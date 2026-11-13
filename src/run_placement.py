"""The flow: read LEF + DEF -> build the problem -> global placement -> evaluate -> write DEF + metrics."""
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from cpp_to_py import def_io, gp_cuda
from utils import parse_def, parse_lef, write_def
from . import gpu_loop
from .calculator import Objective
from .cluster import coarse_problem, expand
from .core import ElectronicDensityLayer, Wirelength
from .database import PlaceData
from .evaluator import evaluate_placement
from .initializer import choose_num_bin, compute_fillers, init_positions
from .resolution import force_num_bin, refine
from .nesterov_optimizer import NesterovOptimizer, estimate_initial_learning_rate
from .param_scheduler import ParamScheduler


def density_uses_kernel(device):
    return bool(gp_cuda) and device.type == "cuda"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _scheduler(density, num_mov, mov_area, total_pins, total_area, num_nodes, args, logger):
    # pitch: the spacing of the cells once they are spread at the target density
    return ParamScheduler(SimpleNamespace(bin=density.bin_np, total_pins=total_pins, total_area=total_area, num_nodes=num_nodes,
                                          pitch=math.sqrt(mov_area / density.target / num_mov), refine=refine(density)),
                          args, logger)


def _place_level(pos, objective, ps, args, logger, first_iter=0):
    """One level of global placement: optimizes `pos` in place until ps.need_to_stop(). -> statistics."""
    bounds = (objective.density.pos_lo, objective.density.pos_hi)
    lr = estimate_initial_learning_rate(pos, objective, args.lr, bounds)
    optimizer = NesterovOptimizer(pos, objective, lr, bounds, stats=lambda: (objective.hpwl, objective.overflow, objective.wa))
    reason, iteration, best, start_iters = "max_iter", first_iter, None, 0
    hpwl = overflow = float("nan")
    for iteration in range(first_iter, args.inner_iter):
        if ps.density_weight > 0 and gpu_loop.usable(optimizer):     # the rest of the level: control on the GPU
            reason, iters, hpwl, overflow = gpu_loop.run_level_on_gpu(optimizer, objective, ps, args, logger, iteration)
            return SimpleNamespace(reason=reason, iters=iters, start_iters=start_iters, hpwl=hpwl, overflow=overflow)
        optimizer.step()
        if optimizer.diverged:
            reason = "diverged"
            break
        hpwl, overflow, wa = optimizer.fetched
        if iteration % args.log_freq == 0:
            logger.info("iter: %d | hpwl: %.4E overflow: %.4f density_weight: %.4E wa_coeff: %.4E"
                        % (iteration, hpwl, overflow, ps.density_weight, ps.wa_coeff))
        if ps.density_weight == 0:              # wirelength-only start
            if ps.start_step(hpwl, wa):
                ps.density_weight, start_iters = args.density_weight * objective.tension_unit(pos), iteration + 1
                logger.info("wirelength-only start over after %d iterations (hpwl %.4E): density weight %.4E"
                            % (start_iters, hpwl, ps.density_weight))
            continue
        ps.step(hpwl, overflow, optimizer.moved, wa)
        stop = ps.need_to_stop()
        if ps.is_best:
            best = (pos.clone(), hpwl, overflow)
        if stop:
            reason = stop
            break
    if best:
        pos.copy_(best[0])
        hpwl, overflow = best[1:]
    return SimpleNamespace(reason=reason, iters=iteration + 1, start_iters=start_iters, hpwl=hpwl, overflow=overflow)


def global_placement_main(data, wirelength, density, pos, node_size, fixed_fraction, args, logger):
    """Optimize `pos` in place. -> dict of GP statistics."""
    device, n = pos.device, data.num_mov
    _sync(device)
    start = time.time()
    ps = _scheduler(density, n, data.mov_area, float(data.cell_pins.sum()), float(node_size.prod(1).sum()), len(node_size), args, logger)
    objective = Objective(wirelength, density, ps)
    stats, evals, first_iter, start_iters = {}, 0, 0, 0
    if args.cluster_ratio > 1:                                  # two levels: the clusters first
        coarse = coarse_problem(data, node_size, wirelength, density, fixed_fraction, args)
        _sync(device)
        stats = {"clusters": coarse.num, "cluster_time_s": time.time() - start, "coarse_num_bin": coarse.density.nbx}
        logger.info("two levels: %d clusters of %d cells (%.2fs), %d x %d bins"
                    % (coarse.num, n, stats["cluster_time_s"], coarse.density.nbx, coarse.density.nby))
        cargs = SimpleNamespace(**vars(args))
        cargs.stop_overflow, cargs.settle_iter, cargs.min_iter = args.handover_overflow, 0, 0
        cps = _scheduler(coarse.density, coarse.num, data.mov_area, coarse.total_pins, coarse.total_area, coarse.num_nodes, cargs, logger)
        cobjective = Objective(coarse.wirelength, coarse.density, cps)
        cpos = torch.cat([pos[:coarse.num], pos[n:]])
        level = _place_level(cpos, cobjective, cps, cargs, logger)
        pos.copy_(expand(cpos, coarse, node_size, args.seed)).clamp_(min=density.pos_lo, max=density.pos_hi)
        # the same share of the wire tension on both levels
        weight = cps.density_weight * objective.tension_unit(pos) / cobjective.tension_unit(cpos)
        overflow = density.forward(pos)[1].item() if args.wa_rule == "overflow" else None     # only the formula needs it
        ps.resume(cps, weight, wirelength.forward(pos, ps.wa_coeff)[2].item(), overflow)
        logger.info("clusters placed after %d iterations (stop: %s, hpwl %.4E, overflow %.4f): the cells take over, density weight %.4E"
                    % (level.iters, level.reason, level.hpwl, level.overflow, weight))
        evals, first_iter, start_iters = cobjective.evals, level.iters, level.start_iters
        stats["coarse_iters"] = level.iters
        del coarse, cobjective, cps, cpos              # the cluster problem is not needed any more: free its memory
    level = _place_level(pos, objective, ps, args, logger, first_iter)
    _sync(device)
    gp_time, iters = time.time() - start, level.iters
    logger.info("GP stop (%s): #iters %d hpwl: %.6E overflow: %.4f GP time: %.4fs per iter: %.6fs (%d gradient calls)"
                % (level.reason, iters, level.hpwl, level.overflow, gp_time, gp_time / iters, evals + objective.evals))
    stats.update({"stop_reason": level.reason, "iters": iters, "start_iters": start_iters or level.start_iters, "gp_time_s": gp_time,
                  "per_iter_s": gp_time / iters, "grad_calls": evals + objective.evals})
    return stats


def run_placement_main(args, logger):
    total_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:%d" % args.gpu if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    if args.gpu >= 0 and device.type != "cuda":
        logger.info("warning: --gpu %d asked for, but no CUDA device is available: the CPU is used" % args.gpu)
    if device.type == "cuda":
        torch.cuda.set_device(device)           # the kernels are launched on the current device

    lef = parse_lef(args.lef)
    design_def = parse_def(args.def_file)
    data = PlaceData(lef, design_def)
    parse_time = time.time() - total_start
    logger.info("design %s: #movable %d, #fixed %d, #nets %d, #pins %d, #rows %d, #blockages %d (parse %.2fs)"
                % (data.name, data.num_mov, int(data.fixed.sum()), data.num_nets, data.num_pins,
                   len(design_def.rows), len(design_def.blockages), parse_time))
    if data.kept_in_place:
        logger.info("%d placed components that are not core cells (pads, blocks, ...) are kept where they are" % data.kept_in_place)
    if design_def.soft_blockages:
        logger.info("warning: %d SOFT / PARTIAL placement blockages ignored" % design_def.soft_blockages)
    if data.unplaced_io_pins:
        logger.info("warning: %d connections to design pins without a location ignored" % data.unplaced_io_pins)

    num_bin = choose_num_bin(data, args.num_bin)                # the grid of the overflow
    evaluate_only = bool(args.eval_def)
    judge_fraction = data.fixed_fraction(num_bin, num_bin)
    free = 1.0 - float(judge_fraction.mean())
    force_bin = num_bin if evaluate_only else (args.force_num_bin or force_num_bin(data.num_mov, free, num_bin))
    fixed_fraction = judge_fraction if force_bin == num_bin else data.fixed_fraction(force_bin, force_bin)
    bin_area = data.width / force_bin * data.height / force_bin
    if evaluate_only:
        node_size = data.size
    else:
        node_size = np.concatenate([data.size, compute_fillers(data, fixed_fraction, bin_area, args, logger)])
    logger.info("#bins %d x %d (bin %.3f x %.3f sites), force on %d x %d bins, target density %.3f, device %s"
                % (num_bin, num_bin, data.width / num_bin, data.height / num_bin, force_bin, force_bin, args.target_density, device))
    if device.type == "cuda" and not gp_cuda:
        logger.info("warning: the CUDA kernels are not built (cpp_to_py/build.sh): PyTorch operators are used, "
                    "several times slower and not reproducible bit for bit")
    if not def_io:
        logger.info("warning: the compiled DEF reader is not built (cpp_to_py/build.sh): the Python reader is used")
    tensor = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    wirelength = Wirelength(tensor(data.pin_cell, torch.long), tensor(data.pin_off, torch.float32),
                            tensor(data.pin_net, torch.long), data.num_nets, data.num_mov, args.ignore_net_degree)
    density = ElectronicDensityLayer(data.width, data.height, force_bin, force_bin, fixed_fraction, args.target_density,
                                     node_size, data.num_mov, device,
                                     np.concatenate([data.cell_pins, np.zeros(len(node_size) - data.num_mov)]))
    if force_bin != num_bin:
        density.judge = ElectronicDensityLayer(data.width, data.height, num_bin, num_bin, judge_fraction,
                                               args.target_density, data.size, data.num_mov, device, field=False)
    metrics = {"design": data.name, "num_movable": data.num_mov, "num_nets": data.num_nets, "num_pins": data.num_pins,
               "num_fillers": len(node_size) - data.num_mov, "num_bin": num_bin, "force_num_bin": force_bin, "target_density": args.target_density,
               "seed": args.seed, "device": str(device), "cuda_kernels": density_uses_kernel(device), "compiled_reader": bool(def_io), "parse_time_s": parse_time}

    if evaluate_only:
        pos = tensor(data.placed_pos(parse_def(args.eval_def)), torch.float32)
    else:
        pos = init_positions(data, node_size, args, device)
        metrics.update(global_placement_main(data, wirelength, density, pos, node_size, fixed_fraction, args, logger))
        if metrics["stop_reason"] != "converged":
            logger.info("warning: global placement ended '%s', not converged (--stop_overflow %.3f)"
                        % (metrics["stop_reason"], args.stop_overflow))
        if not bool(torch.isfinite(pos[:data.num_mov]).all()):
            raise RuntimeError("the placement is not finite (diverged before any usable point): nothing written")
    metrics.update(evaluate_placement(pos, wirelength, density, data))
    logger.info("exact HPWL: %.6E sites = %.1f um, exact overflow: %.4f"
                % (metrics["hpwl_sites"], metrics["hpwl_um"], metrics["overflow"]))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        if not evaluate_only and args.write_def:
            start = time.time()
            write_def(design_def, os.path.join(args.out_dir, "gp.def"), data.mov_index,
                      data.to_def_xy(pos[:data.num_mov].cpu().numpy()))
            metrics["write_time_s"] = time.time() - start
        metrics["total_time_s"] = time.time() - total_start
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    logger.info("total time: %.2fs" % (time.time() - total_start))
    return metrics
