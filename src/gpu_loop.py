"""The rest of a level with the control of the run on the GPU: no transfer to the host per iteration.

What NesterovOptimizer.step (Barzilai-Borwein step, backtracking), ParamScheduler.step (density weight, smoothing,
preconditioner) and ParamScheduler.need_to_stop decide on the host after every iteration is done by one GPU thread
(cpp_to_py: control_kernel), in the same order and in double precision; the kernels read the density weight, the
smoothing and the step size from device memory. A redone step (backtracking) is a pass whose step is not taken: the
next pass starts from the same point with the new step size. The host only looks every `check_every` passes whether
the run has stopped; once it has, the control freezes (later passes change nothing), so the result is that of a
run checked after every iteration and `check_every` costs at most that many unused gradient calls.

The host classes stay the reference: the same code runs on the CPU (and when the kernels are not built), and the
tests compare the two.
"""
import math

import torch

from cpp_to_py import gp_cuda

# layout of the state, as `enum` in control_kernel
FIELDS = ("lam gam pcoef alpha a iter hpwl_prev motion ww mu settled best_hpwl best_ovf is_best improved stop bt bt_total "
          "accept loop_it evals res_hpwl res_ovf has_best").split()
K = {name: i for i, name in enumerate(FIELDS)}
REASON = {1: "converged", 2: "stalled", 3: "diverged", 4: "max_iter"}


def usable(optimizer):
    return bool(gp_cuda) and optimizer.kernel


def _next_coef(a):
    a_next = (1 + math.sqrt(4 * a * a + 1)) / 2
    return (a - 1) / a_next


def run_level_on_gpu(optimizer, objective, ps, args, logger, iteration, check_every=16):
    """Continue the level from `iteration` (the density weight is on) until it stops. Updates `optimizer`, `ps`,
    `objective.evals` as the host loop would. -> (reason, iterations, hpwl, overflow) of the kept placement."""
    device, pos = optimizer.v.device, optimizer.v
    S = [0.0] * len(FIELDS)
    S[K["lam"]], S[K["gam"]], S[K["alpha"]], S[K["a"]] = ps.density_weight, ps.wa_coeff, optimizer.alpha, optimizer.a
    S[K["pcoef"]] = ps.precond_coef
    S[K["iter"]], S[K["hpwl_prev"]], S[K["motion"]], S[K["ww"]] = ps.iter, ps.hpwl[-1], ps.motion, ps.weighted_weight
    S[K["mu"]], S[K["settled"]], S[K["improved"]] = ps.mu, ps.settled, ps.improved_iter
    S[K["best_hpwl"]], S[K["best_ovf"]], S[K["bt_total"]] = ps.best_hpwl, ps.best_overflow, optimizer.backtracks
    S[K["loop_it"]] = iteration
    S = torch.tensor(S, dtype=torch.float64, device=device)
    kp = torch.tensor([ps.wa_coeff, ps.density_weight, ps.precond_coef * ps.density_weight, optimizer.alpha,
                       _next_coef(optimizer.a)],
                      dtype=torch.float32, device=device)
    a = ps.args
    c = torch.tensor([ps.base_gamma, ps.motion_unit, ps.total_pins, ps.total_area, a.ref_motion, a.min_iter,
                      a.stop_overflow, a.settle_iter, a.stall_iter, args.inner_iter, optimizer.max_backtrack],
                     dtype=torch.float64, device=device)
    u, g, lo, hi = optimizer.u, optimizer.g, optimizer.lo, optimizer.hi
    best = pos.clone()
    evals_before, logged = objective.evals, iteration // args.log_freq
    watch = torch.tensor([K["stop"], K["loop_it"], K["lam"], K["gam"], K["res_hpwl"], K["res_ovf"]], device=device)
    while True:
        for _ in range(check_every):
            u_next, v_next = gp_cuda.nesterov_step(u, pos, g, lo, hi, kp[3:4], kp[4:5])
            g_next = objective.on_gpu(v_next, kp)
            gp_cuda.control(S, kp, c, torch.dist(v_next, pos), torch.dist(g_next, g), objective.hpwl, objective.overflow)
            gp_cuda.select(S, u, pos, g, u_next, v_next, g_next, best)
        stop, it, lam, gam, hpwl, ovf = S[watch].tolist()          # the only transfer: every check_every passes
        if int(it) // args.log_freq > logged:
            logged = int(it) // args.log_freq
            logger.info("iter: %d | hpwl: %.4E overflow: %.4f density_weight: %.4E wa_coeff: %.4E" % (it, hpwl, ovf, lam, gam))
        if stop:
            break
    S = S.tolist()
    get = lambda name: S[K[name]]
    ps.density_weight, ps.wa_coeff, ps.iter, ps.motion = get("lam"), get("gam"), int(get("iter")), get("motion")
    ps.precond_coef = get("pcoef")
    ps.weighted_weight, ps.mu, ps.settled = get("ww"), get("mu"), int(get("settled"))
    ps.best_hpwl, ps.best_overflow, ps.improved_iter = get("best_hpwl"), get("best_ovf"), int(get("improved"))
    ps.hpwl.append(get("hpwl_prev"))
    optimizer.u, optimizer.g, optimizer.alpha, optimizer.a = u, g, get("alpha"), get("a")
    optimizer.backtracks = int(get("bt_total"))
    objective.evals = evals_before + int(get("evals"))
    if get("has_best"):
        pos.copy_(best)
    return REASON[int(get("stop"))], int(get("loop_it")), get("res_hpwl"), get("res_ovf")
