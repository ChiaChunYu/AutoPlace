"""AutoPlace: GPU global placement from LEF + DEF (wirelength-driven; first layer).

    python main.py --lef tech.lef cells.lef --def floorplan.def --out_dir out/
    python main.py --lef ... --def floorplan.def --eval_def placed.def       # only evaluate a placement

Writes <out_dir>/gp.def (the input DEF with every movable cell PLACED), metrics.json and placer.log.
"""
import argparse
import os

from src import run_placement_main
from utils import setup_logger


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


def get_option(argv=None):
    p = argparse.ArgumentParser("AutoPlace", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lef", nargs="+", required=True, help="LEF file(s): technology, cells, macros")
    p.add_argument("--def", dest="def_file", required=True, help="DEF with the netlist, rows and fixed objects")
    p.add_argument("--out_dir", default="", help="output folder (nothing is written without it)")
    p.add_argument("--eval_def", default="", help="evaluate this placed DEF (same netlist as --def) instead of placing")
    p.add_argument("--write_def", type=str2bool, default=True, help="write gp.def")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0, help="GPU id, -1 = CPU")
    # global placement
    p.add_argument("--target_density", type=float, default=1.0, help="density every bin may reach")
    p.add_argument("--num_bin", type=int, default=0, help="density grid is num_bin x num_bin; 0 = from the row count")
    p.add_argument("--stop_overflow", type=float, default=0.07, help="stop when the overflow is below this")
    p.add_argument("--inner_iter", type=int, default=10000, help="maximum iterations")
    p.add_argument("--settle_iter", type=int, default=30, help="iterations run after reaching --stop_overflow")
    p.add_argument("--stall_iter", type=int, default=200, help="stop when nothing improved for this many iterations")
    p.add_argument("--min_iter", type=int, default=100, help="no stop before this iteration")
    p.add_argument("--ignore_net_degree", type=int, default=100, help="nets with more pins are not optimized")
    p.add_argument("--start_tol", type=float, default=1e-3, help="the wirelength-only start ends when the WA wirelength falls by less than this share per iteration")
    p.add_argument("--density_weight", type=float, default=1e-3, help="first density weight: the density force as a share of the full wire tension")
    p.add_argument("--wa_coeff", type=float, default=4.0, help="WA smoothing, in bin sizes")
    p.add_argument("--ref_motion", type=float, default=5.0, help="move per iteration (RMS, in cell pitches) at which the density weight stops growing")
    p.add_argument("--cluster_ratio", type=float, default=5, help="clusters of this many cells are placed first, then the cells (two levels); 1 = cells only")
    p.add_argument("--handover_overflow", type=float, default=0.7, help="overflow of the clusters at which the cells take over")
    p.add_argument("--lr", type=float, default=0.01, help="length of the probe step that sets the first step size")
    p.add_argument("--log_freq", type=int, default=100)
    return p.parse_args(argv)


def main(argv=None):
    args = get_option(argv)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.out_dir, "placer.log") if args.out_dir else None)
    logger.info("options: %s" % vars(args))
    try:
        return run_placement_main(args, logger)
    finally:
        for h in list(logger.handlers):         # close placer.log
            logger.removeHandler(h)
            h.close()


if __name__ == "__main__":
    main()
