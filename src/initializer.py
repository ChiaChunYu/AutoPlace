"""Everything prepared once before the optimization: the density grid, the fillers, the start positions."""
import math

import numpy as np
import torch


def choose_num_bin(data, num_bin, max_bin=512):
    """0 = automatic: the largest power of 2 not above the row count, at most max_bin (a bin is >= one row tall)."""
    if num_bin > 0:
        return num_bin
    rows = max(int(data.height / data.row_height), 2)
    return min(max_bin, 2 ** int(math.log2(rows)))


def compute_fillers(data, fixed_fraction, bin_area, args, logger):
    """Fillers take the free area the cells do not need, so every bin can reach the target density.

    -> filler sizes [n, 2] (numpy). May raise args.target_density to the real utilization.
    """
    placeable = float((1.0 - fixed_fraction).sum() * bin_area)
    util = data.mov_area / placeable
    logger.info("region %.1f x %.1f sites, placeable area %.4E, movable area %.4E, utilization %.3f"
                % (data.width, data.height, placeable, data.mov_area, util))
    if util > args.target_density:
        logger.info("utilization %.3f > target density %.3f: target density raised to %.3f"
                    % (util, args.target_density, util))
        args.target_density = util
    widths = np.sort(data.size[:, 0])
    n = len(widths)
    w = float(widths[int(n * 0.05):max(int(n * 0.95), int(n * 0.05) + 1)].mean())
    h = data.row_height
    count = int(round(max(args.target_density * placeable - data.mov_area, 0.0) / (w * h)))
    logger.info("#fillers %d, filler size %.3f x %.3f sites" % (count, w, h))
    return np.tile(np.array([[w, h]]), (count, 1))


def init_positions(data, node_size, args, device):
    """Movable cells: a small Gaussian cloud at the region centre. Fillers: uniform over the region."""
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    region = torch.tensor([data.width, data.height], dtype=torch.float32)
    num_fill = len(node_size) - data.num_mov
    pos = torch.cat([torch.randn(data.num_mov, 2, generator=gen) * region * 0.001 + region * 0.5,
                     torch.rand(num_fill, 2, generator=gen) * region], 0)
    size = torch.tensor(node_size, dtype=torch.float32)
    pos += (torch.rand(len(node_size), 2, generator=gen) - 0.5) * size * args.noise_ratio
    return pos.to(device)
