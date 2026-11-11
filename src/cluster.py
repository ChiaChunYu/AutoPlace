"""Two-level placement: the cells are merged into clusters, the clusters are placed first (a smaller problem with
the same fillers and grid), then every cell starts on its cluster.

Clustering, on the device: rounds of star matching. In a round half of the clusters, drawn at random, are heads;
every other cluster joins the head it shares the most net weight with per unit of merged area (a net of d pins gives
1 / (d - 1) to each pair of its cells), a head taking at most `per_head` of them. Weights and areas are summed as
integers, so the same input gives the same clusters.
"""
from types import SimpleNamespace

import numpy as np
import torch

from .core import ElectronicDensityLayer, Wirelength


def _merge_edges(parts, n):
    """parts: [(i, j, w)] -> one edge per (i, j), sorted by i then j, weights summed; int32 out (only the sort key is
    int64, and only while merging)."""
    key = torch.cat([a.long() * n + b for a, b, _ in parts])
    w = torch.cat([c for _, _, c in parts]).int()
    key, inv = torch.unique(key, return_inverse=True)
    w = torch.zeros(len(key), dtype=torch.int32, device=w.device).index_add_(0, inv, w)
    return (key // n).int(), (key % n).int(), w


def _first_of_group(group, score, limit):
    """-> indices of the `limit` highest scores of every group."""
    order = torch.argsort(score, descending=True, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    g, at = group[order], torch.arange(len(order), device=group.device)
    start = torch.ones_like(g, dtype=torch.bool)
    start[1:] = g[1:] != g[:-1]
    return order[at - torch.cummax(torch.where(start, at, torch.zeros_like(at)), 0)[0] < limit]


def cluster_cells(pin_cell, pin_net, area, ratio, seed, max_degree=8, per_head=3, max_rounds=12):
    """pin_cell / pin_net [P] int32 or long, sorted by net (cell index len(area) = fixed pin); area [n] long.
    -> (cluster of every cell [n] long, area of every cluster [K] long). Nets above `max_degree` pins say little
    about which cells belong together; a cell above 20 x the mean area (a macro) stays alone, and two clusters merge
    only when together they are under 1.5 x `ratio` mean cells (a head that takes several can end above it)."""
    n, device = len(area), area.device
    degree = torch.bincount(pin_net)[pin_net]
    small = degree <= max_degree                    # only these nets give pairs; the pins stay grouped by net
    pin_cell, pin_net, degree = pin_cell[small], pin_net[small], degree[small]
    i = j = w = None
    for k in range(1, max_degree):                  # the pairs of a net: pins k apart in the pin list
        a, b, d = pin_cell[:-k], pin_cell[k:], degree[:-k]
        m = (pin_net[:-k] == pin_net[k:]) & (a < n) & (b < n) & (a != b)
        a, b, d = a[m], b[m], 420 // (d[m] - 1)     # 420 / (d - 1) is an integer for d = 2..8
        parts = [(a, b, d), (b, a, d)] + ([(i, j, w)] if i is not None else [])
        # merged after every k, so the pairs of all k never exist at once (same sums: integers)
        i, j, w = _merge_edges(parts, n)
        del a, b, d, m, parts
    del pin_cell, pin_net, degree                   # only the edges are used from here on
    cid, num = torch.arange(n, device=device), n
    big = 20 * area.sum() // n
    cap = 3 * ratio * area[area < big].sum() // (2 * int((area < big).sum()))
    gen = torch.Generator().manual_seed(seed)
    for _ in range(max_rounds):
        if num * ratio <= n:
            break
        head = (torch.rand(num, generator=gen) < 0.5).to(device)
        total = area[i] + area[j]
        m = ~head[i] & head[j] & (total <= cap) & (area[i] < big) & (area[j] < big)
        tail, to, score = i[m], j[m], w[m].double() / total[m].double()
        del total, m
        best = _first_of_group(tail, score, 1)
        tail, to, score = tail[best], to[best], score[best]
        taken = _first_of_group(to, score, per_head)
        parent = torch.arange(num, device=device)
        parent[tail[taken]] = to[taken].long()
        kept, new = torch.unique(parent, return_inverse=True)
        if len(kept) == num:
            break
        cid, num = new[cid], len(kept)
        area = torch.zeros(num, dtype=torch.long, device=device).index_add_(0, new, area)
        i, j = new[i].int(), new[j].int()
        keep = i != j
        parts = [(i[keep], j[keep], w[keep])]
        del i, j, w, keep, tail, to, score, best, taken, parent, kept, new
        i, j, w = _merge_edges(parts, num)
        del parts
    return cid, area


def coarse_problem(data, node_size, wirelength, density, fixed_fraction, args):
    """-> the problem of the clusters: cid [num_mov], size [K, 2], wirelength, density, and what ParamScheduler needs.

    A merged cluster is a square of the summed cell area with its pins at the centre, one per net; a cell that was
    not merged keeps its shape and pin offsets. Fillers, grid and blockages are those of the cell-level problem."""
    n, device = data.num_mov, density.device
    size = torch.tensor(node_size[:n], dtype=torch.float32, device=device)
    scale = 1024                                    # areas as integers: exact sums, 1 / 1024 of a site area
    cid, area = cluster_cells(wirelength.all["cell"], wirelength.all["net"], (size.prod(1).double() * scale).round().long(),
                              args.cluster_ratio, args.seed)
    pins = wirelength.pins()
    num = len(area)
    side = (area.double() / scale).sqrt().float()
    csize = torch.stack([side, side], 1)
    single = torch.bincount(cid, minlength=num) == 1
    member = torch.zeros(num, dtype=torch.long, device=device)
    member[cid] = torch.arange(n, device=device)
    csize[single] = size[member[single]]
    owner = torch.cat([cid, cid.new_tensor([num])])[pins.pop("cell")]          # num = fixed pin
    keep_off = torch.cat([single, single.new_tensor([True])])[owner]
    # one pin per (net, merged cluster): the first of each run of equal keys (temporaries freed as soon as used)
    key = pins["net"] * (num + 1) + owner
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    del key
    repeat = torch.zeros_like(keep_off)
    repeat[order[1:]] = (sorted_key[1:] == sorted_key[:-1]) & ~keep_off[order[1:]]
    del sorted_key, order
    sel = ~repeat
    del repeat
    off = torch.where(keep_off[sel].unsqueeze(1), pins["off"][sel], torch.zeros(1, 2, device=device))
    c_owner, c_net, num_nets = owner[sel], pins["net"][sel], pins["num_nets"]
    del owner, keep_off, sel, pins
    cwl = Wirelength(c_owner, off, c_net, num_nets, num, args.ignore_net_degree)
    cpins = torch.bincount(c_owner, minlength=num + 1)[:num].double().cpu().numpy()
    del c_owner, c_net, off
    cnode = np.concatenate([csize.double().cpu().numpy(), node_size[n:]])
    cden = ElectronicDensityLayer(data.width, data.height, density.nbx, density.nby, fixed_fraction, density.target,
                                  cnode, num, device, np.concatenate([cpins, np.zeros(len(cnode) - num)]))
    return SimpleNamespace(cid=cid, num=num, size=csize, wirelength=cwl, density=cden, total_pins=float(cpins.sum()),
                           total_area=float(cnode.prod(1).sum()), num_nodes=len(cnode))


def expand(cpos, coarse, node_size, seed):
    """Positions of clusters + fillers -> positions of cells + fillers: a cell at a random point of the square of
    its cluster (so the cells of a cluster start side by side, not on top of each other)."""
    n, device = len(coarse.cid), cpos.device
    gen = torch.Generator().manual_seed(seed)
    room = (coarse.size[coarse.cid] - torch.tensor(node_size[:n], dtype=torch.float32, device=device)).clamp_(min=0)
    cells = cpos[:coarse.num][coarse.cid] + (torch.rand(n, 2, generator=gen) - 0.5).to(device) * room
    return torch.cat([cells, cpos[coarse.num:]])
