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


def _merge_edges(i, j, w, n):
    key, inv = torch.unique(i * n + j, return_inverse=True)
    return key // n, key % n, torch.zeros_like(key).index_add_(0, inv, w)


def _first_of_group(group, score, limit):
    """-> indices of the `limit` highest scores of every group."""
    order = torch.argsort(score, descending=True, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    g, at = group[order], torch.arange(len(order), device=group.device)
    start = torch.ones_like(g, dtype=torch.bool)
    start[1:] = g[1:] != g[:-1]
    return order[at - torch.cummax(torch.where(start, at, torch.zeros_like(at)), 0)[0] < limit]


def cluster_cells(pin_cell, pin_net, area, ratio, seed, max_degree=8, per_head=3, max_rounds=12):
    """pin_cell / pin_net [P] long, sorted by net (cell index len(area) = fixed pin); area [n] long.
    -> (cluster of every cell [n] long, area of every cluster [K] long). Nets above `max_degree` pins say little
    about which cells belong together; a cell above 20 x the mean area (a macro) stays alone, and two clusters merge
    only when together they are under 1.5 x `ratio` mean cells (a head that takes several can end above it)."""
    n, device = len(area), area.device
    degree = torch.bincount(pin_net)[pin_net]
    i, j, w = [], [], []
    for k in range(1, max_degree):                  # the pairs of a net: pins k apart in the pin list
        a, b, d = pin_cell[:-k], pin_cell[k:], degree[:-k]
        m = (pin_net[:-k] == pin_net[k:]) & (d <= max_degree) & (a < n) & (b < n) & (a != b)
        i += [a[m], b[m]]
        j += [b[m], a[m]]
        w += [420 // (d[m] - 1)] * 2                # 420 / (d - 1) is an integer for d = 2..8
    i, j, w = _merge_edges(torch.cat(i), torch.cat(j), torch.cat(w), n)
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
        best = _first_of_group(tail, score, 1)
        tail, to, score = tail[best], to[best], score[best]
        taken = _first_of_group(to, score, per_head)
        parent = torch.arange(num, device=device)
        parent[tail[taken]] = to[taken]
        kept, new = torch.unique(parent, return_inverse=True)
        if len(kept) == num:
            break
        cid, num = new[cid], len(kept)
        area = torch.zeros(num, dtype=torch.long, device=device).index_add_(0, new, area)
        i, j = new[i], new[j]
        i, j, w = _merge_edges(i[i != j], j[i != j], w[i != j], num)
    return cid, area


def coarse_problem(data, node_size, wirelength, density, fixed_fraction, args):
    """-> the problem of the clusters: cid [num_mov], size [K, 2], wirelength, density, and what ParamScheduler needs.

    A merged cluster is a square of the summed cell area with its pins at the centre, one per net; a cell that was
    not merged keeps its shape and pin offsets. Fillers, grid and blockages are those of the cell-level problem."""
    n, device, pins = data.num_mov, density.device, wirelength.all
    size = torch.tensor(node_size[:n], dtype=torch.float32, device=device)
    scale = 1024                                    # areas as integers: exact sums, 1 / 1024 of a site area
    cid, area = cluster_cells(pins["cell"], pins["net"], (size.prod(1).double() * scale).round().long(),
                              args.cluster_ratio, args.seed)
    num = len(area)
    side = (area.double() / scale).sqrt().float()
    csize = torch.stack([side, side], 1)
    single = torch.bincount(cid, minlength=num) == 1
    member = torch.zeros(num, dtype=torch.long, device=device)
    member[cid] = torch.arange(n, device=device)
    csize[single] = size[member[single]]
    owner = torch.cat([cid, cid.new_tensor([num])])[pins["cell"]]             # num = fixed pin
    keep_off = torch.cat([single, single.new_tensor([True])])[owner]
    off = torch.where(keep_off.unsqueeze(1), pins["off"], torch.zeros_like(pins["off"]))
    key = pins["net"] * (num + 1) + owner
    order = torch.argsort(key, stable=True)
    repeat = torch.zeros_like(keep_off)
    repeat[order[1:]] = (key[order][1:] == key[order][:-1]) & ~keep_off[order][1:]
    sel = ~repeat
    cwl = Wirelength(owner[sel], off[sel], pins["net"][sel], pins["num_nets"], num, args.ignore_net_degree)
    cpins = torch.bincount(owner[sel], minlength=num + 1)[:num].double().cpu().numpy()
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
