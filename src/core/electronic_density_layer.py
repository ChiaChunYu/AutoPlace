"""Electrostatic density (ePlace): cells are charges, the density map is the charge density, and the force on a
cell is the electric field of the whole map.

Smoothing. A side of a node narrower than sqrt(2) bins is widened to sqrt(2) bins and the density scaled down,
keeping the charge.
Field. Poisson's equation with zero field across the region boundary, solved by FFT on the mirrored map (the same
solution as the DCT / DST formulation, without a custom transform).
Map and gradient. On the GPU: two CUDA kernels, one thread per node (cpp_to_py/gp_cuda_kernel.cu). The PyTorch code
(`_rasterize`) is the same computation for the CPU and the reference the kernels are tested against: every node is
cut into pieces of at most 2 x 2 bins, so each piece touches 3 x 3 bins and the map is one vectorized scatter.
"""
import math

import numpy as np
import torch

from cpp_to_py import gp_cuda


def make_pieces(size, bin_size):
    """size [n, 2] numpy -> (node, offset from the node centre, half size) of the pieces, each <= 2 x 2 bins."""
    n = np.maximum(np.ceil(size / (2 * bin_size) - 1e-9), 1).astype(np.int64)
    count = n.prod(1)
    node = np.repeat(np.arange(len(size)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    piece = size[node] / n[node]
    return node, -size[node] / 2 + (np.stack([k // n[node, 1], k % n[node, 1]], 1) + 0.5) * piece, piece / 2


class ElectronicDensityLayer(object):
    def __init__(self, width, height, nbx, nby, fixed_fraction, target_density, node_size, num_mov, device,
                 node_pins=None):
        """node_size [num_mov + num_fillers, 2] numpy (real sizes); fixed_fraction [nbx, nby] numpy; node_pins [nodes]
        numpy (only for the preconditioner of `forward`)."""
        device = torch.device(device)
        self.nbx, self.nby, self.num_mov, self.num_nodes = nbx, nby, num_mov, len(node_size)
        self.target, self.device = target_density, device
        self.use_kernel = bool(gp_cuda) and device.type == "cuda"
        self.bin_np = np.array([width / nbx, height / nby])
        self.bin_area = float(self.bin_np.prod())
        self.mov_area = float(np.prod(node_size[:num_mov], 1).sum())
        t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
        self.bin = t(self.bin_np)
        # a blocked bin counts as filled to the target: with fillers, every bin then settles at the target density
        self.fixed_map = t(fixed_fraction * target_density)
        real = np.asarray(node_size, dtype=np.float64)
        smooth = np.maximum(real, math.sqrt(2) * self.bin_np)
        # (sizes, density scale): the smoothed nodes for the optimization, the real cells for the final overflow
        self.smooth = self._nodes(smooth, real.prod(1) / smooth.prod(1), t)
        self.real = self._nodes(real[:num_mov], np.ones(num_mov), t)
        self.pos_lo = self.smooth["size"] / 2               # the whole smoothed node stays inside the region
        self.pos_hi = t([width, height]) - self.pos_lo
        self.steps = torch.arange(3, device=device)
        self.pins = t(np.zeros(len(real)) if node_pins is None else node_pins)
        self.area = t(real.prod(1))
        self.no_grad = torch.zeros(0, 2, device=device)
        mx, my = (np.minimum(np.arange(2 * n), 2 * n - 1 - np.arange(2 * n)) for n in (nbx, nby))
        self.mirror = t((mx[:, None] * nby + my[None, :]).reshape(-1), torch.long)     # map -> mirrored map

        # spectral field of the mirrored map. The 1 / bin_x factor is the ePlace / DREAMPlace / Xplace scale of the
        # density gradient (field in bin units), kept so the density weight and the preconditioner mean the same.
        kx = 2 * math.pi * torch.fft.fftfreq(2 * nbx, d=float(self.bin_np[0])).to(device).unsqueeze(1)
        ky = 2 * math.pi * torch.fft.rfftfreq(2 * nby, d=float(self.bin_np[1])).to(device).unsqueeze(0)
        inv = 1.0 / (kx ** 2 + ky ** 2).clamp_(min=1e-30)
        inv[0, 0] = 0.0
        inv = inv / float(self.bin_np[0])
        kx[nbx, 0] = 0.0                                    # the Nyquist term has no derivative
        ky[0, nby] = 0.0
        self.to_field = torch.stack([-1j * kx * inv, -1j * ky * inv]).to(torch.complex64)      # [2, 2 nbx, nby + 1]

    def _nodes(self, size, ratio, t):
        nodes = {"size": t(size), "ratio": t(ratio)}
        if not self.use_kernel:
            node, off, half = make_pieces(size, self.bin_np)
            nodes.update(node=t(node, torch.long), off=t(off), half=t(half), piece_ratio=t(ratio[node]).view(-1, 1, 1))
        return nodes

    def clamp_(self, pos):
        """Keep every (smoothed) node inside the placement region, in place."""
        return pos.clamp_(min=self.pos_lo, max=self.pos_hi)

    def _rasterize(self, pos, n):
        """-> (flat bin index [Q, 3, 3], weight [Q, 3, 3] = density a piece adds to each bin)."""
        c = pos[n["node"]] + n["off"]
        lo, hi = (c - n["half"]) / self.bin, (c + n["half"]) / self.bin
        edges = lo.floor().clamp_(min=0).unsqueeze(2) + self.steps              # [Q, 2, 3] lower edge of 3 bins
        ov = (torch.min(hi.unsqueeze(2), edges + 1) - torch.max(lo.unsqueeze(2), edges)).clamp_(min=0)
        ix = edges[:, 0].long().clamp_(max=self.nbx - 1)
        iy = edges[:, 1].long().clamp_(max=self.nby - 1)
        return ix.unsqueeze(2) * self.nby + iy.unsqueeze(1), n["piece_ratio"] * ov[:, 0, :, None] * ov[:, 1, None, :]

    def _density(self, pos, n, num_mov):
        """-> (total density mirrored in both axes for the FFT [2 nbx, 2 nby], overflow).

        Total = nodes below num_mov (cells) + blocked area + the rest (fillers); the overflow counts the first two."""
        if self.use_kernel:
            rho, over = gp_cuda.density_map(pos, n["size"], n["ratio"], num_mov, float(self.bin_np[0]), float(self.bin_np[1]),
                                            self.nbx, self.nby, self.fixed_map, self.target)
        else:
            idx, w = self._rasterize(pos, n)
            idx = idx + (n["node"] >= num_mov).long().view(-1, 1, 1) * (self.nbx * self.nby)
            maps = torch.zeros(2 * self.nbx * self.nby, device=self.device).index_add_(0, idx.reshape(-1), w.reshape(-1))
            mov = maps[:self.nbx * self.nby] + self.fixed_map.reshape(-1)
            over = (mov - self.target).clamp_(min=0)
            rho = (mov + maps[self.nbx * self.nby:])[self.mirror].view(2 * self.nbx, 2 * self.nby)
        return rho, over.sum() * (self.bin_area / self.mov_area)

    def exact_overflow(self, pos):
        """Overflow of the movable cells with their real sizes (no smoothing, no fillers)."""
        return self._density(pos[:self.num_mov].contiguous(), self.real, self.num_mov)[1]

    def forward(self, pos, wl_grad=None, weight=1.0, precond=-1.0):
        """pos [num_nodes, 2] -> (gradient [num_nodes, 2], overflow).

        gradient = (wl_grad + weight x gradient of the density energy) / max(1, pins + precond x area); with the
        defaults it is the plain density gradient (wl_grad [num_mov, 2] is added to the movable cells only).
        """
        n, size, bx, by = self.smooth, (2 * self.nbx, 2 * self.nby), float(self.bin_np[0]), float(self.bin_np[1])
        mirrored, overflow = self._density(pos, n, self.num_mov)
        field = torch.fft.irfft2(torch.fft.rfft2(mirrored) * self.to_field, s=size)     # [2, 2 nbx, 2 nby], x and y
        wl_grad = self.no_grad if wl_grad is None else wl_grad
        if self.use_kernel:
            grad = gp_cuda.density_grad(pos, n["size"], n["ratio"], field, wl_grad, self.pins, self.area, weight, precond,
                                        bx, by, self.nbx, self.nby)
        else:                                                                   # -(charge x field), summed per node
            idx, w = self._rasterize(pos, n)
            force = (w.unsqueeze(3) * field[:, :self.nbx, :self.nby].reshape(2, -1).t()[idx]).sum((1, 2))
            grad = torch.zeros_like(pos).index_add_(0, n["node"], force).mul_(-weight)
            grad[:len(wl_grad)] += wl_grad
            if precond >= 0:
                grad /= (self.pins + precond * self.area).clamp_(min=1.0).unsqueeze(1)
        return grad, overflow
