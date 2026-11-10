"""Nesterov's method with a Barzilai-Borwein step and backtracking (ePlace, Algorithm 2).

u is the main solution, v the look-ahead point the gradient is taken at. The step length alpha is the inverse of a
local Lipschitz estimate |v' - v| / |g' - g|; a step is redone with the new estimate when it shrinks by > 5 %.
"""
import math

import torch

from cpp_to_py import gp_cuda


class NesterovOptimizer(object):
    def __init__(self, pos, grad_fn, lr, bounds, stats=None, max_backtrack=10):
        """pos: the tensor to optimize, updated in place (it holds v). grad_fn(x) -> gradient at x.
        bounds = (lo, hi): every point the gradient is taken at is kept inside them.
        stats: optional callable -> 0-d tensors of the last evaluated point; they are brought to the CPU together with
        the step-size estimate (one transfer per iteration instead of one each) and left in `self.fetched`."""
        self.grad_fn = grad_fn
        self.stats, self.fetched = stats or (lambda: ()), []
        self.lo, self.hi = bounds
        self.max_backtrack = max_backtrack
        self.kernel = bool(gp_cuda) and pos.is_cuda
        self.v = pos.clamp_(min=self.lo, max=self.hi)
        self.g = grad_fn(self.v)
        self.u = self.v.clone()
        probe = torch.add(self.v, self.g, alpha=-lr).clamp_(min=self.lo, max=self.hi)
        self.alpha = (torch.dist(self.v, probe) / torch.dist(self.g, grad_fn(probe))).item()
        self.a = 1.0
        self.backtracks = 0

    def step(self):
        a_next = (1 + math.sqrt(4 * self.a * self.a + 1)) / 2
        coef = (self.a - 1) / a_next
        for k in range(self.max_backtrack):
            if self.kernel:             # one pass: u_next = v - alpha g, v_next = u_next + coef (u_next - u), clamped
                u_next, v_next = gp_cuda.nesterov_step(self.u, self.v, self.g, self.lo, self.hi, self.alpha, coef)
            else:
                u_next = torch.add(self.v, self.g, alpha=-self.alpha)
                v_next = torch.lerp(self.u, u_next, 1 + coef).clamp_(min=self.lo, max=self.hi)
            g_next = self.grad_fn(v_next)
            moved, turned, *self.fetched = torch.stack((torch.dist(v_next, self.v), torch.dist(g_next, self.g)) + tuple(self.stats())).tolist()
            alpha_next = moved / turned
            if alpha_next > 0.95 * self.alpha:
                break
            self.alpha = alpha_next
            self.backtracks += 1
        self.u = u_next
        self.v.copy_(v_next)
        self.g = g_next
        self.alpha = alpha_next
        self.a = a_next


def estimate_initial_learning_rate(pos, grad_fn, lr, bounds):
    """Step length from one probe step of length lr (the same Lipschitz estimate as the optimizer)."""
    pos.clamp_(min=bounds[0], max=bounds[1])
    g = grad_fn(pos)
    probe = (pos - lr * g).clamp_(min=bounds[0], max=bounds[1])
    return ((pos - probe).norm() / (g - grad_fn(probe)).norm()).item()
