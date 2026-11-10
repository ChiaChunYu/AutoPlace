"""Nesterov's method with a Barzilai-Borwein step and backtracking (ePlace, Algorithm 2).

u is the main solution, v the look-ahead point the gradient is taken at. The step length alpha is the inverse of a
local Lipschitz estimate |v' - v| / |g' - g|; a step is redone with the new estimate when it shrinks by > 5 %.
"""
import math

import torch


class NesterovOptimizer(object):
    def __init__(self, pos, grad_fn, lr, max_backtrack=10):
        """pos: the tensor to optimize, updated in place (it holds v). grad_fn(x) -> gradient (may clamp x in place)."""
        self.grad_fn = grad_fn
        self.max_backtrack = max_backtrack
        self.v = pos
        self.g = grad_fn(self.v)
        self.u = self.v.clone()
        probe = torch.add(self.v, self.g, alpha=-lr)
        self.alpha = (torch.dist(self.v, probe) / torch.dist(self.g, grad_fn(probe))).item()
        self.a = 1.0
        self.backtracks = 0

    def step(self):
        a_next = (1 + math.sqrt(4 * self.a * self.a + 1)) / 2
        coef = (self.a - 1) / a_next
        for k in range(self.max_backtrack):
            u_next = torch.add(self.v, self.g, alpha=-self.alpha)
            v_next = torch.lerp(self.u, u_next, 1 + coef)              # u_next + coef (u_next - u)
            g_next = self.grad_fn(v_next)
            alpha_next = (torch.dist(v_next, self.v) / torch.dist(g_next, self.g)).item()
            if alpha_next > 0.95 * self.alpha:
                break
            self.alpha = alpha_next
            self.backtracks += 1
        self.u = u_next
        self.v.copy_(v_next)
        self.g = g_next
        self.alpha = alpha_next
        self.a = a_next


def estimate_initial_learning_rate(pos, grad_fn, lr):
    """Step length from one probe step of length lr (the same Lipschitz estimate as the optimizer)."""
    g = grad_fn(pos)
    probe = pos - lr * g
    g2 = grad_fn(probe)                     # grad_fn may clamp `probe` in place
    return ((pos - probe).norm() / (g - g2).norm()).item()
