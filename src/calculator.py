"""The gradient of the objective of global placement:  WA wirelength + density_weight x density energy,
divided per node by the preconditioner max(1, #pins + precond_coef x density_weight x area).

The value of the objective is not needed by the optimizer, so it is never computed; the HPWL and the overflow of the
evaluated point come for free from the same pass.
"""


class Objective(object):
    def __init__(self, wirelength, density, ps):
        self.wirelength, self.density, self.ps = wirelength, density, ps
        self.hpwl = self.overflow = None        # of the last evaluated point (HPWL: the optimized nets)
        self.evals = 0

    def raw_grads(self, pos):
        """-> (wirelength gradient [num_mov, 2], density gradient [num_nodes, 2]), unweighted; pos is clamped in place."""
        self.density.clamp_(pos)
        return self.wirelength.forward(pos, self.ps.wa_coeff)[1], self.density.forward(pos)[0]

    def __call__(self, pos):
        """pos must be inside the region (the optimizer keeps it there: `density.pos_lo`, `density.pos_hi`)."""
        _, wl_grad, self.hpwl = self.wirelength.forward(pos, self.ps.wa_coeff)
        grad, self.overflow = self.density.forward(pos, wl_grad, self.ps.density_weight,
                                                      self.ps.precond_coef * self.ps.density_weight)
        self.evals += 1
        return grad
