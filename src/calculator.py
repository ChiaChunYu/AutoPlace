"""The gradient of the objective of global placement:  WA wirelength + density_weight x density energy,
divided per node by the preconditioner max(1, #pins + precond_coef x density_weight x area).

The value of the objective is not needed by the optimizer, so it is never computed; the HPWL and the overflow of the
evaluated point come for free from the same pass.

While the density weight is 0 (the wirelength-only start) the density map and the field are not computed at all.
"""
import torch


class Objective(object):
    def __init__(self, wirelength, density, ps):
        self.wirelength, self.density, self.ps = wirelength, density, ps
        self.wa = self.hpwl = self.overflow = None      # of the last evaluated point (WA, HPWL: the optimized nets)
        self.evals = 0
        self.not_measured = torch.tensor(float("nan"), device=density.device)
        self.pin_precond = density.pins[:density.num_mov].clamp(min=1.0).unsqueeze(1)

    def tension_unit(self, pos):
        """The density weight at which the density force on the cells equals every pin pulling at full tension:
        #pins / L1 norm of the density gradient. The start of the density weight is given in this unit."""
        n = self.density.num_mov
        return (self.density.pins.sum() / self.density.forward(pos)[0][:n].abs().sum()).item()

    def __call__(self, pos):
        """pos must be inside the region (the optimizer keeps it there: `density.pos_lo`, `density.pos_hi`)."""
        self.wa, wl_grad, self.hpwl = self.wirelength.forward(pos, self.ps.wa_coeff)
        if self.ps.density_weight == 0:
            grad, self.overflow = torch.zeros_like(pos), self.not_measured
            grad[:len(wl_grad)] = wl_grad / self.pin_precond
        else:
            grad, self.overflow = self.density.forward(pos, wl_grad, self.ps.density_weight,
                                                       self.ps.precond_coef * self.ps.density_weight)
        self.evals += 1
        return grad

    def on_gpu(self, pos, kp):
        """The same gradient with the parameters in device memory: kp = (wa smoothing, density weight,
        density weight x precond_coef, ...) as set by the control kernel (src/gpu_loop.py)."""
        self.wa, wl_grad, self.hpwl = self.wirelength.forward(pos, kp[0:1])
        grad, self.overflow = self.density.forward(pos, wl_grad, kp[1:2], kp[2:3])
        self.evals += 1
        return grad
