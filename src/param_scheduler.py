"""The parameters that change during global placement, and when to stop.

    density_weight  starts so that the density gradient is `--density_weight` of the wirelength gradient, then is
                    multiplied every iteration by about 1.05, less when the wirelength rises quickly (ePlace);
                    every 3rd iteration only at the start and while wirelength and density are balanced
    wa_coeff        the smoothing gamma of the WA wirelength: large (smooth) at high overflow, small (close to HPWL)
                    at low overflow
    precond_coef    the preconditioner of a node is #pins + precond_coef x density_weight x area, at least 1 (the
                    diagonal the gradient is divided by); precond_coef doubles every 20 iterations once the overflow
                    is under 0.3
"""
import math

import numpy as np
import torch


class ParamScheduler(object):
    def __init__(self, data, args, logger):
        self.args, self.logger = args, logger
        self.iter = 0
        self.hpwl, self.overflow = [], []
        self.base_gamma = args.wa_coeff * (data.bin[0] + data.bin[1])
        self.wa_coeff = 10 * self.base_gamma
        self.density_weight = 0.0
        self.precond_coef = 1.0
        self.total_pins, self.total_area = data.total_pins, data.total_area      # over every node (fillers: 0 pins)
        self.weighted_weight = 0.0              # share of the density term in the preconditioner, 0..1
        self.mu = 1.0
        self.last_enlarge = -args.enlarge_interval
        self.settled, self.best_hpwl, self.best_overflow, self.is_best, self.improved_iter = 0, float("inf"), float("inf"), False, 0

    def set_init_param(self, wl_grad, density_grad):
        self.density_weight = self.args.density_weight * (wl_grad.norm(p=1) / density_grad.norm(p=1)).item()

    def step(self, hpwl, overflow):
        density_part = self.precond_coef * self.density_weight * self.total_area
        self.weighted_weight = density_part / (self.total_pins + density_part)
        self.hpwl.append(hpwl)
        self.overflow.append(overflow)
        # slow the updates down (every 3rd iteration) at the start and while wirelength and density are balanced
        skip = (self.iter < 50 or 0.5 < self.weighted_weight < 0.95) and self.iter % 3 != 0
        if self.iter >= 1 and not skip:
            self.mu = self.growth()
            self.density_weight *= self.mu
            self.wa_coeff = self.base_gamma * 10 ** ((overflow - 0.1) * 20 / 9 - 1)
        if (self.iter > 15 and overflow > 0.9 and self.iter - self.last_enlarge > self.args.enlarge_interval
                and self._plateau(self.overflow, 25, 0.001)):
            self.last_enlarge = self.iter      # the cells have not started to spread: the density weight is too small
            self.density_weight *= 2
            self.logger.info("overflow plateau at iter %d: density weight doubled" % self.iter)
        if overflow < 0.3 and self.precond_coef < 1024 and self.iter % 20 == 0:
            self.precond_coef *= 2
        self.iter += 1

    def growth(self):
        """ePlace / Xplace: about 1.05 per iteration, less when the HPWL rose by a large part of --ref_hpwl."""
        delta = self.hpwl[-1] - self.hpwl[-2]
        if delta < 0:
            return 1.05 * max(0.9999 ** self.iter, 0.98)
        return 1.05 * float(np.clip(1.05 ** (-delta / self.args.ref_hpwl), 0.95, 1.05))

    @staticmethod
    def _plateau(x, window, threshold):
        if len(x) < window:
            return False
        x = x[-window:]
        return (max(x) - min(x)) / (sum(x) / len(x)) < threshold

    def need_to_stop(self):
        """-> reason to stop, or None; sets `is_best` when the caller should keep the current placement.

        The kept placement is the one with the lowest overflow until the overflow gets below --stop_overflow; from
        there the placement keeps improving for a while (wirelength and overflow both fall), so --settle_iter more
        iterations are run and the one with the lowest HPWL is kept ("converged"). When the density term dominates
        (weighted_weight >= 0.95) and nothing has improved for --stall_iter iterations, the run ends with what it
        kept ("stalled": the overflow cannot get lower, e.g. bins smaller than a cell, and the density weight would
        grow without bound). A wirelength that is not a finite
        positive number ends the run too ("diverged").
        """
        hpwl, overflow = self.hpwl[-1], self.overflow[-1]
        self.is_best = False
        if not (0 <= hpwl < 1e30 and 0 <= overflow < 1e30):
            return "diverged"
        if self.iter >= self.args.min_iter and overflow < self.args.stop_overflow:
            self.settled += 1
            self.is_best = hpwl < self.best_hpwl
            self.best_hpwl = min(hpwl, self.best_hpwl)
            if self.settled > self.args.settle_iter:
                return "converged"
        elif self.settled == 0:
            self.is_best = overflow < self.best_overflow
            if overflow < self.best_overflow * 0.999:               # a real improvement, not the last digits
                self.improved_iter = self.iter
            self.best_overflow = min(overflow, self.best_overflow)
        if (self.is_best and self.settled) or self.weighted_weight < 0.95:
            self.improved_iter = self.iter      # the clock runs only once the density term dominates: before that the
                                                # overflow is not expected to fall (wirelength-driven start)
        if self.iter - self.improved_iter > self.args.stall_iter:
            return "stalled"
        return None
