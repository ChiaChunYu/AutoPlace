"""The parameters that change during global placement, and when to stop.

    density_weight  0 during the wirelength-only start, which ends when the WA wirelength has converged
                    (`--start_tol`); then set so that the density force on the cells is `--density_weight` of the
                    full wire tension (every pin pulling), and from there multiplied every iteration by at most
                    1.05: less the further the nodes moved in the last iteration, against the smoothing gamma of the
                    wirelength model (how far it resolves the order of cells), and less when the HPWL fell (the wirelength is still catching up); updated
                    every iteration (the motion is what slows it down, no fixed slow window)
    wa_coeff        the smoothing gamma of the WA wirelength, starting at 10 x the unit (`--wa_coeff` x bin sizes).
                    --wa_rule accuracy: every iteration multiplied by sqrt(target / error), within 0.9 to 1.1, where
                    error = 1 - WA / HPWL is the measured error of the model and target = overflow / refine (refine =
                    how many times finer the force grid is than the overflow grid): a smooth model while much is to
                    spread, an accurate one at the end, as accurate as the density force can resolve; kept within
                    0.01 to 100 x the unit. --wa_rule overflow: Xplace's formula of the overflow
    precond_coef    the preconditioner of a node is #pins + precond_coef x density_weight x area, at least 1 (the
                    diagonal the gradient is divided by); precond_coef doubles every 20 iterations once the overflow
                    is under 0.3
"""
import math


class ParamScheduler(object):
    def __init__(self, data, args, logger):
        self.args, self.logger = args, logger
        self.iter = 0
        self.wa, self.hpwl, self.overflow = [], [], []
        self.base_gamma = args.wa_coeff * (data.bin[0] + data.bin[1])
        self.refine = data.refine               # force bins / overflow bins per side
        self.wa_coeff = 10 * self.base_gamma
        self.density_weight = 0.0
        self.precond_coef = 1.0
        self.total_pins, self.total_area = data.total_pins, data.total_area      # over every node (fillers: 0 pins)
        self.pitch = data.pitch
        self.motion_unit = data.pitch * math.sqrt(data.num_nodes)     # L2 norm of "every node moved one cell pitch"
        self.motion = 0.0                       # RMS move of the nodes in the last iteration, in cell pitches
        self.weighted_weight = 0.0              # share of the density term in the preconditioner, 0..1
        self.mu = 1.0
        self.settled, self.best_hpwl, self.best_overflow, self.is_best, self.improved_iter = 0, float("inf"), float("inf"), False, 0

    def start_step(self, hpwl, wa, window=5):
        """One iteration of the wirelength-only start -> True when it is over: the WA wirelength fell by less than
        --start_tol (relative) per iteration over the last `window` iterations."""
        self.hpwl.append(hpwl)
        self.wa.append(wa)
        self.iter += 1
        return not (len(self.wa) <= window or self.wa[-1 - window] - wa > self.args.start_tol * window * wa)

    def gamma(self, overflow):
        return self.base_gamma * 10 ** ((overflow - 0.1) * 20 / 9 - 1)

    def resume(self, coarse, density_weight, hpwl, overflow):
        """Go on where the run on the clusters (`coarse`, its scheduler) stopped; no wirelength-only start."""
        self.density_weight, self.precond_coef, self.iter = density_weight, coarse.precond_coef, coarse.iter
        if self.args.wa_rule == "accuracy":     # the same smoothing in units of the bin
            self.wa_coeff = coarse.wa_coeff / coarse.base_gamma * self.base_gamma
        else:
            self.wa_coeff = self.gamma(overflow)
        self.hpwl.append(hpwl)

    def step(self, hpwl, overflow, moved, wa):
        """moved: L2 norm of the move of all nodes in the last iteration (the optimizer has it); wa: WA wirelength."""
        self.motion = moved / self.motion_unit
        density_part = self.precond_coef * self.density_weight * self.total_area
        self.weighted_weight = density_part / (self.total_pins + density_part)
        self.hpwl.append(hpwl)
        self.overflow.append(overflow)
        self.mu = self.growth()
        self.density_weight *= self.mu
        self.wa_coeff = self.smoothing(hpwl, overflow, wa)
        if overflow < 0.3 and self.precond_coef < 1024 and self.iter % 20 == 0:
            self.precond_coef *= 2
        self.iter += 1

    def smoothing(self, hpwl, overflow, wa):
        if self.args.wa_rule != "accuracy":
            return self.gamma(overflow)
        error = max(1e-6, 1 - wa / max(hpwl, 1e-12))
        g = self.wa_coeff * min(1.1, max(0.9, math.sqrt(max(overflow, 1e-3) / self.refine / error)))
        return min(100 * self.base_gamma, max(0.01 * self.base_gamma, g))

    def growth(self):
        """1.05 ^ (1 - move / gamma), not below 1.05 x 0.95, move = the RMS move of the nodes in the last iteration:
        the wirelength model resolves the order of cells closer than about its smoothing gamma, so the weight grows
        slower when the nodes move further than that per iteration (the model can no longer follow). 1.05 x 0.98
        when the HPWL fell."""
        if self.hpwl[-1] < self.hpwl[-2]:
            return 1.05 * 0.98
        return 1.05 * max(1.05 ** (-self.motion * self.pitch / self.wa_coeff), 0.95)

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
