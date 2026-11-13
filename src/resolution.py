"""How the parameters of a run follow the design: every model works at the resolution of what it controls.

    cells, free area        ->  force grid: about one cell per free bin                     (force_num_bin)
    force grid / overflow   ->  refine: how many times finer the force sees than the overflow  (refine)
    grid
    overflow / refine       ->  target of the error of the wirelength model, 1 - WA / HPWL;
                                gamma (the WA smoothing) follows it                             (smoothing)
    gamma                   ->  growth of the density weight: the cells may move per iteration
                                about as far as the model resolves                             (growth)
    growth                  ->  how far gamma may move in one iteration                        (smoothing)
    gamma of the clusters   ->  hand-over to the cells, once the cluster model is as sharp as the cell level
                                starts                                                         (handover)

Each link uses what the one before it set, so a change upstream (smaller cells, a finer grid) carries through: a
finer grid asks a sharper wirelength model, and a sharper model a more careful density weight. The host scheduler
calls these functions; control_kernel (cpp_to_py/gp_cuda_kernel.cu) computes the same in double precision.
"""
import math


def force_num_bin(num_nodes, free_fraction, num_bin):
    """Bins per side of the density force: 2^round(log2 sqrt(nodes / free fraction)), never coarser than `num_bin`
    (the grid of the overflow). A cell is widened to sqrt(2) bins in the density, so with many small cells per bin
    the force cannot see clumps of bin size."""
    return max(num_bin, 2 ** int(round(math.log2(math.sqrt(num_nodes / max(free_fraction, 1e-9))))))


def refine(density):
    """How many times finer per side the force grid of `density` is than the grid its overflow is measured on."""
    return density.nbx / density.judge.nbx if density.judge is not None else 1.0


def smoothing(gamma, hpwl, wa, overflow, refine, unit, mu):
    """Next WA smoothing: gamma x sqrt(target / error), kept within 0.01 to 100 x `unit`; per iteration it moves by
    at most the factor the density weight moves by (mu, or 1 / mu), so the model and the weight keep one pace.
    error = 1 - WA / HPWL, the measured error of the model; target = overflow / refine: a smooth model while much is
    still to spread, and as sharp as the density force resolves at the end."""
    error = max(1e-6, 1 - wa / max(hpwl, 1e-12))
    m = max(mu, 1.0 / mu)
    g = gamma * min(m, max(1.0 / m, math.sqrt(max(overflow, 1e-3) / refine / error)))
    return min(100 * unit, max(0.01 * unit, g))


def growth(motion, pitch, gamma, hpwl_fell):
    """Factor of the density weight for the next iteration: 1.05 ^ (1 - move / gamma), at least 1.05 x 0.95, with
    move = motion x pitch the RMS move of the nodes in the last iteration; 1.05 x 0.98 while the HPWL falls (the
    wirelength is still catching up)."""
    if hpwl_fell:
        return 1.05 * 0.98
    return 1.05 * max(1.05 ** (-motion * pitch / gamma), 0.95)


def handover(gamma, start_gamma):
    """The clusters hand over to the cells once their wirelength model is as sharp as the one the cells start with:
    from there the clusters cannot resolve anything the cells would not."""
    return gamma <= start_gamma
