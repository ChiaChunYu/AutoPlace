"""Weighted-average (WA) wirelength: value, gradient and HPWL in one pass.

On the GPU this is one CUDA kernel per net + one per cell (cpp_to_py/gp_cuda_kernel.cu); the PyTorch code below is the
same computation, used on the CPU and as the reference the kernel is tested against.

For a net with pin coordinates x_i and smoothing gamma:
    WA = sum(x_i e^{x_i/gamma}) / sum(e^{x_i/gamma}) - sum(x_i e^{-x_i/gamma}) / sum(e^{-x_i/gamma})
The exponents are shifted by the net's max / min, so nothing overflows; the max / min are the HPWL, which
therefore costs nothing extra. Both axes are computed together ([pins, 2] tensors).
"""
import torch

from cpp_to_py import gp_cuda


class Wirelength(object):
    def __init__(self, pin_cell, pin_off, pin_net, num_nets, num_mov, ignore_net_degree):
        """pin_cell [P] long (num_mov = fixed pin), pin_off [P, 2], pin_net [P] long: all on the target device.

        Two pin sets: `all` (every net: exact HPWL, clustering) and `opt` (the optimized nets). With the CUDA kernel
        they are kept small: `all` as int32 indices (made long only while used, `pins()`), `opt` only as the
        kernel's view (pins grouped by net and by cell, int32)."""
        self.num_mov = num_mov
        self.kernel = bool(gp_cuda) and pin_off.is_cuda
        degree = torch.bincount(pin_net.cpu(), minlength=num_nets).to(pin_net.device)
        # the optimized nets: 2 <= degree <= ignore_net_degree (a huge net, e.g. reset, only adds noise to the gradient)
        use = (degree >= 2) & (degree <= ignore_net_degree)
        new_id = torch.cumsum(use.long(), 0) - 1
        sel = use[pin_net]
        self.opt = self._pin_set(pin_cell[sel], pin_off[sel], new_id[pin_net[sel]], int(use.sum()))
        self.num_opt_nets = int(use.sum())
        if self.kernel:
            self.all = {"cell": pin_cell.int(), "off": pin_off.contiguous(), "net": pin_net.int(), "num_nets": num_nets}
        else:
            self.all = self._pin_set(pin_cell, pin_off, pin_net, num_nets)

    def _pin_set(self, pin_cell, pin_off, pin_net, num_nets):
        if self.kernel:                              # the kernel's view only: pins grouped by net, and by cell
            start = lambda idx, n: torch.searchsorted(idx, torch.arange(n + 1, device=idx.device)).int()
            order = torch.argsort(pin_cell, stable=True)
            return {"off": pin_off.contiguous(), "num_nets": num_nets, "cell32": pin_cell.int(),
                    "net_start": start(pin_net, num_nets), "cell_start": start(pin_cell[order], self.num_mov),
                    "cell_pin": order.int()}
        return {"cell": pin_cell, "off": pin_off.contiguous(), "net": pin_net, "num_nets": num_nets,
                "net2": pin_net.unsqueeze(1).expand(-1, 2).contiguous()}

    def pins(self):
        """-> every pin of every net: dict(cell, off, net, num_nets) with long indices (temporary on the GPU)."""
        a = self.all
        return {"cell": a["cell"].long(), "off": a["off"], "net": a["net"].long(), "num_nets": a["num_nets"]}

    def _pin_pos(self, pos, s):
        # one extra row of zeros: a fixed pin points at it and carries its absolute position in `off`
        ext = torch.cat([pos[:self.num_mov], pos.new_zeros(1, 2)], 0)
        return ext[s["cell"]] + s["off"]

    @staticmethod
    def _net_minmax(x, s):
        shape = (s["num_nets"], 2)
        net2 = s["net2"] if "net2" in s else s["net"].unsqueeze(1).expand(-1, 2).contiguous()
        hi = x.new_zeros(shape).scatter_reduce_(0, net2, x, "amax", include_self=False)
        lo = x.new_zeros(shape).scatter_reduce_(0, net2, x, "amin", include_self=False)
        return lo, hi

    def hpwl(self, pos):
        """Exact HPWL over every net."""
        s = self.pins() if self.kernel else self.all
        lo, hi = self._net_minmax(self._pin_pos(pos, s), s)
        return (hi - lo).sum()

    def forward(self, pos, gamma):
        """-> (WA wirelength, its gradient [num_mov, 2], HPWL of the optimized nets)."""
        s = self.opt
        if self.kernel:
            grad, out = gp_cuda.wa(pos, s["cell32"], s["off"], s["net_start"], s["cell_start"], s["cell_pin"],
                                   self.num_mov, float(gamma))
            total = out.sum(0)
            return total[0], grad, total[1]
        x = self._pin_pos(pos, s)
        lo, hi = self._net_minmax(x, s)
        net, shape = s["net"], (s["num_nets"], 2)
        ep = torch.exp((x - hi[net]) / gamma)
        en = torch.exp((lo[net] - x) / gamma)
        sp = x.new_zeros(shape).index_add_(0, net, ep)
        sn = x.new_zeros(shape).index_add_(0, net, en)
        xp = x.new_zeros(shape).index_add_(0, net, x * ep) / sp
        xn = x.new_zeros(shape).index_add_(0, net, x * en) / sn
        grad_pin = ep * (1 + (x - xp[net]) / gamma) / sp[net] - en * (1 - (x - xn[net]) / gamma) / sn[net]
        grad = x.new_zeros(self.num_mov + 1, 2).index_add_(0, s["cell"], grad_pin)[:self.num_mov]
        return (xp - xn).sum(), grad, (hi - lo).sum()
