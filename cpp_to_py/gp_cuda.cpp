// Python bindings of gp_cuda_kernel.cu: tensors in, tensors out, all on the GPU.
#include <torch/extension.h>

void wa_wirelength(const float*, const int*, const float*, const int*, const int*, const int*, int, int, const float*,
                   float*, float*, float*);
void density_map(const float*, const float*, const float*, int, int, float, float, int, int, unsigned long long*,
                 const float*, float, float*, float*);
void nesterov_step(const float*, const float*, const float*, const float*, const float*, const float*, const float*, int,
                   float*, float*);
void density_grad(const float*, const float*, const float*, const float*, const float*, const float*, const float*, int,
                  int, const float*, const float*, float, float, int, int, float*);
void control(double*, float*, const double*, const float*, const float*, const float*, const float*, const float*);
void select_step(const double*, float*, float*, float*, const float*, const float*, const float*, float*, int);

using torch::Tensor;
static const float* F(const Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32, "want a contiguous float32 CUDA tensor");
    return t.data_ptr<float>();
}
static double* D(const Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat64, "want a contiguous float64 CUDA tensor");
    return t.data_ptr<double>();
}
static float* W(const Tensor& t) { return const_cast<float*>(F(t)); }
static const int* I(const Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kInt32, "want a contiguous int32 CUDA tensor");
    return t.data_ptr<int>();
}

// -> (gradient [num_mov, 2], per-net (WA, HPWL) [num_nets, 2])
std::tuple<Tensor, Tensor> wa(Tensor pos, Tensor pin_cell, Tensor pin_off, Tensor net_start, Tensor cell_start,
                              Tensor cell_pin, int num_mov, Tensor gamma) {
    int num_nets = net_start.size(0) - 1;
    Tensor pin_grad = torch::empty_like(pin_off), out = pos.new_empty({num_nets, 2}), grad = pos.new_empty({num_mov, 2});
    wa_wirelength(F(pos), I(pin_cell), F(pin_off), I(net_start), I(cell_start), I(cell_pin), num_nets, num_mov, F(gamma),
                  pin_grad.data_ptr<float>(), out.data_ptr<float>(), grad.data_ptr<float>());
    return {grad, out};
}

// -> (total density mirrored for the FFT [2 nbx, 2 nby], density of the cells + blocked area above the target [nbx, nby])
std::tuple<Tensor, Tensor> dmap(Tensor pos, Tensor size, Tensor ratio, int num_mov, float bx, float by, int nbx, int nby,
                                Tensor fixed, float target) {
    Tensor map = torch::zeros({2, nbx, nby}, pos.options().dtype(torch::kInt64));
    Tensor rho = pos.new_empty({2 * nbx, 2 * nby}), over = pos.new_empty({nbx, nby});
    density_map(F(pos), F(size), F(ratio), pos.size(0), num_mov, bx, by, nbx, nby,
                (unsigned long long*)map.data_ptr<int64_t>(), F(fixed), target, rho.data_ptr<float>(), over.data_ptr<float>());
    return {rho, over};
}

// -> (u_next, v_next) of Nesterov's method; v_next is kept inside [lo, hi]
std::tuple<Tensor, Tensor> nstep(Tensor u, Tensor v, Tensor g, Tensor lo, Tensor hi, Tensor alpha, Tensor coef) {
    Tensor u_next = torch::empty_like(v), v_next = torch::empty_like(v);
    nesterov_step(F(u), F(v), F(g), F(lo), F(hi), F(alpha), F(coef), v.numel(), u_next.data_ptr<float>(),
                  v_next.data_ptr<float>());
    return {u_next, v_next};
}

// field [2, 2 nbx, 2 nby]; wl_grad [num_wl, 2] (num_wl may be 0) -> preconditioned gradient [num_nodes, 2]
Tensor dgrad(Tensor pos, Tensor size, Tensor ratio, Tensor field, Tensor wl_grad, Tensor pins, Tensor area, Tensor weight,
             Tensor precond, float bx, float by, int nbx, int nby) {
    Tensor grad = torch::empty_like(pos);
    density_grad(F(pos), F(size), F(ratio), F(field), F(wl_grad), F(pins), F(area), pos.size(0), wl_grad.size(0),
                 F(weight), F(precond), bx, by, nbx, nby, grad.data_ptr<float>());
    return grad;
}

// one pass of the run's control (see control_kernel): state S [float64], kernel parameters kp [float32 x 5],
// constants c [float64]; moved / turned / hpwl / overflow / WA are 0-d float32 tensors of the pass
void ctrl(Tensor S, Tensor kp, Tensor c, Tensor moved, Tensor turned, Tensor hpwl, Tensor ovf, Tensor wa) {
    control(D(S), W(kp), D(c), F(moved), F(turned), F(hpwl), F(ovf), F(wa));
}

// take the pass's step if it was accepted (u, v, g in place), and copy it to best if it is the best so far
void sel(Tensor S, Tensor u, Tensor v, Tensor g, Tensor u_next, Tensor v_next, Tensor g_next, Tensor best) {
    select_step(D(S), W(u), W(v), W(g), F(u_next), F(v_next), F(g_next), W(best), v.numel());
}

PYBIND11_MODULE(gp_cuda, m) {
    m.def("wa", &wa, "WA wirelength: gradient and per-net (WA, HPWL)");
    m.def("density_map", &dmap, "total density (mirrored) and the density above the target");
    m.def("nesterov_step", &nstep, "u_next and the clamped v_next");
    m.def("density_grad", &dgrad, "gradient of the objective of every node from the field");
    m.def("control", &ctrl, "one pass of the run's control on the GPU (step size, density weight, stop)");
    m.def("select", &sel, "take an accepted step and keep the best placement");
}
