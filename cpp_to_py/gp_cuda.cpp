// Python bindings of gp_cuda_kernel.cu: tensors in, tensors out, all on the GPU.
#include <torch/extension.h>

void wa_wirelength(const float*, const int*, const float*, const int*, const int*, const int*, int, int, float, float*,
                   float*, float*);
void density_map(const float*, const float*, const float*, int, int, float, float, int, int, unsigned long long*);
void density_grad(const float*, const float*, const float*, const float*, const float*, const float*, const float*, int,
                  int, float, float, float, float, int, int, float*);

using torch::Tensor;
static const float* F(const Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32, "want a contiguous float32 CUDA tensor");
    return t.data_ptr<float>();
}
static const int* I(const Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kInt32, "want a contiguous int32 CUDA tensor");
    return t.data_ptr<int>();
}

// -> (gradient [num_mov, 2], per-net (WA, HPWL) [num_nets, 2])
std::tuple<Tensor, Tensor> wa(Tensor pos, Tensor pin_cell, Tensor pin_off, Tensor net_start, Tensor cell_start,
                              Tensor cell_pin, int num_mov, float gamma) {
    int num_nets = net_start.size(0) - 1;
    Tensor pin_grad = torch::empty_like(pin_off), out = pos.new_empty({num_nets, 2}), grad = pos.new_empty({num_mov, 2});
    wa_wirelength(F(pos), I(pin_cell), F(pin_off), I(net_start), I(cell_start), I(cell_pin), num_nets, num_mov, gamma,
                  pin_grad.data_ptr<float>(), out.data_ptr<float>(), grad.data_ptr<float>());
    return {grad, out};
}

// -> [2, nbx, nby] density of the movable cells and of the fillers, in bin areas
Tensor dmap(Tensor pos, Tensor size, Tensor ratio, int num_mov, float bx, float by, int nbx, int nby) {
    Tensor map = torch::zeros({2, nbx, nby}, pos.options().dtype(torch::kInt64));
    density_map(F(pos), F(size), F(ratio), pos.size(0), num_mov, bx, by, nbx, nby,
                (unsigned long long*)map.data_ptr<int64_t>());
    return map.to(torch::kFloat32) / 4294967296.0;
}

// field [2, 2 nbx, 2 nby]; wl_grad [num_wl, 2] (num_wl may be 0) -> preconditioned gradient [num_nodes, 2]
Tensor dgrad(Tensor pos, Tensor size, Tensor ratio, Tensor field, Tensor wl_grad, Tensor pins, Tensor area, float weight,
             float precond, float bx, float by, int nbx, int nby) {
    Tensor grad = torch::empty_like(pos);
    density_grad(F(pos), F(size), F(ratio), F(field), F(wl_grad), F(pins), F(area), pos.size(0), wl_grad.size(0), weight,
                 precond, bx, by, nbx, nby, grad.data_ptr<float>());
    return grad;
}

PYBIND11_MODULE(gp_cuda, m) {
    m.def("wa", &wa, "WA wirelength: gradient and per-net (WA, HPWL)");
    m.def("density_map", &dmap, "density maps of the movable cells and the fillers");
    m.def("density_grad", &dgrad, "gradient of the objective of every node from the field");
}
