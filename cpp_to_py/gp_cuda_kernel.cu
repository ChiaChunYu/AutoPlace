// CUDA kernels of the global placement. Raw pointers only (no torch headers): the wrappers are in gp_cuda.cpp.
// Every kernel gives the same bits on every run: sums are taken in a fixed order, or on integers.
#include <cuda_runtime.h>

#define BLOCK 256
#define GRID(n) (((n) + BLOCK - 1) / BLOCK)

// ---------------------------------------------------------------- weighted-average wirelength
// One thread per net, both axes. Pins of net i are [net_start[i], net_start[i + 1]). A pin of a movable cell is at
// pos[cell] + off; a fixed pin (cell == num_mov) is at off. Writes the gradient of every pin and, per net,
// out[i] = (WA wirelength, HPWL).
__global__ void wa_net_kernel(const float* pos, const int* pin_cell, const float* pin_off, const int* net_start,
                              int num_nets, int num_mov, float inv_gamma, float* pin_grad, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nets) return;
    int lo_pin = net_start[i], hi_pin = net_start[i + 1];
    float wa = 0, hpwl = 0;
    for (int a = 0; a < 2; a++) {
        float hi = -1e30f, lo = 1e30f;
        for (int p = lo_pin; p < hi_pin; p++) {
            int c = pin_cell[p];
            float x = (c < num_mov ? pos[2 * c + a] : 0.f) + pin_off[2 * p + a];
            pin_grad[2 * p + a] = x;                     // keep the coordinate for the next two passes
            hi = fmaxf(hi, x);
            lo = fminf(lo, x);
        }
        float sp = 0, sn = 0, xp = 0, xn = 0;
        for (int p = lo_pin; p < hi_pin; p++) {
            float x = pin_grad[2 * p + a], ep = expf((x - hi) * inv_gamma), en = expf((lo - x) * inv_gamma);
            sp += ep; sn += en; xp += x * ep; xn += x * en;
        }
        xp /= sp; xn /= sn;
        for (int p = lo_pin; p < hi_pin; p++) {
            float x = pin_grad[2 * p + a], ep = expf((x - hi) * inv_gamma), en = expf((lo - x) * inv_gamma);
            pin_grad[2 * p + a] = ep * (1 + (x - xp) * inv_gamma) / sp - en * (1 - (x - xn) * inv_gamma) / sn;
        }
        wa += xp - xn;
        hpwl += hi - lo;
    }
    out[2 * i] = wa;
    out[2 * i + 1] = hpwl;
}

// One thread per cell: the sum of its pins' gradients. Pins of cell c are cell_pin[cell_start[c] .. cell_start[c + 1]).
__global__ void wa_cell_kernel(const float* pin_grad, const int* cell_start, const int* cell_pin, int num_mov,
                               float* grad) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_mov) return;
    float gx = 0, gy = 0;
    for (int k = cell_start[c]; k < cell_start[c + 1]; k++) {
        gx += pin_grad[2 * cell_pin[k]];
        gy += pin_grad[2 * cell_pin[k] + 1];
    }
    grad[2 * c] = gx;
    grad[2 * c + 1] = gy;
}

void wa_wirelength(const float* pos, const int* pin_cell, const float* pin_off, const int* net_start,
                   const int* cell_start, const int* cell_pin, int num_nets, int num_mov, float gamma,
                   float* pin_grad, float* out, float* grad) {
    wa_net_kernel<<<GRID(num_nets), BLOCK>>>(pos, pin_cell, pin_off, net_start, num_nets, num_mov, 1.f / gamma,
                                             pin_grad, out);
    wa_cell_kernel<<<GRID(num_mov), BLOCK>>>(pin_grad, cell_start, cell_pin, num_mov, grad);
}

// ---------------------------------------------------------------- density map and its gradient
// One thread per node. `size` is the smoothed size and `ratio` the density scale that keeps the charge. The overlap
// of the node with each bin, in bin areas, goes to map 0 (movable cells) or map 1 (fillers) as fixed point
// (DENSITY_SCALE per unit): integer adds do not depend on their order.
#define DENSITY_SCALE 4294967296.0     // 2^32

struct Span { float lo, hi; int b0, b1; };
__device__ inline Span span(float centre, float size, float inv_bin, int nb) {
    Span s;
    s.lo = (centre - 0.5f * size) * inv_bin;
    s.hi = (centre + 0.5f * size) * inv_bin;
    s.b0 = max((int)floorf(s.lo), 0);
    s.b1 = min((int)floorf(s.hi), nb - 1);
    return s;
}
__device__ inline float overlap(const Span& s, int b) { return fmaxf(fminf(s.hi, b + 1.f) - fmaxf(s.lo, (float)b), 0.f); }

__global__ void density_map_kernel(const float* pos, const float* size, const float* ratio, int num_nodes, int num_mov,
                                   float inv_bx, float inv_by, int nbx, int nby, unsigned long long* map) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    Span sx = span(pos[2 * i], size[2 * i], inv_bx, nbx), sy = span(pos[2 * i + 1], size[2 * i + 1], inv_by, nby);
    unsigned long long* m = map + (i < num_mov ? 0 : nbx * nby);
    for (int bx = sx.b0; bx <= sx.b1; bx++) {
        float wx = ratio[i] * overlap(sx, bx);
        for (int by = sy.b0; by <= sy.b1; by++)
            atomicAdd(&m[bx * nby + by], (unsigned long long)((double)(wx * overlap(sy, by)) * DENSITY_SCALE));
    }
}

// The gradient of the whole objective for node i, preconditioned:
//   (wl_grad[i] + weight x density gradient) / max(1, pins[i] + precond x area[i])
// density gradient = -(charge of the node in each bin) x (field of the bin). `field` is the transform output as it
// is: two planes (x, y) of the mirrored map, each [2 nbx][2 nby]. Nodes from num_wl on (fillers) have no wl_grad;
// precond < 0 switches the division off.
__global__ void density_grad_kernel(const float* pos, const float* size, const float* ratio, const float* field,
                                    const float* wl_grad, const float* pins, const float* area, int num_nodes,
                                    int num_wl, float weight, float precond, float inv_bx, float inv_by, int nbx,
                                    int nby, float* grad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    Span sx = span(pos[2 * i], size[2 * i], inv_bx, nbx), sy = span(pos[2 * i + 1], size[2 * i + 1], inv_by, nby);
    const float* fy = field + 4 * nbx * nby;
    float gx = 0, gy = 0;
    for (int bx = sx.b0; bx <= sx.b1; bx++) {
        float wx = ratio[i] * overlap(sx, bx);
        for (int by = sy.b0; by <= sy.b1; by++) {
            float w = wx * overlap(sy, by);
            gx -= w * field[bx * 2 * nby + by];
            gy -= w * fy[bx * 2 * nby + by];
        }
    }
    gx *= weight; gy *= weight;
    if (i < num_wl) { gx += wl_grad[2 * i]; gy += wl_grad[2 * i + 1]; }
    float scale = precond < 0 ? 1.f : 1.f / fmaxf(pins[i] + precond * area[i], 1.f);
    grad[2 * i] = gx * scale;
    grad[2 * i + 1] = gy * scale;
}

void density_map(const float* pos, const float* size, const float* ratio, int num_nodes, int num_mov, float bx,
                 float by, int nbx, int nby, unsigned long long* map) {
    density_map_kernel<<<GRID(num_nodes), BLOCK>>>(pos, size, ratio, num_nodes, num_mov, 1.f / bx, 1.f / by, nbx, nby, map);
}

void density_grad(const float* pos, const float* size, const float* ratio, const float* field, const float* wl_grad,
                  const float* pins, const float* area, int num_nodes, int num_wl, float weight, float precond, float bx,
                  float by, int nbx, int nby, float* grad) {
    density_grad_kernel<<<GRID(num_nodes), BLOCK>>>(pos, size, ratio, field, wl_grad, pins, area, num_nodes, num_wl,
                                                    weight, precond, 1.f / bx, 1.f / by, nbx, nby, grad);
}
