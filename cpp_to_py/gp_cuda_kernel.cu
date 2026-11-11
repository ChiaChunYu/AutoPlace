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
                              int num_nets, int num_mov, const float* gamma, float* pin_grad, float* out) {
    float inv_gamma = __frcp_rn(*gamma);           // IEEE 1 / gamma (the same bits as on the host, fast math or not)
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
                   const int* cell_start, const int* cell_pin, int num_nets, int num_mov, const float* gamma,
                   float* pin_grad, float* out, float* grad) {
    wa_net_kernel<<<GRID(num_nets), BLOCK>>>(pos, pin_cell, pin_off, net_start, num_nets, num_mov, gamma,
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
                                    int num_wl, const float* weight_p, const float* precond_p, float inv_bx,
                                    float inv_by, int nbx, int nby, float* grad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    float weight = *weight_p, precond = *precond_p;
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

// One thread per bin: from the two fixed-point maps to what the rest of the iteration needs.
//   rho  [2 nbx][2 nby]  total density (cells + blocked area + fillers), mirrored in both axes for the FFT
//   over [nbx][nby]      density of the cells and the blocked area above the target (its sum is the overflow)
__global__ void density_post_kernel(const unsigned long long* map, const float* fixed, float target, int nbx, int nby,
                                    float* rho, float* over) {
    int b = blockIdx.x * blockDim.x + threadIdx.x, nb = nbx * nby;
    if (b >= nb) return;
    int bx = b / nby, by = b - bx * nby, w = 2 * nby, mx = 2 * nbx - 1 - bx, my = w - 1 - by;
    float mov = (float)((double)map[b] / DENSITY_SCALE) + fixed[b];
    float r = mov + (float)((double)map[nb + b] / DENSITY_SCALE);
    over[b] = fmaxf(mov - target, 0.f);
    rho[bx * w + by] = r; rho[bx * w + my] = r; rho[mx * w + by] = r; rho[mx * w + my] = r;
}

void density_map(const float* pos, const float* size, const float* ratio, int num_nodes, int num_mov, float bx,
                 float by, int nbx, int nby, unsigned long long* map, const float* fixed, float target, float* rho,
                 float* over) {
    density_map_kernel<<<GRID(num_nodes), BLOCK>>>(pos, size, ratio, num_nodes, num_mov, 1.f / bx, 1.f / by, nbx, nby, map);
    density_post_kernel<<<GRID(nbx * nby), BLOCK>>>(map, fixed, target, nbx, nby, rho, over);
}

void density_grad(const float* pos, const float* size, const float* ratio, const float* field, const float* wl_grad,
                  const float* pins, const float* area, int num_nodes, int num_wl, const float* weight,
                  const float* precond, float bx, float by, int nbx, int nby, float* grad) {
    density_grad_kernel<<<GRID(num_nodes), BLOCK>>>(pos, size, ratio, field, wl_grad, pins, area, num_nodes, num_wl,
                                                    weight, precond, 1.f / bx, 1.f / by, nbx, nby, grad);
}

// ---------------------------------------------------------------- optimizer step
// One thread per coordinate:  u_next = v - alpha g;  v_next = u_next + coef (u_next - u), kept inside [lo, hi].
__global__ void nesterov_step_kernel(const float* u, const float* v, const float* g, const float* lo, const float* hi,
                                     const float* alpha_p, const float* coef_p, int n, float* u_next, float* v_next) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float alpha = *alpha_p, coef = *coef_p;
    float un = v[i] - alpha * g[i];
    u_next[i] = un;
    v_next[i] = fminf(fmaxf(un + coef * (un - u[i]), lo[i]), hi[i]);
}

void nesterov_step(const float* u, const float* v, const float* g, const float* lo, const float* hi,
                   const float* alpha, const float* coef, int n, float* u_next, float* v_next) {
    nesterov_step_kernel<<<GRID(n), BLOCK>>>(u, v, g, lo, hi, alpha, coef, n, u_next, v_next);
}

// ---------------------------------------------------------------- control of the run, on the GPU
// One thread: what NesterovOptimizer.step (backtracking), ParamScheduler.step and ParamScheduler.need_to_stop do
// on the host, in the same order and in double precision, so that a run needs no transfer to the host per
// iteration. S is the state (layout: FIELDS in src/gpu_loop.py), kp the float parameters the other kernels read
// (gamma, density weight, density weight x precond_coef, alpha, coef), c the constants of the run.
enum { LAM, GAM, PCOEF, ALPHA, A, ITER, HPWL_PREV, MOTION, WW, MU, SETTLED, BEST_HPWL, BEST_OVF, IS_BEST, IMPROVED, STOP, BT,
       BT_TOTAL, ACCEPT, LOOP_IT, EVALS, RES_HPWL, RES_OVF, HAS_BEST };
enum { C_BASE_GAMMA, C_MOTION_UNIT, C_TOTAL_PINS, C_TOTAL_AREA, C_REF_MOTION, C_MIN_ITER, C_STOP_OVF, C_SETTLE,
       C_STALL, C_INNER, C_MAX_BT };

__global__ void control_kernel(double* S, float* kp, const double* c, const float* moved_p, const float* turned_p,
                               const float* hpwl_p, const float* ovf_p) {
    S[ACCEPT] = 0; S[IS_BEST] = 0;
    if (S[STOP] != 0) return;                                      // stopped: every later pass changes nothing
    double moved = *moved_p, turned = *turned_p, hpwl = *hpwl_p, ovf = *ovf_p;
    if (!(isfinite(moved) && isfinite(turned) && isfinite(hpwl) && isfinite(ovf))) {
        S[STOP] = 3; return;                                       // diverged: the step is not taken
    }
    double alpha_next = turned > 0 ? moved / turned : S[ALPHA];
    bool good = alpha_next > 0.95 * S[ALPHA];
    if (!good) S[BT_TOTAL] += 1;
    bool accept = good || S[BT] >= c[C_MAX_BT] - 1;
    S[ALPHA] = alpha_next;
    S[EVALS] += 1;
    if (!accept) { S[BT] += 1; kp[3] = (float)alpha_next; return; }  // the step is redone with the new alpha
    S[BT] = 0; S[ACCEPT] = 1;
    S[A] = (1 + sqrt(4 * S[A] * S[A] + 1)) / 2;
    // ParamScheduler.step
    double iter = S[ITER];
    S[MOTION] = moved / c[C_MOTION_UNIT];
    double part = S[PCOEF] * S[LAM] * c[C_TOTAL_AREA];
    S[WW] = part / (c[C_TOTAL_PINS] + part);
    double mu = hpwl < S[HPWL_PREV] ? 1.05 * 0.98 : 1.05 * fmax(pow(1.05, -S[MOTION] / c[C_REF_MOTION]), 0.95);
    S[HPWL_PREV] = hpwl; S[MU] = mu; S[LAM] *= mu;
    S[GAM] = c[C_BASE_GAMMA] * pow(10.0, (ovf - 0.1) * 20 / 9 - 1);
    if (ovf < 0.3 && S[PCOEF] < 1024 && fmod(iter, 20.0) == 0) S[PCOEF] *= 2;
    S[ITER] = iter = iter + 1;
    // ParamScheduler.need_to_stop
    double stop = 0, is_best = 0;
    if (!(hpwl < 1e30 && ovf < 1e30)) stop = 3;                                               // diverged
    else {
        if (iter >= c[C_MIN_ITER] && ovf < c[C_STOP_OVF]) {
            S[SETTLED] += 1;
            is_best = hpwl < S[BEST_HPWL];
            S[BEST_HPWL] = fmin(hpwl, S[BEST_HPWL]);
            if (S[SETTLED] > c[C_SETTLE]) stop = 1;                                         // converged
        } else if (S[SETTLED] == 0) {
            is_best = ovf < S[BEST_OVF];
            if (ovf < S[BEST_OVF] * 0.999) S[IMPROVED] = iter;
            S[BEST_OVF] = fmin(ovf, S[BEST_OVF]);
        }
        if (stop == 0) {
            if ((is_best && S[SETTLED] > 0) || S[WW] < 0.95) S[IMPROVED] = iter;
            if (iter - S[IMPROVED] > c[C_STALL]) stop = 2;                                  // stalled
        }
    }
    S[IS_BEST] = is_best;
    if (is_best) { S[RES_HPWL] = hpwl; S[RES_OVF] = ovf; S[HAS_BEST] = 1; }
    S[LOOP_IT] += 1;
    if (stop == 0 && S[LOOP_IT] >= c[C_INNER]) stop = 4;                                   // max_iter
    S[STOP] = stop;
    if (!S[HAS_BEST]) { S[RES_HPWL] = hpwl; S[RES_OVF] = ovf; }
    double a_next = (1 + sqrt(4 * S[A] * S[A] + 1)) / 2;
    kp[0] = (float)S[GAM]; kp[1] = (float)S[LAM]; kp[2] = (float)(S[PCOEF] * S[LAM]);
    kp[3] = (float)alpha_next; kp[4] = (float)((S[A] - 1) / a_next);
}

// One thread per coordinate: take the step if it was accepted, keep the placement if it is the best so far.
__global__ void select_kernel(const double* S, float* u, float* v, float* g, const float* u_next, const float* v_next,
                              const float* g_next, float* best, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || S[ACCEPT] == 0) return;
    u[i] = u_next[i]; v[i] = v_next[i]; g[i] = g_next[i];
    if (S[IS_BEST] != 0) best[i] = v_next[i];
}

void control(double* S, float* kp, const double* c, const float* moved, const float* turned, const float* hpwl,
             const float* ovf) {
    control_kernel<<<1, 1>>>(S, kp, c, moved, turned, hpwl, ovf);
}

void select_step(const double* S, float* u, float* v, float* g, const float* u_next, const float* v_next,
                 const float* g_next, float* best, int n) {
    select_kernel<<<GRID(n), BLOCK>>>(S, u, v, g, u_next, v_next, g_next, best, n);
}
