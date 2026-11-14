"""11-09: the release re-distance on the card - core/brush.redistance's own discrete scheme, run in parallel.

Why: on her 0.25 mm lattice one CPU pass over a large stroke took 7.6-9.2 s (the engine blocked). The CPU scheme, kept
exactly so the two agree to well under a micron (and a replay without a card reproduces it):
  F at the region's nodes (+ a 1-brick halo): the card's field program (gpu/geval.py; = the CPU's to ~0.01 um);
  |grad F| near the surface (brush._near_unit_gradient): unchanged test, percentiles on the card;
  seeds (brush.sweep_distance's first loop): node by node, the same formula;
  the fast sweep (8 directions, Gauss-Seidel): run plane by plane (i + j + k = const in each direction's order): a node's
  update reads only its 6 neighbours, and in both orders the 3 upstream ones are already updated and the 3 downstream
  ones are not, so every update sees the same values as in the CPU's lexicographic loop; the same stop (an iteration
  that changes nothing by 1e-3 h) and the same 4-iteration cap;
  the edge ramp (brush.edge_ramp: city-block distance to the region's edge / 16, capped at 1): 16 Jacobi steps;
  the delta update, the finite guard, the snapshots and the brick Lipschitz bounds: as brush.redistance.
redistance_gpu(scene, oid, bricks, rec, ramp, band, src, force) has brush.redistance's signature and return."""
import math
import os
import time

import numpy as np

from ..core import brush as BR

_SRC = r'''
#define BIG 1e30
extern "C" __global__ void scatter_bricks(const float* vals, const long long* off, int n, int nj, int nk, double* F) {
    // vals (n, 729): 9^3 nodes per brick; node (a,b,c) < 8 goes to F[(off_i + a) * nj * nk + ...]
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)n * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int a = q / 64, bb = (q / 8) % 8, c = q % 8;
    long long ox = off[3 * b], oy = off[3 * b + 1], oz = off[3 * b + 2];
    F[((ox + a) * nj + (oy + bb)) * nk + (oz + c)] = (double)vals[(size_t)b * 729 + (a * 9 + bb) * 9 + c];
}

extern "C" __global__ void scatter8(const float* vals, const long long* off, int n, int nj, int nk, double* F) {
    // vals (n, 512): each brick's own 8^3 nodes (geval n8)
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)n * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int a = q / 64, bb = (q / 8) % 8, c = q % 8;
    long long ox = off[3 * b], oy = off[3 * b + 1], oz = off[3 * b + 2];
    F[((ox + a) * nj + (oy + bb)) * nk + (oz + c)] = (double)vals[id];
}

// 11-09: brush.brick_lipschitz / brick_lipschitz_out on the card. One block per brick; delta nodes read through a
// brick table T over bricks [tlo, tlo + tn) (row into rows, -1: no slot -> 0, as program._node); the arithmetic in
// double with explicit rounding (no fused multiply-add), so the bounds equal the CPU kernels'
__device__ __forceinline__ double lip_node(const int* T, const float* rows, int tx, int ty, int tz, int tnx, int tny,
                                           int tnz, int i, int j, int k) {
    int bx = (i >> 3) - tx, by = (j >> 3) - ty, bz = (k >> 3) - tz;
    if (bx < 0 || by < 0 || bz < 0 || bx >= tnx || by >= tny || bz >= tnz) return 0.0;
    int r = T[(bx * tny + by) * tnz + bz];
    if (r < 0) return 0.0;
    return (double)rows[(size_t)r * 512 + ((i & 7) * 8 + (j & 7)) * 8 + (k & 7)];
}

extern "C" __global__ void lip_bricks(const long long* bricks, int nb, const int* T, const float* rows, int tx, int ty,
                                      int tz, int tnx, int tny, int tnz, double h, const double* Fn, int i0, int j0,
                                      int k0, int ni, int nj, int nk, double thr, double* out) {
    // Fn null: every cell (brick_lipschitz); else the cells with a node Fn >= -thr, unknown (1e30) or outside Fn
    __shared__ double best[256];
    int q = blockIdx.x;
    if (q >= nb) return;
    int bi = (int)bricks[3 * q] * 8, bj = (int)bricks[3 * q + 1] * 8, bk = (int)bricks[3 * q + 2] * 8;
    double m = 0.0;
    for (int c = threadIdx.x; c < 729; c += blockDim.x) {
        int i = bi - 1 + c / 81, j = bj - 1 + (c / 9) % 9, k = bk - 1 + c % 9;
        if (Fn) {
            bool use = false;
            for (int t = 0; t < 8 && !use; t++) {
                int a = i + (t & 1) - i0, b = j + ((t >> 1) & 1) - j0, cc = k + ((t >> 2) & 1) - k0;
                if (a < 0 || b < 0 || cc < 0 || a >= ni || b >= nj || cc >= nk) { use = true; break; }
                double f = Fn[((long long)a * nj + b) * nk + cc];
                if (f >= 1e29 || f >= -thr) use = true;
            }
            if (!use) continue;
        }
        double v[8];
        for (int t = 0; t < 8; t++)
            v[t] = lip_node(T, rows, tx, ty, tz, tnx, tny, tnz, i + (t & 1), j + ((t >> 1) & 1), k + ((t >> 2) & 1));
        double gx = fmax(fmax(fabs(v[1] - v[0]), fabs(v[3] - v[2])), fmax(fabs(v[5] - v[4]), fabs(v[7] - v[6])));
        double gy = fmax(fmax(fabs(v[2] - v[0]), fabs(v[3] - v[1])), fmax(fabs(v[6] - v[4]), fabs(v[7] - v[5])));
        double gz = fmax(fmax(fabs(v[4] - v[0]), fabs(v[5] - v[1])), fmax(fabs(v[6] - v[2]), fabs(v[7] - v[3])));
        double g = __ddiv_rn(sqrt(__dadd_rn(__dadd_rn(__dmul_rn(gx, gx), __dmul_rn(gy, gy)), __dmul_rn(gz, gz))), h);
        if (g > m) m = g;
    }
    best[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && best[threadIdx.x + s] > best[threadIdx.x]) best[threadIdx.x] = best[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[q] = best[0];
}

extern "C" __global__ void grad_sel(const double* F, const unsigned char* reg, int ni, int nj, int nk, double h,
                                    double band, double* g) {
    // brush._near_unit_gradient on interior nodes: |grad F| (central) where the node and its 6 neighbours are known,
    // reg and |F| < band; else -1
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)ni * nj * nk;
    if (id >= N) return;
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    g[id] = -1.0;
    if (i < 1 || j < 1 || k < 1 || i >= ni - 1 || j >= nj - 1 || k >= nk - 1) return;
    double f = F[id];
    if (f >= 1e29 || !reg[id]) return;
    long long sx = (long long)nj * nk, sy = nk;
    double a0 = F[id + sx], a1 = F[id - sx], b0 = F[id + sy], b1 = F[id - sy], c0 = F[id + 1], c1 = F[id - 1];
    if (a0 >= 1e29 || a1 >= 1e29 || b0 >= 1e29 || b1 >= 1e29 || c0 >= 1e29 || c1 >= 1e29) return;
    if (fabs(f) >= band) return;
    double gx = a0 - a1, gy = b0 - b1, gz = c0 - c1;
    g[id] = sqrt(gx * gx + gy * gy + gz * gz) / (2.0 * h);
}

extern "C" __global__ void seeds(const double* F, const unsigned char* act, int ni, int nj, int nk, double h,
                                 double* U, unsigned char* fixedv) {
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)ni * nj * nk;
    if (id >= N) return;
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    U[id] = BIG; fixedv[id] = 0;
    if (!act[id]) return;
    double f = F[id];
    if (f == 0.0) { U[id] = 0.0; fixedv[id] = 1; return; }
    int dims[3] = {ni, nj, nk}, p[3] = {i, j, k};
    long long st[3] = {(long long)nj * nk, (long long)nk, 1};
    double inv = 0.0, mb = BIG;
    for (int ax = 0; ax < 3; ax++) {
        double best = BIG;
        for (int sg = -1; sg <= 1; sg += 2) {
            int q = p[ax] + sg;
            if (q < 0 || q >= dims[ax]) continue;
            long long nid = id + sg * st[ax];
            if (!act[nid]) continue;
            double gv = F[nid];
            if ((f > 0.0) != (gv > 0.0)) {
                double th = f / (f - gv);
                if (th * h < best) best = th * h;
            }
        }
        if (best < BIG) { inv += 1.0 / fmax(best * best, 1e-300); mb = fmin(mb, best); }
    }
    if (inv > 0.0) {
        double gg = 0.0;
        for (int ax = 0; ax < 3; ax++) {
            bool okp = p[ax] + 1 < dims[ax] && act[id + st[ax]];
            bool okm = p[ax] - 1 >= 0 && act[id - st[ax]];
            double dd;
            if (okp && okm) dd = (F[id + st[ax]] - F[id - st[ax]]) / (2.0 * h);
            else if (okp) dd = (F[id + st[ax]] - f) / h;
            else if (okm) dd = (f - F[id - st[ax]]) / h;
            else dd = 0.0;
            gg += dd * dd;
        }
        if (gg > 1e-12) U[id] = fmin(fabs(f) / sqrt(gg), mb);
        else U[id] = 1.0 / sqrt(inv);
        fixedv[id] = 1;
    }
}

extern "C" __global__ void sweep_plane(double* U, const unsigned char* act, const unsigned char* fixedv, int ni, int nj,
                                       int nk, double h, int si, int sj, int sk, int pl,
                                       unsigned long long* change) {
    // the nodes of plane ii + jj + kk = pl in sweep direction (si, sj, sk): (ii, jj) per thread
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nij = (long long)ni * nj;
    if (t >= nij) return;
    int ii = (int)(t / nj), jj = (int)(t % nj);
    int kk = pl - ii - jj;
    if (kk < 0 || kk >= nk) return;
    int i = si > 0 ? ii : ni - 1 - ii, j = sj > 0 ? jj : nj - 1 - jj, k = sk > 0 ? kk : nk - 1 - kk;
    long long sx = (long long)nj * nk, sy = nk;
    long long id = (long long)i * sx + (long long)j * sy + k;
    if (!act[id] || fixedv[id]) return;
    double a = BIG, b = BIG, c = BIG;
    if (i > 0 && act[id - sx]) a = U[id - sx];
    if (i < ni - 1 && act[id + sx] && U[id + sx] < a) a = U[id + sx];
    if (j > 0 && act[id - sy]) b = U[id - sy];
    if (j < nj - 1 && act[id + sy] && U[id + sy] < b) b = U[id + sy];
    if (k > 0 && act[id - 1]) c = U[id - 1];
    if (k < nk - 1 && act[id + 1] && U[id + 1] < c) c = U[id + 1];
    double tmp;
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (b > c) { tmp = b; b = c; c = tmp; }
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (a >= BIG) return;
    double u = a + h;
    if (u > b) {
        u = 0.5 * (a + b + sqrt(fmax(2.0 * h * h - (a - b) * (a - b), 0.0)));
        if (u > c) {
            double sm = a + b + c;
            double disc = sm * sm - 3.0 * (a * a + b * b + c * c - h * h);
            u = (sm + sqrt(fmax(disc, 0.0))) / 3.0;
        }
    }
    double cur = U[id];
    if (u < cur) {
        double ch = cur < BIG ? cur - u : h;
        atomicMax(change, (unsigned long long)__double_as_longlong(ch));   // positive doubles order as their bits
        U[id] = u;
    }
}

extern "C" __global__ void jacobi(const double* Uin, double* Uout, const long long* idx, long long n,
                                  const unsigned char* act, int ni, int nj, int nk, double h,
                                  unsigned long long* change) {
    // one Jacobi step of sweep_plane's update over the listed (active, unseeded) nodes: Uin -> Uout
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    long long id = idx[t];
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    long long sx = (long long)nj * nk, sy = nk;
    double a = BIG, b = BIG, c = BIG;
    if (i > 0 && act[id - sx]) a = Uin[id - sx];
    if (i < ni - 1 && act[id + sx] && Uin[id + sx] < a) a = Uin[id + sx];
    if (j > 0 && act[id - sy]) b = Uin[id - sy];
    if (j < nj - 1 && act[id + sy] && Uin[id + sy] < b) b = Uin[id + sy];
    if (k > 0 && act[id - 1]) c = Uin[id - 1];
    if (k < nk - 1 && act[id + 1] && Uin[id + 1] < c) c = Uin[id + 1];
    double tmp;
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (b > c) { tmp = b; b = c; c = tmp; }
    if (a > b) { tmp = a; a = b; b = tmp; }
    double cur = Uin[id];
    if (a >= BIG) { Uout[id] = cur; return; }
    double u = a + h;
    if (u > b) {
        u = 0.5 * (a + b + sqrt(fmax(2.0 * h * h - (a - b) * (a - b), 0.0)));
        if (u > c) {
            double sm = a + b + c;
            double disc = sm * sm - 3.0 * (a * a + b * b + c * c - h * h);
            u = (sm + sqrt(fmax(disc, 0.0))) / 3.0;
        }
    }
    if (u < cur) {
        double ch = cur < BIG ? cur - u : h;
        atomicMax(change, (unsigned long long)__double_as_longlong(ch));
        Uout[id] = u;
    } else Uout[id] = cur;
}

extern "C" __global__ void flag_seed_nbrs(const unsigned char* act, const unsigned char* fixedv, int ni, int nj, int nk,
                                          unsigned char* flag) {
    // the active, unseeded nodes beside a seed: the fast iterative method's first list
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)ni * nj * nk;
    if (id >= N) return;
    if (!act[id] || fixedv[id]) return;
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    long long sx = (long long)nj * nk, sy = nk;
    bool s = (i > 0 && act[id - sx] && fixedv[id - sx]) || (i < ni - 1 && act[id + sx] && fixedv[id + sx]) ||
             (j > 0 && act[id - sy] && fixedv[id - sy]) || (j < nj - 1 && act[id + sy] && fixedv[id + sy]) ||
             (k > 0 && act[id - 1] && fixedv[id - 1]) || (k < nk - 1 && act[id + 1] && fixedv[id + 1]);
    if (s) flag[id] = 1;
}

extern "C" __global__ void relax_active(double* U, const long long* idx, long long n, const unsigned char* act,
                                        const unsigned char* fixedv, int ni, int nj, int nk, double h, double eps,
                                        unsigned char* flag) {
    // the fast iterative method: the listed nodes updated in place (each by its own thread), the 6 neighbours of a
    // node that dropped by more than eps flagged for the next step
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    long long id = idx[t];
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    long long sx = (long long)nj * nk, sy = nk;
    double a = BIG, b = BIG, c = BIG;
    if (i > 0 && act[id - sx]) a = U[id - sx];
    if (i < ni - 1 && act[id + sx] && U[id + sx] < a) a = U[id + sx];
    if (j > 0 && act[id - sy]) b = U[id - sy];
    if (j < nj - 1 && act[id + sy] && U[id + sy] < b) b = U[id + sy];
    if (k > 0 && act[id - 1]) c = U[id - 1];
    if (k < nk - 1 && act[id + 1] && U[id + 1] < c) c = U[id + 1];
    double tmp;
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (b > c) { tmp = b; b = c; c = tmp; }
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (a >= BIG) return;
    double u = a + h;
    if (u > b) {
        u = 0.5 * (a + b + sqrt(fmax(2.0 * h * h - (a - b) * (a - b), 0.0)));
        if (u > c) {
            double sm = a + b + c;
            double disc = sm * sm - 3.0 * (a * a + b * b + c * c - h * h);
            u = (sm + sqrt(fmax(disc, 0.0))) / 3.0;
        }
    }
    double cur = U[id];
    if (u < cur) {
        U[id] = u;
        if (cur - u > eps || cur >= BIG) {
            if (i > 0 && act[id - sx] && !fixedv[id - sx]) flag[id - sx] = 1;
            if (i < ni - 1 && act[id + sx] && !fixedv[id + sx]) flag[id + sx] = 1;
            if (j > 0 && act[id - sy] && !fixedv[id - sy]) flag[id - sy] = 1;
            if (j < nj - 1 && act[id + sy] && !fixedv[id + sy]) flag[id + sy] = 1;
            if (k > 0 && act[id - 1] && !fixedv[id - 1]) flag[id - 1] = 1;
            if (k < nk - 1 && act[id + 1] && !fixedv[id + 1]) flag[id + 1] = 1;
        }
    }
}

__device__ __forceinline__ void push_nb(long long nid, const unsigned char* act, const unsigned char* fixedv,
                                        unsigned int* claim, long long* nxt, unsigned long long* cnt) {
    if (act[nid] && !fixedv[nid] && atomicExch(&claim[nid], 1u) == 0u) {
        unsigned long long p = atomicAdd(cnt, 1ull);
        nxt[p] = nid;
    }
}

extern "C" __global__ void relax_list(double* U, const long long* idx, long long n, const unsigned char* act,
                                      const unsigned char* fixedv, int ni, int nj, int nk, double h, double eps,
                                      unsigned int* claim, long long* nxt, unsigned long long* cnt) {
    // relax_active with the next list appended on the card (a claim per node keeps it once): no full-box scan
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    long long id = idx[t];
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    long long sx = (long long)nj * nk, sy = nk;
    double a = BIG, b = BIG, c = BIG;
    if (i > 0 && act[id - sx]) a = U[id - sx];
    if (i < ni - 1 && act[id + sx] && U[id + sx] < a) a = U[id + sx];
    if (j > 0 && act[id - sy]) b = U[id - sy];
    if (j < nj - 1 && act[id + sy] && U[id + sy] < b) b = U[id + sy];
    if (k > 0 && act[id - 1]) c = U[id - 1];
    if (k < nk - 1 && act[id + 1] && U[id + 1] < c) c = U[id + 1];
    double tmp;
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (b > c) { tmp = b; b = c; c = tmp; }
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (a >= BIG) return;
    double u = a + h;
    if (u > b) {
        u = 0.5 * (a + b + sqrt(fmax(2.0 * h * h - (a - b) * (a - b), 0.0)));
        if (u > c) {
            double sm = a + b + c;
            double disc = sm * sm - 3.0 * (a * a + b * b + c * c - h * h);
            u = (sm + sqrt(fmax(disc, 0.0))) / 3.0;
        }
    }
    double cur = U[id];
    if (u < cur) {
        U[id] = u;
        if (cur - u > eps || cur >= BIG) {
            if (i > 0) push_nb(id - sx, act, fixedv, claim, nxt, cnt);
            if (i < ni - 1) push_nb(id + sx, act, fixedv, claim, nxt, cnt);
            if (j > 0) push_nb(id - sy, act, fixedv, claim, nxt, cnt);
            if (j < nj - 1) push_nb(id + sy, act, fixedv, claim, nxt, cnt);
            if (k > 0) push_nb(id - 1, act, fixedv, claim, nxt, cnt);
            if (k < nk - 1) push_nb(id + 1, act, fixedv, claim, nxt, cnt);
        }
    }
}

extern "C" __global__ void relax_rb(double* U, const long long* idx, long long n, int parity, const unsigned char* act,
                                      const unsigned char* fixedv, int ni, int nj, int nk, double h, double eps,
                                      unsigned int* claim, long long* nxt, unsigned long long* cnt) {
    // 11-09: relax_list on the nodes of one parity ((i + j + k) & 1) only: a node's 6 neighbours all have the other
    // parity, so the in-place update reads values no thread of this launch writes: Gauss-Seidel order, deterministic
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    long long id = idx[t];
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    if (((i + j + k) & 1) != parity) return;
    long long sx = (long long)nj * nk, sy = nk;
    double a = BIG, b = BIG, c = BIG;
    if (i > 0 && act[id - sx]) a = U[id - sx];
    if (i < ni - 1 && act[id + sx] && U[id + sx] < a) a = U[id + sx];
    if (j > 0 && act[id - sy]) b = U[id - sy];
    if (j < nj - 1 && act[id + sy] && U[id + sy] < b) b = U[id + sy];
    if (k > 0 && act[id - 1]) c = U[id - 1];
    if (k < nk - 1 && act[id + 1] && U[id + 1] < c) c = U[id + 1];
    double tmp;
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (b > c) { tmp = b; b = c; c = tmp; }
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (a >= BIG) return;
    double u = a + h;
    if (u > b) {
        u = 0.5 * (a + b + sqrt(fmax(2.0 * h * h - (a - b) * (a - b), 0.0)));
        if (u > c) {
            double sm = a + b + c;
            double disc = sm * sm - 3.0 * (a * a + b * b + c * c - h * h);
            u = (sm + sqrt(fmax(disc, 0.0))) / 3.0;
        }
    }
    double cur = U[id];
    if (u < cur) {
        U[id] = u;
        if (cur - u > eps || cur >= BIG) {
            if (i > 0) push_nb(id - sx, act, fixedv, claim, nxt, cnt);
            if (i < ni - 1) push_nb(id + sx, act, fixedv, claim, nxt, cnt);
            if (j > 0) push_nb(id - sy, act, fixedv, claim, nxt, cnt);
            if (j < nj - 1) push_nb(id + sy, act, fixedv, claim, nxt, cnt);
            if (k > 0) push_nb(id - 1, act, fixedv, claim, nxt, cnt);
            if (k < nk - 1) push_nb(id + 1, act, fixedv, claim, nxt, cnt);
        }
    }
}

extern "C" __global__ void relax_compute(const double* U, const long long* idx, long long n, const unsigned char* act,
                                         const unsigned char* fixedv, int ni, int nj, int nk, double h, double eps,
                                         double* vals, unsigned int* claim, long long* nxt, unsigned long long* cnt) {
    // relax_list's update read from U as it was before this step (vals out, written by relax_write after every read):
    // Jacobi steps over the active list, independent of the threads' order, so a replay reproduces it bit for bit
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    long long id = idx[t];
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    long long sx = (long long)nj * nk, sy = nk;
    double cur = U[id];
    vals[t] = cur;
    double a = BIG, b = BIG, c = BIG;
    if (i > 0 && act[id - sx]) a = U[id - sx];
    if (i < ni - 1 && act[id + sx] && U[id + sx] < a) a = U[id + sx];
    if (j > 0 && act[id - sy]) b = U[id - sy];
    if (j < nj - 1 && act[id + sy] && U[id + sy] < b) b = U[id + sy];
    if (k > 0 && act[id - 1]) c = U[id - 1];
    if (k < nk - 1 && act[id + 1] && U[id + 1] < c) c = U[id + 1];
    double tmp;
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (b > c) { tmp = b; b = c; c = tmp; }
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (a >= BIG) return;
    double u = a + h;
    if (u > b) {
        u = 0.5 * (a + b + sqrt(fmax(2.0 * h * h - (a - b) * (a - b), 0.0)));
        if (u > c) {
            double sm = a + b + c;
            double disc = sm * sm - 3.0 * (a * a + b * b + c * c - h * h);
            u = (sm + sqrt(fmax(disc, 0.0))) / 3.0;
        }
    }
    if (u < cur) {
        vals[t] = u;
        if (cur - u > eps || cur >= BIG) {
            if (i > 0) push_nb(id - sx, act, fixedv, claim, nxt, cnt);
            if (i < ni - 1) push_nb(id + sx, act, fixedv, claim, nxt, cnt);
            if (j > 0) push_nb(id - sy, act, fixedv, claim, nxt, cnt);
            if (j < nj - 1) push_nb(id + sy, act, fixedv, claim, nxt, cnt);
            if (k > 0) push_nb(id - 1, act, fixedv, claim, nxt, cnt);
            if (k < nk - 1) push_nb(id + 1, act, fixedv, claim, nxt, cnt);
        }
    }
}

extern "C" __global__ void relax_write(const long long* idx, long long n, const double* vals, double* U) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) U[idx[t]] = vals[t];
}

extern "C" __global__ void unclaim(const long long* idx, long long n, unsigned int* claim) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) claim[idx[t]] = 0u;
}

extern "C" __global__ void ramp_init(const unsigned char* reg, int ni, int nj, int nk, int cap, int* d) {
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)ni * nj * nk;
    if (id >= N) return;
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    d[id] = (!reg[id] || i == 0 || j == 0 || k == 0 || i == ni - 1 || j == nj - 1 || k == nk - 1) ? 0 : cap;
}

extern "C" __global__ void ramp_step(const int* din, int ni, int nj, int nk, int* dout) {
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)ni * nj * nk;
    if (id >= N) return;
    int k = (int)(id % nk), j = (int)((id / nk) % nj), i = (int)(id / ((long long)nj * nk));
    long long sx = (long long)nj * nk, sy = nk;
    int v = din[id];
    if (v == 0) { dout[id] = 0; return; }
    int m = v;
    if (i > 0) m = min(m, din[id - sx] + 1);
    if (i < ni - 1) m = min(m, din[id + sx] + 1);
    if (j > 0) m = min(m, din[id - sy] + 1);
    if (j < nj - 1) m = min(m, din[id + sy] + 1);
    if (k > 0) m = min(m, din[id - 1] + 1);
    if (k < nk - 1) m = min(m, din[id + 1] + 1);
    dout[id] = m;
}

extern "C" __global__ void update(const double* F, const double* U, const int* d, double w, const long long* off,
                                  int n, int nj, int nk, const float* delta, const float* mask, float* out) {
    // brush.redistance's write, brick by brick: delta + beta (1 - clip(mask)) ok (D - F), non-finite -> unchanged
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)n * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int a = q / 64, bb = (q / 8) % 8, c = q % 8;
    long long nid = ((off[3 * b] + a) * nj + (off[3 * b + 1] + bb)) * nk + (off[3 * b + 2] + c);
    double u = U[nid], f = F[nid];
    bool ok = u < 1e29;
    double beta = fmin(1.0, (double)d[nid] / w);
    double m = fmin(fmax((double)mask[id], 0.0), 1.0);
    double wgt = beta * (1.0 - m) * (ok ? 1.0 : 0.0);
    double D = f < 0.0 ? -u : u;
    double nv = (double)delta[id] + wgt * (ok ? D - f : 0.0);
    out[id] = isfinite(nv) ? (float)nv : delta[id];
}
'''


class _K:
    mod = None
    prev = None      # 11-09: (oid, blo, (ni, nj, nk), F after the last pass on the card) for the release's next pass


def forget():
    """the release is over: the last pass's field leaves the card"""
    _K.prev = None


def _kernels(cp):
    if _K.mod is None:
        _K.mod = cp.RawModule(code=_SRC, options=('-std=c++14',))
    return _K.mod


_NB27 = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int64)


def slots_of(g, B):
    """11-09: the store slots of brick keys B (n, 3) from the grid's dense index in one gather (-1 where none);
    the per-brick dict lookups cost the release ~0.1 s on a large stroke"""
    B = np.asarray(B, np.int64).reshape(-1, 3)
    out = np.full(len(B), -1, np.int64)
    if g.index.size == 0 or not len(B):
        return out
    i = B - np.asarray(g.origin, np.int64)
    ok = np.all((i >= 0) & (i < np.array(g.index.shape, np.int64)), 1)
    out[ok] = g.index[i[ok, 0], i[ok, 1], i[ok, 2]]
    return out


_OFF = np.int64(1 << 20)
_MASK = np.int64((1 << 21) - 1)


def encode_keys(B):
    B = np.asarray(B, np.int64).reshape(-1, 3)
    return (B[:, 0] + _OFF) << 42 | (B[:, 1] + _OFF) << 21 | (B[:, 2] + _OFF)


def unique_keys(B):
    """brick keys sorted as sorted() sorts their tuples, without repeats"""
    B = np.asarray(B, np.int64).reshape(-1, 3)
    if not len(B):
        return B
    # one int64 per key (each coordinate offset into 21 bits): its numeric order is the tuples' order
    e = np.unique((B[:, 0] + _OFF) << 42 | (B[:, 1] + _OFF) << 21 | (B[:, 2] + _OFF))
    return np.stack([(e >> 42) - _OFF, ((e >> 21) & _MASK) - _OFF, (e & _MASK) - _OFF], 1)


def near_keys(g, B):
    """the bricks within one brick (Chebyshev) of B that hold a slot, sorted; and their slots"""
    N = unique_keys((np.asarray(B, np.int64).reshape(-1, 1, 3) + _NB27[None]).reshape(-1, 3))
    sl = slots_of(g, N)
    m = sl >= 0
    return N[m], sl[m]


def _mask_act(cp, F, Fn):
    """the pass was undone (no gain): F on the active nodes (Fn's finite nodes), 1e30 elsewhere"""
    return cp.where(Fn < 1e29, F, 1e30)


def _geval():
    """a card evaluator of its own (needs no viewport: a replay reaches it too)"""
    from .geval import GpuEval
    from .scene import _cp
    cp = _cp()
    if getattr(_K, 'ge', None) is None:
        _K.ge = GpuEval(cp)
    return cp, _K.ge


def _plane_sweep(cp, K, U, act, fixedv, ni, nj, nk, h, change, th, tm, t1):
    """the CPU's Gauss-Seidel fast sweep plane by plane (exactly its updates), one iteration captured as a CUDA graph"""
    sweep = K.get_function('sweep_plane')
    grid = ((ni * nj + th - 1) // th,)
    cp.cuda.runtime.deviceSynchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for s in range(8):
            si = 1 if (s & 1) == 0 else -1
            sj = 1 if (s & 2) == 0 else -1
            sk = 1 if (s & 4) == 0 else -1
            for pl in range(ni + nj + nk - 2):
                sweep(grid, (th,), (U, act, fixedv, np.int32(ni), np.int32(nj), np.int32(nk), np.float64(h),
                                    np.int32(si), np.int32(sj), np.int32(sk), np.int32(pl), change))
        graph = stream.end_capture()
    tm['graph_build'] = time.perf_counter() - t1
    iters = 0
    for it in range(4):
        change.fill(0)
        cp.cuda.runtime.deviceSynchronize()
        graph.launch(stream)
        stream.synchronize()
        iters += 1
        ch = float(np.frombuffer(change.get().tobytes(), np.float64)[0])
        if ch < 1e-3 * h:
            break
    del graph
    tm['sweep'] = time.perf_counter() - t1
    tm['_iters'] = iters
    return U


STATS = {}
JACOBI = __import__('os').environ.get('FS_RD_JACOBI', '1') == '1'   # the sweep as Jacobi steps over active nodes
FIM = __import__('os').environ.get('FS_RD_FIM', '1') == '1'         # the fast iterative method (active lists)
FIM_MAX = 2000
LIP_GPU = os.environ.get('FS_RD_LIP_GPU', '1') == '1'   # 11-09: the brick Lipschitz bounds on the card (lip_bricks)
RB = os.environ.get('FS_RD_SWEEP', 'rb') == 'rb'   # 11-09: red-black in-place steps ('jacobi': each step from the last)
N8 = os.environ.get('FS_RD_N8', '1') == '1'    # 11-09: the field at each brick's own 8^3 nodes (geval eval_nodes8)
FIM_EPS = 1e-6         # x h: a drop smaller than this flags no neighbour
JACOBI_MAX = 400
JACOBI_TOL = 1e-3      # x h: a Jacobi step that changes nothing by this ends it (the CPU's sweep stops at 1e-3 h)
FP32 = __import__('os').environ.get('FS_RD_FP32', '0') == '1'   # the field in single precision (geval fp32)
FP32_FAR = 0.005      # m: bricks holding a value at least this far out are evaluated in double (her store's bounds)


def redistance_gpu(scene, oid, bricks, rec, ramp=16.0, band=BR.BAND, src=None, force=False, reuse_prev=False,
                   lock_zero=False):
    tm = {}
    t0 = time.perf_counter()
    oi = scene.index_of(oid)
    obj = scene.objects[oi]
    g = obj.sculpt
    st = scene.store
    h = float(g.h)
    Ra = unique_keys(bricks)
    Rsl = slots_of(g, Ra)
    Ra = Ra[Rsl >= 0]
    if not len(Ra):
        return None
    if band > 0:
        R = BR.grow_band(scene, oid, [tuple(b) for b in Ra.tolist()], rec, band, src)
        Ra = unique_keys(R)
    R = [tuple(b) for b in Ra.tolist()]
    near_a, near_sl = near_keys(g, Ra)
    L_before = float(st.Lb[near_sl].max())
    BR.redistance.last_grad = (0.0, 0.0, False)
    # reuse_prev: the field a stroke's last pass left (a tuple, history keeps it per stroke), or True for _K.prev
    prev = reuse_prev if isinstance(reuse_prev, tuple) else (_K.prev if reuse_prev else None)
    _K.prev = None
    if L_before <= BR.SKIP_L and not force:
        BR.REDIST_STATS['skip_L'] = BR.REDIST_STATS.get('skip_L', 0) + 1
        return L_before, L_before
    cp, ge = _geval()
    K = _kernels(cp)
    blo = Ra.min(0) - 1
    bhi = Ra.max(0) + 1
    nb = tuple(int(v) for v in (bhi - blo + 1))
    role = np.zeros(nb, np.uint8)
    role[tuple((Ra - blo).T)] = 2
    halo = np.zeros_like(role)
    for d in BR._NB26:
        halo = np.maximum(halo, np.roll(role, d, (0, 1, 2)))
    role = np.where(role == 2, 2, np.where(halo > 0, 1, 0)).astype(np.uint8)
    ni, nj, nk = (int(v) * 8 for v in nb)
    N = ni * nj * nk
    d_role = cp.asarray(role)
    full = cp.repeat(cp.repeat(cp.repeat(d_role, 8, 0), 8, 1), 8, 2)
    reg = cp.ascontiguousarray((full == 2).astype(cp.uint8))
    act = cp.ascontiguousarray((full > 0).astype(cp.uint8))
    del full
    tm['setup'] = time.perf_counter() - t0
    tm['setup_roll_ms'] = None
    # F at the active bricks (the card's field program), 1e30 elsewhere
    t1 = time.perf_counter()
    actb = np.ascontiguousarray(np.argwhere(role > 0) + blo)   # brick keys (C order)
    th = 256
    F = None
    if prev is not None and prev[0] == oid and np.all(blo >= prev[1]) and np.all(blo + np.array(nb) <= prev[1] + np.array(prev[2]) // 8):
        # 11-09: a later pass of the same release: its bricks and their halo lie inside the last pass's active bricks
        # (history: the moving bricks' neighbours among the first pass's bricks), where the last pass left F + its
        # delta change on the card (the field is the base program + the delta at a node): no program evaluation
        o_ = (blo - prev[1]) * 8
        sub = prev[3].reshape(prev[2])[o_[0]:o_[0] + ni, o_[1]:o_[1] + nj, o_[2]:o_[2] + nk]
        a3 = act.reshape(ni, nj, nk) > 0
        if not bool(cp.any(a3 & (sub >= 1e29))):
            F = cp.ascontiguousarray(cp.where(a3, sub, 1e30)).ravel()
            tm['field_reused'] = True
        del sub, a3
    if F is None:
        F = cp.full(N, 1e30, cp.float64)
    for c0 in (range(0, len(actb), 32768) if not tm.get('field_reused') else ()):
        kb = actb[c0:c0 + 32768]
        _te = time.perf_counter()
        vals = ge.eval(scene, obj, (kb * 8 * h).astype(np.float64), kb, fp32=FP32, n8=N8)
        if FP32:
            # single precision matched the CPU to 0.02 um p99 everywhere within 2 mm of her surface, but at her
            # store's far bounds (6-20 mm out) a float32 decision gave values up to 4 um off: a brick holding any
            # value >= FP32_FAR is evaluated again in double
            far = cp.nonzero(cp.abs(vals).max(1) >= FP32_FAR)[0]
            if len(far):
                fh = far.get()
                vals[far] = ge.eval(scene, obj, (kb[fh] * 8 * h).astype(np.float64), kb[fh], n8=N8)
                tm['field_far_bricks'] = tm.get('field_far_bricks', 0) + len(fh)
        _th = time.perf_counter()
        cp.cuda.runtime.deviceSynchronize()      # the evaluator's launch is complete before the scatter reads it
        tm['field_host'] = tm.get('field_host', 0.0) + (_th - _te)
        tm['field_dev'] = tm.get('field_dev', 0.0) + (time.perf_counter() - _th)
        off = cp.asarray(np.ascontiguousarray((kb - blo) * 8, np.int64))   # argwhere is column-major: C order for the kernel
        K.get_function('scatter8' if N8 else 'scatter_bricks')(((len(kb) * 512 + th - 1) // th,), (th,),
                                         (vals, off, np.int32(len(kb)), np.int32(nj), np.int32(nk), F))
        import os as _os
        if _os.environ.get('FS_RD_DEBUG'):
            print('RDDBG chunk', c0, len(kb), 'vals', vals.shape, vals.dtype, 'finite<1e29 in vals', int(cp.count_nonzero(vals < 1e29)),
                  'F filled', int(cp.count_nonzero(F < 1e29)), 'off range', off.min().item(), off.max().item(), (ni, nj, nk), flush=True)
        del vals
    tm['field'] = time.perf_counter() - t1
    import os as _os
    if _os.environ.get('FS_RD_DEBUG'):
        _K.dbg = {'F': F.get().reshape(ni, nj, nk), 'blo': blo.copy(), 'act': act.get()}
    # the near-unit-gradient test (brush._near_unit_gradient)
    t1 = time.perf_counter()
    gsel = cp.empty(N, cp.float64)
    K.get_function('grad_sel')(((N + th - 1) // th,), (th,), (F, reg, np.int32(ni), np.int32(nj), np.int32(nk),
                                                             np.float64(h), np.float64(BR.SKIP_BAND * h), gsel))
    gv = gsel[gsel >= 0.0]
    if len(gv):
        gp1, gp99 = (float(v) for v in cp.percentile(gv, cp.asarray([1.0, 99.0])).get())
        ok_g = BR.SKIP_G_LO <= gp1 and gp99 <= BR.SKIP_G_HI
    else:
        gp1 = gp99 = 0.0
        ok_g = False
    del gsel, gv
    BR.redistance.last_grad = (gp1, gp99, ok_g)
    tm['grad'] = time.perf_counter() - t1
    if ok_g:
        BR.REDIST_STATS['unit_grad'] = BR.REDIST_STATS.get('unit_grad', 0) + 1
        _K.prev = (oid, blo.copy(), (ni, nj, nk), F)
        STATS.update(tm)
        return L_before, L_before
    # seeds + the plane-ordered fast sweep
    t1 = time.perf_counter()
    U = cp.empty(N, cp.float64)
    fixedv = cp.empty(N, cp.uint8)
    K.get_function('seeds')(((N + th - 1) // th,), (th,), (F, act, np.int32(ni), np.int32(nj), np.int32(nk),
                                                          np.float64(h), U, fixedv))
    change = cp.zeros(1, cp.uint64)
    sweep = K.get_function('sweep_plane')
    nij = ni * nj
    grid = ((nij + th - 1) // th,)
    iters = 0
    if FIM:
        # the fast iterative method: from the nodes beside the seeds, each step relaxes only the nodes whose neighbour
        # dropped (in place); the same upwind update to the same fixed point
        flag = cp.zeros(N, cp.uint8)
        nb_seed = K.get_function('flag_seed_nbrs')
        nb_seed(((N + th - 1) // th,), (th,), (act, fixedv, np.int32(ni), np.int32(nj), np.int32(nk), flag))
        idx = cp.nonzero(flag)[0].astype(cp.int64)          # the first list (one scan)
        del flag
        nact = int(cp.count_nonzero(act))
        claim = cp.zeros(N, cp.uint32)
        nxt = cp.empty(max(nact, 1), cp.int64)
        cnt = cp.zeros(1, cp.uint64)
        relax = K.get_function('relax_compute')
        wr = K.get_function('relax_write')
        rbk = K.get_function('relax_rb')
        uncl = K.get_function('unclaim')
        vals = cp.empty(max(nact, 1), cp.float64) if not RB else None
        n_ = int(len(idx))
        while n_ and iters < FIM_MAX:
            cnt.fill(0)
            if RB:
                # red-black: the list's even nodes in place, then its odd nodes (reading the even ones' new values)
                for par in (0, 1):
                    rbk(((n_ + th - 1) // th,), (th,), (U, idx, np.int64(n_), np.int32(par), act, fixedv, np.int32(ni),
                                                         np.int32(nj), np.int32(nk), np.float64(h),
                                                         np.float64(FIM_EPS * h), claim, nxt, cnt))
            else:
                relax(((n_ + th - 1) // th,), (th,), (U, idx, np.int64(n_), act, fixedv, np.int32(ni), np.int32(nj),
                                                       np.int32(nk), np.float64(h), np.float64(FIM_EPS * h), vals, claim,
                                                       nxt, cnt))
                wr(((n_ + th - 1) // th,), (th,), (idx, np.int64(n_), vals, U))
            m_ = int(cnt.get()[0])
            iters += 1
            if m_:
                uncl(((m_ + th - 1) // th,), (th,), (nxt, np.int64(m_), claim))
            idx, nxt = nxt[:m_].copy(), nxt
            n_ = m_
        del claim, nxt, idx, vals
        tm['sweep'] = time.perf_counter() - t1
    elif JACOBI:
        # Jacobi steps over the compacted active, unseeded nodes (the cost follows those nodes, not the box): the same
        # upwind update to its fixed point, the stop where a step changes nothing by JACOBI_TOL h
        idx = cp.nonzero((act.ravel() > 0) & (fixedv == 0))[0].astype(cp.int64)
        U2 = U.copy()
        jac = K.get_function('jacobi')
        nidx = int(len(idx))
        gj = ((nidx + th - 1) // th,)
        for it in range(JACOBI_MAX):
            change.fill(0)
            jac(gj, (th,), (U, U2, idx, np.int64(nidx), act, np.int32(ni), np.int32(nj), np.int32(nk), np.float64(h), change))
            U, U2 = U2, U
            iters += 1
            if it % 4 == 3:
                ch = float(np.frombuffer(change.get().tobytes(), np.float64)[0])
                if ch < JACOBI_TOL * h:
                    break
        del U2, idx
        tm['sweep'] = time.perf_counter() - t1
    else:
        U = _plane_sweep(cp, K, U, act, fixedv, ni, nj, nk, h, change, th, tm, t1)
        iters = tm.pop('_iters')
    # the edge ramp
    t1 = time.perf_counter()
    cap = int(math.ceil(ramp))
    da = cp.empty(N, cp.int32)
    db = cp.empty(N, cp.int32)
    K.get_function('ramp_init')(((N + th - 1) // th,), (th,), (reg, np.int32(ni), np.int32(nj), np.int32(nk),
                                                              np.int32(cap), da))
    step = K.get_function('ramp_step')
    for _ in range(cap):
        step(((N + th - 1) // th,), (th,), (da, np.int32(ni), np.int32(nj), np.int32(nk), db))
        da, db = db, da
    del db
    tm['ramp'] = time.perf_counter() - t1
    # the write (brick by brick, as brush.redistance)
    t1 = time.perf_counter()
    slots = slots_of(g, Ra)
    D0 = st.delta[slots]                 # the region's state before (one gather; the snapshots below are its rows)
    M0 = st.mask[slots]
    Lb0 = st.Lb[slots].astype(np.float64)
    Lold = st.Lb[near_sl].astype(np.float64)
    d_delta = cp.asarray(np.ascontiguousarray(D0, np.float32).reshape(-1))
    d_mask = cp.asarray(np.ascontiguousarray(M0, np.float32).reshape(-1))
    out = cp.empty_like(d_delta)
    off = cp.asarray(np.ascontiguousarray((Ra - blo) * 8, np.int64))
    K.get_function('update')(((len(R) * 512 + th - 1) // th,), (th,),
                             (F, U, da, np.float64(ramp), off, np.int32(len(R)), np.int32(nj), np.int32(nk), d_delta,
                              d_mask, out))
    if lock_zero:
        # 11-09 (I2b, a release paying older strokes' debt): the zero set kept bitwise - every corner of every
        # cell whose 8 nodes change sign (F before this pass) keeps its delta; only band distances change
        Fg = F.reshape(ni, nj, nk)
        mn = mx = None
        for a_ in (0, 1):
            for b_ in (0, 1):
                for c_ in (0, 1):
                    v_ = Fg[a_:a_ + ni - 1, b_:b_ + nj - 1, c_:c_ + nk - 1]
                    mn = v_.copy() if mn is None else cp.minimum(mn, v_)
                    mx = v_.copy() if mx is None else cp.maximum(mx, v_)
        cc_ = (mn < 0) & (mx >= 0) & (mx < 1e29)
        del mn, mx
        lockg = cp.zeros((ni, nj, nk), cp.bool_)
        for a_ in (0, 1):
            for b_ in (0, 1):
                for c_ in (0, 1):
                    lockg[a_:a_ + ni - 1, b_:b_ + nj - 1, c_:c_ + nk - 1] |= cc_
        del cc_
        q = cp.arange(512)
        qa, qb, qc = q // 64, (q // 8) % 8, q % 8
        offc = cp.asarray(np.ascontiguousarray((Ra - blo) * 8, np.int64))
        lk = lockg[offc[:, 0, None] + qa[None], offc[:, 1, None] + qb[None], offc[:, 2, None] + qc[None]].reshape(-1)
        out = cp.where(lk, d_delta, out)
        tm['locked_nodes'] = int(lk.sum())
        del lockg, lk
    newd = out.get().reshape(len(R), 8, 8, 8)
    st.delta[slots] = newd
    tm['write'] = time.perf_counter() - t1
    # the brick Lipschitz bounds (as brush.redistance)
    t1 = time.perf_counter()
    arr = np.ascontiguousarray(near_a)
    if LIP_GPU:
        # the delta rows the near bricks' cells read (the near bricks and one brick around them), after the write
        tk, tsl = near_keys(g, near_a)
        tlo = tk.min(0)
        tn = tk.max(0) - tlo + 1
        Th = np.full(tuple(int(v) for v in tn), -1, np.int32)
        Th[tuple((tk - tlo).T)] = np.arange(len(tk), dtype=np.int32)
        d_T = cp.asarray(Th)
        d_rows = cp.asarray(np.ascontiguousarray(st.delta[tsl], np.float32))
        d_br = cp.asarray(arr)
        d_out = cp.empty(len(arr), cp.float64)
        lipk = K.get_function('lip_bricks')
        targs = (np.int32(tlo[0]), np.int32(tlo[1]), np.int32(tlo[2]), np.int32(tn[0]), np.int32(tn[1]), np.int32(tn[2]))
        lipk((len(arr),), (256,), (d_br, np.int32(len(arr)), d_T, d_rows) + targs +
             (np.float64(h), np.intp(0), np.int32(0), np.int32(0), np.int32(0), np.int32(0), np.int32(0), np.int32(0),
              np.float64(0.0), d_out))
        outL = d_out.get()
    else:
        pk = scene.pack()
        outL = np.zeros(len(near_a))
        BR.brick_lipschitz(pk.index_flat, pk.delta, pk.oi_tab, oi, h, arr, outL)
    Lbase = float(scene.objects[oi].compiled()[4])
    thr = (Lbase + float(outL.max())) * math.sqrt(3.0) * h
    tm['lip_a'] = time.perf_counter() - t1
    # Fn: F where act, + the delta's change on R's bricks; 1e30 elsewhere
    dch = cp.asarray(newd.reshape(-1).astype(np.float64)) - d_delta.astype(cp.float64)
    Fn = cp.where(act.ravel() > 0, F, 1e30)
    K2 = cp.ElementwiseKernel('raw float64 dch, raw int64 off, int32 nj, int32 nk', 'raw float64 Fn',
                              '''long long b = i / 512; int q = i % 512; int a = q / 64, bb = (q / 8) % 8, c = q % 8;
                                 long long nid = ((off[3 * b] + a) * nj + (off[3 * b + 1] + bb)) * nk + (off[3 * b + 2] + c);
                                 if (Fn[nid] < 1e29) Fn[nid] += dch[i];''', 'fs_redist_fn')
    K2(dch, off, np.int32(nj), np.int32(nk), Fn, size=len(R) * 512)
    i0, j0, k0 = (int(v) for v in blo * 8)
    if LIP_GPU:
        lipk((len(arr),), (256,), (d_br, np.int32(len(arr)), d_T, d_rows) + targs +
             (np.float64(h), Fn, np.int32(i0), np.int32(j0), np.int32(k0), np.int32(ni), np.int32(nj), np.int32(nk),
              np.float64(thr), d_out))
        outL = d_out.get()
        del d_T, d_rows, d_br, d_out
    else:
        Fn_h = Fn.get().reshape(ni, nj, nk)
        BR.brick_lipschitz_out(pk.index_flat, pk.delta, pk.oi_tab, oi, h, arr, np.ascontiguousarray(Fn_h), i0, j0, k0,
                               thr, outL)
    tm['lip_fn'] = time.perf_counter() - t1 - tm['lip_a']
    L_after = float(outL.max())
    tm['lipschitz'] = time.perf_counter() - t1
    del U, fixedv, da, act, reg, d_delta, d_mask, out, dch
    if L_after >= L_before and not force:
        BR.REDIST_STATS['no_gain'] = BR.REDIST_STATS.get('no_gain', 0) + 1
        st.delta[slots] = D0
        _K.prev = (oid, blo.copy(), (ni, nj, nk), _mask_act(cp, F, Fn))
        del F, Fn
        STATS.update(tm)
        return L_before, L_before
    _K.prev = (oid, blo.copy(), (ni, nj, nk), Fn)       # F + this pass's delta change, 1e30 off the active bricks
    del F, Fn
    BR.REDIST_STATS['kept'] = BR.REDIST_STATS.get('kept', 0) + 1
    st.Lb[near_sl] = outL
    key = obj.id
    Rset = set(R)
    for i_, b in enumerate(R):
        if (key, b) not in rec.snap and (key, b) not in rec.new_set:
            rec.snap[(key, b)] = (D0[i_].copy(), M0[i_].copy(), float(Lb0[i_]))
    for b, L_ in zip(map(tuple, near_a.tolist()), Lold.tolist()):
        if b not in Rset and (key, b) not in rec.Lsnap and (key, b) not in rec.snap and (key, b) not in rec.new_set:
            rec.Lsnap[(key, b)] = L_
    scene.refresh_object(oi)
    tm['total'] = time.perf_counter() - t0
    tm['dims'] = (ni, nj, nk)
    tm['iters'] = iters
    STATS.clear()
    STATS.update(tm)
    return L_before, L_after
