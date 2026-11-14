"""Hair Studio on the GPU (CuPy, Apache-2.0): the settle's hot parts on the graphics card.

Why (10-31): the port came first after that pass, since a 10-100x speed gain on the GPU saves 10-100 hours
of runs in a single week. On the full grown head (514 guides x 118 segments, 105k
samples) one settle evaluation took 0.235 s on the CPU, three quarters of it (0.17 s) in the hair-hair pressure and
cohesion field (settle.SplatC1: every sample splatted onto a 3-D lattice, blurred, read back as forces) - the work a
GPU is built for. The RTX 5050 sat idle (it only rendered).

    SplatC1GPU    settle.SplatC1 with field / cohesive / pressure / adjoint / gather / alignment on the GPU; the same
                  interface, NumPy in and out, so the settle's code is unchanged. Its kernel, blur and free volume are
                  the CPU class's own (the free-volume table a_v h^3 is computed on the CPU per crop and kept on the
                  card while the crop holds). cohesive_g / pressure_g: the same with CuPy in and out.
    CoreGPU       the whole settle.State.core on the card (bending, gravity, body contact, the field, friction, the
                  sample forces to the directions, the conditioning; the acts through a NumPy view), and fun(): the
                  solver's objective with the preconditioner's transforms as kernels, so only x and the gradient
                  cross. settle.State uses it when HAIR_GPU=1, the contact is C1 and the splat is SplatC1GPU.
    sdf_sample / smooth_distance   the body's distance (grid.SDFGrid.sample, settle.smooth_distance) on the card
    enabled()     HAIR_GPU=1 in the environment, and CuPy can reach a CUDA device

Measured on the full grown head (10-31): the core 0.248 -> 0.028 s per evaluation; a 150-evaluation L-BFGS solve
0.388 -> 0.074 s per evaluation; energies agree with the CPU to 1e-15, gradients to rounding (1e-13).

The splat and the gathers are hand-written kernels, one thread per sample: each computes its own quadratic B-spline
weights and adds into / reads its 27 lattice nodes directly (the first port built them from array operations with
(27, n) temporaries: ~55 ms per evaluation, the card's slow float64 arithmetic spent moving memory). The blur stays
three array passes. The crop has slack (SLACK cells a side) and is kept while the samples stay in it, so a_v is rebuilt
only when the hair leaves it.

Same math, not the same bits: the splat's scatter-add and the sample forces' bincount run as atomic adds, so sums come
in another order; energies and gradients agree with the CPU to rounding, not bit for bit, and an L-BFGS solve takes a
path that differs by rounding (the same energy to ~1e-5 after 150 evaluations). The CPU stays the reference and the
default.
Needs no cuBLAS or cuRAND (this install has neither)."""
import os

import numpy as np

from . import settle as SE

try:                                          # optional: the engine runs without a GPU
    import cupy as cp
except Exception:                             # noqa: BLE001
    cp = None


def enabled():
    """the GPU path is on: HAIR_GPU=1 and a CUDA device CuPy can use"""
    if os.environ.get("HAIR_GPU", "0") not in ("1", "true", "yes") or cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:                         # noqa: BLE001
        return False


_SRC = r"""
__device__ __forceinline__ void wts(double u, double* w, double* d) {
    w[0] = 0.5 * (0.5 - u) * (0.5 - u); w[1] = 0.75 - u * u; w[2] = 0.5 * (0.5 + u) * (0.5 + u);
    d[0] = -(0.5 - u); d[1] = -2.0 * u; d[2] = 0.5 + u;
}

// the splat: each sample's node offsets u (n, 3) and base index B (n), and v (w_x w_y w_z) added into S
extern "C" __global__ void splat(const double* Y, const double* v, const int n, const double lox, const double loy,
                                 const double loz, const double h, const long long c0x, const long long c0y,
                                 const long long c0z, const long long sx, const long long sy, double* S, double* U,
                                 long long* B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double fx = (Y[3 * i] - lox) / h, fy = (Y[3 * i + 1] - loy) / h, fz = (Y[3 * i + 2] - loz) / h;
    double ix = rint(fx), iy = rint(fy), iz = rint(fz);
    double ux = fx - ix, uy = fy - iy, uz = fz - iz;
    U[3 * i] = ux; U[3 * i + 1] = uy; U[3 * i + 2] = uz;
    long long b = ((long long)ix - c0x) * sx + ((long long)iy - c0y) * sy + ((long long)iz - c0z);
    B[i] = b;
    double wx[3], wy[3], wz[3], d[3];
    wts(ux, wx, d); wts(uy, wy, d); wts(uz, wz, d);
    double vi = v[i];
    for (int a = 0; a < 3; a++) for (int e = 0; e < 3; e++) {
        double wxy = wx[a] * wy[e] * vi;
        long long r = b + (a - 1) * sx + (e - 1) * sy;
        for (int c = 0; c < 3; c++) atomicAdd(&S[r + (c - 1)], wxy * wz[c]);
    }
}

// the blur, one axis per launch: out = 2 src + the lower neighbour + the upper one (zero outside), in the order of the
// array passes (src * 2; out[1:] += src[:-1]; out[:-1] += src[1:]), times scale on the last axis (the deferred 1/64):
// the same bits as SplatC1._blur
extern "C" __global__ void blur_ax(const double* src, double* out, const long long n0, const long long n1,
                                   const long long n2, const int ax, const double scale, const int last) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long tot = n0 * n1 * n2;
    if (i >= tot) return;
    long long c, n, s;
    if (ax == 0) { c = i / (n1 * n2); n = n0; s = n1 * n2; }
    else if (ax == 1) { c = (i / n2) % n1; n = n1; s = n2; }
    else { c = i % n2; n = n2; s = 1; }
    double v = src[i] * 2.0;
    if (c > 0) v += src[i - s];
    if (c < n - 1) v += src[i + s];
    if (last) v *= scale;
    out[i] = v;
}

// the field's pointwise work for the cohesive energy (SplatC1GPU.cohesive_g, no alignment): ph = F / av, the energy
// densities e1 = av ph^2 ph, e2 = av pw pw and the adjoint inputs G1 = 3c ph^2, G2 = -2ca pw, in the array form's order
extern "C" __global__ void coh_pointwise(const double* F, const double* Sw, const double* av, const double c3,
                                         const double ca2, const long long tot, double* e1, double* e2, double* G1,
                                         double* G2) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tot) return;
    double a = av[i];
    double ph = F[i] / a;
    double p2 = ph * ph;
    double pw = Sw[i] / a;
    e1[i] = (a * p2) * ph;
    e2[i] = (a * pw) * pw;
    G1[i] = c3 * p2;
    G2[i] = ca2 * pw;
}

// the same for the pressure alone
extern "C" __global__ void prs_pointwise(const double* F, const double* av, const double c3, const long long tot,
                                         double* e1, double* G1) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tot) return;
    double a = av[i];
    double ph = F[i] / a;
    double p2 = ph * ph;
    e1[i] = (a * p2) * ph;
    G1[i] = c3 * p2;
}

// a second field on the same samples: weights vw instead of v
extern "C" __global__ void resplat(const double* U, const long long* B, const double* vw, const int n,
                                   const long long sx, const long long sy, double* S) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double wx[3], wy[3], wz[3], d[3];
    wts(U[3 * i], wx, d); wts(U[3 * i + 1], wy, d); wts(U[3 * i + 2], wz, d);
    double vi = vw[i];
    long long b = B[i];
    for (int a = 0; a < 3; a++) for (int e = 0; e < 3; e++) {
        double wxy = wx[a] * wy[e] * vi;
        long long r = b + (a - 1) * sx + (e - 1) * sy;
        for (int c = 0; c < 3; c++) atomicAdd(&S[r + (c - 1)], wxy * wz[c]);
    }
}

// a field at the samples: sum over the 27 nodes of A w_x w_y w_z
extern "C" __global__ void gather(const double* A, const double* U, const long long* B, const int n,
                                  const long long sx, const long long sy, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double wx[3], wy[3], wz[3], d[3];
    wts(U[3 * i], wx, d); wts(U[3 * i + 1], wy, d); wts(U[3 * i + 2], wz, d);
    long long b = B[i];
    double s = 0.0;
    for (int a = 0; a < 3; a++) for (int e = 0; e < 3; e++) {
        long long r = b + (a - 1) * sx + (e - 1) * sy;
        for (int c = 0; c < 3; c++) s += A[r + (c - 1)] * wx[a] * wy[e] * wz[c];
    }
    out[i] = s;
}

// the body's distance: trilinear value at P inside the lattice (grid._trilinear: floor, cells clipped to the box)
extern "C" __global__ void tri_sdf(const double* A, const long long nx, const long long ny, const long long nz,
                                   const double lox, const double loy, const double loz, const double h,
                                   const double* P, const int n, double* v) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double f[3] = {(P[3 * i] - lox) / h, (P[3 * i + 1] - loy) / h, (P[3 * i + 2] - loz) / h};
    long long sh[3] = {nx, ny, nz}, i0[3];
    double t[3];
    for (int k = 0; k < 3; k++) {
        long long q = (long long)floor(f[k]);
        q = q < 0 ? 0 : (q > sh[k] - 2 ? sh[k] - 2 : q);
        i0[k] = q; t[k] = f[k] - (double)q;
    }
    long long sx = ny * nz, sy = nz;
    long long b = i0[0] * sx + i0[1] * sy + i0[2];
    double c000 = A[b], c001 = A[b + 1], c010 = A[b + sy], c011 = A[b + sy + 1];
    double c100 = A[b + sx], c101 = A[b + sx + 1], c110 = A[b + sx + sy], c111 = A[b + sx + sy + 1];
    double ux = 1 - t[0], uy = 1 - t[1], uz = 1 - t[2];
    double c00 = c000 * ux + c100 * t[0], c01 = c001 * ux + c101 * t[0];
    double c10 = c010 * ux + c110 * t[0], c11 = c011 * ux + c111 * t[0];
    double c0 = c00 * uy + c10 * t[1], c1 = c01 * uy + c11 * t[1];
    v[i] = c0 * uz + c1 * t[2];
}

// the body's distance, C1: quadratic B-spline value and gradient at P (settle._bspline2: nodes clipped to the box)
extern "C" __global__ void bsp_sdf(const double* A, const long long nx, const long long ny, const long long nz,
                                   const double lox, const double loy, const double loz, const double h,
                                   const double* P, const int n, double* v, double* g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double f[3] = {(P[3 * i] - lox) / h, (P[3 * i + 1] - loy) / h, (P[3 * i + 2] - loz) / h};
    long long sh[3] = {nx, ny, nz};
    double w[3][3], d[3][3];
    long long ix[3][3];
    for (int k = 0; k < 3; k++) {
        double r = rint(f[k]);
        double u = f[k] - r;
        wts(u, w[k], d[k]);
        for (int a = 0; a < 3; a++) {
            long long q = (long long)r + a - 1;
            ix[k][a] = q < 0 ? 0 : (q > sh[k] - 1 ? sh[k] - 1 : q);
        }
    }
    long long sx = ny * nz, sy = nz;
    double vv = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        long long base = ix[0][a] * sx + ix[1][b] * sy;
        double wab = w[0][a] * w[1][b], dxab = d[0][a] * w[1][b], dyab = w[0][a] * d[1][b];
        for (int c = 0; c < 3; c++) {
            double val = A[base + ix[2][c]];
            vv += val * wab * w[2][c];
            gx += val * dxab * w[2][c];
            gy += val * dyab * w[2][c];
            gz += val * wab * d[2][c];
        }
    }
    v[i] = vv; g[3 * i] = gx / h; g[3 * i + 1] = gy / h; g[3 * i + 2] = gz / h;
}

// settle.bend_carried, one thread per guide: the frame carried along the strand, the energy and its gradient through
// the transport. Scratch T, N, Tb (G, K, 3), C, A (G, K)
extern "C" __global__ void bend_carried(const double* D, const double* d0, const double* n0, const double* loc,
                                        const double* cb, const int G, const int K, double* T, double* N, double* C,
                                        double* Aa, double* Tb, double* gD, double* Eg) {
    int gi = blockDim.x * blockIdx.x + threadIdx.x;
    if (gi >= G) return;
    long long o = (long long)gi * K * 3, o1 = (long long)gi * K;
    for (int c = 0; c < 3; c++) { T[o + c] = d0[3 * gi + c]; N[o + c] = n0[3 * gi + c]; }
    C[o1] = 1.0; Aa[o1] = 0.0;
    double n[3] = {n0[3 * gi], n0[3 * gi + 1], n0[3 * gi + 2]};
    for (int j = 1; j < K; j++) {
        for (int c = 0; c < 3; c++) T[o + 3 * j + c] = D[o + 3 * (j - 1) + c];
        const double* ta = &T[o + 3 * (j - 1)];
        const double* tb = &T[o + 3 * j];
        double cc = 1.0 + (ta[0] * tb[0] + ta[1] * tb[1] + ta[2] * tb[2]);
        cc = cc > 1e-9 ? cc : 1e-9;
        double a = (n[0] * tb[0] + n[1] * tb[1] + n[2] * tb[2]) / cc;
        for (int c = 0; c < 3; c++) { n[c] = n[c] - a * (ta[c] + tb[c]); N[o + 3 * j + c] = n[c]; }
        C[o1 + j] = cc; Aa[o1 + j] = a;
    }
    double E = 0.0;
    for (int j = 0; j < K; j++) {
        const double* t = &T[o + 3 * j];
        const double* nn = &N[o + 3 * j];
        double ct = loc[o + 3 * j], cn = loc[o + 3 * j + 1], cbn = loc[o + 3 * j + 2];
        double txn[3] = {t[1] * nn[2] - t[2] * nn[1], t[2] * nn[0] - t[0] * nn[2], t[0] * nn[1] - t[1] * nn[0]};
        double gv[3];
        double w = cb[o1 + j];
        for (int c = 0; c < 3; c++) {
            double rho = ct * t[c] + cn * nn[c] + cbn * txn[c];
            double res = D[o + 3 * j + c] - t[c] - rho;
            gv[c] = w * res;
            E += gv[c] * res;
            gD[o + 3 * j + c] = gv[c];
        }
        double nxg[3] = {nn[1] * gv[2] - nn[2] * gv[1], nn[2] * gv[0] - nn[0] * gv[2], nn[0] * gv[1] - nn[1] * gv[0]};
        for (int c = 0; c < 3; c++) Tb[o + 3 * j + c] = -(1.0 + ct) * gv[c] - cbn * nxg[c];
    }
    double carry[3] = {0.0, 0.0, 0.0};
    for (int j = K - 1; j > 0; j--) {
        const double* t = &T[o + 3 * j];
        const double* ta = &T[o + 3 * (j - 1)];
        const double* nn = &N[o + 3 * j];
        double gv[3] = {gD[o + 3 * j], gD[o + 3 * j + 1], gD[o + 3 * j + 2]};
        double ct = loc[o + 3 * j], cn = loc[o + 3 * j + 1], cbn = loc[o + 3 * j + 2];
        (void)ct; (void)nn;
        double gxt[3] = {gv[1] * t[2] - gv[2] * t[1], gv[2] * t[0] - gv[0] * t[2], gv[0] * t[1] - gv[1] * t[0]};
        double nb[3];
        for (int c = 0; c < 3; c++) nb[c] = (-cn * gv[c] - cbn * gxt[c]) + carry[c];
        double cc = C[o1 + j], a = Aa[o1 + j];
        double bc = (nb[0] * (ta[0] + t[0]) + nb[1] * (ta[1] + t[1]) + nb[2] * (ta[2] + t[2])) / cc;
        const double* nprev = &N[o + 3 * (j - 1)];
        for (int c = 0; c < 3; c++) {
            carry[c] = nb[c] - bc * t[c];
            Tb[o + 3 * j + c] += -a * nb[c] - bc * nprev[c] + (bc * a) * ta[c];
            Tb[o + 3 * (j - 1) + c] += -a * nb[c] + (bc * a) * t[c];
        }
    }
    for (int j = 0; j < K - 1; j++)
        for (int c = 0; c < 3; c++) gD[o + 3 * j + c] += Tb[o + 3 * (j + 1) + c];
    Eg[gi] = 0.5 * E;
}

// the solver's variables to the segment vectors (settle.State._U with a factor): L^T y = x by back substitution per
// guide (settle.bsolve_t), then u_k = (y_k - y_{k-1}) / lmap_k. Di, S1, S2 (G, K, 3, 3); x, y, U (G, K, 3)
__device__ __forceinline__ void mtv(const double* M, const double* v, double* r) {
    for (int c = 0; c < 3; c++) r[c] = (M[c] * v[0] + M[3 + c] * v[1]) + M[6 + c] * v[2];
}
__device__ __forceinline__ void mv(const double* M, const double* v, double* r) {
    for (int c = 0; c < 3; c++) r[c] = (M[3 * c] * v[0] + M[3 * c + 1] * v[1]) + M[3 * c + 2] * v[2];
}
extern "C" __global__ void u_of_x(const double* x, const double* Di, const double* S1, const double* S2,
                                  const double* lmap, const int G, const int K, double* y, double* U) {
    int gi = blockDim.x * blockIdx.x + threadIdx.x;
    if (gi >= G) return;
    long long o = (long long)gi * K;
    for (int k = K - 1; k >= 0; k--) {
        long long b = o + k;
        double r[3] = {x[3 * b], x[3 * b + 1], x[3 * b + 2]}, t[3];
        if (k + 1 < K) { mtv(&S1[9 * b], &y[3 * (b + 1)], t); for (int c = 0; c < 3; c++) r[c] -= t[c]; }
        if (k + 2 < K) { mtv(&S2[9 * b], &y[3 * (b + 2)], t); for (int c = 0; c < 3; c++) r[c] -= t[c]; }
        mtv(&Di[9 * b], r, &y[3 * b]);
    }
    for (int k = 0; k < K; k++) {
        long long b = o + k;
        for (int c = 0; c < 3; c++) {
            double u = y[3 * b + c];
            if (k >= 1) u -= y[3 * (b - 1) + c];
            U[3 * b + c] = u / lmap[b];
        }
    }
}

// dE/dU to the solver's variables (settle.State._gx with a factor): g = gU / lmap, z_k = g_k - g_{k+1}, L w = z by
// forward substitution per guide (settle.bsolve), times esc
extern "C" __global__ void gx_of_gu(const double* gU, const double* Di, const double* S1, const double* S2,
                                    const double* lmap, const int G, const int K, double* w) {
    int gi = blockDim.x * blockIdx.x + threadIdx.x;
    if (gi >= G) return;
    long long o = (long long)gi * K;
    for (int k = 0; k < K; k++) {
        long long b = o + k;
        double r[3], t[3];
        for (int c = 0; c < 3; c++) {
            double z = gU[3 * b + c] / lmap[b];
            if (k + 1 < K) z -= gU[3 * (b + 1) + c] / lmap[b + 1];
            r[c] = z;
        }
        if (k >= 1) { mv(&S1[9 * (b - 1)], &w[3 * (b - 1)], t); for (int c = 0; c < 3; c++) r[c] -= t[c]; }
        if (k >= 2) { mv(&S2[9 * (b - 2)], &w[3 * (b - 2)], t); for (int c = 0; c < 3; c++) r[c] -= t[c]; }
        mv(&Di[9 * b], r, &w[3 * b]);
    }
}

// dE/dY from K blurred dE/dF fields H (K, ncell) folded with the per-sample weights VW (K, n)
extern "C" __global__ void gather_grad(const double* H, const int K, const long long ncell, const double* VW,
                                       const double* U, const long long* B, const int n, const long long sx,
                                       const long long sy, const double invh, double* G) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double wx[3], wy[3], wz[3], dx[3], dy[3], dz[3];
    wts(U[3 * i], wx, dx); wts(U[3 * i + 1], wy, dy); wts(U[3 * i + 2], wz, dz);
    long long b = B[i];
    double gx = 0.0, gy = 0.0, gz = 0.0;
    for (int a = 0; a < 3; a++) for (int e = 0; e < 3; e++) {
        long long r = b + (a - 1) * sx + (e - 1) * sy;
        for (int c = 0; c < 3; c++) {
            long long idx = r + (c - 1);
            double hn = 0.0;
            for (int k = 0; k < K; k++) hn += H[k * ncell + idx] * VW[(long long)k * n + i];
            gx += hn * dx[a] * wy[e] * wz[c];
            gy += hn * wx[a] * dy[e] * wz[c];
            gz += hn * wx[a] * wy[e] * dz[c];
        }
    }
    G[3 * i] = gx * invh; G[3 * i + 1] = gy * invh; G[3 * i + 2] = gz * invh;
}
"""
_MOD = None


def _k(name):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC)
    return _MOD.get_function(name)


def _launch(name, n, *args):
    if n:
        _k(name)(((n + 255) // 256,), (256,), args)


def _blur(B, out=None):
    """grid.blur121 on the card (SplatC1._blur's bits): [1, 2, 1] along each axis, one fused kernel per axis (the
    array form - a multiply and two shifted adds per axis - read and wrote the 36 MB crop eight times per axis: 3.4 ms
    a blur on the full head), the 1/4 factors folded into one 1/64 in the last pass. out: a contiguous buffer of B's
    size to blur into (a row of the adjoint's stack)"""
    B = cp.ascontiguousarray(B)
    n0, n1, n2 = B.shape
    tot = B.size
    t1 = cp.empty_like(B)
    t2 = cp.empty_like(B)
    res = cp.empty_like(B) if out is None else out
    for ax, (s, o) in enumerate(((B, t1), (t1, t2), (t2, res))):
        _launch("blur_ax", tot, s, o, np.int64(n0), np.int64(n1), np.int64(n2), np.int32(ax), np.float64(1.0 / 64.0),
                np.int32(ax == 2))
    return res


class SplatC1GPU(SE.SplatC1):
    """settle.SplatC1 with its lattice work on the GPU (see the module notes). NumPy in, NumPy out."""

    SLACK = 4                                  # crop slack, lattice cells per side
    SHRINK = 1.5                               # a kept crop is dropped once it holds this many times the cells needed
                                               # (a wild line-search trial would otherwise leave every later
                                               # evaluation working on its huge crop)
    on_gpu = True                              # settle.State runs its whole core on the card with this splat

    def __init__(self, grid=None, mm=None):
        super().__init__(grid=grid, mm=mm)
        self._g = None                         # the last field's samples on the card: U, B, v, n, sx, sy
        self._avg = None                       # (crop key, a_v h^3 on the card)
        self._crop = None                      # (c0, shape) of the crop with slack
        self._afull_g = None                   # the grid's free fraction a_v on the card (float64)

    def _avox_g(self):
        """a_v h^3 on the crop, built on the card: grid.Splat.free (a slice of the grid's a_v, 1 where the grid does
        not reach) times vox, as the CPU class's _avox - the same values; the CPU build cost ~24 ms a crop"""
        k = (tuple(int(c) for c in self.c0), self.shape)
        if self._avg is None or self._avg[0] != k:
            out = cp.ones(self.shape, cp.float64)
            af = self.base._afull
            if af is not None:
                if getattr(self, "_afull_g", None) is None:
                    self._afull_g = cp.asarray(np.asarray(af, np.float64))
                G = np.array(af.shape)
                c0 = np.asarray(self.c0, np.int64)
                a0 = np.maximum(c0, 0)
                a1 = np.minimum(c0 + np.array(self.shape), G)
                if np.all(a1 > a0):
                    d0, d1 = a0 - c0, a1 - c0
                    out[d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]] = self._afull_g[a0[0]:a1[0], a0[1]:a1[1],
                                                                                a0[2]:a1[2]]
            self._avg = (k, out * self.vox)
        return self._avg[1]

    def _field_g(self, Y, v):
        Yg = cp.ascontiguousarray(Y.reshape(-1, 3) if isinstance(Y, cp.ndarray)
                                  else cp.asarray(np.asarray(Y, np.float64).reshape(-1, 3)))
        n = len(Yg)
        vg = cp.ascontiguousarray(cp.broadcast_to(v, (n,)) if isinstance(v, cp.ndarray)
                                  else cp.asarray(np.broadcast_to(np.asarray(v, np.float64), (n,))))
        lo = np.asarray(self.lo, np.float64)
        if n:
            i = cp.rint((Yg - cp.asarray(lo)) / self.h)
            ext = cp.asnumpy(cp.concatenate([i.min(0), i.max(0)]))
            if not np.isfinite(ext).all():         # the CPU class raises here; on the card a NaN's index would write
                raise FloatingPointError("SplatC1GPU: non-finite sample positions")   # outside the lattice
            ext = ext.astype(np.int64)
            lo_i, hi_i = ext[:3], ext[3:]
            # a crop with slack, kept while the samples stay inside it (a_v reused; nodes the samples do not reach
            # hold zero fibre and add nothing)
            k = self._crop
            need = np.prod(hi_i + 3 + 2 * self.SLACK - (lo_i - 2) + 0.0)
            if k is not None and (lo_i - 2 >= k[0]).all() and (hi_i + 3 <= k[0] + k[1]).all() and \
                    np.prod(k[1] + 0.0) <= self.SHRINK * need:
                c0, cs = k
            else:
                c0 = lo_i - 2 - self.SLACK
                cs = hi_i + 3 + self.SLACK - c0
                self._crop = (c0, cs)
        else:
            c0, cs = np.zeros(3, np.int64), np.full(3, 5, np.int64)
        shape = tuple(int(s) for s in cs)
        sx, sy = shape[1] * shape[2], shape[2]
        S = cp.zeros(shape[0] * sx, cp.float64)
        U = cp.empty((n, 3), cp.float64)
        B = cp.empty(n, cp.int64)
        _launch("splat", n, Yg, vg, np.int32(n), lo[0], lo[1], lo[2], np.float64(self.h), np.int64(c0[0]),
                np.int64(c0[1]), np.int64(c0[2]), np.int64(sx), np.int64(sy), S, U, B)
        F = _blur(S.reshape(shape))
        self._g = (U, B, vg, n, sx, sy)
        self.c0, self.shape = c0, shape
        self.base.c0, self.base.shape = self.c0, self.shape
        return F

    def _subfield_g(self, F, wk):
        U, B, vg, n, sx, sy = self._g
        vw = cp.ascontiguousarray(vg * cp.asarray(wk))
        S = cp.zeros(F.size, cp.float64)
        _launch("resplat", n, U, B, vw, np.int32(n), np.int64(sx), np.int64(sy), S)
        return _blur(S.reshape(F.shape))

    def _gather_g(self, A):
        U, B, _, n, sx, sy = self._g
        out = cp.empty(n, cp.float64)
        _launch("gather", n, cp.ascontiguousarray(A.ravel()), U, B, np.int32(n), np.int64(sx), np.int64(sy), out)
        return out

    def _adjoint_multi_g(self, pairs):
        U, B, _, n, sx, sy = self._g
        H = cp.empty((len(pairs), pairs[0][0].size), cp.float64)          # each blurred straight into its row
        for k, (G, _) in enumerate(pairs):
            _blur(G.reshape(self.shape), out=H[k])
        VW = cp.ascontiguousarray(cp.stack([vw if isinstance(vw, cp.ndarray) else
                                            cp.asarray(np.broadcast_to(np.asarray(vw, np.float64), (n,)))
                                            for _, vw in pairs], 0))
        g = cp.empty((n, 3), cp.float64)
        _launch("gather_grad", n, H, np.int32(len(pairs)), np.int64(H.shape[1]), VW, U, B, np.int32(n),
                np.int64(sx), np.int64(sy), np.float64(1.0 / self.h), g)
        return g

    # ---------------------------------------------------------------- CuPy in and out, for CoreGPU
    # the energies come back as 0-d CuPy arrays, read back with the core's other parts in one transfer
    def cohesive_g(self, Yg, vg, c, ca, wfg, cl=0.0, al=None, eps=1e-6):
        F = self._field_g(Yg, vg)
        vg = self._g[2]
        av = self._avox_g()
        if al is None or cl <= 0:                  # one pointwise kernel for the energies' densities and dE/dF
            Sw = self._subfield_g(F, wfg)
            tot = F.size
            e1, e2, G1, G2 = (cp.empty(tot, cp.float64) for _ in range(4))
            _launch("coh_pointwise", tot, F, Sw, av, np.float64(3.0 * float(c)), np.float64(-2.0 * float(ca)),
                    np.int64(tot), e1, e2, G1, G2)
            g = self._adjoint_multi_g([(G1, vg), (G2, vg * wfg)])
            return float(c) * cp.sum(e1), -float(ca) * cp.sum(e2), g
        ph = F / av
        p2 = ph * ph
        wx, wl = wfg * (1.0 - al), wfg * al
        px = self._subfield_g(F, wx) / av
        pl = self._subfield_g(F, wl) / av
        lg = cp.log(ph + eps)
        g = self._adjoint_multi_g([(3.0 * float(c) * p2 - float(cl) * pl / (ph + eps), vg),
                                   (-2.0 * float(ca) * px, vg * wx), (-float(cl) * lg, vg * wl)])
        return (float(c) * cp.sum(av * p2 * ph),
                -float(ca) * cp.sum(av * px * px) - float(cl) * cp.sum(av * pl * lg), g)

    def pressure_g(self, Yg, vg, c):
        F = self._field_g(Yg, vg)
        av = self._avox_g()
        tot = F.size
        e1, G1 = cp.empty(tot, cp.float64), cp.empty(tot, cp.float64)
        _launch("prs_pointwise", tot, F, av, np.float64(3.0 * float(c)), np.int64(tot), e1, G1)
        return float(c) * cp.sum(e1), self._adjoint_multi_g([(G1, self._g[2])])

    # ---------------------------------------------------------------- the CPU class's interface
    # cohesive() and pressure() return the fibre fraction phi ON THE CARD (a CuPy array): the settle's core drops it
    # and _adhesive passes it straight back to gather() - copying the crop (~11 MB) to the CPU every evaluation was
    # pure waste. cp.asnumpy(phi) where a NumPy copy is wanted.
    def field(self, Y, v):
        return cp.asnumpy(self._field_g(Y, v))

    def gather(self, A):
        return cp.asnumpy(self._gather_g(cp.asarray(A)))

    def adjoint(self, Gf, vw=None):
        vw = self._g[2] if vw is None else cp.asarray(np.asarray(vw, np.float64))
        return cp.asnumpy(self._adjoint_multi_g([(cp.asarray(Gf), vw)]))

    def adjoint_multi(self, pairs):
        return cp.asnumpy(self._adjoint_multi_g([(cp.asarray(G), vw) for G, vw in pairs]))

    def alignment(self, Y, v, d):
        F = self._field_g(Y, v)
        d = np.asarray(d, np.float64)
        Q = np.empty((len(Y), 6))
        for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
            Q[:, k] = cp.asnumpy(self._gather_g(self._subfield_g(F, d[:, a] * d[:, b])))
        tr = np.maximum(Q[:, 0] + Q[:, 1] + Q[:, 2], 1e-300)
        Qn = Q / tr[:, None]
        s2 = (Qn[:, :3] ** 2).sum(1) + 2.0 * (Qn[:, 3:] ** 2).sum(1)
        return np.clip(1.5 * s2 - 0.5, 0.0, 1.0)

    def cohesive(self, Y, v, c, ca, wf, cl=0.0, al=None, eps=1e-6):
        F = self._field_g(Y, v)
        vg = self._g[2]
        wfg = cp.asarray(np.asarray(wf, np.float64))
        av = self._avox_g()
        ph = F / av
        p2 = ph * ph
        if al is None or cl <= 0:
            pw = self._subfield_g(F, wfg) / av
            Es = cp.stack([cp.sum(av * p2 * ph), cp.sum(av * pw * pw)])
            g = self._adjoint_multi_g([(3.0 * float(c) * p2, vg), (-2.0 * float(ca) * pw, vg * wfg)])
            Eh = cp.asnumpy(Es)
            return float(c) * float(Eh[0]), -float(ca) * float(Eh[1]), cp.asnumpy(g), ph
        alg = cp.asarray(np.asarray(al, np.float64))
        wx, wl = wfg * (1.0 - alg), wfg * alg
        px = self._subfield_g(F, wx) / av
        pl = self._subfield_g(F, wl) / av
        lg = cp.log(ph + eps)
        Es = cp.asnumpy(cp.stack([cp.sum(av * p2 * ph), cp.sum(av * px * px), cp.sum(av * pl * lg)]))
        g = self._adjoint_multi_g([(3.0 * float(c) * p2 - float(cl) * pl / (ph + eps), vg),
                                   (-2.0 * float(ca) * px, vg * wx), (-float(cl) * lg, vg * wl)])
        return (float(c) * float(Es[0]), -float(ca) * float(Es[1]) - float(cl) * float(Es[2]), cp.asnumpy(g), ph)

    def pressure(self, Y, v, c, free_weight=True):
        F = self._field_g(Y, v)
        av = self._avox_g()
        ph = F / av
        p2 = ph * ph
        E = float(c) * float(cp.sum(av * p2 * ph))
        g = self._adjoint_multi_g([(3.0 * float(c) * p2, self._g[2])])
        return E, cp.asnumpy(g), ph


# ---------------------------------------------------------------- the body's distance on the card
def _sdf_dev(grid):
    """the body grid's lattices on the card (float64; the grid keeps float32, which converts exactly), kept on the
    grid object"""
    d = getattr(grid, "_gpu_dev", None)
    if d is None:
        def lat(A, lo, hi, h, blend=None):
            return {"A": cp.asarray(np.ascontiguousarray(A, np.float64)), "shape": tuple(int(s) for s in A.shape),
                    "lo": np.asarray(lo, np.float64), "hi": np.asarray(hi, np.float64), "h": float(h),
                    "lo_g": cp.asarray(np.asarray(lo, np.float64)), "hi_g": cp.asarray(np.asarray(hi, np.float64)),
                    "blend": blend}
        d = lat(grid.A, grid.lo, grid.hi, grid.h)
        d["fine"] = [lat(fb.A, fb.lo, fb.hi, fb.h, float(fb.blend)) for fb in getattr(grid, "fine", ())]
        grid._gpu_dev = d
    return d


def _lattice_args(L, P):
    return (L["A"], np.int64(L["shape"][0]), np.int64(L["shape"][1]), np.int64(L["shape"][2]),
            np.float64(L["lo"][0]), np.float64(L["lo"][1]), np.float64(L["lo"][2]), np.float64(L["h"]),
            cp.ascontiguousarray(P), np.int32(len(P)))


def _tri(L, P):
    v = cp.empty(len(P), cp.float64)
    _launch("tri_sdf", len(P), *_lattice_args(L, P), v)
    return v


def _bsp(L, P):
    v = cp.empty(len(P), cp.float64)
    g = cp.empty((len(P), 3), cp.float64)
    _launch("bsp_sdf", len(P), *_lattice_args(L, P), v, g)
    return v, g


def _smooth01(t):
    return t * t * (3 - 2 * t)


def _fine_w(fb, Pi):
    dlo, dhi = Pi - fb["lo_g"], fb["hi_g"] - Pi
    t = cp.clip(cp.minimum(dlo, dhi) / fb["blend"], 0, 1)
    return dlo, dhi, t, _smooth01(t)


def sdf_sample(grid, P):
    """grid.SDFGrid.sample(P) (value only, the fine boxes blended) on the card"""
    L = _sdf_dev(grid)
    Pc = cp.clip(P, L["lo_g"], L["hi_g"])
    o = P - Pc
    v = _tri(L, Pc) + cp.sqrt((o * o).sum(1))
    for fb in L["fine"]:
        inside = cp.nonzero(cp.all((P > fb["lo_g"]) & (P < fb["hi_g"]), 1))[0]
        if len(inside) == 0:
            continue
        Pi = P[inside]
        w = _fine_w(fb, Pi)[3].prod(1)
        v[inside] = (1 - w) * v[inside] + w * _tri(fb, Pi)
    return v


def smooth_distance(grid, P):
    """settle.smooth_distance(grid, P) on the card: the C1 distance and its gradient, the fine boxes blended"""
    L = _sdf_dev(grid)
    Pc = cp.clip(P, L["lo_g"], L["hi_g"])
    o = P - Pc
    do = cp.sqrt((o * o).sum(1))
    v, g = _bsp(L, Pc)
    out = do > 0
    v = v + do
    g = cp.where(out[:, None], g * (o == 0) + o / cp.where(out, do, 1.0)[:, None], g)
    for fb in L["fine"]:
        inside = cp.nonzero(cp.all((P > fb["lo_g"]) & (P < fb["hi_g"]), 1))[0]
        if len(inside) == 0:
            continue
        Pi = P[inside]
        dlo, dhi, t, sm = _fine_w(fb, Pi)
        w = sm.prod(1)
        fv, fg = _bsp(fb, Pi)
        ds = cp.where((t > 0) & (t < 1), 6 * t * (1 - t) / fb["blend"], 0.0) * cp.where(dlo < dhi, 1.0, -1.0)
        gw = cp.stack([ds[:, 0] * sm[:, 1] * sm[:, 2], sm[:, 0] * ds[:, 1] * sm[:, 2],
                       sm[:, 0] * sm[:, 1] * ds[:, 2]], 1)
        v0 = v[inside]
        g[inside] = (1 - w)[:, None] * g[inside] + w[:, None] * fg + (fv - v0)[:, None] * gw
        v[inside] = (1 - w) * v0 + w * fv
    return v, g


def fine_weight(grid, P):
    """settle.fine_weight on the card: the ears' fine crops' weight at P and its gradient"""
    L = _sdf_dev(grid)
    w = cp.zeros(len(P))
    gw = cp.zeros((len(P), 3))
    for fb in L["fine"]:
        inside = cp.nonzero(cp.all((P > fb["lo_g"]) & (P < fb["hi_g"]), 1))[0]
        if len(inside) == 0:
            continue
        dlo, dhi, t, sm = _fine_w(fb, P[inside])
        ds = cp.where((t > 0) & (t < 1), 6 * t * (1 - t) / fb["blend"], 0.0) * cp.where(dlo < dhi, 1.0, -1.0)
        w[inside] = sm.prod(1)
        gw[inside] = cp.stack([ds[:, 0] * sm[:, 1] * sm[:, 2], sm[:, 0] * ds[:, 1] * sm[:, 2],
                               sm[:, 0] * sm[:, 1] * ds[:, 2]], 1)
    return w, gw


# ---------------------------------------------------------------- the settle's core on the card
class CoreGPU:
    """settle.State.core(U[, forces]) on the card: the same terms in the same form - bending (bend_carried as a
    kernel, one thread per guide, or the plain set), gravity, the samples, body contact (C1: the trilinear distance
    picks the samples near the skin, the B-spline distance pushes them), the hair-hair field (SplatC1GPU), friction,
    the acts, the sample forces back to the segment ends and the directions, the conditioning. U and dE/dU cross to and
    from the card; the energy's parts come back in one transfer and are summed in the CPU core's order.

    The acts run on the CPU through a NumPy view (they are on only while styling). The state's arrays are uploaded
    when the settle replaces them (_setup, _anchors, wave_frame assign new arrays; the cache holds the arrays it was
    built from, so an identity cannot be reused); r.rho is edited in place, so the plain bending uploads it each call."""

    def __init__(self):
        self._key = None
        self._held = None
        self.a = {}
        self._scratch = None

    def _sync(self, st):
        r = st.rods
        hx = bool(st.hx_frame)
        held = (st.act, st.lk, st.cb, st.Wk, r.X, r.d0, st.sq, st.st, st.sw, st.sv, st.swf, st.sal, st.f_on,
                st.f_anc, st.f_n, st.f_mn, st.loc if hx else None, st.wave_n0c if hx else None)
        key = (st.Ka, hx) + tuple(id(x) for x in held)
        if key == self._key:
            return self.a

        def up(x):
            return None if x is None else cp.asarray(np.ascontiguousarray(x))
        a = {k: up(x) for k, x in zip(("act", "lk", "cb", "Wk", "X", "d0", "q", "t", "sw", "sv", "swf", "sal", "on",
                                        "anc", "nf", "mn"), held[:16])}
        if hx:
            a["loc"] = up(st.loc[:, :st.Ka])
            a["n0"] = up(st.wave_n0c)
        self.a, self._key, self._held = a, key, held
        return a

    def _bend(self, st, a, D):
        r = st.rods
        G, Ka = r.G, st.Ka
        cb = a["cb"]
        if st.hx_frame:
            if self._scratch is None or self._scratch[0] != (G, Ka):
                self._scratch = ((G, Ka), [cp.empty((G, Ka, 3)), cp.empty((G, Ka, 3)), cp.empty((G, Ka)),
                                           cp.empty((G, Ka)), cp.empty((G, Ka, 3))])
            T, N, C, Aa, Tb = self._scratch[1]
            gD = cp.empty((G, Ka, 3))
            Eg = cp.empty(G)
            _launch("bend_carried", G, D, a["d0"], a["n0"], a["loc"], cb, np.int32(G), np.int32(Ka), T, N, C, Aa,
                    Tb, gD, Eg)
            return Eg.sum(), gD
        rho = cp.asarray(np.ascontiguousarray(r.rho[:, :Ka]))
        Dp = cp.empty_like(D)
        Dp[:, 0] = a["d0"]
        Dp[:, 1:] = D[:, :-1]
        dd = D - Dp - rho
        cdd = cb[..., None] * dd
        gD = cdd.copy()
        gD[:, :-1] -= cdd[:, 1:]
        return 0.5 * (cdd * dd).sum(), gD

    def _factor(self, st):
        """the preconditioner's factor and lmap on the card, refreshed when precondition() builds a new one"""
        F = st.F
        if getattr(self, "_F", (None,))[0] is not F:
            Dc, Di, S1, S2 = F
            self._F = (F, st.lmap, [cp.asarray(np.ascontiguousarray(M, np.float64)) for M in (Di, S1, S2)],
                       cp.asarray(np.ascontiguousarray(st.lmap, np.float64)))
        return self._F[2], self._F[3]

    def fun(self, st, x, keep=False):
        """settle.State.solve's objective with a factor, on the card: x -> U (u_of_x), the core, dE/dU -> the gradient
        in x (gx_of_gu). Only x and the gradient cross. -> (E * ESC, gradient) like solve's f. keep: the evaluation's
        U, dE/dU and the core's force extras stay on the card for kept() (the gated settle's gate reads them at every
        5th accepted iterate; copying them out at every evaluation cost ~30 ms each)"""
        G, Ka = st.rods.G, st.Ka
        (Di, S1, S2), lmap = self._factor(st)
        xg = cp.asarray(np.ascontiguousarray(x, np.float64))
        y = cp.empty((G, Ka, 3))
        U = cp.empty((G, Ka, 3))
        _launch("u_of_x", G, xg, Di, S1, S2, lmap, np.int32(G), np.int32(Ka), y, U)
        E, gU = self._core(st, U, "keep" if keep else False)
        w = cp.empty((G, Ka, 3))
        _launch("gx_of_gu", G, cp.ascontiguousarray(gU), Di, S1, S2, lmap, np.int32(G), np.int32(Ka), w)
        return E * st.ESC, cp.asnumpy(w).ravel() * st.ESC

    def kept(self):
        """the last fun(keep=True) evaluation as NumPy: (U, dE/dU, the core's forces=True extras)"""
        k = self._kept
        h = cp.asnumpy
        o = {n: h(k[n]) for n in ("gD", "D", "P0", "P1", "Y", "fc", "fp")}
        o["parts"] = dict(k["parts"])
        return h(k["U"]), h(k["gU"]), o

    def core(self, st, U, forces=False):
        """settle.State.core(U, forces) on the card; with forces the same extras as the CPU core, as NumPy arrays"""
        r = self._core(st, cp.asarray(np.ascontiguousarray(U, np.float64)), bool(forces))
        if forces:
            E, gU, o = r
            h = cp.asnumpy
            o = dict(o, **{n: h(o[n]) for n in ("gD", "D", "P0", "P1", "Y", "fc", "fp")})
            return E, h(gU), o
        return r[0], cp.asnumpy(r[1])

    def _core(self, st, U, forces=False):
        a = self._sync(st)
        r = st.rods
        G, Ka = r.G, st.Ka
        act, lk, cb, X = a["act"], a["lk"], a["cb"], a["X"]
        nu = cp.where(act, cp.sqrt((U * U).sum(2)), 1.0)
        D = cp.ascontiguousarray(U / nu[..., None])
        P1 = X[:, None, :] + cp.cumsum(lk[..., None] * D, 1)
        P0 = cp.empty_like(P1)
        P0[:, 0] = X
        P0[:, 1:] = P1[:, :-1]
        zero = cp.zeros((), cp.float64)
        Eb, gD = self._bend(st, a, D)
        Wk = a["Wk"]
        Eg = (Wk * (0.5 * (P0[..., 2] + P1[..., 2]) - X[:, None, 2])).sum()
        gP0 = cp.zeros_like(P0)
        gP1 = cp.zeros_like(P1)
        gP0[..., 2] += 0.5 * Wk
        gP1[..., 2] += 0.5 * Wk
        q, t = a["q"], a["t"]
        P0f = P0.reshape(-1, 3)
        P1f = P1.reshape(-1, 3)
        Y = P0f[q] + t[:, None] * (P1f[q] - P0f[q])
        gY = cp.zeros_like(Y)
        fc = fp = None
        Ec = zero
        if st.body is not None:
            near = cp.nonzero(sdf_sample(st.body, Y) < st.h_c + st.margin)[0]
            if len(near):
                sdn, gsn = smooth_distance(st.body, Y[near])
                pen_n = st.h_c - sdn
                hit = pen_n > 0
                kp = cp.where(hit, st.k_c * a["sw"][near] * pen_n, 0.0)
                gc = cp.zeros((len(near), 3))
                ek = float(st.f.get("contact_ear_k", 1.0))
                if ek != 1.0:                   # the ears' stiffer skin (settle contact_ear_k), as the CPU core
                    w, gw = fine_weight(st.body, Y[near])
                    gc = cp.where(hit, 0.5 * (ek - 1.0) * st.k_c * a["sw"][near] * pen_n * pen_n, 0.0)[:, None] * gw
                    kp = kp * (1.0 + (ek - 1.0) * w)
                Ec = 0.5 * (kp * cp.where(hit, pen_n, 0.0)).sum()
                gc = gc - kp[:, None] * gsn
                gY[near] += gc
                if forces:
                    fc = cp.zeros_like(Y)
                    fc[near] = gc
        Ep = Ea = zero
        if st.pressure:
            if st.c_a > 0:
                Ep, Ea, gp = st.splat.cohesive_g(Y, a["sv"], st.c_p, st.c_a, a["swf"], cl=st.c_l, al=a["sal"])
            else:
                Ep, gp = st.splat.pressure_g(Y, a["sv"], st.c_p)
            gY += gp
            if forces:
                fp = gp
        Ef = zero
        if st.friction and st.mu > 0 and len(st.f_on):
            on, nf, mn = a["on"], a["nf"], a["mn"]
            dl = Y[on] - a["anc"]
            tdl = dl - (dl * nf).sum(1, keepdims=True) * nf
            sq_ = cp.sqrt(1.0 + (tdl * tdl).sum(1) / st.eps ** 2)
            Ef = (mn * st.eps * (sq_ - 1.0)).sum()
            gY[on] += (mn / (st.eps * sq_))[:, None] * tdl
        E_acts = 0.0
        gPa = None
        if st.acts is not None and hasattr(st.acts, "energy"):
            v = st._view(cp.asnumpy(P0), cp.asnumpy(P1), cp.asnumpy(D), cp.asnumpy(Y))
            E_acts, ga = st.acts.energy(v)
            ga = ga or {}
            if ga.get("Y") is not None:
                gY += cp.asarray(ga["Y"])
            if ga.get("D") is not None:
                gD += cp.asarray(ga["D"])
            if ga.get("P") is not None:
                gPa = cp.asarray(ga["P"])
        n = G * Ka
        for c in range(3):
            gP1[..., c] += cp.bincount(q, weights=t * gY[:, c], minlength=n).reshape(G, Ka)
            gP0[..., c] += cp.bincount(q, weights=(1 - t) * gY[:, c], minlength=n).reshape(G, Ka)
        gP1[:, :-1] += gP0[:, 1:]
        if gPa is not None:
            gP1 += gPa[:, 1:]
        gD += lk[..., None] * cp.cumsum(gP1[:, ::-1], 1)[:, ::-1]
        gD = cp.where(act[..., None], gD, 0.0)
        Econd = 0.5 * (cb * (nu - 1) ** 2).sum()
        gU = (gD - (gD * D).sum(2, keepdims=True) * D) / nu[..., None] + (cb * (nu - 1))[..., None] * D
        gU = cp.where(act[..., None], gU, 0.0)
        Eh = cp.asnumpy(cp.stack([cp.asarray(e, cp.float64) for e in (Eb, Eg, Ec, Ep, Ea, Ef, Econd)]))
        parts = [float(x) for x in Eh[:6]] + [float(E_acts), float(Eh[6])]   # the CPU core's order
        self.parts = dict(zip(("bend", "gravity", "contact", "pressure", "cohesion", "friction", "acts", "cond"), parts))
        if not forces:
            return sum(parts), gU
        pd = dict(self.parts)
        if not (st.pressure and st.c_a > 0):
            pd.pop("cohesion")                         # the CPU core reports cohesion only with the cohesive field
        # the extras on the card (core() copies them out; fun(keep=True) leaves them for kept())
        o = dict(parts=pd, gD=gD, D=D, P0=P0, P1=P1, Y=Y, fc=cp.zeros_like(Y) if fc is None else fc,
                 fp=cp.zeros_like(Y) if fp is None else fp)
        if forces == "keep":
            self._kept = dict(o, U=U, gU=gU)
            return sum(parts), gU
        return sum(parts), gU, o
