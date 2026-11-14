// Field Studio GPU field program (2026-11-08, r22): core/program.py's stack machine on the card, in double precision,
// for brick admission and volume builds (the true field: no guard / box shortcut, as eval_local_t with tau = 1e9).
// Value at a lattice node = base program + the sculpt layer's node value (nodes coincide with the sculpt lattice, so no
// interpolation). Opcodes and parameter layouts: program.py (kept in step by gpu/tests: the CPU / GPU comparison).
#include <cuda_fp16.h>
#ifndef REAL
#define REAL double
#endif
typedef REAL real;

#define OP_SPHERE 1
#define OP_BOX 2
#define OP_CYL 3
#define OP_CAPSULE 4
#define OP_CONE 5
#define OP_TORUS 6
#define OP_ELLIPSOID 7
#define OP_SLAB 8
#define OP_SAMPLED 9
#define OP_UNION 20
#define OP_SUB 21
#define OP_INTER 22
#define OP_SUNION 23
#define OP_SSUB 24
#define OP_SINTER 25
#define OP_XFORM 30
#define OP_TWIST 31
#define OP_BEND 32
#define OP_TAPER 33
#define OP_POP 39
#define OP_SHELL 40
#define OP_OFFSET 41
#define OP_GUARD 50
#define OP_GUARD_U 51
#define OP_OSWEEP 52
#define OP_FV8 53
#define STK 48

__device__ __forceinline__ real smin_d(real a, real b, real k) {
    real h = fmax(k - fabs(a - b), (real)0.0) / k;
    return fmin(a, b) - h * h * k * (real)0.25;
}

// ---- fs-materials 11-14: OP_MATGEN, the material generators (materials/fieldops.py; same layout, same hash noise) ----
// This block is copied verbatim into gpu/program.cu (between the fs-materials markers); materials/tests check the two
// copies are identical and that the card's values equal the CPU's.
#define OP_MATGEN 60
#define MG_OCT 14
#define MG_CHIP 15
__device__ __forceinline__ unsigned int mg_h32(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    return (x >> 16) ^ x;
}
__device__ __forceinline__ unsigned int mg_hc(int ix, int iy, int iz, int seed) {
    return mg_h32((unsigned int)ix ^ mg_h32((unsigned int)iy ^ mg_h32((unsigned int)iz ^ mg_h32((unsigned int)seed))));
}
__device__ __forceinline__ real mg_fade(real t) { return t * t * t * (t * (t * (real)6.0 - (real)15.0) + (real)10.0); }
__device__ __forceinline__ real mg_grad(unsigned int h, real x, real y, real z) {
    h &= 15u;
    real u = h < 8u ? x : y;
    real v = h < 4u ? y : ((h == 12u || h == 14u) ? x : z);
    return ((h & 1u) == 0u ? u : -u) + ((h & 2u) == 0u ? v : -v);
}
__device__ real mg_noise3(real x, real y, real z, int seed) {
    real fx = floor(x), fy = floor(y), fz = floor(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    real xf = x - fx, yf = y - fy, zf = z - fz;
    real u = mg_fade(xf), v = mg_fade(yf), w = mg_fade(zf);
    real res = (real)0.0;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                real g = mg_grad(mg_hc(ix + dx, iy + dy, iz + dz, seed), xf - dx, yf - dy, zf - dz);
                real wx = dx ? u : (real)1.0 - u, wy = dy ? v : (real)1.0 - v, wz = dz ? w : (real)1.0 - w;
                res += wx * wy * wz * g;
            }
    return res;
}
__device__ real mg_fbm(const real* prm, int b, int n, real x, real y, real z) {
    real s = (real)0.0;
    for (int k = 0; k < n; k++) {
        const real* r = prm + b + k * MG_OCT;
        real f = r[0];
        real X = (r[2] * x + r[3] * y + r[4] * z) * f + r[11];
        real Y = (r[5] * x + r[6] * y + r[7] * z) * f + r[12];
        real Z = (r[8] * x + r[9] * y + r[10] * z) * f + r[13];
        s += r[1] * mg_noise3(X, Y, Z, k * 7919 + 17);
    }
    return s;
}
// noise and its analytic gradient (g[0..2])
__device__ real mg_noise3g(real x, real y, real z, int seed, real* g) {
    real fx = floor(x), fy = floor(y), fz = floor(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    real xf = x - fx, yf = y - fy, zf = z - fz;
    real u = mg_fade(xf), v = mg_fade(yf), w = mg_fade(zf);
    real du = (real)30.0 * xf * xf * (xf - (real)1.0) * (xf - (real)1.0);
    real dv = (real)30.0 * yf * yf * (yf - (real)1.0) * (yf - (real)1.0);
    real dw = (real)30.0 * zf * zf * (zf - (real)1.0) * (zf - (real)1.0);
    real res = (real)0.0, gx = (real)0.0, gy = (real)0.0, gz = (real)0.0;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                unsigned int h = mg_hc(ix + dx, iy + dy, iz + dz, seed) & 15u;
                real su = (h & 1u) == 0u ? (real)1.0 : (real)-1.0, sv = (h & 2u) == 0u ? (real)1.0 : (real)-1.0;
                real g0 = (real)0.0, g1 = (real)0.0, g2 = (real)0.0;
                if (h < 8u) g0 += su; else g1 += su;
                if (h < 4u) g1 += sv; else if (h == 12u || h == 14u) g0 += sv; else g2 += sv;
                real dot = g0 * (xf - dx) + g1 * (yf - dy) + g2 * (zf - dz);
                real wx = dx ? u : (real)1.0 - u, wy = dy ? v : (real)1.0 - v, wz = dz ? w : (real)1.0 - w;
                real dwx = dx ? du : -du, dwy = dy ? dv : -dv, dwz = dz ? dw : -dw;
                real www = wx * wy * wz;
                res += www * dot;
                gx += dwx * wy * wz * dot + www * g0;
                gy += wx * dwy * wz * dot + www * g1;
                gz += wx * wy * dwz * dot + www * g2;
            }
    g[0] = gx; g[1] = gy; g[2] = gz;
    return res;
}
__device__ real mg_fbm_g(const real* prm, int b, int n, real x, real y, real z, real* g) {
    real s = (real)0.0;
    g[0] = g[1] = g[2] = (real)0.0;
    for (int k = 0; k < n; k++) {
        const real* r = prm + b + k * MG_OCT;
        real f = r[0];
        real X = (r[2] * x + r[3] * y + r[4] * z) * f + r[11];
        real Y = (r[5] * x + r[6] * y + r[7] * z) * f + r[12];
        real Z = (r[8] * x + r[9] * y + r[10] * z) * f + r[13];
        real ng[3];
        real v = mg_noise3g(X, Y, Z, k * 7919 + 17, ng);
        real a = r[1], af = a * f;
        s += a * v;
        g[0] += af * (ng[0] * r[2] + ng[1] * r[5] + ng[2] * r[8]);
        g[1] += af * (ng[0] * r[3] + ng[1] * r[6] + ng[2] * r[9]);
        g[2] += af * (ng[0] * r[4] + ng[1] * r[7] + ng[2] * r[10]);
    }
    return s;
}
__device__ __forceinline__ real mg_u01(unsigned int h) { return (real)(h & 0xFFFFFFu) / (real)16777216.0; }
__device__ real mg_pits(real cell, real prob, real depth, real rfac, int seed, real x, real y, real z, real* g) {
    real gx = x / cell, gy = y / cell, gz = z / cell;
    int cx = (int)floor(gx), cy = (int)floor(gy), cz = (int)floor(gz);
    real best = (real)0.0;
    g[0] = g[1] = g[2] = (real)0.0;
    for (int i = cx - 1; i < cx + 2; i++)
        for (int j = cy - 1; j < cy + 2; j++)
            for (int k = cz - 1; k < cz + 2; k++) {
                if (mg_u01(mg_hc(i, j, k, seed)) >= prob) continue;
                real px = i + mg_u01(mg_hc(i, j, k, seed + 1)), py = j + mg_u01(mg_hc(i, j, k, seed + 2)),
                     pz = k + mg_u01(mg_hc(i, j, k, seed + 3));
                real rad = rfac * ((real)0.7 + (real)0.6 * mg_u01(mg_hc(i, j, k, seed + 4)));
                real sx = (real)0.65 + (real)0.8 * mg_u01(mg_hc(i, j, k, seed + 6));
                real sy = (real)0.65 + (real)0.8 * mg_u01(mg_hc(i, j, k, seed + 7));
                real sz = (real)0.65 + (real)0.8 * mg_u01(mg_hc(i, j, k, seed + 8));
                real dx = (gx - px) * sx, dy = (gy - py) * sy, dz = (gz - pz) * sz;
                real r2 = rad * rad;
                real t2 = (dx * dx + dy * dy + dz * dz) / r2;
                real tx = (real)2.0 * dx * sx / (r2 * cell), ty = (real)2.0 * dy * sy / (r2 * cell),
                     tz = (real)2.0 * dz * sz / (r2 * cell);
                if (t2 < (real)1.43) {
                    real wg[3];
                    real wb = mg_noise3g(gx * (real)3.0, gy * (real)3.0, gz * (real)3.0, seed + 9, wg) * (real)3.7;
                    real k3 = (real)3.7 * (real)3.0 / cell;
                    if (wb > (real)1.0 || wb < (real)-1.0) { wb = fmin((real)1.0, fmax((real)-1.0, wb)); k3 = (real)0.0; }
                    real f_ = (real)1.0 + (real)0.3 * wb;
                    tx = tx * f_ + t2 * (real)0.3 * k3 * wg[0];
                    ty = ty * f_ + t2 * (real)0.3 * k3 * wg[1];
                    tz = tz * f_ + t2 * (real)0.3 * k3 * wg[2];
                    t2 = t2 * f_;
                }
                if (t2 < (real)1.0) {
                    real b = (real)1.0 - t2;
                    real dep = depth * ((real)0.5 + mg_u01(mg_hc(i, j, k, seed + 5)));
                    real v = -dep * b * b;
                    if (v < best) {
                        best = v;
                        g[0] = (real)2.0 * dep * b * tx; g[1] = (real)2.0 * dep * b * ty; g[2] = (real)2.0 * dep * b * tz;
                    }
                }
            }
    return best;
}
__device__ real mg_relief(const real* prm, int o, real x, real y, real z, real* g) {
    int n = (int)prm[o + 3];
    real r = mg_fbm_g(prm, o + 4, n, x, y, z, g);
    int p = o + 4 + n * MG_OCT;
    if (prm[p] > (real)0.0) {
        real a[3];
        r += mg_pits(prm[p], prm[p + 1], prm[p + 2], prm[p + 3], (int)prm[p + 4], x, y, z, a);
        g[0] += a[0]; g[1] += a[1]; g[2] += a[2];
    }
    if (prm[p + 9] != (real)0.0) {
        real sp = prm[p + 8];
        real ph = (prm[p + 5] * x + prm[p + 6] * y + prm[p + 7] * z) / sp;
        real hx = prm[p + 5] / sp, hy = prm[p + 6] / sp, hz = prm[p + 7] / sp;
        if (prm[p + 10] != (real)0.0) {
            real wf = prm[p + 11], wg[3];
            real wn = mg_noise3g(x * wf, y * wf, z * wf, 104729, wg);
            real kw = prm[p + 10] * (real)5.0;
            ph += kw * wn;
            hx += kw * wf * wg[0]; hy += kw * wf * wg[1]; hz += kw * wf * wg[2];
        }
        // scalloped strokes; a claw chisel's passes, the deepest cut, each cut its own width (fieldops.py relief)
        real a = prm[p + 9], jit = (real)0.0, best = (real)0.0, kk = (real)0.0, gw = (real)1.0;
        int npass = 1, sd = (int)prm[p + 4];
        bool blows = prm[o + 1] >= (real)((p - o) + 20) && prm[p + 17] > (real)0.0;   // update 20: the blows
        real lb = blows ? prm[p + 17] : (real)1.0, bmg[3] = {(real)0.0, (real)0.0, (real)0.0};
        if (blows) {
            real g2[3];
            real n2 = mg_noise3g(x / lb + (real)0.83, y / lb + (real)0.29, z / lb + (real)0.57, 104779, g2);
            real ko = prm[p + 19] * (real)3.71;
            ph += ko * n2;
            hx += ko * g2[0] / lb; hy += ko * g2[1] / lb; hz += ko * g2[2] / lb;
        }
        if (prm[o + 1] >= (real)((p - o) + 16)) { npass = max(1, (int)prm[p + 14]); jit = prm[p + 15]; }
        if (prm[o + 1] >= (real)((p - o) + 17) && prm[p + 16] > (real)0.0) gw = fmin(prm[p + 16], (real)1.0);
        for (int i = 0; i < npass; i++) {
            real phi = ph, dsc = (real)1.0;
            if (i > 0) {
                phi = ph + mg_u01(mg_hc(i, 13, sd, 104743));
                dsc = (real)0.5 + (real)0.5 * mg_u01(mg_hc(i, 19, sd, 104759));
            }
            real k_ = floor(phi), b0 = k_, w_ = (real)1.0;
            if (jit > (real)0.0) {
                b0 = k_ + jit * (mg_u01(mg_hc((int)k_, 29 + i, sd, 104761)) - (real)0.5);
                if (phi < b0) { k_ -= (real)1.0; b0 = k_ + jit * (mg_u01(mg_hc((int)k_, 29 + i, sd, 104761)) - (real)0.5); }
                real b1 = k_ + (real)1.0 + jit * (mg_u01(mg_hc((int)k_ + 1, 29 + i, sd, 104761)) - (real)0.5);
                if (phi >= b1) {
                    k_ += (real)1.0; b0 = b1;
                    b1 = k_ + (real)1.0 + jit * (mg_u01(mg_hc((int)k_ + 1, 29 + i, sd, 104761)) - (real)0.5);
                }
                w_ = b1 - b0;
            }
            real s_ = ((real)2.0 * (phi - b0) / w_ - (real)1.0) / gw;
            real j_ = (real)0.6 + (real)0.8 * mg_u01(mg_hc((int)k_, 7, 11, 104723 + 97 * i));
            real ri = (real)0.0, ki = (real)0.0;
            if (s_ * s_ < (real)1.0) {
                ri = -a * dsc * j_ * ((real)1.0 - s_ * s_);
                ki = a * dsc * j_ * (real)4.0 * s_ / (w_ * gw);
            }
            real mg[3] = {(real)0.0, (real)0.0, (real)0.0};
            if (blows && ri != (real)0.0) {           // each groove its own blows (fieldops.py relief)
                unsigned int ho = mg_hc((int)k_, 41 + i, sd, 104789);
                real o1 = (real)97.0 * mg_u01(ho), o2 = (real)97.0 * mg_u01(mg_h32(ho ^ 0x2545f491u)),
                     o3 = (real)97.0 * mg_u01(mg_h32(ho ^ 0x68e31da4u));
                real g1[3];
                real n1 = mg_noise3g(x / lb + o1, y / lb + o2, z / lb + o3, 104773, g1);
                real m_ = (real)0.85 + prm[p + 18] * (real)3.71 * n1;
                if (m_ <= (real)0.0) { ri = (real)0.0; ki = (real)0.0; }
                else {
                    real mm = fmin(m_, (real)1.6);
                    if (m_ < (real)1.6) {
                        real kb = prm[p + 18] * (real)3.71 / lb;
                        mg[0] = ri * kb * g1[0]; mg[1] = ri * kb * g1[1]; mg[2] = ri * kb * g1[2];
                    }
                    ri *= mm; ki *= mm;
                }
            }
            if (i == 0 || ri < best) { best = ri; kk = ki; bmg[0] = mg[0]; bmg[1] = mg[1]; bmg[2] = mg[2]; }
        }
        r += best;
        g[0] += kk * hx + bmg[0]; g[1] += kk * hy + bmg[1]; g[2] += kk * hz + bmg[2];
    }
    return r;
}
// materials/fieldops.py relief_value: (F - r) / sqrt(1 + |grad r|^2) within B of the surface, slope 1 / (1 + g_max) beyond
__device__ real mg_relief_value(const real* prm, int o, real F, real x, real y, real z) {
    real g[3];
    real r = mg_relief(prm, o, x, y, z, g);
    real d = F - r;
    int p = o + 4 + (int)prm[o + 3] * MG_OCT;
    real B = prm[p + 12];
    if (B <= (real)0.0) return d;
    real k = sqrt((real)1.0 + g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    real ad = fabs(d);
    if (ad <= B) return d / k;
    real v = B / k + (ad - B) / ((real)1.0 + prm[p + 13]);
    return d > (real)0.0 ? v : -v;
}
__device__ real mg_chips(const real* prm, int o, real x, real y, real z) {
    int K = (int)prm[o + 2], n = (int)prm[o + 3], c0 = o + 4 + n * MG_OCT;
    real best = (real)1e9;
    for (int k = 0; k < K; k++) {
        const real* c = prm + c0 + k * MG_CHIP;
        real dx = x - c[0], dy = y - c[1], dz = z - c[2];
        real rx = c[12], ry = c[13], rz = c[14];
        real rmin = fmin(rx, fmin(ry, rz)), rmax = fmax(rx, fmax(ry, rz));
        real dd = sqrt(dx * dx + dy * dy + dz * dz);
        if (dd - rmax > best && (dd / rmax - (real)1.0) * rmin > best) continue;
        real lx = (c[3] * dx + c[4] * dy + c[5] * dz) / rx;
        real ly = (c[6] * dx + c[7] * dy + c[8] * dz) / ry;
        real lz = (c[9] * dx + c[10] * dy + c[11] * dz) / rz;
        real v = (sqrt(lx * lx + ly * ly + lz * lz) - (real)1.0) * rmin;
        if (v < best) best = v;
    }
    int ce = c0 + K * MG_CHIP;
    real vwear = (real)1e9;
    if (prm[o + 1] >= (real)((ce - o) + 8)) {          // the arris crumble (fieldops.py chips_d)
        real ax = fabs(x) - prm[ce], ay = fabs(y) - prm[ce + 1], az = fabs(z) - prm[ce + 2];
        real mx = fmax(ax, (real)0.0), my = fmax(ay, (real)0.0), mz = fmax(az, (real)0.0);
        real e = fmin(sqrt(ax * ax + ay * ay + mz * mz), fmin(sqrt(ax * ax + az * az + my * my), sqrt(ay * ay + az * az + mx * mx)));
        real sag = prm[ce + 3];
        int nx = (int)prm[o + 1] - (ce - o) - 8;        // extra floats: 5 wear, 2 a second bite octave, 7 both
        if (nx == 5 || nx == 7) {                       // weathering: the arrises worn round (fieldops.py chips_d)
            real lw = prm[ce + 10];
            real nw = mg_noise3(x / lw + (real)0.71, y / lw + (real)0.43, z / lw + (real)0.19, (int)prm[ce + 12]) * (real)3.71;
            real rw = prm[ce + 8] + prm[ce + 9] * fmax((real)1.0 + prm[ce + 11] * nw, (real)0.0);
            sag = rw * (real)0.41421356;
            real bx = ax + rw, by = ay + rw, bz = az + rw;
            real px = fmax(bx, (real)0.0), py = fmax(by, (real)0.0), pz = fmax(bz, (real)0.0);
            real sd = sqrt(px * px + py * py + pz * pz) + fmin(fmax(bx, fmax(by, bz)), (real)0.0) - rw;
            vwear = fmax(-sd, (real)-0.01);             // joined after the roughness (fieldops.py)
        }
        real lam = prm[ce + 5];
        real nn = mg_noise3(x / lam + (real)0.31, y / lam + (real)0.17, z / lam + (real)0.53, (int)prm[ce + 7]) * (real)3.71;
        real w = prm[ce + 4] * fmax(nn - prm[ce + 6], (real)0.0) / ((real)1.0 - prm[ce + 6]);
        if (nx == 2 || nx == 7) {                       // a second bite octave (fieldops.py chips_d)
            int c2 = ce + nx + 6;
            real l2 = prm[c2 + 1];
            real n2 = mg_noise3(x / l2 + (real)0.77, y / l2 + (real)0.41, z / l2 + (real)0.23, (int)prm[ce + 7] + 101) * (real)3.71;
            w += prm[c2] * fmax(n2, (real)0.0);
        }
        real v = e - (sag + w);
        if (v < best) best = v;
    }
    if (n > 0 && best < (real)0.01) best += mg_fbm(prm, o + 4, n, x, y, z);
    return fmin(best, vwear);
}
__device__ real mg_crack(const real* prm, int o, real x, real y, real z) {
    real dx = x - prm[o + 2], dy = y - prm[o + 3], dz = z - prm[o + 4];
    real sn = prm[o + 5] * dx + prm[o + 6] * dy + prm[o + 7] * dz;
    real s = prm[o + 8] * dx + prm[o + 9] * dy + prm[o + 10] * dz;
    real t = prm[o + 11] * dx + prm[o + 12] * dy + prm[o + 13] * dz;
    real D = prm[o + 14], w0 = prm[o + 15], hl = prm[o + 16], tf = prm[o + 18];
    real s_tip = D * ((real)1.0 + prm[o + 17] * mg_noise3(t * tf, (real)0.5, (real)0.25, (int)prm[o + 20]) * (real)5.0);
    if (hl < (real)1.0) {
        real e2 = (real)1.0 - (t / hl) * (t / hl);
        s_tip = e2 > (real)0.0 ? s_tip * sqrt(e2) : (real)0.0;
    }
    int n = (int)prm[o + 21];
    real S = sn + mg_fbm(prm, o + 22, n, x, y, z);
    real w;
    if (s <= (real)0.0) w = w0;
    else if (s_tip <= (real)0.0 || s >= s_tip) w = (real)0.0;
    else {
        real q = (real)1.0 - s / s_tip;
        real m = w0 / ((real)2.0 * s_tip * prm[o + 19]);
        real sq = sqrt(q);
        w = sq >= m ? w0 * sq : w0 * q / m;
    }
    real d = fabs(S) - (real)0.5 * w;
    d = fmax(d, s - s_tip);
    if (hl < (real)1.0) d = fmax(d, fabs(t) - hl);
    return d;
}
__device__ real mg_apply(const real* prm, int o, real F, real x, real y, real z) {
    int kind = (int)prm[o];
    if (kind == 1) return mg_relief_value(prm, o, F, x, y, z);
    if (kind == 2) return fmax(F, -mg_chips(prm, o, x, y, z));
    if (kind == 3) return fmax(F, -mg_crack(prm, o, x, y, z));
    return F;
}
// ---- end fs-materials --------------------------------------------------------------------------------------------------

// milestone (11-08): program.py _fv8 / _fv8_bid / _fv8_raw / _fv8_coarser, same layout and rule (core/fv8base.py): her
// store as an editor sees it (capped samples take the coarser levels' value: fv8_field.pre_estimate)
// 11-09 (pool_measure: her 1.1 GB store went to the card as float64, appended to the parameters): the store's samples
// stay float32 in their own buffer, FD (set by gpu/geval.py before each launch); read as real (double), exactly as the
// float32 -> float64 copy read them
__device__ const float* FD;
__device__ __forceinline__ long long fv8_off(const real* prm, int o, int k) {
    return (long long)prm[o + k] * 1048576LL + (long long)prm[o + k + 1];
}
__device__ int fv8_bid(const real* prm, int o, long long D, int l, real px, real py, real pz, int* bx, int* by, int* bz) {
    real tm = prm[o + 3];
    int nx = (int)prm[o + 4], ny = (int)prm[o + 5], nz = (int)prm[o + 6];
    int tx = (int)floor((px - prm[o]) / tm), ty = (int)floor((py - prm[o + 1]) / tm), tz = (int)floor((pz - prm[o + 2]) / tm);
    if (tx < 0 || ty < 0 || tz < 0 || tx >= nx || ty >= ny || tz >= nz) return -2;
    int nb = (int)prm[o + 16 + l];
    if (nb == 0) return -2;
    int slot = (int)(real)FD[D + fv8_off(prm, o, 24) + ((long long)(tx * ny + ty) * nz + tz) * 4 + l];
    if (slot < 0) return -2;
    real h = prm[o + 8 + l];
    *bx = (int)floor((px - prm[o]) / h * (real)0.125); *by = (int)floor((py - prm[o + 1]) / h * (real)0.125);
    *bz = (int)floor((pz - prm[o + 2]) / h * (real)0.125);
    int lx = *bx - tx * nb, ly = *by - ty * nb, lz = *bz - tz * nb;
    if (lx < 0 || ly < 0 || lz < 0 || lx >= nb || ly >= nb || lz >= nb) return -2;
    return (int)(real)FD[D + fv8_off(prm, o, 26 + 2 * l) + (long long)slot * nb * nb * nb + (lx * nb + ly) * nb + lz];
}
__device__ real fv8_raw(const real* prm, int o, long long D, int l, real px, real py, real pz, bool* ok) {
    int bx, by, bz;
    int bid = fv8_bid(prm, o, D, l, px, py, pz, &bx, &by, &bz);
    *ok = bid >= 0;
    if (bid < 0) return (real)0;
    real h = prm[o + 8 + l];
    real fx = (px - prm[o]) / h - bx * (real)8, fy = (py - prm[o + 1]) / h - by * (real)8, fz = (pz - prm[o + 2]) / h - bz * (real)8;
    int i = min(max((int)floor(fx), 0), 7), j = min(max((int)floor(fy), 0), 7), k = min(max((int)floor(fz), 0), 7);
    real ax = fx - i, ay = fy - j, az = fz - k;
    long long s = D + fv8_off(prm, o, 34 + 2 * l) + (long long)bid * 729 + (i * 9 + j) * 9 + k;
    real a0 = (real)FD[s] + ((real)FD[s + 81] - (real)FD[s]) * ax, a1 = (real)FD[s + 9] + ((real)FD[s + 90] - (real)FD[s + 9]) * ax;
    real a2 = (real)FD[s + 1] + ((real)FD[s + 82] - (real)FD[s + 1]) * ax, a3 = (real)FD[s + 10] + ((real)FD[s + 91] - (real)FD[s + 10]) * ax;
    real b0 = a0 + (a1 - a0) * ay, b1 = a2 + (a3 - a2) * ay;
    return b0 + (b1 - b0) * az;
}
// milestone: a sample's value at a coarser level k, addressed by its exact node index (level-l node gi -> level-k node
// gi / 2^(k - l), exact in float), not by a position rebuilt from float32 header values (a node on a brick face could
// fall into the neighbouring, missing brick: the card read -6 mm where the CPU read 0.09 mm)
__device__ real fv8_raw_node(const real* prm, int o, long long D, int k, real gx, real gy, real gz, bool* ok) {
    *ok = false;
    int nb = (int)prm[o + 16 + k];
    if (nb == 0) return (real)0;
    int nx = (int)prm[o + 4], ny = (int)prm[o + 5], nz = (int)prm[o + 6];
    real tw = (real)(8 * nb);
    int tx = (int)floor(gx / tw), ty = (int)floor(gy / tw), tz = (int)floor(gz / tw);
    if (tx < 0 || ty < 0 || tz < 0 || tx >= nx || ty >= ny || tz >= nz) return (real)0;
    int slot = (int)(real)FD[D + fv8_off(prm, o, 24) + ((long long)(tx * ny + ty) * nz + tz) * 4 + k];
    if (slot < 0) return (real)0;
    int bx = (int)floor(gx / (real)8), by = (int)floor(gy / (real)8), bz = (int)floor(gz / (real)8);
    int lx = bx - tx * nb, ly = by - ty * nb, lz = bz - tz * nb;
    int bid = (int)(real)FD[D + fv8_off(prm, o, 26 + 2 * k) + (long long)slot * nb * nb * nb + (lx * nb + ly) * nb + lz];
    if (bid < 0) return (real)0;
    real fx = gx - bx * (real)8, fy = gy - by * (real)8, fz = gz - bz * (real)8;
    int i = min(max((int)floor(fx), 0), 7), j = min(max((int)floor(fy), 0), 7), kk = min(max((int)floor(fz), 0), 7);
    real ax = fx - i, ay = fy - j, az = fz - kk;
    long long s = D + fv8_off(prm, o, 34 + 2 * k) + (long long)bid * 729 + (i * 9 + j) * 9 + kk;
    real a0 = (real)FD[s] + ((real)FD[s + 81] - (real)FD[s]) * ax, a1 = (real)FD[s + 9] + ((real)FD[s + 90] - (real)FD[s + 9]) * ax;
    real a2 = (real)FD[s + 1] + ((real)FD[s + 82] - (real)FD[s + 1]) * ax, a3 = (real)FD[s + 10] + ((real)FD[s + 91] - (real)FD[s + 10]) * ax;
    real b0 = a0 + (a1 - a0) * ay, b1 = a2 + (a3 - a2) * ay;
    *ok = true;
    return b0 + (b1 - b0) * az;
}
__device__ real fv8_coarser_node(const real* prm, int o, long long D, int l, int gix, int giy, int giz, real v) {
    int last = -1;
    for (int k = 0; k < 4; k++) if ((int)prm[o + 16 + k] > 0) last = k;
    for (int k = l + 1; k < 4; k++) {
        if ((int)prm[o + 16 + k] == 0) continue;
        real sc = (real)1 / (real)(1 << (k - l));
        bool ok;
        real w = fv8_raw_node(prm, o, D, k, gix * sc, giy * sc, giz * sc, &ok);
        if (!ok) continue;
        if (k != last && fabs(w) >= (real)0.999 * prm[o + 42 + k]) continue;
        return w;
    }
    return v;
}
__device__ real fv8_coarser(const real* prm, int o, long long D, int l, real x, real y, real z, real v) {
    int last = -1;
    for (int k = 0; k < 4; k++) if ((int)prm[o + 16 + k] > 0) last = k;
    for (int k = l + 1; k < 4; k++) {
        if ((int)prm[o + 16 + k] == 0) continue;
        bool ok;
        real w = fv8_raw(prm, o, D, k, x, y, z, &ok);
        if (!ok) continue;
        if (k != last && fabs(w) >= (real)0.999 * prm[o + 42 + k]) continue;
        return w;
    }
    return v;
}
// 11-09 (seam): program.py _fv8_node / _fv8_seam, the same rule: a level-l node from any level-l brick holding it
// (faces included), with the cap rule; a point in a level-l cell no brick holds, next to one that does, is the
// trilinear of the cell's nodes (held: level l; else the coarser levels at the node) - her field continuous across L0's
// band edge (it stepped 130-177 um there; the card's node samples did not)
__device__ real fv8_node(const real* prm, int o, long long D, int l, int gx, int gy, int gz, bool* ok) {
    *ok = false;
    int nb = (int)prm[o + 16 + l];
    if (nb == 0) return (real)0;
    int nx = (int)prm[o + 4], ny = (int)prm[o + 5], nz = (int)prm[o + 6];
    int b0x = gx >> 3, b0y = gy >> 3, b0z = gz >> 3;
    for (int cx = 0; cx < 2; cx++) { if (cx && (gx & 7)) continue;
    for (int cy = 0; cy < 2; cy++) { if (cy && (gy & 7)) continue;
    for (int cz = 0; cz < 2; cz++) { if (cz && (gz & 7)) continue;
        int bx = b0x - cx, by = b0y - cy, bz = b0z - cz;
        if (bx < 0 || by < 0 || bz < 0) continue;
        int tx = bx / nb, ty = by / nb, tz = bz / nb;
        if (tx >= nx || ty >= ny || tz >= nz) continue;
        int slot = (int)(real)FD[D + fv8_off(prm, o, 24) + ((long long)(tx * ny + ty) * nz + tz) * 4 + l];
        if (slot < 0) continue;
        int bid = (int)(real)FD[D + fv8_off(prm, o, 26 + 2 * l) + (long long)slot * nb * nb * nb
                                + ((bx - tx * nb) * nb + (by - ty * nb)) * nb + (bz - tz * nb)];
        if (bid < 0) continue;
        real v = (real)FD[D + fv8_off(prm, o, 34 + 2 * l) + (long long)bid * 729 + ((gx - bx * 8) * 9 + (gy - by * 8)) * 9 + (gz - bz * 8)];
        if (fabs(v) >= (real)0.999 * prm[o + 42 + l]) v = fv8_coarser_node(prm, o, D, l, gx, gy, gz, v);
        *ok = true;
        return v;
    }}}
    return (real)0;
}
__device__ real fv8_below_node(const real* prm, int o, long long D, int l, int gx, int gy, int gz) {
    // 11-09: program.py _fv8_below_node
    bool have_c = false;
    real capped = 0;
    for (int k = l + 1; k < 4; k++) {
        if ((int)prm[o + 16 + k] == 0) continue;
        real sc = (real)1 / (real)(1 << (k - l));
        bool ok;
        real w = fv8_raw_node(prm, o, D, k, gx * sc, gy * sc, gz * sc, &ok);
        if (!ok) continue;
        if (fabs(w) >= (real)0.999 * prm[o + 42 + k]) { if (!have_c) { capped = w; have_c = true; } continue; }
        return w;
    }
    if (have_c) return capped;
    return prm[o + 20 + l] - prm[o + 8 + l];
}
__device__ real fv8_seam(const real* prm, int o, long long D, int l, real px, real py, real pz, bool* ok) {
    *ok = false;
    real h = prm[o + 8 + l];
    real fx = (px - prm[o]) / h, fy = (py - prm[o + 1]) / h, fz = (pz - prm[o + 2]) / h;
    int i0 = (int)floor(fx), j0 = (int)floor(fy), k0 = (int)floor(fz);
    if (i0 < 0 || j0 < 0 || k0 < 0) return (real)0;
    real ax = fx - i0, ay = fy - j0, az = fz - k0, acc = 0;
    int held = 0;
    for (int di = 0; di < 2; di++) for (int dj = 0; dj < 2; dj++) for (int dk = 0; dk < 2; dk++) {
        bool hk;
        real v = fv8_node(prm, o, D, l, i0 + di, j0 + dj, k0 + dk, &hk);
        if (hk) held++;
        else v = fv8_below_node(prm, o, D, l, i0 + di, j0 + dj, k0 + dk);
        acc += (di ? ax : 1 - ax) * (dj ? ay : 1 - ay) * (dk ? az : 1 - az) * v;
    }
    if (held == 0) return (real)0;
    *ok = true;
    return acc;
}
// 11-09: program.py _fv8_bound_node / _fv8_V (one value per finest-level node)
__device__ real fv8_bound_node(const real* prm, int o, long long D, int l0, int gx, int gy, int gz) {
    real best = prm[o + 7];
    int nx = (int)prm[o + 4], ny = (int)prm[o + 5], nz = (int)prm[o + 6];
    for (int k = l0; k < 4; k++) {
        int nb = (int)prm[o + 16 + k];
        if (nb == 0) continue;
        real sc = (real)1 / (real)(1 << (k - l0));
        real kx = gx * sc, ky = gy * sc, kz = gz * sc, tw = (real)(8 * nb);
        int tx = (int)floor(kx / tw), ty = (int)floor(ky / tw), tz = (int)floor(kz / tw);
        if (tx < 0 || ty < 0 || tz < 0 || tx >= nx || ty >= ny || tz >= nz) continue;
        int slot = (int)(real)FD[D + fv8_off(prm, o, 24) + ((long long)(tx * ny + ty) * nz + tz) * 4 + k];
        if (slot < 0) continue;
        int bx = (int)floor(kx / (real)8), by = (int)floor(ky / (real)8), bz = (int)floor(kz / (real)8);
        int bid = (int)(real)FD[D + fv8_off(prm, o, 26 + 2 * k) + (long long)slot * nb * nb * nb
                                + ((bx - tx * nb) * nb + (by - ty * nb)) * nb + (bz - tz * nb)];
        if (bid < 0) best = fmax(best, prm[o + 20 + k] - prm[o + 8 + k]);
    }
    return best;
}
__device__ real fv8_V(const real* prm, int o, long long D, int l0, int gx, int gy, int gz) {
    bool hk;
    real v = fv8_node(prm, o, D, l0, gx, gy, gz, &hk);
    if (hk) return v;
    bool have_c = false;
    real capped = 0;
    for (int k = l0 + 1; k < 4; k++) {
        if ((int)prm[o + 16 + k] == 0) continue;
        real sc = (real)1 / (real)(1 << (k - l0));
        bool ok;
        real w = fv8_raw_node(prm, o, D, k, gx * sc, gy * sc, gz * sc, &ok);
        if (!ok) continue;
        if (fabs(w) >= (real)0.999 * prm[o + 42 + k]) { if (!have_c) { capped = w; have_c = true; } continue; }
        return w;
    }
    if (have_c) return capped;
    return fv8_bound_node(prm, o, D, l0, gx, gy, gz);
}
__device__ real fv8_d(const real* prm, int o, real px, real py, real pz) {
    long long D = (long long)prm[o + 15];        // 11-09: the store's offset in FD (float32 on the card)
    real tm = prm[o + 3];
    int nx = (int)prm[o + 4], ny = (int)prm[o + 5], nz = (int)prm[o + 6];
    real gx = (px - prm[o]) / tm, gy = (py - prm[o + 1]) / tm, gz = (pz - prm[o + 2]) / tm;
    real lb = prm[o + 7];
    if (gx < 0 || gy < 0 || gz < 0 || gx >= nx || gy >= ny || gz >= nz) {
        real qx = fmax(fmax(-gx, gx - nx), (real)0) * tm, qy = fmax(fmax(-gy, gy - ny), (real)0) * tm;
        real qz = fmax(fmax(-gz, gz - nz), (real)0) * tm;
        return sqrt(qx * qx + qy * qy + qz * qz) + lb;
    }
    real best = lb;
    for (int l = 0; l < 4; l++) {
        if ((int)prm[o + 16 + l] == 0) continue;
        int bx, by, bz;
        int bid = fv8_bid(prm, o, D, l, px, py, pz, &bx, &by, &bz);
        if (bid < 0) {                                // 11-09: outside the finest level's bricks: the node field
            real h0 = prm[o + 8 + l];
            real fx = (px - prm[o]) / h0, fy = (py - prm[o + 1]) / h0, fz = (pz - prm[o + 2]) / h0;
            int i0 = (int)floor(fx), j0 = (int)floor(fy), k0 = (int)floor(fz);
            real ax = fx - i0, ay = fy - j0, az = fz - k0, acc = 0;
            for (int di = 0; di < 2; di++) for (int dj = 0; dj < 2; dj++) for (int dk = 0; dk < 2; dk++)
                acc += (di ? ax : 1 - ax) * (dj ? ay : 1 - ay) * (dk ? az : 1 - az) * fv8_V(prm, o, D, l, i0 + di, j0 + dj, k0 + dk);
            return acc;
        }
        if (bid == -2) continue;
        real h = prm[o + 8 + l];
        if (bid < 0) { best = fmax(best, prm[o + 20 + l] - h); continue; }
        real fx = (px - prm[o]) / h - bx * (real)8, fy = (py - prm[o + 1]) / h - by * (real)8, fz = (pz - prm[o + 2]) / h - bz * (real)8;
        int i = min(max((int)floor(fx), 0), 7), j = min(max((int)floor(fy), 0), 7), k = min(max((int)floor(fz), 0), 7);
        real ax = fx - i, ay = fy - j, az = fz - k;
        long long s = D + fv8_off(prm, o, 34 + 2 * l) + (long long)bid * 729 + (i * 9 + j) * 9 + k;
        real cap = (real)0.999 * prm[o + 42 + l], acc = 0;
        for (int di = 0; di < 2; di++) for (int dj = 0; dj < 2; dj++) for (int dk = 0; dk < 2; dk++) {
            real v;
            if (i + di == 8 || j + dj == 8 || k + dk == 8) {      // 11-09: one value per node (program.py)
                bool hk;
                v = fv8_node(prm, o, D, l, bx * 8 + i + di, by * 8 + j + dj, bz * 8 + k + dk, &hk);
            } else {
                v = (real)FD[s + di * 81 + dj * 9 + dk];
                if (fabs(v) >= cap)
                    v = fv8_coarser_node(prm, o, D, l, bx * 8 + i + di, by * 8 + j + dj, bz * 8 + k + dk, v);
            }
            acc += (di ? ax : 1 - ax) * (dj ? ay : 1 - ay) * (dk ? az : 1 - az) * v;
        }
        return acc;
    }
    return best;
}

__device__ real sampled_d(const real* prm, int o, real px, real py, real pz) {
    real cx = px - prm[o + 8], cy = py - prm[o + 9], cz = pz - prm[o + 10], cs;
    if (prm[o + 7] > (real)0.5) cs = sqrt(cx * cx + cy * cy + cz * cz) - prm[o + 11];
    else {
        real qx = fabs(cx) - prm[o + 11], qy = fabs(cy) - prm[o + 12], qz = fabs(cz) - prm[o + 13];
        real mx = fmax(qx, (real)0.0), my = fmax(qy, (real)0.0), mz = fmax(qz, (real)0.0);
        cs = sqrt(mx * mx + my * my + mz * mz) + fmin(fmax(qx, fmax(qy, qz)), (real)0.0);
    }
    real h = prm[o + 3];
    int nx = (int)prm[o + 4], ny = (int)prm[o + 5], nz = (int)prm[o + 6];
    real fx = (px - prm[o]) / h, fy = (py - prm[o + 1]) / h, fz = (pz - prm[o + 2]) / h;
    int i = (int)floor(fx), j = (int)floor(fy), k = (int)floor(fz);
    if (i < 0 || j < 0 || k < 0 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return cs > 0.0 ? cs + prm[o + 14] : cs;   // 11-09: + boundary min
    real tx = fx - i, ty = fy - j, tz = fz - k;
    int d = o + (int)prm[o + 15];
    int b = d + (i * ny + j) * nz + k, syz = ny * nz;
    real v000 = prm[b], v001 = prm[b + 1], v010 = prm[b + nz], v011 = prm[b + nz + 1];
    real v100 = prm[b + syz], v101 = prm[b + syz + 1], v110 = prm[b + syz + nz], v111 = prm[b + syz + nz + 1];
    real a = v000 + (v100 - v000) * tx, bb = v010 + (v110 - v010) * tx, c = v001 + (v101 - v001) * tx, dd = v011 + (v111 - v011) * tx;
    real e = a + (bb - a) * ty, f = c + (dd - c) * ty;
    return fmax(e + (f - e) * tz, cs);
}

// cutsdf (11-08): program.py _osweep, same math and layout (header 15: eye, u, v, m, focal, extent, n; n plane points
// (x, y); n unit rays). Exact signed distance to the outline swept from the eye, max'd with the extent cube SDF.
__device__ real osweep_d(const real* prm, int o, real px, real py, real pz) {
    real rx = px - prm[o], ry = py - prm[o + 1], rz = pz - prm[o + 2];
    int n = (int)prm[o + 14], pb = o + 15, ab = o + 15 + 2 * n;
    real rr = rx * rx + ry * ry + rz * rz, best = (real)1e30;
    for (int i = 0; i < n; i++) {
        int j = (i + 1 == n) ? 0 : i + 1;
        real ax = prm[ab + 3 * i], ay = prm[ab + 3 * i + 1], az = prm[ab + 3 * i + 2];
        real t = rx * ax + ry * ay + rz * az;
        // milestone: |r x A|^2, not |r|^2 - (r.A)^2 (that cancelled to 0 in float at 0.4 m from the eye: the card's
        // cutter read 0 where the CPU read 0.1 mm)
        real ex = ry * az - rz * ay, ey = rz * ax - rx * az, ez = rx * ay - ry * ax;
        real d2 = t <= (real)0.0 ? rr : ex * ex + ey * ey + ez * ez;
        if (d2 < best) best = d2;
        real bx = prm[ab + 3 * j], by = prm[ab + 3 * j + 1], bz = prm[ab + 3 * j + 2];
        real nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        real nn = nx * nx + ny * ny + nz * nz;
        if (nn > (real)1e-24) {
            real c1x = ay * rz - az * ry, c1y = az * rx - ax * rz, c1z = ax * ry - ay * rx;
            real c2x = ry * bz - rz * by, c2y = rz * bx - rx * bz, c2z = rx * by - ry * bx;
            if (c1x * nx + c1y * ny + c1z * nz >= (real)0.0 && c2x * nx + c2y * ny + c2z * nz >= (real)0.0) {
                real s = rx * nx + ry * ny + rz * nz;
                d2 = s * s / nn;
                if (d2 < best) best = d2;
            }
        }
    }
    real dist = sqrt(best);
    real z = rx * prm[o + 9] + ry * prm[o + 10] + rz * prm[o + 11];
    bool inside = false;
    if (z > (real)0.0) {
        real f = prm[o + 12];
        real X = f * (rx * prm[o + 3] + ry * prm[o + 4] + rz * prm[o + 5]) / z;
        real Y = f * (rx * prm[o + 6] + ry * prm[o + 7] + rz * prm[o + 8]) / z;
        for (int i = 0, k = n - 1; i < n; k = i++) {
            real xi = prm[pb + 2 * i], yi = prm[pb + 2 * i + 1], xk = prm[pb + 2 * k], yk = prm[pb + 2 * k + 1];
            if ((yi > Y) != (yk > Y) && X < (xk - xi) * (Y - yi) / (yk - yi) + xi) inside = !inside;
        }
    }
    real sd = inside ? -dist : dist;
    real e = prm[o + 13];
    real qx = fabs(px) - e, qy = fabs(py) - e, qz = fabs(pz) - e;
    real mx = fmax(qx, (real)0.0), my = fmax(qy, (real)0.0), mz = fmax(qz, (real)0.0);
    real b = sqrt(mx * mx + my * my + mz * mz) + fmin(fmax(qx, fmax(qy, qz)), (real)0.0);
    return fmax(sd, b);
}

__device__ real eval_prog_d(const int* code, const real* prm, int c0, int c1, real px, real py, real pz) {
    real vs[STK], ps[4 * STK];
    int sp = 0, pp = 0;
    for (int i = c0; i < c1; i++) {
        int op = code[3 * i], o = code[3 * i + 1];
        if (op == OP_SPHERE) vs[sp++] = sqrt(px * px + py * py + pz * pz) - prm[o];
        else if (op == OP_BOX) {
            real r = prm[o + 3];
            real qx = fabs(px) - prm[o] + r, qy = fabs(py) - prm[o + 1] + r, qz = fabs(pz) - prm[o + 2] + r;
            real mx = fmax(qx, (real)0.0), my = fmax(qy, (real)0.0), mz = fmax(qz, (real)0.0);
            vs[sp++] = sqrt(mx * mx + my * my + mz * mz) + fmin(fmax(qx, fmax(qy, qz)), (real)0.0) - r;
        } else if (op == OP_CYL) {
            real rr = prm[o + 2];
            real dx = sqrt(px * px + pz * pz) - prm[o] + rr, dy = fabs(py) - prm[o + 1] + rr;
            real mx = fmax(dx, (real)0.0), my = fmax(dy, (real)0.0);
            vs[sp++] = fmin(fmax(dx, dy), (real)0.0) + sqrt(mx * mx + my * my) - rr;
        } else if (op == OP_CAPSULE) {
            real hh = prm[o];
            real qy = py - fmin(fmax(py, -hh), hh);
            vs[sp++] = sqrt(px * px + qy * qy + pz * pz) - prm[o + 1];
        } else if (op == OP_CONE) {
            real r1 = prm[o], r2 = prm[o + 1], h = prm[o + 2];
            real qx = sqrt(px * px + pz * pz), qy = py, k2x = r2 - r1, k2y = (real)2.0 * h;
            real rr = qy < (real)0.0 ? r1 : r2;
            real cax = qx - fmin(qx, rr), cay = fabs(qy) - h;
            real t = ((r2 - qx) * k2x + (h - qy) * k2y) / (k2x * k2x + k2y * k2y);
            t = fmin(fmax(t, (real)0.0), (real)1.0);
            real cbx = qx - r2 + k2x * t, cby = qy - h + k2y * t;
            real s = (cbx < (real)0.0 && cay < (real)0.0) ? -(real)1.0 : (real)1.0;
            vs[sp++] = s * sqrt(fmin(cax * cax + cay * cay, cbx * cbx + cby * cby));
        } else if (op == OP_TORUS) {
            real qx = sqrt(px * px + pz * pz) - prm[o];
            vs[sp++] = sqrt(qx * qx + py * py) - prm[o + 1];
        } else if (op == OP_ELLIPSOID) {
            real rx = prm[o], ry = prm[o + 1], rz = prm[o + 2], rmin = fmin(rx, fmin(ry, rz));
            real ex = px / rx, ey = py / ry, ez = pz / rz;
            vs[sp++] = (sqrt(ex * ex + ey * ey + ez * ez) - (real)1.0) * rmin;
        } else if (op == OP_SLAB) {
            real d = fabs(px * prm[o] + py * prm[o + 1] + pz * prm[o + 2] - prm[o + 3]) - (real)0.5 * prm[o + 4];
            real e = prm[o + 5];
            real qx = fabs(px) - e, qy = fabs(py) - e, qz = fabs(pz) - e;
            real mx = fmax(qx, (real)0.0), my = fmax(qy, (real)0.0), mz = fmax(qz, (real)0.0);
            real b = sqrt(mx * mx + my * my + mz * mz) + fmin(fmax(qx, fmax(qy, qz)), (real)0.0);
            vs[sp++] = fmax(d, b);
        } else if (op == OP_SAMPLED) vs[sp++] = sampled_d(prm, o, px, py, pz);
        else if (op == OP_OSWEEP) vs[sp++] = osweep_d(prm, o, px, py, pz);
        else if (op == OP_FV8) vs[sp++] = fv8_d(prm, o, px, py, pz);
        else if (op >= OP_UNION && op <= OP_SINTER) {
            sp -= 1;
            real b = vs[sp], a = vs[sp - 1], r;
            if (op == OP_UNION) r = fmin(a, b);
            else if (op == OP_SUB) r = fmax(a, -b);
            else if (op == OP_INTER) r = fmax(a, b);
            else if (op == OP_SUNION) r = smin_d(a, b, prm[o]);
            else if (op == OP_SSUB) r = -smin_d(-a, b, prm[o]);
            else r = -smin_d(-a, -b, prm[o]);
            vs[sp - 1] = r;
        } else if (op == OP_XFORM) {
            ps[4 * pp] = px; ps[4 * pp + 1] = py; ps[4 * pp + 2] = pz; ps[4 * pp + 3] = prm[o + 12];
            pp++;
            real dx = px - prm[o + 9], dy = py - prm[o + 10], dz = pz - prm[o + 11];
            px = prm[o] * dx + prm[o + 1] * dy + prm[o + 2] * dz;
            py = prm[o + 3] * dx + prm[o + 4] * dy + prm[o + 5] * dz;
            pz = prm[o + 6] * dx + prm[o + 7] * dy + prm[o + 8] * dz;
        } else if (op == OP_TWIST || op == OP_BEND || op == OP_TAPER) {
            ps[4 * pp] = px; ps[4 * pp + 1] = py; ps[4 * pp + 2] = pz; ps[4 * pp + 3] = (real)1.0;
            pp++;
            real k = prm[o];
            if (op == OP_TWIST) { real a = k * py, c = cos(a), s = sin(a), nx = c * px + s * pz; pz = -s * px + c * pz; px = nx; }
            else if (op == OP_BEND) { real a = k * px, c = cos(a), s = sin(a), nx = c * px - s * py; py = s * px + c * py; px = nx; }
            else { real g = fmax((real)1.0 + k * py, prm[o + 1]); px = px / g; pz = pz / g; }
        } else if (op == OP_POP) {
            pp -= 1;
            px = ps[4 * pp]; py = ps[4 * pp + 1]; pz = ps[4 * pp + 2];
            vs[sp - 1] *= ps[4 * pp + 3];
        } else if (op == OP_SHELL) vs[sp - 1] = fabs(vs[sp - 1]) - prm[o];
        else if (op == OP_OFFSET) vs[sp - 1] = vs[sp - 1] - prm[o];
        else if (op == OP_MATGEN) vs[sp - 1] = mg_apply(prm, o, vs[sp - 1], px, py, pz);   // fs-materials 11-14
        // OP_GUARD / OP_GUARD_U: never skip (tau = 1e9: the true field)
    }
    return vs[0];
}

// One thread per node of n bricks: rows[b] <- base(node) + sculpt delta(node). origins: (n, 3) real local brick
// origins; dsl: (n, 8) indices into dblk of the sculpt bricks (sx, sy, sz) + {0,1}^3 that the 9^3 nodes fall in (-1: 0)
// r34: node spacing = stride sculpt nodes (1: the sculpt lattice; 4: a coarse level whose nodes are every 4th sculpt
// node). dsl: (n, (stride+1)^3) indices into dblk of the sculpt bricks the 9^3 nodes fall in (-1: no sculpt there).
extern "C" __global__ void eval_bricks(const int* code, const real* prm, int c0, int c1, const double* origins, double h,
                                       const int* dsl, const float* dblk, const int* rows, int n, __half* sdf, float* out,
                                       int stride) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)n * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    double hs = h * stride;
    real x = (real)(origins[3 * b] + i * hs), y = (real)(origins[3 * b + 1] + j * hs), z = (real)(origins[3 * b + 2] + k * hs);
    real f = eval_prog_d(code, prm, c0, c1, x, y, z);
    int gi = i * stride, gj = j * stride, gk = k * stride, m = stride + 1;
    int s = dsl[(size_t)m * m * m * b + ((gi >> 3) * m + (gj >> 3)) * m + (gk >> 3)];
    if (s >= 0) f += (real)dblk[(size_t)s * 512 + ((gi & 7) * 8 + (gj & 7)) * 8 + (gk & 7)];
    if (rows) sdf[(size_t)rows[b] * 729 + q] = __float2half_rn((float)f);
    if (out) out[id] = (float)f;
}

// 11-09 (the release re-distance on the card): eval_bricks at a brick's own 8^3 sculpt nodes only (stride 1; the 9th
// layer is the next brick's first): the same value per node, 512 rather than 729 evaluations per brick
extern "C" __global__ void eval_nodes8(const int* code, const real* prm, int c0, int c1, const double* origins, double h,
                                       const int* dsl, const float* dblk, int n, float* out) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)n * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    real x = (real)(origins[3 * b] + i * h), y = (real)(origins[3 * b + 1] + j * h), z = (real)(origins[3 * b + 2] + k * h);
    real f = eval_prog_d(code, prm, c0, c1, x, y, z);
    int s = dsl[8 * b];
    if (s >= 0) f += (real)dblk[(size_t)s * 512 + q];
    out[id] = (float)f;
}
