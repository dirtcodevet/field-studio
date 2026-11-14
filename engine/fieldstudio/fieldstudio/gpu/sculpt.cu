// Field Studio GPU sculpt (phase 0 / integration option A, 2026-11-08): brush dabs applied on the card's brick atlas.
// The rules are core/brush.py's brush_kernel, node for node (same falloff LUT, same A, same per-tool field rule); the
// field read at displaced points is the object's own atlas field (trilinear inside its bricks, as the viewport
// draws it) instead of the CPU's base program + trilinear delta (they differ by the base's interpolation error,
// ~h^2 kappa / 8: 1.25 um on a 100 mm sphere at h = 1 mm). Masks: not read here (strokes with a mask in use stay on
// the CPU path). Every dab: kernel 1 computes the new values of the dab's bricks into scratch (reads only the pool),
// kernel 2 writes them back, so a dab never reads its own writes.
#include <cuda_fp16.h>
__device__ __forceinline__ float3 mk3(float x, float y, float z) { return make_float3(x, y, z); }

#define NLEV 4
#define VF 48
#define VF_M 0
#define VF_S 24
#define VF_O 25
#define VF_TM 28
#define VF_H 29
#define VF_BAND 33
#define VI 20
#define VI_GOFF 0
#define VI_NX 1
#define VI_NY 2
#define VI_NZ 3
#define VI_TOFF 4
#define VI_BPT 8
#define VI_BOFF 12

// the object's field at local point (x, y, z), level slot `lev` only (one level per Field Studio object);
// outside every brick: 1e30 (r17: no information; fast-math may drop NaN tests; the dab leaves such a node as it was)
__device__ float obj_f(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                       float x, float y, float z, float outside) {
    float tm = vf[VF_TM];
    float qx = (x - vf[VF_O]) / tm, qy = (y - vf[VF_O + 1]) / tm, qz = (z - vf[VF_O + 2]) / tm;
    int tx = (int)floorf(qx), ty = (int)floorf(qy), tz = (int)floorf(qz);
    if (tx < 0 || ty < 0 || tz < 0 || tx >= vi[VI_NX] || ty >= vi[VI_NY] || tz >= vi[VI_NZ]) return 1e30f;
    int slot = grid[vi[VI_GOFF] + ((tx * vi[VI_NY] + ty) * vi[VI_NZ] + tz) * NLEV + lev];
    if (slot < 0) return 1e30f;
    int bpt = vi[VI_BPT + lev];
    float h = vf[VF_H + lev], bm = 8.0f * h;
    float lx = (qx - tx) * tm / bm, ly = (qy - ty) * tm / bm, lz = (qz - tz) * tm / bm;
    int bi = min(max((int)floorf(lx), 0), bpt - 1), bj = min(max((int)floorf(ly), 0), bpt - 1), bk = min(max((int)floorf(lz), 0), bpt - 1);
    int bid = ttab[vi[VI_TOFF + lev] + slot * bpt * bpt * bpt + (bi * bpt + bj) * bpt + bk];
    if (bid < 0) return 1e30f;
    const __half* B = sdf + (size_t)(bid + vi[VI_BOFF + lev]) * 729;
    float ux = (lx - bi) * 8.0f, uy = (ly - bj) * 8.0f, uz = (lz - bk) * 8.0f;
    int i = min(max((int)floorf(ux), 0), 7), j = min(max((int)floorf(uy), 0), 7), k = min(max((int)floorf(uz), 0), 7);
    float fx = ux - i, fy = uy - j, fz = uz - k;
    const __half* c = B + (i * 9 + j) * 9 + k;
    float v000 = __half2float(c[0]), v001 = __half2float(c[1]), v010 = __half2float(c[9]), v011 = __half2float(c[10]);
    float v100 = __half2float(c[81]), v101 = __half2float(c[82]), v110 = __half2float(c[90]), v111 = __half2float(c[91]);
    float a0 = v000 + (v100 - v000) * fx, a1 = v010 + (v110 - v010) * fx, a2 = v001 + (v101 - v001) * fx, a3 = v011 + (v111 - v011) * fx;
    float b0 = a0 + (a1 - a0) * fy, b1 = a2 + (a3 - a2) * fy;
    return b0 + (b1 - b0) * fz;
}

// fs-tools 11-14 (patch 3): the object's field AS IT WAS AT THE PRESS: obj_f's lookup, with a row the stroke has touched
// read from its first-touch values (bslot: atlas row -> pool row, -1 = untouched, so the atlas still holds its start).
// Off the bricks it returns 1e30 as obj_f does; ft_rule2 (toolset/ftools.cuh FT_FAR) takes any read above 1e29 as no
// information and keeps the node's value, so neither reader's off-brick value ever enters a rule
__device__ float obj_f_start(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                             float x, float y, float z, const int* bslot, int nbslot, const __half* bpool) {
    float tm = vf[VF_TM];
    float qx = (x - vf[VF_O]) / tm, qy = (y - vf[VF_O + 1]) / tm, qz = (z - vf[VF_O + 2]) / tm;
    int tx = (int)floorf(qx), ty = (int)floorf(qy), tz = (int)floorf(qz);
    if (tx < 0 || ty < 0 || tz < 0 || tx >= vi[VI_NX] || ty >= vi[VI_NY] || tz >= vi[VI_NZ]) return 1e30f;
    int slot = grid[vi[VI_GOFF] + ((tx * vi[VI_NY] + ty) * vi[VI_NZ] + tz) * NLEV + lev];
    if (slot < 0) return 1e30f;
    int bpt = vi[VI_BPT + lev];
    float h = vf[VF_H + lev], bm = 8.0f * h;
    float lx = (qx - tx) * tm / bm, ly = (qy - ty) * tm / bm, lz = (qz - tz) * tm / bm;
    int bi = min(max((int)floorf(lx), 0), bpt - 1), bj = min(max((int)floorf(ly), 0), bpt - 1), bk = min(max((int)floorf(lz), 0), bpt - 1);
    int bid = ttab[vi[VI_TOFF + lev] + slot * bpt * bpt * bpt + (bi * bpt + bj) * bpt + bk];
    if (bid < 0) return 1e30f;
    int row = bid + vi[VI_BOFF + lev];
    int ps = row < nbslot ? bslot[row] : -1;
    const __half* B = ps >= 0 ? bpool + (size_t)ps * 729 : sdf + (size_t)row * 729;
    float ux = (lx - bi) * 8.0f, uy = (ly - bj) * 8.0f, uz = (lz - bk) * 8.0f;
    int i = min(max((int)floorf(ux), 0), 7), j = min(max((int)floorf(uy), 0), 7), k = min(max((int)floorf(uz), 0), 7);
    float fx = ux - i, fy = uy - j, fz = uz - k;
    const __half* c = B + (i * 9 + j) * 9 + k;
    float v000 = __half2float(c[0]), v001 = __half2float(c[1]), v010 = __half2float(c[9]), v011 = __half2float(c[10]);
    float v100 = __half2float(c[81]), v101 = __half2float(c[82]), v110 = __half2float(c[90]), v111 = __half2float(c[91]);
    float a0 = v000 + (v100 - v000) * fx, a1 = v010 + (v110 - v010) * fx, a2 = v001 + (v101 - v001) * fx, a3 = v011 + (v111 - v011) * fx;
    float b0 = a0 + (a1 - a0) * fy, b1 = a2 + (a3 - a2) * fy;
    return b0 + (b1 - b0) * fz;
}

__device__ __forceinline__ float lut_w(const float* lut, int n, float u) {
    float x = u * (n - 1);
    int i = (int)x;
    if (i >= n - 1) return lut[n - 1];
    float t = x - i;
    return lut[i] + (lut[i + 1] - lut[i]) * t;
}

// brush targeting (11-08 night): core/brush.py target_gate, node for node, on the atlas field (1e30 reads skipped)
__device__ __forceinline__ bool obj_grad(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                                         int lev, float x, float y, float z, float e, float* g) {
    float a = obj_f(vf, vi, grid, ttab, sdf, lev, x + e, y, z, 0.f), b = obj_f(vf, vi, grid, ttab, sdf, lev, x - e, y, z, 0.f);
    float c = obj_f(vf, vi, grid, ttab, sdf, lev, x, y + e, z, 0.f), d = obj_f(vf, vi, grid, ttab, sdf, lev, x, y - e, z, 0.f);
    float p = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z + e, 0.f), q = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z - e, 0.f);
    if (fabsf(a) > 1e29f || fabsf(b) > 1e29f || fabsf(c) > 1e29f || fabsf(d) > 1e29f || fabsf(p) > 1e29f || fabsf(q) > 1e29f) return false;
    float gx = a - b, gy = c - d, gz = p - q;
    float gl = sqrtf(gx * gx + gy * gy + gz * gz);
    if (gl < 1e-12f) return false;            // the direction decides, whatever the size (as the CPU)
    g[0] = gx / gl; g[1] = gy / gl; g[2] = gz / gl;
    return true;
}

__device__ float target_gate(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                             float x, float y, float z, float f0, const float* bp) {
    int mode = (int)bp[15];
    float hs = bp[18];
    float g[3];
    if (!obj_grad(vf, vi, grid, ttab, sdf, lev, x, y, z, hs, g)) return 1.0f;
    float fw = 1.0f;
    if (mode & 1) {
        float cs = g[0] * bp[4] + g[1] * bp[5] + g[2] * bp[6];
        float lo = bp[16], hi = bp[17];
        if (cs <= lo) return 0.0f;
        if (cs < hi) { float t = (cs - lo) / (hi - lo); fw = t * t * (3.0f - 2.0f * t); }
    }
    if ((mode & 2) && fabsf(f0) <= bp[19]) {
        float cx = bp[1], cy = bp[2], cz = bp[3];
        float dx = x - f0 * g[0] - cx, dy = y - f0 * g[1] - cy, dz = z - f0 * g[2] - cz;
        float L = sqrtf(dx * dx + dy * dy + dz * dz);
        if (L > hs) {
            int K = min(12, (int)ceilf(L / (2.0f * hs)));   // sheet-scale samples (as the CPU)
            float a[3];
            float es = 2.0f * hs;                        // sheet-scale chord continuity (as the CPU)
            bool ha = obj_grad(vf, vi, grid, ttab, sdf, lev, cx, cy, cz, es, a);
            for (int i = 1; i <= K; i++) {
                float t = (float)i / K, b[3];
                if (!obj_grad(vf, vi, grid, ttab, sdf, lev, cx + dx * t, cy + dy * t, cz + dz * t, es, b)) continue;
                if (ha && a[0] * b[0] + a[1] * b[1] + a[2] * b[2] < -0.6f) return 0.0f;   // FOLD_DOT (brush.py)
                a[0] = b[0]; a[1] = b[1]; a[2] = b[2]; ha = true;
            }
        }
    }
    return fw;
}

// 11-09 night: Smooth keeps to its own sheet (core/brush.py _sheet_avg, node for node): each of the six samples walks
// out in steps of dl / K; where the field turns back (a medial ridge in a gap, the middle of a thin flap) the sample is
// the node's own sheet carried on linearly. A node on such a ridge, or a 1e30 read, returns 1e30: the node keeps its value.
#define RIDGE_TOL 0.3f
__device__ float sheet_avg(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                           float x, float y, float z, float f0, float dl, float h, float outside) {
    int K = max(1, (int)ceilf(dl / h - 1e-6f));
    float st = dl / K, tol = RIDGE_TOL * st, acc = 0.0f;
    for (int ax = 0; ax < 3; ax++) {
        float ex = ax == 0 ? 1.f : 0.f, ey = ax == 1 ? 1.f : 0.f, ez = ax == 2 ? 1.f : 0.f;
        float pp = obj_f(vf, vi, grid, ttab, sdf, lev, x + st * ex, y + st * ey, z + st * ez, outside);
        float pm = obj_f(vf, vi, grid, ttab, sdf, lev, x - st * ex, y - st * ey, z - st * ez, outside);
        if (fabsf(pp) > 1e29f || fabsf(pm) > 1e29f) return 1e30f;
        float a1 = pp - f0, b1 = pm - f0;
        if (a1 * b1 > 0.0f && fminf(fabsf(a1), fabsf(b1)) > tol) return 1e30f;
        float g = (pp - pm) / (2.0f * st);
        for (int k = 0; k < 2; k++) {
            float sg = k == 0 ? 1.f : -1.f;
            float v = k == 0 ? pp : pm, d1 = v - f0;
            bool crossed = false;
            for (int t = 2; t <= K; t++) {
                float vn = obj_f(vf, vi, grid, ttab, sdf, lev, x + sg * t * st * ex, y + sg * t * st * ey, z + sg * t * st * ez, outside);
                if (fabsf(vn) > 1e29f) return 1e30f;
                float d = vn - v;
                if (d * d1 < 0.0f && fabsf(d) > tol) { crossed = true; break; }
                v = vn;
            }
            acc += crossed ? (f0 + sg * dl * g) : v;
        }
    }
    return acc / 6.0f;
}

// fs-brush 11-11 (design C1, Smooth did not smooth: 2 h averaging moved a 30 mm bump by s (2h/L)^2 ~ 0.4 % a
// dab): Smooth at the brush's scale on the node's own sheet. A node near the surface (|F| < bp[24] h) projects to the
// surface (y = x - F grad F / |grad F|^2), samples F on two rings of the tangent plane at y (radii bp[25] r and bp[26]
// r, 8 samples each), and moves toward the quadric fitted through its surroundings: a = (r2^2 m1 - r1^2 m2) / (r2^2 -
// r1^2) (m: a ring's mean F; a sphere or a cylinder gives a = 0, so the form's own curvature stays, and only the part
// of the surface that departs from it at the brush's scale, the bumps, is removed). A sample belongs to the node's sheet
// only where F still rises along the node's normal there (else a facing sheet or a ridge sets its value: dropped), so
// a facing surface is never pulled. The change per dab is capped at bp[27] h; the live re-distance then rebuilds the
// distances around the moved surface.
#ifndef QSM_E
#define QSM_E 2.0f
#endif
// fs-brush-3 11-13 (grain: Smooth r 8 broke her lower lid's margin up; card A/B grain/sd_sm_*; CPU model
// grain/smooth_fit_sim.py): the brush-scale fit treats the surface as a height field over its rings; where a ring's heights
// reach past their clamp (0.5 rr) the surface turns away within the ring (a margin, a rim) and the extrapolation
// (4 m0 - m1) / 3 turns the clamped heights into node-to-node jumps. The fit fades out there:
// 1 - smoothstep((max |height| / clamp - QSM_HV0) / (QSM_HV1 - QSM_HV0)); a node it leaves keeps the 2 h term (clean on
// the margin). QSM_HV1 <= QSM_HV0: off. Set from gpu/sculpt.py (FSB_QSM_HV, default below)
#ifndef QSM_HV0
#define QSM_HV0 0.0f
#endif
#ifndef QSM_HV1
#define QSM_HV1 0.0f
#endif
// fs-brush 11-11 (M, cpu_pits.py on the CPU path, the same rule: a Smooth pass over the fold under a Grab-pulled lump left
// 1-2 mm pits; with the fit off it stayed clean; the cell-scale term made no difference): the fit with every step
// continuous in the node's position (core/brush.py _quad_fit_robust, the same rule), bp[40] > 1.5:
// - each ring sample's HEIGHT found along the normal (one Newton step from the plane point: F there alone is the distance
//   to whatever surface is nearest, in a fold the lump the plane cuts into), clamped to half the ring radius;
// - the same-sheet test taken at that surface point, as a weight (0 below a rise of 0.1 h, 1 above 0.5 h);
// - opposite samples weighted as pairs (the weaker one's weight), so a lost sample never leaks the surface's tilt;
// - the fit's amplitude fades in with the weaker ring's total weight (3 -> 5) and with |grad F| (0.2 -> 0.4).
// M (CPU): no pits, streaks or terraces in the fold; a bump of width W = r loses 88 / 51 / 52 % at W 4 / 12 / 40 mm
// (the plane fit 89 / 52 / 54 %).
__device__ float quad_fit_robust(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                                 float x, float y, float z, float f0, float h, float r, const float* bp,
                                 float gx, float gy, float gz, float gl, bool* ok) {
    *ok = false;
    float gam = fminf(fmaxf((gl - 0.2f) / 0.2f, 0.f), 1.f);
    if (gam <= 0.f) return 0.f;
    float nx = gx / gl, ny = gy / gl, nz = gz / gl;
    float d = f0 / gl;
    float px = x - d * nx, py = y - d * ny, pz = z - d * nz;
    float ax = fabsf(nx) < 0.9f ? 1.f : 0.f, ay = ax > 0.5f ? 0.f : 1.f;
    float t1x = -nz * ay, t1y = nz * ax, t1z = nx * ay - ny * ax;
    float tl = sqrtf(t1x * t1x + t1y * t1y + t1z * t1z);
    t1x /= tl; t1y /= tl; t1z /= tl;
    float t2x = ny * t1z - nz * t1y, t2y = nz * t1x - nx * t1z, t2z = nx * t1y - ny * t1x;
    float m[2], W[2], hrmax = 0.f;
    for (int ring = 0; ring < 2; ring++) {
        float rr = bp[25 + ring] * r, hc = 0.5f * rr;
        float HG[8], WG[8];
        for (int k = 0; k < 8; k++) {
            HG[k] = 0.f; WG[k] = 0.f;
            float th = (k + 0.5f * ring) * 0.78539816f;
            float c = cosf(th) * rr, sn = sinf(th) * rr;
            float sx = px + c * t1x + sn * t2x, sy = py + c * t1y + sn * t2y, sz = pz + c * t1z + sn * t2z;
            float fs = obj_f(vf, vi, grid, ttab, sdf, lev, sx, sy, sz, 0.f);
            if (fabsf(fs) > 1e29f) continue;
            float t0 = -fs / gl;
            float qx = sx + t0 * nx, qy = sy + t0 * ny, qz = sz + t0 * nz;
            float fq = obj_f(vf, vi, grid, ttab, sdf, lev, qx, qy, qz, 0.f);
            if (fabsf(fq) > 1e29f) continue;
            float hgt = t0 - fq / gl;
            float fn = obj_f(vf, vi, grid, ttab, sdf, lev, qx + h * nx, qy + h * ny, qz + h * nz, 0.f);
            if (fabsf(fn) > 1e29f) continue;
            HG[k] = -fminf(fmaxf(hgt, -hc), hc);
            hrmax = fmaxf(hrmax, fabsf(hgt) / fmaxf(hc, 1e-12f));
            // fs-brush 11-11 ~22:00 (bp[40] > 2.5, 'robust2', new strokes): only a sheet facing the other way is another sheet; a
            // steep sample on the same sheet keeps its weight (the 0.1..0.5 ramp gave 17-26 % of a curved jaw's samples partial
            // weights that switched the pairs' min from node to node: test1's granular jaw). 'robust' (older logs): as it was.
            WG[k] = (bp[40] > 2.5f) ? fminf(fmaxf(((fn - fq) / (h * gl) + 0.1f) / 0.3f, 0.f), 1.f)
                                    : fminf(fmaxf(((fn - fq) / (h * gl) - 0.1f) / 0.4f, 0.f), 1.f);
        }
        float acc = 0.f, wt = 0.f;
        for (int k = 0; k < 4; k++) {
            float wp = fminf(WG[k], WG[k + 4]);
            acc += wp * (HG[k] + HG[k + 4]);
            wt += 2.f * wp;
        }
        m[ring] = wt > 1e-9f ? acc / wt : 0.f;
        W[ring] = wt;
    }
    gam *= fminf(fmaxf((fminf(W[0], W[1]) - 3.f) / 2.f, 0.f), 1.f);
    if (QSM_HV1 > QSM_HV0) {
        float t = fminf(fmaxf((hrmax - QSM_HV0) / (QSM_HV1 - QSM_HV0), 0.f), 1.f);
        gam *= 1.f - t * t * (3.f - 2.f * t);
    }
    if (gam <= 0.f) return 0.f;
    float r1 = bp[25] * r, r2 = bp[26] * r;
    r1 *= r1; r2 *= r2;
    *ok = true;
    return gam * (r2 * m[0] - r1 * m[1]) / (r2 - r1) * gl;
}

__device__ float quad_smooth(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                             float x, float y, float z, float f0, float h, float r, const float* bp, bool* ok) {
    *ok = false;
    float e = (bp[39] > 0.f ? bp[39] : QSM_E) * h;     // fs-brush 11-11: the normal from a 2 h stencil (1 h jittered node to node: speckle, M t1 shots)
    float fx1 = obj_f(vf, vi, grid, ttab, sdf, lev, x + e, y, z, 0.f), fx0 = obj_f(vf, vi, grid, ttab, sdf, lev, x - e, y, z, 0.f);
    float fy1 = obj_f(vf, vi, grid, ttab, sdf, lev, x, y + e, z, 0.f), fy0 = obj_f(vf, vi, grid, ttab, sdf, lev, x, y - e, z, 0.f);
    float fz1 = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z + e, 0.f), fz0 = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z - e, 0.f);
    if (fabsf(fx1) > 1e29f || fabsf(fx0) > 1e29f || fabsf(fy1) > 1e29f || fabsf(fy0) > 1e29f || fabsf(fz1) > 1e29f || fabsf(fz0) > 1e29f) return 0.f;
    float gx = (fx1 - fx0) / (2.f * e), gy = (fy1 - fy0) / (2.f * e), gz = (fz1 - fz0) / (2.f * e);
    float gl = sqrtf(gx * gx + gy * gy + gz * gz);
    if (bp[40] > 1.5f) return quad_fit_robust(vf, vi, grid, ttab, sdf, lev, x, y, z, f0, h, r, bp, gx, gy, gz, gl, ok);
    if (gl < 0.3f) return 0.f;                    // a ridge between sheets or a flat medial region: left as it is
    float nx = gx / gl, ny = gy / gl, nz = gz / gl;
    float d = f0 / gl;
    float px = x - d * nx, py = y - d * ny, pz = z - d * nz;     // the surface point under the node
    float ax = fabsf(nx) < 0.9f ? 1.f : 0.f, ay = ax > 0.5f ? 0.f : 1.f;
    float t1x = ny * 0.f - nz * ay, t1y = nz * ax - nx * 0.f, t1z = nx * ay - ny * ax;
    float tl = sqrtf(t1x * t1x + t1y * t1y + t1z * t1z);
    t1x /= tl; t1y /= tl; t1z /= tl;
    float t2x = ny * t1z - nz * t1y, t2y = nz * t1x - nx * t1z, t2z = nx * t1y - ny * t1x;
    float rr[2] = {bp[25] * r, bp[26] * r}, m[2];
    int cnt[2];
    for (int ring = 0; ring < 2; ring++) {
        float acc = 0.f; int n = 0;
        for (int k = 0; k < 8; k++) {
            float th = (k + 0.5f * ring) * 0.78539816f;
            float c = cosf(th) * rr[ring], sn = sinf(th) * rr[ring];
            float sx = px + c * t1x + sn * t2x, sy = py + c * t1y + sn * t2y, sz = pz + c * t1z + sn * t2z;
            float fs = obj_f(vf, vi, grid, ttab, sdf, lev, sx, sy, sz, 0.f);
            if (fabsf(fs) > 1e29f) continue;
            float fn = obj_f(vf, vi, grid, ttab, sdf, lev, sx + h * nx, sy + h * ny, sz + h * nz, 0.f);
            if (fabsf(fn) > 1e29f) continue;
            if (fn - fs < 0.3f * h * gl) continue;            // F does not rise along this sheet's normal: another sheet
            acc += fs / gl; n++;
        }
        cnt[ring] = n; m[ring] = n ? acc / n : 0.f;
    }
    if (cnt[0] < 5 || cnt[1] < 5) return 0.f;
    float r1 = rr[0] * rr[0], r2 = rr[1] * rr[1];
    float a = (r2 * m[0] - r1 * m[1]) / (r2 - r1);              // distance units: + moves the surface inward
    *ok = true;
    return a * gl;                                               // back in F units at the node
}

// bp: [0] tool (brush.T), [1..3] c, [4..6] n, [7] r, [8] s, [9..11] v, [12] dl (smooth), [13] A, [14] band clamp
// rows (nb): pool rows; org (nb, 3): brick origins (local); out (nb * 729): new values
// fs-brush 11-11: mpool (rows x 729, fp16) holds each row's mask (already inverted under Mask > Invert) when use_mask;
// a dab moves a node by (1 - m) of its change, as core/brush.py brush_kernel (M, fs-trust R9: with any mask in the scene
// every stroke fell to the CPU path, 0-2 fps on a test project)
// fs-brush 11-12: Grab's scope (core/brush.py _scope_at / _scope_dist): the weight of the pressed part's side, trilinear
// in the stroke-start grid mg (255 = its side; bp[23..25] origin, bp[26] spacing, bp[27] > 0.5 on), and the falloff
// distance with the pressed part's thickness bp[38] taken out of the depth along the press normal (bp 4-6)
__device__ __forceinline__ float scope_at(float x, float y, float z, const float* bp, const unsigned char* mg,
                                          int nx, int ny, int nz) {
    if (bp[27] < 0.5f) return 1.0f;
    float sp = bp[26];
    float fx = (x - bp[23]) / sp, fy = (y - bp[24]) / sp, fz = (z - bp[25]) / sp;
    if (fx < 0.f || fy < 0.f || fz < 0.f || fx > nx - 1.f || fy > ny - 1.f || fz > nz - 1.f) return 1.0f;
    int i = min((int)fx, nx - 2), j = min((int)fy, ny - 2), k = min((int)fz, nz - 2);
    float ax = fx - i, ay = fy - j, az = fz - k, v = 0.f;
    for (int di = 0; di < 2; di++) for (int dj = 0; dj < 2; dj++) for (int dk = 0; dk < 2; dk++) {
        float w = (di ? ax : 1.f - ax) * (dj ? ay : 1.f - ay) * (dk ? az : 1.f - az);
        v += w * (float)mg[((size_t)(i + di) * ny + (j + dj)) * nz + (k + dk)];
    }
    return v / 255.0f;
}
__device__ __forceinline__ float scope_dist(float ex, float ey, float ez, const float* bp) {
    float d2 = ex * ex + ey * ey + ez * ez;
    float T = bp[38];
    if (T > 0.f) {
        float dn = ex * bp[4] + ey * bp[5] + ez * bp[6];
        float dd = fmaxf(fabsf(dn) - T, 0.f);
        d2 = d2 - dn * dn + dd * dd;
    }
    return sqrtf(fmaxf(d2, 0.f));
}

extern "C" __global__ void dab_compute(const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                       int vol, int lev, const int* rows, const float* org, int nb,
                                       const float* bp, const float* lut, int nlut, float* out,
                                       const __half* mpool, const int* mslot, int use_mask,
                                       const unsigned char* smask, const int* smidx, int use_smask,
                                       const float* Tb, const int* Tslot,
                                       const float* steps, int nsteps,
                                       const unsigned char* mg, int mnx, int mny, int mnz) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    const float* vf = vft + vol * VF;
    const int* vi = vit + vol * VI;
    float h = vf[VF_H + lev];
    float x = org[3 * b] + i * h, y = org[3 * b + 1] + j * h, z = org[3 * b + 2] + k * h;
    float f0 = __half2float(sdf[(size_t)rows[b] * 729 + q]);
    out[id] = f0;
    if (!isfinite(f0)) return;                       // 11-09 guard: a non-finite node is never computed from
    int typ = (int)bp[0];
    float cx = bp[1], cy = bp[2], cz = bp[3], nx = bp[4], ny = bp[5], nz = bp[6], r = bp[7], s = bp[8];
    float vx = bp[9], vy = bp[10], vz = bp[11], dl = bp[12], A = bp[13], clampv = bp[14];
    float rx = x - cx, ry = y - cy, rz = z - cz;
    float d = sqrtf(rx * rx + ry * ry + rz * rz);
    if (d >= r) return;
    float w = lut_w(lut, nlut, d / r);
    if (w <= 0.0f) return;
    // fs-brush 11-12: the sheet under the brush (core/brush.py _sheet_flood, labelled on the host from the stroke's own
    // first-touch values): a node whose nearest surface is another sheet keeps its value
    if (use_smask && !smask[(size_t)smidx[b] * 729 + q]) return;
    // fs-brush 11-11: mirror feather (core/brush.py NBP_MIRP): where this dab's sphere overlaps its mirror partner's, the
    // node's weight is scaled by (w^2 + w2^2) / (w + w2)^2, so the pair sums to (w^2 + w2^2) / (w + w2): smooth, w alone
    // and where the two coincide (max() creased the midline: cpu_seam.py)
    if (bp[44] > 0.5f) {
        float ex = x - bp[41], ey = y - bp[42], ez = z - bp[43];
        float d2 = sqrtf(ex * ex + ey * ey + ez * ez);
        if (d2 < r) {
            float w2 = lut_w(lut, nlut, d2 / r);
            if (w2 > 0.0f) w = w * (w * w + w2 * w2) / ((w + w2) * (w + w2));   // contraharmonic (max() creased the midline)
        }
    }
    // fs-brush 11-11: a node farther from the surface than the dab's own move plus a band cannot carry the moved surface;
    // it keeps its value here and the live re-distance (rd_*) rebuilds its distance inside the footprint (bp[28] > 0:
    // that reach; 0: every node, as before)
    if (bp[28] > 0.0f && fabsf(f0) > bp[28]) {
        // fs-brush 11-12 (sculpt-contact's fold15 walls; fold17: none with the skip off): one-sided as the CPU (c8040703):
        // bp[45] 1 skips only deep INSIDE (a growing dab: the air ahead is never skipped, so no node there keeps a stale far
        // value while the surface grows into it), -1 only deep outside (a cutting dab), 0 both (as before)
        float sd = bp[45];
        if (sd == 0.0f || (sd > 0.0f && f0 < 0.0f) || (sd < 0.0f && f0 > 0.0f)) return;
    }
    if ((int)bp[15] != 0) { w *= target_gate(vf, vi, grid, ttab, sdf, lev, x, y, z, f0, bp); if (w <= 0.0f) return; }
    float outside = f0 >= 0.0f ? clampv : -clampv;
    float dn = rx * nx + ry * ny + rz * nz;
    if (bp[35] > 0.5f && (typ == 1 || typ == 5))      // fs-brush 11-11: Clay and Flatten on the area plane (area_plane)
        dn = (x - bp[29]) * bp[32] + (y - bp[30]) * bp[33] + (z - bp[31]) * bp[34];
    float f1 = f0;
    // every displaced read is checked where it is taken: 1e30 = outside every brick (no information) -> the node
    // keeps its value (r17b: a sentinel scaled into a sum slipped past a check on the result: smooth gave NaN)
#define RD(X, Y, Z) obj_f(vf, vi, grid, ttab, sdf, lev, (X), (Y), (Z), outside)
#define BAD(v) (fabsf(v) > 1e29f)
    if (typ == 0) { float r0 = RD(x - nx * A * w, y - ny * A * w, z - nz * A * w); if (BAD(r0)) return; f1 = r0; }
    else if (typ == 1) {
        // 11-10: at most A per dab. fs-core 11-14: Ctrl's dig (brush.invert_dab: A < 0) is the mirror rule, max(F, dn - A)
        // under the falloff and at most |A| a dab; min(F, dn - A) with A < 0 adds nothing and the limit max(f1, F - A)
        // then lifted every node of the ball by |A| whatever its weight: a step of |A| at the ball's edge each dab (Clay
        // and Ctrl-Clay alternated over a sphere: the front a foam of fragments, fs-core timing100, HEAD and band alike)
        if (A >= 0.0f) { f1 = f0 + w * (fminf(f0, dn - A) - f0); if (bp[22] > 0.5f) f1 = fmaxf(f1, f0 - A); }
        else { f1 = f0 + w * (fmaxf(f0, dn - A) - f0); if (bp[22] > 0.5f) f1 = fminf(f1, f0 - A); }
    }
    else if (typ == 2) { float r0 = RD(x + nx * A * w, y + ny * A * w, z + nz * A * w); if (BAD(r0)) return; f1 = r0; }
    else if (typ == 3 || typ == 11) {                // 11: swell to contact (the collider clamps it)
        float kk = 1.0f;
        // sculpt-contact 11-12: Swell to contact = the balloon times core/collide.swell_k (the same formula as the CPU):
        // Tb holds the contacts' distance at this row's 729 nodes (local units; Tslot[b] < 0: no contact within reach)
        if (typ == 11 && bp[46] > 0.5f && Tslot[b] >= 0) {
            const float* T = Tb + (size_t)Tslot[b] * 729;
            float gain = bp[47], E = bp[48], R = bp[49], gap = bp[50];
            float gs = fmaxf(T[q] - gap + f0, 0.0f);
            float nearw = fminf(fmaxf((R - gs) / fmaxf(R - r, 1e-12f), 0.0f), 1.0f);
            if (nearw > 0.0f) {
                int ip = min(i + 1, 8), im = max(i - 1, 0), jp = min(j + 1, 8), jm = max(j - 1, 0), kp = min(k + 1, 8), km = max(k - 1, 0);
                float gx = (T[ip * 81 + j * 9 + k] - T[im * 81 + j * 9 + k]) / ((ip - im) * h);
                float gy = (T[i * 81 + jp * 9 + k] - T[i * 81 + jm * 9 + k]) / ((jp - jm) * h);
                float gz = (T[i * 81 + j * 9 + kp] - T[i * 81 + j * 9 + km]) / ((kp - km) * h);
                float e2 = 2.0f * h;                         // the body's normal at 2 cells (SWELL_GRAD_CELLS)
                float ax = RD(x + e2, y, z), bx = RD(x - e2, y, z), ay = RD(x, y + e2, z), by = RD(x, y - e2, z);
                float az = RD(x, y, z + e2), bz = RD(x, y, z - e2);
                float cs = 0.0f;
                if (!(BAD(ax) || BAD(bx) || BAD(ay) || BAD(by) || BAD(az) || BAD(bz))) {
                    float nx_ = ax - bx, ny_ = ay - by, nz_ = az - bz;
                    float gl = sqrtf(gx * gx + gy * gy + gz * gz), bl = sqrtf(nx_ * nx_ + ny_ * ny_ + nz_ * nz_);
                    if (gl > 1e-12f && bl > 1e-12f) cs = -(gx * nx_ + gy * ny_ + gz * nz_) / (gl * bl);
                }
                float face = 0.5f + 0.5f * fminf(fmaxf(cs, -1.0f), 1.0f);
                float ease = 1.0f - expf(-(gs + 0.5f * h) / E);
                kk = 1.0f + nearw * (gain * ease * face - 1.0f);
            }
        }
        f1 = f0 - A * w * kk;
    }
    else if (typ == 4 && bp[23] > 0.5f) {           // fs-brush 11-11: Smooth at the brush's scale (quad_smooth)
        if (fabsf(f0) >= bp[24] * h) return;
        bool ok;
        float a = quad_smooth(vf, vi, grid, ttab, sdf, lev, x, y, z, f0, h, r, bp, &ok);
        if (!ok || (bp[40] > 0.5f && bp[40] < 1.5f)) a = 0.0f;      // bp[40] 1: the fit off (diagnosis)
        // + the lattice's own scale (M, fs-bench test 1 at full resolution: the brush-scale fit alone left the cell-scale
        // roughness of large Grabs, a speckled jaw, that the old 2 h average wiped out): bp[38] 0 = the six neighbours at
        // 2 h, 1 = a binomial 3^3 low-pass at h (core/brush.py NBP_CELL), 2 = none
        float lap = 0.0f;
        if (bp[38] < 0.5f) {
            float e2 = 2.0f * h;
            float a0 = RD(x + e2, y, z), a1 = RD(x - e2, y, z), a2 = RD(x, y + e2, z), a3 = RD(x, y - e2, z), a4 = RD(x, y, z + e2), a5 = RD(x, y, z - e2);
            if (!(BAD(a0) || BAD(a1) || BAD(a2) || BAD(a3) || BAD(a4) || BAD(a5))) lap = (a0 + a1 + a2 + a3 + a4 + a5) / 6.0f - f0;
        } else if (bp[38] < 1.5f) {
            float accb = 0.0f;
            bool bad = false;
            for (int di = -1; di <= 1 && !bad; di++)
                for (int dj = -1; dj <= 1 && !bad; dj++)
                    for (int dk = -1; dk <= 1; dk++) {
                        float v = RD(x + di * h, y + dj * h, z + dk * h);
                        if (BAD(v)) { bad = true; break; }
                        accb += (float)((2 - abs(di)) * (2 - abs(dj)) * (2 - abs(dk))) * v;
                    }
            if (!bad) lap = accb / 64.0f - f0;
        }
        float dv = s * w * (a + lap), cap = bp[27] * h;
        f1 = f0 + fminf(fmaxf(dv, -cap), cap);
    }
    else if (typ == 4 && bp[20] > 0.5f) {
        float av = sheet_avg(vf, vi, grid, ttab, sdf, lev, x, y, z, f0, dl, h, outside);
        if (fabsf(av) > 1e29f) return;     // 1e30: keep the node (no NaN under fast math)
        f1 = f0 + s * w * (av - f0);
    } else if (typ == 4) {
        float a0 = RD(x + dl, y, z), a1 = RD(x - dl, y, z), a2 = RD(x, y + dl, z), a3 = RD(x, y - dl, z), a4 = RD(x, y, z + dl), a5 = RD(x, y, z - dl);
        if (BAD(a0) || BAD(a1) || BAD(a2) || BAD(a3) || BAD(a4) || BAD(a5)) return;
        f1 = f0 + s * w * ((a0 + a1 + a2 + a3 + a4 + a5) / 6.0f - f0);
    } else if (typ == 5) { f1 = f0 + s * w * (dn - f0); if (bp[22] > 0.5f) f1 = fminf(fmaxf(f1, f0 - A), f0 + A); }
    else if (typ == 6 || typ == 7) {
        float tx = rx - nx * dn, ty = ry - ny * dn, tz = rz - nz * dn;
        bool v2 = bp[37] > 0.5f;
        float kk = s * w * ((v2 || typ == 7) ? 1.0f : 0.5f);
        if (v2) {                                   // fs-brush 11-11: toward the stroke LINE (bp[9:12] its direction)
            float vn = vx * nx + vy * ny + vz * nz;
            float dx = vx - nx * vn, dy = vy - ny * vn, dz = vz - nz * vn, dl2 = sqrtf(dx * dx + dy * dy + dz * dz);
            if (dl2 > 1e-9f) { dx /= dl2; dy /= dl2; dz /= dl2; float ta = tx * dx + ty * dy + tz * dz; tx -= dx * ta; ty -= dy * ta; tz -= dz * ta; }
        } else if (typ == 7 && bp[36] > 0.0f) kk = fminf(fmaxf(kk, -bp[36]), bp[36]);   // fs-brush 11-11: Pinch bounded (bp[36])
        float r0 = RD(x + tx * kk, y + ty * kk, z + tz * kk);
        if (BAD(r0)) return;
        f1 = r0;
        if (typ == 6) f1 += A * w * w * w;
    } else if (typ == 8) { float r0 = RD(x - vx * w, y - vy * w, z - vz * w); if (BAD(r0)) return; f1 = r0; }
    else if (typ == 14) {
        // fs-brush 11-12 (toolkit review 4.2): the anchored flow Grab (core/brush.py _flow_back): a kernel of radius rt
        // carried from c0 to c0 + v moves space at falloff x v; the node is walked back to t = 0 (midpoint steps) and reads
        // the stroke-start field there (the stroke's rows were restored to it at the batch's start: gpu/sculpt.py apply)
        float px = x, py = y, pz = z;
        float c0x = bp[29], c0y = bp[30], c0z = bp[31], rt = bp[32];
        int ns = (int)bp[33]; if (ns < 1) ns = 1;
        float dt = 1.0f / (float)ns;
        for (int kk = ns; kk > 0; kk--) {
            float t1 = kk * dt;
            float ex = px - (c0x + t1 * vx), ey = py - (c0y + t1 * vy), ez = pz - (c0z + t1 * vz);
            float dd = scope_dist(ex, ey, ez, bp);
            float w1 = dd < rt ? lut_w(lut, nlut, dd / rt) : 0.0f;
            if (w1 > 0.f) w1 *= scope_at(px, py, pz, bp, mg, mnx, mny, mnz);
            float xm = px - 0.5f * dt * w1 * vx, ym = py - 0.5f * dt * w1 * vy, zm = pz - 0.5f * dt * w1 * vz;
            float tm = t1 - 0.5f * dt;
            ex = xm - (c0x + tm * vx); ey = ym - (c0y + tm * vy); ez = zm - (c0z + tm * vz);
            dd = scope_dist(ex, ey, ez, bp);
            float wm = dd < rt ? lut_w(lut, nlut, dd / rt) : 0.0f;
            if (wm > 0.f) wm *= scope_at(xm, ym, zm, bp, mg, mnx, mny, mnz);
            px -= dt * wm * vx; py -= dt * wm * vy; pz -= dt * wm * vz;
        }
        float r0 = RD(px, py, pz); if (BAD(r0)) return; f1 = r0;
    }
    else if (typ == 16) {
        // fs-brush 11-12 (Snake Hook's flat plug): Snake Hook's step as a flow (core/brush.py snake_back):
        // a kernel of radius rt carried from c0 to c0 + v moves space at its falloff, plus Blender's Magnify (an outward
        // velocity perpendicular to the step, magnify x |v| x w x distance / rt); walked back along the step (midpoint)
        float px = x, py = y, pz = z;
        float c0x = bp[29], c0y = bp[30], c0z = bp[31], rt = bp[32];
        int ns = (int)bp[33];
        float vl = sqrtf(vx * vx + vy * vy + vz * vz);
        if (vl > 1e-12f && ns >= 1) {
            float ax = vx / vl, ay = vy / vl, az = vz / vl;
            float m = bp[34] * vl / rt;
            float ox = bp[38], oy = bp[39], oz = bp[40];   // Rake: the step's turn (axis x angle)
            float dt = 1.0f / (float)ns;
            for (int kk = ns; kk > 0; kk--) {
                float t1 = kk * dt;
                float ex = px - (c0x + t1 * vx), ey = py - (c0y + t1 * vy), ez = pz - (c0z + t1 * vz);
                float dd = sqrtf(ex * ex + ey * ey + ez * ez);
                float w1 = dd < rt ? lut_w(lut, nlut, dd / rt) : 0.0f;
                float ta = ex * ax + ey * ay + ez * az;
                float ux = w1 * (vx + m * (ex - ax * ta) + oy * ez - oz * ey);
                float uy = w1 * (vy + m * (ey - ay * ta) + oz * ex - ox * ez);
                float uz = w1 * (vz + m * (ez - az * ta) + ox * ey - oy * ex);
                float xm = px - 0.5f * dt * ux, ym = py - 0.5f * dt * uy, zm = pz - 0.5f * dt * uz;
                float tm = t1 - 0.5f * dt;
                ex = xm - (c0x + tm * vx); ey = ym - (c0y + tm * vy); ez = zm - (c0z + tm * vz);
                dd = sqrtf(ex * ex + ey * ey + ez * ez);
                float wm = dd < rt ? lut_w(lut, nlut, dd / rt) : 0.0f;
                ta = ex * ax + ey * ay + ez * az;
                px -= dt * wm * (vx + m * (ex - ax * ta) + oy * ez - oz * ey);
                py -= dt * wm * (vy + m * (ey - ay * ta) + oz * ex - ox * ez);
                pz -= dt * wm * (vz + m * (ez - az * ta) + ox * ey - oy * ex);
            }
        }
        float r0 = RD(px, py, pz); if (BAD(r0)) return; f1 = r0;
    }
    else if (typ == 17) {
        // fs-brush 11-12: Snake Hook's whole path composed (core/brush.py snake_path_back): the node walks back through
        // every step (the latest first; each the flow of typ 16 with its own c0, v, rt, substeps, Magnify, Rake) to the
        // stroke-start field (its rows were restored this batch); a node no step reaches is left as it is
        float px = x, py = y, pz = z;
        bool moved = false;
        for (int s = nsteps - 1; s >= 0; s--) {
            const float* S = steps + 12 * s;
            float c0x = S[0], c0y = S[1], c0z = S[2], svx = S[3], svy = S[4], svz = S[5], rt = S[6];
            int ns = (int)S[7];
            float vl = sqrtf(svx * svx + svy * svy + svz * svz);
            if (vl < 1e-12f || ns < 1) continue;
            float ax = svx / vl, ay = svy / vl, az = svz / vl;
            float qx = px - c0x, qy = py - c0y, qz = pz - c0z;
            float tq = fminf(fmaxf(qx * ax + qy * ay + qz * az, 0.0f), vl);
            float ex = qx - ax * tq, ey = qy - ay * tq, ez = qz - az * tq;
            if (ex * ex + ey * ey + ez * ez >= rt * rt) continue;
            moved = true;
            float m = S[8] * vl / rt;
            float ox = S[9], oy = S[10], oz = S[11];
            float dt = 1.0f / (float)ns;
            for (int kk = ns; kk > 0; kk--) {
                float t1 = kk * dt;
                ex = px - (c0x + t1 * svx); ey = py - (c0y + t1 * svy); ez = pz - (c0z + t1 * svz);
                float dd = sqrtf(ex * ex + ey * ey + ez * ez);
                float w1 = dd < rt ? lut_w(lut, nlut, dd / rt) : 0.0f;
                float ta = ex * ax + ey * ay + ez * az;
                float ux = w1 * (svx + m * (ex - ax * ta) + oy * ez - oz * ey);
                float uy = w1 * (svy + m * (ey - ay * ta) + oz * ex - ox * ez);
                float uz = w1 * (svz + m * (ez - az * ta) + ox * ey - oy * ex);
                float xm = px - 0.5f * dt * ux, ym = py - 0.5f * dt * uy, zm = pz - 0.5f * dt * uz;
                float tm = t1 - 0.5f * dt;
                ex = xm - (c0x + tm * svx); ey = ym - (c0y + tm * svy); ez = zm - (c0z + tm * svz);
                dd = sqrtf(ex * ex + ey * ey + ez * ez);
                float wm = dd < rt ? lut_w(lut, nlut, dd / rt) : 0.0f;
                ta = ex * ax + ey * ay + ez * az;
                px -= dt * wm * (svx + m * (ex - ax * ta) + oy * ez - oz * ey);
                py -= dt * wm * (svy + m * (ey - ay * ta) + oz * ex - ox * ez);
                pz -= dt * wm * (svz + m * (ez - az * ta) + ox * ey - oy * ex);
            }
        }
        if (!moved) return;
        float r0 = RD(px, py, pz); if (BAD(r0)) return; f1 = r0;
    }
    else if (typ == 15) {
        // fs-brush 11-12 (toolkit review 4.3): Stretch as an anchored elongation (core/brush.py _stretch_back): the
        // region from the pivot P (bp 29-31, D = bp[32] behind the press point) to the pressed end lengthens by
        // Le = L x falloff(distance from the axis / bp[33]); the end's move fades over bp[23] beyond it; exact inverse
        float L = sqrtf(vx * vx + vy * vy + vz * vz);
        float px = x, py = y, pz = z;
        if (L > 1e-12f) {
            float ax = vx / L, ay = vy / L, az = vz / L;
            float ex = x - bp[29], ey = y - bp[30], ez = z - bp[31];
            float xa = ex * ax + ey * ay + ez * az;
            if (xa > 0.0f) {
                float qx = ex - xa * ax, qy = ey - xa * ay, qz = ez - xa * az;
                float rho = sqrtf(qx * qx + qy * qy + qz * qz);
                float D = bp[32], rr = bp[33], fade = bp[23];
                if (rho < rr) {
                    float Le = L * lut_w(lut, nlut, rho / rr);
                    float xa0 = xa;
                    if (Le > 0.0f) {
                        if (xa <= D + Le) xa0 = xa * D / (D + Le);
                        else if (xa <= D + fade) xa0 = D + (xa - D - Le) * fade / (fade - Le);
                    }
                    float d_ = xa0 - xa;
                    px = x + d_ * ax; py = y + d_ * ay; pz = z + d_ * az;
                }
            }
        }
        float r0 = RD(px, py, pz); if (BAD(r0)) return; f1 = r0;
    }
    else if (typ == 9) {
        float vl = sqrtf(vx * vx + vy * vy + vz * vz);
        if (vl < 1e-12f) return;
        float ax = vx / vl, ay = vy / vl, az = vz / vl;
        float kw = (vl / r) * w;
        float pr = (rx * ax + ry * ay + rz * az) * (kw / (1.0f + kw));
        float r0 = RD(x - ax * pr, y - ay * pr, z - az * pr);
        if (BAD(r0)) return;
        f1 = r0;
    }
    if (!isfinite(f1)) return;                       // 11-09 guard: the node keeps its value
    if (use_mask) {
        int ms = mslot[rows[b]];                         // fs-brush 11-11: the stroke's own rows only (compact buffer)
        float m = ms >= 0 ? __half2float(mpool[(size_t)ms * 729 + q]) : 0.0f;
        m = fminf(fmaxf(m, 0.0f), 1.0f);
        f1 = f0 + (f1 - f0) * (1.0f - m);
    }
    out[id] = f1;
}

// fs-tools 11-14: the approved brush tools (core/brush.FT_TOOLS): toolset/ftools.cuh's rules in their own kernel, launched
// by gpu/sculpt.py _run with dab_compute's arguments (dab_compute itself is unchanged). The field is read as every
// branch of dab_compute reads it (obj_f; 1e30 outside every brick = no information: the node keeps its value); bp[7] is
// the tool's whole ball (radius + Smoothing); their own profile (warp/warp.cuh w_profile), not the lut; the sheet under
// the brush, the skip and the stroke's mask as dab_compute; Stop at contact reads the contact rows (Tb, Tslot).
// gpu/sculpt.py compiles with -I<fieldstudio> and -DFT_TYP_SWELL / SETTLE / FLATTEN / PENCIL = brush.T's codes.
#define WREAL float
#include "warp/warp.cuh"
#include "toolset/ftools.cuh"
struct FtObjRd {
    const float* vf; const int* vi; const int* grid; const int* ttab; const __half* sdf; int lev; float outside;
    __device__ float operator()(float x, float y, float z) const { return obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z, outside); }
};
struct FtStartRd {     // the press-time field (patch 3); without a pool (use 0) the field as it is
    const float* vf; const int* vi; const int* grid; const int* ttab; const __half* sdf; int lev; float outside;
    const int* bslot; int nbslot; const __half* bpool; int use;
    __device__ float operator()(float x, float y, float z) const {
        return use ? obj_f_start(vf, vi, grid, ttab, sdf, lev, x, y, z, bslot, nbslot, bpool)
                   : obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z, outside);
    }
};

extern "C" __global__ void ft_dab(const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                  int vol, int lev, const int* rows, const float* org, int nb,
                                  const float* bp, const float* lut, int nlut, float* out,
                                  const __half* mpool, const int* mslot, int use_mask,
                                  const unsigned char* smask, const int* smidx, int use_smask,
                                  const float* Tb, const int* Tslot,
                                  const float* steps, int nsteps,
                                  const unsigned char* mg, int mnx, int mny, int mnz,
                                  const int* bslot, int nbslot, const __half* bpool, int use_start) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    const float* vf = vft + vol * VF;
    const int* vi = vit + vol * VI;
    float h = vf[VF_H + lev];
    float x = org[3 * b] + i * h, y = org[3 * b + 1] + j * h, z = org[3 * b + 2] + k * h;
    float f0 = __half2float(sdf[(size_t)rows[b] * 729 + q]);
    out[id] = f0;
    if (!isfinite(f0)) return;
    int typ = (int)bp[0];
    if (typ < FT_TYP_SWELL || typ > FT_TYP_PENCIL) return;      // fs-tools patch 4: + Pencil (ftools.cuh ft_pencil)
    float rx = x - bp[1], ry = y - bp[2], rz = z - bp[3];
    if (rx * rx + ry * ry + rz * rz >= bp[7] * bp[7]) return;
    if (use_smask && !smask[(size_t)smidx[b] * 729 + q]) return;
    if (bp[28] > 0.0f && fabsf(f0) > bp[28]) {
        float sd = bp[45];
        if (sd == 0.0f || (sd > 0.0f && f0 < 0.0f) || (sd < 0.0f && f0 > 0.0f)) return;
    }
    FtObjRd rd{vf, vi, grid, ttab, sdf, lev, f0 >= 0.0f ? bp[14] : -bp[14]};
    FtStartRd rs{vf, vi, grid, ttab, sdf, lev, f0 >= 0.0f ? bp[14] : -bp[14], bslot, nbslot, bpool, use_start};
    float tc = (bp[FT_CONTACT] > 0.5f && Tslot[b] >= 0) ? Tb[(size_t)Tslot[b] * 729 + q] : 1e9f;
    float f1 = ft_rule2(rd, rs, typ, bp, x, y, z, f0, tc, use_start != 0);
    if (!isfinite(f1) || fabsf(f1) > 1e29f) return;
    if (use_mask) {
        int ms = mslot[rows[b]];
        float m = ms >= 0 ? __half2float(mpool[(size_t)ms * 729 + q]) : 0.0f;
        m = fminf(fmaxf(m, 0.0f), 1.0f);
        f1 = f0 + (f1 - f0) * (1.0f - m);
    }
    out[id] = f1;
}

extern "C" __global__ void dab_commit(__half* sdf, const int* rows, int nb, const float* vals) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    sdf[(size_t)rows[id / 729] * 729 + id % 729] = __float2half_rn(vals[id]);
}

// 11-09 collider (core/collide.py): a node inside a locked part + gap (B > 0) is kept at or above min(F_start, B)
extern "C" __global__ void collide_clamp(__half* sdf, const int* rows, int nb, const float* before, const float* B, float band) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    float b = B[id];
    if (!(b > band)) return;                     // band: -2 h (core/collide.py BAND_CELLS)
    float lim = fminf(before[id], b);
    size_t a = (size_t)rows[id / 729] * 729 + id % 729;
    float v = __half2float(sdf[a]);
    if (v < lim) sdf[a] = __float2half_ru(lim);
}

// 11-09 T4 candidate 'scale': every node of the stroke's bricks divided by the field's local gradient magnitude
// (central differences of the atlas field at +-h): signs and zeros unchanged, so the zero set moves only by the
// second-order variation of the factor across a cell; |grad F| returns toward 1. Reads the pool, writes out.
extern "C" __global__ void redist_scale(const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                        int vol, int lev, const int* rows, const float* org, int nb, float* out) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    const float* vf = vft + vol * VF;
    const int* vi = vit + vol * VI;
    float h = vf[VF_H + lev];
    float x = org[3 * b] + i * h, y = org[3 * b + 1] + j * h, z = org[3 * b + 2] + k * h;
    float f0 = __half2float(sdf[(size_t)rows[b] * 729 + q]);
    out[id] = f0;
    float a = obj_f(vf, vi, grid, ttab, sdf, lev, x + h, y, z, 0.f), c = obj_f(vf, vi, grid, ttab, sdf, lev, x - h, y, z, 0.f);
    float d = obj_f(vf, vi, grid, ttab, sdf, lev, x, y + h, z, 0.f), e = obj_f(vf, vi, grid, ttab, sdf, lev, x, y - h, z, 0.f);
    float f = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z + h, 0.f), g = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z - h, 0.f);
    if (fabsf(a) > 1e29f || fabsf(c) > 1e29f || fabsf(d) > 1e29f || fabsf(e) > 1e29f || fabsf(f) > 1e29f || fabsf(g) > 1e29f) return;
    float gx = (a - c) / (2.0f * h), gy = (d - e) / (2.0f * h), gz = (f - g) / (2.0f * h);
    float gm = sqrtf(gx * gx + gy * gy + gz * gz);
    if (!(gm > 0.2f)) return;                     // a mid-plane or a flat region: left as it is
    // r2 (r1 moved the surface p99 30 um on the first stroke: the cells holding it were rescaled by varying factors):
    // the nodes of the cells that can hold the surface (|F| < 2 h |grad F|) keep their values, the zero set exactly
    if (fabsf(f0) < 2.0f * h * gm) return;
    out[id] = f0 / fminf(gm, 5.0f);
}

// 11-09 T4 candidate 'scale2' (r64's gate: the pass's own shift p99 <= 5 um, |grad F| p99 <= 1.3 after): pass 1, the
// field's |grad F| at every node of the stroke's bricks (central differences of the atlas at +-h; 0 where unknown)
extern "C" __global__ void grad_mag(const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                    int vol, int lev, const int* rows, const float* org, int nb, float* gmag) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    const float* vf = vft + vol * VF;
    const int* vi = vit + vol * VI;
    float h = vf[VF_H + lev];
    float x = org[3 * b] + i * h, y = org[3 * b + 1] + j * h, z = org[3 * b + 2] + k * h;
    gmag[id] = 0.0f;
    float a = obj_f(vf, vi, grid, ttab, sdf, lev, x + h, y, z, 0.f), c = obj_f(vf, vi, grid, ttab, sdf, lev, x - h, y, z, 0.f);
    float d = obj_f(vf, vi, grid, ttab, sdf, lev, x, y + h, z, 0.f), e = obj_f(vf, vi, grid, ttab, sdf, lev, x, y - h, z, 0.f);
    float f = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z + h, 0.f), g = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z - h, 0.f);
    if (fabsf(a) > 1e29f || fabsf(c) > 1e29f || fabsf(d) > 1e29f || fabsf(e) > 1e29f || fabsf(f) > 1e29f || fabsf(g) > 1e29f) return;
    float gx = (a - c) / (2.0f * h), gy = (d - e) / (2.0f * h), gz = (f - g) / (2.0f * h);
    gmag[id] = sqrtf(gx * gx + gy * gy + gz * gz);
}

// pass 2: each node divided by the mean |grad F| over its 3x3x3 neighbourhood within the brick (a slowly varying
// factor moves a crossing on an edge by about t (1 - t) h dS / S); nodes with no usable gradient keep their value
extern "C" __global__ void scale_smooth(const __half* sdf, const int* rows, int nb, const float* gmag, float* out) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    float f0 = __half2float(sdf[(size_t)rows[b] * 729 + q]);
    float acc = 0.0f; int n = 0;
    for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++) {
        int ii = i + di, jj = j + dj, kk = k + dk;
        if (ii < 0 || jj < 0 || kk < 0 || ii > 8 || jj > 8 || kk > 8) continue;
        float g = gmag[(size_t)b * 729 + (ii * 9 + jj) * 9 + kk];
        if (g > 0.2f) { acc += fminf(g, 5.0f); n++; }
    }
    // r3 (drift r2: one release divided stretched regions by up to 5: the pass's own shift p99 13.6 um, a card hit
    // 2.7 mm off, the probe brush 3.1 x nominal): each release corrects by at most a factor 0.8-1.25
    out[id] = (n >= 8) ? f0 / fminf(fmaxf(acc / n, 0.8f), 1.25f) : f0;
}

// gather bricks (rows) into a dense (nb, 729) fp16 buffer for download
extern "C" __global__ void gather_rows(const __half* sdf, const int* rows, int nb, __half* out) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    out[id] = sdf[(size_t)rows[id / 729] * 729 + id % 729];
}

// r16: apron nodes (index 8 on an axis) are copies of the neighbour brick's node 0 (nodes 0..7 are each brick's own).
// After a batch, every apron of the given bricks is refreshed from its owner where the owner brick exists.
// keys: (nb, 3) brick ijk in the volume grid (tiles of bpt^3 bricks)
extern "C" __global__ void apron_fill(const float* vft, const int* vit, const int* grid, const int* ttab, __half* sdf,
                                      int vol, int lev, const int* rows, const int* keys, int nb) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    if (i < 8 && j < 8 && k < 8) return;
    const int* vi = vit + vol * VI;
    int bpt = vi[VI_BPT + lev];
    int ki = keys[3 * b] + (i == 8), kj = keys[3 * b + 1] + (j == 8), kk = keys[3 * b + 2] + (k == 8);
    int tx = ki / bpt, ty = kj / bpt, tz = kk / bpt;
    if (ki < 0 || kj < 0 || kk < 0 || tx >= vi[VI_NX] || ty >= vi[VI_NY] || tz >= vi[VI_NZ]) return;
    int slot = grid[vi[VI_GOFF] + ((tx * vi[VI_NY] + ty) * vi[VI_NZ] + tz) * NLEV + lev];
    if (slot < 0) return;
    int bid = ttab[vi[VI_TOFF + lev] + slot * bpt * bpt * bpt + ((ki - tx * bpt) * bpt + (kj - ty * bpt)) * bpt + (kk - tz * bpt)];
    if (bid < 0) return;
    int oi = i & 7, oj = j & 7, ok = k & 7;
    sdf[(size_t)rows[b] * 729 + q] = sdf[(size_t)(bid + vi[VI_BOFF + lev]) * 729 + (oi * 9 + oj) * 9 + ok];
}

// r23: re-distance at a stroke's release, on the card. For each node of the given bricks within the narrow band
// (|F| < 6h), the closest point of the zero set is found by Newton projection on the atlas field
// (y <- y - F(y) grad F / |grad F|^2, 5 steps; accepted when |F(foot)| < 0.05 h), and the node moves toward the
// signed distance |x - foot| by a weight continuous in F (1 inside 3h, 0 at 6h) and in the depth inside the stroke's
// union of dab spheres (0 at its edge, 1 at a quarter radius in), so no seam can appear at either edge.
// dabs: (nd, 4) local centre + radius. Writes `out` (nb * 729); dab_commit writes it back.
__device__ __forceinline__ float obj_g(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                                       float x, float y, float z, float e, float* gx, float* gy, float* gz) {
    float a = obj_f(vf, vi, grid, ttab, sdf, lev, x + e, y, z, 0.0f), b = obj_f(vf, vi, grid, ttab, sdf, lev, x - e, y, z, 0.0f);
    float c = obj_f(vf, vi, grid, ttab, sdf, lev, x, y + e, z, 0.0f), d = obj_f(vf, vi, grid, ttab, sdf, lev, x, y - e, z, 0.0f);
    float f = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z + e, 0.0f), g = obj_f(vf, vi, grid, ttab, sdf, lev, x, y, z - e, 0.0f);
    if (fabsf(a) > 1e29f || fabsf(b) > 1e29f || fabsf(c) > 1e29f || fabsf(d) > 1e29f || fabsf(f) > 1e29f || fabsf(g) > 1e29f) return 0.0f;
    *gx = (a - b) / (2.0f * e); *gy = (c - d) / (2.0f * e); *gz = (f - g) / (2.0f * e);
    return 1.0f;
}

extern "C" __global__ void redist(const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                  int vol, int lev, const int* rows, const float* org, int nb,
                                  const float* dabs, int nd, float* out, int* stats) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    const float* vf = vft + vol * VF;
    const int* vi = vit + vol * VI;
    float h = vf[VF_H + lev];
    float x = org[3 * b] + i * h, y = org[3 * b + 1] + j * h, z = org[3 * b + 2] + k * h;
    float f0 = __half2float(sdf[(size_t)rows[b] * 729 + q]);
    out[id] = f0;
    float band = 6.0f * h, core = 3.0f * h;
    if (fabsf(f0) >= band) return;
    float depth = -1e30f, rr = h;                      // depth inside the union of the stroke's dab spheres
    for (int d = 0; d < nd; d++) {
        float dx = x - dabs[4 * d], dy = y - dabs[4 * d + 1], dz = z - dabs[4 * d + 2];
        float dep = dabs[4 * d + 3] - sqrtf(dx * dx + dy * dy + dz * dz);
        if (dep > depth) { depth = dep; rr = dabs[4 * d + 3]; }
    }
    float ws = fminf(fmaxf(depth / (0.25f * rr), 0.0f), 1.0f);
    float wf = fminf(fmaxf((band - fabsf(f0)) / (band - core), 0.0f), 1.0f);
    float w = ws * wf;
    if (w <= 0.0f) return;
    float px = x, py = y, pz = z, e = 0.5f * h;
    for (int it = 0; it < 5; it++) {
        float gx, gy, gz;
        float fy = obj_f(vf, vi, grid, ttab, sdf, lev, px, py, pz, 0.0f);
        if (fabsf(fy) > 1e29f) return;
        if (obj_g(vf, vi, grid, ttab, sdf, lev, px, py, pz, e, &gx, &gy, &gz) == 0.0f) return;
        float g2 = gx * gx + gy * gy + gz * gz;
        if (g2 < 1e-8f) return;
        px -= fy * gx / g2; py -= fy * gy / g2; pz -= fy * gz / g2;
    }
    float ff = obj_f(vf, vi, grid, ttab, sdf, lev, px, py, pz, 0.0f);
    if (!(fabsf(ff) < 0.05f * h)) { atomicAdd(stats, 1); return; }
    float dist = sqrtf((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz));
    if (dist > 1.5f * band) { atomicAdd(stats + 1, 1); return; }
    float D = f0 < 0.0f ? -dist : dist;
    out[id] = f0 + w * (D - f0);
}

// r41: re-distance by a jump-flood closest-point transform over a stroke's bricks (no projection through the field).
// Seeds: the exact zero crossings of the lattice edges at each node (the trilinear field is linear along an edge);
// a node keeps the nearest of its 6 edge crossings. Jump flooding (steps 16, 8, 4, 2, 1, 1 nodes, 26 neighbours)
// spreads the nearest seed; the new value is sign(F) |x - seed|, blended in by a weight continuous in F (1 inside 3h,
// 0 at 6h) and in the depth inside the stroke's union of dab spheres. Nodes are the bricks' own 8^3; neighbours are
// found through rowmap (the volume's cell -> pool row grid) and regmap (pool row -> region index).
__device__ __forceinline__ int node_row(const int* rowmap, int nbx, int nby, int nbz, int I, int J, int K) {
    if (I < 0 || J < 0 || K < 0) return -1;
    int bi = I >> 3, bj = J >> 3, bk = K >> 3;
    if (bi >= nbx || bj >= nby || bk >= nbz) return -1;
    return rowmap[((size_t)bi * nby + bj) * nbz + bk];
}

extern "C" __global__ void jfa_seed(const __half* sdf, const int* rowmap, int nbx, int nby, int nbz,
                                    const int* rows, const int* keys, int nb, float h, float4* seed) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
    float f0 = __half2float(sdf[(size_t)rows[b] * 729 + (i * 9 + j) * 9 + k]);
    float best = 1e30f; float3 sp = mk3(0, 0, 0);
    const int D[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int e = 0; e < 6; e++) {
        int I2 = I + D[e][0], J2 = J + D[e][1], K2 = K + D[e][2];
        int r2 = node_row(rowmap, nbx, nby, nbz, I2, J2, K2);
        if (r2 < 0) continue;
        float f1 = __half2float(sdf[(size_t)r2 * 729 + ((I2 & 7) * 9 + (J2 & 7)) * 9 + (K2 & 7)]);
        if ((f0 < 0.0f) == (f1 < 0.0f)) continue;
        float t = f0 / (f0 - f1);                    // the crossing along the edge, in nodes from this node
        if (t < best) { best = t; sp = mk3(I + D[e][0] * t, J + D[e][1] * t, K + D[e][2] * t); }
    }
    seed[id] = best < 1e29f ? make_float4(sp.x, sp.y, sp.z, 1.0f) : make_float4(0, 0, 0, 0);
}

extern "C" __global__ void jfa_step(const int* rowmap, int nbx, int nby, int nbz, const int* regmap,
                                    const int* keys, int nb, int step, const float4* sin, float4* sout) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
    float4 s0 = sin[id];
    float bd = s0.w > 0.5f ? (s0.x - I) * (s0.x - I) + (s0.y - J) * (s0.y - J) + (s0.z - K) * (s0.z - K) : 1e30f;
    float4 bs = s0;
    for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
        if (!dx && !dy && !dz) continue;
        int I2 = I + dx * step, J2 = J + dy * step, K2 = K + dz * step;
        int r2 = node_row(rowmap, nbx, nby, nbz, I2, J2, K2);
        if (r2 < 0) continue;
        int g = regmap[r2];
        if (g < 0) continue;
        float4 s = sin[(size_t)g * 512 + ((I2 & 7) * 8 + (J2 & 7)) * 8 + (K2 & 7)];
        if (s.w < 0.5f) continue;
        float d = (s.x - I) * (s.x - I) + (s.y - J) * (s.y - J) + (s.z - K) * (s.z - K);
        if (d < bd) { bd = d; bs = s; }
    }
    sout[id] = bs;
}

extern "C" __global__ void jfa_apply(__half* sdf, const int* rows, const int* keys, int nb, float h, const float* org0,
                                     const float* dabs, int nd, const float4* seed, int* stats) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
    size_t a = (size_t)rows[b] * 729 + (i * 9 + j) * 9 + k;
    float f0 = __half2float(sdf[a]);
    float band = 6.0f * h, core = 3.0f * h;
    if (fabsf(f0) >= band) return;
    float4 s = seed[id];
    if (s.w < 0.5f) { atomicAdd(stats, 1); return; }
    float x = org0[0] + I * h, y = org0[1] + J * h, z = org0[2] + K * h;
    float depth = -1e30f, rr = h;
    for (int d = 0; d < nd; d++) {
        float dx = x - dabs[4 * d], dy = y - dabs[4 * d + 1], dz = z - dabs[4 * d + 2];
        float dep = dabs[4 * d + 3] - sqrtf(dx * dx + dy * dy + dz * dz);
        if (dep > depth) { depth = dep; rr = dabs[4 * d + 3]; }
    }
    float w = fminf(fmaxf(depth / (0.25f * rr), 0.0f), 1.0f) * fminf(fmaxf((band - fabsf(f0)) / (band - core), 0.0f), 1.0f);
    if (w <= 0.0f) return;
    float dist = h * sqrtf((s.x - I) * (s.x - I) + (s.y - J) * (s.y - J) + (s.z - K) * (s.z - K));
    float D = f0 < 0.0f ? -dist : dist;
    sdf[a] = __float2half_rn(f0 + w * (D - f0));
}

// 11-09 night (a smooth over narrow folds laid a film of flesh over them;
// smoothing one fold pulled the facing fold over): a brush may reshape a surface but never join two surfaces that were
// apart, close a gap into a film, or tear one apart. Topology-preserving update (simple points, Bertrand & Malandain;
// as in topology-preserving level sets): a node may change sign only if the inside (f < 0, 26-connected) and the
// outside (6-connected) keep their connectivity around it (11-10: holes and cavities may close, flip_ok). Nodes are
// committed in 8 parity colours, so no two
// 26-adjacent nodes change in the same pass. A node refused keeps its sign, at most eps from zero and never deeper than it was.
// 11-10 (a pinhole the guard kept open on a thin fold after smoothing; holes are defects to fix, never to keep):
// counts t26 (26-connected inside components among the 26 neighbours) and t6 (6-connected outside components within
// N18 that touch a face neighbour), each capped at 3. A node may gain material when the inside around it is ONE piece
// (t26 == 1: it may close a pinhole, a channel or a cavity, never bridge two pieces); it may lose material when the
// outside around it is one piece and the inside at most one (t6 == 1, t26 <= 1: no tear, no hole punched through).
__device__ void topo_counts(const bool* in, int* t26o, int* t6o) {
    int lab[27], st[27];
    for (int c = 0; c < 27; c++) lab[c] = 0;
    int ncomp = 0;
    for (int s = 0; s < 27 && ncomp < 3; s++) {
        if (s == 13 || !in[s] || lab[s]) continue;
        ncomp++;
        int sp = 0; st[sp++] = s; lab[s] = ncomp;
        while (sp) {
            int c = st[--sp]; int cx = c / 9, cy = (c / 3) % 3, cz = c % 3;
            for (int d = 0; d < 27; d++) {
                if (d == 13 || !in[d] || lab[d]) continue;
                int dx = d / 9, dy = (d / 3) % 3, dz = d % 3;
                if (abs(dx - cx) <= 1 && abs(dy - cy) <= 1 && abs(dz - cz) <= 1) { lab[d] = ncomp; st[sp++] = d; }
            }
        }
    }
    *t26o = ncomp;
    for (int c = 0; c < 27; c++) lab[c] = 0;
    int nb6 = 0;
    for (int s = 0; s < 27 && nb6 < 3; s++) {
        int sx = s / 9 - 1, sy = (s / 3) % 3 - 1, sz = s % 3 - 1;
        if (s == 13 || in[s] || lab[s] || abs(sx) + abs(sy) + abs(sz) != 1) continue;
        nb6++;
        int sp = 0; st[sp++] = s; lab[s] = 1;
        while (sp) {
            int c = st[--sp]; int cx = c / 9, cy = (c / 3) % 3, cz = c % 3;
            const int D[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
            for (int q = 0; q < 6; q++) {
                int nx = cx + D[q][0], ny = cy + D[q][1], nz = cz + D[q][2];
                if (nx < 0 || ny < 0 || nz < 0 || nx > 2 || ny > 2 || nz > 2) continue;
                if (abs(nx - 1) + abs(ny - 1) + abs(nz - 1) > 2) continue;
                int d = nx * 9 + ny * 3 + nz;
                if (d == 13 || in[d] || lab[d]) continue;
                lab[d] = 1; st[sp++] = d;
            }
        }
    }
    *t6o = nb6;
}

__device__ bool flip_ok(const bool* in, bool to_inside) {
    int t26, t6;
    topo_counts(in, &t26, &t6);
    return to_inside ? (t26 == 1) : (t6 == 1 && t26 <= 1);
}

extern "C" __global__ void dab_topo_commit(const float* vft, const int* vit, const int* grid, const int* ttab, __half* sdf,
                                           int vol, int lev, const int* rows, const float* org, int nb,
                                           float* out, int colour, float eps, int final, int* acc) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    const float* vf = vft + vol * VF;
    const int* vi = vit + vol * VI;
    float h = vf[VF_H + lev];
    float x = org[3 * b] + i * h, y = org[3 * b + 1] + j * h, z = org[3 * b + 2] + k * h;
    int I = (int)floorf((x - vf[VF_O]) / h + 0.5f), J = (int)floorf((y - vf[VF_O + 1]) / h + 0.5f), K = (int)floorf((z - vf[VF_O + 2]) / h + 0.5f);
    if (((I & 1) | ((J & 1) << 1) | ((K & 1) << 2)) != colour) return;
    size_t a = (size_t)rows[b] * 729 + q;
    float fc = __half2float(sdf[a]), fn = out[id];
    if ((fc < 0.0f) != (fn < 0.0f) && isfinite(fc)) {
        bool in[27];
        bool ok = true;
        for (int c = 0; c < 27; c++) {
            if (c == 13) { in[c] = fn < 0.0f; continue; }
            float v = obj_f(vf, vi, grid, ttab, sdf, lev, x + (c / 9 - 1) * h, y + ((c / 3) % 3 - 1) * h, z + (c % 3 - 1) * h, 1e30f);
            if (fabsf(v) > 1e29f) { ok = false; break; }        // no information: keep the sign
            in[c] = v < 0.0f;
        }
        if (!ok || !flip_ok(in, fn < 0.0f)) {
            if (!final) return;                                 // pending: a later sweep may accept it
            fn = fc;            // 11-10: the node keeps its value (an eps clamp left sub-cell membranes: flakes on her)
            out[id] = fn;
        } else atomicAdd(acc, 1);
    }
    sdf[a] = __float2half_rn(fn);
}

// 11-10 (acceptance S7: Grab 21.9 -> 7-8 fps with the guard over every node): the colour sweeps over the FLIPPING nodes
// only (idx: their positions in the dab's rows x 729; the rest committed at once by dab_commit beforehand)
extern "C" __global__ void dab_topo_list(const float* vft, const int* vit, const int* grid, const int* ttab, __half* sdf,
                                         int vol, int lev, const int* rows, const float* org, const long long* idx,
                                         int nidx, const float* out, int colour, int* done, int* acc) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nidx || done[t]) return;
    size_t id = (size_t)idx[t];
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    const float* vf = vft + vol * VF;
    const int* vi = vit + vol * VI;
    float h = vf[VF_H + lev];
    float x = org[3 * b] + i * h, y = org[3 * b + 1] + j * h, z = org[3 * b + 2] + k * h;
    int I = (int)floorf((x - vf[VF_O]) / h + 0.5f), J = (int)floorf((y - vf[VF_O + 1]) / h + 0.5f), K = (int)floorf((z - vf[VF_O + 2]) / h + 0.5f);
    if (((I & 1) | ((J & 1) << 1) | ((K & 1) << 2)) != colour) return;
    size_t a = (size_t)rows[b] * 729 + q;
    float fc = __half2float(sdf[a]), fn = out[id];
    bool in[27];
    for (int c = 0; c < 27; c++) {
        if (c == 13) { in[c] = fn < 0.0f; continue; }
        float v = obj_f(vf, vi, grid, ttab, sdf, lev, x + (c / 9 - 1) * h, y + ((c / 3) % 3 - 1) * h, z + (c % 3 - 1) * h, 1e30f);
        if (fabsf(v) > 1e29f) return;                      // no information: the node keeps its value
        in[c] = v < 0.0f;
    }
    if (!flip_ok(in, fn < 0.0f)) return;                   // refused for now: kept (a later sweep may accept it)
    sdf[a] = __float2half_rn(fn);
    done[t] = 1;
    atomicAdd(acc, 1);
}

// 11-10: one pass: a node keeping its sign is committed, a flipping one is listed for the sweeps (dab_topo_list)
extern "C" __global__ void dab_commit_split(__half* sdf, const int* rows, int nb, const float* vals, long long* idx,
                                            int* count) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    size_t a = (size_t)rows[id / 729] * 729 + id % 729;
    float fc = __half2float(sdf[a]), fn = vals[id];
    if ((fc < 0.0f) == (fn < 0.0f)) { sdf[a] = __float2half_rn(fn); return; }
    int k = atomicAdd(count, 1);
    idx[k] = (long long)id;
}

// ---- fs-brush 11-11: the live re-distance of each batch's dab footprint (a closest-point transform) ------------------
// Why (M, test 1): displacement brushes compose F(p - v w) dab after dab; Grab compressed the field ahead of the pull to
// |grad F| 5-10 (F read 20 mm 4 mm from the neck's surface), so the tracer's sphere steps (which take F as a distance)
// jumped over the surface (the "torn open ring"), and every later brush moved that surface by A / |grad F| instead of A.
// The release re-distance worked in an F-valued band (|F| < band: nodes 2 mm from the surface reading 20 mm were never
// corrected) and ramped to the old values at its region of bricks' edge, which left brick seams (blocky patches) where
// that region met bricks with no sculpt layer (2 915 seam nodes after one neck Grab; 0 without it).
// Here, after each batch of dabs: seeds at the interface nodes (a 6-neighbour of the other sign): the Newton foot
// x - F grad F / |grad F|^2 (central differences of the nodes), so the zero set is kept to second order; jump flooding
// carries the nearest foot to every node of the batch's bricks (geometric, whatever F reads); the new value
// sign(F) |x - foot| is blended in by the depth inside the batch's dab spheres (1 from a quarter radius in, 0 at their
// edge), so where the dabs changed nothing, nothing changes, and no seam can form at any region edge. Signs never
// change (no topology change). Node coordinates: integer node indices of the volume (I = key * 8 + i).
__device__ __forceinline__ float node_val(const __half* sdf, const int* rowmap, int nbx, int nby, int nbz, int I, int J, int K, bool* ok) {
    int r = node_row(rowmap, nbx, nby, nbz, I, J, K);
    if (r < 0) { *ok = false; return 0.0f; }
    *ok = true;
    return __half2float(sdf[(size_t)r * 729 + ((I & 7) * 9 + (J & 7)) * 9 + (K & 7)]);
}

extern "C" __global__ void rd_seed(const __half* sdf, const int* rowmap, int nbx, int nby, int nbz,
                                   const int* rows, const int* keys, int nb, float4* seed) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
    float f0 = __half2float(sdf[(size_t)rows[b] * 729 + (i * 9 + j) * 9 + k]);
    seed[id] = make_float4(0, 0, 0, 0);
    if (!isfinite(f0)) return;
    const int D[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float g[3], best = 1e30f;
    bool iface = (f0 == 0.0f);
    for (int a = 0; a < 3; a++) {
        bool okp, okm;
        float fp = node_val(sdf, rowmap, nbx, nby, nbz, I + D[a][0], J + D[a][1], K + D[a][2], &okp);
        float fm = node_val(sdf, rowmap, nbx, nby, nbz, I - D[a][0], J - D[a][1], K - D[a][2], &okm);
        okp = okp && isfinite(fp); okm = okm && isfinite(fm);
        if (okp && (fp < 0.0f) != (f0 < 0.0f)) { iface = true; best = fminf(best, f0 / (f0 - fp)); }
        if (okm && (fm < 0.0f) != (f0 < 0.0f)) { iface = true; best = fminf(best, f0 / (f0 - fm)); }
        g[a] = (okp && okm) ? 0.5f * (fp - fm) : okp ? (fp - f0) : okm ? (f0 - fm) : 0.0f;   // per node step
    }
    if (!iface) return;
    float gl = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);    // |grad F| x h
    if (!(gl > 1e-12f)) return;
    // the distance in nodes: |F| / |grad F|, capped by the nearest crossing on an axis (a point of the zero set)
    float u = fminf(fabsf(f0) / gl, best);
    float s = f0 < 0.0f ? -1.0f : 1.0f;
    seed[id] = make_float4(I - s * u * g[0] / gl, J - s * u * g[1] / gl, K - s * u * g[2] / gl, 1.0f);
}

extern "C" __global__ void rd_apply(__half* sdf, const int* rows, const int* keys, int nb, float h, const float* org0,
                                    const float* dabs, int nd, const float4* seed, int* stats,
                                    const __half* mpool, const int* mslot, int use_mask) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
    size_t a = (size_t)rows[b] * 729 + (i * 9 + j) * 9 + k;
    float f0 = __half2float(sdf[a]);
    if (!isfinite(f0)) return;
    float4 s = seed[id];
    if (s.w < 0.5f) { atomicAdd(stats, 1); return; }      // no surface reached from here: kept
    float x = org0[0] + I * h, y = org0[1] + J * h, z = org0[2] + K * h;
    float depth = -1e30f, rr = h;
    for (int d = 0; d < nd; d++) {
        float dx = x - dabs[4 * d], dy = y - dabs[4 * d + 1], dz = z - dabs[4 * d + 2];
        float dep = dabs[4 * d + 3] - sqrtf(dx * dx + dy * dy + dz * dz);
        if (dep / fmaxf(dabs[4 * d + 3], 1e-9f) > depth / fmaxf(rr, 1e-9f)) { depth = dep; rr = dabs[4 * d + 3]; }
    }
    float w = fminf(fmaxf(depth / (0.25f * rr), 0.0f), 1.0f);
    if (use_mask) { int ms = mslot[rows[b]]; if (ms >= 0) w *= 1.0f - fminf(fmaxf(__half2float(mpool[(size_t)ms * 729 + (i * 9 + j) * 9 + k]), 0.0f), 1.0f); }
    if (w <= 0.0f) return;
    float dist = h * sqrtf((s.x - I) * (s.x - I) + (s.y - J) * (s.y - J) + (s.z - K) * (s.z - K));
    float Dv = f0 < 0.0f ? -dist : dist;
    float fn = f0 + w * (Dv - f0);
    if (f0 != 0.0f && (fn < 0.0f) != (f0 < 0.0f)) fn = f0 < 0.0f ? -1e-7f : 1e-7f;   // never a sign change
    sdf[a] = __float2half_rn(fn);
    atomicAdd(stats + 1, 1);
}


// ---- fs-brush 11-11: the area plane of Flatten and Clay (fs-bench scorecard: Flatten ADDED material on any convex form,
// +8 mm on the bust, and Clay ribbed 65-89 % along a stroke, because both took their plane from the cursor's tangent
// plane: on a convex form that plane lies above everything else under the brush). As Blender's area plane: the
// falloff-weighted mean position and mean normal of the surface under the brush. 256 samples on a disc of radius
// 0.8 r about the dab centre in the plane across the cursor normal, each moved onto the surface by three Newton steps
// on the card's field; the result goes into bp[29..34] (point, normal) on the card, so the dab reads it with no sync.
__device__ __forceinline__ float lut_wq(const float* lut, int n, float u) {
    if (u >= 1.0f) return 0.0f;
    float x = u * (n - 1);
    int i = (int)x;
    if (i >= n - 1) return lut[n - 1];
    float t = x - i;
    return lut[i] + (lut[i + 1] - lut[i]) * t;
}

extern "C" __global__ void area_plane(const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                      int vol, int lev, float* bp, const float* lut, int nlut) {
    __shared__ float acc[256][7];
    int t = threadIdx.x;
    const float* vf = vft + vol * VF;
    const int* vi = vit + vol * VI;
    float h = vf[VF_H + lev];
    float cx = bp[1], cy = bp[2], cz = bp[3], nx = bp[4], ny = bp[5], nz = bp[6], r = bp[7];
    float ax = fabsf(nx) < 0.9f ? 1.f : 0.f, ay = ax > 0.5f ? 0.f : 1.f;
    float t1x = -nz * ay, t1y = nz * ax, t1z = nx * ay - ny * ax;
    float tl = sqrtf(t1x * t1x + t1y * t1y + t1z * t1z); t1x /= tl; t1y /= tl; t1z /= tl;
    float t2x = ny * t1z - nz * t1y, t2y = nz * t1x - nx * t1z, t2z = nx * t1y - ny * t1x;
    // a sunflower disc: even area coverage
    float rho = 0.8f * r * sqrtf((t + 0.5f) / 256.0f), th = t * 2.39996323f;
    float px = cx + rho * (cosf(th) * t1x + sinf(th) * t2x), py = cy + rho * (cosf(th) * t1y + sinf(th) * t2y),
          pz = cz + rho * (cosf(th) * t1z + sinf(th) * t2z);
    float wsum = 0.f, gx = 0.f, gy = 0.f, gz = 0.f;
    bool ok = true;
    for (int it = 0; it < 4 && ok; it++) {
        float f = obj_f(vf, vi, grid, ttab, sdf, lev, px, py, pz, 0.f);
        float a = obj_f(vf, vi, grid, ttab, sdf, lev, px + h, py, pz, 0.f), b = obj_f(vf, vi, grid, ttab, sdf, lev, px - h, py, pz, 0.f);
        float c = obj_f(vf, vi, grid, ttab, sdf, lev, px, py + h, pz, 0.f), d = obj_f(vf, vi, grid, ttab, sdf, lev, px, py - h, pz, 0.f);
        float e = obj_f(vf, vi, grid, ttab, sdf, lev, px, py, pz + h, 0.f), g = obj_f(vf, vi, grid, ttab, sdf, lev, px, py, pz - h, 0.f);
        if (fabsf(f) > 1e29f || fabsf(a) > 1e29f || fabsf(b) > 1e29f || fabsf(c) > 1e29f || fabsf(d) > 1e29f || fabsf(e) > 1e29f || fabsf(g) > 1e29f) { ok = false; break; }
        gx = (a - b) / (2.f * h); gy = (c - d) / (2.f * h); gz = (e - g) / (2.f * h);
        float g2 = gx * gx + gy * gy + gz * gz;
        if (g2 < 1e-8f) { ok = false; break; }
        if (it == 3) { ok = fabsf(f) < 0.25f * h; break; }
        px -= f * gx / g2; py -= f * gy / g2; pz -= f * gz / g2;
    }
    float w = 0.f;
    if (ok) {
        float dx = px - cx, dy = py - cy, dz = pz - cz;
        float gl = sqrtf(gx * gx + gy * gy + gz * gz);
        if (gx * nx + gy * ny + gz * nz > 0.0f) {       // the brushed side only (a facing sheet is not this surface)
            w = lut_wq(lut, nlut, sqrtf(dx * dx + dy * dy + dz * dz) / r);
            gx /= gl; gy /= gl; gz /= gl;
        }
    }
    acc[t][0] = w * px; acc[t][1] = w * py; acc[t][2] = w * pz;
    acc[t][3] = w * gx; acc[t][4] = w * gy; acc[t][5] = w * gz; acc[t][6] = w;
    __syncthreads();
    for (int sft = 128; sft > 0; sft >>= 1) {
        if (t < sft) for (int k = 0; k < 7; k++) acc[t][k] += acc[t + sft][k];
        __syncthreads();
    }
    if (t == 0) {
        if (acc[0][6] > 1e-6f) {
            float W = acc[0][6];
            float mx = acc[0][3], my = acc[0][4], mz = acc[0][5], ml = sqrtf(mx * mx + my * my + mz * mz);
            bp[29] = acc[0][0] / W; bp[30] = acc[0][1] / W; bp[31] = acc[0][2] / W;
            if (ml > 1e-6f) { bp[32] = mx / ml; bp[33] = my / ml; bp[34] = mz / ml; }
            else { bp[32] = nx; bp[33] = ny; bp[34] = nz; }
        } else {                                         // no surface found under the brush: the cursor's plane
            bp[29] = cx; bp[30] = cy; bp[31] = cz; bp[32] = nx; bp[33] = ny; bp[34] = nz;
        }
    }
}


// fs-brush 11-11: min |F| over each row's own 8^3 nodes (the live re-distance and the apron refresh then run only on the
// bricks near the surface: M, at r 60-80 they took 9-15 and 8-10 ms a batch over every brick of the brush's path)
extern "C" __global__ void row_minabs(const __half* sdf, const int* rows, int nb, float* out) {
    // one block of 256 threads per row (an atomic per node contended 512 ways on one address: M, 2x slower batches)
    __shared__ float m[256];
    int b = blockIdx.x, t = threadIdx.x;
    if (b >= nb) return;
    const __half* B = sdf + (size_t)rows[b] * 729;
    float v = 1e30f;
    for (int q = t; q < 512; q += 256) {
        int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
        float f = fabsf(__half2float(B[(i * 9 + j) * 9 + k]));
        if (f < v) v = f;
    }
    m[t] = v;
    __syncthreads();
    for (int s2 = 128; s2 > 0; s2 >>= 1) {
        if (t < s2 && m[t + s2] < m[t]) m[t] = m[t + s2];
        __syncthreads();
    }
    if (t == 0) out[b] = m[0];
}


// fs-brush 11-11: each row's smallest and largest finite node over its 9^3 (aprons included): a row holding both signs
// holds the surface (the re-distance's reach starts there; gpu/sculpt.py RD_REACH). One block of 256 threads per row.
extern "C" __global__ void row_minmax(const __half* sdf, const int* rows, int nb, float* omin, float* omax) {
    __shared__ float a[256], z[256];
    int b = blockIdx.x, t = threadIdx.x;
    if (b >= nb) return;
    const __half* B = sdf + (size_t)rows[b] * 729;
    float lo = 1e30f, hi = -1e30f;
    for (int q = t; q < 729; q += 256) {
        float f = __half2float(B[q]);
        if (!isfinite(f)) continue;
        lo = fminf(lo, f); hi = fmaxf(hi, f);
    }
    a[t] = lo; z[t] = hi;
    __syncthreads();
    for (int s2 = 128; s2 > 0; s2 >>= 1) {
        if (t < s2) { a[t] = fminf(a[t], a[t + s2]); z[t] = fmaxf(z[t], z[t + s2]); }
        __syncthreads();
    }
    if (t == 0) { omin[b] = a[0]; omax[b] = z[0]; }
}

// fs-brush 11-11: the batch-level keep-apart guard. Rows' values copied (fp16) and restored, so the guard can judge the
// batch's NET sign changes once (dab_commit_split + dab_topo_list), instead of after every dab (a split, a host sync and
// 2-3 colour sweeps per dab: M, 4-6 dabs a frame in the head scene)
// fs-brush 11-12: the sheet under the brush: nodes another sheet owns get their stroke-start values back after the live
// re-distance (keep[id] != 0)
extern "C" __global__ void rows_restore_masked(__half* sdf, const int* rows, int nb, const float* before,
                                               const unsigned char* keep) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    if (keep[id]) sdf[(size_t)rows[id / 729] * 729 + id % 729] = __float2half_rn(before[id]);
}

extern "C" __global__ void rows_restore(__half* sdf, const int* rows, int nb, const __half* buf) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 729) return;
    sdf[(size_t)rows[id / 729] * 729 + id % 729] = buf[id];
}


// fs-brush 11-11: the apron refresh after a batch, all on the card: for each changed brick K and each of the 8 bricks
// K + (0|-1, 0|-1, 0|-1) whose aprons copy K's nodes, the 217 apron nodes (an index 8 on some axis) copied from their
// owner. Bricks found through the volume's row map (cell -> pool row); duplicates write the same values. Replaces the
// host's unique over 8 x the bricks and its row look-ups (M: 4-7 ms a batch in the head scene).
extern "C" __global__ void apron_fill_nb(__half* sdf, const int* rowmap, int nbx, int nby, int nbz, const int* keys, int nb) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 8 * 217) return;
    int b = (int)(id / (8 * 217)), r = (int)(id % (8 * 217));
    int o = r / 217, a = r % 217;
    int i, j, k;
    if (a < 81) { i = 8; j = a / 9; k = a % 9; }
    else if (a < 153) { a -= 81; j = 8; i = a / 9; k = a % 9; }
    else { a -= 153; k = 8; i = a / 8; j = a % 8; }
    int bx = keys[3 * b] - (o & 1), by = keys[3 * b + 1] - ((o >> 1) & 1), bz = keys[3 * b + 2] - ((o >> 2) & 1);
    if (bx < 0 || by < 0 || bz < 0 || bx >= nbx || by >= nby || bz >= nbz) return;
    int row = rowmap[((size_t)bx * nby + by) * nbz + bz];
    if (row < 0) return;
    int ox = bx + (i == 8), oy = by + (j == 8), oz = bz + (k == 8);
    if (ox >= nbx || oy >= nby || oz >= nbz) return;
    int orow = rowmap[((size_t)ox * nby + oy) * nbz + oz];
    if (orow < 0) return;
    sdf[(size_t)row * 729 + (i * 9 + j) * 9 + k] = sdf[(size_t)orow * 729 + ((i & 7) * 9 + (j & 7)) * 9 + (k & 7)];
}


// ---- fs-brush 11-11 (M, fs-bench test 1 shots: the jump-flood re-distance left lattice-aligned ribs and a crumbly jaw
// edge; base code smooth): the live re-distance as core/brush.py sweep_distance's scheme, on the card. Seeds: the nodes
// with a 6-neighbour of the other sign, U = min(|F| / |grad F|, the nearest crossing on an axis) (the CPU's own seed);
// the rest by Jacobi iterations of the Godunov (upwind) update of |grad U| = 1 from their 6 neighbours, which is smooth
// node to node (the nearest-foot distance of the jump flood has kinks between feet). The interface nodes keep their own
// values: the trilinear zero set between them cannot move; only the field off the surface is rebuilt.
// U holds |distance| in metres; 1e30 = not reached.
// fs-core-2 11-14 (fs-core m1 / dbg6: Clay strokes piled on one spot): a surface node keeps its value only while that
// value can be a distance. Two things made kept values impossible and nothing ever corrected them, because the surface's
// own nodes are frozen in every batch: (1) the field at the surface steepened stroke by stroke (|grad F| 1.0 -> 1.4 -> 2.0
// over strokes 1-5, sheet on or off), and a seed with |grad F| = 2 hands the band a value twice its distance; (2) a dab
// wrote a node 3.35 mm inside beside a neighbour it did not write (another sheet, 0.02 mm outside): a 3.4 mm step across
// one 0.67 mm cell (the walls from stroke 6). A node whose central-difference slope exceeds gk (a distance has 1; a kink
// or crease has less) takes |F| / |grad F|, and one whose kept value exceeds gk times the axis crossing beside it takes that
// crossing's distance (|F| <= the crossing's distance holds for every exact distance), so the value never promises more
// room than the crossing the tracer draws next to it. On a violated edge both ends shrink by their own crossing, which is
// the same point: the zero set there stays where it is. gk <= 0: kept as before.
__device__ __forceinline__ float seed_fix(float af, float gl, float best, float h, float gk) {
    if (!(gk > 0.0f)) return af;
    float u = af;
    if (gl > gk * h) u = fminf(u, af * h / gl);
    if (best < 1e29f && af > gk * best * h) u = fminf(u, best * h);
    return u;
}
extern "C" __global__ void rd_eik_init(const __half* sdf, const int* rowmap, int nbx, int nby, int nbz,
                                       const int* rows, const int* keys, int nb, float h, float* U, unsigned char* iface,
                                       float* GL, int seed_kept, float gk) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
    float f0 = __half2float(sdf[(size_t)rows[b] * 729 + (i * 9 + j) * 9 + k]);
    U[id] = 1e30f; iface[id] = 0; GL[id] = 1.0f;
    if (!isfinite(f0)) return;
    const int D[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float g[3], best = 1e30f;
    bool fc = (f0 == 0.0f);
    for (int a = 0; a < 3; a++) {
        bool okp, okm;
        float fp = node_val(sdf, rowmap, nbx, nby, nbz, I + D[a][0], J + D[a][1], K + D[a][2], &okp);
        float fm = node_val(sdf, rowmap, nbx, nby, nbz, I - D[a][0], J - D[a][1], K - D[a][2], &okm);
        okp = okp && isfinite(fp); okm = okm && isfinite(fm);
        if (okp && (fp < 0.0f) != (f0 < 0.0f)) { fc = true; best = fminf(best, f0 / (f0 - fp)); }
        if (okm && (fm < 0.0f) != (f0 < 0.0f)) { fc = true; best = fminf(best, f0 / (f0 - fm)); }
        g[a] = (okp && okm) ? 0.5f * (fp - fm) : okp ? (fp - f0) : okm ? (f0 - fm) : 0.0f;
    }
    float gl = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);     // |grad F| x h
    if (!fc) {
        // a corner of a cell the surface crosses only diagonally: frozen too (its value shapes that cell's zero set)
        bool diag = false;
        for (int dx = -1; dx <= 1 && !diag; dx++) for (int dy = -1; dy <= 1 && !diag; dy++) for (int dz = -1; dz <= 1; dz++) {
            if (abs(dx) + abs(dy) + abs(dz) < 2) continue;
            bool ok;
            float fv = node_val(sdf, rowmap, nbx, nby, nbz, I + dx, J + dy, K + dz, &ok);
            if (ok && isfinite(fv) && (fv < 0.0f) != (f0 < 0.0f)) { diag = true; break; }
        }
        if (!diag) return;
        if (seed_kept) { U[id] = seed_fix(fabsf(f0), gl, 1e30f, h, gk); iface[id] = 1; GL[id] = gl; return; }
        if (!(gl > 1e-12f)) return;
        U[id] = fabsf(f0) / gl * h;
        iface[id] = 1; GL[id] = gl;
        return;
    }
    // fs-brush-3 11-13 (seed_kept, FSB_RD_SEED=kept; CPU model fs-brush/grain/rd_sim.py 'scale_keep'): the surface layer keeps
    // its values (frozen above), so the rebuilt layer must start from THOSE values, not from |F| / |grad F| or the nearest
    // axis crossing: seeding it from an estimate that differs from the kept value left a step between the two layers of
    // ~|F| x |1 / |grad F| - 1| per node, which varies node to node with each node's sub-cell offset and with the
    // gradient where a part is one or two cells thick (the grain; worse with brush size, where strokes stretch the field
    // more). Seeded from the kept values, the field is continuous across the layer (model: no roughness added at r 8, the
    // zero set fixed, ten repeats = one)
    if (seed_kept) { U[id] = seed_fix(fabsf(f0), gl, best, h, gk); iface[id] = 1; GL[id] = gl; return; }
    float u = gl > 1e-12f ? fminf(fabsf(f0) / gl, best) : best;    // in nodes
    U[id] = u * h;
    iface[id] = 1; GL[id] = gl;
}

extern "C" __global__ void rd_eik_step(const int* rowmap, int nbx, int nby, int nbz, const int* regmap,
                                       const int* keys, int nb, float h, const float* Uin, float* Uout,
                                       const unsigned char* iface, int* changed) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    float u0 = Uin[id];
    Uout[id] = u0;
    if (iface[id]) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
    float m[3];
    const int D[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int a = 0; a < 3; a++) {
        float best = 1e30f;
        for (int sg = -1; sg <= 1; sg += 2) {
            int I2 = I + sg * D[a][0], J2 = J + sg * D[a][1], K2 = K + sg * D[a][2];
            int r2 = node_row(rowmap, nbx, nby, nbz, I2, J2, K2);
            if (r2 < 0) continue;
            int g2 = regmap[r2];
            if (g2 < 0) continue;
            float v = Uin[(size_t)g2 * 512 + ((I2 & 7) * 8 + (J2 & 7)) * 8 + (K2 & 7)];
            best = fminf(best, v);
        }
        m[a] = best;
    }
    float a = m[0], bb = m[1], c = m[2], t;
    if (a > bb) { t = a; a = bb; bb = t; }
    if (bb > c) { t = bb; bb = c; c = t; }
    if (a > bb) { t = a; a = bb; bb = t; }
    if (a >= 1e29f) return;
    float u = a + h;
    if (u > bb) {
        u = 0.5f * (a + bb + sqrtf(fmaxf(2.0f * h * h - (a - bb) * (a - bb), 0.0f)));
        if (u > c) {
            float sm = a + bb + c, disc = sm * sm - 3.0f * (a * a + bb * bb + c * c - h * h);
            u = (sm + sqrtf(fmaxf(disc, 0.0f))) / 3.0f;
        }
    }
    if (u < u0 - 1e-4f * h) { Uout[id] = u; changed[0] = 1; }
}

extern "C" __global__ void rd_eik_apply(__half* sdf, const int* rows, const int* keys, int nb, float h, const float* org0,
                                        const float* dabs, int nd, const float* U, const unsigned char* iface, int* stats,
                                        const __half* mpool, const int* mslot, int use_mask, const float* GL, int iface_rw,
                                        const __half* pre, const int* preidx, int use_pre, float chg_eps) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
    size_t a = (size_t)rows[b] * 729 + (i * 9 + j) * 9 + k;
    float f0 = __half2float(sdf[a]);
    float u = U[id];
    if (!isfinite(f0) || u >= 1e29f) { if (u >= 1e29f) atomicAdd(stats, 1); return; }
    // the surface's own nodes keep their values. M (fs-bench test 1 shots): rewriting them each batch with their seed
    // (|F| / |grad F|, capped by the nearest crossing) flaked the surface (t1_fsbrush_7); rewriting only those whose
    // gradient left 0.8-1.25 was unstable: noise grew over the head (t1_fsbrush_8). Frozen, the cranium is smooth.
    // iface_rw (fs-brush 11-11, FSB_RD_RELEASE: once per stroke, at its release, never per batch): the surface's own nodes
    // take their seed too, so a stroke's warp of the field there does not carry into the next stroke
    if (iface[id] && !iface_rw) return;
    float x = org0[0] + I * h, y = org0[1] + J * h, z = org0[2] + K * h;
    float depth = -1e30f, rr = h;
    for (int d = 0; d < nd; d++) {
        float dx = x - dabs[4 * d], dy = y - dabs[4 * d + 1], dz = z - dabs[4 * d + 2];
        float dep = dabs[4 * d + 3] - sqrtf(dx * dx + dy * dy + dz * dz);
        if (dep / fmaxf(dabs[4 * d + 3], 1e-9f) > depth / fmaxf(rr, 1e-9f)) { depth = dep; rr = dabs[4 * d + 3]; }
    }
    float w = fminf(fmaxf(depth / (0.25f * rr), 0.0f), 1.0f);
    if (use_mask) { int ms = mslot[rows[b]]; if (ms >= 0) w *= 1.0f - fminf(fmaxf(__half2float(mpool[(size_t)ms * 729 + ((i * 9 + j) * 9 + k)]), 0.0f), 1.0f); }
    // fs-brush-2 11-13 (grain under every setting, worse at large scale or on
    // sensitive areas; fs-store-3's gate matrix named this re-distance: off, the frozen-beside-moved count fell 2-3x for
    // every tool and stopped growing with the radius): the weight above is the dab sphere's depth only, so in a brush's
    // outer ring, where the dab's own falloff had moved nothing, every node off the surface was rewritten with the
    // first-order eikonal distance while the surface's own nodes stayed exact: on a rim one or two cells thick that
    // rewrite differs node to node (the frozen-beside-moved grain the shading normals read). A node is now re-distanced
    // in proportion to how much this batch changed it (its value before the batch's dabs, pre): one the stroke did not
    // change keeps its exact value
    if (use_pre) {
        int pi = preidx[rows[b]];
        if (pi >= 0) {
            float fp = __half2float(pre[(size_t)pi * 729 + (i * 9 + j) * 9 + k]);
            w *= fminf(fabsf(f0 - fp) / chg_eps, 1.0f);
        }
    }
    if (w <= 0.0f) return;
    float Dv = f0 < 0.0f ? -u : u;
    float fn = f0 + w * (Dv - f0);
    if (f0 != 0.0f && (fn < 0.0f) != (f0 < 0.0f)) fn = f0 < 0.0f ? -1e-7f : 1e-7f;
    sdf[a] = __float2half_rn(fn);
    atomicAdd(stats + 1, 1);
}

// ---- fs-core 11-14: THE TRUNCATED DISTANCE BAND (gpu/sculpt.py TSDF). The batch's rows are every held row within 2 W of
// its balls; U = the eikonal distance from the surface's own nodes (rd_eik_init seed_kept + rd_eik_step to convergence
// within W). A node's nearest surface can only have moved if a ball lies nearer to it than its distance: such a node
// (nearer a ball than min(|F|, W)) takes min(U, W) (its seeds lie within 2 W of the balls: all present); every other node
// keeps its value, clamped to |F| <= W (its true distance is >= min(its old one, its distance to the balls)). The surface's
// own nodes keep their values (the zero set does not move); signs never change. dabs: (cx, cy, cz, r) local.
extern "C" __global__ void rd_tsdf_apply(__half* sdf, const int* rows, const int* keys, int nb, float h, const float* org0,
                                         const float* dabs, int nd, const float* U, const unsigned char* iface, float W,
                                         int* stats, float nb_tol) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)nb * 512) return;
    int b = (int)(id / 512), q = (int)(id % 512);
    int i = q >> 6, j = (q >> 3) & 7, k = q & 7;
    size_t a = (size_t)rows[b] * 729 + (i * 9 + j) * 9 + k;
    float f0 = __half2float(sdf[a]);
    if (!isfinite(f0)) return;
    float af = fabsf(f0), v = af;
    if (iface[id]) {
        v = fminf(fminf(af, U[id]), W);              // fs-core-2: U = the kept value, or its fix (seed_fix)
    } else {
        int I = keys[3 * b] * 8 + i, J = keys[3 * b + 1] * 8 + j, K = keys[3 * b + 2] * 8 + k;
        float x = org0[0] + I * h, y = org0[1] + J * h, z = org0[2] + K * h;
        float db = 1e30f;
        for (int d = 0; d < nd; d++) {
            float dx = x - dabs[4 * d], dy = y - dabs[4 * d + 1], dz = z - dabs[4 * d + 2];
            db = fminf(db, sqrtf(dx * dx + dy * dy + dz * dz) - dabs[4 * d + 3]);
        }
        if (db < fminf(af, W)) v = fminf(U[id], W);
        else v = fminf(af, W);
        // fs-core-2 11-14: a node beside a surface node seed_fix corrected (anywhere in the rows, not only near this
        // batch's balls) keeps no more than the distance rebuilt from the surface nodes: U is the converged (or, beyond
        // the iterations, a larger) eikonal value, so min() only lowers a value that promised more room than U. Kept
        // nodes of a healthy band differ from U by rounding only: nb_tol (0.05 h) leaves them as they are. nb_tol < 0: off
        if (nb_tol >= 0.0f && U[id] < v - nb_tol) v = U[id];
    }
    if (v == af) return;
    if (f0 == 0.0f) return;
    sdf[a] = __float2half_rn(f0 < 0.0f ? -v : v);
    atomicAdd(stats + 1, 1);
}

// fs-core 11-14: the sheet restore under the truncated band (gpu/sculpt.py TSDF). The band pass moves no surface (every
// surface's own nodes are frozen) and its values are true lower bounds; the stroke-start value of a node the restore
// protects is too, unless the worked sheet has since come nearer. A protected node takes the smaller magnitude of the two
// (same sign: the dabs never write a protected node): another sheet keeps its own exact values where it is still the
// nearest surface, and no stroke-start value promises more room than there now is (sh_restore put those back: the
// crossing stroke's stale layer, fs-core ab2_nosheet).
extern "C" __global__ void sh_restore_min(__half* sdf, const int* rows, int n, const __half* bdev, const int* bpos,
                                          const unsigned char* keep) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)n * 729) return;
    if (!keep[id]) return;
    size_t a = (size_t)rows[id / 729] * 729 + id % 729;
    __half hb = bdev[(size_t)bpos[id / 729] * 729 + id % 729];
    float fb = __half2float(hb), fn = __half2float(sdf[a]);
    if (!isfinite(fn) || (fb < 0.0f) != (fn < 0.0f)) { sdf[a] = hb; return; }
    if (fabsf(fb) < fabsf(fn)) sdf[a] = hb;
}

// ---- fs-brush 11-12: THE SHEET UNDER THE BRUSH, labelled on the card (core/brush.py _sheet_flood: the same rule) ----
// A batch's n rows (bricks): row b's 9^3 first-touch values at bdev[bpos[b] * 729 + q] (q = i * 81 + j * 9 + k), its key
// at bkey[3b..3b+2]; kbox: dense int32 over the batch's keys (dims kd, origin kmin) -> batch row or -1. Cells: 8^3 a row
// (c = ci * 64 + cj * 8 + ck), corners at the row's local nodes ci..ci+1 (its apron holds index 8). Global lattice
// indices are relative to the volume's origin; seeds (ns x 3, mm from the origin) and their facing normals snorm.
__device__ __forceinline__ int sh_row(const int* kbox, const int* kd, const int* kmin, int kx, int ky, int kz) {
    int a = kx - kmin[0], b = ky - kmin[1], c = kz - kmin[2];
    if (a < 0 || b < 0 || c < 0 || a >= kd[0] || b >= kd[1] || c >= kd[2]) return -1;
    return kbox[(a * kd[1] + b) * kd[2] + c];
}
__device__ __forceinline__ int fdiv8(int v) { return v >= 0 ? v / 8 : -((-v + 7) / 8); }
__device__ __forceinline__ bool sh_val(const __half* bdev, const int* bpos, const int* kbox, const int* kd, const int* kmin,
                                       int I, int J, int K, float* v) {
    int kx = fdiv8(I), ky = fdiv8(J), kz = fdiv8(K);
    int b = sh_row(kbox, kd, kmin, kx, ky, kz);
    if (b < 0) return false;
    *v = __half2float(bdev[(size_t)bpos[b] * 729 + (I - 8 * kx) * 81 + (J - 8 * ky) * 9 + (K - 8 * kz)]);
    return true;
}
// a cell's 8 corners (from its owning row) -> mixed signs; its (unnormalised) normal into g
__device__ __forceinline__ bool sh_cell(const __half* bdev, const int* bpos, int b, int ci, int cj, int ck, float* g) {
    const __half* r = bdev + (size_t)bpos[b] * 729;
    float v[8];
    bool neg = false, pos = false;
    for (int t = 0; t < 8; t++) {
        int a = ci + (t >> 2), bb = cj + ((t >> 1) & 1), c = ck + (t & 1);
        v[t] = __half2float(r[a * 81 + bb * 9 + c]);
        if (v[t] < 0.0f) neg = true; else pos = true;
    }
    if (g) {
        g[0] = (v[4] + v[5] + v[6] + v[7]) - (v[0] + v[1] + v[2] + v[3]);
        g[1] = (v[2] + v[3] + v[6] + v[7]) - (v[0] + v[1] + v[4] + v[5]);
        g[2] = (v[1] + v[3] + v[5] + v[7]) - (v[0] + v[2] + v[4] + v[6]);
    }
    return neg && pos;
}
// fs-core 11-14: the walk's facing cone applies behind the press point's tangent plane only (deeper than SH_BEHIND h): a
// thin part's back face lies behind its front and stays another sheet; a concave crease's far wall (strokes meeting a
// built-up form) faces back toward the brush too but lies level or in front, and is the same surface: the cone stopped the
// walk there, the dab filled against a protected wall and left specks in the crease (fs-core dbg6, m1 from stroke 6).
// SH_BEHIND < 0: the cone everywhere, as before
#ifndef SH_BEHIND
#define SH_BEHIND 0.5f
#endif
__device__ __forceinline__ bool sh_behind(const float* seeds, const float* snorm, int s, int I, int J, int K, float h) {
    if (SH_BEHIND < 0.0f) return true;
    float dep = ((I + 0.5f) * h - seeds[3 * s]) * snorm[3 * s] + ((J + 0.5f) * h - seeds[3 * s + 1]) * snorm[3 * s + 1]
              + ((K + 0.5f) * h - seeds[3 * s + 2]) * snorm[3 * s + 2];
    return dep < -SH_BEHIND * h;
}
__device__ __forceinline__ bool sh_inreach(const float* seeds, int ns, float x, float y, float z, float reach2) {
    for (int s = 0; s < ns; s++) {
        float dx = x - seeds[3 * s], dy = y - seeds[3 * s + 1], dz = z - seeds[3 * s + 2];
        if (dx * dx + dy * dy + dz * dz <= reach2) return true;
    }
    return false;
}

extern "C" __global__ void sh_cells(const __half* bdev, const int* bpos, int n, signed char* cmix) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)n * 512) return;
    int b = (int)(id / 512), c = (int)(id % 512);
    cmix[id] = sh_cell(bdev, bpos, b, c / 64, (c / 8) % 8, c % 8, nullptr) ? 1 : 0;
}

// one step of the walk: cells at step k reach their face neighbours that the surface crosses into (within the reach, not
// turned past the facing limit), which take step k + 1 and the walker's seed
extern "C" __global__ void sh_step(const __half* bdev, const int* bpos, const int* bkey, const int* kbox, const int* kd,
                                   const int* kmin, int n, const signed char* cmix, int* cstep, int* csid, int k,
                                   const float* seeds, const float* snorm, int ns, float h, float reach2, float cosc,
                                   int* changed) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)n * 512) return;
    if (cstep[id] != k) return;
    int b = (int)(id / 512), c = (int)(id % 512);
    int ci = c / 64, cj = (c / 8) % 8, ck = c % 8;
    int I = bkey[3 * b] * 8 + ci, J = bkey[3 * b + 1] * 8 + cj, K = bkey[3 * b + 2] * 8 + ck;
    int s0 = csid[id];
    const __half* r = bdev + (size_t)bpos[b] * 729;
    for (int d = 0; d < 6; d++) {
        int ax = d >> 1, sg = (d & 1) ? 1 : -1;
        int I2 = I + (ax == 0 ? sg : 0), J2 = J + (ax == 1 ? sg : 0), K2 = K + (ax == 2 ? sg : 0);
        // the shared face's four corners, all in this row (local indices up to 8)
        int fa = ax == 0 ? ci + (sg > 0 ? 1 : 0) : ci, fb = ax == 1 ? cj + (sg > 0 ? 1 : 0) : cj, fc = ax == 2 ? ck + (sg > 0 ? 1 : 0) : ck;
        bool neg = false, pos = false;
        for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) {
            int a = fa, bb = fb, cc = fc;
            if (ax == 0) { bb += u; cc += v; } else if (ax == 1) { a += u; cc += v; } else { a += u; bb += v; }
            float val = __half2float(r[a * 81 + bb * 9 + cc]);
            if (val < 0.0f) neg = true; else pos = true;
        }
        if (!(neg && pos)) continue;
        int kx = fdiv8(I2), ky = fdiv8(J2), kz = fdiv8(K2);
        int b2 = sh_row(kbox, kd, kmin, kx, ky, kz);
        if (b2 < 0) continue;
        size_t id2 = (size_t)b2 * 512 + (I2 - 8 * kx) * 64 + (J2 - 8 * ky) * 8 + (K2 - 8 * kz);
        if (!cmix[id2] || cstep[id2] != 0) continue;
        if (!sh_inreach(seeds, ns, (I2 + 0.5f) * h, (J2 + 0.5f) * h, (K2 + 0.5f) * h, reach2)) continue;
        if (cosc > -1.5f && sh_behind(seeds, snorm, s0, I2, J2, K2, h)) {
            float g[3];
            sh_cell(bdev, bpos, b2, I2 - 8 * kx, J2 - 8 * ky, K2 - 8 * kz, g);
            float gl = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            if (gl > 1e-12f && (g[0] * snorm[3 * s0] + g[1] * snorm[3 * s0 + 1] + g[2] * snorm[3 * s0 + 2]) / gl < cosc) continue;
        }
        if (atomicCAS(&cstep[id2], 0, k + 1) == 0) { csid[id2] = s0; *changed = 1; }
    }
}

// every node of the batch's rows: in the reach, the label of the surface cell nearest its foot (x - F grad F / |grad F|):
// on the walked sheet -> the dabs may change it (smask 1); otherwise not, and after the live re-distance it gets its
// first-touch value back (keep 1) unless an earlier batch of this stroke had it on the sheet (ever_on)
extern "C" __global__ void sh_label(const __half* bdev, const int* bpos, const int* bkey, const int* kbox, const int* kd,
                                    const int* kmin, int n, const signed char* cmix, const int* cstep,
                                    const float* seeds, int ns, float h, float reach2, float inner2, float gmin,
                                    unsigned char* smask, unsigned char* keep, unsigned char* ever_on,
                                    int side, float bandw, unsigned char* labo) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)n * 729) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int i = q / 81, j = (q / 9) % 9, k = q % 9;
    int I = bkey[3 * b] * 8 + i, J = bkey[3 * b + 1] * 8 + j, K = bkey[3 * b + 2] * 8 + k;
    size_t eo = (size_t)bpos[b] * 729 + q;
    float f = __half2float(bdev[eo]);
    if (!sh_inreach(seeds, ns, I * h, J * h, K * h, reach2)) {
        smask[id] = 1;                               // beyond every dab's reach: nothing here is written
        keep[id] = ever_on[eo] ? 0 : 1;              // what a re-distance touched here gets its value back
        return;
    }
    // fs-brush 11-12 (t1_fsbrush_18s: a speck on each carved socket floor with the gate on): a node that was DEEP when the
    // stroke first touched it, on the side the dab moves into (side 1: a growing dab, outside; -1: a cutting dab,
    // inside), is where the worked surface goes: it belongs to this sheet. Its stored value may be stale (the deep
    // interior is re-distanced later), so its foot is not read from it. Other surfaces stay protected by their own
    // near nodes, which never move.
    if (side != 0 && fabsf(f) >= bandw && ((side > 0 && f > 0.0f) || (side < 0 && f < 0.0f))) {
        ever_on[eo] = 1; smask[id] = 1; keep[id] = 0; labo[id] = 1;
        return;
    }
    float vp, vm, g[3];
    for (int ax = 0; ax < 3; ax++) {
        int dI = ax == 0, dJ = ax == 1, dK = ax == 2;
        bool hp = sh_val(bdev, bpos, kbox, kd, kmin, I + dI, J + dJ, K + dK, &vp);
        bool hm = sh_val(bdev, bpos, kbox, kd, kmin, I - dI, J - dJ, K - dK, &vm);
        if (hp && hm) g[ax] = (vp - vm) / (2.0f * h);
        else if (hp) g[ax] = (vp - f) / h;
        else if (hm) g[ax] = (f - vm) / h;
        else g[ax] = 0.0f;
    }
    float gl = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    int lab = 0;
    if (gl >= gmin) {
        float fx = I - f * g[0] / gl / h, fy = J - f * g[1] / gl / h, fz = K - f * g[2] / gl / h;
        int bx = (int)floorf(fx), by = (int)floorf(fy), bz = (int)floorf(fz);
        float best = 1e30f;
        int lx = 0, ly = 0, lz = 0;
        for (int da = -1; da <= 1; da++) for (int db = -1; db <= 1; db++) for (int dc = -1; dc <= 1; dc++) {
            int cx = bx + da, cy = by + db, cz = bz + dc;
            int kx = fdiv8(cx), ky = fdiv8(cy), kz = fdiv8(cz);
            int b2 = sh_row(kbox, kd, kmin, kx, ky, kz);
            if (b2 < 0) continue;
            size_t id2 = (size_t)b2 * 512 + (cx - 8 * kx) * 64 + (cy - 8 * ky) * 8 + (cz - 8 * kz);
            if (!cmix[id2]) continue;
            float dd = (cx + 0.5f - fx) * (cx + 0.5f - fx) + (cy + 0.5f - fy) * (cy + 0.5f - fy) + (cz + 0.5f - fz) * (cz + 0.5f - fz);
            if (dd < best) { best = dd; lab = cstep[id2] > 0 ? 1 : 2; lx = cx; ly = cy; lz = cz; }
        }
        if (lab == 2 && !sh_inreach(seeds, ns, (lx + 0.5f) * h, (ly + 0.5f) * h, (lz + 0.5f) * h, inner2)) lab = 4;
    }
    if (lab == 1) ever_on[eo] = 1;
    smask[id] = lab == 1 ? 1 : 0;
    keep[id] = (lab != 1 && !ever_on[eo]) ? 1 : 0;
    // 3: undecided (no reliable foot) near the surface: sh_fill0 may join it to the sheet; 4: its foot is another sheet
    // beyond the inner reach: never filled
    labo[id] = (unsigned char)(lab == 0 && fabsf(f) < bandw ? 3 : lab);
}

// fs-brush 11-12 (t1_fsbrush_18s: specks on carved socket floors with the gate on): an undecided node near the surface
// (its field had no reliable gradient: a stroke's own distortion) joins the worked sheet unless one of its 26
// neighbours belongs to another surface; in a thin gap or a wall's middle it borders one, so it stays put
extern "C" __global__ void sh_fill0(const int* bkey, const int* kbox, const int* kd, const int* kmin, int n, const int* bpos,
                                    const unsigned char* labo, unsigned char* smask, unsigned char* keep,
                                    unsigned char* ever_on) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)n * 729) return;
    if (labo[id] != 3) return;
    int b = (int)(id / 729), q = (int)(id % 729);
    int I = bkey[3 * b] * 8 + q / 81, J = bkey[3 * b + 1] * 8 + (q / 9) % 9, K = bkey[3 * b + 2] * 8 + q % 9;
    for (int da = -1; da <= 1; da++) for (int db = -1; db <= 1; db++) for (int dc = -1; dc <= 1; dc++) {
        int I2 = I + da, J2 = J + db, K2 = K + dc;
        int kx = fdiv8(I2), ky = fdiv8(J2), kz = fdiv8(K2);
        int b2 = sh_row(kbox, kd, kmin, kx, ky, kz);
        if (b2 < 0) continue;
        unsigned char ln = labo[(size_t)b2 * 729 + (I2 - 8 * kx) * 81 + (J2 - 8 * ky) * 9 + (K2 - 8 * kz)];
        if (ln == 2 || ln == 4) return;          // beside another sheet (labelled, or its foot there beyond the inner reach)
    }
    smask[id] = 1; keep[id] = 0; ever_on[(size_t)bpos[b] * 729 + q] = 1;
}

extern "C" __global__ void sh_restore(__half* sdf, const int* rows, int n, const __half* bdev, const int* bpos,
                                      const unsigned char* keep) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)n * 729) return;
    if (keep[id]) sdf[(size_t)rows[id / 729] * 729 + id % 729] = bdev[(size_t)bpos[id / 729] * 729 + id % 729];
}

// the seed cell of each press point: the nearest surface along its brush axis (alternately out and in, half a cell a
// step) whose cell faces the brush; best[s] = the smallest (step * 8 + which of the 7 cells) found
__device__ __forceinline__ bool sh_seed_cell(const __half* bdev, const int* bpos, const int* kbox, const int* kd,
                                             const int* kmin, const float* seeds, const float* snorm, int s, int ts, int k,
                                             float h, int* bo, int* co) {
    float tt = (float)((ts + 1) / 2) * 0.5f * h * ((ts % 2 == 1) ? 1.0f : -1.0f);
    int cx = (int)floorf((seeds[3 * s] + snorm[3 * s] * tt) / h);
    int cy = (int)floorf((seeds[3 * s + 1] + snorm[3 * s + 1] * tt) / h);
    int cz = (int)floorf((seeds[3 * s + 2] + snorm[3 * s + 2] * tt) / h);
    if (k == 1) cx -= 1; else if (k == 2) cx += 1; else if (k == 3) cy -= 1; else if (k == 4) cy += 1;
    else if (k == 5) cz -= 1; else if (k == 6) cz += 1;
    int kx = fdiv8(cx), ky = fdiv8(cy), kz = fdiv8(cz);
    int b = sh_row(kbox, kd, kmin, kx, ky, kz);
    if (b < 0) return false;
    float g[3];
    if (!sh_cell(bdev, bpos, b, cx - 8 * kx, cy - 8 * ky, cz - 8 * kz, g)) return false;
    if (g[0] * snorm[3 * s] + g[1] * snorm[3 * s + 1] + g[2] * snorm[3 * s + 2] <= 0.0f) return false;
    *bo = b; *co = (cx - 8 * kx) * 64 + (cy - 8 * ky) * 8 + (cz - 8 * kz);
    return true;
}

extern "C" __global__ void sh_seed(const __half* bdev, const int* bpos, const int* kbox, const int* kd, const int* kmin,
                                   const float* seeds, const float* snorm, int ns, int nts, float h, unsigned int* best) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns * nts) return;
    int s = id / nts, ts = id % nts, b, c;
    for (int k = 0; k < 7; k++)
        if (sh_seed_cell(bdev, bpos, kbox, kd, kmin, seeds, snorm, s, ts, k, h, &b, &c)) {
            atomicMin(&best[s], (unsigned int)(ts * 8 + k));
            return;
        }
}

extern "C" __global__ void sh_seed_set(const __half* bdev, const int* bpos, const int* kbox, const int* kd, const int* kmin,
                                       const float* seeds, const float* snorm, int ns, float h, const unsigned int* best,
                                       int* cstep, int* csid) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= ns || best[s] == 0xffffffffu) return;
    int b, c;
    if (sh_seed_cell(bdev, bpos, kbox, kd, kmin, seeds, snorm, s, (int)(best[s] / 8), (int)(best[s] % 8), h, &b, &c)) {
        cstep[(size_t)b * 512 + c] = 1;
        csid[(size_t)b * 512 + c] = s;
    }
}

// fs-brush-4 11-13: sh_step's step for ONE cell of the frontier (a cell at step k): the face neighbours it reaches take step
// k + 1 and its seed, and are appended to the next frontier list (fout, count nout). The rule is sh_step's.
__device__ __forceinline__ void sh_step_cell(const __half* bdev, const int* bpos, const int* bkey, const int* kbox,
                                             const int* kd, const int* kmin, const signed char* cmix, int* cstep, int* csid,
                                             int k, const float* seeds, const float* snorm, int ns, float h, float reach2,
                                             float cosc, size_t id, int* fout, int* nout, int cap) {
    int b = (int)(id / 512), c = (int)(id % 512);
    int ci = c / 64, cj = (c / 8) % 8, ck = c % 8;
    int I = bkey[3 * b] * 8 + ci, J = bkey[3 * b + 1] * 8 + cj, K = bkey[3 * b + 2] * 8 + ck;
    int s0 = csid[id];
    const __half* r = bdev + (size_t)bpos[b] * 729;
    for (int d = 0; d < 6; d++) {
        int ax = d >> 1, sg = (d & 1) ? 1 : -1;
        int I2 = I + (ax == 0 ? sg : 0), J2 = J + (ax == 1 ? sg : 0), K2 = K + (ax == 2 ? sg : 0);
        int fa = ax == 0 ? ci + (sg > 0 ? 1 : 0) : ci, fb = ax == 1 ? cj + (sg > 0 ? 1 : 0) : cj, fc = ax == 2 ? ck + (sg > 0 ? 1 : 0) : ck;
        bool neg = false, pos = false;
        for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) {
            int a = fa, bb = fb, cc = fc;
            if (ax == 0) { bb += u; cc += v; } else if (ax == 1) { a += u; cc += v; } else { a += u; bb += v; }
            float val = __half2float(r[a * 81 + bb * 9 + cc]);
            if (val < 0.0f) neg = true; else pos = true;
        }
        if (!(neg && pos)) continue;
        int kx = fdiv8(I2), ky = fdiv8(J2), kz = fdiv8(K2);
        int b2 = sh_row(kbox, kd, kmin, kx, ky, kz);
        if (b2 < 0) continue;
        size_t id2 = (size_t)b2 * 512 + (I2 - 8 * kx) * 64 + (J2 - 8 * ky) * 8 + (K2 - 8 * kz);
        if (!cmix[id2] || cstep[id2] != 0) continue;
        if (!sh_inreach(seeds, ns, (I2 + 0.5f) * h, (J2 + 0.5f) * h, (K2 + 0.5f) * h, reach2)) continue;
        if (cosc > -1.5f && sh_behind(seeds, snorm, s0, I2, J2, K2, h)) {
            float g[3];
            sh_cell(bdev, bpos, b2, I2 - 8 * kx, J2 - 8 * ky, K2 - 8 * kz, g);
            float gl = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            if (gl > 1e-12f && (g[0] * snorm[3 * s0] + g[1] * snorm[3 * s0 + 1] + g[2] * snorm[3 * s0 + 2]) / gl < cosc) continue;
        }
        if (atomicCAS(&cstep[id2], 0, k + 1) == 0) {
            csid[id2] = s0;
            int o = atomicAdd(nout, 1);
            if (o < cap) fout[o] = (int)id2;
        }
    }
}

// the seed cells as sh_seed_set, each cell entered once (the first seed to reach it), into the first frontier list
extern "C" __global__ void sh_seed_set_f(const __half* bdev, const int* bpos, const int* kbox, const int* kd, const int* kmin,
                                         const float* seeds, const float* snorm, int ns, float h, const unsigned int* best,
                                         int* cstep, int* csid, int* fout, int* nout) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= ns || best[s] == 0xffffffffu) return;
    int b, c;
    if (sh_seed_cell(bdev, bpos, kbox, kd, kmin, seeds, snorm, s, (int)(best[s] / 8), (int)(best[s] % 8), h, &b, &c)) {
        size_t id = (size_t)b * 512 + c;
        if (atomicCAS(&cstep[id], 0, 1) == 0) { csid[id] = s; fout[atomicAdd(nout, 1)] = (int)id; }
    }
}

// fs-brush-4 11-13 (M, speed4/sw_exact: sh_step and the frontier list cost the same, ~6-9 ms for a 160-step walk: each step
// is a separate launch, so the time is the launches' host overhead, ~40 us each, and the walk is up to 1.6 (r + 2 h) / h
// steps): the WHOLE walk in one launch by one block. The frontier is a few hundred cells a step (a ring round the press
// point), which one block covers in one pass; __syncthreads() separates the steps, so step k + 1 starts only after every
// cell of step k has entered its neighbours: the rule, the order and the labels are sh_step's. fa holds the seeds' list
// (count cnt[0]); cnt[2] = the last step taken.
extern "C" __global__ void sh_walk_1b(const __half* bdev, const int* bpos, const int* bkey, const int* kbox, const int* kd,
                                      const int* kmin, const signed char* cmix, int* cstep, int* csid, int geo,
                                      const float* seeds, const float* snorm, int ns, float h, float reach2, float cosc,
                                      int* fa, int* fb, int* cnt, int cap) {
    __shared__ int nf_s;
    int* fin = fa; int* fout = fb;
    int* nin = cnt; int* nout = cnt + 1;
    int k = 1;
    for (; k <= geo; k++) {
        if (threadIdx.x == 0) { nf_s = min(atomicAdd(nin, 0), cap); atomicExch(nout, 0); }
        __syncthreads();
        int nf = nf_s;
        if (nf == 0) break;
        for (int t = threadIdx.x; t < nf; t += blockDim.x)
            sh_step_cell(bdev, bpos, bkey, kbox, kd, kmin, cmix, cstep, csid, k, seeds, snorm, ns, h, reach2, cosc,
                         (size_t)fin[t], fout, nout, cap);
        __syncthreads();
        int* tp = fin; fin = fout; fout = tp;
        tp = nin; nin = nout; nout = tp;
    }
    if (threadIdx.x == 0) cnt[2] = k;
}
