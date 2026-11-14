// Field Studio GPU viewport (phase 0, 2026-11-08): sparse-brick field atlas, tracer, live light.
// No mesh anywhere: every value is a field sample (P14). The atlas uses the FV8 store layout (meta.json "format"):
// a 32 mm tile grid per volume, per level a tile table -> brick id, bricks of 9^3 fp16 samples (8 cells + apron),
// sample (a,b,c) at brick_origin + (a,b,c) h, trilinear inside a brick.
// Light is the FV8 live-light model's per-pixel terms (work/livelight/DESIGN.md section 1): sun visibility by a soft
// shadow march, sky light by stratified cosine hemisphere rays to a short reach (miss = sky radiance from the CC0
// noon_grass environment, hit = 0, no bounce), diffuse R/pi (E_sun vis wrap(n.L) + E_sky), accumulated over passes.
#include <cuda_fp16.h>

#define NLEV 4
// volume float table (VF floats per volume)
#define VF 48
#define VF_M 0        // 12: world -> local 3x4 (row-major rows of [R|t]), local = R p + t
#define VF_MI 12      // 12: local -> world 3x4
#define VF_S 24       // local units per world unit (uniform)
#define VF_O 25       // 3: tile grid origin (local)
#define VF_TM 28      // tile size (local m)
#define VF_H 29       // 4: h per level
#define VF_BAND 33    // 4: band half-width per level (lower bound of |F| where a tile has no brick)
#define VF_LO 37      // 3: local box min
#define VF_HI 40      // 3: local box max
#define VF_ALB 43     // 3: albedo (material 0); material 1 (lips) / 2 (eye) scaled from it
#define VF_VIS 46     // 1 visible
// volume int table (VI ints per volume)
#define VI 20
#define VI_GOFF 0     // offset of the tile grid in grid[] ([tx,ty,tz,level] -> slot or -1)
#define VI_NX 1
#define VI_NY 2
#define VI_NZ 3
#define VI_TOFF 4     // 4: offset of each level's tile table in ttab[]
#define VI_BPT 8      // 4: bricks per tile axis per level (0 = level absent)
#define VI_BOFF 12    // 4: offset added to that level's brick ids (global pool row)
#define VI_FOFF 16    // far field: offset in grid[] of the volume's distance-to-nearest-brick grid (-1: none)
#define VI_FLEV 17    // far field: the level whose brick cells it counts (the coarsest present)
#define VI_MASK 18    // milestone (11-08): index of a volume whose level-2 bricks mask this one (-1: none)
#define VI_MASKER 19  // 11-09: 1 = this volume masks another (its missing bricks are no |F| >= band bound)
__device__ const float* g_vft;   // the volume tables, for the mask lookup (set by scene.py after each upload)
__device__ const int* g_vit;
// fs-brush-4 11-13 (n3, the light rule: each object's own response precomputed once, the light reaching it live): the
// window's sky visibility kept per SURFACE CELL, view-independent (the point, its normal octant, the geometry within the
// sky reach). Every pixel and frame that sees a cell adds its rays until the cell holds g_sky_par[1]; then pixels read it.
// g_sky_mask = slots - 1 (0 = off: today's arithmetic). g_sky_par: [0] base cell (m), [1] rays to hold, [2] cell size in
// pixel footprints, [3] the pixel's angle (rad, set by scene.py per view size / fov), [4] the edit pad (m)
__device__ unsigned long long* g_sky_k;
__device__ float* g_sky_v;
__device__ unsigned int g_sky_mask;
__device__ float g_sky_par[8];
// n3b: each volume's key origin (world, fixed until the cache is cleared): the cell indices are counted from it, so a
// fine level's 16-bit range (+-32768 cells) covers the volume wherever it stands in the world
__device__ float g_sky_org[3 * 64];

#ifdef TRACE_DBG
struct Hit { float t; int vol; int lev; int nm; };   // nm: 11-09 B2 diagnosis (the near-miss return)
#else
struct Hit { float t; int vol; int lev; };
#endif

__device__ __forceinline__ float3 mk(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float3 add(float3 a, float3 b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 sub(float3 a, float3 b) { return mk(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 mul(float3 a, float s) { return mk(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 nrm(float3 a) { float l = rsqrtf(fmaxf(dot3(a, a), 1e-30f)); return mul(a, l); }

__device__ __forceinline__ float3 xf_p(const float* M, float3 p) {
    return mk(M[0] * p.x + M[1] * p.y + M[2] * p.z + M[3], M[4] * p.x + M[5] * p.y + M[6] * p.z + M[7],
              M[8] * p.x + M[9] * p.y + M[10] * p.z + M[11]);
}
__device__ __forceinline__ float3 xf_d(const float* M, float3 d) {
    return mk(M[0] * d.x + M[1] * d.y + M[2] * d.z, M[4] * d.x + M[5] * d.y + M[6] * d.z, M[8] * d.x + M[9] * d.y + M[10] * d.z);
}

// Ray / box in local units. Returns false when missed.
__device__ __forceinline__ bool box_clip(float3 o, float3 d, const float* vf, float& t0, float& t1) {
    float lo[3] = {vf[VF_LO], vf[VF_LO + 1], vf[VF_LO + 2]}, hi[3] = {vf[VF_HI], vf[VF_HI + 1], vf[VF_HI + 2]};
    float oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
    for (int a = 0; a < 3; a++) {
        if (fabsf(dd[a]) < 1e-12f) { if (oo[a] < lo[a] || oo[a] > hi[a]) return false; continue; }
        float inv = 1.0f / dd[a], ta = (lo[a] - oo[a]) * inv, tb = (hi[a] - oo[a]) * inv;
        if (ta > tb) { float s = ta; ta = tb; tb = s; }
        t0 = fmaxf(t0, ta); t1 = fminf(t1, tb);
    }
    return t0 <= t1;
}

// far field (r10): grid[VI_FOFF + cell] = floor(Euclidean distance, in brick cells of level VI_FLEV, from the cell's
// centre to the nearest cell holding a brick). Any point of this cell is then at least (d - sqrt 3) cells from any
// brick, and every zero crossing lies inside a brick: a lower bound of |F| (local units), 0 where unknown.
__device__ __forceinline__ float far_lb(const float* vf, const int* vi, const int* grid, float3 p) {
    if (vi[VI_FOFF] < 0) return 0.0f;
    int l = vi[VI_FLEV];
    float bm = 8.0f * vf[VF_H + l];
    int nx = vi[VI_NX] * vi[VI_BPT + l], ny = vi[VI_NY] * vi[VI_BPT + l], nz = vi[VI_NZ] * vi[VI_BPT + l];
    int i = (int)floorf((p.x - vf[VF_O]) / bm), j = (int)floorf((p.y - vf[VF_O + 1]) / bm), k = (int)floorf((p.z - vf[VF_O + 2]) / bm);
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return 0.0f;
    float d = (float)grid[vi[VI_FOFF] + ((size_t)i * ny + j) * nz + k];
    return fmaxf(d - 1.7321f, 0.0f) * bm;
}

// Field value of one volume at local point p, reading the finest level >= lmin that holds a brick there.
// Returns the value; *lev = the level read (-1: no brick at any level), *empty = 1 when the 32 mm tile holds no
// level at all (the caller steps to the tile's exit).
// The tile grid holds, per tile, a row address into the tile table, or -1 where no band bricks are stored at any
// level: that is true deep inside a solid as well as in open air, so -1 is not an occupancy or "empty" tag. Keep it
// so: no per-tile or per-brick occupancy, inside / outside or sign tags beside the finer entries.
// milestone (11-08): where the mask volume (an editable object's card volume of edited bricks, same local frame)
// holds a level-2 brick, this volume (her native store) has no surface: -> a safe step (the distance to that
// brick's nearest face, so a march cannot jump past the brick's end), else -1
__device__ __forceinline__ float mask_step(const int* vi, const int* grid, const int* ttab, float3 p) {
    int m = vi[VI_MASK];
    if (m < 0) return -1.0f;
    const float* mf = g_vft + m * VF;
    const int* mi = g_vit + m * VI;
    int bpt = mi[VI_BPT + 2];
    if (bpt == 0) return -1.0f;
    float bm = 8.0f * mf[VF_H + 2];
    float qx = (p.x - mf[VF_O]) / bm, qy = (p.y - mf[VF_O + 1]) / bm, qz = (p.z - mf[VF_O + 2]) / bm;
    int bx = (int)floorf(qx), by = (int)floorf(qy), bz = (int)floorf(qz);
    if (bx < 0 || by < 0 || bz < 0) return -1.0f;
    int tx = bx / bpt, ty = by / bpt, tz = bz / bpt;
    if (tx >= mi[VI_NX] || ty >= mi[VI_NY] || tz >= mi[VI_NZ]) return -1.0f;
    int slot = grid[mi[VI_GOFF] + ((tx * mi[VI_NY] + ty) * mi[VI_NZ] + tz) * NLEV + 2];
    if (slot < 0) return -1.0f;
    int bid = ttab[mi[VI_TOFF + 2] + slot * bpt * bpt * bpt + ((bx - tx * bpt) * bpt + (by - ty * bpt)) * bpt + (bz - tz * bpt)];
    if (bid < 0) return -1.0f;
    float fx = qx - bx, fy = qy - by, fz = qz - bz;
    float d = fminf(fminf(fminf(fx, 1.0f - fx), fminf(fy, 1.0f - fy)), fminf(fz, 1.0f - fz)) * bm;
    return fmaxf(d, 0.0f) + 2e-5f;
}

__device__ float vol_f0(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                        float3 p, int lmin, int* lev, int* empty) {
    {
        float ms = mask_step(vi, grid, ttab, p);
        if (ms >= 0.0f) { *lev = -2; *empty = 0; return ms; }   // -2: masked (a step limit, never a surface)
    }
    float tm = vf[VF_TM];
    float qx = (p.x - vf[VF_O]) / tm, qy = (p.y - vf[VF_O + 1]) / tm, qz = (p.z - vf[VF_O + 2]) / tm;
    int tx = (int)floorf(qx), ty = (int)floorf(qy), tz = (int)floorf(qz);
    *lev = -1; *empty = 1;
    // r46: a hit with no brick under it carries level -1; reading from level -1 took vi[VI_BPT - 1] (level 3's
    // tile-table offset) as bricks-per-tile and indexed far past the tables: CUDA illegal address after ~100-150
    // strokes in phase0/size300.py (the fhit read in trace_gbuf)
    lmin = max(lmin, 0);
    if (tx < 0 || ty < 0 || tz < 0 || tx >= vi[VI_NX] || ty >= vi[VI_NY] || tz >= vi[VI_NZ]) return 1e3f;
    const int* g = grid + vi[VI_GOFF] + ((tx * vi[VI_NY] + ty) * vi[VI_NZ] + tz) * NLEV;
    float lb = 0.0f;
    for (int l = lmin; l < NLEV; l++) {
        int bpt = vi[VI_BPT + l];
        if (bpt == 0) continue;
        int slot = g[l];
        if (slot < 0) continue;
        *empty = 0;
        float h = vf[VF_H + l], bm = 8.0f * h;
        float lx = (qx - tx) * tm / bm, ly = (qy - ty) * tm / bm, lz = (qz - tz) * tm / bm;
        int bi = min(max((int)floorf(lx), 0), bpt - 1), bj = min(max((int)floorf(ly), 0), bpt - 1), bk = min(max((int)floorf(lz), 0), bpt - 1);
        int bid = ttab[vi[VI_TOFF + l] + slot * bpt * bpt * bpt + (bi * bpt + bj) * bpt + bk];
        if (bid < 0) { lb = fmaxf(lb, vf[VF_BAND + l]); continue; }
        const __half* B = sdf + (size_t)(bid + vi[VI_BOFF + l]) * 729;
        float ux = (lx - bi) * 8.0f, uy = (ly - bj) * 8.0f, uz = (lz - bk) * 8.0f;
        int i = min(max((int)floorf(ux), 0), 7), j = min(max((int)floorf(uy), 0), 7), k = min(max((int)floorf(uz), 0), 7);
        float fx = ux - i, fy = uy - j, fz = uz - k;
        const __half* c = B + (i * 9 + j) * 9 + k;
        float v000 = __half2float(c[0]), v001 = __half2float(c[1]), v010 = __half2float(c[9]), v011 = __half2float(c[10]);
        float v100 = __half2float(c[81]), v101 = __half2float(c[82]), v110 = __half2float(c[90]), v111 = __half2float(c[91]);
        float a0 = v000 + (v100 - v000) * fx, a1 = v010 + (v110 - v010) * fx, a2 = v001 + (v101 - v001) * fx, a3 = v011 + (v111 - v011) * fx;
        float b0 = a0 + (a1 - a0) * fy, b1 = a2 + (a3 - a2) * fy;
        *lev = l;
        return b0 + (b1 - b0) * fz;
    }
    // no brick at any level >= lmin: |F| >= the band there, and >= the far field's distance to the nearest brick
    return fmaxf(fmaxf(lb, far_lb(vf, vi, grid, p)), 1e-4f);
}

// ---- fs-brush 11-13: THE MOVE'S LIVE VIEW (gpu/sculpt.py MOVE_VIEW) ----------------------------------------------------
// A block-out moves 60-150 mm brushes; writing an anchored move into the 1 mm lattice every frame
// costs the moved VOLUME ((r/h)^3: Stretch r 100 ran 8 fps, r 120 5-6 fps). While an anchored Grab / Stretch is dragged
// the lattice keeps the stroke-start field, and every read of the moved volume inside a move copy's ball is taken at the
// point's pre-image: the node walk of sculpt.cu typ 14 / 15, the same arithmetic, applied to the read point. The value is
// divided by a local bound of the walk's stretch (|grad F(pre(x))| <= |J_pre| |grad F|: a grid of bounds per copy,
// mv_lgrid), so a sphere-traced step never passes the moved surface. The cost is per pixel, whatever the radius; the
// release writes the lattice once (one batch, as the old path's last). g_mv_n = 0 (any module compiling this text that
// does not set it): nothing changes, bit for bit.
#define MV_MAXC 4
#define MV_H 64          // floats per copy: 0..50 the dab's bp (brush.NBP 51), then:
#define MVH_LO 52        // 3: the bound grid's origin (local)
#define MVH_CS 55        // its cell size
#define MVH_NL 56        // its cells per axis
#define MVH_LOFF 57      // its offset in g_mv_L
#define MVH_MN 58        // 3: Grab's scope grid dims (0: none)
__device__ int g_mv_n;                    // active copies (0: off)
__device__ int g_mv_goff;                 // vi[VI_GOFF] of the moved volume (its identity in any copy of the tables)
__device__ const float* g_mv_hdr;         // MV_MAXC x MV_H
__device__ const float* g_mv_L;           // the bound grids
__device__ const float* g_mv_lut;         // the stroke's falloff table
__device__ int g_mv_nlut;
__device__ const unsigned char* g_mv_mg[MV_MAXC];   // Grab's scope grids (uint8, x-major), per copy
// her store object: the native store volume the moved (editable) one masks (-1 none) and its grid offset. Inside a move the
// editable volume draws everything (a pre-image it holds no brick at reads her native store), and the native store is a
// step limit only, so her unmoved surface never shows through the moved one
__device__ int g_mv_nat;
__device__ int g_mv_nat_goff;

__device__ __forceinline__ float mv_lut_w(float u) {
    const float* lut = g_mv_lut; int n = g_mv_nlut;
    float x = u * (n - 1);
    int i = (int)x;
    if (i >= n - 1) return lut[n - 1];
    float t = x - i;
    return lut[i] + (lut[i + 1] - lut[i]) * t;
}
// sculpt.cu scope_at / scope_dist, the same arithmetic
__device__ __forceinline__ float mv_scope_at(float x, float y, float z, const float* bp, const unsigned char* mg,
                                             int nx, int ny, int nz) {
    if (bp[27] < 0.5f || mg == 0 || nx < 2) return 1.0f;
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
__device__ __forceinline__ float mv_scope_dist(float ex, float ey, float ez, const float* bp) {
    float d2 = ex * ex + ey * ey + ez * ez;
    float T = bp[38];
    if (T > 0.f) {
        float dn = ex * bp[4] + ey * bp[5] + ez * bp[6];
        float dd = fmaxf(fabsf(dn) - T, 0.f);
        d2 = d2 - dn * dn + dd * dd;
    }
    return sqrtf(fmaxf(d2, 0.f));
}

// one copy's pre-image of a point inside its ball (sculpt.cu typ 14 / 15 node walk)
__device__ float3 mv_pull_one(int c, float x, float y, float z) {
    const float* bp = g_mv_hdr + c * MV_H;
    int typ = (int)bp[0];
    float vx = bp[9], vy = bp[10], vz = bp[11];
    float px = x, py = y, pz = z;
    if (typ == 14) {
        const unsigned char* mg = g_mv_mg[c];
        int mnx = (int)bp[MVH_MN], mny = (int)bp[MVH_MN + 1], mnz = (int)bp[MVH_MN + 2];
        float c0x = bp[29], c0y = bp[30], c0z = bp[31], rt = bp[32];
        int ns = (int)bp[33]; if (ns < 1) ns = 1;
        float dt = 1.0f / (float)ns;
        for (int kk = ns; kk > 0; kk--) {
            float t1 = kk * dt;
            float ex = px - (c0x + t1 * vx), ey = py - (c0y + t1 * vy), ez = pz - (c0z + t1 * vz);
            float dd = mv_scope_dist(ex, ey, ez, bp);
            float w1 = dd < rt ? mv_lut_w(dd / rt) : 0.0f;
            if (w1 > 0.f) w1 *= mv_scope_at(px, py, pz, bp, mg, mnx, mny, mnz);
            float xm = px - 0.5f * dt * w1 * vx, ym = py - 0.5f * dt * w1 * vy, zm = pz - 0.5f * dt * w1 * vz;
            float tm = t1 - 0.5f * dt;
            ex = xm - (c0x + tm * vx); ey = ym - (c0y + tm * vy); ez = zm - (c0z + tm * vz);
            dd = mv_scope_dist(ex, ey, ez, bp);
            float wm = dd < rt ? mv_lut_w(dd / rt) : 0.0f;
            if (wm > 0.f) wm *= mv_scope_at(xm, ym, zm, bp, mg, mnx, mny, mnz);
            px -= dt * wm * vx; py -= dt * wm * vy; pz -= dt * wm * vz;
        }
    } else if (typ == 15) {
        float L = sqrtf(vx * vx + vy * vy + vz * vz);
        if (L > 1e-12f) {
            float ax = vx / L, ay = vy / L, az = vz / L;
            float ex = x - bp[29], ey = y - bp[30], ez = z - bp[31];
            float xa = ex * ax + ey * ay + ez * az;
            if (xa > 0.0f) {
                float qx = ex - xa * ax, qy = ey - xa * ay, qz = ez - xa * az;
                float rho = sqrtf(qx * qx + qy * qy + qz * qz);
                float D = bp[32], rr = bp[33], fade = bp[23];
                if (rho < rr) {
                    float Le = L * mv_lut_w(rho / rr);
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
    }
    return mk(px, py, pz);
}

__device__ __forceinline__ bool mv_in(int c, float x, float y, float z) {
    const float* bp = g_mv_hdr + c * MV_H;
    float rx = x - bp[1], ry = y - bp[2], rz = z - bp[3];
    return rx * rx + ry * ry + rz * rz < bp[7] * bp[7];        // sculpt.cu dab_compute: d >= r keeps the node
}

__device__ __forceinline__ float mv_bound(int c, float x, float y, float z) {
    const float* H = g_mv_hdr + c * MV_H;
    int NL = (int)H[MVH_NL];
    if (NL <= 0) return 1.0f;
    float cs = H[MVH_CS];
    int i = (int)floorf((x - H[MVH_LO]) / cs), j = (int)floorf((y - H[MVH_LO + 1]) / cs), k = (int)floorf((z - H[MVH_LO + 2]) / cs);
    if (i < 0 || j < 0 || k < 0 || i >= NL || j >= NL || k >= NL) return 1.0f;
    return g_mv_L[(int)H[MVH_LOFF] + (i * NL + j) * NL + k];
}

// the composed pre-image (the copies were written in order, so the last is undone first) and the product of the bounds;
// false when no copy's ball holds p (the point reads its own value)
__device__ __forceinline__ bool mv_map(const int* vi, float3 p, float3* q, float* L) {
    if (g_mv_n <= 0 || vi[VI_GOFF] != g_mv_goff) return false;
    float x = p.x, y = p.y, z = p.z, Lm = 1.0f;
    bool any = false;
    for (int c = g_mv_n - 1; c >= 0; c--) {
        if (!mv_in(c, x, y, z)) continue;
        any = true;
        Lm *= mv_bound(c, x, y, z);
        float3 r = mv_pull_one(c, x, y, z);
        x = r.x; y = r.y; z = r.z;
    }
    *q = mk(x, y, z); *L = Lm;
    return any;
}

// the distance from p to the nearest move ball (< 0 inside one), and along a ray to where it enters one (0 inside, 1e30 never):
// outside the balls the field is the stroke-start one, but its empty-space bounds and tile skips know nothing of the moved
// surface inside a ball, so no step taken from outside may pass a ball's edge (11-13 mv4: a ray's empty-tile skip jumped
// into the ball past the moved chin and was taken deep inside, on planes 8 mm apart)
__device__ __forceinline__ float mv_ball_dist0(float3 p) {
    float best = 1e30f;
    for (int c = 0; c < g_mv_n; c++) {
        const float* bp = g_mv_hdr + c * MV_H;
        float rx = p.x - bp[1], ry = p.y - bp[2], rz = p.z - bp[3];
        best = fminf(best, sqrtf(rx * rx + ry * ry + rz * rz) - bp[7]);
    }
    return best;
}
__device__ __forceinline__ float mv_ball_dist(const int* vi, float3 p) {
    if (g_mv_n <= 0 || vi[VI_GOFF] != g_mv_goff) return 1e30f;
    float best = 1e30f;
    for (int c = 0; c < g_mv_n; c++) {
        const float* bp = g_mv_hdr + c * MV_H;
        float rx = p.x - bp[1], ry = p.y - bp[2], rz = p.z - bp[3];
        best = fminf(best, sqrtf(rx * rx + ry * ry + rz * rz) - bp[7]);
    }
    return best;
}
__device__ __forceinline__ float mv_ray_enter(const int* vi, float3 p, float3 d) {
    if (g_mv_n <= 0 || vi[VI_GOFF] != g_mv_goff) return 1e30f;
    float best = 1e30f;
    for (int c = 0; c < g_mv_n; c++) {
        const float* bp = g_mv_hdr + c * MV_H;
        float ox = p.x - bp[1], oy = p.y - bp[2], oz = p.z - bp[3];
        float b = ox * d.x + oy * d.y + oz * d.z, cc = ox * ox + oy * oy + oz * oz - bp[7] * bp[7];
        if (cc <= 0.0f) return 0.0f;
        float disc = b * b - cc;
        if (disc < 0.0f || b > 0.0f) continue;
        best = fminf(best, -b - sqrtf(disc));
    }
    return best;
}

// the stroke-start field at a pre-image q, as a lower bound where no brick holds it. A tile with no brick at a level has
// |F| >= that level's band at its nodes (the layout's own bound, VF_BAND), so an empty tile reads the largest present
// level's band less one of its cells; outside the tile grid (whose border tiles are empty) the distance to the grid's
// box plus that. (11-13 mv2: the first version read the distance to the empty tile's nearest face, which falls to 0 at
// every face: rays crept there and the tracer's near-miss rule took them as hits, stepped plateaus under the chin.) The
// sign is unknown there and read as outside: a ray meets the band of held bricks before it can reach such a point.
__device__ float mv_read(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                         float3 q, int lmin, int* lev) {
    int emp;
    float g = vol_f0(vf, vi, grid, ttab, sdf, q, lmin, lev, &emp);
    if (*lev >= 0) return g;
    if (g_mv_nat >= 0) {                        // her editable copy holds no brick here: her native store's value
        const float* nf = g_vft + g_mv_nat * VF; const int* ni = g_vit + g_mv_nat * VI;
        int ln, en;
        float gn = vol_f0(nf, ni, grid, ttab, sdf, q, 0, &ln, &en);
        if (ln >= 0) { *lev = max(lmin, 0); return gn; }
        if (ln != -2 && !en) return gn;         // her store's own band bound
    }
    float bb = 0.0f;
    for (int l = 0; l < NLEV; l++)
        if (vi[VI_BPT + l] != 0) bb = fmaxf(bb, vf[VF_BAND + l] - vf[VF_H + l]);
    float tm = vf[VF_TM];
    float qq[3] = {q.x, q.y, q.z};
    int nn[3] = {vi[VI_NX], vi[VI_NY], vi[VI_NZ]};
    float out2 = 0.0f;
    bool outside = false;
    for (int a = 0; a < 3; a++) {
        float u = (qq[a] - vf[VF_O + a]) / tm;
        if (u < 0.0f) { out2 += u * u * tm * tm; outside = true; }
        else if (u > (float)nn[a]) { float e = (u - nn[a]) * tm; out2 += e * e; outside = true; }
    }
    if (outside) return sqrtf(out2) + fmaxf(bb, 1e-4f);
    if (emp) return fmaxf(fminf(g, 1e2f), fmaxf(bb, 1e-4f));
    return g;
}

__device__ float vol_f(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                       float3 p, int lmin, int* lev, int* empty) {
    if (g_mv_n > 0 && vi[VI_GOFF] == g_mv_goff) {
        float3 q; float L;
        if (mv_map(vi, p, &q, &L)) {
            float g = mv_read(vf, vi, grid, ttab, sdf, q, max(lmin, 0), lev);
            *empty = 0;
            if (*lev < 0) *lev = -1;
            return g / fmaxf(L, 1.0f);
        }
        float f = vol_f0(vf, vi, grid, ttab, sdf, p, lmin, lev, empty);
        if (f > 0.0f && *lev != -2) f = fminf(f, fmaxf(mv_ball_dist(vi, p), 1e-4f));
        return f;
    }
    if (g_mv_n > 0 && g_mv_nat >= 0 && vi[VI_GOFF] == g_mv_nat_goff) {
        // her native store inside a move: no surface (the moved editable volume draws it), a step to the ball's edge
        float db = mv_ball_dist0(p);
        if (db < 0.0f) { *lev = -2; *empty = 0; return -db + 2e-5f; }
    }
    return vol_f0(vf, vi, grid, ttab, sdf, p, lmin, lev, empty);
}

// the moved field's gradient at a local point inside a move (central differences of the stroke-start field at the
// pre-images, unscaled): the shading normal there. false outside every move
__device__ bool mv_grad(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                        float3 p, float3* g) {
    if (g_mv_n <= 0 || vi[VI_GOFF] != g_mv_goff) return false;
    float3 q; float L;
    if (!mv_map(vi, p, &q, &L)) return false;
    float e = 0.5f * vf[VF_H + max(lev, 0)];
    float gg[3];
    for (int a = 0; a < 3; a++) {
        float3 dd = mk(a == 0 ? e : 0.0f, a == 1 ? e : 0.0f, a == 2 ? e : 0.0f);
        float3 qa, qb; float La, Lb; int la, lb;
        if (!mv_map(vi, add(p, dd), &qa, &La)) qa = add(p, dd);
        if (!mv_map(vi, sub(p, dd), &qb, &Lb)) qb = sub(p, dd);
        gg[a] = mv_read(vf, vi, grid, ttab, sdf, qa, max(lev, 0), &la) - mv_read(vf, vi, grid, ttab, sdf, qb, max(lev, 0), &lb);
    }
    *g = mk(gg[0], gg[1], gg[2]);
    return true;
}

// the bound grid of copy c: per cell, the largest spectral norm of the pre-image map's Jacobian over 3^3 points of the
// cell (central differences at an eighth of a cell); then mv_ldil takes the 27-cell maximum x a margin
extern "C" __global__ void mv_lgrid(int c, float* Lraw) {
    const float* H = g_mv_hdr + c * MV_H;
    int NL = (int)H[MVH_NL];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= NL * NL * NL) return;
    int i = id / (NL * NL), j = (id / NL) % NL, k = id % NL;
    float cs = H[MVH_CS], e = 0.125f * cs;
    float best = 1.0f;
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int d = 0; d < 3; d++) {
        float x = H[MVH_LO] + (i + 0.5f * a) * cs, y = H[MVH_LO + 1] + (j + 0.5f * b) * cs, z = H[MVH_LO + 2] + (k + 0.5f * d) * cs;
        float J[3][3];
        for (int ax = 0; ax < 3; ax++) {
            float ox = ax == 0 ? e : 0.0f, oy = ax == 1 ? e : 0.0f, oz = ax == 2 ? e : 0.0f;
            float3 pa = mv_in(c, x + ox, y + oy, z + oz) ? mv_pull_one(c, x + ox, y + oy, z + oz) : mk(x + ox, y + oy, z + oz);
            float3 pb = mv_in(c, x - ox, y - oy, z - oz) ? mv_pull_one(c, x - ox, y - oy, z - oz) : mk(x - ox, y - oy, z - oz);
            J[0][ax] = (pa.x - pb.x) / (2.0f * e); J[1][ax] = (pa.y - pb.y) / (2.0f * e); J[2][ax] = (pa.z - pb.z) / (2.0f * e);
        }
        // the spectral norm: power iteration on J^T J
        float M[3][3];
        for (int r = 0; r < 3; r++) for (int s = 0; s < 3; s++) M[r][s] = J[0][r] * J[0][s] + J[1][r] * J[1][s] + J[2][r] * J[2][s];
        float u0 = 0.577f, u1 = 0.577f, u2 = 0.577f, lam = 0.0f;
        for (int it = 0; it < 12; it++) {
            float w0 = M[0][0] * u0 + M[0][1] * u1 + M[0][2] * u2;
            float w1 = M[1][0] * u0 + M[1][1] * u1 + M[1][2] * u2;
            float w2 = M[2][0] * u0 + M[2][1] * u1 + M[2][2] * u2;
            lam = sqrtf(w0 * w0 + w1 * w1 + w2 * w2);
            if (lam < 1e-20f) break;
            u0 = w0 / lam; u1 = w1 / lam; u2 = w2 / lam;
        }
        best = fmaxf(best, sqrtf(fmaxf(lam, 0.0f)));
    }
    Lraw[id] = best;
}
// diagnosis (fs-brush-2 11-13): per world point of volume v: the read the tracer takes (vol_f), its level, whether a move
// maps it, the pre-image, the bound, the stroke-start read at the pre-image and its level -> out[10 * i ..]
extern "C" __global__ void mv_probe(int v, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                    const float* pw, int n, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* vf = vft + v * VF; const int* vi = vit + v * VI;
    float3 p = xf_p(vf + VF_M, mk(pw[3 * i], pw[3 * i + 1], pw[3 * i + 2]));
    int lev = -9, emp = -9;
    float f = vol_f(vf, vi, grid, ttab, sdf, p, 0, &lev, &emp);
    float3 q = p; float L = 1.0f;
    bool m = mv_map(vi, p, &q, &L);
    int l0 = -9, e0 = -9;
    float f0 = vol_f0(vf, vi, grid, ttab, sdf, q, 0, &l0, &e0);
    float* o = out + 10 * i;
    o[0] = f; o[1] = (float)lev; o[2] = m ? 1.0f : 0.0f; o[3] = q.x; o[4] = q.y; o[5] = q.z; o[6] = L; o[7] = f0; o[8] = (float)l0; o[9] = (float)e0;
}

extern "C" __global__ void mv_ldil(int c, const float* Lraw, float* Lout, float margin) {
    const float* H = g_mv_hdr + c * MV_H;
    int NL = (int)H[MVH_NL];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= NL * NL * NL) return;
    int i = id / (NL * NL), j = (id / NL) % NL, k = id % NL;
    float m = 1.0f;
    for (int a = -1; a <= 1; a++) for (int b = -1; b <= 1; b++) for (int d = -1; d <= 1; d++) {
        int ii = i + a, jj = j + b, kk = k + d;
        if (ii < 0 || jj < 0 || kk < 0 || ii >= NL || jj >= NL || kk >= NL) continue;
        m = fmaxf(m, Lraw[(ii * NL + jj) * NL + kk]);
    }
    Lout[(int)H[MVH_LOFF] + id] = m * margin;
}

// ---- fs-materials 11-14: the material look (materials/look.py; inserted into gpu/trace.cu between these markers) -----
// A volume whose object carries a library material is shaded from it in light_pass (trace.cu's hook calls mat_shade):
//  - its ALBEDO is a volumetric composition field in the object's own frame (the stone's grains, each its mineral's
//    colour; fine grains; intergranular pores; mottling; bedding laminae), so it is the same through a cut, a chip or a
//    crack; every term is filtered by the pixel's footprint as a camera's pixel averages it (the arithmetic mean
//    albedo per channel stays the measured one at any zoom);
//  - its SPECULAR is GGX with Smith visibility and Schlick Fresnel from the profile (alpha by finish, F0 from n);
//  - its DIFFUSE under the key is Oren-Nayar (rough, porous surfaces scatter back toward the light) with no wrap.
// Shading comes only from the geometry and the light; nothing here darkens or lightens by position.
// Without a material (g_mat[vol * MATF] == 0) mat_shade returns at once and light_pass is as before, bit for bit.
#define MATF 192
#define MAT_NV 64
__device__ float g_mat[MAT_NV * MATF];
__device__ float g_mat_pa;          // the view's pixel angle (rad): look.py per frame

__device__ __forceinline__ unsigned int lk_h32(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    return (x >> 16) ^ x;
}
__device__ __forceinline__ unsigned int lk_hc(int ix, int iy, int iz, int seed) {
    return lk_h32((unsigned int)ix ^ lk_h32((unsigned int)iy ^ lk_h32((unsigned int)iz ^ lk_h32((unsigned int)seed))));
}
__device__ __forceinline__ float lk_u01(unsigned int h) { return (float)(h & 0xFFFFFFu) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float lk_fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
__device__ __forceinline__ float lk_grad(unsigned int h, float x, float y, float z) {
    h &= 15u;
    float u = h < 8u ? x : y;
    float v = h < 4u ? y : ((h == 12u || h == 14u) ? x : z);
    return ((h & 1u) == 0u ? u : -u) + ((h & 2u) == 0u ? v : -v);
}
__device__ float lk_noise3(float x, float y, float z, int seed) {
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float xf = x - fx, yf = y - fy, zf = z - fz;
    float u = lk_fade(xf), v = lk_fade(yf), w = lk_fade(zf), res = 0.0f;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                float g = lk_grad(lk_hc(ix + dx, iy + dy, iz + dz, seed), xf - dx, yf - dy, zf - dz);
                res += (dx ? u : 1.0f - u) * (dy ? v : 1.0f - v) * (dz ? w : 1.0f - w) * g;
            }
    return res;
}
// unit-SD fBm with wavelengths lam, lam/2, ... (octaves under ~2 pixels faded out); a different seed per octave.
// PATENT GUARD (research/PATENT_GRAIN_RELIEF.md): Perlin 2002 octaves, deliberately; do not switch to band-limited
// (wavelet) noise before Pixar's US 7,672,476 expires (~2028-10).
__device__ float lk_fbm(float3 q, float lam, int oct, int seed, float foot) {
    float s = 0.0f, a = 1.0f, norm = 0.0f;
    for (int k = 0; k < oct; k++) {
        float l = lam / (float)(1 << k);
        float f = fminf(fmaxf((l / fmaxf(foot, 1e-9f) - 1.0f) / 2.0f, 0.0f), 1.0f);   // full at 3 px, none at 1 px
        float sh = 0.37f * k;
        s += a * f * lk_noise3(q.x / l + sh, q.y / l - sh, q.z / l + 0.5f * sh, seed + 31 * k) * 3.71f;
        norm += a * a;
        a *= 0.6f;
    }
    return s / sqrtf(norm);
}
// cellular grains of size cell: the nearest grain's uniform value u (0..1), its class hash, and its distance d1 (in
// cells): grains are round about their centres, the interstices (large d1) are where the pores and the cement are
__device__ void lk_cells(float3 q, float cell, int seed, float* u, unsigned int* cls, float* d1o) {
    float gx = q.x / cell, gy = q.y / cell, gz = q.z / cell;
    int cx = (int)floorf(gx), cy = (int)floorf(gy), cz = (int)floorf(gz);
    float d1 = 1e9f;
    int bi = 0, bj = 0, bk = 0;
    for (int i = cx - 1; i <= cx + 1; i++)
        for (int j = cy - 1; j <= cy + 1; j++)
            for (int k = cz - 1; k <= cz + 1; k++) {
                float px = i + lk_u01(lk_hc(i, j, k, seed + 1)), py = j + lk_u01(lk_hc(i, j, k, seed + 2)),
                      pz = k + lk_u01(lk_hc(i, j, k, seed + 3));
                float dx = gx - px, dy = gy - py, dz = gz - pz, d = sqrtf(dx * dx + dy * dy + dz * dz);
                if (d < d1) { d1 = d; bi = i; bj = j; bk = k; }
            }
    *u = lk_u01(lk_hc(bi, bj, bk, seed));
    *cls = lk_hc(bi, bj, bk, seed + 7);
    *d1o = d1;
}
// 11-14 (update 11): the nearest grain cell's offset r (point - centre, in cells), its hash and distance, for the
// fracture's facets
__device__ void lk_facet(float3 q, float cell, int seed, float3* r, unsigned int* cls, float* d1o) {
    float gx = q.x / cell, gy = q.y / cell, gz = q.z / cell;
    int cx = (int)floorf(gx), cy = (int)floorf(gy), cz = (int)floorf(gz);
    float d1 = 1e9f, bx = 0.0f, by = 0.0f, bz = 0.0f;
    int bi = 0, bj = 0, bk = 0;
    for (int i = cx - 1; i <= cx + 1; i++)
        for (int j = cy - 1; j <= cy + 1; j++)
            for (int k = cz - 1; k <= cz + 1; k++) {
                float px = i + lk_u01(lk_hc(i, j, k, seed + 1)), py = j + lk_u01(lk_hc(i, j, k, seed + 2)),
                      pz = k + lk_u01(lk_hc(i, j, k, seed + 3));
                float dx = gx - px, dy = gy - py, dz = gz - pz, d = sqrtf(dx * dx + dy * dy + dz * dz);
                if (d < d1) { d1 = d; bi = i; bj = j; bk = k; bx = dx; by = dy; bz = dz; }
            }
    *r = mk(bx, by, bz);
    *cls = lk_hc(bi, bj, bk, seed + 7);
    *d1o = d1;
}
// a broken grain's surface: a plane at the grain's own level (proud, or a socket where it was plucked) tilted its own
// way (the break through it), falling toward the cell's edge (the gap round the grain); heights in grain units
__device__ __forceinline__ float lk_facet_h(float3 r, unsigned int cls, float d1) {
    float lev = 1.2f * lk_u01(lk_h32(cls ^ 0x7f4a7c15u)) - 0.2f;
    float th = 6.2831853f * lk_u01(lk_h32(cls ^ 0x3c6ef372u)), ct = 2.0f * lk_u01(lk_h32(cls ^ 0xa54ff53au)) - 1.0f;
    float st = sqrtf(fmaxf(1.0f - ct * ct, 0.0f)), sl = 0.9f * lk_u01(lk_h32(cls ^ 0x510e527fu));
    float tilt = sl * (r.x * st * cosf(th) + r.y * st * sinf(th) + r.z * ct);
    return lev + tilt - 0.6f * d1 * d1;
}
// sparse spots (rust: iron oxide concretions) on a cell grid of size cell: the weight 0..1 of the nearest spot
__device__ float lk_spots(float3 q, float cell, float prob, float rad, int seed) {
    float gx = q.x / cell, gy = q.y / cell, gz = q.z / cell;
    int cx = (int)floorf(gx), cy = (int)floorf(gy), cz = (int)floorf(gz);
    float best = 0.0f;
    for (int i = cx - 1; i <= cx + 1; i++)
        for (int j = cy - 1; j <= cy + 1; j++)
            for (int k = cz - 1; k <= cz + 1; k++) {
                if (lk_u01(lk_hc(i, j, k, seed)) >= prob) continue;
                float px = i + lk_u01(lk_hc(i, j, k, seed + 1)), py = j + lk_u01(lk_hc(i, j, k, seed + 2)),
                      pz = k + lk_u01(lk_hc(i, j, k, seed + 3));
                float r = rad * (0.6f + 0.8f * lk_u01(lk_hc(i, j, k, seed + 4)));
                float dx = gx - px, dy = gy - py, dz = gz - pz, t = (dx * dx + dy * dy + dz * dz) / (r * r);
                if (t < 1.0f) { float w = 1.0f - t; w = w * w * (3.0f - 2.0f * w); best = fmaxf(best, w); }
            }
    return best;
}
// irregular layering along a bed: 1-D noise of the bed coordinate (several octaves: layers of random thickness),
// laterally continuous; unit SD
// The layers are read along an OBLIQUE line through the 3-D noise (11-14, the plaid fix): read along a lattice line
// (y = z = 0) Perlin noise is zero at every lattice point, so the layers came out periodic at the spacing (pale
// bands every ~2 mm crossing the droved strokes). A unit direction with incommensurate components, offset off every
// lattice plane, crosses the cells at irregular intervals: the same spacing statistics, no period.
__device__ float lk_layers(float zb, float spacing, int seed, float foot) {
    float s = 0.0f, a = 1.0f, norm = 0.0f;
    for (int k = 0; k < 4; k++) {
        float l = spacing * (float)(1 << k);
        float f = fminf(fmaxf((l / fmaxf(foot, 1e-9f) - 1.0f) / 2.0f, 0.0f), 1.0f);
        float t = zb / l;
        s += a * f * lk_noise3(0.8090f * t + 0.37f * k + 0.131f, 0.4253f * t + 0.71f * k + 0.293f,
                               0.4057f * t + 0.53f * k + 0.617f, seed + 53 * k) * 3.71f;
        norm += a * a;
        a *= 0.8f;
    }
    return s / sqrtf(norm);
}

// log E[exp(X)] for the terms' laws: a uniform unit-SD value scaled by g; a class of share p with log value k; a band of
// share phi in which the log value ramps 0..k (uniform) -> what each term adds to the log of the arithmetic mean
__device__ __forceinline__ float lk_uni(float g) { float a = 1.7320508f * fabsf(g); return a < 1e-4f ? 0.0f : logf(sinhf(a) / a); }
__device__ __forceinline__ float lk_two(float p, float k) { return logf(1.0f - p + p * expf(k)); }
__device__ __forceinline__ float lk_band(float phi, float k) { return fabsf(k) < 1e-5f ? 0.0f : logf(1.0f - phi + phi * (expf(k) - 1.0f) / k); }

// ---- the grain relief (render-time geometry below the lattice) -------------------------------------------------------
// The same grains as the colour field (lk_cells, the same seed), each a low dome about its centre (flat-topped on a
// sawn face: clipped), the interstices lowest. Its amplitude fades with the pixel footprint (under ~1 px a grain it is
// the BRDF's roughness). The tracer marches it (trace_vol: the hit is ON it), the G-buffer normal is its gradient, and
// mat_shade marches it for the sun's grain-scale shadow and the sky's grain-scale occlusion: one geometry, every use.
// Table: 39 amplitude A (m)   40 dome radius (cells)   41 top clip (0..1; 1 = full domes)   42 mean height (for zero mean)
__device__ __forceinline__ int mat_vol(const float* vf) {
    if (g_vft == 0) return -1;
    long long d = (long long)(vf - g_vft);
    return (d >= 0 && d % VF == 0 && d / VF < MAT_NV) ? (int)(d / VF) : -1;
}
__device__ __forceinline__ float mat_micro_amp(int vol) {
    if (vol < 0) return 0.0f;
    const float* m = g_mat + vol * MATF;
    if (m[0] < 0.5f) return 0.0f;
    float a = (m[39] > 0.0f && m[7] > 0.0f) ? m[39] : 0.0f;
    return (m[65] > 0.0f && m[74] > 0.0f && m[75] > 0.0f) ? fmaxf(a, m[74]) : a;   // the fracture relief's, if larger
}
// 11-14 (fs-materials-2, update 11): BROKEN and WORN surfaces are granular. Below the as-made box (slots 65-68) by
// more than the finish's own relief (76) the surface was broken (a chip scar, a crumble bite, a crack's wall) or worn
// (weathering's rounded arris): there the grain relief becomes a fracture relief (74 amplitude, 75 cell): each grain its
// own level (proud, or a plucked socket), flat-topped (the fracture runs round the grains). 0 on the as-made face.
__device__ __forceinline__ float mat_frac_t(const float* m, float3 q) {
    if (!(m[65] > 0.0f && m[74] > 0.0f && m[75] > 0.0f)) return 0.0f;
    float ax = fabsf(q.x) - m[65] + m[68], ay = fabsf(q.y) - m[66] + m[68], az = fabsf(q.z) - m[67] + m[68];
    float px = fmaxf(ax, 0.0f), py = fmaxf(ay, 0.0f), pz = fmaxf(az, 0.0f);
    float d = -(sqrtf(px * px + py * py + pz * pz) + fminf(fmaxf(ax, fmaxf(ay, az)), 0.0f) - m[68]);   // depth below it
    float t = fminf(fmaxf((d - m[76]) / m[76], 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
__device__ __forceinline__ float mat_micro_fade(const float* m, float foot) {
    return fminf(fmaxf((m[7] / fmaxf(foot, 1e-9f) - 1.0f) / 2.0f, 0.0f), 1.0f);   // full at 3 px a grain, none at 1
}
// the relief height (local units) at local point q; 0 without relief
__device__ float mat_micro(int vol, float3 q, float foot) {
    const float* m = g_mat + vol * MATF;
    float tf = mat_frac_t(m, q), out = 0.0f;
    if (tf < 1.0f && m[39] > 0.0f && m[7] > 0.0f) {
        float fg = mat_micro_fade(m, foot);
        if (fg > 0.0f) {
            float u; unsigned int cls; float d1;
            lk_cells(q, m[7], (int)m[29] * 13 + 101, &u, &cls, &d1);
            float t = 1.0f - (d1 / m[40]) * (d1 / m[40]);
            float h = t > 0.0f ? fminf(t, m[41]) / m[41] : 0.0f;
            h *= 0.7f + 0.6f * lk_u01(lk_h32(cls ^ 0x51ed27u));                  // each grain its own height
            out = (1.0f - tf) * fg * m[39] * (h - m[42]);
        }
    }
    if (tf > 0.0f) {
        // facets at two scales: the cell (m[75]) and 0.4 x it, heights 0.7 and 0.3 (their sum stays under the tracer's 2 x amplitude reach)
        float sc[2] = {m[75], 0.4f * m[75]}, wt[2] = {0.7f, 0.3f};
        for (int k = 0; k < 2; k++) {
            float ff = fminf(fmaxf((sc[k] / fmaxf(foot, 1e-9f) - 1.0f) / 2.0f, 0.0f), 1.0f);   // full at 3 px a grain
            if (ff <= 0.0f) continue;
            float3 r; unsigned int cls; float d1;
            lk_facet(q, sc[k], (int)m[29] * 13 + 977 + 61 * k, &r, &cls, &d1);
            out += tf * ff * wt[k] * m[74] * (lk_facet_h(r, cls, d1) - m[82]);
        }
    }
    return out;
}
__device__ float3 mat_micro_grad(int vol, float3 q, float foot) {
    const float* m = g_mat + vol * MATF;
    float e = fmaxf(0.04f * m[7], 0.25f * foot);
    float gx = mat_micro(vol, add(q, mk(e, 0, 0)), foot) - mat_micro(vol, sub(q, mk(e, 0, 0)), foot);
    float gy = mat_micro(vol, add(q, mk(0, e, 0)), foot) - mat_micro(vol, sub(q, mk(0, e, 0)), foot);
    float gz = mat_micro(vol, add(q, mk(0, 0, e)), foot) - mat_micro(vol, sub(q, mk(0, 0, e)), foot);
    return mul(mk(gx, gy, gz), 0.5f / e);
}
// the G-buffer normal with the relief: n (world, from the lattice) tilted by the relief's gradient
__device__ float3 mat_micro_normal(const float* vf, float3 pw, float3 n, float foot_w) {
    int vol = mat_vol(vf);
    if (mat_micro_amp(vol) <= 0.0f) return n;
    float s = vf[24];
    float3 q = mk(vf[0] * pw.x + vf[1] * pw.y + vf[2] * pw.z + vf[3], vf[4] * pw.x + vf[5] * pw.y + vf[6] * pw.z + vf[7],
                  vf[8] * pw.x + vf[9] * pw.y + vf[10] * pw.z + vf[11]);
    float3 nl = nrm(mk(vf[0] * n.x + vf[1] * n.y + vf[2] * n.z, vf[4] * n.x + vf[5] * n.y + vf[6] * n.z,
                       vf[8] * n.x + vf[9] * n.y + vf[10] * n.z));
    float3 g = mat_micro_grad(vol, q, foot_w * s);
    g = sub(g, mul(nl, dot3(g, nl)));                                   // its tangential part tilts the normal
    float3 gl = sub(nl, g);
    const float* MI = vf + 12;
    return nrm(mk(MI[0] * gl.x + MI[1] * gl.y + MI[2] * gl.z, MI[4] * gl.x + MI[5] * gl.y + MI[6] * gl.z,
                  MI[8] * gl.x + MI[9] * gl.y + MI[10] * gl.z));
}
// grain-scale visibility along a local direction dl from q (normal nl): the relief as a height field over the tangent
// plane, marched over ~3 grains; 1 = open
// PATENT GUARD (fs-materials research/PATENT_GRAIN_RELIEF.md, 11-14, a reading, not legal advice): this is a fixed-step
// CLEARANCE test of the light ray (relief / parallax-occlusion self-shadowing: Policarpo 2005, Tatarchuk 2004/2006).
// Do NOT change it into a search for the point of maximum elevation / horizon angle or into horizon maps before
// ~2031-08 without counsel (Adobe US 8,379,021 B1 claims that route for height-field shadows).
__device__ float mat_micro_vis(int vol, float3 q, float3 nl, float3 dl, float foot) {
    const float* m = g_mat + vol * MATF;
    float cos_t = dot3(nl, dl);
    if (cos_t <= 0.0f) return 0.0f;
    float3 tl = sub(dl, mul(nl, cos_t));
    float tlen = sqrtf(dot3(tl, tl));
    if (tlen < 1e-4f) return 1.0f;
    tl = mul(tl, 1.0f / tlen);
    float slope = cos_t / tlen;                                         // the ray's rise per unit of tangent distance
    float tf = mat_frac_t(m, q);                                        // on a broken surface: its coarser grains
    float cell = tf > 0.5f ? m[75] : m[7], amp = tf > 0.5f ? m[74] : m[39];
    float h0 = mat_micro(vol, q, foot), vis = 1.0f, soft = 0.25f * amp + 1e-9f;
    for (int k = 1; k <= 6; k++) {
        float x = 0.45f * cell * k;
        float hk = mat_micro(vol, add(q, mul(tl, x)), foot);
        vis = fminf(vis, fminf(fmaxf((h0 + slope * x - hk) / soft + 0.5f, 0.0f), 1.0f));
    }
    return vis;
}

// ---- GRANITE: crystals by mineral (11-14 fs-materials-2, update 12; look.py crystal_shares mirrors this code) ----
// 96 coarse cell (m; 0 = off)  97 size spread (additive weight, cells)  98 outline warp (cells)  99 flake cell (m)
// 100 flake share  101 clot wavelength (m)  102 the boundary band (power-distance gap, cells^2)  103 plate aspect
// 104-106 coarse shares: white feldspar, K-feldspar, quartz (biotite books: the rest)
// 107-118 log colour of each kind (white feldspar, K-feldspar, quartz, biotite: RGB each)
// 119-122 the kinds' EFFECTIVE area shares (Monte Carlo)  123 per-crystal lightness (log SD)
// 124 quartz alpha factor  125 biotite alpha factor
__device__ int lk_crystal(float3 q, const float* m, int seed, unsigned int* hco, float3* cno, float* zo, float* gpo) {
    float c = m[96];
    float gx = q.x / c, gy = q.y / c, gz = q.z / c;
    if (m[98] > 0.0f) {                       // irregular, interlocking outlines: the lookup through a smooth warp
        float s = 1.25f;
        float wx = lk_noise3(gx * s + 0.13f, gy * s + 0.71f, gz * s + 0.37f, seed + 11) * 3.71f;
        float wy = lk_noise3(gx * s + 0.53f, gy * s + 0.29f, gz * s + 0.91f, seed + 12) * 3.71f;
        float wz = lk_noise3(gx * s + 0.83f, gy * s + 0.47f, gz * s + 0.19f, seed + 13) * 3.71f;
        gx += m[98] * wx; gy += m[98] * wy; gz += m[98] * wz;
    }
    if (m[148] > 0.0f) {                      // 11-14 (update 18): a fine warp octave: serrated, jagged crystal edges
        float s2 = 4.0f;
        float jx = lk_noise3(gx * s2 + 0.61f, gy * s2 + 0.17f, gz * s2 + 0.83f, seed + 14) * 3.71f;
        float jy = lk_noise3(gx * s2 + 0.29f, gy * s2 + 0.93f, gz * s2 + 0.41f, seed + 15) * 3.71f;
        float jz = lk_noise3(gx * s2 + 0.77f, gy * s2 + 0.53f, gz * s2 + 0.07f, seed + 16) * 3.71f;
        gx += m[148] * jx; gy += m[148] * jy; gz += m[148] * jz;
    }
    int cx = (int)floorf(gx), cy = (int)floorf(gy), cz = (int)floorf(gz);
    float best = 1e9f, best2 = 1e9f;               // a power diagram: straight crystal faces, sizes varied
    int bi = 0, bj = 0, bk = 0;
    for (int i = cx - 1; i <= cx + 1; i++)
        for (int j = cy - 1; j <= cy + 1; j++)
            for (int k = cz - 1; k <= cz + 1; k++) {
                float px = i + lk_u01(lk_hc(i, j, k, seed + 1)), py = j + lk_u01(lk_hc(i, j, k, seed + 2)),
                      pz = k + lk_u01(lk_hc(i, j, k, seed + 3));
                float dx = gx - px, dy = gy - py, dz = gz - pz;
                float d = dx * dx + dy * dy + dz * dz - m[97] * (lk_u01(lk_hc(i, j, k, seed + 5)) - 0.5f);
                if (d < best) { best2 = best; best = d; bi = i; bj = j; bk = k; }
                else if (d < best2) best2 = d;
            }
    unsigned int h = lk_hc(bi, bj, bk, seed + 7);
    float u = lk_u01(h);
    *zo = best + m[97] * (lk_u01(lk_hc(bi, bj, bk, seed + 5)) - 0.5f);   // the squared distance from its own centre
    *gpo = best2 - best;                                                 // the gap to the next crystal (0: a boundary)
    {                                          // the crystal's own optical plane (cleavage / basal): a random normal
        float th0 = 6.2831853f * lk_u01(lk_h32(h ^ 0x9e3779b9u)), ct0 = 2.0f * lk_u01(lk_h32(h ^ 0x85ebca6bu)) - 1.0f;
        float st0 = sqrtf(fmaxf(1.0f - ct0 * ct0, 0.0f));
        *cno = mk(st0 * cosf(th0), st0 * sinf(th0), ct0);
    }
    int kind = u < m[104] ? 0 : (u < m[104] + m[105] ? 1 : (u < m[104] + m[105] + m[106] ? 2 : 3));
    float bnd = m[102] > 0.0f ? fmaxf(1.0f - (best2 - best) / m[102], 0.0f) : 0.0f;   // near a crystal boundary
    if (m[99] > 0.0f && bnd > 0.0f) {         // biotite plates between the crystals, denser in clots
        float cl = lk_noise3(q.x / m[101] + 0.5f, q.y / m[101] + 0.25f, q.z / m[101] + 0.75f, seed + 21) * 3.71f;
        float pf = m[100] * bnd * fminf(fmaxf(0.5f + 0.35f * cl, 0.0f), 1.0f);
        {
            float fc = m[99];
            float fx = q.x / fc, fy = q.y / fc, fz = q.z / fc;
            int ix = (int)floorf(fx), iy = (int)floorf(fy), iz = (int)floorf(fz);
            float e = m[103] * m[103] - 1.0f;
            for (int i = ix - 1; i <= ix + 1; i++)
                for (int j = iy - 1; j <= iy + 1; j++)
                    for (int k = iz - 1; k <= iz + 1; k++) {
                        unsigned int hf = lk_hc(i, j, k, seed + 31);
                        if (lk_u01(hf) >= pf) continue;
                        float dx = fx - (i + lk_u01(lk_hc(i, j, k, seed + 32))), dy = fy - (j + lk_u01(lk_hc(i, j, k, seed + 33))),
                              dz = fz - (k + lk_u01(lk_hc(i, j, k, seed + 34)));
                        float th = 6.2831853f * lk_u01(lk_h32(hf ^ 0x3c6ef372u)), ct = 2.0f * lk_u01(lk_h32(hf ^ 0xa54ff53au)) - 1.0f;
                        float st = sqrtf(fmaxf(1.0f - ct * ct, 0.0f));
                        float ra = dx * st * cosf(th) + dy * st * sinf(th) + dz * ct;
                        float d2 = dx * dx + dy * dy + dz * dz + e * ra * ra;
                        float rr2 = 0.36f;
                        if (m[156] > 0.0f && d2 < 0.36f * (1.0f + m[156]) * (1.0f + m[156]))   // update 23: a frayed edge
                            rr2 *= fmaxf(1.0f + m[156] * lk_noise3(fx * 3.0f + 0.3f, fy * 3.0f + 0.7f, fz * 3.0f + 0.1f, seed + 35) * 3.71f, 0.2f);
                        if (d2 < rr2) { kind = 3; h = hf; *cno = mk(st * cosf(th), st * sinf(th), ct); }   // a plate
                    }
        }
    }
    *hco = h;
    return kind;
}
// update 21 / 25: how strongly water leaves the top at the arris point a (local): the crumble's bite there (the field's
// own bite function) and the top arris's chip scars; ~0..1.3
__device__ float mat_runoff_src(const float* m, float3 a) {
    float n1 = lk_noise3(a.x / m[150] + 0.31f, a.y / m[150] + 0.17f, a.z / m[150] + 0.53f, (int)m[152]) * 3.71f;
    float w = m[149] * fmaxf(n1 - m[151], 0.0f) / (1.0f - m[151]);
    if (m[153] > 0.0f)
        w += m[153] * fmaxf(lk_noise3(a.x / m[154] + 0.77f, a.y / m[154] + 0.41f, a.z / m[154] + 0.23f,
                                      (int)m[152] + 101) * 3.71f, 0.0f);
    float src = fminf(w / (1.6f * (m[149] + m[153])), 1.2f);
    for (int k = 0; k < (int)m[155]; k++) {
        const float* S = m + 160 + 4 * k;
        float dx = a.x - S[0], dz = a.z - S[1], dd = (dx * dx + dz * dz) / (S[2] * S[2]);
        if (dd < 1.0f) src = fmaxf(src, 1.3f * S[3] * (1.0f - dd));
    }
    return src;
}
// the table, per volume (look.py MAT_* keep the same indices):
//  0 on   1-3 base albedo (linear RGB, the copy's measured mean)   4 GGX alpha   5 F0   6 Oren-Nayar sigma (rad)
//  7 grain size (m)   8 grain contrast (log SD)   9-11 grain colour direction (log RGB per unit)
// 12 fine grain size (m)   13 fine contrast (log SD)   14 interstice darkening (log)   15 interstice start (d1, cells)
// 16 mottle wavelength (m)   17 mottle (log SD)   18-20 the iron direction (log RGB: more iron = darker and redder)
// 21 dark-grain share   22 dark-grain log factor   23 lamina spacing (m)   24 lamina (log SD along the iron direction)
// 25-27 bed normal (local)   28 lamina warp (m, the bed's waviness)   29 seed   30 spot cell (m)   31 spot share
// 32 spot radius (cells)   33 spot log strength (along the iron direction)   34 lamina lightness (log SD, grey axis)
// 35 interstice share (P(d1 > start), measured)   36 interstice mean ramp in that share (measured)
// 37 grain-cluster wavelength (m)   38 grain-cluster (log SD along the grain direction)
// 43 grain hue (log SD per grain: clean grains paler and greyer, coated grains warmer)   44-46 its direction (log RGB;
// zero: the iron direction)   47 clear-grain share (clean quartz: glassy fracture faces, so a sharper specular and a
// darker, greyer look, the light going into the grain)
// 48-50 stain direction (log RGB; zero: the iron direction)   51 patch wavelength (m)   52 patch log strength
// 53 patch threshold (unit SD)   54 band spacing (m)   55 band log strength   56 band width (fraction of the spacing)
// 57 ring zone depth (rings)   58 rings per unit of the patch field   59 the rings' mean over the volume
// 64 fired skin on   65-67 the as-fired box's half size (m)   68 its round (m)   69 skin depth (m)   70 transition (m)
// 71-73 the body's log colour shift from the face
__device__ bool mat_shade(int vol, const float* vf, float3 pw, float3 n, const float* L, float ndl, float* wd,
                          float* alb, float* spec, float* sky) {
    if (vol < 0 || vol >= MAT_NV) return false;
    const float* m = g_mat + vol * MATF;
    if (m[0] < 0.5f) return false;
    const float* M = vf;                          // VF_M: world -> local rows
    float3 q = mk(M[0] * pw.x + M[1] * pw.y + M[2] * pw.z + M[3], M[4] * pw.x + M[5] * pw.y + M[6] * pw.z + M[7],
                  M[8] * pw.x + M[9] * pw.y + M[10] * pw.z + M[11]);
    float3 cpos = mk(L[17], L[18], L[19]);
    float3 vd = sub(cpos, pw);
    float dist = sqrtf(dot3(vd, vd));
    vd = mul(vd, 1.0f / fmaxf(dist, 1e-9f));
    float foot = dist * fmaxf(g_mat_pa, 1e-6f) * vf[24];        // the pixel's footprint in local units (VF_S)
    int seed = (int)m[29];
    float la[3] = {logf(fmaxf(m[1], 1e-4f)), logf(fmaxf(m[2], 1e-4f)), logf(fmaxf(m[3], 1e-4f))};
    float var[3] = {0.0f, 0.0f, 0.0f};
    float a_mul = 1.0f;                           // the grain's own surface: clear grains are smoother
    int gk = -1;                                  // granite: the crystal's kind here (-1: none, or too small to see)
    float3 gcn = mk(0.0f, 0.0f, 1.0f);            // its optical plane's normal (local)
    float3 vd0 = vd;                              // the view direction (world), for the quartz depth
    // grains: each its own colour along the grain direction, a dark lithic class, round, with darker interstices (the
    // pores and the cement between them); every term's arithmetic mean is taken back out (lk_uni, lk_two), so the mean
    // albedo stays the measured one at any zoom
    if (m[96] > 0.0f) {
        // GRANITE crystals by mineral: each kind its colour, each crystal its own lightness, the mean taken out with
        // the kinds' effective area shares at this footprint's strength
        float fg = fminf(fmaxf((m[96] / fmaxf(foot, 1e-9f) - 0.6f) / 1.4f, 0.0f), 1.0f);
        if (fg > 0.0f) {
            unsigned int hc;
            float3 cnl;
            float zr, gp;
            int kind = lk_crystal(q, m, seed * 13 + 1301, &hc, &cnl, &zr, &gp);
            // 11-14 (update 16, against flat fills): structure inside the crystal and at its edges, mean kept:
            // ZONING (feldspar grows in zones: a core-to-rim tone change, each crystal its own sign and strength) and
            // BOUNDARY SEAMS (grain-boundary films and hairline gaps: darker within ~0.03 cells^2 of the boundary)
            if (kind <= 1 && m[145] > 0.0f) {
                float zs = (lk_u01(lk_h32(hc ^ 0x1b873593u)) - 0.5f) * 2.0f;          // -1..1
                float zt = fminf(zr / 0.35f, 1.0f) - 0.5f;                            // core -0.5 .. rim +0.5
                for (int ch = 0; ch < 3; ch++) la[ch] += fg * m[145] * zs * zt;
            }
            if (m[146] != 0.0f && m[147] > 0.0f) {
                float ts = fminf(fmaxf(1.0f - gp / m[147], 0.0f), 1.0f);
                float sh = 0.5f * m[147] * 0.6f;                                     // ~ the seams' share (E)
                for (int ch = 0; ch < 3; ch++) la[ch] += fg * m[146] * (ts * ts - sh);
            }
            gk = fg > 0.5f ? kind : -1;
            gcn = cnl;
            if (kind <= 1 && m[143] > 0.0f) {       // feldspar's milky clouding inside the crystal (mean kept)
                float vm = lk_fbm(q, m[144], 2, seed * 59 + 1511, foot);
                for (int ch = 0; ch < 3; ch++) { la[ch] += fg * m[143] * vm; var[ch] += fg * fg * m[143] * m[143]; }
            }
            if (kind == 2 && m[141] > 0.0f && m[142] > 0.0f) {   // QUARTZ: inclusions seen INTO the crystal
                float3 vl = nrm(mk(M[0] * vd0.x + M[1] * vd0.y + M[2] * vd0.z, M[4] * vd0.x + M[5] * vd0.y + M[6] * vd0.z,
                                   M[8] * vd0.x + M[9] * vd0.y + M[10] * vd0.z));
                float3 nl0 = nrm(mk(M[0] * n.x + M[1] * n.y + M[2] * n.z, M[4] * n.x + M[5] * n.y + M[6] * n.z,
                                    M[8] * n.x + M[9] * n.y + M[10] * n.z));
                float ci = fmaxf(dot3(nl0, vl), 0.0f), eta = 1.0f / 1.544f;          // into quartz (n 1.544)
                float k2 = 1.0f - eta * eta * (1.0f - ci * ci);
                float3 rr = k2 > 0.0f ? nrm(sub(mul(vl, -eta), mul(nl0, sqrtf(k2) - eta * ci))) : mul(nl0, -1.0f);
                float acc = 0.0f, wsum = 0.0f, dps[3] = {0.15f, 0.4f, 0.8f};
                for (int k = 0; k < 3; k++) {
                    float dd = dps[k] * 1e-3f * m[141] / 1e-3f;
                    float wk = expf(-dps[k] / 0.6f);
                    acc += wk * lk_spots(add(q, mul(rr, dd)), 0.18e-3f, 0.25f, 0.35f, seed * 61 + 1601 + 7 * k);
                    wsum += wk;
                }
                float sp = acc / fmaxf(wsum, 1e-6f), ph = 0.25f * 4.18879f * 0.35f * 0.35f * 0.35f * 0.9f;
                for (int ch = 0; ch < 3; ch++) la[ch] += fg * m[142] * (sp - ph);   // milky specks: lighter, the mean out
            }
            float v = (lk_u01(lk_h32(hc ^ 0x2545f491u)) - 0.5f) * 3.4641f, g = fg * m[123];
            for (int ch = 0; ch < 3; ch++) {
                float sm = 0.0f;
                for (int k = 0; k < 4; k++) sm += m[119 + k] * expf(fg * m[107 + 3 * k + ch]);
                la[ch] += fg * m[107 + 3 * kind + ch] - logf(fmaxf(sm, 1e-9f)) + g * v - lk_uni(g);
            }
            if (kind == 2) a_mul = 1.0f - fg * (1.0f - m[124]);       // quartz: glassy, a sharper specular
            if (kind == 3) a_mul = 1.0f - fg * (1.0f - m[125]);       // biotite: a shiny cleavage
        }
    }
    {
        float cell = m[96] > 0.0f ? 0.0f : m[7];
        float fg = fminf(fmaxf((cell / fmaxf(foot, 1e-9f) - 0.6f) / 1.4f, 0.0f), 1.0f);   // full at 2 px a grain
        if (cell > 0.0f && fg > 0.0f) {
            float u; unsigned int cls; float d1;
            lk_cells(q, cell, seed * 13 + 101, &u, &cls, &d1);
            float v = (u - 0.5f) * 3.4641f;                     // uniform -> unit SD
            float c = fg * m[8];
            float ca = lk_u01(cls);
            float kd = fg * m[22];
            float uh = lk_u01(lk_h32(cls ^ 0x2545f491u));
            float hv = (uh * uh - 0.33333f) * 3.354f;   // the grain's own hue (unit SD), skewed: a long tail toward the
                                                        // coating colour, a short one toward clean grey quartz
            float t = fminf(fmaxf((d1 - m[15]) / 0.22f, 0.0f), 1.0f);
            t = t * t * (3.0f - 2.0f * t);
            float kp = fg * m[14];
            bool clear = m[47] > 0.0f && lk_u01(lk_h32(cls ^ 0x68e31da4u)) < m[47];
            float kc = clear ? fg * -0.2f : 0.0f;                // log 0.82: the light that goes into a clear grain
            if (clear) { hv = -1.118f; a_mul = 1.0f - 0.75f * fg * (1.0f - t); }   // clean, and glassy on its face
            for (int ch = 0; ch < 3; ch++) {
                float g = c * m[9 + ch];
                float hd = (m[44] != 0.0f || m[45] != 0.0f || m[46] != 0.0f) ? m[44 + ch] : m[18 + ch];
                float gh = fg * m[43] * hd;
                la[ch] += g * v + gh * hv + ((ca < m[21]) ? kd : 0.0f) + kp * t + kc;
                la[ch] -= lk_uni(g) + lk_two(m[21], kd) + lk_two(m[35], kp * m[36]) + lk_two(m[47], fg * -0.2f);
                var[ch] += gh * gh;                                  // the skewed hue: its mean taken out to 2nd order
            }
        }
        float fcell = m[12];
        float ff = fminf(fmaxf((fcell / fmaxf(foot, 1e-9f) - 0.6f) / 1.4f, 0.0f), 1.0f);
        if (fcell > 0.0f && ff > 0.0f) {
            float u; unsigned int cls; float d1;
            lk_cells(q, fcell, seed * 17 + 211, &u, &cls, &d1);
            float v = (u - 0.5f) * 3.4641f, c = ff * m[13];
            for (int ch = 0; ch < 3; ch++) { float g = c * m[9 + ch]; la[ch] += g * v - lk_uni(g); }
        }
    }
    // grain clusters: packing and sorting vary over a few grains (lightness along the grain direction)
    if (m[37] > 0.0f && m[38] > 0.0f) {
        float v = lk_fbm(q, m[37], 2, seed * 37 + 701, foot);
        for (int ch = 0; ch < 3; ch++) { float g = m[38] * m[9 + ch]; la[ch] += g * v; var[ch] += g * g; }
    }
    // rust spots (iron oxide), sparse, a mm or two across
    if (m[30] > 0.0f && m[31] > 0.0f) {
        float fs = fminf(fmaxf((m[30] * m[32] / fmaxf(foot, 1e-9f) - 0.5f) / 1.5f, 0.0f), 1.0f);
        if (fs > 0.0f) {
            float w = lk_spots(q, m[30], m[31], m[32], seed * 29 + 503);
            float phi = m[31] * 4.18879f * m[32] * m[32] * m[32];          // the spots' volume share (= area share, Delesse)
            for (int ch = 0; ch < 3; ch++) {
                float k = fs * m[33] * m[18 + ch];
                la[ch] += k * w - lk_band(fminf(phi, 0.9f), k);
            }
        }
    }
    // iron staining (11-14): PATCHES (cm-scale zones richer in iron oxide, orange against yellow, fairly sharp edges)
    // and LIESEGANG BANDS (thin rhythmic rings: the level sets of a smooth random field, so they nest and wander as on a
    // cut face), both a composition of the stone's volume along the stain direction; their means are taken back out
    {
        float sd[3] = {m[48], m[49], m[50]};
        if (sd[0] == 0.0f && sd[1] == 0.0f && sd[2] == 0.0f) { sd[0] = m[18]; sd[1] = m[19]; sd[2] = m[20]; }
        float v = 0.0f;                                      // the patch field (unit SD): patches and rings read it
        if (m[51] > 0.0f) {
            v = lk_fbm(q, m[51], 3, seed * 41 + 809, foot) + 0.18f * lk_fbm(q, m[51] / 6.0f, 2, seed * 41 + 811, foot);
            v *= 0.984f;                                     // the sum's SD back to 1 (1 / sqrt(1 + 0.18^2))
        }
        if (m[51] > 0.0f && m[52] != 0.0f) {
            float t = fminf(fmaxf((v - m[53]) / 1.0f + 0.5f, 0.0f), 1.0f);   // diffuse edges (R1's stains)
            t = t * t * (3.0f - 2.0f * t);
            float phi = 0.5f * erfcf(m[53] * 0.70710678f);          // the share above the threshold (unit normal)
            for (int ch = 0; ch < 3; ch++) { float k = m[52] * sd[ch]; la[ch] += k * t - lk_two(phi, k); }
        }
        // LIESEGANG RINGS (11-14, by reference: the Quedlinburg and Aztec sandstones, R1's halo): bands precipitated
        // BEHIND the patch's iron front and parallel to it, so the level sets of the patch field itself, inside the
        // patch: dr = rings behind the edge (m[58] rings per unit of the field, the spacing following the front's
        // steepness); each ring a rise from the last one's tail (0.5 w of the spacing) then a graded tail (2 w); each
        // its own strength; the zone m[57] rings deep, fading inward. Their mean over the volume (m[59], look.py) out.
        if (m[51] > 0.0f && m[54] > 0.0f && m[55] != 0.0f && m[57] > 0.0f) {
            float fb = fminf(fmaxf((m[54] * m[56] / fmaxf(foot, 1e-9f) - 0.5f) / 1.5f, 0.0f), 1.0f);
            if (fb > 0.0f) {
                float dr = (v - m[53]) * m[58] + 0.25f * lk_fbm(q, 5.0f * m[54], 2, seed * 43 + 911, foot);
                float b = 0.0f;
                if (dr > 0.0f && dr < m[57]) {
                    float kr = floorf(dr), u = dr - kr;
                    float wf = 0.5f * m[56], wb = 2.0f * m[56];
                    float j0 = 0.3f + 1.4f * lk_u01(lk_hc((int)kr, 3, 5, seed * 43 + 913));
                    if (u < wf) {
                        float jp = kr > 0.5f ? (0.3f + 1.4f * lk_u01(lk_hc((int)kr - 1, 3, 5, seed * 43 + 913)))
                                               * expf(-(1.0f - wf) / wb) : 0.0f;
                        float s = u / wf;
                        s = s * s * (3.0f - 2.0f * s);
                        b = jp + (j0 - jp) * s;
                    } else {
                        b = j0 * expf(-(u - wf) / wb);
                    }
                    b *= 1.0f - dr / m[57];
                }
                for (int ch = 0; ch < 3; ch++) {
                    float k = fb * m[55] * sd[ch];
                    la[ch] += k * b - lk_two(m[59], k);
                }
            }
        }
    }
    // mottling (iron staining, cm scale): faint
    if (m[16] > 0.0f && m[17] > 0.0f) {
        float v = lk_fbm(q, m[16], 4, seed * 19 + 307, foot);
        for (int ch = 0; ch < 3; ch++) { la[ch] += m[17] * v * m[18 + ch]; var[ch] += m[17] * m[17] * m[18 + ch] * m[18 + ch]; }
    }
    // laminae: layers of random thickness along the bed normal, the bed gently wavy
    if (m[23] > 0.0f && (m[24] > 0.0f || m[34] > 0.0f)) {
        float zb = q.x * m[25] + q.y * m[26] + q.z * m[27];
        if (m[28] > 0.0f) zb += m[28] * lk_fbm(q, 40.0f * m[23], 3, seed * 23 + 401, foot);
        float v = lk_layers(zb, m[23], seed * 31 + 601, foot);
        for (int ch = 0; ch < 3; ch++) {
            float a = m[24] * m[18 + ch] - m[34];
            la[ch] += a * v;
            var[ch] += a * a;
        }
    }
    // the fired SKIN (11-14): a brick's face is its skin; deeper than the skin below the AS-FIRED form (the object's
    // base primitive, a rounded box in its own frame: where a chip or a crack has cut through) the body shows, paler
    if (m[64] > 0.5f) {
        float ax = fabsf(q.x) - m[65] + m[68], ay = fabsf(q.y) - m[66] + m[68], az = fabsf(q.z) - m[67] + m[68];
        float px = fmaxf(ax, 0.0f), py = fmaxf(ay, 0.0f), pz = fmaxf(az, 0.0f);
        float d = -(sqrtf(px * px + py * py + pz * pz) + fminf(fmaxf(ax, fmaxf(ay, az)), 0.0f) - m[68]);
        float t = fminf(fmaxf((d - m[69]) / fmaxf(m[70], 1e-6f) + 0.5f, 0.0f), 1.0f);
        t = t * t * (3.0f - 2.0f * t);
        for (int ch = 0; ch < 3; ch++) la[ch] += t * m[71 + ch];
    }
    // WEATHERING (11-14): the exposed surface's measured colour shift (slots 77-79, already x the state w), uneven over
    // the object (a patch field at slot 80, strength 81); the arithmetic mean stays the measured shift (var)
    if (m[77] != 0.0f || m[78] != 0.0f || m[79] != 0.0f) {
        float v = (m[80] > 0.0f && m[81] > 0.0f) ? lk_fbm(q, m[80], 3, seed * 47 + 1013, foot) : 0.0f;
        for (int ch = 0; ch < 3; ch++) {
            la[ch] += m[77 + ch] * (1.0f + m[81] * v);
            var[ch] += (m[81] * m[77 + ch]) * (m[81] * m[77 + ch]);
        }
    }
    for (int ch = 0; ch < 3; ch++) alb[ch] = fminf(fmaxf(expf(la[ch] - 0.5f * var[ch]), 0.005f), 0.95f);   // var: the fBm terms (normal)
    // the DEPOSIT (11-14, update 14): 128 amount (0..1)  129-131 its albedo (linear)  132 up-facing bias  133 pore fill
    // 134 patch wavelength (m)  135 patch strength  136 arris clearing width (m; the as-made box, slots 65-68)
    float dep = 0.0f;
    if (m[128] > 0.0f) {
        // 11-14 round 2 (sheet_12: an opaque coat read as paint): a THIN FILM that lets the stone show through
        // (coverage 1 - exp(-tau), at most 0.55) and the relief's HOLLOWS filled up to a level that rises with the
        // amount (the geometry decides where the dust lies); more on tops, softer larger patches
        float up = fminf(fmaxf(n.y, -1.0f), 1.0f);                       // world up is +y
        // EXPOSURE (137: 0 sheltered .. 1 out in the rain): sheltered dust settles on tops; rain
        // washes exposed tops, runs down the sides in streaks (along the object's up, the as-made box's y) and
        // gathers at the base (the splash zone)
        float ex = m[137];
        float tau = m[128] * fmaxf(1.0f + m[132] * up * (1.0f - 2.0f * ex), 0.0f);
        if (ex > 0.0f) {
            float side = 1.0f - fabsf(up);
            // sides washed to ~40 %, except SPARSE run-off streaks (the noise above 0.9 SD: ~18 % of the face), soft
            float hb0 = m[66] > 0.0f ? fminf(fmaxf((q.y + m[66]) / (2.0f * m[66]), 0.0f), 1.0f) : 0.5f;
            float st = 0.0f, src = 0.0f;
            if (m[149] > 0.0f && m[65] > 0.0f) {
                // update 19 (streaks from the FORM, not a noise row): water on the top leaves where
                // the top arris is bitten (the crumble's bites: its own bite function, evaluated on the arris line
                // above this point); deeper bites run further down (the run thins), opacity varies along the run
                bool zf = fabsf(q.z) / m[67] > fabsf(q.x) / m[65];
                float3 a = zf ? mk(q.x, m[66], copysignf(m[67], q.z)) : mk(copysignf(m[65], q.x), m[66], q.z);
                // update 21: the run DRIFTS a little along the edge as it descends (where the face leads it), so the
                // source is read up the run's own path, not straight above
                float dr = 0.004f * (1.0f - hb0) * lk_noise3(a.x / 0.006f + 0.2f, a.y / 0.006f, a.z / 0.006f + 0.8f, seed * 67 + 1717) * 3.71f;
                if (zf) a.x += dr; else a.z += dr;
                // update 25 (branching): below ~65 % of the height some runs SPLIT: the source is read at
                // two points along the edge moving apart as the run descends (where a smooth per-run field says it
                // branches), so one run divides into two diverging branches
                float bb = fmaxf(lk_noise3(a.x / 0.01f + 0.6f, 0.3f, a.z / 0.01f + 0.2f, seed * 67 + 1721) * 3.71f - 0.2f, 0.0f);
                float sp_ = fmaxf(0.65f - hb0, 0.0f) / 0.65f;
                float del = 0.004f * fminf(bb, 1.5f) * sp_ * sp_;
                if (del > 1e-5f) {
                    float3 a1 = a, a2 = a;
                    if (zf) { a1.x += del; a2.x -= del; } else { a1.z += del; a2.z -= del; }
                    src = fmaxf(mat_runoff_src(m, a1), mat_runoff_src(m, a2));
                } else {
                    src = mat_runoff_src(m, a);
                }
                float thr = 0.25f + 0.6f * (1.0f - hb0);                       // further down only the deepest go on
                thr += 0.12f * lk_noise3(a.x / 0.01f + 0.4f, q.y / 0.012f + 0.9f, a.z / 0.01f + 0.1f, seed * 67 + 1719) * 3.71f;   // width varies along the run
                float t = fminf(fmaxf((src - thr) / 0.4f, 0.0f), 1.0f);              // soft edges (round 2)
                st = t * t * (3.0f - 2.0f * t);
                float va = lk_noise3(q.x / 0.004f + 0.5f, q.y / 0.02f + 0.1f, q.z / 0.004f + 0.7f, seed * 67 + 1713) * 3.71f;
                st *= fminf(fmaxf(0.75f + 0.3f * va, 0.2f), 1.2f);
            } else {
                // (no box / no crumble: the round-3 noise row) soft-edged, fading downward from the top edge
                float wx = 0.002f * lk_noise3(q.y / 0.05f + 0.2f, q.x / 0.05f, q.z / 0.05f + 0.4f, seed * 67 + 1711) * 3.71f;
                float sk = lk_noise3((q.x + wx) / 0.008f + 0.3f, q.y / 0.09f + 0.6f, (q.z + wx) / 0.008f + 0.9f, seed * 67 + 1709) * 3.71f;
                st = fminf(fmaxf((sk - 0.6f) / 1.4f, 0.0f), 1.0f);
                st = st * st * (3.0f - 2.0f * st);
                st *= 0.35f + 0.65f * hb0;
                src = 0.5f;
            }
            tau *= 1.0f + ex * side * (0.4f + 1.1f * st - 1.0f);
            if (m[66] > 0.0f) {
                float hb = (q.y + m[66]) / (2.0f * m[66]);                // 0 at the base .. 1 at the top
                float bz = fminf(fmaxf(1.0f - hb / 0.35f, 0.0f), 1.0f);
                tau *= 1.0f + ex * 1.5f * bz * bz * (0.6f + 0.8f * fminf(src, 1.0f));   // pools under the runs
            }
        }
        if (m[134] > 0.0f) tau *= fmaxf(1.0f + m[135] * lk_fbm(q, m[134], 3, seed * 53 + 1409, foot), 0.0f);
        float c = 0.45f * (1.0f - expf(-tau));
        if (m[133] > 0.0f && mat_micro_amp(vol) > 0.0f) {
            float hgt = mat_micro(vol, q, foot) / fmaxf(mat_micro_amp(vol), 1e-9f);   // ~ -0.5 .. 1.3
            float t = fminf(fmaxf((-0.6f + 0.9f * tau - hgt) / 0.35f, 0.0f), 1.0f);   // round 3: the level rises slower
            c = fmaxf(c, 0.85f * m[133] * t * t * (3.0f - 2.0f * t));
        }
        if (m[136] > 0.0f && m[65] > 0.0f) {                              // exposed arrises wiped clean
            float ax = fabsf(q.x) - m[65], ay = fabsf(q.y) - m[66], az = fabsf(q.z) - m[67];
            float e = fminf(sqrtf(ax * ax + ay * ay + fmaxf(az, 0.0f) * fmaxf(az, 0.0f)),
                            fminf(sqrtf(ax * ax + az * az + fmaxf(ay, 0.0f) * fmaxf(ay, 0.0f)),
                                  sqrtf(ay * ay + az * az + fmaxf(ax, 0.0f) * fmaxf(ax, 0.0f))));
            float t = fminf(fmaxf(e / m[136], 0.0f), 1.0f);
            c *= t * t * (3.0f - 2.0f * t);
        }
        dep = fminf(fmaxf(c, 0.0f), 1.0f);
        for (int ch = 0; ch < 3; ch++) alb[ch] = (1.0f - dep) * alb[ch] + dep * m[129 + ch];
        a_mul = a_mul + dep * (1.0f / fmaxf(m[4], 0.06f) - a_mul);        // the deposit is matte (alpha -> 1)
    }
    // the key: Oren-Nayar diffuse (no wrap) and GGX specular
    float3 Ls = mk(L[0], L[1], L[2]);
    float ndv = fmaxf(dot3(n, vd), 1e-4f), nl = fmaxf(ndl, 0.0f);
    float s2 = m[6] * m[6];
    float A = 1.0f - 0.5f * s2 / (s2 + 0.33f), B = 0.45f * s2 / (s2 + 0.09f);
    float3 lp = sub(Ls, mul(n, ndl)), vp = sub(vd, mul(n, ndv));
    float lpl = sqrtf(dot3(lp, lp)), vpl = sqrtf(dot3(vp, vp));
    float cphi = (lpl > 1e-5f && vpl > 1e-5f) ? fmaxf(dot3(lp, vp) / (lpl * vpl), 0.0f) : 0.0f;
    float si = sqrtf(fmaxf(1.0f - nl * nl, 0.0f)), sv = sqrtf(fmaxf(1.0f - ndv * ndv, 0.0f));
    float sina = nl < ndv ? si : sv, tanb = nl < ndv ? sv / fmaxf(ndv, 1e-4f) : si / fmaxf(nl, 1e-4f);
    *wd = nl * (A + B * cphi * sina * tanb);
    float3 hv = nrm(add(Ls, vd));
    float ndh = fmaxf(dot3(n, hv), 0.0f), vdh = fmaxf(dot3(vd, hv), 0.0f);
    float ae = fmaxf(m[4] * a_mul, 0.06f);
    float a2 = fmaxf(ae * ae, 1e-6f);
    float dd = ndh * ndh * (a2 - 1.0f) + 1.0f;
    float D = a2 / (3.14159265f * dd * dd);
    float Vis = 0.5f / fmaxf(nl * sqrtf(ndv * ndv * (1.0f - a2) + a2) + ndv * sqrtf(nl * nl * (1.0f - a2) + a2), 1e-6f);
    float F = m[5] + (1.0f - m[5]) * powf(1.0f - vdh, 5.0f);
    *spec = nl > 0.0f ? D * Vis * F * nl : 0.0f;
    // granite: the crystal's own plane reflects (feldspar cleavage, biotite basal): a narrow lobe about its normal
    if (gk >= 0 && gk != 2 && nl > 0.0f) {
        float kg = gk == 3 ? m[139] : m[138];
        if (kg > 0.0f) {
            const float* MI = vf + 12;
            float3 cw = nrm(mk(MI[0] * gcn.x + MI[1] * gcn.y + MI[2] * gcn.z, MI[4] * gcn.x + MI[5] * gcn.y + MI[6] * gcn.z,
                               MI[8] * gcn.x + MI[9] * gcn.y + MI[10] * gcn.z));
            if (dot3(cw, n) < 0.0f) cw = mul(cw, -1.0f);   // a plane: the side facing out
            float cg = fmaxf(dot3(cw, hv), 0.0f), ag2 = m[140] * m[140];
            float dg = cg * cg * (ag2 - 1.0f) + 1.0f;
            *spec += kg * ag2 / (3.14159265f * dg * dg) * Vis * F * nl;
        }
    }
    // the grain relief's own shadow (along the key) and its occlusion of the sky (four azimuths, cosine-weighted)
    if (mat_micro_amp(vol) > 0.0f && ((m[39] > 0.0f && mat_micro_fade(m, foot) > 0.0f) ||
                                      (mat_frac_t(m, q) > 0.0f && m[75] > 2.0f * foot))) {
        float3 nl3 = nrm(mk(M[0] * n.x + M[1] * n.y + M[2] * n.z, M[4] * n.x + M[5] * n.y + M[6] * n.z,
                            M[8] * n.x + M[9] * n.y + M[10] * n.z));
        float3 Ll = nrm(mk(M[0] * Ls.x + M[1] * Ls.y + M[2] * Ls.z, M[4] * Ls.x + M[5] * Ls.y + M[6] * Ls.z,
                           M[8] * Ls.x + M[9] * Ls.y + M[10] * Ls.z));
        float sv = mat_micro_vis(vol, q, nl3, Ll, foot);
        *wd *= sv;
        *spec *= sv;
        float3 t1 = fabsf(nl3.x) > 0.5f ? mk(0, 1, 0) : mk(1, 0, 0);
        float3 bu = nrm(mk(nl3.y * t1.z - nl3.z * t1.y, nl3.z * t1.x - nl3.x * t1.z, nl3.x * t1.y - nl3.y * t1.x));
        float3 bv = mk(nl3.y * bu.z - nl3.z * bu.y, nl3.z * bu.x - nl3.x * bu.z, nl3.x * bu.y - nl3.y * bu.x);
        float occ = 0.0f;
        for (int a = 0; a < 4; a++) {
            float ph = 1.5707963f * a + 0.7853982f;
            float3 dl = nrm(add(add(mul(bu, 0.7071f * cosf(ph)), mul(bv, 0.7071f * sinf(ph))), mul(nl3, 0.7071f)));
            occ += mat_micro_vis(vol, q, nl3, dl, foot);
        }
        *sky = 0.25f * occ;
    }
    return true;
}
// ---- end fs-materials look ---------------------------------------------------------------------------------------------

// distance along d (local) to the exit of the current 32 mm tile, + a small push
__device__ __forceinline__ float tile_exit(const float* vf, float3 p, float3 d) {
    float tm = vf[VF_TM], best = 1e30f;
    float pp[3] = {p.x, p.y, p.z}, dd[3] = {d.x, d.y, d.z};
    for (int a = 0; a < 3; a++) {
        float q = (pp[a] - vf[VF_O + a]) / tm, f = floorf(q);
        if (dd[a] > 1e-9f) best = fminf(best, ((f + 1.0f) - q) * tm / dd[a]);
        else if (dd[a] < -1e-9f) best = fminf(best, (q - f) * tm / -dd[a]);
    }
    return best + 1e-5f;
}

__device__ __forceinline__ int lod_level(const float* vf, const int* vi, float foot_local) {
    // finest level whose h is >= half the pixel footprint: detail under half a pixel is not traced
    int l = 0;
    while (l < NLEV - 1 && (vi[VI_BPT + l] == 0 || vf[VF_H + l] < 0.5f * foot_local)) l++;
    while (l > 0 && vi[VI_BPT + l] == 0) l--;
    return l;
}

// Sphere trace one volume over [t0, t1] (world t; local = world * s). Returns hit t or -1.
__device__ float trace_vol(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                           float3 ow, float3 dw, float t0, float t1, float pixang, float lod_bias, int* lev_out, int max_steps) {
    const float* M = vf + VF_M;
    float s = vf[VF_S];
    float3 o = xf_p(M, ow), d = nrm(xf_d(M, dw));
    float a = t0 * s, b = t1 * s;              // local t
    if (!box_clip(o, d, vf, a, b)) return -1.0f;
    float t = a, tprev = a, fprev = 1e30f;
    int lev, emp, near_miss = 0;
    const int mvol = mat_vol(vf);                       // fs-materials 11-14: the grain relief (0 without a material)
    const float mA = mat_micro_amp(mvol);
    for (int it = 0; it < max_steps && t <= b; it++) {
        float3 p = add(o, mul(d, t));
        float foot = t * pixang * lod_bias;
        int lmin = lod_level(vf, vi, foot);
        float f = vol_f(vf, vi, grid, ttab, sdf, p, lmin, &lev, &emp);
        if (mA > 0.0f && !emp && lev >= 0 && f < 2.0f * mA) f -= mat_micro(mvol, p, foot);
        if (emp) { tprev = t; fprev = 1e30f; t += fmaxf(fminf(tile_exit(vf, p, d), mv_ray_enter(vi, p, d) + 1e-5f), f < 1e2f ? f : 0.0f); continue; }
        if (lev == -2) {                              // masked by an editable object's bricks
            // 11-09 (holes under her chin after a large Grab): mask_step is the distance to the masked brick's NEAREST
            // face, so a ray grazing along a face crept in tiny steps, ran out of steps and missed her (her store,
            // unmasked, hit within 20-98 um there): the ray leaves the masked brick along its own direction instead
            // (this volume has no surface inside a masked brick)
            const float* mf = g_vft + vi[VI_MASK] * VF;
            float bmm = 8.0f * mf[VF_H + 2], ex = 1e30f;
            float pp[3] = {p.x, p.y, p.z}, dd[3] = {d.x, d.y, d.z};
            for (int a3 = 0; a3 < 3; a3++) {
                float q = (pp[a3] - mf[VF_O + a3]) / bmm, fl = floorf(q);
                if (dd[a3] > 1e-9f) ex = fminf(ex, ((fl + 1.0f) - q) * bmm / dd[a3]);
                else if (dd[a3] < -1e-9f) ex = fminf(ex, (q - fl) * bmm / -dd[a3]);
            }
            tprev = t; fprev = 1e30f; t += fmaxf(fminf(ex, 1e3f) + 2e-5f, f);
            continue;
        }
        if (lev == -1 && vi[VI_MASKER]) {
            // 11-09 (D1c misses): in an editable store object's volume a brick it lacks can lie right beside
            // her surface (her store draws it), so the band step jumped past the start of the next held brick, where the
            // edited surface was (the CPU surface 349 um before the card's hit): step at most to this brick's exit
            float bm = 8.0f * vf[VF_H + 2], ex = 1e30f;
            float pp[3] = {p.x, p.y, p.z}, dd[3] = {d.x, d.y, d.z};
            for (int a3 = 0; a3 < 3; a3++) {
                float q = (pp[a3] - vf[VF_O + a3]) / bm, fl = floorf(q);
                if (dd[a3] > 1e-9f) ex = fminf(ex, ((fl + 1.0f) - q) * bm / dd[a3]);
                else if (dd[a3] < -1e-9f) ex = fminf(ex, (q - fl) * bm / -dd[a3]);
            }
            tprev = t; fprev = 1e30f; t += fmaxf(fminf(f, ex + 1e-5f), fmaxf(0.25f * foot, 2e-6f * s) * 0.5f);
            continue;
        }
        float eps = fmaxf(0.25f * foot, 2e-6f * s);
        if (f < eps) {
            float ta = t, fa = f, tb = -1.0f, fb = 0.0f;
            if (f < 0.0f) {                                    // stepped past: the previous point brackets
                if (fprev < 1e29f) { ta = tprev; fa = fprev; tb = t; fb = f; }
                else { tb = t; fb = f; ta = t - eps; fa = vol_f(vf, vi, grid, ttab, sdf, add(o, mul(d, ta)), lmin, &lev, &emp);
                       if (mA > 0.0f && fa < 2.0f * mA) fa -= mat_micro(mvol, add(o, mul(d, ta)), foot); }
            } else {                                           // probe forward to a sign change (4 doublings)
                float st = fmaxf(f, eps * 0.5f);
                for (int k = 0; k < 4; k++) {
                    float tt = ta + st, ft = vol_f(vf, vi, grid, ttab, sdf, add(o, mul(d, tt)), lmin, &lev, &emp);
                    if (mA > 0.0f && ft < 2.0f * mA) ft -= mat_micro(mvol, add(o, mul(d, tt)), foot);
                    if (ft < 0.0f) { tb = tt; fb = ft; break; }
                    ta = tt; fa = ft; st *= 2.0f;
                    if (ft > 4.0f * eps) break;
                }
            }
            if (tb >= 0.0f && fa >= 0.0f) {
                for (int k = 0; k < 12; k++) {                 // bisection, then a secant inside the bracket
                    float tm = 0.5f * (ta + tb), fm = vol_f(vf, vi, grid, ttab, sdf, add(o, mul(d, tm)), lmin, &lev, &emp);
                    if (mA > 0.0f && fm < 2.0f * mA) fm -= mat_micro(mvol, add(o, mul(d, tm)), foot);
                    if (fm >= 0.0f) { ta = tm; fa = fm; } else { tb = tm; fb = fm; }
                }
                float th = (fa - fb) != 0.0f ? ta + (tb - ta) * fa / (fa - fb) : 0.5f * (ta + tb);
                vol_f(vf, vi, grid, ttab, sdf, add(o, mul(d, th)), lmin, &lev, &emp);
                *lev_out = lev;
                return th / s;
            }
#ifdef TRACE_DBG
            if (++near_miss > 64) { *lev_out = lev - 1000; return t / s; }   // 11-09 B2 diagnosis: marked (trace_scene)
#else
            if (++near_miss > 64) { *lev_out = lev; return t / s; }   // a ray held beside the surface: take it
#endif
            tprev = ta; fprev = fa; t = ta + fmaxf(fa, eps * 0.5f);
            continue;
        }
        tprev = t; fprev = f;
        t += fmaxf(f, eps * 0.5f);
    }
    return -1.0f;
}

// scene trace over all visible volumes: nearest hit
__device__ Hit trace_scene(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                           float3 o, float3 d, float tmax, float pixang, float lod_bias, int max_steps) {
    Hit h; h.t = tmax; h.vol = -1; h.lev = -1;
#ifdef TRACE_DBG
    h.nm = 0;
#endif
    for (int v = 0; v < nvol; v++) {
        const float* vf = vft + v * VF;
        if (vf[VF_VIS] < 0.5f) continue;
        int lev = -1;
        float t = trace_vol(vf, vit + v * VI, grid, ttab, sdf, o, d, 0.0f, h.t, pixang, lod_bias, &lev, max_steps);
#ifdef TRACE_DBG
        int nm = 0;
        if (lev <= -900) { nm = 1; lev += 1000; }
        if (t >= 0.0f && t < h.t) h.nm = nm;
#endif
        if (t >= 0.0f && t < h.t) { h.t = t; h.vol = v; h.lev = lev; }
    }
    return h;
}

// scene field (world units) for shadow / sky rays: min over volumes; outside a box, the box distance
#define MAXC 16
__device__ float scene_f(const int* cand, int ncand, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                         float3 pw, int lmin_req, float* tile_skip, float3 dw, int* exact) {
    float best = 1e30f; *tile_skip = 1e30f; *exact = 0;
    for (int c = 0; c < ncand; c++) {
        int v = cand[c];
        const float* vf = vft + v * VF; const int* vi = vit + v * VI;
        if (vf[VF_VIS] < 0.5f) continue;
        float s = vf[VF_S];
        float3 p = xf_p(vf + VF_M, pw);
        float dx = fmaxf(fmaxf(vf[VF_LO] - p.x, p.x - vf[VF_HI]), 0.0f), dy = fmaxf(fmaxf(vf[VF_LO + 1] - p.y, p.y - vf[VF_HI + 1]), 0.0f),
              dz = fmaxf(fmaxf(vf[VF_LO + 2] - p.z, p.z - vf[VF_HI + 2]), 0.0f);
        float bd = sqrtf(dx * dx + dy * dy + dz * dz);
        if (bd > 0.0f) { float b2 = (bd + 1e-4f) / s; if (b2 < best) { best = b2; *exact = 0; } continue; }
        int lev, emp;
        int lmin = lmin_req; while (lmin > 0 && vi[VI_BPT + lmin] == 0) lmin--;
        float f = vol_f(vf, vi, grid, ttab, sdf, p, lmin, &lev, &emp);
        if (lev == -2) {                              // masked: a step limit only - to the masked brick's exit along
            float3 d = nrm(xf_d(vf + VF_M, dw));      // the ray (11-09: the nearest-face step crept along faces)
            const float* mf = g_vft + vi[VI_MASK] * VF;
            float bmm = 8.0f * mf[VF_H + 2], ex = 1e30f;
            float pp[3] = {p.x, p.y, p.z}, dd[3] = {d.x, d.y, d.z};
            for (int a3 = 0; a3 < 3; a3++) {
                float q = (pp[a3] - mf[VF_O + a3]) / bmm, fl = floorf(q);
                if (dd[a3] > 1e-9f) ex = fminf(ex, ((fl + 1.0f) - q) * bmm / dd[a3]);
                else if (dd[a3] < -1e-9f) ex = fminf(ex, (q - fl) * bmm / -dd[a3]);
            }
            *tile_skip = fminf(*tile_skip, fmaxf(fminf(ex, 1e3f) + 2e-5f, f) / s);
            continue;
        }
        if (emp) { float3 d = nrm(xf_d(vf + VF_M, dw)); *tile_skip = fminf(*tile_skip, fmaxf(fminf(tile_exit(vf, p, d), mv_ray_enter(vi, p, d) + 1e-5f), f < 1e2f ? f : 0.0f) / s); continue; }
        if (f / s < best) { best = f / s; *exact = (lev >= 0); }
    }
    return best;
}

// milestone: the scene's true field value at a world point (volumes' stored samples only: masked and band-bound
// returns skipped); ok = some volume holds a brick there
// 11-10: the volume holding the point inside it (scene field < 0 there), or -1; *fv its value (world units)
__device__ int inside_vol(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                          float3 pw, float* fv) {
    float best = 1e30f; int bv = -1;
    for (int v = 0; v < nvol; v++) {
        const float* vf = vft + v * VF; const int* vi = vit + v * VI;
        if (vf[VF_VIS] < 0.5f) continue;
        float3 p = xf_p(vf + VF_M, pw);
        if (p.x < vf[VF_LO] || p.y < vf[VF_LO + 1] || p.z < vf[VF_LO + 2] || p.x > vf[VF_HI] || p.y > vf[VF_HI + 1] || p.z > vf[VF_HI + 2]) continue;
        int lev, emp, lm = 0;
        while (lm < NLEV - 1 && vi[VI_BPT + lm] == 0) lm++;
        float f = vol_f(vf, vi, grid, ttab, sdf, p, lm, &lev, &emp);
        if (lev >= 0 && f / vf[VF_S] < best) { best = f / vf[VF_S]; bv = v; }
    }
    *fv = best;
    return best < 0.0f ? bv : -1;
}

__device__ float scene_val(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                          float3 pw, int lmin, bool* ok) {
    float best = 1e30f; *ok = false;
    for (int v = 0; v < nvol; v++) {
        const float* vf = vft + v * VF; const int* vi = vit + v * VI;
        if (vf[VF_VIS] < 0.5f) continue;
        float3 p = xf_p(vf + VF_M, pw);
        if (p.x < vf[VF_LO] || p.y < vf[VF_LO + 1] || p.z < vf[VF_LO + 2] || p.x > vf[VF_HI] || p.y > vf[VF_HI + 1] || p.z > vf[VF_HI + 2]) continue;
        int lev, emp, lm = lmin;
        while (lm > 0 && vi[VI_BPT + lm] == 0) lm--;
        float f = vol_f(vf, vi, grid, ttab, sdf, p, lm, &lev, &emp);
        if (lev >= 0 && f / vf[VF_S] < best) { best = f / vf[VF_S]; *ok = true; }
    }
    return best;
}

// r32: one lattice node's value at level lev (global node ijk from the tile grid origin); false where no brick holds it
__device__ __forceinline__ bool node_at(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                                        int lev, int I, int J, int K, float* v) {
    if (lev < 0 || lev >= NLEV) return false;        // r46
    int bpt = vi[VI_BPT + lev];
    if (bpt == 0 || I < 0 || J < 0 || K < 0) return false;
    int bi = I >> 3, bj = J >> 3, bk = K >> 3;
    int tx = bi / bpt, ty = bj / bpt, tz = bk / bpt;
    if (tx >= vi[VI_NX] || ty >= vi[VI_NY] || tz >= vi[VI_NZ]) return false;
    int slot = grid[vi[VI_GOFF] + ((tx * vi[VI_NY] + ty) * vi[VI_NZ] + tz) * NLEV + lev];
    if (slot < 0) return false;
    int bid = ttab[vi[VI_TOFF + lev] + slot * bpt * bpt * bpt + ((bi - tx * bpt) * bpt + (bj - ty * bpt)) * bpt + (bk - tz * bpt)];
    if (bid < 0) return false;
    *v = __half2float(sdf[(size_t)(bid + vi[VI_BOFF + lev]) * 729 + ((I & 7) * 9 + (J & 7)) * 9 + (K & 7)]);
    return true;
}

// r32: the shading normal from the gradient of the C1 Catmull-Rom reconstruction over the 4^3 nodes around the point
// (it interpolates the nodes, so the field's node values are unchanged; the trilinear field's normal jumps at every
// cell face: the faint lattice-period terracing). Falls back to central differences where a node is missing.
__device__ bool c1_grad(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, int lev,
                        float3 p, float3* g) {
    float h = vf[VF_H + lev];
    float ux = (p.x - vf[VF_O]) / h, uy = (p.y - vf[VF_O + 1]) / h, uz = (p.z - vf[VF_O + 2]) / h;
    int ix = (int)floorf(ux), iy = (int)floorf(uy), iz = (int)floorf(uz);
    float t[3] = {ux - ix, uy - iy, uz - iz}, w[3][4], dw[3][4];
    for (int a = 0; a < 3; a++) {
        float x = t[a], x2 = x * x, x3 = x2 * x;
        w[a][0] = 0.5f * (-x3 + 2.0f * x2 - x); w[a][1] = 0.5f * (3.0f * x3 - 5.0f * x2 + 2.0f);
        w[a][2] = 0.5f * (-3.0f * x3 + 4.0f * x2 + x); w[a][3] = 0.5f * (x3 - x2);
        dw[a][0] = 0.5f * (-3.0f * x2 + 4.0f * x - 1.0f); dw[a][1] = 0.5f * (9.0f * x2 - 10.0f * x);
        dw[a][2] = 0.5f * (-9.0f * x2 + 8.0f * x + 1.0f); dw[a][3] = 0.5f * (3.0f * x2 - 2.0f * x);
    }
    float gx = 0, gy = 0, gz = 0;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++) {
        float v;
        if (!node_at(vf, vi, grid, ttab, sdf, lev, ix - 1 + i, iy - 1 + j, iz - 1 + k, &v)) return false;
        gx += dw[0][i] * w[1][j] * w[2][k] * v;
        gy += w[0][i] * dw[1][j] * w[2][k] * v;
        gz += w[0][i] * w[1][j] * dw[2][k] * v;
    }
    *g = mk(gx, gy, gz);
    return true;
}

__device__ float3 vol_normal(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, float3 pw, int lev) {
    {
        float3 pl = xf_p(vf + VF_M, pw);
        float3 g;
        int l0 = lev < 0 ? 2 : lev;
        if (mv_grad(vf, vi, grid, ttab, sdf, l0, pl, &g)) return nrm(xf_d(vf + VF_MI, g));   // fs-brush 11-13: inside a move
        if (c1_grad(vf, vi, grid, ttab, sdf, l0, pl, &g)) return nrm(xf_d(vf + VF_MI, g));
    }
    float3 p = xf_p(vf + VF_M, pw);
    if (lev < 0) lev = 2;
    float e = 0.75f * vf[VF_H + lev];
    int l, em;
    float gx = vol_f(vf, vi, grid, ttab, sdf, add(p, mk(e, 0, 0)), lev, &l, &em) - vol_f(vf, vi, grid, ttab, sdf, sub(p, mk(e, 0, 0)), lev, &l, &em);
    float gy = vol_f(vf, vi, grid, ttab, sdf, add(p, mk(0, e, 0)), lev, &l, &em) - vol_f(vf, vi, grid, ttab, sdf, sub(p, mk(0, e, 0)), lev, &l, &em);
    float gz = vol_f(vf, vi, grid, ttab, sdf, add(p, mk(0, 0, e)), lev, &l, &em) - vol_f(vf, vi, grid, ttab, sdf, sub(p, mk(0, 0, e)), lev, &l, &em);
    const float* MI = vf + VF_MI;
    return nrm(xf_d(MI, mk(gx, gy, gz)));       // local -> world (rigid + uniform scale: direction only)
}

// soft visibility toward the sun (FV8 sun_vis: one SDF soft-shadow march), and an occlusion test for sky rays
// candidates: the volumes whose world box the ray segment [0, tmax] can reach (the caller's list)
__device__ int ray_cands(int nvol, const float* wbox, const float* vft, float3 o, float3 d, float tmax, float grow, int* cand) {
    int n = 0;
    for (int v = 0; v < nvol && n < MAXC; v++) {
        if (vft[v * VF + VF_VIS] < 0.5f) continue;
        const float* B = wbox + 6 * v;
        float t0 = 0.0f, t1 = tmax;
        float oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
        bool ok = true;
        for (int a = 0; a < 3 && ok; a++) {
            float lo = B[a] - grow, hi = B[3 + a] + grow;
            if (fabsf(dd[a]) < 1e-12f) { if (oo[a] < lo || oo[a] > hi) ok = false; continue; }
            float ta = (lo - oo[a]) / dd[a], tb = (hi - oo[a]) / dd[a];
            if (ta > tb) { float q = ta; ta = tb; tb = q; }
            t0 = fmaxf(t0, ta); t1 = fminf(t1, tb);
            if (t0 > t1) ok = false;
        }
        if (ok) cand[n++] = v;
    }
    return n;
}

__device__ float march_vis(const int* cand, int ncand, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                           float3 o, float3 d, float tmax, float k, int lmin, int max_steps) {
    if (ncand == 0) return 1.0f;
    float t = 0.0f, res = 1.0f;
    for (int it = 0; it < max_steps && t < tmax; it++) {
        float skip; int exact;
        float3 p = add(o, mul(d, t));
        // the hit's own level near the start (a coarser level's surface can lie outside the fine one there, by up
        // to about a millimetre), one level coarser past 1 cm and two past 4 cm
        int lm = min(lmin + (t > 0.01f ? 1 : 0) + (t > 0.04f ? 1 : 0), NLEV - 1);
        float f = scene_f(cand, ncand, vft, vit, grid, ttab, sdf, p, lm, &skip, d, &exact);
        // fs-shell 11-12 (the cast shadow's edge stepped where the view, at the hit's level, is smooth):
        // a coarser level skips empty space, but within a few millimetres of a surface its occluder is blocky (and
        // often a lower bound, so the penumbra was not taken there); near a surface the occluder is read at the hit's
        // own level, so the shadow's edge follows the form the view draws and its soft edge is the sun's. k > 0: the
        // sun's soft shadow; k < 0 (fs-light's area-light rays, 11-12): hard lit / blocked with this same re-read and
        // no penumbra; k = 0: the sky's open / blocked test keeps its coarse reads, as before
        if (k != 0.0f && lm > lmin && f < 4e-3f) {
            float s2; int e2;
            float f2 = scene_f(cand, ncand, vft, vit, grid, ttab, sdf, p, lmin, &s2, d, &e2);
            if (f2 < 1e29f) { f = f2; skip = s2; exact = e2; }
        }
        if (skip < 1e29f && f > 1e29f) { t += skip; continue; }
        if (f >= 1e29f) break;
        if (f < 2e-5f) return 0.0f;
        // penumbra only from true field samples: a band / box lower bound is not a distance to an occluder
        if (k > 0.0f && exact) res = fminf(res, k * f / fmaxf(t, 1e-4f));
        float st = f;
        if (skip < 1e29f) st = fminf(st, skip);
        t += fmaxf(st, 2e-5f);
    }
    return fmaxf(res, 0.0f);
}

__device__ __forceinline__ unsigned int hash3(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int h = a * 0x8da6b343u ^ b * 0xd8163841u ^ c * 0xcb1ab31fu;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}
__device__ __forceinline__ float u01(unsigned int h) { return (h >> 8) * (1.0f / 16777216.0f); }

// equirect env lookup (Blender frame, z up): world y-up dir -> (x, -z, y), rotated about the up axis by (cr, sr)
__device__ float3 env_rad(const __half* env, int ew, int eh, float3 dw, float cr, float sr) {
    float bx = dw.x, by = -dw.z, bz = dw.y;
    float rx = cr * bx - sr * by, ry = sr * bx + cr * by;
    float u = 0.5f + atan2f(ry, rx) * (0.5f / 3.14159265f);
    float v = acosf(fminf(fmaxf(bz, -1.0f), 1.0f)) / 3.14159265f;
    int x = min(max((int)(u * ew), 0), ew - 1), y = min(max((int)(v * eh), 0), eh - 1);
    const __half* e = env + ((size_t)y * ew + x) * 3;
    return mk(__half2float(e[0]), __half2float(e[1]), __half2float(e[2]));
}

__device__ __forceinline__ float3 env_bilin(const __half* env, int ew, int eh, float3 dw, float cr, float sr) {
    float bx = dw.x, by = -dw.z, bz = dw.y;
    float rx = cr * bx - sr * by, ry = sr * bx + cr * by;
    float u = (0.5f + atan2f(ry, rx) * (0.5f / 3.14159265f)) * ew - 0.5f;
    float v = acosf(fminf(fmaxf(bz, -1.0f), 1.0f)) / 3.14159265f * eh - 0.5f;
    int x0 = (int)floorf(u), y0 = (int)floorf(v);
    float fx = u - x0, fy = v - y0;
    float3 c = mk(0, 0, 0);
    for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
        int x = ((x0 + i) % ew + ew) % ew, y = min(max(y0 + j, 0), eh - 1);
        float w = (i ? fx : 1.0f - fx) * (j ? fy : 1.0f - fy);
        const __half* e = env + ((size_t)y * ew + x) * 3;
        c = add(c, mul(mk(__half2float(e[0]), __half2float(e[1]), __half2float(e[2])), w));
    }
    return c;
}

// irradiance from the environment's 9 SH coefficients per channel (cosine-convolved on the host), world frame
__device__ __forceinline__ float3 sh_irr(const float* S, float3 n) {
    float b[9] = {0.282095f, 0.488603f * n.y, 0.488603f * n.z, 0.488603f * n.x, 1.092548f * n.x * n.y, 1.092548f * n.y * n.z,
                  0.315392f * (3.0f * n.z * n.z - 1.0f), 1.092548f * n.x * n.z, 0.546274f * (n.x * n.x - n.y * n.y)};
    float r = 0, g = 0, bb = 0;
    for (int i = 0; i < 9; i++) { r += S[i] * b[i]; g += S[9 + i] * b[i]; bb += S[18 + i] * b[i]; }
    return mk(fmaxf(r, 0.0f), fmaxf(g, 0.0f), fmaxf(bb, 0.0f));
}

// incremental re-light: outside the re-traced rectangle q, only pixels whose light the edit can change: within the
// sky reach of a changed box, or whose sun ray crosses it (boxes: lo, hi world, grown by the caller for the penumbra)
__device__ bool affected(int x, int y, float4 g0, float4 g1, const float* L, int nbox, const float* boxes,
                         int qx0, int qy0, int qx1, int qy1) {
    if (x >= qx0 && x < qx1 && y >= qy0 && y < qy1) return true;
    if (g1.w < 0.0f) return false;
    float3 p = mk(g0.x, g0.y, g0.z), Ls = mk(L[0], L[1], L[2]);
    for (int b = 0; b < nbox; b++) {
        const float* B = boxes + 6 * b;
        float dx = fmaxf(fmaxf(B[0] - p.x, p.x - B[3]), 0.0f), dy = fmaxf(fmaxf(B[1] - p.y, p.y - B[4]), 0.0f),
              dz = fmaxf(fmaxf(B[2] - p.z, p.z - B[5]), 0.0f);
        if (dx * dx + dy * dy + dz * dz < L[13] * L[13] * 1.0001f) return true;
        // the soft shadow reads occluders within t / k of the sun ray (res = k f / t), so the box is grown by the
        // penumbra width at its far side: (distance to the box's far corner) / k
        float cx = 0.5f * (B[0] + B[3]) - p.x, cy = 0.5f * (B[1] + B[4]) - p.y, cz = 0.5f * (B[2] + B[5]) - p.z;
        float rb = 0.5f * sqrtf((B[3] - B[0]) * (B[3] - B[0]) + (B[4] - B[1]) * (B[4] - B[1]) + (B[5] - B[2]) * (B[5] - B[2]));
        float g = (sqrtf(cx * cx + cy * cy + cz * cz) + rb) / fmaxf(L[12], 1.0f);
        float t0 = 0.0f, t1 = L[11];
        float oo[3] = {p.x, p.y, p.z}, dd[3] = {Ls.x, Ls.y, Ls.z};
        bool ok = true;
        for (int a = 0; a < 3 && ok; a++) {
            float lo_ = B[a] - g, hi_ = B[3 + a] + g;
            if (fabsf(dd[a]) < 1e-12f) { if (oo[a] < lo_ || oo[a] > hi_) ok = false; continue; }
            float ta = (lo_ - oo[a]) / dd[a], tb = (hi_ - oo[a]) / dd[a];
            if (ta > tb) { float q = ta; ta = tb; tb = q; }
            t0 = fmaxf(t0, ta); t1 = fminf(t1, tb);
            if (t0 > t1) ok = false;
        }
        if (ok) return true;
    }
    return false;
}

// pass A: trace a rectangle into the G-buffer. jitter index j (0 = pixel centre).
// P: camera P[0..2] pos, [3..5] fwd, [6..8] right, [9..11] up, [12] tan(fov/2), [13] aspect, [14] max steps, [15] tmax,
//    [16] lod bias.  G-buffer: gp (x, y, z, h_world) ; gn (nx, ny, nz, vol + lev/4, or -1: then the view direction)
extern "C" __global__ void trace_gbuf(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                      const float* P, int W, int H, int x0, int y0, int rw, int rh, int jit,
                                      float4* gp, float4* gn, float* depth, int* objid, int* lev_out, float* fhit,
                                      const float* L, int nbox, const float* boxes, int qx0, int qy0, int qx1, int qy1,
                                      unsigned char* rmask, const int* pcount, int aa_passes) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= rw || py >= rh) return;
    int x = x0 + px, y = y0 + py;
    if (x >= W || y >= H) return;
    size_t pi = (size_t)y * W + x;
    if (jit < 0) {                        // per-pixel progressive: this pixel's own pass count decides
        jit = pcount[pi];
        if (jit >= aa_passes) return;
        nbox = 0;
    }
    if (nbox > 0) {                       // the job's pixel set is decided once, at its first pass, and kept
        if (jit == 0) rmask[pi] = affected(x, y, gp[pi], gn[pi], L, nbox, boxes, qx0, qy0, qx1, qy1) ? 1 : 0;
        if (!rmask[pi]) return;
    }
    const float RG[8] = {0.375f, 0.125f, 0.875f, 0.375f, 0.625f, 0.875f, 0.125f, 0.625f};   // 4-sample rotated grid
    float jx = 0.5f, jy = 0.5f;
    if (jit > 0 && jit <= 4) { jx = RG[2 * (jit - 1)]; jy = RG[2 * (jit - 1) + 1]; }
    else if (jit > 4) { unsigned int hs = hash3(x, y, jit * 9781u + 17u); jx = u01(hs); jy = u01(hash3(hs, 3u, 7u)); }
    float sx = (2.0f * (x + jx) / W - 1.0f) * P[12] * P[13], sy = (1.0f - 2.0f * (y + jy) / H) * P[12];
    float3 o = mk(P[0], P[1], P[2]);
    float3 d = nrm(add(mk(P[3], P[4], P[5]), add(mul(mk(P[6], P[7], P[8]), sx), mul(mk(P[9], P[10], P[11]), sy))));
    float pixang = 2.0f * P[12] / H;
    if (P[17] > 0.0f) {   // fs-shell 11-11: orthographic (render/camera.py Camera.ray): parallel rays along f, each
                          // starting P[18] before the camera, offset across the target plane (P[17] = the distance to it);
                          // the pixel footprint is constant: pixang times the ray length near the target gives it
        float3 f = mk(P[3], P[4], P[5]);
        o = add(add(o, mul(add(mul(mk(P[6], P[7], P[8]), sx), mul(mk(P[9], P[10], P[11]), sy)), P[17])), mul(f, -P[18]));
        d = f;
        pixang = 2.0f * P[12] * P[17] / (H * (P[17] + P[18]));
    }
    // fs-shell 11-11 (design 4.4, the Section tool): the view's section plane P[19..21] = n, P[22] = d, P[23] = on: the
    // half n.p > d is cut from the view. A ray from that side starts at the plane (where the plane point is inside a
    // form, the cut-face branch below shows it, facing the cut side); a ray from the kept side stops at the plane.
    // P[23] = 0 (no section): every value below is what it was, so the pixels are too.
    float tsec = 0.0f, tfar = P[15];
    float3 sn = mk(0.0f, 0.0f, 0.0f);
    bool sec_start = false;
    // fs-shell 11-11 (design 4.8): P[23] is a bit field: 1 = the section, 2 = hidden boxes (P[24] = their count, P[25 +
    // 24 k ..] = box k as 6 planes n.x, n.y, n.z, d; inside where n.p <= d). P[23] = 0 or 1 reads as before.
    int clipf = (int)(P[23] + 0.5f);
    if (clipf & 1) {
        sn = mk(P[19], P[20], P[21]);
        float s0 = dot3(sn, o) - P[22], dn = dot3(sn, d);
        if (s0 > 0.0f) {
            if (dn >= 0.0f) {                 // the ray never reaches the kept side: background
                gp[pi] = make_float4(0, 0, 0, 0); gn[pi] = make_float4(d.x, d.y, d.z, -1.0f);
                if (jit == 0) { depth[pi] = 1e30f; objid[pi] = -1; lev_out[pi] = -1; fhit[pi] = 0.0f; }
                return;
            }
            tsec = -s0 / dn;
            o = add(o, mul(d, tsec));
            sec_start = true;
        } else if (dn > 0.0f) {
            tfar = fminf(tfar, -s0 / dn);
        }
    }
    // fs-shell-3 11-14 (fs-tools' connector in Section, win3): a section ray starts ON the plane, and
    // where that point is inside a form the pixel IS the cut face. The cut face was drawn only after a miss (below),
    // which holds for one form but not for SEPARATE forms in contact: a ray starting inside a pillar found the box's top
    // face where it lies under the pillar's foot (both F = 0 there, the union solid through) and drew that buried face,
    // in the pillar's shadow: a dark pocket the fields do not have. The start is tested first, as the CPU view does
    // (render/raymarch.py: the plane point through the field before any trace). A start inside a hidden box is left to
    // the box path below (the box is skipped first). Without a section nothing here runs.
    if (sec_start) {
        bool in_box = false;
        if (clipf & 2) {
            int nb0 = (int)(P[24] + 0.5f);
            for (int b = 0; b < nb0 && !in_box; b++) {
                const float* bp = P + 25 + 24 * b;
                bool ins = true;
                for (int a = 0; a < 6 && ins; a++)
                    if (bp[4 * a] * o.x + bp[4 * a + 1] * o.y + bp[4 * a + 2] * o.z - bp[4 * a + 3] > 0.0f) ins = false;
                in_box = ins;
            }
        }
        if (!in_box) {
            float fq0;
            int cv0 = inside_vol(nvol, vft, vit, grid, ttab, sdf, o, &fq0);
            if (cv0 >= 0) {
                const float* vfc = vft + cv0 * VF;
                gp[pi] = make_float4(o.x, o.y, o.z, vfc[VF_H] / vfc[VF_S]);
                gn[pi] = make_float4(sn.x, sn.y, sn.z, (float)cv0);
                if (jit == 0) { depth[pi] = tsec; objid[pi] = cv0; lev_out[pi] = 0; fhit[pi] = 0.0f; }
                return;
            }
        }
    }
    Hit hh;
    if (clipf & 2) {
        // box hide: the ray's parts inside a box are skipped; leaving a box inside a form shows the cut face there
        int nb = (int)(P[24] + 0.5f);
        float cur = 0.0f;
        hh.vol = -1;
        for (int it = 0; it < 32; it++) {
            bool jumped = false; float3 cn = mk(0.0f, 0.0f, 0.0f);
            for (int r = 0; r <= nb; r++) {
                bool moved = false;
                for (int b = 0; b < nb; b++) {
                    const float* bp = P + 25 + 24 * b;
                    float ta = -1e30f, tb = 1e30f; int ax = 0; bool never = false;
                    for (int a = 0; a < 6; a++) {
                        float3 pn = mk(bp[4 * a], bp[4 * a + 1], bp[4 * a + 2]);
                        float s0 = dot3(pn, o) - bp[4 * a + 3], dn = dot3(pn, d);
                        if (fabsf(dn) < 1e-30f) { if (s0 > 0.0f) never = true; }
                        else { float u = -s0 / dn; if (dn < 0.0f) ta = fmaxf(ta, u); else if (u < tb) { tb = u; ax = a; } }
                    }
                    if (!never && ta <= cur && cur < tb) {
                        cur = tb; moved = true; jumped = true;
                        cn = mk(-bp[4 * ax], -bp[4 * ax + 1], -bp[4 * ax + 2]);
                    }
                }
                if (!moved) break;
            }
            if (cur >= tfar) break;
            if (jumped) {
                float3 q = add(o, mul(d, cur));
                float fq;
                int cv = inside_vol(nvol, vft, vit, grid, ttab, sdf, q, &fq);
                if (cv >= 0) {
                    float3 n = nrm(cn);
                    const float* vfc = vft + cv * VF;
                    gp[pi] = make_float4(q.x, q.y, q.z, vfc[VF_H] / vfc[VF_S]);
                    gn[pi] = make_float4(n.x, n.y, n.z, (float)cv);
                    if (jit == 0) { depth[pi] = cur + tsec; objid[pi] = cv; lev_out[pi] = 0; fhit[pi] = 0.0f; }
                    return;
                }
            }
            float tn = tfar;
            for (int b = 0; b < nb; b++) {
                const float* bp = P + 25 + 24 * b;
                float ta = -1e30f, tb = 1e30f; bool never = false;
                for (int a = 0; a < 6; a++) {
                    float3 pn = mk(bp[4 * a], bp[4 * a + 1], bp[4 * a + 2]);
                    float s0 = dot3(pn, o) - bp[4 * a + 3], dn = dot3(pn, d);
                    if (fabsf(dn) < 1e-30f) { if (s0 > 0.0f) never = true; }
                    else { float u = -s0 / dn; if (dn < 0.0f) ta = fmaxf(ta, u); else tb = fminf(tb, u); }
                }
                if (!never && ta > cur && ta < tn && tb > ta) tn = ta;
            }
            Hit h2 = trace_scene(nvol, vft, vit, grid, ttab, sdf, add(o, mul(d, cur)), d, tn - cur, pixang, P[16], (int)P[14]);
            if (h2.vol >= 0) { hh = h2; hh.t += cur; break; }
            if (tn >= tfar) break;
            cur = tn;
        }
    } else {
        hh = trace_scene(nvol, vft, vit, grid, ttab, sdf, o, d, tfar, pixang, P[16], (int)P[14]);
    }
    if (hh.vol < 0) {
        // (a ray that starts inside solid comes back a miss: only then is the start tested; 11-10 r2, the test on
        // every pixel cost ~35 % of the frame)
            // 11-10 (a sculptor whose view passes her skin): the camera plane cuts her open as a clip plane would. Where this
            // pixel's ray starts INSIDE her (just in front of the camera), the pixel shows the cut face there, flat and
            // facing the camera, instead of a miss (a solid store showed the sky) or her inner faces; where it starts
            // outside (the hollow of an opening), it traces as before
            const float TNEAR = 0.0005f;
            float3 q = add(o, mul(d, TNEAR));
            float fq;
            int cv = inside_vol(nvol, vft, vit, grid, ttab, sdf, q, &fq);
            if (cv >= 0) {
                float3 n = sec_start ? sn : mul(d, -1.0f);      // fs-shell 11-11: a section's cut face faces the cut side
                float3 qq = sec_start ? o : q;                  // fs-shell 11-11: a section's face is ON its plane
                const float* vfc = vft + cv * VF;
                gp[pi] = make_float4(qq.x, qq.y, qq.z, vfc[VF_H] / vfc[VF_S]);
                gn[pi] = make_float4(n.x, n.y, n.z, (float)cv);
                if (jit == 0) { depth[pi] = sec_start ? tsec : TNEAR; objid[pi] = cv; lev_out[pi] = 0; fhit[pi] = 0.0f; }
                return;
            }
        gp[pi] = make_float4(0, 0, 0, 0); gn[pi] = make_float4(d.x, d.y, d.z, -1.0f);
        if (jit == 0) { depth[pi] = 1e30f; objid[pi] = -1; lev_out[pi] = -1; fhit[pi] = 0.0f; }
        return;
    }
    const float* vf = vft + hh.vol * VF; const int* vi = vit + hh.vol * VI;
    float3 p = add(o, mul(d, hh.t));
    float3 n;
    {
        // milestone: where the hit volume lacks a C1 node (an editable object's edited bricks end there), the
        // normal is the scene field's gradient (her store fills in), not that one volume's band bound
        float3 pl = xf_p(vf + VF_M, p), g;
        int l0 = hh.lev < 0 ? 2 : hh.lev;
        if (mv_grad(vf, vi, grid, ttab, sdf, l0, pl, &g)) n = nrm(xf_d(vf + VF_MI, g));   // fs-brush 11-13: inside a move
        else if (c1_grad(vf, vi, grid, ttab, sdf, l0, pl, &g)) n = nrm(xf_d(vf + VF_MI, g));
        else {
            float e = 0.75f * vf[VF_H + l0] / vf[VF_S], gg[3];
            bool ok = true;
            for (int a = 0; a < 3 && ok; a++) {
                float3 dd = mk(a == 0 ? e : 0.0f, a == 1 ? e : 0.0f, a == 2 ? e : 0.0f);
                bool o1, o2;
                float fa = scene_val(nvol, vft, vit, grid, ttab, sdf, add(p, dd), l0, &o1);
                float fb = scene_val(nvol, vft, vit, grid, ttab, sdf, sub(p, dd), l0, &o2);
                ok = o1 && o2; gg[a] = fa - fb;
            }
            n = ok ? nrm(mk(gg[0], gg[1], gg[2])) : vol_normal(vf, vi, grid, ttab, sdf, p, hh.lev);
        }
    }
    n = mat_micro_normal(vf, p, n, hh.t * pixang);   // fs-materials 11-14: the grain relief's gradient (no material: n)
    if (dot3(n, d) > 0.0f) n = mul(n, -1.0f);
    float hl = vf[VF_H + max(hh.lev, 0)] / vf[VF_S];
    gp[pi] = make_float4(p.x, p.y, p.z, hl);
    gn[pi] = make_float4(n.x, n.y, n.z, (float)hh.vol + 0.25f * (float)max(hh.lev, 0));
    if (jit == 0) {
        depth[pi] = hh.t + tsec; objid[pi] = hh.vol; lev_out[pi] = hh.lev;     // fs-shell 11-11: from the camera
#ifdef TRACE_DBG
        if (hh.nm) objid[pi] = hh.vol + 10000;     // 11-09 B2 diagnosis: the near-miss return took this hit
#endif
        int l2, em2; fhit[pi] = vol_f(vf, vi, grid, ttab, sdf, xf_p(vf + VF_M, p), hh.lev, &l2, &em2) / vf[VF_S];
    }
}

__device__ __forceinline__ unsigned long long sky_mix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// the cell key of a surface point: level l (cell c_l = base x 2^l from the pixel's footprint), the cell in world units,
// the volume, the normal's octant. 0 = no key (the point outside the key's range)
__device__ __forceinline__ unsigned long long sky_key(float3 p, float3 n, int vol, float dist) {
    float c = fmaxf(g_sky_par[2] * dist * g_sky_par[3], g_sky_par[0]);
    int l = min(max((int)ceilf(log2f(c / g_sky_par[0]) - 1e-4f), 0), 31);
    float cl = g_sky_par[0] * (float)(1u << l);
    if (vol < 0 || vol > 63) return 0ull;
    p = mk(p.x - g_sky_org[3 * vol], p.y - g_sky_org[3 * vol + 1], p.z - g_sky_org[3 * vol + 2]);   // n3b: from the volume's origin
    int ix = (int)floorf(p.x / cl) + 32768, iy = (int)floorf(p.y / cl) + 32768, iz = (int)floorf(p.z / cl) + 32768;
    if (ix < 0 || iy < 0 || iz < 0 || ix > 65535 || iy > 65535 || iz > 65535) return 0ull;
    unsigned long long oc = (n.x > 0.0f ? 1ull : 0ull) | (n.y > 0.0f ? 2ull : 0ull) | (n.z > 0.0f ? 4ull : 0ull);
    return (1ull << 63) | ((unsigned long long)l << 58) | (oc << 55) | ((unsigned long long)(vol & 63) << 49)
           | ((unsigned long long)ix << 32) | ((unsigned long long)iy << 16) | (unsigned long long)iz;   // bits: 63 used, 58-62 l, 55-57 octant, 49-54 vol, 32-47 / 16-31 / 0-15 cell
}
// find or insert a key (linear probing, 32 slots) -> slot, or -1 (the table is full along the probe)
__device__ __forceinline__ int sky_slot(unsigned long long key) {
    unsigned int h = (unsigned int)sky_mix(key) & g_sky_mask;
    for (int i = 0; i < 32; i++) {
        unsigned int s = (h + i) & g_sky_mask;
        unsigned long long old = atomicCAS(&g_sky_k[s], 0ull, key);
        if (old == 0ull || old == key) return (int)s;
    }
    return -1;
}

// pass B: light the G-buffer (sun soft shadow + sky visibility x SH irradiance, the live-light p1 terms), accumulate.
// L: [0..2] sun dir, [3..5] sun rgb, [6] cr, [7] sr, [8] exposure, [9] background gain, [10] wrap, [11] sun reach,
//    [12] penumbra k, [13] sky reach, [14] sky rays, [15] spec level, [16] spec exponent, [17..19] camera pos,
//    [22] sculpt headlight irradiance (white, from the camera, no shadow; 0 = off: review renders)
extern "C" __global__ void light_pass(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                      const __half* env, int ew, int eh, const float* L, const float* SH,
                                      int W, int H, int x0, int y0, int rw, int rh, int pass,
                                      const float4* gp, const float4* gn, float* accum, float* gvis, int aa_passes, const float* wbox,
                                      float* skyc, float* aob, const unsigned char* rmask, int* pcount, int target,
                                      int nbox, const float* boxes, int qx0, int qy0, int qx1, int qy1) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= rw || py >= rh) return;
    int x = x0 + px, y = y0 + py;
    if (x >= W || y >= H) return;
    size_t pi = (size_t)y * W + x;
    if (pass < 0) {                       // per-pixel progressive
        pass = pcount[pi];
        if (pass >= target) return;
        pcount[pi] = pass + 1;
        nbox = 0;
    }
    float4 g1 = gn[pi];
    if (nbox > 0 && !rmask[pi]) return;
    float3 col;
    float cr = L[6], sr = L[7];
    if (g1.w < 0.0f) {
        col = mul(env_bilin(env, ew, eh, mk(g1.x, g1.y, g1.z), cr, sr), L[9]);
        if (pass == 0) { float* sc = skyc + pi * 4; sc[0] = sc[1] = sc[2] = 0.0f; sc[3] = 1.0f; aob[pi * 2] = 0.0f; aob[pi * 2 + 1] = 0.0f; }
    } else {
        float4 g0 = gp[pi];
        int vol = (int)g1.w, lev = (int)((g1.w - vol) * 4.0f + 0.5f);
        const float* vf = vft + vol * VF;
        float3 p = mk(g0.x, g0.y, g0.z), n = mk(g1.x, g1.y, g1.z);
        float hl = g0.w;
        float3 ps = add(p, mul(n, 3.0f * hl + 1e-4f));
        int lsh = lev;
        float3 Ls = mk(L[0], L[1], L[2]);
        float ndl = dot3(n, Ls), wrap = L[10];
        float vis = 0.0f;
        int cand[MAXC];
        if (pass < aa_passes) {           // the sun term has no noise: once per traced G-buffer
            int nc = ray_cands(nvol, wbox, vft, ps, Ls, L[11], 1e-3f, cand);
            if (ndl > -wrap) vis = march_vis(cand, nc, vft, vit, grid, ttab, sdf, ps, Ls, L[11], L[12], lsh, L[21] > 0.0f ? (int)L[21] : 256);
            gvis[pi] = vis;
        } else vis = gvis[pi];
        float wd = fmaxf((ndl + wrap) / (1.0f + wrap), 0.0f);
        int nsky = (int)L[14];
        unsigned int hs = hash3(x, y, pass * 7907u + 3u);
        float3 t1 = fabsf(n.x) > 0.5f ? mk(0, 1, 0) : mk(1, 0, 0);
        float3 bu = nrm(mk(n.y * t1.z - n.z * t1.y, n.z * t1.x - n.x * t1.z, n.x * t1.y - n.y * t1.x));
        float3 bv = mk(n.y * bu.z - n.z * bu.y, n.z * bu.x - n.x * bu.z, n.x * bu.y - n.y * bu.x);
        int sq = max(1, (int)floorf(sqrtf((float)nsky)));
        float open = 0.0f;
        int cs = -1;                      // n3: this point's sky cell (-1: none, today's per-pixel estimate)
        float c_open = 0.0f, c_rays = 0.0f;
        float ab_prev0 = aob[pi * 2], ab_prev1 = aob[pi * 2 + 1];   // n3d: this pixel's sum before this pass
        if (g_sky_mask != 0u) {
            float3 cpos0 = mk(L[17], L[18], L[19]);
            float3 dc0 = sub(p, cpos0);
            unsigned long long key = sky_key(p, n, vol, sqrtf(dot3(dc0, dc0)));
            if (key != 0ull) cs = sky_slot(key);
            if (cs >= 0) {
                c_rays = atomicAdd(&g_sky_v[2 * cs + 1], 0.0f);
                c_open = atomicAdd(&g_sky_v[2 * cs], 0.0f);
                // the cell holds enough rays: none traced (n3d, g_sky_par[7] > 0: only on a camera-move pass, the first after
                // a restart; a still view's later passes trace their own rays as before, so its converged image is per pixel)
                if (c_rays >= g_sky_par[1] && (g_sky_par[7] <= 0.0f || pass == 0)) nsky = 0;
            }
        }
        int nsc = 0;                      // sky rays reach L[13]: the volumes whose box is within that of the point
        for (int v = 0; v < nvol && nsc < MAXC; v++) {
            if (vft[v * VF + VF_VIS] < 0.5f) continue;
            const float* B = wbox + 6 * v;
            float dx = fmaxf(fmaxf(B[0] - ps.x, ps.x - B[3]), 0.0f), dy = fmaxf(fmaxf(B[1] - ps.y, ps.y - B[4]), 0.0f),
                  dz = fmaxf(fmaxf(B[2] - ps.z, ps.z - B[5]), 0.0f);
            if (dx * dx + dy * dy + dz * dz <= (L[13] + 1e-3f) * (L[13] + 1e-3f)) cand[nsc++] = v;
        }
        for (int k = 0; k < nsky; k++) {
            unsigned int hk = hash3(hs, k * 7919u + 1u, pass + 1u);
            float u1 = ((k % sq) + u01(hk)) / sq, u2 = ((k / sq) % sq + u01(hash3(hk, 5u, 11u))) / sq;
            float r = sqrtf(u1), ph = 6.2831853f * u2;
            float3 dir = nrm(add(add(mul(bu, r * cosf(ph)), mul(bv, r * sinf(ph))), mul(n, sqrtf(fmaxf(1.0f - u1, 0.0f)))));
            open += march_vis(cand, nsc, vft, vit, grid, ttab, sdf, ps, dir, L[13], 0.0f, lsh, L[20] > 0.0f ? (int)L[20] : 64) > 0.0f ? 1.0f : 0.0f;   // r38: runtime caps (a literal cap
            // compiled 2.8x slower: 109 -> 39 ms for 2 rays over a 5.8 Mpx close-up, same results)
        }
        // sky: its colour factor (albedo / pi x SH irradiance) and the rays' visibility kept apart, so resolve can
        // filter the visibility (depth / normal guided) while it converges
        float3 E0 = sh_irr(SH, n);
        float* sc = skyc + pi * 4;
        float* ab = aob + pi * 2;
        float alb[3] = {vf[VF_ALB], vf[VF_ALB + 1], vf[VF_ALB + 2]};
        float mspec = 0.0f, msky = 1.0f;
        bool mon = mat_shade(vol, vf, p, n, L, ndl, &wd, alb, &mspec, &msky);   // fs-materials 11-14: false, nothing changed, without a material
        float fa[3] = {alb[0] * E0.x * msky, alb[1] * E0.y * msky, alb[2] * E0.z * msky};
        if (cs >= 0) {
            // n3: the rays traced go to the cell; the pixel shows the cell's total (resolve filters it as a pixel's own)
            if (nsky > 0 && c_rays < 16.0f * g_sky_par[1]) { atomicAdd(&g_sky_v[2 * cs], open); atomicAdd(&g_sky_v[2 * cs + 1], (float)nsky);
                            c_open += open; c_rays += (float)nsky; }
            if (pass == 0) { sc[0] = fa[0] * L[8] / 3.14159265f; sc[1] = fa[1] * L[8] / 3.14159265f; sc[2] = fa[2] * L[8] / 3.14159265f; sc[3] = 1.0f; }
            else { sc[0] += fa[0] * L[8] / 3.14159265f; sc[1] += fa[1] * L[8] / 3.14159265f; sc[2] += fa[2] * L[8] / 3.14159265f; sc[3] += 1.0f; }
            ab[0] = c_open; ab[1] = c_rays;
            if (g_sky_par[7] > 0.0f) {
                // n3d: the cell's mean is a PRIOR of weight g_sky_par[7] x its fullness, set on the restart's pass with the
                // pixel's own rays of that pass; later passes add the pixel's own rays as today (resolve's filter by count)
                float kw = c_rays > 0.0f ? g_sky_par[7] * fminf(c_rays / g_sky_par[1], 1.0f) : 0.0f;
                float pm = c_rays > 0.0f ? c_open / c_rays : 0.0f;
                if (pass == 0) { ab[0] = pm * kw + open; ab[1] = kw + (float)nsky; }
                else { ab[0] = ab_prev0 + open; ab[1] = ab_prev1 + (float)nsky; }
            }
        } else
        if (pass == 0) { sc[0] = fa[0] * L[8] / 3.14159265f; sc[1] = fa[1] * L[8] / 3.14159265f; sc[2] = fa[2] * L[8] / 3.14159265f; sc[3] = 1.0f;
                         ab[0] = open; ab[1] = (float)nsky; }
        else { sc[0] += fa[0] * L[8] / 3.14159265f; sc[1] += fa[1] * L[8] / 3.14159265f; sc[2] += fa[2] * L[8] / 3.14159265f; sc[3] += 1.0f;
               ab[0] += open; ab[1] += (float)nsky; }
        float3 Es = mk(0, 0, 0);
        float3 cpos = mk(L[17], L[18], L[19]);
        float3 vd = nrm(sub(cpos, p));
        float3 hv = nrm(add(Ls, vd));
        float nh = fmaxf(dot3(n, hv), 0.0f), spec = mon ? mspec * vis : L[15] * powf(nh, L[16]) * (ndl > 0.0f ? 1.0f : 0.0f) * vis;
        float sun[3] = {L[3], L[4], L[5]}, es[3] = {Es.x, Es.y, Es.z}, cc[3];
        float hd = L[22] * fmaxf(dot3(n, vd), 0.0f);       // sculpt headlight (11-08 night: deep recesses were black)
        if (L[22] <= 0.0f) hd = 0.0f;                         // review light: the sum below is as before, bit for bit
        for (int c = 0; c < 3; c++) cc[c] = alb[c] / 3.14159265f * (sun[c] * vis * wd + es[c] + hd) + spec * sun[c];
        col = mk(cc[0], cc[1], cc[2]);
    }
    col = mul(col, L[8]);
    float* a = accum + pi * 4;
    if (pass == 0) { a[0] = col.x; a[1] = col.y; a[2] = col.z; a[3] = 1.0f; }
    else { a[0] += col.x; a[1] += col.y; a[2] += col.z; a[3] += 1.0f; }
}

// accumulation -> 8-bit BGRA (Qt Format_RGB32 order), filmic tone map (ACES fit) + sRGB
__device__ __forceinline__ float aces(float x) { return fminf(fmaxf((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f), 1.0f); }
__device__ __forceinline__ float srgb(float x) { return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f; }
extern "C" __global__ void resolve(const float* accum, unsigned int* out, int W, int H, int x0, int y0, int rw, int rh,
                                   const float* skyc, const float* aob, const float* depth, const float4* gn, float cav) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= rw || py >= rh) return;
    int x = x0 + px, y = y0 + py;
    if (x >= W || y >= H) return;
    size_t pi = (size_t)y * W + x;
    const float* a = accum + pi * 4;
    float inv = 1.0f / fmaxf(a[3], 1.0f);
    float c0 = a[0] * inv, c1 = a[1] * inv, c2 = a[2] * inv;
    const float* sc = skyc + pi * 4;
    float rays = aob[pi * 2 + 1];
    if (rays > 0.0f) {
        // sky visibility: joint bilateral over the neighbours (same surface: depth within 2 % and normals within ~25
        // deg), radius 2 px until 8 rays a pixel, 1 px until 32, then the pixel's own estimate
        int R = rays < 8.0f ? 2 : (rays < 32.0f ? 1 : 0);
        float d0 = depth[pi];
        float4 n0 = gn[pi];
        float so = 0.0f, sn = 0.0f;
        for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) {
            int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
            size_t q = (size_t)yy * W + xx;
            float rq = aob[q * 2 + 1];
            if (rq <= 0.0f) continue;
            float4 nq = gn[q];
            if (fabsf(depth[q] - d0) > 0.02f * d0 || n0.x * nq.x + n0.y * nq.y + n0.z * nq.z < 0.9f) continue;
            so += aob[q * 2]; sn += rq;
        }
        float ao = sn > 0.0f ? so / sn : aob[pi * 2] / rays;
        if (cav != 1.0f) ao = 1.0f - cav * (1.0f - ao);   // cavity strength: 1 = the sky's occlusion as is (review)
        float k = ao / fmaxf(sc[3], 1.0f);
        c0 += sc[0] * k; c1 += sc[1] * k; c2 += sc[2] * k;
    }
    unsigned int r = (unsigned int)(srgb(aces(c0)) * 255.0f + 0.5f), g = (unsigned int)(srgb(aces(c1)) * 255.0f + 0.5f),
                 b = (unsigned int)(srgb(aces(c2)) * 255.0f + 0.5f);
    out[pi] = 0xff000000u | (r << 16) | (g << 8) | b;
}

// write bricks into the pool: rows[i] <- vals[i] (729 fp16 each); used for incremental uploads
extern "C" __global__ void scatter_bricks(__half* sdf, const int* rows, const __half* vals, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * 729) return;
    size_t b = i / 729, k = i % 729;
    sdf[(size_t)rows[b] * 729 + k] = vals[i];
}

// per-pixel progressive restart: pass count -> 0 for every pixel of the rect (nbox == 0) or for the pixels the boxes
// can change (gpu.scene.affected: in the trace rect q, within the sky reach of a box, or a sun ray through one)
extern "C" __global__ void restart(int W, int H, int x0, int y0, int rw, int rh, const float4* gp, const float4* gn,
                                   const float* L, int nbox, const float* boxes, int qx0, int qy0, int qx1, int qy1, int* pcount) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= rw || py >= rh) return;
    int x = x0 + px, y = y0 + py;
    if (x >= W || y >= H) return;
    size_t pi = (size_t)y * W + x;
    if (nbox == 0 || affected(x, y, gp[pi], gn[pi], L, nbox, boxes, qx0, qy0, qx1, qy1)) pcount[pi] = 0;
}

// picks (r12): n rays through the same field the viewport draws, finest level (lod bias 0), exact hit refinement.
// out: t (world, -1 = miss), volume index, normal xyz
extern "C" __global__ void pick_rays(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                     const float* orig, const float* dirs, int n, float* t_out, int* v_out, float* n_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 o = mk(orig[3 * i], orig[3 * i + 1], orig[3 * i + 2]), d = nrm(mk(dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]));
    Hit hh = trace_scene(nvol, vft, vit, grid, ttab, sdf, o, d, 100.0f, 0.0f, 0.0f, 1024);
    t_out[i] = hh.vol < 0 ? -1.0f : hh.t;
    v_out[i] = hh.vol;
    if (hh.vol >= 0) {
        float3 nn = vol_normal(vft + hh.vol * VF, vit + hh.vol * VI, grid, ttab, sdf, add(o, mul(d, hh.t)), hh.lev);
        if (dot3(nn, d) > 0.0f) nn = mul(nn, -1.0f);
        n_out[3 * i] = nn.x; n_out[3 * i + 1] = nn.y; n_out[3 * i + 2] = nn.z;
    }
}

// r39: selection outline alpha, ui/viewport._soft_edge's formula at full resolution: b = box(box(sel, 1), 1) (a 5x5 tent,
// weights (1 2 3 2 1)^2 / 81), a = clip(1.5 (1 - |2b - 1|) - 0.3, 0, 1), as a byte. sel = the pixel's object id is in ids.
extern "C" __global__ void sel_alpha(const short* obj, int W, int H, const short* ids, int nids, unsigned char* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    // r43: two edge-clamped 3x3 box passes, as the CPU's _box(_box(sel, 1), 1) (one clamped 5x5 tent differed at the
    // frame's border: K5 max 0.076 on row 0)
    float b = 0.0f;
    for (int ey = -1; ey <= 1; ey++) for (int ex = -1; ex <= 1; ex++) {
        int cx = min(max(x + ex, 0), W - 1), cy = min(max(y + ey, 0), H - 1);
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            int xx = min(max(cx + dx, 0), W - 1), yy = min(max(cy + dy, 0), H - 1);
            short o = obj[(size_t)yy * W + xx];
            bool s = false;
            for (int k = 0; k < nids; k++) s = s || (o == ids[k]);
            if (s) b += 1.0f;
        }
    }
    b /= 81.0f;
    float a = fminf(fmaxf(1.5f * (1.0f - fabsf(2.0f * b - 1.0f)) - 0.3f, 0.0f), 1.0f);
    out[(size_t)y * W + x] = (unsigned char)(a * 255.0f + 0.5f);
}

// fs-brush-4 11-13 (n3): an edit's boxes (world lo / hi, nb of them) + pad: the sky cells whose cell overlaps one lose their
// rays (the key stays; the next frames' pixels re-converge them)
extern "C" __global__ void sky_invalidate(int nbox, const float* boxes, float pad) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i <= g_sky_mask; i += gridDim.x * blockDim.x) {
        unsigned long long k = g_sky_k[i];
        if (k == 0ull) continue;
        int l = (int)((k >> 58) & 31ull);
        float cl = g_sky_par[0] * (float)(1u << l);
        int vo = (int)((k >> 49) & 63ull);
        float cx = ((float)(int)((k >> 32) & 0xFFFFull) - 32768.0f + 0.5f) * cl + g_sky_org[3 * vo];
        float cy = ((float)(int)((k >> 16) & 0xFFFFull) - 32768.0f + 0.5f) * cl + g_sky_org[3 * vo + 1];
        float cz = ((float)(int)(k & 0xFFFFull) - 32768.0f + 0.5f) * cl + g_sky_org[3 * vo + 2];
        float g = pad + 0.8661f * cl;
        for (int b = 0; b < nbox; b++) {
            const float* B = boxes + 6 * b;
            if (cx >= B[0] - g && cx <= B[3] + g && cy >= B[1] - g && cy <= B[4] + g && cz >= B[2] - g && cz <= B[5] + g) {
                g_sky_v[2 * i] = 0.0f; g_sky_v[2 * i + 1] = 0.0f;
                break;
            }
        }
    }
}
