// The hair fibre's scattering and hit code, pasted by engine/hair/fibre_layer.py before fibre_layer.cuh.
// The single fibre follows Chiang et al. 2016 (the near-field lobes R, TT, TRT and the rest at the true offset h; the
// far field through Table A, the near field integrated over h), with pbrt-v3's importance sampling of those lobes for the
// full-path functions. Also here: float3 ops; the strand curve as centripetal Catmull-Rom spans (Seg4, cr_eval, cr_deriv);
// the oval section and its cone-footprint hit (SecF, seg_hit_q); the per-pixel hit list (HitList8, hits_insert);
// cuticle_dalpha; the constants HT, HSUN, HSUNRGB, HSKY; hhash; and the older lock-medium and bake functions.
// Two corrections are built in: the hit offset h takes the sign the lobes measure phi from (it had shaded every fibre as
// its own mirror image), and the bounce light is shaped over h with the fibre's own near / far ratio for the light's mean
// direction, which keeps its h-mean (Table A is the h-integral of the near field).
#include <cuda_fp16.h>
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ __forceinline__ float3 operator*(float s, float3 a) { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ __forceinline__ float3 operator/(float3 a, float s) { return make_float3(a.x/s, a.y/s, a.z/s); }
__device__ __forceinline__ void operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
__device__ __forceinline__ float hdot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ float3 hcross(float3 a, float3 b) { return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ __forceinline__ float hlen(float3 a) { return sqrtf(hdot(a, a)); }
__device__ __forceinline__ float3 hnorm(float3 a) { return a * (1.f / fmaxf(hlen(a), 1e-30f)); }
#define HK 8                        // fibre hits kept per pixel
#define HNSUB 8                     // chords per curve segment for the bracket, then Newton on the curve
#define HPTS 64
#define HSEG 63
#define TA_TD 64                    // Table A: |theta_o| [0, pi/2] (64 nodes incl. ends), phi [-pi, pi) 128 periodic, sigma 32
#define TA_PH 128
#define TA_SG 32
#define TB_MAXT 24                  // Table B: theta nodes (non-uniform, <= 24), dphi 16 periodic cell centres
#define TB_DP 16
#define TB_K 5
#define TB_D 6
#define TB_S 32
#define SG_LO 0.005f
#define SG_HI 30.0f
#define HPI 3.14159265f

struct HairType { float beta_m, beta_n, eta, alpha, ell, tau, s_len, step, r_eff, ms_const, rr, pad; };
__constant__ HairType HT;
__constant__ const float* HP;              // (S, 64, 3) m
__constant__ const float* HR;              // (S, 64) m
__constant__ const __half* HSIG;           // (S, 64, 3) per unit radius
__constant__ const unsigned* HSEED;        // (S,)
__constant__ const __half* HELL;           // (S,)
__constant__ const int* HLOCK;             // (S,)
__constant__ const unsigned char* LOCKLEV; // (locks,) 0 L0, 1 L1, 2 L2; null in FULL (every fibre is a curve)
__constant__ const __half* L2V;            // (B, 512, 7)
__constant__ const __half* L2BND;          // (B, 7)
__constant__ const int* L2IDX;             // (gx, gy, gz) dense brick index of the 8 mm bricks, -1 empty
__constant__ const unsigned* L2OFF;        // (B + 1)
__constant__ const unsigned* L2SEG;        // s * 63 + k
__constant__ float3 L2ORG; __constant__ int3 L2DIM;
__constant__ const __half* L3T;            // (C,) sun transmittance (fv4_hair_l3.py)
__constant__ const __half* L3SH;           // (C, 9, 3) incoming radiance, world real SH order 2 (sky_rgb included)
__constant__ const int* L3IDX; __constant__ int3 L3DIM;   // 4 mm cells, origin L2ORG
__constant__ const int* L3IJK;             // (C, 3)
__constant__ float* L3N;                   // (C, 3) bake_hair: the lock surface normal (out of the hair)
__constant__ float* L3DEP;                 // (C,)  bake_hair: slab depth through the cell along N, perpendicular mfp
__constant__ int* L3SURF;                  // (C,)  bake_hair: the cell just outside the lock along N (-1: none)
__constant__ const float* TA;              // (64, 128, 32, 4) f32
__constant__ const __half* TB;             // (nt, nt, 16, 5, 6, 32) orders >= 2
__constant__ const __half* TBSH;           // (16, nt, 5, 6, 32) all orders (or order 1 with --hair-ms off)
__constant__ float TBTH[TB_MAXT]; __constant__ int TBNT;   // theta nodes (rad, ascending)
__constant__ float3 HSUN; __constant__ float3 HSUNRGB; __constant__ float3 HSKY; __constant__ float3 HHC;  // head centre
__constant__ int HMS;                      // --hair-ms on|off
__constant__ const float* BRL2;            // (B,) per brick, this frame: the share of its segments in L2 locks
__constant__ const __half* L3J;            // (C, 9, 3) light fix (11-04, fv4_hair_l3ms.py): path-traced in-scattered light J(wo)
__constant__ int HROOM;                    // 1: hair_bounce = L3J (sky all orders + sun orders >= 2); 0: Table B sun + R_sh room

struct HitList8 { int n; float t[HK]; float a[HK]; float h[HK]; float u[HK]; int sk[HK]; float wpx[HK]; };

__device__ __forceinline__ float hhash(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return (x & 0xffffff) * (1.f / 16777216.f); }

// ---------------------------------------------------------------- the curve: centripetal Catmull-Rom (fv4_hair_field.py)
__device__ __forceinline__ float3 hp(int s, int i) {
    if (i < 0) { float3 a = make_float3(HP[(s*HPTS)*3], HP[(s*HPTS)*3+1], HP[(s*HPTS)*3+2]);
                 float3 b = make_float3(HP[(s*HPTS+1)*3], HP[(s*HPTS+1)*3+1], HP[(s*HPTS+1)*3+2]);
                 return make_float3(2.f*a.x-b.x, 2.f*a.y-b.y, 2.f*a.z-b.z); }
    if (i > HPTS-1) { float3 a = make_float3(HP[(s*HPTS+HPTS-1)*3], HP[(s*HPTS+HPTS-1)*3+1], HP[(s*HPTS+HPTS-1)*3+2]);
                      float3 b = make_float3(HP[(s*HPTS+HPTS-2)*3], HP[(s*HPTS+HPTS-2)*3+1], HP[(s*HPTS+HPTS-2)*3+2]);
                      return make_float3(2.f*a.x-b.x, 2.f*a.y-b.y, 2.f*a.z-b.z); }
    return make_float3(HP[(s*HPTS+i)*3], HP[(s*HPTS+i)*3+1], HP[(s*HPTS+i)*3+2]);
}
__device__ __forceinline__ float3 f3l(float3 a, float3 b, float ta, float tb, float t) {
    float wa = (tb - t) / (tb - ta), wb = (t - ta) / (tb - ta);
    return make_float3(wa*a.x + wb*b.x, wa*a.y + wb*b.y, wa*a.z + wb*b.z);
}
struct Seg4 { float3 p0, p1, p2, p3; float t1, t2, t3; };
__device__ __forceinline__ Seg4 seg_load(int s, int k) {
    Seg4 q; q.p0 = hp(s, k-1); q.p1 = hp(s, k); q.p2 = hp(s, k+1); q.p3 = hp(s, k+2);
    float3 d;
    d = q.p1 - q.p0; float d01 = fmaxf(sqrtf(sqrtf(hdot(d, d))), 1e-6f);
    d = q.p2 - q.p1; float d12 = fmaxf(sqrtf(sqrtf(hdot(d, d))), 1e-6f);
    d = q.p3 - q.p2; float d23 = fmaxf(sqrtf(sqrtf(hdot(d, d))), 1e-6f);
    q.t1 = d01; q.t2 = d01 + d12; q.t3 = q.t2 + d23; return q;
}
__device__ __forceinline__ float3 cr_eval(const Seg4& q, float u) {
    float t = q.t1 + u * (q.t2 - q.t1);
    float3 a1 = f3l(q.p0, q.p1, 0.f, q.t1, t), a2 = f3l(q.p1, q.p2, q.t1, q.t2, t), a3 = f3l(q.p2, q.p3, q.t2, q.t3, t);
    float3 b1 = f3l(a1, a2, 0.f, q.t2, t), b2 = f3l(a2, a3, q.t1, q.t3, t);
    return f3l(b1, b2, q.t1, q.t2, t);
}
__device__ __forceinline__ float3 cr_deriv(const Seg4& q, float u) {
    float e = 1e-3f; float3 a = cr_eval(q, fmaxf(u - e, 0.f)), b = cr_eval(q, fminf(u + e, 1.f));
    float inv = 1.f / (fminf(u + e, 1.f) - fmaxf(u - e, 0.f));
    return make_float3((b.x-a.x)*inv, (b.y-a.y)*inv, (b.z-a.z)*inv);
}
__device__ __forceinline__ float strand_arc(int s, int k, float u) {
    float arc = 0.f;
    for (int i = 0; i < k; ++i) arc += hlen(hp(s, i+1) - hp(s, i));
    return arc + u * hlen(hp(s, k+1) - hp(s, k));
}

// ---------------------------------------------------------------- FV8: the section of any fibre span (stored or generated)
struct SecF { float3 g; float ell, psi0, psi1, r0, r1; };   // g: reference vector (world); psi lerps over the span
__device__ __forceinline__ float oval_hw_q(const SecF& f, float u, float3 T, float3 D, float r) {
    float sa = r * sqrtf(f.ell), sb = r / sqrtf(f.ell);
    float3 n = hnorm(f.g - T * hdot(f.g, T)); float3 b = hcross(T, n);
    float psi = f.psi0 + u * (f.psi1 - f.psi0);
    float cp = cosf(psi), sp = sinf(psi);
    float3 e1 = n * cp + b * sp, e2 = b * cp - n * sp;
    float3 w = hcross(T, D); float lw = hlen(w); if (lw < 1e-6f) return r; w = w / lw;
    float c1 = hdot(w, e1), c2 = hdot(w, e2);
    return sqrtf(sa*sa*c1*c1 + sb*sb*c2*c2);
}
__device__ __forceinline__ SecF sec_stored(int s, int k) {   // exactly FV4's oval_halfwidth inputs for strand s, span k
    SecF f; float3 t0 = hnorm(hp(s, 1) - hp(s, 0));
    float3 g = hcross(t0, make_float3(0.f, 0.f, 1.f));
    if (hdot(g, g) < 1e-6f) g = hcross(t0, make_float3(1.f, 0.f, 0.f));
    f.g = g; f.ell = __half2float(HELL[s]);
    float ph = 6.283185307f * (float)HSEED[s] * 2.3283064e-10f;
    f.psi0 = ph + HT.tau * strand_arc(s, k, 0.f); f.psi1 = ph + HT.tau * strand_arc(s, k, 1.f);
    f.r0 = HR[s*HPTS+k]; f.r1 = HR[s*HPTS+k+1]; return f;
}
// ---------------------------------------------------------------- the oval section (H1, unchanged)
__device__ __forceinline__ float oval_halfwidth(int s, int k, float u, float3 T, float3 D, float r) {
    float ell = __half2float(HELL[s]);
    float sa = r * sqrtf(ell), sb = r / sqrtf(ell);
    float3 t0 = hnorm(hp(s, 1) - hp(s, 0));
    float3 g = hcross(t0, make_float3(0.f, 0.f, 1.f));
    if (hdot(g, g) < 1e-6f) g = hcross(t0, make_float3(1.f, 0.f, 0.f));
    float3 n = hnorm(g - T * hdot(g, T)); float3 b = hcross(T, n);
    float psi = 6.283185307f * (float)HSEED[s] * 2.3283064e-10f + HT.tau * strand_arc(s, k, u);
    float cp = cosf(psi), sp = sinf(psi);
    float3 e1 = n * cp + b * sp, e2 = b * cp - n * sp;
    float3 w = hcross(T, D); float lw = hlen(w); if (lw < 1e-6f) return r; w = w / lw;
    float c1 = hdot(w, e1), c2 = hdot(w, e2);
    return sqrtf(sa*sa*c1*c1 + sb*sb*c2*c2);
}

// ---------------------------------------------------------------- one ray (a cone of half-angle pix_half) and one segment
// pix_half = 0: the exact ray (FULL); coverage is then 1 inside the section and 0 outside.
__device__ bool seg_hit_q(const Seg4& q, const SecF& sf, float3 O, float3 D, float pix_half, float tmin, float tmax,
                          float& th, float& alpha, float& hh, float& uh, float& wpx) {
    float best = 1e30f, bu = 0.f;
    float3 a = cr_eval(q, 0.f);
    for (int i = 1; i <= HNSUB; ++i) {
        float3 b = cr_eval(q, (float)i / HNSUB);
        float3 ab = b - a, ao = a - O;
        float A_ = hdot(ab, ab), B_ = hdot(ab, D), C_ = hdot(ab, ao), E_ = hdot(D, ao);
        float den = A_ - B_ * B_;
        float v = den > 1e-20f ? (B_ * E_ - C_) / den : 0.f; v = fminf(fmaxf(v, 0.f), 1.f);
        float3 c = a + ab * v; float tt = hdot(c - O, D); float3 x = O + D * tt - c; float d2 = hdot(x, x);
        if (d2 < best) { best = d2; bu = ((float)(i - 1) + v) / HNSUB; }
        a = b;
    }
    float rmax = fmaxf(sf.r0, sf.r1) * 1.2f;
    float3 cb = cr_eval(q, bu); float tb = hdot(cb - O, D);
    if (sqrtf(best) > rmax + pix_half * fmaxf(tb, 0.f) + 1e-6f) return false;
    float u = bu;
    for (int it = 0; it < 3; ++it) {
        float3 C = cr_eval(q, u), Cu = cr_deriv(q, u);
        float3 rC = C - O; float3 perp = rC - D * hdot(rC, D);
        float3 Cup = Cu - D * hdot(Cu, D);
        float f = hdot(perp, Cu), fp = hdot(Cup, Cup);
        if (fp < 1e-20f) break;
        u = fminf(fmaxf(u - f / fp, 0.f), 1.f);
    }
    float3 C = cr_eval(q, u), T = hnorm(cr_deriv(q, u));
    float t = hdot(C - O, D);
    if (t <= tmin || t >= tmax) return false;
    float3 x = C - (O + D * t);
    float3 wdir = hcross(T, D); float lw = hlen(wdir); if (lw < 1e-6f) return false; wdir = wdir / lw;
    float d = -hdot(x, wdir);
    float r = sf.r0 * (1.f - u) + sf.r1 * u;
    float w = oval_hw_q(sf, u, T, D, r);
    float rp = fmaxf(pix_half * t, 1e-9f);
    float ov = fminf(w, d + rp) - fmaxf(-w, d - rp);
    if (ov <= 0.f) return false;
    // F1: d is the offset along wdir = -Yh; pbrt's h is the offset along +Yh (shade_fibre's phi frame)
    th = t; alpha = fminf(ov / (2.f * rp), 1.f); hh = fminf(fmaxf(-d / w, -1.f), 1.f); uh = u; wpx = w / rp;
    return true;
}
__device__ bool seg_hit(float3 O, float3 D, float pix_half, float tmin, float tmax, int s, int k,
                        float& th, float& alpha, float& hh, float& uh, float& wpx) {
    Seg4 q = seg_load(s, k); SecF sf = sec_stored(s, k);
    return seg_hit_q(q, sf, O, D, pix_half, tmin, tmax, th, alpha, hh, uh, wpx);
}

__device__ __forceinline__ void hits_insert(HitList8& L, int cap, float t, float a, float h, float u, int sk, float wpx) {
    for (int i = 0; i < L.n; ++i) if (L.sk[i] == sk) return;           // the same segment listed in two bricks
    if (L.n == cap && t >= L.t[cap-1]) return;
    int j = L.n < cap ? L.n : cap - 1;
    while (j > 0 && L.t[j-1] > t) { L.t[j] = L.t[j-1]; L.a[j] = L.a[j-1]; L.h[j] = L.h[j-1]; L.u[j] = L.u[j-1]; L.sk[j] = L.sk[j-1]; L.wpx[j] = L.wpx[j-1]; --j; }
    L.t[j] = t; L.a[j] = a; L.h[j] = h; L.u[j] = u; L.sk[j] = sk; L.wpx[j] = wpx;
    if (L.n < cap) ++L.n;
}

// ---------------------------------------------------------------- the brick walk (3-D DDA over the 8 mm bricks)
// mode 0: up to HK hits (rules); 1: the closest hit only (FULL); 2: any hit (FULL shadow rays, returns at the first).
// skip_s / skip_k: the strand and segment a FULL path just left (its neighbours k +- 1 are skipped too).
__device__ float hair_walk(float3 O, float3 D, float pix_half, float tmax, int mode, int skip_s, int skip_k, HitList8& out) {
    const float BE = 8e-3f; const int cap = mode == 0 ? HK : 1;
    out.n = 0;
    float3 inv = make_float3(1.f / (fabsf(D.x) > 1e-12f ? D.x : 1e-12f), 1.f / (fabsf(D.y) > 1e-12f ? D.y : 1e-12f), 1.f / (fabsf(D.z) > 1e-12f ? D.z : 1e-12f));
    float3 lo = (L2ORG - O) * inv, hi = (L2ORG + make_float3(L2DIM.x, L2DIM.y, L2DIM.z) * BE - O) * inv;
    float t0 = fmaxf(fmaxf(fminf(lo.x, hi.x), fminf(lo.y, hi.y)), fmaxf(fminf(lo.z, hi.z), 0.f));
    float t1 = fminf(fminf(fmaxf(lo.x, hi.x), fmaxf(lo.y, hi.y)), fminf(fmaxf(lo.z, hi.z), tmax));
    if (t0 >= t1) return tmax;
    float3 p = O + D * (t0 + 1e-6f);
    int ix = min(max((int)floorf((p.x - L2ORG.x) / BE), 0), L2DIM.x - 1);
    int iy = min(max((int)floorf((p.y - L2ORG.y) / BE), 0), L2DIM.y - 1);
    int iz = min(max((int)floorf((p.z - L2ORG.z) / BE), 0), L2DIM.z - 1);
    int sx = D.x > 0 ? 1 : -1, sy = D.y > 0 ? 1 : -1, sz = D.z > 0 ? 1 : -1;
    float tx = ((L2ORG.x + (ix + (sx > 0)) * BE) - O.x) * inv.x, ty = ((L2ORG.y + (iy + (sy > 0)) * BE) - O.y) * inv.y, tz = ((L2ORG.z + (iz + (sz > 0)) * BE) - O.z) * inv.z;
    float dx = BE * fabsf(inv.x), dy = BE * fabsf(inv.y), dz = BE * fabsf(inv.z);
    float tin = t0;
    for (int guard = 0; guard < 4096; ++guard) {
        float tout = fminf(fminf(tx, ty), fminf(tz, t1));
        int b = L2IDX[(ix * L2DIM.y + iy) * L2DIM.z + iz];
        if (b >= 0) {
            for (unsigned j = L2OFF[b]; j < L2OFF[b+1]; ++j) {
                unsigned sk = L2SEG[j]; int s = sk / HSEG, k = sk - s * HSEG;
                if (s == skip_s && abs(k - skip_k) <= 1) continue;
                if (LOCKLEV && LOCKLEV[HLOCK[s]] == 2) continue;
                float th, al, hh, uh, wpx;
                if (seg_hit(O, D, pix_half, fmaxf(tin - 1e-3f, 1e-7f), fminf(tout + 1e-3f, tmax), s, k, th, al, hh, uh, wpx)) {
                    if (mode == 2) { out.n = 1; out.t[0] = th; out.sk[0] = (int)sk; return th; }
                    hits_insert(out, cap, th, al, hh, uh, (int)sk, wpx);
                }
            }
            if (mode == 1 && out.n > 0 && out.t[0] <= tout) return tout;
            if (mode == 0) {
                float Tr = 1.f; for (int i = 0; i < out.n; ++i) Tr *= 1.f - out.a[i];
                if (out.n > 0 && Tr < 1e-3f && out.t[out.n-1] <= tout) return tout;
                if (out.n == HK && out.t[HK-1] <= tout) return tout;
            }
        }
        if (tout >= t1) break;
        tin = tout;
        if (tx <= ty && tx <= tz) { ix += sx; tx += dx; if (ix < 0 || ix >= L2DIM.x) break; }
        else if (ty <= tz) { iy += sy; ty += dy; if (iy < 0 || iy >= L2DIM.y) break; }
        else { iz += sz; tz += dz; if (iz < 0 || iz >= L2DIM.z) break; }
    }
    return t1;
}
__device__ __forceinline__ float hair_hit(float3 O, float3 D, float pix_half, float tmax, HitList8& out) {
    return hair_walk(O, D, pix_half, tmax, 0, -1, -9, out);
}

// ---------------------------------------------------------------- the single fibre (Chiang 2016, as ms_ref/fibre.py)
__device__ __forceinline__ float i0s(float x) { float v = 0.f, x2i = 1.f, f = 1.f, i4 = 1.f;
    for (int i = 0; i < 10; ++i) { if (i > 1) f *= i; v += x2i / (i4 * f * f); x2i *= x * x; i4 *= 4.f; } return v; }
__device__ __forceinline__ float logi0(float x) { return x > 12.f ? x + 0.5f * (-logf(6.283185307f) + logf(1.f / x) + 1.f / (8.f * x)) : logf(i0s(x)); }
__device__ __forceinline__ float Mp(float ci, float co, float si, float so, float v) {
    float a = ci * co / v, b = si * so / v;
    if (v <= 0.1f) return expf(logi0(a) - b - 1.f / v + 0.6931f + logf(1.f / (2.f * v)));
    return expf(-b) * i0s(a) / (sinhf(1.f / v) * 2.f * v);
}
__device__ __forceinline__ float frd(float ci, float eta) {
    ci = fminf(fmaxf(ci, -1.f), 1.f); float st = sqrtf(fmaxf(1.f - ci*ci, 0.f)) / eta; if (st >= 1.f) return 1.f;
    float ct = sqrtf(fmaxf(1.f - st*st, 0.f)); float rpa = (eta*ci - ct) / (eta*ci + ct), rpe = (ci - eta*ct) / (ci + eta*ct);
    return 0.5f * (rpa*rpa + rpe*rpe);
}
__device__ __forceinline__ float logis(float x, float s) { float e = expf(-fabsf(x) / s); return e / (s * (1.f + e) * (1.f + e)); }
__device__ __forceinline__ float lcdf(float x, float s) { return 1.f / (1.f + expf(-x / s)); }
__device__ __forceinline__ float Np(float phi, int p, float s, float gO, float gT) {
    float d = phi - (2.f * p * gT - 2.f * gO + p * HPI);
    d -= 6.283185307f * floorf((d + HPI) / 6.283185307f);
    return logis(d, s) / (lcdf(HPI, s) - lcdf(-HPI, s));
}
__device__ __forceinline__ float var_b(float bm) { float x = 0.726f*bm + 0.812f*bm*bm + 3.7f*powf(bm, 20.f); return x * x; }
__device__ __forceinline__ float scale_s(float bn) { return 0.626657069f * (0.265f*bn + 1.194f*bn*bn + 5.372f*powf(bn, 22.f)); }
__device__ __forceinline__ float strand_bm(int s) { return HT.beta_m * (1.f + HT.rr * (2.f * hhash((unsigned)s * 2654435761u + 7u) - 1.f)); }

// the attenuations A_p at h (rgb), shared by the near field, its sampling and its pdf
struct Ap4 { float3 a[4]; float gO, gT; };
__device__ __forceinline__ Ap4 hair_ap(float h, float sO, float3 sigma) {
    Ap4 r; float cO = sqrtf(fmaxf(1.f - sO*sO, 0.f)), eta = HT.eta;
    float sT = sO / eta, cT = sqrtf(fmaxf(1.f - sT*sT, 0.f));
    float etap = sqrtf(fmaxf(eta*eta - sO*sO, 0.f)) / fmaxf(cO, 1e-4f);
    float sGT = h / etap, cGT = sqrtf(fmaxf(1.f - sGT*sGT, 0.f));
    r.gT = asinf(fminf(fmaxf(sGT, -1.f), 1.f)); r.gO = asinf(fminf(fmaxf(h, -1.f), 1.f));
    float l = 2.f * cGT / fmaxf(cT, 1e-4f);
    float3 Tr = make_float3(expf(-sigma.x * l), expf(-sigma.y * l), expf(-sigma.z * l));
    float f = frd(cO * sqrtf(fmaxf(1.f - h*h, 0.f)), eta);
    r.a[0] = make_float3(f, f, f); r.a[1] = Tr * ((1.f - f) * (1.f - f)); r.a[2] = r.a[1] * Tr * f;
    r.a[3] = make_float3(r.a[2].x * f * Tr.x / fmaxf(1.f - Tr.x * f, 1e-4f), r.a[2].y * f * Tr.y / fmaxf(1.f - Tr.y * f, 1e-4f), r.a[2].z * f * Tr.z / fmaxf(1.f - Tr.z * f, 1e-4f));
    return r;
}
// the tilted theta_o of lobe p (p = 3: untilted), pbrt-v3's sin2kAlpha table
__device__ __forceinline__ void lobe_tilt(int p, float sO, float cO, float al, float& so, float& co) {
    if (p == 3) { so = sO; co = cO; return; }
    float m = p == 0 ? 2.f : (p == 1 ? 1.f : 4.f), sg = p == 0 ? -1.f : 1.f;
    float sa = sinf(m * al), ca = cosf(m * al);
    so = sO * ca + sg * cO * sa; co = fabsf(cO * ca - sg * sO * sa);
}
// near field at the true h: f |cos theta_i| per unit irradiance. cuticle_dalpha adds to the scale tilt (macro only).
__device__ float3 hair_near(float h, float sI, float sO, float phi, float3 sigma, float cuticle_dalpha, float bm) {
    float cO = sqrtf(fmaxf(1.f - sO*sO, 0.f)), cI = sqrtf(fmaxf(1.f - sI*sI, 0.f));
    Ap4 A = hair_ap(h, sO, sigma);
    float vb = var_b(bm), s = scale_s(HT.beta_n), al = HT.alpha + cuticle_dalpha;
    const float vm[4] = {1.f, 0.25f, 4.f, 4.f};
    float3 r = make_float3(0.f, 0.f, 0.f);
    for (int p = 0; p < 4; ++p) {
        float so, co; lobe_tilt(p, sO, cO, al, so, co);
        float m = Mp(cI, co, sI, so, vm[p] * vb);
        float n = p < 3 ? Np(phi, p, s, A.gO, A.gT) : 1.f / 6.283185307f;
        r += A.a[p] * (m * n);
    }
    return r;
}
// pbrt-v3 HairBSDF::Sample_f / Pdf on the same lobes (FULL). u0..u3 uniform; returns false for a dead sample.
// Out: sI (sin theta_i), dphi (phi_i - phi_o), pdf over solid angle.
__device__ __forceinline__ float hair_pdf(float h, float sI, float sO, float phi, float3 sigma, float bm) {
    float cO = sqrtf(fmaxf(1.f - sO*sO, 0.f)), cI = sqrtf(fmaxf(1.f - sI*sI, 0.f));
    Ap4 A = hair_ap(h, sO, sigma);
    float w[4], ws = 0.f; for (int p = 0; p < 4; ++p) { w[p] = (A.a[p].x + A.a[p].y + A.a[p].z) / 3.f; ws += w[p]; }
    if (ws <= 0.f) return 0.f;
    float vb = var_b(bm), s = scale_s(HT.beta_n); const float vm[4] = {1.f, 0.25f, 4.f, 4.f};
    float pdf = 0.f;
    for (int p = 0; p < 4; ++p) {
        float so, co; lobe_tilt(p, sO, cO, HT.alpha, so, co);
        pdf += Mp(cI, co, sI, so, vm[p] * vb) * (w[p] / ws) * (p < 3 ? Np(phi, p, s, A.gO, A.gT) : 1.f / 6.283185307f);
    }
    return pdf;
}
__device__ bool hair_sample(float h, float sO, float3 sigma, float bm, float u0, float u1, float u2, float u3,
                            float& sI, float& dphi, float& pdf) {
    float cO = sqrtf(fmaxf(1.f - sO*sO, 0.f));
    Ap4 A = hair_ap(h, sO, sigma);
    float w[4], ws = 0.f; for (int p = 0; p < 4; ++p) { w[p] = (A.a[p].x + A.a[p].y + A.a[p].z) / 3.f; ws += w[p]; }
    if (ws <= 0.f) return false;
    int p = 0; float acc = 0.f;
    for (p = 0; p < 3; ++p) { float q = w[p] / ws; if (u0 < acc + q) { u0 = (u0 - acc) / q; break; } acc += q; }
    if (p == 3) u0 = (u0 - acc) / fmaxf(w[3] / ws, 1e-12f);
    u0 = fminf(fmaxf(u0, 0.f), 0.99999994f);
    float vb = var_b(bm), s = scale_s(HT.beta_n); const float vm[4] = {1.f, 0.25f, 4.f, 4.f}; float v = vm[p] * vb;
    float so, co; lobe_tilt(p, sO, cO, HT.alpha, so, co);
    float uu = fmaxf(u1, 1e-5f);
    float ct = 1.f + v * logf(uu + (1.f - uu) * expf(-2.f / v));
    float st = sqrtf(fmaxf(1.f - ct * ct, 0.f)), cp = cosf(6.283185307f * u2);
    sI = fminf(fmaxf(-ct * so + st * cp * co, -1.f), 1.f);
    if (p < 3) {
        float k = lcdf(HPI, s) - lcdf(-HPI, s);
        float x = -s * logf(1.f / (u3 * k + lcdf(-HPI, s)) - 1.f);
        x = fminf(fmaxf(x, -HPI), HPI);
        dphi = 2.f * p * A.gT - 2.f * A.gO + p * HPI + x;
    } else dphi = 6.283185307f * u3;
    pdf = hair_pdf(h, sI, sO, dphi, sigma, bm);
    return pdf > 0.f;
}

// the cuticle (macro only, H1): scale edges every s_len; the face rises HT.step over the exposed length
__device__ __forceinline__ float cuticle_dalpha(float arc, unsigned seed) {
    float x = arc / HT.s_len + (float)(seed & 0xffff) * (1.f / 65536.f); x -= floorf(x);
    float riser = 0.1e-6f / HT.s_len;
    if (x > 1.f - riser) return -1.4f - HT.alpha;
    return atanf(HT.step / HT.s_len) - HT.alpha;
}

// ---------------------------------------------------------------- Table A (H2's layout; f32 upload)
__device__ __forceinline__ void ta_sig(float sg, int& k, float& f) {
    float u = (logf(fminf(fmaxf(sg, SG_LO), SG_HI)) - logf(SG_LO)) / (logf(SG_HI) - logf(SG_LO)) * (TA_SG - 1);
    u = fminf(fmaxf(u, 0.f), TA_SG - 1.0001f); k = (int)u; f = u - k;
}
__device__ __forceinline__ float hgeo(float a, float b, float f) {
    return (a > 0.f && b > 0.f) ? expf((1.f - f) * logf(a) + f * logf(b)) : (1.f - f) * a + f * b;
}
__device__ __forceinline__ float ta_at(int it, int ip, int ks, float fs, int p) {
    long long i = (((long long)it * TA_PH + ip) * TA_SG + ks) * 4 + p;
    return hgeo(TA[i], TA[i + 4], fs);
}
__device__ void tableA4(float sO, float phi, float sigma, float out[4]) {
    float ut = fminf(asinf(fminf(fabsf(sO), 1.f)) / 1.5707963f * (TA_TD - 1), TA_TD - 1.0001f); int it = (int)ut; float ft = ut - it;
    float up = (phi + HPI) / 6.283185307f * TA_PH; up -= floorf(up / TA_PH) * TA_PH;
    int ip = (int)up; float fp = up - ip; ip %= TA_PH;
    int ks; float fs; ta_sig(sigma, ks, fs);
    for (int p = 0; p < 4; ++p) {
        float v = 0.f;
        for (int a = 0; a < 2; ++a) {
#ifdef HAIR_TA_CUBIC
            float c0 = ta_at(it + a, (ip + TA_PH - 1) % TA_PH, ks, fs, p), c1 = ta_at(it + a, ip, ks, fs, p);
            float c2 = ta_at(it + a, (ip + 1) % TA_PH, ks, fs, p), c3 = ta_at(it + a, (ip + 2) % TA_PH, ks, fs, p);
            float t = fp, t2 = t * t, t3 = t2 * t;
            float cv = 0.5f * ((2.f * c1) + (-c0 + c2) * t + (2.f * c0 - 5.f * c1 + 4.f * c2 - c3) * t2 + (-c0 + 3.f * c1 - 3.f * c2 + c3) * t3);
            v += (a ? ft : 1.f - ft) * fmaxf(cv, 0.f);
#else
            v += (a ? ft : 1.f - ft) * ((1.f - fp) * ta_at(it + a, ip, ks, fs, p) + fp * ta_at(it + a, (ip + 1) % TA_PH, ks, fs, p));
#endif
        }
        out[p] = v;
    }
}
// the far field (h-integrated; a fibre narrower than a pixel): f |cos theta_i| = sum_p M_p(theta_i, tilted theta_o) far_p
__device__ float3 hair_far(float sI, float sO, float phi, float3 sigma, float bm) {
    float cO = sqrtf(fmaxf(1.f - sO*sO, 0.f)), cI = sqrtf(fmaxf(1.f - sI*sI, 0.f));
    float vb = var_b(bm); const float vm[4] = {1.f, 0.25f, 4.f, 4.f};
    float fr[4], fg[4], fb[4];
    tableA4(sO, phi, sigma.x, fr); tableA4(sO, phi, sigma.y, fg); tableA4(sO, phi, sigma.z, fb);
    float3 o = make_float3(0.f, 0.f, 0.f);
    for (int p = 0; p < 4; ++p) {
        float so, co; lobe_tilt(p, sO, cO, HT.alpha, so, co);
        float m = Mp(cI, co, sI, so, vm[p] * vb);
        o.x += m * fr[p]; o.y += m * fg[p]; o.z += m * fb[p];
    }
    return o;
}

// ---------------------------------------------------------------- Table B (H2's frame and axes)
__device__ __forceinline__ void tb_nodes(float x, const float* nd, int n, int& i, float& f) {
    if (x <= nd[0]) { i = 0; f = 0.f; return; }
    if (x >= nd[n-1]) { i = n - 2; f = 1.f; return; }
    int k = 0; while (k < n - 2 && x > nd[k+1]) ++k;
    i = k; f = (x - nd[k]) / (nd[k+1] - nd[k]);
}
__device__ __forceinline__ void tb_dphi(float x, int& i0, int& i1, float& f) {
    float u = (x + HPI) / 6.283185307f * TB_DP - 0.5f; u -= floorf(u / TB_DP) * TB_DP;
    i0 = (int)u; f = u - i0; i0 %= TB_DP; i1 = (i0 + 1) % TB_DP;
}
__device__ __forceinline__ void tb_kappa(float kdeg, int& i, float& f) {   // L2's kappa (deg); > 45 toward iso (90)
    const float K[5] = {0.f, 10.f, 25.f, 45.f, 90.f}; tb_nodes(fminf(fmaxf(kdeg, 0.f), 90.f), K, 5, i, f);
}
__device__ __forceinline__ void tb_depth(float d, int& i, float& f) {
    const float Dn[6] = {1.f, 2.f, 3.f, 5.f, 10.f, 1e4f}; tb_nodes(fmaxf(d, 1.f), Dn, 6, i, f);
}
__device__ float tableB1(float ti, float to, float dphi, float kdeg, float dep, float sigma) {
    int a0, b0, c0, c1, kk, dd, ks; float fa, fb, fc, fk, fd, fs; const int NT = TBNT;
    tb_nodes(ti, TBTH, NT, a0, fa); tb_nodes(to, TBTH, NT, b0, fb); tb_dphi(dphi, c0, c1, fc);
    tb_kappa(kdeg, kk, fk); tb_depth(dep, dd, fd); ta_sig(sigma, ks, fs);
    float v = 0.f;
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c)
    for (int k = 0; k < 2; ++k) for (int d = 0; d < 2; ++d) {
        float w = (a ? fa : 1 - fa) * (b ? fb : 1 - fb) * (c ? fc : 1 - fc) * (k ? fk : 1 - fk) * (d ? fd : 1 - fd);
        if (w <= 0.f) continue;
        long long i = (((((long long)(a0 + a) * NT + (b0 + b)) * TB_DP + (c ? c1 : c0)) * TB_K + kk + k) * TB_D + dd + d) * TB_S + ks;
        v += w * hgeo(__half2float(TB[i]), __half2float(TB[i + 1]), fs);
    }
    return v;
}
// the room's response in the lock frame: sum_q c_q (lock-frame SH, 9 used of 16) R_sh[q]
__device__ float3 tableB_room(const float3* c9, float to, float kdeg, float dep, float3 sigma) {
    int b0, kk, dd; float fb, fk, fd; const int NT = TBNT;
    tb_nodes(to, TBTH, NT, b0, fb); tb_kappa(kdeg, kk, fk); tb_depth(dep, dd, fd);
    float sg[3] = {sigma.x, sigma.y, sigma.z}, o[3] = {0.f, 0.f, 0.f};
    for (int ch = 0; ch < 3; ++ch) {
        int ks; float fs; ta_sig(sg[ch], ks, fs);
        for (int q = 0; q < 9; ++q) {
            float v = 0.f;
            for (int b = 0; b < 2; ++b) for (int k = 0; k < 2; ++k) for (int d = 0; d < 2; ++d) {
                float w = (b ? fb : 1 - fb) * (k ? fk : 1 - fk) * (d ? fd : 1 - fd);
                if (w <= 0.f) continue;
                long long i = ((((long long)q * NT + b0 + b) * TB_K + kk + k) * TB_D + dd + d) * TB_S + ks;
                v += w * ((1.f - fs) * __half2float(TBSH[i]) + fs * __half2float(TBSH[i + 1]));   // signed: linear in sigma
            }
            float c = ch == 0 ? c9[q].x : (ch == 1 ? c9[q].y : c9[q].z);
            o[ch] += c * v;
        }
    }
    return make_float3(fmaxf(o[0], 0.f), fmaxf(o[1], 0.f), fmaxf(o[2], 0.f));
}
// world SH (bands 0-2, L3's basis: Y00, Y1-1 (y), Y10 (z), Y11 (x), Y2-2 xy, Y2-1 yz, Y20, Y21 xz, Y22) rotated into the
// frame whose rows are ex, ey, ez: band 1 as the vector v (g(d') = (M v).d'), band 2 as the traceless quadratic form Q.
__device__ void sh_rotate9(const __half* w, float3 ex, float3 ey, float3 ez, float3* o) {
    const float k1 = 0.488603f, kxy = 1.092548f, k20 = 0.315392f, k22 = 0.546274f;
    for (int ch = 0; ch < 3; ++ch) {
        float c[9]; for (int q = 0; q < 9; ++q) c[q] = __half2float(w[q * 3 + ch]);
        float3 v = make_float3(c[3], c[1], c[2]);                       // (x, y, z) coefficients of band 1 (k1 shared)
        float lv[3] = {hdot(ex, v), hdot(ey, v), hdot(ez, v)};
        // Q (symmetric, traceless) with f2(d) = d^T Q d
        float Q[3][3];
        Q[0][0] = c[8] * k22 - c[6] * k20; Q[1][1] = -c[8] * k22 - c[6] * k20; Q[2][2] = 2.f * c[6] * k20;
        Q[0][1] = Q[1][0] = 0.5f * kxy * c[4]; Q[1][2] = Q[2][1] = 0.5f * kxy * c[5]; Q[0][2] = Q[2][0] = 0.5f * kxy * c[7];
        float Mr[3][3] = {{ex.x, ex.y, ex.z}, {ey.x, ey.y, ey.z}, {ez.x, ez.y, ez.z}};
        float MQ[3][3], R2[3][3];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { MQ[i][j] = 0.f; for (int k = 0; k < 3; ++k) MQ[i][j] += Mr[i][k] * Q[k][j]; }
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { R2[i][j] = 0.f; for (int k = 0; k < 3; ++k) R2[i][j] += MQ[i][k] * Mr[j][k]; }
        float l[9];
        l[0] = c[0]; l[1] = lv[1]; l[2] = lv[2]; l[3] = lv[0];
        l[4] = 2.f * R2[0][1] / kxy; l[5] = 2.f * R2[1][2] / kxy; l[7] = 2.f * R2[0][2] / kxy;
        l[6] = R2[2][2] / (2.f * k20); l[8] = (R2[0][0] - R2[1][1]) / (2.f * k22);
        for (int q = 0; q < 9; ++q) { if (ch == 0) o[q].x = l[q]; else if (ch == 1) o[q].y = l[q]; else o[q].z = l[q]; }
    }
}

// ---------------------------------------------------------------- lookups in L2 and L3
struct Vox { float phi, kdeg; float3 t, sig; int b; };
__device__ __forceinline__ float3 oct_dec(float x, float y) {
    float z = 1.f - fabsf(x) - fabsf(y);
    if (z < 0.f) { float ox = x; x = (1.f - fabsf(y)) * (ox >= 0.f ? 1.f : -1.f); y = (1.f - fabsf(ox)) * (y >= 0.f ? 1.f : -1.f); }
    return hnorm(make_float3(x, y, z));
}
__device__ __forceinline__ bool l2_vox(float3 p, Vox& v) {
    int ix = (int)floorf((p.x - L2ORG.x) * 1000.f), iy = (int)floorf((p.y - L2ORG.y) * 1000.f), iz = (int)floorf((p.z - L2ORG.z) * 1000.f);
    if (ix < 0 || iy < 0 || iz < 0) return false;
    int bx = ix >> 3, by = iy >> 3, bz = iz >> 3;
    if (bx >= L2DIM.x || by >= L2DIM.y || bz >= L2DIM.z) return false;
    int b = L2IDX[(bx * L2DIM.y + by) * L2DIM.z + bz]; if (b < 0) return false;
    const __half* q = L2V + ((long long)b * 512 + (((ix & 7) * 8 + (iy & 7)) * 8 + (iz & 7))) * 7;
    v.phi = __half2float(q[0]); if (!(v.phi > 0.f)) return false;
    v.t = oct_dec(__half2float(q[1]), __half2float(q[2])); v.kdeg = __half2float(q[3]);
    v.sig = make_float3(__half2float(q[4]), __half2float(q[5]), __half2float(q[6])); v.b = b;
    return true;
}
__device__ __forceinline__ float l2_phi(float3 p) { Vox v; return l2_vox(p, v) ? v.phi : 0.f; }
// extinction of the fibre medium along d (fv4_hair_l3.py's formula): 2 phi / (pi r_eff) sqrt(1 - E[(t.d)^2])
__device__ __forceinline__ float l2_sigt(const Vox& v, float3 d) {
    float md = hdot(v.t, d), lam = cosf(v.kdeg * 0.0174533f); lam *= lam;
    float e2 = lam * md * md + (1.f - lam) * 0.5f * (1.f - md * md);
    return 2.f * v.phi / (HPI * HT.r_eff) * sqrtf(fmaxf(0.f, 1.f - e2));
}
// the hair as an attenuating medium along a ray (the skin bake): exp(-integral sigma_t), 1 mm steps, jittered start
__device__ float hair_Tr(float3 O, float3 D, float tmax, float jit) {
    const float BE = 8e-3f;
    float3 inv = make_float3(1.f / (fabsf(D.x) > 1e-12f ? D.x : 1e-12f), 1.f / (fabsf(D.y) > 1e-12f ? D.y : 1e-12f), 1.f / (fabsf(D.z) > 1e-12f ? D.z : 1e-12f));
    float3 lo = (L2ORG - O) * inv, hi = (L2ORG + make_float3(L2DIM.x, L2DIM.y, L2DIM.z) * BE - O) * inv;
    float t0 = fmaxf(fmaxf(fminf(lo.x, hi.x), fminf(lo.y, hi.y)), fmaxf(fminf(lo.z, hi.z), 0.f));
    float t1 = fminf(fminf(fmaxf(lo.x, hi.x), fmaxf(lo.y, hi.y)), fminf(fmaxf(lo.z, hi.z), tmax));
    float tau = 0.f;
    for (float t = t0 + jit * 1e-3f; t < t1; t += 1e-3f) {
        Vox v; if (!l2_vox(O + D * t, v)) continue;
        tau += l2_sigt(v, D) * 1e-3f; if (tau > 9.2f) return 0.f;
    }
    return expf(-tau);
}
__device__ __forceinline__ int l3_cell(float3 p) {
    int x = (int)floorf((p.x - L2ORG.x) * 250.f), y = (int)floorf((p.y - L2ORG.y) * 250.f), z = (int)floorf((p.z - L2ORG.z) * 250.f);
    if (x < 0 || y < 0 || z < 0 || x >= L3DIM.x || y >= L3DIM.y || z >= L3DIM.z) return -1;
    return L3IDX[(x * L3DIM.y + y) * L3DIM.z + z];
}
// GPU run (11-04): L3 read trilinearly between the 8 nearest 4 mm cell centres (absent cells drop out and the weights
// are renormalised). A nearest-cell read switched the sun on and off in 4 mm blocks across the fibres (dry run f0).
__device__ __forceinline__ int l3_ijk(int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= L3DIM.x || y >= L3DIM.y || z >= L3DIM.z) return -1;
    return L3IDX[(x * L3DIM.y + y) * L3DIM.z + z];
}
__device__ __forceinline__ int l3_tri(float3 p, int* cid, float* w) {
    float gx = (p.x - L2ORG.x) * 250.f - 0.5f, gy = (p.y - L2ORG.y) * 250.f - 0.5f, gz = (p.z - L2ORG.z) * 250.f - 0.5f;
    int x0 = (int)floorf(gx), y0 = (int)floorf(gy), z0 = (int)floorf(gz);
    float fx = gx - x0, fy = gy - y0, fz = gz - z0, ws = 0.f; int n = 0;
    for (int q = 0; q < 8; ++q) {
        int dx = q & 1, dy = (q >> 1) & 1, dz = q >> 2;
        int c = l3_ijk(x0 + dx, y0 + dy, z0 + dz); if (c < 0) continue;
        float ww = (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy) * (dz ? fz : 1.f - fz);
        if (ww <= 0.f) continue;
        cid[n] = c; w[n] = ww; ws += ww; ++n;
    }
    for (int q = 0; q < n; ++q) w[q] /= ws;
    return n;
}
__device__ __forceinline__ float l3_Ts(float3 p) {
    int cid[8]; float w[8]; int n = l3_tri(p, cid, w);
    if (n == 0) return 1.f;
    float t = 0.f; for (int q = 0; q < n; ++q) t += w[q] * __half2float(L3T[cid[q]]);
    return t;
}

// ---------------------------------------------------------------- bake_hair: the lock frame per L3 cell (once per run)
// N: minus the gradient of phi smoothed over 7 taps at +-2 mm (the outward normal of the hair mass), else the radial
// direction from the head centre; depth: the slab's thickness through the cell along N in perpendicular mean free paths
// (sigma_max = 2 phi / (pi r_eff), H2's definition), marched both ways until 3 empty voxels in a row; surf: the first
// L3 cell beyond the slab's outer face (its T_sun and room SH light the lock's surface; -1 = open sky).
__device__ __forceinline__ float phi7(float3 p) {
    const float e = 1e-3f;
    return (l2_phi(p) + l2_phi(p + make_float3(e, 0, 0)) + l2_phi(p - make_float3(e, 0, 0)) + l2_phi(p + make_float3(0, e, 0))
            + l2_phi(p - make_float3(0, e, 0)) + l2_phi(p + make_float3(0, 0, e)) + l2_phi(p - make_float3(0, 0, e))) * (1.f / 7.f);
}
__device__ float slab_march(float3 p, float3 d, float& dist) {
    float tau = 0.f; int empty = 0; dist = 0.f;
    for (int i = 1; i <= 160; ++i) {
        float ph = l2_phi(p + d * (i * 1e-3f));
        if (ph <= 0.f) { if (++empty >= 3) break; continue; }
        empty = 0; tau += 2.f * ph / (HPI * HT.r_eff) * 1e-3f; dist = i * 1e-3f;
    }
    return tau;
}
extern "C" __global__ void bake_hair(int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ncell) return;
    float3 p = L2ORG + make_float3(L3IJK[3*c] + 0.5f, L3IJK[3*c+1] + 0.5f, L3IJK[3*c+2] + 0.5f) * 4e-3f;
    const float e = 2e-3f;
    float3 g = make_float3(phi7(p + make_float3(e, 0, 0)) - phi7(p - make_float3(e, 0, 0)),
                           phi7(p + make_float3(0, e, 0)) - phi7(p - make_float3(0, e, 0)),
                           phi7(p + make_float3(0, 0, e)) - phi7(p - make_float3(0, 0, e)));
    float3 rad = hnorm(p - HHC);
    float3 N = hlen(g) > 1e-3f ? hnorm(-g) : rad;
    if (hdot(N, rad) < -0.5f) N = rad;          // a gap inside the mass can point the gradient inward: keep it outward
    float du, dd;
    float tu = slab_march(p, N, du), td = slab_march(p, -N, dd);
    L3N[3*c] = N.x; L3N[3*c+1] = N.y; L3N[3*c+2] = N.z;
    L3DEP[c] = tu + td + 2.f * phi7(p) / (HPI * HT.r_eff) * 1e-3f;
    L3SURF[c] = l3_cell(p + N * (du + 4e-3f));
}

// ---------------------------------------------------------------- shading one hair point (a fibre hit or a medium sample)
// T: the fibre tangent (hit) or the voxel's mean tangent (medium); single: the fibre's own first scattering (rgb,
// already including the sun and T_sun), added by the caller. Here: the many-bounce sun (Table B) and the room (R_sh).
// Light fix (11-04, fv4_light_diag.py): Table B's room (a flat slab over black, lit from above its surface only) gave
// 0.3-0.45x the sky light a path trace of the same L2 medium gathers; with HROOM the cell's stored J (the sky by every
// order + the sun from the second order on, path-traced through the real hair mass once, static light) is read instead.
__device__ __forceinline__ float3 l3j_eval(const int* cid, const float* cw, int nc, float3 wo) {
    const float y[9] = {0.282095f, 0.488603f * wo.y, 0.488603f * wo.z, 0.488603f * wo.x, 1.092548f * wo.x * wo.y,
                        1.092548f * wo.y * wo.z, 0.315392f * (3.f * wo.z * wo.z - 1.f), 1.092548f * wo.x * wo.z,
                        0.546274f * (wo.x * wo.x - wo.y * wo.y)};
    float o[3] = {0.f, 0.f, 0.f};
    for (int j = 0; j < nc; ++j) { const __half* J = L3J + (long long)cid[j] * 27;
        for (int q = 0; q < 9; ++q) for (int ch = 0; ch < 3; ++ch) o[ch] += cw[j] * y[q] * __half2float(J[q * 3 + ch]); }
    return make_float3(fmaxf(o[0], 0.f), fmaxf(o[1], 0.f), fmaxf(o[2], 0.f));
}
__device__ float3 hair_bounce(float3 P, float3 T, float3 wo, float3 sigma, const Vox* vp) {
    int cid[8]; float cw[8]; int nc = l3_tri(P, cid, cw);
    if (HROOM && nc > 0) return l3j_eval(cid, cw, nc, wo);
    Vox v; bool ok = vp ? true : l2_vox(P, v); if (vp) v = *vp;
    float kdeg = ok ? v.kdeg : 0.f;
    float3 x = hnorm(ok ? v.t : T); if (hdot(x, T) < 0.f) x = -x;
    float3 z = wo - x * hdot(wo, x); if (hlen(z) < 1e-4f) z = hcross(x, make_float3(0.f, 0.f, 1.f)); z = hnorm(z);
    float3 y = hcross(z, x);
    float dep = 1.f, Tsurf = 1.f; float3 N = z;
    __half shb[27]; bool have_sh = nc > 0;
    if (nc > 0) {                                                // trilinear: depth, normal, the surface cell's T_sun and SH
        dep = 0.f; Tsurf = 0.f; N = make_float3(0.f, 0.f, 0.f); float sh[27];
        for (int q = 0; q < 27; ++q) sh[q] = 0.f;
        for (int j = 0; j < nc; ++j) {
            int c = cid[j], sc = L3SURF[c]; float wj = cw[j];
            dep += wj * L3DEP[c]; N += make_float3(L3N[3*c], L3N[3*c+1], L3N[3*c+2]) * wj;
            Tsurf += wj * (sc >= 0 ? __half2float(L3T[sc]) : 1.f);
            int sh_c = sc >= 0 ? sc : c;
            for (int q = 0; q < 27; ++q) sh[q] += wj * __half2float(L3SH[sh_c * 27 + q]);
        }
        N = hlen(N) > 1e-6f ? hnorm(N) : z;
        for (int q = 0; q < 27; ++q) shb[q] = __float2half(sh[q]);
    }
    float to = asinf(fminf(fmaxf(hdot(wo, x), -1.f), 1.f));
    float3 out = make_float3(0.f, 0.f, 0.f);
    if (HMS && Tsurf > 0.f && hdot(HSUN, N) > 0.f) {
        float ti = asinf(fminf(fmaxf(hdot(HSUN, x), -1.f), 1.f)), dp = atan2f(hdot(HSUN, y), hdot(HSUN, z));
        out += make_float3(tableB1(ti, to, dp, kdeg, dep, sigma.x), tableB1(ti, to, dp, kdeg, dep, sigma.y),
                           tableB1(ti, to, dp, kdeg, dep, sigma.z)) * HSUNRGB * Tsurf;
    }
    if (have_sh) {
        float3 c9[9]; sh_rotate9(shb, x, y, z, c9);
        out += tableB_room(c9, to, kdeg, dep, sigma);
    } else {                                                     // open sky: uniform radiance HSKY -> c00 = HSKY / Y00
        float3 c9[9]; c9[0] = HSKY * 3.5449077f; for (int q = 1; q < 9; ++q) c9[q] = make_float3(0.f, 0.f, 0.f);
        out += tableB_room(c9, to, kdeg, dep, sigma);
    }
    return out;
}
// F2: the h-profile of light arriving from the hair mass's outside (the lock normal N, L3N trilinear; else radial):
// near(h) / far, per channel, clamped [0.2, 5]; its mean over h is 1 (Table A = the h-integral of the near field).
__device__ float3 bounce_hshape(float3 P, float3 T, float3 wo, float h, float3 sg, float bm) {
    int cid[8]; float cw[8]; int nc = l3_tri(P, cid, cw);
    float3 N = make_float3(0.f, 0.f, 0.f);
    for (int j = 0; j < nc; ++j) N += make_float3(L3N[3*cid[j]], L3N[3*cid[j]+1], L3N[3*cid[j]+2]) * cw[j];
    N = hlen(N) > 1e-6f ? hnorm(N) : hnorm(P - HHC);
    float3 Yh = hnorm(hcross(T, wo) + make_float3(1e-9f, 0.f, 0.f)), Zh = hcross(Yh, T);
    float sO = hdot(wo, T), sI = hdot(N, T);
    float phi = atan2f(hdot(N, Zh), hdot(N, Yh)) - atan2f(hdot(wo, Zh), hdot(wo, Yh));
    float3 nr = hair_near(h, sI, sO, phi, sg, 0.f, bm), fr = hair_far(sI, sO, phi, sg, bm);
    return make_float3(fminf(fmaxf(nr.x / fmaxf(fr.x, 1e-6f), 0.2f), 5.f), fminf(fmaxf(nr.y / fmaxf(fr.y, 1e-6f), 0.2f), 5.f),
                       fminf(fmaxf(nr.z / fmaxf(fr.z, 1e-6f), 0.2f), 5.f));
}
__device__ float3 hit_sigma(int s, int k, float u) {
    return make_float3(__half2float(HSIG[(s*HPTS+k)*3]), __half2float(HSIG[(s*HPTS+k)*3+1]), __half2float(HSIG[(s*HPTS+k)*3+2])) * (1.f - u)
         + make_float3(__half2float(HSIG[(s*HPTS+k+1)*3]), __half2float(HSIG[(s*HPTS+k+1)*3+1]), __half2float(HSIG[(s*HPTS+k+1)*3+2])) * u;
}
__device__ float3 shade_fibre(float3 O, float3 D, float pix_half, const HitList8& L, int i) {
    int sk = L.sk[i], s = sk / HSEG, k = sk - s * HSEG; float u = L.u[i];
    Seg4 q = seg_load(s, k); float3 T = hnorm(cr_deriv(q, u)); float3 P = O + D * L.t[i];
    float3 wo = -D, wi = HSUN, sg = hit_sigma(s, k, u);
    float sO = hdot(wo, T), sI = hdot(wi, T), bm = strand_bm(s);
    float3 Yh = hnorm(hcross(T, wo) + make_float3(1e-9f, 0.f, 0.f)), Zh = hcross(Yh, T);
    float phi = atan2f(hdot(wi, Zh), hdot(wi, Yh)) - atan2f(hdot(wo, Zh), hdot(wo, Yh));
    float Ts = l3_Ts(P);
    float3 Lr = make_float3(0.f, 0.f, 0.f);
    if (Ts > 0.f) {                                          // no sun term where the canopy is dark toward the sun
        float3 f;
        if (L.wpx[i] >= 1.f) {
            float cd = pix_half * L.t[i] < 10e-6f ? cuticle_dalpha(strand_arc(s, k, u), HSEED[s]) : 0.f;
            f = hair_near(L.h[i], sI, sO, phi, sg, cd, bm);
        } else f = hair_far(sI, sO, phi, sg, bm);
        Lr += f * HSUNRGB * Ts;
    }
    float3 Lb = hair_bounce(P, T, wo, sg, nullptr);
    if (L.wpx[i] >= 1.f) Lb = Lb * bounce_hshape(P, T, wo, L.h[i], sg, bm);   // F2
    return Lr + Lb;
}

// ---------------------------------------------------------------- the levels per lock, with hysteresis
// wpx = the lock's mean fibre half-width / (pixel half-width at its distance): L0 >= 1, L1 down to 0.2, L2 below; also
// L2 where more than nfib_max fibres share a pixel (fibres per metre across the lock x the pixel width; 0 = off).
// A lock changes level only past the threshold x 1.3 (up) / x 0.7 (down) of its last level.
extern "C" __global__ void hair_lock_levels(const float* lock_c, const float* lock_r, const float* lock_wfib,
                                             const float* lock_fpm, float3 cam, float pix_half, float nfib_max,
                                             unsigned char* lev, int nlocks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nlocks) return;
    float3 c = make_float3(lock_c[3*i], lock_c[3*i+1], lock_c[3*i+2]);
    float dist = fmaxf(hlen(c - cam) - lock_r[i], 1e-4f);
    float wpx = lock_wfib[i] / (pix_half * dist);
    float nper = lock_fpm[i] * 2.f * pix_half * dist;
    int prev = lev[i];
    float h01 = prev == 0 ? 0.7f : 1.3f, h12 = prev == 2 ? 1.3f : 0.7f;
    int l = wpx >= 1.f * h01 ? 0 : (wpx >= 0.2f * h12 ? 1 : 2);
    if (nfib_max > 0.f && l < 2 && nper > nfib_max * (prev == 2 ? 0.7f : 1.3f)) l = 2;
    lev[i] = (unsigned char)l;
}
extern "C" __global__ void hair_brick_l2share(const unsigned* off, const unsigned* seg, const int* lock, const unsigned char* lev,
                                               float* share, int nb) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= nb) return;
    unsigned n = off[b+1] - off[b], m = 0;
    for (unsigned j = off[b]; j < off[b+1]; ++j) m += lev[lock[seg[j] / HSEG]] == 2;
    share[b] = n ? (float)m / n : 0.f;
}

// ---------------------------------------------------------------- per pixel: the hair layer over her skin (RULES)
// The fibre hits and the L2 medium are composited front to back in t order. Out: hair_rgb (premultiplied), hair_T,
// cov (the coverage AOV, 1 - hair_T), nhit.
extern "C" __global__ void hair_layer(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                      int x0, int y0, int w, int h, const float* skin_t, float* hair_rgb, float* hair_T,
                                      float* cov, unsigned char* nhit) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    int X = x0 + px, Y = y0 + py, o = py * w + px;
    float pix_half = 0.5f * pixang;
    float sx = ((X + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (Y + 0.5f)) * pixang;
    float3 D = hnorm(fwd + right * sx + up * sy);
    float tmax = skin_t ? fminf(skin_t[Y * W_ + X], 1e3f) : 1e3f;
    HitList8 L; hair_hit(cam, D, pix_half, tmax, L);
    float3 C = make_float3(0.f, 0.f, 0.f); float Tr = 1.f; int ih = 0;
    bool full = L.n == HK; float tfull = full ? L.t[HK-1] : 1e30f;
    // the medium: 1 mm steps from the grid entry; bricks with no L2 lock (share 0) cost one lookup a step
    const float BE = 8e-3f;
    float3 inv = make_float3(1.f / (fabsf(D.x) > 1e-12f ? D.x : 1e-12f), 1.f / (fabsf(D.y) > 1e-12f ? D.y : 1e-12f), 1.f / (fabsf(D.z) > 1e-12f ? D.z : 1e-12f));
    float3 lo = (L2ORG - cam) * inv, hi = (L2ORG + make_float3(L2DIM.x, L2DIM.y, L2DIM.z) * BE - cam) * inv;
    float t0 = fmaxf(fmaxf(fminf(lo.x, hi.x), fminf(lo.y, hi.y)), fmaxf(fminf(lo.z, hi.z), 0.f));
    float t1 = fminf(fminf(fmaxf(lo.x, hi.x), fmaxf(lo.y, hi.y)), fminf(fmaxf(lo.z, hi.z), tmax));
    for (float t = t0 + 0.5e-3f; t < t1 && Tr > 1e-3f; t += 1e-3f) {
        while (ih < L.n && L.t[ih] <= t && Tr > 1e-3f) {          // the fibre hits in front of this step
            C += shade_fibre(cam, D, pix_half, L, ih) * (Tr * L.a[ih]); Tr *= 1.f - L.a[ih]; ++ih;
        }
        float3 p = cam + D * t; Vox v;
        if (!l2_vox(p, v)) continue;
        float share = (full && t > tfull) ? 1.f : BRL2[v.b];
        if (share <= 0.f) continue;
        float a = 1.f - expf(-l2_sigt(v, D) * share * 1e-3f);
        float3 wo = -D; float sO = hdot(wo, v.t), sI = hdot(HSUN, v.t);
        float3 Yh = hnorm(hcross(v.t, wo) + make_float3(1e-9f, 0.f, 0.f)), Zh = hcross(Yh, v.t);
        float phi = atan2f(hdot(HSUN, Zh), hdot(HSUN, Yh)) - atan2f(hdot(wo, Zh), hdot(wo, Yh));
        float Ts = l3_Ts(p);
        float3 Lr = hair_far(sI, sO, phi, v.sig, HT.beta_m) * HSUNRGB * Ts + hair_bounce(p, v.t, wo, v.sig, &v);
        C += Lr * (Tr * a); Tr *= 1.f - a;
    }
    while (ih < L.n && Tr > 1e-3f) { C += shade_fibre(cam, D, pix_half, L, ih) * (Tr * L.a[ih]); Tr *= 1.f - L.a[ih]; ++ih; }
    hair_rgb[3*o] = C.x; hair_rgb[3*o+1] = C.y; hair_rgb[3*o+2] = C.z;
    hair_T[o] = Tr; cov[o] = 1.f - Tr; nhit[o] = (unsigned char)L.n;
}
// the frame: hair over the skin/eye pass (full frame; the hair layer may be a crop at x0, y0)
extern "C" __global__ void hair_compose(float* out, const float* hair_rgb, const float* hair_T, int W_, int x0, int y0, int w, int h) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    int o = py * w + px; long long f = (long long)(y0 + py) * W_ + (x0 + px);
    float T = hair_T[o];
    out[3*f] = hair_rgb[3*o] + T * out[3*f]; out[3*f+1] = hair_rgb[3*o+1] + T * out[3*f+1]; out[3*f+2] = hair_rgb[3*o+2] + T * out[3*f+2];
}
