// THE FIBRE LAYER (hair lane, 11-10): every hair kind drawn by one kernel from a HairSet (engine/hair/hairset.py).
// Pixel kernels take ty0 (hair-shadow lane 11-11): the first image row of the band this launch covers; FibreLayer launches
// them in bands of rows so no single launch runs long (the card's display watchdog; one scalp layer was one 15 s launch).
// Pasted AFTER fibre_scatter.cuh (float3 ops, Seg4 / cr_eval / cr_deriv, SecF / seg_hit_q, HitList8 /
// hits_insert, hair_ap / hair_near / hair_far (Chiang 2016 with Table A), cuticle_dalpha, HT, HSUN, HSUNRGB, HSKY, hhash).
// Names are F-prefixed or fl_*, so the brow and lash modules' code can sit beside it.
// What it draws (built, not painted): each strand a centripetal Catmull-Rom centreline through NPT points, an oval section
// (a/b = ell) turning with the strand's twist, the radius per point (taper, spindle), the pigment as absorption per unit LOCAL
// radius (pbrt's path is in radii: sigma x r(s) / rref, times the per-point pigment multiplier), the cuticle tilt (HT.alpha)
// and roughness (per-strand beta_m). Light: the key through the fibre's four lobes (true h when the fibre is >= 1 px wide, the
// h-integrated far field below), the room per lobe (R and TRT each lit by what their centre direction sees: the sky where it
// points away from the skin the hair grows from, the still behind the hair where it points into it), and TT carried as the
// coloured transmittance of the composite (the skin seen through a thin translucent tip). Coverage of sub-pixel fibres is
// the cone footprint overlap (seg_hit_q), so fibres far thinner than a pixel read as faint lines, not aliased dashes.
#ifndef FNPT
#define FNPT 24
#endif
#define FSPN (FNPT - 1)
struct FibreDesc { int n; int pad0, pad1, pad2; float amb, wa, wl, pad3; };
// hair2 11-10 night (the three spare slots; 0 = as before): wa = WEATHERING, the cuticle's roughness rising with the fibre's
// age (beta_m x (1 + wa x s / 0.4 m), capped at 1.5 x that distance); wl = the KEY's angular radius (rad): a sized key widens
// the key's longitudinal lobes; pad3 = 1: the open room's direction at a point of a LONG strand is the direction away from
// her body there (her field's gradient), not the root's skin normal (right for a brow hair; for a hair rooted on top of her
// head it pointed straight up through the whole curtain, so the hanging lengths saw no sky)
__constant__ FibreDesc FD;
__constant__ const float* FP;           // (n, FNPT, 3) m
__constant__ const float* FRAD;         // (n, FNPT) m
__constant__ const float* FPIG;         // (n, FNPT) pigment multiplier
__constant__ const float* FSIG;         // (n, 3) sigma_a per unit radius at FRREF
__constant__ const float* FSIGP;        // (n, FNPT, 3) or null: sigma_a per unit radius at FRREF per point (replaces FSIG x FPIG)
__constant__ int FSHADOW;               // 1: the key and the room seen through the hair itself (coloured transmittance)
__constant__ int FSKYMED;               // fs-store-5 11-13 (lane copy, off = 0 = today): 1 the sky, 2 the skin side seen through the hair's MEASURED MEDIUM (fl_Tmed) instead of one wide exact cone
// fs-store-6 11-13 (lane copy; FL_KEYDOM, off = 0 = today): THE KEY'S LIGHT THROUGH THE HAIR, built once a frame in the key's own
// view from the fibres (fl_dom_build / fl_dom_scan below): every reader of the key through the hair (each hit's key in the
// composite, the scalp's shadow on her skin) reads the map (fl_Tdom) instead of tracing exact rays through the span grid
#define FDOM_KL 12
__constant__ int FDOM;                   // 1: the key through the hair is read from the map
__constant__ float* FDOMD0;              // (ncol) each column's first fibre depth s (m; 1e30 = no hair)
__constant__ unsigned int* FDOMACC;      // (ncol, FDOM_KL + 2) the build's fixed-point sums: FDOM_KL - 1 layer bins (grey), then r g b
__constant__ float* FDOMT;               // (ncol, FDOM_KL) the grey optical depth summed from d0 to each layer boundary
__constant__ float* FDOMC;               // (ncol, 4) the column's colour ratio (tau_c / tau_grey)
__constant__ float3 FDOMA; __constant__ float3 FDOMB;   // the map's axes across the key
__constant__ float4 FDOMO;               // u0, v0, texel (m), s_off: s = s_off - P . HSUN (the depth from the key, >= 0 over the hair)
__constant__ int2 FDOMN;                 // nu, nv
__constant__ float FDOMZ[FDOM_KL];       // the layer boundaries after d0 (m)
__device__ float3 fl_Tdom(float3 P, float bias, float tmax);
__constant__ const float* FBODY;        // her body's signed distance (m) on a lattice (null: no body in the hair's light)
__constant__ float3 FBORG; __constant__ int3 FBDIM; __constant__ float FBH;
__constant__ const float* FENV;         // (h, w, 3) the environment's radiance, equirect (z up, the renderer's HDRI with its sun
__constant__ float4 FENVR;              //   removed): null = a uniform sky HSKY. FENVR = (cos rot, sin rot, w, h): world -> map
__constant__ const float* FRREF;        // (n,) m
__constant__ const float* FELL;         // (n,)
__constant__ const float* FTAU;         // (n,) 1/m
__constant__ const float* FBM;          // (n,) beta_m
__constant__ const float* FNRM;         // (n, 3) outward normal of the skin at the root
__constant__ const unsigned* FSEED;     // (n,)
__constant__ const float* FLEN;         // (n,) m
__constant__ float3 FGORG; __constant__ int3 FGDIM; __constant__ float FGH;
__constant__ const unsigned* FGOFF;     // (cells + 1)
__constant__ const int* FGIDX;          // cell -> span ids (strand * FSPN + k)

__device__ __forceinline__ float fh(int i, int c) { return hhash(FSEED[i] * 2654435761u + (unsigned)c * 104729u); }
__device__ __forceinline__ float3 fgp(int i, int j) {        // point j with second-order end ghosts
    const float* g = FP + (size_t)i * FNPT * 3;
    if (j < 0) { float3 a = make_float3(g[0], g[1], g[2]), b = make_float3(g[3], g[4], g[5]), c = make_float3(g[6], g[7], g[8]); return a * 3.f - b * 3.f + c; }
    if (j > FNPT - 1) { int e = 3 * (FNPT - 1);
        float3 a = make_float3(g[e], g[e+1], g[e+2]), b = make_float3(g[e-3], g[e-2], g[e-1]), c = make_float3(g[e-6], g[e-5], g[e-4]); return a * 3.f - b * 3.f + c; }
    return make_float3(g[3*j], g[3*j+1], g[3*j+2]);
}
__device__ __forceinline__ float frad(int i, int k, float u) { const float* r = FRAD + (size_t)i * FNPT; return r[k] * (1.f - u) + r[min(k + 1, FNPT - 1)] * u; }
__device__ __forceinline__ float fpig(int i, int k, float u) { const float* r = FPIG + (size_t)i * FNPT; return r[k] * (1.f - u) + r[min(k + 1, FNPT - 1)] * u; }
__device__ void fl_span(int i, int k, Seg4& q, SecF& sf) {
    q.p0 = fgp(i, k - 1); q.p1 = fgp(i, k); q.p2 = fgp(i, k + 1); q.p3 = fgp(i, k + 2);
    float3 d; d = q.p1 - q.p0; float d01 = fmaxf(sqrtf(sqrtf(hdot(d, d))), 1e-6f);
    d = q.p2 - q.p1; float d12 = fmaxf(sqrtf(sqrtf(hdot(d, d))), 1e-6f); d = q.p3 - q.p2; float d23 = fmaxf(sqrtf(sqrtf(hdot(d, d))), 1e-6f);
    q.t1 = d01; q.t2 = d01 + d12; q.t3 = q.t2 + d23;
    float L = FLEN[i], s0 = L * (float)k / FSPN, s1 = L * (float)(k + 1) / FSPN;
    float3 t0 = hnorm(fgp(i, 1) - fgp(i, 0)); float3 g = hcross(t0, make_float3(FNRM[3*i], FNRM[3*i+1], FNRM[3*i+2]));
    if (hdot(g, g) < 1e-6f) g = hcross(t0, make_float3(0.f, 0.f, 1.f));
    if (hdot(g, g) < 1e-6f) g = hcross(t0, make_float3(1.f, 0.f, 0.f));
    sf.g = g; sf.ell = FELL[i]; float ph = 6.2831853f * fh(i, 5);
    sf.psi0 = ph + FTAU[i] * s0; sf.psi1 = ph + FTAU[i] * s1; sf.r0 = frad(i, k, 0.f); sf.r1 = frad(i, k, 1.f);
}
// hair-shadow lane 11-11: a cheap, conservative pre-test before a span's full hit test (fl_span + seg_hit_q: the curve's 8 chords
// and 3 Newton steps): the span lies within a sphere about its chord's midpoint of radius (chord + its largest radius x 1.2)
// (half the chord, the curve's bulge, generously), so a ray whose cone (half-angle pen) passes farther than that from the centre,
// or outside [tmin, tmax], cannot hit it. Rejects most spans in a cell without touching the curve; the hits are unchanged.
__device__ __forceinline__ bool fl_may_hit(int i, int k, float3 O, float3 D, float pen, float tmin, float tmax) {
    const float* g = FP + ((size_t)i * FNPT + k) * 3;
    float3 a = make_float3(g[0], g[1], g[2]), b = make_float3(g[3], g[4], g[5]);
    float3 c = (a + b) * 0.5f; float R = hlen(b - a) + 1.2f * fmaxf(frad(i, k, 0.f), frad(i, k, 1.f)) + 1e-6f;
    float3 oc = c - O; float tc = hdot(oc, D);
    if (tc + R <= tmin || tc - R >= tmax) return false;
    float3 x = oc - D * tc; float lim = R + pen * fmaxf(tc + R, 0.f);
    return hdot(x, x) <= lim * lim;
}
template <typename F> __device__ void fl_walk(float3 O, float3 D, float tmax, F f) {
    float3 lo = FGORG, hi = FGORG + make_float3(FGDIM.x * FGH, FGDIM.y * FGH, FGDIM.z * FGH);
    float t0 = 0.f, t1 = tmax; float o[3] = {O.x, O.y, O.z}, d[3] = {D.x, D.y, D.z}, l[3] = {lo.x, lo.y, lo.z}, h[3] = {hi.x, hi.y, hi.z};
    for (int a = 0; a < 3; ++a) { float inv = 1.f / (fabsf(d[a]) > 1e-12f ? d[a] : 1e-12f);
        float ta = (l[a] - o[a]) * inv, tb = (h[a] - o[a]) * inv; if (ta > tb) { float s = ta; ta = tb; tb = s; }
        t0 = fmaxf(t0, ta); t1 = fminf(t1, tb); }
    if (t0 > t1) return;
    float3 p = O + D * (t0 + 1e-7f); int dim[3] = {FGDIM.x, FGDIM.y, FGDIM.z};
    int c[3] = {(int)floorf((p.x - lo.x) / FGH), (int)floorf((p.y - lo.y) / FGH), (int)floorf((p.z - lo.z) / FGH)};
    int st[3]; float tn[3], td[3];
    for (int a = 0; a < 3; ++a) { c[a] = min(max(c[a], 0), dim[a] - 1); st[a] = d[a] >= 0.f ? 1 : -1;
        float nb = l[a] + (c[a] + (st[a] > 0 ? 1 : 0)) * FGH; float ad = fabsf(d[a]) > 1e-12f ? d[a] : 1e-12f;
        tn[a] = (nb - o[a]) / ad; td[a] = FGH / fabsf(ad); }
    float tstop = 1e30f; float tc = t0;
    for (int it = 0; it < 8192; ++it) {
        if (tc > tstop || tc > t1) return;
        int cell = (c[0] * dim[1] + c[1]) * dim[2] + c[2];
        for (unsigned j = FGOFF[cell]; j < FGOFF[cell + 1]; ++j) tstop = fminf(tstop, f(FGIDX[j]));
        int a = tn[0] < tn[1] ? (tn[0] < tn[2] ? 0 : 2) : (tn[1] < tn[2] ? 1 : 2);
        tc = tn[a]; c[a] += st[a]; if (c[a] < 0 || c[a] >= dim[a]) return; tn[a] += td[a];
    }
}
__device__ void fl_hits(float3 O, float3 D, float pix_half, float tmax, HitList8& L, float tmin = 1e-5f) {
    L.n = 0;
    fl_walk(O, D, tmax, [&](int sid) -> float {
        int i = sid / FSPN, k = sid - i * FSPN;
        if (fl_may_hit(i, k, O, D, pix_half, tmin, tmax)) { Seg4 q; SecF sf; fl_span(i, k, q, sf);
            float th, al, hh, uh, wpx;
            if (seg_hit_q(q, sf, O, D, pix_half, tmin, tmax, th, al, hh, uh, wpx)) hits_insert(L, HK, th, al, hh, uh, sid, wpx); }
        return L.n == HK ? L.t[HK - 1] + 2.f * FGH : 1e30f;
    });
}
__device__ __forceinline__ float3 fl_sigma(int i, int k, float u) {
    float s = frad(i, k, u) / fmaxf(FRREF[i], 1e-9f);
    if (FSIGP) { const float* a = FSIGP + ((size_t)i * FNPT + k) * 3; int k1 = min(k + 1, FNPT - 1); const float* b = FSIGP + ((size_t)i * FNPT + k1) * 3;
        return make_float3(a[0] * (1.f - u) + b[0] * u, a[1] * (1.f - u) + b[1] * u, a[2] * (1.f - u) + b[2] * u) * s; }
    return make_float3(FSIG[3*i], FSIG[3*i+1], FSIG[3*i+2]) * (s * fpig(i, k, u));
}
// the light that gets through the hair from P along Ldir: EVERY fibre crossed (not only the 8 nearest), each counted once in
// the cell that holds its hit point; per fibre its uncovered share plus what its TT lobe passes (forward scattering: a blond
// fibre passes much of the light on, coloured; a dark one little); cone footprint pen (rad), so far fibres shade softly
// fwd = 1 (hair2 11-11): each fibre passes on its whole FORWARD-scattered light, not only its TT lobe: TT + the share of the R
// lobe that leaves into the forward half-space (0.293: the integral of cos(phi/2)/4 over |phi| > 90 deg, a smooth dielectric
// cylinder) + about 0.15 of TRT (approximate: TRT leaves mostly backward). This is the forward attenuation a_f of dual
// scattering (Zinke et al. 2008) and is what diffuse light inside a mass of hair sees; TT alone compounds to far too little
// over tens of fibres (0.75^20 = 0.003 against 0.82^20 = 0.02 ... 0.9^20 = 0.12 for light blond).
__device__ float3 fl_Trc(float3 O, float3 D, float pen, float tmax, int fwd = 0) {
    float3 Tc = make_float3(1.f, 1.f, 1.f);
    float3 lo = FGORG, hi = FGORG + make_float3(FGDIM.x * FGH, FGDIM.y * FGH, FGDIM.z * FGH);
    float t0 = 0.f, t1 = tmax; float o[3] = {O.x, O.y, O.z}, d[3] = {D.x, D.y, D.z}, l[3] = {lo.x, lo.y, lo.z}, h[3] = {hi.x, hi.y, hi.z};
    for (int a = 0; a < 3; ++a) { float inv = 1.f / (fabsf(d[a]) > 1e-12f ? d[a] : 1e-12f);
        float ta = (l[a] - o[a]) * inv, tb = (h[a] - o[a]) * inv; if (ta > tb) { float s_ = ta; ta = tb; tb = s_; }
        t0 = fmaxf(t0, ta); t1 = fminf(t1, tb); }
    if (t0 > t1) return Tc;
    float3 p = O + D * (t0 + 1e-7f); int dim[3] = {FGDIM.x, FGDIM.y, FGDIM.z};
    int c[3] = {(int)floorf((p.x - lo.x) / FGH), (int)floorf((p.y - lo.y) / FGH), (int)floorf((p.z - lo.z) / FGH)};
    int st[3]; float tn[3], td[3];
    for (int a = 0; a < 3; ++a) { c[a] = min(max(c[a], 0), dim[a] - 1); st[a] = d[a] >= 0.f ? 1 : -1;
        float nb = l[a] + (c[a] + (st[a] > 0 ? 1 : 0)) * FGH; float ad = fabsf(d[a]) > 1e-12f ? d[a] : 1e-12f;
        tn[a] = (nb - o[a]) / ad; td[a] = FGH / fabsf(ad); }
    float tc = t0;
    for (int it = 0; it < 4096; ++it) {
        int a = tn[0] < tn[1] ? (tn[0] < tn[2] ? 0 : 2) : (tn[1] < tn[2] ? 1 : 2);
        float tx = fminf(tn[a], t1);
        int cell = (c[0] * dim[1] + c[1]) * dim[2] + c[2];
        for (unsigned j = FGOFF[cell]; j < FGOFF[cell + 1]; ++j) {
            int sid = FGIDX[j]; int i = sid / FSPN, k = sid - i * FSPN;
            if (!fl_may_hit(i, k, O, D, pen, 1e-6f, tmax)) continue;
            Seg4 q; SecF sf; fl_span(i, k, q, sf);
            float th, al, hh, uh, wpx;
            if (!seg_hit_q(q, sf, O, D, pen, 1e-6f, tmax, th, al, hh, uh, wpx)) continue;
            if (th < tc || th >= tx) continue;                         // counted in the cell that holds the hit point only
            float3 T = hnorm(cr_deriv(q, uh)); Ap4 A = hair_ap(0.f, hdot(D, T), fl_sigma(i, k, uh));
            float3 af = fwd ? A.a[1] + A.a[0] * 0.293f + A.a[2] * 0.15f : A.a[1];
            Tc = Tc * (make_float3(1.f, 1.f, 1.f) * (1.f - al) + af * al);
        }
        if (fmaxf(Tc.x, fmaxf(Tc.y, Tc.z)) < 2e-3f || tx >= t1) return Tc;
        tc = tn[a]; c[a] += st[a]; if (c[a] < 0 || c[a] >= dim[a]) return Tc; tn[a] += td[a];
    }
    return Tc;
}
// the span grid built on the card: pass 0 counts each cell's spans, pass 1 writes them (cur = a copy of the offsets). Each span's
// curve is sampled every 0.4 cell; a sample within the fibre's radius + pad of a cell face lists the span in the neighbour too
extern "C" __global__ void fl_grid(int nspan, float pad, unsigned* cnt, unsigned* cur, int* idx, int pass) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x; if (sid >= nspan) return;
    int i = sid / FSPN, k = sid - i * FSPN; Seg4 q; SecF sf; fl_span(i, k, q, sf);
    float r = (fmaxf(sf.r0, sf.r1) + pad) / FGH; float ch = hlen(q.p2 - q.p1);
    int m = max(2, (int)ceilf(ch / (0.4f * FGH)) + 1); int ring[8]; for (int j = 0; j < 8; ++j) ring[j] = -1; int rp = 0;
    for (int j = 0; j < m; ++j) {
        float3 pp = cr_eval(q, (float)j / (float)(m - 1)); float g[3] = {(pp.x - FGORG.x) / FGH, (pp.y - FGORG.y) / FGH, (pp.z - FGORG.z) / FGH};
        int cc[3], lo_[3], hi_[3];
        for (int a = 0; a < 3; ++a) { cc[a] = (int)floorf(g[a]); float fr = g[a] - cc[a]; lo_[a] = fr < r ? -1 : 0; hi_[a] = fr > 1.f - r ? 1 : 0; }
        for (int dx = lo_[0]; dx <= hi_[0]; ++dx) for (int dy = lo_[1]; dy <= hi_[1]; ++dy) for (int dz = lo_[2]; dz <= hi_[2]; ++dz) {
            int x = cc[0] + dx, y = cc[1] + dy, z = cc[2] + dz;
            if (x < 0 || y < 0 || z < 0 || x >= FGDIM.x || y >= FGDIM.y || z >= FGDIM.z) continue;
            int cell = (x * FGDIM.y + y) * FGDIM.z + z; bool seen = false;
            for (int e = 0; e < 8; ++e) if (ring[e] == cell) { seen = true; break; }
            if (seen) continue;
            ring[rp] = cell; rp = (rp + 1) & 7;
            if (pass == 0) atomicAdd(&cnt[cell], 1u); else { unsigned w = atomicAdd(&cur[cell], 1u); idx[w] = sid; }
        }
    }
}
// her body's signed distance at P (trilinear on the lattice; outside the lattice: far = free)
__device__ __forceinline__ float fl_body(float3 P) {
    float fx = (P.x - FBORG.x) / FBH, fy = (P.y - FBORG.y) / FBH, fz = (P.z - FBORG.z) / FBH;
    if (fx < 0.f || fy < 0.f || fz < 0.f || fx > FBDIM.x - 1.001f || fy > FBDIM.y - 1.001f || fz > FBDIM.z - 1.001f) return 1.f;
    int x = (int)fx, y = (int)fy, z = (int)fz; float u = fx - x, v = fy - y, w = fz - z;
    const float* A = FBODY; long long sx = (long long)FBDIM.y * FBDIM.z, sy = FBDIM.z; long long b = x * sx + y * sy + z;
    float c00 = A[b] * (1 - w) + A[b + 1] * w, c01 = A[b + sy] * (1 - w) + A[b + sy + 1] * w;
    float c10 = A[b + sx] * (1 - w) + A[b + sx + 1] * w, c11 = A[b + sx + sy] * (1 - w) + A[b + sx + sy + 1] * w;
    return (c00 * (1 - v) + c01 * v) * (1 - u) + (c10 * (1 - v) + c11 * v) * u;
}
// the light from direction L reaching P past her body: a soft SDF shadow march (penumbra factor k), 1 = unoccluded
__device__ float fl_body_vis(float3 P, float3 L, float k, float tmax) {
    if (!FBODY) return 1.f;
    float res = 1.f, t = fmaxf(1.5f * FBH, 4e-3f);
    for (int i = 0; i < 96 && t < tmax; ++i) {
        float d = fl_body(P + L * t);
        if (d < 1e-4f) return 0.f;
        res = fminf(res, k * d / t); t += fmaxf(d, 1.0e-3f);
    }
    return fminf(fmaxf(res, 0.f), 1.f);
}
// the sky past her body: soft visibility toward straight up and toward the open side (away from the nearest body surface)
__device__ __forceinline__ float3 fl_body_grad(float3 P) {
    float e = FBH; float3 g = make_float3(fl_body(P + make_float3(e, 0, 0)) - fl_body(P - make_float3(e, 0, 0)),
                                          fl_body(P + make_float3(0, e, 0)) - fl_body(P - make_float3(0, e, 0)),
                                          fl_body(P + make_float3(0, 0, e)) - fl_body(P - make_float3(0, 0, e)));
    return hnorm(g + make_float3(0.f, 0.f, 1e-9f));
}
__device__ __forceinline__ float fl_bm_eff(float bm, float s) {
    bm = bm * (1.f + FD.wa * fminf(s / 0.4f, 1.5f));
    if (FD.wl > 0.f) { float x = sqrtf(var_b(bm) + 0.25f * FD.wl * FD.wl); bm = (-0.726f + sqrtf(0.527076f + 3.248f * x)) / 1.624f; }
    return fminf(bm, 0.95f);
}
__device__ float fl_body_sky(float3 P) {
    if (!FBODY) return 1.f;
    float e = FBH; float3 g = make_float3(fl_body(P + make_float3(e, 0, 0)) - fl_body(P - make_float3(e, 0, 0)),
                                          fl_body(P + make_float3(0, e, 0)) - fl_body(P - make_float3(0, e, 0)),
                                          fl_body(P + make_float3(0, 0, e)) - fl_body(P - make_float3(0, 0, e)));
    float3 o = hdot(g, g) > 1e-12f ? hnorm(g) : make_float3(0.f, 0.f, 1.f);
    return 0.5f * fl_body_vis(P, make_float3(0.f, 0.f, 1.f), 2.f, 0.3f) + 0.5f * fl_body_vis(P, o, 2.f, 0.3f);
}
// light scattered back out of the hair beneath a fibre (multiple scattering, the bulk of light hair's brightness): per fibre its
// back-scattered share b (R, TRT, TRRT) and forward share f (TT), averaged over the offset h; from a deep mass the light that
// comes back after going forward and back any number of times, B = b / (1 - f^2) (clamped). Scaled by how much hair lies
// beneath (1 - the skin side's transmittance through it) and lit by the key that reaches this depth and the sky above.
__device__ float3 fl_backscatter(float3 sg, float sO) {
    float3 b = make_float3(0.f, 0.f, 0.f), fw = make_float3(0.f, 0.f, 0.f); const float hs[4] = {-0.75f, -0.25f, 0.25f, 0.75f};
    for (int k = 0; k < 4; ++k) { Ap4 A = hair_ap(hs[k], sO, sg); b += (A.a[0] + A.a[2] + A.a[3]) * 0.25f; fw += A.a[1] * 0.25f; }
    float3 B = make_float3(b.x / fmaxf(1.f - fw.x * fw.x, 0.1f), b.y / fmaxf(1.f - fw.y * fw.y, 0.1f), b.z / fmaxf(1.f - fw.z * fw.z, 0.1f));
    return make_float3(fminf(B.x, 0.9f), fminf(B.y, 0.9f), fminf(B.z, 0.9f));
}
// the sky's radiance along a world direction: the environment map where there is one (the same map and rotation the renderer
// lights her skin with), else the uniform sky
__device__ float3 fl_env(float3 w) {
    if (!FENV) return HSKY;
    float x = FENVR.x * w.x - FENVR.y * w.y, y = FENVR.y * w.x + FENVR.x * w.y, z = fminf(fmaxf(w.z, -1.f), 1.f);
    float az = atan2f(y, x), el = asinf(z); int W = (int)FENVR.z, H = (int)FENVR.w;
    float u = 0.5f - az / 6.2831853f, v = el / 3.14159265f + 0.5f;
    float fx = u * W - 0.5f, fy = (1.f - v) * H - 0.5f; int x0 = (int)floorf(fx), y0 = (int)floorf(fy); float ax = fx - x0, ay = fy - y0;
    float3 c = make_float3(0.f, 0.f, 0.f);
    for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
        int xx = ((x0 + dx) % W + W) % W, yy = min(max(y0 + dy, 0), H - 1); const float* p = FENV + ((size_t)yy * W + xx) * 3;
        float wgt = (dx ? ax : 1.f - ax) * (dy ? ay : 1.f - ay); c += make_float3(p[0], p[1], p[2]) * wgt;
    }
    return c;
}
// the room seen along w: the sky away from the skin the strand grows from, the still behind it (bg) toward the skin
__device__ __forceinline__ float3 fl_room(float3 w, float3 N, float3 bg) {
    float c = hdot(w, N); float x = fminf(fmaxf((c + 0.25f) / 0.5f, 0.f), 1.f); x = x * x * (3.f - 2.f * x);
    return HSKY * x + bg * (1.f - x);
}
__device__ __forceinline__ float3 fl_room2(float3 w, float3 N, float3 bg, float3 vsky) {
    float c = hdot(w, N); float x = fminf(fmaxf((c + 0.25f) / 0.5f, 0.f), 1.f); x = x * x * (3.f - 2.f * x);
    return fl_env(w) * vsky * x + bg * (1.f - x);
}
__device__ float3 fl_room_h(float h, float sO, float cO, float3 T, float3 Yh, float3 Zh, float phio, float3 sg, float3 N, float3 bg, float3 vsky, float3& tt) {
    Ap4 A = hair_ap(h, sO, sg); tt = A.a[1];
    float sI = -sO, cI = cO; float3 r = make_float3(0.f, 0.f, 0.f);
    float ph0 = phio - 2.f * A.gO, ph2 = phio + 4.f * A.gT - 2.f * A.gO + 2.f * HPI;
    float3 w0 = T * sI + (Yh * cosf(ph0) + Zh * sinf(ph0)) * cI, w2 = T * sI + (Yh * cosf(ph2) + Zh * sinf(ph2)) * cI;
    r += A.a[0] * fl_room2(w0, N, bg, vsky) + A.a[2] * fl_room2(w2, N, bg, vsky) + A.a[3] * (HSKY * vsky * 0.5f + bg * 0.5f);
    return r;
}
__device__ float3 fl_shade(float3 T, float3 wo, float h, float wpx, float3 sg, float bm, float s, float pixw, unsigned seed, float3 vis,
                           float3 N, float3 bg, float3 vsky, float3& tt) {
    float sO = hdot(wo, T), sI = hdot(HSUN, T), cO = sqrtf(fmaxf(1.f - sO * sO, 0.f));
    float3 Yh = hnorm(hcross(T, wo) + make_float3(1e-9f, 0.f, 0.f)), Zh = hcross(Yh, T);
    float phio = atan2f(hdot(wo, Zh), hdot(wo, Yh));
    float phi = atan2f(hdot(HSUN, Zh), hdot(HSUN, Yh)) - phio;
    float3 f, room;
    // the room: what each lobe's centre direction sees, with the sky seen THROUGH the hair above (vsky) and the skin side seen
    // through the hair below (bg already carries that: the caller passes bg x its transmittance toward the skin)
    if (wpx >= 1.f) { float cd = pixw < 10e-6f ? cuticle_dalpha(s, seed) : 0.f; f = hair_near(h, sI, sO, phi, sg, cd, bm);
        room = fl_room_h(h, sO, cO, T, Yh, Zh, phio, sg, N, bg, vsky, tt); }
    else { f = hair_far(sI, sO, phi, sg, bm); room = make_float3(0.f, 0.f, 0.f); tt = make_float3(0.f, 0.f, 0.f);
        const float hs[4] = {-0.75f, -0.25f, 0.25f, 0.75f};
        for (int k = 0; k < 4; ++k) { float3 t1; room += fl_room_h(hs[k], sO, cO, T, Yh, Zh, phio, sg, N, bg, vsky, t1) * 0.25f; tt += t1 * 0.25f; } }
    return f * HSUNRGB * vis + room * FD.amb;
}
__device__ __forceinline__ unsigned fl_hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
__device__ __forceinline__ float fl_u01(unsigned x) { return (x & 0xffffff) * (1.f / 16777216.f); }
// THE LAYER'S TRACE (hair-shadow lane 11-11: fl_layer's body, unchanged, as a function of the ray, so any ray can see the hair):
// along O + t D (unit D, cone half-angle pix_half, up to tz, the room behind = bg): premultiplied rgb C, coloured transmittance
// Tc, scalar coverage transmittance Tr, the first hit's distance d0 and strand id0.
__device__ float3 fl_Tmed(float3 P, float3 w, float pen, float tmax, int set);   // fs-store-5: defined with the medium below
// fs-store-6 11-13 (lane copy, tests only): FREF > 0 = THE REFERENCE for the hair's light: the key's disc and every sky /
// skin-side cone integrated by FREF thin exact rays each (uniform in solid angle inside the cone, R2-stratified, seeded by the
// hit), instead of one sized cone (today) or the medium / the map. Slow; judges which of those is closer to the truth.
__constant__ int FREF;
__device__ __noinline__ float3 fl_cone_ref(float3 O, float3 ax, float half, float tmax, int fwd, int n, unsigned seed) {   // noinline: hs3 ran ptxas out of memory
    float3 e1 = hnorm(hcross(ax, fabsf(ax.z) < 0.9f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f))), e2 = hcross(ax, e1);
    float ch = cosf(half), r1 = fl_u01(fl_hash(seed)), r2 = fl_u01(fl_hash(seed * 7u + 13u)); float3 acc = make_float3(0.f, 0.f, 0.f);
    for (int k = 0; k < n; ++k) {
        float u = r1 + k * 0.7548776662f, v = r2 + k * 0.5698402910f; u -= floorf(u); v -= floorf(v);
        float ct = 1.f - u * (1.f - ch), st = sqrtf(fmaxf(1.f - ct * ct, 0.f)), ph = 6.2831853f * v;
        acc = acc + fl_Trc(O, hnorm(ax * ct + (e1 * cosf(ph) + e2 * sinf(ph)) * st), 1e-7f, tmax, fwd);
    }
    return acc * (1.f / (float)n);
}
// one round of fl_trace: the (up to 8) nearest hits in L, shaded and composited front to back into C, Tc, Tr
__device__ __forceinline__ void fl_trace_round(float3 cam, float3 D, float pix_half, float3 bg, const HitList8& L, float3& C, float3& Tc, float& Tr) {
    float3 wo = -D;
    for (int j = 0; j < L.n && Tr > 1e-3f; ++j) {
        int i = L.sk[j] / FSPN, k = L.sk[j] - i * FSPN; Seg4 q; SecF sf; fl_span(i, k, q, sf);
        float3 T = hnorm(cr_deriv(q, L.u[j])); float s = FLEN[i] * ((float)k + L.u[j]) / (float)FSPN;
        float3 N = make_float3(FNRM[3*i], FNRM[3*i+1], FNRM[3*i+2]); float3 tt;
        if (FBODY && FD.pad3 > 0.5f) { float wn = fminf(fmaxf((s - 0.01f) / 0.03f, 0.f), 1.f); wn = wn * wn * (3.f - 2.f * wn);
            if (wn > 0.f) N = hnorm(N * (1.f - wn) + fl_body_grad(cam + D * L.t[j]) * wn); }
        float3 one = make_float3(1.f, 1.f, 1.f), vis = one, vsk = one, tdn = one;
        if (FBODY && Tr * L.a[j] > 0.02f) {                     // her body's shadow on the hair and its share of the sky
            float3 Pb = cam + D * L.t[j]; vis = one * fl_body_vis(Pb, HSUN, FD.wl > 0.125f ? 1.f / FD.wl : 8.f, 0.4f); vsk = one * fl_body_sky(Pb);
        }
        if (FSHADOW && Tr * L.a[j] > 0.02f) {
            float3 Ph = cam + D * L.t[j]; float rr = frad(i, k, L.u[j]);
            if (FREF > 0) {                                                              // fs-store-6: the reference (tests only)
                unsigned sd = fl_hash((unsigned)L.sk[j] * 9781u ^ __float_as_uint(L.t[j]));
                if (vis.x > 1e-3f) vis = vis * fl_cone_ref(Ph + HSUN * (2.5f * rr), HSUN, fmaxf(0.00465f, FD.wl), 0.25f, 0, FREF, sd);
                if (FD.pad0 == 0) vsk = vsk * fl_cone_ref(Ph + N * (2.5f * rr), N, 0.35f, 0.25f, 0, FREF, sd + 1u);
                else { float3 b1 = hnorm(hcross(N, fabsf(N.z) < 0.9f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f))), b2 = hcross(N, b1);
                    float3 acc = fl_cone_ref(Ph + N * (2.5f * rr), N, 0.35f, 0.25f, 1, FREF, sd + 2u) * 0.25f;
                    const float cz[3] = {1.f, -0.5f, -0.5f}, sz[3] = {0.f, 0.8660254f, -0.8660254f};
                    for (int q3 = 0; q3 < 3; ++q3) { float3 dq = hnorm(N * 0.5f + (b1 * cz[q3] + b2 * sz[q3]) * 0.8660254f);
                        acc = acc + fl_cone_ref(Ph + dq * (2.5f * rr), dq, 0.35f, 0.18f, 1, FREF, sd + 3u + (unsigned)q3) * 0.25f; }
                    vsk = vsk * acc; }
                tdn = fl_cone_ref(Ph - N * (2.5f * rr), -N, 0.35f, 0.05f, 0, FREF, sd + 9u) * vsk;
            } else {
            if (vis.x > 1e-3f) vis = vis * (FDOM ? fl_Tdom(Ph, 2.5f * rr + 0.5f * FDOMO.z, 0.25f) : fl_Trc(Ph + HSUN * (2.5f * rr), HSUN, fmaxf(0.00465f, FD.wl) + 2.f * rr / 0.01f, 0.25f));   // a sized key: soft self-shadow
            if (FD.pad0 == 0) vsk = vsk * ((FSKYMED & 1) ? fl_Tmed(Ph + N * (2.5f * rr), N, 0.35f, 0.25f, 1) : fl_Trc(Ph + N * (2.5f * rr), N, 0.35f, 0.25f));
            else {
                // hair2 11-11: the sky reaches a fibre from the whole hemisphere about the outward direction, not only
                // straight out. One ray along N takes the SHORTEST way out of the mass and so overstates the sky inside
                // it: the hair came out too bright against her skin and too evenly lit inside (lit side / shadow side
                // of the lengths 1.2 against the TARGET's 2.1 while her skin's lit / shadow matched at 2.4). Four cones,
                // cosine-weighted: N (0-30 deg: a quarter of the flux) and three at 60 deg from N (a quarter each).
                float3 a1 = hnorm(hcross(N, fabsf(N.z) < 0.9f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f))), a2 = hcross(N, a1);
                float3 acc = ((FSKYMED & 1) ? fl_Tmed(Ph + N * (2.5f * rr), N, 0.35f, 0.25f, 2) : fl_Trc(Ph + N * (2.5f * rr), N, 0.35f, 0.25f, 1)) * 0.25f;
                const float cz[3] = {1.f, -0.5f, -0.5f}, sz[3] = {0.f, 0.8660254f, -0.8660254f};
                for (int q3 = 0; q3 < 3; ++q3) {
                    float3 dq = hnorm(N * 0.5f + (a1 * cz[q3] + a2 * sz[q3]) * 0.8660254f);
                    acc = acc + ((FSKYMED & 1) ? fl_Tmed(Ph + dq * (2.5f * rr), dq, 0.35f, 0.18f, 2) : fl_Trc(Ph + dq * (2.5f * rr), dq, 0.35f, 0.18f, 1)) * 0.25f;     // (her body's share of the sky is already in vsk)
                }
                vsk = vsk * acc;
            }
            tdn = ((FSKYMED & 2) ? fl_Tmed(Ph - N * (2.5f * rr), -N, 0.35f, 0.05f, 1) : fl_Trc(Ph - N * (2.5f * rr), -N, 0.35f, 0.05f)) * vsk;   // the skin side: through the hair below, the skin itself under the hair above
            }                                                                            // fs-store-6: end of the non-reference branch
        }
        float3 sgh = fl_sigma(i, k, L.u[j]);
        float3 f = fl_shade(T, wo, L.h[j], L.wpx[j], sgh, fl_bm_eff(FBM[i], s), s, pix_half * L.t[j], FSEED[i], vis, N, bg * tdn, vsk, tt);
        if (FSHADOW && Tr * L.a[j] > 0.02f) {
            float dense = 1.f - (tdn.x + tdn.y + tdn.z) / 3.f;
            if (dense > 0.f) { float3 Bk = fl_backscatter(sgh, hdot(wo, T));
                f += Bk * (HSUNRGB * vis * 0.5f + HSKY * vsk * 1.5708f) * (dense * 0.3183f) * FD.amb; }
        }
        C += f * Tc * L.a[j]; Tr *= 1.f - L.a[j];
        Tc = Tc * (make_float3(1.f, 1.f, 1.f) * (1.f - L.a[j]) + tt * L.a[j]);
    }
}
__device__ void fl_trace(float3 cam, float3 D, float pix_half, float tz, float3 bg, float3& C, float3& Tc, float& Tr, float& d0, int& id0) {
    C = make_float3(0.f, 0.f, 0.f); Tc = make_float3(1.f, 1.f, 1.f); Tr = 1.f;
    HitList8 L; fl_hits(cam, D, pix_half, tz, L);
    d0 = L.n ? L.t[0] : 1e30f; id0 = L.n ? L.sk[0] / FSPN : -1;
    // rounds of 8 hits, nearest first: a dense mass of sub-pixel fibres needs more than 8 to close a pixel (up to 6 rounds)
    for (int rnd = 0; rnd < 6; ++rnd) {
        fl_trace_round(cam, D, pix_half, bg, L, C, Tc, Tr);
        if (L.n < HK || Tr <= 0.02f) break;
        float tl = L.t[HK - 1]; fl_hits(cam, D, pix_half, tz, L, tl * (1.f + 1e-6f) + 1e-7f);
        if (L.n == 0) break;
    }
}
// THE LAYER: per pixel premultiplied rgb, scalar coverage transmittance T, coloured transmittance Tc (1 - a + a A_TT per hit),
// the first hit's depth and strand id. zbuf: the still's primary-hit distance along the same normalised ray (rules3c depth[]),
// so only fibre in front of her surface is drawn; bgimg: the still (linear), the room behind the hair. Composite: rgb + Tc x still.
// The ray is the engine's (pixang = 2 tan(hfov / 2) / w).
extern "C" __global__ void fl_layer(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                    const float* zbuf, const float* bgimg, float* rgb, float* Tout, float* Tcout, float* depth, int* idout, int ty0) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = ty0 + blockIdx.y * blockDim.y + threadIdx.y; if (px >= W_ || py >= H_) return;
    int o = py * W_ + px; float pix_half = 0.5f * pixang;
    float sx = ((px + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (py + 0.5f)) * pixang;
    float tz = fminf(zbuf[o], 1e3f);
    float3 bg = bgimg ? make_float3(bgimg[3*o], bgimg[3*o+1], bgimg[3*o+2]) : HSKY;
    float3 D = hnorm(fwd + right * sx + up * sy);
    float3 C, Tc; float Tr, d0; int id0; fl_trace(cam, D, pix_half, tz, bg, C, Tc, Tr, d0, id0);
    rgb[3*o] = C.x; rgb[3*o+1] = C.y; rgb[3*o+2] = C.z; Tout[o] = Tr; Tcout[3*o] = Tc.x; Tcout[3*o+1] = Tc.y; Tcout[3*o+2] = Tc.z;
    depth[o] = d0; idout[o] = id0;
}
// THE LAYER IN ROUNDS (hair-shadow lane 11-11): fl_layer split so one launch does one round (the 8 nearest hits) per pixel and
// carries the rest in its outputs (C, Tr, Tc as is; st: the next round's start distance, or < 0 when the pixel is done), so no
// launch waits on a dense pixel's whole 6 rounds (one 32-row band of the scalp took 1.5 s). The same arithmetic in the same
// order as fl_layer: the same image.
extern "C" __global__ void fl_layer_rnd(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                        const float* zbuf, const float* bgimg, float* rgb, float* Tout, float* Tcout, float* depth, int* idout,
                                        float* st, int rnd, int ty0) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = ty0 + blockIdx.y * blockDim.y + threadIdx.y; if (px >= W_ || py >= H_) return;
    int o = py * W_ + px; if (rnd > 0 && st[o] < 0.f) return;
    float pix_half = 0.5f * pixang;
    float sx = ((px + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (py + 0.5f)) * pixang;
    float tz = fminf(zbuf[o], 1e3f);
    float3 bg = bgimg ? make_float3(bgimg[3*o], bgimg[3*o+1], bgimg[3*o+2]) : HSKY;
    float3 D = hnorm(fwd + right * sx + up * sy);
    float3 C, Tc; float Tr; HitList8 L;
    if (rnd == 0) {
        C = make_float3(0.f, 0.f, 0.f); Tc = make_float3(1.f, 1.f, 1.f); Tr = 1.f;
        fl_hits(cam, D, pix_half, tz, L);
        depth[o] = L.n ? L.t[0] : 1e30f; idout[o] = L.n ? L.sk[0] / FSPN : -1;
    } else {
        C = make_float3(rgb[3*o], rgb[3*o+1], rgb[3*o+2]); Tr = Tout[o]; Tc = make_float3(Tcout[3*o], Tcout[3*o+1], Tcout[3*o+2]);
        fl_hits(cam, D, pix_half, tz, L, st[o]);
    }
    if (rnd > 0 && L.n == 0) { st[o] = -1.f; return; }
    fl_trace_round(cam, D, pix_half, bg, L, C, Tc, Tr);
    rgb[3*o] = C.x; rgb[3*o+1] = C.y; rgb[3*o+2] = C.z; Tout[o] = Tr; Tcout[3*o] = Tc.x; Tcout[3*o+1] = Tc.y; Tcout[3*o+2] = Tc.z;
    st[o] = (L.n < HK || Tr <= 0.02f || rnd == 5) ? -1.f : L.t[HK - 1] * (1.f + 1e-6f) + 1e-7f;
}
// THE HAIR ALONG GIVEN RAYS (hair-shadow lane 11-11, L-1027: her eyes' reflections see her lashes and hair). ray: n x 11 floats
// (origin 3, unit direction 3, cone half-angle, far limit, the radiance behind 3); out: n x 8 (C 3, Tc 3, Tr, d0). ns > 1: the
// cone sampled by ns x ns stratified exact sub-rays (the fibre-resolved reference); i0: the first ray of this launch.
extern "C" __global__ void fl_rays(int n, const float* ray, int ns, float* out, int i0, int a0, int a1) {
    int i = i0 + blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const float* r = ray + (size_t)i * 11;
    float3 O = make_float3(r[0], r[1], r[2]), D = hnorm(make_float3(r[3], r[4], r[5])), bg = make_float3(r[8], r[9], r[10]);
    float half = r[6], tz = r[7];
    float3 C, Tc; float Tr, d0; int id0;
    if (ns <= 1) fl_trace(O, D, half, tz, bg, C, Tc, Tr, d0, id0);
    else {
        float3 e1 = hnorm(hcross(D, fabsf(D.z) < 0.9f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f))), e2 = hcross(D, e1);
        // ns > 1 (11-12): the sub-rays [a0, a1) of this launch ADDED to out (the caller zeroes it, splits the ns x ns sub-rays over
        // launches so none runs long, and divides by ns^2 at the end); d0 the least
        float* oo = out + (size_t)i * 8;
        C = make_float3(oo[0], oo[1], oo[2]); Tc = make_float3(oo[3], oo[4], oo[5]); Tr = oo[6]; d0 = oo[7]; float th = tanf(half);
        for (int ab = a0; ab < a1 && ab < ns * ns; ++ab) { int a = ab / ns, b = ab - a * ns;
            unsigned s1 = fl_hash((unsigned)i * 7919u + (unsigned)(a * ns + b) * 131u + 7u), s2 = fl_hash(s1 + 3u);
            float u = 2.f * (a + fl_u01(s1)) / ns - 1.f, v = 2.f * (b + fl_u01(s2)) / ns - 1.f;    // a square of the cone's width (its
            float3 Ds = hnorm(D + (e1 * u + e2 * v) * th);                                        // footprint on the pixel's grid)
            float3 c_, t_; float tr_, d_; int id_; fl_trace(O, Ds, 1e-7f, tz, bg, c_, t_, tr_, d_, id_);
            C = C + c_; Tc = Tc + t_; Tr += tr_; d0 = fminf(d0, d_);
        }
    }
    float* o = out + (size_t)i * 8; o[0] = C.x; o[1] = C.y; o[2] = C.z; o[3] = Tc.x; o[4] = Tc.y; o[5] = Tc.z; o[6] = Tr; o[7] = d0;
}
// the key's transmittance through the hair from a point (the hair's shadow on what lies under it); pen widens each fibre's
// footprint to the shadow's cone (a fibre thinner than the penumbra casts a soft partial shadow, not an aliased hard one)
__device__ float fl_Tr(float3 P, float3 Ldir, float pen, float tmax) {
    HitList8 L; fl_hits(P, Ldir, pen, tmax, L); float Tr = 1.f;
    for (int j = 0; j < L.n; ++j) Tr *= 1.f - L.a[j];
    return Tr;
}
// per still pixel: the key's transmittance through the hair from the pixel's surface point (1: unshaded)
extern "C" __global__ void fl_skin_shade(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                         const float* zbuf, float pen_min, float* Ts, int ty0) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = ty0 + blockIdx.y * blockDim.y + threadIdx.y; if (px >= W_ || py >= H_) return;
    int o = py * W_ + px; float z = zbuf[o]; if (!(z < 1e3f)) { Ts[o] = 1.f; return; }
    float sx = ((px + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (py + 0.5f)) * pixang;
    float3 D = hnorm(fwd + right * sx + up * sy); float3 P = cam + D * (z - 2e-6f);
    float pen = fmaxf(pen_min, 0.5f * pixang * z / 3e-3f);
    Ts[o] = fl_Tr(P, HSUN, pen, 0.05f);
}

// hair2 11-10: the hair's occlusion of HER SKIN (the scalp under the hair, the part, the forehead edge): per still pixel the
// coloured transmittance through the hair toward the key (wsun) and over six sky directions (up, four at 50 deg from up, and
// back toward the camera: the room), each through fl_Trc (every fibre crossed, its TT lobe passing light on, coloured)
extern "C" __global__ void fl_skin_occ(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                       const float* zbuf, float wsun, float* Ts, int ty0) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = ty0 + blockIdx.y * blockDim.y + threadIdx.y; if (px >= W_ || py >= H_) return;
    int o = py * W_ + px; float z = zbuf[o];
    if (!(z < 1e3f)) { Ts[3*o] = 1.f; Ts[3*o+1] = 1.f; Ts[3*o+2] = 1.f; return; }
    float sx = ((px + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (py + 0.5f)) * pixang;
    float3 D = hnorm(fwd + right * sx + up * sy); float3 P = cam + D * (z - 2e-4f);
    float3 ts = fl_Trc(P, HSUN, 0.00465f + 0.02f, 0.06f);
    const float c = 0.6428f, s = 0.7660f;
    float3 dirs[6] = {make_float3(0.f, 0.f, 1.f), make_float3(s, 0.f, c), make_float3(-s, 0.f, c), make_float3(0.f, s, c),
                      make_float3(0.f, -s, c), make_float3(-D.x, -D.y, -D.z)};
    float3 sk = make_float3(0.f, 0.f, 0.f);
    for (int k = 0; k < 6; ++k) sk = sk + fl_Trc(P, hnorm(dirs[k]), 0.12f, 0.06f);
    sk = sk * (1.f / 6.f);
    float3 r = ts * wsun + sk * (1.f - wsun);
    Ts[3*o] = r.x; Ts[3*o+1] = r.y; Ts[3*o+2] = r.z;
}

// hair-shadow lane 11-11: the light reaching a point through the hair from a LIGHT OF SIZE (a cap of angular radius cr about
// direction c), integrated with ns x ns stratified rays over the cap, each ray EXACT (pen 1e-7: every fibre it crosses is found,
// since the span grid lists a span in every cell within its radius + 0.1 mm, and covers the ray fully). The point is jittered
// over the pixel's footprint on the surface (jit: +-0.5 px along right / up at that depth) with the same strata, so the
// shadow is anti-aliased at the pixel and its softness comes only from the light's size.
// Why: one cone of the light's radius through the span grid (fl_Trc with pen = cr) visits only the cells on the cone's centre
// line, so a fibre 1 mm off the axis but inside a 6 mm footprint is never seen and a fibre on the axis is weighted by the
// whole footprint: under a 12 deg softbox the face hair's shadow came out as sharp as the 0.5 mm cells, and the overlap of
// many lashes in that thin tube read as a long solid band (streaks, 11-11).
__device__ float3 fl_cap_T(float3 cam, float3 D0, float3 jr, float3 ju, float z, float3 c, float cr, int ns, unsigned seed, float tmax, float spen = 1e-7f) {
    float3 e1 = hnorm(hcross(c, fabsf(c.z) < 0.9f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f))), e2 = hcross(c, e1);
    float ck = cosf(cr), rot = fl_u01(fl_hash(seed ^ 0x9e3779b9u)); float3 acc = make_float3(0.f, 0.f, 0.f);
    for (int a = 0; a < ns; ++a) for (int b = 0; b < ns; ++b) {
        unsigned s1 = fl_hash(seed + (unsigned)a * 131u + (unsigned)b * 7919u + 1u), s2 = fl_hash(s1 + 3u), s3 = fl_hash(s2 + 5u), s4 = fl_hash(s3 + 7u);
        float u = (a + fl_u01(s1)) / ns, v = (b + fl_u01(s2)) / ns;
        float3 D = hnorm(D0 + jr * (fl_u01(s3) - 0.5f) + ju * (fl_u01(s4) - 0.5f)); float3 P = cam + D * (z - 2e-4f);
        float ct = 1.f - u * (1.f - ck), st = sqrtf(fmaxf(1.f - ct * ct, 0.f)), ph = 6.2831853f * (v + rot);
        float3 L = hnorm(c * ct + (e1 * cosf(ph) + e2 * sinf(ph)) * st);
        acc = acc + fl_Trc(P, L, spen, tmax);
    }
    return acc * (1.f / (float)(ns * ns));
}
// as fl_skin_occ2 (the key's share wsun, the six sky directions the rest), each light integrated by fl_cap_T: the key a cap of
// radius key_r (ns_key^2 rays), each sky direction a cap of radius sky_r (ns_sky^2 rays). Short hair (brows, lashes) only: the
// exact rays cost ns^2 walks, cheap through the face hair's sparse 0.5 mm grid, not through a scalp mass. key_spen / sky_spen: each
// sub-ray's own cone half-angle (1e-7 = exact; ~ the stratum's size = a soft sub-cone, exact only while its footprint stays under the cells)
extern "C" __global__ void fl_skin_occ3(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                        const float* zbuf, float wsun, float key_r, int ns_key, float sky_r, int ns_sky, float* Ts, int ty0,
                                        float key_spen, float sky_spen) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = ty0 + blockIdx.y * blockDim.y + threadIdx.y; if (px >= W_ || py >= H_) return;
    int o = py * W_ + px; float z = zbuf[o];
    if (!(z < 1e3f)) { Ts[3*o] = 1.f; Ts[3*o+1] = 1.f; Ts[3*o+2] = 1.f; return; }
    float sx = ((px + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (py + 0.5f)) * pixang;
    float3 D0 = fwd + right * sx + up * sy; float3 jr = right * pixang, ju = up * pixang; float3 D = hnorm(D0);
    unsigned seed = fl_hash((unsigned)o * 9781u + 17u);
    float3 ts = fl_cap_T(cam, D0, jr, ju, z, HSUN, key_r, ns_key, seed, 0.06f, key_spen);
    const float c = 0.6428f, s = 0.7660f;
    float3 dirs[6] = {make_float3(0.f, 0.f, 1.f), make_float3(s, 0.f, c), make_float3(-s, 0.f, c), make_float3(0.f, s, c),
                      make_float3(0.f, -s, c), make_float3(-D.x, -D.y, -D.z)};
    float3 sk = make_float3(0.f, 0.f, 0.f);
    for (int k = 0; k < 6; ++k) sk = sk + fl_cap_T(cam, D0, jr, ju, z, hnorm(dirs[k]), sky_r, ns_sky, fl_hash(seed + 1000003u * (unsigned)(k + 1)), 0.06f, sky_spen);
    sk = sk * (1.f / 6.f);
    float3 r = ts * wsun + sk * (1.f - wsun);
    Ts[3*o] = r.x; Ts[3*o+1] = r.y; Ts[3*o+2] = r.z;
}

// hair2 11-10 night: as fl_skin_occ, with the key's angular radius (rad): a softbox key casts a soft hair shadow on her skin
extern "C" __global__ void fl_skin_occ2(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                        const float* zbuf, float wsun, float pen_sun, float* Ts, int ty0) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = ty0 + blockIdx.y * blockDim.y + threadIdx.y; if (px >= W_ || py >= H_) return;
    int o = py * W_ + px; float z = zbuf[o];
    if (!(z < 1e3f)) { Ts[3*o] = 1.f; Ts[3*o+1] = 1.f; Ts[3*o+2] = 1.f; return; }
    float sx = ((px + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (py + 0.5f)) * pixang;
    float3 D = hnorm(fwd + right * sx + up * sy); float3 P = cam + D * (z - 2e-4f);
    float3 ts = fl_Trc(P, HSUN, pen_sun, 0.06f);
    const float c = 0.6428f, s = 0.7660f;
    float3 dirs[6] = {make_float3(0.f, 0.f, 1.f), make_float3(s, 0.f, c), make_float3(-s, 0.f, c), make_float3(0.f, s, c),
                      make_float3(0.f, -s, c), make_float3(-D.x, -D.y, -D.z)};
    float3 sk = make_float3(0.f, 0.f, 0.f);
    for (int k = 0; k < 6; ++k) sk = sk + fl_Trc(P, hnorm(dirs[k]), 0.12f, 0.06f);
    sk = sk * (1.f / 6.f);
    float3 r = ts * wsun + sk * (1.f - wsun);
    Ts[3*o] = r.x; Ts[3*o+1] = r.y; Ts[3*o+2] = r.z;
}

// ======================================================================================================================
// THE HAIR AS A MEASURED ABSORBING MEDIUM, FOR THE FAR PART OF A SOFT SHADOW RAY (hair-shadow lane 11-11, for
// the scalp hair's own shadow on her skin). One cone of the light's size walking the span grid sees only the cells on its
// centre line, so once its footprint is wider than a cell it misses most of the fibres inside it and weights the rest by the
// whole footprint: the scalp's shadow came out as hard black masses with isolated black specks (measured: probe scalp_t34).
// Here the shadow ray is split where its footprint reaches FMED_NEAR of a cell: before that the fibres themselves (fl_Trc,
// exact enough while the footprint stays inside the cells it visits: the fine strand shadows at the hairline); after it the
// fibres' own extinction gathered into the span grid's cells (each span's projected width x length x (1 - its TT pass) per
// cell volume, for the light's direction exactly, and orientation-averaged for the sky), mip-mapped, and read at the level
// whose cell matches the footprint: the cone's integral over every fibre inside it. Built from the fibres, not a look.
__constant__ float* FMED;          // levels packed: per cell rgb extinction (1/m) along HSUN, then the same orientation-averaged
__constant__ int FMEDL;            // number of levels
__constant__ int3 FMEDDIM[6];      // per level dims (level 0 = the span grid's)
__constant__ long long FMEDOFF[18];// per level offsets (floats): [0..5] the key set, [6..11] the isotropic set, [12..17] the isotropic FORWARD set (fs-store-5: 1 - (TT + 0.293 R + 0.15 TRT), as fl_Trc's fwd)
__constant__ float FMED_NEAR;      // the near part ends where the footprint radius reaches this x the cell size
// level 0: every span sampled at ~1/4 cell; each sample adds its projected area x (1 - TT) / cell volume to its cell (atomics)
// fs-store-6 11-13: the sums are FIXED POINT (unsigned, 1 / FL_MEDQ per unit, then fl_med_fix converts in place): integer
// atomics add in any order to the same total, so a frame's medium is bit-identical run to run (float atomics left 101 px of
// a 1080p frame differing by <= 0.1 % between two frames of one scene, hs1 f2 vs f5)
#define FL_MEDQ 16384.f
__device__ __forceinline__ void fl_madd(float* m, float v) { atomicAdd((unsigned int*)m, (unsigned int)fminf(rintf(fmaxf(v, 0.f) * FL_MEDQ), 4.0e9f)); }
extern "C" __global__ void fl_med_fix(long long n, float* med) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    unsigned int u = ((const unsigned int*)med)[i]; med[i] = (float)u * (1.f / FL_MEDQ);
}
extern "C" __global__ void fl_med_build(int nspan, float* med) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x; if (sid >= nspan) return;
    int i = sid / FSPN, k = sid - i * FSPN; Seg4 q; SecF sf; fl_span(i, k, q, sf);
    float ch = hlen(q.p2 - q.p1); int m = max(2, (int)ceilf(ch / (0.25f * FGH)) + 1);
    float V = FGH * FGH * FGH; long long n0 = (long long)FGDIM.x * FGDIM.y * FGDIM.z;
    for (int j = 0; j < m; ++j) {
        float u = ((float)j + 0.5f) / (float)m; float3 pp = cr_eval(q, u), T = hnorm(cr_deriv(q, u));
        float ds = ch / (float)m; float r = frad(i, k, u); float3 sg = fl_sigma(i, k, u);
        int cx = (int)floorf((pp.x - FGORG.x) / FGH), cy = (int)floorf((pp.y - FGORG.y) / FGH), cz = (int)floorf((pp.z - FGORG.z) / FGH);
        if (cx < 0 || cy < 0 || cz < 0 || cx >= FGDIM.x || cy >= FGDIM.y || cz >= FGDIM.z) continue;
        long long c = ((long long)cx * FGDIM.y + cy) * FGDIM.z + cz;
        float cw = hdot(HSUN, T); float sw = sqrtf(fmaxf(1.f - cw * cw, 0.f));
        Ap4 A = hair_ap(0.f, cw, sg); float3 a1 = A.a[1];
        float e = 2.f * r * ds / V;
        fl_madd(&med[3 * c], e * sw * (1.f - a1.x)); fl_madd(&med[3 * c + 1], e * sw * (1.f - a1.y)); fl_madd(&med[3 * c + 2], e * sw * (1.f - a1.z));
        Ap4 B = hair_ap(0.f, 0.f, sg); float3 b1 = B.a[1];                                     // the sky: |T x w| averaged = pi / 4
        long long ci = 3 * (n0 + c);
        fl_madd(&med[ci], e * 0.7853982f * (1.f - b1.x)); fl_madd(&med[ci + 1], e * 0.7853982f * (1.f - b1.y)); fl_madd(&med[ci + 2], e * 0.7853982f * (1.f - b1.z));
        float3 bf = B.a[1] + B.a[0] * 0.293f + B.a[2] * 0.15f; long long cf = 3 * (2 * n0 + c);    // fs-store-5: the forward set (level 0 after the two above)
        fl_madd(&med[cf], e * 0.7853982f * fmaxf(1.f - bf.x, 0.f)); fl_madd(&med[cf + 1], e * 0.7853982f * fmaxf(1.f - bf.y, 0.f)); fl_madd(&med[cf + 2], e * 0.7853982f * fmaxf(1.f - bf.z, 0.f));
    }
}
// the next level: the mean of each 2 x 2 x 2 block (cells past the edge count as empty)
extern "C" __global__ void fl_med_down(const float* src, int3 sd, float* dst, int3 dd) {
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long nd = (long long)dd.x * dd.y * dd.z; if (c >= nd) return;
    int z = (int)(c % dd.z), y = (int)((c / dd.z) % dd.y), x = (int)(c / ((long long)dd.y * dd.z));
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int e = 0; e < 2; ++e) {
        int X = 2 * x + a, Y = 2 * y + b, Z = 2 * z + e; if (X >= sd.x || Y >= sd.y || Z >= sd.z) continue;
        long long s = ((long long)X * sd.y + Y) * sd.z + Z; s0 += src[3 * s]; s1 += src[3 * s + 1]; s2 += src[3 * s + 2];
    }
    dst[3 * c] = s0 * 0.125f; dst[3 * c + 1] = s1 * 0.125f; dst[3 * c + 2] = s2 * 0.125f;
}
__device__ __forceinline__ float3 fl_med_at(int set, int l, float3 p) {     // trilinear, cell centres; outside = empty
    float h = FGH * (float)(1 << l); int3 d = FMEDDIM[l]; const float* M = FMED + FMEDOFF[set * 6 + l];
    float fx = (p.x - FGORG.x) / h - 0.5f, fy = (p.y - FGORG.y) / h - 0.5f, fz = (p.z - FGORG.z) / h - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy), z0 = (int)floorf(fz); float ax = fx - x0, ay = fy - y0, az = fz - z0;
    float3 c = make_float3(0.f, 0.f, 0.f);
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int e = 0; e < 2; ++e) {
        int X = x0 + a, Y = y0 + b, Z = z0 + e; if (X < 0 || Y < 0 || Z < 0 || X >= d.x || Y >= d.y || Z >= d.z) continue;
        float w = (a ? ax : 1.f - ax) * (b ? ay : 1.f - ay) * (e ? az : 1.f - az); long long s = 3 * (((long long)X * d.y + Y) * d.z + Z);
        c = c + make_float3(M[s], M[s + 1], M[s + 2]) * w;
    }
    return c;
}
// the light through the hair from P along w for a cone of half-angle pen: the fibres themselves while the footprint stays within
// FMED_NEAR of a cell, then the medium at the footprint's level (set 0: the key's direction, 1: orientation-averaged)
__device__ float3 fl_Tmed(float3 P, float3 w, float pen, float tmax, int set) {
    float tn = fminf(tmax, FMED_NEAR * FGH / fmaxf(pen, 1e-6f));
    float3 Tn = fl_Trc(P, w, pen, tn);
    if (tn >= tmax || fmaxf(Tn.x, fmaxf(Tn.y, Tn.z)) < 2e-3f) return Tn;
    float3 lo = FGORG, hi = FGORG + make_float3(FGDIM.x * FGH, FGDIM.y * FGH, FGDIM.z * FGH);
    float t0 = tn, t1 = tmax; float o[3] = {P.x, P.y, P.z}, d[3] = {w.x, w.y, w.z}, l[3] = {lo.x, lo.y, lo.z}, h[3] = {hi.x, hi.y, hi.z};
    for (int a = 0; a < 3; ++a) { float inv = 1.f / (fabsf(d[a]) > 1e-12f ? d[a] : 1e-12f);
        float ta = (l[a] - o[a]) * inv, tb = (h[a] - o[a]) * inv; if (ta > tb) { float s_ = ta; ta = tb; tb = s_; }
        t0 = fmaxf(t0, ta); t1 = fminf(t1, tb); }
    float3 tau = make_float3(0.f, 0.f, 0.f); float t = t0;
    for (int it = 0; it < 512 && t < t1; ++it) {
        float fp = 2.f * pen * t; float lv = log2f(fmaxf(fp / FGH, 1.f)); int lq = min((int)lv, FMEDL - 1);
        float hl = FGH * (float)(1 << lq); float ds = fminf(0.5f * hl, t1 - t);
        float3 s = fl_med_at(set, lq, P + w * (t + 0.5f * ds));
        tau = tau + s * ds; t += ds;
        if (fminf(tau.x, fminf(tau.y, tau.z)) > 9.f) break;
    }
    return make_float3(Tn.x * expf(-tau.x), Tn.y * expf(-tau.y), Tn.z * expf(-tau.z));
}
// as fl_skin_occ2 (the key's share wsun, the six sky directions the rest), each through fl_Tmed (the scalp's mass)
extern "C" __global__ void fl_skin_occ4(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                        const float* zbuf, float wsun, float pen_sun, float* Ts, int ty0) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = ty0 + blockIdx.y * blockDim.y + threadIdx.y; if (px >= W_ || py >= H_) return;
    int o = py * W_ + px; float z = zbuf[o];
    if (!(z < 1e3f)) { Ts[3*o] = 1.f; Ts[3*o+1] = 1.f; Ts[3*o+2] = 1.f; return; }
    float sx = ((px + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (py + 0.5f)) * pixang;
    float3 D = hnorm(fwd + right * sx + up * sy); float3 P = cam + D * (z - 2e-4f);
    float3 ts = fl_Tmed(P, HSUN, pen_sun, 0.06f, 0);
    const float c = 0.6428f, s = 0.7660f;
    float3 dirs[6] = {make_float3(0.f, 0.f, 1.f), make_float3(s, 0.f, c), make_float3(-s, 0.f, c), make_float3(0.f, s, c),
                      make_float3(0.f, -s, c), make_float3(-D.x, -D.y, -D.z)};
    float3 sk = make_float3(0.f, 0.f, 0.f);
    for (int k = 0; k < 6; ++k) sk = sk + fl_Tmed(P, hnorm(dirs[k]), 0.12f, 0.06f, 1);
    sk = sk * (1.f / 6.f);
    float3 r = ts * wsun + sk * (1.f - wsun);
    Ts[3*o] = r.x; Ts[3*o+1] = r.y; Ts[3*o+2] = r.z;
}

// ======================================================================================================================
// THE SCALP'S SHADOW ON HER SKIN, EXACT AND LEAN (hair-shadow lane 11-11). The key's disc and the six sky
// caps sampled with EXACT rays (as fl_skin_occ3, measured against the 144-ray reference: total shadow within 0.03 %), in
// passes: each launch adds the samples [s0, s1) of a per-pixel low-discrepancy sequence (R2, rotated per pixel) and keeps the
// running sums and the spread of what the rays saw. After the pilot pass a pixel whose rays all agree (lit, or under the mass)
// is done; the rest take the remaining passes. Short launches (a few rays each), so no launch runs long.
// acc: per pixel key rgb sum, sky rgb sum, key lum min / max, sky lum min / max, key n, sky n (12 floats)
__device__ __forceinline__ float fl_frac(float x) { return x - floorf(x); }
extern "C" __global__ void fl_occ_pass(float3 cam, float3 fwd, float3 right, float3 up, float pixang, int W_, int H_,
                                       const float* zbuf, float key_r, float sky_r, int k0, int k1, int q0, int q1, float conv, int kp,
                                       float* acc, int ty0) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = ty0 + blockIdx.y * blockDim.y + threadIdx.y; if (px >= W_ || py >= H_) return;
    int o = py * W_ + px; float z = zbuf[o]; if (!(z < 1e3f)) return;
    float* a = acc + (size_t)o * 12;
    if (k0 >= kp && a[7] - a[6] < conv && a[9] - a[8] < conv) return;         // past the pilot (kp key rays) and its rays agreed: done
    float sx = ((px + 0.5f) - 0.5f * W_) * pixang, sy = (0.5f * H_ - (py + 0.5f)) * pixang;
    float3 D0 = fwd + right * sx + up * sy; float3 jr = right * pixang, ju = up * pixang; float3 D = hnorm(D0);
    unsigned seed = fl_hash((unsigned)o * 9781u + 17u);
    float r1 = fl_u01(seed), r2 = fl_u01(fl_hash(seed + 1u)), r3 = fl_u01(fl_hash(seed + 2u)), r4 = fl_u01(fl_hash(seed + 3u));
    const float A1 = 0.7548776662f, A2 = 0.5698402910f;                       // R2 (the plastic number)
    // the key
    float3 e1 = hnorm(hcross(HSUN, fabsf(HSUN.z) < 0.9f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f))), e2 = hcross(HSUN, e1);
    float ck = cosf(key_r);
    if (FDOM) {                                                                  // fs-store-6: the key from the map, once
        if (k0 == 0 && k1 > 0) { float3 t = fl_Tdom(cam + D * (z - 2e-4f), 0.f, 0.06f); float l = 0.2126f * t.x + 0.7152f * t.y + 0.0722f * t.z;
            a[0] += t.x; a[1] += t.y; a[2] += t.z; a[6] = fminf(a[6], l); a[7] = fmaxf(a[7], l); a[10] += 1.f; }
    } else
    for (int n = k0; n < k1; ++n) {
        float u = fl_frac(r1 + n * A1), v = fl_frac(r2 + n * A2), jx = fl_frac(r3 + n * A1) - 0.5f, jy = fl_frac(r4 + n * A2) - 0.5f;
        float3 Dj = hnorm(D0 + jr * jx + ju * jy); float3 P = cam + Dj * (z - 2e-4f);
        float ct = 1.f - u * (1.f - ck), st = sqrtf(fmaxf(1.f - ct * ct, 0.f)), ph = 6.2831853f * v;
        float3 L = hnorm(HSUN * ct + (e1 * cosf(ph) + e2 * sinf(ph)) * st);
        float3 t = fl_Trc(P, L, 1e-7f, 0.06f); float l = 0.2126f * t.x + 0.7152f * t.y + 0.0722f * t.z;
        a[0] += t.x; a[1] += t.y; a[2] += t.z; a[6] = fminf(a[6], l); a[7] = fmaxf(a[7], l); a[10] += 1.f;
    }
    // the sky: sample n goes to direction n % 6 (each a cap of radius sky_r)
    const float c = 0.6428f, s = 0.7660f;
    float3 dirs[6] = {make_float3(0.f, 0.f, 1.f), make_float3(s, 0.f, c), make_float3(-s, 0.f, c), make_float3(0.f, s, c),
                      make_float3(0.f, -s, c), make_float3(-D.x, -D.y, -D.z)};
    float cs = cosf(sky_r);
    for (int n = q0; n < q1; ++n) {
        int kd = n % 6, m = n / 6; float3 cd = hnorm(dirs[kd]);
        float3 f1 = hnorm(hcross(cd, fabsf(cd.z) < 0.9f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f))), f2 = hcross(cd, f1);
        float u = fl_frac(r2 + m * A1 + kd * 0.37f), v = fl_frac(r1 + m * A2 + kd * 0.61f), jx = fl_frac(r4 + n * A1) - 0.5f, jy = fl_frac(r3 + n * A2) - 0.5f;
        float3 Dj = hnorm(D0 + jr * jx + ju * jy); float3 P = cam + Dj * (z - 2e-4f);
        float ct = 1.f - u * (1.f - cs), st = sqrtf(fmaxf(1.f - ct * ct, 0.f)), ph = 6.2831853f * v;
        float3 L = hnorm(cd * ct + (f1 * cosf(ph) + f2 * sinf(ph)) * st);
        float3 t = fl_Trc(P, L, 1e-7f, 0.06f); float l = 0.2126f * t.x + 0.7152f * t.y + 0.0722f * t.z;
        a[3] += t.x; a[4] += t.y; a[5] += t.z; a[8] = fminf(a[8], l); a[9] = fmaxf(a[9], l); a[11] += 1.f;
    }
}
// the passes' sums -> the occlusion (key share wsun, sky the rest); pixels off her surface stay 1
extern "C" __global__ void fl_occ_finish(int n, const float* zbuf, const float* acc, float wsun, float* Ts) {
    int o = blockIdx.x * blockDim.x + threadIdx.x; if (o >= n) return;
    if (!(zbuf[o] < 1e3f)) { Ts[3*o] = 1.f; Ts[3*o+1] = 1.f; Ts[3*o+2] = 1.f; return; }
    const float* a = acc + (size_t)o * 12; float nk = fmaxf(a[10], 1.f), ns = fmaxf(a[11], 1.f);
    for (int c = 0; c < 3; ++c) Ts[3*o+c] = wsun * a[c] / nk + (1.f - wsun) * a[3 + c] / ns;
}

// ===== fs-store-6 11-13: the key's light through the hair, built once a frame (FL_KEYDOM) =====
// Columns FDOMO.z wide across the key. Pass 0: each column's first fibre depth d0 (atomicMin on the float's bits: s >= 0).
// Pass 1: each fibre sample adds its projected area across the key over the column's area x (1 - its TT lobe) (fl_med_build's
// key set) to the layer of the column that holds it, the layers FDOMZ after d0 (fine at the lit face, doubling with depth);
// fixed-point sums, so a frame's map is bit-identical run to run. The samples are spaced half a column across the key.
#define FDOM_Q 4096.f
__device__ __forceinline__ void fl_uadd(unsigned int* m, float v) { atomicAdd(m, (unsigned int)fminf(rintf(fmaxf(v, 0.f) * FDOM_Q), 4.0e9f)); }
__device__ __forceinline__ int fl_dom_col(float3 p, float& s) {
    float u = (hdot(p, FDOMA) - FDOMO.x) / FDOMO.z, v = (hdot(p, FDOMB) - FDOMO.y) / FDOMO.z; s = FDOMO.w - hdot(p, HSUN);
    int iu = (int)floorf(u), iv = (int)floorf(v);
    if (!(u >= 0.f) || !(v >= 0.f) || iu >= FDOMN.x || iv >= FDOMN.y) return -1;
    return iv * FDOMN.x + iu;
}
extern "C" __global__ void fl_dom_build(int s0, int nspan, int pass) {
    int sid = s0 + blockIdx.x * blockDim.x + threadIdx.x; if (sid >= nspan) return;
    int i = sid / FSPN, k = sid - i * FSPN; Seg4 q; SecF sf; fl_span(i, k, q, sf);
    float3 ch3 = q.p2 - q.p1; float ch = hlen(ch3); float chp = hlen(ch3 - HSUN * hdot(ch3, HSUN));
    int m = min(max(2, (int)ceilf(chp / (0.5f * FDOMO.z)) + 1), 2048); float A = FDOMO.z * FDOMO.z;
    for (int j = 0; j < m; ++j) {
        float u = ((float)j + 0.5f) / (float)m; float3 pp = cr_eval(q, u); float s;
        int col = fl_dom_col(pp, s); if (col < 0) continue;
        if (pass == 0) { atomicMin((int*)&FDOMD0[col], __float_as_int(fmaxf(s, 0.f))); continue; }
        float3 T = hnorm(cr_deriv(q, u)); float cw = hdot(HSUN, T); float sw = sqrtf(fmaxf(1.f - cw * cw, 0.f));
        float r = frad(i, k, u); Ap4 Ap = hair_ap(0.f, cw, fl_sigma(i, k, u)); float3 a1 = Ap.a[1];
        float e = 2.f * r * (ch / (float)m) * sw / A;
        float er = e * fmaxf(1.f - a1.x, 0.f), eg_ = e * fmaxf(1.f - a1.y, 0.f), eb = e * fmaxf(1.f - a1.z, 0.f);
        float dd = s - FDOMD0[col]; int kb = 0; while (kb < FDOM_KL - 2 && dd >= FDOMZ[kb + 1]) ++kb;
        unsigned int* a = FDOMACC + (size_t)col * (FDOM_KL + 2);
        fl_uadd(&a[kb], 0.2126f * er + 0.7152f * eg_ + 0.0722f * eb);
        fl_uadd(&a[FDOM_KL - 1], er); fl_uadd(&a[FDOM_KL], eg_); fl_uadd(&a[FDOM_KL + 1], eb);
    }
}
extern "C" __global__ void fl_dom_scan(int ncol) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ncol) return;
    const unsigned int* a = FDOMACC + (size_t)c * (FDOM_KL + 2); float* T = FDOMT + (size_t)c * FDOM_KL; float* C = FDOMC + (size_t)c * 4;
    float acc = 0.f; T[0] = 0.f;
    for (int k = 0; k < FDOM_KL - 1; ++k) { acc += (float)a[k] * (1.f / FDOM_Q); T[k + 1] = acc; }
    float g = fmaxf(acc, 1e-9f), q = 1.f / FDOM_Q;
    C[0] = acc > 0.f ? (float)a[FDOM_KL - 1] * q / g : 1.f; C[1] = acc > 0.f ? (float)a[FDOM_KL] * q / g : 1.f;
    C[2] = acc > 0.f ? (float)a[FDOM_KL + 1] * q / g : 1.f; C[3] = 0.f;
}
__device__ __forceinline__ float fl_dom_tau(int col, float s) {
    float d0 = FDOMD0[col]; if (!(d0 < 1e29f)) return 0.f;
    float dd = s - d0; if (!(dd > 0.f)) return 0.f;
    const float* T = FDOMT + (size_t)col * FDOM_KL;
    if (dd >= FDOMZ[FDOM_KL - 1]) return T[FDOM_KL - 1];
    int k = 0; while (k < FDOM_KL - 2 && dd >= FDOMZ[k + 1]) ++k;
    return T[k] + (T[k + 1] - T[k]) * ((dd - FDOMZ[k]) / (FDOMZ[k + 1] - FDOMZ[k]));
}
// the key's light at P through the hair between depth s(P) - bias - tmax and s(P) - bias (bilinear over four columns)
__device__ float3 fl_Tdom(float3 P, float bias, float tmax) {
    float u = (hdot(P, FDOMA) - FDOMO.x) / FDOMO.z - 0.5f, v = (hdot(P, FDOMB) - FDOMO.y) / FDOMO.z - 0.5f;
    if (!(u > -1.f) || !(v > -1.f) || u > (float)FDOMN.x || v > (float)FDOMN.y) return make_float3(1.f, 1.f, 1.f);
    float s1 = FDOMO.w - hdot(P, HSUN) - bias, s0 = s1 - tmax;
    int u0 = (int)floorf(u), v0 = (int)floorf(v); float fu = u - (float)u0, fv = v - (float)v0;
    float tr = 0.f, tg = 0.f, tb = 0.f;
    for (int bb = 0; bb < 2; ++bb) for (int aa = 0; aa < 2; ++aa) {
        int iu = u0 + aa, iv = v0 + bb; if (iu < 0 || iv < 0 || iu >= FDOMN.x || iv >= FDOMN.y) continue;
        int col = iv * FDOMN.x + iu; float t = fl_dom_tau(col, s1) - fl_dom_tau(col, s0); if (!(t > 0.f)) continue;
        float w = (aa ? fu : 1.f - fu) * (bb ? fv : 1.f - fv); const float* C = FDOMC + (size_t)col * 4;
        tr += C[0] * t * w; tg += C[1] * t * w; tb += C[2] * t * w;
    }
    return make_float3(expf(-tr), expf(-tg), expf(-tb));
}
