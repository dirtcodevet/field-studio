"""View modes on the card (fs-shell, 2026-11-11; design chunk 9: matcap, cavity and curvature on the card, matcap at
full frame rate; the gap table: a peer's staple, so forms are judged the way sculptors do).

A drop-in for GpuScene.k_light with light_pass's exact arguments (fs-light's split view and look-dev cache use the same
slot): the G-buffer the render kernel traced (position, normal, object) is shaded by the mode instead of the review
light. Compiled once from trace.cu's own text plus the kernel below (its helpers: scene_val, env_bilin), so trace.cu is
not edited. Modes:
  'matcap'     a sculptor's matcap lit in VIEW space (it turns with the camera, as Blender's and ZBrush's do): clay,
               red wax or grey studio, computed analytically (no licensed images: P13); cavity shading on top
  'curvature'  the form's mean curvature from the field itself (the Laplacian of F at the hit, 6 samples at a set scale,
               default 1 mm or 2 lattice steps): convex warm, concave cool, flat grey
  'normal'     the world normal as colour (as the CPU's Normals)
Cavity: concave forms darkened by their curvature (exact from the field, not screen-space), convex edges a touch lighter;
its strength 0-1. The colours are written through the inverse of the resolve's ACES curve, so they show as designed.
Only the anti-aliasing passes shade (a view mode has no noise to converge): full frame rate. Behind the form: the CPU
view's neutral grey, not the sky (a form is judged against a plain ground).
The engine draws these modes on the card (ui/engine.py's card_modes); the CPU keeps its own Matcap / Normals as a
fallback ('curvature' falls back to the CPU's matcap)."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {'shaded': 0, 'matcap': 1, 'curvature': 2, 'normal': 3, 'silhouette': 4}
MATCAPS = {'clay': 0, 'red wax': 1, 'grey': 2}

SRC = r'''
// ---- fs-shell 11-11: view modes (gpu/viewmodes.py) ---------------------------------------------------------------------
// VM[0] mode (1 matcap, 2 curvature, 3 normals), VM[1] cavity strength 0-1, VM[2] the curvature colour scale (m: a
// radius of curvature this size reads full colour), VM[3] matcap kind (0 clay, 1 red wax, 2 grey), VM[4] the curvature
// sampling scale (m; at least 2 lattice steps); fs-shell 11-13: mode 4 = silhouette (VM[6] the ground, VM[7] the form:
// display values, flat)
__device__ float VM[16];

__device__ __forceinline__ float3 vm_cross(float3 a, float3 b) {
    return mk(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float vm_inv_aces(float y) {
    y = fminf(fmaxf(y, 0.0f), 0.985f);
    float a = 2.51f - 2.43f * y, b = 0.03f - 0.59f * y, c = -0.14f * y;
    return (-b + sqrtf(fmaxf(b * b - 4.0f * a * c, 0.0f))) / (2.0f * a);
}

__device__ float vm_curv(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                         float3 p, int lev, float e) {
    // the mean curvature (1/m) from the field's Laplacian (F a distance near the surface: lap F = 2 H)
    bool ok0;
    float f0 = scene_val(nvol, vft, vit, grid, ttab, sdf, p, lev, &ok0);
    float s = 0.0f;
    bool all = ok0;
    for (int a = 0; a < 3; a++) {
        float3 d = mk(a == 0 ? e : 0.0f, a == 1 ? e : 0.0f, a == 2 ? e : 0.0f);
        bool o1, o2;
        s += scene_val(nvol, vft, vit, grid, ttab, sdf, add(p, d), lev, &o1)
           + scene_val(nvol, vft, vit, grid, ttab, sdf, sub(p, d), lev, &o2);
        all = all && o1 && o2;
    }
    if (!all) return 0.0f;
    return 0.5f * (s - 6.0f * f0) / (e * e);
}

extern "C" __global__ void view_pass(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
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
    if (pass < 0) {                       // per-pixel progressive (as light_pass)
        pass = pcount[pi];
        if (pass >= target) return;
        pcount[pi] = pass + 1;
        nbox = 0;
    }
    float4 g1 = gn[pi];
    if (nbox > 0 && !rmask[pi]) return;
    if (pass >= aa_passes) return;        // nothing to converge: the anti-aliasing passes are the image
    float3 col;
    if ((int)(VM[0] + 0.5f) == 4) {       // fs-shell 11-13: the silhouette (flat: the form one value, the ground another)
        float v = g1.w < 0.0f ? VM[6] : VM[7];
        col = mk(vm_inv_aces(v), vm_inv_aces(v), vm_inv_aces(v));
    } else if (g1.w < 0.0f) {             // a neutral studio grey behind the form (as the CPU view's), not the sky
        float bg = 0.085f + 0.05f * (1.0f - (y + 0.5f) / (float)H);
        col = mk(vm_inv_aces(bg), vm_inv_aces(bg), vm_inv_aces(bg * 1.08f));
    } else {
        float4 g0 = gp[pi];
        int vol = (int)g1.w, lev = (int)((g1.w - vol) * 4.0f + 0.5f);
        float3 p = mk(g0.x, g0.y, g0.z), n = nrm(mk(g1.x, g1.y, g1.z));
        float hl = g0.w;
        int mode = (int)(VM[0] + 0.5f);
        float3 vd = nrm(sub(mk(L[17], L[18], L[19]), p));
        float3 up = mk(0.0f, 1.0f, 0.0f);
        float3 r = vm_cross(up, vd);
        if (dot3(r, r) < 1e-8f) r = mk(1.0f, 0.0f, 0.0f);
        r = nrm(r);
        float3 u = vm_cross(vd, r);
        float3 nv = mk(dot3(n, r), dot3(n, u), dot3(n, vd));            // the normal in view space
        float c3[3];
        if (mode == 3) {
            c3[0] = n.x * 0.5f + 0.5f; c3[1] = n.y * 0.5f + 0.5f; c3[2] = n.z * 0.5f + 0.5f;
        } else {
            float3 l1 = nrm(mk(-0.50f, 0.62f, 0.60f)), l2 = nrm(mk(0.65f, -0.15f, 0.75f));
            float d1 = fmaxf(dot3(nv, l1), 0.0f), d2 = fmaxf(dot3(nv, l2), 0.0f);
            float rim = powf(1.0f - fmaxf(nv.z, 0.0f), 3.0f);
            float3 hv = nrm(add(l1, mk(0.0f, 0.0f, 1.0f)));
            float nh = fmaxf(dot3(nv, hv), 0.0f);
            int kind = (int)(VM[3] + 0.5f);
            float base[3], spec, dif;
            if (mode == 2 || kind == 2) { base[0] = 0.46f; base[1] = 0.46f; base[2] = 0.48f; spec = 0.12f * powf(nh, 30.0f);
                                          dif = 0.20f + 0.65f * d1 + 0.18f * d2; }
            else if (kind == 1) { base[0] = 0.62f; base[1] = 0.13f; base[2] = 0.09f;
                                  float w = 0.5f * dot3(nv, l1) + 0.5f;
                                  dif = 0.22f + 0.78f * w * w + 0.12f * d2; spec = 0.30f * powf(nh, 48.0f); }
            else { base[0] = 0.66f; base[1] = 0.56f; base[2] = 0.48f; spec = 0.10f * powf(nh, 18.0f);
                   dif = 0.20f + 0.66f * d1 + 0.16f * d2; }
            for (int c = 0; c < 3; c++) c3[c] = base[c] * dif + spec + 0.18f * rim * base[c];
            float cav = VM[1];
            if (mode == 2 || cav > 0.0f) {
                float e = fmaxf(VM[4], 2.0f * hl);
                float Hm = vm_curv(nvol, vft, vit, grid, ttab, sdf, p, lev, e);
                float t = fminf(fmaxf(Hm * VM[2], -1.0f), 1.0f);
                if (mode == 2) {
                    float wa[3] = {0.95f, 0.55f, 0.28f}, co[3] = {0.25f, 0.50f, 0.95f};
                    for (int c = 0; c < 3; c++) {
                        float tint = t > 0.0f ? wa[c] : co[c];
                        float a = fabsf(t);
                        c3[c] = c3[c] * (1.0f - 0.75f * a) + tint * dif * 0.75f * a;
                    }
                } else {
                    float k = fmaxf(-t, 0.0f), kp = fmaxf(t, 0.0f);
                    for (int c = 0; c < 3; c++) c3[c] = c3[c] * (1.0f - 0.7f * cav * k) + 0.06f * cav * kp;
                }
            }
        }
        col = mk(vm_inv_aces(c3[0]), vm_inv_aces(c3[1]), vm_inv_aces(c3[2]));
    }
    float* a = accum + pi * 4;
    if (pass == 0) { a[0] = col.x; a[1] = col.y; a[2] = col.z; a[3] = 1.0f; }
    else { a[0] += col.x; a[1] += col.y; a[2] += col.z; a[3] += 1.0f; }
    aob[pi * 2] = 0.0f; aob[pi * 2 + 1] = 0.0f;            // no sky term in a view mode (resolve adds none)
    float* sc = skyc + pi * 4;
    sc[0] = sc[1] = sc[2] = 0.0f; sc[3] = 1.0f;
}

// ---- fs-shell 11-13: X-ray (Tier 0 part 4: adjustable transparency in the view types, to see structural layers
// through objects). A composite after the frame (Clay or Rendered): from each pixel's first
// hit (the frame's own colour, depth and object) the ray goes on through the volumes ONE BY ONE (each volume's own entry
// and exit, so a form inside another, a skull in a head, shows as its own layer, which the union's min would hide), every
// entry shaded as clay with a hue per object, composited front to back with alpha = (1 - xray) x the object's opacity
// (vft[v * VF + VF_OPA], 1 by default). Pixels without a hit, or whose front alpha is 1, keep the frame's colour.
// Blending in display-linear (the frame's colour un-encoded from sRGB), written back sRGB.
#define VF_OPA 47
#define XR_MAXV 48
// one volume's value at a world point (world units) and whether a brick holds it (*brick 1). A tile with no brick gives
// only a POSITIVE lower bound of |F| (trace.cu vol_f0: the store keeps no sign away from the surface), inside or
// outside alike, so a sign is read only where *brick is 1; elsewhere the value is a safe step, nothing more. Outside the
// volume's box: *brick 2 (truly outside)
__device__ __noinline__ float xr_f(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf, float3 pw,
                                   int* brick) {
    float3 p = xf_p(vf + VF_M, pw);
    if (p.x < vf[VF_LO] || p.y < vf[VF_LO + 1] || p.z < vf[VF_LO + 2] || p.x > vf[VF_HI] || p.y > vf[VF_HI + 1] || p.z > vf[VF_HI + 2]) {
        *brick = 2;
        return 1e30f;
    }
    int lev, emp, lm = 0;
    while (lm < NLEV - 1 && vi[VI_BPT + lm] == 0) lm++;
    float f = vol_f(vf, vi, grid, ttab, sdf, p, lm, &lev, &emp);
    *brick = lev >= 0 ? 1 : 0;
    return f / vf[VF_S];
}
__device__ float xr_hmin(const float* vf, const int* vi) {
    int lm = 0;
    while (lm < NLEV - 1 && vi[VI_BPT + lm] == 0) lm++;
    return 0.5f * vf[VF_H + lm] / vf[VF_S];
}
// from t (inside the volume) along the ray to where it leaves: the first point where a brick reads F > 0 (or the box
// ends); through brickless tiles (the solid's inside, no sign stored) the step is their |F| bound; at least half a cell
__device__ __noinline__ float xr_exit(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                         float3 o, float3 d, float t, float tmax) {
    float hm = xr_hmin(vf, vi);
    for (int i = 0; i < 1024 && t < tmax; i++) {
        int br;
        float f = xr_f(vf, vi, grid, ttab, sdf, add(o, mul(d, t)), &br);
        if (br == 2 || (br == 1 && f > 0.0f)) return t;
        t += fmaxf(fabsf(f), hm);
    }
    return tmax;
}
// the next entry of one volume beyond t (one instantiation of trace_vol: NVRTC inlines device functions, and two call
// sites of the tracer doubled the module's compile time)
__device__ __noinline__ float xr_entry(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                                       float3 o, float3 d, float t, float tmax, float pixang, float lodb, int steps) {
    int lev;
    float te = trace_vol(vf, vi, grid, ttab, sdf, o, d, t, tmax, pixang, lodb, &lev, steps);
    return te >= 0.0f ? te : 1e30f;
}
__device__ __noinline__ float3 xr_normal(const float* vf, const int* vi, const int* grid, const int* ttab, const __half* sdf,
                                         float3 p) {
    float3 n, g, pl = xf_p(vf + VF_M, p);
    int l0 = 0;
    while (l0 < NLEV - 1 && vi[VI_BPT + l0] == 0) l0++;
    if (c1_grad(vf, vi, grid, ttab, sdf, l0, pl, &g)) n = nrm(xf_d(vf + VF_MI, g));
    else n = vol_normal(vf, vi, grid, ttab, sdf, p, l0);
    return n;
}
__device__ __noinline__ void xr_clay(float3 n, float3 d, int v, int layer, float c3[3]) {
    float3 vd = mul(d, -1.0f);
    float3 up = mk(0.0f, 1.0f, 0.0f);
    float3 r = vm_cross(up, vd);
    if (dot3(r, r) < 1e-8f) r = mk(1.0f, 0.0f, 0.0f);
    r = nrm(r);
    float3 u = vm_cross(vd, r);
    float3 nv = mk(dot3(n, r), dot3(n, u), dot3(n, vd));
    float3 l1 = nrm(mk(-0.50f, 0.62f, 0.60f)), l2 = nrm(mk(0.65f, -0.15f, 0.75f));
    float d1 = fmaxf(dot3(nv, l1), 0.0f), d2 = fmaxf(dot3(nv, l2), 0.0f);
    float dif = 0.22f + 0.64f * d1 + 0.16f * d2;
    // (no view-angle rim term on a layer: 11-14, a patent precaution: QuantifiCare US 11,094,139 claims a rim glow
    // 1 - |N.V| drawn over another model; X-ray here is each object's own lit surface composited at its opacity)
    // a hue per object (golden-ratio steps), low saturation: the layers read apart without shouting
    float h = fmodf(0.61803f * (float)v + 0.08f, 1.0f) * 6.0f;
    float k[3] = {fabsf(h - 3.0f) - 1.0f, 2.0f - fabsf(h - 2.0f), 2.0f - fabsf(h - 4.0f)};
    float sat = 0.30f, dim = 1.0f - 0.06f * (float)min(layer, 5);
    for (int c = 0; c < 3; c++) {
        float hue = fminf(fmaxf(k[c], 0.0f), 1.0f);
        float base = (0.60f * (1.0f - sat) + 0.60f * sat * hue) * dim;
        c3[c] = base * dif;
    }
}
// fs-shell-4 11-14 (the known gap: "X-ray ignores section and box-hide clips beyond the first surface"): the frame's
// first hit honours the view's clips (trace.cu's render kernel: P[23] bit 1 = the section, the half n.p > d cut away;
// bit 2 = hidden boxes, P[24] of them, 6 planes each, inside where n.p <= d), but the layers behind it came from each
// volume's own entries and exits, which know nothing of them: a form in the cut-away half or inside a hidden box still
// showed through, and a form the clip cuts open showed its raw skin, not its cut face. The cause is in the walk, so the
// walk now carries the same rule: a point of the ray is KEPT when it lies in the section's kept span and in no hidden
// box (the clipped solid is each form intersected with the kept space). An entry is drawn only where kept; where the ray
// passes INTO kept space while inside a form, that form's cut face is drawn there, facing the ray, as the frame draws
// the first cut face. Without clips (P[23] = 0) none of this runs: the pixels are the same bit for bit.
#define XR_CEPS 2e-5f
__device__ bool xr_kept(const float* P, int clipf, float k0, float k1, float3 o, float3 d, float t) {
    if (t < k0 || t >= k1) return false;
    if (clipf & 2) {
        int nb = (int)(P[24] + 0.5f);
        float3 q = add(o, mul(d, t));
        for (int b = 0; b < nb; b++) {
            const float* bp = P + 25 + 24 * b;
            bool ins = true;
            for (int a = 0; a < 6 && ins; a++)
                if (bp[4 * a] * q.x + bp[4 * a + 1] * q.y + bp[4 * a + 2] * q.z - bp[4 * a + 3] > 0.0f) ins = false;
            if (ins) return false;
        }
    }
    return true;
}
// the next place beyond t where the ray crosses a clip boundary (the section plane, a hidden box's face), and that
// boundary's plane normal (*nout); 1e30 when none
__device__ float xr_next_clip(const float* P, int clipf, float k0, float k1, float3 sn, float3 o, float3 d, float t,
                              float3* nout) {
    float tn = 1e30f;
    if (clipf & 1) {
        if (k0 > t && k0 < tn) { tn = k0; *nout = sn; }
        if (k1 > t && k1 < tn) { tn = k1; *nout = sn; }
    }
    if (clipf & 2) {
        int nb = (int)(P[24] + 0.5f);
        for (int b = 0; b < nb; b++) {
            const float* bp = P + 25 + 24 * b;
            float ta = -1e30f, tb = 1e30f; int aa = 0, ab = 0; bool never = false;
            for (int a = 0; a < 6; a++) {
                float3 pn = mk(bp[4 * a], bp[4 * a + 1], bp[4 * a + 2]);
                float s0 = dot3(pn, o) - bp[4 * a + 3], dn = dot3(pn, d);
                if (fabsf(dn) < 1e-30f) { if (s0 > 0.0f) never = true; }
                else { float u = -s0 / dn; if (dn < 0.0f) { if (u > ta) { ta = u; aa = a; } } else if (u < tb) { tb = u; ab = a; } }
            }
            if (never || tb <= ta) continue;
            if (ta > t && ta < tn) { tn = ta; *nout = mk(bp[4 * aa], bp[4 * aa + 1], bp[4 * aa + 2]); }
            if (tb > t && tb < tn) { tn = tb; *nout = mk(bp[4 * ab], bp[4 * ab + 1], bp[4 * ab + 2]); }
        }
    }
    return tn;
}
extern "C" __global__ void xray_comp(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                     const float* P, int W, int H, int x0, int y0, int rw, int rh,
                                     const float* depth, const int* objid, const unsigned int* in, unsigned int* out,
                                     float xray, int max_layers) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= rw || py >= rh) return;
    int x = x0 + px, y = y0 + py;
    if (x >= W || y >= H) return;
    size_t pi = (size_t)y * W + x;
    unsigned int c0 = in[pi];
    out[pi] = c0;
    int v0 = objid[pi];
    if (v0 < 0 || v0 >= nvol || nvol > XR_MAXV) return;
    float a0 = (1.0f - xray) * fminf(fmaxf(vft[v0 * VF + VF_OPA], 0.0f), 1.0f);
    if (a0 >= 0.999f) return;
    // the pixel-centre ray, as trace_gbuf's pass 0 (perspective or orthographic)
    float sx = (2.0f * (x + 0.5f) / W - 1.0f) * P[12] * P[13], sy = (1.0f - 2.0f * (y + 0.5f) / H) * P[12];
    float3 o = mk(P[0], P[1], P[2]);
    float3 d = nrm(add(mk(P[3], P[4], P[5]), add(mul(mk(P[6], P[7], P[8]), sx), mul(mk(P[9], P[10], P[11]), sy))));
    float pixang = 2.0f * P[12] / H;
    if (P[17] > 0.0f) {
        float3 f = mk(P[3], P[4], P[5]);
        o = add(add(o, mul(add(mul(mk(P[6], P[7], P[8]), sx), mul(mk(P[9], P[10], P[11]), sy)), P[17])), mul(f, -P[18]));
        d = f;
        pixang = 2.0f * P[12] * P[17] / (H * (P[17] + P[18]));
    }
    float tmax = P[15];
    float t1 = depth[pi];
    // fs-shell-4 11-14: the view's clips on this ray (trace.cu's rule): the section's kept span [k0, k1), the hidden
    // boxes tested at a point (xr_kept); tcl the next clip boundary past the first hit (the frame drew the ones before)
    int clipf = (int)(P[23] + 0.5f);
    float k0 = 0.0f, k1 = 1e30f;
    float3 sn = mk(0.0f, 0.0f, 0.0f), ncl = mk(0.0f, 0.0f, 0.0f);
    if (clipf & 1) {
        sn = mk(P[19], P[20], P[21]);
        float s0 = dot3(sn, o) - P[22], dn = dot3(sn, d);
        if (s0 > 0.0f) k0 = dn < 0.0f ? -s0 / dn : 1e30f;
        else if (dn > 0.0f) k1 = -s0 / dn;
    }
    float tcl = clipf ? xr_next_clip(P, clipf, k0, k1, sn, o, d, t1 + XR_CEPS, &ncl) : 1e30f;
    float lin[3];
    {
        float e[3] = {(float)((c0 >> 16) & 255u) / 255.0f, (float)((c0 >> 8) & 255u) / 255.0f, (float)(c0 & 255u) / 255.0f};
        for (int c = 0; c < 3; c++) lin[c] = e[c] <= 0.04045f ? e[c] / 12.92f : powf((e[c] + 0.055f) / 1.055f, 2.4f);
    }
    float C[3] = {a0 * lin[0], a0 * lin[1], a0 * lin[2]};
    float T = 1.0f - a0;
    // per volume: whether the ray is inside it just past the first hit, and its next event (entry or exit). The state
    // comes from the volume's own entries and exits replayed from the camera (the camera outside every form, as the
    // frame's tracer has it), never from the field's sign: brickless tiles keep none (xr_f). The hit volume v0 is
    // entered at the first hit t1 itself (the frame's surface)
    unsigned long long inside = 0ull;
    float tev[XR_MAXV];
    for (int v = 0; v < nvol; v++) {
        const float* vf = vft + v * VF; const int* vi = vit + v * VI;
        tev[v] = 1e30f;
        if (vf[VF_VIS] < 0.5f) continue;
        float eps = xr_hmin(vf, vi);
        float tpast = t1 + 2.0f * eps;
        if (v == v0) {
            inside |= (1ull << v);
            tev[v] = xr_exit(vf, vi, grid, ttab, sdf, o, d, t1 + eps, tmax);
            continue;
        }
        bool in = false;
        float te = xr_entry(vf, vi, grid, ttab, sdf, o, d, 0.0f, tmax, pixang, P[16], (int)P[14]);
        for (int k = 0; k < 16 && te < tpast; k++) {     // events before the first hit (a form partly in front of it)
            if (!in) { in = true; te = xr_exit(vf, vi, grid, ttab, sdf, o, d, te + eps, tmax); }
            else { in = false; te = xr_entry(vf, vi, grid, ttab, sdf, o, d, te + eps, tmax, pixang, P[16], (int)P[14]); }
        }
        if (in) inside |= (1ull << v);
        tev[v] = te;
    }
    int layer = 1;
    for (int it = 0; it < 64 && T > 0.02f && layer <= max_layers; it++) {
        int vb = -1; float tb = 1e29f;
        for (int v = 0; v < nvol; v++) if (tev[v] < tb) { tb = tev[v]; vb = v; }
        if (tcl < tb) {                                   // fs-shell-4 11-14: a clip boundary comes first
            if (!xr_kept(P, clipf, k0, k1, o, d, tcl - XR_CEPS) && xr_kept(P, clipf, k0, k1, o, d, tcl + XR_CEPS)) {
                // into kept space inside forms: each form the ray is inside shows its cut face here
                float3 n = ncl;
                if (dot3(n, d) > 0.0f) n = mul(n, -1.0f);
                for (int v = 0; v < nvol && T > 0.02f && layer <= max_layers; v++) {
                    if (!(inside & (1ull << v))) continue;
                    float c3[3];
                    xr_clay(n, d, v, layer, c3);
                    float a = (1.0f - xray) * fminf(fmaxf(vft[v * VF + VF_OPA], 0.0f), 1.0f);
                    for (int c = 0; c < 3; c++) C[c] += T * a * c3[c];
                    T *= 1.0f - a;
                    layer++;
                }
            }
            tcl = xr_next_clip(P, clipf, k0, k1, sn, o, d, tcl + XR_CEPS, &ncl);
            continue;
        }
        if (vb < 0) break;
        const float* vf = vft + vb * VF; const int* vi = vit + vb * VI;
        float eps = xr_hmin(vf, vi);
        if (inside & (1ull << vb)) {                      // leaving vb: its next entry beyond
            inside &= ~(1ull << vb);
            tev[vb] = xr_entry(vf, vi, grid, ttab, sdf, o, d, tb + eps, tmax, pixang, P[16], (int)P[14]);
            continue;
        }
        // entering vb: a layer (fs-shell-4 11-14: where the ray is in kept space; a clipped entry draws nothing, the form
        // is still entered, so its cut face shows where the ray comes back into kept space)
        if (clipf == 0 || xr_kept(P, clipf, k0, k1, o, d, tb)) {
            float3 p = add(o, mul(d, tb));
            float3 n = xr_normal(vf, vi, grid, ttab, sdf, p);
            if (dot3(n, d) > 0.0f) n = mul(n, -1.0f);
            float c3[3];
            xr_clay(n, d, vb, layer, c3);
            float a = (1.0f - xray) * fminf(fmaxf(vf[VF_OPA], 0.0f), 1.0f);   // each object at its own opacity
            for (int c = 0; c < 3; c++) C[c] += T * a * c3[c];
            T *= 1.0f - a;
            layer++;
        }
        inside |= (1ull << vb);
        tev[vb] = xr_exit(vf, vi, grid, ttab, sdf, o, d, tb + eps, tmax);
    }
    float bg = 0.10f + 0.05f * (1.0f - (y + 0.5f) / (float)H);     // the ground behind every layer (view modes' grey)
    for (int c = 0; c < 3; c++) C[c] += T * bg;
    unsigned int r = (unsigned int)(srgb(fminf(C[0], 1.0f)) * 255.0f + 0.5f), gg = (unsigned int)(srgb(fminf(C[1], 1.0f)) * 255.0f + 0.5f),
                 b = (unsigned int)(srgb(fminf(C[2], 1.0f)) * 255.0f + 0.5f);
    out[pi] = 0xff000000u | (r << 16) | (gg << 8) | b;
}

// ---- fs-shell-2 11-14: select inside, part 3 (the selected object's outline through what covers it) and part 4 (the
// Scene list's hovered object): the ID image of ONE volume on its own. Per pixel: bit 0 = this pixel's ray enters volume
// vsel anywhere (whatever is in front of it), bit 1 = vsel is the frame's front surface here (the front-ID image).
// PATENT GUARDS (QuantifiCare US 11,094,139 B2, the full-claim read in design/LIBRARIES_AND_PATENTS_2026-11-14.md,
// addendum bab2b321; not legal advice), kept by this kernel and by ui/viewport.py's _id_outline that draws from it:
//   1. the outline comes ONLY from edges of these ID images (this mask and the front-ID image), never from N.V and
//      never from a contour of the field's gradient against the view direction;
//   2. its colour and alpha are constant along the line (antialiasing the only variation): no fade, glow or width by
//      angle;
//   3. the hidden object is never drawn shaded, tinted or translucent by this feature (that is X-ray's job).
// This kernel writes a mask and nothing else: no normal is read, no colour is made.
extern "C" __global__ void vol_mask(int nvol, const float* vft, const int* vit, const int* grid, const int* ttab, const __half* sdf,
                                    const float* P, int W, int H, int x0, int y0, int rw, int rh, const int* objid,
                                    int vsel, unsigned char* out) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= rw || py >= rh) return;
    int x = x0 + px, y = y0 + py;
    if (x >= W || y >= H) return;
    size_t pi = (size_t)y * W + x;
    unsigned char m = 0;
    if (objid[pi] == vsel) {
        m = 3;
    } else if (vsel >= 0 && vsel < nvol) {
        const float* vf = vft + vsel * VF; const int* vi = vit + vsel * VI;
        if (vf[VF_VIS] >= 0.5f) {
            float sx = (2.0f * (x + 0.5f) / W - 1.0f) * P[12] * P[13], sy = (1.0f - 2.0f * (y + 0.5f) / H) * P[12];
            float3 o = mk(P[0], P[1], P[2]);
            float3 d = nrm(add(mk(P[3], P[4], P[5]), add(mul(mk(P[6], P[7], P[8]), sx), mul(mk(P[9], P[10], P[11]), sy))));
            float pixang = 2.0f * P[12] / H;
            if (P[17] > 0.0f) {
                float3 f = mk(P[3], P[4], P[5]);
                o = add(add(o, mul(add(mul(mk(P[6], P[7], P[8]), sx), mul(mk(P[9], P[10], P[11]), sy)), P[17])), mul(f, -P[18]));
                d = f;
                pixang = 2.0f * P[12] * P[17] / (H * (P[17] + P[18]));
            }
            float te = xr_entry(vf, vi, grid, ttab, sdf, o, d, 0.0f, P[15], pixang, P[16], (int)P[14]);
            if (te < P[15]) m = 1;
        }
    }
    out[pi] = m;
}
'''


class ViewModes:
    """the compiled view-mode kernel for one GpuScene, and its settings"""

    def __init__(self, G):
        import cupy as cp
        self.cp = cp
        src = open(os.path.join(HERE, 'trace.cu'), encoding='utf-8').read() + SRC
        self.mod = cp.RawModule(code=src, options=('--use_fast_math', '-std=c++14'))
        self.k = self.mod.get_function('view_pass')
        self.G = G
        self.saved = None                  # the k_light in place before a mode took the slot (fs-light's may be there)
        self.mode = 'shaded'
        self.settings = {'cavity': 0.6, 'curv_scale': 0.004, 'matcap': 'clay', 'sample': 0.001,
                         'sil_ground': 0.86, 'sil_form': 0.06}      # fs-shell 11-13: the silhouette's two values
        self.k_xray = self.mod.get_function('xray_comp')
        self.k_volmask = self.mod.get_function('vol_mask')     # fs-shell-2 11-14: select inside parts 3 / 4
        self.d_xout = None
        self._pushed = None
        G.view_modes = self                # fs-light 11-12 (b47d6ae1): while a mode holds the light slot, its light
        #                                    runtime writes its kernel into .saved (reads .mode, .k, .saved only)

    def _push(self, mode):
        s = self.settings
        vals = np.zeros(16, np.float32)
        vals[0] = MODES.get(mode, 1)
        vals[1] = float(s['cavity'])
        vals[2] = float(s['curv_scale'])
        vals[3] = MATCAPS.get(s['matcap'], 0)
        vals[4] = float(s['sample'])
        vals[6] = float(s.get('sil_ground', 0.86))
        vals[7] = float(s.get('sil_form', 0.06))
        key = vals.tobytes()
        if key != self._pushed:
            ptr = self.mod.get_global('VM')                     # a device pointer: viewed as 16 float32 to write
            self.cp.ndarray((16,), self.cp.float32, ptr)[...] = self.cp.asarray(vals)
            self._pushed = key
            return True
        return False

    def apply(self, gv, mode):
        """take or give back the light slot for this frame's mode -> True when the image must restart"""
        G = gv.gpu
        changed = False
        if mode in ('matcap', 'curvature', 'normal', 'silhouette'):
            if G.k_light is not self.k:
                self.saved = G.k_light
                G.k_light = self.k
                changed = True
            changed |= self._push(mode)
        elif G.k_light is self.k:
            G.k_light = self.saved if self.saved is not None else (getattr(G, '_k_light0', None) or G.mod.get_function('light_pass'))
            self.saved = None
            self._pushed = None
            changed = True
        if mode != self.mode:
            changed = True
            self.mode = mode
        if changed:
            gv.key = None                  # the next frame restarts every pixel (as fs-light's swap does)
        return changed


    def xray(self, G, cam, W, H, xray, max_layers=6):
        """fs-shell 11-13: the X-ray composite over the frame G just resolved (d_out): -> the device buffer to show.
        Through the launch guard (kind 'xray'). xray: 0 (the front surface opaque) .. 0.95 (nearly clear)"""
        cp = self.cp
        if self.d_xout is None or self.d_xout.shape != G.d_out.shape:
            self.d_xout = cp.empty_like(G.d_out)
        P = cp.asarray(G.cam_params(cam, W, H))
        geo = (np.int32(len(G.vols)), G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf)
        k = self.k_xray

        def issue(tx, ty, tw, th):
            k(((tw + 15) // 16, (th + 7) // 8, 1), (16, 8, 1),
              geo + (P, np.int32(W), np.int32(H), np.int32(tx), np.int32(ty), np.int32(tw), np.int32(th),
                     G.d_depth, G.d_obj, G.d_out, self.d_xout, np.float32(xray), np.int32(max_layers)))
        G.guard.run('xray', W, H, (0, 0, W, H), issue, units=1, gen=G._guard_gen())
        return self.d_xout


def vol_mask(vm, G, cam, W, H, vsel):
    """fs-shell-2 11-14: the ID image of volume vsel on its own (bit 0: on this ray, bit 1: the front surface), for the
    outline of a covered object (select inside part 3, the Scene list's hover part 4; the patent guards are on the
    kernel). Through the launch guard (kind 'volmask'). -> a device (H, W) uint8 array"""
    cp = vm.cp
    out = cp.zeros((H, W), cp.uint8)
    P = cp.asarray(G.cam_params(cam, W, H))
    geo = (np.int32(len(G.vols)), G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf)
    k = vm.k_volmask

    def issue(tx, ty, tw, th):
        k(((tw + 15) // 16, (th + 7) // 8, 1), (16, 8, 1),
          geo + (P, np.int32(W), np.int32(H), np.int32(tx), np.int32(ty), np.int32(tw), np.int32(th),
                 G.d_obj, np.int32(vsel), out))
    G.guard.run('volmask', W, H, (0, 0, W, H), issue, units=1, gen=G._guard_gen())
    return out


def install(engine):
    """the engine draws matcap / curvature / normals on the card through this kernel"""
    engine.card_modes = ('shaded', 'matcap', 'curvature', 'normal', 'silhouette')
    orig = engine._render_gpu

    def _render_gpu(view, interactive, orig=orig, engine=engine):
        gv = engine.gpu_view
        vm = getattr(gv, 'viewmodes', None)
        mode = view.get('mode', 'shaded')
        if vm is None and mode != 'shaded':
            vm = gv.viewmodes = ViewModes(gv.gpu)
        if vm is not None:
            if vm.G is not gv.gpu:             # a new GpuScene (the view was rebuilt): a new module for it
                vm = gv.viewmodes = ViewModes(gv.gpu)
            vm.settings.update(getattr(engine, 'vm_settings', None) or {})   # the window's matcap kind / cavity
            vm.apply(gv, mode)
        return orig(view, interactive)
    engine._render_gpu = _render_gpu
    return engine
