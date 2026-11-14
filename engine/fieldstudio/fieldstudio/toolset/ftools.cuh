// The approved brush tools' rules on the card (fs-tools, 11-14): the same rules as toolset/ftools.py (the CPU
// reference), device functions only, for fs-core's dab kernel to call for typ >= 30 (stage 2 proposal,
// 11-14). The rules are templated on a READER: any object with `float operator()(float x, float y, float z)
// const` giving the field (local units) at a local point, the way the includer's other branches read it (fs-core wraps
// obj_f with its arguments; a test wraps a dense grid). Include warp/warp.cuh first (the shared profile, w_profile).
// Parameter slots (bp, NBP grown to 64; 1-3 c, 4-6 n, 7 r are the dab's own as for every tool):
#ifndef FTOOLS_CUH
#define FTOOLS_CUH
#define FT_HARD 51        // Hardness (0-1)
#define FT_SMOOTH 52      // Smoothing (local units)
#define FT_FALL 53        // the foot's falloff (0 soft, 1 even, 2 sharp)
#define FT_FAMILY 54      // 0 the Shape family, 1 the Swell family (a sphere's dome at Hardness 1)
#define FT_A 55           // swell / settle: the quantum's rise (local units, signed: + swell, - settle); flatten /
                          // smooth: rate x dt
#define FT_MODE 56        // swell: 0 follow, 1 flat
#define FT_GAP 57         // Stop at contact: the gap (local units)
#define FT_CONTACT 58     // 1: Stop at contact (the includer passes the contact distance tc)
#define FT_P 59           // 3: flatten's plane point (local); its facing is bp[4..6]
#define FT_H 62           // the lattice spacing (local units): the gradient's step
#define FT_R 63           // the profile's radius (local units; 0: bp[7]). In fs-core's dab kernel bp[7] is the dab's
                          // whole ball (radius + Smoothing), as for every tool: nothing changes beyond it
// the typ codes: core/brush.TOOLS numbering (gpu/sculpt.py passes -DFT_TYP_SWELL=... from brush.T); 30-32 standalone
#ifndef FT_TYP_SWELL
#define FT_TYP_SWELL 30
#define FT_TYP_SETTLE 31
#define FT_TYP_FLATTEN 32
#endif
#ifndef FT_TYP_PENCIL
#define FT_TYP_PENCIL 33
#endif
// Pencil (the tip acts where it is at each dab; the window lays the dabs along the path at the tip's spacing): FT_R
// the tip's half-width,
// FT_A the depth or height, FT_MODE the tip (0 point, 1 round, 2 square) + 4 for a ridge, FT_SMOOTH the blend width k,
// FT_P..+2 the stroke's direction at the dab (local)
#define FT_MIRP 41        // fs-core's mirror partner (core/brush.NBP_MIRP): its centre 41-43 (local), 44 > 0.5 = on
#define FT_FAR 1e29f      // a reader's 'no information' (fs-core's obj_f: 1e30 outside every brick): the node keeps f0

// the node's anchor: its own point projected onto the surface along the field's gradient (on the card the press-time
// anchor of the CPU reference is read this way, from the field as it is; the same point while the surface rises along
// its facing)
template <class Rd>
__device__ __forceinline__ bool ft_anchor(const Rd& rd, float x, float y, float z, float f0, float h, float* q) {
    float a = rd(x + h, y, z), b = rd(x - h, y, z), c = rd(x, y + h, z), d = rd(x, y - h, z);
    float e = rd(x, y, z + h), f = rd(x, y, z - h);
    if (fabsf(a) > FT_FAR || fabsf(b) > FT_FAR || fabsf(c) > FT_FAR || fabsf(d) > FT_FAR || fabsf(e) > FT_FAR
        || fabsf(f) > FT_FAR) return false;              // a read outside the held bricks: no anchor
    float gx = a - b, gy = c - d, gz = e - f;
    float gl = sqrtf(gx * gx + gy * gy + gz * gz);
    if (gl < 1e-12f) { q[0] = x; q[1] = y; q[2] = z; return true; }
    float s = f0 / gl;
    q[0] = x - s * gx; q[1] = y - s * gy; q[2] = z - s * gz;
    return true;
}

// the mirror feather (as fs-core's dab_compute): where this dab's profile overlaps its mirror partner's, the weight is
// scaled by (w^2 + w2^2) / (w + w2)^2, so the pair sums to (w^2 + w2^2) / (w + w2): smooth, w alone and where the two
// coincide. dx, dy, dz: the point the weight is read at (the anchor, or the node) minus the partner's centre; lateral:
// the distance across the facing n (flat tools), else the full distance (follow).
__device__ __forceinline__ float ft_feather(const float* bp, float w, float px, float py, float pz, bool lateral,
                                            float nx, float ny, float nz, float r, float hard, float sm, int fall, int fam) {
    if (!(bp[FT_MIRP + 3] > 0.5f) || w <= 0.0f) return w;
    float ex = px - bp[FT_MIRP], ey = py - bp[FT_MIRP + 1], ez = pz - bp[FT_MIRP + 2];
    if (lateral) { float t = ex * nx + ey * ny + ez * nz; ex -= t * nx; ey -= t * ny; ez -= t * nz; }
    float w2 = (float)w_profile((wr)sqrtf(ex * ex + ey * ey + ez * ez), (wr)r, (wr)hard, (wr)sm, fall, fam);
    if (w2 <= 0.0f) return w;
    return w * (w * w + w2 * w2) / ((w + w2) * (w + w2));
}

// the unit facing of the field at a point: central differences of trilinear reads at +-h (toolset/ftools.py _grad_pt)
template <class Rd>
__device__ __forceinline__ bool ft_facing(const Rd& rd, float x, float y, float z, float h, float* n) {
    float a = rd(x + h, y, z), b = rd(x - h, y, z), c = rd(x, y + h, z), d = rd(x, y - h, z);
    float e = rd(x, y, z + h), f = rd(x, y, z - h);
    if (fabsf(a) > FT_FAR || fabsf(b) > FT_FAR || fabsf(c) > FT_FAR || fabsf(d) > FT_FAR || fabsf(e) > FT_FAR
        || fabsf(f) > FT_FAR) return false;
    float gx = a - b, gy = c - d, gz = e - f;
    float gl = fmaxf(sqrtf(gx * gx + gy * gy + gz * gz), 1e-12f);
    n[0] = gx / gl; n[1] = gy / gl; n[2] = gz / gl;
    return true;
}

// Pencil (TOOLSET_DESIGN 3.3): the tip acts where it is, at the dab's point, on the surface AS IT WAS AT THE PRESS
// (rs): the
// point projected along the press-time facing (four steps, as the CPU reference), the tip's frame from that facing and
// the stroke's direction, the depth limited to half the part's thickness under it (carve), then the tip's field T at the
// node: carve f = smooth max(f, -T), ridge f = smooth min(f, T) (k = the blend width). The tip acts only where it
// is, never a continuous search for its best placement along the path (the patent route: MERL US 8,265,909);
// within one drag a spot is never recut, and the depth is constant whatever the speed. Every node of the dab
// computes the same frame from the same reads, so the dab is one placement of the tip. toolset/ftools.py
// FtStroke._pencil is the reference.
template <class Rs>
__device__ float ft_pencil(const Rs& rs, const float* bp, float x, float y, float z, float f0) {
    float h = bp[FT_H], r = bp[FT_R], depth = bp[FT_A], k = bp[FT_SMOOTH];
    int md = (int)bp[FT_MODE];
    bool ridge = md >= 4;
    int tip = md & 3;
    float q[3] = {bp[1], bp[2], bp[3]}, n[3];
    if (!ft_facing(rs, q[0], q[1], q[2], h, n)) return f0;
    for (int it = 0; it < 4; ++it) {
        float v = rs(q[0], q[1], q[2]);
        if (fabsf(v) > FT_FAR) return f0;
        q[0] -= v * n[0]; q[1] -= v * n[1]; q[2] -= v * n[2];
    }
    if (!ft_facing(rs, q[0], q[1], q[2], h, n)) return f0;
    float t[3] = {bp[FT_P], bp[FT_P + 1], bp[FT_P + 2]};
    float tn = t[0] * n[0] + t[1] * n[1] + t[2] * n[2];
    t[0] -= tn * n[0]; t[1] -= tn * n[1]; t[2] -= tn * n[2];
    float tl = sqrtf(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    if (tl > 1e-9f) { t[0] /= tl; t[1] /= tl; t[2] /= tl; }
    else {                                             // no direction yet (the first dab): any frame on the surface
        float a0 = fabsf(n[0]) < 0.9f ? 1.0f : 0.0f, a1 = fabsf(n[0]) < 0.9f ? 0.0f : 1.0f;
        float an = a0 * n[0] + a1 * n[1];
        t[0] = a0 - an * n[0]; t[1] = a1 - an * n[1]; t[2] = -an * n[2];
        tl = fmaxf(sqrtf(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]), 1e-12f);
        t[0] /= tl; t[1] /= tl; t[2] /= tl;
    }
    float v[3] = {n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]};
    float dl = depth;
    if (!ridge) {                                      // the part's thickness under the point, inward along -n
        float tmax = 4.0f * depth + 4.0f * h, s = 0.5f * h, th = tmax;
        while (s < tmax) {
            float fv = rs(q[0] - s * n[0], q[1] - s * n[1], q[2] - s * n[2]);
            if (fabsf(fv) > FT_FAR) break;                  // off the held bricks: no limit read
            if (fv > 0.0f) { th = s; break; }
            s += fmaxf(fabsf(fv), 0.25f * h);
        }
        dl = fminf(depth, 0.5f * th);
    }
    float ex = x - q[0], ey = y - q[1], ez = z - q[2];
    float a = ex * t[0] + ey * t[1] + ez * t[2];          // along the stroke
    float b = ex * v[0] + ey * v[1] + ez * v[2];          // across it
    float w = ex * n[0] + ey * n[1] + ez * n[2];          // along the facing (0 = the surface)
    if (ridge) w = -w;
    float T;
    if (tip == 0) {                                    // point: a cone, half-width r at the surface
        float rho = sqrtf(a * a + b * b), ww = w + dl, L = sqrtf(r * r + dl * dl);
        float ca = dl / L, sa = r / L;
        T = (ww * ca + rho * sa < 0.0f) ? sqrtf(rho * rho + ww * ww) : rho * ca - ww * sa;
    } else if (tip == 1) {                             // round: a ball whose bottom is at the depth
        float cz = r - dl;
        T = sqrtf(a * a + b * b + (w - cz) * (w - cz)) - r;
    } else {                                           // square: a flat chisel, half-width r, flat bottom
        float qx = fabsf(b) - r, qy = fabsf(a) - r, qz = fabsf(w - (r - dl)) - r;
        float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f), mz = fmaxf(qz, 0.0f);
        T = sqrtf(mx * mx + my * my + mz * mz) + fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);
    }
    if (!ridge) {
        float c = -T;
        if (k > 0.0f) { float hh = fmaxf(k - fabsf(f0 - c), 0.0f) / k; return fmaxf(f0, c) + hh * hh * k * 0.25f; }
        return fmaxf(f0, c);
    }
    if (k > 0.0f) { float hh = fmaxf(k - fabsf(f0 - T), 0.0f) / k; return fminf(f0, T) - hh * hh * k * 0.25f; }
    return fminf(f0, T);
}

// one quantum of an approved tool at node (x, y, z) with value f0; tc = the contact distance (when FT_CONTACT).
// rd reads the field as it is; rs the field AS IT WAS AT THE PRESS (fs-tools patch 3): Follow's anchors are the nodes'
// nearest press-time surface points, and the cursor's own, so the dome is measured along the press-time surface and a
// held swell grows linearly in time everywhere (the CPU reference's anchors); rs = rd gives the earlier rule.
template <class Rd, class Rs>
__device__ float ft_rule2(const Rd& rd, const Rs& rs, int typ, const float* bp, float x, float y, float z, float f0, float tc,
                          bool press) {
    float r = bp[FT_R] > 0.0f ? bp[FT_R] : bp[7], hard = bp[FT_HARD], sm = bp[FT_SMOOTH];
    int fall = (int)bp[FT_FALL], fam = (int)bp[FT_FAMILY];
    float cx = bp[1], cy = bp[2], cz = bp[3], nx = bp[4], ny = bp[5], nz = bp[6];
    if (typ == FT_TYP_SWELL || typ == FT_TYP_SETTLE) {
        float a = bp[FT_A];
        float f1;
        if ((int)bp[FT_MODE] == 0) {                     // follow: each point along its own facing
            float q[3], c0[3] = {cx, cy, cz};
            float fs = press ? rs(x, y, z) : f0;
            if (fabsf(fs) > FT_FAR || !ft_anchor(rs, x, y, z, fs, bp[FT_H], q)) return f0;
            for (int it = 0; press && it < 2; ++it) {                 // the cursor's point (on the surface as it is) onto the press-time one
                float fc = rs(c0[0], c0[1], c0[2]);
                float cn[3];
                if (fabsf(fc) > FT_FAR || !ft_anchor(rs, c0[0], c0[1], c0[2], fc, bp[FT_H], cn)) break;
                c0[0] = cn[0]; c0[1] = cn[1]; c0[2] = cn[2];
            }
            float d = sqrtf((q[0] - c0[0]) * (q[0] - c0[0]) + (q[1] - c0[1]) * (q[1] - c0[1]) + (q[2] - c0[2]) * (q[2] - c0[2]));
            float w = (float)w_profile((wr)d, (wr)r, (wr)hard, (wr)sm, fall, fam);
            w = ft_feather(bp, w, q[0], q[1], q[2], false, nx, ny, nz, r, hard, sm, fall, fam);
            if (w <= 0.0f) return f0;
            f1 = f0 - a * w;
        } else {                                         // flat: a held flow along the press facing n
            float ex = x - cx, ey = y - cy, ez = z - cz;
            float t = ex * nx + ey * ny + ez * nz;
            float lx = ex - t * nx, ly = ey - t * ny, lz = ez - t * nz;
            float w = (float)w_profile((wr)sqrtf(lx * lx + ly * ly + lz * lz), (wr)r, (wr)hard, (wr)sm, fall, fam);
            w = ft_feather(bp, w, x, y, z, true, nx, ny, nz, r, hard, sm, fall, fam);
            if (w <= 0.0f) return f0;
            float s = a * w;
            f1 = rd(x - s * nx, y - s * ny, z - s * nz);
            if (fabsf(f1) > FT_FAR) return f0;
        }
        if (bp[FT_CONTACT] > 0.5f) f1 = fmaxf(f1, bp[FT_GAP] - tc);   // Stop at contact: never into the other field
        return f1;
    }
    if (typ == FT_TYP_FLATTEN) {                         // onto the plane through the point under the cursor
        float ex = x - bp[FT_P], ey = y - bp[FT_P + 1], ez = z - bp[FT_P + 2];
        float P = ex * nx + ey * ny + ez * nz;
        float lx = ex - P * nx, ly = ey - P * ny, lz = ez - P * nz;
        float w = (float)w_profile((wr)sqrtf(lx * lx + ly * ly + lz * lz), (wr)r, (wr)hard, (wr)sm, fall, fam);
        w = ft_feather(bp, w, x, y, z, true, nx, ny, nz, r, hard, sm, fall, fam);
        if (w <= 0.0f) return f0;
        return f0 + (P - f0) * (1.0f - expf(-bp[FT_A] * w));
    }
    if (typ == FT_TYP_PENCIL) return ft_pencil(rs, bp, x, y, z, f0);
    return f0;
}

// the one-reader form (the field as it is for the anchors too): sculpt.cu before patch 3, and tests on a fixed grid
template <class Rd>
__device__ float ft_rule(const Rd& rd, int typ, const float* bp, float x, float y, float z, float f0, float tc) {
    return ft_rule2(rd, rd, typ, bp, x, y, z, f0, tc, false);
}
#endif
