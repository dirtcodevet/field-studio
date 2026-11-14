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
