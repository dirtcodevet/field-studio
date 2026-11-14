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
