// The one warp engine on the card (fs-tools, 11-14): the same flow as warp/flow.py, device functions only, so any card
// kernel (a live Shape or Transform drag in the sculpt path, a bake) maps points through the ONE engine. Layouts are
// flow.py's (FP_* parameter block, ST_* stage rows, 3 rows per RK4 step); kept in step by warp/card.py's parity check.
// WREAL: float for live work (sub-micrometre on positions near the support), double for the parity check.
#ifndef WARP_CUH
#define WARP_CUH
#ifndef WREAL
#define WREAL float
#endif
typedef WREAL wr;

#define FP_METRIC 0
#define FP_R 1
#define FP_HARD 2
#define FP_SMOOTH 3
#define FP_FALL 4
#define FP_FAMILY 5
#define FP_C0 6
#define FP_N 9
#define FP_E1 12
#define FP_E2 15
#define FP_REACH 18
#define FP_EPSMIN 19
#define FP_KEEPVOL 20
#define FP_CONTACT 21
#define FP_CEPS 22
#define FP_CGAP 23
#define FP_IMG_PIX 24
#define FP_IMG_W 25
#define FP_IMG_H 26
#define FP_IMG_U0 27
#define FP_IMG_V0 28
#define FP_G_O 29
#define FP_G_H 32
#define FP_C_O 33
#define FP_C_H 36
#define FP_NSTEP 37
#define FP_GRADEPS 38
#define FP_TRAVEL 39
#define ST_A 12
#define ST_TAU 21
#define ST_P 24
#define ST_AX 27
#define ST_T 28
#define ST_N 29

struct WarpArgs {
    const wr* fp; const wr* img; int imgc;            // img: imgc x H x W
    const wr* G; int gx, gy, gz;                      // the region weight grid
    const wr* CG; int cx, cy, cz;                     // the contact distance grid
    const wr* st;                                     // stage rows
};

__device__ __forceinline__ wr w_tri(const wr* Gd, int nx, int ny, int nz, wr ox, wr oy, wr oz, wr h, wr x, wr y, wr z,
                                    wr outside) {
    wr fx = (x - ox) / h, fy = (y - oy) / h, fz = (z - oz) / h;
    if (fx < 0 || fy < 0 || fz < 0 || fx > nx - 1 || fy > ny - 1 || fz > nz - 1) return outside;
    int i = min((int)fx, nx - 2), j = min((int)fy, ny - 2), k = min((int)fz, nz - 2);
    wr tx = fx - i, ty = fy - j, tz = fz - k;
#define GV(a, b, c) Gd[((long)(a) * ny + (b)) * nz + (c)]
    wr a = GV(i, j, k) + (GV(i + 1, j, k) - GV(i, j, k)) * tx;
    wr b = GV(i, j + 1, k) + (GV(i + 1, j + 1, k) - GV(i, j + 1, k)) * tx;
    wr c = GV(i, j, k + 1) + (GV(i + 1, j, k + 1) - GV(i, j, k + 1)) * tx;
    wr d = GV(i, j + 1, k + 1) + (GV(i + 1, j + 1, k + 1) - GV(i, j + 1, k + 1)) * tx;
#undef GV
    wr e = a + (b - a) * ty, f = c + (d - c) * ty;
    return e + (f - e) * tz;
}

__device__ __forceinline__ wr w_img(const WarpArgs& A, int c, wr u1, wr u2) {
    const wr* fp = A.fp;
    int W = (int)fp[FP_IMG_W], H = (int)fp[FP_IMG_H];
    wr px = (u1 - fp[FP_IMG_U0]) / fp[FP_IMG_PIX], py = (u2 - fp[FP_IMG_V0]) / fp[FP_IMG_PIX];
    px = fmin(fmax(px, (wr)0), (wr)(W - 1.000001));
    py = fmin(fmax(py, (wr)0), (wr)(H - 1.000001));
    int i = (int)px, j = (int)py;
    wr tx = px - i, ty = py - j;
    const wr* I = A.img + (long)c * H * W;
    wr a = I[j * W + i] + (I[j * W + i + 1] - I[j * W + i]) * tx;
    wr b = I[(j + 1) * W + i] + (I[(j + 1) * W + i + 1] - I[(j + 1) * W + i]) * tx;
    return a + (b - a) * ty;
}

__device__ __forceinline__ wr w_ramp(wr t) {
    if (t <= 0) return 1;
    if (t >= 1) return 0;
    return 1 - t * t * (3 - 2 * t);
}

// the shared profile (warp/profile.py): the peak (Hardness) and the foot (Smoothing, its falloff)
__device__ __forceinline__ void w_peak(wr u, wr hard, int family, wr& pk, wr& m) {
    const wr PI = (wr)3.14159265358979323846;
    wr uu = fmin(fmax(u, (wr)0), (wr)1);
    if (family == 1) {
        wr a = cos((wr)0.5 * PI * uu);
        wr dome = sqrt(fmax((wr)1 - uu * uu, (wr)0));
        pk = (1 - hard) * a * a + hard * dome;
        m = -((1 - hard) * (wr)0.25 * PI + hard * 2);
        return;
    }
    pk = (1 - hard) * cos((wr)0.5 * PI * uu) + hard * (1 - uu);
    m = -((1 - hard) * (wr)0.5 * PI + hard);
}

__device__ __forceinline__ wr w_smax0(wr a, wr k, int fall) {
    if (k <= 0) return fmax(a, (wr)0);
    wr h = k - fabs(a);
    if (h <= 0) return fmax(a, (wr)0);
    h = h / k;
    if (fall == 0) return fmax(a, (wr)0) + k * h * h * (wr)0.25;
    if (fall == 1) return fmax(a, (wr)0) + k * h * h * h / 6;
    return fmax(a, (wr)0) + k * h * h * h * h * (wr)0.125;
}

__device__ __forceinline__ wr w_raw(wr d, wr r, wr hard, wr smooth, int fall, int family) {
    wr u = d / r, pk, m;
    w_peak(u, hard, family, pk, m);
    wr a = u <= 1 ? pk : m * (u - 1);
    return w_smax0(a, -m * smooth / r, fall);
}

__device__ __forceinline__ wr w_profile(wr d, wr r, wr hard, wr smooth, int fall, int family) {
    if (d >= r + smooth) return 0;
    wr w0 = w_raw(0, r, hard, smooth, fall, family);
    return fmin(w_raw(d, r, hard, smooth, fall, family) / w0, (wr)1);
}

__device__ __forceinline__ void w_travel(const WarpArgs& A, int s, wr x, wr y, wr z, wr& sd, wr& dsda, wr& tl) {
    const wr* fp = A.fp; const wr* S = A.st + (long)s * ST_N;
    tl = sqrt(S[ST_TAU] * S[ST_TAU] + S[ST_TAU + 1] * S[ST_TAU + 1] + S[ST_TAU + 2] * S[ST_TAU + 2]);
    wr a = S[ST_T] * tl;
    wr ex = x - fp[FP_C0], ey = y - fp[FP_C0 + 1], ez = z - fp[FP_C0 + 2];
    wr u1 = ex * fp[FP_E1] + ey * fp[FP_E1 + 1] + ez * fp[FP_E1 + 2];
    wr u2 = ex * fp[FP_E2] + ey * fp[FP_E2 + 1] + ez * fp[FP_E2 + 2];
    wr b = w_img(A, 2, u1, u2);
    wr k = fp[FP_EPSMIN];
    wr hh = fmax(k - fabs(a - b), (wr)0) / k;
    sd = fmin(a, b) - hh * hh * k * (wr)0.25;
    dsda = a <= b ? 1 - (wr)0.5 * hh : (wr)0.5 * hh;
}

__device__ wr w_body_weight(const WarpArgs& A, int s, wr x, wr y, wr z) {
    const wr* fp = A.fp; const wr* S = A.st + (long)s * ST_N;
    wr yx, yy, yz, ax;
    if (fp[FP_TRAVEL] > (wr)0.5) {
        wr sd, ds, tl;
        w_travel(A, s, x, y, z, sd, ds, tl);
        wr f = tl > 0 ? sd / tl : 0;
        yx = x - f * S[ST_TAU]; yy = y - f * S[ST_TAU + 1]; yz = z - f * S[ST_TAU + 2];
        ax = f * (S[ST_TAU] * fp[FP_N] + S[ST_TAU + 1] * fp[FP_N + 1] + S[ST_TAU + 2] * fp[FP_N + 2]);
    } else {
        yx = S[0] * x + S[1] * y + S[2] * z + S[3];
        yy = S[4] * x + S[5] * y + S[6] * z + S[7];
        yz = S[8] * x + S[9] * y + S[10] * z + S[11];
        ax = S[ST_AX];
    }
    int m = (int)fp[FP_METRIC];
    wr r = fp[FP_R];
    if (m == 2) return w_tri(A.G, A.gx, A.gy, A.gz, fp[FP_G_O], fp[FP_G_O + 1], fp[FP_G_O + 2], fp[FP_G_H], yx, yy, yz, 0);
    wr ex = yx - fp[FP_C0], ey = yy - fp[FP_C0 + 1], ez = yz - fp[FP_C0 + 2];
    if (m == 0) {
        wr d = sqrt(ex * ex + ey * ey + ez * ez);
        return w_profile(d, r, fp[FP_HARD], fp[FP_SMOOTH], (int)fp[FP_FALL], (int)fp[FP_FAMILY]);
    }
    wr u1 = ex * fp[FP_E1] + ey * fp[FP_E1 + 1] + ez * fp[FP_E1 + 2];
    wr u2 = ex * fp[FP_E2] + ey * fp[FP_E2 + 1] + ez * fp[FP_E2 + 2];
    wr w = w_profile(sqrt(u1 * u1 + u2 * u2), r, fp[FP_HARD], fp[FP_SMOOTH], (int)fp[FP_FALL], (int)fp[FP_FAMILY]);
    if (w <= 0) return 0;
    int reach = (int)fp[FP_REACH];
    if (reach < 0 || reach == 2) return w;
    wr zb = ex * fp[FP_N] + ey * fp[FP_N + 1] + ez * fp[FP_N + 2];
    wr ze = w_img(A, 1, u1, u2) - ax;
    wr zs = fmin(w_img(A, 0, u1, u2), ze - fp[FP_EPSMIN]);
    if (zb <= zs) return w;
    return w * w_ramp((zb - zs) / (ze - zs));
}

__device__ __forceinline__ void w_column_point(const WarpArgs& A, int s, wr& x, wr& y, wr& z) {
    const wr* fp = A.fp; const wr* S = A.st + (long)s * ST_N;
    if ((int)fp[FP_METRIC] != 1 || (int)fp[FP_REACH] != 1) return;
    wr yx = S[0] * x + S[1] * y + S[2] * z + S[3];
    wr yy = S[4] * x + S[5] * y + S[6] * z + S[7];
    wr yz = S[8] * x + S[9] * y + S[10] * z + S[11];
    wr ex = yx - fp[FP_C0], ey = yy - fp[FP_C0 + 1], ez = yz - fp[FP_C0 + 2];
    wr u1 = ex * fp[FP_E1] + ey * fp[FP_E1 + 1] + ez * fp[FP_E1 + 2];
    wr u2 = ex * fp[FP_E2] + ey * fp[FP_E2 + 1] + ez * fp[FP_E2 + 2];
    wr zb = ex * fp[FP_N] + ey * fp[FP_N + 1] + ez * fp[FP_N + 2];
    wr zs = w_img(A, 0, u1, u2);
    if (zb >= zs) return;
    wr dz = zs - zb;
    x += dz * fp[FP_N]; y += dz * fp[FP_N + 1]; z += dz * fp[FP_N + 2];
}

__device__ void w_velocity(const WarpArgs& A, int s, wr x, wr y, wr z, wr* out) {
    const wr* fp = A.fp; const wr* S = A.st + (long)s * ST_N;
    wr W = w_body_weight(A, s, x, y, z);
    bool keep = fp[FP_KEEPVOL] > (wr)0.5;
    if (W <= 0 && !keep) { out[0] = out[1] = out[2] = 0; return; }
    wr px = x - S[ST_P], py = y - S[ST_P + 1], pz = z - S[ST_P + 2];
    wr v0x = S[ST_A] * px + S[ST_A + 1] * py + S[ST_A + 2] * pz + S[ST_TAU];
    wr v0y = S[ST_A + 3] * px + S[ST_A + 4] * py + S[ST_A + 5] * pz + S[ST_TAU + 1];
    wr v0z = S[ST_A + 6] * px + S[ST_A + 7] * py + S[ST_A + 8] * pz + S[ST_TAU + 2];
    bool travel = fp[FP_TRAVEL] > (wr)0.5;
    if (travel) { wr sd, ds, tl; w_travel(A, s, x, y, z, sd, ds, tl); W *= ds; }
    wr vx = W * v0x, vy = W * v0y, vz = W * v0z;
    if (keep) {
        wr e = fp[FP_GRADEPS];
        wr gx = (w_body_weight(A, s, x + e, y, z) - w_body_weight(A, s, x - e, y, z)) / (2 * e);
        wr gy = (w_body_weight(A, s, x, y + e, z) - w_body_weight(A, s, x, y - e, z)) / (2 * e);
        wr gz = (w_body_weight(A, s, x, y, z + e) - w_body_weight(A, s, x, y, z - e)) / (2 * e);
        if (gx != 0 || gy != 0 || gz != 0) {
            wr tx = S[ST_TAU], ty = S[ST_TAU + 1], tz = S[ST_TAU + 2];
            wr wx = S[ST_A + 7], wy = S[ST_A + 2], wz = S[ST_A + 3];
            wr q2 = px * px + py * py + pz * pz;
            wr sx = (wr)0.5 * (ty * pz - tz * py) - (wr)0.5 * q2 * wx;
            wr sy = (wr)0.5 * (tz * px - tx * pz) - (wr)0.5 * q2 * wy;
            wr sz = (wr)0.5 * (tx * py - ty * px) - (wr)0.5 * q2 * wz;
            vx += gy * sz - gz * sy; vy += gz * sx - gx * sz; vz += gx * sy - gy * sx;
        }
    }
    if (fp[FP_CONTACT] > (wr)0.5 && !travel && (vx != 0 || vy != 0 || vz != 0)) {
        wr h = fp[FP_C_H], ox = fp[FP_C_O], oy = fp[FP_C_O + 1], oz = fp[FP_C_O + 2];
        wr cxp = x, cyp = y, czp = z;
        w_column_point(A, s, cxp, cyp, czp);
        wr d = w_tri(A.CG, A.cx, A.cy, A.cz, ox, oy, oz, h, cxp, cyp, czp, (wr)1e9) - fp[FP_CGAP];
        wr ce = fp[FP_CEPS];
        if (d < ce) {
#define CT(a, b, c) w_tri(A.CG, A.cx, A.cy, A.cz, ox, oy, oz, h, a, b, c, (wr)1e9)
            wr nx = CT(cxp + (wr)0.5 * h, cyp, czp) - CT(cxp - (wr)0.5 * h, cyp, czp);
            wr ny = CT(cxp, cyp + (wr)0.5 * h, czp) - CT(cxp, cyp - (wr)0.5 * h, czp);
            wr nz = CT(cxp, cyp, czp + (wr)0.5 * h) - CT(cxp, cyp, czp - (wr)0.5 * h);
#undef CT
            wr nl = sqrt(nx * nx + ny * ny + nz * nz), vl = sqrt(vx * vx + vy * vy + vz * vz);
            wr app = (nl < (wr)1e-30 || fabs(nx) > (wr)1e8 || fabs(ny) > (wr)1e8 || fabs(nz) > (wr)1e8) ? (wr)1 :
                     fmin(fmax((wr)-4 * (vx * nx + vy * ny + vz * nz) / (vl * nl), (wr)0), (wr)1);
            wr f = 1 - w_ramp(fmax(d, (wr)0) / ce);
            wr kk = 1 - (1 - f) * app;
            vx *= kk; vy *= kk; vz *= kk;
        }
    }
    out[0] = vx; out[1] = vy; out[2] = vz;
}

// phi (inverse false) or phi^-1 (inverse true) of one point: RK4 over the stage rows (3 a step)
__device__ void w_map(const WarpArgs& A, wr& x, wr& y, wr& z, bool inverse) {
    int n = (int)A.fp[FP_NSTEP];
    wr hs = (wr)1 / n;
    wr k1[3], k2[3], k3[3], k4[3];
    for (int q = 0; q < n; q++) {
        int k = inverse ? n - 1 - q : q;
        int s0 = inverse ? 3 * k + 2 : 3 * k, s1 = 3 * k + 1, s2 = inverse ? 3 * k : 3 * k + 2;
        wr h = inverse ? -hs : hs;
        w_velocity(A, s0, x, y, z, k1);
        w_velocity(A, s1, x + (wr)0.5 * h * k1[0], y + (wr)0.5 * h * k1[1], z + (wr)0.5 * h * k1[2], k2);
        w_velocity(A, s1, x + (wr)0.5 * h * k2[0], y + (wr)0.5 * h * k2[1], z + (wr)0.5 * h * k2[2], k3);
        w_velocity(A, s2, x + h * k3[0], y + h * k3[1], z + h * k3[2], k4);
        wr c = h / 6;
        x += c * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        y += c * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        z += c * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
    }
}
#endif
