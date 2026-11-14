"""Hair in the viewport (fs-hair, 11-11): every strand drawn into the frame on the card, as curves (P14: no meshes).

The viewport's frame (ui/viewport.py) is the field tracer's picture with a per-pixel ray distance (`depth`). Hair is
composited into it as a viewport overlay (the grids' pattern: `vp.overlays`, `active()` / `composite(fr, rgb)`), so the
viewport's own code is unchanged.

How a strand is drawn (three launches, each far under the display watchdog's ~1 s):
1. NEAREST: each segment is walked in screen space one pixel at a time along its major axis (Wu's two-pixel split on
   the minor axis); a sample in front of the form (ray distance < the frame's depth, minus a band) keeps the nearest
   hair distance per pixel (atomicMin on the float's bits).
2. GATHER: the same walk; samples within a few fibre widths of the nearest hair add their colour and their coverage.
   Coverage is the fibre's projected width in pixels (true size: a 60 um fibre at 0.25 mm a pixel covers 24 %), so
   density reads as it is: a thin patch shows the form through it, a full head does not.
3. RESOLVE: the frame's colour under the hair's transmittance exp(-sum coverage), plus the hair's mean colour.

Shading is a working light, not the review renderer's (the look comes from engine/hair/fibre_layer in the review
render): a Kajiya-Kay diffuse term and two shifted highlights (the white R toward the root, the coloured TRT toward the
tip), lit by the viewport's sun and a soft fill from the camera. Colour per point comes from the strand's pigment.

LOD: past a strand budget the draw takes every s-th strand and scales its coverage by s, so the expected transmittance
(and so the density read) is unchanged while orbiting.
"""
import math
import time

import numpy as np

_SRC = r'''
extern "C" {

__device__ __forceinline__ bool project(const float* C, float x, float y, float z, float W, float H,
                                        float& sx, float& sy, float& t)
{
    // C: pos 0-2, f 3-5, r 6-7-8, u 9-11, th 12, aspect 13, ortho 14, D 15, B 16
    float qx = x - C[0], qy = y - C[1], qz = z - C[2];
    float zf = qx * C[3] + qy * C[4] + qz * C[5];
    float xr = qx * C[6] + qy * C[7] + qz * C[8];
    float yu = qx * C[9] + qy * C[10] + qz * C[11];
    float px, py;
    if (C[14] > 0.5f) {
        px = xr / C[15]; py = yu / C[15]; t = zf + C[16];
        if (t <= 1e-4f) return false;
    } else {
        if (zf <= 1e-4f) return false;
        px = xr / zf; py = yu / zf; t = sqrtf(qx * qx + qy * qy + qz * qz);
    }
    sx = (px / (C[12] * C[13]) + 1.0f) * 0.5f * W - 0.5f;
    sy = (1.0f - py / C[12]) * 0.5f * H - 0.5f;
    return true;
}

__device__ __forceinline__ float strand_u(int s, int salt)
{
    // a strand's own number in [-1, 1] (its cuticle tilt and twist differ from its neighbours')
    unsigned int h = (unsigned int)s * 2654435761u ^ (unsigned int)salt * 2246822519u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return (h & 0xFFFFFF) / 8388607.5f - 1.f;
}

__device__ __forceinline__ float sh9(const float* S, float x, float y, float z)
{
    // the tracer's sh_irr (trace.cu), one channel: irradiance on a surface facing (x, y, z) (world, unit)
    float b[9] = {0.282095f, 0.488603f * y, 0.488603f * z, 0.488603f * x, 1.092548f * x * y, 1.092548f * y * z,
                  0.315392f * (3.0f * z * z - 1.0f), 1.092548f * x * z, 0.546274f * (x * x - y * y)};
    float r = 0.f;
    for (int i = 0; i < 9; i++) r += S[i] * b[i];
    return fmaxf(r, 0.f);
}

__device__ __forceinline__ void shade(const float* C, const float* L, const float* tan_, const float* base, float tipf,
                                      float* out, int s, const float* pos, const float* ks, const float* ss, const float* bk)
{
    // ks: the key light's transmittance through the hair to this point (its three lobes); ss: the sky's (fs-hair 00:2x:
    // the shadow scales the light, not the albedo)
    // view vector toward the camera (perspective: from the point; the caller passes the camera's -f for ortho)
    float vx = -C[3], vy = -C[4], vz = -C[5];
    float tx = tan_[0], ty = tan_[1], tz = tan_[2];
    float tl = tx * L[0] + ty * L[1] + tz * L[2];
    float tv = tx * vx + ty * vy + tz * vz;
    float dif = sqrtf(fmaxf(0.f, 1.f - tl * tl));
    // fill light from the camera (keeps the hair readable where the sun does not reach)
    float fill = sqrtf(fmaxf(0.f, 1.f - tv * tv));
    // Marschner-style shifted cones: R shifted toward the root, TRT toward the tip (cuticle tilt ~3 deg)
    // shifted by the cuticle tilt (about 3 deg, R toward the root, TRT toward the tip), each strand its own (+-1 deg)
    float sR = 0.10f + 0.035f * strand_u(s, 1), sT = -0.07f + 0.035f * strand_u(s, 2);
    float aR = tl + sR, aV = tv;
    float cR = sqrtf(fmaxf(0.f, 1.f - aR * aR)) * sqrtf(fmaxf(0.f, 1.f - aV * aV)) - aR * aV;
    float aT = tl + sT;
    float cT = sqrtf(fmaxf(0.f, 1.f - aT * aT)) * sqrtf(fmaxf(0.f, 1.f - aV * aV)) - aT * aV;
    float specR = powf(fmaxf(cR, 0.f), 90.f) * 0.30f;
    float specT = powf(fmaxf(cT, 0.f), 22.f) * 0.55f;
    if (L[5] > 0.5f) {
        // the window's light (one light system): each light's irradiance on the fibre, exposed as the tracer exposes
        // the skin (scene-linear; the resolve tone-maps as the frame is); the sky as the ambient
        int nl = (int)L[0];
        float acc3[3] = {0.f, 0.f, 0.f};
        for (int i = 0; i < nl; i++) {
            const float* li = L + 8 + 6 * i;
            float tli = tx * li[0] + ty * li[1] + tz * li[2];
            float difi = sqrtf(fmaxf(0.f, 1.f - tli * tli));
            float aRi = tli + sR, aTi = tli + sT;
            float cRi = sqrtf(fmaxf(0.f, 1.f - aRi * aRi)) * sqrtf(fmaxf(0.f, 1.f - aV * aV)) - aRi * aV;
            float cTi = sqrtf(fmaxf(0.f, 1.f - aTi * aTi)) * sqrtf(fmaxf(0.f, 1.f - aV * aV)) - aTi * aV;
            float sRi = powf(fmaxf(cRi, 0.f), 90.f) * 0.30f, sTi = powf(fmaxf(cTi, 0.f), 22.f) * 0.55f;
            bool kl = (i == (int)L[6]);
            for (int c = 0; c < 3; c++)
                acc3[c] += (kl ? ks[c] : 1.f) * li[3 + c] * (base[c] * 0.8f * difi + sRi + sTi * base[c] * 1.6f);
        }
        float sky[3] = {L[2], L[3], L[4]};
        if (L[103] > 0.5f) {
            // the sky as the skin sees it: bright above, dim below (the tracer's SH), along the outward direction
            float ox = pos[0] - L[91], oy = pos[1] - L[92], oz = pos[2] - L[93];
            float on = sqrtf(ox * ox + oy * oy + oz * oz) + 1e-12f;
            ox /= on; oy /= on; oz /= on;
            float wx = L[94] * ox + L[95] * oy + L[96] * oz;
            float wy = L[97] * ox + L[98] * oy + L[99] * oz;
            float wz = L[100] * ox + L[101] * oy + L[102] * oz;
            for (int c = 0; c < 3; c++) sky[c] = sh9(L + 64 + 9 * c, wx, wy, wz);
        }
        int kq = (int)L[6];
        for (int c = 0; c < 3; c++) {
            float keyE = (kq >= 0 && kq < nl) ? L[8 + 6 * kq + 3 + c] : 0.f;
            // the multiple scattering (run_hair's backscatter: Bk x dense x (key x vis x 0.5 + sky x vsk x pi/2), HSKY a radiance)
            float ms = bk[c] * (keyE * ks[c] * 0.5f + sky[c] * ss[c] * 0.5f);
            out[c] = L[1] * (acc3[c] + sky[c] * base[c] * 0.6f * ss[c] + ms);
        }
        return;
    }
    float sun = L[3], amb = L[4];
    for (int c = 0; c < 3; c++) {
        float b = base[c];
        out[c] = b * (amb + 0.22f * fill) * ss[c] + sun * ks[c] * (b * 0.8f * dif + specR + specT * b * 1.6f)
                 + bk[c] * (sun * ks[c] * 0.5f + amb * ss[c] * 0.5f);
    }
}

// one segment walk shared by the two passes; mode 0 nearest, 1 gather
__device__ void walk(int mode, const float* C, const float* L, const float* P, const float* RAD, const float* COL,
                     const unsigned short* FLB,
                     int seg, int npt, int stride, float covscale, int W, int H, const float* depth,
                     unsigned int* tmin, float* acc, float band, float pxang, float minvis, float win)
{
    int s = seg / (npt - 1), k = seg % (npt - 1);
    if (s % stride) return;
    int i0 = s * npt + k, i1 = i0 + 1;
    const float* a = P + 3 * i0;
    const float* b = P + 3 * i1;
    if (a[0] != a[0] || b[0] != b[0]) return;           // NaN marks a cut strand's dead tail
    float r0 = RAD[i0], r1 = RAD[i1];
    if (r0 <= 0.f || r1 <= 0.f) return;
    float ax, ay, at, bx, by, bt;
    if (!project(C, a[0], a[1], a[2], (float)W, (float)H, ax, ay, at)) return;
    if (!project(C, b[0], b[1], b[2], (float)W, (float)H, bx, by, bt)) return;
    float dx = bx - ax, dy = by - ay;
    float adx = fabsf(dx), ady = fabsf(dy);
    float n = fmaxf(adx, ady);
    if ((ax < -2 && bx < -2) || (ay < -2 && by < -2) || (ax > W + 1 && bx > W + 1) || (ay > H + 1 && by > H + 1)) return;
    int ns = (int)ceilf(n);
    if (ns < 1) ns = 1;
    if (ns > 4096) return;                               // a segment through the camera
    bool xmaj = adx >= ady;
    float tan_[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    float tl_ = sqrtf(tan_[0] * tan_[0] + tan_[1] * tan_[1] + tan_[2] * tan_[2]) + 1e-12f;
    tan_[0] /= tl_; tan_[1] /= tl_; tan_[2] /= tl_;
    float base[3], col[3], pos[3];
    for (int j = 0; j < ns; j++) {
        float u = (j + 0.5f) / ns;
        float sx = ax + dx * u, sy = ay + dy * u, t = at + (bt - at) * u;
        float rad = r0 + (r1 - r0) * u;
        float cov = 2.f * rad / fmaxf(t * pxang, 1e-9f);
        cov = fminf(fmaxf(cov, minvis), 1.f) * covscale;
        if (mode == 1) {
            int nc = (int)L[7];
            if (nc < 3) nc = 3;
            for (int c = 0; c < 3; c++) base[c] = COL[nc * i0 + c] + (COL[nc * i1 + c] - COL[nc * i0 + c]) * u;
            float ks[3] = {1.f, 1.f, 1.f}, ss[3] = {1.f, 1.f, 1.f}, bk[3] = {0.f, 0.f, 0.f};
            if (L[104] > 0.5f) {
                // run_hair's light (fibre_light.py, hair-shadow's method) at every K-th point (and the tip): the key's
                // and the sky's coloured transmittance and the backscatter, interpolated along the strand
                int K = (int)L[105], nsF = (int)L[106];
                float f = (k + u) / (float)K;
                int q0 = (int)f;
                if (q0 > nsF - 2) q0 = nsF - 2;
                float lo = (float)(q0 * K), hi = fminf((float)((q0 + 1) * K), (float)(npt - 1));
                float w = fminf(fmaxf((k + u - lo) / fmaxf(hi - lo, 1e-6f), 0.f), 1.f);
                const unsigned short* A = FLB + ((size_t)s * nsF + q0) * 9;
                const unsigned short* B = A + 9;
                for (int c = 0; c < 3; c++) {
                    ks[c] = ((float)A[c] + ((float)B[c] - (float)A[c]) * w) * (1.f / 65535.f);
                    ss[c] = ((float)A[3 + c] + ((float)B[3 + c] - (float)A[3 + c]) * w) * (1.f / 65535.f);
                    bk[c] = ((float)A[6 + c] + ((float)B[6 + c] - (float)A[6 + c]) * w) * (1.f / 65535.f);
                }
            } else if (nc >= 5) {
                float k1 = COL[nc * i0 + 3] + (COL[nc * i1 + 3] - COL[nc * i0 + 3]) * u;
                float s1 = COL[nc * i0 + 4] + (COL[nc * i1 + 4] - COL[nc * i0 + 4]) * u;
                for (int c = 0; c < 3; c++) { ks[c] = k1; ss[c] = s1; }
            }
            for (int c = 0; c < 3; c++) pos[c] = a[c] + (b[c] - a[c]) * u;
            shade(C, L, tan_, base, (k + u) / (npt - 1), col, s, pos, ks, ss, bk);
        }
        // a tent footprint on the minor axis: half-width 1 is Wu's split; wider when the LOD thins the strands
        float mi = xmaj ? sy : sx;
        float hw = covscale >= 3.f ? 1.75f : 1.f;
        int ma = (int)floorf((xmaj ? sx : sy) + 0.5f);
        for (int mm = (int)floorf(mi - hw) + 1; mm <= (int)floorf(mi + hw); mm++) {
            float w = (1.f - fabsf((float)mm - mi) / hw) / hw;
            if (w <= 0.f) continue;
            int px = xmaj ? ma : mm, py = xmaj ? mm : ma;
            if (px < 0 || py < 0 || px >= W || py >= H) continue;
            int pi = py * W + px;
            float dep = depth[pi];
            if (t > dep - band) continue;                // behind the form
            if (mode == 0) {
                atomicMin(tmin + pi, __float_as_uint(t));
            } else {
                float tm = __uint_as_float(tmin[pi]);
                float g = __expf(-(t - tm) / win);
                float aa = cov * w;
                float wt = aa * g;
                atomicAdd(acc + 5 * pi + 0, col[0] * wt);
                atomicAdd(acc + 5 * pi + 1, col[1] * wt);
                atomicAdd(acc + 5 * pi + 2, col[2] * wt);
                atomicAdd(acc + 5 * pi + 3, wt);
                atomicAdd(acc + 5 * pi + 4, -__logf(fmaxf(1.f - fminf(aa, 0.995f), 1e-6f)));
            }
        }
    }
}

__global__ void hair_nearest(const float* C, const float* L, const float* P, const float* RAD, int nseg, int npt,
                             int stride, float covscale, int W, int H, const float* depth, unsigned int* tmin,
                             float band, float pxang, float minvis, int seg0)
{
    int g = seg0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nseg) return;
    walk(0, C, L, P, RAD, 0, 0, g, npt, stride, covscale, W, H, depth, tmin, 0, band, pxang, minvis, 1.f);
}

__global__ void hair_gather(const float* C, const float* L, const float* P, const float* RAD, const float* COL,
                            const unsigned short* FLB, int nseg, int npt, int stride, float covscale, int W, int H, const float* depth,
                            unsigned int* tmin, float* acc, float band, float pxang, float minvis, int seg0, float win)
{
    int g = seg0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nseg) return;
    walk(1, C, L, P, RAD, COL, FLB, g, npt, stride, covscale, W, H, depth, tmin, acc, band, pxang, minvis, win);
}

__device__ __forceinline__ float srgb_f(float x) { return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f; }
__device__ __forceinline__ float srgb_inv(float y) { return y <= 0.04045f ? y / 12.92f : powf((y + 0.055f) / 1.055f, 2.4f); }
__device__ __forceinline__ float aces_f(float x) { return fminf(fmaxf((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f), 1.0f); }
__device__ __forceinline__ float aces_inv(float y)
{
    y = fminf(fmaxf(y, 0.f), 0.999f);
    float a = 2.43f * y - 2.51f, b = 0.59f * y - 0.03f, c = 0.14f * y;
    return (-b - sqrtf(fmaxf(b * b - 4.f * a * c, 0.f))) / (2.f * a);
}

__global__ void hair_resolve(const unsigned char* rgb, const float* acc, unsigned char* out, int n, int tonemap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float wsum = acc[5 * i + 3];
    float T = __expf(-acc[5 * i + 4]);
    for (int c = 0; c < 3; c++) {
        float bg = rgb[3 * i + c] / 255.f;
        float hc = wsum > 0.f ? acc[5 * i + c] / wsum : 0.f;
        float v;
        if (tonemap) {
            // the frame is the tracer's ACES fit + sRGB of exposed radiance: undo it, add the hair's exposed radiance
            // over the transmitted background, map again (hair and skin under one exposure and one tone map)
            float x = aces_inv(srgb_inv(bg));
            v = srgb_f(aces_f(x * T + hc * (1.f - T)));
        } else {
            float bgl = bg * bg;                          // ~linear
            v = sqrtf(fmaxf(bgl * T + hc * (1.f - T), 0.f));
        }
        out[3 * i + c] = (unsigned char)fminf(255.f, v * 255.f + 0.5f);
    }
}
}
'''

_MOD = {}
_TILE = {}


def _mod():
    if 'm' not in _MOD:
        import cupy as cp
        m = cp.RawModule(code=_SRC, options=('-std=c++14', '--use_fast_math'))
        _MOD['m'] = m
        _MOD['near'] = m.get_function('hair_nearest')
        _MOD['gather'] = m.get_function('hair_gather')
        _MOD['resolve'] = m.get_function('hair_resolve')
    return _MOD


def cam_vec(cam, w, h):
    """the kernel's camera: pos, f, r, u, tan(fov/2), aspect, ortho flag, D, B"""
    f, r, u = cam.basis()
    th = math.tan(math.radians(cam.fov) * 0.5)
    if getattr(cam, 'ortho', False):
        D, B = cam.ortho_params()
        o = [1.0, D, B]
    else:
        o = [0.0, 0.0, 0.0]
    return np.array(list(cam.pos) + list(f) + list(r) + list(u) + [th, w / float(h)] + o, np.float32)


def pixel_angle(cam, h):
    a = 2.0 * math.tan(math.radians(cam.fov) * 0.5) / h
    if getattr(cam, 'ortho', False):
        D, B = cam.ortho_params()
        a *= D / max(B + D, 1e-9)
    return a


def _dev(cp, a, dtype=np.float32, chunk=1 << 22):
    """host -> card by pageable copies in chunks (ndarray.set: no pinned staging, which failed under a job's memory cap)"""
    a = np.ascontiguousarray(a, dtype).reshape(-1)
    out = cp.empty(a.shape, dtype)
    for i in range(0, a.size, chunk):
        out[i:i + chunk].set(a[i:i + chunk])
    return out


class StrandBuffers:
    """strands on the card: P (n, k, 3) world, RAD (n, k), COL (n, k, 3) linear base colour, or (n, k, 5) with the key's
    and the sky's transmittance through the hair (hair_ui.shaded); uploaded when the key changes"""

    def __init__(self):
        self.key = None
        self.col_key = None
        self.P = self.RAD = self.COL = None
        self.n = self.k = 0
        self.nc = 3
        self.FLB = None                       # run_hair's light at sampled points (uint16 (n, ns, 9)), or None
        self.fl_key = None
        self.fl_K = self.fl_ns = 0

    def set(self, key, P, RAD, COL, rows=None, col_key=None):
        """upload the strands; rows (indices): only those strands changed since the last upload (a stroke's), so a dab
        sends a few MB instead of the whole hair; col_key: the colour array's identity (a new shade re-sends it)"""
        import cupy as cp
        if (key == self.key or getattr(self, 'hold', False)) and self.P is not None:
            return                            # hold: a card stroke is editing the copy here (hair/card_style.py)
        n, k = P.shape[:2]
        nc = int(COL.shape[2]) if np.ndim(COL) == 3 else 3
        same = self.P is not None and (self.n, self.k) == (n, k)
        if same and rows is not None and len(rows) < n // 2:
            r = _dev(cp, np.asarray(rows, np.int64), np.int64)
            self.P.reshape(n, k * 3)[r] = _dev(cp, P[rows]).reshape(len(rows), -1)
            self.RAD.reshape(n, k)[r] = _dev(cp, RAD[rows]).reshape(len(rows), -1)
            if col_key != self.col_key or nc != self.nc:
                self.COL = _dev(cp, COL)
            else:
                self.COL.reshape(n, k * nc)[r] = _dev(cp, COL[rows]).reshape(len(rows), -1)
        else:
            if (self.n, self.k) != (n, k):
                self.clear_light()
            self.n, self.k = n, k
            self.P = _dev(cp, P)
            self.RAD = _dev(cp, RAD)
            self.COL = _dev(cp, COL)
        self.nc = nc
        self.key = key
        self.col_key = col_key

    def set_light(self, key, FLB, K):
        """run_hair's light at every K-th point (and the tip) of each strand: uint16 (n, ns, 9), fixed point / 65535"""
        import cupy as cp
        if key == self.fl_key and self.FLB is not None:
            return
        FLB = np.asarray(FLB)
        if FLB.shape[0] != self.n or FLB.dtype != np.uint16:
            self.clear_light()
            return
        self.FLB = _dev(cp, FLB, np.uint16)
        self.fl_key, self.fl_K, self.fl_ns = key, int(K), int(FLB.shape[1])

    def clear_light(self):
        self.FLB = None
        self.fl_key = None
        self.fl_K = self.fl_ns = 0

    def free(self):
        self.clear_light()
        self.P = self.RAD = self.COL = None
        self.key = None


def draw(buf, cam, W, H, depth, light_dir=(0.35, 0.8, 0.45), sun=1.0, amb=0.18, budget=2_000_000, band_mm=0.35,
         minvis=0.0, tile_segments=1 << 20, stats=None, cov_mult=1.0, win_mm=0.12, keep_on_card=False, lights=None):
    """depth (H, W) ray distance (inf off the form) -> the hair layer (HC (H, W, 3) linear colour, T (H, W)
    transmittance); resolve(rgb, HC, T) composites it.
    budget: segments drawn per frame at most (LOD by strand stride, coverage scaled to keep the density read)."""
    import cupy as cp
    M = _mod()
    t0 = time.perf_counter()
    if buf.P is None or buf.n == 0:
        return np.zeros((H, W, 3), np.float32), np.ones((H, W), np.float32)
    nseg = buf.n * (buf.k - 1)
    stride = max(1, int(math.ceil(nseg / float(budget))))
    C = cp.asarray(cam_vec(cam, W, H))
    Lh = light_array(lights, light_dir, sun, amb)
    Lh[7] = getattr(buf, 'nc', 3)
    flb = getattr(buf, 'FLB', None)
    if flb is not None and lights is not None:
        Lh[104], Lh[105], Lh[106] = 1.0, float(buf.fl_K), float(buf.fl_ns)
    else:
        flb = cp.zeros(9, cp.uint16)
    L = cp.asarray(Lh)
    d = cp.asarray(np.ascontiguousarray(np.where(np.isfinite(depth), depth, 1e30), np.float32).reshape(-1))
    tmin = cp.full(W * H, np.float32(1e30).view(np.uint32), dtype=cp.uint32)
    acc = cp.zeros(W * H * 5, cp.float32)
    pxang = np.float32(pixel_angle(cam, H))
    band = np.float32(band_mm * 1e-3)
    bs = 256
    # tiles timed one by one: a tile past ~150 ms halves the next (up close a segment spans hundreds of pixels; the
    # card also drives the display, and a launch near ~1-2 s resets the driver for every lane)
    tile = int(min(tile_segments, _TILE.get('n', tile_segments)))
    worst = 0.0
    for kern in ('near', 'gather'):
        s0 = 0
        while s0 < nseg:
            n_ = min(tile, nseg - s0)
            t1 = time.perf_counter()
            if kern == 'near':
                M['near'](((n_ + bs - 1) // bs,), (bs,), (C, L, buf.P, buf.RAD, np.int32(s0 + n_), np.int32(buf.k),
                                                           np.int32(stride), np.float32(stride * cov_mult), np.int32(W),
                                                           np.int32(H), d, tmin, band, pxang, np.float32(minvis),
                                                           np.int32(s0)))
            else:
                M['gather'](((n_ + bs - 1) // bs,), (bs,), (C, L, buf.P, buf.RAD, buf.COL, flb, np.int32(s0 + n_),
                                                             np.int32(buf.k), np.int32(stride),
                                                             np.float32(stride * cov_mult), np.int32(W), np.int32(H), d,
                                                             tmin, acc, band, pxang, np.float32(minvis), np.int32(s0),
                                                             np.float32(win_mm * 1e-3)))
            cp.cuda.get_current_stream().synchronize()
            dt = time.perf_counter() - t1
            worst = max(worst, dt)
            if dt > 1.0:
                # fs-shell 11-14 (Tier 0 part 1): a tile past 1 s (half the display watchdog) stops the draw; the next
                # draw starts from the smallest tile (nothing relaunched now)
                cp.cuda.Device().synchronize()
                _TILE['n'] = 4096
                raise RuntimeError('hair draw: a %s tile took %.0f ms (%d segments): stopped, nothing more launched'
                                   % (kern, dt * 1e3, n_))
            s0 += n_
            if dt > 0.15 and tile > 4096:
                tile = max(4096, tile // 2)
            elif dt < 0.03 and tile < tile_segments:
                tile = min(tile_segments, tile * 2)
    _TILE['n'] = tile
    if keep_on_card:
        if stats is not None:
            stats.update(ms=(time.perf_counter() - t0) * 1000.0, segments=nseg, stride=stride, path='card',
                         longest_launch_ms=round(worst * 1000.0, 1), tile=tile)
        return CardLayer(acc, W, H, tonemap=1 if lights is not None else 0)
    a = acc.reshape(H, W, 5)
    wsum = a[..., 3:4]
    HC = cp.where(wsum > 0, a[..., :3] / cp.maximum(wsum, 1e-30), 0.0).astype(cp.float32)
    T = cp.exp(-a[..., 4]).astype(cp.float32)
    HC, T = HC.get(), T.get()
    if stats is not None:
        stats.update(ms=(time.perf_counter() - t0) * 1000.0, segments=nseg, stride=stride, path='card',
                     longest_launch_ms=round(worst * 1000.0, 1), tile=tile)
    return HC, T


class CardLayer:
    """a drawn hair layer kept on the card (the gather's sums): resolve() composites a frame through it there, so only
    the 8-bit frame crosses each way (3 MB at 1080p instead of a 17 MB float layer and a numpy composite)"""

    def __init__(self, acc, W, H, tonemap=0):
        self.acc, self.W, self.H, self.tonemap = acc, W, H, int(tonemap)

    def resolve(self, rgb):
        import cupy as cp
        M = _mod()
        src = cp.asarray(np.ascontiguousarray(rgb).reshape(-1))
        out = cp.empty_like(src)
        n = self.W * self.H
        M['resolve'](((n + 255) // 256,), (256,), (src, self.acc, out, np.int32(n), np.int32(self.tonemap)))
        return out.get().reshape(self.H, self.W, 3)


def light_array(lights=None, light_dir=(0.35, 0.8, 0.45), sun=1.0, amb=0.18):
    """the kernels' L: lights None -> the old working light (v1: one key + ambient + camera fill, display units);
    lights = {'dirs': (n, 3) toward each light, 'rgb': (n, 3) irradiance, 'expo': float, 'sky': (3,)} -> v2
    (scene-linear and exposed; up to 8 lights)"""
    L = np.zeros(128, np.float32)
    if lights is None:
        Ld = np.asarray(light_dir, float)
        Ld = Ld / max(np.linalg.norm(Ld), 1e-12)
        L[:5] = [Ld[0], Ld[1], Ld[2], sun, amb]
        return L
    dirs = np.asarray(lights['dirs'], float).reshape(-1, 3)[:8]
    rgb = np.asarray(lights['rgb'], float).reshape(-1, 3)[:8]
    L[0] = len(dirs)
    L[1] = float(lights.get('expo', 1.0))
    L[6] = float(lights.get('key', 0))           # the light shadowed through the hair (its transmittance in COL[3])
    L[2:5] = np.asarray(lights.get('sky', (0.0, 0.0, 0.0)), float)
    L[5] = 1.0
    for i, (dv, cv) in enumerate(zip(dirs, rgb)):
        dv = dv / max(np.linalg.norm(dv), 1e-12)
        L[8 + 6 * i: 8 + 6 * i + 3] = dv
        L[8 + 6 * i + 3: 8 + 6 * i + 6] = cv
    sh = lights.get('sky_sh')
    if sh is not None and lights.get('centre') is not None:
        L[64:91] = np.asarray(sh, float).reshape(27)
        L[91:94] = np.asarray(lights['centre'], float)
        L[94:103] = np.asarray(lights.get('rot', np.eye(3)), float).reshape(9)
        L[103] = 1.0
    return L


def _aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def _aces_inv(y):
    y = np.clip(y, 0.0, 0.999)
    a, b, c = 2.43 * y - 2.51, 0.59 * y - 0.03, 0.14 * y
    return (-b - np.sqrt(np.maximum(b * b - 4 * a * c, 0.0))) / (2 * a)


def _srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def _srgb_inv(y):
    return np.where(y <= 0.04045, y / 12.92, np.power((y + 0.055) / 1.055, 2.4))


def resolve_tm(rgb, HC, T):
    """the CPU twin of the card's tone-mapped resolve (v2 light)"""
    x = _aces_inv(_srgb_inv(np.asarray(rgb, np.float32) / 255.0))
    v = _srgb(_aces(x * T[..., None] + HC * (1.0 - T[..., None])))
    return np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)


def resolve(rgb, HC, T):
    """the frame under the hair: colour x T + hair colour x (1 - T), in ~linear light"""
    bg = (np.asarray(rgb, np.float32) / 255.0) ** 2
    v = np.sqrt(np.clip(bg * T[..., None] + HC * (1.0 - T[..., None]), 0, None))
    return np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---- the CPU path (no card, or FIELDSTUDIO_GPU=0): the same walk, serial numba; a smaller strand budget ----------------
_NB = {}


def _nb():
    if 'walk' in _NB:
        return _NB
    import numba as nb

    @nb.njit(nogil=True, cache=True, fastmath=True)
    def _project(C, x, y, z, W, H):
        qx, qy, qz = x - C[0], y - C[1], z - C[2]
        zf = qx * C[3] + qy * C[4] + qz * C[5]
        xr = qx * C[6] + qy * C[7] + qz * C[8]
        yu = qx * C[9] + qy * C[10] + qz * C[11]
        if C[14] > 0.5:
            px, py, t = xr / C[15], yu / C[15], zf + C[16]
            if t <= 1e-4:
                return False, 0.0, 0.0, 0.0
        else:
            if zf <= 1e-4:
                return False, 0.0, 0.0, 0.0
            px, py, t = xr / zf, yu / zf, math.sqrt(qx * qx + qy * qy + qz * qz)
        sx = (px / (C[12] * C[13]) + 1.0) * 0.5 * W - 0.5
        sy = (1.0 - py / C[12]) * 0.5 * H - 0.5
        return True, sx, sy, t

    @nb.njit(nogil=True, cache=True, fastmath=True)
    def _walk(mode, C, L, P, RAD, COL, stride, covscale, W, H, depth, tmin, acc, band, pxang, minvis, win, FLB, FK):
        n, k = P.shape[0], P.shape[1]
        col = np.zeros(3)
        ksv = np.ones(3)
        ssv = np.ones(3)
        bkv = np.zeros(3)
        for s in range(0, n, stride):
            for kk in range(k - 1):
                a = P[s, kk]
                b = P[s, kk + 1]
                if a[0] != a[0] or b[0] != b[0]:
                    continue
                r0, r1 = RAD[s, kk], RAD[s, kk + 1]
                if r0 <= 0.0 or r1 <= 0.0:
                    continue
                ok, ax, ay, at = _project(C, a[0], a[1], a[2], W, H)
                if not ok:
                    continue
                ok, bx, by, bt = _project(C, b[0], b[1], b[2], W, H)
                if not ok:
                    continue
                if (ax < -2 and bx < -2) or (ay < -2 and by < -2) or (ax > W + 1 and bx > W + 1) or \
                        (ay > H + 1 and by > H + 1):
                    continue
                dx, dy = bx - ax, by - ay
                nn = max(abs(dx), abs(dy))
                ns = max(1, int(math.ceil(nn)))
                if ns > 4096:
                    continue
                xmaj = abs(dx) >= abs(dy)
                tx, ty, tz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                tl_ = math.sqrt(tx * tx + ty * ty + tz * tz) + 1e-12
                tx, ty, tz = tx / tl_, ty / tl_, tz / tl_
                for j in range(ns):
                    u = (j + 0.5) / ns
                    sx, sy, t = ax + dx * u, ay + dy * u, at + (bt - at) * u
                    rad = r0 + (r1 - r0) * u
                    cov = min(max(2.0 * rad / max(t * pxang, 1e-9), minvis), 1.0) * covscale
                    if mode == 1:
                        vx, vy, vz = -C[3], -C[4], -C[5]
                        tlv = tx * L[0] + ty * L[1] + tz * L[2]
                        tv = tx * vx + ty * vy + tz * vz
                        dif = math.sqrt(max(0.0, 1.0 - tlv * tlv))
                        fill = math.sqrt(max(0.0, 1.0 - tv * tv))
                        hsh = (s * 2654435761) & 0xFFFFFFFF
                        hsh ^= hsh >> 15
                        u1 = (hsh & 0xFFFF) / 32767.5 - 1.0
                        u2 = ((hsh >> 16) & 0xFFFF) / 32767.5 - 1.0
                        aR = tlv + 0.10 + 0.035 * u1
                        cR = math.sqrt(max(0.0, 1 - aR * aR)) * math.sqrt(max(0.0, 1 - tv * tv)) - aR * tv
                        aT = tlv - 0.07 + 0.035 * u2
                        cT = math.sqrt(max(0.0, 1 - aT * aT)) * math.sqrt(max(0.0, 1 - tv * tv)) - aT * tv
                        specR = max(cR, 0.0) ** 90 * 0.30
                        specT = max(cT, 0.0) ** 22 * 0.55
                        for c in range(3):
                            ksv[c] = 1.0
                            ssv[c] = 1.0
                            bkv[c] = 0.0
                        if COL.shape[2] >= 12:
                            for c in range(3):
                                ksv[c] = COL[s, kk, 3 + c] + (COL[s, kk + 1, 3 + c] - COL[s, kk, 3 + c]) * u
                                ssv[c] = COL[s, kk, 6 + c] + (COL[s, kk + 1, 6 + c] - COL[s, kk, 6 + c]) * u
                                bkv[c] = COL[s, kk, 9 + c] + (COL[s, kk + 1, 9 + c] - COL[s, kk, 9 + c]) * u
                        elif COL.shape[2] >= 5:
                            k1 = COL[s, kk, 3] + (COL[s, kk + 1, 3] - COL[s, kk, 3]) * u
                            s1 = COL[s, kk, 4] + (COL[s, kk + 1, 4] - COL[s, kk, 4]) * u
                            for c in range(3):
                                ksv[c] = k1
                                ssv[c] = s1
                        if FK > 0:
                            # the small light buffer (every FK-th point, uint16), as the card's kernel reads it
                            nsF = FLB.shape[1]
                            q0 = int((kk + u) / FK)
                            if q0 > nsF - 2:
                                q0 = nsF - 2
                            lo_ = q0 * FK
                            hi_ = min((q0 + 1) * FK, k - 1)
                            wq = min(max((kk + u - lo_) / max(hi_ - lo_, 1e-6), 0.0), 1.0)
                            for c in range(3):
                                a0 = float(FLB[s, q0, c]); a1 = float(FLB[s, q0 + 1, c])
                                ksv[c] = (a0 + (a1 - a0) * wq) / 65535.0
                                a0 = float(FLB[s, q0, 3 + c]); a1 = float(FLB[s, q0 + 1, 3 + c])
                                ssv[c] = (a0 + (a1 - a0) * wq) / 65535.0
                                a0 = float(FLB[s, q0, 6 + c]); a1 = float(FLB[s, q0 + 1, 6 + c])
                                bkv[c] = (a0 + (a1 - a0) * wq) / 65535.0
                        if L[5] > 0.5:
                            nl = int(L[0])
                            kl = int(L[6])
                            for c in range(3):
                                bc = COL[s, kk, c] + (COL[s, kk + 1, c] - COL[s, kk, c]) * u
                                a3 = 0.0
                                for li in range(nl):
                                    o = 8 + 6 * li
                                    tli = tx * L[o] + ty * L[o + 1] + tz * L[o + 2]
                                    difi = math.sqrt(max(0.0, 1.0 - tli * tli))
                                    aRi = tli + 0.10 + 0.035 * u1
                                    aTi = tli - 0.07 + 0.035 * u2
                                    cRi = math.sqrt(max(0.0, 1 - aRi * aRi)) * math.sqrt(max(0.0, 1 - tv * tv)) - aRi * tv
                                    cTi = math.sqrt(max(0.0, 1 - aTi * aTi)) * math.sqrt(max(0.0, 1 - tv * tv)) - aTi * tv
                                    a3 += (ksv[c] if li == kl else 1.0) * L[o + 3 + c] * (
                                        bc * 0.8 * difi + max(cRi, 0.0) ** 90 * 0.30 + max(cTi, 0.0) ** 22 * 0.55 * bc * 1.6)
                                skyc = L[2 + c]
                                if L[103] > 0.5:
                                    px_ = a[0] + (b[0] - a[0]) * u - L[91]
                                    py_ = a[1] + (b[1] - a[1]) * u - L[92]
                                    pz_ = a[2] + (b[2] - a[2]) * u - L[93]
                                    on = math.sqrt(px_ * px_ + py_ * py_ + pz_ * pz_) + 1e-12
                                    px_, py_, pz_ = px_ / on, py_ / on, pz_ / on
                                    wx = L[94] * px_ + L[95] * py_ + L[96] * pz_
                                    wy = L[97] * px_ + L[98] * py_ + L[99] * pz_
                                    wz = L[100] * px_ + L[101] * py_ + L[102] * pz_
                                    o9 = 64 + 9 * c
                                    skyc = max(0.0, L[o9] * 0.282095 + L[o9 + 1] * 0.488603 * wy + L[o9 + 2] * 0.488603 * wz
                                               + L[o9 + 3] * 0.488603 * wx + L[o9 + 4] * 1.092548 * wx * wy
                                               + L[o9 + 5] * 1.092548 * wy * wz + L[o9 + 6] * 0.315392 * (3 * wz * wz - 1)
                                               + L[o9 + 7] * 1.092548 * wx * wz + L[o9 + 8] * 0.546274 * (wx * wx - wy * wy))
                                keyE = L[8 + 6 * kl + 3 + c] if 0 <= kl < nl else 0.0
                                ms = bkv[c] * (keyE * ksv[c] * 0.5 + skyc * ssv[c] * 0.5)
                                col[c] = L[1] * (a3 + skyc * bc * 0.6 * ssv[c] + ms)
                        else:
                            for c in range(3):
                                bc = COL[s, kk, c] + (COL[s, kk + 1, c] - COL[s, kk, c]) * u
                                col[c] = (bc * (L[4] + 0.22 * fill) * ssv[c] + L[3] * ksv[c] * (bc * 0.8 * dif + specR + specT * bc * 1.6)
                                          + bkv[c] * (L[3] * ksv[c] * 0.5 + L[4] * ssv[c] * 0.5))
                    mi = sy if xmaj else sx
                    hw = 1.75 if covscale >= 3.0 else 1.0
                    ma = int(math.floor((sx if xmaj else sy) + 0.5))
                    for mm in range(int(math.floor(mi - hw)) + 1, int(math.floor(mi + hw)) + 1):
                        w = (1.0 - abs(mm - mi) / hw) / hw
                        if w <= 0.0:
                            continue
                        px = ma if xmaj else mm
                        py = mm if xmaj else ma
                        if px < 0 or py < 0 or px >= W or py >= H:
                            continue
                        if t > depth[py, px] - band:
                            continue
                        if mode == 0:
                            if t < tmin[py, px]:
                                tmin[py, px] = t
                        else:
                            g = math.exp(-(t - tmin[py, px]) / win)
                            aa = cov * w
                            wt = aa * g
                            acc[py, px, 0] += col[0] * wt
                            acc[py, px, 1] += col[1] * wt
                            acc[py, px, 2] += col[2] * wt
                            acc[py, px, 3] += wt
                            acc[py, px, 4] += -math.log(max(1.0 - min(aa, 0.995), 1e-6))

    _NB['walk'] = _walk
    return _NB


def draw_cpu(P, RAD, COL, cam, W, H, depth, light_dir=(0.35, 0.8, 0.45), sun=1.0, amb=0.18, budget=400_000,
             band_mm=0.35, minvis=0.0, stats=None, cov_mult=1.0, win_mm=0.12, lights=None, flb=None, flk=0):
    """the CPU twin of draw(): P (n, k, 3), RAD (n, k), COL (n, k, 3) as arrays on the host"""
    t0 = time.perf_counter()
    P = np.ascontiguousarray(P, np.float32)
    n, k = P.shape[:2]
    if n == 0:
        return np.zeros((H, W, 3), np.float32), np.ones((H, W), np.float32)
    nseg = n * (k - 1)
    stride = max(1, int(math.ceil(nseg / float(budget))))
    C = cam_vec(cam, W, H).astype(np.float64)
    L = light_array(lights, light_dir, sun, amb).astype(np.float64)
    dep = np.ascontiguousarray(np.where(np.isfinite(depth), depth, 1e30), np.float64)
    tmin = np.full((H, W), 1e30)
    acc = np.zeros((H, W, 5))
    pxang = pixel_angle(cam, H)
    band = band_mm * 1e-3
    walk = _nb()['walk']
    R = np.ascontiguousarray(RAD, np.float32)
    Cc = np.ascontiguousarray(COL, np.float32)
    win = win_mm * 1e-3
    if flb is not None and lights is not None and int(flk) > 0:
        FLB, FK = np.ascontiguousarray(flb, np.uint16), int(flk)
    else:
        FLB, FK = np.zeros((1, 2, 9), np.uint16), 0
    walk(0, C, L, P, R, Cc, stride, float(stride * cov_mult), W, H, dep, tmin, acc, band, pxang, minvis, win, FLB, FK)
    walk(1, C, L, P, R, Cc, stride, float(stride * cov_mult), W, H, dep, tmin, acc, band, pxang, minvis, win, FLB, FK)
    T = np.exp(-acc[..., 4]).astype(np.float32)
    HC = np.where(acc[..., 3:4] > 0, acc[..., :3] / np.maximum(acc[..., 3:4], 1e-30), 0.0).astype(np.float32)
    if stats is not None:
        stats.update(ms=(time.perf_counter() - t0) * 1000.0, segments=nseg, stride=stride, path='cpu')
    return HC, T
