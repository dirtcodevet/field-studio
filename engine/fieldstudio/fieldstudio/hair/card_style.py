"""The style brushes on the card (fs-hair, 11-11): Comb, Cut, Puff, Smooth, Clump run where the strands already are for
drawing (hair/draw.py StrandBuffers), one thread per strand, so a dab on her 138k hairs x 64 points costs about a
millisecond instead of the CPU's 50-250 ms.

During a stroke the card copy is the working hair: dabs edit it in place, the view draws it as it is (no upload). At
the release the strands the stroke touched come back to the CPU copy (groom.cur), whose rows from before the stroke
are the undo step, so undo, states and save stay on the CPU as before. A cut marks each strand's cut point on the card
(the hair past it stops drawing at once); the exact resample to the new length runs once on the CPU at the release
(style.resample_cut, the same as the CPU path). The rules are style.py's: roots stay, lengths are kept from the root
out (follow-the-leader), no point inside the form (the body lattice, a fibre radius plus 0.2 mm), scissors close on
the hair in front of the form and off the skin. Every launch is one thread per strand (~1 ms on 138k strands)."""
import numpy as np

_SRC = r'''
extern "C" {

__device__ __forceinline__ bool proj(const float* C, float x, float y, float z, float W, float H,
                                     float& sx, float& sy, float& t)
{
    float qx = x - C[0], qy = y - C[1], qz = z - C[2];
    float zf = qx * C[3] + qy * C[4] + qz * C[5];
    float xr = qx * C[6] + qy * C[7] + qz * C[8];
    float yu = qx * C[9] + qy * C[10] + qz * C[11];
    float px, py;
    if (C[14] > 0.5f) { px = xr / C[15]; py = yu / C[15]; t = zf + C[16]; if (t <= 1e-5f) return false; }
    else { if (zf <= 1e-5f) return false; px = xr / zf; py = yu / zf; t = sqrtf(qx * qx + qy * qy + qz * qz); }
    sx = (px / (C[12] * C[13]) + 1.0f) * 0.5f * W - 0.5f;
    sy = (1.0f - py / C[12]) * 0.5f * H - 0.5f;
    return true;
}

__device__ __forceinline__ bool visible(const float* depth, int W, int H, float sx, float sy, float t, float band)
{
    int ix = (int)(sx + 0.5f), iy = (int)(sy + 0.5f);
    if (ix < 0 || iy < 0 || ix >= W || iy >= H) return false;
    return t <= depth[iy * W + ix] + band;
}

// one lattice of the pair: F and its gradient, trilinear; false outside it
__device__ __forceinline__ bool tri1(const float* A, long long off, float lo0, float lo1, float lo2, float h, int nx,
                                     int ny, int nz, float x, float y, float z, float& f, float& gx, float& gy,
                                     float& gz)
{
    float fx = (x - lo0) / h, fy = (y - lo1) / h, fz = (z - lo2) / h;
    if (fx < 0 || fy < 0 || fz < 0 || fx > nx - 1.001f || fy > ny - 1.001f || fz > nz - 1.001f) return false;
    int i = (int)fx, j = (int)fy, k = (int)fz;
    float u = fx - i, v = fy - j, w = fz - k;
    int sy = nz, sx = ny * nz;
    const float* a = A + off + (long long)i * sx + j * sy + k;
    float c000 = a[0], c001 = a[1], c010 = a[sy], c011 = a[sy + 1];
    float c100 = a[sx], c101 = a[sx + 1], c110 = a[sx + sy], c111 = a[sx + sy + 1];
    float c00 = c000 + (c100 - c000) * u, c10 = c010 + (c110 - c010) * u;
    float c01 = c001 + (c101 - c001) * u, c11 = c011 + (c111 - c011) * u;
    float c0 = c00 + (c10 - c00) * v, c1 = c01 + (c11 - c01) * v;
    gx = (((c100 - c000) * (1 - v) + (c110 - c010) * v) * (1 - w) + ((c101 - c001) * (1 - v) + (c111 - c011) * v) * w) / h;
    gy = (((c010 - c000) * (1 - u) + (c110 - c100) * u) * (1 - w) + ((c011 - c001) * (1 - u) + (c111 - c101) * u) * w) / h;
    gz = ((c01 - c00) * (1 - v) + (c11 - c10) * v) / h;
    f = c0 + (c1 - c0) * w;
    return true;
}

// the body lattices (style.BodyGrid.packed): the fine one where it covers the point, else the coarse; outside both:
// far from the form. M: the header [coarse lo 0-2, h 3, dims 4-6, offset 7, fine lo 8-10, h 11, dims 12-14, offset 15]
__device__ __forceinline__ float tri(const float* A, const float* M, float x, float y, float z,
                                     float& gx, float& gy, float& gz)
{
    float f;
    gx = gy = gz = 0.f;
    if (M[11] > 0.f && tri1(A, (long long)M[15], M[8], M[9], M[10], M[11], (int)M[12], (int)M[13], (int)M[14],
                            x, y, z, f, gx, gy, gz)) return f;
    if (M[3] > 0.f && tri1(A, (long long)M[7], M[0], M[1], M[2], M[3], (int)M[4], (int)M[5], (int)M[6],
                           x, y, z, f, gx, gy, gz)) return f;
    gx = gy = gz = 0.f;
    return 1.f;
}

// lengths from the root out and clear of the form, from point k0 on
__device__ void ftl_push(float* p, const float* seg, int k, int k0, const float* A, const float* M,
                         float clear)
{
    for (int i = max(k0, 1); i < k; i++) {
        float* q = p + 3 * i;
        const float* r = p + 3 * (i - 1);
        if (q[0] != q[0]) break;
        float L = seg[i - 1];
        float dx = q[0] - r[0], dy = q[1] - r[1], dz = q[2] - r[2];
        float d = sqrtf(dx * dx + dy * dy + dz * dz);
        if (d < 1e-12f) continue;
        float x = r[0] + dx / d * L, y = r[1] + dy / d * L, z = r[2] + dz / d * L;
        for (int it = 0; it < 8; it++) {              // 8: a lock wrapped tight over a thin form (an ear)
            float gx, gy, gz;
            float f = tri(A, M, x, y, z, gx, gy, gz);
            if (f >= clear) break;
            float gn = sqrtf(gx * gx + gy * gy + gz * gz);
            if (gn < 1e-6f) break;
            float push = (clear - f) / gn;
            x += gx / gn * push; y += gy / gn * push; z += gz / gn * push;
            dx = x - r[0]; dy = y - r[1]; dz = z - r[2];
            d = sqrtf(dx * dx + dy * dy + dz * dz) + 1e-12f;
            x = r[0] + dx / d * L; y = r[1] + dy / d * L; z = r[2] + dz / d * L;
        }
        {
            // still inside where the hair crosses a crease: turn the segment toward the outward normal at the point
            // before it (clear) until it clears; the length stays exact (style.py's rule)
            float gx, gy, gz, nx, ny, nz;
            if (L > 1e-9f && tri(A, M, x, y, z, gx, gy, gz) < clear) {
                tri(A, M, r[0], r[1], r[2], nx, ny, nz);
                float nn = sqrtf(nx * nx + ny * ny + nz * nz);
                if (nn > 1e-6f) {
                    nx /= nn; ny /= nn; nz /= nn;
                    float ux = (x - r[0]) / L, uy = (y - r[1]) / L, uz = (z - r[2]) / L;
                    for (int st = 1; st <= 10; st++) {
                        float w = st / 10.f;
                        float vx = ux * (1 - w) + nx * w, vy = uy * (1 - w) + ny * w, vz = uz * (1 - w) + nz * w;
                        float vn = sqrtf(vx * vx + vy * vy + vz * vz) + 1e-12f;
                        x = r[0] + vx / vn * L; y = r[1] + vy / vn * L; z = r[2] + vz / vn * L;
                        if (tri(A, M, x, y, z, gx, gy, gz) >= clear) break;
                    }
                }
            }
        }
        q[0] = x; q[1] = y; q[2] = z;
    }
}

__device__ __forceinline__ float brush_w(const float* C, int W, int H, const float* depth, float cx, float cy,
                                         float rpx, float band, const float* q)
{
    float sx, sy, t;
    if (!proj(C, q[0], q[1], q[2], (float)W, (float)H, sx, sy, t)) return 0.f;
    float ex = sx - cx, ey = sy - cy, d2 = ex * ex + ey * ey;
    if (d2 > rpx * rpx || !visible(depth, W, H, sx, sy, t, band)) return 0.f;
    float qq = sqrtf(d2) / rpx;
    return (1.f - qq * qq) * (1.f - qq * qq);
}

__global__ void comb_k(float* P, const float* SEG, const float* RAD, int n, int k, const float* C, int W, int H,
                       const float* depth, float cx, float cy, float rpx, float dx, float dy, float dz,
                       float strength, float band, const float* A, const float* M, float clear,
                       unsigned char* touched, const unsigned char* act)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n || !act[s]) return;                      // pinned out of the section
    float* p = P + (long long)s * k * 3;
    int first = k;
    for (int i = 1; i < k; i++) {
        float* q = p + 3 * i;
        if (q[0] != q[0] || RAD[(long long)s * k + i] <= 0.f) break;
        float w = brush_w(C, W, H, depth, cx, cy, rpx, band, q);
        if (w <= 0.f) continue;
        float ramp = fminf(1.f, i / (0.06f * k));
        float a = w * strength * ramp;
        q[0] += dx * a; q[1] += dy * a; q[2] += dz * a;
        if (i < first) first = i;
    }
    if (first < k) {
        touched[s] = 1;
        ftl_push(p, SEG + (long long)s * (k - 1), k, first, A, M, clear);
    }
}

// scissors: the first point (from the root) in the brush, in front of the form and off the skin, marks the cut;
// the hair past it stops drawing at once (radius 0); the CPU resamples it at the release
__global__ void cut_k(const float* P, float* RAD, int n, int k, const float* C, int W, int H, const float* depth,
                      float cx, float cy, float rpx, float band, int min_pts, const float* A, const float* M, float min_off, float* cut_at, unsigned char* touched, const unsigned char* act)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n || !act[s]) return;                      // pinned out of the section
    const float* p = P + (long long)s * k * 3;
    for (int i = min_pts; i < k; i++) {
        const float* q = p + 3 * i;
        if (q[0] != q[0] || (float)i > cut_at[s] || RAD[(long long)s * k + i] <= 0.f) break;
        if (brush_w(C, W, H, depth, cx, cy, rpx, band, q) <= 0.f) continue;
        float gx, gy, gz;
        if (tri(A, M, q[0], q[1], q[2], gx, gy, gz) < min_off) continue;
        cut_at[s] = (float)i;
        touched[s] = 1;
        for (int j = i + 1; j < k; j++) RAD[(long long)s * k + j] = 0.f;
        break;
    }
}

__global__ void puff_k(float* P, const float* SEG, const float* NRM, int n, int k, const float* C, int W, int H,
                       const float* depth, float cx, float cy, float rpx, float strength, float band,
                       const float* A, const float* M, float clear, unsigned char* touched, const unsigned char* act)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n || !act[s]) return;                      // pinned out of the section
    float* p = P + (long long)s * k * 3;
    float w = brush_w(C, W, H, depth, cx, cy, rpx, band, p);
    if (w <= 0.f) return;
    int kk = 0;
    for (int i = 0; i < k; i++) { if (p[3 * i] != p[3 * i]) break; kk++; }
    const float* sg = SEG + (long long)s * (k - 1);
    for (int i = 1; i < kk; i++) {
        float f = 1.f - (i - 1) / fmaxf(0.4f * (kk - 1), 1.f);
        if (f <= 0.f) break;
        float a = w * strength * 0.25f * f;
        float* q = p + 3 * i;
        const float* r = p + 3 * (i - 1);
        float dx = q[0] - r[0], dy = q[1] - r[1], dz = q[2] - r[2];
        float L = sqrtf(dx * dx + dy * dy + dz * dz) + 1e-12f;
        dx = dx / L + NRM[3 * s] * a; dy = dy / L + NRM[3 * s + 1] * a; dz = dz / L + NRM[3 * s + 2] * a;
        float L2 = sqrtf(dx * dx + dy * dy + dz * dz) + 1e-12f;
        float ox = r[0] + dx / L2 * sg[i - 1] - q[0], oy = r[1] + dy / L2 * sg[i - 1] - q[1];
        float oz = r[2] + dz / L2 * sg[i - 1] - q[2];
        for (int j = i; j < kk; j++) { p[3 * j] += ox; p[3 * j + 1] += oy; p[3 * j + 2] += oz; }
    }
    touched[s] = 1;
    ftl_push(p, sg, k, 1, A, M, clear);
}

__global__ void smooth_k(float* P, const float* SEG, int n, int k, const float* C, int W, int H, const float* depth,
                         float cx, float cy, float rpx, float strength, float band, const float* A, const float* M, float clear, unsigned char* touched, const unsigned char* act)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n || !act[s]) return;                      // pinned out of the section
    float* p = P + (long long)s * k * 3;
    bool hit = false;
    for (int i = 1; i < k - 1; i++) {
        float* q = p + 3 * i;
        if (q[0] != q[0] || q[3] != q[3]) break;
        float w = brush_w(C, W, H, depth, cx, cy, rpx, band, q);
        if (w <= 0.f) continue;
        float f = w * strength * 0.5f;
        for (int a = 0; a < 3; a++) q[a] += (0.5f * (q[a - 3] + q[a + 3]) - q[a]) * f;
        hit = true;
    }
    if (hit) { touched[s] = 1; ftl_push(p, SEG + (long long)s * (k - 1), k, 1, A, M, clear); }
}

__global__ void clump_acc_k(const float* P, int n, int k, const float* C, int W, int H, const float* depth,
                            float cx, float cy, float rpx, float band, float* acc, const unsigned char* act)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n || !act[s]) return;                      // pinned out of the section
    const float* p = P + (long long)s * k * 3;
    for (int i = 0; i < k; i++) {
        const float* q = p + 3 * i;
        if (q[0] != q[0]) break;
        float w = brush_w(C, W, H, depth, cx, cy, rpx, band, q);
        if (w <= 0.f) continue;
        atomicAdd(acc + 4 * i, q[0] * w); atomicAdd(acc + 4 * i + 1, q[1] * w);
        atomicAdd(acc + 4 * i + 2, q[2] * w); atomicAdd(acc + 4 * i + 3, w);
    }
}

__global__ void clump_k(float* P, const float* SEG, int n, int k, const float* C, int W, int H, const float* depth,
                        float cx, float cy, float rpx, float strength, float band, const float* acc,
                        const float* A, const float* M, float clear, unsigned char* touched, const unsigned char* act)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n || !act[s]) return;                      // pinned out of the section
    float* p = P + (long long)s * k * 3;
    bool hit = false;
    for (int i = 1; i < k; i++) {
        float* q = p + 3 * i;
        if (q[0] != q[0]) break;
        float cn = acc[4 * i + 3];
        if (cn <= 0.f) continue;
        float w = brush_w(C, W, H, depth, cx, cy, rpx, band, q);
        if (w <= 0.f) continue;
        float f = w * strength * 0.3f * fminf(1.f, i / (0.25f * k));
        for (int a = 0; a < 3; a++) q[a] += (acc[4 * i + a] / cn - q[a]) * f;
        hit = true;
    }
    if (hit) { touched[s] = 1; ftl_push(p, SEG + (long long)s * (k - 1), k, 1, A, M, clear); }
}
}
'''

_MOD = {}


def _mod():
    if 'm' not in _MOD:
        import cupy as cp
        m = cp.RawModule(code=_SRC, options=('-std=c++14', '--use_fast_math'))
        for nm in ('comb_k', 'cut_k', 'puff_k', 'smooth_k', 'clump_acc_k', 'clump_k'):
            _MOD[nm] = m.get_function(nm)
        _MOD['m'] = m
    return _MOD


class CardStroke:
    """one stroke on the card: the groom's strands in the draw buffer (buf) edited in place; finish() brings the touched
    strands back to the CPU copy and returns their indices"""

    def __init__(self, groom, buf, cam_l, W, H, depth_l, tool):
        import cupy as cp
        from .draw import cam_vec
        self.g, self.buf, self.tool, self.W, self.H = groom, buf, tool, int(W), int(H)
        n, k = groom.cur['P'].shape[:2]
        self.n, self.k = n, k
        self.C = cp.asarray(cam_vec(cam_l, W, H))
        d = np.where(np.isfinite(depth_l), depth_l, 1e30).astype(np.float32)
        self.depth = cp.asarray(np.ascontiguousarray(d).reshape(-1))
        self.SEG = cp.asarray(np.ascontiguousarray(groom.cur['SEG'], np.float32).reshape(-1))
        pool, meta, _ = groom._grid_args()
        gk = (id(groom.grid), id(pool))
        if getattr(groom, '_card_grid', None) is None or groom._card_grid[0] != gk:
            groom._card_grid = (gk, cp.asarray(pool.astype(np.float32)), cp.asarray(meta.astype(np.float32)))
        _, self.A, self.M = groom._card_grid
        self.touched = cp.zeros(n, cp.uint8)
        sec = getattr(groom, 'section', None)
        self.act = cp.asarray(sec.astype(np.uint8)) if sec is not None and len(sec) == n else cp.ones(n, cp.uint8)
        self.cut_at = cp.full(n, float(k - 1), cp.float32) if tool == 'cut' else None
        self.NRM = cp.asarray(groom.root_normals().astype(np.float32)) if tool == 'puff' else None
        self.clear = np.float32(groom.clearance())
        self.bs = 256
        self.grid_ = ((n + self.bs - 1) // self.bs,)
        buf.hold = True                        # the view draws the card copy as it is (no upload from the CPU)
        _mod()

    def dab(self, cx, cy, rpx, strength=0.8, drag=(0.0, 0.0, 0.0), band=0.004):
        import cupy as cp
        M = _mod()
        b = self.buf
        f32, i32 = np.float32, np.int32
        common = (self.C, i32(self.W), i32(self.H), self.depth, f32(cx), f32(cy), f32(rpx))
        if self.tool == 'comb':
            M['comb_k'](self.grid_, (self.bs,), (b.P, self.SEG, b.RAD, i32(self.n), i32(self.k)) + common + (
                f32(drag[0]), f32(drag[1]), f32(drag[2]), f32(strength), f32(band), self.A, self.M,
                self.clear, self.touched, self.act))
        elif self.tool == 'cut':
            M['cut_k'](self.grid_, (self.bs,), (b.P, b.RAD, i32(self.n), i32(self.k)) + common + (
                f32(0.002), i32(max(1, int(0.04 * self.k))), self.A, self.M, f32(0.003), self.cut_at,
                self.touched, self.act))
        elif self.tool == 'puff':
            M['puff_k'](self.grid_, (self.bs,), (b.P, self.SEG, self.NRM, i32(self.n), i32(self.k)) + common + (
                f32(strength), f32(band), self.A, self.M, self.clear, self.touched, self.act))
        elif self.tool == 'smooth':
            M['smooth_k'](self.grid_, (self.bs,), (b.P, self.SEG, i32(self.n), i32(self.k)) + common + (
                f32(strength), f32(band), self.A, self.M, self.clear, self.touched, self.act))
        elif self.tool == 'clump':
            acc = cp.zeros(self.k * 4, cp.float32)
            M['clump_acc_k'](self.grid_, (self.bs,), (b.P, i32(self.n), i32(self.k)) + common + (f32(band), acc,
                                                                                                    self.act))
            M['clump_k'](self.grid_, (self.bs,), (b.P, self.SEG, i32(self.n), i32(self.k)) + common + (
                f32(strength), f32(band), acc, self.A, self.M, self.clear, self.touched, self.act))
        cp.cuda.get_current_stream().synchronize()

    def finish(self):
        """the touched strands back to the CPU copy; -> (idx, old rows for undo)"""
        import cupy as cp
        from . import style as ST
        g, b = self.g, self.buf
        t = cp.nonzero(self.touched)[0]
        idx = cp.asnumpy(t).astype(np.int64)
        c = g.cur
        old = {kk: c[kk][idx].copy() for kk in ('P', 'RAD', 'COL', 'SEG')}
        if len(idx):
            if self.tool == 'cut':
                cut_at = np.full(self.n, float(self.k - 1))
                cut_at[idx] = cp.asnumpy(self.cut_at[t]).astype(np.float64)
                ST.resample_cut(c['P'], c['RAD'], c['COL'], c['SEG'], cut_at)
                g.pushout(idx)                     # re-spaced points can sit on chords through a thin form
            else:
                Pk = b.P.reshape(self.n, self.k, 3)[t]
                c['P'][idx] = cp.asnumpy(Pk)
        b.hold = False
        b.key = None                           # the next draw re-sends the hair (the CPU copy is now the truth)
        return idx, old
