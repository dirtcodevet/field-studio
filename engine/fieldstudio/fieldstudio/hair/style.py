"""Style acts on hair strands (fs-hair, 11-11): comb, cut, length, push-out, settle. Every strand is edited directly, as
Blender's curves sculpt mode does (each brush acts on all the curves under it, children are not a separate layer), so
what the user combs is what they see.

Strands: P (n, k, 3) float32 in one frame (the caller's), roots at P[:, 0] fixed to the skin. Rest segment lengths
SEG (n, k-1). Points past a strand's end are NaN (a cut strand keeps k points: it is resampled over its new length, so
its detail stays). Every act ends with the follow-the-leader pass (lengths restored from the root out, so a comb never
stretches a hair) and the push-out (no point inside the form: the body field as a grid, a fibre radius plus a gap).

The body field is a dense lattice of F (a signed distance, metres) in the strands' frame: BodyGrid (lo, h, A). Built
from Field Studio's scene by the caller (hair/generate.body_grid).

The kernels run on one thread (measured 11-11 20:58 with the other lanes' jobs holding the CPU at 100 %: sampling 1M
points 11 ms on one thread against 29-145 ms with numba's parallel pool at 2-16 threads, and 1k points 0.01 ms against
25-106 ms; the pool's threads wait for the busy cores). The card brushes (card_style.py) are the fast path."""
import math

import numpy as np
import numba as nb


# ---- the body grid ----------------------------------------------------------------------------------------------------
class BodyGrid:
    """the body as lattices of F: a coarse one (lo, h, A) over the hair's whole reach and, where it is set, a FINE one
    over the head (fine = (lo2, h2, A2)), where the hair leaves the skin and the brushes work: a 3 mm lattice misplaces
    thin, curved forms (an ear) by more than the 0.2 mm clearance (fs-hair 21:40, her T3: 565 of 228k points ended
    inside her true field after a comb over the ear, down to 0.63 mm). The kernels read both through packed(): one
    float32 pool and a float64 header [coarse lo 0-2, h 3, dims 4-6, offset 7, fine lo 8-10, h 11, dims 12-14,
    offset 15] (fine h 0: none)."""

    def __init__(self, lo, h, A, fine=None):
        self.lo = np.asarray(lo, np.float64)
        self.h = float(h)
        self.A = np.ascontiguousarray(A, np.float32)
        self.fine = None
        if fine is not None:
            self.set_fine(*fine)
        self._packed = None

    def set_fine(self, lo2, h2, A2):
        self.fine = (np.asarray(lo2, np.float64), float(h2), np.ascontiguousarray(A2, np.float32))
        self._packed = None

    def packed(self):
        if self._packed is None:
            meta = np.zeros(16, np.float64)
            meta[0:3], meta[3] = self.lo, self.h
            meta[4:7] = self.A.shape
            parts = [self.A.reshape(-1)]
            if self.fine is not None:
                lo2, h2, A2 = self.fine
                meta[8:11], meta[11] = lo2, h2
                meta[12:15] = A2.shape
                meta[15] = self.A.size
                parts.append(A2.reshape(-1))
            self._packed = (np.ascontiguousarray(np.concatenate(parts)), meta, 0.0)
        return self._packed

    def sample(self, P):
        P = np.ascontiguousarray(np.asarray(P, np.float64).reshape(-1, 3))
        f = np.empty(len(P))
        g = np.empty((len(P), 3))
        A, meta, h = self.packed()
        _grid_fg(A, meta, h, P, f, g)
        return f, g


def no_grid():
    """the packed arguments for 'no body' (every point far from any form)"""
    return np.zeros(2, np.float32), np.zeros(16, np.float64), 0.0


@nb.njit(nogil=True, cache=True, fastmath=True, inline='always')
def _tri1(A, off, lo0, lo1, lo2, h, nx, ny, nz, x, y, z):
    fx, fy, fz = (x - lo0) / h, (y - lo1) / h, (z - lo2) / h
    if fx < 0 or fy < 0 or fz < 0 or fx > nx - 1.001 or fy > ny - 1.001 or fz > nz - 1.001:
        return False, 1.0, 0.0, 0.0, 0.0
    i, j, k = int(fx), int(fy), int(fz)
    u, v, w = fx - i, fy - j, fz - k
    sy = nz
    sx = ny * nz
    b = off + i * sx + j * sy + k
    c000, c001, c010, c011 = A[b], A[b + 1], A[b + sy], A[b + sy + 1]
    c100, c101, c110, c111 = A[b + sx], A[b + sx + 1], A[b + sx + sy], A[b + sx + sy + 1]
    c00 = c000 + (c100 - c000) * u
    c10 = c010 + (c110 - c010) * u
    c01 = c001 + (c101 - c001) * u
    c11 = c011 + (c111 - c011) * u
    c0 = c00 + (c10 - c00) * v
    c1 = c01 + (c11 - c01) * v
    f = c0 + (c1 - c0) * w
    gx = ((c100 - c000) * (1 - v) + (c110 - c010) * v) * (1 - w) + ((c101 - c001) * (1 - v) + (c111 - c011) * v) * w
    gy = ((c010 - c000) * (1 - u) + (c110 - c100) * u) * (1 - w) + ((c011 - c001) * (1 - u) + (c111 - c101) * u) * w
    gz = (c01 - c00) * (1 - v) + (c11 - c10) * v
    return True, f, gx / h, gy / h, gz / h


@nb.njit(nogil=True, cache=True, fastmath=True, inline='always')
def _tri(A, lo, h, x, y, z):
    """F and its gradient at (x, y, z): the fine lattice where it covers the point, else the coarse (lo is the packed
    header; h is unused, kept for the kernels' signatures). Outside both: far from the form (1 m, no gradient)"""
    if lo[11] > 0.0:
        ok, f, gx, gy, gz = _tri1(A, int(lo[15]), lo[8], lo[9], lo[10], lo[11], int(lo[12]), int(lo[13]),
                                  int(lo[14]), x, y, z)
        if ok:
            return f, gx, gy, gz
    if lo[3] > 0.0:
        ok, f, gx, gy, gz = _tri1(A, int(lo[7]), lo[0], lo[1], lo[2], lo[3], int(lo[4]), int(lo[5]), int(lo[6]),
                                  x, y, z)
        if ok:
            return f, gx, gy, gz
    return 1.0, 0.0, 0.0, 0.0


@nb.njit(nogil=True, cache=True, fastmath=True)
def _grid_fg(A, lo, h, P, f, g):
    for i in nb.prange(P.shape[0]):
        a, b, c, d = _tri(A, lo, h, P[i, 0], P[i, 1], P[i, 2])
        f[i] = a
        g[i, 0] = b
        g[i, 1] = c
        g[i, 2] = d


# ---- follow the leader + push-out (per strand, from the root) ------------------------------------------------------------
@nb.njit(nogil=True, cache=True, fastmath=True, inline='always')
def _ftl_pushout_one(P, SEG, s, A, lo, h, clear, k0):
    k = P.shape[1]
    for i in range(max(k0, 1), k):
        if P[s, i, 0] != P[s, i, 0]:
            break
        L = SEG[s, i - 1]
        dx = P[s, i, 0] - P[s, i - 1, 0]
        dy = P[s, i, 1] - P[s, i - 1, 1]
        dz = P[s, i, 2] - P[s, i - 1, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < 1e-12:
            continue
        x = P[s, i - 1, 0] + dx / d * L
        y = P[s, i - 1, 1] + dy / d * L
        z = P[s, i - 1, 2] + dz / d * L
        if lo[3] > 0.0:
            for it in range(8):                     # 8: a lock wrapped tight over a thin form (an ear) needed more than 3
                f, gx, gy, gz = _tri(A, lo, h, x, y, z)
                if f >= clear:
                    break
                gn = math.sqrt(gx * gx + gy * gy + gz * gz)
                if gn < 1e-6:
                    break
                push = (clear - f) / gn
                x += gx / gn * push
                y += gy / gn * push
                z += gz / gn * push
                # back onto the segment's length sphere (the push may stretch it)
                dx, dy, dz = x - P[s, i - 1, 0], y - P[s, i - 1, 1], z - P[s, i - 1, 2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
                x = P[s, i - 1, 0] + dx / d * L
                y = P[s, i - 1, 1] + dy / d * L
                z = P[s, i - 1, 2] + dz / d * L
            f, gx, gy, gz = _tri(A, lo, h, x, y, z)
            if f < clear and L > 1e-9:
                # pushing and keeping the length fight where the hair crosses a crease (fs-hair 22:4x, her ear): turn
                # the segment from its direction toward the outward normal at the point before it (which is clear)
                # until the point clears; the length stays exact
                f0, nx_, ny_, nz_ = _tri(A, lo, h, P[s, i - 1, 0], P[s, i - 1, 1], P[s, i - 1, 2])
                nn = math.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_)
                if nn > 1e-6:
                    nx_, ny_, nz_ = nx_ / nn, ny_ / nn, nz_ / nn
                    ux = (x - P[s, i - 1, 0]) / L
                    uy = (y - P[s, i - 1, 1]) / L
                    uz = (z - P[s, i - 1, 2]) / L
                    for st in range(1, 11):
                        w = st / 10.0
                        vx, vy, vz = ux * (1 - w) + nx_ * w, uy * (1 - w) + ny_ * w, uz * (1 - w) + nz_ * w
                        vn = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-12
                        x = P[s, i - 1, 0] + vx / vn * L
                        y = P[s, i - 1, 1] + vy / vn * L
                        z = P[s, i - 1, 2] + vz / vn * L
                        f, gx, gy, gz = _tri(A, lo, h, x, y, z)
                        if f >= clear:
                            break
        P[s, i, 0] = x
        P[s, i, 1] = y
        P[s, i, 2] = z


@nb.njit(nogil=True, cache=True, fastmath=True)
def ftl_pushout(P, SEG, sel, A, lo, h, clear):
    """lengths restored from the root out and every point kept clear of the form, for the strands in sel"""
    for j in nb.prange(sel.shape[0]):
        _ftl_pushout_one(P, SEG, sel[j], A, lo, h, clear, 1)


# ---- screen helpers ---------------------------------------------------------------------------------------------------------
@nb.njit(nogil=True, cache=True, fastmath=True, inline='always')
def _proj(C, x, y, z, W, H):
    qx, qy, qz = x - C[0], y - C[1], z - C[2]
    zf = qx * C[3] + qy * C[4] + qz * C[5]
    xr = qx * C[6] + qy * C[7] + qz * C[8]
    yu = qx * C[9] + qy * C[10] + qz * C[11]
    if C[14] > 0.5:
        px, py, t = xr / C[15], yu / C[15], zf + C[16]
    else:
        if zf <= 1e-5:
            return -1e9, -1e9, -1.0
        px, py, t = xr / zf, yu / zf, math.sqrt(qx * qx + qy * qy + qz * qz)
    sx = (px / (C[12] * C[13]) + 1.0) * 0.5 * W - 0.5
    sy = (1.0 - py / C[12]) * 0.5 * H - 0.5
    return sx, sy, t


@nb.njit(nogil=True, cache=True, fastmath=True, inline='always')
def _visible(depth, sx, sy, t, band):
    H, W = depth.shape
    ix, iy = int(sx + 0.5), int(sy + 0.5)
    if ix < 0 or iy < 0 or ix >= W or iy >= H:
        return False
    return t <= depth[iy, ix] + band


# ---- comb: the points under the brush follow the drag, roots stay ---------------------------------------------------------
@nb.njit(nogil=True, cache=True, fastmath=True)
def comb_dab(P, SEG, Pw, M, Minv, C, W, H, depth, cx, cy, rpx, dlt, strength, band, A, lo, h, clear, touched, sel,
             root_m=0.0):
    """Pw: the strands in world (n, k, 3) (read for the screen test); P: the same in the strands' frame (edited).
    M / Minv: 4x4 world <- frame and frame <- world. dlt: the drag in world (3,) at the brush's depth.
    A point within rpx of the brush centre (screen) and in front of the form moves by dlt x falloff x strength
    (its root's distance along the strand also lowers the weight near the root: the first 2 % of a strand barely
    moves, as a comb pulls a hair from its root's lie). Then lengths and clearance from the root out."""
    n, k = P.shape[0], P.shape[1]
    # dlt in the frame (a direction: no translation)
    dx = Minv[0, 0] * dlt[0] + Minv[0, 1] * dlt[1] + Minv[0, 2] * dlt[2]
    dy = Minv[1, 0] * dlt[0] + Minv[1, 1] * dlt[1] + Minv[1, 2] * dlt[2]
    dz = Minv[2, 0] * dlt[0] + Minv[2, 1] * dlt[1] + Minv[2, 2] * dlt[2]
    r2 = rpx * rpx
    for j in nb.prange(sel.shape[0]):
        s = sel[j]
        hit = False
        first = k
        arc = 0.0
        for i in range(1, k):
            x = Pw[s, i, 0]
            if x != x:
                break
            arc += SEG[s, i - 1]
            sx, sy, t = _proj(C, x, Pw[s, i, 1], Pw[s, i, 2], W, H)
            ex, ey = sx - cx, sy - cy
            d2 = ex * ex + ey * ey
            if d2 > r2 or t < 0:
                continue
            if not _visible(depth, sx, sy, t, band):
                continue
            q = math.sqrt(d2) / rpx
            fo = (1.0 - q * q) * (1.0 - q * q)                 # smooth falloff
            if root_m > 0.0:
                # the follicle holds the first root_m along its own direction; the comb bends the hair over the next
                # root_m (a clamped fibre turns over a few mm), so the roots keep covering the skin at a part
                q2 = (arc - root_m) / root_m
                ramp = 0.0 if q2 <= 0.0 else (1.0 if q2 >= 1.0 else q2 * q2 * (3.0 - 2.0 * q2))
                if ramp <= 0.0:
                    continue
            else:
                ramp = min(1.0, i / (0.06 * k))                   # near the root a hair turns, it does not slide
            wgt = fo * strength * ramp
            P[s, i, 0] += dx * wgt
            P[s, i, 1] += dy * wgt
            P[s, i, 2] += dz * wgt
            if i < first:
                first = i
            hit = True
        if hit:
            touched[s] = 1
            # the points past the brush follow (a comb drags the whole lower strand along, not only what it touches)
            _ftl_pushout_one(P, SEG, s, A, lo, h, clear, first)


# ---- cut: each strand under the brush ends where it first enters the brush (blunt, as scissors leave it) ----------------
@nb.njit(nogil=True, cache=True, fastmath=True)
def cut_dab(Pw, C, W, H, depth, cx, cy, rpx, band, min_pts, cut_at, sel, A, lo, h, min_off):
    """cut_at (n,) float: the arc index (fractional) where the strand is cut, k-1 = uncut; lowered here, never raised.
    Scissors close on the hair in front: a point is cut only in front of the form (or with no form behind it) and at
    least min_off (m) off the skin, so the hair lying against the scalp behind the lengths is not cut to stubble"""
    n, k = Pw.shape[0], Pw.shape[1]
    r2 = rpx * rpx
    for j in nb.prange(sel.shape[0]):
        s = sel[j]
        for i in range(min_pts, k):
            x = Pw[s, i, 0]
            if x != x or i > cut_at[s]:
                break
            sx, sy, t = _proj(C, x, Pw[s, i, 1], Pw[s, i, 2], W, H)
            ex, ey = sx - cx, sy - cy
            if ex * ex + ey * ey > r2 or t <= 0 or not _visible(depth, sx, sy, t, band):
                continue
            if lo[3] > 0.0:
                f, gx, gy, gz = _tri(A, lo, h, x, Pw[s, i, 1], Pw[s, i, 2])
                if f < min_off:
                    continue
            if True:
                if float(i) < cut_at[s]:
                    cut_at[s] = float(i)
                break


@nb.njit(nogil=True, cache=True, fastmath=True)
def _resample_cut(P, RAD, COL, SEG, idx, cut_at, blunt):
    n, k = P.shape[0], P.shape[1]
    for j in nb.prange(idx.shape[0]):
        s = idx[j]
        kk = 0
        for i in range(k):
            if P[s, i, 0] != P[s, i, 0]:
                break
            kk += 1
        if kk < 2:
            continue
        c = min(float(cut_at[s]), kk - 1.0)
        if c < 1.0:
            c = 1.0
        arc = np.empty(kk)
        arc[0] = 0.0
        for i in range(1, kk):
            arc[i] = arc[i - 1] + SEG[s, i - 1]
        i0 = int(math.floor(c))
        fr = c - i0
        Lnew = arc[i0] + (SEG[s, i0] * fr if i0 < kk - 1 else 0.0)
        oldP = P[s, :kk].copy()
        oldR = RAD[s, :kk].copy()
        oldC = COL[s, :kk].copy()
        rmax = 0.0
        for i in range(kk):
            rmax = max(rmax, oldR[i])
        # the shaft's radius at mid-length (the scissors leave the end at the shaft's width)
        q = 0
        while q < kk - 2 and arc[q + 1] < Lnew * 0.5:
            q += 1
        w = (Lnew * 0.5 - arc[q]) / max(arc[q + 1] - arc[q], 1e-12)
        shaft = min(oldR[q] + (oldR[q + 1] - oldR[q]) * min(max(w, 0.0), 1.0), rmax)
        q = 0
        for i in range(k):
            t = Lnew * i / (k - 1)
            while q < kk - 2 and arc[q + 1] < t:
                q += 1
            w = (t - arc[q]) / max(arc[q + 1] - arc[q], 1e-12)
            w = min(max(w, 0.0), 1.0)
            for a in range(3):
                P[s, i, a] = oldP[q, a] + (oldP[q + 1, a] - oldP[q, a]) * w
                COL[s, i, a] = oldC[q, a] + (oldC[q + 1, a] - oldC[q, a]) * w
            r = oldR[q] + (oldR[q + 1] - oldR[q]) * w
            if blunt and t > Lnew * 0.5:
                r = max(r, shaft)
            RAD[s, i] = r
        for i in range(k - 1):
            SEG[s, i] = Lnew / (k - 1)


def resample_cut(P, RAD, COL, SEG, cut_at, blunt=True):
    """strands with cut_at < k-1 are shortened to that arc index and resampled to k points over the new length.
    blunt: the radius stays the shaft's to the end (a scissor cut), not the natural taper."""
    n, k = P.shape[:2]
    idx = np.where(cut_at < k - 1 - 1e-6)[0].astype(np.int64)
    if not len(idx):
        return 0
    _resample_cut(P, RAD, COL, SEG, idx, np.asarray(cut_at, np.float64), bool(blunt))
    return len(idx)


def length_cut(P, RAD, COL, SEG, max_len, sel=None):
    """every strand (or sel) longer than max_len (m) is cut to it (a straight trim across the lengths)"""
    n, k = P.shape[:2]
    L = np.nansum(SEG, axis=1)
    cut_at = np.full(n, float(k - 1))
    over = L > max_len
    if sel is not None:
        m = np.zeros(n, bool)
        m[sel] = True
        over &= m
    for s in np.where(over)[0]:
        arc = np.concatenate([[0.0], np.cumsum(SEG[s].astype(np.float64))])
        i = int(np.searchsorted(arc, max_len) - 1)
        i = min(max(i, 0), k - 2)
        cut_at[s] = i + (max_len - arc[i]) / max(SEG[s, i], 1e-12)
    return resample_cut(P, RAD, COL, SEG, cut_at)


# ---- settle: gravity, lengths, bend toward the rest shape, contact with friction (position-based, a few dozen steps) ----
@nb.njit(nogil=True, cache=True, fastmath=True)
def settle(P, SEG, REST, sel, g, iters, bend, A, lo, h, clear, mu):
    """P edited in place for the strands sel. REST (n, k, 3): the shape the hair holds (its set); bend 0..1 how strongly
    each segment's direction returns to the rest's (relative to its parent segment: a stiff fibre keeps its curl and the
    comb's set); g (3,) the gravity step per iteration in the frame (m); mu: friction on the form (the tangential slip
    of a point in contact is cancelled up to mu x the normal correction)."""
    k = P.shape[1]
    for j in nb.prange(sel.shape[0]):
        s = sel[j]
        for it in range(iters):
            for i in range(1, k):
                if P[s, i, 0] != P[s, i, 0]:
                    break
                ox, oy, oz = P[s, i, 0], P[s, i, 1], P[s, i, 2]
                # gravity
                x, y, z = ox + g[0], oy + g[1], oz + g[2]
                # bend: toward the rest segment direction, carried on the current parent
                rx = REST[s, i, 0] - REST[s, i - 1, 0]
                ry = REST[s, i, 1] - REST[s, i - 1, 1]
                rz = REST[s, i, 2] - REST[s, i - 1, 2]
                tx = P[s, i - 1, 0] + rx
                ty = P[s, i - 1, 1] + ry
                tz = P[s, i - 1, 2] + rz
                x += (tx - x) * bend
                y += (ty - y) * bend
                z += (tz - z) * bend
                # length
                L = SEG[s, i - 1]
                dx, dy, dz = x - P[s, i - 1, 0], y - P[s, i - 1, 1], z - P[s, i - 1, 2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
                x = P[s, i - 1, 0] + dx / d * L
                y = P[s, i - 1, 1] + dy / d * L
                z = P[s, i - 1, 2] + dz / d * L
                # contact with friction
                f, gx, gy, gz = _tri(A, lo, h, x, y, z)
                if f < clear:
                    gn = math.sqrt(gx * gx + gy * gy + gz * gz)
                    if gn > 1e-6:
                        nx_, ny_, nz_ = gx / gn, gy / gn, gz / gn
                        pn = clear - f
                        x += nx_ * pn
                        y += ny_ * pn
                        z += nz_ * pn
                        # friction: the step's tangential part, from the old position
                        sx_, sy_, sz_ = x - ox, y - oy, z - oz
                        sn = sx_ * nx_ + sy_ * ny_ + sz_ * nz_
                        tx_, ty_, tz_ = sx_ - sn * nx_, sy_ - sn * ny_, sz_ - sn * nz_
                        tl = math.sqrt(tx_ * tx_ + ty_ * ty_ + tz_ * tz_)
                        if tl > 1e-12:
                            keep = max(0.0, 1.0 - mu * pn / tl)
                            x = ox + sn * nx_ + tx_ * keep
                            y = oy + sn * ny_ + ty_ * keep
                            z = oz + sn * nz_ + tz_ * keep
                P[s, i, 0] = x
                P[s, i, 1] = y
                P[s, i, 2] = z
        _ftl_pushout_one(P, SEG, s, A, lo, h, clear, 1)


# ---- more style brushes (as Blender's curves sculpting: Puff, Pinch / clump, Smooth), each length-keeping ----------------
@nb.njit(nogil=True, cache=True, fastmath=True, inline='always')
def _hit_weight(C, W, H, depth, cx, cy, rpx, band, x, y, z):
    sx, sy, t = _proj(C, x, y, z, W, H)
    if t < 0:
        return 0.0
    ex, ey = sx - cx, sy - cy
    d2 = ex * ex + ey * ey
    if d2 > rpx * rpx or not _visible(depth, sx, sy, t, band):
        return 0.0
    q = math.sqrt(d2) / rpx
    return (1.0 - q * q) * (1.0 - q * q)


@nb.njit(nogil=True, cache=True, fastmath=True)
def puff_dab(P, SEG, NRM, C, W, H, depth, cx, cy, rpx, strength, band, A, lo, h, clear, touched, sel):
    """lift: strands whose root is under the brush turn toward their root's normal, most at the root, fading over the
    first 40 % of the length (volume at the roots, as a puff or a root lift)"""
    n, k = P.shape[0], P.shape[1]
    for j in nb.prange(sel.shape[0]):
        s = sel[j]
        w = _hit_weight(C, W, H, depth, cx, cy, rpx, band, P[s, 0, 0], P[s, 0, 1], P[s, 0, 2])
        if w <= 0.0:
            continue
        kk = 0
        for i in range(k):
            if P[s, i, 0] != P[s, i, 0]:
                break
            kk += 1
        for i in range(1, kk):
            f = 1.0 - (i - 1) / max(0.4 * (kk - 1), 1.0)
            if f <= 0.0:
                break
            a = w * strength * 0.25 * f
            dx = P[s, i, 0] - P[s, i - 1, 0]
            dy = P[s, i, 1] - P[s, i - 1, 1]
            dz = P[s, i, 2] - P[s, i - 1, 2]
            L = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
            dx = dx / L + NRM[s, 0] * a
            dy = dy / L + NRM[s, 1] * a
            dz = dz / L + NRM[s, 2] * a
            L2 = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
            # move this point and carry the rest of the strand rigidly with it
            ox = P[s, i - 1, 0] + dx / L2 * SEG[s, i - 1] - P[s, i, 0]
            oy = P[s, i - 1, 1] + dy / L2 * SEG[s, i - 1] - P[s, i, 1]
            oz = P[s, i - 1, 2] + dz / L2 * SEG[s, i - 1] - P[s, i, 2]
            for q in range(i, kk):
                P[s, q, 0] += ox
                P[s, q, 1] += oy
                P[s, q, 2] += oz
        touched[s] = 1
        _ftl_pushout_one(P, SEG, s, A, lo, h, clear, 1)


@nb.njit(nogil=True, cache=True, fastmath=True)
def _clump_targets(P, C, W, H, depth, cx, cy, rpx, band, acc, cnt, sel):
    n, k = P.shape[0], P.shape[1]
    for j in range(sel.shape[0]):
        s = sel[j]
        for i in range(k):
            x = P[s, i, 0]
            if x != x:
                break
            w = _hit_weight(C, W, H, depth, cx, cy, rpx, band, x, P[s, i, 1], P[s, i, 2])
            if w > 0:
                acc[i, 0] += x * w
                acc[i, 1] += P[s, i, 1] * w
                acc[i, 2] += P[s, i, 2] * w
                cnt[i] += w


@nb.njit(nogil=True, cache=True, fastmath=True)
def _clump_apply(P, SEG, C, W, H, depth, cx, cy, rpx, band, strength, acc, cnt, A, lo, h, clear, touched, sel):
    n, k = P.shape[0], P.shape[1]
    for j in nb.prange(sel.shape[0]):
        s = sel[j]
        hit = False
        for i in range(1, k):
            x = P[s, i, 0]
            if x != x:
                break
            if cnt[i] <= 0:
                continue
            w = _hit_weight(C, W, H, depth, cx, cy, rpx, band, x, P[s, i, 1], P[s, i, 2])
            if w <= 0:
                continue
            f = w * strength * 0.3 * min(1.0, i / (0.25 * k))        # a lock gathers toward the ends, not the roots
            for a in range(3):
                P[s, i, a] += (acc[i, a] / cnt[i] - P[s, i, a]) * f
            hit = True
        if hit:
            touched[s] = 1
            _ftl_pushout_one(P, SEG, s, A, lo, h, clear, 1)


def clump_dab(P, SEG, C, W, H, depth, cx, cy, rpx, strength, band, A, lo, h, clear, touched, sel):
    """pinch: the points under the brush draw toward the brush's mean hair at the same point along the strand (a lock
    forming, as water, oil or product binds neighbours)"""
    k = P.shape[1]
    acc = np.zeros((k, 3))
    cnt = np.zeros(k)
    _clump_targets(P, C, W, H, depth, cx, cy, rpx, band, acc, cnt, sel)
    _clump_apply(P, SEG, C, W, H, depth, cx, cy, rpx, band, strength, acc, cnt, A, lo, h, clear, touched, sel)


@nb.njit(nogil=True, cache=True, fastmath=True)
def smooth_dab(P, SEG, C, W, H, depth, cx, cy, rpx, strength, band, A, lo, h, clear, touched, sel):
    """smooth: each point under the brush moves toward the mean of its neighbours along the strand (kinks relax)"""
    n, k = P.shape[0], P.shape[1]
    for j in nb.prange(sel.shape[0]):
        s = sel[j]
        hit = False
        for i in range(1, k - 1):
            x = P[s, i, 0]
            if x != x or P[s, i + 1, 0] != P[s, i + 1, 0]:
                break
            w = _hit_weight(C, W, H, depth, cx, cy, rpx, band, x, P[s, i, 1], P[s, i, 2])
            if w <= 0:
                continue
            f = w * strength * 0.5
            for a in range(3):
                P[s, i, a] += (0.5 * (P[s, i - 1, a] + P[s, i + 1, a]) - P[s, i, a]) * f
            hit = True
        if hit:
            touched[s] = 1
            _ftl_pushout_one(P, SEG, s, A, lo, h, clear, 1)


# ---- the stroke's candidates: each strand's box on screen, so a dab tests only strands near the brush ----------------------
@nb.njit(nogil=True, cache=True, fastmath=True)
def strand_boxes(P, C, W, H, sel, box):
    """box (n, 4): x0, x1, y0, y1 of each strand's points on screen (frame px), for the strands in sel; a strand
    behind the camera gets an empty box"""
    k = P.shape[1]
    for j in nb.prange(sel.shape[0]):
        s = sel[j]
        x0, x1, y0, y1 = 1e9, -1e9, 1e9, -1e9
        for i in range(k):
            x = P[s, i, 0]
            if x != x:
                break
            sx, sy, t = _proj(C, x, P[s, i, 1], P[s, i, 2], W, H)
            if t < 0:
                continue
            x0 = min(x0, sx)
            x1 = max(x1, sx)
            y0 = min(y0, sy)
            y1 = max(y1, sy)
        box[s, 0], box[s, 1], box[s, 2], box[s, 3] = x0, x1, y0, y1
