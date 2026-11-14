"""Hair Studio CPU preview (numpy + numba, Apache-2.0): strands splatted through a pinhole camera as coverage over a
z-buffer - free, no renderer, no GPU. Moved from the the test model HS2 preview (its splat and its colour ramp kept, so the
measures stay comparable).

Each segment is stepped at 0.5 px; its projected area (length x the strand's true diameter at that depth) is added to
its pixel unless the body (the z-buffer) is in front; alpha = 1 - exp(-area); the colour is the coverage-weighted mean.

    project(P, C, f_px, res) -> uv (px, v down), depth (m)          C: a 4 x 4 camera matrix_world (-Z forward, Y up)
    splat(P, radius, colour, C, f_px, res, zb) -> alpha (H, W), colour (H, W, 3)
    splat_front(P, radius, attr, C, f_px, res, zb) -> the front-most strand point's depth and attr per pixel
    melanin_ramp(melp, tang) -> BGR 0..1: melanin 0.70 -> L* 32 (the reference crown), 90 L* per unit, a/b 7/13,
                                a direction shade +-12 % (so locks read) - HS2's ramp
    look_at(loc, tgt) -> C
"""
import cv2
import numba
import numpy as np


def look_at(loc, tgt):
    z = -(np.asarray(tgt, float) - np.asarray(loc, float))
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    y = up - (up @ z) * z
    y /= np.linalg.norm(y)
    M = np.eye(4)
    M[:3, 0] = np.cross(y, z)
    M[:3, 1] = y
    M[:3, 2] = z
    M[:3, 3] = loc
    return M


def project(P, C, f_px, res):
    C = np.asarray(C, np.float64)
    Xc = (np.asarray(P, np.float64) - C[:3, 3]) @ C[:3, :3]
    d = -Xc[:, 2]
    return np.stack([res[0] / 2 + f_px * Xc[:, 0] / d, res[1] / 2 - f_px * Xc[:, 1] / d], 1), d


@numba.njit(cache=True)
def _splat(uv, dep, wpx, col, off, zb, acc, cacc):
    H, W = zb.shape
    n = len(off) - 1
    for s in range(n):
        for k in range(off[s], off[s + 1] - 1):
            x0, y0, x1, y1 = uv[k, 0], uv[k, 1], uv[k + 1, 0], uv[k + 1, 1]
            L = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            m = int(L / 0.5) + 1
            for j in range(m):
                t = (j + 0.5) / m
                x = x0 + t * (x1 - x0)
                y = y0 + t * (y1 - y0)
                xi, yi = int(x), int(y)
                if xi < 0 or yi < 0 or xi >= W or yi >= H:
                    continue
                d = dep[k] + t * (dep[k + 1] - dep[k])
                if d > zb[yi, xi] + 0.0002:
                    continue
                a = (wpx[k] + t * (wpx[k + 1] - wpx[k])) * L / m
                acc[yi, xi] += a
                for c in range(3):
                    cacc[yi, xi, c] += a * col[k, c]


@numba.njit(cache=True)
def _splat_front(uv, dep, attr, off, zb, zf, af):
    H, W = zb.shape
    n = len(off) - 1
    for s in range(n):
        for k in range(off[s], off[s + 1] - 1):
            x0, y0, x1, y1 = uv[k, 0], uv[k, 1], uv[k + 1, 0], uv[k + 1, 1]
            L = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            m = int(L / 0.5) + 1
            for j in range(m):
                t = (j + 0.5) / m
                xi, yi = int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
                if xi < 0 or yi < 0 or xi >= W or yi >= H:
                    continue
                d = dep[k] + t * (dep[k + 1] - dep[k])
                if d > zb[yi, xi] + 0.0002 or d >= zf[yi, xi]:
                    continue
                zf[yi, xi] = d
                af[yi, xi] = attr[k] + t * (attr[k + 1] - attr[k])


def _radii(r_root, r_tip, K):
    t = np.linspace(0, 1, K)
    return (r_root[:, None] * (1 - t) + r_tip[:, None] * t).ravel()


def splat(P, r_root, r_tip, colour, C, f_px, res, zb, chunk=40000):
    n, K = P.shape[:2]
    acc = np.zeros(zb.shape)
    cacc = np.zeros(zb.shape + (3,))
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        m = sl.stop - sl.start
        uv, dep = project(P[sl].reshape(-1, 3), C, f_px, res)
        wpx = 2 * _radii(r_root[sl], r_tip[sl], K) * f_px / dep
        off = (np.arange(m + 1) * K).astype(np.int64)
        _splat(uv, dep, wpx, np.ascontiguousarray(colour[sl].reshape(-1, 3), np.float64), off, zb, acc, cacc)
    al = 1 - np.exp(-acc)
    return al, cacc / np.maximum(acc, 1e-12)[..., None]


def splat_front(P, attr, C, f_px, res, zb, chunk=40000):
    n, K = P.shape[:2]
    zf = np.full(zb.shape, np.inf)
    af = np.full(zb.shape, np.nan)
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        m = sl.stop - sl.start
        uv, dep = project(P[sl].reshape(-1, 3), C, f_px, res)
        off = (np.arange(m + 1) * K).astype(np.int64)
        _splat_front(uv, dep, np.ascontiguousarray(attr[sl].reshape(-1), np.float64), off, zb, zf, af)
    return zf, af


_RAMP = None


def melanin_ramp(melp, tang, light=(0.3, -0.2, 0.93)):
    """HS2's preview colour (BGR 0..1) from melanin per point: L* = 32 + (0.70 - m) 90, a* 7, b* 13; x (0.94 + 0.12
    |tangent . light|)"""
    global _RAMP
    if _RAMP is None:
        _RAMP = np.stack([cv2.cvtColor(np.array([[[l, 7.0, 13.0]]], np.float32), cv2.COLOR_LAB2BGR)[0, 0]
                          for l in np.linspace(5, 80, 76)])
    Lst = 32 + (0.70 - np.asarray(melp, np.float64)) * 90
    c = _RAMP[np.clip(np.round(Lst - 5).astype(int), 0, 75)]
    lv = np.asarray(light, np.float64)
    sh = 1 + 0.12 * np.clip(np.abs(tang @ lv), 0, 1) - 0.06
    return np.clip(c * sh[:, None], 0, 1)
