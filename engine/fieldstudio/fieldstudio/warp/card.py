"""The warp engine's map on the card (warp.cuh: the same flow as flow.py), with bounded launches.

map_points(w, P, inverse) -> (n, 3): phi or phi^-1 of world points for a built warp (warp/build.Warp). Launches are cut
so none runs past ~40 ms (the driver resets a launch near 2 s on this machine: the measured rule); the first chunk is
small and timed, the rest sized from it. float on the card by default (WREAL), double for the parity check.
parity(w, P) compares the card with the CPU kernels (flow.map_points): the check that keeps the two in step.
A card job runs through engine/core/gpu_lock.py (label fstools-*) unless the window owns the card.
"""
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_MOD = {}
TARGET_MS = 40.0
FIRST = 4096

_SRC = r'''
#include "warp.cuh"
extern "C" __global__ void warp_map(const wr* pts, wr* out, long n, int inverse, const wr* box,
                                    const wr* fp, const wr* img, int imgc, const wr* G, int gx, int gy, int gz,
                                    const wr* CG, int cx, int cy, int cz, const wr* st) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    wr x = pts[3 * i], y = pts[3 * i + 1], z = pts[3 * i + 2];
    if (!(x < box[0] || y < box[1] || z < box[2] || x > box[3] || y > box[4] || z > box[5])) {
        WarpArgs A;
        A.fp = fp; A.img = img; A.imgc = imgc; A.G = G; A.gx = gx; A.gy = gy; A.gz = gz;
        A.CG = CG; A.cx = cx; A.cy = cy; A.cz = cz; A.st = st;
        w_map(A, x, y, z, inverse != 0);
    }
    out[3 * i] = x; out[3 * i + 1] = y; out[3 * i + 2] = z;
}
'''


def _module(dtype):
    import cupy as cp
    key = np.dtype(dtype).name
    m = _MOD.get(key)
    if m is None:
        # the included header's text in the options: CuPy's compile cache keys on the source text and the options only,
        # so an edit to warp.cuh would otherwise run stale, across processes too (fs-tools s2)
        import hashlib
        ih = hashlib.sha1(open(os.path.join(HERE, 'warp.cuh'), 'rb').read()).hexdigest()[:12]
        opts = ('-I' + HERE, '-DWREAL=%s' % ('double' if key == 'float64' else 'float'), '--std=c++14',
                '-DFS_INC_SHA=0x%s' % ih)
        m = _MOD[key] = cp.RawModule(code=_SRC, options=opts)
    return m.get_function('warp_map')


def map_points(w, P, inverse=True, dtype=np.float32):
    import cupy as cp
    P = np.ascontiguousarray(np.asarray(P, float).reshape(-1, 3))
    n = len(P)
    out = np.empty_like(P)
    if n == 0:
        return out
    k = _module(dtype)
    dt = np.dtype(dtype)
    dev = lambda a: cp.asarray(np.ascontiguousarray(a, dt))
    fp, img, G, CG, st, box = dev(w.fp), dev(w.img), dev(w.G), dev(w.CG), dev(w.st), dev(w.box)
    imgc = int(w.img.shape[0])
    gx, gy, gz = (int(v) for v in w.G.shape)
    cx, cy, cz = (int(v) for v in w.CG.shape)
    i = 0
    chunk = FIRST
    while i < n:
        j = min(n, i + chunk)
        dp = dev(P[i:j])
        do = cp.empty_like(dp)
        t0 = time.perf_counter()
        k(((j - i + 127) // 128,), (128,), (dp, do, np.int64(j - i), np.int32(bool(inverse)), box, fp, img,
                                            np.int32(imgc), G, np.int32(gx), np.int32(gy), np.int32(gz), CG,
                                            np.int32(cx), np.int32(cy), np.int32(cz), st))
        cp.cuda.runtime.deviceSynchronize()
        ms = (time.perf_counter() - t0) * 1e3
        out[i:j] = cp.asnumpy(do).astype(float)
        # the next launch sized for TARGET_MS from this one's measured rate (never more than 4x a step)
        per = ms / max(j - i, 1)
        i = j
        chunk = int(min(max(TARGET_MS / max(per, 1e-9), 1024), 4 * chunk, 1 << 22))
    return out


def parity(w, P, dtype=np.float64):
    """max |card - CPU| (m) of phi^-1 over points P"""
    a = map_points(w, P, True, dtype)
    b = w.inverse(P)
    return float(np.abs(a - b).max())
