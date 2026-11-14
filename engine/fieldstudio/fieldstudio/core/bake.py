"""An object's field evaluated at many points (the CPU core's program), for the card mirror and the tools.
(The band bake that wrote these values into a tagged band store is not part of this release.)"""
import math

import numpy as np
from numba import njit, prange

from . import program as P
BR = 8                                   # nodes per brick axis


@njit(cache=True, nogil=True, parallel=True)
def eval_local_many(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts, out):
    n = pts.shape[0]
    nch = 64
    for c in prange(nch):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        for m in range(c * n // nch, (c + 1) * n // nch):
            out[m] = P.eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts[m, 0], pts[m, 1], pts[m, 2],
                                  vs, ps)


@njit(cache=True, nogil=True, parallel=True)
def eval_brick_nodes(code, prm, oi_tab, of_tab, index_flat, delta, oi, keys, h, out):
    """out (M,8,8,8) = F at the nodes of global bricks keys (M,3)."""
    for n in prange(keys.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        bx = keys[n, 0] * BR
        by = keys[n, 1] * BR
        bz = keys[n, 2] * BR
        for i in range(BR):
            for j in range(BR):
                for k in range(BR):
                    out[n, i, j, k] = P.eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                                   (bx + i) * h, (by + j) * h, (bz + k) * h, vs, ps)


def object_field_fn(scene, oi):
    pk = scene.pack()

    def f(pts):
        pts = np.ascontiguousarray(pts, np.float64).reshape(-1, 3)
        out = np.empty(len(pts))
        eval_local_many(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, pts, out)
        return out
    return f, float(pk.of_tab[oi, P.OF_L])


