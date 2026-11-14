"""Build GPU volumes (FV8 brick layout) from a field function on the CPU: the narrow band only.

volume_from_field(fn, lo, hi, h, ...) samples fn (vectorised: (N, 3) local points -> (N,) values) at the 9^3 nodes of
every 8h brick whose centre value says the surface may pass within the band, keeps the bricks that hold |F| < band,
and lays them out as one level of a Volume (tiles of bpt^3 bricks). Used for test scenes and as the first path for
Field Studio objects (the CPU core is the field's authority; this only samples it).
"""
import numpy as np

from .scene import NLEV, Volume

_NODE = np.stack(np.meshgrid(np.arange(9), np.arange(9), np.arange(9), indexing='ij'), -1).reshape(-1, 3).astype(float)


def brick_nodes(origins, h):
    """(N, 3) brick origins -> (N, 729, 3) node positions."""
    return origins[:, None, :] + _NODE[None] * h


def volume_from_field(name, fn, lo, hi, h, band=None, level=2, bpt=4, albedo=(0.66, 0.56, 0.49), chunk=4096,
                      lipschitz=1.0):
    """-> (Volume, bricks (N, 9, 9, 9) fp16). The volume holds one level `level` (its h); the others are absent."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    band = 4.0 * h if band is None else float(band)
    bm = 8.0 * h
    tm = bm * bpt
    nt = np.maximum(np.ceil((hi - lo) / tm).astype(int), 1)
    nb = nt * bpt
    ii = np.stack(np.meshgrid(*[np.arange(n) for n in nb], indexing='ij'), -1).reshape(-1, 3)
    centres = lo + (ii + 0.5) * bm
    fc = np.empty(len(ii))
    for i in range(0, len(ii), 1 << 18):
        fc[i:i + (1 << 18)] = fn(centres[i:i + (1 << 18)])
    keep = np.abs(fc) <= lipschitz * (0.5 * np.sqrt(3) * bm) + band
    ii = ii[keep]
    vals, ijk = [], []
    for i in range(0, len(ii), chunk):
        part = ii[i:i + chunk]
        pts = brick_nodes(lo + part * bm, h).reshape(-1, 3)
        v = fn(pts).reshape(len(part), 729)
        ok = np.abs(v).min(1) < band
        vals.append(v[ok])
        ijk.append(part[ok])
    vals = np.concatenate(vals) if vals else np.zeros((0, 729))
    ijk = np.concatenate(ijk) if ijk else np.zeros((0, 3), int)
    vals = np.clip(vals, -band * 1.5, band * 1.5)
    tiles = ijk // bpt
    uniq, tinv = np.unique(tiles, axis=0, return_inverse=True)
    tinv = tinv.reshape(-1)
    grid = np.full(tuple(nt) + (NLEV,), -1, np.int32)
    grid[uniq[:, 0], uniq[:, 1], uniq[:, 2], level] = np.arange(len(uniq), dtype=np.int32)
    ttab = np.full((len(uniq), bpt ** 3), -1, np.int32)
    li = ijk - tiles * bpt
    ttab[tinv, (li[:, 0] * bpt + li[:, 1]) * bpt + li[:, 2]] = np.arange(len(ijk), dtype=np.int32)
    levels = [None] * NLEV
    levels[level] = {'h': float(h), 'band': float(band), 'bpt': int(bpt), 'tile_table': ttab}
    v = Volume(name, lo, tm, tuple(nt), levels, lo, lo + nt * tm, albedo)
    v.grid = grid
    v.brick_ijk = ijk
    v.level = level
    bricks = [None] * NLEV
    bricks[level] = vals.reshape(-1, 9, 9, 9).astype(np.float16)
    return v, bricks


# ---- analytic test shapes (world = local) ---------------------------------------------------------------
def sd_box(c, half, r=0.0):
    c = np.asarray(c, float)
    half = np.asarray(half, float)

    def f(p):
        q = np.abs(p - c) - half + r
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - r
    return f


def sd_sphere(c, rad):
    c = np.asarray(c, float)
    return lambda p: np.linalg.norm(p - c, axis=-1) - rad


