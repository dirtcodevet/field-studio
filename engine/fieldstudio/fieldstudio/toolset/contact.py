"""Contact between objects, read from their fields (exact distances, no meshes).

touching(scene, a, b) -> (gap_half, point): the smallest value over space of max(F_a, F_b), and where. For two
separate objects it is half their gap (at the middle of the closest pair of points); <= 0 where they overlap. Two
objects touch when it is within the tolerance.
"""
import numpy as np

from ..warp.bake import object_field
from .common import world_box

TOUCH_TOL = 1e-4          # m: objects closer than 2 x this touch (Place on surface leaves gap 0 +- 0.1 mm)


def touching(scene, a, b, tol=TOUCH_TOL, n=40):
    """-> (m, point): m = min over x of max(F_a(x), F_b(x)) (world m), point where; m = +inf when their boxes are
    farther apart than the search reach"""
    la, ha = world_box(a)
    lb, hb = world_box(b)
    reach = 4.0 * tol + 1e-3
    lo = np.maximum(la, lb) - reach
    hi = np.minimum(ha, hb) + reach
    if np.any(lo > hi):
        return float('inf'), None
    pk = scene.pack()
    fa = object_field(scene, a, pk)
    fb = object_field(scene, b, pk)
    ext = hi - lo
    d = float(max(ext.max() / (n - 1), 1e-5))
    ax = [np.linspace(lo[i], hi[i], max(2, int(np.ceil(ext[i] / d)) + 1)) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    m = np.maximum(fa(G), fb(G))
    order = np.argsort(m)[:6]
    best = (float(m[order[0]]), G[order[0]])
    # refine from the best seeds: a pattern search on max(F_a, F_b) (non-smooth at the bisector; no gradients needed)
    for i0 in order:
        x = G[i0].copy()
        fx = float(max(fa(x[None])[0], fb(x[None])[0]))
        step = d
        dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                         [1, 1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
                         [1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1]], float)
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        while step > tol * 0.05:
            P = x[None, :] + step * dirs
            v = np.maximum(fa(P), fb(P))
            j = int(np.argmin(v))
            if v[j] < fx:
                x, fx = P[j], float(v[j])
            else:
                step *= 0.5
        if fx < best[0]:
            best = (fx, x)
    return best


def touch_graph(scene, oids, tol=TOUCH_TOL):
    """-> (connected: bool, pairs [(a, b, m)] that touch, all pair values)"""
    objs = [scene.get(o) for o in oids]
    pairs, vals = [], {}
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            m, p = touching(scene, objs[i], objs[j], tol)
            vals[(objs[i].id, objs[j].id)] = m
            if m <= tol:
                pairs.append((objs[i].id, objs[j].id, m))
    # connected components over the touch pairs
    comp = {o: o for o in oids}

    def find(x):
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x
    for a, b, _ in pairs:
        comp[find(a)] = find(b)
    roots = {find(o) for o in oids}
    return len(roots) == 1, pairs, vals


def nearest_pair(scene, a, b, iters=12):
    """the closest points of two objects at any distance (alternating projections onto each one's surface through its
    field), and their gap (m): for the cue of objects far apart (touching() searches only near their boxes' overlap)"""
    from .common import world_box
    pk = scene.pack()
    fa = object_field(scene, a, pk)
    fb = object_field(scene, b, pk)

    def proj(f, x):
        for _ in range(6):
            e = 1e-5
            g = np.array([(f((x + e * np.eye(3)[k])[None])[0] - f((x - e * np.eye(3)[k])[None])[0]) / (2 * e)
                          for k in range(3)])
            x = x - f(x[None])[0] * g / max(g @ g, 1e-12)
        return x
    la, ha = world_box(a)
    lb, hb = world_box(b)
    pb = 0.5 * (lb + hb)
    pa = proj(fa, 0.5 * (la + ha))
    for _ in range(iters):
        pb = proj(fb, pa)
        pa = proj(fa, pb)
    return float(np.linalg.norm(pb - pa)), pa, pb
