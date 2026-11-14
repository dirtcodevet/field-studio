"""Measure and section (fs-shell, 2026-11-11; design Appendix D1: the sizes of what is being worked on
at a glance, such as a pant leg, an inseam or a waist for a belt; queue item 5: a field gives these exactly).

Read-only measures of an object's exact field (its sculpt layer included: tools.protected_merge.field_of), world metres:
- section_loops: the closed curves where a plane cuts the object's surface. The field is sampled on the plane, its zero
  contour followed cell by cell and linked into loops, then every vertex is moved onto the zero set itself by Newton
  steps along the in-plane gradient (to ~1e-7 m), so the curve lies on the surface, not on a sampling grid.
- girth: a loop's length two ways: along the surface's outline (every concavity followed) and as a tailor's tape sits
  (the loop's convex hull: the tape bridges the hollows, e.g. the navel or the small of the back).
- tape: the straight distance between two surface points and the distance along the surface: the normal-section
  curve between them (the plane through both points and their mean normal), shortened toward a geodesic by projected
  curve-shortening (each interior point moved to its neighbours' mid-point and back onto the zero set).
Curves only (P14: no triangles, no element meshes)."""
import math

import numpy as np

from .protected_merge import field_of, world_box


class FieldFn:
    """a field given as a function of world points, with its box: measures work on any field, not only a scene
    object's (another engine's store read through its own module, a test field). banded: the function returns NaN
    (or +-cap) away from the surface; section planes then take each unknown sample's sign from the nearest known one,
    as core.region does in 3-D."""

    def __init__(self, f, box, banded=False):
        self.f, self.box, self.banded = f, (np.asarray(box[0], float), np.asarray(box[1], float)), banded

    def get(self, oid):
        return self

    def pack(self):
        return None


def _F(scene, pk, o, P):
    P = np.ascontiguousarray(np.asarray(P, float).reshape(-1, 3))
    if isinstance(scene, FieldFn):
        return np.asarray(scene.f(P), float)
    return field_of(scene, pk, o, P)


def _box(scene, o):
    return scene.box if isinstance(scene, FieldFn) else world_box(o)


def _grad(scene, pk, o, P, h=2e-5):
    P = np.asarray(P, float).reshape(-1, 3)
    g = np.empty_like(P)
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        g[:, a] = (_F(scene, pk, o, P + e) - _F(scene, pk, o, P - e)) / (2 * h)
    return g


def plane_basis(n):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def _newton(scene, pk, o, P, n, iters, max_move, tol=1e-7):
    """F = 0 by Newton steps along the gradient (n given: its part in that plane, so points stay in their plane).
    fs-shell 11-12 (fs-bench-full F6; fs-cloth's F7 finding): the step f / |g|^2 has no bound where the gradient (in the
    plane: where the plane runs almost along the surface, under her chin, where her thighs meet) goes to about 0, and it
    threw points tens to thousands of metres off her (a tape there then resampled that path every 2 mm: +15 GB). Now a
    step is damped: taken only where it brings |F| down, halved up to four times otherwise; a point no step improves
    is left where it is for the remaining iterations; and with max_move (a length, or one per point) no point ends
    farther than that from where it started (the caller's cell: a section's grid step, a refined segment's length).
    tol: a point within 0.1 um of the surface is done (her store's own precision is coarser: going on to 0.01 um, as
    before, only spent evaluations; fs-cloth's whole-body profile 167 -> see t38)."""
    P = np.array(P, float).reshape(-1, 3)
    P0 = P.copy()
    lim = None if max_move is None else np.broadcast_to(np.asarray(max_move, float), (len(P),))
    f = _F(scene, pk, o, P)
    stuck = np.zeros(len(P), bool)
    for _ in range(iters):
        live = np.isfinite(f) & (np.abs(f) >= tol) & ~stuck
        if not live.any():
            break
        idx = np.flatnonzero(live)
        g = _grad(scene, pk, o, P[idx])
        if n is not None:
            g = g - (g @ n)[:, None] * n[None]
        step = (f[idx] / np.maximum((g * g).sum(1), 1e-18))[:, None] * g
        good = np.isfinite(step).all(1)          # a banded field: a step that would leave the band is not taken
        stuck[idx[~good]] = True
        idx, step = idx[good], step[good]
        alpha = np.ones(len(idx))
        for _h in range(5):
            if not len(idx):
                break
            Q = P[idx] - alpha[:, None] * step
            if lim is not None:
                d = Q - P0[idx]
                dl = np.sqrt((d * d).sum(1))
                Q = P0[idx] + d * np.minimum(1.0, lim[idx] / np.maximum(dl, 1e-30))[:, None]
            fq = _F(scene, pk, o, Q)
            better = np.isfinite(fq) & (np.abs(fq) < np.abs(f[idx]))
            P[idx[better]] = Q[better]
            f[idx[better]] = fq[better]
            keep = ~better
            idx, step, alpha = idx[keep], step[keep], 0.5 * alpha[keep]
        stuck[idx] = True                       # no step improved these: they stay
    return P


def project_in_plane(scene, pk, o, P, n, iters=6, max_move=None):
    """move points onto F = 0 along the in-plane gradient (they stay in their plane); damped and bounded (_newton)"""
    n = np.asarray(n, float) / np.linalg.norm(n)
    return _newton(scene, pk, o, P, n, iters, max_move)


def project_to_surface(scene, pk, o, P, iters=6, max_move=None):
    """move points onto F = 0 along the gradient; damped and bounded (_newton)"""
    return _newton(scene, pk, o, P, None, iters, max_move)


def refine_loop(scene, pk, o, W, n, closed, tol=1e-5, passes=12, max_pts=200000, on_tol=None):
    """insert each segment's mid-point (moved onto the surface in the plane) wherever it lies off the chord by more
    than tol: the curve converges into sharp corners (a box's edges) and onto tight bends that the sampling cut short"""
    for _ in range(passes):
        if len(W) > max_pts:
            break
        A = W
        B = np.vstack([W[1:], W[:1]]) if closed else W[1:]
        A = A if closed else W[:-1]
        M = project_in_plane(scene, pk, o, 0.5 * (A + B), n, iters=5,
                             max_move=np.linalg.norm(B - A, axis=1))    # a mid-point stays within its segment's reach
        d = np.linalg.norm(M - 0.5 * (A + B), axis=1)
        add = (d > tol) & np.isfinite(d)
        if on_tol is not None:                  # fs-shell 11-12: only a mid-point that reached the surface goes in
            fm = np.abs(_F(scene, pk, o, M))
            add &= np.isfinite(fm) & (fm <= on_tol)
        if not add.any():
            break
        out = []
        for k in range(len(A)):
            out.append(A[k])
            if add[k]:
                out.append(M[k])
        if not closed:
            out.append(W[-1])
        W = np.array(out)
    return W


def _march(Fg):
    """marching squares on a (ny, nx) grid of values: -> list of segments ((y, x), (y, x)) in fractional grid coords"""
    ny, nx = Fg.shape
    segs = []
    s = Fg < 0

    def interp(a, b, fa, fb):
        t = fa / (fa - fb) if fa != fb else 0.5
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    for j in range(ny - 1):
        for i in range(nx - 1):
            c = (int(s[j, i]) | (int(s[j, i + 1]) << 1) | (int(s[j + 1, i + 1]) << 2) | (int(s[j + 1, i]) << 3))
            if c in (0, 15):
                continue
            p = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
            f = [Fg[j, i], Fg[j, i + 1], Fg[j + 1, i + 1], Fg[j + 1, i]]
            e = {}
            for k in range(4):
                a, b = k, (k + 1) % 4
                if (f[a] < 0) != (f[b] < 0):
                    e[k] = interp(p[a], p[b], f[a], f[b])
            ks = sorted(e)
            if len(ks) == 2:
                segs.append(((j, i, ks[0]), e[ks[0]], (j, i, ks[1]), e[ks[1]]))
            elif len(ks) == 4:                     # a saddle: split by the cell's centre value
                fc = sum(f) / 4.0
                pairs = ((0, 1), (2, 3)) if (fc < 0) == (f[0] < 0) else ((0, 3), (1, 2))
                for a, b in pairs:
                    segs.append(((j, i, a), e[a], (j, i, b), e[b]))
    return segs


def _march_fast(Fg):
    """marching squares, vectorized: the same segments as _march, with Python work only on the crossing cells"""
    s = Fg < 0
    a, b, c, d = s[:-1, :-1], s[:-1, 1:], s[1:, 1:], s[1:, :-1]
    code = a.astype(np.int8) | (b.astype(np.int8) << 1) | (c.astype(np.int8) << 2) | (d.astype(np.int8) << 3)
    js, is_ = np.nonzero((code != 0) & (code != 15))
    if not len(js):
        return []
    f0, f1, f2, f3 = Fg[js, is_], Fg[js, is_ + 1], Fg[js + 1, is_ + 1], Fg[js + 1, is_]
    P = [(js, is_), (js, is_ + 1), (js + 1, is_ + 1), (js + 1, is_)]
    F = [f0, f1, f2, f3]
    cross, pt = [], []
    for k in range(4):
        ka, kb = k, (k + 1) % 4
        x = (F[ka] < 0) != (F[kb] < 0)
        den = F[ka] - F[kb]
        t = np.where(np.abs(den) > 0, F[ka] / np.where(den == 0, 1.0, den), 0.5)
        y_ = P[ka][0] + (P[kb][0] - P[ka][0]) * t
        x_ = P[ka][1] + (P[kb][1] - P[ka][1]) * t
        cross.append(x)
        pt.append((y_, x_))
    segs = []
    fc = (f0 + f1 + f2 + f3) / 4.0
    for m in range(len(js)):
        j, i = int(js[m]), int(is_[m])
        ks = [k for k in range(4) if cross[k][m]]
        e = {k: (float(pt[k][0][m]), float(pt[k][1][m])) for k in ks}
        if len(ks) == 2:
            segs.append(((j, i, ks[0]), e[ks[0]], (j, i, ks[1]), e[ks[1]]))
        elif len(ks) == 4:
            pairs = ((0, 1), (2, 3)) if (fc[m] < 0) == (f0[m] < 0) else ((0, 3), (1, 2))
            for a_, b_ in pairs:
                segs.append(((j, i, a_), e[a_], (j, i, b_), e[b_]))
    return segs


def _edge_key(j, i, k):
    """a cell edge's shared identity (so neighbouring cells link)"""
    return {0: ('h', j, i), 1: ('v', j, i + 1), 2: ('h', j + 1, i), 3: ('v', j, i)}[k]


# a refined mid-point within this of the surface is on it (one that converges lands within a micrometre; one stuck
# where the plane runs along the surface stays farther and is not inserted)
ON_SURFACE = 5e-5


def _on_runs(W, on, closed):
    """a curve's runs of points on the surface: [(points, closed)]; the whole curve when every point is on it"""
    if on.all():
        return [(W, closed)]
    if closed:                                  # start just after an off point so no run wraps around the start
        k = int(np.flatnonzero(~on)[0])
        W, on = np.roll(W, -k - 1, axis=0), np.roll(on, -k - 1)
    out, i, n = [], 0, len(W)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i >= 2:
                out.append((W[i:j], False))
            i = j
        else:
            i += 1
    return out


def section_loops(scene, oid, point, normal, res=0.0008, margin=0.01, max_n=700, refine=True, window=None):
    """-> list of loops: each {'pts' (N, 3) world points ON the surface, 'closed', 'length'} for the plane through
    `point` with `normal`, cutting object oid's surface. refine: move each loop into its sharp corners (False: leave
    that to the caller, for the one loop it measures). window: (half-width in-plane, m) around the point instead of the
    object's whole box (a live girth on a large body)"""
    o = scene.get(oid)
    pk = scene.pack()
    n, u, v = plane_basis(normal)
    c = np.asarray(point, float)
    lo, hi = _box(scene, o)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - c
    us, vs = corners @ u, corners @ v
    u0, u1 = us.min() - margin, us.max() + margin
    v0, v1 = vs.min() - margin, vs.max() + margin
    if window is not None:
        wu0, wu1, wv0, wv1 = ((-float(window), float(window), -float(window), float(window))
                              if np.isscalar(window) else [float(x) for x in window])
        u0, u1 = max(u0, wu0), min(u1, wu1)
        v0, v1 = max(v0, wv0), min(v1, wv1)
    h = max(res, (u1 - u0) / max_n, (v1 - v0) / max_n)
    nu, nv = int(math.ceil((u1 - u0) / h)) + 1, int(math.ceil((v1 - v0) / h)) + 1
    U, V = np.meshgrid(u0 + np.arange(nu) * h, v0 + np.arange(nv) * h)
    P = c[None] + U.reshape(-1, 1) * u[None] + V.reshape(-1, 1) * v[None]
    Fg = _F(scene, pk, o, P).reshape(nv, nu)
    bad = ~np.isfinite(Fg)
    if bad.any():                          # a banded field: unknown samples take the nearest known sample's sign
        if bad.all():
            return []
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        Fg = Fg.copy()
        Fg[bad] = np.sign(Fg[tuple(ix[bad] for ix in idx)]) * max(float(np.nanmax(np.abs(Fg))), 1e-3)
    segs = _march_fast(Fg)
    if not segs:
        return []
    # link the segments through their shared edges into polylines
    by_edge = {}
    pts = {}
    for k, (a_, pa, b_, pb) in enumerate(segs):
        ka, kb = _edge_key(*a_), _edge_key(*b_)
        pts[ka], pts[kb] = pa, pb
        by_edge.setdefault(ka, []).append((k, kb))
        by_edge.setdefault(kb, []).append((k, ka))
    used = set()
    loops = []
    for k0 in range(len(segs)):
        if k0 in used:
            continue
        a_, pa, b_, pb = segs[k0]
        start, cur = _edge_key(*a_), _edge_key(*b_)
        chain = [start, cur]
        used.add(k0)
        closed = False
        while True:
            nxt = [(k, e) for k, e in by_edge.get(cur, []) if k not in used]
            if not nxt:
                break
            k, e = nxt[0]
            used.add(k)
            if e == start:
                closed = True
                break
            chain.append(e)
            cur = e
        uv = np.array([pts[e] for e in chain])                # (y, x) fractional grid coordinates
        W = c[None] + (u0 + uv[:, 1:2] * h) * u[None] + (v0 + uv[:, 0:1] * h) * v[None]
        W = project_in_plane(scene, pk, o, W, n, max_move=1.5 * h)      # the sampled curve lies within a cell
        # fs-shell 11-12: a point the projection leaves farther from the surface than a grid step is not on her (a
        # plane tangent to her over an area, under her jaw: the sampling's contours there run through her store's
        # unknown space, 3 mm off); the curve is split there and only its runs on her are kept. A point a grazing spot
        # left within a step (under the breasts, at the thighs' junction) stays: its loop is a real section
        fW = np.abs(_F(scene, pk, o, W))
        on = np.isfinite(fW) & (fW <= h)
        for W, closed in _on_runs(W, on, closed):
            if refine:
                W = refine_loop(scene, pk, o, W, n, closed, on_tol=ON_SURFACE)
            seg = np.linalg.norm(np.diff(np.vstack([W, W[:1]]) if closed else W, axis=0), axis=1)
            loops.append({'pts': W, 'closed': closed, 'length': float(seg.sum()), 'plane': (c, n, u, v)})
    loops.sort(key=lambda L: -L['length'])
    return loops


def hull_length(loop):
    """the tape's length around a closed loop: its convex hull's perimeter in the loop's plane"""
    c, n, u, v = loop['plane']
    Q = loop['pts'] - c[None]
    p = sorted(set(map(tuple, np.round(np.stack([Q @ u, Q @ v], 1), 9))))
    if len(p) < 3:
        return float(loop['length'])

    def cross(o_, a, b):
        return (a[0] - o_[0]) * (b[1] - o_[1]) - (a[1] - o_[1]) * (b[0] - o_[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = np.array(lower[:-1] + upper[:-1])
    return float(np.linalg.norm(np.diff(np.vstack([hull, hull[:1]]), axis=0), axis=1).sum())


def girth(scene, oid, point, normal=(0.0, 1.0, 0.0), res=0.0008, refine=True, window=None):
    """the loop of the plane through `point` (default horizontal) nearest the point: -> {'pts', 'length' (along the
    outline), 'tape' (the convex hull: as a tape sits), 'closed', 'plane'} or None"""
    loops = [L for L in section_loops(scene, oid, point, normal, res=res, refine=False, window=window)
             if L['closed']]
    if not loops:
        return None
    p = np.asarray(point, float)
    L = min(loops, key=lambda L: float(np.min(np.linalg.norm(L['pts'] - p[None], axis=1))))
    L = dict(L)
    c, n, u, v = L['plane']
    if refine:
        L['pts'] = refine_loop(scene, scene.pack(), scene.get(oid), L['pts'], n, True, on_tol=ON_SURFACE)
    L['length'] = float(np.linalg.norm(np.diff(np.vstack([L['pts'], L['pts'][:1]]), axis=0), axis=1).sum())
    L['tape'] = hull_length(L)
    return L


def _arc_between(loops, a, b, trust_open=False, near=0.003):
    """the section's path from a to b: on the curve that passes within `near` of both, the shorter way round (a closed
    curve) or the way along it (an open one). An open curve was cut by the section's window; its way between the points
    is taken only when the other way round is surely longer (that way runs to the curve's two cut ends and between
    them at least their straight distance); otherwise None (the caller widens the window). trust_open: the curve is
    open for another reason (the object's own box), take the way along it."""
    for L in loops:
        P = L['pts']
        ia = int(np.argmin(np.linalg.norm(P - a[None], axis=1)))
        ib = int(np.argmin(np.linalg.norm(P - b[None], axis=1)))
        if np.linalg.norm(P[ia] - a) > near or np.linalg.norm(P[ib] - b) > near:
            continue
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        i0, i1 = min(ia, ib), max(ia, ib)
        arc1 = P[ia:ib + 1] if ia <= ib else P[ib:ia + 1][::-1]
        l1 = seg[i1] - seg[i0]
        if L['closed']:
            arc2 = np.vstack([P[ib:], P[:ia + 1]])[::-1] if ia <= ib else np.vstack([P[ia:], P[:ib + 1]])
            l2 = (seg[-1] - l1) + float(np.linalg.norm(P[-1] - P[0]))
            return arc1 if l1 <= l2 or len(arc2) < 2 else arc2
        other = (seg[i0] - seg[0]) + (seg[-1] - seg[i1]) + float(np.linalg.norm(P[-1] - P[0]))
        if trust_open or l1 <= other:
            return arc1
        return None
    return None


def tape(scene, oid, a, b, iters=80, spacing=0.002, windowed=True, max_n=700):
    """two surface points -> {'straight', 'surface', 'pts' (the path on the surface)}: the path starts as the normal
    section between them (the plane through both and their mean normal; the shorter arc), then is shortened toward
    a geodesic with every point kept on the zero set. 'section': how the plane was cut (around the points, widened,
    the whole section; or a projected line when no curve of the plane holds both points). windowed=False: the whole
    section from the start (max_n: the section's samples a side; the whole section of a large body at the window's
    grid step needs more)"""
    o = scene.get(oid)
    pk = scene.pack()
    a = project_to_surface(scene, pk, o, a, max_move=0.01)[0]     # picked on the drawn surface: within a pixel's depth
    b = project_to_surface(scene, pk, o, b, max_move=0.01)[0]
    straight = float(np.linalg.norm(b - a))
    if straight < 1e-6:
        return {'straight': 0.0, 'surface': 0.0, 'pts': np.stack([a, b])}
    g = _grad(scene, pk, o, np.stack([a, b]))
    m = g.sum(0)
    m /= max(np.linalg.norm(m), 1e-12)
    ab = (b - a) / straight
    n = np.cross(ab, m)
    if np.linalg.norm(n) < 1e-6:
        n = np.cross(ab, [0.0, 1.0, 0.0] if abs(ab[1]) < 0.9 else [1.0, 0.0, 0.0])
    n /= np.linalg.norm(n)
    # fs-shell 11-12 (F6: a path that bulges far from the line between the points must still be measured
    # whole, never cut short silently): the section is cut only in a window around the two points (0.75 x their
    # distance + 2 cm about the chord's middle), so the cost follows the measured distance, and widened (doubled, up to
    # the object's whole section) while the path between them may leave it: when no curve in the window holds both
    # points, or the curve holding them is cut by the window and its other way round could be the shorter one
    _, pu, pv = plane_basis(n)
    half = 0.5 * (b - a)
    blo, bhi = _box(scene, o)
    span = float(np.linalg.norm(np.asarray(bhi) - np.asarray(blo)))
    reach = (0.75 * straight + 0.02) if windowed else 2.0 * span
    path, widened, whole = None, 0, False
    while path is None:
        whole = reach >= span
        win = None if whole else (float(half @ pu) - reach, float(half @ pu) + reach,
                                  float(half @ pv) - reach, float(half @ pv) + reach)
        # refine=False: the path is resampled every 2 mm and each point put on the surface below; the sections'
        # 10 um corner refinement of every curve in the window was most of the tape's time on her (48 s -> see t38c)
        loops = section_loops(scene, oid, a, n, res=min(0.0008, straight / 40.0), window=win, refine=False,
                              max_n=max_n)
        best = _arc_between(loops, a, b, trust_open=whole)
        if best is not None:
            path = np.vstack([a[None], best[1:-1], b[None]]) if len(best) > 2 else np.stack([a, b])
        elif whole:
            break
        else:
            reach *= 2.0
            widened += 1
    how = 'whole section' if whole else ('around the points' if not widened else 'widened %dx' % widened)
    if path is None:                           # no curve of the plane holds both points: said in the result
        how = 'projected line (no section path between the points)'
        path = np.linspace(a, b, max(3, int(straight / spacing)))
        path = project_to_surface(scene, pk, o, path, max_move=straight)
    # resample evenly, then shorten with the end points fixed
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    k = max(3, int(d[-1] / spacing) + 1)
    s = np.linspace(0.0, d[-1], k)
    path = np.stack([np.interp(s, d, path[:, j]) for j in range(3)], 1)
    for _ in range(iters):
        mid = 0.5 * (path[:-2] + path[2:])
        path[1:-1] = project_to_surface(scene, pk, o, mid, iters=3, max_move=spacing)
    surf = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return {'straight': straight, 'surface': surf, 'pts': path, 'section': how}


# ---- sizes: exact extents and volume ------------------------------------------------------------------------------------
def own_axes(o):
    """the object's own axes in the world: the columns of its rotation (a turned board's length, height, depth)"""
    from ..core import ops
    return np.asarray(ops.rot_matrix(o.transform['rotate']), float)


def extents(scene, oid, n=48, axes=None):
    """-> (lo, hi) (3,) each: the solid's exact extent along three axes (axes: 3 x 3, its columns unit world vectors;
    None = the world's x, y, z: world coordinates as before). The field is sampled over the object's box; from the
    extreme inside sample on each side the point climbs along the surface (its tangent part of the axis, then back onto
    F = 0) until the surface faces the axis: the true extreme, not the sampling's. fs-shell 11-11: with axes =
    own_axes(o) the object's own size however it is turned (the Dimensions default)."""
    A = np.eye(3) if axes is None else np.asarray(axes, float)
    o = scene.get(oid)
    pk = scene.pack()
    blo, bhi = world_box(o)
    blo, bhi = np.asarray(blo, float), np.asarray(bhi, float)
    ax = [np.linspace(blo[k], bhi[k], n) for k in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    F = _F(scene, pk, o, G)
    ins = G[F < 0]
    if not len(ins):
        return None
    lo, hi = np.empty(3), np.empty(3)
    for k in range(3):
        for sgn, out in ((1.0, hi), (-1.0, lo)):
            e = sgn * A[:, k]
            q = project_to_surface(scene, pk, o, ins[np.argmax(sgn * (ins @ A[:, k]))][None])[0]
            big = float(np.max(bhi - blo)) / n
            q_prev = n_prev = None
            for _ in range(60):
                g = _grad(scene, pk, o, q[None])[0]
                nrm = g / max(np.linalg.norm(g), 1e-12)
                t = e - (e @ nrm) * nrm
                tl = float(np.linalg.norm(t))
                if tl < 1e-7:
                    break
                # the step from the surface's curvature along the climb (a sphere: straight to its extreme); flat:
                # one cell; a sharp edge: shrinks onto it
                if q_prev is not None and np.linalg.norm(q - q_prev) > 1e-12:
                    kap = float(np.linalg.norm(nrm - n_prev)) / float(np.linalg.norm(q - q_prev))
                    s_ = min(big, tl / max(kap, 1e-9))
                else:
                    s_ = big * 0.25
                moved = False
                for _try in range(40):
                    cand = project_to_surface(scene, pk, o, (q + s_ * t / tl)[None])[0]
                    if cand @ e > q @ e + 1e-12:
                        q_prev, n_prev = q, nrm
                        q = cand
                        moved = True
                        break
                    s_ *= 0.35
                    if s_ < 1e-9:
                        break
                if not moved:
                    break
            out[k] = q @ A[:, k]
    return lo, hi


def volume(scene, oid, n=96):
    """-> m^3: the solid's volume, the smoothed inside fraction (a Heaviside of half a cell) summed over a grid of
    n^3 cells across the object's box (its error falls with the cell squared on a smooth form; a sharp edge costs
    O(h) along it)"""
    o = scene.get(oid)
    pk = scene.pack()
    blo, bhi = world_box(o)
    blo, bhi = np.asarray(blo, float) - 1e-4, np.asarray(bhi, float) + 1e-4
    h = (bhi - blo) / n
    ax = [blo[k] + (np.arange(n) + 0.5) * h[k] for k in range(3)]
    eps = 0.5 * float(h.min())
    tot = 0.0
    for i in range(n):                       # one slab at a time: bounded memory
        X, Y, Z = np.meshgrid(ax[0][i:i + 1], ax[1], ax[2], indexing='ij')
        P = np.stack([X, Y, Z], -1).reshape(-1, 3)
        F = _F(scene, pk, o, P)
        tot += float(np.clip(0.5 - F / (2.0 * eps), 0.0, 1.0).sum())
    return tot * float(np.prod(h))


# densities (kg/m^3) of common materials: standard engineering and anatomy tables (public facts; P13 does not apply)
DENSITY = {'soft tissue / skin': 1060.0, 'whole human body (average)': 1010.0, 'water': 1000.0, 'pine': 500.0,
           'oak': 750.0, 'maple': 705.0, 'clay (wet)': 1800.0, 'brick': 1900.0, 'concrete': 2400.0, 'granite': 2700.0,
           'marble': 2700.0, 'glass': 2500.0, 'aluminium': 2700.0, 'steel': 7850.0, 'PLA plastic': 1240.0,
           'cotton (fibre)': 1540.0}


def girth_follow(scene, oid, prev, point, res=0.003, margin=0.03):
    """a live girth: the section through `point` in the plane parallel to the previous loop's, sampled only over the
    previous loop's in-plane box plus a margin (a neighbouring arm or leg stays out), vertices exact on F = 0, not
    corner-refined (the release measures it fully). -> girth dict or None"""
    c, n, u, v = prev['plane']
    p = np.asarray(point, float)
    Q = prev['pts'] - p[None]
    win = (float((Q @ u).min()) - margin, float((Q @ u).max()) + margin,
           float((Q @ v).min()) - margin, float((Q @ v).max()) + margin)
    loops = [L for L in section_loops(scene, oid, p, n, res=res, refine=False, window=win) if L['closed']]
    if not loops:
        return None
    # the loop that continues the previous one: the nearest centroid in the plane
    pc = prev['pts'].mean(0)
    L = dict(min(loops, key=lambda L: float(np.linalg.norm(((L['pts'].mean(0) - pc) - ((L['pts'].mean(0) - pc) @ n) * n)))))
    L['tape'] = hull_length(L)
    return L


# ---- girth profiles along any axis (a log, a column, a dragon's neck) -------------------------------------------------
def long_axis(scene, oid, n=40):
    """the form's own long axis: the first principal direction of its inside (the field sampled over its box)"""
    o = scene.get(oid)
    pk = scene.pack()
    lo, hi = _box(scene, o)
    ax = [np.linspace(lo[k], hi[k], n) for k in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    F = _F(scene, pk, o, G)
    ins = G[np.isfinite(F) & (F < 0)]
    if len(ins) < 10:
        return np.array([0.0, 1.0, 0.0])
    c = ins - ins.mean(0)
    w, V = np.linalg.eigh(c.T @ c)
    a = V[:, int(np.argmax(w))]
    k = int(np.argmax(np.abs(a)))
    return a * (1.0 if a[k] > 0 else -1.0)


def girth_profile(scene, oid, axis, seed=None, step=None, res=0.003, max_sections=400):
    """sections perpendicular to `axis` (world) through the form, the loop at `seed` (default: the largest loop across
    the form's middle) followed section by section both ways (girth_follow) until it ends; -> {'axis', 's' (m along
    the axis), 'length', 'tape', 'width' (the loop's largest in-plane extent), 'pts' (each loop), 'ms'}"""
    import time as _t
    t0 = _t.perf_counter()
    o = scene.get(oid)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    lo, hi = _box(scene, o)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    smin, smax = float((corners @ a).min()), float((corners @ a).max())
    if step is None:
        step = max(0.001, (smax - smin) / 200.0)
    if seed is None:
        # the largest loop across the form's middle; a seed plane off any face it might lie on (a column's join)
        cen = corners.mean(0)
        L0 = None
        for f_ in (0.5137, 0.4471, 0.5803, 0.3807, 0.6469):
            seed = cen + (smin + f_ * (smax - smin) - float(cen @ a)) * a
            loops = [L for L in section_loops(scene, oid, seed, a, res=res, refine=False) if L['closed']]
            if loops:
                L0 = max(loops, key=lambda L: L['length'])
                break
        if L0 is None:
            return None
    else:
        seed = np.asarray(seed, float)
        L0 = girth(scene, oid, seed, a, res=res, refine=False)
        if L0 is None:
            return None
    L0 = dict(L0)
    L0['tape'] = hull_length(L0)
    rows = {float(L0['pts'].mean(0) @ a): L0}
    for sgn in (1.0, -1.0):
        prev = L0
        for _ in range(max_sections // 2):
            pc = prev['pts'].mean(0)
            nxt = pc + sgn * step * a
            if not (smin - step < float(nxt @ a) < smax + step):
                break
            g = girth_follow(scene, oid, prev, nxt, res=res)
            if g is None or g['length'] < 1e-4:
                break
            rows[float(g['pts'].mean(0) @ a)] = g
            prev = g
    ss = sorted(rows)
    out = {'axis': a.tolist(), 's': np.array(ss), 'length': np.array([rows[s]['length'] for s in ss]),
           'tape': np.array([rows[s]['tape'] for s in ss]), 'pts': [rows[s]['pts'] for s in ss],
           'width': np.array([float(max(np.ptp(rows[s]['pts'] @ rows[s]['plane'][2]),
                                       np.ptp(rows[s]['pts'] @ rows[s]['plane'][3]))) for s in ss]),
           'step': step, 'ms': (_t.perf_counter() - t0) * 1000.0}
    out['features'] = profile_features(out)
    return out


def profile_features(prof, end_frac=0.06, jump_factor=4.0, rise=0.02):
    """what the tool finds on any profile: its waists (local minima with the girth rising at least `rise` on both
    sides; 'narrowest' the smallest), its bulges (local maxima; 'widest' the largest) and its sudden changes (the slope
    several times its typical value) -> [{'kind', 's', 'tape', 'i'}]. Ends that only taper are not waists."""
    s, g = np.asarray(prof['s']), np.asarray(prof['tape'])
    n = len(s)
    if n < 5:
        return []
    feats = []

    def extrema(sign):
        out = []
        for i in range(1, n - 1):
            if not (sign * g[i] <= sign * g[i - 1] and sign * g[i] <= sign * g[i + 1]):
                continue
            left, right = g[:i], g[i + 1:]
            if sign > 0:                     # a waist: higher on both sides by `rise`
                ok = left.max() >= g[i] * (1 + rise) and right.max() >= g[i] * (1 + rise)
            else:                            # a bulge: lower on both sides, or the form's end beyond it
                ok = (left.min() <= g[i] * (1 - rise) or i <= 2) and (right.min() <= g[i] * (1 - rise) or i >= n - 3)
            if ok and not any(abs(i - j) < 3 for j in out):
                out.append(i)
        return out
    waists = extrema(1.0)
    bulges = extrema(-1.0)
    if waists:
        i = min(waists, key=lambda k: g[k])
        feats.append({'kind': 'narrowest', 'i': int(i), 's': float(s[i]), 'tape': float(g[i])})
        for k in waists:
            if k != i:
                feats.append({'kind': 'waist', 'i': int(k), 's': float(s[k]), 'tape': float(g[k])})
    if bulges:
        i = max(bulges, key=lambda k: g[k])
        feats.append({'kind': 'widest', 'i': int(i), 's': float(s[i]), 'tape': float(g[i])})
        for k in bulges:
            if k != i:
                feats.append({'kind': 'bulge', 'i': int(k), 's': float(s[k]), 'tape': float(g[k])})
    slope = np.abs(np.diff(g) / np.maximum(np.diff(s), 1e-9))
    typ = float(np.median(slope)) + 1e-9
    cand = np.where(slope > jump_factor * typ)[0]
    last = -10
    for j in cand:                          # local peaks of the slope, a few sections apart
        if j - last < 3:
            continue
        lo_, hi_ = max(0, j - 2), min(len(slope), j + 3)
        if slope[j] == slope[lo_:hi_].max():
            feats.append({'kind': 'sudden change', 'i': int(j), 's': float(s[j]), 'tape': float(g[j]),
                          'step_mm': float((g[j + 1] - g[j]) * 1000.0)})
            last = j
    return feats


def section_at(scene, oid, prof, s_val, res=0.0015):
    """the profile's loop re-measured at position s_val along its axis (following the nearest profile section)"""
    a = np.asarray(prof['axis'], float)
    k = int(np.argmin(np.abs(np.asarray(prof['s']) - s_val)))
    prev = {'pts': prof['pts'][k], 'plane': (prof['pts'][k].mean(0), a, *plane_basis(a)[1:])}
    pc = prof['pts'][k].mean(0)
    g = girth_follow(scene, oid, prev, pc + (s_val - float(pc @ a)) * a, res=res)
    if g is None:
        return None
    c, n, u, v = g['plane']
    g['pts'] = refine_loop(scene, scene.pack(), scene.get(oid), g['pts'], n, True, on_tol=ON_SURFACE)
    g['length'] = float(np.linalg.norm(np.diff(np.vstack([g['pts'], g['pts'][:1]]), axis=0), axis=1).sum())
    g['tape'] = hull_length(g)
    return g
