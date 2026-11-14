"""True depth under an object's surface (fs-shell, 2026-11-11; the sculpt-exact wall for Cut > Conform).

After sculpting, an object's field F = base + sculpt delta is re-distanced only inside the sculpt's band of bricks:
deeper than that F keeps the base's value, so -F is not the depth under the sculpted surface (probe t24: 8 mm under a
sculpted tip F read -0.90 mm where the true distance was 1.17 mm). A wall 4 mm from the surface needs the true
distance to the surface as sculpted. Here: the surface's points, dense, from the object's full field (grid crossings,
Newton-projected onto F = 0: F's SIGN is right everywhere, so its zero set is the surface), then a k-d tree gives every
lattice node its distance to them. Accuracy: the point spacing s gives an error under s^2 / (8 R) on a curved surface
(R its radius of curvature) plus the projection's 1e-6 m.

  surface_points(scene, oid, step)                    -> (n, 3) world points on the surface (and their mask values)
  depth_grid(scene, oid, step, cap)                   -> {'lo', 'h', 'dims', 'depth'}: the signed true depth at lattice
                                                         nodes (+ inside, - outside), clamped at +-cap (a k-d tree bound)
  conform_lattice(scene, oid, wall, step, openings)   -> the cutter's field on that lattice: wall - depth (< 0 deeper than
                                                         the wall), and, with openings (mask >= 0.5 on the surface),
                                                         min(that, distance to the masked patch - wall * 1.5)"""
import time

import numpy as np


def _crossings(scene, pk, o, lo, hi, step):
    from .protected_merge import field_of
    ax = [np.arange(lo[a], hi[a] + step, step) for a in range(3)]
    pts = []
    # slab by slab along x (memory: one slab of the grid at a time)
    Y, Z = np.meshgrid(ax[1], ax[2], indexing='ij')
    prev = None
    for i, x in enumerate(ax[0]):
        P = np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1)
        f = field_of(scene, pk, o, P).reshape(Y.shape)
        for a in (0, 1):                                   # crossings inside the slab along y and z
            s0 = [slice(None)] * 2
            s1 = [slice(None)] * 2
            s0[a], s1[a] = slice(0, -1), slice(1, None)
            f0, f1 = f[tuple(s0)], f[tuple(s1)]
            m = np.isfinite(f0) & np.isfinite(f1) & ((f0 < 0) != (f1 < 0))
            if m.any():
                PP = P.reshape(Y.shape + (3,))
                t = (f0[m] / (f0[m] - f1[m]))[:, None]
                pts.append(PP[tuple(s0)][m] + t * (PP[tuple(s1)][m] - PP[tuple(s0)][m]))
        if prev is not None:                               # crossings between this slab and the last, along x
            fp, Pp = prev
            m = np.isfinite(fp) & np.isfinite(f) & ((fp < 0) != (f < 0))
            if m.any():
                t = (fp[m] / (fp[m] - f[m]))[:, None]
                A = Pp.reshape(Y.shape + (3,))[m]
                B = P.reshape(Y.shape + (3,))[m]
                pts.append(A + t * (B - A))
        prev = (f, P)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def surface_points(scene, oid, step=0.0005, box=None, newton=4):
    """-> (world points on the object's surface (n, 3), their mask values (n,) or None)"""
    from .protected_merge import field_of, world_box
    o = scene.get(oid)
    pk = scene.pack()
    lo, hi = (np.asarray(v, float) for v in (box or world_box(o)))
    P = _crossings(scene, pk, o, lo - step, hi + step, step)
    for _ in range(newton):                                # onto F = 0 (its sign is right; its zero set is the surface)
        f = field_of(scene, pk, o, P)
        g = np.empty_like(P)
        for a in range(3):
            e = np.zeros(3)
            e[a] = 2e-5
            g[:, a] = (field_of(scene, pk, o, P + e) - field_of(scene, pk, o, P - e)) / 4e-5
        gg = np.maximum((g * g).sum(1), 1e-18)
        stp = (f / gg)[:, None] * g
        ok = np.isfinite(stp).all(1) & (np.linalg.norm(stp, axis=1) < 2 * step)
        P[ok] -= stp[ok]
    N = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    mask = _mask_at(scene, pk, o, P)
    surface_points.normals = N                         # the outward normals at the points (the last Newton gradient)
    return P, mask


def _mask_at(scene, pk, o, P):
    """the object's mask (0-1, its invert flag applied) at world points, or None when it has none"""
    try:
        from ..core import program as PR
        from .protected_merge import _to_local
        if not o.sculpt.slots:
            return None
        oi = scene.index_of(o.id)
        X = _to_local(o, P)
        out = np.empty(len(X))
        for i in range(len(X)):
            out[i] = PR.grid_sample(pk.index_flat, pk.mask, pk.oi_tab, oi, o.sculpt.h, X[i, 0], X[i, 1], X[i, 2])
        out = np.clip(out, 0.0, 1.0)
        if getattr(o, 'mask_inv', False):
            out = 1.0 - out
        return out
    except Exception:
        return None


def depth_grid(scene, oid, step=0.001, cap=0.02, surf_step=None, box=None):
    """the signed true depth (+ inside, - outside) at the lattice nodes of the object's world box, clamped at +-cap"""
    from scipy.spatial import cKDTree
    from .protected_merge import field_of, world_box
    t0 = time.perf_counter()
    o = scene.get(oid)
    pk = scene.pack()
    lo, hi = (np.asarray(v, float) for v in (box or world_box(o)))
    lo, hi = lo - 2 * step, hi + 2 * step
    ss = surf_step or min(step, 0.0005)
    S, mask = surface_points(scene, oid, ss, (lo, hi))
    N = surface_points.normals
    tree = cKDTree(S)
    ax = [np.arange(lo[a], hi[a] + step * 0.5, step) for a in range(3)]
    dims = np.array([len(a) for a in ax])
    depth = np.empty(tuple(dims), np.float32)
    Y, Z = np.meshgrid(ax[1], ax[2], indexing='ij')
    for i, x in enumerate(ax[0]):
        P = np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1)
        f = field_of(scene, pk, o, P)
        d = np.full(len(P), cap)
        # |F| never exceeds the true distance (a Lipschitz-1 field): nodes with |F| >= cap are past the cap already
        q = np.abs(f) < cap
        if q.any():
            dq, j = tree.query(P[q], distance_upper_bound=cap, workers=-1)
            near = np.isfinite(dq) & (dq < 3 * ss)
            if near.any():
                # near the surface the samples' spacing dominates: the distance to the nearest sample's tangent plane
                jj = j[near]
                Pq = P[q]
                dq[near] = np.abs(((Pq[near] - S[jj]) * N[jj]).sum(1))
            d[q] = np.minimum(dq, cap)
        depth[i] = np.where(f < 0, d, -d).reshape(Y.shape)
    return {'lo': lo, 'h': float(step), 'dims': dims, 'depth': depth, 'surface': S, 'mask': mask,
            'ms': (time.perf_counter() - t0) * 1000}


def mask_line(S, mask, spacing):
    """the mask's 0.5 line on the surface as points: interpolated on each masked-unmasked pair of neighbouring samples
    (within 1.5 sample spacings)"""
    from scipy.spatial import cKDTree
    m = np.asarray(mask, float)
    inn = m >= 0.5
    if not (inn.any() and (~inn).any()):
        return np.zeros((0, 3))
    pr = cKDTree(S).query_pairs(1.5 * float(spacing), output_type='ndarray')
    if not len(pr):
        return np.zeros((0, 3))
    a, b = pr[:, 0], pr[:, 1]
    x = inn[a] != inn[b]
    a, b = a[x], b[x]
    t = np.clip((0.5 - m[a]) / np.where(m[b] != m[a], m[b] - m[a], 1.0), 0.0, 1.0)
    return S[a] + t[:, None] * (S[b] - S[a])


def masked_patch(S, mask, spacing):
    """the masked patch as points: the surface samples with mask >= 0.5 plus the mask's 0.5 line between neighbouring
    samples (interpolated on each masked-unmasked pair within 1.5 sample spacings). The opening's rim then follows the
    mask's own line, not which 1 mm sample happened to fall inside it (that left a +-0.5 mm scallop on the rim)"""
    from scipy.spatial import cKDTree
    m = np.asarray(mask, float)
    inn = m >= 0.5
    P = [S[inn]]
    if inn.any() and (~inn).any():
        tr = cKDTree(S)
        pr = tr.query_pairs(1.5 * float(spacing), output_type='ndarray')
        if len(pr):
            a, b = pr[:, 0], pr[:, 1]
            x = inn[a] != inn[b]
            a, b = a[x], b[x]
            t = np.clip((0.5 - m[a]) / np.where(m[b] != m[a], m[b] - m[a], 1.0), 0.0, 1.0)
            P.append(S[a] + t[:, None] * (S[b] - S[a]))
    return np.vstack(P)


def conform_lattice(scene, oid, wall, step=0.001, openings=True, cap=None):
    """the conform cutter's field on the lattice: wall - true depth (< 0 deeper than the wall); with openings, where the
    surface is masked (>= 0.5) the cut reaches the surface: min(that, distance to the masked patch - 1.5 wall)"""
    from scipy.spatial import cKDTree
    cap = cap or max(wall * 2.0, wall + 4 * step)
    G = depth_grid(scene, oid, step, cap)
    C = (wall - G['depth']).astype(np.float32)
    n_open = 0
    if openings and G['mask'] is not None and np.any(G['mask'] >= 0.5):
        n_open = int(np.sum(G['mask'] >= 0.5))
        Mp = masked_patch(G['surface'], G['mask'], step)
        tr = cKDTree(Mp)
        lo, h, dims = G['lo'], G['h'], G['dims']
        ax = [lo[a] + np.arange(dims[a]) * h for a in range(3)]
        Y, Z = np.meshgrid(ax[1], ax[2], indexing='ij')
        for i, x in enumerate(ax[0]):
            P = np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1)
            d, _ = tr.query(P, distance_upper_bound=3 * cap, workers=-1)
            d = np.minimum(d, 3 * cap).reshape(Y.shape)
            inside = G['depth'][i] > 0
            C[i] = np.where(inside, np.minimum(C[i], d - 1.5 * wall), C[i])
    G['cutter'] = C
    G['openings'] = n_open
    return G


def surface_samples(scene, oid, h=None, openings=False, max_nodes=24_000_000):
    """the surface, fast: the field on the object's own lattice in big batches, its sign changes, Newton onto F = 0
    (world) -> (points (n, 3), normals (n, 3), mask values or None, (lo, h, dims, f, W) of the lattice)"""
    from .protected_merge import field_of, _to_world, _rs
    o = scene.get(oid)
    pk = scene.pack()
    _R, _t, s = _rs(o)
    h = float(h or o.sculpt.h)
    lo, hi = (np.asarray(v, float) for v in o.local_box())
    lo, hi = lo - 3 * h, hi + 3 * h
    n = float(np.prod((hi - lo) / h + 1))
    if n > max_nodes:
        h *= (n / max_nodes) ** (1.0 / 3.0) * 1.01
    ss = h * s
    ax = [np.arange(lo[a], hi[a] + 0.5 * h, h) for a in range(3)]
    dims = tuple(len(a) for a in ax)
    Q = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    W = _to_world(o, Q)
    f = np.empty(len(W))
    for a in range(0, len(W), 2_000_000):
        f[a:a + 2_000_000] = field_of(scene, pk, o, W[a:a + 2_000_000]) / s
    fg = f.reshape(dims)
    Wg = W.reshape(dims + (3,))
    pts = []
    for a in range(3):
        s0, s1 = [slice(None)] * 3, [slice(None)] * 3
        s0[a], s1[a] = slice(0, -1), slice(1, None)
        f0, f1 = fg[tuple(s0)], fg[tuple(s1)]
        m = np.isfinite(f0) & np.isfinite(f1) & ((f0 < 0) != (f1 < 0))
        if m.any():
            t = (f0[m] / (f0[m] - f1[m]))[:, None]
            A, Bq = Wg[tuple(s0)][m], Wg[tuple(s1)][m]
            pts.append(A + t * (Bq - A))
    S = np.vstack(pts) if pts else np.zeros((0, 3))
    g = np.zeros_like(S)
    for _ in range(4):
        fs = field_of(scene, pk, o, S)
        for a in range(3):
            e = np.zeros(3)
            e[a] = 2e-5
            g[:, a] = (field_of(scene, pk, o, S + e) - field_of(scene, pk, o, S - e)) / 4e-5
        gg = np.maximum((g * g).sum(1), 1e-18)
        stp = (fs / gg)[:, None] * g
        ok = np.isfinite(stp).all(1) & (np.linalg.norm(stp, axis=1) < 2 * ss)
        S[ok] -= stp[ok]
    N = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    mask = _mask_at(scene, pk, o, S) if openings else None
    return S, N, mask, (lo, h, dims, f, W, s, ss)


def depth_at(S, N, tree, P, ss):
    """the true depth (unsigned) at world points P from the surface samples: the nearest sample's tangent plane where
    it is close to the foot of the perpendicular, else the point distance"""
    dq, j = tree.query(P, workers=-1)
    dpl = np.abs(((P - S[j]) * N[j]).sum(1))
    lateral = np.sqrt(np.maximum(dq * dq - dpl * dpl, 0.0))
    return np.where(lateral < 1.5 * ss, dpl, dq)


def conform_local(scene, oid, wall, step=None, openings=True, max_nodes=24_000_000, open_margin=0.0):
    """the conform cutter as a lattice in the OBJECT's own frame (its local units), for a 'lattice' cutter object that
    shares the object's transform: C = wall - true depth (< 0 deeper than the wall); with openings, the masked surface
    reached. The depth: the nearest surface sample's tangent plane (its error on a curved surface ~ (R - d) theta^2 / 2,
    under 0.01 mm at 1 mm samples), from ONE batched k-d query over the only nodes that can be near the wall (|F| is a
    lower bound of the depth: outside nodes, and inside ones deeper than wall + 3 h by |F|, need none).
    Openings (fs-shell 11-11: the hole is what the user paints): within the wall, a point whose foot on the
    surface lies inside the mask's 0.5 line is cut: the cutter there is the signed distance from that line measured along
    the surface (sqrt(d_line^2 - depth^2); negative inside the mask), so the hole's rim sits ON the line and its sides
    run straight through the wall; open_margin (m, default 0) widens it past the line by that much.
    -> {'lo' (local), 'h' (local), 'cutter' (nx, ny, nz) float32, 'n_surface', 'openings', 'n_query', 'ms'}"""
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    S, N, mask, (lo, h, dims, f, W, s, ss) = surface_samples(scene, oid, step, openings, max_nodes)
    wall_l = float(wall) / s                              # local units
    tree = cKDTree(S)
    margin = 3.0 * h
    C = np.where(f >= 0.0, wall_l + f, -margin)           # outside: positive; deep (by |F|): clamped below zero
    q = np.nonzero((f < 0.0) & (-f < wall_l + margin))[0]
    if len(q):
        depth = depth_at(S, N, tree, W[q], ss) / s
        C[q] = np.maximum(wall_l - depth, -margin)
    n_open = 0
    if openings and mask is not None and np.any(mask >= 0.5):
        n_open = int(np.sum(mask >= 0.5))
        # on BOTH sides of the surface (continuous: a jump at the surface made a ragged rim), within the wall
        near = np.nonzero(f < wall_l + margin)[0]
        d = np.abs(f[near])                               # to the surface: outside by F, inside by the true depth
        if len(q):
            pos = np.full(len(f), -1, np.int64)
            pos[q] = np.arange(len(q))
            k = pos[near]
            d[k >= 0] = depth[k[k >= 0]]
        _, j = tree.query(W[near], workers=-1)
        Wn = W[near]
        foot = Wn - ((Wn - S[j]) * N[j]).sum(1)[:, None] * N[j]
        pk_ = scene.pack()
        mf = _mask_at(scene, pk_, scene.get(oid), foot)
        inside = (mf >= 0.5) if mf is not None else np.zeros(len(near), bool)
        Lp = mask_line(S, mask, ss)
        if len(Lp):
            dl, _ = cKDTree(Lp).query(Wn, workers=-1)
            dl = dl / s
            lat = np.sqrt(np.maximum(dl * dl - d * d, 0.0))   # along the surface from the line
        else:
            lat = np.full(len(near), 3.0 * (wall_l + margin))
        C[near] = np.minimum(C[near], np.where(inside, -lat, lat) - float(open_margin) / s)
    return {'lo': lo, 'h': h, 'cutter': C.reshape(dims).astype(np.float32), 'n_surface': int(len(S)),
            'openings': n_open, 'n_query': int(len(q)), 'ms': (time.perf_counter() - t0) * 1000}
