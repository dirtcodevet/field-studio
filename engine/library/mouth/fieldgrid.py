"""engine/library/mouth/fieldgrid.py: sample a field function on a lattice, re-distance it to a true distance (L = 1, as
Field Studio's 'lattice' node and its tracer need), and draw it for form checks (CPU, numba). fs-mouth lane, 11-11.

No meshes (P14): the zero set is found as sign changes along lattice edges (points), never as triangles.

    G, lo = sample(fn, lo, hi, h)               # fn(P (N, 3) mm) -> (N,) ; G (nx, ny, nz) float32, node i at lo + i h
    D = redistance(G, h)                         # true signed distance (mm) to the zero set of G
    img = render_ortho([(D, lo, h, albedo), ...], view_dir, up, ...)   # clay preview (several grids: nearest hit)
"""
import numpy as np
from scipy.spatial import cKDTree
import numba as nb


def sample(fn, lo, hi, h, chunk=1_500_000):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    axes = [lo[k] + np.arange(n[k]) * h for k in range(3)]
    G = np.empty(n, np.float32)
    nyz = n[1] * n[2]
    step = max(1, chunk // nyz)
    Y, Z = np.meshgrid(axes[1], axes[2], indexing="ij")
    Y, Z = Y.ravel(), Z.ravel()
    for i0 in range(0, n[0], step):
        xs = axes[0][i0:i0 + step]
        P = np.empty((len(xs) * nyz, 3))
        P[:, 0] = np.repeat(xs, nyz)
        P[:, 1] = np.tile(Y, len(xs))
        P[:, 2] = np.tile(Z, len(xs))
        G[i0:i0 + len(xs)] = fn(P).reshape(len(xs), n[1], n[2])
    return G, lo


def sample_band(fn, lo, hi, h, band, coarse=4, margin=2.0, chunk=1_500_000):
    """sample() on the lattice, but the fine nodes evaluated only near the surface (fs-mouth 11-12: the mouth set's
    dense evaluation took 7.6 min, most of it on nodes that redistance() then clamps to +-band). A coarse lattice
    (coarse x h) is evaluated first; a coarse cell's fine nodes are evaluated where its corner values come within
    band + margin x the cell's diagonal of zero (the fields are distance-like: |grad f| <= ~1; margin 2 covers the
    blends' slack), the rest take the coarse value trilinearly, clamped beyond +-band (where redistance() puts +-band
    anyway). -> (G, lo), the same lattice as sample()"""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    nc = (n - 1 + coarse - 1) // coarse + 1                      # coarse nodes covering the fine lattice
    Gc, _ = sample(fn, lo, lo + (nc - 1) * coarse * h, coarse * h, chunk)
    hc = coarse * h
    reach = band + margin * hc * np.sqrt(3.0)
    m = np.abs(Gc) < reach
    near = np.zeros(tuple(nc - 1), bool)                         # a cell is near when any of its 8 corners is
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                near |= m[a:a + nc[0] - 1, b:b + nc[1] - 1, c:c + nc[2] - 1]
    # the fine lattice from the coarse values (trilinear), then the near cells' nodes evaluated exactly
    fx = [np.arange(n[k]) / coarse for k in range(3)]
    i0 = [np.minimum(np.floor(f).astype(int), nc[k] - 2) for k, f in enumerate(fx)]
    t = [fx[k] - i0[k] for k in range(3)]
    G = np.empty(tuple(n), np.float32)
    for i in range(n[0]):
        a0, ta = i0[0][i], t[0][i]
        S = Gc[a0] * (1 - ta) + Gc[a0 + 1] * ta                  # (ncy, ncz)
        S = S[i0[1]] * (1 - t[1])[:, None] + S[i0[1] + 1] * t[1][:, None]
        S = S[:, i0[2]] * (1 - t[2])[None, :] + S[:, i0[2] + 1] * t[2][None, :]
        G[i] = S
    ci = [np.minimum(np.arange(n[k]) // coarse, nc[k] - 2) for k in range(3)]
    sel = near[ci[0][:, None, None], ci[1][None, :, None], ci[2][None, None, :]]
    idx = np.argwhere(sel)
    for s0 in range(0, len(idx), chunk):
        q = idx[s0:s0 + chunk]
        G[q[:, 0], q[:, 1], q[:, 2]] = fn(lo + q * h)
    far = ~sel
    G[far] = np.clip(G[far], -band, band)
    return G, lo


def crossings(G, lo, h):
    """the zero set as points on lattice edges (linear interpolation along each edge with a sign change)"""
    pts = []
    for ax in range(3):
        a = np.moveaxis(G, ax, 0)
        f0, f1 = a[:-1], a[1:]
        m = (f0 < 0) != (f1 < 0)
        idx = np.argwhere(m)
        t = f0[m] / (f0[m] - f1[m])
        p = idx.astype(np.float64)
        p[:, 0] += t
        # moveaxis(ax -> 0) then indices are (ax, rest...): put them back in x, y, z order
        if ax == 1:
            p = p[:, [1, 0, 2]]
        elif ax == 2:
            p = p[:, [1, 2, 0]]
        pts.append(p)
    P = np.concatenate(pts, 0) * h + lo
    return P


def redistance(G, h, lo=None, band=None, chunk=2_000_000):
    """true signed distance (mm) from the sign of G: the distance to the zero-set points (sign changes along edges), and
    at the nodes next to the zero set the first-order value G / |grad G| (more exact there). band: beyond this distance
    (mm) the value is clamped to +-band (None: exact everywhere)."""
    lo0 = np.zeros(3) if lo is None else np.asarray(lo, float)
    G = np.asarray(G, np.float32)
    P = crossings(G, lo0, h)
    tree = cKDTree(P)
    n = G.shape
    D = np.empty(n, np.float32)
    axes = [lo0[k] + np.arange(n[k]) * h for k in range(3)]
    nyz = n[1] * n[2]
    step = max(1, chunk // nyz)
    Y, Z = np.meshgrid(axes[1], axes[2], indexing="ij")
    Y, Z = Y.ravel(), Z.ravel()
    for i0 in range(0, n[0], step):
        xs = axes[0][i0:i0 + step]
        Q = np.empty((len(xs) * nyz, 3))
        Q[:, 0] = np.repeat(xs, nyz)
        Q[:, 1] = np.tile(Y, len(xs))
        Q[:, 2] = np.tile(Z, len(xs))
        d, _ = tree.query(Q, workers=-1, distance_upper_bound=(np.inf if band is None else band * 1.01))
        d = np.minimum(d, 1e3 if band is None else band)
        D[i0:i0 + len(xs)] = d.reshape(len(xs), n[1], n[2])
    # nodes next to the zero set: G / |grad G|
    gx, gy, gz = np.gradient(G.astype(np.float64), h)
    gn = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
    near = np.zeros(n, bool)
    for ax in range(3):
        s = (G < 0)
        d_ = np.diff(s.astype(np.int8), axis=ax) != 0
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax] = slice(0, -1)
        sl1[ax] = slice(1, None)
        near[tuple(sl0)] |= d_
        near[tuple(sl1)] |= d_
    dn = np.abs(G / gn)
    D = np.where(near, np.minimum(D, dn), D).astype(np.float32)
    return np.where(G < 0, -D, D).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- CPU preview tracer
@nb.njit(cache=True, fastmath=True)
def _tri(D, lo, h, x, y, z):
    nx, ny, nz = D.shape
    fx = (x - lo[0]) / h
    fy = (y - lo[1]) / h
    fz = (z - lo[2]) / h
    # outside the box: distance to the box + a large value (rays skip empty space)
    if fx < 0 or fy < 0 or fz < 0 or fx > nx - 1.001 or fy > ny - 1.001 or fz > nz - 1.001:
        cx = min(max(fx, 0.0), nx - 1.001)
        cy = min(max(fy, 0.0), ny - 1.001)
        cz = min(max(fz, 0.0), nz - 1.001)
        dd = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2 + (fz - cz) ** 2) * h
        return dd + max(_tri_in(D, cx, cy, cz), 0.0)
    return _tri_in(D, fx, fy, fz)


@nb.njit(cache=True, fastmath=True)
def _tri_in(D, fx, fy, fz):
    i = int(fx)
    j = int(fy)
    k = int(fz)
    tx = fx - i
    ty = fy - j
    tz = fz - k
    c00 = D[i, j, k] * (1 - tx) + D[i + 1, j, k] * tx
    c10 = D[i, j + 1, k] * (1 - tx) + D[i + 1, j + 1, k] * tx
    c01 = D[i, j, k + 1] * (1 - tx) + D[i + 1, j, k + 1] * tx
    c11 = D[i, j + 1, k + 1] * (1 - tx) + D[i + 1, j + 1, k + 1] * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


@nb.njit(parallel=True, cache=True, fastmath=True)
def _trace(Ds, los, hs, nobj, org, dx, dy, dz, W, Hh, tmax, out_t, out_id):
    for py in nb.prange(Hh):
        for px in range(W):
            ox = org[py, px, 0]
            oy = org[py, px, 1]
            oz = org[py, px, 2]
            t = 0.0
            hit = -1
            for it in range(1200):
                x = ox + dx * t
                y = oy + dy * t
                z = oz + dz * t
                best = 1e9
                bi = -1
                for o in range(nobj):
                    d = _tri(Ds[o], los[o], hs[o], x, y, z)
                    if d < best:
                        best = d
                        bi = o
                if best < 0.002:
                    hit = bi
                    break
                t += max(best * 0.9, 0.002)
                if t > tmax:
                    break
            out_t[py, px] = t
            out_id[py, px] = hit


def render_ortho(objs, view, up, centre, width_mm, W=600, H=None, light=(0.4, 0.6, 0.7), bg=(0.12, 0.12, 0.13)):
    """objs: list of (D, lo, h, albedo rgb). view: direction the camera looks along (into the scene). Orthographic.
    Returns an (H, W, 3) float image in [0, 1] (clay shading: key + fill + rim; a form check, not the look)."""
    H = H or W
    view = np.asarray(view, float)
    view /= np.linalg.norm(view)
    up = np.asarray(up, float)
    r = np.cross(view, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, view)
    px = width_mm / W
    xs = (np.arange(W) - W / 2 + 0.5) * px
    ys = (H / 2 - np.arange(H) - 0.5) * px
    X, Y = np.meshgrid(xs, ys)
    far = 200.0
    org = np.asarray(centre, float) + X[..., None] * r + Y[..., None] * u - view * far / 2
    # equalise shapes for numba: pad the grids into one list (numba typed list of arrays)
    from numba.typed import List
    Ds = List()
    for o in objs:
        Ds.append(np.ascontiguousarray(o[0], np.float32))
    los = np.array([o[1] for o in objs], np.float64)
    hs = np.array([o[2] for o in objs], np.float64)
    out_t = np.zeros((H, W))
    out_id = np.zeros((H, W), np.int64)
    _trace(Ds, los, hs, len(objs), np.ascontiguousarray(org), view[0], view[1], view[2], W, H, far, out_t, out_id)
    P = org + out_t[..., None] * view
    # normals by central differences of the hit object's grid
    img = np.zeros((H, W, 3))
    img[:] = bg
    # camera-relative clay lights: key from upper left front, fill from the right, rim behind
    L = -view * 0.75 + u * 0.55 - r * 0.45
    L /= np.linalg.norm(L)
    for o, (D, lo, h, alb) in enumerate(objs):
        m = out_id == o
        if not m.any():
            continue
        Q = P[m]
        e = 0.5 * h
        n = np.stack([_eval_grid(D, lo, h, Q + e * np.eye(3)[k]) - _eval_grid(D, lo, h, Q - e * np.eye(3)[k]) for k in range(3)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        ndl = np.clip(n @ L, 0, 1)
        F2 = -view * 0.6 + r * 0.7 - u * 0.1
        F2 /= np.linalg.norm(F2)
        fill = np.clip(n @ F2, 0, 1) * 0.28
        rim = (1 - np.clip(-(n @ view), 0, 1)) ** 3 * 0.25
        hv = L - view
        hv /= np.linalg.norm(hv)
        spec = np.clip(n @ hv, 0, 1) ** 40 * 0.35
        sh = 0.10 + 0.75 * ndl + fill + rim
        img[m] = np.clip(np.asarray(alb)[None] * sh[:, None] + spec[:, None], 0, 1)
    return img


def _eval_grid(D, lo, h, Q):
    from scipy.ndimage import map_coordinates
    f = (Q - np.asarray(lo)) / h
    return map_coordinates(D, f.T, order=1, mode="nearest")
