"""Protected parts from the Mask brush (milestone, 11-08): the surface a person paints with the Mask brush on an
object becomes a separate protected part, a lattice object that keeps its own surface while the body around it is
sculpted (a lid against an eye, a fold, any part that must stay separate in contact).

  out_dir_for     where a store version's parts are saved (<version>_protected_parts/, or FIELDSTUDIO_PARTS_DIR)
  build_from_marks  each marked part -> a 'lattice' field file: P = max(F, C), C a smooth bisector cut between the
                  part's marks, the other parts' marks and the unmarked surface around them
  masked_surface  the object's surface nodes that carry the Mask brush's paint
  object_field_fn the object's field as an editor reads it
  protect_masked  the masked surface becomes the named part (and every marked part is rebuilt)
"""
import json
import os
import time

import numpy as np

from . import fv8base, ops


def out_dir_for(field_dir):
    # test harnesses set FIELDSTUDIO_PARTS_DIR so their marks never land in a store's own folder
    alt = os.environ.get('FIELDSTUDIO_PARTS_DIR')
    if alt:
        return os.path.join(alt, os.path.basename(os.path.abspath(field_dir)) + '_protected_parts')
    return os.path.abspath(field_dir) + '_protected_parts'


def build_from_marks(F_fn, org, h, marks, others, out_dir, depth=0.006, outside=0.001, margin=0.003):
    """F_fn(X (n,3) local) -> her field; marks {name: (n,3) local surface points the user marked}; others (m,3): her surface
    points around them that belong to no part. Each marked part -> a 'lattice' file in out_dir: P = max(F, C),
    C = max((d_own - d_other) / 2, d_own - depth) (smooth bisector cuts against the other parts' marks and her unmarked
    surface). -> {name: {'file', 'sha', 'lo', 'dims', 'region_nodes'}}"""
    from scipy.spatial import cKDTree
    os.makedirs(out_dir, exist_ok=True)
    allp = [np.asarray(v, float).reshape(-1, 3) for v in marks.values()] + [np.asarray(others, float).reshape(-1, 3)]
    labs = np.concatenate([np.full(len(p), i + 1, np.int32) for i, p in enumerate(allp[:-1])] + [np.zeros(len(allp[-1]), np.int32)])
    Z = np.concatenate(allp)
    names = list(marks)
    out = {}
    for i, name in enumerate(names):
        own = labs == i + 1
        if not own.any():
            continue
        P0 = Z[own]
        lo = P0.min(0) - depth - margin
        hi = P0.max(0) + depth + margin
        i0 = np.floor((lo - org) / h).astype(np.int64)
        i1 = np.ceil((hi - org) / h).astype(np.int64)
        dims = tuple(int(v) for v in (i1 - i0 + 1))
        ax = [org[k] + (i0[k] + np.arange(dims[k])) * h for k in range(3)]
        X = np.ascontiguousarray(np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3))
        F = F_fn(X)
        cand = np.nonzero(F <= outside)[0]
        t_own = cKDTree(P0)
        t_oth = cKDTree(Z[~own]) if (~own).any() else None
        C = np.full(len(X), -1.0)     # r4: outside her (F > outside) the part's field is hers (C = 1 there made 1 m
        #                                values beside her surface: the first PK1 run's 130 mm error)
        for a0 in range(0, len(cand), 1 << 20):
            q = cand[a0:a0 + (1 << 20)]
            d_o, _ = t_own.query(X[q], k=1)
            d_x = t_oth.query(X[q], k=1)[0] if t_oth is not None else np.full(len(q), np.inf)
            C[q] = np.maximum(0.5 * (d_o - d_x), d_o - depth)
        Pk = np.maximum(F, C).reshape(dims).astype(np.float32)
        tmp = os.path.join(out_dir, '_build_tmp.npz')
        np.savez(tmp, data=Pk, lo=org + i0 * h, h=np.float64(h))
        sha = fv8base.file_sha(tmp)
        f = os.path.join(out_dir, '%s_%s.npz' % (name.replace(' ', '_'), sha[:12]))   # never overwritten: logs name it
        os.replace(tmp, f)
        out[name] = {'file': f, 'sha': sha, 'lo': (org + i0 * h).tolist(), 'dims': list(dims),
                     'region_nodes': int((C < 0).sum())}
    return out


def masked_surface(scene, o, thr=0.5):
    """the object's surface nodes (local) that carry the Mask brush's paint (mask > thr, |F| <= 1.5 h)"""
    from . import program as P
    g = o.sculpt
    st = scene.store
    h = float(g.h)
    pts = []
    for b, s in g.slots.items():
        if s < 0 or not st.mask[s].any():
            continue
        idx = np.argwhere(st.mask[s] > thr)
        if len(idx):
            pts.append((np.asarray(b) * 8 + idx) * h)
    if not pts:
        return np.zeros((0, 3))
    X = np.ascontiguousarray(np.concatenate(pts))
    pk = scene.pack()
    f = np.empty(len(X))
    oi = scene.index_of(o.id)
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, X, f)
    X, f = X[np.abs(f) <= 1.5 * h], f[np.abs(f) <= 1.5 * h]
    if not st.has('mdir') or not st.mdir.any() or not len(X):
        return X
    # direction-tagged paint (11-09): each marked node's surface point, kept only where the direction-aware mask
    # there (its own surface normal) is still above thr: nodes shared with a facing sheet mark the painted sheet only
    from . import brush as BR
    import math as _m
    def grad(Y, e):
        G = np.zeros((len(Y), 3))
        for ax in range(3):
            a = np.empty(len(Y)); b = np.empty(len(Y)); d = np.zeros(3); d[ax] = e
            P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(Y + d), a)
            P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(Y - d), b)
            G[:, ax] = a - b
        return G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    Y = X - f[:, None] * grad(X, 0.5 * h)
    for _ in range(3):
        fy = np.empty(len(Y))
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(Y), fy)
        Y = Y - fy[:, None] * grad(Y, 0.05 * h)
    N = grad(Y, 0.02 * h)
    cd = _m.cos(_m.radians(BR.MASK_DIR_CUTOFF_DEG))
    keep = np.array([P.mask_sample_dir(pk.index_flat, pk.mask, pk.mdir, pk.oi_tab, oi, h, *map(float, y), *map(float, nn), cd) > thr
                     for y, nn in zip(Y, N)], bool)
    return Y[keep]


def object_field_fn(scene, o):
    from . import program as P
    pk = scene.pack()
    oi = scene.index_of(o.id)

    def F(X):
        X = np.ascontiguousarray(X, np.float64)
        out = np.empty(len(X))
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, X, out)
        return out
    return F


def protect_masked(scene, o, name, out_dir, org=None, depth=0.006, around=0.012):
    """the user's marks for `name` = the object's masked surface nodes; the other parts' saved marks and her unmarked surface
    around them are the 'others'. Saves marks.npz and builds every marked part again (so the cut walls between parts
    stay bisectors between the marks). -> (part info for name, all parts info, n marked points)"""
    if not str(name).strip():
        raise ValueError('a protected part needs a name')
    M = masked_surface(scene, o)
    if len(M) < 20:
        raise ValueError('paint the part with the Mask brush first (found %d marked surface points)' % len(M))
    h = float(o.sculpt.h)
    org = np.zeros(3) if org is None else np.asarray(org, float)
    os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, 'marks.npz')
    marks = {}
    if os.path.exists(mp):
        with np.load(mp) as z:
            marks = {k.replace('_', ' '): np.asarray(z[k]) for k in z.files}
    marks[name] = M
    np.savez(mp, **{k.replace(' ', '_'): v for k, v in marks.items()})
    F = object_field_fn(scene, o)
    allm = np.concatenate(list(marks.values()))
    lo = allm.min(0) - around
    hi = allm.max(0) + around
    i0 = np.floor((lo - org) / h).astype(np.int64)
    i1 = np.ceil((hi - org) / h).astype(np.int64)
    # her unmarked surface around the marks (every 2nd node: the bisector needs the surface, not every sample)
    ax = [org[k] + (i0[k] + np.arange(0, int(i1[k] - i0[k]) + 1, 2)) * h for k in range(3)]
    X = np.ascontiguousarray(np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3))
    f = F(X)
    S = X[np.abs(f) <= 1.0 * h]
    from scipy.spatial import cKDTree
    dm = cKDTree(allm).query(S, k=1)[0]
    others = S[dm > 1.5 * h]
    info = build_from_marks(F, org, h, marks, others, out_dir, depth=depth)
    json.dump({'h': h, 'parts': info, 'marks': {k: int(len(v)) for k, v in marks.items()}},
              open(os.path.join(out_dir, 'parts.json'), 'w'), indent=1)
    return info[name], info, int(len(M))
