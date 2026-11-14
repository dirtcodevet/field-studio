"""Write an editable store object's edits into a NEW version of her store (milestone item 2, 11-08 night).

The object's field is F_obj = its base (her store, plus any merged shapes) + its sculpt layer, in her store's frame.
The edit is dF = F_obj - F_store. It is non-zero only near the sculpted bricks and the merged parts (their footprint
+ the blend margin), so the support is split into connected clusters (face, hands, ...). Each cluster becomes one
'delta_grid' op of the store (fv8_field.py: applied at every level's own samples, bricks allocated where the moved
surface leaves a level's band), on a lattice at the object's h aligned to her store's nodes. Clusters are disjoint,
so no point takes two deltas. commit() writes the version (unchanged arrays hard-linked); the published version and
FIELD_CURRENT.json are never touched.
"""
import os
import time

import numpy as np
from numba import njit, prange

from . import fv8base, ops, program as P
from . import region


@njit(cache=True, parallel=True)
def _eval_many(code, prm, X, out):
    for i in prange(X.shape[0]):
        vs = np.zeros(P.STACK)
        ps = np.zeros(4 * P.STACK)
        out[i] = P.eval_prog(code, prm, 0, code.shape[0], X[i, 0], X[i, 1], X[i, 2], vs, ps)


def _store_node(o):
    for n in o.base.walk():
        if n.kind == 'fv8store':
            return n
    return None


def support_boxes(scene, o, margin):
    """-> list of (lo, hi) local boxes holding every point where F_obj may differ from her store"""
    boxes = []
    g = o.sculpt
    st = scene.store
    B = 8 * g.h
    for b, s in g.slots.items():
        if s >= 0 and (st.delta[s].any()):
            lo = np.asarray(b, float) * B
            boxes.append((lo - margin, lo + B + margin))

    def walk(n):
        if not any(m.kind == 'fv8store' for m in n.walk()):
            code, prm, lo, hi, L = ops.compile_tree(n)
            boxes.append((np.asarray(lo) - margin, np.asarray(hi) + margin))
            return
        for c in n.children:
            walk(c)
    walk(o.base)
    return boxes


def clusters(boxes, cell):
    """union of boxes -> connected clusters (bounding boxes) on a grid of `cell`"""
    if not boxes:
        return []
    from scipy import ndimage
    lo = np.min([b[0] for b in boxes], 0)
    hi = np.max([b[1] for b in boxes], 0)
    n = np.ceil((hi - lo) / cell).astype(int) + 1
    occ = np.zeros(n, bool)
    for a, b in boxes:
        i0 = np.floor((a - lo) / cell).astype(int)
        i1 = np.ceil((b - lo) / cell).astype(int)
        occ[i0[0]:i1[0] + 1, i0[1]:i1[1] + 1, i0[2]:i1[2] + 1] = True
    lab, k = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    out = []
    for sl in ndimage.find_objects(lab):
        a = lo + np.array([s.start for s in sl]) * cell
        b = lo + np.array([s.stop for s in sl]) * cell
        out.append((a, b))
    return out


T_NODES = 128               # nodes per tile axis (32 mm at 0.25 mm)


def prepare(scene, oid, blend_margin=0.004):
    """(under the scene lock; fast) -> what write() needs: a snapshot of the scene's program and sculpt layer (the
    pack), her store's own program, and the 32 mm tiles (T_NODES^3, on her store's lattice) that can hold a change:
    tiles of the non-zero sculpt bricks (+ the blend margin), and for every base part that is not her store, the tiles
    where that part's field and her field are within reach of the tile (a Cut's cutter is a 3.5 m box: its box tiled
    her whole body; the cone through her is what changes). r-stream (11-08 night): the deltas are computed and applied
    one tile at a time in write() (keeping them all reached 22 GB in the app: the RAM guard stopped it)."""
    o = scene.get(oid)
    node = _store_node(o)
    if node is None:
        raise ValueError('%s is not an editable store object' % o.name)
    hs = float(o.sculpt.h)                   # the sculpt layer's lattice (its bricks)
    F8m = region.fv8(node.params['source'])
    # fs-trust 11-11: the safe writer (the default) takes its target on L0's lattice whatever the object's h
    h = float(F8m.LEVEL_H[0]) if _writer(F8m) == 'txn' else hs
    pk = scene.pack()
    oi = scene.index_of(oid)
    store_code, store_prm, _lo, _hi, _L = ops.compile_tree(node)
    S = fv8base.load(node.params['source'], node.params.get('source_id') or None)
    org = np.asarray(S['hdr'][0:3])
    tl = T_NODES * h
    tiles = set()
    m = blend_margin + 2 * max(h, hs)
    g = o.sculpt
    st = scene.store
    B = 8 * hs
    for bk, sl in g.slots.items():
        if sl >= 0 and st.delta[sl].any():
            lo = np.asarray(bk, float) * B - m
            hi = lo + B + 2 * m
            a0 = np.floor((lo - org) / tl).astype(int)
            a1 = np.floor((hi - org) / tl).astype(int)
            for i in range(a0[0], a1[0] + 1):
                for j in range(a0[1], a1[1] + 1):
                    for k in range(a0[2], a1[2] + 1):
                        tiles.add((i, j, k))

    reach = 0.5 * np.sqrt(3) * tl + m

    def walk(n):
        if not any(q.kind == 'fv8store' for q in n.walk()):
            code, prm, lo, hi, L = ops.compile_tree(n)
            a0 = np.floor((np.asarray(lo) - org) / tl).astype(int) - 1
            a1 = np.floor((np.asarray(hi) - org) / tl).astype(int) + 1
            sl0 = np.floor((np.asarray(S['lo']) - org) / tl).astype(int)
            sl1 = np.floor((np.asarray(S['hi']) - org) / tl).astype(int)
            a0, a1 = np.maximum(a0, sl0), np.minimum(a1, sl1)
            if np.any(a1 < a0):
                return
            T = np.stack(np.meshgrid(*[np.arange(a0[i], a1[i] + 1) for i in range(3)], indexing='ij'), -1).reshape(-1, 3)
            C = np.ascontiguousarray(org + (T + 0.5) * tl)
            fs = np.empty(len(C)); _eval_many(code, prm, C, fs)
            fh = np.empty(len(C)); _eval_many(store_code, store_prm, C, fh)
            keep = (fs <= reach) & (fh <= reach + 0.02)
            for t in map(tuple, T[keep].tolist()):
                tiles.add(t)
            return
        for cc in n.children:
            walk(cc)
    walk(o.base)
    return {'source': node.params['source'], 'source_id': node.params.get('source_id'), 'name': o.name,
            'pk': pk, 'oi': oi, 'store_code': store_code, 'store_prm': store_prm, 'org': org, 'h': h,
            'tiles': sorted(tiles)}


def _writer(F8):
    """fs-trust 11-11: 'txn' (the default: fv8_write's write_txn) or 'legacy' (FV8_WRITER=legacy: the old per-tile
    delta_grid ops), as the store module decides (fv8_field.writer_mode); a store module from before the switch: the env"""
    if hasattr(F8, 'writer_mode'):
        return F8.writer_mode()
    return 'txn' if os.environ.get('FV8_WRITER') == 'txn' else 'legacy'


def tile_delta(prep, t, want_obj=False):
    """one tile's delta lattice (origin, D) or None when it holds no change; want_obj: (origin, D, F_obj) (FV8_WRITER=txn)"""
    h = prep['h']
    T = T_NODES
    t0 = np.asarray(t) * T
    origin = prep['org'] + t0 * h
    ax = [origin[k] + np.arange(T) * h for k in range(3)]       # nodes t0 .. t0 + T - 1: tiles are disjoint
    X = np.ascontiguousarray(np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3))
    pk = prep['pk']
    fo = np.empty(len(X))
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, prep['oi'], X, fo)
    fs = np.empty(len(X))
    _eval_many(prep['store_code'], prep['store_prm'], X, fs)
    D = (fo - fs).reshape(T, T, T).astype(np.float32)
    D[np.abs(D) < 1e-9] = 0.0
    if not np.any(D): return None
    return (origin, D, fo.reshape(T, T, T).astype(np.float32)) if want_obj else (origin, D)


def tile_groups(tiles, max_tiles=27):
    """26-connected groups of tiles, each split along x into runs of at most max_tiles tiles (memory bound)"""
    left = set(tuple(int(v) for v in t) for t in tiles); groups = []
    while left:
        seed = left.pop(); g = [seed]; todo = [seed]
        while todo:
            c = todo.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        n = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if n in left: left.remove(n); g.append(n); todo.append(n)
        g.sort()
        for i in range(0, len(g), max_tiles): groups.append(g[i:i + max_tiles])
    return groups


def _dense_cloud(F, origin, h, lim, sub=4):
    """F_obj's trilinear zero set on a dense lattice (F, node (i,j,k) at origin + ijk h) as points with unit normals:
    crossings of a sub-lattice inside each crossing cell, as fv8_write.ZeroCloud does for the store's bricks (no
    triangles). -> (points, normals, crossing-cell mask (nx-1, ny-1, nz-1))"""
    c = [F[a:F.shape[0] - 1 + a, b_:F.shape[1] - 1 + b_, d:F.shape[2] - 1 + d] for a in (0, 1) for b_ in (0, 1) for d in (0, 1)]
    lo = np.minimum.reduce(c)
    hi = np.maximum.reduce(c)
    amin = np.minimum.reduce([np.abs(x) for x in c])
    cross = (lo < 0) & (hi >= 0) & (amin < lim)
    del lo, hi, amin
    ci, cj, ck = np.nonzero(cross)
    t = np.arange(sub + 1) / sub
    P_, N_ = [], []
    for m0 in range(0, len(ci), 20000):
        sl = slice(m0, m0 + 20000)
        Cc = np.stack([x[ci[sl], cj[sl], ck[sl]] for x in c], 1).astype(np.float64)       # (m, 8), corner order (a, b, d)
        m = len(Cc)
        wx = np.stack([1 - t, t], 0)
        Sg = np.zeros((m, sub + 1, sub + 1, sub + 1))
        for q, (a, b_, d) in enumerate([(a, b_, d) for a in (0, 1) for b_ in (0, 1) for d in (0, 1)]):
            Sg += Cc[:, q, None, None, None] * (wx[a][None, :, None, None] * wx[b_][None, None, :, None] * wx[d][None, None, None, :])
        base = origin + np.stack([ci[sl], cj[sl], ck[sl]], 1).astype(np.float64) * h
        for axk in range(3):
            A_ = np.take(Sg, np.arange(sub), axis=axk + 1)
            B_ = np.take(Sg, np.arange(1, sub + 1), axis=axk + 1)
            mk = (np.sign(A_) * np.sign(B_) < 0) | ((A_ == 0) & (B_ != 0))
            if not mk.any():
                continue
            e, i, j, k = np.nonzero(mk)
            va = A_[e, i, j, k]
            vb = B_[e, i, j, k]
            U = np.stack([i, j, k], 1).astype(np.float64)
            U[:, axk] += va / (va - vb)
            U /= sub
            cc = Cc[e]
            u, v, w_ = U[:, 0], U[:, 1], U[:, 2]

            def cv(a, b_, d):
                return cc[:, (a * 2 + b_) * 2 + d]
            gx = ((1 - v) * (1 - w_) * (cv(1, 0, 0) - cv(0, 0, 0)) + v * (1 - w_) * (cv(1, 1, 0) - cv(0, 1, 0))
                  + (1 - v) * w_ * (cv(1, 0, 1) - cv(0, 0, 1)) + v * w_ * (cv(1, 1, 1) - cv(0, 1, 1)))
            gy = ((1 - u) * (1 - w_) * (cv(0, 1, 0) - cv(0, 0, 0)) + u * (1 - w_) * (cv(1, 1, 0) - cv(1, 0, 0))
                  + (1 - u) * w_ * (cv(0, 1, 1) - cv(0, 0, 1)) + u * w_ * (cv(1, 1, 1) - cv(1, 0, 1)))
            gz = ((1 - u) * (1 - v) * (cv(0, 0, 1) - cv(0, 0, 0)) + u * (1 - v) * (cv(1, 0, 1) - cv(1, 0, 0))
                  + (1 - u) * v * (cv(0, 1, 1) - cv(0, 1, 0)) + u * v * (cv(1, 1, 1) - cv(1, 1, 0)))
            g = np.stack([gx, gy, gz], 1)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
            P_.append(base[e] + U * h)
            N_.append(g)
    P = np.concatenate(P_) if P_ else np.zeros((0, 3))
    N = np.concatenate(N_) if N_ else np.zeros((0, 3))
    return P, N, cross


def _dist_target(prep, origin, Tg, F8, sub=4):
    """fs-trust 11-11 (the store's contract item d: a stored value never exceeds the distance to the surface; M on her,
    Field Studio milestone 1 W3: thousands of samples over-stated at every level, up to 1.7 mm): the object's field F_obj
    is not a distance away from its surface (strokes stretch it: |grad F| up to 1.3 by the release rule, more farther
    out), and the writer takes a target as given and restricts it to the coarse levels. So the write-back gives the store
    the DISTANCE to F_obj's own zero set, signed by F_obj: a target node that is a corner of a crossing cell keeps F_obj
    exactly (the surface does not move); every other one gets sign(F_obj) x a lower bound of the distance to F_obj's zero
    set (fv8_write.ZeroCloud's rule on F_obj sampled on L0's lattice over the group and its cap's margin), capped at L0's
    cap. FS_WB_DIST=0 turns it off. -> (the new target, report)"""
    FW = F8.fv8_write_module()
    h0 = float(F8.LEVEL_H[0])
    S_ = F8.FieldStore(prep['source'])
    cap0 = float(S_.lv[0].get('cap_m', 0.003))
    del S_
    M = int(np.ceil(cap0 / h0)) + 2
    shp = np.array(Tg.shape)
    eshp = shp + 2 * M
    eo = np.asarray(origin, np.float64) - M * h0
    pk = prep['pk']
    F = np.empty(tuple(int(v) for v in eshp), np.float32)
    ay = eo[1] + np.arange(eshp[1]) * h0
    az = eo[2] + np.arange(eshp[2]) * h0
    YY, ZZ = np.meshgrid(ay, az, indexing='ij')
    for i in range(int(eshp[0])):
        X = np.ascontiguousarray(np.stack([np.full(YY.size, eo[0] + i * h0), YY.ravel(), ZZ.ravel()], 1))
        fo = np.empty(len(X))
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, prep['oi'], X, fo)
        F[i] = fo.reshape(YY.shape)
    lim = min(2e-3, 0.999 * cap0)
    Pp, Nn, cross = _dense_cloud(F, eo, h0, lim, sub)
    keep = np.zeros(F.shape, bool)
    for a in (0, 1):
        for b_ in (0, 1):
            for d in (0, 1):
                keep[a:F.shape[0] - 1 + a, b_:F.shape[1] - 1 + b_, d:F.shape[2] - 1 + d] |= cross
    keep = keep[M:M + shp[0], M:M + shp[1], M:M + shp[2]]
    fin = np.isfinite(Tg)
    # crossing-cell corners keep F_obj exactly: the surface stays as sculpted. (Milestone 1 runs 8-9 tried lowering them where F_obj
    # over-states the distance: per node to the cloud distance moved the surface up to 21.5 um, by F_obj's gradient length
    # 18.3 um; F_obj's |grad| near her edited surface reaches 2.76 (p99), the brushes' field. The coarse levels are kept from
    # over-stating by the writer itself (fv8_write._rebuild_level, contract item d).)
    ci = np.argwhere(fin & keep)
    gl = np.ones(len(ci))
    if len(ci):
        e = ci + M
        g2 = np.zeros(len(ci))
        for ax in range(3):
            up = e.copy(); up[:, ax] += 1; dn_ = e.copy(); dn_[:, ax] -= 1
            g2 += ((F[up[:, 0], up[:, 1], up[:, 2]].astype(np.float64) - F[dn_[:, 0], dn_[:, 1], dn_[:, 2]]) / (2 * h0)) ** 2
        gl = np.sqrt(g2)
    del F
    T = Tg.copy()
    todo = fin & ~keep
    rep = {'target_nodes': int(fin.sum()), 'crossing_corners': int((fin & keep).sum()),
           'corner_grad_p50_p99': [round(float(np.percentile(gl, q)), 3) for q in (50, 99)] if len(ci) else None,
           'redistanced': int(todo.sum()),
           'cloud_points': int(len(Pp))}
    if todo.any() and len(Pp):
        from scipy.spatial import cKDTree
        zc = FW.ZeroCloud.__new__(FW.ZeroCloud)
        zc.P, zc.N, zc.spacing, zc.kd, zc.cells = Pp, Nn, h0 / sub, cKDTree(Pp), int(cross.sum())
        idx = np.argwhere(todo)
        sp = h0 / sub
        for c0 in range(0, len(idx), 1 << 20):
            q = idx[c0:c0 + (1 << 20)]
            X = np.asarray(origin, np.float64) + q * h0
            d = zc.dist(X, upper=cap0 + sp)
            d = np.maximum(d - sp * sp / (8.0 * np.maximum(d, sp)) - 1e-6, 0.0)      # the point distance over-states by <= s^2 / 8d
            v = Tg[q[:, 0], q[:, 1], q[:, 2]]
            T[q[:, 0], q[:, 1], q[:, 2]] = (np.where(v < 0, -1.0, 1.0) * np.minimum(d, cap0)).astype(np.float32)
        dv = np.abs(T[todo]).astype(np.float64) - np.abs(Tg[todo]).astype(np.float64)
        rep['abs_change_max_um'] = round(float(np.abs(dv).max()) * 1e6, 1)
        rep['lowered_p99_um'] = round(float(np.percentile(-dv, 99)) * 1e6, 1)
        rep['raised_max_um'] = round(float(dv.max()) * 1e6, 1)
    elif todo.any():
        T[todo] = (np.where(Tg[todo] < 0, -1.0, 1.0) * cap0).astype(np.float32)
    return T, rep


def _write_txn(prep, S, F8, out_dir, note, check, timeout, t0):
    """FV8_WRITER=txn: each tile group's change as ONE fv8_write 'write_txn' op whose target is the object's own field
    (F_obj) at every node where it differs from her store, so the store's allocation, shared faces and coarse levels are
    written from that one target against one fixed source (engine lane, 11-10)"""
    FW = F8.fv8_write_module()
    h = prep['h']; T = T_NODES; reps, tiles_out = [], []
    for grp in tile_groups(prep['tiles']):
        ga = np.array(grp); lo = ga.min(0); n = (ga.max(0) - lo + 1) * T
        Tg = np.full(tuple(int(v) for v in n), np.nan, np.float32); nz = 0
        for t in grp:
            td = tile_delta(prep, t, want_obj=True)
            if td is None: continue
            origin_t, D, fo = td; o = (np.asarray(t) - lo) * T
            Tg[o[0]:o[0] + T, o[1]:o[1] + T, o[2]:o[2] + T] = np.where(D != 0, fo, np.nan); nz += int((D != 0).sum())
            del D, fo
        if not nz: continue
        origin = prep['org'] + lo * T * h
        drep = None
        if os.environ.get('FS_WB_DIST', '1') != '0':           # fs-trust 11-11: the store gets distances (contract item d)
            Tg, drep = _dist_target(prep, origin, Tg, F8)
        op = FW.target_op(origin, Tg, l0_rebuild=os.environ.get('FV8_TXN_L0REBUILD', '1') != '0',
                          source='Field Studio editable store write-back (txn)', object=prep['name'])
        r = S.apply(op)
        reps.append({k: r.get(k) for k in ('target_nodes', 'surface_nodes_changed', 'check', 'write_s', 'latency_ms')})
        reps[-1]['distance_target'] = drep
        tiles_out.append({'origin': origin.tolist(), 'dims': list(Tg.shape), 'tiles': len(grp), 'nonzero': nz})
        del Tg, op
    if not tiles_out:
        raise ValueError('nothing to write: the object equals her store')
    t1 = time.perf_counter()
    nt = note or 'Field Studio write-back of %s (%d groups, write_txn); NOT published' % (prep['name'], len(reps))
    attr_rep = None
    if os.environ.get('FV8_TXN_ATTR', '1') != '0':     # the skin's attributes go with the moved surface (engine lane 11-11; F7)
        import json, shutil
        FA = F8.fv8_attrs_module(); pre = out_dir + '.pre_attr'
        S.commit(pre, note=nt)
        ch = json.load(open(os.path.join(pre, 'version.json'))).get('changed_bricks', {}).get('L0', [])
        r_ = FA.transport(F8.FieldStore(prep['source']), pre, out_dir, None, None, ids=np.asarray(ch, np.int64), both_sides=True,
                          note='Field Studio write-back (txn): attributes moved with the surface')
        attr_rep = {k: r_.get(k) for k in ('bricks', 'lattice_nodes', 'nodes_transported', 'nodes_refused_far_source', 'rest_moved_mm_p50_max')}
        shutil.rmtree(pre)
    else:
        S.commit(out_dir, note=nt)
    rep = {'source': prep['source'], 'new_version': out_dir, 'writer': 'write_txn', 'attr_transport': attr_rep, 'clusters': tiles_out,
           'tiles_considered': len(prep['tiles']), 'ops': reps, 'apply_s': round(t1 - t0, 2),
           'commit_s': round(time.perf_counter() - t1, 2), 'published': False}
    if check:
        rep['guard'] = region.guard_check(out_dir, timeout=timeout)
    return rep


def write(prep, out_dir, note='', check=True, timeout=900):
    """compute and apply the tiles' deltas one at a time to her source version and commit a NEW version at out_dir"""
    t0 = time.perf_counter()
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError('%s exists: write-back makes a NEW version folder only' % out_dir)
    src = prep['source']
    if prep.get('source_id') and fv8base.source_id(src) != prep['source_id']:
        raise ValueError('her source version changed since it was opened')
    F8 = region.fv8(src)
    S = F8.FieldStore(src)
    reps, tiles_out = [], []
    if _writer(F8) == 'txn':          # engine lane 11-10; the default since 11-11 (fs-trust): one transaction per tile group
        return _write_txn(prep, S, F8, out_dir, note, check, timeout, t0)
    for t in prep['tiles']:
        td = tile_delta(prep, t)
        if td is None:
            continue
        origin, D = td
        op = F8.delta_grid_op(origin, prep['h'], D, allocate=True, source='Field Studio editable store write-back',
                              object=prep['name'], **({'writer': 'legacy'} if hasattr(F8, 'writer_mode') else {}))
        r = S.apply(op)
        reps.append({k: r.get(k) for k in ('delta_max_abs_um', 'lip_bound_D', 'latency_ms')})
        tiles_out.append({'origin': origin.tolist(), 'dims': list(D.shape), 'nonzero': int((D != 0).sum())})
        del D, op
    if not tiles_out:
        raise ValueError('nothing to write: the object equals her store')
    t1 = time.perf_counter()
    S.commit(out_dir, note=note or 'Field Studio write-back of %s (%d tiles); NOT published' % (prep['name'], len(reps)))
    rep = {'source': src, 'new_version': out_dir, 'clusters': tiles_out, 'tiles_considered': len(prep['tiles']),
           'ops': reps, 'apply_s': round(t1 - t0, 2), 'commit_s': round(time.perf_counter() - t1, 2), 'published': False}
    if check:
        rep['guard'] = region.guard_check(out_dir, timeout=timeout)
    return rep
