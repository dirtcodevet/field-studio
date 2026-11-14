"""Protected parts as colliders for every edit (11-09; design: a lid shaped toward the eye slides
over the globe at the gap instead of sinking into it).

Rule (applied to the nodes an edit changed, after each dab on the CPU / each batch on the card):
    B(x)  = gap - min over the locked protected parts of F_part(x)        (> 0: inside a part dilated by the gap)
    F'(x) = max(F_new(x), min(F_start(x), B(x)))   where B(x) > 0;   F'(x) = F_new(x) elsewhere
so an edit never moves the editable surface into a part + gap, a node that already overlapped (her body still holds
copies of her eyes and other parts: missing anatomy, a separate lane builds it) is never pushed deeper, and a node the
edit did not change keeps its bits (F_new = F_start >= min(F_start, B)).
Parts that are unlocked (Edit > Unprotect) are editable and do not collide; locked parts collide whether shown or not.
Contacts between protected parts themselves are not touched (they are never edited while locked).
Settings: 'collide' (the panel turns it on for every brush; missing = off, so older strokes replay unchanged) and
'collide_gap' (m, default GAP).
"""
import os

import numpy as np

from .scene import is_protected

GAP = 0.0003
LAST_SETTLE = {}     # the last Settle dab's amount and clearance (diagnostics)
BAND_CELLS = 2.0   # r2 (test_swell_settle r1: the surface stopped 0.25 mm from a part at gap 0.3): the rule also holds a band
                   # of 2 lattice cells OUTSIDE the part + gap (B > -2 h), so the field next to the contact follows B (a
                   # distance) and the zero crossing lands at the gap instead of being pinned to the last clamped node


def part_partners(scene, target_id, modes=('touch', 'slide')):
    """fs-contact 11-11: the store parts (core/fv8parts.py part records) paired with the target part by one of `modes`
    (touch / slide: they touch, never fuse, so a stroke on one never pushes it into the other; bond and pass do not collide)"""
    try:
        t = scene.get(target_id)
    except Exception:
        return []
    tp = getattr(t, 'part', None)
    if not tp:
        return []
    pairs = tp.get('pairs') or {}
    out = []
    for o in scene.objects:
        op = getattr(o, 'part', None)
        if o.id == target_id or not op or op.get('source') != tp.get('source'):
            continue
        if pairs.get(str(op.get('pid'))) in modes:
            out.append(o)
    return out


def colliders(scene, target_id, extra=()):
    """locked protected parts other than the target, plus `extra` object ids (Swell / Settle contacts: the selected
    objects), plus (fs-contact 11-11) the target's touch / slide partners among her store parts"""
    ex = set(int(i) for i in (extra or ()))
    out = [o for o in scene.objects if o.id != target_id and (is_protected(o) or o.id in ex)]
    ids = {o.id for o in out}
    return out + [o for o in part_partners(scene, target_id) if o.id not in ids]


def _parts_near(parts, lo_w, hi_w, gap):
    from ..tools.protected_merge import world_box
    out = []
    for p in parts:
        a, b = world_box(p)
        if np.all(a - gap <= hi_w) and np.all(b + gap >= lo_w):
            out.append(p)
    return out


def bound(scene, pk, parts, W, gap):
    """B at world points W (n, 3): gap - min_part F_part (parts culled by their world boxes)"""
    from ..tools.protected_merge import field_of
    B = np.full(len(W), -np.inf)
    if not len(W):
        return B
    for p in _parts_near(parts, W.min(0), W.max(0), gap + 0.002):   # + the outside band (BAND_CELLS h, h <= 1 mm)
        gp = 0.0 if getattr(p, 'part', None) else gap                # fs-contact 11-11: store parts touch (gap 0); others keep the gap
        Bp = gp - field_of(scene, pk, p, W)
        if getattr(p, 'root', None):            # 11-09: an attached part's root is fused to the body: no collision there
            from .attach import root_weight
            from ..tools.protected_merge import _to_local
            Bp[root_weight(p, _to_local(p, W)) > 0.0] = -np.inf
        B = np.maximum(B, Bp)
    return B


def _bound_bricks(scene, pk, parts, obj, bl, idx, h, gap, lo_w, hi_w):
    """fs-brush 11-12: bound() at the 512 nodes of each brick in bl (m, 3), as one field evaluation per part -> (m, 512),
    equal brick by brick to bound(scene, pk, parts, the brick's world nodes, gap): a part counts only for the bricks whose
    node box (lo_w / hi_w: the corners' world box) its world box + gap + 2 mm meets, as bound's cull does"""
    from ..tools.protected_merge import field_of, _to_world, world_box
    m = len(bl)
    B = np.full((m, 512), -np.inf)
    if not m:
        return B
    W = _to_world(obj, ((np.asarray(bl, np.int64)[:, None, :] * 8 + idx[None]) * h).reshape(-1, 3)).reshape(m, 512, 3)
    pad = gap + 0.002
    for p in parts:
        a, b = world_box(p)
        sel = np.all(a - pad <= hi_w, 1) & np.all(b + pad >= lo_w, 1)
        if not sel.any():
            continue
        Ws = np.ascontiguousarray(W[sel].reshape(-1, 3))
        gp = 0.0 if getattr(p, 'part', None) else gap                # as bound(): store parts touch (gap 0)
        Bp = gp - field_of(scene, pk, p, Ws)
        if getattr(p, 'root', None):
            from .attach import root_weight
            from ..tools.protected_merge import _to_local
            Bp[root_weight(p, _to_local(p, Ws)) > 0.0] = -np.inf
        B[sel] = np.maximum(B[sel], Bp.reshape(-1, 512))
    return B


def clamp_dab(scene, pk, obj, oi, lo, h, df, settings):
    """CPU brush: df (the dab's change on the node grid starting at node lo) clamped in place by the rule. -> nodes
    clamped"""
    if not settings.get('collide', False) or settings.get('tool') == 'mask':
        return 0
    parts = colliders(scene, obj.id, settings.get('contact_ids'))
    if not parts:
        return 0
    from ..tools.protected_merge import _to_world
    from . import program as P
    gap = float(settings.get('collide_gap', GAP))
    # r2: the dab's box against the parts' boxes first (no node is evaluated when it misses every part + gap)
    c_ = np.array([[x, y, z] for x in (0, df.shape[0]) for y in (0, df.shape[1]) for z in (0, df.shape[2])], float)
    Cw = _to_world(obj, (np.asarray(lo, float)[None] + c_) * h)
    if not _parts_near(parts, Cw.min(0), Cw.max(0), gap + BAND_CELLS * h * float(obj.transform['scale'])):
        return 0
    idx = np.argwhere(df != 0.0)
    if not len(idx):
        return 0
    X = (idx + np.asarray(lo)[None]) * h
    W = _to_world(obj, X)
    B = bound(scene, pk, parts, W, gap)
    act = B > -BAND_CELLS * h * float(obj.transform['scale'])
    if not act.any():
        return 0
    idx, X, B = idx[act], X[act], B[act]
    F0 = np.empty(len(X))
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X), F0)
    d = df[idx[:, 0], idx[:, 1], idx[:, 2]]
    lim = np.minimum(0.0, B / max(float(obj.transform['scale']), 1e-9) - F0)   # local units (B is world)
    dn = np.maximum(d, lim)
    ch = dn != d
    df[idx[ch, 0], idx[ch, 1], idx[ch, 2]] = dn[ch]
    return int(ch.sum())


def overlap_count(scene, obj, gap=GAP, n_max=200000):
    """'new overlaps only' check helper: surface nodes of obj (|F| <= h) inside a locked part + gap, minus those that were
    inside already in `baseline` (a set of node keys) when given by the caller. -> (count, node keys)"""
    from ..tools.protected_merge import _to_world
    from . import program as P
    parts = colliders(scene, obj.id)
    g = obj.sculpt
    st = scene.store
    if not parts or not g.slots:
        return 0, set()
    pk = scene.pack()
    oi = scene.index_of(obj.id)
    h = float(g.h)
    idx = np.stack(np.meshgrid(*[np.arange(8)] * 3, indexing='ij'), -1).reshape(-1, 3)
    keys = []
    for b, s in g.slots.items():
        if s >= 0 and st.delta[s].any():
            keys.append(np.asarray(b) * 8 + idx)
    if not keys:
        return 0, set()
    K = np.concatenate(keys)[:n_max]
    X = K * h
    F = np.empty(len(X))
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X), F)
    near = np.abs(F) <= h
    K, X, F = K[near], X[near], F[near]
    B = bound(scene, pk, parts, _to_world(obj, X), gap)
    bad = (B > 1e-9) & (F < B - 1e-6)
    return int(bad.sum()), set(map(tuple, K[bad].tolist()))


def clamp_bricks(scene, oid, bricks, rec, settings):
    """after a stroke's re-distance (CPU): the rule over the stroke's bricks, with F_start from the stroke record (r2:
    the re-distance moved clamped nodes back by 36-62 um, test_collide's diagnosis). -> nodes changed"""
    if not settings.get('collide', False) or settings.get('tool') == 'mask':
        return 0
    obj = scene.get(oid)
    parts = colliders(scene, oid, settings.get('contact_ids'))
    if not parts or not bricks:
        return 0
    from ..tools.protected_merge import _to_world
    from . import program as P
    gap = float(settings.get('collide_gap', GAP))
    g = obj.sculpt
    st = scene.store
    h = float(g.h)
    oi = scene.index_of(oid)
    idx = np.stack(np.meshgrid(*[np.arange(8)] * 3, indexing='ij'), -1).reshape(-1, 3)
    sc_ = max(float(obj.transform['scale']), 1e-9)
    n = 0
    pk = scene.pack()
    changed = set()
    # 11-09 (a large stroke's release on her spent 3.2 s here, the parts' world boxes recomputed for each of 30k
    # bricks): the same cull bound() applies (a part counts where its world box + gap + 2 mm meets the nodes' world box)
    # done once for every brick; a brick no part reaches gets B = -inf everywhere and is skipped, as before
    from ..tools.protected_merge import world_box
    allb = np.array(sorted(set(map(tuple, bricks))), np.int64).reshape(-1, 3)
    if not len(allb):
        return 0
    pad = gap + 0.002
    pbox = [world_box(p_) for p_ in parts]
    corners = np.array([[x, y, z] for x in (0, 7) for y in (0, 7) for z in (0, 7)], np.int64)
    Cw = _to_world(obj, ((allb[:, None, :] * 8 + corners[None]) * h).reshape(-1, 3)).reshape(len(allb), 8, 3)
    lo_w, hi_w = Cw.min(1), Cw.max(1)
    near = np.zeros(len(allb), bool)
    for a_, b_ in pbox:
        near |= np.all(a_ - pad <= hi_w, 1) & np.all(b_ + pad >= lo_w, 1)
    # fs-brush 11-12 (fs-bench-full F3, M f3/f3_cpu_probe.py: a Smooth r 4 lid stroke on her CPU path spent 10.5 of its
    # 13.3 s here, bound() called once per brick near her eyes: 401 serial calls of 512 points, ~26 ms each): the parts'
    # bound for every near brick at once, one evaluation per part (each part still counted only for the bricks its box
    # meets, as bound's cull): the same values, so the same clamp
    nbl = allb[near]
    Bb = _bound_bricks(scene, pk, parts, obj, nbl, idx, h, gap, lo_w[near], hi_w[near])
    for i_, b in enumerate(map(tuple, nbl.tolist())):
        s = g.get_slot(b)
        # 11-09 (the script API's replay check): a missing brick is treated as an empty one. Live scenes keep empty
        # bricks (a cleared mask's) that a saved or replayed scene does not have, and skipping the missing ones made a
        # Settle stroke's result depend on which bricks happened to exist; it is allocated (through rec) only if the
        # rule changes it.
        X = (np.asarray(b) * 8 + idx) * h
        B = Bb[i_] / sc_
        act = B > -BAND_CELLS * h
        if not act.any():
            continue
        Fn = np.empty(len(X))
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X), Fn)
        d_now = st.delta[s].reshape(-1).astype(np.float64) if s >= 0 else np.zeros(512)
        snap = rec.snap.get((oid, b)) if rec is not None else None
        d0 = snap[0].reshape(-1).astype(np.float64) if snap is not None else np.zeros(512)
        F0 = Fn - d_now + d0
        lim = np.minimum(F0, B)
        up = act & (Fn < lim)
        if up.any():
            if s < 0:
                if g.ensure_index(np.asarray(b), np.asarray(b)):
                    scene.touch(index=True)
                s = st.alloc()
                g.set_slot(b, s)
                if rec is not None:
                    rec.new.append((oid, b))
                    rec.new_set.add((oid, b))
                scene.touch(index=True)
                pk = scene.pack()
            dd = st.delta[s].reshape(-1)
            dd[up] = (d_now[up] + (lim[up] - Fn[up])).astype(np.float32)
            n += int(up.sum())
            changed.add(b)
    if n:
        scene.refresh_object(oi)
        from . import brush as BR
        BR.measure_bricks(scene, oi, changed)        # the tracer's bound of the clamped bricks (was left as before)
        scene.touch()
    return n


def new_overlaps(scene, obj, gap=GAP, max_nodes=3000000):
    """'new overlaps only': surface nodes (|F| <= h) of the editable object obj that lie inside a locked part + gap (F < B)
    where its START did not (its base field, the sculpt layer left out: F_base >= B). Nodes over the parts' world
    boxes + gap. -> (count, mode label)"""
    from ..tools.protected_merge import _eval_exact, _to_local, _to_world, _rs, world_box
    from . import program as P
    parts = colliders(scene, obj.id)
    if not parts:
        return 0, 'new overlaps only'
    pk = scene.pack()
    oi = scene.index_of(obj.id)
    h = float(obj.sculpt.h)
    tab = pk.oi_tab.copy()
    tab[oi, P.OI_NX] = 0
    n = 0
    for p in parts:
        lo, hi = world_box(p)
        cw = np.array([[x, y, z] for x in (lo[0] - gap, hi[0] + gap) for y in (lo[1] - gap, hi[1] + gap)
                       for z in (lo[2] - gap, hi[2] + gap)])
        cl = _to_local(obj, cw)
        a = np.floor(cl.min(0) / h).astype(np.int64); b = np.ceil(cl.max(0) / h).astype(np.int64)
        if np.prod(b - a + 1) > max_nodes:
            step = int(np.ceil((np.prod(b - a + 1) / max_nodes) ** (1 / 3)))
        else:
            step = 1
        K = np.stack(np.meshgrid(*[np.arange(a[i], b[i] + 1, step) for i in range(3)], indexing='ij'), -1).reshape(-1, 3)
        X = np.ascontiguousarray(K * h, np.float64)
        F = np.empty(len(X))
        _eval_exact(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, X, F)
        s_ = _rs(obj)[2]
        near = np.abs(F) <= h
        X, F = X[near], F[near] * s_
        if not len(X):
            continue
        B = bound(scene, pk, parts, _to_world(obj, X), gap)
        c = (B > 1e-9) & (F < B - 1e-6)
        if not c.any():
            continue
        Fb = np.empty(int(c.sum()))
        _eval_exact(pk.code, pk.prm, tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X[c]), Fb)
        n += int(np.sum(Fb * s_ >= B[c] - 1e-6))
    return n, 'new overlaps only'


def settle_dab(scene, pk, obj, oi, lo, h, c, n, r, strength, lut, df, settings):
    """Settle onto (11-09): each node in the dab's sphere reads the field from a point further from the nearest
    contact (its field's gradient direction g), F'(x) = F(x + g A w): the surface moves toward the contact by A w,
    and clamp_dab (called after) stops it at the gap, so repeated dabs rest it on the contact and it conforms.
    A = strength r / 2 (the brushes' amount). df (the dab grid from node lo) is filled in place."""
    from ..tools.protected_merge import _to_world, _to_local, field_of, _rs
    from . import program as P
    from .brush import _lut
    parts = colliders(scene, obj.id, settings.get('contact_ids'))
    if not parts:
        return 0
    ni, nj, nk = df.shape
    K = np.stack(np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing='ij'), -1).reshape(-1, 3)
    X = (K + np.asarray(lo)[None]) * h
    d = np.linalg.norm(X - np.asarray(c)[None], axis=1)
    sel = d < r
    if not sel.any():
        return 0
    K, X, d = K[sel], X[sel], d[sel]
    w = np.array([_lut(lut, float(u)) for u in d / r])
    keep = w > 0
    K, X, w = K[keep], X[keep], w[keep]
    W = _to_world(obj, X)
    near = _parts_near(parts, W.min(0), W.max(0), 0.05)
    if not near:
        return 0
    near_parts = near
    e = 0.5 * h * _rs(obj)[2]
    Tm = np.full(len(W), np.inf)
    G = np.zeros((len(W), 3))
    for p in near:
        Tp = field_of(scene, pk, p, W)
        Gp = np.stack([field_of(scene, pk, p, W + e * a) - field_of(scene, pk, p, W - e * a) for a in np.eye(3)], 1)
        better = Tp < Tm
        Tm[better] = Tp[better]
        G[better] = Gp[better]
    G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    Rm = _rs(obj)[0]
    Gl = G @ Rm                                    # world direction -> the object's local frame (rotation only)
    # r6 (test_settle_thickness r1: a settled 2 mm slab -9.2 % in the middle, +5.5 % at the edges): the falloff was
    # measured from the dab centre, so the two faces of a sheet moved by different amounts. Now it is measured ACROSS
    # the move direction (a column through the sheet moves as one), with a taper only far along it (|along| > r / 2),
    # so a thick body is not sheared at the brush's edge
    dx_ = X - np.asarray(c)[None]
    # r7 (edges -7.7 %: each node moved along its own direction to the contact, which fans out, so a column's two faces
    # moved by different amounts): one direction per dab, the contact's at the node nearest the dab centre
    gc = Gl[int(np.argmin(np.linalg.norm(dx_, axis=1)))]
    Gl = np.broadcast_to(gc, Gl.shape).copy()
    da = np.abs((dx_ * Gl).sum(1))
    dt = np.linalg.norm(dx_ - (dx_ * Gl).sum(1)[:, None] * Gl, axis=1)
    tp = np.clip((r - da) / (0.5 * r), 0.0, 1.0)
    w = np.array([_lut(lut, float(min(u, 1.0))) for u in dt / r]) * (tp * tp * (3.0 - 2.0 * tp))
    from .brush import dab_amount               # 11-09 night: the amount per pass, as every brush (core/brush.py)
    A = dab_amount(dict(settings, strength=strength), r)
    F0 = np.empty(len(X))
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X), F0)
    # r2 (contact_sheet r1: a 2 mm slab settled onto an eye lost its middle: once its near surface was held at contact,
    # the far surface kept coming down): a dab moves by at most the CLEARANCE under the brush, the smallest distance
    # from the body's target-facing surface in the brush core (w >= 0.5) to the contact; touching, it stops and keeps
    # its thickness; later dabs settle what is not touching yet
    gap = float(settings.get('collide_gap', GAP))
    sc_ = _rs(obj)[2]
    COLA = None
    near = (np.abs(F0) <= h) & ((w > 0.0) if SETTLE_COLUMNS else (w >= 0.5))
    if near.any():
        Xn = np.ascontiguousarray(X[near])
        Gb = np.zeros((len(Xn), 3))
        for ax in range(3):
            a_ = np.empty(len(Xn)); b_ = np.empty(len(Xn)); d_ = np.zeros(3); d_[ax] = 0.5 * h
            P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(Xn + d_), a_)
            P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(Xn - d_), b_)
            Gb[:, ax] = (a_ - b_) / h                   # the field's gradient (local), its magnitude kept
        gm = np.maximum(np.linalg.norm(Gb, axis=1), 1e-12)
        Gn = Gb / gm[:, None]
        facing = (Gn * Gl[near]).sum(1) < -0.5          # the body's outward normal points toward the contact
        facing &= Tm[near] >= gap                     # r4: nodes inside the contact + gap hold clamped values, not geometry
        if facing.any():
            # at the SURFACE the node belongs to: one Newton step x - F grad / |grad|^2 (r3: read at the nodes the slab
            # stopped 0.15 mm short; r5: F + T at the nodes, 0.053 mm short: Settle stretches the field, |grad F| != 1)
            Xs = Xn[facing] - (F0[near][facing] / gm[facing] ** 2)[:, None] * Gb[facing]
            Ws = _to_world(obj, Xs)
            Ts = np.full(len(Ws), np.inf)
            for p in near_parts:
                Ts = np.minimum(Ts, field_of(scene, pk, p, Ws))
            clear = float(np.min(Ts - gap)) / sc_
            if SETTLE_COLUMNS:
                # sculpt-contact 12:2x (sheets 1 and 5: Settle never moved her lid, because one point under the brush
                # already touched her eye and A = min over the brush core = 0): each column across the move direction
                # settles by ITS OWN clearance (the facing surface point nearest it across the move), so the parts not
                # touching yet come down while the touching ones hold; a column keeps its thickness
                from scipy.spatial import cKDTree
                g_ = Gl[0]
                b1 = np.cross(g_, [1.0, 0.0, 0.0] if abs(g_[0]) < 0.9 else [0.0, 1.0, 0.0]); b1 /= np.linalg.norm(b1)
                b2 = np.cross(g_, b1)
                tq = lambda Z: np.stack([Z @ b1, Z @ b2], 1)
                d_c, j_c = cKDTree(tq(Xs)).query(tq(X), k=1)
                col = np.maximum((Ts - gap) / sc_, 0.0)[j_c]
                col[d_c > 2.0 * h] = np.inf               # no facing surface measured in that column: free (clamp_dab after)
                COLA = np.minimum(A * w, col)
            else:
                A = min(A, max(clear, 0.0))
    LAST_SETTLE.clear(); LAST_SETTLE.update(A=A, clear=locals().get('clear'), n_facing=int(facing.sum()) if near.any() else 0)
    if COLA is not None:
        if not np.any(COLA > 0.0):
            return 0
        Y = np.ascontiguousarray(X + Gl * COLA[:, None])
        F1 = np.empty(len(Y))
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, Y, F1)
        df[K[:, 0], K[:, 1], K[:, 2]] = F1 - F0
        return int(np.sum(COLA > 0.0))
    if A <= 0.0:
        return 0
    Y = np.ascontiguousarray(X + Gl * (A * w)[:, None])
    F1 = np.empty(len(Y))
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, Y, F1)
    df[K[:, 0], K[:, 1], K[:, 2]] = F1 - F0
    return int(len(K))


# sculpt-contact 11-12 (the true state, sheets 1-3: Swell was plain inflate, a fixed rate along every normal: a lid
# swelled toward her eye grew 1.1-1.45 mm OUTWARD, and 8 strokes closed a fold only from 3.55 to 1.25 mm toward its
# locked neighbour): Swell to contact grows the surface that FACES a contact TOWARD it, by a share of the gap that remains,
# so each pass eases it closer and it comes to rest touching (the collider still clamps); the far side does not grow.
SETTLE_COLUMNS = os.environ.get('SC_SETTLE_COLUMNS', '1') == '1'   # Settle: each column by its own clearance
SWELL_TOWARD = os.environ.get('SC_SWELL_TOWARD', '1') == '1'
SWELL_MODE = os.environ.get('SC_SWELL_MODE', 'ease')     # 'ease': the balloon (inflate along the normals) easing into
#   the contact (a 'toward' growth was tried and rejected: t_swell_synth r1, a flat mesa with horns wrapping the part)
SWELL_EASE = float(os.environ.get('SC_SWELL_EASE', 0.25))  # the balloon slows over the last EASE x r of its gap
SWELL_GRAD_CELLS = float(os.environ.get('SC_SWELL_GRAD_CELLS', 2.0))
SWELL_GAIN = float(os.environ.get('SC_SWELL_GAIN', 3.0))   # Swell to contact's amount over Inflate's (t_swell_synth r2:
#   at Inflate's 0.36 mm a stroke (r 4, s 0.5) a 3 mm gap took 8+ strokes; the ease keeps the approach smooth)
SWELL_REACH = 2.0          # a gap longer than REACH x r is not pulled across (it fades out from r to REACH x r)


LAST_SWELL = {}


def swell_params(settings, r, h, sc_):
    """Swell to contact's constants for a dab (one place for the CPU and the card, gpu/sculpt.cu typ 11 with bp[46]):
    (gain, ease length E, reach R, contact gap), lengths in the object's local units"""
    gap = float(settings.get('collide_gap', GAP)) / max(sc_, 1e-12)
    return SWELL_GAIN, max(SWELL_EASE * r, 2.0 * h), SWELL_REACH * r, gap


def swell_k(T, F0, gT, gB, r, h, gain, E, R, gap):
    """the multiplier of the balloon's own growth at nodes (local units): T the contacts' distance at the node, F0 the
    body's field, gT the contacts' gradient, gB the body's (normal) gradient.
       gs   = max(T - gap + F0, 0)           the gap from the body's surface there to the contact
       near = 1 within r of a contact, 0 beyond R (linear between): far from every contact Swell is the balloon as it was
       ease = 1 - exp(-(gs + h/2) / E)       slows over the last E of the gap; h/2 so the collider finishes the touch
       face = (1 + cos) / 2                  full where the surface faces the contact, half side-on, none facing away
       k    = 1 + near (gain ease face - 1)"""
    gs = np.maximum(T - gap + F0, 0.0)
    near = np.clip((R - gs) / max(R - r, 1e-12), 0.0, 1.0)
    ease = 1.0 - np.exp(-(gs + 0.5 * h) / E)
    gl = np.linalg.norm(gT, axis=1); bl = np.linalg.norm(gB, axis=1)
    ok = (gl > 1e-12) & (bl > 1e-12)
    cs = np.zeros(len(T))
    cs[ok] = -(gT[ok] * gB[ok]).sum(1) / (gl[ok] * bl[ok])
    face = 0.5 + 0.5 * np.clip(cs, -1.0, 1.0)
    return 1.0 + near * (gain * ease * face - 1.0)


def swell_ease(scene, pk, obj, oi, lo, h, r, df, settings):
    """Swell to contact on the CPU path (sculpt-contact 11-12): the kernel's balloon (df: inflate along the normals, a
    soft dome from the falloff) times swell_k at each node it moves; collide.clamp_dab (after) keeps every node out of the
    contacts. The card computes the same in gpu/sculpt.cu (typ 11, bp[46]). -> nodes whose growth changed"""
    from ..tools.protected_merge import _to_world, field_of, _rs
    from . import program as P
    idx = np.argwhere(df < 0.0)
    if not len(idx):
        return 0
    parts = colliders(scene, obj.id, settings.get('contact_ids'))
    if not parts:
        return 0
    Rm, _, sc_ = _rs(obj)
    gain, E, R, gap = swell_params(settings, r, h, sc_)
    X = np.ascontiguousarray((idx + np.asarray(lo)[None]) * h, np.float64)
    W = _to_world(obj, X)
    near = _parts_near(parts, W.min(0), W.max(0), (R + h) * sc_ + gap * sc_)
    if not near:
        return 0
    T = np.full(len(W), np.inf)
    gT = np.zeros((len(W), 3))
    e = h * sc_
    for p in near:
        Tp = field_of(scene, pk, p, W)
        better = Tp < T
        if not better.any():
            continue
        T[better] = Tp[better]
        Wb = W[better]
        gT[better] = np.stack([field_of(scene, pk, p, Wb + e * a) - field_of(scene, pk, p, Wb - e * a) for a in np.eye(3)], 1)
    T = T / sc_
    gT = gT @ Rm                                   # world -> the object's local frame (rotation only)
    F0 = np.empty(len(X))
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, X, F0)
    eg = SWELL_GRAD_CELLS * h                      # fold8: the body's normal at the form's scale, not its texture's
    gB = np.zeros((len(X), 3))
    for ax in range(3):
        a_ = np.empty(len(X)); b_ = np.empty(len(X)); dd = np.zeros(3); dd[ax] = eg
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X + dd), a_)
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X - dd), b_)
        gB[:, ax] = a_ - b_
    fin = np.isfinite(T)
    k = np.ones(len(X))
    if fin.any():
        k[fin] = swell_k(T[fin], F0[fin], gT[fin], gB[fin], r, h, gain, E, R, gap)
    df[idx[:, 0], idx[:, 1], idx[:, 2]] *= k
    LAST_SWELL.clear(); LAST_SWELL.update(n=int(len(idx)), k_mean=float(k.mean()), k_max=float(k.max()))
    return int(np.sum(np.abs(k - 1.0) > 1e-9))


def warm_cpu_paths():
    """the CPU brush paths (the tagged Mask, the collider, Swell / Settle) run once on a throwaway scene so their
    compiled code is loaded before the first real stroke (app run 11-09: 1.75-1.9 s on the first Mask dab, then ~50 ms)"""
    from .history import History, AddObject, BrushStroke
    h = History()
    b = h.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.02}, 'name': 'warm', 'h': 0.0005}))
    h.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.004}, 'name': 'warm part (protected)',
                      'translate': [0.0, 0.0225, 0.0], 'h': 0.0005}))
    smp = [{'p': [0.0, 0.02, 0.0], 'n': [0.0, 1.0, 0.0]}]
    for st in ({'tool': 'mask', 'radius': 0.003, 'strength': 0.5, 'paint_dir': True, 'connected': True},
               {'tool': 'clay', 'radius': 0.003, 'strength': 0.5, 'collide': True},
               {'tool': 'swell', 'radius': 0.003, 'strength': 0.5}, {'tool': 'settle', 'radius': 0.003, 'strength': 0.5}):
        h.push(BrushStroke(b.id, st, smp, redistance=False))
