"""The card's collider step for brush strokes (11-09: core/collide.py's rule on the card; fs-brush 11-12: moved out of
gpu/sculpt.py into its own module at the split agreed with sculpt-contact: sculpt-contact owns this
file, core/collide.py and the Swell / Settle behaviour; fs-brush's stroke pipeline calls it through GpuSculpt._collide /
_collide_rows, and the kernel (collide_clamp) stays in gpu/sculpt.cu as gs.k_collide). Moved verbatim (self -> gs).

collide(gs, scene, settings, plans, live): after a batch's dabs, every touched row near a collider part gets
F := max(F, min(F_start, B)) where B > 0 (B the parts' bound at its 729 nodes, cached for the stroke).
collide_rows(gs, ...): the same for given rows (the release calls it after the re-distance).
rows_far(gs, ...): True when the box of all the given bricks misses every part's box + the per-row margin."""
import os

import numpy as np

COLLIDE_BOX = os.environ.get('FSB_COLLIDE_BOX', '1') != '0'   # fs-brush 11-12: the collider step skips batches far from every part


def collide(gs, scene, settings, plans, live):
    """11-09: core/collide.py's rule on the card, after the batch's dabs: per touched row, B at its 729 nodes (cached for
    the stroke; rows wholly outside every part + gap are skipped), then F := max(F, min(F_start, B)) where B > 0"""
    from ..core import collide as CL
    from ..tools.protected_merge import _to_world
    cp = gs.cp
    n = 0
    gap = float(settings.get('collide_gap', CL.GAP))
    pk = None
    for oi in sorted({p['oi'] for p in plans}):
        obj = scene.objects[oi]
        parts = CL.colliders(scene, obj.id, settings.get('contact_ids'))
        if not parts:
            continue
        if COLLIDE_BOX:
            # fs-brush 11-12 (decay series dk1: chest strokes on her paid 60-218 ms a stroke here for her EYES, 0.2 m
            # away, in per-row Python): the batch's rows as ONE box first; when it misses every part's box + the
            # margin _collide_rows tests each row's box with, every row's does too: none is near, the same result
            vol0 = next(p['vol'] for p in plans if p['oi'] == oi)
            ka = np.concatenate([np.asarray(p['keys'], np.int64).reshape(-1, 3) for p in plans if p['oi'] == oi])
            if len(ka) == 0 or rows_far(gs, obj, vol0, ka, parts, live, oi, gap):
                continue
        if pk is None:
            pk = scene.pack()
        cache = live.setdefault('B', {}).setdefault(oi, {})
        rk = {}
        for p in plans:
            if p['oi'] == oi:
                for r_, k_ in zip(p['rows'].tolist(), p['keys'].tolist()):
                    rk[r_] = k_
        vol = next(p['vol'] for p in plans if p['oi'] == oi)
        n += collide_rows(gs, scene, obj, oi, parts, rk, vol, live, gap, pk)
    return n

def rows_far(gs, obj, vol, keys, parts, live, oi, gap):
    """fs-brush 11-12: True when the box of all these GPU bricks (ijk) misses every collider part's world box + the
    margin _collide_rows tests each row with (so every row would be decided 'not near' there)"""
    from ..core import collide as CL
    from ..tools.protected_merge import _to_world
    pb = live.setdefault('pboxes', {}).get(oi)
    if pb is None:
        from ..tools.protected_merge import world_box
        pb = np.array([np.r_[world_box(p_)] for p_ in parts]).reshape(-1, 2, 3)
        live['pboxes'][oi] = pb
    h = float(obj.sculpt.h)
    bm = 8 * h
    klo, khi = keys.min(0).astype(float), (keys.max(0) + 1).astype(float)
    cb = np.array([[x, y, z] for x in (klo[0], khi[0]) for y in (klo[1], khi[1]) for z in (klo[2], khi[2])])
    Cb = _to_world(obj, vol.origin[None] + cb * bm)
    blo, bhi = Cb.min(0), Cb.max(0)
    mg_ = gap + CL.BAND_CELLS * h * float(obj.transform['scale'])
    return not any(np.all(blo <= b_ + mg_) and np.all(bhi >= a_ - mg_) for a_, b_ in pb)

def collide_rows(gs, scene, obj, oi, parts, rk, vol, live, gap, pk):
    from ..core import collide as CL
    from ..tools.protected_merge import _to_world
    cp = gs.cp
    n = 0
    if COLLIDE_BOX and rk:
        # fs-brush 11-12: the release's call (every row the stroke touched) asks the one-box question first too
        if rows_far(gs, obj, vol, np.array(list(rk.values()), np.int64).reshape(-1, 3), parts, live, oi, gap):
            return n
    if True:
        cache = live.setdefault('B', {}).setdefault(oi, {})
        new = [r for r in rk if r not in cache]
        if new:
            h = float(obj.sculpt.h)
            bm = 8 * h
            # r2 (app: 10-30 ms a batch): rows whose box misses every part's world box + gap are decided by their
            # corners alone, before any node is built or evaluated
            pb = live.setdefault('pboxes', {}).get(oi)
            if pb is None:
                from ..tools.protected_merge import world_box
                pb = np.array([np.r_[world_box(p_)] for p_ in parts]).reshape(-1, 2, 3)
                live['pboxes'][oi] = pb
            ks_ = np.array([rk[r] for r in new], float)
            cor = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * bm
            Cw = _to_world(obj, (vol.origin[None, None] + ks_[:, None] * bm + cor[None]).reshape(-1, 3)).reshape(-1, 8, 3)
            rlo, rhi = Cw.min(1), Cw.max(1)
            near = np.zeros(len(new), bool)
            for a_, b_ in pb:
                mg_ = gap + CL.BAND_CELLS * h * float(obj.transform['scale'])    # + the outside band
                near |= np.all(rlo <= b_ + mg_, 1) & np.all(rhi >= a_ - mg_, 1)
            for r in [r for r, nr in zip(new, near) if not nr]:
                cache[r] = None
            new = [r for r, nr in zip(new, near) if nr]
        if new:
            ijk = np.stack(np.meshgrid(*[np.arange(9)] * 3, indexing='ij'), -1).reshape(-1, 3) * h
            org = vol.origin[None] + np.array([rk[r] for r in new], float) * bm
            X = (org[:, None, :] + ijk[None]).reshape(-1, 3)
            B = CL.bound(scene, pk, parts, _to_world(obj, X), gap) / max(float(obj.transform['scale']), 1e-9)
            B = B.reshape(len(new), 729).astype(np.float32)
            for i, r in enumerate(new):
                cache[r] = B[i] if np.any(B[i] > -CL.BAND_CELLS * h) else None
        act = [r for r in rk if cache.get(r) is not None]
        if not act:
            return n
        Bf = np.concatenate([cache[r] for r in act])
        bf = gs._ft_rows(live, oi, act).reshape(-1)   # fs-brush-3 11-13: 'before' or the first-touch blocks (the move view keeps no dict)
        th = 256
        gs.k_collide(((len(act) * 729 + th - 1) // th,), (th,),
                       (gs.view.gpu.sdf, cp.asarray(np.array(act, np.int32)), np.int32(len(act)), cp.asarray(bf), cp.asarray(Bf),
                        np.float32(-CL.BAND_CELLS * float(obj.sculpt.h))))
        n += len(act)
    return n


def contact_rows(gs, scene, settings, p, live):
    """sculpt-contact 11-12: Swell to contact on the card: the contacts' distance (local units) at the 729 nodes of each
    of the plan's rows, as gpu/sculpt.cu dab_compute reads them (typ 11, bp[46]): -> (Tb (n x 729 float32 on the card),
    Tslot (len(rows) int32: the row's slot in Tb, -1 = no contact within reach)). Per stroke cache in live['T'][oi]: the
    parts do not move during a stroke"""
    from ..core import collide as CL
    from ..tools.protected_merge import _to_world, field_of, world_box, _rs
    cp = gs.cp
    oi = p['oi']
    obj = scene.objects[oi]
    rows, keys, vol, bm = p['rows'], p['keys'], p['vol'], p['bm']
    h = float(obj.sculpt.h)
    sc_ = _rs(obj)[2]
    parts = CL.colliders(scene, obj.id, settings.get('contact_ids'))
    cache = live.setdefault('T', {}).setdefault(oi, {})
    reach = float(p['bp'][49]) + float(p['bp'][50]) + 2.0 * h
    new = [int(r) for r in rows.tolist() if int(r) not in cache]
    if new and parts:
        kk = {int(r): k for r, k in zip(rows.tolist(), np.asarray(keys).reshape(-1, 3).tolist())}
        ks_ = np.array([kk[r] for r in new], float)
        cor = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * bm
        Cw = _to_world(obj, (vol.origin[None, None] + ks_[:, None] * bm + cor[None]).reshape(-1, 3)).reshape(-1, 8, 3)
        rlo, rhi = Cw.min(1), Cw.max(1)
        near = np.zeros(len(new), bool)
        mg = reach * sc_
        for p_ in parts:
            a_, b_ = world_box(p_)
            near |= np.all(rlo <= b_ + mg, 1) & np.all(rhi >= a_ - mg, 1)
        for r in [r for r, nr in zip(new, near) if not nr]:
            cache[r] = None
        new = [r for r, nr in zip(new, near) if nr]
        if new:
            ijk = np.stack(np.meshgrid(*[np.arange(9)] * 3, indexing='ij'), -1).reshape(-1, 3) * h
            org = vol.origin[None] + np.array([kk[r] for r in new], float) * bm
            W = _to_world(obj, (org[:, None, :] + ijk[None]).reshape(-1, 3))
            pk = scene.pack()
            T = np.full(len(W), np.inf)
            for p_ in CL._parts_near(parts, W.min(0), W.max(0), mg):
                T = np.minimum(T, field_of(scene, pk, p_, W))
            T = (T / sc_).reshape(len(new), 729)
            for i, r in enumerate(new):
                cache[r] = T[i].astype(np.float32) if np.isfinite(T[i]).all() and T[i].min() < reach else None
    elif new:
        for r in new:
            cache[r] = None
    slots = np.full(len(rows), -1, np.int32)
    blocks = []
    for b, r in enumerate(rows.tolist()):
        t = cache.get(int(r))
        if t is not None:
            slots[b] = len(blocks)
            blocks.append(t)
    Tb = cp.asarray(np.concatenate(blocks)) if blocks else cp.zeros(729, cp.float32)
    return Tb, cp.asarray(slots)
