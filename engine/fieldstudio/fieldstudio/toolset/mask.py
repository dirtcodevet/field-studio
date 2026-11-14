"""Mask / Protect by a property of the form (TOOLSET_DESIGN 3.2, approved 11-14): "Mask by: thickness (everything thinner
than x mm), cavity, distance from a part, or a region"; "Grow / Shrink / Soften / Invert / Clear".

MaskBy(oid, by, value, op)
- by 'thickness': every surface point where the form is thinner than `value` (m) along its own facing: from the point
  inward until the far face (a march through the field: exact, "no mesh tool has 'everything thinner than 2 mm'");
- by 'distance': every surface point within `value` of another object (a part, a contact partner);
- op 'set' (replace), 'add' (max with what is there), 'remove' (clear where it holds).
MaskOp(oid, op, amount): 'grow' / 'shrink' by `amount` (m) along the form, 'soften' (blur over `amount`), 'invert', 'clear'.
The mask lives on the sculpt lattice's band around the surface (bricks are added to hold it; the field is unchanged);
undo is exact (a StrokeRecord with the mask's prior values).
"""
import numpy as np
from scipy import ndimage

from ..core import brush as BR
from ..core.history import Command, command
from ..warp.bake import RecUndo
from ..warp.bake import object_field
from . import common as C

B = C.B


def _band_bricks(scene, o, band):
    """the object's bricks within `band` (local) of its surface: (keys, local nodes, F at nodes)"""
    lo, hi = o.local_box()
    h = o.sculpt.h
    keys = C.bricks_in_box(lo - band, hi + band, h, pad=0)
    Xc = (keys + 0.5) * B * h
    Fc = C.field_local(scene, o, Xc)
    keys = keys[np.abs(Fc) < band + 0.87 * B * h]
    X = C.brick_nodes(keys, h).reshape(-1, 3)
    F = C.field_local(scene, o, X)
    return keys, X, F


def thickness_at(scene, o, X, F, hmin):
    """the form's thickness through each local point X (near its surface): inward along -grad F to the far face"""
    s = C.frame(o)[2]
    e = 0.25 * o.sculpt.h
    g = np.stack([(C.field_local(scene, o, X + e * np.eye(3)[a]) - C.field_local(scene, o, X - e * np.eye(3)[a])) / (2 * e)
                  for a in range(3)], 1)
    n = g / np.maximum(np.linalg.norm(g, axis=1), 1e-12)[:, None]
    P0 = X - F[:, None] * n                            # onto the surface
    t = np.full(len(X), hmin)
    out = np.full(len(X), np.inf)
    act = np.ones(len(X), bool)
    lo, hi = o.local_box()
    tmax = float(np.linalg.norm(hi - lo))
    for _ in range(400):
        idx = np.nonzero(act)[0]
        if not len(idx):
            break
        f = C.field_local(scene, o, P0[idx] - t[idx, None] * n[idx])
        out_ = f > 0
        out[idx[out_]] = t[idx[out_]]
        act[idx[out_]] = False
        t[idx] += np.maximum(np.abs(f), hmin)
        act &= t < tmax
    return out * s


@command
class MaskBy(RecUndo, Command):
    label = 'Mask by'

    def __init__(self, oid, by='thickness', value=0.002, op='set', other=None):
        self.oid = int(oid)
        self.by = str(by)
        self.value = float(value)
        self.op = str(op)
        self.other = None if other is None else [int(v) for v in other]
        self.rec = None
        self.info = {}

    def args(self):
        d = {'oid': self.oid, 'by': self.by, 'value': self.value, 'op': self.op}
        if self.other is not None:
            d['other'] = self.other
        return d

    def do(self, scene):
        o = scene.get(self.oid)
        h = o.sculpt.h
        s = C.frame(o)[2]
        band = 2.0 * h
        keys, X, F = _band_bricks(scene, o, band)
        near = np.abs(F) < band
        sel = np.zeros(len(X), bool)
        if self.by == 'thickness':
            th = np.full(len(X), np.inf)
            th[near] = thickness_at(scene, o, X[near], F[near], 0.25 * h)
            sel = near & (th < self.value)
            self.info['thin_nodes'] = int(sel.sum())
        elif self.by == 'distance':
            others = [scene.get(i) for i in (self.other or [x.id for x in scene.objects if x.id != self.oid])]
            W = C.to_world(o, X[near])
            d = np.full(near.sum(), np.inf)
            for x in others:
                d = np.minimum(d, object_field(scene, x)(W))
            sel[np.nonzero(near)[0]] = d < self.value
        else:
            raise ValueError(self.by)
        st = scene.store
        st.ensure('mask')
        g = o.sculpt
        cur = np.stack([st.mask[g.get_slot(tuple(k))].astype(np.float64) if g.get_slot(tuple(k)) >= 0 else
                        np.zeros((B, B, B)) for k in keys])
        selb = sel.reshape(cur.shape)
        if self.op == 'set':
            new = np.where(near.reshape(cur.shape), selb.astype(float), cur)
        elif self.op == 'add':
            new = np.maximum(cur, selb.astype(float))
        else:
            new = np.where(selb, 0.0, cur)
        ch = np.any(new != cur, axis=(1, 2, 3))
        self.rec = BR.StrokeRecord()
        _write_mask(scene, o, keys[ch], new[ch], self.rec)
        self.info['bricks'] = int(ch.sum())
        return self.info

    def undo(self, scene):
        if self.rec is not None:
            BR.undo_record(scene, self.rec)
            self.rec = None


def _write_mask(scene, o, keys, vals, rec):
    """write mask values into o's bricks (adding bricks with delta 0 where needed), recording undo"""
    st = scene.store
    g = o.sculpt
    keys = [tuple(int(v) for v in k) for k in keys]
    newk = [k for k in keys if g.get_slot(k) < 0]
    if newk:
        ns = st.alloc_many(len(newk))
        st.delta[ns] = 0.0
        st.Lb[ns] = 0.0
        st.mask[ns] = 0.0
        g.set_slots(newk, ns)
        for k in newk:
            rec.new.append((o.id, k))
            rec.new_set.add((o.id, k))
    for k, v in zip(keys, vals):
        sl = g.get_slot(k)
        if (o.id, k) not in rec.snap and (o.id, k) not in rec.new_set:
            rec.snap[(o.id, k)] = (st.delta[sl].copy(), st.mask[sl].copy(), float(st.Lb[sl]))
        st.mask[sl] = np.asarray(v, np.float32)
    scene.touch(index=bool(newk))
    scene.refresh_object(scene.index_of(o.id))


@command
class MaskOp(RecUndo, Command):
    """'grow' / 'shrink' (by amount, m, along the form), 'soften' (a blur over amount), 'invert', 'clear'"""
    label = 'Mask'

    def __init__(self, oid, op, amount=0.001):
        self.oid = int(oid)
        self.op = str(op)
        self.amount = float(amount)
        self.rec = None

    def args(self):
        return {'oid': self.oid, 'op': self.op, 'amount': self.amount}

    def do(self, scene):
        o = scene.get(self.oid)
        h = o.sculpt.h
        s = C.frame(o)[2]
        st = scene.store
        st.ensure('mask')
        band = 2.0 * h
        keys, X, F = _band_bricks(scene, o, band)
        g = o.sculpt
        cur = np.stack([st.mask[g.get_slot(tuple(k))].astype(np.float64) if g.get_slot(tuple(k)) >= 0 else
                        np.zeros((B, B, B)) for k in keys])
        # the band's nodes on one dense grid (the operations run over it), then back to the bricks
        kmin = keys.min(0)
        shape = tuple(int(v) for v in (keys.max(0) - kmin + 1) * B)
        M = np.zeros(shape)
        inb = np.zeros(shape, bool)
        Fd = np.full(shape, 1e9)
        for q, k in enumerate(keys):
            a = (k - kmin) * B
            sl = (slice(a[0], a[0] + B), slice(a[1], a[1] + B), slice(a[2], a[2] + B))
            M[sl] = cur[q]
            Fd[sl] = F.reshape(len(keys), B, B, B)[q]
        inb = np.abs(Fd) < band
        r = max(1, int(round(self.amount / s / h)))
        if self.op in ('grow', 'shrink'):
            # along the form: dilate / erode within the band (the band is a thin shell: the 3-D step stays on the form)
            m = M > 0.5
            st_ = ndimage.generate_binary_structure(3, 1)
            for _ in range(r):
                m = (ndimage.binary_dilation(m, st_) if self.op == 'grow' else ndimage.binary_erosion(m, st_)) & inb
            N = np.where(inb, m.astype(float), M)
        elif self.op == 'soften':
            N = np.where(inb, ndimage.gaussian_filter(M * inb, r / 2.0) /
                         np.maximum(ndimage.gaussian_filter(inb.astype(float), r / 2.0), 1e-6), M)
        elif self.op == 'invert':
            N = np.where(inb, 1.0 - M, M)
        elif self.op == 'clear':
            N = np.zeros_like(M)
        else:
            raise ValueError(self.op)
        new = np.stack([N[(k - kmin)[0] * B:(k - kmin)[0] * B + B, (k - kmin)[1] * B:(k - kmin)[1] * B + B,
                          (k - kmin)[2] * B:(k - kmin)[2] * B + B] for k in keys])
        ch = np.any(np.abs(new - cur) > 1e-7, axis=(1, 2, 3))
        self.rec = BR.StrokeRecord()
        _write_mask(scene, o, keys[ch], new[ch], self.rec)
        return {'bricks': int(ch.sum())}

    def undo(self, scene):
        if self.rec is not None:
            BR.undo_record(scene, self.rec)
            self.rec = None
