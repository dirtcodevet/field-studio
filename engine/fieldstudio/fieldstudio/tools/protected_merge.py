"""Merge with protected parts (milestone, 11-08 night; block-in: shapes merged into her body, never fused to the
eyeball or to other protected parts).

MergeProtected(target, shapes, protected=(), use_mask=False, k=0.003, gap=0.0003)
    The shapes (unsculpted objects: spheres, rounded boxes, capsules, ellipsoids, scaled on any axis) are merged into
    the target's field, baked into the target's SCULPT layer on its own lattice (so her base, her store, is never
    copied or recompiled, and the result undoes, replays and saves like a stroke):
        S   = min over the shapes (their exact fields, in the target's frame)
        P_d = min over the protected objects' exact fields, and (use_mask) the distance to the target's MASKED
              surface nodes (painted with the Mask brush), each minus gap
        S'  = max(S, -P_d)                        (the shape minus the protected parts dilated)
        F'  = max(smin_k(F, S'), min(F, -P_d))    (program._smin's smooth union; the max: inside the dilation no
                                                   new inside, her own field unchanged; r2 after test PM2's first run)
        delta += F' - F at every target node of the shapes' footprint (+ k + 2 h)
    Guarantees (D): wherever the blend is inactive (|F - S'| >= k with S' >= F) F' == F bit for bit, so the stored layer
    outside the footprint is unchanged; a new surface point where F >= 0 has S' < k / 4, so it lies more than `gap` from
    every protected part (P_d >= 0 there, P_d = P - gap); where a shape covers a protected part, the new surface hugs it
    between gap and gap + k / 4 (the blend's dip just outside the dilation). The protected objects stay separate objects, untouched. The shapes leave the scene (undo restores).
"""
import copy

import numpy as np
from numba import njit, prange

from ..core import program as P
from ..core import brush as BR
from ..core.history import Command, command


@njit(cache=True, parallel=True)
def _eval_exact(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts, out):
    for n in prange(pts.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        out[n] = P.eval_local_t(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts[n, 0], pts[n, 1], pts[n, 2],
                                vs, ps, 1e9)


@njit(cache=True)
def _smin_all(F, S, k, out):
    for i in range(F.shape[0]):
        a = F[i]
        b = S[i]
        h = max(k - abs(a - b), 0.0) / k
        out[i] = min(a, b) - h * h * k * 0.25


def _rs(o):
    from ..core import ops
    t = o.transform
    return ops.rot_matrix(t['rotate']), np.asarray(t['translate'], float), max(float(t['scale']), 1e-9)


def _to_world(o, X):
    R, t, s = _rs(o)
    return (X * s) @ R.T + t


def _to_local(o, W):
    R, t, s = _rs(o)
    return ((W - t) @ R) / s


def field_of(scene, pk, o, W):
    """object o's exact field (world units) at world points W"""
    X = np.ascontiguousarray(_to_local(o, W))
    out = np.empty(len(X))
    _eval_exact(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, scene.index_of(o.id), X, out)
    return out * _rs(o)[2]


def min_scale(o):
    """the smallest scale factor along any path of the object's tree (transform nodes' per-axis scales, multiplied
    down the tree) times the object's own: a shape's field is a distance bound scaled by it, so a smooth union's blend
    reaches k / min_scale from the shape (r3: the locality gate's scale_merge run found a 173 um step where the footprint
    stopped at k)"""
    def walk(n, acc):
        if n.kind == 'transform':
            acc = acc * min(abs(float(v)) for v in n.params.get('scale', [1.0, 1.0, 1.0]))
        if not n.children:
            return acc
        return min(walk(c, acc) for c in n.children)
    return max(walk(o.base, 1.0), 1e-3)


def world_box(o):
    lo, hi = o.local_box()
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = _to_world(o, c)
    return w.min(0), w.max(0)


@command
class MergeProtected(Command):
    label = 'Merge with protected parts'

    def __init__(self, target, shapes, protected=(), use_mask=False, k=0.003, gap=0.0003):
        self.target = int(target)
        self.shapes = [int(s) for s in shapes]
        self.protected = [int(p) for p in protected]
        self.use_mask = bool(use_mask)
        self.k = float(k)
        self.gap = float(gap)
        self.rec = None
        self.removed = None
        self.info = {}

    def args(self):
        return {'target': self.target, 'shapes': self.shapes, 'protected': self.protected, 'use_mask': self.use_mask,
                'k': self.k, 'gap': self.gap}

    def nbytes(self):
        return 256 + (self.rec.nbytes() if self.rec is not None else 0)

    def do(self, scene):
        if self.rec is not None and self.removed is not None:     # redo: re-apply the recorded result
            self._apply_values(scene, self._redo_vals)
            self._remove_shapes(scene)
            return
        tgt = scene.get(self.target)
        if any(scene.get(s).sculpt.slots for s in self.shapes):
            raise ValueError('merge with protected parts takes unsculpted shapes (sculpt the result after merging)')
        if self.target in self.shapes or self.target in self.protected or set(self.shapes) & set(self.protected):
            raise ValueError('target, shapes and protected parts must be different objects')
        h = float(tgt.sculpt.h)
        pk = scene.pack()
        _, _, st_ = _rs(tgt)
        # the footprint: the shapes' world boxes + k + 2 h, in the target's frame, on its lattice
        los, his = zip(*[world_box(scene.get(s)) for s in self.shapes])
        wlo, whi = np.min(los, 0), np.max(his, 0)
        m = self.k / min(min_scale(scene.get(s_)) for s_ in self.shapes) + 2 * h * st_   # r3: the blend's reach
        c = np.array([[x, y, z] for x in (wlo[0] - m, whi[0] + m) for y in (wlo[1] - m, whi[1] + m)
                      for z in (wlo[2] - m, whi[2] + m)])
        lc = _to_local(tgt, c)
        i0 = np.floor(lc.min(0) / h).astype(np.int64)
        i1 = np.ceil(lc.max(0) / h).astype(np.int64)
        i0 = (i0 // 8) * 8                                   # whole sculpt bricks
        i1 = (i1 // 8) * 8 + 7
        ax = [np.arange(i0[a], i1[a] + 1) * h for a in range(3)]
        Xl = np.ascontiguousarray(np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3))
        W = _to_world(tgt, Xl)
        F = field_of(scene, pk, tgt, W)
        S = np.min([field_of(scene, pk, scene.get(s), W) for s in self.shapes], 0)
        g = self.gap                                         # r2: the clamp below keeps the gap; no k/4 margin needed
        Pd = np.full(len(W), np.inf)
        for p in self.protected:
            Pd = np.minimum(Pd, field_of(scene, pk, scene.get(p), W) - g)
        if self.use_mask:
            Pd = np.minimum(Pd, self._mask_distance(scene, tgt, i0, i1, h) * st_ - g)
        Sp = np.maximum(S, -Pd) if np.isfinite(Pd).any() else S
        Fn = np.empty(len(F))
        _smin_all(F, Sp, self.k, Fn)
        if np.isfinite(Pd).any():
            # r2 (test PM2 first run: the blend's dip, up to k/4, made surface inside the eye where F was just above 0):
            # inside the protected dilation nothing new becomes inside, and her own field stays as it was
            Fn = np.maximum(Fn, np.minimum(F, -Pd))
        dF = ((Fn - F) / st_).reshape(tuple(int(v) for v in (i1 - i0 + 1)))
        self.rec = BR.StrokeRecord()
        vals = self._write(scene, tgt, i0, dF)
        self._redo_vals = vals
        self._remove_shapes(scene)
        self.info = {'nodes': int(dF.size), 'changed': int((dF != 0).sum()), 'bricks': len(vals)}
        return self.info

    def _mask_distance(self, scene, tgt, i0, i1, h):
        """distance (target local units) from each footprint node to the nearest masked surface node of the target"""
        from scipy.ndimage import distance_transform_edt
        g = tgt.sculpt
        st = scene.store
        dims = tuple(int(v) for v in (i1 - i0 + 1))
        M = np.zeros(dims, bool)
        pk = scene.pack()
        mg = 6 * 8                                            # masked nodes found within the footprint + 6 bricks
        for b, s in g.slots.items():
            if s < 0 or not st.mask[s].any():
                continue
            o = np.asarray(b) * 8 - i0
            if np.any(o + 8 <= -mg) or np.any(o >= np.array(dims) + mg):
                continue
            mk = st.mask[s] > 0.5
            if not mk.any():
                continue
            idx = np.argwhere(mk)
            X = (np.asarray(b) * 8 + idx) * h
            f = np.empty(len(X))
            _eval_exact(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, scene.index_of(tgt.id),
                        np.ascontiguousarray(X), f)
            surf = idx[np.abs(f) <= 1.5 * h] + o
            ok = np.all((surf >= 0) & (surf < np.array(dims)), 1)
            surf = surf[ok]
            M[surf[:, 0], surf[:, 1], surf[:, 2]] = True
        if not M.any():
            return np.full(int(np.prod(dims)), np.inf)
        return distance_transform_edt(~M).reshape(-1) * h

    def _write(self, scene, tgt, i0, dF):
        """delta += dF over the bricks where dF is non-zero, through self.rec; -> {brick: delta after}"""
        st = scene.store
        g = tgt.sculpt
        nb = np.array(dF.shape) // 8
        D = dF.reshape(nb[0], 8, nb[1], 8, nb[2], 8).transpose(0, 2, 4, 1, 3, 5)
        nz = np.argwhere(np.any(D != 0, axis=(3, 4, 5)))
        vals = {}
        if not len(nz):
            return vals
        keys = nz + i0 // 8
        g.ensure_index(keys.min(0), keys.max(0))
        for (a, b, c), key in zip(nz.tolist(), map(tuple, keys.tolist())):
            s = g.get_slot(key)
            k2 = (tgt.id, key)
            if s < 0:
                s = st.alloc()
                g.set_slot(key, s)
                self.rec.new.append(k2)
                self.rec.new_set.add(k2)
            elif k2 not in self.rec.snap:
                self.rec.snap[k2] = (st.delta[s].copy(), st.mask[s].copy(), float(st.Lb[s]))
            st.delta[s] = (st.delta[s] + D[a, b, c]).astype(np.float32)
            vals[key] = st.delta[s].copy()
        scene.touch(index=True)
        BR.measure_bricks(scene, scene.index_of(tgt.id), set(vals.keys()))
        return vals

    def _apply_values(self, scene, vals):
        tgt = scene.get(self.target)
        g = tgt.sculpt
        st = scene.store
        self.rec = BR.StrokeRecord()
        if vals:
            ks = np.array(list(vals.keys()), np.int64)
            g.ensure_index(ks.min(0), ks.max(0))
        for key, v in vals.items():
            s = g.get_slot(key)
            k2 = (tgt.id, key)
            if s < 0:
                s = st.alloc()
                g.set_slot(key, s)
                self.rec.new.append(k2)
                self.rec.new_set.add(k2)
            else:
                self.rec.snap[k2] = (st.delta[s].copy(), st.mask[s].copy(), float(st.Lb[s]))
            st.delta[s] = v
        scene.touch(index=True)
        BR.measure_bricks(scene, scene.index_of(tgt.id), set(vals.keys()))

    def _remove_shapes(self, scene):
        self.removed = []
        for s in self.shapes:
            i = scene.index_of(s)
            self.removed.append((i, scene.objects[i]))
        for i, o in sorted(self.removed, key=lambda t: -t[0]):
            scene.objects.pop(i)
        scene.touch(structural=True, index=True)

    def undo(self, scene):
        for i, o in sorted(self.removed, key=lambda t: t[0]):
            scene.objects.insert(i, o)
        scene.touch(structural=True, index=True)
        BR.undo_record(scene, self.rec)
