"""SPLICE and FILL AND SPLICE (DESIGN_NOTES 10d).

SPLICE: pick a face on A (point pa, normal na) and a face on B (pb, nb). B turns so nb = -na (align) and moves so
pb lands on pa (pushed `overlap` into A), then joins A's tree as a transform node under a smooth union of blend k:
    Splice N (union, blend k)
      |- A's tree
      |- tooldata:{splice} transform (B's frame in A's frame) -> B's tree (still live and editable)
B's sculpt layer is resampled into A's lattice (trilinear, values scaled s_B/s_A) through the brush undo record,
and B is removed from the scene. Undo puts everything back.

FILL AND SPLICE: a pillar lofted from face A to face B; its cross-section is a rounded rectangle (width, depth,
round, angle) that morphs linearly from profile A to profile B (circle, rect, or a drawn polygon fitted by its
principal axes and its fill ratio). The pillar is `slices` rounded-box segments along the axis (overlapping by
their rounding), guarded, and smooth-unioned into A with blend k at both faces. B (if another object) joins A
the same way as in SPLICE but without moving.
"""
import copy
import math

import numpy as np

from ..core import ops
from ..core import brush as BR
from ..core.history import Command, command
from . import common as C


def _resample_sculpt(scene, ia, ib, b_frame_new, rec):
    """Write B's sculpt delta (in B's new placement) into A's lattice. Returns the number of nodes written."""
    A = scene.objects[ia]
    B = scene.objects[ib]
    if not B.sculpt.slots:
        return 0
    rb = B.sculpt.region_box()
    Ra, ta, sa = C.obj_frame(A)
    Rb, tb, sb = b_frame_new
    corners = ops._box_corners(rb[0], rb[1])
    world = (Rb @ (corners * sb).T).T + tb
    qa = (Ra.T @ (world - ta).T).T / sa
    h = A.sculpt.h
    lo = np.floor(qa.min(0) / h).astype(np.int64) - 1
    hi = np.ceil(qa.max(0) / h).astype(np.int64) + 1
    dims = hi - lo + 1
    if int(np.prod(dims)) > 40_000_000:
        raise ValueError('sculpt region too large to carry (%d nodes)' % int(np.prod(dims)))
    df = np.zeros(tuple(int(d) for d in dims))
    # slab by slab along x keeps memory small
    jj = (np.arange(dims[1]) + lo[1]) * h
    kk = (np.arange(dims[2]) + lo[2]) * h
    J, K = np.meshgrid(jj, kk, indexing='ij')
    for a in range(int(dims[0])):
        x = (lo[0] + a) * h
        pa = np.column_stack([np.full(J.size, x), J.ravel(), K.ravel()])
        w = (Ra @ (pa * sa).T).T + ta
        qb = (Rb.T @ (w - tb).T).T / sb
        df[a] = (C.sample_sculpt(scene, ib, qb) * (sb / sa)).reshape(J.shape)
    df[np.abs(df) < 1e-12] = 0.0
    return C.write_nodes(scene, ia, lo, df, rec)


class _JoinBase(C.BaseEditMixin, Command):
    """Shared do/undo: replace A's base, remove B, carry B's sculpt into A."""

    def _join(self, scene, A, B, b_frame, extra_node, blend, name, ids, data):
        ia = scene.index_of(A.id)
        tree = A.base
        if B is not None:
            ib = scene.index_of(B.id)
            rel = C.relative_transform(C.obj_frame(A), b_frame)
            bnode = ops.Node('transform', rel, [C.renumber(B.base, ids)], nid=ids())
            C.set_data(bnode, dict(data, b_name=B.name, b_id=B.id))
            tree = ops.Node('union', {'blend': blend}, [tree, bnode], nid=ids(), name=name)
            self.rec = BR.StrokeRecord()
            self.written = _resample_sculpt(scene, ia, ib, b_frame, self.rec)
        if extra_node is not None:
            tree = ops.Node('union', {'blend': blend}, [tree, extra_node], nid=ids(), name=name)
        self._commit(scene, A, tree, ids)
        if B is not None:
            self._b_pos = scene.index_of(B.id)
            self._b_obj = scene.objects.pop(self._b_pos)
            scene.touch(structural=True, index=True)

    def _unjoin(self, scene):
        if getattr(self, '_b_obj', None) is not None:
            scene.objects.insert(self._b_pos, self._b_obj)
            scene.touch(structural=True, index=True)
            self._b_obj = None
        if getattr(self, 'rec', None) is not None:
            BR.undo_record(scene, self.rec)
            self.rec = None
        self._restore(scene, self.oid_a)

    def nbytes(self):
        r = getattr(self, 'rec', None)
        return 512 + (r.nbytes() if r is not None else 0)


@command
class Splice(_JoinBase):
    """Move B onto A face to face and join with a smooth union (blend k, metres). World-space picks."""
    label = 'Splice'

    def __init__(self, oid_a, pa, na, oid_b, pb, nb, blend=0.004, align=True, overlap=0.0):
        self.oid_a, self.oid_b = int(oid_a), int(oid_b)
        self.pa, self.na = [float(v) for v in pa], [float(v) for v in na]
        self.pb, self.nb = [float(v) for v in pb], [float(v) for v in nb]
        self.blend, self.align, self.overlap = float(blend), bool(align), float(overlap)

    def args(self):
        return {'oid_a': self.oid_a, 'pa': self.pa, 'na': self.na, 'oid_b': self.oid_b, 'pb': self.pb,
                'nb': self.nb, 'blend': self.blend, 'align': self.align, 'overlap': self.overlap}

    def do(self, scene):
        if self.oid_a == self.oid_b:
            raise ValueError('splice joins two different objects')
        A = self._begin(scene, self.oid_a)
        B = scene.get(self.oid_b)
        Rb, tb, sb = C.obj_frame(B)
        pa, pb = np.asarray(self.pa), np.asarray(self.pb)
        na = C.unit(self.na)
        Ral = C.rotation_between(self.nb, -na) if self.align else np.eye(3)
        Rb2 = Ral @ Rb
        tb2 = Ral @ (tb - pb) + pa - na * self.overlap
        self.moved = {'R': Rb2.tolist(), 't': tb2.tolist()}
        ids = C.Ids(scene.next_id)
        k = sum(1 for n in A.base.walk() if n.name.startswith('Splice')) + 1
        self._join(scene, A, B, (Rb2, tb2, sb), None, self.blend, 'Splice %d' % k, ids,
                   {'tool': 'splice', 'pa': C.r7(pa), 'na': C.r7(na), 'pb': C.r7(pb)})
        return A.id

    def undo(self, scene):
        self._unjoin(scene)


def profile_to_rr(p, frame2=None):
    """Profile -> (width, depth, round, angle_deg). p: {'shape': 'circle', 'radius'} | {'shape': 'rect',
    'width', 'depth', 'round', 'angle'} | {'shape': 'polygon', 'points': [[x, y], ...]} (cross-section plane
    coords, metres: x along the pillar frame's x, y along its z)."""
    s = p.get('shape', 'circle')
    if s == 'circle':
        r = float(p['radius'])
        return 2 * r, 2 * r, r, 0.0
    if s == 'rect':
        w, d = float(p['width']), float(p['depth'])
        return w, d, min(float(p.get('round', 0.0)), 0.5 * min(w, d)), float(p.get('angle', 0.0))
    if s == 'polygon':
        q = np.asarray(p['points'], float).reshape(-1, 2)
        c = q.mean(0)
        x = q - c
        ev, evec = np.linalg.eigh(x.T @ x)
        ax = evec[:, 1]                                    # major axis
        ang = math.degrees(math.atan2(ax[1], ax[0]))
        u = x @ ax
        v = x @ np.array([-ax[1], ax[0]])
        w = float(u.max() - u.min())
        d = float(v.max() - v.min())
        xs, ys = q[:, 0], q[:, 1]
        area = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)))
        ratio = area / max(w * d, 1e-18)                   # 1 = rectangle, pi/4 = ellipse
        fr = min(max((1.0 - ratio) / (1.0 - math.pi / 4.0), 0.0), 1.0)
        return w, d, fr * 0.5 * min(w, d), -ang            # angle about the pillar's y (x toward z negative)
    raise ValueError('profile shape must be circle, rect or polygon')


def build_pillar(qa, qb, prof_a, prof_b, slices, ids, join_k=0.0):
    """Pillar from local qa to qb, rounded-rectangle sections morphing a -> b. Returns (node, frame, info)."""
    axis = np.asarray(qb, float) - np.asarray(qa, float)
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        raise ValueError('faces are too close for a pillar')
    ey = axis / L
    k = int(np.argmin(np.abs(ey)))
    e = np.zeros(3)
    e[k] = 1.0
    F = C.frame_from(ey, e)                                # x, y = axis, z
    a = np.array(profile_to_rr(prof_a))
    b = np.array(profile_to_rr(prof_b))
    segs = []
    n = max(int(slices), 1)
    kseg = 0.1 * max(min(a[0], a[1], b[0], b[1]), 1e-5)
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        tm = 0.5 * (t0 + t1)
        w, d, rr, ang = a + (b - a) * tm
        rr = min(rr, 0.5 * min(w, d) * 0.999)
        seg = L / n
        ext = rr + 0.05 * seg
        if i == 0 or i == n - 1:
            ext += 0.0
        box = ops.Node('box', {'size': [w, seg + 2 * ext, d], 'round': rr}, nid=ids())
        th = math.radians(ang)
        Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
        mid = np.asarray(qa) + axis * tm
        segs.append(C.framed(box, mid, F @ Ry, [1.0, 1.0, 1.0], ids))
    body = C.chunked_union(segs, kseg, ids, join_k=join_k)     # inner guards padded for the join (pencil-locality fix)
    return body, F, {'length': L, 'segments': n, 'a': a.tolist(), 'b': b.tolist()}


@command
class FillSplice(_JoinBase):
    """A pillar lofted from face A (pa, na) to face B (pb, nb), profile_a -> profile_b, smooth-joined (blend).
    oid_b may equal oid_a (two faces of one object). inset: how far the pillar ends sink into each face
    (default: a quarter of the larger profile size)."""
    label = 'Fill and splice'

    def __init__(self, oid_a, pa, na, oid_b, pb, nb, profile_a, profile_b=None, blend=0.003, slices=12,
                 inset=None):
        self.oid_a, self.oid_b = int(oid_a), int(oid_b)
        self.pa, self.na = [float(v) for v in pa], [float(v) for v in na]
        self.pb, self.nb = [float(v) for v in pb], [float(v) for v in nb]
        self.profile_a = copy.deepcopy(profile_a)
        self.profile_b = copy.deepcopy(profile_b if profile_b is not None else profile_a)
        self.blend, self.slices = float(blend), int(slices)
        self.inset = None if inset is None else float(inset)

    def args(self):
        return {'oid_a': self.oid_a, 'pa': self.pa, 'na': self.na, 'oid_b': self.oid_b, 'pb': self.pb,
                'nb': self.nb, 'profile_a': self.profile_a, 'profile_b': self.profile_b, 'blend': self.blend,
                'slices': self.slices, 'inset': self.inset}

    def do(self, scene):
        A = self._begin(scene, self.oid_a)
        B = None if self.oid_b == self.oid_a else scene.get(self.oid_b)
        sa = C.obj_frame(A)[2]
        qa = C.to_local(A, self.pa)
        qb = C.to_local(A, self.pb)
        na = C.unit(C.dir_to_local(A, self.na))
        nb = C.unit(C.dir_to_local(A, self.nb))
        wa = profile_to_rr(self.profile_a)
        wb = profile_to_rr(self.profile_b)
        inset = self.inset if self.inset is not None else 0.25 * max(wa[0], wa[1], wb[0], wb[1]) * sa
        inset_l = inset / sa
        ids = C.Ids(scene.next_id)
        # profiles are given in world metres: convert to A's local units
        pa_l = _scale_profile(self.profile_a, 1.0 / sa)
        pb_l = _scale_profile(self.profile_b, 1.0 / sa)
        body, F, info = build_pillar(qa - na * inset_l, qb - nb * inset_l, pa_l, pb_l, self.slices, ids,
                                     join_k=self.blend / sa)
        pillar = C.guarded(body, self.blend / sa, ids)
        C.set_data(pillar, {'tool': 'fill_splice', 'pa': C.r7(qa), 'pb': C.r7(qb), 'info': info})
        k = sum(1 for n in A.base.walk() if n.name.startswith('Fill')) + 1
        b_frame = C.obj_frame(B) if B is not None else None
        self.info = info
        self._join(scene, A, B, b_frame, pillar, self.blend / sa, 'Fill and splice %d' % k, ids,
                   {'tool': 'fill_splice_b'})
        return A.id

    def undo(self, scene):
        self._unjoin(scene)


def _scale_profile(p, f):
    q = copy.deepcopy(p)
    for key in ('radius', 'width', 'depth', 'round'):
        if key in q:
            q[key] = float(q[key]) * f
    if 'points' in q:
        q['points'] = [[float(x) * f, float(y) * f] for x, y in q['points']]
    return q
