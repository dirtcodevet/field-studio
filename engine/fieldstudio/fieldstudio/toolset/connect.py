"""Connect (TOOLSET_DESIGN 3.3, approved 11-14).

Design: two surfaces, one placed above the other, are joined by a pillar. Its path is drawn freehand (a visible
line shows what was drawn), or a shape is chosen and the place where it starts is picked on each surface. One
button builds the connection between the two objects in that shape; the result is three objects, all connected,
which can then be selected and merged, or each modified as it is, its own properties included.

ConnectObjects(a, b, p_a, p_b, profile_a, profile_b, twist)
- the path runs straight from the picked point on a to the picked point on b, or as an arc (a bulge), or along a
  drawn curve (Catmull-Rom through its points), the profile carried along it by rotation-minimizing frames;
- the cross-section changes smoothly from outline a to outline b (their 2-D distances blended by a smoothstep along the
  path; any two outlines: circle, square, rounded square, or a drawn polygon), turning by `twist` degrees;
- the tube runs on into both objects and is then cut EXACTLY by their own fields: connector = max(tube, -F_a, -F_b), so
  its two ends are the two surfaces themselves (no gap, no overlap: flush over curved or uneven surfaces);
- the connector is its own object (its own material and label), in contact with both: three objects; Merge joins them.
It is stored like any object: an ellipsoid that holds it as its base and its exact shape in its own sculpt lattice (a
field, no mesh: P14). Patent note (LIBRARIES_AND_PATENTS 6.2): interpolating two 2-D outlines along a path is 1990s
practice, clear as far as found.
"""
import copy
import math

import numpy as np
from numba import njit, prange

from ..core import ops
from ..core import brush as BR
from ..core.history import Command, command, release_object_slots
from ..core.scene import SceneObject
from ..core.combine import _mat_to_euler, _measure_all
from ..warp.bake import object_field, W_BAND
from . import common as C

B = C.B


def profile_sdf(p, w):
    """2-D signed distance of a profile at points w (n, 2). p: {'shape': 'circle' | 'square' | 'rounded' | 'polygon',
    'size': m (radius / half side), 'rot': deg, 'points': [[u, v], ...] (polygon)}"""
    shape = p.get('shape', 'circle')
    a = float(p.get('size', 0.01))
    th = math.radians(float(p.get('rot', 0.0)))
    c, s = math.cos(th), math.sin(th)
    q = np.stack([c * w[:, 0] + s * w[:, 1], -s * w[:, 0] + c * w[:, 1]], 1)
    if shape == 'circle':
        return np.linalg.norm(q, axis=1) - a
    if shape in ('square', 'rounded'):
        rr = 0.3 * a if shape == 'rounded' else 0.0
        d = np.abs(q) - (a - rr)
        return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) - rr
    if shape == 'polygon':
        P = np.asarray(p['points'], float)
        n = len(P)
        dmin = np.full(len(q), np.inf)
        inside = np.zeros(len(q), bool)
        for i in range(n):
            a_, b_ = P[i], P[(i + 1) % n]
            e = b_ - a_
            t = np.clip(((q - a_) @ e) / max(e @ e, 1e-30), 0.0, 1.0)
            dmin = np.minimum(dmin, np.linalg.norm(q - (a_ + t[:, None] * e), axis=1))
            cond = ((a_[1] > q[:, 1]) != (b_[1] > q[:, 1]))
            xint = a_[0] + (q[:, 1] - a_[1]) * e[0] / np.where(e[1] == 0, 1e-30, e[1])
            inside ^= cond & (q[:, 0] < xint)
        return np.where(inside, -dmin, dmin)
    raise ValueError(shape)


def _rmax(p):
    if p.get('shape') == 'polygon':
        return float(np.linalg.norm(np.asarray(p['points'], float), axis=1).max())
    return float(p.get('size', 0.01)) * (math.sqrt(2.0) if p.get('shape') in ('square', 'rounded') else 1.0)


def _frame_for(e):
    """an orthonormal frame with its y axis along e: columns (u, e, v)"""
    e = np.asarray(e, float) / np.linalg.norm(e)
    a = np.array([1.0, 0.0, 0.0]) if abs(e[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(e, a)
    u /= np.linalg.norm(u)
    v = np.cross(u, e)
    return np.stack([u, e, v], 1)




@command
class ConnectObjects(Command):
    """A connector between objects a and b, from world point p_a (on a) to p_b (on b)."""
    label = 'Connect'

    def __init__(self, a, b, p_a, p_b, profile_a, profile_b=None, twist=0.0, name=None, h=None, bow=0.0,
                 bow_dir=None, path=None):
        self.a = int(a)
        self.b = int(b)
        self.p_a = [float(v) for v in p_a]
        self.p_b = [float(v) for v in p_b]
        self.profile_a = copy.deepcopy(profile_a)
        self.profile_b = copy.deepcopy(profile_b if profile_b is not None else profile_a)
        self.twist = float(twist)
        self.name = name
        self.h = None if h is None else float(h)
        self.bow = float(bow)                      # Path: arc = a bulge of `bow` m sideways at the middle
        self.bow_dir = None if bow_dir is None else [float(v) for v in bow_dir]
        self.path = None if path is None else [[float(v) for v in q] for q in path]   # Path: a drawn curve (inner points)
        if self.bow or self.path:
            raise ValueError('Connect along a curve is not part of this release (straight connectors only)')
        self.obj = None
        self.info = {}

    def args(self):
        d = {'a': self.a, 'b': self.b, 'p_a': self.p_a, 'p_b': self.p_b, 'profile_a': self.profile_a,
             'profile_b': self.profile_b, 'twist': self.twist}
        if self.name is not None:
            d['name'] = self.name
        if self.h is not None:
            d['h'] = self.h
        if self.bow:
            d['bow'] = self.bow
            d['bow_dir'] = self.bow_dir
        if self.path:
            d['path'] = self.path
        return d

    # ---- the path ----------------------------------------------------------------------------------------------------
    def curved(self):
        return False                    # Connect along a curve is not part of this release

    def field(self, scene, W):
        """the connector's world field at world points W (as built: tube cut by both objects)"""
        A, Bo = scene.get(self.a), scene.get(self.b)
        pa, pb = np.asarray(self.p_a), np.asarray(self.p_b)
        L = float(np.linalg.norm(pb - pa))
        Fr = _frame_for(pb - pa)
        X = (W - pa) @ Fr                                   # (u, y, v): y along the path
        s = np.clip(X[:, 1] / L, 0.0, 1.0)
        m = s * s * (3.0 - 2.0 * s)
        th = np.radians(self.twist) * s
        c, sn = np.cos(th), np.sin(th)
        w = np.stack([c * X[:, 0] + sn * X[:, 2], -sn * X[:, 0] + c * X[:, 2]], 1)
        d2 = (1.0 - m) * profile_sdf(self.profile_a, w) + m * profile_sdf(self.profile_b, w)
        # the tube runs on into both objects by `ext` (then cut by them)
        ext = self._ext
        dy = np.maximum(-ext - X[:, 1], X[:, 1] - (L + ext))
        tube = np.maximum(d2, dy)
        return np.maximum(tube, np.maximum(-object_field(scene, A)(W), -object_field(scene, Bo)(W)))

    def do(self, scene):
        if self.obj is not None:
            scene.objects.insert(min(self.at, len(scene.objects)), self.obj)
            scene.next_id = self.after_next
            scene.touch(structural=True, index=True)
            return self.obj.id
        A, Bo = scene.get(self.a), scene.get(self.b)
        self.prev_next = scene.next_id
        pa, pb = np.asarray(self.p_a), np.asarray(self.p_b)
        L = float(np.linalg.norm(pb - pa))
        if L < 1e-6:
            raise ValueError('Connect needs two different points')
        R = max(_rmax(self.profile_a), _rmax(self.profile_b))
        hw = self.h or min(A.sculpt.h * C.frame(A)[2], Bo.sculpt.h * C.frame(Bo)[2])
        self._ext = max(4.0 * hw, 0.5 * R)                  # into each object (the cut by its field ends it)
        Fr = _frame_for(pb - pa)
        # the object: a cylinder that holds the connector, its exact shape in the lattice (local y = the path)
        band = 2.0 * B * hw
        # the holding shape is an ellipsoid: its field is smooth everywhere but at its centre (deep in the pillar), so the
        # lattice's interpolation of the difference adds no crease error at the surfaces (a cylinder's interior crease,
        # side wall vs end cap, put the ends 0.11 mm off flush: measured 11-14, fs-tools t1/k2)
        tr = {'translate': [float(v) for v in pa], 'rotate': _mat_to_euler(Fr), 'scale': 1.0}
        Ro = ops.rot_matrix(tr['rotate'])
        if self.curved():
            self._poly = self.polyline()
            Pl = (self._poly[0] - pa) @ Ro                # the path in the object's frame
            plo, phi = Pl.min(0) - (R + band), Pl.max(0) + (R + band)
        else:
            plo = np.array([-(R + band), -self._ext - band, -(R + band)])
            phi = np.array([R + band, L + self._ext + band, R + band])
        cen = 0.5 * (plo + phi)
        ell = ops.Node('ellipsoid', {'radii': [float(v) for v in 0.5 * (phi - plo) * 1.05]}, nid=scene.new_id())
        base = ops.Node('transform', {'translate': [float(v) for v in cen], 'rotate': [0.0, 0.0, 0.0],
                                      'scale': [1.0, 1.0, 1.0]}, [ell], nid=scene.new_id(), name='connector')
        M = SceneObject(scene.new_id(), self.name or 'Connector (%s - %s)' % (A.name, Bo.name), base, transform=tr,
                        colour=copy.deepcopy(A.colour), material=copy.deepcopy(A.material), h=hw)
        keys = C.bricks_in_box(plo, phi, hw, pad=1)
        X = C.brick_nodes(keys, hw).reshape(-1, 3)
        W = X @ Ro.T + pa                                   # the object's own rotation (the Euler round trip, not Fr)
        F = self.field(scene, W)
        code, prm, _, _, _ = ops.compile_tree(base)
        Fb = np.empty(len(X))
        C.eval_tree(code, prm, np.ascontiguousarray(X), Fb)
        Fk = F.reshape(len(keys), -1)
        keep = np.any(np.minimum(np.abs(Fk), np.abs(Fb.reshape(len(keys), -1))) < band, 1) | \
            np.any(np.sign(Fk) != np.sign(Fb.reshape(len(keys), -1)), 1)
        st = scene.store
        kk = [tuple(int(v) for v in keys[i]) for i in np.nonzero(keep)[0]]
        if kk:
            sl = st.alloc_many(len(kk))
            vals = (np.clip(F, -W_BAND * hw, W_BAND * hw) - Fb).reshape(len(keys), B, B, B)[keep]   # fs-core's band
            st.delta[sl] = vals.astype(np.float32)
            st.Lb[sl] = 0.0
            if st.has('mask'):
                st.mask[sl] = 0.0
            M.sculpt.set_slots(kk, sl)
        self.obj = M
        self.at = len(scene.objects)
        scene.objects.append(M)
        scene.touch(structural=True, index=True)
        _measure_all(scene, scene.index_of(M.id))
        self.after_next = scene.next_id
        self.info = {'length_mm': round(L * 1e3, 3), 'bricks': len(kk), 'h_mm': round(hw * 1e3, 4)}
        return M.id

    def undo(self, scene):
        scene.objects.remove(self.obj)
        scene.next_id = self.prev_next
        scene.touch(structural=True, index=True)

    def discard(self, scene, state):
        if state == 'undone' and self.obj is not None:
            release_object_slots(scene, self.obj)

    def undo_state(self):
        if self.obj is None:
            return None
        meta = {'id': int(self.obj.id), 'prev_next': int(self.prev_next), 'at': int(self.at),
                'after_next': int(self.after_next)}
        return lambda: (meta, {})

    def restore_undo_state(self, scene, meta, A):
        self.obj = scene.get(meta['id'])
        self.prev_next, self.at, self.after_next = meta['prev_next'], meta['at'], meta['after_next']

    def nbytes(self):
        return 256 + (len(self.obj.sculpt.slots) if self.obj is not None else 0) * 2 * 4 * B ** 3
