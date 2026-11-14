"""Place on surface (TOOLSET_DESIGN 3.2, approved 11-14): contact placement now; the physics settle is a hook for later.

Design: the object is placed on a surface, and it is to interact with physics too.
- The object follows the cursor across OTHER surfaces, sitting on them, turned to their facing (Align: surface /
  upright / keep), spun by the wheel; a click drops it. It never sinks in unless Offset says so (+ sinks, - floats).
- Its contact face comes from its own field: its support along the surface's facing (the lowest points), read once at
  the press from a sample of its inside, then refined by the exact contact read from both fields on the drop
  (toolset/contact.touching: min over space of max(F_object, F_surface), half the gap).
- The drop is one SetObjectProps (its final frame), so a replay needs no contact solve.
- Settle (gravity until rest) needs rigid contact, which the contact lane will give: settle_hook is where it goes.
"""
import math

import numpy as np

from ..core import ops
from ..core.history import SetObjectProps
from ..core.combine import _mat_to_euler
from ..warp.bake import object_field
from ..warp.lens import march
from . import common as C
from .contact import touching

settle_hook = None          # fn(scene, oid) -> transform: the physics settle (the contact lane), when it exists


def _rot_between(a, b):
    """the rotation taking unit a to unit b (shortest arc)"""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-12:
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 0, 1.0])
        v = np.cross(a, p)
        v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _axis_angle(n, ang):
    n = np.asarray(n, float) / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


class Placer:
    """one placement drag of object oid: press (read its inside once), move (cursor ray -> frame), drop (exact)"""

    def __init__(self, scene, oid, up_local=(0.0, 1.0, 0.0), samples=24):
        self.scene = scene
        self.oid = int(oid)
        o = scene.get(oid)
        self.t0 = dict(o.transform)
        self.R0 = ops.rot_matrix(o.transform['rotate'])
        self.s = float(o.transform['scale'])
        lo, hi = o.local_box()
        ax = [np.linspace(lo[a], hi[a], samples) for a in range(3)]
        X = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
        F = C.field_local(scene, o, X)
        cell = float((hi - lo).max() / (samples - 1))
        self.inside = X[F < 0.5 * cell]                  # its inside and skin (local): the support is read from these
        if not len(self.inside):
            self.inside = X[np.argsort(F)[:64]]
        self.up_local = np.asarray(up_local, float)
        others = [x for x in scene.objects if x.id != self.oid and scene.shown(x)]
        pk = scene.pack()
        self.fields = [object_field(scene, x, pk) for x in others]
        self.others = others

    def surface_field(self, P):
        v = np.full(len(P), 1e9)
        for f in self.fields:
            v = np.minimum(v, f(P))
        return v

    def hit(self, ray):
        """the first OTHER surface under the cursor ray (origin, direction): point and its facing, or None"""
        o, d = np.asarray(ray[0], float), np.asarray(ray[1], float)
        d = d / np.linalg.norm(d)
        if not self.fields:
            return None
        r = march(self.surface_field, o[None], d[None], 20.0, L=2.0, hmin=1e-5)
        t = r[0, 0]
        if not np.isfinite(t):
            return None
        p = o + t * d
        e = 1e-5
        g = np.array([(self.surface_field((p + e * np.eye(3)[a])[None])[0] -
                       self.surface_field((p - e * np.eye(3)[a])[None])[0]) / (2 * e) for a in range(3)])
        return p, g / max(np.linalg.norm(g), 1e-12)

    def frame(self, p, n, align='surface', offset=0.0, spin=0.0):
        """the object's frame (rotation matrix, translation) sitting on the surface at p with facing n"""
        if align == 'surface':
            R = _axis_angle(n, spin) @ _rot_between(self.R0 @ self.up_local, n) @ self.R0
        elif align == 'upright':
            R = _axis_angle([0, 1, 0], spin) @ self.R0
        else:
            R = self.R0
        W = (self.inside * self.s) @ R.T                 # the object's inside, turned, relative to its origin
        sup = float((W @ n).min())                       # its lowest point along the facing
        t = p - sup * n - offset * n
        return R, t

    def transform(self, R, t):
        return {'translate': [float(v) for v in t], 'rotate': _mat_to_euler(R), 'scale': self.s}

    def drop(self, R, t, n, offset=0.0, tol=2e-5, iters=6):
        """the exact contact: move along n until the gap (both fields) is the offset; -> its transform"""
        o = self.scene.get(self.oid)
        keep = dict(o.transform)
        t = np.asarray(t, float).copy()
        try:
            for _ in range(iters):
                o.transform = self.transform(R, t)
                self.scene.touch(structural=True)
                m = min((touching(self.scene, o, x)[0] for x in self.others), default=float('inf'))
                if not np.isfinite(m):
                    break
                gap = 2.0 * m                            # touching gives half the gap (< 0: half the overlap depth)
                move = gap + offset
                if abs(move) < tol:
                    break
                t = t - move * n
            return self.transform(R, t)
        finally:
            o.transform = keep
            self.scene.touch(structural=True)


# ---- jobs (the window posts them during a drag) ----------------------------------------------------------------------
def place_press(engine, oid):
    from .common import refuse_protected, Protected
    try:
        refuse_protected(engine.hist.scene, [oid], 'Place on surface')
    except Protected as e:
        from ..ui.engine import Refused
        engine.tool_result = {'tool': 'place', 'oid': int(oid), 'refused': 'protected', 'offer': 'unprotect'}
        raise Refused(str(e))
    engine._placer = Placer(engine.hist.scene, oid)
    engine.tool_result = {'tool': 'place', 'oid': int(oid)}
    return False


def place_drag(engine, oid, ray, align='surface', offset=0.0, spin=0.0):
    """the object rides the surface under the cursor (its transform, no history step until the drop)"""
    pl = getattr(engine, '_placer', None)
    if pl is None or pl.oid != int(oid):
        pl = engine._placer = Placer(engine.hist.scene, oid)
    h = pl.hit(ray)
    if h is None:
        return False
    p, n = h
    R, t = pl.frame(p, n, align, offset, math.radians(spin))
    o = engine.hist.scene.get(oid)
    o.transform = pl.transform(R, t)
    engine.hist.scene.touch(structural=True)
    engine.tool_result = {'tool': 'place', 'oid': int(oid), 'at': p.tolist(), 'normal': n.tolist()}
    pl.last = (R, t, n, offset)
    return True


def place_release(engine, oid):
    """the drop: the exact contact, then one undoable step (from where it was at the press)"""
    pl = getattr(engine, '_placer', None)
    if pl is None or getattr(pl, 'last', None) is None:
        return False
    R, t, n, offset = pl.last
    tr = pl.drop(R, t, n, offset)
    o = engine.hist.scene.get(oid)
    o.transform = dict(pl.t0)                          # the step records the move from the press frame
    engine.hist.scene.touch(structural=True)
    engine._push(SetObjectProps(oid, {'transform': tr}), 'Place on surface')
    if settle_hook is not None:
        pass                                            # the physics settle: the contact lane's, when it lands
    engine._placer = None
    engine.tool_result = {'tool': 'place', 'oid': int(oid), 'transform': tr}
    return True


JOBS = {'tools.place_press': place_press, 'tools.place_drag': place_drag, 'tools.place_release': place_release}
