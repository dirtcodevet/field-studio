"""The chain (fs-shell, 2026-11-11; design 4.5: "plus a chain laid click by click as capsules, like ZSpheres").

Create > Mass > Chain: the first click on a form picks the object; each next click lays a capsule mass from the last
point to this one (on the form under the pointer, or, off the form, in the view plane through the last point), joined
smoothly by the object's blend dial; Esc ends the chain. A neck, a limb, a finger, an ear's rim, laid out in a few
clicks, each segment one undo step and its own mass (the gizmo moves and stretches it afterwards)."""
import math

import numpy as np


def _rot_y_to(d):
    """the rotation (Euler XYZ degrees, ops.rot_matrix's convention) taking +y onto the unit vector d"""
    from .engine import euler_from_matrix
    y = np.array([0.0, 1.0, 0.0])
    d = np.asarray(d, float) / max(np.linalg.norm(d), 1e-12)
    ax = np.cross(y, d)
    s, c = float(np.linalg.norm(ax)), float(y @ d)
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        ax /= s
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        ang = math.atan2(s, c)
        R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
    return euler_from_matrix(R)


class ChainMode:
    """tools.register's mode protocol (on_press / on_drag / on_release on the engine thread)"""

    def __init__(self, hooks, radius=0.012):
        self.hooks = hooks
        self.radius = float(radius)
        self.oid = None
        self.last = None          # the last point (world)
        self.kind = 'chain'
        self.segments = 0

    def overlay(self):
        return []

    def on_press(self, x, y):
        pass

    def on_drag(self, x, y):
        pass

    def on_release(self, x, y):
        from ..tools.masses import AddMass
        from ..tools.protected_merge import _rs
        r = self.hooks.pick(x, y)
        sc = self.hooks.history.scene
        if self.oid is None:
            if not getattr(r, 'hit', False):
                self.hooks.status('Chain: the first click goes on the form the chain grows from')
                return None
            self.oid = int(r.object_id)
            self.last = np.asarray(r.point, float)
            self.hooks.status('Chain: click the next point (on the form or off it); Esc ends')
            return None
        if getattr(r, 'hit', False) and int(r.object_id) == self.oid:
            p = np.asarray(r.point, float)
        else:
            # off the form: the point on the view plane through the last point
            cam = self.hooks.get_camera()
            w, h = self.hooks.viewport_size()
            o, d = cam.ray(float(x), float(y), w, h)
            f = np.asarray(cam.basis()[0], float)
            den = float(np.asarray(d) @ f)
            if abs(den) < 1e-9:
                return None
            t = float((self.last - np.asarray(o)) @ f) / den
            p = np.asarray(o, float) + np.asarray(d, float) * t
        if sc.index_of(self.oid) < 0:
            return None
        ob = sc.get(self.oid)
        R, tr, s = _rs(ob)
        a = ((self.last - tr) @ R) / s
        b = ((p - tr) @ R) / s
        L = float(np.linalg.norm(b - a))
        if L < 1e-4:
            return None
        self.hooks.history.push(AddMass(self.oid, 'capsule', {'radius': self.radius / s, 'height': L},
                                        translate=[float(v) for v in 0.5 * (a + b)], rotate=_rot_y_to(b - a)))
        self.last = p
        self.segments += 1
        self.hooks.status('Chain: %d segment(s); click the next point; Esc ends' % self.segments)
        return None


def start(win, radius=None):
    if radius is None:
        v = win.ask('Chain', {'radius': 0.012})
        if v is None:
            return None
        radius = float(v['radius'])
    m = ChainMode(win.tool_hooks, radius)
    win.set_tool_mode(m)
    return m
