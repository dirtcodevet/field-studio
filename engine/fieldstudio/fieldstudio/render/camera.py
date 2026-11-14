"""Perspective camera with orbit / pan / zoom / frame / preset views (no GPU, plain numpy).

fs-shell (11-11, design 4.3): an orthographic mode. With ortho on, every pixel's ray runs along the view direction f;
the view's half height at the target is D tan(fov / 2) (D = the distance to the target), so zooming (moving the camera
toward its target) and framing work unchanged, and every ray starts on a plane B = max(0, ORTHO_REACH - D) before the
camera, so nothing near the target is cut when zoomed in. ray(), project() and raymarch._cam_array (the kernels'
camera) all use this one definition."""
import math

import numpy as np

ORTHO_REACH = 1.5        # m: an orthographic ray starts at least this far before the target

PRESETS = {  # number keys: direction FROM target TO camera
    1: ('front', (0.0, 0.0, 1.0)), 3: ('right', (1.0, 0.0, 0.0)), 7: ('top', (0.0, 1.0, 0.0001)),
    9: ('back', (0.0, 0.0, -1.0)), 4: ('left', (-1.0, 0.0, 0.0)), 2: ('bottom', (0.0, -1.0, 0.0001)),
    5: ('three-quarter', (0.6, 0.35, 0.72)),
}


BASIS_CACHE = __import__('os').environ.get('FSB_BASIS_CACHE', '1') != '0'   # fs-brush-4 11-13: Camera.basis kept for the camera's values


class Camera:
    def __init__(self, pos=(0.35, 0.25, 0.6), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov=35.0, ortho=False,
                 section=None):
        self.pos = np.array(pos, float)
        self.target = np.array(target, float)
        self.up = np.array(up, float)
        self.fov = float(fov)
        self.ortho = bool(ortho)
        # fs-shell 11-11 (design 4.4, the Section tool): a view section plane (n unit, d): the half n.p > d is cut away
        # from the view (the tracers start or stop rays at the plane; the cut face is filled from the field); None: off
        self.section = None if section is None else (np.asarray(section[0], float), float(section[1]))

    def copy(self):
        c = Camera(self.pos.copy(), self.target.copy(), self.up.copy(), self.fov, self.ortho, self.section)
        bc = getattr(self, '_basis_c', None)
        if bc is not None:
            # fs-shell 11-14 (the time inventory: every frame works on a copy, and each copy rebuilt the basis on its first
            # use): the cache travels with the copy; it is keyed on the values, so a copy moved later rebuilds as before
            c._basis_c = bc
        return c

    def to_dict(self):
        d = {'pos': self.pos.tolist(), 'target': self.target.tolist(), 'up': self.up.tolist(), 'fov': self.fov,
             'ortho': self.ortho}
        if self.section is not None:
            d['section'] = [list(map(float, self.section[0])), float(self.section[1])]
        return d

    @staticmethod
    def from_dict(d):
        return Camera(d['pos'], d['target'], d.get('up', (0, 1, 0)), d.get('fov', 35.0), d.get('ortho', False),
                      d.get('section'))

    def section_vec(self):
        """-> [nx, ny, nz, d, 1] with a section plane, zeros without (the kernels' camera, raymarch._cam_array)"""
        if self.section is None:
            return np.zeros(5)
        n = np.asarray(self.section[0], float)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        return np.array([n[0], n[1], n[2], float(self.section[1]), 1.0])

    def ortho_params(self):
        """-> (D, B): the distance to the target (the view's half height there is D tan(fov / 2)) and how far before
        the camera the parallel rays start"""
        D = float(np.linalg.norm(self.target - self.pos))
        return D, max(0.0, ORTHO_REACH - D)

    def ortho_vec(self):
        """-> [1, D, B] in the orthographic view, [0, 0, 0] in perspective (the kernels' camera, raymarch._cam_array)"""
        if not self.ortho:
            return np.zeros(3)
        D, B = self.ortho_params()
        return np.array([1.0, D, B])

    def basis(self):
        # fs-brush-4 11-13 (M, fs-brush/speed4/paint1: the window's paint ran ~211 project() calls a frame, each rebuilding this
        # basis, ~31 us: ~7-13 ms of the GUI thread's time per paint, under the Python lock the engine needs): kept for the
        # camera's current VALUES (pos / target / up may be moved in place by orbit or pan, so the key is their bytes, never
        # the arrays' identity); copies are returned, as before every caller got fresh arrays
        if BASIS_CACHE:
            key = (np.asarray(self.pos, float).tobytes(), np.asarray(self.target, float).tobytes(), np.asarray(self.up, float).tobytes())
            c = getattr(self, '_basis_c', None)
            if c is not None and c[0] == key:
                return c[1].copy(), c[2].copy(), c[3].copy()
            f, r, u = self._basis()
            self._basis_c = (key, f, r, u)
            return f.copy(), r.copy(), u.copy()
        return self._basis()

    def _basis(self):
        f = self.target - self.pos
        f /= max(np.linalg.norm(f), 1e-12)
        r = np.cross(f, self.up)
        if np.linalg.norm(r) < 1e-9:
            r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return f, r, u

    def ray(self, x, y, w, h):
        """Pixel (x, y) (float, top-left origin) of a w x h image -> (origin, unit direction)."""
        f, r, u = self.basis()
        th = math.tan(math.radians(self.fov) * 0.5)
        a = w / float(h)
        px = (2.0 * (x + 0.5) / w - 1.0) * th * a
        py = (1.0 - 2.0 * (y + 0.5) / h) * th
        if self.ortho:
            D, B = self.ortho_params()
            return self.pos + (r * px + u * py) * D - f * B, f.copy()
        d = f + r * px + u * py
        return self.pos.copy(), d / np.linalg.norm(d)

    def project(self, p, w, h):
        """World point -> pixel (x, y) and view depth. Orthographic: the depth returned is D for every point in front of
        the rays' start (the depth at which the perspective formulas give the orthographic scale), -1 behind it."""
        f, r, u = self.basis()
        v = np.asarray(p, float) - self.pos
        z = float(v @ f)
        th = math.tan(math.radians(self.fov) * 0.5)
        a = w / float(h)
        if self.ortho:
            D, B = self.ortho_params()
            x = ((v @ r) / (D * th * a) + 1.0) * 0.5 * w - 0.5
            y = (1.0 - (v @ u) / (D * th)) * 0.5 * h - 0.5
            return x, y, (D if z > -B else -1.0)
        x = ((v @ r) / (z * th * a) + 1.0) * 0.5 * w - 0.5
        y = (1.0 - (v @ u) / (z * th)) * 0.5 * h - 0.5
        return x, y, z

    def orbit(self, dyaw_deg, dpitch_deg):
        v = self.pos - self.target
        dist = np.linalg.norm(v)
        yaw = math.atan2(v[0], v[2]) - math.radians(dyaw_deg)
        pitch = math.asin(max(-1.0, min(1.0, v[1] / max(dist, 1e-12)))) + math.radians(dpitch_deg)
        pitch = max(-1.55, min(1.55, pitch))
        self.pos = self.target + dist * np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch),
                                                  math.cos(pitch) * math.cos(yaw)])

    def orbit_about(self, P, dyaw_deg, dpitch_deg):
        """11-10: orbit as orbit() does, but about the point P (the surface under the cursor at the press): position and
        target both turn about P, so P stays where it is on the screen and the view keeps its aim relative to it"""
        P = np.asarray(P, float)

        def rot(axis, ang):
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        up = np.array([0.0, 1.0, 0.0])
        Ry = rot(up, -math.radians(dyaw_deg))
        pos, tgt = P + Ry @ (self.pos - P), P + Ry @ (self.target - P)
        v = pos - tgt
        side = np.cross(v, up)
        if np.linalg.norm(side) > 1e-9 and dpitch_deg:
            Rp = rot(side, math.radians(dpitch_deg))
            pos2, tgt2 = P + Rp @ (pos - P), P + Rp @ (tgt - P)
            v2 = pos2 - tgt2
            if abs(v2[1]) / max(np.linalg.norm(v2), 1e-12) < math.sin(1.55):     # as orbit(): never over the pole
                pos, tgt = pos2, tgt2
        self.pos, self.target = pos, tgt

    def zoom_about(self, P, factor):
        """11-10: zoom toward the point P (under the cursor): position and target scaled about P, the aim unchanged"""
        P = np.asarray(P, float)
        f = max(float(factor), 1e-4)
        self.pos = P + (self.pos - P) * f
        self.target = P + (self.target - P) * f

    def pan(self, dx_frac, dy_frac):
        """Pan by fractions of the view height at the target distance."""
        f, r, u = self.basis()
        dist = np.linalg.norm(self.target - self.pos)
        span = 2.0 * dist * math.tan(math.radians(self.fov) * 0.5)
        move = (-r * dx_frac + u * dy_frac) * span
        self.pos += move
        self.target += move

    def zoom(self, factor):
        v = self.pos - self.target
        self.pos = self.target + v * max(float(factor), 1e-4)

    def frame(self, lo, hi, margin=1.25, aspect=None, sphere_margin=0.0):
        """Aim at the box centre from the current direction. aspect (w / h) given (hand2b, verifier item 9): the
        distance at which all 8 box corners project inside the view (vertical and horizontal half-angles), times
        margin. Without it: the old bounding-sphere rule (a tall box such as a standing figure was cut off: the
        sphere rule with margin 0.8 left her head at the top edge and her feet below the frame)."""
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        c = (lo + hi) * 0.5
        rad = max(0.5 * np.linalg.norm(hi - lo), 1e-4)
        v = self.pos - self.target
        v /= max(np.linalg.norm(v), 1e-12)
        if aspect is None:
            dist = rad * margin / math.sin(math.radians(self.fov) * 0.5)
        else:
            self.target = c
            self.pos = c + v
            f, r, u = self.basis()
            th = math.tan(math.radians(self.fov) * 0.5)
            dist = 1e-4
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        q = np.array([x, y, z]) - c
                        # camera at c - f dist: depth = dist + f.q; need |u.q| <= depth th, |r.q| <= depth th aspect
                        dist = max(dist, abs(q @ u) / th - q @ f, abs(q @ r) / (th * float(aspect)) - q @ f)
            dist *= float(margin)
            # never nearer than the bounding-sphere rule at sphere_margin: a compact scene keeps the slack that lets
            # it be orbited after framing (the selftest frames, then orbits 18 / 14 deg)
            dist = max(dist, rad * float(sphere_margin) / math.sin(math.radians(self.fov) * 0.5))
        self.target = c
        self.pos = c + v * dist

    def preset(self, key):
        name, d = PRESETS[key]
        d = np.array(d, float)
        d /= np.linalg.norm(d)
        dist = np.linalg.norm(self.pos - self.target)
        self.pos = self.target + d * dist
        return name
