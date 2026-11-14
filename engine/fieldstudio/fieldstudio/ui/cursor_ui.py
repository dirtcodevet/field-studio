"""The brush cursor on the surface, with live mm (fs-shell 11-11; design 4.6 "Brush cursor": "a ring drawn on the surface
(on the field's zero set within the radius), with falloff shading and the mirror cursor. Live mm: the radius, and with
Alt held the gap to the facing sheet and the thickness under the cursor, both read from the field"; chunk 9).

The ring: in 64 directions around the hover point, the point of the surface the view shows (the last frame's depth) at
exactly the brush radius from the centre: the brush's own footprint (a ball of that radius) where it meets the form,
draped over bumps and curvature, broken where the surface turns away or another form is in front. A fainter inner ring
at half the radius shades the falloff. With Mirror on, the mirrored footprint is drawn the same way (dashed). Beside
the ring: the radius in the scene's unit; with Alt held, the thickness under the cursor (inward along the normal to
where the solid ends) and the gap to the facing sheet (outward to the next surface), from the scene's exact field
(the one inside rule), refreshed as the cursor moves.

No geometry, nothing in the scene: drawn from the frame on the GUI thread; the Alt readout is one engine 'call'."""
import math

import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen
from PySide6.QtWidgets import QApplication

N_RING = 64


def thick_gap(scene, q, n, span=0.06, step=0.00025):
    """on the engine thread: from the surface point q with outward normal n -> (thickness, gap) in metres (None where
    nothing within span): the solid inward along -n until the field turns outside, the air outward along +n until it
    turns inside (the scene's exact field: every shown object's, the smallest)"""
    from ..tools.protected_merge import field_of
    pk = scene.pack()
    objs = [o for o in scene.objects if scene.shown(o)]
    if not objs:
        return None, None
    q = np.asarray(q, float)
    n = np.asarray(n, float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)

    def F(P):
        v = np.full(len(P), np.inf)
        for o in objs:
            f = field_of(scene, pk, o, P)
            v = np.minimum(v, np.where(np.isfinite(f), f, np.inf))
        return v
    s = np.arange(step, span, step)

    def cross(sign, inside_first):
        P = q[None] + sign * s[:, None] * n[None]
        f = F(np.ascontiguousarray(P))
        want = (f > 0) if inside_first else (f < 0)
        k = np.nonzero(want & (s > 2 * step))[0]
        if not len(k):
            return None
        j = int(k[0])
        a, b = (s[j - 1] if j > 0 else 0.0), s[j]
        for _ in range(30):                        # bisection on the crossing
            m = 0.5 * (a + b)
            fm = float(F((q + sign * m * n)[None])[0])
            if (fm > 0) == inside_first:
                b = m
            else:
                a = m
        return 0.5 * (a + b)
    return cross(-1.0, True), cross(1.0, False)


CURSOR_VEC = __import__('os').environ.get('FSB_CURSOR_VEC', '1') != '0'   # fs-brush-4 11-13: the cursor's ring projected in one call

class SurfaceCursor(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.vp = win.viewport
        self.alt = None                     # {'q', 'thick', 'gap'} the last Alt readout
        self._alt_q = None
        self._busy = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(90)
        self.timer.timeout.connect(self._ask_alt)
        self.win.engine.callDone.connect(self._called)
        self._tok = None
        self.last_ring = None               # (world points (n, 3) or None per direction) of the last drawn ring

    # ---- the ring on the surface ---------------------------------------------------------------------------------------
    def _surface(self, fr, Q, oid):
        """the surface points the frame shows along the rays through world points Q's pixels (depth read between pixels,
        bilinear where the 4 neighbours are the same object's surface) -> (points (n, 3), ok (n,))"""
        from ..tools.regions import project_many, rays_at
        cam = fr.get('cam') or self.vp.cam
        d = fr.get('depth')
        ob = fr.get('obj')
        h, w = d.shape[:2]
        x, y, z = project_many(cam, Q, w, h)
        ok = (z > 0) & (x >= 0) & (y >= 0) & (x <= w - 1.001) & (y <= h - 1.001)
        xi = np.clip(np.floor(x).astype(np.int64), 0, w - 2)
        yi = np.clip(np.floor(y).astype(np.int64), 0, h - 2)
        fx, fy = np.clip(x - xi, 0, 1), np.clip(y - yi, 0, 1)
        d00, d10, d01, d11 = d[yi, xi], d[yi, xi + 1], d[yi + 1, xi], d[yi + 1, xi + 1]
        D4 = np.stack([d00, d10, d01, d11], 1).astype(np.float64)
        good = np.all(np.isfinite(D4) & (D4 > 0) & (D4 < 1e29), 1)
        if ob is not None and oid is not None:
            O4 = np.stack([ob[yi, xi], ob[yi, xi + 1], ob[yi + 1, xi], ob[yi + 1, xi + 1]], 1)
            good &= np.all(O4 == oid, 1)
        t = (D4[:, 0] * (1 - fx) * (1 - fy) + D4[:, 1] * fx * (1 - fy) + D4[:, 2] * (1 - fx) * fy
             + D4[:, 3] * fx * fy)
        O, Dr = rays_at(cam, x, y, w, h)
        ok &= good
        P = O + Dr * np.where(ok, t, 0.0)[:, None]
        return P, ok

    def ring(self, c, n, r, oid, count=N_RING):
        """world points of the footprint ring (None where the surface turns away or another form is in front)"""
        fr = self.vp.frame
        if fr is None or fr.get('depth') is None:
            return None
        n = n / max(np.linalg.norm(n), 1e-12)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, a)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        ang = 2 * np.pi * np.arange(count) / count
        U_ = np.cos(ang)[:, None] * t1[None] + np.sin(ang)[:, None] * t2[None]
        rho = np.full(count, float(r))
        alive = np.ones(count, bool)
        P = np.zeros((count, 3))
        for _ in range(8):                      # rho so that the surface point there lies r from the centre
            Pk, ok = self._surface(fr, c[None] + rho[:, None] * U_, oid)
            alive &= ok
            P = np.where(alive[:, None], Pk, P)
            dist = np.linalg.norm(P - c[None], axis=1)
            alive &= dist > 1e-9
            if not alive.any():
                break
            done = np.abs(dist - r) < 0.002 * r
            if np.all(done | ~alive):
                break
            rho = np.where(alive, rho * r / np.maximum(dist, 1e-12), rho)
            alive &= rho < 4 * r
        dist = np.linalg.norm(P - c[None], axis=1)
        alive &= np.abs(dist - r) < 0.01 * r
        return [P[k] if alive[k] else None for k in range(count)]

    def _path(self, pts):
        W, H = self.vp.width(), self.vp.height()
        cam = self.vp.cam
        path = QPainterPath()
        pen_down = False
        n = len(pts)
        # fs-brush-4 11-13: the points projected in one call (tools/regions.project_many, Camera.project's own formulas)
        from ..tools.regions import project_many
        idx = [k for k in range(n) if pts[k] is not None]
        XYZ = {}
        if idx and not CURSOR_VEC:
            XYZ = {k: cam.project(pts[k], W, H) for k in idx}
        elif idx:
            X_, Y_, Z_ = project_many(cam, np.array([pts[k] for k in idx], float), W, H)
            XYZ = {k: (float(X_[j]), float(Y_[j]), float(Z_[j])) for j, k in enumerate(idx)}
        for k in range(n + 1):
            p = pts[k % n]
            if p is None:
                pen_down = False
                continue
            x, y, z = XYZ[k % n]
            if z <= 0:
                pen_down = False
                continue
            if pen_down:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                pen_down = True
        return path

    def draw(self, p):
        vp = self.vp
        if (vp.hover is None or vp.tool == 'select' or vp.brush is None or vp.tool in ('move', 'rotate', 'scale')
                or getattr(vp, 'drag', None) in ('orbit', 'pan') or getattr(vp, 'tool_mode', None) is not None):
            return
        c = np.asarray(vp.hover['point'], float)
        n = np.asarray(vp.hover['normal'], float)
        oid = vp.hover.get('oid')
        r = float(vp.brush['radius'])
        pts = self.ring(c, n, r, oid)
        if pts is None:
            return
        self.last_ring = pts
        col = {'carve': QColor(255, 140, 120), 'mask': QColor(170, 170, 255), 'smooth': QColor(160, 230, 255),
               'grab': QColor(255, 220, 120), 'stretch': QColor(255, 220, 120), 'snake': QColor(255, 220, 120),
               'elastic': QColor(255, 220, 120)}.get(vp.tool, QColor(140, 190, 255))
        path = self._path(pts)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 120), 3.0))
        p.drawPath(path)
        p.setPen(QPen(col, 1.6))
        p.drawPath(path)
        inner = self.ring(c, n, 0.5 * r, oid, 32)      # the falloff, shaded by a fainter inner ring
        if inner is not None:
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 90), 1.0))
            p.drawPath(self._path(inner))
        W, H = vp.width(), vp.height()
        x, y, z = vp.cam.project(c, W, H)
        p.setPen(QPen(col, 1.6))
        p.drawLine(QPointF(x - 4, y), QPointF(x + 4, y))
        p.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))
        # the mirror cursor: the mirrored footprint, drawn the same way
        sym = vp.brush.get('symmetry', [False] * 3)
        tr = vp.transforms.get(oid) if hasattr(vp, 'transforms') else None
        if any(sym) and tr is not None:
            from ..core.ops import rot_matrix
            R = rot_matrix(tr['rotate'])
            s = max(float(tr['scale']), 1e-6)
            t = np.asarray(tr['translate'], float)
            for ax in range(3):
                if not sym[ax]:
                    continue
                cl = R.T @ (c - t) / s
                nl = R.T @ n
                cl[ax] = -cl[ax]
                nl[ax] = -nl[ax]
                mpts = self.ring(R @ (cl * s) + t, R @ nl, r, oid)
                if mpts is not None:
                    p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 140), 1.2, Qt.DashLine))
                    p.drawPath(self._path(mpts))
        # live mm
        from . import units as U
        unit = getattr(self.win, 'units', 'mm')
        lines = ['r %s' % U.format_length(r, unit)]
        if QApplication.keyboardModifiers() & Qt.AltModifier:
            self._want_alt(c, n)
            a = self.alt
            if a is not None and np.linalg.norm(np.asarray(a['q']) - c) < 0.002:
                lines.append('thickness %s' % (U.format_length(a['thick'], unit) if a['thick'] is not None
                                                 else '> 60 mm'))
                lines.append('gap %s' % (U.format_length(a['gap'], unit) if a['gap'] is not None else 'none within 60 mm'))
            else:
                lines.append('reading...')
        xs = [vp.cam.project(q, W, H)[0] for q in pts if q is not None]
        lx = (max(xs) if xs else x) + 10
        p.setFont(QFont('Segoe UI', 9))
        fm = p.fontMetrics()
        for i, txt in enumerate(lines):
            yy = y - 8 + i * (fm.height() + 2)
            p.setPen(QPen(QColor(0, 0, 0, 170), 1.0))
            p.drawText(QPointF(lx + 1, yy + 1), txt)
            p.setPen(QPen(QColor(235, 238, 245), 1.0))
            p.drawText(QPointF(lx, yy), txt)

    # ---- the Alt readout -----------------------------------------------------------------------------------------------
    def _want_alt(self, c, n):
        if self._alt_q is not None and np.linalg.norm(self._alt_q[0] - c) < 1e-4:
            return
        self._alt_q = (c.copy(), n.copy())
        if not self.timer.isActive():
            self.timer.start()

    def _ask_alt(self):
        if self._busy or self._alt_q is None:
            if self._alt_q is not None:
                self.timer.start()
            return
        c, n = self._alt_q
        self._busy = True
        tok = object()
        self._tok = tok
        self.win.engine.post('call', lambda e, c=c, n=n: (c, thick_gap(e.hist.scene, c, n)), tok)

    def _called(self, rr):
        if rr[0] is not self._tok:
            return
        self._busy = False
        res = rr[1]
        if res is not None and not getattr(res, 'failed', False):     # a failed call answers with a falsy CallFailed (fs-brush cba5ef61); the engine reports the failure itself
            c, (th, gp) = res
            self.alt = {'q': c, 'thick': th, 'gap': gp}
            self.vp.update()


class _AltFilter(QObject):
    """Alt pressed or released over the view: redraw at once (the thickness / gap readout comes and goes)"""

    def __init__(self, sc):
        super().__init__(sc)
        self.sc = sc

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if ev.type() in (QEvent.KeyPress, QEvent.KeyRelease) and ev.key() == Qt.Key_Alt:
            self.sc.vp.update()
        return False


def install(win):
    sc = SurfaceCursor(win)
    win.surface_cursor = sc
    vp = win.viewport
    vp._draw_ring = lambda p: None                  # the flat tangent-plane circle gives way to the surface ring
    flat = vp._draw_cursor

    def _draw_cursor(p, flat=flat, vp=vp):
        # the 2-D screen circle stays off the form (and for the F-drag readout); on the form the surface ring shows it.
        # fs-shell-5 11-14: a key's read-out ([ ] - = Shift+[ ]) shows on the form too, for its moment
        import time as _t
        fl_ = getattr(vp, '_flash', None)
        flashing = fl_ is not None and _t.perf_counter() < fl_[1]
        if (vp.hover is not None and getattr(vp, 'adjust', None) is None and sc.last_ring is not None
                and not flashing):
            return
        flat(p)
    vp._draw_cursor = _draw_cursor
    sc.filter = _AltFilter(sc)
    win.viewport.installEventFilter(sc.filter)
    return sc
