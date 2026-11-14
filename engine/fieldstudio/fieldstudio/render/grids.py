"""Grids for scale (fs-shell, 2026-11-11; design Appendix D2: a grid overlay, toggled on or off, that gives the
measurements for scale from any viewing angle, its scale chosen by the user, to millimetre or inch precision).

Two of D2's three grids (the surface contours are chunk 9, in the tracer):
- Floor grid (perspective views): the plane under the scene (y = its lowest point, from the objects' compiled bounds:
  the figure's feet at 0, a bust's underside), depth-tested against the frame, so the model hides it where the model is
  nearer; X axis red, Z axis blue; it fades toward the horizon.
- View grid (axis views: the view within ~2 deg of a world axis, orthographic or the near-orthographic lens): graph
  paper in the view plane through the view's target, on the empty background only, so the reference photos (drawn
  after it) sit on it like a draughtsman's sheet.
Spacing: chosen (metres) or adaptive (finer as you zoom, as Blender's floor): the finest step of the unit's ladder whose
lines stay >= MIN_PX apart; major lines every ladder step up, labelled in the scene's unit (the viewport's overlay
draws the labels: draw_labels). Snapping (the gizmo's Ctrl) snaps to the minor step of the grid in use.
Mesh-free: lines are a per-pixel distance to the nearest multiple of the step, antialiased by the pixel's footprint."""
import math

import numpy as np
from numba import njit, prange

METRIC = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
IMPERIAL = [0.0254 / 16, 0.0254 / 4, 0.0254, 0.0254 * 6, 0.3048, 0.3048 * 5, 0.3048 * 10, 0.3048 * 50,
            0.3048 * 100]
MIN_PX = 9.0                 # adaptive: minor lines at least this many pixels apart
AXIS_COS = math.cos(math.radians(2.0))
COL_MINOR = (0.62, 0.66, 0.72)
COL_MAJOR = (0.80, 0.83, 0.88)
COL_X = (0.92, 0.32, 0.32)
COL_Z = (0.36, 0.56, 0.96)
COL_Y = (0.43, 0.80, 0.37)


def ladder(unit):
    return IMPERIAL if unit in ('in', 'ft-in') else METRIC


def steps_for(px_per_m, unit, chosen=None):
    """-> (minor, major) spacing in metres: the chosen minor (its major the ladder's next step up, or 10x), or adaptive"""
    L = ladder(unit)
    if chosen:
        minor = float(chosen)
        up = [s for s in L if s > minor * 1.0001]
        return minor, (up[0] if up and up[0] / minor <= 12.5 else minor * 10.0)
    for i, s in enumerate(L):
        if s * px_per_m >= MIN_PX:
            return s, (L[i + 1] if i + 1 < len(L) else s * 10.0)
    return L[-1], L[-1] * 10.0


def view_kind(cam):
    """'view' (axis view: the view grid), or 'floor'"""
    f = np.asarray(cam.basis()[0], float)
    if getattr(cam, 'ortho', False) or float(np.max(np.abs(f))) >= AXIS_COS:
        if float(np.max(np.abs(f))) >= AXIS_COS:
            return 'view'
    return 'floor'


@njit(cache=True)
def _line_cov(a, step, fp):
    """coverage (0..1) of the line nearest coordinate a (lines at multiples of step) for a pixel footprint fp"""
    r = a / step
    d = abs(r - math.floor(r + 0.5)) * step
    w = 0.6 * fp
    return max(0.0, 1.0 - d / max(w, 1e-12))


@njit(cache=True)
def _over(r, g, b, a, k, cr, cg, cb):
    """layer a line of colour (cr, cg, cb) and opacity k over (r, g, b, a) (premultiplied)"""
    if k <= 0.0:
        return r, g, b, a
    return r * (1.0 - k) + cr * k, g * (1.0 - k) + cg * k, b * (1.0 - k) + cb * k, a + k * (1.0 - a)


@njit(parallel=True, cache=True, nogil=True)
def floor_layer(cam, w, h, fy, minor, major, fade, T, C):
    """per pixel: the floor's hit (T: ray t, inf none) and its RGBA (C)"""
    th = cam[12]
    orth = cam.shape[0] > 18 and cam[16] > 0.5
    oD = cam[17] if orth else 0.0
    oB = cam[18] if orth else 0.0
    pix = 2.0 * th / h
    for j in prange(h):
        for i in range(w):
            T[j, i] = np.inf
            C[j, i, 3] = 0.0
            px = (2.0 * (i + 0.5) / w - 1.0) * th * w / h
            py = (1.0 - 2.0 * (j + 0.5) / h) * th
            if orth:
                ox = cam[0] + (cam[6] * px + cam[9] * py) * oD - cam[3] * oB
                oy = cam[1] + (cam[7] * px + cam[10] * py) * oD - cam[4] * oB
                oz = cam[2] + (cam[8] * px + cam[11] * py) * oD - cam[5] * oB
                dx, dy, dz = cam[3], cam[4], cam[5]
            else:
                ox, oy, oz = cam[0], cam[1], cam[2]
                dx = cam[3] + cam[6] * px + cam[9] * py
                dy = cam[4] + cam[7] * px + cam[10] * py
                dz = cam[5] + cam[8] * px + cam[11] * py
                dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= dl
                dy /= dl
                dz /= dl
            if abs(dy) < 1e-6:
                continue
            t = (fy - oy) / dy
            if t <= 0.0:
                continue
            x = ox + dx * t
            z = oz + dz * t
            # the pixel's footprint on the floor: its angular size times t, stretched by the grazing angle
            graze = max(abs(dy), 0.02)
            fp = (pix * (oD if orth else t)) / graze
            # fs-shell 11-12 (t52): lines fade in with their spacing on screen (none under 3 px apart, whole from 9 px),
            # so the far floor thins out smoothly instead of a dense mesh with a hard edge where the minors stop
            wm = min(max((minor / fp - 3.0) / 6.0, 0.0), 1.0)
            wM = min(max((major / fp - 3.0) / 6.0, 0.0), 1.0)
            cm = max(_line_cov(x, minor, fp), _line_cov(z, minor, fp)) * wm if wm > 0.0 else 0.0
            cM = max(_line_cov(x, major, fp), _line_cov(z, major, fp)) * wM if wM > 0.0 else 0.0
            cx = _line_cov(z, 1e9, fp) if abs(z) < 4.0 * fp else 0.0     # the X axis (z = 0)
            cz = _line_cov(x, 1e9, fp) if abs(x) < 4.0 * fp else 0.0     # the Z axis (x = 0)
            fadef = math.exp(-(t / fade) * (t / fade)) if fade > 0.0 else 1.0   # t52: held near, gone far
            r, g, b, a = _over(0.0, 0.0, 0.0, 0.0, cm * 0.22, 0.62, 0.66, 0.72)
            r, g, b, a = _over(r, g, b, a, cM * 0.42, 0.80, 0.83, 0.88)
            r, g, b, a = _over(r, g, b, a, cx * 0.85, 0.92, 0.32, 0.32)
            r, g, b, a = _over(r, g, b, a, cz * 0.85, 0.36, 0.56, 0.96)
            if a * fadef <= 0.002:
                continue
            T[j, i] = t
            C[j, i, 0] = r / a
            C[j, i, 1] = g / a
            C[j, i, 2] = b / a
            C[j, i, 3] = a * fadef


@njit(parallel=True, cache=True, nogil=True)
def view_layer(w, h, u0, v0, minor, major, mpp, C):
    """the view grid: graph paper in the view plane through the target; u0 / v0 the target's coordinates along the
    screen's right / up (a world axis, signed so it grows rightward / upward), mpp metres per pixel at the target"""
    for j in prange(h):
        for i in range(w):
            C[j, i, 3] = 0.0
            u = (i + 0.5 - 0.5 * w) * mpp + u0
            v = (0.5 * h - (j + 0.5)) * mpp + v0
            fp = mpp
            cm = max(_line_cov(u, minor, fp), _line_cov(v, minor, fp)) if mpp < minor * 0.5 else 0.0
            cM = max(_line_cov(u, major, fp), _line_cov(v, major, fp)) if mpp < major * 0.5 else 0.0
            c0u = _line_cov(u, 1e9, fp) if abs(u) < 4.0 * fp else 0.0     # the world axis through 0 along v
            c0v = _line_cov(v, 1e9, fp) if abs(v) < 4.0 * fp else 0.0
            r, g, b, al = _over(0.0, 0.0, 0.0, 0.0, cm * 0.16, 0.62, 0.66, 0.72)
            r, g, b, al = _over(r, g, b, al, cM * 0.34, 0.80, 0.83, 0.88)
            r, g, b, al = _over(r, g, b, al, c0u * 0.6, 0.85, 0.85, 0.85)
            r, g, b, al = _over(r, g, b, al, c0v * 0.6, 0.85, 0.85, 0.85)
            if al <= 0.002:
                continue
            C[j, i, 0] = r / al
            C[j, i, 1] = g / al
            C[j, i, 2] = b / al
            C[j, i, 3] = al


@njit(cache=True, nogil=True)
def blend(rgb, depth, obj, T, C, mode, out):
    """per frame: mode 0 (floor): where the floor is nearer than the model (or no model); mode 1 (view grid): on the
    empty background only. depth / obj may be at a lower resolution than rgb (nearest)."""
    h, w = rgb.shape[0], rgb.shape[1]
    dh, dw = obj.shape[0], obj.shape[1]
    for j in range(h):
        jj = min(j * dh // h, dh - 1)
        for i in range(w):
            a = C[j, i, 3]
            if a <= 0.0:
                continue
            ii = min(i * dw // w, dw - 1)
            if mode == 1:
                if obj[jj, ii] >= 0:
                    continue
            else:
                if obj[jj, ii] >= 0 and not (T[j, i] < depth[jj, ii]):
                    continue
            for c in range(3):
                v = rgb[j, i, c] * (1.0 - a) + C[j, i, c] * 255.0 * a
                out[j, i, c] = np.uint8(min(max(v + 0.5, 0.0), 255.0))


class Grid:
    """the viewport's grids: settings, the cached layer per view, composite(), labels and the snapping step"""

    def __init__(self, vp):
        self.vp = vp
        self.floor = False
        self.view = False
        self.spacing = None            # metres (chosen) or None (adaptive)
        self.floor_y = 0.0
        self.fade = 4.0                # m: the floor fades with distance
        self._cache = {'key': None}
        self.last = None               # (kind, minor, major) of the grid drawn last
        self._lows = {}                # oid -> (key, lowest y) from the base tree's compiled bounds

    def scene_changed(self, snap):
        """the floor under the scene: the lowest point of the visible objects' base forms (world y), 0 when empty"""
        from ..core import ops
        lows = []
        for o in (snap or {}).get('objects', []):
            if not o.get('visible', True) or o.get('hidden') or str(o.get('name', '')).startswith('Cutter'):
                continue
            t = o.get('transform') or {}
            key = (o.get('nodes'), repr(t), repr(o.get('base'))[:4000])
            c = self._lows.get(o['id'])
            if c is None or c[0] != key:
                try:
                    _c, _p, lo, hi, _L = ops.compile_tree(ops.Node.from_dict(o['base']))
                    R = ops.rot_matrix(t.get('rotate', [0.0, 0.0, 0.0]))
                    s_ = float(t.get('scale', 1.0))
                    tr = np.asarray(t.get('translate', [0.0, 0.0, 0.0]), float)
                    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                    low = float((corners @ R.T * s_ + tr)[:, 1].min())
                except Exception:
                    low = None
                c = (key, low)
                self._lows[o['id']] = c
            if c[1] is not None:
                lows.append(c[1])
        fy = round(min(lows), 4) if lows else 0.0
        if fy != self.floor_y:
            self.floor_y = fy
            return True
        return False

    def unit(self):
        try:
            return getattr(self.vp.window(), 'units', 'mm')
        except Exception:
            return 'mm'

    def active(self):
        return self.floor or self.view

    def _px_per_m(self, cam, h):
        th = math.tan(math.radians(cam.fov) * 0.5)
        if getattr(cam, 'ortho', False):
            D, B = cam.ortho_params()
            return h / (2.0 * th * D)
        d = float(np.linalg.norm(np.asarray(cam.target, float) - np.asarray(cam.pos, float)))
        return h / (2.0 * th * max(d, 1e-6))

    def state(self, cam, h):
        kind = view_kind(cam)
        if (kind == 'view' and not self.view) or (kind == 'floor' and not self.floor):
            return None
        ppm = self._px_per_m(cam, h)
        minor, major = steps_for(ppm, self.unit(), self.spacing)
        return kind, minor, major, ppm

    @staticmethod
    def _uv0(cam):
        """the view's target along the screen's right and up (each a world axis, signed to grow right / up)"""
        f, r, u = (np.asarray(v, float) for v in cam.basis())
        ui_, uj = int(np.argmax(np.abs(r))), int(np.argmax(np.abs(u)))
        tg = np.asarray(cam.target, float)
        return tg[ui_] * np.sign(r[ui_]), tg[uj] * np.sign(u[uj])

    def step(self):
        """the snapping step of the grid in use in this view (m), or None"""
        st = self.state(self.vp.cam, max(self.vp.height(), 1))
        return st[1] if st else None

    def composite(self, fr, rgb):
        from .raymarch import _cam_array
        cam = fr['cam']
        h, w = rgb.shape[:2]
        st = self.state(cam, h)
        if st is None:
            self.last = None
            return rgb
        kind, minor, major, ppm = st
        ca = _cam_array(cam)
        # t52: the floor fades over a distance that follows the view (Blender's grid thins out the same way): ~5x the
        # distance to what the view looks at, never beyond self.fade x 3
        dv = float(np.linalg.norm(np.asarray(cam.target, float) - np.asarray(cam.pos, float)))
        fade = min(max(5.0 * dv, 0.25), 3.0 * self.fade)
        key = (kind, tuple(np.round(ca, 12)), w, h, minor, major, self.floor_y, fade)
        c = self._cache
        if c['key'] != key:
            C = np.zeros((h, w, 4), np.float32)
            T = np.full((h, w), np.inf, np.float32)
            if kind == 'floor':
                floor_layer(ca, w, h, float(self.floor_y), float(minor), float(major), float(fade), T, C)
            else:
                u0, v0 = self._uv0(cam)
                view_layer(w, h, float(u0), float(v0), float(minor), float(major), 1.0 / ppm, C)
            c.update(key=key, T=T, C=C)
        out = rgb.copy()
        depth = fr.get('depth')
        obj = np.ascontiguousarray(fr['obj'])
        depth = np.ascontiguousarray(depth if depth is not None else np.full(obj.shape, np.inf, np.float32),
                                     dtype=np.float32)
        blend(np.ascontiguousarray(rgb), depth, obj, c['T'], c['C'], 0 if kind == 'floor' else 1, out)
        self.last = (kind, minor, major)
        return out

    def draw_labels(self, p):
        """the major lines' values in the scene's unit (the viewport's overlay painter)"""
        if self.last is None:
            return
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QFont
        from ..ui import units as U
        vp = self.vp
        kind, minor, major = self.last
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        cam = vp.cam
        unit = self.unit()
        p.setFont(QFont('Segoe UI', 7))
        p.setPen(QColor(200, 205, 214, 210))
        fr = getattr(vp, 'frame', None)
        obj = None if fr is None else np.asarray(fr.get('obj'))
        dep = None if fr is None or fr.get('depth') is None else np.asarray(fr['depth'])

        def hidden(sx, sy, t):
            """a label point the model hides (the frame's hit nearer than the point)"""
            if obj is None:
                return False
            jj = min(max(int(sy * obj.shape[0] / H), 0), obj.shape[0] - 1)
            ii = min(max(int(sx * obj.shape[1] / W), 0), obj.shape[1] - 1)
            if obj[jj, ii] < 0:
                return False
            return dep is None or float(dep[jj, ii]) < t
        if kind == 'view':
            f, r, u = (np.asarray(v, float) for v in cam.basis())
            ui_, uj = int(np.argmax(np.abs(r))), int(np.argmax(np.abs(u)))
            ppm = self._px_per_m(cam, H)
            u0, v0 = self._uv0(cam)
            fm = p.fontMetrics()
            # fs-shell 11-11 (the 21:xx rerun's shot): at a fine major step the labels ran into each other
            # ("90 mm100 mm110 mm"); a label is left out when it would touch the one before it (every other one, or
            # more, stays: the lines keep their spacing), as the floor grid's labels already were
            k0 = math.ceil((u0 - 0.5 * W / ppm) / major)
            k1 = math.floor((u0 + 0.5 * W / ppm) / major)
            right = -1e9
            for k in range(int(k0), int(k1) + 1):
                x = 0.5 * W + (k * major - u0) * ppm
                text = U.format_length(k * major * np.sign(r[ui_]), unit, digits=None).replace('.0 mm', ' mm')
                if x + 2 < right + 8:
                    continue
                p.drawText(QPointF(x + 2, H - 6), text)
                right = x + 2 + fm.horizontalAdvance(text)
            k0 = math.ceil((v0 - 0.5 * H / ppm) / major)
            k1 = math.floor((v0 + 0.5 * H / ppm) / major)
            last = -1e9
            for k in range(int(k0), int(k1) + 1):
                y = 0.5 * H - (k * major - v0) * ppm
                if abs(y - last) < fm.height() + 2:
                    continue
                p.drawText(QPointF(4, y - 2), U.format_length(k * major * np.sign(u[uj]), unit).replace('.0 mm', ' mm'))
                last = y
        else:
            for ax in (0, 2):
                last = None
                for k in sorted(range(-12, 13), key=abs):
                    if k == 0:
                        continue
                    P = np.zeros(3)
                    P[1] = self.floor_y
                    P[ax] = k * major
                    sx, sy, z = cam.project(P, W, H)
                    if z <= 1e-6 or not (0 <= sx < W and 0 <= sy < H):
                        continue
                    t = float(np.linalg.norm(P - np.asarray(cam.pos, float)))
                    if hidden(sx, sy, t):
                        continue
                    placed = getattr(self, '_placed', [])
                    if any((sx - a) ** 2 + (sy - b) ** 2 < 70.0 ** 2 for a, b in placed):
                        continue                 # crowded toward the horizon: every other label left out
                    placed.append((sx, sy))
                    self._placed = placed
                    p.drawText(QPointF(sx + 3, sy - 3), ('x ' if ax == 0 else 'z ') +
                               U.format_length(k * major, unit).replace('.0 mm', ' mm'))
            self._placed = []
        p.setPen(QColor(150, 158, 170, 220))
        p.drawText(QPointF(W - 230, H - 22), 'grid: %s, major %s' % (U.format_length(minor, unit),
                                                                   U.format_length(major, unit)))


# ---- D2's third grid: contour lines on the model ------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def contour_kernel(cam, depth, obj, step, axes, alpha, rgb_in, out):
    """lines on the model wherever x / y / z crosses a multiple of step (axes: 3 flags), from each pixel's hit point
    (its ray and depth: the tracer's own t), antialiased by the hit point's change to the next pixel (the line's width
    in the pixel's own world footprint). depth / obj at the frame's size."""
    h, w = depth.shape[0], depth.shape[1]
    th = cam[12]
    orth = cam.shape[0] > 18 and cam[16] > 0.5
    oD = cam[17] if orth else 0.0
    oB = cam[18] if orth else 0.0
    P = np.empty((h, w, 3))
    for j in range(h):
        for i in range(w):
            if obj[j, i] < 0 or not (depth[j, i] < 1e20):
                P[j, i, 0] = np.nan
                continue
            px = (2.0 * (i + 0.5) / w - 1.0) * th * w / h
            py = (1.0 - 2.0 * (j + 0.5) / h) * th
            if orth:
                ox = cam[0] + (cam[6] * px + cam[9] * py) * oD - cam[3] * oB
                oy = cam[1] + (cam[7] * px + cam[10] * py) * oD - cam[4] * oB
                oz = cam[2] + (cam[8] * px + cam[11] * py) * oD - cam[5] * oB
                dx, dy, dz = cam[3], cam[4], cam[5]
            else:
                ox, oy, oz = cam[0], cam[1], cam[2]
                dx = cam[3] + cam[6] * px + cam[9] * py
                dy = cam[4] + cam[7] * px + cam[10] * py
                dz = cam[5] + cam[8] * px + cam[11] * py
                dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= dl
                dy /= dl
                dz /= dl
            t = depth[j, i]
            P[j, i, 0] = ox + dx * t
            P[j, i, 1] = oy + dy * t
            P[j, i, 2] = oz + dz * t
    cols = ((0.95, 0.40, 0.40), (0.45, 0.85, 0.45), (0.45, 0.62, 1.0))
    for j in range(h):
        for i in range(w):
            if P[j, i, 0] != P[j, i, 0]:
                continue
            r = rgb_in[j, i, 0] / 255.0
            g = rgb_in[j, i, 1] / 255.0
            b = rgb_in[j, i, 2] / 255.0
            for a in range(3):
                if not axes[a]:
                    continue
                v = P[j, i, a]
                # the footprint: the hit point's change to the neighbours (same object, no depth jump)
                fp = 0.0
                for (jj, ii) in ((j, i + 1), (j + 1, i), (j, i - 1), (j - 1, i)):
                    if 0 <= jj < h and 0 <= ii < w and P[jj, ii, 0] == P[jj, ii, 0] and obj[jj, ii] == obj[j, i]:
                        dv = abs(P[jj, ii, a] - v)
                        if dv < 4.0 * step and dv > fp:
                            fp = dv
                if fp <= 0.0 or fp > 0.5 * step:
                    continue                    # lines closer than two pixels here: left out (no moire)
                q = v / step
                d = abs(q - math.floor(q + 0.5)) * step
                cov = max(0.0, 1.0 - d / (0.75 * fp))
                if cov <= 0.0:
                    continue
                k = cov * alpha
                cr, cg, cb = cols[a]
                r = r * (1.0 - k) + cr * k
                g = g * (1.0 - k) + cg * k
                b = b * (1.0 - k) + cb * k
            out[j, i, 0] = np.uint8(min(max(r * 255.0 + 0.5, 0.0), 255.0))
            out[j, i, 1] = np.uint8(min(max(g * 255.0 + 0.5, 0.0), 255.0))
            out[j, i, 2] = np.uint8(min(max(b * 255.0 + 0.5, 0.0), 255.0))


class Contours:
    """design D2's surface contours: on the model, every `spacing` along the chosen axes (default: y, the height),
    spacing as the grid's (adaptive) or chosen; drawn from the frame's own hit points (no geometry, no tracer change)"""

    def __init__(self, vp, grid):
        self.vp, self.grid = vp, grid
        self.on = False
        self.axes = np.array([False, True, False])
        self.alpha = 0.75
        self.last_step = None

    def active(self):
        return self.on

    def composite(self, fr, rgb):
        from .raymarch import _cam_array
        depth = fr.get('depth')
        obj = fr.get('obj')
        if depth is None or obj is None:
            return rgb
        depth = np.ascontiguousarray(depth, dtype=np.float64)
        obj = np.ascontiguousarray(obj)
        if depth.shape != rgb.shape[:2] or obj.shape != rgb.shape[:2]:
            return rgb                          # a preview frame at another size: the next full frame draws them
        cam = fr['cam']
        h = rgb.shape[0]
        ppm = self.grid._px_per_m(cam, h)
        step = self.grid.spacing or steps_for(ppm, self.grid.unit(), None)[0]
        self.last_step = step
        out = rgb.copy()
        contour_kernel(_cam_array(cam), depth, obj, float(step), self.axes, float(self.alpha),
                       np.ascontiguousarray(rgb), out)
        return out


def warmup():
    """fs-shell-5 11-14 (t68: the first frame's grid composite compiled on the WINDOW thread, 1.2 s waiting on
    numba's compiler lock while the engine compiled): every grid kernel compiled in the engine's warm-up, on tiny frames
    of each type a frame arrives in (card frames carry int16 ids, CPU frames int32)"""
    from .camera import Camera
    from .raymarch import _cam_array
    ca = _cam_array(Camera(pos=(0.0, 0.3, 0.6), target=(0, 0, 0), fov=35.0))
    w, h = 8, 6
    T = np.full((h, w), np.inf, np.float32)
    C = np.zeros((h, w, 4), np.float32)
    floor_layer(ca, w, h, 0.0, 0.01, 0.05, 1.0, T, C)
    view_layer(w, h, 0.0, 0.0, 0.01, 0.05, 0.001, C)
    rgb = np.zeros((h, w, 3), np.uint8)
    out = rgb.copy()
    for od in (np.int16, np.int32):
        obj = np.full((h, w), -1, od)
        for mode in (0, 1):
            blend(rgb, np.full((h, w), np.inf, np.float32), obj, T, C, mode, out)
        contour_kernel(ca, np.full((h, w), np.inf, np.float64), obj, 0.01, np.array([False, True, False]), 0.75, rgb,
                       out)
