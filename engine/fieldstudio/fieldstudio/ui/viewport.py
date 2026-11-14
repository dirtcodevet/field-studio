"""The viewport: shows the engine's CPU-traced frames, takes navigation and brush input, draws overlays
(brush ring on the surface, selection outline, XYZ gizmo, field read-out). It never touches the scene: picks,
strokes and renders are engine jobs."""
import math
import time

import numpy as np
from numba import njit, prange
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ..render import Camera

SEL_COLOUR = np.array([255, 160, 40], np.uint8)
HOVER_COLOUR = np.array([120, 200, 255], np.uint8)     # fs-shell-2 11-14: the Scene list's hovered object (part 4)


def _cover_label(m, obj, win, selected):
    """fs-shell-2 11-14: '<selected> · hidden behind <the object most in front of its covered outline>' and where
    (fractions of the frame: the covered part's top centre); read from the ID images (no geometry), None when little
    of it is covered"""
    if obj is None or obj.shape != m.shape:
        return None
    full = (m & 1) > 0
    covered = full & ((m & 2) == 0)
    if covered.sum() < 40:
        return None
    ids, counts = np.unique(obj[covered], return_counts=True)
    keep = ids >= 0
    if not keep.any():
        return None
    top = int(ids[keep][np.argmax(counts[keep])])
    names = {o['id']: o['name'] for o in ((getattr(win, 'snap', None) or {}).get('objects') or [])}
    who = names.get(selected[0], 'object') if selected else 'object'
    ys, xs = np.nonzero(covered)
    return ('%s  \u00b7  hidden behind %s' % (who, names.get(top, 'object %d' % top)),
            float(xs.mean()) / m.shape[1], float(ys.min()) / m.shape[0])


def _id_outline(rgb, m, colour, visible=True):
    """fs-shell-2 11-14: select inside part 3 / part 4. m: an object's own ID image (gpu/viewmodes.vol_mask: bit 0 the
    object is on this pixel's ray, bit 1 it is the front surface here). Its outline: solid where it is the front surface
    (when visible=True), dashed where something covers it.
    PATENT GUARDS (QuantifiCare US 11,094,139 B2; design/LIBRARIES_AND_PATENTS_2026-11-14.md addendum bab2b321):
      1. the line is ONLY the edge of this ID image (a pixel of the object with a neighbour off it), never from N.V and
         never from a contour of the field gradient against the view direction;
      2. one constant colour and alpha along the whole line (no fade, glow or width by angle);
      3. the hidden object itself is never drawn shaded, tinted or translucent here: only its edge pixels are set."""
    full = (m & 1) > 0
    if not full.any():
        return rgb
    inner = full.copy()
    inner[1:, :] &= full[:-1, :]
    inner[:-1, :] &= full[1:, :]
    inner[:, 1:] &= full[:, :-1]
    inner[:, :-1] &= full[:, 1:]
    edge = full & ~inner
    front = (m & 2) > 0
    yy, xx = np.indices(m.shape)
    dash = ((xx + yy) // 4) % 2 == 0
    covered = edge & ~front & dash
    out = rgb.copy()
    out[covered] = colour
    if visible:
        out[edge & front] = colour
    return out


def _box(m, r):
    """Box mean of a 2-D float array over (2r+1)^2 (edge-clamped). r47: separable shifted sums in float32 (exact for
    the outline's small counts; r43's float64 cumulative sums fixed the 0.01 alpha lost by float32 cumulative sums over
    the frame, but cost the UI thread ~20 ms more a frame at 1600 x 900); 2x faster than either."""
    m = np.asarray(m, np.float32)
    H, W = m.shape
    p = np.pad(m, ((0, 0), (r, r)), mode='edge')
    h = np.zeros((H, W), np.float32)
    for d in range(2 * r + 1):
        h += p[:, d:d + W]
    p = np.pad(h, ((r, r), (0, 0)), mode='edge')
    v = np.zeros((H, W), np.float32)
    for d in range(2 * r + 1):
        v += p[d:d + H]
    return v / float((2 * r + 1) ** 2)


@njit(cache=True, nogil=True, parallel=True)
def _blend_alpha(rgb, a, colour, out):
    """r39: out = rgb blended toward colour by the byte alpha a (the card computed it)"""
    H, W = a.shape
    for y in prange(H):
        for x in range(W):
            al = a[y, x] / 255.0
            for c in range(3):
                v = rgb[y, x, c]
                out[y, x, c] = np.uint8(v * (1.0 - al) + colour[c] * al) if al > 0.02 else v


def warmup_composites():
    """fs-shell-5 11-14: the selection blend compiled in the engine's warm-up, not at the first selection (t68:
    490 ms on the window thread)"""
    rgb = np.zeros((6, 8, 3), np.uint8)
    _blend_alpha(rgb, np.zeros((6, 8), np.uint8), SEL_COLOUR, rgb.copy())


def _draw_soft_edge(rgb, sel, scale, colour):
    """Blend the soft outline into a copy of rgb; works only on the mask's edge band (cheap on the UI thread)."""
    r = max(1, int(scale))
    pad = 2 * r + 2
    # the blur runs over the mask's bounding box (+ its reach) only; the blend touches edge pixels only
    rows = np.nonzero(sel.any(1))[0]
    cols = np.nonzero(sel.any(0))[0]
    y0 = max(int(rows[0]) - pad, 0)
    y1 = min(int(rows[-1]) + pad + 1, sel.shape[0])
    x0 = max(int(cols[0]) - pad, 0)
    x1 = min(int(cols[-1]) + pad + 1, sel.shape[1])
    sub = sel[y0:y1, x0:x1]
    a = _soft_edge(sub, r)
    idx = np.nonzero(a > 0.02)
    out = rgb.copy()
    reg = out[y0:y1, x0:x1]
    al = a[idx][:, None]
    reg[idx] = (reg[idx] * (1.0 - al) + np.asarray(colour, np.float32) * al).astype(np.uint8)
    return out


def _soft_edge(sel, scale):
    """Outline alpha peaked on the mask boundary, about 1.5 px wide, smoothed over the preview's pixel scale."""
    r = max(1, int(scale))
    b = _box(_box(sel.astype(np.float32), r), r)
    return np.clip(1.5 * (1.0 - np.abs(2.0 * b - 1.0)) - 0.3, 0.0, 1.0)

RING_VEC = __import__('os').environ.get('FSB_RING_VEC', '1') != '0'   # fs-brush-4 11-13: the brush ring projected in one call (_draw_ring)


def _strength_txt(tool, v):
    """fs-shell-4 11-14: 'strength 4.0 mm/s' / 'strength 50 %/s' for the engine's timed brushes (ui/toolreg's
    STRENGTH_SHOW, the strip's own units); 'strength 0.50' for today's brushes, as before"""
    try:
        from . import toolreg as _TRs
        t = _TRs.strength_text(tool, v)
    except Exception:
        t = None
    return ('strength ' + t) if t else ('strength %.2f' % float(v))


def _tool_name(tool):
    """fs-shell-4 11-14: a timed brush's name (Swell / Settle / Flatten), not its engine id ('ft_swell')"""
    try:
        from . import toolreg as _TRs
        return _TRs.FT_NAMES.get(tool, tool)
    except Exception:
        return tool


class Viewport(QWidget):
    selectRequest = Signal(int, bool)        # object id (-1 none), additive
    contextRequest = Signal(int, object)     # hand2f: right-click without a drag: object id (-1 empty), QPointF
    statsChanged = Signal(object)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.cam = Camera(pos=(0.35, 0.25, 0.6), target=(0, 0, 0), fov=35.0)
        self.mode = 'shaded'
        self.xray = 0.0                       # fs-shell 11-13: X-ray 0 (opaque) .. 0.9 (Clay and Rendered; gpu/viewmodes.py)
        self.obj_opacity = {}                 # fs-shell 11-13: {object id: opacity 0-1} (view only, for the X-ray)
        self.quality = 'auto'
        self.res_scale = 1.0
        self.show_mask = False
        self.tool = 'select'
        self.brush = None                     # settings dict (owned by the tool panel)
        self.selected = []
        self.frame = None
        self.image = None
        self.hover = None
        self.drag = None
        self.press_pos = None
        self.last_pos = None
        self.seq = 0
        self.warm = False
        self.warm_msg = 'Compiling the field kernels (first run about 1 minute, then a few seconds)...'
        self.warm_frac = 0.0                  # launch progress 0..1 (Engine.progress)
        self.frame_times = []
        self.objects_info = {}
        self.transforms = {}
        self.tool_mode = None                 # a fieldstudio.tools mode (pencil / cut / splice / fill / layer)
        self.mode_label = ''
        self.mode_path = []                   # the drag in widget pixels (drawn at once; the engine runs the mode)
        # hand1: object transform tools, the 2-D brush cursor, F / Shift+F radius and strength drags, references
        self.mouse_pos = None                 # last mouse position over the view (widget px), for the cursor circle
        self.adjust = None                    # (kind, x0, y0, value0) while F / Shift+F is held
        self.xform = None                     # the transform drag: oid, transform at press, press pos
        self.xform_axis = 'all'               # Scale tool: 'all' or 'x' / 'y' / 'z' (non-uniform: a base stretch)
        self.stretches = {}                   # oid -> [sx, sy, sz] of its 'object scale' node (from snapshots)
        self.shears = {}                      # 11-10: oid -> its 'object scale' node's shear (6 values)
        self.handle_hover = None              # 11-10: the transform handle under the mouse
        from . import gizmo as _GZ             # fs-shell 11-11 (chunk 6): the transform gizmo lives in ui/gizmo.py
        _GZ.install(self)
        self.pressure = 1.0                   # 11-10: the pen's pressure (tabletEvent); 1.0 with a mouse
        self._tablet_t = -1.0
        self.lazy_pt = None                   # 11-10: the stroke stabilizer's brush point (widget px)
        # 11-10 (close work on a lip or a lid: the view turned about a far target and swung the work out of sight):
        # orbit about the surface point under the cursor at the press, and zoom toward the cursor (View menu, saved)
        self.orbit_cursor = True
        self.zoom_cursor = True
        self.orbit_pivot = None
        self.panel = None                     # the ToolPanel (radius / strength changes from the view go through it)
        from ..render.refplanes import RefSet
        self.refs = RefSet([])
        self.ref_tex = {}
        self.ref_sizes = {}                   # name -> (W, H) of the loaded image
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        engine.frameReady.connect(self.on_frame)
        engine.hoverReady.connect(self.on_hover)

    # ---- render requests --------------------------------------------------------------------------------------
    def view(self):
        w = max(16, int(self.width() * self.res_scale))
        h = max(16, int(self.height() * self.res_scale))
        return {'cam': self.cam.copy(), 'w': w, 'h': h, 'mode': self.mode, 'show_mask': self.show_mask,
                'quality': self.quality, 'seq': self.seq, 'selected': list(self.selected),
                'tool': getattr(self, 'tool', None),
                'xray': float(self.xray) if self.mode in ('shaded', 'matcap') else 0.0,
                'opacity': dict(self.obj_opacity) if self.mode in ('shaded', 'matcap') else {},
                'strength': (self.brush or {}).get('strength')}   # fs-brush 11-11: the field tools read it (Pencil depth)

    def request(self, interactive=True):
        self.seq += 1
        self.engine.request_render(self.view(), interactive)

    def to_render(self, pos):
        v = self.view()
        return pos.x() * v['w'] / max(self.width(), 1), pos.y() * v['h'] / max(self.height(), 1), v

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.request(True)

    # ---- frames -------------------------------------------------------------------------------------------------
    def on_frame(self, fr):
        self.frame = fr
        now = time.perf_counter()
        self.frame_times = [t for t in self.frame_times if now - t < 1.0] + [now]
        # fs-brush 11-12 (test 1's window read 'FPS 3' after a Smooth r 50 that ran 14 fps; fs-brush/fps fpc2 on the card:
        # after a 39 fps stroke the readout fell 40 -> 27 -> 13 -> 3 as the light's later frames came in): the readout
        # counted every frame of the last second, so after the release it showed the passes that follow, not the stroke. It
        # now counts the frames that arrive during work (a stroke, an orbit, a pan, a transform drag held; a wheel turn
        # within 0.3 s) and keeps that rate after the release; the window's timer shows '0 (idle)' once nothing has been
        # drawn for a second (main_window.tick)
        if self.drag in ('stroke', 'orbit', 'pan', 'xform', 'mode') or now - getattr(self, '_wheel_t', -1e9) < 0.3:
            self.work_times = [t for t in getattr(self, 'work_times', []) if now - t < 1.0] + [now]
            self.work_fps = len(self.work_times)
        fps = getattr(self, 'work_fps', len(self.frame_times))
        self._compose()
        self.statsChanged.emit({'ms': fr['stats']['ms'], 'fps': fps, 'stats': fr['stats'],
                                'final': fr['final'], 'res': (fr['w'], fr['h'])})

    def set_selected(self, oids):
        self.selected = list(oids)
        if self.frame is not None:
            self._compose()
            # fs-shell-2 11-14 (select inside part 3): the covered outline comes from the card's frame for these ids (the
            # selected objects' own ID image): a new selection asks the card for a frame (the outline above is
            # recomposited from the last one meanwhile)
            if self.frame.get('renderer') == 'gpu' and self.selected and self.frame.get('sel_ids') != self.selected:
                self.request(False)

    def _compose(self):
        fr = self.frame
        rgb = fr['rgb']
        self.hidden_label = None
        for ov in getattr(self, 'overlays', ()):   # fs-shell 11-11: the grids (D2), the cutter's preview (4.18)
            if self.mode == 'silhouette' and 'grid' in type(ov).__name__.lower():
                continue                            # fs-shell 11-14: the silhouette is flat: no floor grid over it
            if ov.active():                         # under the reference planes
                try:
                    rgb = ov.composite(fr, rgb)
                    fr = dict(fr, rgb=rgb)
                except Exception as e:
                    self.ref_error = 'overlay: %s' % e
        if self.refs.planes and self.ref_tex:
            try:
                from ..render import refplanes
                rgb = refplanes.composite(fr, self.refs, self.ref_tex)
            except Exception as e:                # a reference problem must never stop the view
                self.ref_error = str(e)
        # 11-10 (in use on her eyelid: after a stroke the selection outline traced the lid margin and
        # the eye opening right where the brush works): no selection outline while a brush or a field tool is in hand, as
        # Blender's Sculpt mode; Select / Move / Rotate / Scale keep it
        outline = self.tool in ('select', 'move', 'rotate', 'scale') and self.tool_mode is None
        if outline and self.selected and fr['stats']['mode'] != 'slice':
            obj = fr['obj']
            sa = fr.get('sel_alpha')
            if sa is not None and fr.get('sel_ids') == list(self.selected) and sa.shape == rgb.shape[:2]:
                out = np.empty_like(rgb)
                _blend_alpha(np.ascontiguousarray(rgb), sa, SEL_COLOUR, out)   # r39: alpha from the card
                rgb = out
                sel = None
            else:
                sel = np.isin(obj, self.selected)
            if sel is not None and sel.any():
                # hand3: a soft outline. The object-id buffer of a preview frame is upsampled nearest, so a hard
                # 1-px edge traced its stair steps (serrated neck/torso and hip outline, round-2 F12). The mask is
                # box-blurred over the frame's own pixel scale and the edge drawn with coverage alpha.
                ir = fr['stats'].get('internal_res') or (rgb.shape[1], rgb.shape[0])
                sc = max(1, int(round(rgb.shape[1] / max(int(ir[0]), 1))))
                rgb = _draw_soft_edge(rgb, sel, sc, SEL_COLOUR)
            hid = fr.get('sel_hidden')                 # fs-shell-2 11-14: part 3, dashed where something covers it
            if hid is not None and hid.shape == rgb.shape[:2] and fr.get('sel_ids') == list(self.selected):
                rgb = _id_outline(rgb, hid, SEL_COLOUR, visible=False)   # the visible part: the outline above
                self.hidden_label = _cover_label(hid, fr.get('obj'), self.window(), self.selected)
        hm = fr.get('hover_mask')                      # fs-shell-2 11-14: part 4, the Scene list's hovered object
        if hm is not None and hm.shape == rgb.shape[:2] and getattr(self, 'outline_hover', None) is not None:
            rgb = _id_outline(rgb, hm, HOVER_COLOUR, visible=True)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        self.image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def refs_changed(self):
        """hand1: a reference setting changed: recomposite the last frame (no engine render)."""
        if self.frame is not None:
            self._compose()
        self.update()

    def on_hover(self, hv):
        self.hover = hv if hv.get('hit') else None
        self.update()

    # ---- painting -------------------------------------------------------------------------------------------------
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(24, 26, 30))
        if self.image is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(QRectF(self.rect()), self.image)
        if not self.warm:
            self._draw_warm(p)
        self._draw_refs(p)
        self._draw_ring(p)
        self._draw_cursor(p)
        self._draw_tool_badge(p)
        self._draw_mode(p)
        self._draw_handles(p)
        if self.lazy_pt is not None and self.drag == 'stroke' and self.mouse_pos is not None \
                and float((self.brush or {}).get('lazy', 0.0)) > 0:
            p.setPen(QPen(QColor(255, 210, 120, 200), 1.2, Qt.DashLine))
            p.drawLine(self.lazy_pt, self.mouse_pos)
            p.setBrush(QColor(255, 210, 120, 200))
            p.drawEllipse(self.lazy_pt, 2.5, 2.5)
            p.setBrush(Qt.NoBrush)
        self._draw_gizmo(p)
        if getattr(self, 'cues', None):           # fs-shell-2 11-14: the tool's cues (toolset/cues.py, measured)
            from .cues_draw import draw as _draw_cues
            _draw_cues(p, self.cam, self.width(), self.height(), self.cues)
        rm_ = getattr(self, 'reg_mouse', None)       # fs-shell-3 11-14: a gesture's live cues (the blade, the connector)
        if rm_ is not None:
            rm_.draw(p)
        hl = getattr(self, 'hidden_label', None)  # fs-shell-2 11-14: 'Inner · hidden behind Cube' by its outline
        if hl:
            text, fx, fy = hl
            p.setFont(QFont('Segoe UI', 9))
            fm_ = p.fontMetrics()
            r_ = QRectF(fx * self.width() - fm_.horizontalAdvance(text) / 2 - 6, fy * self.height() - fm_.height() - 14,
                        fm_.horizontalAdvance(text) + 12, fm_.height() + 6)
            p.fillRect(r_, QColor(16, 18, 22, 215))
            p.setPen(QColor(int(SEL_COLOUR[0]), int(SEL_COLOUR[1]), int(SEL_COLOUR[2])))
            p.drawText(r_, int(Qt.AlignCenter), text)
        self._draw_info(p)
        p.end()

    def set_warm_progress(self, i, n, label):
        self.warm_steps = getattr(self, 'warm_steps', []) + [label]
        self.warm_frac = i / max(n, 1)
        self.warm_msg = ('Getting the field kernels ready (%d of %d): %s.' '\n' 'The first launch after an update '
                         'compiles them (about 1 minute); later launches take a few seconds.' % (min(i + 1, n), n, label))
        self.update()

    def _draw_warm(self, p):
        r = self.rect()
        w = min(520, r.width() - 40)
        x = r.center().x() - w // 2
        y = r.center().y() - 10
        p.setPen(QColor(220, 225, 232))
        p.setFont(QFont('Segoe UI', 11))
        p.drawText(QRectF(x - 60, y - 70, w + 120, 60), Qt.AlignHCenter | Qt.AlignBottom | Qt.TextWordWrap,
                   self.warm_msg)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(52, 56, 64))
        p.drawRoundedRect(QRectF(x, y, w, 10), 5, 5)
        p.setBrush(QColor(96, 160, 230))
        p.drawRoundedRect(QRectF(x, y, max(10.0, w * self.warm_frac), 10), 5, 5)

    def _draw_ring(self, p):
        if self.hover is None or self.tool == 'select' or self.brush is None:
            return
        c = np.asarray(self.hover['point'], float)
        n = np.asarray(self.hover['normal'], float)
        n = n / max(np.linalg.norm(n), 1e-12)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, a)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        r = float(self.brush['radius'])
        W, H = self.width(), self.height()
        path = QPainterPath()
        if RING_VEC:
            # fs-brush-4 11-13 (M, speed4/gil1: the window's thread spent ~19 ms a frame drawing the brush ring, cursor and
            # handles, holding the Python lock the engine's batch and render need; without them Draw r 20 21.8 -> 29.9 fps):
            # the ring's 65 points projected in one call (tools/regions.project_many, Camera.project's own formulas)
            # instead of 65 Camera.project calls of ~34 us each, on every paint (every mouse move)
            from ..tools.regions import project_many
            ang = 2 * math.pi * np.arange(65) / 64
            Q = c[None] + r * (np.cos(ang)[:, None] * t1[None] + np.sin(ang)[:, None] * t2[None])
            X_, Y_, Z_ = project_many(self.cam, Q, W, H)
            if np.any(Z_ <= 0):
                return
            path.moveTo(float(X_[0]), float(Y_[0]))
            for k in range(1, 65):
                path.lineTo(float(X_[k]), float(Y_[k]))
        else:
            for k in range(65):
                ang = 2 * math.pi * k / 64
                q = c + r * (math.cos(ang) * t1 + math.sin(ang) * t2)
                x, y, z = self.cam.project(q, W, H)
                if z <= 0:
                    return
                if k == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
        tool_col = {'carve': QColor(255, 140, 120), 'mask': QColor(170, 170, 255), 'smooth': QColor(160, 230, 255),
                    'grab': QColor(255, 220, 120), 'stretch': QColor(255, 220, 120)}.get(self.tool,
                                                                                         QColor(140, 190, 255))
        p.setPen(QPen(QColor(0, 0, 0, 120), 3.0))
        p.drawPath(path)
        p.setPen(QPen(tool_col, 1.6))
        p.drawPath(path)
        x, y, z = self.cam.project(c, W, H)
        p.drawLine(QPointF(x - 4, y), QPointF(x + 4, y))
        p.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))
        sym = self.brush.get('symmetry', [False] * 3)
        tr = self.transforms.get(self.hover.get('oid'))
        if any(sym) and tr is not None:
            from ..core.ops import rot_matrix
            R = rot_matrix(tr['rotate'])
            s = max(float(tr['scale']), 1e-6)
            t = np.asarray(tr['translate'], float)
            for ax in range(3):
                if sym[ax]:
                    loc = R.T @ (c - t) / s          # the core mirrors in the object's local frame
                    loc[ax] = -loc[ax]
                    m = R @ (loc * s) + t
                    mx, my, mz = self.cam.project(m, W, H)
                    if mz > 0:
                        p.setPen(QPen(QColor(tool_col.red(), tool_col.green(), tool_col.blue(), 110), 1.0,
                                      Qt.DashLine))
                        p.drawEllipse(QPointF(mx, my), 5, 5)

    def _cam(self):
        return self.frame['cam'] if self.frame is not None else self.cam

    def _draw_refs(self, p):
        """hand1: the reference planes' frames, the vertical line where they cross, and each centre mark."""
        rs = self.refs
        if not rs.planes or not rs.visible or not (rs.show_frame or rs.show_cross):
            return
        from ..render import refplanes
        cam = self._cam()
        W, H = self.width(), self.height()
        cols = {'front': QColor(90, 150, 255, 170), 'side': QColor(110, 220, 120, 170)}
        nvis = 0
        for pl in rs.planes:
            if not pl.visible or pl.name not in self.ref_sizes:
                continue
            nvis += 1
            iw, ih = self.ref_sizes[pl.name]
            corners, centre = refplanes.outline(pl, rs.anchor_mm, iw, ih)
            pts = [cam.project(c, W, H) for c in corners]
            if any(z <= 0 for _, _, z in pts):
                continue
            col = cols.get(pl.name, QColor(200, 200, 200, 170))
            if rs.show_frame:
                p.setPen(QPen(col, 1.2))
                p.setBrush(Qt.NoBrush)
                path = QPainterPath()
                path.moveTo(pts[0][0], pts[0][1])
                for x, y, _ in pts[1:]:
                    path.lineTo(x, y)
                path.closeSubpath()
                p.drawPath(path)
            x, y, z = cam.project(centre, W, H)
            if z > 0 and rs.show_cross:
                p.setPen(QPen(col, 1.6))
                p.drawLine(QPointF(x - 6, y), QPointF(x + 6, y))
                p.drawLine(QPointF(x, y - 6), QPointF(x, y + 6))
        if rs.show_cross and nvis >= 2:
            a = np.asarray(rs.anchor_mm, float) / 1000.0
            x0, y0, z0 = cam.project(a + np.array([0, 0.2, 0]), W, H)
            x1, y1, z1 = cam.project(a - np.array([0, 0.2, 0]), W, H)
            if z0 > 0 and z1 > 0:
                p.setPen(QPen(QColor(255, 230, 120, 140), 1.0, Qt.DashLine))
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    def brush_px(self):
        """hand1: the brush radius in widget pixels at the hover point (or at the view target off the form)."""
        if self.brush is None:
            return 0.0
        cam = self.cam
        if self.hover is not None:
            d = float(np.linalg.norm(np.asarray(self.hover['point'], float) - cam.pos))
        else:
            d = float(np.linalg.norm(cam.target - cam.pos))
        f_px = self.height() / (2.0 * math.tan(math.radians(cam.fov) * 0.5))
        return float(self.brush['radius']) * f_px / max(d, 1e-6)

    # ---- fs-shell-5 11-14 (the three primary settings quick and easy to adjust:
    # radius, strength and softness): while one changes (F / Shift+F / Ctrl+F drag, or a key: [ ] - = { }), the
    # cursor shows it live: the ring at the radius, a dashed ring where the brush acts at half strength (the falloff,
    # which the softness moves), the brush's profile, and the number on a dark plate
    def flash_adjust(self, kind):
        from PySide6.QtCore import QTimer
        self._flash = (kind, time.perf_counter() + 1.3)
        self.update()
        QTimer.singleShot(1350, self.update)

    def _falloff_samples(self):
        b = self.brush or {}
        tool = str(b.get('tool', self.tool) or '')
        key = (tool, b.get('falloff'), repr(b.get('curve'))[:200], round(float(b.get('hardness', 0.0)), 3),
               round(float(b.get('smoothing', 0.0)), 5), round(float(b.get('radius', 0.01)), 5), b.get('foot'))
        c = getattr(self, '_fo_cache', None)
        if c is not None and c[0] == key:
            return c[1]
        w = None
        try:
            if tool.startswith('ft_'):
                from ..warp import profile as PR
                r = float(b.get('radius', 0.01))
                s_ = float(b.get('smoothing', 0.0))
                fam = PR.PEAK_SWELL if tool in ('ft_swell', 'ft_settle') else PR.PEAK_SHAPE
                d, w = PR.profile_curve(r, float(b.get('hardness', 0.0)), s_, b.get('foot', 'soft') or 'soft', fam,
                                        n=65)
                w = np.interp(np.linspace(0.0, d[-1], 65), d, w)
            else:
                from ..core import brush as BR
                w = BR.falloff_lut(b.get('falloff', 'smooth'), b.get('curve'))
        except Exception:
            w = None
        self._fo_cache = (key, w)
        return w

    def _draw_plate(self, p, x0, y0, txt):
        f = QFont('Segoe UI')
        f.setPixelSize(15)
        f.setBold(True)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(txt)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(14, 16, 20, 215))
        p.drawRoundedRect(QRectF(x0 + 14, y0 - 15, tw + 12, 30), 5, 5)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QPointF(x0 + 20, y0 + 5), txt)

    def _draw_readout(self, p, x0, y0, kind):
        if isinstance(kind, tuple):                    # fs-shell-5 11-14: a registry tool's own setting, by name
            from . import quick_settings as QS
            self._draw_plate(p, x0, y0, QS.readout(self.window(), kind[1]))
            return
        rp = self.brush_px()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 140), 3.0))
        p.drawEllipse(QPointF(x0, y0), rp, rp)
        p.setPen(QPen(QColor(255, 255, 255, 235), 1.5))
        p.drawEllipse(QPointF(x0, y0), rp, rp)
        w = self._falloff_samples()
        if w is not None and len(w) > 2:
            ws = np.asarray(w, float)
            half = np.nonzero(ws < 0.5)[0]
            if len(half):
                u = half[0] / float(len(ws) - 1)
                p.setPen(QPen(QColor(255, 196, 90, 230), 1.4, Qt.DashLine))
                p.drawEllipse(QPointF(x0, y0), rp * u, rp * u)
        b = self.brush or {}
        if kind == 'radius':
            txt = 'Radius %.1f mm' % (float(b.get('radius', 0.0)) * 1000)
        elif kind == 'hardness':
            txt = 'Softness / Hardness %d %%' % int(round(float(b.get('hardness', 0.0)) * 100))
        else:
            txt = 'Strength ' + _strength_txt(self.tool, b.get('strength', 0.0))
        f = QFont('Segoe UI')
        f.setPixelSize(15)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        tx = x0 + min(rp, 300) + 12
        tw = fm.horizontalAdvance(txt)
        pw = 64 if w is not None else 0
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(14, 16, 20, 215))
        p.drawRoundedRect(QRectF(tx - 6, y0 - 15, tw + 12 + (pw + 8 if pw else 0), 30), 5, 5)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QPointF(tx, y0 + 5), txt)
        if pw:                                     # the profile across the brush: soft dome to hard edge
            ws = np.asarray(w, float)
            gx, gy, gw, gh = tx + tw + 8, y0 - 10, pw, 20
            path = QPainterPath()
            for i, x in enumerate(np.linspace(-1.0, 1.0, 41)):
                yv = float(ws[min(len(ws) - 1, int(abs(x) * (len(ws) - 1)))])
                px_, py_ = gx + (x + 1) * 0.5 * gw, gy + gh - yv * gh
                if i == 0:
                    path.moveTo(px_, py_)
                else:
                    path.lineTo(px_, py_)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 196, 90), 2.0))
            p.drawPath(path)

    def _draw_cursor(self, p):
        """hand1: a 2-D circle of the brush size that follows the mouse, also off the form, and the F-drag read-out."""
        if self.adjust is not None and self.brush is not None:
            kind, x0, y0, _ = self.adjust
            self._draw_readout(p, x0, y0, kind)
            return
        fl = getattr(self, '_flash', None)          # fs-shell-5 11-14: a key changed it: the same read-out a moment
        if (fl is not None and time.perf_counter() < fl[1] and self.mouse_pos is not None
                and isinstance(fl[0], tuple)):
            self._draw_readout(p, self.mouse_pos.x(), self.mouse_pos.y(), fl[0])
            return
        if (fl is not None and time.perf_counter() < fl[1] and self.brush is not None and self.mouse_pos is not None
                and self.tool not in ('select', 'move', 'rotate', 'scale')):
            self._draw_readout(p, self.mouse_pos.x(), self.mouse_pos.y(), fl[0])
            return
        if (self.mouse_pos is None or self.tool in ('select', 'move', 'rotate', 'scale') or self.brush is None
                or self.tool_mode is not None or self.drag in ('orbit', 'pan')):
            return
        rp = self.brush_px()
        if rp < 2:
            return
        p.setPen(QPen(QColor(255, 255, 255, 70 if self.hover is not None else 130), 1.0,
                      Qt.DotLine if self.hover is not None else Qt.SolidLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(self.mouse_pos, rp, rp)

    TOOL_NAMES = {'draw': 'Draw', 'clay': 'Clay', 'carve': 'Carve', 'inflate': 'Inflate', 'smooth': 'Smooth',
                  'flatten': 'Flatten', 'crease': 'Crease', 'pinch': 'Pinch', 'grab': 'Grab', 'stretch': 'Stretch',
                  'mask': 'Mask', 'select': 'Select', 'move': 'Move object', 'rotate': 'Rotate object',
                  'scale': 'Scale object'}

    def _draw_tool_badge(self, p):
        """V1.3: the active tool, always shown top-left (a field tool shows its own label in _draw_mode)."""
        if self.tool_mode is not None or not self.warm:
            return
        if self.tool == 'select' or self.brush is None:
            text = 'Select  -  click a form to select it, drag to orbit'
            if self.selected and self.tool == 'select':      # fs-shell 11-14: the transform strip sits below
                text = 'Select  -  move, rotate or scale the selection with the buttons below'
            col = QColor(200, 206, 214)
            win_ = self.window()
            if getattr(win_, 'reg_tool', None) is not None:  # fs-shell-3 11-14: a registry tool is in hand: its name
                try:                                         # and the first clause of what to do with it
                    from . import toolreg as _TRh
                    h_ = _TRh.hint(win_)
                    text = '%s  -  %s' % (_TRh.card(win_)[0], h_.split(':')[0].split(';')[0].split('. ')[0].rstrip('.'))
                    col = QColor(255, 210, 120)
                except Exception:
                    pass
        elif self.tool in ('move', 'rotate', 'scale'):
            text = {'move': 'Move object (W)  -  drag an arrow to move along it; the centre or the form: the view plane',
                    'rotate': 'Rotate object (R)  -  drag a ring to turn about its axis; elsewhere: turn / tilt',
                    'scale': 'Scale object (Y)  -  drag a box to stretch one axis (Ctrl: skew); the centre: evenly'}[self.tool]
            if self.tool == 'scale' and self.xform_axis != 'all':
                text += '  -  %s axis only (stretch)' % self.xform_axis.upper()
            rx = getattr(self.window(), 'region_xform', None)       # fs-shell-3 11-14: the mask leaves part free
            if rx is not None and rx.active():
                text = text.replace('Move object', 'Move the unmasked part').replace('Rotate object',
                                                                                    'Rotate the unmasked part')
            col = QColor(255, 210, 120)
        else:
            b = self.brush
            layer = b.get('layer', 'sculpt')
            extra = ' (%s band)' % layer if self.tool == 'smooth' and layer in ('detail', 'shape') else ''
            text = '%s%s  -  radius %.1f mm  -  %s%s' % (
                self.TOOL_NAMES.get(self.tool) or _tool_name(self.tool), extra, float(b['radius']) * 1000,
                _strength_txt(self.tool, b['strength']),
                '  -  mirror ' + ''.join('XYZ'[i] for i in range(3) if b.get('symmetry', [0] * 3)[i])
                if any(b.get('symmetry', [])) else '')
            col = QColor(140, 190, 255)
        p.setFont(QFont('Segoe UI', 10, QFont.DemiBold))
        fm = p.fontMetrics()
        r = QRectF(12, 10, min(self.width() - 140, fm.horizontalAdvance(text) + 24), 26)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(20, 22, 26, 200))
        p.drawRoundedRect(r, 5, 5)
        p.setBrush(Qt.NoBrush)
        p.setPen(col)
        p.drawText(r.adjusted(10, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, text)

    def _draw_mode(self, p):
        m = self.tool_mode
        if m is None:
            return
        col = QColor(255, 200, 90)
        if len(self.mode_path) >= 2:
            path = QPainterPath()
            path.moveTo(self.mode_path[0])
            for q in self.mode_path[1:]:
                path.lineTo(q)
            if type(m).__name__ == 'CutMode':
                path.closeSubpath()
            p.setPen(QPen(QColor(0, 0, 0, 140), 3.4))
            p.drawPath(path)
            p.setPen(QPen(col, 1.6, Qt.DashLine if type(m).__name__ == 'CutMode' else Qt.SolidLine))
            p.drawPath(path)
        first = getattr(m, 'first', None)
        if first is not None and getattr(first, 'hit', False):
            x, y, z = self.cam.project(np.asarray(first.point, float), self.width(), self.height())
            if z > 0:
                p.setPen(QPen(col, 2.0))
                p.drawEllipse(QPointF(x, y), 7, 7)
                p.drawText(QPointF(x + 10, y - 8), 'A')
        if self.mode_label:
            p.setFont(QFont('Segoe UI', 10, QFont.DemiBold))
            r = QRectF(12, 10, min(self.width() - 140, 9 * len(self.mode_label) + 24), 26)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(20, 22, 26, 200))
            p.drawRoundedRect(r, 5, 5)
            p.setBrush(Qt.NoBrush)
            p.setPen(col)
            p.drawText(r.adjusted(10, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, self.mode_label)

    # tool-mode events (the mouse and the self-test use the same calls)
    def _mode_post(self, method, pos):
        if self.tool_mode is None:             # V1.3: the tool was left (Esc) during its drag: drop the event
            return
        x, y, v = self.to_render(pos)
        self.engine.post('mode', self.tool_mode, method, v, x, y)

    def mode_press(self, pos):
        self.mode_path = [QPointF(pos)]
        self._mode_post('on_press', pos)
        self.update()

    def mode_move(self, pos):
        last = self.mode_path[-1] if self.mode_path else None
        if last is None or (pos.x() - last.x()) ** 2 + (pos.y() - last.y()) ** 2 >= 4:
            self.mode_path.append(QPointF(pos))
        self._mode_post('on_drag', pos)
        self.update()

    def mode_release(self, pos):
        self._mode_post('on_release', pos)
        self.mode_path = []
        self.update()

    def _draw_gizmo(self, p):
        f, r, u = self.cam.basis()
        cx, cy, L = self.width() - 60, 62, 34
        axes = [('X', np.array([1, 0, 0]), QColor(235, 80, 80)), ('Y', np.array([0, 1, 0]), QColor(110, 210, 90)),
                ('Z', np.array([0, 0, 1]), QColor(80, 140, 245))]
        axes.sort(key=lambda a: float(a[1] @ f), reverse=True)          # far axes first
        p.setFont(QFont('Segoe UI', 9, QFont.DemiBold))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 18))
        p.drawEllipse(QPointF(cx, cy), L + 12, L + 12)
        for name, a, col in axes:
            dx, dy = float(a @ r) * L, -float(a @ u) * L
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 90), 2))
            p.drawLine(QPointF(cx, cy), QPointF(cx - dx * 0.6, cy - dy * 0.6))
            p.setPen(QPen(col, 2.4))
            p.drawLine(QPointF(cx, cy), QPointF(cx + dx, cy + dy))
            p.setBrush(col)
            p.drawEllipse(QPointF(cx + dx, cy + dy), 3.5, 3.5)
            p.setPen(col)
            p.drawText(QPointF(cx + dx * 1.3 - 4, cy + dy * 1.3 + 4), name)
        p.setBrush(Qt.NoBrush)

    # ---- 11-10 (the primitive shapes lacked an intuitive way to stretch, shape, sculpt
    # form or skew): handles on the selected object, in its own axes, as Blender's transform gizmos.
    # Move: an arrow per axis moves along it. Scale: a box per axis stretches that axis (Ctrl: skews it toward the
    # other axis the drag runs along); the centre disc scales evenly. Rotate: a ring per axis turns about it.
    # Away from the handles the drag works as before (move in the view plane, turn, scale evenly).
    HANDLE_PX = 90
    AXIS_COL = (QColor(235, 80, 80), QColor(110, 210, 90), QColor(80, 140, 245))

    def _handles(self):
        return self.gizmo.geo()

    def _handles_pre_gizmo(self):
        if self.tool not in ('move', 'rotate', 'scale') or not self.selected or self.drag not in (None, 'xform'):
            return None
        x0 = self.xform
        if x0 is not None and x0.get('gz') is not None:
            return x0['gz']['geo']
        oid = self.selected[0]
        tr = self.transforms.get(oid)
        if tr is None:
            return None
        from ..core.ops import rot_matrix
        W, H = max(self.width(), 1), max(self.height(), 1)
        c = np.asarray(tr['translate'], float)
        R = rot_matrix(tr['rotate'])
        sx, sy, z = self.cam.project(c, W, H)
        if z <= 1e-6:
            return None
        k = self.HANDLE_PX * 2.0 * z * math.tan(math.radians(self.cam.fov) * 0.5) / H   # world length of a handle
        axes = []
        for i in range(3):
            ex, ey, _ = self.cam.project(c + R[:, i] * k, W, H)
            d = np.array([ex - sx, ey - sy])
            L = float(np.linalg.norm(d))
            axes.append((d / L if L > 1e-6 else np.zeros(2), L))
        rings = []
        if self.tool == 'rotate':
            for i in range(3):
                j, l = (i + 1) % 3, (i + 2) % 3
                pts = [self.cam.project(c + 0.85 * k * (math.cos(a) * R[:, j] + math.sin(a) * R[:, l]), W, H)[:2]
                       for a in np.linspace(0.0, 2 * math.pi, 49)]
                rings.append(np.array(pts))
        return {'oid': oid, 'c': c, 'R': R, 'k': k, 'sc': np.array([sx, sy]), 'axes': axes, 'rings': rings}

    def _handle_at(self, pos):
        return self.gizmo.hit(pos)

    def _handle_at_pre_gizmo(self, pos):
        g = self._handles()
        if g is None:
            return None
        q = np.array([pos.x(), pos.y()])
        if np.linalg.norm(q - g['sc']) <= 10:
            return ('centre', -1)
        if self.tool == 'rotate':
            best = None
            f = self.cam.basis()[0]
            for i, ring in enumerate(g['rings']):
                d = float(np.min(np.linalg.norm(ring - q, axis=1)))
                score = d + 4.0 * (1.0 - abs(float(g['R'][:, i] @ f)))     # where rings cross: the one facing you
                if d <= 7 and (best is None or score < best[0]):
                    best = (score, i)
            return ('ring', best[1]) if best else None
        best = None
        for i, (dv, L) in enumerate(g['axes']):
            if L < 8:
                continue
            v = q - g['sc']
            t = float(v @ dv)
            if not (0.2 * L <= t <= L + 9):
                continue
            d = float(np.linalg.norm(v - t * dv))
            if d <= 8 and (best is None or d < best[0]):
                best = (d, i)
        return ('axis', best[1]) if best else None

    def _draw_handles(self, p):
        self.gizmo.draw(p)

    def _draw_handles_pre_gizmo(self, p):
        g = self._handles()
        if g is None:
            return
        hov = self.handle_hover
        act = self.xform['gz']['hit'] if (self.xform is not None and self.xform.get('gz')) else None
        sc = QPointF(float(g['sc'][0]), float(g['sc'][1]))
        f = self.cam.basis()[0]
        order = sorted(range(3), key=lambda i: -float(g['R'][:, i] @ f))    # far axes first
        for i in order:
            col = QColor(self.AXIS_COL[i])
            lit = hov in (('axis', i), ('ring', i)) or act in (('axis', i), ('ring', i))
            if lit:
                col = col.lighter(150)
            if self.tool == 'rotate':
                ring = g['rings'][i]
                path = QPainterPath(QPointF(float(ring[0][0]), float(ring[0][1])))
                for pt in ring[1:]:
                    path.lineTo(QPointF(float(pt[0]), float(pt[1])))
                p.setPen(QPen(col, 3.2 if lit else 2.0))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
                continue
            dv, L = g['axes'][i]
            if L < 8:
                continue
            tp = g['sc'] + dv * L
            tip = QPointF(float(tp[0]), float(tp[1]))
            p.setPen(QPen(col, 3.0 if lit else 2.0))
            p.drawLine(sc, tip)
            p.setBrush(col)
            p.setPen(Qt.NoPen)
            if self.tool == 'move':
                n = np.array([-dv[1], dv[0]])
                a = g['sc'] + dv * (L + 10)
                b = g['sc'] + dv * (L - 4)
                path = QPainterPath(QPointF(float(a[0]), float(a[1])))
                path.lineTo(QPointF(float(b[0] + n[0] * 5), float(b[1] + n[1] * 5)))
                path.lineTo(QPointF(float(b[0] - n[0] * 5), float(b[1] - n[1] * 5)))
                path.closeSubpath()
                p.drawPath(path)
            else:
                p.drawRect(QRectF(tip.x() - 5, tip.y() - 5, 10, 10))
        p.setPen(QPen(QColor(235, 235, 235, 220), 2.0))
        p.setBrush(QColor(255, 255, 255, 60) if hov == ('centre', -1) else Qt.NoBrush)
        p.drawEllipse(sc, 7, 7)
        p.setBrush(Qt.NoBrush)

    def _xform_handle(self, x, pos, mods):
        """the job for a handle drag (the press's geometry, kept for the whole drag)"""
        from ..core.ops import rot_matrix
        from .engine import euler_from_matrix
        gz = x['gz']
        g = gz['geo']
        kind, i = gz['hit']
        d = np.array([pos.x() - x['p0'].x(), pos.y() - x['p0'].y()])
        t0 = x['t0']
        if kind == 'ring':
            v = np.array([pos.x(), pos.y()]) - g['sc']
            ang = math.atan2(v[1], v[0]) - gz['a0']
            ax = g['R'][:, i]
            f = x['cam'].basis()[0]
            # screen angles run clockwise (y down); seen from the axis tip a positive turn runs anticlockwise
            ang = ang if float(ax @ f) > 0 else -ang
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rr = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
            Rn = Rr @ rot_matrix(t0['rotate'])
            return ('props', x['oid'], {'transform': {'rotate': [round(float(v_), 4) for v_ in euler_from_matrix(Rn)]}},
                    x['drag'])
        dv, L = g['axes'][i]
        s_px = float(d @ dv)
        if self.tool == 'move':
            w = s_px / max(L, 1e-6) * g['k']
            t = np.asarray(t0['translate'], float) + g['R'][:, i] * w
            return ('props', x['oid'], {'transform': {'translate': [round(float(v_), 6) for v_ in t]}}, x['drag'])
        st = list(x['stretch0'])
        sh = list(x.get('shear0') or [0.0] * 6)
        if mods & Qt.ControlModifier:
            # skew: the axis-i tip slides toward the other axis the drag runs along most
            cand = [a for a in range(3) if a != i and g['axes'][a][1] >= 8]    # an axis seen end-on cannot be dragged along
            if not cand:
                return None
            j = max(cand, key=lambda a: abs(float(d @ g['axes'][a][0])))
            dj, Lj = g['axes'][j]
            amt = float(d @ dj) / max(Lj, 1e-6)          # in handle lengths
            slot = {(0, 1): 0, (0, 2): 1, (1, 0): 2, (1, 2): 3, (2, 0): 4, (2, 1): 5}[(j, i)]
            sh[slot] = round(max(min(sh[slot] + amt, 3.0), -3.0), 4)
            return ('oscale', x['oid'], st, x['drag'], sh)
        gfac = math.exp(s_px / 150.0)
        st[i] = round(max(st[i] * gfac, 0.05), 5)
        return ('oscale', x['oid'], st, x['drag'], sh if any(sh) else None)

    def _draw_info(self, p):
        if self.frame is None:
            return
        # fs-shell-2 11-14 (design rule 3: Lipschitz, bricks and the like only on an Inspect page): in the approved
        # layout these show in the inspect views (field slice, normals, curvature) or with View > Debug views >
        # 'Engine numbers in the view'; FS_LAYOUT=classic keeps them always
        on = getattr(self, 'engine_info', None)
        if on is None:
            on = __import__('os').environ.get('FS_LAYOUT', '') == 'classic'
        if not on and self.mode not in ('slice', 'normal', 'curvature'):
            return
        st = self.frame['stats']
        info = self.objects_info
        lines = [('Objects', '%d' % info.get('objects', 0)),
                 ('Base nodes', '%d' % info.get('nodes', 0)),
                 ('Sculpt bricks', '%d (8³ at %.1f mm)' % (info.get('bricks', 0), info.get('h', 0.5))),
                 ('Lipschitz L', '%.3f' % st.get('lipschitz', 1.0)),
                 ('Rep', 'Implicit field F(X) = 0 (no mesh)'),
                 ('Frame', '%dx%d q%d %s, %.0f ms, %s unresolved' % (
                     st['internal_res'][0], st['internal_res'][1], st['quality'], st['mode'], st['ms'],
                     st.get('unresolved', 0)))]
        p.setFont(QFont('Segoe UI', 8))
        y = self.height() - 12 - 15 * len(lines)
        # hand2b (verifier item 12): a dark translucent backing, so the light text reads over a light model too
        fm = p.fontMetrics()
        wmax = max(98 - 14 + fm.horizontalAdvance(v) for _, v in lines)
        p.fillRect(QRectF(8, y - 13, wmax + 14, 15 * len(lines) + 6), QColor(16, 18, 22, 170))
        for k, v in lines:
            p.setPen(QColor(150, 158, 168))
            p.drawText(QPointF(14, y), k + ':')
            p.setPen(QColor(220, 225, 232))
            p.drawText(QPointF(98, y), v)
            y += 15

    # ---- input ------------------------------------------------------------------------------------------------------
    def _obj_at(self, pos):
        fr = self.frame
        if fr is None:
            return -1
        obj = fr['obj']
        h, w = obj.shape
        x = int(pos.x() * w / max(self.width(), 1))
        y = int(pos.y() * h / max(self.height(), 1))
        if 0 <= x < w and 0 <= y < h:
            return int(obj[y, x])
        return -1

    def mousePressEvent(self, e):
        self.setFocus()
        if self.drag is not None:              # V1.3: a second button during a drag is ignored (no lost stroke end)
            return
        self.drag_button = e.button()
        self.press_pos = e.position()
        self._pivot_set, self.orbit_pivot = False, None
        self.last_pos = e.position()
        mods = e.modifiers()
        if self.adjust is not None and e.button() == Qt.LeftButton:
            self.end_adjust()                  # hand1: a click ends an F / Shift+F drag
            self.drag = None
            return
        if e.button() == Qt.LeftButton:
            if self.gizmo.picking_pivot and not (mods & Qt.AltModifier):
                self.gizmo.pick_pivot(e.position())          # fs-shell 11-11: Pivot > Pick a surface point
                self.drag = None
                return
            rm = getattr(self, 'reg_mouse', None)            # fs-shell-3 11-14: a registry tool acting at the pointer
            if rm is not None and not (mods & Qt.AltModifier):
                took = rm.press(e.position(), mods)
                if took:
                    self.drag = 'reg' if took is True else None
                    return
            if mods & Qt.AltModifier:
                self.drag = 'orbit'
            elif self.tool_mode is not None:
                self.drag = 'mode'
                self.mode_press(e.position())
            elif self.tool == 'select':
                self.drag = 'click-add' if mods & (Qt.ShiftModifier | Qt.ControlModifier) else 'click'
            elif self.tool in ('move', 'rotate', 'scale'):
                self.drag = 'xform' if self.begin_xform(e.position()) else 'orbit'
                if self.xform is not None:
                    self.xform['mods'] = mods          # 11-10: Ctrl at the press skews (Scale handles)
            elif self.brush is not None:
                # fs-brush 11-11 (design C5; 22 of 166 presses near the silhouette orbited in test 1): with a brush in
                # hand a press never orbits. The engine starts the stroke on the form under the press, or on the nearest
                # surface within the brush's radius of the ray, or (Blender) when the drag first reaches the form.
                # Orbit: Alt + left drag or the middle button.
                self.drag = 'stroke'
                self.begin_stroke(e.position(), bool(mods & Qt.ControlModifier), bool(mods & Qt.ShiftModifier))
            else:
                self.drag = 'orbit'
        elif e.button() == Qt.RightButton:
            self.drag = 'rclick'               # hand2f: a click opens the action menu; 4 px of travel makes it a pan
        elif e.button() == Qt.MiddleButton:
            self.drag = 'pan' if mods & Qt.ShiftModifier else 'orbit'    # fs-brush 11-11: as Blender (MMB orbits)

    # ---- 11-10 (the first finding: not yet as refined as Blender): pen pressure and a stroke stabilizer,
    # two things a sculptor reaches for in fine work. A tablet's pressure scales each dab's strength (recorded in the
    # dab, so replay is exact); with a mouse it stays 1. The stabilizer (brush panel, px; 0 = off, as Blender's
    # Stabilize Stroke) keeps the brush on a string behind the pointer: it moves only when the string is pulled taut,
    # so a hand's tremor in a lid crease or a lip line does not reach the surface.
    def tabletEvent(self, e):
        self.pressure = max(0.05, min(float(e.pressure()), 1.0))
        self._tablet_t = time.perf_counter()
        e.ignore()                             # Qt then sends the matching mouse event, which drives the stroke

    def _stroke_view(self, pos):
        x, y, v = self.to_render(pos)
        if time.perf_counter() - self._tablet_t < 0.25:
            v['pressure'] = round(self.pressure, 3)
        t0 = getattr(self, '_stroke_t0', None)        # fs-shell-3 11-14: seconds since the press,
        if t0 is not None:                            # on every sample (a held tool's rate x time reads it)
            v['t'] = round(time.perf_counter() - t0, 5)
        return x, y, v

    # ---- fs-shell-3 11-14 (decided; fs-tools' BRUSH_ENGINE: while held, stroke_move every
    # ~1/120 s with 't'): a held tool keeps acting while the cursor is still, so the window sends a timed sample at
    # ~120 Hz during its stroke, at the brush's place now (the stabilizer's point when it is on: a slack string never
    # stops them). Only for the brushes the engine times (core/brush.FT_TOOLS: ft_swell / ft_settle / ft_flatten since
    # ee9e0f9b; one dab per sample covering the time since the last): today's brushes never get a still sample, so their
    # strokes are as they were (t58/t_ab.py). engine.timed_samples = False switches them off (a harness's A/B).
    HELD_HZ = 120.0

    def _held_tool(self):
        if getattr(self.engine, 'timed_samples', True) is False:
            return False
        tool = (self.brush or {}).get('tool') or getattr(self, 'tool', None)
        try:
            from ..core import brush as _BRft
            return tool in getattr(_BRft, 'FT_TOOLS', ())
        except Exception:
            return False

    def _held_start(self):
        if not self._held_tool():
            return
        from PySide6.QtCore import QTimer
        tm = getattr(self, '_held_timer', None)
        if tm is None:
            tm = self._held_timer = QTimer(self)
            tm.setTimerType(Qt.PreciseTimer)
            tm.timeout.connect(self._held_tick)
        tm.start(max(1, int(round(1000.0 / self.HELD_HZ))))

    def _held_stop(self):
        tm = getattr(self, '_held_timer', None)
        if tm is not None:
            tm.stop()

    def _held_tick(self):
        if self.drag != 'stroke' or getattr(self, '_stroke_t0', None) is None:
            self._held_stop()
            return
        R = float((self.brush or {}).get('lazy', 0.0))
        q = self.lazy_pt if (R > 0 and self.lazy_pt is not None) else getattr(self, '_stroke_last', None)
        if q is None:
            return
        x, y, v = self._stroke_view(q)
        self.held_samples = getattr(self, 'held_samples', 0) + 1
        self.engine.post('stroke_move', v, x, y)

    def _lazy_pos(self, pos):
        R = float((self.brush or {}).get('lazy', 0.0))
        if R <= 0 or self.lazy_pt is None:
            return pos
        dx, dy = pos.x() - self.lazy_pt.x(), pos.y() - self.lazy_pt.y()
        L = math.hypot(dx, dy)
        if L <= R:
            return None                        # the string is slack: the brush stays
        k = (L - R) / L
        self.lazy_pt = QPointF(self.lazy_pt.x() + dx * k, self.lazy_pt.y() + dy * k)
        return QPointF(self.lazy_pt)

    def begin_stroke(self, pos, invert=False, smooth=False):
        self.lazy_pt = QPointF(pos)
        self._stroke_t0 = None                       # fs-shell-3 11-14: the press's view as before (no 't'); the
        self._stroke_last = QPointF(pos)             # moves after it carry 't', seconds since this press
        s = dict(self.brush)
        s['symmetry'] = list(s.get('symmetry', [False] * 3))
        if smooth and s['tool'] != 'mask':     # hand1: Shift held = Smooth for this stroke (as in ZBrush / Blender)
            s['tool'] = 'smooth'
            s['layer'] = 'sculpt'
            invert = False
        if invert:
            if s['tool'] == 'draw':
                s['tool'] = 'carve'
            elif s['tool'] == 'carve':
                s['tool'] = 'draw'
            elif s['tool'] == 'mask':
                s['strength'] = -abs(s['strength'])
            else:
                s['invert'] = True             # fs-brush 11-11: every other brush has its inverse (brush.invert_dab)
        x, y, v = self._stroke_view(pos)
        self.engine.post('stroke_begin', v, x, y, s)
        self._stroke_t0 = time.perf_counter()
        self._held_start()                           # fs-shell-3 11-14: a held tool's timed samples (engine permitting)

    def move_stroke(self, pos):
        q = self._lazy_pos(pos)
        if q is None:
            self.update()                      # the stabilizer's string is drawn; the brush has not moved
            return
        self._stroke_last = QPointF(q)
        x, y, v = self._stroke_view(q)
        self.engine.post('stroke_move', v, x, y)
        if getattr(self, 'tool', None) == 'mask':
            self.request(True)        # 11-08 night: mask paint changes no surface, so no frame came to show it

    def end_stroke(self):
        self._held_stop()                            # fs-shell-3 11-14: no timed sample after the release
        self._stroke_t0 = None
        self.lazy_pt = None
        self.engine.post('stroke_end')
        if getattr(self, 'tool', None) == 'mask':
            self.request(False)

    # ---- hand1: object transform drags (Move W / Rotate R / Scale Y) ------------------------------------------------
    def begin_xform(self, pos):
        oid = self._obj_at(pos)
        if self.selected and self._handle_at(pos) is not None:
            oid = self.selected[0]                 # 11-10: a handle belongs to the selected object
        if oid < 0 and self.selected:
            oid = self.selected[0]
        tr = self.transforms.get(oid)
        if oid < 0 or tr is None:
            return False
        import copy
        self.xform_seq = getattr(self, 'xform_seq', 0) + 1      # hand1b: one drag = one undo step (engine merge id)
        self.xform = {'oid': oid, 't0': copy.deepcopy(tr), 'p0': QPointF(pos), 'cam': self.cam.copy(),
                      'drag': self.xform_seq, 'gz': None,
                      'stretch0': list(self.stretches.get(oid, [1.0, 1.0, 1.0])),
                      'shear0': list(self.shears.get(oid, [0.0] * 6))}
        self.gizmo.begin(self.xform, pos)           # fs-shell 11-11: the handle (or deformer) hit and its geometry
        # fs-shell-6 11-14 (the eye workflow: select both, drag them back together): the rest of the selection
        # (or, on an object not selected, the rest of its group) goes with the drag
        groups = getattr(self, 'groups', None) or {}
        if oid in self.selected:
            party = [o for o in self.selected if o != oid]
        else:
            g = groups.get(oid)
            party = [o for o, g2 in groups.items() if g is not None and g2 == g and o != oid]
        self.xform['party'] = [(o, copy.deepcopy(self.transforms[o])) for o in party if o in self.transforms]
        if oid not in self.selected:
            self.selectRequest.emit(oid, False)
        return True

    def with_party(self, x, job):
        """fs-shell-6 11-14: a drag of one object of a selection moves the whole selection as one: the dragged one's change
        (old -> new transform: turn R, scale k, move) is applied to each member about the same point, so they keep
        their places relative to each other; -> ('props_many', [(oid, props), ...], drag). A stretch along one axis
        (oscale), a deformer or a mass's own transform stays the dragged one's alone"""
        party = x.get('party') or []
        if not party or job is None or job[0] != 'props' or x.get('mass') or 'transform' not in job[2]:
            return job
        from ..core.ops import rot_matrix
        from .engine import euler_from_matrix
        t0 = x['t0']
        new = dict(t0)
        new.update(job[2]['transform'])
        R0, R1 = rot_matrix(t0.get('rotate', [0, 0, 0])), rot_matrix(new.get('rotate', [0, 0, 0]))
        s0, s1 = float(t0.get('scale', 1.0) or 1.0), float(new.get('scale', 1.0) or 1.0)
        c0, c1 = np.asarray(t0['translate'], float), np.asarray(new['translate'], float)
        dR, k = R1 @ R0.T, s1 / max(s0, 1e-12)
        items = [(x['oid'], job[2])]
        for oid, tm in party:
            out = {'translate': [round(float(v), 6) for v in c1 + k * (dR @ (np.asarray(tm['translate'], float) - c0))]}
            if 'rotate' in job[2]['transform']:
                out['rotate'] = [round(float(v), 4) for v in euler_from_matrix(dR @ rot_matrix(tm.get('rotate', [0, 0, 0])))]
            if 'scale' in job[2]['transform']:
                out['scale'] = round(max(float(tm.get('scale', 1.0) or 1.0) * k, 1e-3), 6)
            items.append((oid, {'transform': out}))
        return ('props_many', items, job[3] if len(job) > 3 else None)

    def xform_at(self, pos):
        """-> the job for the drag so far: ('props', oid, {'transform': ...}) or ('oscale', oid, [sx, sy, sz])."""
        x = self.xform
        if x is None:
            return None
        # fs-shell 11-11 (chunk 6): every transform drag's job comes from the gizmo (handles, deformers, the typed value,
        # X / Y / Z lock, Ctrl snapping, the pivot); the code below is the pre-gizmo free drag, kept for reference
        return self.gizmo.job(x, pos, getattr(self, '_mods', Qt.NoModifier) | x.get('mods', Qt.NoModifier))
        from ..core.ops import rot_matrix
        from .engine import euler_from_matrix
        cam = x['cam']
        f, r, u = cam.basis()
        t0 = x['t0']
        dx = pos.x() - x['p0'].x()
        dy = pos.y() - x['p0'].y()
        c = np.asarray(t0['translate'], float)
        if self.tool == 'move':
            d = float(np.linalg.norm(c - cam.pos))
            k = 2.0 * d * math.tan(math.radians(cam.fov) * 0.5) / max(self.height(), 1)
            t = c + r * dx * k - u * dy * k
            return ('props', x['oid'], {'transform': {'translate': [round(float(v), 6) for v in t]}}, x['drag'])
        if self.tool == 'rotate':
            a = math.radians(dx * 0.5)
            b = math.radians(dy * 0.5)
            Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            ax = r / max(np.linalg.norm(r), 1e-12)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rb = np.eye(3) + math.sin(b) * K + (1 - math.cos(b)) * (K @ K)
            R = Rb @ Ry @ rot_matrix(t0['rotate'])
            return ('props', x['oid'], {'transform': {'rotate': [round(float(v), 4) for v in euler_from_matrix(R)]}},
                    x['drag'])
        g = math.exp((dx - dy) / 300.0)
        if self.xform_axis == 'all':
            return ('props', x['oid'], {'transform': {'scale': round(max(float(t0['scale']) * g, 1e-3), 6)}}, x['drag'])
        st = list(x['stretch0'])
        i = 'xyz'.index(self.xform_axis)
        st[i] = round(max(st[i] * g, 0.05), 5)
        return ('oscale', x['oid'], st, x['drag'])

    def mouseMoveEvent(self, e):
        pos = e.position()
        self.mouse_pos = QPointF(pos)
        self._mods = e.modifiers()
        if self.drag is None and self.tool in ('move', 'rotate', 'scale'):
            hv = self._handle_at(pos)
            if hv != self.handle_hover:
                self.handle_hover = hv
                self.update()
        if self.adjust is not None:
            self.adjust_move(pos)
            return
        if self.drag == 'xform':
            job = self.xform_at(pos)
            if job is not None:
                self.engine.post(*job)
            self.update()
            return
        if self.drag == 'mode':
            self.mode_move(pos)
            return
        if self.drag == 'reg':
            self.reg_mouse.move(pos)                # fs-shell-3 11-14: a registry tool's drag
            self.update()
            return
        if self.drag is None:
            rm = getattr(self, 'reg_mouse', None)
            if rm is not None and rm.hover(pos):     # fs-shell-3 11-14: Place's picked-up object follows the cursor
                self.update()
                return
            if self.tool not in ('select', 'move', 'rotate', 'scale') and self.warm and self.tool_mode is None:
                x, y, v = self.to_render(pos)
                self.engine.request_hover(v, x, y)
            self.update()
            return
        dx = pos.x() - self.last_pos.x()
        dy = pos.y() - self.last_pos.y()
        self.last_pos = pos
        if self.drag in ('click', 'click-add'):
            if abs(pos.x() - self.press_pos.x()) + abs(pos.y() - self.press_pos.y()) > 4:
                self.drag = 'orbit'
            else:
                return
        if self.drag == 'rclick':
            if abs(pos.x() - self.press_pos.x()) + abs(pos.y() - self.press_pos.y()) > 4:
                self.drag = 'pan'
            else:
                return
        if self.drag == 'orbit':
            if not getattr(self, '_pivot_set', False):      # 11-10: the pivot, once per orbit drag, at the press
                self._pivot_set = True
                self.orbit_pivot = self.surface_at(self.press_pos) if (self.orbit_cursor and self.press_pos is not None) else None
            if self.orbit_pivot is not None:
                self.cam.orbit_about(self.orbit_pivot, dx * 0.4, dy * 0.4)
            else:
                self.cam.orbit(dx * 0.4, dy * 0.4)
            self.request(True)
        elif self.drag == 'pan':
            self.cam.pan(dx / max(self.height(), 1), dy / max(self.height(), 1))
            self.request(True)
        elif self.drag == 'stroke':
            self.move_stroke(pos)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.drag is not None and e.button() != getattr(self, 'drag_button', e.button()):
            return                             # V1.3: only the button that started the drag ends it
        d = self.drag
        self.drag = None
        if d == 'xform':
            self.gizmo.finish(self.xform)          # fs-shell-3 11-14: a masked region's drag posts its one move here
            self.xform = None
            self.request(False)
        elif d == 'mode':
            self.mode_release(e.position())
        elif d == 'reg':
            self.reg_mouse.release(e.position())    # fs-shell-3 11-14: a registry tool's release
        elif d == 'stroke':
            self.end_stroke()
        elif d in ('click', 'click-add'):
            si = getattr(self.window(), 'select_inside', None)   # fs-shell 11-14: click again to go deeper (A+click: back)
            oid = si.click(e.position()) if (si is not None and d == 'click') else self._obj_at(e.position())
            self.selectRequest.emit(oid, d == 'click-add')
        elif d == 'rclick':
            self.contextRequest.emit(self._obj_at(e.position()), QPointF(e.position()))
        elif d in ('orbit', 'pan'):
            self.request(False)

    def contextMenuEvent(self, e):
        """hand2f: the Menu key / Shift+F10 open the action menu at the mouse (the mouse button opens it on release
        in mouseReleaseEvent, so a right-drag can still pan); a mouse-reason event is ignored here."""
        from PySide6.QtGui import QContextMenuEvent
        if e.reason() == QContextMenuEvent.Mouse:
            e.accept()
            return
        pos = self.mouse_pos or QPointF(self.width() / 2, self.height() / 2)
        if self.drag is None:
            self.contextRequest.emit(self._obj_at(pos), QPointF(pos))
        e.accept()

    def surface_at(self, pos):
        """11-10: the world point the last frame shows under pos (its depth along the pixel's ray), or None"""
        fr = self.frame
        if fr is None or fr.get('depth') is None:
            return None
        d = fr['depth']
        h, w = d.shape[:2]
        x = min(max(int(pos.x() * w / max(self.width(), 1)), 0), w - 1)
        y = min(max(int(pos.y() * h / max(self.height(), 1)), 0), h - 1)
        t = float(d[y, x])
        if not np.isfinite(t) or t <= 0 or t > 1e29:
            return None
        o, dr = self.cam.ray(x, y, w, h)
        return o + dr * t

    def wheelEvent(self, e):
        self._wheel_t = time.perf_counter()        # fs-brush 11-12: a wheel turn counts as work for the FPS readout
        steps = e.angleDelta().y() / 120.0
        rm = getattr(self, 'reg_mouse', None)
        if rm is not None and rm.wheel(steps):     # fs-shell-3 11-14: the wheel spins an object being placed
            return
        P = self.surface_at(e.position()) if self.zoom_cursor else None
        if P is not None:
            self.cam.zoom_about(P, 0.88 ** steps)
        else:
            self.cam.zoom(0.88 ** steps)
        if self.brush is not None and self.brush.get('radius_follow') and self.panel is not None:
            # 11-10: the brush keeps its size on the screen (the distance to what is in view scaled by 0.88^steps)
            self.panel.set_radius(float(self.brush['radius']) * 0.88 ** steps)
        self.request(True)

    def leaveEvent(self, e):
        self.hover = None
        self.mouse_pos = None
        self.update()

    def enterEvent(self, e):
        # hand1: the view takes the keyboard when the mouse enters it, unless a text field (not a number box) has it
        from PySide6.QtWidgets import QApplication, QLineEdit, QAbstractSpinBox
        fw = QApplication.focusWidget()
        typing = isinstance(fw, QLineEdit) and not isinstance(fw.parent(), QAbstractSpinBox)
        if not typing and self.window().isActiveWindow():
            self.setFocus()
        super().enterEvent(e)

    # ---- hand1: F-drag radius, Shift+F-drag strength ----------------------------------------------------------
    def begin_adjust(self, kind):
        # fs-shell-5 11-14 (any tool): a registry tool that is not a brush drags its own settings
        try:
            from . import quick_settings as QS
            win_ = self.window()
            if self.drag is None and QS.own_slots(win_) is not None:
                slot = {'radius': 0, 'strength': 1, 'hardness': 2}.get(kind, 0)
                v0 = QS.slot_value(win_, slot)
                if v0 is None:
                    return False
                mp = self.mouse_pos or QPointF(self.width() / 2, self.height() / 2)
                self.adjust = (('quick', slot), mp.x(), mp.y(), v0)
                self.update()
                return True
        except Exception:
            pass
        if self.brush is None or self.tool in ('select', 'move', 'rotate', 'scale') or self.drag is not None:
            return False
        mp = self.mouse_pos or QPointF(self.width() / 2, self.height() / 2)
        v0 = float(self.brush['radius'] if kind == 'radius' else self.brush.get('hardness', 0.0) if kind == 'hardness'
                   else self.brush['strength'])
        self.adjust = (kind, mp.x(), mp.y(), v0)
        self.update()
        return True

    def adjust_move(self, pos):
        kind, x0, y0, v0 = self.adjust
        dx = pos.x() - x0
        if isinstance(kind, tuple):                    # fs-shell-5 11-14: a registry tool's own setting
            from . import quick_settings as QS
            v = QS.drag_value(self.window(), kind[1], v0, dx)
            if v is not None:
                QS.set_slot(self.window(), kind[1], v)
            self.update()
            return
        if self.panel is None:
            return
        if kind == 'radius':
            f_px = self.height() / (2.0 * math.tan(math.radians(self.cam.fov) * 0.5))
            if self.hover is None:
                d = float(np.linalg.norm(self.cam.target - self.cam.pos))
            else:
                d = float(np.linalg.norm(np.asarray(self.hover['point'], float) - self.cam.pos))
            r_px = max(2.0, v0 * f_px / max(d, 1e-6) + dx)
            self.panel.set_radius(r_px * max(d, 1e-6) / f_px)
        elif kind == 'hardness':                       # fs-shell-5 11-14: Ctrl+F-drag, 0-100 % over 300 px
            self.panel.set_hardness(min(max(v0 + dx / 300.0, 0.0), 1.0))
        else:
            # fs-brush 11-11 (design 4.7: "Shift+F drags past 1"; every brush now rises up to 2): 0-2
            self.panel.set_strength(min(max(v0 + dx / 300.0, 0.0), 2.0))
        self.update()

    def end_adjust(self):
        self.adjust = None
        self.update()
