"""The Section tool (fs-shell, 2026-11-11; design 4.4: a plane dragged (or aligned to the view or
an axis) cuts the view; the cut face is filled from the field, since inside is known everywhere; a mm ruler
shows the gap or thickness under the cursor, on any object).

The plane lives on the view's camera (Camera.section = (n, d): the half n.p > d is cut away); the tracers start or stop
rays at it and fill the cut face from the field (render/raymarch.py on the CPU; the card half as agreed 11-11).
View > Section (or Tools > Section tool) puts a plane through the selected object (else the view's target) facing the
view, so the view looks into the cut. A bar over the view: align it to the view or to X / Y / Z, flip it, move it (mm, or
Shift+drag in the view along its normal; Ctrl snaps to 1 mm), the ruler, off. Under the cursor, on the plane: the wall's
thickness (inside a form) or the gap (between forms), measured in the plane along the field's gradient, both from the
field. The ruler: two clicks on the plane, a line with mm ticks and its length."""
import math

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton

SEC_COL = QColor(120, 200, 255)


def plane_hit(cam, pos, W, H, n, d):
    """the widget point's ray on the plane n.p = d -> world point or None"""
    o, dv = cam.ray(pos.x(), pos.y(), W, H)
    dn = float(np.dot(n, dv))
    if abs(dn) < 1e-9:
        return None
    t = (d - float(np.dot(n, o))) / dn
    if t <= 0:
        return None
    return np.asarray(o, float) + np.asarray(dv, float) * t


def across(scene, n, q, span=0.4, step=0.001):
    """on the engine thread: the wall or gap through q in the plane (normal n), along the field's in-plane gradient
    -> {'q', 'a', 'b', 'inside', 'width'} or None. The field is the scene's (the smallest of the objects near q)."""
    from ..tools.protected_merge import field_of, world_box
    pk = scene.pack()
    q = np.asarray(q, float)
    objs = []
    for o in scene.objects:
        if not getattr(o, 'visible', True):
            continue
        try:
            lo, hi = world_box(o)
        except Exception:
            continue
        if np.all(q >= np.asarray(lo) - span) and np.all(q <= np.asarray(hi) + span):
            objs.append(o)
    if not objs:
        return None

    def F(P):
        P = np.ascontiguousarray(np.asarray(P, float).reshape(-1, 3))
        v = np.full(len(P), np.inf)
        for o in objs:
            f = field_of(scene, pk, o, P)
            v = np.minimum(v, np.where(np.isfinite(f), f, np.inf))
        return v
    e = 1e-4
    s = np.arange(step, span, step)
    f0 = float(F(q)[0])
    if not np.isfinite(f0):
        return None
    inside = f0 < 0

    def chord(g):
        """the wall (or gap) through q along +-g: both ends where the sign changes (a 1 mm scan, then bisection to
        ~1 um), or None"""
        ends = []
        for sgn in (1.0, -1.0):
            P = q[None] + sgn * s[:, None] * g[None]
            f = F(P)
            k = np.nonzero(np.isfinite(f) & ((f < 0) != inside))[0]
            if not len(k):
                return None
            j = k[0]
            a_, b_ = (0.0 if j == 0 else s[j - 1]), s[j]
            for _ in range(10):
                m_ = 0.5 * (a_ + b_)
                fm = float(F(q + sgn * m_ * g)[0])
                if np.isfinite(fm) and (fm < 0) == inside:
                    a_ = m_
                else:
                    b_ = m_
            ends.append(q + sgn * 0.5 * (a_ + b_) * g)
        return ends

    # along the field's in-plane gradient; where it gives no direction (a stepped store's flat interior), the
    # shortest of 12 directions in the plane (the wall's local thickness through q)
    g = np.array([(F(q + e * a) - F(q - e * a))[0] for a in np.eye(3)]) / (2 * e)
    g = g - float(g @ n) * n
    best = None
    if np.all(np.isfinite(g)) and float(np.linalg.norm(g)) > 0.05:
        best = chord(g / np.linalg.norm(g))
    if best is None:
        u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        for ang in np.radians(np.arange(0.0, 180.0, 15.0)):
            ch = chord(np.cos(ang) * u + np.sin(ang) * v)
            if ch is not None and (best is None or np.linalg.norm(ch[1] - ch[0]) < np.linalg.norm(best[1] - best[0])):
                best = ch
    if best is None:
        return None
    a, b = best
    return {'q': q, 'a': a, 'b': b, 'inside': bool(inside), 'width': float(np.linalg.norm(b - a))}


class SectionTool(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.on = False
        self.c = np.zeros(3)                 # a point on the plane
        self.n = np.array([0.0, 0.0, 1.0])   # the normal: its side is cut away
        self.readout = None                  # the wall / gap under the cursor
        self.ruler = []                      # the ruler's points on the plane (0, 1 or 2)
        self.ruler_on = False
        self._busy = False
        self._want = None
        self.drag = None
        self.bar = SectionBar(self)

    # ---- the plane ---------------------------------------------------------------------------------------------------
    @property
    def d(self):
        return float(self.n @ self.c)

    def apply(self):
        vp = self.win.viewport
        vp.cam.section = (self.n.copy(), self.d) if self.on else None
        vp.request(False)
        vp.update()
        self.bar.sync()

    def start(self, on=True):
        vp = self.win.viewport
        if on and not self.on:
            sel = list(vp.selected or [])
            tr = vp.transforms.get(sel[0]) if sel else None
            self.c = np.asarray(tr['translate'], float) if tr else np.asarray(vp.cam.target, float)
            self.n = -np.asarray(vp.cam.basis()[0], float)       # facing the view: the view looks into the cut
            self._base = self.c.copy()
        self.on = bool(on)
        if not on:
            self.readout = None
            self.ruler = []
            self.ruler_on = False
        a = getattr(self, 'action', None)
        if a is not None and a.isChecked() != self.on:
            a.setChecked(self.on)
        self.bar.setVisible(self.on)
        self.apply()

    def align(self, which):
        vp = self.win.viewport
        if which == 'view':
            self.n = -np.asarray(vp.cam.basis()[0], float)
        else:
            i = 'xyz'.index(which)
            n = np.zeros(3)
            n[i] = 1.0
            f = np.asarray(vp.cam.basis()[0], float)
            self.n = n if float(n @ f) < 0 else -n                # the side toward the view is cut away
        self._base = self.c.copy()
        self.readout = None
        self.apply()

    def flip(self):
        self.n = -self.n
        self.readout = None
        self.apply()

    def offset_to(self, mm):
        """the plane at mm along its normal from where the tool put it"""
        base = getattr(self, '_base', None)
        if base is None:
            base = self._base = self.c.copy()
        self.c = base + self.n * (float(mm) / 1000.0)
        self.apply()

    # ---- under the cursor ------------------------------------------------------------------------------------------------
    def follow(self, pos):
        """the wall or gap under the cursor (latest wins: one engine job at a time)"""
        if not self.on:
            return
        vp = self.win.viewport
        q = plane_hit(vp.cam, pos, max(vp.width(), 1), max(vp.height(), 1), self.n, self.d)
        if q is None:
            self.readout = None
            return
        if self._busy:
            self._want = pos
            return
        self._busy = True
        n = self.n.copy()
        eng = self.win.engine
        tok = object()

        def fn(e):
            try:
                return across(e.hist.scene, n, q)
            except Exception:
                return None

        def done(rr, tk=tok):
            if rr[0] is not tk:
                return
            eng.callDone.disconnect(done)
            self._busy = False
            self.readout = rr[1]
            vp.update()
            if self._want is not None:
                p_, self._want = self._want, None
                self.follow(p_)
        eng.callDone.connect(done)
        eng.post('call', fn, tok)

    def ruler_click(self, pos):
        vp = self.win.viewport
        q = plane_hit(vp.cam, pos, max(vp.width(), 1), max(vp.height(), 1), self.n, self.d)
        if q is None:
            return
        if len(self.ruler) >= 2:
            self.ruler = []
        self.ruler.append(q)
        vp.update()

    # ---- drawing -----------------------------------------------------------------------------------------------------
    def draw(self, p):
        if not self.on:
            return
        vp = self.win.viewport
        if getattr(vp.cam, 'section', None) is None:             # the view's camera was replaced (a preset, a load)
            vp.cam.section = (self.n.copy(), self.d)
            vp.request(False)
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        cam = vp.cam
        from . import units as U
        unit = getattr(self.win, 'units', 'mm')
        # the plane: a square around its point, its normal
        f, r, u = cam.basis()
        e1 = np.cross(self.n, u if abs(float(self.n @ u)) < 0.9 else r)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.n, e1)
        R = 0.6 * float(np.linalg.norm(cam.pos - cam.target)) * math.tan(math.radians(cam.fov) * 0.5)
        sq = [self.c + R * (a * e1 + b * e2) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pr = [cam.project(q, W, H) for q in sq]
        if all(z > 0 for _, _, z in pr):
            p.setPen(QPen(QColor(SEC_COL.red(), SEC_COL.green(), SEC_COL.blue(), 140), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            for i in range(4):
                a, b = pr[i], pr[(i + 1) % 4]
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
        p.setFont(QFont('Segoe UI', 9, QFont.Bold))
        fm = p.fontMetrics()
        ro = self.readout
        if ro is not None:
            a = cam.project(ro['a'], W, H)
            b = cam.project(ro['b'], W, H)
            if a[2] > 0 and b[2] > 0:
                col = QColor(255, 214, 90) if ro['inside'] else QColor(120, 230, 160)
                p.setPen(QPen(col, 2.0))
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
                for q in (a, b):
                    p.drawEllipse(QPointF(q[0], q[1]), 3, 3)
                text = '%s %s' % ('thickness' if ro['inside'] else 'gap', U.format_length(ro['width'], unit))
                m = QPointF(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
                rr = QRectF(m.x() + 12, m.y() - 22, fm.horizontalAdvance(text) + 12, fm.height() + 6)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(16, 18, 22, 220))
                p.drawRoundedRect(rr, 4, 4)
                p.setPen(col)
                p.drawText(rr, Qt.AlignCenter, text)
        if self.ruler:
            pts = [cam.project(q, W, H) for q in self.ruler]
            p.setPen(QPen(SEC_COL, 2.0))
            for q in pts:
                p.drawEllipse(QPointF(q[0], q[1]), 3, 3)
            if len(self.ruler) == 2 and all(q[2] > 0 for q in pts):
                A, B = self.ruler
                L = float(np.linalg.norm(B - A))
                p.drawLine(QPointF(pts[0][0], pts[0][1]), QPointF(pts[1][0], pts[1][1]))
                # ticks every 1 mm (5 and 10 longer), while they stay at least 4 px apart; their length in pixels
                dv = (B - A) / max(L, 1e-12)
                sx, sy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
                sl = max(math.hypot(sx, sy), 1e-9)
                nx_, ny_ = -sy / sl, sx / sl                      # across the ruler on the screen
                step = 0.001
                px_per_m = sl / max(L, 1e-12)
                while step * px_per_m < 4 and step < 1.0:
                    step *= 10
                k = 0
                while k * step <= L + 1e-12 and k < 2000:
                    big = k % 10 == 0
                    mid = k % 5 == 0
                    tl = 13.0 if big else (9.0 if mid else 6.0)
                    q0 = cam.project(A + dv * k * step, W, H)
                    p.setPen(QPen(SEC_COL, 1.8 if big else 1.3))
                    p.drawLine(QPointF(q0[0], q0[1]), QPointF(q0[0] + nx_ * tl, q0[1] + ny_ * tl))
                    k += 1
                text = 'ruler %s' % U.format_length(L, unit)
                m = QPointF(0.5 * (pts[0][0] + pts[1][0]), 0.5 * (pts[0][1] + pts[1][1]))
                rr = QRectF(m.x() + 10, m.y() + 8, fm.horizontalAdvance(text) + 12, fm.height() + 6)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(16, 18, 22, 220))
                p.drawRoundedRect(rr, 4, 4)
                p.setPen(SEC_COL)
                p.drawText(rr, Qt.AlignCenter, text)
                self.ruler_len = L


class SectionBar(QFrame):
    """the controls over the view while the section is on"""

    def __init__(self, tool):
        vp = tool.win.viewport
        super().__init__(vp)
        self.tool = tool
        self.setObjectName('sectionBar')
        self.setStyleSheet('#sectionBar { background: rgba(18, 26, 34, 236); border: 1px solid #78c8ff; '
                           'border-radius: 6px; } QLabel { color: #d8ecff; } QPushButton { padding: 2px 9px; }')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 6, 4)
        lay.addWidget(QLabel('Section'))
        self.buttons = {}
        for key, text, tip in (('view', 'View', 'face the view'), ('x', 'X', 'across X'), ('y', 'Y', 'across Y'),
                               ('z', 'Z', 'across Z')):
            b = QPushButton(text)
            b.setToolTip('Align the plane: ' + tip)
            b.clicked.connect(lambda _=False, k=key: tool.align(k))
            lay.addWidget(b)
            self.buttons[key] = b
        fl = QPushButton('Flip')
        fl.setToolTip('Cut the other side away')
        fl.clicked.connect(tool.flip)
        lay.addWidget(fl)
        lay.addWidget(QLabel('move'))
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-2000.0, 2000.0)
        self.offset.setDecimals(1)
        self.offset.setSingleStep(1.0)
        self.offset.setSuffix(' mm')
        self.offset.setToolTip('Move the plane along its normal (or Shift+drag in the view; Ctrl snaps to 1 mm)')
        self.offset.valueChanged.connect(lambda v: tool.offset_to(v) if not self._sync else None)
        lay.addWidget(self.offset)
        self.rb = QPushButton('Ruler')
        self.rb.setCheckable(True)
        self.rb.setToolTip('Two clicks on the plane: a mm ruler')
        self.rb.toggled.connect(lambda v: setattr(tool, 'ruler_on', bool(v)))
        lay.addWidget(self.rb)
        off = QPushButton('Off')
        off.clicked.connect(lambda: tool.start(False))
        lay.addWidget(off)
        self._sync = False
        self.hide()

    def sync(self):
        t = self.tool
        base = getattr(t, '_base', None)
        self._sync = True
        self.offset.setValue(0.0 if base is None else float(t.n @ (t.c - base)) * 1000.0)
        self._sync = False
        if self.rb.isChecked() != t.ruler_on:
            self.rb.setChecked(t.ruler_on)
        self.place()

    def place(self):
        """bottom right of the view (the stats box sits bottom left)"""
        vp = self.parentWidget()
        self.adjustSize()
        w = min(vp.width() - 24, self.sizeHint().width())
        self.setGeometry(int(vp.width() - w - 14), vp.height() - self.sizeHint().height() - 14, int(w),
                         self.sizeHint().height())


class _SectionFilter(QObject):
    """on the viewport while the section is on: the cursor's readout; with the ruler, left clicks set its ends;
    Shift + left drag moves the plane along its normal"""

    def __init__(self, tool):
        super().__init__(tool)
        self.tool = tool

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        T = self.tool
        if not T.on:
            return False
        t = ev.type()
        if t == QEvent.MouseMove:
            if T.drag is not None:
                vp = T.win.viewport
                dy = ev.position().y() - T.drag['y0']
                dist = float(np.linalg.norm(vp.cam.pos - vp.cam.target))
                m_per_px = 2.0 * dist * math.tan(math.radians(vp.cam.fov) * 0.5) / max(vp.height(), 1)
                off = T.drag['off0'] - dy * m_per_px * 1000.0
                if ev.modifiers() & Qt.ControlModifier:
                    off = round(off)
                T.bar.offset.setValue(off)
                return True
            T.follow(ev.position())
            return False
        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            if T.ruler_on:
                T.ruler_click(ev.position())
                return True
            if ev.modifiers() & Qt.ShiftModifier:
                T.drag = {'y0': ev.position().y(), 'off0': T.bar.offset.value()}
                return True
        if t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and T.drag is not None:
            T.drag = None
            return True
        if t == QEvent.Resize:
            QTimer.singleShot(0, T.bar.place)
        return False


def install(win):
    T = SectionTool(win)
    win.section = T
    T.filter = _SectionFilter(T)
    win.viewport.installEventFilter(T.filter)
    win.viewport.setMouseTracking(True)
    vm = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'View' in a.menu().title()), None)
    if vm is not None:
        a = vm.addAction('Section (cut the view with a plane; thickness, gap, mm ruler)')
        a.setCheckable(True)
        a.toggled.connect(lambda v: T.start(v))
        T.action = a
    tm = getattr(win, 'menu_tools', None)
    if tm is not None:
        b = tm.addAction('Section tool (cut the view with a plane)')
        b.triggered.connect(lambda: T.start(True))
    return T
