"""The Reference Rig dialog (fs-shell, 2026-11-11; design 4.3): the front and the side photo side by side, set on one
centre and one scale by two clicks per image (crown, chin) and a centre column (the midline in the front, the ear canal
in the profile), or by Auto-fit from each photo's silhouette (render/refrig.py). Once an image has its crown and chin,
it is drawn at the rig's scale, so the two images stand crown to crown and chin to chin on the same screen rows: what
the viewport will show is visible here first. A check line (Shift+click the same landmark in each image) shows how
far the photos disagree at that height, amber past 1 mm.

Keys: Enter applies, Esc cancels, A runs Auto-fit, Backspace clears the image under the mouse, 1 / 2 / 3 pick the
mark the next click sets (crown / chin / centre). Clicking sets the next unset mark in order; dragging a mark moves it."""
import copy

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QKeySequence, QShortcut
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

from ..render import refrig as RR

ORDER = ('crown', 'chin', 'centre')
COL = {'crown': QColor(80, 230, 120), 'chin': QColor(70, 200, 255), 'centre': QColor(255, 210, 70)}
NAMES = {'front': {'crown': 'crown', 'chin': 'chin', 'centre': 'midline'},
         'side': {'crown': 'crown', 'chin': 'chin', 'centre': 'ear canal'}}
SECOND = ('chin', 'nose base', 'pupils', 'mouth', 'ear lobe')
TOP_MARGIN = 0.10          # the crown's screen row, as a fraction of the canvas height, once a scale is set
SPAN_FRAC = 0.62           # crown-to-chin as a fraction of the canvas height


class RigCanvas(QWidget):
    changed = Signal()
    checkPicked = Signal(str, float)            # plane name, image row (Shift+click)

    def __init__(self, name, rgba, marks, parent=None):
        super().__init__(parent)
        self.name = name
        self.rgba = rgba
        h, w = rgba.shape[:2]
        self.img = QImage(np.ascontiguousarray(rgba).data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        self.W, self.H = w, h
        self.marks = dict(marks or {})
        self.next_mark = None
        self.drag = None
        self.pending = None
        self.press = None
        self.frozen = None                     # the view transform held during a drag
        self.check_rows = {}                   # name -> image row of the check landmark in that image
        self.check_world = None                # world height (mm, chin = 0 reference) of the line
        self.peer = None
        self.mouse = None
        self.setMouseTracking(True)
        self.setMinimumSize(360, 520)
        self.setCursor(Qt.CrossCursor)

    # ---- transform: image px <-> widget px -----------------------------------------------------------------------
    def scale_set(self):
        return 'crown' in self.marks and 'chin' in self.marks and abs(self.marks['chin'][1] - self.marks['crown'][1]) > 4

    def xform(self):
        if self.frozen is not None:
            return self.frozen
        cw, ch = self.width(), self.height()
        if self.scale_set():
            s = SPAN_FRAC * ch / abs(self.marks['chin'][1] - self.marks['crown'][1])
            cx = self.marks.get('centre', [None])[0] if 'centre' in self.marks else self.marks['crown'][0]
            ox = 0.5 * cw - s * cx
            oy = TOP_MARGIN * ch - s * self.marks['crown'][1]
        else:
            s = min(cw / self.W, ch / self.H)
            ox, oy = 0.5 * (cw - s * self.W), 0.5 * (ch - s * self.H)
        return s, ox, oy

    def to_w(self, x, y):
        s, ox, oy = self.xform()
        return QPointF(ox + s * x, oy + s * y)

    def to_img(self, p):
        s, ox, oy = self.xform()
        return (p.x() - ox) / s, (p.y() - oy) / s

    # ---- painting ----------------------------------------------------------------------------------------------
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(20, 22, 26))
        s, ox, oy = self.xform()
        p.drawImage(QRectF(ox, oy, s * self.W, s * self.H), self.img)
        nm = dict(NAMES[self.name], chin=getattr(self.window(), 'second', 'chin'))
        if 'centre' in self.marks:
            x = self.to_w(self.marks['centre'][0], 0).x()
            p.setPen(QPen(COL['centre'], 1.4, Qt.DashLine))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            self._label(p, QPointF(x + 4, self.height() - 8), nm['centre'], COL['centre'])
        for k in ('crown', 'chin'):
            if k in self.marks:
                q = self.to_w(*self.marks[k])
                p.setPen(QPen(COL[k], 1.0, Qt.DotLine))
                p.drawLine(QPointF(0, q.y()), QPointF(self.width(), q.y()))
                p.setPen(QPen(COL[k], 2.0))
                p.drawLine(q + QPointF(-9, 0), q + QPointF(9, 0))
                p.drawLine(q + QPointF(0, -9), q + QPointF(0, 9))
                self._label(p, q + QPointF(10, -6), nm[k], COL[k])
        # the check line: each image's own pick solid; the other image's pick dashed at the same screen row (both
        # images stand at the rig's scale, so one screen row is one height); green when they agree within 1 mm
        if self.check_rows:
            info = getattr(self.window(), 'check_info', None)
            col = QColor(240, 240, 240) if not info else (QColor(90, 220, 120) if info['ok'] else QColor(255, 190, 60))
            for nme, row in self.check_rows.items():
                src = self if nme == self.name else self.peer
                if src is None or (src is not self and not (src.scale_set() and self.scale_set())):
                    continue
                y = src.to_w(0, row).y()
                p.setPen(QPen(col, 1.4, Qt.SolidLine if src is self else Qt.DashLine))
                p.drawLine(QPointF(0, y), QPointF(self.width(), y))
            if info:
                self._label(p, QPointF(self.width() - 170, 20), 'check: differ %.1f mm' % info['delta_mm'], col)
        # title and the next click
        title = 'FRONT' if self.name == 'front' else 'SIDE (profile)'
        self._label(p, QPointF(10, 20), title, QColor(230, 230, 230), bold=True)
        nxt = self.next_unset()
        nm = dict(NAMES[self.name], chin=getattr(self.window(), 'second', 'chin'))
        if nxt:
            self._label(p, QPointF(10, 40), 'click: %s' % nm[nxt], COL[nxt])
        # loupe: 4x around the mouse, in the corner away from it
        if self.mouse is not None:
            ix, iy = self.to_img(self.mouse)
            if 0 <= ix < self.W and 0 <= iy < self.H:
                L = 130
                src = QRectF(ix - L / 8.0, iy - L / 8.0, L / 4.0, L / 4.0)
                right = self.mouse.x() < self.width() / 2
                dst = QRectF(self.width() - L - 8 if right else 8, self.height() - L - 8, L, L)
                p.fillRect(dst, QColor(0, 0, 0))
                p.drawImage(dst, self.img, src)
                p.setPen(QPen(QColor(255, 255, 255, 180), 1))
                p.drawRect(dst)
                c = dst.center()
                p.drawLine(c + QPointF(-8, 0), c + QPointF(8, 0))
                p.drawLine(c + QPointF(0, -8), c + QPointF(0, 8))
        p.end()

    @staticmethod
    def _label(p, at, text, col, bold=False):
        f = QFont('Segoe UI', 9)
        f.setBold(bold)
        p.setFont(f)
        p.setPen(QColor(0, 0, 0, 200))
        p.drawText(at + QPointF(1, 1), text)
        p.setPen(col)
        p.drawText(at, text)

    # ---- marks ---------------------------------------------------------------------------------------------------
    def next_unset(self):
        if self.next_mark:
            return self.next_mark
        for k in ORDER:
            if k not in self.marks:
                return k
        return None

    def _near(self, pos):
        best = None
        for k in ('crown', 'chin'):
            if k in self.marks:
                q = self.to_w(*self.marks[k])
                d = (q - pos).manhattanLength()
                if d < 14 and (best is None or d < best[0]):
                    best = (d, k)
        if 'centre' in self.marks:
            x = self.to_w(self.marks['centre'][0], 0).x()
            d = abs(x - pos.x())
            if d < 7 and (best is None or d < best[0]):
                best = (d, 'centre')
        return None if best is None else best[1]

    def mousePressEvent(self, e):
        pos = e.position()
        ix, iy = self.to_img(pos)
        if e.modifiers() & Qt.ShiftModifier:
            self.checkPicked.emit(self.name, float(iy))
            return
        self.frozen = self.xform()
        k = self._near(pos)
        self.press = pos
        self.pending = None
        if k is None:
            k = self.next_unset()
            if k is None:
                self.frozen = None
                return
            self._set_mark(k, ix, iy)
            self.drag = k
        else:
            # a press on a mark drags it once the mouse moves; a click without a move sets the next unset mark there
            # (the crown sits on the front's midline, so a midline click near it must not grab the crown)
            self.pending = k
        self.update()

    def _set_mark(self, k, ix, iy):
        if k == 'centre' and 'centre' in self.marks and self.drag == 'centre':
            self.marks['centre'] = [float(ix), float(self.marks['centre'][1])]
        else:
            self.marks[k] = [float(ix), float(iy)]
        self.next_mark = None

    def mouseMoveEvent(self, e):
        self.mouse = e.position()
        if self.pending and (e.position() - self.press).manhattanLength() > 3:
            self.drag, self.pending = self.pending, None
        if self.drag:
            ix, iy = self.to_img(e.position())
            self._set_mark(self.drag, ix, iy)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.pending:
            k = self.next_unset()
            if k is not None:
                ix, iy = self.to_img(self.press)
                self._set_mark(k, ix, iy)
            self.pending = None
            self.frozen = None
            self.changed.emit()
        elif self.drag:
            self.drag = None
            self.frozen = None
            self.changed.emit()
        else:
            self.frozen = None
        self.update()

    def leaveEvent(self, e):
        self.mouse = None
        self.update()

    def clear(self):
        self.marks = {}
        self.update()
        self.changed.emit()


class RigDialog(QDialog):
    """apply_cb(marks_by_plane: {'front': {...}, 'side': {...}}, height_mm) places the planes."""

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.setWindowTitle('Reference Rig: front and side on one centre')
        self.resize(1180, 860)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        hint = QLabel('Click the crown, then the chin, then the centre (front: the midline; side: the ear canal) in each '
                      'photo, or press Auto-fit. Drag any mark to fix it. Shift+click the same landmark in both photos '
                      'to check them against each other.')
        hint.setWordWrap(True)
        hint.setObjectName('dim')
        top.addWidget(hint, 1)
        lay.addLayout(top)
        bar = QHBoxLayout()
        self.b_auto = QPushButton('Auto-fit (A)')
        self.b_auto.setToolTip('Find each photo\'s head outline and propose the crown, chin and centre')
        self.b_auto.clicked.connect(self.autofit)
        bar.addWidget(self.b_auto)
        bar.addSpacing(12)
        from PySide6.QtWidgets import QComboBox
        self.second = 'chin'
        self.second_box = QComboBox()
        for k in SECOND:
            self.second_box.addItem('Crown to %s' % k, k)
        self.second_box.setToolTip('The second mark in each photo: the chin, or a landmark both photos show when a beard '
                                   'or a collar hides it (Auto-fit proposes the chin only)')
        self.second_box.currentIndexChanged.connect(self._second_changed)
        bar.addWidget(self.second_box)
        self.height_mm = QDoubleSpinBox()
        self.height_mm.setRange(20.0, 3000.0)
        self.height_mm.setDecimals(1)
        self.height_mm.setSuffix(' mm')
        self.height_mm.setValue(float(getattr(panel, 'rig_height_mm', None) or 230.0))
        self.height_mm.setToolTip('The real distance from the crown to the second mark (crown to chin, vertex to menton: '
                                  'adults about 215-235 mm)')
        self.height_mm.valueChanged.connect(self._update_info)
        bar.addWidget(self.height_mm)
        self.b_fit = QPushButton('Fit to pairs')
        self.b_fit.setToolTip('Least squares over the crown, the second mark and every landmark you Shift+clicked in '
                              'both photos: the height and scale of the side photo are set so they all agree as well as the '
                              'photos allow (its disagreement spread over them, not left on one landmark)')
        self.b_fit.setEnabled(False)
        self.b_fit.clicked.connect(self.fit_pairs)
        bar.addWidget(self.b_fit)
        bar.addSpacing(12)
        self.info = QLabel('')
        bar.addWidget(self.info, 1)
        b = QPushButton('Cancel')
        b.clicked.connect(self.reject)
        bar.addWidget(b)
        self.b_ok = QPushButton('Apply (Enter)')
        self.b_ok.setDefault(True)
        self.b_ok.clicked.connect(self.accept)
        bar.addWidget(self.b_ok)
        lay.addLayout(bar)
        row = QHBoxLayout()
        self.canvases = {}
        for name in ('front', 'side'):
            arr = panel.rgba.get(name)
            if arr is None:
                continue
            p = panel.vp.refs.get(name)
            marks = copy.deepcopy(getattr(p, 'marks', None) or {})
            c = RigCanvas(name, arr, marks)
            c.changed.connect(self._update_info)
            c.checkPicked.connect(self._check)
            row.addWidget(c, 1)
            self.canvases[name] = c
        if len(self.canvases) == 2:
            self.canvases['front'].peer = self.canvases['side']
            self.canvases['side'].peer = self.canvases['front']
        lay.addLayout(row, 1)
        self.check = {}
        self.pairs = []                          # (front row, side row) of each landmark checked in both photos
        self._pend = {}
        self.check_info = None
        QShortcut(QKeySequence('A'), self, activated=self.autofit)
        for i, k in enumerate(ORDER):
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda kk=k: self._pick_next(kk))
        QShortcut(QKeySequence(Qt.Key_Backspace), self, activated=self._clear_under_mouse)
        if all(not c.marks for c in self.canvases.values()):
            self.autofit()
        self._update_info()

    def _second_changed(self, i):
        self.second = self.second_box.itemData(i)
        self._update_info()

    def _pick_next(self, k):
        for c in self.canvases.values():
            if c.underMouse():
                c.next_mark = k
                c.update()

    def _clear_under_mouse(self):
        for c in self.canvases.values():
            if c.underMouse():
                c.clear()

    def autofit(self):
        notes = []
        for name, c in self.canvases.items():
            m = self.panel.mattes.get(name)
            if m is None or not m.get('keyed'):
                notes.append('%s: no plain background to find the head on' % name)
                continue
            af = RR.autofit(m, c.rgba, name)
            if af is None:
                notes.append('%s: no head found' % name)
                continue
            cy = af['ear'][1] if 'ear' in af else 0.5 * (af['crown'][1] + af['chin'][1])
            c.marks = {'crown': list(af['crown']), 'chin': list(af['chin']), 'centre': [af['centre_x'], cy]}
            if name == 'side':
                c.marks['flip_h'] = bool(af.get('flip_h', False))
            c.update()
        self.autofit_notes = notes
        self._update_info()

    def marks(self):
        out = {}
        for name, c in self.canvases.items():
            mk = c.marks
            if 'crown' in mk and 'chin' in mk:
                d = {'crown': list(mk['crown']), 'chin': list(mk['chin']),
                     'centre_x': float(mk['centre'][0]) if 'centre' in mk else float(mk['crown'][0])}
                if 'centre' in mk:
                    d['centre'] = list(mk['centre'])
                if 'flip_h' in mk:
                    d['flip_h'] = mk['flip_h']
                d['second'] = self.second
                out[name] = d
        return out

    def _check(self, name, row):
        # one landmark is a pair of Shift+clicks, one in each photo, in either order; a third click starts a new pair
        if name in self._pend or len(self._pend) == 2:
            self._pend = {}
        self._pend[name] = row
        self.check = dict(self._pend)
        if len(self._pend) == 2:
            self.pairs.append((float(self._pend['front']), float(self._pend['side'])))   # a landmark seen in both
            self.b_fit.setEnabled(True)
        for c in self.canvases.values():
            c.check_rows = dict(self.check)
            c.check_world = True
            c.update()
        self._update_info()

    def fit_pairs(self):
        """move the side photo's crown and second mark so the side maps to the front by the least-squares line through
        every shared landmark (rows: crown, second mark, the checked pairs). -> the residuals in mm"""
        mk = self.marks()
        if len(mk) < 2:
            return None
        hm = self.height_mm.value()
        f, s_ = mk['front'], mk['side']
        mpp_f = hm / abs(f['chin'][1] - f['crown'][1])
        rows = [(f['crown'][1], s_['crown'][1]), (f['chin'][1], s_['chin'][1])] + list(self.pairs)
        rf = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows])
        yf = (rf[0] - rf) * mpp_f                       # the front's heights below its crown (mm)
        A = np.stack([rs, np.ones_like(rs)], 1)
        (a, b), *_ = np.linalg.lstsq(A, yf, rcond=None)  # side row -> the front's height below the crown
        c = self.canvases['side']
        # the side's crown and second mark where the line puts the front's crown (0) and second mark heights
        c.marks['crown'][1] = float((0.0 - b) / a)
        c.marks['chin'][1] = float((yf[1] - b) / a)
        c.update()
        res = (a * rs + b) - yf
        self.fit_residuals = [round(float(v), 2) for v in res]
        self._update_info()
        return self.fit_residuals

    def _update_info(self, *_):
        mk = self.marks()
        hm = self.height_mm.value()
        parts = []
        for name in ('front', 'side'):
            if name in mk:
                rows = abs(mk[name]['chin'][1] - mk[name]['crown'][1])
                parts.append('%s %.3f mm/px' % (name, hm / max(rows, 1)))
            elif name in self.canvases:
                parts.append('%s: set crown and chin' % name)
        self.check_info = None
        if len(self.check) == 2 and len(mk) == 2:
            ys = {}
            for name, row in self.check.items():
                m = mk[name]
                mpp = hm / abs(m['chin'][1] - m['crown'][1])
                ys[name] = (0.5 * (m['crown'][1] + m['chin'][1]) - row) * mpp
            d = ys['front'] - ys['side']
            self.check_info = {'front_mm': ys['front'], 'side_mm': ys['side'], 'delta_mm': d, 'ok': abs(d) <= 1.0}
            parts.append('<span style="color:%s">check: front %.1f, side %.1f mm, differ %.1f mm</span>' % (
                '#4ade80' if abs(d) <= 1.0 else '#fbbf24', ys['front'], ys['side'], d))
        if self.pairs and len(mk) == 2:
            mf = hm / abs(mk['front']['chin'][1] - mk['front']['crown'][1])
            ms = hm / abs(mk['side']['chin'][1] - mk['side']['crown'][1])
            ds = [((rf - mk['front']['crown'][1]) * mf - (rs - mk['side']['crown'][1]) * ms) for rf, rs in self.pairs]
            worst = max(abs(v) for v in ds)
            parts.append('<span style="color:%s">%d pair%s: worst %.1f mm</span>' % (
                '#4ade80' if worst <= 1.0 else '#fbbf24', len(ds), 's' if len(ds) > 1 else '', worst))
        for n in getattr(self, 'autofit_notes', []) or []:
            parts.append('<span style="color:#fbbf24">%s</span>' % n)
        self.info.setText(' | '.join(parts))
        self.b_ok.setEnabled(len(mk) >= 1)
        for c in self.canvases.values():
            c.update()
