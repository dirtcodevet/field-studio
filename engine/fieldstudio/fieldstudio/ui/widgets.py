"""Small widgets: slider + spin field, vector field, the falloff curve editor, painted icons."""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF, QRadialGradient)
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QSlider, QWidget

from ..core import brush as BR


class ElideLabel(QLabel):
    """fs-shell 11-11: a one-line hint that never sets the window's width: it takes its full text's width when there is
    room and is cut with ... when there is not (the full text in its tooltip). The status bar's and the context bar's
    hints held the window at 1632 px minimum before."""

    def __init__(self, text='', parent=None, min_w=80):
        super().__init__(text, parent)
        self._min_w = min_w

    def setText(self, t):
        super().setText(t)
        self.setToolTip(t)
        self.updateGeometry()
        self.update()

    def minimumSizeHint(self):
        s = super().minimumSizeHint()
        s.setWidth(self._min_w)
        return s

    def paintEvent(self, e):
        p = QPainter(self)
        p.setPen(self.palette().color(self.foregroundRole()))
        p.setFont(self.font())
        r = self.contentsRect()
        p.drawText(r, int(self.alignment() | Qt.AlignVCenter),
                   self.fontMetrics().elidedText(self.text(), Qt.ElideRight, r.width()))
        p.end()


class FloatField(QWidget):
    """Label + slider + spin box for one float. value() / setValue() in SI units; shown as value * factor."""
    valueChanged = Signal(float)
    editStarted = Signal()

    def __init__(self, label, lo, hi, factor=1.0, decimals=2, log=False, suffix='', label_w=78, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.factor, self.log = lo, hi, factor, log and lo > 0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.label = QLabel(label)
        self.label.setFixedWidth(label_w)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(lo * factor, hi * factor)
        self.spin.setSingleStep((hi - lo) * factor / 100.0)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(78)
        self.spin.setKeyboardTracking(False)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._block = False
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderPressed.connect(self.editStarted)
        self.spin.valueChanged.connect(self._from_spin)

    def _to_pos(self, v):
        v = min(max(v, self.lo), self.hi)
        if self.log:
            return int(round(1000 * math.log(v / self.lo) / math.log(self.hi / self.lo)))
        return int(round(1000 * (v - self.lo) / max(self.hi - self.lo, 1e-12)))

    def _from_pos(self, i):
        if self.log:
            return self.lo * (self.hi / self.lo) ** (i / 1000.0)
        return self.lo + (self.hi - self.lo) * i / 1000.0

    def value(self):
        return self.spin.value() / self.factor

    def setValue(self, v, emit=False):
        self._block = True
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.spin.setValue(v * self.factor)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        self._block = False
        if emit:
            self.valueChanged.emit(self.value())

    def busy(self):
        return self.slider.isSliderDown() or self.spin.hasFocus()

    def _from_slider(self, i):
        v = self._from_pos(i)
        self.spin.blockSignals(True)
        self.spin.setValue(v * self.factor)
        self.spin.blockSignals(False)
        self.valueChanged.emit(v)

    def _from_spin(self, sv):
        v = sv / self.factor
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


class VecField(QWidget):
    """Label + 3 spin boxes (x, y, z) for a vector parameter."""
    valueChanged = Signal(list)

    def __init__(self, label, lo, hi, factor=1.0, decimals=1, step=None, label_w=78, parent=None):
        super().__init__(parent)
        self.factor = factor
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lb = QLabel(label)
        lb.setFixedWidth(label_w)
        lay.addWidget(lb)
        self.spins = []
        cols = ['#eb5050', '#6ed25a', '#508cf5']
        for i in range(3):
            s = QDoubleSpinBox()
            s.setDecimals(decimals)
            s.setRange(lo * factor, hi * factor)
            s.setSingleStep(step if step else (hi - lo) * factor / 200.0)
            s.setKeyboardTracking(False)
            s.setStyleSheet('QDoubleSpinBox { border-left: 2px solid %s; }' % cols[i])
            s.valueChanged.connect(self._changed)
            lay.addWidget(s, 1)
            self.spins.append(s)

    def value(self):
        return [s.value() / self.factor for s in self.spins]

    def setValue(self, v):
        for s, x in zip(self.spins, v):
            s.blockSignals(True)
            s.setValue(float(x) * self.factor)
            s.blockSignals(False)

    def busy(self):
        return any(s.hasFocus() for s in self.spins)

    def _changed(self, _):
        self.valueChanged.emit(self.value())


class FalloffCurve(QWidget):
    """Draws the brush falloff w(u) (u = distance / radius); in 'custom' mode its 6 control points drag."""
    curveChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kind = 'smooth'
        self.curve = [1.0, 0.95, 0.8, 0.5, 0.2, 0.0]
        self.drag = -1
        self.setMinimumHeight(54)
        self.setMaximumHeight(64)

    def set_kind(self, kind, curve=None):
        self.kind = kind
        if curve:
            self.curve = list(curve)
        self.update()

    def _pt(self, i, v):
        r = self.rect().adjusted(8, 6, -8, -6)
        return QPointF(r.left() + r.width() * i / (len(self.curve) - 1), r.bottom() - r.height() * v)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('#23272e'))
        r = self.rect().adjusted(8, 6, -8, -6)
        lut = BR.falloff_lut(self.kind, self.curve if self.kind == 'custom' else None)
        # mirror: a brush profile across the dab (-1..1)
        path = QPainterPath()
        n = len(lut)
        pts = []
        for k in range(-n + 1, n):
            w = lut[abs(k)]
            x = r.left() + r.width() * (k + n - 1) / (2 * n - 2)
            pts.append(QPointF(x, r.bottom() - r.height() * w))
        path.moveTo(QPointF(r.left(), r.bottom()))
        for q in pts:
            path.lineTo(q)
        path.lineTo(QPointF(r.right(), r.bottom()))
        g = QLinearGradient(0, r.top(), 0, r.bottom())
        g.setColorAt(0, QColor(160, 170, 185, 200))
        g.setColorAt(1, QColor(60, 66, 76, 120))
        p.fillPath(path, QBrush(g))
        p.setPen(QPen(QColor(220, 225, 232), 1.4))
        p.drawPolyline(QPolygonF(pts))
        if self.kind == 'custom':
            p.setPen(QPen(QColor('#3b82f6'), 1))
            p.setBrush(QColor('#3b82f6'))
            for i, v in enumerate(self.curve):
                q = self._pt(i, v)
                q.setX(r.center().x() + (q.x() - r.left()) / 2.0)
                p.drawEllipse(q, 3.5, 3.5)
        p.end()

    def _index_at(self, pos):
        r = self.rect().adjusted(8, 6, -8, -6)
        best, bi = 1e9, -1
        for i, v in enumerate(self.curve):
            q = self._pt(i, v)
            q.setX(r.center().x() + (q.x() - r.left()) / 2.0)
            d = abs(q.x() - pos.x())
            if d < best:
                best, bi = d, i
        return bi if best < 18 else -1

    def mousePressEvent(self, e):
        if self.kind == 'custom':
            self.drag = self._index_at(e.position())
            self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e):
        if self.drag < 0:
            return
        r = self.rect().adjusted(8, 6, -8, -6)
        v = min(max((r.bottom() - e.position().y()) / max(r.height(), 1), 0.0), 1.0)
        if self.drag == len(self.curve) - 1:
            v = 0.0
        self.curve[self.drag] = round(v, 3)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.drag >= 0:
            self.drag = -1
            self.curveChanged.emit(list(self.curve))


# ---- painted icons (no image files) ---------------------------------------------------------------------------

def _pix(size=32):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    return pm


def sphere_icon(col=(170, 175, 185), size=28):
    pm = _pix(size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    g = QRadialGradient(size * 0.38, size * 0.35, size * 0.6)
    c = QColor(*col)
    g.setColorAt(0, c.lighter(150))
    g.setColorAt(0.6, c)
    g.setColorAt(1, c.darker(260))
    p.setBrush(QBrush(g))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QRectF(3, 3, size - 6, size - 6))
    p.end()
    return QIcon(pm)


def tool_icon(tool, size=28):
    cols = {'select': (150, 160, 175), 'draw': (200, 170, 140), 'clay': (205, 150, 110), 'carve': (120, 150, 210),
            'inflate': (210, 190, 120), 'smooth': (150, 200, 220), 'flatten': (170, 170, 170),
            'crease': (190, 130, 150), 'pinch': (180, 140, 200), 'grab': (220, 200, 110), 'stretch': (220, 170, 90),
            'mask': (110, 110, 150), 'swell': (200, 160, 200), 'settle': (140, 190, 160)}
    if tool == 'select':
        pm = _pix(size)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.moveTo(8, 4)
        path.lineTo(8, 23)
        path.lineTo(13, 18)
        path.lineTo(17, 25)
        path.lineTo(20, 23)
        path.lineTo(16, 16)
        path.lineTo(23, 16)
        path.closeSubpath()
        p.setBrush(QColor(215, 220, 228))
        p.setPen(QPen(QColor(40, 40, 40), 1))
        p.drawPath(path)
        p.end()
        return QIcon(pm)
    return sphere_icon(cols.get(tool, (170, 170, 170)), size)


def prim_icon(kind, size=30):
    pm = _pix(size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    s = float(size)
    light, mid, dark = QColor(205, 210, 220), QColor(150, 156, 168), QColor(95, 100, 112)
    p.setPen(QPen(QColor(40, 42, 48), 1))
    if kind == 'sphere':
        p.end()
        return sphere_icon((175, 180, 190), size)
    if kind == 'box':
        top = QPainterPath()
        top.moveTo(s * .5, s * .12)
        top.lineTo(s * .88, s * .3)
        top.lineTo(s * .5, s * .48)
        top.lineTo(s * .12, s * .3)
        top.closeSubpath()
        left = QPainterPath()
        left.moveTo(s * .12, s * .3)
        left.lineTo(s * .5, s * .48)
        left.lineTo(s * .5, s * .9)
        left.lineTo(s * .12, s * .72)
        left.closeSubpath()
        right = QPainterPath()
        right.moveTo(s * .88, s * .3)
        right.lineTo(s * .5, s * .48)
        right.lineTo(s * .5, s * .9)
        right.lineTo(s * .88, s * .72)
        right.closeSubpath()
        for path, c in ((top, light), (left, mid), (right, dark)):
            p.setBrush(c)
            p.drawPath(path)
    elif kind in ('cylinder', 'cone'):
        g = QLinearGradient(s * .15, 0, s * .85, 0)
        g.setColorAt(0, dark)
        g.setColorAt(0.4, light)
        g.setColorAt(1, dark)
        body = QPainterPath()
        if kind == 'cylinder':
            body.moveTo(s * .18, s * .22)
            body.lineTo(s * .18, s * .78)
            body.arcTo(QRectF(s * .18, s * .68, s * .64, s * .2), 180, 180)
            body.lineTo(s * .82, s * .22)
        else:
            body.moveTo(s * .5, s * .1)
            body.lineTo(s * .14, s * .78)
            body.arcTo(QRectF(s * .14, s * .68, s * .72, s * .2), 180, 180)
        body.closeSubpath()
        p.setBrush(QBrush(g))
        p.drawPath(body)
        if kind == 'cylinder':
            p.setBrush(light)
            p.drawEllipse(QRectF(s * .18, s * .12, s * .64, s * .2))
    elif kind == 'capsule':
        g = QLinearGradient(s * .25, 0, s * .75, 0)
        g.setColorAt(0, dark)
        g.setColorAt(0.4, light)
        g.setColorAt(1, dark)
        p.setBrush(QBrush(g))
        p.drawRoundedRect(QRectF(s * .28, s * .08, s * .44, s * .84), s * .22, s * .22)
    elif kind == 'torus':
        path = QPainterPath()
        path.addEllipse(QRectF(s * .06, s * .22, s * .88, s * .56))
        inner = QPainterPath()
        inner.addEllipse(QRectF(s * .32, s * .4, s * .36, s * .2))
        g = QRadialGradient(s * .5, s * .38, s * .5)
        g.setColorAt(0, light)
        g.setColorAt(1, dark)
        p.setBrush(QBrush(g))
        p.drawPath(path.subtracted(inner))
    elif kind == 'ellipsoid':
        g = QRadialGradient(s * .4, s * .38, s * .55)
        g.setColorAt(0, light)
        g.setColorAt(1, dark)
        p.setBrush(QBrush(g))
        p.drawEllipse(QRectF(s * .06, s * .24, s * .88, s * .52))
    elif kind == 'slab':
        path = QPainterPath()
        path.moveTo(s * .1, s * .55)
        path.lineTo(s * .45, s * .35)
        path.lineTo(s * .92, s * .45)
        path.lineTo(s * .57, s * .65)
        path.closeSubpath()
        p.setBrush(light)
        p.drawPath(path)
        side = QPainterPath()
        side.moveTo(s * .1, s * .55)
        side.lineTo(s * .57, s * .65)
        side.lineTo(s * .57, s * .74)
        side.lineTo(s * .1, s * .64)
        side.closeSubpath()
        p.setBrush(mid)
        p.drawPath(side)
    else:
        p.setBrush(mid)
        p.drawRect(QRectF(s * .2, s * .2, s * .6, s * .6))
    p.end()
    return QIcon(pm)


def mutation_icon(kind, size=22):
    pm = _pix(size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(200, 206, 216), 1.6))
    s = float(size)
    if kind == 'twist':
        for k in range(3):
            y = s * (0.25 + 0.25 * k)
            p.drawLine(QPointF(s * (0.25 + 0.1 * k), y), QPointF(s * (0.75 - 0.1 * k), y + s * 0.05))
        p.drawArc(QRectF(s * .2, s * .1, s * .6, s * .8), 30 * 16, 120 * 16)
    elif kind == 'bend':
        p.drawArc(QRectF(s * .15, s * .3, s * .7, s * .9), 20 * 16, 140 * 16)
    elif kind == 'taper':
        p.drawLine(QPointF(s * .35, s * .15), QPointF(s * .2, s * .85))
        p.drawLine(QPointF(s * .65, s * .15), QPointF(s * .8, s * .85))
    elif kind == 'stretch':
        p.drawLine(QPointF(s * .5, s * .1), QPointF(s * .5, s * .9))
        p.drawLine(QPointF(s * .35, s * .25), QPointF(s * .5, s * .1))
        p.drawLine(QPointF(s * .65, s * .25), QPointF(s * .5, s * .1))
        p.drawLine(QPointF(s * .35, s * .75), QPointF(s * .5, s * .9))
        p.drawLine(QPointF(s * .65, s * .75), QPointF(s * .5, s * .9))
    elif kind == 'shell':
        p.drawEllipse(QRectF(s * .15, s * .15, s * .7, s * .7))
        p.drawEllipse(QRectF(s * .3, s * .3, s * .4, s * .4))
    elif kind == 'offset':
        p.drawRoundedRect(QRectF(s * .15, s * .15, s * .7, s * .7), s * .25, s * .25)
    else:
        p.drawRect(QRectF(s * .25, s * .25, s * .5, s * .5))
        p.drawLine(QPointF(s * .5, s * .05), QPointF(s * .5, s * .95))
    p.end()
    return QIcon(pm)


def falloff_values(kind, curve=None):
    return np.asarray(BR.falloff_lut(kind, curve))
