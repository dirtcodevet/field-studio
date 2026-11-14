"""File > New: the start page (fs-shell, 2026-11-11; design 4.5 "a start that is already right").

Four starts, each one click (or its number key, then Enter):
- Head: the bust sphere (240 mm) with a neck stub blended in, one object on a 1 mm lattice (brushes up to 80 mm),
  Mirror X on, a 40 mm Grab in hand, framed from the front; "then load front + side photos" (on by default) opens the
  file picker and the Reference Rig straight after (test 1's start).
- Figure: the standing 1680 mm blockout (core/figure.py), Mirror X, references at figure scale.
- Hand: a palm, four fingers and a thumb as one object on a 0.5 mm lattice, at an adult's size (hand length ~185 mm).
- Empty: nothing in the scene.
The thumbnails are drawn here (our own line drawings, P13)."""
import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QToolButton, QVBoxLayout

ALBEDO = [0.66, 0.56, 0.49]

STARTS = [
    ('head', 'Head', 'A 240 mm sphere with a neck, 1 mm lattice, Mirror X. Sculpt a head against front and side photos.'),
    ('figure', 'Figure', 'A standing 1680 mm blockout: head, torso, pelvis and limb pairs, Mirror X.'),
    ('hand', 'Hand', 'A palm, four fingers and a thumb at adult size, one object on a 0.5 mm lattice.'),
    ('empty', 'Empty', 'An empty scene. Add primitives from the Create bar.'),
]


def head_spec(r=0.12, neck_r=0.052, blend=0.025, h=0.001):
    """the Head start: one object, a sphere blended with a neck stub (a cylinder under the back half of the sphere,
    where a neck leaves a skull), so the neck is never pulled out of the sphere with Grab (test 1's torn neck ring)"""
    neck = {'id': 5004, 'kind': 'cylinder', 'name': None,
            'params': {'radius': neck_r, 'height': 0.16, 'round': 0.0}, 'children': []}
    base = {'id': 5001, 'kind': 'union', 'name': 'Head', 'params': {'blend': blend}, 'children': [
        {'id': 5002, 'kind': 'sphere', 'name': None, 'params': {'radius': r}, 'children': []},
        {'id': 5003, 'kind': 'transform', 'name': 'neck',
         'params': {'translate': [0.0, -r - 0.03, -0.02], 'rotate': [0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0]},
         'children': [neck]}]}
    return {'name': 'Head', 'h': float(h), 'base': base, 'translate': [0.0, 0.0, 0.0],
            'colour': {'albedo': list(ALBEDO), 'variation': 0.0}}


def hand_spec(h=0.0005):
    """the Hand start (adult, E: hand length ~185 mm, palm 85 x 95 x 28 mm): the palm a rounded box at the origin,
    the fingers capsules toward +y, the thumb toward +x and forward; one object, a small blend at the knuckles"""
    nid = [6000]

    def node(kind, params, children=(), name=None):
        nid[0] += 1
        return {'id': nid[0], 'kind': kind, 'name': name, 'params': params, 'children': list(children)}

    def placed(kind, params, at, rot=(0.0, 0.0, 0.0), name=None):
        return node('transform', {'translate': list(at), 'rotate': list(rot), 'scale': [1.0, 1.0, 1.0]},
                    [node(kind, params)], name)
    kids = [node('box', {'size': [0.085, 0.095, 0.028], 'round': 0.010}, name='palm')]
    for i, (x, length, r) in enumerate(((-0.031, 0.072, 0.0085), (-0.010, 0.082, 0.0092), (0.011, 0.076, 0.0088),
                                         (0.031, 0.062, 0.0078))):
        kids.append(placed('capsule', {'radius': r, 'height': length - 2 * r},
                           (x, 0.0475 + 0.5 * length - 0.004, 0.0), name='finger %d' % (i + 1)))
    kids.append(placed('capsule', {'radius': 0.0105, 'height': 0.045}, (0.050, -0.012, 0.010), (0.0, 0.0, -40.0),
                       name='thumb'))
    base = kids[0]                                    # a boolean takes two children: a left-nested chain of unions
    for k in kids[1:]:
        base = node('union', {'blend': 0.004}, [base, k])
    base['name'] = 'Hand'
    return {'name': 'Hand', 'h': float(h), 'base': base, 'translate': [0.0, 0.0, 0.0],
            'colour': {'albedo': list(ALBEDO), 'variation': 0.0}}


def _thumb(kind, size=96):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    line = QPen(QColor(230, 232, 236), 2.2)
    fill = QColor(120, 128, 140, 90)
    p.setPen(line)
    p.setBrush(fill)
    c = size / 2.0
    if kind == 'head':
        p.drawEllipse(QPointF(c, c - 8), 28, 28)
        p.drawRoundedRect(QRectF(c - 12, c + 14, 24, 26), 6, 6)
        p.setPen(QPen(QColor(255, 210, 70), 1.4, Qt.DashLine))
        p.drawLine(QPointF(c, 6), QPointF(c, size - 6))
    elif kind == 'figure':
        p.drawEllipse(QPointF(c, 14), 7, 9)
        path = QPainterPath()
        path.addRoundedRect(QRectF(c - 13, 25, 26, 32), 8, 8)
        p.drawPath(path)
        for sx in (-1, 1):
            p.drawRoundedRect(QRectF(c + sx * 16 - 4, 27, 8, 30), 4, 4)
            p.drawRoundedRect(QRectF(c + sx * 7 - 5, 56, 10, 34), 5, 5)
    elif kind == 'hand':
        p.drawRoundedRect(QRectF(c - 20, c - 2, 40, 36), 8, 8)
        for i, (dx, ln) in enumerate(((-15, 30), (-5, 36), (5, 33), (15, 26))):
            p.drawRoundedRect(QRectF(c + dx - 4.5, c - 2 - ln, 9, ln + 6), 4.5, 4.5)
        p.save()
        p.translate(c + 22, c + 14)
        p.rotate(-40)
        p.drawRoundedRect(QRectF(-5, -24, 10, 28), 5, 5)
        p.restore()
    else:
        p.setPen(QPen(QColor(120, 128, 140), 1.0))
        for i in range(1, 6):
            p.drawLine(QPointF(8, i * size / 6.0), QPointF(size - 8, i * size / 6.0))
            p.drawLine(QPointF(i * size / 6.0, 8), QPointF(i * size / 6.0, size - 8))
    p.end()
    return QIcon(pm)


class StartPage(QDialog):
    """-> self.choice in ('head', 'figure', 'hand', 'empty') and self.with_refs (bool) after accept"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('New')
        self.choice = None
        lay = QVBoxLayout(self)
        t = QLabel('Start from')
        t.setStyleSheet('font-size: 13pt; font-weight: 600;')
        lay.addWidget(t)
        row = QHBoxLayout()
        self.buttons = {}
        for i, (key, title, tip) in enumerate(STARTS):
            b = QToolButton()
            b.setIcon(_thumb(key))
            b.setIconSize(QSize(96, 96))
            b.setText('%d  %s' % (i + 1, title))
            b.setToolTip(tip)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setFixedSize(150, 150)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self.pick(k, accept=True))
            row.addWidget(b)
            self.buttons[key] = b
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda k=key: self.pick(k))
        lay.addLayout(row)
        self.desc = QLabel('')
        self.desc.setObjectName('dim')
        self.desc.setWordWrap(True)
        lay.addWidget(self.desc)
        self.refs = QCheckBox('Head: then load front + side photos and align them (Reference Rig)')
        self.refs.setChecked(True)
        lay.addWidget(self.refs)
        QShortcut(QKeySequence(Qt.Key_Return), self, activated=self._enter)
        QShortcut(QKeySequence(Qt.Key_Enter), self, activated=self._enter)
        self.pick('head')

    def pick(self, key, accept=False):
        self.choice = key
        for k, b in self.buttons.items():
            b.setChecked(k == key)
        self.desc.setText(dict((k, d) for k, _, d in STARTS)[key])
        self.refs.setEnabled(key == 'head')
        if accept:
            self.accept()

    def _enter(self):
        if self.choice:
            self.accept()

    @property
    def with_refs(self):
        return self.choice == 'head' and self.refs.isChecked()
