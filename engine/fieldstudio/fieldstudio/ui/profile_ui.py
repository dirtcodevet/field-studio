"""The girth profile panel (fs-shell, 2026-11-11; design: the tool finds girths itself, but
not body regions, which would not work for animals, dragons, blocks or logs).

Tools > Measure > Girth profile of the selected object: sections along an axis (the form's own long axis found by the
tool, or X / Y / Z, or the view's up) measured off the field on the engine thread; a docked chart of girth by position
(the tape solid, the outline thin), with what the tool finds on any form (the narrowest and widest sections, waists,
bulges, sudden changes), a body-plan template's named landmarks where one fits (Human torso: hips, waist, underbust,
bust), and the user's own named sections. Hover the chart: that section shows on the model. Click: pin a girth there. Right-
click: name the section (pinned, labelled with that name)."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QComboBox, QDockWidget, QHBoxLayout, QInputDialog, QLabel, QMenu, QPushButton,
                               QVBoxLayout, QWidget)

FEAT_COL = {'narrowest': QColor(134, 239, 172), 'widest': QColor(251, 146, 60), 'waist': QColor(134, 239, 172),
            'bulge': QColor(251, 146, 60), 'sudden change': QColor(248, 113, 113)}
NAMED_COL = QColor(167, 139, 250)


class ProfileChart(QWidget):
    hovered = Signal(object)          # index into the profile, or None
    clicked = Signal(int)
    named = Signal(int)

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.setMouseTracking(True)
        self.setMinimumHeight(170)
        self.hover = None

    def _frame(self):
        return QRectF(56, 10, max(10, self.width() - 70), max(10, self.height() - 44))

    def _xy(self, s, g):
        pr = self.panel.prof
        r = self._frame()
        s0, s1 = float(pr['s'].min()), float(pr['s'].max())
        g0, g1 = 0.0, float(max(pr['length'].max(), pr['tape'].max())) * 1.05
        return QPointF(r.left() + (s - s0) / max(s1 - s0, 1e-9) * r.width(), r.bottom() - (g - g0) / max(g1 - g0, 1e-9) * r.height())

    def _index_at(self, x):
        pr = self.panel.prof
        if pr is None:
            return None
        r = self._frame()
        s0, s1 = float(pr['s'].min()), float(pr['s'].max())
        s = s0 + (x - r.left()) / max(r.width(), 1) * (s1 - s0)
        return int(np.argmin(np.abs(pr['s'] - s)))

    def paintEvent(self, e):
        from .units import format_length as F
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(24, 26, 31))
        pr = self.panel.prof
        u = getattr(self.panel.win, 'units', 'mm')
        p.setFont(QFont('Segoe UI', 8))
        if pr is None:
            p.setPen(QColor(150, 156, 166))
            p.drawText(self.rect(), Qt.AlignCenter, self.panel.status or 'Tools > Measure > Girth profile')
            p.end()
            return
        r = self._frame()
        p.setPen(QPen(QColor(60, 64, 72), 1))
        p.drawRect(r)
        g1 = float(max(pr['length'].max(), pr['tape'].max())) * 1.05
        for k in range(5):
            gv = g1 * k / 4
            y = r.bottom() - r.height() * k / 4
            p.setPen(QColor(120, 126, 136))
            p.drawText(QPointF(4, y + 4), F(gv, u).replace('.0 mm', ' mm'))
        for key, col, w in (('length', QColor(250, 204, 21, 160), 1.0), ('tape', QColor(56, 189, 248), 2.0)):
            path = QPainterPath()
            for i, (s, g) in enumerate(zip(pr['s'], pr[key])):
                q = self._xy(s, g)
                path.moveTo(q) if i == 0 else path.lineTo(q)
            p.setPen(QPen(col, w))
            p.drawPath(path)
        # position ticks
        s0, s1 = float(pr['s'].min()), float(pr['s'].max())
        p.setPen(QColor(120, 126, 136))
        for k in range(6):
            sv = s0 + (s1 - s0) * k / 5
            q = self._xy(sv, 0.0)
            p.drawLine(QPointF(q.x(), r.bottom()), QPointF(q.x(), r.bottom() + 3))
            p.drawText(QPointF(q.x() - 14, r.bottom() + 13), F(sv - s0, u).replace('.0 mm', ' mm'))
        # features, landmarks, given names: a landmark on a found feature shares one label; labels never overlap
        fm = p.fontMetrics()
        placed = []
        for f in self.panel.features():
            q = self._xy(f['s'], f['tape'])
            col = NAMED_COL if f.get('named') else FEAT_COL.get(f['kind'], QColor(220, 220, 220))
            p.setPen(QPen(col, 1.5))
            p.setBrush(col)
            p.drawEllipse(q, 3.5, 3.5)
            text = f.get('label', f['kind'])
            box = QRectF(q.x() + 5, q.y() - 16, fm.horizontalAdvance(text) + 4, 13)
            if box.right() > r.right():
                box.moveRight(q.x() - 5)
            for _ in range(12):
                if not any(box.intersects(b) for b in placed):
                    break
                box.translate(0, 14 if box.top() < r.center().y() else -14)
            placed.append(box)
            p.setBrush(QColor(24, 26, 31, 200))
            p.setPen(Qt.NoPen)
            p.drawRect(box)
            p.setPen(col)
            p.drawText(box, Qt.AlignLeft | Qt.AlignVCenter, text)
        if self.hover is not None and 0 <= self.hover < len(pr['s']):
            q = self._xy(pr['s'][self.hover], pr['tape'][self.hover])
            p.setPen(QPen(QColor(235, 235, 235, 160), 1, Qt.DashLine))
            p.drawLine(QPointF(q.x(), r.top()), QPointF(q.x(), r.bottom()))
            p.setPen(QColor(235, 235, 235))
            p.drawText(QPointF(min(q.x() + 6, r.right() - 210), r.top() + 12),
                       'at %s: tape %s, outline %s' % (F(pr['s'][self.hover] - pr['s'].min(), u),
                                                       F(pr['tape'][self.hover], u), F(pr['length'][self.hover], u)))
        p.setPen(QColor(150, 156, 166))
        p.drawText(QPointF(r.left(), self.height() - 6), 'position along the axis (click: pin a girth there; '
                                                         'right-click: name it)')
        p.end()

    def mouseMoveEvent(self, e):
        i = self._index_at(e.position().x())
        if i != self.hover:
            self.hover = i
            self.hovered.emit(i)
            self.update()

    def leaveEvent(self, e):
        self.hover = None
        self.hovered.emit(None)
        self.update()

    def mousePressEvent(self, e):
        i = self._index_at(e.position().x())
        if i is None:
            return
        if e.button() == Qt.LeftButton:
            self.clicked.emit(i)
        elif e.button() == Qt.RightButton:
            m = QMenu(self)
            a = m.addAction('Name this section...')
            b = m.addAction('Pin a girth here')
            got = m.exec(e.globalPosition().toPoint())
            if got is a:
                self.named.emit(i)
            elif got is b:
                self.clicked.emit(i)


class ProfilePanel(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.prof = None
        self.oid = None
        self.status = ''
        self.marks = {}            # template landmarks: name -> girth (refined)
        self.names = []            # the user's named sections: (name, index)
        self.pinned = {}           # profile index -> the pinned measure (so naming it renames it)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        row = QHBoxLayout()
        self.axis = QComboBox()
        for text, key in (('its long axis', 'long'), ('up (Y)', 'y'), ('X', 'x'), ('Z', 'z'), ("the view's up", 'view')):
            self.axis.addItem(text, key)
        row.addWidget(QLabel('Along'))
        row.addWidget(self.axis)
        b = QPushButton('Measure the selected object')
        b.clicked.connect(self.run)
        row.addWidget(b)
        self.tpl = QComboBox()
        self.tpl.addItem('Body plan: none', None)
        from ..tools.bodyplan import TEMPLATES
        for k in TEMPLATES:
            self.tpl.addItem('Body plan: ' + k, k)
        row.addWidget(self.tpl)
        b2 = QPushButton('Find landmarks')
        b2.setToolTip('Name the template\'s sections on this profile (measured again at 1 mm) and pin them')
        b2.clicked.connect(self.landmarks)
        row.addWidget(b2)
        self.info = QLabel('')
        self.info.setObjectName('dim')
        row.addWidget(self.info, 1)
        lay.addLayout(row)
        self.chart = ProfileChart(self)
        self.chart.hovered.connect(self._hover)
        self.chart.clicked.connect(lambda i: self.pin(i))
        self.chart.named.connect(self._name)
        lay.addWidget(self.chart, 1)

    # ---- measuring on the engine thread -------------------------------------------------------------------------------
    def _axis_vec(self):
        key = self.axis.currentData()
        if key in ('x', 'y', 'z'):
            return {'x': [1.0, 0.0, 0.0], 'y': [0.0, 1.0, 0.0], 'z': [0.0, 0.0, 1.0]}[key]
        if key == 'view':
            return [float(v) for v in self.win.viewport.cam.basis()[2]]
        return None

    def run(self, oid=None):
        vp = self.win.viewport
        if oid is None:
            if not vp.selected:
                self.win.status.showMessage('Select an object first.', 6000)
                return
            oid = vp.selected[0]
        self.oid = int(oid)
        ax = self._axis_vec()
        self.status = 'Measuring the profile...'
        self.prof, self.marks, self.names = None, {}, []
        self.chart.update()

        def fn(e):
            from ..tools import measure as M
            sc = e.hist.scene
            a = ax if ax is not None else M.long_axis(sc, oid)
            return M.girth_profile(sc, oid, a)
        self._call(fn, self._got)

    def _got(self, prof):
        self.prof = prof
        self.status = '' if prof is not None else 'No section through this object along that axis.'
        if prof is not None:
            self.info.setText('%d sections in %.1f s' % (len(prof['s']), prof['ms'] / 1000.0))
        self.chart.update()

    def _call(self, fn, then):
        eng = self.win.engine
        tok = object()

        def done(r, tk=tok):
            if r[0] is tk:
                eng.callDone.disconnect(done)
                then(r[1])
        eng.callDone.connect(done)
        eng.post('call', fn, tok)

    # ---- what the chart shows -----------------------------------------------------------------------------------------
    def features(self):
        from .units import format_length as F
        u = getattr(self.win, 'units', 'mm')
        out = []
        if self.prof is None:
            return out
        named = []
        for name, g in self.marks.items():
            s = float(g['pts'].mean(0) @ np.asarray(self.prof['axis']))
            named.append({'kind': name, 's': s, 'tape': g['tape'], 'named': True})
        step = float(self.prof.get('step', 0.005))
        for f in self.prof.get('features', []):
            if f['kind'] not in ('narrowest', 'widest', 'sudden change'):
                continue
            hit = [n for n in named if abs(n['s'] - f['s']) <= 3 * step]
            if hit:                              # the landmark and the feature are one section: one label
                hit[0].setdefault('also', []).append(f['kind'])
                continue
            out.append(dict(f, label='%s %s' % (f['kind'], F(f['tape'], u))))
        for n in named:
            extra = (' = %s' % ', '.join(n['also'])) if n.get('also') else ''
            out.append(dict(n, label='%s%s %s' % (n['kind'].capitalize(), extra, F(n['tape'], u))))
        for name, i in self.names:
            out.append({'kind': name, 's': float(self.prof['s'][i]), 'tape': float(self.prof['tape'][i]), 'named': True,
                        'label': name})
        return out

    def _hover(self, i):
        ms = getattr(self.win, 'measures', None)
        if ms is None:
            return
        if i is None or self.prof is None:
            ms.preview = None
        else:
            P = self.prof['pts'][i]
            ms.preview = {'pts': P, 'length': float(self.prof['length'][i]), 'tape': float(self.prof['tape'][i]),
                          'oid': self.oid}
        self.win.viewport.update()

    # ---- pins: a girth there, given names, the template's landmarks -----------------------------------------------------
    def pin(self, i, name=None, pts=None):
        """pin a girth at profile section i (or at a given section's own points: a refined landmark)"""
        from .measure_ui import _local, _dir_local, compute
        if self.prof is None:
            return
        P = self.prof['pts'][i] if pts is None else pts
        a = np.asarray(self.prof['axis'], float)
        oid = self.oid
        layer = self.win.measures

        def fn(e):
            sc = e.hist.scene
            o = sc.get(oid)
            m = {'kind': 'girth', 'oid': oid, 'a_local': _local(o, P[0])[0].tolist(), 'n_local': _dir_local(o, a).tolist()}
            if name:
                m['name'] = name
            return compute(sc, m)

        def got(m):
            if m and not m.get('error'):
                layer._add(m)
                if pts is None:
                    self.pinned[int(i)] = layer.items[-1] if layer.items else None
        self._call(fn, got)

    def _name(self, i):
        text, ok = QInputDialog.getText(self, 'Name this section', 'A name for this section (e.g. "neck base", '
                                                                   '"knot", "wrist"):')
        if ok and text.strip():
            self.name_section(i, text.strip())

    def name_section(self, i, text):
        """name the section at profile index i: a girth already pinned there (within a section) is renamed, not
        pinned again"""
        self.names = [(n, k) for n, k in self.names if abs(k - int(i)) > 1]
        self.names.append((text, int(i)))
        near = [k for k in self.pinned if abs(k - int(i)) <= 1 and self.pinned[k] in self.win.measures.items]
        if near:
            self.pinned[near[0]]['name'] = text
            self.win.viewport.update()
        else:
            self.pin(i, text)
        self.chart.update()

    def landmarks(self):
        key = self.tpl.currentData()
        if self.prof is None or key is None:
            self.win.status.showMessage('Measure a profile and choose a body plan first.', 6000)
            return
        prof, oid = self.prof, self.oid

        def fn(e):
            from ..tools import bodyplan as BP
            sc = e.hist.scene
            run = BP.TEMPLATES[key]
            marks = run(prof)
            return BP.refine(sc, oid, prof, marks, kinds=run.kinds)

        def got(fine):
            self.marks = fine or {}
            for name, g in self.marks.items():
                self.pin(0, name.capitalize(), pts=g['pts'])     # pinned at the landmark's own (refined) section
            self.chart.update()
        self._call(fn, got)


def install(win):
    panel = ProfilePanel(win)
    dock = QDockWidget('Girth profile', win)
    dock.setObjectName('girth_profile')
    dock.setWidget(panel)
    win.addDockWidget(Qt.BottomDockWidgetArea, dock)
    dock.hide()
    win.profile_panel, win.profile_dock = panel, dock
    return panel


def open_panel(win):
    win.profile_dock.show()
    win.profile_dock.raise_()
    if win.viewport.selected:
        win.profile_panel.run()
