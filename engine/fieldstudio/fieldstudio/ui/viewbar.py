"""The view bar (fs-shell, 2026-11-14; the approved interface, section 3.4 of
engine/fieldstudio/design/INTERFACE_DESIGN_2026-11-13.md).

Left: the selected object, its name and its own opacity (to see structural layers through objects: per object).
Right: Clay / Look / Silhouette; X-ray (0-100 %: real per-object compositing along the ray, gpu/viewmodes.py; not a
rim glow); Section (a cutting plane with a mm ruler); Measure (tape, girth, dimensions); Refs (the front and side
references); Mirror X (the global symmetry toggle). Until the Look / Render panel lands, the live quality and a
screenshot sit at the bar's right end.

Every control drives the window's existing action or setting (set_mode, set_xray, the menu actions), so keys, menus and
tests see one state. It replaces the old View row (kept hidden: set_mode / set_xray keep working);
FS_LAYOUT=classic keeps the old row."""
import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy, QSlider,
                               QToolButton, QVBoxLayout, QWidget)

from .widgets import ElideLabel

ROW_H = 38

VIEWS = (('matcap', 'Clay', 'The sculpting view: matcap and cavity (Shift+1)'),
         ('shaded', 'Look', 'The real material and light, refining as you work (Shift+2)'),
         ('silhouette', 'Silhouette', 'Flat, one colour, to check the outline against the references (Shift+3)'))


def _btn(text, tip, checkable=False):
    b = QToolButton()
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setStyleSheet('QToolButton { padding: 4px 7px; color: #e8ebef; border-radius: 4px; }'   # fs-shell-5: 10 -> 7 px
                    'QToolButton:hover { background: #2e3440; }'
                    'QToolButton:checked { background: #3b6fd8; color: white; }')
    return b


def _lbl(text, col='#c3cad3', px=13, bold=False):          # fs-shell-5 11-14: was #9aa3ae at 12 px (5-6:1)
    l_ = QLabel(text)
    f = QFont()
    f.setPixelSize(px)
    f.setBold(bold)
    l_.setFont(f)
    l_.setStyleSheet('color: %s; background: transparent;' % col)
    return l_


class ViewBar(QFrame):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setObjectName('viewBar')
        self.setFixedHeight(38)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # never widens the window: it compacts
        self.setStyleSheet('#viewBar { background: #1d2026; border-bottom: 1px solid #2a2f38; }')
        from . import shell as SH
        self.items = {it.key: it for it in SH.items(win)}
        # fs-shell-2 11-14 (step 6, measured at 1366 x 768 with a panel open): squeezed into one row the buttons were cut
        # ('Si...te', 'M...X'); one row when it holds every name whole, else two (Section ... Screenshot below)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 3, 10, 3)
        outer.setSpacing(0)
        lay = QHBoxLayout()
        lay.setSpacing(8)
        outer.addLayout(lay)
        self.row1 = lay
        self.row2 = QHBoxLayout()
        self.row2.setSpacing(8)
        outer.addLayout(self.row2)
        self._two = False
        # left: the selected object and its opacity (a long name is cut with ...; the whole name in its hover)
        self.obj = ElideLabel('Nothing selected', min_w=80)
        _f = QFont()
        _f.setPixelSize(13)
        _f.setBold(True)
        self.obj.setFont(_f)
        self.obj.setStyleSheet('color: #e6e9ee; background: transparent;')
        self.obj.setMinimumWidth(80)
        lay.addWidget(self.obj)
        self.op_lbl = _lbl('Opacity')
        lay.addWidget(self.op_lbl)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setFixedWidth(90)
        self.opacity.setToolTip("How see-through the selected object is in Clay and Look (a view setting: not saved, no "
                                "undo step)")
        self.opacity.valueChanged.connect(self._set_opacity)
        lay.addWidget(self.opacity)
        self.op_val = _lbl('100%')
        self.op_val.setFixedWidth(34)
        lay.addWidget(self.op_val)
        lay.addStretch(1)
        # right: the views, X-ray, Section, Measure, Refs, Mirror X
        self.views = QButtonGroup(self)
        self.views.setExclusive(True)
        for i, (mode, text, tip) in enumerate(VIEWS):
            b = _btn(text, tip, True)
            b.clicked.connect(lambda _=False, m=mode: win.set_mode(m))
            self.views.addButton(b, i)
            lay.addWidget(b)
        lay.addSpacing(6)
        lay.addWidget(_lbl('X-ray'))
        self.xray = QSlider(Qt.Horizontal)
        self.xray.setRange(0, 100)
        self.xray.setFixedWidth(100)
        self.xray.setToolTip('X-ray: every form turns that transparent and the parts inside show through (Alt+Q). '
                             'Real transparency: each object composited along the ray at its opacity.')
        self.xray.valueChanged.connect(self._set_xray)
        lay.addWidget(self.xray)
        self.xray_val = _lbl('off')
        self.xray_val.setFixedWidth(34)
        lay.addWidget(self.xray_val)
        lay.addSpacing(6)
        self.tail = QWidget()                  # Section ... Screenshot: on the first row, or the second when narrow
        lay.addWidget(self.tail)
        lay = QHBoxLayout(self.tail)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.section = _btn('Section', 'A cutting plane with a mm ruler', True)
        self.section.clicked.connect(lambda: self._run('action:View/Section (cut the view with a plane; thickness, gap, mm ruler)',
                                                       'action:Tools/Section tool (cut the view with a plane)'))
        lay.addWidget(self.section)
        self.measure = _btn('Measure', 'Tape, girth and dimensions')
        self.measure.clicked.connect(self._measure_menu)
        lay.addWidget(self.measure)
        self.refs = _btn('Refs', 'The front and side references shown (Alt+R)', True)
        self.refs.clicked.connect(lambda: self._run('action:View/Reference planes visible'))
        lay.addWidget(self.refs)
        self.mirror = _btn('Mirror X', 'The symmetry across the middle, for every tool (Alt+X)', True)
        self.mirror.clicked.connect(lambda: self._run('action:Sculpt/Mirror X on / off'))
        lay.addWidget(self.mirror)
        lay.addSpacing(10)
        # until the Look / Render panel: the live quality, a screenshot
        self.quality = QComboBox()
        for t, d in (('Fast', 0), ('Auto', 'auto'), ('Full', 2)):
            self.quality.addItem(t, d)
        self.quality.setCurrentIndex(1)
        self.quality.setToolTip('Live quality while you work (Auto: fast while moving, full when still)')
        self.quality.currentIndexChanged.connect(lambda i: win.set_quality(self.quality.itemData(i)))
        lay.addWidget(self.quality)
        shot = _btn('Screenshot', 'Save this view as a picture (Ctrl+S)')
        shot.clicked.connect(lambda: getattr(win, 'screenshot', lambda: None)())
        lay.addWidget(shot)
        lay.addStretch(1)                      # fs-shell-2 11-14: packed left (on its own row it was spread out)
        self.shot = shot
        # when narrow, the least needed go first (they stay in the menus): Screenshot, the quality, the value labels
        self._drop = [self.shot, self.quality, self.op_val, self.xray_val, self.op_lbl]
        self._t = QTimer(self)
        self._t.timeout.connect(self.sync)
        self._t.start(150)
        self.sync()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fit()

    def _need(self, layout):
        m = self.layout().contentsMargins()
        return layout.sizeHint().width() + m.left() + m.right()

    def fit(self):
        """one row if every name fits whole, else two; then, only if still short, the least needed go (they stay in
        the menus): Screenshot, the quality, the value labels"""
        for w_ in self._drop:
            w_.setVisible(True)
        if self._two:                          # try one row
            self.row2.removeWidget(self.tail)
            self.row1.addWidget(self.tail)
            self._two = False
        if self.width() < self._need(self.row1):
            self.row1.removeWidget(self.tail)
            self.row2.addWidget(self.tail)
            self._two = True
        for w_ in self._drop:                  # only from a row that is still short (two rows: row 2 has room)
            row = self.row2 if (self._two and self.tail.isAncestorOf(w_)) else self.row1
            if self.width() < self._need(row):
                w_.setVisible(False)
        h = ROW_H * (2 if self._two else 1) - (4 if self._two else 0)
        if self.height() != h:
            self.setFixedHeight(h)

    def _run(self, *keys):
        for k in keys:
            it = self.items.get(k)
            if it is not None:
                it.run()
                self.win.viewport.setFocus()
                return True
        return False

    def _measure_menu(self):
        from PySide6.QtGui import QCursor
        m = QMenu(self)
        for k, it in self.items.items():
            if k.startswith('action:Tools/Measure') or k.startswith('action:View/Dimensions') or \
                    k.startswith('action:Tools/Girth') or k == 'action:Tools/Clear all measures':
                m.addAction(it.name, it.run)
        m.popup(QCursor.pos())
        self._last_menu = m

    def _selected(self):
        return list(getattr(self.win.viewport, 'selected', []) or [])

    def _set_opacity(self, v):
        if getattr(self, '_busy', False):
            return
        vp = self.win.viewport
        for oid in self._selected():
            if v >= 100:
                vp.obj_opacity.pop(oid, None)
            else:
                vp.obj_opacity[oid] = v / 100.0
        self.op_val.setText('%d%%' % v)
        if v < 100 and vp.mode not in ('shaded', 'matcap'):
            self.win.set_mode('matcap')
        vp.request(False)

    def _set_xray(self, v):
        if getattr(self, '_busy', False):
            return
        w = self.win
        if hasattr(w, 'xray_slider'):
            w.xray_slider.setValue(int(v))          # the window's own slider (set_xray, Alt+Q's memory)
        else:
            w.set_xray(int(v))

    @staticmethod
    def _action_checked(it):
        try:
            a = it.run.__defaults__[0]
            return bool(a.isChecked())
        except Exception:
            return None

    def sync(self):
        w = self.win
        vp = w.viewport
        self._busy = True
        try:
            sel = self._selected()
            objs = {o['id']: o for o in (getattr(w, 'snap', None) or {}).get('objects', [])}
            names = [objs[o]['name'] for o in sel if o in objs]
            self.obj.setText(names[0] + (' +%d' % (len(names) - 1) if len(names) > 1 else '') if names else 'Nothing selected')
            has = bool(names)
            for x in (self.op_lbl, self.opacity, self.op_val):
                x.setEnabled(has)
            if has:
                v = int(round(100 * float(vp.obj_opacity.get(sel[0], 1.0))))
                if self.opacity.value() != v:
                    self.opacity.setValue(v)
                self.op_val.setText('%d%%' % v)
            idx = {m: i for i, (m, _t, _tip) in enumerate(VIEWS)}.get(vp.mode, -1)
            if idx >= 0:
                b = self.views.button(idx)
                if not b.isChecked():
                    b.setChecked(True)
            else:                                      # a debug view: none of the three lit
                self.views.setExclusive(False)
                for b in self.views.buttons():
                    b.setChecked(False)
                self.views.setExclusive(True)
            xv = int(round(100 * float(getattr(vp, 'xray', 0.0))))
            if self.xray.value() != xv:
                self.xray.setValue(xv)
            self.xray_val.setText('off' if xv == 0 else '%d%%' % xv)
            self.xray.setEnabled(vp.mode in ('shaded', 'matcap'))
            for btn, key in ((self.refs, 'action:View/Reference planes visible'),
                             (self.mirror, 'action:Sculpt/Mirror X on / off')):
                it = self.items.get(key)
                st = self._action_checked(it) if it is not None else None
                if st is not None and btn.isChecked() != st:
                    btn.setChecked(st)
            sec = getattr(vp.cam, 'section', None) is not None
            if self.section.isChecked() != sec:
                self.section.setChecked(sec)
        finally:
            self._busy = False


def install(win):
    if os.environ.get('FS_LAYOUT', '') == 'classic':
        return None
    bar = ViewBar(win)
    cl = win.centralWidget().layout()
    vp_i = cl.indexOf(win.viewport)
    cl.insertWidget(vp_i, bar)
    old = win.mode_combo.parentWidget() if hasattr(win, 'mode_combo') else None
    if old is not None and old is not bar:
        old.hide()
    win.view_bar = bar
    return bar
