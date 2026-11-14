"""The tool strip above the view (fs-shell, 2026-11-14; the approved interface, sections 3.3 and 4 of
engine/fieldstudio/design/INTERFACE_DESIGN_2026-11-13.md: each tool's definition and
its modifiers had so many layers that a box, a slider or a linear / constant switch gave no idea what it changed).

From left to right, the same places for every shaping tool:
  1. the tool: its picture, its name (22 px), its kind ("FOLLOWS THE SURFACE · held"), one or two lines of what it does;
  2. Radius, Strength, Hardness, Smoothing: always these four slots, each with its unit and a plain line under it;
     Smoothing carries the three foot pictures (soft / even / sharp: the one falloff choice) and a live profile;
  3. Reach (Near side / Whole part / Through) and Stop at contact with its gap;
  4. the tool's own settings, at most three.
Object tools (Transform) show their own buttons instead of the four slots.

The values are today's brush settings (ui/panels.py ToolPanel: radius, strength, falloff, 'Protect surfaces behind',
collide / contact gap), written through its own setters, so every stroke, preset and per-brush memory keeps working.
Hardness and Smoothing's distance do not exist in today's brushes: their slots stay in place, dim, saying so, until
fs-tools' tools fill them (its registry). FS_LAYOUT=classic keeps the old context bar."""
import os

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
                               QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from . import glyphs as GL

STRIP_H = 118
OBJECT_TOOLS = ('select', 'move', 'rotate', 'scale')
# today's tools, in the approved words (until fs-tools' registry carries them): name, kind, what it does
SPEC = {
    'select': ('Select', 'OBJECTS · click', 'Click a form to select it (Shift adds); drag on empty space to turn the view.'),
    'move': ('Transform', 'OBJECTS · anchored', 'Move, turn or scale the whole form with its handles; type a value for an exact one.'),
    'rotate': ('Transform', 'OBJECTS · anchored', 'Move, turn or scale the whole form with its handles; type a value for an exact one.'),
    'scale': ('Transform', 'OBJECTS · anchored', 'Move, turn or scale the whole form with its handles; type a value for an exact one.'),
    'stretch': ('Shape', 'FACES FLAT · anchored', 'Pulls the surface where you press, as far as you drag.'),
    'grab': ('Grab', 'FOLLOWS THE SURFACE · anchored', 'Drags the surface under the brush with the mouse.'),
    'snake': ('Snake Hook', 'FOLLOWS THE SURFACE · path', 'Pulls a form out along the drag, keeping its thickness.'),
    'elastic': ('Elastic Grab', 'FOLLOWS THE SURFACE · anchored', 'Pulls like a lump of clay; the far side stays put.'),
    'swell': ('Swell / Settle', 'FOLLOWS THE SURFACE · held', 'Grows the surface out until it meets what is next to it, then stops at the gap.'),
    'settle': ('Swell / Settle', 'FOLLOWS THE SURFACE · held', 'Lets the surface settle onto what is under it, stopping at the gap.'),
    'smooth': ('Smooth', 'FOLLOWS THE SURFACE · held', 'Evens out bumps under the brush.'),
    'flatten': ('Flatten', 'FACES FLAT · held', 'Brings the surface onto a plane under the cursor.'),
    'mask': ('Mask / Protect', 'FOLLOWS THE SURFACE · held', 'Paints what the other tools must not change (Ctrl erases).'),
    'draw': ('Draw', 'FOLLOWS THE SURFACE · path', 'Pushes the surface out along its normal (Ctrl: carve).'),
    'carve': ('Carve', 'FOLLOWS THE SURFACE · path', 'Pushes the surface in along its normal (Ctrl: draw).'),
    'clay': ('Clay', 'FOLLOWS THE SURFACE · path', 'Builds up towards a plane above the surface, like laying clay.'),
    'inflate': ('Inflate', 'FOLLOWS THE SURFACE · held', 'Grows the form evenly under the brush.'),
    'crease': ('Crease', 'FOLLOWS THE SURFACE · path', 'Cuts a sharp groove along the stroke.'),
    'pinch': ('Pinch', 'FOLLOWS THE SURFACE · path', 'Draws the surface together towards the stroke.'),
}
STRENGTH_LINE = {'stretch': '1 = the surface follows the cursor', 'grab': '1 = the surface follows the cursor',
                 'elastic': '1 = the surface follows the cursor', 'snake': '1 = the surface follows the cursor',
                 'smooth': 'how much of a bump goes in one pass', 'mask': 'how opaque the paint is',
                 'swell': 'how fast it grows', 'settle': 'how fast it settles'}
FEET = [('soft', 'smooth', 'Soft: the change meets the surface gently'),
        ('even', 'linear', 'Even: a straight slope to the surface'),
        ('sharp', 'sharp', 'Sharp: a crisp foot, still no tear')]
SLOT_W = 162            # fs-shell-5 11-14: 150 -> 162 for the 13 px lines (the keys sit on the title row)
ROW2_H = 44            # fs-shell-2 11-14: the wrapped strip's second row (Reach on one line, the tool's own), was 100
#                        fs-shell-5 11-14: 46 -> 44 (ui2/check_1366: the view at 40.0 % of 1366 x 768, 1 px under its floor)


def _one_sentence(says, lbl):
    """fs-shell-3 11-14 (t54, the 1366 / 1920 check run on the registry rows: Merge's, Cut's and Connect's sentences cut
    at two lines): the card holds one plain sentence (design 3.3); a longer text gives its first clause there (up to
    ';', ': ' or '. ') when the whole would not fit the label's two lines, and keeps the whole in its hover"""
    fm = lbl.fontMetrics()
    w = 236                                       # the card's text at its widest (the tool block's 300 px less the picture)

    def lines(t):
        return fm.boundingRect(0, 0, w, 10000, int(Qt.TextWordWrap), t).height() // max(fm.lineSpacing(), 1)
    if not says or lines(says) <= 2:
        return says
    for sep in ('; ', ': ', '. '):
        if sep in says:
            head = says.split(sep)[0].rstrip('.') + '.'
            if lines(head) <= 2:
                return head
    # fs-shell-5 11-14 (ui2/check_1366: an elided sentence word-wrapped back to three lines at 13 px): cut at a word
    # until it wraps to two lines
    t = says
    while lines(t) > 2 and len(t) > 12:
        t = t[:max(12, int(len(t) * 0.9))].rsplit(' ', 1)[0].rstrip(',;:.') + '...'
    return t


def _lbl(text, px=12, bold=False, col='#e6e9ee', wrap=False):
    l_ = QLabel(text)
    f = QFont()
    f.setPixelSize(px)
    f.setBold(bold)
    l_.setFont(f)
    l_.setStyleSheet('color: %s; background: transparent;' % col)
    l_.setWordWrap(wrap)
    return l_


def profile_pixmap(kind, w=96, h=34, col=(240, 150, 60), dim=False):
    """the brush's profile across its width (core/brush.falloff_lut, the same curve the stroke uses)"""
    from ..core import brush as BR
    try:
        lut = BR.falloff_lut(kind)
    except Exception:
        lut = BR.falloff_lut('smooth')
    pm = QPixmap(w, h)
    pm.fill(QColor(30, 33, 40))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(QColor(70, 76, 88), 1))
    p.drawLine(QPointF(2, h - 4), QPointF(w - 2, h - 4))
    path = QPainterPath()
    n = len(lut)
    xs = np.linspace(-1.0, 1.0, 81)
    for i, x in enumerate(xs):
        y = float(lut[min(n - 1, int(abs(x) * (n - 1)))])
        px_, py_ = 4 + (x + 1) * 0.5 * (w - 8), (h - 4) - y * (h - 9)
        path.moveTo(px_, py_) if i == 0 else path.lineTo(px_, py_)
    c = QColor(*col) if not dim else QColor(110, 116, 126)
    p.setPen(QPen(c, 2))
    p.drawPath(path)
    p.end()
    return pm


def reach_pixmap(which, size=26, on=False):
    """front and back surfaces, and what the tool reaches: Near side / Whole part / Through"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = size / 26.0
    fg, ac = QColor(150, 156, 168), QColor(240, 150, 60) if on else QColor(200, 204, 212)
    p.setPen(QPen(fg, 1.4 * s))
    p.drawArc(QRectF(3 * s, 4 * s, 20 * s, 10 * s), 0, 180 * 16)          # the near surface
    p.drawArc(QRectF(3 * s, 12 * s, 20 * s, 10 * s), 0, 180 * 16)         # the back surface
    p.setPen(QPen(ac, 2.4 * s))
    if which in ('near', 'part', 'through'):
        p.drawArc(QRectF(3 * s, 4 * s, 20 * s, 10 * s), 30 * 16, 120 * 16)
    if which in ('part', 'through'):
        p.drawArc(QRectF(3 * s, 12 * s, 20 * s, 10 * s), 30 * 16, 120 * 16)
    if which == 'through':
        p.drawLine(QPointF(13 * s, 1 * s), QPointF(13 * s, 25 * s))
    p.end()
    return pm


def _legacy_hardness():
    """fs-shell-5 11-14: today's brushes have a Hardness once fs-tools defines it for them (warp/profile.legacy_curve);
    the timed brushes (ft_) always have one"""
    try:
        from ..warp.profile import legacy_curve  # noqa: F401
        return True
    except Exception:
        return False


_KEYS_CACHE = {}


def _slot_keys(win):
    """fs-shell-5 11-14: the keys of the slots as the keymap has them now (they can be rebound), e.g. {'radius': '[ ]'}
    with the drag in the hover; kept 30 s (a rebinding clears it: keymap._save)"""
    import time as _t
    c = _KEYS_CACHE.get(id(win))
    if c is not None and _t.perf_counter() - c[0] < 30.0:
        return c[1]
    out = {'radius': 'F', 'strength': 'Shift+F', 'hardness': 'Ctrl+F'}
    drag = {'radius': 'F-drag', 'strength': 'Shift+F-drag', 'hardness': 'Ctrl+F-drag'}
    try:
        from . import keymap as KM
        acts = KM.actions(win)

        def k(aid):
            a = acts.get(aid)
            if a is None or a.shortcut().isEmpty():
                return ''
            return KM.native(KM.norm(a.shortcut()))
        for slot, lo, hi in (('radius', 'Sculpt/Brush radius smaller', 'Sculpt/Brush radius larger'),
                             ('strength', 'Sculpt/Brush strength lower', 'Sculpt/Brush strength higher'),
                             ('hardness', 'Sculpt/Brush softer (softness / hardness)', 'Sculpt/Brush harder (softness / hardness)')):
            a_, b_ = k(lo), k(hi)
            if a_ or b_:
                out[slot] = '%s %s' % (a_, b_)
            out[slot + '_tip'] = 'Keys: %s / %s (a step each), or hold %s over the view and drag' % (
                a_ or '-', b_ or '-', drag[slot].replace('-drag', ''))
    except Exception:
        pass
    _KEYS_CACHE[id(win)] = (_t.perf_counter(), out)
    return out


class _Slot(QFrame):
    """one of the four shared controls: title, value, a picture (optional), a plain line"""

    def __init__(self, title, line, parent=None):
        super().__init__(parent)
        self.setFixedWidth(SLOT_W)
        self.setStyleSheet('QFrame { background: #20242b; border-radius: 6px; }')
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 5, 8, 5)
        v.setSpacing(2)
        self.title = _lbl(title, 14, True, '#eef1f5')         # fs-shell-5 11-14: 12 / 11 px greys -> 14 / 13 px, >= 8:1
        # fs-shell-5 11-14: the slot's keys on its title row, so its line keeps its words
        tr = QHBoxLayout()
        tr.setSpacing(4)
        tr.addWidget(self.title)
        tr.addStretch(1)
        self.keys = _lbl('', 12, True, '#9cceff')
        tr.addWidget(self.keys)
        v.addLayout(tr)
        self.row = QHBoxLayout()
        self.row.setSpacing(6)
        v.addLayout(self.row)
        self.line = _lbl(line, 13, False, '#c3cad3', wrap=True)
        v.addWidget(self.line)
        v.addStretch(1)

    def set_dim(self, dim, line=None):
        self.setEnabled(not dim)
        if line is not None:
            self.line.setText(line)


class ToolStrip(QFrame):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setObjectName('toolStrip')
        self.setFixedHeight(STRIP_H)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # never widens the window: it wraps (design 3.6)
        self.setStyleSheet('#toolStrip { background: #1a1d22; border-bottom: 1px solid #2a2f38; }')
        self.band = QFrame(self)                         # the category's colour band on the strip's left edge
        self.band.setGeometry(0, 0, 5, 2 * STRIP_H)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 6, 10, 6)
        outer.setSpacing(6)
        lay = QHBoxLayout()
        lay.setSpacing(8)
        outer.addLayout(lay)
        self.row2 = QHBoxLayout()
        self.row2.setSpacing(8)
        outer.addLayout(self.row2)
        self.row1 = lay
        # 1. the tool
        tb = QWidget()
        tb.setMinimumWidth(170)
        tb.setMaximumWidth(300)
        self.tool_block = tb
        th = QHBoxLayout(tb)
        th.setContentsMargins(0, 0, 0, 0)
        th.setSpacing(10)
        self.pic = QLabel()
        self.pic.setFixedSize(44, 44)
        th.addWidget(self.pic, 0, Qt.AlignTop)
        tv = QVBoxLayout()
        tv.setSpacing(1)
        self.name = _lbl('', 22, True)
        self.kind = _lbl('', 12, True, '#c3cad3')
        self.says = _lbl('', 13, False, '#dde2e8', wrap=True)
        self.says.setMaximumHeight(3 * self.says.fontMetrics().lineSpacing() + 2)   # fs-shell-2 11-14: at most 3 lines
        self.says.setMaximumWidth(236)            # fs-shell-5 11-14: the card's text column (_one_sentence measures it)
        for w in (self.name, self.kind, self.says):
            tv.addWidget(w)
        tv.addStretch(1)
        th.addLayout(tv, 1)
        lay.addWidget(tb, 0, Qt.AlignLeft)        # flush left for every tool; its own width (the slots follow it)
        # 2. the four shared controls
        self.shared = QWidget()
        sh = QHBoxLayout(self.shared)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.setSpacing(6)
        tp = win.tools
        self.s_radius = _Slot('Radius', 'the inner ring')
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.5, 120.0)
        self.radius.setDecimals(1)
        self.radius.setSuffix(' mm')
        self.radius.setSingleStep(0.5)
        self.radius.valueChanged.connect(lambda v: self._guard(lambda: tp.set_radius(v / 1000.0)))
        self.s_radius.row.addWidget(self.radius)
        self.s_strength = _Slot('Strength', '')
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.0, 2.0)
        self.strength.setDecimals(2)
        self.strength.setSingleStep(0.05)
        self._sfac = 1.0                                  # fs-shell-3 11-14: shown = rate x factor (mm/s, % a second)
        self.strength.valueChanged.connect(lambda v: self._guard(lambda: tp.set_strength(v / self._sfac)))
        self.s_strength.row.addWidget(self.strength)
        # fs-shell-2 11-14: two lines (three ran out of the slot); the whole sentence in its hover
        self.s_hard = _Slot('Softness / Hardness', 'the peak: soft dome to pin-point (coming)')
        self.s_hard.setToolTip('Hardness: the shape of the peak, from a soft dome to a pin-point. It comes with the new '
                               'tools (fs-tools); today\'s brushes shape their peak by their falloff.')
        self.hard_pic = QLabel()
        self.s_hard.row.addWidget(self.hard_pic)
        # fs-shell-3 11-14: the engine's timed brushes read Hardness (0-1) and a Smoothing distance (m): their own boxes
        self.hard_spin = QSpinBox()
        self.hard_spin.setRange(0, 100)
        self.hard_spin.setSuffix(' %')
        self.hard_spin.setToolTip('Hardness: 0 a soft dome, 100 the sharpest peak')
        # fs-shell-5 11-14: through ToolPanel.set_hardness (today's brushes take it as their falloff curve)
        self.hard_spin.valueChanged.connect(lambda v: self._guard(lambda: tp.set_hardness(v / 100.0)))
        self.hard_spin.hide()
        self.s_hard.row.addWidget(self.hard_spin)
        self.smooth_mm = QDoubleSpinBox()
        self.smooth_mm.setRange(0.0, 50.0)
        self.smooth_mm.setDecimals(1)
        self.smooth_mm.setSuffix(' mm')
        self.smooth_mm.setToolTip('Smoothing: how far beyond the radius the change fades into the surface')
        self.smooth_mm.valueChanged.connect(lambda v: self._guard(lambda: tp._set('smoothing', v / 1000.0)))
        self.smooth_mm.hide()
        self.s_smooth = _Slot('Smoothing', 'the foot: how the change meets the surface')
        self.feet = QButtonGroup(self)
        self.feet.setExclusive(True)
        for i, (word, kind, tip) in enumerate(FEET):
            b = QToolButton()
            b.setCheckable(True)
            b.setIcon(QPixmap(profile_pixmap(kind, 30, 18)))
            b.setIconSize(QSize(30, 18))
            b.setFixedSize(38, 26)                # fs-shell-2 11-14: the three feet fit the 150 px slot (wanted 181: cut)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=kind: self._guard(lambda: self._set_foot(k)))
            self.feet.addButton(b, i)
            self.s_smooth.row.addWidget(b)
        self.smooth_pic = QLabel()
        self.s_smooth.layout().insertWidget(2, self.smooth_mm)   # fs-shell-3 11-14: under the feet (ft brushes only)
        for s_ in (self.s_radius, self.s_strength, self.s_hard, self.s_smooth):
            sh.addWidget(s_)
        lay.addWidget(self.shared)
        # 3. Reach and Stop at contact
        self.reach_box = QFrame()
        self.reach_box.setStyleSheet('QFrame { background: #20242b; border-radius: 6px; }')
        # fs-shell-2 11-14: a grid, so that wrapped (narrow windows) Reach is one line: 'Reach  [Near side] [Whole part]
        # [Through]  [x] Stop at contact' (the strip's second row 46 px, not 100)
        from PySide6.QtWidgets import QGridLayout
        self.reach_grid = rv = QGridLayout(self.reach_box)
        rv.setContentsMargins(8, 5, 8, 5)
        rv.setHorizontalSpacing(4)
        rv.setVerticalSpacing(2)
        self.reach_title = _lbl('Reach', 14, True, '#eef1f5')
        rv.addWidget(self.reach_title, 0, 0, 1, 3)
        self.reach_btns = []
        self.reach = QButtonGroup(self)
        self.reach.setExclusive(True)
        for i, (k, word, tip) in enumerate((('near', 'Near side', 'Only the surface you are working on; anything behind it stays as it was'),
                                            ('part', 'Whole part', 'The near side and the part behind it'),
                                            ('through', 'Through', 'Everything along the way (comes with the new tools)'))):
            b = QToolButton()
            b.setCheckable(True)
            b.setText(word)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setIcon(QPixmap(reach_pixmap(k)))
            b.setIconSize(QSize(26, 26))
            b.setToolTip(tip)
            b.setEnabled(k != 'through')
            b.clicked.connect(lambda _=False, kk=k: self._guard(lambda: self._set_reach(kk)))
            self.reach.addButton(b, i)
            self.reach_btns.append(b)
            rv.addWidget(b, 1, i)
        self.contact = QCheckBox('Stop at contact')
        self.contact.setToolTip('The surface stops at the gap from anything it meets, never passing into it')
        self.contact.toggled.connect(lambda on: self._guard(lambda: tp._set('collide', bool(on))))
        rv.addWidget(self.contact, 2, 0, 1, 3)
        # 4. the tool's own (at most three), and the object tools' buttons
        self.own = QWidget()
        self.own_lay = QHBoxLayout(self.own)
        self.own_lay.setContentsMargins(0, 0, 0, 0)
        self.own_lay.setSpacing(6)
        # Reach and the tool's own: on the first row when it has room, else on a second (the strip grows to 2 rows)
        self.tail = QWidget()
        tl = QHBoxLayout(self.tail)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(8)
        tl.addWidget(self.reach_box)
        tl.addWidget(self.own, 1)
        lay.addWidget(self.tail, 1)
        self._wrapped = False
        self._r1_end = None                      # the first row's closing stretch while wrapped
        self._busy = False
        self._sig = None
        self._t = QTimer(self)
        self._t.timeout.connect(self.sync)
        self._t.start(150)
        self.sync()

    FULL_W = 300 + 4 * SLOT_W + 260 + 220             # one row: the tool, the four, Reach, the tool's own

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._wrap(self.width() < self.FULL_W)     # by the width only: the view never jumps when the tool changes
        # fs-shell-5 11-14 (ui2/check_1366: wrapped at 1366 the card shrank to its text, 170-192 px, and its sentence
        # ran to four lines): the card keeps its 300 px whenever the first row has room for it and the four slots
        self.tool_block.setMinimumWidth(300 if self.width() >= 300 + 4 * SLOT_W + 60 else 170)

    def _wrap(self, two):
        if two == self._wrapped:
            return
        self._wrapped = two
        (self.row1 if two else self.row2).removeWidget(self.tail)
        (self.row2 if two else self.row1).addWidget(self.tail, 1)
        if two:                                   # fs-shell-2 11-14: wrapped, the four follow the tool card (the
            self.row1.addStretch(1)               # card's cell took the row's spare width: a gap before Radius)
            self._r1_end = self.row1.itemAt(self.row1.count() - 1)
        elif self._r1_end is not None:
            self.row1.removeItem(self._r1_end)
            self._r1_end = None
        self._reach_compact(two)
        self._sig = None                          # fs-shell-5 11-14: the tool's own settings laid out again (beside / above)
        self.setFixedHeight(STRIP_H + ROW2_H if two else STRIP_H)
        # fs-shell-2 11-14: the sentence stays when the strip wraps (hidden, an object tool's card was an empty box)

    def _reach_compact(self, on):
        """fs-shell-2 11-14: Reach as one line (wrapped) or as its block (one row)"""
        g = self.reach_grid
        for w_ in [self.reach_title] + self.reach_btns + [self.contact]:
            g.removeWidget(w_)
        if on:
            g.addWidget(self.reach_title, 0, 0)
            for i, b in enumerate(self.reach_btns):
                b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                b.setIconSize(QSize(18, 18))
                g.addWidget(b, 0, 1 + i)
            g.addWidget(self.contact, 0, 4)
        else:
            g.addWidget(self.reach_title, 0, 0, 1, 3)
            for i, b in enumerate(self.reach_btns):
                b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                b.setIconSize(QSize(26, 26))
                g.addWidget(b, 1, i)
            g.addWidget(self.contact, 2, 0, 1, 3)

    # ---- writes, never echoed back while syncing ---------------------------------------------------------------------
    def _guard(self, fn):
        if not self._busy:
            fn()

    def _set_foot(self, kind):
        tp = self.win.tools
        if hasattr(tp, 'set_legacy_foot') and tp.set_legacy_foot(kind):   # fs-shell-5 11-14: the Hardness curve's foot
            return
        try:
            tp.falloff.setCurrentText(kind)          # ToolPanel's own control: its handler keeps the per-brush memory
        except Exception:
            tp._set('falloff', kind)
        if str(tp.settings.get('tool', '')).startswith('ft_'):   # fs-shell-3 11-14: the timed brushes' foot word
            tp._set('foot', {k: w for w, k, _t in FEET}.get(kind, 'soft'))

    def _set_reach(self, k):
        tp = self.win.tools
        on = k == 'near'
        if hasattr(tp, 'sheet_only'):
            tp.sheet_only.setChecked(on)              # 'Protect surfaces behind', per brush
        else:
            tp._set('sheet_only', on)

    # ---- the tool's own -----------------------------------------------------------------------------------------------
    def _rebuild_own(self, tool, field_mode):
        while self.own_lay.count():
            w = self.own_lay.takeAt(0).widget()
            if w is not None:
                w.hide()                         # fs-shell-2 11-14: a removed control leaves the screen now
                w.deleteLater()
        tp = self.win.tools
        if getattr(self.win, 'reg_tool', None) is not None:
            return self._rebuild_reg()

        def button(text, tip, fn, checked=None):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setStyleSheet('QToolButton { padding: 6px 10px; } QToolButton:checked { background: #3b6fd8; color: white; }')
            if checked is not None:
                b.setCheckable(True)
                b.setChecked(checked)
            b.clicked.connect(fn)
            self.own_lay.addWidget(b)
            return b
        if field_mode is not None:
            return
        if tool in OBJECT_TOOLS:
            for k, w in (('move', 'Move  W'), ('rotate', 'Rotate  R'), ('scale', 'Scale  Y')):
                button(w, 'Pick %s; then drag its handles on the form, or type a value' % w.split()[0].lower(),
                       lambda _=False, kk=k: tp.set_tool(kk), checked=(tool == k))
            rx = getattr(self.win, 'region_xform', None)
            if rx is not None and tool in ('move', 'rotate', 'scale') and rx.masked():
                return self._region_own(rx, tool)
            self.own_lay.addWidget(_lbl('While dragging: X / Y / Z lock an axis, type a number, Ctrl snaps.', 13,
                                        False, '#c3cad3', wrap=True), 1)
        elif tool == 'smooth':
            cur = tp.settings.get('layer', 'sculpt')
            for k, w, tip in (('sculpt', 'Form', 'Smooth the form'), ('detail', 'Detail', 'Smooth only the fine detail')):
                button(w, tip, lambda _=False, kk=k: tp.set_layer(kk), checked=(cur == k))
        elif tool in ('swell', 'settle'):
            for k, w in (('swell', 'Swell'), ('settle', 'Settle')):
                button(w, '', lambda _=False, kk=k: tp.set_tool(kk), checked=(tool == k))
        self.own_lay.addStretch(1)

    # ---- fs-shell-3 11-14: Transform on a masked object (TOOLSET_DESIGN 3.2; ui/region_xform.py) ---------------------
    def _region_own(self, rx, tool):
        """Moves [the unmasked part | the whole object], Soft edge (the registry's shared Smoothing for Transform, mm),
        Pivot (the registry's choices a region has); the line says what the drag will do"""
        def box(label, w, tip):
            b = QWidget()
            bl = (QHBoxLayout if self._wrapped else QVBoxLayout)(b)   # fs-shell-5: beside, in the wrapped row
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(1)
            bl.addWidget(_lbl(label, 13, True, '#e2e6eb'))
            w.setToolTip(tip)
            bl.addWidget(w)
            bl.addStretch(1)
            self.own_lay.addWidget(b)
            return w
        mv = QComboBox()
        mv.addItem('The unmasked part', True)
        mv.addItem('The whole object', False)
        mv.setCurrentIndex(0 if rx.on else 1)
        mv.currentIndexChanged.connect(lambda i: self._region_set(rx, 'on', bool(mv.itemData(i))))
        box('Moves', mv, 'The Mask protects part of this object: the drag moves only what it leaves free (its '
                         'sculpting with it), or the whole object')
        se = QDoubleSpinBox()
        se.setRange(0.0, 50.0)
        se.setDecimals(1)
        se.setSingleStep(0.5)
        se.setSuffix(' mm')
        se.setValue(rx.smoothing * 1e3)
        se.setMinimumWidth(90)
        se.valueChanged.connect(lambda v: self._region_set(rx, 'smoothing', float(v) / 1e3))
        box('Soft edge', se, 'Smoothing: how far inside the free part its move fades in from the masked border (the '
                             'masked part itself moves 0.0 mm)')
        pv = QComboBox()
        from .region_xform import PIVOTS
        for k, w in PIVOTS:
            pv.addItem(w, k)
        pv.setCurrentIndex([k for k, _w in PIVOTS].index(rx.pivot) if rx.pivot in [k for k, _w in PIVOTS] else 0)
        pv.currentIndexChanged.connect(lambda i: self._region_pivot(rx, pv.itemData(i)))
        box('Pivot', pv, 'Where the part turns about: its centre, the middle of its border (its root, as a nose turned '
                         'about where it meets the face), or a point you click on the form')
        st = rx.state()
        if st == 'all':
            line = 'All of this object is masked: nothing is free to move. Clear or shrink the mask, or move the whole object.'
        elif tool == 'scale':
            line = 'Scale acts on the whole object: a masked part moves and turns (Move W, Rotate R); its scale comes later.'
        elif rx.on:
            line = ('Drag the handles: the tinted part moves, its sculpting with it; the masked part stays (0.0 mm). '
                    'One step on release; Esc puts it back.')
        else:
            line = 'The whole object moves, the mask with it.'
        self.region_line = _lbl(line, 13, False, '#ffd278' if st == 'all' or tool == 'scale' else '#c3cad3', wrap=True)
        self.own_lay.addWidget(self.region_line, 1)

    def _region_set(self, rx, k, v):
        if self._busy:
            return
        setattr(rx, k, v)                         # 'on' is in the strip's sig: its line follows at the next sync
        self.win.viewport.update()

    def _region_pivot(self, rx, k):
        if self._busy:
            return
        rx.pivot = k
        g = getattr(self.win.viewport, 'gizmo', None)
        if g is not None:
            if k == 'surface point':
                g.set_pivot_mode('pick')          # the next click on the form sets it (Esc: back to the centre)
            elif g.pivot_mode == 'point' or g.picking_pivot:
                g.set_pivot_mode('centre')
        self.win.viewport.update()

    # ---- a tool run through fs-tools' own jobs (fs-shell-2 11-14; ui/toolreg.py, tools/registry.py) ----------------
    def _rebuild_reg(self):
        """its own settings from the registry, its action button, and an offer when the action was refused
        ('Not touching: Group instead')"""
        from . import toolreg as TR
        win = self.win
        # fs-shell-5 11-14: a setting that is one of the tool's quick three shows its keys
        try:
            from . import quick_settings as _QS
            qk_ = {}
            sl_ = _QS.own_slots(win) or []
            ks_ = _slot_keys(win)
            for i_, (q_, s_) in enumerate(sl_[:3]):
                if s_ is not None:
                    qk_[s_.key] = ks_[('radius', 'strength', 'hardness')[i_]]
        except Exception:
            qk_ = {}
        for s in TR.own_settings(win):
            box = QWidget()
            bl = (QHBoxLayout if self._wrapped else QVBoxLayout)(box)   # fs-shell-5: beside, in the wrapped row
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(1)
            v = TR.value(win, s)
            if s.kind == 'toggle':
                w = QCheckBox(s.label + (('   ' + qk_[s.key]) if s.key in qk_ else ''))
                w.setChecked(bool(v))
                w.toggled.connect(lambda on, ss=s: TR.set_value(win, ss, bool(on)))
            else:
                lr_ = QHBoxLayout()
                lr_.setSpacing(6)
                lr_.addWidget(_lbl(s.label, 13, True, '#e2e6eb'))
                if s.key in qk_:
                    lr_.addWidget(_lbl(qk_[s.key], 12, True, '#9cceff'))
                lr_.addStretch(1)
                bl.addLayout(lr_)
                lo, hi = tuple(s.range) if s.range else (0.0, 100.0)
                if s.kind == 'choice':
                    w = QComboBox()
                    for c in s.choices:
                        w.addItem(str(c), c)
                    w.setCurrentIndex(list(s.choices).index(v) if v in s.choices else 0)
                    w.currentIndexChanged.connect(lambda i, ww=w, ss=s: TR.set_value(win, ss, ww.itemData(i)))
                elif isinstance(s.default, int) and not isinstance(s.default, bool):
                    w = QSpinBox()
                    w.setRange(int(lo), int(hi))
                    w.setValue(int(v or 0))
                    w.valueChanged.connect(lambda x, ss=s: TR.set_value(win, ss, int(x)))
                else:
                    f = 1000.0 if s.unit == 'm' else 1.0          # metres shown in mm (the display units)
                    w = QDoubleSpinBox()
                    w.setDecimals(1 if f > 1 else 2)
                    w.setRange(lo * f, hi * f)
                    w.setSuffix(' mm' if f > 1 else (' %s' % s.unit if s.unit else ''))
                    w.setValue(float(v or 0.0) * f)
                    w.valueChanged.connect(lambda x, ss=s, ff=f: TR.set_value(win, ss, float(x) / ff))
                w.setMinimumWidth(96)
            w.setToolTip(s.says or s.label)
            bl.addWidget(w)
            bl.addStretch(1)                      # the label and its box together at the top
            self.own_lay.addWidget(box)
        if TR.is_brush(win):                     # fs-shell-3 11-14: a held brush acts by the press: no action button
            self.reg_act_btn = None
            self.reg_extra_btns = []
            self.reg_offer_btn = QToolButton()
            self.reg_offer_btn.hide()
            # fs-shell-5 11-14 (ui2/check_1366 at 1920: the strip there is 1578 px; the wrapped hint got 60 px and
            # needed 374 px of height, and pushed the four slots): one line, cut with ... to the room left, the whole
            # hint in its hover (the card's sentence above says the same)
            self.reg_line = self._hint_line(TR.hint(win))
            self.own_lay.addWidget(self.reg_line, 1)
            return
        act = QToolButton()
        act.setText(TR.action_text(win))
        act.setToolTip('Do it to the selection (one undo step)')
        act.setStyleSheet('QToolButton { padding: 7px 14px; background: #3b6fd8; color: white; border-radius: 5px; '
                          'font-weight: 600; } QToolButton:hover { background: #4a7de6; }')
        act.clicked.connect(self._reg_act)
        self.own_lay.addWidget(act)
        self.reg_act_btn = act
        self.reg_extra_btns = []
        for text, tip, fn in TR.extras(win):      # fs-shell-3 11-14: Add's 'Hollow the selected', 'More shapes ...'
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setStyleSheet('QToolButton { padding: 6px 10px; border: 1px solid #4a5260; border-radius: 5px; '
                            'background: #262b33; } QToolButton:hover { background: #303642; }')
            b.clicked.connect(lambda _=False, f=fn: f())
            self.own_lay.addWidget(b)
            self.reg_extra_btns.append(b)
        off = QToolButton()
        off.setStyleSheet('QToolButton { padding: 7px 12px; background: #3a2f1a; color: #ffd278; border: 1px solid '
                          '#a8812e; border-radius: 5px; }')
        off.clicked.connect(self._reg_offer)
        off.hide()
        self.own_lay.addWidget(off)
        self.reg_offer_btn = off
        self.reg_line = self._hint_line(TR.hint(win))   # fs-shell-5 11-14: one line, the whole in its hover
        self.own_lay.addWidget(self.reg_line, 1)

    @staticmethod
    def _hint_line(text):
        """fs-shell-5 11-14 (ui2/check_1366: a wrapped hint at 13 px set the strip's row height, 72-119 px): the tool's
        hint as one line cut with ... to the room left; the whole hint in its hover"""
        from .widgets import ElideLabel
        lb = ElideLabel(text, min_w=0)
        f_ = QFont()
        f_.setPixelSize(13)
        lb.setFont(f_)
        lb.setStyleSheet('color: #c3cad3; background: transparent;')
        lb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return lb

    def _reg_act(self):
        from . import toolreg as TR
        line = TR.act(self.win)
        if line:
            self.reg_line.setText(line)
            try:
                self.win.status.showMessage(line, 5000)
            except Exception:
                pass

    def _reg_offer(self):
        from . import toolreg as TR
        o = TR.offer(self.win)
        if o is not None:
            _text, job, args = o
            self.win.engine.tool_result = None
            self.win.engine.post(job, *args)
            self.reg_line.setText('Grouped: they select and move together; Ungroup undoes it.')
        self.reg_offer_btn.hide()

    def _sync_reg(self):
        from . import toolreg as TR
        b = getattr(self, 'reg_offer_btn', None)
        if b is None:
            return
        try:
            o = TR.offer(self.win)
        except RuntimeError:
            return
        if o is not None:
            if b.text() != o[0]:
                b.setText(o[0])
            if not b.isVisible():
                b.show()
                self.reg_line.setText('They do not touch, so Merge left them apart (nothing changed).')
        elif b.isVisible():
            b.hide()

    # ---- the view of today's settings --------------------------------------------------------------------------------
    def sync(self):
        win = self.win
        tp = win.tools
        vp = win.viewport
        tool = tp.tool
        fm = vp.tool_mode
        st = tp.settings
        reg = getattr(win, 'reg_tool', None)
        rx = getattr(win, 'region_xform', None)       # fs-shell-3 11-14: a masked object brings its region controls
        rxs = (rx.state(), rx.on) if (rx is not None and tool in ('move', 'rotate', 'scale')) else None
        sig = (tool, type(fm).__name__ if fm is not None else None, st.get('layer'), reg, rxs)
        name, kind, says = SPEC.get(tool, (tool, '', ''))
        if reg is not None:                       # fs-shell-2 11-14: a tool run through fs-tools' own jobs
            from . import toolreg as TR
            name, kind, says = TR.card(win)
        elif fm is not None:
            nm = type(fm).__name__
            name, kind, says = {'PencilMode': ('Pencil', 'FOLLOWS THE SURFACE · line', 'Draw a line across the form: a groove or a ridge, constant depth along it.'),
                                'CutMode': ('Cut', 'FACES FLAT · line', 'Draw the cut outline; it closes itself.')}.get(
                nm, (nm, '', ''))
        from .categories import CAT_COL
        cat = {'stretch': 'manipulate', 'mask': 'manipulate', 'move': 'manipulate', 'rotate': 'manipulate',
               'scale': 'manipulate', 'select': 'manipulate', 'swell': 'build', 'settle': 'build', 'smooth': 'corrective',
               'flatten': 'corrective'}.get(tool, 'build')
        if reg is not None:
            sp_ = TR.spec(reg[0])
            cat = sp_.category if sp_ is not None and sp_.category in CAT_COL else 'build'
        col = CAT_COL[cat]
        self._busy = True
        try:
            if sig != self._sig:
                self._sig = sig
                self.band.setStyleSheet('background: rgb(%d,%d,%d);' % col)
                try:
                    if reg is not None:
                        from .categories import _pic
                        # fs-shell-3 11-14: the column row's own picture (Cut is drawn by its glyph, not a 'p:' picture)
                        pk_ = {'copies': 'p:' + (reg[1] or 'duplicate'), 'add': 'p:shapes', 'cut': 'cut'}.get(
                            reg[0], 'p:' + reg[0])
                        # fs-shell-4 11-14 (Swell / Settle's card glyph box was EMPTY): the guess 'p:swell'
                        # is no picture; the card takes the column row's own picture key (Swell / Flatten: glyphs)
                        pk_ = (getattr(getattr(win, 'categories', None), 'row_pics', None) or {}).get(
                            'reg:%s%s' % (reg[0], ':' + reg[1] if reg[1] else ''), pk_)
                        self.pic.setPixmap(_pic(pk_, 44, col) if pk_.startswith('p:') else GL.glyph_pixmap(pk_, 44))
                    else:
                        self.pic.setPixmap(GL.glyph_pixmap(tool, 44))
                except Exception:
                    self.pic.clear()
                self.name.setText(name)
                self.kind.setText(kind)
                self.says.setText(_one_sentence(says, self.says))
                # fs-shell-5 11-14 (ui2/check_1366: Cut's sentence got 30 px of the 34 its two lines need): the card
                # holds the sentence's lines whatever its neighbours ask for
                fm_ = self.says.fontMetrics()
                n_ = fm_.boundingRect(0, 0, 236, 10000, int(Qt.TextWordWrap), self.says.text()).height() // max(
                    fm_.lineSpacing(), 1)
                self.says.setMinimumHeight(min(3, max(1, n_)) * fm_.lineSpacing() + 2)
                self.says.setToolTip(says)
                self._rebuild_own(tool, fm)
            from . import toolreg as _TRb
            ftb = reg is not None and _TRb.is_brush(win)     # fs-shell-3 11-14: a held brush row: the shared controls
            objt = tool in OBJECT_TOOLS or fm is not None or (reg is not None and not ftb)
            if reg is not None:
                self._sync_reg()
            try:                                  # the registry tool's cues on the model (or none)
                from . import toolreg as _TRc
                _TRc.update_cues(win)
            except Exception:
                pass
            if self.shared.isVisible() == objt:
                self.shared.setVisible(not objt)
                self.reach_box.setVisible(not objt)
                pass
            if not objt:
                r_mm = float(st.get('radius', 0.012)) * 1000.0
                if abs(self.radius.value() - r_mm) > 0.04 and not self.radius.hasFocus():
                    self.radius.setValue(r_mm)
                ft_ = str(tool).startswith('ft_')
                fac, suf, lo_, hi_, dec_ = _TRb.STRENGTH_SHOW.get(tool, (1.0, '', 0.0, 2.0, 2))
                if (fac, suf) != (self._sfac, self.strength.suffix()):
                    self._sfac = fac                          # the rate in display units (mm/s, % a second) for a timed brush
                    self.strength.setDecimals(dec_)
                    self.strength.setRange(lo_, hi_)
                    self.strength.setSuffix(suf)
                    self.strength.setSingleStep(0.5 if fac > 1 else 0.05)
                s = float(st.get('strength', 0.5)) * self._sfac
                if abs(self.strength.value() - s) > 0.5 * 10.0 ** (-self.strength.decimals()) and not self.strength.hasFocus():
                    self.strength.setValue(s)
                self.s_strength.line.setText({'ft_swell': 'how fast it swells while held',
                                              'ft_settle': 'how fast it settles while held',
                                              'ft_flatten': 'how much of the way to the plane each second'}.get(
                    tool, STRENGTH_LINE.get(tool, 'how far one pass moves the surface')))
                hard_ = ft_ or _legacy_hardness()        # fs-shell-5 11-14: live on every brush with a definition
                self.hard_spin.setVisible(hard_)
                self.hard_pic.setVisible(not hard_)
                self.smooth_mm.setVisible(ft_)
                self.s_smooth.line.setVisible(not ft_)
                if hard_:
                    hv = int(round(float(st.get('hardness', 0.0)) * 100))
                    if self.hard_spin.value() != hv and not self.hard_spin.hasFocus():
                        self.hard_spin.setValue(hv)
                    sv = float(st.get('smoothing', 0.0)) * 1000.0
                    if abs(self.smooth_mm.value() - sv) > 0.04 and not self.smooth_mm.hasFocus():
                        self.smooth_mm.setValue(sv)
                fo = st.get('falloff', 'smooth')
                if fo == 'custom' and st.get('foot'):      # fs-shell-5 11-14: a Hardness curve: its foot
                    fo = {'soft': 'smooth', 'even': 'linear', 'sharp': 'sharp'}.get(st.get('foot'), 'smooth')
                idx = {k: i for i, (_w, k, _t) in enumerate(FEET)}.get(fo, -1)
                b = self.feet.button(idx) if idx >= 0 else None
                if b is not None and not b.isChecked():
                    b.setChecked(True)
                key = ('hp', fo)
                if getattr(self, '_hp_key', None) != key:
                    self._hp_key = key
                    self.hard_pic.setPixmap(profile_pixmap(fo, 96, 30, col, dim=True))
                keys_ = _slot_keys(win)
                self.s_hard.set_dim(not hard_, '0 soft, 100 hard' if hard_ else
                                    'the peak: soft dome to pin-point (coming)')
                for sl_, k_ in ((self.s_radius, 'radius'), (self.s_strength, 'strength'), (self.s_hard, 'hardness')):
                    if sl_.keys.text() != keys_[k_]:
                        sl_.keys.setText(keys_[k_])
                        sl_.keys.setToolTip(keys_.get(k_ + '_tip', ''))
                so = bool(st.get('sheet_only', False))
                rb = self.reach.button(0 if so else 1)
                if rb is not None and not rb.isChecked():
                    rb.setChecked(True)
                c_on = bool(st.get('collide', False))
                if self.contact.isChecked() != c_on:
                    self.contact.setChecked(c_on)
        finally:
            self._busy = False


def install(win):
    """the strip above the view, in place of the old context bar (FS_LAYOUT=classic keeps it)"""
    if os.environ.get('FS_LAYOUT', '') == 'classic':
        return None
    strip = ToolStrip(win)
    # fs-shell-2 11-14 (design rule 5 + 3.6, measured at 1366 x 768): a panel open on the side narrowed the strip with
    # the view (the strip sat in the central widget): Smoothing ran past its edge, captions were cut. The strip is the
    # core: it lives in a fixed top dock that owns both top corners, so it spans over side panels; a panel narrows
    # only the view (rule 5: "the view narrows instead").
    from PySide6.QtWidgets import QDockWidget, QWidget as _QW
    dock = QDockWidget('Tool strip', win)
    dock.setObjectName('toolStripDock')
    dock.setTitleBarWidget(_QW())                  # no title bar: it is part of the window, not a panel
    dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
    dock.toggleViewAction().setVisible(False)      # never offered for hiding in the docks' right-click menu
    dock.setWidget(strip)
    win.addDockWidget(Qt.TopDockWidgetArea, dock)
    try:                                           # fs-shell-2 11-14: the cue job (toolset/cues.py on the engine thread)
        from . import toolreg as _TRi
        _TRi.install_cues(win)
        _TRi.hide_old_merges(win)                 # ONE Merge (fs-tools'): the fixed-width smooth union's entry goes
    except Exception as e_:
        if hasattr(win, '_install_failed'):           # fs-shell-5 11-14: shown to the user, not only logged
            win._install_failed('tool cues', e_)
        else:
            print('tool cues not installed: %r' % (e_,))
    win.setCorner(Qt.TopLeftCorner, Qt.TopDockWidgetArea)
    win.setCorner(Qt.TopRightCorner, Qt.TopDockWidgetArea)
    win.tool_strip_dock = dock
    cb = getattr(win, 'context_bar', None)
    if cb is not None:
        cb.hide()
    win.tool_strip = strip
    return strip
