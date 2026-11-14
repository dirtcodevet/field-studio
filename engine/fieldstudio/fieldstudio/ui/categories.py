"""The four categories, the window's left column (fs-shell, 2026-11-14; the approved interface, section 3.2 of
engine/fieldstudio/design/INTERFACE_DESIGN_2026-11-13.md; design: the categories prominent and bold enough to read
on the screen).

Add Objects / Manipulate / Building Tools / Corrective: each name bold at 21 px in its colour with a colour band on the
left edge; every tool one 38 px row (a picture of what it does, its plain name at 16 px, its key); the active tool
outlined in its category's colour; nothing folded, nothing scrolled (16 rows and 4 headings fit a 768 px screen).

Where the rows come from: fs-tools' registry (fieldstudio/tools/registry.py, TOOLS of ToolSpec)
when it exists. Until then TOOL_MAP below maps today's tools and actions onto the approved names; a tool
the toolset has not built yet shows dim, with "coming" and today's nearest in its hover line. Today's other brushes
(Draw, Clay, Carve, Inflate, Crease, Pinch, Grab, Snake, Select) stay reachable in a dim group at the foot, folded by
default (they are not in the approved toolset; they stay until the new tools replace them).
FS_LAYOUT=classic keeps the old shelf instead."""
import os

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy, QToolBar, QVBoxLayout, QWidget

from . import glyphs as GL

CATS = [('add', 'Add Objects', (64, 196, 190)), ('manipulate', 'Manipulate', (160, 122, 232)),
        ('build', 'Building Tools', (240, 150, 60)), ('corrective', 'Corrective', (232, 112, 152))]
CAT_COL = {k: c for k, _n, c in CATS}

# (category, name, what runs it: 'tool:<key>' / 'field:<label>' / 'action:<menu path>' / 'menu:<what>' / None = coming,
#  the tool keys that light this row, one plain sentence, picture key)
TOOL_MAP = [
    ('add', 'Shapes', 'menu:shapes', (), 'Add a shape: sphere, cube, cylinder ... , a mass, a cutter, or a new head',
     'p:shapes'),
    ('add', 'Duplicate', 'action:Edit/Duplicate object (with its sculpt)', (), 'A copy of the selected form, sculpt and all',
     'p:duplicate'),
    ('add', 'Mirror', 'action:Edit/Mirror-duplicate across X (world, with its sculpt)', (),
     'A mirrored copy of the selected form across the middle (right-click: other planes)', 'p:mirror'),
    ('add', 'Array', None, (), 'Copies along a line, a circle or a path, each a little different (coming from fs-tools)',
     'p:array'),
    ('manipulate', 'Transform', 'tool:move', ('move', 'rotate', 'scale'),
     'Move, turn or scale a whole form with handles; type a value for an exact one', 'move'),
    ('manipulate', 'Shape', 'tool:stretch', ('stretch',), 'Pull the surface where you press, as far as you drag', 'stretch'),
    ('manipulate', 'Place on surface', None, (), 'Set a form down on another, touching it (coming from fs-tools)', 'p:place'),
    ('manipulate', 'Mask / Protect', 'tool:mask', ('mask',), 'Paint what the other tools must not change', 'mask'),
    ('build', 'Pencil', 'field:Pencil ridge', (), 'Draw a line on the form: a groove, a ridge or paint', 'pencil ridge'),
    ('build', 'Swell / Settle', 'tool:swell', ('swell', 'settle'), 'Grow the surface out, or let it settle onto what is '
     'under it, stopping at contact', 'swell'),
    ('build', 'Cut', 'field:Cut (remove)', (), 'Cut along a line or a shape', 'cut'),
    ('build', 'Merge (Join)', 'action:Tools/Merge selected (sculpt kept; last settings)', (),
     'Join the selected forms into one, blended where they touch', 'p:merge'),
    ('build', 'Connect', None, (), 'Bridge two forms with a profile along a path (coming from fs-tools)', 'p:connect'),
    ('corrective', 'Smooth', 'tool:smooth', ('smooth',), 'Even out bumps under the brush', 'smooth'),
    ('corrective', 'Flatten', 'tool:flatten', ('flatten',), 'Bring the surface onto a plane that follows the cursor',
     'flatten'),
    ('corrective', 'Blend', None, (), 'Round the corner where two forms meet (coming from fs-tools; today: Create > '
     'Soften the joins)', 'p:blend'),
]
LEGACY = ['tool:select', 'tool:grab', 'tool:draw', 'tool:clay', 'tool:carve', 'tool:inflate', 'tool:crease', 'tool:pinch',
          'tool:snake', 'tool:elastic', 'tool:settle', 'field:Pencil valley', 'field:Cut and separate', 'field:Splice',
          'field:Fill and splice']


def _pic(key, size, col):
    """a picture for the rows that have no glyph yet (what the tool does to a form, drawn plainly)"""
    pm = QPixmap(size, size)
    pm.fill(GL.BG)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = size / 32.0
    fg = QColor(214, 218, 226)
    ac = QColor(*col)
    p.setPen(QPen(fg, 1.6 * s))
    p.setBrush(Qt.NoBrush)
    k = key[2:]
    if k == 'shapes':
        p.drawEllipse(QRectF(4 * s, 9 * s, 13 * s, 13 * s))
        p.drawRect(QRectF(16 * s, 12 * s, 12 * s, 12 * s))
        p.setPen(QPen(ac, 2.0 * s))
        p.drawLine(QPointF(24 * s, 3 * s), QPointF(24 * s, 9 * s)); p.drawLine(QPointF(21 * s, 6 * s), QPointF(27 * s, 6 * s))
    elif k == 'duplicate':
        p.drawRoundedRect(QRectF(5 * s, 5 * s, 15 * s, 15 * s), 3 * s, 3 * s)
        p.setPen(QPen(ac, 1.8 * s))
        p.drawRoundedRect(QRectF(12 * s, 12 * s, 15 * s, 15 * s), 3 * s, 3 * s)
    elif k == 'mirror':
        p.drawPie(QRectF(3 * s, 7 * s, 18 * s, 18 * s), 90 * 16, 180 * 16)
        p.setPen(QPen(ac, 1.8 * s))
        p.drawPie(QRectF(11 * s, 7 * s, 18 * s, 18 * s), -90 * 16, 180 * 16)
        p.setPen(QPen(fg, 1.0 * s, Qt.DashLine))
        p.drawLine(QPointF(16 * s, 3 * s), QPointF(16 * s, 29 * s))
    elif k == 'array':
        for i in range(3):
            p.setPen(QPen(fg if i == 0 else ac, 1.6 * s))
            p.drawRoundedRect(QRectF((3 + 9.5 * i) * s, 11 * s, 7.5 * s, 10 * s), 2 * s, 2 * s)
    elif k == 'place':
        p.drawLine(QPointF(3 * s, 25 * s), QPointF(29 * s, 25 * s))
        p.setPen(QPen(ac, 1.8 * s))
        p.drawEllipse(QRectF(10 * s, 9 * s, 12 * s, 12 * s))
        p.drawLine(QPointF(16 * s, 2 * s), QPointF(16 * s, 7 * s))
    elif k == 'merge':
        p.drawEllipse(QRectF(3 * s, 9 * s, 14 * s, 14 * s))
        p.setPen(QPen(ac, 1.8 * s))
        p.drawEllipse(QRectF(15 * s, 9 * s, 14 * s, 14 * s))
    elif k == 'connect':
        p.drawEllipse(QRectF(2 * s, 11 * s, 9 * s, 9 * s)); p.drawEllipse(QRectF(21 * s, 11 * s, 9 * s, 9 * s))
        p.setPen(QPen(ac, 2.2 * s))
        p.drawLine(QPointF(11 * s, 15.5 * s), QPointF(21 * s, 15.5 * s))
    elif k == 'blend':
        p.drawLine(QPointF(5 * s, 4 * s), QPointF(5 * s, 27 * s)); p.drawLine(QPointF(5 * s, 27 * s), QPointF(28 * s, 27 * s))
        p.setPen(QPen(ac, 2.0 * s))
        p.drawArc(QRectF(5 * s, 9 * s, 36 * s, 36 * s), 90 * 16, 90 * 16)
    p.fillRect(QRectF(0, 0, max(2.0, size / 16.0), size), ac)
    p.end()
    return pm


class _Row(QFrame):
    clicked = Signal()
    rightClicked = Signal()

    def __init__(self, name, key, says, pm, col, coming=False, height=38, font_px=16, parent=None):
        super().__init__(parent)
        self.col = col
        self.coming = coming
        self.active = False
        self.setFixedHeight(height)
        self.setCursor(Qt.ArrowCursor if coming else Qt.PointingHandCursor)
        self.setToolTip(says)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.setSpacing(10)
        self.pic = QLabel()
        self.pic.setPixmap(pm)
        self.pic.setFixedSize(pm.size())
        lay.addWidget(self.pic)
        self.name = QLabel(name)
        f = QFont()
        f.setPixelSize(font_px)
        self.name.setFont(f)
        self.name.setStyleSheet('color: %s;' % ('#7b818b' if coming else '#e6e9ee'))
        lay.addWidget(self.name, 1)
        self.key = QLabel('coming' if coming else (key or ''))
        fk = QFont()
        fk.setPixelSize(max(11, font_px - 3))
        self.key.setFont(fk)
        self.key.setStyleSheet('color: #b4bcc7;')            # fs-shell-5 11-14: was #7f8792 (5.8:1 at 13 px)
        lay.addWidget(self.key)
        self._style()

    def set_active(self, on):
        if on != self.active:
            self.active = on
            self._style()

    def _style(self):
        c = 'rgb(%d,%d,%d)' % self.col
        if self.active:
            self.setStyleSheet('_Row, QFrame { background: #2a2f38; border: 2px solid %s; border-radius: 6px; }'
                               'QLabel { border: none; background: transparent; }' % c)
        else:
            self.setStyleSheet('QFrame { background: transparent; border: 2px solid transparent; border-radius: 6px; }'
                               'QFrame:hover { background: #23272f; }'
                               'QLabel { border: none; background: transparent; }')

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and not self.coming:
            self.clicked.emit()
        elif e.button() == Qt.RightButton:
            self.rightClicked.emit()


class CategoryColumn(QWidget):
    """the left column: the four categories, every tool a row (runs today's tools until fs-tools' registry lands)"""

    def __init__(self, win, width=300, compact=False):
        super().__init__()
        self.win = win
        self.setObjectName('categoryColumn')
        self.setStyleSheet('#categoryColumn { background: #1b1e24; }')
        self.setFixedWidth(width)
        self.compact = compact
        if not compact:
            # fs-shell-6 11-14: the full column lives only in a window >= 900 px tall (small_window); its own minimum
            # (827 px) held a window jumped from 1920 x 1080 to 1366 x 768 at 911 px: Qt clamps the resize to the old
            # minimum before the column turns compact (_Fit, 200 ms later), and nothing asks it to shrink after
            self.setMinimumHeight(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(1)
        from . import shell as SH
        self.items = {it.key: it for it in SH.items(win)}
        self.rows = []                                   # (row, light keys)
        self.row_pics = {}                               # fs-shell-4 11-14: run -> the row's picture key
        # fs-shell-2 11-14 (step 6, measured): compact rows were 34 px: the column's 718 px + the top bar + the status bar
        # held the window at 792 px, taller than a 768 px screen; 30 px rows bring the column to ~630 (window ~700)
        fpx, hp, rh = (14, 16, 30) if compact else (16, 21, 38)
        for ck, cname, col in CATS:
            h = QLabel(cname)
            fh = QFont()
            fh.setPixelSize(hp)
            fh.setBold(True)
            h.setFont(fh)
            h.setStyleSheet('color: rgb(%d,%d,%d); border-left: 5px solid rgb(%d,%d,%d); padding: 4px 0 2px 8px;'
                            % (col + col))
            lay.addWidget(h)
            for cat, name, run, light, says, pk in _rows(win):
                if cat != ck:
                    continue
                it = self.items.get(run) if run and not run.startswith(('menu:', 'reg:')) else None
                coming = run is None or (run is not None and not run.startswith(('menu:', 'reg:')) and it is None)
                key = it.shortcut if it is not None else ''
                if run and run.startswith('reg:merge'):   # fs-shell-2 11-14: Ctrl+J runs this Merge now (ONE Merge)
                    mk = self.items.get('action:Tools/Merge selected (sculpt kept; last settings)')
                    key = mk.shortcut if mk is not None else key
                ra = (getattr(win, 'reg_key_actions', None) or {}).get(run)   # fs-shell-3 11-14: its key (REG_KEYS)
                if ra is not None and not ra.shortcut().isEmpty():
                    from PySide6.QtGui import QKeySequence as _QKS
                    key = ra.shortcut().toString(_QKS.NativeText)
                ps = 24 if compact else 28                # fs-shell-2 11-14: fits the compact 30 px row
                pm = _pic(pk, ps, col) if pk.startswith('p:') else self._glyph(pk, col, ps)
                row = _Row(name, key, says, pm, col, coming=coming, height=rh, font_px=fpx)
                if not coming:
                    row.clicked.connect(lambda r=run: self.run(r))
                if pk == 'p:mirror':
                    row.rightClicked.connect(self._mirror_menu)
                lay.addWidget(row)
                self.rows.append((row, set(light), run))
                if run:                                   # fs-shell-4 11-14: the strip's tool card shows this same picture
                    self.row_pics[run] = pk
            lay.addSpacing(3 if compact else 6)
        # today's other tools: reachable, dim, folded
        self.legacy_head = QLabel('▸ Today\'s other brushes')
        self.legacy_head.setCursor(Qt.PointingHandCursor)
        self.legacy_head.setStyleSheet('color: #b4bcc7; padding: 6px 0 2px 8px;')
        self.legacy_head.mousePressEvent = lambda e: self.fold_legacy(not self.legacy_box.isVisible())
        lay.addWidget(self.legacy_head)
        self.legacy_box = QWidget()
        lb = QVBoxLayout(self.legacy_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(0)
        for k in LEGACY:
            it = self.items.get(k)
            if it is None:
                continue
            gk = it.glyph or k.split(':', 1)[1]
            row = _Row(it.name, it.shortcut, it.desc or it.name, self._glyph(gk, (120, 126, 136), 22), (120, 126, 136),
                       height=28, font_px=13)
            row.clicked.connect(lambda r=k: self.run(r))
            lb.addWidget(row)
            self.rows.append((row, {k.split(':', 1)[1]} if k.startswith('tool:') else set(), k))
        lay.addWidget(self.legacy_box)
        lay.addStretch(1)
        self.fold_legacy(False)
        self._t = QTimer(self)
        self._t.timeout.connect(self.sync)
        self._t.start(200)
        self.sync()

    @staticmethod
    def _glyph(key, col, size=28):
        try:
            return GL.glyph_pixmap(key, size)
        except Exception:
            return _pic('p:' + key, size, col)

    def fold_legacy(self, show):
        self.legacy_box.setVisible(bool(show))
        self.legacy_head.setText(('▾ ' if show else '▸ ') + "Today's other brushes")
        self.legacy_head.setToolTip("Draw, Clay, Carve, Inflate, Crease, Pinch, Grab, Snake, Select ...: today's brushes, "
                                    "kept until the new tools replace them (they are not in the approved toolset)")

    def run(self, key):
        if key.startswith('reg:'):              # fs-shell-2 11-14: a tool run through fs-tools' own jobs
            from . import toolreg as TR
            TR.select(self.win, key)
            vp = getattr(self.win, 'viewport', None)
            if vp is not None:
                vp.setFocus()
            return None
        self.win.reg_tool = None
        if key == 'menu:shapes':
            return self._shapes_menu()
        it = self.items.get(key)
        if it is not None:
            it.run()
            vp = getattr(self.win, 'viewport', None)
            if vp is not None:
                vp.setFocus()

    def _menu_at(self, m):
        from PySide6.QtGui import QCursor
        m.popup(QCursor.pos())
        self._last_menu = m
        return m

    def _shapes_menu(self):
        m = QMenu(self)
        for path in ('File/New head (sphere + neck, 1 mm lattice, Mirror X; then photos)',
                     'File/New bust (a plain sphere, 1 mm lattice)'):
            it = self.items.get('action:' + path)
            if it is not None:
                m.addAction(it.name, it.run)
        m.addSeparator()
        groups = {'Shapes': [], 'Masses (block-in)': [], 'Cutters': []}
        for k, it in self.items.items():
            if not k.startswith('action:Create/'):
                continue
            n = it.name
            if n.endswith(' mass'):
                groups['Masses (block-in)'].append(it)
            elif n.endswith(' cutter'):
                groups['Cutters'].append(it)
            elif n in ('Sphere', 'Cube', 'Cylinder', 'Capsule', 'Cone', 'Torus', 'Ellipsoid', 'Slab'):
                groups['Shapes'].append(it)
        for g, its in groups.items():
            if not its:
                continue
            sub = m if g == 'Shapes' else m.addMenu(g)
            for it in its:
                sub.addAction(it.name, it.run)
        return self._menu_at(m)

    def _mirror_menu(self):
        m = QMenu(self)
        for k, it in self.items.items():
            if k.startswith('action:Edit/Mirror'):
                m.addAction(it.name, it.run)
        return self._menu_at(m)

    def sync(self):
        tool = getattr(getattr(self.win, 'tools', None), 'tool', None)
        reg = getattr(self.win, 'reg_tool', None)
        cur = ('reg:%s%s' % (reg[0], ':' + reg[1] if reg[1] else '')) if reg else None
        for row, light, run in self.rows:
            row.set_active(bool(run == cur) if cur else bool(tool in light))


def install(win):
    """the column on the window's left, in place of the old shelf (FS_LAYOUT=classic keeps the shelf)"""
    if os.environ.get('FS_LAYOUT', '') == 'classic':
        return None
    from PySide6.QtGui import QGuiApplication
    scr = QGuiApplication.primaryScreen()
    small = scr is not None and scr.availableGeometry().height() < 900     # until the window has its size
    col = CategoryColumn(win, width=240 if small else 300, compact=small)
    tb = QToolBar('Categories')
    tb.setObjectName('categories')
    tb.setMovable(False)
    tb.setOrientation(Qt.Vertical)
    win._categories_act = tb.addWidget(col)
    win.addToolBar(Qt.LeftToolBarArea, tb)
    old = getattr(win, 'shelf_bar', None)
    if old is not None:
        old.hide()
    win.categories, win.categories_bar = col, tb
    win.reg_tool = None                          # fs-shell-2 11-14: picking any other tool leaves the registry tool
    win.tools.toolChanged.connect(lambda _k: setattr(win, 'reg_tool', None))
    win._categories_fit = _Fit(win)
    win.installEventFilter(win._categories_fit)
    return col


def _rows(win):
    """fs-shell-2 11-14: the rows from fs-tools' registry (ui/toolreg.py: names, sentences, status; the tools the
    window runs through their own jobs) over today's runners and pictures; TOOL_MAP alone without the registry"""
    try:
        from . import toolreg as TR
        if TR.registry() is not None:
            return TR.rows(win, TOOL_MAP)
    except Exception as e_:
        print('tool registry not read: %r' % (e_,))
    return TOOL_MAP


def small_window(win):
    """design 3.6: the 1366 x 768 rules apply to a window under 1600 wide or 900 high"""
    return win.width() < 1600 or win.height() < 900


class _Fit(QObject):
    """fs-shell-2 11-14 (step 6): the column follows the WINDOW's size (240 px, 14 px names below 1600 x 900;
    300 px, 16 px above), rebuilt when the window crosses the line; it was chosen once from the screen's height"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.t = QTimer(self)
        self.t.setSingleShot(True)
        self.t.setInterval(200)
        self.t.timeout.connect(self.refit)

    def eventFilter(self, obj, ev):
        if obj is self.win and ev.type() == QEvent.Resize:
            self.t.start()
        return False

    def refit(self):
        win = self.win
        col = getattr(win, 'categories', None)
        small = small_window(win)
        if col is None or col.compact == small:
            return
        tb = win.categories_bar
        legacy = col.legacy_box.isVisible()
        new = CategoryColumn(win, width=240 if small else 300, compact=small)
        new.fold_legacy(legacy)
        col._t.stop()
        act = win._categories_act
        tb.removeAction(act)
        act.deleteLater()                        # takes the old column with it
        win._categories_act = tb.addWidget(new)
        win.categories = new
        # fs-shell-6 11-14: the window's minimum follows the new column now (it kept the old one's until the next
        # layout pass)
        tb.updateGeometry()
        win.layout().invalidate()
