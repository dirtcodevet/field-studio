"""The top bar (fs-shell-2, 2026-11-14; the approved interface, step 5; design INTERFACE_DESIGN_2026-11-13.md 3.1 and
mocks/mock_window.png): one row, 40 px.

- Field Studio, then four menus: File, Edit, View, Help. The window's other menus (Create, Sculpt, Tools, Render, and
  any a lane adds later, e.g. Library) are folded into those four as submenus, so every action keeps its key, its
  place in the search and in the Keys window; nothing core lives only in a menu (the core is the categories column).
- The search box (Space): it opens the search of every tool and action (shell.Palette) under the box.
- Live state on the right, in plain words (mock_window.png): Render ON / OFF (ON = the Look view, the model seen as it
  will render), fps or idle, "CPU view" when the card is not drawing, the card memory Field Studio holds, the undo
  memory. The engine numbers (frame time, RAM, field memory, Lipschitz L, quality) moved off the bar into its hover
  (design rule 3: no engine terms on screen).
The old row (workspace tabs, the Create shelf, the telemetry) is hidden, not removed: win.ws / win.telemetry still
work for the code and tests that use them. FS_LAYOUT=classic keeps the old top."""
import os

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy, QWidget

KEEP = ('File', 'Edit', 'Add', 'Sculpt', 'Object', 'View', 'Panels', 'Help')   # fs-shell-5 11-14: ui/menus.py's eight
# a folded menu -> (the menu it goes into, its title there)
FOLD = {'Create': ('Edit', 'Add objects (shapes, masses, cutters, garments)'),
        'Sculpt': ('Edit', 'Sculpt brushes (keys, radius, strength, mirror, mask)'),
        'Tools': ('Edit', 'Object tools (mutate, merge, protect, measure, replay)'),
        'Render': ('View', 'Render a still (full detail, converged)'),
        'Library': ('File', 'Library (mouth: teeth, gums, the set)')}
BAR_H = 40


def _clean(t):
    return t.replace('&', '').strip()


class _Folder(QObject):
    """keeps the menu bar at four menus: folds what is there now and any top-level menu added later"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.folded = {}                         # clean title -> the submenu it became
        self.where = {}                          # clean title -> where it is now ('Edit > Object tools (...)')
        self._pending = False

    def menus(self):
        return [a for a in self.win.menuBar().actions() if a.menu() is not None]

    def target(self, title):
        for a in self.menus():
            if _clean(a.menu().title()) == title:
                return a.menu()
        return None

    def fold_all(self):
        self._pending = False
        mb = self.win.menuBar()
        for a in self.menus():
            name = _clean(a.menu().title())
            if name in KEEP:
                continue
            into, title = FOLD.get(name, ('Object', name))   # fs-shell-5: a lane's later menu goes under Object
            dest = self.target(into)
            if dest is None:
                continue
            m = a.menu()
            mb.removeAction(a)
            have = self.folded.get(name)
            if have is not None and have is not m:   # a lane made the same menu again: its actions join the first
                for x in list(m.actions()):
                    m.removeAction(x)
                    have.addAction(x)
                continue
            m.setTitle(title)
            # its actions keep their old path ('Tools/Merge selected ...'): rebound keys are saved by path
            # (keymap.actions), the categories / view bar / search find actions by it, the Keys window groups by it
            m.setProperty('fs_menu_root', name)
            self.where[name] = '%s > %s' % (into, title)
            if name not in self.folded:
                dest.addSeparator()
            dest.addMenu(m)
            self.folded[name] = m
        # Quit stays last in File
        fm = self.target('File')
        if fm is not None:
            q = next((x for x in fm.actions() if x.text().replace('&', '') == 'Quit'), None)
            if q is not None and fm.actions()[-1] is not q:
                fm.removeAction(q)
                fm.addAction(q)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.ActionAdded and not self._pending:
            act = ev.action()
            if act is not None and act.menu() is not None and _clean(act.menu().title()) not in KEEP:
                self._pending = True
                QTimer.singleShot(0, self.fold_all)  # after the lane has filled it
        return False


SEARCH_FULL = '  Search tools and actions...   (Space)'
SEARCH_SHORT = '  Search...  (Space)'


class SearchBox(QPushButton):
    """looks like a search field; a click (or Space anywhere) opens the search of every tool and action"""

    def __init__(self, win):
        super().__init__(SEARCH_FULL)
        self.win = win
        self.setObjectName('topSearch')
        self.setCursor(Qt.IBeamCursor)
        self.setToolTip('Find any tool or action by name or by what you want ("crease", "hollow", "x-ray"). '
                        'Space opens it anywhere.')
        self.setStyleSheet('#topSearch { text-align: left; color: #b4bcc7; background: #14171b; '
                           'border: 1px solid #343a44; border-radius: 13px; padding: 3px 12px; }'
                           '#topSearch:hover { border-color: #3b82f6; }')
        self.setFixedHeight(28)
        self.setMinimumWidth(170)
        self.setMaximumWidth(520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clicked.connect(self.open)

    def open(self):
        p = getattr(self.win, 'palette', None)
        if p is None:
            return
        g = self.mapToGlobal(QPoint(0, self.height() + 4))
        p.open_at(QPoint(g.x() + 40, g.y() + 20))       # open_at places its corner 40 / 20 px up-left of the point


class LiveState(QLabel):
    """render state, fps, card memory, undo memory, in plain words; the engine numbers in its hover.
    fs-shell-6 11-14 (06_1366.png: the text was cut at its left at 1366 px): `compact` shortens it when the row is
    narrow (TopRight.fit): 1 the undo memory moves to the hover, 2 the card memory as held GB only, 3 Render and fps
    only. The hover always has the whole line."""

    def __init__(self, win):
        super().__init__('')
        self.win = win
        self.compact = 0
        self.on_longer = None                    # TopRight.fit_more: called when the text grows
        self.setObjectName('topLive')
        self.setStyleSheet('#topLive { color: #c9d0d8; padding: 0 10px; }')
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.t = QTimer(self)
        self.t.timeout.connect(self.refresh)
        self.t.start(500)
        self.refresh()

    def _tel(self, k):
        try:
            return self.win.telemetry.values[k].text()
        except Exception:
            return '-'

    @staticmethod
    def card_memory():
        """(held GB, card GB) from CuPy's own counters (no call into the card), or None when the card is not in use;
        GB here = 2^30 bytes, the way the card is sold (an 8 GB card read 9 in powers of ten)"""
        import sys
        # fs-shell-5 11-14 (t68: this readout froze the window 1.8 s): 'import cupy' here waited on Python's
        # import lock while the engine thread was still importing cupy at start. The modules are read only once their
        # import has finished (never imported, never waited on)
        cp = sys.modules.get('cupy')
        GS = sys.modules.get('fieldstudio.gpu.scene')
        for m in (cp, GS):
            if m is None or getattr(getattr(m, '__spec__', None), '_initializing', False):
                return None
        try:
            held = cp.get_default_memory_pool().total_bytes()
            total = getattr(GS._cp, 'card_total', 0) or 0
            return held / 2 ** 30, total / 2 ** 30
        except Exception:
            return None

    def parts(self, compact=None):
        """the live state's parts at a compactness (None: its own)"""
        c = self.compact if compact is None else compact
        snap = getattr(self.win, 'snap', None) or {}
        cpu = snap.get('render_device') == 'cpu' or os.environ.get('FIELDSTUDIO_GPU', '1') == '0'
        fps = self._tel('fps')
        idle = 'idle' in fps or fps in ('-', '0')
        # Render ON = the model seen as it will render (the Look view; mock_panels' "Render always on"); Clay,
        # Silhouette and the debug views are OFF
        look = getattr(self.win.viewport, 'mode', '') == 'shaded'
        parts = ['Render %s' % ('ON' if look else 'OFF'), 'idle' if idle else '%s fps' % fps.split()[0]]
        if c >= 3:
            return parts
        parts.append('CPU view' if cpu else None)
        cm = self.card_memory()
        if cm is not None:
            parts.append('card %.1f / %.0f GB' % cm if cm[1] and c < 2 else 'card %.1f GB' % cm[0])
        if c < 1:
            parts.append('undo %s' % self._tel('undo'))
        return [p for p in parts if p]

    def refresh(self):
        text = '  ·  '.join(self.parts())
        if text != self.text():
            longer = len(text) > len(self.text())
            self.setText(text)
            self.setMinimumWidth(0)              # QLabel's sizeHint grows to its minimum: measure the text alone
            self.setMinimumWidth(self.sizeHint().width())
            if longer and self.on_longer is not None:
                self.on_longer()
        self.setToolTip('Now: %s\nEngine numbers (the Inspect page):\nframe %s · RAM %s · field %s · Lipschitz L %s · %s\n'
                        'Card memory: what Field Studio holds on the card (its pool) of the card\'s total.'
                        % ('  ·  '.join(self.parts(0)), self._tel('ms'), self._tel('ram'), self._tel('field'),
                           self._tel('L'), self._tel('cpu')))


def keyboard_icon(w=22, h=14, col=(236, 240, 245)):
    """fs-shell-5 11-14: a small keyboard picture for the Keyboard map button"""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(QColor(*col), 1.3))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(QRectF(0.7, 0.7, w - 1.4, h - 1.4), 2.5, 2.5)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(*col))
    for row, n, y in ((0, 6, 3.2), (1, 5, 6.4)):
        for i in range(n):
            p.drawRect(QRectF(3.2 + i * 2.9 + row * 1.4, y, 1.7, 1.7))
    p.drawRect(QRectF(6.0, 9.6, w - 12.0, 1.8))
    p.end()
    return pm


TOP_BTN = ('QPushButton { color: #f2f4f7; background: #262c36; border: 1px solid #56606e; border-radius: 6px; '
           'padding: 3px 12px; font-size: 13px; } QPushButton:hover { border-color: #5b9bff; background: #2e3644; }'
           'QPushButton:checked { background: #24489a; border-color: #7fb0ff; color: white; }')   # 8:1 (was 5.9:1)


# fs-shell-6 11-14: the row's steps when it is narrow (1366 x 768 on the card needs step 3), each shorter than the
# one before; fit() takes the first that fits whole, so nothing in the row is ever cut:
#   (Keyboard map button, Quick keys button, the live state's compactness, the search box's text)
STEPS = (('  Keyboard map', 'Quick keys', 0, SEARCH_FULL),
         ('  Keyboard map', 'Quick keys', 1, SEARCH_FULL),      # the undo memory to the hover
         ('  Keys', 'Quick keys', 1, SEARCH_FULL),
         ('  Keys', 'Quick keys', 1, SEARCH_SHORT),
         ('  Keys', 'Quick keys', 2, SEARCH_SHORT),             # card memory as held GB
         ('  Keys', 'Quick', 3, SEARCH_SHORT))                  # Render and fps only


class TopRight(QWidget):
    """the search box, the keyboard map and quick keys buttons (fs-shell-5 11-14) and the live state:
    fills the menu bar's row right of the menus. fs-shell-6 11-14: at narrow widths the row steps down (STEPS) instead
    of cutting the live state at its left (06_1366.png)"""

    def __init__(self, win, mb):
        super().__init__()
        self.mb = mb
        self.win = win
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 6, 0)      # fs-shell-6: 24 -> 12, the stretch keeps the gap after the menus
        lay.setSpacing(10)
        self.search = SearchBox(win)
        self.live = LiveState(win)
        # the keyboard map one button click away on the top bar
        self.keys_btn = QPushButton('  Keyboard map')
        from PySide6.QtGui import QIcon
        self.keys_btn.setIcon(QIcon(keyboard_icon()))
        self.keys_btn.setIconSize(QSize(22, 14))
        self.keys_btn.setToolTip('Every key: what each one does, with the current tool, and a keyboard map (F1)')
        self.keys_btn.setStyleSheet(TOP_BTN)
        self.keys_btn.setFixedHeight(28)
        self.keys_btn.setFocusPolicy(Qt.NoFocus)
        self.keys_btn.clicked.connect(lambda: getattr(win, 'keys_window', None) is not None and win.keys_window.toggle())
        self.quick_btn = QPushButton('Quick keys')
        self.quick_btn.setCheckable(True)
        self.quick_btn.setToolTip('Show or hide the common keys on the view (brush size, strength, hardness, the tools, '
                                  'copy, mirror, undo)')
        self.quick_btn.setStyleSheet(TOP_BTN)
        self.quick_btn.setFixedHeight(28)
        self.quick_btn.setFocusPolicy(Qt.NoFocus)
        qk = getattr(win, 'quick_keys', None)
        self.quick_btn.setChecked(bool(qk is not None and qk.shown))
        self.quick_btn.toggled.connect(lambda on: getattr(win, 'quick_keys', None) is not None
                                       and win.quick_keys.set_shown(on))
        lay.addStretch(1)
        lay.addWidget(self.search, 3)
        lay.addSpacing(8)
        lay.addWidget(self.keys_btn)
        lay.addWidget(self.quick_btn)
        lay.addStretch(1)
        lay.addWidget(self.live)
        self.setFixedHeight(BAR_H - 4)
        self.step = -1
        self.live.on_longer = self.fit_more
        self.fit()

    def _apply(self, i):
        keys, quick, compact, search = STEPS[i]
        self.step = i
        for b, t in ((self.keys_btn, keys), (self.quick_btn, quick), (self.search, search)):
            if b.text() != t:
                b.setText(t)
            b.setMinimumWidth(0)
            b.setMinimumWidth(b.sizeHint().width())          # its text is whole, never squeezed
        self.live.compact = compact
        cb, self.live.on_longer = self.live.on_longer, None        # no re-entry from inside a fit
        try:
            self.live.refresh()
        finally:
            self.live.on_longer = cb
        self.live.setMinimumWidth(0)
        self.live.setMinimumWidth(self.live.sizeHint().width())

    def _need(self):
        lay = self.layout()
        lay.invalidate()
        return lay.minimumSize().width()

    def fit(self, start=0):
        """the first step from `start` whose row fits this width whole (the last step if none does)"""
        w = self.width()
        for i in range(start, len(STEPS)):
            self._apply(i)
            if self._need() <= w:
                return i
        return self.step

    def fit_more(self):
        """the live text grew: step down only if it no longer fits (never back up on a text change, so the row does
        not flicker as the fps comes and goes; a resize starts again from the full row)"""
        if self._need() > self.width() and self.step < len(STEPS) - 1:
            self.fit(self.step + 1)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if ev.oldSize().width() != ev.size().width():
            self.fit()

    def _menus_right(self):
        """the x the four menus end at, from the font (actionGeometry here would re-enter the menu bar's layout)"""
        fm = self.mb.fontMetrics()
        r = sum(fm.horizontalAdvance(a.text().replace('&', '')) + 26 for a in self.mb.actions() if a.isVisible())
        left = self.mb.cornerWidget(Qt.TopLeftCorner)
        return r + (left.sizeHint().width() if left is not None else 0)

    def sizeHint(self):
        # QMenuBar gives a corner widget its size hint (asked again on every resize): the row right of the menus
        w = self.mb.width() - self._menus_right() - 8
        return QSize(max(self.minimumSizeHint().width(), w), BAR_H - 4)

    def minimumSizeHint(self):
        # a floor only (the menu bar sizes this row from sizeHint): the search box's least, the two buttons
        return QSize(170 + 40 + self.keys_btn.sizeHint().width() + self.quick_btn.sizeHint().width(), BAR_H - 4)


def install(win):
    if os.environ.get('FS_LAYOUT', '') == 'classic':
        return None
    mb = win.menuBar()
    title = QLabel('Field Studio')
    f = QFont()
    f.setPixelSize(17)
    f.setBold(True)
    title.setFont(f)
    title.setStyleSheet('color: #e6e9ee; padding: 0 18px 0 12px;')
    title.setFixedHeight(BAR_H - 4)
    mb.setCornerWidget(title, Qt.TopLeftCorner)
    title.show()
    right = TopRight(win, mb)
    mb.setCornerWidget(right, Qt.TopRightCorner)
    right.show()
    mb.setMinimumHeight(BAR_H)
    from .theme import PANEL2
    # the four menu names centred in the 40 px row (the theme's 4 px items sat at the row's top beside the corners)
    mb.setStyleSheet('QMenuBar::item { padding: 11px 10px; background: transparent; }'
                     'QMenuBar::item:selected { background: %s; }' % PANEL2)
    fold = _Folder(win)
    fold.fold_all()
    mb.installEventFilter(fold)
    # the view's engine numbers (objects, bricks, Lipschitz L, frame) live on the inspect side now (design rule 3)
    vm = fold.target('View')
    dv = next((a.menu() for a in (vm.actions() if vm is not None else []) if a.menu() is not None
               and _clean(a.menu().title()) == 'Debug views'), None)
    if dv is not None:
        ei = dv.addAction('Engine numbers in the view (objects, bricks, Lipschitz L, frame)')
        ei.setCheckable(True)
        ei.setChecked(bool(getattr(win.viewport, 'engine_info', False)))
        ei.toggled.connect(lambda on: (setattr(win.viewport, 'engine_info', bool(on)), win.viewport.update()))
        win.act_engine_info = ei
    # fs-shell-5 11-14 (the first hands-on test: one very long, unsorted list of options): eight short menus
    # grouped by what the user is doing (ui/menus.py); every action keeps its id, key and search entry
    try:
        from . import menus as MN
        rep = MN.build(win)
        if rep.get('missing'):
            print('menus: %d entries not found (renamed by a lane?): %s' % (len(rep['missing']), rep['missing'][:8]))
    except Exception as e_:
        if hasattr(win, '_install_failed'):
            win._install_failed('the menus', e_)
        else:
            print('menus not regrouped: %r' % (e_,))
    old = getattr(win, 'top_bar', None)          # the workspace tabs, the Create shelf, the telemetry
    if old is not None:
        old.hide()
        old.toggleViewAction().setText('Old top row (workspace tabs, Create shelf, engine numbers)')
    win.topbar = right
    win.menu_folder = fold
    return right
