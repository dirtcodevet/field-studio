"""The toggled panels (fs-shell, 2026-11-14; the approved interface, rule 5 and section 5 of
engine/fieldstudio/design/INTERFACE_DESIGN_2026-11-13.md: one panel at a time, with a pin to keep several open;
a pop-out option; panels movable to the left, right, top or bottom of the screen, so each person can arrange
them as they choose).

- Tabs on the window's edge (right by default): Materials & Generators, Animation, Look / Render, then today's Scene,
  References, Layers, Regions and Hair panels. A tab opens its panel; one is open at a time.
- Every panel's title bar: Pin (stays open when another opens: several can be open), Pop out (its own window, e.g. on a
  second monitor; closing that window docks it back), Dock (left / right / top / bottom: the tabs and the panels move
  together), Close.
- Each person's arrangement is remembered (the app's settings: the edge, the pinned panels, the open ones, the popped-out
  ones and where they were).
- A docked panel never covers the categories, the tool strip or the view bar: the view narrows instead (Qt docks).
Materials & Generators and Animation are the window's half: fs-materials and fs-tools supply what runs inside them;
until then they say plainly what is coming and point to today's nearest controls.
FS_LAYOUT=classic keeps the old docks as they were."""
import os

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QPainter
from PySide6.QtWidgets import (QDockWidget, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QToolBar, QToolButton,
                               QVBoxLayout, QWidget)

AREAS = {'left': Qt.LeftDockWidgetArea, 'right': Qt.RightDockWidgetArea, 'top': Qt.TopDockWidgetArea,
         'bottom': Qt.BottomDockWidgetArea}
TB_AREAS = {'left': Qt.LeftToolBarArea, 'right': Qt.RightToolBarArea, 'top': Qt.TopToolBarArea,
            'bottom': Qt.BottomToolBarArea}
# (name on the tab, the existing dock's key in win.docks or None for a new one)
ORDER = [('Materials & Generators', None), ('Animation (not built yet)', None), ('Look / Render', 'Light'), ('Scene', 'Properties'),
         ('References', 'References'), ('Layers', 'Layers'), ('Regions', 'Regions'), ('Hair', 'Hair')]


class _Tab(QToolButton):
    """a tab on the edge: its name written along the edge when the tabs stand on the left or right"""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.label = text
        self.vertical = True
        self.setCheckable(True)
        self.setToolTip(text)
        self.setStyleSheet('QToolButton { background: #20242b; color: #dfe3ea; border: none; border-radius: 4px; margin: 2px; }'
                           'QToolButton:hover { background: #2e3440; }'
                           'QToolButton:checked { background: #3b6fd8; color: white; }')

    def sizeHint(self):
        fm = self.fontMetrics()
        w = fm.horizontalAdvance(self.label) + 22
        return QSize(30, w) if self.vertical else QSize(w, 28)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setPen(QColor('white') if self.isChecked() else QColor(223, 227, 234))
        if self.vertical:
            p.translate(self.width() / 2.0 + 5, self.height() / 2.0)
            p.rotate(90)
            r = QRectF(-self.height() / 2.0, -self.width() / 2.0 - 5, self.height(), self.width())
            p.drawText(r, Qt.AlignCenter, self.label)
        else:
            p.drawText(self.rect(), Qt.AlignCenter, self.label)
        p.end()

    def text(self):
        return ''


def _placeholder(title, lines, buttons=()):
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(12, 12, 12, 12)
    v.setSpacing(8)
    h = QLabel(title)
    f = QFont()
    f.setPixelSize(16)
    f.setBold(True)
    h.setFont(f)
    v.addWidget(h)
    for ln in lines:
        l_ = QLabel(ln)
        l_.setWordWrap(True)
        l_.setStyleSheet('color: #c8ced6;')
        v.addWidget(l_)
    for text, fn in buttons:
        b = QPushButton(text)
        b.clicked.connect(fn)
        v.addWidget(b)
    v.addStretch(1)
    return w


class PanelManager:
    def __init__(self, win):
        self.win = win
        self.docks = {}                      # tab name -> QDockWidget
        self.tabs = {}                       # tab name -> _Tab
        self.pinned = set()
        self.edge = 'right'
        self._restoring = False
        self.bar = QToolBar('Panels')
        self.bar.setObjectName('panelTabs')
        self.bar.setMovable(False)
        self.bar.setStyleSheet('QToolBar { background: #1b1e24; border: none; spacing: 2px; }')
        self._make_docks()
        for name, _k in ORDER:
            if name not in self.docks:
                continue
            t = _Tab(name)
            t.clicked.connect(lambda _=False, n=name: self.toggle(n))
            self.bar.addWidget(t)
            self.tabs[name] = t
        self.restore()
        self._t = QTimer(win)
        self._t.timeout.connect(self._sync_tabs)
        self._t.start(250)

    # ---- the docks -------------------------------------------------------------------------------------------------------
    def _make_docks(self):
        win = self.win
        items = {}
        try:
            from . import shell as SH
            items = {it.key: it for it in SH.items(win)}
        except Exception:
            pass

        def run(k):
            it = items.get(k)
            return (lambda: it.run()) if it is not None else (lambda: None)
        for name, key in ORDER:
            if key is not None:
                d = win.docks.get(key)
                if d is None:
                    continue
                if name == 'Look / Render':
                    self._wrap_look(d)
                d.setWindowTitle(name)
            elif name == 'Materials & Generators' and isinstance(getattr(win, 'docks', None), dict)                     and win.docks.get('Materials') is not None:
                # fs-shell-2 11-14: fs-materials' own dock (materials_ui.install, ee5db02b), adopted like Scene and
                # Look / Render: before, it opened beside Scene outside the panels (327 px of view, rule 5)
                d = win.docks['Materials']
                d.setWindowTitle(name)
            elif name == 'Materials & Generators':
                d = QDockWidget(name)
                body = None
                try:                        # fs-materials' panel (ui/materials_ui.py, its file): make_panel(win) -> QWidget
                    from .materials_ui import make_panel as _mat_panel
                    body = _mat_panel(win)
                except ModuleNotFoundError:
                    body = None
                except Exception as e_:
                    if hasattr(win, '_install_failed'):   # fs-shell-5 11-14: shown to the user, not only logged
                        win._install_failed('The Materials & Generators panel', e_)
                    else:
                        print('Materials & Generators panel not built: %r' % (e_,))
                d.setWidget(body if body is not None else _placeholder(name, [
                    'Apply a material to the selected object or a region of it: a library by family (Stone, Brick & '
                    'clay, Wood, Metal, Glass, Ground, Water, Fabric, Pigment), swatches rendered from each material, '
                    'its plain description, its finishes.',
                    'Seeded generators: stone surface, cracks and chips, weathering; for bodies the iris, skin and hair '
                    'generators. The same seed always gives the same result; Vary across copies gives each copy its own.',
                    'Coming from fs-materials (the library made real in Field Studio). Today: the selected object\'s '
                    'albedo and roughness are in Scene.'], [('Open Scene (albedo, roughness)', lambda: self.open('Scene'))]))
                d.setObjectName('panel_materials')
                win.addDockWidget(Qt.RightDockWidgetArea, d)
            else:
                d = QDockWidget(name)
                # fs-shell-5 11-14 (animation / rigging does not exist yet and is never faked): the tab says
                # so, and the panel says what exists today instead of listing what does not
                d.setWidget(_placeholder('Animation and rigging: not built yet', [
                    'There is no skeleton, pose or keyframe system in Field Studio yet; its design is approved and '
                    'waits to be built.',
                    'What exists today: Transform (W move, R rotate, Y scale) moves, turns and scales a form, or the part '
                    'of it the Mask leaves free, one undo step at a time.']))
                d.setObjectName('panel_animation')
                win.addDockWidget(Qt.RightDockWidgetArea, d)
            d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
            d.setTitleBarWidget(self._title_bar(name, d))
            d.topLevelChanged.connect(lambda fl, n=name: self._floated(n, fl))
            d.visibilityChanged.connect(lambda vis, n=name: self._vis(n, vis))
            self.docks[name] = d

    def _wrap_look(self, d):
        """Look / Render: the live quality, the view, the render still, then today's Light panel"""
        win = self.win
        light = d.widget()
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        row.addWidget(QLabel('Live quality'))
        for t, q in (('Fast', 0), ('Auto', 'auto'), ('Full', 2)):
            b = QPushButton(t)
            b.clicked.connect(lambda _=False, qq=q: win.set_quality(qq))
            row.addWidget(b)
        v.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('View'))
        for t, m in (('Clay', 'matcap'), ('Look', 'shaded'), ('Silhouette', 'silhouette')):
            b = QPushButton(t)
            b.clicked.connect(lambda _=False, mm=m: win.set_mode(mm))
            row2.addWidget(b)
        v.addLayout(row2)
        rs = QPushButton('Render this view, converged (full resolution)')
        rs.clicked.connect(self._render_still)
        v.addWidget(rs)
        v.addWidget(light, 1)
        d.setWidget(box)

    def _render_still(self):
        try:
            from .render_still import converge_window_view
            converge_window_view(self.win)
        except Exception as e:
            self.win.status.showMessage('Render still: %s' % e, 10000)

    def _title_bar(self, name, d):
        w = QFrame()
        w.setStyleSheet('QFrame { background: #262a32; } QToolButton { color: #dfe3ea; padding: 2px 6px; border: none; }'
                        'QToolButton:hover { background: #3a404b; } QToolButton:checked { background: #3b6fd8; color: white; }')
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 2, 4, 2)
        h.setSpacing(2)
        lb = QLabel(name)
        lb.setStyleSheet('color: #e6e9ee; font-weight: 600; background: transparent;')
        h.addWidget(lb, 1)
        pin = QToolButton()
        pin.setText('Pin')
        pin.setCheckable(True)
        pin.setToolTip('Keep this panel open when another opens')
        pin.toggled.connect(lambda on, n=name: self.pin(n, on))
        h.addWidget(pin)
        pop = QToolButton()
        pop.setText('Pop out')
        pop.setToolTip('This panel in its own window (closing the window docks it back)')
        pop.clicked.connect(lambda _=False, n=name: self.pop(n))
        h.addWidget(pop)
        dk = QToolButton()
        dk.setText('Dock')
        dk.setToolTip('Move the panels and their tabs to the left, right, top or bottom of the window')
        m = QMenu(dk)
        for e_ in ('left', 'right', 'top', 'bottom'):
            m.addAction(e_.capitalize(), lambda ee=e_: self.set_edge(ee))
        dk.setMenu(m)
        dk.setPopupMode(QToolButton.InstantPopup)
        h.addWidget(dk)
        cl = QToolButton()
        cl.setText('×')
        cl.setToolTip('Close')
        cl.clicked.connect(lambda _=False, n=name: self.close(n))
        h.addWidget(cl)
        w.pin_button = pin
        return w

    # ---- behaviour -------------------------------------------------------------------------------------------------------
    def toggle(self, name):
        d = self.docks[name]
        if d.isVisible() and not d.isFloating():
            self.close(name)
        else:
            self.open(name)

    def open(self, name):
        d = self.docks[name]
        for n, o in self.docks.items():
            if n != name and n not in self.pinned and not o.isFloating() and o.isVisible():
                o.hide()
        if not d.isFloating() and self.win.dockWidgetArea(d) != AREAS[self.edge]:
            self._add(d)
        d.show()
        d.raise_()
        self._fit_width(name, d)
        self.save()

    PANEL_W_SMALL = 360

    def _fit_width(self, name, d):
        """fs-shell-2 11-14 (step 6, the view's area): in a window under 1600 px a side panel opens 360 px wide (it was
        ~430: at 1366 the view kept 658 px); once per panel per session, so the user's own resizing is kept"""
        sized = self.__dict__.setdefault('_sized', set())
        if name in sized or d.isFloating() or self.edge not in ('left', 'right'):
            return
        try:
            from .categories import small_window
            if small_window(self.win):
                self.win.resizeDocks([d], [self.PANEL_W_SMALL], Qt.Horizontal)
                sized.add(name)
        except Exception:
            pass

    def _add(self, d):
        """fs-shell-2 11-14: a panel docked to the top goes under the tool strip (the strip spans the top: rule 5)"""
        self.win.addDockWidget(AREAS[self.edge], d)
        sd = getattr(self.win, 'tool_strip_dock', None)
        if self.edge == 'top' and sd is not None:
            self.win.splitDockWidget(sd, d, Qt.Vertical)

    def close(self, name):
        d = self.docks[name]
        d.hide()
        if d.isFloating():
            d.setFloating(False)
        self.save()

    def pin(self, name, on):
        (self.pinned.add if on else self.pinned.discard)(name)
        tb = self.docks[name].titleBarWidget()
        if tb is not None and tb.pin_button.isChecked() != bool(on):
            tb.pin_button.setChecked(bool(on))
        self.save()

    def pop(self, name):
        d = self.docks[name]
        d.setFloating(True)
        d.show()
        d.raise_()
        self.save()

    def _floated(self, name, floating):
        self.save()

    def _vis(self, name, visible):
        d = self.docks.get(name)
        if d is not None and not visible and d.isFloating() and not self._restoring:
            QTimer.singleShot(0, lambda: d.setFloating(False))   # closing a popped-out window docks it back
        if not self._restoring:
            QTimer.singleShot(0, self.save)

    def set_edge(self, edge):
        self.edge = edge
        win = self.win
        win.removeToolBar(self.bar)
        win.addToolBar(TB_AREAS[edge], self.bar)
        self.bar.setOrientation(Qt.Vertical if edge in ('left', 'right') else Qt.Horizontal)
        for t in self.tabs.values():
            t.vertical = edge in ('left', 'right')
            t.updateGeometry()
        self.bar.show()
        for d in self.docks.values():
            if not d.isFloating():
                vis = d.isVisible()
                self._add(d)
                d.setVisible(vis)
        self.save()

    def _sync_tabs(self):
        for n, t in self.tabs.items():
            on = self.docks[n].isVisible()
            if t.isChecked() != on:
                t.setChecked(on)

    # ---- remembered per person -----------------------------------------------------------------------------------------
    def _cfg(self):
        from .main_window import app_settings
        return app_settings()

    def save(self):
        if self._restoring:
            return
        try:
            c = self._cfg()
            c.setValue('panels/edge', self.edge)
            c.setValue('panels/pinned', ','.join(sorted(self.pinned)))
            c.setValue('panels/open', ','.join(n for n, d in self.docks.items() if d.isVisible() and not d.isFloating()))
            fl = []
            for n, d in self.docks.items():
                if d.isFloating() and d.isVisible():
                    g = d.geometry()
                    fl.append('%s@%d:%d:%d:%d' % (n, g.x(), g.y(), g.width(), g.height()))
            c.setValue('panels/floating', ';'.join(fl))
        except Exception:
            pass

    def restore(self):
        self._restoring = True
        try:
            c = self._cfg()
            edge = str(c.value('panels/edge', 'right'))
            pinned = [x for x in str(c.value('panels/pinned', '') or '').split(',') if x in self.docks]
            opened = [x for x in str(c.value('panels/open', 'Scene') or '').split(',') if x in self.docks]
            floating = [x for x in str(c.value('panels/floating', '') or '').split(';') if '@' in x]
        except Exception:
            edge, pinned, opened, floating = 'right', [], ['Scene'], []
        self.edge = edge if edge in AREAS else 'right'
        self.win.addToolBar(TB_AREAS[self.edge], self.bar)
        self.bar.setOrientation(Qt.Vertical if self.edge in ('left', 'right') else Qt.Horizontal)
        for t in self.tabs.values():
            t.vertical = self.edge in ('left', 'right')
        for n, d in self.docks.items():
            self._add(d)
            d.hide()
        for n in pinned:
            self.pin(n, True)
        for n in (opened or ['Scene'])[: 1 + len(pinned)]:
            if n in self.docks:
                self.docks[n].show()
        for rec in floating:
            n, g = rec.split('@', 1)
            if n in self.docks:
                try:
                    x, y, w, h = (int(v) for v in g.split(':'))
                    d = self.docks[n]
                    d.setFloating(True)
                    d.setGeometry(x, y, w, h)
                    d.show()
                except Exception:
                    pass
        self._restoring = False


def install(win):
    if os.environ.get('FS_LAYOUT', '') == 'classic':
        return None
    pm = PanelManager(win)
    win.panels = pm
    from PySide6.QtCore import QEvent, QObject

    class _FitOnResize(QObject):              # fs-shell-2 11-14: the panel open at start, once the window has its size
        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Resize:
                QTimer.singleShot(0, lambda: [pm._fit_width(n, d) for n, d in pm.docks.items() if d.isVisible()])
            return False
    pm._fit_filter = _FitOnResize(win)
    win.installEventFilter(pm._fit_filter)
    vm = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'View' in a.menu().title()), None)
    if vm is not None:
        sub = vm.addMenu('Panels')
        for n in pm.docks:
            a = QAction(n.replace('&', '&&'), win)       # fs-shell-5: '&' read as a mnemonic ate the word
            a.triggered.connect(lambda _=False, nn=n: pm.open(nn))
            sub.addAction(a)
    return pm
