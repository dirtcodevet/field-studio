"""Regions in the window (fs-shell, 2026-11-11; design 4.8 "Regions (face sets)": "Regions are painted with the Region
brush, filled by lasso, or made from a mask or a protected part; each has a name and colour. H hides the region under
the cursor, Shift+H isolates it, Alt+H shows all", and "Hide part of one object ... plus box hide (drag a box;
everything inside hides)").

  The Regions panel (a tab beside Properties): the selected object's regions (colour, name, nodes, shown), and
    From mask / From part (another object, within mm) / Lasso (drag an outline on the model; Through takes every
    depth) / Box hide (drag a rectangle) / Rename / Colour / Delete / Hide / Isolate / Show all / Overlay.
  Keys over the view: H hides the region under the cursor, Shift+H isolates it (again: the object whole). Over an object
    with no regions, H and Shift+H do what they did (hide the selected objects / show all). Alt+H shows every region,
    box and object again.
  Lasso: left-drag an outline; Shift+drag adds to the selected region, Ctrl+drag erases from it.
Everything that changes the model is a logged, undoable command (tools/regions.py); hiding is view state (not saved,
not in the hash), drawn by the tracers' view-clip list (render/raymarch.view_clips)."""
import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDockWidget, QDoubleSpinBox, QHBoxLayout,
                               QInputDialog, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..tools import regions as RG


def _post_view_refresh(eng):
    eng.post('view', 'hide', [])                 # a no-op view job: the scene's new view state is drawn and published


class RegionsUI(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.eng = win.engine
        self.vp = win.viewport
        self.mode = None                         # 'lasso' | 'box' while that tool is on
        self.path = []                           # the outline (lasso) or the two corners (box) being drawn
        self.mods = Qt.NoModifier
        self.info = {}                           # the panel's object: {oid, name, regions, counts, hidden}
        self.active = None                       # the selected region id
        self.last = None                         # the last command's report
        self.eng.callDone.connect(self._called)
        self._tok = {}

    # ---- engine calls ------------------------------------------------------------------------------------------------
    def call(self, fn, then=None):
        tok = object()
        self._tok[id(tok)] = (tok, then)
        self.eng.post('call', fn, tok)

    def _called(self, rr):
        ent = self._tok.pop(id(rr[0]), None)
        if ent is None or ent[0] is not rr[0]:
            return
        if ent[1] is not None:
            ent[1](rr[1])

    def oid(self):
        sel = list(getattr(self.vp, 'selected', []) or [])
        return sel[0] if sel else None

    def refresh(self, *_):
        try:
            self.panel.parts.refresh()
        except Exception as e:
            print('parts tree: %r' % (e,))
        oid = self.oid()
        if oid is None:
            self.info = {}
            self.panel.show_info({})
            return

        def read(e, oid=oid):
            sc = e.hist.scene
            if sc.index_of(oid) < 0:
                return {}
            o = sc.get(oid)
            return {'oid': oid, 'name': o.name, 'regions': {int(k): dict(v) for k, v in (o.regions or {}).items()},
                    'counts': RG.counts(sc, oid),
                    'hidden': set((getattr(sc, 'view_regions_hidden', None) or {}).get(oid, set())),
                    'boxes': len(getattr(sc, 'view_boxes', None) or []),
                    'overlay': bool(getattr(sc, 'view_region_overlay', False)),
                    'others': [(p.id, p.name) for p in sc.objects if p.id != oid]}
        self.call(read, self._got)

    def _got(self, info):
        self.info = info or {}
        if self.active not in (self.info.get('regions') or {}):
            self.active = None
        self.panel.show_info(self.info)

    def push(self, cmd):
        def after(_r=None):
            self.last = getattr(cmd, 'report', None)
            self.refresh()
        self.eng.post('push', cmd)
        self.call(lambda e: None, after)          # runs after the push (the engine's one job queue)

    def view(self, fn):
        self.call(lambda e: fn(e.hist.scene), lambda _r: (self.refresh()))
        _post_view_refresh(self.eng)

    # ---- the writers -------------------------------------------------------------------------------------------------
    def from_mask(self):
        oid = self.oid()
        if oid is not None:
            self.push(RG.RegionFromMask(oid))

    def from_part(self, part, within_mm):
        oid = self.oid()
        if oid is not None and part is not None:
            self.push(RG.RegionFromPart(oid, part, within=float(within_mm) / 1000.0))

    def lasso_done(self, poly, mods):
        oid = self.oid()
        if oid is None or len(poly) < 3:
            return
        mode, rid = 'new', None
        if mods & Qt.ShiftModifier and self.active is not None:
            mode, rid = 'add', self.active
        elif mods & Qt.ControlModifier:
            mode, rid = 'erase', self.active
        cam = self.vp.cam.copy()
        cam.section = None
        self.push(RG.RegionLasso(oid, cam.to_dict(), poly, self.vp.width(), self.vp.height(),
                                 through=self.panel.through.isChecked(), mode=mode, rid=rid))

    def box_done(self, a, b):
        cam = self.vp.cam.copy()
        W, H = self.vp.width(), self.vp.height()
        self.view(lambda sc: RG.box_hide(sc, cam, a[0], a[1], b[0], b[1], W, H))

    # ---- hide / isolate / show ---------------------------------------------------------------------------------------
    def hide(self, rid=None):
        oid, rid = self.oid(), self.active if rid is None else rid
        if oid is not None and rid is not None:
            self.view(lambda sc: RG.hide_region(sc, oid, rid))

    def isolate(self, rid=None):
        oid, rid = self.oid(), self.active if rid is None else rid
        if oid is not None and rid is not None:
            self.view(lambda sc: RG.isolate_region(sc, oid, rid))

    def show(self, rid):
        oid = self.oid()
        if oid is not None:
            self.view(lambda sc: RG.show_region(sc, oid, rid))

    def show_all(self):
        self.view(RG.show_all)

    def overlay(self, on):
        def f(sc, on=bool(on)):
            sc.view_region_overlay = on
            sc.touch(structural=True)
        self.view(f)

    def under_cursor(self, pos):
        """(object id, world point) the last frame shows under pos, or (None, None)"""
        fr = self.vp.frame
        if fr is None or fr.get('obj') is None:
            return None, None
        ob = fr['obj']
        h, w = ob.shape[:2]
        x = min(max(int(pos.x() * w / max(self.vp.width(), 1)), 0), w - 1)
        y = min(max(int(pos.y() * h / max(self.vp.height(), 1)), 0), h - 1)
        oid = int(ob[y, x])
        if oid < 0:
            return None, None
        return oid, self.vp.surface_at(pos)

    def has_regions(self, oid):
        for o in (self.win.snap or {}).get('objects', []):
            if o.get('id') == oid:
                return bool(o.get('regions'))
        return False

    def key_region(self, pos, isolate):
        oid, P = self.under_cursor(pos)
        if oid is None or P is None:
            return

        def f(e, oid=oid, P=np.asarray(P, float), iso=isolate):
            sc = e.hist.scene
            if sc.index_of(oid) < 0:
                return None
            rid = int(RG.region_at(sc, oid, [P])[0])
            (RG.isolate_region if iso else RG.hide_region)(sc, oid, rid)
            return (oid, rid)

        def done(r):
            if r is not None:
                o = next((d for d in (self.win.snap or {}).get('objects', []) if d.get('id') == r[0]), {})
                nm = ((o.get('regions') or {}).get(r[1]) or {}).get('name') if r[1] else 'the surface with no region'
                self.win.statusBar().showMessage('%s %s (Alt+H shows all)' % ('Isolated' if isolate else 'Hid',
                                                                          nm or 'region %d' % r[1]), 4000)
            self.refresh()
        self.call(f, done)
        _post_view_refresh(self.eng)

    # ---- the view: the outline being drawn ---------------------------------------------------------------------------
    def draw(self, p):
        if not self.path or self.mode is None:
            return
        p.setBrush(Qt.NoBrush)
        if self.mode == 'lasso':
            col = QColor(255, 210, 90) if not (self.mods & Qt.ControlModifier) else QColor(255, 110, 110)
            p.setPen(QPen(col, 1.6, Qt.DashLine))
            path = QPainterPath(QPointF(*self.path[0]))
            for q in self.path[1:]:
                path.lineTo(QPointF(*q))
            path.closeSubpath()
            p.drawPath(path)
        elif self.mode == 'box' and len(self.path) == 2:
            (x0, y0), (x1, y1) = self.path
            p.setPen(QPen(QColor(140, 200, 255), 1.6, Qt.DashLine))
            p.drawRect(int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))

    def set_mode(self, m):
        self.mode = m
        self.path = []
        self.panel.lasso_b.setChecked(m == 'lasso')
        self.panel.box_b.setChecked(m == 'box')
        if m == 'lasso':
            self.win.statusBar().showMessage('Region lasso: drag an outline on the model (Shift adds to the selected '
                                             'region, Ctrl erases). Esc ends.', 6000)
        elif m == 'box':
            self.win.statusBar().showMessage('Box hide: drag a rectangle; everything in it hides (Alt+H shows all). '
                                             'Esc ends.', 6000)
        self.vp.update()


class _Filter(QObject):
    """over the viewport: the lasso and box drags, and H / Shift+H over an object with regions"""

    def __init__(self, ui):
        super().__init__(ui)
        self.ui = ui

    def eventFilter(self, obj, ev):
        U = self.ui
        t = ev.type()
        if t in (QEvent.ShortcutOverride, QEvent.KeyPress) and ev.key() == Qt.Key_H:
            m = ev.modifiers() & ~Qt.KeypadModifier
            if m in (Qt.NoModifier, Qt.ShiftModifier):
                pos = U.vp.mapFromGlobal(U.vp.cursor().pos())
                oid, _ = U.under_cursor(QPointF(pos))
                if oid is not None and U.has_regions(oid):
                    if t == QEvent.KeyPress and not ev.isAutoRepeat():
                        U.key_region(QPointF(pos), isolate=bool(m & Qt.ShiftModifier))
                    ev.accept()
                    return True
            return False
        if U.mode is None:
            return False
        if t == QEvent.KeyPress and ev.key() == Qt.Key_Escape:
            U.set_mode(None)
            return True
        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            q = (float(ev.position().x()), float(ev.position().y()))
            U.mods = ev.modifiers()
            U.path = [q] if U.mode == 'lasso' else [q, q]
            U.vp.update()
            return True
        if t == QEvent.MouseMove and U.path and (ev.buttons() & Qt.LeftButton):
            q = (float(ev.position().x()), float(ev.position().y()))
            if U.mode == 'lasso':
                if np.hypot(q[0] - U.path[-1][0], q[1] - U.path[-1][1]) >= 3.0:
                    U.path.append(q)
            else:
                U.path[1] = q
            U.vp.update()
            return True
        if t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and U.path:
            path, U.path = U.path, []
            if U.mode == 'lasso':
                U.lasso_done(path, U.mods)
            elif abs(path[1][0] - path[0][0]) > 3 and abs(path[1][1] - path[0][1]) > 3:
                U.box_done(path[0], path[1])
            U.vp.update()
            return True
        return False


class RegionsPanel(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        L = QVBoxLayout(self)
        L.setContentsMargins(6, 6, 6, 6)
        from .parts_tree import PartsTree               # design 4.17: the inside parts as a tree (fs-contact's parts)
        self.parts = PartsTree(ui.win)
        self.parts.setVisible(False)
        L.addWidget(self.parts)
        self.title = QLabel('Select an object to see its regions.')
        self.title.setWordWrap(True)
        L.addWidget(self.title)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Region', 'Nodes', 'Shown'])
        self.tree.setRootIsDecorated(False)
        self.tree.itemSelectionChanged.connect(self._pick)
        self.tree.itemChanged.connect(self._toggled)
        self.tree.itemDoubleClicked.connect(lambda *_: self._rename())
        L.addWidget(self.tree, 1)

        def row(*ws):
            h = QHBoxLayout()
            for w in ws:
                h.addWidget(w)
            L.addLayout(h)

        def btn(text, tip, fn, checkable=False):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.clicked.connect(fn)
            return b
        self.lasso_b = btn('Lasso', 'Drag an outline on the model: a new region (Shift adds to the selected one, Ctrl '
                           'erases from it)', lambda v: ui.set_mode('lasso' if v else None), True)
        self.through = QCheckBox('Through')
        self.through.setToolTip('The lasso takes every depth (off: only the surface the view shows)')
        row(btn('From mask', 'The masked surface becomes a region', ui.from_mask), self.lasso_b, self.through)
        self.part = QComboBox()
        self.part.setToolTip('Another object (a protected part: the eyeball, a lid, a tooth)')
        self.within = QDoubleSpinBox()
        self.within.setRange(0.0, 50.0)
        self.within.setDecimals(2)
        self.within.setValue(0.5)
        self.within.setSuffix(' mm')
        self.within.setToolTip('The surface within this distance of the part becomes the region')
        row(btn('From part', 'Where this object touches or nears the chosen object becomes a region',
                lambda: ui.from_part(self.part.currentData(), self.within.value())), self.part, self.within)
        row(btn('Rename', 'Rename the selected region (or double-click it)', self._rename),
            btn('Colour', 'The selected region\'s colour', self._colour),
            btn('Delete', 'The selected region gone (its surface back to no region)', self._delete))
        self.box_b = btn('Box hide', 'Drag a rectangle: everything in it hides, every depth (Alt+H shows all)',
                         lambda v: ui.set_mode('box' if v else None), True)
        row(btn('Hide', 'Hide the selected region (H over it in the view)', lambda: ui.hide()),
            btn('Isolate', 'Only the selected region shown (Shift+H over it)', lambda: ui.isolate()),
            btn('Show all', 'Every region, box and object shown again (Alt+H)', ui.show_all), self.box_b)
        self.overlay = QCheckBox('Colour the regions in the view')
        self.overlay.toggled.connect(ui.overlay)
        L.addWidget(self.overlay)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        L.addWidget(self.status)
        self._filling = False

    def show_info(self, info):
        self._filling = True
        self.tree.clear()
        self.part.clear()
        if not info:
            self.title.setText('Select an object to see its regions.')
            self._filling = False
            return
        regs, cnt, hid = info.get('regions', {}), info.get('counts', {}), info.get('hidden', set())
        self.title.setText('<b>%s</b>: %d region%s%s' % (info['name'], len(regs), '' if len(regs) == 1 else 's',
                                                        ('; %d box hide%s' % (info['boxes'], '' if info['boxes'] == 1
                                                                              else 's')) if info.get('boxes') else ''))
        rows = [(0, {'name': '(no region)', 'colour': [0.55, 0.55, 0.55]})] if regs else []
        rows += sorted(regs.items())
        for k, v in rows:
            it = QTreeWidgetItem([v.get('name') or 'Region %d' % k, '%d' % cnt.get(k, 0) if k else '', ''])
            c = v.get('colour') or [0.7, 0.7, 0.7]
            it.setForeground(0, QColor.fromRgbF(*[min(max(float(x), 0.0), 1.0) for x in c[:3]]))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(2, Qt.Unchecked if k in hid else Qt.Checked)
            it.setData(0, Qt.UserRole, int(k))
            self.tree.addTopLevelItem(it)
            if k == self.ui.active:
                it.setSelected(True)
        for pid, nm in info.get('others', []):
            self.part.addItem(nm, pid)
        self.overlay.blockSignals(True)
        self.overlay.setChecked(bool(info.get('overlay')))
        self.overlay.blockSignals(False)
        r = self.ui.last
        if r:
            self.status.setText('Last: %s, %d nodes%s.' % ({'new': 'made region %s' % r.get('rid'),
                                                           'add': 'added to region %s' % r.get('rid'),
                                                           'erase': 'erased'}.get(r.get('mode'), ''), r.get('nodes', 0),
                                                          (', %d new bricks' % r['new_bricks'])
                                                          if r.get('new_bricks') else ''))
        self._filling = False

    def _pick(self):
        it = self.tree.selectedItems()
        self.ui.active = it[0].data(0, Qt.UserRole) if it else None
        ms = getattr(self.ui.win, 'measures', None)
        if ms is not None and getattr(ms, 'dims_on', False):
            ms.schedule_dims()                   # design D1: the overlay shows the selected region's size

    def _toggled(self, it, col):
        if self._filling or col != 2:
            return
        k = it.data(0, Qt.UserRole)
        if it.checkState(2) == Qt.Checked:
            self.ui.show(k)
        else:
            self.ui.hide(k)

    def _rename(self):
        oid, k = self.ui.oid(), self.ui.active
        if oid is None or not k:
            return
        cur = ((self.ui.info.get('regions') or {}).get(k) or {}).get('name', '')
        nm, ok = QInputDialog.getText(self, 'Rename region', 'Name:', text=cur)
        if ok and nm.strip():
            self.ui.push(RG.RegionProps(oid, k, name=nm.strip()))

    def _colour(self):
        oid, k = self.ui.oid(), self.ui.active
        if oid is None or not k:
            return
        c0 = ((self.ui.info.get('regions') or {}).get(k) or {}).get('colour') or [0.8, 0.5, 0.5]
        c = QColorDialog.getColor(QColor.fromRgbF(*c0[:3]), self, 'Region colour')
        if c.isValid():
            self.ui.push(RG.RegionProps(oid, k, colour=[c.redF(), c.greenF(), c.blueF()]))

    def _delete(self):
        oid, k = self.ui.oid(), self.ui.active
        if oid is not None and k:
            self.ui.push(RG.RegionDelete(oid, k))


def install(win):
    U = RegionsUI(win)
    U.panel = RegionsPanel(U)
    win.regions_ui = U
    dock = QDockWidget('Regions', win)
    dock.setObjectName('regions_panel')
    dock.setWidget(U.panel)
    dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
    win.addDockWidget(Qt.RightDockWidgetArea, dock)
    docks = getattr(win, 'docks', None)
    if isinstance(docks, dict) and 'Properties' in docks:
        win.tabifyDockWidget(docks['Properties'], dock)
        docks['Properties'].raise_()
        docks['Regions'] = dock
    vm = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'View' in a.menu().title()), None)
    if vm is not None:
        a = dock.toggleViewAction()
        a.setText('Regions (paint, lasso, from mask or part; hide, isolate, box hide)')
        vm.addAction(a)
        b = vm.addAction('Show all regions, boxes and objects')
        b.setShortcut(QKeySequence('Alt+H'))
        # fs-shell 11-12 (fs-bench-full F1, card run 2): application-wide like every key the window's _act makes, so
        # Alt+H works while another window of the app is the active one (a floating panel, a pop-out, a dialog left
        # open); with Qt's default (the window's own) it died there while Ctrl+Z still worked
        b.setShortcutContext(Qt.ApplicationShortcut)
        b.triggered.connect(U.show_all)
    else:
        sc = QShortcut(QKeySequence('Alt+H'), win)
        sc.setContext(Qt.ApplicationShortcut)
        sc.activated.connect(U.show_all)
    U.filter = _Filter(U)
    win.viewport.installEventFilter(U.filter)
    win.props.selectionChanged.connect(lambda *_: QTimer.singleShot(0, U.refresh))
    win.engine.sceneChanged.connect(lambda *_: QTimer.singleShot(0, U.refresh) if dock.isVisible() else None)
    U.dock = dock
    return U
