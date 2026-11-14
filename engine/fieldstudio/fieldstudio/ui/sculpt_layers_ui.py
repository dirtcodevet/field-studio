"""The Layers tab (fs-shell, 2026-11-11; design 4.10: "Each object gets a layer stack: name, strength slider (-1 to 2),
on / off, solo, merge down, and record (strokes go to the selected layer) ... A lid pass on its own layer can be dialled
from 0 to 1 for comparison, with nothing lost"; snapshots to compare and go back to).

  Record a new layer: from now on the strokes on the selected object go into it; Stop: the layer is made (what the
  strokes changed since, from the history's own records: tools/sculpt_layers.capture).
  Each layer: on / off, a strength slider -1 to 2 (applied when the slider is let go: a dial re-distances the changed
  bricks), Solo (only this layer on), Merge into base, Delete (its change taken out).
  Snapshots: Take (a named state of the object), Restore (all), Restore where masked, Delete.
A dial is a blend of two fields, not the strokes at another strength (the brushes are not linear): the tooltip says so."""
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QDockWidget, QDoubleSpinBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from ..tools import sculpt_layers as SL

DIAL_TIP = ('A layer at a strength blends the field before and after its strokes: 0 = without them, 1 = as made, 2 = '
            'twice the change. It is a blend of two fields, not the strokes run again at another strength (the '
            'brushes are not linear).')


class LayersUI(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.eng = win.engine
        self.vp = win.viewport
        self.info = {}
        self._tok = {}
        self.eng.callDone.connect(self._called)

    def oid(self):
        sel = list(getattr(self.vp, 'selected', []) or [])
        return sel[0] if sel else None

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

    def refresh(self, *_):
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
            import copy
            return {'oid': oid, 'name': o.name, 'layers': copy.deepcopy(o.layers), 'snapshots': copy.deepcopy(o.snapshots)}
        self.call(read, self._got)

    def _got(self, info):
        self.info = info or {}
        self.panel.show_info(self.info)

    def push(self, cmds, label=None):
        cmds = [c for c in (cmds if isinstance(cmds, (list, tuple)) else [cmds]) if c is not None]
        if not cmds:
            return
        if len(cmds) > 1:
            cmds[-1].group_n = len(cmds)
            cmds[0].group_first_n = len(cmds)
        for c in cmds:
            self.eng.post('push', c)
        self.call(lambda e: None, lambda _r: self.refresh())

    # ---- actions -----------------------------------------------------------------------------------------------------
    def record(self):
        oid = self.oid()
        if oid is None:
            return
        if any(L.get('recording') for L in self.info.get('layers', [])):
            self.stop()
            return
        name, ok = QInputDialog.getText(self.panel, 'Record a layer', 'Name of the new layer:',
                                        text='Layer %d' % (len(self.info.get('layers', [])) + 1))
        if ok:
            self.push(SL.LayerNew(oid, name.strip() or None))
            self.win.statusBar().showMessage('Recording into "%s": your strokes on %s go into it. Stop when the pass '
                                             'is done.' % (name, self.info.get('name', 'the object')), 8000)

    def stop(self):
        oid = self.oid()
        if oid is None:
            return

        def cap(e, oid=oid):
            return SL.stop_recording(e.hist, oid)

        def done(cmd):
            if cmd is None:
                self.win.statusBar().showMessage('Nothing was recording on this object.', 5000)
                return
            self.push(cmd)
            self.win.statusBar().showMessage('Layer made: %d bricks of change.' % cmd.bricks, 6000)
        self.call(cap, done)

    def dial(self, lid, strength=None, on=None):
        oid = self.oid()
        if oid is not None:
            self.push(SL.LayerDial(oid, lid, strength=strength, on=on))

    def solo(self, lid):
        oid = self.oid()
        Ls = self.info.get('layers', [])
        if oid is None or not Ls:
            return
        others_on = [L for L in Ls if int(L['id']) != int(lid) and L.get('on', True)]
        if others_on:                                   # solo: the others off (one undo step)
            cmds = [SL.LayerDial(oid, L['id'], on=False) for L in others_on]
            me = next(L for L in Ls if int(L['id']) == int(lid))
            if not me.get('on', True):
                cmds.append(SL.LayerDial(oid, lid, on=True))
        else:                                           # solo again: every layer on
            cmds = [SL.LayerDial(oid, L['id'], on=True) for L in Ls if not L.get('on', True)]
        self.push(cmds)

    def delete(self, lid, merge=False):
        oid = self.oid()
        if oid is not None:
            self.push(SL.LayerDelete(oid, lid, merge=merge))

    def snapshot(self):
        oid = self.oid()
        if oid is None:
            return
        name, ok = QInputDialog.getText(self.panel, 'Snapshot', 'Name:',
                                        text='Snapshot %d' % (len(self.info.get('snapshots', [])) + 1))
        if ok:
            self.push(SL.SnapshotTake(oid, name.strip() or None))

    def restore(self, name, masked=False):
        oid = self.oid()
        if oid is not None and name:
            self.push(SL.SnapshotRestore(oid, name, masked=masked))

    def snap_delete(self, name):
        oid = self.oid()
        if oid is not None and name:
            self.push(SL.SnapshotDelete(oid, name))


class LayerRow(QWidget):
    def __init__(self, ui, L):
        super().__init__()
        self.ui, self.lid = ui, int(L['id'])
        h = QHBoxLayout(self)
        h.setContentsMargins(2, 1, 2, 1)
        self.on = QCheckBox()
        self.on.setChecked(bool(L.get('on', True)))
        self.on.setToolTip('On / off (off keeps the layer; on puts it back at its strength)')
        self.on.toggled.connect(lambda v: ui.dial(self.lid, on=v))
        h.addWidget(self.on)
        rec = L.get('recording')
        name = QLabel(('● ' if rec else '') + L.get('name', 'Layer'))
        if rec:
            name.setStyleSheet('color: #ff6b6b; font-weight: 600;')
            name.setToolTip('Recording: strokes on this object go into this layer until Stop')
        h.addWidget(name, 1)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-100, 200)
        self.slider.setValue(int(round(float(L.get('strength', 1.0)) * 100)))
        self.slider.setToolTip(DIAL_TIP)
        self.slider.setEnabled(not rec)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-1.0, 2.0)
        self.spin.setSingleStep(0.05)
        self.spin.setDecimals(2)
        self.spin.setValue(float(L.get('strength', 1.0)))
        self.spin.setEnabled(not rec)
        self.spin.setToolTip(DIAL_TIP)
        self.slider.valueChanged.connect(lambda v: self.spin.blockSignals(True) or self.spin.setValue(v / 100.0)
                                         or self.spin.blockSignals(False))
        self.slider.sliderReleased.connect(lambda: ui.dial(self.lid, strength=self.slider.value() / 100.0))
        self.spin.editingFinished.connect(lambda: ui.dial(self.lid, strength=round(self.spin.value(), 2)))
        h.addWidget(self.slider, 2)
        h.addWidget(self.spin)
        for text, tip, fn in (('S', 'Solo: only this layer on (again: all on)', lambda: ui.solo(self.lid)),
                              ('M', 'Merge into the base (its change kept, the layer gone)',
                               lambda: ui.delete(self.lid, merge=True)),
                              ('×', 'Delete (its change taken out)', lambda: ui.delete(self.lid))):
            b = QPushButton(text)
            b.setFixedWidth(26)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            b.setEnabled(not rec)
            h.addWidget(b)


class LayersPanel(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        L = QVBoxLayout(self)
        L.setContentsMargins(6, 6, 6, 6)
        self.title = QLabel('Select an object to see its layers.')
        self.title.setWordWrap(True)
        L.addWidget(self.title)
        self.rows = QVBoxLayout()
        L.addLayout(self.rows)
        self.rec_b = QPushButton('Record a layer')
        self.rec_b.setToolTip('Strokes from now on go into a new layer; press again (Stop) when the pass is done')
        self.rec_b.clicked.connect(ui.record)
        L.addWidget(self.rec_b)
        L.addWidget(QLabel('<b>Snapshots</b>'))
        self.snaps = QListWidget()
        self.snaps.setMaximumHeight(110)
        L.addWidget(self.snaps)
        h = QHBoxLayout()
        for text, tip, fn in (('Take', 'A named state of the object to compare with or go back to', ui.snapshot),
                              ('Restore', 'Back to the chosen snapshot (all of the object)',
                               lambda: ui.restore(self._snap())),
                              ('Restore masked', 'Back to the snapshot only where you masked',
                               lambda: ui.restore(self._snap(), masked=True)),
                              ('Delete', 'Delete the chosen snapshot', lambda: ui.snap_delete(self._snap()))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
        L.addLayout(h)
        L.addStretch(1)

    def _snap(self):
        it = self.snaps.currentItem()
        return it.text().split('  (')[0] if it else None

    def show_info(self, info):
        while self.rows.count():
            w = self.rows.takeAt(0).widget()
            if w is not None:
                w.hide()                         # fs-shell-2 11-14: a removed row leaves the screen now
                w.deleteLater()
        self.snaps.clear()
        if not info:
            self.title.setText('Select an object to see its layers.')
            self.rec_b.setText('Record a layer')
            return
        Ls = info.get('layers', [])
        recording = any(x.get('recording') for x in Ls)
        self.title.setText('<b>%s</b>: %d layer%s (top first)%s' % (info['name'], len(Ls), '' if len(Ls) == 1 else 's',
                                                                    ' - <span style="color:#ff6b6b">recording</span>'
                                                                    if recording else ''))
        for x in reversed(Ls):
            self.rows.addWidget(LayerRow(self.ui, x))
        self.rec_b.setText('Stop recording' if recording else 'Record a layer')
        for s in info.get('snapshots', []):
            self.snaps.addItem('%s  (%d bricks)' % (s['name'], int(s.get('bricks', 0))))


def install(win):
    U = LayersUI(win)
    U.panel = LayersPanel(U)
    win.layers_ui = U
    dock = QDockWidget('Layers', win)
    dock.setObjectName('layers_panel')
    dock.setWidget(U.panel)
    dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
    win.addDockWidget(Qt.RightDockWidgetArea, dock)
    docks = getattr(win, 'docks', None)
    if isinstance(docks, dict) and 'Properties' in docks:
        win.tabifyDockWidget(docks['Properties'], dock)
        docks['Properties'].raise_()
        docks['Layers'] = dock
    vm = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'View' in a.menu().title()), None)
    if vm is not None:
        a = dock.toggleViewAction()
        a.setText('Layers (record a pass, dial it 0-1, snapshots)')
        vm.addAction(a)
    win.props.selectionChanged.connect(lambda *_: QTimer.singleShot(0, U.refresh))
    win.engine.sceneChanged.connect(lambda *_: QTimer.singleShot(0, U.refresh) if dock.isVisible() else None)
    U.dock = dock
    return U
