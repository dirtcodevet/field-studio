"""Topology in the window (fs-shell, 2026-11-11; design D4 "a stroke breaks nothing that existed before it", the UI half;
the counter and the stroke check are fs-brush's: core/topo.py, engine.topoChanged, the engine's 'fill_cavity' job).

What the user sees:
  * the count: the selected object's parts, tunnels (through openings) and pockets (sealed cavities) on the status bar,
    from the whole-object check (Tools > Check topology); after a stroke that changed the topology the line says what
    that stroke did and that the whole count needs a fresh check (a stroke's counts are local to its box: a join inside
    the box can be a new loop of the whole object, so the whole count is never patched from a stroke's numbers);
  * the prompt: a stroke that closed or opened an opening (engine: 'opening_closed') is never applied silently: a bar over
    the view says what changed and offers Keep it / Undo the stroke (the stroke is one undo step; once other steps came
    after it the bar says so instead of undoing the wrong step). A pocket the stroke itself sealed is filled by the
    engine and reported on the status line ('filled');
  * the hidden-void view (View > Hidden voids): each sealed pocket drawn through the model at its place, with its volume
    and size; double-click one to frame it, right-click for Frame it / Fill it solid (the engine's 'fill_cavity', one
    undo step) / hide the view.
Interface agreed with fs-brush 11-11 (their messages 17:5x): core/topo.object_topology(scene, oid, lo, hi) -> betti +
{'cavities': [{'centroid_m', 'volume_m3', 'bbox_m', 'nodes'}], 'nodes'}; engine.topoChanged({'oid', 'stroke_step',
'before', 'after', 'box_m', 'cavities', 'filled', 'opening_closed'}); engine.post('fill_cavity', oid, cavity)."""
import time

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

POCKET_COL = QColor(196, 120, 255)
BIG_CHECK = 60_000_000          # lattice nodes above which the whole-object check asks first (RAM ~ 10 B / node)


def _plural(n, one, many):
    return '%d %s' % (n, one if n == 1 else many)


def counts_text(b):
    return '%s, %s, %s' % (_plural(b['b0'], 'part', 'parts'), _plural(b['b1'], 'tunnel', 'tunnels'),
                           _plural(b['b2'], 'pocket', 'pockets'))


def change_text(before, after):
    """what a stroke did, in plain words, from its local before / after counts"""
    out = []
    d0, d1, d2 = (after[k] - before[k] for k in ('b0', 'b1', 'b2'))
    if d1 < 0:
        out.append('closed an opening' if d1 == -1 else 'closed %d openings' % -d1)
    elif d1 > 0:
        out.append('opened a way through the form' if d1 == 1 else 'opened %d ways through the form' % d1)
    if d0 < 0:
        out.append('joined two parts' if d0 == -1 else 'joined %d parts into one' % (1 - d0))
    elif d0 > 0:
        out.append('split off a part' if d0 == 1 else 'split off %d parts' % d0)
    if d2 > 0:
        out.append('sealed a pocket inside' if d2 == 1 else 'sealed %d pockets inside' % d2)
    elif d2 < 0:
        out.append('opened a sealed pocket to the outside' if d2 == -1 else 'opened %d sealed pockets' % -d2)
    return ', '.join(out) or 'changed nothing in the topology'


def size_text(c):
    lo, hi = np.asarray(c['bbox_m'][0], float), np.asarray(c['bbox_m'][1], float)
    v = float(c['volume_m3']) * 1e6
    vol = ('%.2f cm3' % v) if v >= 0.01 else ('%.1f mm3' % (v * 1000.0))
    ext = (hi - lo) * 1000.0
    return '%s, %.0f x %.0f x %.0f mm' % (vol, ext[0], ext[1], ext[2])


def whole_check(engine, oid, pad=0.003):
    """on the engine thread (under its lock): the object's whole-lattice topology over its world box (+ pad)"""
    from ..core import topo
    sc = engine.hist.scene
    if sc.index_of(oid) < 0:
        return {'error': 'the object is gone'}
    o = sc.get(oid)
    _, lo, hi = sc._object_row(o)
    lo, hi = np.asarray(lo, float) - pad, np.asarray(hi, float) + pad
    t0 = time.perf_counter()
    try:
        b = topo.object_topology(sc, oid, lo, hi)
    except ValueError as e:
        return {'error': str(e), 'limit': 'rotated'}
    b['s'] = time.perf_counter() - t0
    b['name'] = o.name
    b['h'] = float(o.sculpt.h)
    return b


class TopoLayer(QObject):
    """the count, the prompt and the hidden-void view; win.topo"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.known = {}            # oid -> the last whole-object check {'b0', 'b1', 'b2', 'cavities', 's', 'name'}
        self.stale = {}            # oid -> what the last topology-changing stroke did (the whole count needs a check)
        self.events = []           # the stroke events as they came (newest last)
        self.voids_on = False
        self.marker_rects = []     # (QRectF, oid, cavity index) from the last draw
        self.busy = False
        self.pending_prompt = None
        self.label = QLabel('')
        self.label.setTextFormat(Qt.RichText)
        self.label.setToolTip('Parts, tunnels (through openings) and pockets (sealed cavities) of the selected object, '
                              'from Tools > Check topology')
        self.label.linkActivated.connect(lambda _l: self.check())
        self.label.hide()
        win.status.addPermanentWidget(self.label)
        self.bar = PromptBar(self)

    # ---- the whole-object check ------------------------------------------------------------------------------------
    def selected(self):
        sel = list(getattr(self.win.viewport, 'selected', []) or [])
        return sel[0] if sel else None

    def check(self, oid=None, ask=True):
        """Tools > Check topology: the selected object's parts, tunnels and pockets over its whole lattice"""
        oid = self.selected() if oid is None else oid
        if oid is None:
            self.win.status.showMessage('Check topology: select an object first.', 6000)
            return None
        if self.busy:
            return None
        objs = {o['id']: o for o in (self.win.snap or {}).get('objects', [])}
        o = objs.get(oid)
        if o is not None and ask and not getattr(self.win, 'no_prompt', False):
            n = self._estimate_nodes(o)
            if n is not None and n > BIG_CHECK:
                from PySide6.QtWidgets import QMessageBox
                r = QMessageBox.question(
                    self.win, 'Check topology',
                    '%s holds about %.0f million lattice nodes at its %.2f mm spacing. The whole check reads every '
                    'one (about %.0f s and %.1f GB of memory). Run it?' % (o.get('name', 'This object'), n / 1e6,
                                                                       o.get('sculpt_h', 0.001) * 1000,
                                                                       n / 1.05e7, n * 1.1e-8))
                if r != QMessageBox.Yes:
                    return None
        self.busy = True
        self.label.setText('Checking topology...')
        self.label.show()
        eng = self.win.engine
        tok = object()

        def done(rr, tk=tok):
            if rr[0] is not tk:
                return
            eng.callDone.disconnect(done)
            self.busy = False
            self.on_check(oid, rr[1])
        eng.callDone.connect(done)
        eng.post('call', lambda e: whole_check(e, oid), tok)
        return tok

    def _estimate_nodes(self, o):
        try:
            sc = self.win.engine.hist.scene
            ob = sc.get(o['id'])
            lo, hi = ob.local_box()
            h = float(o.get('sculpt_h') or ob.sculpt.h)
            return float(np.prod((np.asarray(hi) - np.asarray(lo)) / h + 1))
        except Exception:
            return None

    def on_check(self, oid, b):
        if b is None or 'error' in b:
            err = (b or {}).get('error', 'no result')
            if (b or {}).get('limit') == 'rotated':
                err = ('this object is rotated or scaled, and the counter reads only unrotated lattices for now '
                       '(fs-brush adds the rotated lattice with D4)')
            self.win.status.showMessage('Check topology: %s.' % err, 15000)
            self.sync()
            return
        self.known[oid] = b
        self.stale.pop(oid, None)
        cav = b.get('cavities') or []
        msg = '%s: %s (%.1f M lattice nodes, %.1f s).' % (b.get('name', 'Object'), counts_text(b), b['nodes'] / 1e6,
                                                         b['s'])
        if cav:
            big = max(cav, key=lambda c: c['volume_m3'])
            msg += ' The largest pocket: %s. View > Hidden voids shows them.' % size_text(big)
        self.win.status.showMessage(msg, 20000)
        self.sync()
        self.win.viewport.update()

    # ---- the status-bar count ---------------------------------------------------------------------------------------
    def sync(self, *_):
        oid = self.selected()
        if self.busy:
            return
        if oid is None or (oid not in self.known and oid not in self.stale):
            self.label.hide()
            return
        b = self.known.get(oid)
        if oid in self.stale:
            t = ('Last stroke %s &middot; <a href="check" style="color:#9ec8ff">check the whole object</a>'
                 % self.stale[oid])
        else:
            t = counts_text(b).replace(', ', ' &middot; ')
            if b['b2']:
                t = t.replace(_plural(b['b2'], 'pocket', 'pockets'),
                              '<span style="color:#c478ff">%s</span>' % _plural(b['b2'], 'pocket', 'pockets'))
        self.label.setText(t)
        self.label.show()

    # ---- the stroke events (engine.topoChanged) ---------------------------------------------------------------------
    def on_topo(self, ev):
        """a stroke changed the local counts (fs-brush's engine emits this only then)"""
        self.events.append(dict(ev, t=time.time()))
        oid = ev.get('oid')
        if ev.get('error') or 'before' not in ev or 'after' not in ev:
            # fs-brush's engine: {'oid', 'error'} when the stroke-end check itself failed: say so, count unknown
            self.stale[oid] = 'could not be checked (%s)' % str(ev.get('error', 'no counts'))[:60]
            self.sync()
            return
        what = change_text(ev['before'], ev['after'])
        self.stale[oid] = what
        if ev.get('filled'):
            cav = ev.get('cavities') or []
            size = (' (%s)' % size_text(cav[0])) if len(cav) == 1 else (' (%d pockets)' % len(cav) if cav else '')
            self.win.status.showMessage('That stroke sealed a pocket inside the form%s; it was filled solid (undo brings '
                                        'it back).' % size, 15000)
            self.stale[oid] = 'sealed a pocket; it was filled solid'
        if ev.get('opening_closed'):
            self.bar.show_for(ev, what)
        self.sync()
        self.win.viewport.update()

    # ---- the hidden-void view ---------------------------------------------------------------------------------------
    def set_voids(self, on):
        self.voids_on = bool(on)
        a = getattr(self, 'voids_action', None)
        if a is not None and a.isChecked() != self.voids_on:
            a.setChecked(self.voids_on)
        if self.voids_on:
            oid = self.selected()
            if oid is not None and oid not in self.known and not self.busy:
                self.check(oid)          # the view needs the pockets: count them once
        self.win.viewport.update()

    def pockets(self):
        """[(oid, index, cavity)] drawn in the view: every checked object still in the scene"""
        objs = {o['id'] for o in (self.win.snap or {}).get('objects', []) if o.get('visible', True)}
        out = []
        for oid, b in self.known.items():
            if oid in objs:
                out += [(oid, k, c) for k, c in enumerate(b.get('cavities') or [])]
        return out

    def draw(self, p):
        self.bar.place()
        self.marker_rects = []
        if not self.voids_on:
            return
        vp = self.win.viewport
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        cam = vp.cam
        p.setFont(QFont('Segoe UI', 9))
        fm = p.fontMetrics()
        placed = []
        for oid, k, c in self.pockets():
            lo, hi = np.asarray(c['bbox_m'][0], float), np.asarray(c['bbox_m'][1], float)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            pr = [cam.project(q, W, H) for q in corners]
            if any(d <= 0 for _, _, d in pr):
                continue
            xy = np.array([(x, y) for x, y, _ in pr])
            cx, cy, cd = cam.project(np.asarray(c['centroid_m'], float), W, H)
            if cd <= 0:
                continue
            p.setPen(QPen(POCKET_COL, 1.3, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            for i in range(8):                         # the box's 12 edges: corners differing in one coordinate
                for j in range(i + 1, 8):
                    if bin(i ^ j).count('1') == 1:
                        p.drawLine(QPointF(*xy[i]), QPointF(*xy[j]))
            r_px = max(4.0, 0.5 * float(np.max(xy.max(0) - xy.min(0))) * 0.35)
            p.setPen(QPen(POCKET_COL, 1.6))
            p.setBrush(QColor(POCKET_COL.red(), POCKET_COL.green(), POCKET_COL.blue(), 70))
            p.drawEllipse(QPointF(cx, cy), min(r_px, 14.0), min(r_px, 14.0))
            text = 'pocket %d: %s' % (k + 1, size_text(c))
            tw = fm.horizontalAdvance(text) + 12
            r = QRectF(cx + 14, cy - 22, tw, fm.height() + 6)
            while any(r.intersects(q) for q in placed):
                r.translate(0, fm.height() + 8)
            placed.append(r)
            p.setPen(QPen(QColor(POCKET_COL.red(), POCKET_COL.green(), POCKET_COL.blue(), 130), 1.0))
            p.drawLine(QPointF(cx, cy), QPointF(r.left(), r.center().y()))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(16, 18, 22, 215))
            p.drawRoundedRect(r, 4, 4)
            p.setPen(POCKET_COL)
            p.drawText(r, Qt.AlignCenter, text)
            self.marker_rects.append((r, oid, k))
            self.marker_rects.append((QRectF(cx - 10, cy - 10, 20, 20), oid, k))

    def pocket_at(self, pos):
        for r, oid, k in self.marker_rects:
            if r.contains(pos):
                return oid, k
        return None

    def frame_pocket(self, oid, k):
        c = self.known[oid]['cavities'][k]
        lo, hi = np.asarray(c['bbox_m'][0], float), np.asarray(c['bbox_m'][1], float)
        pad = max(0.01, float(np.max(hi - lo)))
        vp = self.win.viewport
        vp.cam.frame(lo - pad, hi + pad, margin=1.0, aspect=max(vp.width(), 1) / float(max(vp.height(), 1)))
        vp.request(False)
        vp.update()

    def can_fill(self):
        return getattr(self.win.engine, 'topoChanged', None) is not None

    def fill_pocket(self, oid, k):
        c = self.known[oid]['cavities'][k]
        self.win.engine.post('fill_cavity', oid, c)
        self.stale[oid] = 'filled pocket %d solid' % (k + 1)
        self.known[oid] = dict(self.known[oid], cavities=[q for j, q in enumerate(self.known[oid]['cavities'])
                                                          if j != k])
        self.sync()
        self.win.viewport.update()


class PromptBar(QFrame):
    """the bar over the view after a stroke closed or opened an opening: what changed, Keep it / Undo the stroke"""

    def __init__(self, layer):
        vp = layer.win.viewport
        super().__init__(vp)
        self.layer = layer
        self.ev = None
        self.setObjectName('topoPrompt')
        self.setStyleSheet('#topoPrompt { background: rgba(28, 22, 36, 236); border: 1px solid #c478ff; '
                           'border-radius: 6px; } QLabel { color: #ece6f4; } '
                           'QPushButton { padding: 3px 12px; }')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 8, 6)
        self.text = QLabel('')
        self.keep = QPushButton('Keep it')
        self.undo = QPushButton('Undo the stroke')
        lay.addWidget(self.text, 1)
        lay.addWidget(self.keep)
        lay.addWidget(self.undo)
        self.keep.clicked.connect(self.dismiss)
        self.undo.clicked.connect(self.do_undo)
        self.hide()

    def show_for(self, ev, what):
        self.ev = ev
        objs = {o['id']: o for o in (self.layer.win.snap or {}).get('objects', [])}
        name = objs.get(ev.get('oid'), {}).get('name', 'the object')
        b0, b1 = ev['before'], ev['after']
        self.text.setText('That stroke %s on %s (tunnels %d to %d). Keep it?' % (what, name, b0['b1'], b1['b1']))
        self.keep.setText('Keep it closed' if b1['b1'] < b0['b1'] else 'Keep it')
        self.refresh()
        self.place()
        self.show()
        self.raise_()

    def is_last(self):
        """is the stroke still the newest step? (stroke_step: the stroke's serial in the history, snap['log_ids'])"""
        ids = (self.layer.win.snap or {}).get('log_ids') or []
        st = (self.ev or {}).get('stroke_step')
        return st is None or (bool(ids) and ids[-1] == st)

    def refresh(self, *_):
        if self.ev is None:
            return
        st = self.ev.get('stroke_step')
        if st is not None and st not in ((self.layer.win.snap or {}).get('log_ids') or []):
            self.dismiss()             # undone by hand (or undo went back past it)
            return
        last = self.is_last()
        self.undo.setEnabled(last)
        self.undo.setToolTip('' if last else 'Other steps came after that stroke: Edit > Undo or the History steps '
                                             'go back to it')

    def do_undo(self):
        if not self.is_last():
            return
        self.layer.win.engine.post('undo')
        oid = self.ev.get('oid')
        self.layer.stale.pop(oid, None)
        self.dismiss()
        self.layer.sync()

    def dismiss(self):
        self.ev = None
        self.hide()

    def place(self):
        if not self.isVisible() and self.ev is None:
            return
        vp = self.parentWidget()
        w = min(vp.width() - 24, max(520, self.sizeHint().width()))
        # below the tool's mode label (top-left, ~12..36 px), so neither hides the other
        self.setGeometry(int((vp.width() - w) / 2), 46, int(w), self.sizeHint().height())


class _VoidFilter(QObject):
    """on the viewport while the hidden-void view is on: double-click a pocket frames it; right-click offers Frame it /
    Fill it solid / hide the view (other clicks pass through to the view)"""

    def __init__(self, layer):
        super().__init__(layer)
        self.layer = layer

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        L = self.layer
        if not L.voids_on or ev.type() not in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
            return False
        hit = L.pocket_at(ev.position())
        if hit is None:
            return False
        if ev.type() == QEvent.MouseButtonDblClick and ev.button() == Qt.LeftButton:
            L.frame_pocket(*hit)
            return True
        if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.RightButton:
            from PySide6.QtWidgets import QMenu
            from PySide6.QtGui import QCursor
            m = QMenu(L.win)
            fr = m.addAction('Frame this pocket')
            fi = m.addAction('Fill it solid' if L.can_fill() else 'Fill it solid (comes with the D4 engine half)')
            fi.setEnabled(L.can_fill())
            m.addSeparator()
            hd = m.addAction('Hide the hidden voids')
            got = m.exec(QCursor.pos())
            if got is fr:
                L.frame_pocket(*hit)
            elif got is fi:
                L.fill_pocket(*hit)
            elif got is hd:
                L.set_voids(False)
            return True
        return False


def install(win):
    layer = TopoLayer(win)
    win.topo = layer
    eng = win.engine
    sig = getattr(eng, 'topoChanged', None)
    if sig is not None:
        sig.connect(layer.on_topo)
    eng.sceneChanged.connect(lambda _s: (layer.bar.refresh(), layer.sync()))
    win.props.selectionChanged.connect(lambda _l: QTimer.singleShot(0, layer.sync))
    layer.void_filter = _VoidFilter(layer)
    win.viewport.installEventFilter(layer.void_filter)
    tm = getattr(win, 'menu_tools', None)
    if tm is not None:
        a = tm.addAction('Check topology of the selected object (parts, tunnels, pockets)')
        a.triggered.connect(lambda: layer.check())
        layer.check_action = a
    vm = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'View' in a.menu().title()), None)
    if vm is not None:
        a = vm.addAction('Hidden voids (sealed pockets inside the selected object)')
        a.setCheckable(True)
        a.toggled.connect(layer.set_voids)
        layer.voids_action = a
    return layer
