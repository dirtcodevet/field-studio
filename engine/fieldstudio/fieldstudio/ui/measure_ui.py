"""Measure in the window (fs-shell, 2026-11-11; design D1, queue item 5 "measure and section").

Tools > Measure: girth (one click on a form: the horizontal section through that point; Shift+click: a vertical section
across the screen, for an arm held out sideways; Ctrl+click: the section facing the view) and tape (two clicks: straight
and along the surface). Each result is kept in the scene as a pinned measure: its anchors live in the object's own
frame, so a measure follows the object when it moves and is measured again when its form changes (the engine thread,
debounced). Measures are drawn from the gizmo's overlay painter (curve + label in the scene's unit) and saved with the
scene (the project's extra 'measures'). Right-click a label: delete it; Tools > Measure > Clear all.
The clicks run as a tool mode on the engine thread (the threading rule); results come back by a queued signal."""
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen

COL = {'girth': QColor(250, 204, 21), 'tape': QColor(56, 189, 248)}


def _local(obj, P):
    from ..tools.protected_merge import _to_local
    return _to_local(obj, np.asarray(P, float).reshape(-1, 3))


def _world(obj, Q):
    from ..tools.protected_merge import _to_world
    return _to_world(obj, np.asarray(Q, float).reshape(-1, 3))


def _dir_local(obj, d):
    from ..tools.protected_merge import _rs
    R, t, s = _rs(obj)
    return np.asarray(d, float) @ R


def _dir_world(obj, d):
    from ..tools.protected_merge import _rs
    R, t, s = _rs(obj)
    return R @ np.asarray(d, float)


def compute(scene, m):
    """(re)measure one pinned measure on the engine thread: -> its world curve and numbers (or an error)"""
    from ..tools import measure as M
    if scene.index_of(m['oid']) < 0:
        return dict(m, error='object gone', pts=None)
    o = scene.get(m['oid'])
    try:
        if m['kind'] == 'volume':
            v = M.volume(scene, m['oid'])
            lo, hi = M.extents(scene, m['oid'])
            top = np.array([0.5 * (lo[0] + hi[0]), hi[1], 0.5 * (lo[2] + hi[2])])
            return dict(m, pts=np.stack([top, top]), volume=v, mass=(v * m['density'] if m.get('density') else None),
                        error=None, closed=False)
        if m['kind'] == 'girth':
            p = _world(o, m['a_local'])[0]
            n = _dir_world(o, m['n_local'])
            g = M.girth(scene, m['oid'], p, n)
            if g is None:
                return dict(m, error='no closed section there', pts=None)
            return dict(m, pts=g['pts'], length=g['length'], tape=g['tape'], error=None, closed=True)
        a = _world(o, m['a_local'])[0]
        b = _world(o, m['b_local'])[0]
        t = M.tape(scene, m['oid'], a, b)
        return dict(m, pts=t['pts'], length=t['surface'], straight=t['straight'], error=None, closed=False,
                    section=t.get('section'))
    except Exception as e:                      # a measure must never stop the engine
        return dict(m, error=str(e)[:200], pts=None)


class MeasureMode:
    """the tool mode (tools.register's protocol: on_press / on_drag / on_release on the engine thread)"""

    def __init__(self, hooks, layer, kind='girth'):
        self.hooks, self.layer, self.kind = hooks, layer, kind
        self.first = None
        self.mods = Qt.NoModifier

    def overlay(self):
        return []

    def on_press(self, x, y):
        """the girth tool: a fast girth at the press, shown live while the drag moves it (released: measured fully)"""
        if self.kind != 'girth':
            return None
        from ..tools import measure as M
        r = self.hooks.pick(x, y)
        if not getattr(r, 'hit', False):
            return None
        sc = self.hooks.history.scene
        n = self._normal()
        t0 = __import__('time').perf_counter()
        g = M.girth(sc, int(r.object_id), np.asarray(r.point, float), n, res=0.003, refine=False)
        if g is not None:
            g['oid'] = int(r.object_id)
            g['ms'] = (__import__('time').perf_counter() - t0) * 1000.0
            self.layer.previewReady.emit(g)
        return None

    def on_drag(self, x, y):
        pass

    def _normal(self):
        cam = self.hooks.get_camera()
        f, rgt, up = (np.asarray(v, float) for v in cam.basis())
        mods = self.layer.mods_at_press
        if mods & Qt.ControlModifier:
            return f
        if mods & Qt.ShiftModifier:
            return rgt
        return np.array([0.0, 1.0, 0.0])

    def on_release(self, x, y):
        self.layer.previewReady.emit(None)
        r = self.hooks.pick(x, y)
        if not getattr(r, 'hit', False):
            return None
        sc = self.hooks.history.scene
        o = sc.get(int(r.object_id))
        p = np.asarray(r.point, float)
        if self.kind == 'girth':
            n = self._normal()                            # horizontal; Shift: vertical across; Ctrl: facing the view
            m = {'kind': 'girth', 'oid': int(r.object_id), 'a_local': _local(o, p)[0].tolist(),
                 'n_local': _dir_local(o, n).tolist()}
        else:
            if self.first is None:
                self.first = (int(r.object_id), p)
                self.hooks.status('Tape: now click the second point (on the same object)')
                return None
            oid, a = self.first
            self.first = None
            if oid != int(r.object_id):
                self.hooks.status('Tape: both points must be on the same object')
                return None
            m = {'kind': 'tape', 'oid': oid, 'a_local': _local(o, a)[0].tolist(), 'b_local': _local(o, p)[0].tolist()}
        res = compute(sc, m)
        self.layer.resultReady.emit(res)
        return None


class MeasureLayer(QObject):
    resultReady = Signal(object)
    updated = Signal(object)
    previewReady = Signal(object)

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.items = []                     # pinned measures (dicts; 'pts' world curve)
        self.mods_at_press = Qt.NoModifier
        self.resultReady.connect(self._add)
        self.updated.connect(self._replace)
        self.preview = None                 # the live girth while the girth tool drags
        self.previewReady.connect(self._preview)
        self._follow_busy = False
        self._follow_want = None
        self.live_ms = []
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._remeasure)
        self._last_version = None
        self._scene_gen = None              # fs-shell 11-12 (F4): the scene generation the measures belong to
        self.label_rects = []

    # ---- results -----------------------------------------------------------------------------------------------------
    def _add(self, m):
        if m.get('error'):
            self.win.status.showMessage('Measure: %s' % m['error'], 8000)
            return
        self.items.append(m)
        self.win.status.showMessage(self.text(m), 15000)
        self.win.viewport.update()

    def _preview(self, g):
        self.preview = g
        self.win.viewport.update()

    def follow(self, pos):
        """the girth tool's drag: the live girth at the height of the surface under the cursor (latest wins: one engine
        job at a time, always for the newest cursor)"""
        if self.preview is None:
            return
        self._follow_want = pos
        if self._follow_busy:
            return
        self._follow_busy = True
        pos = self._follow_want
        self._follow_want = None
        vp = self.win.viewport
        x, y, v = vp.to_render(pos)
        prev = self.preview
        tok = object()
        eng = self.win.engine

        def fn(e):
            import time as _t
            from ..tools import measure as M
            t0 = _t.perf_counter()
            r = e._pick(v, x, y)
            if not getattr(r, 'hit', False) or int(r.object_id) != prev['oid']:
                return None
            g = M.girth_follow(e.hist.scene, prev['oid'], prev, np.asarray(r.point, float))
            if g is not None:
                g['oid'] = prev['oid']
                g['ms'] = (_t.perf_counter() - t0) * 1000.0
            return g

        def done(rr, tk=tok):
            if rr[0] is not tk:
                return
            eng.callDone.disconnect(done)
            self._follow_busy = False
            if rr[1] is not None and self.preview is not None:
                self.preview = rr[1]
                self.live_ms.append(rr[1].get('ms', 0.0))
                vp.update()
            if self._follow_want is not None:
                self.follow(self._follow_want)
        eng.callDone.connect(done)
        eng.post('call', fn, tok)

    def _replace(self, ms):
        self.items = [m for m in ms if m is not None]
        self.win.viewport.update()

    def text(self, m):
        from .units import format_length as F
        u = getattr(self.win, 'units', 'mm')
        if m.get('live'):
            return 'Girth %s (tape %s)  live' % (F(m['length'], u), F(m['tape'], u))
        if m.get('name'):
            return '%s: %s' % (m['name'], self.text(dict(m, name=None)))
        if m['kind'] == 'volume':
            v = m.get('volume') or 0.0
            if u in ('in', 'ft-in'):
                vs = ('%.2f in\u00b3' % (v / 0.0254 ** 3)) if v < 0.3048 ** 3 else ('%.3f ft\u00b3' % (v / 0.3048 ** 3))
                ws = '' if m.get('mass') is None else ', weight %.2f lb' % (m['mass'] / 0.45359237)
            else:
                vs = ('%.1f cm\u00b3' % (v * 1e6)) if v < 1e-3 else ('%.3f L' % (v * 1e3))
                ws = '' if m.get('mass') is None else ', weight %.3f kg' % m['mass']
            return 'Volume %s%s%s' % (vs, ws, (' (%s)' % m['material']) if m.get('material') else '')
        if m['kind'] == 'girth':
            return 'Girth %s (as a tape sits %s)' % (F(m['length'], u), F(m['tape'], u))
        s = 'Tape %s along the surface (straight %s)' % (F(m['length'], u), F(m['straight'], u))
        if str(m.get('section') or '').startswith('projected'):     # fs-shell 11-12: said, never silent
            s += ': no section path between the points, measured along a line projected onto the surface'
        return s

    def scene_changed(self, snap):
        # fs-shell 11-12 (fs-bench-full F4; fs-hair's scene_gen, its on_scene the pattern): a new scene (open her, new,
        # open a region, open a project) restarts the object ids, so a pinned measure kept by id would measure whatever
        # takes its id: on a new scene generation the measures go. A project's own come back through load() (its
        # projectLoaded arrives before its snapshot and takes that generation, so they stay)
        gen = (snap or {}).get('scene_gen')
        if gen is not None and gen != self._scene_gen:
            first = self._scene_gen is None
            self._scene_gen = gen
            if not first and (self.items or getattr(self, 'dims', None) is not None
                              or getattr(self, 'preview', None) is not None):
                n = len(self.items)
                self.items, self.dims, self.preview = [], None, None
                self._tok = None                    # a remeasure still running for the old scene is not taken
                if n:
                    self.win.status.showMessage('Measures cleared: a new scene (%d)' % n, 6000)
                self.win.viewport.update()
        v = (snap or {}).get('version')
        if self.items and v != self._last_version:
            self._last_version = v
            self.timer.start()

    def _remeasure(self):
        if not self.items:
            return
        ms = [dict(m) for m in self.items]

        def fn(e):
            return [compute(e.hist.scene, m) for m in ms]
        self._tok = object()
        eng = self.win.engine

        def done(r, tok=self._tok):
            if r[0] is tok:
                eng.callDone.disconnect(done)
                if tok is not self._tok:            # F4: the scene changed meanwhile; these are the old scene's
                    return
                self.updated.emit([m for m in r[1] if not m.get('error') or m.get('error') != 'object gone'])
        eng.callDone.connect(done)
        eng.post('call', fn, self._tok)

    def clear(self):
        self.items = []
        self.win.viewport.update()

    def to_list(self):
        keep = ('kind', 'oid', 'a_local', 'b_local', 'n_local', 'density', 'material', 'name')
        return [{k: v for k, v in m.items() if k in keep} for m in self.items]

    def add_volume(self, oid, material=None):
        """Measure > Volume and weight: the selected object's volume (and weight from a material's density), pinned"""
        from ..tools.measure import DENSITY
        m = {'kind': 'volume', 'oid': int(oid), 'material': material, 'density': DENSITY.get(material)}
        tok = object()
        eng = self.win.engine
        gen0 = self._scene_gen

        def done(r, tk=tok):
            if r[0] is tk:
                eng.callDone.disconnect(done)
                if self._scene_gen == gen0:         # F4: not into a scene that came after it was asked for
                    self._add(r[1])
        eng.callDone.connect(done)
        eng.post('call', lambda e: compute(e.hist.scene, m), tok)

    # ---- the dimensions overlay (design D1: W x H x D of the selected object, exact from the field) ---------------------
    def set_dims_world(self, on):
        """'world extents' (the space it takes up in the scene, along the world's x, y, z) instead of the
        object's own size"""
        self.dims_world = bool(on)
        self.dims = None
        if getattr(self, 'dims_on', False):
            self.schedule_dims()
        self.win.viewport.update()

    def set_dims(self, on):
        self.dims_on = bool(on)
        self.dims = None
        if on:
            self.schedule_dims()
        self.win.viewport.update()

    def schedule_dims(self, *_):
        if getattr(self, 'dims_on', False):
            if not hasattr(self, 'dims_timer'):
                self.dims_timer = QTimer(self)
                self.dims_timer.setSingleShot(True)
                self.dims_timer.setInterval(200)
                self.dims_timer.timeout.connect(self._dims_run)
            self.dims_timer.start()

    def _dims_run(self):
        vp = self.win.viewport
        if not vp.selected:
            self.dims = None
            vp.update()
            return
        oid = int(vp.selected[0])
        tok = object()
        eng = self.win.engine
        self._dims_tok = tok
        rg = getattr(self.win, 'regions_ui', None)        # design D1: a region selected in the Regions panel: its size
        rid = rg.active if (rg is not None and rg.active and rg.oid() == oid) else None
        world = bool(getattr(self, 'dims_world', False))   # own size by default, world extents an option

        def fn(e):
            from ..tools import measure as M
            if e.hist.scene.index_of(oid) < 0:
                return None
            A = np.eye(3) if world else M.own_axes(e.hist.scene.get(oid))
            base = {'oid': oid, 'axes': A.tolist(), 'world': world}
            if rid:
                from ..tools import regions as RG
                r = RG.region_extents(e.hist.scene, oid, rid, axes=A)
                if r is not None:
                    return dict(base, lo=r[0], hi=r[1], rid=rid)
            r = M.extents(e.hist.scene, oid, axes=A)
            return None if r is None else dict(base, lo=r[0], hi=r[1])

        def done(r, tk=tok):
            if r[0] is tk:
                eng.callDone.disconnect(done)
                self.dims = r[1]
                vp.update()
        eng.callDone.connect(done)
        eng.post('call', fn, tok)

    def draw_dims(self, p):
        d = getattr(self, 'dims', None)
        if not getattr(self, 'dims_on', False):
            return
        from .units import format_length as F
        vp = self.win.viewport
        if d is None or d['oid'] not in vp.selected:
            want = int(vp.selected[0]) if vp.selected else None
            if want is not None and getattr(self, '_dims_asked', None) != want:
                self._dims_asked = want              # the selection changed: measure the new one (once)
                self.schedule_dims()
            return
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        u = getattr(self.win, 'units', 'mm')
        lo, hi = np.asarray(d['lo']), np.asarray(d['hi'])
        A = np.asarray(d.get('axes', np.eye(3)), float)        # the box along these axes (own: it turns with the object)
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ A.T
        pr = np.array([vp.cam.project(c, W, H) for c in C])
        if np.any(pr[:, 2] <= 1e-6):
            return
        col = QColor(226, 232, 240)
        p.setPen(QPen(QColor(226, 232, 240, 110), 1.0, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        for a_ in range(8):
            for b_ in range(a_ + 1, 8):
                if bin(a_ ^ b_).count('1') == 1:
                    p.drawLine(QPointF(pr[a_, 0], pr[a_, 1]), QPointF(pr[b_, 0], pr[b_, 1]))
        # the three edges through the corner nearest the viewer, each labelled with its length
        f = np.asarray(vp.cam.basis()[0], float)
        k0 = int(np.argmin(C @ f))
        p.setFont(QFont('Segoe UI', 9))
        fm = p.fontMetrics()
        names = ('world X', 'world Y', 'world Z') if d.get('world') else ('W', 'H', 'D')
        for ax, name in enumerate(names):
            k1 = k0 ^ (1 << (2 - ax))
            a, b = pr[k0], pr[k1]
            if np.hypot(*(b[:2] - a[:2])) < 12:
                continue                       # an edge seen end-on
            p.setPen(QPen(col, 1.6))
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
            text = '%s%s %s' % ('region ' if d.get('rid') else '', name, F(hi[ax] - lo[ax], u))
            mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            r = QRectF(mx - fm.horizontalAdvance(text) / 2 - 5, my - 10, fm.horizontalAdvance(text) + 10, 20)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(16, 18, 22, 215))
            p.drawRoundedRect(r, 4, 4)
            p.setPen(col)
            p.drawText(r, Qt.AlignCenter, text)

    def load(self, lst):
        # the loaded project's scene generation: its own measures stay when its snapshot arrives (F4)
        self._scene_gen = getattr(getattr(getattr(self.win, 'engine', None), 'hist', None), 'gen', None)
        self.items = []
        if lst:
            self.items = [dict(m, pts=None) for m in lst]
            self.timer.start()

    # ---- drawing (the gizmo's overlay painter) ---------------------------------------------------------------------------
    def _hidden(self, P, pr, W, H):
        """per curve point: is the model in front of it (the frame's hit nearer)?"""
        vp = self.win.viewport
        fr = getattr(vp, 'frame', None)
        if fr is None or fr.get('obj') is None or fr.get('depth') is None:
            return np.zeros(len(P), bool)
        obj, dep = np.asarray(fr['obj']), np.asarray(fr['depth'])
        jj = np.clip((pr[:, 1] * obj.shape[0] / H).astype(int), 0, obj.shape[0] - 1)
        ii = np.clip((pr[:, 0] * obj.shape[1] / W).astype(int), 0, obj.shape[1] - 1)
        cam = vp.cam
        if getattr(cam, 'ortho', False):
            f = np.asarray(cam.basis()[0], float)
            t = (np.asarray(P) - np.asarray(cam.pos, float)[None]) @ f + cam.ortho_params()[1]
        else:
            t = np.linalg.norm(np.asarray(P) - np.asarray(cam.pos, float)[None], axis=1)
        d = dep[jj, ii].astype(float)
        return (obj[jj, ii] >= 0) & (d < t - np.maximum(0.0015, 0.004 * t))

    def draw(self, p):
        self.draw_dims(p)
        items = list(self.items)
        if self.preview is not None:
            items.append(dict(self.preview, kind='girth', closed=True, live=True))
        if not items:
            self.label_rects = []
            return
        vp = self.win.viewport
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        cam = vp.cam
        self.label_rects = []
        p.setFont(QFont('Segoe UI', 9))
        fm = p.fontMetrics()
        for k, m in enumerate(items):
            P = m.get('pts')
            if P is None or len(P) < 2:
                continue
            P = np.asarray(P, float)
            pr = np.array([cam.project(q, W, H) for q in P])
            ok = pr[:, 2] > 1e-6
            if ok.sum() < 2:
                continue
            hid = self._hidden(P, pr, W, H)
            col = COL.get(m['kind'], QColor(240, 240, 240))
            n = len(P)
            idx = list(range(n)) + ([0] if m.get('closed') else [])
            # visible stretches solid over a dark halo, hidden ones (behind the form) faint and dashed
            for want_hidden in (True, False):
                path = QPainterPath()
                pen_down = False
                for a_, b_ in zip(idx[:-1], idx[1:]):
                    seg_ok = ok[a_] and ok[b_] and (bool(hid[a_] or hid[b_]) == want_hidden)
                    if not seg_ok:
                        pen_down = False
                        continue
                    if not pen_down:
                        path.moveTo(QPointF(pr[a_, 0], pr[a_, 1]))
                        pen_down = True
                    path.lineTo(QPointF(pr[b_, 0], pr[b_, 1]))
                p.setBrush(Qt.NoBrush)
                if want_hidden:
                    c2 = QColor(col)
                    c2.setAlpha(110)
                    p.setPen(QPen(c2, 1.0, Qt.DashLine))
                    p.drawPath(path)
                else:
                    p.setPen(QPen(QColor(0, 0, 0, 140), 3.2))
                    p.drawPath(path)
                    p.setPen(QPen(col, 1.6))
                    p.drawPath(path)
            if m['kind'] == 'tape':
                p.setPen(QPen(col, 1.6))
                p.setBrush(col)
                for j in (0, n - 1):
                    if ok[j]:
                        p.drawEllipse(QPointF(pr[j, 0], pr[j, 1]), 3.5, 3.5)
            # the label beside the curve's rightmost visible point, moved down until it overlaps no other label
            vis = pr[ok & ~hid] if (ok & ~hid).any() else pr[ok]
            j = int(np.argmax(vis[:, 0]))
            x, y = float(vis[j, 0]) + 10, float(vis[j, 1])
            text = self.text(m)
            r = QRectF(x, y - 10, fm.horizontalAdvance(text) + 10, 20)
            if r.right() > W:
                r.moveRight(W - 4)
            for _ in range(40):
                if not any(r.intersects(r2.adjusted(-2, -2, 2, 2)) for r2, _k in self.label_rects):
                    break
                r.translate(0, 24)
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 120), 1.0))
            p.drawLine(QPointF(vis[j, 0], vis[j, 1]), QPointF(r.left(), r.center().y()))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(16, 18, 22, 215))
            p.drawRoundedRect(r, 4, 4)
            p.setPen(col)
            p.drawText(r, Qt.AlignCenter, text)
            self.label_rects.append((r, k))

    def label_at(self, pos):
        for r, k in self.label_rects:
            if r.contains(pos):
                return k
        return None


class _ViewFilter(QObject):
    """on the viewport: remembers the modifiers of each press (the girth's plane), and a right-click on a measure's
    label offers to delete it (before the view's own action menu)"""

    def __init__(self, layer):
        super().__init__(layer)
        self.layer = layer

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if ev.type() == QEvent.MouseMove and (ev.buttons() & Qt.LeftButton) and self.layer.preview is not None:
            mode = getattr(self.layer.win.viewport, 'tool_mode', None)
            if type(mode).__name__ == 'MeasureMode' and mode.kind == 'girth':
                self.layer.follow(ev.position())        # the girth follows the cursor's height, live
        if ev.type() == QEvent.MouseButtonPress:
            self.layer.mods_at_press = ev.modifiers()
            if ev.button() == Qt.RightButton:
                k = self.layer.label_at(ev.position())
                if k is not None and k < len(self.layer.items):
                    from PySide6.QtWidgets import QMenu
                    from PySide6.QtGui import QCursor
                    m = QMenu(self.layer.win)
                    nm = m.addAction('Name this measure...')
                    a = m.addAction('Delete this measure')
                    b = m.addAction('Clear all measures')
                    got = m.exec(QCursor.pos())
                    if got is nm:
                        from PySide6.QtWidgets import QInputDialog
                        text, ok = QInputDialog.getText(self.layer.win, 'Name this measure', 'Name:',
                                                        text=self.layer.items[k].get('name') or '')
                        if ok:
                            self.layer.items[k]['name'] = text.strip() or None
                    elif got is a:
                        del self.layer.items[k]
                    elif got is b:
                        self.layer.items = []
                    self.layer.win.viewport.update()
                    return True
        return False


def install(win):
    layer = MeasureLayer(win)
    win.measures = layer
    win.engine.sceneChanged.connect(layer.scene_changed)
    win.engine.sceneChanged.connect(layer.schedule_dims)
    layer.dims_on = False
    layer.view_filter = _ViewFilter(layer)
    win.viewport.installEventFilter(layer.view_filter)
    return layer


def start(win, kind):
    """Tools > Measure > Girth / Tape: the measure tool mode (Esc leaves it)"""
    m = MeasureMode(win.tool_hooks, win.measures, kind)
    win.set_tool_mode(m)
    win.viewport.mode_label = ('Girth: click a form (Shift: vertical across the screen, Ctrl: facing the view); Esc '
                               'leaves' if kind == 'girth' else
                               'Tape: click two points on a form (straight and along the surface); Esc leaves')
    win.viewport.update()
    return m
