"""The view's presses for fs-tools' registry tools that act at the pointer (fs-shell-3, 2026-11-14; registry stage 2b;
the tools' jobs are fs-tools' toolset/*.py, registered on the engine by toolset/jobs.register).

- Place on surface (TOOLSET_DESIGN 3.2): press on an object and drag: it rides the OTHER surfaces under the cursor,
  sitting on them, turned to their facing (Align), spun by the wheel; the release drops it with the exact contact
  (tools.place_press / place_drag / place_release: one undo step). Or Pick up (the strip's button, or P again): the
  selected object follows the cursor with no button held; a click drops it. Esc puts it back.
The drag jobs are coalesced: the window keeps only the newest cursor ray, and one 'shell.place_drag' job at a time
reads it on the engine thread, so a slow surface read never queues a backlog behind the hand.
"""
import math
import threading

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, Qt


def _ray(vp, pos):
    o, d = vp.cam.ray(float(pos.x()), float(pos.y()), max(vp.width(), 1), max(vp.height(), 1))
    return [[float(v) for v in o], [float(v) for v in d]]


# ---- the Add tool's shapes: today's Create menu shapes (main_window.PRIMS), at the Size set ----------------------------
SHAPES = ('sphere', 'box', 'cylinder', 'capsule', 'cone', 'torus', 'ellipsoid')
_LENGTHS = ('radius', 'size', 'round', 'height', 'radius_bottom', 'radius_top', 'major', 'minor', 'radii', 'thickness',
            'extent', 'offset')
_EXT = {}


def shape_params(kind, size):
    """the shape's parameters with its largest extent = size (m): the Create menu's proportions, scaled"""
    from .main_window import PRIMS
    from ..core import ops
    base = next((dict(p) for k, _n, p in PRIMS if k == kind), None)
    if base is None:
        raise ValueError('no shape %r' % (kind,))
    if kind not in _EXT:
        _c, _p, lo, hi, _L = ops.compile_tree(ops.Node(kind, dict(base)))
        _EXT[kind] = float((np.asarray(hi, float) - np.asarray(lo, float)).max())
    f = float(size) / max(_EXT[kind], 1e-9)
    out = {}
    for k, v in base.items():
        if k in _LENGTHS:
            out[k] = [round(float(x) * f, 7) for x in v] if isinstance(v, (list, tuple)) else round(float(v) * f, 7)
        else:
            out[k] = v
    return out


class RegMouse:
    """one per window (win.reg_mouse; the viewport calls press / move / release / wheel / key)"""

    def __init__(self, win):
        self.win = win
        self.st = None                         # the gesture in progress: {'tool', 'oid', ...}
        self._lock = threading.Lock()
        self._latest = None                    # place: (oid, ray, align, offset, spin) the next drag job reads
        self._inflight = False
        self.cues = None                       # the live cues drawn while a gesture runs (the blade, the connector)
        self._cue_req = None                   # the newest cue request (name, args) the cue job reads
        self._cue_inflight = False
        self._cue_gen = 0
        eng = win.engine
        for name, fn in (('shell.place_drag', self._place_drag_job), ('shell.place_cancel', self._place_cancel_job),
                         ('shell.cue_live', self._cue_job)):
            try:
                eng.register_job(name, fn)
            except ValueError:
                pass

    # ---- which tool ---------------------------------------------------------------------------------------------
    def tool(self):
        from . import toolreg as TR
        cur = TR.current(self.win)
        return cur[0] if cur else None

    def _val(self, key, skey, default=None):
        from . import toolreg as TR
        sp = TR.spec(key)
        s = next((x for x in (sp.own if sp else ()) if x.key == skey), None)
        if s is None:
            return default
        return (getattr(self.win, 'reg_values', {}) or {}).get((key, skey), s.default)

    def busy(self):
        return self.st is not None

    # ---- the viewport's events ------------------------------------------------------------------------------------------
    def press(self, pos, mods):
        """a left press in the view; -> True when a registry tool took it (the viewport then sends move / release)"""
        t = self.tool()
        if self.st is not None and self.st.get('follow'):
            self._place_end(pos)               # Pick up: a click drops it
            return 'done'
        if self.st is not None and self.st.get('tool') != t:
            self.cancel()                      # another tool was picked mid-gesture: that gesture is dropped
        if t == 'place':
            vp = self.win.viewport
            oid = vp._obj_at(pos)
            if oid < 0:
                return False
            self._place_begin(oid, pos, follow=False)
            return True
        if t == 'add':
            self.add_at(pos)
            return 'done'
        if t == 'cut':
            if str(self._val('cut', 'type', 'line')) == 'shape':
                self.cut_shape_at(pos)
                return 'done'
            if self.st is not None and self.st.get('tool') == 'cut' and self.st.get('await_b'):
                self._cut_line_end(pos)
                return 'done'
            vp = self.win.viewport
            self.st = {'tool': 'cut', 'a': QPointF(pos), 'b': QPointF(pos), 'ray_a': _ray(vp, pos),
                       'oid_a': int(vp._obj_at(pos))}
            self._say('Cut: drag the line across the form (or click where it ends); the blade shows before it cuts.')
            return True
        if t == 'connect':
            self._connect_click(pos)
            return 'done'
        return False

    def move(self, pos):
        if self.st is None:
            return
        if self.st['tool'] == 'place':
            self._place_move(pos)
        elif self.st['tool'] == 'cut':
            self.st['b'] = QPointF(pos)
            self._cut_cue()

    def hover(self, pos):
        """the mouse moves with no button held: a picked-up object follows it; a line cut's second end, the
        connector's second end follow it"""
        if self.st is None:
            return False
        if self.st.get('follow'):
            self._place_move(pos)
            return True
        if self.st.get('tool') == 'cut' and self.st.get('await_b'):
            self.st['b'] = QPointF(pos)
            self._cut_cue()
            return True
        if self.st.get('tool') == 'connect':
            self._connect_cue(pos)
            return True
        return False

    def release(self, pos):
        if self.st is None:
            return
        if self.st['tool'] == 'place' and not self.st.get('follow'):
            self._place_end(pos)
        elif self.st['tool'] == 'cut':
            a = self.st['a']
            if abs(pos.x() - a.x()) + abs(pos.y() - a.y()) > 6:
                self._cut_line_end(pos)
            else:
                self.st['await_b'] = True       # a click: the line's end is the next click
                self._say('Cut: click where the line ends (Esc: cancel).')

    def wheel(self, steps):
        """the wheel while placing spins the object about the surface's facing (15 deg a notch)"""
        if self.st is None or self.st['tool'] != 'place':
            return False
        self.st['spin'] = (self.st.get('spin', 0.0) + 15.0 * float(steps)) % 360.0
        if self.st.get('pos') is not None:
            self._place_move(self.st['pos'])
        return True

    def cancel(self):
        """Esc: the gesture is dropped; a placed object goes back where it was"""
        if self.st is None:
            return False
        if self.st['tool'] == 'place':
            self.win.engine.post('shell.place_cancel', self.st['oid'])
            self._say('Place on surface: put back.')
        else:
            self._say('%s: cancelled.' % self.st['tool'].capitalize())
        self.st = None
        self.cues = None
        self.win.viewport.update()
        return True

    # ---- live cues (toolset/cues.py on the engine thread; the newest request only) --------------------------------------
    def _request_cue(self, name, *args):
        with self._lock:
            self._cue_gen += 1
            self._cue_req = (self._cue_gen, name, args)
            go = not self._cue_inflight
            if go:
                self._cue_inflight = True
        if go:
            self.win.engine.post('shell.cue_live')

    def _cue_job(self, engine):
        from ..toolset import cues as CU
        with self._lock:
            req = self._cue_req
            self._cue_req = None
            self._cue_inflight = False
        if req is None:
            return False
        gen, name, args = req
        try:
            if name == 'cut_line':
                out = CU.cut_line(engine.hist.scene, *args)
            elif name == 'connect':
                out = CU.connect(*args)
            else:
                out = []
        except Exception as e:
            out = [{'kind': 'text', 'at': list(map(float, args[1][0] if name == 'cut_line' else args[0])),
                    'text': 'cue not measured: %s' % type(e).__name__, 'style': 'label'}]
        if self.st is not None and gen == self._cue_gen:
            self.cues = out
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(self.win.viewport, 'update', Qt.QueuedConnection)
        return False

    def draw(self, p):
        if self.st is None or not self.cues:
            return
        from .cues_draw import draw as _draw_cues
        vp = self.win.viewport
        _draw_cues(p, vp.cam, vp.width(), vp.height(), self.cues)

    # ---- Cut (line: the blade from the first surface, Length deep; shape: its volume to the container) -------------------
    def _cut_oid(self, st):
        """the object the line cuts: the one selected, else the one under its start, its end, or its middle"""
        vp = self.win.viewport
        sel = list(getattr(vp, 'selected', None) or [])
        if len(sel) == 1:
            return int(sel[0])
        for q in (st['a'], st['b'], QPointF(0.5 * (st['a'].x() + st['b'].x()), 0.5 * (st['a'].y() + st['b'].y()))):
            o = int(vp._obj_at(q))
            if o >= 0:
                return o
        return -1

    def _cut_cue(self):
        st = self.st
        oid = self._cut_oid(st)
        if oid < 0:
            self.cues = []                     # the line crosses no object yet: no blade
            return
        vp = self.win.viewport
        self._request_cue('cut_line', oid, st['ray_a'], _ray(vp, st['b']),
                          float(self._val('cut', 'length', 0.05) or 0.05))

    def _cut_line_end(self, pos):
        st = self.st
        st['b'] = QPointF(pos)
        oid = self._cut_oid(st)
        self.st = None
        self.cues = None
        if oid < 0:
            self._say('Cut: the line misses the objects (nothing cut).')
            return
        vp = self.win.viewport
        length = float(self._val('cut', 'length', 0.05) or 0.05)
        self.win.engine.post('tools.cut_line', oid, st['ray_a'], _ray(vp, pos), length)
        self._say('Cutting %.1f mm deep from the first surface ... (one undo step)' % (length * 1e3))
        vp.update()

    def cut_shape_at(self, pos):
        """a click on a form: the shape (its size: Shape size) centred on the surface point there; its volume leaves the
        form and becomes the container's piece (Delete it or Place it)"""
        vp = self.win.viewport
        oid = int(vp._obj_at(pos))
        P = vp.surface_at(pos)
        if oid < 0 or P is None:
            return self._say('Cut with a shape: click on the form to cut.')
        shape = str(self._val('cut', 'shape', 'box'))
        size = float(self._val('cut', 'shape_size', 0.02) or 0.02)
        try:
            params = shape_params(shape, size)
        except Exception as e:
            return self._say('Cut: %s' % e)
        at = {'translate': [float(v) for v in P], 'rotate': [0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0]}
        self.win.engine.post('tools.cut_shape', oid, shape, params, at)
        return self._say('Cutting a %s of %.0f mm out ... the piece goes to the container (Delete it, or Place it).'
                         % (shape, size * 1e3))

    def held_piece(self):
        r = getattr(self.win.engine, 'tool_result', None) or {}
        return r.get('held') if r.get('tool') == 'cut' else None

    def place_piece(self):
        pid = self.held_piece()
        if pid is None:
            return self._say('No cut piece is held.')
        self.win.props.set_selection([int(pid)])
        from . import toolreg as TR
        TR.select(self.win, 'reg:place')
        self.win.viewport.selected = [int(pid)]
        return self.pick_up()

    # ---- Connect (two surfaces, the profile carried between them: three objects in contact) -----------------------------
    def _profile(self):
        return {'shape': str(self._val('connect', 'profile', 'circle')),
                'size': float(self._val('connect', 'size', 0.005) or 0.005)}

    def _connect_click(self, pos):
        vp = self.win.viewport
        oid = int(vp._obj_at(pos))
        P = vp.surface_at(pos)
        if oid < 0 or P is None:
            return self._say('Connect: click on a surface.')
        if self.st is None or self.st.get('tool') != 'connect':
            self.st = {'tool': 'connect', 'a': oid, 'p_a': [float(v) for v in P]}
            return self._say('Connect: now click the surface of the other object.')
        if oid == self.st['a']:
            return self._say('Connect: click a surface of ANOTHER object (the connector runs between two).')
        st = self.st
        self.st = None
        self.cues = None
        twist = float(self._val('connect', 'twist', 0.0) or 0.0)
        self.win.engine.post('tools.connect', st['a'], oid, st['p_a'], [float(v) for v in P], self._profile(), None,
                             twist)
        vp.update()
        return self._say('Connecting ... three objects, all in contact (one undo step); Merge joins them.')

    def _connect_cue(self, pos):
        P = self.win.viewport.surface_at(pos)
        if P is None:
            return
        twist = float(self._val('connect', 'twist', 0.0) or 0.0)
        self._request_cue('connect', self.st['p_a'], [float(v) for v in P], self._profile(), None, twist)

    # ---- Place on surface -----------------------------------------------------------------------------------------------
    def pick_up(self):
        """the strip's 'Pick up' (or P again): the selected object follows the cursor; a click drops it"""
        vp = self.win.viewport
        sel = list(getattr(vp, 'selected', None) or [])
        if len(sel) != 1:
            return 'Place on surface: select one object first.'
        pos = vp.mouse_pos or QPointF(vp.width() / 2.0, vp.height() / 2.0)
        self._place_begin(int(sel[0]), pos, follow=True)
        vp.setFocus()
        return 'Move the mouse: the object rides the surface under it. A click drops it; the wheel spins it; Esc puts it back.'

    def _place_begin(self, oid, pos, follow):
        vp = self.win.viewport
        if oid not in (getattr(vp, 'selected', None) or []):
            vp.selectRequest.emit(int(oid), False)
        self.st = {'tool': 'place', 'oid': int(oid), 'spin': 0.0, 'follow': bool(follow), 'pos': QPointF(pos),
                   'moved': False}
        self._latest = None
        self.win.engine.post('tools.place_press', int(oid))
        self._say('Place on surface: drag over another surface; release drops it (wheel: spin, Esc: put back).'
                  if not follow else 'Place on surface: the object follows the cursor; a click drops it.')

    def _place_move(self, pos):
        st = self.st
        st['pos'] = QPointF(pos)
        st['moved'] = True
        vp = self.win.viewport
        with self._lock:
            self._latest = (st['oid'], _ray(vp, pos), str(self._val('place', 'align', 'surface')),
                            float(self._val('place', 'offset', 0.0) or 0.0), float(st.get('spin', 0.0)))
            go = not self._inflight
            if go:
                self._inflight = True
        if go:
            self.win.engine.post('shell.place_drag')

    def _place_end(self, pos):
        st = self.st
        self.st = None
        if st is None:
            return
        if st.get('moved'):
            self._place_move_final(st, pos)
            self.win.engine.post('tools.place_release', st['oid'])
            self._say('Placed on the surface (one undo step).')
        else:
            self.win.engine.post('shell.place_cancel', st['oid'])
        self.win.viewport.update()

    def _place_move_final(self, st, pos):
        vp = self.win.viewport
        with self._lock:
            self._latest = (st['oid'], _ray(vp, pos), str(self._val('place', 'align', 'surface')),
                            float(self._val('place', 'offset', 0.0) or 0.0), float(st.get('spin', 0.0)))
            self._inflight = True
        self.win.engine.post('shell.place_drag')     # the release's own ray, before the drop (the queue keeps order)

    def _place_drag_job(self, engine):
        """engine thread: the newest ray only (older ones were never queued)"""
        from ..toolset import place as PL
        with self._lock:
            lt = self._latest
            self._latest = None
            self._inflight = False
        if lt is None:
            return False
        oid, ray, align, offset, spin = lt
        try:
            return PL.place_drag(engine, oid, ray, align, offset, spin)
        except Exception as e:
            engine.tool_result = {'tool': 'place', 'error': repr(e)}
            return False

    @staticmethod
    def _place_cancel_job(engine, oid):
        pl = getattr(engine, '_placer', None)
        if pl is None or pl.oid != int(oid):
            return False
        o = engine.hist.scene.get(int(oid))
        o.transform = dict(pl.t0)
        engine.hist.scene.touch(structural=True)
        engine._placer = None
        return True

    # ---- Add (the shape library, with Hollow) -----------------------------------------------------------------------------
    def add_at(self, pos=None):
        """a click in the view (or the Add button / A: the view's centre): the shape lands on the surface under it,
        sitting on it, or at the view's target when nothing is under it (tools.add_at: one undo step, Hollow included)"""
        vp = self.win.viewport
        if pos is None:
            pos = QPointF(vp.width() / 2.0, vp.height() / 2.0)
        kind = str(self._val('add', 'kind', 'sphere'))
        size = float(self._val('add', 'size', 0.05) or 0.05)
        try:
            params = shape_params(kind, size)
        except Exception as e:
            return self._say('Add: %s' % e)
        hollow = bool(self._val('add', 'hollow', False))
        wall = float(self._val('add', 'wall', 0.005) or 0.005)
        objs = (getattr(self.win, 'snap', None) or {}).get('objects', [])
        h = min([ob['sculpt_h'] for ob in objs if ob.get('sculpt_h')] or [0.0005])
        fallback = [float(v) for v in vp.cam.target]
        self.win.engine.post('tools.add_at', kind, params, _ray(vp, pos), fallback, wall if hollow else None,
                             'surface', 0.0, h)
        line = 'Added a %s, %.0f mm%s.' % (kind, size * 1e3, (', hollow with a %.1f mm wall' % (wall * 1e3)) if hollow else '')
        self._say(line)
        return line

    def hollow_selected(self):
        """'Hollow the selected': its inside derived at the Wall, the outside unmoved (tools.hollow; Re-hollow after)"""
        sel = list(getattr(self.win.viewport, 'selected', None) or [])
        if len(sel) != 1:
            return self._say('Hollow: select one object.')
        wall = float(self._val('add', 'wall', 0.005) or 0.005)
        self.win.engine.post('tools.hollow', int(sel[0]), wall)
        return self._say('Hollowing with a %.1f mm wall ... (one undo step)' % (wall * 1e3))

    # ---- helpers -------------------------------------------------------------------------------------------------------
    def _say(self, line):
        try:
            self.win.status.showMessage(line, 6000)
        except Exception:
            pass
        s = getattr(self.win, 'tool_strip', None)
        rl = getattr(s, 'reg_line', None) if s is not None else None
        if rl is not None:
            try:
                rl.setText(line)
            except RuntimeError:
                pass
        return line


class _EscKeys(QObject):
    """Esc during a registry tool's gesture puts it back (the app's shortcuts would otherwise take it)"""

    def __init__(self, rm):
        super().__init__(rm.win)
        self.rm = rm

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.ShortcutOverride, QEvent.KeyPress) and ev.key() == Qt.Key_Escape and self.rm.busy():
            if t == QEvent.ShortcutOverride:
                ev.accept()
                return True
            self.rm.cancel()
            return True
        return False


def install(win):
    from PySide6.QtWidgets import QApplication
    win.reg_mouse = RegMouse(win)
    win.viewport.reg_mouse = win.reg_mouse
    win.reg_mouse_keys = _EscKeys(win.reg_mouse)
    QApplication.instance().installEventFilter(win.reg_mouse_keys)
    return win.reg_mouse
