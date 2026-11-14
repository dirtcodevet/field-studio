"""Field Studio tools (DESIGN_NOTES 10d): PENCIL, CUT, SPLICE, FILL AND SPLICE, edit by LAYER.

Importing this package registers the tool commands with core.history (so replay() and load_project() logs that
contain them work). Every tool is a Command (do / undo / to_log), deterministic, built from core nodes (L is
tracked by the core) or written through the brush undo record (per-brick L re-measured).

register(ui_hooks) adds the tools to the app; see its docstring and CONTRACT.md 'tools'.
"""
import numpy as np

from .common import node_label, get_data, TOOLDATA
from .cut import Cut, outline_from_screen, build_cutter
from .shapecut import ShapeCut               # fs-shell 11-11 (design 4.18): cut with a shape, registered for replay
from .masses import AddMass, SetMassBlend, RemoveMass   # fs-shell 11-11 (design 4.5): masses, registered for replay
from .splice import Splice, FillSplice, profile_to_rr
from .layers import LayerSmooth, DEFAULT_LAYER_SETTINGS, bands
from .cavity import FillCavity             # fs-brush 11-11 (design D4): fill a cavity solid, registered for replay
from ..core import brush as BR

__all__ = ['Cut',
           'outline_from_screen', 'Splice', 'FillSplice', 'LayerSmooth', 'register', 'node_label', 'get_data',
           'CutMode', 'PickPairMode', 'LayerSmoothMode']


def _hook(h, name, default=None):
    if isinstance(h, dict):
        return h.get(name, default)
    return getattr(h, name, default)


class _Mode:
    """Base for interactive tool modes: the viewport sends on_press / on_drag / on_release (pixel x, y)."""

    def __init__(self, hooks):
        self.hooks = hooks
        self.hist = _hook(hooks, 'history')

    def _pick(self, x, y):
        f = _hook(self.hooks, 'pick')
        if f is not None:
            return f(x, y)
        from ..render import pick
        w, h = _hook(self.hooks, 'viewport_size')()
        return pick(self.hist.scene, _hook(self.hooks, 'get_camera')(), x, y, w, h)

    def _status(self, msg):
        f = _hook(self.hooks, 'status')
        if f:
            f(msg)

    def _done(self):
        f = _hook(self.hooks, 'refresh')
        if f:
            f()

    def overlay(self):
        """Screen polyline to draw while the mode is active ([(x, y), ...]), or []."""
        return []


class CutMode(_Mode):
    """Draw a closed outline on the screen; on release it cuts the selected object (or every visible one)."""

    def __init__(self, hooks, mode='remove', distance=0.02, kerf=0.0, blend=0.0):
        super().__init__(hooks)
        self.mode, self.distance, self.kerf, self.blend = mode, distance, kerf, blend
        self.screen = []

    def on_press(self, x, y):
        self.screen = [(x, y)]

    def on_drag(self, x, y):
        lx, ly = self.screen[-1]
        if (x - lx) ** 2 + (y - ly) ** 2 >= 9:
            self.screen.append((x, y))

    def on_release(self, x, y):
        self.on_drag(x, y)
        out = None
        if len(self.screen) >= 3:
            sel = _hook(self.hooks, 'get_selection')
            oid = sel() if sel else None
            sc = self.hist.scene
            oids = [oid] if oid is not None else [o.id for o in sc.objects if o.visible]
            w, h = _hook(self.hooks, 'viewport_size')()
            eye, dirs = outline_from_screen(_hook(self.hooks, 'get_camera')(), self.screen, w, h)
            out = self.hist.push(Cut(oids, eye, dirs, self.mode, self.distance, self.kerf, self.blend))
            self._status('Cut (%s): %d new object(s)' % (self.mode, len(out)))
        self.screen = []
        self._done()
        return out

    def overlay(self):
        return list(self.screen) + (self.screen[:1] if self.screen else [])


class PickPairMode(_Mode):
    """Two clicks: face A then face B. kind 'splice' -> Splice; 'fill' -> FillSplice with `profile`."""

    def __init__(self, hooks, kind='splice', blend=0.004, profile=None, profile_b=None):
        super().__init__(hooks)
        self.kind, self.blend = kind, blend
        self.profile = profile or {'shape': 'circle', 'radius': 0.006}
        self.profile_b = profile_b
        self.first = None

    def on_press(self, x, y):
        pass

    def on_drag(self, x, y):
        pass

    def on_release(self, x, y):
        r = self._pick(x, y)
        if not r.hit:
            return None
        if self.first is None:
            self.first = r
            self._status('Face A picked: now pick face B')
            return None
        a, b = self.first, r
        self.first = None
        if self.kind == 'splice':
            out = self.hist.push(Splice(a.object_id, a.point.tolist(), a.normal.tolist(), b.object_id,
                                        b.point.tolist(), b.normal.tolist(), self.blend))
        else:
            out = self.hist.push(FillSplice(a.object_id, a.point.tolist(), a.normal.tolist(), b.object_id,
                                            b.point.tolist(), b.normal.tolist(), self.profile, self.profile_b,
                                            self.blend))
        self._done()
        return out


class LayerSmoothMode(_Mode):
    """A brush drag that smooths one layer ('detail' or 'shape'); one undo step per drag."""

    def __init__(self, hooks, settings=None):
        super().__init__(hooks)
        self.settings = dict(DEFAULT_LAYER_SETTINGS)
        self.settings.update(settings or {})
        self.oid = None
        self.last = None

    def _samples(self, r):
        if self.last is None:
            return [{'p': r.point.tolist(), 'n': r.normal.tolist()}]
        smp = BR.space_samples([self.last, r.point], [r.normal, r.normal], self.settings['radius'],
                               self.settings['spacing'])
        return smp[1:]

    def on_press(self, x, y):
        r = self._pick(x, y)
        self.oid, self.last = None, None
        if r.hit:
            self.oid = r.object_id
            self.hist.push(LayerSmooth(self.oid, self.settings, self._samples(r)))
            self.last = r.point

    def on_drag(self, x, y):
        if self.oid is None:
            return
        r = self._pick(x, y)
        if r.hit and r.object_id == self.oid:
            s = self._samples(r)
            if s:
                self.hist.extend_stroke(s)
                self.last = np.asarray(s[-1]['p'])

    def on_release(self, x, y):
        self.oid, self.last = None, None
        self._done()


def register(ui_hooks):
    """Add the tools to the app. ui_hooks: a dict or object with
        history                      core.history.History (required)
        add_action(menu, label, callback, shortcut=None)   add a menu entry / button (required)
        get_selection() -> oid|None  the selected object (required for cut / edit)
        get_camera() -> Camera, viewport_size() -> (w, h)  (required for the screen tools)
        set_tool_mode(mode|None)     route viewport mouse events to mode.on_press/on_drag/on_release(x, y)
                                     and draw mode.overlay() (a screen polyline); None restores orbit/brush
        pick(x, y) -> PickResult     optional (default: render.pick with the camera)
        ask(title, fields) -> dict|None   optional dialog: fields {name: default}; returns edited values
        status(msg), refresh()       optional
        push(cmd)                    optional: run a command where the app runs its edits (default history.push)
    Returns {label: callback} of everything added (the app can also bind them to buttons).
    """
    add = _hook(ui_hooks, 'add_action')
    set_mode = _hook(ui_hooks, 'set_tool_mode') or (lambda m: None)
    ask = _hook(ui_hooks, 'ask')
    hist = _hook(ui_hooks, 'history')
    added = {}

    def _with(title, fields):
        if ask is None:
            return dict(fields)
        return ask(title, fields)

    def cut(mode):
        def cb():
            v = _with('Cut (%s)' % mode, {'distance': 0.02, 'kerf': 0.0, 'blend': 0.0})
            if v is not None:
                set_mode(CutMode(ui_hooks, mode, float(v['distance']), float(v['kerf']), float(v['blend'])))
        return cb

    def pair(kind):
        def cb():
            fields = {'blend': 0.004}
            if kind == 'fill':
                fields.update({'radius_a': 0.006, 'radius_b': 0.006})
            v = _with('Splice' if kind == 'splice' else 'Fill and splice', fields)
            if v is None:
                return
            pa = {'shape': 'circle', 'radius': float(v.get('radius_a', 0.006))}
            pb = {'shape': 'circle', 'radius': float(v.get('radius_b', 0.006))}
            set_mode(PickPairMode(ui_hooks, kind, float(v['blend']), pa, pb))
        return cb

    def layer(which):
        def cb():
            v = _with('Smooth %s' % which, {'radius': 0.01, 'strength': 1.0, 'detail_scale': 0.002})
            if v is not None:
                v['layer'] = which
                set_mode(LayerSmoothMode(ui_hooks, v))
        return cb

    def tool_off():
        set_mode(None)

    entries = [('Tools', 'Cut (remove)', cut('remove'), 'K'),
               ('Tools', 'Cut and separate', cut('separate'), 'Shift+K'),
               ('Tools', 'Splice', pair('splice'), 'J'),
               ('Tools', 'Fill and splice', pair('fill'), 'Shift+J'),
               ('Tools', 'Smooth detail (keep shape)', layer('detail'), None),
               ('Tools', 'Smooth shape (keep detail)', layer('shape'), None),
               ('Tools', 'Exit tool', tool_off, 'Esc')]
    for menu, label, cb, sc in entries:
        if add is not None:
            add(menu, label, cb, sc)
        added[label] = cb
    return added
