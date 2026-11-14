"""11-09 (READY 1): the scripted sculpt API - Field Studio driven from Python, on a COPY of a project.

    from fieldstudio.script import Session
    s = Session('my_sculpt.fsproj', work='my_run')   # copies it first
    her = s.find('editable copy')
    p, n = s.surface(her, near=[0.03, 1.55, 0.09])                   # a point on her surface + its normal
    blk = s.add_primitive('box', {'size': [0.018, 0.016, 0.012], 'round': 0.003}, at=p + 0.002 * n, scale=[1, .45, 1])
    s.merge(her, [blk], k=0.003, gap=0.0005)                            # protected parts kept apart (the eyes)
    s.stroke(her, 'mask', [q0, q1, q2], radius=0.002)                   # world path; projected onto her surface
    lid = s.protect_masked(her, 'lid'); s.unprotect(lid); s.set_root(lid); s.protect(lid)
    s.stroke(her, 'swell', [q], radius=0.002, contacts=[lid])
    s.stroke(her, 'grab', [q], radius=0.01, drag=[0, 0.002, 0])
    s.save('result.fsproj')

Everything goes through the same commands, GPU hooks (gpu/hooks.py) and part actions (core/actions.py) as the app,
so the result is what the app's buttons would give for the same dabs, it lands in the undo history and the project's log,
and a saved result replays to the same state hash. The app's brush defaults (ui/panels.py) are used unless a stroke
overrides them. Paths are world points; each is moved onto the object's surface (Newton steps on its field) and the
dabs are laid along the polyline at spacing x radius, as the app lays them along the cursor's picks.

A Session refuses to save over the project it was opened from, and never writes into it (it works on a copy)."""
import copy
import math
import os
import shutil
import time

import numpy as np

from .core import history as HIST
from .core.history import History, AddObject, BrushStroke, load_project, save_project
from .core import actions as ACT
from .core import brush as BR
from .core import ops

MIN_BRUSH_RADIUS = 0.0005           # ui/engine.py: the core lifts anything under 2 lattice steps to 2 h anyway
MAX_BRUSH_RADIUS = 0.12             # ui/engine.py (phase0/script_api_test.py checks these equal the engine's)
NODES_PER_RADIUS = 80


def max_radius_for(h):
    """ui/engine.py max_radius_for (the app's per-lattice brush limit; kept here so a script needs no Qt)"""
    return min(MAX_BRUSH_RADIUS, max(0.04, NODES_PER_RADIUS * float(h)))


def app_settings(tool):
    """the brush settings a stroke of `tool` carries in the app by default (ui/panels.py ToolPanel)"""
    s = copy.deepcopy(BR.DEFAULT_SETTINGS)
    s.update({'tool': tool, 'radius': 0.012, 'strength': 0.5, 'curve': [1.0, 0.95, 0.8, 0.5, 0.2, 0.0],
              'collide': True, 'collide_gap': 0.0003, 'contact_gap': 0.0, 'use_mask': True, 'face_cone': 80.0,
              'face_only': False, 'connected': tool == 'mask', 'paint_dir': tool == 'mask'})
    return s


class _LogHistory(History):
    """ui/engine.py UIHistory (kept here so a script needs no Qt): an opened project's saved log is the op log's
    baseline, so replay and save carry the whole history (History.log() alone holds only this session's commands)"""

    def __init__(self, scene=None):
        super().__init__(scene)
        self.base_log = []

    def log(self):
        return list(self.base_log) + super().log()


class Session:
    def __init__(self, project=None, work=None, gpu=True, log=print):
        """project: a .fsproj folder (copied into work/ first) or None (an empty scene). work: the run's folder
        (created). gpu: strokes on the card as in the app (False: the CPU brushes)."""
        self.log = log or (lambda *a: None)
        self.work = os.path.abspath(work or os.path.join(os.getcwd(), 'fs_script_run'))
        os.makedirs(self.work, exist_ok=True)
        self.source = os.path.abspath(project) if project else None
        self.times = []
        t0 = time.perf_counter()
        if project:
            cp = os.path.join(self.work, 'opened_copy.fsproj')
            shutil.rmtree(cp, ignore_errors=True)
            shutil.copytree(self.source, cp)
            loaded = load_project(cp)                     # hash-checked
            self.hist = _LogHistory(loaded.scene)          # as the app's File > Open (ui/engine.py _open)
            self.hist.base_log = list(loaded.loaded_log)
            if getattr(loaded, 'loaded_done', None):                  # fs-trust 11-11: the undo journal
                self.hist.base_log = list(loaded.loaded_base)
                self.hist.done = list(loaded.loaded_done)
        else:
            self.hist = _LogHistory()
        self.gv = self.gs = None
        if gpu:
            from .gpu.viewport import GpuView
            from .gpu import hooks as GH
            self.gv = GpuView()
            self.gs = GH.install(self.gv, lambda scene: scene is self.hist.scene, lambda m: self.log(m))
        self._t('open', t0)

    # ---- marks (fs-shell 11-11, design 4.14) -----------------------------------------------------------------------
    def marks(self, shared=False):
        """the user's marks: those saved in the opened project, or (shared=True) the shared view being marked in now
        (tools/marks.read_shared): [{'n', 'kind' pin / region, 'name', 'note', 'object', 'centre_mm', 'size_mm', 'still',
        'camera', 'q' / 'outline_q' / 'samples_q' (the object's own frame, m)}]"""
        from .tools import marks as MK
        if shared:
            return MK.read_shared().get('marks', [])
        return MK.project_marks(os.path.join(self.work, 'opened_copy.fsproj')) if self.source else []

    # ---- bookkeeping ---------------------------------------------------------------------------------------------
    def _t(self, what, t0):
        self.times.append((what, round(time.perf_counter() - t0, 3)))

    @property
    def scene(self):
        return self.hist.scene

    def push(self, cmd, label=None):
        if label:
            cmd.ui_label = label
        out = self.hist.push(cmd)
        if self.gv is not None:
            self.gv.note_change()
        return out

    def objects(self):
        return {o.name: o.id for o in self.scene.objects}

    def find(self, text):
        """the id of the one object whose name contains text"""
        hits = [o.id for o in self.scene.objects if text in o.name]
        if len(hits) != 1:
            raise KeyError('%d objects match %r: %s' % (len(hits), text, list(self.objects())))
        return hits[0]

    def state_hash(self):
        return self.scene.state_hash()

    # ---- geometry --------------------------------------------------------------------------------------------------
    def field(self, oid, P):
        from .tools.protected_merge import field_of
        return field_of(self.scene, self.scene.pack(), self.scene.get(oid), np.atleast_2d(np.asarray(P, float)))

    def surface(self, oid, near, iters=8):
        """-> (point, unit normal) on the object's surface nearest to `near` (world): Newton steps on its field"""
        p = np.asarray(near, float).copy()
        e = 2e-5
        for _ in range(iters):
            f = self.field(oid, p)[0]
            g = np.array([(self.field(oid, p + e * a)[0] - self.field(oid, p - e * a)[0]) / (2 * e) for a in np.eye(3)])
            gg = float(g @ g)
            if gg < 1e-12:
                break
            p = p - f * g / gg
            if abs(f) < 1e-8:
                break
        g = np.array([(self.field(oid, p + e * a)[0] - self.field(oid, p - e * a)[0]) / (2 * e) for a in np.eye(3)])
        return p, g / max(np.linalg.norm(g), 1e-12)

    # ---- objects ---------------------------------------------------------------------------------------------------
    def add(self, spec, label=None):
        t0 = time.perf_counter()
        o = self.push(AddObject(spec), label or 'Add %s' % spec.get('name', 'object'))
        self._t('add ' + spec.get('name', ''), t0)
        return o.id

    def add_primitive(self, kind, params, at, scale=(1.0, 1.0, 1.0), rotate=(0.0, 0.0, 0.0), name=None, h=0.00025):
        """a primitive ('box' {size, round}, 'sphere' {radius}, ... any ops.Node kind) scaled per axis and rotated
        (deg) about its centre, placed at `at` (world)"""
        base = ops.Node('transform', {'translate': [0, 0, 0], 'rotate': list(map(float, rotate)),
                                      'scale': list(map(float, scale))}, [ops.Node(kind, dict(params))]).to_dict()
        return self.add({'name': name or kind, 'base': base, 'translate': np.asarray(at, float).tolist(), 'h': h})

    def protected_ids(self):
        from .core.scene import is_protected
        return [o.id for o in self.scene.objects if is_protected(o)]

    def merge(self, target, shapes, protected=None, use_mask=False, k=0.003, gap=0.0005):
        """Merge with protected parts (the app's Merge dialog): shapes into target, the protected parts (default:
        every locked protected part) kept apart by gap. -> the command's info"""
        from .tools.protected_merge import MergeProtected
        t0 = time.perf_counter()
        prot = self.protected_ids() if protected is None else list(protected)
        info = self.push(MergeProtected(target, list(shapes), prot, use_mask, k, gap), 'Merge with protected parts')
        self._t('merge', t0)
        return info

    # ---- parts -----------------------------------------------------------------------------------------------------
    def protect_masked(self, oid, name):
        t0 = time.perf_counter()
        msg, changed = ACT.protect_masked(self.hist, oid, name)
        self._after(msg)
        self._t('protect_masked', t0)
        return self.find('%s (protected)' % name)

    def unprotect(self, pid):
        t0 = time.perf_counter()
        self._after(ACT.set_locked(self.hist, pid, False, self.push)[0])
        self._t('unprotect', t0)

    def protect(self, pid):
        """Protect again (an attached part's root is fused); then the app's overlap check. -> new overlap nodes"""
        t0 = time.perf_counter()
        self._after(ACT.set_locked(self.hist, pid, True, self.push)[0])
        n = self.check_overlaps()
        self._t('protect', t0)
        return n

    def set_root(self, pid):
        t0 = time.perf_counter()
        msg, ok = ACT.set_root(self.hist, pid, self.push)
        self._after(msg)
        self._t('set_root', t0)
        if not ok:
            raise RuntimeError(msg)

    def check_overlaps(self):
        msg, tot = ACT.check_overlaps(self.hist)
        self.log(msg)
        return tot

    def _after(self, msg):
        if self.gv is not None:
            self.gv.note_change()
        self.log(msg)

    # ---- strokes ---------------------------------------------------------------------------------------------------
    def stroke(self, oid, tool, path, radius=0.002, strength=0.5, drag=None, contacts=None, settings=None, batch=16):
        """One brush stroke as the app makes it. path: world points (moved onto the object's surface). Grab /
        stretch: path[0] is the press point, drag the world vector it is pulled by (laid in 0.35 r steps, as the
        app). contacts: Swell / Settle contacts besides the locked parts. -> the BrushStroke"""
        from .core.scene import is_protected
        t0 = time.perf_counter()
        o = self.scene.get(oid)
        if is_protected(o):
            raise RuntimeError('%s is protected: unprotect it to sculpt it' % o.name)
        s = app_settings(tool)
        s.update(settings or {})
        s['radius'] = float(radius)
        s['strength'] = float(strength)
        if tool in ('swell', 'settle'):
            s['contact_ids'] = [int(i) for i in (contacts or []) if int(i) != int(oid)]
            s['collide'] = True
            s['collide_gap'] = float(s.get('contact_gap', 0.0))
        rmax = max_radius_for(o.sculpt.h) * max(float(o.transform['scale']), 1e-6)
        s['radius'] = min(max(s['radius'], MIN_BRUSH_RADIUS), rmax)
        rad = s['radius']
        pts = [self.surface(oid, q) for q in np.atleast_2d(np.asarray(path, float))]
        dabs = []
        if tool in ('grab', 'stretch'):
            anchor, n0 = pts[0]
            v = np.asarray(drag, float)
            L = float(np.linalg.norm(v))
            k = max(1, int(math.ceil(L / (0.35 * rad))))
            a = anchor.copy()
            for _ in range(k):
                c = a if tool == 'grab' else anchor
                dabs.append({'p': c.tolist(), 'n': n0.tolist(), 'v': (v / k).tolist()})
                if tool == 'grab':
                    a = a + v / k
            first = []
        else:
            step = max(float(s['spacing']) * rad, 1e-5)
            p0, n0 = pts[0]
            first = [{'p': p0.tolist(), 'n': n0.tolist()}]
            carry = 0.0
            prev, prev_n = p0, n0
            for p, n in pts[1:]:
                seg = p - prev
                L = float(np.linalg.norm(seg))
                if L > 4.0 * rad:
                    dabs.append({'p': p.tolist(), 'n': n.tolist()})
                    carry = 0.0
                elif L > 0:
                    t = step - carry
                    while t <= L:
                        q = prev + seg * (t / L)
                        nn = prev_n + (n - prev_n) * (t / L)
                        nn = nn / max(np.linalg.norm(nn), 1e-12)
                        dabs.append({'p': q.tolist(), 'n': nn.tolist()})
                        t += step
                    carry = L - (t - step)
                prev, prev_n = p, n
            if carry >= 0.2 * step:
                # the app's release rule (ui/engine.py, hand2 slow strokes): the carried path under one spacing laid
                # as a partial dab at the end, strength scaled by carry / step (deposit per unit path unchanged)
                dabs.append({'p': prev.tolist(), 'n': prev_n.tolist(), 'k': round(float(carry) / step, 4)})
        cmd = BrushStroke(oid, s, first)
        cmd.live = True
        self.push(cmd, 'Brush stroke (%s)' % tool)
        for i in range(0, len(dabs), batch):
            self.hist.extend_stroke(dabs[i:i + batch])
        if cmd.samples or first:
            self.hist.end_stroke()
        else:                                            # a press that never moved: no undo step (as the app)
            self.hist.undo()
            self.hist.drop_undone_last()
        if self.gv is not None:
            self.gv.note_change()
        self._t('stroke %s r %.1f mm (%d dabs)' % (tool, rad * 1e3, len(dabs) + len(first)), t0)
        return cmd

    def idle(self, steps=None):
        """11-09: the app's idle time after a release: the last stroke's further release passes (all, or `steps`) ->
        the passes run"""
        n = 0
        while self.hist.idle_pending() and (steps is None or n < steps):
            self.hist.idle_step()
            n += 1
        if n and self.gv is not None:
            self.gv.note_change()
        return n

    # ---- history, files, views -------------------------------------------------------------------------------------
    def undo(self, n=1):
        for _ in range(n):
            self.hist.undo()
        if self.gv is not None:
            self.gv.note_change()

    def redo(self, n=1):
        for _ in range(n):
            self.hist.redo()
        if self.gv is not None:
            self.gv.note_change()

    def save(self, path):
        """save as a project (with its Open in Field Studio.cmd); refuses the project this session was opened from
        and any existing folder"""
        path = os.path.abspath(path)
        if self.source and os.path.normcase(path) == os.path.normcase(self.source):
            raise RuntimeError('refused: %s is the project this session was opened from (it works on a copy)' % path)
        if os.path.exists(path):
            raise RuntimeError('refused: %s exists' % path)
        t0 = time.perf_counter()
        save_project(self.hist, path)
        self._t('save', t0)
        return path

    def replay_hash(self):
        """the state hash a fresh scene reaches by replaying this session's log (the app's File > Verify replay)"""
        return HIST.replay(self.hist.log()).scene.state_hash()

    def render(self, path, target, dist, view=(0.0, 0.0, 1.0), fov=35.0, w=900, h=700, passes=60, headlight=0.6):
        """a view through the card (as the app's viewport, the sculpt headlight on) saved as PNG -> path"""
        from PIL import Image
        from .render.camera import Camera
        if self.gv is None:
            raise RuntimeError('render needs gpu=True')
        d = np.asarray(view, float)
        d = d / np.linalg.norm(d)
        cam = Camera(pos=tuple(np.asarray(target, float) + dist * d), target=tuple(np.asarray(target, float)), fov=fov)
        self.gv.aid.update({'headlight': headlight})
        self.gv.note_change()
        fr = None
        for _ in range(passes):
            fr = self.gv.frame(self.scene, cam, w, h)
        Image.fromarray(np.asarray(fr.rgb)).save(path)
        return path
