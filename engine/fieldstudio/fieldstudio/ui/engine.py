"""The engine thread: the ONLY thread that touches the scene (edits, picks, renders).

The UI thread posts jobs and view requests; results come back as Qt signals (queued to the UI thread). This keeps
the contract's threading rule by construction: a render never runs while an edit runs, because both run here, one
after the other. The numba kernels release the GIL (nogil=True), so the UI keeps running during a render.

Jobs (tuples, processed in order, all before the next render):
  ('add', spec)                      AddObject; the new object becomes the selection
  ('param', oid, nid, params)        SetNodeParams, merged with the previous edit of the same key within 1.5 s
  ('props', oid, props)              SetObjectProps, merged the same way
  ('wrap', oid, nid, kind, params)   put a mutation node above node nid (SetBase)
  ('deform', oid, kind, params, drag) fs-shell 11-11: the gizmo's deformer (bend / twist / taper / offset) on the
                                      object's root, added once and then dialled; one drag = one undo step
  ('push', command)                  fs-shell 11-11: a ready Command (the shape cutter's), one undo step; the object
                                      ids it returns become the selection
  ('unwrap', oid, nid)               remove node nid (a mutation -> its child; a boolean operand -> the other one)
  ('boolean', a, b, kind, blend)     B (in A's frame) joins A's base tree under a boolean node; B is removed
                                      (two commands, one undo step)
  ('remove', [oids]) ('duplicate', oid) ('undo',) ('redo',) ('goto', n)
  hand2f: ('merge', [oids], op, k_m, frame)  CombineObjects (Merge / Subtract / Intersect, sculpt carried exactly);
          'boolean' now runs the same command, so a boolean keeps both sculpt layers
          ('mirror_dup', oid) MirrorDuplicate about x = 0   ('mask', oid, 'clear'|'invert') MaskEdit
          ('view', 'hide'|'isolate'|'show_all'|'toggle', [oids])  view-only hide (not logged, not in the hash)
          ('autosave', [paths], extra)  snapshot under the lock, written on a worker thread
  ('stroke_begin', view, x, y, settings) ('stroke_move', view, x, y) ('stroke_end',)
  ('new',) ('open', path) ('save', path, extra) ('frame', [oids]) ('verify_replay',) ('call', fn, token)
"""
import collections
import copy
import math
import os
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..core import (History, AddObject, RemoveObject, SetBase, SetNodeParams, SetObjectProps, BrushStroke,
                    save_project, load_project, replay, ops)
from ..core import brush as BR
from ..core.combine import CombineObjects, DuplicateObject, MirrorDuplicate, MaskEdit
from ..core.history import snapshot_project, write_snapshot
from ..render import Camera, render, pick
from .. import tools as FT          # registers the tool commands (pencil, cut, splice, fill, layer smooth) for replay

MERGE_WINDOW = 1.5        # s: consecutive edits of the same parameter merge into one undo step
REFINE_DELAY = 0.15       # s idle before the full-quality frame
INTERACTIVE_HOLD = 0.25   # s after an interaction during which frames render at preview quality
PW_IDLE_MS = float(os.environ.get('FSB_PW_IDLE_MS', '25'))   # fs-brush-2 11-13: a write behind a release, per turn, no stroke live
PW_DRAG_MS = float(os.environ.get('FSB_PW_DRAG_MS', '8'))    # ... and per turn beside a live stroke's frames



def _warm_cpu_brushes():
    """11-09: the CPU brush paths loaded from the compile cache at start-up (core/collide.warm_cpu_paths)"""
    try:
        from ..core import collide as _CL
        _CL.warm_cpu_paths()
    except Exception:
        pass


HEADLIGHT_DEFAULT = 0.6     # sculpt headlight, a fraction of the key's irradiance (11-08 night; 0 in review renders)

class Refused(Exception):
    """V1.3: a job the engine declines (a status message, not a failure)."""


SCRUB_IDLE_S = float(os.environ.get('FSB_SCRUB_IDLE_S', '1.0'))   # fs-brush 11-12: idle seconds before the store check slices
SCRUB_ON = os.environ.get('FSB_SCRUB', '1') != '0'
POOL_IDLE_S = 2.0 if not os.environ.get('FS_NO_POOL_IDLE') else 1e6            # 11-09: seconds idle before CuPy's cached blocks are released (gpu/scene.release_cached)
POOL_RELEASE_AFTER = ('open', 'merge', 'open_her', 'her_writeback', 'consolidate', 'new', 'open_region')
MIN_BRUSH_RADIUS = 0.0005   # V1.3: m; the core lifts anything under 2 lattice steps to 2h anyway
MAX_BRUSH_RADIUS = 0.12     # hand1: m, the absolute ceiling (was 0.04 for every object)
NODES_PER_RADIUS = 80       # hand1: a dab touches (2r/h)^3 nodes; r <= 80 h keeps it at or under 160^3 = 4 M nodes,
                            # the V1.3 bound (40 mm at the 0.5 mm lattice; 80 mm on a 1 mm bust lattice)
STROKE_BUDGET_MS = 60.0     # hand1: dab time per engine batch; hand2b: the dabs past it wait for the next batch
BACKLOG_BATCHES = 4         # hand2b: a backlog longer than this many batches is reduced (continuity kept)
MAX_SPACING = 0.4           # hand2b: brush dabs of a reduced backlog stay within 0.4 r of each other (no beads)


def max_radius_for(h, tool=None):
    """hand1: the largest brush radius for a sculpt lattice spacing h (m). fs-brush 11-12 (a block-out test: a large
    Stretch on a sphere should stretch the full shape when used large enough; tk4: 120 / 100
    mm Stretches were cut to 80 mm on New bust's 1 mm lattice): the move tools (Grab, Stretch, Snake Hook) reach
    MOVE_NODES_PER_RADIUS h, up to MAX_MOVE_RADIUS: their cost is the moved region, not a dab ball per mouse event"""
    if tool in ('grab', 'stretch', 'snake') and MOVE_NODES_PER_RADIUS > 0:
        return min(MAX_MOVE_RADIUS, max(0.04, MOVE_NODES_PER_RADIUS * float(h)))
    return min(MAX_BRUSH_RADIUS, max(0.04, NODES_PER_RADIUS * float(h)))


MOVE_NODES_PER_RADIUS = float(os.environ.get('FSB_MOVE_NODES', '200'))   # 0: the move tools as every brush (80 h)
MAX_MOVE_RADIUS = float(os.environ.get('FSB_MOVE_RMAX', '0.25'))


class UIHistory(History):
    """History whose op log can start with a loaded project's log (when its replay did not reproduce the hash)."""

    _gen = 0

    def __init__(self, scene=None):
        super().__init__(scene)
        self.base_log = []
        UIHistory._gen += 1
        self.gen = UIHistory._gen        # a new history = a new scene (ids restart): F4, id-keyed plug-ins clear on a change

    def log(self):
        return list(self.base_log) + super().log()


def _label(c):
    lab = getattr(c, 'ui_label', None)
    if lab:
        return lab
    if isinstance(c, AddObject):
        sp = c.spec
        return 'Add %s' % (sp.get('name') or sp.get('kind') or (sp.get('base') or {}).get('kind', 'object'))
    if isinstance(c, FT.LayerSmooth):
        return 'Smooth %s layer (%d dabs)' % (c.settings.get('layer', '?'), len(c.samples))
    if isinstance(c, BrushStroke):
        return '%s stroke (%d dabs)' % (c.settings['tool'].capitalize(), len(c.samples))
    if isinstance(c, SetNodeParams):
        k = ', '.join(sorted(c.params))
        return 'Edit %s' % k
    if isinstance(c, SetObjectProps):
        return 'Object %s' % ', '.join(sorted(c.props))
    if isinstance(c, RemoveObject):
        return 'Delete object %d' % c.oid
    if isinstance(c, CombineObjects):
        return '%s %d objects%s' % ({'union': 'Merge', 'subtract': 'Subtract', 'intersect': 'Intersect'}[c.op],
                                    len(c.oids), ' (smooth %.0f mm)' % (c.k * 1000) if c.k > 0 else '')
    if isinstance(c, MirrorDuplicate):
        return 'Mirror-duplicate X'
    if isinstance(c, DuplicateObject):
        return 'Duplicate'
    if isinstance(c, MaskEdit):
        return 'Mask %s' % c.op
    return getattr(c, 'label', type(c).__name__)


def euler_from_matrix(R):
    """R = Rz Ry Rx -> Euler XYZ degrees (inverse of ops.rot_matrix)."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    b = math.asin(sy)
    if abs(sy) < 0.999999:
        a = math.atan2(R[2, 1], R[2, 2])
        c = math.atan2(R[1, 0], R[0, 0])
    else:
        a = math.atan2(-R[1, 2], R[1, 1])
        c = 0.0
    return [math.degrees(a), math.degrees(b), math.degrees(c)]


def _renumber(tree, start):
    """Give every node of a tree dict a fresh id from start; -> (tree, next free id)."""
    t = copy.deepcopy(tree)
    n = [start]

    def walk(d):
        d['id'] = n[0]
        n[0] += 1
        for ch in d.get('children', []):
            walk(ch)
    walk(t)
    return t, n[0]


def _find_parent(tree, nid):
    """-> (parent dict or None, index, node dict) in a tree dict."""
    if tree['id'] == nid:
        return None, -1, tree
    stack = [tree]
    while stack:
        d = stack.pop()
        for i, ch in enumerate(d.get('children', [])):
            if ch['id'] == nid:
                return d, i, ch
            stack.append(ch)
    return None, -1, None


def _ray_plane(cam, x, y, w, h, p0, n):
    o, d = cam.ray(x, y, w, h)
    den = float(d @ n)
    if abs(den) < 1e-9:
        return None
    t = float((p0 - o) @ n) / den
    return o + d * t


# fs-brush 11-12 (the toolkit review, P1: full 3D, the same from any view): a move tool's drag is
# resolved on a constraint taken from the FORM, not the camera. Default: the plane through the press point that holds
# the press normal and faces the camera as much as it can (seen from the side it is the view plane, as before; from 3/4
# a drag along the normal's on-screen direction moves along the true normal); as the view turns head-on it turns smoothly
# into the tangent plane (a slide). Ctrl (invert): the normal only, from any view (head-on: the drag up / down drives it).
# FSB_DRAG_FORM=0: the camera's plane, as before.
# fs-brush 11-12, decided: the screen plane by default. Grab / Stretch / Snake carry the surface in the screen
# plane by default (as Blender / ZBrush; tk4: a block-out from the side, 18.9 / 21.6 / 14.3 mm vs the form rule's 19.6 /
# 23.0 / 19.1), the form rule on a visible switch: the brush panel's 'Drag along the form' (settings['drag_form'])
DRAG_FORM = os.environ.get('FSB_DRAG_FORM', '0') != '0'
GRAB_FLOW = os.environ.get('FSB_GRAB_FLOW', '1') != '0'
STRETCH_FLOW = os.environ.get('FSB_STRETCH_FLOW', '1') != '0'   # fs-brush 11-12: Stretch as an anchored elongation   # fs-brush 11-12: the anchored flow Grab (CPU path for now)
DRAG_HEADON = 0.0872                        # sin 5 deg: below it the plane turns into the tangent plane (15 deg slid a pull from 80 deg 32 mm sideways)


def drag_constraint(f, n0):
    """-> the drag plane's normal for camera forward f and press normal n0 (see DRAG_FORM)"""
    f = np.asarray(f, float)
    n0 = np.asarray(n0, float) / max(float(np.linalg.norm(n0)), 1e-12)
    m = f - float(f @ n0) * n0
    s = float(np.linalg.norm(m))
    if s >= DRAG_HEADON:
        return m / s
    beta = s / DRAG_HEADON
    pn = (beta * m / s if s > 1e-12 else 0.0 * m) + (1.0 - beta) * n0
    return pn / max(float(np.linalg.norm(pn)), 1e-12)


def drag_on_normal(cam, x, y, w, h, p0, n0, up):
    """Ctrl: the point on the line p0 + t n0 nearest the cursor's ray; head-on (the ray along the line) the view-plane
    drag's upward part drives t"""
    o, d = cam.ray(x, y, w, h)
    n0 = np.asarray(n0, float)
    b = float(d @ n0)
    den = 1.0 - b * b
    if den > DRAG_HEADON * DRAG_HEADON:
        w0 = o - p0
        t = (float(w0 @ n0) - b * float(w0 @ d)) / den
        return p0 + t * n0
    q = _ray_plane(cam, x, y, w, h, p0, cam.basis()[0])
    if q is None:
        return None
    return p0 + float((q - p0) @ np.asarray(up, float)) * n0



class CallFailed:
    """fs-brush 11-12: the result an engine 'call' sends with callDone when its function raised: .error (the exception),
    .trace (its traceback text). Falsy, so a listener that tests its result sees a failure; failed is True."""
    failed = True

    def __init__(self, error, trace=''):
        self.error = error
        self.trace = trace

    def __bool__(self):
        return False

    def __repr__(self):
        return 'CallFailed(%r)' % (self.error,)


def _pressure_error(e):
    """fs-brush 11-11: a card failure that is memory pressure (device or pinned host memory), not a broken card"""
    t = (type(e).__name__ + ' ' + str(e))
    return any(k in t for k in ('OutOfMemory', 'cudaErrorMemoryAllocation', 'cudaErrorAlreadyMapped',
                                'cudaErrorHostMemoryAlreadyRegistered', 'out of memory', 'resource already mapped'))

class Engine(QObject):
    frameReady = Signal(object)      # dict: rgb, obj, depth, stats, cam, w, h, seq, final
    sceneChanged = Signal(object)    # snapshot dict
    hoverReady = Signal(object)      # dict: hit, point, normal, oid, seq
    boundsReady = Signal(object)     # (lo, hi)
    projectLoaded = Signal(object)   # extra dict of a loaded project (camera ...)
    message = Signal(str)
    failed = Signal(str)
    ready = Signal(float)            # warm-up seconds
    progress = Signal(int, int, str)  # warm-up step, steps, what is compiling (the launch progress)
    callDone = Signal(object)        # (token, result)
    regionDone = Signal(object)      # V1.2: the write-back report (or {'error': ...})
    # fs-brush 11-11 (design D4): a stroke changed the form's topology (parts / tunnels / cavities): {'oid',
    # 'stroke_step' (the stroke command's serial, as in snap['log_ids']), 'before', 'after', 'box_m', 'cavities',
    # 'filled' (a pocket the stroke made was filled), 'opening_closed' (a part fused or split, a tunnel closed, a cavity
    # lost: the window asks Keep / Undo)}; fs-shell's ui/topo_ui.py listens
    topoChanged = Signal(object)
    # fs-store 11-12 (an 18:24 save never finished: the write ran on a daemon thread the quit killed): every project
    # save's end, {'path', 'ok', 'hash', 'error', 'seconds'}; the window shows a failure and keeps the work marked unsaved
    saveDone = Signal(object)

    def __init__(self):
        super().__init__()
        self.hist = UIHistory()
        self.lock = threading.RLock()        # held around every scene access (edits, pack + render, picks)
        self._cv = threading.Condition()
        self._jobs = collections.deque()
        self._view = None
        self._render_due = False
        self._interactive_until = 0.0
        self._refine_at = None
        self._hover = None
        self._stop = False
        self._busy = False
        self.warm = False
        self._merge = None
        self._stroke = None
        self._select = None
        self._select_node = None
        self.frames = 0
        self.last_render_ms = 0.0
        self.threads = 0
        self._mode_view = None               # the view of the tool-mode event being handled (engine thread)
        self._dab_ms = 5.0                   # hand1: running estimate of one dab's cost (ms), for the stroke budget
        self._q0_ms = 30.0                   # hand1: the last preview frame's time, for the adaptive preview scale
        self.thinned = 0                     # hand1: dabs dropped or merged by the budget (telemetry)
        self.autosave_busy = False           # hand2f: a worker thread is writing an autosave
        self.last_autosave = None            # hand2f: (paths, hash, seconds) of the last finished autosave
        self.last_merge = None
        self.replaying = False               # hand2b: an opened project's undo history is being rebuilt (worker)
        self._adopted = False
        # phase 0 (2026-11-08): the viewport draws through the GPU field renderer (gpu/viewport.py); the CPU renderer
        # stays for the other display modes (matcap, normals, slice, mask) and as the fallback without a usable card.
        # FIELDSTUDIO_GPU=0 turns the GPU path off.
        self.use_gpu = os.environ.get('FIELDSTUDIO_GPU', '1') != '0'
        self.gpu_view = None
        self.gpu_error = None
        self.view_aid = {'headlight': HEADLIGHT_DEFAULT, 'cavity': 1.0, 'ev': 0.0}
        self.parts_view = 'shown'           # 11-09: protected parts shown (default) / ghost / hidden: VIEW state
        self.show_overlaps = True           # red 'new overlaps only' on settled frames
        self._pv_hidden = set()
        self.loop_prof = collections.deque(maxlen=4000)   # r13: per-loop phase times (ms)
        self.pw_waits = collections.deque(maxlen=200)     # fs-brush-2 11-13: (job, ms) a write behind a release was finished for
        self.thread = threading.Thread(target=self._run, name='field-engine', daemon=True)

    # ---- UI-thread API ---------------------------------------------------------------------------------------
    def start(self):
        # fs-brush 11-11 (M, head test stage 5, p44: median 15.0 -> 29.3 fps, frame p95 93 -> 59 ms, the frame's readbacks
        # 14.2 -> 2.9 ms, a dab batch 29 -> 11 ms): this thread sculpts and renders while the UI thread composites each
        # frame (reference planes, overlays: 5-7 ms of Python and numpy); at Python's default 5 ms GIL switch interval
        # every card call that released the GIL here waited up to 5 ms to get it back. 0.5 ms hands it back promptly.
        # FS_SWITCH_MS overrides (0 = Python's default).
        import sys
        ms = float(os.environ.get('FS_SWITCH_MS', '0.5'))
        if ms > 0:
            sys.setswitchinterval(ms / 1000.0)
        self.thread.start()

    def stop(self):
        # fs-store 11-12: a project save queued or being written finishes before the engine stops (the engine thread
        # drops queued jobs at a stop, and the interpreter's exit killed the old daemon writer mid-write: an 18:24
        # save). No time limit: a save is never cut short. The window waits here first with a 'Saving...' dialog.
        self.wait_saves()
        # fs-shell 11-13 (Tier 0, the exit rule; L-1142): STOP LAUNCHING, SYNC, THEN EXIT. Before: a 5 s join of this
        # thread, no device sync, the hair threads never stopped. Now ui/exit_rule.py: the launch guard refuses new passes,
        # this thread ends the job it is in (joined up to FS_EXIT_WAIT_S, 120 s), the hair's card threads are joined,
        # the device is synchronised; the record goes to ~/FieldStudio/last_exit.json
        from .exit_rule import stop_card_work
        if getattr(self, 'last_exit', None) is None:      # the window's close and app.run's end both call stop()
            self.last_exit = stop_card_work(self)
        t = getattr(self, '_autosave_thread', None)
        if t is not None and t.is_alive():     # fs-store 11-12: an autosave being written ends cleanly too
            t.join(timeout=120.0)

    def stop_thread(self, wait_s=120.0):
        """fs-shell 11-13: tell the engine thread to stop (it starts no new job and draws no new frame) and join it ->
        True when it is still alive after wait_s"""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self.thread.is_alive() and self.thread is not threading.current_thread():   # hand1b: closed before it started
            self.thread.join(timeout=wait_s)
        return self.thread.is_alive()

    def save_pending(self):
        """fs-store 11-12: a project save is queued, being snapshotted or being written"""
        with self._cv:
            if getattr(self, '_saves_open', 0) > 0 or any(j and j[0] == 'save' for j in self._jobs):
                return True
        t = getattr(self, '_save_thread', None)
        return bool(t is not None and t.is_alive())

    def _save_closed(self):
        with self._cv:
            self._saves_open = max(0, getattr(self, '_saves_open', 0) - 1)
            self._cv.notify_all()

    def wait_saves(self, timeout=None):
        """fs-store 11-12: wait until no project save is queued or being written (the engine thread runs the queued
        'save' job; then its writer thread is joined) -> True when none is left"""
        t0 = time.perf_counter()
        while self.save_pending():
            if timeout is not None and time.perf_counter() - t0 > timeout:
                return False
            if not self.thread.is_alive():
                # the engine thread is gone with a save still queued: run it here, so it is not lost
                with self._cv:
                    jobs = [j for j in self._jobs if j and j[0] == 'save']
                    for j in jobs:
                        self._jobs.remove(j)
                for j in jobs:
                    self._do(j)
            t = getattr(self, '_save_thread', None)
            if t is not None and t.is_alive():
                t.join(0.05)
            else:
                time.sleep(0.01)
        return True

    def post(self, *job):
        with self._cv:
            if job and job[0] == 'save':
                # fs-store 11-12: a save counts as pending from its post to its writer's end (the engine thread takes
                # jobs off the queue before running them: a check of the queue alone missed a save being snapshotted)
                self._saves_open = getattr(self, '_saves_open', 0) + 1
            self._jobs.append(job)
            if job[0] in ('param', 'props', 'props_many', 'oscale', 'stroke_begin', 'stroke_move'):   # hand1b: + oscale
                self._interactive_until = time.perf_counter() + INTERACTIVE_HOLD
            self._cv.notify_all()

    def request_render(self, view, interactive=True):
        with self._cv:
            self._view = view
            self._render_due = True
            if interactive:
                self._interactive_until = time.perf_counter() + INTERACTIVE_HOLD
            self._cv.notify_all()

    def request_hover(self, view, x, y):
        with self._cv:
            self._hover = (view, x, y)
            self._cv.notify_all()

    def idle(self):
        with self._cv:
            if not (not self._jobs and not self._busy and not self._render_due and self._refine_at is None
                    and self._hover is None and self.warm and not self.replaying):
                return False
        if self._pw_ready():                             # fs-brush-2 11-13: a stroke's write behind its release
            return False
        return not self.save_pending()                   # fs-store 11-12: a save in flight is work in progress

    # ---- the thread --------------------------------------------------------------------------------------------
    def _run(self):
        try:
            import numba
            n = int(os.environ.get('FIELDSTUDIO_THREADS', '0') or 0) or max(2, (os.cpu_count() or 4) - 2)
            n = max(1, min(n, numba.config.NUMBA_NUM_THREADS))
            numba.set_num_threads(n)
            self.threads = n
            t0 = time.perf_counter()
            self._warmup()
            self.warm = True
            self.ready.emit(time.perf_counter() - t0)
        except Exception:
            self.failed.emit('warm-up failed:\n' + traceback.format_exc())
            self.warm = True
        self._emit_scene()
        while True:
            with self._cv:
                while (not self._stop and not self._jobs and self._hover is None and not self._render_due
                       and not self._refine_ready() and not self._idle_rd_ready() and not self._scrub_ready()
                       and not self._pw_ready()):
                    to = None
                    if self._refine_at is not None:
                        to = max(0.005, self._refine_at - time.perf_counter())
                    elif self._scrub_pending():
                        # fs-brush 11-12: wake when the idle check of the sculpt store may run
                        to = max(0.02, SCRUB_IDLE_S - (time.perf_counter() - getattr(self, '_last_job_t', 0.0)))
                    elif getattr(self, '_pool_dirty', False) and self._stroke is None:
                        # 11-09: idle for POOL_IDLE_S with GPU work since the last release: the cached blocks go back
                        to = max(0.02, POOL_IDLE_S - (time.perf_counter() - getattr(self, '_last_job_t', 0.0)))
                        if not self._cv.wait(to):
                            self._pool_dirty = False
                            from ..gpu import scene as _GS
                            _GS.release_cached('idle')
                        continue
                    self._cv.wait(to)
                if self._stop:
                    return
                jobs = list(self._jobs)
                self._jobs.clear()
                hover = self._hover
                self._hover = None
                render_due = self._render_due
                self._render_due = False
                view = self._view
                self._busy = True
            try:
                _t0 = time.perf_counter()
                _tm = {'jobs': len(jobs)}
                changed = False
                moves = []
                for job in jobs:
                    if self._stop:                   # fs-shell 11-13: the exit rule (no new job once the app closes)
                        break
                    if job[0] == 'stroke_move':
                        moves.append(job)
                        continue
                    if moves:
                        changed |= self._moves_guarded(moves)
                        moves = []
                    try:
                        changed |= bool(self._do(job))
                    except Refused as e:
                        self.message.emit(str(e))
                    except Exception as e:
                        self.failed.emit('%s failed: %s\n%s' % (job[0], e, traceback.format_exc(limit=4)))
                if moves:
                    changed |= self._moves_guarded(moves)
                if self._pw_ready():
                    # fs-brush-2 11-13 (the release must never be a pause): a slice of the last
                    # stroke's write behind its release, every turn (the stroke stays in the view as its formula until
                    # the fine dab lands); any job needing the written field finished it first (_do)
                    _tp = time.perf_counter()
                    try:
                        with self.lock:
                            # bigger slices while no stroke is live (each turn stays well under a 100 ms gap), small
                            # ones beside a live stroke's frames
                            done_ = self.gpu_sculpt.pw_step(PW_DRAG_MS if self._stroke is not None else PW_IDLE_MS)
                    except Exception as e:
                        done_ = True
                        with self.lock:
                            n_ = self.gpu_sculpt.pw_abort_failed()
                        self.failed.emit('the stroke could not be written (%d card rows put back): %s -- %s'
                                         % (n_, e, traceback.format_exc(limit=6)))
                    if done_:
                        self.hist._notify()
                        changed = True
                    _tm['pw_ms'] = (time.perf_counter() - _tp) * 1000
                if not jobs and self._idle_rd_ready() and not self._pw_ready():
                    # 11-09: the last stroke's further release passes, one per idle turn
                    # (~0.2 s on her; a press arriving meanwhile waits for that one pass, then stops the rest)
                    _ti = time.perf_counter()
                    try:
                        with self.lock:
                            self.hist.idle_step()
                    except Exception as e:
                        # 11-09 night: a failing idle pass is not retried every idle turn (run 2 retried a dead
                        # card context forever): the refinement stops here, its region stays owed, and idle
                        # refinement stays off for the session after a card error
                        with self.lock:
                            try:
                                self.hist.idle_stop()
                            except Exception:
                                pass
                        if 'cuda' in (type(e).__module__ + type(e).__name__ + str(e)).lower():
                            self._idle_rd_off = True
                        self.failed.emit('idle refinement stopped: %s\n%s' % (e, traceback.format_exc(limit=6)))
                    if self.gpu_view is not None:
                        self.gpu_view.note_change()
                    changed = True
                    _tm['idle_rd_ms'] = (time.perf_counter() - _ti) * 1000
                if not jobs and self._scrub_ready() and not self._pw_ready():
                    # fs-brush 11-12: one idle slice of the mirror's check of the sculpt store (gpu/mirror.py scrub_step):
                    # the safety net under the light syncs at card strokes' releases
                    _ts = time.perf_counter()
                    try:
                        with self.lock:
                            r_ = self.gpu_view.scrub(self.hist.scene)
                        if r_ and r_.get('differ'):
                            changed = True
                            self.scrub_log.append((time.perf_counter(), r_))
                    except Exception as e:
                        self._scrub_off = True
                        self.failed.emit('store check stopped: %s\n%s' % (e, traceback.format_exc(limit=6)))
                    _tm['scrub_ms'] = (time.perf_counter() - _ts) * 1000
                if jobs:
                    self._last_job_t = time.perf_counter()
                    if os.environ.get('FS_DBG_JOBS'):
                        print('JOBS', round(self._last_job_t, 2), [j[0] for j in jobs][:6], flush=True)
                if self.gpu_view is not None and jobs:
                    self._pool_dirty = True
                    big = [j[0] for j in jobs if j[0] in POOL_RELEASE_AFTER]
                    if big:                              # 11-09: after the render that rebuilds for it (below)
                        self._pool_release_after = big[0]
                _t1 = time.perf_counter()
                _tm['work_ms'] = (_t1 - _t0) * 1000
                if changed:
                    if self._stroke is not None and not [j for j in jobs if j[0] not in ('stroke_move', 'stroke_flush')]:
                        # fs-brush 11-11 (M: 10-27 ms a frame in a long session): a live stroke's batch changes nothing
                        # the panels show until its release (which emits); the frame only needs the view told
                        if self.gpu_view is not None:
                            self.gpu_view.note_change()
                    else:
                        self._emit_scene()
                    render_due = True
                _t2 = time.perf_counter()
                _tm['emit_ms'] = (_t2 - _t1) * 1000
                if self._stop:                       # fs-shell 11-13: the exit rule (no new frame once the app closes)
                    hover = view = None
                if hover is not None and self._stroke is None:
                    self._do_hover(*hover)
                if view is not None:
                    interactive = time.perf_counter() < self._interactive_until or self._stroke is not None
                    if render_due:
                        self._render(view, interactive)
                    elif self._refine_ready():
                        self._render(view, False)
                    _tm['render_ms'] = (time.perf_counter() - _t2) * 1000
                    if (getattr(self, '_pool_dirty', False) and not jobs and self._stroke is None
                            and time.perf_counter() - getattr(self, '_last_job_t', 0.0) >= POOL_IDLE_S):
                        # 11-09: no job for POOL_IDLE_S (the view may still be refining its light): the cached
                        # blocks go back (the refine passes keep only what they use)
                        self._pool_dirty = False
                        from ..gpu import scene as _GS
                        _GS.release_cached('idle')
                    if getattr(self, '_pool_release_after', None) and render_due:
                        from ..gpu import scene as _GS     # 11-09: the large operation's temporaries go back
                        _GS.release_cached(self._pool_release_after)
                        self._pool_release_after = None
                    _tm['stroke'] = self._stroke is not None
                    if self.gpu_view is not None:
                        _tm['gpu'] = dict(self.gpu_view.last.get('times_ms', {}), w=view['w'], h=view['h'],
                                          mode=self.gpu_view.last.get('mode'), left=self.gpu_view.last.get('passes_left_min'),
                                          pending=str(self.gpu_view.last.get('pending')))
                    _tm['pick_ms'] = getattr(self, '_pick_ms', 0.0)
                    _tm['moves_pre_ms'] = getattr(self, '_prof_budget_ms', 0.0)
                    _tm['extend_ms'] = getattr(self, '_prof_extend_ms', 0.0)
                    self._prof_extend_ms = 0.0
                    if getattr(self, 'gpu_sculpt', None) is not None:
                        _tm['gs'] = dict(self.gpu_sculpt.stats)
                    self._pick_ms = 0.0
                    self.loop_prof.append(_tm)
                elif self._refine_ready():
                    self._refine_at = None
            except Exception:
                self.failed.emit(traceback.format_exc())
            finally:
                with self._cv:
                    self._busy = False

    scrub_log = []

    def _scrub_pending(self):
        """fs-brush 11-12: the mirror's idle check of the sculpt store has slots left (after a light sync)"""
        gv = self.gpu_view
        if gv is None or self._stroke is not None or getattr(self, '_scrub_off', False) or not SCRUB_ON:
            return False
        m = gv.mirror
        S = getattr(m, '_S', None)
        n = self.hist.scene.store.used
        return S is not None and len(S) == n and int(getattr(m, 'scrub_pos', n)) < n

    def _scrub_ready(self):
        return (self._scrub_pending()
                and time.perf_counter() - getattr(self, '_last_job_t', 0.0) >= SCRUB_IDLE_S)

    # fs-brush-2 11-13: jobs that may run while a stroke's write is behind its release (they neither read nor change the
    # written field or the history); every other job finishes the write first
    PW_OK = ('gpu_light', 'gpu_aid', 'parts_view', 'call', 'stroke_move', 'stroke_flush', 'stroke_end', 'view')

    def _pw_ready(self):
        gs_ = getattr(self, 'gpu_sculpt', None)
        return gs_ is not None and gs_.pw_pending()

    def _pw_finish(self, why=''):
        if self._pw_ready():
            _t = time.perf_counter()
            with self.lock:
                self.gpu_sculpt.pw_complete()
            self.hist._notify()
            self.pw_waits.append((why, round((time.perf_counter() - _t) * 1000, 1)))

    def _idle_rd_ready(self):
        return self._stroke is None and not getattr(self, '_idle_rd_off', False) and self.hist.idle_pending()

    def _refine_ready(self):
        return (self._refine_at is not None and time.perf_counter() >= self._refine_at and self._stroke is None)

    def _warmup(self):
        """Compile / load every kernel once on a scratch scene, with progress (first run ~30-60 s, then ~1-3 s).
        V1.1: also the field tools' kernels (pencil, cut, splice, fill, layer smooth) and all 11 brushes, so the
        first action after launch does not stall on a compile."""
        steps = []

        def step(label, fn):
            steps.append((label, fn))

        st = {}

        def scene():
            h = History()
            o = h.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.05}}))
            h.push(AddObject({'base': {'id': 50, 'kind': 'twist', 'params': {'rate': 4.0}, 'children': [
                {'id': 51, 'kind': 'box', 'params': {'size': [0.05, 0.1, 0.05], 'round': 0.004}}]},
                'translate': (0.1, 0, 0)}))
            st.update(h=h, o=o, cam=Camera(pos=(0.0, 0.0, 0.4)))

        def frames():
            for q in (0, 1, 2):
                render(st['h'].scene, st['cam'], 32, 18, quality=q)

        def slice_pick():
            render(st['h'].scene, st['cam'], 32, 18, quality=0, mode='slice')
            pr = pick(st['h'].scene, st['cam'], 16, 9, 32, 18)
            st['p'] = list(pr.point if pr.hit else np.array([0, 0, 0.05]))

        def brushes():
            for tool in BR.TOOLS:
                st['h'].push(BrushStroke(st['o'].id, {'tool': tool, 'radius': 0.01},
                                         [{'p': st['p'], 'n': [0, 0, 1], 'v': [0.001, 0, 0]}]))

        def cut_splice():
            h, cam = st['h'], st['cam']
            eye, dirs = FT.outline_from_screen(cam, [(10, 5), (22, 5), (22, 13), (10, 13)], 32, 18)
            h.push(FT.Cut([st['o'].id], eye, dirs, 'separate', distance=0.01))
            b = h.push(AddObject({'kind': 'box', 'params': {'size': [0.02, 0.02, 0.02]},
                                  'translate': (-0.12, 0, 0)}))
            c = h.push(AddObject({'kind': 'box', 'params': {'size': [0.02, 0.02, 0.02]},
                                  'translate': (-0.2, 0, 0)}))
            h.push(FT.FillSplice(b.id, [-0.11, 0, 0], [1, 0, 0], c.id, [-0.19, 0, 0], [1, 0, 0],
                                 {'shape': 'circle', 'radius': 0.004}))
            d = h.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.01}, 'translate': (0, 0.2, 0)}))
            e = h.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.01}, 'translate': (0, 0.3, 0)}))
            h.push(FT.Splice(d.id, [0, 0.21, 0], [0, 1, 0], e.id, [0, 0.29, 0], [0, -1, 0]))

        def layers():
            for layer in ('detail', 'shape'):
                st['h'].push(FT.LayerSmooth(st['o'].id, {'layer': layer, 'radius': 0.004},
                                            [{'p': st['p'], 'n': [0, 0, 1]}]))
            st['h'].scene.evaluate(np.zeros((1, 3)))
            st['h'].scene.gradient(np.zeros((1, 3)))

        step('scratch scene', scene)
        step('the renderer (preview, refine, final)', frames)
        step('the field slice and picking', slice_pick)
        step('the 11 sculpt brushes', brushes)
        step('cut, splice, fill and splice', cut_splice)
        step('the layer smoothing (detail, shape)', layers)

        def refs():
            from ..render import refplanes
            refplanes.warmup()
        step('the reference planes', refs)

        def view_overlays():
            # fs-shell-5 11-14 (t68): the grids, contours and selection blend compiled here, behind the
            # progress bar, not on the window thread at the first frame (1.2 s) and first selection (0.5 s)
            from ..render import grids
            from . import viewport as _VP
            grids.warmup()
            _VP.warmup_composites()
        step("the view's grids and selection", view_overlays)

        def merge():
            # hand2f: Merge (the combine kernel), Mirror-duplicate, Duplicate, mask edits and world symmetry
            h = History()
            a = h.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.02}, 'h': 0.002}))
            b = h.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.02}, 'translate': (0.03, 0, 0), 'h': 0.002}))
            h.push(BrushStroke(a.id, {'tool': 'inflate', 'radius': 0.01, 'symmetry': [True, False, False],
                                      'sym_world': True}, [{'p': [0.0, 0.02, 0.0], 'n': [0, 1, 0]}]))
            h.push(MirrorDuplicate(a.id))
            h.push(DuplicateObject(b.id))
            h.push(MaskEdit(a.id, 'invert'))
            h.push(CombineObjects([a.id, b.id], 'union', 0.004))
        step('merge, mirror-duplicate and world symmetry', merge)

        def gpu():
            # phase 0: the GPU viewport (kernels compiled, the environment's SH, a tiny frame through the mirror)
            if not self.use_gpu:
                return
            try:
                from ..gpu.viewport import GpuView
                gv = GpuView()
                h = History()
                h.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.02}, 'h': 0.002}))
                gv.frame(h.scene, Camera(pos=(0, 0, 0.2), target=(0, 0, 0)), 64, 36)
                gv.key = None                    # the next sync drops the warm-up object (or rebuilds its id)
                gv.changed = True
                gv.aid.update(self.view_aid)         # the sculpting visibility aid (headlight on by default)
                self.gpu_view = gv
                self._install_gpu_sculpt(gv)
                threading.Thread(target=_warm_cpu_brushes, name='cpu-brush-warmup', daemon=True).start()
            except Exception:
                self.gpu_error = traceback.format_exc()
                self.gpu_view = None
                self.message.emit('GPU viewport not available, drawing on the CPU: %s' % self.gpu_error.splitlines()[-1])
        step('the GPU viewport', gpu)
        n = len(steps)
        for i, (label, fn) in enumerate(steps):
            self.progress.emit(i, n, label)
            try:
                fn()
            except Exception:
                if i < 4:                     # the core kernels must compile; a tool's warm-up is best effort
                    raise
                self.message.emit('warm-up of %s skipped: %s' % (label, traceback.format_exc().splitlines()[-1]))
        self.progress.emit(n, n, 'ready')

    def _install_gpu_sculpt(self, gv):
        """integration option A (2026-11-08): brush strokes on the app's live scene run on the GPU atlas
        (gpu/sculpt.py) and write their change into the sculpt layer; FIELDSTUDIO_GPU_SCULPT=0 keeps the CPU brushes."""
        if os.environ.get('FIELDSTUDIO_GPU_SCULPT', '1') == '0':
            return
        from ..gpu import hooks as GH                # 11-09: one implementation with the script API
        self.gpu_sculpt = GH.install(gv, lambda scene: self.gpu_view is not None and scene is self.hist.scene,
                                     self.message.emit)
        self.gpu_sculpt.pw_host = True            # fs-brush-2 11-13: this engine runs the writes behind the release

    # ---- scene snapshot (what the panels read) -------------------------------------------------------------
    def _emit_scene(self):
        if self.gpu_view is not None:
            self.gpu_view.note_change()
        with self.lock:
            sc = self.hist.scene
            objs = []
            for o in sc.objects:
                code, prm, lo, hi, Lb = o.compiled()
                objs.append({'id': o.id, 'name': o.name, 'base': o.base.to_dict(),
                             'transform': copy.deepcopy(o.transform), 'colour': copy.deepcopy(o.colour),
                             'material': copy.deepcopy(o.material), 'detail': copy.deepcopy(o.detail),
                             'visible': o.visible, 'bricks': len(o.sculpt.slots), 'sculpt_h': o.sculpt.h,
                             'hidden': o.id in sc.view_hidden, 'mask_inv': bool(getattr(o, 'mask_inv', False)),
                             'locked': getattr(o, 'locked', None),
                             'regions': {int(k): dict(v) for k, v in (getattr(o, 'regions', None) or {}).items()},
                             'L_base': Lb, 'L_sculpt': o.sculpt.lipschitz(sc.store),
                             'nodes': sum(1 for _ in o.base.walk())})
                if getattr(o, 'group', None) is not None:      # fs-shell-6 11-14 (fs-shell-5's t73): a group selects
                    objs[-1]['group'] = int(o.group)            # and drags as one
            snap = {'objects': objs, 'can_undo': self.hist.can_undo(), 'can_redo': self.hist.can_redo(),
                    'log': [_label(c) for c in self.hist.done], 'redo_log': [_label(c) for c in reversed(self.hist.undone)],
                    'base_log': len(self.hist.base_log), 'hist_mem': self.hist.memory_bytes(),
                    'field_mem': sc.field_memory_bytes(), 'L': sc.lipschitz_bound(), 'version': sc.version,
                    'select': self._select, 'select_node': self._select_node,
                    # hand2f: identity of the done steps (the unsaved-changes and autosave checks compare it)
                    'log_ids': [getattr(c, 'serial', -id(c)) for c in self.hist.done], 'committed': len(self.hist.committed),
                    # fs-shell 11-11: the Timeline shows one entry per undo step: each command's
                    # group size where a group starts (_push_group, Soften the joins), else 0
                    'log_groups': [int(getattr(c, 'group_first_n', 0) or 0) for c in self.hist.done],
                    'redo_groups': [int(getattr(c, 'group_first_n', 0) or 0) for c in reversed(self.hist.undone)],
                    'adopted': bool(getattr(self, '_adopted', False)),
                    # fs-brush 11-11 (a failed card view must not pass for the card): which device drew the
                    # last frame ('card' / 'cpu' / None before the first) and why the card did not
                    'render_device': getattr(self, 'render_device', None), 'card_error': getattr(self, 'card_error', None),
                    'card_expected': os.environ.get('FIELDSTUDIO_GPU', '1') != '0',
                    'scene_gen': getattr(self.hist, 'gen', 0)}
            self._adopted = False
        self._select = None
        self._select_node = None
        self.sceneChanged.emit(snap)

    # ---- jobs ------------------------------------------------------------------------------------------------
    def _push(self, cmd, label=None):
        if label:
            cmd.ui_label = label
        with self.lock:
            return self.hist.push(cmd)

    def _push_nolock(self, cmd, label=None):
        """_push for callers already holding self.lock (core.actions)"""
        if label:
            cmd.ui_label = label
        return self.hist.push(cmd)

    def _push_group(self, cmds, label):
        out = []
        with self.lock:
            for c in cmds:
                c.ui_label = label
                out.append(self.hist.push(c))
        cmds[-1].group_n = len(cmds)
        cmds[0].group_first_n = len(cmds)
        return out

    def _merge_push_group(self, key, cmds, drag=None):
        """fs-shell-6 11-14: _merge_push for several commands that undo as one (_push_group's marks): the previous
        push of the same key (the same drag, or within MERGE_WINDOW) is undone and dropped, all of its commands"""
        now = time.perf_counter()
        m = self._merge
        if drag is not None:
            key = (key, 'drag', drag)
        with self.lock:
            if (m and m[0] == key and (drag is not None or now - m[1] < MERGE_WINDOW) and self.hist.done
                    and self.hist.done[-1] is m[2]):
                for _ in range(int(getattr(m[2], 'group_n', 1) or 1)):
                    self.hist.undo()
                    self.hist.drop_undone_last()
            for c in cmds:
                c.ui_label = 'Move together'
                self.hist.push(c)
        if len(cmds) > 1:
            cmds[-1].group_n = len(cmds)
            cmds[0].group_first_n = len(cmds)
        self._merge = (key, now, cmds[-1])

    def _merge_push(self, key, cmd, drag=None):
        """Merge with the previous push of the same key. hand1b: a transform DRAG (W / R / Y) carries a drag id and
        merges for exactly that drag, press to release, however long it pauses; other edits keep the 1.5 s window."""
        now = time.perf_counter()
        m = self._merge
        if drag is not None:
            key = (key, 'drag', drag)
        with self.lock:
            if (m and m[0] == key and (drag is not None or now - m[1] < MERGE_WINDOW) and self.hist.done
                    and self.hist.done[-1] is m[2]):
                self.hist.undo()
                self.hist.drop_undone_last()
            self.hist.push(cmd)
        self._merge = (key, now, cmd)

    def _do(self, job):
        if job and job[0] not in self.PW_OK and self._pw_ready():
            st_ = job[4] if job[0] == 'stroke_begin' and len(job) > 4 and isinstance(job[4], dict) else None
            if st_ is not None and st_.get('tool') in ('grab', 'stretch') and not st_.get('snake', False):
                # fs-brush-2 11-13 (back to back, at a working pace): a Grab / Stretch pressed while the last stroke's write is behind
                # its release starts at once over it, once that stroke's fine dab has landed (the lattice holds it; its
                # CPU write-back carries on beside the new stroke's frames; gpu/sculpt.py lands it first if the new
                # stroke would write the lattice)
                _t = time.perf_counter()
                with self.lock:
                    self.gpu_sculpt.pw_until_switch()
                self.pw_waits.append(('stroke_begin:overlap', round((time.perf_counter() - _t) * 1000, 1)))
            else:
                self._pw_finish(job[0])
        try:
            return self._do_job(job)
        except KeyError as e:
            # V1.3: a job queued for an object that is gone by the time it runs (deleted, undone, a new scene)
            if e.args and isinstance(e.args[0], (int, np.integer)) and self.hist.scene.index_of(int(e.args[0])) < 0:
                raise Refused('%s skipped: object %d no longer exists.' % (job[0], int(e.args[0])))
            raise
        except ValueError as e:
            if job[0] == 'mode':               # a field tool refusing the picks (e.g. both faces on one object)
                raise Refused('%s' % e)
            raise

    # ---- fs-shell 11-14: jobs registered by the tool lanes (the window / tools contract) ------------------------------
    BUILTIN_JOBS = ('add', 'adopt_replay', 'autosave', 'boolean', 'call', 'check_overlaps', 'cmd', 'consolidate', 'deform',
                    'duplicate', 'fill_cavity', 'frame', 'fv8store', 'goto', 'gpu_aid', 'gpu_light', 'gpu_ref_store',
                    'her_writeback', 'mask', 'merge', 'merge_protected', 'mirror_dup', 'mode', 'new', 'open', 'open_her',
                    'open_region', 'oscale', 'param', 'parts_view', 'props', 'props_many', 'protect_masked', 'push', 'redo', 'remove',
                    'save', 'set_locked', 'set_root', 'show_overlaps', 'stroke_begin', 'stroke_end', 'stroke_flush',
                    'stroke_move', 'undo', 'unwrap', 'verify_replay', 'view', 'wrap', 'write_region')

    def register_job(self, name, fn):
        """fs-tools / fs-materials: a job of their own, run on this engine thread under its lock as
        fn(engine, *args) -> True when the scene changed (posted as engine.post(name, *args)). A name the engine
        already handles is refused (no silent override)."""
        jobs = self.__dict__.setdefault('_job_handlers', {})
        if name in jobs or name in getattr(self, 'BUILTIN_JOBS', ()):
            raise ValueError('job %r is already registered' % (name,))
        jobs[name] = fn

    def _do_job(self, job):
        kind = job[0]
        sc = self.hist.scene
        h_ = getattr(self, '_job_handlers', None)
        if h_ and kind in h_:
            with self.lock:
                return bool(h_[kind](self, *job[1:]))
        if kind == 'gpu_ref_store':             # r33: View > Published field beside the figure (view only)
            if self.gpu_view is None:
                self.message.emit('The published field needs the GPU viewport.')
                return False
            path, on = job[1], job[2]
            copies = job[3] if len(job) > 3 else 1
            t0 = time.perf_counter()
            with self.lock:
                n = self.gpu_view.set_reference_store(path, on, copies=copies)
            if on:
                self.message.emit('Published field %s: %d volume(s) on the card in %.1f s (view only: not saved, not '
                                  'editable)' % (os.path.basename(path or ''), n, time.perf_counter() - t0))
            return True
        if kind == 'parts_view':                # 11-09: View > Protected parts: shown / ghosted / hidden
            self.parts_view = job[1]
            self._render_due = True
            return True
        if kind == 'show_overlaps':
            self.show_overlaps = bool(job[1])
            self._render_due = True
            return True
        if kind == 'check_overlaps':
            from ..core import actions as ACT          # 11-09: one implementation with the script API
            with self.lock:
                msg, _tot = ACT.check_overlaps(self.hist)
            self.message.emit(msg)
            return False
        if kind == 'gpu_aid':                   # 11-08 night: View > sculpt headlight / cavity / exposure
            self.view_aid.update(job[1])
            if self.gpu_view is not None:
                self.gpu_view.aid.update(job[1])
            self._render_due = True
            return True
        if kind == 'gpu_light':                 # r26: View > Sculpting key light (or the HDRI's sun)
            if self.gpu_view is not None:
                self.gpu_view.light_mode = job[1]
            return True
        if kind == 'add':
            o = self._push(AddObject(job[1]))
            self._select = [o.id]
            return True
        if kind == 'param':
            _, oid, nid, params = job
            self._merge_push(('param', oid, nid, tuple(sorted(params))), SetNodeParams(oid, nid, params))
            return True
        if kind == 'props_many':
            # fs-shell-6 11-14 (the eye workflow: select both, then drag them back together): one drag of a selection
            # sets every member's transform; merged per drag id across the members, press to release = one undo step
            items, drag = job[1], (job[2] if len(job) > 2 else None)
            cmds = [SetObjectProps(int(o), p) for o, p in items]
            self._merge_push_group(('props_many', tuple(sorted(int(o) for o, _p in items))), cmds, drag)
            return True
        if kind == 'props':
            oid, props = job[1], job[2]
            drag = job[3] if len(job) > 3 else None
            key = ('props', oid, tuple(sorted((k, tuple(sorted(v)) if isinstance(v, dict) else None)
                                              for k, v in props.items())))
            self._merge_push(key, SetObjectProps(oid, props), drag)
            return True
        if kind == 'oscale':
            # hand1: non-uniform object scale (Scale tool, one axis): a root 'transform' node named 'object scale'
            # carries [sx, sy, sz] on the base tree (the sculpt layer is not stretched). A drag merges into one step.
            oid, svec = job[1], job[2]
            drag = job[3] if len(job) > 3 else None
            shear = job[4] if len(job) > 4 else None     # 11-10: a skew (Scale handles, Ctrl+drag; core/ops 'shear')
            key = ('oscale', oid) if drag is None else ('oscale', oid, 'drag', drag)
            now = time.perf_counter()
            m = self._merge
            with self.lock:
                if (m and m[0] == key and (drag is not None or now - m[1] < MERGE_WINDOW) and self.hist.done
                        and self.hist.done[-1] is m[2]):
                    self.hist.undo()
                    self.hist.drop_undone_last()
                o = sc.get(oid)
                tree = o.base.to_dict()
                nxt = None
                if tree['kind'] == 'transform' and tree.get('name') == 'object scale':
                    tree['params']['scale'] = [float(v) for v in svec]
                else:
                    nid = sc.next_id
                    tree = {'id': nid, 'kind': 'transform', 'name': 'object scale',
                            'params': {'translate': [0.0, 0.0, 0.0], 'rotate': [0.0, 0.0, 0.0],
                                       'scale': [float(v) for v in svec]}, 'children': [tree]}
                    nxt = nid + 1
                if shear is not None:
                    tree['params']['shear'] = [round(float(v), 5) for v in shear]
                cmd = SetBase(oid, tree, next_id=nxt) if nxt else SetBase(oid, tree)
                cmd.ui_label = ('Skew %s' % o.name) if shear is not None else 'Scale %s %s' % (o.name, 'x'.join('%.2f' % v for v in svec))
                self.hist.push(cmd)
            self._merge = (key, now, cmd)
            if o.sculpt.slots:
                self.message.emit('Axis scale stretches the base form only; the sculpt layer keeps its shape.')
            return True
        if kind == 'deform':
            # fs-shell 11-11 (design 4.5, deformers on the gizmo): one node per kind named 'gizmo <kind>' at the root of
            # the base tree (under the axis scale's 'object scale' node when there is one), added by the first drag and
            # dialled by later ones; a drag merges into one undo step by its id, as 'oscale' does
            oid, mkind, params = job[1], job[2], dict(job[3])
            drag = job[4] if len(job) > 4 else None
            key = ('deform', oid, mkind) if drag is None else ('deform', oid, mkind, 'drag', drag)
            now = time.perf_counter()
            m = self._merge
            with self.lock:
                if (m and m[0] == key and (drag is not None or now - m[1] < MERGE_WINDOW) and self.hist.done
                        and self.hist.done[-1] is m[2]):
                    self.hist.undo()
                    self.hist.drop_undone_last()
                o = sc.get(oid)
                tree = o.base.to_dict()
                name = 'gizmo ' + mkind
                node, nxt = tree, None
                while node is not None and node.get('name') != name and \
                        (node.get('name') == 'object scale' or str(node.get('name', '')).startswith('gizmo ')):
                    node = node['children'][0] if node.get('children') else None
                if node is not None and node.get('name') == name:
                    node['params'].update(params)
                else:
                    nid = sc.next_id
                    full = dict(ops.MUTATIONS.get(mkind, {}))
                    full.update(params)
                    holder = tree if (tree['kind'] == 'transform' and tree.get('name') == 'object scale') else None
                    child = holder['children'][0] if holder is not None else tree
                    new = {'id': nid, 'kind': mkind, 'name': name, 'params': full, 'children': [child]}
                    if holder is not None:
                        holder['children'][0] = new
                    else:
                        tree = new
                    nxt = nid + 1
                cmd = SetBase(oid, tree, next_id=nxt) if nxt else SetBase(oid, tree)
                cmd.ui_label = '%s %s %s' % (mkind.capitalize(), o.name,
                                             ', '.join('%s %.4g' % kv for kv in sorted(params.items())))
                self.hist.push(cmd)
            self._merge = (key, now, cmd)
            return True
        if kind == 'push':
            out = self._push(job[1])
            if isinstance(out, list) and out:
                self._select = [int(v) for v in out]
            elif isinstance(out, dict):                # fs-shell 11-11: a mass added: its object and its node selected
                if out.get('select'):
                    self._select = [int(v) for v in out['select']]
                if out.get('node') is not None:
                    self._select_node = int(out['node'])
            return True
        if kind == 'wrap':
            _, oid, nid, mkind, params = job
            with self.lock:
                o = sc.get(oid)
                tree = o.base.to_dict()
                new_id = sc.next_id
                parent, i, node = _find_parent(tree, nid if nid is not None else tree['id'])
                if node is None:
                    raise KeyError('node %s' % nid)
                wrapped = {'id': new_id, 'kind': mkind, 'name': mkind, 'params': copy.deepcopy(params or {}),
                           'children': [node]}
                if parent is None:
                    tree = wrapped
                else:
                    parent['children'][i] = wrapped
            self._push(SetBase(oid, tree, next_id=new_id + 1), 'Add %s' % mkind)
            self._select = [oid]
            self._select_node = new_id
            return True
        if kind == 'unwrap':
            _, oid, nid = job
            with self.lock:
                tree = sc.get(oid).base.to_dict()
            parent, i, node = _find_parent(tree, nid)
            if node is None:
                return False
            if node['kind'] in ops.PRIMITIVES:
                if parent is None or parent['kind'] not in ops.BOOLEANS:
                    self.message.emit('A primitive at the root (or under a mutation) is the shape itself; '
                                      'delete the object instead.')
                    return False
                # remove this operand: the boolean becomes its other child
                gp, gi, _ = _find_parent(tree, parent['id'])
                repl = parent['children'][1 - i]
                node, i, parent = parent, gi, gp
            else:
                repl = node['children'][0]
            if parent is None:
                tree = repl
            else:
                parent['children'][i] = repl
            self._push(SetBase(oid, tree), 'Remove %s' % node['kind'])
            self._select = [oid]
            self._select_node = repl['id']
            return True
        if kind == 'boolean':
            # hand2f: Combine runs CombineObjects: B's sculpt is carried (before: dropped), B is removed, A keeps its id
            _, a, b, bkind, blend = job
            return self._do_merge([a, b], bkind, float(blend), 'first')
        if kind == 'merge':
            _, oids, op, k = job[:4]
            frame = job[4] if len(job) > 4 else 'first'
            return self._do_merge(list(oids), op, float(k), frame)
        if kind == 'mirror_dup':
            with self.lock:
                o = sc.get(job[1])
                m = self.hist.push(MirrorDuplicate(o.id))
            self._select = [m.id]
            self.message.emit('Mirror-duplicate of %s across x = 0 (exact, sculpt included): %s.' % (o.name, m.name))
            return True
        if kind == 'mask':
            with self.lock:
                o = sc.get(job[1])
                self.hist.push(MaskEdit(o.id, job[2]))
                inv = o.mask_inv
            if job[2] == 'clear':
                self.message.emit('%s: mask cleared.' % o.name)
            else:
                self.message.emit('%s: mask inverted%s.' % (o.name, '' if inv else ' back'))
            return True
        if kind == 'view':
            _, mode, oids = job
            with self.lock:
                ids = [o.id for o in sc.objects]
                hid = sc.view_hidden
                if mode == 'hide':
                    hid.update(o for o in oids if o in ids)
                elif mode == 'toggle':
                    for o in oids:
                        if o in hid:
                            hid.discard(o)
                        elif o in ids:
                            hid.add(o)
                elif mode == 'isolate':
                    others = set(ids) - set(oids)
                    if others and others <= hid and not (set(oids) & hid):
                        hid.clear()                      # a second Isolate shows everything again
                    else:
                        hid.clear()
                        hid.update(others)
                elif mode == 'show_all':
                    hid.clear()
                sc.touch(structural=True)
            return True
        if kind == 'autosave':
            return self._autosave(job[1], job[2] if len(job) > 2 else None)
        if kind == 'remove':
            oids = [o for o in job[1] if self.hist.scene.index_of(o) >= 0]
            if not oids:
                return False
            self._push_group([RemoveObject(o) for o in oids], 'Delete %d object(s)' % len(oids))
            self._select = []
            return True
        if kind == 'duplicate':
            with self.lock:
                o = sc.get(job[1])
                n = self.hist.push(DuplicateObject(o.id))          # hand2f: the sculpt bricks copied exactly
            self._select = [n.id]
            return True
        if kind == 'mode':
            # a field tool's mouse event (tools.PencilMode / CutMode / PickPairMode / LayerSmoothMode), run HERE so
            # its picks and pushes keep the threading rule; the hooks read self._mode_view for camera and size
            _, mode, method, view, x, y = job
            self._mode_view = view
            with self.lock:
                v0, n0 = self.hist.scene.version, len(self.hist.done)
                getattr(mode, method)(x, y)
                n1 = len(self.hist.done)
                changed = self.hist.scene.version != v0 or n1 != n0
            if n1 > n0 and method == 'on_release':
                self.message.emit('%s: done (Ctrl+Z undoes it).' % _label(self.hist.done[-1]))
            return changed
        if kind == 'cmd':
            self._push(job[1], job[2] if len(job) > 2 else None)
            return True
        if kind == 'undo':
            return self._undo()
        if kind == 'redo':
            return self._redo()
        if kind == 'goto':
            n = int(job[1])
            ch = False
            while len(self.hist.done) > n and self.hist.can_undo():
                with self.lock:
                    self.hist.undo()
                ch = True
            while len(self.hist.done) < n and self.hist.can_redo():
                with self.lock:
                    self.hist.redo()
                ch = True
            self._merge = None
            return ch
        if kind == 'stroke_begin':
            return self._stroke_begin(*job[1:])
        if kind == 'adopt_replay':
            return self._adopt_replay(*job[1:])
        if kind == 'stroke_flush':
            return self._stroke_moves([])           # hand2b: the backlog of the last batch
        if kind == 'stroke_end':
            return self._stroke_end()
        if kind == 'open_her':
            # milestone: her published store as an editable object (a copy: write-back makes new versions only)
            from ..core import fv8base, ops as OPS
            _, src, hh, eyes = job
            t0 = time.perf_counter()
            with self.lock:
                self.hist = UIHistory()
            self._merge = None
            self._select = []
            spec = {'name': 'field (editable copy)', 'base': OPS.Node('fv8store', fv8base.params_for(src)).to_dict(),
                    'translate': fv8base.TRANSFORM['translate'], 'rotate': fv8base.TRANSFORM['rotate'], 'h': float(hh)}
            from ..core import fv8parts as FVP            # fs-contact 11-11: a parts version opens as one object per part
            pspecs = []
            if FVP.has_parts(src):
                her_base, pspecs, her_part, _pinfo = FVP.open_specs(src, float(hh))
                spec['base'] = her_base; spec['part'] = her_part
            self._push(AddObject(spec), 'Open her field (editable copy)')
            for ps in pspecs:
                self._push(AddObject(ps), 'Add %s' % ps['name'])
            n_eyes = 0
            if eyes:
                for es in fv8base.eye_specs(src):
                    eo = self._push(AddObject(es), 'Add %s' % es['name'])
                    # 11-09: protected parts stay VISIBLE (View > Protected parts ghosts or hides them as a view)
                    n_eyes += 1
            self._warm_store_objects()
            self.message.emit('Her field %s opened as an editable copy (%.2f mm lattice; %d protected eye objects) in '
                              '%.1f s. Edits are written back only as a NEW version (File > Write her edits...).'
                              % (os.path.basename(src), hh * 1e3, n_eyes, time.perf_counter() - t0))
            return True
        if kind == 'fill_cavity':
            # fs-brush 11-11 (design D4): the explicit 'Fill it solid' (fs-shell's ui/topo_ui.py prompt, the hidden-voids
            # view): the enclosed cavity filled on the object's own lattice (tools/cavity.py), one undo step
            _, oid, cav = job
            try:
                cmd = FT.FillCavity(oid, cav)
                self._push(cmd, 'Fill a cavity')
                self.message.emit('Filled a cavity solid (%d lattice nodes). Ctrl+Z undoes it.' % int(cmd.filled))
            except Exception as e:
                self.message.emit('Fill a cavity FAILED: %s: %s' % (type(e).__name__, e))
            return True
        if kind == 'her_writeback':
            from ..core import fv8writeback as WB
            _, oid, out_dir = job
            with self.lock:
                her_o = self.hist.scene.get(oid)
                prep = None if getattr(her_o, 'part', None) else WB.prepare(self.hist.scene, oid)
                if prep is None:                           # fs-contact 11-11: her parts version (core/fv8parts.py)
                    from ..core import fv8parts as FVP
                    pprep = FVP.prepare_writeback(self.hist.scene, oid)

            def work():
                try:
                    if prep is None:                       # fs-contact 11-11: her parts version: rebuilt from every part object
                        rep = FVP.write_parts(pprep, out_dir)
                        self.message.emit('Her parts written to a NEW version %s (%d parts; union, entries, leaves rebuilt from them). '
                                          'Not published.' % (out_dir, len(rep.get('objects', {}))))
                        return
                    rep = WB.write(prep, out_dir, check=True)
                    g = rep.get('guard') or {}
                    self.message.emit('Her edits written to a NEW version %s (%d tiles; apply %.0f s, commit %.0f s; '
                                      'face guard: %s). Not published.' % (out_dir, len(rep['clusters']), rep['apply_s'],
                                                                         rep['commit_s'], 'same' if g.get('ok') else
                                                                         'CHANGED (expected for face edits)'))
                except Exception as e:
                    self.message.emit('Write-back FAILED: %s: %s' % (type(e).__name__, e))
            threading.Thread(target=work, name='her-writeback', daemon=True).start()
            self.message.emit('Writing her edits to %s in the background (sculpting can go on) ...' % out_dir)
            return False
        if kind == 'set_root':
            # 11-09 attached parts: the root painted on the (unprotected) part with the Mask brush becomes its root
            from ..core import actions as ACT
            _, pid = job
            with self.lock:
                msg, ok = ACT.set_root(self.hist, pid, self._push_nolock)
            self.message.emit(msg)
            return ok
        if kind == 'set_locked':
            # 11-08 night: Unprotect (editable, shown) / Protect again (locked; brushes pick through it)
            from ..core import actions as ACT
            _, oid, lock = job
            with self.lock:
                msg, unprot = ACT.set_locked(self.hist, oid, lock, self._push_nolock)
            if unprot:
                self._select = [int(oid)]            # brushes now work on the part (see _brush_exclude)
            if lock:
                self.post('check_overlaps')
            self.message.emit(msg)
            return True
        if kind == 'protect_masked':
            # milestone (design choice A): the surface painted with the Mask brush becomes a protected part
            from ..core import actions as ACT
            _, oid, name = job
            try:
                with self.lock:
                    msg, _changed = ACT.protect_masked(self.hist, oid, name)
                self.message.emit(msg)
            except Exception as e:
                self.message.emit('Protect the masked part refused: %s' % e)
            return True
        if kind == 'merge_protected':
            from ..tools.protected_merge import MergeProtected
            _, target, shapes, protected, use_mask, k, gap = job
            t0 = time.perf_counter()
            try:
                with self.lock:
                    cmd = MergeProtected(target, shapes, protected, use_mask, k, gap)
                    cmd.ui_label = 'Merge with protected parts'
                    info = self.hist.push(cmd)
                self._select = [target]
                self.message.emit('Merged %d shape(s) into the target, %d protected part(s)%s, k %.1f mm, gap %.2f mm: '
                                  '%d bricks changed (%.1f s).' % (len(shapes), len(protected), ' + the painted mask' if use_mask
                                                                   else '', k * 1e3, gap * 1e3,
                                                                   (info or {}).get('bricks', 0), time.perf_counter() - t0))
            except Exception as e:
                self.message.emit('Merge with protected parts refused: %s' % e)
            return True
        if kind == 'new':
            with self.lock:
                self.hist = UIHistory()
            self._merge = None
            self._select = []
            self.message.emit('New scene.')
            return True
        if kind == 'open':
            return self._open(job[1])
        if kind == 'save':
            # r48: the state is snapshotted under the lock (the sculpt chunks copied, ~0.1 s per 100 MB); the write
            # (only changed chunks compressed) runs on a worker, so sculpting goes on (before: 40-72 s held here)
            path = job[1] if len(job) > 1 else None
            t0 = time.perf_counter()
            try:
                _, path, extra = job[:3]
                ref_copies = list(job[3]) if len(job) > 3 else []      # r49: copied by the worker, into its own folder
                self.join_save()
                t0 = time.perf_counter()
                with self.lock:
                    doc, chunks_ = snapshot_project(self.hist, extra)
            except Exception as e:
                # fs-store 11-12: a snapshot that fails is a failed save (before, only 'save failed' in the log while
                # the window already counted the work as saved)
                self.last_save = (path, None, 0.0, time.perf_counter() - t0, {'error': repr(e)})
                self.message.emit('SAVE FAILED for %s: %s: %s' % (path, type(e).__name__, e))
                self.saveDone.emit({'path': path, 'ok': False, 'hash': None, 'error': '%s: %s' % (type(e).__name__, e),
                                    'seconds': time.perf_counter() - t0})
                self._save_closed()
                return False
            held = time.perf_counter() - t0
            n = len(doc['log'])

            def work():
                t1 = time.perf_counter()
                res = {'path': path, 'ok': False, 'hash': None, 'error': None}
                try:
                    hsh = write_snapshot(doc, chunks_, path, refs=ref_copies)
                    st = write_snapshot.last_stats or {}
                    self.last_save = (path, hsh, time.perf_counter() - t1, held, st)
                    res.update(ok=True, hash=hsh)
                    self.message.emit('Saved %s (state hash %s..., %d log entries; %d of %d sculpt chunks written, '
                                      '%.1f s in the background, %.2f s held).' % (
                                          path, hsh[:12], n, st.get('written', 0), st.get('chunks', 0),
                                          time.perf_counter() - t1, held))
                except BaseException as e:
                    self.last_save = (path, None, time.perf_counter() - t1, held, {'error': repr(e)})
                    res['error'] = '%s: %s' % (type(e).__name__, e)
                    self.message.emit('SAVE FAILED for %s: %s: %s' % (path, type(e).__name__, e))
                finally:
                    res['seconds'] = time.perf_counter() - t1
                    try:
                        self.saveDone.emit(res)
                    except RuntimeError:                 # the window is gone: the file is what counts
                        pass
                    self._save_closed()
            # fs-store 11-12: NOT a daemon thread. Quit -> Save posted this job and closed the window ~0.1 s later; the
            # interpreter's exit killed the daemon writer mid-write (an 18:24 save: a .tmp with no project.json, no
            # message). The window and Engine.stop now wait for it; a non-daemon thread also holds the exit until the
            # file is complete, whatever path the process takes to its end.
            try:
                self._save_thread = threading.Thread(target=work, name='save', daemon=False)
                self._save_thread.start()
            except BaseException:
                self._save_closed()
                raise
            return False
        if kind == 'frame':
            with self.lock:
                oids = job[1] or [o.id for o in sc.objects if o.visible]
                lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
                for oid in oids:
                    if sc.index_of(oid) < 0:
                        continue
                    _, w0, w1 = sc._object_row(sc.get(oid))
                    lo, hi = np.minimum(lo, w0), np.maximum(hi, w1)
            if np.all(np.isfinite(lo)):
                self.boundsReady.emit((lo, hi))
            return False
        if kind == 'consolidate':
            self.join_save()                                 # r48
            # r44: fold the old op-log entries into one checkpoint (replays run without the lock; adopted only if
            # nothing changed meanwhile)
            from ..core.history import consolidate_prepare, consolidate_commit
            keep = int(job[1]) if len(job) > 1 else 50
            t0 = time.perf_counter()
            with self.lock:
                self.hist.end_stroke()
                snap = (self.hist.log(), len(self.hist.done), self.hist.scene.state_hash())
            try:
                plan = consolidate_prepare(self.hist, keep, snap)
            except Exception as e:
                self.message.emit('History not consolidated (%s: %s); nothing changed.' % (type(e).__name__, e))
                return False
            if plan is None:
                self.message.emit('Nothing to consolidate: the log has %d entries, %d kept for undo.'
                                  % (len(snap[0]), min(keep, snap[1])))
                return False
            with self.lock:
                ok = consolidate_commit(self.hist, plan)
            self.message.emit(
                ('History consolidated: %d entries folded into one checkpoint, %d undo steps kept; the new log '
                 'replays to the same state hash %s... (%.1f s).' % (plan['n_fold'], snap[1] - plan['n_fold_done'],
                                                                     plan['hash'][:12], time.perf_counter() - t0))
                if ok else 'History not consolidated: the model changed while it ran; nothing changed.')
            return False
        if kind == 'verify_replay':
            t0 = time.perf_counter()
            with self.lock:
                log = self.hist.log()
                want = self.hist.scene.state_hash()
            got = replay(log).scene.state_hash()
            ok = got == want
            self.message.emit('Replay of %d log entries: %s (%.1f s) - hash %s...' % (
                len(log), 'SAME state hash' if ok else 'DIFFERENT hash', time.perf_counter() - t0, want[:12]))
            return False
        if kind == 'open_region':
            return self._open_region(job[1], job[2] if len(job) > 2 else None)
        if kind == 'write_region':
            return self._write_region(job[1], job[2], job[3] if len(job) > 3 else '')
        if kind == 'call':
            _, fn, token = job
            try:
                with self.lock:
                    res = fn(self)
            except Exception as e:
                # fs-brush 11-12: a failed call still ends with callDone, carrying a
                # CallFailed result, so a panel that chains calls (her natural base's comb lattice) stops instead of waiting
                # forever; the failure is then reported as before (the per-job handler)
                self.callDone.emit((token, CallFailed(e, traceback.format_exc())))
                raise
            self.callDone.emit((token, res))
            return False
        raise ValueError('unknown job %r' % kind)

    # ---- hand2f: merge, autosave ------------------------------------------------------------------------------------
    def _do_merge(self, oids, op, k, frame):
        oids = [o for o in oids if self.hist.scene.index_of(o) >= 0]
        if len(oids) < 2:
            raise Refused('Merge needs two or more objects selected (Shift+click adds).')
        t0 = time.perf_counter()
        with self.lock:
            names = [self.hist.scene.get(o).name for o in oids]
            cmd = CombineObjects(oids, op, k, frame)
            try:
                m = self.hist.push(cmd)
            except ValueError as e:              # hand2b: the size guard (refused before anything is allocated)
                raise Refused(str(e))
        rep = cmd.report
        verb = {'union': 'Merged', 'subtract': 'Subtracted', 'intersect': 'Intersected'}[op]
        msg = '%s %s (%s, %.1f s): sculpt %d bricks carried' % (
            verb, ', '.join(names), 'smooth %.0f mm' % (k * 1000) if k > 0 else 'sharp', time.perf_counter() - t0,
            rep.get('bricks', 0))
        if 'max_err_near_m' in rep:
            msg += ', max field error %.1f um near the surface' % (rep['max_err_near_m'] * 1e6)
        if rep.get('apart'):
            msg += '. Not touching the others: %s (one object in separate pieces)' % ', '.join(rep['apart'])
        if rep.get('resampled'):
            msg += '. Resampled the %s sculpt onto the %.2g mm lattice' % (
                ', '.join(rep['resampled']), m.sculpt.h * float(m.transform['scale']) * 1e3)
        self.message.emit(msg + '. Ctrl+Z undoes it.')
        self._select = [m.id]
        self.last_merge = rep
        return True

    def _autosave(self, paths, extra):
        if self.autosave_busy or not paths:
            return False
        t0 = time.perf_counter()
        with self.lock:
            doc, arrays = snapshot_project(self.hist, extra)
        self.autosave_busy = True

        def work():
            try:
                for p in paths:
                    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
                    hsh = write_snapshot(doc, arrays, p)          # r48b: the hash is computed by the writer
                self.last_autosave = (list(paths), hsh, time.perf_counter() - t0)
                self.message.emit('Autosaved (%d log entries, %.1f s): %s' % (
                    len(doc['log']), time.perf_counter() - t0, paths[0]))
            except Exception as e:
                self.message.emit('Autosave failed: %s: %s' % (type(e).__name__, e))
            finally:
                self.autosave_busy = False
        # fs-store 11-12: kept, so Engine.stop lets a running autosave end (a daemon cut mid-write leaves a .tmp)
        self._autosave_thread = threading.Thread(target=work, name='autosave', daemon=True)
        self._autosave_thread.start()
        return False

    # ---- V1.2 field regions ---------------------------------------------------------------------------------------
    def _open_region(self, params, name=None):
        """a crop of a brick-field version as a new object (base = the sampled region node, neutral colour)"""
        from ..core import region
        t0 = time.perf_counter()
        self.message.emit('Loading the field region from %s ...' % params['source'])
        R = region.load(params)                       # outside the lock: the cache makes the compile instant
        sc = self.hist.scene
        with self.lock:
            nid = sc.next_id + 1
            spec = {'base': {'id': nid, 'kind': 'sampled', 'name': 'field region', 'params': params, 'children': []},
                    'name': name or 'region %s' % os.path.basename(params['source']),
                    'translate': list(params['anchor']),
                    'colour': {'albedo': [0.6, 0.6, 0.6], 'variation': 0.0},
                    'material': {'roughness': 0.5, 'specular': 0.3}}
            o = self._push(AddObject(spec), 'Open field region')
            _, w0, w1 = sc._object_row(o)
        self._select = [o.id]
        self.boundsReady.emit((w0, w1))
        rp = R['report']
        self.message.emit('Field region: level L%d, %s samples, L %.2f, loaded in %.1f s.' % (
            rp['level'], 'x'.join(str(v) for v in rp['dims']), rp['L_cell_max'], time.perf_counter() - t0))
        return True

    def _write_region(self, oid, out_dir, note=''):
        """the region's sculpt delta -> a NEW store version + the face shape guard check (a worker thread; never
        publishes). The result arrives as regionDone."""
        from ..core import region
        with self.lock:
            prep = region.prepare_write_back(self.hist.scene, oid)
        self.message.emit('Writing the region back to a new version: %s ...' % out_dir)

        def work():
            try:
                rep = region.write_back_prepared(prep, out_dir, note)
            except Exception as e:
                rep = {'error': '%s: %s' % (type(e).__name__, e), 'trace': traceback.format_exc(limit=4)}
            self.regionDone.emit(rep)
        threading.Thread(target=work, name='region-write-back', daemon=True).start()
        return False

    def _undo(self):
        if not self.hist.can_undo():
            return False
        n = getattr(self.hist.done[-1], 'group_n', 1)
        with self.lock:
            for _ in range(n):
                self.hist.undo()
        self._merge = None
        return True

    def _redo(self):
        if not self.hist.can_redo():
            return False
        n = getattr(self.hist.undone[-1], 'group_first_n', 1)
        with self.lock:
            for _ in range(n):
                self.hist.redo()
        self._merge = None
        return True

    def _warm_store_objects(self):
        """milestone: her store's data to the card and the double-precision evaluator compiled now, at Open, so the first
        stroke does not pay it (acceptance A2: the first Clay stroke ran at 2 fps)"""
        gv = self.gpu_view
        if gv is None:
            return
        try:
            with self.lock:
                for o in self.hist.scene.objects:
                    if any(n.kind == 'fv8store' for n in o.base.walk()):
                        gv.mirror.geval._program(o)
        except Exception as e:
            self.message.emit('Card warm-up for her store skipped: %s' % e)

    def join_save(self, timeout=None):
        """r48: wait for a Save being written in the background (before Open, another Save, Consolidate)"""
        t = getattr(self, '_save_thread', None)
        if t is not None and t.is_alive():
            t.join(timeout)
        return not (t is not None and t.is_alive())

    def _open(self, path):
        """hand2b (verifier item 6): the saved state is loaded and shown at once (load_project: the arrays and a hash
        check, ~2 s for a 60 MB project); the undo history is rebuilt by replaying the op log on a WORKER thread into
        its own scene (nothing shared with the live one), then adopted here, on the engine thread, if the replay
        reproduces the saved hash and nothing was edited meanwhile. Before, the replay ran first on this thread
        (76.6 s for a 39-step full-body project; minutes for a real session) and blocked every job."""
        self.join_save()                                     # r48
        t0 = time.perf_counter()
        with self.lock:
            loaded = load_project(path)                      # hash-checked
            want = loaded.scene.state_hash()
            log = loaded.loaded_log
            h = UIHistory(loaded.scene)
            h.base_log = list(log)
            jdone = list(getattr(loaded, 'loaded_done', None) or [])   # fs-trust 11-11: the undo journal
            if jdone:
                h.base_log = list(loaded.loaded_base)
                h.done = jdone                       # rebuilt from saved records: no replay
                log = []
            self.hist = h
        self._merge = None
        self._select = []
        self._warm_store_objects()
        import json
        with open(os.path.join(path, 'project.json'), encoding='utf-8') as f:
            extra = json.load(f).get('extra', {})
        self.projectLoaded.emit(extra)
        self.message.emit('Opened %s: hash %s... verified (%.1f s).%s' % (
            path, want[:12], time.perf_counter() - t0,
            ' Rebuilding the undo history of %d steps in the background; sculpting can start now.' % len(log)
            if log else ''))
        if log:
            token = h

            def work():
                t1 = time.perf_counter()
                try:
                    r = replay(log)
                    ok = r.scene.state_hash() == want
                except Exception as e:
                    r, ok = None, False
                    self.message.emit('Undo history not rebuilt (%s: %s); the history starts at the opened state.'
                                      % (type(e).__name__, e))
                finally:
                    self.replaying = False
                self.post('adopt_replay', token, r if ok else None, time.perf_counter() - t1, len(log))
            self.replaying = True
            threading.Thread(target=work, name='open-replay', daemon=True).start()
        else:
            self._upgrade_eyes()
        return True

    def _upgrade_eyes(self):
        """fs-store-3 11-13 (decided: one eye per side, the eye engine's, used everywhere): a project saved with the
        old stand-in eye objects (the 12.17 mm eyeball sphere U the cornea sphere, hard) gets the drawn eye's surface
        (core/fv8base.eye_base: the coat sphere U the cornea sphere, smooth 0.15 mm, as the real renderer draws it): one
        undoable 'Edit base' step per eye, after the open's history is settled (so the replay's hash check is untouched)"""
        from ..core import fv8base
        from ..core.history import SetBase
        try:
            ups = fv8base.eye_upgrades(self.hist.scene)
        except Exception as e:
            self.message.emit('Eye objects not checked (%s: %s).' % (type(e).__name__, e))
            return 0
        for oid, tree, side in ups:
            self._push(SetBase(oid, tree), "Eye %s: the drawn eye's surface" % side)
        if ups:
            self.message.emit("Eye objects updated to the drawn eye's surface (coat + cornea, as the renderer draws them): "
                              '%s. Undo restores the old ones; saving keeps the new.' % ', '.join(s for _o, _t, s in ups))
        return len(ups)

    def _adopt_replay(self, token, r, secs, n):
        """the open's background replay finished (engine thread): take its done / undone stacks when the live history
        is still the one opened and untouched"""
        if r is None:
            if token is self.hist:
                self.message.emit('Undo history: the replay did not reproduce the saved state; the history starts at '
                                  'the opened state.')
                self._upgrade_eyes()                          # fs-store-3 11-13: after the history is settled
            return False
        if token is not self.hist:
            return False                                      # another scene was opened or created meanwhile
        if self.hist.done or self.hist.undone or self._stroke is not None:
            self.message.emit('Undo history of the opened project not attached: edits were made while it was '
                              'rebuilt (%.0f s). Undo works for the steps since opening.' % secs)
            if self._stroke is None:
                self._upgrade_eyes()
            return False
        with self.lock:
            h2 = UIHistory(r.scene)
            h2.done, h2.undone, h2.committed = r.done, r.undone, r.committed
            h2.scene.view_hidden = set(self.hist.scene.view_hidden)
            h2.gen = getattr(self.hist, 'gen', 0)   # the undo-attach swaps in an EQUAL scene (same objects, ids): same gen (fs-brush)
            self.hist = h2
            # fs-tools-2 11-14: the rebuilt steps join the undo journal (the replay keeps its undo data in RAM only, with no
            # uid), so the next save carries the whole history and the next Open needs no replay. Before, a project once
            # saved without a journal replayed at every Open for good (fs-tools u3: the milestone re-saved with 12 x null)
            if getattr(h2, 'journal', False):
                from ..core import undojournal as _UJ
                for c_ in h2.done:
                    if c_.uid is None:
                        c_.uid = _UJ.new_uid()
                for c_ in h2.done:
                    h2._journal(c_)
        self._merge = None
        self._adopted = True                 # the window keeps its clean / dirty state across the swap
        self.message.emit('Undo history rebuilt in the background: %d steps (%.0f s).' % (n, secs))
        self._upgrade_eyes()                 # fs-store-3 11-13: the old stand-in eyes, after the history is settled
        return True

    # ---- picking and brush strokes ---------------------------------------------------------------------------
    def _pick(self, view, x, y):
        _tp = time.perf_counter()
        try:
            return self._pick_(view, x, y)
        finally:
            self._pick_ms = getattr(self, '_pick_ms', 0.0) + (time.perf_counter() - _tp) * 1000

    def _brush_exclude(self, view):
        """With a brush in hand, the object ids brush picks pass through.
        fs-brush 11-12: a locked protected part STOPS the pick, so
        brushing it does nothing (the press says it is protected; drag dabs that land on it are skipped). Until now
        (11-08 night) picks passed through it to the surface beneath, an edit the user could not see: on her, her body's copy
        of each eye under the protected eye. Edit > Unprotect still makes a part sculptable."""
        from ..core import brush as _BR
        if view.get('tool') not in _BR.TOOLS:
            return ()
        sc = self.hist.scene
        sel = set(int(i) for i in (view.get('selected') or []))
        if sel and any(getattr(o, 'locked', None) is False and o.id in sel for o in sc.objects):
            # 11-09: an unprotected part is a copy of her surface where it sits: with it selected, brushes work on the
            # selection only (a pick could otherwise land on her body, the same surface)
            return tuple(o.id for o in sc.objects if o.id not in sel)
        return ()

    def _pick_(self, view, x, y):
        with self.lock:
            if self.gpu_view is not None:
                # phase 0 (r12): pick on the card's field, the one the viewport draws (one kernel launch; the CPU
                # march slowed with every stroke as the field steepened)
                try:
                    from ..render.raymarch import PickResult
                    hit, p, n, oid, t = self.gpu_view.pick(self.hist.scene, view['cam'], x, y, view['w'], view['h'],
                                                           self._brush_exclude(view))
                    if hit and oid >= 0:
                        return PickResult(True, p, n, oid, t)
                    return PickResult(False, None, None, -1, float('inf'))
                except Exception:
                    self.gpu_error = traceback.format_exc()
            return pick(self.hist.scene, view['cam'], x, y, view['w'], view['h'])

    def _do_hover(self, view, x, y):
        pr = self._pick(view, x, y)
        self.hoverReady.emit({'hit': pr.hit, 'point': pr.point, 'normal': pr.normal, 'oid': pr.object_id,
                              'x': x, 'y': y})

    def _near_miss(self, view, x, y, radius):
        """fs-brush 11-11 (design C5): a press whose ray misses every form: the closest approach of the ray to an
        editable form's zero set (the minimum of |F| along the ray, CPU field, 0.5 mm steps over the forms' box). Within
        the brush's radius: the surface point there (two Newton steps) and its normal, as a pick result; else None."""
        from ..render.raymarch import PickResult
        from ..core.scene import is_protected
        from ..tools.protected_merge import field_of
        cam = view['cam']
        o, d = cam.ray(x, y, view['w'], view['h'])
        best = None
        with self.lock:
            sc = self.hist.scene
            pk = sc.pack()
            excl = set(self._brush_exclude(view))
            for ob in sc.objects:
                if is_protected(ob) or ob.id in excl or not getattr(ob, 'visible', True) or ob.id in sc.view_hidden:
                    continue
                from ..tools.protected_merge import world_box
                lo, hi = world_box(ob)
                lo = np.asarray(lo, float) - radius; hi = np.asarray(hi, float) + radius
                inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
                t0 = max(float(np.max(np.minimum((lo - o) * inv, (hi - o) * inv))), 0.0)
                t1 = float(np.min(np.maximum((lo - o) * inv, (hi - o) * inv)))
                if t1 <= t0:
                    continue
                tt = np.arange(t0, t1, 0.0005)
                F = field_of(sc, pk, ob, o[None] + tt[:, None] * d[None])
                k = int(np.argmin(np.abs(F)))
                if best is None or abs(F[k]) < best[0]:
                    best = (abs(float(F[k])), ob, o + tt[k] * d)
            if best is None or best[0] > radius:
                return None
            _, ob, p = best
            e = 2e-4
            for _ in range(3):
                f0 = float(field_of(sc, pk, ob, p[None])[0])
                g = np.array([(field_of(sc, pk, ob, (p + dd)[None])[0] - field_of(sc, pk, ob, (p - dd)[None])[0]) / (2 * e)
                              for dd in np.eye(3) * e])
                gl = float(np.linalg.norm(g))
                if gl < 1e-9:
                    return None
                p = p - f0 * g / (gl * gl)
            n = g / gl
        return PickResult(True, p, n, ob.id, float(np.dot(p - o, d)))

    def _stroke_begin(self, view, x, y, settings, pending_ok=True):
        if settings.get('tool') == 'snake':        # fs-brush 11-11 (design 4.5): Snake Hook = Grab, flat-topped falloff
            from ..core import brush as _BRs
            if _BRs.SNAKE_PATH:
                # fs-brush 11-12 (the flat-topped plug): a PATH tool (Blender's Snake Hook: it keeps
                # picking up what is under the brush as it moves; never the anchored flow), a smooth falloff, and
                # Magnify keeping the girth (core/brush.py snake_step_back)
                # fs-shell-6 11-14 (all three primary settings on every brush): the Softness /
                # Hardness slot's curve ('custom' + 'curve', fs-tools' legacy_curve) is kept; it was replaced here by
                # SNAKE_FALLOFF with curve=None, so Softness 0 and 100 gave the same Snake stroke (t74/check_softness_any)
                own = settings.get('falloff') == 'custom' and settings.get('curve')
                settings = dict(settings, tool='grab', falloff='custom' if own else _BRs.SNAKE_FALLOFF,
                                curve=list(settings['curve']) if own else None, snake=True, grab_flow=False)
            else:
                settings = dict(settings, tool='grab', falloff='custom', curve=list(_BRs.SNAKE_CURVE), snake=True)
        self._stroke = None
        self._stroke_pending = None
        pr = self._pick(view, x, y)
        if not pr.hit:
            try:
                nm = self._near_miss(view, x, y, float(settings.get('radius', 0.01)))
            except Exception:
                self.gpu_error = traceback.format_exc()
                nm = None
            if nm is not None:
                pr = nm
        if not pr.hit:
            # hand2b: a press on a silhouette pixel (the viewport's object buffer, from a coarser preview, says the
            # form is there; the exact ray grazes past it) takes the nearest hit within 2 px. After the framing fit
            # (hand2b) the selftest's S4 presses landed on such pixels and two strokes were lost.
            for r_ in (1.0, 2.0):
                for dx, dy in ((r_, 0), (-r_, 0), (0, r_), (0, -r_), (r_, r_), (-r_, r_), (r_, -r_), (-r_, -r_)):
                    pr = self._pick(view, x + dx, y + dy)
                    if pr.hit:
                        break
                if pr.hit:
                    break
            if not pr.hit:
                if pending_ok and settings.get('tool') not in ('grab', 'stretch'):
                    self._stroke_pending = (settings,)    # Blender: the stroke starts where the drag reaches the form
                return False
        from ..core.scene import is_protected as _isp
        if _isp(self.hist.scene.get(pr.object_id)):
            self.message.emit('%s is protected: Edit > Unprotect a part to sculpt it.' % self.hist.scene.get(pr.object_id).name)
            return False
        s = copy.deepcopy(settings)
        # fs-brush 11-11 (design 4.8): a hidden region is never stroked, and 'This region only' (the auto-mask menu's
        # 'only_region') keeps the stroke to the region under the press. Both go into the stroke's own settings, so a
        # replay never depends on what the view hides (brush.region_gate)
        if s.get('tool') != 'region':
            _hid = (getattr(self.hist.scene, 'view_regions_hidden', None) or {}).get(pr.object_id) or ()
            if _hid:
                s['skip_regions'] = sorted(int(i) for i in _hid)
            if s.pop('only_region', False):
                from ..core import brush as _BRr
                with self.lock:
                    s['automask_region'] = int(_BRr.region_at(self.hist.scene, pr.object_id, pr.point))
        else:
            s.pop('only_region', None)
        if s.get('tool') in ('swell', 'settle'):  # 11-09: contacts = locked parts + the other selected objects
            s['contact_ids'] = [int(i) for i in (view.get('selected') or []) if int(i) != int(pr.object_id)]
            s['collide'] = True
            # 11-09 (contact means touching, not close): Swell / Settle rest ON the contact (gap 0 by default; the
            # brushes' no-new-overlap gap stays the collider's own setting). test_contact_gap: |d| <= 2.7 um at gap 0
            s['collide_gap'] = float(s.get('contact_gap', 0.0))
        # fs-brush 11-11: a part with touch / slide partners collides with them under every brush, so
        # touching parts never fuse (core/collide.py part_partners; an explicit collide=False still wins)
        try:
            from ..core import collide as _CLp
            with self.lock:
                if _CLp.part_partners(self.hist.scene, pr.object_id):
                    s.setdefault('collide', True)
        except AttributeError:
            pass
        s.setdefault('smooth_quad', True)         # fs-brush 11-11: Smooth at the brush's scale (gpu/sculpt.cu quad_smooth)
        # fs-brush 11-11: its fit made robust (no pits in a fold beside a pulled form; brush.NBP_CELL), its normal from 2 h
        s.setdefault('smooth_fit', os.environ.get('FSB_SMOOTH_FIT', 'robust2'))   # 22:00: robust2 (the same-sheet weight)
        s.setdefault('smooth_qse', 2.0)
        s.setdefault('mid_snap', True)            # fs-brush 11-11: one centred dab near a mirror plane (brush.mirror_dabs)
        s.setdefault('snap_smooth', False)        # fs-brush 11-11: Smooth mirrors as a feathered pair, never snapped (brush.mirror_plan)
        # fs-brush 11-11 (brush.mirror_plan): under one mirror axis the snap keeps midline forms single (no doubling) and a
        # Grab's drag across the plane goes to a feathered pair, so a sideways Grab near the midline widens both sides (the
        # snap alone cancelled it: test1 'neck width' 0.03 of the drag; the plain pair split a midline nose)
        s.setdefault('mirror_feather', os.environ.get('FSB_MIRROR_FEATHER', '1') != '0')
        s.setdefault('area_plane', True)          # fs-brush 11-11: Clay / Flatten on the area plane (gpu/sculpt.cu area_plane)
        s.setdefault('pinch_bound', True)         # fs-brush 11-11: Pinch bounded so it never folds the field
        s.setdefault('strength_v2', os.environ.get('FSB_STRENGTH_V2', '1') != '0')   # fs-brush 11-11: every tool's effect rises with strength
        from ..core import brush as _BRa
        s.setdefault('autosmooth', _BRa.AUTOSMOOTH.get(s.get('tool'), 0.0))   # fs-brush 11-11 (design 4.7)
        r0 = float(s.get('radius', 0.01))
        with self.lock:
            o = self.hist.scene.get(pr.object_id)
            rmax = max_radius_for(o.sculpt.h, s.get('tool') if not s.get('snake') else 'snake') * max(float(o.transform['scale']), 1e-6)
        s['radius'] = min(max(r0, MIN_BRUSH_RADIUS), rmax)     # V1.3: a dab's lattice stays bounded (hand1: per lattice)
        if s['radius'] != r0:
            self.message.emit('Brush radius %.1f mm is outside %.1f - %.0f mm on this object (%.1f mm lattice); '
                              'using %.1f mm. File > New bust uses a 1 mm lattice for brushes up to 80 mm.' % (
                r0 * 1000, MIN_BRUSH_RADIUS * 1000, rmax * 1000, o.sculpt.h * 1000, s['radius'] * 1000))
        if s['tool'] in _BRa.FT_LINE:
            # fs-tools patch 4: the Pencil's tip acts where it is at each dab; dabs every tenth of the tip's width or half a
            # lattice step, whichever is closer (TOOLSET_DESIGN 3.3; toolset/ftools.py FtStroke._pencil), along the drag
            hw_ = float(o.sculpt.h) * max(float(o.transform['scale']), 1e-6)
            kt_ = 0.07 if s.get('tip', 'round') == 'point' else 0.2   # a cone's floor rises spacing / (2 r) between dabs
            s['spacing'] = max(min(kt_ * s['radius'], 0.5 * hw_), 1e-6) / s['radius']
        cam = view['cam']
        f, r, u = cam.basis()
        st = {'oid': pr.object_id, 'settings': s, 'last': pr.point.copy(), 'last_n': pr.normal.copy(),
              'carry': 0.0, 'anchor': pr.point.copy(), 'anchor0': pr.point.copy(), 'n0': pr.normal.copy(),
              'plane_n': f.copy(), 'plane_last': pr.point.copy(), 'view': view, 't_press': time.perf_counter()}
        if s['tool'] == 'grab':                            # fs-brush 11-12: the anchored flow Grab (history.py, gpu/sculpt.py)
            s.setdefault('grab_flow', GRAB_FLOW)
        if s['tool'] == 'stretch':                         # fs-brush 11-12: Stretch as an anchored elongation
            s.setdefault('stretch_flow', STRETCH_FLOW)
        if s['tool'] in ('grab', 'stretch') and s.get('drag_form', DRAG_FORM):   # fs-brush 11-12: the drag's constraint from the form (P1)
            st['plane_n'] = drag_constraint(f, pr.normal)
        if s['tool'] in ('grab', 'stretch') and s.get('invert', False):
            # fs-brush 11-12: Ctrl = along the press normal only, in either drag rule (Blender's Ctrl on Grab: normal weight)
            st['drag_line'] = True
            st['cam_up'] = u.copy()
        first = [] if s['tool'] in ('grab', 'stretch') else [{'p': pr.point.tolist(), 'n': pr.normal.tolist()}]
        if first and view.get('pressure') is not None:          # 11-10: the pen's pressure scales the dab
            first[0]['k'] = round(float(view['pressure']), 4)
        layer = s.pop('layer', 'sculpt')
        dscale = s.pop('detail_scale', 0.002)
        if s['tool'] == 'smooth' and layer in ('detail', 'shape'):
            # edit by LAYER (DESIGN_NOTES 10d): smooth one band of the sculpt layer and keep the other
            ls = {k: s[k] for k in ('radius', 'strength', 'falloff', 'curve', 'spacing', 'symmetry', 'use_mask')
                  if k in s}
            ls.update(layer=layer, detail_scale=float(dscale))
            cmd = FT.LayerSmooth(pr.object_id, ls, first)
        else:
            cmd = BrushStroke(pr.object_id, s, first)
            cmd.live = True                    # hand1: no re-distance at the press (only on release)
        t0 = time.perf_counter()
        self._push(cmd)
        if first:
            self._dab_ms = 0.5 * self._dab_ms + 0.5 * (time.perf_counter() - t0) * 1000.0
        st['cmd'] = cmd
        from ..core import brush as _BR
        st['_angle_n0'] = len(_BR.MASK_ANGLE_LOG)
        self._stroke = st
        self._select = [pr.object_id]
        self.hoverReady.emit({'hit': True, 'point': pr.point, 'normal': pr.normal, 'oid': pr.object_id,
                              'x': x, 'y': y})
        return True

    def _moves_guarded(self, moves):
        """fs-brush 11-12 (fs-bench-full F5, probe5's stacks): _stroke_moves ran outside the per-job try, so a failing batch
        (the card out of memory on a whole-body Smooth on her) went to the turn's outer except and every other job taken
        in that turn was dropped: the stroke's release never ran, nor anything behind it, and the window looked hung.
        Now the failure is reported, the stroke is halted (its further moves are skipped; its release still ends it,
        keeping what it did), and the turn's other jobs run. Out of card memory: the sculpt scratch and the pool's
        cached blocks are given back first."""
        try:
            return bool(self._stroke_moves(moves))
        except Exception as e:
            oom = 'OutOfMemory' in type(e).__name__ or 'out of memory' in str(e).lower()
            if self._stroke is not None:
                self._stroke['halted'] = True
            if oom:
                try:
                    gs = getattr(self, 'gpu_sculpt', None)
                    if gs is not None and getattr(gs, '_scratch', None):
                        gs._scratch.clear()
                    from ..gpu import scene as _GSm
                    _GSm.release_cached('stroke out of memory')
                except Exception:
                    pass
                self.failed.emit('The card ran out of memory in this stroke, so it stopped here (what it did is kept; '
                                 'release the mouse to finish it). Try a smaller brush or a closer view. ' +
                                 os.linesep + traceback.format_exc(limit=-8))   # fs-brush 11-12: the innermost frames
            else:
                self.failed.emit('stroke_move failed (the stroke stopped here; release to finish it): %s ' % e +
                                 os.linesep + traceback.format_exc(limit=-8))   # where it failed, not the engine loop
            return True

    def _stroke_moves(self, moves):
        from ..core import brush as _BRsn
        self._prof_t_moves = time.perf_counter()
        st = self._stroke
        if st is not None and st.get('halted'):
            return False                         # fs-brush 11-12: a failed stroke takes no more moves (its release ends it)
        if st is None and getattr(self, '_stroke_pending', None) is not None and moves:
            # fs-brush 11-11: a press off the form with a brush: the stroke starts at the first move over the form
            sett = self._stroke_pending[0]
            for i, (_, view, x, y) in enumerate(moves):
                if self._stroke_begin(view, x, y, sett, pending_ok=False):
                    moves = moves[i + 1:]
                    break
                self._stroke_pending = (sett,)
            st = self._stroke
        if st is None:
            return False
        s = st['settings']
        tool = s['tool']
        rad = float(s['radius'])
        dabs = []
        pre = {}
        if tool not in ('grab', 'stretch') and self.gpu_view is not None and len(moves) > 1:
            # r30: every queued move picked in one launch on the card's field
            try:
                with self.lock:
                    res = self.gpu_view.pick_many(self.hist.scene, [(m[1]['cam'], m[2], m[3], m[1]['w'], m[1]['h'])
                                                                    for m in moves], self._brush_exclude(moves[0][1]))
                from ..render.raymarch import PickResult
                for i, (hit, p_, n_, oid, t_) in enumerate(res):
                    pre[i] = PickResult(True, p_, n_, oid, t_) if hit and oid >= 0 else PickResult(False, None, None, -1, float('inf'))
            except Exception:
                pre = {}
        for mi, (_, view, x, y) in enumerate(moves):
            n_before = len(dabs)
            self._apply_pressure_from = (n_before, view.get('pressure'))
            cam = view['cam']
            if tool in ('grab', 'stretch'):
                if st.get('drag_line'):
                    p = drag_on_normal(cam, x, y, view['w'], view['h'], st['anchor0'], st['n0'], st['cam_up'])
                else:
                    p = _ray_plane(cam, x, y, view['w'], view['h'], st['anchor0'], st['plane_n'])
                if p is None:
                    continue
                v = p - st['plane_last']
                if s.get('strength_v2', False):
                    # fs-brush 11-11 (Grab and Stretch ignored strength): the surface moves strength x
                    # the drag (1 = it follows the cursor; capped there), and the anchor with it
                    v = v * min(max(float(s.get('strength', 1.0)), 0.0), 1.0)
                L = float(np.linalg.norm(v))
                if L < 1e-9:
                    st['plane_last'] = p
                    continue
                if s.get('snake', False) and _BRsn.SNAKE_PATH:
                    # fs-brush 11-12: Snake Hook's steps every SNAKE_SPACING r of travel (core/brush.py snake_steps)
                    for vv_ in _BRsn.snake_steps(st.setdefault('snake_step_state', {}), v, rad):
                        dabs.append({'p': st['anchor'].tolist(), 'n': st['n0'].tolist(), 'v': vv_.tolist()})
                        st['anchor'] = st['anchor'] + vv_
                    st['plane_last'] = p
                    st['hover'] = (st['anchor'], st['n0'])
                    continue
                k = max(1, int(math.ceil(L / (0.35 * rad))))      # hand1: was 0.2 r (16 dabs per 40 mm at 60 mm)
                for _ in range(k):
                    c = st['anchor'] if tool == 'grab' else st['anchor0']
                    dabs.append({'p': c.tolist(), 'n': st['n0'].tolist(), 'v': (v / k).tolist()})
                    if tool == 'grab':
                        st['anchor'] = st['anchor'] + v / k
                st['plane_last'] = p
                st['hover'] = (st['anchor'], st['n0'])
                continue
            pr = pre[mi] if mi in pre else self._pick(view, x, y)
            if not pr.hit or pr.object_id != st['oid']:
                continue
            p, n = pr.point, pr.normal
            if tool in _BRsn.FT_TOOLS:
                # fs-tools 11-14: the approved held tools (Strength is a rate): one dab per timed sample at the point under
                # the cursor, covering the time since the last sample ('t': s since the press, posted ~120 a second while
                # the button is held; without it the engine's clock). Flat Swell / Settle keep the press facing.
                t_ = view.get('t')
                t_ = float(t_) if t_ is not None else time.perf_counter() - st.get('t_press', time.perf_counter())
                dt_ = max(t_ - float(st.get('ft_t', 0.0)), 0.0)
                st['ft_t'] = max(t_, float(st.get('ft_t', 0.0)))
                nd_ = st['n0'] if (tool != 'ft_flatten' and s.get('follow', 'follow') == 'flat') else n
                if dt_ > 0.0:
                    dabs.append({'p': np.asarray(p).tolist(), 'n': np.asarray(nd_).tolist(), 'dt': round(dt_, 6)})
                st['last'] = p.copy()
                st['last_n'] = n.copy()
                st['hover'] = (p, n)
                continue
            prev = st['last']
            seg = p - prev
            L = float(np.linalg.norm(seg))
            step = max(float(s['spacing']) * rad, 1e-5)
            jump_ = 4.0 * (max(rad, 0.0025) if tool in _BRsn.FT_LINE else rad)   # fs-tools patch 4: a fine Pencil tip's
            if L > jump_:                         # a jump (over a gap): restart the path here   # fast stroke isn't a jump
                dabs.append({'p': p.tolist(), 'n': n.tolist()})
                st['carry'] = 0.0
            elif L > 0:
                t = step - st['carry']
                line = (tool in ('pinch', 'crease') and s.get('strength_v2', False)) or tool in _BRsn.FT_LINE   # + Pencil's
                # direction (fs-tools patch 4: the square tip turns with the stroke)
                if line:
                    st['dir'] = (seg / L).tolist()      # fs-brush 11-11: Pinch / Crease pull toward the stroke line
                while t <= L:
                    q = prev + seg * (t / L)
                    nn = st['last_n'] + (n - st['last_n']) * (t / L)
                    nn = nn / max(np.linalg.norm(nn), 1e-12)
                    dabs.append({'p': q.tolist(), 'n': nn.tolist()} if not line else {'p': q.tolist(), 'n': nn.tolist(), 'v': st['dir']})
                    t += step
                st['carry'] = L - (t - step)
            st['last'] = p.copy()
            st['last_n'] = n.copy()
            st['hover'] = (p, n)
            pres = view.get('pressure')
            if pres is not None:                  # 11-10: this move's dabs carry the pen's pressure
                for d_ in dabs[n_before:]:
                    d_['k'] = round(float(d_.get('k', 1.0)) * float(pres), 4)
        if st.get('snake_flush') and s.get('snake', False) and _BRsn.SNAKE_PATH:
            st['snake_flush'] = False              # fs-brush 11-12: the release lays Snake Hook's carried travel
            for vv_ in _BRsn.snake_steps(st.setdefault('snake_step_state', {}), np.zeros(3), rad, flush=True):
                dabs.append({'p': st['anchor'].tolist(), 'n': st['n0'].tolist(), 'v': vv_.tolist()})
                st['anchor'] = st['anchor'] + vv_
        if tool not in ('grab', 'stretch') and 'hover' in st:
            # hand2: slow strokes. The carried path (under one spacing) was held until a whole spacing was covered,
            # so a slow 2 mm move did nothing at r 40 (10 mm spacing). Lay it now as a partial dab at the cursor,
            # strength scaled by carry / step: the deposit per unit of path stays the same. Recorded as a sample
            # ('k'), so replay is exact.
            step = max(float(s['spacing']) * rad, 1e-5)
            if st['carry'] >= 0.2 * step:
                p, n = st['hover']
                kk = float(st['carry']) / step
                pres = moves[-1][1].get('pressure') if moves else None       # 11-10: the pen's pressure here too
                if os.environ.get('FSB_FRAC', '1') != '0':
                    # fs-brush 11-11 (M: repeated Smooth strokes differed 5.2 %): the path share is its own field 'f',
                    # scaling the dab's AMOUNT linearly; 'k' (the pen) stays a strength. As a strength, the share went
                    # through the concave strength response of Smooth / Flatten / Pinch / Crease (two half dabs smoothed
                    # ~16 % more than one whole), so a stroke's total depended on how its frames fell.
                    d_ = {'p': np.asarray(p).tolist(), 'n': np.asarray(n).tolist(), 'f': round(kk, 4)}
                    if pres is not None:
                        d_['k'] = round(float(pres), 4)
                    dabs.append(d_)
                else:
                    if pres is not None:
                        kk *= float(pres)
                    dabs.append({'p': np.asarray(p).tolist(), 'n': np.asarray(n).tolist(), 'k': round(kk, 4)})
                if st.get('dir') is not None and tool in ('pinch', 'crease'):
                    dabs[-1]['v'] = st['dir']
                st['carry'] = 0.0
        # hand2b: the dabs the last batch had no time for come first (the stroke is laid in order, never thinned
        # into beads or merged past the budget: verifier items 3, 18)
        dabs = st.pop('backlog', []) + dabs
        if not dabs:
            if 'hover' in st:
                self._emit_stroke_hover(st)
            return False
        if tool in ('grab', 'stretch') and getattr(st.get('cmd'), 'on_gpu', False) and len(dabs) > 1:
            dabs = self._merge_drag(dabs, rad)
        dabs, later = self._budget(dabs, tool)
        if later:
            st['backlog'] = later
            self.post('stroke_flush')            # the rest goes in the next batch, after a frame
        t0 = time.perf_counter()
        self._prof_budget_ms = (t0 - self._prof_t_moves) * 1000 if hasattr(self, '_prof_t_moves') else 0.0
        if s.get('snake', False):
            # fs-brush 11-12 (Rake, Blender's Snake Hook): each dab carries the path's direction before / after it, read
            # over chords (core/brush.py snake_stamp), in the order the stroke applies them (after the merge and budget)
            from ..core import brush as _BRk
            _BRk.snake_stamp(st.setdefault('snake_rake_state', {}), dabs, rad)
        with self.lock:
            if self.hist.done and self.hist.done[-1] is st['cmd']:
                self.hist.extend_stroke(dabs)
                self._prof_extend_ms = (time.perf_counter() - t0) * 1000
            else:
                self._stroke = None
                return False
        ms = (time.perf_counter() - t0) * 1000.0 / len(dabs)
        self._dab_ms = 0.6 * self._dab_ms + 0.4 * ms
        self._emit_stroke_hover(st)
        return True

    def _budget(self, dabs, tool):
        """-> (dabs to apply now, dabs for the next batch). hand2b (verifier items 3, 18): one batch's dab work stays
        near STROKE_BUDGET_MS; the dabs past it are DEFERRED to the next batch (a frame is drawn in between), so the
        form follows the cursor's whole path in order and catches up when the cursor slows. Before, hand1 thinned the
        queued path (gaps: Clay / Draw strokes drawn as strings of beads) and merged Grab dabs, which hand2 split back
        to 0.7 n dabs at a 0.5 r cap (no saving). Only a backlog over BACKLOG_BATCHES batches is reduced, and never
        past continuity: brush dabs no farther apart than MAX_SPACING r, grab / stretch dabs merged to at most
        0.5 r of drag each (the fold limit of p - v w is 0.67 r). The applied dabs are what the stroke records, so
        replay is exact."""
        n = len(dabs)
        st_ = self._stroke
        if st_ is not None and getattr(st_.get('cmd'), 'on_gpu', False):
            return dabs, []                  # r30: GPU dabs (~1 ms each): the whole backlog in this batch
        keep = max(1, int(STROKE_BUDGET_MS / max(self._dab_ms, 0.05)))
        if n <= keep:
            return dabs, []
        if n > BACKLOG_BATCHES * keep:
            red = self._reduce(dabs, tool)
            self.thinned += n - len(red)
            dabs = red
        return dabs[:keep], dabs[keep:]

    GRAB_MERGE = float(os.environ.get('FSB_GRAB_MERGE', '0.25'))   # fs-brush 11-11: a batch's drag dabs merged up to this r fraction

    def _merge_drag(self, dabs, rad):
        """fs-brush 11-11 (M: Grab 5 fps at r 80 with 13-21 full-ball dabs a frame, one per mouse event): the frame's
        consecutive drag dabs merged into one (the first anchor, the summed drag) while the sum stays within GRAB_MERGE r,
        as Blender applies a grab once per stroke step, not per mouse event. The merged dabs are what the stroke
        records, so replay is exact."""
        out = []
        acc = None
        lim = self.GRAB_MERGE * rad
        for d in dabs:
            v = np.asarray(d['v'], float)
            if acc is not None and 'k' not in d and float(np.linalg.norm(acc['v'] + v)) <= lim:
                acc['v'] = acc['v'] + v
                continue
            if acc is not None:
                out.append({'p': acc['p'], 'n': acc['n'], 'v': acc['v'].tolist()})
            acc = {'p': d['p'], 'n': d['n'], 'v': v.copy()}
        if acc is not None:
            out.append({'p': acc['p'], 'n': acc['n'], 'v': acc['v'].tolist()})
        return out

    def _reduce(self, dabs, tool):
        """the continuity-safe reduction of a long backlog (see _budget)"""
        n = len(dabs)
        st = self._stroke
        rad = float(st['settings']['radius']) if st else 0.0
        if rad <= 0.0 or n < 3:
            return dabs
        if tool in ('grab', 'stretch'):
            out = []
            acc = None
            for d in dabs:
                v = np.asarray(d['v'], float)
                if acc is not None and float(np.linalg.norm(acc['v'] + v)) <= 0.5 * rad:
                    acc['v'] = acc['v'] + v
                    continue
                if acc is not None:
                    out.append({'p': acc['p'], 'n': acc['n'], 'v': acc['v'].tolist()})
                acc = {'p': d['p'], 'n': d['n'], 'v': v.copy()}
            out.append({'p': acc['p'], 'n': acc['n'], 'v': acc['v'].tolist()})
            return out
        from ..core import brush as _BRr
        if tool in _BRr.FT_LINE:
            return dabs                   # fs-tools patch 4: every Pencil dab is kept (a dropped one scallops the line)
        if tool in _BRr.FT_TOOLS:
            # fs-tools 11-14: held tools integrate time (Strength is a rate): a dropped dab's dt goes to the next kept one,
            # so the backlog deposits what it held (a still cursor's dabs merge exactly: the rules are linear in dt).
            # A dab is kept when the next one lies farther than MAX_SPACING r from the last kept one, or at the end.
            out = []
            acc = 0.0
            last = np.asarray(dabs[0]['p'], float)
            lim = MAX_SPACING * rad
            for i in range(n):
                acc += float(dabs[i].get('dt', 0.0))
                q = np.asarray(dabs[i]['p'], float)
                nxt = np.asarray(dabs[i + 1]['p'], float) if i + 1 < n else None
                if nxt is None or float(np.linalg.norm(nxt - last)) > lim:
                    d_ = dict(dabs[i])
                    d_['dt'] = round(acc, 6)
                    out.append(d_)
                    acc = 0.0
                    last = q
            return out
        # brush dabs: keep a dab whenever the next one would be farther than MAX_SPACING r from the last kept one
        out = [dabs[0]]
        last = np.asarray(dabs[0]['p'], float)
        lim = MAX_SPACING * rad
        for i in range(1, n):
            q = np.asarray(dabs[i]['p'], float)
            nxt = np.asarray(dabs[i + 1]['p'], float) if i + 1 < n else None
            if 'k' in dabs[i] or 'f' in dabs[i] or nxt is None or float(np.linalg.norm(nxt - last)) > lim:
                out.append(dabs[i])
                last = q
        return out

    def _emit_stroke_hover(self, st):
        p, n = st['hover']
        self.hoverReady.emit({'hit': True, 'point': np.asarray(p), 'normal': np.asarray(n), 'oid': st['oid'],
                              'x': None, 'y': None})

    def _log_mask_angles(self, st):
        """11-09: per tagged Mask stroke, the closest opposite-facing surface's angle over its dabs, appended
        to %LOCALAPPDATA%/FieldStudio/mask_angles.jsonl; a message when it comes within 5 deg of the cutoff"""
        from ..core import brush as _BR
        if st['settings'].get('tool') != 'mask' or not st['settings'].get('paint_dir'):
            return
        n0 = st.get('_angle_n0', 0)
        a = [x for x in _BR.MASK_ANGLE_LOG[n0:] if x is not None]
        rec = {'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'dabs': len(_BR.MASK_ANGLE_LOG) - n0,
               'closest_opposite_deg': round(min(a), 2) if a else None, 'cutoff_deg': _BR.MASK_DIR_CUTOFF_DEG}
        d = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'FieldStudio')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'mask_angles.jsonl'), 'a', encoding='utf-8') as f:
            import json as _json
            f.write(_json.dumps(rec) + '\n')
        if a and min(a) < _BR.MASK_DIR_CUTOFF_DEG + 5.0:
            self.message.emit('Mask: a facing surface came within %.1f deg of the "other side" cutoff (%.0f deg); paint '
                              'there may be shared. Check it with the mask shown.' % (min(a), _BR.MASK_DIR_CUTOFF_DEG))

    def _stroke_end(self):
        self._stroke_pending = None
        _tr = time.perf_counter()
        self.prof_release = {}
        try:
            return self._stroke_end_()
        finally:
            self.prof_release['total_ms'] = (time.perf_counter() - _tr) * 1000

    def _stroke_end_(self):
        st = self._stroke
        if st is not None and st['settings'].get('snake', False) and not st.get('halted'):
            st['snake_flush'] = True                 # fs-brush 11-12: Snake Hook's carried travel first (snake_steps)
            self._stroke_moves([])
        if st is not None and st.get('backlog'):
            # hand2b: the release lays the rest of the path first (in budget-sized batches, all of it)
            while st.get('backlog') and self._stroke is st:
                self._stroke_moves([])
        self._stroke = None
        if st is None:
            return False
        if not st['cmd'].samples and self.hist.done and self.hist.done[-1] is st['cmd']:
            with self.lock:                        # a click with grab / stretch that never moved: no undo step
                self.hist.undo()
                self.hist.drop_undone_last()
            return True
        _t = time.perf_counter()
        if self._view is not None and self.hist.done and getattr(self.hist.done[-1], 'pending', False):
            # hand1: show the finished stroke first (a preview frame), then re-distance (0.5-2.5 s on a 1 mm bust
            # lattice at 30-60 mm, M), so the release never looks frozen
            self._emit_scene()
            self._render(self._view, True)
        self.prof_release['preview_ms'] = (time.perf_counter() - _t) * 1000
        _t = time.perf_counter()
        with self.lock:                            # mouse release: re-distance the stroke's bricks (core)
            self.hist.end_stroke()
        self.prof_release['end_stroke_ms'] = (time.perf_counter() - _t) * 1000
        gs_ = getattr(self, 'gpu_sculpt', None)
        if gs_ is not None:
            self.prof_release['finish_ms'] = gs_.stats.get('finish_ms')
            ev = getattr(gs_, 'topo_event', None)
            if ev:                                 # fs-brush 11-11 (design D4): the stroke-end topology check's report
                gs_.topo_event = None
                self.topoChanged.emit(dict(ev, stroke_step=getattr(st['cmd'], 'serial', None)))
        # fs-brush 11-11 (R3): the frames just after a release are quick ones (one pass where the stroke changed the
        # picture); the light's refinement follows on idle frames, so a next press is never held behind a 40 ms frame
        self._interactive_until = max(getattr(self, '_interactive_until', 0.0), time.perf_counter() + 0.15)
        try:
            self._log_mask_angles(st)
        except Exception:
            pass
        return True

    # ---- render -------------------------------------------------------------------------------------------------
    def _render(self, view, interactive):
        # 11-08 night: the mask is drawn over the card's frame too (gpu/mask_overlay.py); before, Show mask meant the CPU
        if self.gpu_view is not None and view.get('mode', 'shaded') in getattr(self, 'card_modes', ('shaded',)) and \
                time.perf_counter() >= getattr(self, '_card_retry_at', 0.0):
            try:
                try:
                    fr = self._render_gpu(view, interactive)
                    if getattr(self, '_card_fails', 0):
                        self.message.emit('The card view is back.')
                    self._card_fails = 0
                    self.render_device, self.card_error = 'card', None
                    return fr
                except Exception as e:
                    # 11-09 night: out of card memory is not a dead card: give the pool's cached blocks back and try
                    # once more before drawing on the CPU (a 22:37 run gave up the card on one failed allocation).
                    # fs-brush 11-11 (twice in one night her store failed to load under RAM pressure with
                    # cudaErrorAlreadyMapped / cudaHostAlloc on a pinned host buffer): host-side allocation failures
                    # are memory pressure too
                    if not _pressure_error(e):
                        raise
                    import gc as _gc
                    import cupy as _cp
                    _gc.collect()
                    _cp.get_default_memory_pool().free_all_blocks()
                    _cp.get_default_pinned_memory_pool().free_all_blocks()
                    self.message.emit('The card ran short of memory (%s); cached blocks released, trying again.'
                                      % str(e).splitlines()[0][:80])
                    fr = self._render_gpu(view, interactive)
                    self._card_fails = 0
                    self.render_device, self.card_error = 'card', None
                    return fr
            except Exception as e2:
                self.gpu_error = traceback.format_exc()
                self._card_fails = getattr(self, '_card_fails', 0) + 1
                self.card_error = str(e2).splitlines()[0][:200] if str(e2) else type(e2).__name__
                if _pressure_error(e2) and self._card_fails < 3:
                    # fs-brush 11-11: pressure is not a dead card: this frame on the CPU, the card again in 3 s (the
                    # window says so: render_device 'cpu' + card_error until it is back)
                    self._card_retry_at = time.perf_counter() + 3.0
                    self.failed.emit('The card could not draw this frame (%s); drawing on the CPU, trying the card '
                                     'again in 3 s (%d of 3).' % (self.card_error, self._card_fails))
                else:
                    self.gpu_view = None
                    self.failed.emit('GPU viewport failed; drawing on the CPU from now on:\n' + self.gpu_error)
        self.render_device = 'cpu'
        return self._render_cpu(view, interactive)

    def _apply_parts_view(self):
        from ..core.scene import is_protected
        sc = self.hist.scene
        want = {o.id for o in sc.objects if is_protected(o)} if self.parts_view in ('ghost', 'hidden') else set()
        if want != self._pv_hidden:
            sc.view_hidden.difference_update(self._pv_hidden - want)
            sc.view_hidden.update(want)
            self._pv_hidden = want
            if self.gpu_view is not None:
                self.gpu_view.note_change()

    def _render_gpu(self, view, interactive):
        with self.lock:
            self._apply_parts_view()
            self._apply_xray(view)
            fr = self.gpu_view.frame(self.hist.scene, view['cam'], view['w'], view['h'],
                                     interactive=interactive or self._stroke is not None,
                                     selected=view.get('selected'))
        st = fr.stats
        st['renderer'] = 'gpu'
        if view.get('show_mask') or view.get('tool') == 'mask':
            # the Mask brush's paint, visible while painting (protected parts are marked with it)
            from ..gpu import mask_overlay
            with self.lock:
                rgb0 = fr.rgb
                nrm = None                       # 11-09: direction-tagged paint reads the surface's own (C1) normal
                try:
                    gn = self.gpu_view.gpu.d_gn
                    if tuple(gn.shape[:2]) == tuple(np.asarray(fr.depth).shape[:2]):
                        cp_ = self.gpu_view.gpu.cp
                        flat = gn.reshape(-1, 4)
                        nrm = lambda ix: flat[cp_.asarray(ix)][:, :3].get()   # r2: only the painted pixels' normals
                except Exception:
                    nrm = None
                fr.rgb = mask_overlay.overlay(self.hist.scene, view['cam'], fr.rgb, fr.depth, fr.obj, nrm)
            st['mask_overlay_px'] = int((fr.rgb != rgb0).any(2).sum()) if fr.rgb is not rgb0 else 0
        if self.parts_view == 'ghost' and self._pv_hidden:
            from ..gpu import parts_overlay
            with self.lock:
                fr.rgb = parts_overlay.ghost(self.gpu_view.gpu, view['cam'], fr.rgb, np.asarray(fr.depth), self._pv_hidden)
        if self.show_overlaps and st.get('converged') and not interactive and self._stroke is None:
            from ..gpu import parts_overlay
            with self.lock:
                fr.rgb, nov = parts_overlay.overlaps(self.hist.scene, view['cam'], fr.rgb, np.asarray(fr.depth),
                                                     np.asarray(fr.obj))
            st['overlap_px'] = nov
            if nov and nov != getattr(self, '_last_nov', 0):
                self.message.emit('New overlaps (new overlaps only mode): %d px shown red. Edit > Check overlaps '
                                  'counts them.' % nov)
            self._last_nov = nov
        st['quality'] = 2 if st['converged'] else 0          # the UI formats q%d: 0 = still converging, 2 = converged
        with self.lock:                                       # fs-shell 11-12 (the card read 0.000, the CPU 1.000):
            st['lipschitz'] = self.hist.scene.lipschitz_bound()   # the scene's tracked bound, as the CPU frame reports
        self.frames += 1
        self.last_render_ms = st['ms']
        final = bool(st['converged'])
        # progressive: keep refining while the light converges (each pass is a few ms at UHD)
        self._refine_at = None if final else time.perf_counter()
        self.frameReady.emit({'rgb': fr.rgb, 'obj': fr.obj, 'depth': fr.depth, 'stats': st,
                              'cam': view['cam'], 'w': view['w'], 'h': view['h'], 'seq': view.get('seq', 0),
                              'final': final, 'quality': st['quality'], 'renderer': 'gpu',
                              'sel_alpha': fr.sel_alpha, 'sel_ids': fr.sel_ids,
                              # fs-shell-2 11-14: select inside parts 3 / 4 (gpu/viewmodes.vol_mask)
                              'sel_hidden': getattr(fr, 'sel_hidden', None), 'hover_mask': getattr(fr, 'hover_mask', None)})

    def _apply_xray(self, view):
        """fs-shell 11-13 (Tier 0 part 4): the view's X-ray amount and per-object opacities onto the card's volumes (view
        state only: never in the scene, its hash or a save); the frame composites the layers (gpu/viewmodes.py xray)"""
        gv = self.gpu_view
        op = view.get('opacity') or {}
        changed = False
        for v in gv.gpu.vols:
            o = float(op.get(getattr(v, 'oid', None), 1.0))
            if abs(getattr(v, 'opacity', 1.0) - o) > 1e-6:
                v.opacity = o
                changed = True
        if changed:
            gv.gpu.refresh_transforms()
        gv.xray = float(view.get('xray', 0.0) or 0.0)
        gv.xray_any = changed or any(float(x) < 0.999 for x in op.values())

    def _render_cpu(self, view, interactive):
        q = view.get('quality', 'auto')
        if q == 'auto':
            q = 0 if interactive else 2
        else:
            q = int(q)
            if interactive and q > 0:
                q = 0
        scale = None
        if q == 0:
            # hand1: adaptive preview resolution: 2 normally, 3 or 4 when the last preview was slow (the window
            # size and the sculpt's L set the cost), so orbiting and strokes keep a usable frame rate
            scale = 2 if self._q0_ms < 45 else (3 if self._q0_ms < 90 else 4)
        sil = view['mode'] == 'silhouette'          # fs-shell 11-14: the CPU's silhouette: its matcap's hits, flattened
        with self.lock:
            res = render(self.hist.scene, view['cam'], view['w'], view['h'], quality=q,
                         mode='matcap' if sil else view['mode'],
                         show_mask=view.get('show_mask', False), scale=scale, upsample=False)
        if sil:
            hit = np.asarray(res.obj) >= 0
            res = res._replace(rgb=np.repeat(np.where(hit, np.uint8(15), np.uint8(219)).astype(np.uint8)[..., None], 3, 2))
        if q == 0:
            self._q0_ms = res.stats['ms'] * (scale / 2.0) ** 2      # the cost at scale 2, so it can step back down
        self.frames += 1
        self.last_render_ms = res.stats['ms']
        final = not interactive
        self._refine_at = None if final else time.perf_counter() + REFINE_DELAY
        self.frameReady.emit({'rgb': res.rgb, 'obj': res.obj, 'depth': res.depth, 'stats': res.stats,
                              'cam': view['cam'], 'w': view['w'], 'h': view['h'], 'seq': view.get('seq', 0),
                              'final': final, 'quality': q})
