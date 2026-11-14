"""Commands, undo/redo, the deterministic op log, project save/load, state hash.

Every model change is a Command: do(scene) / undo(scene) / to_log(). The op log is the list of done commands'
log entries (JSON); replaying it into an empty Scene reproduces the same state hash (deterministic: no random
numbers, no timing, single-writer numba kernels).
"""
import copy
import itertools
import json
import os

import numpy as np

from . import brush as BR
from . import gpu_results as GR
from . import ops
from . import undojournal as UJ
from .scene import Scene, SceneObject, DEFAULT_H

# integration option A (2026-11-08): the app sets GPU_STROKE = fn(scene, cmd, samples) -> bool. A True return means
# the dabs were applied on the GPU atlas and written into the sculpt layer through cmd.rec (gpu/sculpt.py); the
# CPU apply is skipped and the stroke runs no CPU re-distance. None (tests, replays, no card): the CPU path.
GPU_STROKE = None
GPU_PRESS_NO_TOUCH = os.environ.get('FSB_PRESS_TOUCH', '0') == '0'   # fs-brush 11-12: a card stroke's press leaves the scene key
GPU_STROKE_END = None    # fn(scene, cmd): a GPU stroke's end (its write-back into the sculpt layer)
POOL_TRIM = None         # fs-trust 11-11: fn(reason): the card pool's unused blocks back at a safe point (gpu/scene.trim)


def _pool_trim(reason):
    if POOL_TRIM is not None:
        try:
            POOL_TRIM(reason)
        except Exception:
            pass
# r66: the T4 fallback: re-distance a GPU stroke's bricks on the CPU at release (FIELDSTUDIO_GPU_END_REDIST=1)
# r68: ...'force' (the default since 11-09 04:25): without its two skips (the delta's L bound <= 1.3;
# no lower L after it); '0' turns it off
GPU_END_REDIST = os.environ.get('FIELDSTUDIO_GPU_END_REDIST', 'force') == '1'
GPU_END_REDIST_FORCE = os.environ.get('FIELDSTUDIO_GPU_END_REDIST', 'force') == 'force'
GPU_END_REDIST = GPU_END_REDIST or GPU_END_REDIST_FORCE
# 11-09: the GPU release re-distance iterated to convergence (new strokes record 'force_iter')
def _redist_fn(gpu):
    """11-09: brush.redistance's scheme on the card (gpu/redist.redistance_gpu: the same result to well under 1 um,
    deterministic) for strokes recorded 'gpu_iter'; the CPU's where no card is reachable"""
    if gpu:
        try:
            from ..gpu import redist as _GRD
            return _GRD.redistance_gpu
        except Exception:
            pass
    return BR.redistance


REDIST_BAND_BRICKS = 2       # 11-09 ('gpu_iter'): bricks whose centre lies within this many bricks of the surface


def _band_bricks(scene, oid, bricks):
    """11-09: the stroke's bricks whose centre has |F| <= REDIST_BAND_BRICKS bricks (8 h each), by the CPU field at
    the centres (one point per brick; the same choice live, on the card, and in a replay). A large stroke's bricks are
    mostly deep inside or far outside her (30,551 for an Inflate r 30 mm on her, of which the surface holds a few
    thousand): the release re-distance now works where the surface and the brush's strength are; elsewhere the
    stroke's own field stays (its bound still measured), the edge ramp joining the two."""
    if len(bricks) == 0:
        return np.zeros((0, 3), np.int64)
    from . import program as P
    o = scene.get(oid)
    h = float(o.sculpt.h)
    B = np.asarray(bricks, np.int64).reshape(-1, 3)
    X = np.ascontiguousarray((B + 0.5) * 8 * h, np.float64)
    pk = scene.pack()
    f = np.empty(len(X))
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, scene.index_of(oid), X, f)
    keep = np.abs(f) <= REDIST_BAND_BRICKS * 8 * h
    return B[keep]


REDIST_ITER_CAP = 4
OWED_BUDGET = int(os.environ.get('FS_OWED_BUDGET', '1500'))           # r84: owed bricks one release pays (touched first, then the oldest debt)
OWED_CAP = int(os.environ.get('FS_OWED_CAP', '1'))   # r82: owed passes at one release, at most
OWED_LOCK = os.environ.get('FS_OWED_LOCK', '1') == '1'   # owed passes keep sign-changing cells' corners (gpu/redist lock_zero)
OWED_PASS = os.environ.get('FS_OWED_PASS', '1') == '1'   # 11-09: a release pays the debt its band touches (_owed_pass)
REDIST_ITER_TOL = 1e-6        # m: a pass that moves no node by this much ends the iteration
REDIST_ITER_P99 = 1.3         # the band's |grad F| p99 at or under this ends the iteration (the acceptance test)
REDIST_ITER_BUDGET_S = 1.2    # s: no further pass that would end past this (a 30-60 mm stroke's pass costs ~1 s)

REGISTRY = {}
_SERIAL = itertools.count(1)   # hand2b: a monotonic serial per pushed command (the unsaved-changes marker; id() is
                               # reused by CPython after a command is freed: verifier item 24)


def command(cls):
    REGISTRY[cls.__name__] = cls
    return cls


def _json_exact(d):
    """fs-trust: d through JSON, or None when JSON would not give back an equal value (then the step bounds the saved
    undo history instead of coming back different)"""
    try:
        r = json.loads(json.dumps(d))
    except (TypeError, ValueError):
        return None
    return r if r == d else None


class Command:
    label = 'command'
    uid = None               # fs-trust 11-11: the undo journal's id (core/undojournal.py), given at push

    def do(self, scene):
        raise NotImplementedError

    def undo(self, scene):
        raise NotImplementedError

    def to_log(self):
        e = getattr(self, '_entry', None)
        if e is not None:
            return copy.deepcopy(e)          # fs-trust: a step rebuilt at Open keeps its saved log entry as it was
        d = {'type': type(self).__name__}
        d.update(self.args())
        return d

    # ---- fs-trust 11-11: the undo journal (core/undojournal.py) ---------------------------------------------------
    def undo_state(self):
        """-> a function returning (meta dict, arrays dict): everything undo() needs beyond the log entry, or None when
        this command cannot be undone from saved data (it then bounds the undo history an Open rebuilds)"""
        return None

    def restore_undo_state(self, scene, meta, arrays):
        raise NotImplementedError

    def evict_undo(self):
        """drop the undo data from RAM (it is on disk); False when this command cannot"""
        return False

    def args(self):
        return {}

    def nbytes(self):
        return 256

    def discard(self, scene, state):
        """hand2b: the command leaves the history for good. state 'undone': it was undone and is now superseded (a new
        push cleared the redo stack): free what only its redo needed. state 'committed': it was done and the undo cap
        dropped it: free what only its undo needed. Store slots are the only resource held this way."""
        return None

    @staticmethod
    def from_log(d):
        d = dict(d)
        cls = REGISTRY[d.pop('type')]
        if cls is BrushStroke and 'redistance' not in d:
            d['redistance'] = False          # logs written before re-distancing replay exactly as they were
        if cls is BrushStroke and 'dab_gain' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), dab_gain=False)   # r67: logs before the gain replay as they ran
        if cls is BrushStroke and 'sheet_smooth' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), sheet_smooth=False)   # 11-09: logs before it replay as they ran
        if cls is BrushStroke and 'amount_pass' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), amount_pass=False)    # 11-09: the old per-dab amount on replay
        if cls is BrushStroke and 'keep_apart' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), keep_apart=False)     # 11-09: logs before it replay as they ran
        if cls is BrushStroke and d.get('settings', {}).get('tool') == 'swell' and 'swell_toward' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), swell_toward=False)  # sculpt-contact: older Swells replay as inflate
        if cls is BrushStroke and 'smooth_local' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), smooth_local=False)   # 11-09: logs before it replay as they ran
        if cls is BrushStroke and 'dab_limit' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), dab_limit=False)      # 11-10: logs before it replay as they ran
        if cls is BrushStroke and 'cpu_band' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), cpu_band=False)       # fs-brush 11-12: logs before it replay as they ran
        if cls is BrushStroke and 'cpu_band_side' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), cpu_band_side=False)  # fs-brush 11-12: the two-sided band replays as it ran
        if cls is BrushStroke and 'sheet_only' not in d.get('settings', {}):
            d['settings'] = dict(d.get('settings', {}), sheet_only=False)     # fs-brush 11-12: strokes before the sheet gate replay as they ran
        return cls(**d)


@command
class AddObject(Command):
    """spec: {kind, params, name, translate, h, base (tree dict, optional), colour, material}."""
    label = 'Add object'

    def __init__(self, spec):
        self.spec = copy.deepcopy(spec)
        self.obj = None

    def args(self):
        return {'spec': self.spec}

    def do(self, scene):
        self.prev_next = scene.next_id
        if self.obj is None:
            sp = self.spec
            oid = scene.new_id()
            if sp.get('base'):
                base = ops.Node.from_dict(sp['base'])
                scene.next_id = max(scene.next_id, max(n.id for n in base.walk()) + 1)
            else:
                base = ops.Node(sp.get('kind', 'sphere'), sp.get('params') or {}, nid=scene.new_id())
            o = SceneObject(oid, sp.get('name') or '%s.%d' % (base.kind, oid), base,
                            transform={'translate': [float(v) for v in sp.get('translate', (0, 0, 0))],
                                       'rotate': [float(v) for v in sp.get('rotate', (0, 0, 0))],
                                       'scale': float(sp.get('scale', 1.0))},
                            colour=copy.deepcopy(sp.get('colour')), material=copy.deepcopy(sp.get('material')),
                            h=float(sp.get('h', DEFAULT_H)))
            if sp.get('part'):                     # fs-contact 11-11: a store part (core/fv8parts.py)
                o.part = copy.deepcopy(sp['part'])
            if sp.get('garment'):                  # fs-shell 11-11: a garment's record (fs-cloth: a later Re-fit)
                o.garment = copy.deepcopy(sp['garment'])
            self.obj = o
            self.after_next = scene.next_id
        else:
            scene.next_id = self.after_next
        scene.objects.append(self.obj)
        scene.touch(structural=True, index=True)
        return self.obj

    def undo(self, scene):
        scene.objects.remove(self.obj)
        scene.next_id = self.prev_next
        # fs-mouth 11-12: an object out of the scene keeps no compiled program (its float64 copy of a lattice base, ~0.9
        # GB for the mouth set, stayed alive in the redo stack); a redo recompiles it from its base
        self.obj.invalidate()
        scene.touch(structural=True, index=True)

    def undo_state(self):
        if self.obj is None:
            return None
        st = {'oid': int(self.obj.id), 'prev_next': int(self.prev_next), 'after_next': int(self.after_next)}
        return lambda: (st, {})

    def restore_undo_state(self, scene, meta, arrays):
        self.obj = scene.get(meta['oid'])          # the object as it stands once every later step is undone
        self.prev_next, self.after_next = meta['prev_next'], meta['after_next']


@command
class RemoveObject(Command):
    label = 'Delete object'

    def __init__(self, oid):
        self.oid = int(oid)

    def args(self):
        return {'oid': self.oid}

    def do(self, scene):
        self.pos = scene.index_of(self.oid)
        self.obj = scene.objects.pop(self.pos)
        self.obj.invalidate()                      # fs-mouth 11-12: kept for its undo, not its compiled program
        scene.touch(structural=True, index=True)

    def undo(self, scene):
        scene.objects.insert(self.pos, self.obj)
        scene.touch(structural=True, index=True)

    def nbytes(self):
        o = getattr(self, 'obj', None)
        return 256 + (len(o.sculpt.slots) * 2 * 4 * 512 if o is not None else 0)   # its sculpt stays allocated

    def discard(self, scene, state):
        o = getattr(self, 'obj', None)
        if state == 'committed' and o is not None:     # it can never come back: free its sculpt bricks
            release_object_slots(scene, o)


_SNAP_B = 2 * 8 * 8 * 8 * 4          # one brick's delta + mask, float32 (a StrokeRecord snapshot, a kept redo state)


def _rec_bytes(r):
    """brush.StrokeRecord.nbytes in O(1): its snapshots are 8^3 float32 delta + mask pairs"""
    if r is None:
        return 0
    return len(r.snap) * _SNAP_B + 64 * (len(r.new) + len(r.Lsnap))


def _ensure_bricks(scene, c):
    """fs-trust: a step rebuilt at Open restores bricks the saved project did not hold (a save keeps non-zero bricks
    only): those are allocated, zero, before its undo writes into them (an all-zero brick is outside the state hash)"""
    st = scene.store
    need = {}
    for r in (getattr(c, 'rec', None), getattr(c, 'rd', None)):
        if r is None:
            continue
        for (oid, b) in UJ.record_keys(r):
            if scene.index_of(oid) >= 0 and scene.get(oid).sculpt.get_slot(b) < 0:
                need.setdefault(oid, set()).add(b)
    for oid, bs in need.items():
        bl = sorted(bs)
        sl = st.alloc_many(len(bl))
        st.delta[sl] = 0.0
        st.mask[sl] = 0.0
        st.mdir[sl] = 0
        st.Lb[sl] = 0.0
        scene.get(oid).sculpt.set_slots(bl, sl)
    if need:
        scene.touch(index=True)
    # bricks whose bound no record restores get one measured after the undo (History.undo)
    have_L = set()
    for r in (getattr(c, 'rec', None), getattr(c, 'rd', None)):
        if r is not None:
            have_L |= set(r.snap) | set(r.Lsnap)
    return {oid: {b for b in bs if (oid, b) not in have_L} for oid, bs in need.items()}


def release_object_slots(scene, o):
    """hand2b: return every store slot of an object that has left the history for good (verifier item 16)"""
    st = scene.store
    for b, s_ in list(o.sculpt.slots.items()):
        if s_ >= 0:
            st.release(s_)
        o.sculpt.set_slot(b, -1)


@command
class SetBase(Command):
    """Replace an object's whole base tree (generic undoable base edit: wrap, boolean, delete node, ...)."""
    label = 'Edit base'

    def __init__(self, oid, tree, next_id=None):
        self.oid = int(oid)
        self.tree = copy.deepcopy(tree)
        self.next_id = next_id

    def args(self):
        return {'oid': self.oid, 'tree': self.tree, 'next_id': self.next_id}

    def do(self, scene):
        o = scene.get(self.oid)
        self.old = o.base.to_dict()
        self.prev_next = scene.next_id
        o.base = ops.Node.from_dict(self.tree)
        o.compiled()                       # validate now: raises before anything is half-applied
        if self.next_id is not None:
            scene.next_id = max(scene.next_id, int(self.next_id))
        o.invalidate()
        scene.touch(structural=True)

    def undo(self, scene):
        o = scene.get(self.oid)
        o.base = ops.Node.from_dict(self.old)
        o.invalidate()
        scene.next_id = self.prev_next
        scene.touch(structural=True)

    def undo_state(self):
        st = _json_exact({'old': self.old, 'prev_next': self.prev_next})
        return None if st is None else (lambda: (st, {}))

    def restore_undo_state(self, scene, meta, arrays):
        self.old, self.prev_next = meta['old'], meta['prev_next']


@command
class SetNodeParams(Command):
    label = 'Edit node'

    def __init__(self, oid, nid, params):
        self.oid = int(oid)
        self.nid = int(nid)
        self.params = copy.deepcopy(params)

    def args(self):
        return {'oid': self.oid, 'nid': self.nid, 'params': self.params}

    def do(self, scene):
        o = scene.get(self.oid)
        n = o.base.find(self.nid)
        if n is None:
            raise KeyError('node %d' % self.nid)
        self.old = {k: copy.deepcopy(n.params[k]) for k in self.params}
        for k, v in self.params.items():
            if k not in n.params:
                raise KeyError(k)
            n.params[k] = copy.deepcopy(v)
        o.invalidate()
        scene.touch(structural=True)

    def undo(self, scene):
        o = scene.get(self.oid)
        n = o.base.find(self.nid)
        n.params.update(copy.deepcopy(self.old))
        o.invalidate()
        scene.touch(structural=True)

    def undo_state(self):
        st = _json_exact({'old': self.old})
        return None if st is None else (lambda: (st, {}))

    def restore_undo_state(self, scene, meta, arrays):
        self.old = meta['old']


@command
class SetObjectProps(Command):
    """Change transform / colour / material / detail / name / visible (dict of top-level keys)."""
    label = 'Object properties'
    KEYS = ('transform', 'colour', 'material', 'detail', 'name', 'visible', 'locked', 'root')

    def __init__(self, oid, props):
        self.oid = int(oid)
        self.props = copy.deepcopy(props)

    def args(self):
        return {'oid': self.oid, 'props': self.props}

    def do(self, scene):
        o = scene.get(self.oid)
        self.old = {}
        for k, v in self.props.items():
            if k not in self.KEYS:
                raise KeyError(k)
            self.old[k] = copy.deepcopy(getattr(o, k))
            if isinstance(v, dict) and isinstance(getattr(o, k), dict):
                nv = copy.deepcopy(getattr(o, k))
                nv.update(copy.deepcopy(v))
                setattr(o, k, nv)
            else:
                setattr(o, k, copy.deepcopy(v))
        scene.touch(structural=True)

    def undo(self, scene):
        o = scene.get(self.oid)
        for k, v in self.old.items():
            setattr(o, k, copy.deepcopy(v))
        scene.touch(structural=True)

    def undo_state(self):
        st = _json_exact({'old': self.old})
        return None if st is None else (lambda: (st, {}))

    def restore_undo_state(self, scene, meta, arrays):
        self.old = meta['old']


SHEET_NEW = os.environ.get('FSB_SHEET', '1') != '0'   # fs-brush 11-12: the sheet gate for new strokes without the panel's key (on by default; FSB_SHEET=0 off)


@command
class BrushStroke(Command):
    """One stroke = its settings + its samples (world space). Undo restores the touched bricks exactly.

    redistance (default True for new strokes): when the stroke ends, the bricks it touched are re-distanced
    (brush.redistance) so |grad F| ~ 1 again; that is part of this same command and of its undo. A live drag
    (History.extend_stroke) first undoes the last re-distance, adds the dabs, and re-distances again at
    History.end_stroke (mouse release) - the same result as doing all the dabs at once and re-distancing once."""
    label = 'Brush stroke'

    def __init__(self, oid, settings, samples, redistance=True, gpu_result=None):
        self.gpu_result = gpu_result         # r42: a GPU stroke's recorded result (core/gpu_results.py)
        self.oid = int(oid)
        s = copy.deepcopy(BR.DEFAULT_SETTINGS)
        s.update(copy.deepcopy(settings))
        s.setdefault('dab_gain', True)       # r67: new strokes carry the small-dab gain (from_log marks older ones off)
        # 11-09 night: the sheet stencil (brush.py _sheet_avg) left thin folds granular; superseded by keep_apart
        s.setdefault('sheet_smooth', False)
        s.setdefault('keep_apart', True)     # 11-09 night: surfaces never joined, filmed over or torn (gpu/sculpt.cu)
        s.setdefault('smooth_local', True)   # 11-09 night: Smooth averages at 2 h, not r / 4 (brush.py smooth_step)
        s.setdefault('dab_limit', True)      # 11-10: Clay / Flatten move a node by at most the dab's amount per dab
        s.setdefault('amount_pass', True)    # 11-09 night: the amount per pass, not per dab (brush.py dab_amount)
        s.setdefault('cpu_band', True)       # fs-brush 11-12 (F3): CPU dabs of the skip tools work the surface band only
        s.setdefault('cpu_band_side', True)  # fs-brush 11-12: ... skipping only the side the surface moves away from
        s.setdefault('sheet_only', SHEET_NEW)  # fs-brush 11-12: the sheet under the brush (core/brush.py SHEET_TOOLS)
        if s.get('tool') == 'swell':         # sculpt-contact 11-12: new Swell strokes are Swell to contact (collide.swell_ease)
            from . import collide as _CLs
            s.setdefault('swell_toward', _CLs.SWELL_TOWARD)
            s.setdefault('swell_mode', _CLs.SWELL_MODE)
        self.settings = s
        self.samples = copy.deepcopy(samples)
        self.rec = None
        self.redistance = bool(redistance)
        self.rd = None                       # undo record of the re-distance
        self.rd_info = None
        self.pending = False                 # dabs added since the last re-distance (a live drag)

    def args(self):
        d = {'oid': self.oid, 'settings': self.settings, 'samples': self.samples}
        if getattr(self, 'gpu_result', None):
            d['gpu_result'] = self.gpu_result
        if getattr(self, 'redistance', False):
            d['redistance'] = True
        return d

    def _redistance(self, scene):
        self.pending = False
        if not getattr(self, 'redistance', False) or self.settings.get('tool') == 'mask' or self.rec is None:
            return
        keys = list(self.rec.snap.keys()) + list(self.rec.new)
        # hand2f: a world-symmetry stroke also writes the mirrored object(s): each is re-distanced (stroke object first)
        oids = [self.oid] + sorted({k[0] for k in keys} - {self.oid})
        self.rd = BR.StrokeRecord()
        if getattr(self, '_gpu_rd', False):
            self._gpu_release(scene, oids, keys)
            return
        for oid in oids:
            if scene.index_of(oid) < 0:
                continue
            o = scene.get(oid)
            if any(n.kind == 'sampled' for n in o.base.walk()):
                # V1.2 field region: the stored field is the authority; re-distancing would write the store's own
                # (not exactly distance) field into the delta over every touched brick, outside the brush's support
                continue
            bricks = [b for (k, b) in keys if k == oid]
            if not bricks:
                continue
            # 11-09 (design: a converged release, once per stroke): 'force_iter' repeats the pass on the stroke's
            # bricks (and those the passes touched) until the band's |grad F| is near 1 (brush.redistance skips: p1 >=
            # 0.8 and p99 <= 1.3) or a pass moves no node by 1 um, at most REDIST_ITER_CAP passes, all in one record
            npass = REDIST_ITER_CAP if getattr(self, '_iter_rd', False) else 1
            # the passes a stroke ran are recorded (its time budget depends on the machine's load): a replay runs
            # exactly that many, without the clock
            fixed = (self.settings.get('gpu_end_redist_passes') or {}).get(str(oid)) if getattr(self, '_iter_rd', False) else None
            if fixed is not None:
                npass = int(fixed)
            st = scene.store
            g = o.sculpt
            info = None
            done = 0
            from ..gpu import redist as _GV         # numpy-only helpers (no card needed)
            # 11-09: the bricks as one sorted key array (sorted() over tuples, vectorized: the per-brick Python cost
            # ~0.3 s of a large stroke's release)
            cur = _GV.unique_keys(np.array(list(bricks) + [b for (k, b) in list(self.rd.snap.keys()) + list(self.rd.new)
                                                           if k == oid], np.int64).reshape(-1, 3))
            if getattr(self, '_gpu_rd', False):
                cur = _band_bricks(scene, oid, cur)         # 11-09: the release re-distances near the surface only
            _dbg = []
            first_cur = cur
            _t_rd = __import__('time').perf_counter()
            for it in range(npass):
                _tp = __import__('time').perf_counter()
                bsl = _GV.slots_of(g, cur)
                has = bsl >= 0
                before = st.delta[bsl[has]]
                cur_l = [tuple(b) for b in cur.tolist()]
                _fn = _redist_fn(getattr(self, '_gpu_rd', False))
                _kw = {'reuse_prev': it > 0} if _fn is not BR.redistance else {}
                inf = _fn(scene, oid, cur_l, self.rd, src=self.rec, force=bool(getattr(self, '_force_rd', False)), **_kw)
                info = inf if info is None else info
                done += 1
                _dbg.append((len(cur), round((__import__('time').perf_counter() - _tp) * 1000), BR.redistance.last_grad[1]))
                if BR.redistance.last_grad[2]:          # converged before this pass: it changed nothing
                    break
                _el = __import__('time').perf_counter() - _t_rd
                if fixed is None and _el + _el / done > REDIST_ITER_BUDGET_S:
                    break                                # another pass would end past the release's time budget
                # 11-09 (in the app a Clay 30 / Grab 60 release took 3.3-5.2 s with every pass over all the stroke's
                # bricks): the next pass covers only the bricks this one still moved by REDIST_ITER_TOL, and their
                # neighbours (a node's change reaches the neighbours' aprons)
                sl2 = _GV.slots_of(g, cur[has])
                ok2 = sl2 >= 0
                mv = np.zeros(len(sl2), bool)
                if ok2.any():
                    mv[ok2] = (np.abs(st.delta[sl2[ok2]] - before[ok2]).reshape(int(ok2.sum()), -1).max(1)
                               >= REDIST_ITER_TOL)
                moving = cur[has][mv]
                if not len(moving):
                    break
                nb_, nsl_ = _GV.near_keys(g, moving)
                # those of the first pass's bricks (sorted, as before)
                cur = nb_[np.isin(_GV.encode_keys(nb_), _GV.encode_keys(first_cur))]
            if getattr(self, '_gpu_rd', False):
                _GV.forget()
            self.rd_passes = done
            if getattr(self, '_iter_rd', False):
                rp = dict(self.settings.get('gpu_end_redist_passes') or {})
                rp[str(oid)] = done
                self.settings['gpu_end_redist_passes'] = rp
            self.rd_dbg = _dbg
            if os.environ.get('FS_DBG_REDIST'):
                print('REDIST', self.settings.get('tool'), 'r', self.settings.get('radius'), 'passes (bricks, ms, p99 before)', _dbg, flush=True)
            if oid == self.oid:
                self.rd_info = info
            if self.settings.get('collide', False) or self.settings.get('tool') in ('swell', 'settle'):
                # 11-09: the collider rule again after the re-distance (it moved clamped nodes back by tens of um)
                from . import collide as CL
                rb = [b for (k, b) in list(self.rd.snap.keys()) + list(self.rd.new) if k == oid]
                CL.clamp_bricks(scene, oid, sorted(set(bricks) | set(rb)), self.rec, dict(self.settings, collide=True))
            # fs-brush 11-12 (the sheet under the brush; thinwall/sf5_lid: with the gate the lid's underside and the eye
            # moved 0 um without this re-distance, 16 / 15 um with it): the release re-distance rewrote nodes beside the
            # other surfaces across the gap. Every node the stroke's reach saw that is not on the worked sheet gets its
            # value back from before this re-distance (its own surface is exactly as before the stroke).
            shl = getattr(self.rec, 'sheet', None)
            if shl is not None:
                # every re-distanced node NOT on the worked sheet: the other surfaces' nodes and those beyond the reach
                # (grow_band's bricks), which no dab of this stroke changed (lid case: 1.8 / 3.5 um left otherwise)
                g_ = scene.get(oid).sculpt
                st_ = scene.store
                for (o_, b_), sn_ in list(self.rd.snap.items()):
                    if o_ != oid:
                        continue
                    lab_ = shl.get((oid, int(b_[0]), int(b_[1]), int(b_[2])))
                    keep_ = np.ones((8, 8, 8), bool) if lab_ is None else ((lab_ & 3) != 1)
                    if not keep_.any():
                        continue
                    sl_ = g_.get_slot(b_)
                    if sl_ < 0:
                        continue
                    d_ = st_.delta[sl_]
                    d_[keep_] = sn_[0][keep_]
                scene.refresh_object(scene.index_of(oid))

    # ---- 11-09 (decided): 'gpu_iter' re-distances once at the release, the further passes in idle time
    def _gpu_release(self, scene, oids, keys):
        """The release: one pass on the card per object; further passes run in idle time (idle_redist_step), each into
        this stroke's record self.rd, until converged, nothing moves, or REDIST_ITER_CAP. The passes an object got are
        recorded in settings['gpu_end_redist_passes']; a replay (self._replaying) runs exactly that many, in the live
        order (the first pass of every object, then each object's further passes), each followed by the collider when
        it is on, as live: the same state whatever the timing was"""
        fixed = dict(self.settings.get('gpu_end_redist_passes') or {}) if getattr(self, '_replaying', False) else None
        self._rdst = {}
        self._rdscene = scene
        self._debt_added, self._debt_taken = [], []
        for oid in oids:
            if scene.index_of(oid) < 0:
                continue
            o = scene.get(oid)
            if any(n.kind == 'sampled' for n in o.base.walk()):
                continue                         # V1.2 field region: the stored field is the authority (as above)
            bricks = [b for (k, b) in keys if k == oid]
            if not bricks:
                continue
            from ..gpu import redist as _GV
            cur = _band_bricks(scene, oid, _GV.unique_keys(np.array(bricks, np.int64).reshape(-1, 3)))
            self._rdst[oid] = {'cur': cur, 'first': cur, 'done': 0, 'more': len(cur) > 0, 'cache': None,
                               'bricks': bricks, 'dbg': [], 'info': None}
            if not len(cur):
                self._rd_count(oid)
        for oid in list(self._rdst):
            self._gpu_rd_pass(scene, oid)
            if OWED_PASS:
                self._owed_pass(scene, oid)
        if fixed is not None:
            for oid in list(self._rdst):
                want = int(fixed.get(str(oid), 1))
                while self._rdst[oid]['more'] and self._rdst[oid]['done'] < want:
                    self._gpu_rd_pass(scene, oid)
            self.idle_redist_stop()
            for _ in range(int(self.settings.get('gpu_end_redist_debtpay', 0))):
                self._debt_pay_one(scene)              # r87: the idle-time debt payments it made, in order
        else:
            self._debt_open = True                     # r87: once its own passes are done, idle time pays older debt
        S = self._rdst.get(self.oid) if self._rdst else None
        if S is not None:
            self.rd_info = S['info']
            self.rd_passes = S['done']
            self.rd_dbg = S['dbg']
        if not self.idle_redist_pending():
            self._rdst = None

    def _owed_pass(self, scene, oid):
        """11-09 (strokes in quick bursts get no idle time): the bricks earlier strokes still owed passes on
        when the next command stopped them (scene._rd_debt) and that this stroke's band touches get one pass now, at
        the release, into this stroke's record. brush.redistance's own |grad F| test (p1 >= 0.8 and p99 <= 1.3 near
        the surface) skips it where the field is still healthy. The debt is a
        function of the recorded passes alone, so a replay takes the same decision."""
        S = self._rdst[oid]
        debt = [e for e in getattr(scene, '_rd_debt', []) if e['oid'] == oid and len(e['enc'])]
        if not debt or not len(S['first']):
            return
        from ..gpu import redist as _GV
        mine = _GV.encode_keys(S['first'])
        hit = [np.intersect1d(e['enc'], mine) for e in debt]
        # r87 (I2b: the bricks one release pays <= OWED_BUDGET): the touched debt too, oldest entries first
        cap_ = OWED_BUDGET
        for i_ in range(len(hit)):
            if len(hit[i_]) > cap_:
                hit[i_] = hit[i_][:max(cap_, 0)]
            cap_ -= len(hit[i_])
        # r84 (back to back on v0013b, D1a still 3.31: debt no later band touches was never paid): then the other debt,
        # oldest first, up to OWED_BUDGET bricks in all (a brick count, not a clock: a replay takes the same bricks)
        left = OWED_BUDGET - int(sum(len(x) for x in hit))
        for i_, e in enumerate(debt):
            if left <= 0:
                break
            rest = np.setdiff1d(e['enc'], hit[i_])
            if len(rest):
                take = rest[:left]
                hit[i_] = np.union1d(hit[i_], take)
                left -= len(take)
        D = np.unique(np.concatenate(hit)) if hit else np.zeros(0, np.int64)
        if not len(D):
            return
        for e, hk in zip(debt, hit):
            if len(hk):
                e['enc'] = np.setdiff1d(e['enc'], hk)
                self._debt_taken.append((e, hk))
        keys = np.stack([(D >> 42) - (1 << 20), ((D >> 21) & ((1 << 21) - 1)) - (1 << 20),
                         (D & ((1 << 21) - 1)) - (1 << 20)], 1)
        import time as _time
        _tp = _time.perf_counter()
        fn = _redist_fn(True)
        # r82 (design: the band healthy before the release returns): passes over the owed bricks until brush's
        # |grad F| test passes there (p1 >= 0.8 and p99 <= 1.3 near the surface: the pass is skipped, the field read
        # only), at most OWED_CAP; forced past the bound skip (the bricks' Lipschitz bound is no sign of a healthy band)
        lk = [tuple(b) for b in keys.tolist()]
        npay = 0
        for _ in range(OWED_CAP):
            kw = {'reuse_prev': S['cache']} if fn is not BR.redistance and S['cache'] is not None else {}
            if fn is not BR.redistance:
                kw['lock_zero'] = OWED_LOCK          # I2b: paying debt changes band distances only, never the surface
            fn(scene, oid, lk, self.rd, src=self.rec, force=True, **kw)
            S['cache'] = getattr(_GV._K, 'prev', None) if fn is not BR.redistance else None
            _GV._K.prev = None
            if BR.redistance.last_grad[2]:
                break
            npay += 1
        S['dbg'].append(('owed', len(keys), npay, round((_time.perf_counter() - _tp) * 1000), BR.redistance.last_grad[1]))
        self.rd_owed = (len(keys), npay)
        if self.settings.get('collide', False) or self.settings.get('tool') in ('swell', 'settle'):
            from . import collide as CL
            rb = [b for (k, b) in list(self.rd.snap.keys()) + list(self.rd.new) if k == oid]
            CL.clamp_bricks(scene, oid, sorted(set(S['bricks']) | set(rb)), self.rec, dict(self.settings, collide=True))
            S['cache'] = None
        scene.touch()

    def _debt_undo(self):
        """undo: the debt this stroke left goes, the debt it paid comes back (the state is as before it)"""
        sc = getattr(self, '_rdscene', None)
        if sc is None:
            return
        added = {id(e) for e in getattr(self, '_debt_added', [])}
        if added and hasattr(sc, '_rd_debt'):
            sc._rd_debt = [x for x in sc._rd_debt if id(x) not in added]     # by identity (entries hold arrays)
        for e, hk in reversed(getattr(self, '_debt_taken', [])):
            e['enc'] = np.union1d(e['enc'], hk)

    def _debt_redo(self):
        sc = getattr(self, '_rdscene', None)
        if sc is None:
            return
        for e, hk in getattr(self, '_debt_taken', []):
            e['enc'] = np.setdiff1d(e['enc'], hk)
        if not hasattr(sc, '_rd_debt'):
            sc._rd_debt = []
        for e in getattr(self, '_debt_added', []):
            sc._rd_debt.append(e)

    def _rd_count(self, oid):
        rp = dict(self.settings.get('gpu_end_redist_passes') or {})
        rp[str(oid)] = self._rdst[oid]['done']
        self.settings['gpu_end_redist_passes'] = rp

    def _gpu_rd_pass(self, scene, oid):
        """one pass of oid's release re-distance; -> True when another is wanted"""
        S = self._rdst[oid]
        if not S['more']:
            return False
        import time as _time
        from ..gpu import redist as _GV
        st = scene.store
        g = scene.get(oid).sculpt
        cur = S['cur']
        _tp = _time.perf_counter()
        bsl = _GV.slots_of(g, cur)
        has = bsl >= 0
        before = st.delta[bsl[has]]
        fn = _redist_fn(True)
        kw = {'reuse_prev': S['cache'] if S['done'] > 0 else None} if fn is not BR.redistance else {}
        S['cache'] = None
        inf = fn(scene, oid, [tuple(b) for b in cur.tolist()], self.rd, src=self.rec, force=True, **kw)
        # the field this pass left on the card (F + its change), held by this stroke for its next pass only: whether a
        # pass reuses it depends on this stroke's own passes alone, live and in a replay
        S['cache'], _GV._K.prev = getattr(_GV._K, 'prev', None), None
        S['info'] = inf if S['info'] is None else S['info']
        S['done'] += 1
        S['dbg'].append((len(cur), round((_time.perf_counter() - _tp) * 1000), BR.redistance.last_grad[1]))
        self._rd_count(oid)
        if os.environ.get('FS_DBG_REDIST'):
            print('REDIST', self.settings.get('tool'), 'oid', oid, 'pass', S['done'], '(bricks, ms, p99 before)',
                  S['dbg'][-1], flush=True)
        if self.settings.get('collide', False) or self.settings.get('tool') in ('swell', 'settle'):
            # the collider rule again after each pass (a re-distance moves clamped nodes back by tens of um); the field
            # it changes is not the cached one: the next pass evaluates afresh
            from . import collide as CL
            rb = [b for (k, b) in list(self.rd.snap.keys()) + list(self.rd.new) if k == oid]
            CL.clamp_bricks(scene, oid, sorted(set(S['bricks']) | set(rb)), self.rec, dict(self.settings, collide=True))
            S['cache'] = None
        more = not BR.redistance.last_grad[2] and S['done'] < REDIST_ITER_CAP
        if more:
            # the next pass covers the bricks this one still moved by REDIST_ITER_TOL and their neighbours, among the
            # first pass's bricks
            sl2 = _GV.slots_of(g, cur[has])
            ok2 = sl2 >= 0
            mv = np.zeros(len(sl2), bool)
            if ok2.any():
                mv[ok2] = (np.abs(st.delta[sl2[ok2]] - before[ok2]).reshape(int(ok2.sum()), -1).max(1)
                           >= REDIST_ITER_TOL)
            moving = cur[has][mv]
            if len(moving):
                nb_, _nsl = _GV.near_keys(g, moving)
                S['cur'] = nb_[np.isin(_GV.encode_keys(nb_), _GV.encode_keys(S['first']))]
                more = len(S['cur']) > 0
            else:
                more = False
        S['more'] = more
        if not more:
            S['cache'] = None
        scene.touch()
        return more

    # ---- r87 (11-09 ~12:50): idle time pays the debt the releases left (oldest first, OWED_BUDGET bricks per
    # payment, lock_zero), into this stroke's record while it is the last command; a replay makes the same payments
    def idle_debt_pending(self):
        sc = getattr(self, '_rdscene', None)
        return (bool(getattr(self, '_debt_open', False)) and not self.idle_redist_pending() and sc is not None
                and any(len(e['enc']) for e in getattr(sc, '_rd_debt', [])))

    def idle_debt_close(self):
        self._debt_open = False

    def idle_debt_step(self, scene):
        if not self.idle_debt_pending():
            return False
        self._debt_pay_one(scene)
        self.settings['gpu_end_redist_debtpay'] = int(self.settings.get('gpu_end_redist_debtpay', 0)) + 1
        return self.idle_debt_pending()

    def _debt_pay_one(self, scene):
        """one payment: the oldest debt of one object, up to OWED_BUDGET bricks, OWED_CAP passes, the zero set locked"""
        debt = [e for e in getattr(scene, '_rd_debt', []) if len(e['enc'])]
        if not debt:
            return 0
        oid = debt[0]['oid']
        if scene.index_of(oid) < 0:
            for e in debt:
                if e['oid'] == oid:
                    self._debt_taken.append((e, e['enc'].copy())); e['enc'] = e['enc'][:0]
            return 0
        left = OWED_BUDGET; take = []
        for e in debt:
            if e['oid'] != oid or left <= 0:
                continue
            t_ = e['enc'][:left]
            e['enc'] = e['enc'][len(t_):]
            self._debt_taken.append((e, t_))
            take.append(t_); left -= len(t_)
        D = np.unique(np.concatenate(take))
        keys = np.stack([(D >> 42) - (1 << 20), ((D >> 21) & ((1 << 21) - 1)) - (1 << 20),
                         (D & ((1 << 21) - 1)) - (1 << 20)], 1)
        if self.rd is None:
            self.rd = BR.StrokeRecord()
        fn = _redist_fn(True)
        lk = [tuple(b) for b in keys.tolist()]
        for _ in range(OWED_CAP):
            kw = {'lock_zero': OWED_LOCK} if fn is not BR.redistance else {}
            fn(scene, oid, lk, self.rd, src=self.rec, force=True, **kw)
            from ..gpu import redist as _GV
            _GV._K.prev = None
            if BR.redistance.last_grad[2]:
                break
        scene.touch()
        return len(lk)

    def idle_redist_pending(self):
        return bool(getattr(self, '_rdst', None)) and any(S['more'] for S in self._rdst.values())

    def idle_redist_step(self, scene):
        """11-09: one further release pass in idle time (the first object still wanting one). -> True if more remain"""
        if not self.idle_redist_pending():
            self._rdst = None
            return False
        for oid, S in self._rdst.items():
            if S['more']:
                self._gpu_rd_pass(scene, oid)
                if oid == self.oid:
                    self.rd_passes = S['done']
                break
        if not self.idle_redist_pending():
            self.idle_redist_stop()
            return False
        return True

    def idle_redist_stop(self):
        """the next command, an undo or a save comes first: no further passes (the recorded count is what ran); the
        bricks the next pass would have covered become debt (_owed_pass)"""
        if getattr(self, '_rdst', None):
            sc = getattr(self, '_rdscene', None)
            from ..gpu import redist as _GV
            for oid, S in self._rdst.items():
                if S['more'] and len(S['cur']) and sc is not None and OWED_PASS:
                    if not hasattr(sc, '_rd_debt'):
                        sc._rd_debt = []
                    e = {'oid': oid, 'enc': np.unique(_GV.encode_keys(S['cur']))}
                    sc._rd_debt.append(e)
                    self._debt_added.append(e)
                S['more'] = False
                S['cache'] = None
        self._rdst = None

    def do(self, scene):
        if getattr(self, '_post', None) is not None:
            self._redo_fast(scene)
            return
        self.rec = BR.StrokeRecord()
        res = GR.get(self.gpu_result) if getattr(self, 'gpu_result', None) else None
        if res is not None:
            # r42: a GPU stroke replayed from the log: its recorded result, exactly (the CPU brushes would give it
            # only to a few um, and the log would no longer replay to the same hash)
            apply_gpu_result(scene, res, self.rec)
            self.redistance = False
            self.pending = False
            if self.settings.get('gpu_end_redist'):
                self._replaying = True                # 11-09: the recorded passes, all now (idle time included)
                try:
                    self.cpu_redist_after_gpu(scene)      # r66: replayed as it ran (CPU, deterministic)
                finally:
                    self._replaying = False
            scene.touch()
            return
        if GPU_STROKE is not None and GPU_STROKE(scene, self, self.samples):
            self.redistance = False
            self.on_gpu = True                  # the whole stroke stays on one path (the card runs the flow Grab itself)
        elif self._flow_on():
            self._flow_c0 = self._flow_v = None
            self._snake_all = []                    # fs-brush 11-12: the composed Snake starts over on a redo
            self._flow_apply(scene, self.samples)       # fs-brush 11-12: the anchored flow Grab (replays as it ran)
        else:
            BR.apply_stroke(scene, self.oid, self.settings, self.samples, self.rec)
        if getattr(self, 'live', False):
            # hand1: a live drag's first push re-distances only on release (finish): the next move undid it anyway
            # (extend), so the end state is the same and the press no longer waits 0.2-7.6 s (M, lag diagnosis)
            self.live = False
            self.pending = bool(getattr(self, 'redistance', False))
        else:
            self._redistance(scene)
        if not (getattr(self, 'on_gpu', False) and GPU_PRESS_NO_TOUCH):
            scene.touch()
        # fs-brush 11-12 (brush-study STUDY 1e: this touch made the stroke's first batch run the full mirror sync, ~280 ms
        # at 226 k slots, in the press's first frame): a card stroke changes nothing on the CPU until its release, which
        # touches the scene itself; FSB_PRESS_TOUCH=1 touches here as before

    def _flow_on(self):
        """fs-brush 11-12 (toolkit review 4.2): a Grab with settings['grab_flow'] on the CPU path is ANCHORED: every
        batch restores the stroke-start field and applies one flow warp with the total drag (core/brush.py _flow_back)"""
        t_ = self.settings.get('tool')
        on = ((t_ == 'grab' and bool(self.settings.get('grab_flow', False))) or
              (t_ == 'stretch' and bool(self.settings.get('stretch_flow', False))) or
              (t_ == 'grab' and bool(self.settings.get('snake', False)) and BR.SNAKE_PATH and BR.SNAKE_COMPOSE))
        return on and not getattr(self, 'on_gpu', False)

    def _flow_apply(self, scene, samples):
        if self.settings.get('snake', False):
            return self._snake_apply(scene, samples)
        tot = getattr(self, '_flow_v', None)
        for smp in samples:
            if getattr(self, '_flow_c0', None) is None:
                self._flow_c0 = np.asarray(smp['p'], float)
                self._flow_n0 = list(smp.get('n', (0.0, 1.0, 0.0)))
                tot = np.zeros(3)
            tot = tot + np.asarray(smp.get('v', (0.0, 0.0, 0.0)), float)
        if tot is None:
            return
        self._flow_v = tot
        if self.rec is not None and (self.rec.snap or self.rec.new or self.rec.Lsnap):
            BR.undo_record(scene, self.rec)
        self.rec = BR.StrokeRecord()
        got = BR.anchored_sample(scene, self.oid, self.settings, self._flow_c0, self._flow_n0, tot)
        if got is None:
            return
        BR.apply_stroke(scene, self.oid, got[0], [got[1]], self.rec)

    def _snake_apply(self, scene, samples):
        """fs-brush 11-12: Snake Hook composed on the CPU path: the stroke-start field back, then the whole path so far
        (every sample's step, per mirror copy) applied as one composed warp per copy (brush.snake_path_back)"""
        self._snake_all = list(getattr(self, '_snake_all', [])) + list(samples)
        if self.rec is not None and (self.rec.snap or self.rec.new or self.rec.Lsnap):
            BR.undo_record(scene, self.rec)
        self.rec = BR.StrokeRecord()
        oi = scene.index_of(self.oid)
        obj = scene.objects[oi]
        from ..tools.protected_merge import _to_world
        s_ = max(float(obj.transform['scale']), 1e-6)
        for steps in BR.snake_path_steps(obj, self._snake_all, self.settings):
            if not len(steps):
                continue
            c_, r_ = BR.snake_path_ball(steps)
            p_w = _to_world(obj, np.asarray(c_, float)[None, :])[0]
            st = dict(self.settings, tool='snakepath', radius=r_ * s_, snake_steps_local=steps, autosmooth=0.0,
                      symmetry=[False, False, False], snake=False, sym_world=False)
            BR.apply_stroke(scene, self.oid, st, [{'p': np.asarray(p_w).tolist(), 'n': [0.0, 0.0, 1.0],
                                                    'v': [0.0, 0.0, 0.0]}], self.rec)

    def extend(self, scene, samples):
        """Live stroke: apply more dabs to this (already done) stroke. Same result as doing all at once."""
        samples = copy.deepcopy(samples)
        if self.rd is not None:
            BR.undo_record(scene, self.rd)
            self.rd = None
        if self._flow_on():
            self._flow_apply(scene, samples)
            self.samples.extend(samples)
            self.pending = self.redistance
            scene.touch()
            return
        if GPU_STROKE is not None and getattr(self, 'on_gpu', False) and GPU_STROKE(scene, self, samples):
            self.redistance = False
        else:
            BR.apply_stroke(scene, self.oid, self.settings, samples, self.rec)
        self.samples.extend(samples)
        self.pending = self.redistance
        if not getattr(self, 'on_gpu', False):      # a GPU stroke changes the CPU layer only at its release
            scene.touch()

    def cpu_redist_after_gpu(self, scene):
        """r66 (T4 fallback): a GPU stroke's bricks re-distanced on the CPU at its release (brush.redistance with its
        own skip / keep-only-if-lower-L rules, then the collider again), after the write-back into the sculpt layer.
        Recorded in self.rd (undone with the stroke); settings['gpu_end_redist'] makes a replay repeat it."""
        was = self.redistance
        self.redistance = True
        mode = self.settings.get('gpu_end_redist') or ('gpu_iter' if GPU_END_REDIST_FORCE else True)
        self._force_rd = mode in ('force', 'force_iter', 'gpu_iter')   # a replay repeats the variant the stroke ran with
        self._iter_rd = mode in ('force_iter', 'gpu_iter')  # 11-09: iterated to convergence ('force': one pass)
        self._gpu_rd = mode == 'gpu_iter'                  # 11-09: the same scheme on the card (gpu/redist.py)
        try:
            self._redistance(scene)
        finally:
            self.redistance = was
            self._force_rd = False
            self._iter_rd = False
            self._gpu_rd = False
        self.settings['gpu_end_redist'] = mode
        scene.touch()

    def finish(self, scene):
        """Mouse release: re-distance the touched bricks (when dabs were added since the last one)."""
        if self.pending:
            self._redistance(scene)
            scene.touch()

    def undo(self, scene):
        self.idle_redist_stop()                  # 11-09: its idle passes end here (the count so far is recorded)
        self._debt_undo()
        # hand2b (verifier item 7): keep the stroke's END state of every brick it touched, so a redo writes it back
        # (bit-exact, milliseconds) instead of recomputing every dab and the re-distance (up to 7.3 s per step on
        # the full-body figure). The records stay too, so the redone stroke undoes again exactly as before.
        st = scene.store
        post = {}
        fast = type(self).do is BrushStroke.do         # subclasses with their own do() (LayerSmooth) recompute
        for r in ((self.rec, getattr(self, 'rd', None)) if fast else ()):
            if r is None:
                continue
            for key in list(r.snap) + list(r.new) + list(r.Lsnap):
                if key in post:
                    continue
                oid, b = key
                if scene.index_of(oid) < 0:
                    continue
                sl = scene.get(oid).sculpt.get_slot(b)
                if sl >= 0:
                    post[key] = (st.delta[sl].copy(), st.mask[sl].copy(), float(st.Lb[sl]))
        self._post = post if fast else None
        # fs-brush 11-11 (design 4.8): a Region stroke's painted ids are part of its end state (kept only for strokes
        # that painted ids: 1 KB a brick, so other strokes keep their undo memory as before)
        self._post_rid = None
        if fast and self.rec is not None and getattr(self.rec, 'rsnap', None) is not None and \
                (self.rec.rsnap or self.settings.get('tool') == 'region'):
            self._post_rid = {}
            for key in list(self.rec.rsnap) + list(self.rec.new):
                oid, b = key
                if scene.index_of(oid) >= 0:
                    sl = scene.get(oid).sculpt.get_slot(b)
                    if sl >= 0:
                        self._post_rid[key] = st.rid[sl].copy()
        # fs-brush 11-12: a Mask stroke's paint directions are part of its end state too (redo wrote the mask back but
        # left the directions as undone: store/t_redo_bisect.py, live 11-12); kept only for strokes that tag directions
        self._post_mdir = None
        if fast and self.rec is not None and self.settings.get('tool') == 'mask' and self.settings.get('paint_dir', False):
            self._post_mdir = {}
            for key in list(getattr(self.rec, 'dsnap', {})) + list(self.rec.new):
                oid, b = key
                if scene.index_of(oid) >= 0:
                    sl = scene.get(oid).sculpt.get_slot(b)
                    if sl >= 0:
                        self._post_mdir[key] = st.mdir[sl].copy()
        self._keep = (self.rec, getattr(self, 'rd', None), getattr(self, 'rd_info', None)) if fast else None
        if getattr(self, 'rd', None) is not None:
            BR.undo_record(scene, self.rd)
            self.rd = None
        BR.undo_record(scene, self.rec)
        self.rec = None

    def _redo_fast(self, scene):
        """redo from the end state kept by undo: allocate the bricks the stroke created, write every touched brick's
        end contents and bound back"""
        rec, rd, info = self._keep
        st = scene.store
        oids = set()
        for r in (rec, rd):
            if r is None:
                continue
            byo = {}                                   # r28: allocate the created bricks per object in bulk
            for oid, b in r.new:
                byo.setdefault(oid, []).append(b)
                oids.add(oid)
            for oid, bl in byo.items():
                g = scene.get(oid).sculpt
                need = [b for b in bl if g.get_slot(b) < 0]
                if need:
                    g.set_slots(need, [st.alloc() for _ in need])
        sls, ds, ms, Ls = [], [], [], []                  # r28: written back in one assignment
        for (oid, b), (d, m, L) in self._post.items():
            sl = scene.get(oid).sculpt.get_slot(b)
            if sl < 0:
                continue
            sls.append(sl); ds.append(d); ms.append(m); Ls.append(L)
            oids.add(oid)
        if sls:
            sl_ = np.array(sls, np.int64)
            st.delta[sl_] = np.stack(ds)
            st.mask[sl_] = np.stack(ms)
            st.Lb[sl_] = np.array(Ls)
        for (oid, b), rr in (getattr(self, '_post_rid', None) or {}).items():   # fs-brush 11-11: the painted ids
            sl = scene.get(oid).sculpt.get_slot(b)
            if sl >= 0:
                st.rid[sl] = rr
        self._post_rid = None
        for (oid, b), dd in (getattr(self, '_post_mdir', None) or {}).items():   # fs-brush 11-12: the paint directions
            sl = scene.get(oid).sculpt.get_slot(b)
            if sl >= 0:
                st.mdir[sl] = dd
        self._post_mdir = None
        self.rec, self.rd, self.rd_info = rec, rd, info
        self._post = None
        self._keep = None
        self._debt_redo()
        self.pending = False
        scene.touch(index=True)
        for oid in oids:
            scene.refresh_object(scene.index_of(oid))
        scene.touch()

    def discard(self, scene, state):
        if state == 'undone':                    # the kept end state is only for a redo
            self._post = None
            self._keep = None
            self._post_rid = self._post_mdir = None   # fs-brush 11-12: their ids / directions too

    def nbytes(self):
        # fs-trust 11-11 (fs-brush, M: History._cap runs every drag batch and re-summed every record's arrays in Python,
        # 6.3 ms per frame at 112 steps): O(1) per step from the record lengths (every snapshot is one brick's delta +
        # mask, 8^3 float32 each: what brush.StrokeRecord.nbytes sums)
        n = 256 + _rec_bytes(self.rec)
        n += _rec_bytes(getattr(self, 'rd', None))
        post = getattr(self, '_post', None)
        if post:
            n += len(post) * _SNAP_B
            for r in self._keep[:2]:
                n += _rec_bytes(r)
        return n

    # ---- fs-trust 11-11: the undo journal (core/undojournal.py)
    def undo_state(self):
        """a done stroke's records (rec: the bricks before it and those it created; rd: its release re-distance). Taken
        by reference: a done stroke's records no longer change once the next command is pushed (its idle passes and
        debt were closed by History.push)"""
        rec, rd = self.rec, getattr(self, 'rd', None)
        if rec is None or getattr(self, '_rdst', None):
            return None

        def make():
            A = {}
            UJ.record_arrays('rec', rec, A)
            if rd is not None:
                UJ.record_arrays('rd', rd, A)
            return {'kind': 'stroke', 'rd': rd is not None}, A
        return make

    def restore_undo_state(self, scene, meta, A):
        self.rec = UJ.record_from('rec', A)
        self.rd = UJ.record_from('rd', A) if meta.get('rd') else None
        self.pending = False
        self.live = False

    def evict_undo(self):
        if self.rec is None or getattr(self, '_post', None) is not None or getattr(self, '_rdst', None):
            return False
        self.rec = None
        self.rd = None
        return True


def _build_objects(sc, scene_doc, data):
    """r44: add a described scene's objects (with their sculpt arrays) to sc, then re-measure the bricks' Lipschitz
    bounds (load_project and Checkpoint share it)"""
    sc.next_id = scene_doc['next_id']
    for od in scene_doc['objects']:
        o = SceneObject(od['id'], od['name'], ops.Node.from_dict(od['base']), transform=od['transform'],
                        colour=od['colour'], material=od['material'], h=od.get('sculpt_h', DEFAULT_H))
        o.detail = od['detail']
        o.visible = od['visible']
        o.mask_inv = bool(od.get('mask_invert', False))
        o.locked = od.get('locked')
        o.root = od.get('root')
        o.part = od.get('part')                  # fs-contact 11-11
        o.garment = od.get('garment')            # fs-shell 11-11 (fs-cloth's garment record)
        o.layers = copy.deepcopy(od.get('layers') or [])          # fs-shell 11-11: sculpt layers
        o.snapshots = copy.deepcopy(od.get('snapshots') or [])    # fs-shell 11-11: snapshots
        o.regions = {int(k): dict(v) for k, v in (od.get('regions') or {}).items()}   # fs-shell 11-11
        o.group = od.get('group')                # fs-tools 11-14: Group / Ungroup
        key = 'o%d_bricks' % o.id
        if key in data:
            bricks = data[key]
            o.sculpt.ensure_index(bricks.min(0), bricks.max(0))
            dl = data['o%d_delta' % o.id]
            mk = data['o%d_mask' % o.id]
            mdk = data['o%d_mdir' % o.id] if ('o%d_mdir' % o.id) in data else None
            rdk = data['o%d_rid' % o.id] if ('o%d_rid' % o.id) in data else None   # fs-shell 11-11: regions
            for i, b in enumerate(map(tuple, bricks.tolist())):
                s = sc.store.alloc()
                sc.store.delta[s] = dl[i]
                sc.store.mask[s] = mk[i]
                if mdk is not None:
                    sc.store.mdir[s] = mdk[i]
                if rdk is not None:
                    sc.store.rid[s] = rdk[i]
                o.sculpt.set_slot(b, s)
        sc.objects.append(o)
    sc.touch(structural=True, index=True)
    pk = sc.pack()
    for oi, o in enumerate(sc.objects):
        if o.sculpt.slots:
            keys = list(o.sculpt.slots.keys())
            out = np.zeros(len(keys))
            if BR.TSDF_L:
                # fs-core 11-14: under the truncated band, L over the band's cells only (core/brush.py
                # brick_lipschitz_band), as the card's write-back measures it; the field at the bricks' nodes from the CPU
                arr = np.array(keys, np.int64)
                F9 = np.empty((len(keys), 9, 9, 9), np.float32)
                for a in range(0, len(keys), 4096):
                    blk = np.empty((min(4096, len(keys) - a), 9, 9, 9))
                    BR.brick_F9(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, arr[a:a + 4096],
                                float(o.sculpt.h), blk)
                    F9[a:a + len(blk)] = blk
                BR.brick_lipschitz_band(pk.index_flat, pk.delta, pk.oi_tab, oi, o.sculpt.h, arr, F9,
                                        BR.TSDF_W_H * float(o.sculpt.h), out)
            else:
                BR.brick_lipschitz(pk.index_flat, pk.delta, pk.oi_tab, oi, o.sculpt.h, np.array(keys, np.int64), out)
            for b, L in zip(keys, out):
                sc.store.Lb[o.sculpt.slots[b]] = L
            sc.refresh_object(oi)


def _scene_arrays(sc, copy_=False):
    arrays = {}
    st = sc.store
    for o in sc.objects:
        keys = sorted(b for b, s in o.sculpt.slots.items() if st.delta[s].any() or st.mask[s].any() or st.rid[s].any())
        if not keys:
            continue
        slots = [o.sculpt.slots[b] for b in keys]
        arrays['o%d_bricks' % o.id] = np.array(keys, np.int64)
        arrays['o%d_delta' % o.id] = st.delta[slots].copy() if copy_ else st.delta[slots]
        arrays['o%d_mask' % o.id] = st.mask[slots].copy() if copy_ else st.mask[slots]
        if st.mdir[slots].any():
            arrays['o%d_mdir' % o.id] = st.mdir[slots].copy()
        if st.rid[slots].any():                  # fs-shell 11-11: region ids, only when any
            arrays['o%d_rid' % o.id] = st.rid[slots].copy()
    return arrays


@command
class Checkpoint(Command):
    """r44 (consolidate history): the state that a run of folded op-log entries built, as one log entry. It stands
    only at the head of a log (replay into an empty scene); its data lives in core/gpu_results' checkpoint store and
    is saved with the project (checkpoint_<cid>.npz). Its do() checks the stored state hash."""
    label = 'Consolidated history'

    def __init__(self, cid, n_folded=0):
        self.cid = cid
        self.n_folded = int(n_folded)

    def args(self):
        return {'cid': self.cid, 'n_folded': self.n_folded}

    def do(self, scene):
        d = GR.ck_get(self.cid)
        if d is None:
            raise ValueError('checkpoint %s is missing (its checkpoint_%s.npz)' % (self.cid, self.cid))
        if scene.objects:
            raise ValueError('a checkpoint replays only into an empty scene (the head of the log)')
        self.prev_next = scene.next_id
        _build_objects(scene, d['scene'], d['arrays'])
        got = scene.state_hash()
        if got != d['hash']:
            raise ValueError('checkpoint %s hash mismatch: stored %s, built %s' % (self.cid, d['hash'][:12], got[:12]))

    def undo(self, scene):
        for o in list(scene.objects):
            release_object_slots(scene, o)
        scene.objects.clear()
        scene.next_id = self.prev_next
        scene.touch(structural=True, index=True)


def consolidate_prepare(hist, keep=50, log_snapshot=None):
    """r44: the slow half of consolidate (two replays), safe to run off the engine thread when given a snapshot
    (hist.log(), len(done), state hash) taken under the lock. -> plan dict, or None when there is nothing to fold."""
    if log_snapshot is None:
        hist.end_stroke()
        log_snapshot = (hist.log(), len(hist.done), hist.scene.state_hash())
    log, n_done, h_before = log_snapshot
    keep = max(0, int(keep))
    n_fold_done = max(0, n_done - keep)
    n_fold = len(log) - n_done + n_fold_done          # base + committed + the oldest done
    folded, kept_log = log[:n_fold], log[n_fold:]
    if n_fold <= 1:
        return None
    r = replay(folded)
    cid = GR.ck_new_id()
    n_prev = sum(int(e.get('n_folded', 1)) if e.get('type') == 'Checkpoint' else 1 for e in folded)
    GR.ck_put(cid, {'scene': r.scene.describe(), 'hash': r.scene.state_hash(), 'arrays': _scene_arrays(r.scene, True)})
    del r
    entry = Checkpoint(cid, n_prev).to_log()
    got = replay([entry] + kept_log).scene.state_hash()
    if got != h_before:
        GR.ck_forget(cid)
        raise ValueError('consolidate: the new log replays to %s, the scene is %s; nothing changed'
                         % (got[:12], h_before[:12]))
    return {'entry': entry, 'n_fold': n_fold, 'n_fold_done': n_fold_done, 'log_len': len(log), 'n_done': n_done,
            'hash': h_before, 'cid': cid}


def consolidate_commit(hist, plan):
    """r44: adopt a prepared plan when the history is still the one it was made from (same log length, same done
    count, same state hash); -> True when adopted. The folded done steps leave the undo stack."""
    if (len(hist.log()) != plan['log_len'] or len(hist.done) != plan['n_done']
            or hist.scene.state_hash() != plan['hash']):
        GR.ck_forget(plan['cid'])
        return False
    k = plan['n_fold_done']
    for c in hist.done[:k]:
        c.discard(hist.scene, 'committed')
    del hist.done[:k]
    hist.committed = [plan['entry']]
    if hasattr(hist, 'base_log'):
        hist.base_log = []
    hist.note = 'History consolidated: %d entries folded into one checkpoint; %d undo steps kept.' % (
        plan['n_fold'], len(hist.done))
    hist._notify()
    return True


def consolidate(hist, keep=50):
    """r44: fold the op log's oldest entries into one Checkpoint so a long session's project stays small (the GPU
    strokes' recorded results that only the folded entries named are no longer saved).

    Folded: everything that can no longer be undone (a UIHistory's base_log, the committed entries) and the done
    steps older than the newest `keep` (those leave the undo stack, as when the memory cap drops steps). The newest
    `keep` steps and the redo stack are untouched, and so are the live scene and its state: the checkpoint's state is
    built by replaying the folded entries into a scene of their own.

    Checked: the state hash before; the new log (checkpoint + kept steps) replays to that same hash, else ValueError
    and nothing changes. -> {'folded', 'kept', 'hash', 'cid', 'secs'}"""
    import time as _t
    t0 = _t.perf_counter()
    plan = consolidate_prepare(hist, keep)
    if plan is None:
        return {'folded': 0, 'kept': len(hist.done), 'hash': hist.scene.state_hash(), 'cid': None,
                'secs': round(_t.perf_counter() - t0, 2)}
    assert consolidate_commit(hist, plan)
    return {'folded': plan['n_fold'], 'kept': len(hist.done), 'hash': plan['hash'], 'cid': plan['cid'],
            'secs': round(_t.perf_counter() - t0, 2)}


class History:
    """Undo/redo stacks; the op log is the done stack's entries."""

    MEM_CAP = 1 << 30            # undo memory cap (bytes); the oldest steps' undo data is dropped past it

    # fs-trust 11-11: the undo journal is on for histories that take edits (the app, the script API, tests); a replay
    # (replay(), consolidate, Verify, Open's old background rebuild) builds a history of its own with it off
    JOURNAL = os.environ.get('FIELDSTUDIO_UNDO_JOURNAL', '1') == '1'

    def __init__(self, scene=None):
        self.scene = scene or Scene()
        self.done = []
        self.undone = []
        self.listeners = []
        self.committed = []      # log entries of the oldest steps whose undo data the cap dropped (no undo past them)
        self.mem_cap = self.MEM_CAP
        self.note = ''           # status line for the UI (set when the cap drops steps)
        self.journal = self.JOURNAL

    def push(self, cmd):
        self.end_stroke()
        self.idle_stop()                         # 11-09: the next command starts on the field as it stands
        result = cmd.do(self.scene)
        cmd.serial = next(_SERIAL)
        if self.journal and cmd.uid is None:
            cmd.uid = UJ.new_uid()
        self.done.append(cmd)
        self.clear_undone()
        if self.journal and len(self.done) > 1:
            self._journal(self.done[-2])          # the step before this one is final now: its records go to disk
        self._cap()
        self._notify()
        return result

    def _journal(self, c):
        """fs-trust: queue c's undo state for the session's undo folder (once; the one spill worker writes it)"""
        if c.uid is None or getattr(c, '_jstate', None) is not None:
            return
        mk = c.undo_state()
        if mk is None:
            c._jstate = 'barrier'                 # cannot be rebuilt from saved data: an Open's undo history stops here
            return
        c._jstate = 'queued'

        def done_(p, n, c=c):
            c._jbytes = int(n)
            c._jstate = 'disk'

        def job(c=c, mk=mk):
            meta, A = mk()
            p = os.path.join(UJ._session_dir(), c.uid + '.npz')
            n = UJ.write_file(p, meta, A)
            UJ._add(c.uid, p)
            done_(p, n)
        GR.submit(job)

    def _ensure_undo(self, c):
        """an evicted or Open-rebuilt step reads its undo state back from its file"""
        if not getattr(c, '_evicted', False):
            return
        p = UJ.path_of(c.uid)
        if p is None:
            raise RuntimeError('undo data of this step is missing (%s.npz)' % c.uid)
        meta, A = UJ.read_file(p)
        c.restore_undo_state(self.scene, meta, A)
        c._measure_after_undo = _ensure_bricks(self.scene, c)
        c._evicted = False

    def resident_bytes(self):
        return self.memory_bytes()

    def journal_bytes(self):
        """bytes of undo data held on disk only (evicted steps)"""
        return sum(int(getattr(c, '_jbytes', 0) or 0) for c in self.done if getattr(c, '_evicted', False))

    def memory_report(self):
        """fs-trust: what the app holds for history, in bytes (the status bar's memory line)"""
        return {'undo_ram': self.memory_bytes(), 'undo_disk': self.journal_bytes(), 'results_ram': GR.ram_bytes(),
                'checkpoints_ram': GR.ck_ram_bytes(), 'undo_ram_budget': UJ.RAM_BUDGET if self.journal else self.mem_cap,
                'steps': len(self.done), 'steps_on_disk': sum(1 for c in self.done if getattr(c, '_evicted', False))}

    def _cap(self):
        """Keep the undo memory under mem_cap: the oldest done steps become committed (their log entries stay in
        the op log, so replay and the state hash are unchanged; only undo past them is gone).
        fs-trust 11-11 (journal on): RAM holds at most UJ.RAM_BUDGET of undo data: the oldest steps whose records are
        on disk drop them from RAM (read back when undone); the depth cap counts RAM and disk together against
        max(mem_cap, UJ.DISK_CAP)."""
        mem = self.memory_bytes()
        cap = self.mem_cap
        if self.journal:
            i = 0
            while mem > UJ.RAM_BUDGET and i < len(self.done) - UJ.KEEP_RESIDENT:
                c = self.done[i]
                i += 1
                if getattr(c, '_jstate', None) == 'disk' and not getattr(c, '_evicted', False):
                    nb = c.nbytes()
                    if c.evict_undo():
                        c._evicted = True
                        mem -= nb - c.nbytes()
            mem += self.journal_bytes()
            cap = max(self.mem_cap, UJ.DISK_CAP)
        n = 0
        while mem > cap and len(self.done) > 1:
            c = self.done.pop(0)
            mem -= c.nbytes() + (int(getattr(c, '_jbytes', 0) or 0) if getattr(c, '_evicted', False) else 0)
            self.committed.append(c.to_log())
            c.discard(self.scene, 'committed')
            n += 1
        if n:
            self.note = ('Undo memory over %.0f MB: the oldest %d step(s) can no longer be undone '
                         '(%d committed in all).' % (cap / 2 ** 20, n, len(self.committed)))

    def clear_undone(self):
        """hand2b: the redo stack is superseded: each command frees what only its redo needed (verifier item 16)"""
        while self.undone:
            self.undone.pop().discard(self.scene, 'undone')

    def drop_undone_last(self):
        """hand2b: drop the command just undone (an edit merged into its successor): freed like a superseded one"""
        self.undone.pop().discard(self.scene, 'undone')

    def end_stroke(self):
        """The mouse released: finish the live stroke (re-distance its bricks). Safe to call any time."""
        c = self.done[-1] if self.done else None
        if c is not None and getattr(c, 'on_gpu', False) and GPU_STROKE_END is not None:
            GPU_STROKE_END(self.scene, c)
            self._notify()
        if c is not None and getattr(c, 'pending', False):
            c.finish(self.scene)
            self._notify()
        # fs-trust 11-11: no pool trim here (R9d, a project copy, 15 min: a trim at each stroke's end freed ~1 GB the next stroke
        # allocated again, up to 258 ms on the release path; fs-brush's condition: off the stroke path). The pool goes back
        # at a save (snapshot_project), after Open, and when the engine has been idle (ui/engine.py POOL_IDLE_S).

    def idle_pending(self):
        """11-09: the last stroke still has release passes for idle time, or (r87) debt to pay"""
        c = self.done[-1] if self.done else None
        return c is not None and hasattr(c, 'idle_redist_pending') and (c.idle_redist_pending() or c.idle_debt_pending())

    def idle_step(self):
        """11-09: one idle-time step of the last stroke (its own release pass, else r87 one debt payment) -> True if
        more remain"""
        if not self.idle_pending():
            return False
        c = self.done[-1]
        if c.idle_redist_pending():
            c.idle_redist_step(self.scene)
        else:
            c.idle_debt_step(self.scene)
        self._notify()
        return self.idle_pending()

    def idle_stop(self):
        c = self.done[-1] if self.done else None
        if c is not None and hasattr(c, 'idle_redist_stop'):
            c.idle_redist_stop()
            c.idle_debt_close()

    def undo(self):
        if not self.done:
            return False
        self._ensure_undo(self.done[-1])         # fs-trust: an evicted / Open-rebuilt step reads its records back
        c = self.done.pop()
        c.undo(self.scene)
        m = getattr(c, '_measure_after_undo', None)
        if m:
            for oid, bs in m.items():
                if bs and self.scene.index_of(oid) >= 0:
                    BR.measure_bricks(self.scene, self.scene.index_of(oid), bs)
            c._measure_after_undo = None
        self.undone.append(c)
        self._notify()
        return True

    def redo(self):
        if not self.undone:
            return False
        c = self.undone.pop()
        c.do(self.scene)
        self.done.append(c)
        self._notify()
        return True

    def extend_stroke(self, samples):
        """Add dabs to the last pushed BrushStroke (the mouse drag in progress); one undo step per drag."""
        c = self.done[-1] if self.done else None
        if not isinstance(c, BrushStroke):
            raise ValueError('last command is not a brush stroke')
        c.extend(self.scene, samples)
        self._cap()
        self._notify()

    def can_undo(self):
        return bool(self.done)

    def can_redo(self):
        return bool(self.undone)

    def log(self):
        return list(self.committed) + [c.to_log() for c in self.done]

    def memory_bytes(self):
        return sum(c.nbytes() for c in self.done) + sum(c.nbytes() for c in self.undone)

    def _notify(self):
        for f in self.listeners:
            f()


def apply_gpu_result(scene, data, rec):
    """r42: write a recorded GPU stroke result (per object: sculpt brick keys, delta after) through rec, as
    brush.apply_sample allocates and snapshots."""
    st = scene.store
    for oid, (keys, after) in data.items():
        if scene.index_of(oid) < 0:
            continue
        g = scene.get(oid).sculpt
        keys = np.asarray(keys, np.int64).reshape(-1, 3)
        if not len(keys):
            continue
        g.ensure_index(keys.min(0), keys.max(0))
        slots = np.empty(len(keys), np.int64)
        for i, b in enumerate(map(tuple, keys.tolist())):
            key = (oid, b)
            s = g.get_slot(b)
            if s < 0:
                s = st.alloc()
                g.set_slot(b, s)
                rec.new.append(key)
                rec.new_set.add(key)
            elif key not in rec.snap and key not in rec.new_set:
                rec.snap[key] = (st.delta[s].copy(), st.mask[s].copy(), float(st.Lb[s]))
            slots[i] = s
        if after.dtype == np.uint32:                 # r45: the XOR form (gpu_results.encode)
            st.delta[slots] = (np.ascontiguousarray(st.delta[slots]).view(np.uint32) ^ after).view(np.float32)
        else:
            st.delta[slots] = np.asarray(after, np.float32)
        scene.touch(index=True)
        BR.measure_bricks(scene, scene.index_of(oid), set(map(tuple, keys.tolist())))


def replay(log):
    """Build a new History by replaying an op log from an empty scene."""
    hist = History(Scene())
    hist.journal = False                     # fs-trust: a replay keeps its undo data in RAM only
    for entry in log:
        hist.push(Command.from_log(entry))
    return hist


def state_hash(scene):
    return scene.state_hash()


# ---- project save / load (.fsproj folder: project.json + sculpt.npz) --------------------------------------

FORMAT = 'fieldstudio-project/1'


def save_project(hist, path, extra=None):
    if hasattr(hist, 'idle_stop'):
        hist.idle_stop()                         # 11-09: the saved state and log are final for the last stroke
    """r48: = snapshot_project + write_snapshot (the sculpt layer as chunk files; only changed chunks compressed)"""
    doc, chunks_ = snapshot_project(hist, extra, copy_=False)
    return write_snapshot(doc, chunks_, path)


def snapshot_project(hist, extra=None, copy_=True):
    """hand2f / r48: -> (doc, chunks) of the current state, the sculpt chunks COPIED (copy_), so the slow part (the
    write) can run on another thread while sculpting goes on (write_snapshot)."""
    from . import chunks as CH
    if hasattr(hist, 'idle_stop'):
        hist.idle_stop()     # fs-trust 11-11: the last stroke's log entry, records and arrays are final at the snapshot
    _pool_trim('save')       # fs-trust 11-11: a save is a safe point for the card pool
    sc = hist.scene
    # r48b: the state hash is computed by write_snapshot from these copies (off the engine lock; 3.4 s held before)
    doc = {'format': FORMAT, 'scene': sc.describe(), 'log': copy.deepcopy(hist.log()), 'hash': None,
           'object_ids': [o.id for o in sc.objects], 'extra': extra or {}}
    _undo_snapshot(hist, doc)
    return doc, CH.chunk_arrays(sc, copy_)


def _undo_snapshot(hist, doc):
    """fs-trust 11-11: doc['undo'] names the undo-journal file of every done step (None: not rebuildable), oldest first;
    a step not on disk yet (the last one) has its state taken now, consistent with the arrays (doc['_undo_pending'],
    written by write_snapshot, not saved in project.json)"""
    if not getattr(hist, 'journal', False):
        return
    uids, pend = [], {}
    budget = UJ.SAVE_QUOTA                    # bytes of undo files a project carries (the newest steps first)
    used = 0
    for c in reversed(hist.done):
        st = getattr(c, '_jstate', None)
        size = int(getattr(c, '_jbytes', 0) or 0) or c.nbytes() // 2      # on disk, or about half its RAM (deflate)
        if c.uid is None or st == 'barrier' or used + size > budget:
            uids.append(None)
            budget = -1                           # nothing older is carried: the reopened history ends here
            continue
        used += size
        if st in ('disk', 'queued'):
            uids.append(c.uid)
        else:
            mk = c.undo_state()
            if mk is None:
                uids.append(None)
                budget = -1
            else:
                pend[c.uid] = mk()
                uids.append(c.uid)
    uids.reverse()
    # the window's grouping (several commands undone as one step) and labels come back with the steps
    attrs = [{k: getattr(c, k) for k in ('group_n', 'group_first_n', 'ui_label') if getattr(c, k, None) is not None}
             for c in hist.done]
    doc['undo'] = {'format': 'undojournal/1', 'done': uids, 'attrs': attrs}
    doc['_undo_pending'] = pend


OPEN_CMD = 'Open in Field Studio.cmd'


def write_open_cmd(folder):
    """(11-08 night) a .fsproj is a folder, so Windows cannot open it by double-click: every save puts this
    small launcher in it (Field Studio --open on its own folder, no console). An added file only: not in the state hash,
    not read by Open."""
    run = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'run_field_studio.py'))
    txt = ('@echo off\r\n'
           'rem Opens this project in Field Studio (written by Field Studio at each save; safe to delete).\r\n'
           'where pythonw >nul 2>nul && (start "" pythonw "%s" --open "%%~dp0." & goto :eof)\r\n'
           'python "%s" --open "%%~dp0."\r\n' % (run, run))
    with open(os.path.join(folder, OPEN_CMD), 'w', encoding='ascii', newline='') as f:
        f.write(txt)


def _replace_retry(src, dst, tries=60):
    """os.replace that waits out a transient Windows sharing violation (up to ~10 s), then raises"""
    import time as _t
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            _t.sleep(0.05 if i < 20 else 0.25)


def recover_interrupted_save(path):
    """fs-trust 11-11: a save that was cut between its two renames leaves <path>.old (the previous complete save) and
    no <path>: it is put back (the half-written <path>.tmp is left for the next save to clear). -> True if recovered"""
    old = path + '.old'
    if not os.path.isdir(path) and os.path.isfile(os.path.join(old, 'project.json')):
        _replace_retry(old, path)
        return True
    return False


def write_snapshot(doc, chunks_, path, refs=None):
    """Write a snapshot as a .fsproj folder ATOMICALLY: into <path>.tmp, then swapped in (the old copy is removed
    only after the new one is complete), so a crash mid-write leaves the previous version intact. r48: chunks,
    GPU stroke results and checkpoints unchanged since the folder's previous version are hard-linked from it, not
    re-compressed or copied. -> the state hash"""
    import shutil
    from . import chunks as CH
    tmp = path + '.tmp'
    old = path + '.old'
    recover_interrupted_save(path)         # fs-trust: never clear .old while it is the only complete save
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    prev = (path, CH.manifest_of(path)) if os.path.isdir(path) else None
    if refs:
        # r49: the reference images this save names, copied into the new folder before the swap (a source may be the
        # current folder's own refs/ copy: it still exists until the swap)
        rd = os.path.join(tmp, 'refs')
        os.makedirs(rd, exist_ok=True)
        for src, name in refs:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(rd, name))
    manifest, stats = CH.write(tmp, chunks_, prev)
    doc = dict(doc)
    ids = doc.pop('object_ids', None)
    if doc.get('hash') is None:
        doc['hash'] = CH.state_hash(doc['scene'], chunks_, ids)
    doc['sculpt'] = {'kind': CH.KIND, 'manifest': manifest}
    GR.save_into(tmp, GR.rids_in(doc.get('log', [])), prev=path if os.path.isdir(path) else None)   # r42 / r48
    GR.ck_save_into(tmp, GR.ck_ids_in(doc.get('log', [])), prev=path if os.path.isdir(path) else None)   # r44 / r48
    pend = doc.pop('_undo_pending', None)
    if doc.get('undo'):                                   # fs-trust 11-11: the undo journal's files of the done steps
        UJ.flush()                                        # steps queued for the session folder are written first
        UJ.save_into(tmp, doc['undo']['done'], prev=path if os.path.isdir(path) else None, pending=pend)
    with open(os.path.join(tmp, 'project.json'), 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    write_open_cmd(tmp)                                   # 11-08 night: double-click to open (not part of the state)
    foreign = []
    if os.path.isdir(path):
        # r48: whatever else the folder holds (refs/ the window copied in for this save, the user's own files) moves over
        own = {'project.json', 'sculpt', 'sculpt.npz', 'gpu_strokes', 'gpu_strokes.npz', 'undo'}
        for f in os.listdir(path):
            if f in own or (f.startswith('checkpoint_') and f.endswith('.npz')) or os.path.exists(os.path.join(tmp, f)):
                continue
            _replace_retry(os.path.join(path, f), os.path.join(tmp, f))
            foreign.append(f)
    # fs-trust 11-11 (M, R9 round 12 on a project copy: the folder went to .old and the new one never took its place;
    # Open then found no project): the swap retries while Windows holds a handle (a scanner or the indexer opening the
    # files just written), and when it still cannot, the previous save is put back where it was, with the user's files
    shutil.rmtree(old, ignore_errors=True)
    moved = False
    if os.path.exists(path):
        _replace_retry(path, old)
        moved = True
    try:
        _replace_retry(tmp, path)
    except Exception:
        if moved and not os.path.exists(path):
            _replace_retry(old, path)
            for f in foreign:
                if os.path.exists(os.path.join(tmp, f)) and not os.path.exists(os.path.join(path, f)):
                    _replace_retry(os.path.join(tmp, f), os.path.join(path, f))
        raise
    shutil.rmtree(old, ignore_errors=True)
    GR.repoint(path)            # r48: registered result files now live in the new folder
    GR.ck_repoint(path)         # fs-trust 11-11: saved checkpoints leave RAM (read from the file when needed)
    UJ.repoint(path)            # fs-trust 11-11: the undo journal's files in the new folder
    write_snapshot.last_stats = stats
    write_snapshot.last_hash = doc['hash']
    return doc['hash']


write_snapshot.last_stats = None


def load_project(path):
    """-> History whose scene equals the saved one (hash-checked); the saved log becomes the op log baseline
    (undo history is not restored; replaying 'log' reproduces the state)."""
    from .program import BRICK
    recover_interrupted_save(path)         # fs-trust 11-11: a save cut between its renames: the last complete one
    with open(os.path.join(path, 'project.json'), encoding='utf-8') as f:
        doc = json.load(f)
    GR.load_from(path)                                # r42: GPU strokes' results, for replay
    GR.ck_load_from(path)                             # r44: checkpoints (consolidated history)
    if doc.get('format') != FORMAT:
        raise ValueError('not a Field Studio project: %r' % doc.get('format'))
    sc = Scene()
    npz_path = os.path.join(path, 'sculpt.npz')
    if (doc.get('sculpt') or {}).get('kind') == 'chunks/1':     # r48
        from . import chunks as CH
        data = CH.read(path, doc['sculpt']['manifest'])
    else:
        data = np.load(npz_path) if os.path.exists(npz_path) else {}
    _build_objects(sc, doc['scene'], data)            # r44: shared with Checkpoint
    hist = History(sc)
    hist.loaded_log = doc.get('log', [])
    got = sc.state_hash()
    if got != doc.get('hash'):
        raise ValueError('project hash mismatch: saved %s, loaded %s' % (doc.get('hash'), got))
    hist.loaded_base = list(hist.loaded_log)
    hist.loaded_done = _rebuild_done(path, doc, hist)
    return hist


def _rebuild_done(path, doc, hist):
    """fs-trust 11-11: the undo history from the undo journal, no replay. The done steps are made from their log
    entries; each reads its undo records from <path>/undo/<uid>.npz when it is undone. The rebuilt run is the newest
    steps back to the first one without a file (or not rebuildable); hist.loaded_base keeps the entries before it (the
    base no undo goes past) and hist.loaded_log stays the whole saved log, as before. The caller adopts them (ui/engine
    _open, script.Session): base_log = loaded_base, done = loaded_done. -> the done steps (oldest first), [] when the
    project has no journal"""
    u = doc.get('undo') or {}
    uids = list(u.get('done') or [])
    log = doc.get('log', [])
    if not uids or len(uids) > len(log):
        return []
    UJ.load_from(path)
    k = len(uids)
    while k > 0 and uids[k - 1] and UJ.path_of(uids[k - 1]):
        k -= 1
    if k == len(uids):
        return []
    first = len(log) - len(uids) + k
    attrs = list(u.get('attrs') or [])
    attrs = attrs[k:] if len(attrs) == len(uids) else [{}] * (len(uids) - k)
    out = []
    for e, uid, at in zip(log[first:], uids[k:], attrs):
        c = Command.from_log(e)
        for a_, v_ in at.items():
            if a_ == 'group_first_n' or a_ == 'group_n':
                v_ = int(v_)
            setattr(c, a_, v_)
        c._entry = copy.deepcopy(e)
        c.uid = uid
        c._evicted = True
        c._jstate = 'disk'
        p = UJ.path_of(uid)
        c._jbytes = os.path.getsize(p) if p else 0
        out.append(c)
    hist.loaded_base = log[:first]
    return out


# fs-materials 11-14: the material commands (AssignMaterial, MaterialBatch) register themselves here, so a log that holds
# them replays wherever the history is loaded (materials/commands.py)
from ..materials import commands as _fsmat_commands  # noqa: E402,F401
