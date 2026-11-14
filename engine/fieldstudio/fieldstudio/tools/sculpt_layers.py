"""Sculpt layers and snapshots (fs-shell, 2026-11-11; design 4.10 "Sculpt layers with strength": "a layer is a stored
field delta ... Each object gets a layer stack: name, strength slider (-1 to 2), on / off, solo, merge down, and record
(strokes go to the selected layer) ... A lid pass on its own layer can be dialled from 0 to 1 for comparison, with
nothing lost"; "Snapshot saves a named state of the object's bricks ... A Restore ... back toward a snapshot").

The data shape, agreed with fs-brush (11-11):
  * The object's sculpt delta stays the COMPOSITE every reader reads (base + each layer x its strength): nothing in the
    brushes, the colliders, the tracers or the card changes.
  * A layer's content is composite - origin over the bricks touched while it recorded: origin[b] is the brick's value
    the first time anything touched it during the recording (each later command's own undo record, first touch wins:
    rec.snap, then zeros for rec.new, then rd.snap, then zeros for rd.new), so idle re-distance passes, debt, the
    collider clamp and undos during the recording all come out right with no hook into a stroke's timing. It is
    computed when the recording stops (capture) and stored by the sha of its bytes (tools/bakes.write_arrays), named in
    the object's record (so the state hash covers it and saves carry the file).
  * A layer records at strength 1. Dialling it s0 -> s1 adds (s1 - s0) x the layer to the composite over its bricks; a
    change over one lattice step is re-distanced (brush.redistance) so the brushes read F as a distance again; bounds
    re-measured, the card's rows re-uploaded. Undo is by snapshot (a StrokeRecord, first touch): bit exact on replay.
  * A dial is a blend of two fields (the brushes are not linear): the panel says so.
Commands (logged, undoable): LayerNew, LayerSet (the capture's result), LayerDial (strength / on), LayerDelete (remove its
contribution, or merge it into the base), SnapshotTake, SnapshotRestore (all, or blended by the mask), SnapshotDelete."""
import copy

import numpy as np

from ..core.history import Command, command
from ..core.scene import B
from . import bakes as BK


def _layer(o, lid):
    for L in getattr(o, 'layers', None) or []:
        if int(L['id']) == int(lid):
            return L
    raise ValueError('object %s has no layer %s' % (o.id, lid))


def eff(L):
    return float(L.get('strength', 1.0)) if L.get('on', True) else 0.0


# ---- capture: a recording's content, from the history's own undo records ------------------------------------------------
def capture(hist, oid, lid):
    """on the engine thread: the bricks the recording of layer lid touched, composite - origin -> (keys (n, 3) int64,
    delta (n, 8, 8, 8) float32), or None when the recording's start is not in the done steps"""
    done = list(hist.done)
    i0 = None
    for i, c in enumerate(done):
        if type(c).__name__ == 'LayerNew' and int(c.oid) == int(oid) and int(c.lid or -1) == int(lid):
            i0 = i
    if i0 is None:
        return None
    origin = {}
    for c in done[i0 + 1:]:
        try:
            hist._ensure_undo(c)
        except Exception:
            pass
        for rec in (getattr(c, 'rec', None), getattr(c, 'rd', None)):
            if rec is None or not hasattr(rec, 'snap'):
                continue
            for (o_, b), v in rec.snap.items():
                if int(o_) == int(oid) and b not in origin:
                    origin[b] = np.array(v[0], np.float32)
            for (o_, b) in getattr(rec, 'new', []):
                if int(o_) == int(oid) and b not in origin:
                    origin[b] = np.zeros((B, B, B), np.float32)
    sc = hist.scene
    o = sc.get(oid)
    st = sc.store
    keys, D = [], []
    for b in sorted(origin):
        s = o.sculpt.get_slot(b)
        cur = st.delta[s] if s >= 0 else np.zeros((B, B, B), np.float32)
        d = (cur - origin[b]).astype(np.float32)
        if np.any(d != 0.0):
            keys.append(b)
            D.append(d)
    if not keys:
        return np.zeros((0, 3), np.int64), np.zeros((0, B, B, B), np.float32)
    return np.array(keys, np.int64), np.stack(D)


# ---- the commands -------------------------------------------------------------------------------------------------------
@command
class LayerNew(Command):
    """a new layer on top of the stack, recording (strength 1): strokes from here on go into it when it is captured"""
    label = 'New layer'

    def __init__(self, oid, name=None, lid=None):
        self.oid = int(oid)
        self.name = name
        self.lid = None if lid is None else int(lid)

    def args(self):
        return {'oid': self.oid, 'name': self.name, 'lid': self.lid}

    def do(self, scene):
        o = scene.get(self.oid)
        if any(L.get('recording') for L in o.layers):
            raise ValueError('a layer is already recording on %s: stop it first' % o.name)
        if self.lid is None:
            self.lid = max([int(L['id']) for L in o.layers] + [0]) + 1
        self.name = self.name or 'Layer %d' % self.lid
        o.layers.append({'id': self.lid, 'name': self.name, 'strength': 1.0, 'on': True, 'sha': None, 'file': None,
                         'bricks': 0, 'recording': True})
        scene.touch(structural=True)
        return [self.oid]

    def undo(self, scene):
        o = scene.get(self.oid)
        o.layers = [L for L in o.layers if int(L['id']) != self.lid]
        scene.touch(structural=True)
        return [self.oid]


@command
class LayerSet(Command):
    """the end of a recording: the layer's content (stored by sha) named in its record"""
    label = 'Layer recorded'

    def __init__(self, oid, lid, sha, file, bricks=0):
        self.oid, self.lid = int(oid), int(lid)
        self.sha, self.file, self.bricks = sha, file, int(bricks)

    def args(self):
        return {'oid': self.oid, 'lid': self.lid, 'sha': self.sha, 'file': self.file, 'bricks': self.bricks}

    def do(self, scene):
        L = _layer(scene.get(self.oid), self.lid)
        self._old = copy.deepcopy(L)
        L.update({'sha': self.sha, 'file': self.file, 'bricks': self.bricks, 'recording': False})
        scene.touch(structural=True)
        return [self.oid]

    def undo(self, scene):
        L = _layer(scene.get(self.oid), self.lid)
        L.clear()
        L.update(self._old)
        scene.touch(structural=True)
        return [self.oid]


class _BrickWrite(Command):
    """shared: write deltas into an object's bricks with a first-touch StrokeRecord as the undo"""

    def _begin(self, scene):
        from ..core.brush import StrokeRecord
        self.rec = StrokeRecord()
        self._touched = set()

    def _slot(self, scene, o, b):
        st = scene.store
        s = o.sculpt.get_slot(b)
        if s < 0:
            o.sculpt.ensure_index(b, b)
            s = st.alloc()
            st.delta[s] = 0.0
            st.mask[s] = 0.0
            st.mdir[s] = 0
            st.rid[s] = 0
            st.Lb[s] = 0.0
            o.sculpt.set_slot(b, s)
            self.rec.new.append((o.id, b))
            self.rec.new_set.add((o.id, b))
        elif (o.id, b) not in self.rec.snap and (o.id, b) not in self.rec.new_set:
            self.rec.snap[(o.id, b)] = (st.delta[s].copy(), st.mask[s].copy(), float(st.Lb[s]))
        self._touched.add(b)
        return s

    def _finish(self, scene, o, redist):
        from ..core import brush as BR
        if not self._touched:
            return
        if redist:
            try:
                BR.redistance(scene, o.id, sorted(self._touched), self.rec)
            except Exception as e:                 # re-distance is a quality step: the dial stands without it
                self.report = dict(getattr(self, 'report', {}), redistance_error=str(e))
        scene.touch(structural=True, index=True)
        BR.measure_bricks(scene, scene.index_of(o.id), set(self._touched))
        scene.refresh_object(scene.index_of(o.id))

    def _undo_bricks(self, scene):
        from ..core import brush as BR
        if getattr(self, 'rec', None) is not None:
            BR.undo_record(scene, self.rec)
        scene.touch(structural=True, index=True)

    def nbytes(self):
        r = getattr(self, 'rec', None)
        return 256 + (r.nbytes() if r is not None else 0)


@command
class LayerDial(_BrickWrite):
    """a layer's strength (-1 to 2) and / or on / off: the composite gets (new - old) x the layer over its bricks"""
    label = 'Layer strength'

    def __init__(self, oid, lid, strength=None, on=None):
        self.oid, self.lid = int(oid), int(lid)
        self.strength = None if strength is None else float(strength)
        self.on = None if on is None else bool(on)

    def args(self):
        return {'oid': self.oid, 'lid': self.lid, 'strength': self.strength, 'on': self.on}

    def do(self, scene):
        o = scene.get(self.oid)
        L = _layer(o, self.lid)
        if L.get('recording'):
            raise ValueError('stop recording the layer before dialling it')
        if self.strength is not None and not (-1.0 <= self.strength <= 2.0):
            raise ValueError('a layer strength is between -1 and 2')
        self._old = copy.deepcopy(L)
        e0 = eff(L)
        if self.strength is not None:
            L['strength'] = self.strength
        if self.on is not None:
            L['on'] = self.on
        d = eff(L) - e0
        self._begin(scene)
        self.report = {'d': d, 'bricks': 0}
        if d != 0.0 and L.get('sha'):
            A = BK.read_arrays(L.get('file'), L['sha'])
            keys, D = A['keys'], A['delta']
            st = scene.store
            for b, dl in zip(map(tuple, keys.tolist()), D):
                s = self._slot(scene, o, b)
                st.delta[s] = st.delta[s] + np.float32(d) * dl
            big = float(np.max(np.abs(D))) * abs(d) if len(D) else 0.0
            self.report = {'d': d, 'bricks': len(keys), 'max_change_m': big}
            self._finish(scene, o, redist=big > o.sculpt.h)
        scene.touch(structural=True)
        return [self.oid]

    def undo(self, scene):
        self._undo_bricks(scene)
        L = _layer(scene.get(self.oid), self.lid)
        L.clear()
        L.update(self._old)
        scene.touch(structural=True)
        return [self.oid]


@command
class LayerDelete(_BrickWrite):
    """a layer gone: its contribution taken out of the composite (merge=False), or folded into the base (merge=True:
    the composite unchanged, the layer's record dropped)"""
    label = 'Delete layer'

    def __init__(self, oid, lid, merge=False):
        self.oid, self.lid, self.merge = int(oid), int(lid), bool(merge)

    def args(self):
        return {'oid': self.oid, 'lid': self.lid, 'merge': self.merge}

    def do(self, scene):
        o = scene.get(self.oid)
        L = _layer(o, self.lid)
        self._old_layers = copy.deepcopy(o.layers)
        self._begin(scene)
        if not self.merge and L.get('sha') and eff(L) != 0.0:
            A = BK.read_arrays(L.get('file'), L['sha'])
            st = scene.store
            d = -eff(L)
            for b, dl in zip(map(tuple, A['keys'].tolist()), A['delta']):
                s = self._slot(scene, o, b)
                st.delta[s] = st.delta[s] + np.float32(d) * dl
            big = float(np.max(np.abs(A['delta']))) * abs(d) if len(A['delta']) else 0.0
            self._finish(scene, o, redist=big > o.sculpt.h)
        o.layers = [x for x in o.layers if int(x['id']) != self.lid]
        scene.touch(structural=True)
        return [self.oid]

    def undo(self, scene):
        self._undo_bricks(scene)
        scene.get(self.oid).layers = copy.deepcopy(self._old_layers)
        scene.touch(structural=True)
        return [self.oid]


@command
class SnapshotTake(Command):
    """a named state of the object's bricks (delta and mask), stored by the sha of its bytes"""
    label = 'Snapshot'

    def __init__(self, oid, name=None, sha=None, file=None):
        self.oid = int(oid)
        self.name, self.sha, self.file = name, sha, file

    def args(self):
        return {'oid': self.oid, 'name': self.name, 'sha': self.sha, 'file': self.file}

    def do(self, scene):
        o = scene.get(self.oid)
        st = scene.store
        if self.sha is None:
            keys = sorted(b for b, s in o.sculpt.slots.items() if st.delta[s].any() or st.mask[s].any())
            K = np.array(keys, np.int64).reshape(-1, 3)
            sl = [o.sculpt.slots[b] for b in keys]
            D = st.delta[sl].copy() if sl else np.zeros((0, B, B, B), np.float32)
            M = st.mask[sl].copy() if sl else np.zeros((0, B, B, B), np.float32)
            r = BK.write_arrays(keys=K, delta=D, mask=M)
            self.sha, self.file = r['sha'], r['file']
            self.bricks = len(keys)
        if not self.name:
            self.name = 'Snapshot %d' % (len(o.snapshots) + 1)
        if any(x['name'] == self.name for x in o.snapshots):
            raise ValueError('a snapshot named %r exists on %s' % (self.name, o.name))
        o.snapshots.append({'name': self.name, 'sha': self.sha, 'file': self.file,
                            'bricks': int(getattr(self, 'bricks', 0))})
        scene.touch(structural=True)
        return [self.oid]

    def undo(self, scene):
        o = scene.get(self.oid)
        o.snapshots = [x for x in o.snapshots if x['name'] != self.name]
        scene.touch(structural=True)
        return [self.oid]


@command
class SnapshotRestore(_BrickWrite):
    """back to a snapshot: everywhere (delta and mask as they were; bricks added since go to zero), or only where the
    user masked (the delta blended toward the snapshot by the mask; the mask kept)"""
    label = 'Restore snapshot'

    def __init__(self, oid, name, masked=False):
        self.oid, self.name, self.masked = int(oid), str(name), bool(masked)

    def args(self):
        return {'oid': self.oid, 'name': self.name, 'masked': self.masked}

    def do(self, scene):
        o = scene.get(self.oid)
        S = next((x for x in o.snapshots if x['name'] == self.name), None)
        if S is None:
            raise ValueError('%s has no snapshot %r' % (o.name, self.name))
        A = BK.read_arrays(S.get('file'), S['sha'])
        snap = {tuple(b): i for i, b in enumerate(A['keys'].tolist())}
        st = scene.store
        self._begin(scene)
        bricks = set(snap) | set(o.sculpt.slots)
        inv = bool(getattr(o, 'mask_inv', False))
        big = 0.0
        for b in sorted(bricks):
            i = snap.get(b)
            tgt_d = A['delta'][i] if i is not None else np.zeros((B, B, B), np.float32)
            tgt_m = A['mask'][i] if i is not None else np.zeros((B, B, B), np.float32)
            s0 = o.sculpt.get_slot(b)
            cur_d = st.delta[s0] if s0 >= 0 else np.zeros((B, B, B), np.float32)
            if self.masked:
                m = st.mask[s0] if s0 >= 0 else np.zeros((B, B, B), np.float32)
                m = (1.0 - m) if inv else m
                if not np.any(m > 0) or np.array_equal(cur_d, tgt_d):
                    continue
                s = self._slot(scene, o, b)
                new = (cur_d + m * (tgt_d - cur_d)).astype(np.float32)
                big = max(big, float(np.max(np.abs(new - cur_d))))
                st.delta[s] = new
            else:
                cur_m = st.mask[s0] if s0 >= 0 else np.zeros((B, B, B), np.float32)
                if np.array_equal(cur_d, tgt_d) and np.array_equal(cur_m, tgt_m):
                    continue
                s = self._slot(scene, o, b)
                big = max(big, float(np.max(np.abs(tgt_d - cur_d))))
                st.delta[s] = tgt_d
                st.mask[s] = tgt_m
        self.report = {'bricks': len(self._touched), 'max_change_m': big}
        self._finish(scene, o, redist=False)
        scene.touch(structural=True)
        return [self.oid]

    def undo(self, scene):
        self._undo_bricks(scene)
        scene.touch(structural=True)
        return [self.oid]


@command
class SnapshotDelete(Command):
    label = 'Delete snapshot'

    def __init__(self, oid, name):
        self.oid, self.name = int(oid), str(name)

    def args(self):
        return {'oid': self.oid, 'name': self.name}

    def do(self, scene):
        o = scene.get(self.oid)
        self._old = copy.deepcopy(o.snapshots)
        o.snapshots = [x for x in o.snapshots if x['name'] != self.name]
        scene.touch(structural=True)
        return [self.oid]

    def undo(self, scene):
        scene.get(self.oid).snapshots = copy.deepcopy(self._old)
        scene.touch(structural=True)
        return [self.oid]


def stop_recording(hist, oid):
    """on the engine thread: the recording layer's capture -> a LayerSet to push (or None when none records)"""
    o = hist.scene.get(oid)
    L = next((x for x in o.layers if x.get('recording')), None)
    if L is None:
        return None
    got = capture(hist, oid, L['id'])
    if got is None:
        return None
    keys, D = got
    r = BK.write_arrays(keys=keys, delta=D)
    return LayerSet(oid, L['id'], r['sha'], r['file'], len(keys))
