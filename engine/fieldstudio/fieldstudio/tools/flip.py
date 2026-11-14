"""Flip and mirror on any axis (fs-shell, 2026-11-11; design: beside dragging to rotate and scale, a quick
flip in place on X / Y / Z and a mirror-copy across any axis through the world origin or the object's own centre,
one undo step each, usable with the gizmo up).

Both are exact on a field: a reflection is an isometry, so the object's tree gets a 'transform' node with scale -1 on
the axis, and its sculpt layer's nodes are mirrored across the same local plane (node i -> -i on that axis, delta, mask
and the mask's paint direction alike).
  FlipObject(oid, axis)                     in place, on the object's OWN axis (the gizmo's), through its centre; the
                                            object keeps its id, name, place and transform
  MirrorCopy(oid, axis, through)            a mirrored copy: through='world' across the world plane axis = 0 (a body's
                                            midline for X), through='centre' across the object's own axis plane
                                            through its centre (an overlapping copy to move away)
MirrorDuplicate (core/combine.py, Mirror X) is unchanged, so older logs replay as they were."""
import copy

import numpy as np

from ..core import ops
from ..core.combine import _ObjectSwap, _frame, _mat_to_euler, _measure_all, _renumber
from ..core.history import command, release_object_slots
from ..core.scene import B, SceneObject

AXES = {'x': 0, 'y': 1, 'z': 2}


def _axis(a):
    a = str(a).lower()
    if a not in AXES:
        raise ValueError('axis %r (x, y or z)' % a)
    return AXES[a]


def mirror_bricks(store, slots, ax):
    """{brick: slot} -> {brick: (delta, mask, mdir, rid)} with node n -> -n on axis ax (exact: each node one source).
    Node n = b * B + l: -n = (-b - 1) * B + (B - l) for l >= 1, and (-b) * B + 0 for l = 0."""
    out = {}

    def tgt(b, dv):
        tb = list(b)
        tb[ax] = dv
        tb = tuple(tb)
        if tb not in out:
            out[tb] = (np.zeros((B, B, B), np.float32), np.zeros((B, B, B), np.float32),
                       np.zeros((B, B, B, 3), np.int8), np.zeros((B, B, B), np.uint16))
        return out[tb]

    sl = [slice(None)] * 3
    for b, s in slots.items():
        D, Mk, Md, Rd = store.delta[s], store.mask[s], store.mdir[s].copy(), store.rid[s]
        Md[..., ax] = -Md[..., ax]                         # the paint direction's component on the axis flips too
        t1 = tgt(b, -b[ax] - 1)
        t0 = tgt(b, -b[ax])
        dst, src = list(sl), list(sl)
        dst[ax], src[ax] = slice(1, B), slice(B - 1, 0, -1)    # new local l = 1..B-1 <- old B-l
        for k, arr in enumerate((D, Mk, Md, Rd)):
            t1[k][tuple(dst)] = arr[tuple(src)]
        dst[ax], src[ax] = slice(0, 1), slice(0, 1)            # new local 0 <- old 0
        for k, arr in enumerate((D, Mk, Md, Rd)):
            t0[k][tuple(dst)] = arr[tuple(src)]
    return out


def _mirrored_object(scene, o, ax, oid, name, transform, label=None):
    """a new SceneObject: o's tree and sculpt mirrored on its local axis ax, with the given id, name and transform"""
    sub = _renumber(o.base, scene)
    sc = [1.0, 1.0, 1.0]
    sc[ax] = -1.0
    base = ops.Node('transform', {'translate': [0.0, 0.0, 0.0], 'rotate': [0.0, 0.0, 0.0], 'scale': sc}, [sub],
                    nid=scene.new_id(), name=label or 'mirror %s' % 'xyz'[ax])
    M = SceneObject(oid, name, base, transform=transform, colour=copy.deepcopy(o.colour),
                    material=copy.deepcopy(o.material), h=o.sculpt.h)
    M.detail = copy.deepcopy(o.detail)
    M.mask_inv = o.mask_inv
    for k in ('locked', 'root', 'part', 'garment'):
        if getattr(o, k, None) is not None:
            setattr(M, k, copy.deepcopy(getattr(o, k)))
    st = scene.store
    mb = mirror_bricks(st, o.sculpt.slots, ax)
    keys = sorted(b for b, (d, m, md, r) in mb.items() if d.any() or m.any() or md.any() or r.any())
    M.regions = {k: dict(v) for k, v in getattr(o, 'regions', {}).items()}
    if keys:
        arr = np.array(keys, np.int64)
        M.sculpt.ensure_index(arr.min(0), arr.max(0))
        for b in keys:
            s = st.alloc()
            st.delta[s], st.mask[s], st.mdir[s], st.rid[s] = mb[b]
            M.sculpt.set_slot(b, s)
    return M


@command
class FlipObject(_ObjectSwap):
    """Flip an object in place on its own X, Y or Z (the gizmo's axes), through its centre. Exact; one undo step."""
    label = 'Flip'

    def __init__(self, oid, axis='x'):
        self.oid = int(oid)
        self.axis = str(axis).lower()
        self.obj = None

    def args(self):
        return {'oid': self.oid, 'axis': self.axis}

    def nbytes(self):
        n = len(self.obj.sculpt.slots) if self.obj is not None else 0
        return 256 + n * 2 * 4 * B ** 3

    def discard(self, scene, state):
        if state == 'undone' and self.obj is not None:
            release_object_slots(scene, self.obj)
        elif state == 'committed':
            for o in getattr(self, 'remove_objs', []):
                release_object_slots(scene, o)

    def do(self, scene):
        if self.obj is not None:
            self._apply(scene)
            return [self.obj.id]
        ax = _axis(self.axis)
        o = scene.get(self.oid)
        self.prev_next = scene.next_id
        self.obj = _mirrored_object(scene, o, ax, o.id, o.name, copy.deepcopy(o.transform),
                                    label='flip %s' % 'xyz'[ax].upper())
        self.after_next = scene.next_id
        self.remove_objs = [o]
        self.insert_at = scene.index_of(o.id)
        self._apply(scene)
        _measure_all(scene, scene.index_of(self.obj.id))
        return [self.obj.id]                              # the engine's 'push' keeps it selected


@command
class MirrorCopy(_ObjectSwap):
    """A mirrored copy across X, Y or Z: through='world' (the world plane axis = 0) or 'centre' (the object's own axis
    plane through its centre). Exact; one undo step."""
    label = 'Mirror copy'

    def __init__(self, oid, axis='x', through='world', name=None):
        self.oid = int(oid)
        self.axis = str(axis).lower()
        self.through = str(through)
        self.name = name
        self.obj = None

    def args(self):
        d = {'oid': self.oid, 'axis': self.axis, 'through': self.through}
        if self.name is not None:
            d['name'] = self.name
        return d

    def nbytes(self):
        n = len(self.obj.sculpt.slots) if self.obj is not None else 0
        return 256 + n * 2 * 4 * B ** 3

    def discard(self, scene, state):
        if state == 'undone' and self.obj is not None:
            release_object_slots(scene, self.obj)

    def do(self, scene):
        if self.obj is not None:
            self._apply(scene)
            return [self.obj.id]
        ax = _axis(self.axis)
        if self.through not in ('world', 'centre'):
            raise ValueError("through %r ('world' or 'centre')" % self.through)
        o = scene.get(self.oid)
        self.prev_next = scene.next_id
        mid = scene.new_id()
        nm = self.name or '%s mirror %s' % (o.name, 'xyz'[ax].upper())
        if self.through == 'centre':
            # its own axis through its centre: the local axis mirrored, the transform as it is
            M = _mirrored_object(scene, o, ax, mid, nm, copy.deepcopy(o.transform),
                                 label='mirror on its own %s' % 'xyz'[ax].upper())
        else:
            # the world plane axis = 0: world' = S (R s q + t); with the local x mirrored, R' = S R Sx, t' = S t
            S = np.eye(3)
            S[ax, ax] = -1.0
            Sx = np.diag([-1.0, 1.0, 1.0])
            R, t, s = _frame(o.transform)
            tr = {'translate': [float(v) + 0.0 for v in S @ t], 'rotate': _mat_to_euler(S @ R @ Sx), 'scale': float(s)}
            M = _mirrored_object(scene, o, 0, mid, nm, tr, label='mirror across world %s' % 'xyz'[ax].upper())
        self.obj = M
        self.after_next = scene.next_id
        self.remove_objs = []
        self.insert_at = scene.index_of(o.id) + 1
        self._apply(scene)
        _measure_all(scene, scene.index_of(M.id))
        return [M.id]                                     # the engine's 'push' selects the copy
