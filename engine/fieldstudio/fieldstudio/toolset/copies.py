"""Duplicate / Mirror / Array (TOOLSET_DESIGN 3.1, approved 11-14).

- Duplicate: core/combine.DuplicateObject (the sculpt bricks copied exactly); the window drops it with Place on surface.
- MirrorAcross(oid, plane): a mirrored copy across a world plane (x, y or z = offset) or the object's own plane through
  its origin (object_x / _y / _z). (Named apart from fs-shell's tools/flip.MirrorCopy: one name, one command in the
  replay registry.) Flip in place is tools/flip.FlipObject (job tools.flip). EXACT: the copy's base is the original under a reflection and its sculpt bricks are
  reflected node for node (no resampling, no seam).
- ArrayCopies(oid, mode, count, ...): many copies along a line (a step), around an axis (an angle), along a drawn path
  (equal spacing), or scattered over another object's surface at a spacing, each sitting on it turned to its facing;
  with a library material, Vary also gives each copy its own look and form from the material's generators
  (fs-materials' reseed); Vary (a seed) gives each copy its own size and turn, so "five varied bricks" are five bricks of one
  stone, not five clones. Copies are separate objects inheriting material, label and sculpt (copied exactly).
"""
import copy
import math

import numpy as np

from ..core import ops
from ..core.history import Command, command, release_object_slots
from ..core.scene import SceneObject
from ..core.combine import _frame, _mat_to_euler, _renumber, _measure_all
from . import common as C

B = C.B
AX = {'x': 0, 'y': 1, 'z': 2}


def _flip_bricks(store, slots, axis, K):
    """{brick: slot} -> {brick: (delta, mask, rid)} with node index n_axis -> K - n_axis (exact: each node one source)"""
    out = {}
    for b, s in slots.items():
        D, Mk, R = store.delta[s], store.mask[s], store.rid[s]
        for l in range(B):
            n = b[axis] * B + l
            m = K - n
            tb = list(b)
            tb[axis] = m // B
            tb = tuple(tb)
            if tb not in out:
                out[tb] = (np.zeros((B, B, B), np.float32), np.zeros((B, B, B), np.float32),
                           np.zeros((B, B, B), np.uint16))
            sl_src = [slice(None)] * 3
            sl_src[axis] = l
            sl_dst = [slice(None)] * 3
            sl_dst[axis] = m % B
            out[tb][0][tuple(sl_dst)] = D[tuple(sl_src)]
            out[tb][1][tuple(sl_dst)] = Mk[tuple(sl_src)]
            out[tb][2][tuple(sl_dst)] = R[tuple(sl_src)]
    return out


def _copy_object(scene, o, transform, base=None, bricks=None, name=None):
    """a new object: o's base (renumbered, or `base`), its bricks copied (or `bricks` {key: (d, m, r)}), at transform"""
    M = SceneObject(scene.new_id(), name or o.name + ' copy', base if base is not None else _renumber(o.base, scene),
                    transform=transform, colour=copy.deepcopy(o.colour), material=copy.deepcopy(o.material),
                    h=o.sculpt.h)
    M.detail = copy.deepcopy(o.detail)
    M.mask_inv = o.mask_inv
    M.regions = {k: dict(v) for k, v in getattr(o, 'regions', {}).items()}
    st = scene.store
    if bricks is None:
        keys = sorted(o.sculpt.slots)
        if keys:
            sl = st.alloc_many(len(keys))
            for k, s in zip(keys, sl):
                s0 = o.sculpt.slots[k]
                st.delta[s] = st.delta[s0]
                if st.has('mask'):
                    st.mask[s] = st.mask[s0]
                if st.has('mdir'):
                    st.mdir[s] = st.mdir[s0]
                if st.has('rid'):
                    st.rid[s] = st.rid[s0]
                st.Lb[s] = st.Lb[s0]
            M.sculpt.set_slots(keys, sl)
    else:
        keys = sorted(k for k, (d, m, r) in bricks.items() if d.any() or m.any() or r.any())
        if keys:
            sl = st.alloc_many(len(keys))
            for k, s in zip(keys, sl):
                st.delta[s] = bricks[k][0]
                st.mask[s] = bricks[k][1]
                st.rid[s] = bricks[k][2]
            M.sculpt.set_slots(keys, sl)
    return M


def _reseed(scene, o, M, seed):
    """Vary's material side (fs-materials' materials.commands.reseed): the copy's own look, colour and (an unsculpted
    copy) its form's generators re-run with its seed; a crack stays the original's. Nothing for a plain material."""
    try:
        from ..materials.commands import reseed
    except Exception:
        return False
    r = reseed(scene, o.id, int(seed))
    if not r:
        return False
    M.material = r['material']
    M.colour = r['colour']
    if not o.sculpt.slots and r.get('tree') is not None:
        t = r['tree']
        M.base = t if isinstance(t, ops.Node) else ops.Node.from_dict(t)
        M.invalidate()
    scene.next_id = max(scene.next_id, int(r.get('next_id') or 0))
    return True


class _Adds(Command):
    """bookkeeping for commands that add objects: undo removes them (kept for an exact redo)"""

    def _insert(self, scene, objs, at):
        for i, M in enumerate(objs):
            scene.objects.insert(min(at + i, len(scene.objects)), M)
        scene.touch(structural=True, index=True)
        for M in objs:
            _measure_all(scene, scene.index_of(M.id))

    def undo(self, scene):
        for M in self.objs:
            scene.objects.remove(M)
        scene.next_id = self.prev_next
        scene.touch(structural=True, index=True)

    def _redo(self, scene):
        self._insert(scene, self.objs, self.at)
        scene.next_id = self.after_next
        return [M.id for M in self.objs]

    def discard(self, scene, state):
        if state == 'undone':
            for M in getattr(self, 'objs', []):
                release_object_slots(scene, M)

    def undo_state(self):
        """the undo journal: the ids it added and the id counter before (the objects are in the saved scene)"""
        if not getattr(self, 'objs', None):
            return None
        meta = {'ids': [int(M.id) for M in self.objs], 'prev_next': int(self.prev_next), 'at': int(self.at),
                'after_next': int(self.after_next)}
        return lambda: (meta, {})

    def restore_undo_state(self, scene, meta, A):
        self.objs = [scene.get(i) for i in meta['ids']]
        self.prev_next, self.at, self.after_next = meta['prev_next'], meta['at'], meta['after_next']

    def nbytes(self):
        return 256 + sum(len(M.sculpt.slots) for M in getattr(self, 'objs', [])) * 2 * 4 * B ** 3


@command
class MirrorAcross(_Adds):
    """plane: 'world_x' | 'world_y' | 'world_z' (with offset, m: the plane x = offset ...), or 'object_x' | 'object_y' |
    'object_z' (the object's own plane through its origin)"""
    label = 'Mirror'

    def __init__(self, oid, plane='world_x', offset=0.0, name=None):
        self.oid = int(oid)
        self.plane = str(plane)
        self.offset = float(offset)
        self.name = name
        self.objs = None

    def args(self):
        d = {'oid': self.oid, 'plane': self.plane, 'offset': self.offset}
        if self.name is not None:
            d['name'] = self.name
        return d

    def do(self, scene):
        if self.objs is not None:
            return self._redo(scene)
        o = scene.get(self.oid)
        self.prev_next = scene.next_id
        kind, ax = self.plane.split('_')
        a = AX[ax]
        S = np.eye(3)
        S[a, a] = -1.0
        R, t, s = _frame(o.transform)
        if kind == 'world':
            # reflect across the world plane: the copy's frame is the conjugate S R S, its local axis a reflected
            # (exact when the plane is world-aligned: proof in core/combine.MirrorDuplicate's for x)
            R2 = S @ R @ S
            e = np.zeros(3)
            e[a] = 1.0
            t2 = S @ t + 2.0 * self.offset * e
            tr = {'translate': [float(v) for v in t2], 'rotate': _mat_to_euler(R2), 'scale': float(s)}
        elif kind == 'object':
            tr = copy.deepcopy(o.transform)
        else:
            raise ValueError(self.plane)
        sub = _renumber(o.base, scene)
        sc = [1.0, 1.0, 1.0]
        sc[a] = -1.0
        base = ops.Node('transform', {'translate': [0.0, 0.0, 0.0], 'rotate': [0.0, 0.0, 0.0], 'scale': sc}, [sub],
                        nid=scene.new_id(), name='mirror %s' % ax)
        if kind == 'world':
            # the copy's local axis a = the original's local axis a reflected only if R maps it to the world plane's
            # normal; the conjugate frame makes the copy's local reflection S equal the world one in every case
            pass
        mb = _flip_bricks(scene.store, o.sculpt.slots, a, 0)
        M = _copy_object(scene, o, tr, base=base, bricks=mb, name=self.name or o.name + ' mirror')
        self.objs = [M]
        self.at = scene.index_of(o.id) + 1
        self._insert(scene, self.objs, self.at)
        self.after_next = scene.next_id
        return [M.id]


@command
class ArrayCopies(_Adds):
    """mode 'line': step (world m, per copy); 'radial': axis, centre (world), angle (deg, the whole turn the copies
    span: 360 = all the way round); 'path': points (world polyline), equal spacing along it. count = copies made
    (the original stays). vary: {'seed', 'size' (fraction), 'turn' (deg)}: each copy's own scale and turn."""
    label = 'Array'

    def __init__(self, oid, mode='line', count=4, step=None, axis=None, centre=None, angle=360.0, points=None,
                 vary=None, name=None, on=None, spacing=0.05, up=None):
        self.oid = int(oid)
        self.mode = str(mode)
        self.count = int(count)
        self.step = None if step is None else [float(v) for v in step]
        self.axis = None if axis is None else [float(v) for v in axis]
        self.centre = None if centre is None else [float(v) for v in centre]
        self.angle = float(angle)
        self.points = None if points is None else [[float(v) for v in p] for p in points]
        self.vary = copy.deepcopy(vary) if vary else None
        self.name = name
        self.on = None if on is None else int(on)
        self.spacing = float(spacing)
        self.up = None if up is None else [float(v) for v in up]
        self.objs = None

    def args(self):
        d = {'oid': self.oid, 'mode': self.mode, 'count': self.count, 'angle': self.angle}
        if self.mode == 'scatter':
            d.update({'on': self.on, 'spacing': self.spacing})
        for k in ('step', 'axis', 'centre', 'points', 'vary', 'name', 'up'):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d

    def placements(self, scene):
        """[(R 3x3, t 3, s)] world frames of the copies (before Vary)"""
        o = scene.get(self.oid)
        R, t, s = _frame(o.transform)
        out = []
        if self.mode == 'line':
            st = np.asarray(self.step if self.step is not None else [0.1, 0.0, 0.0], float)
            for i in range(1, self.count + 1):
                out.append((R, t + i * st, s))
        elif self.mode == 'radial':
            ax = np.asarray(self.axis or [0.0, 1.0, 0.0], float)
            ax /= np.linalg.norm(ax)
            c = np.asarray(self.centre if self.centre is not None else [0.0, 0.0, 0.0], float)
            n = self.count + 1
            full = abs(self.angle - 360.0) < 1e-9
            dth = math.radians(self.angle) / (n if full else max(n - 1, 1))
            for i in range(1, self.count + 1):
                th = i * dth
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                Q = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
                out.append((Q @ R, Q @ (t - c) + c, s))
        elif self.mode == 'path':
            P = np.asarray(self.points, float)
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            L = np.concatenate([[0.0], np.cumsum(seg)])
            for i in range(1, self.count + 1):
                u = L[-1] * i / self.count
                k = min(int(np.searchsorted(L, u, side='right') - 1), len(seg) - 1)
                f = (u - L[k]) / max(seg[k], 1e-12)
                p = P[k] + f * (P[k + 1] - P[k])
                out.append((R, t + (p - P[0]), s))
        elif self.mode == 'scatter':
            out = self._scatter(scene, o, R, s)
        else:
            raise ValueError(self.mode)
        return out

    def _scatter(self, scene, o, R0, s0):
        """copies scattered over another object's surface (self.on) at a spacing, each sitting on it and turned to its
        facing (Place on surface's frame): seeded, deterministic; points closer than the spacing are dropped"""
        from .place import Placer
        tgt = scene.get(int(self.on))
        lo, hi = C.world_box(tgt)
        rng = np.random.default_rng(int((self.vary or {}).get('seed', 0)) * 7919 + 1)
        from ..warp.bake import object_field
        f = object_field(scene, tgt)
        P = rng.uniform(lo, hi, (40 * max(self.count, 1), 3))
        for _ in range(4):                                   # onto the surface (Newton along the gradient)
            e = 1e-5
            g = np.stack([(f(P + e * np.eye(3)[a]) - f(P - e * np.eye(3)[a])) / (2 * e) for a in range(3)], 1)
            P = P - (f(P) / np.maximum((g * g).sum(1), 1e-12))[:, None] * g
        ok = np.abs(f(P)) < 1e-6
        P, g = P[ok], g[ok]
        if self.up is not None:                              # only the side facing `up` (a table's top)
            up = np.asarray(self.up, float)
            k = (g @ up) / np.maximum(np.linalg.norm(g, axis=1), 1e-12) > 0.5
            P, g = P[k], g[k]
        keep = []
        for i in range(len(P)):                              # Poisson-disk: at least `spacing` apart
            if all(np.linalg.norm(P[i] - P[j]) >= self.spacing for j in keep):
                keep.append(i)
            if len(keep) >= self.count:
                break
        pl = Placer(scene, self.oid)
        out = []
        self._normals = []
        for i in keep:
            n = g[i] / np.linalg.norm(g[i])
            R, t = pl.frame(P[i], n, 'surface', 0.0, float(rng.uniform(0.0, 2 * math.pi)))
            out.append((R, t, s0))
            self._normals.append(n)
        return out

    def do(self, scene):
        if self.objs is not None:
            return self._redo(scene)
        o = scene.get(self.oid)
        self.prev_next = scene.next_id
        v = self.vary or {}
        seed = int(v.get('seed', 0))
        vs = float(v.get('size', 0.0))
        vt = math.radians(float(v.get('turn', 0.0)))
        lo, hi = o.local_box()
        cl = 0.5 * (lo + hi)                                   # vary about the object's own centre
        objs = []
        for i, (R, t, s) in enumerate(self.placements(scene)):
            if seed and (vs or vt):
                rng = np.random.default_rng(seed * 100003 + i)
                u1, u2 = rng.uniform(-1.0, 1.0, 2)
                f = 1.0 + vs * u1
                th = vt * u2
                Q = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
                c_w = R @ (cl * s) + t                         # the copy's centre stays where it was
                R = R @ Q
                s = s * f
                t = c_w - R @ (cl * s)
            tr = {'translate': [float(x) for x in t], 'rotate': _mat_to_euler(R), 'scale': float(s)}
            M = _copy_object(scene, o, tr, name=(self.name or o.name) + ' %d' % (i + 1))
            if seed:
                _reseed(scene, o, M, seed * 100003 + i)
            objs.append(M)
        self.objs = objs
        self.at = scene.index_of(o.id) + 1
        self._insert(scene, objs, self.at)
        if self.mode == 'scatter':
            # each copy, after its Vary, set down on the surface by the exact contact (Place on surface's drop)
            from .place import Placer
            for M, n in zip(objs, self._normals):
                pl = Placer(scene, M.id)
                Rm = ops.rot_matrix(M.transform['rotate'])
                M.transform = pl.drop(Rm, np.asarray(M.transform['translate'], float), n)
            scene.touch(structural=True)
        self.after_next = scene.next_id
        return [M.id for M in objs]
