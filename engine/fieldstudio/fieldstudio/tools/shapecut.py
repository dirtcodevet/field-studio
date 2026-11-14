"""CUT WITH A SHAPE (fs-shell, 2026-11-11; design 4.18, the cut tool, 11-10: choose a shape (a sphere,
a cube or any other), then choose what is cut away (the interior or the exterior), then delete it or paste it
elsewhere).

A cutter is any object (a primitive placed with the gizmo: sphere, box, capsule, cylinder, cone, torus, slab, with its
stretch and gizmo deformers) applied to one or more target objects, exactly (a min / max of two distances, no
triangles, P14), in ONE undo step; the cutter object is used up.
  side 'inside'  : what lies inside the cutter goes (a hole); target := target - cutter.
  side 'outside' : what lies outside the cutter goes (a trim); target := target AND cutter.
  result 'delete': the part that goes is dropped.
  result 'lift'  : the part that goes becomes a new object (its colour, material and a copy of the sculpt layer),
                   moved by `distance` along `offset_dir` (default: from the target's centre toward the cutter's).
  result 'keep'  : as 'lift' with no move: both pieces stay, their cut faces touching.
The cutter enters each target's local frame as a 'transform' node over a renumbered copy of its base tree
(common.relative_transform, as CombineObjects places B in A's frame). Like Cut, the target's sculpt layer is kept as it
is (it stays a delta on the new base). Conform with a wall (4.18's second row) needs regions (chunk 7): not here."""
import copy

import numpy as np

from ..core import ops
from ..core.history import Command, command
from ..core.scene import SceneObject
from . import common as C
from .cut import _has_material

SIDES = ('inside', 'outside')
RESULTS = ('delete', 'lift', 'keep')


def is_cutter(obj_or_dict):
    """a cutter object: named 'Cutter ...' by Create > Cutter"""
    name = obj_or_dict['name'] if isinstance(obj_or_dict, dict) else getattr(obj_or_dict, 'name', '')
    return str(name).startswith('Cutter')


def _bool(kind, a, b, k):
    """the program's boolean of a (the object) and b (the cutter) as OP_SUB / OP_INTER (and their smooth forms) do it"""
    if k <= 1e-9:
        return np.maximum(a, -b) if kind == 'subtract' else np.maximum(a, b)

    def smin(x, y):
        h = np.maximum(k - np.abs(x - y), 0.0) / k
        return np.minimum(x, y) - h * h * k * 0.25
    return -smin(-a, b) if kind == 'subtract' else -smin(-a, -b)


def _cut_sculpt(scene, o, base_tree, cutter_node, kind, k, keep_old=None, tree_k=None):
    """a cut of a SCULPTED object: the sculpt layer is a delta added after the tree, so where its band reaches into the
    cut it would put material back. At every node of every brick: delta := op(F_old, C) - op(base_old, C) (F_old = the
    old base + delta), so base_new + delta_new = the cut of the full field there; the sculpt is unchanged away from the
    cut and 0 inside it. keep_old: a dict to keep the old deltas in (for the undo). -> bricks changed"""
    from .common import _eval_tree
    from ..core.scene import B
    st = scene.store
    if not o.sculpt.slots:
        return 0
    cb, pb, _l, _h, _L = ops.compile_tree(base_tree)
    cc, pc, _l, _h, _L = ops.compile_tree(cutter_node)
    hh = float(o.sculpt.h)
    l = np.arange(B)
    L3 = np.stack(np.meshgrid(l, l, l, indexing='ij'), -1).reshape(-1, 3)
    n = 0
    for b, slot in list(o.sculpt.slots.items()):
        pts = np.ascontiguousarray((np.asarray(b, np.int64) * B + L3) * hh, dtype=np.float64)
        bv = np.empty(len(pts))
        cv = np.empty(len(pts))
        _eval_tree(cb, pb, pts, bv)
        _eval_tree(cc, pc, pts, cv)
        d0 = st.delta[slot].reshape(-1).astype(np.float64)
        # the total at the node: the cut of the full field, SHARP (k); the tree's own cut (tree_k) may be smooth, so
        # between the nodes the trilinear delta never has to follow a crease
        d1 = _bool(kind, bv + d0, cv, k) - _bool(kind, bv, cv, tree_k if tree_k is not None else k)
        if np.allclose(d1, d0, atol=1e-12):
            continue
        if keep_old is not None:
            keep_old[(o.id, b)] = (slot, st.delta[slot].copy())
        st.delta[slot] = d1.reshape(B, B, B).astype(st.delta.dtype)
        n += 1
    return n


def _cover_bricks(scene, o, base_tree, cutter_node, kind, tree_k, band=0.003, new=None):
    """bricks wherever the tree's SMOOTH cut (tree_k) differs from the sharp cut near the surface, so the sculpt layer
    can make the cut exact there too (a brick-less stretch kept the smooth cut: a visible seam at the bricks' edge, a
    rounded opening rim). new: a list that gets (b, slot) of each brick made (for the undo) -> bricks made"""
    from .common import _eval_tree
    from ..core.scene import B
    if tree_k <= 1e-9:
        return 0
    cc, pc, clo, chi, _L = ops.compile_tree(cutter_node)
    cb, pb, blo, bhi, _L = ops.compile_tree(base_tree)
    lo = np.maximum(np.asarray(clo, float), np.asarray(blo, float)) - tree_k
    hi = np.minimum(np.asarray(chi, float), np.asarray(bhi, float)) + tree_k
    if np.any(hi <= lo):
        return 0
    hh = float(o.sculpt.h)
    n0 = np.floor(lo / hh).astype(np.int64)
    n1 = np.ceil(hi / hh).astype(np.int64)
    ax = [np.arange(n0[a], n1[a] + 1) for a in range(3)]
    st = scene.store
    made = 0
    for i0 in range(0, len(ax[0]), 32):                 # slabs of nodes (memory)
        X = ax[0][i0:i0 + 32]
        G = np.stack(np.meshgrid(X, ax[1], ax[2], indexing='ij'), -1).reshape(-1, 3)
        pts = np.ascontiguousarray(G * hh, dtype=np.float64)
        bv = np.empty(len(pts))
        cv = np.empty(len(pts))
        _eval_tree(cb, pb, pts, bv)
        _eval_tree(cc, pc, pts, cv)
        sharp = _bool(kind, bv, cv, 0.0)
        smooth = _bool(kind, bv, cv, tree_k)
        need = (np.abs(sharp - smooth) > 1e-7) & (np.abs(sharp) < band)
        if not need.any():
            continue
        for b in {tuple(v) for v in np.floor_divide(G[need], B).tolist()}:
            if b in o.sculpt.slots:
                continue
            slot = st.alloc()
            st.delta[slot] = 0.0
            st.mask[slot] = 0.0
            st.mdir[slot] = 0
            o.sculpt.ensure_index(b, b)
            o.sculpt.set_slot(b, slot)
            if new is not None:
                new.append((b, slot))
            made += 1
    return made


@command
class ShapeCut(Command):
    """Cut targets with a cutter object (side inside / outside; result delete / lift / keep)."""
    label = 'Cut with shape'

    def __init__(self, targets, cutter, side='inside', result='delete', distance=0.03, offset_dir=None, blend=0.0):
        self.targets = [int(o) for o in targets]
        self.cutter = int(cutter)
        self.side = str(side)
        self.result = str(result)
        self.distance = float(distance)
        self.offset_dir = None if offset_dir is None else [float(v) for v in offset_dir]
        self.blend = float(blend)

    def args(self):
        return {'targets': self.targets, 'cutter': self.cutter, 'side': self.side, 'result': self.result,
                'distance': self.distance, 'offset_dir': self.offset_dir, 'blend': self.blend}

    def do(self, scene):
        if self.side not in SIDES or self.result not in RESULTS:
            raise ValueError('side must be %s, result %s' % (SIDES, RESULTS))
        if self.cutter in self.targets:
            raise ValueError('the cutter cannot cut itself')
        self._prev_next = scene.next_id
        self._old = {}
        self._pieces = []
        self._placed = {}
        self._piece_cut = {}
        self._old_delta = {}
        self._new_bricks = []
        ids = C.Ids(scene.next_id)
        cut = scene.get(self.cutter)
        cf = C.obj_frame(cut)
        keep_op, piece_op = ('subtract', 'intersect') if self.side == 'inside' else ('intersect', 'subtract')
        new_trees, made = {}, []
        for oid in self.targets:
            o = scene.get(oid)
            self._old[oid] = o.base.to_dict()
            rel = C.relative_transform(C.obj_frame(o), cf)

            def placed():
                return ops.Node('transform', copy.deepcopy(rel), [C.renumber(cut.base, ids)], nid=ids(),
                                name='cutter %s' % cut.name)
            node = placed()
            C.set_data(node, {'tool': 'shapecut', 'side': self.side, 'result': self.result, 'cutter': cut.name})
            # a sculpted target (fs-shell 11-11): the tree's cut smooth over 2 lattice steps where the cut is sharp, so
            # the sculpt layer's 1 mm trilinear never meets a crease (it makes the total the sharp cut at every node)
            tk = max(self.blend, 4.0 * float(o.sculpt.h)) if (o.sculpt.slots and self.blend <= 1e-9) else self.blend
            new_trees[oid] = ops.Node(keep_op, {'blend': tk}, [o.base, node], nid=ids(),
                                      name='Cut (%s)' % self.side)
            self._placed[oid] = (o.base, node, tk)
            if self.result != 'delete':
                pnode = placed()
                piece = ops.Node(piece_op, {'blend': self.blend}, [C.renumber(o.base, ids), pnode], nid=ids(),
                                 name='Cut piece')
                made.append((o, piece))
                self._piece_cut[id(piece)] = (o.base, pnode)
        if self.result == 'lift':
            if self.offset_dir is not None:
                off = C.unit(np.asarray(self.offset_dir, float)) * self.distance
            else:
                d = np.asarray(cut.transform['translate'], float) - np.mean(
                    [scene.get(o).transform['translate'] for o in self.targets], axis=0)
                if np.linalg.norm(d) < 1e-6:
                    d = np.array([1.0, 0.0, 0.0])
                off = C.unit(d) * self.distance
        else:
            off = np.zeros(3)
        for o, piece in made:
            if not _has_material(o, piece):
                continue
            pid = ids()
            t = copy.deepcopy(o.transform)
            t['translate'] = [float(a + b) for a, b in zip(t['translate'], off)]
            p = SceneObject(pid, '%s piece' % o.name, piece, transform=t, colour=copy.deepcopy(o.colour),
                            material=copy.deepcopy(o.material), h=o.sculpt.h)
            p.detail = copy.deepcopy(o.detail)
            slots = C.copy_sculpt(scene, o, p)
            ob, pn = self._piece_cut[id(piece)]
            _cut_sculpt(scene, p, ob, pn, piece_op, self.blend)       # the piece's copied sculpt cut the same way
            self._pieces.append((p, slots))
        for oid, tree in new_trees.items():
            o = scene.get(oid)
            ob, nd, tk = self._placed[oid]
            if tk > self.blend:
                made_ = []
                _cover_bricks(scene, o, ob, nd, keep_op, tk, new=made_)
                self._new_bricks.extend((oid, b, sl) for b, sl in made_)
            _cut_sculpt(scene, o, ob, nd, keep_op, self.blend, self._old_delta, tree_k=tk)   # fs-shell 11-11: sculpted
            o.base = tree
            o.invalidate()
            o.compiled()
        for p, slots in self._pieces:
            scene.objects.append(p)
        # the cutter is used up (kept here for the undo)
        self._cutter_index = scene.index_of(self.cutter)
        self._cutter_obj = cut
        scene.objects.remove(cut)
        scene.next_id = max(scene.next_id, ids.next_id)
        scene.touch(structural=True, index=True)
        if self._old_delta or self._pieces:
            from ..core.combine import _measure_all
            for oi_ in {scene.index_of(oid) for (oid, _b) in self._old_delta} | {scene.index_of(p.id) for p, _s in
                                                                                   self._pieces}:
                if oi_ >= 0:
                    _measure_all(scene, oi_)               # the bricks' Lipschitz bounds after the new deltas
        return [p.id for p, s in self._pieces]

    def undo(self, scene):
        for p, slots in reversed(self._pieces):
            scene.objects.remove(p)
            C.release_sculpt(scene, p, set(slots))
        for (oid, b), (slot, arr) in getattr(self, '_old_delta', {}).items():
            scene.store.delta[slot] = arr
        for oid, b, slot in reversed(getattr(self, '_new_bricks', [])):      # the bricks the cut made
            if scene.index_of(oid) >= 0:
                scene.get(oid).sculpt.set_slot(b, -1)
            scene.store.release(slot)
        for oid, tree in self._old.items():
            o = scene.get(oid)
            o.base = ops.Node.from_dict(tree)
            o.invalidate()
        if getattr(self, '_old_delta', None):
            from ..core.combine import _measure_all
            for oid in {oid for (oid, _b) in self._old_delta}:
                if scene.index_of(oid) >= 0:
                    _measure_all(scene, scene.index_of(oid))
        scene.objects.insert(min(self._cutter_index, len(scene.objects)), self._cutter_obj)
        scene.next_id = self._prev_next
        self._pieces = []
        scene.touch(structural=True, index=True)


# ---- Conform with a wall (design 4.18: hollow out a skull with a 4 mm wall) -----------------------------------------------
CUTTER_COLOUR = [0.86, 0.24, 0.20]


@command
class ConformCutter(Command):
    """a cutter that is the region's inward offset at a wall distance: the volume {x inside the region : depth > wall},
    exactly from the region's field (an 'offset' node over a copy of its tree: F + wall < 0). It is an ordinary cutter
    (named 'Cutter ...', red): the live preview shows what will go, Apply cut (side inside) leaves a wall of that
    thickness everywhere, Lift out keeps the inside as a new object; until then it can be moved like any cutter.
    Its tree is the region's base tree: a sculpt layer on the region is not in it (wall_report measures the wall the
    cut would leave on the full field, sculpt included)."""
    label = 'Conform cutter'

    def __init__(self, target, wall, mode='offset', bake=None, openings=True, open_margin=0.0):
        self.target = int(target)
        self.wall = float(wall)
        self.open_margin = float(open_margin)           # fs-shell 11-11: an opening this much wider than the mask
        # 'offset': an offset of the region's tree (the first version, exact where the field is a distance: primitives,
        # masses, stores); 'true' (fs-shell 11-11): the true distance from the actual surface, sculpt
        # included (tools/truedist), baked as a lattice (tools/bakes) whose file is recorded here so a replay loads it
        self.mode = str(mode)
        self.bake = None if bake is None else dict(bake)
        self.openings = bool(openings)

    def args(self):
        d = {'target': self.target, 'wall': self.wall}
        if self.mode != 'offset':                    # older logs (no mode) replay as the offset they were
            d.update({'mode': self.mode, 'bake': self.bake, 'openings': self.openings})
            if self.open_margin:
                d['open_margin'] = self.open_margin
        return d

    def do(self, scene):
        if self.wall <= 0:
            raise ValueError('the wall must be thicker than 0')
        self._prev_next = scene.next_id
        o = scene.get(self.target)
        ids = C.Ids(scene.next_id)
        cid = ids()
        if self.mode == 'auto':
            # the field is a distance through the wall on an analytic object with no sculpt layer (an offset of its
            # tree is exact there, sharp creases included); a sculpt layer, a store or a lattice needs the true depth
            kinds = {n.kind for n in o.base.walk()}
            self.mode = 'true' if (o.sculpt.slots or kinds & {'fv8store', 'lattice', 'sampled'}) else 'offset'
        if self.mode == 'true':
            if self.bake is None:
                from . import truedist as TD, bakes as BK
                G = TD.conform_local(scene, self.target, self.wall, openings=self.openings,
                                     open_margin=self.open_margin)                          # its own lattice
                self.bake = BK.write(G['cutter'], G['lo'], G['h'])
                self.bake.update({'n_surface': G['n_surface'], 'openings': G['openings'], 'ms': round(G['ms'], 1),
                                  'h': G['h']})
            inner = ops.Node('lattice', {'file': self.bake['file'], 'sha': self.bake['sha']}, [], nid=ids())
        else:
            inner = ops.Node('offset', {'distance': -self.wall}, [C.renumber(o.base, ids)], nid=ids(),
                             name='conform %s, wall %.1f mm' % (o.name, self.wall * 1000.0))
        C.set_data(inner, {'tool': 'conform', 'target': o.name, 'target_id': o.id, 'wall': self.wall,
                           'mode': self.mode})
        self._obj = SceneObject(cid, 'Cutter (conform %s, %.1f mm wall)' % (o.name, self.wall * 1000.0), inner,
                                transform=copy.deepcopy(o.transform),
                                colour={'albedo': list(CUTTER_COLOUR), 'variation': 0.0}, h=o.sculpt.h)
        scene.objects.append(self._obj)
        scene.next_id = max(scene.next_id, ids.next_id)
        scene.touch(structural=True, index=True)
        return [cid]

    def undo(self, scene):
        scene.objects.remove(self._obj)
        scene.next_id = self._prev_next
        scene.touch(structural=True, index=True)


def conform_info(scene, cutter_oid):
    """a live sculpt-exact Conform cutter's {'target_id', 'wall', 'mode'} (its lattice node's tool data), or None"""
    if scene.index_of(cutter_oid) < 0:
        return None
    o = scene.get(cutter_oid)
    d = C.get_data(o.base) if o.base is not None else None
    if not d or d.get('tool') != 'conform' or o.base.kind != 'lattice':
        return None
    return d


@command
class RebakeConform(Command):
    """the live Conform cutter's wall follows the sculpting (refreshed after a stroke is released,
    never during one): its baked lattice replaced by one computed from the surface as it is now (bake: computed off the
    engine thread by ui/cutprev and recorded here, so a replay loads the same file); one undo step with the stroke"""
    label = 'Conform follows the sculpt'

    def __init__(self, cutter, bake):
        self.cutter = int(cutter)
        self.bake = dict(bake)

    def args(self):
        return {'cutter': self.cutter, 'bake': self.bake}

    def do(self, scene):
        o = scene.get(self.cutter)
        self._old = dict(o.base.params)
        o.base.params = dict(o.base.params, file=self.bake['file'], sha=self.bake['sha'])
        o.invalidate()
        scene.touch(structural=True, index=True)
        return None

    def undo(self, scene):
        o = scene.get(self.cutter)
        o.base.params = dict(self._old)
        o.invalidate()
        scene.touch(structural=True, index=True)


def wall_report(scene, target_oid, cutter_oid, step=0.0015, max_nodes=6_000_000, tol=0.0001):
    """the wall a cut would leave: on the cutter's surface inside the target, the target's depth there -> {'n', 'min',
    'median', 'max', 'p01', 'p99', 'thin' (points under wall - tol), 'true_depth', 'ms'} (metres). The depth is -F where
    the field is a distance (no sculpt layer), else the distance to the actual surface (tools/truedist: a sculpt layer
    is a distance only inside its band). A wall exactly as set reads min = max = the wall."""
    import time as _t
    from .protected_merge import field_of, world_box
    t0 = _t.perf_counter()
    tg, cut = scene.get(target_oid), scene.get(cutter_oid)
    if tg is None or cut is None:
        return None
    pk = scene.pack()
    lo, hi = (np.asarray(v, float) for v in world_box(cut))
    tlo, thi = (np.asarray(v, float) for v in world_box(tg))
    lo, hi = np.maximum(lo, tlo), np.minimum(hi, thi)
    if np.any(hi <= lo):
        return {'n': 0, 'ms': 0.0}
    n_ax = np.ceil((hi - lo) / step) + 1
    if float(np.prod(n_ax)) > max_nodes:
        step = float(((np.prod(hi - lo)) / max_nodes) ** (1.0 / 3.0)) * 1.01
    ax = [np.arange(lo[a], hi[a] + step, step) for a in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1)
    Fc = field_of(scene, pk, cut, G.reshape(-1, 3)).reshape(G.shape[:3])
    pts = []
    for a in range(3):
        s0, s1 = [slice(None)] * 3, [slice(None)] * 3
        s0[a], s1[a] = slice(0, -1), slice(1, None)
        f0, f1 = Fc[tuple(s0)], Fc[tuple(s1)]
        m = np.isfinite(f0) & np.isfinite(f1) & ((f0 < 0) != (f1 < 0))
        if m.any():
            t = (f0[m] / (f0[m] - f1[m]))[:, None]
            pts.append(G[tuple(s0)][m] + t * (G[tuple(s1)][m] - G[tuple(s0)][m]))
    if not pts:
        return {'n': 0, 'ms': (_t.perf_counter() - t0) * 1000}
    P = np.vstack(pts)
    # onto the cutter's surface exactly (Newton on its field), then the target's depth there
    for _ in range(4):
        f = field_of(scene, pk, cut, P)
        g = np.empty_like(P)
        for a in range(3):
            e = np.zeros(3)
            e[a] = 2e-5
            g[:, a] = (field_of(scene, pk, cut, P + e) - field_of(scene, pk, cut, P - e)) / 4e-5
        gg = np.maximum((g * g).sum(1), 1e-18)
        stp = (f / gg)[:, None] * g
        ok = np.isfinite(stp).all(1)
        P[ok] -= stp[ok]
    Ft = field_of(scene, pk, tg, P)
    keep = np.isfinite(Ft) & (Ft < 0)
    if not keep.any():
        return {'n': 0, 'ms': (_t.perf_counter() - t0) * 1000}
    w = -Ft[keep]
    true_depth = False
    if tg.sculpt.slots:
        # a sculpt layer: deeper than its band the field is not the depth (tools/truedist): the distance from each
        # point to the actual surface (its points from the full field, a k-d tree)
        from scipy.spatial import cKDTree
        from . import truedist as TD
        S, N, _m, lat = TD.surface_samples(scene, tg.id)
        if len(S):
            w = TD.depth_at(S, N, cKDTree(S), P[keep], lat[6])
            true_depth = True
    wall = None
    try:
        d = C.get_data(cut.base) if hasattr(C, 'get_data') else None
        wall = float((d or {}).get('wall')) if d else None
    except Exception:
        wall = None
    out = {'n': int(len(w)), 'min': float(w.min()), 'median': float(np.median(w)), 'max': float(w.max()),
           'p01': float(np.percentile(w, 1)), 'p99': float(np.percentile(w, 99)), 'step': step,
           'true_depth': true_depth, 'ms': (_t.perf_counter() - t0) * 1000}
    if wall is not None:
        out['wall'] = wall
        out['thin'] = int((w < wall - tol).sum())
    return out


# ---- the live preview (design 4.18: "a live preview tints the volume that will go") -------------------------------------
def preview_mask(scene, cam, W, H, cutter_oid, target_oids, side='inside', step=3, max_steps=96, max_rays=40000):
    """-> {'x0','y0','step','mask' (h, w) bool, 't', 'n_rays', 'ms'} or None: where, seen through the camera, a ray
    passes through the volume the cut would take away: target AND cutter (side 'inside') or target AND NOT cutter
    ('outside'), exactly from the fields (the targets' sculpt layers included), by sphere tracing max(F_t, s F_c)
    inside the overlap of their world boxes. Only the cutter's screen box (inside) or the targets' (outside) is traced,
    at one ray per `step` pixels. Field-native: no triangles, no card needed."""
    import math as _m
    import time as _t
    from .protected_merge import field_of, world_box
    from ..gpu.parts_overlay import _rays, _origins
    t_start = _t.perf_counter()
    cut = scene.get(cutter_oid)
    tg = [scene.get(o) for o in target_oids if scene.index_of(o) >= 0 and o != cutter_oid]
    if cut is None or not tg:
        return None
    pk = scene.pack()
    cbox = world_box(cut)
    tbox = np.array([world_box(o) for o in tg])                      # (n, 2, 3)
    tlo, thi = tbox[:, 0].min(0), tbox[:, 1].max(0)
    if side == 'inside':
        lo, hi = np.maximum(cbox[0], tlo), np.minimum(cbox[1], thi)
    else:
        lo, hi = tlo, thi
    if np.any(hi <= lo):
        return {'x0': 0, 'y0': 0, 'step': step, 'mask': np.zeros((0, 0), bool), 't': None, 'n_rays': 0,
                'ms': 0.0}
    # the box's screen rectangle (its corners projected), clipped to the view
    cs = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pr = np.array([cam.project(c, W, H) for c in cs])
    if np.all(pr[:, 2] <= 1e-6):
        return None
    if np.any(pr[:, 2] <= 1e-6):
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0 = int(max(0, _m.floor(pr[:, 0].min()) - step))
        y0 = int(max(0, _m.floor(pr[:, 1].min()) - step))
        x1 = int(min(W, _m.ceil(pr[:, 0].max()) + step))
        y1 = int(min(H, _m.ceil(pr[:, 1].max()) + step))
    if x1 <= x0 or y1 <= y0:
        return None
    while ((x1 - x0) // step) * ((y1 - y0) // step) > max_rays:
        step += 1
    D = _rays(cam, W, H, step)
    O = _origins(cam, W, H, step)
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    iy = np.nonzero((ys >= y0) & (ys < y1))[0]
    ix = np.nonzero((xs >= x0) & (xs < x1))[0]
    d = D[iy][:, ix].reshape(-1, 3)
    o = np.ascontiguousarray(np.broadcast_to(O, D.shape)[iy][:, ix].reshape(-1, 3))
    L = max(1.0, float(cut.sculpt.lipschitz(scene.store)) if cut.sculpt.slots else 1.0,
            *[float(o_.sculpt.lipschitz(scene.store)) if o_.sculpt.slots else 1.0 for o_ in tg])
    sgn = 1.0 if side == 'inside' else -1.0
    th = _m.tan(_m.radians(cam.fov) * 0.5)

    def trace(o, d, px_step):
        """sphere-trace max(F_t, s F_c) along rays inside the box -> (hit, closest approach in screen pixels)"""
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        ta, tb = (lo[None] - o) * inv, (hi[None] - o) * inv
        tn = np.maximum(np.minimum(ta, tb).max(1), 0.0)
        tf = np.maximum(ta, tb).min(1)
        act = np.nonzero(tf > tn)[0]
        t = tn.copy()
        hit = np.zeros(len(d), bool)
        near = np.full(len(d), np.inf)
        for _ in range(max_steps):
            if not len(act):
                break
            P = o[act] + d[act] * t[act][:, None]
            Ft = np.min(np.stack([field_of(scene, pk, o_, P) for o_ in tg]), axis=0)
            Fc = field_of(scene, pk, cut, P)
            F = np.maximum(Ft, sgn * Fc)
            pxw = (np.maximum(2.0 * th / H * t[act], 1e-9) if not getattr(cam, 'ortho', False)
                   else np.full(len(act), 2.0 * th * cam.ortho_params()[0] / H))
            near[act] = np.minimum(near[act], F / L / pxw)
            eps = np.maximum(pxw * px_step / 3.0, 0.0002)
            h_ = F < eps
            hit[act[h_]] = True
            t[act] += np.maximum(F / L * 0.9, eps * 0.5)
            keep = (~h_) & (t[act] < tf[act])
            act = act[keep]
        return hit, near

    # 1. coarse: one ray per `step` pixels
    hit, near = trace(o, d, step)
    hc, wc = len(iy), len(ix)
    mask_c = hit.reshape(hc, wc)
    sd_c = np.where(hit, -1.0, np.minimum(near, 4.0 * step)).reshape(hc, wc)
    # 2. the boundary cells (the mask differs from a neighbour) traced again at every pixel, so the edge is as fine as
    #    the frame (the coarse edge stair-stepped)
    s_ = int(step)
    gx0, gy0 = (int(xs[ix[0]]) if len(ix) else 0), (int(ys[iy[0]]) if len(iy) else 0)   # cell k: pixels [k s, (k+1) s)
    full_m = np.repeat(np.repeat(mask_c, s_, 0), s_, 1)
    full_sd = np.repeat(np.repeat(sd_c, s_, 0), s_, 1).astype(np.float32)
    if s_ > 1 and mask_c.size:
        m = mask_c
        pad = np.pad(m, 1, mode='edge')
        diff = np.zeros_like(m)
        for dy_ in (-1, 0, 1):
            for dx_ in (-1, 0, 1):
                diff |= pad[1 + dy_:1 + dy_ + hc, 1 + dx_:1 + dx_ + wc] != m
        cy_, cx_ = np.nonzero(diff)
        if len(cy_):
            oy_, ox_ = np.meshgrid(np.arange(s_), np.arange(s_), indexing='ij')
            py_ = (cy_[:, None] * s_ + oy_.ravel()[None]).ravel()
            px_ = (cx_[:, None] * s_ + ox_.ravel()[None]).ravel()
            # the fine rays (each pixel's centre), as gpu.parts_overlay._rays / _origins
            f_, r_, u_ = (np.asarray(v, float) for v in cam.basis())
            a_ = W / float(H)
            sx_ = (2.0 * (gx0 + px_ + 0.5) / W - 1.0) * th * a_
            sy_ = (1.0 - 2.0 * (gy0 + py_ + 0.5) / H) * th
            if getattr(cam, 'ortho', False):
                D_, B_ = cam.ortho_params()
                dd = np.broadcast_to(f_, (len(px_), 3)).copy()
                oo = np.asarray(cam.pos, float)[None] + (r_[None] * sx_[:, None] + u_[None] * sy_[:, None]) * D_ - f_[None] * B_
            else:
                dd = f_[None] + r_[None] * sx_[:, None] + u_[None] * sy_[:, None]
                dd /= np.linalg.norm(dd, axis=1, keepdims=True)
                oo = np.broadcast_to(np.asarray(cam.pos, float), (len(px_), 3)).copy()
            fh, fn = trace(np.ascontiguousarray(oo), dd, 1)
            ok_ = (py_ < full_m.shape[0]) & (px_ < full_m.shape[1])
            full_m[py_[ok_], px_[ok_]] = fh[ok_]
            full_sd[py_[ok_], px_[ok_]] = np.where(fh, -1.0, np.minimum(fn, 4.0))[ok_]
    return {'x0': gx0, 'y0': gy0, 'step': 1, 'mask': full_m, 'sd_px': full_sd,
            'n_rays': int(len(d)) + int(len(cy_) * s_ * s_ if s_ > 1 and mask_c.size and len(cy_) else 0),
            'coarse_step': s_,
            'ms': (_t.perf_counter() - t_start) * 1000.0}
