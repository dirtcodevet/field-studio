"""Cut (TOOLSET_DESIGN 3.3, approved 11-14): a straight line cut with a Length, and a shape cut into the container.

Design: a straight line cut has a set length; it cuts straight away from the view, starting where the click is
and ending where the drag finishes, and a length long enough cuts clean through the object, leaving two objects.
Q5: the Length is measured from the first surface. A shape cut shows the shape in 3-D; it is scaled to size and
dragged to exactly where the cut goes (inside an object or on its surface), and 'cut away' makes it a container:
that shape is removed from the object and is now in the container. One click deletes it, or it is moved to another
place and a different click puts it there.

LineCut(oid, (o_a, d_a), (o_b, d_b), length, kerf)
- the blade is the plane through the two view rays where the drag started and ended (away from the view), from the
  first surface the rays meet (in front of it by a hair) to Length behind it;
- if the blade cuts the object into separate pieces (a component count on the field), the pieces are split EXACTLY by
  the blade's plane, with no kerf: the volume is unchanged and the faces are flat (the cut piece is the part on its side
  of the plane within its own box: a region R, piece = F AND R, the rest = F MINUS R);
- otherwise it is a slit of that depth, `kerf` wide (default one lattice cell);
- the sculpt layer is carried by the EXACT rule: delta' = OP(base + delta, cutter) - OP(base, cutter), so a sculpted
  bulge never fills the cut and nothing moves where the cutter does not reach.
CutAway(oid, shape, params, at) : the shape (a primitive, placed and scaled) is removed from the object and becomes
  its own object (the container's held piece), in place and in contact; Delete or Place it after.
"""
import copy

import numpy as np

from ..core import ops
from ..core import brush as BR
from ..core.history import Command, command
from ..core.scene import SceneObject
from ..warp.bake import object_field
from ..warp.lens import march
from . import common as C


def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-300)


class _Ids:
    def __init__(self, scene):
        self.scene = scene

    def __call__(self):
        return self.scene.new_id()


def _renumber(node, ids):
    t = ops.Node.from_dict(node.to_dict())
    for n in t.walk():
        n.id = ids()
    return t


def _halfspace(n, c, extent, ids):
    """{x : n.x <= c} inside a cube of half size extent (a slab of thickness 2 extent whose upper face is the plane)"""
    n = _unit(n)
    return ops.Node('slab', {'normal': [float(v) for v in n], 'offset': float(c - extent), 'thickness': float(2 * extent),
                             'extent': float(extent)}, nid=ids())


def _obox(centre, axes, half, ids):
    """an oriented box: centre, axes (3x3 columns: u, v, w), half sizes"""
    from ..core.combine import _mat_to_euler
    R = np.asarray(axes, float)
    if np.linalg.det(R) < 0:
        R = R.copy()
        R[:, 2] *= -1
    b = ops.Node('box', {'size': [float(2 * h) for h in half], 'round': 0.0}, nid=ids())
    return ops.Node('transform', {'translate': [float(v) for v in centre], 'rotate': _mat_to_euler(R),
                                  'scale': [1.0, 1.0, 1.0]}, [b], nid=ids())


def _tree_values(tree, X):
    code, prm, lo, hi, L = ops.compile_tree(tree)
    out = np.empty(len(X))
    C.eval_tree(code, prm, np.ascontiguousarray(X), out)
    return out


class _ExactOps:
    """apply a boolean with a cutter tree to an object (base and sculpt, exact), and make a piece object; undoable"""

    def _apply(self, scene, o, op, cutter, rec, full=None, extra=None):
        """o := OP(o, cutter): base gets the node, the delta of every held brick is rebuilt by the exact rule. full: the
        cutter's whole field (an object cutter's sculpt included; local units) where the tree holds only its base;
        extra: more brick keys where that sculpt can decide values"""
        keys = sorted(set(o.sculpt.slots) | set(map(tuple, extra if extra is not None else [])))
        if keys:
            X = C.brick_nodes(keys, o.sculpt.h).reshape(-1, 3)
            b = C.field_local(scene, o, X, base_only=True)
            f = C.field_local(scene, o, X)
            c = _tree_values(cutter, X)
            cf = c if full is None else full(X)
            sg = -1.0 if op == 'subtract' else 1.0
            cc, ccf = sg * c, sg * cf
            new = np.maximum(f, ccf) - np.maximum(b, cc)
            hit = ((ccf > f) | (cc > b) | (ccf != cc)).reshape(len(keys), -1)   # where the cutter decides a value
            sel = [i for i in range(len(keys)) if hit[i].any()]
            if sel:
                st = scene.store
                d = np.stack([st.delta[o.sculpt.get_slot(keys[i])].astype(np.float64) if o.sculpt.get_slot(keys[i]) >= 0
                              else np.zeros((8, 8, 8)) for i in sel])
                nv = np.where(hit[sel].reshape(d.shape), new.reshape(len(keys), 8, 8, 8)[sel], d)
                C.write_bricks(scene, o, [keys[i] for i in sel], nv, rec)
        o.base = ops.Node(op, {'blend': 0.0}, [o.base, cutter], nid=scene.new_id(), name='Cut')
        o.invalidate()
        o.compiled()
        scene.touch(structural=True, index=True)

    def _piece(self, scene, o, cutter, name, full=None, extra=None):
        """a new object = o AND cutter, in o's frame, its sculpt carried exactly (only bricks that hold a change)"""
        ids = _Ids(scene)
        tree = ops.Node('intersect', {'blend': 0.0}, [_renumber(o.base, ids), _renumber(cutter, ids)], nid=ids(),
                        name='Cut piece')
        p = SceneObject(scene.new_id(), '%s piece' % o.name, tree, transform=copy.deepcopy(o.transform),
                        colour=copy.deepcopy(o.colour), material=copy.deepcopy(o.material), h=o.sculpt.h)
        p.detail = copy.deepcopy(o.detail)
        keys = sorted(set(o.sculpt.slots) | set(map(tuple, extra if extra is not None else [])))
        vals = None
        if keys:
            X = C.brick_nodes(keys, o.sculpt.h).reshape(-1, 3)
            b = C.field_local(scene, o, X, base_only=True)
            f = C.field_local(scene, o, X)
            c = _tree_values(cutter, X)
            cf = c if full is None else full(X)
            vals = (np.maximum(f, cf) - np.maximum(b, c)).reshape(len(keys), 8, 8, 8)
        return p, keys, vals

    def _insert_piece(self, scene, p, keys, vals, rec):
        scene.objects.append(p)
        scene.touch(structural=True, index=True)
        if keys:
            nz = [i for i in range(len(keys)) if np.any(vals[i] != 0)]
            if nz:
                C.write_bricks(scene, p, [keys[i] for i in nz], vals[nz], rec)
                BR.measure_bricks(scene, scene.index_of(p.id), {tuple(keys[i]) for i in nz})
                scene.refresh_object(scene.index_of(p.id))

    def undo_state(self):
        """the undo journal: the base trees before, the pieces' ids, and the brick record (the pieces' bricks leave with
        them)"""
        if getattr(self, 'rec', None) is None:
            return None
        from ..core import undojournal as UJ
        rec = self.rec
        meta = {'old_base': {str(k): v for k, v in self.old_base.items()}, 'pieces': [int(p.id) for p in self.pieces],
                'prev_next': int(self.prev_next)}

        def make():
            A = {}
            UJ.record_arrays('rec', rec, A)
            return meta, A
        return make

    def restore_undo_state(self, scene, meta, A):
        from ..core import undojournal as UJ
        self.rec = UJ.record_from('rec', A)
        self.old_base = {int(k): v for k, v in meta['old_base'].items()}
        self.pieces = [scene.get(i) for i in meta['pieces']]
        self.prev_next = meta['prev_next']

    def _undo_all(self, scene):
        for p in reversed(getattr(self, 'pieces', [])):
            if scene.index_of(p.id) >= 0:
                from ..core.history import release_object_slots
                release_object_slots(scene, p)
                scene.objects.remove(p)
        if getattr(self, 'rec', None) is not None:
            # the piece's bricks were released with it: drop them from the record before restoring the rest
            self.rec.new = [k for k in self.rec.new if scene.index_of(k[0]) >= 0]
            self.rec.new_set = set(self.rec.new)
            self.rec.snap = {k: v for k, v in self.rec.snap.items() if scene.index_of(k[0]) >= 0}
            self.rec.Lsnap = {k: v for k, v in self.rec.Lsnap.items() if scene.index_of(k[0]) >= 0}
            BR.undo_record(scene, self.rec)
            self.rec = None
        for oid, tree in getattr(self, 'old_base', {}).items():
            o = scene.get(oid)
            o.base = ops.Node.from_dict(tree)
            o.invalidate()
        scene.next_id = self.prev_next
        scene.touch(structural=True, index=True)


def _components(field, lo, hi, cutter_fn, cell):
    """label the solid's connected components on a grid (cell spacing), with the cutter removed"""
    from scipy import ndimage
    n = np.maximum(np.ceil((hi - lo) / cell).astype(int) + 1, 2)
    ax = [lo[a] + cell * np.arange(n[a]) for a in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    f = field(G)
    c = cutter_fn(G)
    solid = ((np.maximum(f, -c)) < 0).reshape(tuple(n))
    lab, k = ndimage.label(solid)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, k + 1)) if k else np.array([])
    return lab, k, sizes, ax


@command
class LineCut(_ExactOps, Command):
    """A straight cut: rays (origin, direction) where the drag started (a) and ended (b), world; length from the
    first surface (m); kerf for a slit (m, default one lattice cell of the object)."""
    label = 'Cut'

    def __init__(self, oid, ray_a, ray_b, length, kerf=None, cell=None):
        self.oid = int(oid)
        self.ray_a = [[float(v) for v in ray_a[0]], [float(v) for v in ray_a[1]]]
        self.ray_b = [[float(v) for v in ray_b[0]], [float(v) for v in ray_b[1]]]
        self.length = float(length)
        self.kerf = None if kerf is None else float(kerf)
        self.cell = None if cell is None else float(cell)
        self.info = {}

    def args(self):
        d = {'oid': self.oid, 'ray_a': self.ray_a, 'ray_b': self.ray_b, 'length': self.length}
        if self.kerf is not None:
            d['kerf'] = self.kerf
        if self.cell is not None:
            d['cell'] = self.cell
        return d

    def blade(self, scene):
        """the blade in world: plane (n, c), its axes, the first-surface depth -> dict (also the window's cue)"""
        o = scene.get(self.oid)
        oa, da = np.asarray(self.ray_a[0]), _unit(self.ray_a[1])
        ob, db = np.asarray(self.ray_b[0]), _unit(self.ray_b[1])
        m = _unit(da + db)
        if np.linalg.norm(np.cross(da, db)) < 1e-9:
            n = _unit(np.cross(m, ob - oa))              # parallel rays (orthographic): the plane holds both
        else:
            n = _unit(np.cross(da, db))                  # rays from one eye: the plane through both
        c = float(n @ oa)
        # the first surface along rays between a and b (64 rays)
        f = object_field(scene, o)
        ts = np.linspace(0.0, 1.0, 64)
        orgs = oa[None, :] * (1 - ts[:, None]) + ob[None, :] * ts[:, None]
        dirs = np.array([_unit(x) for x in da[None, :] * (1 - ts[:, None]) + db[None, :] * ts[:, None]])
        lo, hi = C.world_box(o)
        far = float(np.linalg.norm(hi - lo)) + float(max(np.linalg.norm(lo - oa), np.linalg.norm(hi - oa)))
        hit = march(f, orgs, dirs, far, L=max(scene.pack().of_tab[scene.index_of(o.id), 20], 1.0),
                    hmin=max(o.sculpt.h * C.frame(o)[2] * 0.5, 2e-5))
        t1 = hit[:, 0]
        ok = np.isfinite(t1)
        if not ok.any():
            return None
        P = orgs[ok] + dirs[ok] * t1[ok, None]
        d0 = float(((P - oa) @ m).min())                 # depth along the mean view direction from a's origin
        u = _unit(ob - oa - ((ob - oa) @ m) * m) if np.linalg.norm(ob - oa) > 1e-9 else _unit(np.cross(n, m))
        return {'n': n, 'c': c, 'm': m, 'u': u, 'oa': oa, 'da': da, 'ob': ob, 'db': db, 'd0': d0,
                'first': P[np.argmin((P - oa) @ m)], 'hits': P}

    def _cutter_local(self, scene, o, bl, kerf, ids):
        """the slit (kerf wide) in o's local frame: the wedge between the two rays (an outline swept along the view rays
        from the eye; parallel rays: a box), intersected with the depth slab [d0 - eps, d0 + length]"""
        s = C.frame(o)[2]
        eps = max(2.0 * kerf, 1e-4)
        depth_lo = bl['d0'] - eps
        depth_hi = bl['d0'] + self.length
        m, n, u = bl['m'], bl['n'], bl['u']
        oa, ob, da, db = bl['oa'], bl['ob'], bl['da'], bl['db']
        # the blade's quad in world: the two rays between depth_lo and depth_hi (along m)
        ta_lo, ta_hi = depth_lo / (da @ m), depth_hi / (da @ m)
        tb_lo = ((oa - ob) @ m + depth_lo) / (db @ m)
        tb_hi = ((oa - ob) @ m + depth_hi) / (db @ m)
        quad = np.array([oa + da * ta_lo, oa + da * ta_hi, ob + db * tb_hi, ob + db * tb_lo])
        # a thin solid: the quad's plane +- kerf / 2, bounded by the quad's four edges (each edge's outward plane)
        ql = C.to_local(o, quad)
        nl = _unit(C.dir_to_local(o, n))
        cen = ql.mean(0)
        parts = [ops.Node('slab', {'normal': [float(v) for v in nl], 'offset': float(nl @ cen),
                                   'thickness': float(kerf / s), 'extent': 8.0 * float(np.abs(ql).max() + 1.0)},
                          nid=ids())]
        ext = 8.0 * float(np.abs(ql).max() + 1.0)
        for i in range(4):
            p0, p1 = ql[i], ql[(i + 1) % 4]
            e = _unit(np.cross(p1 - p0, nl))
            if e @ (cen - p0) > 0:
                e = -e
            parts.append(_halfspace(e, float(e @ p0), ext, ids))        # inside: e.x <= e.p0 (toward the centre)
        acc = parts[0]
        for p in parts[1:]:
            acc = ops.Node('intersect', {'blend': 0.0}, [acc, p], nid=ids())
        return acc, quad

    def do(self, scene):
        C.refuse_protected(scene, [self.oid], 'Cut')
        o = scene.get(self.oid)
        self.prev_next = scene.next_id
        self.old_base = {o.id: o.base.to_dict()}
        self.pieces = []
        self.rec = BR.StrokeRecord()
        bl = self.blade(scene)
        if bl is None:
            raise ValueError('The cut line does not cross the object')
        hw = o.sculpt.h * C.frame(o)[2]
        kerf = self.kerf if self.kerf is not None else hw
        ids = _Ids(scene)
        slit, quad = self._cutter_local(scene, o, bl, kerf, ids)
        # does the blade cut it through? components of the solid with a slit wide enough for the census grid
        lo, hi = C.world_box(o)
        cell = self.cell or float(max(hw, (hi - lo).max() / 120.0))
        census, _ = self._cutter_local(scene, o, bl, max(kerf, 2.5 * cell), _Ids(scene))
        cc = ops.compile_tree(census)
        f = object_field(scene, o)

        def cut_fn(W):
            out = np.empty(len(W))
            C.eval_tree(cc[0], cc[1], np.ascontiguousarray(C.to_local(o, W)), out)
            return out * C.frame(o)[2]
        lab, k, sizes, ax = _components(f, lo - 2 * cell, hi + 2 * cell, cut_fn, cell)
        big = [i + 1 for i in range(k) if sizes[i] >= 8]
        self.info = {'first_surface': [round(float(v), 6) for v in bl['first']], 'kerf_mm': round(kerf * 1e3, 4),
                     'components': len(big), 'quad': quad.tolist()}
        if len(big) >= 2:
            done = self._split(scene, o, bl, lab, big, ax, cell, ids)
            if done:
                return self.info
            self.info['through_split'] = 'pieces interlock across the plane: kept as one object with a slit'
        self._apply(scene, o, 'subtract', slit, self.rec)
        self.info['result'] = 'slit'
        scene.next_id = max(scene.next_id, self.prev_next)
        return self.info

    def _split(self, scene, o, bl, lab, big, ax, cell, ids):
        """split exactly by the blade's plane: every piece but the largest becomes its own object"""
        n, c = bl['n'], bl['c']
        G = np.stack(np.meshgrid(*ax, indexing='ij'), -1)
        sizes = {i: int((lab == i).sum()) for i in big}
        keep = max(big, key=lambda i: sizes[i])
        pieces = []
        for i in big:
            if i == keep:
                continue
            P = G[lab == i]
            side = np.sign(np.median(P @ n - c))
            if side == 0:
                return False
            lo = P.min(0) - 2 * cell
            hi = P.max(0) + 2 * cell
            # the region R: its side of the plane, within its box; nothing of the other pieces may be inside it
            inR = np.all((G >= lo) & (G <= hi), axis=-1) & ((G @ n - c) * side >= -0.5 * cell)
            others = (lab > 0) & (lab != i)
            if np.any(inR & others):
                return False
            pieces.append((i, side, lo, hi))
        if not pieces:
            return False
        llo, lhi = o.local_box()
        ext = 8.0 * float(max(np.abs(llo).max(), np.abs(lhi).max())) + 1.0
        regions = []
        for i, side, lo, hi in pieces:
            nl = _unit(C.dir_to_local(o, n * side))       # the piece is where nl.x >= cl (its side)
            p0 = C.to_local(o, n * c)                      # a point on the plane
            cl = float(nl @ p0)
            hs = _halfspace(-nl, -cl, ext, ids)            # {x : -nl.x <= -cl}
            Lb, Hb = C.local_box_of_world(o, lo, hi)
            box = _obox(0.5 * (Lb + Hb), np.eye(3), 0.5 * (Hb - Lb), ids)
            regions.append(ops.Node('intersect', {'blend': 0.0}, [hs, box], nid=ids(), name='cut region'))
        # the pieces first (from the uncut object), then the object minus every region
        made = [self._piece(scene, o, R, '%s piece' % o.name) for R in regions]
        for R in regions:
            self._apply(scene, o, 'subtract', R, self.rec)
        for p, keys, vals in made:
            self._insert_piece(scene, p, keys, vals, self.rec)
            self.pieces.append(p)
        self.info['result'] = 'through'
        self.info['pieces'] = [p.id for p in self.pieces]
        return True

    def undo(self, scene):
        self._undo_all(scene)


@command
class CutAway(_ExactOps, Command):
    """Cut with a shape: shape kind + params (core/ops.PRIMITIVES), placed at world `at` = {translate, rotate (deg),
    scale [x, y, z]}. The object loses the shape's volume; the volume becomes its own object, in place (the container's
    held piece: Delete it, or Place it)."""
    label = 'Cut away'

    def __init__(self, oid, shape, params, at):
        self.oid = int(oid)
        self.shape = str(shape)
        self.params = copy.deepcopy(params)
        self.at = copy.deepcopy(at)

    def args(self):
        return {'oid': self.oid, 'shape': self.shape, 'params': self.params, 'at': self.at}

    def _cutter_local(self, scene, o, ids):
        R0, t0, s0 = C.frame(o)
        Ra = ops.rot_matrix(self.at.get('rotate', [0.0, 0.0, 0.0]))
        ta = np.asarray(self.at.get('translate', [0.0, 0.0, 0.0]), float)
        sa = np.asarray(self.at.get('scale', [1.0, 1.0, 1.0]), float) * np.ones(3)
        # the shape's local frame in the object's local frame: x_obj = (R0^T (Ra diag(sa) q + ta - t0)) / s0
        from ..core.combine import _mat_to_euler
        Rl = R0.T @ Ra
        tl = (R0.T @ (ta - t0)) / s0
        prim = ops.Node(self.shape, dict(ops.PRIMITIVES[self.shape], **self.params), nid=ids())
        return ops.Node('transform', {'translate': [float(v) for v in tl], 'rotate': _mat_to_euler(Rl),
                                      'scale': [float(v / s0) for v in sa]}, [prim], nid=ids())

    def do(self, scene):
        C.refuse_protected(scene, [self.oid], 'Cut away')
        o = scene.get(self.oid)
        self.prev_next = scene.next_id
        self.old_base = {o.id: o.base.to_dict()}
        self.pieces = []
        self.rec = BR.StrokeRecord()
        ids = _Ids(scene)
        cutter = self._cutter_local(scene, o, ids)
        p, keys, vals = self._piece(scene, o, cutter, '%s piece' % o.name)
        if not _has_material(p):
            raise ValueError('The shape holds none of the object: nothing to cut away')
        self._apply(scene, o, 'subtract', cutter, self.rec)
        self._insert_piece(scene, p, keys, vals, self.rec)
        self.pieces.append(p)
        return p.id

    def undo(self, scene):
        self._undo_all(scene)


def _has_material(obj, n=24):
    code, prm, lo, hi, L = ops.compile_tree(obj.base)
    ax = [np.linspace(lo[a], hi[a], n) for a in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    out = np.empty(len(g))
    C.eval_tree(code, prm, np.ascontiguousarray(g), out)
    return bool((out < 0).any())


@command
class CutWithObject(_ExactOps, Command):
    """Cut with another object as the cutter (TOOLSET_DESIGN 3.3: "any object or part used as the cutter"): the cutter
    is used up (it leaves the scene; undo puts it back).
    keep 'both' (the default): the target loses the cutter's volume, which becomes its own object (the container's held
    piece); 'rest' = Subtract (no piece); 'piece' = Intersect (the target becomes what the cutter held, in place).
    Exact: the cutter's base goes into the target's tree (in the target's frame); its sculpt, and the target's, by the
    exact rule at the target's lattice nodes."""
    label = 'Cut with object'

    def __init__(self, oid, cutter, keep='both'):
        self.oid = int(oid)
        self.cutter = int(cutter)
        self.keep = str(keep)

    def args(self):
        return {'oid': self.oid, 'cutter': self.cutter, 'keep': self.keep}

    def _cutter_tree(self, scene, o, q, ids):
        from ..core.combine import _frame as _fr, _mat_to_euler
        RM, tM, sM = _fr(o.transform)
        Rq, tq, sq = _fr(q.transform)
        R = RM.T @ Rq
        t = RM.T @ (tq - tM) / sM
        sc = sq / sM
        return ops.Node('transform', {'translate': [float(v) for v in t], 'rotate': _mat_to_euler(R),
                                      'scale': [sc, sc, sc]}, [_renumber(q.base, ids)], nid=ids(), name='cutter %s' % q.name)

    def do(self, scene):
        C.refuse_protected(scene, [self.oid, self.cutter], 'Cut with object')
        o = scene.get(self.oid)
        q = scene.get(self.cutter)
        if self.keep not in ('both', 'rest', 'piece'):
            raise ValueError('keep: both, rest or piece')
        self.prev_next = scene.next_id
        self.old_base = {o.id: o.base.to_dict()}
        self.pieces = []
        self.rec = BR.StrokeRecord()
        ids = _Ids(scene)
        tree = self._cutter_tree(scene, o, q, ids)
        sM = C.frame(o)[2]
        fq = object_field(scene, q)
        full = (lambda X: fq(C.to_world(o, X)) / sM) if q.sculpt.slots else None
        extra = None
        if q.sculpt.slots:
            la, ha = C.world_box(q)
            lo_, hi_ = C.local_box_of_world(o, la, ha)
            ol, oh = o.local_box()
            lo_, hi_ = np.maximum(lo_, ol), np.minimum(hi_, oh)
            extra = C.bricks_in_box(lo_, hi_, o.sculpt.h, pad=1) if np.all(lo_ <= hi_) else None
        if self.keep in ('both',):
            p, keys, vals = self._piece(scene, o, tree, '%s piece' % o.name, full, extra)
            if not _has_material(p):
                raise ValueError('The cutter holds none of the object: nothing to cut away')
        op = 'intersect' if self.keep == 'piece' else 'subtract'
        self._apply(scene, o, op, tree, self.rec, full, extra)
        if self.keep == 'both':
            self._insert_piece(scene, p, keys, vals, self.rec)
            self.pieces.append(p)
        self.cut_at = scene.objects.index(q)
        scene.objects.remove(q)
        self.used = q
        scene.touch(structural=True, index=True)
        return self.pieces[0].id if self.pieces else o.id

    def undo(self, scene):
        scene.objects.insert(min(self.cut_at, len(scene.objects)), self.used)
        self._undo_all(scene)

    def undo_state(self):
        return None                                   # the cutter left the saved scene: the undo history stops here

    def discard(self, scene, state):
        if state == 'committed':
            from ..core.history import release_object_slots
            release_object_slots(scene, self.used)
