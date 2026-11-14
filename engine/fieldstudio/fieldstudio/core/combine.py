"""hand2f (2026-11-08): Merge and the object commands that carry the sculpt layer exactly.

CombineObjects(oids, op, k, frame)   Merge (op 'union') / Subtract / Intersect of 2+ objects into ONE new object M.
    Base: a left-deep fold in selection order, op_k(op_k(A', B'), C') ... (value stack depth 2), each source X' its
    renumbered tree under a transform node (its frame relative to M's) and, where it is safe, a 'cull' box guard, so
    a full-body object's ray step pays one box test for every part far from the point.
    Sculpt (EXACT rule, not the additive splice carry): over S = the sources' allocated bricks mapped into M's lattice
    (+1 brick), delta_M(x) = OP_k(s_i (b_i + d_i)(q_i)) / s_M - OP_k(s_i b_i(q_i)) / s_M, both folds evaluated the same
    way (unguarded), so delta_M = 0 bit-for-bit wherever every source delta is 0 (S's edge is seamless). With the
    same h, rotation and scale and a whole-node offset, the deltas are read at nodes (exact); otherwise trilinear
    (error reported). Mask: the source whose surface is nearest (smallest |F_i|) owns the node.
    The sources leave the scene and stay in memory; undo puts them back at their places (exact, cheap).
DuplicateObject(oid, offset)         a copy with its sculpt bricks copied exactly (same h, same frame).
MirrorDuplicate(oid)                 a copy reflected about the world plane x = 0: t' = Sx t, R' = Sx R Sx, base under a
                                     transform of scale [-1, 1, 1], sculpt node (i, j, k) -> (-i, j, k): EXACT.
MaskEdit(oid, op)                    'clear' (mask 0 everywhere, invert off) / 'invert' (the object's invert flag).

All are deterministic Commands (do / undo / to_log): replay of the op log reproduces the state hash.
"""
import copy

import numpy as np
from numba import njit, prange

from . import ops
from . import program as P
from . import brush as BR
from .scene import SceneObject
from .history import Command, command, release_object_slots

B = P.BRICK
OPS = {'union': 0, 'subtract': 1, 'intersect': 2}
CUT = 0.10            # m: a source whose box is farther than this (+ k) from a node is left out of a union's fold (D:
                      # its value there is >= the distance to its box only for exact SDFs; the ellipsoid's is >= that
                      # times r_min / r_max, so it cannot be the minimum where another part's surface is within 1-3 cm)


def _frame(t):
    return ops.rot_matrix(t['rotate']), np.asarray(t['translate'], float), max(float(t['scale']), 1e-6)


def _mat_to_euler(R):
    import math
    R = np.asarray(R, float)
    sy = min(1.0, max(-1.0, -R[2, 0]))
    y = math.asin(sy)
    if abs(math.cos(y)) > 1e-9:
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        z = 0.0
        x = math.atan2(-R[1, 2], R[1, 1])
    return [math.degrees(x), math.degrees(y), math.degrees(z)]


def _renumber(node, scene):
    t = ops.Node.from_dict(node.to_dict())
    for n in t.walk():
        n.id = scene.new_id()
    return t


# ---- the merge kernel -------------------------------------------------------------------------------------------

@njit(cache=True, inline='always')
def _fold(acc, v, first, op, k):
    if first:
        return v
    if op == 0:
        return min(acc, v) if k <= 0.0 else P._smin(acc, v, k)
    if op == 1:
        return max(acc, -v) if k <= 0.0 else -P._smin(-acc, v, k)
    return max(acc, v) if k <= 0.0 else -P._smin(-acc, -v, k)


@njit(cache=True, inline='always')
def _at(code, prm, oi_tab, of_tab, index_flat, delta, mask, srcs, op, k, sM, px, py, pz, vs, ps, res):
    """At world point p: res = (fold with sculpt, fold of the bases, mask of the nearest source), in M-local units."""
    ff = 0.0
    fb = 0.0
    best = 1e30
    mk = 0.0
    first = True
    for t in range(srcs.shape[0]):
        oi = srcs[t]
        lx, ly, lz = P.to_local(of_tab, oi, px, py, pz)
        s = of_tab[oi, P.OF_S]
        if op == 0:
            dx = max(of_tab[oi, P.OF_BMIN] - lx, lx - of_tab[oi, P.OF_BMAX])
            dy = max(of_tab[oi, P.OF_BMIN + 1] - ly, ly - of_tab[oi, P.OF_BMAX + 1])
            dz = max(of_tab[oi, P.OF_BMIN + 2] - lz, lz - of_tab[oi, P.OF_BMAX + 2])
            mx = max(dx, 0.0)
            my = max(dy, 0.0)
            mz = max(dz, 0.0)
            if s * np.sqrt(mx * mx + my * my + mz * mz) > CUT + k * sM:
                continue
        b = P.eval_prog_t(code, prm, oi_tab[oi, P.OI_C0], oi_tab[oi, P.OI_C1], lx, ly, lz, vs, ps, 1e30)
        h = of_tab[oi, P.OF_H]
        d = P.grid_sample(index_flat, delta, oi_tab, oi, h, lx, ly, lz)
        vb = s * b / sM
        vf = s * (b + d) / sM
        fb = _fold(fb, vb, first, op, k)
        ff = _fold(ff, vf, first, op, k)
        first = False
        if abs(vf) < best:
            best = abs(vf)
            m = P.grid_sample(index_flat, mask, oi_tab, oi, h, lx, ly, lz)
            m = min(max(m, 0.0), 1.0)
            if of_tab[oi, P.OF_MINV] > 0.5:
                m = 1.0 - m
            mk = m
    res[0] = ff
    res[1] = fb
    res[2] = mk


@njit(parallel=True, cache=True, nogil=True)
def combine_kernel(code, prm, oi_tab, of_tab, index_flat, delta, mask, srcs, op, k, RM, tM, sM, hM, bricks, slots,
                   nonzero):
    """For each M brick (prange; each node written once, no reductions: deterministic under threads): SET delta and
    mask at its 8^3 nodes. RM: M's rotation (local -> world)."""
    for q in prange(bricks.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        res = np.empty(3)
        slot = slots[q]
        nz = 0
        for a in range(B):
            for b in range(B):
                for c in range(B):
                    x = (bricks[q, 0] * B + a) * hM
                    y = (bricks[q, 1] * B + b) * hM
                    z = (bricks[q, 2] * B + c) * hM
                    px = sM * (RM[0, 0] * x + RM[0, 1] * y + RM[0, 2] * z) + tM[0]
                    py = sM * (RM[1, 0] * x + RM[1, 1] * y + RM[1, 2] * z) + tM[1]
                    pz = sM * (RM[2, 0] * x + RM[2, 1] * y + RM[2, 2] * z) + tM[2]
                    _at(code, prm, oi_tab, of_tab, index_flat, delta, mask, srcs, op, k, sM, px, py, pz, vs, ps, res)
                    dv = np.float32(res[0] - res[1])
                    mv = np.float32(res[2])
                    delta[slot, a, b, c] = dv
                    mask[slot, a, b, c] = mv
                    if dv != 0.0 or mv != 0.0:
                        nz = 1
        nonzero[q] = nz


@njit(cache=True)
def _fold_points(code, prm, oi_tab, of_tab, index_flat, delta, mask, srcs, op, k, sM, pts, out):
    vs = np.empty(P.STACK)
    ps = np.empty(4 * P.STACK)
    res = np.empty(3)
    for n in range(pts.shape[0]):
        _at(code, prm, oi_tab, of_tab, index_flat, delta, mask, srcs, op, k, sM, pts[n, 0], pts[n, 1], pts[n, 2],
            vs, ps, res)
        out[n] = res[0]


@njit(cache=True)
def _eval_exact(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts, out):
    """object oi's local field at local points with every guard opened (tau = inf): the exact value"""
    vs = np.empty(P.STACK)
    ps = np.empty(4 * P.STACK)
    for n in range(pts.shape[0]):
        out[n] = (P.eval_prog_t(code, prm, oi_tab[oi, P.OI_C0], oi_tab[oi, P.OI_C1], pts[n, 0], pts[n, 1], pts[n, 2],
                                vs, ps, 1e30)
                  + P.grid_sample(index_flat, delta, oi_tab, oi, of_tab[oi, P.OF_H], pts[n, 0], pts[n, 1], pts[n, 2]))


MAX_MERGE_BRICKS = 250000   # hand2b: a merge whose sculpt support would exceed this many bricks (2 x 2 KB each, about
                            # 1 GB of store) is refused BEFORE anything is allocated (verifier items 13, 14: a
                            # coarse-scaled source resampled onto a fine lattice asked for 274,625 bricks, ~1.07 GB)


def _brick_ranges(src, RM, tM, sM, hM):
    """-> (lo, hi) (n, 3) int64: the M-lattice brick range covering each of src's allocated bricks (undilated)"""
    Rs, ts, ss = _frame(src.transform)
    hs = src.sculpt.h
    keys = np.array(sorted(src.sculpt.slots), np.int64)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.float64)
    c = (keys[:, None, :] + corners[None]) * (B * hs)              # (n, 8, 3) src local
    w = np.einsum('ij,nkj->nki', Rs, c * ss) + ts                   # world
    m = np.einsum('ji,nkj->nki', RM, w - tM) / sM                   # M local (RM.T @ (w - tM))
    lo = np.floor(m.min(1) / hM).astype(np.int64) >> 3
    hi = np.floor(m.max(1) / hM).astype(np.int64) >> 3
    return lo, hi


def _support_core(src, RM, tM, sM, hM):
    """the sum of the UNdilated M-brick ranges' volumes src maps to (cheap, no allocation): about the support's
    size when the lattices match, and the blow-up when a coarse part is resampled onto a fine lattice"""
    if not src.sculpt.slots:
        return 0
    lo, hi = _brick_ranges(src, RM, tM, sM, hM)
    return int(np.prod(hi - lo + 1, axis=1).sum())


def _unique_rows(a):
    """READY 2 (merge-all): np.unique(a, axis=0) for (n, 3) int64 bricks, the same sorted rows, through one packed
    int64 key (np.unique over axis 0 sorted structured views: 0.15 s of a full-figure merge)"""
    a = np.asarray(a, np.int64).reshape(-1, 3)
    if not len(a):
        return a
    lo = a.min(0)
    ext = a.max(0) - lo + 1
    q = a - lo
    key = (q[:, 0] * ext[1] + q[:, 1]) * ext[2] + q[:, 2]
    u = np.unique(key)
    z = u % ext[2]
    y = (u // ext[2]) % ext[1]
    x = u // (ext[1] * ext[2])
    return np.stack([x, y, z], 1) + lo


def _ranges_union(lo, hi):
    """the distinct bricks of the boxes [lo_i, hi_i] (inclusive), as an (n, 3) int64 array"""
    ext = hi - lo + 1
    E = ext.max(0)
    out = []
    for dx in range(int(E[0])):
        for dy in range(int(E[1])):
            for dz in range(int(E[2])):
                d = np.array([dx, dy, dz], np.int64)
                ok = np.all(d[None] < ext, axis=1)
                if ok.any():
                    out.append(lo[ok] + d)
    return _unique_rows(np.concatenate(out))


def _carry_rid(st, srcs, M, arr, slots, RM, tM, sM, hM):
    """fs-shell 11-11 (regions, design 4.8): the merged object's region ids, each M node taking the id of the nearest
    node of the first source that labels it (pulled, so a finer M lattice has no gaps). The first source keeps its ids;
    a later source's ids that collide are renumbered (M.regions holds every table). -> (len(arr),) bool: the M bricks
    that hold ids (kept even where delta and mask are zero)."""
    M.regions = {}
    has = np.zeros(len(arr), bool)
    pos = {tuple(b): i for i, b in enumerate(arr.tolist())}
    g = np.indices((B, B, B)).reshape(3, -1).T                     # (512, 3) node offsets in a brick
    for o in srcs:
        lab = [b for b, s in o.sculpt.slots.items() if st.rid[s].any()]
        regs = getattr(o, 'regions', {}) or {}
        if not lab:
            continue
        remap = {}
        for k in sorted(regs):
            nk = k if k not in M.regions else max(M.regions) + 1
            remap[k] = nk
            M.regions[nk] = dict(regs[k])
        lut = np.arange(int(max(st.rid[o.sculpt.slots[b]].max() for b in lab)) + 1, dtype=np.uint16)
        for k, nk in remap.items():
            if k < len(lut):
                lut[k] = nk
        src = _Fake(o, lab)
        lo, hi = _brick_ranges(src, RM, tM, sM, hM)
        want = set()
        for a, c in zip(lo.tolist(), hi.tolist()):
            for x in range(a[0] - 1, c[0] + 2):
                for y in range(a[1] - 1, c[1] + 2):
                    for z_ in range(a[2] - 1, c[2] + 2):
                        if (x, y, z_) in pos:
                            want.add(pos[(x, y, z_)])
        if not want:
            continue
        idx = np.array(sorted(want), np.int64)
        Rs, ts, ss = _frame(o.transform)
        hs = o.sculpt.h
        nodes = (arr[idx][:, None, :] * B + g[None]).reshape(-1, 3).astype(np.float64)
        w = (nodes * hM * sM) @ RM.T + tM                              # M local -> world
        loc = ((w - ts) @ Rs) / ss                                     # world -> src local (Rs.T @ (w - ts))
        n = np.rint(loc / hs).astype(np.int64)
        sb, si = n >> 3, n & 7
        got = np.zeros(len(n), np.uint16)
        lk = np.array(lab, np.int64)
        lsl = np.array([o.sculpt.slots[b] for b in lab], np.int64)
        pk_l = _pack3(lk)
        order = np.argsort(pk_l)
        pk_l, lsl = pk_l[order], lsl[order]
        q = _pack3(sb)
        at = np.clip(np.searchsorted(pk_l, q), 0, len(pk_l) - 1)
        hit = pk_l[at] == q
        if hit.any():
            got[hit] = st.rid[lsl[at[hit]], si[hit, 0], si[hit, 1], si[hit, 2]]
        got = lut[np.minimum(got, len(lut) - 1)] * (got > 0)
        got = got.reshape(len(idx), B, B, B)
        for r, i in enumerate(idx.tolist()):
            if got[r].any():
                cur = st.rid[slots[i]]
                st.rid[slots[i]] = np.where(cur > 0, cur, got[r])
                has[i] = True
    return has


def _pack3(k):
    k = np.asarray(k, np.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


class _Fake:
    """an object stand-in whose sculpt holds only some bricks (for _brick_ranges)"""
    def __init__(self, o, keys):
        self.transform = o.transform

        class _S:
            pass
        self.sculpt = _S()
        self.sculpt.h = o.sculpt.h
        self.sculpt.slots = {b: 0 for b in keys}


def _mapped_bricks(src, RM, tM, sM, hM):
    """M-lattice bricks covering src's allocated bricks (each brick's 8 corners mapped through world), +1 dilation.
    hand2b: vectorised (the per-brick Python loops took seconds on a sculpted full-body merge); the undilated union
    first, then the dilation of that union (27 offsets) -> (n, 3) int64."""
    if not src.sculpt.slots:
        return np.zeros((0, 3), np.int64)
    lo, hi = _brick_ranges(src, RM, tM, sM, hM)
    core = _ranges_union(lo, hi)
    offs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], np.int64)
    return _unique_rows((core[:, None, :] + offs[None]).reshape(-1, 3))


def _measure_all(scene, oi):
    o = scene.objects[oi]
    if not o.sculpt.slots:
        return
    pk = scene.pack()
    keys = sorted(o.sculpt.slots)
    out = np.zeros(len(keys))
    BR.brick_lipschitz(pk.index_flat, pk.delta, pk.oi_tab, oi, o.sculpt.h, np.array(keys, np.int64), out)
    for b, L in zip(keys, out):
        scene.store.Lb[o.sculpt.slots[b]] = L
    scene.refresh_object(oi)


def _reset_box(g):
    if g.slots:
        ks = np.array(list(g.slots), np.int64)
        g.bmin, g.bmax = ks.min(0), ks.max(0)
    else:
        g.bmin = g.bmax = None


class _ObjectSwap(Command):
    """do/undo bookkeeping: insert new objects, remove old ones; undo restores both exactly (objects kept)."""

    def _apply(self, scene):
        self._swap_store(scene)
        self.removed = []
        live = []
        for o in self.remove_objs:                     # by identity: a merge keeps the first source's id
            i = next((k for k, x in enumerate(scene.objects) if x is o), -1)
            if i < 0:
                # fs-core-3 11-14 (undo all 13 / redo all 13 did not round-trip, fs-tools-2's u3/mile): a redo of an
                # earlier step that makes its objects anew (LineCut's pieces: new instances, the same ids) leaves the
                # scene without the instance this step removed when it ran, and the redo raised 'not in list'. The
                # source is the scene's object with that id (self.obj is not in the scene yet, so an id it shares
                # with its first source finds the source); the live instance is the one removed, kept for the undo
                # and freed at 'committed'
                i = scene.index_of(o.id)
                if i < 0:
                    raise ValueError('%s: object %d is no longer in the scene' % (type(self).__name__, o.id))
            ob = scene.objects.pop(i)
            self.removed.append((i, ob))
            live.append(ob)
        self.remove_objs = live
        scene.objects.insert(min(self.insert_at, len(scene.objects)), self.obj)
        scene.next_id = self.after_next
        scene.touch(structural=True, index=True)

    def undo(self, scene):
        scene.objects.remove(self.obj)
        for i, o in reversed(self.removed):
            scene.objects.insert(i, o)
        scene.next_id = self.prev_next
        scene.touch(structural=True, index=True)

    # ---- the undo journal (fs-tools-2 11-14; fs-shell-4's t67): a Merge at the end of a session gave
    # no undo state, so its save carried no journal (core/history._undo_snapshot stops at the first such step), the reopen
    # replayed every step in the background and an edit meanwhile dropped the history. The swap's bookkeeping and every
    # REMOVED object in full (its description, its bricks: delta, mask, directions, region ids, L) go to the journal, so an
    # undo after a reopen puts the sources back exactly; the inserted object is the scene's own (by id)
    def _swap_store(self, scene):
        self._st = scene.store                     # the store the removed objects' bricks stay in while this is done

    def undo_state(self):
        if self.obj is None or getattr(self, 'removed', None) is None or getattr(self, '_st', None) is None:
            return None
        st = self._st
        removed = list(self.removed)
        meta = {'obj': int(self.obj.id), 'insert_at': int(self.insert_at), 'prev_next': int(self.prev_next),
                'after_next': int(self.after_next), 'removed': [[int(i), o.describe()] for i, o in removed]}
        # the removed objects' slots don't change while this step is done (they leave the store at 'committed'), so the
        # arrays are read when the journal writes them, by reference as BrushStroke's records
        slots = {o.id: dict(o.sculpt.slots) for _, o in removed}

        def make():
            A = {}
            for oid, sl in slots.items():
                keys = sorted(b for b, s_ in sl.items() if s_ >= 0)
                if not keys:
                    continue
                ss = [sl[b] for b in keys]
                A['o%d_bricks' % oid] = np.array(keys, np.int64)
                A['o%d_delta' % oid] = st.delta[ss].copy()
                A['o%d_Lb' % oid] = np.asarray(st.Lb[ss], np.float64).copy()
                if st.has('mask'):
                    A['o%d_mask' % oid] = st.mask[ss].copy()
                if st.has('mdir'):
                    A['o%d_mdir' % oid] = st.mdir[ss].copy()
                if st.has('rid'):
                    A['o%d_rid' % oid] = st.rid[ss].copy()
            return meta, A
        return make

    def restore_undo_state(self, scene, meta, A):
        self.obj = scene.get(int(meta['obj']))
        self.insert_at = int(meta['insert_at'])
        self.prev_next = int(meta['prev_next'])
        self.after_next = int(meta['after_next'])
        st = scene.store
        self._st = st
        self.removed = []
        for i, od in meta['removed']:
            o = SceneObject(od['id'], od['name'], ops.Node.from_dict(od['base']), transform=od['transform'],
                            colour=od['colour'], material=od['material'], h=od.get('sculpt_h'))
            o.detail = od['detail']
            o.visible = od['visible']
            o.mask_inv = bool(od.get('mask_invert', False))
            for k in ('locked', 'root', 'part', 'garment', 'group'):
                setattr(o, k, od.get(k))
            o.layers = copy.deepcopy(od.get('layers') or [])
            o.snapshots = copy.deepcopy(od.get('snapshots') or [])
            o.regions = {int(k): dict(v) for k, v in (od.get('regions') or {}).items()}
            key = 'o%d_bricks' % o.id
            if key in A:
                bricks = np.asarray(A[key])
                o.sculpt.ensure_index(bricks.min(0), bricks.max(0))
                for q, b in enumerate(map(tuple, bricks.tolist())):
                    s_ = st.alloc()
                    st.delta[s_] = A['o%d_delta' % o.id][q]
                    st.Lb[s_] = A['o%d_Lb' % o.id][q]
                    for nm in ('mask', 'mdir', 'rid'):
                        if ('o%d_%s' % (o.id, nm)) in A:
                            st.ensure(nm)
                            getattr(st, nm)[s_] = A['o%d_%s' % (o.id, nm)][q]
                        elif st.has(nm):
                            getattr(st, nm)[s_] = 0
                    o.sculpt.set_slot(b, s_)
            self.removed.append((int(i), o))
        self.remove_objs = [o for _, o in self.removed]

    def evict_undo(self):
        return False


@command
class CombineObjects(_ObjectSwap):
    """Merge (op 'union') / Subtract / Intersect of 2+ objects into one new object (it keeps the first-selected
    object's id, colour and place in the list; hand2b: named 'A + B' / 'Merged (A, B +n)' unless a name is given); the sculpt layers carried exactly (module note). k: blend in metres (0 = sharp). frame: 'first' (the first-selected object's frame) or
    'world' (identity: local Mirror X is then the world midline). h: M's lattice (default min of the sources')."""
    label = 'Merge'

    def __init__(self, oids, op='union', k=0.0, frame='first', h=None, name=None):
        self.oids = [int(o) for o in oids]
        self.op = str(op)
        self.k = float(k)
        self.frame = str(frame)
        self.h = None if h is None else float(h)
        self.name = name
        self.obj = None
        self.report = {}

    def args(self):
        d = {'oids': self.oids, 'op': self.op, 'k': self.k, 'frame': self.frame}
        if self.h is not None:
            d['h'] = self.h
        if self.name is not None:
            d['name'] = self.name
        return d

    def nbytes(self):
        # hand2b: M's bricks AND the sources' (kept in memory for an exact undo) count against the undo cap
        n = len(self.obj.sculpt.slots) if self.obj is not None else 0
        n += sum(len(o.sculpt.slots) for o in getattr(self, 'remove_objs', []))
        return 256 + n * 2 * 4 * B ** 3

    def discard(self, scene, state):
        """hand2b (verifier item 16): undone and superseded -> M's bricks are freed; committed (no undo past it) ->
        the sources' bricks are freed"""
        if state == 'undone' and self.obj is not None:
            release_object_slots(scene, self.obj)
        elif state == 'committed':
            for o in getattr(self, 'remove_objs', []):
                release_object_slots(scene, o)

    def do(self, scene):
        if self.obj is not None:                       # redo: the same object back, the sources out again
            self._apply(scene)
            return self.obj
        if len(self.oids) < 2 or len(set(self.oids)) != len(self.oids):
            raise ValueError('Merge needs two or more different objects')
        if self.op not in OPS:
            raise ValueError('op %r' % self.op)
        srcs = [scene.get(o) for o in self.oids]
        self.prev_next = scene.next_id
        first = srcs[0]
        if self.frame == 'world':
            tM = {'translate': [0.0, 0.0, 0.0], 'rotate': [0.0, 0.0, 0.0], 'scale': 1.0}
        else:
            tM = copy.deepcopy(first.transform)
        RM, tMv, sM = _frame(tM)
        # hand2b (verifier item 14): the lattices compared in WORLD units (h x the object's scale); M's local h is the
        # finest world spacing over M's own scale. Before, local h values of differently scaled frames were mixed.
        # Only SCULPTED parts set it (an unsculpted part has nothing to carry; its finer lattice would only multiply
        # the support), else the finest of all.
        hw = [o.sculpt.h * _frame(o.transform)[2] for o in srcs]
        hws = [x for o, x in zip(srcs, hw) if o.sculpt.slots] or hw
        hM = self.h if self.h else min(hws) / sM
        kl = self.k / sM
        est = sum(_support_core(o, RM, tMv, sM, hM) for o in srcs)   # >= the undilated union's size

        def _refuse(n_need):
            nb = sum(len(o.sculpt.slots) for o in srcs)
            raise ValueError('Merge refused before allocating: the %d sculpt bricks of the parts would need about %d '
                             'bricks (about %.0f MB) on the merged %.2f mm lattice (limit %d). Parts of very different '
                             'scale or lattice are the usual cause; merge them at a coarser lattice (h) or rescale '
                             'first.' % (nb, n_need, n_need * 2 * 4 * B ** 3 / 2 ** 20, hM * sM * 1e3,
                                         MAX_MERGE_BRICKS))
        if est > 8 * MAX_MERGE_BRICKS:               # a clear blow-up: refused before even listing the support
            _refuse(est)
        mid = first.id                                 # M keeps the first source's id (selection, jobs, tools follow)
        parts = []
        for i, o in enumerate(srcs):
            Rs, ts, ss = _frame(o.transform)
            R = RM.T @ Rs
            t = RM.T @ (ts - tMv) / sM
            s = ss / sM
            sub = _renumber(o.base, scene)
            x = ops.Node('transform', {'translate': [float(v) for v in t], 'rotate': _mat_to_euler(R),
                                       'scale': [s, s, s]}, [sub], nid=scene.new_id(), name='part %s' % o.name)
            # hand2b (verifier items 4, 5a): only a union's operands after the first carry a guard, and it is the
            # exact one (skips only where min / smin with the parts before it is exact: program.OP_GUARD_U). The
            # hand2f box guard on every part returned its box bound wherever a part's sculpt had moved the surface
            # outside the part's own box, and the merged delta (computed against the full field) then put the
            # surface off along the box boundary. Subtract / intersect operands are never guarded (a lower bound
            # cannot stand in for B of max(a, -b) or max(a, b)).
            if self.op == 'union' and i > 0:
                x = ops.Node('cull', {'pad': 0.0, 'union_k': float(kl)}, [x], nid=scene.new_id(),
                             name='part %s' % o.name)
            parts.append(x)
        acc = parts[0]
        for p in parts[1:]:
            acc = ops.Node(self.op, {'blend': float(kl)}, [acc, p], nid=scene.new_id(),
                           name={'union': 'merge', 'subtract': 'subtract', 'intersect': 'intersect'}[self.op])
        # hand2b (verifier item 8): the merged object is named for what it holds, not after the first part alone
        # ('Head' for a whole merged figure); its id and colour still come from the first-selected part
        if self.name:
            name = self.name
        elif len(srcs) == 2:
            name = '%s + %s' % (srcs[0].name, srcs[1].name)
        else:
            name = 'Merged (%s, %s +%d)' % (srcs[0].name, srcs[1].name, len(srcs) - 2)
        M = SceneObject(mid, name, acc, transform=tM, colour=copy.deepcopy(first.colour),
                        material=copy.deepcopy(first.material), h=hM)
        # sculpt: the support S over every sculpted source
        S = [_mapped_bricks(o, RM, tMv, sM, hM) for o in srcs]
        S = _unique_rows(np.concatenate(S)) if S else np.zeros((0, 3), np.int64)
        if len(S) > MAX_MERGE_BRICKS:
            scene.next_id = self.prev_next            # the part trees' ids are given back
            _refuse(len(S))
        insert_at = min(scene.index_of(o) for o in self.oids)
        st = scene.store
        scene.objects.append(M)                        # while the kernel reads the sources (removed below)
        self.report = {'bricks': 0, 'support': len(S)}
        try:
            if len(S):
                arr = np.ascontiguousarray(S, np.int64)   # sorted (lexicographic rows)
                M.sculpt.ensure_index(arr.min(0), arr.max(0))
                slots = st.alloc_many(len(arr))           # READY 2: at once (a Python loop per brick before)
                M.sculpt.set_slots(arr, slots)
                scene.touch(structural=True, index=True)
                pk = scene.pack()
                src_oi = np.array([scene.objects.index(o) for o in srcs], np.int64)
                nonzero = np.zeros(len(S), np.int64)
                if not st.has('mask') and any(getattr(o, 'mask_inv', False) for o in srcs):
                    # fs-brush 11-12 (fs-shell's t49 warm-up; lazyfix/repro_merge.py): a source with Mask > Invert gives
                    # the merged object stored mask 1 - 0 = 1, written by the kernel below; while the store's mask is
                    # lazy (0fb997e4) st.kview('mask') is ONE zero brick seen at every slot, so those writes went into it
                    # (every slot aliased, threads racing) and were lost - silently, or raised at the next pack. The mask
                    # is made real first, as every other writer does
                    st.ensure('mask')
                    pk = scene.pack()
                combine_kernel(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, st.delta, st.kview('mask'), src_oi,
                               OPS[self.op], kl, np.ascontiguousarray(RM), np.ascontiguousarray(tMv), sM, hM, arr,
                               slots, nonzero)
                z = nonzero == 0
                if st.has('rid') and any(st.rid[o.sculpt.slots[b]].any() for o in srcs for b in o.sculpt.slots):
                    z &= ~_carry_rid(st, srcs, M, arr, slots, RM, tMv, sM, hM)   # fs-shell 11-11: region ids
                if z.any():
                    st.release_many(slots[z])
                    M.sculpt.set_slots(arr[z], np.full(int(z.sum()), -1, np.int64))
                _reset_box(M.sculpt)
                self.report['bricks'] = len(M.sculpt.slots)
                # the check: max |F_M - OP(F_i)| at seeded off-lattice points inside the kept support
                kept = np.array(sorted(M.sculpt.slots), np.int64)
                if len(kept):
                    rng = np.random.default_rng(20261108)
                    pick = kept[rng.integers(0, len(kept), 4096)]
                    loc = (pick * B + rng.random((4096, 3)) * B) * hM
                    world = (RM @ (loc * sM).T).T + tMv
                    ref = np.empty(len(world))
                    _fold_points(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, st.delta, st.kview('mask'), src_oi,
                                 OPS[self.op], kl, sM, np.ascontiguousarray(world), ref)
                    self._check = (loc, ref * sM)
        except Exception:
            scene.objects.remove(M)
            for b, s_ in list(M.sculpt.slots.items()):
                st.release(s_)
            scene.next_id = self.prev_next
            scene.touch(structural=True, index=True)
            raise
        scene.objects.remove(M)
        self.obj = M
        self.after_next = scene.next_id
        self.remove_objs = list(srcs)
        self.insert_at = insert_at
        self._apply(scene)
        oi = scene.objects.index(M)
        _measure_all(scene, oi)
        chk = getattr(self, '_check', None)
        if chk is not None:
            pk = scene.pack()
            got = np.empty(len(chk[0]))
            _eval_exact(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi,
                        np.ascontiguousarray(chk[0]), got)
            err = np.abs(got * sM - chk[1])
            near = np.abs(chk[1]) < 0.003
            self.report['max_err_m'] = float(err.max())
            self.report['max_err_near_m'] = float(err[near].max()) if near.any() else 0.0
            self._check = None
        # touching check (a message, not a refusal)
        boxes = []
        for o in srcs:
            _, w0, w1 = scene._object_row(o)
            boxes.append((w0, w1))
        apart = []
        for i in range(1, len(srcs)):
            gap = min(float(np.max(np.maximum(boxes[i][0] - boxes[j][1], boxes[j][0] - boxes[i][1])))
                      for j in range(len(srcs)) if j != i)
            if gap > self.k + 2 * hM:
                apart.append(srcs[i].name)
        self.report['apart'] = apart
        # hand2b: compared in world units (a fine part merged into a coarser frame is reported too)
        self.report['resampled'] = sorted({'%.2g mm' % (hw_ * 1e3) for o, hw_ in zip(srcs, hw)
                                           if o.sculpt.slots and abs(hw_ - hM * sM) > 1e-12})
        return M


@command
class DuplicateObject(_ObjectSwap):
    """A copy of one object beside it (offset in metres, world), the sculpt bricks copied exactly."""
    label = 'Duplicate'

    def __init__(self, oid, offset=None, name=None):
        self.oid = int(oid)
        self.offset = None if offset is None else [float(v) for v in offset]
        self.name = name
        self.obj = None

    def args(self):
        d = {'oid': self.oid}
        if self.offset is not None:
            d['offset'] = self.offset
        if self.name is not None:
            d['name'] = self.name
        return d

    def nbytes(self):
        n = len(self.obj.sculpt.slots) if self.obj is not None else 0
        return 256 + n * 2 * 4 * B ** 3

    def discard(self, scene, state):
        if state == 'undone' and self.obj is not None:      # hand2b: the copy's bricks (verifier item 16)
            release_object_slots(scene, self.obj)

    def do(self, scene):
        if self.obj is not None:
            self._apply(scene)
            return self.obj
        o = scene.get(self.oid)
        self.prev_next = scene.next_id
        t = copy.deepcopy(o.transform)
        off = self.offset
        if off is None:
            lo, hi = o.local_box()
            off = [float(hi[0] - lo[0]) * float(t['scale']) * 1.1, 0.0, 0.0]
        t['translate'] = [float(a) + float(b) for a, b in zip(t['translate'], off)]
        mid = scene.new_id()
        base = _renumber(o.base, scene)
        M = SceneObject(mid, self.name or o.name + ' copy', base, transform=t, colour=copy.deepcopy(o.colour),
                        material=copy.deepcopy(o.material), h=o.sculpt.h)
        M.detail = copy.deepcopy(o.detail)
        M.mask_inv = o.mask_inv
        M.regions = {k: dict(v) for k, v in getattr(o, 'regions', {}).items()}   # fs-shell 11-11
        st = scene.store
        keys = sorted(o.sculpt.slots)
        if keys:
            arr = np.array(keys, np.int64)
            M.sculpt.ensure_index(arr.min(0), arr.max(0))
            for b in keys:
                s0 = o.sculpt.slots[b]
                s = st.alloc()
                st.delta[s] = st.delta[s0]
                st.mask[s] = st.mask[s0]
                st.mdir[s] = st.mdir[s0]
                st.rid[s] = st.rid[s0]           # fs-shell 11-11: region ids
                st.Lb[s] = st.Lb[s0]
                M.sculpt.set_slot(b, s)
        self.obj = M
        self.after_next = scene.next_id
        self.remove_objs = []
        self.insert_at = scene.index_of(o.id) + 1
        self._apply(scene)
        return M


def _mirror_x_bricks(store, slots):
    """{brick: slot} -> {brick: (delta, mask, rid)} with node (i, j, k) -> (-i, j, k). Exact (each node one source).
    (fs-shell 11-11: the region ids mirrored with them)"""
    out = {}
    for (bx, by, bz), s in slots.items():
        D = store.delta[s]
        Mk = store.mask[s]
        R = store.rid[s]
        for tb in ((-bx - 1, by, bz), (-bx, by, bz)):
            if tb not in out:
                out[tb] = (np.zeros((B, B, B), np.float32), np.zeros((B, B, B), np.float32),
                           np.zeros((B, B, B), np.uint16))
        t1 = out[(-bx - 1, by, bz)]
        t0 = out[(-bx, by, bz)]
        t1[0][1:B] = D[B - 1:0:-1]          # new local l = 1..7  <-  old a = 8 - l = 7..1
        t1[1][1:B] = Mk[B - 1:0:-1]
        t1[2][1:B] = R[B - 1:0:-1]
        t0[0][0] = D[0]                     # new local 0  <-  old a = 0
        t0[1][0] = Mk[0]
        t0[2][0] = R[0]
    return out


@command
class MirrorDuplicate(_ObjectSwap):
    """A mirrored copy about the world plane x = 0 (her left built, her right made from it). Exact."""
    label = 'Mirror-duplicate X'

    def __init__(self, oid, plane='world_x', name=None):
        self.oid = int(oid)
        self.plane = str(plane)
        self.name = name
        self.obj = None

    def args(self):
        d = {'oid': self.oid, 'plane': self.plane}
        if self.name is not None:
            d['name'] = self.name
        return d

    def nbytes(self):
        n = len(self.obj.sculpt.slots) if self.obj is not None else 0
        return 256 + n * 2 * 4 * B ** 3

    def discard(self, scene, state):
        if state == 'undone' and self.obj is not None:      # hand2b: the copy's bricks (verifier item 16)
            release_object_slots(scene, self.obj)

    def do(self, scene):
        if self.obj is not None:
            self._apply(scene)
            return self.obj
        if self.plane != 'world_x':
            raise ValueError('plane %r (world_x only)' % self.plane)
        o = scene.get(self.oid)
        self.prev_next = scene.next_id
        Sx = np.diag([-1.0, 1.0, 1.0])
        R, t, s = _frame(o.transform)
        R2 = Sx @ R @ Sx
        t2 = Sx @ t
        mid = scene.new_id()
        sub = _renumber(o.base, scene)
        base = ops.Node('transform', {'translate': [0.0, 0.0, 0.0], 'rotate': [0.0, 0.0, 0.0],
                                      'scale': [-1.0, 1.0, 1.0]}, [sub], nid=scene.new_id(), name='mirror x')
        tr = {'translate': [float(v) + 0.0 for v in t2], 'rotate': _mat_to_euler(R2), 'scale': float(s)}
        nm = self.name or (o.name + ' mirror')
        M = SceneObject(mid, nm, base, transform=tr, colour=copy.deepcopy(o.colour),
                        material=copy.deepcopy(o.material), h=o.sculpt.h)
        M.detail = copy.deepcopy(o.detail)
        M.mask_inv = o.mask_inv
        st = scene.store
        mb = _mirror_x_bricks(st, o.sculpt.slots)
        keys = sorted(b for b, (d, m, r) in mb.items() if d.any() or m.any() or r.any())
        M.regions = {k: dict(v) for k, v in getattr(o, 'regions', {}).items()}   # fs-shell 11-11
        if keys:
            arr = np.array(keys, np.int64)
            M.sculpt.ensure_index(arr.min(0), arr.max(0))
            for b in keys:
                sl = st.alloc()
                st.delta[sl] = mb[b][0]
                st.mask[sl] = mb[b][1]
                st.rid[sl] = mb[b][2]
                M.sculpt.set_slot(b, sl)
        self.obj = M
        self.after_next = scene.next_id
        self.remove_objs = []
        self.insert_at = scene.index_of(o.id) + 1
        self._apply(scene)
        _measure_all(scene, scene.index_of(M.id))
        return M


@command
class MaskEdit(Command):
    """'clear': the object's mask 0 everywhere and invert off; 'invert': toggle the object's invert flag (the brushes
    and the mask display read 1 - m, so the unsculpted area inverts too). One undo step each."""
    label = 'Mask'

    def __init__(self, oid, op='clear'):
        self.oid = int(oid)
        self.op = str(op)

    def args(self):
        return {'oid': self.oid, 'op': self.op}

    def nbytes(self):
        return 256 + sum(m.nbytes for m in getattr(self, 'old', {}).values())

    def do(self, scene):
        o = scene.get(self.oid)
        self.inv0 = o.mask_inv
        self.old = {}
        self.oldd = {}
        if self.op == 'clear':
            st = scene.store
            for b, s in o.sculpt.slots.items():
                if st.mask[s].any() or st.mdir[s].any():
                    self.old[b] = st.mask[s].copy()
                    self.oldd[b] = st.mdir[s].copy()
                    st.mask[s] = 0.0
                    st.mdir[s] = 0
            o.mask_inv = False
        elif self.op == 'invert':
            o.mask_inv = not o.mask_inv
        else:
            raise ValueError('mask op %r' % self.op)
        scene.touch(structural=True)

    def undo(self, scene):
        o = scene.get(self.oid)
        st = scene.store
        for b, m in self.old.items():
            st.mask[o.sculpt.slots[b]] = m
        for b, dd in getattr(self, 'oldd', {}).items():
            st.mdir[o.sculpt.slots[b]] = dd
        o.mask_inv = self.inv0
        scene.touch(structural=True)

    # fs-trust 11-11: the undo journal (core/undojournal.py): the old mask planes, so a reopen's history goes past it
    def undo_state(self):
        old, oldd, inv0 = dict(self.old), dict(getattr(self, 'oldd', {})), bool(self.inv0)

        def make():
            ks = sorted(old)
            A = {'keys': np.array(ks, np.int64).reshape(-1, 3)}
            if ks:
                A['mask'] = np.stack([old[k] for k in ks]).astype(np.float32, copy=False)
                A['mdir'] = np.stack([oldd.get(k, np.zeros(old[k].shape + (3,), np.int8)) for k in ks]).astype(np.int8, copy=False)
            return {'kind': 'mask_edit', 'inv0': inv0}, A
        return make

    def restore_undo_state(self, scene, meta, A):
        ks = [tuple(int(v) for v in r) for r in np.asarray(A['keys']).reshape(-1, 3).tolist()]
        self.old = {k: np.array(A['mask'][i]) for i, k in enumerate(ks)} if ks else {}
        self.oldd = {k: np.array(A['mdir'][i]) for i, k in enumerate(ks)} if ks else {}
        self.inv0 = bool(meta['inv0'])
