"""Shared helpers for the tools: frames, Euler angles, guarded subtrees, tool data on nodes, sculpt-grid writes.

Everything here builds core nodes (ops.Node) or writes the sculpt grid through the same undo record the brushes
use (brush.StrokeRecord), so the core keeps tracking L and the hash. No mesh anywhere.

Guard (D): taper(rate=0) is the identity map with L multiplier 1, and the core compiles it with a box guard
(child box + margin). Wrapped as offset(-e)( taper0( offset(+e)(child) ) ) the value inside the guard is the
child's (F - e + e) and outside it is >= margin + e, a lower bound of the child's value. With e = 2k, a smooth
boolean of blend k with this subtree has the same zero set as with the unguarded subtree (proof in CONTRACT.md,
'tools').
"""
import copy
import itertools
import json
import math

import numpy as np
from numba import njit, prange

from ..core import ops
from ..core import program as P
from ..core import brush as BR

TOOLDATA = 'tooldata:'


# ---- ids, frames, angles --------------------------------------------------------------------------------

class Ids:
    """Deterministic node-id counter starting at scene.next_id."""

    def __init__(self, start):
        self._c = itertools.count(int(start))
        self.last = int(start) - 1

    def __call__(self):
        self.last = next(self._c)
        return self.last

    @property
    def next_id(self):
        return self.last + 1


def mat_to_euler(R):
    """3x3 rotation -> Euler XYZ degrees with R = Rz Ry Rx (ops.rot_matrix's convention)."""
    R = np.asarray(R, float)
    sy = -R[2, 0]
    sy = min(1.0, max(-1.0, sy))
    y = math.asin(sy)
    if abs(math.cos(y)) > 1e-9:
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        z = 0.0
        x = math.atan2(-R[1, 2], R[1, 1])
    return [math.degrees(x), math.degrees(y), math.degrees(z)]


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-15 else v * 0.0


def frame_from(ey, ez_hint):
    """Right-handed orthonormal frame with y = ey, z as close to ez_hint as possible. Columns [ex, ey, ez]."""
    ey = unit(ey)
    ez = np.asarray(ez_hint, float) - ey * float(np.dot(ez_hint, ey))
    if np.linalg.norm(ez) < 1e-9:
        k = int(np.argmin(np.abs(ey)))
        e = np.zeros(3)
        e[k] = 1.0
        ez = e - ey * ey[k]
    ez = unit(ez)
    ex = np.cross(ey, ez)
    return np.column_stack([ex, ey, ez])


def rotation_between(a, b):
    """Smallest rotation taking unit a to unit b (Rodrigues); 180 deg about a deterministic perpendicular."""
    a = unit(a)
    b = unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-12:
        k = int(np.argmin(np.abs(a)))
        e = np.zeros(3)
        e[k] = 1.0
        ax = unit(np.cross(a, e))
        return 2.0 * np.outer(ax, ax) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def framed(child, origin, R, scale, ids, name=None):
    """transform node: child space -> local, child x axis = R[:,0] * scale[0], etc."""
    return ops.Node('transform', {'translate': [float(v) for v in origin], 'rotate': mat_to_euler(R),
                                  'scale': [float(v) for v in scale]}, [child], nid=ids(), name=name)


def guarded(child, k, ids, name=None):
    """Box guard around a subtree (see the module note). k: the blend radius the subtree will be joined with."""
    e = 2.0 * max(float(k), 0.0)
    inner = child
    if e > 0:
        inner = ops.Node('offset', {'distance': e}, [inner], nid=ids(), name='guard+')
    g = ops.Node('taper', {'rate': 0.0, 'g_min': 0.2}, [inner], nid=ids(), name='guard')
    if e > 0:
        g = ops.Node('offset', {'distance': -e}, [g], nid=ids(), name='guard-')
    if name:
        g.name = name
    return g


def chain(nodes, kind, blend, ids, name=None):
    """Left-deep boolean chain (value stack depth 2)."""
    acc = nodes[0]
    for n in nodes[1:]:
        acc = ops.Node(kind, {'blend': float(blend)}, [acc, n], nid=ids())
    if name:
        acc.name = name
    return acc


def chunked_union(nodes, blend, ids, chunk=8, join_k=0.0):
    """Union of many small nodes, guarded in chunks of 8 (a one-level bounding-box hierarchy).
    join_k: the blend the WHOLE union is later joined with (see tree_union: every guard inside is padded for it)."""
    if len(nodes) <= chunk:
        return chain(nodes, 'union', blend, ids)
    pad = max(float(blend), float(join_k))
    groups = []
    for i in range(0, len(nodes), chunk):
        part = nodes[i:i + chunk]
        groups.append(guarded(chain(part, 'union', blend, ids), pad, ids) if len(part) > 1 else part[0])
    return chain(groups, 'union', blend, ids)


def tree_union(nodes, blend, ids, chunk=8, join_k=0.0):
    """Union of many small nodes as a guarded tree (a bounding-box hierarchy, chunk-ary): a point far from a group
    pays one guard test for the whole group, so long strokes of many pieces stay cheap to evaluate.

    join_k: the blend radius the whole union will be joined with (a pencil's softness). pencil-locality fix
    (2026-11-08): every inner guard is padded for it (e = 2 max(blend, join_k), the module note's proof), not only
    the outer one. Before, the inner groups were guarded with the pieces' own blend (0 for a pencil), so a skipped
    group returned its box bound (>= its margin m, 2 mm for a stroke-sized group) and that bound went straight into
    the smooth join with the surface below: wherever bound - F_below < softness the join moved, in the shape of the
    group boxes. With softness > m the zero set itself moved outside the stroke (M: softness 3 mm, 83 um at 521 of
    400 000 surface points; 4 mm, 225 um at 705, ridge and valley alike, rectangles around the stroke); at the
    default 1.5 mm the traced field changed in box-shaped regions near the surface. With the pad a skipped group
    returns >= m + 2 join_k, so the join is exact wherever F_below <= m + join_k (union) or >= -(m + join_k)
    (subtract): the tracer's field equals the true field over the whole band."""
    pad = max(float(blend), float(join_k))
    level = list(nodes)
    while len(level) > chunk:
        nxt = []
        for i in range(0, len(level), chunk):
            part = level[i:i + chunk]
            nxt.append(guarded(chain(part, 'union', blend, ids), pad, ids) if len(part) > 1 else part[0])
        level = nxt
    return chain(level, 'union', blend, ids)


def set_data(node, data):
    node.name = TOOLDATA + json.dumps(data, sort_keys=True, separators=(',', ':'))


def get_data(node):
    if node is None or not node.name.startswith(TOOLDATA):
        return None
    return json.loads(node.name[len(TOOLDATA):])


def node_label(node):
    """Display name for a node (the UI can use this: tool-data nodes show as '<tool> data')."""
    d = get_data(node)
    if d is not None:
        return '%s data' % d.get('tool', 'tool')
    return node.name


def max_id(tree):
    return max(n.id for n in tree.walk())


def renumber(tree, ids):
    """Copy of a tree with fresh ids (for a subtree copied into another tree)."""
    t = ops.Node.from_dict(tree.to_dict()) if isinstance(tree, ops.Node) else ops.Node.from_dict(tree)
    for n in t.walk():
        n.id = ids()
    return t


def r7(v):
    """Round coordinates to 0.1 um for compact, deterministic tool data."""
    return [round(float(x), 7) for x in v]


# ---- object frames -------------------------------------------------------------------------------------

def obj_frame(obj):
    t = obj.transform
    return ops.rot_matrix(t['rotate']), np.asarray(t['translate'], float), max(float(t['scale']), 1e-6)


def to_local(obj, p):
    R, t, s = obj_frame(obj)
    return (R.T @ (np.asarray(p, float) - t)) / s


def dir_to_local(obj, d):
    R, t, s = obj_frame(obj)
    return R.T @ np.asarray(d, float)


def to_world(obj, q):
    R, t, s = obj_frame(obj)
    return R @ (np.asarray(q, float) * s) + t


def relative_transform(a_frame, b_frame):
    """Transform-node params that place B's local field inside A's local frame (B's values in A's units)."""
    Ra, ta, sa = a_frame
    Rb, tb, sb = b_frame
    R = Ra.T @ Rb
    t = Ra.T @ (tb - ta) / sa
    s = sb / sa
    return {'translate': [float(v) for v in t], 'rotate': mat_to_euler(R), 'scale': [s, s, s]}


# ---- evaluating a tree (and an object's sculpt) at points ------------------------------------------------

@njit(cache=True)
def _eval_tree(code, prm, pts, out):
    vs = np.empty(P.STACK)
    ps = np.empty(4 * P.STACK)
    for n in range(pts.shape[0]):
        out[n] = P.eval_prog(code, prm, 0, code.shape[0], pts[n, 0], pts[n, 1], pts[n, 2], vs, ps)


@njit(cache=True, nogil=True, parallel=True)
def _eval_range(code, prm, c0, c1, pts, out):
    # fs-core 11-14 (14 window stalls seen in the detail-band Smooth, 20:07-20:11): the layer tools evaluate the base at
    # every node of a brush box (3-7 M points a sample at r 22.9) on the engine thread; holding the GIL that starved the
    # window (fs-core/gil_probe.py: 81 ms a million points). Without the GIL and over the cores, chunk by chunk, the same
    # values point for point
    n = pts.shape[0]
    nch = (n + 4095) // 4096
    for c in prange(nch):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        for i in range(c * 4096, min(n, (c + 1) * 4096)):
            out[i] = P.eval_prog(code, prm, c0, c1, pts[i, 0], pts[i, 1], pts[i, 2], vs, ps)


@njit(cache=True)
def _sample_grid(index_flat, pool, oi_tab, oi, h, pts, out):
    for n in range(pts.shape[0]):
        out[n] = P.grid_sample(index_flat, pool, oi_tab, oi, h, pts[n, 0], pts[n, 1], pts[n, 2])


def eval_tree(tree, pts):
    code, prm, lo, hi, L = ops.compile_tree(tree)
    pts = np.ascontiguousarray(np.asarray(pts, float).reshape(-1, 3))
    out = np.empty(len(pts))
    _eval_tree(code, prm, pts, out)
    return out


def sample_sculpt(scene, oi, pts):
    """delta-F of scene.objects[oi]'s sculpt layer at local points."""
    pk = scene.pack()
    pts = np.ascontiguousarray(np.asarray(pts, float).reshape(-1, 3))
    out = np.empty(len(pts))
    _sample_grid(pk.index_flat, pk.delta, pk.oi_tab, oi, scene.objects[oi].sculpt.h, pts, out)
    return out


def local_field(scene, oi, tree=None):
    """f(local pts) = F_tree + dF_sculpt (tree defaults to the object's base)."""
    obj = scene.objects[oi]
    t = obj.base if tree is None else tree
    code, prm, lo, hi, L = ops.compile_tree(t)

    def f(pts):
        pts = np.ascontiguousarray(np.asarray(pts, float).reshape(-1, 3))
        out = np.empty(len(pts))
        _eval_tree(code, prm, pts, out)
        if obj.sculpt.slots:
            out += sample_sculpt(scene, oi, pts)
        return out
    return f


def grad(f, pts, eps=1e-5):
    pts = np.asarray(pts, float).reshape(-1, 3)
    g = np.empty_like(pts)
    for a in range(3):
        e = np.zeros(3)
        e[a] = eps
        g[:, a] = (f(pts + e) - f(pts - e)) / (2 * eps)
    return g


def project(f, pts, iters=4, max_step=None):
    """Newton projection of points onto the zero set of f; returns (points, unit normals)."""
    q = np.asarray(pts, float).reshape(-1, 3).copy()
    for _ in range(iters):
        v = f(q)
        g = grad(f, q)
        gg = np.maximum((g * g).sum(1), 1e-12)
        step = (v / gg)[:, None] * g
        if max_step is not None:
            sl = np.linalg.norm(step, axis=1)
            k = np.minimum(1.0, max_step / np.maximum(sl, 1e-15))
            step *= k[:, None]
        q -= step
    g = grad(f, q)
    n = g / np.maximum(np.linalg.norm(g, axis=1), 1e-12)[:, None]
    return q, n


# ---- sculpt-grid writes with an undo record (shared by layer smoothing and splice) ----------------------

_GATHER_BRICKS = __import__('os').environ.get('FS_GATHER_BRICKS', '1') != '0'   # fs-core 11-14: 0 = node by node


def gather_nodes(scene, oi, lo, dims, channel='delta'):
    """Values of the sculpt lattice nodes lo .. lo+dims-1 (int lattice coords) -> array (dims)."""
    g = scene.objects[oi].sculpt
    st = scene.store
    pool = st.delta if channel == 'delta' else st.mask
    out = np.zeros(tuple(int(d) for d in dims), np.float64)
    if not g.slots:
        return out
    if _GATHER_BRICKS and g.index.size:
        # fs-core 11-14: brick by brick (a brush box of 7 M nodes built three int64 index grids and a fancy-indexed read:
        # ~0.5 s and ~170 MB a layer-Smooth sample); the same values
        lo = np.asarray(lo, np.int64)
        hi = lo + np.asarray(dims, np.int64)            # exclusive
        b0 = lo >> 3
        b1 = (hi - 1) >> 3
        org = np.asarray(g.origin, np.int64)
        sh = np.array(g.index.shape, np.int64)
        r0 = np.maximum(b0 - org, 0)
        r1 = np.minimum(b1 - org + 1, sh)
        if np.any(r1 <= r0):
            return out
        sub = g.index[r0[0]:r1[0], r0[1]:r1[1], r0[2]:r1[2]]
        for t in np.argwhere(sub >= 0).tolist():
            slot = int(sub[t[0], t[1], t[2]])
            bk = (np.array(t, np.int64) + r0 + org) * 8          # the brick's first node
            n0 = np.maximum(bk, lo)
            n1 = np.minimum(bk + 8, hi)
            out[n0[0] - lo[0]:n1[0] - lo[0], n0[1] - lo[1]:n1[1] - lo[1], n0[2] - lo[2]:n1[2] - lo[2]] = \
                pool[slot, n0[0] - bk[0]:n1[0] - bk[0], n0[1] - bk[1]:n1[1] - bk[1], n0[2] - bk[2]:n1[2] - bk[2]]
        return out
    ii = np.arange(dims[0]) + lo[0]
    jj = np.arange(dims[1]) + lo[1]
    kk = np.arange(dims[2]) + lo[2]
    I, J, K = np.meshgrid(ii, jj, kk, indexing='ij')
    bx = (I >> 3) - g.origin[0]
    by = (J >> 3) - g.origin[1]
    bz = (K >> 3) - g.origin[2]
    sh = g.index.shape
    ok = (bx >= 0) & (by >= 0) & (bz >= 0) & (bx < sh[0]) & (by < sh[1]) & (bz < sh[2])
    slot = np.full(I.shape, -1, np.int64)
    slot[ok] = g.index[bx[ok], by[ok], bz[ok]]
    has = slot >= 0
    out[has] = pool[slot[has], I[has] & 7, J[has] & 7, K[has] & 7]
    return out


def write_nodes(scene, oi, lo, df, rec):
    """delta += df on lattice nodes lo .. lo+shape-1 of scene.objects[oi]; allocates only bricks with a
    non-zero write; snapshots for undo into rec (brush.StrokeRecord); re-measures the per-brick L."""
    obj = scene.objects[oi]
    g = obj.sculpt
    st = scene.store
    lo = np.asarray(lo, np.int64)
    nz = np.argwhere(df != 0.0)
    if not len(nz):
        return 0
    nodes = nz + lo
    # fs-core 11-14 (14 window stalls, 20:07-20:11, all here): np.unique(axis=0) sorts row views holding the GIL (1.2 s
    # for a million nodes, the window frozen throughout: fs-core/gil_probe.py); the brick keys packed into one int64 each
    # sort as plain integers. The same bricks, in the same (lexicographic) order
    bk = nodes >> 3
    b0 = bk.min(0)
    d_ = bk - b0
    m_ = d_.max(0) + 1
    u_ = np.unique((d_[:, 0] * m_[1] + d_[:, 1]) * m_[2] + d_[:, 2])
    bricks = np.stack([u_ // (m_[1] * m_[2]), (u_ // m_[2]) % m_[1], u_ % m_[2]], 1) + b0
    blo, bhi = bricks.min(0), bricks.max(0)
    if g.ensure_index(blo - 1, bhi + 1):
        scene.touch(index=True)
    touched = set()
    for b in map(tuple, bricks.tolist()):
        key = (obj.id, b)
        slot = g.get_slot(b)
        if slot < 0:
            slot = st.alloc()
            g.set_slot(b, slot)
            rec.new.append(key)
            rec.new_set.add(key)
        elif key not in rec.snap and key not in rec.new_set:
            rec.snap[key] = (st.delta[slot].copy(), st.mask[slot].copy(), float(st.Lb[slot]))
        touched.add(b)
    pk = scene.pack()
    scene.refresh_object(oi)
    pk = scene.pack()
    dm = np.zeros_like(df)
    BR.write_kernel(pk.index_flat, pk.delta, pk.mask, pk.oi_tab, oi, int(lo[0]), int(lo[1]), int(lo[2]),
                    np.ascontiguousarray(df, np.float64), dm)
    # re-measure L on the touched bricks and their -1 neighbours (cells straddle into them)
    near = set()
    for (x, y, z) in touched:
        for dx in (-1, 0):
            for dy in (-1, 0):
                for dz in (-1, 0):
                    b = (x + dx, y + dy, z + dz)
                    if g.get_slot(b) >= 0:
                        near.add(b)
    near = sorted(near)
    for b in near:
        key = (obj.id, b)
        if key not in rec.snap and key not in rec.new_set and key not in rec.Lsnap:
            rec.Lsnap[key] = float(st.Lb[g.get_slot(b)])
    arr = np.array(near, np.int64)
    out = np.zeros(len(near))
    BR.brick_lipschitz(pk.index_flat, pk.delta, pk.oi_tab, oi, g.h, arr, out)
    for b, L in zip(near, out):
        st.Lb[g.get_slot(b)] = L
    scene.refresh_object(oi)
    return len(nz)


def copy_sculpt(scene, src, dst):
    """Copy src's sculpt bricks into dst (same lattice spacing and frame); returns the slots made."""
    st = scene.store
    made = []
    if not src.sculpt.slots:
        return made
    dst.sculpt.h = src.sculpt.h
    keys = sorted(src.sculpt.slots)
    arr = np.array(keys, np.int64)
    dst.sculpt.ensure_index(arr.min(0), arr.max(0))
    for b in keys:
        s0 = src.sculpt.slots[b]
        s = st.alloc()
        st.delta[s] = st.delta[s0]
        st.mask[s] = st.mask[s0]
        st.mdir[s] = st.mdir[s0]
        st.rid[s] = st.rid[s0]                   # fs-shell 11-11: region ids
        st.Lb[s] = st.Lb[s0]
        dst.sculpt.set_slot(b, s)
        made.append(s)
    if getattr(src, 'regions', None):
        dst.regions = {k: dict(v) for k, v in src.regions.items()}
    return made


def release_sculpt(scene, obj, slots):
    st = scene.store
    for b in list(obj.sculpt.slots):
        s = obj.sculpt.slots[b]
        if s in slots:
            st.release(s)
            obj.sculpt.set_slot(b, -1)
    obj.sculpt.bmin = obj.sculpt.bmax = None
    if obj.sculpt.slots:
        ks = np.array(list(obj.sculpt.slots), np.int64)
        obj.sculpt.bmin, obj.sculpt.bmax = ks.min(0), ks.max(0)


class BaseEditMixin:
    """do/undo bookkeeping for commands that replace one object's base tree."""

    def _begin(self, scene, oid):
        o = scene.get(oid)
        self._old_tree = o.base.to_dict()
        self._prev_next = scene.next_id
        return o

    def _commit(self, scene, o, tree, ids):
        o.base = tree
        o.invalidate()
        o.compiled()
        scene.next_id = max(scene.next_id, ids.next_id, max_id(tree) + 1)
        scene.touch(structural=True)

    def _restore(self, scene, oid):
        o = scene.get(oid)
        o.base = ops.Node.from_dict(self._old_tree)
        o.invalidate()
        scene.next_id = self._prev_next
        scene.touch(structural=True)
