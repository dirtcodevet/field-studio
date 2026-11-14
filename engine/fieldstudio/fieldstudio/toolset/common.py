"""Small shared helpers for the toolset (frames, boxes, lattice nodes, the exact sculpt rule)."""
import numpy as np
from numba import njit, prange

from ..core import ops
from ..core import program as P

B = P.BRICK


def frame(o):
    t = o.transform
    return ops.rot_matrix(t['rotate']), np.asarray(t['translate'], float), max(float(t['scale']), 1e-9)


def to_world(o, X):
    R, t, s = frame(o)
    return (np.asarray(X, float) * s) @ R.T + t


def to_local(o, W):
    R, t, s = frame(o)
    return ((np.asarray(W, float) - t) @ R) / s


def dir_to_local(o, d):
    R, t, s = frame(o)
    return np.asarray(d, float) @ R


def world_box(o):
    lo, hi = o.local_box()
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = to_world(o, c)
    return w.min(0), w.max(0)


def brick_nodes(keys, h):
    """local node coordinates of bricks (n,3 int keys) -> (n, 8, 8, 8, 3)"""
    keys = np.asarray(keys, np.int64).reshape(-1, 3)
    o = np.arange(B)
    I, J, K = np.meshgrid(o, o, o, indexing='ij')
    off = np.stack([I, J, K], -1)
    return (keys[:, None, None, None, :] * B + off[None]).astype(np.float64) * h


def bricks_in_box(lo_local, hi_local, h, pad=1):
    blo = np.floor(np.asarray(lo_local) / (B * h)).astype(np.int64) - pad
    bhi = np.floor(np.asarray(hi_local) / (B * h)).astype(np.int64) + pad
    rng = [np.arange(blo[a], bhi[a] + 1) for a in range(3)]
    return np.stack(np.meshgrid(*rng, indexing='ij'), -1).reshape(-1, 3)


def local_box_of_world(o, lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    L = to_local(o, c)
    return L.min(0), L.max(0)


@njit(cache=True, parallel=True)
def eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts, out):
    for n in prange(pts.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        out[n] = P.eval_local_t(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts[n, 0], pts[n, 1], pts[n, 2],
                                vs, ps, 1e9)


@njit(cache=True, parallel=True)
def eval_base(code, prm, oi_tab, oi, pts, out):
    for n in prange(pts.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        out[n] = P.eval_prog_t(code, prm, oi_tab[oi, P.OI_C0], oi_tab[oi, P.OI_C1], pts[n, 0], pts[n, 1], pts[n, 2],
                               vs, ps, 1e9)


@njit(cache=True, parallel=True)
def eval_tree(code, prm, pts, out):
    """a compiled base tree (ops.compile_tree's code, prm) at local points: no sculpt"""
    for n in prange(pts.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        out[n] = P.eval_prog_t(code, prm, 0, code.shape[0], pts[n, 0], pts[n, 1], pts[n, 2], vs, ps, 1e9)


def field_local(scene, o, X, base_only=False):
    """object o's field (local units) at local points X, as the scene is now"""
    pk = scene.pack()
    oi = scene.index_of(o.id)
    X = np.ascontiguousarray(np.asarray(X, float).reshape(-1, 3))
    out = np.empty(len(X))
    if base_only:
        eval_base(pk.code, pk.prm, pk.oi_tab, oi, X, out)
    else:
        eval_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, X, out)
    return out


def write_bricks(scene, o, keys, new_delta, rec, need_new=None):
    """write delta values (n, 8, 8, 8) into object o's bricks (allocating those not held where need_new says so),
    recording undo data in rec (a brush.StrokeRecord). -> the bricks written"""
    g = o.sculpt
    st = scene.store
    keys = [tuple(int(v) for v in k) for k in np.asarray(keys, np.int64).reshape(-1, 3)]
    newk = [k for i, k in enumerate(keys) if g.get_slot(k) < 0 and (need_new is None or need_new[i])]
    if newk:
        ns = st.alloc_many(len(newk))
        st.delta[ns] = 0.0
        st.Lb[ns] = 0.0
        if st.has('mask'):
            st.mask[ns] = 0.0
        g.set_slots(newk, ns)
        for k in newk:
            rec.new.append((o.id, k))
            rec.new_set.add((o.id, k))
    done = []
    for i, k in enumerate(keys):
        sl = g.get_slot(k)
        if sl < 0:
            continue
        key = (o.id, k)
        if key not in rec.snap and key not in rec.new_set:
            rec.snap[key] = (st.delta[sl].copy(), st.mask[sl].copy(), float(st.Lb[sl]))
        st.delta[sl] = np.asarray(new_delta[i], np.float32)
        done.append(k)
    scene.touch(index=bool(newk))
    return done


def finish_bricks(scene, o, done, rec, redistance=True):
    """the bound of each written brick (and its neighbours) re-measured; the release re-distance (fs-core's)"""
    from ..core import brush as BR
    oi = scene.index_of(o.id)
    near = set(done) | {(b[0] + dx, b[1] + dy, b[2] + dz) for b in done for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)}
    for b in near:
        sl = o.sculpt.get_slot(b)
        if sl >= 0 and (o.id, b) not in rec.snap and (o.id, b) not in rec.new_set and (o.id, b) not in rec.Lsnap:
            rec.Lsnap[(o.id, b)] = float(scene.store.Lb[sl])
    BR.measure_bricks(scene, oi, near)
    scene.refresh_object(oi)
    info = {}
    if redistance and done:
        try:
            info['redist'] = BR.redistance(scene, o.id, list(done), rec, force=True)
        except Exception as ex:
            info['redist_error'] = repr(ex)
    return info


class Protected(ValueError):
    """a tool refused: the part is protected (Mask / Protect > Protect part: "it is locked for every tool", design 3.2;
    core/scene.is_protected). offer = 'unprotect' (Edit > Unprotect makes it editable)"""
    offer = 'unprotect'


def refuse_protected(scene, oids, what):
    """raise Protected when any of oids is a protected part (every tool that would change it calls this first)"""
    from ..core.scene import is_protected
    for i in oids:
        o = scene.get(int(i))
        if o is not None and is_protected(o):
            raise Protected("%s: '%s' is protected; Unprotect it to change it (Edit > Unprotect)" % (what, o.name))
