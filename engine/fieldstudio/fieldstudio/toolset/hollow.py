"""Hollow / Shell, computed INWARD (TOOLSET_DESIGN 3.1, approved 11-14).

Design: add a shape (a primitive, a model or any other), choose 'hollow/shell' and the wall's width, and the
whole form is hollowed to its exact shape. Q6: the hollow shell is computed over the whole object, at the exact
distance (thickness) of the wall the user sets; it is not a brush.

HollowObject(oid, wall)
- the outside does not move: F' = max(F, delta - wall) keeps F (bit for bit) wherever the wall decides nothing;
- delta is the TRUE distance from the outer surface, measured to surface points read from the field (each lattice node
  beside the surface projected onto it: sub-cell exact; the nearest one found by an exact Euclidean distance transform,
  linear time; no mesh), so the wall is the set thickness
  everywhere, whatever the field's health deep inside (a sculpted object's stored values are distances only near its
  surface);
- where the form is thinner than twice the wall the cavity simply doesn't reach: no hole is made;
- the hollow is sealed until a Cut opens it (a sphere made hollow with a 5 mm wall, then cut in half, is a
  bowl);
- RE-HOLLOW is the same command on a hollow object: any enclosed cavity is filled first (the empty region not connected
  to the outside), then the inside is derived again from the outside at the wall given (edits to the inner face are
  replaced: the window warns first).
Today's core Shell (ops 'shell': |F| - t / 2) centres the wall on the surface and grows the outside by half the wall; it
stays for old trees. Live-until-Apply (the inside following later outside edits) needs the object's field evaluated as
max(G, inside(G)) at every reader (CPU program, card program, card strokes): a proposal for fs-shell / fs-core.
"""
import numpy as np
from scipy import ndimage

from ..core import brush as BR
from ..core.history import Command, command
from ..warp.bake import RecUndo, W_BAND
from . import common as C

B = C.B
CAVITY_BAND = 1.0         # bricks: cavity values beyond the inner face are truncated at this distance (a lower bound)
MAX_NODES = 48_000_000    # the census grid's cap at the object's lattice spacing (a sculpted head at 0.67 mm: 18 M,
                          # ~0.7 GB peak: float32 grids, the gradient only at surface nodes, chunked lookups)
SLAB = 8                  # x planes of the grid evaluated at a time (the field's points never all in memory)


def _grid(o, pad=2):
    """a brick-aligned grid over the object's box: (h, i0 (a multiple of the brick), i1)"""
    h = o.sculpt.h
    lo, hi = o.local_box()
    i0 = (np.floor(lo / h / B).astype(np.int64) - pad) * B
    i1 = (np.floor(hi / h / B).astype(np.int64) + pad + 1) * B - 1
    return h, i0, i1


def _field_grid(scene, o, h, i0, shape):
    F = np.empty(shape, np.float32)
    ay = (i0[1] + np.arange(shape[1])) * h
    az = (i0[2] + np.arange(shape[2])) * h
    for x0 in range(0, shape[0], SLAB):
        ax = (i0[0] + np.arange(x0, min(x0 + SLAB, shape[0]))) * h
        X = np.stack(np.meshgrid(ax, ay, az, indexing='ij'), -1).reshape(-1, 3)
        F[x0:x0 + len(ax)] = C.field_local(scene, o, X).reshape(len(ax), shape[1], shape[2])
    return F


def inner_distance(scene, o, F=None, grid=None, upto=None, exact_at=None):
    """-> (h, i0, shape, F, exterior, delta): the field on the object's lattice over its box (float32), the exterior
    (F > 0 and connected to the box's boundary), and the true distance of every non-exterior node to the outer surface
    (m, local; float32)"""
    h, i0, i1 = grid or _grid(o)
    shape = tuple(int(v) for v in (i1 - i0 + 1))
    if int(np.prod(shape)) > MAX_NODES:
        raise ValueError('Hollow: the object spans %s lattice nodes (cap %d): hollow it at a coarser lattice'
                         % ('x'.join(map(str, shape)), MAX_NODES))
    if F is None:
        F = _field_grid(scene, o, h, i0, shape)
    lab, n = ndimage.label(F > 0)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    exterior = np.isin(lab, border[border > 0])
    del lab
    solid = ~exterior
    # surface points: the solid nodes beside the exterior, each projected onto the zero set (sub-cell exact); the
    # gradient read at those nodes only (central differences on the grid)
    edge = solid & ndimage.binary_dilation(exterior)
    eidx = np.argwhere(edge)
    mx = np.array(shape) - 1
    ge = np.empty((len(eidx), 3))
    for a in range(3):
        ip = eidx.copy()
        im = eidx.copy()
        ip[:, a] = np.minimum(ip[:, a] + 1, mx[a])
        im[:, a] = np.maximum(im[:, a] - 1, 0)
        ge[:, a] = (F[ip[:, 0], ip[:, 1], ip[:, 2]].astype(float) - F[im[:, 0], im[:, 1], im[:, 2]]) / \
            ((ip[:, a] - im[:, a]) * h)
    gl2 = np.maximum((ge * ge).sum(-1), 1e-12)
    Pe = (eidx + i0) * h - (F[edge].astype(float) / gl2)[:, None] * ge
    eid = np.full(shape, -1, np.int32)
    eid[edge] = np.arange(len(eidx), dtype=np.int32)
    delta = np.where(solid, np.maximum(-F, 0.0), -1.0).astype(np.float32)
    # every solid node's nearest surface node by an exact Euclidean distance transform (linear time), then the distance
    # to that node's surface point; nodes within half a cell of the surface keep their own value (nearer than any wall
    # the lattice holds); an enclosed cavity's nodes are measured too (Re-hollow)
    deep = solid & ((F < -0.5 * h) | (F > 0))
    ind = np.empty((3,) + shape, np.int32)
    ndimage.distance_transform_edt(~edge, return_distances=False, return_indices=True, indices=ind)
    del edge
    flat = np.flatnonzero(deep)
    del deep
    tree = None
    CH = 1 << 21
    for c0 in range(0, len(flat), CH):
        fi = flat[c0:c0 + CH]
        di = np.stack(np.unravel_index(fi, shape), 1)
        nr = ind[:, di[:, 0], di[:, 1], di[:, 2]]
        X = (di + i0) * h
        d = np.linalg.norm(X - Pe[eid[nr[0], nr[1], nr[2]]], axis=1)
        if exact_at is not None:
            # near the inner face the distance decides where the face is: there it is measured exactly to every
            # surface point (a k-d tree); elsewhere the transform's nearest point is within a cell
            sh = np.abs(d - exact_at) < 3.0 * h
            if sh.any():
                if tree is None:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(Pe)
                d[sh], _ = tree.query(X[sh], workers=-1)
        if upto is not None:
            d = np.minimum(d, upto)
        delta.reshape(-1)[fi] = d
    return h, i0, shape, F, exterior, delta


@command
class HollowObject(RecUndo, Command):
    """Hollow the object inward to a wall (m, world). On a hollow object: Re-hollow (its cavity filled, then derived
    again from the outside)."""
    label = 'Hollow'

    def __init__(self, oid, wall):
        self.oid = int(oid)
        self.wall = float(wall)
        self.rec = None
        self.info = {}

    def args(self):
        return {'oid': self.oid, 'wall': self.wall}

    def do(self, scene):
        C.refuse_protected(scene, [self.oid], 'Hollow')
        o = scene.get(self.oid)
        s = C.frame(o)[2]
        t = self.wall / s
        h, i0, shape, F, exterior, delta = inner_distance(scene, o, upto=t + CAVITY_BAND * B * o.sculpt.h, exact_at=t)
        solid = ~exterior
        del exterior
        Fn = np.where(solid & (F > 0), -delta, F)            # an old cavity filled (its distance to the outside)
        np.maximum(Fn, delta - t, out=Fn, where=solid)       # the cavity: farther than the wall from the outside
        change = Fn != F
        self.info = {'wall_mm': round(self.wall * 1e3, 4), 'grid': list(shape),
                     'cavity_nodes': int((solid & (Fn > 0)).sum()), 'refilled_nodes': int((solid & (F > 0)).sum())}
        if not change.any():
            self.info['note'] = 'thinner than twice the wall everywhere: no cavity'
            self.rec = BR.StrokeRecord()
            return self.info
        # the bricks holding a changed node (the grid is brick-aligned): written whole (base + new value), the rest of
        # their nodes unchanged
        nb = np.array(shape) // B
        bmask = change.reshape(nb[0], B, nb[1], B, nb[2], B).any(axis=(1, 3, 5))
        bl = np.argwhere(bmask)
        bkeys = bl + i0 // B
        st = scene.store
        g = o.sculpt
        self.rec = BR.StrokeRecord()
        done = []
        CHB = 4096
        for c0 in range(0, len(bkeys), CHB):
            kk = bkeys[c0:c0 + CHB]
            ll = bl[c0:c0 + CHB]
            sl = [(slice(a * B, a * B + B), slice(b * B, b * B + B), slice(c * B, c * B + B)) for a, b, c in ll]
            Fnb = np.stack([Fn[q] for q in sl]).astype(np.float64)
            chb = np.stack([change[q] for q in sl])
            X = C.brick_nodes(kk, h).reshape(-1, 3)
            Fb = C.field_local(scene, o, X, base_only=True).reshape(Fnb.shape)
            cur = np.stack([st.delta[g.get_slot(tuple(k))].astype(np.float64) if g.get_slot(tuple(k)) >= 0
                            else np.zeros((B, B, B)) for k in kk])
            nd = np.where(chb, np.clip(Fnb, -W_BAND * h, W_BAND * h) - Fb, cur)      # fs-core's band
            done += C.write_bricks(scene, o, kk, nd, self.rec)
        self.info.update(C.finish_bricks(scene, o, done, self.rec, redistance=False))
        self.info['bricks'] = len(done)
        return self.info

    def undo(self, scene):
        if self.rec is not None:
            BR.undo_record(scene, self.rec)
            self.rec = None

    def nbytes(self):
        return 256 if self.rec is None else self.rec.nbytes()


def measure_wall(scene, o, n_dirs=200, seed=0):
    """the wall along rays from the object's centre outward: (outer radius - inner radius) per direction (world m),
    for directions where the ray meets the inner face; a check of 'reads 5.0 +- 0.1 mm everywhere it can'"""
    from ..warp.bake import object_field
    f = object_field(scene, o)
    lo, hi = C.world_box(o)
    c = 0.5 * (lo + hi)
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n_dirs, 3))
    D /= np.linalg.norm(D, axis=1)[:, None]
    R = float(np.linalg.norm(hi - lo))
    ts = np.linspace(0.0, R, 4000)
    out = []
    for d in D:
        v = f(c[None, :] + ts[:, None] * d[None, :])
        sgn = v < 0
        ch = np.nonzero(sgn[1:] != sgn[:-1])[0]
        if len(ch) >= 2 and not sgn[0]:
            def root(i):
                a, b = ts[i], ts[i + 1]
                for _ in range(40):
                    m = 0.5 * (a + b)
                    if (f((c + m * d)[None])[0] < 0) == sgn[i]:
                        a = m
                    else:
                        b = m
                return 0.5 * (a + b)
            r_in, r_out = root(ch[0]), root(ch[-1])
            out.append(r_out - r_in)
    return np.array(out)
