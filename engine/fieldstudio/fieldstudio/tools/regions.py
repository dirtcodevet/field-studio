"""Regions (fs-shell, 2026-11-11; design 4.8: "Each lattice node of an object carries a region id (16 bits). Regions are
painted with the Region brush, filled by lasso, or made from a mask or a protected part; each has a name and colour.")

The data (core/scene.py): scene.store.rid, a uint16 per sculpt-lattice node (node (i, j, k) of brick b at (b*8 + i)*h in
object local, as the sculpt delta); obj.regions = {id: {'name', 'colour'}}; id 0 = no region. A brick may hold ids with
zero delta and mask (kept, saved, hashed). The Region brush's dab write is fs-brush's (core/brush.py).

The writers here are logged, undoable commands. Each picks a set of nodes and writes one id there:
  RegionFromMask(oid, thr=0.5)                    the nodes whose mask >= thr (the Mask brush's paint)
  RegionFromPart(oid, part, within=0.5 mm)        the nodes on the object's surface band whose foot point lies within
                                                  `within` of another object (a protected part: the eyeball, the
                                                  lid, a tooth), i.e. where the surface touches or nears it
  RegionLasso(oid, cam, poly, w, h, through=False) the surface-band nodes whose foot point falls inside a screen
                                                  outline; through=False keeps only what the view shows (front facing,
                                                  not behind another surface), through=True takes every depth
  mode: 'new' (a new id; name and colour given or chosen), 'add' (into an existing id `rid`), 'erase' (back to 0: the
  given id only, or any id when rid is None)
"The surface band" = the nodes with |F| <= band (default 1.5 h): every node the tracer, the brush and a label read when
they ask "which region is this surface point in" (region_at: the nearest node). Nodes outside existing bricks get new
bricks (zero delta and mask), budgeted (max_new_bricks; a refusal says how many it would need).

Not here: hiding (the view state in the engine's 'view' job and the tracers' view-clip list), the Region brush, the
panel (ui/regions_ui.py)."""
import copy

import numpy as np

from ..core.history import Command, command
from ..core.scene import B
from .protected_merge import field_of, _to_world, _to_local, _rs

PALETTE = [(0.90, 0.42, 0.40), (0.38, 0.62, 0.92), (0.45, 0.80, 0.45), (0.95, 0.75, 0.30), (0.70, 0.50, 0.90),
           (0.30, 0.80, 0.80), (0.95, 0.55, 0.75), (0.65, 0.65, 0.35), (0.55, 0.40, 0.30), (0.40, 0.45, 0.70)]
MAX_NEW_BRICKS = 60_000          # ~0.55 GB of brick pools (delta, mask, mdir, rid, Lb): a refusal above it
_G = np.indices((B, B, B)).reshape(3, -1).T.astype(np.float64)        # (512, 3) node offsets in a brick


def next_id(o):
    return max([int(k) for k in (getattr(o, 'regions', None) or {})] + [0]) + 1


def default_colour(rid):
    return list(PALETTE[(int(rid) - 1) % len(PALETTE)])


def region_at(scene, oid, P):
    """the region id at world point(s) P (the nearest lattice node's; 0 where no brick holds one) -> int array"""
    o = scene.get(oid)
    P = np.atleast_2d(np.asarray(P, float))
    out = np.zeros(len(P), np.int64)
    if not getattr(o, 'regions', None):
        return out
    n = np.rint(_to_local(o, P) / o.sculpt.h).astype(np.int64)
    st = scene.store
    for j, (b, l) in enumerate(zip((n >> 3).tolist(), (n & 7).tolist())):
        s = o.sculpt.get_slot(tuple(b))
        if s >= 0:
            out[j] = int(st.rid[s][l[0], l[1], l[2]])
    return out


def counts(scene, oid):
    """{id: labelled nodes} of an object"""
    o = scene.get(oid)
    st = scene.store
    out = {}
    if not getattr(o, 'regions', None) or not st.has('rid'):
        return out                               # no table, no ids (the writers always add a row); none written anywhere
    for s in o.sculpt.slots.values():
        r = st.rid[s]
        if r.any():
            v, c = np.unique(r[r > 0], return_counts=True)
            for a, b in zip(v.tolist(), c.tolist()):
                out[a] = out.get(a, 0) + b
    return out


# ---- the surface band -------------------------------------------------------------------------------------------------
def _grad(scene, pk, o, W, e):
    g = np.empty_like(W)
    for a in range(3):
        d = np.zeros(3)
        d[a] = e
        g[:, a] = field_of(scene, pk, o, W + d) - field_of(scene, pk, o, W - d)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-30)


def band_nodes(scene, o, bricks, band):
    """the surface-band nodes of the given bricks (local brick coords (n, 3)) -> (brick index (m,), node (m, 3) int,
    foot points world (m, 3), normals world (m, 3)); band in object-local units"""
    pk = scene.pack()
    h = o.sculpt.h
    s = _rs(o)[2]
    if not len(bricks):
        z = np.zeros((0, 3))
        return np.zeros(0, np.int64), np.zeros((0, 3), np.int64), z, z
    bi_all, nd_all, W_all, F_all = [], [], [], []
    step = max(1, 2_000_000 // (B ** 3))
    for a in range(0, len(bricks), step):
        bb = bricks[a:a + step]
        X = ((bb[:, None, :] * B + _G[None]) * h).reshape(-1, 3)
        W = _to_world(o, X)
        F = field_of(scene, pk, o, W) / s
        k = np.abs(F) <= band
        if not k.any():
            continue
        idx = np.nonzero(k)[0]
        bi_all.append(a + idx // (B ** 3))
        nd_all.append(_G[idx % (B ** 3)].astype(np.int64))
        W_all.append(W[idx])
        F_all.append(F[idx])
    if not bi_all:
        z = np.zeros((0, 3))
        return np.zeros(0, np.int64), np.zeros((0, 3), np.int64), z, z
    bi, nd, W, F = (np.concatenate(v) for v in (bi_all, nd_all, W_all, F_all))
    N = _grad(scene, pk, o, W, 0.25 * h * s)
    foot = W - (F * s)[:, None] * N
    return bi, nd, foot, N


def surface_bricks(scene, o, lo, hi, band, limit=8_000_000):
    """the object's lattice bricks in the local box [lo, hi] that can hold surface-band nodes (the field at each brick's
    centre within its half diagonal + band, x1.5 for a field that is not a distance far from the surface)"""
    h = o.sculpt.h
    bs = B * h
    b0 = np.floor(np.asarray(lo, float) / bs).astype(np.int64)
    b1 = np.floor(np.asarray(hi, float) / bs).astype(np.int64)
    n = int(np.prod(b1 - b0 + 1))
    if n > limit:
        raise ValueError('Region: the area spans %d lattice bricks to search (limit %d); outline a smaller area or '
                         'zoom in' % (n, limit))
    ax = [np.arange(b0[a], b1[a] + 1) for a in range(3)]
    Bk = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    pk = scene.pack()
    s = _rs(o)[2]
    keep = []
    for a in range(0, len(Bk), 2_000_000):
        c = (Bk[a:a + 2_000_000] + 0.5) * bs
        F = field_of(scene, pk, o, _to_world(o, c)) / s
        keep.append(np.abs(F) <= 1.5 * (0.87 * bs + band))
    return Bk[np.concatenate(keep)] if keep else Bk[:0]


def _uniq(a):
    return np.unique(np.asarray(a, np.int64).reshape(-1, 3), axis=0) if len(a) else np.zeros((0, 3), np.int64)


# ---- the commands -----------------------------------------------------------------------------------------------------
class _RegionWrite(Command):
    """the shared write and undo: self.pick(scene, o) -> {brick tuple: (8, 8, 8) bool}"""
    label = 'Region'

    def _init(self, oid, mode, rid, name, colour, band, max_new_bricks):
        self.oid = int(oid)
        self.mode = str(mode)
        if self.mode not in ('new', 'add', 'erase'):
            raise ValueError('mode %r (new, add or erase)' % mode)
        self.rid = None if rid is None else int(rid)
        self.name = name
        self.colour = None if colour is None else [float(c) for c in colour]
        self.band = None if band is None else float(band)
        self.max_new_bricks = int(max_new_bricks)
        self._picked = None
        self.report = {}

    def _args(self):
        d = {'oid': self.oid, 'mode': self.mode, 'rid': self.rid, 'name': self.name, 'colour': self.colour}
        if self.band is not None:
            d['band'] = self.band
        if self.max_new_bricks != MAX_NEW_BRICKS:
            d['max_new_bricks'] = self.max_new_bricks
        return d

    def _band(self, o):
        return self.band / _rs(o)[2] if self.band is not None else 1.5 * o.sculpt.h

    def nbytes(self):
        n = len(getattr(self, '_old', {}) or {})
        return 256 + n * 2 * B ** 3 * 2

    def do(self, scene):
        o = scene.get(self.oid)
        st = scene.store
        if self._picked is None:
            self._picked = self.pick(scene, o)
        sel = {b: m for b, m in self._picked.items() if m.any()}
        new = [b for b in sel if o.sculpt.get_slot(b) < 0]
        if self.mode == 'erase':
            sel = {b: m for b, m in sel.items() if o.sculpt.get_slot(b) >= 0}
            new = []
        if len(new) > self.max_new_bricks:
            raise ValueError('Region: %d new lattice bricks would be needed (about %.0f MB; limit %d). Outline a '
                             'smaller area.' % (len(new), len(new) * 17 * B ** 3 / 2 ** 20, self.max_new_bricks))
        self._old_table = copy.deepcopy(getattr(o, 'regions', {}) or {})
        if self.mode == 'new' and self.rid is None:
            self.rid = next_id(o)                       # fixed on the first do: the log and a redo use the same id
        if self.mode != 'erase':
            if self.rid is None or self.rid <= 0 or self.rid > 65535:
                raise ValueError('Region: an id 1-65535 is needed (got %r)' % self.rid)
            if self.mode == 'add' and self.rid not in o.regions:
                raise ValueError('Region: object %d has no region %d' % (self.oid, self.rid))
        self._new = []
        if new:
            arr = np.array(new, np.int64)
            o.sculpt.ensure_index(arr.min(0), arr.max(0))
            for b in new:
                s = st.alloc()
                st.delta[s] = 0.0
                st.mask[s] = 0.0
                st.mdir[s] = 0
                st.rid[s] = 0
                st.Lb[s] = 0.0
                o.sculpt.set_slot(b, s)
                self._new.append(b)
        self._old = {}
        n = 0
        if self.mode != 'erase':
            st.ensure('rid')                             # fs-brush 11-12: painted in place below (lazy until now)
        for b, m in sel.items():
            s = o.sculpt.get_slot(b)
            r = st.rid[s]
            if self.mode == 'erase':
                m = m & ((r > 0) if self.rid is None else (r == self.rid))
                if not m.any():
                    continue
            self._old[b] = r.copy()
            r[m] = 0 if self.mode == 'erase' else self.rid
            n += int(m.sum())
        if self.mode == 'new':
            o.regions[self.rid] = {'name': self.name or 'Region %d' % self.rid,
                                   'colour': self.colour or default_colour(self.rid)}
        elif self.mode == 'erase':
            left = counts(scene, self.oid)
            for k in list(o.regions):
                if k not in left:
                    del o.regions[k]                     # an id with no node left leaves the table
        self.report = {'rid': self.rid, 'nodes': n, 'new_bricks': len(self._new), 'mode': self.mode}
        scene.touch(structural=True, index=bool(self._new))
        return [self.oid]

    def undo(self, scene):
        o = scene.get(self.oid)
        st = scene.store
        for b, r in self._old.items():
            st.rid[o.sculpt.get_slot(b)] = r
        for b in reversed(self._new):
            s = o.sculpt.get_slot(b)
            if s >= 0:
                st.release(s)
                o.sculpt.set_slot(b, -1)
        o.regions = copy.deepcopy(self._old_table)
        scene.touch(structural=True, index=bool(self._new))
        return [self.oid]


@command
class RegionFromMask(_RegionWrite):
    """the masked nodes (mask >= thr; an inverted mask read as it is seen) -> a region"""
    label = 'Region from mask'

    def __init__(self, oid, thr=0.5, mode='new', rid=None, name=None, colour=None, band=None,
                 max_new_bricks=MAX_NEW_BRICKS):
        self._init(oid, mode, rid, name, colour, band, max_new_bricks)
        self.thr = float(thr)

    def args(self):
        return dict(self._args(), thr=self.thr)

    def pick(self, scene, o):
        st = scene.store
        out = {}
        for b, s in o.sculpt.slots.items():
            m = st.mask[s]
            if getattr(o, 'mask_inv', False):
                m = 1.0 - m
            k = m >= self.thr
            if k.any():
                out[b] = k
        return out


@command
class RegionFromPart(_RegionWrite):
    """the object's surface where it touches or nears another object (within `within` m of it) -> a region named after
    that object"""
    label = 'Region from part'

    def __init__(self, oid, part, within=0.0005, mode='new', rid=None, name=None, colour=None, band=None,
                 max_new_bricks=MAX_NEW_BRICKS):
        self._init(oid, mode, rid, name, colour, band, max_new_bricks)
        self.part = int(part)
        self.within = float(within)

    def args(self):
        return dict(self._args(), part=self.part, within=self.within)

    def do(self, scene):
        if self.name is None and self.mode == 'new':
            self.name = scene.get(self.part).name
        return _RegionWrite.do(self, scene)

    def pick(self, scene, o):
        P = scene.get(self.part)
        _, w0, w1 = scene._object_row(P)
        pad = self.within + 3 * o.sculpt.h * _rs(o)[2]
        w0, w1 = np.asarray(w0, float) - pad, np.asarray(w1, float) + pad
        corners = np.array([[x, y, z] for x in (w0[0], w1[0]) for y in (w0[1], w1[1]) for z in (w0[2], w1[2])])
        L = _to_local(o, corners)
        band = self._band(o)
        bk = surface_bricks(scene, o, L.min(0), L.max(0), band)
        bi, nd, foot, _N = band_nodes(scene, o, bk, band)
        if not len(bi):
            return {}
        fp = field_of(scene, scene.pack(), P, foot)
        k = fp <= self.within
        self.report_pick = {'band_nodes': int(len(bi)), 'picked': int(k.sum())}
        return _to_bricks(bk, bi[k], nd[k])


@command
class RegionLasso(_RegionWrite):
    """the surface inside a screen outline -> a region. cam: Camera.to_dict(); poly: [[x, y], ...] pixels of a w x h
    view. through=False: only what the view shows (front facing, nothing in front of it); True: every depth."""
    label = 'Region lasso'

    def __init__(self, oid, cam, poly, w, h, through=False, mode='new', rid=None, name=None, colour=None, band=None,
                 max_new_bricks=MAX_NEW_BRICKS):
        self._init(oid, mode, rid, name, colour, band, max_new_bricks)
        self.cam = dict(cam)
        self.poly = [[float(x), float(y)] for x, y in poly]
        self.w, self.h = int(w), int(h)
        self.through = bool(through)

    def args(self):
        return dict(self._args(), cam=self.cam, poly=self.poly, w=self.w, h=self.h, through=self.through)

    def pick(self, scene, o):
        from ..render.camera import Camera
        cam = Camera.from_dict(self.cam)
        poly = np.asarray(self.poly, float)
        if len(poly) < 3:
            return {}
        band = self._band(o)
        hh = o.sculpt.h
        bs = B * hh
        s = _rs(o)[2]
        if self.through:
            _, w0, w1 = scene._object_row(o)
            w0, w1 = np.asarray(w0, float), np.asarray(w1, float)
            corners = np.array([[x, y, z] for x in (w0[0], w1[0]) for y in (w0[1], w1[1]) for z in (w0[2], w1[2])])
            L = _to_local(o, corners)
            lo, hi = L.min(0), L.max(0)
            # the object's bricks whose centre projects inside the outline's box (+ a brick), then the surface test
            ax = [np.arange(np.floor(lo[a] / bs), np.floor(hi[a] / bs) + 1).astype(np.int64) for a in range(3)]
            n = int(np.prod([len(a) for a in ax]))
            if n > 8_000_000:
                raise ValueError('Region lasso (through): %d bricks to search; zoom in or outline less' % n)
            Bk = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
            C = _to_world(o, (Bk + 0.5) * bs)
            x, y, z = project_many(cam, C, self.w, self.h)
            pad = _px_size(cam, C, self.w, self.h, bs * s)
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0)
            k = (z > 0) & (x >= x0 - pad) & (x <= x1 + pad) & (y >= y0 - pad) & (y <= y1 + pad)
            bk = surface_bricks_of(scene, o, Bk[k], band)
        else:
            hits = self._visible_hits(scene, cam, poly, o)
            if hits is None:
                return {}
            Hw, foot_px = hits
            nb = _to_local(o, Hw) / bs
            r = int(min(4, np.ceil(foot_px / bs) + 1))
            base = np.floor(nb).astype(np.int64)
            offs = np.array([[a, b, c] for a in range(-r, r + 1) for b in range(-r, r + 1) for c in range(-r, r + 1)],
                            np.int64)
            base = _uniq(base)
            bk = _uniq((base[:, None, :] + offs[None]).reshape(-1, 3))
            bk = surface_bricks_of(scene, o, bk, band)
        bi, nd, foot, N = band_nodes(scene, o, bk, band)
        if not len(bi):
            return {}
        x, y, z = project_many(cam, foot, self.w, self.h)
        k = (z > 0) & inside(poly, x, y)
        if not self.through:
            f, _r, _u = cam.basis()
            to_cam = (-f[None]) if cam.ortho else (cam.pos[None] - foot)
            k &= (N * to_cam).sum(1) > 0
            k &= self._unoccluded(cam, foot, N, x, y, hh * s)
        self.report_pick = {'band_nodes': int(len(bi)), 'picked': int(k.sum())}
        return _to_bricks(bk, bi[k], nd[k])

    # the view's own surface: rays through every pixel inside the outline (its depth image is the occlusion test)
    def _visible_hits(self, scene, cam, poly, o):
        from ..render.raymarch import trace_rays
        x0, y0 = np.floor(poly.min(0)).astype(int)
        x1, y1 = np.ceil(poly.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.w - 1), min(y1, self.h - 1)
        if x1 < x0 or y1 < y0:
            return None
        X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        X, Y = X.ravel(), Y.ravel()
        k = inside(poly, X.astype(float), Y.astype(float))
        X, Y = X[k], Y[k]
        if not len(X):
            return None
        Og, Dg = rays_at(cam, X, Y, self.w, self.h)
        t, oi, _st = trace_rays(scene, Og, Dg)
        pk = scene.pack()
        ids = np.asarray(pk.ids)
        hit = np.isfinite(t) & (oi >= 0)
        self._depth = {'x0': x0, 'y0': y0, 'img': np.full((y1 - y0 + 1, x1 - x0 + 1), np.inf)}
        self._depth['img'][Y[hit] - y0, X[hit] - x0] = t[hit]
        mine = hit.copy()
        mine[hit] = ids[oi[hit]] == o.id
        if not mine.any():
            return None
        Hw = Og[mine] + Dg[mine] * t[mine][:, None]
        pixang = 2.0 * np.tan(np.radians(cam.fov) * 0.5) / self.h
        foot_px = float(np.max(t[mine])) * pixang / _rs(o)[2] if not cam.ortho else \
            2.0 * cam.ortho_params()[0] * np.tan(np.radians(cam.fov) * 0.5) / self.h / _rs(o)[2]
        return Hw, foot_px

    def _unoccluded(self, cam, foot, N, x, y, hs):
        d = self._depth
        img = d['img']
        xi = np.clip(np.rint(x).astype(np.int64) - d['x0'], 0, img.shape[1] - 1)
        yi = np.clip(np.rint(y).astype(np.int64) - d['y0'], 0, img.shape[0] - 1)
        Og, Dg = rays_at(cam, np.rint(x), np.rint(y), self.w, self.h)
        tq = ((foot - Og) * Dg).sum(1)
        pixang = 2.0 * np.tan(np.radians(cam.fov) * 0.5) / self.h
        fp = (tq * pixang) if not cam.ortho else np.full(len(tq), 2.0 * cam.ortho_params()[0] *
                                                         np.tan(np.radians(cam.fov) * 0.5) / self.h)
        cosv = np.maximum(np.abs((N * Dg).sum(1)), 0.2)
        tol = 3.0 * hs + 2.0 * fp / cosv
        return tq <= img[yi, xi] + tol


def surface_bricks_of(scene, o, Bk, band):
    """of the given bricks, those that can hold surface-band nodes"""
    if not len(Bk):
        return Bk
    bs = B * o.sculpt.h
    pk = scene.pack()
    s = _rs(o)[2]
    keep = []
    for a in range(0, len(Bk), 2_000_000):
        c = (Bk[a:a + 2_000_000] + 0.5) * bs
        F = field_of(scene, pk, o, _to_world(o, c)) / s
        keep.append(np.abs(F) <= 1.5 * (0.87 * bs + band))
    return Bk[np.concatenate(keep)]


def _to_bricks(bk, bi, nd):
    out = {}
    for j in np.unique(bi).tolist():
        m = np.zeros((B, B, B), bool)
        q = nd[bi == j]
        m[q[:, 0], q[:, 1], q[:, 2]] = True
        out[tuple(int(v) for v in bk[j])] = m
    return out


# ---- screen helpers ---------------------------------------------------------------------------------------------------
def project_many(cam, P, w, h):
    """Camera.project for many world points -> x, y, depth (depth <= 0 behind the camera / the ortho start plane)"""
    f, r, u = cam.basis()
    v = np.asarray(P, float) - cam.pos[None]
    z = v @ f
    th = np.tan(np.radians(cam.fov) * 0.5)
    a = w / float(h)
    if cam.ortho:
        D, Bo = cam.ortho_params()
        x = ((v @ r) / (D * th * a) + 1.0) * 0.5 * w - 0.5
        y = (1.0 - (v @ u) / (D * th)) * 0.5 * h - 0.5
        return x, y, np.where(z > -Bo, D, -1.0)
    zz = np.where(np.abs(z) < 1e-12, 1e-12, z)
    x = ((v @ r) / (zz * th * a) + 1.0) * 0.5 * w - 0.5
    y = (1.0 - (v @ u) / (zz * th)) * 0.5 * h - 0.5
    return x, y, z


def rays_at(cam, X, Y, w, h):
    """Camera.ray for many pixels -> origins (n, 3), unit directions (n, 3)"""
    f, r, u = cam.basis()
    th = np.tan(np.radians(cam.fov) * 0.5)
    a = w / float(h)
    px = (2.0 * (np.asarray(X, float) + 0.5) / w - 1.0) * th * a
    py = (1.0 - 2.0 * (np.asarray(Y, float) + 0.5) / h) * th
    if cam.ortho:
        D, Bo = cam.ortho_params()
        O = cam.pos[None] + (r[None] * px[:, None] + u[None] * py[:, None]) * D - f[None] * Bo
        return np.ascontiguousarray(O), np.ascontiguousarray(np.repeat(f[None], len(O), 0))
    d = f[None] + r[None] * px[:, None] + u[None] * py[:, None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.ascontiguousarray(np.repeat(cam.pos[None], len(d), 0)), np.ascontiguousarray(d)


def _px_size(cam, P, w, h, size):
    """a world size at points P in pixels"""
    if cam.ortho:
        D, _ = cam.ortho_params()
        return np.full(len(P), size / (2.0 * D * np.tan(np.radians(cam.fov) * 0.5)) * h)
    z = np.maximum((np.asarray(P, float) - cam.pos[None]) @ cam.basis()[0], 1e-9)
    return size / (2.0 * z * np.tan(np.radians(cam.fov) * 0.5)) * h


def inside(poly, x, y):
    """even-odd point in polygon, vectorised"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    res = np.zeros(x.shape, bool)
    px, py = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = px[i], py[i], px[j], py[j]
        c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) if yj != yi else 1e-12) + xi)
        res ^= c
        j = i
    return res


@command
class RegionProps(Command):
    """rename / recolour a region (the table only; one undo step)"""
    label = 'Region'

    def __init__(self, oid, rid, name=None, colour=None):
        self.oid = int(oid)
        self.rid = int(rid)
        self.name = name
        self.colour = None if colour is None else [float(c) for c in colour]

    def args(self):
        return {'oid': self.oid, 'rid': self.rid, 'name': self.name, 'colour': self.colour}

    def do(self, scene):
        o = scene.get(self.oid)
        if self.rid not in o.regions:
            raise ValueError('Region: object %d has no region %d' % (self.oid, self.rid))
        self._old = copy.deepcopy(o.regions[self.rid])
        if self.name is not None:
            o.regions[self.rid]['name'] = str(self.name)
        if self.colour is not None:
            o.regions[self.rid]['colour'] = list(self.colour)
        scene.touch(structural=True)
        return [self.oid]

    def undo(self, scene):
        scene.get(self.oid).regions[self.rid] = copy.deepcopy(self._old)
        scene.touch(structural=True)
        return [self.oid]


# ---- the view state: hide, isolate, box hide (not in the hash or the log; render/raymarch.view_clips reads it) --------
def _vh(scene):
    if not isinstance(getattr(scene, 'view_regions_hidden', None), dict):
        scene.view_regions_hidden = {}
    return scene.view_regions_hidden


def hide_region(scene, oid, rid):
    _vh(scene).setdefault(int(oid), set()).add(int(rid))
    scene.touch(structural=True)


def isolate_region(scene, oid, rid):
    """only this region of the object shown (its other regions and its unlabelled surface hidden); a second isolate of
    the same region shows the object whole again"""
    o = scene.get(oid)
    others = ({0} | {int(k) for k in counts(scene, oid)} | {int(k) for k in (o.regions or {})}) - {int(rid)}
    vh = _vh(scene)
    if vh.get(int(oid)) == others:
        vh.pop(int(oid), None)
    else:
        vh[int(oid)] = others
    scene.touch(structural=True)


def show_region(scene, oid, rid):
    s = _vh(scene).get(int(oid))
    if s:
        s.discard(int(rid))
        if not s:
            _vh(scene).pop(int(oid), None)
    scene.touch(structural=True)


def show_all(scene):
    """Alt+H: every region, box and object shown again"""
    scene.view_regions_hidden = {}
    scene.view_boxes = []
    scene.view_hidden.clear()
    scene.touch(structural=True)


def box_hide(scene, cam, x0, y0, x1, y1, w, h):
    """Box hide: everything the screen rectangle covers (the view's frustum through it, every depth) hides"""
    planes = screen_box_planes(cam, x0, y0, x1, y1, w, h)
    if not isinstance(getattr(scene, 'view_boxes', None), list):
        scene.view_boxes = []
    scene.view_boxes.append({'planes': planes})
    scene.touch(structural=True)
    return planes


def screen_box_planes(cam, x0, y0, x1, y1, w, h):
    """the four planes of the view's frustum through a screen rectangle (inside where n.p <= d)"""
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    X = np.array([x0, x1, x1, x0]) - 0.5
    Y = np.array([y0, y0, y1, y1]) - 0.5
    O, Dr = rays_at(cam, X, Y, w, h)
    Oc, Dc = rays_at(cam, np.array([0.5 * (x0 + x1) - 0.5]), np.array([0.5 * (y0 + y1) - 0.5]), w, h)
    pc = Oc[0] + Dc[0] * 1.0
    out = []
    for a in range(4):
        b = (a + 1) % 4
        n = np.cross(Dr[a], Dr[b]) if not cam.ortho else np.cross(O[b] - O[a], Dr[a])
        ln = np.linalg.norm(n)
        if ln < 1e-15:
            continue
        n = n / ln
        d = float(n @ O[a])
        if n @ pc - d > 0:
            n, d = -n, -d
        out.append([float(n[0]), float(n[1]), float(n[2]), d])
    return out


@command
class RegionDelete(_RegionWrite):
    """a region gone: its nodes back to no region, its row out of the table"""
    label = 'Delete region'

    def __init__(self, oid, rid):
        self._init(oid, 'erase', rid, None, None, None, MAX_NEW_BRICKS)

    def args(self):
        return {'oid': self.oid, 'rid': self.rid}

    def pick(self, scene, o):
        st = scene.store
        out = {}
        for b, s in o.sculpt.slots.items():
            k = st.rid[s] == self.rid
            if k.any():
                out[b] = k
        return out

    def do(self, scene):
        r = _RegionWrite.do(self, scene)
        scene.get(self.oid).regions.pop(self.rid, None)
        return r


def region_extents(scene, oid, rid, axes=None):
    """design D1 ("the selected object's or region's width x height x depth"): the world extent of a region's surface:
    its labelled nodes carried onto the surface (x - F n) -> (lo, hi) or None"""
    o = scene.get(oid)
    st = scene.store
    h = o.sculpt.h
    X = []
    for b, s in o.sculpt.slots.items():
        k = np.argwhere(st.rid[s] == int(rid))
        if len(k):
            X.append((np.asarray(b) * B + k) * h)
    if not X:
        return None
    W = _to_world(o, np.vstack(X).astype(np.float64))
    pk = scene.pack()
    F = field_of(scene, pk, o, W)
    N = _grad(scene, pk, o, W, 0.25 * h * _rs(o)[2])
    foot = W - F[:, None] * N
    if axes is not None:                         # along the object's own axes (or any three): coordinates on them
        foot = foot @ np.asarray(axes, float)
    return foot.min(0), foot.max(0)
