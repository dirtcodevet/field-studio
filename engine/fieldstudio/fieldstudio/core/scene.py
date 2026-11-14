"""Scene, objects and their layers.

Object = {id, name, layers: base (Node tree), sculpt (sparse delta-F brick grid), detail (placeholder, off),
colour (albedo params), material}, transform (translate, rotate deg XYZ, uniform scale).

F_object(p_world) = s * (F_base(q) + dF_sculpt(q)), q = R^T (p - t) / s
F_scene = min over visible objects. Everything is evaluated by the numba program (program.py) through a pack of
flat arrays, rebuilt lazily when something structural changes; brush writes go into the shared pools in place.
"""
import copy
import hashlib
import json

import numpy as np

from . import ops
from . import program as P

DEFAULT_H = 0.0005       # sculpt lattice spacing, 0.5 mm
B = P.BRICK


# fs-brush 11-12 (after p74, within a 16 GB RAM budget): the store's side arrays, per lattice node. Each is
# LAZY until an edit writes something non-zero into it: zeros, held as one brick seen at every slot (no memory; on her
# body strokes these are 69 % of the store's 6.4 KB a brick and stay zero). Their shapes and types:
_SIDE = {'mask': ((B, B, B), np.float32),          # the mask
         'mdir': ((B, B, B, 3), np.int8),          # the mask's paint direction x 127 (0 = untagged)
         'rid': ((B, B, B), np.uint16)}            # fs-shell 11-11 (design 4.8, regions): region id (0 = none)


class _LazySide(np.ndarray):
    """fs-brush 11-12: a side array no edit has written yet (see _SIDE): every slot reads zeros from one shared brick.
    A Python write of anything non-zero makes the store's real array first (SculptStore.ensure) and goes there; a write of
    zeros changes nothing and is dropped. What indexing returns is a plain READ-ONLY array, so a write through it (an
    in-place op on st.mask[s], np.copyto into it) fails loudly instead of reaching the shared brick. Numba kernels get
    SculptStore.kview() instead (writable; a kernel that may write non-zero calls ensure() first: pack() checks)."""

    def __getitem__(self, k):
        r = np.ndarray.__getitem__(self, k)
        if not isinstance(r, np.ndarray):
            return r
        r = r.view(np.ndarray)
        if np.may_share_memory(r, self._zero):
            r.flags.writeable = False
        return r

    def __setitem__(self, k, v):
        if not np.any(v):                         # zeros onto zeros
            return
        st = self._store() if self._store is not None else None
        if st is None:
            raise ValueError('a write to a lazy side array with no store')
        st.ensure(self._name)[k] = v

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        """whole-array maths reads the zeros as a plain array (results are plain arrays); out= into it fails loudly"""
        if any(isinstance(o, _LazySide) for o in kw.get('out', ()) or ()):
            raise ValueError('out= into a lazy side array of the sculpt store: call SculptStore.ensure() first')
        inputs = tuple(i.view(np.ndarray) if isinstance(i, _LazySide) else i for i in inputs)
        return getattr(ufunc, method)(*inputs, **kw)

    @property
    def nbytes(self):
        """the memory it holds: none (one shared brick); memory reports read SculptStore.nbytes()"""
        return 0

    def __array_finalize__(self, obj):
        self._store = getattr(obj, '_store', None)
        self._name = getattr(obj, '_name', None)
        self._zero = getattr(obj, '_zero', None)


class SculptStore:
    """Scene-wide brick pools shared by every object's sculpt grid (delta-F and mask), with per-brick L.

    fs-brush 11-12 (p74: five body strokes on her took scene.store 1.6 -> 3,332 MB, the app 6.5 -> 11.6 GB with a 17.8 GB
    transient; a 16 GB RAM budget): the pools grow WITHOUT copying (core/vmem.py: address space reserved once, pages
    committed as slots are needed, so a growth never holds the old and new arrays together and the commit follows the
    slots used, not a doubled capacity), and the side arrays (mask, mdir, rid: _SIDE) cost nothing until an edit writes
    something non-zero into them. The arrays are properties: st.delta / st.Lb are real; st.mask / st.mdir / st.rid are
    real once written, else a _LazySide (reads zeros; Python writes go through). Numba kernels take st.kview(name);
    st.has(name) says whether a side array holds anything; st.ensure(name) makes it real. Every growth or ensure()
    bumps st.generation (the pack takes the new views)."""

    def __init__(self, cap=256):
        from . import vmem
        self._vmem = vmem
        self.cap = int(cap)
        self._pool = {'delta': vmem.Pool((B, B, B), np.float32, self.cap), 'Lb': vmem.Pool((), np.float64, self.cap)}
        self._zero = {n: np.zeros(shp, dt) for n, (shp, dt) in _SIDE.items()}   # the one zero brick of each side array
        self._v = {}
        self._kv = {}
        self.free = []
        self.used = 0
        self.generation = 0      # bumps when the pools grow or a side array becomes real

    # ---- the arrays ----------------------------------------------------------------------------------------------
    def _lazy(self, name, writable):
        z = self._zero[name]
        v = np.lib.stride_tricks.as_strided(z, shape=(self.cap,) + z.shape, strides=(0,) + z.strides,
                                            writeable=writable)
        if writable:
            return v
        import weakref
        v = v.view(_LazySide)
        v._store, v._name, v._zero = weakref.ref(self), name, z
        return v

    def _get(self, name):
        v = self._v.get(name)
        if v is None:
            p = self._pool.get(name)
            v = p.array if p is not None else self._lazy(name, False)
            self._v[name] = v
        return v

    delta = property(lambda self: self._get('delta'))
    Lb = property(lambda self: self._get('Lb'))
    mask = property(lambda self: self._get('mask'))
    mdir = property(lambda self: self._get('mdir'))
    rid = property(lambda self: self._get('rid'))

    def kview(self, name):
        """the array to hand a numba kernel: the real one, or (lazy) a writable zero view (one brick seen at every slot;
        a kernel that may write non-zero values into it calls ensure(name) first)"""
        if name in self._pool:
            return self._get(name)
        v = self._kv.get(name)
        if v is None:
            v = self._kv[name] = self._lazy(name, True)
        return v

    def has(self, name):
        """False while a side array has never been written (all zeros, no memory)"""
        return name in self._pool

    def ensure(self, name):
        """make a side array real (zeros, committed for the current capacity) and return it"""
        if name not in self._pool:
            shp, dt = _SIDE[name]
            self._pool[name] = self._vmem.Pool(shp, dt, self.cap)
            self._v.pop(name, None)
            self._kv.pop(name, None)
            self.generation += 1
        return self._get(name)

    def check_lazy(self):
        """the shared zero bricks are still zero (a kernel wrote into a lazy side array without ensure(): a fault)"""
        for n, z in self._zero.items():
            if z.any():
                z[...] = 0
                raise RuntimeError('a write reached the lazy %s array of the sculpt store without ensure() (fs-brush '
                                   '11-12); the write was lost' % n)

    def _grow(self, need):
        """capacity for at least `need` slots. Without copying (core/vmem.py) a step costs a commit and new views, so the
        steps are small: +1/16 (at least 8192 slots), the committed slack under ~6 %; the numpy fallback copies, so it
        doubles as before"""
        if need <= self.cap:
            return
        step = self.cap // 16 if self._pool['delta'].virtual else self.cap
        cap = max(int(need), self.cap + max(step, 8192))
        for p in self._pool.values():
            p.grow(cap)
        self.cap = cap
        self._v.clear()
        self._kv.clear()
        self.generation += 1

    def alloc(self):
        if self.free:
            return self.free.pop()
        if self.used >= self.cap:
            self._grow(self.used + 1)
        s = self.used
        self.used += 1
        return s

    def alloc_many(self, n):
        """READY 2 (merge-all): n slots at once, the same slots in the same order as n alloc() calls"""
        out = []
        while self.free and len(out) < n:
            out.append(self.free.pop())
        k = n - len(out)
        if k:
            self._grow(self.used + k)
            out.extend(range(self.used, self.used + k))
            self.used += k
        return np.array(out, np.int64)

    def release_many(self, slots):
        """release() of each slot, in order"""
        slots = np.asarray(slots, np.int64)
        if not len(slots):
            return
        for n in self._pool:                     # a lazy side array is zero already
            self._get(n)[slots] = 0
        self.free.extend(slots.tolist())

    def release(self, slot):
        for n in self._pool:
            self._get(n)[slot] = 0
        self.free.append(slot)

    def nbytes(self):
        """host memory the pools hold (committed)"""
        return sum(p.nbytes() for p in self._pool.values())


class SculptGrid:
    """One object's sculpt layer: a sparse lattice of delta-F (and mask) on a fixed lattice in the object's
    local frame, nodes at integer multiples of h; bricks of 8^3 nodes, a dense brick index that grows on demand."""

    def __init__(self, h=DEFAULT_H):
        self.h = float(h)
        self.origin = np.zeros(3, np.int64)
        self.index = np.full((0, 0, 0), -1, np.int32)
        self.slots = {}                  # (bx, by, bz) -> slot
        self.bmin = None                 # allocated brick range (inclusive), for the object box
        self.bmax = None

    def ensure_index(self, blo, bhi):
        """Grow the index to cover bricks blo..bhi (inclusive). Returns True when it reallocated."""
        blo = np.asarray(blo, np.int64)
        bhi = np.asarray(bhi, np.int64)
        if self.index.size:
            cur_lo = self.origin
            cur_hi = self.origin + np.array(self.index.shape) - 1
            if np.all(blo >= cur_lo) and np.all(bhi <= cur_hi):
                return False
            pad = 4
            nlo = np.minimum(blo - pad, cur_lo)
            nhi = np.maximum(bhi + pad, cur_hi)
        else:
            pad = 4
            nlo, nhi = blo - pad, bhi + pad
        shape = tuple(int(v) for v in (nhi - nlo + 1))
        new = np.full(shape, -1, np.int32)
        if self.index.size:
            o = self.origin - nlo
            sx, sy, sz = self.index.shape
            new[o[0]:o[0] + sx, o[1]:o[1] + sy, o[2]:o[2] + sz] = self.index
        self.index = new
        self.origin = nlo
        return True

    def get_slot(self, b):
        return self.slots.get(b, -1)

    def set_slot(self, b, slot):
        i = np.asarray(b, np.int64) - self.origin
        self.index[i[0], i[1], i[2]] = slot
        if slot >= 0:
            self.slots[b] = slot
            bb = np.asarray(b, np.int64)
            self.bmin = bb.copy() if self.bmin is None else np.minimum(self.bmin, bb)
            self.bmax = bb.copy() if self.bmax is None else np.maximum(self.bmax, bb)
        else:
            if self.slots.pop(b, None) is not None:
                self._box_dirty = True           # 11-09: the range follows the bricks held, not a high-water mark

    def set_slots(self, keys, slots):
        """r28: set_slot for many bricks at once (same end state as one set_slot per key, in order)."""
        keys = np.asarray(keys, np.int64).reshape(-1, 3)
        slots = np.asarray(slots, np.int64).reshape(-1)
        if not len(keys):
            return
        self.ensure_index(keys.min(0), keys.max(0))
        i = keys - self.origin
        self.index[i[:, 0], i[:, 1], i[:, 2]] = slots
        tk = list(map(tuple, keys.tolist()))
        pos = slots >= 0
        for k, s in zip(tk, slots.tolist()):
            if s >= 0:
                self.slots[k] = s
            elif self.slots.pop(k, None) is not None:
                self._box_dirty = True
        if pos.any():
            kp = keys[pos]
            self.bmin = kp.min(0) if self.bmin is None else np.minimum(self.bmin, kp.min(0))
            self.bmax = kp.max(0) if self.bmax is None else np.maximum(self.bmax, kp.max(0))

    def region_box(self):
        """Local-space box of all allocated bricks (incl. the +1 node interpolation apron).
        11-09 (the script API's replay check): bmin / bmax grew on every allocation and never shrank when a brick was
        freed (a reverted re-distance, the GPU write-back's paths), so a live scene's box stayed larger than the same
        state loaded or replayed; the colliders' box culls then gave Settle a different result. A free marks the range
        stale and it is recomputed here from the bricks held."""
        if getattr(self, '_box_dirty', False):
            self._box_dirty = False
            if self.slots:
                ks = np.array(list(self.slots.keys()), np.int64)
                self.bmin, self.bmax = ks.min(0), ks.max(0)
            else:
                self.bmin = self.bmax = None
        if self.bmin is None:
            return None
        lo = self.bmin * B * self.h
        hi = (self.bmax + 1) * B * self.h
        return lo.astype(float), hi.astype(float)

    def lipschitz(self, store):
        if not self.slots:
            return 0.0
        return float(store.Lb[list(self.slots.values())].max())


class SceneObject:
    def __init__(self, oid, name, base, transform=None, colour=None, material=None, h=DEFAULT_H):
        self.id = int(oid)
        self.name = name
        self.base = base                         # ops.Node
        self.sculpt = SculptGrid(h)
        self.detail = {'enabled': False, 'amplitude': 0.0, 'scale': 0.002, 'seed': 0}
        self.colour = colour or {'albedo': [0.72, 0.62, 0.55], 'variation': 0.0}
        self.material = material or {'roughness': 0.5, 'specular': 0.3}
        self.transform = transform or {'translate': [0.0, 0.0, 0.0], 'rotate': [0.0, 0.0, 0.0], 'scale': 1.0}
        self.visible = True
        self.mask_inv = False                    # hand2f: Mask > Invert (a flag, so unsculpted areas invert too)
        self.locked = None                       # 11-08 night: protected (locked) part; None = by name (see is_protected)
        self.root = None                         # 11-09: an attached part's root {'file', 'sha'} (core/attach.py)
        self.part = None                         # fs-contact 11-11: a store part {source, pid, name, pairs} (core/fv8parts.py)
        self.regions = {}                        # fs-shell 11-11 (design 4.8): region id -> {'name', 'colour'}
        self.garment = None                      # fs-shell 11-11 (for fs-cloth): {garment, panel, fabric, fit, run}
        self.layers = []                         # fs-shell 11-11 (design 4.10): [{id, name, strength, on, sha, file, ...}]
        self.snapshots = []                      # fs-shell 11-11 (design 4.10): [{name, sha, file, bricks}]
        self.group = None                        # fs-tools 11-14: a group id (Group / Ungroup); None = ungrouped
        self._compiled = None                    # (code, prm, lo, hi, L), cleared on base edits

    def compiled(self):
        if self._compiled is None:
            self._compiled = ops.compile_tree(self.base)
        return self._compiled

    def invalidate(self):
        self._compiled = None

    def describe(self):
        """Canonical description (no sculpt data) for save / hash."""
        d = {'id': self.id, 'name': self.name, 'base': self.base.to_dict(), 'detail': self.detail,
             'colour': self.colour, 'material': self.material, 'transform': self.transform,
             'visible': self.visible, 'sculpt_h': self.sculpt.h}
        if self.mask_inv:                        # hand2f: only when set, so older hashes and projects are unchanged
            d['mask_invert'] = True
        if getattr(self, 'regions', None):       # fs-shell 11-11: only when it has regions (hashes unchanged)
            d['regions'] = {str(k): dict(v) for k, v in sorted(self.regions.items())}
        if getattr(self, 'garment', None):       # fs-shell 11-11: only when set (hashes unchanged)
            d['garment'] = copy.deepcopy(self.garment)
        if getattr(self, 'layers', None):        # fs-shell 11-11: sculpt layers (their content by sha), only when any
            d['layers'] = copy.deepcopy(self.layers)
        if getattr(self, 'snapshots', None):     # fs-shell 11-11: snapshots (by sha), only when any
            d['snapshots'] = copy.deepcopy(self.snapshots)
        if self.locked is not None:              # only when set: older hashes and projects unchanged
            d['locked'] = bool(self.locked)
        if getattr(self, 'group', None) is not None:   # fs-tools 11-14: only when grouped (hashes unchanged)
            d['group'] = int(self.group)
        if self.root:
            d['root'] = dict(self.root)
        if self.part:                            # fs-contact 11-11: only when set (older hashes and projects unchanged)
            d['part'] = copy.deepcopy(self.part)
        return d

    def local_box(self):
        code, prm, lo, hi, L = self.compiled()
        rb = self.sculpt.region_box()
        if rb is not None:
            lo, hi = np.minimum(lo, rb[0]), np.maximum(hi, rb[1])
        return lo, hi


def is_protected(o):
    """a protected part: locked (not sculpted or picked by brushes, kept apart by merges). Unprotect sets locked = False
    (it becomes an ordinary editable object until Protect relocks it); without the flag, the '(protected)' name"""
    lk = getattr(o, 'locked', None)
    return bool(lk) if lk is not None else '(protected)' in o.name


class Pack:
    """Flat arrays the numba kernels take (see program.py for the table layouts)."""
    __slots__ = ('code', 'prm', 'oi_tab', 'of_tab', 'index_flat', 'delta', 'mask', 'mdir', 'ids', 'lo', 'hi', 'lbase')

    def args(self):
        return self.code, self.prm, self.oi_tab, self.of_tab, self.index_flat, self.delta


class Scene:
    def __init__(self):
        self.objects = []
        self.store = SculptStore()
        self.next_id = 1
        self.version = 0
        self._pack = None
        self._index_dirty = True
        self._store_gen = -1
        self.view_hidden = set()     # hand2f: Hide / Isolate are VIEW state: not in describe(), the hash or the op log

    def shown(self, o):
        """hand2f: drawn and picked: the object's own 'visible' (a logged property) and not hidden in the view"""
        return o.visible and o.id not in self.view_hidden

    # ---- ids and lookup --------------------------------------------------------------------------------
    def new_id(self):
        i = self.next_id
        self.next_id += 1
        return i

    def get(self, oid):
        for o in self.objects:
            if o.id == oid:
                return o
        raise KeyError(oid)

    def index_of(self, oid):
        for i, o in enumerate(self.objects):
            if o.id == oid:
                return i
        return -1

    def touch(self, structural=False, index=False):
        """Mark the pack stale. structural: base trees / transforms / object list changed."""
        self.version += 1
        if structural or index:
            self._pack = None
        if index:
            self._index_dirty = True

    # ---- packing -------------------------------------------------------------------------------------------
    def pack(self):
        st = self.store
        st.check_lazy()                          # fs-brush 11-12: no kernel wrote into a lazy side array
        if self._pack is not None and self._store_gen == st.generation and not self._index_dirty:
            return self._pack
        if self._pack is not None and not self._index_dirty:
            # fs-brush 11-12: only the store changed (its pools grew without copying, or a side array became real): the
            # same pack takes the new views; the index and the programs are as they were
            pk = self._pack
            pk.delta, pk.mask, pk.mdir = st.delta, st.kview('mask'), st.kview('mdir')
            self._store_gen = st.generation
            return pk
        codes, prms, oi_rows, of_rows = [], [], [], []
        c_off = 0
        p_off = 0
        if self._index_dirty or self._pack is None or self._store_gen != st.generation:
            sizes = [o.sculpt.index.size for o in self.objects]
            flat = np.full(max(sum(sizes), 1), -1, np.int32)
            off = 0
            for o, n in zip(self.objects, sizes):
                if n:
                    flat[off:off + n] = o.sculpt.index.ravel()
                    o.sculpt.index = flat[off:off + n].reshape(o.sculpt.index.shape)   # view: writes go live
                off += n
            self._index_dirty = False
        else:
            flat = self._pack.index_flat
        idx_off = 0
        wlo = np.full(3, np.inf)
        whi = np.full(3, -np.inf)
        for o in self.objects:
            code, prm, blo, bhi, Lb = o.compiled()
            code = code.copy()
            code[:, 1] += p_off
            for r in range(code.shape[0]):
                if code[r, 0] == P.OP_GUARD or code[r, 0] == P.OP_GUARD_U:
                    code[r, 2] += c_off
            codes.append(code)
            prms.append(prm)
            g = o.sculpt
            n = g.index.size
            nx, ny, nz = g.index.shape if n else (0, 0, 0)
            oi_rows.append([c_off, c_off + code.shape[0], idx_off, nx, ny, nz,
                            int(g.origin[0]), int(g.origin[1]), int(g.origin[2])])
            idx_off += n
            c_off += code.shape[0]
            p_off += prm.shape[0]
            row, w0, w1 = self._object_row(o)
            of_rows.append(row)
            if self.shown(o):
                wlo = np.minimum(wlo, w0)
                whi = np.maximum(whi, w1)
        pk = Pack()
        pk.code = np.ascontiguousarray(np.vstack(codes) if codes else np.zeros((0, 3), np.int32), dtype=np.int32)
        # milestone (11-08): one object's program is used as is (her store's is 2 GB; a copy cost that again)
        pk.prm = np.ascontiguousarray((prms[0] if len(prms) == 1 else np.concatenate(prms)) if prms else np.zeros(1),
                                      dtype=np.float64)
        pk.oi_tab = np.array(oi_rows, np.int32).reshape(-1, P.OI_N)
        pk.of_tab = np.array(of_rows, np.float64).reshape(-1, P.OF_N)
        pk.index_flat = flat
        pk.delta = st.delta
        pk.mask = st.kview('mask')               # fs-brush 11-12: kernels take kview (a zero view while lazy)
        pk.mdir = st.kview('mdir')
        pk.ids = [o.id for o in self.objects]
        # V1.3b: each object's base-tree L, for the tracer's per-brick bound L_base + Lb[brick] (raymarch.local_step)
        pk.lbase = np.array([float(o.compiled()[4]) for o in self.objects] or [1.0], np.float64)
        if not np.all(np.isfinite(wlo)):
            wlo, whi = np.zeros(3), np.zeros(3)
        pk.lo, pk.hi = wlo, whi
        self._pack = pk
        self._store_gen = st.generation
        return pk

    def _object_row(self, o):
        code, prm, blo, bhi, Lb = o.compiled()
        g = o.sculpt
        lo, hi = o.local_box()
        m = ops._margin(lo, hi)
        elo, ehi = lo - m, hi + m
        t = o.transform
        R = ops.rot_matrix(t['rotate'])
        s = max(float(t['scale']), 1e-6)
        L = Lb + g.lipschitz(self.store)
        row = np.zeros(P.OF_N)
        row[0:9] = R.T.ravel()
        row[9:12] = t['translate']
        row[12] = s
        row[13:16] = elo
        row[16:19] = ehi
        row[19] = m
        row[20] = max(L, 1.0)
        row[21] = g.h
        row[22:25] = o.colour['albedo']
        row[25] = 1.0 if self.shown(o) else 0.0
        row[26] = o.material.get('roughness', 0.5)
        row[27] = 1.0 if getattr(o, 'mask_inv', False) else 0.0
        c = ops._box_corners(elo, ehi)
        w = (R @ c.T).T * s + np.asarray(t['translate'], float)
        return row, w.min(0), w.max(0)

    def refresh_object(self, oi):
        """After a brush write: update one object's row (box, margin, L) in the live pack without a repack."""
        if self._pack is None:
            return
        row, w0, w1 = self._object_row(self.objects[oi])
        self._pack.of_tab[oi] = row
        if self.shown(self.objects[oi]):
            self._pack.lo = np.minimum(self._pack.lo, w0)
            self._pack.hi = np.maximum(self._pack.hi, w1)
        self.version += 1

    # ---- queries (contract) --------------------------------------------------------------------------------
    def evaluate(self, points):
        """points (N,3) world -> F (N,) float64 (min over visible objects; not an exact SDF after edits)."""
        pk = self.pack()
        pts = np.ascontiguousarray(np.asarray(points, np.float64).reshape(-1, 3))
        f = np.empty(len(pts))
        d = np.empty(len(pts))
        o = np.empty(len(pts), np.int64)
        P.eval_points(*pk.args(), pts, f, d, o)
        return f

    def evaluate_full(self, points):
        """-> (F, distance lower bound F_i/L_i, object index into self.objects or -1)."""
        pk = self.pack()
        pts = np.ascontiguousarray(np.asarray(points, np.float64).reshape(-1, 3))
        f = np.empty(len(pts))
        d = np.empty(len(pts))
        o = np.empty(len(pts), np.int64)
        P.eval_points(*pk.args(), pts, f, d, o)
        return f, d, o

    def gradient(self, points, eps=1e-5):
        """Central-difference gradient of F, (N,3)."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        g = np.empty_like(pts)
        for a in range(3):
            e = np.zeros(3)
            e[a] = eps
            g[:, a] = (self.evaluate(pts + e) - self.evaluate(pts - e)) / (2 * eps)
        return g

    def lipschitz_bound(self):
        """Max over visible objects of the tracked bound (world units; uniform scale keeps L)."""
        pk = self.pack()
        if not len(pk.of_tab):
            return 1.0
        vis = pk.of_tab[:, P.OF_VIS] > 0.5
        return float(pk.of_tab[vis, P.OF_L].max()) if vis.any() else 1.0

    def bounds(self):
        pk = self.pack()
        return pk.lo.copy(), pk.hi.copy()

    def field_memory_bytes(self):
        n = self.store.nbytes()
        for o in self.objects:
            n += o.sculpt.index.nbytes
        return n

    # ---- describe / hash -----------------------------------------------------------------------------------
    def describe(self):
        return {'next_id': self.next_id, 'objects': [o.describe() for o in self.objects]}

    def state_hash(self):
        """sha256 of the canonical description + every non-zero sculpt brick (by object id, brick coords)."""
        h = hashlib.sha256()
        h.update(json.dumps(self.describe(), sort_keys=True, separators=(',', ':')).encode())
        st = self.store
        for o in self.objects:
            for b in sorted(o.sculpt.slots):
                s = o.sculpt.slots[b]
                d = st.delta[s]
                m = st.mask[s]
                r = st.rid[s]
                if not d.any() and not m.any() and not r.any():
                    continue
                h.update(np.array([o.id, b[0], b[1], b[2]], np.int64).tobytes())
                h.update(np.ascontiguousarray(d).tobytes())
                h.update(np.ascontiguousarray(m).tobytes())
                if st.mdir[s].any():             # only when tagged: older hashes unchanged
                    h.update(np.ascontiguousarray(st.mdir[s]).tobytes())
                if r.any():                      # fs-shell 11-11: region ids, only when any (hashes unchanged)
                    h.update(b'rid' + np.ascontiguousarray(r).tobytes())
        return h.hexdigest()


def make_object(scene, kind='sphere', name=None, params=None, translate=(0, 0, 0), h=DEFAULT_H, base=None):
    """Create (not insert) an object with one primitive (or a given base tree)."""
    oid = scene.new_id()
    if base is None:
        base = ops.Node(kind, params or {}, nid=scene.new_id())
    return SceneObject(oid, name or ('%s.%d' % (base.kind, oid)), base,
                       transform={'translate': [float(v) for v in translate], 'rotate': [0.0, 0.0, 0.0],
                                  'scale': 1.0}, h=h)
