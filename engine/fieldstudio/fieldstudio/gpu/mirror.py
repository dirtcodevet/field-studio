"""Mirror a Field Studio scene onto the GPU atlas (phase 0, 2026-11-08).

The CPU core stays the one authority for every field (program.eval_local through core.bake.object_field_fn). The
mirror keeps one GPU volume per object, in the object's local frame at its sculpt lattice h: 8h bricks of 9^3 nodes
(the FV8 layout), the narrow band only, in tiles of 4^3 bricks whose grid is aligned to the sculpt bricks (both start
at multiples of 8h), so sculpt brick k feeds GPU bricks k and k-1 on each axis (the apron node).

What is sent is found by comparing, never guessed:
  - the base program (code + parameters + lattice h) per object: changed -> the volume is rebuilt;
  - each sculpt brick's checksum (a fixed random projection of its 512 values): the bricks that changed, appeared or
    left since the last sync are exactly the edited ones (strokes, mirrored and world-symmetry dabs, the release
    re-distance, undo, redo, merges, duplicates...), and only the GPU bricks they feed are re-sampled;
  - transform, visibility and colour: the volume's row only.
A change that reaches past the volume's grid (its box plus a growth margin) rebuilds the volume.
"""
import hashlib
import os
import math
import time

import numpy as np

from numba import njit, prange

from .scene import NLEV, Volume
from ..core import program as P


def ops_compile(node):
    from ..core import ops as _ops
    return _ops.compile_tree(node)


@njit(cache=True, nogil=True, parallel=True)
def _eval_nodes9(code, prm, oi_tab, of_tab, index_flat, delta, oi, origins, h, out):
    """r11: out (N, 729) = the object's local field at the 9^3 nodes of bricks with the given origins."""
    for n in prange(origins.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    # r21: the true field (guard threshold 1e9: no box / guard shortcut). eval_local returns a box-
                    # distance bound outside the object's box, and that box grows with the sculpt, so held bricks kept
                    # bounds from an older box (452 bricks off by up to 37 mm; brush rules on them made brick patches)
                    out[n, (i * 9 + j) * 9 + k] = P.eval_local_t(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                                                 origins[n, 0] + i * h, origins[n, 1] + j * h,
                                                                 origins[n, 2] + k * h, vs, ps, 1e9)


def _ukeys(a):
    """unique rows of an int (N, 3) array (r11: np.unique(axis=0) sorted void views, ~19 ms a call)"""
    a = np.asarray(a, np.int64)
    if len(a) == 0:
        return a.reshape(0, 3)
    lo = a.min(0)
    d = a - lo
    m = d.max(0) + 1
    code = (d[:, 0] * m[1] + d[:, 1]) * m[2] + d[:, 2]
    u = np.unique(code)
    return np.stack([u // (m[1] * m[2]), (u // m[2]) % m[1], u % m[2]], 1) + lo

ADMIT_CHUNK = 65536   # 11-09: candidate cells evaluated per chunk in _admit_cells (bounds the card memory of a rebuild)
LEVEL = 2          # the object's lattice goes in level slot 2 (its own h; slot meaning is per volume)
BPT = 4            # bricks per tile axis (tile = 32 h)
BOX_FAST = __import__('os').environ.get('FSB_BOX_FAST', '1') != '0'   # fs-brush 11-12: no full key scan for an exact box
SCRUB_SLOTS = int(__import__('os').environ.get('FSB_SCRUB_SLOTS', '16384'))   # fs-brush 11-12: idle check slice (32 MB)
BAND_H = 4.0       # band half-width in h (|F| < 4 h is stored)
GROW = 0.25        # grid margin: 25 % of the box on each side, at least 4 tiles
BLOCKOUT_GROW = float(os.environ.get('FSB_BLOCKOUT_GROW', '2.0'))   # fs-brush 11-12: small objects' margin (0: as before); fs-brush-3 11-13: 0.9 -> 2.0
BLOCKOUT_CAP = float(os.environ.get('FSB_BLOCKOUT_CAP', '0.6'))    # fs-brush-3 11-13: 0.15 -> 0.6 m (block-out S3 regrew the
# grid mid-drag, 740-918 ms in one frame; with 2x / 0.6 m: none, render cost unchanged, speed/f2_tk4_big vs _def). Only for
# objects whose box is at most BLOCKOUT_MAX a side, so her body (1.7 m: 25 %) and its store's cell map are unchanged
BLOCKOUT_MAX = float(os.environ.get('FSB_BLOCKOUT_MAX', '0.5'))
COARSE_FLUSH_KEYS = int(os.environ.get('FSB_COARSE_FLUSH_KEYS', '1500'))   # fs-brush-2 11-13: fine bricks per frame (0 = all)
RING = int(__import__("os").environ.get("FS_STORE_RING", "1"))
LEVEL_C = 3        # r34: the coarse level slot: nodes every C_STRIDE sculpt nodes, one brick per tile
C_STRIDE = 4
C_BAND_H = 16.0    # its band in sculpt h (4 coarse cells)
_W = np.random.default_rng(12345).standard_normal(512).astype(np.float32)
_W64 = _W.astype(np.float64)


@njit(parallel=True, cache=True)
def _row_ck(D, W, out):
    for i in prange(D.shape[0]):
        acc = 0.0
        for j in range(D.shape[1]):
            acc += np.float64(D[i, j]) * W[j]
        out[i] = acc


def _ck(D):
    """fs-brush 11-12: each slot's checksum (its 512 delta values against _W), the same bits whatever batch it is
    computed in: one row in a fixed order (float64). The matrix product it replaces gave 82 of 226 k rows different bits
    in 16 k-row slices than in one product (the BLAS blocks by shape), so the full sync took slots marked current at a
    release (a product over that stroke's slots) as changed and resampled them, and the idle scrub raised false alarms
    (syn1: 1,385)."""
    D = np.ascontiguousarray(D, np.float32).reshape(len(D), -1)
    out = np.empty(len(D), np.float64)
    if len(D):
        _row_ck(D, _W64, out)
    return out
_OFFS = np.array([[a, b, c] for a in (0, -1) for b in (0, -1) for c in (0, -1)], np.int64)


def _rot(of_row):
    return np.asarray(of_row[0:9], float).reshape(3, 3).T     # of_tab holds R^T row-major



RM_LOG_MAX = 4_000_000     # fs-brush 11-12: logged cells before the card copy is simply uploaded whole again


def _rm_note(ov, keys, rows):
    """fs-brush 11-12 (M p65 on her: each card batch uploaded the object's WHOLE cell -> row map, her box at 0.25 mm =
    83 M cells / 332 MB, after an astype copy: 0.4-0.85 s in the live re-distance and as much again in the apron
    refresh, whatever the batch's size): the cells changed since the card copy was made, so gpu/sculpt.py _rowmap_dev
    applies only them. None = not logging (no copy yet, or too many changes: the next use uploads the map whole)."""
    lg = getattr(ov, 'rm_log', None)
    if lg is None:
        return
    keys = np.asarray(keys, np.int64).reshape(-1, 3)
    lg.append((np.ravel_multi_index((keys[:, 0], keys[:, 1], keys[:, 2]), ov.rowmap.shape), np.asarray(rows, np.int64)))
    ov.rm_logn = getattr(ov, 'rm_logn', 0) + len(keys)
    if ov.rm_logn > RM_LOG_MAX:
        ov.rm_log = None
        ov.rm_logn = 0

class ObjVolume:
    def __init__(self, oid):
        self.oid = oid
        self.vol = None          # gpu.scene.Volume
        self.base_sig = None
        self.key_to_row = {}     # (i, j, k) GPU brick ijk in the volume grid -> pool row
        self.far_dirty = True
        self.ck = {}             # sculpt brick key -> checksum at the last sync
        self.boff = None         # GPU brick ijk of sculpt brick (0, 0, 0)


class SceneMirror:
    def __init__(self, gpu):
        self.gpu = gpu
        self.objs = {}           # oid -> ObjVolume
        self.stats = {}
        from .geval import GpuEval
        self.geval = GpuEval(gpu.cp)     # r22: node values evaluated on the card

    @staticmethod
    def _base_sig(o):
        code, prm, lo, hi, L = o.compiled()
        h = hashlib.sha1(np.ascontiguousarray(code).tobytes())
        if prm.size > 10_000_000:      # milestone: her store (2 GB): the tree names it by its source id, exactly
            import json as _json
            h.update(_json.dumps(o.base.to_dict(), sort_keys=True).encode())
        else:
            h.update(np.ascontiguousarray(prm).tobytes())
        h.update(np.float64(o.sculpt.h).tobytes())
        return h.hexdigest()

    @staticmethod
    def _checksums(scene, o):
        if not o.sculpt.slots:
            return {}
        keys = list(o.sculpt.slots.keys())
        slots = np.fromiter((o.sculpt.slots[k] for k in keys), np.int64, len(keys))
        c = _ck(scene.store.delta[slots].reshape(len(keys), 512))
        return dict(zip(keys, c.tolist()))

    @staticmethod
    def checksums_of(scene, o, keys):
        """r15: checksums of the given sculpt brick keys only (missing keys skipped)."""
        sl = o.sculpt.slots
        ks = [k for k in keys if k in sl]
        if not ks:
            return {}
        slots = np.fromiter((sl[k] for k in ks), np.int64, len(ks))
        c = _ck(scene.store.delta[slots].reshape(len(ks), 512))
        return dict(zip(ks, c.tolist()))

    # ---- sync ------------------------------------------------------------------------------------------------
    def sync(self, scene, known_current=False):
        """Bring the GPU atlas up to the scene. -> {'rebuilt': [oid], 'bricks': n resampled, 'dirty_world': [(lo, hi)],
        'ms': ...}. dirty_world: world boxes of the incremental changes (for the frame's dirty rectangles).
        r17: one checksum per store slot for the whole store (one matrix product), compared per slot with the last
        sync; the changed slots are mapped to their objects' brick keys (now and at the last sync, so a slot that
        moved or was freed counts for both). The base signature is recomputed only when an object's compiled
        program object changed.
        fs-brush 11-12 (brush-study STUDY 1e: this sync ran twice per stroke, ~280 ms a call at 226 k slots, most of it
        the whole-store checksum and the owner dict): known_current=True is the release of a card stroke whose every
        written brick was marked current (mark_current) and before which nothing else on the CPU changed since the sync
        its first batch saw (gpu/sculpt.py checks the scene key): the change search is skipped; the rest (rows, boxes,
        far fields, transforms) runs as always. A full sync still runs at idle (scrub_step) to catch any writer that
        changed the store without a touch."""
        t0 = time.perf_counter()
        pk = scene.pack()
        st = scene.store
        n = st.used
        changed = {}
        if known_current and self._S is not None and len(self._S) == n and not getattr(self, 'scrub_dirty', False):
            self.light_syncs = getattr(self, 'light_syncs', 0) + 1
            self.scrub_pos = 0                       # the idle check walks the store again (gpu/viewport.py scrub)
        else:
            S = _ck(st.delta[:n].reshape(n, 512)) if n else np.zeros(0, np.float64)
            owner = {}
            for o in scene.objects:
                for k, sl in o.sculpt.slots.items():
                    owner[sl] = (o.id, k)
            prevS = self._S if self._S is not None else np.zeros(0, np.float64)
            m = min(len(prevS), n)
            ch = np.nonzero(S[:m] != prevS[:m])[0].tolist() + list(range(m, n))
            for sl in ch:
                for ow in (owner.get(sl), self._owner.get(sl)):
                    if ow is not None:
                        changed.setdefault(ow[0], set()).add(ow[1])
            for sl, ow in self._owner.items():          # a slot whose owner changed (moved or freed) without a value change
                if owner.get(sl) != ow:
                    changed.setdefault(ow[0], set()).add(ow[1])
                    if owner.get(sl) is not None:
                        changed.setdefault(owner[sl][0], set()).add(owner[sl][1])
            self._S, self._owner = S, owner
            self.full_sync_t = time.perf_counter()
            self.scrub_dirty = False
            self.scrub_pos = 0                       # a full check just ran: the idle scrub starts over
        rebuilt, nb, dirty = [], 0, []
        seen = set()
        for oi, o in enumerate(scene.objects):
            seen.add(o.id)
            ov = self.objs.get(o.id)
            if ov is None:
                ov = self.objs[o.id] = ObjVolume(o.id)
            comp = o.compiled()
            if getattr(ov, '_comp', None) is not comp:
                ov._comp = comp
                ov._sig = self._base_sig(o)
            sig = ov._sig
            if ov.vol is None or sig != ov.base_sig:
                nb += self._rebuild(scene, oi, ov)
                ov.base_sig = sig
                rebuilt.append(o.id)
            elif changed.get(o.id):
                ok, nn, box = self._update_keys(scene, oi, ov, sorted(changed[o.id]))
                if not ok:
                    nb += self._rebuild(scene, oi, ov)
                    rebuilt.append(o.id)
                else:
                    nb += nn
                    if box is not None:
                        dirty.append(self._world_box(pk.of_tab[oi], box))
            self._set_row(ov, pk, oi, scene)
        gone = [oid for oid in list(self.objs) if oid not in seen]
        for oid in gone:
            self._drop(self.objs.pop(oid))
            self.geval.forget(oid)                       # fs-mouth 11-12: its device program goes with it
        if gone:                                         # a structural change (never a stroke): the pool's blocks back
            from .scene import release_cached
            release_cached('mirror_drop')
        for ov in self.objs.values():                # r10/r11: far fields and tight boxes of changed brick sets
            if ov.vol is None or not ov.far_dirty:
                continue
            new = getattr(ov, 'new_keys', None)
            is_store = getattr(ov, 'is_store', False)
            far_full = (not is_store) and (getattr(ov.vol, 'far', None) is None or new is None)
            # fs-brush 11-12 (brush-study A2; her 0.42-1.1 M keys made an array here after every admitting stroke, 0.16-0.46 s,
            # only to recompute the tracer's box, which _alloc_bricks already grew by the new keys): with no brick freed
            # since the box was last computed from every key, that box is exact, so the scan is skipped
            box_fast = BOX_FAST and new is not None and not getattr(ov, 'freed_since_box', False) and not far_full
            keys = None if box_fast else np.array(list(ov.key_to_row.keys()), np.int64).reshape(-1, 3)
            if is_store:
                pass                                   # milestone: her store volume bounds the empty space; no far field
            elif far_full:
                self.gpu.far_field(ov.vol, LEVEL, keys)
            elif len(new):
                self._far_add(ov.vol, np.concatenate(new))
            ov.new_keys = []
            if keys is not None:
                self._tight_box(ov.vol, keys)
                ov.freed_since_box = False
            ov.far_dirty = False
            if is_store and box_fast:
                self.gpu.touch_box()                   # no table entry changed here (admissions sent theirs)
            else:
                self.gpu.touch_tables(ov.vol)
        self.gpu.refresh_transforms()
        res = {'rebuilt': rebuilt, 'bricks': nb, 'dirty_world': dirty, 'ms': (time.perf_counter() - t0) * 1000,
               'light': bool(known_current and not changed)}
        self.stats = res
        return res

    def scrub_step(self, scene, budget_slots=SCRUB_SLOTS):
        """fs-brush 11-12 (brush-study's safety net for the light syncs): at idle, one slice of the sculpt store checked
        against the checksums the mirror holds (and the slots' owners). A slot that differs was changed by a writer that
        never touched the scene (the light sync relies on touch discipline as the version gate always did): its object's
        bricks are re-sampled by the next sync, which the caller forces (GpuView.scrub). -> {'checked', 'differ',
        'done'} or None (nothing to check)."""
        st = scene.store
        n = st.used
        if self._S is None or len(self._S) != n:
            return None
        a = int(getattr(self, 'scrub_pos', 0))
        if a >= n:
            return {'checked': 0, 'differ': 0, 'done': True}
        b = min(n, a + int(budget_slots))
        c = _ck(st.delta[a:b].reshape(b - a, 512))
        idx = np.nonzero(c != self._S[a:b])[0]
        self.scrub_pos = b
        if len(idx):
            self.scrub_found = getattr(self, 'scrub_found', 0) + int(len(idx))
            self._S[idx + a] = np.nan                # NaN equals nothing: the next full sync finds exactly these slots
            self.scrub_dirty = True                  # and the next sync is a full one
        return {'checked': b - a, 'differ': int(len(idx)), 'done': b >= n}

    _S = None
    _owner = {}
    grow_hint = {}             # r40: oid -> (lo, hi) local box the next rebuild's grid must cover
    coarse_pending = []        # r37: [(oid, GPU brick keys)] coarse refreshes queued for an idle frame
    live_cells = None          # fn(oid) -> set of GPU cell keys a live GPU stroke holds (gpu/sculpt.py), or None

    def mark_current(self, scene, o, keys):
        """r17: these sculpt bricks of object o are current on the card already (GPU strokes wrote them): record their
        slots' checksums so the next sync does not re-sample them."""
        st = scene.store
        if self._S is None:
            return
        sl = [o.sculpt.slots[k] for k in keys if k in o.sculpt.slots]
        if not sl:
            return
        sl = np.array(sl, np.int64)
        if sl.max() >= len(self._S):
            n0 = len(self._S)
            S = np.zeros(st.used, np.float64)
            S[:n0] = self._S
            S[n0:] = _ck(st.delta[n0:st.used].reshape(st.used - n0, 512))   # r20: real checksums, not zeros
            self._S = S
        self._S[sl] = _ck(st.delta[sl].reshape(len(sl), 512))
        own = self._owner                            # r24: in place (a copy of the whole map per release cost ms)
        sls = o.sculpt.slots
        for k in keys:
            s_ = sls.get(k)
            if s_ is not None:
                own[s_] = (o.id, k)

    @staticmethod
    def _world_box(row, box):
        lo, hi = box
        R = _rot(row)
        s = row[12]
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = (R @ c.T).T * s + row[9:12]
        return w.min(0), w.max(0)

    def _set_row(self, ov, pk, oi, scene):
        v = ov.vol
        if v is None:
            return
        row = pk.of_tab[oi]
        o = scene.objects[oi]
        v.set_transform(R=_rot(row), t=row[9:12], s=1.0 / max(row[12], 1e-9))
        v.visible = bool(scene.shown(o))
        v.albedo = tuple(float(x) for x in o.colour['albedo'])
        v.oid = o.id
        sv = getattr(ov, 'store_vol', None)
        if sv is not None:                           # milestone: her native store, as her object
            sv.set_transform(R=_rot(row), t=row[9:12], s=1.0 / max(row[12], 1e-9))
            sv.visible = v.visible
            sv.albedo = v.albedo
            sv.oid = o.id

    def _drop(self, ov, keep_store=False):
        sv = getattr(ov, 'store_vol', None)
        if sv is not None and not keep_store:
            if sv in self.gpu.vols:
                self.gpu.vols.remove(sv)
            ov.store_vol = None
            self.gpu._dirty_tables = True
        if ov.vol is not None and ov.vol in self.gpu.vols:
            self.gpu.vols.remove(ov.vol)
            self.gpu.free_rows(list(ov.key_to_row.values()) + list(getattr(ov, 'coarse', {}).values()))
            ov.coarse = {}
            self.gpu._dirty_tables = True
        ov.key_to_row = {}
        ov.vol = None

    # ---- whole-object build ---------------------------------------------------------------------------------
    def _field(self, scene, oi):
        from ..core.bake import object_field_fn
        f, L = object_field_fn(scene, oi)
        self._pk = scene.pack()
        self._oi = oi
        self._scene = scene
        return f, max(L, 1.0)

    @staticmethod
    def _store_node(o):
        """milestone: the 'fv8store' node of an object's base (her store as an editable base), or None"""
        for n in o.base.walk():
            if n.kind == 'fv8store':
                return n
        return None

    def _store_cells(self, o, ov, lo_g, bm, nt):
        """milestone: the cells an editable store object needs on the card: its non-zero sculpt bricks (+1 cell) and
        the footprint of every base part that is not her store (merged shapes, protected parts), + 2 cells"""
        from ..core import ops as OPS
        out = []
        st = None
        g = o.sculpt
        ks = [k for k, s_ in g.slots.items() if s_ >= 0]
        if ks:
            k = np.array(ks, np.int64) + ov.boff
            off = np.stack(np.meshgrid(*[np.arange(-RING, RING + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
            out.append((k[:, None, :] + off[None]).reshape(-1, 3))
        def walk(n):
            has = any(m.kind == 'fv8store' for m in n.walk())
            if not has:
                # milestone r2: the cells where this part can change her field, found per tile (BPT cells): the part's
                # field within reach (tile radius + band + blend) and her surface within reach. Its box alone was a
                # 3.5 m cube for a Cut's cutter (every cell of her body at 0.25 mm: the card was never built)
                from ..core.fv8writeback import _eval_many
                code, prm, lo, hi, L = OPS.compile_tree(n)
                tmc = bm * BPT
                a = np.maximum(np.floor((np.asarray(lo) - lo_g) / tmc).astype(np.int64) - 1, 0)
                b = np.minimum(np.ceil((np.asarray(hi) - lo_g) / tmc).astype(np.int64) + 1, np.asarray(nt))
                if not np.all(b > a):
                    return
                T = np.stack(np.meshgrid(*[np.arange(a[i], b[i]) for i in range(3)], indexing='ij'), -1).reshape(-1, 3)
                C = np.ascontiguousarray(lo_g + (T + 0.5) * tmc)
                reach = 0.5 * math.sqrt(3) * tmc + BAND_H * bm / 8 + 0.004 + 2 * bm
                fs = np.empty(len(C))
                _eval_many(code, prm, C, fs)
                T, C = T[fs <= reach], C[fs <= reach]
                if not len(T):
                    return
                fo = ov.f(C)
                T = T[np.abs(fo) <= reach]
                if not len(T):
                    return
                off = np.stack(np.meshgrid(*[np.arange(BPT)] * 3, indexing='ij'), -1).reshape(-1, 3)
                out.append((T[:, None, :] * BPT + off[None]).reshape(-1, 3))
                return
            if n.kind == 'fv8store':
                return
            for c in n.children:
                walk(c)
        walk(o.base)
        if not out:
            return np.zeros((0, 3), np.int64)
        c = _ukeys(np.concatenate(out))
        n = np.asarray(nt) * BPT
        return c[np.all((c >= 0) & (c < n), 1)]

    def _ensure_store_volume(self, o, ov, node):
        sv = getattr(ov, 'store_vol', None)
        if sv is None or getattr(sv, 'src', None) != node.params['source']:
            if sv is not None and sv in self.gpu.vols:
                self.gpu.vols.remove(sv)
            sv = self.gpu.add_fv8_store(node.params['source'], name='store%d' % o.id)
            sv.src = node.params['source']
            ov.store_vol = sv
        sv.mask_vol = ov.vol
        self.gpu._dirty_tables = True

    def _rebuild(self, scene, oi, ov):
        o = scene.objects[oi]
        h = float(o.sculpt.h)
        store = self._store_node(o)
        f, L = self._field(scene, oi)
        lo, hi = o.local_box()
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        gh = self.grow_hint.pop(o.id, None)          # r40: a dab's reach that overflowed the old grid
        if gh is not None:
            lo, hi = np.minimum(lo, gh[0]), np.maximum(hi, gh[1])
        tm = 8 * h * BPT
        g = np.maximum(GROW * (hi - lo), 4 * tm)
        # fs-brush 11-12 (a block-out, tk4: a sphere stretched into a head grows ~1.6x; the 25 % margin overflowed
        # mid-stroke and the rebuild cost a 0.74-0.8 s frame gap): small objects get up to BLOCKOUT_GROW of their box,
        # capped at BLOCKOUT_CAP m a side (large objects keep their 25 %: it already exceeds the cap)
        if float(np.max(hi - lo)) <= BLOCKOUT_MAX:     # fs-brush-3 11-13: small objects only (see BLOCKOUT_CAP)
            g = np.maximum(g, np.minimum(BLOCKOUT_GROW * (hi - lo), BLOCKOUT_CAP))
        t_lo = np.floor((lo - g) / tm).astype(np.int64)
        t_hi = np.ceil((hi + g) / tm).astype(np.int64)
        nt = t_hi - t_lo
        lo_g = t_lo * tm
        band = BAND_H * h
        levels = [None] * NLEV
        levels[LEVEL] = {'h': h, 'band': band, 'bpt': BPT, 'tile_table': np.zeros((0, BPT ** 3), np.int32)}
        levels[LEVEL_C] = {'h': C_STRIDE * h, 'band': C_BAND_H * h, 'bpt': 1, 'tile_table': np.zeros((0, 1), np.int32)}
        if store is not None:
            levels[LEVEL_C] = None                  # her store draws her; a coarse copy would be a second surface
        v = Volume('obj%d' % o.id, lo_g, tm, tuple(int(x) for x in nt), levels, lo_g, lo_g + nt * tm,
                   tuple(o.colour['albedo']))
        v.grid = np.full(tuple(int(x) for x in nt) + (NLEV,), -1, np.int32)
        self._drop(ov, keep_store=True)
        ov.vol = v
        # r15: cell -> pool row (milestone: int32 for an editable store object: her whole box at 0.25 mm is 83 M cells)
        ov.rowmap = np.full(tuple(int(x) * BPT for x in nt), -1, np.int32 if store is not None else np.int64)
        # fs-brush 11-12: a new map: the card's copy (gpu/sculpt.py _rowmap_dev) is uploaded whole once, then follows the
        # cells logged below
        ov.rm_gen = getattr(ov, 'rm_gen', 0) + 1
        ov.rm_log = None
        ov.is_store = store is not None
        ov.far_dirty = True
        ov.new_keys = None                         # a full far field at the next sync
        ov.boff = -t_lo * BPT                      # GPU brick ijk of sculpt brick 0
        ov.f, ov.L = f, L
        self.gpu.vols.append(v)
        self.gpu._dirty_tables = True
        bm = 8 * h
        if store is not None:
            from ..core.fv8writeback import _eval_many
            sc_, sp_, *_r = ops_compile(store)

            def f_store(X, _c=sc_, _p=sp_):
                X = np.ascontiguousarray(X, np.float64)
                out = np.empty(len(X))
                _eval_many(_c, _p, X, out)
                return out
            ov.f_store = f_store
            cells = self._store_cells(o, ov, lo_g, bm, nt)
            n = self._admit_cells(ov, cells)
            ov.coarse = {}
            self._ensure_store_volume(o, ov, store)
            return n
        i0 = np.floor((lo - lo_g) / bm).astype(np.int64) - 1
        i1 = np.ceil((hi - lo_g) / bm).astype(np.int64) + 1
        rng = [np.arange(max(i0[a], 0), min(i1[a], nt[a] * BPT)) for a in range(3)]
        cells = np.stack(np.meshgrid(*rng, indexing='ij'), -1).reshape(-1, 3)
        n = self._admit_cells(ov, cells)
        ov.coarse = {}
        self.coarse_refresh(scene, oi, ov, None)
        return n

    def coarse_flush(self, scene):
        """r37: run the queued coarse refreshes. -> world boxes changed (for the frame's re-light).
        fs-brush-2 11-13 (mv8g: after a 25 k-brick stroke the whole refresh ran in one idle frame, 95-196 ms, the longest
        turn after the release on either release path): at most COARSE_FLUSH_KEYS fine bricks a frame (each piece is
        recomputed from the current field, so splitting a stroke's keys changes no value); the rest stays queued"""
        q, rest, n = [], [], 0
        for oid, keys in self.coarse_pending:
            if COARSE_FLUSH_KEYS <= 0 or n < COARSE_FLUSH_KEYS:
                k = len(keys) if COARSE_FLUSH_KEYS <= 0 else max(COARSE_FLUSH_KEYS - n, 0)
                q.append((oid, keys[:k]))
                n += min(k, len(keys))
                if k < len(keys):
                    rest.append((oid, keys[k:]))
            else:
                rest.append((oid, keys))
        self.coarse_pending = rest
        boxes = []
        pk = scene.pack()
        for oid, keys in q:
            oi = scene.index_of(oid)
            ov = self.objs.get(oid)
            if oi < 0 or ov is None or ov.vol is None:
                continue
            if not len(keys):
                continue
            self.coarse_refresh(scene, oi, ov, keys)
            bm = 8 * ov.vol.levels[LEVEL]['h']
            lo = ov.vol.origin + (keys.min(0) - BPT) * bm
            hi = ov.vol.origin + (keys.max(0) + 1 + BPT) * bm
            boxes.append(self._world_box(pk.of_tab[oi], (lo, hi)))
        return boxes

    def coarse_refresh(self, scene, oi, ov, fine_keys):
        """r34: (re)build the coarse level over the tiles holding fine_keys (GPU brick ijk) and their neighbours, or
        over every tile near the surface (None). Values from the card's evaluator (coarse nodes are sculpt nodes)."""
        v = ov.vol
        if v is None or v.levels[LEVEL_C] is None:
            return 0
        o = scene.objects[oi]
        h = float(o.sculpt.h)
        tm = v.tile_m
        nt = np.array(v.dims)
        if fine_keys is None:
            tiles = np.stack(np.meshgrid(*[np.arange(n) for n in nt], indexing='ij'), -1).reshape(-1, 3)
            f, L = self._field(scene, oi)
            fc = f(v.origin + (tiles + 0.5) * tm)
            tiles = tiles[np.abs(fc) <= L * 0.8661 * tm + C_BAND_H * h]
        else:
            t0 = np.asarray(fine_keys, np.int64).reshape(-1, 3) // BPT
            offs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
            tiles = _ukeys((t0[:, None, :] + offs[None]).reshape(-1, 3))
            tiles = tiles[np.all((tiles >= 0) & (tiles < nt), 1)]
        if len(tiles) == 0:
            return 0
        cp = self.gpu.cp
        vals = self.geval.eval(scene, o, v.origin + tiles * tm, tiles * BPT - ov.boff, stride=C_STRIDE)
        band_c = C_BAND_H * h
        coarse = getattr(ov, 'coarse', None)
        if coarse is None:
            coarse = ov.coarse = {}
        held = np.array([tuple(t) in coarse for t in tiles.tolist()], bool)
        keep = held | (cp.abs(vals).min(1).get() < band_c)
        tiles, vals = tiles[keep], vals[cp.asarray(keep)]
        if len(tiles) == 0:
            return 0
        lv = v.levels[LEVEL_C]
        tt = lv['tile_table']
        rows = np.empty(len(tiles), np.int64)
        new = [i for i, t in enumerate(tiles.tolist()) if tuple(t) not in coarse]
        for i, t in enumerate(tiles.tolist()):
            if tuple(t) in coarse:
                rows[i] = coarse[tuple(t)]
        if new:
            nr = self.gpu.alloc_rows(len(new))
            base = tt.shape[0]
            tt = np.concatenate([tt, np.full((len(new), 1), -1, np.int32)])
            for j, i in enumerate(new):
                t = tuple(tiles[i].tolist())
                v.grid[t[0], t[1], t[2], LEVEL_C] = base + j
                tt[base + j, 0] = nr[j]
                coarse[t] = int(nr[j])
                rows[i] = nr[j]
            lv['tile_table'] = tt
            v.brick_off[LEVEL_C] = 0
            self.gpu.touch_tables(v)
        self.gpu.sdf[cp.asarray(rows)] = vals.astype(cp.float16)
        return len(tiles)

    @staticmethod
    def _far_add(vol, cells, reach=12):
        """r11: new bricks can only lower the distance to the nearest brick: min-update the far field within `reach`
        cells of them (beyond it the stored value stays a lower bound only if it is <= reach, so it is capped there).
        Freed bricks leave smaller (still valid, conservative) values until the next full rebuild."""
        far = vol.far
        n = np.array(far.shape)
        lo = np.maximum(cells.min(0) - reach, 0)
        hi = np.minimum(cells.max(0) + reach + 1, n)
        win = far[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        from scipy.ndimage import distance_transform_edt
        occ = np.ones(win.shape, bool)
        c = cells - lo
        occ[c[:, 0], c[:, 1], c[:, 2]] = False
        best = distance_transform_edt(occ)
        np.minimum(win, np.minimum(np.floor(best), 255).astype(np.uint8), out=win)
        # a cell outside the window may now be nearer than its stored value to a new brick: cap the far field at
        # reach everywhere outside the window's distance-safe core
        np.minimum(far, np.uint8(reach), out=far, where=far > reach)

    @staticmethod
    def _tight_box(vol, keys):
        """r11: the box the tracer clips to = the allocated bricks' extent (not the grid's growth margin), so rays and
        shadow candidates skip a volume wherever it holds nothing."""
        bm = 8 * vol.levels[LEVEL]['h']
        if len(keys) == 0:
            vol.lo = vol.origin.copy()
            vol.hi = vol.origin.copy()
            return
        vol.lo = vol.origin + keys.min(0) * bm
        vol.hi = vol.origin + (keys.max(0) + 1) * bm

    # ---- incremental --------------------------------------------------------------------------------------------
    def _update_keys(self, scene, oi, ov, keys):
        """Re-sample the GPU bricks fed by the changed sculpt bricks. -> (ok, bricks, local box or None)."""
        v = ov.vol
        o = scene.objects[oi]
        h = float(o.sculpt.h)
        if abs(h - v.levels[LEVEL]['h']) > 1e-15:
            return False, 0, None
        ov.f, ov.L = self._field(scene, oi)
        k = np.asarray(keys, np.int64).reshape(-1, 3) + ov.boff
        cells = _ukeys((k[:, None, :] + _OFFS[None]).reshape(-1, 3))
        nb = np.array(v.dims) * BPT
        if np.any(cells < 0) or np.any(cells >= nb):
            return False, 0, None
        live = self.live_cells(o.id) if self.live_cells is not None else None
        if live:                                     # r20: a live GPU stroke's bricks are newer on the card than
            keep = np.array([tuple(c) not in live for c in cells.tolist()], bool)   # in the CPU layer until its release
            cells = cells[keep]
        n = self._admit_cells(ov, cells)
        self.coarse_refresh(scene, oi, ov, cells)            # r34
        bm = 8 * h
        lo = v.origin + cells.min(0) * bm
        hi = v.origin + (cells.max(0) + 1) * bm
        return True, n, (lo, hi)

    def _admit_cells(self, ov, cells):
        """(Re)sample the given GPU brick cells (ijk): keep the band, free what left it."""
        if len(cells) == 0:
            return 0
        v = ov.vol
        lv = v.levels[LEVEL]
        h, band = lv['h'], lv['band']
        bm = 8 * h
        # r20: a brick the volume already holds is refreshed, never freed (GPU strokes hold every brick their dabs
        # cover, deep ones too; freeing them here let a live stroke's rows be reused by its own admissions: 46 rows
        # aliased, bricks off by 22-50 mm in the app). New bricks are admitted only where the band can be.
        held = ov.rowmap[cells[:, 0], cells[:, 1], cells[:, 2]] >= 0
        centres = v.origin + (cells + 0.5) * bm
        fc = ov.f(centres)
        reach = ov.L * 0.5 * math.sqrt(3) * bm + band
        cand = held | (np.abs(fc) <= reach)
        near_old = None
        if getattr(ov, 'f_store', None) is not None:
            # milestone (r-ghost): an editable store object's card bricks also cover her store's OWN surface wherever it
            # is in the candidate cells: those bricks mask it; else an edit that moved her surface more than the band
            # left the old surface drawn (a ghost that shadowed the light: black patches after a reopen)
            fsx = ov.f_store(centres)
            # only where the edit changed her field (else her store draws there and no border is made)
            near_old = (np.abs(fsx) <= reach) & (np.abs(fc - fsx) > 1e-7)
            # 11-09 ('spikes' at a merged block's ends, asHim_4: her old surface drawn through thin fins where
            # the card's hits lay up to 0.34 mm inside the new body): the change was judged at the cell's CENTRE only;
            # where it lies in a corner the cell was never admitted. A cell now also counts as changed when any sculpt
            # brick its nodes read holds a non-zero change
            o_ = self._scene.objects[self._oi]
            g_ = o_.sculpt
            if g_.slots:
                st_ = self._scene.store
                sk = cells - ov.boff
                corn = np.array([[a_, b_, c_] for a_ in (0, 1) for b_ in (0, 1) for c_ in (0, 1)], np.int64)
                chg = np.zeros(len(cells), bool)
                nzb = {b_ for b_, s_ in g_.slots.items() if s_ >= 0 and st_.delta[s_].any()}
                if nzb:
                    for cc in corn:
                        kk = sk + cc                      # a cell reads its own brick and the next ones (node 8)
                        chg |= np.array([tuple(x) in nzb for x in kk.tolist()], bool)
                near_old |= (np.abs(fsx) <= reach) & chg
            cand |= near_old
        keys = cells[cand]
        held = held[cand]
        if near_old is not None:
            near_old = near_old[cand]
        kept = []
        if len(keys):
            cp = self.gpu.cp
            o = self._scene.objects[self._oi]
            # 11-09 (an out-of-memory in the app: a grown torso's rebuild evaluated ~760k candidate cells at once, 2.2 GB
            # on top of 4.7 GB in use): the candidates in chunks, each evaluated, filtered and written before the next
            for c0 in range(0, len(keys), ADMIT_CHUNK):
                kc = keys[c0:c0 + ADMIT_CHUNK]
                vals = self.geval.eval(self._scene, o, v.origin + kc * bm, kc - ov.boff)   # r22: on the card
                ok = held[c0:c0 + ADMIT_CHUNK] | (cp.abs(vals).min(1).get() < band)
                if near_old is not None:
                    ok |= near_old[c0:c0 + ADMIT_CHUNK]
                kc, vals = kc[ok], vals[cp.asarray(ok)]
                if not len(kc):
                    continue
                # r14: true values (fp16 holds them); a clamp lost the far side of the band that a dab can bring in
                rows = np.empty(len(kc), np.int64)
                new = []
                for n_, k in enumerate(map(tuple, kc.tolist())):
                    r = ov.key_to_row.get(k)
                    if r is None:
                        new.append(n_)
                    else:
                        rows[n_] = r
                if new:
                    new = np.array(new)
                    rows[new] = self._alloc_bricks(ov, kc[new])
                # 11-09 guard: fp16 holds +-65504; a larger (or non-finite) value became inf on the card (Smooth then
                # made NaN, the ray-march stepped by it); her field is a distance in metres, so +-10 m bounds every
                # true value
                vals = cp.clip(cp.nan_to_num(vals, nan=10.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
                self.gpu.sdf[cp.asarray(rows)] = vals.astype(cp.float16)
                del vals
                kept.append(kc)
        keys = np.concatenate(kept) if kept else np.zeros((0, 3), np.int64)
        if len(keys):
            self.fill_aprons(ov, keys)
        return int(len(keys))

    # r43: every node is evaluated on its own, so a brick's apron (its +1 neighbours' node 0) and the neighbour's owned
    # node could differ by an fp16 ulp where the two were sampled by different calls (K3: 10,971 of 17.4 M apron nodes).
    # The aprons of the admitted bricks and of their lower neighbours become copies of the owners, bit for bit.
    _K_APRON = None
    _APRON_SRC = r"""
    #include <cuda_fp16.h>
    extern "C" __global__ void mirror_apron(__half* sdf, const long long* rows, const long long* nb, int n) {
        // r47: one thread per (brick, apron node): node (i, j, k) with any index 8 copies node (i&7, j&7, k&7) of the
        // neighbour brick at offset (i==8, j==8, k==8); nb[7 b + o - 1] = that neighbour's row (-1: none)
        long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= (long long)n * 729) return;
        int b = (int)(id / 729), q = (int)(id % 729);
        int i = q / 81, j = (q / 9) % 9, k = q % 9;
        int o = (i == 8) * 4 + (j == 8) * 2 + (k == 8);
        if (o == 0) return;
        long long r = nb[7LL * b + o - 1];
        if (r < 0) return;
        sdf[rows[b] * 729 + q] = sdf[r * 729 + ((i & 7) * 9 + (j & 7)) * 9 + (k & 7)];
    }
    """
    _OFFS = np.array([[(o >> 2) & 1, (o >> 1) & 1, o & 1] for o in range(1, 8)], np.int64)

    def fill_aprons(self, ov, keys):
        """r43/r47: the aprons of the given bricks and of their lower neighbours become copies of the owners' nodes,
        bit for bit (every node is evaluated on its own, so an apron and its owner sampled by different calls could
        differ by an fp16 ulp: K3, 10,971 of 17.4 M). r47: on the card, 7 neighbour rows per brick (r43 built 217
        index pairs per brick on the host: ~0.5 GB for a full-figure rebuild)."""
        if len(keys) == 0 or getattr(ov, 'rowmap', None) is None:
            return
        cp = self.gpu.cp
        if SceneMirror._K_APRON is None:
            SceneMirror._K_APRON = cp.RawKernel(self._APRON_SRC, 'mirror_apron', options=('-std=c++14',))
        keys = np.asarray(keys, np.int64).reshape(-1, 3)
        allk = _ukeys((keys[:, None, :] - self._OFFS[None]).reshape(-1, 3))
        allk = _ukeys(np.concatenate([keys, allk]))
        n = np.array(ov.rowmap.shape)
        allk = allk[np.all((allk >= 0) & (allk < n), 1)]
        if not len(allk):
            return
        rw = ov.rowmap[allk[:, 0], allk[:, 1], allk[:, 2]]
        allk, rw = allk[rw >= 0], rw[rw >= 0].astype(np.int64)
        if not len(rw):
            return
        nk = allk[:, None, :] + self._OFFS[None]
        ok = np.all(nk < n, 2)
        nb = np.full(nk.shape[:2], -1, np.int64)
        nb[ok] = ov.rowmap[nk[ok][:, 0], nk[ok][:, 1], nk[ok][:, 2]]
        th = 256
        SceneMirror._K_APRON(((len(rw) * 729 + th - 1) // th,), (th,),
                             (self.gpu.sdf, cp.asarray(rw), cp.asarray(np.ascontiguousarray(nb)), np.int32(len(rw))))

    def _alloc_bricks(self, ov, keys):
        v = ov.vol
        lv = v.levels[LEVEL]
        rows = self.gpu.alloc_rows(len(keys))
        tiles = keys // BPT
        tt = lv['tile_table']
        need = sorted({tuple(t) for t in tiles.tolist() if v.grid[t[0], t[1], t[2], LEVEL] < 0})
        g_idx = None
        if need:
            base = int(lv.get('n_tiles', tt.shape[0]))
            for n_, t in enumerate(need):
                v.grid[t[0], t[1], t[2], LEVEL] = base + n_
            nd = np.asarray(need, np.int64).reshape(-1, 3)
            g_idx = np.ravel_multi_index((nd[:, 0], nd[:, 1], nd[:, 2], np.full(len(nd), LEVEL)), v.grid.shape)
            if base + len(need) > tt.shape[0]:
                grow = max(len(need), 64, tt.shape[0] // 4)       # spare rows: the table layout holds (Grab fps)
                tt = np.concatenate([tt, np.full((grow, BPT ** 3), -1, np.int32)])
            lv['n_tiles'] = base + len(need)
        li = keys - tiles * BPT
        s = v.grid[tiles[:, 0], tiles[:, 1], tiles[:, 2], LEVEL]
        col = (li[:, 0] * BPT + li[:, 1]) * BPT + li[:, 2]
        tt[s, col] = rows
        t_idx = s.astype(np.int64) * tt.shape[1] + col
        for row, key in zip(rows.tolist(), map(tuple, keys.tolist())):
            ov.key_to_row[key] = row
        if getattr(ov, 'rowmap', None) is not None:
            ov.rowmap[keys[:, 0], keys[:, 1], keys[:, 2]] = rows
            _rm_note(ov, keys, rows)                 # fs-brush 11-12: for the card's copy of the map
        lv['tile_table'] = tt
        v.brick_off[LEVEL] = 0               # the tile table holds pool rows directly
        # milestone: the tracer's box grows with the bricks at once (an editable store object starts with none; its
        # strokes' bricks were masked from her store but outside the box the rays clip to: holes)
        bm = 8 * lv['h']
        klo = v.origin + keys.min(0) * bm
        khi = v.origin + (keys.max(0) + 1) * bm
        if np.all(np.asarray(v.lo) == np.asarray(v.hi)):
            v.lo, v.hi = klo, khi
        else:
            v.lo, v.hi = np.minimum(v.lo, klo), np.maximum(v.hi, khi)
        ov.far_dirty = True
        if getattr(ov, 'new_keys', None) is not None:
            ov.new_keys.append(np.asarray(keys, np.int64))
        # fs-brush 11-12: the entries written above, by flat index (gpu/scene.py touch_tables); a grown tile table changes
        # the layout, which uploads the whole tables anyway
        self.gpu.touch_tables(v, grid_idx=g_idx, tt_idx={LEVEL: t_idx})
        return rows

    def _free_brick(self, ov, key):
        v = ov.vol
        lv = v.levels[LEVEL]
        row = ov.key_to_row.pop(key)
        if getattr(ov, 'rowmap', None) is not None:
            ov.rowmap[key] = -1
            _rm_note(ov, np.asarray(key, np.int64).reshape(1, 3), np.full(1, -1, np.int64))   # fs-brush 11-12
        t = (key[0] // BPT, key[1] // BPT, key[2] // BPT)
        s = v.grid[t[0], t[1], t[2], LEVEL]
        li = (key[0] - t[0] * BPT, key[1] - t[1] * BPT, key[2] - t[2] * BPT)
        col = (li[0] * BPT + li[1]) * BPT + li[2]
        lv['tile_table'][s, col] = -1
        self.gpu.free_rows([row])
        self.gpu.touch_tables(v, tt_idx={LEVEL: np.array([int(s) * lv['tile_table'].shape[1] + col], np.int64)})
        ov.far_dirty = True
        ov.freed_since_box = True        # fs-brush 11-12: the tracer's box may shrink: sync recomputes it from every key
