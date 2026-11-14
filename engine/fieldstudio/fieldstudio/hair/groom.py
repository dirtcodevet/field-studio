"""One hair system on one object (fs-hair, 11-11): the natural base, the current hair, its states, and the acts on it.

Styles as states (design 11-11: the natural base first, styles as states on it): the generator grows
the NATURAL BASE once (how the hair grows and falls, from its causes); every style act (cut, comb, settle) changes the
CURRENT hair; a named STATE keeps a styled hair as a change from the base; Water (the wash) erases the set and returns
the hair to its natural fall. A cut is not erased by water (a cut hair stays cut): the base keeps its lengths only until
a cut, which re-bases the lengths (water then restores the base's shape at the new lengths).

Frames: every array is in the OBJECT's own frame (it follows the object when moved). The window's camera and depth are
carried into that frame for drawing and brushing (uniform scale s: distances / s), so nothing is copied to world.
"""
import json
import math
import os
import time

import numpy as np

from . import style as ST


def _rot(rx, ry, rz):
    """Field Studio's object rotation (degrees, X then Y then Z, as tools.protected_merge)"""
    from ..tools.protected_merge import _to_world

    class _O:
        transform = {'translate': [0, 0, 0], 'rotate': [rx, ry, rz], 'scale': 1.0}
    E = _to_world(_O(), np.eye(3))
    O = _to_world(_O(), np.zeros((1, 3)))
    return (E - O).T          # columns: the images of the local axes


def comb_root_mm(v=None):
    """how much of each hair the follicle holds against a comb (mm along the strand); FS_HAIR_COMB_ROOT_MM overrides"""
    if v is not None:
        return float(v)
    return float(os.environ.get('FS_HAIR_COMB_ROOT_MM', '8'))


class Frame:
    """world <- local: x_w = s R x_l + t"""

    def __init__(self, transform=None):
        tr = transform or {}
        self.t = np.asarray(tr.get('translate', (0, 0, 0)), float)
        r = tr.get('rotate', (0, 0, 0))
        self.s = float(tr.get('scale', 1.0))
        self.R = _rot(*r) if any(abs(float(v)) > 1e-12 for v in r) else np.eye(3)
        self.key = (tuple(np.round(self.t, 9)), tuple(np.round(np.asarray(r, float), 9)), round(self.s, 9))

    def to_world(self, P):
        P = np.asarray(P, np.float64)
        return (P @ self.R.T) * self.s + self.t

    def to_local(self, P):
        P = np.asarray(P, np.float64)
        return ((P - self.t) / self.s) @ self.R

    def dir_local(self, D):
        return np.asarray(D, np.float64) @ self.R

    def matrix(self):
        M = np.eye(4)
        M[:3, :3] = self.R * self.s
        M[:3, 3] = self.t
        return M

    def camera(self, cam):
        """the window's camera carried into the local frame (a Camera with the same fov)"""
        c = cam.copy()
        c.pos = self.to_local(cam.pos[None])[0]
        c.target = self.to_local(cam.target[None])[0]
        c.up = self.dir_local(cam.up[None])[0]
        return c


OPTICS = ('ELL', 'TAU', 'BM', 'PIG', 'SEED', 'KIND')      # the base's own fibre optics, kept for the export (fs-hair 11-12)


def by_arc(A, SEGb, SEGc):
    """per-point values A (n, k[, c]) on the base strands -> the same at each current point's arc length from the root (n, k[, c])"""
    n, k = A.shape[:2]
    sb = np.zeros((n, k))
    sb[:, 1:] = np.cumsum(np.nan_to_num(np.asarray(SEGb, np.float64)), 1)
    sc = np.zeros((n, k))
    sc[:, 1:] = np.cumsum(np.nan_to_num(np.asarray(SEGc, np.float64)), 1)
    span = float(max(sb[:, -1].max(), sc[:, -1].max())) + 1.0
    off = np.arange(n, dtype=np.float64)[:, None] * span
    Gl = (sb + off).ravel()
    Q = (np.minimum(sc, sb[:, -1:]) + off).ravel()
    j = np.searchsorted(Gl, Q, side='right') - 1
    row = np.repeat(np.arange(n), k)
    j = np.clip(j, row * k, row * k + k - 2)
    w = np.clip((Q - Gl[j]) / np.maximum(Gl[j + 1] - Gl[j], 1e-12), 0.0, 1.0)
    Af = A.reshape(n * k, -1).astype(np.float64)
    out = Af[j] * (1.0 - w)[:, None] + Af[j + 1] * w[:, None]
    return out.reshape(A.shape).astype(A.dtype)


def seg_of(P):
    """each segment's length (n, k-1) float32 from the points P (n, k, 3)"""
    P = np.asarray(P, np.float32)
    return np.sqrt(np.sum(np.diff(P, axis=1) ** 2, axis=2), dtype=np.float32).astype(np.float32)


def _compact(idx, P, RAD, COL, SEG):
    """an undo / state snapshot: P and SEG (the rest lengths) float32 exact, RAD and COL float16"""
    return {'idx': idx, 'P': np.array(P, np.float32), 'RAD': np.array(RAD, np.float16), 'COL': np.array(COL, np.float16),
            'SEG': np.array(SEG, np.float32)}


def _snap_bytes(s):
    return sum(int(getattr(s.get(k), 'nbytes', 0)) for k in ('P', 'RAD', 'COL', 'SEG', 'idx'))


class HairGroom:
    def __init__(self, oid, name='hair', settings=None):
        self.oid = int(oid)
        self.name = name
        self.settings = dict(settings or {})
        self.region = None              # dict Q, N, area (local)
        self.base = None                # the natural base: P, RAD, COL, SEG (local)
        self.cur = None
        self.grid = None                # BodyGrid in the local frame
        self.undo_stack = []
        self.redo_stack = []
        self.states = {}                # name -> sparse change from the base
        self.log = []                   # every act, with its numbers (the record)
        self.version = 0
        self.visible = True
        self.report = {}

    # ---- the base ----------------------------------------------------------------------------------------------------
    def set_base(self, P, RAD, COL, grid=None, report=None, extra=None):
        self._box_version = getattr(self, '_box_version', 0) + 1
        P = np.ascontiguousarray(P, np.float32)
        # the base compact (fs-hair 11-12, to save memory): P as is, RAD and COL in half precision, the
        # lengths from P when wanted; _b(k) gives full-precision copies
        # SEG stays exact: the rest lengths along each strand (a resample's arc steps), not always |dP| (11-12 test)
        SEG = np.linalg.norm(np.diff(P.astype(np.float64), axis=1), axis=2).astype(np.float32)
        self.base = {'P': P, 'RAD': np.ascontiguousarray(RAD, np.float16), 'COL': np.ascontiguousarray(COL, np.float16),
                     'SEG': SEG}
        if extra:
            self.base.update({k: v for k, v in extra.items() if k not in ('P', 'RAD', 'COL', 'SEG')})
        self.cur = {'P': P.copy(), 'RAD': np.ascontiguousarray(RAD, np.float32).copy(),
                    'COL': np.ascontiguousarray(COL, np.float32).copy(), 'SEG': SEG.copy()}
        self.grid = grid
        self.undo_stack, self.redo_stack = [], []
        self.report = dict(report or {})
        self.version += 1
        self._log('base', strands=int(len(P)), points=int(P.shape[1]))

    def _b(self, k):
        """the base's array k in full precision (a new array): SEG from P"""
        b = self.base
        if k == 'SEG':
            return b['SEG'].astype(np.float32) if 'SEG' in b else seg_of(b['P'])
        return np.array(b[k], np.float32)

    def undo_bytes(self):
        return sum(_snap_bytes(s) for s in self.undo_stack + self.redo_stack)

    def _trim_undo(self):
        """the undo memory budget (FS_HAIR_UNDO_MB, default 400): the oldest acts dropped first, the last kept"""
        cap = float(os.environ.get('FS_HAIR_UNDO_MB', '400')) * 1e6
        while len(self.undo_stack) > 1 and self.undo_bytes() > cap:
            self.undo_stack.pop(0)
        while self.redo_stack and self.undo_bytes() > cap:
            self.redo_stack.pop(0)

    def cov_mult(self):
        """a preview grows 1 in 1/p hairs: each is drawn covering 1/p (the density read kept)"""
        p = float((self.settings or {}).get('preview', 1.0) or 1.0)
        return 1.0 / max(min(p, 1.0), 1e-3)

    @property
    def n(self):
        return 0 if self.cur is None else len(self.cur['P'])

    def _log(self, act, **kw):
        kw.update(act=act, time=time.strftime('%H:%M:%S'))
        self.log.append(kw)

    # ---- undo by stroke ------------------------------------------------------------------------------------------------
    def begin(self, idx=None):
        """before an act: keep what it may change (all strands, or idx) for undo"""
        c = self.cur
        if idx is None:
            snap = _compact(None, c['P'], c['RAD'], c['COL'], c['SEG'])
        else:
            idx = np.asarray(idx, np.int64)
            snap = _compact(idx, c['P'][idx], c['RAD'][idx], c['COL'][idx], c['SEG'][idx])
        self._pending = snap

    def commit(self, touched=None, label=''):
        """after an act: keep only the strands it touched (sparse undo)"""
        snap = getattr(self, '_pending', None)
        self._pending = None
        if snap is None:
            return
        if snap['idx'] is None and touched is not None:
            t = np.where(np.asarray(touched) > 0)[0]
            if not len(t):
                return
            snap = {'idx': t, **{k: snap[k][t] for k in ('P', 'RAD', 'COL', 'SEG')}}
        snap['label'] = label
        self.undo_stack.append(snap)
        if len(self.undo_stack) > 64:
            self.undo_stack.pop(0)
        self.redo_stack = []
        self._trim_undo()
        self.version += 1

    def _swap(self, snap):
        c = self.cur
        idx = snap['idx']
        if idx is None:
            back = _compact(None, c['P'], c['RAD'], c['COL'], c['SEG'])
            c['P'] = np.array(snap['P'], np.float32)
            c['RAD'] = np.array(snap['RAD'], np.float32)
            c['COL'] = np.array(snap['COL'], np.float32)
            c['SEG'] = np.array(snap['SEG'], np.float32) if 'SEG' in snap else seg_of(c['P'])
        else:
            back = _compact(idx, c['P'][idx], c['RAD'][idx], c['COL'][idx], c['SEG'][idx])
            c['P'][idx] = snap['P']
            c['RAD'][idx] = snap['RAD']
            c['COL'][idx] = snap['COL']
            c['SEG'][idx] = snap['SEG'] if 'SEG' in snap else seg_of(c['P'][idx])
        back['label'] = snap.get('label', '')
        self.version += 1
        return back

    def undo(self):
        self._box_version = getattr(self, '_box_version', 0) + 1
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._swap(self.undo_stack.pop()))
        self._trim_undo()
        self._log('undo')
        return True

    def redo(self):
        self._box_version = getattr(self, '_box_version', 0) + 1
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._swap(self.redo_stack.pop()))
        self._trim_undo()
        self._log('redo')
        return True

    # ---- acts ------------------------------------------------------------------------------------------------------------
    def clearance(self):
        r = float(np.nanmedian(self.cur['RAD'][:, 0])) if self.n else 35e-6
        return r + 0.0002

    def _grid_args(self):
        """the body lattices packed for the kernels (style.BodyGrid.packed: coarse + fine where set)"""
        g = self.grid
        if g is None:
            return ST.no_grid()
        return g.packed()

    def _cand(self, C, W, H, cx, cy, rpx):
        """the strands whose screen box meets the brush (the boxes kept per view: all projected once, then only the
        strands a dab may move are re-projected after it)"""
        key = (tuple(np.round(C, 9)), int(W), int(H), self.n)
        bx = getattr(self, '_boxes', None)
        if bx is None or bx[0] != key or getattr(self, '_boxes_v', None) != self._box_version:
            box = np.empty((self.n, 4), np.float64)
            ST.strand_boxes(self.cur['P'], C, W, H, np.arange(self.n, dtype=np.int64), box)
            self._boxes = bx = (key, box)
            self._boxes_v = self._box_version
        box = bx[1]
        m = (box[:, 0] - rpx <= cx) & (box[:, 1] + rpx >= cx) & (box[:, 2] - rpx <= cy) & (box[:, 3] + rpx >= cy)
        sec = getattr(self, 'section', None)
        if sec is not None and len(sec) == self.n:
            m &= sec.astype(bool)                       # the strands pinned out of the section are not touched
        return np.nonzero(m)[0].astype(np.int64), box

    _box_version = 0

    def _boxes_after(self, C, W, H, cand, box):
        if len(cand):
            ST.strand_boxes(self.cur['P'], C, W, H, cand, box)

    def comb_dab(self, cam_l, W, H, depth_l, cx, cy, rpx, drag_l, strength, touched, band=0.004, root_mm=None):
        """one dab of the comb; drag_l: the drag in the local frame (m). touched (n,) uint8 marks the strands moved"""
        c = self.cur
        A, lo, h = self._grid_args()
        from .draw import cam_vec
        C = cam_vec(cam_l, W, H).astype(np.float64)
        I = np.eye(4)
        cand, box = self._cand(C, W, H, cx, cy, rpx)
        ST.comb_dab(c['P'], c['SEG'], c['P'], I, I, C, W, H, np.ascontiguousarray(depth_l, np.float64), float(cx),
                    float(cy), float(rpx), np.asarray(drag_l, np.float64), float(strength), float(band), A, lo, h,
                    float(self.clearance()), touched, cand, float(comb_root_mm(root_mm)) / 1000.0)
        self._boxes_after(C, W, H, cand, box)
        self.version += 1

    def pushout(self, idx):
        """the strands idx kept clear of the form, lengths kept from the root out (style.ftl_pushout)"""
        idx = np.asarray(idx, np.int64)
        if not len(idx) or self.grid is None:
            return
        A, meta, hh = self._grid_args()
        ST.ftl_pushout(self.cur['P'], self.cur['SEG'], idx, A, meta, hh, float(self.clearance()))

    def root_normals(self):
        """the skin's normal at each root (the body lattice's gradient; the first segment where there is none)"""
        c = self.cur
        key = (id(self.grid), self.n)
        if getattr(self, '_nrm', None) is not None and self._nrm[0] == key:
            return self._nrm[1]
        R = c['P'][:, 0].astype(np.float64)
        if self.grid is not None:
            _, gr = self.grid.sample(R)
            N = gr / np.maximum(np.linalg.norm(gr, axis=1, keepdims=True), 1e-9)
        else:
            d = c['P'][:, 1] - c['P'][:, 0]
            N = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        N = np.ascontiguousarray(N, np.float64)
        self._nrm = (key, N)
        return N

    def style_dab(self, tool, cam_l, W, H, depth_l, cx, cy, rpx, strength, touched, band=0.004):
        """Puff / Clump / Smooth: one dab (length kept, clear of the form)"""
        c = self.cur
        A, lo, h = self._grid_args()
        from .draw import cam_vec
        C = cam_vec(cam_l, W, H).astype(np.float64)
        dep = np.ascontiguousarray(depth_l, np.float64)
        args = (C, W, H, dep, float(cx), float(cy), float(rpx))
        cand, box = self._cand(C, W, H, cx, cy, rpx)
        if tool == 'puff':
            ST.puff_dab(c['P'], c['SEG'], self.root_normals(), *args, float(strength), float(band), A, lo, h,
                        float(self.clearance()), touched, cand)
        elif tool == 'clump':
            ST.clump_dab(c['P'], c['SEG'], *args, float(strength), float(band), A, lo, h, float(self.clearance()),
                         touched, cand)
        elif tool == 'smooth':
            ST.smooth_dab(c['P'], c['SEG'], *args, float(strength), float(band), A, lo, h, float(self.clearance()),
                          touched, cand)
        self._boxes_after(C, W, H, cand, box)
        self.version += 1

    def cut_dab(self, cam_l, W, H, depth_l, cx, cy, rpx, band=0.002, min_frac=0.04, min_off=0.003):
        """one dab of the scissors: each strand under the brush ends where it first enters it (never longer)"""
        c = self.cur
        n, k = c['P'].shape[:2]
        from .draw import cam_vec
        C = cam_vec(cam_l, W, H).astype(np.float64)
        cut_at = np.full(n, float(k - 1))
        cand, box = self._cand(C, W, H, cx, cy, rpx)
        A, lo, h = self._grid_args()
        ST.cut_dab(c['P'], C, W, H, np.ascontiguousarray(depth_l, np.float64), float(cx), float(cy), float(rpx),
                   float(band), max(1, int(min_frac * k)), cut_at, cand, A, lo, h, float(min_off))
        m = ST.resample_cut(c['P'], c['RAD'], c['COL'], c['SEG'], cut_at)
        idx = np.where(cut_at < k - 1 - 1e-6)[0]
        if m:
            self.pushout(idx)                      # re-spaced points can sit on chords through a thin form (an ear)
            self._boxes_after(C, W, H, cand, box)
            self.version += 1
        return idx

    def trim(self, max_len_m, sel=None):
        self._box_version = getattr(self, '_box_version', 0) + 1
        c = self.cur
        self.begin()
        m = ST.length_cut(c['P'], c['RAD'], c['COL'], c['SEG'], float(max_len_m), sel=sel)
        touched = np.zeros(self.n, np.uint8)
        L = np.nansum(c['SEG'], axis=1)
        touched[np.abs(L - float(max_len_m)) < 1e-6] = 1
        self.commit(touched, 'trim %.0f mm' % (max_len_m * 1000))
        self._log('trim', max_mm=round(max_len_m * 1000, 1), cut=int(m))
        return m

    def settle(self, sel=None, iters=40, set_frac=0.85, mu=0.4, up_local=(0, 1, 0), g_mm=0.6, label='settle'):
        self._box_version = getattr(self, '_box_version', 0) + 1
        """gravity, the fibre's bend back toward its set shape, contact with friction: the hair comes to rest as it
        lies after a comb. set_frac: how much of the current (combed) shape the hair holds (water + drying, or a
        product, set it); the rest is the base's own shape (its natural curl)"""
        c = self.cur
        n = self.n
        sel = np.arange(n) if sel is None else np.asarray(sel, np.int64)
        if not len(sel):
            return
        self.begin(sel)
        REST = c['P'].copy()                    # the shape it holds: its own (combed) segments, carried on its parents
        A, lo, h = self._grid_args()
        g = -np.asarray(up_local, float)
        g = g / max(np.linalg.norm(g), 1e-12) * (g_mm * 1e-3)
        ST.settle(c['P'], c['SEG'], REST, sel, g, int(iters), float(set_frac), A, lo, h, float(self.clearance()),
                  float(mu))
        t = np.zeros(n, np.uint8)
        t[sel] = 1
        self.commit(t, label)
        self._log('settle', strands=int(len(sel)), iters=iters, set_frac=set_frac, mu=mu)

    def take_rows(self, sel):
        """a copy of the strands sel and the version (taken on the UI thread, for a worker)"""
        c = self.cur
        sel = np.asarray(sel, np.int64)
        return (self.version, sel, np.ascontiguousarray(c['P'][sel].copy()), np.ascontiguousarray(c['SEG'][sel].copy()),
                self._grid_args(), float(self.clearance()))

    def settle_copy(self, taken, iters=24, set_frac=0.9, mu=0.4, up_local=(0, 1, 0), g_mm=0.6):
        """the settle on a copy taken by take_rows (on a worker thread): -> (version it started from, sel, P)"""
        v0, sel, P, SEG, (A, lo, h), clear = taken
        g = -np.asarray(up_local, float)
        g = g / max(np.linalg.norm(g), 1e-12) * (g_mm * 1e-3)
        ST.settle(P, SEG, P.copy(), np.arange(len(sel), dtype=np.int64), g, int(iters), float(set_frac), A, lo, h,
                  clear, float(mu))
        return v0, sel, P

    def apply_rows(self, v0, sel, P, label='rest after comb'):
        """put a worker's result in (an undo step of its own) if nothing changed the hair since it started"""
        if self.version != v0:
            return False
        self._box_version = getattr(self, '_box_version', 0) + 1
        self.begin(sel)
        self.cur['P'][sel] = P
        t = np.zeros(self.n, np.uint8)
        t[sel] = 1
        self.commit(t, label)
        self._log('settle', strands=int(len(sel)), worker=True)
        return True

    def wash(self):
        self._box_version = getattr(self, '_box_version', 0) + 1
        """water erases the set: the hair returns to its natural fall (the base's shapes) at its current lengths"""
        if self.base is None:
            return
        c, b = self.cur, self.base
        self.begin()
        Lc = np.nansum(c['SEG'], axis=1)
        Lb = np.nansum(self._b('SEG'), axis=1)
        for k in ('P', 'RAD', 'COL', 'SEG'):
            c[k] = self._b(k)
        cut = Lc < Lb - 1e-6
        if cut.any():
            ST.length_cut(c['P'], c['RAD'], c['COL'], c['SEG'], 1e9)      # no-op shape
            idx = np.where(cut)[0]
            k_ = c['P'].shape[1]
            cut_at = np.full(self.n, float(k_ - 1))
            for s in idx:
                arc = np.concatenate([[0.0], np.cumsum(c['SEG'][s].astype(np.float64))])
                i = int(np.clip(np.searchsorted(arc, Lc[s]) - 1, 0, k_ - 2))
                cut_at[s] = i + (Lc[s] - arc[i]) / max(float(c['SEG'][s, i]), 1e-12)
            ST.resample_cut(c['P'], c['RAD'], c['COL'], c['SEG'], cut_at)
        self.commit(np.ones(self.n, np.uint8), 'wash')
        self._log('wash', kept_cuts=int(cut.sum()))

    # ---- sections and stylist cuts (11-11: the scissor work left a single straight cut, with no way to vary the
    # length within one section) ---------------------------------------------------------------------------------------
    def active(self):
        """the strands the brushes and cuts act on: the section, or every strand"""
        sec = getattr(self, 'section', None)
        if sec is None or len(sec) != self.n:
            return np.arange(self.n, dtype=np.int64)
        return np.nonzero(sec)[0].astype(np.int64)

    def set_section(self, mask=None):
        """mask (n,) bool / uint8, or None for every strand; the rest is pinned out of the way (not cut, combed or
        styled; drawn dimmed or hidden)"""
        self.section = None if mask is None else np.asarray(mask, np.uint8).copy()
        self.section_version = getattr(self, 'section_version', 0) + 1
        self._box_version = getattr(self, '_box_version', 0) + 1
        self.version += 1

    def section_dab(self, cam_l, W, H, depth_l, cx, cy, rpx, add=True, band=0.004):
        """the Section brush: the strands whose ROOT is under the brush (in front of the form) join (or leave) the
        section. -> how many changed"""
        from .draw import cam_vec
        R = self.cur['P'][:, 0].astype(np.float64)
        C = cam_vec(cam_l, W, H).astype(np.float64)
        f, r, u = cam_l.basis()
        q = R - cam_l.pos
        zf = q @ f
        ok = zf > 1e-5
        th = C[12]
        px = (q @ r) / np.maximum(zf, 1e-9)
        py = (q @ u) / np.maximum(zf, 1e-9)
        sx = (px / (th * C[13]) + 1.0) * 0.5 * W - 0.5
        sy = (1.0 - py / th) * 0.5 * H - 0.5
        inb = ok & ((sx - cx) ** 2 + (sy - cy) ** 2 <= rpx * rpx)
        ix = np.clip(np.round(sx).astype(int), 0, W - 1)
        iy = np.clip(np.round(sy).astype(int), 0, H - 1)
        t = np.linalg.norm(q, axis=1)
        inb &= t <= np.asarray(depth_l)[iy, ix] + band
        sec = getattr(self, 'section', None)
        if sec is None or len(sec) != self.n:
            sec = np.zeros(self.n, np.uint8) if add else np.ones(self.n, np.uint8)
        before = sec.copy()
        sec[inb] = 1 if add else 0
        self.section = sec
        changed = int((sec != before).sum())
        if changed:
            self.section_version = getattr(self, 'section_version', 0) + 1
            self._box_version = getattr(self, '_box_version', 0) + 1
            self.version += 1
        return changed

    def cut_to_lengths(self, idx, target_m, label='cut', point_mm=0.0, seed=0):
        """each strand idx shortened to target_m (never lengthened); point_mm: each end shortened by a random 0..point
        (the stylist's point cut: a soft edge instead of a blunt line). -> strands cut"""
        c = self.cur
        n, k = c['P'].shape[:2]
        idx = np.asarray(idx, np.int64)
        tgt = np.asarray(target_m, np.float64).reshape(-1)
        if point_mm > 0:
            tgt = tgt - np.random.default_rng(int(seed) + 977).uniform(0.0, point_mm / 1000.0, len(idx))
        L = np.nansum(c['SEG'][idx].astype(np.float64), axis=1)
        do = tgt < L - 1e-6
        tgt = np.maximum(tgt, 0.002)                                   # 2 mm stays: the scissors do not reach the skin
        cut_at = np.full(n, float(k - 1))
        for s, T in zip(idx[do], tgt[do]):
            arc = np.concatenate([[0.0], np.cumsum(c['SEG'][s].astype(np.float64))])
            i = int(np.clip(np.searchsorted(arc, T) - 1, 0, k - 2))
            cut_at[s] = i + (T - arc[i]) / max(float(c['SEG'][s, i]), 1e-12)
        sel = idx[do]
        if not len(sel):
            return 0
        self._box_version = getattr(self, '_box_version', 0) + 1
        self.begin(sel)
        ST.resample_cut(c['P'], c['RAD'], c['COL'], c['SEG'], cut_at)
        self.pushout(sel)
        t = np.zeros(n, np.uint8)
        t[sel] = 1
        self.commit(t, label)
        return int(len(sel))

    def cut_same_length(self, length_mm, to_mm=None, axis_local=None, point_mm=0.0, seed=0):
        """a stylist's travelling guide: every hair of the section cut to length_mm; with to_mm, the length runs from
        length_mm to to_mm along axis_local (graduation, a fringe longer at the sides, a length front to back)"""
        idx = self.active()
        if not len(idx):
            return 0
        tgt = np.full(len(idx), float(length_mm) / 1000.0)
        if to_mm is not None and axis_local is not None:
            a = np.asarray(axis_local, float)
            a = a / max(np.linalg.norm(a), 1e-12)
            pr = self.cur['P'][idx, 0].astype(np.float64) @ a
            span = max(float(pr.max() - pr.min()), 1e-9)
            w = (pr - pr.min()) / span
            tgt = (float(length_mm) + (float(to_mm) - float(length_mm)) * w) / 1000.0
        m = self.cut_to_lengths(idx, tgt, label='same length %.0f mm' % length_mm, point_mm=point_mm, seed=seed)
        self._log('cut same length', length_mm=length_mm, to_mm=to_mm, point_mm=point_mm, cut=m, section=int(len(idx)))
        return m

    def lifted(self, idx, elev_deg, up_local):
        """the section's hair lifted to an elevation as a stylist holds it before cutting, combed straight: each strand
        laid out from its root along its lift direction (0 deg down the head, 90 straight out from the skin, 180 straight
        up), then kept clear of the form with its lengths kept (so lifted hair wraps the head where it must). ->
        (P lifted (m, k, 3), the lift directions)"""
        c = self.cur
        P0 = c['P'][idx].astype(np.float64)
        SEG = c['SEG'][idx].astype(np.float32)
        R = P0[:, 0]
        N = self.root_normals()[idx]
        up = np.asarray(up_local, float)
        up = up / max(np.linalg.norm(up), 1e-12)

        def slerp(a, b, t):
            a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
            b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
            dd = np.clip((a * b).sum(-1, keepdims=True), -1, 1)
            w = np.arccos(dd)
            sw = np.maximum(np.sin(w), 1e-9)
            out = np.where(w < 1e-6, a, (np.sin((1 - t) * w) * a + np.sin(t * w) * b) / sw)
            return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
        e = np.radians(float(elev_deg))
        down = np.broadcast_to(-up, N.shape)
        upb = np.broadcast_to(up, N.shape)
        d = slerp(down, N, min(e, np.pi / 2) / (np.pi / 2)) if e <= np.pi / 2 else slerp(N, upb, (e - np.pi / 2) / (np.pi / 2))
        arc = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(SEG.astype(np.float64), 1)], 1)
        PL = (R[:, None, :] + d[:, None, :] * arc[..., None]).astype(np.float32)
        A, meta, hh = self._grid_args()
        ST.ftl_pushout(PL, SEG, np.arange(len(idx), dtype=np.int64), A, meta, hh, float(self.clearance()))
        return PL, d

    def cut_lift(self, elev_deg, guide_mm, angle_deg=0.0, point_mm=0.0, up_local=(0, 1, 0), guide=None, seed=0):
        """a stylist's stationary guide: the section lifted to elev_deg (lifted()), cut by the plane through the guide
        hair's point at guide_mm, square to the guide's lifted direction and tilted by angle_deg (an A-line, a face
        frame), then let fall: each hair keeps the length the plane left it. 0 deg: one length; 90: layers; 180: long
        layers (shorter on top, longer under). guide: the guide strand (default the section's central root)."""
        idx = self.active()
        if not len(idx):
            return 0
        PL, d = self.lifted(idx, elev_deg, up_local)
        if float(elev_deg) <= 30.0:
            # low elevation: the hair is cut as it hangs (a one-length line on the hanging hair), not laid straight
            # down from roots on top of the head (that line starts inside the head)
            PL = self.cur['P'][idx].astype(np.float32).copy()
        R = PL[:, 0].astype(np.float64)
        gi = int(np.argmin(np.linalg.norm(R - R.mean(0), axis=1))) if guide is None else int(np.nonzero(idx == guide)[0][0])
        seg = np.linalg.norm(np.diff(PL[gi].astype(np.float64), axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        gm = min(float(guide_mm) / 1000.0, arc[-1])
        j = int(np.clip(np.searchsorted(arc, gm) - 1, 0, len(seg) - 1))
        w = (gm - arc[j]) / max(seg[j], 1e-12)
        Cp = PL[gi, j] + (PL[gi, j + 1] - PL[gi, j]) * w
        # the cutting line is square to the guide's lift direction at its root (level at 0 and 180 deg, square to the
        # skin at 90), not to where the guide ended up after wrapping round the head (that tilted a one-length line)
        nc = np.asarray(d[gi], np.float64).copy()
        nc /= max(np.linalg.norm(nc), 1e-12)
        if abs(float(angle_deg)) > 1e-6:
            up = np.asarray(up_local, float) / max(np.linalg.norm(up_local), 1e-12)
            ax = np.cross(up, nc)
            if np.linalg.norm(ax) < 1e-6:
                Ng = self.root_normals()[idx[gi]]
                ax = np.cross(Ng, up)
            ax /= max(np.linalg.norm(ax), 1e-12)
            a = np.radians(float(angle_deg))
            nc = nc * np.cos(a) + np.cross(ax, nc) * np.sin(a) + ax * (ax @ nc) * (1 - np.cos(a))
        # each lifted hair's first crossing of the plane, as an arc length
        s = (PL.astype(np.float64) - Cp) @ nc                     # signed distance of every point to the plane
        good = np.isfinite(s)
        s = np.where(good, s, -1e9)
        cross = (s[:, 1:] >= 0) & (s[:, :-1] < 0)
        has = cross.any(1)
        jj = np.argmax(cross, 1)
        segl = np.linalg.norm(np.diff(PL.astype(np.float64), axis=1), axis=2)
        arcl = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(segl, 1)], 1)
        r_ = np.arange(len(idx))
        s0, s1 = s[r_, jj], s[r_, jj + 1]
        f = np.where(np.abs(s1 - s0) > 1e-12, -s0 / (s1 - s0), 0.0)
        Lc = arcl[r_, jj] + f * segl[r_, jj]
        m = self.cut_to_lengths(idx[has], Lc[has], label='lift %d deg, %.0f mm' % (elev_deg, guide_mm), point_mm=point_mm,
                                seed=seed)
        self._log('cut lift', elev_deg=elev_deg, guide_mm=guide_mm, angle_deg=angle_deg, point_mm=point_mm, cut=m,
                  section=int(len(idx)))
        return m

    # ---- states ------------------------------------------------------------------------------------------------------------
    def save_state(self, name):
        c, b = self.cur, self.base
        d = np.zeros(self.n, bool)
        for k in ('P', 'SEG'):
            a = np.nan_to_num(c[k]) != np.nan_to_num(self._b(k))
            d |= a.reshape(self.n, -1).any(1)
        idx = np.where(d)[0]
        self.states[name] = dict(_compact(idx, c['P'][idx], c['RAD'][idx], c['COL'][idx], c['SEG'][idx]),
                                 time=time.strftime('%H:%M:%S'))
        self._log('state saved', name=name, changed=int(len(idx)))
        return len(idx)

    def show_state(self, name):
        self._box_version = getattr(self, '_box_version', 0) + 1
        c, b = self.cur, self.base
        st = self.states.get(name)
        self.begin()
        for k in ('P', 'RAD', 'COL', 'SEG'):
            c[k] = self._b(k)
        if st is not None:
            for k in ('P', 'RAD', 'COL'):
                c[k][st['idx']] = st[k]
            c['SEG'][st['idx']] = st['SEG'] if 'SEG' in st else seg_of(c['P'][st['idx']])
        self.commit(np.ones(self.n, np.uint8), 'state %s' % name)
        self._log('state shown', name=name)

    # ---- measures (the live checks) -----------------------------------------------------------------------------------------
    def stats(self):
        c = self.cur
        if c is None:
            return {}
        L = np.nansum(c['SEG'], axis=1) * 1000
        out = {'strands': int(self.n), 'length_mm_p10_50_90': np.percentile(L, [10, 50, 90]).round(1).tolist()}
        if self.grid is not None and self.n:
            sub = c['P'][:: max(1, self.n // 4000), 1:].reshape(-1, 3)      # roots sit in the skin by design
            sub = sub[np.isfinite(sub[:, 0])]
            f, _ = self.grid.sample(sub)
            out['points_inside_form'] = int((f < 0).sum())
            out['clear_mm_p1'] = round(float(np.percentile(f, 1) * 1000), 3)
        return out

    # ---- save / load (arrays in a sidecar npz; the record in json) -------------------------------------------------------
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, 'hair_%d' % self.oid)
        arrs = {}
        for pre, d in (('b_', self.base or {}), ('c_', self.cur or {})):
            for k in ('P', 'RAD', 'COL', 'SEG', 'SIGP', 'SIG') + OPTICS:
                if k in d:
                    arrs[pre + k] = d[k]
        if self.grid is not None:
            arrs.update(g_A=self.grid.A, g_lo=self.grid.lo, g_h=np.array([self.grid.h]))
            if self.grid.fine is not None:
                arrs.update(gf_A=self.grid.fine[2], gf_lo=self.grid.fine[0], gf_h=np.array([self.grid.fine[1]]))
        for nm, st in self.states.items():
            for k in ('idx', 'P', 'RAD', 'COL', 'SEG'):
                if k in st:
                    arrs['s_%s__%s' % (k, nm)] = st[k]
        np.savez(stem + '.npz', **arrs)
        meta = {'oid': self.oid, 'name': self.name, 'settings': self.settings, 'report': self.report,
                'states': list(self.states), 'log': self.log[-500:], 'visible': self.visible}
        with open(stem + '.json', 'w') as fh:
            json.dump(meta, fh, indent=1, default=float)
        return stem

    @staticmethod
    def load(stem):
        meta = json.load(open(stem + '.json'))
        g = HairGroom(meta['oid'], meta.get('name', 'hair'), meta.get('settings'))
        z = np.load(stem + '.npz')
        g.base = {k[2:]: z[k] for k in z.files if k.startswith('b_')}
        g.cur = {k[2:]: z[k].copy() for k in z.files if k.startswith('c_')}
        if 'g_A' in z.files:
            g.grid = ST.BodyGrid(z['g_lo'], float(z['g_h'][0]), z['g_A'])
            if 'gf_A' in z.files:
                g.grid.set_fine(z['gf_lo'], float(z['gf_h'][0]), z['gf_A'])
        for nm in meta.get('states', []):
            g.states[nm] = {k: z['s_%s__%s' % (k, nm)] for k in ('idx', 'P', 'RAD', 'COL', 'SEG')
                            if 's_%s__%s' % (k, nm) in z.files}
        g.report = meta.get('report', {})
        g.log = meta.get('log', [])
        g.visible = meta.get('visible', True)
        return g

    def to_hairset(self, frame=None, kind='scalp'):
        """the current hair as an engine HairSet (for the review renderer); in the object's frame, or world via frame"""
        import sys
        from .generate import ROOT
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from engine.hair.hairset import HairSet
        c = self.cur
        P = c['P'].astype(np.float64)
        if frame is not None:
            P = frame.to_world(P.reshape(-1, 3)).reshape(P.shape)
        P = P.astype(np.float32)
        n, k = P.shape[:2]
        sig = self.base.get('SIG') if self.base else None
        if sig is None:
            sig = np.tile(np.array([[0.4, 0.6, 1.0]], np.float32), (n, 1))
        d0 = P[:, 1] - P[:, 0]
        N = d0 / np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-9)
        a = {'P': P, 'RAD': c['RAD'].astype(np.float32), 'PIG': np.ones((n, k), np.float32),
             'SIG': np.asarray(sig, np.float32), 'RREF': c['RAD'][:, 0].astype(np.float32),
             'ELL': np.full(n, 1.2, np.float32), 'TAU': np.full(n, 40.0, np.float32), 'BM': np.full(n, 0.10, np.float32),
             'NRM': N.astype(np.float32),
             'SEED': (np.arange(1, n + 1, dtype=np.uint64) * 2654435761 % 2 ** 32).astype(np.uint32),
             'KIND': np.zeros(n, np.int32), 'LEN': np.nansum(c['SEG'], axis=1).astype(np.float32)}
        b = self.base or {}
        # the base's own fibre optics (fs-hair 01:5x, 11-12: the export had ELL 1.2 / TAU 40 defaults where her n2c has 1.32 /
        # 69.6): per strand as they are; per point (the pigment along the strand) by arc length from the root, so a cut end
        # shows the shaft the scissors exposed, not the old tip (as the window's colour, resampled on a cut)
        for key in ('ELL', 'TAU', 'BM', 'SEED', 'KIND'):
            if key in b and len(b[key]) == n:
                a[key] = np.asarray(b[key]).astype(a[key].dtype)
        bS = self._b('SEG') if 'P' in b else None
        if bS is not None and bS.shape == c['SEG'].shape:
            for key in ('SIGP', 'PIG'):
                if key in b and b[key].shape[:2] == (n, k):
                    a[key] = by_arc(np.asarray(b[key], np.float32), bS, c['SEG'])
        return HairSet(a, [kind], {'what': 'Field Studio hair (fs-hair) %s' % self.name})
