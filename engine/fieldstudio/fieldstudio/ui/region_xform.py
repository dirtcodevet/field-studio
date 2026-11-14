"""Transform of a masked region (fs-shell-3, 2026-11-14; TOOLSET_DESIGN 3.2, approved: "a region: a masked or selected
part of an object, moved with a soft edge (the shared Smoothing) and carrying its sculpting"; "Sees: the gizmo at the
pivot; the region's soft edge as a gradient tint"). The move itself is fs-tools' tools.region_move (toolset/moves.py):
'unmasked' = everything the Mask does not protect; each point carries its nearest surface point's weight, so a nose
moves with its inside; the soft edge lies inside the region, so the protected part moves 0.0 mm.

In the window, with Move (W) or Rotate (R) and ONE selected object whose mask protects part of it:
- the part that will move is tinted on the model (its soft edge fading to its border), measured off the window thread
  from the field and the mask (job 'shell.region');
- the gizmo sits at the region's pivot (Pivot: the part's centre, the middle of its border = its root, or a picked
  surface point) in the object's axes; handles, X / Y / Z, typed values, Ctrl snapping work as for objects;
- the drag changes nothing in the scene: the part's ghost follows the hand;
- the release (or Enter) posts ONE tools.region_move = one undo step; Esc puts it back (nothing posted).
- the strip: Moves [Unmasked part | Whole object], Soft edge (mm), Pivot.
Scale: fs-tools' region move is rigid (turn + move), so Scale acts on the whole object and the strip says so.
"""
import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QColor, QPen, QPolygonF

NMAX = 3000                     # ghost points drawn (a fixed-seed subsample of the region's surface)
CAP = 64                        # the measuring grid's nodes along the object's longest side
PIVOTS = (('centre', 'Centre of the part'), ('border', 'Its border (root)'), ('surface point', 'A picked point'))
TINT = QColor(250, 214, 80)     # the meaning colours (interface 3.2): yellow = what the tool caught


# ---- the measure (engine thread, under its lock) -----------------------------------------------------------------------
def measure(scene, oid, smoothing, cap=CAP, nmax=NMAX):
    """-> the region of object oid the Mask leaves free: its surface samples (world), their facing, their weight (the
    soft edge: 0 at the border rising to 1 over `smoothing` inward, as toolset/moves.region_grid), its centre and the
    middle of its border; state 'none' (no mask: the whole object moves), 'partial', 'all' (all of it masked)"""
    from ..toolset import common as C
    t0 = time.perf_counter()
    o = scene.get(int(oid))
    st = scene.store
    inv = bool(getattr(o, 'mask_inv', False))
    out = {'oid': int(oid), 'state': 'none', 'inv': inv, 'smoothing': float(smoothing)}
    slots = np.fromiter(o.sculpt.slots.values(), np.int64) if o.sculpt.slots else np.zeros(0, np.int64)
    painted = False
    if st.has('mask') and len(slots):
        for i in range(0, len(slots), 4096):           # chunks: a big object never copies its whole mask at once
            if float(st.mask[slots[i:i + 4096]].max()) > 0.5:
                painted = True
                break
    if not painted and not inv:
        out['ms'] = round(1e3 * (time.perf_counter() - t0), 1)
        return out
    lo, hi = C.world_box(o)
    hmin = o.sculpt.h * C.frame(o)[2]
    S = _samples(scene, o, lo, hi, max(float((hi - lo).max()) / (cap - 1), hmin), inv)
    if S is None:                              # no surface in the box (an empty object)
        out['ms'] = round(1e3 * (time.perf_counter() - t0), 1)
        return out
    free = S[3]
    out['free_frac'] = float(free.mean())      # over the whole object's surface (the coarse pass)
    out['state'] = 'all' if not free.any() else ('none' if free.all() else 'partial')
    if out['state'] == 'partial':
        # a second, finer pass over the free part's box (+ the soft edge): the ghost at the part's own scale
        Pf = S[0][free]
        flo, fhi = Pf.min(0) - smoothing - 3 * S[4], Pf.max(0) + smoothing + 3 * S[4]
        flo, fhi = np.maximum(flo, lo - 2 * S[4]), np.minimum(fhi, hi + 2 * S[4])
        h2 = max(float((fhi - flo).max()) / (cap - 1), hmin)
        if h2 < 0.75 * S[4]:
            S2 = _samples(scene, o, flo, fhi, h2, inv)
            if S2 is not None and S2[3].any() and not S2[3].all():
                S = S2
        _finish(out, S, smoothing, nmax)
    out['ms'] = round(1e3 * (time.perf_counter() - t0), 1)
    return out


def _samples(scene, o, lo, hi, hw, inv):
    """the object's surface on a grid of spacing hw over (lo, hi) (+ 2 hw): points (world), facing, mask, free, hw"""
    from ..toolset import common as C
    from ..warp.bake import _mask_at, object_field
    st = scene.store
    lo, hi = lo - 2 * hw, hi + 2 * hw
    n = np.maximum(np.ceil((hi - lo) / hw).astype(int) + 1, 2)
    ax = [lo[a] + hw * np.arange(n[a]) for a in range(3)]
    Wp = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    F = object_field(scene, o)(Wp).reshape(tuple(n))
    surf = np.zeros(tuple(n), bool)
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        ch = (F[tuple(s0)] < 0) != (F[tuple(s1)] < 0)
        surf[tuple(s0)] |= ch
        surf[tuple(s1)] |= ch
    idx = np.nonzero(surf)
    if not len(idx[0]):
        return None
    G = np.stack(np.gradient(F, hw), -1)[idx]
    gn = np.maximum(np.linalg.norm(G, axis=1), 1e-12)
    N = G / gn[:, None]
    P = Wp.reshape(tuple(n) + (3,))[idx] - (F[idx] / gn)[:, None] * N       # onto the surface (one Newton step)
    Xl = np.ascontiguousarray(C.to_local(o, P))
    m = np.zeros(len(Xl))
    if st.has('mask'):
        pk = scene.pack()
        _mask_at(pk.index_flat, st.kview('mask'), pk.oi_tab, scene.index_of(o.id), o.sculpt.h, Xl, m)
    m = np.clip(m, 0.0, 1.0)
    if inv:
        m = 1.0 - m
    return P, N, m, (1.0 - m) > 0.5, hw


def _finish(out, S, smoothing, nmax):
    """the free part's samples, their soft-edge weight, its centre and the middle of its border, into out"""
    from scipy.spatial import cKDTree
    P, N, _m, free, hw = S
    Pf, Nf = P[free], N[free]
    d, _ = cKDTree(P[~free]).query(Pf)
    if smoothing > 0:
        t = np.clip(d / smoothing, 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
    else:
        w = np.ones(len(Pf))
    border = Pf[d <= 1.75 * hw]
    out['centre'] = Pf.mean(0).tolist()
    out['border'] = (border.mean(0) if len(border) else Pf.mean(0)).tolist()
    out['n_free'] = int(free.sum())
    if len(Pf) > nmax:
        k = np.random.default_rng(0).choice(len(Pf), nmax, replace=False)
        Pf, Nf, w = Pf[k], Nf[k], w[k]
    out['pts'], out['nrm'], out['w'] = Pf, Nf, w
    out['h_mm'] = round(hw * 1e3, 3)


def _region_job(engine, oid, smoothing, sig):
    try:
        info = measure(engine.hist.scene, oid, smoothing)
    except Exception as e:                     # a region that cannot be measured is not offered (and says why once)
        info = {'oid': int(oid), 'state': 'none', 'error': repr(e)}
    engine.region_info = (sig, info)
    return False


def _proj(cam, w, h, P):
    """vectorised render.camera.project for perspective views: -> x, y, z (z <= 0 behind the camera)"""
    f, r, u = cam.basis()
    v = np.asarray(P, float) - cam.pos
    z = v @ f
    th = math.tan(math.radians(cam.fov) * 0.5)
    a = w / float(h)
    if cam.ortho:
        D, _B = cam.ortho_params()
        x = ((v @ r) / (D * th * a) + 1.0) * 0.5 * w - 0.5
        y = (1.0 - (v @ u) / (D * th)) * 0.5 * h - 0.5
        return x, y, np.full(len(v), D)
    zz = np.where(z > 1e-9, z, 1e-9)
    x = ((v @ r) / (zz * th * a) + 1.0) * 0.5 * w - 0.5
    y = (1.0 - (v @ u) / (zz * th)) * 0.5 * h - 0.5
    return x, y, z


# ---- the window side -----------------------------------------------------------------------------------------------------
class RegionXform:
    """the window's region transform: what the Mask leaves free, offered to Move / Rotate; one per window"""

    def __init__(self, win):
        self.win = win
        self.on = True                         # Moves: the unmasked part (True) / the whole object
        self.smoothing = 0.004                 # the soft edge (m): the registry's shared Smoothing for Transform
        self.pivot = 'centre'                  # the registry's Pivot ('centre' | 'border' | 'surface point')
        self.info = None
        self.sig = None
        self.pending = None                    # after a release: (motion, the log length then) until the scene changes
        self.last = None                       # the last region move posted (oid, R, t, pivot, smoothing)
        self._t = QTimer(win)
        self._t.timeout.connect(self.sync)
        self._t.start(150)

    # -- what is offered
    def _oid(self):
        vp = self.win.viewport
        sel = list(getattr(vp, 'selected', None) or [])
        return int(sel[0]) if len(sel) == 1 else None

    def _ver(self):
        snap = getattr(self.win, 'snap', None) or {}
        return (len(snap.get('log') or ()), len(snap.get('redo_log') or ()), snap.get('version'))

    def wanted(self):
        vp = self.win.viewport
        return getattr(vp, 'tool', None) in ('move', 'rotate', 'scale') and self._oid() is not None

    def sync(self):
        vp = self.win.viewport
        if self.pending is not None and self._ver()[:2] != self.pending[1]:
            self.pending = None                # the move landed (or was refused): the scene says so
            vp.update()
        if not self.wanted():
            return
        oid = self._oid()
        sig = (oid, self._ver(), round(self.smoothing, 6))
        if sig != self.sig:
            self.sig = sig
            self.win.engine.post('shell.region', oid, self.smoothing, sig)
        ri = getattr(self.win.engine, 'region_info', None)
        if ri is not None and ri[0] == sig and self.info is not ri[1]:
            self.info = ri[1]
            vp.update()

    def state(self):
        """'partial' / 'all' / 'none' for the selected object (None until measured)"""
        i = self.info
        if i is None or i.get('oid') != self._oid():
            return None
        return i.get('state')

    def masked(self):
        return self.state() in ('partial', 'all')

    def active(self, oid=None):
        """the gizmo moves the region (not the object): Move / Rotate, Moves = the unmasked part, part of it masked"""
        vp = self.win.viewport
        i = self.info
        return (self.on and getattr(vp, 'tool', None) in ('move', 'rotate') and i is not None
                and i.get('state') == 'partial' and i.get('oid') == (self._oid() if oid is None else oid)
                and self._oid() is not None)

    def pivot_point(self):
        g = getattr(self.win.viewport, 'gizmo', None)
        if self.pivot == 'surface point' and g is not None and g.pivot_mode == 'point' and g.pivot_point is not None:
            return np.asarray(g.pivot_point, float)
        i = self.info or {}
        return np.asarray(i.get('border' if self.pivot == 'border' else 'centre'), float)

    def ctx(self, oid=None):
        """-> {'oid', 'pivot'} for the gizmo while the region is what moves, else None"""
        if not self.active(oid):
            return None
        return {'oid': self.info['oid'], 'pivot': self.pivot_point()}

    # -- the drag's motion (gizmo.job hands its object-style job here)
    @staticmethod
    def motion(x, j):
        """the gizmo's ('props', oid, {'transform': ...}) for an object at the region's pivot -> (R, t, pivot): the
        region's points go p -> R (p - pivot) + pivot + t"""
        from ..core.ops import rot_matrix
        t0 = x['t0']
        c = np.asarray(t0['translate'], float)
        R0 = rot_matrix(t0['rotate'])
        tr = j[2].get('transform', {}) if j[0] == 'props' else {}
        R1 = rot_matrix(tr['rotate']) if 'rotate' in tr else R0
        Rr = R1 @ R0.T
        c1 = np.asarray(tr.get('translate', c), float)
        # the gizmo rounds a position to 1 um; the move is taken to the same 1 um (an X move has no Y / Z part)
        return Rr, np.round(c1 - c, 6) + 0.0, c

    def commit(self, x):
        """the release (or Enter): ONE tools.region_move with the drag's motion = one undo step"""
        m = x.get('region_motion')
        if m is None:
            return False
        R, t, c = m
        if float(np.abs(R - np.eye(3)).max()) < 1e-9 and float(np.abs(t).max()) < 1e-9:
            return False                       # no motion: nothing to post
        oid = x['region']['oid']
        self.win.engine.post('tools.region_move', int(oid), R.tolist(), [float(v) for v in t], [float(v) for v in c],
                             float(self.smoothing), 'unmasked')
        self.last = (int(oid), R.tolist(), [float(v) for v in t], [float(v) for v in c], float(self.smoothing))
        self.pending = ((R, t, c), self._ver()[:2])
        try:
            self.win.status.showMessage('Moving the unmasked part (soft edge %.1f mm) ... one undo step'
                                        % (self.smoothing * 1e3), 8000)
        except Exception:
            pass
        return True

    # -- the tint / ghost
    def draw(self, p):
        vp = self.win.viewport
        i = self.info
        if not self.active() or i is None or 'pts' not in i:
            return
        P, N, w = i['pts'], i['nrm'], i['w']
        x = vp.xform
        mot = (x.get('region_motion') if (x is not None and x.get('region') is not None) else None)
        moving = mot is not None
        if mot is None and self.pending is not None:
            mot = self.pending[0]
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        cam = vp.cam
        if mot is not None:
            R, t, c = mot
            Q = P + w[:, None] * (((P - c) @ R.T + c + t) - P)
            Nq = N @ R.T
        else:
            Q, Nq = P, N
        xs, ys, zs = _proj(cam, W, H, Q)
        if cam.ortho:
            facing = (Nq @ (-cam.basis()[0])) > 0.0
        else:
            facing = np.einsum('ij,ij->i', Nq, cam.pos[None, :] - Q) > 0.0
        ok = facing & (zs > 1e-6) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        fr = getattr(vp, 'frame', None)
        if mot is None and fr is not None and fr.get('depth') is not None and not cam.ortho:
            # at rest: only what the frame shows (its depth along each pixel's ray)
            d = fr['depth']
            dh, dw = d.shape[:2]
            ix = np.clip((xs * dw / W).astype(int), 0, dw - 1)
            iy = np.clip((ys * dh / H).astype(int), 0, dh - 1)
            dist = np.linalg.norm(Q - cam.pos[None, :], axis=1)
            dep = np.asarray(d, float)[iy, ix]
            ok &= dist <= dep + max(0.002, 3e-3 * float(np.median(dist)) if len(dist) else 0.002)
        for lo_, hi_, a in ((0.0, 0.34, 70), (0.34, 0.67, 130), (0.67, 1.01, 220)):
            sel = ok & (w >= lo_) & (w < hi_)
            if not sel.any():
                continue
            col = QColor(TINT)
            col.setAlpha(a if not moving else min(255, a + 30))
            p.setPen(QPen(col, 2.6 if moving else 2.0))
            p.drawPoints(QPolygonF([QPointF(float(a_), float(b_)) for a_, b_ in zip(xs[sel], ys[sel])]))


def install(win):
    """the 'shell.region' job on the engine, the window's RegionXform (win.region_xform)"""
    try:
        win.engine.register_job('shell.region', _region_job)
    except ValueError:
        pass
    win.region_xform = RegionXform(win)
    return win.region_xform
