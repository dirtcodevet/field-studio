"""Building warps: the motion (anchored translation, a rigid or affine motion, a path), the support (ball, lens, a
sampled region), the options (Keep volume, Stop at contact), and the swept box outside which nothing moves.

    w = anchored(support, D)                         Shape: the pressed point follows the hand by D (world)
    w = affine(support, A, tau, pivot)               Transform of a region: generator A (x - pivot) + tau, time 1
    w = rigid(support, R, t, pivot)                  the same from a rotation matrix and a translation
    w = along_path(support, points)                  Draw out: the support rides along a path
    w.inverse(P) / w.forward(P) / w.jacobian(P)

Every warp is a pure function of its spec (the press-time field is read once, when it is built), so an anchored stroke
re-built for the hand's current position gives the result that depends only on where the hand is now ("drag back and
it's undone; a wiggle leaves nothing").
"""
import math

import numpy as np
from scipy.linalg import expm

from . import flow as FL
from .profile import FALLOFF, PEAK_SHAPE


class Support:
    """Where the tool acts and with what weight. metric 'ball' | 'lens' | 'grid'."""

    def __init__(self, metric, c0=(0, 0, 0), radius=0.01, hardness=0.0, smoothing=0.0, falloff='soft',
                 family=PEAK_SHAPE, lens=None, reach='none', grid=None, grid_origin=None, grid_h=None, eps_min=None):
        self.metric = metric
        self.c0 = np.asarray(c0, float)
        self.r = float(radius)
        self.hard = float(hardness)
        self.smooth = float(smoothing)
        self.fall = FALLOFF[falloff] if isinstance(falloff, str) else int(falloff)
        self.family = int(family)
        self.lens = lens                         # warp/lens.depth_images(...) for metric 'lens' with a Reach gate
        self.reach = reach if metric == 'lens' else 'none'
        self.grid = grid
        self.grid_origin = None if grid_origin is None else np.asarray(grid_origin, float)
        self.grid_h = grid_h
        self.eps_min = eps_min
        self.front_pad = None                    # the band kept in front of a lens's caught surface (set by the bake)

    @property
    def reach_r(self):
        return self.r + self.smooth

    def box(self):
        """the support's box at the press (world): lo, hi"""
        R = self.reach_r
        if self.metric == 'ball':
            return self.c0 - R, self.c0 + R
        if self.metric == 'grid':
            G = self.grid
            nz = np.argwhere(G > 0)
            if not len(nz):
                return self.grid_origin, self.grid_origin
            lo = self.grid_origin + (nz.min(0) - 1) * self.grid_h
            hi = self.grid_origin + (nz.max(0) + 1) * self.grid_h
            return lo, hi
        L = self.lens
        n, e1, e2 = L['n'], L['e1'], L['e2']
        img = L['img']
        if self.reach == 'through':
            z0, z1 = -L['half'] * 2.0, float(np.nanmax(np.where(np.isfinite(L['zfar']), L['zfar'], -np.inf))) + R
            if not np.isfinite(z1):
                z1 = 4.0 * R
        else:
            # in front of the caught surface the air moves with it: the band in front (front_pad) is what the new
            # surface's field needs; the swept box adds the motion itself
            z0 = float(img[0].min()) - (2.0 * R if self.front_pad is None else self.front_pad)
            z1 = float(img[1].max())
        corners = []
        for a in (-R, R):
            for b in (-R, R):
                for z in (z0, z1):
                    corners.append(self.c0 + a * e1 + b * e2 + z * n)
        c = np.array(corners)
        return c.min(0), c.max(0)

    def lipschitz(self):
        """an upper bound of |grad W| (per metre): the profile's steepest slope, the depth gate's"""
        m = (0.5 * math.pi * (1 - self.hard) + self.hard) / max(self.r, 1e-9)
        if self.metric == 'grid':
            G = self.grid
            g = np.gradient(G, self.grid_h)
            m = float(np.sqrt(sum(x * x for x in g)).max()) if G.size else 0.0
        elif self.metric == 'lens' and self.reach in ('near', 'whole'):
            eps = self.lens['eps']
            m = m + 1.5 / eps + 1.5 * self._img_slope()
        return m

    def _img_slope(self):
        img = self.lens['img']
        pix = self.lens['pix']
        gy, gx = np.gradient(img[0], pix)
        return float(np.sqrt(gx * gx + gy * gy).max()) / max(self.lens['eps'], 1e-9)


class Warp:
    """A built warp: its parameter block, stage rows, images, grids and the swept box."""

    def __init__(self, fp, st, img, G, CG, box, spec=None):
        self.fp = fp
        self.st = st
        self.img = img
        self.G = G
        self.CG = CG
        self.box = box
        self.spec = spec or {}

    @property
    def steps(self):
        return int(self.fp[FL.FP_NSTEP])

    def _map(self, P, inverse):
        P = np.ascontiguousarray(np.asarray(P, float).reshape(-1, 3))
        out = np.empty_like(P)
        FL.map_points(self.fp, self.img, self.G, self.CG, self.st, P, inverse, self.box, out)
        return out

    def forward(self, P):
        return self._map(P, False)

    def inverse(self, P):
        return self._map(P, True)

    def jacobian(self, P, eps=None, inverse=False):
        P = np.ascontiguousarray(np.asarray(P, float).reshape(-1, 3))
        eps = 1e-3 * self.fp[FL.FP_R] if eps is None else eps
        out = np.empty(len(P))
        FL.jacobian_det(self.fp, self.img, self.G, self.CG, self.st, P, eps, inverse, out)
        return out


def _fp(sup, keep_volume, contact, nsteps):
    fp = np.zeros(FL.FP_N_)
    fp[FL.FP_METRIC] = FL.METRIC[sup.metric]
    fp[FL.FP_R] = sup.r
    fp[FL.FP_HARD] = sup.hard
    fp[FL.FP_SMOOTH] = sup.smooth
    fp[FL.FP_FALL] = sup.fall
    fp[FL.FP_FAMILY] = sup.family
    fp[FL.FP_C0:FL.FP_C0 + 3] = sup.c0
    fp[FL.FP_REACH] = FL.REACH[sup.reach]
    img = np.zeros((2, 2, 2))
    if sup.metric == 'lens':
        L = sup.lens
        fp[FL.FP_N:FL.FP_N + 3] = L['n']
        fp[FL.FP_E1:FL.FP_E1 + 3] = L['e1']
        fp[FL.FP_E2:FL.FP_E2 + 3] = L['e2']
        fp[FL.FP_IMG_PIX] = L['pix']
        fp[FL.FP_IMG_W] = L['res']
        fp[FL.FP_IMG_H] = L['res']
        fp[FL.FP_IMG_U0] = L['u0']
        fp[FL.FP_IMG_V0] = L['v0']
        fp[FL.FP_EPSMIN] = L['eps'] if sup.eps_min is None else sup.eps_min
        img = L['img']
    G = np.zeros((2, 2, 2))
    if sup.metric == 'grid':
        G = np.ascontiguousarray(sup.grid, np.float64)
        fp[FL.FP_G_O:FL.FP_G_O + 3] = sup.grid_origin
        fp[FL.FP_G_H] = sup.grid_h
    fp[FL.FP_KEEPVOL] = 1.0 if keep_volume else 0.0
    fp[FL.FP_GRADEPS] = max(1e-3 * sup.r, 1e-6) if sup.metric != 'grid' else 0.25 * sup.grid_h
    CG = np.zeros((2, 2, 2))
    if contact is not None:
        fp[FL.FP_CONTACT] = 1.0
        CG = np.ascontiguousarray(contact['grid'], np.float64)
        fp[FL.FP_C_O:FL.FP_C_O + 3] = contact['origin']
        fp[FL.FP_C_H] = contact['h']
        fp[FL.FP_CEPS] = contact.get('eps', 2.0 * contact['h'])
        fp[FL.FP_CGAP] = contact.get('gap', 0.0)
    fp[FL.FP_NSTEP] = nsteps
    return fp, np.ascontiguousarray(img, np.float64), G, CG


def _steps(sup, speed, extra=0.0, lo=8, hi=256, h_target=2.0):
    """RK4 steps from a bound of the velocity's Lipschitz constant (h_target per step; measured 11-14, fs-tools t1/w3:
    a lens pull of 0.6 x its radius is within 1.1 um of a 600-step reference at 8 steps, 3.3 nm at 64)"""
    Lv = speed * sup.lipschitz() + extra
    return int(min(max(math.ceil(Lv / h_target), lo), hi))


def _swept_box(sup, Ms):
    """the union of the support's box carried by each stage's motion (Ms: 4x4 forward motions)"""
    lo, hi = sup.box()
    c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts = np.concatenate([(c @ M.T)[:, :3] for M in Ms])
    pad = 1e-4 + 0.0 * sup.reach_r
    return np.concatenate([pts.min(0) - pad, pts.max(0) + pad])


def _stage_rows(Ms, gens, axial, ts=None):
    st = np.zeros((len(Ms), FL.ST_N))
    if ts is not None:
        st[:, FL.ST_T] = ts
    for s, (M, (A, tau, p), ax) in enumerate(zip(Ms, gens, axial)):
        Mi = np.linalg.inv(M)
        st[s, FL.ST_MINV:FL.ST_MINV + 12] = Mi[:3, :4].ravel()
        st[s, FL.ST_A:FL.ST_A + 9] = np.asarray(A, float).ravel()
        st[s, FL.ST_TAU:FL.ST_TAU + 3] = tau
        st[s, FL.ST_P:FL.ST_P + 3] = p
        st[s, FL.ST_AX] = ax
    return st


def _per_step(N, fn):
    """3 rows per RK step k: fn(t, k) at t_k, t_k + h/2, t_k + h, each with step k's generator"""
    Ms, gens, axial = [], [], []
    for k in range(N):
        for f in (0.0, 0.5, 1.0):
            M, g, ax = fn((k + f) / N, k)
            Ms.append(M)
            gens.append(g)
            axial.append(ax)
    return Ms, gens, axial


def _contact_extra(contact, speed, travel=False):
    """the contact gate's own rate: a point slows to rest over eps, so a step must resolve eps (fs-tools t1/f1: at ~1.5
    steps per eps the flap stopped 1.07 mm short)"""
    if contact is None or travel:
        return 0.0
    return 6.0 * speed / max(contact.get('eps', 2.0 * contact['h']), 1e-6)


def anchored(sup, D, keep_volume=False, contact=None, steps=None):
    """Shape: the support translates with the hand; the pressed point (weight 1) moves exactly by D."""
    D = np.asarray(D, float)
    nax = sup.lens['n'] if (sup.metric == 'lens' and sup.lens is not None) else np.zeros(3)
    N = steps or _steps(sup, float(np.linalg.norm(D)), extra=_contact_extra(
        contact, float(np.linalg.norm(D)), sup.metric == 'lens' and sup.lens is not None and 'travel' in sup.lens))

    def fn(t, k):
        M = np.eye(4)
        M[:3, 3] = t * D
        return M, (np.zeros((3, 3)), D, sup.c0 + t * D), float(t * (D @ nax))
    Ms, gens, axial = _per_step(N, fn)
    fp, img, G, CG = _fp(sup, keep_volume, contact, N)
    ts = np.array([(k + f) / N for k in range(N) for f in (0.0, 0.5, 1.0)])
    if sup.metric == 'lens' and sup.lens is not None and 'travel' in sup.lens:
        # Whole part with Stop at contact: each column's free travel (img[2]); no point gate
        fp[FL.FP_TRAVEL] = 1.0
        img = np.ascontiguousarray(np.concatenate([img[:2], sup.lens['travel'][None]]), np.float64)
    return Warp(fp, _stage_rows(Ms, gens, axial, ts), img, G, CG, _swept_box(sup, Ms),
                {'kind': 'anchored', 'D': D.tolist()})


def affine(sup, A, tau, pivot, keep_volume=False, contact=None, steps=None):
    """A region moved by the affine motion exp(X), X = [[A, tau - A pivot], [0, 0]]: its weight-1 part exactly so."""
    A = np.asarray(A, float)
    tau = np.asarray(tau, float)
    p = np.asarray(pivot, float)
    X = np.zeros((4, 4))
    X[:3, :3] = A
    X[:3, 3] = tau - A @ p
    lo, hi = sup.box()
    ext = float(np.linalg.norm(hi - lo))
    speed = float(np.linalg.norm(tau)) + float(np.linalg.norm(A, 2)) * (ext + float(np.linalg.norm(p - 0.5 * (lo + hi))))
    N = steps or _steps(sup, speed, extra=float(np.linalg.norm(A, 2)) + _contact_extra(contact, speed))
    Ms, gens, axial = _per_step(N, lambda t, k: (expm(t * X), (A, tau, p), 0.0))
    fp, img, G, CG = _fp(sup, keep_volume, contact, N)
    return Warp(fp, _stage_rows(Ms, gens, axial), img, G, CG, _swept_box(sup, Ms),
                {'kind': 'affine', 'A': A.tolist(), 'tau': tau.tolist(), 'pivot': p.tolist()})


def rigid(sup, R, t, pivot, keep_volume=False, contact=None, steps=None):
    """a region rotated by R about pivot and moved by t (world), as the screw motion's flow: weight 1 = exactly x ->
    R (x - pivot) + pivot + t"""
    from scipy.linalg import logm
    p = np.asarray(pivot, float)
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, float)
    M[:3, 3] = p + np.asarray(t, float) - np.asarray(R, float) @ p
    X = np.real(logm(M))
    A = X[:3, :3]
    A = 0.5 * (A - A.T)                     # a rotation's log is skew (numerically cleaned)
    tau = X[:3, 3] + A @ p                  # generator form A (x - p) + tau  <=>  A x + (tau - A p)
    return affine(sup, A, tau, p, keep_volume=keep_volume, contact=contact, steps=steps)


def along_path(sup, points, keep_volume=False, contact=None, sub=None):
    """Draw out: the support rides along a polyline from points[0] (the press) to points[-1]; each segment is a
    translation of the support at the segment's own velocity (a path flow: the result is the path's, not only its end's)"""
    P = np.asarray(points, float).reshape(-1, 3)
    segs = np.diff(P, axis=0)
    nseg = len(segs)
    if nseg == 0:
        return anchored(sup, np.zeros(3), keep_volume, contact, steps=4)
    Lw = sup.lipschitz()
    nax = sup.lens['n'] if sup.metric == 'lens' else np.zeros(3)
    sub_n = [sub or int(min(max(math.ceil(float(np.linalg.norm(s_)) * Lw / 0.2), 1), 64)) for s_ in segs]
    N = int(sum(sub_n))
    starts = np.concatenate([[0], np.cumsum(sub_n)])
    offs = np.concatenate([np.zeros((1, 3)), np.cumsum(segs, axis=0)])

    def fn(t, k):
        seg = int(np.searchsorted(starts, k, side='right') - 1)
        f = (t * N - starts[seg]) / sub_n[seg]
        d = offs[seg] + f * segs[seg]
        M = np.eye(4)
        M[:3, 3] = d
        v = segs[seg] * (N / sub_n[seg])
        return M, (np.zeros((3, 3)), v, sup.c0 + d), float(d @ nax)
    Ms, gens, axial = _per_step(N, fn)
    fp, img, G, CG = _fp(sup, keep_volume, contact, N)
    return Warp(fp, _stage_rows(Ms, gens, axial), img, G, CG, _swept_box(sup, Ms),
                {'kind': 'path', 'points': P.tolist()})
