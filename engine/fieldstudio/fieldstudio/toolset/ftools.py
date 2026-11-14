"""The approved brush tools' rules and their ONE stroke engine, CPU reference (fs-tools, 11-14; design:
fs_build/fs-tools/BRUSH_ENGINE.md; TOOLSET_DESIGN 1.3-1.7 and 3.3-3.4, approved 11-14).

The engine does everything the tools share; each tool is one rule at a point:
- TIMING: Strength is a RATE for held tools: the stroke's timed samples (cursor point + its time since the press) are
  integrated in fixed quanta of QUANTUM s, still or moving, so the same path at the same pace gives the same result at
  any mouse rate or frame rate (TOOLSET 1.6);
- THE FORM AT THE PRESS: the field on a dense local grid over the stroke's reach, read once; the sheet under the press
  (Near side) labelled from it: surface nodes flood-filled from the pressed one within the reach, then every node takes
  the label of its nearest surface node (an exact distance transform), so a tool never reaches across a gap or the back
  of a thin part;
- DIRECTION: follows (each point along its own facing) or flat (one direction, the facing at the press);
- STOP AT CONTACT: F >= gap - D_contact (a max with the other objects' field): each point stops where it touches, the
  rest goes on; never fuses; no corrugation;
- THE PROFILE: warp/profile.py (Radius, Hardness = the peak, Smoothing = the foot, the falloff choice);
- THE UPDATE: the grid's change written into the object's lattice (bricks it changed), clamped to fs-core's band, one
  StrokeRecord (exact undo, the undo journal).
The rules (w = profile x the gate; dt = QUANTUM; a = rate x dt):
    swell / settle (follow)  F -= +-a w                         (each point moves along its own facing)
    swell / settle (flat)    F(x) <- F(x -+ a w n_press)        (a held flow: a mound off the surface, one direction)
    flatten                  F += (P - F) (1 - exp(-rate dt w)), P = (x - p) . n, the plane through the point under the
                             cursor facing the surface there, read again each quantum (it always follows the cursor)
    smooth                   F += (1 - exp(-rate dt w)) (d - blur3(d)), d = blur(F) - F (a band-pass: bumps go, the form
                             stays, held for seconds)
The card runs the same rules through fs-core's batch path (stage 2 proposal, 11-14); this module is
the reference they are checked against.
"""
import math

import numpy as np
from numba import njit, prange
from scipy import ndimage

from ..core import brush as BR
from ..core.history import Command, command
from ..warp.profile import weight as _w, FALLOFF, PEAK_SHAPE, PEAK_SWELL
from ..warp.bake import object_field, RecUndo, W_BAND
from . import common as C

B = C.B
QUANTUM = 1.0 / 240.0          # s: the rate's time step (TOOLSET 1.6: counted in time, never per event or frame)
SMOOTH_KAPPA = 0.373           # Smooth: the share removed in 1 s = 1 - exp(-KAPPA x rule rate) on a bump of width r


# ---- the rules on the dense grid (local units) ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _swell_follow(G, gate, Dc, use_c, gap, o, h, c, r, hard, sm, fall, fam, a, Q):
    """each point along its own facing; its weight read at its nearest press-time surface point Q (the dome measured
    along the surface, so a rising surface keeps its weight: 2 s rises twice 1 s)"""
    nx, ny, nz = G.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                g = gate[i, j, k]
                if g <= 0.0:
                    continue
                d = math.sqrt((Q[i, j, k, 0] - c[0]) ** 2 + (Q[i, j, k, 1] - c[1]) ** 2 + (Q[i, j, k, 2] - c[2]) ** 2)
                w = _w(d, r, hard, sm, fall, fam) * g
                if w <= 0.0:
                    continue
                v = G[i, j, k] - a * w
                if use_c:
                    v = max(v, gap - Dc[i, j, k])
                G[i, j, k] = v


@njit(cache=True)
def _tri(G, fx, fy, fz):
    nx, ny, nz = G.shape
    fx = min(max(fx, 0.0), nx - 1.000001)
    fy = min(max(fy, 0.0), ny - 1.000001)
    fz = min(max(fz, 0.0), nz - 1.000001)
    i = int(fx)
    j = int(fy)
    k = int(fz)
    tx = fx - i
    ty = fy - j
    tz = fz - k
    a = G[i, j, k] + (G[i + 1, j, k] - G[i, j, k]) * tx
    b = G[i, j + 1, k] + (G[i + 1, j + 1, k] - G[i, j + 1, k]) * tx
    cc = G[i, j, k + 1] + (G[i + 1, j, k + 1] - G[i, j, k + 1]) * tx
    d = G[i, j + 1, k + 1] + (G[i + 1, j + 1, k + 1] - G[i, j + 1, k + 1]) * tx
    e = a + (b - a) * ty
    f = cc + (d - cc) * ty
    return e + (f - e) * tz


@njit(cache=True, parallel=True)
def _swell_flat(G, Gout, gate, Dc, use_c, gap, o, h, c, n, r, hard, sm, fall, fam, a):
    """a held flow along n: F(x) <- F(x - a w(x) n), w the profile of the distance ACROSS n (a lens) x the gate"""
    nx, ny, nz = G.shape
    for i in prange(nx):
        x = o[0] + i * h
        for j in range(ny):
            y = o[1] + j * h
            for k in range(nz):
                Gout[i, j, k] = G[i, j, k]
                g = gate[i, j, k]
                if g <= 0.0:
                    continue
                z = o[2] + k * h
                ex = x - c[0]
                ey = y - c[1]
                ez = z - c[2]
                t = ex * n[0] + ey * n[1] + ez * n[2]
                lx = ex - t * n[0]
                ly = ey - t * n[1]
                lz = ez - t * n[2]
                w = _w(math.sqrt(lx * lx + ly * ly + lz * lz), r, hard, sm, fall, fam) * g
                if w <= 0.0:
                    continue
                s = a * w / h
                v = _tri(G, i - s * n[0], j - s * n[1], k - s * n[2])
                if use_c:
                    v = max(v, gap - Dc[i, j, k])
                Gout[i, j, k] = v


@njit(cache=True, parallel=True)
def _flatten(G, gate, o, h, p, n, r, hard, sm, fall, fam, rdt):
    nx, ny, nz = G.shape
    for i in prange(nx):
        x = o[0] + i * h
        for j in range(ny):
            y = o[1] + j * h
            for k in range(nz):
                g = gate[i, j, k]
                if g <= 0.0:
                    continue
                z = o[2] + k * h
                ex = x - p[0]
                ey = y - p[1]
                ez = z - p[2]
                P = ex * n[0] + ey * n[1] + ez * n[2]
                lx = ex - P * n[0]
                ly = ey - P * n[1]
                lz = ez - P * n[2]
                w = _w(math.sqrt(lx * lx + ly * ly + lz * lz), r, hard, sm, fall, fam) * g
                if w <= 0.0:
                    continue
                G[i, j, k] += (P - G[i, j, k]) * (1.0 - math.exp(-rdt * w))


@njit(cache=True, parallel=True)
def _stamp(G, gate, o, h, q, n, t, r, depth, tip, ridge, k):
    """the Pencil tip at one point: its shape at surface point q (facing n, along t), carving (max with -tip) or raising
    (min with tip), its edge blended over k (the shared Smoothing); only on the sheet under the press (the gate)"""
    nx, ny, nz = G.shape
    v = np.empty(3)
    v[0] = n[1] * t[2] - n[2] * t[1]
    v[1] = n[2] * t[0] - n[0] * t[2]
    v[2] = n[0] * t[1] - n[1] * t[0]
    reach = r + depth + k + 2 * h
    i0 = max(int((q[0] - reach - o[0]) / h), 0)
    i1 = min(int((q[0] + reach - o[0]) / h) + 2, nx)
    j0 = max(int((q[1] - reach - o[1]) / h), 0)
    j1 = min(int((q[1] + reach - o[1]) / h) + 2, ny)
    k0 = max(int((q[2] - reach - o[2]) / h), 0)
    k1 = min(int((q[2] + reach - o[2]) / h) + 2, nz)
    for i in prange(i0, i1):
        for j in range(j0, j1):
            for kk in range(k0, k1):
                if gate[i, j, kk] <= 0.0:
                    continue
                ex = o[0] + i * h - q[0]
                ey = o[1] + j * h - q[1]
                ez = o[2] + kk * h - q[2]
                a = ex * t[0] + ey * t[1] + ez * t[2]          # along the stroke
                b = ex * v[0] + ey * v[1] + ez * v[2]          # across it
                w = ex * n[0] + ey * n[1] + ez * n[2]          # along the facing (0 = the surface)
                if ridge:
                    w = -w                                     # a ridge: the tip turned up out of the surface
                # the tip's field, its deepest point at depth below the surface (w = -depth)
                if tip == 0:                                   # point: a cone, half-width r at the surface
                    rho = math.sqrt(a * a + b * b)
                    ww = w + depth
                    L = math.sqrt(r * r + depth * depth)
                    ca = depth / L
                    sa = r / L
                    if ww * ca + rho * sa < 0.0:
                        T = math.sqrt(rho * rho + ww * ww)
                    else:
                        T = rho * ca - ww * sa
                elif tip == 1:                                 # round: a ball whose bottom is at the depth
                    cz = r - depth
                    T = math.sqrt(a * a + b * b + (w - cz) ** 2) - r
                else:                                          # square: a flat chisel, half-width r, flat bottom
                    qx = abs(b) - r
                    qy = abs(a) - r
                    qz = abs(w - (r - depth)) - r
                    mx = max(qx, 0.0)
                    my = max(qy, 0.0)
                    mz = max(qz, 0.0)
                    T = math.sqrt(mx * mx + my * my + mz * mz) + min(max(qx, max(qy, qz)), 0.0)
                f = G[i, j, kk]
                if not ridge:
                    c = -T                                     # carve: f = smooth max(f, -T)
                    if k > 0.0:
                        hh = max(k - abs(f - c), 0.0) / k
                        G[i, j, kk] = max(f, c) + hh * hh * k * 0.25
                    else:
                        G[i, j, kk] = max(f, c)
                else:
                    if k > 0.0:                                # ridge: f = smooth min(f, T)
                        hh = max(k - abs(f - T), 0.0) / k
                        G[i, j, kk] = min(f, T) - hh * hh * k * 0.25
                    else:
                        G[i, j, kk] = min(f, T)


def thickness_under(F0, o, h, q, n, tmax):
    """the part's thickness under surface point q, inward along -n (for Pencil's depth limit: half of it)"""
    t = 0.5 * h
    while t < tmax:
        if _tri(F0, *((q - t * n - o) / h)) > 0:
            return t
        t += max(abs(_tri(F0, *((q - t * n - o) / h))), 0.25 * h)
    return tmax


# ---- the engine ------------------------------------------------------------------------------------------------------
TOOLS = {'swell': ('held', 'follow'), 'settle': ('held', 'follow'), 'flatten': ('held', 'flat'),
         'smooth': ('held', 'follow'), 'pencil': ('line', 'follow'), 'mask': ('held', 'follow'),
         'blend': ('held', 'follow')}
TIPS = {'point': 0, 'round': 1, 'square': 2}


class _Grid:
    """the dense local grid over the stroke's reach (lattice-aligned nodes, brick-aligned box)"""

    def __init__(self, scene, o, lo, hi):
        h = o.sculpt.h
        self.h = h
        self.i0 = (np.floor(lo / h / B).astype(np.int64) - 1) * B
        self.i1 = (np.floor(hi / h / B).astype(np.int64) + 2) * B - 1
        self.shape = tuple(int(v) for v in self.i1 - self.i0 + 1)
        if int(np.prod(self.shape)) > 24_000_000:
            raise ValueError('the stroke reaches %s nodes: too large for the CPU reference' % (self.shape,))
        self.o = self.i0 * h
        ax = [self.o[a] + h * np.arange(self.shape[a]) for a in range(3)]
        self.X = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
        self.F0 = C.field_local(scene, o, self.X).reshape(self.shape)
        self.Fb = C.field_local(scene, o, self.X, base_only=True).reshape(self.shape)

    def node(self, p_local):
        return np.clip(np.rint((np.asarray(p_local) - self.o) / self.h).astype(int), 0, np.array(self.shape) - 1)


def surface_anchor(F0, o, h):
    """every node's nearest press-time surface point (local), the surface nodes projected onto the zero set"""
    surf = _surface_nodes(F0)
    _, ind = ndimage.distance_transform_edt(~surf, return_indices=True)
    g = np.stack(np.gradient(F0, h), -1)
    idx = np.moveaxis(ind, 0, -1)
    Fs = F0[ind[0], ind[1], ind[2]]
    gs = g[ind[0], ind[1], ind[2]]
    gl2 = np.maximum((gs * gs).sum(-1), 1e-12)
    Q = o + idx * h - (Fs / gl2)[..., None] * gs
    # near the surface each node's own projection along its gradient (what the card's rule reads: ftools.cuh ft_anchor;
    # the distance transform's nearest node put follow-swell 53 um apart from the card on a 417 um step, fs-tools t1/g2)
    near = np.abs(F0) < 3.0 * h
    gl2n = np.maximum((g * g).sum(-1), 1e-12)
    X = o + np.stack(np.meshgrid(*[np.arange(n) for n in F0.shape], indexing='ij'), -1) * h
    Qn = X - (F0 / gl2n)[..., None] * g
    Q[near] = Qn[near]
    return np.ascontiguousarray(Q)


def _surface_nodes(F0):
    surf = np.zeros(F0.shape, bool)
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        ch = (F0[tuple(s0)] < 0) != (F0[tuple(s1)] < 0)
        surf[tuple(s0)] |= ch
        surf[tuple(s1)] |= ch
    return surf


def sheet_gate(F0, seed, reach_mask):
    """Near side: 1 for nodes whose nearest surface node lies on the sheet under the press (surface nodes connected to
    the pressed one within the reach), else 0. The surface nodes are the SOLID side's only (F <= 0 beside air): an air
    node between two facing sheets is not a surface node, so two sheets with one air node between them stay apart (with
    both sides' nodes, a 0.5 mm gap on a 0.25 mm lattice joined them: fs-tools t1/pn1)"""
    solid = F0 <= 0
    surf = np.zeros(F0.shape, bool)
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        surf[tuple(s0)] |= solid[tuple(s0)] & ~solid[tuple(s1)]
        surf[tuple(s1)] |= solid[tuple(s1)] & ~solid[tuple(s0)]
    if not surf.any():
        return np.zeros(F0.shape)
    _, ind = ndimage.distance_transform_edt(~surf, return_indices=True)
    sn = tuple(ind[:, seed[0], seed[1], seed[2]])            # the press's nearest surface node
    lab, _ = ndimage.label(surf & reach_mask, structure=np.ones((3, 3, 3)))
    L = lab[sn]
    if L == 0:
        return np.zeros(F0.shape)
    on = lab == L
    return on[ind[0], ind[1], ind[2]].astype(np.float64)


@command
class FtStroke(RecUndo, Command):
    """one stroke of an approved brush tool on the CPU reference. samples: [{'p': world point under the cursor,
    't': s since the press}] (timed, as the window sends them; a still cursor repeats its point). settings: radius (m),
    strength (the tool's unit: swell m/s at the centre; flatten / smooth 1/s), hardness, smoothing (m), foot, reach
    ('near' | 'through'), follow ('follow' | 'flat'), stop_contact, gap (m)."""
    label = 'Stroke'

    def __init__(self, oid, tool, settings, samples):
        self.oid = int(oid)
        self.tool = str(tool)
        self.settings = dict(settings)
        self.samples = [{'p': [float(v) for v in s['p']], 't': float(s['t'])} for s in samples]
        self.rec = None
        self.info = {}

    def args(self):
        return {'oid': self.oid, 'tool': self.tool, 'settings': self.settings, 'samples': self.samples}

    def do(self, scene):
        C.refuse_protected(scene, [self.oid], 'Stroke')
        o = scene.get(self.oid)
        st = self.settings
        s_ = C.frame(o)[2]
        r = float(st.get('radius', 0.01)) / s_
        sm = float(st.get('smoothing', 0.0)) / s_
        hard = float(st.get('hardness', 0.0))
        fall = FALLOFF[st.get('foot', 'soft')]
        fam = PEAK_SWELL if self.tool in ('swell', 'settle') else PEAK_SHAPE
        rate = float(st.get('strength', 0.002))
        P = C.to_local(o, np.array([s['p'] for s in self.samples]))
        T = np.array([s['t'] for s in self.samples])
        reach = r + sm
        grow = rate / s_ * (T[-1] - T[0] + QUANTUM) if self.tool in ('swell', 'settle') else 0.0
        if self.tool == 'pencil':
            grow = rate / s_ + 2 * o.sculpt.h
        if self.tool == 'smooth':
            grow = 9.0 * max(r / 3.0, 0.75 * o.sculpt.h)          # the band-pass's wide blur sees past the reach
        lo = P.min(0) - reach - grow - 4 * o.sculpt.h
        hi = P.max(0) + reach + grow + 4 * o.sculpt.h
        g = _Grid(scene, o, lo, hi)
        G = g.F0.copy()
        # the sheet under the press, within the stroke's reach (Near side); Through: everything in reach
        near = st.get('reach', 'near') != 'through'
        D = np.full(g.shape, np.inf)
        for q in P:
            D = np.minimum(D, np.linalg.norm(g.X - q, axis=1).reshape(g.shape))
        reach_mask = D <= reach + 2 * g.h
        gate = sheet_gate(g.F0, g.node(P[0]), reach_mask) if near else np.ones(g.shape)
        use_c = bool(st.get('stop_contact', False))
        Dc = np.full(g.shape, 1e9)
        if use_c:
            others = [x for x in scene.objects if x.id != o.id and scene.shown(x)]
            Wp = C.to_world(o, g.X)
            for x in others:
                Dc = np.minimum(Dc, (object_field(scene, x)(Wp) / s_).reshape(g.shape))
        gap = float(st.get('gap', 0.0)) / s_
        # the facing at the press (flat tools' one direction): the field's gradient there; follow tools' anchors
        n0 = _grad_at(g.F0, g.node(P[0]), g.h)
        Q = surface_anchor(g.F0, g.o, g.h)
        if self.tool == 'pencil':
            nq = self._pencil(G, g, gate, P, st, s_)
            return self._finish(scene, o, g, G, nq, s_)
        if self.tool == 'mask':
            return self._mask(scene, o, g, gate, P, T, st, s_, r, hard, sm, fall, fam, rate)
        if self.tool == 'blend':
            Tg = blend_target(g.F0, g.h, r)                   # the press form rounded at radius r, both ways
        # time quanta from the first sample to the last; the cursor where the samples put it at that time
        nq = max(1, int(round((T[-1] - T[0]) / QUANTUM)))
        for q in range(nq):
            tq = T[0] + (q + 0.5) * QUANTUM
            c = np.array([np.interp(tq, T, P[:, a]) for a in range(3)])
            a = rate * QUANTUM / s_
            if self.tool in ('swell', 'settle'):
                sg = 1.0 if self.tool == 'swell' else -1.0
                if st.get('follow', 'follow') == 'flat':
                    Gout = np.empty_like(G)
                    _swell_flat(G, Gout, gate, Dc, use_c, gap, g.o, g.h, c, n0, r, hard, sm, fall, fam, sg * a)
                    G = Gout
                else:
                    _swell_follow(G, gate, Dc, use_c, gap, g.o, g.h, c, r, hard, sm, fall, fam, sg * a, Q)
            elif self.tool == 'flatten':
                # the plane through the surface point under the cursor, facing the surface there (a small patch's
                # average facing, so a grain of texture doesn't tip it), read from the field as it is NOW
                pnt, nrm = _surface_under(G, g, c, n0, r)
                _flatten(G, gate, g.o, g.h, pnt, nrm, r, hard, sm, fall, fam, rate * QUANTUM)
            elif self.tool == 'blend':
                d = np.linalg.norm(Q - c, axis=-1)
                w = np.vectorize(lambda dd: _w(float(dd), r, hard, sm, fall, fam))(d) * gate
                k_ = 1.0 - np.exp(-rate * QUANTUM * w)
                G += k_ * (Tg - G)
            elif self.tool == 'smooth':
                # Strength = the share of a bump of width r removed per second (TOOLSET 1.3): the rule's rate from the
                # measured response (fs-tools t1/fl1: 0.7 / s removed 23 % in 1 s -> kappa 0.373)
                k_s = -math.log(1.0 - min(max(rate, 0.0), 0.99)) / SMOOTH_KAPPA
                _smooth_step(G, gate, g, c, r, hard, sm, fall, fam, k_s * QUANTUM, Q)
            else:
                raise ValueError(self.tool)
        return self._finish(scene, o, g, G, nq, s_)

    def _pencil(self, G, g, gate, P, st, s_):
        """Pencil (TOOLSET 3.3): the tip acts where it is at each moment, at close spacing (<= a tenth of its width)
        along the path, on the surface as it was at the press: no search along the drawn path, within one drag a
        spot is never recut, and the depth is constant whatever the speed (Strength = depth or height); the depth
        limited to half the part's thickness there ("amber"). Patent route: no continuous swept search (MERL US
        8,265,909)."""
        r = float(st.get('radius', 0.001)) / s_
        depth = min(max(float(st.get('strength', 0.0005)), 0.0), PENCIL_MAX_DEPTH, 2.0 * r * s_) / s_
        k = float(st.get('smoothing', 0.0)) / s_
        tip = TIPS[st.get('tip', 'round')]
        ridge = st.get('mode', 'carve') == 'ridge'
        # a cone's floor rises by spacing / (2 r) between points: the point tip needs 0.1 r for 5 % (design 3.3: depth
        # constant within 5 %; at 0.2 r it read 0.451-0.472 of 0.5 mm, at 0.1 r a 5.5 % spread through the window: fs-tools
        # s3/cpu.py, reg_copy3), a ball's by spacing^2 / (8 r)
        step = max(min((0.07 if tip == 0 else 0.2) * r, 0.5 * g.h), 1e-6)
        seg = np.diff(P, axis=0)
        L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        n_st = max(int(math.ceil(L[-1] / step)), 1)
        limited = 0
        for i in range(n_st + 1):
            u = min(L[-1] * i / n_st, L[-1])
            kk = min(int(np.searchsorted(L, u, side='right') - 1), max(len(seg) - 1, 0))
            if len(seg):
                f = (u - L[kk]) / max(L[kk + 1] - L[kk], 1e-12)
                c = P[kk] + f * seg[kk]
                t = seg[kk] / max(np.linalg.norm(seg[kk]), 1e-12)
            else:
                c, t = P[0], np.array([1.0, 0.0, 0.0])
            # on the surface as it was at the press: c projected onto F0, its facing there
            # the facing at the point itself (trilinear reads at +-h: the card's ft_facing; the nearest node's facing put
            # the card's tip a fraction of a node off the reference's)
            n = _grad_pt(g.F0, g.o, g.h, c)
            for _ in range(4):
                c = c - _tri(g.F0, *((c - g.o) / g.h)) * n
            n = _grad_pt(g.F0, g.o, g.h, c)
            t = t - (t @ n) * n
            t = t / max(np.linalg.norm(t), 1e-12) if np.linalg.norm(t) > 1e-9 else _frame_any(n)
            dl = min(depth, 0.5 * thickness_under(g.F0, g.o, g.h, c, n, 4 * depth + 4 * g.h)) if not ridge else depth
            limited += dl < depth - 1e-9
            _stamp(G, gate, g.o, g.h, c, n, t, r, dl, tip, ridge, k)
        self.info['stamps'] = n_st + 1
        self.info['depth_limited_stamps'] = int(limited)
        return n_st + 1

    def _mask(self, scene, o, g, gate, P, T, st, s_, r, hard, sm, fall, fam, rate):
        """Mask paint (TOOLSET 3.2): opacity per second while held (Strength a rate), on the sheet under the press only
        (masking the upper lip never reaches the lower); the field is unchanged"""
        from ..warp.bake import _mask_at
        from .mask import _write_mask
        pk = scene.pack()
        oi = scene.index_of(o.id)
        stt = scene.store
        stt.ensure('mask')
        M0 = np.empty(len(g.X))
        _mask_at(pk.index_flat, stt.kview('mask'), pk.oi_tab, oi, o.sculpt.h, np.ascontiguousarray(g.X), M0)
        M0 = M0.reshape(g.shape)
        M = M0.copy()
        Q = surface_anchor(g.F0, g.o, g.h)
        inv = st.get('erase', False)
        nq = max(1, int(round((T[-1] - T[0]) / QUANTUM)))
        for q in range(nq):
            tq = T[0] + (q + 0.5) * QUANTUM
            c = np.array([np.interp(tq, T, P[:, a]) for a in range(3)])
            d = np.linalg.norm(Q - c, axis=-1)
            w = np.vectorize(lambda dd: _w(float(dd), r, hard, sm, fall, fam))(d) * gate
            M = np.clip(M + (-1.0 if inv else 1.0) * rate * QUANTUM * w, 0.0, 1.0)
        ch = np.abs(M - M0) > 1e-9
        nb = np.array(g.shape) // B
        bm = ch.reshape(nb[0], B, nb[1], B, nb[2], B).any(axis=(1, 3, 5))
        bl = np.argwhere(bm)
        self.rec = BR.StrokeRecord()
        vals = [M[a * B:a * B + B, b * B:b * B + B, c_ * B:c_ * B + B] for a, b, c_ in bl]
        if len(bl):
            _write_mask(scene, o, bl + g.i0 // B, np.array(vals), self.rec)
        self.info.update({'quanta': nq, 'bricks': int(len(bl)), 'max_mask': float(M.max())})
        return self.info

    def _finish(self, scene, o, g, G, nq, s_):
        # only what the tool changed is written (clamped to fs-core's band); the rest of the grid stays as it was (the
        # whole grid's clamp made the result depend on the grid's extent: 0.21 mm between 10 and 300 cursor samples)
        ch = np.abs(G - g.F0) > 1e-12
        G = np.where(ch, np.clip(G, -W_BAND * g.h, W_BAND * g.h), g.F0)
        ch = np.abs(G - g.F0) > 1e-12
        self.rec = BR.StrokeRecord()
        done = _write_grid(scene, o, g, G, ch, self.rec)
        self.info.update({'quanta': nq, 'bricks': len(done), 'grid': list(g.shape),
                          'max_change_mm': round(float(np.abs(G - g.F0).max()) * s_ * 1e3, 4)})
        self.info.update(C.finish_bricks(scene, o, done, self.rec, redistance=False))
        return self.info

    def undo(self, scene):
        if self.rec is not None:
            BR.undo_record(scene, self.rec)
            self.rec = None


def _frame_any(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t = a - (a @ n) * n
    return t / np.linalg.norm(t)


def _grad_pt(F, o, h, p):
    """the unit facing of grid field F at point p (local): central differences of trilinear reads at +-h (ftools.cuh
    ft_facing, the same reads)"""
    f = (np.asarray(p, float) - o) / h
    g = np.array([_tri(F, f[0] + 1.0, f[1], f[2]) - _tri(F, f[0] - 1.0, f[1], f[2]),
                  _tri(F, f[0], f[1] + 1.0, f[2]) - _tri(F, f[0], f[1] - 1.0, f[2]),
                  _tri(F, f[0], f[1], f[2] + 1.0) - _tri(F, f[0], f[1], f[2] - 1.0)])
    return g / max(np.linalg.norm(g), 1e-12)


def _grad_at(F, ijk, h):
    i, j, k = (int(v) for v in ijk)
    i = min(max(i, 1), F.shape[0] - 2)
    j = min(max(j, 1), F.shape[1] - 2)
    k = min(max(k, 1), F.shape[2] - 2)
    g = np.array([F[i + 1, j, k] - F[i - 1, j, k], F[i, j + 1, k] - F[i, j - 1, k], F[i, j, k + 1] - F[i, j, k - 1]])
    return g / max(np.linalg.norm(g), 1e-12)


def _surface_under(G, g, c, n0, r):
    """the surface point near c (along the press facing) and the patch's average facing (radius r / 4) in the field now"""
    # march from above c along -n0 to the surface of G (trilinear), then the facing averaged over a small patch
    p = np.asarray(c, float) + 2.0 * r * n0
    for _ in range(200):
        v = _tri(G, *((p - g.o) / g.h))
        if abs(v) < 1e-7:
            break
        p = p - v * n0
    rr = max(r / 4.0, 2 * g.h)
    off = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * rr
    acc = np.zeros(3)
    for q in p + off:
        f = (q - g.o) / g.h
        e = 0.5
        acc += np.array([_tri(G, f[0] + e, f[1], f[2]) - _tri(G, f[0] - e, f[1], f[2]),
                         _tri(G, f[0], f[1] + e, f[2]) - _tri(G, f[0], f[1] - e, f[2]),
                         _tri(G, f[0], f[1], f[2] + e) - _tri(G, f[0], f[1], f[2] - e)])
    return p, acc / max(np.linalg.norm(acc), 1e-12)


def _smooth_step(G, gate, g, c, r, hard, sm, fall, fam, rdt, Q):
    """F += (1 - exp(-rdt w)) (blur(F) - F) in the reach: bumps up to about the radius go, the form stays"""
    sgh = max(r / 3.0, 0.75 * g.h)
    m_ = r + sm + 9.0 * sgh                       # the blurs see well past the reach (no edge padding inside it)
    lo = np.maximum(((c - m_ - g.o) / g.h).astype(int) - 2, 0)
    hi = np.minimum(((c + m_ - g.o) / g.h).astype(int) + 3, np.array(g.shape))
    sl = tuple(slice(lo[a], hi[a]) for a in range(3))
    sub = G[sl]
    bl = ndimage.gaussian_filter(sub, sigma=max(r / 3.0 / g.h, 0.75), mode='nearest')
    ax = [g.o[a] + g.h * np.arange(lo[a], hi[a]) for a in range(3)]
    d = np.linalg.norm(Q[sl] - c, axis=-1)                    # measured along the surface (the anchors)
    w = np.vectorize(lambda dd: _w(float(dd), r, hard, sm, fall, fam))(d) * gate[sl]
    k = 1.0 - np.exp(-rdt * w)
    # a band-pass: the blur's change minus its own wider blur. A form much larger than r gives a nearly uniform change
    # (its curvature's shrink) which the wider blur takes back out, so held for seconds the form stays; a bump of about
    # the radius gives a local change, which goes. (The plain blur, held 10 s, sank a sphere's pole 0.32 mm; spreading a
    # bump's volume over the brush sank it 0.49 mm: fs-tools t1/fl1.)
    sg = max(r / 3.0 / g.h, 0.75)
    dl = bl - sub
    G[sl] = sub + k * (dl - ndimage.gaussian_filter(dl, sigma=3.0 * sg, mode='nearest'))


def redistance_exact(F, h):
    """the signed distance (grid units of length) to F's zero set: surface points interpolated on every sign-changing
    grid edge, the nearest one for each node by a k-d tree (sub-cell exact), the sign F's"""
    from scipy.spatial import cKDTree
    pts = []
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        f0, f1 = F[tuple(s0)], F[tuple(s1)]
        m = (f0 <= 0) != (f1 <= 0)
        idx = np.argwhere(m).astype(float)
        t = f0[m] / (f0[m] - f1[m])
        idx[:, a] += t
        pts.append(idx)
    # nodes lying on the zero set are surface points themselves (a value of +-1e-18 there hid them: no sign change
    # to either side, the nearest point found a cell away)
    pts.append(np.argwhere(np.abs(F) <= 1e-9 * h).astype(float))
    P = np.concatenate(pts) * h
    if not len(P):
        return F.copy()
    X = np.stack(np.meshgrid(*[np.arange(n) * h for n in F.shape], indexing='ij'), -1).reshape(-1, 3)
    d, _ = cKDTree(P).query(X, workers=-1)
    return np.where(F.reshape(-1) <= 0, -d, d).reshape(F.shape)


def blend_target(F0, h, rho):
    """the solid rounded at radius rho both ways (Blend, TOOLSET 3.4: a crease filled, a sharp edge rounded): its
    closing then its opening by a ball of radius rho, as offsets of signed distances re-distanced exactly at each step
    (thresholded node distance transforms lost about a cell a step: a 5 mm round came out 3 mm and flat faces moved
    0.75 mm, fs-tools t1/bl1)"""
    D = redistance_exact(F0 - rho, h)            # dilate
    Cl = redistance_exact(D + rho, h)            # erode: the closing
    E = redistance_exact(Cl + rho, h)            # erode
    return E - rho                               # dilate: the opening of the closing


def _write_grid(scene, o, g, G, ch, rec):
    nb = np.array(g.shape) // B
    bmask = ch.reshape(nb[0], B, nb[1], B, nb[2], B).any(axis=(1, 3, 5))
    bl = np.argwhere(bmask)
    keys = bl + g.i0 // B
    st = scene.store
    vals = []
    for (a, b, c), k in zip(bl, keys):
        sl = (slice(a * B, a * B + B), slice(b * B, b * B + B), slice(c * B, c * B + B))
        cur = st.delta[o.sculpt.get_slot(tuple(k))].astype(np.float64) if o.sculpt.get_slot(tuple(k)) >= 0 else \
            np.zeros((B, B, B))
        vals.append(np.where(ch[sl], G[sl] - g.Fb[sl], cur))
    return C.write_bricks(scene, o, keys, np.array(vals), rec) if len(keys) else []


# ---- the card path's parameter block (fs-core's _plan calls this for typ >= 30; ftools.cuh reads it) -----------------
FT_TYP = {'swell': 30, 'settle': 31, 'flatten': 32, 'pencil': 33}
PENCIL_MAX_DEPTH = 0.005           # m: the deepest a Pencil line goes (its Depth / Height), and never more than twice
                                   # the tip's half-width: a line, not a hole (the tip's ball is tip + 2 depth)
FT_SLOTS = {'hard': 51, 'smooth': 52, 'fall': 53, 'family': 54, 'a': 55, 'mode': 56, 'gap': 57, 'contact': 58,
            'p': 59, 'h': 62, 'r': 63}


def ft_bp(tool, settings, h, dt, s=1.0, plane_point=None, direction=None):
    """the values of bp slots 51..63 (local units; s = the object's scale) for one dab of an approved tool covering dt
    seconds of the stroke (held tools: Strength is a rate). Pencil: one tip point; direction = the stroke's direction there
    (local; None: any frame). -> {slot: value}"""
    st = settings
    if tool == 'pencil':
        # the tip's half-width (FT_R; never the brushes' 2 h floor), its depth or height (FT_A), the tip + 4 for a ridge
        # (FT_MODE), the blend width (FT_SMOOTH), the direction (FT_P); ftools.cuh ft_pencil reads them
        r_tip = float(st['radius_local']) if 'radius_local' in st else float(st.get('radius', 0.001)) / s
        # Strength is the depth in metres (the strip's Depth); bounded, so a unitless default never becomes a 0.5 m
        # tip (the window's warm-up stroke passes strength 0.5: its ball tried to commit 3 GB, fs-tools s3/reg_copy2)
        depth = min(max(float(st.get('strength', 0.0005)), 0.0), PENCIL_MAX_DEPTH, 2.0 * r_tip * s)
        out = {FT_SLOTS['r']: r_tip, FT_SLOTS['a']: depth / s,
               FT_SLOTS['mode']: TIPS[st.get('tip', 'round')] + (4 if st.get('mode', 'carve') == 'ridge' else 0),
               FT_SLOTS['smooth']: float(st.get('smoothing', 0.0)) / s, FT_SLOTS['h']: float(h),
               FT_SLOTS['hard']: float(st.get('hardness', 0.0)), FT_SLOTS['fall']: 0, FT_SLOTS['family']: PEAK_SHAPE,
               FT_SLOTS['gap']: 0.0, FT_SLOTS['contact']: 0.0}
        d = np.zeros(3) if direction is None else np.asarray(direction, float)
        for i in range(3):
            out[FT_SLOTS['p'] + i] = float(d[i])
        return out
    r_sm = float(st.get('smoothing', 0.0)) / s
    out = {FT_SLOTS['hard']: float(st.get('hardness', 0.0)), FT_SLOTS['smooth']: r_sm,
           FT_SLOTS['fall']: FALLOFF[st.get('foot', 'soft')],
           FT_SLOTS['family']: PEAK_SWELL if tool in ('swell', 'settle') else PEAK_SHAPE,
           FT_SLOTS['mode']: 1 if st.get('follow', 'follow') == 'flat' else 0,
           FT_SLOTS['gap']: float(st.get('gap', 0.0)) / s, FT_SLOTS['contact']: 1.0 if st.get('stop_contact') else 0.0,
           FT_SLOTS['h']: float(h)}
    if 'radius_local' in st:                     # the profile's radius (fs-core's bp[7] is the dab's whole ball)
        out[FT_SLOTS['r']] = float(st['radius_local'])
    rate = float(st.get('strength', 0.0))
    if tool in ('swell', 'settle'):
        out[FT_SLOTS['a']] = (1.0 if tool == 'swell' else -1.0) * rate * dt / s
    else:
        out[FT_SLOTS['a']] = rate * dt
    if plane_point is not None:
        for i in range(3):
            out[FT_SLOTS['p'] + i] = float(plane_point[i])
    return out
