"""Transform of a region, and Bend / Twist / Taper carrying the sculpting (TOOLSET_DESIGN 3.2, approved 11-14).

The design: "a region: a masked or selected part of an object, moved with a soft edge (the shared Smoothing) and
carrying its sculpting"; "Bend, Twist and Taper take an axis, a centre and an extent drawn on screen. They act on the
whole field, sculpting included; today they bend only the base form and leave the sculpting behind."

Both go through the one warp engine's bake (warp/bake.py: F'(x) = F(phi^-1(x)) into the sculpt lattice, exact undo):
- REGION: the part to move is a painted region (its region id) or everything the Mask does not protect (Mask protects:
  "masked areas change 0.0 mm under every tool"). Its weight is carried from the surface into the material behind it:
  each point takes the region weight of its NEAREST SURFACE POINT (an exact distance transform), so a nose moves with
  its inside, not as a skin; the soft edge is the shared Smoothing, measured inward from the region's border (the
  protected part outside it never moves). The move is a
  rigid motion (rotate about a pivot, then translate) as the flow of its screw generator with the weight riding the
  region (warp/build.rigid): weight 1 moves exactly rigidly, weight 0 never moves; optional Stop at contact and Keep
  volume.
- BEND / TWIST / TAPER: closed-form one-to-one maps with their inverse (Barr's deformations with an extent: inside the
  extent the deformation, beyond it the end's rigid continuation), about an axis through a centre, of the whole object.
"""
import math

import numpy as np
from numba import njit, prange
from scipy import ndimage

from ..core import program as P
from ..warp.flow import _tri
from ..warp import build as WB
from ..warp.bake import WarpObject, object_field
from . import common as C


# ---- region weights --------------------------------------------------------------------------------------------------
def region_grid(scene, o, source='unmasked', smoothing=0.0, spacing=None, cap=128, soft=False, pad=None, field=False):
    """-> (G, origin, h): the region's weight on a world grid over the object's box (+ smoothing): 1 in the region,
    its soft edge rising from 0 at its border to 1 over `smoothing` inward. soft: the carried weight itself (each point
    its nearest surface point's value, a soft mask kept soft), no threshold. pad: the grid's margin beyond the object's
    box (default smoothing + 4 lattice steps); field: also return the object's field on the grid (G, origin, h, F)"""
    from ..warp.bake import _mask_at
    lo, hi = C.world_box(o)
    if pad is None:
        pad = smoothing + 4.0 * o.sculpt.h * C.frame(o)[2]
    lo, hi = lo - pad, hi + pad
    h = spacing or float(max(o.sculpt.h * C.frame(o)[2], (hi - lo).max() / (cap - 1)))
    n = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    ax = [lo[a] + h * np.arange(n[a]) for a in range(3)]
    Wp = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    F = object_field(scene, o)(Wp).reshape(tuple(n))
    Xl = np.ascontiguousarray(C.to_local(o, Wp))
    pk = scene.pack()
    oi = scene.index_of(o.id)
    st = scene.store
    if isinstance(source, str) and source.startswith('rid:'):
        rid = int(source.split(':')[1])
        vals = np.zeros(len(Xl))
        if st.has('rid'):
            # region ids are labels: the nearest node's id (no interpolation)
            g = o.sculpt
            idx = np.rint(Xl / g.h).astype(np.int64)
            for q, (i, j, k) in enumerate(idx):
                b = (i >> 3, j >> 3, k >> 3)
                sl = g.get_slot(b)
                if sl >= 0 and st.rid[sl][i & 7, j & 7, k & 7] == rid:
                    vals[q] = 1.0
    else:
        m = np.zeros(len(Xl))
        if st.has('mask'):
            _mask_at(pk.index_flat, st.kview('mask'), pk.oi_tab, oi, o.sculpt.h, Xl, m)
        m = np.clip(m, 0.0, 1.0)
        if getattr(o, 'mask_inv', False):
            m = 1.0 - m
        vals = 1.0 - m                                   # everything the Mask does not protect
    vals = vals.reshape(tuple(n))
    surf = np.zeros(tuple(n), bool)
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1)
        sl1[a] = slice(1, None)
        ch = (F[tuple(sl0)] < 0) != (F[tuple(sl1)] < 0)
        surf[tuple(sl0)] |= ch
        surf[tuple(sl1)] |= ch
    if not surf.any():
        return (np.zeros(tuple(n)), lo, h, F) if field else (np.zeros(tuple(n)), lo, h)
    _, ind = ndimage.distance_transform_edt(~surf, return_indices=True)
    carried = vals[ind[0], ind[1], ind[2]]               # each point: its nearest surface point's weight
    if soft:
        return (carried, lo, h, F) if field else (carried, lo, h)
    R = carried > 0.5
    if smoothing > 0:
        # the soft edge lies INSIDE the region (the part that moves), so whatever is outside it (the Mask's protected
        # part) moves 0.0 mm: "masked areas change 0.0 mm under every tool"
        din = ndimage.distance_transform_edt(R) * h
        t = np.clip(din / smoothing, 0.0, 1.0)
        G = np.where(R, t * t * (3.0 - 2.0 * t), 0.0)
    else:
        G = R.astype(float)
    return (G, lo, h, F) if field else (G, lo, h)


# ---- closed-form deformations ------------------------------------------------------------------------------------------
class ClosedWarp:
    """Barr's bend / twist / taper with an extent, about an axis through a centre (world), and skew. amount: bend and
    twist in degrees over the extent, taper the scale at the extent's far end (1 = none), skew the lean in degrees (the
    sections slide along `direction` by tan(amount) per m along the axis: the 11-10 skew). Inverse and forward closed
    form."""

    def __init__(self, kind, centre, axis, extent, amount, direction=None, box=None):
        self.kind = kind
        self.c = np.asarray(centre, float)
        a = np.asarray(axis, float)
        self.a = a / np.linalg.norm(a)
        d = np.asarray(direction if direction is not None else
                       ([1.0, 0, 0] if abs(self.a[0]) < 0.9 else [0, 0, 1.0]), float)
        d = d - (d @ self.a) * self.a
        self.d = d / np.linalg.norm(d)
        self.b = np.cross(self.a, self.d)
        self.s0, self.s1 = float(extent[0]), float(extent[1])
        L = self.s1 - self.s0
        if kind in ('bend', 'twist'):
            self.k = math.radians(float(amount)) / L             # rad per m
        elif kind == 'skew':
            self.k = math.tan(math.radians(float(amount)))       # slide per m along the axis
        else:
            self.k = (float(amount) - 1.0) / L                   # scale change per m
        self.box = np.array([-1e9] * 3 + [1e9] * 3) if box is None else np.asarray(box, float)
        self.steps = 0
        self.spec = {'kind': kind}

    def _coords(self, Pw):
        q = Pw - self.c
        return q @ self.d, q @ self.a, q @ self.b

    def _back(self, u, s, w):
        return self.c + u[:, None] * self.d + s[:, None] * self.a + w[:, None] * self.b

    def forward(self, Pw):
        Pw = np.asarray(Pw, float).reshape(-1, 3)
        u, s, w = self._coords(Pw)
        sh = np.clip(s, self.s0, self.s1)
        if self.kind == 'twist':
            th = self.k * (sh - self.s0)
            c, sn = np.cos(th), np.sin(th)
            return self._back(c * u - sn * w, s, sn * u + c * w)
        if self.kind == 'taper':
            g = 1.0 + self.k * (sh - self.s0)
            return self._back(g * u, s, g * w)
        if self.kind == 'skew':
            return self._back(u + self.k * (sh - self.s0), s, w)
        if abs(self.k) < 1e-12:
            return Pw.copy()
        rho = 1.0 / self.k
        th = self.k * (sh - self.s0)
        c, sn = np.cos(th), np.sin(th)
        e = s - sh
        A = rho - u
        u2 = rho - A * c + e * sn
        s2 = self.s0 + A * sn + e * c
        return self._back(u2, s2, w)

    def inverse(self, Pw):
        Pw = np.asarray(Pw, float).reshape(-1, 3)
        u, s, w = self._coords(Pw)
        if self.kind == 'twist':
            sh = np.clip(s, self.s0, self.s1)
            th = -self.k * (sh - self.s0)
            c, sn = np.cos(th), np.sin(th)
            return self._back(c * u - sn * w, s, sn * u + c * w)
        if self.kind == 'taper':
            sh = np.clip(s, self.s0, self.s1)
            g = 1.0 + self.k * (sh - self.s0)
            return self._back(u / g, s, w / g)
        if self.kind == 'skew':
            sh = np.clip(s, self.s0, self.s1)
            return self._back(u - self.k * (sh - self.s0), s, w)
        if abs(self.k) < 1e-12:
            return Pw.copy()
        rho = 1.0 / self.k
        th1 = self.k * (self.s1 - self.s0)
        # angle of the point about the centre of curvature (u = rho, s = s0)
        phi = np.arctan2((s - self.s0) * np.sign(rho), (rho - u) * np.sign(rho))
        lo_t, hi_t = min(0.0, th1), max(0.0, th1)
        tc = np.clip(phi, lo_t, hi_t)
        c, sn = np.cos(tc), np.sin(tc)
        A = (rho - u) * c + (s - self.s0) * sn
        e = -(rho - u) * sn + (s - self.s0) * c
        sh = self.s0 + tc / self.k
        return self._back(rho - A, sh + e, w)


    def inverse_scaled(self, Pw, W):
        """the inverse of this deform at W x its amount, per point (W in 0..1): a twist turns back by W x its angle (along
        the arc), a bend by W x its curvature, a skew slides back W x its slide, a taper scales by g^W. W = 1 is inverse()
        itself, bit for bit; W = 0 leaves the point where it is"""
        Pw = np.asarray(Pw, float).reshape(-1, 3)
        W = np.asarray(W, float).reshape(-1)
        out = Pw.copy()
        full = W >= 1.0
        if full.any():
            out[full] = self.inverse(Pw[full])
        m = (W > 0.0) & ~full
        if not m.any():
            return out
        P, w_ = Pw[m], W[m]
        u, s, w = self._coords(P)
        sh = np.clip(s, self.s0, self.s1)
        if self.kind == 'twist':
            th = -self.k * w_ * (sh - self.s0)
            c, sn = np.cos(th), np.sin(th)
            out[m] = self._back(c * u - sn * w, s, sn * u + c * w)
        elif self.kind == 'taper':
            g = (1.0 + self.k * (sh - self.s0)) ** w_
            out[m] = self._back(u / g, s, w / g)
        elif self.kind == 'skew':
            out[m] = self._back(u - w_ * self.k * (sh - self.s0), s, w)
        else:
            k = self.k * w_
            tiny = np.abs(k) < 1e-9
            k = np.where(tiny, 1e-9, k)
            rho = 1.0 / k
            th1 = k * (self.s1 - self.s0)
            phi = np.arctan2((s - self.s0) * np.sign(rho), (rho - u) * np.sign(rho))
            tc = np.clip(phi, np.minimum(0.0, th1), np.maximum(0.0, th1))
            c, sn = np.cos(tc), np.sin(tc)
            A = (rho - u) * c + (s - self.s0) * sn
            e = -(rho - u) * sn + (s - self.s0) * c
            shb = self.s0 + tc / k
            r_ = self._back(rho - A, shb + e, w)
            r_[tiny] = P[tiny]
            out[m] = r_
        return out


def _ids_spec(spec):
    return spec


# ---- a whole-object deform over a masked patch: the MOTION weighted by the mask ---------------------------------------
# Design 3.2: "Masked areas change 0.0 mm under every tool", and a deform "acts on the whole field, sculpting included".
# A value blend (the warped value mixed back to the old one by the mask) tears where staying meets moving (fs-tools
# t1/mk2's Skew case). Instead the deform's MOTION is weighted: each node's preimage is the deform's inverse at W(x) x
# its amount (ClosedWarp.inverse_scaled: a twist turns back along the arc by W x its angle, a bend by W x its curvature),
#     psi(x) = phi_{W(x)}^-1(x),        F'(x) = F(psi(x))
# with phi the deform and W = 1 - the mask CARRIED into the material behind the surface (region_grid(soft=True): each
# point its nearest surface point's mask), its edge widened over the moving side:
# - a masked node has W = 0: psi(x) = x, its value kept bit for bit (0.0 mm);
# - W = 1: psi = phi^-1 exactly, the deform itself (the sculpting carried);
# - between, the motion fades over the edge. psi is one-to-one while the slide x |grad W| stays well under one, so the
#   edge is at least EDGE_PER_SLIDE x the deform's slide at the hold's border (deform_weight).
# The displacement blended instead (psi = x + W (phi^-1(x) - x)) moved partly weighted points along the chord of a
# turn, not its arc: under a 90 deg twist the sections shrank and the box's top flared (candy wrapper; mk3/final, 11-14).
# Measured before this (fs-tools mk3 close-ups, 11-14): a weighted VELOCITY (the flow W(x) u_t(x)) kept the patch exact
# but stopped the material at the hold's edge, so everything sweeping toward the patch piled into a layer ~50:1 thinner
# than it came, below the lattice: dust, creases and terraced rings at a skew 25 / twist 30 / twist 90. A weighted
# motion compresses by at most the slide x |grad W| (bounded by the edge), and costs one closed form and one read a node.
class MaskedDeform:
    """the deform cw (a ClosedWarp) with its displacement weighted by W (see above): inverse (the bake's) in closed form,
    forward by Newton's method (cues and checks)"""

    def __init__(self, cw, G, origin, h, smoothing=0.0):
        self.cw = cw
        self.kind = cw.kind
        self.G = np.ascontiguousarray(G, np.float64)
        self.o = np.asarray(origin, float)
        self.h = float(h)
        self.smoothing = float(smoothing)
        self.box = np.asarray(cw.box, float)
        self.local = False                      # the bake must not value-blend: the motion holds the mask
        self.steps = 0
        self.spec = {'kind': 'masked_' + cw.kind}
        self.stats = {}

    def weight(self, P):
        P = np.ascontiguousarray(np.asarray(P, float).reshape(-1, 3))
        out = np.empty(len(P))
        _wgrid(self.G, self.o[0], self.o[1], self.o[2], self.h, P, out)
        return out

    def inverse(self, Pw):
        P = np.asarray(Pw, float).reshape(-1, 3)
        W = self.weight(P)
        out = P.copy()
        m = W > 0.0
        if m.any():
            out[m] = self.cw.inverse_scaled(P[m], W[m])
        self.stats = {'points': len(P), 'held': int((W <= 0.0).sum()), 'full': int((W >= 1.0).sum()),
                      'edge': int(((W > 0.0) & (W < 1.0)).sum())}
        return out

    def forward(self, Pw, iters=30):
        """psi^-1 by Newton's method (central-difference Jacobian), from the deform's own image blended by W"""
        P = np.asarray(Pw, float).reshape(-1, 3)
        Wp = self.weight(P)
        y = P + Wp[:, None] * (self.cw.forward(P) - P)        # a start: the displacement blended
        e = 1e-6
        for _ in range(iters):
            r = self.inverse(y) - P
            if np.abs(r).max() < 1e-12:
                break
            J = np.empty((len(y), 3, 3))
            for a in range(3):
                ea = np.eye(3)[a] * e
                J[:, :, a] = (self.inverse(y + ea) - self.inverse(y - ea)) / (2 * e)
            y = y - np.linalg.solve(J, r[:, :, None])[:, :, 0]
        return y


@njit(cache=True, parallel=True)
def _wgrid(G, ox, oy, oz, h, P, out):
    for i in prange(P.shape[0]):
        out[i] = _tri(G, ox, oy, oz, h, P[i, 0], P[i, 1], P[i, 2], 1.0)


EDGE_PER_SLIDE = 2.5    # the moving part's soft edge is at least this x the deform's slide at the hold's border


def deform_weight(scene, o, smoothing=0.0, cap=128, cw=None, info=None):
    """W on a world grid over the object's box (+ a margin): 0 where held, 1 where the deform acts in full, a smooth
    edge between; 1 at the grid's edge and beyond it. -> (W, origin, h), or None when nothing is masked.
    - the hold: the mask carried into the material behind the surface (region_grid(soft=True)), and in air only within
      the masked surface's own lattice cells (a trilinear read there takes corners up to sqrt(3) steps out), widened by
      one grid step so the masked surface is held to its border (a grid coarser than the lattice smeared the border: a
      masked point read W 0.25 and moved);
    - the edge: W rises from the hold over S = max(Smoothing, EDGE_PER_SLIDE x the deform's slide at the hold's border),
      as a smoothstep of the distance from the hold, in material and air alike. Design 1.3: Smoothing 0 is "a crisp foot,
      still continuous: never a tear"; the floor keeps the weighted motion one-to-one (slide x |grad W| <= 1.5 /
      EDGE_PER_SLIDE = 0.6), so a small deform keeps a crisp edge and a large one gets the room the material needs."""
    lo, hi = C.world_box(o)
    ext = float((hi - lo).max())
    hl = o.sculpt.h * C.frame(o)[2]
    pad = 8.0 * max(hl, ext / (cap - 17))
    for _ in range(2):
        # the grid reaches past the box by the edge (+ 8 steps), so W reaches 1 inside it: no step at its border
        hg = max(hl, (ext + 2.0 * pad) / (cap - 1))
        G, glo, h, F = region_grid(scene, o, 'unmasked', 0.0, spacing=hg, soft=True, pad=pad, field=True)
        G = np.clip(G, 0.0, 1.0)
        if G.min() >= 1.0:
            return None
        G = np.where(F > 1.75 * h, 1.0, G)              # air holds only within the masked surface's own cells
        G = ndimage.minimum_filter(G, size=3, mode='nearest')
        held = G < 0.5
        S = float(smoothing)
        if cw is not None and held.any():
            border = held & ndimage.binary_dilation(~held, np.ones((3, 3, 3), bool))
            if border.any():
                P_ = glo + np.argwhere(border) * h
                slide = float(np.linalg.norm(cw.forward(P_) - P_, axis=1).max())
                S = max(S, EDGE_PER_SLIDE * slide)
                if info is not None:
                    info['slide_mm'] = round(slide * 1e3, 2)
        if S + 4.0 * h <= pad:
            break
        pad = S + 8.0 * h
    if info is not None:
        info['edge_mm'] = round(S * 1e3, 2)
        info['grid_mm'] = round(h * 1e3, 3)
    if S > 0 and held.any():
        din = ndimage.distance_transform_edt(~held) * h
        t = np.clip(din / S, 0.0, 1.0)
        G = G * (t * t * (3.0 - 2.0 * t))
    return G, glo, h


# ---- the jobs (engine thread, under its lock) --------------------------------------------------------------------------
def region_move(engine, oid, R, t, pivot, smoothing=0.0, source='unmasked', keep_volume=False, contact=False, gap=0.0):
    """move a region of an object: rotation R (3x3) about pivot, then translation t (world)"""
    spec = {'motion': 'rigid', 'use_mask': True, 'R': np.asarray(R, float).tolist(), 't': [float(v) for v in t],
            'pivot': [float(v) for v in pivot], 'keep_volume': bool(keep_volume),
            'support': {'metric': 'region', 'source': source, 'smoothing': float(smoothing), 'c0': list(pivot),
                        'radius': 1.0}}
    if contact:
        spec['contact'] = {'objects': 'others', 'gap': float(gap)}
    cmd = WarpObject(oid, spec)
    info = engine._push(cmd, 'Move region')
    engine.tool_result = {'tool': 'transform', 'info': info}
    return True


def deform(engine, oid, kind, centre, axis, extent, amount, direction=None, smoothing=0.0):
    """bend / twist / taper / skew the whole object, sculpting included; a masked patch stays (the motion weighted by the
    mask, MaskedDeform), the moving part's edge softened over `smoothing` (m) beyond the mask's own softness"""
    spec = {'motion': 'deform', 'use_mask': True, 'mask_motion': True, 'smoothing': float(smoothing), 'kind': kind,
            'centre': [float(v) for v in centre],
            'axis': [float(v) for v in axis],
            'extent': [float(v) for v in extent], 'amount': float(amount),
            'direction': None if direction is None else [float(v) for v in direction],
            'support': {'metric': 'none', 'c0': [float(v) for v in centre], 'radius': 1.0}}
    cmd = WarpObject(oid, spec)
    info = engine._push(cmd, kind.capitalize())
    engine.tool_result = {'tool': 'transform', 'info': info}
    return True


JOBS = {'tools.region_move': region_move, 'tools.deform': deform}
