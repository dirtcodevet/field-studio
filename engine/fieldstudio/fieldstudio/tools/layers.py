"""Edit by LAYER (DESIGN_NOTES 10d): brush commands that change one layer and keep the others.

Bands of an object's field F = base + delta (delta = the sculpt layer), split at a length `detail_scale` (s):
    detail   D = delta - G_s * delta              (the sculpt's high band: pores, creases, scratches)
    shape    S = base + G_s * delta               (the analytic base plus the sculpt's low band)
    colour   the albedo params (SetObjectProps 'colour'); no colour field in V1, so no colour brush yet
G_s is a Gaussian low-pass with sigma = s (scipy.ndimage, deterministic).
V1.1: the two ops are split with a GUARD BAND between them: the detail op removes everything finer than
g_d*s (g_d = band_gap, default 1.5) and the shape op works only on the part coarser than g_s*s (g_s = shape_gap,
default 4), so neither op reaches into the other's band through the soft edge of one Gaussian (V1 used one
Gaussian for both: smooth-detail removed 21 % of the detail, smooth-shape changed it by 36 %).
  layer 'detail' (smooth detail, keep shape): delta += w*str*(G_{g_d s}*delta - delta)  -> D removed, S kept
  layer 'shape'  (smooth shape, keep detail): S_lo = base + G_{g_s s}*delta (the shape band, the guard band
                 excluded); delta += w*str*(G_r*S_lo - S_lo), sigma_r = max(radius/4, s) -> S smoothed at the
                 brush scale, D kept (E: the detail stays at its lattice position, so it rides the moved surface
                 only for moves smaller than its extent)
The gaps come from a sweep on the L1/L2 scene (M, V1_RESULTS.md 'V1.1'). With Gaussian bands and the smooth
falloff, no linear detail filter tried reached both L1 targets at once (>= 50 % removed, < 10 % low-band change).
LayerSmooth subclasses BrushStroke, so History.extend_stroke works for live drags (one undo step per drag).
"""
import copy

import numpy as np
from scipy import ndimage

from ..core import brush as BR
from ..core import program as P
from ..core.history import BrushStroke, command
from . import common as C

import os as _os
# fs-core 11-14 (the 'detail band' Smooth froze the window 2-2.7 s a sample, 20:07-20:11; a proposal: move all the
# layers onto the card): the band split's Gaussians run on the card when it is there (the same separable filter, mode 'nearest',
# truncate 3, in float32, the precision the sculpt pool keeps; fs-core/ly_check.py: the same delta to 0.004 um). OFF by
# default (FS_LAYER_CARD=1 on): it took a shape sample at r 22.9 only from 2.07 to 1.68 s; the rest is the brush box
# itself (~7.5 M nodes a sample for a ~165 k-node ball). The window no longer waits on any of it: the GIL is released
# (common.py _eval_range, write_nodes)
_CARD = _os.environ.get('FS_LAYER_CARD', '0') != '0'
_CARD_MIN = 1 << 18           # smaller boxes stay on the CPU (the transfer costs more than the filter)


def _gauss(a, sigma):
    if _CARD and a.size >= _CARD_MIN:
        try:
            import cupy as _cp
            from cupyx.scipy import ndimage as _cnd
            d = _cnd.gaussian_filter(_cp.asarray(a, dtype=_cp.float32), float(sigma), mode='nearest', truncate=3.0)
            out = _cp.asnumpy(d).astype(np.float64)
            del d
            return out
        except Exception:
            pass
    return ndimage.gaussian_filter(a, sigma, mode='nearest', truncate=3.0)


LAYERS = ('detail', 'shape')
DEFAULT_LAYER_SETTINGS = {'tool': 'layer_smooth', 'layer': 'detail', 'radius': 0.01, 'strength': 1.0,
                          'falloff': 'smooth', 'curve': None, 'spacing': 0.25, 'detail_scale': 0.002, 'band_gap': 1.5,
                          'shape_gap': 4.0,
                          'symmetry': [False, False, False], 'use_mask': True}


def bands(scene, oi, lo, dims, detail_scale, filtered=True):
    """(base, delta, low band, high band) on lattice nodes lo..lo+dims-1 (for tests and the UI's readouts)."""
    obj = scene.objects[oi]
    h = obj.sculpt.h
    pk = scene.pack()
    ii = [(np.arange(dims[a]) + lo[a]) * h for a in range(3)]
    X = np.stack(np.meshgrid(*ii, indexing='ij'), -1).reshape(-1, 3)
    base = np.empty(len(X))
    C._eval_range(pk.code, pk.prm, int(pk.oi_tab[oi, P.OI_C0]), int(pk.oi_tab[oi, P.OI_C1]),
                  np.ascontiguousarray(X), base)
    base = base.reshape(tuple(int(d) for d in dims))
    delta = C.gather_nodes(scene, oi, lo, dims)
    if not filtered:
        return base, delta, None, None
    sig = detail_scale / h
    low = ndimage.gaussian_filter(delta, sig, mode='nearest', truncate=3.0)
    return base, delta, low, delta - low


def apply_layer_sample(scene, oi, settings, c, rec, lut):
    obj = scene.objects[oi]
    h = obj.sculpt.h
    r = max(float(settings['radius_local']), 2 * h)
    s_det = max(float(settings['detail_scale_local']), h)
    if settings['layer'] == 'detail':     # the detail op removes everything finer than band_gap * s
        s_gap = s_det * max(float(settings.get('band_gap', 1.5)), 1.0)
    else:                                  # the shape op sees only what is coarser than shape_gap * s
        s_gap = s_det * max(float(settings.get('shape_gap', 4.0)), 1.0)
    sig_shape = max(0.25 * r, s_det)
    apron = 3.0 * s_gap + (3.0 * sig_shape if settings['layer'] == 'shape' else 0.0) + 2 * h
    c = np.asarray(c, float)
    lo = np.floor((c - r - apron) / h).astype(np.int64)
    hi = np.ceil((c + r + apron) / h).astype(np.int64)
    dims = hi - lo + 1
    if not _REGION:
        return _apply_layer_sample_box(scene, oi, settings, c, rec, lut, h, r, s_det, s_gap, sig_shape, lo, dims)
    # fs-core 11-14 (the 'detail band' Smooth: 2-3.7 s a sample at r 22.9, the stroke catching up for 3-20 s after the
    # release): the change is written in the brush's ball only, so only what the ball's nodes read is computed. The box is
    # the same; each Gaussian runs as its three 1-D passes over the region the next pass or the ball needs (_gauss_region:
    # the same scipy filter on slabs, cropped where its 'nearest' edge never reaches), and the base is evaluated where the
    # shape filter reads it. The ball's values are the full box's, bit for bit (fs-core/ly_region.py).
    ci = np.floor((c - lo * h) / h).astype(np.int64)
    rr = int(np.ceil(r / h)) + 1
    blo = np.maximum(ci - rr, 0)
    bhi = np.minimum(ci + rr + 1, dims)
    if np.any(bhi <= blo):
        return 0
    delta = C.gather_nodes(scene, oi, lo, dims)
    ii = [(np.arange(blo[a], bhi[a]) + lo[a]) * h for a in range(3)]
    X, Y, Z = np.meshgrid(*ii, indexing='ij')
    u = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / r
    w = np.interp(np.clip(u, 0, 1), np.linspace(0, 1, len(lut)), lut)
    w[u >= 1.0] = 0.0
    if settings.get('use_mask', True):
        m = C.gather_nodes(scene, oi, lo + blo, bhi - blo, 'mask')
        w *= 1.0 - np.clip(m, 0.0, 1.0)
    st = float(settings['strength'])
    bsl = tuple(slice(int(blo[a]), int(bhi[a])) for a in range(3))
    if settings['layer'] == 'detail':
        coarse_b = _gauss_region(delta, s_gap / h, blo, bhi)
        df = w * st * (coarse_b - delta[bsl])
    else:
        rs = int(3.0 * (sig_shape / h) + 0.5)
        slo = np.maximum(blo - rs, 0)
        shi = np.minimum(bhi + rs, dims)
        coarse_s = _gauss_region(delta, s_gap / h, slo, shi)
        base_s = _base_region(scene, oi, lo + slo, shi - slo, h)
        S = base_s + coarse_s
        Ss_b = _gauss_region(S, sig_shape / h, blo - slo, bhi - slo)
        S_b = S[tuple(slice(int(blo[a] - slo[a]), int(bhi[a] - slo[a])) for a in range(3))]
        df = w * st * (Ss_b - S_b)
    df[w <= 0.0] = 0.0
    df = df.astype(np.float32).astype(np.float64)     # what the float32 pool will hold
    return C.write_nodes(scene, oi, lo + blo, df, rec)


_REGION = _os.environ.get('FS_LAYER_REGION', '1') != '0'    # fs-core 11-14: 0 = the whole box, as before


def _gauss_region(a, sigma, out_lo, out_hi):
    """the Gaussian of a (sigma in nodes, mode 'nearest', truncate 3: scipy's gaussian_filter, axis 0, 1, 2 in turn) at the
    index box out_lo .. out_hi only: each 1-D pass over the region the later passes read"""
    rad = int(3.0 * float(sigma) + 0.5)
    lo_ = np.asarray(out_lo, np.int64)
    hi_ = np.asarray(out_hi, np.int64)
    n = np.array(a.shape, np.int64)
    cur = a
    off = np.zeros(3, np.int64)                     # index of cur[0, 0, 0] in a
    for ax in range(3):
        # the region this pass outputs: the final range on the axes done and this one, +- rad on the axes still to come
        olo = np.array([lo_[k] if k <= ax else max(lo_[k] - rad, 0) for k in range(3)])
        ohi = np.array([hi_[k] if k <= ax else min(hi_[k] + rad, n[k]) for k in range(3)])
        ilo = olo.copy(); ihi = ohi.copy()
        ilo[ax] = max(olo[ax] - rad, 0); ihi[ax] = min(ohi[ax] + rad, n[ax])
        slab = cur[tuple(slice(int(ilo[k] - off[k]), int(ihi[k] - off[k])) for k in range(3))]
        f = ndimage.gaussian_filter1d(slab, float(sigma), axis=ax, mode='nearest', truncate=3.0)
        cur = f[tuple(slice(int(olo[k] - ilo[k]), int(ohi[k] - ilo[k])) for k in range(3))]
        off = olo
    return cur


def _base_region(scene, oi, lo, dims, h):
    """the object's base program at lattice nodes lo .. lo + dims - 1 (as bands() evaluates it)"""
    pk = scene.pack()
    ii = [(np.arange(dims[a]) + lo[a]) * h for a in range(3)]
    Xn = np.stack(np.meshgrid(*ii, indexing='ij'), -1).reshape(-1, 3)
    out = np.empty(len(Xn))
    C._eval_range(pk.code, pk.prm, int(pk.oi_tab[oi, P.OI_C0]), int(pk.oi_tab[oi, P.OI_C1]), np.ascontiguousarray(Xn), out)
    return out.reshape(tuple(int(d) for d in dims))


def _apply_layer_sample_box(scene, oi, settings, c, rec, lut, h, r, s_det, s_gap, sig_shape, lo, dims):
    """the whole-box form (FS_LAYER_REGION=0): every node of the brush box, as before 11-14"""
    base, delta, _, _ = bands(scene, oi, lo, dims, s_det, filtered=False)
    coarse = _gauss(delta, s_gap / h)
    ii = [(np.arange(dims[a]) + lo[a]) * h for a in range(3)]
    X, Y, Z = np.meshgrid(*ii, indexing='ij')
    u = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / r
    w = np.interp(np.clip(u, 0, 1), np.linspace(0, 1, len(lut)), lut)
    w[u >= 1.0] = 0.0
    if settings.get('use_mask', True):
        m = C.gather_nodes(scene, oi, lo, dims, 'mask')
        w *= 1.0 - np.clip(m, 0.0, 1.0)
    st = float(settings['strength'])
    if settings['layer'] == 'detail':
        df = w * st * (coarse - delta)
    else:
        S = base + coarse
        Ss = _gauss(S, sig_shape / h)
        df = w * st * (Ss - S)
    df[w <= 0.0] = 0.0
    df = df.astype(np.float32).astype(np.float64)     # what the float32 pool will hold
    return C.write_nodes(scene, oi, lo, df, rec)


def apply_layer_stroke(scene, oid, settings, samples, rec):
    oi = scene.index_of(oid)
    obj = scene.objects[oi]
    lut = BR.falloff_lut(settings.get('falloff', 'smooth'), settings.get('curve'))
    sym = settings.get('symmetry', [False, False, False])
    for smp in samples:
        cl, nl, vl, s = BR.world_to_local(obj, smp['p'], smp.get('n', (0, 1, 0)))
        st = dict(settings)
        st['radius_local'] = float(settings['radius']) / s
        st['detail_scale_local'] = float(settings['detail_scale']) / s
        for sg in BR._symmetry_signs(sym):
            apply_layer_sample(scene, oi, st, cl * sg, rec, lut)


@command
class LayerSmooth(BrushStroke):
    """Smooth one layer and keep the other: settings['layer'] = 'detail' (smooth detail, keep shape) or 'shape'
    (smooth shape, keep detail). samples as for BrushStroke ({'p', 'n'} world)."""
    label = 'Layer smooth'

    def __init__(self, oid, settings, samples):
        self.oid = int(oid)
        s = copy.deepcopy(DEFAULT_LAYER_SETTINGS)
        s.update(copy.deepcopy(settings))
        if s['layer'] not in LAYERS:
            raise ValueError('layer must be one of %s' % (LAYERS,))
        self.settings = s
        self.samples = copy.deepcopy(samples)
        self.rec = None

    def do(self, scene):
        self.rec = BR.StrokeRecord()
        apply_layer_stroke(scene, self.oid, self.settings, self.samples, self.rec)
        scene.touch()

    def extend(self, scene, samples):
        samples = copy.deepcopy(samples)
        apply_layer_stroke(scene, self.oid, self.settings, samples, self.rec)
        self.samples.extend(samples)
        scene.touch()
