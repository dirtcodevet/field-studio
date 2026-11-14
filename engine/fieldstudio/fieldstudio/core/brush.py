"""Sculpt brushes on the sculpt layer (delta-F on the object's lattice).

Each brush sample computes, at every lattice node p inside the bounded support |p - c| < r, a new field value
F_new(p) from the current F (base + sculpt), and stores delta += (F_new - F_old) * (1 - mask(p)). Nodes outside
the support are untouched (bounded support). After each sample the per-brick Lipschitz bound of the sculpt grid
is re-measured from the stored nodes (exact for the trilinear interpolant), so the object's L stays tracked.

Strength is 0..1. Displacement brushes move the surface by up to A = strength * radius * 0.5 (E: exact only where
|grad F| = 1). Brush types and their field rules (c = centre, n = normal, w = falloff(|p - c| / r)):
  draw     F(p - n A w)                      clay     F + w (min(F, (p-c).n - A) - F)
  carve    F(p + n A w)                      inflate  F - A w
  smooth   F + s w (mean6(F, p +- delta) - F) flatten  F + s w ((p-c).n - F)
  crease   F(p + s w t / 2) + A w^3, t = tangential (p - c)
  pinch    F(p + s w t)                      grab     F(p - v w)       (v = drag vector)
  stretch  F(c + M^-1 (p - c)), M = I + (|v|/r) w a a^T, a = v/|v|
  mask     mask += s w (strength < 0 erases)
"""
import math
import os

import numpy as np
from numba import njit, prange

from . import program as P
from .program import STACK, eval_local, grid_sample, _node
from ..warp.profile import weight as _ft_w       # fs-tools 11-14: the approved tools' shared profile (warp/profile.py)

TOOLS = ['draw', 'clay', 'carve', 'inflate', 'smooth', 'flatten', 'crease', 'pinch', 'grab', 'stretch', 'mask',
         'swell', 'settle',          # 11-09: swell to contact, settle onto (core/collide.py; CPU path)
         'region',                   # fs-brush 11-11 (design 4.8): paints a region id (region_dab; no field change)
         'grabflow',                 # fs-brush 11-12 (toolkit review 4.2): the anchored flow Grab (core/history.py)
         'stretchflow',              # fs-brush 11-12 (toolkit review 4.3): Stretch as an anchored elongation (_stretch_back)
         'snakeflow',                # fs-brush 11-12 (Snake Hook's flat plug): Grab's path step + Magnify
         'snakepath',                # fs-brush 11-12: Snake Hook's whole path composed from the stroke-start field
         'ft_swell', 'ft_settle', 'ft_flatten',   # fs-tools 11-14: the approved brush tools (toolset/ftools.cuh rules;
                                     # held: one dab per timed sample, its amount = rate x dt; bp slots 51-63)
         'ft_pencil']                # fs-tools 11-14 (patch 4): Pencil, a line tool: the tip acts where it is at each dab
T = {name: i for i, name in enumerate(TOOLS)}
FT_TOOLS = ('ft_swell', 'ft_settle', 'ft_flatten')     # the HELD approved tools (Strength a rate: 'dt' per dab)
FT_LINE = ('ft_pencil',)            # the line tools: the tip acts where it is, each fixed spacing of travel, no 'dt'
FT_ALL = FT_TOOLS + FT_LINE         # every approved tool: sculpt.cu ft_dab, the press-time pool, bp slots 51-63
FT_T0 = T['ft_swell']               # the rules' codes are FT_T0 + 0 / 1 / 2 / 3 (gpu/sculpt.py compiles them into sculpt.cu)
LUT_N = 65

DEFAULT_SETTINGS = {'tool': 'draw', 'radius': 0.01, 'strength': 0.5, 'falloff': 'smooth', 'curve': None,
                    'spacing': 0.25, 'symmetry': [False, False, False], 'use_mask': True}


def falloff_lut(kind='smooth', curve=None):
    """w(u), u = d / r in [0, 1] -> LUT_N samples. curve: list of values at evenly spaced u (custom)."""
    u = np.linspace(0.0, 1.0, LUT_N)
    if kind == 'linear':
        w = 1.0 - u
    elif kind == 'sharp':
        w = (1.0 - u) ** 3
    elif kind == 'constant':
        w = np.where(u < 0.9, 1.0, (1.0 - u) / 0.1)
    elif kind == 'custom' and curve:
        cu = np.linspace(0.0, 1.0, len(curve))
        w = np.interp(u, cu, np.asarray(curve, float))
    else:                                   # smooth: 1 - smoothstep
        w = 1.0 - (3 * u * u - 2 * u * u * u)
    w = np.clip(w, 0.0, 1.0)
    w[-1] = 0.0
    return w.astype(np.float64)


@njit(cache=True, inline='always')
def _lut(lut, u):
    x = u * (lut.shape[0] - 1)
    i = int(x)
    if i >= lut.shape[0] - 1:
        return lut[lut.shape[0] - 1]
    t = x - i
    return lut[i] + (lut[i + 1] - lut[i]) * t



# r67 (11-09): small-dab strength. The falloff is written at lattice nodes and read back trilinearly, so a dab a few
# cells wide loses its peak (one Inflate dab, r = 2 h: 0.71-0.85 of the continuous rule's peak; r = 4 h: 0.93-0.97;
# random sub-cell centres, scratch experiment). Per dab, the lattice's own peak displacement on a plane is solved next
# to the continuous one (one 1-D bisection each) and the dab's amount is scaled by their ratio: 0.95-1.02 at r = 2 h
# for Inflate, Clay and Flatten in that experiment. Applied to A (draw, carve, inflate, swell, clay) or s (flatten).
GAIN_A_TOOLS = ('draw', 'carve', 'inflate', 'swell', 'clay')
GAIN_S_TOOLS = ('flatten',)
GAIN_LO, GAIN_HI = 0.8, 1.6


@njit(cache=True)
def _lat_w(lut, cx, cy, cz, px, py, pz, r, h):
    fx, fy, fz = px / h, py / h, pz / h
    i0, j0, k0 = math.floor(fx), math.floor(fy), math.floor(fz)
    tx, ty, tz = fx - i0, fy - j0, fz - k0
    acc = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                x = (i0 + a) * h - cx
                y = (j0 + b) * h - cy
                z = (k0 + c) * h - cz
                u = math.sqrt(x * x + y * y + z * z) / r
                w = _lut(lut, u) if u < 1.0 else 0.0
                acc += w * (tx if a else 1.0 - tx) * (ty if b else 1.0 - ty) * (tz if c else 1.0 - tz)
    return acc


@njit(cache=True)
def dab_gain(lut, cx, cy, cz, nx, ny, nz, r, A, h):
    """continuous peak / lattice peak of an Inflate-like dab (F - A w) on the plane through c with normal n."""
    if A <= 0.0 or r <= 0.0:
        return 1.0
    lo, hi = 0.0, A
    for _ in range(40):
        m = 0.5 * (lo + hi)
        u = m / r
        if m - A * (_lut(lut, u) if u < 1.0 else 0.0) < 0.0:
            lo = m
        else:
            hi = m
    tc = 0.5 * (lo + hi)
    lo, hi = 0.0, A
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if m - A * _lat_w(lut, cx, cy, cz, cx + m * nx, cy + m * ny, cz + m * nz, r, h) < 0.0:
            lo = m
        else:
            hi = m
    tl = 0.5 * (lo + hi)
    if tl <= 1e-15:
        return 1.0
    return min(max(tc / tl, GAIN_LO), GAIN_HI)


@njit(cache=True)
def flatten_gain(lut, cx, cy, cz, nx, ny, nz, r, h):
    """r67b: Flatten moves a curved surface by s w (dn - F) ~ s w rho^2 / 2R (rho: distance across the normal), largest
    part-way out, not at the centre: the continuous profile's maximum over the lattice's (trilinear of the nodes' w
    rho^2) over a polar sampling of the dab's plane."""
    ax, ay, az = (0.0, 0.0, 1.0) if abs(nz) < 0.9 else (1.0, 0.0, 0.0)
    ux, uy, uz = ny * az - nz * ay, nz * ax - nx * az, nx * ay - ny * ax
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / ul, uy / ul, uz / ul
    vx, vy, vz = ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux
    mc, ml = 0.0, 0.0
    for ir in range(1, 17):
        rho = r * ir / 17.0
        wc = _lut(lut, rho / r) * rho * rho
        if wc > mc:
            mc = wc
        for ia in range(12):
            a = 2.0 * math.pi * ia / 12.0
            px = cx + rho * (math.cos(a) * ux + math.sin(a) * vx)
            py = cy + rho * (math.cos(a) * uy + math.sin(a) * vy)
            pz = cz + rho * (math.cos(a) * uz + math.sin(a) * vz)
            fx, fy, fz = px / h, py / h, pz / h
            i0, j0, k0 = math.floor(fx), math.floor(fy), math.floor(fz)
            tx, ty, tz = fx - i0, fy - j0, fz - k0
            acc = 0.0
            for a_ in range(2):
                for b_ in range(2):
                    for c_ in range(2):
                        x = (i0 + a_) * h - cx
                        y = (j0 + b_) * h - cy
                        z = (k0 + c_) * h - cz
                        d2 = x * x + y * y + z * z
                        u = math.sqrt(d2) / r
                        dn = x * nx + y * ny + z * nz
                        w = _lut(lut, u) if u < 1.0 else 0.0
                        acc += w * (d2 - dn * dn) * (tx if a_ else 1.0 - tx) * (ty if b_ else 1.0 - ty) * (tz if c_ else 1.0 - tz)
            if acc > ml:
                ml = acc
    if ml <= 0.0:
        return 1.0
    return min(max(mc / ml, GAIN_LO), GAIN_HI)


# 11-09 night (many tools had far too strong an impact): a dab carried strength * r * 0.5, and
# dabs every spacing * r stack about 1 / spacing of them on each point of the path (the falloff's integral is 1/2 per
# side), so one pass at strength 0.5 built 0.7 - 0.8 r (phase0/tool_strength.py: 3 mm with a 4 mm brush; Clay 8 mm at
# strength 1). The amount is now set per PASS: one pass at strength s raises about AMOUNT_PER_PASS * s * r, whatever
# the spacing (a dab carries AMOUNT_PER_PASS * spacing * s * r). New strokes carry 'amount_pass'; older logs replay
# with the old amount (core/history.py from_log).
AMOUNT_PER_PASS = 0.2


def smooth_step(settings, h, r):
    """Smooth's stencil step. 11-09 night (a smooth over narrow folds laid a film of flesh over
    them): the step was r / 4 (2.5 mm at r 10 mm), a volume blur spanning the folds that fills any concavity. New
    strokes ('smooth_local') average at the lattice's own scale, 2 h, whatever the radius, as a surface smooth averages
    a point with its near neighbours: fine bumps go fast, large forms slowly, facing surfaces are not drawn together."""
    if settings.get('smooth_local', False):
        return 2.0 * h
    return max(2 * h, 0.25 * r)


def dab_amount(settings, r):
    """the displacement amount A of one dab (draw, carve, inflate, swell, clay, crease)"""
    s = abs(float(settings['strength']))
    if settings.get('amount_pass', False):
        return s * r * AMOUNT_PER_PASS * float(settings.get('spacing', 0.25))
    return s * r * 0.5


def gained(settings, tool, c, n, r, A, s, h, lut):
    """-> (A, s) with the r67 small-dab gain when settings['dab_gain'] (new strokes; older logs replay without it)."""
    if not settings.get('dab_gain', False) or (tool not in GAIN_A_TOOLS and tool not in GAIN_S_TOOLS):
        return A, s
    g = dab_gain(lut, float(c[0]), float(c[1]), float(c[2]), float(n[0]), float(n[1]), float(n[2]), float(r),
                 float(A), float(h))
    if tool in GAIN_A_TOOLS:
        return A * g, s
    g = flatten_gain(lut, float(c[0]), float(c[1]), float(c[2]), float(n[0]), float(n[1]), float(n[2]), float(r), float(h))
    s2 = s * g
    return A, (min(s2, 1.0) if s >= 0 else max(s2, -1.0))

FOLD_DOT = -0.6           # brush targeting: the gradient turning past ~127 deg between chord samples = a fold or gap
                          # (r: -0.25 cut jagged sheets whose facets differ by 120 deg; direction tags separate the
                          # opposite sheets of a gap for the Mask brush)
MASK_ANGLE_LOG = []         # per tagged Mask dab: the closest opposite-facing surface's angle (deg) or None
MASK_DIR_CUTOFF_DEG = 110.0   # direction-tagged mask: paint is not seen by surfaces facing more than this away
NBP = 64                  # fs-tools 11-14: 51-63 the approved brush tools (toolset/ftools.cuh FT_*: hardness, smoothing,
                          # foot, family, amount, follow / flat, gap, Stop at contact, plane point (3), h, the profile's
                          # radius; bp[7] is their whole ball). Before: 51. sculpt-contact 11-12: 46-50 Swell to contact (NBP_SWC: flag, gain, ease E, reach R, gap). Before: 46                  # fs-brush 11-11: 41-44 the mirror partner (NBP_MIRP); 38-40 Smooth's cell-scale term, normal stencil, fit off (NBP_CELL). 23-27 Smooth at the brush's scale (gpu/sculpt.cu quad_smooth), 28 the card's skip reach, 29-35 the area plane, 36 Pinch's bound, 37 the strength response (v2). Before: 23 slots (15-19: targeting, 20: Smooth keeps to its sheet, 21: keep apart,
                          # 22: Clay / Flatten move a node by at most the dab's amount)


def target_params(settings, h):
    """-> bp[15:20]: mode bits (1 brush-facing, 2 connected), cos of the cone's outer / inner edge, chord step, band.
    Missing keys = off, so strokes recorded before targeting replay unchanged."""
    mode = (1 if settings.get('face_only', False) else 0) | (2 if settings.get('connected', False) else 0)
    if settings.get('tool') == 'mask' and settings.get('paint_dir', False):
        mode |= 4                      # tagged paint: nodes facing away are painted too (their tag keeps it apart)
    cone = math.radians(min(max(float(settings.get('face_cone', 80.0)), 10.0), 170.0))
    ramp = math.radians(10.0)
    return [float(mode), math.cos(min(cone + ramp, math.pi)), math.cos(max(cone - ramp, 0.0)), float(h), 4.0 * float(h)]


# fs-brush 11-11 (design C1): Smooth at the brush's scale on the node's own sheet (gpu/sculpt.cu quad_smooth): the flag,
# the band it changes (in h), the two rings of the tangent plane (fractions of r), the most a dab moves the surface (in h),
# and the per-pass factor (one pass at strength 0.5 takes about half of a bump as wide as the brush, as Blender's default)
NBP_QSM = 23
NBP_PLANE = 29            # fs-brush 11-11: Clay / Flatten area plane (point 29-31, normal 32-34, flag 35; gpu/sculpt.cu area_plane)
NBP_STR = 37              # fs-brush 11-11: bp[37] > 0.5: bp[8] is already the per-dab factor (strength_dab); Pinch /
                          # Crease pull toward the stroke line (bp[9:12] = the stroke's direction)
NBP_PINCH = 36            # fs-brush 11-11: Pinch's per-dab factor bound (0.4: the forward map's Jacobian stays >= 0.5)
PINCH_K = 0.4
# fs-brush 11-11, decided: a low-strength smooth smooths less than a high-strength one; a low-strength
# crease, ridge or pencil indent has less impact than a high-strength one;
# the same across all of the tools. M (fs-bench strength sweep, base_1111): Grab
# and Stretch ignored strength, Pinch peaked at 0.5, Crease saturated from 0.75, the factors clamped at 1. Every factor
# that must stay below a limit per dab now approaches it as k_max (1 - exp(-k s_eff / k_max)): proportional at low
# strength (slope k), still rising at 2 (the slider's top), never past the limit. s_eff = s * spacing / 0.25: the
# same amount per pass whatever the spacing. Draw / Carve / Inflate / Clay / Swell were already proportional (A ~ s).
STRENGTH_RESP = {'smooth': (1.6, 1.0), 'flatten': (1.4, 1.0), 'pinch': (0.25, PINCH_K), 'crease': (0.15, 0.3)}
if os.environ.get('FSB_SMOOTH_RESP'):     # fs-brush 11-11: tuning switch (k,kmax) for Smooth's response, benches only
    STRENGTH_RESP['smooth'] = tuple(float(x) for x in os.environ['FSB_SMOOTH_RESP'].split(','))


def smooth_cell_bp(bp, settings):
    """fs-brush 11-11: Smooth's cell-scale term, normal stencil and fit switch from the stroke's settings (NBP_CELL)"""
    bp[NBP_CELL] = SMOOTH_CELL.get(settings.get('smooth_cell', '2h'), 0.0)
    bp[NBP_QSME] = float(settings.get('smooth_qse', 0.0))
    bp[NBP_QOFF] = 1.0 if settings.get('smooth_quad_off', False) else SMOOTH_FIT.get(settings.get('smooth_fit', 'plane'), 0.0)


def mirror_partner_bp(bp, settings):
    """fs-brush 11-11: the mirror feather's partner centre into bp (NBP_MIRP)"""
    mp = settings.get('mirror_partner')
    if mp is not None:
        bp[NBP_MIRP:NBP_MIRP + 3] = np.asarray(mp, float)
        bp[NBP_MIRP + 3] = 1.0


def feather_axis(settings, sym):
    """-> the one mirrored axis when the stroke feathers its mirror pair (new strokes), else None"""
    on = [a for a in range(3) if sym[a]]
    return on[0] if settings.get('mirror_feather', False) and len(on) == 1 else None


def mirror_plan(c, n, v, r, sym, settings, tool):
    """fs-brush 11-11: the dabs one sample makes under symmetry, with the mirror feather's partners:
    [(c, n, v, weight, partner centre or None)] (local frame).
    The midline snap (mirror_dabs, C4) stays for everything mirror-symmetric: a dab near the plane becomes one centred
    dab, so a midline form is single and not doubled (M, test1 t1_fsbrush_13: with the plain feathered pair, a nose
    pressed in the side view, whose pick lands a few mm off the midline, pulled two lobes at +-x, 0.47 of the drag at
    the midline). Its zeroed sideways drag (the snap cancelled a widening Grab: test1 'neck width' 0.03 of the drag)
    is carried by a FEATHERED PAIR at the original centres with only the drag's component across the plane, weighted
    as the snap's centred share (1 - t): both sides widen in full, and the midline itself doesn't move sideways.
    With the snap off ('mid_snap' False) the plain pair is feathered for every tool."""
    c = np.asarray(c, float); n = np.asarray(n, float); v = np.asarray(v, float)
    snap = bool(settings.get('mid_snap', False))
    if tool == 'smooth' and not settings.get('snap_smooth', True):
        # fs-brush 11-11 ~22:00 (test1 t1_fsbrush_15's granular jaw; t1fix/s3_variants.py, CPU, one Smooth band r 70 s 1.0
        # across the face): the snap moves every dab within 0.8 r of the plane onto it, so with a big brush a stroke ACROSS
        # the face collapses into one stack of dabs at the midline, and the jaw at its edge took ~15 weak passes of the
        # robust fit (left jaw roughness 1.72 -> 2.15; snap off 1.53 / 1.56, both sides even). Smoothing doubles nothing at
        # the midline, and the feathered pair already evens the overlap: new strokes ('snap_smooth' False) smooth with the
        # pair; logs without the key replay with the snap, as they ran.
        snap = False
    fax = feather_axis(settings, sym)
    base = mirror_dabs(c, n, v, r, sym, snap)
    if fax is None:
        return [(c_, n_, v_, w_, None) for c_, n_, v_, w_ in base]
    sg = np.ones(3); sg[fax] = -1.0
    if not snap:
        return [(c_, n_, v_, w_, c_ * sg) for c_, n_, v_, w_ in base]
    out = [(c_, n_, v_, w_, None) for c_, n_, v_, w_ in base]
    # fs-brush-4 11-13 (a live Grab, test 1's 'neck width' / 'jaw width' 0.000: since the anchored flow Grab became the
    # default (abf61389) the tool here is 'grabflow' / 'stretchflow', so the snap zeroed a sideways pull near the mirror
    # plane and this pair was never added): every move tool
    if tool in MOVE_TOOLS and abs(float(v[fax])) > 1e-12:
        d = abs(float(c[fax]))
        t = min(max((d - SNAP_LO * r) / max((1.0 - SNAP_LO) * r, 1e-12), 0.0), 1.0)
        if t < 1.0:
            va = np.zeros(3); va[fax] = v[fax]
            out.append((c, n, va, 1.0 - t, c * sg))
            out.append((c * sg, n * sg, va * sg, 1.0 - t, c))
    return out


def with_partner(st, c, ax):
    """st with 'mirror_partner' = c mirrored on axis ax (a copy; st itself is shared between dabs)"""
    pc = np.array(c, float)
    pc[ax] = -pc[ax]
    s2 = dict(st)
    s2['mirror_partner'] = pc
    return s2


def frac_of(settings):
    """fs-brush 11-11: a partial dab's share of a spacing step (sample 'f'; 1 for a whole dab). It scales the dab's
    amount linearly: strength_dab answers for the whole strength (strength / frac) and its factor is multiplied by frac,
    so a stroke deposits the same whether its frames ended on whole or partial dabs."""
    fr = float(settings.get('frac', 1.0))
    return fr if fr > 1e-6 else 1.0


def strength_dab(tool, s, spacing):
    """-> the per-dab factor of a strength s (signed; Ctrl inverts) for Smooth / Flatten / Pinch / Crease's pinch"""
    k, kmax = STRENGTH_RESP[tool]
    se = abs(float(s)) * float(spacing) / 0.25
    return math.copysign(kmax * (1.0 - math.exp(-k * se / kmax)), s)
NBP_SIDE = 45             # fs-brush 11-12: the CPU band skip's side (1 inside only, -1 outside only, 0 both)
NBP_SWC = 46              # sculpt-contact 11-12: bp[46] > 0.5: Swell to contact (core/collide.swell_k on the card)
NBP_SKIP = 28             # fs-brush 11-11: card dabs skip nodes farther than this from the surface (live re-distance on)
# not Grab / Stretch: they carry values across the whole ball; with the skip a 30 mm pull at r 60 followed 0.61 of the
# drag (0.97 without; M, probe_follow.py p19)
SKIP_TOOLS = ('draw', 'carve', 'inflate', 'clay', 'swell')
KA_SWEEPS_BY_MOVE = os.environ.get('FSB_KA_SWEEPS', '1') != '0'   # fs-brush-4 11-13: keep-apart's sweeps by the dab's move (keep_apart_sweeps)
SKIP_BAND_H = 6.0
BAND_ONE_SIDED = os.environ.get('FSB_BAND_ONE_SIDED', '1') != '0'   # fs-brush 11-12: the CPU band skips one side only
BAND_DEEP = float(os.environ.get('FSB_BAND_DEEP', '1'))   # fs-brush 11-12: bricks the one-sided band reaches past its band on the skipped side (1 exact in band/t_band_synth.py; 0 drifts)
BAND_BALL = os.environ.get('FSB_BAND_BALL', '0') == '1'   # fs-brush 11-12: the band's bricks only where the dab's ball reaches (OFF: it changes brush.redistance's percentile skip, test_brush_target BT5)
QSM_BAND_H, QSM_R1, QSM_R2, QSM_CAP_H = 6.0, 0.3, 0.6, 2.0
# fs-brush-3 11-13: the fit fades out where a ring's |height| reaches past its clamp (no height field there: a lid margin,
# a rim); (start, end) in units of the clamp; FSB_QSM_HV='a-b', '0-0' = off. The card's QSM_HV0 / QSM_HV1 (gpu/sculpt.cu)
QSM_HV = tuple(float(v) for v in os.environ.get('FSB_QSM_HV', '1.2-2.0').split('-'))
if os.environ.get('FSB_QSM_RINGS'):       # fs-brush 11-11: tuning switch for the fit's ring radii (x r), benches only
    QSM_R1, QSM_R2 = (float(x) for x in os.environ['FSB_QSM_RINGS'].split(','))
# fs-brush 11-11 (M, cpu_pits.py: a Smooth pass over the fold under a Grab-pulled lump left 1-2 mm pits; the old 2 h
# average alone left it clean): Smooth's cell-scale term. bp[NBP_CELL]: 0 = the six neighbours at 2 h (11-11 to now;
# a 2 h stencil couples only nodes of the same parity, so the eight interleaved sub-lattices are never averaged with
# each other), 1 = a binomial 3 x 3 x 3 low-pass at h ((1 2 1) / 4 on each axis: zero response at the lattice's own
# Nyquist), 2 = none. bp[NBP_QSME]: the fit's normal stencil in h (0: 1 h, the CPU's first rule; the card uses 2).
# bp[NBP_QOFF] > 0.5: the brush-scale fit off (diagnosis). Older logs carry none of them and replay as before.
NBP_CELL, NBP_QSME, NBP_QOFF = 38, 39, 40
SMOOTH_CELL = {'2h': 0.0, 'box': 1.0, 'none': 2.0}
SMOOTH_FIT = {'plane': 0.0, 'off': 1.0, 'robust': 2.0, 'robust2': 3.0}   # bp[NBP_QOFF]: the fit as 11-11 / off /
# _quad_fit_robust / the same with the same-sheet weight of ~22:00 (only a sheet facing away is another sheet)
QSM_PASS_K = 1.2
NBP_LIM = 22              # 11-10: bp[22] > 0: |change| per dab <= bp[13] for Clay and Flatten (a hollow filled in one dab)
NBP_TOPO = 21             # bp[21] > 0.5: surfaces kept apart (no join, film or tear; gpu/sculpt.cu dab_topo_commit)
TOPO_EPS = 0.05           # a refused node keeps its sign at +-TOPO_EPS * h
NBP_SHEET = 20            # bp[20] > 0.5: Smooth keeps to its own sheet (11-09 night); older logs replay without it
RIDGE_TOL = 0.3           # a step counts as turning back when its change exceeds this fraction of the step


@njit(cache=True, inline='always')
def _sheet_avg(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, f0, dl, h, vs, ps):
    """11-09 night (smoothing one fold pulled the facing fold over to it). Smooth's six samples at +-dl
    (dl = r / 4, 0.75 mm at r 3 mm) reached across her 0.5 mm gaps and through thin flaps, so a node on one sheet
    averaged in the facing sheet and the ridge between them. Each sample now walks out in lattice steps; where the
    field turns back (a medial ridge in a gap, or the middle of a thin flap) the sample is the node's own sheet carried
    on linearly instead. A node that itself sits on such a ridge is left as it is (NaN). Nothing crossed: the same
    six values as before."""
    K = max(1, int(math.ceil(dl / h - 1e-9)))
    st = dl / K
    tol = RIDGE_TOL * st
    acc = 0.0
    for ax in range(3):
        ex = 1.0 if ax == 0 else 0.0
        ey = 1.0 if ax == 1 else 0.0
        ez = 1.0 if ax == 2 else 0.0
        pp = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x + st * ex, y + st * ey, z + st * ez, vs, ps)
        pm = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x - st * ex, y - st * ey, z - st * ez, vs, ps)
        a1, b1 = pp - f0, pm - f0
        if a1 * b1 > 0.0 and min(abs(a1), abs(b1)) > tol:
            return math.nan
        g = (pp - pm) / (2.0 * st)
        for sg in (1.0, -1.0):
            v = pp if sg > 0 else pm
            d1 = v - f0
            crossed = False
            for t in range(2, K + 1):
                vn = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x + sg * t * st * ex,
                                y + sg * t * st * ey, z + sg * t * st * ez, vs, ps)
                d = vn - v
                if d * d1 < 0.0 and abs(d) > tol:
                    crossed = True
                    break
                v = vn
            acc += (f0 + sg * dl * g) if crossed else v
    return acc / 6.0


@njit(cache=True)
def _quad_smooth(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, f0, h, r, bp, vs, ps):
    """fs-brush 11-11: gpu/sculpt.cu quad_smooth on the CPU field (the same rule; see there). -> (a, ok)"""
    e = (bp[NBP_QSME] if bp.shape[0] > NBP_QSME and bp[NBP_QSME] > 0.0 else 1.0) * h
    gx = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x + e, y, z, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x - e, y, z, vs, ps)) / (2.0 * e)
    gy = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y + e, z, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y - e, z, vs, ps)) / (2.0 * e)
    gz = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z + e, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z - e, vs, ps)) / (2.0 * e)
    gl = math.sqrt(gx * gx + gy * gy + gz * gz)
    if bp.shape[0] > NBP_QOFF and bp[NBP_QOFF] > 1.5:
        return _quad_fit_robust(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, f0, h, r, bp, vs, ps,
                                gx, gy, gz, gl)
    if gl < 0.3:
        return 0.0, False
    nx, ny, nz = gx / gl, gy / gl, gz / gl
    d = f0 / gl
    px, py, pz = x - d * nx, y - d * ny, z - d * nz
    ax_ = 1.0 if abs(nx) < 0.9 else 0.0
    ay_ = 0.0 if ax_ > 0.5 else 1.0
    t1x, t1y, t1z = -nz * ay_, nz * ax_, nx * ay_ - ny * ax_
    tl = math.sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
    t1x, t1y, t1z = t1x / tl, t1y / tl, t1z / tl
    t2x, t2y, t2z = ny * t1z - nz * t1y, nz * t1x - nx * t1z, nx * t1y - ny * t1x
    m0 = 0.0
    m1 = 0.0
    c0 = 0
    c1 = 0
    for ring in range(2):
        rr = (bp[NBP_QSM + 2] if ring == 0 else bp[NBP_QSM + 3]) * r
        acc = 0.0
        n = 0
        for k in range(8):
            th = (k + 0.5 * ring) * 0.78539816
            c = math.cos(th) * rr
            sn = math.sin(th) * rr
            sx, sy, sz = px + c * t1x + sn * t2x, py + c * t1y + sn * t2y, pz + c * t1z + sn * t2z
            fs = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, sx, sy, sz, vs, ps)
            fn = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, sx + h * nx, sy + h * ny, sz + h * nz, vs, ps)
            if fn - fs < 0.3 * h * gl:
                continue
            acc += fs / gl
            n += 1
        if ring == 0:
            c0 = n
            m0 = acc / n if n else 0.0
        else:
            c1 = n
            m1 = acc / n if n else 0.0
    if c0 < 5 or c1 < 5:
        return 0.0, False
    r1 = (bp[NBP_QSM + 2] * r) ** 2
    r2 = (bp[NBP_QSM + 3] * r) ** 2
    return (r2 * m0 - r1 * m1) / (r2 - r1) * gl, True


@njit(cache=True)
def _quad_fit_robust(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, f0, h, r, bp, vs, ps, gx, gy, gz, gl):
    """fs-brush 11-11 (M, cpu_pits.py: the fit pitted the fold under a Grab-pulled lump; with the fit off it stayed
    clean): the brush-scale fit with every step continuous in the node's position, so neighbouring nodes agree.
    - each ring sample's HEIGHT is found along the normal (one Newton step from the plane: F at the plane point alone
      is a distance to whatever surface is nearest, which in a fold is the lump the plane cuts into);
    - the same-sheet test is taken at that surface point, as a weight (0 below a rise of 0.1 h, 1 above 0.5 h);
    - a height beyond half the ring radius counts as that much (a clamp: a fading weight made a terrace ring);
    - opposite samples are weighted as pairs (the weaker one's weight), so a lost sample never leaks the surface's tilt
      into the mean (unpaired weights made streaks);
    - the fit's amplitude fades in with the weaker ring's total weight (3 -> 5 samples) and with |grad F| (0.2 -> 0.4).
    M (cpu_pits.py, cpu_fitcheck.py): no pits, streaks or terraces in the fold; bumps W 4 / 12 / 40 lose 88 / 51 / 52 %.
    -> (a, ok) as _quad_smooth"""
    gam = min(max((gl - 0.2) / 0.2, 0.0), 1.0)
    if gam <= 0.0:
        return 0.0, False
    nx, ny, nz = gx / gl, gy / gl, gz / gl
    d = f0 / gl
    px, py, pz = x - d * nx, y - d * ny, z - d * nz
    ax_ = 1.0 if abs(nx) < 0.9 else 0.0
    ay_ = 0.0 if ax_ > 0.5 else 1.0
    t1x, t1y, t1z = -nz * ay_, nz * ax_, nx * ay_ - ny * ax_
    tl = math.sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
    t1x, t1y, t1z = t1x / tl, t1y / tl, t1z / tl
    t2x, t2y, t2z = ny * t1z - nz * t1y, nz * t1x - nx * t1z, nx * t1y - ny * t1x
    m0 = 0.0
    m1 = 0.0
    W0 = 0.0
    W1 = 0.0
    HG = np.empty(8)
    WG = np.empty(8)
    hrmax = 0.0
    for ring in range(2):
        rr = (bp[NBP_QSM + 2] if ring == 0 else bp[NBP_QSM + 3]) * r
        for k in range(8):
            th = (k + 0.5 * ring) * 0.78539816
            c = math.cos(th) * rr
            sn = math.sin(th) * rr
            sx, sy, sz = px + c * t1x + sn * t2x, py + c * t1y + sn * t2y, pz + c * t1z + sn * t2z
            fs = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, sx, sy, sz, vs, ps)
            t0 = -fs / gl                                  # the surface along the normal: one Newton step
            qx, qy, qz = sx + t0 * nx, sy + t0 * ny, sz + t0 * nz
            fq = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, qx, qy, qz, vs, ps)
            hgt = t0 - fq / gl                             # height of the surface above the plane (along n)
            fn = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, qx + h * nx, qy + h * ny, qz + h * nz, vs, ps)
            # fs-brush 11-11 ~22:00 ('robust2', bp > 2.5, new strokes; t1fix/fit_probe2-3.py, s3_variants.py): only a sheet
            # facing the other way is another sheet; a steep sample on the same sheet keeps its weight (the 0.1..0.5 ramp
            # gave 17-26 % of a curved jaw's samples partial weights that switched the pairs' min from node to node:
            # test1's granular jaw). 'robust' (older logs) replays as it was.
            if bp[NBP_QOFF] > 2.5:
                ws = min(max(((fn - fq) / (h * gl) + 0.1) / 0.3, 0.0), 1.0)
            else:
                ws = min(max(((fn - fq) / (h * gl) - 0.1) / 0.4, 0.0), 1.0)
            # a surface departing from the plane by more than half the ring radius counts as departing by that much
            # (a clamp, continuous; a fading weight made a terrace where it switched the fit off)
            hc = 0.5 * rr
            HG[k] = -min(max(hgt, -hc), hc)                 # + = the surface lies below the plane (as fs / gl)
            hrmax = max(hrmax, abs(hgt) / max(hc, 1e-12))   # fs-brush-3 11-13: the height against its clamp
            WG[k] = ws
        # opposite samples weighted as pairs (the weaker one's weight for both): the surface's tilt over the plane cancels
        # within each pair, so losing a sample on one side never leaks the tilt into the mean
        acc = 0.0
        wt = 0.0
        for k in range(4):
            wp = min(WG[k], WG[k + 4])
            acc += wp * (HG[k] + HG[k + 4])
            wt += 2.0 * wp
        if ring == 0:
            m0 = acc / wt if wt > 1e-9 else 0.0
            W0 = wt
        else:
            m1 = acc / wt if wt > 1e-9 else 0.0
            W1 = wt
    gam *= min(max((min(W0, W1) - 3.0) / 2.0, 0.0), 1.0)
    if QSM_HV[1] > QSM_HV[0]:                    # fs-brush-3 11-13: the height-field fade (gpu/sculpt.cu, the same rule)
        t_ = min(max((hrmax - QSM_HV[0]) / (QSM_HV[1] - QSM_HV[0]), 0.0), 1.0)
        gam *= 1.0 - t_ * t_ * (3.0 - 2.0 * t_)
    if gam <= 0.0:
        return 0.0, False
    r1 = (bp[NBP_QSM + 2] * r) ** 2
    r2 = (bp[NBP_QSM + 3] * r) ** 2
    return gam * (r2 * m0 - r1 * m1) / (r2 - r1) * gl, True


@njit(cache=True, inline='always')
def _grad_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, e, vs, ps):
    gx = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x + e, y, z, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x - e, y, z, vs, ps))
    gy = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y + e, z, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y - e, z, vs, ps))
    gz = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z + e, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z - e, vs, ps))
    return gx, gy, gz


@njit(cache=True)
def target_gate(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, f0, cx, cy, cz, nx, ny, nz, bp, vs, ps):
    """brush targeting weight in [0, 1] at node (x, y, z) (bp[15] mode, see target_params)"""
    mode = int(bp[15])
    hs = bp[18]
    gx, gy, gz = _grad_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, hs, vs, ps)
    gl = math.sqrt(gx * gx + gy * gy + gz * gz)
    if gl < 1e-12:                   # the gradient's DIRECTION decides, whatever its size: inside a narrow gap every
        return 1.0                   # node's gradient is weak but points away from the face it belongs to (r2/r3)
    gx /= gl
    gy /= gl
    gz /= gl
    fw = 1.0
    if (mode & 4) and gx * nx + gy * ny + gz * nz < 0.0:
        return 1.0
    if mode & 1:
        cs = gx * nx + gy * ny + gz * nz
        lo = bp[16]
        hi = bp[17]
        if cs <= lo:
            return 0.0
        if cs < hi:
            t = (cs - lo) / (hi - lo)
            fw = t * t * (3.0 - 2.0 * t)
    if (mode & 2) and abs(f0) <= bp[19]:
        px = x - f0 * gx
        py = y - f0 * gy
        pz = z - f0 * gz
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        if L > hs:
            K = min(12, int(math.ceil(L / (2.0 * hs))))   # r (app: 130-215 ms a Mask dab on her): samples at the
            es = 2.0 * hs                                  # sheet scale the chord is judged at (2 cells), not every cell                    # chord continuity judged at sheet scale (r: a 1-cell step flipped between the
                                             # facets of a rough / jagged sheet, MD2 coverage 0.16)
            ax, ay, az = _grad_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, cx, cy, cz, es, vs, ps)
            al = math.sqrt(ax * ax + ay * ay + az * az)
            if al > 1e-12:
                ax /= al
                ay /= al
                az /= al
            if mode & 4:
                # tagged paint (11-09): opposite-facing sheets are kept apart by the paint's direction tag, so the chord
                # only has to catch a SAME-facing fold: the chord leaves the brush's facing side (grad . n < -0.2) and
                # comes back to it (> 0.2). r: the turn test cut a jagged sheet whose dab centre sat 0.05 mm under the
                # facing sheet (the first chord samples read that sheet's gradient; MD2 coverage 0.35)
                same = al > 1e-12 and ax * nx + ay * ny + az * nz > 0.2
                away = False
                for i in range(1, K + 1):
                    t = i / K
                    bx, by, bz = _grad_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                             cx + dx * t, cy + dy * t, cz + dz * t, es, vs, ps)
                    bl = math.sqrt(bx * bx + by * by + bz * bz)
                    if bl < 1e-12:
                        continue
                    sd = (bx * nx + by * ny + bz * nz) / bl
                    if sd > 0.2:
                        if away:
                            return 0.0
                        same = True
                    elif sd < -0.2 and same:
                        away = True
                return fw
            for i in range(1, K + 1):
                t = i / K
                bx, by, bz = _grad_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                         cx + dx * t, cy + dy * t, cz + dz * t, es, vs, ps)
                bl = math.sqrt(bx * bx + by * by + bz * bz)
                if bl < 1e-12:
                    continue
                bx /= bl
                by /= bl
                bz /= bl
                if al > 1e-12 and ax * bx + ay * by + az * bz < FOLD_DOT:
                    return 0.0
                ax, ay, az, al = bx, by, bz, 1.0
    return fw


@njit(cache=True, inline='always', nogil=True)
def _scope_at(x, y, z, bp, mg):
    """fs-brush 11-12 (scope: Grab moves the pressed part's own faces together,
    the other sheet across a gap 0): the scope weight at a point, trilinear in the stroke-start grid mg (uint8, 255 = the
    pressed part's side; bp NBP_QSM..+3 its local origin and spacing, NBP_QSM + 4 > 0.5 on); 1 off the grid / off"""
    if bp[NBP_QSM + 4] < 0.5:
        return 1.0
    sp = bp[NBP_QSM + 3]
    fx = (x - bp[NBP_QSM]) / sp
    fy = (y - bp[NBP_QSM + 1]) / sp
    fz = (z - bp[NBP_QSM + 2]) / sp
    nx, ny, nz = mg.shape[0], mg.shape[1], mg.shape[2]
    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx > nx - 1.0 or fy > ny - 1.0 or fz > nz - 1.0:
        return 1.0
    i = min(int(fx), nx - 2)
    j = min(int(fy), ny - 2)
    k = min(int(fz), nz - 2)
    ax, ay, az = fx - i, fy - j, fz - k
    v = 0.0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                wgt = (ax if di else 1.0 - ax) * (ay if dj else 1.0 - ay) * (az if dk else 1.0 - az)
                v += wgt * mg[i + di, j + dj, k + dk]
    return v / 255.0


@njit(cache=True, inline='always', nogil=True)
def _scope_dist(ex, ey, ez, bp):
    """the falloff distance from the moving centre: plain, or with the pressed part's thickness T (bp[NBP_CELL]) taken
    out of the depth along the press normal (bp 4-6), so a part thinner than the brush moves through its thickness"""
    d2 = ex * ex + ey * ey + ez * ez
    T = bp[NBP_CELL]
    if T > 0.0:
        dn = ex * bp[4] + ey * bp[5] + ez * bp[6]
        dd = max(abs(dn) - T, 0.0)
        d2 = d2 - dn * dn + dd * dd
    return math.sqrt(max(d2, 0.0))


@njit(cache=True, inline='always', nogil=True)
def _flow_back(x, y, z, bp, lut, mg):
    """fs-brush 11-12 (toolkit review 4.2, 'moves as flows'): the anchored flow Grab. A kernel of radius rt whose centre
    travels from c0 to c0 + v (t in 0..1) carries space with it at the kernel's falloff times v: the press point follows
    the drag exactly, the map is one-to-one at any drag (a flow never folds), and it depends only on the total drag (a
    wiggle back to the start is the start). This walks a node back from t = 1 to t = 0 (midpoint steps): the point of the
    stroke-start field it now shows. bp: NBP_PLANE..+2 c0, +3 rt, +4 steps (the area-plane slots: unused by Grab)"""
    c0x, c0y, c0z = bp[NBP_PLANE], bp[NBP_PLANE + 1], bp[NBP_PLANE + 2]
    rt = bp[NBP_PLANE + 3]
    n = int(bp[NBP_PLANE + 4])
    vx, vy, vz = bp[9], bp[10], bp[11]
    dt = 1.0 / n
    for k in range(n, 0, -1):
        t1 = k * dt
        ex, ey, ez = x - (c0x + t1 * vx), y - (c0y + t1 * vy), z - (c0z + t1 * vz)
        d = _scope_dist(ex, ey, ez, bp)
        w1 = _lut(lut, d / rt) if d < rt else 0.0
        if w1 > 0.0:
            w1 *= _scope_at(x, y, z, bp, mg)
        xm, ym, zm = x - 0.5 * dt * w1 * vx, y - 0.5 * dt * w1 * vy, z - 0.5 * dt * w1 * vz
        tm = t1 - 0.5 * dt
        ex, ey, ez = xm - (c0x + tm * vx), ym - (c0y + tm * vy), zm - (c0z + tm * vz)
        d = _scope_dist(ex, ey, ez, bp)
        wm = _lut(lut, d / rt) if d < rt else 0.0
        if wm > 0.0:
            wm *= _scope_at(xm, ym, zm, bp, mg)
        x, y, z = x - dt * wm * vx, y - dt * wm * vy, z - dt * wm * vz
    return x, y, z


@njit(cache=True, inline='always', nogil=True)
def snake_back(x, y, z, bp, lut):
    """fs-brush 11-12 (Snake Hook made a flat-topped plug: the form was wrong, not pulled;
    Blender's Snake Hook, toolkit_review/A 1.16): one step of a PATH tool, as a flow: a kernel of radius rt carried from
    c0 to c0 + v (the step) moves space at its falloff (Grab's step; it keeps picking up what is under the brush now),
    plus Blender's Magnify: an outward velocity perpendicular to the step, magnify x |v| x w x (distance from the moving
    centre's line) / rt, which keeps the pulled form's girth (without it the form thins to a spike along a long pull;
    with a flat falloff it was a plug). Integrated backward along the step (midpoint), so the material at the moving
    centre follows the cursor exactly (s5: a first-order step lost 3.5 mm a straight 80 mm pull and the tip in a bend).
    -> the point of the field before the step that a node now shows. bp: NBP_PLANE..+2 c0, +3 rt, +4 steps, +5 magnify"""
    c0x, c0y, c0z = bp[NBP_PLANE], bp[NBP_PLANE + 1], bp[NBP_PLANE + 2]
    rt = bp[NBP_PLANE + 3]
    n = int(bp[NBP_PLANE + 4])
    vx, vy, vz = bp[9], bp[10], bp[11]
    vl = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vl < 1e-12 or n < 1:
        return x, y, z
    ax, ay, az = vx / vl, vy / vl, vz / vl
    m = bp[NBP_PLANE + 5] * vl / rt
    ox, oy, oz = bp[NBP_CELL], bp[NBP_CELL + 1], bp[NBP_CELL + 2]   # Rake: the step's turn (axis x angle)
    dt = 1.0 / n
    for k in range(n, 0, -1):
        t1 = k * dt
        ex, ey, ez = x - (c0x + t1 * vx), y - (c0y + t1 * vy), z - (c0z + t1 * vz)
        d = math.sqrt(ex * ex + ey * ey + ez * ez)
        w1 = _lut(lut, d / rt) if d < rt else 0.0
        ta = ex * ax + ey * ay + ez * az
        ux = w1 * (vx + m * (ex - ax * ta) + oy * ez - oz * ey)
        uy = w1 * (vy + m * (ey - ay * ta) + oz * ex - ox * ez)
        uz = w1 * (vz + m * (ez - az * ta) + ox * ey - oy * ex)
        xm, ym, zm = x - 0.5 * dt * ux, y - 0.5 * dt * uy, z - 0.5 * dt * uz
        tm = t1 - 0.5 * dt
        ex, ey, ez = xm - (c0x + tm * vx), ym - (c0y + tm * vy), zm - (c0z + tm * vz)
        d = math.sqrt(ex * ex + ey * ey + ez * ez)
        wm = _lut(lut, d / rt) if d < rt else 0.0
        ta = ex * ax + ey * ay + ez * az
        x = x - dt * wm * (vx + m * (ex - ax * ta) + oy * ez - oz * ey)
        y = y - dt * wm * (vy + m * (ey - ay * ta) + oz * ex - ox * ez)
        z = z - dt * wm * (vz + m * (ez - az * ta) + ox * ey - oy * ex)
    return x, y, z


SNAKE_ROW = 12            # a composed Snake step: c0 (3), v (3), rt, substeps, magnify, Rake omega (3) - local frame


@njit(cache=True, inline='always', nogil=True)
def snake_path_back(x, y, z, steps, lut):
    """fs-brush 11-12 (tk3 / s5 at 1 mm: a path tool that re-reads its own field every step diffuses its form: an 80 mm
    arm thinned 8 -> 3.5-5.5 mm and a bend lost its end): Snake Hook's WHOLE path composed: a node walks back through
    every step of the path (the latest first), each step the flow of snake_back (the kernel carried c0 -> c0 + v,
    Magnify, Rake), to the point of the STROKE-START field it now shows: one read a batch, nothing accumulates, and the
    path still shapes the form. A step whose kernel never comes within rt of the node is skipped. steps: (n, SNAKE_ROW)"""
    for s in range(steps.shape[0] - 1, -1, -1):
        c0x, c0y, c0z = steps[s, 0], steps[s, 1], steps[s, 2]
        vx, vy, vz = steps[s, 3], steps[s, 4], steps[s, 5]
        rt = steps[s, 6]
        n = int(steps[s, 7])
        vl = math.sqrt(vx * vx + vy * vy + vz * vz)
        if vl < 1e-12 or n < 1:
            continue
        ax, ay, az = vx / vl, vy / vl, vz / vl
        qx, qy, qz = x - c0x, y - c0y, z - c0z         # the node's distance from the step's segment
        tq = min(max(qx * ax + qy * ay + qz * az, 0.0), vl)
        ex, ey, ez = qx - ax * tq, qy - ay * tq, qz - az * tq
        if ex * ex + ey * ey + ez * ez >= rt * rt:
            continue
        m = steps[s, 8] * vl / rt
        ox, oy, oz = steps[s, 9], steps[s, 10], steps[s, 11]
        dt = 1.0 / n
        for k in range(n, 0, -1):
            t1 = k * dt
            ex, ey, ez = x - (c0x + t1 * vx), y - (c0y + t1 * vy), z - (c0z + t1 * vz)
            d = math.sqrt(ex * ex + ey * ey + ez * ez)
            w1 = _lut(lut, d / rt) if d < rt else 0.0
            ta = ex * ax + ey * ay + ez * az
            ux = w1 * (vx + m * (ex - ax * ta) + oy * ez - oz * ey)
            uy = w1 * (vy + m * (ey - ay * ta) + oz * ex - ox * ez)
            uz = w1 * (vz + m * (ez - az * ta) + ox * ey - oy * ex)
            xm, ym, zm = x - 0.5 * dt * ux, y - 0.5 * dt * uy, z - 0.5 * dt * uz
            tm = t1 - 0.5 * dt
            ex, ey, ez = xm - (c0x + tm * vx), ym - (c0y + tm * vy), zm - (c0z + tm * vz)
            d = math.sqrt(ex * ex + ey * ey + ez * ez)
            wm = _lut(lut, d / rt) if d < rt else 0.0
            ta = ex * ax + ey * ay + ez * az
            x = x - dt * wm * (vx + m * (ex - ax * ta) + oy * ez - oz * ey)
            y = y - dt * wm * (vy + m * (ey - ay * ta) + oz * ex - ox * ez)
            z = z - dt * wm * (vz + m * (ez - az * ta) + ox * ey - oy * ex)
    return x, y, z


def snake_path_steps(obj, samples, settings):
    """fs-brush 11-12: Snake Hook's composed path per mirror copy: [(n, SNAKE_ROW) local arrays], in mirror_plan's copy
    order, from ALL the stroke's samples (each dab's step, its Rake turns, Magnify); the object's own frame only
    (sym_world strokes take no copies but the first)"""
    sym = settings.get('symmetry', [False, False, False])
    h = float(obj.sculpt.h)
    mag = float(settings.get('snake_magnify', SNAKE_MAGNIFY))
    copies = {}
    bp_ = np.zeros(NBP)
    for smp in samples:
        c, n, v, s = world_to_local(obj, smp['p'], smp.get('n', (0, 1, 0)), smp.get('v', (0, 0, 0)))
        r_l = max(float(settings['radius']) / s, 2 * h)
        vps = snake_vps(obj, smp, c, n, r_l, sym, settings, 'grab')
        for ci, (c_, n_, v_, w_, pc_) in enumerate(mirror_plan(c, n, v, r_l, sym, settings, 'grab')):
            vv = np.asarray(v_, float) * w_
            L = float(np.linalg.norm(vv))
            if L < 1e-12:
                continue
            st_ = dict(settings, snake_vp=vps[ci][0], snake_vr=vps[ci][1]) if vps is not None else settings
            snake_bp(bp_, st_, c_, r_l, vv)
            copies.setdefault(ci, []).append(np.r_[c_, vv, r_l, bp_[NBP_PLANE + 4], mag, bp_[NBP_CELL:NBP_CELL + 3]])
    return [np.ascontiguousarray(np.array(rows, np.float64)) for _, rows in sorted(copies.items())]


def snake_path_ball(steps):
    """the ball (centre, radius) holding a composed path's every kernel: its segments' bounds + the radius"""
    P = np.concatenate([steps[:, 0:3], steps[:, 0:3] + steps[:, 3:6]])
    c = 0.5 * (P.min(0) + P.max(0))
    return c, float(np.max(np.linalg.norm(P - c, axis=1)) + steps[:, 6].max())


def snake_path_near(ctr, steps, reach):
    """which points (n, 3) lie within reach + the kernel radius of any of the path's segments"""
    keep = np.zeros(len(ctr), bool)
    for s in steps:
        c0, v = s[0:3], s[3:6]
        L2 = float(v @ v)
        t = np.clip(((ctr - c0) @ v) / max(L2, 1e-24), 0.0, 1.0)
        d = np.linalg.norm(ctr - c0 - t[:, None] * v[None, :], axis=1)
        keep |= d <= s[6] + reach
    return keep


def snake_dab(c, v, r):
    """fs-brush 11-12: a Snake Hook step's dab: the ball (centre, radius) over the kernel's travel c -> c + v, as the
    anchored Grab's (anchored_sample): -> (centre, radius)"""
    L = float(np.linalg.norm(v))
    return np.asarray(c, float) + 0.5 * np.asarray(v, float), r + 0.5 * L


def snake_bp(bp, settings, c0, rt, v):
    """fs-brush 11-12: a Snake Hook dab's slots (the stroke's tool is 'grab' with snake=True; ui/engine.py), the same on
    the CPU (apply_sample) and the card (gpu/sculpt.py _plan): c0 the step's start, rt the brush radius"""
    L = float(np.linalg.norm(v))
    bp[0] = T['snakeflow']
    bp[NBP_PLANE:NBP_PLANE + 3] = c0
    bp[NBP_PLANE + 3] = rt
    bp[NBP_PLANE + 4] = min(16, max(2, int(math.ceil(L / (0.15 * rt)))))
    bp[NBP_PLANE + 5] = float(settings.get('snake_magnify', SNAKE_MAGNIFY))
    bp[NBP_PLANE + 6] = 0.0
    bp[NBP_CELL:NBP_CELL + 3] = 0.0
    vp, vr = settings.get('snake_vp'), settings.get('snake_vr')
    if vp is not None and vr is not None:       # Rake: the carried region turns by the path's turn (snake_stamp)
        vp, vr = np.asarray(vp, float), np.asarray(vr, float)
        lp, lr = float(np.linalg.norm(vp)), float(np.linalg.norm(vr))
        if lp > 1e-12 and lr > 1e-12:
            ax_ = np.cross(vp / lp, vr / lr)
            sn = float(np.linalg.norm(ax_))
            cs = float(np.dot(vp / lp, vr / lr))
            if sn > 1e-9:
                bp[NBP_CELL:NBP_CELL + 3] = ax_ / sn * math.atan2(sn, cs) * float(settings.get('snake_rake', SNAKE_RAKE))


GRAB_SCOPE = os.environ.get('FSB_GRAB_SCOPE', '1') != '0'
SCOPE_SHEET = 0.5          # a pressed part thinner than this x the brush radius moves through its thickness
SCOPE_DIV = float(os.environ.get('FSB_SCOPE_DIV', '24'))   # the scope grid's spacing: max(h, r / SCOPE_DIV)
SCOPE_BOX = 1.5            # the scope grid covers +- SCOPE_BOX r about the press point
_SCOPE_CACHE = {}


def scope_mask(scene, oid, c0, n0, r):
    """fs-brush 11-12 (scope: Grab moves the PRESSED part's own faces together, a
    lid's front and back, and leaves the other sheet across a gap, the eye, alone): from the stroke-start field over a
    box of 2 r about the press point (object frame): the inside region's connected parts (26-connected), the pressed
    part = the one just inside the press point; every grid node's weight 255 where it is nearer the pressed part than
    any other part, 0 where nearer another, a 2-node ramp between. -> {'mg', 'origin', 'sp', 'T'} (local), or None when
    the box holds one part only (nothing to keep apart) and the part is thicker than SCOPE_SHEET r. Cached per stroke
    (the press point); the stroke-start field is the CPU field, untouched during a card stroke"""
    key = (id(scene), oid, tuple(np.round(np.asarray(c0, float), 9)), round(float(r), 9))
    if key in _SCOPE_CACHE:           # fs-brush-3 11-13: the stroke's own scope, from its stroke-start field, before the
        return _SCOPE_CACHE[key]      # staleness rule (its release ran without scope: the slit's far plate moved 0.50 mm)
    if CPU_STALE:                     # fs-brush-2 11-13: the CPU field is behind the card: no scope until it lands
        return None
    from scipy import ndimage
    from ..tools.protected_merge import field_of, _to_world
    _SCOPE_CACHE.clear()
    o = scene.get(oid)
    h = float(o.sculpt.h)
    cl, nl, _, s = world_to_local(o, c0, n0, None)
    rl = r / s
    # tk2_card6: a 2 r box at r / 40 (4 M points: the field, labels, two distance transforms) stalled the press (New
    # bust Grab r 120: 28.9 -> 5.0 fps; the slit 1.2 fps): 1.5 r at max(h, r / 24), at most 72^3 points; a gap
    # narrower than the spacing is one part to a brush that big
    sp = max(h, rl / SCOPE_DIV)
    R = SCOPE_BOX * rl
    m = int(math.ceil(2 * R / sp)) + 1
    org = cl - R
    ax = org[None, :] + np.arange(m)[:, None] * sp
    G = np.stack(np.meshgrid(ax[:, 0], ax[:, 1], ax[:, 2], indexing='ij'), -1).reshape(-1, 3)
    pk = scene.pack()
    F = np.asarray(field_of(scene, pk, o, np.ascontiguousarray(_to_world(o, G)))).reshape(m, m, m) / s
    ins = F < 0
    lab, nlab = ndimage.label(ins, structure=np.ones((3, 3, 3), bool))
    T = form_depth(scene, oid, c0, np.asarray(n0, float), np.asarray(n0, float), max(h, 0.0005)) / s
    sheet = T if (0.0 < T < SCOPE_SHEET * rl) else 0.0
    pl = 0
    for tt in np.arange(0.5, 6.0, 0.5):                 # the pressed part: just inside the press point
        q = np.round((cl - tt * sp * np.asarray(nl, float) - org) / sp).astype(int)
        if np.all(q >= 0) and np.all(q < m) and lab[tuple(q)] > 0:
            pl = int(lab[tuple(q)])
            break
    if pl == 0 or (nlab <= 1 and sheet <= 0.0):
        _SCOPE_CACHE[key] = None
        return None
    if nlab <= 1:
        mg = np.full((m, m, m), 255, np.uint8)
    else:
        dp = ndimage.distance_transform_edt(lab != pl) * sp
        do = ndimage.distance_transform_edt(~((lab > 0) & (lab != pl))) * sp
        mg = np.clip(127.5 + 127.5 * (do - dp) / (0.5 * sp), 0, 255).astype(np.uint8)   # a 1-node ramp (tk2_card7: 2 nodes reached the slit's lower face, 0.36 mm)
    out = {'mg': np.ascontiguousarray(mg), 'origin': org, 'sp': sp, 'T': sheet, 'parts': int(nlab)}
    _SCOPE_CACHE[key] = out
    return out


def scope_bp(bp, settings):
    """fill a grabflow dab's scope slots from settings['scope'] -> its grid (uint8) or None (scope off)"""
    sc = settings.get('scope')
    if not sc:
        bp[NBP_QSM + 4] = 0.0
        bp[NBP_CELL] = 0.0
        return None
    bp[NBP_QSM:NBP_QSM + 3] = sc['origin']
    bp[NBP_QSM + 3] = sc['sp']
    bp[NBP_QSM + 4] = 1.0 if sc['parts'] > 1 else 0.0
    bp[NBP_CELL] = sc['T']
    return sc['mg']


def stretch_bp(bp, c, v, r, settings):
    """fs-brush 11-12: _stretch_back's slots from the dab (c = the ball's centre, r its radius, v the total drag; the
    depth, the axis falloff radius and the fade come as ratios of r: the same in any frame)"""
    L = float(np.linalg.norm(v))
    a = np.asarray(v, float) / max(L, 1e-12)
    D = r * float(settings.get('flow_D_ratio', 1.0))
    fade = r * float(settings.get('flow_fade_ratio', 1.0))
    bp[29:32] = np.asarray(c, float) - a * 0.5 * (D + fade)
    bp[32] = D
    bp[33] = r * float(settings.get('flow_ratio', 1.0))
    bp[23] = fade


CPU_STALE = False      # fs-brush-2 11-13: set by gpu/sculpt.py while an earlier stroke's write-back is behind its release


def form_depth(scene, oid, c0, a, n0, step):
    """fs-brush 11-12: how far the form reaches behind the press point c0 against the stretch axis a (world): the
    first exit along -a from just inside the surface (c0 - 2 step n0); 0 when the line leaves the form at once"""
    if CPU_STALE:                     # the CPU field is behind the card here: the depth's bound (anchored_sample) instead
        return 1e3
    from ..tools.protected_merge import field_of
    o = scene.get(oid)
    pk = scene.pack()
    start = np.asarray(c0, float) - 2.0 * step * np.asarray(n0, float)
    ts = np.arange(0.0, 2.0, step)
    P = start[None, :] - ts[:, None] * np.asarray(a, float)[None, :]
    f = field_of(scene, pk, o, np.ascontiguousarray(P))
    out = np.nonzero(f > 0.0)[0]
    if not len(out):
        return float(ts[-1])
    return float(ts[out[0]])


def anchored_sample(scene, oid, settings, c0, n0, vtot):
    """fs-brush 11-12 (toolkit review 4.2): the one dab an anchored Grab / Stretch applies for the total drag so far:
    -> (settings, sample) or None. Grab: a ball over the flow's capsule. Stretch: the pivot behind the press point at
    the form's depth along the axis (at least the brush radius: a pull along the surface stretches a brush-sized
    piece), a ball over the cylinder from the pivot to the fade's end"""
    L = float(np.linalg.norm(vtot))
    if L < 1e-9:
        return None
    r = float(settings['radius'])
    c0 = np.asarray(c0, float)
    if settings.get('tool') == 'grab':
        st = dict(settings, tool='grabflow', radius=r + 0.5 * L, flow_ratio=r / (r + 0.5 * L), autosmooth=0.0)
        if GRAB_SCOPE and settings.get('grab_scope', True):
            sc_ = scope_mask(scene, oid, c0, n0, r)
            if sc_ is not None:
                st['scope'] = sc_
        return st, {'p': (c0 + 0.5 * vtot).tolist(), 'n': list(n0), 'v': list(vtot)}
    a = np.asarray(vtot, float) / L
    h = float(scene.get(oid).sculpt.h) * max(float(scene.get(oid).transform.get('scale', 1.0)), 1e-9)
    D = form_depth(scene, oid, c0, a, n0, max(h, 0.0005))
    if STRETCH_DK > 0.0:
        # fs-brush 11-12 (design: the pivot depth is bounded, and a Stretch used large enough stretches the
        # full shape): the pivot no deeper than the brush's own radius x STRETCH_DK, so a
        # large brush stretches the whole shape and a small one stays local; at least a quarter radius
        D = min(max(D, 0.25 * r), STRETCH_DK * r)
    else:
        D = max(D, r)                         # before: the whole form's depth behind the press point
    fade = max(2.0 * L, r)
    half = 0.5 * (D + fade)
    rb = math.sqrt(half * half + r * r) + 2.0 * h
    st = dict(settings, tool='stretchflow', radius=rb, flow_ratio=r / rb, flow_D_ratio=D / rb,
              flow_fade_ratio=fade / rb, autosmooth=0.0)
    P = c0 - a * D
    return st, {'p': (P + a * half).tolist(), 'n': list(n0), 'v': list(vtot)}


@njit(cache=True, inline='always', nogil=True)
def _stretch_back(x, y, z, bp, lut):
    """fs-brush 11-12 (toolkit review 4.3: Stretch must stretch the whole region worked on): the
    region is elongated along the axis a about a pivot P on the far side of the form (D behind the press point): along
    a, the stretch from P to the press point grows from D to D + Le (Le = the drag's length L times the brush falloff of
    the distance from the axis, so the ring's edge stays still), the pressed end moves Le (the press point: L, it
    follows the drag), nothing behind the pivot moves, and beyond the pressed end the move fades out over 'fade'. Each
    piece is linear and increasing along a, so the map is one-to-one at any drag (no crater, no inner surface); this is
    its exact inverse: the stroke-start point a node now shows. bp: 29-31 P, 32 D, 33 r (axis falloff), 23 fade, 9-11 v"""
    Px, Py, Pz = bp[29], bp[30], bp[31]
    D = bp[32]
    rr = bp[33]
    fade = bp[23]
    vx, vy, vz = bp[9], bp[10], bp[11]
    L = math.sqrt(vx * vx + vy * vy + vz * vz)
    if L < 1e-12:
        return x, y, z
    ax, ay, az = vx / L, vy / L, vz / L
    ex, ey, ez = x - Px, y - Py, z - Pz
    xa = ex * ax + ey * ay + ez * az
    if xa <= 0.0:
        return x, y, z
    qx, qy, qz = ex - xa * ax, ey - xa * ay, ez - xa * az
    rho = math.sqrt(qx * qx + qy * qy + qz * qz)
    if rho >= rr:
        return x, y, z
    Le = L * _lut(lut, rho / rr)
    if Le <= 0.0:
        return x, y, z
    if xa <= D + Le:                             # forward: x = x0 (D + Le) / D on 0..D
        xa0 = xa * D / (D + Le)
    elif xa <= D + fade:                         # forward: x = x0 + Le (1 - (x0 - D) / fade) on D..D+fade (fade > Le)
        xa0 = D + (xa - D - Le) * fade / (fade - Le)
    else:
        xa0 = xa
    d_ = xa0 - xa
    return x + d_ * ax, y + d_ * ay, z + d_ * az


@njit(cache=True, nogil=True)
def _ft_feather(bp, w, px, py, pz, lateral, nx, ny, nz, r, hard, sm, fall, fam):
    """fs-tools 11-14: the mirror feather for the approved tools (toolset/ftools.cuh ft_feather; dab_compute's pair rule)"""
    if not (bp[NBP_MIRP + 3] > 0.5) or w <= 0.0:
        return w
    ex, ey, ez = px - bp[NBP_MIRP], py - bp[NBP_MIRP + 1], pz - bp[NBP_MIRP + 2]
    if lateral:
        t = ex * nx + ey * ny + ez * nz
        ex, ey, ez = ex - t * nx, ey - t * ny, ez - t * nz
    w2 = _ft_w(math.sqrt(ex * ex + ey * ey + ez * ez), r, hard, sm, fall, fam)
    if w2 <= 0.0:
        return w
    return w * (w * w + w2 * w2) / ((w + w2) * (w + w2))


@njit(cache=True, nogil=True)
def _ft_facing(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, hh, vs, ps):
    """fs-tools 11-14 (patch 4): the unit facing at a point, central differences at +-hh (toolset/ftools.cuh ft_facing)"""
    gx = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x + hh, y, z, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x - hh, y, z, vs, ps))
    gy = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y + hh, z, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y - hh, z, vs, ps))
    gz = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z + hh, vs, ps)
          - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z - hh, vs, ps))
    gl = max(math.sqrt(gx * gx + gy * gy + gz * gz), 1e-12)
    return gx / gl, gy / gl, gz / gl


def _ft_press_frame(scene, oi, rec, c, ftv, h):
    """fs-tools patch 4: the Pencil tip's frame on the field AS IT WAS AT THE PRESS, for the CPU path (the card reads its
    press-time pool, sculpt.cu obj_f_start): the stroke's own undo snapshot of the bricks around the tip is put back
    for the reads, then restored. -> {slot: value}: q (bp 9-11), n (15-17), the depth limit (18), 1 at 19, t (59-61).
    The same arithmetic as toolset/ftools.cuh ft_pencil (four projections along the facing, the facing at +-h, the
    part's thickness inward). Without it the twin framed the tip in the groove it had just cut: 0.5 mm lines came out
    1.2-2.8 mm deep (fs-tools s3/cpu.py)."""
    from ..toolset.common import field_local
    o = scene.objects[oi]
    g = o.sculpt
    st = scene.store
    r_tip, depth = float(ftv[63]), float(ftv[55])
    ridge = int(ftv[56]) >= 4
    reach = r_tip + 4.0 * depth + 8.0 * h
    c = np.asarray(c, float)
    blo = np.floor((c - reach) / h).astype(np.int64) >> 3
    bhi = np.floor((c + reach) / h).astype(np.int64) >> 3
    near = lambda b: all(blo[i] <= b[i] <= bhi[i] for i in range(3))
    swaps = []
    try:
        done = set()
        for (oid, b), val in (rec.snap.items() if rec is not None else ()):
            if oid == o.id and near(b):
                sl = g.get_slot(b)
                if sl >= 0:
                    swaps.append((sl, st.delta[sl].copy()))
                    st.delta[sl] = val[0]
                    done.add(b)
        for (oid, b) in (rec.new if rec is not None else ()):
            if oid == o.id and near(b) and b not in done:
                sl = g.get_slot(b)
                if sl >= 0:
                    swaps.append((sl, st.delta[sl].copy()))
                    st.delta[sl] = 0.0                  # a brick the stroke made: the base field at the press
        F = lambda P: field_local(scene, o, np.asarray(P, float).reshape(-1, 3))
        E = np.eye(3) * h

        def facing(p):
            v = F(np.concatenate([p + E, p - E]))
            gv = v[:3] - v[3:]
            return gv / max(float(np.linalg.norm(gv)), 1e-12)
        n = facing(c)
        q = c.copy()
        for _ in range(4):
            q = q - float(F(q)[0]) * n
        n = facing(q)
        t = np.array([ftv[59], ftv[60], ftv[61]], float)
        t = t - (t @ n) * n
        if np.linalg.norm(t) > 1e-9:
            t = t / np.linalg.norm(t)
        else:
            a_ = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t = a_ - (a_ @ n) * n
            t = t / max(np.linalg.norm(t), 1e-12)
        dl = depth
        if not ridge:
            tmax = 4.0 * depth + 4.0 * h
            s_ = 0.5 * h
            th = tmax
            while s_ < tmax:
                fv = float(F(q - s_ * n)[0])
                if fv > 0.0:
                    th = s_
                    break
                s_ += max(abs(fv), 0.25 * h)
            dl = min(depth, 0.5 * th)
    finally:
        for sl, v in reversed(swaps):
            st.delta[sl] = v
    out = {9: q[0], 10: q[1], 11: q[2], 15: n[0], 16: n[1], 17: n[2], 18: dl, 19: 1.0, 59: t[0], 60: t[1], 61: t[2]}
    return {k: float(v) for k, v in out.items()}


@njit(cache=True, nogil=True)
def _ft_pencil(code, prm, oi_tab, of_tab, index_flat, delta, oi, bp, x, y, z, f0, vs, ps):
    """fs-tools 11-14 (patch 4): the Pencil tip acting at a node (toolset/ftools.cuh ft_pencil, the same arithmetic) on the
    CPU field AS IT IS (the card reads the press-time pool; this twin runs only with the card off, where a stroke
    crossing itself reads its own groove)"""
    hh = bp[62]
    r = bp[63]
    depth = bp[55]
    k = bp[52]
    md = int(bp[56])
    ridge = md >= 4
    tip = md & 3
    if bp[19] > 0.5:                             # the frame read on the press-time field (_ft_press_frame)
        return _ft_stamp(bp, bp[9], bp[10], bp[11], bp[15], bp[16], bp[17], bp[59], bp[60], bp[61], bp[18], x, y, z, f0)
    qx, qy, qz = bp[1], bp[2], bp[3]
    nx, ny, nz = _ft_facing(code, prm, oi_tab, of_tab, index_flat, delta, oi, qx, qy, qz, hh, vs, ps)
    for _ in range(4):
        v = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, qx, qy, qz, vs, ps)
        qx, qy, qz = qx - v * nx, qy - v * ny, qz - v * nz
    nx, ny, nz = _ft_facing(code, prm, oi_tab, of_tab, index_flat, delta, oi, qx, qy, qz, hh, vs, ps)
    tx, ty, tz = bp[59], bp[60], bp[61]
    tn = tx * nx + ty * ny + tz * nz
    tx, ty, tz = tx - tn * nx, ty - tn * ny, tz - tn * nz
    tl = math.sqrt(tx * tx + ty * ty + tz * tz)
    if tl > 1e-9:
        tx, ty, tz = tx / tl, ty / tl, tz / tl
    else:
        a0 = 1.0 if abs(nx) < 0.9 else 0.0
        a1 = 0.0 if abs(nx) < 0.9 else 1.0
        an = a0 * nx + a1 * ny
        tx, ty, tz = a0 - an * nx, a1 - an * ny, -an * nz
        tl = max(math.sqrt(tx * tx + ty * ty + tz * tz), 1e-12)
        tx, ty, tz = tx / tl, ty / tl, tz / tl
    dl = depth
    if not ridge:
        tmax = 4.0 * depth + 4.0 * hh
        s_ = 0.5 * hh
        th = tmax
        while s_ < tmax:
            fv = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, qx - s_ * nx, qy - s_ * ny, qz - s_ * nz, vs, ps)
            if fv > 0.0:
                th = s_
                break
            s_ += max(abs(fv), 0.25 * hh)
        dl = min(depth, 0.5 * th)
    return _ft_stamp(bp, qx, qy, qz, nx, ny, nz, tx, ty, tz, dl, x, y, z, f0)


@njit(cache=True, nogil=True)
def _ft_stamp(bp, qx, qy, qz, nx, ny, nz, tx, ty, tz, dl, x, y, z, f0):
    """fs-tools patch 4: the Pencil tip's field at a node and its blend (ftools.cuh ft_pencil's last part)"""
    r = bp[63]
    k = bp[52]
    md = int(bp[56])
    ridge = md >= 4
    tip = md & 3
    vx, vy, vz = ny * tz - nz * ty, nz * tx - nx * tz, nx * ty - ny * tx
    ex, ey, ez = x - qx, y - qy, z - qz
    a = ex * tx + ey * ty + ez * tz
    b = ex * vx + ey * vy + ez * vz
    w = ex * nx + ey * ny + ez * nz
    if ridge:
        w = -w
    if tip == 0:
        rho = math.sqrt(a * a + b * b)
        ww = w + dl
        L = math.sqrt(r * r + dl * dl)
        ca = dl / L
        sa = r / L
        T = math.sqrt(rho * rho + ww * ww) if ww * ca + rho * sa < 0.0 else rho * ca - ww * sa
    elif tip == 1:
        czz = r - dl
        T = math.sqrt(a * a + b * b + (w - czz) ** 2) - r
    else:
        qx_ = abs(b) - r
        qy_ = abs(a) - r
        qz_ = abs(w - (r - dl)) - r
        mx, my, mz = max(qx_, 0.0), max(qy_, 0.0), max(qz_, 0.0)
        T = math.sqrt(mx * mx + my * my + mz * mz) + min(max(qx_, max(qy_, qz_)), 0.0)
    if not ridge:
        c_ = -T
        if k > 0.0:
            hk = max(k - abs(f0 - c_), 0.0) / k
            return max(f0, c_) + hk * hk * k * 0.25
        return max(f0, c_)
    if k > 0.0:
        hk = max(k - abs(f0 - T), 0.0) / k
        return min(f0, T) - hk * hk * k * 0.25
    return min(f0, T)


@njit(cache=True, nogil=True)
def _ft_node(code, prm, oi_tab, of_tab, index_flat, delta, oi, rt, bp, x, y, z, f0, vs, ps):
    """fs-tools 11-14: one approved tool's rule at a node (toolset/ftools.cuh ft_rule, the same arithmetic; rt 0 Swell,
    1 Settle, 2 Flatten, 3 Pencil). Stop at contact is the card's (its contact rows); here the CPU field only."""
    if rt == 3:
        return _ft_pencil(code, prm, oi_tab, of_tab, index_flat, delta, oi, bp, x, y, z, f0, vs, ps)
    r = bp[63] if bp[63] > 0.0 else bp[7]
    hard = bp[51]
    sm = bp[52]
    fall = int(bp[53])
    fam = int(bp[54])
    cx, cy, cz = bp[1], bp[2], bp[3]
    nx, ny, nz = bp[4], bp[5], bp[6]
    if rt <= 1:
        av = bp[55]
        if int(bp[56]) == 0:                     # follow: the weight at the node's anchor (its point on the surface)
            hh = bp[62]
            gx = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x + hh, y, z, vs, ps)
                  - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x - hh, y, z, vs, ps))
            gy = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y + hh, z, vs, ps)
                  - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y - hh, z, vs, ps))
            gz = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z + hh, vs, ps)
                  - eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z - hh, vs, ps))
            gl = math.sqrt(gx * gx + gy * gy + gz * gz)
            qx, qy, qz = x, y, z
            if gl >= 1e-12:
                sg = f0 / gl
                qx, qy, qz = x - sg * gx, y - sg * gy, z - sg * gz
            dq = math.sqrt((qx - cx) ** 2 + (qy - cy) ** 2 + (qz - cz) ** 2)
            w = _ft_feather(bp, _ft_w(dq, r, hard, sm, fall, fam), qx, qy, qz, False, nx, ny, nz, r, hard, sm, fall, fam)
            if w <= 0.0:
                return f0
            return f0 - av * w
        ex, ey, ez = x - cx, y - cy, z - cz             # flat: a held flow along the press facing
        t = ex * nx + ey * ny + ez * nz
        lx, ly, lz = ex - t * nx, ey - t * ny, ez - t * nz
        w = _ft_feather(bp, _ft_w(math.sqrt(lx * lx + ly * ly + lz * lz), r, hard, sm, fall, fam), x, y, z, True,
                        nx, ny, nz, r, hard, sm, fall, fam)
        if w <= 0.0:
            return f0
        sv = av * w
        return eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x - sv * nx, y - sv * ny, z - sv * nz, vs, ps)
    ex, ey, ez = x - bp[59], y - bp[60], z - bp[61]     # flatten: onto the plane through the point under the cursor
    Pd = ex * nx + ey * ny + ez * nz
    lx, ly, lz = ex - Pd * nx, ey - Pd * ny, ez - Pd * nz
    w = _ft_feather(bp, _ft_w(math.sqrt(lx * lx + ly * ly + lz * lz), r, hard, sm, fall, fam), x, y, z, True,
                    nx, ny, nz, r, hard, sm, fall, fam)
    if w <= 0.0:
        return f0
    return f0 + (Pd - f0) * (1.0 - math.exp(-bp[55] * w))


@njit(parallel=True, cache=True, nogil=True)
def brush_kernel(code, prm, oi_tab, of_tab, index_flat, delta, mask, oi, i0, j0, k0, h, bp, lut, out_f, out_m,
                 blo0, blo1, blo2, bmask, steps, mgrid):
    ni, nj, nk = out_f.shape
    typ = int(bp[0])
    cx, cy, cz = bp[1], bp[2], bp[3]
    nx, ny, nz = bp[4], bp[5], bp[6]
    r = bp[7]
    s = bp[8]
    vx, vy, vz = bp[9], bp[10], bp[11]
    dl = bp[12]
    A = bp[13]
    use_mask = bp[14] > 0.5
    tmode = int(bp[15]) if bp.shape[0] > 15 else 0
    minv = of_tab[oi, P.OF_MINV] > 0.5          # hand2f: Mask > Invert (the stored mask reads as 1 - m)
    use_bm = bmask.shape[0] > 1 or bmask.shape[1] > 1 or bmask.shape[2] > 1   # fs-brush 11-12: the band's bricks
    skipb = bp[NBP_SKIP] if bp.shape[0] > NBP_SKIP else 0.0
    sideb = bp[NBP_SIDE] if bp.shape[0] > NBP_SIDE else 0.0
    for a in prange(ni):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        for b in range(nj):
            for c in range(nk):
                out_f[a, b, c] = 0.0
                out_m[a, b, c] = 0.0
                x = (i0 + a) * h
                y = (j0 + b) * h
                z = (k0 + c) * h
                rx = x - cx
                ry = y - cy
                rz = z - cz
                d = math.sqrt(rx * rx + ry * ry + rz * rz)
                if d >= r:
                    continue
                if use_bm and bmask[((i0 + a) >> 3) - blo0, ((j0 + b) >> 3) - blo1, ((k0 + c) >> 3) - blo2] == 0:
                    continue                     # fs-brush 11-12: a brick beyond the band (not allocated): untouched
                if typ >= FT_T0 and typ <= FT_T0 + 3:
                    # fs-tools 11-14: the approved brush tools, the rules of toolset/ftools.cuh on the CPU field (the card's
                    # twin; their own profile, not the lut; the sheet / skip / mask as the other tools)
                    mf = 0.0
                    if use_mask:
                        mf = grid_sample(index_flat, mask, oi_tab, oi, h, x, y, z)
                        if minv:
                            mf = 1.0 - min(max(mf, 0.0), 1.0)
                    f0 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps)
                    if skipb > 0.0 and abs(f0) > skipb:
                        if sideb == 0.0 or (sideb > 0.0 and f0 < 0.0) or (sideb < 0.0 and f0 > 0.0):
                            continue
                    f1 = _ft_node(code, prm, oi_tab, of_tab, index_flat, delta, oi, typ - FT_T0, bp, x, y, z, f0, vs, ps)
                    out_f[a, b, c] = (f1 - f0) * (1.0 - mf)
                    continue
                w = _lut(lut, d / r)
                if w <= 0.0:
                    continue
                if bp.shape[0] > NBP_MIRP + 3 and bp[NBP_MIRP + 3] > 0.5:     # fs-brush 11-11: mirror feather
                    ex_ = x - bp[NBP_MIRP]
                    ey_ = y - bp[NBP_MIRP + 1]
                    ez_ = z - bp[NBP_MIRP + 2]
                    d2_ = math.sqrt(ex_ * ex_ + ey_ * ey_ + ez_ * ez_)
                    if d2_ < r:
                        w2_ = _lut(lut, d2_ / r)
                        if w2_ > 0.0:
                            # the pair sums to the contraharmonic mean T = (w^2 + w2^2) / (w + w2): smooth (max() has a
                            # kink where the weights cross, on the midline itself, and the pair's move creased there:
                            # cpu_seam.py), T = w where one acts alone and where the two coincide (no doubling)
                            w = w * (w * w + w2_ * w2_) / ((w + w2_) * (w + w2_))
                if tmode != 0:                   # brush targeting (11-08 night)
                    f0g = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps)
                    w *= target_gate(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, f0g,
                                     cx, cy, cz, nx, ny, nz, bp, vs, ps)
                    if w <= 0.0:
                        continue
                m = 0.0
                if use_mask or typ == 10:
                    m = grid_sample(index_flat, mask, oi_tab, oi, h, x, y, z)
                if typ == 10:
                    if minv:                     # painting the shown (inverted) mask lowers the stored one
                        nm = min(max(m - s * w, 0.0), 1.0)
                    else:
                        nm = min(max(m + s * w, 0.0), 1.0)
                    out_m[a, b, c] = nm - m
                    continue
                if minv and use_mask:
                    m = 1.0 - min(max(m, 0.0), 1.0)
                f0 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps)
                if skipb > 0.0 and abs(f0) > skipb:
                    # fs-brush 11-12: as the card (sculpt.cu bp[28]): beyond the move + band; one-sided (bp[NBP_SIDE]):
                    # only the side the surface moves away from (1: inside, f0 < 0; -1: outside, f0 > 0)
                    if sideb == 0.0 or (sideb > 0.0 and f0 < 0.0) or (sideb < 0.0 and f0 > 0.0):
                        continue
                dn = rx * nx + ry * ny + rz * nz
                if (typ == 1 or typ == 5) and bp.shape[0] > NBP_PLANE + 6 and bp[NBP_PLANE + 6] > 0.5:
                    # fs-brush 11-11: Clay / Flatten on the area plane (area_plane_cpu; gpu/sculpt.cu area_plane)
                    dn = ((x - bp[NBP_PLANE]) * bp[NBP_PLANE + 3] + (y - bp[NBP_PLANE + 1]) * bp[NBP_PLANE + 4]
                          + (z - bp[NBP_PLANE + 2]) * bp[NBP_PLANE + 5])
                if typ == 0:
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                    x - nx * A * w, y - ny * A * w, z - nz * A * w, vs, ps)
                elif typ == 1:
                    if A >= 0.0:
                        f1 = f0 + w * (min(f0, dn - A) - f0)
                        if bp.shape[0] > NBP_LIM and bp[NBP_LIM] > 0.5:
                            f1 = max(f1, f0 - A)           # 11-10: a hollow under the plane fills over passes
                    else:                                  # fs-core 11-14: Ctrl's dig, the mirror rule (gpu/sculpt.cu)
                        f1 = f0 + w * (max(f0, dn - A) - f0)
                        if bp.shape[0] > NBP_LIM and bp[NBP_LIM] > 0.5:
                            f1 = min(f1, f0 - A)
                elif typ == 2:
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                    x + nx * A * w, y + ny * A * w, z + nz * A * w, vs, ps)
                elif typ == 3 or typ == 11:      # 11: swell to contact = inflate, clamped by the contacts
                    f1 = f0 - A * w
                elif typ == 4 and bp.shape[0] > NBP_QSM + 4 and bp[NBP_QSM] > 0.5:
                    f1 = f0                      # fs-brush 11-11: Smooth at the brush's scale (_quad_smooth)
                    if abs(f0) < bp[NBP_QSM + 1] * h:
                        aq, okq = _quad_smooth(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, f0, h, r, bp, vs, ps)
                        if not okq or (bp.shape[0] > NBP_QOFF and 0.5 < bp[NBP_QOFF] < 1.5):
                            aq = 0.0
                        cm = bp[NBP_CELL] if bp.shape[0] > NBP_CELL else 0.0
                        if cm < 0.5:
                            e2 = 2.0 * h            # + the lattice's own scale (as gpu/sculpt.cu)
                            lap = (eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x + e2, y, z, vs, ps)
                                   + eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x - e2, y, z, vs, ps)
                                   + eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y + e2, z, vs, ps)
                                   + eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y - e2, z, vs, ps)
                                   + eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z + e2, vs, ps)
                                   + eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z - e2, vs, ps)) / 6.0 - f0
                        elif cm < 1.5:                 # binomial 3^3 at h (NBP_CELL)
                            accb = 0.0
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    for dk in range(-1, 2):
                                        wb = (2 - abs(di)) * (2 - abs(dj)) * (2 - abs(dk))
                                        accb += wb * eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                                                x + di * h, y + dj * h, z + dk * h, vs, ps)
                            lap = accb / 64.0 - f0
                        else:
                            lap = 0.0
                        capq = bp[NBP_QSM + 4] * h
                        f1 = f0 + min(max(s * w * (aq + lap), -capq), capq)
                elif typ == 4:
                    if bp.shape[0] > NBP_SHEET and bp[NBP_SHEET] > 0.5:
                        av = _sheet_avg(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, f0, dl, h, vs, ps)
                        f1 = f0 if math.isnan(av) else f0 + s * w * (av - f0)
                    else:
                        acc = 0.0
                        acc += eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x + dl, y, z, vs, ps)
                        acc += eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x - dl, y, z, vs, ps)
                        acc += eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y + dl, z, vs, ps)
                        acc += eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y - dl, z, vs, ps)
                        acc += eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z + dl, vs, ps)
                        acc += eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z - dl, vs, ps)
                        f1 = f0 + s * w * (acc / 6.0 - f0)
                elif typ == 5:
                    f1 = f0 + s * w * (dn - f0)
                    if bp.shape[0] > NBP_LIM and bp[NBP_LIM] > 0.5:
                        f1 = min(max(f1, f0 - A), f0 + A)  # 11-10: at most the dab's amount per dab
                elif typ == 6 or typ == 7:
                    tx = rx - nx * dn
                    ty = ry - ny * dn
                    tz = rz - nz * dn
                    v2 = bp.shape[0] > NBP_STR and bp[NBP_STR] > 0.5
                    kk = s * w * (1.0 if (v2 or typ == 7) else 0.5)
                    if v2:                       # fs-brush 11-11: toward the stroke LINE (Blender 2.91+), not its point
                        dx_, dy_, dz_ = vx - nx * (vx * nx + vy * ny + vz * nz), vy - ny * (vx * nx + vy * ny + vz * nz), vz - nz * (vx * nx + vy * ny + vz * nz)
                        dl_ = math.sqrt(dx_ * dx_ + dy_ * dy_ + dz_ * dz_)
                        if dl_ > 1e-9:
                            dx_, dy_, dz_ = dx_ / dl_, dy_ / dl_, dz_ / dl_
                            ta_ = tx * dx_ + ty * dy_ + tz * dz_
                            tx, ty, tz = tx - dx_ * ta_, ty - dy_ * ta_, tz - dz_ * ta_
                    elif typ == 7 and bp.shape[0] > NBP_PINCH and bp[NBP_PINCH] > 0.0:   # fs-brush 11-11: bounded
                        kk = min(max(kk, -bp[NBP_PINCH]), bp[NBP_PINCH])
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                    x + tx * kk, y + ty * kk, z + tz * kk, vs, ps)
                    if typ == 6:
                        f1 += A * w * w * w
                elif typ == 12:                  # settle: computed outside the kernel (collide.settle_dab)
                    continue
                elif typ == 8:
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                    x - vx * w, y - vy * w, z - vz * w, vs, ps)
                elif typ == 16:                  # fs-brush 11-12: Snake Hook's step as a flow (snake_back)
                    xb, yb, zb = snake_back(x, y, z, bp, lut)
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, xb, yb, zb, vs, ps)
                elif typ == 17:                  # fs-brush 11-12: Snake Hook's whole path composed (snake_path_back)
                    xb, yb, zb = snake_path_back(x, y, z, steps, lut)
                    if SNAKE_SKIP_UNTOUCHED and xb == x and yb == y and zb == z:
                        continue                 # no step of the path reaches this node: untouched
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, xb, yb, zb, vs, ps)
                elif typ == 14:                  # fs-brush 11-12: the anchored flow Grab (_flow_back)
                    xb, yb, zb = _flow_back(x, y, z, bp, lut, mgrid)
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, xb, yb, zb, vs, ps)
                elif typ == 15:                  # fs-brush 11-12: Stretch as an anchored elongation (_stretch_back)
                    xb, yb, zb = _stretch_back(x, y, z, bp, lut)
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, xb, yb, zb, vs, ps)
                else:
                    vl = math.sqrt(vx * vx + vy * vy + vz * vz)
                    if vl < 1e-12:
                        continue
                    ax = vx / vl
                    ay = vy / vl
                    az = vz / vl
                    kw = (vl / r) * w
                    pr = (rx * ax + ry * ay + rz * az) * (kw / (1.0 + kw))
                    f1 = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                    x - ax * pr, y - ay * pr, z - az * pr, vs, ps)
                out_f[a, b, c] = (f1 - f0) * (1.0 - m)


@njit(cache=True)
def _topo_counts(inn):
    """inn (27,) bool, centre 13 ignored -> (t26, t6), each capped at 3: gpu/sculpt.cu topo_counts"""
    lab = np.zeros(27, np.int64)
    st = np.zeros(27, np.int64)
    ncomp = 0
    for s0 in range(27):
        if ncomp >= 3:
            break
        if s0 == 13 or not inn[s0] or lab[s0] != 0:
            continue
        ncomp += 1
        sp = 0
        st[sp] = s0
        sp += 1
        lab[s0] = ncomp
        while sp > 0:
            sp -= 1
            c = st[sp]
            cx, cy, cz = c // 9, (c // 3) % 3, c % 3
            for d in range(27):
                if d == 13 or not inn[d] or lab[d] != 0:
                    continue
                if abs(d // 9 - cx) <= 1 and abs((d // 3) % 3 - cy) <= 1 and abs(d % 3 - cz) <= 1:
                    lab[d] = ncomp
                    st[sp] = d
                    sp += 1
    lab[:] = 0
    nb6 = 0
    for s0 in range(27):
        if nb6 >= 3:
            break
        sx, sy, sz = s0 // 9 - 1, (s0 // 3) % 3 - 1, s0 % 3 - 1
        if s0 == 13 or inn[s0] or lab[s0] != 0 or abs(sx) + abs(sy) + abs(sz) != 1:
            continue
        nb6 += 1
        sp = 0
        st[sp] = s0
        sp += 1
        lab[s0] = 1
        while sp > 0:
            sp -= 1
            c = st[sp]
            cx, cy, cz = c // 9, (c // 3) % 3, c % 3
            for q in range(6):
                nx, ny, nz = cx, cy, cz
                if q == 0:
                    nx += 1
                elif q == 1:
                    nx -= 1
                elif q == 2:
                    ny += 1
                elif q == 3:
                    ny -= 1
                elif q == 4:
                    nz += 1
                else:
                    nz -= 1
                if nx < 0 or ny < 0 or nz < 0 or nx > 2 or ny > 2 or nz > 2:
                    continue
                if abs(nx - 1) + abs(ny - 1) + abs(nz - 1) > 2:
                    continue
                d = nx * 9 + ny * 3 + nz
                if d == 13 or inn[d] or lab[d] != 0:
                    continue
                lab[d] = 1
                st[sp] = d
                sp += 1
    return ncomp, nb6


@njit(cache=True)
def _flip_ok(inn, to_inside):
    """11-10: gain material only where the inside around is one piece (a pinhole, channel or cavity may close, two
    pieces never join); lose it only where the outside around is one piece and the inside at most one (no tear, no
    hole punched through): gpu/sculpt.cu flip_ok"""
    t26, t6 = _topo_counts(inn)
    if to_inside:
        return t26 == 1
    return t6 == 1 and t26 <= 1


@njit(cache=True)
def _simple_point(inn):
    t26, t6 = _topo_counts(inn)
    return t26 == 1 and t6 == 1


@njit(parallel=True, cache=True, nogil=True)
def _field_box(code, prm, oi_tab, of_tab, index_flat, delta, oi, i0, j0, k0, ni, nj, nk, h, out):
    for a in prange(ni):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        for b in range(nj):
            for c in range(nk):
                out[a, b, c] = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                          (i0 + a) * h, (j0 + b) * h, (k0 + c) * h, vs, ps)


@njit(parallel=True, cache=True, nogil=True)
def _field_box_band(code, prm, oi_tab, of_tab, index_flat, delta, oi, i0, j0, k0, ni, nj, nk, h, out, blo0, blo1, blo2,
                    bmask, bval):
    """fs-brush 11-12 (F3): _field_box, but a node of a brick the band left out (bmask 0) takes its brick centre's value
    (bval: the right sign, farther than the band) instead of an evaluation: the keep-apart sweeps read only the signs of
    nodes next to a node changing sign, and those all lie inside the band's bricks"""
    nbx, nby, nbz = bmask.shape
    for a in prange(ni):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        for b in range(nj):
            for c in range(nk):
                bx = ((i0 + a) >> 3) - blo0
                by = ((j0 + b) >> 3) - blo1
                bz = ((k0 + c) >> 3) - blo2
                if 0 <= bx < nbx and 0 <= by < nby and 0 <= bz < nbz and bmask[bx, by, bz] == 0:
                    out[a, b, c] = bval[bx, by, bz]
                else:
                    out[a, b, c] = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                              (i0 + a) * h, (j0 + b) * h, (k0 + c) * h, vs, ps)


@njit(cache=True)
def _topo_sweeps(F, T, i0, j0, k0, eps, sweeps):
    """F (padded by 1, current), T (inner target = F + df): the nodes are moved to T in 8 parity colours, a sign change
    only where it keeps the topology, sweeps repeated until nothing more is accepted; refused nodes keep their sign
    (at most eps from zero, never deeper). F is updated in place."""
    ni, nj, nk = T.shape
    inn = np.zeros(27, np.bool_)
    pend = np.zeros((ni, nj, nk), np.bool_)
    for a in range(ni):
        for b in range(nj):
            for c in range(nk):
                f = F[a + 1, b + 1, c + 1]
                t = T[a, b, c]
                if (f < 0.0) == (t < 0.0):
                    F[a + 1, b + 1, c + 1] = t
                else:
                    pend[a, b, c] = True
    for sw in range(sweeps):
        acc = 0
        for col in range(8):
            for a in range(ni):
                if ((i0 + a) & 1) != (col & 1):
                    continue
                for b in range(nj):
                    if (((j0 + b) & 1) << 1) != (col & 2):
                        continue
                    for c in range(nk):
                        if (((k0 + c) & 1) << 2) != (col & 4) or not pend[a, b, c]:
                            continue
                        for q in range(27):
                            inn[q] = F[a + q // 9, b + (q // 3) % 3, c + q % 3] < 0.0
                        inn[13] = T[a, b, c] < 0.0
                        if _flip_ok(inn, T[a, b, c] < 0.0):
                            F[a + 1, b + 1, c + 1] = T[a, b, c]
                            pend[a, b, c] = False
                            acc += 1
        if acc == 0:
            break
    # 11-10: a node still refused keeps its value (an eps clamp left sub-cell membranes: flakes on her)


def keep_apart_cpu(pk, oi, lo, h, df, band=None, sweeps=16):
    """the card's dab_topo_commit on the CPU path (11-09 night): df (the dab's change over the box from lo) is cut back
    where it would join, film over or tear surfaces. band (fs-brush 11-12, F3): (blo, bmask, bval) of a band-limited
    dab: the field evaluated in the band's bricks only (_field_box_band)"""
    ni, nj, nk = df.shape
    F = np.empty((ni + 2, nj + 2, nk + 2))
    if band is not None:
        blo_, bm_, bv_ = band
        _field_box_band(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, int(lo[0]) - 1,
                        int(lo[1]) - 1, int(lo[2]) - 1, ni + 2, nj + 2, nk + 2, h, F, int(blo_[0]), int(blo_[1]),
                        int(blo_[2]), bm_, bv_)
    else:
        _field_box(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, int(lo[0]) - 1, int(lo[1]) - 1,
                   int(lo[2]) - 1, ni + 2, nj + 2, nk + 2, h, F)
    F0 = F[1:-1, 1:-1, 1:-1].copy()
    T = F0 + df
    if not np.any((F0 < 0) != (T < 0)):
        return
    _topo_sweeps(F, T, int(lo[0]), int(lo[1]), int(lo[2]), TOPO_EPS * h, int(sweeps))
    df[...] = F[1:-1, 1:-1, 1:-1] - F0


def keep_apart_sweeps(tool, bp, settings, h):
    """fs-brush-4 11-13 (the composed Snake's lost tip, fs-brush/snake4/tip_dbg2.py: the kernel moved the arm's tip nodes
    29-30 mm, but keep_apart_cpu's FIXED 16 sweeps accept a front only so far from the old surface, so the farthest flips
    were refused and those nodes kept their stroke-start values): the sweeps a dab needs grow with how far it can move a
    surface, as the card's _run (n_sw = max(TOPO_SWEEPS, ceil(move / h) + 4)); for the composed Snake the move is the
    whole path's length. The loop stops at the first sweep that accepts nothing, so a small move costs no more"""
    if tool == 'snakepath':
        st_ = settings.get('snake_steps_local')
        mv = float(np.linalg.norm(np.asarray(st_, float)[:, 3:6], axis=1).sum()) if st_ is not None and len(st_) else 0.0
    elif tool in ('grab', 'stretch', 'grabflow', 'stretchflow', 'snakeflow'):
        mv = float(np.linalg.norm(bp[9:12]))
    else:
        mv = abs(float(bp[13]))
    return int(min(KEEP_APART_MAX_SWEEPS, max(16, math.ceil(mv / max(float(h), 1e-12)) + 4)))


KEEP_APART_MAX_SWEEPS = int(os.environ.get('FSB_KA_MAX_SWEEPS', '512'))


@njit(parallel=True, cache=True, nogil=True)
def write_kernel(index_flat, delta, mask, oi_tab, oi, i0, j0, k0, df, dm):
    ni, nj, nk = df.shape
    off = oi_tab[oi, P.OI_IDX]
    nx = oi_tab[oi, P.OI_NX]
    ny = oi_tab[oi, P.OI_NY]
    nz = oi_tab[oi, P.OI_NZ]
    ox = oi_tab[oi, P.OI_OX]
    oy = oi_tab[oi, P.OI_OY]
    oz = oi_tab[oi, P.OI_OZ]
    for a in prange(ni):
        i = i0 + a
        for b in range(nj):
            j = j0 + b
            for c in range(nk):
                if df[a, b, c] == 0.0 and dm[a, b, c] == 0.0:
                    continue
                k = k0 + c
                slot = index_flat[off + (((i >> 3) - ox) * ny + ((j >> 3) - oy)) * nz + ((k >> 3) - oz)]
                if df[a, b, c] != 0.0:
                    delta[slot, i & 7, j & 7, k & 7] = np.float32(delta[slot, i & 7, j & 7, k & 7] + df[a, b, c])
                if dm[a, b, c] != 0.0:
                    mask[slot, i & 7, j & 7, k & 7] = np.float32(mask[slot, i & 7, j & 7, k & 7] + dm[a, b, c])


@njit(parallel=True, cache=True, nogil=True)
def brick_lipschitz(index_flat, delta, oi_tab, oi, h, bricks, out):
    """bricks (n,3) global brick coords -> out (n,): max over cells with lower corner in [8b-1, 8b+7] of
    sqrt(gx^2 + gy^2 + gz^2), g_axis = max |edge difference| / h over the cell's 4 parallel edges."""
    off = oi_tab[oi, P.OI_IDX]
    nx = oi_tab[oi, P.OI_NX]
    ny = oi_tab[oi, P.OI_NY]
    nz = oi_tab[oi, P.OI_NZ]
    ox = oi_tab[oi, P.OI_OX]
    oy = oi_tab[oi, P.OI_OY]
    oz = oi_tab[oi, P.OI_OZ]
    for q in prange(bricks.shape[0]):
        best = 0.0
        bi = bricks[q, 0] * 8
        bj = bricks[q, 1] * 8
        bk = bricks[q, 2] * 8
        for i in range(bi - 1, bi + 8):
            for j in range(bj - 1, bj + 8):
                for k in range(bk - 1, bk + 8):
                    v = np.empty(8)
                    for t in range(8):
                        v[t] = _node(index_flat, delta, off, nx, ny, nz, ox, oy, oz,
                                     i + (t & 1), j + ((t >> 1) & 1), k + ((t >> 2) & 1))
                    gx = max(max(abs(v[1] - v[0]), abs(v[3] - v[2])), max(abs(v[5] - v[4]), abs(v[7] - v[6])))
                    gy = max(max(abs(v[2] - v[0]), abs(v[3] - v[1])), max(abs(v[6] - v[4]), abs(v[7] - v[5])))
                    gz = max(max(abs(v[4] - v[0]), abs(v[5] - v[1])), max(abs(v[6] - v[2]), abs(v[7] - v[3])))
                    g = math.sqrt(gx * gx + gy * gy + gz * gz) / h
                    if g > best:
                        best = g
        out[q] = best


@njit(parallel=True, cache=True, nogil=True)
def write_dir(index_flat, mask, mdir, oi_tab, oi, i0, j0, k0, dm, nx, ny, nz):
    """after write_kernel: each node the Mask paint raised takes the direction blend (old direction x old paint +
    this dab's sheet direction x its added paint), stored x 127"""
    ni, nj, nk = dm.shape
    off = oi_tab[oi, P.OI_IDX]
    nxo = oi_tab[oi, P.OI_NX]
    ny_ = oi_tab[oi, P.OI_NY]
    nz_ = oi_tab[oi, P.OI_NZ]
    ox = oi_tab[oi, P.OI_OX]
    oy = oi_tab[oi, P.OI_OY]
    oz = oi_tab[oi, P.OI_OZ]
    for a in prange(ni):
        i = i0 + a
        for b in range(nj):
            j = j0 + b
            for c in range(nk):
                if dm[a, b, c] <= 0.0:
                    continue
                k = k0 + c
                slot = index_flat[off + (((i >> 3) - ox) * ny_ + ((j >> 3) - oy)) * nz_ + ((k >> 3) - oz)]
                m1 = mask[slot, i & 7, j & 7, k & 7]
                m0 = max(m1 - dm[a, b, c], 0.0)
                dx = mdir[slot, i & 7, j & 7, k & 7, 0] / 127.0 * m0 + nx * dm[a, b, c]
                dy = mdir[slot, i & 7, j & 7, k & 7, 1] / 127.0 * m0 + ny * dm[a, b, c]
                dz = mdir[slot, i & 7, j & 7, k & 7, 2] / 127.0 * m0 + nz * dm[a, b, c]
                L = math.sqrt(dx * dx + dy * dy + dz * dz)
                if L < 1e-9:
                    continue
                mdir[slot, i & 7, j & 7, k & 7, 0] = np.int8(round(dx / L * 127.0))
                mdir[slot, i & 7, j & 7, k & 7, 1] = np.int8(round(dy / L * 127.0))
                mdir[slot, i & 7, j & 7, k & 7, 2] = np.int8(round(dz / L * 127.0))


def sheet_normal(pk, oi, c, n, r, h, nmax=1500):
    """the painted sheet's direction at SHEET scale: the mean field gradient over the dab's near-surface nodes
    (|F| <= 1.5 h, within r) on the brush's own side (grad . n > 0), so the facets of a rough or jagged sheet
    inherit their sheet's orientation and the facing sheet across a gap (grad . n < 0) is left out of the mean"""
    lo = np.floor((c - r) / h).astype(np.int64)
    hi = np.ceil((c + r) / h).astype(np.int64)
    g = np.stack(np.meshgrid(*[np.arange(lo[a], hi[a] + 1) for a in range(3)], indexing='ij'), -1).reshape(-1, 3) * h
    g = g[np.linalg.norm(g - c, axis=1) <= r]
    if len(g) > 8 * nmax:
        g = g[np.linspace(0, len(g) - 1, 8 * nmax).astype(np.int64)]
    X = np.ascontiguousarray(g, np.float64)
    f = np.empty(len(X))
    P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, X, f)
    X = X[np.abs(f) <= 1.5 * h]
    if len(X) > nmax:
        X = X[np.linspace(0, len(X) - 1, nmax).astype(np.int64)]
    if not len(X):
        return n
    e = 0.5 * h
    G = np.zeros((len(X), 3))
    for ax in range(3):
        a = np.empty(len(X)); b = np.empty(len(X))
        d = np.zeros(3); d[ax] = e
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X + d), a)
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X - d), b)
        G[:, ax] = a - b
    G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    G_all = G
    # r (her 0.5 mm gap site: the plain mean over every node with grad . n > 0 came out 43 deg off the brush, pulled by
    # side-facing walls 2-3 mm away): only surfaces within ~73 deg of the brush (a +-60 deg facet still counts), weighted
    # toward the dab centre by the falloff
    keep = G @ n > 0.3
    if not keep.any():
        return n
    w = 1.0 - np.minimum(np.linalg.norm(X[keep] - c, axis=1) / r, 1.0)
    s = (G[keep] * w[:, None]).sum(0)
    L = np.linalg.norm(s)
    ns = s / L if L > 1e-9 else n
    # 11-09: the closest OPPOSITE-facing surface's angle to the paint direction, per dab (the 110 deg
    # cutoff had a 116-118 deg margin at one of her sites): logged so a crossing would show
    cs_ = G_all @ ns
    opp = cs_ < 0.0
    MASK_ANGLE_LOG.append(float(np.degrees(np.arccos(np.clip(cs_[opp].max(), -1, 1)))) if opp.any() else None)
    del MASK_ANGLE_LOG[:-5000]
    return ns


class StrokeRecord:
    """Undo data for one stroke: bricks it created, and the prior contents of bricks it changed."""

    def __init__(self):
        self.new = []        # (oid, brick)
        self.new_set = set()
        self.snap = {}       # (oid, brick) -> (delta, mask, L)
        self.dsnap = {}      # (oid, brick) -> mask paint directions before (tagged Mask strokes)
        self.rsnap = {}      # fs-brush 11-11 (design 4.8): (oid, brick) -> region ids before (Region strokes)
        self.Lsnap = {}      # (oid, brick) -> L  (neighbours whose L was re-measured)

    def nbytes(self):
        return sum(v[0].nbytes + v[1].nbytes for v in self.snap.values()) + 64 * (len(self.new) + len(self.Lsnap))


def _symmetry_signs(sym):
    signs = [np.ones(3)]
    for ax in range(3):
        if sym[ax]:
            more = []
            for s in signs:
                t = s.copy()
                t[ax] = -1.0
                more.append(t)
            signs += more
    return signs


# fs-brush 11-11 (design 4.7: Clay and Grab 0.1). M (fs-bench fsbrush_1): on Grab it cut the bust's underside pull to
# 0.39 of the drag, because a Smooth at the Grab's own scale removes the very form being pulled; Grab 0 (as Blender).
# Autosmooth runs at AUTOSMOOTH_R of the brush's radius, so it takes the dabs' ripples and leaves the brush's form.
# Clay 0 by default too (M, bench_perf p18: Clay r 10 fell 40 -> 12 fps with its autosmooth dabs, each with its own guard
# sweeps); 'autosmooth' in a stroke's settings turns it on (0-1)
AUTOSMOOTH = {}
AUTOSMOOTH_R = 0.5
MOVE_TOOLS = ('grab', 'stretch', 'grabflow', 'stretchflow')   # fs-brush-4 11-13: the move tools, incl. the anchored flows
SNAP_LO = 0.8             # fs-brush 11-11 (design C4): midline snap: centre within SNAP_LO r of a mirror plane = one dab
# fs-brush 11-11 (fs-bench test1 t1_fsbrush_12: under Mirror X a 60 mm Grab pressed at the neck's
# rim 25 mm off the midline pulled 0.03 of the drag; the snap had merged it and its mirror into one centred dab whose
# sideways drag cancels): MIRROR FEATHER. The pair is kept, each dab carries its partner's centre, and where the two
# spheres overlap each node's weight is scaled by (w^2 + w2^2) / (w + w2)^2, so the pair sums to the contraharmonic mean
# T = (w^2 + w2^2) / (w + w2) (smooth; max() creased the midline): T = w alone and where the two coincide, so a
# midline stroke is not doubled, a sideways pull cancels only where the two sides meet, and away from the midline each
# dab acts in full. bp[41:44] the partner's centre (local), bp[44] > 0.5 on. One mirror axis ('mirror_feather').
NBP_MIRP = 41


def mirror_dabs(c, n, v, r, sym, snap=False):
    """the dabs one sample makes under symmetry (local frame): [(c, n, v, weight)]. snap False (older logs): the plain
    mirrored copies (_symmetry_signs). snap True (fs-brush 11-11, design C4: midline forms came out doubled, the nose a
    double ridge): on each mirrored axis, a centre within SNAP_LO r of the plane becomes ONE dab projected onto the
    plane (its normal and drag made symmetric); between SNAP_LO r and r the centred dab and the mirrored pair share the
    weight (t = (d - SNAP_LO r) / ((1 - SNAP_LO) r)), so nothing jumps at the threshold; beyond r, the pair."""
    c = np.asarray(c, float); n = np.asarray(n, float); v = np.asarray(v, float)
    if not snap:
        return [(c * sg, n * sg, v * sg, 1.0) for sg in _symmetry_signs(sym)]
    out = [(c, n, v, 1.0)]
    for ax in range(3):
        if not sym[ax]:
            continue
        nxt = []
        for c_, n_, v_, w_ in out:
            d = abs(float(c_[ax]))
            t = min(max((d - SNAP_LO * r) / max((1.0 - SNAP_LO) * r, 1e-12), 0.0), 1.0)
            sg = np.ones(3); sg[ax] = -1.0
            if t > 0.0:
                nxt.append((c_, n_, v_, w_ * t))
                nxt.append((c_ * sg, n_ * sg, v_ * sg, w_ * t))
            if t < 1.0:
                cc, nn, vv = c_.copy(), n_.copy(), v_.copy()
                cc[ax] = 0.0; nn[ax] = 0.0; vv[ax] = 0.0
                ln = float(np.linalg.norm(nn))
                nn = nn / ln if ln > 1e-9 else n_
                nxt.append((cc, nn, vv, w_ * (1.0 - t)))
        out = nxt
    return out


def invert_dab(tool, A, s):
    """fs-brush 11-11 (design 4.7, 'Ctrl inverts every brush'): (A, s) of the inverse brush. Inflate -> Deflate, Clay ->
    dig, Swell -> shrink, Crease cut -> raise a ridge (A); Flatten -> Contrast, Pinch -> Magnify, Smooth -> enhance (s).
    Draw / Carve swap tools and Mask erases (the viewport); Grab and Stretch have no inverse here."""
    if tool in ('inflate', 'clay', 'swell', 'crease'):
        return -A, s
    if tool in ('flatten', 'pinch', 'smooth'):
        return A, -s
    return A, s


def area_plane_cpu(pk, oi, c, n, r, h, lut):
    """fs-brush 11-11: gpu/sculpt.cu area_plane on the CPU field: the falloff-weighted mean position and normal of the
    surface under the brush (256 sunflower samples on a 0.8 r disc across n, three Newton steps each) -> (P, N)"""
    from .program import eval_points_local
    c = np.asarray(c, float); n = np.asarray(n, float) / max(np.linalg.norm(n), 1e-12)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
    k = np.arange(256)
    rho = 0.8 * r * np.sqrt((k + 0.5) / 256.0); th = k * 2.39996323
    P = c[None] + rho[:, None] * (np.cos(th)[:, None] * t1[None] + np.sin(th)[:, None] * t2[None])

    def ev(X):
        out = np.empty(len(X))
        eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(X), out)
        return out
    E = np.eye(3) * h
    for it in range(4):
        f = ev(P)
        G = np.stack([(ev(P + E[i]) - ev(P - E[i])) / (2 * h) for i in range(3)], 1)
        g2 = (G * G).sum(1)
        if it == 3:
            break
        P = P - (f / np.maximum(g2, 1e-12))[:, None] * G
    ok = (np.abs(f) < 0.25 * h) & (g2 > 1e-8) & ((G @ n) > 0)
    if not ok.any():
        return c, n
    u = np.linalg.norm(P - c[None], axis=1) / r
    w = np.interp(np.clip(u, 0, 1), np.linspace(0, 1, len(lut)), lut) * ok * (u < 1)
    if w.sum() <= 1e-9:
        return c, n
    Pm = (w[:, None] * P).sum(0) / w.sum()
    Nm = (w[:, None] * G / np.sqrt(np.maximum(g2, 1e-12))[:, None]).sum(0)
    ln = np.linalg.norm(Nm)
    return Pm, (Nm / ln if ln > 1e-9 else n)


def strength_v2_bp(bp, tool, settings, h):
    """fs-brush 11-11: bp[8] (and Smooth's cap) from strength_dab for the tools whose per-dab factor has a limit; the
    flag bp[NBP_STR] makes the kernels take it as given (and Pinch / Crease pull toward the stroke line, bp[9:12])"""
    sp = float(settings.get('spacing', 0.25))
    fr = frac_of(settings)
    st = float(settings['strength']) / fr          # a partial dab: the whole strength's response, times its share
    if settings.get('invert', False) and tool in ('flatten', 'pinch', 'smooth'):
        st = -st
    if tool in STRENGTH_RESP:
        bp[8] = strength_dab(tool, st, sp) * fr
        bp[NBP_STR] = 1.0
    if tool == 'smooth':
        bp[NBP_QSM + 4] = QSM_CAP_H * max(1.0, abs(st))
    if tool == 'crease':
        bp[NBP_STR] = 1.0


def _box_rid(st, g, lo, shape):
    """the region ids of the lattice nodes lo .. lo + shape - 1 (local node indices; 0 where no brick)"""
    out = np.zeros(shape, np.uint16)
    lo = np.asarray(lo, np.int64)
    hi = lo + np.asarray(shape, np.int64) - 1
    for bx in range(lo[0] >> 3, (hi[0] >> 3) + 1):
        for by in range(lo[1] >> 3, (hi[1] >> 3) + 1):
            for bz in range(lo[2] >> 3, (hi[2] >> 3) + 1):
                sl = g.get_slot((bx, by, bz))
                if sl < 0:
                    continue
                b0 = np.array([bx, by, bz], np.int64) * 8
                a0 = np.maximum(b0, lo)
                a1 = np.minimum(b0 + 7, hi)
                if np.any(a1 < a0):
                    continue
                out[a0[0] - lo[0]:a1[0] - lo[0] + 1, a0[1] - lo[1]:a1[1] - lo[1] + 1, a0[2] - lo[2]:a1[2] - lo[2] + 1] = \
                    st.rid[sl, a0[0] - b0[0]:a1[0] - b0[0] + 1, a0[1] - b0[1]:a1[1] - b0[1] + 1, a0[2] - b0[2]:a1[2] - b0[2] + 1]
    return out


# fs-brush 11-12 (design: a tool changes only the surface it works on and nothing behind it, unless a setting or
# toggle says otherwise; thinwall/tw1: one object, a 4 mm wall
# + a ball 2 mm behind it: today's Draw r 20 moved the wall's back 3.9 mm and the ball 3.5 mm; the 11-08 face / chord
# gates leaked above r 6: a same-facing third surface passes the facing test, and nodes on medial ridges / beyond 4 h
# were written ungated). THE SHEET UNDER THE BRUSH: per dab, the surface band region connected to the press point is
# grown over the lattice's nodes, never across a medial ridge (where the field's gradient collapses or turns back: the
# middle of a thin wall, of a gap, of a fold); only those nodes change. Deeper nodes keep their values (signs right) and
# the stroke's re-distance brings them back to distances. Grab-type moves are not gated (their reach is the form's own,
# Form-aware Grab's item).
SHEET_TOOLS = ('draw', 'carve', 'inflate', 'swell', 'clay', 'flatten', 'smooth', 'crease', 'pinch', 'layer',
               'ft_swell', 'ft_settle', 'ft_flatten',   # fs-core-2 11-14: the approved held tools (ee9e0f9b) too
               'ft_pencil')                             # fs-tools patch 4: Pencil is Near side always (design 3.3)
SHEET_DEFAULT = os.environ.get('FSB_SHEET', '1') != '0'   # 11-12: 'Protect surfaces behind' ON by default (the panel's default for a brush never set); FSB_SHEET=0 off
SHEET_GMIN = float(os.environ.get('FSB_SHEET_GMIN', '0.6'))     # |grad F| below this (central differences) = a ridge
SHEET_CONE = float(os.environ.get('FSB_SHEET_CONE', '100'))    # deg: the walk stops where the surface turns further than
                                                                # this from the brush (a lid's margin onto its back face)
SHEET_GEO = float(os.environ.get('FSB_SHEET_GEO', '2.0'))       # cell walk / straight distance allowed (an axis walk
                                                                # over a 45 deg surface is 1.41x the straight line)
# fs-core 11-14: 1.6 -> 2.0: a surface diagonal on all three axes takes sqrt 3 = 1.73x its straight distance in axis
# steps, so the walk stopped short of the ball's edge there and left part of the worked surface labelled 'another
# sheet' (fs-core dbg6: neighbours either side of the stroke-start surface, one written, one not: crease specks)


@njit(parallel=True, cache=True, nogil=True)
def _node_field(code, prm, oi_tab, of_tab, index_flat, delta, oi, i0, j0, k0, h, F):
    ni, nj, nk = F.shape
    for a in prange(ni):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        for b in range(nj):
            for c in range(nk):
                F[a, b, c] = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, (i0 + a) * h, (j0 + b) * h,
                                        (k0 + c) * h, vs, ps)


@njit(cache=True, inline='always')
def _ngrad(F, a, b, c, h):
    ni, nj, nk = F.shape
    a0 = max(a - 1, 0); a1 = min(a + 1, ni - 1)
    b0 = max(b - 1, 0); b1 = min(b + 1, nj - 1)
    c0 = max(c - 1, 0); c1 = min(c + 1, nk - 1)
    gx = (F[a1, b, c] - F[a0, b, c]) / (max(a1 - a0, 1) * h)
    gy = (F[a, b1, c] - F[a, b0, c]) / (max(b1 - b0, 1) * h)
    gz = (F[a, b, c1] - F[a, b, c0]) / (max(c1 - c0, 1) * h)
    return gx, gy, gz


@njit(cache=True, inline='always')
def _cell_mixed(F, a, b, c):
    neg = False
    pos = False
    for da in range(2):
        for db in range(2):
            for dc in range(2):
                v = F[a + da, b + db, c + dc]
                if math.isnan(v):
                    return False
                if v < 0.0:
                    neg = True
                else:
                    pos = True
    return neg and pos


@njit(cache=True, inline='always')
def _foot_label(F, h, a, b, c, gmin, st, cm, i0, j0, k0, seeds, inner2):
    """-> 0 / 1 / 2: the label of the surface cell nearest to node (a, b, c)'s foot (x - F grad F / |grad F|)"""
    ni, nj, nk = F.shape
    ci, cj, ck = ni - 1, nj - 1, nk - 1
    f = F[a, b, c]
    if math.isnan(f):
        return 0
    gx, gy, gz = _ngrad(F, a, b, c, h)
    gl = math.sqrt(gx * gx + gy * gy + gz * gz)
    if not (gl >= gmin):                         # a ridge, or a field the stroke distorted: no reliable foot
        return 0
    fx = a - f * gx / gl / h
    fy = b - f * gy / gl / h
    fz = c - f * gz / gl / h
    ba = int(math.floor(fx)); bb = int(math.floor(fy)); bc = int(math.floor(fz))
    lab = 0
    best = 1e30
    la = 0; lb = 0; lc = 0
    for da in range(-1, 2):
        for db in range(-1, 2):
            for dc in range(-1, 2):
                ea = ba + da; eb = bb + db; ec = bc + dc
                if ea < 0 or eb < 0 or ec < 0 or ea >= ci or eb >= cj or ec >= ck:
                    continue
                m = cm[ea, eb, ec]               # memo: -1 not yet known, 0 no surface, 1 a surface cell
                if m < 0:
                    m = 1 if _cell_mixed(F, ea, eb, ec) else 0
                    cm[ea, eb, ec] = m
                if m == 0:
                    continue
                dd = (ea + 0.5 - fx) ** 2 + (eb + 0.5 - fy) ** 2 + (ec + 0.5 - fz) ** 2
                if dd < best:
                    best = dd
                    lab = 1 if st[ea, eb, ec] > 0 else 2
                    la = ea; lb = eb; lc = ec
    if lab == 2:
        # 'another sheet' only for a foot cell well inside the reach (one near its edge may be this sheet for a later dab)
        x2 = (i0 + la + 0.5) * h; y2 = (j0 + lb + 0.5) * h; z2 = (k0 + lc + 0.5) * h
        for s in range(seeds.shape[0]):
            dx = x2 - seeds[s, 0]; dy = y2 - seeds[s, 1]; dz = z2 - seeds[s, 2]
            if dx * dx + dy * dy + dz * dz <= inner2:
                return 2
        return 3                                 # another sheet, near the reach's edge: undecided, never filled
    return lab


@njit(parallel=True, cache=True, nogil=True)
def _sheet_label(F, h, i0, j0, k0, seeds, r, gmin, st, cm, inner2, bandw, out, side=0):
    """every node in the reach: near the surface (|F| < bandw) the label of its own foot's cell; deeper nodes the label
    of their 8^3 block's centre (one projection a block: a deep node's foot is its block's, within the block's size)"""
    ni, nj, nk = F.shape
    ns = seeds.shape[0]
    reach2 = (r + 2.0 * h) * (r + 2.0 * h)
    nba = (ni + 7) // 8
    for ba_ in prange(nba):
        for bb_ in range((nj + 7) // 8):
            for bc_ in range((nk + 7) // 8):
                ca = min(ba_ * 8 + 4, ni - 1); cb = min(bb_ * 8 + 4, nj - 1); cc = min(bc_ * 8 + 4, nk - 1)
                blk = -1                         # the block's deep label, computed when first needed
                for a in range(ba_ * 8, min(ba_ * 8 + 8, ni)):
                    for b in range(bb_ * 8, min(bb_ * 8 + 8, nj)):
                        for c in range(bc_ * 8, min(bc_ * 8 + 8, nk)):
                            if (out[a, b, c] & 3) != 0:
                                continue
                            f = F[a, b, c]
                            if math.isnan(f):
                                continue
                            x = (i0 + a) * h; y = (j0 + b) * h; z = (k0 + c) * h
                            inb = False
                            for s in range(ns):
                                dx = x - seeds[s, 0]; dy = y - seeds[s, 1]; dz = z - seeds[s, 2]
                                if dx * dx + dy * dy + dz * dz <= reach2:
                                    inb = True
                                    break
                            if not inb:
                                continue
                            out[a, b, c] |= 4    # seen in a dab's reach (the stroke's re-distance keeps it unless on the sheet)
                            if side != 0 and abs(f) >= bandw and ((side > 0 and f > 0.0) or (side < 0 and f < 0.0)):
                                out[a, b, c] = 4 | 1    # deep on the side the dab moves into: the worked sheet's (gpu sh_label)
                                continue
                            if abs(f) < bandw:
                                lab = _foot_label(F, h, a, b, c, gmin, st, cm, i0, j0, k0, seeds, inner2)
                            else:
                                if blk < 0:
                                    blk = _foot_label(F, h, ca, cb, cc, gmin, st, cm, i0, j0, k0, seeds, inner2)
                                lab = blk
                            if lab == 3:
                                out[a, b, c] = 4 | 8      # bit 8: its foot is another sheet (beyond the inner reach)
                            elif lab != 0:
                                out[a, b, c] = 4 | lab


@njit(cache=True)
def _sheet_fill0(F, bandw, out):
    """gpu/sculpt.cu sh_fill0: an undecided node near the surface (seen in the reach, no label) joins the worked sheet
    unless one of its 26 neighbours belongs to another surface"""
    ni, nj, nk = F.shape
    upd = []
    for a in range(ni):
        for b in range(nj):
            for c in range(nk):
                o = out[a, b, c]
                if (o & 4) == 0 or (o & 3) != 0 or (o & 8) != 0 or math.isnan(F[a, b, c]) or abs(F[a, b, c]) >= bandw:
                    continue
                ok = True
                for da in range(-1, 2):
                    for db in range(-1, 2):
                        for dc in range(-1, 2):
                            a2 = a + da; b2 = b + db; c2 = c + dc
                            if a2 < 0 or b2 < 0 or c2 < 0 or a2 >= ni or b2 >= nj or c2 >= nk:
                                continue
                            if (out[a2, b2, c2] & 3) == 2 or (out[a2, b2, c2] & 8) != 0:
                                ok = False       # beside another sheet (labelled, or its foot there beyond the inner reach)
                if ok:
                    upd.append((a * nj + b) * nk + c)
    for idx in upd:
        out[idx // (nj * nk), (idx // nk) % nj, idx % nk] = 4 | 1


@njit(cache=True)
def _sheet_flood(F, h, i0, j0, k0, seeds, snorm, r, geo, gmin, cosc, bandw, out, side=0):
    """out (in / out) per node: 0 not seen yet this stroke, 1 its nearest surface is the sheet under the brush, 2 it is
    another surface's (or a ridge's). THE SHEET = the lattice cells the surface passes through (corners of both signs),
    flooded from the cell(s) under the press point across shared faces the surface crosses (a face with corners of both
    signs), no further than geo steps along the surface (a cell walk is never shorter than the surface: the geodesic
    reach, so a sheet joined only far round, a crevice's floor, a flap's rim, stays out). Each node then takes the label
    of the surface cell at its foot (x - F grad F / |grad F|). FIRST SIGHT: a labelled node keeps its label for the
    stroke (the stroke distorts the field it leaves behind; the labels were taken before)."""
    ni, nj, nk = F.shape
    ci, cj, ck = ni - 1, nj - 1, nk - 1
    st = np.zeros((ci, cj, ck), np.int16)       # 0 not a reached surface cell, > 0: steps + 1 from a seed
    sid = np.zeros((ci, cj, ck), np.int16)      # which seed's walk reached it (its own facing direction)
    cap = min(ci * cj * ck, 4 * (ci * cj + cj * ck + ci * ck) + 64)   # surface cells: an area, not a volume
    q = np.empty(cap, np.int64)
    qh = 0
    qt = 0
    ns = seeds.shape[0]
    reach2 = (r + 2.0 * h) * (r + 2.0 * h)
    inner = max(0.5 * r, r - 4.0 * h)
    inner2 = inner * inner
    for s in range(ns):
      for tstep in range(0, 2 * int(r / h) + 2):
        # the press point's cells first; when they hold no surface facing the brush (the stroke has moved the surface
        # away from where the sample was taken), the nearest surface along the brush's axis, alternately out and in
        tt = (tstep + 1) // 2 * 0.5 * h * (1.0 if tstep % 2 == 1 else -1.0)
        px_ = seeds[s, 0] + snorm[s, 0] * tt; py_ = seeds[s, 1] + snorm[s, 1] * tt; pz_ = seeds[s, 2] + snorm[s, 2] * tt
        sa = int(math.floor(px_ / h)) - i0
        sb = int(math.floor(py_ / h)) - j0
        sc = int(math.floor(pz_ / h)) - k0
        q0 = qt
        for da in range(-1, 2):
            for db in range(-1, 2):
                for dc in range(-1, 2):
                    a = sa + da; b = sb + db; c = sc + dc
                    if a < 0 or b < 0 or c < 0 or a >= ci or b >= cj or c >= ck or st[a, b, c] > 0:
                        continue
                    if not _cell_mixed(F, a, b, c):
                        continue
                    # the cell's own normal must face the brush (not the far face of a thin flap under the press)
                    gx = (F[a + 1, b, c] + F[a + 1, b + 1, c] + F[a + 1, b, c + 1] + F[a + 1, b + 1, c + 1]
                          - F[a, b, c] - F[a, b + 1, c] - F[a, b, c + 1] - F[a, b + 1, c + 1])
                    gy = (F[a, b + 1, c] + F[a + 1, b + 1, c] + F[a, b + 1, c + 1] + F[a + 1, b + 1, c + 1]
                          - F[a, b, c] - F[a + 1, b, c] - F[a, b, c + 1] - F[a + 1, b, c + 1])
                    gz = (F[a, b, c + 1] + F[a + 1, b, c + 1] + F[a, b + 1, c + 1] + F[a + 1, b + 1, c + 1]
                          - F[a, b, c] - F[a + 1, b, c] - F[a, b + 1, c] - F[a + 1, b + 1, c])
                    if gx * snorm[s, 0] + gy * snorm[s, 1] + gz * snorm[s, 2] <= 0.0:
                        continue
                    if qt >= cap:
                        continue
                    st[a, b, c] = 1
                    sid[a, b, c] = s
                    q[qt] = (a * cj + b) * ck + c
                    qt += 1
        if qt > q0:
            break
    while qh < qt:
        idx = q[qh]
        qh += 1
        a = idx // (cj * ck)
        b = (idx // ck) % cj
        c = idx % ck
        d0 = st[a, b, c]
        if d0 > geo:
            continue
        for k in range(6):
            a2 = a; b2 = b; c2 = c
            if k == 0: a2 -= 1
            elif k == 1: a2 += 1
            elif k == 2: b2 -= 1
            elif k == 3: b2 += 1
            elif k == 4: c2 -= 1
            else: c2 += 1
            if a2 < 0 or b2 < 0 or c2 < 0 or a2 >= ci or b2 >= cj or c2 >= ck or st[a2, b2, c2] > 0:
                continue
            # the shared face's four corners
            neg = False
            pos = False
            for u in range(2):
                for v in range(2):
                    if k < 2:
                        fa = a if k == 0 else a + 1
                        val = F[fa, b + u, c + v]
                    elif k < 4:
                        fb = b if k == 2 else b + 1
                        val = F[a + u, fb, c + v]
                    else:
                        fc = c if k == 4 else c + 1
                        val = F[a + u, b + v, fc]
                    if val < 0.0:
                        neg = True
                    else:
                        pos = True
            if not (neg and pos):
                continue
            x = (i0 + a2 + 0.5) * h; y = (j0 + b2 + 0.5) * h; z = (k0 + c2 + 0.5) * h
            inb = False
            for s in range(ns):
                dx = x - seeds[s, 0]; dy = y - seeds[s, 1]; dz = z - seeds[s, 2]
                if dx * dx + dy * dy + dz * dz <= reach2:
                    inb = True
                    break
            if not inb:
                continue
            if cosc > -1.0:
                e0 = sid[a, b, c]
                gx = (F[a2 + 1, b2, c2] + F[a2 + 1, b2 + 1, c2] + F[a2 + 1, b2, c2 + 1] + F[a2 + 1, b2 + 1, c2 + 1]
                      - F[a2, b2, c2] - F[a2, b2 + 1, c2] - F[a2, b2, c2 + 1] - F[a2, b2 + 1, c2 + 1])
                gy = (F[a2, b2 + 1, c2] + F[a2 + 1, b2 + 1, c2] + F[a2, b2 + 1, c2 + 1] + F[a2 + 1, b2 + 1, c2 + 1]
                      - F[a2, b2, c2] - F[a2 + 1, b2, c2] - F[a2, b2, c2 + 1] - F[a2 + 1, b2, c2 + 1])
                gz = (F[a2, b2, c2 + 1] + F[a2 + 1, b2, c2 + 1] + F[a2, b2 + 1, c2 + 1] + F[a2 + 1, b2 + 1, c2 + 1]
                      - F[a2, b2, c2] - F[a2 + 1, b2, c2] - F[a2, b2 + 1, c2] - F[a2 + 1, b2 + 1, c2])
                gn = math.sqrt(gx * gx + gy * gy + gz * gz)
                if gn > 1e-12 and (gx * snorm[e0, 0] + gy * snorm[e0, 1] + gz * snorm[e0, 2]) / gn < cosc:
                    continue                     # the surface has turned away from the brush (the default's limit)
            if qt >= cap:
                continue
            sid[a2, b2, c2] = sid[a, b, c]
            st[a2, b2, c2] = d0 + 1
            q[qt] = (a2 * cj + b2) * ck + c2
            qt += 1
    cm = np.full((ci, cj, ck), -1, np.int8)
    _sheet_label(F, h, i0, j0, k0, seeds, r, gmin, st, cm, inner2, bandw, out, side)
    _sheet_fill0(F, bandw, out)
    return qt


def _sheet_cache_io(cache, oid, lo, lab, write):
    """the stroke's first-sight labels (dict brick -> uint8 8^3) <-> the dab's box"""
    lo = np.asarray(lo, np.int64)
    hi = lo + np.asarray(lab.shape, np.int64) - 1
    for bi in range(lo[0] >> 3, (hi[0] >> 3) + 1):
        for bj in range(lo[1] >> 3, (hi[1] >> 3) + 1):
            for bk in range(lo[2] >> 3, (hi[2] >> 3) + 1):
                b0 = np.array([bi, bj, bk], np.int64) * 8
                s0 = np.maximum(b0, lo); s1 = np.minimum(b0 + 7, hi) + 1
                box = tuple(slice(int(s0[d] - lo[d]), int(s1[d] - lo[d])) for d in range(3))
                bb = tuple(slice(int(s0[d] - b0[d]), int(s1[d] - b0[d])) for d in range(3))
                key = (oid, bi, bj, bk)
                if write:
                    blk = cache.get(key)
                    if blk is None:
                        if not lab[box].any():
                            continue
                        blk = cache[key] = np.zeros((8, 8, 8), np.uint8)
                    blk[bb] = lab[box]
                else:
                    blk = cache.get(key)
                    if blk is not None:
                        lab[box] = blk[bb]


@njit(parallel=True, cache=True, nogil=True)
def sheet_box_fill(box, vals, off):
    """the card's rows (n, 729: 9^3 nodes, i-major) into the dense box at their node offsets"""
    for t in prange(vals.shape[0]):
        a0 = off[t, 0]; b0 = off[t, 1]; c0 = off[t, 2]
        for q in range(729):
            box[a0 + q // 81, b0 + (q // 9) % 9, c0 + q % 9] = vals[t, q]


@njit(parallel=True, cache=True, nogil=True)
def sheet_box_take(out, off, on, seen_off):
    """the labels back per row: on (n, 729) = on the worked sheet, seen_off = in the reach and not on it"""
    for t in prange(off.shape[0]):
        a0 = off[t, 0]; b0 = off[t, 1]; c0 = off[t, 2]
        for q in range(729):
            o = out[a0 + q // 81, b0 + (q // 9) % 9, c0 + q % 9]
            on[t, q] = (o & 3) == 1
            seen_off[t, q] = (o & 4) != 0 and (o & 3) != 1


def sheet_on(settings):
    """the sheet under the brush applies to this stroke (new strokes: on unless 'sheet_only' is False; strokes recorded
    before it carry no key and replay as they were only if FSB_SHEET=0 - see from_log)"""
    return settings.get('sheet_only', SHEET_DEFAULT) is not False and settings.get('tool') in SHEET_TOOLS


SHEET_DEEPSIDE = os.environ.get('FSB_SHEET_DEEPSIDE', '0') != '0'   # fs-brush 11-12: OFF (it let warps pull another
                                                                    # surface forward: lid case, eye 173 um under Draw r 8)
# fs-core 11-14: ON under the truncated band (gpu/sculpt.py TSDF, default). There every surface's own nodes are frozen in the
# band pass and a node off them takes its distance to the true zero set, so what a dab writes into a node deeper than the
# sheet band on its moving side never reaches another surface; labelled from its foot instead, such a node at the band's
# clamp (no reliable gradient) came out undecided and the dab could not move the worked surface through it (fs-core m1:
# walls from the 2nd overlapping stroke; deep side on: none, as with the sheet off)
if os.environ.get('FSB_TSDF', '1') != '0' and 'FSB_SHEET_DEEPSIDE' not in os.environ:
    SHEET_DEEPSIDE = True
# fs-core 11-14: the truncated band's width (lattice spacings; gpu/sculpt.py TSDF_W_H reads the same variable) and the
# per-brick L measured over the band's cells (brick_lipschitz_band) when the band is on
TSDF_W_H = float(os.environ.get('FSB_TSDF_W', '12'))
TSDF_L = os.environ.get('FSB_TSDF', '1') != '0'


def sheet_side(settings, bp):
    """fs-brush 11-12: the side a dab moves the surface into: 1 a growing dab (outward), -1 a cutting dab (inward), 0 both
    or neither (Smooth, Flatten, Pinch). bp: the dab's parameters after Ctrl's inversion (bp[13] = its amount)"""
    if not SHEET_DEEPSIDE:
        return 0
    tool = settings.get('tool')
    A = float(bp[13])
    if tool == 'carve':
        return 1 if A < 0 else -1
    if tool in ('draw', 'inflate', 'swell', 'clay', 'layer'):
        return 1 if A > 0 else -1
    if tool == 'crease':
        return -1 if A > 0 else 1
    return 0


def sheet_band(A, h, settings):
    """the band (mm) within which a node takes its own foot's label (deeper nodes: their block's)"""
    b = max(SKIP_BAND_H * h, 1.6 * abs(float(A)) + SKIP_BAND_H * h)
    if settings.get('tool') == 'smooth':
        b = max(b, QSM_BAND_H * h + 2 * h)
    return b


def sheet_reach(settings, r, h):
    """-> (geo steps along the surface, cos of the facing limit or -2 for none)"""
    geo = int(math.ceil(SHEET_GEO * (float(r) + 2 * h) / h))
    cone = settings.get('sheet_cone', SHEET_CONE)                # None / >= 180: the whole connected piece (both faces)
    cosc = -2.0 if cone is None or float(cone) >= 180.0 else math.cos(math.radians(float(cone)))
    return geo, cosc


def sheet_gate(pk, oi, lo, shape, h, c, n, r, A, bp, settings, rec=None, oid=0):
    """-> uint8 array (shape) of the nodes this dab may change: its own sheet (None = no gate). rec: the stroke's undo
    record, which carries the stroke's first-sight labels (rec.sheet)"""
    if not sheet_on(settings):
        return None
    F = np.empty(tuple(int(x) for x in shape), np.float64)
    _node_field(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, int(lo[0]), int(lo[1]), int(lo[2]),
                h, F)
    seeds = [np.asarray(c, float)]
    norms = [np.asarray(n, float) / max(float(np.linalg.norm(n)), 1e-12)]
    if bp.shape[0] > NBP_MIRP + 3 and bp[NBP_MIRP + 3] > 0.5:
        seeds.append(np.asarray(bp[NBP_MIRP:NBP_MIRP + 3], float))
        sym = settings.get('symmetry') or [False, False, False]
        nm = np.asarray(n, float).copy()
        for ax in range(3):
            if sym[ax]:
                nm[ax] = -nm[ax]
        norms.append(nm)
    geo, cosc = sheet_reach(settings, r, h)
    out = np.zeros(F.shape, np.uint8)
    cache = None
    if rec is not None:
        cache = getattr(rec, 'sheet', None)
        if cache is None:
            cache = rec.sheet = {}
        _sheet_cache_io(cache, oid, lo, out, False)
    _sheet_flood(F, h, int(lo[0]), int(lo[1]), int(lo[2]), np.ascontiguousarray(np.array(seeds)),
                 np.ascontiguousarray(np.array(norms)), float(r), geo, SHEET_GMIN, cosc, sheet_band(A, h, settings), out,
                 sheet_side(settings, bp))
    if cache is not None:
        _sheet_cache_io(cache, oid, lo, out, True)
    return ((out & 3) == 1).astype(np.uint8)


def region_gate(st, g, lo, shape, settings):
    """fs-brush 11-11 (design 4.8): the 0 / 1 weight of a dab's nodes from the stroke's own settings (so a replay never
    depends on the view): 'skip_regions' (ids hidden when the stroke began: never touched) and 'automask_region' (the id
    under the press when 'This region only' was on: nothing else touched). -> None when neither applies."""
    skip = settings.get('skip_regions') or ()
    only = settings.get('automask_region')
    if not skip and only is None:
        return None
    ids = _box_rid(st, g, lo, shape)
    allow = np.ones(shape, np.float64)
    if skip:
        allow[np.isin(ids, np.asarray(list(skip), np.uint16))] = 0.0
    if only is not None:
        allow[ids != np.uint16(int(only))] = 0.0
    return allow


# fs-brush 11-11 (block-in kit, design 4.5: Snake Hook pulls a form out along the path keeping its section): today's
# Grab (its anchor moves with the drag) with a falloff flat to 0.8 r and a cosine edge to r. M (elastic/snake_probe.py,
# sphere r 60, r 15, an 80 mm pull): the pulled form's section 13.3 -> 11.8 mm base to tip (the smooth falloff: 8.4 -> 1.0,
# a spike), side roughness 60 um (203); around a 90 deg bend the arm thins to ~half (to refine). ui/engine.py turns
# tool 'snake' into this Grab at the press, so logs hold an ordinary Grab with its curve.
# fs-brush 11-12: Snake Hook as a PATH tool with a smooth falloff and Magnify (snake_back);
# FSB_SNAKE_PATH=0 = the flat-topped Grab above. SNAKE_MAGNIFY: measured (fs-brush/snake/), the girth kept along a pull
STRETCH_DK = float(os.environ.get('FSB_STRETCH_DK', '1.0'))   # Stretch's pivot depth <= DK x r (0: the form's whole depth)
SNAKE_PATH = os.environ.get('FSB_SNAKE_PATH', '1') != '0'
SNAKE_MAGNIFY = float(os.environ.get('FSB_SNAKE_MAGNIFY', '1.0'))   # s5 (r 15, 80 mm pull): 0 -> 7.0 / 3.25 / 2.0 mm base / mid / tip, 1.0 -> 8.5 / 7.0 / 5.5
SNAKE_FALLOFF = os.environ.get('FSB_SNAKE_FALLOFF', 'smooth')
SNAKE_SKIP_UNTOUCHED = os.environ.get('FSB_SNAKE_SKIPU', '1') != '0'
SNAKE_COMPOSE = os.environ.get('FSB_SNAKE_COMPOSE', '0') != '0'   # OFF by default: in progress (CONTINUE 22:5x)   # the whole path from the stroke-start field (snake_path_back)
SNAKE_RAKE = float(os.environ.get('FSB_SNAKE_RAKE', '1.0'))     # Blender's Snake Hook default rake 1.0 (toolkit_review/A 1.16)


SNAKE_RAKE_CHORD = 0.25   # Rake reads the path's direction over chords of this share of r (tk3 card1: per mouse event,
                          # ~1 px steps on a 0.49 mm grid, the direction jumped by up to 45 deg a step and twisted the arm)


SNAKE_SPACING = float(os.environ.get('FSB_SNAKE_SPACING', '0.2'))   # a Snake Hook step every 0.2 r of travel


def snake_steps(state, v, r, flush=False):
    """fs-brush 11-12 (s5 S5_PIX: one step per mouse event, ~0.5 mm, thinned an 80 mm arm to 4.5 mm where ~4 mm steps
    kept 6.25: every step re-reads the field, so the form depended on the mouse rate): Snake Hook's steps laid every
    SNAKE_SPACING r of TRAVEL (Blender's spacing), the rest carried to the next move; flush=True (the release) lays the
    rest, so the arm ends at the cursor. -> the step vectors to apply now (each <= 0.35 r)"""
    pend = state.get('pend', np.zeros(3)) + np.asarray(v, float)
    L = float(np.linalg.norm(pend))
    if L < 1e-12 or (L < SNAKE_SPACING * r and not flush):
        state['pend'] = pend
        return []
    k = max(1, int(math.ceil(L / (0.35 * r))))
    state['pend'] = np.zeros(3)
    return [pend / k] * k


def snake_stamp(state, dabs, r):
    """fs-brush 11-12 (Rake): each Snake Hook dab gets the path's direction before it ('vp') and after it ('vr'), both
    scaled to the dab's step, read over chords of SNAKE_RAKE_CHORD r of travel so mouse-event jitter turns nothing; the
    turn happens at the dab where a chord completes, by the angle between successive chords (a 90 deg bend turns the
    carried region 90 deg in total). state: the stroke's dict (dir, acc). These directions are recorded, so replay is exact"""
    for d_ in dabs:
        v_ = np.asarray(d_['v'], float)
        L_ = float(np.linalg.norm(v_))
        if L_ < 1e-12:
            continue
        cur = state.get('dir')
        if cur is None:
            cur = v_ / L_
        acc = state.get('acc', np.zeros(3)) + v_
        new = cur
        la = float(np.linalg.norm(acc))
        if la >= SNAKE_RAKE_CHORD * r:
            new = acc / la
            acc = np.zeros(3)
        d_['vp'] = (cur * L_).tolist()
        d_['vr'] = (new * L_).tolist()
        state['dir'], state['acc'] = new, acc


def snake_vps(obj, smp, c, n, r_l, sym, settings, tool):
    """fs-brush 11-12 (Rake): each mirror copy's path direction before and after a Snake Hook sample (local, the same
    order as mirror_plan's copies): [(vp, vr)], or None when the dab carries none / Rake is off"""
    if not (tool == 'grab' and settings.get('snake', False) and SNAKE_PATH and 'vp' in smp
            and float(settings.get('snake_rake', SNAKE_RAKE)) > 0.0):
        return None
    _, _, vp, _ = world_to_local(obj, smp['p'], smp.get('n', (0, 1, 0)), smp['vp'])
    _, _, vr, _ = world_to_local(obj, smp['p'], smp.get('n', (0, 1, 0)), smp.get('vr', smp['v']))
    return [(a[2], b[2]) for a, b in zip(mirror_plan(c, n, vp, r_l, sym, settings, tool),
                                         mirror_plan(c, n, vr, r_l, sym, settings, tool))]
SNAKE_CURVE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8536, 0.5, 0.1464, 0.0]


REGION_REACH = 0.85       # the Region brush paints the nodes within this share of its radius (the falloff's visible circle)


def region_dab(scene, oi, settings, c, n, r, lo, shape, lut, pk):
    """fs-brush 11-11 (design 4.8): the Region brush's dab. Paints settings['region_id'] (Ctrl / invert: 0, erasing) on the
    lattice nodes within REGION_REACH r of the centre that lie in the surface's band (|F| < max(3 h, 0.25 r)) and on the
    sheet under the brush (the field's gradient within 80 degrees of the pressed normal, so paint never crosses a gap to a
    facing sheet: the mask's direction rule). Hard-edged: an id cannot blend. Bricks were allocated (and their ids
    snapshotted for undo) by apply_sample's brick loop. -> the number of nodes painted."""
    from .program import eval_points_local
    obj = scene.objects[oi]
    g = obj.sculpt
    st = scene.store
    h = g.h
    rid = 0 if settings.get('invert', False) else int(settings.get('region_id', 0))
    I = np.stack(np.meshgrid(*[np.arange(int(lo[a]), int(lo[a]) + shape[a]) for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
    P = I * h
    d = np.linalg.norm(P - c[None], axis=1)
    sel = d <= REGION_REACH * r
    if not np.any(sel):
        return 0
    I = I[sel]
    P = np.ascontiguousarray(P[sel])
    F = np.empty(len(P))
    eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, P, F)
    band = max(3.0 * h, 0.25 * r)
    keep = np.abs(F) < band
    if not np.any(keep):
        return 0
    I = I[keep]
    P = np.ascontiguousarray(P[keep])
    G = np.zeros((len(P), 3))
    for a in range(3):
        Pp = P.copy()
        Pp[:, a] += h
        Pm = P.copy()
        Pm[:, a] -= h
        Fp = np.empty(len(P))
        Fm = np.empty(len(P))
        eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(Pp), Fp)
        eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(Pm), Fm)
        G[:, a] = Fp - Fm
    gl = np.maximum(np.linalg.norm(G, axis=1), 1e-12)
    facing = (G @ n) / gl > math.cos(math.radians(80.0))
    I = I[facing]
    if not len(I):
        return 0
    # connected to the pressed sheet: inside nodes (F <= 0) in the dab, 26-connected through the material to the node
    # nearest the press, and the band nodes outside them that touch that piece (a lip beside the pressed one - same
    # facing, joined only through air within the brush - stays unpainted)
    from scipy import ndimage as _ndi
    lo_ = np.asarray(lo, np.int64)
    box = np.zeros(shape, bool)
    Fb = np.full(shape, np.inf)
    Ib = np.stack(np.meshgrid(*[np.arange(int(lo[a]), int(lo[a]) + shape[a]) for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
    Fa = np.empty(len(Ib))
    eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(Ib * h), Fa)
    Fb = Fa.reshape(shape)
    inside = (Fb <= 0.0) & (np.linalg.norm(Ib * h - c[None], axis=1).reshape(shape) <= REGION_REACH * r)
    lab, nl = _ndi.label(inside, structure=np.ones((3, 3, 3), bool))
    if nl == 0:
        return 0
    ci = np.clip(np.round(c / h).astype(np.int64) - lo_, 0, np.asarray(shape) - 1)
    pts = np.argwhere(lab > 0)
    best = lab[tuple(pts[np.argmin(np.sum((pts - ci) ** 2, 1))])]
    piece = lab == best
    near = _ndi.binary_dilation(piece, structure=np.ones((3, 3, 3), bool), iterations=max(1, int(math.ceil(band / h))))
    J = I - lo_
    I = I[near[J[:, 0], J[:, 1], J[:, 2]] & (piece[J[:, 0], J[:, 1], J[:, 2]] | (Fb[J[:, 0], J[:, 1], J[:, 2]] > 0.0))]
    if not len(I):
        return 0
    B = I >> 3
    W = I & 7
    count = 0
    for bk in np.unique(B, axis=0):
        sl = g.get_slot((int(bk[0]), int(bk[1]), int(bk[2])))
        if sl < 0:
            continue
        m = np.all(B == bk, axis=1)
        w = W[m]
        st.rid[sl, w[:, 0], w[:, 1], w[:, 2]] = rid
        count += int(m.sum())
    return count


def region_at(scene, oid, p_world):
    """the region id of the lattice node nearest a world point on the object (0 = none)"""
    oi = scene.index_of(oid)
    obj = scene.objects[oi]
    g = obj.sculpt
    c, _, _, _ = world_to_local(obj, p_world)
    i = np.round(np.asarray(c, float) / g.h).astype(np.int64)
    sl = g.get_slot((int(i[0]) >> 3, int(i[1]) >> 3, int(i[2]) >> 3))
    if sl < 0:
        return 0
    return int(scene.store.rid[sl, i[0] & 7, i[1] & 7, i[2] & 7])


def apply_sample(scene, oi, settings, c, n, v, rec, lut=None, measure=True, touched=None):
    """Apply one brush sample to scene.objects[oi]; c, n, v in the object's LOCAL frame (radius local too).
    hand1 (2026-11-07): measure=False skips the per-dab brick Lipschitz re-measure and adds the dab's bricks to
    `touched` (a set); apply_stroke measures the union once after all its dabs. Same final Lb (D: a brick's L reads
    only nodes of its own and the neighbouring bricks, and every dab that writes those nodes has the brick in its
    range, so the last measurement of each brick is of the final state either way); 7-52 ms per dab (M, lag diag)."""
    obj = scene.objects[oi]
    g = obj.sculpt
    st = scene.store
    h = g.h
    tool = settings['tool']
    r = float(settings['radius_local'])
    snake_ = tool == 'grab' and settings.get('snake', False) and SNAKE_PATH
    if snake_:                                   # fs-brush 11-12: the dab covers the kernel's travel (snake_dab)
        snake_c0, snake_rt = np.asarray(c, float), max(r, 2 * h)
        c, r = snake_dab(c, v, snake_rt)
    r_tip = r
    if r < 2 * h:
        r = 2 * h
    ftv = None
    if tool in FT_TOOLS:                         # fs-tools 11-14: as gpu/sculpt.py _plan (the ball = radius + Smoothing)
        from ..toolset.ftools import ft_bp as _ft_bp
        ftv = _ft_bp(tool[3:], dict(settings, radius_local=r), h, float(settings.get('dt', 0.0)),
                     float(settings.get('s_local', 1.0)), np.asarray(c, float) if tool == 'ft_flatten' else None)
        r = r + ftv[52]
    elif tool in FT_LINE:                        # fs-tools patch 4: Pencil, the tip where it is (its own size, never the 2 h floor)
        from ..toolset.ftools import ft_bp as _ft_bp
        ftv = _ft_bp('pencil', dict(settings, radius_local=r_tip), h, 0.0, float(settings.get('s_local', 1.0)),
                     direction=settings.get('pencil_dir'))
        ftv.update(_ft_press_frame(scene, oi, rec, np.asarray(c, float), ftv, h))
        r = ftv[63] + 2.0 * ftv[55] + ftv[52] + 2.0 * h       # the ball: the tip, its depth from the press surface, k
    if lut is None:
        lut = falloff_lut(settings.get('falloff', 'smooth'), settings.get('curve'))
    c = np.asarray(c, float)
    n = np.asarray(n, float)
    v = np.asarray(v, float)
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(n)) and np.all(np.isfinite(v))):
        return                                  # hand2: a missed pick (NaN centre) made ensure_index ask for TBs
    n = n / max(np.linalg.norm(n), 1e-12)
    lo = np.floor((c - r) / h).astype(np.int64)
    hi = np.ceil((c + r) / h).astype(np.int64)
    blo = lo >> 3
    bhi = hi >> 3
    if g.ensure_index(blo - 1, bhi):
        scene.touch(index=True)
    # fs-brush 11-12 (fs-bench-full F3, M f3_cpu_probe.py B: a torso Draw r 25 on her CPU path allocated and evaluated
    # EVERY brick of each dab's 50 mm cube, 23 972 bricks for 2 dabs, the GBs and minutes of F3): new strokes of the
    # skip tools work the surface band only, as the card does (nodes beyond the dab's move x 1.6 + 6 h keep their
    # values; the release re-distance rebuilds the band). A brick of the cube whose centre lies farther than that +
    # 1.5 x its half-diagonal is not allocated, snapshotted or evaluated; the kernel never touches its nodes (bmask)
    band_keep = None
    skip_cpu = 0.0
    band_side = 0.0
    if settings.get('cpu_band', False) and tool in SKIP_TOOLS:
        A0 = dab_amount(settings, r)
        A0, _s0 = gained(settings, tool, c, n, r, A0, float(settings['strength']), h, lut)
        skip_cpu = abs(A0) * 1.6 + SKIP_BAND_H * h
        nbx = tuple(int(v) for v in (bhi - blo + 1))
        BK = np.stack(np.meshgrid(*[np.arange(blo[i], bhi[i] + 1) for i in range(3)], indexing='ij'), -1).reshape(-1, 3)
        pk0 = scene.pack()
        Fc = np.empty(len(BK))
        P.eval_points_local(pk0.code, pk0.prm, pk0.oi_tab, pk0.of_tab, pk0.index_flat, pk0.delta, oi,
                            np.ascontiguousarray((BK * 8 + 3.5) * h, np.float64), Fc)
        lim_ = skip_cpu + 1.5 * 0.8661 * 8 * h
        if BAND_ONE_SIDED and settings.get('cpu_band_side', False):
            # fs-brush 11-12 (found with band/t_band_synth.py: 8 Inflate strokes kinked the top 160 um vs 90
            # with every node worked, and grew it 22.52 vs 22.30 mm): a brick skipped on the side the surface moves TOWARD
            # keeps a stale value (too far) that its own band test then keeps reading, so the surface grows into stale bricks.
            # Skip only the side the surface moves AWAY from (it stays on its side of the surface and is never reached):
            # a dab that grows the form (Draw, Inflate, Clay, Swell; Carve inverted) skips deep INSIDE only, one that cuts
            # in (Carve; the others inverted) deep OUTSIDE only. The zero set then reads only current values, as with the
            # band off; the far half of the cube is still saved (fine profile, band/t_band_synth.py: the live band's step
            # 28.6 um at stroke 5, the notch where the top crosses a brick row; one-sided 5-8 um as with the band off)
            A_eff = A0
            if settings.get('invert', False):
                A_eff = invert_dab(tool, A0, 1.0)[0]
            grow = (A_eff < 0) if tool == 'carve' else (A_eff > 0)
            band_side = 1.0 if grow else -1.0
            # the kept side reaches BAND_DEEP bricks further than the band (1: identical to the band off over 8 strokes,
            # 0: drifts, 22.54 vs 22.30 mm): the edge of what the stroke touched must lie away from the surface.
            # M (f3/f3_cpu_probe.py B, her torso Draw r 25, 2 dabs, CPU, warm): 5,248 bricks 2.2 s -> 16,008 / 5.3 s (none: 23,972 / 11.8)
            deep_ = BAND_DEEP * 8 * h
            skip_cpu += deep_
            lim_ += deep_
            band_keep = (Fc >= -lim_) if grow else (Fc <= lim_)
            band_keep = band_keep.reshape(nbx)
        else:
            band_keep = (np.abs(Fc) <= lim_).reshape(nbx)
        if BAND_BALL and settings.get('cpu_band_side', False):
            # fs-brush 11-12: only bricks the dab's ball reaches (the kernel writes no node at d >= r, so a cube-corner
            # brick would be allocated with a zero delta: the same field, fewer bricks)
            dc = np.linalg.norm((BK * 8 + 3.5) * h - c[None, :], axis=1)
            band_keep &= (dc <= r + 0.8661 * 8 * h).reshape(nbx)
    for bx in range(blo[0] - 1, bhi[0] + 1):
        for by in range(blo[1] - 1, bhi[1] + 1):
            for bz in range(blo[2] - 1, bhi[2] + 1):
                b = (bx, by, bz)
                inside = bx >= blo[0] and by >= blo[1] and bz >= blo[2]
                if inside and band_keep is not None and not band_keep[bx - blo[0], by - blo[1], bz - blo[2]]:
                    inside = False                # fs-brush 11-12: beyond the band: as a brick outside the cube
                slot = g.get_slot(b)
                key = (obj.id, b)
                if slot < 0:
                    if not inside:
                        continue
                    slot = st.alloc()
                    g.set_slot(b, slot)
                    rec.new.append(key)
                elif inside:
                    if key not in rec.snap and key not in rec.new_set:
                        rec.snap[key] = (st.delta[slot].copy(), st.mask[slot].copy(), float(st.Lb[slot]))
                    if tool == 'mask' and key not in rec.new_set and key not in getattr(rec, 'dsnap', {}):
                        if not hasattr(rec, 'dsnap'):
                            rec.dsnap = {}
                        rec.dsnap[key] = st.mdir[slot].copy()
                    if tool == 'region' and key not in rec.new_set and key not in getattr(rec, 'rsnap', {}):
                        if not hasattr(rec, 'rsnap'):
                            rec.rsnap = {}
                        rec.rsnap[key] = st.rid[slot].copy()
                else:
                    if key not in rec.snap and key not in rec.new_set and key not in rec.Lsnap:
                        rec.Lsnap[key] = float(st.Lb[slot])
    rec.new_set.update(rec.new)
    pk = scene.pack()
    scene.refresh_object(oi)
    pk = scene.pack()
    dims = (hi - lo + 1)
    if tool == 'region':                         # fs-brush 11-11 (design 4.8): region ids only; the field is untouched
        region_dab(scene, oi, settings, c, n, r, lo, tuple(int(x) for x in dims), lut, pk)
        if not measure and touched is not None:
            touched.update((x, y, z) for x in range(blo[0] - 1, bhi[0] + 1) for y in range(blo[1] - 1, bhi[1] + 1)
                           for z in range(blo[2] - 1, bhi[2] + 1))
        scene.refresh_object(oi)
        return
    df = np.zeros(tuple(int(x) for x in dims), np.float64)
    dm = np.zeros_like(df)
    bp = np.zeros(NBP)
    bp[0] = T[tool]
    bp[1:4] = c
    bp[4:7] = n
    bp[7] = r
    bp[8] = float(settings['strength'])
    bp[9:12] = v
    bp[12] = smooth_step(settings, h, r)
    bp[13] = dab_amount(settings, r)
    bp[13], bp[8] = gained(settings, tool, c, n, r, bp[13], bp[8], h, lut)
    if tool == 'smooth' and settings.get('smooth_quad', False):     # fs-brush 11-11: as gpu/sculpt.py _plan
        bp[NBP_QSM] = 1.0
        bp[NBP_QSM + 1:NBP_QSM + 5] = (QSM_BAND_H, QSM_R1, QSM_R2, QSM_CAP_H)
        bp[8] = math.copysign(min(abs(float(settings['strength'])) * float(settings.get('spacing', 0.25)) / 0.25 * QSM_PASS_K, 1.0),
                              float(settings['strength']))
        smooth_cell_bp(bp, settings)
    if tool in ('clay', 'flatten') and settings.get('area_plane', False):   # fs-brush 11-11: as gpu/sculpt.py _plan
        Pm, Nm = area_plane_cpu(pk, oi, c, n, r, h, lut)
        bp[NBP_PLANE:NBP_PLANE + 3] = Pm
        bp[NBP_PLANE + 3:NBP_PLANE + 6] = Nm
        bp[NBP_PLANE + 6] = 1.0
    if tool == 'stretchflow':                    # fs-brush 11-12: Stretch as an anchored elongation (_stretch_back)
        stretch_bp(bp, c, v, r, settings)
    if snake_:
        snake_bp(bp, settings, snake_c0, snake_rt, v)   # fs-brush 11-12: Snake Hook's step as a flow + Magnify (snake_back)
    if tool == 'grabflow':                       # fs-brush 11-12: the anchored flow Grab (_flow_back; core/history.py)
        rt = r * float(settings.get('flow_ratio', 1.0))
        bp[NBP_PLANE:NBP_PLANE + 3] = c - 0.5 * v   # the press point (the ball sits on the drag's midpoint)
        bp[NBP_PLANE + 3] = rt
        bp[NBP_PLANE + 4] = min(64, max(4, int(math.ceil(float(np.linalg.norm(v)) / (0.15 * rt)))))
        bp[NBP_PLANE + 6] = 0.0
    if tool == 'pinch' and settings.get('pinch_bound', False):     # fs-brush 11-11: as gpu/sculpt.py _plan
        bp[NBP_PINCH] = PINCH_K
    if settings.get('strength_v2', False):      # fs-brush 11-11 (decided): as gpu/sculpt.py _plan
        strength_v2_bp(bp, tool, settings, h)
    if settings.get('invert', False):           # fs-brush 11-11: Ctrl inverts every brush (as gpu/sculpt.py _plan)
        A_i, s_i = invert_dab(tool, bp[13], bp[8])
        bp[13] = A_i
        if not (settings.get('strength_v2', False) and tool in STRENGTH_RESP):
            bp[8] = s_i
    bp[NBP_SHEET] = 1.0 if settings.get('sheet_smooth', False) else 0.0
    bp[NBP_LIM] = 1.0 if settings.get('dab_limit', False) else 0.0
    bp[14] = 1.0 if settings.get('use_mask', True) else 0.0
    bp[15:20] = target_params(settings, h)
    mirror_partner_bp(bp, settings)
    if ftv is not None:                          # fs-tools 11-14: the approved tools' slots (gpu/sculpt.py _plan)
        bp[15:20] = 0.0
        for k_, v_ in ftv.items():
            bp[k_] = v_
    if band_keep is not None:
        bp[NBP_SKIP] = skip_cpu
        bp[NBP_SIDE] = band_side                 # fs-brush 11-12: which side the skip may keep (0: both, as before)
        bmask_ = np.ascontiguousarray(band_keep.astype(np.uint8))
    else:
        bmask_ = np.ones((1, 1, 1), np.uint8)
    steps_ = settings.get('snake_steps_local') if tool == 'snakepath' else None
    mg_ = scope_bp(bp, settings) if tool == 'grabflow' else None   # fs-brush 11-12: Grab's scope (_scope_at)
    if mg_ is None:
        mg_ = np.full((2, 2, 2), 255, np.uint8)
    if steps_ is None:
        steps_ = np.zeros((1, SNAKE_ROW))
    brush_kernel(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, pk.mask, oi,
                 int(lo[0]), int(lo[1]), int(lo[2]), h, bp, lut, df, dm, int(blo[0]), int(blo[1]), int(blo[2]), bmask_,
                 np.ascontiguousarray(steps_, np.float64), mg_)
    if tool == 'swell' and settings.get('swell_toward', False) and not settings.get('invert', False):
        from . import collide as CL               # sculpt-contact 11-12: Swell to contact (collide.swell_ease)
        CL.swell_ease(scene, pk, obj, oi, lo, h, r, df, settings)
    if tool == 'settle':
        from . import collide as CL               # 11-09: the surface moves toward its contacts (see collide.settle_dab)
        CL.settle_dab(scene, pk, obj, oi, lo, h, c, n, r, float(settings['strength']), lut, df, settings)
    sg_ = sheet_gate(pk, oi, lo, df.shape, h, c, n, r, bp[13], bp, settings, rec, obj.id)   # fs-brush 11-12: the sheet under the brush
    if sg_ is not None:
        df *= sg_
    if settings.get('collide', False) or tool in ('swell', 'settle'):
        from . import collide as CL               # 11-09: locked protected parts collide with every edit
        CL.clamp_dab(scene, pk, obj, oi, lo, h, df, dict(settings, collide=True))
    if settings.get('keep_apart', False) and tool != 'mask':
        keep_apart_cpu(pk, oi, lo, h, df,         # 11-09 night: as the card (gpu/sculpt.cu dab_topo_commit)
                       band=(blo, bmask_, np.ascontiguousarray(Fc.reshape(band_keep.shape))) if band_keep is not None else None,
                       sweeps=keep_apart_sweeps(tool, bp, settings, h) if KA_SWEEPS_BY_MOVE else 16)
    allow = region_gate(st, g, lo, df.shape, settings)   # fs-brush 11-11 (design 4.8): hidden regions / 'this region only'
    if allow is not None:
        df *= allow
        dm *= allow
    if not st.has('mask') and dm.any():          # fs-brush 11-12: the first non-zero mask write makes the mask real
        st.ensure('mask')
        scene.pack()                              # (the same pack takes the new view)
    write_kernel(pk.index_flat, pk.delta, pk.mask, pk.oi_tab, oi, int(lo[0]), int(lo[1]), int(lo[2]), df, dm)
    if tool == 'mask' and settings.get('paint_dir', False):
        if not st.has('mdir'):
            st.ensure('mdir')
            scene.pack()
        ns = sheet_normal(pk, oi, c, n, r, h)
        write_dir(pk.index_flat, pk.mask, st.mdir, pk.oi_tab, oi, int(lo[0]), int(lo[1]), int(lo[2]), dm,
                  float(ns[0]), float(ns[1]), float(ns[2]))
    if not measure:
        if touched is not None:
            touched.update((x, y, z) for x in range(blo[0] - 1, bhi[0] + 1) for y in range(blo[1] - 1, bhi[1] + 1)
                           for z in range(blo[2] - 1, bhi[2] + 1))
        scene.refresh_object(oi)
        return
    br = [b for b in ((x, y, z) for x in range(blo[0] - 1, bhi[0] + 1) for y in range(blo[1] - 1, bhi[1] + 1)
                      for z in range(blo[2] - 1, bhi[2] + 1)) if g.get_slot(b) >= 0]
    if br:
        arr = np.array(br, np.int64)
        out = np.zeros(len(br))
        brick_lipschitz(pk.index_flat, pk.delta, pk.oi_tab, oi, h, arr, out)
        for b, L in zip(br, out):
            st.Lb[g.get_slot(b)] = L
    scene.refresh_object(oi)


def undo_record(scene, rec):
    st = scene.store
    touched = set()
    sls, ds, ms, Ls = [], [], [], []                      # r28: restored in one assignment
    for (oid, b), (d, m, L) in rec.snap.items():
        sls.append(scene.get(oid).sculpt.get_slot(b)); ds.append(d); ms.append(m); Ls.append(L)
        touched.add(oid)
    if sls:
        sl_ = np.array(sls, np.int64)
        st.delta[sl_] = np.stack(ds)
        st.mask[sl_] = np.stack(ms)
        st.Lb[sl_] = np.array(Ls)
    for (oid, b), dd in getattr(rec, 'dsnap', {}).items():
        s_ = scene.get(oid).sculpt.get_slot(b)
        if s_ >= 0:
            st.mdir[s_] = dd
    for (oid, b), rr in getattr(rec, 'rsnap', {}).items():     # fs-brush 11-11 (design 4.8)
        s_ = scene.get(oid).sculpt.get_slot(b)
        if s_ >= 0:
            st.rid[s_] = rr
    for (oid, b), L in rec.Lsnap.items():
        g = scene.get(oid).sculpt
        st.Lb[g.get_slot(b)] = L
        touched.add(oid)
    for (oid, b) in reversed(rec.new):
        g = scene.get(oid).sculpt
        s = g.get_slot(b)
        if s >= 0:
            st.release(s)
            g.set_slot(b, -1)
        touched.add(oid)
    for oid in touched:
        g = scene.get(oid).sculpt
        if g.slots:
            ks = np.array(list(g.slots.keys()), np.int64)
            g.bmin, g.bmax = ks.min(0), ks.max(0)
        else:
            g.bmin = g.bmax = None
        scene.refresh_object(scene.index_of(oid))
    scene.touch()


def world_to_local(obj, p, n=None, v=None):
    from .ops import rot_matrix
    t = obj.transform
    R = rot_matrix(t['rotate'])
    s = max(float(t['scale']), 1e-6)
    pl = R.T @ (np.asarray(p, float) - np.asarray(t['translate'], float)) / s
    nl = None if n is None else R.T @ np.asarray(n, float)
    vl = None if v is None else R.T @ np.asarray(v, float) / s
    return pl, nl, vl, s


def mirror_target(scene, p, exclude_hidden=True):
    """hand2f: world symmetry. -> index of the object whose surface is nearest the world point p (smallest |F_i|,
    world units; ties by scene order), or -1. Deterministic (the logged scene state alone: objects whose logged
    'visible' is off are skipped; view-hidden ones are NOT), so replay picks the same one."""
    pk = scene.pack()
    best, bi = 1e30, -1
    pt = np.ascontiguousarray(np.asarray(p, np.float64).reshape(1, 3))
    out = np.empty(1)
    for oi, o in enumerate(scene.objects):
        # hand2b: only the LOGGED 'visible' property may decide (verifier item 15): view-hide is view state, not in
        # the op log, so filtering by it sent the mirrored dab to another object on replay (hash mismatch)
        if exclude_hidden and not o.visible:
            continue
        q, _, _, s = world_to_local(o, p)
        P.eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi,
                            np.ascontiguousarray(q.reshape(1, 3)), out)
        f = abs(float(out[0])) * s
        if f < best:
            best, bi = f, oi
    return bi


def apply_stroke(scene, oid, settings, samples, rec):
    """samples: list of {'p': world point, 'n': world normal, 'v': world drag vector (grab/stretch)}.
    hand2f: settings['sym_world'] True mirrors about the WORLD planes through the origin (x = 0 is her midline): each
    mirrored dab goes to the object whose surface is nearest the mirrored world point (mirror_target), so a stroke on
    a left-arm object lands on the right-arm object. False (default, and every older log): the object's own frame."""
    oi = scene.index_of(oid)
    obj = scene.objects[oi]
    lut = falloff_lut(settings.get('falloff', 'smooth'), settings.get('curve'))
    sym = settings.get('symmetry', [False, False, False])
    world = bool(settings.get('sym_world', False))
    touched = {oi: set()}
    pl_ = None
    for smp in samples:
        c, n, v, s = world_to_local(obj, smp['p'], smp.get('n', (0, 1, 0)), smp.get('v', (0, 0, 0)))
        st = dict(settings)
        st['radius_local'] = float(settings['radius']) / s
        if st['tool'] in FT_TOOLS:              # fs-tools 11-14: held tools: the dab covers dt seconds (gpu/sculpt.py _expand)
            st['dt'] = float(smp.get('dt', 0.0))
            st['s_local'] = float(s)
        if st['tool'] in FT_LINE:               # fs-tools patch 4: Pencil's direction: the dab's 'v' (the window's path
            st['s_local'] = float(s)            # direction), else from the previous dab of this call (local)
            if np.linalg.norm(v) > 0.0:
                st['pencil_dir'] = np.asarray(v, float).tolist()
            elif pl_ is not None:
                st['pencil_dir'] = (np.asarray(c, float) - pl_).tolist()
            pl_ = np.asarray(c, float)
        line_ = st['tool'] in FT_LINE           # fs-tools patch 4: Pencil's depth is constant (by design): no pen
        if 'k' in smp and not line_:            # hand2: a partial dab (slow strokes) carries its share of strength
            st['strength'] = float(settings['strength']) * float(smp['k'])
        if 'f' in smp and not line_:            # fs-brush 11-11: a partial dab's path share (frac_of)
            st['frac'] = float(smp['f'])
            st['strength'] = float(st['strength']) * st['frac']
        if not world:
            # fs-brush 11-11 (design C4): midline snap (mirror_dabs); 'mid_snap' off in older logs
            r_l = max(float(st['radius_local']), 2 * float(obj.sculpt.h))
            # fs-tools 11-14: the approved tools reach radius + Smoothing: the midline snap and the feather pair see that
            r_m = r_l + (float(st.get('smoothing', 0.0)) / s if st['tool'] in FT_ALL else 0.0)
            vps_ = snake_vps(obj, smp, c, n, r_l, sym, settings, st['tool'])   # fs-brush 11-12: Rake
            for ci_, (c_, n_, v_, w_, pc_) in enumerate(mirror_plan(c, n, v, r_m, sym, settings, st['tool'])):
                st_w = st if w_ >= 1.0 else dict(st, strength=float(st['strength']) * w_)
                if pc_ is not None:
                    st_w = dict(st_w); st_w['mirror_partner'] = pc_
                if vps_ is not None:
                    st_w = dict(st_w, snake_vp=vps_[ci_][0], snake_vr=vps_[ci_][1])
                if st['tool'] in FT_LINE and ci_ > 0 and np.linalg.norm(v_) > 0.0:
                    # fs-core-3 11-14 (landing patch 4): a mirrored tip its own direction, as gpu/sculpt.py _expand
                    st_w = dict(st_w, pencil_dir=np.asarray(v_, float).tolist())
                vv = v_ * w_ if st['tool'] in MOVE_TOOLS else v_
                apply_sample(scene, oi, st_w, c_, n_, vv, rec, lut, measure=False, touched=touched[oi])
                a_s = float(settings.get('autosmooth', AUTOSMOOTH.get(st['tool'], 0.0)))
                if st['tool'] in FT_ALL:                # fs-tools 11-14: their Smoothing is the profile's foot, no pass
                    a_s = 0.0
                if a_s > 0.0:
                    # fs-brush 11-11: autosmooth on the CPU path too (as gpu/sculpt.py _expand)
                    k_ = 1.0
                    c_s = c_
                    if st['tool'] in MOVE_TOOLS:
                        k_ = min(float(np.linalg.norm(vv)) / max(float(settings.get('spacing', 0.25)) * r_l, 1e-12), 1.0)
                        c_s = c_ + vv
                    if k_ > 0.0:
                        st_s = dict(st, tool='smooth', strength=a_s * k_ * w_, smooth_quad=True, invert=False,
                                    radius_local=float(st['radius_local']) * AUTOSMOOTH_R)
                        apply_sample(scene, oi, st_s, c_s, n_, np.zeros(3), rec, falloff_lut(st_s.get('falloff', 'smooth'), st_s.get('curve')),
                                     measure=False, touched=touched[oi])
            continue
        for gi, sg in enumerate(_symmetry_signs(sym)):
            if gi == 0:
                apply_sample(scene, oi, st, c * sg, n * sg, v * sg, rec, lut, measure=False, touched=touched[oi])
                continue
            pw = np.asarray(smp['p'], float) * sg
            ti = mirror_target(scene, pw)
            if ti < 0:
                continue
            o2 = scene.objects[ti]
            c2, n2, v2, s2 = world_to_local(o2, pw, np.asarray(smp.get('n', (0, 1, 0)), float) * sg,
                                            np.asarray(smp.get('v', (0, 0, 0)), float) * sg)
            st2 = dict(st)
            st2['radius_local'] = float(settings['radius']) / s2
            if 's_local' in st2:
                st2['s_local'] = float(s2)
            apply_sample(scene, ti, st2, c2, n2, v2, rec, lut, measure=False, touched=touched.setdefault(ti, set()))
    for k in sorted(touched):
        measure_bricks(scene, k, touched[k])


def measure_bricks(scene, oi, bricks):
    """hand1: re-measure the Lipschitz bound of the given bricks (the allocated ones) once, after a batch of dabs."""
    g = scene.objects[oi].sculpt
    st = scene.store
    br = sorted(b for b in bricks if g.get_slot(b) >= 0)
    if not br:
        return
    pk = scene.pack()
    arr = np.array(br, np.int64)
    out = np.zeros(len(br))
    brick_lipschitz(pk.index_flat, pk.delta, pk.oi_tab, oi, g.h, arr, out)
    for b, L in zip(br, out):
        st.Lb[g.get_slot(b)] = L
    scene.refresh_object(oi)


@njit(parallel=True, cache=True, nogil=True)
def brick_lipschitz_band(index_flat, delta, oi_tab, oi, h, bricks, F9, W, out):
    """fs-core 11-14 (the truncated band, gpu/sculpt.py TSDF): brick_lipschitz over the cells the surface can be in only.
    F9 (n, 9, 9, 9): the field at each brick's 8^3 nodes and its +1 apron (the card's values); a cell (lower corner
    8 b .. 8 b + 7) counts when one of its 8 corners has |F| < W. A cell with every corner at or beyond the band holds
    lower bounds of the distance (the band clamps there, against the unclamped field next to it), so a tracer's step
    F / L from it never passes a surface whatever its slope; measured, those clamp steps read as slopes of 30-700 (the HUD's
    L after one click: 52) and the mirror's admission test, which scales with L, took every cell for a candidate"""
    off = oi_tab[oi, P.OI_IDX]
    nx = oi_tab[oi, P.OI_NX]
    ny = oi_tab[oi, P.OI_NY]
    nz = oi_tab[oi, P.OI_NZ]
    ox = oi_tab[oi, P.OI_OX]
    oy = oi_tab[oi, P.OI_OY]
    oz = oi_tab[oi, P.OI_OZ]
    Wn = 0.98 * W                     # a clamped node reads back as W to a rounding (fp16 on the card, f32 sums on the CPU)
    for q in prange(bricks.shape[0]):
        best = 0.0
        bi = bricks[q, 0] * 8
        bj = bricks[q, 1] * 8
        bk = bricks[q, 2] * 8
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    near = False
                    for t in range(8):
                        if abs(F9[q, i + (t & 1), j + ((t >> 1) & 1), k + ((t >> 2) & 1)]) < Wn:
                            near = True
                    if not near:
                        continue
                    v = np.empty(8)
                    for t in range(8):
                        v[t] = _node(index_flat, delta, off, nx, ny, nz, ox, oy, oz,
                                     bi + i + (t & 1), bj + j + ((t >> 1) & 1), bk + k + ((t >> 2) & 1))
                    gx = max(max(abs(v[1] - v[0]), abs(v[3] - v[2])), max(abs(v[5] - v[4]), abs(v[7] - v[6])))
                    gy = max(max(abs(v[2] - v[0]), abs(v[3] - v[1])), max(abs(v[6] - v[4]), abs(v[7] - v[5])))
                    gz = max(max(abs(v[4] - v[0]), abs(v[5] - v[1])), max(abs(v[6] - v[2]), abs(v[7] - v[3])))
                    g = math.sqrt(gx * gx + gy * gy + gz * gz) / h
                    if g > best:
                        best = g
        out[q] = best


@njit(parallel=True, cache=True, nogil=True)
def brick_F9(code, prm, oi_tab, of_tab, index_flat, delta, oi, bricks, h, out):
    """fs-core 11-14: the object's local field at each sculpt brick's 9^3 nodes (its 8^3 and the +1 apron) -> out
    (n, 9, 9, 9): what brick_lipschitz_band needs where no card values are at hand (a project opened)"""
    for q in prange(bricks.shape[0]):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    out[q, i, j, k] = P.eval_local_t(code, prm, oi_tab, of_tab, index_flat, delta, oi,
                                                     (bricks[q, 0] * 8 + i) * h, (bricks[q, 1] * 8 + j) * h,
                                                     (bricks[q, 2] * 8 + k) * h, vs, ps, 1e9)


def measure_bricks_band(scene, oi, bricks, F9, W):
    """fs-core 11-14: measure_bricks over the band's cells (brick_lipschitz_band); bricks a list of keys, F9 their
    (n, 9, 9, 9) field values in the same order"""
    g = scene.objects[oi].sculpt
    st = scene.store
    idx = [t for t, b in enumerate(bricks) if g.get_slot(b) >= 0]
    if not idx:
        return
    pk = scene.pack()
    arr = np.array([bricks[t] for t in idx], np.int64)
    out = np.zeros(len(idx))
    brick_lipschitz_band(pk.index_flat, pk.delta, pk.oi_tab, oi, g.h, arr,
                         np.ascontiguousarray(np.asarray(F9, np.float32)[idx]), float(W), out)
    for b, L in zip(arr.tolist(), out):
        st.Lb[g.get_slot(tuple(b))] = L
    scene.refresh_object(oi)


def space_samples(points, normals, radius, spacing):
    """Resample a polyline of surface hits to one sample every spacing * radius (the stroke's dabs)."""
    pts = np.asarray(points, float).reshape(-1, 3)
    nrm = np.asarray(normals, float).reshape(-1, 3)
    step = max(float(spacing) * float(radius), 1e-6)
    out = [{'p': pts[0].tolist(), 'n': nrm[0].tolist()}]
    acc = 0.0
    for i in range(1, len(pts)):
        seg = pts[i] - pts[i - 1]
        L = float(np.linalg.norm(seg))
        t = step - acc
        while t <= L:
            q = pts[i - 1] + seg * (t / L)
            nn = nrm[i - 1] + (nrm[i] - nrm[i - 1]) * (t / L)
            out.append({'p': q.tolist(), 'n': (nn / max(np.linalg.norm(nn), 1e-12)).tolist()})
            t += step
        acc = L - (t - step)
    return out


# ---- re-distancing after a stroke (the stroke-level fix for steep fields) ---------------------------------------
# Dabs stack along a stroke (each re-samples the field at a displaced point), so |grad F| grows with the stroke
# length (one Carve stroke: L 1.6 -> 18). After the stroke, the touched bricks are re-distanced: F at their lattice
# nodes is replaced by the distance to the zero set (sign kept), so |grad F| ~ 1 and the tracked L drops again.
# Zero set held: the nodes next to a crossing get the subcell intercept distance (exact for a planar front), the
# rest come from fast sweeping (Godunov, 8 orderings, single thread = deterministic). A 1-brick read-only halo
# gives the crossings just outside the region; a 16-node ramp at the region's edge blends into the untouched field
# (M, one Carve stroke: a 4-node ramp gained nothing, 8 -> sculpt L 2.79, 16 -> 2.24, 32 -> 2.93).

@njit(parallel=True, cache=True, nogil=True)
def node_field(code, prm, oi_tab, of_tab, index_flat, delta, oi, i0, j0, k0, h, act, out):
    """F = F_base + delta at lattice nodes (i0 + a, j0 + b, k0 + c) where act > 0 (else 1e30)."""
    ni, nj, nk = out.shape
    off = oi_tab[oi, P.OI_IDX]
    nx = oi_tab[oi, P.OI_NX]
    ny = oi_tab[oi, P.OI_NY]
    nz = oi_tab[oi, P.OI_NZ]
    ox = oi_tab[oi, P.OI_OX]
    oy = oi_tab[oi, P.OI_OY]
    oz = oi_tab[oi, P.OI_OZ]
    c0 = oi_tab[oi, P.OI_C0]
    c1 = oi_tab[oi, P.OI_C1]
    for a in prange(ni):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        for b in range(nj):
            for c in range(nk):
                if act[a, b, c] == 0:
                    out[a, b, c] = 1e30
                    continue
                fb = P.eval_prog(code, prm, c0, c1, (i0 + a) * h, (j0 + b) * h, (k0 + c) * h, vs, ps)
                out[a, b, c] = fb + _node(index_flat, delta, off, nx, ny, nz, ox, oy, oz, i0 + a, j0 + b, k0 + c)


@njit(cache=True, nogil=True)
def sweep_distance(F, act, h, U):
    """Unsigned distance to the zero set of F over the active nodes. U out (1e30 = unreachable)."""
    ni, nj, nk = F.shape
    BIG = 1e30
    fixed = np.zeros(F.shape, np.uint8)
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                U[i, j, k] = BIG
                if act[i, j, k] == 0:
                    continue
                f = F[i, j, k]
                if f == 0.0:
                    U[i, j, k] = 0.0
                    fixed[i, j, k] = 1
                    continue
                inv = 0.0
                mb = BIG
                for ax in range(3):
                    best = BIG
                    for sg in (-1, 1):
                        a = i + sg if ax == 0 else i
                        b = j + sg if ax == 1 else j
                        c = k + sg if ax == 2 else k
                        if a < 0 or b < 0 or c < 0 or a >= ni or b >= nj or c >= nk:
                            continue
                        if act[a, b, c] == 0:
                            continue
                        g = F[a, b, c]
                        if (f > 0.0) != (g > 0.0):
                            th = f / (f - g)
                            if th * h < best:
                                best = th * h
                    if best < BIG:
                        inv += 1.0 / max(best * best, 1e-300)
                        mb = min(mb, best)
                if inv > 0.0:
                    # hand2 (2026-11-07): seed |F| / |grad F| (central differences, one-sided at the region edge):
                    # it keeps the old zero set to second order and is smooth node to node; the per-axis intercept
                    # formula (kept as the fallback) moved each crossing on its own and wrote 1 mm grain into the
                    # delta (M, grain diag H3/H3b; L within 0.1 of before)
                    gg = 0.0
                    for ax in range(3):
                        ap_ = i + 1 if ax == 0 else i
                        bp_ = j + 1 if ax == 1 else j
                        cp_ = k + 1 if ax == 2 else k
                        am_ = i - 1 if ax == 0 else i
                        bm_ = j - 1 if ax == 1 else j
                        cm_ = k - 1 if ax == 2 else k
                        okp = ap_ < ni and bp_ < nj and cp_ < nk and act[ap_, bp_, cp_] > 0
                        okm = am_ >= 0 and bm_ >= 0 and cm_ >= 0 and act[am_, bm_, cm_] > 0
                        if okp and okm:
                            dd = (F[ap_, bp_, cp_] - F[am_, bm_, cm_]) / (2.0 * h)
                        elif okp:
                            dd = (F[ap_, bp_, cp_] - f) / h
                        elif okm:
                            dd = (f - F[am_, bm_, cm_]) / h
                        else:
                            dd = 0.0
                        gg += dd * dd
                    if gg > 1e-12:
                        # 11-09 (a NaN): at a thin sheet or ridge the central differences cancel (|grad| ~1e-6)
                        # and |F| / |grad F| reached 0.6 m - 350 km, written into the delta by a forced re-distance (the
                        # card's fp16 then held inf; Smooth made NaN). A crossing found on an axis is a point of the
                        # zero set, so the distance is at most the nearest one: the seed is capped there.
                        U[i, j, k] = min(abs(f) / math.sqrt(gg), mb)
                    else:
                        U[i, j, k] = 1.0 / math.sqrt(inv)
                    fixed[i, j, k] = 1
    for it in range(4):
        change = 0.0
        for s in range(8):
            si = 1 if (s & 1) == 0 else -1
            sj = 1 if (s & 2) == 0 else -1
            sk = 1 if (s & 4) == 0 else -1
            for ii in range(ni):
                i = ii if si > 0 else ni - 1 - ii
                for jj in range(nj):
                    j = jj if sj > 0 else nj - 1 - jj
                    for kk in range(nk):
                        k = kk if sk > 0 else nk - 1 - kk
                        if act[i, j, k] == 0 or fixed[i, j, k] == 1:
                            continue
                        a = BIG
                        if i > 0 and act[i - 1, j, k] > 0:
                            a = U[i - 1, j, k]
                        if i < ni - 1 and act[i + 1, j, k] > 0 and U[i + 1, j, k] < a:
                            a = U[i + 1, j, k]
                        b = BIG
                        if j > 0 and act[i, j - 1, k] > 0:
                            b = U[i, j - 1, k]
                        if j < nj - 1 and act[i, j + 1, k] > 0 and U[i, j + 1, k] < b:
                            b = U[i, j + 1, k]
                        c = BIG
                        if k > 0 and act[i, j, k - 1] > 0:
                            c = U[i, j, k - 1]
                        if k < nk - 1 and act[i, j, k + 1] > 0 and U[i, j, k + 1] < c:
                            c = U[i, j, k + 1]
                        if a > b:
                            a, b = b, a
                        if b > c:
                            b, c = c, b
                        if a > b:
                            a, b = b, a
                        if a >= BIG:
                            continue
                        u = a + h
                        if u > b:
                            u = 0.5 * (a + b + math.sqrt(max(2.0 * h * h - (a - b) * (a - b), 0.0)))
                            if u > c:
                                sm = a + b + c
                                disc = sm * sm - 3.0 * (a * a + b * b + c * c - h * h)
                                u = (sm + math.sqrt(max(disc, 0.0))) / 3.0
                        if u < U[i, j, k]:
                            if U[i, j, k] < BIG:
                                change = max(change, U[i, j, k] - u)
                            else:
                                change = max(change, h)
                            U[i, j, k] = u
        if change < 1e-3 * h:
            break


@njit(cache=True, nogil=True)
def edge_ramp(reg, w, out):
    """out = min(1, (city-block node distance to the nearest node outside reg) / w) over reg nodes, else 0."""
    ni, nj, nk = reg.shape
    BIG = 1 << 20
    d = np.empty(reg.shape, np.int64)
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                if reg[i, j, k] == 0 or i == 0 or j == 0 or k == 0 or i == ni - 1 or j == nj - 1 or k == nk - 1:
                    d[i, j, k] = 0
                else:
                    d[i, j, k] = BIG
    for i in range(1, ni):
        for j in range(1, nj):
            for k in range(1, nk):
                v = min(d[i - 1, j, k], min(d[i, j - 1, k], d[i, j, k - 1])) + 1
                if v < d[i, j, k]:
                    d[i, j, k] = v
    for i in range(ni - 2, -1, -1):
        for j in range(nj - 2, -1, -1):
            for k in range(nk - 2, -1, -1):
                v = min(d[i + 1, j, k], min(d[i, j + 1, k], d[i, j, k + 1])) + 1
                if v < d[i, j, k]:
                    d[i, j, k] = v
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                out[i, j, k] = min(1.0, d[i, j, k] / w)


_NB26 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


@njit(parallel=True, cache=True, nogil=True)
def brick_lipschitz_out(index_flat, delta, oi_tab, oi, h, bricks, Fn, i0, j0, k0, thr, out):
    """V1.3: brick_lipschitz over the cells that can hold F >= 0 only: a cell counts when one of its 8 nodes has
    Fn >= -thr, or a node lies outside Fn / is unknown (1e30). thr = L_all * sqrt(3) * h, L_all a bound valid
    everywhere, so a skipped cell has F < 0 at every point (D). The tracer only needs |grad F| <= L on F >= 0: from
    a point outside, the path to the first surface point stays in F >= 0, so F(p) / L is still a safe step."""
    off = oi_tab[oi, P.OI_IDX]
    nx = oi_tab[oi, P.OI_NX]
    ny = oi_tab[oi, P.OI_NY]
    nz = oi_tab[oi, P.OI_NZ]
    ox = oi_tab[oi, P.OI_OX]
    oy = oi_tab[oi, P.OI_OY]
    oz = oi_tab[oi, P.OI_OZ]
    ni, nj, nk = Fn.shape
    for q in prange(bricks.shape[0]):
        best = 0.0
        bi = bricks[q, 0] * 8
        bj = bricks[q, 1] * 8
        bk = bricks[q, 2] * 8
        v = np.empty(8)
        for i in range(bi - 1, bi + 8):
            for j in range(bj - 1, bj + 8):
                for k in range(bk - 1, bk + 8):
                    use = False
                    for t in range(8):
                        a = i + (t & 1) - i0
                        b = j + ((t >> 1) & 1) - j0
                        c = k + ((t >> 2) & 1) - k0
                        if a < 0 or b < 0 or c < 0 or a >= ni or b >= nj or c >= nk:
                            use = True
                            break
                        f = Fn[a, b, c]
                        if f >= 1e29 or f >= -thr:
                            use = True
                            break
                    if not use:
                        continue
                    for t in range(8):
                        v[t] = _node(index_flat, delta, off, nx, ny, nz, ox, oy, oz,
                                     i + (t & 1), j + ((t >> 1) & 1), k + ((t >> 2) & 1))
                    gx = max(max(abs(v[1] - v[0]), abs(v[3] - v[2])), max(abs(v[5] - v[4]), abs(v[7] - v[6])))
                    gy = max(max(abs(v[2] - v[0]), abs(v[3] - v[1])), max(abs(v[6] - v[4]), abs(v[7] - v[5])))
                    gz = max(max(abs(v[4] - v[0]), abs(v[5] - v[1])), max(abs(v[6] - v[2]), abs(v[7] - v[3])))
                    g = math.sqrt(gx * gx + gy * gy + gz * gz) / h
                    if g > best:
                        best = g
        out[q] = best


# V1.3b: the narrow-band growth is measured but OFF by default (BAND = 0): it changes the re-distance result, so
# saved logs would replay to new hashes, and under full CPU load the demo's first stroke took 11.7 s (M). Set
# BAND = 3 to use it (one Carve stroke: object L 2.67 -> 2.16, M).
BAND = 0
# hand2: re-distance is skipped when the region's DELTA bound (st.Lb) is at or under this. hand2b (verifier item 19):
# its comment claimed '|grad F| near 1', which it does not test (the tracer reads L_base + L_delta); the measured
# premise is in redistance(). A second skip tests |grad F| itself: central differences of F = base + delta at the
# region's nodes within SKIP_BAND h of the surface, 1st..99th percentile inside [SKIP_G_LO, SKIP_G_HI] (the field is
# already a distance where the tracer meets it)
SKIP_L = 1.3
SKIP_G_LO = 0.8
SKIP_G_HI = 1.3
SKIP_BAND = 2.0


def _near_unit_gradient(F, reg, h):
    """-> (ok, p1, p99): |grad F| on region nodes with |F| < SKIP_BAND h whose 6 neighbours are known. ok False when
    there are no such nodes."""
    big = F >= 1e29
    c = (slice(1, -1),) * 3
    Fc = F[c]
    gx = F[2:, 1:-1, 1:-1] - F[:-2, 1:-1, 1:-1]
    gy = F[1:-1, 2:, 1:-1] - F[1:-1, :-2, 1:-1]
    gz = F[1:-1, 1:-1, 2:] - F[1:-1, 1:-1, :-2]
    known = ~(big[2:, 1:-1, 1:-1] | big[:-2, 1:-1, 1:-1] | big[1:-1, 2:, 1:-1] | big[1:-1, :-2, 1:-1]
              | big[1:-1, 1:-1, 2:] | big[1:-1, 1:-1, :-2] | big[c])
    sel = known & (reg[c] > 0) & (np.abs(np.where(big[c], 1e30, Fc)) < SKIP_BAND * h)
    if not sel.any():
        return False, 0.0, 0.0
    g = np.sqrt(gx[sel] ** 2 + gy[sel] ** 2 + gz[sel] ** 2) / (2.0 * h)
    p1, p99 = (float(v) for v in np.percentile(g, [1.0, 99.0]))
    return (SKIP_G_LO <= p1 and p99 <= SKIP_G_HI), p1, p99


def _surface_bricks(scene, oi, R):
    """Bricks of R whose nodes hold both signs of F (the new zero set passes through them)."""
    g = scene.objects[oi].sculpt
    Ra = np.array(R, np.int64)
    blo = Ra.min(0)
    nb = tuple(int(v) for v in (Ra.max(0) - blo + 1))
    role = np.zeros(nb, np.uint8)
    role[tuple((Ra - blo).T)] = 1
    act = np.ascontiguousarray(np.repeat(np.repeat(np.repeat(role, 8, 0), 8, 1), 8, 2))
    pk = scene.pack()
    F = np.empty(act.shape, np.float64)
    i0, j0, k0 = (int(v) for v in blo * 8)
    node_field(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, i0, j0, k0, g.h, act, F)
    out = []
    for b in R:
        a = (np.array(b) - blo) * 8
        f = F[a[0]:a[0] + 8, a[1]:a[1] + 8, a[2]:a[2] + 8]
        if f.min() <= 0.0 <= f.max():
            out.append(b)
    return out


def grow_band(scene, oid, R, rec, n=3, src=None, moved=4.0):
    """V1.3b: the re-distance region = the stroke's bricks + every brick within n bricks (Chebyshev) of a brick the
    new zero set passes through. Cause (M, one Carve stroke on a cube): the stroke's bricks end about 6 mm under a
    10 mm trench floor, so the 16-node edge ramp, which carries the old-minus-new distance mismatch (about the
    trench depth), lay right under the floor in cells with F >= -thr, where the tracer reads L (sculpt L 1.67 there).
    With n = 3 the ramp (2 bricks) starts at least 3 bricks from the surface, past thr. Bricks it adds that hold no
    sculpt are allocated (zero delta) into rec.new, so the stroke's undo releases them.
    Only surface bricks where this stroke moved the field by >= moved * h (src = the stroke's own record: its
    snapshots hold the delta before the stroke) seed the band: the mismatch the ramp carries is about that move, and
    growing around every surface brick cost 6x in stroke time (M, core T3: 15 -> 100 s for 50 strokes)."""
    oi = scene.index_of(oid)
    obj = scene.objects[oi]
    g = obj.sculpt
    st = scene.store
    S = _surface_bricks(scene, oi, R)
    if src is not None:
        lim = moved * g.h
        keep = []
        for b in S:
            key = (obj.id, b)
            d = st.delta[g.get_slot(b)]
            if key in src.snap:
                mv = float(np.abs(d - src.snap[key][0]).max())
            elif key in src.new_set:
                mv = float(np.abs(d).max())
            else:
                continue
            if mv >= lim:
                keep.append(b)
        S = keep
    if not S:
        return R
    Rs = set(R)
    add = set()
    rng = range(-n, n + 1)
    for b in S:
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    q = (b[0] + dx, b[1] + dy, b[2] + dz)
                    if q not in Rs:
                        add.add(q)
    if not add:
        return R
    A = np.array(sorted(add), np.int64)
    if g.ensure_index(A.min(0) - 1, A.max(0) + 1):
        scene.touch(index=True)
    for b in sorted(add):
        if g.get_slot(b) < 0:
            g.set_slot(b, st.alloc())
            rec.new.append((obj.id, b))
            rec.new_set.add((obj.id, b))
    scene.pack()
    scene.refresh_object(oi)
    return sorted(Rs | add)


REDIST_STATS = {}   # r68: why each re-distance ended (skip_L, unit_grad, no_gain, kept)


def redistance(scene, oid, bricks, rec, ramp=16.0, band=BAND, src=None, force=False):
    """Re-distance the sculpt field over `bricks` (brick coords of object oid) and the narrow band around the new
    surface (grow_band). Undo data goes into rec (a StrokeRecord). Kept only when it lowers the region's measured
    L; otherwise reverted. -> (L before, L after)."""
    oi = scene.index_of(oid)
    obj = scene.objects[oi]
    g = obj.sculpt
    st = scene.store
    h = g.h
    R = sorted(b for b in set(bricks) if g.get_slot(b) >= 0)
    if not R:
        return None
    if band > 0:
        R = grow_band(scene, oid, R, rec, band, src)
    # the region's L before (bricks R and their neighbours, whose apron reads R's nodes)
    near = sorted({(b[0] + dx, b[1] + dy, b[2] + dz) for b in R for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for dz in (-1, 0, 1)})
    near = [b for b in near if g.get_slot(b) >= 0]
    L_before = float(max(st.Lb[g.get_slot(b)] for b in near))
    redistance.last_grad = (0.0, 0.0, False)
    if L_before <= SKIP_L and not force:
        REDIST_STATS['skip_L'] = REDIST_STATS.get('skip_L', 0) + 1
        # hand2b: the premise stated as measured: the tracked bound here is the DELTA's (st.Lb), and after a
        # re-distance the delta is D - F_base, whose gradient is about the angle between the old and the new surface
        # normals, so a delta bound already <= 1.3 is not lowered: re-distance ran and was reverted for no gain on
        # 6 of 12 bench strokes (hand2) and on single Inflate dabs of 80 / 120 mm on a 1 mm bust (L_delta 0.64 /
        # 0.85; 2.5 / 7.9 s each, M 2026-11-08). The tracer's L of the region is L_base + this (not |grad F|).
        return L_before, L_before
    Ra = np.array(R, np.int64)
    blo = Ra.min(0) - 1
    bhi = Ra.max(0) + 1
    nb = tuple(int(v) for v in (bhi - blo + 1))
    role = np.zeros(nb, np.uint8)
    role[tuple((Ra - blo).T)] = 2
    halo = np.zeros_like(role)
    for d in _NB26:
        halo = np.maximum(halo, np.roll(role, d, (0, 1, 2)))
    role = np.where(role == 2, 2, np.where(halo > 0, 1, 0)).astype(np.uint8)
    full = np.repeat(np.repeat(np.repeat(role, 8, 0), 8, 1), 8, 2)
    reg = np.ascontiguousarray((full == 2).astype(np.uint8))
    act = np.ascontiguousarray((full > 0).astype(np.uint8))
    i0, j0, k0 = (int(v) for v in blo * 8)
    pk = scene.pack()
    F = np.empty(act.shape, np.float64)
    node_field(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, i0, j0, k0, h, act, F)
    ok_g, gp1, gp99 = _near_unit_gradient(F, reg, h)
    redistance.last_grad = (gp1, gp99, ok_g)
    if ok_g:
        REDIST_STATS['unit_grad'] = REDIST_STATS.get('unit_grad', 0) + 1
        # hand2b: nothing to gain: |grad F| is already near 1 where the tracer meets the surface (SKIP_G_*)
        return L_before, L_before
    U = np.empty_like(F)
    sweep_distance(F, act, h, U)
    beta = np.empty_like(F)
    edge_ramp(reg, float(ramp), beta)
    snaps = {}
    for b in R:
        s = g.get_slot(b)
        snaps[b] = (st.delta[s].copy(), st.mask[s].copy(), float(st.Lb[s]))
    Lold = {b: float(st.Lb[g.get_slot(b)]) for b in near}
    for b in R:
        s = g.get_slot(b)
        a = (np.array(b) - blo) * 8
        sl = (slice(a[0], a[0] + 8), slice(a[1], a[1] + 8), slice(a[2], a[2] + 8))
        u = U[sl]
        f = F[sl]
        ok = u < 1e29
        if not ok.any():
            continue
        wgt = beta[sl] * (1.0 - np.clip(st.mask[s].astype(np.float64), 0.0, 1.0)) * ok
        D = np.where(f < 0.0, -u, u)
        new = st.delta[s].astype(np.float64) + wgt * np.where(ok, D - f, 0.0)
        new = np.where(np.isfinite(new), new, st.delta[s].astype(np.float64))   # 11-09 guard: never a non-finite node
        st.delta[s] = new.astype(np.float32)
    arr = np.array(near, np.int64)
    out = np.zeros(len(near))
    brick_lipschitz(pk.index_flat, pk.delta, pk.oi_tab, oi, h, arr, out)
    # V1.3: measure L where the tracer reads it (F >= 0). Under a carve the inside keeps the OLD distance below the
    # region (D - F_base ~ the trench depth there), a mismatch no local ramp can take out; the tracer never steps
    # from inside, so those cells no longer set L. The all-cells bound above gives the threshold that keeps it sound.
    Lbase = float(scene.objects[oi].compiled()[4])
    thr = (Lbase + float(out.max())) * math.sqrt(3.0) * h
    Fn = np.where(act > 0, F, 1e30)
    for b in R:
        a = (np.array(b) - blo) * 8
        sl = (slice(a[0], a[0] + 8), slice(a[1], a[1] + 8), slice(a[2], a[2] + 8))
        s = g.get_slot(b)
        Fn[sl] = np.where(act[sl] > 0, F[sl] + (st.delta[s].astype(np.float64) - snaps[b][0].astype(np.float64)),
                          1e30)
    brick_lipschitz_out(pk.index_flat, pk.delta, pk.oi_tab, oi, h, arr, np.ascontiguousarray(Fn), i0, j0, k0,
                        thr, out)
    L_after = float(out.max())
    if L_after >= L_before and not force:       # no gain (e.g. a steep base field): keep the stroke as it was
        REDIST_STATS['no_gain'] = REDIST_STATS.get('no_gain', 0) + 1
        for b, (d, m, L) in snaps.items():
            st.delta[g.get_slot(b)] = d
        return L_before, L_before
    REDIST_STATS['kept'] = REDIST_STATS.get('kept', 0) + 1
    for b, L in zip(near, out):
        st.Lb[g.get_slot(b)] = L
    key = obj.id
    for b in R:
        # 11-09 (iterated release re-distance): the FIRST snapshot is the undo state; a later pass recording into the
        # same rec keeps it (one pass per rec before, so this was the same)
        if (key, b) not in rec.snap and (key, b) not in rec.new_set:
            rec.snap[(key, b)] = snaps[b]
    for b in near:
        if b not in snaps and (key, b) not in rec.Lsnap and (key, b) not in rec.snap and (key, b) not in rec.new_set:
            rec.Lsnap[(key, b)] = Lold[b]
    scene.refresh_object(oi)
    return L_before, L_after
