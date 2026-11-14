"""Hair Studio colour (numpy only, Apache-2.0): the fibre's absorption from pigments, as the Principled Hair BSDF
(Chiang et al. 2016 as implemented in Cycles) uses it, plus a neutral ("ash") absorber.

Cycles (melanin parametrisation): concentration q = -ln(max(1 - m, 1e-4)) for melanin m; eumelanin = q (1 - redness),
pheomelanin = q redness; sigma_a = eumelanin EU + pheomelanin PHEO (+ the tint's sigma, ~1-2 % here). The absorption
parametrisation takes sigma_a directly, so an ash tone is possible: a share `ash` of the absorption made neutral
(grey) at the same luminance-weighted strength - same darkness, a flatter spectrum (less yellow).

Chiang's map between a fibre colour c and sigma_a at radial roughness b: sigma = (ln c / D(b))^2,
D(b) = 5.969 - 0.215 b + 2.532 b^2 - 10.73 b^3 + 5.574 b^4 + 0.245 b^5.

    sigma(m, redness, ash)            (..., 3) absorption per unit radius
    fibre_colour(sigma, b), sigma_from_colour(c, b)
    melanin_equivalent(sigma)         the melanin with the same luminance-weighted absorption (redness 0)
    predict_render(sigma, gain, white, b)   a fitted scene model: rendered linear RGB = gain * fibre colour + white
    srgb_lin_to_lab(rgb), lab_to_srgb_lin(L, a, b)

The colour stage (10-31: the hair set once per person, the style unable to change it, the colour changeable
without re-growing). Colour used to be drawn inline in generate from the pipeline's shared random streams, so it could
only change by growing again (~35 min at 145k). colourize derives every colour array from what the npz saves - the
final points, kinds, din, the lock / sub-lock / bundle ids - plus the scalp's frame, the style's colour and material
blocks (and the two hair keys colour reads: hairline.baby.lighter, locks.bundle_r_mm) and streams of its OWN seeded by
the colour seed. generate calls it on the arrays as it saves them and engine/hair/recolour.py calls it on a saved npz,
so the two give the same colours by construction, and a new colour block changes nothing but the colour arrays.

    colourize(pts, kind, din, lock, sub, bun, scalp, st, seed)
        -> dict(melp, mel, red, mel_root, sigma, material_json, report)
    lock_table(lock, sub, bun)        form.melanin_field's lock membership rebuilt from the saved ids
"""
import json

import numpy as np

EU = np.array([0.506, 0.841, 1.653])
PHEO = np.array([0.343, 0.733, 1.924])
LUM = np.array([0.2126, 0.7152, 0.0722])
GREY = float(EU @ LUM)      # 0.828: the neutral absorber of eumelanin's luminance-weighted strength
# a violet TONER (hair2 lane 11-10 night): the salon cause of beige blond - a pigment absorbing mid (green-yellow) light, at
# eumelanin's luminance-weighted strength. Melanin + neutral ash alone cannot give a warm, low-yellow beige (ash lowers the
# yellow and the warmth together); measured need: the TARGET's lit lengths R/G 1.55, G/B 1.79 (linear) against ours 1.47, 2.07
TONER = np.array([0.25, 1.0, 0.35]); TONER = TONER / float(TONER @ LUM) * GREY


def conc(m):
    return -np.log(np.maximum(1 - np.asarray(m, np.float64), 1e-4))


def sigma(m, redness=0.0, ash=0.0, toner=0.0):
    q = conc(m)[..., None]
    r = np.asarray(redness, np.float64)[..., None]
    a = np.asarray(ash, np.float64)[..., None]
    t = np.asarray(toner, np.float64)[..., None]
    pig = (1 - r) * EU + r * PHEO
    return q * ((1 - a - t) * pig + a * GREY + t * TONER)


def D(b):
    return ((((0.245 * b + 5.574) * b - 10.73) * b + 2.532) * b - 0.215) * b + 5.969


def fibre_colour(sig, b=0.55):
    return np.exp(-np.sqrt(np.maximum(sig, 0)) * D(b))


def sigma_from_colour(c, b=0.55):
    return (np.log(np.clip(c, 1e-6, 1.0)) / D(b)) ** 2


def melanin_equivalent(sig):
    s = np.asarray(sig, np.float64) @ LUM
    return 1 - np.exp(-s / GREY)


def predict_render(sig, gain, white, b=0.55):
    return np.asarray(gain) * fibre_colour(sig, b) + np.asarray(white)


def srgb_lin_to_lab(rgb):
    rgb = np.asarray(rgb, np.float64)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = rgb @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb_lin(L, a, b):
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    finv = lambda f: np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787)  # noqa: E731
    X, Y, Z = 0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    return np.stack([X, Y, Z], -1) @ M.T


def solve_pigments(sig_target):
    """the (q_eu, q_pheo, q_grey) >= 0 whose absorption is nearest sig_target (3,) (least squares, non-negative):
    -> melanin m, redness, ash, residual"""
    from scipy.optimize import nnls
    A = np.stack([EU, PHEO, np.full(3, GREY)], 1)
    x, res = nnls(A, np.asarray(sig_target, np.float64))
    q = x.sum()
    if q <= 0:
        return 0.0, 0.0, 0.0, float(res)
    # sigma(m, r, a) = q ((1 - a)((1 - r) EU + r PHEO) + a GREY): q (1 - a)(1 - r) = x0, q (1 - a) r = x1, q a = x2
    a = x[2] / q
    r = x[1] / max(x[0] + x[1], 1e-12)
    m = 1 - np.exp(-q)
    return float(m), float(r), float(a), float(res)


# ---------------------------------------------------------------------------------------------------- the colour stage
# its own streams under the colour seed (numpy SeedSequence entropy [seed, tag]): the main strands' lock / sub-lock /
# strand / balayage / redness draws, the bundles' draws, the baby hairs' draws. None is shared with the geometry, so
# drawing the colour again never moves a point, and a baby-hair count change never shifts the main strands' colour.
STREAM_MAIN, STREAM_BUNDLE, STREAM_BABY = 0xC0, 0xC1, 0xC2


def lock_table(lock, sub, bun=None):
    """form.melanin_field's membership (li, si, bun, CL, nl, nsub) rebuilt from the saved ids of the MAIN strands (no
    baby hairs): li = lock, si = sub, bun = each strand's bundle id (a strand index: its bundle's centre strand), CL =
    each lock's first member (the strand whose root reads the balayage's framing zone), nl / nsub one past the largest
    ids. A negative id (none written today) becomes a lock / sub-lock of its own; bun None (an npz from before the
    colour stage, 10-31) or an id outside the strands gives each strand its own bundle - the bundle offset then acts as
    one more per-strand offset."""
    li = np.array(lock, np.int64)
    si = np.array(sub, np.int64)
    n = len(li)
    for a in (li, si):
        neg = a < 0
        if neg.any():
            a[neg] = (a[~neg].max() + 1 if (~neg).any() else 0) + np.arange(int(neg.sum()))
    own = np.arange(n)
    bu = own.copy() if bun is None else np.array(bun, np.int64)
    bad = (bu < 0) | (bu >= n)
    bu[bad] = own[bad]
    nl = int(li.max()) + 1 if n else 0
    nsub = int(si.max()) + 1 if n else 0
    CL = np.zeros(nl, np.int64)                       # a lock id no strand carries (a gap) reads strand 0: never used
    u, first = np.unique(li, return_index=True)
    CL[u] = first
    return dict(li=li, si=si, bun=bu, CL=CL, nl=nl, nsub=nsub,
                stats={"locks": nl, "sub_locks": nsub, "bundle_ids": "strand-level (bun absent)" if bun is None
                       else int(len(np.unique(bu))), "bundle_ids_replaced": int(bad.sum()) if bun is not None else 0})


def colourize(pts, kind, din, lock, sub, bun, scalp, st, seed):
    """the colour stage: every colour array of the npz from the saved arrays alone.

    pts (n, K, 3) the final points (as saved, f32); kind (n,) i8, the baby hairs (kind 3) last, as generate writes
    them; din (n,) the roots' mm inside the hairline (as saved, f32); lock, sub (n,) the saved ids (-1 on baby hairs);
    bun (n,) the bundle id per strand (-1 on baby hairs) or None for an older npz; scalp: anything with .sph,
    .d_inside_mm and .ears ({side: {"z_top": z}}) - the case's Scalp, or recolour's from the npz's centre / grid / thb /
    ear_z_top; st: the EFFECTIVE style (style.effective: the oil's roughness in the material); seed: the colour seed.

    The main strands go through form.melanin_field (base + lock + sub-lock + bundle + strand offsets, roots darker,
    lengths lighter, balayage locks, the front edge and the edge hairs lighter; redness per strand), phi_r of each lock
    read at its first member's root (pts[:, 0]); the baby hairs take the base less front_lighter and
    hairline.baby.lighter, plus a strand offset, all along the strand (generate's formula since HS2).
    -> dict(melp (n, K) f32, mel (n,) f32, red (n,) f32, mel_root (n,) f32 zeros, sigma (n, K, 3) f16, material_json,
    report)"""
    from . import strand_util as FO                   # melanin_field (strand_util.py since P14, 11-09; was form.py)
    P = np.asarray(pts)
    kind = np.asarray(kind)
    n, K = P.shape[:2]
    nm = int((kind != 3).sum())
    if (kind[nm:] != 3).any():
        raise ValueError("colourize: the baby hairs (kind 3) must come after every other strand, as generate writes them")
    seed = int(seed)
    Cc = st["colour"]
    melp = np.empty((n, K), np.float32)
    red = np.empty(n, np.float32)
    rep = {"seed": seed, "streams": [[seed, STREAM_MAIN], [seed, STREAM_BUNDLE], [seed, STREAM_BABY]],
           "strands": int(n), "baby_hairs": int(n - nm)}
    if nm:
        lk = lock_table(np.asarray(lock)[:nm], np.asarray(sub)[:nm], None if bun is None else np.asarray(bun)[:nm])
        Pm = P[:nm].astype(np.float64)
        phi_r, _, _ = scalp.sph(Pm[:, 0])
        melp[:nm], red[:nm], cst = FO.melanin_field(
            Pm, Pm[:, 0], kind[:nm], np.asarray(din, np.float64)[:nm], phi_r, scalp, st, lk,
            np.random.default_rng([seed, STREAM_MAIN]), np.random.default_rng([seed, STREAM_BUNDLE]))
        del Pm
        rep.update(cst)
        rep["lock_table"] = lk["stats"]
    if n > nm:
        m = n - nm
        Bb = st["hairline"]["baby"]
        rb = np.random.default_rng([seed, STREAM_BABY])
        mb = np.clip(Cc["melanin"] - Cc["front_lighter"] - Bb["lighter"] + rb.normal(0, Cc["strand_sd"], m), 0.1, 0.95)
        melp[nm:] = mb[:, None]
        red[nm:] = np.clip(Cc["redness"] + rb.normal(0, Cc["redness_sd"], m), 0, 0.9)
    return dict(melp=melp, mel=melp.mean(1).astype(np.float32), red=red, mel_root=np.zeros(n, np.float32),
                sigma=sigma(melp, red[:, None], Cc["ash"]).astype(np.float16),
                material_json=json.dumps(st["material"]), report=rep)
