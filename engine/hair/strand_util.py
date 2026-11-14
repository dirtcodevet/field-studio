"""Hair strand utilities (numpy only, Apache-2.0): smoothstep, the arc length along strands, the oil model (sebum +
product -> load -> liquid-bridge adhesion) and the per-point melanin field, moved out of form.py / scalp.py (the old
triangle-scalp Hair Studio chain) so the live fibre and colour modules need not import it (P14 mesh-free, 11-09).

Verbatim copies: smoothstep from scalp.py (sha256 cbfda4305345), arc / melanin_field / OIL_DEFAULTS / oil_load /
adhesion_per_length from form.py (sha256 fb266bc55186). melanin_field reads its `scalp` argument only through .sph(P),
.d_inside_mm(P) and .ears[side]["z_top"]: any head frame with those (closed-form, field- or state-based) serves; no
triangle surface is needed.
"""
import numpy as np

FORM_PY_SHA256_12 = "fb266bc55186"
SCALP_PY_SHA256_12 = "cbfda4305345"


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def arc(P):
    return np.concatenate([np.zeros(P.shape[:-2] + (1,)), np.cumsum(np.linalg.norm(np.diff(P, axis=-2), axis=-1), -1)],
                          -1)


def melanin_field(P, roots, kind, din, phi_r, scalp, st, lk, rng, rng_b, s_tie=None, batch=12000):
    """melanin per point (n, K): base + lock + sub-lock + bundle + strand offsets; roots darker; the lengths lighter
    below the temple level; balayage locks (more of the face-framing ones) lighter below the upper ear; the front edge
    zone and the edge hairs lighter. Redness per strand."""
    Cc = st["colour"]
    n, K = P.shape[:2]
    li, si, bun, CL = lk["li"], lk["si"], lk["bun"], lk["CL"]
    nl, nsub = lk["nl"], lk["nsub"]
    ears = scalp.ears
    z_et = 0.5 * sum(e["z_top"] for e in ears.values())
    lock_off = rng.normal(0, 1.0, nl) * Cc["lock_sd"]
    sub_off = rng.normal(0, 1.0, nsub) * Cc["sub_sd"]
    str_off = rng.normal(0, Cc["strand_sd"], n)
    bun_off = rng_b.normal(0, Cc["bundle_sd"], n)
    phl = phi_r[CL]
    dinl = din[CL]
    framing = (dinl < 30) & (np.abs(phl) > 15) & (np.abs(phl) < 80)
    bal = np.where(rng.random(nl) < np.where(framing, Cc["balayage_framing_frac"], Cc["balayage_frac"]),
                   rng.uniform(*Cc["balayage_amount"], nl), 0.0)
    zb = np.where(framing, z_et + 0.07, z_et + 0.03)
    melp = np.empty((n, K), np.float32)
    for b0 in range(0, n, batch):
        sl = slice(b0, min(n, b0 + batch))
        Pb = P[sl]
        s = arc(Pb)
        l = li[sl]
        z = Pb[..., 2]
        m = Cc["melanin"] + lock_off[l][:, None] + sub_off[si[sl]][:, None] + str_off[sl][:, None]
        if st["locks"]["bundle_r_mm"] > 0:
            m = m + bun_off[bun[sl]][:, None]
        php, _, _ = scalp.sph(Pb)
        dinp = scalp.d_inside_mm(Pb.reshape(-1, 3)).reshape(Pb.shape[:2])
        m = m + Cc["root_add"] * np.exp(-s / max(Cc["root_mm"] / 1000, 1e-6)) * (din[sl] > 15)[:, None]
        m = m - Cc["lengths_lighter"] * (1 - smoothstep(z_et - 0.04, z_et + 0.03, z))
        m = m - bal[l][:, None] * (1 - smoothstep(zb[l][:, None] - 0.03, zb[l][:, None] + 0.01, z)) * \
            smoothstep(8.0, 20.0, dinp)
        m = m - Cc["front_lighter"] * (1 - smoothstep(3.0, 14.0, dinp)) * (1 - smoothstep(35.0, 50.0, np.abs(php)))
        m = m - Cc["edge_lighter"] * (kind[sl] > 0)[:, None]
        melp[sl] = np.clip(m, 0.10, 0.95)
    red = np.clip(Cc["redness"] + rng.normal(0, Cc["redness_sd"], n), 0.0, 0.9).astype(np.float32)
    stats = {"balayage_locks": int((bal > 0).sum()), "framing_locks": int(framing.sum()),
             "melanin_point_p5_50_95": np.percentile(melp, [5, 50, 95]).round(3).tolist()}
    return melp, red, stats


OIL_DEFAULTS = {
    "days_since_wash": (1.0, (0.0, 7.0), "days since the hair was washed; sebum spreads down from the scalp over them "
                                         "(second-day hair grips, 2-8x withdrawal force: hair_styling P58)"),
    "product": (0.0, (0.0, 1.0), "the style's oil / serum / leave-in load along the whole length, 0 none .. 1 heavy "
                                 "(hair_styling 7 row 22: liquid bridges -> ribbons; overdone -> rat tails)"),
    "sebum_mm_per_day": (50.0, (10.0, 200.0), "how far sebum creeps down the hair per day (estimate, not measured: "
                                              "roots greasy on day 2, the lengths by day 3-5 on fine hair; fine hair "
                                              "spreads it faster, hair_styling 7 row 'Fine vs coarse')"),
    "sebum_load": (0.6, (0.0, 2.0), "the sebum load on the hair near the scalp once spread, on the product's scale"),
    "gamma_mN_m": (30.0, (20.0, 72.0), "surface tension of the bridging liquid: sebum / oils ~30 mN/m, water 72"),
    "load_half": (0.5, (0.05, 5.0), "the load at which the bridges wet half the fibre's perimeter; W = gamma pi r "
                                    "o / (o + load_half)"),
    "capture_mm": (1.0, (0.1, 5.0), "groups closer than this (surface to surface, packed) meet as the hair is handled "
                                    "and drying; bridges cannot form across a wider gap"),
}


def oil_load(s, days, product, sebum_mm_per_day, sebum_load):
    """the oil load (product's scale) at arc s (m): the product everywhere, plus sebum from the root down to the front it
    has crept to in `days` (smooth over the last third of that front)"""
    front = max(days * sebum_mm_per_day / 1000.0, 1e-9)
    seb = sebum_load * (1.0 - smoothstep(0.66 * front, front, s)) if days > 0 else 0.0 * s
    return product + seb


def adhesion_per_length(o, r, gamma, load_half):
    """adhesion energy per length of fibre-fibre contact (J/m) of a liquid bridge: gamma x the wetted width, which grows
    with the load toward the fibre's half perimeter pi r"""
    o = np.maximum(o, 0.0)
    return gamma * np.pi * r * o / (o + load_half)
