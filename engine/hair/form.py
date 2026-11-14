"""Hair Studio form (numpy + scipy, Apache-2.0): what turns combed strands into hair with form and texture - locks,
sub-locks and bundles, clumping, flat S-waves, crown lift, root jitter, wander, frizz, fly-aways - and the colour
field per point. Generalised from the the test model HS2 step 2 (its measured choices kept as the defaults; see the style
schema), batch-wise, no N x N matrices.

Order (form): locks -> per-lock parameters -> waves + root jitter + crown lift -> lock / sub-lock / bundle clumping
(each toward its centre strand's curve at the same arc length, shifted by the roots' along-flow offset) -> wander,
frizz, fly-aways. Gathered strands (s_tie > 0): before the tie they are sleek (clumping, waves, frizz x (1 - sleek)),
the waves start after the tie.
"""
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .scalp import smoothstep


def arc(P):
    return np.concatenate([np.zeros(P.shape[:-2] + (1,)), np.cumsum(np.linalg.norm(np.diff(P, axis=-2), axis=-1), -1)],
                          -1)


def at_arc(C, LC, s, K):
    m = len(C)
    f = np.clip(s / np.maximum(LC, 1e-9)[:, None] * (K - 1), 0, K - 1)
    i0 = np.minimum(np.floor(f).astype(np.int64), K - 2)
    w = (f - i0)[..., None]
    r = np.arange(m)[:, None]
    out = C[r, i0] * (1 - w) + C[r, i0 + 1] * w
    d = C[:, -1] - C[:, -2]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    out += np.maximum(s - LC[:, None], 0)[..., None] * d[:, None, :]
    return out


def poisson(key, cand, r, rng):
    """Poisson-disk subset of cand (indices into key, mm) with minimum distance r (scalar or per candidate)"""
    if len(cand) == 0:
        return np.zeros(0, np.int64)
    tree = cKDTree(key[cand])
    rr = np.broadcast_to(np.asarray(r, np.float64), (len(cand),))
    blocked = np.zeros(len(cand), bool)
    out = []
    for j in rng.permutation(len(cand)):
        if blocked[j]:
            continue
        out.append(cand[j])
        blocked[tree.query_ball_point(key[cand[j]], rr[j])] = True
    return np.array(out, np.int64)


def frames(P, c):
    """tangent T, the in-sheet side B (T x the outward direction), the sheet normal N2, the outward direction Nout:
    radial from the head centre c, its downward part dropped (so on the top of the head the sheet is the scalp's
    tangent plane, and on hanging hair it is the outward horizontal; HS2 used the horizontal everywhere, which turned
    the waves and the root jitter of the strands crossing the top into the scalp's normal: spikes)"""
    T = np.gradient(P, axis=1)
    T /= np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-12)
    Nout = P - np.asarray(c, np.float64)
    Nout[..., 2] = np.maximum(Nout[..., 2], 0)
    Nout /= np.maximum(np.linalg.norm(Nout, axis=2, keepdims=True), 1e-12)
    B = np.cross(T, Nout)
    nb = np.linalg.norm(B, axis=2, keepdims=True)
    alt = np.cross(T, np.array([0.0, 0.0, 1.0]))
    B = np.where(nb > 0.2, B, alt)
    B /= np.maximum(np.linalg.norm(B, axis=2, keepdims=True), 1e-12)
    N2 = np.cross(B, T)
    return T, B, N2, Nout


def locks(X, gside, main, din, st, rng, rng_b):
    """lock / sub-lock / bundle membership: Poisson-disk centres among the main strands' roots (the part's sides kept
    apart), Voronoi cells. Returns dict(lock, li, nl, sub, si, nsub, bun, CL, SC, BC, stats)"""
    L = st["locks"]
    n = len(X)
    # the part's two sides are kept apart only in the part zone (behind it the hair is one sheet: no line down the back)
    pz = st.get("_part_zone")
    key = np.c_[X * 1000.0, gside * 60.0 * (pz if pz is not None else 1.0)]
    cand = np.nonzero(main)[0]
    r_lock = L["lock_r_mm"] * np.where(din[cand] < L["front_zone_mm"], L["front_scale"], 1.0)
    CL = poisson(key, cand, r_lock, rng)
    lock = CL[cKDTree(key[CL]).query(key)[1]]
    r_sub = L["sub_r_mm"] * np.where(din[cand] < L["front_zone_mm"], min(1.0, L["front_scale"] * 1.15), 1.0)
    SC = poisson(key, cand, r_sub, rng)
    sc_lock = lock[SC]
    _, ii = cKDTree(key[SC]).query(key, k=8)
    match = sc_lock[ii] == lock[:, None]
    has = match.any(1)
    sub = np.where(has, SC[ii[np.arange(n), np.argmax(match, 1)]], lock)
    del ii, match
    if L["bundle_r_mm"] > 0:
        BC = poisson(key, cand, L["bundle_r_mm"], rng_b)
        _, ii = cKDTree(key[BC]).query(key, k=8)
        match = sub[BC][ii] == sub[:, None]
        hasb = match.any(1)
        bun = np.where(hasb, BC[ii[np.arange(n), np.argmax(match, 1)]], np.arange(n))
        del ii, match
    else:
        BC, bun = np.zeros(0, np.int64), np.arange(n)
    nl = len(CL)
    lid = -np.ones(n, np.int64)
    lid[CL] = np.arange(nl)
    li = lid[lock]
    sid = -np.ones(n, np.int64)
    sid[SC] = np.arange(len(SC))
    si = np.where(sid[sub] >= 0, sid[sub], len(SC))
    cell = np.bincount(li, minlength=nl)
    stats = {"locks": int(nl), "sub_locks": int(len(SC)), "bundles": int(len(BC)),
             "strands_per_lock_p10_50_90": np.percentile(cell, [10, 50, 90]).tolist()}
    if len(BC):
        stats["strands_per_bundle_median"] = float(np.median(np.bincount(bun)[BC]))
    return dict(lock=lock, li=li, nl=nl, sub=sub, si=si, nsub=len(SC) + 1, bun=bun, CL=CL, SC=SC, BC=BC, stats=stats)


def shape(P, roots, kind, din, phi_r, th_r, side_x, s_tie, scalp, st, lk, rng, rng_b, log=print, batch=12000):
    """the form on strands P (n, K, 3) float64 (in place): waves, root jitter, crown lift, clumping, wander / frizz /
    fly-aways. s_tie (n,): arc of the tie (0 = loose). Returns P, fly mask, stats"""
    n, K = P.shape[:2]
    main = kind == 0
    li, si, bun = lk["li"], lk["si"], lk["bun"]
    nl, nsub = lk["nl"], lk["nsub"]
    W, C, Fz, Lk, G = st["waves"], st["clump"], st["frizz"], st["locks"], st["gather"]
    sleek = float(G["sleek"]) if G["type"] != "none" else 0.0
    ears = scalp.ears
    z_eb = min(e["z_bot"] for e in ears.values())
    y_back = max(e["y_back"] for e in ears.values())
    stats = {}
    # per lock / sub-lock / strand parameters
    k_lock = rng.uniform(*C["lock"], nl)
    phase = rng.uniform(0, 2 * np.pi, nl)
    pf = float(W.get("phase_field_mm", 0.0))
    if pf > 0:
        # coherent waves (10-29 night): neighbouring locks wave together, as a set wave does (each lock at its own random
        # phase reads as crimp): the phase a smooth field over the lock's root (random Fourier features at ~pf mm) plus a
        # per-lock jitter
        rp = np.random.default_rng(int(st["seed"]) + 53)
        kv = rp.normal(0, 1, (24, 3)) / (pf / 1000)
        bv = rp.uniform(0, 2 * np.pi, 24)
        fld = np.cos(roots[lk["CL"]] @ kv.T + bv).sum(1) / np.sqrt(12)
        phase = np.pi * fld + rp.normal(0, float(W.get("phase_jitter", 0.6)), nl)
    lam = W["wavelength_mm"] / 1000 * rng.uniform(1 - W["wavelength_jitter"], 1 + W["wavelength_jitter"], nl)
    amp = rng.uniform(*W["amp_mm"], nl) / 1000.0
    s0 = rng.uniform(*W["start_mm"], nl) / 1000
    hl = rng.uniform(0.8, 1.2, nl)
    k_sub = rng.uniform(*C["sub"], nsub)
    k_sub[-1] = 0
    ph_sub = rng.normal(0, 0.25, nsub)
    ph_str = rng.normal(0, 0.08, n)
    amp_str = rng.uniform(0.9, 1.1, n)
    tilt = np.radians(rng.uniform(*W["tilt_deg"], nl)) * rng.choice([-1.0, 1.0], nl)
    jit = np.radians(rng.normal(0, Lk["root_jitter_deg"], n))
    if Lk["bundle_r_mm"] > 0:
        bjit = np.radians(rng_b.normal(0, Lk["bundle_jitter_deg"], n))
        a0, a1 = Lk["bundle_jitter_phi_deg"]
        bjit_w = 1 - smoothstep(a0, a1, np.abs(phi_r)) if a1 > a0 else np.ones(n)
        jit = jit + bjit[bun] * bjit_w
        k_bun = rng_b.uniform(*C["bundle"], n)
        stats["bundle_jitter_kept_mean"] = round(float(bjit_w.mean()), 3)
    fly = main & (rng.random(n) < Fz["fly_frac"])
    stats["fly_aways"] = int(fly.sum())
    wave_on = W["wavelength_mm"] > 0 and W["amp_mm"][1] > 0
    for b0 in range(0, n, batch):
        sl = slice(b0, min(n, b0 + batch))
        Pb = P[sl]
        s = arc(Pb)
        T, B, N2, Nout = frames(Pb, scalp.c)
        l = li[sl]
        st0 = s_tie[sl][:, None]
        disp = np.zeros_like(Pb)
        if wave_on:
            near = np.zeros_like(s)
            if W["ears_free"]:
                near = smoothstep(0.050, 0.065, np.abs(Pb[..., 0])) * smoothstep(z_eb - 0.04, z_eb - 0.01, Pb[..., 2]) * \
                    (1 - smoothstep(y_back + 0.015, y_back + 0.04, Pb[..., 1]))
            sw = st0 + s0[l][:, None]
            A = amp[l][:, None] * amp_str[sl][:, None] * smoothstep(sw, sw + 0.5 * lam[l][:, None], s) \
                * (0.85 + 0.3 * smoothstep(sw, sw + 0.25, s)) * (1 - near)
            argw = 2 * np.pi * s / lam[l][:, None] + phase[l][:, None] + ph_sub[si[sl]][:, None] + ph_str[sl][:, None]
            ca, sa = np.cos(tilt[l])[:, None, None], np.sin(tilt[l])[:, None, None]
            Wd, Wp = ca * B + sa * N2, -sa * B + ca * N2
            disp += A[..., None] * (np.sin(argw)[..., None] * Wd + W["perp"] * np.cos(argw)[..., None] * Wp)
        # root jitter: the strand turns jit about its root normal over its first 30 mm
        disp += (jit[sl][:, None] * np.minimum(s, 0.03) * (1 - sleek * (s_tie[sl] > 0))[:, None])[..., None] * B * \
            main[sl][:, None, None]
        # crown lift: radial over the top, not at the front hairline
        if st["lift"]["crown_mm"] > 0:
            v = Pb - scalp.c
            u = v / np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-9)
            php, thp, _ = scalp.sph(Pb)
            dinp = scalp.d_inside_mm(Pb.reshape(-1, 3)).reshape(Pb.shape[:2])
            wf = 1 - (1 - smoothstep(12.0, 45.0, dinp)) * (1 - smoothstep(70.0, 90.0, np.abs(php)))
            lift = st["lift"]["crown_mm"] / 1000.0 * hl[l][:, None] * (1 - smoothstep(25.0, 55.0, thp)) * wf * \
                (1 - np.exp(-s / 0.02))
            disp += (lift * main[sl][:, None])[..., None] * u
        P[sl] = Pb + disp
    log("  form: waves + jitter + crown lift")
    # clumping
    keep_edge = np.ones(n)
    if C["temple_free"]:
        keep_edge *= 1 - (1 - smoothstep(3.0, 20.0, din)) * smoothstep(30.0, 45.0, np.abs(phi_r))
    if C["part_free"] and st["part"]["type"] != "none":
        part_zone = (din < 80.0) & (th_r < 70.0)
        keep_edge *= np.where(part_zone, 0.25 + 0.75 * smoothstep(0.003, 0.012, np.abs(side_x)), 1.0)
    stats["clump_kept_lt_0p5"] = int((keep_edge < 0.5).sum())
    Ltot = arc(P)[:, -1]
    sfly = np.where(fly, rng.uniform(*Fz["fly_start"], n) * Ltot, 1e9)
    rise_lock = np.where(din < Lk["front_zone_mm"], C["front_rise_mm"], C["lock_rise_mm"]) / 1000
    tip_gain = float(C.get("tip_gain", 0.0))

    def clump(Psrc, centre_idx, kfac, lam_c):
        out = np.empty_like(Psrc)
        lam_c = np.broadcast_to(np.asarray(lam_c, np.float64), (n,))
        for b0 in range(0, n, batch):
            sl = slice(b0, min(n, b0 + batch))
            Pb = Psrc[sl]
            ci = centre_idx[sl]
            Cc = Psrc[ci]
            s = arc(Pb)
            LC = arc(Cc)[:, -1]
            t0c = Cc[:, 1] - Cc[:, 0]
            t0c /= np.maximum(np.linalg.norm(t0c, axis=1, keepdims=True), 1e-12)
            delta = np.einsum("ij,ij->i", Pb[:, 0] - Cc[:, 0], t0c)
            sc = np.maximum(s + delta[:, None], 0)
            Cs = at_arc(Cc, LC, sc, K)
            c = kfac[sl][:, None] * (1 - np.exp(-s / lam_c[sl][:, None]))
            if tip_gain > 0:          # real locks gather toward the ends (ribbons), loosest at the root
                c = np.minimum(c * (1 + tip_gain * s / np.maximum(s[:, -1:], 1e-6)), 0.95)
            c *= np.clip(1 - (sc - LC[:, None]) / 0.03, 0, 1)
            c *= np.where(main[sl], 1.0, C["edge_factor"])[:, None]
            c *= keep_edge[sl][:, None]
            c *= 1 - smoothstep(sfly[sl][:, None], sfly[sl][:, None] + 0.02, s)
            if sleek > 0:
                c *= 1 - sleek * (s < s_tie[sl][:, None])
            out[sl] = Pb + c[..., None] * (Cs - Pb)
        return out
    P[:] = clump(P, lk["lock"], k_lock[li], rise_lock)
    P[:] = clump(P, lk["sub"], k_sub[si], C["sub_rise_mm"] / 1000)
    if Lk["bundle_r_mm"] > 0:
        P[:] = clump(P, bun, np.where(bun == np.arange(n), 0.0, k_bun[bun]), C["bundle_rise_mm"] / 1000)
    log("  form: clumping")
    # wander, frizz, fly-aways
    sw = rng.uniform(*Fz["wander_mm"], n) / 1000.0
    sf = rng.uniform(*Fz["frizz_mm"], n) / 1000.0
    afly = rng.uniform(*Fz["fly_mm"], n) / 1000
    for b0 in range(0, n, batch):
        sl = slice(b0, min(n, b0 + batch))
        Pb = P[sl]
        nb = len(Pb)
        s = arc(Pb)
        T, B, N2, Nout = frames(Pb, scalp.c)
        ramp = 1 - np.exp(-s / 0.02)
        if sleek > 0:
            ramp = ramp * (1 - sleek * (s < s_tie[sl][:, None]))
        wn = gaussian_filter1d(rng.normal(0, 1, (nb, K, 2)), 3.0, axis=1)
        wn /= np.maximum(wn.std(axis=1, keepdims=True), 1e-9)
        fz = rng.normal(0, 1, (nb, K, 2))
        fz[:, 0] = 0
        off = sw[sl][:, None, None] * wn * ramp[..., None] + sf[sl][:, None, None] * fz * \
            (np.minimum(1, s / 0.005) * (ramp if sleek > 0 else 1))[..., None]
        disp = off[..., :1] * B + off[..., 1:] * N2
        fl = fly[sl]
        if fl.any():
            Lb = s[:, -1]
            g = smoothstep(sfly[sl][:, None], np.maximum(Lb, sfly[sl] + 0.01)[:, None], s) ** 1.5
            if W["ears_free"]:
                g = g * (1 - smoothstep(0.050, 0.065, np.abs(Pb[..., 0])) * smoothstep(z_eb - 0.04, z_eb - 0.01, Pb[..., 2])
                         * (1 - smoothstep(y_back + 0.015, y_back + 0.04, Pb[..., 1])))
            dn = gaussian_filter1d(rng.normal(0, 1, (nb, K, 2)), 5.0, axis=1)
            dn /= np.maximum(np.linalg.norm(dn, axis=2, keepdims=True), 1e-9)
            dirv = 0.7 * Nout + 0.5 * (dn[..., :1] * B + dn[..., 1:] * T * 0.3)
            dirv /= np.maximum(np.linalg.norm(dirv, axis=2, keepdims=True), 1e-9)
            disp += np.where(fl[:, None, None], (afly[sl][:, None] * g)[..., None] * dirv, 0.0)
        P[sl] = Pb + disp
    log("  form: wander / frizz / fly-aways")
    return P, fly, stats


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


# ----------------------------------------------------------------------------------------------------------------------
# Locks from causes (grow mode; DESIGN_v2 4.9, hair_styling.md points 3, 8, 10, 22, 32, hair.md point 2).
#
# The legacy `shape` above pulls strands toward lock centres it picked (a stand-in). Here a lock is what the fibres do:
#   1. follicular units (children.follicular_units): 1-4 hairs leave one follicle opening together - the seed of a clump;
#   2. the drying section: hairs dried in one section took one wave phase (hair_styling 7 rows 19, 32: "in phase only
#      inside a clump formed while wet"), so their waves nest; across sections the phases differ;
#   3. adhesion: sebum (spreading down from the scalp over the days since washing) and the style's product make liquid
#      bridges between touching fibres, an adhesion energy per length of contact W (J/m). Two neighbouring groups merge
#      when W, over the length they can zip, pays for bending them together against their own stiffness EI (elastocapillary
#      coalescence, Bico, Roman, Moulin & Boudaoud, Nature 432:690, 2004) and for forcing their waves into step;
#   4. inside a merged group the bridges' cohesion squeezes the fibres against the hair mass's own swelling pressure
#      P = k E phi^3 (Van Wyk; Goldstein, Warren & Ball 2012): the group packs to phi_lock = sqrt(W / (pi r^2 k E)),
#      tighter with more oil (ribbons; overdone: rat tails), not at all when phi_lock is below the mass's own packing.
# No width, count or centre is chosen: oil 0 gives W = 0 and no locks; fine hair (EI ~ r^4) merges more than thick.
# ----------------------------------------------------------------------------------------------------------------------

# oil.* and locks.* keys read here, in style.py's (default, (lo, hi), note) format
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
LOCK_CAUSE_DEFAULTS = {
    "section_mm": (35.0, (10.0, 80.0), "the drying section's width on the scalp (dry.section_mm when the acts carry it; "
                                       "hair_styling 7 row 32: no wider than the barrel, 25-50 mm)"),
    "phase_field_mm": (0.0, (0.0, 60.0), "> 0: the wave's phase from a coherent field over the roots at about this "
                                          "scale (11-12, n2d: an air-dried natural wave is in step only inside a clump "
                                          "formed while wet, so neighbouring locks loosely together and the head's patches "
                                          "apart) instead of one phase per drying section (a barrel's width, 35 mm, read "
                                          "as even horizontal bands); the sections still set the wavelength, plane, grouping"),
    "phase_jitter": (0.15, (0.0, 3.0), "rad, each hair's own phase about its section's (a natural wave is not "
                                       "perfectly in step even inside one section)"),
    "rounds": (60, (1, 200), "coalescence rounds, a cap: it stops when no pair merges (each round merges mutual best "
                            "partners: Bico's hierarchical coalescence). 12 cut it short on gr3 (153 merges in round 12)"),
    "phi_max": (0.5, (0.1, 0.8), "the tightest a lock can pack (fibres touching, hexagonal is 0.9)"),
    "brush_lane_mm": (6.0, (1.0, 20.0), "with brush.through (the stroke drawn to the ends over the dried hair): the "
                                        "clear gap between the brush's neighbouring pins / tufts across the stroke. "
                                        "Every hair in one lane is gathered against its lane-mates whatever section "
                                        "it was dried in, so groups whose facing edges lie within a lane meet. A "
                                        "cushion brush's ball-tipped nylon pins are ~2 mm thick at ~7-10 mm pitch "
                                        "(~6 mm clear); a boar-bristle brush's tufts leave ~2-4 mm. Estimates from "
                                        "brush construction, not measured; no source in hair_styling.md"),
    "split": (0, (0, 4), "the hand's finger-split after the set (hair_styling 7 row 18 / row 32, the look recipe 7.2a: "
                         "'a finger-split into 10-20 mm ribbons'; fingers split each lock into 2-4 sub-locks, P22/P54): "
                         "each lock is parted into this many ribbons along its natural seams - the largest piece in the "
                         "hand at the seam between the two halves that zipped separately before they joined (its last "
                         "merge) - and the bridges across a split are broken (the pieces do not re-merge). 0 / 1 = no "
                         "split. Belongs to the finish (finish.split) when actions.py is next opened"),
    "pack": ("group", ["group", "mass"], "how a zipped lock packs. group: each lock's own cross-section at the layer "
                                          "(its hairs' spread across the lock, as a uniform ellipse 4 pi sqrt(det)) "
                                          "pulled in to the area its bridges pack it to (N pi r^2 / phi_lock), across "
                                          "the lock only (bridges pull hairs side by side, they do not slide them "
                                          "along each other). mass: the old estimate, every lock scaled by sqrt("
                                          "phi_mass / phi_lock) toward its centre at the same arc, as if each lock "
                                          "already sat at the mass's mean packing"),
}


def _cfg(st, block, defaults):
    out = {k: v[0] for k, v in defaults.items()}
    out.update({k: v for k, v in ((st or {}).get(block) or {}).items() if k in defaults})
    return out


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


def phase_field(X, scale_mm, seed, jitter):
    """a wave phase per hair from a coherent field over the roots X (m): random Fourier features at about scale_mm (the
    10-29 phase field), so hairs a few mm apart share a phase and hairs several scales apart do not, plus each hair's own
    jitter (rad). -> (n,)"""
    rp = np.random.default_rng(int(seed) + 53)
    kv = rp.normal(0, 1, (24, 3)) / (float(scale_mm) / 1000.0)
    bv = rp.uniform(0, 2 * np.pi, 24)
    fld = np.cos(np.asarray(X, np.float64) @ kv.T + bv).sum(1) / np.sqrt(12.0)
    return np.pi * fld + rp.normal(0, float(jitter), len(fld))


def section_phase(X, section_mm, seed, jitter, parts=False):
    """a wave phase per hair from its drying section: Poisson-disk section centres section_mm apart over the roots, each
    hair takes its nearest centre's phase plus its own jitter (rad). Returns (phase (n,), section id (n,)); with parts
    also each section's phase (C,), so a caller can read a hair's own jitter as phase - ph_sec[sec]"""
    rng = np.random.default_rng(int(seed) + 61)
    key = np.asarray(X, np.float64) * 1000.0
    cand = rng.choice(len(key), size=min(len(key), 20000), replace=False)
    C = poisson(key, cand, float(section_mm), rng)
    sec = cKDTree(key[C]).query(key)[1]
    ph_sec = rng.uniform(0, 2 * np.pi, len(C))
    phase = ph_sec[sec] + rng.normal(0, jitter, len(key))
    return (phase, sec, ph_sec) if parts else (phase, sec)


def wave_follow(P, dphi, lam, s0=0.0, batch=20000):
    """each child's own phase about the wave it took from its guides (DESIGN_v2 5, waves phase 2; in place on P (n, K, 3)
    m). The rods carry the natural wave; a child interpolated from them carries their wave at their phase. A hair's wave
    is its own follicle's (Thibaut et al. 2005), dried in its section with the others, so it sits a little off the
    section's step: its own jitter dphi (n,) rad about its guide's. Moving the wave along the hair by dphi / k is moving
    each segment's direction along the arc by that much (the large bends, ~100s of mm, move by a few mm, nothing a
    viewer reads); the shift ramps in with the wave (waves.start_mm, s0 m, over half a wavelength) so the root's exit
    is untouched, and every segment keeps its length (hair does not stretch). lam: wavelength (m), scalar or (n,)."""
    dphi = np.asarray(dphi, np.float64)
    lam = np.broadcast_to(np.asarray(lam, np.float64), dphi.shape)
    for b0 in range(0, len(P), batch):
        sl = slice(b0, min(len(P), b0 + batch))
        Pb = P[sl].astype(np.float64)
        m, K = Pb.shape[:2]
        d = np.diff(Pb, axis=1)
        seg = np.linalg.norm(d, axis=2)
        t = d / np.maximum(seg, 1e-15)[..., None]
        sc = np.cumsum(seg, 1) - 0.5 * seg                       # each segment's mid arc (m)
        ds = dphi[sl] / (2 * np.pi / np.maximum(lam[sl], 1e-6))  # the wave's shift along the hair (m)
        q = sc + ds[:, None] * smoothstep(s0, s0 + 0.5 * lam[sl, None], sc)
        # per-row interpolation of the directions at arc q: offset each row so one sorted search serves all rows
        big = float(sc.max() + np.abs(ds).max() + 1.0)
        off = big * 2 * np.arange(m)[:, None]
        j = np.searchsorted((sc + off).ravel(), (np.clip(q, sc[:, :1], sc[:, -1:]) + off).ravel()).reshape(m, K - 1)
        j = np.clip(j - np.arange(m)[:, None] * (K - 1), 1, K - 2)
        r = np.take_along_axis(sc, j - 1, 1)
        w = np.clip((q - r) / np.maximum(np.take_along_axis(sc, j, 1) - r, 1e-15), 0.0, 1.0)[..., None]
        tn = (1 - w) * np.take_along_axis(t, (j - 1)[..., None], 1) + w * np.take_along_axis(t, j[..., None], 1)
        tn /= np.maximum(np.linalg.norm(tn, axis=2, keepdims=True), 1e-15)
        P[sl] = Pb[:, :1] + np.concatenate([np.zeros((m, 1, 3)), np.cumsum(tn * seg[..., None], 1)], 1)
    return P


def wave_shift(P, dphi, lam, s0=0.0, batch=20000):
    """each strand's WAVE moved along its arc by dphi (rad; the wave's phase + dphi), its mean line kept (fs-hair 11-12, the
    phase field's shifts reach pi). P (n, K, 3) m in place (any node spacing); lam (m) scalar or (n,); the shift ramps in
    from s0 over half a wavelength, as wave_follow. The mean line: the one-wavelength arc mean (arc_mean) of the strand
    resampled evenly by arc; the wave: the rest. The nodes back at their own arcs along the new curve, the root fixed. -> P"""
    dphi = np.asarray(dphi, np.float64)
    lam = np.broadcast_to(np.asarray(lam, np.float64), dphi.shape)
    for b0 in range(0, len(P), batch):
        sl = slice(b0, min(len(P), b0 + batch))
        Pb = np.asarray(P[sl], np.float64)
        m, K = Pb.shape[:2]
        seg = np.linalg.norm(np.diff(Pb, axis=1), axis=2)                     # (m, K-1)
        s = np.concatenate([np.zeros((m, 1)), np.cumsum(seg, 1)], 1)          # node arcs
        L = s[:, -1]
        M = 2 * K
        u = np.linspace(0.0, 1.0, M)[None, :] * L[:, None]                    # even arcs
        R = np.empty((m, M, 3))
        for i in range(m):
            for c in range(3):
                R[i, :, c] = np.interp(u[i], s[i], Pb[i, :, c])
        # each strand's OWN hanging wavelength (gravity stretches the rest wave: n2d's children p50 85 mm against 70): the
        # zero crossings of its wave about a wide (2 lambda) mean, along the wave's main direction, over the middle 20-90 %;
        # with the rest wavelength the one-wavelength mean kept ~1/5 of a stretched wave, and moving the rest of it cost
        # ~8 % of the wave (n2d32ub). Fewer than 3 crossings: the rest wavelength
        lr = np.asarray(lam[sl], np.float64)
        le = lr.copy()
        W2 = R - arc_mean(R, L, 2.0 * lr)
        a0, a1 = int(0.2 * M), int(0.9 * M)
        for i in range(m):
            Wm = W2[i, a0:a1] - W2[i, a0:a1].mean(0)
            try:
                v = np.linalg.svd(Wm, full_matrices=False)[2][0]
            except np.linalg.LinAlgError:
                continue
            x = Wm @ v
            zc = np.nonzero(np.diff(np.sign(x)) != 0)[0]
            if len(zc) >= 3:
                le[i] = np.clip(2.0 * (u[i, a0 + zc[-1]] - u[i, a0 + zc[0]]) / (len(zc) - 1), 0.8 * lr[i], 1.8 * lr[i])
        Mn = arc_mean(R, L, le)
        Wv = R - Mn
        ds = dphi[sl] / (2 * np.pi / np.maximum(le, 1e-6))                     # the wave's shift along the arc (m)
        q = u + ds[:, None] * smoothstep(s0, s0 + 0.5 * le[:, None], u)
        R2 = np.empty_like(R)
        for i in range(m):
            qi = np.clip(q[i], 0.0, L[i])
            for c in range(3):
                R2[i, :, c] = Mn[i, :, c] + np.interp(qi, u[i], Wv[i, :, c])
        # the nodes back at their own arcs ALONG THE NEW CURVE (its shape kept exactly, the strand's length and node spacing
        # kept; past the new curve's end, along its last direction), so nothing drifts from the root
        s2 = np.concatenate([np.zeros((m, 1)), np.cumsum(np.linalg.norm(np.diff(R2, axis=1), axis=2), 1)], 1)
        Q = np.empty_like(Pb)
        for i in range(m):
            L2 = s2[i, -1]
            inside = s[i] <= L2
            for c in range(3):
                Q[i, :, c] = np.interp(s[i], s2[i], R2[i, :, c])
            if not inside.all():
                e = R2[i, -1] - R2[i, -2]
                e = e / max(np.linalg.norm(e), 1e-15)
                Q[i, ~inside] = R2[i, -1] + (s[i, ~inside] - L2)[:, None] * e[None]
        Q[:, 0] = Pb[:, 0]
        P[sl] = Q
    return P


def _evenly(P, L, s):
    """points of evenly spaced strands P (n, K, 3) with lengths L at arcs s (n, m) (clamped to the strand)"""
    n, K = P.shape[:2]
    f = np.clip(s / np.maximum(L, 1e-12)[:, None] * (K - 1), 0, K - 1)
    i0 = np.minimum(np.floor(f).astype(np.int64), K - 2)
    w = (f - i0)[..., None]
    r = np.arange(n)[:, None]
    return P[r, i0] * (1 - w) + P[r, i0 + 1] * w


def _fibre_fraction(P, r, Q, h):
    """the fibre volume fraction of strands P (n, K, 3) m with radii r (n,) m in the cubic cell of side h (m) holding
    each query point Q (m, 3): sum of segment length x pi r^2 over the segments whose midpoints fall in that cell, over
    h^3. Cells are aligned on the strands' lowest corner."""
    P = np.asarray(P, np.float64)
    lo = P.reshape(-1, 3).min(0) - h
    hi = P.reshape(-1, 3).max(0) + h
    dims = np.floor((hi - lo) / h).astype(np.int64) + 1
    lins, vols = [], []
    for b0 in range(0, len(P), 20000):
        Pb = P[b0:b0 + 20000]
        mid = 0.5 * (Pb[:, 1:] + Pb[:, :-1]).reshape(-1, 3)
        vol = (np.linalg.norm(np.diff(Pb, axis=1), axis=2) * np.pi * r[b0:b0 + 20000, None] ** 2).ravel()
        ijk = np.floor((mid - lo) / h).astype(np.int64)
        lins.append((ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2])
        vols.append(vol)
    u, inv = np.unique(np.concatenate(lins), return_inverse=True)
    phi = np.bincount(inv, np.concatenate(vols)) / h ** 3
    q = np.floor((np.asarray(Q, np.float64) - lo) / h).astype(np.int64)
    ql = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    pos = np.clip(np.searchsorted(u, ql), 0, len(u) - 1)
    return np.where(u[pos] == ql, phi[pos], 0.0)


def arc_mean(Y, L, win):
    """the running mean of Y (n, K, 3) (points evenly spaced along each hair, length L (n,) m) over an arc window win
    (m, scalar or (n,)) centred on each point, the window shrunk at the root and the tip: a hair's mean line, its wave
    taken out (a box one wavelength wide has a zero at the wave's own frequency)"""
    n, K = Y.shape[:2]
    ds = np.maximum(np.asarray(L, np.float64), 1e-12) / (K - 1)
    hw = 0.5 * np.broadcast_to(np.asarray(win, np.float64), (n,)) / ds          # half-window in samples
    C = np.concatenate([np.zeros((n, 1, Y.shape[2])), np.cumsum(0.5 * (Y[:, 1:] + Y[:, :-1]), 1)], 1)
    i = np.arange(K, dtype=np.float64)[None, :]
    a = np.clip(i - hw[:, None], 0, K - 1)
    b = np.clip(i + hw[:, None], 0, K - 1)

    def at(f):
        j = np.minimum(np.floor(f).astype(np.int64), K - 2)
        w = (f - j)[..., None]
        return np.take_along_axis(C, j[..., None], 1) * (1 - w) + np.take_along_axis(C, j[..., None] + 1, 1) * w
    return (at(b) - at(a)) / np.maximum(b - a, 1e-9)[..., None]


def wave_rms_by_third(P, lam, n=2000, seed=0, min_mm=250.0):
    """the hairs' waviness: P (m, K, 3) m (any node spacing; a repeated end node is a zero-length segment); up to n hairs
    longer than min_mm resampled at 2 mm by arc, the residual from their
    one-wavelength (lam m) arc mean, its rms (mm) by third of the length, p50 over the hairs (the first and last half
    wavelength left out, where the window shrinks). A diagnostic for the report, not a target."""
    Pn = np.asarray(P, np.float64)
    L = np.linalg.norm(np.diff(Pn, axis=1), axis=2).sum(1)
    idx = np.nonzero(L > min_mm / 1000.0)[0]
    if len(idx) == 0 or lam <= 0:
        return None
    idx = np.random.default_rng(int(seed)).choice(idx, min(n, len(idx)), replace=False)
    Pi, Li = Pn[idx], L[idx]
    out = [[], [], []]
    for Q, Lq in zip(Pi, Li):
        sq = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
        m = int(Lq / 0.002) + 1
        u = np.linspace(0, Lq, m)
        R = np.stack([np.interp(u, sq, Q[:, d]) for d in range(3)], 1)     # by arc: the nodes need not be even
        res = R - arc_mean(R[None], np.array([Lq]), lam)[0]
        e = int(np.ceil(0.5 * lam / 0.002))
        for t in range(3):
            lo, hi = max(e, t * m // 3), min(m - e, (t + 1) * m // 3)
            if hi - lo >= 10:
                out[t].append(np.sqrt((res[lo:hi] ** 2).sum(1).mean()) * 1000)
    return [round(float(np.median(o)), 3) if o else None for o in out]


def lock_causes(P, unit, r, phase, fibre, st, section=None, s_layer=None, keep_wave_m=0.0, log=print):
    """locks from causes on children P (n, K, 3) m, evenly spaced along each hair (children.interpolate's output), in
    place. unit (n,) the follicular unit of each hair (children.follicular_units; arange(n) = none); r (n,) m each
    hair's radius; phase (n,) rad (section_phase); fibre a fibre.Fibre (E, k, rho); st the style (oil.*, locks.*,
    waves.*). section (n,): the drying section of each hair (section_phase) when the look's dry act dried the hair in
    sections: a section was held together wet in the brush or hand, so its groups met (captured) whatever their gap now;
    None = only groups within oil.capture_mm meet. With brush.on and brush.through (the look's stroke drawn from the root
    to the ends after the hair is dried) the brush carries neighbouring sections together: its pins split the mass into
    lanes locks.brush_lane_mm wide across the stroke, and every hair in one lane is pressed against its lane-mates, so
    two groups meet when their facing edges across the stroke lie within a lane and their depth apart is within the
    bristles' reach (brush.reach_mm). The stroke runs along the hair, so the offset along the hair does not part them.
    Meeting is all the brush gives: whether the pair then zips is still the bridges against the bending and the waves
    (Weff, the zipping length). s_layer (m): the absolute arc the groups are found at (default half
    the median length; a hair shorter than it is taken at its end). At one absolute arc the hairs from neighbouring
    roots lie across the hair's direction, as a cut through the mass; at one arc FRACTION hairs of different lengths
    sit at different heights and the mass reads 20-80x too loose (measured on gr3: phi 0.0005).

    keep_wave_m (m, scalar or (n,)): when the hairs carry their natural wave (the rods' rest curvature, settle.rest_waves), the pack
    moves each hair by its mean line over this arc (one wavelength) toward the lock's mean line, and the hair keeps its
    own wave. The bridges draw hairs side by side; they do not iron a wavy fibre straight: a lock's members dried in
    one section nest wave in wave, and a lock that holds two sections (the brush's carry) has two trains whose plain
    mean is near straight, so a pull toward that mean took (1 - compaction) of every member's wave away (round 17:
    guides 3.0 mm rms, children 0.73). 0 = toward the centre curve point by point, as before.

    Returns (P, lock id (n,), report). The hair's length is kept (the moved polyline is rebuilt from the root with its own
    segment lengths)."""
    t0 = time.time()
    oc = _cfg(st, "oil", OIL_DEFAULTS)
    lc = _cfg(st, "locks", LOCK_CAUSE_DEFAULTS)
    W8 = (st or {}).get("waves") or {}
    n, K = P.shape[:2]
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    L = seg.sum(1)
    s_l = 0.5 * float(np.median(L)) if s_layer is None else float(s_layer)
    s_j = np.minimum(L, s_l)
    # a hair joins a lock only over the length it shares with the lock. A hair that ends before the layer arc is not
    # at the layer: it is not zipped there, does not bridge two groups, is not counted in a group's centre, size or
    # packing, and is not pulled toward a lock's centre curve it never reaches. It keeps its own place (a group of
    # its own, never merged). Measured on gr8 (generate, 141k hairs, layer 160.9 mm): with the short hairs counted
    # at their ends the coalescence chained 50 groups across the head and the pack drew 1009 short front hairs
    # (35-77 mm) 23-76 mm onto the forehead.
    at_layer = L >= s_l
    r = np.asarray(r, np.float64)
    EI = fibre.E * np.pi * r ** 4 / 4
    wl = fibre.rho * 9.81 * fibre.wet_w * np.pi * r ** 2              # N/m, weight per length
    gamma = oc["gamma_mN_m"] / 1000.0
    o = oil_load(s_j, oc["days_since_wash"], oc["product"], oc["sebum_mm_per_day"], oc["sebum_load"])
    Wa = adhesion_per_length(o, r, gamma, oc["load_half"])
    kE = max(fibre.kE, 1.0)
    phi_lock = np.minimum(np.sqrt(Wa / (np.pi * r ** 2 * kE)), lc["phi_max"])
    # the waves at the layer: amplitude pulled out by the weight below (Miller et al. 2014; DESIGN_v2 4.9 envelope)
    lam = float(W8.get("wavelength_mm", 0) or 0) / 1000.0
    am = W8.get("amp_mm", [0, 0])
    A0 = 0.5 * (float(am[0]) + float(am[1])) / 1000.0 if lam > 0 else 0.0
    kw = 2 * np.pi / lam if lam > 0 else 0.0
    A = A0 / (1.0 + wl * (L - s_j) / np.maximum(EI * kw ** 2, 1e-30)) if lam > 0 else np.zeros(n)
    Q = _evenly(P.astype(np.float64), L, s_j[:, None])[:, 0]
    # the mass's own packing where each hair crosses the layer: the fibre volume (segment length x pi r^2 of every
    # hair) in the 3 mm cell around the hair's layer point, over the cell's volume - the volume fraction the settle and
    # the research measure. The old estimate (hairs within 2 mm at the same absolute arc, as one cross-section) read
    # 9x too loose on gr3 (p50 0.0020 against 0.0184 by volume): hairs from roots at different heights cross the layer
    # at different heights, so one arc is not one cross-section. That made every lock pack to ~0.17 of its width.
    phi_mass = np.clip(_fibre_fraction(P, r, Q, 0.003), 1e-4, 0.6)
    a_r = np.pi * r ** 2                                              # m2, each fibre's section
    # the brush's carry (brush.through): the hair's direction at the layer and the head's outward direction there, to
    # split a pair's offset into along the hair (the stroke's direction, no parting), across it (the lanes) and depth
    bb = (st or {}).get("brush") or {}
    carry = bool(bb.get("on", True)) and bool(bb.get("through", False))
    if carry:
        lane = float(lc["brush_lane_mm"]) / 1000.0
        reach = float(bb.get("reach_mm", 15.0)) / 1000.0
        Tq = (_evenly(P.astype(np.float64), L, (s_j + 0.003)[:, None])[:, 0]
              - _evenly(P.astype(np.float64), L, np.maximum(s_j - 0.003, 0.0)[:, None])[:, 0])
        Tq /= np.maximum(np.linalg.norm(Tq, axis=1, keepdims=True), 1e-15)
        cen = (st or {}).get("_centre", None)
        cen = np.asarray(cen if cen is not None else P[:, 0].mean(0), np.float64)
        Nq = Q - cen
        Nq /= np.maximum(np.linalg.norm(Nq, axis=1, keepdims=True), 1e-15)
    sec_arr = None if section is None else np.asarray(section)
    # coalescence: groups start as follicular units (the hairs that reach the layer); a hair that does not reach it is
    # a group of its own that never takes part (keyed past every unit id, so it never shares a unit's group)
    uk = np.asarray(unit).astype(np.int64)
    uk = np.where(at_layer, uk, int(uk.max()) + 1 + np.arange(n))
    _, lab = np.unique(uk, return_inverse=True)
    lab0 = lab.copy()
    m0 = int(lab.max()) + 1
    # the merge tree (for the finger-split's seams): leaves 0..m0-1 are the starting groups, internal node m0 + j is the
    # j-th merge, joining nodes t_left[j] and t_right[j] and holding t_size[j] hairs
    node_of = np.arange(m0)
    t_left, t_right, t_size = [], [], []
    cap = oc["capture_mm"] / 1000.0
    merges = []
    carried = []                                                      # merges whose pair met only by the brush, per round
    refused = {}
    for it in range(int(lc["rounds"])):
        m = lab.max() + 1
        nc = np.bincount(lab, minlength=m).astype(np.float64)
        C = np.stack([np.bincount(lab, Q[:, d], m) for d in range(3)], 1) / nc[:, None]
        EIc = np.bincount(lab, EI, m) / nc
        Wc = np.bincount(lab, Wa, m) / nc
        sc = np.bincount(lab, s_j, m) / nc
        Ac = np.bincount(lab, A, m) / nc
        ph = np.arctan2(np.bincount(lab, np.sin(phase), m), np.bincount(lab, np.cos(phase), m))
        area = np.bincount(lab, a_r, m)
        pl = np.maximum(np.bincount(lab, phi_lock, m) / nc, np.bincount(lab, phi_mass, m) / nc)
        R = np.sqrt(area / (np.pi * pl))                              # the group packed as its bridges pack it
        # only groups at the layer are candidates (the hairs that end before it are their own groups, left out)
        go = np.nonzero(np.bincount(lab, at_layer.astype(np.float64), m) > 0)[0]
        kq = min(7, len(go))
        if kq < 2:
            break
        d, nb = cKDTree(C[go]).query(C[go], k=kq)
        nb = go[nb]
        # the candidate pairs: each group's 6 nearest centres, taken both ways round. Whether two neighbouring groups
        # meet and zip does not depend on which of them lists the other among its 6 nearest; with one-way lists a big
        # group's own 6 nearest can all be groups it cannot take while small groups list it as their best, no pair is
        # mutually best and the coalescence stopped early (measured on gr3's full count: stopped at round 29 with 2184
        # of 5628 groups still holding a merge their own rule allows; whole-head width p50 6 mm against 15.6 on the
        # back patch, where the groups are big).
        a = np.repeat(go, kq - 1)
        nb = nb[:, 1:].ravel()
        lo_, hi_ = np.minimum(a, nb), np.maximum(a, nb)
        _, ui = np.unique(lo_ * np.int64(m) + hi_, return_index=True)
        a, nb, d = lo_[ui], hi_[ui], d[:, 1:].ravel()[ui]
        gap = np.maximum(d - R[a] - R[nb], 0.0)
        na, nbb = nc[a], nc[nb]
        mu = na * nbb / (na + nbb)
        EIm = 0.5 * (EIc[a] + EIc[nb])
        # adhesion gained per length of zip: the bridges along the contact line between the groups (~ sqrt of the smaller)
        gain = np.minimum(Wc[a], Wc[nb]) * np.sqrt(np.minimum(na, nbb))
        # forcing their waves into step: bending the difference 2 A sin(dphi / 2) at wavenumber k, per length
        dA = 2 * 0.5 * (Ac[a] + Ac[nb]) * np.abs(np.sin(0.5 * (ph[a] - ph[nb])))
        cost = mu * EIm * kw ** 4 * dA ** 2 / 4
        Weff = gain - cost
        # the zipping length: E(l) = 3 EI mu g^2 / (2 l^3) - Weff (L - l) is least at l* = (9 EI mu g^2 / (2 Weff))^(1/4)
        lz = (9 * EIm * mu * gap ** 2 / (2 * np.maximum(Weff, 1e-30))) ** 0.25
        met0 = gap < cap
        if sec_arr is not None:
            secg = np.zeros(m, np.int64)
            secg[lab] = sec_arr
            met0 |= secg[a] == secg[nb]
        met = met0
        if carry:
            # the pair's own frame: t along the hair, n outward (off the head), b across the stroke (the lanes)
            Tc = np.stack([np.bincount(lab, Tq[:, d], m) for d in range(3)], 1)
            Nc = np.stack([np.bincount(lab, Nq[:, d], m) for d in range(3)], 1)
            t = Tc[a] + Tc[nb]
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-15)
            nv = Nc[a] + Nc[nb]
            nv -= (nv * t).sum(1)[:, None] * t
            nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-15)
            bv = np.cross(t, nv)
            # each group's half-width across the stroke as it lies (not packed): sqrt(3) x its rms across its own
            # stroke direction, the half-width of a uniform strip
            Tg = Tc / np.maximum(np.linalg.norm(Tc, axis=1, keepdims=True), 1e-15)
            Ng = Nc - (Nc * Tg).sum(1)[:, None] * Tg
            Bg = np.cross(Tg, Ng / np.maximum(np.linalg.norm(Ng, axis=1, keepdims=True), 1e-15))
            hw = np.sqrt(3.0 * np.bincount(lab, ((Q - C[lab]) * Bg[lab]).sum(1) ** 2, m) / nc)
            D = C[nb] - C[a]
            edge = np.abs((D * bv).sum(1)) - hw[a] - hw[nb]
            met = met0 | ((edge < lane) & (np.abs((D * nv).sum(1)) < reach))
        zip_ok = lz < 0.5 * (sc[a] + sc[nb])
        ok = (Weff > 0) & met & zip_ok
        refused = {"pairs": int(len(a)), "not_met": int((~met).sum()), "met_but_waves_beat_bridges":
                   int((met & (Weff <= 0)).sum()), "met_but_zip_longer_than_layer_arc":
                   int((met & (Weff > 0) & ~zip_ok).sum()), "allowed": int(ok.sum())}
        if not ok.any():
            break
        # each group's best partner over every pair it is in, ties to the lower partner id: then the lowest-id group
        # holding a least-score pair and its partner are always mutually best, so a round merges whenever any pair can
        e = np.nonzero(ok)[0]
        ends = np.concatenate([a[e], nb[e]])
        other = np.concatenate([nb[e], a[e]])
        o_ = np.lexsort((other, np.concatenate([lz[e], lz[e]]), ends))
        head = np.ones(len(o_), bool)
        head[1:] = ends[o_][1:] != ends[o_][:-1]
        best = -np.ones(m, np.int64)
        best[ends[o_][head]] = other[o_][head]
        mutual = (best >= 0) & (best[np.maximum(best, 0)] == np.arange(m))
        pa = np.nonzero(mutual & (np.arange(m) < best))[0]
        if len(pa) == 0:
            break
        # the merges whose pair met only by the brush's carry (the pair keys a * m + nb are sorted: np.unique above)
        pi_ = np.searchsorted(a * np.int64(m) + nb, pa * np.int64(m) + best[pa])
        carried.append(int((~met0[pi_]).sum()))
        ids = m0 + len(t_left) + np.arange(len(pa))
        t_left.extend(node_of[pa].tolist())
        t_right.extend(node_of[best[pa]].tolist())
        t_size.extend((nc[pa] + nc[best[pa]]).tolist())
        node_new = node_of.copy()
        node_new[pa] = ids
        new = np.arange(m)
        new[best[pa]] = pa
        uq, new = np.unique(new, return_inverse=True)
        node_of = node_new[uq]
        lab = new[lab]
        merges.append(int(len(pa)))
    m = lab.max() + 1
    groups_coalesced = int(m)
    # the finger-split (locks.split): the hand parts each lock into k ribbons. A lock is a tree of merges; its natural
    # seams are the merge lines, and the seam between the two halves of the piece now in the hand is the one the
    # fingers open (the piece's last merge: the two halves zipped on their own before they joined). The largest piece
    # is split first, until the lock is in k pieces or every piece is a single starting group. The bridges across a
    # split are broken: the pieces are not run through the coalescence again.
    k_split = int(lc["split"])
    split_rep = None
    if k_split >= 2 and len(t_left):
        J = len(t_left)
        tl, tr = np.asarray(t_left, np.int64), np.asarray(t_right, np.int64)
        size = np.concatenate([np.bincount(lab0, minlength=m0).astype(np.float64), np.asarray(t_size)])
        is_piece = np.zeros(m0 + J, bool)
        for g in range(int(m)):
            pieces = [int(node_of[g])]
            while len(pieces) < k_split:
                cand = [p for p in pieces if p >= m0]
                if not cand:
                    break
                p = max(cand, key=lambda q: size[q])
                pieces.remove(p)
                pieces += [int(tl[p - m0]), int(tr[p - m0])]
            is_piece[pieces] = True
        # each node's piece: top down (a parent's id is always above its children's)
        parent = -np.ones(m0 + J, np.int64)
        parent[tl] = m0 + np.arange(J)
        parent[tr] = m0 + np.arange(J)
        piece = -np.ones(m0 + J, np.int64)
        piece[is_piece] = np.arange(int(is_piece.sum()))
        for v in range(m0 + J - 1, -1, -1):
            if piece[v] < 0 and parent[v] >= 0:
                piece[v] = piece[parent[v]]
        _, lab = np.unique(piece[lab0], return_inverse=True)
        m = lab.max() + 1
        split_rep = {"k": k_split, "locks_before": groups_coalesced, "pieces_after": int(m)}
    nc = np.bincount(lab, minlength=m).astype(np.float64)
    # compaction toward the group's own centre curve, each group by its bridges' packing; from the root over the
    # zipping length of the group's root spread (the follicle holds the root apart)
    pack = str(lc["pack"])
    if pack == "group":
        # the lock's own cross-section now: its hairs' offsets at the layer across the lock's direction, as a uniform
        # ellipse (area 4 pi sqrt of the product of the two variances); the bridges pack it to N pi r^2 / phi_lock
        # (phi_lock: the bridges' pull against the van Wyk resistance of the wavy mass, so the hairs touch at the
        # mass's contact spacing, not side by side). A lock already as tight stays as it is.
        Tq2 = (_evenly(P.astype(np.float64), L, (s_j + 0.003)[:, None])[:, 0]
               - _evenly(P.astype(np.float64), L, np.maximum(s_j - 0.003, 0.0)[:, None])[:, 0])
        Tg2 = np.stack([np.bincount(lab, Tq2[:, d], m) for d in range(3)], 1)
        Tg2 /= np.maximum(np.linalg.norm(Tg2, axis=1, keepdims=True), 1e-15)
        Cq = np.stack([np.bincount(lab, Q[:, d], m) for d in range(3)], 1) / nc[:, None]
        V = Q - Cq[lab]
        V -= (V * Tg2[lab]).sum(1)[:, None] * Tg2[lab]
        cov = np.zeros((m, 3, 3))
        for i in range(3):
            for j in range(i, 3):
                cov[:, i, j] = cov[:, j, i] = np.bincount(lab, V[:, i] * V[:, j], m) / nc
        ev = np.linalg.eigvalsh(cov)                                  # ascending; the smallest is along the lock (~0)
        area_now = 4 * np.pi * np.sqrt(np.maximum(ev[:, 2] * ev[:, 1], 0.0))
        area_pk = np.bincount(lab, a_r, m) / np.maximum(np.bincount(lab, phi_lock, m) / nc, 1e-9)
        fac = np.sqrt(np.clip(area_pk / np.maximum(area_now, 1e-30), 0.0, 1.0))
        del Tq2, V, cov
    else:
        fac = np.sqrt(np.clip((np.bincount(lab, phi_mass, m) / nc) / np.maximum(np.bincount(lab, phi_lock, m) / nc,
                                                                                  1e-9), 0.0, 1.0))
    fac = np.maximum(fac, 0.1)
    X0 = P[:, 0].astype(np.float64)
    Cr = np.stack([np.bincount(lab, X0[:, d], m) for d in range(3)], 1) / nc[:, None]
    spread = np.sqrt(np.bincount(lab, ((X0 - Cr[lab]) ** 2).sum(1), m) / nc)
    Wl = np.maximum(np.bincount(lab, Wa, m) / nc, 1e-30)
    EIl = np.bincount(lab, EI, m) / nc
    lzip = np.minimum((9 * EIl * spread ** 2 / (2 * Wl)) ** 0.25, 0.5)
    Lmax = float(L.max())
    ng = 48
    sg = np.linspace(0, Lmax, ng)
    Cg = np.zeros((m, ng, 3))
    Ng = np.zeros((m, ng))
    for b0 in range(0, n, 20000):
        sl = slice(b0, min(n, b0 + 20000))
        Pg = _evenly(P[sl].astype(np.float64), L[sl], np.broadcast_to(sg, (sl.stop - b0, ng)))
        on = (sg[None, :] <= L[sl, None] + 1e-9).astype(np.float64)
        lb = lab[sl]
        for g in range(ng):
            Ng[:, g] += np.bincount(lb, on[:, g], m)
            for d in range(3):
                Cg[:, g, d] += np.bincount(lb, Pg[:, g, d] * on[:, g], m)
    Cg /= np.maximum(Ng, 1e-9)[..., None]
    if pack == "group":
        Tcg = np.gradient(Cg, axis=1)                                  # the centre curve's direction along the arc
        Tcg /= np.maximum(np.linalg.norm(Tcg, axis=2, keepdims=True), 1e-15)
    alone = nc[lab] < 2
    kw_m = np.broadcast_to(np.asarray(keep_wave_m, np.float64), (n,))
    for b0 in range(0, n, 20000):
        sl = slice(b0, min(n, b0 + 20000))
        Pb = P[sl].astype(np.float64)
        s = L[sl, None] * np.linspace(0, 1, K)[None, :]
        f = np.clip(s / max(Lmax, 1e-12) * (ng - 1), 0, ng - 1)
        i0 = np.minimum(np.floor(f).astype(np.int64), ng - 2)
        w = (f - i0)[..., None]
        lb = lab[sl][:, None]
        Cs = Cg[lb, i0] * (1 - w) + Cg[lb, i0 + 1] * w
        pull = (1 - fac[lab[sl]])[:, None] * smoothstep(0.0, np.maximum(lzip[lab[sl]], 1e-4)[:, None], s)
        pull *= ~alone[sl, None]
        dC = Cs - Pb
        if pack == "group":
            # across the lock only: the bridges pull hairs side by side, they do not slide them along each other
            tc = Tcg[lb, i0] * (1 - w) + Tcg[lb, i0 + 1] * w
            tc /= np.maximum(np.linalg.norm(tc, axis=2, keepdims=True), 1e-15)
            dC -= (dC * tc).sum(2)[..., None] * tc
        if kw_m.max() > 0:
            dC = arc_mean(dC, L[sl], kw_m[sl])                          # the mean lines move; each wave stays its own
        Pn = Pb + pull[..., None] * dC
        # length kept: the new directions, each segment its own length, built from the root
        dv = np.diff(Pn, axis=1)
        dv /= np.maximum(np.linalg.norm(dv, axis=2, keepdims=True), 1e-15)
        Pn = Pb[:, :1] + np.concatenate([np.zeros((len(Pb), 1, 3)), np.cumsum(dv * seg[sl, :, None], 1)], 1)
        P[sl] = Pn
    sizes = nc[lab]
    rep = {"groups": int(m), "merge_rounds": merges, "hairs_per_group_p10_50_90_hairweighted":
           np.percentile(sizes, [10, 50, 90]).tolist(),
           "oil_load_mid_p50": round(float(np.median(o)), 3), "W_uJ_per_m_p50": round(float(np.median(Wa)) * 1e6, 3),
           "phi_lock_p50": round(float(np.median(phi_lock)), 4), "phi_mass_p50": round(float(np.median(phi_mass)), 4),
           "compaction_p50_hairweighted": round(float(np.median(fac[lab])), 3),
           "zip_mm_p50": round(float(np.median(lzip[lab])) * 1000, 2), "layer_mm": round(s_l * 1000, 1),
           "last_round_pairs": refused, "pack": pack, "keep_wave_mm": round(float(kw_m.mean()) * 1000, 2),
           "split": split_rep,
           "locks_coalesced": groups_coalesced,
           "hairs_reaching_layer": int(at_layer.sum()), "hairs_short_of_layer_left_alone": int((~at_layer).sum()),
           "groups_at_layer": int(len(np.unique(lab[at_layer]))),
           "hairs_per_layer_group_p10_50_90_hairweighted": (np.percentile(sizes[at_layer], [10, 50, 90]).tolist()
                                                           if at_layer.any() else None),
           "brush_carry": ({"lane_mm": round(lane * 1000, 2), "reach_mm": round(reach * 1000, 2),
                            "merges_met_only_by_brush": int(sum(carried)), "merges": int(sum(merges))}
                           if carry else None)}
    if sec_arr is not None:
        # how many drying sections each lock holds (the brush's carry is what lets one lock hold more than one)
        ls = np.unique(lab.astype(np.int64) * (int(sec_arr.max()) + 1) + sec_arr)
        nsec = np.bincount(ls // (int(sec_arr.max()) + 1), minlength=m)
        rep["hairs_in_locks_of_2plus_sections_share"] = round(float((nsec[lab] >= 2).mean()), 3)
    rep["seconds"] = round(time.time() - t0, 2)
    log("  form: locks from causes - %d groups, %s rounds" % (m, merges))
    return P, lab, rep


def section_waves(P, phase, r, fibre, st):
    """the phase-1 wave on children P (n, K, 3) (in place), each hair at its own (section) phase, the amplitude pulled
    out by the weight below the point (DESIGN_v2 4.9: A / (1 + T / (EI k^2)), T = w x the length below). Displaced in
    the sheet's side direction B (frames) about a head centre st['_centre'] (m) or the roots' mean."""
    W8 = (st or {}).get("waves") or {}
    lam = float(W8.get("wavelength_mm", 0) or 0) / 1000.0
    if lam <= 0:
        return P
    am = W8.get("amp_mm", [0, 0])
    A0 = 0.5 * (float(am[0]) + float(am[1])) / 1000.0
    s0 = 0.5 * sum(W8.get("start_mm", [0, 0])) / 1000.0 if "start_mm" in W8 else 0.0
    kw = 2 * np.pi / lam
    r = np.asarray(r, np.float64)
    EI = fibre.E * np.pi * r ** 4 / 4
    wl = fibre.rho * 9.81 * fibre.wet_w * np.pi * r ** 2
    c = st.get("_centre", None) if st else None
    c = np.asarray(c if c is not None else P[:, 0].mean(0), np.float64)
    for b0 in range(0, len(P), 20000):
        sl = slice(b0, min(len(P), b0 + 20000))
        Pb = P[sl].astype(np.float64)
        s = arc(Pb)
        L = s[:, -1:]
        T, B, N2, Nout = frames(Pb, c)
        A = A0 / (1 + wl[sl, None] * (L - s) / (EI[sl, None] * kw ** 2)) * smoothstep(s0, s0 + 0.5 * lam, s)
        Pn = Pb + (A * np.sin(kw * s + phase[sl, None]))[..., None] * B
        # hair does not stretch: the waved directions, each segment its own length, built from the root (a 5 mm wave
        # at 90 mm otherwise adds ~(A k)^2 / 4 of the length: 12 mm on a 400 mm hair, measured on gr3)
        seg = np.linalg.norm(np.diff(Pb, axis=1), axis=2)
        dv = np.diff(Pn, axis=1)
        dv /= np.maximum(np.linalg.norm(dv, axis=2, keepdims=True), 1e-15)
        P[sl] = Pb[:, :1] + np.concatenate([np.zeros((len(Pb), 1, 3)), np.cumsum(dv * seg[..., None], 1)], 1)
    return P
