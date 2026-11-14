"""Hair Studio generator (numpy + scipy, Apache-2.0): a scalp + a STYLE -> strands, deterministic for a seed,
batch-wise (peak RAM ~ 1 GB for 145k strands x 32 points).

    generate(scalp, collider, style, menton_z, log=print) -> (arrays dict for the npz, report dict)
    generate(..., head=None, head_cache=None, regrow=False, case=None, passes="auto")    grow mode, flow.order 'after'
    finish_head(scalp, collider, style, menton_z, head)       the rested head finished as it lies (no style stage)

Grow mode (flow.source 'grow') has two orders (flow.order, a sim key). 'during' (the default; every earlier run
reproduces): _grow, the acts on while the hair grows. 'after' (DESIGN_grow_then_style, 10-31: the hair is grown
once, then styled from there): _grow_after - the Hair stage is a grown head (head.py: given, or found in
head_cache by its key, or grown with no act and saved there), instantiated at its 'rested' snapshot; the Style stage
puts the style's product on it, builds the acts on those (cut) rods, runs the passes (passes.py's run_passes when it is
there, else acts_passes: the act energies as one lagged block), then the same tail as the legacy order (the water set,
release, tousle, the final); then the same finish (children, form, baby hairs, radius, push-out, colour).

Steps
1. Roots: area-weighted on the scalp inside the hairline, feathered over edge_mm; the part (centre / side / none) gives
   each root a side, and a share of the roots on the line's front cross it (no scalp line). Edge hairs within edge_mm
   (fine), the closest of them short.
2. Lengths: loose styles - tips at the menton minus tip_below_menton_mm (+ spread, a share shorter, an A-line tilt,
   face-framing pieces higher); a layered cut takes the crown roots shorter. Gathered styles - each strand keeps its
   loose length (her hair, gathered), from a first loose walk.
3. Flow: the comb walk (flow.walk) along the style's direction field, hung under gravity, kept out of the body
   (collider), tucked behind the ears; gathered styles walk to the gather point, bridge into the tie and follow the
   tail (ponytail) or the coiled rope (bun). Resampled to `points` per strand.
4. Form (form.shape): locks, clumping, waves, crown lift, jitter, wander, frizz, fly-aways; bevel (turn under).
5. Baby hairs at the hairline (short, fine, lighter, curled, standing a little).
6. Colour per point (form.melanin_field -> colour.sigma with the style's redness and ash).
7. Collision: every point pushed out of the collider (exact distance), then reported: points inside the skin (target
   0) and the minimum clearance; the tie (an analytic torus: its parameters, no triangles).
Arrays: pts (n, K, 3) f32 m; r_root, r_tip (n,) f32 m; radius (n, K) f32 m (grow mode: the fibre's radius per point); kind (n,) i8 (0 main, 1 edge fine, 2 edge short, 3 baby);
melp (n, K) f32; mel, red, mel_root (n,) f32; sigma (n, K, 3) f16 (absorption per unit radius); lock, sub (n,) i32;
fly, cross (n,) bool; din, s_tie (n,) f32; tie_torus (15,) f64 (gathered: centre 3, axis 3, U 3, W 3, major R, minor
r, width; metres); grid, thb, centre; style_json, material_json (str).
Mesh-free since 11-14 (fs-hairclean, P14): the body and the scalp are fields (field_collider); the scalp cap (triangles
offset off the skin) is gone, the renderer draws her skin; the tie is its torus's parameters, not a triangle tube.
"""
import copy
import json
import os
import time

import numpy as np

from . import colour as CO
from . import flow as FL
from . import form as FO
from . import style as SY
from .ram import peak_mb
from .scalp import smoothstep


STACK_SIGMA_MM, STACK_MAX_MM = 4.0, 80.0          # the stack's smoothing and its cap (40 clipped the back at packing 0.025: 10-29)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def resample_arc(P, K):
    """polylines (m, S, 3) with any spacing (repeated points allowed) -> K points uniform in arc, and the lengths"""
    m, S = P.shape[:2]
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    cum = np.concatenate([np.zeros((m, 1)), np.cumsum(seg, 1)], 1)
    L = cum[:, -1]
    t = np.linspace(0, 1, K)[None, :] * L[:, None]
    idx = np.empty((m, K), np.int64)
    for k in range(K):
        idx[:, k] = (cum <= t[:, k:k + 1] + 1e-12).sum(1) - 1
    idx = np.clip(idx, 0, S - 2)
    r = np.arange(m)[:, None]
    c0, c1 = cum[r, idx], cum[r, idx + 1]
    w = np.clip((t - c0) / np.maximum(c1 - c0, 1e-12), 0, 1)[..., None]
    return P[r, idx] * (1 - w) + P[r, idx + 1] * w, L


def torus(T, A, U, W, R, r, width=0.0):
    """a tie: a torus about axis A at T (major R, minor r), U and W spanning its plane; width > 2r stretches it along A (a
    wrapped band). An analytic form, kept as its 15 parameters (a field: the distance to it is closed-form)"""
    return np.concatenate([np.asarray(T, np.float64), np.asarray(A, np.float64), np.asarray(U, np.float64),
                           np.asarray(W, np.float64), [float(R), float(r), float(width)]])


def heights(st, X, din, phi_r, th_r, rng):
    """per strand the plateau height Hp (m) and the lift angle's tangent: points are kept h(s) above the skin"""
    Li = st["lift"]
    n = len(X)
    upstream = np.clip((102.0 - th_r) / (102.0 - 23.0), 0, 1)
    front_up = np.where(np.abs(phi_r) < 60, np.clip(1 - din / 30.0, 0, 1), 0)
    p0, p1 = Li["plateau_mm"]
    Hpl = (p0 + (p1 - p0) * np.maximum(upstream, front_up) + rng.uniform(0, 1.5, n)) / 1000
    if Li["front_flat"]:
        f0, f1 = Li["front_mm"]
        frontal = np.abs(phi_r) < 70
        capped = np.minimum(Hpl, (f0 + (f1 - f0) * smoothstep(0.0, 40.0, din) + rng.uniform(0, 1.0, n)) / 1000)
        # 10-29: the cap held EVERY frontal root at front_mm[1] - the whole front half of the head, not only the hairline as
        # documented; front_release_mm [a, b] fades it out between a and b mm in from the hairline ([0, 0]: never - the
        # behaviour before, kept so earlier styles reproduce)
        r0, r1 = Li.get("front_release_mm", [0.0, 0.0])
        keep = 1.0 if r1 <= 0 else 1 - smoothstep(r0, r1, din)
        Hpl = np.where(frontal, keep * capped + (1 - keep) * Hpl, Hpl)
    tl = np.tan(np.radians(rng.uniform(*Li["angle_deg"], n)))
    return Hpl, tl


def h_of(s, Hp, tl):
    Ll = 2 * Hp / tl
    h = np.where(s < Ll, s * tl * (1 - s / (2 * Ll)), Hp)
    return np.maximum(h, 0.00005)


def _roots_part_kinds(scalp, st, rng, rep, tick):
    """step 1: roots, part, kinds (shared by the comb and grow branches; the RNG draws are exactly the legacy
    order, so the comb branch is byte-identical). Returns a dict of the arrays and the comb frame."""
    seed = int(st["seed"])
    R, HL, PT, FLW = st["roots"], st["hairline"], st["part"], st["flow"]
    # 1. roots, part, kinds
    PA = float(R.get("patch_amp", 0.0))
    X, NR, scalp_cm2 = scalp.sample_roots(int(round(R["count"] * (1 + PA))), R["edge_mm"], R["edge_floor"], rng)
    if PA > 0:
        # natural density patches (10-29: the volume comes from the hair growing in denser patches): a smooth field of
        # Gaussian blobs (+-1, sigma patch_mm) over the scalp, normalised to +-1; the roots kept with probability
        # (1 + PA f) / (1 + PA), so the count stays ~R["count"] and the density varies +-PA patch to patch
        rp = np.random.default_rng(seed + 31)
        sig = R["patch_mm"] / 1000
        nb = max(8, int(scalp_cm2 * 1e-4 / sig ** 2))
        cb = X[rp.choice(len(X), nb, replace=False)]; wb = rp.choice([-1.0, 1.0], nb)
        f = np.zeros(len(X))
        for k0 in range(0, len(X), 20000):
            d2 = ((X[k0:k0 + 20000, None, :] - cb[None]) ** 2).sum(-1)
            f[k0:k0 + 20000] = (np.exp(-0.5 * d2 / sig ** 2) * wb[None]).sum(1)
        f = f / max(np.abs(f).max(), 1e-9)
        keep = rp.random(len(X)) < (1 + PA * f) / (1 + PA)
        X, NR = X[keep], NR[keep]
    n = len(X)
    din = scalp.d_inside_mm(X)
    phi_r, th_r, _ = scalp.sph(X)
    x_part = PT["offset_mm"] / 1000 if PT["type"] == "side" else 0.0
    side_x = X[:, 0] - x_part
    side = np.where(side_x >= 0, 1, -1)
    cross = np.zeros(n, bool)
    if PT["type"] != "none" and PT["cross_frac"] > 0:
        fr = np.abs(phi_r) < 90
        y_front = float(X[fr & (din < 3.0), 1].min()) if (fr & (din < 3.0)).any() else float(X[:, 1].min())
        zone = (np.abs(side_x) < PT["cross_mm"] / 1000) & (X[:, 1] < y_front + PT["cross_front_mm"] / 1000) & \
            (th_r < 70)
        p = PT["cross_frac"] * (1 - np.abs(side_x) / max(PT["cross_mm"] / 1000, 1e-9))
        cross = zone & (rng.random(n) < p)
    gside = np.where(cross, -side, side)
    # the comb's frame: the front hairline's y, the part's depth, the whorl (the part's end on the scalp)
    frn = (np.abs(phi_r) < 20) & (din < 4.0)
    y_front = float(np.median(X[frn, 1])) if frn.any() else float(X[:, 1].min())
    HV = scalp.V
    wsel = (np.abs(HV[:, 0] - x_part) < 0.004) & (np.abs(HV[:, 1] - (y_front + PT["depth_mm"] / 1000)) < 0.004)
    whorl = HV[wsel][np.argmax(HV[wsel][:, 2])] if wsel.any() else None
    FLW = dict(FLW, _y_front=y_front, _part_depth=(PT["depth_mm"] / 1000 if PT["type"] != "none" else 0.0),
               _whorl=whorl)
    if FLW.get("drape_front", 0.0) > 0:     # the strands rooted in front of the ears' line drape in front of the shoulders
        d0_, d1_ = FLW.get("drape_phi_deg", [95.0, 120.0])
        FLW["_drape_w"] = 1 - smoothstep(d0_, d1_, np.abs(phi_r))
    kind = np.zeros(n, np.int8)
    edge = din < R["edge_mm"]
    kind[edge] = 1
    short = edge & (din < 3.0) & (rng.random(n) < HL["edge_short_frac"])
    # the sideburns: roots in front of an ear and below its top are short when the hair is tucked behind the ears
    # (the long hair from above covers them; walked long, they climbed over the ear and hung outside it)
    # A comb-path stand-in only: in grow mode (DESIGN_v2: no sideburn cut) those roots grow to their own lengths and
    # the tuck lays them back over the ear with the side section (it forced 889 of 915 roots in front of the ear to
    # short hair on gr18, so no long hair grew below the ear top + 3 mm where the hairline reaches ~20 mm below it).
    # The block draws no random numbers, so skipping it leaves the comb path's stream unchanged.
    # grow.sideburn 'short' (DESIGN_sections 2) takes gr18's short lengths back in grow mode: the reference's hair in
    # front of the ear is short and fine, lying down and back, ending at the ear's front edge; long hair from those
    # roots can reach behind the ear only over the helix, across its face or under the lobe (gr19: 179 / 221 strands
    # on the helix's top band). A stand-in until the preauricular follicles get their own growth cycle.
    sideburn = np.zeros(n, bool)
    sb_short = str((st.get("grow") or {}).get("sideburn", "own")) == "short"
    if FLW["behind_ears"] and (FLW.get("source", "comb") != "grow" or sb_short):
        for e_s, e in scalp.ears.items():
            sideburn |= (e_s * X[:, 0] > 0.04) & (X[:, 1] < e["y_back"] + 0.002) & (X[:, 2] < e["z_top"] + 0.003)
        short |= sideburn
        rep["sideburn_short"] = int(sideburn.sum())
    elif FLW.get("source", "comb") == "grow":
        rep["sideburn_short"] = 0   # kept so reports compare across runs: grow mode has no sideburn cut
    FLW["_sideburn"] = sideburn
    kind[short] = 2
    main = kind == 0
    rep["comb"] = {"y_front": round(y_front, 4), "whorl": None if whorl is None else whorl.round(4).tolist()}
    rep["roots"] = {"strands": int(n), "scalp_cm2": round(scalp_cm2, 1), "per_cm2": round(n / scalp_cm2, 1),
                    "edge_fine": int((kind == 1).sum()), "edge_short": int((kind == 2).sum()),
                    "part_cross": int(cross.sum()), "x_part_mm": x_part * 1000}
    tick("roots %d on %.0f cm2" % (n, scalp_cm2))
    return dict(X=X, NR=NR, scalp_cm2=scalp_cm2, n=n, din=din, phi_r=phi_r, th_r=th_r, x_part=x_part,
                side_x=side_x, side=side, cross=cross, gside=gside, y_front=y_front, whorl=whorl, FLW=FLW,
                kind=kind, edge=edge, short=short, sideburn=sideburn, main=main, PA=PA, scalp_cm2_=scalp_cm2)


def generate(scalp, collider, style, menton_z, log=print, workers=4, head=None, head_cache=None, regrow=False,
             case=None, passes="auto", style_stage=True, from_state=None, save_state=None):
    """a scalp + a style -> (arrays, report). Grow mode with flow.order 'after' only: head - a head.GrownHead or a
    saved head's path, used as given; head_cache - the directory of saved heads (<key16>.npz), looked up by the head's
    key and written to when a head is grown (None: grown in memory, not saved); regrow - grow a new head even when the
    cache has one; case - the case's ids for the head's key (e.g. {"case": "my_case"}); passes - the style stage's hook
    ('auto', 'acts' or a callable: _passes_hook); style_stage False - the rested head finished as it lies
    (finish_head). The comb path and flow.order 'during' take none of them."""
    t0 = time.time()
    st = SY.effective(style)
    seed = int(st["seed"])
    rng = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 7)
    K = int(st["points"])
    R, HL, PT, FLW, LN, CU, GA = (st["roots"], st["hairline"], st["part"], st["flow"], st["length"], st["cut"],
                                  st["gather"])
    if FLW.get("source", "comb") != "grow" and (head is not None or regrow or not style_stage):
        # a grown head is the grow branch's (review 10-31): the comb path would ignore it without a word
        raise ValueError("a grown head (head / regrow / style_stage) needs flow.source 'grow'; this style combs")
    gathered = GA["type"] != "none"
    rep = {"style": st["name"], "seed": seed, "points": K, "gather": GA["type"]}
    tick = lambda msg: log("%s  (%.1f s)" % (msg, time.time() - t0))  # noqa: E731

    # 1. roots, part, kinds
    RP = _roots_part_kinds(scalp, st, rng, rep, tick)
    if FLW.get("source", "comb") == "grow":
        order = str(FLW.get("order", "during"))
        if order == "after":
            return _grow_after(scalp, collider, style, st, menton_z, RP, rep, rng, rng_b, tick, log, t0, head=head,
                               head_cache=head_cache, regrow=regrow, case=case, passes=passes,
                               style_stage=style_stage, from_state=from_state, save_state=save_state)
        if order != "during":
            raise ValueError("flow.order %r: 'during' (the acts while the hair grows) or 'after' (grow, then style)"
                             % order)
        if head is not None or regrow or not style_stage or from_state is not None or save_state is not None:
            raise ValueError("a grown head (head, regrow, finish_head) is flow.order 'after''s; this style grows with "
                             "its acts on (flow.order 'during')")
        return _grow(scalp, collider, style, st, menton_z, RP, rep, rng, rng_b, tick, log, t0)
    (X, NR, n, din, phi_r, th_r, x_part, side_x, gside, cross, FLW, kind, sideburn, main, PA) = (
        RP[k] for k in ("X", "NR", "n", "din", "phi_r", "th_r", "x_part", "side_x", "gside", "cross", "FLW", "kind",
                        "sideburn", "main", "PA"))

    # heights and the start directions
    Hpl, tl = heights(st, X, din, phi_r, th_r, rng)
    DK = float(st["lift"].get("density_k", 0.0))
    if DK > 0:
        # the stack follows the hair: the local root density (roots within 6 mm) over its median, ^k
        from scipy.spatial import cKDTree
        cnt = np.array([len(q) for q in cKDTree(X).query_ball_point(X, 0.006)], float)
        rel = cnt / max(np.median(cnt[din > 10]), 1.0)
        Hpl = Hpl * np.clip(rel, 0.5, 2.0) ** DK
        rep["density"] = {"patch_amp": PA, "density_k": DK, "rel_p5_p50_p95": [round(float(np.percentile(rel, q)), 3) for q in (5, 50, 95)]}
    if gathered:
        sl_ = float(GA["sleek"])
        Hpl = Hpl * (1 - 0.7 * sl_) + 0.0005 * sl_
    FLW["_Hp"], FLW["_Hp_max"] = Hpl, float(np.percentile(Hpl, 95))
    turn = np.radians(rng.normal(0, FLW["turn_deg"], n))
    Lshort = rng.uniform(*HL["edge_short_mm"], n) / 1000
    # the sideburns lie flat, down and back, 5-15 mm (aimed over the ear like the long hair, they stood out beside it)
    Lshort = np.where(sideburn, rng.uniform(0.005, 0.015, n), Lshort)

    # 2. the length targets
    ztip = menton_z - LN["tip_below_menton_mm"] / 1000 + rng.normal(0, LN["tip_sd_mm"] / 1000, n)
    ztip += np.where(rng.random(n) < LN["short_frac"], rng.uniform(*LN["short_mm"], n) / 1000, 0.0)
    ztip -= LN["a_line_mm"] / 1000 * (np.cos(np.radians(phi_r)) + 1) / 2
    if CU["type"] == "face_framing":
        a0, a1 = CU["framing_phi_deg"]
        wfr = (1 - smoothstep(0.0, CU["framing_zone_mm"], din)) * smoothstep(a0 - 5, a0, np.abs(phi_r)) * \
            (1 - smoothstep(a1, a1 + 10, np.abs(phi_r)))
        ztip += CU["framing_rise_mm"] / 1000 * wfr
    layer = np.zeros(n)
    if CU["type"] in ("layered", "face_framing") and CU["crown_layer"] > 0:
        c0, c1 = CU["crown_theta_deg"]
        w_cr = (1 - smoothstep(c0, c1, th_r)) * smoothstep(25.0, 45.0, din) * main
        layer = np.clip(CU["crown_layer"] * w_cr * rng.uniform(0.7, 1.3, n), 0, 0.45)
    Lcap = LN["max_mm"] / 1000
    step = FLW["step_mm"] / 1000
    ears_cfg, corr_cfg = st["ears"], st["corridor"]
    surf = collider

    def comb_loose(P, s, ids, Pv, sd):
        return FL.comb_field(scalp, FLW, ears_cfg, corr_cfg, gside, P, s, ids, Pv, sd)

    tuck = (lambda P: FL.ear_tuck(scalp, ears_cfg, P)) if FLW["behind_ears"] else None

    def start_dirs(comb, ids):
        Xb, Nb = X[ids], NR[ids]
        sd, j = surf.sd(Xb)
        C = comb(Xb, np.zeros(len(ids)), ids, surf.V[j], sd)
        C = C[0] if isinstance(C, tuple) else C
        C = C + np.array([0.0, 0.0, -FLW["gravity"] / max(FLW["comb_strength"], 1e-6)])
        C = C - np.einsum("ij,ij->i", C, Nb)[:, None] * Nb
        C = _unit(C + 1e-9 * np.array([0, 1.0, 0]))
        r_ = turn[ids][:, None]
        C = C * np.cos(r_) + np.cross(Nb, C) * np.sin(r_)
        return C

    BATCH = 6000
    pts = np.empty((n, K, 3), np.float32)
    L_out = np.zeros(n)
    s_tie = np.zeros(n, np.float32)
    viol = np.zeros(n, bool)
    reached = np.ones(n, bool)
    hfn = lambda s, ids: h_of(s, Hpl[ids], tl[ids])  # noqa: E731

    def loose_batch(ids):
        Lw = np.where(kind[ids] == 2, Lshort[ids] + step, Lcap)
        poly, _, _ = FL.walk(surf, X[ids], NR[ids], start_dirs(comb_loose, ids), Lw, step, FLW["stiffness"],
                             FLW["gravity"], FLW["comb_strength"], comb_loose, hfn, constraint=tuck, ids=ids)
        L, has = FL.first_below(poly, step, ztip[ids])
        L = np.where(kind[ids] == 2, np.minimum(Lshort[ids], L), L)
        L = L * (1 - layer[ids])
        L = np.clip(L, 0.012, Lcap)
        return poly, L, has

    tie = None
    PK = float(st["lift"].get("stack_packing", 0.0))
    if PK > 0 and not gathered:
        # THE STACK (10-29): strands do not collide with one another, so the layer's height was a style constant (the
        # plateau) and the hanging mass fell thin. Here the hair's own amount sets it: a first walk deposits each strand's
        # volume (arc x pi r^2) on a 2 mm voxel grid with the skin's area; both smoothed (sigma STACK_SIGMA_MM); the layer's
        # thickness T = hair volume / skin area / packing, read at every skin vertex; with the furthest arc seen there
        # (s_max). The second walk keeps each strand T(j) x (s / s_max(j)) above the skin - the hair that has come furthest
        # (from upstream) lies on top - after the usual rise from the root. More hair, a thicker layer: the volume is the
        # hair's own ("a little bit of extra volume here and there as the hair is in denser patches")
        from scipy.ndimage import gaussian_filter
        vox = 0.002
        lo_ = surf.V.min(0) - 0.06; hi_ = surf.V.max(0) + 0.06
        shp = tuple(np.ceil((hi_ - lo_) / vox).astype(int) + 1)
        HV = np.zeros(shp, np.float32); SM = np.zeros(len(surf.V), np.float32)
        # each place's histogram of the arc the passing hair has travelled (4 mm voxels, NB bins): a strand's place in the
        # stack is its RANK there (the share of the hair under it that has come less far) - g13 used s / s_max, which put a
        # median strand a fifth of the way up and thinned the layer
        NB = 24; s_bin = float(Lcap) / NB
        shp4 = tuple(np.ceil((hi_ - lo_) / 0.004).astype(int) + 1)
        SH = np.zeros(shp4 + (NB,), np.float32)
        r_m = 0.5 * (st["radius"]["root_mm"] + st["radius"]["tip_mm"]) / 1000
        for b0 in range(0, n, BATCH):
            ids = np.arange(b0, min(n, b0 + BATCH))
            poly, L, has = loose_batch(ids)
            sg = np.arange(poly.shape[1]) * step
            ok = sg[None, :] <= L[:, None]
            _, jj = surf.sd(poly)
            q = np.clip(((surf.V[jj[ok]] - lo_) / vox).astype(int), 0, np.array(shp) - 1)
            np.add.at(HV, (q[:, 0], q[:, 1], q[:, 2]), step * np.pi * r_m ** 2)
            np.maximum.at(SM, jj[ok], np.broadcast_to(sg, poly.shape[:2])[ok].astype(np.float32))
            q4 = np.clip(((surf.V[jj[ok]] - lo_) / 0.004).astype(int), 0, np.array(shp4) - 1)
            sb = np.clip((np.broadcast_to(sg, poly.shape[:2])[ok] / s_bin).astype(int), 0, NB - 1)
            np.add.at(SH, (q4[:, 0], q4[:, 1], q4[:, 2], sb), 1.0)
        va = surf.sample_area()                  # each surface sample's share of the skin (m2; field_collider)
        AV = np.zeros(shp, np.float32); qv = np.clip(((surf.V - lo_) / vox).astype(int), 0, np.array(shp) - 1)
        np.add.at(AV, (qv[:, 0], qv[:, 1], qv[:, 2]), va)
        sgm = STACK_SIGMA_MM / 1000 / vox
        HVs, AVs = gaussian_filter(HV, sgm), gaussian_filter(AV, sgm)
        T = (HVs / np.maximum(AVs, 1e-9))[qv[:, 0], qv[:, 1], qv[:, 2]] / PK
        T = np.clip(T, 0.0, STACK_MAX_MM / 1000)
        SMs = np.maximum(SM, step)
        rep["stack"] = {"packing": PK, "T_mm_p50_p90_max_on_hair": [round(float(np.percentile(T[SM > 0], q_) * 1000), 2) for q_ in (50, 90, 100)],
                        "sigma_mm": STACK_SIGMA_MM}
        for b_ in range(NB):
            SH[..., b_] = gaussian_filter(SH[..., b_], 1.0)
        q4v = np.clip(((surf.V - lo_) / 0.004).astype(int), 0, np.array(shp4) - 1)
        Hv = SH[q4v[:, 0], q4v[:, 1], q4v[:, 2]]                                       # (nV, NB)
        CDF = np.cumsum(Hv, 1) / np.maximum(Hv.sum(1, keepdims=True), 1e-9)
        CDF0 = np.c_[np.zeros(len(CDF)), CDF[:, :-1]]
        del SH, Hv

        def rank(s_, j_):
            b_ = np.clip((s_ / s_bin).astype(int), 0, NB - 1); fr_ = np.clip(s_ / s_bin - b_, 0, 1)
            return CDF0[j_, b_] + fr_ * (CDF[j_, b_] - CDF0[j_, b_])
        hfn = lambda s, ids, j: h_of(s, np.maximum(T[j] * rank(s, j), 0.0005), tl[ids])  # noqa: E731
        tick("stack: the layer from the hair's own amount (packing %.2f; T p50 / p90 %s mm)" % (PK, rep["stack"]["T_mm_p50_p90_max_on_hair"][:2]))
    if not gathered:
        for b0 in range(0, n, BATCH):
            ids = np.arange(b0, min(n, b0 + BATCH))
            poly, L, has = loose_batch(ids)
            reached[ids] = has | (kind[ids] == 2)
            pts[ids] = FL.resample_poly(poly, step, L, K).astype(np.float32)
            L_out[ids] = L
            if (b0 // BATCH) % 5 == 0:
                tick("  walk %d / %d" % (b0 // BATCH + 1, (n + BATCH - 1) // BATCH))
    else:
        # the loose length of every strand (her hair, gathered)
        if LN["gathered_from"] == "loose":
            Lnat = np.empty(n)
            for b0 in range(0, n, BATCH):
                ids = np.arange(b0, min(n, b0 + BATCH))
                _, L, _ = loose_batch(ids)
                Lnat[ids] = L
            tick("  loose lengths p10/50/90 %s mm" % np.percentile(Lnat * 1000, [10, 50, 90]).round(0).tolist())
        else:
            Lnat = rng.uniform(*LN["fixed_mm"], n) / 1000
        tie_r = GA["tie_r_mm"] / 1000
        G, nG, T, A, U, W = FL.tie_frame(scalp, GA["phi_deg"], GA["theta_deg"], tie_r, GA["tail_dir"])
        conv = GA["converge_mm"] / 1000
        # each strand's place in the tie: its angle about the axis (from the root), radius by its distance to G
        dX = X - T
        theta_i = np.arctan2(dX @ W, dX @ U)
        dG = np.linalg.norm(X - G, axis=1)
        rank = np.argsort(np.argsort(dG)) / max(n - 1, 1)
        rho = np.sqrt(np.clip(rank, 0, 1)) * 0.92 + 0.04

        def comb_g(P, s, ids, Pv, sd):
            return FL.comb_gather(scalp, G, GA["pull"], P, s, ids, Pv, sd)

        def stop_g(P, s, ids):
            return np.linalg.norm(P - G, axis=1) < conv

        # the tail's centre line, shared
        L_tail_est = np.clip(Lnat - dG * 1.25 - conv, 0.0, None)
        Lt_max = float(np.percentile(L_tail_est, 99.5)) + 0.03
        width = GA["tie_width_mm"] / 1000
        spread = GA["tail_spread_mm"] / 1000
        if GA["type"] == "ponytail":
            def rfun(t):
                t = np.asarray(t, np.float64)
                r = tie_r + (spread - tie_r) * smoothstep(width, width + 0.06, t)
                return r * (1 - GA["tail_taper"] * smoothstep(width + 0.05, Lt_max, t))
            spine = FL.spine_walk(surf, T, A, Lt_max, step, GA["tail_stiffness"], 1.0, lambda t: 0.8 * float(rfun(t)))
            twist = GA["twist_mm"] / 1000
        else:
            Rb = GA["bun_r_mm"] / 1000
            turns = float(np.clip(np.percentile(L_tail_est, 85) / (2 * np.pi * Rb), 0.8, GA["bun_turns"]))
            rope0 = tie_r * 0.85

            def rfun(t):
                t = np.asarray(t, np.float64)
                return rope0 * (1 - 0.45 * smoothstep(0.0, Lt_max, t))
            spine = FL.coil_spine(T, A, U, W, Lt_max, step, tie_r * 0.3, Rb, turns, GA["bun_rise_mm"] / 1000,
                                  GA["bun_tuck_mm"] / 1000)
            twist = GA["bun_twist_mm"] / 1000
            rep["bun_turns"] = round(turns, 2)
        Ts, Us, Ws = FL.rmf(spine, U)
        # the strands' angles carried into the spine's frame at the tie
        rep["tie"] = {"G": G.round(4).tolist(), "T": T.round(4).tolist(), "axis": A.round(3).tolist(),
                      "tail_len_mm": round(Lt_max * 1000, 1), "tie_r_mm": GA["tie_r_mm"]}
        tie = torus(T + A * width * 0.5, A, U, W, tie_r + GA["tie_tube_mm"] / 1000, GA["tie_tube_mm"] / 1000,
                    width=width)
        Lmax_walk = min(Lcap, float(dG.max()) * 2.2 + 0.05)
        n_ext = [0]
        # loose tendrils (10-29 night): a share of the temples' front roots left out of the tie, hanging beside the face
        tend = np.zeros(n, bool)
        if GA.get("tendril_frac", 0.0) > 0:
            rt_ = np.random.default_rng(seed + 71)
            a0, a1 = GA["tendril_phi_deg"]
            tz = (din < GA["tendril_zone_mm"]) & (np.abs(phi_r) >= a0) & (np.abs(phi_r) <= a1) & (kind != 2)
            tend = tz & (rt_.random(n) < GA["tendril_frac"])
            L_tend = rt_.uniform(*GA["tendril_len_mm"], n) / 1000
            FLW_t = dict(FLW, behind_ears=False, face_back=GA.get("tendril_back", 0.3), back=0.0, away_from_part=0.0,
                         _sideburn=None)

            def comb_t(P, s, ids, Pv, sd):
                return FL.comb_field(scalp, FLW_t, ears_cfg, None, gside, P, s, ids, Pv, sd)
            rep["tendrils"] = int(tend.sum())
        for b0 in range(0, n, BATCH):
            ids = np.arange(b0, min(n, b0 + BATCH))
            m = len(ids)
            Lw = np.full(m, Lmax_walk)
            poly, stop_at, _ = FL.walk(surf, X[ids], NR[ids], start_dirs(comb_g, ids), Lw, step, FLW["stiffness"],
                                       FLW["gravity"] * 0.3, GA["pull"], comb_g, hfn, stop=stop_g, ids=ids)
            arrived = stop_at < poly.shape[1] - 1
            ls = stop_at * step
            r_ = np.arange(m)
            Pa = poly[r_, stop_at]
            da = _unit(poly[r_, stop_at] - poly[r_, np.maximum(stop_at - 1, 0)] + 1e-12)
            D = T + (rho[ids] * tie_r)[:, None] * (np.cos(theta_i[ids])[:, None] * U + np.sin(theta_i[ids])[:, None] * W)
            B, Lb = FL.bridge(Pa, da, D, A, step)
            # tail: each strand along the spine from the tie
            Lt = np.maximum(Lnat[ids] - ls - Lb, GA["min_tail_mm"] / 1000)
            n_ext[0] += int((arrived & (Lnat[ids] - ls - Lb < GA["min_tail_mm"] / 1000)).sum())
            tail = FL.place(spine, Ts, Us, Ws, step, 0.0, rho[ids], theta_i[ids], rfun, twist, Lt)
            # scalp part up to the stop (the rest repeats the stop point), the bridge, the tail
            Sw = poly.shape[1]
            kk = np.minimum(np.arange(Sw)[None, :], stop_at[:, None])
            scal = np.take_along_axis(poly, kk[..., None], axis=1)
            full = np.concatenate([scal, B, tail], 1)
            kk2 = np.minimum(np.arange(Sw)[None, :], np.ceil(np.minimum(Lnat[ids], Lmax_walk) / step).astype(np.int64)[:, None])
            loose = np.take_along_axis(poly, kk2[..., None], axis=1)
            full = np.where(arrived[:, None, None], full, np.concatenate(
                [loose, np.repeat(loose[:, -1:], B.shape[1] + tail.shape[1], 1)], 1))
            P32, Lf = resample_arc(full, K)
            pts[ids] = P32.astype(np.float32)
            L_out[ids] = Lf
            s_tie[ids] = np.where(arrived, ls + Lb, Lf)
            reached[ids] = arrived
            if (b0 // BATCH) % 5 == 0:
                tick("  gather walk %d / %d" % (b0 // BATCH + 1, (n + BATCH - 1) // BATCH))
        ti_ = np.nonzero(tend)[0]
        for b0 in range(0, len(ti_), BATCH):
            ids = ti_[b0:b0 + BATCH]
            Lw = np.minimum(L_tend[ids], Lcap)
            poly, _, _ = FL.walk(surf, X[ids], NR[ids], start_dirs(comb_t, ids), Lw, step, FLW["stiffness"],
                                 FLW["gravity"], FLW["comb_strength"], comb_t, hfn, ids=ids)
            pts[ids] = FL.resample_poly(poly, step, Lw, K).astype(np.float32)
            L_out[ids] = Lw
            s_tie[ids] = 0.0
            reached[ids] = False
        rep["gather_arrived"] = int(reached.sum())
        rep["gather_lengthened_to_min_tail"] = n_ext[0]
    tick("flow done")

    # the hair's own volume (volume.py): strands relaxed until they hold no more than the packing's share of the space
    VO = st.get("volume", {})
    if float(VO.get("packing", 0.0)) > 0 and not gathered:
        from . import volume as VL
        RD = st["radius"]
        es = float(np.mean(RD["edge_scale"]))
        rr = np.where(kind > 0, es, 1.0) * RD["root_mm"] / 1000
        rt = np.where(kind > 0, es, 1.0) * RD["tip_mm"] / 1000
        pts, rep["volume"] = VL.relax(pts, rr, rt, surf, VO, log=lambda m: tick(m), key=np.c_[X, gside * 0.06])
        tick("volume: %.0f cm3 of fibre fills %.0f cm3 (packing %.3f)" % (rep["volume"]["fibre_cm3"],
                                                                         rep["volume"]["hair_volume_cm3"], VO["packing"]))

    # the corridor: strands over the face skin are dropped (counted)
    for b0 in range(0, n, 20000):
        sl = slice(b0, min(n, b0 + 20000))
        Pb = pts[sl].astype(np.float64)
        sd, j = surf.sd(Pb)
        vj = surf.V[j].reshape(-1, 3)
        # only the head's vertices are in the scalp's frame: points over the body are never over the face
        fm = FL.corridor_mask(scalp, corr_cfg, vj, sd.reshape(-1)).reshape(Pb.shape[:2])
        viol[sl] = fm[:, 1:].any(1)
    rep["corridor_dropped"] = int(viol.sum())

    keep = ~viol
    idx = np.nonzero(keep)[0]
    X, NR, din, phi_r, th_r, kind, gside, cross, side_x = (a[idx] for a in (X, NR, din, phi_r, th_r, kind, gside, cross,
                                                                             side_x))
    P = pts[idx].astype(np.float64)
    del pts
    L_out, s_tie, reached = L_out[idx], s_tie[idx], reached[idx]
    n = len(idx)
    main = kind == 0
    rep["length_mm_main_p10_50_90"] = np.percentile(L_out[main] * 1000, [10, 50, 90]).round(1).tolist()
    rep["tip_z_main_p10_50_90"] = np.percentile(P[main][:, -1, 2], [10, 50, 90]).round(4).tolist()
    rep["reached_tip_plane"] = int(reached.sum()) if not gathered else None

    # bevel: the ends turn under toward the neck
    if CU["turn_under_mm"] > 0:
        s = FO.arc(P)
        Lb_ = s[:, -1:]
        f = np.clip((s - (Lb_ - 0.03)) / 0.03, 0, 1) ** 2
        tgt = np.stack([np.zeros_like(P[..., 0]), np.full_like(P[..., 1], scalp.c[1] + 0.02), P[..., 2]], -1)
        dvec = tgt - P
        dvec[..., 2] = 0
        P += (CU["turn_under_mm"] / 1000 * f)[..., None] * _unit(dvec)

    # 4. form
    st["_part_zone"] = ((X[:, 1] - FLW["_y_front"] < PT["depth_mm"] / 1000) & (th_r < 75)).astype(np.float64) \
        if (PT["type"] != "none" and not gathered) else np.zeros(n)
    if gathered:        # pulled back: no part, so no side separation and no part-free clumping
        st["clump"]["part_free"] = False
    lk = FO.locks(X, gside, main, din, st, rng, rng_b)
    rep["locks"] = lk["stats"]
    tick("locks %d, sub-locks %d, bundles %d" % (lk["nl"], lk["nsub"] - 1, len(lk["BC"])))
    P_pre = P.astype(np.float32)
    P, fly, fst = FO.shape(P, X, kind, din, phi_r, th_r, side_x, s_tie.astype(np.float64), scalp, st, lk, rng, rng_b,
                           log=lambda m: tick(m))
    rep["form"] = fst
    # constraints: behind the ears (never further than the walk left it), the corridor (revert)
    if FLW["behind_ears"]:
        for b0 in range(0, n, 20000):
            sl = slice(b0, min(n, b0 + 20000))
            Pb = P[sl]
            Q = P_pre[sl].astype(np.float64)
            for e_s, e in scalp.ears.items():
                zc = Pb[..., 2]
                wz = smoothstep(e["z_bot"] - 0.020, e["z_bot"] - 0.002, zc) * \
                    (1 - smoothstep(e["z_top"] + 0.04, e["z_top"] + 0.06, zc))
                xl = ears_cfg["x_max_mm"] / 1000 + (ears_cfg["x_top_mm"] - ears_cfg["x_max_mm"]) / 1000 * \
                    smoothstep(e["z_top"] + 0.004, e["z_top"] + 0.04, zc)
                xl = np.maximum(xl, e_s * Q[..., 0])
                behind = Pb[..., 1] > e["y_back"] + 0.001
                dx = np.maximum(0, e_s * Pb[..., 0] - xl) * wz * behind
                Pb[..., 0] -= e_s * dx
            P[sl] = Pb
    nrev = 0
    for b0 in range(0, n, 20000):
        sl = slice(b0, min(n, b0 + 20000))
        Pb = P[sl]
        sd, j = surf.sd(Pb)
        fm = FL.corridor_mask(scalp, corr_cfg, surf.V[j].reshape(-1, 3), sd.reshape(-1)).reshape(Pb.shape[:2])
        v = fm[:, 1:].any(1)
        nrev += int(v.sum())
        Pb[v] = P_pre[sl][v]
        P[sl] = Pb
    rep["corridor_reverted_after_form"] = nrev
    del P_pre

    # 5. baby hairs
    Bb = HL["baby"]
    nb_in_band = int(((din > Bb["band_mm"][0]) & (din < Bb["band_mm"][1]) & (np.abs(phi_r) < Bb["phi_deg"])).sum())
    n_baby = int(Bb["frac"] * nb_in_band) + int(Bb["extra"])
    baby = None
    if n_baby > 0:
        Xb, Nb, _ = scalp.sample_roots(n_baby * 3, 0, 1, rng, band=tuple(Bb["band_mm"]))
        phb, _, _ = scalp.sph(Xb)
        okb = np.nonzero(np.abs(phb) < Bb["phi_deg"])[0][:n_baby]
        Xb, Nb = Xb[okb], Nb[okb]
        m = len(Xb)
        sd, j = surf.sd(Xb)
        gs = np.where(Xb[:, 0] - x_part >= 0, 1, -1)
        C = FL.comb_field(scalp, dict(FLW, _sideburn=None, _Hp=None), ears_cfg, None, gs, Xb, np.zeros(m), np.arange(m),
                          surf.V[j], sd)[0]
        C = _unit(C - np.einsum("ij,ij->i", C, Nb)[:, None] * Nb + 1e-9)
        # baby hairs grow out of the hairline, toward the face (away from the crown), not combed into the hair
        out = Xb - (scalp.c + np.array([0.0, 0.0, 0.12]))
        out = _unit(out - np.einsum("ij,ij->i", out, Nb)[:, None] * Nb + 1e-9)
        C = _unit(Bb["outward"] * out + (1 - Bb["outward"]) * C)
        rot = np.radians(rng.normal(0, 35.0, m))[:, None]
        C = C * np.cos(rot) + np.cross(Nb, C) * np.sin(rot)
        lift = np.radians(rng.uniform(*Bb["lift_deg"], m))[:, None]
        d0 = np.cos(lift) * C + np.sin(lift) * Nb
        side_v = _unit(np.cross(Nb, C))
        bend_ax = _unit(np.cos(rng.uniform(0, np.pi, m))[:, None] * side_v + np.sin(rng.uniform(0, np.pi, m))[:, None]
                        * np.cross(d0, side_v))
        Lb = rng.uniform(*Bb["length_mm"], m) / 1000
        tot = Bb["curl"] * 2 * np.pi * rng.uniform(0.5, 1.5, m) * rng.choice([-1.0, 1.0], m)
        u = np.linspace(0, 1, K)
        # an arc: direction turns about bend_ax by tot * u
        ang = tot[:, None] * u[None, :]
        dd = d0[:, None, :] * np.cos(ang)[..., None] + np.cross(bend_ax, d0)[:, None, :] * np.sin(ang)[..., None]
        seg = dd * (Lb / (K - 1))[:, None, None]
        Pbaby = Xb[:, None, :] + np.concatenate([np.zeros((m, 1, 3)), np.cumsum(seg[:, :-1], 1)], 1)
        Pbaby, _ = surf.push_out(Pbaby, 0.00015, iters=3, exact=True)
        baby = dict(P=Pbaby, X=Xb, m=m)
        rep["baby_hairs"] = int(m)
        tick("baby hairs %d" % m)

    # 6. colour
    melp, red, cst = FO.melanin_field(P, X, kind, din, phi_r, scalp, st, lk, rng, rng_b)
    rep["colour"] = cst
    Cc = st["colour"]
    R_ROOT = st["radius"]["root_mm"] / 1000
    r_root = np.full(n, R_ROOT) * rng.uniform(1 - st["radius"]["jitter"], 1 + st["radius"]["jitter"], n)
    r_tip = np.full(n, st["radius"]["tip_mm"] / 1000) * rng.uniform(1 - st["radius"]["jitter"], 1 + st["radius"]["jitter"], n)
    fk = rng.uniform(*st["radius"]["edge_scale"], n)
    fine = kind > 0
    r_root = np.where(fine, r_root * fk, r_root)
    r_tip = np.where(fine, r_tip * fk, r_tip)
    lock_i, sub_i = lk["li"].astype(np.int32), lk["si"].astype(np.int32)
    if baby is not None:
        m = baby["m"]
        P = np.concatenate([P, baby["P"]])
        mb = np.clip(Cc["melanin"] - Cc["front_lighter"] - Bb["lighter"] + rng.normal(0, Cc["strand_sd"], m), 0.1, 0.95)
        melp = np.concatenate([melp, np.repeat(mb[:, None], K, 1).astype(np.float32)])
        red = np.concatenate([red, np.clip(Cc["redness"] + rng.normal(0, Cc["redness_sd"], m), 0, 0.9).astype(np.float32)])
        rb = R_ROOT * rng.uniform(*Bb["radius_scale"], m)
        r_root = np.concatenate([r_root, rb])
        r_tip = np.concatenate([r_tip, rb * 0.45])
        kind = np.concatenate([kind, np.full(m, 3, np.int8)])
        din = np.concatenate([din, scalp.d_inside_mm(baby["X"])])
        cross = np.concatenate([cross, np.zeros(m, bool)])
        fly = np.concatenate([fly, np.zeros(m, bool)])
        s_tie = np.concatenate([s_tie, np.zeros(m, np.float32)])
        lock_i = np.concatenate([lock_i, np.full(m, -1, np.int32)])
        sub_i = np.concatenate([sub_i, np.full(m, -1, np.int32)])
    n = len(P)

    # 7. collision: pushed out of the body (exact distance), then reported
    tl12 = np.tan(np.radians(12.0))
    moved = 0
    for b0 in range(0, n, 8000):
        sl = slice(b0, min(n, b0 + 8000))
        s = FO.arc(P[sl])
        hmin = np.clip(s * tl12, 0.0002, 0.0015)
        hmin = np.where(kind[sl][:, None] == 3, 0.0001, hmin)
        P[sl], mv = surf.push_out(P[sl], hmin, iters=4, exact=True)
        moved += mv
    rep["collision_pushed_points"] = int(moved)
    tick("collision push")
    col, _ = surf.report(P)
    rep["collision"] = col
    tick("collision report: %d points inside, min clearance %.3f mm" % (col["points_inside"], col["min_clearance_mm"]))

    sig = CO.sigma(melp, red[:, None], Cc["ash"]).astype(np.float16)
    out = dict(pts=P.astype(np.float32), r_root=r_root.astype(np.float32), r_tip=r_tip.astype(np.float32), kind=kind,
               melp=melp, mel=melp.mean(1).astype(np.float32), red=red, mel_root=np.zeros(n, np.float32), sigma=sig,
               lock=lock_i, sub=sub_i, fly=fly, cross=cross, din=din.astype(np.float32), s_tie=s_tie,
               grid=scalp.grid, thb=scalp.thb, centre=scalp.c,
               style_json=json.dumps({k: v for k, v in style.items() if not k.startswith("_")}),
               material_json=json.dumps(st["material"]))
    if tie is not None:
        out["tie_torus"] = tie
    L_all = FO.arc(P)[:, -1] * 1000
    rep.update(strands=int(n), points_total=int(n * K), kinds={str(k): int((kind == k).sum()) for k in range(4)},
               length_mm_all_p10_50_90=np.percentile(L_all, [10, 50, 90]).round(1).tolist(),
               material=st["material"], ash=Cc["ash"],
               sigma_crown_mean=CO.sigma(np.float64(Cc["melanin"]), Cc["redness"], Cc["ash"]).round(4).tolist(),
               seconds=round(time.time() - t0, 1), peak_ram_mb=peak_mb())
    return out, rep


# ---------------------------------------------------------------- grow mode (DESIGN_v2 4.2-4.9, section 6)
def _keep_wave(lam_sec, st, fb):
    """the window (m, per section) over which the locks' pack moves only the mean lines: the waves block's wavelength
    when it sets one, else the hair type's coil turn (fibre.hair_type turn_m) - so the pack keeps each curl"""
    from . import settle as SE
    lam_sec = np.asarray(lam_sec, np.float64)
    if float((st.get("waves") or {}).get("wavelength_mm", 0.0) or 0.0) > 0 or not SE.helix_on(fb):
        return lam_sec
    return np.full(lam_sec.shape, float(fb.hair["turn_m"]))


def _ear_boxes(scalp, margin_mm=15.0, inner_mm=10.0):
    """fine SDF crops around the ears (m): the ear's box from scalp.ears widened by margin_mm, from inner_mm inside
    its outer rim; the ear rim is thin (~2 mm radius) and a 3 mm lattice rounds it (grid.py notes)"""
    m = margin_mm / 1000.0
    out = []
    for e_s, e in scalp.ears.items():
        xo = float(e.get("x_out", 0.09))
        xa, xb = xo - inner_mm / 1000.0 - m, xo + m
        lo = np.array([min(e_s * xa, e_s * xb), e.get("y_front", e["y_back"] - 0.055) - m, e["z_bot"] - m])
        hi = np.array([max(e_s * xa, e_s * xb), e["y_back"] + m, e["z_top"] + m])
        out.append((lo, hi))
    return out


def _real_locks(lk, il, gid):
    """form.locks' membership with the long children il carried by their REAL lock gid (il,): the ringlet they coil in
    (children_mode 'ringlet') or the group the gather packed them into (locks.causes). form.locks' ids are Poisson
    cells of the ROOTS: at mid-length two hairs within 1.5 mm share one only 22 % of the time on gr18, 36 % on the wavy
    head, 45 % on the ringlet head (95 % at the roots; review of 10-31, buildHS3/README) - so a colour per lock and the
    saved lock id followed root patches, not the ribbons or ringlets. The real locks take ids 0 .. m-1 (one sub-lock
    each); every other hair keeps its Poisson lock and sub-lock after them; CL (the strand at each lock's centre, read
    for the balayage's framing zone) is a lock's first member."""
    u, inv = np.unique(np.asarray(gid), return_inverse=True)
    m = len(u)
    li = np.asarray(lk["li"], np.int64) + m
    si = np.asarray(lk["si"], np.int64) + m
    li[il] = inv
    si[il] = inv
    first = np.empty(m, np.int64)
    first[inv[::-1]] = np.asarray(il)[::-1]
    return dict(lk, li=li, si=si, nl=int(lk["nl"]) + m, nsub=int(lk["nsub"]) + m,
                CL=np.concatenate([first, np.asarray(lk["CL"], np.int64)]), real_locks=m)


def _tip_rule(st, menton_z, phi_r, din, rng):
    """each hair's cut plane (z, m): the length.* plane with its spread, the A-line tilt and the face-framing rise -
    the scissors' line on hair in its natural fall (hair_styling.md 7: perimeter, face frame)"""
    LN, CU = st["length"], st["cut"]
    n = len(phi_r)
    ztip = menton_z - LN["tip_below_menton_mm"] / 1000 + rng.normal(0, LN["tip_sd_mm"] / 1000, n)
    ztip -= LN["a_line_mm"] / 1000 * (np.cos(np.radians(phi_r)) + 1) / 2
    if CU["type"] == "face_framing":
        a0, a1 = CU["framing_phi_deg"]
        wfr = (1 - smoothstep(0.0, CU["framing_zone_mm"], din)) * smoothstep(a0 - 5, a0, np.abs(phi_r)) * \
            (1 - smoothstep(a1, a1 + 10, np.abs(phi_r)))
        ztip += CU["framing_rise_mm"] / 1000 * wfr
    return ztip


def _cut_plan(st, menton_z, phi_r, th_r, din, main, seed):
    """the scissors' plan, read on hair in its natural fall: each root's cut plane (_tip_rule, its own stream seed +
    233) and its crown layer fraction (cut.crown_layer: the over-directed crown sections cut shorter, seed + 239).
    One copy for the legacy grow and the grown head (head.py), so the head is cut exactly as cut.taken 'fall' cuts.
    -> (ztip (n,) m, layer (n,))"""
    ztip = _tip_rule(st, menton_z, phi_r, din, np.random.default_rng(seed + 233))
    CU = st["cut"]
    n = len(phi_r)
    layer = np.zeros(n)
    if CU["type"] in ("layered", "face_framing") and CU["crown_layer"] > 0:
        c0, c1 = CU["crown_theta_deg"]
        w_cr = (1 - smoothstep(c0, c1, th_r)) * smoothstep(25.0, 45.0, din) * main
        layer = np.clip(CU["crown_layer"] * w_cr * np.random.default_rng(seed + 239).uniform(0.7, 1.3, n), 0, 0.45)
    return ztip, layer


def _cut(rods, ztip, layer, cap):
    """the cut on the settled guides (DESIGN_v2 4.7): each member ends where its guide's settled curve first goes
    below the member's cut plane (arc, linear between nodes), shortened by its layer fraction (the over-directed
    crown sections cut shorter), never longer than it grew (cap, m). -> (cut lengths (M,) m, reached the plane)"""
    P = rods.positions()
    A = rods.node_arcs()
    g = rods.mg
    Lg = A[g, -1]
    out_L = np.empty(len(g))
    has = np.zeros(len(g), bool)
    for c0 in range(0, len(g), 20000):                               # members x nodes in chunks
        sl = slice(c0, c0 + 20000)
        gg = g[sl]
        z = P[gg, :, 2]
        below = z <= ztip[sl, None]
        h = below[:, 1:].any(1)
        j = np.where(h, np.argmax(below[:, 1:], 1) + 1, 1)
        r = np.arange(len(gg))
        z0, z1 = z[r, j - 1], z[r, j]
        a0, a1 = A[gg, j - 1], A[gg, j]
        w = np.clip((z0 - ztip[sl]) / np.maximum(z0 - z1, 1e-12), 0, 1)
        out_L[sl] = np.where(h, a0 + w * (a1 - a0), np.maximum(Lg[sl], cap[sl]))
        has[sl] = h
    Lc = out_L * (1 - layer)
    return np.clip(np.minimum(Lc, cap), 0.004, None), has


def _short_out_by_region(cano, body, Pg, rods, Ps, X, L, phi, th, din, dx_part, zone, gcfg):
    """how many of the short hairs' points stand out of the long hair's layer (more than the layer's fuzzy onset,
    grow.canopy_onset_mm, above its outer surface), by region of the scalp: the halo the canopy exists to remove.
    Measured against the same layer whether or not the canopy loaded them (canopy off: one is built for the measure).
    Regions, in this order of precedence: hairline (din < 15 mm from the hairline), part (within 10 mm of the part
    line in the part's zone), crown (theta < 35 deg from the top), back (|phi| > 120 deg), sides (the rest)."""
    from . import children as CH
    dist = CH._distance(body)
    if dist is None or len(X) == 0:
        return {}
    c = cano if cano is not None else CH.build_canopy((Pg, rods.W, rods.V, rods.l), X, L, dist, gcfg)
    n, m = Ps.shape[:2]
    pc, _ = CH.canopy_depth(c, dist, np.asarray(Ps[:, 1:], np.float64).reshape(-1, 3))
    out = (pc.reshape(n, m - 1) > float(gcfg["canopy_onset_mm"]) / 1000.0)
    reg = np.full(n, "sides", dtype=object)
    reg[np.abs(phi) > 120] = "back"
    reg[th < 35] = "crown"
    reg[(zone > 0) & (np.abs(dx_part) < 0.010)] = "part"
    reg[din < 15] = "hairline"
    res = {"all": {"hairs": int(n), "points_out_share": round(float(out.mean()), 4),
                   "hairs_out_share": round(float(out.any(1).mean()), 4)}}
    for r in ("hairline", "part", "crown", "sides", "back"):
        k = reg == r
        if k.any():
            res[r] = {"hairs": int(k.sum()), "points_out_share": round(float(out[k].mean()), 4),
                      "hairs_out_share": round(float(out[k].any(1).mean()), 4)}
    return res


def _prelude(scalp, collider, st, RP, rep, tick, cache_dir=None):
    """everything the grow branch computes before growth, from the roots RP (_roots_part_kinds): the follicles (d0,
    the whorl), the fibre fb, the grown lengths mL and radius scales ms, the part sides and the children's key, the
    guides gi and their members mg, the body grid, the Rods and their rest waves. Shared by _grow and the grown head
    (head.py: grown once, cached, styled after), so both start from the same rods; every stream is its own seeded RNG
    (nothing here draws from the legacy rng), and no act is built here - the acts belong to a style, the head has none.
    The grid cache directory is cache_dir or $HAIR_GRID_CACHE (none: the 3 mm body grid is built, ~10 s).
    -> dict: T (stage seconds), d0, Wh, fb, mL, ms, side, x_part, y_front, zone, gcfg, gwarn, key, gi, mg, counts, f,
    swarn, body, rods, wv_rods, ph_all, sec_all, ph_sec, lam_sec, tilt_sec (None without rod waves), and fresh_rods():
    a new Rods with its rest waves built exactly as rods was (the natural fall's second growth, an instantiated head)"""
    import os
    from . import actions as AC
    from . import children as CH
    from . import fibre as FB
    from . import follicle as FOL
    from . import grid as GR
    from . import settle as SE
    seed = int(st["seed"])
    K = int(st["points"])
    X, NR, n, din, phi_r, th_r, kind = (RP[k] for k in ("X", "NR", "n", "din", "phi_r", "th_r", "kind"))
    HL, PT = st["hairline"], st["part"]
    surf = collider
    stg = time.time
    T = {}

    # follicles: growth direction and exit angle per root (own RNG: the legacy stream is untouched)
    t = stg()
    _, alpha, d0, Wh, frep = FOL.field(scalp, X, NR, din, phi_r, th_r, st, np.random.default_rng(seed + 229))
    rep["follicle"] = {k: frep[k] for k in ("whorl", "alpha_p1_10_50_90_99", "regions", "warnings") if k in frep}
    # fibre and grown lengths (the growth cycle), radius per hair (jitter, fine edge hairs)
    # the whole oil block (not only its amount): with fibre.adhesion on the fibre gets the oil's liquid bridges, whose
    # load joins the settle's friction (fibre.Fibre.adhesive_load) - one oil state for the locks and the friction
    fb = FB.props(st.get("fibre"), st["radius"], dict(st["oil"]))
    mL = FB.cycle_lengths(n, st.get("fibre"), np.random.default_rng(seed + 223), max_mm=float(st["length"]["max_mm"]))
    rr = np.random.default_rng(seed + 227)
    jit = float(st["radius"]["jitter"])
    ms = rr.uniform(1 - jit, 1 + jit, n)
    es = st["radius"]["edge_scale"]
    ms = np.where(kind > 0, ms * rr.uniform(es[0], es[1], n), ms)
    # the short edge hairs at the hairline (hairline.edge_short_*) are short by nature, not by a cut
    Lshort = rr.uniform(*HL["edge_short_mm"], n) / 1000
    mL = np.where(kind == 2, np.minimum(mL, Lshort), mL)
    T["follicle_fibre"] = stg() - t

    # part sides (actions.part_side: the faint line's crossing roots flipped) and the children's key
    t = stg()
    side, x_part = AC.part_side(scalp, X, st, np.random.default_rng(seed + 307))
    frn = (np.abs(phi_r) < 20) & (din < 4.0)
    y_front = float(np.median(X[frn, 1])) if frn.any() else float(X[:, 1].min())
    zone = ((X[:, 1] - y_front < PT["depth_mm"] / 1000) & (th_r < 75)).astype(np.float64) \
        if PT["type"] != "none" else np.zeros(n)
    gcfg = CH.settings(st)
    gcfg, gwarn = gcfg if isinstance(gcfg, tuple) else (gcfg, [])          # children.py's settings: dict (or with warnings)
    gcfg = dict(gcfg, children_points=K)
    key = CH.part_key(X, side, zone, gcfg["part_sep_mm"])
    gi, _ = CH.guides(X, key, int(st["grow"]["guides"]), np.random.default_rng(seed + 211), log=None)
    mg, counts, _ = CH.members(key[gi], key)
    T["guides"] = stg() - t
    tick("grow: %d roots, %d guides, %.0f hairs per guide" % (n, len(gi), n / max(len(gi), 1)))

    # the body and the rods
    t = stg()
    f, swarn = SE.settings(st)
    own = set(CH.CHILDREN_DEFAULTS) | set(CH.FU_DEFAULTS) | {"canopy"}                 # children.py / this branch read those
    swarn = [w for w in swarn if w.split("grow.")[-1] not in own]
    cache_dir = cache_dir or os.environ.get("HAIR_GRID_CACHE") or None
    body = GR.SDFGrid(surf, f["grid_mm"], cache_dir=cache_dir, fine=_ear_boxes(scalp), log=lambda m: None)
    # the natural wave in the rods (grow.rod_waves, DESIGN_v2 5 waves phase 2): one drying-section phase field over ALL
    # the roots, so a guide and the children rooted beside it in its section share the wave's step; the guides take
    # theirs here (each guide its own jitter about the section's), the children read theirs from the same field below
    # the hair type's rest helix (fibre.type, fibre.hair_type) rides the same path: the section phase, the in-section
    # child blend and the pack that moves only the mean lines (window: the coil's turn when no waves block sets one)
    wv_rods = bool(f.get("rod_waves", False)) or SE.helix_on(fb)
    ph_all = sec_all = ph_sec = lam_sec = tilt_sec = None
    if wv_rods:
        lcs_w = FO._cfg(st, "locks", FO.LOCK_CAUSE_DEFAULTS)
        ph_all, sec_all, ph_sec = FO.section_phase(X, lcs_w["section_mm"], seed, lcs_w["phase_jitter"], parts=True)
        if float(lcs_w.get("phase_field_mm", 0.0)) > 0:
            # 11-12 (n2d): the phase from a coherent field over the roots; the sections keep the rest (FO.phase_field)
            ph_all = FO.phase_field(X, lcs_w["phase_field_mm"], seed, lcs_w["phase_jitter"])
        # one wave train per section: its wavelength and plane shared by the guides and the children dried in it
        lam_sec, tilt_sec = SE.section_wave(len(ph_sec), st.get("waves") or {}, seed)

    def fresh_rods():
        """the guides as rods, uncut and ungrown (every segment along d0), with their rest waves - the one recipe for
        the styled rods, the natural fall's and an instantiated head's (rest_waves draws from its own seeded stream)"""
        r = SE.Rods(X[gi], NR[gi], d0[gi], fb, mg, mL, ms, seg_root_mm=f["seg_root_mm"], seg_growth=f["seg_growth"],
                    seg_max_mm=f["seg_max_mm"], sample_mm=f["sample_mm"], min_tail_mm=f["min_tail_mm"])
        if wv_rods:
            SE.rest_waves(r, st.get("waves") or {}, seed, phase=ph_all[gi], random_mm=float(f["wave_random_mm"]),
                          lam=lam_sec[sec_all[gi]], tilt=tilt_sec[sec_all[gi]])
        return r
    rods = fresh_rods()
    T["body_rods"] = stg() - t
    tick("grow: body grid %s (from cache %s), rods K %d" % (body.report.get("shape"), body.report.get("from_cache"),
                                                             rods.K))
    return dict(T=T, d0=d0, Wh=Wh, fb=fb, mL=mL, ms=ms, side=side, x_part=x_part, y_front=y_front, zone=zone, gcfg=gcfg,
                gwarn=gwarn, key=key, gi=gi, mg=mg, counts=counts, f=f, swarn=swarn, body=body, rods=rods,
                wv_rods=wv_rods, ph_all=ph_all, sec_all=sec_all, ph_sec=ph_sec, lam_sec=lam_sec, tilt_sec=tilt_sec,
                fresh_rods=fresh_rods)


def _grow(scalp, collider, style, st, menton_z, RP, rep, rng, rng_b, tick, log, t0, cache_dir=None):
    """grow mode: follicles -> fibre and lengths -> guides and members -> growth with the acts on (settle.grow) ->
    the cut on the settled hair -> the water set -> release -> the final gated settle -> children -> form (no crown
    lift) -> baby hairs -> colour -> a constant 0.2 mm push-out -> the report. Never calls heights, h_of, density_k,
    the stack, the corridor drop or revert, or the ear clamps (DESIGN_v2 section 6). Everything before growth is
    _prelude's (shared with the grown head, head.py); the acts are built here, on the prelude's rods; everything after
    the final settle is _finish's (shared with flow.order 'after')."""
    from . import actions as AC
    from . import settle as SE
    seed = int(st["seed"])
    X, din, phi_r, th_r, kind, main = (RP[k] for k in ("X", "din", "phi_r", "th_r", "kind", "main"))
    surf = collider
    stg = time.time
    rep["mode"] = "grow"
    rep["retired_key_warnings"] = SY.retired(style)
    pre = _prelude(scalp, collider, st, RP, rep, tick, cache_dir=cache_dir)
    (T, Wh, fb, mL, gi, f, body, rods, fresh_rods) = (pre[k] for k in (
        "T", "Wh", "fb", "mL", "gi", "f", "body", "rods", "fresh_rods"))
    # the acts on the uncut rods (they read X, s, l, W; rest_waves neither reads nor writes what they read, so building
    # them after the waves leaves both as before)
    t = stg()
    acts = AC.Acts(scalp, body, st, rods, W=Wh, coll_V=surf.V, seed=seed, log=lambda m: None)
    T["body_rods"] += stg() - t

    # styling in sections (DESIGN_sections, off by default): the cut first in natural fall (a salon cuts wet, then
    # blow-dries), then section by section from the nape to the front - grown, brushed, held while it dries and
    # cools, let go and dropped onto the sections laid before it
    sec_cfg, sec_warn = AC.sections_settings(st)
    sec_on = bool(sec_cfg["on"])
    slog = lambda m: tick("  " + m)  # noqa: E731
    ssec_all = None
    if sec_on:
        rep["sections"] = {"warnings": list(sec_warn), "order": list(sec_cfg["order"])}
        ssec_all = AC.section_of(scalp, X, st["dry"], sec_cfg["order"])
        ssec_g = ssec_all[gi]
        rep["sections"]["guides"] = {nm: int((ssec_g == i).sum()) for i, nm in enumerate(sec_cfg["order"])}
        rep["sections"]["roots"] = {nm: int((ssec_all == i).sum()) for i, nm in enumerate(sec_cfg["order"])}
        tick("sections: %s guides" % rep["sections"]["guides"])
        if str(st["cut"].get("taken", "held")) != "fall":
            rep["sections"]["warnings"].append("cut.taken '%s' refused with sections on: 'fall' used (the cut comes "
                                               "first, on wet hair in its natural fall)" % st["cut"].get("taken"))
        t = stg()
        ztip, layer = _cut_plan(st, menton_z, phi_r, th_r, din, main, seed)
        rods_f = fresh_rods()
        SE.grow(rods_f, body, acts=None, cfg=st, pressure=True, friction=True, log=lambda m: None)
        Lcut, reached = _cut(rods_f, ztip, layer, mL)
        rep["cut_fall"] = {"seconds": round(stg() - t, 1),
                           "guide_tip_z_mm_p10_50_90": np.round(1000 * np.percentile(
                               rods_f.positions()[:, -1, 2] - menton_z, [10, 50, 90]), 1).tolist()}
        del rods_f
        Lcut = np.where(kind == 2, mL, Lcut)
        rods.members(cut=Lcut)
        T["cut"] = stg() - t
        rep["cut"] = {"reached_plane": int(reached.sum()), "members": int(len(Lcut)),
                      "length_mm_p10_50_90": np.percentile(Lcut * 1000, [10, 50, 90]).round(1).tolist(),
                      "grown_mm_p10_50_90": np.percentile(mL * 1000, [10, 50, 90]).round(1).tolist(),
                      "held_at_grown_length": int((Lcut >= mL - 1e-9).sum()), "taken": "fall"}
        tick("cut (first, in natural fall): %d of %d hairs reached their plane; lengths p10/50/90 %s mm" % (
            reached.sum(), len(Lcut), rep["cut"]["length_mm_p10_50_90"]))
        t = stg()
        state = SE.grow_sections(rods, ssec_g, body, acts=acts, cfg=st, pressure=True, friction=True, log=slog,
                                 hold=lambda s_, on_: AC.section_hold(s_, acts, on_, slog), stick=AC.lag_stick,
                                 drop_evals=int(sec_cfg["drop_evals"]))
        T["grow"] = stg() - t
        rep["sections"]["evals"] = {sec_cfg["order"][d["section"]]: d["evals"] for d in state.sections}
    else:
        # growth by continuation with the acts on (DESIGN_v2 4.6)
        t = stg()
        state = SE.grow(rods, body, acts=acts, cfg=st, pressure=True, friction=True, log=slog)
        T["grow"] = stg() - t

    if not sec_on:  # (with sections on the cut came first, above)
        # the cut on the hair as it now falls, capped by what grew (4.7); then 2 sub-solves on the cut rods
        t = stg()
        ztip, layer = _cut_plan(st, menton_z, phi_r, th_r, din, main, seed)
        CU = st["cut"]
        # where the scissors read the length (cut.taken): 'held' (as before) on the hair as the growth left it, the brush
        # still carrying it - at the temples that path runs back round the ear and down the back, a longer way to the plane
        # than the hair's own fall, so those lengths came out ~100 mm longer than the back's (gr10: temple guides cut to
        # 477-520 mm, the back's 352-453) - a cut taken over-directed, which leaves hair "longer where it was pulled from"
        # (hair_styling.md 7, over-direction row, P9). 'fall': the perimeter cut as a stylist takes it, on wet hair combed
        # down in its natural fall (the solid 0 deg row, P1-P3; the length map: "hang the strands in natural fall, gravity
        # and growth field only"), then the blow-dry: the same guides grown a second time with every act off (~30 s at 700
        # guides), the planes read on that fall, then the styled growth goes on as grown.
        taken = str(CU.get("taken", "held"))
        if taken == "fall":
            tf = stg()
            rods_f = fresh_rods()            # the same hair, the same natural wave, in its natural fall
            SE.grow(rods_f, body, acts=None, cfg=st, pressure=True, friction=True, log=lambda m: None)
            Lcut, reached = _cut(rods_f, ztip, layer, mL)
            rep["cut_fall"] = {"seconds": round(stg() - tf, 1),
                               "guide_tip_z_mm_p10_50_90": np.round(1000 * np.percentile(
                                   rods_f.positions()[:, -1, 2] - menton_z, [10, 50, 90]), 1).tolist()}
            del rods_f
        else:
            Lcut, reached = _cut(rods, ztip, layer, mL)
        Lcut = np.where(kind == 2, mL, Lcut)
        rods.members(cut=Lcut)
        # a new state on the cut rods: every surviving segment stays as it lies (none is re-hung); anchors taken now
        st2 = SE.State(rods, body, st, True, True, acts, state.splat, slog)
        st2.act_full = rods.l > 0
        st2.activate(float(rods.L.max()) + 1e-9)
        st2.evals, st2.eval_s, st2.hist, st2.stage_index, st2.t0 = state.evals, state.eval_s, state.hist, \
            state.stage_index, state.t0
        state = st2
        acts.lag(state.view())
        for k in range(2):
            state.solve(max(2, int(round(f["evals_stage"] / 2))), label="cut %d" % k)
            acts.lag(state.view())
            state.lag()
        T["cut"] = stg() - t
        rep["cut"] = {"reached_plane": int(reached.sum()), "members": int(len(Lcut)),
                      "length_mm_p10_50_90": np.percentile(Lcut * 1000, [10, 50, 90]).round(1).tolist(),
                      "grown_mm_p10_50_90": np.percentile(mL * 1000, [10, 50, 90]).round(1).tolist(),
                      "held_at_grown_length": int((Lcut >= mL - 1e-9).sum()), "taken": taken}
        tick("cut: %d of %d hairs reached their plane; lengths p10/50/90 %s mm" % (
            reached.sum(), len(Lcut), rep["cut"]["length_mm_p10_50_90"]))

    # the water set (dry.*), release (acts off but the pins that stay), tousle, the final gated settle
    t = stg()
    state, nset = _tail(state, acts, f, slog, sec_on=sec_on)
    T["dry_release_final"] = stg() - t
    _settled(rep, state, scalp, menton_z, nset, tick, sec_on=sec_on)
    return _finish(scalp, collider, style, st, RP, pre, rods, Lcut, acts, fb, rep, rng, rng_b, tick, t0, T,
                   sec=(sec_cfg, ssec_all) if sec_on else None)


def _tail(state, acts, f, slog, sec_on=False):
    """the style's end, one copy for both orders: the water set (dry.*) and its hold, release (every act off but the
    pins that stay), the release solves with the friction anchors kept by actions.lag_stick, the tousle strokes, then
    settle.release's final gated settle (its own release solves off: they would re-anchor every sample). With sections
    on every section was already held, set and dropped in turn: only the release. f: the settle's settings
    (release_solves, evals_release). -> (state, joints set)"""
    from . import actions as AC
    from . import settle as SE
    if sec_on:
        # every section was held, set and dropped in turn; nothing is held now
        nset = int(sum(int(x) for x in getattr(state, "sections_set", [])))
        acts.on_g = None
        acts.release()
        state.stage_index += 1
    else:
        if getattr(state, "dry_done", None) is not None:
            nset = int(state.dry_done)         # dry.when 'before_hands': set in the style stage, before the hands
        else:
            nset = AC.dry_set(state, acts)
            AC.dry_hold(state, acts, log=slog)     # the section stays in the brush while it cools (dry.hold), then lets go
        acts.release()
        state.stage_index += 1
        nrel = max(1, int(f["release_solves"]))
        if f["evals_release"] > 0:
            per = max(2, int(round(f["evals_release"] / nrel)))
            for k in range(nrel):
                state.solve(per, label="release %d" % k)
                AC.lag_stick(state)            # static friction: hair that held keeps its anchor (as actions.settle_look)
    for k in range(int(acts.v["tousle"]["strokes"])):
        acts.tousle_stroke(k)
        acts.lag(state.view())
        state.solve(int(acts.v["tousle"]["evals"]), label="tousle %d" % k)
        AC.lag_stick(state)
    acts.tousle_off()
    keep = state.f["evals_release"]
    state.f["evals_release"] = 0
    try:
        state = SE.release(state, acts=acts)
    finally:
        state.f["evals_release"] = keep
    return state, nset


def _settled(rep, state, scalp, menton_z, nset, tick, sec_on=False):
    """the settle's report on the guides as they now lie, into rep: settle, dry_joints_set (and the sections' slip)"""
    from . import settle as SE
    srep = SE.report(state, {"centre": scalp.c, "chin_z": menton_z})
    rep["settle"] = {k: srep[k] for k in ("guides", "members", "energies_per_guide_weight_mm", "gate",
                                          "clamp_first_segment_vs_d0_deg_p50_p90_max", "fibre_fraction", "layers",
                                          "evals", "seconds_per_eval") if k in srep}
    rep["dry_joints_set"] = nset
    if sec_on:
        rep["sections"]["slip_from_drop_to_end"] = SE.section_slip(state)
    tick("settle: gate %s (max %.3f, p99 %.4f), %d evaluations" % (
        state.gate["passed"], state.gate["max_force_ratio"], state.gate["p99_force_ratio"], state.evals))


# ---------------------------------------------------------------- grow, then style (flow.order 'after')
def head_style(style):
    """the style as the Hair stage reads it: a copy without flow.order. The order says how a run uses the head (grow
    then style, or the acts while it grows), not what the head is; the head's key covers every sim leaf
    (head.settings_of), so it is taken without this one - one head whichever order asks for it, and the same key as a
    head grown before the switch existed"""
    s = copy.deepcopy(style)
    if isinstance(s.get("flow"), dict):
        s["flow"].pop("order", None)
    return s


def _the_head(scalp, collider, sh, menton_z, head, head_cache, regrow, case, log):
    """the Hair stage's grown head: given (a head.GrownHead or a saved path: used as it is), else the cache's
    <key16>.npz when its full key is this style's (head.head_key: the hair, environment and sim settings, the style
    keys the head reads, the case, the code), else grown now by head.grow_head (no act, oil.product 0) and saved
    there. Says 'head: cached <key16>' or 'head: grown <key16> in N s'. -> (GrownHead, report {key, path, cached,
    given, regrow, seconds})"""
    from . import head as HD
    t = time.time()
    if head is not None:
        path = os.path.abspath(head) if isinstance(head, str) else None
        hd = HD.load(path) if path else head
        log("head: given %s%s" % (hd.key[:16], (" (%s)" % path) if path else " (in memory)"))
        return hd, {"key": hd.key, "path": path, "cached": True, "given": True, "regrow": False,
                    "seconds": round(time.time() - t, 1)}
    key, _ = HD.head_key(sh, scalp, collider, menton_z, case)
    path = os.path.join(os.path.abspath(head_cache), key[:16] + ".npz") if head_cache else None
    hd = None
    if path and not regrow and os.path.isfile(path) and os.path.isfile(path[:-4] + ".json"):
        hd = HD.load(path)
        if hd.key != key:                 # another head under the same 16 characters: this one is grown over it
            hd = None
    if hd is not None:
        log("head: cached %s" % key[:16])
        return hd, {"key": key, "path": path, "cached": True, "given": False, "regrow": False,
                    "seconds": round(time.time() - t, 1)}
    hd = HD.grow_head(scalp, collider, sh, menton_z, case=case, log=lambda m: log("  head " + m))
    if hd.key != key:
        raise HD.HeadMismatch("grow_head keyed the head %s, the run %s" % (hd.key[:16], key[:16]))
    if path:
        HD.save(hd, path)
    s = round(time.time() - t, 1)
    log("head: grown %s in %.1f s%s" % (key[:16], s, (" -> %s" % path) if path else " (no head cache: not saved)"))
    return hd, {"key": key, "path": path, "cached": False, "given": False, "regrow": bool(regrow), "seconds": s}


def _style_product(state, fb):
    """the style's product put on the grown head (the head grew with oil.product 0: product is a style pass, put on
    hair that is already there). The rods take the style's fibre (fibre.props with the style's whole oil block) and
    rebuild their member sums: only B changes, by the bind's oil bond (fibre.bind oil_beta x the oil's wetting); l, s,
    W, D and the rest turning stay as they lie. The State re-reads what its constructor takes from the fibre (c_p, mu,
    mu_s, the bridges' adh, c_a, c_l) and re-sets its stage (_setup: the wetting swf, the anchors' arrays, the
    scaling). The friction anchors keep their loads until the next lag (static friction is history); the alignment
    sal stays as its stage set it. Here, not in settle.py: it sets the attributes the State's own constructor sets.
    -> report"""
    r = state.rods
    sal = state.sal
    B0, swf0 = r.B.copy(), state.swf
    r.fibre = fb
    r.members()
    state.c_p = fb.c_p
    state.mu = fb.mu
    state.mu_s = float(getattr(fb, "mu_skin", fb.mu))
    state.adh = getattr(fb, "adhesion", None) if state.friction else None
    state.c_a = fb.cohesion_c0() if (state.pressure and getattr(fb, "cohesion", False)) else 0.0
    state.c_l = fb.cohesion_sigma() if (state.c_a > 0 and getattr(fb, "cohesion_count", "crossing") == "aligned") \
        else 0.0
    state._setup(state.Ka)
    if sal is not None and state.c_l > 0:
        state.sal = sal
    on = B0 > 0
    pq = lambda v: None if v is None or not len(v) else [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])]  # noqa: E731
    return {"B_over_head_p10_50_90": pq(r.B[on] / B0[on]), "wetting_p10_50_90": pq(state.swf),
            "wetting_head_p10_50_90": pq(swf0), "c_a0_pa": round(float(state.c_a), 2)}


def _style_moves(st):
    """what in the style acts on the grown head: the brush, the tuck, the water set (dry.set, its lift, the hold), the
    tousle, the product. Empty: the style stage is the identity - nothing touches the hair, so nothing is solved (a
    solve from the rested head would move it by the solver's own steps, with no cause)"""
    from . import actions as AC
    v, _ = AC.settings(st)
    dv = v["dry"]
    return [nm for nm, on in (("brush", bool(v["brush"]["on"])), ("tuck", bool(v["tuck"]["on"])),
                              ("dry", float(dv["set"]) > 0 or float(dv["lift_deg"]) > 0 or
                               (bool(dv["hold"]) and int(dv["hold_evals"]) > 0)),
                              ("tousle", int(v["tousle"]["strokes"]) > 0),
                              ("product", float((st.get("oil") or {}).get("product", 0.0)) != 0.0)) if on]


def _after_skipped(st, acts=None):
    """the acts that only make sense while the hair grows: off in the style stage (acts.face_on and acts.hand_on are
    switched off here; the others have nothing to act on) and listed in the report (after_skipped), each with whether
    the style had it on. -> {key: {on, why}}"""
    from . import actions as AC
    v, _ = AC.settings(st)
    g = st.get("grow") or {}
    out = {
        "grow.grow_along_stroke": {
            "on": bool(g.get("grow_along_stroke", True)) and bool(v["brush"]["on"]) and bool(v["brush"]["through"]),
            "why": "lays each NEW length along the brush's stroke as it emerges (settle.State.activate); on a grown "
                   "head no length emerges"},
        "hand": {
            "on": bool(v["hand"]["on"]),
            "why": "hand.stages counts growth stages from the first whose lengths reach the shoulders, then lets go; "
                   "a grown head has no growth stages (the lengths laid by a hand come back as a pass)"},
        "face.barrier": {
            "on": bool(v["face"]["barrier"]),
            "why": "growth only (actions.py): a hand keeps the hair off the face while it is brushed as it grows; "
                   "the grown head's hair already lies where it fell"},
        "sections": {
            "on": bool(AC.sections_settings(st)[0]["on"]),
            "why": "styling in sections grows the head section by section (settle.grow_sections)"},
        "cut.taken": {
            "on": str((st.get("cut") or {}).get("taken", "held")) != "fall",
            "why": "the head is cut on its natural fall (head.grow_head); 'held' reads the length on hair the acts "
                   "carried while it grew"},
    }
    if acts is not None:
        acts.face_on = False
        acts.hand_on = False
    return out


def acts_passes(state, acts, st, log=None, probe=None):
    """the style stage's hook until passes.py (DESIGN_grow_then_style piece 3): the existing act energies on the grown
    head as ONE lagged block - the acts' lagged data taken on the hair as it lies, then settle's release block
    (release_solves solves, evals_release evaluations in all) with the acts ON, the friction anchors re-taken between by
    actions.lag_stick (static friction: hair that held keeps its anchor; settle.lag re-lags the acts). Nothing is tuned
    for it: it is the baseline the passes replace (a brush pulls on the heading only, so on grown hair it cannot lift
    a curtain off the face - DESIGN 'A stroke as a pass'). -> report (json values)"""
    from . import actions as AC
    f = state.f
    e0 = state.evals
    state.stage_index += 1
    acts.lag(state.view())
    out = {"hook": "acts_passes", "solves": [], "lag_stick_stuck_slid": []}
    nrel = max(1, int(f["release_solves"]))
    if f["evals_release"] > 0:
        per = max(2, int(round(f["evals_release"] / nrel)))
        for k in range(nrel):
            state.solve(per, label="acts %d" % k)
            h = state.hist[-1]
            out["solves"].append({"evals": h["evals"], "E0": h["E0"], "E1": h["E1"]})
            out["lag_stick_stuck_slid"].append(list(AC.lag_stick(state)))
            if probe is not None:
                probe("acts %d" % k, state)
    _, o = state.energies()
    out["energies_per_guide_weight_mm_end"] = {k: round(float(v) * state.ESC, 4) for k, v in o["parts"].items()}
    out["evals"] = int(state.evals - e0)
    return out


def _passes_hook(passes):
    """the style stage's hook -> (fn(state, acts, st, log, probe=None) -> report, name). 'auto': passes.py's
    run_passes when engine/hair/passes.py is there and defines it (piece 3: a stroke as a pass), else acts_passes;
    'acts': acts_passes always (the baseline the passes replace); a callable: itself. passes.py is imported only here,
    so a run that never styles after never loads it"""
    if callable(passes):
        return passes, getattr(passes, "__name__", "callable")
    if passes == "acts":
        return acts_passes, "acts_passes"
    if passes != "auto":
        raise ValueError("passes %r: 'auto', 'acts' or a callable" % (passes,))
    if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "passes.py")):
        from . import passes as PS
        fn = getattr(PS, "run_passes", None)
        if callable(fn):
            return fn, "passes.run_passes"
    return acts_passes, "acts_passes"


def _grow_after(scalp, collider, style, st, menton_z, RP, rep, rng, rng_b, tick, log, t0, head=None, head_cache=None,
                regrow=False, case=None, passes="auto", style_stage=True, from_state=None, save_state=None):
    """flow.order 'after' (DESIGN_grow_then_style): grow once, then style. The Hair stage is a grown head (_the_head:
    given, cached by its key, or grown now with no act and oil.product 0), live through head.instantiate at its
    'rested' snapshot - its cut, its anchors, its rest. The Style stage: the style's product (_style_product), the acts
    built on those cut rods AFTER the head (actions.Acts; the growth-only acts off and listed: _after_skipped), the
    passes (_passes_hook), then the legacy tail (_tail: the water set, release with lag_stick, tousle, the final). A
    style with nothing on (_style_moves empty) leaves the rested head as it lies: no solve at all. Then the same finish
    (_finish). style_stage False (finish_head): no product, no acts - the rested head finished as it lies. RP, rng,
    rng_b: generate's (the finish draws from rng, rng_b in the legacy order). -> (arrays, report)"""
    from . import actions as AC
    from . import fibre as FB
    from . import head as HD
    seed = int(st["seed"])
    stg = time.time
    slog = lambda m: tick("  " + m)  # noqa: E731
    rep["mode"] = "grow"
    rep["order"] = "after"
    rep["retired_key_warnings"] = SY.retired(style)
    T = {}
    # the Hair stage: the grown head
    t = stg()
    sh = head_style(style)
    hd, rep["head"] = _the_head(scalp, collider, sh, menton_z, head, head_cache, regrow, case, log)
    T["head"] = stg() - t
    # styling is cumulative (10-31: further refinements to the style are further passes over what is already
    # done, as in real life):
    # from_state - a styled state saved by an earlier run (save_state) on THIS grown head - is where the passes start
    snap = "rested"
    if from_state is not None:
        hs = HD.load(from_state) if isinstance(from_state, str) else from_state
        if hs.key != hd.key:
            raise HD.HeadMismatch("the styled state was made on another grown head (%s, this hair's is %s)"
                                  % (hs.key[:16], hd.key[:16]))
        if "styled" not in hs.snaps:
            raise HD.HeadMismatch("%s holds no styled state" % (from_state,))
        hd, snap = hs, "styled"
        rep["state_from"] = {"path": from_state if isinstance(from_state, str) else None,
                             "history": hs.meta.get("styled_history")}
    # the rested head, live: instantiate is the only way to it (the prelude rebuilt and checked, the history put back)
    t = stg()
    hfl = (hd.meta.get("settings") or {}).get("flow") or {}
    if "order" in hfl:                   # a head grown by head.grow_head from a style that carried the key
        sh["flow"] = dict(sh.get("flow") or {}, order=hfl["order"])
    ctx = HD.rebuild(hd, scalp, collider, st=sh, log=slog)
    rods, state = HD.instantiate(hd, scalp, collider, snap, ctx=ctx, log=slog)
    pre = ctx["pre"]
    if not (np.array_equal(ctx["RP"]["X"], RP["X"]) and np.array_equal(ctx["RP"]["kind"], RP["kind"])):
        raise HD.HeadMismatch("the run's roots differ from the head's rebuilt ones")
    T["instantiate"] = stg() - t
    hr = hd.meta.get("report") or {}
    rep["head"].update(code_drift=ctx["code_drift"], name=hd.meta.get("name"),
                       rested={k: (hr.get("rested") or {}).get(k) for k in ("gate", "evals")})
    rep["cut"] = copy.deepcopy(hr.get("cut"))
    for k in ("roots", "follicle"):         # the prelude's own report (the follicles' whorl, angles, warnings)
        if k in ctx.get("prelude_report", {}) and k not in rep:
            rep[k] = ctx["prelude_report"][k]
    Lcut = rods.mc.copy()                # the head's cut (kind 2 at its grown length), as rods.members took it
    tick("head %s rested: %d guides, K %d, gate %s, %d evaluations; cut on the natural fall, lengths p10/50/90 %s mm"
         % (hd.key[:16], rods.G, rods.K, (state.gate or {}).get("passed"), state.evals,
            (rep["cut"] or {}).get("length_mm_p10_50_90")))
    acts = None
    nset = 0
    if style_stage:
        # the Style stage, on hair that is already there
        t = stg()
        fb = FB.props(st.get("fibre"), st["radius"], dict(st["oil"]))      # the style's fibre: the head's + its product
        prod = float(st["oil"].get("product", 0.0))
        ss = {"product": prod}
        if prod != 0.0:
            ss["product_applied"] = _style_product(state, fb)
        acts = AC.Acts(scalp, pre["body"], st, rods, W=pre["Wh"], coll_V=collider.V, seed=seed, log=lambda m: None)
        rep["after_skipped"] = _after_skipped(st, acts)
        ss["acts_on"] = _style_moves(st)
        if ss["acts_on"]:
            state.acts = acts
            hook, ss["passes"] = _passes_hook(passes)
            tick("style stage: %s on the grown head (%s); skipped (growth only): %s" % (
                ", ".join(ss["acts_on"]), ss["passes"],
                ", ".join(k for k, v in rep["after_skipped"].items() if v["on"]) or "none"))
            e0 = state.evals
            pr = hook(state, acts, st, slog)
            ss["passes_report"] = json.loads(json.dumps(pr, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                                        else str(o)))
            ss["evals_passes"] = int(state.evals - e0)
            T["style"] = stg() - t
            t = stg()
            state, nset = _tail(state, acts, pre["f"], slog)
            T["dry_release_final"] = stg() - t
            ss["evals_all"] = int(state.evals - e0)
        else:
            ss["identity"] = "no act on and no product: the rested head is the styled head (no solve)"
            tick("style stage: nothing on - the %s as it lies" % ("styled state" if snap == "styled" else "rested head"))
            T["style"] = stg() - t
        rep["style_stage"] = ss
        if save_state is not None:
            # the styled state, for the next refinement to start from: the rested head kept beside it, the history
            # of styles applied, the fresh rods' fingerprints (a style's product is re-applied by the next stage)
            hist = list(hd.meta.get("styled_history") or []) + [
                {"style": st["name"], "acts_on": ss.get("acts_on"), "product": prod, "passes": ss.get("passes")}]
            meta = copy.deepcopy(hd.meta)
            meta["styled_history"] = hist
            meta["fingerprints"]["styled"] = meta["fingerprints"]["rested"]
            meta.setdefault("state", {})["styled"] = {"gate": copy.deepcopy(state.gate), "hist": []}
            p_state = HD.save(HD.GrownHead(meta, {"rested": hd.snaps["rested"], "styled": HD._snapshot(rods, state)}),
                              save_state)
            rep["state_saved"] = {"path": p_state, "history": hist}
            tick("styled state saved: %s (%d styles on this head)" % (p_state, len(hist)))
    else:
        fb = pre["fb"]                                                      # the head's fibre (oil.product 0)
        st = dict(st, oil=dict(st["oil"], product=0.0))
        rep["style_stage"] = {"skipped": "finish_head: the rested head finished as it lies"}
    _settled(rep, state, scalp, menton_z, nset, tick)
    return _finish(scalp, collider, style, st, RP, pre, rods, Lcut, acts, fb, rep, rng, rng_b, tick, t0, T)


def finish_head(scalp, collider, style, menton_z, head, log=print):
    """the rested head finished as it lies: no product, no act, no solve - _finish (children, form, baby hairs, radius,
    push-out, colour) on the head's 'rested' snapshot, through the same path as flow.order 'after'. What a style that
    moves nothing gives (the identity proof's other side), and the npz to judge 'this head of hair' before any style.
    head: a head.GrownHead or its saved path; style: the run's style (its hair, sim and environment the head's).
    -> (arrays, report)"""
    s = dict(style, flow=dict(style.get("flow") or {}, order="after"))
    return generate(scalp, collider, s, menton_z, log=log, head=head, style_stage=False)


def _finish(scalp, collider, style, st, RP, pre, rods, Lcut, acts, fb, rep, rng, rng_b, tick, t0, T, sec=None):
    """the finish of grow mode, one copy for both orders (flow.order 'during': _grow; 'after': _grow_after and
    finish_head): the settled guides rods -> children -> form (no crown lift) -> baby hairs -> radius -> a constant
    0.2 mm push-out -> the face count -> colour (colour.colourize) -> the out dict and the report. RP: the run's roots
    (_roots_part_kinds); pre: _prelude's dict (the head's rebuilt prelude in 'after'); Lcut (M,) m: the cut lengths
    (rods.mc); acts: the style's actions.Acts, or None (finish_head: no act - nothing was held, set or tucked); fb: the
    fibre the hair has now (the style's product in it); rng, rng_b: generate's streams as the roots left them (form and
    colour draw from them, in the legacy order); T: the stage seconds so far; sec: (sections settings, each root's
    section) when styled in sections, else None. -> (arrays, report)"""
    from . import children as CH
    from . import follicle as FOL
    from . import settle as SE
    seed = int(st["seed"])
    K = int(st["points"])
    X, n, din, phi_r, th_r, kind, main = (RP[k] for k in ("X", "n", "din", "phi_r", "th_r", "kind", "main"))
    HL, R = st["hairline"], st["roots"]
    surf = collider
    stg = time.time
    (d0, mL, ms, side, x_part, zone, gcfg, gwarn, key, gi, swarn, body, wv_rods, ph_all, sec_all, ph_sec,
     lam_sec) = (pre[k] for k in ("d0", "mL", "ms", "side", "x_part", "zone", "gcfg", "gwarn", "key", "gi", "swarn",
                                  "body", "wv_rods", "ph_all", "sec_all", "ph_sec", "lam_sec"))
    sec_on = sec is not None
    sec_cfg, ssec_all = sec if sec_on else (None, None)

    # children: long hairs interpolated from the guides of their own part side, short ones as single rods (4.8)
    t = stg()
    Pg = rods.positions()
    long_ = Lcut >= gcfg["short_mm"] / 1000
    P = np.zeros((n, K, 3))
    il = np.nonzero(long_)[0]
    # locks from causes (locks.causes, off by default): hairs leave the scalp in follicular units of 1-4 that share an
    # opening and a follicle direction (hair.md point 2), so the children start from the units' roots. Off, the roots and
    # the RNG streams are exactly as before.
    causes = bool((st.get("locks") or {}).get("causes", False))
    unit = None
    if causes:
        Xu, du, unit, urep = CH.follicular_units(X[il], d0[il], seed, st)
        X = X.copy()
        d0 = d0.copy()
        X[il], d0[il] = Xu, du
    # the children blend only guides of their own drying section (rod_waves) and, with sections on and
    # sections.children_within, of their own staged section: a front guide laid over the temple blended with a side
    # guide tucked behind the ear would put a child between the two layers (DESIGN_sections 3.2)
    sc_c = sec_all[il] if wv_rods else None
    sc_g = sec_all[gi] if wv_rods else None
    if sec_on and bool(sec_cfg["children_within"]):
        nd = int(sec_all.max()) + 1 if wv_rods else 1
        sc_c = ssec_all[il] * nd + (sc_c if wv_rods else 0)
        sc_g = ssec_all[gi] * nd + (sc_g if wv_rods else 0)
    # 11-12 (fs-hair, n2d): with the phase field the guides of one section no longer wave in step, so a child's blend of
    # them averaged the wave flat (n2d: 70-82 % of the guides' wave kept vs n2c's 97 %). For the blend only, each guide is
    # shifted to its section's mean phase (the engine's wave_follow: directions moved along the arc, lengths kept); each
    # child is then shifted from that mean to its own phase (below). Field off: Pg as before.
    Pg_b, ph_bar = Pg, None
    if wv_rods and float(FO._cfg(st, "locks", FO.LOCK_CAUSE_DEFAULTS).get("phase_field_mm", 0.0)) > 0:
        sgi = sec_all[gi]
        zs = np.zeros(int(sec_all.max()) + 1, complex)
        np.add.at(zs, sgi, np.exp(1j * ph_all[gi]))
        ph_bar = np.angle(zs)
        dg = np.angle(np.exp(1j * (ph_bar[sgi] - ph_all[gi])))
        W8b = st.get("waves") or {}
        s0b = 0.5 * sum(W8b.get("start_mm", [0, 0])) / 1000.0 if "start_mm" in W8b else 0.0
        # (fs-hair 11-12, n2d's loose crown strands) the wave alone moved; wave_follow carried the large bends with it
        Pg_b = FO.wave_shift(np.array(Pg, np.float64), dg, lam_sec[sgi], s0=s0b)
        rep["wave_blend_dephase_rad_p50_p90"] = [round(float(v), 3) for v in np.percentile(np.abs(dg), [50, 90])]
    Pl, crep = CH.interpolate(Pg_b, rods.L, rods.X, X[il], d0[il], key[il], Lcut[il], key_g=key[gi], d0_g=rods.d0,
                              arcs=rods.node_arcs(), cfg=gcfg, log=None,
                              sec_c=sc_c, sec_g=sc_g)
    del Pg_b
    P[il] = Pl
    real_gid = crep.get("ringlet_guide")              # the ringlet each long child coils in (children_mode 'ringlet')
    if wv_rods:  # the wave's path to the children, measured at each step (rms mm by third of the length, p50)
        lam_w = float((st.get("waves") or {}).get("wavelength_mm", 70.0)) / 1000.0
        rep["wave_by_third_mm"] = {"guides": FO.wave_rms_by_third(Pg[rods.L > 0.25], lam_w, seed=seed),
                                   "children_blend": FO.wave_rms_by_third(Pl, lam_w, seed=seed)}
    ish = np.nonzero(~long_)[0]
    shrep = {}
    if len(ish):
        # the short hairs rest under the long hair lying over them (grow.canopy, on by default): the settled guides'
        # layer (its height and areal weight from rods.W, rods.V over the skin) is lifted only by doing work against
        # its weight, so a short hair under a full layer bends down inside it instead of standing through it as a halo
        cano = None
        if bool(st["grow"].get("canopy", True)):
            cano = CH.build_canopy((Pg, rods.W, rods.V, rods.l), X[ish], Lcut[ish], CH._distance(body), gcfg)
        # the short hairs are in the brush with their section (dry.hold, span 'held'): the stroke drawn through
        # carries them on the guides' held path and they set while held, as the long hair around them does
        held = r_set = shrep_laid = None
        P_held = getattr(acts, "P_held", None)
        if bool(st["grow"].get("short_set", True)) and P_held is not None:
            held, _ = CH.interpolate(P_held, rods.L, rods.X, X[ish], d0[ish], key[ish], Lcut[ish], key_g=key[gi],
                                     d0_g=rods.d0, arcs=rods.node_arcs(), cfg=gcfg, log=None)
            r_set = CH.set_share(acts.v["dry"], acts.v["wear"], acts.v["finish"], phi_r[ish], din[ish])
            if (bool(st["grow"].get("short_laid", False)) and getattr(acts, "tuck_pin", None) is not None
                    and acts.Qp is not None and acts.tuck_pin.any()):
                # the tucked side section's short hairs held on the hand's laid route over and behind the ear
                # (children.laid_held); rods.node_arcs() are the arcs the route was laid at (acts._trace_paths)
                held, shrep_laid = CH.laid_held(held, acts.Qp, acts.tuck_pin.any(1), rods.node_arcs(), key[gi], X[ish],
                                                key[ish], Lcut[ish], cfg=gcfg, pin=acts.tuck_pin)
        Ps, shrep = CH.short_hairs(X[ish], d0[ish], Lcut[ish], fb, body, scale=ms[ish], cfg=gcfg, log=None,
                                   canopy=cano, held=held, set_r=r_set)
        P[ish] = Ps
        if shrep_laid is not None:
            shrep["laid"] = shrep_laid
        shrep["out_of_layer_by_region"] = _short_out_by_region(
            cano, body, Pg, rods, Ps, X[ish], Lcut[ish], phi_r[ish], th_r[ish], din[ish],
            X[ish, 0] - scalp.c[0] - x_part, zone[ish], gcfg)
    T["children"] = stg() - t
    rep["children"] = {"long": int(len(il)), "short": int(len(ish)),
                       "interpolate": {k: v for k, v in crep.items() if not isinstance(v, np.ndarray)},
                       "short_hairs": {k: v for k, v in shrep.items() if not isinstance(v, np.ndarray)}}
    tick("children: %d long, %d short" % (len(il), len(ish)))

    # form: locks, clumping, waves, jitter, wander, frizz, fly-aways - never the crown lift in grow mode (4.9)
    t = stg()
    stf = dict(st)
    stf["lift"] = dict(st["lift"], crown_mm=0.0)
    stf["_part_zone"] = zone
    gside = side.astype(np.int64)
    s_tie = np.zeros(n)
    lk = FO.locks(X, gside, main, din, stf, rng, rng_b)
    rep["locks"] = lk["stats"]
    if causes:
        # the wave and the locks from their causes on the long children (form.py, DESIGN_v2 4.9): each drying section
        # sets one wave phase, the wave's amplitude is pulled out by the weight below; liquid bridges (sebum days and
        # product, oil.*) zip neighbouring groups where the adhesion beats the bending, then each lock packs as its
        # bridges pack it. The legacy stand-ins (lock / sub / bundle clumping toward Poisson centres, a random wave per
        # lock) are switched off in form.shape below so nothing is applied twice; jitter, wander, frizz, fly-aways stay.
        t_c = stg()
        lcs = FO._cfg(st, "locks", FO.LOCK_CAUSE_DEFAULTS)
        r_l = st["radius"]["root_mm"] / 1000 * ms[il]
        stc = dict(st, _centre=np.asarray(scalp.c, np.float64))
        Pc = P[il].astype(np.float64)
        if wv_rods:
            # the rods carry the wave (DESIGN_v2 6, form.py row: the phase-1 displacement switches off): each child
            # took its guides' wave; it keeps its own jitter about its section's step, read from the one field the
            # guides were waved from (its jitter less its nearest guide's - both about the same section phase)
            phase, sec = ph_all[il], sec_all[il]
            from scipy.spatial import cKDTree as _KD
            ng = gi[_KD(X[gi]).query(X[il])[1]]
            # 11-12 (fs-hair, n2d): with the phase field (locks.phase_field_mm > 0) a hair's jitter is its phase less the
            # smooth field at its root (form.phase_field without jitter: the same draws), not less its old drying
            # section's: against the section, a child whose nearest guide lay across a 35 mm section border took the
            # border's random phase step as its jitter (n2du: 8.5 % of the long hairs, p50 1.4 rad, up to pi)
            lcs_j = FO._cfg(st, "locks", FO.LOCK_CAUSE_DEFAULTS)
            if float(lcs_j.get("phase_field_mm", 0.0)) > 0:
                ph_ref = FO.phase_field(RP["X"], lcs_j["phase_field_mm"], seed, 0.0)
            else:
                ph_ref = ph_sec[sec_all]
            jit = lambda ii: ph_all[ii] - ph_ref[ii]  # noqa: E731
            W8 = st.get("waves") or {}
            s0w = 0.5 * sum(W8.get("start_mm", [0, 0])) / 1000.0 if "start_mm" in W8 else 0.0
            dphi = jit(il) - jit(ng)
            if ph_bar is not None:
                # the blend sits at its section's mean phase (above): each child to its own phase from the field
                dphi = np.angle(np.exp(1j * (ph_all[il] - ph_bar[sec_all[il]])))
            rep["wave_follow_dphi_rad_p50_p90_max"] = [round(float(v), 3) for v in np.percentile(np.abs(dphi), [50, 90, 100])]
            if ph_bar is not None:
                FO.wave_shift(Pc, dphi, lam_sec[sec], s0=s0w)     # the wave alone (the field's shifts reach pi)
            else:
                FO.wave_follow(Pc, dphi, lam_sec[sec], s0=s0w)
            rep["wave_by_third_mm"]["children_follow"] = FO.wave_rms_by_third(Pc, lam_w, seed=seed)
        else:
            phase, sec = FO.section_phase(Pc[:, 0], lcs["section_mm"], seed, lcs["phase_jitter"])
            if float(lcs.get("phase_field_mm", 0.0)) > 0:
                phase = FO.phase_field(Pc[:, 0], lcs["phase_field_mm"], seed, lcs["phase_jitter"])
            FO.section_waves(Pc, phase, r_l, fb, stc)
        tip0 = Pc[:, -1].copy()
        Pc, lock_id, lrep = FO.lock_causes(Pc, unit, r_l, phase, fb, stc, section=sec, log=lambda m: None,
                                           keep_wave_m=(_keep_wave(lam_sec, st, fb)[sec] if wv_rods else 0.0))
        # the gather's groups (lock_id) are NOT the saved locks yet: on look A they chain into ~42 head-wide locks of
        # ~3.4k hairs and every hair short of the layer is a group of one, so colour would turn to a few dozen flat
        # blocks plus single-hair speckle (review 10-31). Swap them in once the chaining is capped (locks.split).
        if wv_rods:
            rep["wave_by_third_mm"]["children_locks"] = FO.wave_rms_by_third(Pc, lam_w, seed=seed)
        # how far the locks moved each tip (mm): the hairs that end short of the layer must not move at all (a hair
        # joins a lock only over the length it shares with it), the long ones by their pack
        Lc = np.linalg.norm(np.diff(Pc, axis=1), axis=2).sum(1)
        mv = np.linalg.norm(Pc[:, -1] - tip0, axis=1) * 1000
        for nm_, sel_ in (("short_of_layer", Lc < lrep["layer_mm"] / 1000), ("reach_layer", Lc >= lrep["layer_mm"] / 1000)):
            lrep["tip_move_mm_p50_99_max_" + nm_] = (np.percentile(mv[sel_], [50, 99, 100]).round(3).tolist()
                                                     if sel_.any() else None)
        P[il] = Pc
        lrep["units"] = {k: v for k, v in urep.items() if not isinstance(v, np.ndarray)}
        lrep["seconds_all"] = round(stg() - t_c, 2)
        rep["locks_causes"] = lrep
        stf["clump"] = dict(st["clump"], lock=[0.0, 0.0], sub=[0.0, 0.0], bundle=[0.0, 0.0])
        stf["waves"] = dict(st["waves"], amp_mm=[0.0, 0.0])
        # the root jitter is the stand-in for the follicle's own turn, which the units now carry (fu_spread_deg, the
        # follicle field): left on it shifts every strand 3-6 mm across the sheet past 30 mm of arc, one by one, and
        # breaks the locks just formed (measured on the 20k run: 9 % of strands in clumps at 1 mm)
        stf["locks"] = dict(st["locks"], root_jitter_deg=0.0, bundle_jitter_deg=0.0)
        tick("form: locks from causes, %d groups at the layer (%d hairs reach it, %d end short of it and stay as they "
             "lie) from %d units, compaction p50 %.3f" % (
                 lrep["groups_at_layer"], lrep["hairs_reaching_layer"], lrep["hairs_short_of_layer_left_alone"],
                 urep.get("units", -1), lrep["compaction_p50_hairweighted"]))
    if wv_rods:  # the rods carry the wave: form.shape's per-lock stand-in wave stays off whatever the locks do
        stf["waves"] = dict(st["waves"], amp_mm=[0.0, 0.0])
    P, fly, fst = FO.shape(P, X, kind, din, phi_r, th_r, X[:, 0] - x_part, s_tie, scalp, stf, lk, rng, rng_b,
                           log=lambda m: None)
    rep["form"] = fst
    T["form"] = stg() - t

    # baby hairs at the hairline: their own follicle direction, short, fine, curled (the legacy step 5's shape)
    t = stg()
    Bb = HL["baby"]
    Cc = st["colour"]
    rb_ = np.random.default_rng(seed + 241)
    nb_in_band = int(((din > Bb["band_mm"][0]) & (din < Bb["band_mm"][1]) & (np.abs(phi_r) < Bb["phi_deg"])).sum())
    n_baby = int(Bb["frac"] * nb_in_band) + int(Bb["extra"])
    baby = None
    if n_baby > 0:
        Xb, Nb, _ = scalp.sample_roots(n_baby * 3, 0, 1, rb_, band=tuple(Bb["band_mm"]))
        phb, _, _ = scalp.sph(Xb)
        okb = np.nonzero(np.abs(phb) < Bb["phi_deg"])[0][:n_baby]
        Xb, Nb = Xb[okb], Nb[okb]
        m = len(Xb)
        gb, _, _, _, _ = FOL.field(scalp, Xb, Nb, None, None, None, st, np.random.default_rng(seed + 251))
        out = Xb - (scalp.c + np.array([0.0, 0.0, 0.12]))
        out = _unit(out - np.einsum("ij,ij->i", out, Nb)[:, None] * Nb + 1e-9)
        C = _unit(Bb["outward"] * out + (1 - Bb["outward"]) * gb)
        rot = np.radians(rb_.normal(0, 35.0, m))[:, None]
        C = C * np.cos(rot) + np.cross(Nb, C) * np.sin(rot)
        lift = np.radians(rb_.uniform(*Bb["lift_deg"], m))[:, None]
        db = np.cos(lift) * C + np.sin(lift) * Nb
        side_v = _unit(np.cross(Nb, C))
        bend_ax = _unit(np.cos(rb_.uniform(0, np.pi, m))[:, None] * side_v + np.sin(rb_.uniform(0, np.pi, m))[:, None]
                        * np.cross(db, side_v))
        Lb = rb_.uniform(*Bb["length_mm"], m) / 1000
        tot = Bb["curl"] * 2 * np.pi * rb_.uniform(0.5, 1.5, m) * rb_.choice([-1.0, 1.0], m)
        ang = tot[:, None] * np.linspace(0, 1, K)[None, :]
        dd = db[:, None, :] * np.cos(ang)[..., None] + np.cross(bend_ax, db)[:, None, :] * np.sin(ang)[..., None]
        seg = dd * (Lb / (K - 1))[:, None, None]
        Pbaby = Xb[:, None, :] + np.concatenate([np.zeros((m, 1, 3)), np.cumsum(seg[:, :-1], 1)], 1)
        baby = dict(P=Pbaby, X=Xb, m=m)
        rep["baby_hairs"] = int(m)

    # colour per point: a stage of its own now (colour.colourize, called below on the arrays as they are saved). Here
    # only the ids it reads: the REAL locks (the ringlets or the gather's groups) where the hair has them; the saved
    # lock / sub ids are the same, so a lock-wise step downstream (colour, a Blender clump split_by_lock) sees them;
    # and each strand's bundle (form.locks: its bundle's centre strand), saved as bun
    lk_c = _real_locks(lk, il, real_gid) if real_gid is not None and len(il) else lk
    if lk_c is lk:
        rep["locks_saved"] = "poisson roots (form.locks)"
    else:
        rep["locks_saved"] = "real: %d %s" % (lk_c["real_locks"], "ringlets" if real_gid is crep.get("ringlet_guide")
                                              else "gathered groups")
    R_ROOT = st["radius"]["root_mm"] / 1000
    r_root = R_ROOT * ms
    r_tip = st["radius"]["tip_mm"] / 1000 * ms
    lock_i, sub_i = lk_c["li"].astype(np.int32), lk_c["si"].astype(np.int32)
    bun_i = np.asarray(lk_c["bun"]).astype(np.int32)
    cross = side != np.where(X[:, 0] - x_part >= 0, 1, -1)
    if baby is not None:
        m = baby["m"]
        P = np.concatenate([P, baby["P"]])
        # the baby hairs' melanin and redness draws, kept and dropped: their colour is the colour stage's (its own
        # streams), and these two leave rb_ in the same state as before for the radius draw below - no geometry moves
        rb_.normal(0, Cc["strand_sd"], m)
        rb_.normal(0, Cc["redness_sd"], m)
        rbb = R_ROOT * rb_.uniform(*Bb["radius_scale"], m)
        r_root = np.concatenate([r_root, rbb])
        r_tip = np.concatenate([r_tip, rbb * 0.45])
        kind = np.concatenate([kind, np.full(m, 3, np.int8)])
        din = np.concatenate([din, scalp.d_inside_mm(baby["X"])])
        cross = np.concatenate([cross, np.zeros(m, bool)])
        fly = np.concatenate([fly, np.zeros(m, bool)])
        s_tie = np.concatenate([s_tie, np.zeros(m)])
        lock_i = np.concatenate([lock_i, np.full(m, -1, np.int32)])
        sub_i = np.concatenate([sub_i, np.full(m, -1, np.int32)])
        bun_i = np.concatenate([bun_i, np.full(m, -1, np.int32)])
    nt = len(P)
    # the radius along each strand, as the settle's fibre has it (10-30: the weight, density and properties of
    # the hairs as simulated): constant to within fibre.taper_mm of the hair's natural tip, so a cut hair ends blunt and an
    # uncut one thins over its last 30 mm. The renderer's old linear root-to-tip taper drew about two thirds of the hair the
    # physics settled (mean radius ~17 um against ~27). Baby hairs keep their own linear taper.
    nm = len(mL)
    fr = (np.clip((mL[:, None] - FO.arc(P[:nm])) / fb.taper, 0.0, 1.0) if fb.taper > 0
          else np.ones((nm, K)))
    rad = np.empty((nt, K), np.float32)
    rad[:nm] = r_tip[:nm, None] + (r_root[:nm, None] - r_tip[:nm, None]) * fr
    if nt > nm:
        tl = np.linspace(0.0, 1.0, K)[None, :]
        rad[nm:] = r_root[nm:, None] * (1 - tl) + r_tip[nm:, None] * tl
    rep["radius_um_mean_p50"] = round(float(np.median(rad[:nm].mean(1))) * 1e6, 2)
    T["baby_colour"] = stg() - t

    # the final push-out: exact distance, a constant 0.2 mm clearance (4.9), then the report
    t = stg()
    moved = 0
    for b0 in range(0, nt, 8000):
        sl = slice(b0, min(nt, b0 + 8000))
        P[sl], mv = surf.push_out(P[sl], 0.0002, iters=4, exact=True)
        moved += mv
    rep["collision_pushed_points"] = int(moved)
    col, _ = surf.report(P)
    rep["collision"] = col
    # points over the face: counted, never deleted (the corridor is a report and a barrier in grow mode)
    nface = 0
    sface = 0
    for b0 in range(0, nt, 20000):
        sl = slice(b0, min(nt, b0 + 20000))
        sd, j = surf.sd(P[sl])
        fm = FL.corridor_mask(scalp, st["corridor"], surf.V[j].reshape(-1, 3), sd.reshape(-1)).reshape(P[sl].shape[:2])
        nface += int(fm[:, 1:].sum())
        sface += int(fm[:, 1:].any(1).sum())
    rep["face"] = {"points_over_face": nface, "strands_touching_face": sface,
                   "share_points": round(nface / max(nt * (K - 1), 1), 5), "deleted": 0}
    T["push_report"] = stg() - t
    tick("collision: %d points inside, min clearance %.3f mm; %d points over the face (kept)" % (
        col["points_inside"], col["min_clearance_mm"], nface))

    # what each act did (through, blow, friction, tuck ...), so a run shows which acts were on (none: finish_head)
    rep["acts_counts"] = json.loads(json.dumps(acts.counts if acts is not None else {},
                                           default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    # the colour stage on the arrays exactly as saved (the final points, kinds, din, lock / sub / bundle ids) with its
    # own streams under the colour seed: engine/hair/recolour.py recomputes these same colours from the npz (scalp
    # frame from centre / grid / thb / ear_z_top) and a new colour block changes nothing else
    t = stg()
    pts_s, din_s = P.astype(np.float32), din.astype(np.float32)
    col = CO.colourize(pts_s, kind, din_s, lock_i, sub_i, bun_i, scalp, st, seed)
    rep["colour"] = col["report"]
    T["colour"] = stg() - t
    out = dict(pts=pts_s, r_root=r_root.astype(np.float32), r_tip=r_tip.astype(np.float32), kind=kind,
               radius=rad,
               melp=col["melp"], mel=col["mel"], red=col["red"], mel_root=col["mel_root"], sigma=col["sigma"],
               lock=lock_i, sub=sub_i, fly=fly, cross=cross, din=din_s, s_tie=s_tie.astype(np.float32),
               grid=scalp.grid, thb=scalp.thb, centre=scalp.c,
               guides=Pg.astype(np.float32), guide_L=rods.L.astype(np.float32),
               style_json=json.dumps({k: v for k, v in style.items() if not k.startswith("_")}),
               material_json=col["material_json"],
               bun=bun_i, colour_seed=np.int64(seed),
               ear_z_top=np.array([float(e["z_top"]) for e in scalp.ears.values()], np.float64))
    L_all = FO.arc(P)[:, -1] * 1000
    rep.update(strands=int(nt), points_total=int(nt * K), kinds={str(k): int((kind == k).sum()) for k in range(4)},
               length_mm_all_p10_50_90=np.percentile(L_all, [10, 50, 90]).round(1).tolist(),
               length_mm_main_p10_50_90=np.percentile(Lcut[main] * 1000, [10, 50, 90]).round(1).tolist(),
               corridor_dropped=0,
               material=st["material"], ash=Cc["ash"], stand_ins_used=list(SE.STAND_INS_USED),
               warnings_grow=gwarn + swarn + list(getattr(acts, "warnings", [])),
               stage_seconds={k: round(v, 1) for k, v in T.items()},
               seconds=round(time.time() - t0, 1), peak_ram_mb=peak_mb())
    return out, rep
