"""Hair Studio preflight (numpy, Apache-2.0): does our environment hold the contact laws at this guide count?
TRANSLATION.md rows 6-8, worker piece 1. Reads the resolved style and a strand count, solves nothing, takes < 1 s.

    hairs per guide   N / G
    guide spacing     S = sqrt(scalp area / G) at the scalp, against the pressure kernel's sd: the settle's SplatC1
                      (quadratic B-spline + [1, 2, 1] blur, variance 0.75 h^2) gives sd = 0.866 h, 2.6 mm at grid_mm 3;
                      with grow.splat_mm 'auto' its lattice is h = max(grid_mm, splat_sd_frac S / 0.866) (settle.
                      splat_lattice_mm), so sd >= splat_sd_frac S (0.25 S by default)
    inflation         the MEASURED lateral factor (TRANSLATION row 6's measured sub-row, tests/test_pressure_scale.py):
                      k_eff / k as H_meas^3 / H^3 on the flat patch, against S / sd - x4.2 at 7.7, x1.1 at 3.85, ~1 at
                      1.9 (k 1e-5; the guides stack nearly continuously across the layer, so only the lateral spacing
                      inflates). Log-log interpolated between the three points (slope 1.93 between the top two), 1
                      below 1.9, and past 7.7 the 1-D tube lattice's far law (S / sd)^2 from x4.2. The square lattice
                      of tubes (tube_inflation, kept as the reference and reported) overstated it: x30 / x2.0 / x1.0
    k_eff             k_type x inflation (k_type = fibre.props' k: k_vw, the type's curvature, the oil)
    H                 the analytic Van Wyk layer on a flat patch of fibre volume per area Q (m): P(z) = kE phi^3 holding
                      the weight above, dP/dz = -rho g phi, gives H = (3Q/2)^(2/3) (3 kE / (2 rho g))^(1/3). Q is the
                      column of hair lying over the scalp: a spherical cap of the scalp's area about a head of radius R,
                      hair flowing down the meridians from the crown, each hair covering at most its length L:
                      q(theta) = (N pi r^2 / A) R (cos(max(theta - L/R, 0)) - cos theta) / sin theta, the cap's area
                      mean (and its edge value reported)
    stand-off         the root's bend length with the binding: l_g F^(1/3), F the ramped root factor - the mean of
                      fibre.bind_factor over the root's own bend span [0, l_g F^(1/3)] (a fixed point). The bond
                      ramps in from the root (bind.start_mm, TRANSLATION row 8 piece 2), so bind_factor(0) is 1; the
                      span past the ramp still carries m^beta(s) and holds the root off the head (test_bind ramp:
                      the ramp alone took a curly guide's stand-off from 3.9x to 3.8x the loose one's).
                      l_g = (EI / w)^(1/3) of one hair, 37-50 mm at 27-40 um

STOP when H (with k_eff, at the mean Q) or the stand-off is over the type's band (BANDS). Both bands are design values (d):
no published layer-thickness table was found (TRANSLATION row 7); the stand-off's 70 mm is the bottom of the 70-93 mm
the bound root gave (row 8), against one hair's 37-50 mm.

    from engine.hair import preflight as PF
    rep = PF.check(style, count=None, scalp_area_mm2=None, head_r_mm=None)   # rep["ok"], rep["stops"], the numbers
    print(PF.format_report(rep))
"""
import numpy as np

from . import fibre as FB

G = 9.81
SD_PER_CELL = float(np.sqrt(2.0 / 3.0))     # grid.Splat's kernel sd per grid cell (trilinear tent + [1, 2, 1] blur)
SD_SPLAT_C1 = float(np.sqrt(0.75))          # settle.SplatC1's (quadratic B-spline + blur): the settle's splat, both modes
# the measured lateral model: (S / sd, k_eff / k = H_meas^3 / H^3) on the flat patch, sd the SplatC1's 2.6 mm (TRANSLATION
# row 6, measured 2026-10-31: H_meas / H 1.61 / 1.03 / 0.94 at 25 / 100 / 400 guides, k 1e-5)
LATERAL_MEASURED = ((1.9, 1.0), (3.85, 1.1), (7.7, 4.2))
SCALP_AREA_MM2 = 62339.0                    # her FH3 cap (types_*.npz cap_V / cap_F), the default when none is given
HEAD_R_MM = 90.0                            # the head radius of the cap model (piece 2's sphere) when none is given

# the type's band (d): H_max_mm the Van Wyk layer at k_eff over the scalp's mean column, standoff_max_mm the bound root
BANDS = {
    "straight": {"H_max_mm": 20.0, "standoff_max_mm": 70.0},
    "wavy": {"H_max_mm": 30.0, "standoff_max_mm": 70.0},
    "curly": {"H_max_mm": 50.0, "standoff_max_mm": 70.0},
}


def root_factor(fb, iters=30, n=400):
    """the ramped root factor F and the bound root's bend length l (m): F = the mean of fb.bind_factor over [0, l],
    l = l_g F^(1/3), iterated to its fixed point (1, l_g when the binding is off)"""
    lg = float(fb.l_g)
    if getattr(fb, "bind", None) is None:
        return 1.0, lg
    F, l = 1.0, lg
    for _ in range(iters):
        F = float(np.mean(fb.bind_factor((np.arange(n) + 0.5) / n * l)))
        l2 = lg * F ** (1.0 / 3.0)
        if abs(l2 - l) < 1e-9:
            break
        l = l2
    return F, lg * F ** (1.0 / 3.0)


def kernel_sd_mm(grid_mm):
    """sd (mm) of grid.Splat: trilinear onto grid_mm cells, then [1, 2, 1] / 4 once per axis (the square-lattice
    reference's sd, TRANSLATION row 6's first table; the settle's own kernel is splat_sd_mm)"""
    return float(grid_mm) * SD_PER_CELL


def lateral_inflation(S_mm, sd_mm, table=LATERAL_MEASURED):
    """the measured lateral (stacked) inflation of k at guide spacing S against the kernel sd: log-log through the
    table's points, 1 below the first, (S / sd)^2 past the last (the 1-D tube lattice's far law)"""
    x = float(S_mm) / float(sd_mm)
    xs = np.log([p[0] for p in table])
    ys = np.log([p[1] for p in table])
    if x <= table[0][0]:
        return float(table[0][1])
    if x >= table[-1][0]:
        return float(table[-1][1] * (x / table[-1][0]) ** 2)
    return float(np.exp(np.interp(np.log(x), xs, ys)))


def tube_inflation(S_mm, sd_mm, n=512):
    """<phi^3> / <phi>^3 of a square lattice (spacing S) of Gaussian tubes of sd: the lattice of a separable Gaussian
    is separable, phi(x, y) = g(x) g(y) with g the 1-D periodic sum, so the ratio is the 1-D ratio squared (exact)"""
    S, sd = float(S_mm), float(sd_mm)
    x = (np.arange(n) + 0.5) / n * S
    m = int(np.ceil(6.0 * sd / S)) + 1
    g = np.exp(-((x[:, None] - S * np.arange(-m, m + 1)[None, :]) ** 2) / (2.0 * sd * sd)).sum(1)
    r1 = float(np.mean(g ** 3) / np.mean(g) ** 3)
    return r1 * r1


def mean_length_mm(fibre_cfg, max_mm=None):
    """the mean grown length (mm) from the growth cycle (fibre.cycle_lengths in expectation), capped at length.max_mm"""
    cy = FB.resolve(fibre_cfg)["cycle"]
    rate_y = float(cy["rate_mm_month"]) * 12.0
    a0, a1 = (float(v) for v in cy["anagen_years"])
    tf = float(cy["telogen_frac"]) if cy["on"] else 1.0
    L = rate_y * 0.5 * (a0 + a1) * (0.5 * (1.0 - tf) + tf)
    return float(min(L, float(max_mm))) if max_mm is not None else float(L)


def column_Q(N, r_m, area_mm2, head_r_mm, L_mm, n=600):
    """the fibre volume per area (m) lying over the scalp: the cap model above -> (area mean, edge value)"""
    A, R, L = float(area_mm2) * 1e-6, float(head_r_mm) * 1e-3, float(L_mm) * 1e-3
    tb = float(np.arccos(np.clip(1.0 - A / (2.0 * np.pi * R * R), -0.95, 1.0)))
    th = (np.arange(n) + 0.5) / n * tb
    na = float(N) * np.pi * float(r_m) ** 2 / A
    q = na * R * (np.cos(np.maximum(th - L / R, 0.0)) - np.cos(th)) / np.sin(th)
    w = np.sin(th)
    return float((q * w).sum() / w.sum()), float(q[-1])


def layer_H(Q_m, kE_pa, rho_g):
    """the analytic Van Wyk layer (m) on a flat patch: H = (3Q/2)^(2/3) (3 kE / (2 rho g))^(1/3)"""
    return (1.5 * float(Q_m)) ** (2.0 / 3.0) * (3.0 * float(kE_pa) / (2.0 * float(rho_g))) ** (1.0 / 3.0)


def scalp_measure(scalp, outside_mm=0.3, min_cos=0.35, min_r=0.05):
    """area (mm2) and area-weighted radius (mm) of the scalp the roots are sampled on (field_collider.FieldScalp's
    samples and their areas, with sample_roots' selection). Mesh-free since 11-14: the triangle cap's measure
    (cap_measure) went with the meshes"""
    V, N, a = np.asarray(scalp.V, np.float64), np.asarray(scalp.VN, np.float64), np.asarray(scalp.area, np.float64)
    v = V - scalp.c
    rv = np.linalg.norm(v, axis=1)
    cosv = np.einsum("ij,ij->i", N, v) / np.maximum(rv, 1e-12)
    ok = (cosv > min_cos) & ~np.asarray(scalp.ear_mask, bool) & (rv > min_r) & (V[:, 2] > scalp.z_min)
    sel = ok & (scalp.d_inside_mm(V) > -outside_mm)
    return float(a[sel].sum() * 1e6), float((a[sel] * rv[sel]).sum() / a[sel].sum() * 1e3)


def check(st, count=None, scalp_area_mm2=None, head_r_mm=None, bands=None):
    """the preflight on a resolved style (dict). count: the strands (default roots.count). -> report dict:
    ok (bool), stops (messages), and the numbers"""
    fcfg = st.get("fibre") or {}
    fb = FB.props(fcfg, st["radius"], st.get("oil") or {})
    tname = str(((fcfg.get("type") or "straight") if not isinstance(fcfg.get("type"), dict)
                 else fcfg["type"].get("base", "straight")))
    grow = st.get("grow") or {}
    G_n = int(grow.get("guides", 3000))
    grid_mm = float(grow.get("grid_mm", 3.0))
    splat_mode = grow.get("splat_mm", "grid")
    N = int(count if count is not None else st["roots"]["count"])
    A = float(scalp_area_mm2 or SCALP_AREA_MM2)
    R = float(head_r_mm or HEAD_R_MM)
    band = dict((bands or BANDS).get(tname, BANDS["straight"]))

    S = float(np.sqrt(A / G_n))
    # the settle's splat (SplatC1, sd 0.866 h): 'grid' h = grid_mm; 'auto' settle.splat_lattice_mm's
    # h = max(grid_mm, splat_sd_frac S / 0.866) (S = splat_spacing_mm, else the guide spacing)
    h_sp = grid_mm
    if splat_mode == "auto":
        frac = float(grow.get("splat_sd_frac", 0.25))
        S_sp = float(grow.get("splat_spacing_mm") or 0.0) or S
        h_sp = max(grid_mm, frac * S_sp / SD_SPLAT_C1)
    sd = h_sp * SD_SPLAT_C1
    infl = lateral_inflation(S, sd)
    infl_sq = tube_inflation(S, sd)                     # the square tube lattice's (the old model), for the record
    k_vw = float(FB.resolve(fcfg)["k_vw"])
    k_eff = fb.k * infl
    rho_g = fb.rho * G * fb.wet_w
    L = mean_length_mm(fcfg, (st.get("length") or {}).get("max_mm"))
    Qm, Qe = column_Q(N, fb.r_root, A, R, L)
    H_type = layer_H(Qm, fb.k * fb.E, rho_g) * 1e3
    H_eff = layer_H(Qm, k_eff * fb.E, rho_g) * 1e3
    H_eff_edge = layer_H(Qe, k_eff * fb.E, rho_g) * 1e3
    f0 = float(fb.bind_factor(0.0))                     # at the follicle: 1 with the ramp (bind.start_mm > 0)
    froot, l_root = root_factor(fb)
    lg = fb.l_g * 1e3
    stand = l_root * 1e3

    stops = []
    if H_eff > band["H_max_mm"]:
        stops.append("Van Wyk layer H %.1f mm over the %s band %.0f mm (k_eff = k_type %.3g x inflation %.2f = %.0f x "
                     "k_vw; at k_type alone H is %.1f mm)" % (H_eff, tname, band["H_max_mm"], fb.k, infl,
                                                               k_eff / k_vw, H_type))
    if stand > band["standoff_max_mm"]:
        stops.append("bound root stand-off %.1f mm over the %s band %.0f mm (l_g %.1f mm x (ramped root factor %.2f)"
                     "^(1/3), %.2f at the follicle: the bound span past the ramp holds the root off, TRANSLATION row "
                     "8)" % (stand, tname, band["standoff_max_mm"], lg, froot, f0))
    return {
        "ok": not stops, "stops": stops, "type": tname, "band": band,
        "strands": N, "guides": G_n, "hairs_per_guide": round(N / G_n, 1),
        "scalp_area_cm2": round(A / 100.0, 1), "head_r_mm": round(R, 1),
        "splat_mm": splat_mode, "splat_lattice_mm": round(h_sp, 2), "guide_spacing_mm": round(S, 2),
        "kernel_sd_mm": round(sd, 3), "spacing_over_sd": round(S / sd, 2),
        "inflation": round(infl, 2), "inflation_model": "measured lateral (TRANSLATION row 6)",
        "inflation_square_lattice": round(infl_sq, 2),
        "k_vw": k_vw, "k_type": fb.k, "k_type_over_k_vw": round(fb.k / k_vw, 3),
        "k_eff": k_eff, "k_eff_over_k_vw": round(k_eff / k_vw, 1),
        "mean_length_mm": round(L, 1), "Q_mean_mm": round(Qm * 1e3, 4), "Q_edge_mm": round(Qe * 1e3, 4),
        "H_type_mm": round(H_type, 1), "H_eff_mm": round(H_eff, 1), "H_eff_edge_mm": round(H_eff_edge, 1),
        "l_g_mm": round(lg, 1), "bind_factor_root": round(froot, 3), "bind_factor_follicle": round(f0, 3),
        "standoff_mm": round(stand, 1),
    }


def format_report(rep):
    """the numbers in a few lines, then the verdict"""
    r = rep
    out = ["PREFLIGHT %s: %d strands / %d guides = %.0f hairs per guide; scalp %.0f cm2, R %.0f mm" % (
               r["type"], r["strands"], r["guides"], r["hairs_per_guide"], r["scalp_area_cm2"], r["head_r_mm"]),
           "  spacing S %.1f mm vs kernel sd %.2f mm (splat %s, lattice %.2f mm; S/sd %.2f) -> measured lateral "
           "inflation x%.2f (square tube lattice x%.2f)" % (
               r["guide_spacing_mm"], r["kernel_sd_mm"], r["splat_mm"], r.get("splat_lattice_mm", 0.0),
               r["spacing_over_sd"], r["inflation"], r.get("inflation_square_lattice", float("nan"))),
           "  k_type %.3g (%.2f x k_vw) -> k_eff %.3g (%.1f x k_vw)" % (
               r["k_type"], r["k_type_over_k_vw"], r["k_eff"], r["k_eff_over_k_vw"]),
           "  Van Wyk layer H: %.1f mm at k_eff (%.1f at k_type; %.1f at the cap edge), band %.0f mm; Q %.3f mm, L %.0f mm"
           % (r["H_eff_mm"], r["H_type_mm"], r["H_eff_edge_mm"], r["band"]["H_max_mm"], r["Q_mean_mm"],
              r["mean_length_mm"]),
           "  root stand-off %.1f mm (l_g %.1f x ramped root factor %.2f^(1/3); %.2f at the follicle), band %.0f mm"
           % (r["standoff_mm"], r["l_g_mm"], r["bind_factor_root"], r.get("bind_factor_follicle", 1.0),
              r["band"]["standoff_max_mm"])]
    out.append("  verdict: PASS" if r["ok"] else "  verdict: STOP\n" + "\n".join("   - " + s for s in r["stops"]))
    return "\n".join(out)
