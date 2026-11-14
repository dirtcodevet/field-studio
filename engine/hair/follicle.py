"""Hair Studio follicles (numpy, Apache-2.0): where each hair leaves the skin, in which direction and how steeply. This
is the head's anatomy. A case sets it once per head (its base style's `follicle` block); a look never changes it.

A hair grows out of an oblique follicle. Two things describe it at the skin:
- the growth direction g: a unit vector in the skin's tangent plane. Over a head it streams out of one whorl behind
  the vertex in a spiral, points forward over the front onto the forehead, turns down at the sides and nape, and is
  bent back toward the ear at the temples (hair.md T2, T3; Bernstein 2005, hair.md R19);
- the exit angle alpha between the skin plane and the hair: flat at the hairline (front 10-20 deg, temples 5-10),
  30-45 over the mid-scalp, 40-55 on the occiput, close to upright near the whorl (hair.md T3; DESIGN_v2 section 2).
The hair's first direction is then d0 = cos(alpha) g + sin(alpha) N. In grow mode the follicle clamps d0: the rod
settle (settle.py) keeps the first segment's direction there, so the root stands at its follicle's angle and nothing
later presses it flat.

Frame (scalp.py): metres; phi = azimuth about the head centre (0 = the face, -y; + = the subject's left, +x),
theta = polar angle from +z, din = mm inside the hairline along the meridian. The subject's right is -x.

    field(scalp, X, N, din, phi, theta, cfg, rng) -> g, alpha (deg), d0, W, report
    whorl(scalp, f)       -> W (m), its skin normal, a report dict
    settings(cfg)         -> the plain follicle values: FOLLICLE_DEFAULTS <- cfg (a style or its 'follicle' block)
    regions(din, phi, theta, dW, f) -> {region: (mask, configured exit angle or None)} for the report and the tests
    FOLLICLE_DEFAULTS     the style block 'follicle.*' in style.py's (default, (lo, hi) or [choices], note) format

DESIGN_v2 section 4.2 is followed. Where it left a choice open, the more physical option was taken:
- the exit-angle jitter is logit-normal on [0, 90] deg, not additive: the angle is bounded by the skin and by the
  normal, so its spread narrows toward both. jitter_deg is the sd at exit_mid; the median is kept exactly. Measured
  spreads grow with the angle (front edge 17.4 (14.3-23.1) deg against 9.6 (8.6-11.3) for curved-blade slits, Ahmad
  2018, hair.md R17). An additive N(0, 4) put about 4 % of temple roots below 3 deg and some into the skin.
- the whorl's right shift is made over the skin (a re-cast ray), and W is where the ray leaves the body's field
  (scalp.ray_exit: field_collider.field_ray_exit, refined by bisection; the triangle hit it once was went with the
  meshes, 11-14), not the nearest sample;
- the occiput value holds fully inside the stated region (|phi| >= 125, theta 80-110) and fades over
  occiput_ramp_deg outside it; the hairline's edge value changes from front to temple to nape across the azimuth
  ranges edge_temple_phi_deg and edge_nape_phi_deg;
- d_W is the straight (chord) distance to W: within the 15-18 mm over which the whorl terms act it differs from the
  distance over the skin by under 1 %.
"""
import copy

import numpy as np

from .scalp import smoothstep

# (default, (lo, hi) or [choices] or None, note) - style.py's format, so the integrator can paste it in as "follicle"
FOLLICLE_DEFAULTS = {
    "whorl_polar_deg": (32.0, (0, 90), "the whorl W: the scalp point on the ray from the head centre at this polar angle "
                                       "from +z. One parietal whorl behind the vertex in 85 % of heads (hair.md T2: "
                                       "Sechi et al. 2020, R14)"),
    "whorl_azimuth_deg": (180.0, (-180, 180), "... at this azimuth (180 = straight back)"),
    "whorl_right_mm": (8.0, (-40, 40), "... then moved this far over the skin to the subject's right (- = left): right "
                                       "of the midline 56 %, left 30 %, midline 14 % (Smith & Gong 1974 via hair.md "
                                       "R14)"),
    "sense": ("cw", ["cw", "ccw"], "the spiral seen from above: clockwise in 70-84 % of single whorls (hair.md T2: R14; "
                                   "Klar 2003, R15)"),
    "swirl_deg": (60.0, (0, 120), "the spiral: the growth direction turned about the skin normal from straight out of W "
                                  "by swirl x exp(-d_W / swirl_mm) (DESIGN_v2 4.2)"),
    "swirl_mm": (15.0, (1, 80), "... fading over this distance from W"),
    "down_sides": ([65.0, 100.0], (0, 180), "the direction bends down at the sides and the nape: + 2 smoothstep(a, b, "
                                            "theta) x the downward tangent (hair.md T3 direction field, Bernstein 2005, "
                                            "R19)"),
    "temple_back": (0.35, (0, 2), "the direction bent back at the temples and above the ears (|phi| 40-130, theta "
                                  "60-85 and below): forward at the front, down and forward at the temple, down and back "
                                  "at the temple apex (hair.md T3, R19; hair_styling.md T3 'Temple and sideburn')"),
    "exit_mid_deg": (35.0, (5, 85), "exit angle (skin plane to hair) over the mid-scalp: 30-45 deg (hair.md T3, clinic "
                                    "sources, W; DESIGN_v2 section 2 table)"),
    "exit_whorl_deg": (70.0, (5, 89), "... at the whorl: close to upright within about 2 cm of it (DESIGN_v2 section 2; "
                                      "hair_styling.md T3 'Growth scan': hair cut short over a whorl stands up)"),
    "whorl_mm": (18.0, (1, 80), "... the whorl's steep angle fades to exit_mid as exp(-d_W / whorl_mm)"),
    "exit_occiput_deg": (45.0, (5, 85), "... on the occiput: 40-55 deg, 53 measured in vivo (Hariri et al. 2020, "
                                        "PMC7602872); posterior mid-scalp 45-60 (hair.md T3, Gharwade 2016, R16)"),
    "occiput_phi_deg": (125.0, (0, 180), "the occiput: |phi| at least this ..."),
    "occiput_theta_deg": ([80.0, 110.0], (0, 180), "... and theta within this range: the full occiput angle inside, "
                                                   "fading over occiput_ramp_deg outside"),
    "occiput_ramp_deg": (10.0, (0.5, 45), "the occiput's fade, in degrees of phi and theta"),
    "exit_edge_deg": {
        "front": (15.0, (1, 60), "at the front hairline: 10-20 deg; measured 17.4 (straight blade) and 9.6 (curved "
                                 "blade) in recipient slits (hair.md T3: Ahmad 2018, R17)"),
        "temple": (10.0, (1, 60), "at the temple edge: 5-10 deg, frontotemporal 10-15 (hair.md T3, clinic source, W)"),
        "nape": (25.0, (1, 60), "at the nape hairline: lower occiput 10-20 deg (hair.md T3, R16) or 15-20; 25 in "
                                "DESIGN_v2 (the nape edge stands more than the temple's)"),
    },
    "edge_band_mm": (25.0, (1, 80), "the edge angle holds at the hairline and blends to the inner value over this many "
                                    "mm inward"),
    "edge_temple_phi_deg": ([30.0, 50.0], (0, 180), "the edge angle changes from front to temple between these azimuths "
                                                    "(the frontotemporal corner)"),
    "edge_nape_phi_deg": ([105.0, 135.0], (0, 180), "... and from temple to nape between these (behind the ear)"),
    "jitter_deg": (4.0, (0, 20), "follicle-to-follicle spread of the exit angle: the sd at exit_mid. Logit-normal on "
                                 "[0, 90] deg, so it narrows toward the skin and the normal and keeps the median "
                                 "(measured spreads grow with the angle, R17)"),
    "azimuth_jitter_deg": (8.0, (0, 45), "follicle-to-follicle spread (sd) of the growth direction about the skin normal"),
    "min_deg": (3.0, (0, 30), "no follicle leaves flatter than this (at 0 deg a hair would lie in the skin); the roots "
                              "held at it are counted in the report (expected 0)"),
    # hair3 (11-11, sheet 17: the follicle direction corrected so that the hair can be swept back off her face
    # naturally to the back and sides). The default field streams straight out of the whorl, so everything
    # ahead of it points forward onto the forehead and long hair falls over the face (sheet 17's veil). A person whose
    # front hair lies back off her face has follicles there that point back and away from her part. 0 = as before.
    "front_back": (0.0, (0, 4), "the growth direction at the front turned BACK over the scalp: + front_back x w x the "
                                "backward tangent (0 = the plain whorl field)"),
    "front_side": (0.0, (0, 4), "... and to the SIDE away from the part line (part_x_mm): + front_side x w x the "
                                "sideways tangent"),
    "front_din_mm": ([0.0, 70.0], (0, 200), "w: full at the hairline, fading out between these mm inside it"),
    "front_phi_deg": ([45.0, 80.0], (0, 180), "... and full within |phi| of the first value, fading out by the second "
                                              "(the front, not the temples, which temple_back already turns)"),
    "top_side": (0.0, (0, 4), "ahead of the whorl on top (theta below 70 deg), the direction also turned away from the "
                              "part line: hair on each side of the part grows toward its own side"),
    "part_x_mm": (0.0, (-60, 60), "the part line the sides split at (+ = the subject's left, as part.offset_mm)"),
}

# the temple term's support, as DESIGN_v2 4.2 writes it: |phi| 40-70 rising, 110-130 falling; theta 60-85 rising
TEMPLE_PHI_IN, TEMPLE_PHI_OUT, TEMPLE_THETA = (40.0, 70.0), (110.0, 130.0), (60.0, 85.0)


def _plain(node):
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    return copy.deepcopy(node[0])


FOLLICLE = _plain(FOLLICLE_DEFAULTS)


def _merge(base, over, pre=""):
    out = copy.deepcopy(base)
    warn = []
    for k, v in over.items():
        if k not in base:
            warn.append("unknown key follicle.%s%s" % (pre, k))
            continue
        if isinstance(base[k], dict):
            out[k], w = _merge(base[k], v, pre + k + ".")
            warn += w
        else:
            out[k] = copy.deepcopy(v)
    return out, warn


def settings(cfg=None):
    """the follicle values: FOLLICLE_DEFAULTS <- cfg. cfg may be a whole style (its 'follicle' block is used; a style
    without one gets the defaults) or the block itself. -> (values dict, warnings)"""
    cfg = cfg or {}
    if isinstance(cfg.get("follicle"), dict):
        blk = cfg["follicle"]
    elif set(cfg) & set(FOLLICLE_DEFAULTS):
        blk = cfg
    else:
        blk = {}
    return _merge(FOLLICLE, blk)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _tangent(v, n):
    """the part of v in the tangent plane of the unit normal n (rows)"""
    v = np.broadcast_to(v, n.shape)
    return v - np.einsum("ij,ij->i", v, n)[:, None] * n


def _turn(v, n, ang):
    """tangent vectors v turned by ang (rad) about the unit normals n; + = counter-clockwise seen from outside"""
    return v * np.cos(ang)[:, None] + np.cross(n, v) * np.sin(ang)[:, None]


def whorl(scalp, f):
    """W: the scalp point on the ray from the head centre at (whorl_polar_deg, whorl_azimuth_deg), moved
    whorl_right_mm over the skin to the subject's right (-x). -> W (m), its unit normal, report dict"""
    c = scalp.c
    th, ph = np.radians(f["whorl_polar_deg"]), np.radians(f["whorl_azimuth_deg"])
    d = np.array([np.sin(th) * np.sin(ph), -np.sin(th) * np.cos(ph), np.cos(th)])
    P0, N0 = scalp.ray_exit(c, d)
    if P0 is None:
        raise ValueError("the whorl ray (polar %.1f, azimuth %.1f deg) misses the scalp" % (f["whorl_polar_deg"],
                                                                                         f["whorl_azimuth_deg"]))
    t_r = _unit(_tangent(np.array([-1.0, 0.0, 0.0]), N0[None])[0])
    P1 = P0 + f["whorl_right_mm"] / 1000 * t_r
    W, NW = scalp.ray_exit(c, P1 - c)
    if W is None:
        W, NW = P0, N0
    ph_w, th_w, r_w = scalp.sph(W[None])
    rep = {"W_mm": (W * 1000).round(2).tolist(), "W_rel_centre_mm": ((W - c) * 1000).round(2).tolist(),
           "phi_deg": round(float(ph_w[0]), 2), "theta_deg": round(float(th_w[0]), 2),
           "moved_mm": round(float(np.linalg.norm(W - P0) * 1000), 2), "normal": NW.round(4).tolist(),
           "sense": f["sense"]}
    return W, NW, rep


def _occiput_weight(aphi, theta, f):
    r = f["occiput_ramp_deg"]
    t0, t1 = f["occiput_theta_deg"]
    p0 = f["occiput_phi_deg"]
    return smoothstep(p0 - r, p0, aphi) * smoothstep(t0 - r, t0, theta) * (1 - smoothstep(t1, t1 + r, theta))


def _edge_value(aphi, f):
    E = f["exit_edge_deg"]
    a0, a1 = f["edge_temple_phi_deg"]
    b0, b1 = f["edge_nape_phi_deg"]
    e = E["front"] + (E["temple"] - E["front"]) * smoothstep(a0, a1, aphi)
    return e + (E["nape"] - e) * smoothstep(b0, b1, aphi)


def exit_profile(din, phi, theta, dW, f):
    """the configured exit angle before the follicle-to-follicle spread (deg): the mid-scalp value, steeper toward the
    whorl, the occiput's value on the occiput, and the hairline's value at the edge"""
    aphi = np.abs(phi)
    a = f["exit_mid_deg"] + (f["exit_whorl_deg"] - f["exit_mid_deg"]) * np.exp(-dW / f["whorl_mm"])
    a = a + _occiput_weight(aphi, theta, f) * (f["exit_occiput_deg"] - a)
    w_e = 1 - smoothstep(0.0, f["edge_band_mm"], din)
    return a + w_e * (_edge_value(aphi, f) - a)


def regions(din, phi, theta, dW, f):
    """the report's regions, each placed where one configured value holds alone (the other terms' weight ~0):
    {name: (mask, configured exit angle in deg, or None where the value is the whorl's profile)}"""
    aphi = np.abs(phi)
    band = f["edge_band_mm"]
    a0, a1 = f["edge_temple_phi_deg"]
    b0, b1 = f["edge_nape_phi_deg"]
    w_occ = _occiput_weight(aphi, theta, f)
    far = dW > 4 * f["whorl_mm"]          # the whorl term under 2 % of (exit_whorl - exit_mid)
    E = f["exit_edge_deg"]
    return {
        "front_hairline": ((din < 4) & (aphi < a0 - 5), E["front"]),
        "temples": ((din < 4) & (aphi > a1 + 5) & (aphi < b0 - 5), E["temple"]),
        "mid_scalp": ((din > band + 5) & far & (w_occ <= 0), f["exit_mid_deg"]),
        "crown_whorl": (dW < 10.0, None),
        "occiput": ((din > band) & far & (w_occ >= 0.999), f["exit_occiput_deg"]),
        "nape": ((din < 4) & (aphi > b1 + 10), E["nape"]),
    }


def field(scalp, X, N, din=None, phi=None, theta=None, cfg=None, rng=None):
    """the follicle at every root.
    X (n, 3) roots (m), N (n, 3) skin normals (outward), din (mm), phi, theta (deg) as scalp.d_inside_mm / scalp.sph
    give them (computed when None), cfg a style or its follicle block, rng a numpy Generator (two normal draws of n).
    -> g (n, 3) unit growth directions in the tangent plane, alpha (n,) exit angles (deg), d0 (n, 3) unit start
       directions cos(alpha) g + sin(alpha) N, W (3,) the whorl (m), report dict"""
    f, warn = settings(cfg)
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.asarray(X, np.float64)
    N = _unit(np.asarray(N, np.float64))
    n = len(X)
    if din is None:
        din = scalp.d_inside_mm(X)
    if phi is None or theta is None:
        phi, theta, _ = scalp.sph(X)
    din, phi, theta = (np.asarray(a, np.float64) for a in (din, phi, theta))
    aphi = np.abs(phi)
    W, NW, wrep = whorl(scalp, f)
    dW = np.linalg.norm(X - W, axis=1) * 1000

    # the growth direction: straight out of W along the skin, turned into the spiral near W
    out = _tangent(X - W, N)
    lo = np.linalg.norm(out, axis=1) < 1e-9                 # a root on W itself: the direction there is undefined
    if lo.any():
        out[lo] = _tangent(np.array([0.0, 1.0, 0.0]), N[lo])
    g = _unit(out)
    sgn = -1.0 if f["sense"] == "cw" else 1.0               # cw seen from outside = a negative turn about the normal
    g = _turn(g, N, sgn * np.radians(f["swirl_deg"]) * np.exp(-dW / f["swirl_mm"]))
    # ... bent down at the sides and nape, and back at the temples
    kd = 2.0 * smoothstep(f["down_sides"][0], f["down_sides"][1], theta)
    kt = f["temple_back"] * smoothstep(*TEMPLE_PHI_IN, aphi) * (1 - smoothstep(*TEMPLE_PHI_OUT, aphi)) * \
        smoothstep(*TEMPLE_THETA, theta)
    g = g + kd[:, None] * _tangent(np.array([0.0, 0.0, -1.0]), N) + kt[:, None] * _tangent(np.array([0.0, 1.0, 0.0]), N)
    g = _unit(_tangent(g, N))
    # hair3: the front and the top ahead of the whorl grow back and away from the part (FOLLICLE_DEFAULTS front_*)
    if f["front_back"] > 0 or f["front_side"] > 0 or f["top_side"] > 0:
        wf = (1 - smoothstep(f["front_din_mm"][0], f["front_din_mm"][1], din)) * \
            (1 - smoothstep(f["front_phi_deg"][0], f["front_phi_deg"][1], aphi))
        side = np.where(X[:, 0] >= f["part_x_mm"] / 1000.0, 1.0, -1.0)
        ts = _tangent(side[:, None] * np.array([1.0, 0.0, 0.0]), N)
        tb = _tangent(np.array([0.0, 1.0, 0.0]), N)
        wt = (1 - smoothstep(55.0, 75.0, theta)) * (X[:, 1] < W[1]).astype(float)
        g = g + (wf * f["front_back"])[:, None] * tb + (wf * f["front_side"] + wt * f["top_side"])[:, None] * ts
        g = _unit(_tangent(g, N))
    # the follicle-to-follicle spread: direction about the normal, then the exit angle
    z_az = rng.standard_normal(n)
    z_al = rng.standard_normal(n)
    g = _unit(_tangent(_turn(g, N, np.radians(f["azimuth_jitter_deg"]) * z_az), N))

    # the exit angle: the regional profile, then a logit-normal spread on [0, 90] deg (median kept)
    a0 = exit_profile(din, phi, theta, dW, f)
    p = np.clip(a0 / 90.0, 1e-6, 1 - 1e-6)
    pm = f["exit_mid_deg"] / 90.0
    s = f["jitter_deg"] / (90.0 * pm * (1 - pm))            # d(alpha)/d(logit p) = 90 p (1 - p) deg
    alpha = 90.0 / (1 + np.exp(-(np.log(p / (1 - p)) + s * z_al)))
    floored = alpha < f["min_deg"]
    alpha = np.maximum(alpha, f["min_deg"])
    al = np.radians(alpha)
    d0 = np.cos(al)[:, None] * g + np.sin(al)[:, None] * N

    rep = {"whorl": wrep, "roots": int(n), "warnings": warn,
           "alpha_p1_10_50_90_99": np.percentile(alpha, [1, 10, 50, 90, 99]).round(2).tolist(),
           "below_3deg_share": float(np.mean(alpha < 3.0)), "floored_to_min": int(floored.sum()),
           "regions": {}}
    for name, (m, conf) in regions(din, phi, theta, dW, f).items():
        if not m.any():
            rep["regions"][name] = {"n": 0}
            continue
        rep["regions"][name] = {"n": int(m.sum()), "median_deg": round(float(np.median(alpha[m])), 2),
                                "configured_deg": round(float(conf if conf is not None else np.median(a0[m])), 2),
                                "p10_p90_deg": np.percentile(alpha[m], [10, 90]).round(2).tolist()}
    return g, alpha, d0, W, rep
