"""Hair Studio fibre (numpy, Apache-2.0): what one hair is made of, and how long each hair has grown.

A scalp hair is a slender keratin rod. Its shape under gravity comes from three numbers per unit length - how hard it
is to bend (EI), how much it weighs (w) and how much room it takes (a) - and the mass of hair pushes back when it is
squeezed (the fibre-mass law P = k E phi^3). Every hair is also at its own point of its growth cycle, so the lengths on
one head are out of step: that is the natural thinning toward the ends and the short new hairs of the halo. This
module holds those causes and nothing about a look (DESIGN_v2 section 4.3).

    props(cfg, radius_cfg, oil=0.0) -> Fibre
        .r(s, L, scale), .EI(s, L, scale), .w(s, L, scale), .a(s, L, scale)   per fibre, at arc s of a hair of grown
                                                                             length L (both m, broadcast)
        .hang(s, L, scale)   weight of the hair below s (N)  ->  .l_T(s, L, scale) = sqrt(EI / hang), the bend length
                             under that tension (m)
        .adhesive_load(s, phi) -> N/m   the normal load per metre of fibre the oil's liquid bridges press it onto
                             its neighbours with, in a mass of fibre fraction phi (0 without the oil block)
        .E, .k, .kE, .c_p, .mu (hair on hair), .mu_skin (hair on skin), .kappa0, .l_g, .adhesion, .report()
    cycle_lengths(n, cfg, rng, max_mm=None, return_phase=False) -> grown length per hair (m) [, resting flag]

Units: the style file holds mm, GPa and months (as style.py does); everything returned is SI (m, N, N m^2, Pa).
`cfg` is the style's `fibre` block (FIBRE_DEFAULTS fill what it leaves out), `radius_cfg` the `radius` block.

Sources (keys in engine/packs/human/research/hair.md and hair_styling.md, and DESIGN_v2 section 2's constants table):
- E 4 GPa dry at ~65 % RH, ~2 GPa wet: Wortmann et al. 2022; Goldstein, Warren & Ball 2012 (arXiv 1204.0371).
  hair.md T8: 3.1 +- 0.16 GPa (Sayahi 2016, humidity not stated).
- rho 1300 kg/m3: Goldstein et al. 2012 (hair.md T8 assumes 1320).
- radius 27-40 um, kept to the end of the hair: Lasisi et al. 2016; hair.md T4.
- natural curvature 0.042 +- 0.02 /mm European, 'straight' ~0.039: Lasisi et al. 2016; Goldstein et al. 2012.
- P = k E phi^3, k ~1e-5 at kappa0 0.039 /mm: Goldstein, Warren & Ball 2012, PRL 108:078101 (the Van Wyk law).
- friction 0.1-0.3 (0.13 dry, 0.25 wet): Fair & Gupta 1982; tribology.rs 2025.
- growth 0.36-0.40 mm/day, anagen 2-7 y, telogen 10-14 %: hair.md T5 (Loussouarn 2001, Wolfram 2003), D7, D8.
- water taken up by virgin hair ~30 % of its weight: hair_styling.md T3 (Robbins, via P29).
- liquid bridges: a meniscus between two crossed fibres of radius r pulls them together with ~4 pi gamma r cos(theta)
  (Israelachvili, Intermolecular and Surface Forces, 3rd ed., ch. 17: crossed cylinders of equal radius act as a sphere
  of that radius on a plane). Friction carries the external load plus this adhesive load (Derjaguin 1934). Sebum gamma
  ~30 mN/m. 1 % synthetic sebum raised fibre withdrawal force ~8x loose, 2-3x dense (hair_styling P58).
- contacts per length of fibre in a mass of fibre fraction phi: 8 phi f / (pi d), f the mean |sin| of the crossing
  angle (Toll 1998, Polym. Eng. Sci. 38:1337; f = pi / 4, isotropic, gives van Wyk's ~2 phi / d).

The adhesive load, worked (fine hair r = 27 um, gamma = 0.03 N/m, cos(theta) = 1): one fully wetted bridge pulls with
4 pi x 0.03 x 27e-6 = 1.02e-5 N, the weight of ~340 mm of that hair (rho g pi r^2 = 1300 x 9.81 x 2.29e-9 = 2.9e-5 N/m).
Per metre of fibre, n_c F_b = (8 phi f / (2 pi r)) (4 pi gamma r w) = 16 f gamma phi w (the radius cancels): at phi
0.02, f 0.2, wetting w 0.6 (day 2 near the root) 1.15e-3 N/m, ~40 x the fibre's own weight per length. Fresh hair: 0.
"""
import numpy as np

G = 9.81                    # m/s2
WATER_UPTAKE = 0.30         # soaked virgin hair carries ~30 % of its dry weight in water (bleached ~45 %), hair_styling T3
KAPPA_REF_MM = 0.039        # the curvature Goldstein's k ~1e-5 was fitted at, 1/mm
DAYS_PER_MONTH = 365.25 / 12
BRIDGE_COS = 1.0            # contact angle of sebum / oils on hair ~0: they spread on the lipid-covered cuticle
BIND_START_MM = 20.0        # (d) the arc from the root over which the hairs' own follicle paths converge into a lock

# The `fibre.*` block in style.py's format: (default, (lo, hi) or None, note).
FIBRE_DEFAULTS = {
    "E_gpa": (4.0, (0.5, 10.0), "Young's modulus of the dry fibre at room humidity (~65 % RH), GPa: 4 (Wortmann et al. "
                                "2022; Goldstein et al. 2012); hair.md T8 gives 3.1 +- 0.16 (humidity not stated). "
                                "Bending stiffness EI = E pi r^4 / 4"),
    "rho": (1300.0, (1100.0, 1500.0), "fibre density, kg/m3 (Goldstein et al. 2012; hair.md T8 assumes 1320). Weight per "
                                      "length w = rho g pi r^2"),
    "taper_mm": (30.0, (0.0, 200.0), "an uncut hair keeps its radius until this far from its natural tip, then thins to "
                                     "radius.tip_mm (the tip grew first, while the follicle was still ramping up). "
                                     "Scissors remove it: a cut hair ends blunt. 30 mm is DESIGN_v2's value (13.4); no "
                                     "measurement found (hair.md section 4)"),
    "kappa0_mm": (0.04, (0.0, 0.5), "the fibre's natural curvature, rms, 1/mm: European 0.042 +- 0.02, 'straight' ~0.039 "
                                    "(Lasisi et al. 2016; Goldstein et al. 2012). Wavier fibres resist packing harder: "
                                    "k x (kappa0 / 0.039)^2"),
    "k_vw": (1e-5, (1e-7, 1e-3), "Van Wyk constant of the fibre-mass law P = k E phi^3, at kappa0 0.039 /mm (Goldstein, "
                                 "Warren & Ball 2012). Uncertain: measure once per hair on one reference, then freeze it "
                                 "for every look on that hair (DESIGN_v2 12)"),
    "friction": (0.25, (0.0, 1.0), "hair-on-hair friction coefficient: 0.1-0.3 measured, 0.13 dry and 0.25 wet (Fair & "
                                   "Gupta 1982; tribology.rs 2025); 0.25 is DESIGN_v2's default and the prototypes' tested "
                                   "value. Grit raises it, conditioner and silicone lower it (hair_styling.md 7.1 rows "
                                   "23-24)"),
    "friction_skin": (0.5, (0.0, 1.5), "hair-on-SKIN friction coefficient, for the load the body's contact carries "
                                       "(settle.State.lag): dry skin against fibres and textiles 0.27-0.71, typically "
                                       "~0.5 (Derler, Schrade & Gerhardt 2007; Derler & Gerhardt 2012 review); it rises "
                                       "26-43 % from dry to normally moist skin (Gerhardt et al. 2008). The hair-hair "
                                       "value (friction, oiled cuticle on cuticle) is about half of it"),
    "adhesion": (True, None, "the oil's liquid bridges press touching fibres together and friction carries that load "
                             "too (Derjaguin's two-term law): second-day hair holds a laid route, freshly washed hair "
                             "slides. Needs the style's oil block passed whole to props(); a bare amount gives none"),
    "cohesion": (True, None, "the same bridges also pull the mass together: the hair-hair free energy gains -c_a phi_w^2 "
                             "(phi_w the fibre fraction weighted by each fibre's wetting o / (o + load_half)), c_a = 2 "
                             "gamma cos(theta) / r, so the mass rests where the bridges balance its swelling pressure "
                             "(phi* = c_a w^2 / (2 c_p)). Needs the oil block as for adhesion; no oil, no attraction"),
    "cohesion_count": ("crossing", ["crossing", "aligned"], "how the bridges are counted in the settle's free energy. "
                                    "crossing: Toll's crossing contacts of a random mat, f = c_p phi^3 - c_a phi_w^2 "
                                    "(phi* = c_a w^2 / (2 c_p)). aligned: each sample's share in line contacts is the "
                                    "local alignment A of the fibre (the order parameter of the splatted direction "
                                    "tensor); that share follows the locks' law (form.py: P = k E phi^3 - (W / pi r^2) "
                                    "phi, phi_lock = sqrt(W / (pi r^2 k E))), the rest the crossing count, so the settle "
                                    "and the locks share one law"),
    "crossing": (0.2, (0.01, 1.0), "f, the mean |sin| of the angle neighbouring hairs cross at, in Toll's contact count "
                                   "8 phi f / (pi d): pi/4 for an isotropic felt, ~0.2 for combed hair whose neighbours "
                                   "run within ~10-15 deg (a choice, not a measurement)"),
    "wet": (0.0, (0.0, 1.0), "0 dry .. 1 soaked: E x (1 - 0.5 wet) (about 2 GPa wet, Wortmann et al. 2022), weight x "
                             "(1 + 0.30 wet) for the water the fibre takes up (hair_styling.md T3)"),
    "cycle": {
        "on": (True, None, "growth-cycle lengths: each follicle is at its own point of its cycle, so the hairs are out of "
                           "step - thinner toward the ends, and a share of short new hairs (the halo). Off: every hair "
                           "at the end of its growth phase"),
        "rate_mm_month": (12.0, (3.0, 35.0), "growth, mm per month (30.4 d): 12 = 0.39 mm/day. Caucasian 0.396 +- 0.055 "
                                             "mm/day (Loussouarn 2001), 0.367 +- 0.056 (hair.md T5 R10), textbook 1 cm "
                                             "per 28 d; African descent 0.26-0.28"),
        "anagen_years": ([3.0, 7.0], (0.1, 40.0), "the growth phase per follicle, years, uniform between these: 2-7 "
                                                  "(2-8) (hair.md T5), 3-5 (Wolfram 2003). Terminal length = rate x "
                                                  "anagen: 0.39 mm/day x 7 y ~ 1 m (hair.md T5: 91-101 cm)"),
        "telogen_frac": (0.10, (0.0, 0.5), "share of follicles resting; a resting hair keeps its final length: ~10 % "
                                           "(Wolfram 2003), 10-14 % (hair.md D8), Caucasian 14 +- 11 % (Loussouarn "
                                           "2001)"),
    },
    "k_curvature": ("kappa0", ["kappa0", "type"], "which curvature scales the Van Wyk k (k = k_vw (kappa / 0.039 /mm)^2, "
                                                  "Goldstein, Warren & Ball 2012: k grows with the fibre's rms natural "
                                                  "curvature). kappa0: fibre.kappa0_mm (today's). type: the hair type's "
                                                  "rest helix when it has one, rms over the helix kappa sqrt((1 + flat^2) "
                                                  "/ 2) - a round curly coil (CD 10 mm) 0.2 /mm, k x 26; no rest curl "
                                                  "(straight): kappa0_mm as before"),
    "bind": {
        "on": (False, None, "BINDING: hairs bound into a lock / ringlet bend like one thicker rod. A guide of n hairs as "
                            "n / m locks of m hairs bonded to degree beta has B = n m^beta EI (loose beta 0: n EI, the "
                            "sum of the members, Euler-Bernoulli; fully bonded beta 1: n m EI, a lock of m^2 EI - the "
                            "parallel-axis theorem gives m^2 / phi EI for a round bundle packed at phi, m^2 taken; "
                            "between them the partial-interaction composite beam, Newmark, Siess & Viest 1951, whose "
                            "stiffness rises with the interface's shear bond; m^beta is a design interpolation (d) on "
                            "the log scale). The weight is the members' sum either way, so a bound lock deflects m^beta "
                            "less under its own weight and stands off the head and its neighbours. Off: today's B"),
        "m": (None, (1.0, 2000.0), "hairs per bonded lock; None: the hair type's lock_m (HAIR_TYPES), 1 without a type"),
        "beta": (None, (0.0, 1.0), "the bond degree from the fibre's own shape (the coils' interlock); None: the type's "
                                   "bond (straight 0, wavy 0.1, curly 0.35 - design values, d)"),
        "oil_beta": (0.4, (0.0, 1.0), "the bond the liquid bridges / product add at full wetting (the oil block's load, "
                                      "Fibre.wetting(s), needs fibre.adhesion): beta(s) = 1 - (1 - beta)(1 - oil_beta "
                                      "w(s)) - two independent bond mechanisms (d). A wetting share of 1 bonds a "
                                      "capillary-zipped straight lock (Bico et al. 2004) to beta 0.4"),
        "start_mm": (None, (0.0, 500.0), "the bond ramps in from the root over this arc (TRANSLATION row 8, piece 2): "
                                         "near the scalp each hair leaves its own follicle (one clamp per hair, n EI), "
                                         "they bind into locks further down, so factor(s) = 1 + (m^beta(s) - 1) "
                                         "smoothstep(s / start) - the shape bond and the oil bond alike. None: the "
                                         "type's start_mm + ramp_mm (where its curl is full) when over 0, else 20 mm "
                                         "(d). 0: the bond from s = 0 (the old clamp at the follicle)"),
    },
    "type": (None, None, "the hair type - the fibre's genetic characteristic (HAIR_TYPES, hair_type): 'straight' | "
                         "'wavy' | 'curly', or {\"base\": name, trait: value} overriding cd_mm (curve diameter) or "
                         "kappa_mm, curl_index or tau_mm, flat, ellipticity, diameter_um, start_mm, ramp_mm, cd_jitter. "
                         "Its rest helix rides the settle's rest turning (settle.rest_waves). None / 'straight' with "
                         "its defaults: today's fibre (no rest curl)"),
}

# DESIGN_v2 section 7 spells three keys differently; they are read when the house names are absent.
_ALIASES = {"rho": "density"}
_CYCLE_ALIASES = {"telogen_frac": "telogen"}


def _plain(node):
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    return node[0]


def resolve(cfg=None):
    """the fibre block with every key filled from FIBRE_DEFAULTS (plain values, the file's units)"""
    cfg = dict(cfg or {})
    out = _plain(FIBRE_DEFAULTS)
    for k in out:
        if k in ("cycle", "bind"):
            continue
        if k in cfg:
            out[k] = cfg[k]
        elif _ALIASES.get(k) in cfg:
            out[k] = cfg[_ALIASES[k]]
    bd = dict(cfg.get("bind") or {})
    for k in out["bind"]:
        if k in bd:
            out["bind"][k] = bd[k]
    cy = dict(cfg.get("cycle") or {})
    for k in out["cycle"]:
        if k in cy:
            out["cycle"][k] = cy[k]
        elif _CYCLE_ALIASES.get(k) in cy:
            out["cycle"][k] = cy[_CYCLE_ALIASES[k]]
    if "rate_mm_month" not in cy and "rate_mm_day" in cy:
        out["cycle"]["rate_mm_month"] = float(cy["rate_mm_day"]) * DAYS_PER_MONTH
    return out


class Fibre:
    """One hair's material along its arc. s and L are metres and broadcast against each other; L is the hair's GROWN
    length (cycle_lengths), not its cut length, because the taper sits at the natural tip and the scissors take it off.
    `scale` multiplies the radius (the caller's per-hair jitter, edge hairs' radius.edge_scale). Past the tip (s > L)
    the radius is 0, so sums over a guide's members drop the hairs that have ended (DESIGN_v2 4.4: n_i(s))."""

    def __init__(self, E, rho, r_root, r_tip, taper, wet_w, k, mu, kappa0, mu_skin=None):
        self.E = float(E)                    # Pa
        self.rho = float(rho)                # kg/m3
        self.r_root = float(r_root)          # m
        self.r_tip = float(r_tip)            # m
        self.taper = float(taper)            # m
        self.wet_w = float(wet_w)            # weight factor from absorbed water
        self.k = float(k)                    # Van Wyk constant (dimensionless)
        self.kE = self.k * self.E            # Pa: P = kE phi^3
        self.c_p = 0.5 * self.kE             # Pa: the settle's free-energy density f(phi) = c_p phi^3 gives
        #                                      P = phi f' - f = 2 c_p phi^3 = k E phi^3 (DESIGN_v2 4.5)
        self.mu = float(mu)                  # hair-on-hair friction
        self.mu_skin = float(mu if mu_skin is None else mu_skin)   # hair-on-skin friction
        self.kappa0 = float(kappa0)          # 1/m, rms natural curvature (the phase-2 rest waves use it)
        self.EI_root = self.E * np.pi * self.r_root ** 4 / 4
        self.w_root = self.rho * G * self.wet_w * np.pi * self.r_root ** 2
        # gravito-bending length: over about this span a hair holds its own direction against its weight
        # (EI / w)^(1/3) = (E r^2 / 4 rho g)^(1/3); it scales as radius^(2/3)
        self.l_g = (self.EI_root / self.w_root) ** (1.0 / 3.0)
        self.adhesion = None                 # the oil state (dict) when the liquid bridges are on (props sets it)
        self.crossing = 0.2                  # f, the mean |sin| of the crossing angle (fibre.crossing)
        self.cohesion = False                # the bridges' attraction in the settle's free energy (props sets it)
        self.cohesion_count = "crossing"     # "crossing" (Toll) or "aligned" (line contacts by the local alignment)
        self.hair = None                     # the hair type's traits (hair_type; props sets it): the rest helix
        self.ellipticity = None              # cross-section a/b of the type (stored; not in the settle's bending yet)
        self.bind = None                     # the binding (fibre.bind on: {m, beta, oil_beta}; props sets it)

    def bind_factor(self, s):
        """the guide's stiffness factor from binding at arc s (m): 1 + (m^beta(s) - 1) smoothstep(s / start), beta(s) =
        1 - (1 - beta)(1 - oil_beta w(s)) (fibre.bind). The ramp takes the shape bond and the oil bond in together:
        at the follicle every hair is its own clamp (factor 1), past `start` the lock is bound (m^beta(s)). start 0:
        m^beta(s) from s = 0. 1 when the binding is off"""
        s = np.asarray(s, np.float64)
        b = self.bind
        if b is None:
            return np.ones(s.shape)
        beta = 1.0 - (1.0 - b["beta"]) * (1.0 - b["oil_beta"] * self.wetting(s))
        F = float(b["m"]) ** np.clip(beta, 0.0, 1.0)
        start = float(b.get("start_m", 0.0))
        if start <= 0.0:
            return F
        x = np.clip(s / start, 0.0, 1.0)
        return 1.0 + (F - 1.0) * (x * x * (3.0 - 2.0 * x))

    def wetting(self, s):
        """the share of the fibre's half perimeter the oil wets at arc s (m), o / (o + load_half) with o the oil load
        (form.oil_load): 0 without the oil state"""
        s = np.asarray(s, np.float64)
        a = self.adhesion
        if a is None:
            return np.zeros(s.shape)
        from . import strand_util as FO              # P14 (11-09): the oil model, out of form.py
        o = np.maximum(FO.oil_load(s, a["days_since_wash"], a["product"], a["sebum_mm_per_day"], a["sebum_load"]), 0.0)
        return o / (o + a["load_half"])

    def cohesion_c0(self):
        """c_a at full wetting (Pa): the bridges' adhesion energy per volume of the mass is c_a0 phi_w^2.
        Contacts per volume (Toll 1998, as adhesive_load): fibre length per volume phi / (pi r^2) x crossings per length
        8 phi f / (pi 2 r) / 2 (each shared by two fibres) = 2 f phi^2 / (pi^2 r^3). A bridge at a crossing at angle a
        wets the overlap of the two fibres' wetted strips, each pi r w wide: area (pi r w)^2 / sin(a), so its adhesion
        work is gamma cos(theta) (pi r)^2 w_i w_j / sin(a). The count carries sin(a) (f is its mean), the area 1 / sin(a):
        the angle cancels, and the energy per volume is 2 gamma cos(theta) phi_w^2 / r, phi_w = sum of phi x w. 0 when
        the oil or the cohesion is off"""
        a = self.adhesion
        if a is None or not self.cohesion:
            return 0.0
        if a["product"] <= 0 and (a["days_since_wash"] <= 0 or a["sebum_load"] <= 0):
            return 0.0                                  # freshly washed, nothing on it: no bridges anywhere
        return 2.0 * self.adhesion["gamma_mN_m"] / 1000.0 * BRIDGE_COS / self.r_root

    def phi_star(self, s):
        """the packing where the bridges balance the swelling pressure: f(phi) = c_p phi^3 - c_a phi^2 with c_a =
        c_a0 w(s)^2 gives P = phi f' - f = 2 c_p phi^3 - c_a phi^2 = 0 at phi* = c_a / (2 c_p); 0 with no oil"""
        return self.cohesion_c0() * self.wetting(s) ** 2 / (2.0 * self.c_p)

    def cohesion_sigma(self):
        """sigma_0 (Pa): the line bridges' stress at full wetting, W / (pi r^2) with W = gamma cos(theta) pi r (form.py's
        adhesion_per_length at full wetting) = gamma cos(theta) / r = c_a0 / 2. Combed hair is aligned: neighbours touch
        along lines, and a bridge along a contact line pulls with gamma x the wetted width (elastocapillary coalescence,
        Bico, Roman, Moulin & Boudaoud, Nature 432:690, 2004; hair_styling P43). 0 when c_a0 is"""
        return 0.5 * self.cohesion_c0()

    def phi_lock(self, s):
        """the locks' packing (form.py lock_causes): sqrt(W / (pi r^2 k E)) = sqrt(sigma_0 w(s) / kE), at r_root. The
        settle's aligned count reproduces it at full alignment (phi_star_mixed with A = 1)"""
        return np.sqrt(self.cohesion_sigma() * self.wetting(s) / self.kE)

    def phi_star_mixed(self, s, A):
        """the packing where P = 2 c_p phi^3 - c_a0 (1 - A)^2 w^2 phi^2 - sigma_0 A w phi = 0 (the settle's free energy
        with a share A of the fibre in line contacts): phi* = (B + sqrt(B^2 + 8 c_p C)) / (4 c_p), B = c_a0 (1 - A)^2
        w^2, C = sigma_0 A w. A = 0 gives phi_star, A = 1 phi_lock"""
        w = self.wetting(s)
        A = np.asarray(A, np.float64)
        B = self.cohesion_c0() * (1.0 - A) ** 2 * w * w
        C = self.cohesion_sigma() * A * w
        return (B + np.sqrt(B * B + 8.0 * self.c_p * C)) / (4.0 * self.c_p)

    def adhesive_load(self, s, phi):
        """the normal load per metre of fibre (N/m) that the oil's liquid bridges press a hair onto its neighbours
        with, at arc s (m) in a mass of fibre fraction phi: contacts per length 8 phi f / (pi 2 r) (Toll 1998) x each
        bridge's pull 4 cos(theta) W, where W = gamma pi r o / (o + load_half) is form.py's bridge adhesion per length
        (one oil state drives the locks and this friction) and o the oil load at s (product + sebum crept down
        days x sebum_mm_per_day). cos(theta) = 1. The radius cancels: 16 f gamma cos(theta) phi o / (o + load_half)."""
        s, phi = np.broadcast_arrays(np.asarray(s, np.float64), np.asarray(phi, np.float64))
        a = self.adhesion
        if a is None:
            return np.zeros(s.shape)
        from . import strand_util as FO              # the oil model (strand_util.py since P14, 11-09; was form.py)
        o = FO.oil_load(s, a["days_since_wash"], a["product"], a["sebum_mm_per_day"], a["sebum_load"])
        W = FO.adhesion_per_length(o, self.r_root, a["gamma_mN_m"] / 1000.0, a["load_half"])
        n_c = 8.0 * np.maximum(phi, 0.0) * self.crossing / (np.pi * 2.0 * self.r_root)
        return n_c * 4.0 * BRIDGE_COS * W

    def r(self, s, L, scale=1.0):
        """fibre radius (m): constant to within `taper` of the natural tip, then linear down to the tip radius. A hair
        shorter than the taper is still in its first weeks of growth and is thinner at the root as well"""
        s = np.asarray(s, np.float64)
        L = np.asarray(L, np.float64)
        u = L - s                                           # distance to the natural tip
        f = np.clip(u / self.taper, 0.0, 1.0) if self.taper > 0 else np.ones(np.broadcast(s, L).shape)
        r = (self.r_tip + (self.r_root - self.r_tip) * f) * scale
        return np.where(u >= 0, r, 0.0)

    def a(self, s, L, scale=1.0):
        """fibre cross-section (m2): the room it takes in the hair mass (the dry area; wet swelling is not modelled)"""
        return np.pi * self.r(s, L, scale) ** 2

    def EI(self, s, L, scale=1.0):
        """bending stiffness (N m2) = E pi r^4 / 4, a round fibre"""
        return self.E * np.pi * self.r(s, L, scale) ** 4 / 4

    def w(self, s, L, scale=1.0):
        """weight per length (N/m) = rho g pi r^2, plus the water a wet fibre carries"""
        return self.rho * G * self.wet_w * self.a(s, L, scale)

    def hang(self, s, L, scale=1.0):
        """weight of the hair below arc s (N), exact for the taper: u = distance from the tip, r(u) = r_tip + c u over
        the taper, so the integral of r^2 over [0, U] is r_tip^2 U + r_tip c U^2 + c^2 U^3 / 3, then r_root^2 beyond"""
        s = np.asarray(s, np.float64)
        L = np.asarray(L, np.float64)
        U = np.maximum(L - s, 0.0)
        rt, rr, tp = self.r_tip, self.r_root, self.taper
        if tp > 0:
            c = (rr - rt) / tp
            U1 = np.minimum(U, tp)
            I = rt * rt * U1 + rt * c * U1 ** 2 + c * c * U1 ** 3 / 3 + rr * rr * np.maximum(U - tp, 0.0)
        else:
            I = rr * rr * U
        return self.rho * G * self.wet_w * np.pi * I * np.asarray(scale, np.float64) ** 2

    def l_T(self, s, L, scale=1.0):
        """bend length under the tension of the hair hanging below (m): sqrt(EI / T); inf at the tip (no tension).
        Above about this length from a bend the hair is pulled straight by its own hanging length"""
        EI, T = np.broadcast_arrays(self.EI(s, L, scale), self.hang(s, L, scale))
        out = np.full(EI.shape, np.inf)
        np.divide(EI, T, out=out, where=T > 0)
        return np.sqrt(out)

    def report(self):
        """the material in the units a person reads"""
        out = {"E_gpa": self.E / 1e9, "rho": self.rho, "r_root_um": self.r_root * 1e6, "r_tip_um": self.r_tip * 1e6,
               "taper_mm": self.taper * 1000, "wet_weight_x": self.wet_w, "EI_root_Nm2": self.EI_root,
               "w_root_N_per_m": self.w_root, "l_g_mm": self.l_g * 1000, "k": self.k, "kE_pa": self.kE,
               "friction": self.mu, "kappa0_mm": self.kappa0 / 1000}
        if self.bind is not None:                       # only when on: earlier reports unchanged
            out["bind"] = dict(self.bind, factor_root=float(self.bind_factor(0.0)))
        return out


def oil_state(oil):
    """the style's `oil` block filled from form.OIL_DEFAULTS (days_since_wash, product, sebum_mm_per_day, sebum_load,
    gamma_mN_m, load_half): the one oil state the locks and the friction both read"""
    from . import strand_util as FO                  # P14 (11-09): OIL_DEFAULTS, out of form.py
    d = {k: v[0] for k, v in FO.OIL_DEFAULTS.items()}
    d.update({k: float(oil[k]) for k in d if k in oil})
    return d


def props(cfg, radius_cfg, oil=0.0):
    """the fibre's material from the style's `fibre` block (None = defaults), its `radius` block (root_mm, tip_mm) and
    the oil: its amount (oil.amount, 0..1), or the style's whole `oil` block (dict), which with fibre.adhesion on also
    gives the fibre its liquid bridges (Fibre.adhesive_load). A bare amount gives none (the old behaviour).
    E = E_gpa (1 - 0.5 wet). k = k_vw (kappa0 / 0.039 mm^-1)^2 (1 - 0.5 oil): wavier fibres push apart harder, oil lets
    them pack (DESIGN_v2 4.3)."""
    f = resolve(cfg)
    block = oil if isinstance(oil, dict) else None
    amount = float(block.get("amount", 0.0)) if block is not None else float(oil)
    wet = float(f["wet"])
    E = float(f["E_gpa"]) * 1e9 * (1.0 - 0.5 * wet)
    # the hair type (fibre.type, HAIR_TYPES): its rest helix for the settle, its section, its diameter when given
    hair = hair_type((cfg or {}).get("type"))
    kap_mm = float(f["kappa0_mm"])
    if str(f["k_curvature"]) == "type" and hair is not None and hair["curl"]:
        # the rms of the helix's curvature vector kappa (cos th, flat sin th) over a turn of th
        kap_mm = hair["kappa"] / 1000.0 * np.sqrt((1.0 + float(hair["flat"]) ** 2) / 2.0)
    k = float(f["k_vw"]) * (kap_mm / KAPPA_REF_MM) ** 2 * (1.0 - 0.5 * amount)
    r_root = float(radius_cfg["root_mm"]) / 1000
    if hair is not None and hair.get("diameter_um"):
        r_root = float(hair["diameter_um"]) / 2e6
    fb = Fibre(E=E, rho=float(f["rho"]), r_root=r_root,
               r_tip=float(radius_cfg["tip_mm"]) / 1000, taper=float(f["taper_mm"]) / 1000,
               wet_w=1.0 + WATER_UPTAKE * wet, k=k, mu=float(f["friction"]), kappa0=float(f["kappa0_mm"]) * 1000)
    fb.crossing = float(f["crossing"])
    fb.mu_skin = float(f["friction_skin"])
    fb.hair = hair
    fb.ellipticity = float(hair["ellipticity"]) if hair is not None else None
    fb.k_curvature_mm = kap_mm
    bd = f["bind"]
    if bool(bd["on"]):
        m = bd["m"] if bd["m"] is not None else (hair.get("lock_m", 1.0) if hair is not None else 1.0)
        beta = bd["beta"] if bd["beta"] is not None else (hair.get("bond", 0.0) if hair is not None else 0.0)
        start_mm = bd.get("start_mm")
        if start_mm is None:                            # the type's: where its curl is full, else 20 mm (d)
            t_mm = (float(hair.get("start_mm") or 0.0) + float(hair.get("ramp_mm") or 0.0)) if hair is not None else 0.0
            start_mm = t_mm if t_mm > 0 else BIND_START_MM
        fb.bind = {"m": max(float(m), 1.0), "beta": float(np.clip(float(beta), 0.0, 1.0)),
                   "oil_beta": float(np.clip(float(bd["oil_beta"]), 0.0, 1.0)),
                   "start_m": max(float(start_mm), 0.0) / 1000.0}
    if block is not None and bool(f["adhesion"]):
        fb.adhesion = oil_state(block)
        fb.cohesion = bool(f["cohesion"])
        fb.cohesion_count = str(f["cohesion_count"])
    return fb


def cycle_lengths(n, cfg, rng, max_mm=None, return_phase=False):
    """grown length of each of n hairs (m), from the growth cycle (DESIGN_v2 4.3, hair.md T5).
    Each follicle has its own growth phase A ~ U(anagen_years); it is at time t ~ U(0, A / (1 - telogen_frac)) of its
    cycle, and t is held at A once it rests (a resting hair keeps its final length), so the resting share is
    telogen_frac. L = rate x t, capped at length.max_mm when given (the style's ceiling, e.g. earlier trims).
    cycle.on false: every hair at the end of its growth phase, rate x A. return_phase also returns the resting flag."""
    cy = resolve(cfg)["cycle"]
    rate_mm_y = float(cy["rate_mm_month"]) * 12.0
    a0, a1 = (float(x) for x in cy["anagen_years"])
    A = rng.uniform(a0, a1, n)
    if cy["on"]:
        t = rng.random(n) * A / (1.0 - float(cy["telogen_frac"]))
        resting = t >= A
        t = np.minimum(t, A)
    else:
        t, resting = A, np.zeros(n, bool)
    L = rate_mm_y * t
    if max_mm is not None:
        L = np.minimum(L, float(max_mm))
    L = L / 1000.0
    return (L, resting) if return_phase else L


# ---------------------------------------------------------------- the hair type (the fibre's genetic characteristic)
# What makes a hair straight, wavy or curly is grown into it. The follicle's curvature programmes the fibre in the bulb
# (Thibaut et al. 2005, Br J Dermatol 152:632): the two sides of the cortex keratinise differently, so the hardened
# fibre bends like a bimetallic strip, and the cross-section goes from round (straight) to oval (wavy) to flat (curly),
# so it bends easily one way and twists. Every non-straight hair is therefore a HELIX - a curvature whose bending
# direction turns along the length: curly a tight round coil, wavy a loose helix flattened against its neighbours that
# reads as an S. Disulfide bonds hold that natural shape (the rest turning); hydrogen bonds, broken by water and
# re-formed on drying, hold a temporary set (settle.rewet). The hair's own weight pulls the top straight (Miller,
# Lazarus, Audoly & Reis 2014, PRL 112:068103) - the settle does that; the type gives only the rest shape.
#
# Classification: de la Mettrie, Loussouarn et al. 2007 (Hum Biol 79:265; Loussouarn et al. 2007, Int J Dermatol 46
# S1:2) - eight types I-VIII from the curve diameter (CD), the curl index (i = stretched length / end-to-end) and the
# number of waves; CD thresholds (hair_styling.md D10, P42): type I > 108 mm, II 57-108, III 31-57, IV 12-31 mm.
# European mean curvature 0.042 /mm (Lasisi et al. 2016) is a CD of ~48 mm, type III. Below IV: the curl-meter ring of
# 9.8 mm (hair.md R20) and curl radii from 0.6 cm (R21). Cross-section a/b: Asian ~1.25, European ~1.3-1.35, African
# descent ~1.75-1.83 (Franbourg et al. 2003, J Am Acad Dermatol 48:S115; hair.md R10). The per-type curl index values
# are design values (d): hair.md still lists de la Mettrie's per-type i table as to fetch; they sit inside what the CD
# classes imply and are overridable.
#
# The rest shape (settle.rest_waves / wave_natural): in the frame carried along the strand by parallel transport
# (t, n, b) the fibre's rest curvature vector is kappa0 (cos th, flat sin th), th = tau0 (s - start) + phase. flat 1:
# a round helix of curvature kappa0 and torsion tau0 (curve diameter CD = 2 / kappa0; the coil's own diameter
# 2 kappa0 / (kappa0^2 + tau0^2)); flat 0: the planar wave (amplitude kappa0 / tau0^2, wavelength 2 pi / tau0) that
# the round-17 `waves` block draws. tau0 comes from the curl index unless tau_mm is given.
HAIR_TYPES = {
    # type I-II: a CD over ~108 mm reads straight over a haircut's length; no rest curl, so 'straight' with its default
    # traits is exactly today's fibre (the radius block, no rest turning); a cd_mm of 120-200 gives it a slight bend
    # lock_m / bond (fibre.bind, design values d): the hairs per bonded lock and the bond degree from the fibre's own
    # shape. Straight fibres nest and slide: no shape bond (only oil zips them, Bico et al. 2004), locks of ~20 (a
    # 0.5 mm zipped strip at phi 0.5, r 32.5 um: 0.5 (0.25 / 0.0325)^2 ~ 30). Wavy: S-waves in register interlock
    # weakly. Curly: coils in one phase thread into each other (the ringlet; curls grow out curled in one drying
    # section) - a ringlet's fibre rope ~1 mm across at phi 0.3 holds 0.3 (0.5 / 0.0325)^2 ~ 70 hairs, 50 taken
    "straight": {"cd_mm": None, "curl_index": 1.0, "flat": 0.0, "ellipticity": 1.25, "diameter_um": None,
                 "start_mm": 0.0, "ramp_mm": 0.0, "cd_jitter": 0.0, "lock_m": 20.0, "bond": 0.0},
    # type II-III: CD 31-108 mm (P42), 48 mm the European mean (Lasisi 2016); a loose helix flattened against the
    # neighbours (flat 0.25) - the S; i 1.10 (d); oval section 1.3 (European, R10); the root straighter (ramp 25 mm,
    # Walker 2B-2C)
    "wavy": {"cd_mm": 48.0, "curl_index": 1.10, "flat": 0.25, "ellipticity": 1.3, "diameter_um": None,
             "start_mm": 0.0, "ramp_mm": 25.0, "cd_jitter": 0.0, "lock_m": 30.0, "bond": 0.1},
    # type V-VI: below IV's 12 mm (P42), the curl-meter ring 9.8 mm (R20); a round tight coil (flat 1) from the root;
    # i 2.0 (d: a coil about half its stretched length); flat section a/b 1.75 (African descent 1.75-1.83, R10)
    "curly": {"cd_mm": 10.0, "curl_index": 2.0, "flat": 1.0, "ellipticity": 1.75, "diameter_um": None,
              "start_mm": 0.0, "ramp_mm": 0.0, "cd_jitter": 0.0, "lock_m": 50.0, "bond": 0.35},
}
HAIR_TRAIT_NOTES = {
    "cd_mm": "curve diameter, mm: 2 / kappa0, the circle the free fibre's tightest arcs fit (None: no rest curl). Or "
             "kappa_mm (1/mm) directly",
    "curl_index": "i = stretched length / end-to-end of a long free fibre (the screw motion's advance); sets tau0. Or "
                  "tau_mm (1/mm), the turning rate of the bending direction (the torsion when flat is 1)",
    "flat": "0 a planar wave (the S), 1 a round helix (the coil): the out-of-plane share of the curvature",
    "ellipticity": "cross-section a/b - STORED ONLY: the bending anisotropy in the material frame is not in the settle",
    "diameter_um": "fibre diameter at the root, um (None: the style's radius block)",
    "start_mm": "the curl starts this far from the root", "ramp_mm": "and ramps in over this arc (smoothstep)",
    "cd_jitter": "per-guide CD factor U(1 -/+ jitter), drawn per drying section so a lock keeps one curl",
}


def _rot(v, ax, c, s):
    """v turned about the unit axis ax by the angle of cos c, sin s (Rodrigues)"""
    return v * c + np.cross(ax, v) * s + ax * float(ax @ v) * (1.0 - c)


def helix_path(kappa, tau, flat=1.0, L=0.1, phase=0.0, start=0.0, ramp=0.0, steps=None, frame=False):
    """the free rest shape of a fibre of the type (m, 1/m): its curvature vector in the parallel-transported frame is
    kappa (cos th, flat sin th), th = tau (s - start) + phase, ramped in from start over ramp (smoothstep). Starts at
    the origin along +z with n = +x; an exact rotation per step. -> P (steps + 1, 3) [, t, n at the end]"""
    steps = int(steps or max(200, np.ceil(L * max(kappa, tau, 1.0) * 40)))
    ds = L / steps
    t, nv = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    P = np.zeros((steps + 1, 3))
    for i in range(steps):
        s = (i + 0.5) * ds
        x = 1.0 if ramp <= 0 else min(max((s - start) / ramp, 0.0), 1.0)
        amp = (x * x * (3 - 2 * x)) if s >= start else 0.0
        th = tau * (s - start) + phase
        kv = amp * kappa * (np.cos(th) * nv + flat * np.sin(th) * np.cross(t, nv))
        km = float(np.linalg.norm(kv))
        if km * ds > 1e-14:
            ax = np.cross(t, kv) / km
            h = 0.5 * km * ds
            P[i + 1] = P[i] + ds * np.sinc(h / np.pi) * _rot(t, ax, np.cos(h), np.sin(h))
            c, sn = np.cos(2 * h), np.sin(2 * h)
            t, nv = _rot(t, ax, c, sn), _rot(nv, ax, c, sn)
        else:
            P[i + 1] = P[i] + ds * t
    return (P, t, nv) if frame else P


def curl_index(kappa, tau, flat=1.0):
    """the curl index of a long free fibre: arc / axial advance of the screw motion over one period of the bending
    direction (2 pi / tau). A round helix: sqrt(kappa^2 + tau^2) / tau exactly."""
    if kappa <= 0:
        return 1.0
    if tau <= 0:
        return float("inf")
    if flat >= 1.0:
        return float(np.hypot(kappa, tau) / tau)
    Pi = 2 * np.pi / tau
    P, t, nv = helix_path(kappa, tau, flat, Pi, steps=400, frame=True)
    R = np.stack([nv, np.cross(t, nv), t], 1)          # the end frame (n, b, t) in the start's axes (x, y, z)
    d = P[-1]
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])     # 2 sin(phi) axis
    if np.linalg.norm(w) > 1e-3:
        u = w / np.linalg.norm(w)
    elif np.trace(R) > 1.0:                             # ~ the identity: the chord is the advance
        u = d / max(float(np.linalg.norm(d)), 1e-300)
    else:                                               # ~ a half turn: the axis from (R + I) / 2 = u u^T
        M = 0.5 * (R + np.eye(3))
        c = M[:, int(np.argmax(np.linalg.norm(M, axis=0)))]
        u = c / np.linalg.norm(c)
    adv = abs(float(d @ u))
    return float(Pi / max(adv, 1e-300))


def helix_tau(kappa, flat, index):
    """tau (1/m) giving the curl index at curvature kappa (1/m) and the flatness: closed form for a round helix,
    bisection on curl_index otherwise (the index falls as tau grows)"""
    index = float(index)
    if kappa <= 0 or index <= 1.0:
        return 0.0
    if flat >= 1.0:
        return float(kappa / np.sqrt(index * index - 1.0))
    lo, hi = kappa * 1e-2, kappa * 1e2
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if curl_index(kappa, mid, flat) > index:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def hair_type(spec):
    """the hair type from the style's fibre.type: None / '' (no type: today's fibre), a name in HAIR_TYPES, or a dict
    {"base": name, <trait>: value, ...} overriding any trait (HAIR_TRAIT_NOTES; kappa_mm / tau_mm may replace cd_mm /
    curl_index). -> None, or the traits plus kappa, tau (1/m), curl (bool: a rest helix), index (the curl index
    reached), period_m (arc of one turn of the bending direction, 2 pi / tau), turn_m (arc of one turn of the coil,
    2 pi / sqrt(flat^2 kappa^2 + tau^2)), coil_d_mm (the round coil's own diameter)"""
    if spec is None or spec == "" or spec is False:
        return None
    spec = {"base": spec} if isinstance(spec, str) else dict(spec)
    base = str(spec.pop("base", spec.pop("name", "straight")))
    if base not in HAIR_TYPES:
        raise ValueError("fibre.type: unknown hair type %r (known: %s)" % (base, ", ".join(HAIR_TYPES)))
    out = dict(HAIR_TYPES[base])
    out.update(spec)
    out["type"] = base
    if out.get("kappa_mm") is not None:
        kappa = float(out["kappa_mm"]) * 1000.0
        out["cd_mm"] = 2.0 / float(out["kappa_mm"]) if kappa > 0 else None
    else:
        kappa = 2000.0 / float(out["cd_mm"]) if out.get("cd_mm") else 0.0
    flat = float(out.get("flat", 1.0))
    if kappa <= 0:
        tau = 0.0
    elif out.get("tau_mm") is not None:
        tau = float(out["tau_mm"]) * 1000.0
    else:
        tau = helix_tau(kappa, flat, out.get("curl_index", 1.0))
    out.update(kappa=kappa, tau=tau, curl=bool(kappa > 0 and tau > 0))
    out["index"] = curl_index(kappa, tau, flat) if out["curl"] else 1.0
    out["period_m"] = 2 * np.pi / tau if tau > 0 else 0.0
    out["turn_m"] = 2 * np.pi / np.hypot(flat * kappa, tau) if out["curl"] else 0.0
    out["coil_d_mm"] = 2000.0 * kappa / (kappa * kappa + tau * tau) if out["curl"] else 0.0
    return out


def measure_curl(P, turn_m=None, skip_m=0.0):
    """a hair's curve diameter and curl index from its points (n, 3) m, as the type defines them: CD = 2 / the 95th
    percentile of the discrete curvature (turning angle / mean adjacent segment length) past skip_m of arc; index =
    arc / end-to-end of the mean line (the points averaged over turn_m of arc, one turn) between its first and last
    centre; whole_index = arc / end-to-end of the whole part past skip_m. -> dict"""
    P = np.asarray(P, np.float64)
    ln = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ln)])
    keep = s >= skip_m - 1e-12
    P, s = P[keep], s[keep]
    seg = np.diff(P, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    u = seg / np.maximum(ln, 1e-300)[:, None]
    ang = np.arccos(np.clip((u[1:] * u[:-1]).sum(1), -1.0, 1.0))
    kap = ang / np.maximum(0.5 * (ln[1:] + ln[:-1]), 1e-300)
    k95 = float(np.percentile(kap, 95)) if len(kap) else 0.0
    arc = float(s[-1] - s[0])
    out = {"cd_mm": (2000.0 / k95) if k95 > 0 else float("inf"), "arc_mm": arc * 1000.0,
           "whole_index": arc / max(float(np.linalg.norm(P[-1] - P[0])), 1e-300)}
    if turn_m and arc > 1.5 * turn_m:
        ns = max(int(np.ceil(arc / (turn_m / 64))), 64)
        sg = np.linspace(s[0], s[-1], ns + 1)
        Q = np.stack([np.interp(sg, s, P[:, k]) for k in range(3)], 1)
        w = max(1, int(round(turn_m / (sg[1] - sg[0]))))
        c = np.cumsum(np.vstack([np.zeros((1, 3)), Q]), 0)
        M = (c[w:] - c[:-w]) / w
        span = (len(M) - 1) * (sg[1] - sg[0])
        out["index"] = span / max(float(np.linalg.norm(M[-1] - M[0])), 1e-300)
    else:
        out["index"] = out["whole_index"]
    return out
