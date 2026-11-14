"""Hair kinds (numpy only, Apache-2.0): the different kinds of human hair, each with its own fibre traits, and what the hair
code needs from a kind.

Skin differs by region; so does hair. Every hair is the same fibre (cuticle, cortex with pigment, an optional medulla), but
each follicle runs its own programme: its cycle sets the length, its bulb's shape the curl and the cross-section, its size
the diameter, its depth and angle the emergence, its melanocytes the colour. A KIND is that programme written as
measurable fibre traits with published ranges; every physical number the hair code uses is derived from them. A body
carries a MAP of kinds (a region may hold a mixture, e.g. vellus and terminal on the forearm), and treatment (trimming,
shaving, plucking, products) and environment (water, humidity, heat) act on top of the kind, never inside it.

    HAIR_KINDS[name]          {"label", "regions", "research", "traits": {trait: {value, range, unit, label, source, note}}}
    REQUIRED                  the traits every kind carries (each with a label M / D / E / U and a source)
    REGION_KINDS[region]      the kinds that grow on a body region of an adult woman, as shares (glabrous regions: {})
    kind(name)                a copy of one kind;  v(name, trait) its value
    hair_type_spec(name)      the fibre.type dict for engine/hair/fibre.py (props, hair_type): base HAIR_TYPES name + traits
    fibre_cfg(name)           (fibre block, radius block) for fibre.props(cfg, radius_cfg)
    curl(name)                kappa, tau (1/m), the rest helix (fibre.hair_type), the free planar wave's length / amplitude
    section(name)             root / tip radius, ellipticity, EI (round and both axes of the oval), weight per length,
                              gravity length, self-weight sag of a free horizontal hair at its natural length
    mechanics(d_um, E_gpa, rho, ell=1, L_mm=None)   the same for any fibre (the research files' T4 / T5 / T6 numbers)
    length_cap_mm(name)       growth x anagen: (lo, value, hi)
    cycle_lengths(name, n, rng, treatment=None)  grown length per hair (mm) from the growth cycle, the resting flag, and
                              blunt tips where a treatment cut them
    pigment(name, scalp, scalp_d_um=None)   melanin / redness / ash of the kind from the person's scalp colour block
                              (engine/hair/colour.py), and sigma_a with the fibre's path length (diameter) folded in
    follicles(name)           density, spacing, emergence angle and spread, hairs per follicular unit, rows
    field_desc(name, scalp=None, over=None)   the field hair's description (the fv8 lock DESC keys: radius, ellipticity,
                              twist, beta_m / beta_n / eta / alpha, length and its jitter, taper, sigma) plus the curl
    check()                   every rule the tests hold: required traits, labels, sources, value inside its range, the
                              scalp's curl traits equal to fibre.HAIR_TYPES, every derived (D) value re-derived

Labels: M measured in the source; D derived (the arithmetic is in DERIVED and re-run by check()); E estimated (an
author's statement without data, a carried scalp value, or a reasoned placeholder, the note says which); U unknown /
untested (a value kept only so the code runs). Sources are keys of the research files in engine/packs/human/research/:
hair_kinds_head.md (Kn, T1-T4), hair_kinds_vellus_body.md (Bn, PV-Vn, MB-R15, SS-*, T1-T8), hair_kinds_underarm_lower_body.md
(APn, V7 / V9, T1-T6), hair.md (Rn, Tn, Dn), hair_styling.md (Pn, Dn), and fibre.py's FIBRE_DEFAULTS / HAIR_TYPES; the
eyelashes' lid laws (11-13) also cite LIDS / SHEETS / LASHREF below (estimates, each labelled E).

Units: lengths mm (diameter um), angles deg, growth mm/day, anagen days, E GPa, rho kg/m3, twist and curvature 1/m in the
derived outputs. Every function returns plain numbers; nothing here knows any person: a case maps the kinds onto its own
body and overrides a kind's values with its own measured ones (the case's private files).
"""
import copy
import math

import numpy as np

from . import colour as HC
from . import fibre as FB

G = 9.81
LABELS = ("M", "D", "E", "U")
REQUIRED = ("diameter_um", "ellipticity", "medulla_index", "medullated_share", "cuticle_layers", "alpha_deg", "beta_m",
            "beta_n", "eta", "E_gpa", "rho", "growth_mm_day", "anagen_days", "telogen_share", "length_mm", "taper_mm",
            "tip_ratio", "curl_type", "cd_mm", "curl_index", "flat", "twist_per_m", "emergence_deg", "density",
            "unit_hairs", "pigment_x", "redness_add")
OPTIONAL = ("mid_ratio", "rows", "count", "major_axis",
            # the lid laws (eyelashes, 11-13): how a lash row grows along its lid margin, read by a grower that knows only the
            # kind and the lid's region (rows and spacing, the density and length laws, lift / sweep / curl, clusters, pigment)
            "length_sd_mm", "row_share", "row_spacing_mm", "punctum_bare_mm", "medial_sparse", "length_law", "growing_fraction",
            "lift_deg", "sweep_deg", "curl_tip_radius_mm", "root_curl_share", "cluster_size", "cluster_bind", "pigment_law")
NONE_OK = ("curl_type", "cd_mm")          # None = no rest curve (straight) / no HAIR_TYPES preset

HEAD = "hair_kinds_head.md"
BODY = "hair_kinds_vellus_body.md"
AXP = "hair_kinds_underarm_lower_body.md"
LIDS = "the anatomy texts' lid-margin laws (" + HEAD + " T3 R29 and the texts a case's lash notes cite, 11-05)"
SHEETS = "by eye on rendered lids against reference eyes (a case's sheet rounds g02-g06, 11-10)"
LASHREF = "11 natural-lash photographs (Wikimedia Commons; reference content/lashes_natural_2026-11-13, 11-13)"


def T(value, lo=None, hi=None, unit="", label="E", source="", note=""):
    """one trait: value, (lo, hi) range or None, unit, label M / D / E / U, source key(s), note"""
    return {"value": value, "range": None if lo is None else (lo, hi), "unit": unit, "label": label, "source": source,
            "note": note}


def _keratin(beta_m=None, note_b=""):
    """the fibre material every kind shares unless measured otherwise: keratin's index, stiffness and density, and the
    longitudinal / azimuthal roughness of the cuticle (no per-kind measurement exists: E carried from scalp hair)"""
    return {
        "eta": T(1.55, 1.53, 1.56, "", "M", "hair.md T7 R32 R33", "keratin's refractive index (pbrt, Cycles default)"),
        "E_gpa": T(4.0, 3.1, 10.0, "GPa", "E", "fibre.py E_gpa (Wortmann 2022; Goldstein 2012); hair.md R22 3.1 tensile; "
                   + HEAD + " K19 ~10 by bending",
                   "measured on scalp hair only; carried to this kind (no kind-specific test found)"),
        "rho": T(1300.0, 1300.0, 1320.0, "kg/m3", "E", "fibre.py rho (Goldstein 2012); hair.md T8 1320", "scalp value carried"),
        "beta_m": beta_m or T(0.10, 0.08, 0.28, "", "E", "hair.md T7 R34 (specular FWHM 9-10 deg -> beta_m 0.08-0.1, D); "
                              "R32 pbrt typical 0.25",
                              "the scalp's measured highlight width carried; per-kind roughness NOT FOUND " + note_b),
        "beta_n": T(0.30, 0.10, 0.55, "", "E", "hair.md T7 R32 (0.1-0.3 in examples); R33 Cycles radial roughness",
                    "azimuthal roughness; no measurement per kind"),
    }


def _kind(label, regions, research, **traits):
    return {"label": label, "regions": regions, "research": research, "traits": traits}


# ======================================================================================================== the kinds
HAIR_KINDS = {
    # ------------------------------------------------------------------------------------------- scalp, main
    "scalp": _kind(
        "Scalp terminal hair (top, crown, occiput, nape, above the ears)",
        "hair-bearing scalp inside the hairline; the edge band is 'scalp_edge'", HEAD + " section 1, T1; hair.md",
        diameter_um=T(70.0, 58.0, 90.0, "um", "M", HEAD + " T1 (R10 Caucasian 70; K17 trichoscopy frontal 61 / occiput 58)",
                      "equivalent circle; trichoscopy reads ~10 um under micrometer / OCT: compare within one method"),
        ellipticity=T(1.32, 1.25, 1.38, "a/b", "M", "hair.md R10 (1.3-1.35 European); " + BODY + " B1 women 1.38",
                      "weak (W) per source"),
        medulla_index=T(0.20, 0.0, 0.45, "width ratio", "E", HEAD + " T1 K24 (one donor, ~20 % of the section)",
                        "medulla width / shaft width when present; often absent or broken in fine Caucasian hair"),
        medullated_share=T(0.4, 0.2, 0.6, "share", "E", AXP + " T1 AP1 (scalp 18 of 30 women medullated); " + HEAD
                           + " K24", "fine European hair is medullated less often than the AP1 cohort (E)"),
        cuticle_layers=T(6.0, 4.0, 8.0, "layers", "M", "hair.md R8", "scale-edge interval 8.3-10.7 um (K18)"),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "M", "hair.md T7 R34 (2.5-3 measured); " + HEAD + " T4 (2.7-3.4 D from K18)",
                    "cuticle scale tilt: highlight shifts R +2a, TT -a, TRT -4a"),
        **_keratin(),
        growth_mm_day=T(0.36, 0.30, 0.45, "mm/day", "M", "hair.md T5 (Loussouarn 2001 0.396; R10 0.367); " + HEAD
                        + " K20 (women faster than men)", "finer hairs grow slower (~0.25 at 20-30 um, R36)"),
        anagen_days=T(1826.0, 730.0, 2555.0, "days", "M", "hair.md T5 (2-7 y), Wolfram 2003 3-5 y",
                      "per-follicle growth phase; drawn uniform over the range (fibre.py cycle.anagen_years)"),
        telogen_share=T(0.10, 0.10, 0.14, "share", "M", "hair.md D8 (10-14 %), Wolfram 2003 ~10 %", ""),
        length_mm=T(657.0, 261.0, 912.0, "mm", "D", "growth x anagen (" + HEAD + " T1: 26-91 cm)",
                    "the natural (uncut) length cap; a haircut, not growth, sets scalp length"),
        taper_mm=T(30.0, 0.0, 200.0, "mm", "E", "fibre.py taper_mm (DESIGN_v2 13.4; no measurement, hair.md section 4)",
                   "an uncut hair thins over this arc to its natural tip; scissors remove it"),
        tip_ratio=T(0.25, 0.1, 0.5, "tip/root", "E", "fibre.py radius block (tip / root of an uncut hair)", ""),
        curl_type=T("wavy", label="M", source="hair.md R21 (European straight 41 / wavy 47 / curly 13 %, W); fibre.py "
                    "HAIR_TYPES", note="per PERSON: a case names its type; 'wavy' is the commonest European class"),
        cd_mm=T(48.0, 31.0, 108.0, "mm", "M", "fibre.py HAIR_TYPES['wavy'] (Lasisi 2016 mean 0.042 /mm = CD 48; P42 classes)",
                "curve diameter of the free fibre; equals HAIR_TYPES[curl_type]"),
        curl_index=T(1.10, 1.0, 1.3, "i", "E", "fibre.py HAIR_TYPES['wavy'] (design value d)", ""),
        flat=T(0.25, 0.0, 1.0, "", "E", "fibre.py HAIR_TYPES['wavy']", "0 planar S, 1 round helix"),
        twist_per_m=T(69.59, 40.0, 120.0, "1/m", "D", "the wavy helix's torsion (fibre.hair_type('wavy')['tau'])",
                      "the oval section turns with the bending direction; equals the field groom's measured 69.6 / m"),
        emergence_deg=T(40.0, 30.0, 60.0, "deg", "E", "hair.md T3 / " + HEAD + " T1 (R16 R17: 30-60 mid and back, 10-20 "
                        "lower occiput)", "angle to the skin plane; whorl steeper; per region in follicle.py"),
        density=T(220.0, 154.0, 263.0, "1/cm2", "M", "hair.md D1 / " + HEAD + " T1 (R1 Caucasian 214-230; R2 Thai 134-163)",
                  "follicles per cm2; temporoparietal ~13-18 % lower"),
        unit_hairs=T(1.55, 1.4, 2.0, "hairs/unit", "D", "hair.md T2 R5 (1/2/3/4 hairs = 60/29/9/3 %)",
                     "mean hairs per follicular unit (surgical series ~2.0)"),
        pigment_x=T(1.0, 1.0, 1.0, "x scalp", "D", "definition (the reference kind)",
                    "eumelanin concentration relative to the person's scalp root"),
        redness_add=T(0.0, 0.0, 0.0, "", "D", "definition", ""),
    ),
    # ------------------------------------------------------------------------------------------- scalp edge
    "scalp_edge": _kind(
        "Scalp edge: front hairline and temples (baby hairs)",
        "5-10 mm band inside the front hairline, temples and temple apex, sideburn", HEAD + " section 1, T1",
        diameter_um=T(35.0, 15.0, 60.0, "um", "E", HEAD + " T1 K22 (vellus < 30, intermediate 30-60 um), K17",
                      "a mix of vellus and intermediate hairs; 35 um is the T4 worked example"),
        ellipticity=T(1.2, 1.1, 1.3, "a/b", "E", HEAD + " section 1 (NOT FOUND; E 1.1-1.3)", ""),
        medulla_index=T(0.0, 0.0, 0.2, "width ratio", "M", HEAD + " T1 K22 (vellus: no medulla)", ""),
        medullated_share=T(0.1, 0.0, 0.3, "share", "E", HEAD + " K22", "intermediate hairs only"),
        cuticle_layers=T(4.0, 3.0, 6.0, "layers", "E", HEAD + " section 1 (NOT FOUND)", "fewer layers on a thin fibre"),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", "scalp value carried (hair.md R34)", ""),
        **_keratin(),
        growth_mm_day=T(0.25, 0.20, 0.30, "mm/day", "M", HEAD + " T1 R36 (20-30 um hairs ~0.25)", "weak (W)"),
        anagen_days=T(154.0, 90.0, 220.0, "days", "M", HEAD + " T1 K21 (temple 22 weeks, 3 men, via secondary)",
                      "no women's data; weak"),
        telogen_share=T(0.15, 0.10, 0.25, "share", "E", HEAD + " section 1 (NOT FOUND)", ""),
        length_mm=T(38.5, 18.0, 66.0, "mm", "D", "growth x anagen; " + HEAD + " T1 (E 2-10 cm)",
                    "no measured baby-hair length distribution"),
        taper_mm=T(8.0, 3.0, 15.0, "mm", "E", HEAD + " T1 (natural pointed tips)", "uncut: the whole tip thins"),
        tip_ratio=T(0.25, 0.1, 0.5, "tip/root", "E", "as scalp", ""),
        curl_type=T(None, label="E", source=HEAD + " T1 (straighter than the lengths: too short to show a wave)"),
        cd_mm=T(None, label="E", source=HEAD + " T1", note="no rest curve (straight); a case may give its scalp's"),
        curl_index=T(1.0, 1.0, 1.1, "i", "E", HEAD + " T1", ""),
        flat=T(0.0, 0.0, 1.0, "", "E", "-", "unused without a curve"),
        twist_per_m=T(40.0, 0.0, 120.0, "1/m", "E", "no measurement (the scalp's wavy twist ~70; a straight hair less)", ""),
        emergence_deg=T(15.0, 5.0, 20.0, "deg", "E", HEAD + " T1 R16 R17 (front 10-20, temples 5-10; clinic, W)", ""),
        density=T(110.0, 80.0, 150.0, "1/cm2", "D", HEAD + " T1 P50 (about half the density behind it)", ""),
        unit_hairs=T(1.0, 1.0, 1.2, "hairs/unit", "M", HEAD + " T1 K17 (mostly single-hair units at the temple)", ""),
        pigment_x=T(1.0, 0.7, 1.0, "x scalp", "E", HEAD + " section 1 (same follicle colour; finer hairs look lighter)",
                    "the lighter look comes from the shorter path through a thin fibre (pigment() folds it in)"),
        redness_add=T(0.0, 0.0, 0.0, "", "E", "-", ""),
    ),
    # ------------------------------------------------------------------------------------------- eyebrow
    "eyebrow": _kind(
        "Eyebrow (supercilia)",
        "head (medial 0.5-0.7 cm), body (~2.5 cm, densest), tail (lateral ~2 cm); women's brow 5.0 x 1.3 cm",
        HEAD + " section 1, T2",
        diameter_um=T(60.0, 45.0, 100.0, "um", "E", AXP + " T1 AP1 (eyebrow 46.7 um, 30 Ghanaian women 15-20 y, M); "
                      + HEAD + " T2 (healthy adults NOT FOUND; E 60-100 in the body, K9)", "finer at head and tail"),
        ellipticity=T(1.2, 1.1, 1.3, "a/b", "E", HEAD + " section 1 (NOT FOUND; E 1.1-1.3)", ""),
        medulla_index=T(0.2, 0.0, 0.3, "width ratio", "E", HEAD + " T2 K27 (more often medullated than scalp)", ""),
        medullated_share=T(0.6, 0.4, 0.8, "share", "E", HEAD + " T2 K27 (lower body > axilla > eyebrow > scalp)",
                           "between the scalp (0.4) and the axilla (0.9)"),
        cuticle_layers=T(6.0, 4.0, 8.0, "layers", "E", HEAD + " T2 K18 (interval like leg hair, qualitative)", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", HEAD + " T4 (no data; scalp carried)", ""),
        **_keratin(),
        growth_mm_day=T(0.15, 0.14, 0.16, "mm/day", "M", HEAD + " T2 K10 (via secondary)", "weak (W)"),
        anagen_days=T(70.0, 60.0, 120.0, "days", "M", HEAD + " T2 R36 (about 70 d vs 4-7 mo: conflict)",
                      "the shorter value fits untrimmed brow lengths (D in T2)"),
        telogen_share=T(0.6, 0.4, 0.8, "share", "E", HEAD + " T2 R36 (telogen up to ~9 mo)", ""),
        length_mm=T(10.5, 9.0, 19.2, "mm", "D", "growth x anagen (" + HEAD + " T2: 10.5-18 mm)", "measured: NOT FOUND"),
        taper_mm=T(3.0, 1.5, 5.0, "mm", "E", HEAD + " T1 (natural tips unless trimmed)", ""),
        tip_ratio=T(0.2, 0.1, 0.3, "tip/root", "E", "-", ""),
        curl_type=T(None, label="E", source=HEAD + " T1 (curl NOT FOUND; E a gentle bend toward the skin)"),
        cd_mm=T(30.0, 15.0, 60.0, "mm", "E", HEAD + " T1", "a single gentle arc toward the skin, not a helix"),
        curl_index=T(1.0, 1.0, 1.05, "i", "E", "-", "an arc shorter than a turn: no repeat"),
        flat=T(0.0, 0.0, 0.0, "", "E", "-", "planar"),
        twist_per_m=T(40.0, 0.0, 120.0, "1/m", "E", "no measurement", ""),
        emergence_deg=T(12.0, 8.0, 15.0, "deg", "E", HEAD + " T2 K16 (10-15, < 15; surgeons, clinics)", ""),
        density=T(66.7, 40.0, 100.0, "1/cm2", "D", HEAD + " T2 (300 hairs / (5.0 x 0.9 cm), W counts)", "tail lowest (K13)"),
        unit_hairs=T(1.0, 1.0, 1.2, "hairs/unit", "E", HEAD + " T2 K16 (single; some pairs centrally)", ""),
        pigment_x=T(1.4, 1.0, 2.0, "x scalp", "E", HEAD + " T2 K11 (follows the scalp imperfectly, own loci); " + AXP
                    + " T4 AP1 (eyebrow very dark vs scalp brownish, M qual); " + BODY + " T3 B12",
                    "a person's brows are their own call (natural range shown); head hairs lighter (K9)"),
        redness_add=T(0.0, -0.05, 0.05, "", "E", "-", ""),
        count=T(300.0, 200.0, 400.0, "per brow", "E", HEAD + " T2 R9 K16", ""),
    ),
    # ------------------------------------------------------------------------------------------- eyelash, upper
    "eyelash_upper": _kind(
        "Eyelash, upper lid (cilia)", "anterior upper lid margin, 2-6 rows; medial / central / lateral thirds",
        HEAD + " section 1, T3",
        diameter_um=T(71.0, 61.0, 80.0, "um", "M", HEAD + " T3 K4 (eq. circle 71.3; major 77.8, minor 63.9); R28 "
                      "Caucasian 61 vs Asian 72", "root diameter; the shaft is spindle-shaped (mid_ratio)"),
        ellipticity=T(1.21, 1.11, 1.33, "a/b", "D", HEAD + " T3 K4 (minor/major 0.826 +- 0.073)",
                      "1 / 0.826; the flat face bends 1.47x easier"),
        medulla_index=T(0.28, 0.2, 0.35, "width ratio", "D", HEAD + " T3 K4 (7.6 % of the section: sqrt(0.076))", ""),
        medullated_share=T(0.9, 0.7, 1.0, "share", "E", HEAD + " T3 K3 K4 (present, TEM)", ""),
        cuticle_layers=T(7.0, 6.5, 8.0, "layers", "M", HEAD + " T3 R28 (6.5 Caucasian, 8.0 Asian), K3", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", HEAD + " T4 (no data; scalp carried)", "mascara / curlers crack it (K4)"),
        **_keratin(),
        growth_mm_day=T(0.13, 0.12, 0.14, "mm/day", "M", HEAD + " T3 R27 (0.12 +- 0.05), R29 (0.12-0.14)", ""),
        anagen_days=T(60.0, 34.0, 90.0, "days", "M", HEAD + " T3 R28 (~2 mo), R27 (34 d), K3 (1-3 mo): conflict",
                      "0.14 x 60 = 8.4 mm fits the measured lengths; 34 d does not"),
        telogen_share=T(0.83, 0.80, 0.85, "share", "M", HEAD + " T3 K3 (anagen ~15 %), R28 (17.8 % Asian)", ""),
        length_mm=T(7.3, 4.0, 10.1, "mm", "M", HEAD + " T3 K2 (7.33 +- 1.15, 177 US women); K1 7.5 Japanese women", ""),
        taper_mm=T(3.0, 2.0, 4.0, "mm", "E", HEAD + " T3 K3 (a long fine tip); LASH plan (last ~40 %)", ""),
        tip_ratio=T(0.1, 0.06, 0.14, "tip/root", "M", HEAD + " T3 K3 (5-10 um tip / 70-80 root, Liotet 1977)", ""),
        mid_ratio=T(1.3, 1.0, 1.5, "mid/root", "M", HEAD + " T3 K3 (80-120 mid-shaft / 70-80 root)", "spindle"),
        curl_type=T(None, label="M", source=HEAD + " T3 R27 K3 K1 (curved; bulb like curly hair; 'curly' in 76 % of "
                    "Japanese women)", note="a single upward arc, not a helix"),
        cd_mm=T(12.0, 10.0, 20.0, "mm", "E", HEAD + " T3 (curvature radius NOT FOUND)", "arc radius 5-10 mm"),
        curl_index=T(1.0, 1.0, 1.05, "i", "E", "-", "shorter than a turn"),
        flat=T(0.0, 0.0, 0.0, "", "E", "-", "planar arc"),
        twist_per_m=T(40.0, 0.0, 120.0, "1/m", "E", "no measurement", ""),
        emergence_deg=T(75.0, 64.0, 86.0, "deg", "M", HEAD + " T3 K5 (root at 75 +- 11 deg to the lid skin)",
                        "lateral view 71.8 deg from the vertical in women (K1)"),
        density=T(7.62, 5.5, 9.7, "1/mm", "D", HEAD + " T3 K1 (15.24 per central 2 mm, women)", "per mm of lid margin"),
        unit_hairs=T(1.0, 1.0, 1.0, "hairs/unit", "E", HEAD + " T3 (single follicles in rows)", ""),
        rows=T(3.0, 2.0, 6.0, "rows", "E", HEAD + " T3 K3 (2-4), R29 (5-6): conflict", ""),
        count=T(171.5, 90.0, 180.0, "per lid", "D", HEAD + " T3 R28 (1715 / 10 Caucasian women); R29 90-160", ""),
        pigment_x=T(1.75, 1.5, 3.0, "x scalp", "E", HEAD + " T3 R27 (dense melanin, DCT keeps it), K3; " + BODY
                    + " T3 B12 (lashes darker than scalp)", "not measured relative to a blond scalp anywhere found"),
        redness_add=T(0.0, -0.05, 0.05, "", "E", "-", ""),
        # the lid laws (11-13): read by a grower that knows only the kind and the lid's region
        length_sd_mm=T(1.15, 0.9, 1.5, "mm", "M", HEAD + " T3 K2 (7.33 +- 1.15, 177 US women)", "full-grown spread"),
        row_share=T((0.45, 0.33, 0.22), label="E", source=LIDS + " (rows 2-3 irregular)", note="front row fullest"),
        row_spacing_mm=T(0.12, 0.10, 0.40, "mm", "E", LIDS + " (back rows <= 0.40 mm along the margin)", ""),
        punctum_bare_mm=T(5.5, 5.0, 6.0, "mm", "E", LIDS + " (none nasal to the punctum, ~5-6 mm from the medial canthus)", ""),
        medial_sparse=T(0.40, 0.2, 0.6, "share", "E", SHEETS + " g02 (density thins toward the inner corner)",
                        "density x (1 - share x (1 - smoothstep(f / 0.35))), f = 0 medial -> 1 lateral"),
        length_law=T((0.62, 0.34, 0.62), label="E", source=LIDS + " (8-12 mm centrally, 6-7 at the ends); " + SHEETS
                     + " g02 (longest in the centre-outer third)", note="(peak f, width, floor): longest at f = peak"),
        growing_fraction=T((0.15, 0.95), label="E", source=HEAD + " T3 (growth 0.13 mm a day, uniform in anagen time)",
                           note="a growing lash's length / full: uniform over this span"),
        lift_deg=T((-5.0, 7.0, -18.0, 16.0), label="E", source=SHEETS + " g04 (forward and a little down, curling up later)",
                   note="(mean, sd, lo, hi) from the anterior normal toward the lid's up"),
        sweep_deg=T((-6.0, 34.0, 6.0), label="E", source=LIDS + " (medial 5-10 deg nasal, lateral third 25-40 deg temporal); "
                    + SHEETS + " g02", note="(medial, lateral, sd)"),
        curl_tip_radius_mm=T(9.0, 6.0, 12.0, "mm", "E", SHEETS + " g06 (radius 9 mm: ~40 deg of curl over a 9-10 mm lash)", ""),
        root_curl_share=T(0.30, 0.2, 0.55, "share", "E", SHEETS + " g04 (the root's share of the tip curvature)", ""),
        cluster_size=T((1, 6), label="E", source=SHEETS + " g02; " + LASHREF + " (crossing lashes gathered into small pointed "
                       "groups)", note="neighbours per group"),
        cluster_bind=T((0.40, 0.22), label="E", source=SHEETS + " g02; " + LASHREF, note="(mean, sd) share the distal parts are drawn together"),
        pigment_law=T({"root_mm": 1.5, "root_dark": 0.30, "tip_mm": 1.5, "tip_light": 0.55}, label="E",
                      source=LIDS + " (dark root, almost translucent over the last 1-2 mm)",
                      note="per point on the fibre's own absorption: darker over root_mm, lighter toward the tip over tip_mm; "
                           "ranges root_mm 0.5-3, root_dark 0-0.8, tip_mm 0.5-2.5, tip_light 0.2-0.8"),
    ),
    # ------------------------------------------------------------------------------------------- eyelash, lower
    "eyelash_lower": _kind(
        "Eyelash, lower lid (cilia)", "anterior lower lid margin, 3-4 rows; medial / central / lateral thirds",
        HEAD + " section 1, T3",
        diameter_um=T(60.0, 50.0, 71.0, "um", "E", HEAD + " T1 (NOT FOUND; E a little thinner than upper)", ""),
        ellipticity=T(1.21, 1.11, 1.33, "a/b", "E", "as upper (" + HEAD + " T1)", ""),
        medulla_index=T(0.28, 0.2, 0.35, "width ratio", "E", "as upper", ""),
        medullated_share=T(0.9, 0.7, 1.0, "share", "E", "as upper", ""),
        cuticle_layers=T(7.0, 6.5, 8.0, "layers", "E", "as upper", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", "as upper", ""),
        **_keratin(),
        growth_mm_day=T(0.13, 0.12, 0.14, "mm/day", "E", "as upper (" + HEAD + " T1: NOT FOUND separately)", ""),
        anagen_days=T(45.0, 34.0, 60.0, "days", "E", "as upper, shorter to fit the shorter lashes (D)", ""),
        telogen_share=T(0.83, 0.80, 0.85, "share", "E", "as upper", ""),
        length_mm=T(5.6, 5.0, 6.2, "mm", "M", HEAD + " T3 K1 (5.56 +- 0.60 Japanese women); K3 6-8 textbook", ""),
        taper_mm=T(2.0, 1.5, 3.0, "mm", "E", "as upper", ""),
        tip_ratio=T(0.1, 0.06, 0.14, "tip/root", "E", "as upper", ""),
        mid_ratio=T(1.3, 1.0, 1.5, "mid/root", "E", "as upper", ""),
        curl_type=T(None, label="M", source=HEAD + " T3 K1 ('curly' in all 50 Japanese)", note="a single downward arc"),
        cd_mm=T(20.0, 12.0, 30.0, "mm", "E", HEAD + " T3 (NOT FOUND)", "gentler than the upper"),
        curl_index=T(1.0, 1.0, 1.05, "i", "E", "-", ""),
        flat=T(0.0, 0.0, 0.0, "", "E", "-", ""),
        twist_per_m=T(40.0, 0.0, 120.0, "1/m", "E", "no measurement", ""),
        emergence_deg=T(75.0, 64.0, 86.0, "deg", "E", "as upper (" + HEAD + " T3 K5: depth 1.8 mm lower)",
                        "lateral view 99.8 deg from the vertical in women (K1); frontal fan 137 / 72-77 / 25-37 deg (K7 K6)"),
        density=T(3.12, 2.5, 3.7, "1/mm", "D", HEAD + " T3 K1 (6.24 per central 2 mm, women)", "per mm of lid margin"),
        unit_hairs=T(1.0, 1.0, 1.0, "hairs/unit", "E", "-", ""),
        rows=T(3.0, 3.0, 4.0, "rows", "E", HEAD + " T3 R29", ""),
        count=T(77.0, 75.0, 80.0, "per lid", "E", HEAD + " T3 R29 (textbooks)", ""),
        pigment_x=T(1.75, 1.5, 3.0, "x scalp", "E", "as upper", ""),
        redness_add=T(0.0, -0.05, 0.05, "", "E", "-", ""),
        # the lid laws (11-13)
        length_sd_mm=T(0.60, 0.4, 0.8, "mm", "M", HEAD + " T3 K1 (5.56 +- 0.60 Japanese women)", ""),
        row_share=T((0.7, 0.3), label="E", source=LIDS + " (1-2 rows)", note=""),
        row_spacing_mm=T(0.10, 0.08, 0.20, "mm", "E", LIDS + " (rows 1-2)", ""),
        punctum_bare_mm=T(6.0, 5.0, 7.0, "mm", "E", LIDS + " (~6 mm)", ""),
        medial_sparse=T(0.30, 0.1, 0.5, "share", "E", SHEETS + " g02", ""),
        length_law=T((0.65, 0.40, 0.55), label="E", source=SHEETS + " g02", note="(peak f, width, floor)"),
        growing_fraction=T((0.15, 0.95), label="E", source="as upper", note=""),
        lift_deg=T((30.0, 7.0, 15.0, 45.0), label="E", source=LIDS + " (forward, 15-25 deg down); " + SHEETS + " g02 (down and "
                   "out)", note="from the anterior normal toward the lid's down"),
        sweep_deg=T((-2.0, 40.0, 8.0), label="E", source=LIDS + " (similar to the upper, a little less)", note="(medial, lateral, sd)"),
        curl_tip_radius_mm=T(12.0, 10.0, 20.0, "mm", "E", LIDS + " (downward, gentler)", ""),
        root_curl_share=T(0.30, 0.2, 0.5, "share", "E", "as upper", ""),
        cluster_size=T((1, 4), label="E", source=SHEETS + " g02; " + LASHREF, note=""),
        cluster_bind=T((0.45, 0.20), label="E", source=SHEETS + " g02; " + LASHREF, note=""),
        pigment_law=T({"root_mm": 1.5, "root_dark": 0.30, "tip_mm": 1.5, "tip_light": 0.55}, label="E", source="as upper", note=""),
    ),
    # ------------------------------------------------------------------------------------------- vellus, face (K1)
    "vellus_face": _kind(
        "Facial vellus (peach fuzz)", "forehead, cheek, nose, upper lip, chin, ears, neck, eyelid skin",
        BODY + " section 1 K1, T1-T7",
        diameter_um=T(16.0, 10.0, 30.0, "um", "M", BODY + " T1 PV-V1 (forehead 16), SS-L17 (15 fit), B10 (< 30)", ""),
        ellipticity=T(1.1, 1.0, 1.2, "a/b", "E", BODY + " section 1 (NOT FOUND; round, thin)", ""),
        medulla_index=T(0.0, 0.0, 0.0, "width ratio", "M", BODY + " T2 SRL-S27 (vellus: none)", ""),
        medullated_share=T(0.0, 0.0, 0.0, "share", "M", BODY + " T2", ""),
        cuticle_layers=T(3.0, 2.0, 4.0, "layers", "E", BODY + " T2 (4 x 0.5 um would be 25 % of the radius)", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", BODY + " T2 (scalp carried)", ""),
        **_keratin(),
        growth_mm_day=T(0.03, 0.03, 0.13, "mm/day", "M", BODY + " T4 PV-V6 (forehead 0.03)", ""),
        anagen_days=T(41.7, 17.0, 67.0, "days", "D", BODY + " T4 (0.5-2 mm / 0.03 mm/day = 17-67 d; the mid value)", ""),
        telogen_share=T(0.51, 0.4, 0.6, "share", "D", BODY + " T4 PV-V6 (49 % growing)", ""),
        length_mm=T(1.0, 0.5, 2.0, "mm", "M", BODY + " T4 SS-L17 (fit), SS-IG07 ('about 1 mm')", ""),
        taper_mm=T(0.4, 0.2, 0.8, "mm", "E", BODY + " T2 (uncut body hairs taper; length NOT FOUND)", ""),
        tip_ratio=T(0.3, 0.1, 0.5, "tip/root", "E", "-", ""),
        curl_type=T(None, label="E", source=BODY + " section 1 (NOT FOUND; near straight over 1-2 mm)"),
        cd_mm=T(None, label="E", source=BODY + " section 1"),
        curl_index=T(1.0, 1.0, 1.0, "i", "E", "-", ""),
        flat=T(0.0, 0.0, 0.0, "", "E", "-", ""),
        twist_per_m=T(0.0, 0.0, 40.0, "1/m", "E", "no measurement", ""),
        emergence_deg=T(15.0, 5.0, 30.0, "deg", "E", BODY + " T5 (NOT FOUND; lies near the skin)", ""),
        density=T(290.0, 25.0, 439.0, "1/cm2", "M", BODY + " T5 MB-R15 (forehead 285), PV-V6 (439), SS-L17 (25-100 "
                  "visible)", "forehead; cheek, lip, ear values NOT FOUND"),
        unit_hairs=T(1.0, 1.0, 1.0, "hairs/unit", "E", BODY + " section 1", ""),
        pigment_x=T(0.05, 0.0, 0.3, "x scalp", "M", BODY + " T3 B3 (50 % of women's facial hairs 'unpigmented'); "
                    "SS-L17 (melanin 0.001 fit)", "nearly colourless"),
        redness_add=T(0.0, 0.0, 0.0, "", "E", "-", ""),
    ),
    # ------------------------------------------------------------------------------------------- vellus, body (K2)
    "vellus_body": _kind(
        "Trunk and limb vellus", "chest, breast, back, abdomen, buttocks, upper arm, proximal forearm, neck",
        BODY + " section 1 K2",
        diameter_um=T(25.0, 15.0, 30.0, "um", "E", BODY + " T1 (< 30 um by definition, B10; site values NOT FOUND)", ""),
        ellipticity=T(1.1, 1.0, 1.2, "a/b", "E", BODY + " section 1 (NOT FOUND)", ""),
        medulla_index=T(0.0, 0.0, 0.0, "width ratio", "M", BODY + " T2 (definition)", ""),
        medullated_share=T(0.0, 0.0, 0.0, "share", "M", BODY + " T2", ""),
        cuticle_layers=T(3.0, 2.0, 4.0, "layers", "E", BODY + " T2", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", BODY + " T2", ""),
        **_keratin(),
        growth_mm_day=T(0.13, 0.10, 0.18, "mm/day", "M", BODY + " T4 PV-V6 (back vellus)", ""),
        anagen_days=T(22.0, 15.0, 40.0, "days", "M", BODY + " T4 PV-V8 (women's upper arm / thigh anagen)", "across sites"),
        telogen_share=T(0.685, 0.6, 0.8, "share", "D", BODY + " T4 PV-V6 (31.5 % growing, back)", ""),
        length_mm=T(2.86, 1.5, 7.2, "mm", "D", BODY + " T4 (0.13 x 22 d)", "weak: mixes sites"),
        taper_mm=T(0.8, 0.3, 1.5, "mm", "E", BODY + " T2", ""),
        tip_ratio=T(0.3, 0.1, 0.5, "tip/root", "E", "-", ""),
        curl_type=T(None, label="E", source=BODY + " section 1 (NOT FOUND)"),
        cd_mm=T(None, label="E", source=BODY + " section 1"),
        curl_index=T(1.0, 1.0, 1.0, "i", "E", "-", ""),
        flat=T(0.0, 0.0, 0.0, "", "E", "-", ""),
        twist_per_m=T(0.0, 0.0, 40.0, "1/m", "E", "no measurement", ""),
        emergence_deg=T(20.0, 10.0, 35.0, "deg", "E", BODY + " T5 (NOT FOUND)", ""),
        density=T(30.0, 15.6, 85.0, "1/cm2", "M", BODY + " T5 MB-R15 (chest 28.4, abdomen 16.9, upper arm 45.9, neck "
                  "47.3, lower back 24.7), PV-V6 (back 85)", "per site in REGION_DENSITY"),
        unit_hairs=T(1.0, 1.0, 1.0, "hairs/unit", "E", BODY + " T5", ""),
        pigment_x=T(0.15, 0.0, 0.3, "x scalp", "E", BODY + " T3 (hypopigmented by definition)", ""),
        redness_add=T(0.0, 0.0, 0.0, "", "E", "-", ""),
    ),
    # ------------------------------------------------------------------------------------------- intermediate (K3)
    "intermediate_body": _kind(
        "Intermediate body hair", "thigh, upper arm, distal forearm; upper lip in some women", BODY + " section 1 K3",
        diameter_um=T(40.0, 29.0, 60.0, "um", "M", BODY + " T1 (30-60 by class, B10; thigh 29, PV-V1 weak)", ""),
        ellipticity=T(1.05, 1.0, 1.15, "a/b", "M", BODY + " T2 PV-V9 ('almost round', qualitative)", ""),
        medulla_index=T(0.0, 0.0, 0.1, "width ratio", "M", BODY + " T2 PV-V9 (arm: none visible)", ""),
        medullated_share=T(0.05, 0.0, 0.2, "share", "E", BODY + " T2", ""),
        cuticle_layers=T(4.0, 3.0, 6.0, "layers", "E", BODY + " T2", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", BODY + " T2", ""),
        **_keratin(),
        growth_mm_day=T(0.19, 0.13, 0.25, "mm/day", "M", BODY + " T4 B6 (thigh 0.16-0.25 adult, Myers via secondary)", "weak"),
        anagen_days=T(22.0, 18.0, 30.0, "days", "M", BODY + " T4 PV-V8 (women's thigh and upper arm 22 d)", ""),
        telogen_share=T(0.77, 0.74, 0.79, "share", "D", BODY + " T4 PV-V8 (22 / 84 and 22 / 106 d)", ""),
        length_mm=T(4.2, 2.9, 5.5, "mm", "D", BODY + " T4 (22 d x 0.13-0.25)", ""),
        taper_mm=T(1.0, 0.5, 2.0, "mm", "E", BODY + " T2", ""),
        tip_ratio=T(0.3, 0.1, 0.5, "tip/root", "E", "-", ""),
        curl_type=T(None, label="E", source=BODY + " section 1 ('arc' curve, E)"),
        cd_mm=T(25.0, 12.0, 50.0, "mm", "E", BODY + " section 1 (an arc; curvature NOT FOUND)", ""),
        curl_index=T(1.0, 1.0, 1.05, "i", "E", "-", ""),
        flat=T(0.0, 0.0, 0.0, "", "E", "-", ""),
        twist_per_m=T(20.0, 0.0, 80.0, "1/m", "E", "no measurement", ""),
        emergence_deg=T(20.0, 10.0, 30.0, "deg", "E", BODY + " T5 (NOT FOUND)", ""),
        density=T(30.0, 21.0, 46.0, "1/cm2", "M", BODY + " T5 MB-R15 (thigh 21.0, forearm 40.3, upper arm 45.9 follicles)",
                  "share of follicles intermediate vs vellus NOT FOUND"),
        unit_hairs=T(1.0, 1.0, 1.0, "hairs/unit", "E", BODY + " T5", ""),
        pigment_x=T(0.4, 0.2, 0.7, "x scalp", "E", BODY + " T3 PV-V9 (arm hair the lightest body site)", ""),
        redness_add=T(0.0, 0.0, 0.0, "", "E", "-", ""),
    ),
    # ------------------------------------------------------------------------------------------- terminal limb (K4)
    "terminal_limb": _kind(
        "Terminal limb hair", "lower leg (calf, shin), distal forearm", BODY + " section 1 K4",
        diameter_um=T(42.0, 30.0, 60.0, "um", "M", BODY + " T1 (calf 42 mixed sexes incl. vellus, weak; women's "
                      "leg in um NOT FOUND)", "v3 11-05: was 50 (E) inside 42-66; the measured calf 42 used, range widened down "
                      "(E) for a fair woman's fine leg hair: a visual check (V) read 50 um terminal as a man's leg"),
        ellipticity=T(2.23, 2.16, 2.30, "a/b", "M", BODY + " T2 B1 (women's lower leg 2.23 +- 0.07, n 16)",
                      "the major axis lies parallel to the skin"),
        major_axis=T("skin", label="M", source=BODY + " T2 B1", note="AT THE EXIT the wide face lies along the skin (B1, stated): "
                     "there it bends toward / away. Along the lying run of a hair that curls sideways in the skin's plane "
                     "(bh01's C shapes) the curl's plane holds the MINOR axis, so the section turns on edge (D; a flat fibre "
                     "twists far more easily than it bends sideways). Resolved 11-05 close-out (body hair v5): both hold, the "
                     "roll turns from flat at the exit to on edge along the curled run; the GPU kernel (fv8_body_hair4/5) "
                     "draws the run edge-on and does not model the turn in the first ~0.3 mm (U, small at 39-65 um/px)"),
        medulla_index=T(0.1, 0.0, 0.2, "width ratio", "E", BODY + " T2 PV-V9 (less visible than axilla, qualitative)", ""),
        medullated_share=T(0.3, 0.1, 0.5, "share", "E", BODY + " T2", ""),
        cuticle_layers=T(7.0, 5.0, 9.0, "layers", "E", AXP + " T1 V9 (lower leg among the most overlapping layers)", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", BODY + " T2", ""),
        **_keratin(),
        growth_mm_day=T(0.14, 0.14, 0.30, "mm/day", "M", BODY + " T4 B2 (women's leg 0.14); PV-V7 0.18-0.485 (conflict)", ""),
        anagen_days=T(30.0, 20.0, 45.0, "days", "E", BODY + " T4 (lower-leg anagen NOT FOUND); PV-V8 women's thigh 22 d",
                      "v3 11-05: was 60 (30-110, E): 15.4 mm max read long (V); a fair woman's leg reference "
                      "(references/body_hair/bh01, CC BY 2.0) shows 2.5-5 mm hairs (E: scale from the calf's width)"),
        telogen_share=T(0.7, 0.4, 0.85, "share", "M", BODY + " T4 B5 (40-85 % of body hair in telogen)", "weak"),
        length_mm=T(4.2, 2.8, 6.3, "mm", "D", "growth x anagen (E anagen)", "natural length NOT FOUND; v3: was 8.4 (4.2-15.4)"),
        taper_mm=T(2.0, 1.0, 4.0, "mm", "E", BODY + " T2", ""),
        tip_ratio=T(0.25, 0.1, 0.5, "tip/root", "E", "-", ""),
        curl_type=T(None, label="E", source=BODY + " T1 B5 ('curlier than scalp', men, weak)"),
        cd_mm=T(20.0, 10.0, 40.0, "mm", "E", BODY + " section 1 (intrinsic curvature NOT FOUND)", ""),
        curl_index=T(1.05, 1.0, 1.2, "i", "E", "-", ""),
        flat=T(0.0, 0.0, 0.5, "", "E", "-", "the flat section bends in the skin's normal plane"),
        twist_per_m=T(20.0, 0.0, 80.0, "1/m", "E", "no measurement", ""),
        emergence_deg=T(15.0, 8.0, 25.0, "deg", "E", BODY + " T5 PV-V9 (among the smallest of six sites, qualitative)", ""),
        density=T(15.6, 10.0, 21.0, "1/cm2", "M", BODY + " T5 MB-R15 (calf 15.6 follicles)", ""),
        unit_hairs=T(1.0, 1.0, 1.1, "hairs/unit", "M", BODY + " T5 B12b (mostly single)", "weak"),
        pigment_x=T(0.35, 0.2, 0.9, "x scalp", "E", BODY + " T3 PV-V9 ranking (between arm, the lightest, and axilla, the darkest)",
                    "v3 11-05: was 0.95 (D, the mean of intermediate 0.4 and axillary 1.5 = her scalp root's melanin): read dark "
                    "(V); for HER fair ash-blond phenotype no darker than her scalp lengths, fitted on the GPU shin against "
                    "bh01's hair / skin contrast (BODY_HAIR.md v3); v4 11-05: 0.55 -> 0.35, chosen from v3's GPU probes "
                    "(0.35 the closest to bh01; range floor 0.3 -> 0.2)"),
        redness_add=T(0.0, 0.0, 0.0, "", "E", "-", ""),
    ),
    # ------------------------------------------------------------------------------------------- terminal midline (K5)
    "terminal_midline": _kind(
        "Terminal midline and areolar hair", "lower-abdomen midline (linea alba), areola rim; upper lip / chin in some women",
        BODY + " section 1 K5",
        diameter_um=T(66.0, 60.0, 75.0, "um", "M", BODY + " T1 B3 (women's upper-lip terminal 66 +- 6)",
                      "linea alba and areola NOT FOUND"),
        ellipticity=T(1.3, 1.1, 1.5, "a/b", "E", BODY + " section 1 (NOT FOUND)", ""),
        medulla_index=T(0.15, 0.0, 0.25, "width ratio", "E", "-", ""),
        medullated_share=T(0.5, 0.2, 0.8, "share", "E", "-", ""),
        cuticle_layers=T(6.0, 4.0, 8.0, "layers", "E", "-", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", "-", ""),
        **_keratin(),
        growth_mm_day=T(0.3, 0.18, 0.45, "mm/day", "E", BODY + " T4 PV-V7 (body sites 0.18-0.485)", ""),
        anagen_days=T(90.0, 60.0, 150.0, "days", "E", "NOT FOUND", ""),
        telogen_share=T(0.7, 0.4, 0.85, "share", "E", BODY + " T4 B5", ""),
        length_mm=T(27.0, 10.8, 67.5, "mm", "D", "growth x anagen (E)", ""),
        taper_mm=T(3.0, 1.0, 6.0, "mm", "E", "-", ""),
        tip_ratio=T(0.25, 0.1, 0.5, "tip/root", "E", "-", ""),
        curl_type=T(None, label="U", source="NOT FOUND"),
        cd_mm=T(20.0, 10.0, 40.0, "mm", "U", "NOT FOUND", ""),
        curl_index=T(1.1, 1.0, 1.4, "i", "U", "NOT FOUND", ""),
        flat=T(0.3, 0.0, 1.0, "", "U", "NOT FOUND", ""),
        twist_per_m=T(40.0, 0.0, 120.0, "1/m", "U", "NOT FOUND", ""),
        emergence_deg=T(20.0, 10.0, 40.0, "deg", "U", "NOT FOUND", ""),
        density=T(3.0, 0.0, 15.0, "1/cm2", "U", BODY + " T8 (prevalence by mFG only, B4, PV-V10, PV-V11)",
                  "a few hairs where present; prevalence differs widely by cohort"),
        unit_hairs=T(1.0, 1.0, 1.0, "hairs/unit", "E", "-", ""),
        pigment_x=T(1.0, 0.7, 1.5, "x scalp", "E", BODY + " T3 PV-V9 (terminal body hair darker than arm hair)", ""),
        redness_add=T(0.0, 0.0, 0.0, "", "E", "-", ""),
    ),
    # ------------------------------------------------------------------------------------------- axillary
    "axillary": _kind(
        "Axillary hair (adult woman)", "axillary vault: central (densest, fastest), thoracic and brachial margins; dome-shaped",
        AXP + " section 1, T1-T6",
        diameter_um=T(73.0, 58.0, 81.0, "um", "M", AXP + " T1 AP1 (73.1, 95 % CI 65-81, Ghanaian women 15-20 y); AP10 "
                      "58-60 Korean women", "about 1.03 x the same person's scalp diameter (AP1, D)"),
        ellipticity=T(1.4, 1.3, 1.5, "a/b", "E", AXP + " section 1 ('elliptical', V9; a/b NOT FOUND)", ""),
        medulla_index=T(0.19, 0.17, 0.21, "width ratio", "M", AXP + " T1 AP1 (women 0.19, SE 0.01)", ""),
        medullated_share=T(0.9, 0.8, 0.95, "share", "M", AXP + " T1 AP1 (27 of 30 women: 24 continuous, 3 discontinuous)", ""),
        cuticle_layers=T(8.0, 6.0, 10.0, "layers", "E", AXP + " T1 V9 (most overlapping layers, qualitative)", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", AXP + " T1 (scalp carried, hair.md R34)", ""),
        **_keratin(T(0.15, 0.08, 0.3, "", "E", AXP + " section 0 (abraded tips, thick cuticle: wider lobes, E)",
                     "no measurement; the scalp's 0.10 widened for wear")),
        growth_mm_day=T(0.39, 0.30, 0.45, "mm/day", "M", AXP + " T2 AP14 (~12 mm/month), V7 (near scalp rates)", "weak"),
        anagen_days=T(120.0, 90.0, 150.0, "days", "M", AXP + " T2 AP15 (~4 months)",
                      "AP13's 44-77 weeks conflicts; the shorter fits the observed 10-50 mm"),
        telogen_share=T(0.7, 0.5, 0.7, "share", "M", AXP + " T2 AP15 (30 / 70 %)", "weak"),
        length_mm=T(30.0, 10.0, 50.0, "mm", "E", AXP + " T2 (forensic summary 10-50 mm); cap 0.4 x 120 = 48 D", ""),
        taper_mm=T(2.0, 0.5, 4.0, "mm", "E", AXP + " T1 (blunt, abraded or frayed tips)", ""),
        tip_ratio=T(0.5, 0.3, 0.8, "tip/root", "E", AXP + " T1 (abraded tips)", ""),
        curl_type=T(None, label="E", source=AXP + " section 1 (straighter than lower-body hair; curl NOT FOUND)"),
        cd_mm=T(40.0, 20.0, 60.0, "mm", "E", AXP + " section 1 (placeholder 20-60 mm)", ""),
        curl_index=T(1.1, 1.0, 1.3, "i", "E", "-", ""),
        flat=T(0.5, 0.0, 1.0, "", "E", "-", ""),
        twist_per_m=T(60.0, 20.0, 150.0, "1/m", "E", "no measurement", ""),
        emergence_deg=T(20.0, 10.0, 30.0, "deg", "E", AXP + " T3 V9 (among the smallest of body sites; degrees unread)", ""),
        density=T(15.0, 13.0, 65.0, "1/cm2", "M", AXP + " T3 AP10 (13.8-14.9 Korean women, folliscope); AP15 65 (table)",
                  "conflict; 65 x 30 % anagen = 19.5 growing (D)"),
        unit_hairs=T(1.2, 1.0, 1.5, "hairs/unit", "E", AXP + " T3 (NOT FOUND; body hair mostly 1-2-hair units)", ""),
        pigment_x=T(1.5, 1.0, 2.5, "x scalp", "E", AXP + " T4 AP18 (near the brows), PV-V9 (darkest body sites)", ""),
        redness_add=T(0.0, 0.0, 0.1, "", "E", AXP + " T4 (darker or redder than a blond scalp, report)", ""),
    ),
    # ------------------------------------------------------------------------------------------- lower_body
    "lower_body": _kind(
        "Lower-body hair (adult woman)", "the lower-body hair field (central denser) and the medial thigh edge "
        "(Tanner 5)", AXP + " section 1, T1-T6",
        diameter_um=T(88.0, 78.0, 100.0, "um", "M", AXP + " T1 AP1 (88.4, 95 % CI 78-99, Ghanaian women 15-20 y)",
                      "coarsest on the body; ~1.24 x the same person's scalp (AP1, D)"),
        ellipticity=T(1.65, 1.5, 1.8, "a/b", "E", AXP + " section 1 (placeholder 1.5-1.8; irregular, twisted contours)", ""),
        medulla_index=T(0.24, 0.18, 0.30, "width ratio", "M", AXP + " T1 AP1 (women 0.24, SE 0.03)", ""),
        medullated_share=T(0.87, 0.75, 0.95, "share", "M", AXP + " T1 AP1 (26 of 30 women)", ""),
        cuticle_layers=T(9.0, 7.0, 11.0, "layers", "E", AXP + " T1 AP3 (more scales, thicker cuticle than scalp)", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "E", AXP + " T1 (scalp carried)", ""),
        **_keratin(T(0.15, 0.08, 0.3, "", "E", AXP + " section 0 (abraded tips, thick cuticle, E)",
                     "no measurement; the scalp's 0.10 widened")),
        growth_mm_day=T(0.30, 0.25, 0.40, "mm/day", "M", AXP + " T2 AP14 (~9 mm/month)", "weak"),
        anagen_days=T(150.0, 120.0, 180.0, "days", "M", AXP + " T2 AP15 ('months'), AP14",
                      "AP13's 44-77 weeks conflicts with the observed 10-60 mm"),
        telogen_share=T(0.7, 0.5, 0.7, "share", "M", AXP + " T2 AP15 (30 / 70 %), AP6 (women more telogen)", "weak"),
        length_mm=T(35.0, 10.0, 60.0, "mm", "E", AXP + " T2 (forensic 10-60 mm); cap 36-54 D", ""),
        taper_mm=T(3.0, 1.0, 6.0, "mm", "E", AXP + " T1 (uncut tips tapered, often abraded)", ""),
        tip_ratio=T(0.3, 0.1, 0.6, "tip/root", "E", "-", ""),
        curl_type=T("curly", label="E", source=AXP + " section 1 (curly, wiry, buckled; AP12)", note="its own cd below"),
        cd_mm=T(15.0, 5.0, 30.0, "mm", "E", AXP + " section 1 (placeholder 5-30 mm)", "with abrupt kinks (buckling)"),
        curl_index=T(1.6, 1.3, 2.0, "i", "E", "-", ""),
        flat=T(0.8, 0.5, 1.0, "", "E", "-", ""),
        twist_per_m=T(102.1, 60.0, 200.0, "1/m", "D", "the kind's helix torsion (fibre.helix_tau at cd 15, i 1.6, flat 0.8)",
                      "'twisted and constricted' (forensic text)"),
        emergence_deg=T(60.0, 30.0, 60.0, "deg", "M", AXP + " T3 AP4 (~60 deg centrally, flatter at the periphery)",
                        "V9 ranks it among the smallest (method unknown)"),
        density=T(70.0, 18.0, 70.0, "1/cm2", "E", AXP + " T3 AP15 (table, origin unverified); AP4 centre > periphery", ""),
        unit_hairs=T(1.2, 1.0, 1.5, "hairs/unit", "E", AXP + " T3 (NOT FOUND)", ""),
        pigment_x=T(1.4, 1.0, 2.5, "x scalp", "E", AXP + " T4 AP18 ('most similar to the eyebrows')",
                    "set equal to the brow kind by default"),
        redness_add=T(0.0, 0.0, 0.1, "", "E", AXP + " T4", ""),
    ),
    # ------------------------------------------------------------------------------------------- lower_body_sparse
    "lower_body_sparse": _kind(
        "Sparse lower-body hair (adult woman)", "the lower body's rear: present only sometimes, a sparse outrider of the lower-body field",
        AXP + " section 1, T6",
        diameter_um=T(70.0, 50.0, 90.0, "um", "U", AXP + " T6 (NOT FOUND; lower-body-like assumed)", ""),
        ellipticity=T(1.5, 1.2, 1.8, "a/b", "U", "NOT FOUND", ""),
        medulla_index=T(0.2, 0.0, 0.3, "width ratio", "U", "NOT FOUND", ""),
        medullated_share=T(0.7, 0.3, 0.9, "share", "U", "NOT FOUND", ""),
        cuticle_layers=T(8.0, 6.0, 10.0, "layers", "U", "NOT FOUND", ""),
        alpha_deg=T(3.0, 2.5, 3.5, "deg", "U", "NOT FOUND", ""),
        **_keratin(),
        growth_mm_day=T(0.3, 0.2, 0.4, "mm/day", "U", "NOT FOUND (lower body carried)", ""),
        anagen_days=T(90.0, 60.0, 150.0, "days", "U", "NOT FOUND", ""),
        telogen_share=T(0.7, 0.5, 0.8, "share", "U", "NOT FOUND", ""),
        length_mm=T(20.0, 5.0, 40.0, "mm", "U", "NOT FOUND", ""),
        taper_mm=T(2.0, 1.0, 4.0, "mm", "U", "NOT FOUND", ""),
        tip_ratio=T(0.3, 0.1, 0.6, "tip/root", "U", "NOT FOUND", ""),
        curl_type=T(None, label="U", source="NOT FOUND"),
        cd_mm=T(20.0, 5.0, 40.0, "mm", "U", "NOT FOUND", ""),
        curl_index=T(1.3, 1.0, 2.0, "i", "U", "NOT FOUND", ""),
        flat=T(0.6, 0.0, 1.0, "", "U", "NOT FOUND", ""),
        twist_per_m=T(60.0, 0.0, 150.0, "1/m", "U", "NOT FOUND", ""),
        emergence_deg=T(20.0, 10.0, 40.0, "deg", "U", "NOT FOUND", ""),
        density=T(5.0, 0.0, 20.0, "1/cm2", "U", AXP + " T6 R36 ('sometimes'), AP21", ""),
        unit_hairs=T(1.0, 1.0, 1.2, "hairs/unit", "U", "NOT FOUND", ""),
        pigment_x=T(1.4, 1.0, 2.5, "x scalp", "U", "as lower body", ""),
        redness_add=T(0.0, 0.0, 0.1, "", "U", "as lower body", ""),
    ),
}

# the derived (D) values and how they re-derive (check() and the tests re-run each; tolerance 2 %)
_k = lambda n, t: HAIR_KINDS[n]["traits"][t]["value"]  # noqa: E731
DERIVED = {
    ("scalp", "length_mm"): lambda: _k("scalp", "growth_mm_day") * _k("scalp", "anagen_days"),
    ("scalp", "unit_hairs"): lambda: (1 * 60 + 2 * 29 + 3 * 9 + 4 * 3) / 101.0,
    ("scalp", "twist_per_m"): lambda: FB.hair_type("wavy")["tau"],
    ("scalp_edge", "length_mm"): lambda: _k("scalp_edge", "growth_mm_day") * _k("scalp_edge", "anagen_days"),
    ("scalp_edge", "density"): lambda: 0.5 * _k("scalp", "density"),
    ("eyebrow", "length_mm"): lambda: _k("eyebrow", "growth_mm_day") * _k("eyebrow", "anagen_days"),
    ("eyebrow", "density"): lambda: 300.0 / (5.0 * 0.9),
    ("eyelash_upper", "ellipticity"): lambda: 1.0 / 0.826,
    ("eyelash_upper", "medulla_index"): lambda: math.sqrt(0.076),
    ("eyelash_upper", "density"): lambda: 15.24 / 2.0,
    ("eyelash_upper", "count"): lambda: 1715.0 / 10.0,
    ("eyelash_lower", "density"): lambda: 6.24 / 2.0,
    ("vellus_face", "anagen_days"): lambda: 0.5 * (0.5 / 0.03 + 2.0 / 0.03),
    ("vellus_face", "telogen_share"): lambda: 1.0 - 0.49,
    ("vellus_body", "telogen_share"): lambda: 1.0 - 0.315,
    ("vellus_body", "length_mm"): lambda: 0.13 * 22.0,
    ("intermediate_body", "telogen_share"): lambda: 1.0 - 22.0 / (0.5 * (84.0 + 106.0)),
    ("intermediate_body", "length_mm"): lambda: _k("intermediate_body", "growth_mm_day") * 22.0,
    ("terminal_limb", "length_mm"): lambda: _k("terminal_limb", "growth_mm_day") * _k("terminal_limb", "anagen_days"),
    ("scalp", "pigment_x"): lambda: 1.0,
    ("scalp", "redness_add"): lambda: 0.0,
    ("terminal_midline", "length_mm"): lambda: _k("terminal_midline", "growth_mm_day") * _k("terminal_midline", "anagen_days"),
    ("lower_body", "twist_per_m"): lambda: FB.helix_tau(2000.0 / _k("lower_body", "cd_mm"), _k("lower_body", "flat"), _k("lower_body", "curl_index")),
}

# where each kind grows on an adult woman (literature, no person): region -> {kind: share of the follicles}. Region names
# are the common anatomical zones (the skin-zone names a case's chart uses); sub-regions carry their own rows. {} = no hair.
REGION_KINDS = {
    "scalp": {"scalp": 1.0}, "scalp_edge": {"scalp_edge": 1.0},
    "brow": {"eyebrow": 1.0}, "lid_margin_upper": {"eyelash_upper": 1.0}, "lid_margin_lower": {"eyelash_lower": 1.0},
    "forehead": {"vellus_face": 1.0}, "cheek": {"vellus_face": 1.0}, "nose": {"vellus_face": 1.0},
    "chin": {"vellus_face": 1.0}, "upper_lip": {"vellus_face": 1.0}, "eyelids": {"vellus_face": 1.0},
    "ear": {"vellus_face": 1.0}, "face_rest": {"vellus_face": 1.0}, "neck": {"vellus_face": 0.5, "vellus_body": 0.5},
    "chest": {"vellus_body": 1.0}, "breast": {"vellus_body": 1.0}, "abdomen": {"vellus_body": 1.0},
    "upper_back": {"vellus_body": 1.0}, "lower_back": {"vellus_body": 1.0}, "buttocks": {"vellus_body": 1.0},
    "upper_arm": {"vellus_body": 0.5, "intermediate_body": 0.5}, "elbow": {"vellus_body": 0.7, "intermediate_body": 0.3},
    "forearm_proximal": {"vellus_body": 0.7, "intermediate_body": 0.3},
    "forearm_distal": {"intermediate_body": 0.5, "terminal_limb": 0.5},
    "hand_dorsum": {"vellus_body": 1.0}, "thigh": {"intermediate_body": 1.0}, "knee": {"intermediate_body": 1.0},
    "lower_leg": {"terminal_limb": 1.0}, "foot_dorsum": {"vellus_body": 1.0},
    "axilla": {"axillary": 1.0}, "lower_body_front": {"lower_body": 1.0}, "lower_body_sides": {"lower_body": 1.0}, "thigh_medial_edge": {"lower_body": 1.0},
    "lower_body_rear": {"lower_body_sparse": 1.0},
    "areola_rim": {"terminal_midline": 1.0}, "linea_alba": {"terminal_midline": 1.0},
    "palm": {}, "sole": {}, "lips": {}, "nipple_areola": {}, "lower_body_inner": {}, "nail": {}, "mucosa": {},
}
REGION_KINDS_SRC = {
    "_": BODY + " section 2 (region map), " + AXP + " section 2, " + HEAD + " section 1. Shares between kinds in a mixed "
         "region are E (the share of follicles per kind is NOT FOUND anywhere); glabrous skin has none (palm, sole, lips, "
         "nipple, inner lower-body skin, nail bed, mucosa)",
    "cheek": "PV-V10: no terminal cheek hair in 125 Norwegian women (vellus only, M)",
    "upper_lip": "B3: 0 +- 1 terminal hairs per view in non-hirsute women (M); B4 mFG >= 1 in 76 % of a Pakistani cohort (W)",
    "forearm_distal": "terminal toward the wrist, vellus toward the elbow (Wikipedia 'Arm hair', W); B9 forearm converges on the elbow",
    "areola_rim": "PV-V10: breast terminal hair only in a small circumareolar area, in some women; prevalence NOT FOUND",
    "linea_alba": "not in the female Tanner-5 picture (gen R30); mFG 'lower abdomen' >= 1 common in B4 (W); Western sample NOT FOUND",
    "thigh_medial_edge": "Tanner 5: the lower-body field spreads to the medial thighs (gen R30, W)",
    "lower_body_rear": "R36 / AP21: present 'sometimes'",
}
# follicles per cm2 by region where measured (MB-R15 Jonsson 2017, 8 F of 15 adults; PV-V6 back vellus), else the kind's
REGION_DENSITY = {"forehead": 285.0, "neck": 47.3, "upper_arm": 45.9, "forearm_proximal": 40.3, "forearm_distal": 40.3,
                  "chest": 28.4, "lower_back": 24.7, "thigh": 21.0, "abdomen": 16.9, "lower_leg": 15.6,
                  "_src": BODY + " T5 MB-R15 (follicles per cm2 by site, cyanoacrylate stripping), M"}

# treatment acts on the kind's grown hair, as a layer on top of it (grooming never changes the kind)
TREATMENTS = {
    "natural": "untreated: every hair at its own point of its growth cycle, natural tapered tips",
    "trim": "cut at max_mm: longer hairs end blunt at max_mm (the taper is cut off)",
    "shave": "cut at the skin `days` ago: growing hairs are back to rate x days, blunt; resting hairs shed (E: 0)",
    "remove": "waxed / lasered / plucked recently: no hair",
}


# ======================================================================================================== access
def kind(name):
    """a deep copy of one kind"""
    if name not in HAIR_KINDS:
        raise KeyError("unknown hair kind %r (known: %s)" % (name, ", ".join(HAIR_KINDS)))
    return copy.deepcopy(HAIR_KINDS[name])


def v(name, trait):
    """one trait's value"""
    return HAIR_KINDS[name]["traits"][trait]["value"]


def rng_of(name, trait):
    r = HAIR_KINDS[name]["traits"][trait]["range"]
    val = v(name, trait)
    return (val, val) if r is None else r


# ======================================================================================================== the fibre
def hair_type_spec(name):
    """the dict engine/hair/fibre.py reads as the style's fibre.type: {"base": HAIR_TYPES name, cd_mm, curl_index, flat,
    ellipticity, diameter_um}. A kind without a curl_type takes 'straight' as its base with its own traits (a lash's
    single arc: cd_mm with curl_index 1 = a constant curvature, no helix)"""
    t = HAIR_KINDS[name]["traits"]
    return {"base": t["curl_type"]["value"] or "straight", "cd_mm": t["cd_mm"]["value"],
            "curl_index": float(t["curl_index"]["value"]), "flat": float(t["flat"]["value"]),
            "ellipticity": float(t["ellipticity"]["value"]), "diameter_um": float(t["diameter_um"]["value"])}


def curl(name):
    """the kind's rest curve through fibre.hair_type: kappa, tau (1/m), curl (a helix), index, period_m, coil_d_mm, plus
    the free planar wave it would draw (wavelength 2 pi / tau, amplitude kappa / tau^2; fibre.py's planar limit)"""
    h = FB.hair_type(hair_type_spec(name))
    out = {k: h[k] for k in ("kappa", "tau", "curl", "index", "period_m", "coil_d_mm", "cd_mm")}
    out["arc_radius_mm"] = 1000.0 / h["kappa"] if h["kappa"] > 0 else float("inf")
    out["wave_lambda_mm"] = 2000.0 * math.pi / h["tau"] if h["tau"] > 0 else 0.0
    out["wave_amp_mm"] = 1000.0 * h["kappa"] / h["tau"] ** 2 if h["tau"] > 0 else 0.0
    return out


def mechanics(d_um, E_gpa=3.1, rho=1320.0, ell=1.0, L_mm=None):
    """a fibre of equivalent-circle diameter d_um: EI round / weak / strong axis of the equal-area oval (a/b = ell:
    I x 1/ell and x ell), weight per length, gravity length (EI / w)^(1/3), and the self-weight tip sag of a free horizontal
    hair of length L_mm, w L^4 / (8 EI) (research T4 / T5 / T6's formulas)"""
    r = 0.5 * d_um * 1e-6
    E = E_gpa * 1e9
    EI = E * math.pi * r ** 4 / 4.0
    w = rho * G * math.pi * r * r
    out = {"EI_Nm2": EI, "EI_weak_Nm2": EI / ell, "EI_strong_Nm2": EI * ell, "w_N_per_m": w,
           "mass_mg_per_m": rho * math.pi * r * r * 1e6, "gravity_length_mm": 1000.0 * (EI / w) ** (1.0 / 3.0)}
    if L_mm is not None:
        L = L_mm / 1000.0
        out["sag_mm"] = 1000.0 * w * L ** 4 / (8.0 * EI)
        out["sag_share"] = out["sag_mm"] / L_mm
    return out


def section(name):
    """root / mid / tip radius (um), ellipticity, the kind's own stiffness and weight, and how gravity acts on it"""
    t = HAIR_KINDS[name]["traits"]
    d = float(t["diameter_um"]["value"])
    mid = float(t["mid_ratio"]["value"]) if "mid_ratio" in t else 1.0
    m = mechanics(d, float(t["E_gpa"]["value"]), float(t["rho"]["value"]), float(t["ellipticity"]["value"]),
                  float(t["length_mm"]["value"]))
    m.update(r_root_um=0.5 * d, r_mid_um=0.5 * d * mid, r_tip_um=0.5 * d * float(t["tip_ratio"]["value"]),
             ell=float(t["ellipticity"]["value"]), taper_mm=float(t["taper_mm"]["value"]),
             gravity_shapes_it=bool(m["gravity_length_mm"] < float(t["length_mm"]["value"])))
    return m


def fibre_cfg(name):
    """(fibre block, radius block) for fibre.props(cfg, radius_cfg): the kind's E, rho, taper, natural curvature, type and
    growth cycle in fibre.py's units (GPa, kg/m3, mm, 1/mm, mm per month, years). Note: fibre.py's cycle ranges were set
    for scalp hair (rate >= 3 mm/month); vellus is slower - resolve() does not clip, the values pass through"""
    t = HAIR_KINDS[name]["traits"]
    cd = t["cd_mm"]["value"]
    a0, a1 = rng_of(name, "anagen_days")
    cfg = {"E_gpa": float(t["E_gpa"]["value"]), "rho": float(t["rho"]["value"]), "taper_mm": float(t["taper_mm"]["value"]),
           "kappa0_mm": (2.0 / float(cd)) if cd else FB.KAPPA_REF_MM, "type": hair_type_spec(name),
           "cycle": {"on": True, "rate_mm_month": float(t["growth_mm_day"]["value"]) * FB.DAYS_PER_MONTH,
                     "anagen_years": [a0 / 365.25, a1 / 365.25], "telogen_frac": float(t["telogen_share"]["value"])}}
    d = float(t["diameter_um"]["value"])
    rad = {"root_mm": d / 2000.0, "tip_mm": d / 2000.0 * float(t["tip_ratio"]["value"])}
    return cfg, rad


# ======================================================================================================== length
def length_cap_mm(name):
    """the natural length cap = growth x anagen: (from the ranges' low ends, the values, the high ends)"""
    g0, g1 = rng_of(name, "growth_mm_day")
    a0, a1 = rng_of(name, "anagen_days")
    return g0 * a0, v(name, "growth_mm_day") * v(name, "anagen_days"), g1 * a1


def cycle_lengths(name, n, rng, treatment=None):
    """grown length per hair (mm) from the kind's cycle (as fibre.cycle_lengths): each follicle's anagen A ~ U(the
    anagen range), its time t ~ U(0, A / (1 - telogen)), held at A once it rests; L = rate x t. treatment: None /
    {"mode": "natural" | "trim" (max_mm) | "shave" (days) | "remove"}. -> (L mm, resting bool, blunt bool)"""
    rate = float(v(name, "growth_mm_day"))
    a0, a1 = rng_of(name, "anagen_days")
    tel = float(v(name, "telogen_share"))
    A = rng.uniform(a0, a1, n)
    t = rng.random(n) * A / (1.0 - tel)
    resting = t >= A
    L = rate * np.minimum(t, A)
    blunt = np.zeros(n, bool)
    tr = dict(treatment or {"mode": "natural"})
    mode = tr.get("mode", "natural")
    if mode == "trim":
        mx = float(tr["max_mm"]); blunt = L > mx; L = np.minimum(L, mx)
    elif mode == "shave":                                   # growing hairs regrow rate x days; resting club hairs: 0 (E)
        L = np.where(resting, 0.0, rate * float(tr["days"])); blunt = L > 0
    elif mode == "remove":
        L = np.zeros(n); blunt = np.zeros(n, bool)
    elif mode != "natural":
        raise ValueError("treatment mode %r (known: %s)" % (mode, ", ".join(TREATMENTS)))
    return L, resting, blunt


# ======================================================================================================== colour
def pigment(name, scalp, scalp_d_um=None):
    """the kind's pigment from the person's scalp colour block (melanin, redness, ash, root_add; engine/hair/colour.py's
    absorption model): eumelanin + pheomelanin concentration q = conc(scalp root melanin) x pigment_x, redness + redness_add,
    the scalp's ash. Chiang's sigma_a is per unit RADIUS (the fibre normalised to r = 1), so the same melanin in a thinner
    fibre absorbs less along its path: sigma_kind = sigma(m, red, ash) x d_kind / d_scalp (hair_kinds_vellus_body.md T3, D).
    scalp_d_um: the person's scalp diameter (default the scalp kind's). -> dict(melanin, redness, ash, path_x, sigma (3,))"""
    m_root = float(scalp.get("melanin", 0.45)) + float(scalp.get("root_add", 0.0))
    q = float(HC.conc(m_root)) * float(v(name, "pigment_x"))
    m = 1.0 - math.exp(-q)
    red = float(np.clip(float(scalp.get("redness", 0.0)) + float(v(name, "redness_add")), 0.0, 1.0))
    ash = float(scalp.get("ash", 0.0))
    d0 = float(scalp_d_um or v("scalp", "diameter_um"))
    path = float(v(name, "diameter_um")) / d0
    sig = np.asarray(HC.sigma(m, red, ash), np.float64) * path
    return {"melanin": m, "redness": red, "ash": ash, "path_x": path, "sigma": sig, "scalp_root_melanin": m_root}


# ======================================================================================================== follicles
def follicles(name, region=None):
    """where and how the kind leaves the skin: density (region's measured value when known), mean spacing, emergence
    angle to the skin plane and its spread (a quarter of the range), hairs per follicular unit, rows (lashes)"""
    t = HAIR_KINDS[name]["traits"]
    dens = float(t["density"]["value"])
    unit = t["density"]["unit"]
    if region is not None and region in REGION_DENSITY and unit == "1/cm2":
        dens = float(REGION_DENSITY[region])
    e0, e1 = rng_of(name, "emergence_deg")
    out = {"density": dens, "density_unit": unit, "emergence_deg": float(t["emergence_deg"]["value"]),
           "emergence_sd_deg": (e1 - e0) / 4.0, "unit_hairs": float(t["unit_hairs"]["value"])}
    out["spacing_mm"] = (10.0 / math.sqrt(dens) if unit == "1/cm2" else 1.0 / dens) if dens > 0 else float("inf")
    if "rows" in t:
        out["rows"] = float(t["rows"]["value"])
    if "count" in t:
        out["count"] = float(t["count"]["value"])
    return out


# ======================================================================================================== the field hair
def field_desc(name, scalp=None, over=None):
    """the field hair's description for one kind, in the keys the fv8 field-hair lock DESC uses (mm, um, 1/m, deg):
    length_mm (the natural length), length_jitter (half the length range / length, capped 0.9), taper_mm, r_root_um,
    r_tip_um, r_jitter (a quarter of the diameter range / diameter), ell, tau_per_m (the section's twist), beta_m,
    beta_n, eta, alpha_deg; sigma_p50 (3,) when the person's scalp colour block is given; and the kind's curl (kappa
    per m, cd_mm, curl_index, flat, curl_type), emergence and density. over: the case's own values on top (its
    measured look for this kind)"""
    t = HAIR_KINDS[name]["traits"]
    d = float(t["diameter_um"]["value"]); d0, d1 = rng_of(name, "diameter_um")
    L = float(t["length_mm"]["value"]); l0, l1 = rng_of(name, "length_mm")
    c = curl(name)
    out = {"kind": name, "length_mm": L, "length_jitter": min(0.9, 0.5 * (l1 - l0) / L), "taper_mm": float(t["taper_mm"]["value"]),
           "r_root_um": 0.5 * d, "r_tip_um": 0.5 * d * float(t["tip_ratio"]["value"]),
           "r_mid_ratio": float(t["mid_ratio"]["value"]) if "mid_ratio" in t else 1.0,
           "r_jitter": 0.25 * (d1 - d0) / d, "ell": float(t["ellipticity"]["value"]), "tau_per_m": float(t["twist_per_m"]["value"]),
           "beta_m": float(t["beta_m"]["value"]), "beta_n": float(t["beta_n"]["value"]), "eta": float(t["eta"]["value"]),
           "alpha_deg": float(t["alpha_deg"]["value"]), "medulla_index": float(t["medulla_index"]["value"]),
           "curl_type": t["curl_type"]["value"], "cd_mm": t["cd_mm"]["value"], "curl_index": float(t["curl_index"]["value"]),
           "flat": float(t["flat"]["value"]), "kappa_per_m": c["kappa"], "helix_tau_per_m": c["tau"],
           "emergence_deg": float(t["emergence_deg"]["value"]), "density": float(t["density"]["value"]),
           "density_unit": t["density"]["unit"], "growth_mm_day": float(t["growth_mm_day"]["value"])}
    if scalp is not None:
        p = pigment(name, scalp)
        out["sigma_p50"] = [float(x) for x in p["sigma"]]
        out["melanin"] = p["melanin"]; out["redness"] = p["redness"]; out["ash"] = p["ash"]
    if over:
        out.update(over)
    return out


# ======================================================================================================== checks
def check(tol=0.02):
    """every rule the tests hold -> list of problems (empty = all good)"""
    bad = []
    for n, K in HAIR_KINDS.items():
        for f in ("label", "regions", "research"):
            if not K.get(f):
                bad.append("%s: no %s" % (n, f))
        tr = K["traits"]
        for t in REQUIRED:
            if t not in tr:
                bad.append("%s: missing %s" % (n, t)); continue
        for t, x in tr.items():
            if t not in REQUIRED and t not in OPTIONAL:
                bad.append("%s.%s: not a known trait" % (n, t))
            if x.get("label") not in LABELS:
                bad.append("%s.%s: label %r" % (n, t, x.get("label")))
            if not str(x.get("source", "")).strip():
                bad.append("%s.%s: no source" % (n, t))
            val = x.get("value")
            if val is None and t not in NONE_OK:
                bad.append("%s.%s: no value" % (n, t))
            r = x.get("range")
            if r is not None and isinstance(val, (int, float)):
                lo, hi = r
                if not (lo <= hi and lo - 1e-9 <= val <= hi + 1e-9):
                    bad.append("%s.%s: %s outside %s" % (n, t, val, r))
        ct = tr.get("curl_type", {}).get("value")
        if ct is not None and ct not in FB.HAIR_TYPES:
            bad.append("%s: curl_type %r not in fibre.HAIR_TYPES" % (n, ct))
    ty = FB.HAIR_TYPES[v("scalp", "curl_type")]            # the scalp's curl traits are its HAIR_TYPES preset's
    for t in ("cd_mm", "curl_index", "flat"):
        if abs(float(v("scalp", t)) - float(ty[t])) > 1e-9:
            bad.append("scalp.%s %s != HAIR_TYPES[%s] %s" % (t, v("scalp", t), v("scalp", "curl_type"), ty[t]))
    for n, K in HAIR_KINDS.items():                         # every D value carries its arithmetic
        for t, x in K["traits"].items():
            if x.get("label") == "D" and (n, t) not in DERIVED:
                bad.append("%s.%s: labelled D without a derivation in DERIVED" % (n, t))
    for (n, t), f in DERIVED.items():
        x = HAIR_KINDS[n]["traits"][t]
        want = float(f())
        if x["label"] != "D":
            bad.append("%s.%s: has a derivation but label %s" % (n, t, x["label"]))
        if abs(float(x["value"]) - want) > tol * max(abs(want), 1e-9):
            bad.append("%s.%s: %s does not re-derive (%s)" % (n, t, x["value"], round(want, 4)))
    for r, mix in REGION_KINDS.items():
        if mix and abs(sum(mix.values()) - 1.0) > 1e-9:
            bad.append("region %s: shares sum %s" % (r, sum(mix.values())))
        for k in mix:
            if k not in HAIR_KINDS:
                bad.append("region %s: unknown kind %s" % (r, k))
    return bad


def table(names=None, traits=("diameter_um", "ellipticity", "medulla_index", "growth_mm_day", "anagen_days", "length_mm",
                              "cd_mm", "emergence_deg", "density", "pigment_x")):
    """a plain-text table of the kinds (value [label])"""
    names = names or list(HAIR_KINDS)
    rows = ["kind | " + " | ".join(traits)]
    for n in names:
        tr = HAIR_KINDS[n]["traits"]
        rows.append(n + " | " + " | ".join("%s [%s]" % (tr[t]["value"], tr[t]["label"]) for t in traits))
    return "\n".join(rows)
