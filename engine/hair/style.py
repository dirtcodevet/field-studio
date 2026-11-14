"""Hair Studio STYLE schema (Apache-2.0; numpy comes in with the grow blocks): every parameter of a hairstyle with its default, its range and a
note; style files (json) that extend one another; validation; the hair-oil coupling.

A style file is a json object with any subset of DEFAULTS' keys (nested the same way) plus optional
    "extends": "<other style name or path>"   - deep-merged under this one (chains allowed)
    "notes": "..."                              - free text
Lengths are millimetres in the file (metres inside the generator), angles degrees, colours in melanin units (0..1,
the Principled Hair BSDF's melanin parametrisation: Cycles maps m -> concentration -ln(1 - m)).

    python -m engine.hair.style show <style.json>      the resolved style (defaults + extends + the file)
    python -m engine.hair.style schema                 the schema table (key, default, range, note)
"""
import copy
import json
import os
import sys

# (default, (lo, hi) or None, note)
S = {
    "name": ("unnamed", None, "the style's name"),
    "seed": (1028, None, "random seed (deterministic output)"),
    "points": (32, (8, 96), "points per strand, uniform in arc length"),
    "body": {
        "collider": ("head", ["head", "torso"], "the case's collider the hair rests on: head = to the upper chest, torso = "
                                               "to the upper thighs with the arms (long hair needs the body under it)"),
        "head": ("hs2", ["hs2", "fh1", "fh2", "fh3"], "the case's head snapshot: hs2 = as first built, fh1-fh3 = its forehead "
                                               "rebuilt from the bone (the hairline carried with the skin)"),
    },
    "roots": {
        "count": (145000, (1000, 400000), "strands (roots) on the scalp; ~244 / cm2 on 595 cm2 = a natural blonde head"),
        "edge_mm": (4.5, (0, 15), "the feathered hairline: density ramps from edge_floor to 1 over this many mm"),
        "edge_floor": (0.30, (0, 1), "relative density at the boundary"),
        "patch_amp": (0.0, (0, 0.5), "natural density patches: the root density varies +- this over the scalp (0 = uniform)"),
        "patch_mm": (18.0, (5, 60), "... in patches about this size"),
    },
    "hairline": {
        "drop_mm": (2.0, (-10, 10), "the root boundary this far below the traced hairline (the 50 % coverage edge lands ~2 mm"
                                    " inside the roots)"),
        "temple_mm": (2.5, (-10, 10), "+ this much lower at the temples (|phi| temple_phi_deg)"),
        "temple_phi_deg": ([25.0, 50.0], None, "the temple drop ramps in between these azimuths"),
        "edge_short_frac": (0.5, (0, 1), "share of the edge hairs within 3 mm of the boundary that are short"),
        "edge_short_mm": ([15.0, 60.0], (1, 200), "their length"),
        "baby": {
            "frac": (0.0, (0, 0.5), "share of the roots in the band that are baby hairs (vellus-like wisps)"),
            "band_mm": ([-1.5, 3.5], (-6, 10), "the band about the boundary (- = outside it) where baby hairs root"),
            "extra": (0, (0, 20000), "extra baby-hair roots in the band (on top of count)"),
            "length_mm": ([5.0, 24.0], (1, 80), "their length"),
            "radius_scale": ([0.35, 0.50], (0.1, 1), "their radius x the main root radius"),
            "lift_deg": ([8.0, 35.0], (0, 80), "they leave the skin at this angle (they stand, curl and do not lie combed)"),
            "curl": (0.5, (0, 2), "curl: turns of a loose arc over the hair's length"),
            "lighter": (0.10, (0, 0.5), "melanin off (finer, lighter)"),
            "phi_deg": (80.0, (0, 180), "only at the front and temples (|phi| < this)"),
            "outward": (0.75, (0, 1), "their direction: this share out of the hairline toward the face, the rest the comb"),
        },
    },
    "part": {
        "type": ("centre", ["centre", "side", "none"], "part line: centre, side (offset_mm from the midline) or none"),
        "offset_mm": (0.0, (-60, 60), "side part: the line this far off the midline (+ = the subject's left, +x)"),
        "depth_mm": (110.0, (0, 250), "the part runs back this far from the front hairline"),
        "cross_frac": (0.40, (0, 1), "share of the roots on the line that cross it (no scalp line)"),
        "cross_mm": (8.0, (0, 30), "crossing roots within this distance of the line"),
        "cross_front_mm": (50.0, (0, 200), "only in the front this many mm"),
    },
    "flow": {
        "source": ("comb", ["comb", "grow"], "comb = the legacy walk over the scalp (direction field + gravity + "
                                             "collision); grow = hair grown from follicles as elastic rods and settled "
                                             "(DESIGN_v2: follicle, fibre, grow, brush/dry/tuck/hand/tousle/face). "
                                             "'guides' was never built and is refused"),
        "order": ("during", ["during", "after"], "grow mode: when the style's acts meet the hair. during = the legacy "
                                                 "order (generate._grow: the acts on while the hair grows, each new "
                                                 "length combed as it emerges); after = grow once, then style "
                                                 "(DESIGN_grow_then_style: the head grown with no act and cached - "
                                                 "engine/hair/head.py - then the product, the acts on hair that is "
                                                 "already there, release, the final). An engine switch: the sim layer "
                                                 "owns it"),
        "away_from_part": (1.0, (0, 5), "comb: lateral, away from the part (on top of the head)"),
        "back": (0.10, (-3, 5), "comb: toward the back of the head (+ = back), everywhere"),
        "side": (0.0, (-5, 5), "comb: all hair toward one side (+ = the subject's left)"),
        "up": (0.0, (0, 5), "comb: up toward the crown (updos)"),
        "face_back": (1.4, (0, 5), "comb: extra back-sweep over the front and the temples, above the ears"),
        "behind_ears": (True, None, "sweep and tuck the hair behind the ears"),
        "drape_front": (0.0, (0, 4), "the strands rooted in front of the ears' line are drawn forward below the ears until "
                                     "they hang in front of the shoulders (0 = they fall where gravity takes them)"),
        "drape_phi_deg": ([95.0, 120.0], None, "... rooted within the first azimuth, fading out by the second"),
        "temple_sweep": (0.0, (0, 5), "comb: over the temple and the side above the ear, aim at the tuck point behind the "
                                      "ear's top (back and down): the hair from above covers the temple on its way"),
        "gravity": (1.0, (0, 5), "gravity's weight against the comb"),
        "volume": (0.30, (0, 1), "off the scalp the upper layers fan out (outward force x the strand's layer height); "
                                 "the strands do not collide with one another, so without it the mass falls thin"),
        "comb_strength": (2.5, (0, 10), "the comb's weight while a strand lies on the scalp"),
        "stiffness": (0.72, (0, 0.98), "share of the previous direction kept per step (bending stiffness)"),
        "step_mm": (4.0, (1, 10), "walk step"),
        "turn_deg": (4.0, (0, 30), "random turn of each strand's start about its root normal"),
        "guide_k": (4, (1, 12), "guides: the k nearest guides on the same side of the part"),
    },
    "lift": {
        "angle_deg": ([10.0, 20.0], (0, 80), "strands rise from the skin at this angle ..."),
        "plateau_mm": ([1.0, 8.0], (0, 40), "... to a plateau: nape .. crown (the upstream hair lies on top)"),
        "front_flat": (True, None, "the front roots lie flat: plateau front_mm[0] at the hairline -> front_mm[1] 40 mm in"),
        "front_mm": ([1.5, 5.5], (0, 20), "the front plateau"),
        "stack_packing": (0.0, (0, 0.6), "the layer from the hair's own amount: a first walk counts the hair over each skin "
                                         "place; thickness = hair volume / area / packing (loose hair ~0.1-0.2); 0 = the plateau"),
        "front_release_mm": ([0.0, 0.0], (0, 200), "the front-flat cap fades out between these distances in from the hairline "
                                                   "([0, 0] = never: the cap over the whole front half - before 10-29)"),
        "crown_mm": (10.0, (0, 40), "extra radial lift over the top (volume), none at the front hairline"),
        "density_k": (0.0, (0, 2), "the plateau x (local root density / mean)^k: where more hair grows, more stacks and the "
                                   "hair stands a little higher (the volume from the hair itself, not a shape)"),
    },
    "volume": {
        "packing": (0.0, (0, 0.5), "the hair's own volume (volume.py): strands relaxed until the fibre fills no more than this "
                                   "share of the space (loose dry hair ~0.02-0.05, sleek ~0.1+); 0 = off (the lift's layer only)"),
        "voxel_mm": (2.5, (1, 8), "the density grid"),
        "smooth_vox": (1.0, (0, 4), "the density smoothed this many voxels"),
        "iters": (40, (1, 200), "relaxation passes"),
        "step_mm": (1.0, (0.1, 5), "the most a point moves in one pass"),
        "reach_vox": (4.0, (1, 20), "the step fades inside a uniform mass beyond this many voxels from its edge"),
        "relax": (0.5, (0.05, 1), "share of the clearing step taken per pass"),
        "along_smooth": (1.5, (0, 6), "the move smoothed along the strand (points): it bends, it does not kink"),
        "hold_mm": (4.0, (0, 20), "within this of the skin only the outward part of the move is kept (the comb's tension)"),
        "clear_mm": (0.6, (0, 5), "kept this far out of the body during the passes"),
        "guide_every": (1, (1, 16), "relax every g-th strand only (radius x sqrt g) and carry the moves to the rest (speed)"),
        "cleanup": (0, (0, 20), "grow mode only: volume.py passes that remove only local over-packing left by the "
                                "children's interpolation (DESIGN_v2 4.9); 0 = none"),
    },
    "ears": {
        "back_mm": (4.0, (0, 20), "behind the ears: kept this far behind the ear's back edge at the ear's height"),
        "x_max_mm": (81.0, (60, 120), "at the ear's height no point lateral of this (the ear hides what is behind it)"),
        "x_top_mm": (92.0, (60, 140), "the limit rises to this 40 mm above the ear"),
        "over_mm": (6.0, (0, 30), "lift over the ear's top"),
    },
    "corridor": {
        "phi_deg": (75.0, (0, 180), "no hair over the face skin within this azimuth of the front (above the ears' lower "
                                    "end, > outside_mm outside the hairline)"),
        "outside_mm": (3.0, (0, 30), ""),
        "eyes_only": (False, None, "only keep the hair off the eyes and the face's centre (bob / framing styles)"),
    },
    "gather": {
        "type": ("none", ["none", "ponytail", "bun"], "gather the hair to a point with a tie"),
        "phi_deg": (180.0, (-180, 180), "the gather point on the scalp: azimuth (0 = front, 180 = back)"),
        "theta_deg": (45.0, (0, 150), "polar angle from the top (high ponytail ~40-55, low bun ~105-120)"),
        "tie_r_mm": (12.0, (4, 30), "the tie's inner radius (the bundle's radius: 145k x 0.055 mm hairs ~ 11 mm)"),
        "tie_tube_mm": (1.6, (0.5, 6), "the tie's tube radius"),
        "tie_width_mm": (6.0, (1, 30), "the tie's width along the tail (wraps)"),
        "converge_mm": (30.0, (5, 80), "strands leave the scalp this far from the gather point"),
        "pull": (4.0, (0, 20), "the comb toward the gather point (sleek, pulled)"),
        "tail_dir": ([0.0, 1.0, 0.35], None, "the tail leaves the tie along this (+ the scalp normal), then falls"),
        "tail_spread_mm": (30.0, (5, 80), "the tail's radius after the tie (it fans out) ..."),
        "tail_taper": (0.45, (0, 0.9), "... and narrows this much toward the ends"),
        "tail_stiffness": (0.86, (0, 0.98), "the tail's bending stiffness"),
        "twist_mm": (0.0, (0, 500), "rope twist pitch in the tail (0 = none)"),
        "bun_r_mm": (17.0, (5, 60), "bun: coil radius about the tie axis"),
        "bun_turns": (2.4, (0.5, 6), "bun: turns of the coil"),
        "bun_rise_mm": (9.0, (0, 40), "bun: each turn sits this much further out"),
        "bun_twist_mm": (70.0, (0, 500), "bun: the rope's twist pitch"),
        "bun_tuck_mm": (35.0, (0, 120), "bun: the last part of the rope tucks under, toward the axis"),
        "sleek": (0.85, (0, 1), "on the scalp the hair is pulled smooth: clumping, waves, frizz x (1 - sleek)"),
        "min_tail_mm": (40.0, (0, 300), "a strand too short to reach the tie with this much tail is lengthened to it (the "
                                        "nape hair of a one-length cut barely reaches a high ponytail)"),
        "tendril_frac": (0.0, (0, 1), "loose tendrils: this share of the temples' front roots stays out of the tie"),
        "tendril_zone_mm": (10.0, (0, 40), "... roots within this of the hairline"),
        "tendril_phi_deg": ([25.0, 60.0], None, "... at these azimuths (the temples)"),
        "tendril_len_mm": ([90.0, 150.0], (10, 400), "... this long, hanging beside the face"),
        "tendril_back": (0.3, (0, 3), "... combed back this much (0 = straight down)"),
    },
    "length": {
        "tip_below_menton_mm": (110.0, (-200, 400), "loose styles: tips at the menton minus this (a horizontal cut)"),
        "tip_sd_mm": (12.0, (0, 60), "tip height spread"),
        "short_frac": (0.25, (0, 1), "share of the strands up to short_mm shorter (layers)"),
        "short_mm": ([0.0, 60.0], (0, 300), ""),
        "max_mm": (480.0, (20, 1200), "no strand longer than this"),
        "a_line_mm": (0.0, (-80, 80), "the cut's tilt: tips this much lower at the front than at the back"),
        "gathered_from": ("loose", ["loose", "fixed"], "gathered styles: each strand's length = its loose length "
                                                        "(her hair, gathered) or fixed_mm"),
        "fixed_mm": ([250.0, 330.0], (20, 1200), ""),
    },
    "cut": {
        "type": ("layered", ["blunt", "layered", "face_framing"], "blunt / layered (the crown shorter) / face-framing"),
        "crown_layer": (0.30, (0, 0.8), "layered: the crown roots cut this much shorter (x U(0.7, 1.3))"),
        "crown_theta_deg": ([25.0, 55.0], None, "layered: the crown is the polar angle below the first, fading to the second"),
        "framing_rise_mm": (90.0, (0, 300), "face-framing: the front pieces' tips this much higher at the face ..."),
        "framing_zone_mm": (30.0, (0, 100), "... for roots within this distance of the front hairline"),
        "framing_phi_deg": ([10.0, 85.0], None, "... between these azimuths off the front"),
        "turn_under_mm": (0.0, (0, 30), "bevel: the last 30 mm of each strand curls this much toward the neck"),
        "taken": ("held", ["held", "fall"], "where the scissors read the length (grow mode): held = on the acted growth "
                                            "(over-directed - lengths come out longer where they were pulled from); fall = "
                                            "on the hair grown with every act off (natural fall, the 0 deg perimeter)"),
    },
    "locks": {
        "lock_r_mm": (6.5, (1, 30), "Poisson radius of the lock centres (cells ~1.4x)"),
        "sub_r_mm": (2.4, (0.5, 10), "sub-locks"),
        "bundle_r_mm": (1.6, (0, 5), "bundles inside the sub-locks (0 = none)"),
        "front_scale": (1.0, (0.3, 1.5), "lock radius x this within front_zone_mm of the front hairline"),
        "front_zone_mm": (40.0, (0, 120), ""),
        "root_jitter_deg": (6.0, (0, 30), "each strand leaves its root turned this (sd) in the hair sheet"),
        "bundle_jitter_deg": (10.0, (0, 30), "bundles leave the root together, turned this (sd)"),
        "bundle_jitter_phi_deg": ([180.0, 181.0], None, "... fading out between these azimuths (the sides; 180 = none)"),
    },
    "clump": {
        "lock": ([0.3, 0.9], (0, 1), "lock clumping k per lock"),
        "lock_rise_mm": (15.0, (1, 200), "c(s) = k (1 - exp(-s / rise))"),
        "front_rise_mm": (15.0, (1, 200), "the lock rise within locks.front_zone_mm of the front hairline"),
        "sub": ([0.35, 0.75], (0, 1), "sub-lock clumping"),
        "sub_rise_mm": (12.0, (1, 200), ""),
        "bundle": ([0.3, 0.6], (0, 1), "bundle clumping"),
        "bundle_rise_mm": (6.0, (1, 200), ""),
        "edge_factor": (0.25, (0, 1), "edge hairs clump this much"),
        "temple_free": (True, None, "no clumping at the temples' edge (it pulls the temple hairline up)"),
        "part_free": (True, None, "x0.25 at the front of the part (it opens a scalp line)"),
        "tip_gain": (0.0, (0, 4), "clumping x (1 + tip_gain x s / L): locks gather toward the ends (ribbons), loosest at the root"),
    },
    "waves": {
        "wavelength_mm": (88.0, (0, 400), "flat S-waves (0 = straight)"),
        "wavelength_jitter": (0.15, (0, 0.5), "x U(1 - j, 1 + j) per lock"),
        "amp_mm": ([6.0, 9.0], (0, 60), "side swing per lock"),
        "start_mm": ([80.0, 120.0], (0, 600), "from this far below the root (gathered styles: after the tie)"),
        "tilt_deg": ([25.0, 65.0], (0, 90), "the wave plane turned out of the hair sheet per lock"),
        "perp": (0.15, (0, 1), "a perpendicular part a quarter wave later"),
        "ears_free": (True, None, "no waves at the ears"),
        "phase_field_mm": (0.0, (0, 300), "coherent waves: the lock's phase a smooth field over the scalp at about this scale "
                                         "(neighbouring locks wave together, as a set wave does); 0 = a random phase per lock"),
        "phase_jitter": (0.6, (0, 3.2), "... plus this much per lock (radians)"),
    },
    "frizz": {
        "wander_mm": ([0.3, 0.8], (0, 10), "smooth wander per strand"),
        "frizz_mm": ([0.05, 0.2], (0, 5), "per-point frizz"),
        "fly_frac": (0.02, (0, 0.2), "fly-aways: share of the main strands"),
        "fly_mm": ([4.0, 12.0], (0, 80), "their drift out of the lock"),
        "fly_start": ([0.25, 0.7], (0, 1), "they leave the lock at this fraction of the length"),
    },
    "oil": {
        "amount": (0.3, (0, 1), "hair oil: 0 dry .. 1 wet-look. Clumping x (1 + 0.5 a) (capped 0.95), frizz and "
                                "wander x (1 - 0.6 a), fly-aways x (1 - 0.7 a), roughness x (1 - 0.35 a)"),
        "shine": (0.0, (0, 1), "extra: longitudinal roughness x (1 - 0.3 shine), coat-free"),
    },
    "radius": {
        "root_mm": (0.0275, (0.005, 0.1), "root radius (0.055 mm diameter)"),
        "tip_mm": (0.0075, (0.001, 0.1), "tip radius"),
        "jitter": (0.10, (0, 0.5), "x U(1 - j, 1 + j)"),
        "edge_scale": ([0.4, 0.6], (0.1, 1), "edge hairs"),
    },
    "colour": {
        "model": ("absorption", ["absorption", "melanin"], "absorption: per-point sigma_a = melanin + pheomelanin + a "
                                                           "neutral (ash) absorber; melanin: the BSDF's melanin input"),
        "melanin": (0.62, (0, 0.99), "base melanin (dark blonde ~0.55-0.70)"),
        "redness": (0.06, (0, 1), "pheomelanin share"),
        "redness_sd": (0.02, (0, 0.3), ""),
        "ash": (0.0, (0, 1), "share of the absorption made neutral (grey): cools the colour at the same darkness"),
        "lock_sd": (0.09, (0, 0.3), "melanin sd per lock"),
        "sub_sd": (0.05, (0, 0.3), "per sub-lock"),
        "bundle_sd": (0.06, (0, 0.3), "per bundle"),
        "strand_sd": (0.035, (0, 0.3), "per strand"),
        "root_add": (0.05, (-0.3, 0.5), "roots darker by this over root_mm (not the front edge)"),
        "root_mm": (40.0, (0, 200), ""),
        "lengths_lighter": (0.22, (0, 0.5), "melanin off below the temple level (sun-lightened lengths)"),
        "balayage_frac": (0.33, (0, 1), "share of the locks lighter in the lengths"),
        "balayage_framing_frac": (0.55, (0, 1), "share of the face-framing locks"),
        "balayage_amount": ([0.15, 0.28], (0, 0.6), ""),
        "front_lighter": (0.16, (0, 0.5), "the front edge zone lighter"),
        "edge_lighter": (0.08, (0, 0.5), "edge hairs lighter still"),
    },
    "material": {
        "roughness": (0.26, (0.02, 1), "longitudinal roughness (dry; oil lowers it)"),
        "radial_roughness": (0.55, (0.02, 1), ""),
        "ior": (1.55, (1, 2), ""),
        "coat": (0.0, (0, 1), ""),
        "random_roughness": (0.12, (0, 1), ""),
        "random_color": (0.10, (0, 1), "melanin model only (Cycles ignores it for absorption)"),
    },
}



# grow mode's blocks (DESIGN_v2 section 7), owned by the modules that read them
from .follicle import FOLLICLE_DEFAULTS  # noqa: E402
from .fibre import FIBRE_DEFAULTS  # noqa: E402
from .settle import GROW_DEFAULTS  # noqa: E402
from .children import CHILDREN_DEFAULTS  # noqa: E402
from .actions import ACTS_DEFAULTS  # noqa: E402
from .children import FU_DEFAULTS  # noqa: E402
from .form import LOCK_CAUSE_DEFAULTS, OIL_DEFAULTS  # noqa: E402

S["follicle"] = FOLLICLE_DEFAULTS
S["fibre"] = FIBRE_DEFAULTS
S["grow"] = dict(GROW_DEFAULTS, **{k: v for k, v in CHILDREN_DEFAULTS.items() if k not in GROW_DEFAULTS})
# the follicular units (children.follicular_units) and the short hairs' canopy switch are read by the grow branch from
# the grow block; settle.settings does not read them (generate drops its 'unknown key' for every key named here)
S["grow"].update({k: v for k, v in FU_DEFAULTS.items() if k not in S["grow"]})
S["grow"]["canopy"] = (True, None, "short hairs rest under the long hair lying over them (children.short_hairs with "
                                   "the settled guides' layer): off, they stand at their exit angle through the layer, "
                                   "a halo no real head shows (hair.md: short regrowth lies under the canopy)")
for _k, _v in ACTS_DEFAULTS.items():
    S[_k] = _v
# locks and the wave from their causes (form.lock_causes, DESIGN_v2 4.9): the drying section, sebum days and product;
# off by default (locks.causes false: the legacy Poisson locks and per-lock waves)
S["locks"] = dict(S["locks"], causes=(False, None, "grow mode: locks and the wave from their causes - follicular units, "
                                                   "the drying section's wave phase, liquid bridges from sebum and "
                                                   "product (form.lock_causes) - in place of the Poisson stand-ins"),
                  **{k: v for k, v in LOCK_CAUSE_DEFAULTS.items() if k not in S["locks"]})
S["oil"] = dict(S["oil"], **{k: v for k, v in OIL_DEFAULTS.items() if k not in S["oil"]})

# keys retired in grow mode (DESIGN_v2 section 7): still honoured by "comb", ignored with a warning by "grow"
RETIRED_GROW = ["flow.gravity", "flow.stiffness", "flow.step_mm", "flow.volume", "flow.turn_deg", "flow.temple_sweep",
                "flow.behind_ears", "flow.drape_front", "flow.drape_phi_deg", "flow.comb_strength", "flow.guide_k",
                "lift.*", "volume.packing", "volume.voxel_mm", "volume.smooth_vox", "volume.iters", "volume.step_mm",
                "volume.reach_vox", "volume.relax", "volume.along_smooth", "volume.hold_mm", "volume.clear_mm",
                "volume.guide_every", "ears.x_max_mm", "ears.x_top_mm", "length.short_frac", "length.short_mm",
                "gather.sleek", "gather.tail_spread_mm", "gather.tail_taper", "gather.tail_stiffness",
                "gather.min_tail_mm"]

def _defaults(node):
    if isinstance(node, dict):
        return {k: _defaults(v) for k, v in node.items()}
    return copy.deepcopy(node[0])


DEFAULTS = _defaults(S)


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _find(name, dirs):
    if os.path.isfile(name):
        return name
    for d in dirs:
        for c in (os.path.join(d, name), os.path.join(d, name + ".json")):
            if os.path.isfile(c):
                return c
    raise FileNotFoundError("style %r not found in %s" % (name, dirs))


def load(path, dirs=()):
    """the resolved style: DEFAULTS <- extends chain <- the file; returns (style, warnings)"""
    path = _find(path, list(dirs))
    raw = json.load(open(path))
    chain = [raw]
    seen = {os.path.abspath(path)}
    d0 = [os.path.dirname(os.path.abspath(path))] + list(dirs)
    while chain[-1].get("extends"):
        p = _find(chain[-1]["extends"], d0)
        if os.path.abspath(p) in seen:
            raise ValueError("extends loop at %s" % p)
        seen.add(os.path.abspath(p))
        chain.append(json.load(open(p)))
    st = copy.deepcopy(DEFAULTS)
    for c in reversed(chain):
        st = deep_merge(st, {k: v for k, v in c.items() if k not in ("extends",)})
    st["_file"] = os.path.abspath(path)
    st["_chain"] = [c.get("name", "?") for c in chain]
    return st, validate(st)


def validate(st, node=S, pre=""):
    """unknown keys and out-of-range values (clipped in place); in grow mode also the retired keys a style sets;
    returns warnings"""
    w = retired(st) if node is S else []
    for k, v in list(st.items()):
        if k.startswith("_") or k in ("extends", "notes", "description"):
            continue
        if k not in node:
            w.append("unknown key %s%s" % (pre, k))
            continue
        spec = node[k]
        if isinstance(spec, dict):
            if isinstance(v, dict):
                w += validate(v, spec, pre + k + ".")
            else:
                w.append("%s%s should be an object" % (pre, k))
            continue
        rng = spec[1]
        if isinstance(rng, list) and v not in rng:
            w.append("%s%s = %r not in %s" % (pre, k, v, rng))
        elif isinstance(rng, tuple):
            lo, hi = rng
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if v < lo or v > hi:
                    st[k] = min(max(v, lo), hi)
                    w.append("%s%s = %r clipped to [%s, %s]" % (pre, k, v, lo, hi))
            elif isinstance(v, list):
                c = [min(max(x, lo), hi) for x in v]
                if c != v:
                    st[k] = c
                    w.append("%s%s = %r clipped to [%s, %s]" % (pre, k, v, lo, hi))
    return w



def retired(st):
    """grow mode: the retired keys a style sets away from their defaults (ignored there, with this warning)"""
    if (st.get("flow") or {}).get("source", "comb") != "grow":
        return []
    w = []
    for key in RETIRED_GROW:
        blk, k = key.split(".")
        have, base = st.get(blk) or {}, DEFAULTS.get(blk) or {}
        for kk in (base if k == "*" else [k]):
            if kk in have and have[kk] != base.get(kk):
                w.append("%s.%s = %r is retired in grow mode (ignored)" % (blk, kk, have[kk]))
    return w


def effective(st):
    """the oil coupling applied: returns a copy with the effective clumping, frizz, fly-aways and roughness"""
    e = copy.deepcopy(st)
    a = float(st["oil"]["amount"])
    sh = float(st["oil"]["shine"])
    for k in ("lock", "sub", "bundle"):
        e["clump"][k] = [min(0.95, x * (1 + 0.5 * a)) for x in st["clump"][k]]
    e["frizz"]["wander_mm"] = [x * (1 - 0.6 * a) for x in st["frizz"]["wander_mm"]]
    e["frizz"]["frizz_mm"] = [x * (1 - 0.6 * a) for x in st["frizz"]["frizz_mm"]]
    e["frizz"]["fly_frac"] = st["frizz"]["fly_frac"] * (1 - 0.7 * a)
    e["material"]["roughness"] = st["material"]["roughness"] * (1 - 0.35 * a) * (1 - 0.3 * sh)
    return e


def schema_rows(node=S, pre=""):
    rows = []
    for k, v in node.items():
        if isinstance(v, dict):
            rows += schema_rows(v, pre + k + ".")
        else:
            rows.append((pre + k, v[0], v[1], v[2]))
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "show":
        st, w = load(a[1])
        print(json.dumps(st, indent=1))
        for x in w:
            print("WARNING", x)
    else:
        for r in schema_rows():
            print("| `%s` | %s | %s | %s |" % r)
