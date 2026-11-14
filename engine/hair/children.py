"""Hair Studio children (numpy + scipy, Apache-2.0): the guides that stand for the hairs, and every hair grown back from
them (DESIGN_v2 4.4 guide half, 4.8 child half).

Guide half (4.4):
    guides(X, key, n, rng)            about n guide roots, a Poisson-disk subset of the roots under the part key
    members(key_g, key_c, L_c, s)     every root a member of its nearest guide on its own part side; n_i(s), the
                                      member hairs longer than each arc s (the guide's arrays are n_i x the fibre's)
Child half (4.8):
    interpolate(G, Lg, X_guides, X_c, d0_c, key_c, L_c, key_g=..., d0_g=...)
                                      each hair from the k nearest guides on its own part side, at the same absolute
                                      arc, its own follicle direction blended in at the root, its own cut length
    short_hairs(X, N, d0, L, fibre, body)
                                      hairs shorter than grow.short_mm: each a single elastic rod from its own d0 under
                                      its own weight against the body (no hair-hair term). They stand at their exit
                                      angle until they meet the long hair lying over them (canopy=, the settled
                                      guides): out of that layer a node lifts the layer's weight over its share, so
                                      under a full layer they bend down inside it; the fly-aways are what can lift it
    Canopy, build_canopy, canopy_depth
                                      the long hair's layer from the settled guides: its outer height over the skin
                                      and its areal weight (N/m2), tabled on a lattice, read C1
    part_key(X, side, zone)           the key the guides, members and children share: (X mm, sep x side x zone)

The part key. Hair on the two sides of a part comes from different combing; a guide that took members from both sides
would bend one side's hair over the part. So the key is (X in mm, 60 mm x part side x part zone): within the part zone
the two sides sit 120 mm apart in key space, behind it (zone 0) the hair is one sheet, as form.locks keys the locks.
The neighbour search here is exact rather than relying on the distance alone: a root with side label +1 (the sign of
its key's 4th coordinate) can take only guides labelled +1 or 0, a root labelled -1 only -1 or 0, a root labelled 0 any
guide. Within that set the nearest guides are found by the key distance, as the design writes it.

The shape (4.8): c(s) = X_c + sum_a w_a (G_a(s) - X_a) + (1 - smoothstep(0, root_blend, s)) s (d0_c - sum_a w_a d0_a),
w_a ~ 1 / (d_a^2 + soft^2) with d_a the key distance in mm and soft 0.05 mm, a follicle's radius (the
design's 0.25 mm^2 left 2 % of the weight to the neighbours of a child at a guide's own root). A guide shorter
than s is extended straight along its last direction. Near the root c'(0) = d0_c exactly, so every hair leaves its own
follicle at its own exit angle whatever its guides do.

The route rule (grow.children_route_split). A hair lies with the hair beside it: where a section was split - the
part's two sides, a tucked section beside one left, the hair either side of the ear - neighbouring guides go two ways,
and their blend would lie between them, where no hair went (gr18: the 102 long children at the front of the part, each
blended from guides swept to both sides). So a neighbour whose route parts from the nearest guide's (the chord
directions from the roots to L/4 .. L differ by more than the split) is dropped from the child's blend, and the child
goes its nearest guide's way. The part key already does this where the part zone reaches; the rule reads the routes
themselves, so it holds wherever the acts split the hair.

Hair does not stretch. The blend of k guides' displacements is not an arc-length curve (averaging directions shortens
it, the root term lengthens it), so the blended curve is sampled finely, its true arc length measured, and the hair
ends where that arc reaches its own cut length. The output points are spaced evenly along the hair and the polyline
through them is scaled (two passes) so its own length is the cut length: what is drawn is as long as the hair is.

Short hairs (4.8). A hair under ~2 bending lengths (l_g ~ 39 mm) stands: its own weight bends it only a little. Each is
an inextensible rod of `short_segments` equal segments clamped at its follicle direction d0, minimising bending +
gravity + body contact, the settle's own terms (settle.py; the contact a penalty on the C1 body distance
settle.smooth_distance, h_c 0.5 mm, stiffness contact_k x the node's weight + contact_kb x the rod's own bending
stiffness EI / l^3 at the node: the skin holds a short stiff hair against its bending as well as its weight), with the
fibre's own EI and weight (taper included: a new hair is thinner at the root too). Each rod starts built out from its follicle and pushed out of the body node by
node. The energy has no hair-hair term, so it is a sum over hairs: each hair is solved by its own L-BFGS (own history,
own step, own stop), all of them vectorised in one pass, in Jacobi-scaled variables (DESIGN_v2 4.5's diagonal: bending +
tension). The skin is far stiffer than the hair, so where a hair touches, the contact springs couple every segment
upstream of the node (moving any of them moves the node): a diagonal scaling cannot see that, and L-BFGS crawled on
the hairs pressed into the skin. So each moving hair's L-BFGS starts every step from a dense preconditioner, its own
3m x 3m Gauss-Newton matrix of bending (the joint chain), tension (4.5) and the springs of the nodes it touches,
refreshed every short_refresh iterations (the settle's banded preconditioner does the same with its contact springs).
The contact clearance ramps up from the root over the arc the follicle's own exit takes to rise: a hair leaving the
skin is not lying on it; it is raised by the C2 onset where the follicle's straight exit allows (never above it).

Units: positions m, arcs m, every *_mm argument and key mm, angles deg.
"""
import time

import numpy as np
from scipy.spatial import cKDTree

from . import form as FM

# grow.* keys this module reads, in style.py's (default, (lo, hi) | [choices] | None, note) format. guides, children_k,
# root_blend_mm and short_mm are the same keys and values as settle.GROW_DEFAULTS; the rest are new.
CHILDREN_DEFAULTS = {
    "guides": (3000, (50, 20000), "guide rods standing for the hairs (DESIGN_v2 4.4); about 48 hairs each at 145k"),
    "children_k": (4, (1, 16), "children: nearest guides on the same part side (DESIGN_v2 4.8)"),
    "root_blend_mm": (15.0, (0.0, 100.0), "children: their own follicle direction blends in over this arc (4.8)"),
    "short_mm": (80.0, (0.0, 400.0), "children: hairs shorter than this are single-rod sweeps (4.8); ~2 l_g, the "
                                     "length below which a hair stands under its own weight"),
    "children_soft_mm": (0.05, (0.005, 5.0), "children: the weights' softening, w = 1 / (d^2 + soft^2) with d the "
                                             "key distance in mm; about a follicle's radius. It only keeps the weight "
                                             "finite at a guide's own root, where the child IS that guide's hair: "
                                             "there the guide takes 1 / (1 + soft^2 sum 1/d_other^2) of the weight, "
                                             "99.97 % at 5 mm spacing (the design's 0.25 mm^2 gave 97.9 %, and the "
                                             "2 % left to neighbours moved such a child up to 4.7 mm off its guide)"),
    "part_sep_mm": (60.0, (10.0, 1000.0), "the part key's 4th coordinate, x part side x part zone (as form.locks)"),
    "children_points": (32, (4, 512), "points per child, evenly spaced along the hair"),
    "children_batch": (20000, (1000, 200000), "children per batch (DESIGN_v2 4.8: 20k); bounds the RAM"),
    "children_mode": ("blend", ["blend", "ringlet"], "children: 'blend' - each hair from its k nearest guides "
                                                     "(4.8); 'ringlet' - each hair a member of ONE guide's lock, "
                                                     "gathered off its follicle into a tube of ringlet_r_mm round the "
                                                     "guide and carried round with the guide's coil. A curly head is "
                                                     "ringlets, not 100k separate coils: the hairs of a ringlet coil "
                                                     "in step. A blend of 4 guide coils at unrelated phases cancels "
                                                     "the coil (types_curly2, sheet 66: guides' curl index 2.19, "
                                                     "children's 1.68) and read as frizz"),
    "ringlet_r_mm": (1.2, (0.05, 10.0), "children_mode 'ringlet': the lock's radius (a member's offset from the "
                                        "guide, uniform over the disc); ~80-300 fibres of 65 um at a loose packing of "
                                        "0.05-0.2"),
    "ringlet_gather_mm": (15.0, (1.0, 100.0), "children_mode 'ringlet': the arc over which a member leaves its own "
                                              "follicle and joins its lock's tube"),
    "ringlet_twist_mm": (40.0, (0.0, 1000.0), "children_mode 'ringlet': the arc over which a member winds once "
                                              "round its lock (0: no twist). The hairs of a ringlet twist round each "
                                              "other; held at one place in the carried frame a member on the inside of "
                                              "a flat loop stays inside and comes out up to 37 % short (types_curly2 "
                                              "guides, 1023 of 9.7k members > 2 % short)"),
    "ringlet_merge_mm": (0.0, (0.0, 50.0), "children_mode 'ringlet': ringlets whose guides lie together merge into one "
                                           "lock - two neighbouring guides whose paths past 40 mm stay within this "
                                           "mean distance (about a coil diameter: their coils touch and thread into "
                                           "each other) share the leader's path, so the head carries ringlets of "
                                           "different sizes. 0: off. On types_curly3's guides 10 mm merges 23 % "
                                           "(the nearest neighbour's mean path distance p10 / 50 8.1 / 13.4 mm)"),
    "ringlet_merge_max": (3, (2, 20), "children_mode 'ringlet': the most guides one merged lock may take"),
    "ringlet_r_sd": (0.0, (0.0, 1.5), "children_mode 'ringlet': the lock radius's lognormal spread per lock. With it "
                                      "(or merging) on, each lock's radius also scales with the square root of its "
                                      "hair count about the median lock (one packing): a merged lock is thicker. 0 and "
                                      "no merging: every lock ringlet_r_mm (types_curly3, sheet 67: every ringlet "
                                      "the same - a wig's evenness)"),
    "children_fine": (160, (16, 2000), "samples per child of the blended curve before it is cut at its length; "
                                       "spaced as (j / n)^1.5, ~0.03 mm at the root and ~6 mm at 400 mm"),
    "children_route_split": (0.5, (0.05, 2.0), "children: a hair follows its own neighbours, never a blend of two "
                                               "routes. A neighbour guide is dropped from a child's blend when its "
                                               "route parts from the nearest guide's: the chord directions from the "
                                               "roots to arcs L/4, L/2, 3L/4, L (L the child's own length) differ "
                                               "by more than this (|D_a(s) - D_0(s)| / s; 0.5 ~ 29 deg). Where a "
                                               "section was split (the part's two sides, a tucked section against "
                                               "the one left, the hair either side of the ear) neighbouring guides "
                                               "go two ways and their blend lies between, where no hair went (gr18: "
                                               "the 102 long children at the front of the part, each blended from "
                                               "guides swept to both sides, chord spread p10/50/90 0.93 / 1.06 / "
                                               "1.22; all long children p50 / p90 0.19 / 0.45). 2.0 keeps every "
                                               "neighbour (the plain blend)"),
    "short_segments": (16, (4, 128), "segments per short hair (equal length; 80 mm -> 5 mm, l_g is 39 mm)"),
    "short_evals": (600, (10, 20000), "iteration cap of the short hairs' per-hair L-BFGS (each hair stops on its "
                                      "own when balanced)"),
    "short_tol": (1e-3, (1e-8, 0.1), "a short hair is at rest when no segment carries an unbalanced moment over "
                                     "this x (the weight it carries x its length, floored by short_kink_urad)"),
    "contact_mm": (0.5, (0.0, 5.0), "short hairs rest this far off the skin (the settle's h_c)"),
    "contact_k": (3e5, (1e3, 1e9), "short hairs' contact stiffness per metre of penetration x the node's weight "
                                   "(the settle's value)"),
    "contact_onset_mm": (0.05, (0.0, 1.0), "short hairs: the contact stiffness rises from 0 to full over this much "
                                           "penetration of the clearance (C2: energy k p^3 / 6 delta, then k (p^2 / 2 "
                                           "- delta p / 2 + delta^2 / 6)), so the force's slope has no jump for the "
                                           "solver to stall on; the clearance a node keeps is raised by it, so a node "
                                           "at rest is still outside the body"),
    "short_kink_urad": (1.0, (0.0, 1000.0), "short hairs: the balance test's floor, the moment of a kink this many "
                                            "microradians at the joint (EI / l x angle). A short hair is at rest when "
                                            "no segment's unbalanced moment exceeds short_tol x the larger of (the "
                                            "weight it carries x its length) and this: a 0.04 mm new hair weighs "
                                            "1e-8 of its own bending moment at a microradian, so the weight alone "
                                            "asked its joints to balance to 1e-14 rad, under the float's resolution"),
    "short_refresh": (10, (0, 1000), "short hairs: iterations between refreshes of each moving hair's dense "
                                     "preconditioner (bending + tension + the contact springs it touches); 0 = the "
                                     "plain Jacobi-scaled L-BFGS"),
    "contact_kb": (1e3, (0.0, 1e6), "short hairs' contact stiffness x the rod's own bending stiffness at the node, "
                                    "EI / l^3. The skin is far stiffer than a hair, so it must hold a node against the "
                                    "hair's bending as firmly as against its weight: a 20 mm hair by the ear, clamped "
                                    "at a follicle aimed along the skin, presses with ~25 000 x its node weight, and "
                                    "the weight-only stiffness let it 1.4 mm into the body (measured). At 1e3 the "
                                    "penetration is ~1/1000 of the bending deflection, about a micrometre"),
    "canopy_phi": (0.03, (0.005, 0.3), "short hairs under the long hair: the long hair's fibre fraction in its layer "
                                       "(hair.md: hanging hair 0.02-0.04). The outermost guide's own hairs lie half "
                                       "their layer's thickness, (fibre area per width) / phi / 2, out from it"),
    "canopy_onset_mm": (0.5, (0.01, 5.0), "short hairs under the long hair: the layer's outer surface is fuzzy by "
                                          "about one fibre spacing, r sqrt(pi / phi) ~ 0.35 mm at phi 0.03, so the "
                                          "layer's load on a node rises over this much (C2, as the body's onset)"),
    "canopy_grid_mm": (3.0, (1.0, 10.0), "short hairs under the long hair: the lattice the layer's height and load "
                                         "are tabled on (the body grid's spacing)"),
    "canopy_height": ("mass", ["mass", "max"], "short hairs under the long hair: the layer's outer surface over a skin "
                                               "point. 'mass': where the long hair's fibre is - the fibre-weighted mean "
                                               "of the guide segments' outer sides within R (the children are that same "
                                               "weighted blend of their guides), never under the stack the fibre makes "
                                               "lying on the skin at canopy_phi; 'max': the highest guide segment within "
                                               "R (round 9). On the test model (gr11) one guide standing 12-18 mm off the temple "
                                               "set 'max' to 16 mm over 500-600 short roots while the long hair itself "
                                               "lay within ~8 mm: the short hairs stood free up to 16 mm, through the "
                                               "long hair - the wispy temples"),
    "short_set": (True, None, "the short hairs are brushed and set with their section: where the section was held "
                              "while it dried (dry.hold, span 'held'), each short hair takes the held path of the "
                              "guides around it (the brush drawn through carries every hair in its bristles, short or "
                              "long) and keeps dry.set x retention of that turning (the fixative's where it went); "
                              "released from that shape, it rests under its set, weight, skin and the layer over it. "
                              "Off: the short hair rests from its follicle alone"),
    "short_laid": (False, None, "with short_set: where the section was tucked (actions' tuck: the hand carries the side "
                               "section over the top of the ear and lays it behind it), the short hairs in it are held "
                               "on that laid route - the tucked guides' laid path (acts.Qp), in place, not offset by "
                               "the root's distance from the guides - so a short hair rooted beside or below the guides "
                               "is gathered up onto the route over the ear, never laid on the helix. Off: every short "
                               "hair takes the guides' held path, offset to its root (round 9). Off by default: on the test model (gr10's guides, round 10) the held paths over the ear's top 12 mm fell 940 -> 27 (404 with the reach rule), but released with dry.set x retention ~0.54 of the turn kept the short hairs sprang back onto the helix - the short / baby hairs' seen cover of the ear 0.070 / 0.092 (round 9) -> 0.114 / 0.129"),
    "short_friction": (True, None, "with short_set: the long hair lying over a set short hair holds it where the "
                                   "brush laid it. Pressed into the layer by the layer's weight over its share (the "
                                   "canopy's areal weight x the node's strip) and by the oil's liquid bridges to the long "
                                   "hairs around it (fibre.adhesive_load at canopy_phi), each node of the held shape is a "
                                   "static friction anchor, mu N in the skin's tangent plane (the settle's smoothed "
                                   "Coulomb law, lagged: N read once at the held shape). A node out of the layer (the "
                                   "forehead, where no long hair lies) has no load and no anchor. Off: the short hair "
                                   "springs back from the held shape against nothing but its own set (round 9) - on "
                                   "the test model (gr12) ~100-200 hairline short hairs sprang forward onto the forehead under "
                                   "the layer's weight, tip springback p50 / p90 19.5 / 45.5 mm"),
    "short_friction_eps_mm": (1.0, (0.01, 10.0), "short_friction: the smoothed Coulomb law is linear within this "
                                                 "slip, mu N beyond (settle.friction_eps_mm)"),
    "short_friction_load": ("contacts", ["contacts", "strip"], "short_friction: how much of the layer's weight over a "
                            "short hair's strip presses that hair. 'contacts' (round 13): the weight reaches the skin "
                            "through the fibres lying on it, so the short hair takes its own share of the contacts in "
                            "its strip - the long hairs of the layer's bottom row beside it carry theirs. A layer at "
                            "fibre fraction phi has 2 phi / (pi r) fibres per metre of width lying within one diameter "
                            "of the skin, so a strip as wide as the short hairs' spacing sp holds 2 phi sp / (pi r) long "
                            "ones and the short hair takes 1 / (1 + that) of its weight; for sparse short hairs that is "
                            "the weight over one bottom-row spacing, pi r / (2 phi) ~ 1.4 mm at phi 0.03, r 27 um. "
                            "'strip' (round 12): the whole strip's weight on the short hair - on the test model (gr12) "
                            "N 1196 / 5778 x the node's own weight p50 / p90, and the hold pushed 38 nodes into the "
                            "body (24 without it)"),
    "short_friction_stiff": (True, None, "short_friction: add mu N / contact onset to a held node's skin stiffness, "
                                         "so the skin holds the node as firmly as the friction pulls it (where the "
                                         "anchor's tangent plane is not the skin's under the node, part of the pull "
                                         "points into the skin). On the test model (gr13's guides) the contact share alone "
                                         "left 37 nodes inside the body (18 beyond 2 mm), against 40 (20) for the "
                                         "strip and 26 (6) with no hold"),
    "short_laid_share": (0.5, (0.0, 1.0), "short_laid: a short hair is in the tucked section when the tucked guides "
                                          "carry at least this share of its interpolation weight (the hand gathers a "
                                          "hair with the most of the hair around it)"),
}
CHILDREN = {k: v[0] for k, v in CHILDREN_DEFAULTS.items()}

_Z = np.array([0.0, 0.0, 1.0])
_ROUTE_FRAC = np.array([0.25, 0.5, 0.75, 1.0])      # the arcs (x the child's length) the route rule reads


def settings(cfg=None):
    """plain values from a whole style (its 'grow' block) or the block itself; missing keys take the defaults, keys of
    the grow block this module does not read are left alone (settle.py reads them)"""
    out = dict(CHILDREN)
    if cfg:
        blk = cfg.get("grow", cfg) if isinstance(cfg, dict) else {}
        for k in CHILDREN:
            if k in blk:
                v = blk[k]
                out[k] = v[0] if isinstance(v, tuple) else v
    return out


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / max(b - a, 1e-300), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------- the part key and the side sets
def part_key(X, side, zone=None, sep_mm=60.0):
    """(n, 4) mm: X in mm and sep x side x zone. side: +1 / -1 per root (the side of the part it is combed to; 0 for
    no part); zone: 1 inside the part zone, 0 behind it (default 1 everywhere)"""
    X = np.asarray(X, np.float64)
    s = np.asarray(side, np.float64).ravel()
    z = np.ones(len(X)) if zone is None else np.asarray(zone, np.float64).ravel()
    return np.c_[X * 1000.0, sep_mm * s * z]


def side_label(key):
    """-1 / 0 / +1: the sign of the key's 4th coordinate (0 behind the part zone or with no part)"""
    key = np.asarray(key, np.float64)
    if key.shape[1] < 4:
        return np.zeros(len(key), np.int8)
    k4 = key[:, 3]
    return np.where(k4 > 1e-9, 1, np.where(k4 < -1e-9, -1, 0)).astype(np.int8)


class SideTrees:
    """nearest guides that a root may take: its own side label or 0 (never the other side of the part), by the key
    distance. One KD-tree per label set."""

    def __init__(self, key_g):
        self.key_g = np.asarray(key_g, np.float64)
        self.lab = side_label(self.key_g)
        self.sets = {}
        for c in (-1, 0, 1):
            idx = np.nonzero(self.lab * c >= 0)[0] if c != 0 else np.arange(len(self.key_g))
            self.sets[c] = (cKDTree(self.key_g[idx]) if len(idx) else None, idx)

    def query(self, key_c, k):
        """(d (n, k) mm, gi (n, k) guide index); with fewer allowed guides than k the missing columns have d = inf and
        gi = -1"""
        key_c = np.asarray(key_c, np.float64)
        lab = side_label(key_c)
        n = len(key_c)
        d = np.full((n, k), np.inf)
        gi = np.full((n, k), -1, np.int64)
        for c in (-1, 0, 1):
            rows = np.nonzero(lab == c)[0]
            tree, idx = self.sets[c]
            if len(rows) == 0 or tree is None:
                continue
            kk = min(k, len(idx))
            dd, ii = tree.query(key_c[rows], k=kk, workers=-1)
            dd, ii = dd.reshape(len(rows), kk), ii.reshape(len(rows), kk)
            d[rows, :kk] = dd
            gi[rows, :kk] = idx[ii]
        return d, gi


# ---------------------------------------------------------------- guide half (DESIGN_v2 4.4)
def guides(X, key, n, rng, cand_mult=10, log=None):
    """about n guide roots: a Poisson-disk subset (form.poisson) of a random 10 n candidates, under the part key, so no
    guide spans the part. The disk radius comes from the roots' own spacing (a uniform scatter of density lambda has a
    mean nearest-neighbour distance 1 / (2 sqrt lambda)); the pass keeps ~0.58 / r^2 per area (measured, test_settle)
    and is redone once with r scaled by sqrt(count / n) if the count is off by more than 5 %.
    X (m, 3) roots m, key (m, 4) mm. Returns (gi sorted indices into X, report)."""
    X = np.asarray(X, np.float64)
    key = np.asarray(key, np.float64)
    m = len(X)
    cand = np.sort(rng.choice(m, size=min(m, int(cand_mult * n)), replace=False))
    sub = rng.choice(m, size=min(m, 20000), replace=False)
    dnn = cKDTree(X * 1000.0).query(X[sub] * 1000.0, k=2)[0][:, 1]
    area = m * 4.0 * float(np.mean(dnn)) ** 2                     # mm^2 of scalp the roots cover
    r = np.sqrt(0.58 * area / n)
    gi = FM.poisson(key, cand, r, rng)
    first = len(gi)
    if abs(len(gi) - n) > 0.05 * n:
        r *= np.sqrt(len(gi) / n)
        gi = FM.poisson(key, cand, r, rng)
    gi = np.sort(gi)
    rep = dict(guides=int(len(gi)), asked=int(n), r_mm=float(r), first_pass=int(first), area_cm2=area / 100.0,
               roots_per_guide=m / max(len(gi), 1), by_side={int(c): int((side_label(key[gi]) == c).sum())
                                                             for c in (-1, 0, 1)})
    if log:
        log("guides: %d (asked %d), Poisson r %.2f mm, %.0f roots each" % (len(gi), n, r, rep["roots_per_guide"]))
    return gi, rep


def members(key_g, key_c, L_c=None, s=None, trees=None):
    """every root a member of its nearest guide on its own part side (SideTrees).
    Returns (mg (m,) guide index per root, counts (G,), n) where n (G, K) = n_i(s), the member hairs longer than each
    arc s (G, K) m (None without L_c and s). The guide's stiffness, weight and fibre volume at s are n_i(s) x the
    fibre's (settle.Rods integrates them exactly per member with the taper; this count is the design's n_i)."""
    trees = trees or SideTrees(key_g)
    G = len(trees.key_g)
    _, gi = trees.query(key_c, 1)
    mg = gi[:, 0]
    counts = np.bincount(mg, minlength=G)
    n = None
    if L_c is not None and s is not None:
        L_c = np.asarray(L_c, np.float64).ravel()
        s = np.asarray(s, np.float64)
        big = 2.0 * max(float(L_c.max()), float(s.max())) + 1.0
        kL = np.sort(mg * big + L_c)
        end = np.cumsum(counts)
        q = np.arange(G)[:, None] * big + s
        n = end[:, None] - np.searchsorted(kL, q.ravel(), side="right").reshape(s.shape)
    return mg, counts, n


# ---------------------------------------------------------------- child half (DESIGN_v2 4.8)
def _guide_table(G, Lg, X_guides, arcs, h, S):
    """the guides' displacements from their roots, G_a(s) - X_a, on a uniform arc table s = 0, h, 2h, .. >= S (float32,
    (Gn, M, 3)), each guide extended straight along its last direction past its length Lg"""
    G = np.asarray(G, np.float64)
    Gn, K = G.shape[:2]
    if arcs is None:
        arcs = np.concatenate([np.zeros((Gn, 1)), np.cumsum(np.linalg.norm(np.diff(G, axis=1), axis=2), 1)], 1)
    arcs = np.asarray(arcs, np.float64)
    Lg = np.minimum(np.asarray(Lg, np.float64).ravel(), arcs[:, -1])
    M = int(np.ceil(S / h)) + 2
    sg = np.arange(M) * h
    big = float(arcs.max()) + S + 1.0
    flat = (arcs + np.arange(Gn)[:, None] * big).ravel()
    T = np.empty((Gn, M, 3), np.float32)
    # the last direction: the last segment that ends at or before Lg (inactive nodes past Lg are ignored)
    iL = np.clip((arcs <= Lg[:, None] + 1e-12).sum(1) - 1, 1, K - 1)
    rows = np.arange(Gn)
    dL = _unit(G[rows, iL] - G[rows, iL - 1])
    PL = G[rows, iL]
    sL = arcs[rows, iL]
    for g0 in range(0, Gn, 512):
        g1 = min(Gn, g0 + 512)
        rr = np.arange(g0, g1)
        q = np.minimum(sg[None, :], sL[rr, None])
        j = np.searchsorted(flat, (q + rr[:, None] * big).ravel(), side="right").reshape(len(rr), M) - 1 \
            - rr[:, None] * K
        j = np.clip(j, 0, K - 2)
        a0 = arcs[rr[:, None], j]
        a1 = arcs[rr[:, None], j + 1]
        f = np.clip((q - a0) / np.maximum(a1 - a0, 1e-300), 0.0, 1.0)[..., None]
        P = G[rr[:, None], j] * (1 - f) + G[rr[:, None], j + 1] * f
        ext = np.maximum(sg[None, :] - sL[rr, None], 0.0)[..., None]
        P = np.where(ext > 0, PL[rr, None] + ext * dL[rr, None], P)
        T[g0:g1] = (P - np.asarray(X_guides, np.float64)[rr, None]).astype(np.float32)
    return T


def _frame_table(T):
    """the carried frame (N1, N2) along each guide's arc table T (Gn, M, 3): N1 transported down the guide by projecting
    off each new tangent (rotation-minimising to O(h)), N2 = t x N1. A member held at a fixed (N1, N2) offset turns
    with the guide's coil, so the lock's hairs coil in step"""
    Td = np.asarray(T, np.float64)
    t = _unit(np.diff(Td, axis=1))
    t = np.concatenate([t, t[:, -1:]], 1)
    Gn, M = t.shape[:2]
    # past a guide's end its nodes repeat (settle.Rods.positions) and the table stands still: carry the last tangent
    # on, or N2 = t x N1 is zero there and a member ending at its lock's end kinks onto the guide (review 10-31)
    nz = np.linalg.norm(t, axis=2) > 0
    t = t[np.arange(Gn)[:, None], np.maximum.accumulate(np.where(nz, np.arange(M)[None, :], 0), axis=1)]
    ref = np.where(np.abs(t[:, 0, 2:3]) < 0.9, _Z[None, :], np.array([[1.0, 0.0, 0.0]]))
    n1 = _unit(np.cross(t[:, 0], ref))
    N1 = np.empty((Gn, M, 3), np.float32)
    N1[:, 0] = n1
    for j in range(1, M):
        n1 = _unit(n1 - (n1 * t[:, j]).sum(1, keepdims=True) * t[:, j])
        N1[:, j] = n1
    N2 = np.cross(t, N1.astype(np.float64)).astype(np.float32)
    return N1, N2


def _ringlet_merge(T, h, Lg, Xg, trees, key_g, merge_m, max_n, s_lo=0.04):
    """the locks after merging (children_mode 'ringlet', ringlet_merge_mm): each guide against its 4 nearest allowed
    neighbours (SideTrees: never across the part), the mean distance between the two settled paths over the arcs past
    s_lo that both reach (21 samples); pairs under merge_m join, closest first, while a lock has at most max_n guides
    (union-find). Returns (leader (Gn,) the guide whose path each guide's members follow, report)"""
    Gn = len(T)
    Xg = np.asarray(Xg, np.float64)
    _, nb = trees.query(key_g, min(5, Gn))
    S = np.linspace(0.0, 1.0, 21)
    allg = np.arange(Gn)
    pairs = []
    for a in range(1, nb.shape[1]):
        j = nb[:, a]
        ok = j >= 0
        jj = np.where(ok, j, allg)
        Lm = np.minimum(Lg, Lg[jj])
        ok &= Lm > s_lo + 0.02
        sm = s_lo + np.maximum(Lm - s_lo, 0.0)[:, None] * S[None, :]
        Pa = Xg[:, None] + _lookup(T, h, allg, sm)
        Pb = Xg[jj][:, None] + _lookup(T, h, jj, sm)
        dist = np.linalg.norm(Pa - Pb, axis=2).mean(1)
        sel = np.nonzero(ok & (dist < merge_m) & (jj != allg))[0]
        pairs += list(zip(dist[sel].tolist(), sel.tolist(), jj[sel].tolist()))
    parent = np.arange(Gn)
    size = np.ones(Gn, np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    joined = 0
    for _, a, b in sorted(pairs):
        ra, rb_ = find(a), find(b)
        if Lg[rb_] > Lg[ra]:            # the longest guide leads: its members end with it, never cut to a shorter path
            ra, rb_ = rb_, ra
        if ra != rb_ and size[ra] + size[rb_] <= max_n:
            parent[rb_] = ra
            size[ra] += size[rb_]
            joined += 1
    leader = np.array([find(i) for i in range(Gn)], np.int64)
    sz = np.bincount(leader, minlength=Gn)
    return leader, dict(pairs_under=len(pairs) // 2, joins=joined, guides_merged=int((leader != allg).sum()),
                        locks_by_guides={int(k): int((sz == k).sum()) for k in range(1, max_n + 1)})


def _lookup(T, h, gi, s):
    """G_a(s) - X_a for guide gi (b,) at arcs s (b, M): linear in the uniform table"""
    u = s / h
    i0 = np.clip(np.floor(u).astype(np.int64), 0, T.shape[1] - 2)
    f = (u - i0)[..., None]
    return T[gi[:, None], i0] * (1 - f) + T[gi[:, None], i0 + 1] * f


def _resample(C, sig, tau):
    """points at arcs tau (b, n) along polylines C (b, M, 3) with cumulative arcs sig (b, M), row by row"""
    b, M = sig.shape
    big = float(sig[:, -1].max()) + float(tau.max()) + 1.0
    off = np.arange(b)[:, None] * big
    j = np.searchsorted((sig + off).ravel(), (tau + off).ravel(), side="right").reshape(tau.shape) - 1 \
        - np.arange(b)[:, None] * M
    j = np.clip(j, 0, M - 2)
    r = np.arange(b)[:, None]
    s0, s1 = sig[r, j], sig[r, j + 1]
    f = np.clip((tau - s0) / np.maximum(s1 - s0, 1e-300), 0.0, 1.0)[..., None]
    return C[r, j] * (1 - f) + C[r, j + 1] * f


def interpolate(G, Lg, X_guides, X_c, d0_c, key_c, L_c, key_g=None, d0_g=None, arcs=None, cfg=None, probe_mm=None,
                log=None, sec_c=None, sec_g=None):
    """children from the guides (DESIGN_v2 4.8).

    G (Gn, K, 3) m: the settled guides' nodes, G[:, 0] the roots; Lg (Gn,) m their lengths (nodes past Lg are ignored);
    X_guides (Gn, 3) m; X_c (n, 3) m the children's roots; d0_c (n, 3) their follicle directions (unit); key_c (n, 4) mm
    and key_g (Gn, 4) mm the part keys (part_key); L_c (n,) m each child's own cut length; d0_g (Gn, 3) the guides'
    follicle directions (default: their first segment's direction); arcs (Gn, K) m the guides' node arcs (default: the
    polyline's own); cfg the grow block or a style.
    probe_mm: optional arcs (mm) at which c(s) is also returned unchanged by the length step (for the root checks).
    sec_c (n,) and sec_g (Gn,): the drying section of each child and guide (form.section_phase), when the guides carry
    the natural wave (settle.rest_waves). A hair dried in a section nests into that section's wave train, so a child
    takes its blend from the guides of its own section among its k neighbours; a blend across two sections averages
    two trains at unrelated phases (two at random phases keep 2 / pi of the wave on average) and the child hangs
    straighter than any hair round it. A child with no neighbour of its own section keeps the plain blend.

    Returns (P (n, points, 3) float32 m, report) and, with probe_mm, a third value (n, len(probe), 3) float64 m."""
    t0 = time.time()
    c = settings(cfg)
    k = int(c["children_k"])
    soft2 = float(c["children_soft_mm"]) ** 2
    rb = float(c["root_blend_mm"]) / 1000.0
    npts = int(c["children_points"])
    nb = int(c["children_batch"])
    Mf = int(c["children_fine"])
    split = float(c.get("children_route_split", 2.0))
    X_c = np.asarray(X_c, np.float64)
    d0_c = _unit(np.asarray(d0_c, np.float64))
    L_c = np.asarray(L_c, np.float64).ravel()
    n = len(X_c)
    G = np.asarray(G, np.float64)
    X_guides = np.asarray(X_guides, np.float64)
    if key_g is None:
        key_g = part_key(X_guides, np.zeros(len(X_guides)))
    if d0_g is None:
        d0_g = _unit(G[:, 1] - G[:, 0])
    d0_g = _unit(np.asarray(d0_g, np.float64))
    trees = SideTrees(key_g)
    lab_g = trees.lab
    ext_f = 1.6                                   # the blended curve is sampled to 1.6 x the cut length, then cut
    S = float(L_c.max()) * ext_f + 0.01 if n else 0.01
    if str(c.get("children_mode", "blend")) == "ringlet" and float(c["ringlet_merge_mm"]) > 0:
        # the merge reads the guides' settled paths to their ends: the table reaches them whatever these children's
        # lengths (generate's held call for the short hairs would otherwise merge on straight extensions)
        S = max(S, float(np.max(np.asarray(Lg, np.float64))) + 0.01)
    h = 0.0005                                    # 0.5 mm table: under 0.02 mm from the guide's own polyline
    T = _guide_table(G, Lg, X_guides, arcs, h, S)
    ring = str(c.get("children_mode", "blend")) == "ringlet"
    if ring:
        # children_mode 'ringlet': each hair joins ONE guide's lock (its nearest allowed guide after the side, route and
        # section rules), gathers off its follicle into the lock's tube and turns with the guide's coil in its carried
        # frame; it ends at its own cut or at the guide's end, never on a straight extension
        F1, F2 = _frame_table(T)
        R_r = float(c["ringlet_r_mm"]) / 1000.0
        gath = float(c["ringlet_gather_mm"]) / 1000.0
        tw = float(c["ringlet_twist_mm"]) / 1000.0
        Lg_r = np.asarray(Lg, np.float64).ravel()
        rng_r = np.random.default_rng(7919)
        rr_u = np.sqrt(rng_r.random(n))                    # the member's place in its lock's disc (x the lock radius)
        th_all = 2.0 * np.pi * rng_r.random(n)
        ring_g = np.empty(n, np.int64)
        Gn = len(G)
        leader = np.arange(Gn)
        mrep = None
        mrg = float(c["ringlet_merge_mm"]) / 1000.0
        if mrg > 0:
            leader, mrep = _ringlet_merge(T, h, Lg_r, X_guides, trees, key_g, mrg, int(c["ringlet_merge_max"]))
        rsd = float(c["ringlet_r_sd"])
        R_lock = np.full(Gn, R_r)
        if rsd > 0 or mrg > 0:
            # one packing: the lock's radius goes with the square root of its hairs about the median lock, times a
            # lognormal spread; the counts from the nearest guide (the side rule, not the route / section rules)
            g_near = trees.query(key_c, 1)[1][:, 0]
            cnt0 = np.bincount(leader[np.maximum(g_near, 0)], minlength=Gn).astype(np.float64)
            med = float(np.median(cnt0[cnt0 > 0])) if (cnt0 > 0).any() else 1.0
            fac = np.exp(np.random.default_rng(7927).normal(0.0, rsd, Gn)) if rsd > 0 else np.ones(Gn)
            R_lock = R_r * fac * np.sqrt(np.maximum(cnt0, 1.0) / med)
    t_tab = time.time() - t0
    P = np.empty((n, npts, 3), np.float32)
    probe = None if probe_mm is None else np.empty((n, len(probe_mm), 3))
    viol = 0
    wnear = np.empty(n)
    short_ext = 0
    route_cut = 0
    route_kids = np.zeros(n, bool)
    sec_kids = 0
    len_err = np.empty(n)
    u = (np.arange(Mf) / (Mf - 1.0)) ** 1.5
    for b0 in range(0, n, nb):
        b1 = min(n, b0 + nb)
        b = b1 - b0
        d, gi = trees.query(key_c[b0:b1], k)
        ok = gi >= 0
        lab_c = side_label(key_c[b0:b1])
        viol += int(((lab_g[np.where(ok, gi, 0)] * lab_c[:, None] < 0) & ok).sum())
        w = np.where(ok, 1.0 / (d * d + soft2), 0.0)
        gi = np.where(ok, gi, gi[:, :1])
        if split < 2.0 and k > 1:
            # the route rule (children_route_split): a hair lies with the hair beside it, so where the nearest guide
            # and another neighbour went two ways (a section split - the part, the tuck, the ear) the child goes the
            # nearest guide's way instead of between them. The spread is read at the child's own arcs.
            sj = L_c[b0:b1, None] * _ROUTE_FRAC[None, :]
            D0 = _lookup(T, h, gi[:, 0], sj)
            for a in range(1, k):
                dev = (np.linalg.norm(_lookup(T, h, gi[:, a], sj) - D0, axis=2) / np.maximum(sj, 1e-9)).max(1)
                cut = ok[:, a] & (dev > split)
                w[:, a] = np.where(cut, 0.0, w[:, a])
                route_cut += int(cut.sum())
                route_kids[b0:b1] |= cut
        if sec_c is not None and sec_g is not None and k > 1:
            same = ok & (np.asarray(sec_g)[gi] == np.asarray(sec_c)[b0:b1, None]) & (w > 0)
            has = same.any(1)
            w = np.where(has[:, None], np.where(same, w, 0.0), w)
            sec_kids += int((has & ~same.all(1)).sum())
        w /= w.sum(1, keepdims=True)
        wnear[b0:b1] = w[:, 0]
        if ring:
            own = gi[np.arange(b), np.argmax(w, 1)]         # its nearest allowed guide
            g1 = leader[own]                                # and that guide's lock
            ring_g[b0:b1] = g1
            dgx = X_guides[g1] - X_c[b0:b1]                # the follicle to its lock's root
            rb_ = (rr_u[b0:b1] * R_lock[g1])[:, None, None]
            thb = th_all[b0:b1, None]
            corr = d0_c[b0:b1] - d0_g[g1]
            # a member rooted far from its lock's root (a merged lock) takes a longer arc to join it
            gth_c = np.where(g1 != own, np.maximum(gath, 1.5 * np.linalg.norm(dgx, axis=1)), gath)[:, None]

            def shape(s):
                gt = np.clip(s / gth_c, 0.0, 1.0)
                gth = (gt * gt * (3.0 - 2.0 * gt))[..., None]
                ang = (thb + (2.0 * np.pi * s / tw if tw > 0 else 0.0))[..., None]
                out = _lookup(T, h, g1, s) + gth * (dgx[:, None, :] + rb_ * (np.cos(ang) * _lookup(F1, h, g1, s)
                                                                            + np.sin(ang) * _lookup(F2, h, g1, s)))
                out += ((1.0 - smoothstep(0.0, rb, s)) * s)[..., None] * corr[:, None, :]
                return out
        else:
            dbar = (w[..., None] * d0_g[gi]).sum(1)
            corr = d0_c[b0:b1] - dbar                      # the child's own follicle against its guides'

            def shape(s):
                out = np.zeros(s.shape + (3,))
                for a in range(k):
                    out += w[:, a, None, None] * _lookup(T, h, gi[:, a], s)
                out += ((1.0 - smoothstep(0.0, rb, s)) * s)[..., None] * corr[:, None, :]
                return out

        if probe is not None:
            sp = np.broadcast_to(np.asarray(probe_mm, np.float64) / 1000.0, (b, len(probe_mm)))
            probe[b0:b1] = X_c[b0:b1, None] + shape(sp)
        Lb = L_c[b0:b1]
        s = ((np.minimum(Lb * ext_f, Lg_r[g1]) if ring else Lb * ext_f))[:, None] * u[None, :]
        C = shape(s)
        seg = np.linalg.norm(np.diff(C, axis=1), axis=2)
        sig = np.concatenate([np.zeros((b, 1)), np.cumsum(seg, 1)], 1)
        # past the sampled curve (only if 1.6 x the cut length of blended curve is still shorter than the hair):
        # straight on along the last direction (never for a ringlet member: it ends with its lock)
        need = np.maximum(1.2 * Lb - sig[:, -1], 0.0) * (not ring)
        short_ext += int((need > 0).sum())
        dl = _unit(C[:, -1] - C[:, -2])
        C = np.concatenate([C, C[:, -1:] + (need + 1e-6)[:, None, None] * dl[:, None, :]], 1)
        sig = np.concatenate([sig, sig[:, -1:] + need[:, None] + 1e-6], 1)
        lam = Lb.copy()
        tj = np.arange(npts) / (npts - 1.0)
        for _ in range(3):
            Q = _resample(C, sig, lam[:, None] * tj[None, :])
            chord = np.linalg.norm(np.diff(Q, axis=1), axis=2).sum(1)
            lam = np.minimum(lam * Lb / np.maximum(chord, 1e-300), sig[:, -1])
        Q = _resample(C, sig, lam[:, None] * tj[None, :])
        len_err[b0:b1] = np.linalg.norm(np.diff(Q, axis=1), axis=2).sum(1) / np.maximum(Lb, 1e-300) - 1.0
        P[b0:b1] = (X_c[b0:b1, None] + Q).astype(np.float32)
    rep = dict(children=int(n), k=k, other_side_neighbours=viol, nearest_weight_p10_50_90=(
        np.percentile(wnear, [10, 50, 90]).round(4).tolist() if n else []),
        length_rel_err_max=float(np.abs(len_err).max()) if n else 0.0, extended_past_sample=short_ext,
        route_split=split, route_neighbours_dropped=route_cut, route_children=int(route_kids.sum()),
        section_children=(sec_kids if sec_c is not None else None),
        table_mb=T.nbytes / 2 ** 20, table_s=round(t_tab, 3), seconds=round(time.time() - t0, 3),
        mode=("ringlet" if ring else "blend"))
    if ring and n:
        cnt = np.bincount(ring_g, minlength=len(G))
        rep.update(ringlet_r_mm=R_r * 1000, ringlet_gather_mm=gath * 1000, ringlet_twist_mm=tw * 1000,
                   members_per_lock_p10_50_90=np.percentile(cnt[cnt > 0], [10, 50, 90]).round(1).tolist(),
                   locks_used=int((cnt > 0).sum()), ended_with_lock_short_2pct=int((len_err < -0.02).sum()),
                   ringlet_guide=ring_g, ringlet_merge=mrep, ringlet_r_sd=rsd,
                   lock_r_mm_p10_50_90=np.percentile(R_lock[cnt > 0] * 1000, [10, 50, 90]).round(2).tolist())
    if log:
        log("children: %d in %.1f s (table %.1f MB), other-side neighbours %d, length err max %.2e" % (
            n, rep["seconds"], rep["table_mb"], viol, rep["length_rel_err_max"]))
    if probe is not None:
        return P, rep, probe
    return P, rep


# ---------------------------------------------------------------- short hairs (DESIGN_v2 4.8)
def _distance(body):
    """the body's signed distance and gradient: settle.smooth_distance (C1, the quadratic B-spline of the grid) for a
    grid.SDFGrid, so the contact force has no jumps at the cell faces and the line search sees a smooth energy; any
    other body its own .sample(P, grad=True)"""
    if body is None:
        return None
    if hasattr(body, "A") and hasattr(body, "lo") and hasattr(body, "h"):
        from . import settle as SE
        return lambda P: SE.smooth_distance(body, P)
    return lambda P: body.sample(P, grad=True)


def _c2_contact(p, dl):
    """the contact energy per unit stiffness and its slope at clearance deficit p (m): 0 for p <= 0, p^3 / (6 dl) up
    to dl, then p^2 / 2 - dl p / 2 + dl^2 / 6 (value, slope and curvature continuous at 0 and dl); dl 0 is the plain
    one-sided spring p^2 / 2"""
    if dl <= 0.0:
        a = np.maximum(p, 0.0)
        return 0.5 * a * a, a
    a = np.clip(p, 0.0, dl)
    full = p >= dl
    e = np.where(full, 0.5 * p * p - 0.5 * dl * p + dl * dl / 6.0, a ** 3 / (6.0 * dl))
    d1 = np.where(full, p - 0.5 * dl, a * a / (2.0 * dl))
    return e, d1


def _c2_ramp(t):
    """the canopy's load profile at t = deficit / onset: energy / (F onset) = t^3 - t^4 / 2 up to 1, then t - 1/2; its
    slope (the share of the full load F) 3 t^2 - 2 t^3 then 1, and the slope's slope / onset (6 t - 6 t^2) then 0.
    The force rises smoothly from nothing to the layer's full load over the onset and stays there (C2)"""
    tc = np.clip(t, 0.0, 1.0)
    far = t > 1.0
    e = np.where(far, t - 0.5, tc ** 3 - 0.5 * tc ** 4)
    d1 = np.where(far, 1.0, 3 * tc * tc - 2 * tc ** 3)
    d2 = np.where(far, 0.0, 6 * tc - 6 * tc * tc)
    return e, d1, d2


class Canopy:
    """the long hair's layer over the scalp, as the short hairs under it meet it (10-30: a growth-cycle
    short hair rises at its exit angle until it meets the long hair lying over it, then bends under that hair's weight;
    only where the long hair is thin - the part, the hairline - does it stand out as a fly-away).

    Built from the settled guides: node positions P (G, K + 1, 3) m, and per segment its member hairs' weight W (G, K) N,
    fibre volume V (G, K) m3 and length l (G, K) m (settle.Rods.W / .V / .l). Each segment stands at its foot on the skin
    (its midpoint moved down the body's distance gradient by its distance), at its height there. Its outer side is that
    height plus half the thickness of that guide's own hairs (the guide is their centre line): (fibre area V / l per
    guide spacing) / phi / 2. The layer's outer surface over a skin point (grow.canopy_height 'mass', the default) is
    the mean of the outer sides of the segments whose feet lie within R (1.5 guide spacings), weighted by the fibre each
    puts there (kernel x V): where the long hair is, as the children blend their guides. 'max' (round 9) takes the
    highest segment within R, so one guide arching off the skin holds the layer up over its whole reach.
    The load the layer puts on what props it up is its areal weight there: the weight of the segments whose feet lie
    within R, spread by the kernel (1 - d^2 / R^2)^2 (area pi R^2 / 3), N/m2. The full weight is taken, not its
    component along the skin's normal: on the sides the hanging layer presses in by its tension x the skull's
    curvature, T / R_head with T the weight hanging below (~3 x the local weight for 300 mm lengths over a 90 mm
    skull), so the weight alone is the smaller of the two there.

    Both are tabled on a lattice over the short roots' region (each lattice node at its own foot) and read by the
    settle's quadratic B-spline (C1, value and exact gradient), so the contact's energy and force agree. The height is
    floored at the body clearance + both onsets: a hair can always lie on the skin under the layer."""

    def __init__(self, P, W, V, l, dist, lo, hi, phi=CHILDREN["canopy_phi"], grid_mm=CHILDREN["canopy_grid_mm"],
                 floor=0.00105, reach=0.09, log=None, height=CHILDREN["canopy_height"]):
        from . import settle as SE
        t0 = time.time()
        self._bs = SE._bspline2
        P = np.asarray(P, np.float64)
        G = P.shape[0]
        l = np.asarray(l, np.float64)
        act = l > 0
        gi = np.nonzero(act)[0]
        M = (0.5 * (P[:, :-1] + P[:, 1:]))[act]
        Wk = np.asarray(W, np.float64)[act]
        Vk = np.asarray(V, np.float64)[act]
        lk = l[act]
        sd, g = dist(M)
        F = M - sd[:, None] * _unit(g)
        # the guides' spacing across the layer: the mean distance from each guide's root to its 4 nearest
        dd, _ = cKDTree(P[:, 0]).query(P[:, 0], k=min(5, G))
        sg = dd[:, 1:].mean(1) if G > 1 else np.full(G, 0.008)
        self.R = 1.5 * float(np.median(sg))
        hb = 0.5 * (Vk / lk) / (float(phi) * sg[gi])
        o = sd + hb
        h = float(grid_mm) / 1000.0
        lo = np.asarray(lo, np.float64)
        shape = np.maximum(np.ceil((np.asarray(hi) - lo) / h).astype(int) + 1, 3)
        ax = [lo[i] + h * np.arange(shape[i]) for i in range(3)]
        Q = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        Hq = np.full(len(Q), float(floor))
        Wq = np.zeros(len(Q))
        tree = cKDTree(F)
        kq = min(96, len(F))
        R = self.R
        area = np.pi * R * R / 3.0
        shell = 0
        for c0 in range(0, len(Q), 40000):
            q = Q[c0:c0 + 40000]
            s, gq = dist(q)
            ins = np.nonzero((s > -0.008) & (s < reach))[0]
            if len(ins) == 0:
                continue
            shell += len(ins)
            fq = q[ins] - s[ins, None] * _unit(gq[ins])
            d, j = tree.query(fq, k=kq, distance_upper_bound=R)
            ok = np.isfinite(d)
            jj = np.where(ok, j, 0)
            kern = np.where(ok, (1.0 - (np.where(ok, d, 0.0) / R) ** 2) ** 2, 0.0)
            if height == "max":
                Hq[c0 + ins] = np.maximum(np.where(ok, o[jj], -np.inf).max(1), floor)
            else:
                # where the long hair's fibre is: its outer sides o averaged by the fibre volume each segment puts over
                # this point (the same kernel as the load), as the children blend their guides - one guide arching off
                # the skin does not lift the whole layer to its height. No separate stack term: the settle's hair-hair
                # pressure already spaced the guides by the layer's packing, so their heights carry the stack (a stack
                # of fibre volume / area / canopy_phi on top counted it twice: 12-15 mm over gr11's temples, where the
                # long children lie at 5.8 / 9-10 mm p50 / p90)
                kv = kern * Vk[jj]
                sv = kv.sum(1)
                Hm = (kv * o[jj]).sum(1) / np.maximum(sv, 1e-300)
                Hq[c0 + ins] = np.maximum(np.where(sv > 0, Hm, floor), floor)
            Wq[c0 + ins] = (np.where(ok, Wk[jj] * kern, 0.0)).sum(1) / area
        self.lo, self.h, self.shape = lo, h, tuple(int(v) for v in shape)
        self.hi = lo + h * (shape - 1)
        self.AH = Hq.reshape(self.shape)
        self.AW = Wq.reshape(self.shape)
        # the lowest height over each node's B-spline support (its 3 x 3 x 3 neighbours): a point closer to the skin
        # than that is inside the layer for certain, and its load and gradient are exactly 0, so it is not read
        from scipy.ndimage import minimum_filter
        self.AHmin = minimum_filter(self.AH, size=3, mode="nearest")
        self.floor = float(floor)
        self.report = dict(guides=int(G), segments=int(len(M)), R_mm=round(self.R * 1e3, 2),
                           guide_spacing_mm_p50=round(float(np.median(sg)) * 1e3, 2),
                           half_bundle_mm_p50_p90=[round(float(np.percentile(hb, q)) * 1e3, 3) for q in (50, 90)],
                           lattice=list(self.shape), shell_nodes=int(shell), seconds=round(time.time() - t0, 3))
        if log:
            log("canopy: %d guide segments, R %.1f mm, lattice %s, %.1f s" % (len(M), self.R * 1e3, self.shape,
                                                                             time.time() - t0))

    def maybe_out(self, P, sd):
        """the points that may stand out of the layer (sd over the lowest height of their B-spline support); the rest
        are inside for certain"""
        i = np.clip(np.rint((P - self.lo) / self.h).astype(np.int64), 0, np.array(self.shape) - 1)
        return sd > self.AHmin[i[:, 0], i[:, 1], i[:, 2]]

    def sample(self, P):
        """the layer's outer height over the skin (m) and its gradient, and the layer's areal weight (N/m2) and its
        gradient, at P (n, 3); outside the lattice the edge values with no gradient (the lattice covers every short
        hair's reach)"""
        Pc = np.clip(P, self.lo, self.hi)
        Hc, gH = self._bs(self.AH, self.lo, self.h, Pc)
        wa, gw = self._bs(self.AW, self.lo, self.h, Pc)
        out = np.any(P != Pc, 1)
        if out.any():
            gH[out] = 0.0
            gw[out] = 0.0
        return Hc, gH, np.maximum(wa, 0.0), gw


class _ShortRods:
    """the short hairs' energy, separable per hair (no hair-hair term), so each hair can be solved on its own while
    all of them are evaluated in one vectorised pass. Variables are Jacobi-scaled directions x = u s (DESIGN_v2 4.5).

    With a Canopy, each node outside the long hair's layer lifts it: the work of lifting the layer's weight over the
    node's share of it by the node's height above the layer, F p with F = (the layer's areal weight) x (the node's
    segment length x the spacing of the short hairs there, at most R: the strip of layer the hair props up), switched
    on over the layer's onset (_c2_ramp). A stiff short hair under a thin layer can still lift it and stand; under a
    full layer the weight bends it down to lie inside."""

    def __init__(self, X, d0, L, fibre, dist, sc, m, hc, kc, kb=CHILDREN["contact_kb"],
                 onset=CHILDREN["contact_onset_mm"] / 1000.0, kink=CHILDREN["short_kink_urad"] * 1e-6,
                 canopy=None, canopy_onset=CHILDREN["canopy_onset_mm"] / 1000.0, held=None, set_r=None,
                 fibre_fric=None, eps=CHILDREN["short_friction_eps_mm"] / 1000.0, phi=CHILDREN["canopy_phi"],
                 fload=CHILDREN["short_friction_load"], fstiff=CHILDREN["short_friction_stiff"]):
        n = len(X)
        self.X, self.d0, self.L, self.m, self.kc, self.dist = X, d0, L, m, kc, dist
        # the set (dry.span 'held', as actions.set_held does for the guides): a short hair is in the brush with its
        # section, held on the stroke while it dries, so every joint takes rest turning rho = r (held - natural) with
        # natural 0 (a straight fibre), held the turning of the shape it was held in (Dh, the segment directions) and
        # r = dry.set x retention (wear, finish.hold where sprayed). Released, it rests where its own set, its weight,
        # the skin and the layer over it balance - and it starts from the held shape, as it leaves the brush.
        self.Dh = None
        self.rho = None
        if held is not None:
            Dh = _unit(np.asarray(held, np.float64))
            r = np.broadcast_to(np.asarray(1.0 if set_r is None else set_r, np.float64), (n,))
            self.Dh = Dh
            self.r_set = r
        self.cano = canopy if dist is not None else None
        self.dlc = float(canopy_onset)
        if self.cano is not None:
            # each node's share of the layer it props up (m2): its segment length (half at the tip) x the spacing of
            # the short hairs around its root, from the 6th nearest (density 6 / pi d6^2), at most the layer's R
            k6 = min(7, n)
            if n > 1:
                d6 = cKDTree(X).query(X, k=k6)[0][:, -1]
                sp = np.minimum(d6 * np.sqrt(np.pi / max(k6 - 1, 1)), self.cano.R)
            else:
                sp = np.full(n, self.cano.R)
            share = np.ones(m)
            share[-1] = 0.5
            self.Ac = (L / m * sp)[:, None] * share[None, :]
            self.spacing = sp
        self.dl = float(onset)
        self.l = L / m                                                    # (n,) segment length
        sj = (np.arange(m + 1) / m)[None, :] * L[:, None]                 # node arcs, root first
        smid = ((np.arange(m) + 0.5) / m)[None, :] * L[:, None]
        B = fibre.EI(sj[:, :m], L[:, None], sc[:, None])                   # joint j: between segment j-1 and j (j=0: d0)
        self.Wseg = fibre.w(smid, L[:, None], sc[:, None]) * self.l[:, None]   # N
        self.Wn = np.concatenate([0.5 * (self.Wseg[:, :-1] + self.Wseg[:, 1:]), 0.5 * self.Wseg[:, -1:]], 1)  # nodes 1..m
        self.cb = B / self.l[:, None]
        # each node's contact stiffness (N/m): its weight's share and its own bending stiffness's share (contact_kb)
        self.kw = kc * self.Wn + kb * fibre.EI(sj[:, 1:], L[:, None], sc[:, None]) / self.l[:, None] ** 3
        self.Tj = np.cumsum(self.Wseg[:, ::-1], 1)[:, ::-1]              # weight hanging from segment j on
        # the balance test's moment scale per segment (N m): its carried weight x its length, floored at a kink of
        # `kink` rad against its own joint's bending (short_kink_urad)
        self.Mref = np.maximum(self.Tj * self.l[:, None], self.cb * float(kink))
        Bn = np.concatenate([B[:, 1:], np.zeros((n, 1))], 1)
        s2 = 2 * self.cb + 2 * Bn / self.l[:, None] + self.l[:, None] * self.Tj / 2
        self.Esc = self.Wseg.sum(1) * 1e-3                                 # per hair: its weight x 1 mm
        self.s = np.sqrt(s2 / self.Esc[:, None])[..., None]
        # the contact clearance. A hair rests h_c off the skin (the settle's layer), but it leaves the skin at its own
        # follicle: on its first arcs it is not lying on the scalp, it is coming out of it. So the clearance a node
        # must keep ramps from the root's own reading up to h_c over the arc its straight exit takes to rise 2 h_c
        # (half the rise of the follicle's own direction), and the follicle's exit is never itself a contact. The C2
        # onset lets a node at rest sit up to half the onset inside its clearance, so the clearance is raised by the
        # onset, but never above 3/4 of the follicle's own straight exit (sd0 + s sin a): a requirement the straight
        # hair meets with room to spare can never press a hair, however short (a 0.04 mm new hair cannot rise 0.05 mm,
        # and one resting exactly on its requirement sits in the stiffest part of the spring). Where the
        # straight exit has left the collider the node must keep at least the onset (or the exit's own height), so
        # it rests outside; where it has not (a root reading inside the collider: the scalp mesh and the collider
        # differ by up to ~0.4 mm on the head), the ramp from the root's own reading stands.
        if dist is not None:
            sd0, g0 = dist(X)
            sina = np.clip((d0 * _unit(g0)).sum(1), np.sin(np.radians(3.0)), 1.0)
            self.sd_root = sd0
            self.sina = sina
            ramp = 0.5 * sj[:, 1:] * sina[:, None]
            self.hreq = np.minimum(hc + self.dl, np.maximum(sd0[:, None] + ramp + np.minimum(self.dl, 0.5 * ramp),
                                                            np.minimum(self.dl, sd0[:, None] + 1.5 * ramp)))
        else:
            self.sd_root = np.zeros(n)
            self.sina = np.ones(n)
            self.hreq = None
        if self.Dh is not None:
            # the held shape as the brush laid it: the stroke's path carried to this root can pass through the skin
            # where the head is not round (by the ear the guides' path, moved a few mm, runs up to ~1 mm inside the
            # auricle - measured on gr9), but a brush lays hair on the skin, never in it. So the held shape is built
            # segment by segment from the follicle with each node pushed out to its clearance (start), and the set
            # takes the turning of that shape
            if dist is not None:
                U0, _ = self.start()
                self.Dh = _unit(U0 / self.s)
            Dh = self.Dh
            self.rho = self.r_set[:, None, None] * (Dh - np.concatenate([d0[:, None], Dh[:, :-1]], 1))
        # the layer's hold (short_friction): each held node a static friction anchor under the long hair lying over it
        self.fA = self.fn = self.fmuN = self.fN_w = None
        self.feps = float(eps)
        if self.Dh is not None and fibre_fric is not None and dist is not None and self.cano is not None:
            A = X[:, None] + np.cumsum(self.l[:, None, None] * self.Dh, 1)        # the held nodes 1..m
            Af = A.reshape(-1, 3)
            sdA, gA = dist(Af)
            Hc, _, wa, _ = self.cano.sample(Af)
            wa = wa.reshape(n, m)
            under = (sdA - Hc <= self.dlc).reshape(n, m) & (wa > 0)              # inside the long hair's layer
            # the layer's weight over the node's strip (N), and the oil bridges to the long hairs around it (N/m x
            # the node's share of fibre: its segment, half at the tip)
            share = np.ones(m)
            share[-1] = 0.5
            Na = fibre_fric.adhesive_load(sj[:, 1:], phi) * (self.l[:, None] * share[None, :])
            # the layer's weight over the strip reaches the skin through the fibres lying on it (short_friction_load
            # 'contacts'): the strip, as wide as the short hairs' spacing, holds 2 phi sp / (pi r) long hairs of the
            # layer's bottom row (fibres within one diameter of the skin, at the layer's fibre fraction), each carrying
            # its own share; the short hair takes 1 / (1 + that). 'strip': the whole strip's weight (round 12)
            if fload == "contacts":
                n_bot = 2.0 * float(phi) * self.spacing / (np.pi * float(fibre_fric.r_root))
                self.fshare = 1.0 / (1.0 + n_bot)
            else:
                self.fshare = np.ones(n)
            Nw = wa * self.Ac * self.fshare[:, None]
            N = np.where(under, Nw + Na, 0.0)
            self.fA = A
            self.fn = _unit(gA).reshape(n, m, 3)
            self.fmuN = float(fibre_fric.mu) * N
            self.fN_w = N / np.maximum(self.Wn, 1e-300)                           # load over the node's own weight
            self.fNw_w = np.where(under, Nw, 0.0) / np.maximum(self.Wn, 1e-300)   # its layer-weight part
            self.fNa_w = np.where(under, Na, 0.0) / np.maximum(self.Wn, 1e-300)   # its oil-bridge part
            if fstiff and self.hreq is not None:
                # the skin as firm as the friction: mu N / onset more stiffness on each held node, so a pull of mu N
                # into the skin (anchor plane tilted against the skin's under the node) meets ~1.5 onsets of travel
                self.kw = self.kw + self.fmuN / max(self.dl, 1e-6)

    def core(self, idx, x, ratio=False):
        """per hair energy / Esc (b,) and its gradient (b, m, 3) at x (b, m, 3) for hairs idx; with ratio also each
        hair's worst segment balance, |the unbalanced moment on segment j| / (the weight it carries x its length), and
        the nodes 1..m (b, m, 3) m"""
        m, l = self.m, self.l[idx]
        s = self.s[idx]
        U = x / s
        nu = np.linalg.norm(U, axis=2, keepdims=True)
        D = U / nu
        X = self.X[idx]
        cb, Wseg = self.cb[idx], self.Wseg[idx]
        P = X[:, None] + np.cumsum(l[:, None, None] * D, 1)               # nodes 1..m
        Dp = np.concatenate([self.d0[idx][:, None], D[:, :-1]], 1)
        dd = D - Dp
        if self.rho is not None:
            dd = dd - self.rho[idx]                                       # the set's rest turning
        E = 0.5 * (cb[..., None] * dd * dd).sum((1, 2))
        gD = cb[..., None] * dd
        gD[:, :-1] -= cb[:, 1:, None] * dd[:, 1:]
        # gravity on the segment midpoints, heights from the root (keeps the energy's float64 resolution)
        zmid = np.cumsum(l[:, None] * D[..., 2], 1) - 0.5 * l[:, None] * D[..., 2]
        E += (Wseg * zmid).sum(1)
        gD[..., 2] += l[:, None] * (np.cumsum(Wseg[:, ::-1], 1)[:, ::-1] - 0.5 * Wseg)
        if self.dist is not None:
            sd, g = self.dist(P.reshape(-1, 3))
            sd = sd.reshape(len(idx), m)
            kw = self.kw[idx]
            ec, fc = _c2_contact(self.hreq[idx] - sd, self.dl)
            E += (kw * ec).sum(1)
            gP = -(kw * fc)[..., None] * g.reshape(len(idx), m, 3)
            if self.cano is not None:
                Pf = P.reshape(-1, 3)
                sdf = sd.ravel()
                q = np.nonzero(self.cano.maybe_out(Pf, sdf))[0]          # the rest: inside, no load, no gradient
                if len(q):
                    Hc, gH, wa, gw = self.cano.sample(Pf[q])
                    er, dr, _ = _c2_ramp((sdf[q] - Hc) / self.dlc)
                    A = self.Ac[idx].ravel()[q]
                    F = wa * A
                    Eq = np.zeros(len(sdf))
                    Eq[q] = F * self.dlc * er
                    E += Eq.reshape(len(idx), m).sum(1)
                    gq = np.zeros((len(sdf), 3))
                    gq[q] = (A * self.dlc * er)[:, None] * gw + (F * dr)[:, None] * (g.reshape(-1, 3)[q] - gH)
                    gP += gq.reshape(len(idx), m, 3)
            if self.fA is not None:
                # static friction against the layer: mu N eps (sqrt(1 + |Pi_t (P - A)|^2 / eps^2) - 1), the settle's
                # smoothed Coulomb law in the skin's tangent plane at the anchor
                dP = P - self.fA[idx]
                nA = self.fn[idx]
                dt = dP - (dP * nA).sum(2, keepdims=True) * nA
                eps = self.feps
                sq = np.sqrt(1.0 + (dt * dt).sum(2) / (eps * eps))
                muN = self.fmuN[idx]
                E += (muN * eps * (sq - 1.0)).sum(1)
                gP += (muN / (eps * sq))[..., None] * dt
            gD += l[:, None, None] * np.cumsum(gP[:, ::-1], 1)[:, ::-1]
        # conditioning: keep |u| near 1 (DESIGN_v2 4.5)
        E += 0.5 * (cb * (nu[..., 0] - 1) ** 2).sum(1)
        gt = gD - (gD * D).sum(2, keepdims=True) * D
        gU = gt / nu + (cb * (nu[..., 0] - 1))[..., None] * D
        esc = self.Esc[idx]
        out = (E / esc, gU / s / esc[:, None, None])
        if ratio:
            r = np.linalg.norm(gt, axis=2) / np.maximum(self.Mref[idx], 1e-300)
            out = out + (r.max(1), P)
        return out

    def precond(self, idx, x, chunk=2000):
        """each hair's dense Gauss-Newton matrix (b, 3m, 3m) in the scaled variables x, of the energy / Esc: bending
        (the joint chain, its neighbour coupling included), tension (DESIGN_v2 4.5's l T / 2), the |u| term, and the
        contact springs of the nodes inside their clearance, each at its own C2 curvature (k x min(p / onset, 1)). A
        node's spring couples every segment upstream of it (dP_q / dU_j = l (I - D_j D_j^T) / |U_j| for j <= q),
        which is what the diagonal scaling missed. (Taking the full spring from one onset short of contact, to make
        a hair about to touch step as if held, froze the new hairs under 0.05 mm tall: every node of theirs is
        that close, and they crawled to the iteration cap.)"""
        m = self.m
        out = np.empty((len(idx), 3 * m, 3 * m))
        I3 = np.eye(3)
        jj = np.arange(m)
        tri = np.tril(np.ones((m, m)))[None, :, :, None]
        for c0 in range(0, len(idx), chunk):
            ii = idx[c0:c0 + chunk]
            b = len(ii)
            s = self.s[ii]
            U = x[c0:c0 + chunk] / s
            nu = np.linalg.norm(U, axis=2)
            D = U / nu[..., None]
            Pj = I3[None, None] - D[..., :, None] * D[..., None, :]            # (b, m, 3, 3)
            J = Pj / nu[..., None, None]                                        # dD_j / dU_j
            JJ = Pj / (nu * nu)[..., None, None]                                # J^T J (P idempotent)
            cb, l = self.cb[ii], self.l[ii]
            cbn = np.concatenate([cb[:, 1:], np.zeros((b, 1))], 1)
            H = np.zeros((b, m, 3, m, 3))
            dg = ((cb + cbn + 0.5 * l[:, None] * self.Tj[ii])[..., None, None] * JJ
                  + cb[..., None, None] * D[..., :, None] * D[..., None, :])
            H[:, jj, :, jj, :] = dg.transpose(1, 0, 2, 3)
            off = -cb[:, 1:, None, None] * np.einsum("bjxy,bjyz->bjxz", J[:, :-1], J[:, 1:])
            H[:, jj[:-1], :, jj[1:], :] = off.transpose(1, 0, 2, 3)
            H[:, jj[1:], :, jj[:-1], :] = off.transpose(1, 0, 3, 2)
            if self.dist is not None:
                P = self.X[ii][:, None] + np.cumsum(l[:, None, None] * D, 1)
                sd, g = self.dist(P.reshape(-1, 3))
                sd, g = sd.reshape(b, m), g.reshape(b, m, 3)
                pq = self.hreq[ii] - sd                                          # (b, m), nodes 1..m
                kq = self.kw[ii] * (np.clip(pq / self.dl, 0.0, 1.0) if self.dl > 0 else (pq > 0))
                h = np.nonzero(kq.any(1))[0]
                if len(h):
                    V = np.einsum("bjxy,bqy->bqjx", J[h], g[h])                  # (h, q, j, 3): J_j g_q
                    V *= tri * (l[h, None, None, None] * np.sqrt(kq[h])[:, :, None, None])
                    H[h] += np.einsum("bqjx,bqky->bjxky", V, V)
                if self.cano is not None:
                    # the layer's load where it is switching on (its onset's curvature F (6t - 6t^2) / onset) along
                    # the layer's normal, coupled upstream like the body's springs
                    Pf = P.reshape(-1, 3)
                    q = np.nonzero(self.cano.maybe_out(Pf, sd.ravel()))[0]
                    kc_ = np.zeros(b * m)
                    gH = np.zeros((b * m, 3))
                    if len(q):
                        Hc, gH[q], wa, _ = self.cano.sample(Pf[q])
                        _, _, d2 = _c2_ramp((sd.ravel()[q] - Hc) / self.dlc)
                        kc_[q] = wa * self.Ac[ii].ravel()[q] * d2 / self.dlc
                    kc_ = kc_.reshape(b, m)
                    h2 = np.nonzero((kc_ > 0).any(1))[0]
                    if len(h2):
                        nq = g[h2] - gH.reshape(b, m, 3)[h2]
                        V = np.einsum("bjxy,bqy->bqjx", J[h2], nq)
                        V *= tri * (l[h2, None, None, None] * np.sqrt(kc_[h2])[:, :, None, None])
                        H[h2] += np.einsum("bqjx,bqky->bjxky", V, V)
                if self.fA is not None:
                    # the friction springs, mu N / (eps sqrt(1 + |d_t|^2 / eps^2)) along the two tangent directions
                    muN = self.fmuN[ii]
                    h3 = np.nonzero((muN > 0).any(1))[0]
                    if len(h3):
                        nA = self.fn[ii][h3]
                        dP = P[h3] - self.fA[ii][h3]
                        dt = dP - (dP * nA).sum(2, keepdims=True) * nA
                        kf = muN[h3] / (self.feps * np.sqrt(1.0 + (dt * dt).sum(2) / self.feps ** 2))
                        a_ = np.where(np.abs(nA[..., 0:1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
                        t1 = _unit(np.cross(nA, a_))
                        t2 = np.cross(nA, t1)
                        for t_ in (t1, t2):
                            V = np.einsum("bjxy,bqy->bqjx", J[h3], t_)
                            V *= tri * (l[h3, None, None, None] * np.sqrt(kf)[:, :, None, None])
                            H[h3] += np.einsum("bqjx,bqky->bjxky", V, V)
            H = H.reshape(b, 3 * m, 3 * m)
            sv = np.repeat(s[..., 0], 3, axis=1)                                # (b, 3m)
            H /= sv[:, :, None] * sv[:, None, :] * self.Esc[ii][:, None, None]
            out[c0:c0 + chunk] = 0.5 * (H + H.transpose(0, 2, 1))
        return out

    def start(self):
        """each rod built segment by segment from its follicle, straight on, each node pushed out to its clearance
        along the body's normal (a hair starts lying over the skin it would have entered, never inside it)"""
        n, m = len(self.X), self.m
        U0 = np.empty((n, m, 3))
        P = self.X.copy()
        D = self.d0.copy()
        moved = 0
        for j in range(m):
            D = D.copy() if self.Dh is None else self.Dh[:, j].copy()     # held: it leaves the brush in that shape
            for _ in range(8):
                Q = P + self.l[:, None] * D
                if self.dist is None:
                    break
                sd, g = self.dist(Q)
                need = self.hreq[:, j] + 1e-5 - sd
                bad = need > 0
                if not bad.any():
                    break
                moved += int(bad.sum())
                D[bad] = _unit(Q[bad] + need[bad, None] * _unit(g[bad]) - P[bad])
            U0[:, j] = D
            P = P + self.l[:, None] * D
        return U0 * self.s, moved


def _lbfgs_per_hair(rods, x0, tol, max_iter, mem=8, flat_de=1e-9, refresh=10):
    """L-BFGS run separately for every hair, vectorised over the hairs still moving: each has its own history, its own
    step (Armijo backtracking) and its own stop (worst segment balance under tol). A hair leaves the batch when it is
    balanced, so the cost falls as they settle. Each step's initial inverse Hessian is the hair's own dense
    preconditioner (rods.precond, inverted), refreshed for the moving hairs every `refresh` iterations; refresh 0 is
    the scaled identity (the plain Jacobi-scaled L-BFGS). A hair stops as 'stalled' when its line search finds no decrease in
    30 cuts (x 1/4), or when its energy fell by under flat_de (x its weight x 1 mm) over its last 20 iterations: it
    is at rest to well under a micrometre of its own weight's work (a node creeping along the contact onset, where
    the penalty's curvature jumps, can otherwise take hundreds of tiny steps). Returns (x, worst ratio per hair,
    report)."""
    n = len(x0)
    shp = x0.shape[1:]
    nv = int(np.prod(shp))
    x = x0.reshape(n, nv).copy()
    E, g, r, _ = rods.core(np.arange(n), x.reshape((n,) + shp), ratio=True)
    g = g.reshape(n, nv)
    S = np.zeros((n, mem, nv))
    Y = np.zeros((n, mem, nv))
    rho = np.zeros((n, mem))
    gam = np.ones(n)
    iters = np.zeros(n, np.int64)
    state = np.where(r < tol, 1, 0)                     # 0 moving, 1 balanced, 2 stalled
    Eh = np.zeros((n, 20))
    hair_evals, calls, k = n, 1, 0
    pc = refresh > 0 and hasattr(rods, "precond")
    row = np.full(n, -1, np.int64)            # each moving hair's row in Minv, set at its last refresh
    Minv = None
    refreshes, pc_s = 0, 0.0
    while k < max_iter:
        a = np.nonzero(state == 0)[0]
        if len(a) == 0:
            break
        if pc and k % refresh == 0:
            tp = time.time()
            Minv = None                                                    # free the old store first
            M = rods.precond(a, x[a].reshape((len(a),) + shp))
            M[:, np.arange(nv), np.arange(nv)] += 1e-9 * np.trace(M, axis1=1, axis2=2)[:, None] / nv
            Minv = np.linalg.inv(M)
            del M
            Minv = 0.5 * (Minv + Minv.transpose(0, 2, 1))
            row[:] = -1
            row[a] = np.arange(len(a))
            refreshes += 1
            pc_s += time.time() - tp
        q = g[a].copy()
        al = np.zeros((len(a), mem))
        slots = [(k - 1 - i) % mem for i in range(mem)]
        for sl in slots:                                                   # newest to oldest
            al[:, sl] = rho[a, sl] * (S[a, sl] * q).sum(1)
            q -= al[:, sl, None] * Y[a, sl]
        if pc:
            rr = np.einsum("bij,bj->bi", Minv[row[a]], q)
        else:
            rr = gam[a, None] * q
        for sl in slots[::-1]:
            be = rho[a, sl] * (Y[a, sl] * rr).sum(1)
            rr += S[a, sl] * (al[:, sl] - be)[:, None]
        d = -rr
        gd = (g[a] * d).sum(1)
        bad = ~(gd < 0)
        if bad.any():                                                      # not a descent direction: restart that hair
            if pc:                                                         # on the preconditioned gradient
                d[bad] = -np.einsum("bij,bj->bi", Minv[row[a[bad]]], g[a[bad]])
            else:
                d[bad] = -g[a[bad]]
            gd[bad] = (g[a[bad]] * d[bad]).sum(1)
            rho[a[bad]] = 0.0
            gam[a[bad]] = 1.0
        t = np.ones(len(a))
        xn, En, gn, rn = np.empty((len(a), nv)), np.empty(len(a)), np.empty((len(a), nv)), np.empty(len(a))
        acc = np.zeros(len(a), bool)
        pend = np.arange(len(a))
        for _ in range(30):
            idx = a[pend]
            xt = x[idx] + t[pend, None] * d[pend]
            Et, gt, rt, _ = rods.core(idx, xt.reshape((len(idx),) + shp), ratio=True)
            calls += 1
            hair_evals += len(idx)
            ok = Et <= E[idx] + 1e-4 * t[pend] * gd[pend]
            okp = pend[ok]
            xn[okp], En[okp], gn[okp], rn[okp] = xt[ok], Et[ok], gt.reshape(len(idx), nv)[ok], rt[ok]
            acc[okp] = True
            pend = pend[~ok]
            if len(pend) == 0:
                break
            t[pend] *= 0.25
        sl = k % mem
        ai = a[acc]
        sv = xn[acc] - x[ai]
        yv = gn[acc] - g[ai]
        sy = (sv * yv).sum(1)
        yy = (yv * yv).sum(1)
        good = sy > 1e-300
        S[ai, sl], Y[ai, sl] = sv, yv
        rho[ai, sl] = np.where(good, 1.0 / np.where(good, sy, 1.0), 0.0)
        gam[ai] = np.where(good, sy / np.maximum(yy, 1e-300), gam[ai])
        rho[a[~acc], sl] = 0.0
        x[ai], E[ai], g[ai], r[ai] = xn[acc], En[acc], gn[acc], rn[acc]
        pos = iters[ai] % 20
        old = Eh[ai, pos]
        Eh[ai, pos] = En[acc]
        flat = (iters[ai] >= 20) & (old - En[acc] < flat_de)
        iters[ai] += 1
        state[ai[flat]] = 2
        state[ai[rn[acc] < tol]] = 1
        state[a[~acc]] = 2
        k += 1
    st2 = state == 2
    rep = dict(iterations=int(k), calls=int(calls), hair_evaluations=int(hair_evals),
               stalled_ratio_max=float(r[st2].max()) if st2.any() else 0.0,
               balanced=int((state == 1).sum()), stalled=int((state == 2).sum()), moving=int((state == 0).sum()),
               iters_p50_p99_max=[float(np.percentile(iters, 50)), float(np.percentile(iters, 99)), int(iters.max())],
               precond_refreshes=int(refreshes), precond_s=round(pc_s, 3))
    return x.reshape(x0.shape), r, rep


def build_canopy(canopy, X, L, dist, cfg=None, log=None):
    """a Canopy over the short roots X (n, 3) m with lengths L (n,) m: canopy is a Canopy (returned as it is) or the
    settled guides as (P (G, K + 1, 3), W (G, K) N, V (G, K) m3, l (G, K) m) - settle.Rods' positions(), W, V, l"""
    if canopy is None or isinstance(canopy, Canopy):
        return canopy
    c = settings(cfg)
    P, W, V, l = canopy
    reach = float(np.max(L)) + 0.012 if len(L) else 0.09
    lo = X.min(0) - reach
    hi = X.max(0) + reach
    floor = (float(c["contact_mm"]) + float(c["contact_onset_mm"]) + float(c["canopy_onset_mm"])) / 1000.0
    return Canopy(P, W, V, l, dist, lo, hi, phi=float(c["canopy_phi"]), grid_mm=float(c["canopy_grid_mm"]),
                  floor=floor, reach=reach, log=log, height=str(c.get("canopy_height", "mass")))


def canopy_depth(cano, dist, P):
    """how far each point P (n, 3) m stands out of the long hair's layer (m; > 0 outside) and the layer's areal
    weight there (N/m2)"""
    sd = dist(P)[0]
    Hc, _, wa, _ = cano.sample(P)
    return sd - Hc, wa


def set_share(dry, wear, finish, phi_deg, din_mm):
    """the share r (n,) of its held turning each short hair keeps (actions.set_held's r for the guides): dry.set x
    retention(wear.age_h, wear.rh, finish.hold where the fixative went). finish.where 'hairline' sprays the roots at the
    front hairline - |phi| < 40 deg and within 15 mm of the hairline, actions' front_g rule, read at the hair's own
    root; 'all' sprays every hair. phi_deg, din_mm (n,) the roots' azimuth from the front and depth inside the hairline"""
    from .actions import retention
    n = len(np.atleast_1d(phi_deg))
    if (finish or {}).get("where", "all") == "all":
        sprayed = np.ones(n, bool)
    else:
        sprayed = (np.abs(np.asarray(phi_deg, np.float64)) < 40.0) & (np.asarray(din_mm, np.float64) < 15.0)
    hold = np.where(sprayed, float((finish or {}).get("hold", 0.0)), 0.0)
    return float(dry.get("set", 0.0)) * retention(float(wear.get("age_h", 0.0)), float(wear.get("rh", 45.0)), hold)


def laid_held(held, Qp, tucked, arcs_g, key_g, X_c, key_c, L_c, cfg=None, pin=None):
    """the short hairs of a tucked section held on the section's laid route (grow.short_laid).

    The tuck (actions.Acts._tuck_turn) is a route in space: the hand lifts the side section up and back to a height
    over the top of the ear, carries it over and lays it on the skin behind the ear. interpolate() moves a guide's shape
    to the child's root (offsets G_a(s) - X_a), so a short hair rooted 10-20 mm below its guides is "held" 10-20 mm
    below the route - across the helix (gr10: the ear's seen cover by short hairs 0.02 -> 0.13). A hand gathering a
    section does not translate it: every hair in the section is brought onto the one line the hand draws. So here a
    short hair whose interpolation weight is mostly on tucked guides (>= short_laid_share) is held on the tucked guides'
    laid path in place, A(s) = sum_a w_a Qp_a(s) (weights over the tucked neighbours only, renormalised), and is
    gathered onto it from its own root: H(s) = A(s) + (X_c - A(0)) (1 - smoothstep(0, g, s)), g = 2 |X_c - A(0)| (at
    least 5 mm) - the hair rises onto the route at about 1 in 2 from its follicle; the slope is not sourced, chosen as
    the gentlest that reaches the route within a short hair's length. The rest keep the held path they had.

    held (n, p, 3) m, interpolate's held path of the short hairs (root first); Qp (Gn, K + 1, 3) m, acts.Qp, the brush /
    hand position at each guide node's arc; tucked (Gn,) bool, guides with a tucked span (acts.tuck_pin.any(1));
    arcs_g (Gn, K + 1) m, the guides' node arcs Qp is laid at; key_g (Gn, 4), key_c (n, 4) the part keys; X_c (n, 3) m
    the short roots; L_c (n,) m their lengths; pin (Gn, K + 1) bool, acts.tuck_pin (the tucked span's nodes): with it a
    hair too short to reach 10 mm past where the route turns down behind the ear is not laid (actions' rule for the
    guides). Returns (held (n, p, 3) float32, report)."""
    c = settings(cfg)
    k = int(c["children_k"])
    soft2 = float(c["children_soft_mm"]) ** 2
    held = np.array(held, np.float32, copy=True)
    n, p = held.shape[:2]
    tucked = np.asarray(tucked, bool)
    rep = dict(tucked_guides=int(tucked.sum()), laid=0)
    if n == 0 or not tucked.any():
        return held, rep
    X_c = np.asarray(X_c, np.float64)
    L_c = np.asarray(L_c, np.float64).ravel()
    trees = SideTrees(key_g)
    d, gi = trees.query(key_c, k)
    ok = gi >= 0
    w = np.where(ok, 1.0 / (d * d + soft2), 0.0)
    w /= w.sum(1, keepdims=True)
    gi = np.where(ok, gi, gi[:, :1])
    wt = np.where(tucked[gi], w, 0.0)
    share = wt.sum(1)
    insec = share >= float(c["short_laid_share"])
    if pin is not None:
        # actions' own rule for the guides (_trace_paths: 'use'): a strand too short to reach 10 mm past where the route
        # comes over the ear and turns down behind it is not tucked - the hand cannot lay behind the ear what does not
        # get there. That arc per tucked guide: the last pinned node still within 2 mm of the route's highest pinned
        # point (the level run over the top ends where the descent behind the ear begins)
        pin = np.asarray(pin, bool)
        Zp = np.where(pin, np.asarray(Qp, np.float64)[..., 2], -np.inf)
        top = (Zp >= Zp.max(1, keepdims=True) - 0.002) & pin
        last = np.where(top.any(1), pin.shape[1] - 1 - np.argmax(top[:, ::-1], 1), 0)
        a2 = np.asarray(arcs_g, np.float64)[np.arange(len(pin)), last]
        need = (wt * a2[gi]).sum(1) / np.maximum(share, 1e-300) + 0.010
        rep["too_short_to_tuck"] = int((insec & (L_c < need)).sum())
        insec &= L_c >= need
    sel = np.nonzero(insec)[0]
    rep.update(laid=int(len(sel)), share_p10_50_90=np.percentile(share, [10, 50, 90]).round(3).tolist())
    if len(sel) == 0:
        return held, rep
    wt = wt[sel] / share[sel, None]
    gis = gi[sel]
    h = 0.0005
    S = float(L_c[sel].max()) * 1.6 + 0.01
    Qp = np.asarray(Qp, np.float64)
    T = _guide_table(Qp, np.asarray(arcs_g, np.float64)[:, -1], Qp[:, 0], arcs_g, h, S)   # displacements from the roots
    Mf = int(c["children_fine"])
    u = (np.arange(Mf) / (Mf - 1.0)) ** 1.5
    Ls = L_c[sel]
    s = (Ls * 1.6)[:, None] * u[None, :]
    A = np.zeros(s.shape + (3,))
    A0 = np.zeros((len(sel), 3))
    for a in range(k):
        g_ = gis[:, a]
        A += wt[:, a, None, None] * (_lookup(T, h, g_, s) + Qp[g_, 0][:, None, :])
        A0 += wt[:, a, None] * Qp[g_, 0]
    off = X_c[sel] - A0
    g = np.maximum(2.0 * np.linalg.norm(off, axis=1), 0.005)
    t = np.clip(s / g[:, None], 0.0, 1.0)
    C = A + (1.0 - t * t * (3.0 - 2.0 * t))[..., None] * off[:, None, :]
    sig = np.concatenate([np.zeros((len(sel), 1)), np.cumsum(np.linalg.norm(np.diff(C, axis=1), axis=2), 1)], 1)
    lam = np.minimum(Ls, sig[:, -1])
    held[sel] = _resample(C, sig, lam[:, None] * (np.arange(p) / (p - 1.0))[None, :]).astype(np.float32)
    rep["gather_mm_p10_50_90"] = np.percentile(1000 * np.linalg.norm(off, axis=1), [10, 50, 90]).round(1).tolist()
    return held, rep


def held_directions(held, L, m):
    """the m segment directions (n, m, 3) of a held shape: each polyline held (n, p, 3) m, root first, resampled at
    arcs (j / m) x its own length (at most L (n,) m) - the shape a short hair was held in while it dried"""
    held = np.asarray(held, np.float64)
    sig = np.concatenate([np.zeros((len(held), 1)), np.cumsum(np.linalg.norm(np.diff(held, axis=1), axis=2), 1)], 1)
    Lh = np.minimum(np.asarray(L, np.float64).ravel(), sig[:, -1])
    Q = _resample(held, sig, (np.arange(m + 1) / m)[None, :] * Lh[:, None])
    return _unit(np.diff(Q, axis=1) + 1e-15)


def short_hairs(X, d0, L, fibre, body=None, scale=1.0, cfg=None, log=None, canopy=None, held=None, set_r=None):
    """hairs shorter than grow.short_mm: each a single inextensible rod of m equal segments, clamped at its own
    follicle direction d0, at rest under bending + its own weight + body contact (no hair-hair term) + with canopy the
    long hair's layer lying over it (Canopy: a node out of the layer lifts the layer's weight over its share).
    canopy None (as before), a Canopy, or the settled guides (P, W, V, l) - generate passes
    canopy=(rods.positions(), rods.W, rods.V, rods.l) after the final settle.
    held (n, p, 3) m, root first: the shape each short hair was held in while its section dried - the brush drawn
    through the section carries the short hairs with the long ones on the stroke (generate passes the children of the
    guides as they were held, actions' P_held, cut to each short hair's length). With it the hair takes the set
    (rest turning set_r (n,) x its held turning: dry.set x retention, the fixative where finish.where says) and starts
    from the held shape, released. held None: no set, a straight start (as before, byte-identical). With held, a canopy
    and grow.short_friction (on), the long hair lying over the short hair holds it where the brush laid it: each held
    node under the layer is a static friction anchor of mu N, N the short hair's own share of the contacts carrying the
    layer's weight over its strip (grow.short_friction_load) plus the oil's liquid bridges to the long hairs
    (fibre.adhesive_load at canopy_phi), and with grow.short_friction_stiff the skin under a held node stiffened by
    mu N / onset; rep["friction"] reports it.

    X (n, 3) m roots; d0 (n, 3) unit follicle directions; L (n,) m lengths; fibre a fibre.Fibre (EI and weight with
    the taper; L is passed as the hair's grown length, so a new hair is thin at the root too); body a grid.SDFGrid
    (contact on settle.smooth_distance, C1) or anything with .sample(P, grad=True) -> (signed distance m, gradient)
    (None: no body); scale the per-hair radius factor. Each rod starts pushed out of the body (_ShortRods.start) and
    is converged on its own (_lbfgs_per_hair) to grow.short_tol. Returns (P (n, points, 3) float32 m, report)."""
    t0 = time.time()
    c = settings(cfg)
    m = int(c["short_segments"])
    npts = int(c["children_points"])
    X = np.asarray(X, np.float64)
    d0 = _unit(np.asarray(d0, np.float64))
    L = np.asarray(L, np.float64).ravel()
    n = len(X)
    sc = np.broadcast_to(np.asarray(scale, np.float64), (n,))
    if n == 0:
        return np.zeros((0, npts, 3), np.float32), dict(short=0)
    dist = _distance(body)
    cano = build_canopy(canopy, X, L, dist, cfg, log=log) if dist is not None else None
    rods = _ShortRods(X, d0, L, fibre, dist, sc, m, float(c["contact_mm"]) / 1000.0, float(c["contact_k"]),
                      float(c["contact_kb"]), float(c["contact_onset_mm"]) / 1000.0, float(c["short_kink_urad"]) * 1e-6,
                      canopy=cano, canopy_onset=float(c["canopy_onset_mm"]) / 1000.0,
                      held=None if held is None else held_directions(held, L, m), set_r=set_r,
                      fibre_fric=fibre if bool(c["short_friction"]) else None,
                      eps=float(c["short_friction_eps_mm"]) / 1000.0, phi=float(c["canopy_phi"]),
                      fload=str(c["short_friction_load"]), fstiff=bool(c["short_friction_stiff"]))
    x0, moved = rods.start()
    t_start = time.time() - t0
    x, _, lrep = _lbfgs_per_hair(rods, x0, float(c["short_tol"]), int(c["short_evals"]),
                                 refresh=int(c["short_refresh"]))
    relag = 0
    if rods.fA is not None:
        # the layer's hold is static friction: a node that slid past eps is in kinetic friction and its anchor trails
        # it (the settle's lag_stick), eps behind along the slip, in the skin's tangent plane where it now lies. Left
        # at its first anchor, a node that slid far is pulled back along the chord - through the curved skin (gr12: a
        # 21.9 mm slip). One re-anchor and a re-solve from where the hairs rest
        Pn = rods.core(np.arange(n), x, ratio=True)[3]
        dP = Pn - rods.fA
        dt = dP - (dP * rods.fn).sum(2, keepdims=True) * rods.fn
        sl = np.linalg.norm(dt, axis=2)
        mv = sl > rods.feps
        relag = int(mv.sum())
        if relag:
            rods.fA = np.where(mv[..., None], Pn - rods.feps * dt / np.maximum(sl, 1e-300)[..., None], rods.fA)
            _, gA = dist(rods.fA[mv])
            rods.fn[mv] = _unit(gA)
            x, _, lrep2 = _lbfgs_per_hair(rods, x, float(c["short_tol"]), int(c["short_evals"]),
                                          refresh=int(c["short_refresh"]))
            for k_ in ("balanced", "stalled", "moving"):
                lrep[k_] = lrep2[k_]
            lrep["iterations"] = lrep["iterations"] + lrep2["iterations"]
    _, _, ratio, Pn = rods.core(np.arange(n), x, ratio=True)
    C = np.concatenate([X[:, None], Pn], 1)
    sig = (np.arange(m + 1) / m)[None, :] * L[:, None]
    Q = _resample(C, sig, (np.arange(npts) / (npts - 1.0))[None, :] * L[:, None])
    rep = dict(short=int(n), start_pushes=moved, start_s=round(t_start, 3), **lrep,
               force_ratio_p50_p99_max=[float(np.percentile(ratio, 50)), float(np.percentile(ratio, 99)),
                                        float(ratio.max())],
               seconds=round(time.time() - t0, 3))
    rep["evals"] = rep["iterations"]
    if dist is not None:
        sd = dist(Pn.reshape(-1, 3))[0].reshape(n, m)
        far = sig[:, 1:] >= 0.002
        rep["min_clearance_mm"] = float(sd.min() * 1000.0)
        rep["min_clearance_beyond_2mm_mm"] = float(sd[far].min() * 1000.0) if far.any() else None
        rep["root_reading_mm_min_p50_max"] = [float(rods.sd_root.min() * 1e3), float(np.median(rods.sd_root) * 1e3),
                                              float(rods.sd_root.max() * 1e3)]
        rep["inside_body_nodes"] = int((sd < 0).sum())
        rep["below_root_reading_nodes"] = int((sd < np.minimum(rods.sd_root, 0.0)[:, None] - 1e-7).sum())
        rep["touching_share"] = float((sd - rods.hreq < 1e-4).any(1).mean())
        if rods.Dh is not None:
            # how far each hair sprang back from the held shape on release (its tip, mm) and the set's share
            Ph = X[:, None] + np.cumsum(rods.l[:, None, None] * rods.Dh, 1)
            sr = np.broadcast_to(np.asarray(1.0 if set_r is None else set_r, np.float64), (n,))
            rep["set"] = dict(r_min_max=[float(sr.min()), float(sr.max())],
                              tip_springback_mm_p50_90_max=[float(v) for v in np.percentile(
                                  np.linalg.norm(Pn[:, -1] - Ph[:, -1], axis=1) * 1e3, [50, 90, 100])])
        if rods.fA is not None:
            # the layer's hold: how many held nodes it anchors, their load over their own weight, how far they slid
            on = rods.fmuN > 0
            slip = np.linalg.norm(Pn - rods.fA, axis=2) * 1e3
            rep["friction"] = dict(mu=float(fibre.mu), eps_mm=rods.feps * 1e3, re_anchored_nodes=relag,
                                   anchored_nodes_share=float(on.mean()),
                                   anchored_hairs_share=float(on.any(1).mean()),
                                   load_over_weight_p50_p90=[float(v) for v in np.percentile(rods.fN_w[on], [50, 90])]
                                   if on.any() else None,
                                   load=str(c["short_friction_load"]), stiff=bool(c["short_friction_stiff"]),
                                   weight_part_over_weight_p50_p90=[float(v) for v in np.percentile(
                                       rods.fNw_w[on], [50, 90])] if on.any() else None,
                                   bridge_part_over_weight_p50_p90=[float(v) for v in np.percentile(
                                       rods.fNa_w[on], [50, 90])] if on.any() else None,
                                   contact_share_p10_50_90=[float(v) for v in np.percentile(rods.fshare, [10, 50, 90])],
                                   slip_mm_anchored_p50_90_max=[float(v) for v in np.percentile(slip[on], [50, 90, 100])]
                                   if on.any() else None)
        if cano is not None:
            pc, wa = canopy_depth(cano, dist, Pn.reshape(-1, 3))
            pc = pc.reshape(n, m)
            rep["canopy"] = dict(cano.report, outside_nodes_share=float((pc > rods.dlc).mean()),
                                 outside_hairs_share=float((pc > rods.dlc).any(1).mean()),
                                 outside_mm_p99_max=[float(np.percentile(pc, 99) * 1e3), float(pc.max() * 1e3)],
                                 spacing_mm_p50=float(np.median(rods.spacing) * 1e3))
        if hasattr(body, "A"):
            st = body.sample(Pn.reshape(-1, 3)).reshape(n, m)
            rep["min_clearance_trilinear_mm"] = float(st.min() * 1000.0)
            rep["inside_body_nodes_trilinear"] = int((st < 0).sum())
            rep["inside_body_nodes_trilinear_beyond_2mm"] = int((st[far] < 0).sum())
    if log:
        log("short hairs: %d in %.1f s, %d iterations (%d hair evaluations), balanced %d stalled %d moving %d, "
            "balance p99 %.2e max %.2e" % (n, rep["seconds"], rep["iterations"], rep["hair_evaluations"],
                                            rep["balanced"], rep["stalled"], rep["moving"],
                                            rep["force_ratio_p50_p99_max"][1], rep["force_ratio_p50_p99_max"][2]))
    return Q.astype(np.float32), rep


# follicular units (hair.md point 2): follicles sit in units of 1-4 (rarely 5) hairs, about 1 unit per mm2 on the
# Caucasian scalp; mean 1.55 hairs in a trichoscopy series, about 2.0 in surgical series. The hairs of one unit leave
# one opening together (within ~0.3 mm), close to parallel: the natural seed of a clump (hair_styling 7 row 32: "Bundle
# = a follicular unit or small group").
FU_DEFAULTS = {
    "fu_sizes": ([0.30, 0.45, 0.20, 0.05], None, "share of units holding 1, 2, 3, 4 hairs (mean 2.0, the surgical "
                                                 "series; hair.md point 2)"),
    "fu_reach_mm": (1.5, (0.3, 5.0), "a unit gathers its hairs from the roots within this (about 1 unit per mm2)"),
    "fu_open_mm": (0.15, (0.02, 0.5), "the hairs of a unit sit within this of the unit's opening (~0.3 mm across)"),
    "fu_spread_deg": (4.0, (0.0, 20.0), "the spread of the unit's hairs' directions about their shared follicle "
                                        "direction"),
}


def follicular_units(X, d0, seed, cfg=None):
    """group the children's own roots X (n, 3) m into follicular units of 1-4 (deterministic from seed): each unit's
    hairs are moved to within fu_open_mm of the unit's centre (on the scalp; the chord is well under a micron off it)
    and share the unit's follicle direction within fu_spread_deg. d0 (n, 3) unit vectors.
    Returns (X (n, 3), d0 (n, 3), unit (n,) int, report)."""
    c = {k: v[0] for k, v in FU_DEFAULTS.items()}
    c.update({k: v for k, v in (((cfg or {}).get("grow") or cfg or {}) if isinstance(cfg, dict) else {}).items()
              if k in FU_DEFAULTS})
    rng = np.random.default_rng(int(seed) + 71)
    X = np.asarray(X, np.float64)
    d0 = _unit(np.asarray(d0, np.float64))
    n = len(X)
    p = np.asarray(c["fu_sizes"], np.float64)
    p = p / p.sum()
    want = rng.choice(np.arange(1, len(p) + 1), size=n, p=p)
    kq = min(len(p) + 2, n)
    dd, nb = cKDTree(X).query(X, k=kq)
    reach = float(c["fu_reach_mm"]) / 1000.0
    unit = -np.ones(n, np.int64)
    nu = 0
    order = rng.permutation(n).tolist()
    nbl, ddl, wl = nb.tolist(), dd.tolist(), want.tolist()
    for i in order:
        if unit[i] >= 0:
            continue
        unit[i] = nu
        need = wl[i] - 1
        for jj, dj in zip(nbl[i][1:], ddl[i][1:]):
            if need <= 0 or dj > reach:
                break
            if unit[jj] < 0:
                unit[jj] = nu
                need -= 1
        nu += 1
    cnt = np.bincount(unit, minlength=nu).astype(np.float64)
    Cx = np.stack([np.bincount(unit, X[:, d], nu) for d in range(3)], 1) / cnt[:, None]
    Cd = _unit(np.stack([np.bincount(unit, d0[:, d], nu) for d in range(3)], 1))
    off = X - Cx[unit]
    ro = np.linalg.norm(off, axis=1, keepdims=True)
    rad = rng.uniform(0.3, 1.0, (n, 1)) * float(c["fu_open_mm"]) / 1000.0
    Xn = np.where(cnt[unit][:, None] > 1, Cx[unit] + off / np.maximum(ro, 1e-12) * np.minimum(ro, rad), X)
    sp = np.radians(float(c["fu_spread_deg"]))
    dn = Cd[unit] + rng.normal(0, sp, (n, 3))
    dn = _unit(dn - 0.0)
    dn = np.where(cnt[unit][:, None] > 1, dn, d0)
    hist = np.bincount(cnt.astype(np.int64), minlength=len(p) + 1)[1:]
    rep = {"units": int(nu), "hairs_per_unit_mean": round(float(n / max(nu, 1)), 3),
           "unit_size_counts_1_2_3_4": hist[:len(p)].tolist(),
           "units_per_mm2": None, "moved_um_p50_max": [round(float(np.median(np.linalg.norm(Xn - X, axis=1))) * 1e6, 1),
                                                     round(float(np.linalg.norm(Xn - X, axis=1).max()) * 1e6, 1)]}
    return Xn, dn, unit, rep
