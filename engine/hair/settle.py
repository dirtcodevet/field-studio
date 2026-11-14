"""Hair Studio settle (numpy + scipy, Apache-2.0): guides as elastic rods clamped at their follicles, grown by
continuation and settled by minimising one energy (DESIGN_v2 4.4-4.6, implementation steps 4 and 5).

A long hair is a stiff, inextensible elastic rod. It leaves its follicle at the follicle's angle, arcs over under its own
weight and lands on the skin or on the hair already there. A guide stands for the hairs rooted around it: it carries
their bending stiffness, their weight and the room they take, and all three thin toward the ends as the hairs end. The
shape comes from the balance of bending, gravity, contact with the body, hair-hair pressure and friction - never from a
height, a plateau, an outline or a packing target.

    Rods(X, N, d0, fibre, member_guide, member_L, member_scale=1, member_cut=None, seg_root_mm=0.5, seg_growth=1.2,
         seg_max_mm=14, sample_mm=5, min_tail_mm=1)          (None: the GROW_DEFAULTS value)
        .members(cut=None)  the guide arrays from the member hairs (again after a cut)
        .l, .s, .lbar (G, K) m; .B (G, K) N m2 at the joints; .W (G, K) N and .V (G, K) m3 per segment;
        .n (G, K) member hairs longer than each joint's arc; .D (G, K, 3) current directions; .rho (G, K, 3) plastic set
        .positions() -> nodes (G, K + 1, 3); .node_arcs() -> (G, K + 1)
    State(rods, body=None, cfg=None, pressure=True, friction=True, acts=None, splat=None, log=print)
        what one settle carries: the active segments, the samples, the lagged friction anchors, the history
    energy(x, state) -> (E, grad)      the scaled energy of the active segments and its gradient, x = Jacobi-scaled u
    grow(rods, body=None, acts=None, stages_mm=None, cfg=None, pressure=True, friction=True, log=print) -> State
    release(state, acts=None, log=print) -> State     acts off (but what acts.release() keeps), lagged relaxes, the
                                                      final gated stage
    report(state, frame=None) -> dict  energies by term, the gate, fibre fraction p50 / p95 by region, layer tops,
                                       the curtain, the follicle clamp, timing
    GROW_DEFAULTS, settings(cfg)       the `grow.*` block in style.py's format
    rest_waves(rods, waves, seed=0, phase=None, random_mm=0, section_mm=35, jitter=0.15) -> rods   the natural wave
        (grow.rod_waves; State does it from the style's waves.* when the rods have none): rods.wave, rods.wave_n0
    wave_natural(rods) -> (G, K, 3)    each joint's natural rest turning in its carried frame (t, n, b)
    transport_frames(rods, D=None) -> T, Nn, Bn (G, K, 3)    the frame carried from the follicle (parallel transport)
    bend_carried(D, d0, n0, loc, cb) -> (E, dE/dD, rho)       the bending of a hair type's helix (fibre.hair_type):
        the rest turning loc (t, n, b) read in the frame carried along the current D, gradient through the transport
        (State.hx_frame; other fibres keep the lagged rho below)
    wave_rest_shape(rods) -> (G, K + 1, 3)    the natural rest shape alone
    State.wave_frame()                 rho = natural + the acts' edits, in the frame carried along the strand now
    rewet(state, wet=1.0)              the set's share of rho x (1 - wet): fully wet hair is back to its natural wave
All positions in metres, forces in newtons, energies in joules; arguments named *_mm are millimetres.

The model (DESIGN_v2 4.5). Unknowns: one free 3-vector u per segment, direction d = u / |u|; node positions
x_{j+1} = x_j + l_j d_j from the root X, so the rod is inextensible by construction. The energy is

    sum_j B_j / (2 lbar_j) |d_j - d_{j-1} - rho_j|^2       bending; d_{-1} = d0 (the follicle clamp); rho = plastic set
  + sum_j W_j (z(mid_j) - z_root)                          gravity (W_j the segment's weight)
  + sum_y 1/2 k_c w_y (h_c - SDF(y))_+^2                   body contact on the samples y (grid.SDFGrid)
  + sum_v a_v h^3 c phi_v^3,  c = k E / 2                  hair-hair pressure P = k E phi^3 (grid.Splat)
  - sum_v a_v h^3 c_a phi_w,v^2                            the oil bridges' cohesion (fibre.cohesion_c0): phi_w the
                                                           fibre fraction weighted by each sample's wetting; with
                                                           uniform wetting w, P = 2 c phi^3 - c_a w^2 phi^2, zero at
                                                           phi* = c_a w^2 / (2 c). No oil: c_a = 0 (pressure only)
                                                           fibre.cohesion_count "aligned": phi_w splits by each
                                                           sample's local alignment A into a crossing share (the term
                                                           above on phi_x) and a line share phi_l, which adds
  - sum_v a_v h^3 sigma_0 phi_l,v ln(phi_v + eps)          the line bridges (form.py's lock law, P = -sigma_0 w phi)
  + sum_y mu N_y eps (sqrt(1 + |Pi_t (y - ybar_y)|^2 / eps^2) - 1)    lagged friction
  + E_acts                                                 the stylist's acts (the hook below)
  + sum_j B_j / (2 lbar_j) (|u_j| - 1)^2                   keeps |u| near 1 (conditioning only)

Gradient chain: forces on the samples go to the segment end points; the direction gradient is
g_d_j = l_j sum_{k >= j} g_x_{k+1} (a reverse cumulative sum); then g_u = (I - d d^T) g_d / |u|.
Solver: scipy L-BFGS-B in Jacobi-scaled variables x = s u, s^2 = ESC (2 B_j / lbar_j + 2 B_{j+1} / lbar_{j+1} +
l_j T_j / 2) with T_j the weight hanging from segment j on, and the energy scaled by ESC = 1 / (G x mean guide weight x
1 mm), so the scaled energy is per guide in guide-weight millimetres and its Hessian is about 1 on the diagonal. Every
solve has a fixed evaluation cap (the final one may stop earlier once the gate passes), so a seed always gives the same
result. Growth (4.6): the active arc grows stage by stage (30, 60, 100, 160, 240, 330, 430 mm, then the longest guide);
new segments start hanging, unit(d_prev - 0.6 z) (along the brush's path when the stroke is drawn through:
grow_along_stroke), pushed out of the body; after each stage the friction anchors and
normal forces are stored again (lagged friction).

Where DESIGN_v2 left a choice open, the more physical option was taken (also in the notes of GROW_DEFAULTS):
- lbar at the root joint is l_0 / 2, not l_0: d0 is the hair's tangent at the skin and a segment's direction is the
  tangent at its middle, so the turn between them spans half a segment. With l_0 the clamp would be twice too soft.
- Samples: n_s = max(2, ceil(l / sample_mm)) per segment at (k + 1/2) / n_s (the design's 1/4 and 3/4 when n_s = 2).
  Two samples on a 14 mm segment are 7 mm apart, while the splat kernel (trilinear + [1, 2, 1] on a 3 mm grid) has an
  sd of 2.45 mm per axis: the line density of a hanging guide would ripple by about 18 % and the pressure (phi^3) by
  about 50 % along it, a false bumpy force. At 5 mm the ripple is under 1 %. A hair is continuous.
- Segment weight and fibre volume are exact integrals over the members (fibre.Fibre.hang with the taper and the cut),
  not midpoint values. The last segment of a guide ends at the guide's own length (a tail under min_tail_mm joins the
  segment before it), so no weight hangs beyond the longest hair.
- A cut member ends blunt at its cut length (scissors remove the taper); before the cut every member is at its grown
  length.
- Gravity is measured from each guide's root height. Energy differences are unchanged; the gate's relative energy
  change is then relative to the hair's own potential energy, not to where the world's origin happens to be.
- The gate's force is the unbalanced force at each node once the rod's tension has taken its share: the transverse
  force carried by segment j is tau_j = (I - d_j d_j^T) g_d_j / l_j, and the node between segments j - 1 and j is out of
  balance by tau_{j-1} - tau_j (the tip by tau_last). It is divided by the node's local weight (half of each adjacent
  segment's) - the design's "max |force| / local segment weight" - or, when larger, by the sum of the magnitudes of
  each term's force on the node (bending, gravity, contact, pressure, friction): the forces it actually balances. At
  the graded root the follicle moment, the tension and the scalp contact each pass ~25 local weights through a node,
  and a balance to 0.05 local weight there is under the float64 resolution of a head's energy. In a freely hanging
  chain the sum is the local weight. The root node is held by the follicle.
- Friction: N_y = |force of the body and the hair-hair pressure on y| as the design writes it (+ the oil bridges'
  load); its coefficient is the interface's that carries the load - fibre.friction_skin (~0.5, Derler & Gerhardt
  2012) for the share the body's contact carries, fibre.friction (hair on hair, ~0.25) for the rest.

The acts hook (actions.py comes next). `acts` is an object; settle calls only the methods it has:
    acts.energy(view) -> (E (J), {"Y": dE/dY (n, 3), "P": dE/dP (G, Ka + 1, 3), "D": dE/dD (G, Ka, 3)})
                         any key may be missing or None. Y are the samples, P the nodes (P[:, 0] are the roots and are
                         held), D the unit segment directions (a brush on the heading acts on D)
    acts.stage(view, index, arc_m)  at the start of each growth stage (switch acts on or off by stage)
    acts.lag(view)       after each sub-solve: update lagged data (brush headings, pin targets)
    acts.precond(view) -> (G, Ka) extra s^2 in J (the act's stiffness on each segment's direction), or None
    acts.release()       switch every act off except those that stay (tie pins)
`view` (State.view()) carries P, D, Y, sample_seg (flat index into G x Ka), sample_t, sample_w (N), sample_v (m3),
sample_arc (m), act, l, s0, B, lbar, W (G, Ka), Ka, the rods and the state.

Sources: the rod statics of Bertails et al. 2005 and Bergou et al. 2008 (discrete elastic rods, reduced coordinates);
the fibre-mass law P = k E phi^3 (Van Wyk 1946; Goldstein, Warren & Ball 2012, PRL 108:078101); level-set contact
(Bridson, Marino & Fedkiw 2003); lagged friction as in DESIGN_v2's D1 prototype; L-BFGS-B (Byrd, Lu, Nocedal & Zhu
1995) via scipy.
"""
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import minimize

from . import fibre as FB

# The `grow.*` block in style.py's format: (default, (lo, hi) or None, note). DESIGN_v2 section 7 plus the settle's own
# numerical settings.
GROW_DEFAULTS = {
    "guides": (3000, (50, 20000), "guide rods standing for the hairs (DESIGN_v2 4.4); about 48 hairs each at 145k"),
    "seg_root_mm": (0.5, (0.1, 10.0), "segment length at the root; graded x seg_growth per segment up to seg_max_mm. "
                                      "DESIGN_v2 gives 3 mm, but a long hair's root bend under its weight is 0.05-0.14 "
                                      "/mm (measured: single rods, 250 guides on a head), so a 3 mm first segment turns "
                                      "4-12 deg off d0 although the hair's tangent at the skin is d0 exactly, and the "
                                      "design's own check (first segment within 2 deg) cannot hold. Graded x1.2, 0.5 mm "
                                      "costs 47 segments for 480 mm instead of 38"),
    "seg_growth": (1.2, (1.0, 2.0), "segment length ratio from one segment to the next"),
    "seg_max_mm": (14.0, (1.0, 40.0), "longest segment (the gravity bending length is 37-50 mm; a curly type's coil "
                                      "needs <= 1.8 mm, ~15 per 27 mm turn - test_hair_type)"),
    "min_tail_mm": (1.0, (0.0, 10.0), "a guide's last segment shorter than this joins the one before it"),
    "sample_mm": (5.0, (1.0, 20.0), "contact and pressure samples at most this far apart along a segment (at least 2 per "
                                    "segment): the splat kernel's sd on a 3 mm grid is 2.45 mm, so 7 mm spacing would "
                                    "ripple the line density ~18 % and the pressure ~50 %; 5 mm keeps it under 1 %"),
    "stages_mm": ([30.0, 60.0, 100.0, 160.0, 240.0, 330.0, 430.0], None, "growth by continuation: the active arc ends "
                                                                         "here stage by stage, then at the longest guide "
                                                                         "(DESIGN_v2 4.6)"),
    "sideburn": ("own", None, "grow mode's sideburn lengths (DESIGN_sections 2): 'own' the roots in front of an ear "
                              "and below its top grow to their own lengths (round 17); 'short' they are short edge "
                              "hairs (hairline.edge_short_mm) as gr18 had them - the reference's preauricular hair is "
                              "short and fine, lying down and back, ending at the ear's front edge"),
    "evals_stage": (40, (2, 2000), "energy evaluations per growth stage (split over `subsolves`)"),
    "subsolves": (2, (1, 10), "solves per growth stage; the acts' lagged data are updated between them"),
    "evals_release": (135, (0, 5000), "evaluations of the release (split over release_solves, friction lagged between)"),
    "release_solves": (3, (1, 10), "solves of the release"),
    "evals_final": (4000, (10, 20000), "cap of the final, gated stage; it stops as soon as the gate passes. DESIGN_v2 "
                                       "gives 300 and says to raise it before changing any physics (section 12): 250 "
                                       "guides on a head needed ~1200; ~700 guides on a head and torso with the acts "
                                       "(2026-10-30) needed 2690 and failed at 2500 (max 0.65). The last nodes to "
                                       "balance were free hair ends hanging off the back, 6-30 mm from the body, "
                                       "whose last segment is the cut's 1-3 mm stub (~1/30000 of the guide's "
                                       "weight): they settle with more evaluations, no physics was missing. A run "
                                       "that passes earlier stops earlier and is unchanged"),
    "final_chunk": (300, (25, 5000), "the final stage's scaling is rebuilt every this many evaluations"),
    "grid_mm": (3.0, (1.0, 10.0), "the body's signed-distance grid and the fibre splat lattice"),
    "splat_mm": ("grid", ["grid", "auto"], "the pressure splat's lattice (TRANSLATION row 6, piece 4). 'grid': grid_mm, "
                                           "today's kernel (sd 0.866 x 3 = 2.6 mm), every earlier run bit-identical. "
                                           "'auto': h = max(grid_mm, splat_sd_frac x S / 0.866), S the guide spacing, "
                                           "so the kernel's sd is at least splat_sd_frac x S and the field is a mass, "
                                           "not a set of tubes. The body's SDF stays at grid_mm"),
    "splat_sd_frac": (0.25, (0.1, 2.0), "'auto': the kernel's sd as a share of the guide spacing S. Measured on the "
                                        "flat patch (tests/test_pressure_scale.py, 25 / 100 / 400 guides, k 1e-5 and "
                                        "2.6e-4): 0.25 puts the layer at 0.94-1.05 H at every count; 0.5 (the design's "
                                        "'half S') over-smooths across the layer too (the kernel is isotropic, sd 10 mm "
                                        "against H 22 mm) and the layer sinks to 0.53-0.62 H"),
    "splat_spacing_mm": (0.0, (0.0, 100.0), "'auto': the guide spacing S (mm); 0 = the median nearest-neighbour "
                                            "distance of the guide roots (S exactly on a square lattice; preflight "
                                            "uses sqrt(scalp area / guides))"),
    "contact_mm": (0.5, (0.0, 5.0), "contact distance h_c: hair rests this far off the skin (penalty below it)"),
    "contact_k": (3e5, (1e3, 1e9), "contact stiffness per metre of penetration, x the sample's weight: 1 mm inside "
                                   "pushes back with 300 sample weights. DESIGN_v2 gives 2e6; that is 1e3-1e5 times the "
                                   "hair's own tension stiffness in the same directions and L-BFGS needed ~10k "
                                   "evaluations for 250 guides (measured). At 3e5 a sample sinks 3 um per weight it "
                                   "carries; the deepest on a head (hair drawn tight over the ear rim) was 0.6 mm past "
                                   "contact_mm, against 0.25 mm at 2e6 and 1.6 mm at 1e5"),
    "contact_c1": (True, None, "contact on a quadratic B-spline of the body grid (and of its fine crops, blended as the "
                               "grid blends them): its gradient is continuous. The trilinear grid's gradient jumps at "
                               "every 3 mm cell face; under a hair drawn tight over the body that kink holds a node "
                               "out of balance (measured 0.1 x its weight where nothing else is left) - a catch the "
                               "smooth skin does not have. The B-spline rounds the surface by ~h^2 / 8 x curvature: "
                               "0.01 mm on the skull, ~0.3 mm on an ear rim of 2 mm radius (2 mm crop)"),
    "contact_ear_k": (1.0, (1.0, 100.0), "the skin's contact stiffness x this in the ears' fine crops (the grid's ear "
                                         "boxes, blended in by the crops' own smoothstep weight): the ear is "
                                         "cartilage, not give. Measured 11-01 (buildHS3 README): at contact_k alone a "
                                         "guide bent over the 3-5 mm rim sits ~0.9 mm (to 2.4 mm) inside it - its own "
                                         "stiffness (29-110 node weights) against the contact - and ~70 guides a "
                                         "side passed through the ear. The ear's shape is unchanged: the groove "
                                         "behind it stays open, so hair can still be tucked there"),
    "contact_margin_mm": (1.5, (0.5, 20.0), "the smooth distance is evaluated only where the grid's own is within this "
                                            "of contact_mm (beyond it there is no contact either way)"),
    "friction_eps_mm": (1.0, (0.01, 10.0), "lagged friction is a smoothed Coulomb law: linear within this slip, mu N "
                                           "beyond (DESIGN_v2 4.5)"),
    "friction_reach_mm": (12.0, (0.0, 100.0), "the oil's adhesive load joins a sample's friction only within this of "
                                              "the body: there the hair rests on the skin or on the layer lying on it, "
                                              "which the body holds still. The lagged anchors are fixed in space, so "
                                              "hair hanging free (whose neighbours swing with it) must not be held "
                                              "against them. 12 mm: the layer on the head (crown p90 12.1 mm, round 4)"),
    "hang_drop": (0.6, (0.0, 5.0), "new segments start along unit(d_prev - hang_drop z) (DESIGN_v2 4.6)"),
    "grow_along_stroke": (True, None, "with the stroke drawn through (actions brush.through, brush on): new segments "
                                      "start along the brush path's heading at their arc, not hanging - the hair is "
                                      "in the bristles as the brush is drawn from the root to the ends, so the length "
                                      "that grows goes where the stroke carries it. Hung instead, a length that drops "
                                      "in front of the shoulder cannot be pulled back behind it through the body "
                                      "(gr17 / pA0: every guide ending in front was there before the hold)"),
    "rod_waves": (False, None, "the hair's natural wave as the rods' rest turning (DESIGN_v2 5, waves phase 2): each "
                               "joint's rest turning is a rotation by kappa(s) lbar in a frame carried from the "
                               "follicle along the guide (parallel transport, refreshed every sub-solve), kappa(s) = A "
                               "(2 pi / lambda)^2 sin(2 pi s / lambda + the section's phase) from the style's waves.* "
                               "(amp_mm the free amplitude). Off: every earlier run unchanged"),
    "wave_random_mm": (0.0, (0.0, 0.2), "rms (1/mm) of a smooth random rest curvature on top of the coherent wave, in "
                                        "both directions across the fibre. DESIGN_v2 names fibre.kappa0_mm here, but "
                                        "Lasisi's kappa0 (0.04 /mm) is the fibre's WHOLE curvature, which a 95 mm wave "
                                        "of 7-11 mm already carries (A k^2 / sqrt 2 = 0.02-0.035 /mm); adding kappa0 "
                                        "again would count it twice, so the extra is 0 unless a look asks for it"),
    "maxcor": (30, (3, 60), "L-BFGS memory pairs. DESIGN_v2 gives 10-20; the slow modes left at the end (a guide sliding "
                            "round the head, a tip swinging below the shoulder) converged in about half the evaluations "
                            "with 40 (measured, 300 guides)"),
    "precond": ("banded", ["banded", "jacobi"], "the solver's variables. jacobi: x = s u with DESIGN_v2's diagonal "
                                                "(bending + tension); banded: x = R u with R^T R each guide's "
                                                "stiffness in node space - contact, tension from the force balance, "
                                                "bending, friction - a block-banded matrix factored per guide. In "
                                                "direction variables one contact couples every segment upstream of "
                                                "it, so the diagonal alone needed ~10k evaluations on a head"),
    "gate": (0.05, (1e-4, 1.0), "converged when max |unbalanced node force| / max(local segment weight, sum of the "
                                "magnitudes of each term's force on the node) is under this ..."),
    "gate_de": (1e-4, (1e-9, 1e-1), "... and the relative energy change over the last 20 evaluations is under this"),
    "children_k": (4, (1, 16), "children: nearest guides on the same part side (children.py)"),
    "root_blend_mm": (15.0, (0.0, 100.0), "children: their own follicle direction blends in over this arc (children.py)"),
    "short_mm": (80.0, (0.0, 400.0), "children: hairs shorter than this are single-rod sweeps (children.py)"),
}

# The stand-ins of the legacy walk (DESIGN_v2 9.A2): none of them exists in this module.
STAND_INS_USED = []

_Z = np.array([0.0, 0.0, 1.0])


def _plain(node):
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    return copy.deepcopy(node[0])


GROW = _plain(GROW_DEFAULTS)


def settings(cfg=None):
    """the grow values: GROW_DEFAULTS <- cfg (a whole style, whose 'grow' block is used, or the block itself)
    -> (values dict, warnings)"""
    cfg = cfg or {}
    blk = cfg["grow"] if isinstance(cfg.get("grow"), dict) else (cfg if set(cfg) & set(GROW_DEFAULTS) else {})
    out = copy.deepcopy(GROW)
    warn = []
    for k, v in blk.items():
        if k not in out:
            warn.append("unknown key grow.%s" % k)
        else:
            out[k] = copy.deepcopy(v)
    return out, warn


def _unit(v):
    return v / np.maximum(np.sqrt((v * v).sum(-1, keepdims=True)), 1e-300)


def wpercentile(x, w, q):
    """weighted percentiles q (0-100) of x with weights w (the value a typical unit of weight sees)"""
    x = np.asarray(x, np.float64).ravel()
    w = np.asarray(w, np.float64).ravel()
    if len(x) == 0 or w.sum() <= 0:
        return [float("nan")] * len(np.atleast_1d(q))
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return [float(np.interp(qq / 100.0, c, x)) for qq in np.atleast_1d(q)]


# ---------------------------------------------------------------- a body distance with a continuous gradient
def _bspline2(A, lo, h, P):
    """quadratic B-spline of lattice A (first node lo, spacing h) at points P inside it: value and exact gradient. C1
    (the weights of the three nearest nodes per axis are 0.5 (0.5 - u)^2, 0.75 - u^2, 0.5 (0.5 + u)^2, u = the offset
    from the nearest node); it averages the nodes over about +-0.5 h, so it does not interpolate them"""
    shp = np.array(A.shape)
    f = (P - lo) / h
    i = np.rint(f).astype(np.int64)
    u = f - i
    wts = (0.5 * (0.5 - u) ** 2, 0.75 - u * u, 0.5 * (0.5 + u) ** 2)
    dws = (-(0.5 - u), -2.0 * u, 0.5 + u)
    Af = A.ravel()
    sx, sy = int(shp[1] * shp[2]), int(shp[2])
    v = np.zeros(len(P))
    g = np.zeros((len(P), 3))
    ix = [np.clip(i[:, 0] + a, 0, shp[0] - 1) * sx for a in (-1, 0, 1)]
    iy = [np.clip(i[:, 1] + a, 0, shp[1] - 1) * sy for a in (-1, 0, 1)]
    iz = [np.clip(i[:, 2] + a, 0, shp[2] - 1) for a in (-1, 0, 1)]
    for a in range(3):
        for b in range(3):
            base = ix[a] + iy[b]
            wab = wts[a][:, 0] * wts[b][:, 1]
            dxab = dws[a][:, 0] * wts[b][:, 1]
            dyab = wts[a][:, 0] * dws[b][:, 1]
            for c in range(3):
                val = Af[base + iz[c]].astype(np.float64)
                v += val * wab * wts[c][:, 2]
                g[:, 0] += val * dxab * wts[c][:, 2]
                g[:, 1] += val * dyab * wts[c][:, 2]
                g[:, 2] += val * wab * dws[c][:, 2]
    return v, g / h


def smooth_distance(grid, P):
    """the body's signed distance (m) and its gradient at P (n, 3) from a grid.SDFGrid, C1: the B-spline of the
    coarse lattice, then each fine crop's B-spline blended in with the grid's own smoothstep weights at the box faces;
    outside the lattice the box-edge value plus the distance out, as SDFGrid.sample does"""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    Pc = np.clip(P, grid.lo, grid.hi)
    o = P - Pc
    do = np.sqrt((o * o).sum(1))
    v, g = _bspline2(grid.A, grid.lo, grid.h, Pc)
    out = do > 0
    if out.any():
        v[out] += do[out]
        g[out] = g[out] * (o[out] == 0) + o[out] / do[out, None]
    for fb in getattr(grid, "fine", ()):
        inside = np.nonzero(np.all((P > fb.lo) & (P < fb.hi), 1))[0]
        if len(inside) == 0:
            continue
        Pi = P[inside]
        dlo, dhi = Pi - fb.lo, fb.hi - Pi
        t = np.clip(np.minimum(dlo, dhi) / fb.blend, 0, 1)
        sm = t * t * (3 - 2 * t)
        w = sm.prod(1)
        fv, fg = _bspline2(fb.A, fb.lo, fb.h, Pi)
        ds = np.where((t > 0) & (t < 1), 6 * t * (1 - t) / fb.blend, 0.0) * np.where(dlo < dhi, 1.0, -1.0)
        gw = np.stack([ds[:, 0] * sm[:, 1] * sm[:, 2], sm[:, 0] * ds[:, 1] * sm[:, 2], sm[:, 0] * sm[:, 1] * ds[:, 2]], 1)
        v0 = v[inside]
        g[inside] = (1 - w)[:, None] * g[inside] + w[:, None] * fg + (fv - v0)[:, None] * gw
        v[inside] = (1 - w) * v0 + w * fv
    return v, g


def fine_weight(grid, P):
    """the weight of the body grid's fine crops (the ears' boxes) at P (n, 3) and its gradient: 0 outside every crop,
    1 well inside one, the grid's own smoothstep over each crop's blend at its faces (as smooth_distance blends the
    crops in) -> (w (n,), gw (n, 3))"""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    w = np.zeros(len(P))
    gw = np.zeros((len(P), 3))
    for fb in getattr(grid, "fine", ()):
        inside = np.nonzero(np.all((P > fb.lo) & (P < fb.hi), 1))[0]
        if len(inside) == 0:
            continue
        Pi = P[inside]
        dlo, dhi = Pi - fb.lo, fb.hi - Pi
        t = np.clip(np.minimum(dlo, dhi) / fb.blend, 0, 1)
        sm = t * t * (3 - 2 * t)
        ds = np.where((t > 0) & (t < 1), 6 * t * (1 - t) / fb.blend, 0.0) * np.where(dlo < dhi, 1.0, -1.0)
        w[inside] = sm.prod(1)
        gw[inside] = np.stack([ds[:, 0] * sm[:, 1] * sm[:, 2], sm[:, 0] * ds[:, 1] * sm[:, 2],
                               sm[:, 0] * sm[:, 1] * ds[:, 2]], 1)
    return w, gw


# ---------------------------------------------------------------- a fibre splat with a continuous gradient
class SplatC1:
    """the hair-hair pressure on a quadratic B-spline splat (then grid's [1, 2, 1] blur), in float64.

    Why not grid.Splat: its trilinear weights have a gradient that jumps at every lattice cell face, so the pressure
    force on a sample jumps as it crosses one (3 c phi^2 x v / h). With the pressure on, a node resting on a face is held
    out of balance by that jump and L-BFGS-B's line search stalls (measured on the 90 mm sphere: max force / local
    weight 0.355, energy change 0 over 20 evaluations, ABNORMAL line search). The quadratic B-spline's weights
    0.5 (0.5 - u)^2, 0.75 - u^2, 0.5 (0.5 + u)^2 (u the offset from the nearest node) are C1, so the energy has a
    continuous gradient, as the contact's smooth_distance has. float64 because float32 rounding of the field (1e-7
    relative) is above the energy decrease L-BFGS asks for near convergence.
    The kernel's variance per axis is h^2 / 4 + h^2 / 2 (the blur) = 0.75 h^2, sd 2.6 mm on a 3 mm lattice (trilinear +
    blur: 2.45 mm), so the sample spacing note of sample_mm still holds. Same lattice, free-volume fraction a_v and
    energy as grid.Splat.pressure with free_weight (E = sum a_v h^3 c phi^3, P = 2 c phi^3 = k E phi^3).

    Speed: the 27 node indices and weights of every sample are built once per field() as (27, n) arrays (node offset
    first, so np.bincount adds in the same order as the per-offset loop it replaced: the field is bit-identical), the
    scatter is one bincount and the adjoint one fancy-index gather of those indices, contracted axis by axis
    (z, then y, then x) with the per-axis weights and their derivatives. A separable per-axis splat was not used: the
    samples are scattered, so the z weights cannot be applied after the x-y scatter (each sample's x-y weights differ),
    and the 27-point form is already one pass of numpy per step. The (27, n) index and weight arrays (40 MB each at
    186k samples) are kept and refilled while n is unchanged: a fresh 40 MB array costs about as much again in page
    faults as filling it (the kept work arrays are about 125 MB at 186k samples). The blur defers its three 1/4
    factors to one 1/64 at the end (powers of two: exact, the same bits as grid.blur121) and works in two kept crop
    buffers. a_v h^3 over the crop is cached while the crop is unchanged. Measured on a head-sized cloud of 186k
    samples (crop 79 x 79 x 164): 0.14-0.23 s per pressure evaluation (field + energy + adjoint) in the loop form,
    0.06-0.09 s here, the ranges being the machine's load (tests/test_settle.py splat)."""

    # node offsets (-1, 0, 1) per axis, x slowest: the order of the 27 rows
    _OFF = np.indices((3, 3, 3)).reshape(3, 27).T - 1
    SD_PER_CELL = float(np.sqrt(0.75))         # the kernel's sd per lattice spacing (B-spline h^2/4 + blur h^2/2)

    def __init__(self, grid=None, mm=None):
        from .grid import Splat
        # with a grid, mm (when given and not the grid's) is the splat's own spacing: grid.Splat keeps the grid's
        # first node and samples its free-volume fraction a_v once at that spacing
        self.base = Splat(grid, mm=mm, dtype=np.float64)
        self.h, self.lo, self.vox = self.base.h, self.base.lo, self.base.vox
        self.c0 = self.shape = None
        self._st = None
        self._av = None                        # (crop key, a_v h^3) of the last crop
        self._nb = {}                          # kept work arrays by name, refilled while their shape holds

    def _buf(self, name, shape, dtype=np.float64):
        a = self._nb.get(name)
        if a is None or a.shape != shape or a.dtype != dtype:
            a = self._nb[name] = np.empty(shape, dtype)
        return a

    _NT = max(1, int(os.environ.get("HAIR_BLUR_THREADS", "4")))
    _POOL = None
    _SLAB = 2                                  # planes per block: 2 x 88 x 214 doubles ~ 300 kB, cache-resident

    @staticmethod
    def _blur(B, C, D):
        """grid.blur121 of B, bit for bit, into C (returned), D a second crop-shaped work array; B is not changed.
        [1, 2, 1] along each axis with the 1/4 factors left to one exact 1/64 (folded into the last axis' pass).
        Each axis' three ops run block by block (_SLAB planes across an axis the pass does not run along), so a block
        is read once from memory and stays in cache for its three ops, and the blocks go to HAIR_BLUR_THREADS threads
        (numpy releases the GIL); every node gets the same operations in the same order: bit-identical to the plain
        whole-array passes (measured on a 97 x 88 x 214 crop: 24 -> 15.6 ms one thread, 8 ms four, both identical)."""
        src = B
        outs = (C, D, C)
        nt, slab = SplatC1._NT, SplatC1._SLAB
        if nt > 1 and SplatC1._POOL is None:
            SplatC1._POOL = ThreadPoolExecutor(nt)
        for ax in range(3):
            out = outs[ax]
            sax = 1 if ax == 0 else 0
            n = out.shape[sax]
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax], b[ax] = slice(1, None), slice(None, -1)
            a, b = tuple(a), tuple(b)

            def work(r, src=src, out=out, ax=ax, sax=sax, a=a, b=b):
                for s0 in range(r[0], r[1], slab):
                    sl = [slice(None)] * 3
                    sl[sax] = slice(s0, min(s0 + slab, r[1]))
                    sl = tuple(sl)
                    so, oo = src[sl], out[sl]
                    np.multiply(so, 2.0, out=oo)
                    np.add(oo[a], so[b], out=oo[a])
                    np.add(oo[b], so[a], out=oo[b])
                    if ax == 2:
                        oo *= 1.0 / 64.0
            if nt == 1 or n < 2 * nt:
                work((0, n))
            else:
                cuts = np.linspace(0, n, nt + 1).astype(int)
                list(SplatC1._POOL.map(work, [(cuts[i], cuts[i + 1]) for i in range(nt)]))
            src = out
        return C

    def field(self, Y, v):
        Y = np.asarray(Y, np.float64).reshape(-1, 3)
        v = np.broadcast_to(np.asarray(v, np.float64), (len(Y),))
        n = len(Y)
        f = (Y - self.lo) / self.h
        i = np.rint(f).astype(np.int64)
        u = f - i
        if n:
            c0 = i.min(0) - 2                  # the kernel reaches one node, the blur one more
            cs = i.max(0) + 3 - c0
        else:
            c0, cs = np.zeros(3, np.int64), np.full(3, 5, np.int64)
        shape = tuple(int(s) for s in cs)
        sx, sy = shape[1] * shape[2], shape[2]
        li = i - c0
        b = li[:, 0] * sx + li[:, 1] * sy + li[:, 2]
        # per-axis weights and their derivatives d/du, (3 offsets, n) each, contiguous along the samples
        w = [np.stack([0.5 * (0.5 - u[:, k]) ** 2, 0.75 - u[:, k] * u[:, k], 0.5 * (0.5 + u[:, k]) ** 2], 0)
             for k in range(3)]
        dw = [np.stack([-(0.5 - u[:, k]), -2.0 * u[:, k], 0.5 + u[:, k]], 0) for k in range(3)]
        off = self._OFF @ np.array([sx, sy, 1], np.int64)                   # (27,) linear node offsets
        lin = np.add(b[None, :], off[:, None], out=self._buf("lin", (27, n), np.int64))
        # (w_x w_y v) w_z, multiplied in the order of the loop this replaced
        wxy = np.multiply(w[0][:, None, :], w[1][None, :, :], out=self._buf("wxy", (3, 3, n)))
        np.multiply(wxy, v, out=wxy)
        W = np.multiply(wxy[:, :, None, :], w[2][None, None, :, :], out=self._buf("W", (3, 3, 3, n)))
        S = np.bincount(lin.ravel(), weights=W.ravel(), minlength=sx * shape[0]).reshape(shape)
        F = self._blur(S, np.empty(shape), self._buf("D", shape))
        self._st = (lin, w, dw, v)
        self.c0, self.shape = c0, shape
        self.base.c0, self.base.shape = self.c0, self.shape
        return F

    def free(self):
        return self.base.free()

    def phi(self, F):
        return F / (self.free() * self.vox)

    def _avox(self):
        """a_v h^3 on the crop of the last field(), reused while the crop is unchanged"""
        k = (tuple(int(c) for c in self.c0), self.shape)
        if self._av is None or self._av[0] != k:
            self._av = (k, self.free() * self.vox)
        return self._av[1]

    def adjoint(self, Gf, vw=None):
        """dE/dY (n, 3) from dE/dF on the crop of the last field(); vw: the per-sample weights the field was splatted
        with, when not the field's own v (the wetted field of cohesive())"""
        lin, w, dw, v = self._st
        v = v if vw is None else vw
        Gf = np.asarray(Gf, np.float64)
        H = self._blur(Gf, self._buf("C", Gf.shape), self._buf("D", Gf.shape)).ravel()
        n = lin.shape[1]
        Hn = H[lin].reshape(3, 3, 3, n)                                     # (x, y, z offsets, n)
        wx, wy, wz = w
        dx, dy, dz = dw
        # contract z with w_z and dw_z, then y, then x (einsum: one pass each, no temporaries)
        Tz = np.einsum("abcn,cn->abn", Hn, wz, out=self._buf("wxy", (3, 3, n)))      # (3, 3, n)
        Td = np.einsum("abcn,cn->abn", Hn, dz, out=self._buf("Td", (3, 3, n)))
        g = np.empty((n, 3))
        np.einsum("abn,an,bn->n", Tz, dx, wy, out=g[:, 0])
        np.einsum("abn,an,bn->n", Tz, wx, dy, out=g[:, 1])
        np.einsum("abn,an,bn->n", Td, wx, wy, out=g[:, 2])
        g *= (v / self.h)[:, None]
        return g

    def adjoint_multi(self, pairs):
        """sum over (dE/dF_k, vw_k) of adjoint(dE/dF_k, vw_k), for fields splatted with the same node indices and
        weights but per-sample weights vw_k: the blurred dE/dF_k are stacked and gathered in ONE fancy index, folded
        with their vw_k, and contracted once (the round-12 cost note: one gather and one contraction, not k)"""
        lin, w, dw, _ = self._st
        n = lin.shape[1]
        shp = tuple(np.shape(pairs[0][0]))
        Hs = self._buf("Hs", (len(pairs),) + shp)                        # each blurred straight into its kept row
        for k, (G, _) in enumerate(pairs):
            self._blur(np.asarray(G, np.float64), Hs[k], self._buf("D", shp))
        Hk = Hs.reshape(len(pairs), -1)[:, lin]                             # (k, 27, n): one gather
        Vk = np.stack([np.broadcast_to(np.asarray(vw, np.float64), (n,)) for _, vw in pairs], 0)
        Hn = np.einsum("kjn,kn->jn", Hk, Vk).reshape(3, 3, 3, n)
        wx, wy, wz = w
        dx, dy, dz = dw
        Tz = np.einsum("abcn,cn->abn", Hn, wz)
        Td = np.einsum("abcn,cn->abn", Hn, dz)
        g = np.empty((n, 3))
        np.einsum("abn,an,bn->n", Tz, dx, wy, out=g[:, 0])
        np.einsum("abn,an,bn->n", Tz, wx, dy, out=g[:, 1])
        np.einsum("abn,an,bn->n", Td, wx, wy, out=g[:, 2])
        g *= 1.0 / self.h
        return g

    def gather(self, A):
        """A (a field on the crop of the last field()) at the samples, with the same B-spline weights the splat used"""
        lin, w, _, _ = self._st
        n = lin.shape[1]
        An = np.asarray(A, np.float64).ravel()[lin].reshape(3, 3, 3, n)
        return np.einsum("abcn,an,bn,cn->n", An, w[0], w[1], w[2])

    def _subfield(self, F, wk):
        """a second field on the last field()'s indices and weights, the per-sample weights scaled by wk"""
        lin = self._st[0]
        W = self._nb["W"]
        Ww = np.multiply(W, np.asarray(wk, np.float64), out=self._buf("Ww", W.shape))
        Sw = np.bincount(lin.ravel(), weights=Ww.ravel(), minlength=F.size).reshape(F.shape)
        return self._blur(Sw, self._buf("Fw", F.shape), self._buf("D", F.shape))   # kept: read before the next call

    def alignment(self, Y, v, d):
        """the local alignment of the fibre at each sample: A = (3 tr(Q^2) - 1) / 2 clipped to [0, 1], Q the splatted
        direction tensor sum v d d^T over its trace (the nematic order parameter's invariant form: 0 isotropic, 1 all
        parallel; a flat random mat reads 0.25). Sign-free in d, as a contact line is"""
        F = self.field(Y, v)
        d = np.asarray(d, np.float64)
        Q = np.empty((len(Y), 6))
        for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
            Q[:, k] = self.gather(self._subfield(F, d[:, a] * d[:, b]))
        tr = np.maximum(Q[:, 0] + Q[:, 1] + Q[:, 2], 1e-300)
        Qn = Q / tr[:, None]
        s2 = (Qn[:, :3] ** 2).sum(1) + 2.0 * (Qn[:, 3:] ** 2).sum(1)
        return np.clip(1.5 * s2 - 0.5, 0.0, 1.0)

    def cohesive(self, Y, v, c, ca, wf, cl=0.0, al=None, eps=1e-6):
        """the hair-hair free energy with the oil bridges' attraction, phi the fibre fraction (weights v):
          crossing (al None): E = sum a_v h^3 (c phi^3 - ca phi_w^2), phi_w the wetted fibre fraction (weights v wf, wf
          each sample's wetting share);
          aligned (al the per-sample line-contact share A, cl = sigma_0): E = sum a_v h^3 (c phi^3 - ca phi_x^2 - cl
          phi_l ln(phi + eps)), phi_x weights v wf (1 - A) (the crossing share), phi_l weights v wf A (the line share).
          The line term's pressure phi f' - f is -cl w phi: form.py's lock law (the bridges' stress W / pi r^2 times
          phi), the unique free energy for it up to a term linear in phi, which carries no force (sum a_v h^3 phi is
          the fibre volume). eps (a fibre fraction) keeps ln finite in empty nodes; it changes nothing where hair is.
        The extra fields reuse the first's node indices and weights (one bincount and blur each); the adjoint is one
        stacked gather. Returns (E_pressure, E_cohesion, dE/dY, phi)"""
        F = self.field(Y, v)
        v = np.asarray(v, np.float64)
        wf = np.asarray(wf, np.float64)
        av = self._avox()
        shp = F.shape
        ph = F / av                                     # returned: a fresh array, as before
        if al is None or cl <= 0:
            # the same products in the same order as below, in kept crop buffers (no fresh crop-sized temporaries)
            p2 = np.multiply(ph, ph, out=self._buf("p2", shp))
            T = np.multiply(av, p2, out=self._buf("T", shp))
            np.multiply(T, ph, out=T)
            Ep = float(c) * float(np.sum(T))
            pw = np.divide(self._subfield(F, wf), av, out=self._buf("pw", shp))
            np.multiply(av, pw, out=T)
            np.multiply(T, pw, out=T)
            Ea = -float(ca) * float(np.sum(T))
            np.multiply(3.0 * float(c), p2, out=p2)
            np.multiply(-2.0 * float(ca), pw, out=pw)
            g = self.adjoint_multi([(p2, v), (pw, v * wf)])
            return Ep, Ea, g, ph
        p2 = ph * ph
        Ep = float(c) * float(np.sum(av * p2 * ph))
        al = np.asarray(al, np.float64)
        wx, wl = wf * (1.0 - al), wf * al
        px = self._subfield(F, wx) / av
        pl = self._subfield(F, wl) / av
        lg = np.log(ph + eps)
        Ea = -float(ca) * float(np.sum(av * px * px)) - float(cl) * float(np.sum(av * pl * lg))
        g = self.adjoint_multi([(3.0 * float(c) * p2 - float(cl) * pl / (ph + eps), v),
                                (-2.0 * float(ca) * px, v * wx), (-float(cl) * lg, v * wl)])
        return Ep, Ea, g, ph

    def pressure(self, Y, v, c, free_weight=True):
        F = self.field(Y, v)
        av = self._avox()
        ph = F / av
        p2 = ph * ph
        E = float(c) * float(np.sum(av * p2 * ph))
        return E, self.adjoint(3.0 * float(c) * p2), ph


def guide_spacing_mm(rods):
    """the guide spacing S (mm) at the roots: the median nearest-neighbour distance of rods.X"""
    from scipy.spatial import cKDTree
    X = np.asarray(rods.X, np.float64).reshape(-1, 3)
    if len(X) < 2:
        return float("inf")
    d, _ = cKDTree(X).query(X, k=2)
    return float(np.median(d[:, 1]) * 1000.0)


def splat_lattice_mm(f, rods=None, spacing_mm=None):
    """the pressure splat's lattice spacing (mm) from the grow values f (splat_mm 'grid' / 'auto') -> (mm, S or None)"""
    if f.get("splat_mm", "grid") in (None, "grid"):
        return float(f["grid_mm"]), None
    S = float(spacing_mm or f.get("splat_spacing_mm") or 0.0) or guide_spacing_mm(rods)
    h = float(f.get("splat_sd_frac", 0.5)) * S / SplatC1.SD_PER_CELL
    return max(float(f["grid_mm"]), h), S


def make_splat(body, f, rods=None, spacing_mm=None):
    """the settle's SplatC1 for the grow values f: on the body's grid (its a_v) when the body has one; splat_mm
    'grid' gives exactly the splat every earlier run used"""
    mm, _ = splat_lattice_mm(f, rods, spacing_mm)
    cls = SplatC1
    from . import gpu as _gpu               # HAIR_GPU=1: the same field on the graphics card (engine/hair/gpu.py)
    if _gpu.enabled():
        cls = _gpu.SplatC1GPU
    if body is not None and hasattr(body, "a"):
        return cls(body) if abs(mm - float(f["grid_mm"])) < 1e-9 else cls(body, mm=mm)
    return cls(mm=mm)


# ---------------------------------------------------------------- block-banded factors (one per guide, batched)
def _mv(M, v):
    """batched 3x3 matrix times vector: (G, 3, 3) @ (G, 3)"""
    return (M[:, :, 0] * v[:, 0:1] + M[:, :, 1] * v[:, 1:2]) + M[:, :, 2] * v[:, 2:3]


def _mtv(M, v):
    """batched transposed 3x3 matrix times vector"""
    return (M[:, 0, :] * v[:, 0:1] + M[:, 1, :] * v[:, 1:2]) + M[:, 2, :] * v[:, 2:3]


def bchol(A0, A1, A2):
    """Cholesky A = L L^T of block-banded symmetric matrices (3x3 blocks, two sub-diagonals), batched over guides.
    A0[:, k] = A_kk, A1[:, k] = A_{k+1,k}, A2[:, k] = A_{k+2,k} (zero past the end) -> (L_kk, L_kk^-1, L_{k+1,k},
    L_{k+2,k})"""
    G, K = A0.shape[:2]
    Dc = np.empty_like(A0)
    Di = np.empty_like(A0)
    S1 = np.zeros_like(A0)
    S2 = np.zeros_like(A0)
    for k in range(K):
        M = A0[:, k].copy()
        if k >= 1:
            M -= S1[:, k - 1] @ S1[:, k - 1].transpose(0, 2, 1)
        if k >= 2:
            M -= S2[:, k - 2] @ S2[:, k - 2].transpose(0, 2, 1)
        Dk = np.linalg.cholesky(M)
        Dc[:, k] = Dk
        Dik = np.linalg.inv(Dk)
        Di[:, k] = Dik
        DiT = Dik.transpose(0, 2, 1)
        B1 = A1[:, k].copy()
        if k >= 1:
            B1 -= S2[:, k - 1] @ S1[:, k - 1].transpose(0, 2, 1)
        S1[:, k] = B1 @ DiT
        S2[:, k] = A2[:, k] @ DiT
    return Dc, Di, S1, S2


def bmul_t(F, y):
    """x = L^T y for a factor F = bchol(...)"""
    Dc, _, S1, S2 = F
    K = y.shape[1]
    x = np.empty_like(y)
    for k in range(K):
        r = _mtv(Dc[:, k], y[:, k])
        if k + 1 < K:
            r += _mtv(S1[:, k], y[:, k + 1])
        if k + 2 < K:
            r += _mtv(S2[:, k], y[:, k + 2])
        x[:, k] = r
    return x


def bsolve_t(F, x):
    """y with L^T y = x (back substitution)"""
    _, Di, S1, S2 = F
    K = x.shape[1]
    y = np.empty_like(x)
    for k in range(K - 1, -1, -1):
        r = x[:, k].copy()
        if k + 1 < K:
            r -= _mtv(S1[:, k], y[:, k + 1])
        if k + 2 < K:
            r -= _mtv(S2[:, k], y[:, k + 2])
        y[:, k] = _mtv(Di[:, k], r)
    return y


def bsolve(F, z):
    """w with L w = z (forward substitution)"""
    _, Di, S1, S2 = F
    K = z.shape[1]
    w = np.empty_like(z)
    for k in range(K):
        r = z[:, k].copy()
        if k >= 1:
            r -= _mv(S1[:, k - 1], w[:, k - 1])
        if k >= 2:
            r -= _mv(S2[:, k - 2], w[:, k - 2])
        w[:, k] = _mv(Di[:, k], r)
    return w


# ---------------------------------------------------------------- the guides as rods
class Rods:
    """guide rods: roots X (G, 3) m, skin normals N, follicle directions d0 (unit), the fibre (fibre.Fibre) and the
    member hairs (flat: member_guide (M,) guide index, member_L (M,) grown length m, member_scale radius factor,
    member_cut (M,) cut length m, default = grown).

    Members along the guide (DESIGN_v2 4.4): n_i(s) = the member hairs longer than s. The guide's stiffness at a joint
    is the sum of its members' EI there, B = sum EI_f(s); its segment weight W and fibre volume V are the exact
    integrals of the members' weight and cross-section over the segment. EI and weight scale together, so a guide has
    the fibre's own bending length whatever its member count, and its ends thin as members end.
    Segments are graded: seg_root_mm at the root, x seg_growth per segment up to seg_max_mm, shared by every guide;
    each guide's last segment ends at its longest member."""

    def __init__(self, X, N, d0, fibre, member_guide, member_L, member_scale=1.0, member_cut=None, seg_root_mm=None,
                 seg_growth=None, seg_max_mm=None, sample_mm=None, min_tail_mm=None):
        g = GROW
        seg_root_mm = g["seg_root_mm"] if seg_root_mm is None else seg_root_mm
        seg_growth = g["seg_growth"] if seg_growth is None else seg_growth
        seg_max_mm = g["seg_max_mm"] if seg_max_mm is None else seg_max_mm
        sample_mm = g["sample_mm"] if sample_mm is None else sample_mm
        min_tail_mm = g["min_tail_mm"] if min_tail_mm is None else min_tail_mm
        self.X = np.ascontiguousarray(X, np.float64).reshape(-1, 3)
        self.N = _unit(np.asarray(N, np.float64).reshape(-1, 3))
        self.d0 = _unit(np.asarray(d0, np.float64).reshape(-1, 3))
        self.G = len(self.X)
        self.fibre = fibre
        self.mg = np.asarray(member_guide, np.int64).ravel()
        self.mL = np.asarray(member_L, np.float64).ravel()
        self.ms = np.broadcast_to(np.asarray(member_scale, np.float64), self.mL.shape).copy()
        self.mc = self.mL.copy() if member_cut is None else np.minimum(np.asarray(member_cut, np.float64), self.mL)
        self.seg_root, self.seg_growth, self.seg_max = seg_root_mm / 1000.0, float(seg_growth), seg_max_mm / 1000.0
        self.sample_m = sample_mm / 1000.0
        self.min_tail = min_tail_mm / 1000.0
        self.D = None
        self.rho = None
        self.members()

    def members(self, cut=None):
        """(re)build the segments and the member arrays; with cut (M,) m the members end there (blunt). Directions of
        segments that survive are kept."""
        if cut is not None:
            self.mc = np.minimum(np.asarray(cut, np.float64), self.mL)
        G = self.G
        Lg = np.zeros(G)
        np.maximum.at(Lg, self.mg, self.mc)
        self.L = Lg
        lb = [self.seg_root]
        while sum(lb) < Lg.max() - 1e-12:
            lb.append(min(lb[-1] * self.seg_growth, self.seg_max))
        lb = np.array(lb)
        sb = np.cumsum(lb) - lb
        K = len(lb)
        l = np.clip(Lg[:, None] - sb[None, :], 0.0, lb[None, :])
        nseg = (l > 0).sum(1)
        r = np.arange(G)
        last = np.maximum(nseg - 1, 0)
        tail = l[r, last]
        short = (nseg > 1) & (tail < self.min_tail) & (tail < lb[last] - 1e-12)
        l[r[short], last[short] - 1] += tail[short]
        l[r[short], last[short]] = 0.0
        s = np.cumsum(l, 1) - l
        self.K, self.l, self.s = K, l, s
        self.lbar = np.zeros((G, K))
        self.lbar[:, 0] = 0.5 * l[:, 0]
        self.lbar[:, 1:] = 0.5 * (l[:, 1:] + l[:, :-1])
        self.lbar = np.where(l > 0, self.lbar, 1.0)
        # member sums, in chunks (145k members x ~40 joints)
        B = np.zeros(G * K)
        W = np.zeros(G * K)
        n = np.zeros(G * K)
        fb = self.fibre
        jj = np.arange(K)
        for c0 in range(0, len(self.mg), 20000):
            sl = slice(c0, c0 + 20000)
            g = self.mg[sl]
            Lm, Lc, sc = self.mL[sl, None], self.mc[sl, None], self.ms[sl, None]
            S0 = s[g]
            S1 = S0 + l[g]
            on = S0 < Lc
            idx = (g[:, None] * K + jj[None, :]).ravel()
            B += np.bincount(idx, weights=(fb.EI(S0, Lm, sc) * on).ravel(), minlength=G * K)
            hc = fb.hang(Lc, Lm, sc)
            H0 = fb.hang(np.minimum(S0, Lc), Lm, sc) - hc          # weight between s and the cut end
            H1 = fb.hang(np.minimum(S1, Lc), Lm, sc) - hc
            W += np.bincount(idx, weights=(H0 - H1).ravel(), minlength=G * K)
            n += np.bincount(idx, weights=on.ravel().astype(np.float64), minlength=G * K)
        act = l > 0
        if getattr(fb, "bind", None) is not None:      # binding (fibre.bind): B = n m^beta(s) EI, the weight unchanged
            B = B * fb.bind_factor(s).ravel()
        self.B = np.where(act, B.reshape(G, K), 0.0)
        self.W = np.where(act, W.reshape(G, K), 0.0)
        self.V = self.W / (fb.rho * FB.G * fb.wet_w)               # fibre volume (m3): weight / (rho g wet)
        self.n = n.reshape(G, K)
        self.Wtot = self.W.sum(1)
        # directions: keep what survives, the rest start along d0 (growth sets them hanging when they switch on)
        D = np.repeat(self.d0[:, None, :], K, 1)
        rho = np.zeros((G, K, 3))
        if self.D is not None:
            k = min(K, self.D.shape[1])
            D[:, :k] = self.D[:, :k]
            rho[:, :k] = self.rho[:, :k]
        self.D, self.rho = D, rho
        # the sample table over every segment (a stage uses those of its active segments)
        ns = np.where(act, np.maximum(2, np.ceil(l / self.sample_m - 1e-9)), 0).astype(np.int64).ravel()
        self.q_all = np.repeat(np.arange(G * K), ns)
        st = np.cumsum(ns) - ns
        kk = np.arange(len(self.q_all)) - np.repeat(st, ns)
        nr = np.repeat(ns, ns).astype(np.float64)
        self.t_all = (kk + 0.5) / nr
        self.f_all = 1.0 / nr
        return self

    def positions(self, D=None):
        """node positions (G, K + 1, 3); past a guide's end the last node repeats"""
        D = self.D if D is None else D
        return np.concatenate([self.X[:, None, :], self.X[:, None, :] + np.cumsum(self.l[..., None] * D, 1)], 1)

    def node_arcs(self):
        return np.concatenate([np.zeros((self.G, 1)), np.cumsum(self.l, 1)], 1)


# ---------------------------------------------------------------- the natural wave (DESIGN_v2 5, waves phase 2)
# What makes a wave: the follicle is curved and the cortex is asymmetric (ortho- / para-cortex on the inner and outer
# side of the bend), so the fibre grows with an intrinsic curvature (Thibaut et al. 2005, Br J Dermatol 152:632; Bryson
# et al. 2009). Wavy European hair is a flattened helix close to a plane: curve diameters of a few cm and 1-3 waves per
# 10 cm (de la Mettrie, Loussouarn et al. 2007, Hum Biol 79:265; Loussouarn et al. 2007, Int J Dermatol 46 S1:2);
# mean curvature 0.042 +- 0.02 /mm (Lasisi et al. 2016). That curvature is a material property: it is carried in the
# fibre's own frame and turns with the fibre (Bertails et al. 2006, Super-helices, SIGGRAPH: the rest curvature lives in
# the material frame). Hanging, the weight below a point pulls its wave out: amplitude / free ~ 1 / (1 + s_below /
# (l_g^3 k^2)), l_g^3 = EI / w (Miller, Lazarus, Audoly & Reis 2014, PRL 112:068103) - straight near the root, waving
# toward the ends. Here the fibre carries only that rest turning; gravity, the body and the pressure do the rest.
#
# The rest turning of joint j (d_{j-1} -> d_j) is kept in the frame (t, n, b) carried from the follicle by parallel
# transport along the guide's current directions: rho_j = c_t t + c_n n + c_b b. Each sub-solve sees it as a fixed
# world vector (lagged); between sub-solves the frame is carried along the strand again and rho turns with it. A rest
# turning written in world axes and never turned (the prototypes' fixed lagged axes) gave a flat crown and pushed points
# into the body: the strand had to turn back to meet axes that no longer matched it.
#
# The set (actions.set_held: rho <- rho + r (held - rho)) and any other act that writes rho is read as an edit to the
# rest turning in the frame of the moment and kept, in the same material frame, beside the natural wave:
# rho = natural + set. After the set, rho = (1 - r) natural + r held (r = 0 leaves the natural wave); rewet() takes
# the set away (water breaks the hydrogen bonds that held it) and fully wet hair is back to its natural wave.

def transport_frames(rods, D=None):
    """the frame carried from each follicle along its guide by parallel transport (rotation-minimising, no twist): at
    joint j the tangent t_j = d_{j-1} (d_{-1} = d0); n_j is n_{j-1} turned by the rotation that takes t_{j-1} to t_j;
    b_j = t_j x n_j. The root's n is rods.wave_n0 (the wave's plane at the follicle). -> T, Nn, Bn (G, K, 3)"""
    D = rods.D if D is None else D
    G, K = D.shape[:2]
    T = np.empty((G, K, 3))
    T[:, 0] = rods.d0
    T[:, 1:] = D[:, :-1]
    Nn = np.empty_like(T)
    n = rods.wave_n0
    n = _unit(n - (n * T[:, 0]).sum(1, keepdims=True) * T[:, 0])
    Nn[:, 0] = n
    for j in range(1, K):
        ta, tb = T[:, j - 1], T[:, j]
        c = 1.0 + (ta * tb).sum(1)
        ok = c > 1e-9                                    # a reversal (never in a rod) leaves n as it was
        n = n - np.where(ok, (n * tb).sum(1) / np.where(ok, c, 1.0), 0.0)[:, None] * (ta + tb)
        n = _unit(n - (n * tb).sum(1, keepdims=True) * tb)
        Nn[:, j] = n
    return T, Nn, np.cross(T, Nn)


# The type's helix (fibre.hair_type) in the energy itself (discrete elastic rods: Bergou, Wardetzky, Robinson, Audoly &
# Grinspun 2008; the material frame here is the twist-free frame carried from the follicle, the zero-twist limit of
# Bergou et al. 2010's quasi-static twist). The rest turning of joint j, loc_j = (c_t, c_n, c_b), is read in the frame
# (t_j, n_j, b_j) carried from the follicle along the CURRENT directions, and the gradient goes through the transport:
#     E = sum_j cb_j / 2 |d_j - t_j - c_t t_j - c_n n_j - c_b t_j x n_j|^2,   t_j = d_{j-1} (t_0 = d0),
#     n_j = n_{j-1} - (n_{j-1} . t_j) / (1 + t_{j-1} . t_j) (t_{j-1} + t_j)   (the exact minimal rotation; n_0 fixed)
# One energy with one minimum at the rest shape (E = 0 free), so a tight coil has nothing to drift on: the lagged
# world-space rho refreshed between sub-solves (piece A) was a fixed-point iteration without that minimum's gradient
# (the frame's dependence on the upstream directions was missing), unstable around a 16-rad coil. Cost: one forward and
# one reverse sweep over the joints, vectorised over the guides (O(G K)).
# The settle has no twist energy, no torsional stiffness and no shear modulus: keep it so beside the oil bridges'
# cohesion and adhesion (docs/PATENTS.md, Hair).
def bend_carried(D, d0, n0, loc, cb):
    """the bending energy with the rest turning loc (G, K, 3) in the frame carried along D (G, K, 3) from the follicle
    (d0, n0 unit, n0 . d0 = 0), weights cb (G, K) = B / lbar -> (E, dE/dD (G, K, 3), rho (G, K, 3) world)"""
    G, K = D.shape[:2]
    T = np.empty((G, K, 3))
    T[:, 0] = d0
    T[:, 1:] = D[:, :-1]
    Nn = np.empty_like(T)
    Nn[:, 0] = n0
    C = np.ones((G, K, 1))
    A = np.zeros((G, K, 1))
    n = n0
    for j in range(1, K):
        ta, tb = T[:, j - 1], T[:, j]
        c = np.maximum(1.0 + np.einsum("ij,ij->i", ta, tb), 1e-9)[:, None]
        a = np.einsum("ij,ij->i", n, tb)[:, None] / c
        n = n - a * (ta + tb)
        Nn[:, j] = n
        C[:, j] = c
        A[:, j] = a
    ct, cn, cbn = loc[..., 0:1], loc[..., 1:2], loc[..., 2:3]
    rho = ct * T + cn * Nn + cbn * np.cross(T, Nn)
    res = D - T - rho
    g = cb[..., None] * res
    E = 0.5 * float((g * res).sum())
    Tb = -(1.0 + ct) * g - cbn * np.cross(Nn, g)          # dE/dt_j, direct
    Nb = -cn * g - cbn * np.cross(g, T)                    # dE/dn_j, direct
    carry = np.zeros((G, 3))
    for j in range(K - 1, 0, -1):                          # back through the transport
        nb = Nb[:, j] + carry
        ta, tb = T[:, j - 1], T[:, j]
        c, a = C[:, j], A[:, j]
        bc = np.einsum("ij,ij->i", nb, ta + tb)[:, None] / c
        carry = nb - bc * tb
        Tb[:, j] += -a * nb - bc * Nn[:, j - 1] + (bc * a) * ta
        Tb[:, j - 1] += -a * nb + (bc * a) * tb
    gD = g.copy()
    gD[:, :-1] += Tb[:, 1:]
    return E, gD, rho


def section_wave(n_sec, waves=None, seed=0):
    """one wave train per drying section: the wavelength (m) and the wave plane's tilt (rad) of each of n_sec sections.
    The hairs of a section are wetted, brushed and dried together in one brush or hand, and dry nested into one wave
    train (hair_styling.md 6: brushing out joins neighbours into one train; the pre-rebuild phase-1 path used one
    wavelength for all), so they share its step and its plane; each keeps its own amplitude (its own follicle's curl,
    Thibaut et al. 2005) and its own small phase jitter. A wavelength drawn per hair instead dephases neighbours along
    the length - at 300 mm, 70 mm +/- 10 % puts two hairs ~2.7 rad apart - and a child blended from 2-4 such guides
    (or a lock packed toward their mean) averages the wave away (round 17: guides 3.0 mm rms, children 0.73).
    waves.wavelength_mm x U(1 -/+ waves.wavelength_jitter) and waves.tilt_deg (uniform in its range), per section.
    -> lam (n_sec,) m, tilt (n_sec,) rad"""
    W8 = waves or {}
    rng = np.random.default_rng(int(seed) + 919)
    lj = float(W8.get("wavelength_jitter", 0.0))
    lam = max(float(W8.get("wavelength_mm", 0.0) or 0.0), 1e-6) / 1000.0 * rng.uniform(1.0 - lj, 1.0 + lj, int(n_sec))
    tv = W8.get("tilt_deg", 0.0)
    tv = list(tv) if isinstance(tv, (list, tuple)) else [tv, tv]
    tilt = np.radians(rng.uniform(float(tv[0]), float(tv[1]), int(n_sec)))
    return lam, tilt


def rest_waves(rods, waves=None, seed=0, phase=None, random_mm=0.0, section_mm=35.0, jitter=0.15, lam=None,
               tilt=None, helix=None):
    """give the rods their natural wave (in place): per guide the free amplitude A (m, uniform in waves.amp_mm), the
    wavelength lambda (waves.wavelength_mm x U(1 -/+ wavelength_jitter)), the start (waves.start_mm: the wave ramps in
    over half a wavelength from there, as form.section_waves does), the plane (waves.tilt_deg: turned out of the hair
    sheet - the side direction along the scalp - toward the skin normal) and the phase: phase (G,) rad, or each guide's
    drying section's phase from form.section_phase (section_mm, jitter) so neighbouring guides wave together.
    random_mm: rms (1/mm) of an extra smooth random curvature (4 sinusoids per direction, 40-200 mm). lam (G,) m and
    tilt (G,) rad: each guide's drying section's wavelength and plane (section_wave), so the guides of one section wave
    as one train; None draws them per guide as before (neighbours then dephase along the length). -> rods"""
    W8 = waves or {}
    G = rods.G
    rng = np.random.default_rng(int(seed) + 911)
    lam_in, tilt_in = lam, tilt

    def rng2(key, dflt):
        v = W8.get(key, dflt)
        v = list(v) if isinstance(v, (list, tuple)) else [v, v]
        return rng.uniform(float(v[0]), float(v[1]), G)
    lam_mm = float(W8.get("wavelength_mm", 0.0) or 0.0)
    lj = float(W8.get("wavelength_jitter", 0.0))
    lam = max(lam_mm, 1e-6) / 1000.0 * rng.uniform(1.0 - lj, 1.0 + lj, G)
    A = rng2("amp_mm", 0.0) / 1000.0 if lam_mm > 0 else np.zeros(G)
    s0 = rng2("start_mm", 0.0) / 1000.0
    tilt = np.radians(rng2("tilt_deg", 0.0))
    if phase is None:
        from engine.hair import form as FO               # here: form is not needed by a settle without waves
        phase = FO.section_phase(rods.X, section_mm, seed, jitter)[0]
    side = _unit(np.cross(rods.N, rods.d0))               # along the scalp, across the exit direction
    up = _unit(np.cross(rods.d0, side))                   # the skin normal's part across the exit direction
    if lam_in is not None:   # the section's train (the per-guide draws above are kept so the other draws do not move)
        lam = np.asarray(lam_in, np.float64).reshape(G)
    if tilt_in is not None:
        tilt = np.asarray(tilt_in, np.float64).reshape(G)
    rods.wave_n0 = np.cos(tilt)[:, None] * side + np.sin(tilt)[:, None] * up
    M = 4
    rnd = None
    if random_mm > 0:
        # each sinusoid of amplitude a has rms a / sqrt 2; M of them per direction: a = rms sqrt(2 / M)
        rnd = dict(a=float(random_mm) * 1000.0 * np.sqrt(2.0 / M),
                   k=2 * np.pi / (rng.uniform(0.040, 0.200, (2, G, M))), p=rng.uniform(0, 2 * np.pi, (2, G, M)))
    rods.wave = dict(A=A, lam=lam, s0=s0, phase=np.asarray(phase, np.float64).reshape(G), rnd=rnd)
    # the hair type's rest helix (fibre.hair_type; helix None = the fibre's, False = none): on the same phase as the
    # wave - each guide's drying section's - so the hairs of one lock or ringlet share one curl, each with its small
    # jitter about the section's step; cd_jitter scales kappa and tau together (the same curl index) per section
    hx = getattr(rods.fibre, "hair", None) if helix is None else (helix or None)
    if hx is not None and hx.get("curl"):
        kap, tau = np.full(G, float(hx["kappa"])), np.full(G, float(hx["tau"]))
        jit = float(hx.get("cd_jitter") or 0.0)
        if jit > 0:
            from engine.hair import form as FO
            _, sec, ph_sec = FO.section_phase(rods.X, section_mm, seed, jitter, parts=True)
            fac = np.random.default_rng(int(seed) + 917).uniform(1.0 - jit, 1.0 + jit, len(ph_sec))[sec]
            kap, tau = kap / fac, tau / fac
        rods.wave["hx"] = dict(kappa=kap, tau=tau, flat=float(hx.get("flat", 1.0)),
                               s0=float(hx.get("start_mm") or 0.0) / 1000.0, ramp=float(hx.get("ramp_mm") or 0.0) / 1000.0)
    rods.wave_x = None
    rods.wave_rho_w = None
    return rods


def helix_on(fibre):
    """True when the fibre's hair type carries a rest helix (fibre.hair_type: a curl, not 'straight')"""
    hx = getattr(fibre, "hair", None)
    return bool(hx is not None and hx.get("curl"))


def wave_natural(rods):
    """the natural rest turning of every joint in its carried frame, (G, K, 3) components along (t, n, b): a rotation
    by theta = kappa(s_j) lbar_j toward n (the coherent wave) and b (the random part)"""
    w = rods.wave
    s, lb = rods.s, rods.lbar
    k = 2 * np.pi / w["lam"][:, None]
    a, b = w["s0"][:, None], (w["s0"] + 0.5 * w["lam"])[:, None]
    x = np.clip((s - a) / (b - a), 0.0, 1.0)
    ramp = x * x * (3 - 2 * x)
    kn = w["A"][:, None] * k * k * np.sin(k * s + w["phase"][:, None])
    kb = np.zeros_like(kn)
    if w["rnd"] is not None:
        r = w["rnd"]
        kn = kn + r["a"] * np.sin(r["k"][0][:, None, :] * s[..., None] + r["p"][0][:, None, :]).sum(-1)
        kb = kb + r["a"] * np.sin(r["k"][1][:, None, :] * s[..., None] + r["p"][1][:, None, :]).sum(-1)
    on = rods.l > 0
    tn = np.where(on, kn * ramp * lb, 0.0)
    tb = np.where(on, kb * ramp * lb, 0.0)
    hx = w.get("hx")
    if hx is not None:
        # the type's helix: curvature kappa (cos th, flat sin th) in the carried frame, th = tau (s - start) + phase
        # (fibre.hair_type) - the bending direction turns along the fibre; flat 0 is the planar wave above
        x = np.clip((s - hx["s0"]) / hx["ramp"], 0.0, 1.0) if hx["ramp"] > 0 else np.ones_like(s)
        rh = np.where(s >= hx["s0"], x * x * (3 - 2 * x), 0.0)
        ang = hx["tau"][:, None] * (s - hx["s0"]) + w["phase"][:, None]
        kh = hx["kappa"][:, None] * rh * lb
        tn = tn + np.where(on, kh * np.cos(ang), 0.0)
        tb = tb + np.where(on, hx["flat"] * kh * np.sin(ang), 0.0)
    th = np.hypot(tn, tb)
    sth = np.where(th > 1e-15, np.sin(th) / np.where(th > 1e-15, th, 1.0), 1.0)
    return np.stack([np.cos(th) - 1.0, sth * tn, sth * tb], -1)


def wave_rest_shape(rods):
    """the natural rest shape alone (no gravity, no body): each joint's turning applied in the carried frame from the
    follicle -> nodes (G, K + 1, 3)"""
    nat = wave_natural(rods)
    G, K = rods.G, rods.K
    D = np.empty((G, K, 3))
    t = rods.d0
    n = _unit(rods.wave_n0 - (rods.wave_n0 * t).sum(1, keepdims=True) * t)
    for j in range(K):
        if j > 0:
            ta, tb = t, D[:, j - 1]
            c = 1.0 + (ta * tb).sum(1)
            n = _unit(n - ((n * tb).sum(1) / np.maximum(c, 1e-9))[:, None] * (ta + tb))
            n = _unit(n - (n * tb).sum(1, keepdims=True) * tb)
            t = tb
        b = np.cross(t, n)
        D[:, j] = _unit(t + nat[:, j, 0:1] * t + nat[:, j, 1:2] * n + nat[:, j, 2:3] * b)
    return rods.positions(D)


def rewet(state, wet=1.0):
    """water breaks the hydrogen bonds the set re-formed: the set's share of the rest turning goes, x wet (0-1); fully
    wet hair (wet 1) is back to its natural wave. Needs the rod waves (state.wv_on). -> state"""
    if not state.wv_on:
        return state
    state.wave_frame()
    state.rods.wave_x *= (1.0 - float(wet))
    state.wave_frame()
    return state


# ---------------------------------------------------------------- one settle's state
class View:
    """what an act (or a report) sees of the current configuration"""
    pass


class State:
    """the settle of one set of rods: body (grid.SDFGrid or anything with .sample(P, grad)), the splat (grid.Splat)
    when the hair-hair pressure is on, the active segments of the current stage, the lagged friction, the history."""

    def __init__(self, rods, body=None, cfg=None, pressure=True, friction=True, acts=None, splat=None, log=print):
        self.rods = rods
        self.body = body
        self.f, self.warnings = settings(cfg)
        self.log = log or (lambda m: None)
        fb = rods.fibre
        self.h_c = self.f["contact_mm"] / 1000.0
        self.k_c = float(self.f["contact_k"])
        self.eps = self.f["friction_eps_mm"] / 1000.0
        self.pressure = bool(pressure)
        self.friction = bool(friction)
        self.c_p = fb.c_p
        self.mu = fb.mu
        # hair on skin: the share of a sample's load the body's contact carries meets the skin's coefficient
        # (fibre.mu_skin, ~0.5 measured), the share the neighbours' pressure carries the hair-hair one (lag)
        self.mu_s = float(getattr(fb, "mu_skin", fb.mu))
        # the oil's liquid bridges (fibre.Fibre.adhesive_load): None = friction carries only contact and pressure
        self.adh = getattr(fb, "adhesion", None) if self.friction else None
        self.reach = self.f["friction_reach_mm"] / 1000.0
        self.adh_rep = []
        # the same bridges' attraction in the hair-hair free energy (fibre.Fibre.cohesion_c0): 0 = pressure only
        self.c_a = fb.cohesion_c0() if (self.pressure and getattr(fb, "cohesion", False)) else 0.0
        # the aligned count (fibre.cohesion_count "aligned"): the line contacts' stress sigma_0 = gamma cos / r, each
        # sample's line share its local alignment A (set per stage in _setup, as the wetting is: the brush set the
        # alignment, the settle moves it little), the locks' law on that share
        self.c_l = fb.cohesion_sigma() if (self.c_a > 0 and getattr(fb, "cohesion_count", "crossing") == "aligned") \
            else 0.0
        self.swf = None
        self.sal = None
        if splat is None and self.pressure:
            splat = make_splat(body, self.f, rods)
            if self.f.get("splat_mm", "grid") != "grid":
                mm, S = splat_lattice_mm(self.f, rods)
                self.log("pressure splat: lattice %.2f mm (kernel sd %.2f mm) for guide spacing S %.2f mm" % (
                    mm, mm * SplatC1.SD_PER_CELL, S))
        self.splat = splat
        self.acts = acts
        self.c1 = bool(self.f["contact_c1"]) and body is not None and hasattr(body, "A") and hasattr(body, "lo")
        self.margin = self.f["contact_margin_mm"] / 1000.0
        # the whole core on the graphics card (gpu.CoreGPU; HAIR_GPU=1 and the GPU splat; C1 contact only), forces and
        # the solver's objective included - the CPU core below stays the reference and the default
        self._gpu = None
        if self.c1 and (not self.pressure or getattr(self.splat, "on_gpu", False)):
            from . import gpu as _gpu
            if _gpu.enabled():
                self._gpu = _gpu.CoreGPU()
        self.kind = self.f["precond"]
        self.F = None
        G = rods.G
        self.ESC = 1.0 / (G * max(float(rods.Wtot.mean()), 1e-300) * 1e-3)
        ns = len(rods.q_all)
        self.anc = np.zeros((ns, 3))
        self.anc_N = np.zeros(ns)
        self.anc_M = np.zeros(ns)            # mu N, the friction's limit force per anchored sample (N)
        self.anc_n = np.zeros((ns, 3))
        self.act_full = np.zeros((G, rods.K), bool)
        self.Ka = 0
        self.hist = []
        self.trace = []
        self.evals = 0
        self.eval_s = 0.0
        self.gate = None
        self.stage_index = -1
        self.t0 = time.time()
        # the natural wave (rod_waves): the rods carry it (rest_waves), so a second State on the same rods (the one
        # after the cut) carries on from the first's bookkeeping instead of counting the wave twice
        self.wv_on = bool(self.f["rod_waves"]) or helix_on(rods.fibre)     # the type's helix needs the rest turning
        self.wv_n = 0
        if self.wv_on and getattr(rods, "wave", None) is None:
            sty = cfg if isinstance(cfg, dict) else {}
            lk = sty.get("locks") if isinstance(sty.get("locks"), dict) else {}
            rest_waves(rods, sty.get("waves") if isinstance(sty.get("waves"), dict) else {}, sty.get("seed", 0),
                       random_mm=float(self.f["wave_random_mm"]), section_mm=float(lk.get("section_mm", 35.0)),
                       jitter=float(lk.get("phase_jitter", 0.15)))
        # the type's helix (fibre.hair_type): the bending term carries the frame along the strand as it is now, with
        # its gradient (bend_carried) - no lagged world-space rest turning. Other fibres keep the lagged rho.
        self.hx_frame = helix_on(rods.fibre)
        self.loc = self.wave_n0c = None
        if self.hx_frame:
            self.wave_frame()

    def wave_frame(self):
        """the rest turning carried along each guide as it is now (rod_waves; called once per sub-solve): the acts'
        edits to rho since the last call are read in today's frame and kept, then rho = natural + those edits, both
        in the frame carried from the follicle along the current strand"""
        if not self.wv_on:
            return
        r = self.rods
        G, K = r.G, r.K
        for key in ("wave_x", "wave_rho_w"):
            old = getattr(r, key, None)
            new = np.zeros((G, K, 3))
            if old is not None:
                k = min(K, old.shape[1])
                new[:, :k] = old[:, :k]
            setattr(r, key, new)
        T, Nn, Bn = transport_frames(r)
        dl = r.rho - r.wave_rho_w                                  # what the acts wrote since (the set, the lift)
        r.wave_x += np.stack([(dl * T).sum(2), (dl * Nn).sum(2), (dl * Bn).sum(2)], -1)
        loc = wave_natural(r) + r.wave_x
        rho = loc[..., 0:1] * T + loc[..., 1:2] * Nn + loc[..., 2:3] * Bn
        rho[r.l <= 0] = 0.0
        r.rho[...] = rho
        r.wave_rho_w = rho.copy()
        self.wv_n += 1
        if self.hx_frame:          # the energy carries the frame itself (bend_carried): only the local turning is kept
            self.loc = np.where((r.l > 0)[..., None], loc, 0.0)
            self.wave_n0c = Nn[:, 0].copy()

    # -------------------------------------------------- the active set
    def activate(self, arc):
        """switch on the segments that start before arc (m); new ones start hanging, pushed out of the body. arc is
        one arc for every guide, or (G,) one per guide (styling in sections, DESIGN_sections 3.2: the section being
        worked at its stage arc, the laid sections at full length, the sections not yet started at 0 - clipped up,
        no segments)"""
        r = self.rods
        if np.ndim(arc) == 0:
            act = (r.l > 0) & (r.s < arc - 1e-12)
        else:
            act = (r.l > 0) & (r.s < np.asarray(arc, np.float64)[:, None] - 1e-12)
        new = act & ~self.act_full
        Ka = int(np.nonzero(act.any(0))[0].max()) + 1
        D = r.D
        drop = np.array([0.0, 0.0, -float(self.f["hang_drop"])])
        ac = self.acts
        Tp = getattr(ac, "T_path", None) if (ac is not None and bool(self.f["grow_along_stroke"]) and
                                             getattr(ac, "through", False) and getattr(ac, "brush_on", False)) else None
        P = np.zeros((r.G, Ka + 1, 3))
        P[:, 0] = r.X
        lact = r.l * act
        # the hair type's helix (fibre.hair_type): the fibre leaves the follicle already curled - a new segment emerges
        # turned from the one before by its natural rest turning in the frame carried along the strand (the settle
        # then only meets gravity and the body; a tight coil grown straight and left to the lagged frame does not
        # find its rest shape). Without a type the new segments hang as before.
        emerge = helix_on(r.fibre) and getattr(r, "wave", None) is not None
        if emerge:
            nat = wave_natural(r)
            t_c = r.d0.copy()
            n_c = _unit(r.wave_n0 - (r.wave_n0 * t_c).sum(1, keepdims=True) * t_c)
        for j in range(Ka):
            m = new[:, j]
            if emerge and j > 0:                  # carry the frame from joint j - 1 to j along the strand as it is
                tb = D[:, j - 1]
                c = 1.0 + (t_c * tb).sum(1)
                ok = c > 1e-9
                n_c = n_c - np.where(ok, (n_c * tb).sum(1) / np.where(ok, c, 1.0), 0.0)[:, None] * (t_c + tb)
                n_c = _unit(n_c - (n_c * tb).sum(1, keepdims=True) * tb)
                t_c = tb
            if m.any():
                if emerge:
                    b_c = np.cross(t_c[m], n_c[m])
                    D[m, j] = _unit(t_c[m] + nat[m, j, 0:1] * t_c[m] + nat[m, j, 1:2] * n_c[m] + nat[m, j, 2:3] * b_c)
                else:
                    D[m, j] = r.d0[m] if j == 0 else _unit(D[m, j - 1] + drop)
                if Tp is not None and j > 0 and j < Tp.shape[1]:
                    t = Tp[m, j]                   # in the brush: along the stroke at this arc (where it has one)
                    ok = np.isfinite(t).all(1) & (np.abs((t * t).sum(1) - 1.0) < 1e-3)
                    mi = np.nonzero(m)[0][ok]
                    if getattr(ac, "steer", False) and self.body is not None and len(mi):
                        # the brush steers the heading only (through_hold 'steer'): the new length takes the stroke's
                        # heading in the skin's tangent plane and keeps the elevation it would have had (curled or
                        # hanging), instead of starting flat along the path on the skin
                        _, gn = self.body.sample(P[mi, j], grad=True)
                        nn = _unit(gn)
                        en = np.clip((D[mi, j] * nn).sum(1), -1.0, 1.0)
                        tt = _unit(t[ok] - (t[ok] * nn).sum(1, keepdims=True) * nn)
                        D[mi, j] = _unit(tt * np.sqrt(1.0 - en * en)[:, None] + nn * en[:, None])
                    else:
                        D[mi, j] = t[ok]
                if self.body is not None:
                    for _ in range(2):
                        pe = P[m, j] + r.l[m, j, None] * D[m, j]
                        sd, g = self.body.sample(pe, grad=True)
                        bad = sd < self.h_c + 1e-3
                        if not bad.any():
                            break
                        pe[bad] += (self.h_c + 1e-3 - sd[bad])[:, None] * _unit(g[bad])
                        dn = _unit(pe - P[m, j])
                        D[np.nonzero(m)[0][bad], j] = dn[bad]
            P[:, j + 1] = P[:, j] + lact[:, j, None] * D[:, j]
        self.act_full = act
        self._setup(Ka)
        return int(new.sum())

    def _setup(self, Ka):
        r = self.rods
        self.Ka = Ka
        act = self.act_full[:, :Ka]
        self.act = act
        self.lk = np.where(act, r.l[:, :Ka], 0.0)
        self.cb = np.where(act, r.B[:, :Ka] / r.lbar[:, :Ka], 0.0)
        self.Wk = np.where(act, r.W[:, :Ka], 0.0)
        self.s0 = r.s[:, :Ka]
        sm = self.act_full.ravel()[r.q_all]
        self.sidx = np.nonzero(sm)[0]
        q = r.q_all[self.sidx]
        i, j = q // r.K, q % r.K
        self.sq = i * Ka + j
        self.st = r.t_all[self.sidx]
        self.sw = r.W[i, j] * r.f_all[self.sidx]
        self.sv = r.V[i, j] * r.f_all[self.sidx]
        self.sarc = r.s[i, j] + self.st * r.l[i, j]
        self.swf = r.fibre.wetting(self.sarc) if self.c_a > 0 else None
        self.sal = self._alignment(i, j) if self.c_l > 0 else None
        self._anchors()
        self.precondition()

    def _alignment(self, i, j):
        """each active sample's local alignment A (SplatC1.alignment) in the configuration now: the samples on their
        segments, each with its segment's direction"""
        r = self.rods
        Ka = self.Ka
        D = r.D[:, :Ka]
        P1 = r.X[:, None, :] + np.cumsum(self.lk[..., None] * D, 1)
        P0 = np.concatenate([r.X[:, None, :], P1[:, :-1]], 1)
        Y = P0[i, j] + self.st[:, None] * (P1[i, j] - P0[i, j])
        return self.splat.alignment(Y, self.sv, D[i, j])

    def _anchors(self):
        """the friction anchors of the active samples that have one: index, anchor, normal, mu N"""
        Nf = self.anc_N[self.sidx]
        on = np.nonzero(Nf > 0)[0]
        self.f_on = on
        self.f_anc = self.anc[self.sidx][on]
        self.f_n = self.anc_n[self.sidx][on]
        self.f_mn = self.anc_M[self.sidx][on]

    def precondition(self, view=None):
        """the solver's variables for the next solve (lagged: built from the configuration now).
        jacobi (DESIGN_v2 4.5): x = s u, s^2 = ESC (2 B_j / lbar_j + 2 B_{j+1} / lbar_{j+1} + l_j T_j / 2) (+ the acts'
        own stiffness), T_j the active weight hanging from segment j on.
        banded: x = R u with R^T R = ESC L^T A L, where L maps the segments to the nodes (y_k = sum_{j <= k} l_j u_j)
        and A is the guide's stiffness in node space: a string of stiffness lambda_j / l_j along each segment
        (lambda_j its axial force from the force balance now, at least 0.1 x the weight below: the geometric
        stiffness of an inextensible chain), the bending (B / lbar on the second differences), the contact springs
        k_c w n n^T of the samples touching the body, and the friction springs mu N / eps in the tangent plane. In node
        space all of these are local, so A is block-banded (two sub-diagonals of 3x3 blocks) and its Cholesky factor
        costs O(K) per guide; a contact, which couples every segment upstream of it in direction variables, is then
        exact in the scaling. The hair-hair pressure is left out (per sample ~1e-3 of the tension's stiffness)."""
        cb = self.cb
        T = np.cumsum(self.Wk[:, ::-1], 1)[:, ::-1]
        extra = None
        if self.acts is not None and hasattr(self.acts, "precond"):
            extra = self.acts.precond(view if view is not None else self.view())
        if self.kind == "jacobi":
            cbn = np.zeros_like(cb)
            cbn[:, :-1] = cb[:, 1:]
            s2 = 2 * cb + 2 * cbn + 0.5 * self.lk * T
            if extra is not None:
                s2 = s2 + extra
            self.sc = np.where(self.act, np.sqrt(np.maximum(s2 * self.ESC, 1e-300)), 1.0)
            self.F = None
            return
        r = self.rods
        G, Ka, act = r.G, self.Ka, self.act
        lm = np.where(act, self.lk, 1.0)
        self.lmap = lm
        U = r.D[:, :Ka]
        self.F = None
        _, _, o = self.core(U, forces=True)
        D, gD = o["D"], o["gD"]
        # the axial (tension) force of each segment: minus the direction gradient's component along it, over l
        lam = np.where(act, np.maximum(-(gD * D).sum(2) / lm, 0.1 * T), 0.0)
        kap = lam / lm
        if extra is not None:
            kap = kap + np.where(act, extra / lm ** 2, 0.0)
        alpha = np.where(act, 1.0 / lm, 0.0)
        beta = np.zeros_like(alpha)
        beta[:, 1:] = alpha[:, :-1]
        a0 = kap.copy()
        a0[:, :-1] += kap[:, 1:]
        a1 = np.zeros((G, Ka))
        a2 = np.zeros((G, Ka))
        a1[:, :-1] -= kap[:, 1:]
        a0 += cb * alpha ** 2
        a0[:, :-1] += (cb * (alpha + beta) ** 2)[:, 1:]
        a0[:, :-2] += (cb * beta ** 2)[:, 2:]
        a1[:, :-1] -= (cb * alpha * (alpha + beta))[:, 1:]
        a1[:, :-2] -= (cb * (alpha + beta) * beta)[:, 2:]
        a2[:, :-2] += (cb * alpha * beta)[:, 2:]
        I3 = np.eye(3)
        A0 = a0[..., None, None] * I3
        A1 = a1[..., None, None] * I3
        A2 = a2[..., None, None] * I3
        # the contact and friction springs on the samples, shared by the segment's two end nodes
        Hs = np.zeros((len(self.sq), 3, 3))
        fc = o["fc"]
        nc = np.sqrt((fc * fc).sum(1))
        hit = nc > 0
        if hit.any():
            n = fc[hit] / nc[hit, None]
            kh = self.k_c * self.sw[hit]
            ek = float(self.f.get("contact_ear_k", 1.0))
            if ek != 1.0:                               # the ears' stiffer skin (contact_ear_k)
                kh = kh * (1.0 + (ek - 1.0) * fine_weight(self.body, o["Y"][hit])[0])
            Hs[hit] += kh[:, None, None] * n[:, :, None] * n[:, None, :]
        if self.friction and self.mu > 0 and len(self.f_on):
            on, nf = self.f_on, self.f_n
            dl = o["Y"][on] - self.f_anc
            tdl = dl - (dl * nf).sum(1, keepdims=True) * nf
            sq_ = np.sqrt(1.0 + (tdl * tdl).sum(1) / self.eps ** 2)
            kf = self.f_mn / (self.eps * sq_)
            Hs[on] += kf[:, None, None] * (I3 - nf[:, :, None] * nf[:, None, :])
        t = self.st
        j = self.sq % Ka
        n_ = G * Ka
        has_a = j >= 1
        for comp in range(9):
            a_, b_ = divmod(comp, 3)
            h = Hs[:, a_, b_]
            A0[..., a_, b_] += np.bincount(self.sq, weights=t * t * h, minlength=n_).reshape(G, Ka)
            A0[..., a_, b_] += np.bincount(self.sq[has_a] - 1, weights=((1 - t) ** 2 * h)[has_a],
                                           minlength=n_).reshape(G, Ka)
            A1[..., a_, b_] += np.bincount(self.sq[has_a] - 1, weights=(t * (1 - t) * h)[has_a],
                                           minlength=n_).reshape(G, Ka)
        A0 *= self.ESC
        A1 *= self.ESC
        A2 *= self.ESC
        A0[~act] = I3
        A1[~act] = 0.0
        A2[~act] = 0.0
        self.F = bchol(A0, A1, A2)

    # -------------------------------------------------- the energy
    def _U(self, x):
        """the segment vectors u from the solver's variables"""
        Ka = self.Ka
        if self.F is None:
            U = self.rods.D[:, :Ka].copy()
            U[self.act] = x.reshape(-1, 3) / self.sc[self.act][:, None]
            return U
        y = bsolve_t(self.F, x.reshape(self.rods.G, Ka, 3))
        U = y.copy()
        U[:, 1:] -= y[:, :-1]
        return U / self.lmap[..., None]

    def x_of(self, D=None):
        """the solver's variables of the directions D (default: the current ones)"""
        D = self.rods.D[:, :self.Ka] if D is None else D
        if self.F is None:
            return (D * self.sc[..., None])[self.act].ravel()
        return bmul_t(self.F, np.cumsum(self.lmap[..., None] * D, 1)).ravel()

    def _gx(self, gU):
        """the scaled energy's gradient in the solver's variables, from dE/dU"""
        if self.F is None:
            return (gU / self.sc[..., None])[self.act].ravel() * self.ESC
        g = gU / self.lmap[..., None]
        z = g.copy()
        z[:, :-1] -= g[:, 1:]
        return bsolve(self.F, z).ravel() * self.ESC

    def core(self, U, forces=False):
        """energy (J) and its gradient with respect to U (G, Ka, 3) of the active segments; with forces also the parts,
        the physical direction gradient (no conditioning) and the contact and pressure forces on the samples"""
        if getattr(self, "_gpu", None) is not None:
            return self._gpu.core(self, U, forces=forces)
        r = self.rods
        act, lk, cb = self.act, self.lk, self.cb
        G, Ka = r.G, self.Ka
        nu = np.sqrt((U * U).sum(2))
        nu = np.where(act, nu, 1.0)
        D = U / nu[..., None]
        P1 = r.X[:, None, :] + np.cumsum(lk[..., None] * D, 1)
        P0 = np.empty_like(P1)
        P0[:, 0] = r.X
        P0[:, 1:] = P1[:, :-1]
        parts = {}
        # bending against the plastic set; the first joint against the follicle direction
        if self.hx_frame:          # the type's helix: the rest turning in the frame carried along the strand now
            parts["bend"], gD, _ = bend_carried(D, r.d0, self.wave_n0c, self.loc[:, :Ka], cb)
        else:
            Dp = np.empty_like(D)
            Dp[:, 0] = r.d0
            Dp[:, 1:] = D[:, :-1]
            dd = D - Dp - r.rho[:, :Ka]
            cdd = cb[..., None] * dd
            parts["bend"] = 0.5 * float((cdd * dd).sum())
            gD = cdd.copy()
            gD[:, :-1] -= cdd[:, 1:]
        # gravity at the segment midpoints, from each guide's root height
        Wk = self.Wk
        parts["gravity"] = float((Wk * (0.5 * (P0[..., 2] + P1[..., 2]) - r.X[:, None, 2])).sum())
        gP0 = np.zeros_like(P0)
        gP1 = np.zeros_like(P1)
        gP0[..., 2] += 0.5 * Wk
        gP1[..., 2] += 0.5 * Wk
        # the samples
        q, t = self.sq, self.st
        P0f = P0.reshape(-1, 3)
        P1f = P1.reshape(-1, 3)
        Y = P0f[q] + t[:, None] * (P1f[q] - P0f[q])
        gY = np.zeros_like(Y)
        fc = fp = None
        parts["contact"] = 0.0
        if self.body is not None:
            if self.c1:
                near = np.nonzero(self.body.sample(Y) < self.h_c + self.margin)[0]
                sdn, gsn = smooth_distance(self.body, Y[near])
                pen_n = self.h_c - sdn
                h_ = pen_n > 0
                hit_i, pen, gs = near[h_], pen_n[h_], gsn[h_]
            else:
                sd, gs_all = self.body.sample(Y, grad=True)
                hit_i = np.nonzero(sd < self.h_c)[0]
                pen, gs = self.h_c - sd[hit_i], gs_all[hit_i]
            if len(hit_i):
                kp = self.k_c * self.sw[hit_i] * pen
                gc = np.zeros_like(Y)
                ek = float(self.f.get("contact_ear_k", 1.0))
                if ek != 1.0:
                    # contact_ear_k: the skin stiffer by this factor in the ears' fine crops (fine_weight blends it in
                    # as the grid blends the crops): 1/2 k_c (1 + (ek - 1) w) w_s pen^2, its gradient with the w term
                    w, gw = fine_weight(self.body, Y[hit_i])
                    m = 1.0 + (ek - 1.0) * w
                    gc[hit_i] = (0.5 * (ek - 1.0) * self.k_c * self.sw[hit_i] * pen * pen)[:, None] * gw
                    kp = kp * m
                parts["contact"] = 0.5 * float((kp * pen).sum())
                gc[hit_i] += -kp[:, None] * gs
                gY += gc
                if forces:
                    fc = gc
        if forces and fc is None:
            fc = np.zeros_like(Y)
        parts["pressure"] = 0.0
        if self.pressure and self.c_a > 0:
            Ep, Ea, gp, _ = self.splat.cohesive(Y, self.sv, self.c_p, self.c_a, self.swf, cl=self.c_l, al=self.sal)
            parts["pressure"] = float(Ep)
            parts["cohesion"] = float(Ea)
            gY += gp
            if forces:
                fp = gp
        elif self.pressure:
            Ep, gp, _ = self.splat.pressure(Y, self.sv, self.c_p)
            parts["pressure"] = float(Ep)
            gY += gp
            if forces:
                fp = gp
        if forces and fp is None:
            fp = np.zeros_like(Y)
        parts["friction"] = 0.0
        if self.friction and self.mu > 0 and len(self.f_on):
            on, nf, mn = self.f_on, self.f_n, self.f_mn
            dl = Y[on] - self.f_anc
            tdl = dl - (dl * nf).sum(1, keepdims=True) * nf
            qq = (tdl * tdl).sum(1)
            sq_ = np.sqrt(1.0 + qq / self.eps ** 2)
            parts["friction"] = float((mn * self.eps * (sq_ - 1.0)).sum())
            gY[on] += (mn / (self.eps * sq_))[:, None] * tdl
        parts["acts"] = 0.0
        gPa = None
        if self.acts is not None and hasattr(self.acts, "energy"):
            v = self._view(P0, P1, D, Y)
            Ea, ga = self.acts.energy(v)
            parts["acts"] = float(Ea)
            ga = ga or {}
            if ga.get("Y") is not None:
                gY += ga["Y"]
            if ga.get("D") is not None:
                gD += ga["D"]
            gPa = ga.get("P")
        # sample forces to the segment end points, then to the directions
        n = G * Ka
        for c in range(3):
            gP1f = np.bincount(q, weights=t * gY[:, c], minlength=n)
            gP0f = np.bincount(q, weights=(1 - t) * gY[:, c], minlength=n)
            gP1[..., c] += gP1f.reshape(G, Ka)
            gP0[..., c] += gP0f.reshape(G, Ka)
        gP1[:, :-1] += gP0[:, 1:]
        if gPa is not None:
            gP1 += gPa[:, 1:]
        gD += lk[..., None] * np.cumsum(gP1[:, ::-1], 1)[:, ::-1]
        gD[~act] = 0.0
        parts["cond"] = 0.5 * float((cb * (nu - 1) ** 2).sum())
        E = sum(parts.values())
        gU = (gD - (gD * D).sum(2, keepdims=True) * D) / nu[..., None] + (cb * (nu - 1))[..., None] * D
        gU[~act] = 0.0
        if forces:
            return E, gU, dict(parts=parts, gD=gD, D=D, P0=P0, P1=P1, Y=Y, fc=fc, fp=fp)
        return E, gU

    def _view(self, P0, P1, D, Y):
        v = View()
        v.P = np.concatenate([P0[:, :1], P1], 1)
        v.D, v.Y = D, Y
        v.sample_seg, v.sample_t, v.sample_w, v.sample_v, v.sample_arc = self.sq, self.st, self.sw, self.sv, self.sarc
        r = self.rods
        v.act, v.l, v.s0, v.Ka = self.act, self.lk, self.s0, self.Ka
        v.B, v.lbar, v.W = r.B[:, :self.Ka], r.lbar[:, :self.Ka], self.Wk
        v.rods, v.state, v.stage = r, self, self.stage_index
        return v

    def view(self):
        """the current configuration as an act sees it"""
        D = self.rods.D[:, :self.Ka]
        P1 = self.rods.X[:, None, :] + np.cumsum(self.lk[..., None] * D, 1)
        P0 = np.concatenate([self.rods.X[:, None, :], P1[:, :-1]], 1)
        P0f, P1f = P0.reshape(-1, 3), P1.reshape(-1, 3)
        Y = P0f[self.sq] + self.st[:, None] * (P1f[self.sq] - P0f[self.sq])
        return self._view(P0, P1, D, Y)

    # -------------------------------------------------- solves, lag, gate
    def solve(self, evals, gated=False, label=""):
        """one L-BFGS-B solve of the active segments with a fixed evaluation cap, in variables scaled from the
        configuration at its start (precondition); gated: stop once the gate passes"""
        t0 = time.time()
        self.precondition()
        x0 = self.x_of()
        self.trace = []
        best = [np.inf]
        last = {}

        on_gpu = getattr(self, "_gpu", None) is not None and self.F is not None

        def f(x):
            te = time.time()
            if on_gpu:             # the whole objective on the card (gpu.CoreGPU.fun): only x and the gradient cross;
                # gated, the evaluation's forces stay there until the gate reads them (CoreGPU.kept)
                Es, g = self._gpu.fun(self, x, keep=gated)
                self.evals += 1
                best[0] = min(best[0], Es)
                self.trace.append(best[0])
                last["x"] = x
                self.eval_s += time.time() - te
                return Es, g
            U = self._U(x)
            # gated: keep the term forces of this evaluation for the gate's measure (core returns them at no cost)
            E, gU, o = self.core(U, forces=True) if gated else (*self.core(U), None)
            g = self._gx(gU)
            Es = E * self.ESC
            self.evals += 1
            best[0] = min(best[0], Es)
            self.trace.append(best[0])
            last["x"], last["U"], last["gU"], last["o"] = x, U, gU, o
            self.eval_s += time.time() - te
            return Es, g

        cb = None
        nacc = [0]
        if gated:
            def cb(intermediate_result):
                # the accepted iterate is (almost always) the last evaluated point, whose gradient is cached. L-BFGS
                # accepts nearly every evaluation and the gate's per-term chain costs ~40 % of one on a head, so it is
                # checked at every 5th accepted iterate (the stop comes at most 5 evaluations late)
                nacc[0] += 1
                if nacc[0] % 5 == 0 and len(self.trace) > 21 and np.array_equal(intermediate_result.x, last.get("x")):
                    U_, gU_, o_ = self._gpu.kept() if on_gpu else (last["U"], last["gU"], last["o"])
                    if self._gate_from(U_, gU_, o_)["passed"]:
                        raise StopIteration
        e0 = None
        res = minimize(f, x0, jac=True, method="L-BFGS-B", callback=cb,
                       options=dict(maxiter=int(evals), maxfun=int(evals), maxcor=int(self.f["maxcor"]), ftol=0.0,
                                    gtol=0.0))
        e0 = self.trace[0] if self.trace else None
        U = self._U(res.x)
        D = _unit(U)
        Dfull = self.rods.D
        Dk = Dfull[:, :self.Ka]
        Dk[self.act] = D[self.act]
        h = dict(stage=self.stage_index, label=label, Ka=self.Ka, evals=len(self.trace), E0=e0, E1=float(res.fun),
                 message=str(res.message)[:60], seconds=round(time.time() - t0, 2))
        self.hist.append(h)
        self.log("  %-10s Ka %2d  %4d evals  E %.6g -> %.6g  (%.1f s)  %s" % (
            label, self.Ka, h["evals"], e0 if e0 is not None else np.nan, h["E1"], h["seconds"], h["message"]))
        return res

    def lag(self, acts_too=True):
        """store the friction anchors (the samples now) and the normal forces |body + hair-hair force| on them; update
        the acts' lagged data"""
        U = self.rods.D[:, :self.Ka]
        _, _, o = self.core(U, forces=True)
        fn = o["fc"] + o["fp"]
        N = np.sqrt((fn * fn).sum(1))
        self.anc[self.sidx] = o["Y"]
        Na = self._adhesive(o["Y"], N) if self.adh is not None else 0.0
        self.anc_N[self.sidx] = N + Na
        # the coefficient of the interface that carries the load: a sample on the skin is held by the skin's friction
        # (mu_s) for the share of its load the body's contact carries, by the hair-hair friction (mu) for the share
        # the neighbours' pressure carries; the oil bridges press hair onto hair (mu). The load N is unchanged.
        Nc = np.sqrt((o["fc"] * o["fc"]).sum(1))
        Np = np.sqrt((o["fp"] * o["fp"]).sum(1))
        sc = Nc / np.maximum(Nc + Np, 1e-300)
        self.anc_M[self.sidx] = (self.mu_s * sc + self.mu * (1.0 - sc)) * N + self.mu * Na
        self.anc_n[self.sidx] = fn / np.maximum(N, 1e-300)[:, None]
        self._anchors()
        if self.hx_frame:          # the acts read and edit rho as it is now (the energy does not keep it current)
            self.wave_frame()
        if acts_too and self.acts is not None and hasattr(self.acts, "lag"):
            self.acts.lag(self.view())
        self.wave_frame()

    def _adhesive(self, Y, N):
        """the oil bridges' normal load on each active sample (N), lagged with the anchors: the members' fibre length in
        the sample (its fibre volume / pi r^2) x fibre.adhesive_load at the sample's arc and fibre fraction (the splat's
        phi, gathered with the splat's own B-spline weights). Only within friction_reach_mm of the body (the layer the
        body holds); N, the contact and pressure load, is passed for the report. The friction keeps the contact /
        pressure normal for its tangent plane (a layer slides along itself), so the energy's form is unchanged."""
        fb = self.rods.fibre
        if self.splat is None:
            return np.zeros(len(Y))
        _, _, ph = self.splat.pressure(Y, self.sv, self.c_p)
        phi = self.splat.gather(ph)
        on = np.ones(len(Y), bool) if self.body is None else self.body.sample(Y) < self.h_c + self.reach
        Na = np.where(on, fb.adhesive_load(self.sarc, phi) * self.sv / (np.pi * fb.r_root ** 2), 0.0)
        hit = Na > 0
        w = np.maximum(self.sw, 1e-300)
        self.adh_rep.append({
            "stage": int(self.stage_index), "samples": int(len(Y)), "with_bridges": int(hit.sum()),
            "phi_p50_p90": [round(float(np.percentile(phi[hit], q)), 4) for q in (50, 90)] if hit.any() else None,
            "adhesive_over_weight_p50_p90": [round(float(np.percentile((Na / w)[hit], q)), 2) for q in (50, 90)]
            if hit.any() else None,
            "adhesive_over_contact_pressure_p50": round(float(np.median(Na[hit] / np.maximum(N[hit], 1e-300))), 2)
            if hit.any() else None})
        return Na

    def gate_check(self, U=None):
        """the convergence gate (DESIGN_v2 4.5): max unbalanced node force / local segment weight < gate, and the
        relative energy change over the last 20 evaluations < gate_de"""
        U = self.rods.D[:, :self.Ka] if U is None else U
        _, gU = self.core(U)
        return self._gate_from(U, gU)

    def _gate_from(self, U, gU, o=None):
        """the gate from a configuration and its gradient: the conditioning term is radial (along d), so the
        transverse part of g_d is |u| times the transverse part of g_u"""
        nu = np.where(self.act, np.sqrt((U * U).sum(2)), 1.0)
        D = U / nu[..., None]
        lk = np.where(self.act, self.lk, 1.0)
        tau = (gU - (gU * D).sum(2, keepdims=True) * D) * (nu / lk)[..., None]
        tau[~self.act] = 0.0
        tz = np.concatenate([tau, np.zeros((self.rods.G, 1, 3))], 1)
        res = tz[:, :-1] - tz[:, 1:]                        # node j + 1 (between segment j and j + 1); tip included
        Wz = np.concatenate([self.Wk, np.zeros((self.rods.G, 1))], 1)
        wl = 0.5 * (Wz[:, :-1] + Wz[:, 1:])
        # a node's unbalanced force against the forces it balances: its weight, or the sum of each term's force on
        # it when larger. At the graded root (0.5 mm segments, ~1/1000 of a guide's weight) the follicle moment, the
        # guide's tension and the scalp contact each pass ~25 local weights through a node, and balancing them to
        # 0.05 local weight is worth ~1e-17 J, under the float64 resolution of a head's energy (5e-17 J; measured,
        # 1466 guides). In a freely hanging chain the sum is the local weight, as before.
        ratio = np.sqrt((res * res).sum(2)) / np.maximum(np.maximum(wl, self._gross(U, o)), 1e-300)
        ratio = ratio[self.act]
        tr = self.trace
        k = min(21, len(tr))                                # a solve that stopped converged early has fewer
        de = abs(tr[-k] - tr[-1]) / max(abs(tr[-1]), 1e-300) if k >= 2 else float("nan")
        mx = float(ratio.max()) if ratio.size else 0.0
        out = dict(max_force_ratio=mx, p99_force_ratio=float(np.percentile(ratio, 99)) if ratio.size else 0.0,
                   p50_force_ratio=float(np.median(ratio)) if ratio.size else 0.0, rel_dE_20=float(de),
                   gate=self.f["gate"], gate_de=self.f["gate_de"])
        out["passed"] = bool(mx < self.f["gate"] and de == de and de < self.f["gate_de"])
        return out

    def _gross(self, U, o=None):
        """per node, the sum over the terms (bending, gravity, contact, pressure, friction) of the magnitude of each
        term's unbalanced force, through the gate's own chain; acts are left out (a smaller sum is stricter)"""
        r = self.rods
        G, Ka = r.G, self.Ka
        if o is None:
            _, _, o = self.core(U, forces=True)
        D, P0, Y = o["D"], o["P0"], o["Y"]
        if self.hx_frame:
            gb = bend_carried(D, r.d0, self.wave_n0c, self.loc[:, :Ka], self.cb)[1]
        else:
            Dp = np.concatenate([r.d0[:, None], D[:, :-1]], 1)
            cdd = self.cb[..., None] * (D - Dp - r.rho[:, :Ka])
            gb = cdd.copy()
            gb[:, :-1] -= cdd[:, 1:]
        gw = np.zeros_like(P0)
        gw[..., 2] = 0.5 * self.Wk
        gf = np.zeros_like(Y)
        if self.friction and self.mu > 0 and len(self.f_on):
            on, nf = self.f_on, self.f_n
            dl = Y[on] - self.f_anc
            tdl = dl - (dl * nf).sum(1, keepdims=True) * nf
            gf[on] = (self.f_mn / (self.eps * np.sqrt(1.0 + (tdl * tdl).sum(1) / self.eps ** 2)))[:, None] * tdl
        zP, zD = np.zeros_like(P0), np.zeros_like(D)
        lk = np.where(self.act, self.lk, 1.0)
        out = np.zeros((G, Ka))
        for gD0, gP0, gP1, gY in ((gb, zP, zP, None), (zD, gw, gw, None), (zD, zP, zP, o["fc"]),
                                  (zD, zP, zP, o["fp"]), (zD, zP, zP, gf)):
            gP0, gP1 = gP0.copy(), gP1.copy()
            if gY is not None:
                for c in range(3):
                    gP1[..., c] += np.bincount(self.sq, weights=self.st * gY[:, c], minlength=G * Ka).reshape(G, Ka)
                    gP0[..., c] += np.bincount(self.sq, weights=(1 - self.st) * gY[:, c],
                                               minlength=G * Ka).reshape(G, Ka)
            gP1[:, :-1] += gP0[:, 1:]
            gD = gD0 + self.lk[..., None] * np.cumsum(gP1[:, ::-1], 1)[:, ::-1]
            tau = (gD - (gD * D).sum(2, keepdims=True) * D) / lk[..., None]
            tau[~self.act] = 0.0
            tz = np.concatenate([tau, np.zeros((G, 1, 3))], 1)
            out += np.sqrt(((tz[:, :-1] - tz[:, 1:]) ** 2).sum(2))
        return out

    def energies(self):
        U = self.rods.D[:, :self.Ka]
        E, _, o = self.core(U, forces=True)
        return E, o


# ---------------------------------------------------------------- the public steps
def energy(x, state):
    """the scaled energy of the state's active segments at Jacobi-scaled variables x, and its gradient"""
    U = state._U(np.asarray(x, np.float64))
    E, gU = state.core(U)
    return E * state.ESC, state._gx(gU)


def stage_arcs(rods, stages_mm):
    """the growth stages (m): those of stages_mm shorter than the longest guide, then the longest guide"""
    Lmax = float(rods.L.max())
    arcs = [a / 1000.0 for a in stages_mm if a / 1000.0 < Lmax - 1e-9]
    return arcs + [Lmax + 1e-9]


def grow(rods, body=None, acts=None, stages_mm=None, cfg=None, pressure=True, friction=True, splat=None, log=print,
         state=None):
    """grow the guides by continuation (DESIGN_v2 4.6): stage by stage the active arc lengthens, new segments start
    hanging and pushed out of the body, 1-2 sub-solves settle them (the acts' lagged data updated between), then the
    friction anchors and normal forces are stored for the next stage. -> State"""
    st = state or State(rods, body, cfg, pressure, friction, acts, splat, log)
    f = st.f
    arcs = stage_arcs(rods, f["stages_mm"] if stages_mm is None else stages_mm)
    nsub = max(1, int(f["subsolves"]))
    per = max(2, int(round(f["evals_stage"] / nsub)))
    for si, arc in enumerate(arcs):
        st.stage_index = si
        nnew = st.activate(arc)
        if st.acts is not None and hasattr(st.acts, "stage"):
            st.acts.stage(st.view(), si, arc)
        st.wave_frame()                                   # the new segments' rest turning in their own frame
        st.log(" stage %d: arc %.0f mm, %d segments switched on, %d active, %d samples" % (
            si, 1000 * min(arc, rods.L.max()), nnew, int(st.act.sum()), len(st.sidx)))
        for k in range(nsub):
            st.solve(per, label="grow %d.%d" % (si, k))
            if k < nsub - 1 and st.acts is not None and hasattr(st.acts, "lag"):
                st.acts.lag(st.view())
            if k < nsub - 1:
                st.wave_frame()
        st.lag()
    return st


def grow_sections(rods, sec_g, body=None, acts=None, stages_mm=None, cfg=None, pressure=True, friction=True,
                  splat=None, log=print, hold=None, stick=None, drop_evals=45, state=None):
    """styling in sections (DESIGN_sections 3.2): a stylist works the head section by section from the bottom up -
    each section is grown (combed out), brushed, held while it dries and cools, let go and dropped onto the sections
    already laid, which stay live (they answer the new section's weight). The last-laid section lies on top because
    it was laid last, never by a rule. sec_g (G,) int: the guide's section, worked in the order 0, 1, ...; -1 = none
    (grown with the first section). acts.on_g (if acts has it) is the section being worked, so the brush, its through
    hold and the face barrier act on that section only. hold(state, on) holds and sets the section (actions'
    dry_set + dry_hold); stick(state) is the friction lag (actions.lag_stick: anchors kept by the hair that stuck);
    None = State.lag. The drop is drop_evals evaluations in 2 solves with the friction lag between. -> State with
    .sections: per section its guides, evaluations and its samples' positions at the end of its drop (slip
    measure)"""
    st = state or State(rods, body, cfg, pressure, friction, acts, splat, log)
    f = st.f
    lag = stick or (lambda s: s.lag())
    nsub = max(1, int(f["subsolves"]))
    per = max(2, int(round(f["evals_stage"] / nsub)))
    sec_g = np.asarray(sec_g).copy()
    sec_g[sec_g < 0] = 0
    full = rods.L + 1e-9
    laid = np.zeros(rods.G, bool)
    st.sections = []
    st.sections_set = []
    si = -1
    for k in range(int(sec_g.max()) + 1):
        on = sec_g == k
        if not on.any():
            continue
        e0 = st.evals
        if acts is not None:
            acts.on_g = on.copy()
        sub = Rods.__new__(Rods)
        sub.L = rods.L[on]
        arcs = stage_arcs(sub, f["stages_mm"] if stages_mm is None else stages_mm)
        for ai, arc in enumerate(arcs):
            si += 1
            st.stage_index = si
            arc_g = np.where(laid, full, np.where(on, arc, 0.0))
            nnew = st.activate(arc_g)
            if st.acts is not None and hasattr(st.acts, "stage"):
                st.acts.stage(st.view(), si, arc)
            st.wave_frame()
            st.log(" section %d stage %d: arc %.0f mm, %d segments switched on, %d active, %d samples" % (
                k, ai, 1000 * min(arc, sub.L.max()), nnew, int(st.act.sum()), len(st.sidx)))
            for j in range(nsub):
                st.solve(per, label="section %d grow %d.%d" % (k, ai, j))
                if j < nsub - 1 and st.acts is not None and hasattr(st.acts, "lag"):
                    st.acts.lag(st.view())
                if j < nsub - 1:
                    st.wave_frame()
            lag(st)
        if hold is not None:
            st.sections_set.append(hold(st, on) or 0)
        # let go: the section is out of the brush and drops onto the ones under it
        if acts is not None:
            acts.on_g = np.zeros(rods.G, bool)
            if hasattr(acts, "lag"):
                acts.lag(st.view())
        laid |= on
        if drop_evals > 0:
            for j in range(2):
                st.wave_frame()
                st.solve(max(2, int(round(drop_evals / 2))), label="section %d drop %d" % (k, j))
                lag(st)
        v = st.view()
        g_s = st.sq // st.Ka
        st.sections.append({"section": k, "guides": int(on.sum()), "evals": int(st.evals - e0),
                            "sidx": st.sidx[on[g_s]].copy(), "Y": v.Y[on[g_s]].copy()})
    return st


def section_slip(state):
    """how far each laid section moved (mm) from the end of its drop to now, the same samples matched by sample index:
    the whole movement and its tangential part (the slip along the skin or the layer under it: the movement less its
    part along the body's distance gradient at the sample, so a layer pressed down by the one laid on it is not
    counted as slip; DESIGN_sections T4), p50 / p95 / max each. -> list per section"""
    out = []
    if not getattr(state, "sections", None):
        return out
    pos = {int(i): j for j, i in enumerate(state.sidx)}
    Y = state.view().Y
    for s in state.sections:
        j = np.array([pos.get(int(i), -1) for i in s["sidx"]])
        ok = j >= 0
        if not ok.any():
            out.append({"section": s["section"], "samples": 0})
            continue
        dv = Y[j[ok]] - s["Y"][ok]
        d = 1000 * np.sqrt((dv * dv).sum(1))
        row = {"section": s["section"], "guides": s["guides"], "evals": s["evals"], "samples": int(ok.sum()),
               "move_mm_p50_p95_max": np.round([np.percentile(d, 50), np.percentile(d, 95), d.max()], 2).tolist()}
        if state.body is not None:
            _, g = state.body.sample(Y[j[ok]], grad=True)
            n = _unit(g)
            tv = dv - (dv * n).sum(1, keepdims=True) * n
            dt = 1000 * np.sqrt((tv * tv).sum(1))
            row["slip_mm_p50_p95_max"] = np.round([np.percentile(dt, 50), np.percentile(dt, 95), dt.max()], 2).tolist()
        out.append(row)
    return out


def release(state, acts=None, log=None):
    """release (DESIGN_v2 4.7): every act off except what acts.release() keeps (tie pins), release_solves lagged
    sub-solves, then the final solve, gated (it stops once the gate passes, at most evals_final evaluations)"""
    st = state
    if log is not None:
        st.log = log
    acts = acts if acts is not None else st.acts
    if acts is not None and hasattr(acts, "release"):
        acts.release()
    st.acts = acts
    st.stage_index += 1
    f = st.f
    nrel = max(1, int(f["release_solves"]))
    if f["evals_release"] > 0:
        per = max(2, int(round(f["evals_release"] / nrel)))
        for k in range(nrel):
            st.solve(per, label="release %d" % k)
            st.lag()
    # the final stage: the friction anchors stay; the scaling is rebuilt every final_chunk evaluations (contacts
    # change as the hair slides, and a stale scaling slows L-BFGS down), and it stops once the gate passes
    left = int(f["evals_final"])
    chunk = max(25, int(f["final_chunk"]))
    k = 0
    while left > 0:
        n = min(chunk, left)
        st.wave_frame()                                   # the rest turning follows the strand between chunks
        st.solve(n, gated=True, label="final %d" % k)
        left -= len(st.trace)
        k += 1
        st.gate = st.gate_check()
        if st.gate["passed"] or len(st.trace) < 5:          # passed, or the line search can make no more progress
            break
    st.gate = st.gate_check()
    st.log(" gate: %s (max force / local weight %.4f, p99 %.4f; energy change over 20 evaluations %.2e)" % (
        "PASSED" if st.gate["passed"] else "NOT PASSED", st.gate["max_force_ratio"], st.gate["p99_force_ratio"],
        st.gate["rel_dE_20"]))
    return st


# ---------------------------------------------------------------- the report
def _sph(c, P):
    v = P - c
    r = np.sqrt((v * v).sum(-1))
    phi = np.degrees(np.arctan2(v[..., 0], -v[..., 1]))
    th = np.degrees(np.arccos(np.clip(v[..., 2] / np.maximum(r, 1e-12), -1, 1)))
    return phi, th, r


def _region_masks(phi, th, z, frame):
    a = np.abs(phi)
    cd = frame["crown_deg"]
    up = z >= frame["z_hang"]
    return {"crown": (th < cd) & up,
            "back": (a > 120) & (th >= cd) & up,
            "sides": (a > 50) & (a <= 120) & (th >= cd) & up,
            "front": (a <= 50) & (th >= cd) & up,
            "hanging": ~up}


def default_frame(rods, frame=None):
    fr = dict(frame or {})
    fr.setdefault("centre", rods.X.mean(0))
    fr["centre"] = np.asarray(fr["centre"], np.float64)
    fr.setdefault("crown_deg", 40.0)
    fr.setdefault("z_hang", float(rods.X[:, 2].min()) - 0.01)
    fr.setdefault("near_mm", 40.0)
    fr.setdefault("chin_z", None)
    return fr


def fibre_fraction(state, frame=None, Y=None, v=None):
    """fibre fraction phi on the splat lattice by region, weighted by fibre volume (the packing a typical fibre sits
    in): p50 / p95, the voxel maximum, and the share of fibre in voxels over 0.3. Regions by the node's position about
    the head centre: crown (theta < crown_deg), back, sides, front (the rest of the head, within near_mm of the skin)
    and the hanging lengths (below z_hang)"""
    fr = default_frame(state.rods, frame)
    if Y is None:
        Y, v = state.view().Y, state.sv
    body = state.body
    # measured with the kernel the pressure energy uses
    sp = make_splat(body, state.f, state.rods)
    F = sp.field(Y, v)
    ph = sp.phi(F).astype(np.float64)
    occ = F > 0
    idx = np.argwhere(occ)
    Q = sp.lo + sp.h * (idx + sp.c0)
    Fw = F[occ].astype(np.float64)
    pv = ph[occ]
    sd = body.sample(Q) if body is not None else np.full(len(Q), np.inf)
    phi, th, _ = _sph(fr["centre"], Q)
    masks = _region_masks(phi, th, Q[:, 2], fr)
    out = {"lattice_mm": round(sp.h * 1000, 3), "voxel_max": round(float(pv.max()), 4) if len(pv) else 0.0,
           "fibre_share_over_0.3": round(float(Fw[pv > 0.3].sum() / max(Fw.sum(), 1e-300)), 5),
           "voxels_over_0.3": int((pv > 0.3).sum())}
    near = fr["near_mm"] / 1000.0
    for k, m in masks.items():
        if k != "hanging":
            m = m & (sd < near)
        m = m & (sd > -0.5 * sp.h)
        if m.sum() < 3:
            out[k] = None
            continue
        p50, p95 = wpercentile(pv[m], Fw[m], [50, 95])
        out[k] = {"p50": round(p50, 4), "p95": round(p95, 4), "max": round(float(pv[m].max()), 4),
                  "fibre_mm3": round(float(Fw[m].sum() * 1e9), 1)}
    return out


def layers(state, frame=None):
    """the hair's outer surface off the skin by region: per 5 mm cell of the samples' foot points (the closest skin
    point, from the body's distance and gradient) the 95th percentile height, then p10 / p50 / p90 over the cells
    with at least 3 samples (samples within 60 mm of the skin); the root loops (each guide's greatest height over its
    first 60 mm of arc, guides longer than 200 mm); the curtain behind the head at chin height (horizontal radius
    from the centre's vertical axis, p50 / p90)"""
    fr = default_frame(state.rods, frame)
    v = state.view()
    Y = v.Y
    c = fr["centre"]
    out = {}
    if state.body is not None:
        sd, g = state.body.sample(Y, grad=True)
        foot = Y - sd[:, None] * _unit(g)
        phi, th, _ = _sph(c, foot)
        masks = _region_masks(phi, th, foot[:, 2], fr)
        ok = (sd < 0.06) & (sd > -0.01)
        cell = np.floor(foot / 0.005).astype(np.int64)
        key = (cell[:, 0] * 1000003 + cell[:, 1]) * 1000003 + cell[:, 2]
        for k in ("crown", "back", "sides", "front"):
            m = masks[k] & ok
            if m.sum() < 10:
                out[k] = None
                continue
            kk, h = key[m], sd[m] * 1000
            o = np.argsort(kk, kind="stable")
            kk, h = kk[o], h[o]
            cuts = np.nonzero(np.diff(kk))[0] + 1
            tops = [np.percentile(x, 95) for x in np.split(h, cuts) if len(x) >= 3]
            if len(tops) < 3:
                out[k] = None
                continue
            out[k] = {"layer_top_cells_p10_p50_p90_mm": np.round(np.percentile(tops, [10, 50, 90]), 2).tolist(),
                      "cells": len(tops), "samples_p90_mm": round(float(np.percentile(h, 90)), 2)}
        # root loops of the long guides
        r = state.rods
        gi = state.sq // state.Ka
        early = state.sarc < 0.06
        hmax = np.full(r.G, -np.inf)
        np.maximum.at(hmax, gi[early], sd[early])
        longg = r.L > 0.2
        _, thr, _ = _sph(c, r.X)
        top = longg & (thr < fr["crown_deg"] + 20)                 # rooted on the convex top of the head
        if longg.any():
            out["root_loop_max_first60_long_guides_p10_p50_p90_mm"] = np.round(
                np.percentile(hmax[longg] * 1000, [10, 50, 90]), 2).tolist()
        if top.sum() >= 3:
            out["root_loop_max_first60_long_top_guides_p10_p50_p90_mm"] = np.round(
                np.percentile(hmax[top] * 1000, [10, 50, 90]), 2).tolist()
            out["root_loop_top_guides"] = int(top.sum())
    if fr.get("chin_z") is not None:
        zc = fr["chin_z"]
        m = (np.abs(Y[:, 2] - zc) < 0.01) & (Y[:, 1] - c[1] > 0.02)
        if m.sum() >= 10:
            rr = np.hypot(Y[m, 0] - c[0], Y[m, 1] - c[1]) * 1000
            out["curtain_behind_chin_radius_p50_p90_mm"] = np.round(np.percentile(rr, [50, 90]), 1).tolist()
            out["curtain_samples"] = int(m.sum())
    return out


def _aligned_report(state):
    """the aligned count's numbers: sigma_0, the line share A by sample and the mixed phi* it gives; None when off"""
    if getattr(state, "c_l", 0.0) <= 0 or state.sal is None or not len(state.sal):
        return None
    fb = state.rods.fibre
    pq = lambda a: [round(float(np.percentile(a, q)), 4) for q in (10, 50, 90)]
    return {"sigma0_pa": round(float(state.c_l), 2), "alignment_p10_p50_p90": pq(state.sal),
            "phi_star_mixed_p10_p50_p90": pq(fb.phi_star_mixed(state.sarc, state.sal)),
            "phi_lock_p10_p50_p90": pq(fb.phi_lock(state.sarc))}


def report(state, frame=None):
    """energies by term (J, and per guide in guide-weight mm), the gate, the follicle clamp, fibre fraction by region,
    the layer tops and the curtain, the evaluations and their time"""
    r = state.rods
    E, o = state.energies()
    sc = state.ESC
    D = r.D
    ang = np.degrees(np.arccos(np.clip((D[:, 0] * r.d0).sum(1), -1, 1)))
    rep = {"guides": r.G, "members": int(len(r.mg)), "segments_max": r.K, "segments_active": int(state.act.sum()),
           "samples": int(len(state.sidx)), "guide_length_p50_max_mm": [round(float(np.median(r.L) * 1000), 1),
                                                                        round(float(r.L.max() * 1000), 1)],
           "hairs_per_guide_root_p50": float(np.median(r.n[:, 0])),
           "pressure": state.pressure, "friction": state.friction,
           "friction_adhesion": state.adh_rep[-1] if state.adh_rep else None,
           "cohesion": {"c_a0_pa": round(float(state.c_a), 2), "c_p_pa": round(float(state.c_p), 2),
                        "wetting_p10_p50_p90": [round(float(np.percentile(state.swf, q)), 3) for q in (10, 50, 90)],
                        "phi_star_p10_p50_p90": [round(float(np.percentile(state.c_a * state.swf ** 2 / (2 * state.c_p),
                                                                           q)), 4) for q in (10, 50, 90)]}
           if state.c_a > 0 and state.swf is not None and len(state.swf) else None,
           "cohesion_aligned": _aligned_report(state),
           "energies_J": {k: float(v) for k, v in o["parts"].items()},
           "energies_per_guide_weight_mm": {k: round(float(v) * sc, 4) for k, v in o["parts"].items()},
           "energy_total_scaled": round(float(E) * sc, 6),
           "gate": state.gate,
           "clamp_first_segment_vs_d0_deg_p50_p90_max": [round(float(np.median(ang)), 3),
                                                         round(float(np.percentile(ang, 90)), 3),
                                                         round(float(ang.max()), 3)],
           "evals": state.evals, "eval_seconds": round(state.eval_s, 2),
           "seconds_per_eval": round(state.eval_s / max(state.evals, 1), 4),
           "seconds": round(time.time() - state.t0, 1), "stages": state.hist,
           "stand_ins_used": list(STAND_INS_USED)}
    rep["fibre_fraction"] = fibre_fraction(state, frame)
    rep["layers"] = layers(state, frame)
    return rep
