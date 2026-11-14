"""engine/hair/actions.py - the stylist's acts as causes in the settle (DESIGN_v2 section 5, steps 6 and 9).

A hairdresser does not place hair: she brushes it along a stroke, parts it, dries it in the brushed shape, tucks it
behind an ear, lays the lengths in front of the shoulders with a hand, shakes it out with her fingers. Each act here
is one of those causes, written as an energy (or a lagged pin) the settle (settle.py) minimises together with the
hair's own bending, weight, contact with the body, hair-hair pressure and friction. None of them names a height, an
outline or a lift: the shape is what the material does under the acts.

The acts and how they enter (settle.py's acts hook: energy, stage, lag, precond, release):
- brush: bristles drag the hair along the stroke; they decide where it goes ALONG the head (its heading), never how
  far it stands off it. Per segment E = 1/2 kappa_b (B / lbar) a (1 - cos psi), cos psi = (d_t . b) / sqrt(|d_t|^2 +
  eps^2), d_t = d - (d . n) n the segment's direction in the skin's tangent plane, b the unit stroke tangent, eps 0.05
  (the heading is undefined within ~3 deg of the skin normal; there the term turns the strand toward b). a = 1 -
  smoothstep(reach / 2, reach, SDF) is the bristles' reach (bristle length 10-20 mm). n, b and a are lagged (taken at
  the start of each sub-solve): a live a(SDF) would give a spurious outward push on any misaligned segment (dE/dSDF <
  0), and a brush does not push hair off the head. `brush.mode` "direction" is the old comb model (the stroke pulls the
  whole direction, cos psi = d . b, elevation included) and exists only for the A7 comparison.
- the stroke field b: flow.comb_field's style terms only (flow.away_from_part by part side in the part zone, the crown
  term away from the follicle whorl W, flow.back / side / up / face_back) plus brush.down_sides (the side strokes go
  down), brush.toward_gather (to the tie) and a coherent brush.noise_deg rotation about the normal (finger-combing).
  Not taken from comb_field: its ear tuck, sideburn, temple sweep, volume fan and face corridor force - the ears are
  contact, the tuck is its own act, and the face is a barrier and a report.
- part: brushing away from a line on both sides. part_side() gives each root its side (cross_frac roots near the
  line flip, the faint line); children must use the same sides and never interpolate across.
- dry / set (dry_set): hydrogen bonds re-form in the brushed shape. Over s <= dry.over_mm, rho <- rho + set (dd -
  rho), dd the current turning at each joint; dry.lift_deg adds a rest turning toward the skin normal over the same
  span (a round-brush blow-dry; 0 when air-dried).
- face barrier: a hand keeps hair off the face while it is brushed: 1/2 k (margin - SDF)_+^2 on the samples over the
  face skin (k the settle's contact stiffness x the sample's weight), growth only.
- tuck: a hand lifts the strands that pass over or in front of an ear over its top and lays them behind it: a soft
  pin at the node where the strand crosses the ear, first to a point over the ear's top, then (once there) behind the
  ear's top (tuck.behind_mm, tuck.over_mm). Released after the first release solve; the ear's contact then holds it.
- hand: the lengths of the roots with |phi| < hand.phi_deg are held in front of the shoulder plane (a one-sided
  penalty on their nodes below the shoulders' top) for hand.stages growth stages counted from the first stage their
  lengths reach the shoulders, then released; shoulders and chest hold them by contact.
- tousle: after release, tousle.strokes lagged sub-solves under a seeded smooth random force (tousle.force x the
  segment's weight) over the crown, each stroke a new field, then released.
- release: every act off except pins that stay (tie pins; the tuck until the first release solve is done).
- the stroke drawn through (brush.through): a brush is drawn from the root to the ends, and its bristles hold the hair
  wherever the hair is, not only within reach of the skin. The brush's path is traced once per guide from its root
  along the stroke field on the skin (over the top and down the back from the hairline; at the sides, with tuck.on,
  back over the ear's top and behind it - the tuck is the side stroke's end, no pin) and off the lengths under
  gravity where the skin turns under. Each node is held within brush.reach_mm (the bristles) of the brush's position
  at its own rest arc: 1/2 k (|P - Q(s)| - reach)_+^2, k = the segment's weight / through_sag_mm. As the strand grows
  by continuation its new length is carried along the path (the brush advances from root to tip). No height, outline
  or region is named: within the bristles the hair's own bending, weight, contact and pressure decide.
- the front sections of a parted look (brush.part_section, tuck.fall): a section parted off at the front is brushed
  as one along its whole stroke - away from the part, down the side over the temple and back to the ear, where the
  tuck lays it behind the ear - and, coming to the ear from above, it falls down the side under its own weight while
  the hand leads it back (the path in front of the ear is the straight line to the point over the helix), instead of
  being carried back above the temple at its height. Both off by default (earlier runs unchanged).
- dry.hold: the section stays in the brush while it cools with its new rest shape (dry_hold), then is let go; the
  round brush's set (blow_set) never bends a joint tighter than 1 / R (R the brush's radius).

Sources: hair_styling.md section 7 (each technique as a cause, the ranges below), DESIGN_v2 sections 5, 7 and 8;
Robbins, Chemical and Physical Behavior of Human Hair (5th ed., 2012) ch. 8-9 for the water set (hydrogen bonds
re-formed on drying; set retained 10-40 % air-dried, ~40-70 % blow-dried under tension).
"""
import copy
import os

import numpy as np
from scipy.spatial import cKDTree

from . import settle as SE

# style blocks in style.py's format: (default, (lo, hi) | [choices] | None, note)
ACTS_DEFAULTS = {
    "brush": {
        "on": (True, None, "brush during growth (released before the final settle)"),
        "strength": (3.0, (0.0, 20.0), "kappa_b: the brush's pull on the heading, in units of the segment's own bending "
                                       "stiffness B / lbar (DESIGN_v2 5; looks use 1.5 finger-combed - 4 sleek)"),
        "reach_mm": (15.0, (2.0, 40.0), "bristle length: full drag within reach / 2 of the skin, none beyond reach "
                                        "(paddle and cushion brushes 10-20 mm; a fine comb 5 mm; hair_styling 7 T1)"),
        "eps": (0.05, (0.005, 0.3), "heading regulariser: below ~eps rad off the skin normal the heading is undefined"),
        "down_sides": (0.5, (0.0, 3.0), "the side strokes go down (over |phi| 50-130, below theta 55)"),
        "lengths": (0.25, (0.0, 2.0), "every stroke ends going down the lengths (a brush is drawn to the ends): a "
                                      "downward stroke term, which only acts where the skin is not level"),
        "toward_gather": (0.0, (0.0, 10.0), "strokes toward the tie (gathered looks)"),
        "noise_deg": (0.0, (0.0, 90.0), "coherent random turn of the stroke about the normal (finger-combing)"),
        "noise_mm": (40.0, (5.0, 200.0), "length scale of that turn"),
        "mode": ("heading", ["heading", "direction"], "heading: bristles act on the heading only (DESIGN_v2 5); "
                                                      "direction: the old comb model, the whole direction pulled "
                                                      "(kept only for the A7 comparison)"),
        "part_section": (False, None, "a parted look's front sections are drawn away from the part along their whole "
                                      "stroke - down the side over the temple and back to the ear (where the tuck "
                                      "takes them behind it), not straight back over the crown once the brush leaves "
                                      "the part zone: each strand keeps its root's part-zone weight on the "
                                      "away_from_part term in front of the ears' back edge and above their lower end. "
                                      "With a centre part the top hair falls away from the part under its own weight "
                                      "and is laid down and back to behind the ear (hair_styling 7 T1, T3: the "
                                      "section is brushed as one along its stroke)"),
        "through": (False, None, "the stroke drawn through to the ends (DESIGN_v2 5): the brush travels a path from "
                                 "the root over the head along the stroke field and off the lengths under gravity; "
                                 "its bristles hold each point of the strand within reach_mm of the brush at the "
                                 "same arc, wherever the hair is - not only near the skin"),
        "through_hold": ("steer", ["none", "steer", "position"], "what the bristles hold while the stroke is drawn "
                                                         "through. 'none' (10-31: brushed hair is not pinned to the scalp; "
                                                         "it stays in place through physics, its characteristics, oil, "
                                                         "moisture, product and its relationship with the hair around "
                                                         "it): nothing - the brush "
                                                         "or a hand passes through the WHOLE depth of the hair, down "
                                                         "to the scalp, and turns each segment's heading toward the "
                                                         "stroke as it passes (the bristles' heading pull at every "
                                                         "height, not only within reach_mm of the skin); no point is "
                                                         "held anywhere. "
                                                         "'steer': only where the strand goes ACROSS the stroke - "
                                                         "each node within reach_mm of the brush's lane, sideways "
                                                         "(along the binormal T x n of the path at its arc); the "
                                                         "strand slides along the brush and its height off the head "
                                                         "is its own, as the bristles' heading pull already is. "
                                                         "'position' (before 10-31): within reach_mm of the path's "
                                                         "point at the node's arc in every direction. That path lies "
                                                         "ON the skin and is matched by arc, so a strand standing "
                                                         "off a round head trails its point and was dragged forward "
                                                         "and down: no layer over ~7.4 mm survived over the top "
                                                         "(every through-brushed head: crown hair above 15 mm "
                                                         "0.02-0.10 % against 7-12 % finger-combed; gr18 / V1 / A3 "
                                                         "crown p50 6.1-7.1 mm), and the friction anchors kept it "
                                                         "there after release (buildHS3 README 10-31). A brush rides "
                                                         "on the hair; it does not tie it to the scalp. The tucked "
                                                         "span (the hand) is held in position either way"),
        "through_sag_mm": (1.0, (0.1, 20.0), "the bristles' give: a point is pulled this far past the bristles' reach "
                                             "by its own segment's weight (the stylist's tension holds the section; "
                                             "softer than the skin's contact by ~300x)"),
        "strokes": (1, (1, 20), "the style stage (passes.py): how many times the brush is drawn through the grown hair "
                                "from the roots to the ends - a person brushes long hair several strokes; one stroke "
                                "on a full head left a curtain over the face (sheet 72)"),
        "fingers_face_lift_mm": (0.0, (0.0, 60.0), "the face hand's root lift (11-10): its fingers hold the section this "
                                                   "far off the scalp at the roots as they pass (rough-drying with the "
                                                   "fingers under the roots, hair_styling 7.2 (d): 'roots rough-dried'), "
                                                   "fading to 0 at dry.over_mm along the hair; a one-sided push off the "
                                                   "skin like the face barrier. 0 = the hand in the skin's tangent plane "
                                                   "only (before 11-10). With dry.when 'with_hands' the set is taken "
                                                   "while the hand holds it"),
        "fingers_face": (False, None, "the style stage (passes.py): after the brush, a hand drawn back through the hair "
                                      "around the face (the guides rooted within 40 mm of the front hairline and the "
                                      "temples), fingers through to the scalp, sweeping it back off the face; holds "
                                      "nothing once done"),
        "fingers_ears": (False, None, "the style stage: a hand lifts the side hair over each ear and lays it behind (the "
                                      "tuck's route: tuck.section / front_mm / behind_mm / over_mm / route), then lets "
                                      "go - the ear's contact, friction and the hair's weight behind keep it, or it "
                                      "springs back; no pin"),
        "fingers_shoulders": (False, None, "the style stage: a hand lays the lengths back over the shoulders to behind "
                                           "them (brush.shoulder_mm), then lets go; no pin"),
        "flow_map": ("", None, "the hair's flow measured from a reference photo and laid on the head (an npz from a case's "
                               "flow tool, e.g. buildHS3/hs3_flow_map.py: a (y, z) grid in m, the direction dy, dz and "
                               "a weight w per cell; repo-relative or absolute). The stroke turns toward it by its "
                               "weight, in the skin's plane, so the brush and the hands travel the reference's own flow "
                               "instead of the stroke terms above (10-30: flow comes from measurement, not from drawn "
                               "flow strokes). '' = none"),
        "over_shoulder": (False, None, "with through: the stroke's end - where the head turns under (and on the neck "
                                       "below the ears) the brush is drawn back and down over the shoulder to behind "
                                       "its crest, then the lengths hang down the back (a stylist lays long hair "
                                       "behind the shoulders; its weight there pulls the span over the ear back). "
                                       "Off: the old end, straight down where the head turns under"),
        "shoulder_mm": (40.0, (0.0, 100.0), "how far behind the shoulders' crest line the brush lays the lengths "
                                            "before they hang (a hand's width past the crest is ~60-80 mm; 40 mm "
                                            "puts them on the upper back's slope - a choice, not a sourced number)"),
        "blade": (False, None, "with over_shoulder: past the crest the brush is drawn down the back toward the "
                               "shoulder blade, not straight down where the lengths left the crest - a stylist lays "
                               "the longest lengths on the back, where it faces backward and holds them, not down the "
                               "outside of the upper arm (where they slide forward under their own weight: the test model rs1, "
                               "the 650 mm temple guides laid at x 135-158 mm, dy +48 -> -52 mm after release). The "
                               "back's lateral limit at each height is measured from the collider (back_table: the "
                               "contiguous back surface from the spine out to where it turns more than 45 deg to the "
                               "side); each length is aimed straight at that limit over its remaining length, so the "
                               "medial sweep is the body's own shape, no angle is set. Measured on the test model (ra1): "
                               "at the set the 650 mm temple guides lay at x 91-111 mm (rs1 135-158), but after "
                               "release contact and the pressure moved them back out to 137-163 mm and they slid "
                               "forward: ends in front 6 / on 6 (rs1 10 / 2) - the lay alone does not hold them"),
    },
    "dry": {
        "set": (0.15, (0.0, 1.0), "share of the turning kept as rest turning: air-dried after brushing 0.1-0.15, "
                                  "brushed while drying 0.4, blow-dry 0.7 (hair_styling 7 T4; Robbins 2012 ch. 8)"),
        "lift_deg": (0.0, (0.0, 60.0), "round-brush root lift: rest turning toward the skin normal over over_mm "
                                       "(0 air-dried; 15-35 deg a round-brush blow-dry, hair_styling 7 T5)"),
        "over_mm": (40.0, (0.0, 300.0), "the span from the root the set acts over (the length held on the brush); "
                                        "blow: the root length wound on the round brush, 20-50 mm (hair_styling 7.1 "
                                        "row 11; P19, P21, P24)"),
        "when": ("after_hands", ["after_hands", "before_hands", "with_hands"], "the style stage (flow.order 'after'): when the water "
                                    "set is taken. after_hands (before 11-10): after the brush and the hands, so the set "
                                    "keeps the shape the hands left (the face hand lays the front at the scalp: the test model "
                                    "proofs 11-10, 27 -> 16 mm off the scalp at 40 mm of arc, whatever the lift or set). "
                                    "before_hands: after the brush strokes, before the hands - dried and set first, then "
                                    "finger-styled (hair_styling 7.2 (a), (c): dry and set, then the brush-out and the "
                                    "fingers): the hands move hair that already holds its dried shape, and on release it "
                                    "springs back toward that shape. with_hands: taken at the end of the face hand while "
                                    "it still holds the section (dried in the hand, brush.fingers_face_lift_mm)"),
        "method": ("air", ["air", "blow"], "air: the brushed shape kept by `set` (air-dried); blow: a round-brush "
                                           "blow-dry - the root held at the base angle of its zone and wound on a "
                                           "brush of radius brush_mm / 2, cooled there; the set is that shape, kept "
                                           "by `set` (40-70 % heat-set under tension and cooled, Robbins 2012 ch. 8)"),
        "brush_mm": (50.0, (19.0, 75.0), "round-brush diameter: 50-55 mm for long hair, smoothness and volume with "
                                         "little curl; 25 mm gives bend (hair_styling D8, T2; P19, P23)"),
        "crown_brush_mm": (38.0, (19.0, 75.0), "the crown's brush: 38 mm over-directed 45 deg in one bouncy blowout "
                                               "(T2 round brush; P23)"),
        "crown_deg": (40.0, (0.0, 90.0), "the crown: roots within this polar angle of the vertex take crown_brush_mm"),
        "base_deg": ([135.0, 90.0, 45.0], (0.0, 180.0), "the angle the section is held at from its final direction, "
                                                        "toward and past the skin normal: on-base above the parietal "
                                                        "ridge and at the hairline (125-135; 135 = over-directed 45 "
                                                        "deg), half-off ridge to occipital (90), off-base below (45) "
                                                        "(hair_styling D6, T2 base control; P20, P21)"),
        "ridge_deg": (55.0, (20.0, 90.0), "the parietal ridge's polar angle from the vertex: on-base above it"),
        "occiput_deg": (95.0, (60.0, 130.0), "the occipital's polar angle: half-off between the ridge and it, at the back"),
        "front_phi_deg": (35.0, (0.0, 90.0), "the hairline in front (|phi| below this, fading over 10 deg) is rolled "
                                             "away from the face on base whatever its polar angle (T2 hairline swept "
                                             "back; P21)"),
        "root_mm": (4.0, (0.5, 15.0), "the hair turns from its follicle direction to the held angle over this length "
                                      "at the skin: the bend length sqrt(EI / T) of a fine hair held taut on the "
                                      "brush (fibre l_T is 13 mm under 300 mm of its own weight; a stylist's pull of "
                                      "~10 weights gives ~4 mm; no newton values are published, T2 row 27)"),
        "hold": (False, None, "the section held while it dries: the brush keeps it on its stroke (the brush still "
                              "in the hair) for hold_evals evaluations, then the set is taken from the shape it is "
                              "held in (span 'held') and the brush lets go"),
        "hold_evals": (60, (0, 1000), "evaluations the section is held (drying and cooling) before release"),
        "span": ("held", ["held", "band"], "held: every joint the brush or hand held while the hair dried takes rest "
                                           "turning natural + r (held - natural), r = set x wear retention, from the "
                                           "shape at the end of dry.hold - the hydrogen bonds re-form in whatever shape "
                                           "the fibre is held in as it dries and cools, the whole held length (hair_"
                                           "styling points 4, 7; P19-P35); with brush.through that is the whole strand, "
                                           "otherwise the over_mm the brush works at the root; a blow-dry's wound span "
                                           "is the round brush's shape, capped at 1 / R. band: the old split (over_mm "
                                           "root band + hold_set), kept for comparison"),
        "hold_set": (0.0, (0.0, 1.0), "ALIAS, span 'band' only: the lengths beyond over_mm keep this share of their "
                                      "turning (brushed while drying 0.4, hair_styling 7 T4); with span 'held' the "
                                      "whole held length takes set x retention and this key is ignored"),
    },
    "tuck": {
        "on": (False, None, "put the hair behind the ears"),
        "behind_mm": (4.0, (0.0, 20.0), "the tuck point this far behind the ear's back edge"),
        "over_mm": (6.0, (0.0, 20.0), "and this far above the ear's top"),
        "sag_mm": (2.0, (0.2, 20.0), "pin softness: the pin gives this much under the guide's whole weight"),
        "hold": (False, None, "with brush.through: keep the tucked lengths (the nodes the side stroke laid in the "
                              "ear's corridor) within behind_mm of where the hand laid them through the release and "
                              "the final settle - used only because the ear's contact alone did not hold them "
                              "(the test model, 398 guides: lateral rim cover 0.81 / 0.86 with contact alone)"),
        "groove_mm": (3.0, (0.0, 10.0), "with brush.through: the side section is laid down the groove between the "
                                        "ear's back and the skull (the auriculocephalic sulcus) this far in front of "
                                        "the ear's back edge, for the ear's full height and 10 mm below it"),
        "hand_mm": (4.0, (0.5, 20.0), "with brush.through: while the brush is on and the section held, the hand keeps "
                                      "the tucked span within this of its path (a hand pressed on the section, not "
                                      "the bristles' 15 mm reach - the reach would let it lie on the ear's face)"),
        "front_mm": (70.0, (0.0, 120.0), "with brush.through: the side section the hand carries starts this far in "
                                         "front of the ear (the temple) and runs down to the sideburn"),
        "route": ("groove", ["groove", "behind", "arc"], "with brush.through: where the hand lays the section down past the "
                                                  "ear's top. groove: in the sulcus under the auricle (groove_mm in "
                                                  "front of its back edge). behind: over the top of the helix and down "
                                                  "the head's skin behind_mm BEHIND the auricle's back edge (the "
                                                  "mastoid), so the ear stands in front of it. Measured on the test model "
                                                  "(gr6, 696 guides): every guide over the rim or the ear's top 15 mm "
                                                  "was a tucked guide whose laid path ran 13-15 mm medial of the rim, "
                                                  "threaded under the helix's back arc; once released its weight "
                                                  "pulled it straight, out of the wedge and over the ear's back edge "
                                                  "(rim 0.60 / 0.71). 'behind' (398 guides, oil adhesion): behind "
                                                  "share at mid height 0.95 / 0.88 (groove 0.82 / 0.77), rim 0.28 / "
                                                  "0.40 (groove 0.08 / 0.44) - the temple strands still lift ~14 mm "
                                                  "off the head onto the helix's back edge after release; not yet "
                                                  "the fix. arc: the hand carries the section round the helix's "
                                                  "top-back arc, not over a corner - the arc's centre and radius "
                                                  "measured from the ear's outline (_ear_arc), the path over_mm "
                                                  "outside the rim at the top, its radius easing to the descent line "
                                                  "by the arc's back end; the descent laid on the skull in the gap "
                                                  "between the auricle's free back part and the skull, midway between "
                                                  "the attachment line (per height, the back-most ear vertex lying "
                                                  "within gap_mm of the skull) and the free edge, where that gap "
                                                  "exists; elsewhere behind_mm behind the edge. Springback across a "
                                                  "near-right-angle corner (route behind) is what put hair over the "
                                                  "helix's top-back corner (the test model rs1: bending 99 % of the energy "
                                                  "released on the tucked span). Measured on the test model (ra1, 398 "
                                                  "guides, torso, oil): NOT the fix - the outline's top-back arc is "
                                                  "R_h 23-24 mm, the mesh shows almost no free gap behind the auricle "
                                                  "(3 mm: 0 / 26 rows +1, 2 / 26 rows -1), so the descent falls back "
                                                  "to behind_mm; the path hugs the rim ~3 mm out and the ear's seen "
                                                  "cover rose: set 0.15 / 0.11, final 0.40 / 0.31 (route behind, rs1: "
                                                  "0.06 / 0.08, 0.26 / 0.26); still bending-driven springback"),
        "section": ("ridge", ["ridge", "ear"], "with brush.through: which side strokes the hand takes into the tuck. "
                                               "ridge: the whole side section - every stroke that passes the ear's "
                                               "front (front_mm in front of it to its back) on the side of the head "
                                               "below the parietal ridge, the skin there facing more sideways than up "
                                               "(normal within 45 deg of horizontal: hair below the ridge is the "
                                               "exterior that hangs over the ear, hair above it the interior; "
                                               "hair_styling 7 rows 120 / 181, P4; the side wrap-dried back over the "
                                               "ear, row 14, P21). ear: the old box, strokes no higher than 30 mm "
                                               "above the ear's top (its tuck point + 30 mm). Measured on the test model "
                                               "(gr14 / gr15 / gr16, 696-713 guides, re-traced): the long temple "
                                               "guides root at z 195-250 mm and the ear box's top was z 226; 13-16 "
                                               "long temple guides per side fell above it each run, were never "
                                               "tucked, and the plain stroke wrapped them round the back of the head "
                                               "to the midline (laid end lateral ~0 mm); which ones fell above it "
                                               "was the guide sample's choice"),
        "fall": (False, None, "with brush.through: a section that reaches the ear's side from above (the front "
                              "sections laid away from the part, the hair over the temple) falls down the side of "
                              "the head under its own weight while the hand leads it back to over the helix: in "
                              "front of the ear its path is the straight line to that point, descending up to 6:1, "
                              "so it lies over the temple in front of the ear; off: the descent is capped at 1:1 and "
                              "the section is carried back above the temple and comes down behind the ear's front. "
                              "Measured on the test model (gr18 / gr19, 145k): 84-85 % of the hair seen over the temple was "
                              "the temple's own roots, 4-7 % the front hairline's beside the part; in the reference "
                              "the front hair beside the part falls sideways and sweeps down over the temple. With "
                              "brush.part_section, 40k / 700 guides, two seeds (side/sdA, sdB): NOT the fix - the "
                              "front roots' share of the temple's seen cover did not rise (+1 0.10 / 0.10, -1 0.14 / "
                              "0.19 against 0.14 / 0.07 and 0.11 / 0.10 at 40k without it), the long hair over the "
                              "face rose to 180-183 strands (51-77 without), a temple rope came back (+1 85 / 20, -1 "
                              "59 / 41 points) and the ear's seen cover rose: the steep fall in front of the ear "
                              "does not hold once released (the reference's temple streaks descend 34-50 deg)"),
        "gap_mm": (3.0, (0.5, 10.0), "route arc: the auricle stands free of the skull where its vertex lies more than "
                                     "this lateral of the skull's skin (the section's thickness: ~3 mm, a few "
                                     "hundred fine hairs side by side - a choice, not a sourced number)"),
    },
    "wear": {
        "age_h": (0.0, (0.0, 72.0), "hours since the hair was dried and set (0: freshly set, retention 1); the set "
                                    "decays toward the natural shape (hair_styling 7.1 row 20; P25, P27, P29, P35)"),
        "rh": (45.0, (20.0, 90.0), "relative humidity of the air worn in, %: water taken back into the fibre (6.0 / "
                                   "9.8 / 13.6 % at 29 / 50 / 70 % RH, D12) re-breaks the set's bonds"),
    },
    "finish": {
        "hold": (0.0, (0.0, 1.0), "hairspray / gel fixative: the set's decay clock runs 10^(-2 hold) as fast "
                                  "(0.5 a flexible spray: 45 % kept at 8 h and 90 % RH; 1 a good fixative: 90 %; "
                                  "hair_styling D11, row 25; P25, P27, P55)"),
        "where": ("all", ["all", "hairline"], "where the fixative went: the whole head, or the front hairline guides "
                                              "only (sprayed after it was rolled back and cooled)"),
    },
    "hand": {
        "on": (False, None, "lay the front lengths in front of the shoulders"),
        "phi_deg": ([95.0, 120.0], (0.0, 180.0), "roots with |phi| below the first value are laid forward, fading out "
                                                 "by the second"),
        "stages": (3, (1, 10), "growth stages the hand holds them, counted from the first that reaches the shoulders"),
        "front_mm": (10.0, (0.0, 60.0), "the lengths are held this far in front of the shoulder plane"),
    },
    "tousle": {
        "strokes": (0, (0, 12), "finger shakes after release (4-8 for a tousled look; 0 none)"),
        "force": (0.5, (0.0, 3.0), "the shake's force in segment weights (DESIGN_v2 5: about half)"),
        "scale_mm": (40.0, (5.0, 200.0), "the shake field's length scale (a hand's width)"),
        "evals": (30, (5, 200), "evaluations per stroke"),
    },
    "face": {
        "barrier": (True, None, "a hand keeps hair off the face while brushing (growth only)"),
        "margin_mm": (3.0, (0.0, 20.0), "the barrier's distance off the face skin"),
    },
}


def _plain(node):
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    return node[0]


ACTS = _plain(ACTS_DEFAULTS)


def settings(st=None):
    """the acts' values from a whole style (its brush / dry / tuck / hand / tousle / face blocks) with missing keys
    filled from ACTS_DEFAULTS -> (values, warnings)"""
    st = st or {}
    out, warn = {}, []
    for blk, node in ACTS_DEFAULTS.items():
        got = st.get(blk) or {}
        vals = dict(ACTS[blk])
        for k, v in got.items():
            if k not in node:
                warn.append("%s.%s: unknown key" % (blk, k))
                continue
            vals[k] = v
        out[blk] = vals
    return out, warn


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _unit(v):
    n = np.sqrt((v * v).sum(-1, keepdims=True))
    return v / np.maximum(n, 1e-300)


def part_side(scalp, X, style, rng):
    """each root's part side (+1 / -1), the side it is brushed toward: by the part line (x = part.offset_mm for a side
    part, the midline for a centre part), with part.cross_frac of the roots near the line (within cross_mm, in front of
    cross_front_mm behind the hairline, above theta 70) on the other side, fading out with the distance (a faint line).
    Children take the side of their guides and never interpolate across it. part.type none: all +1, no part term."""
    PT = dict(style.get("part") or {})
    typ = PT.get("type", "centre")
    n = len(X)
    if typ == "none":
        return np.ones(n, np.int8), 0.0
    x_part = PT.get("offset_mm", 0.0) / 1000 if typ == "side" else 0.0
    sx = X[:, 0] - scalp.c[0] - x_part
    side = np.where(sx >= 0, 1, -1)
    cf = float(PT.get("cross_frac", 0.0))
    if cf > 0:
        din = scalp.d_inside_mm(X)
        phi, th, _ = scalp.sph(X)
        fr = np.abs(phi) < 90
        y_front = float(X[fr & (din < 3.0), 1].min()) if (fr & (din < 3.0)).any() else float(X[:, 1].min())
        cm = PT.get("cross_mm", 8.0) / 1000
        zone = (np.abs(sx) < cm) & (X[:, 1] < y_front + PT.get("cross_front_mm", 50.0) / 1000) & (th < 70)
        p = cf * (1 - np.abs(sx) / max(cm, 1e-9))
        side = np.where(zone & (rng.random(n) < p), -side, side)
    return side.astype(np.int8), x_part


def _noise_field(rng, nmodes=6):
    """a seeded smooth random scalar field of unit rms: a sum of plane waves (wave vectors on a unit sphere)"""
    k = _unit(rng.normal(size=(nmodes, 3)))
    ph = rng.uniform(0, 2 * np.pi, nmodes)
    amp = np.sqrt(2.0 / nmodes)
    return lambda P, L: amp * np.cos((P @ k.T) / L + ph).sum(-1)


class Acts:
    """the stylist's acts for one settle (settle.py's acts hook).
    scalp: scalp.Scalp (c, ears, V, d_inside_mm, sph); body: grid.SDFGrid; style: the whole style (flow, part,
    corridor and the act blocks); rods: settle.Rods; W: the follicle whorl (m, follicle.field); coll_V: the collider's
    vertices (m) for the shoulder plane (only needed with hand.on); seed: the acts' own RNG (noise, cross, tousle)."""

    def __init__(self, scalp, body, style, rods, W=None, coll_V=None, seed=0, log=print):
        self.scalp, self.body, self.rods, self.log = scalp, body, rods, log or (lambda m: None)
        self.v, self.warnings = settings(style)
        fl = dict(style.get("flow") or {})
        self.fl = {k: float(fl.get(k, 0.0)) for k in ("away_from_part", "back", "side", "up", "face_back")}
        self.rng = np.random.default_rng(seed + 307)
        X = rods.X
        self.side, self.x_part = part_side(scalp, X, style, self.rng)
        PT = dict(style.get("part") or {})
        din = scalp.d_inside_mm(X)
        phi_r, th_r, _ = scalp.sph(X)
        frn = (np.abs(phi_r) < 20) & (din < 4.0)
        self.y_front = float(np.median(X[frn, 1])) if frn.any() else float(X[:, 1].min())
        self.part_depth = PT.get("depth_mm", 20.0) / 1000 if PT.get("type", "centre") != "none" else 0.0
        self.W = None if W is None else np.asarray(W, np.float64)
        self.phi_r = phi_r
        # each strand's part section (brush.part_section): its root's part-zone weight, carried along its stroke
        self.wf_root = self.part_zone(X, th_r)
        self.front_g = (np.abs(phi_r) < 40) & (din < 15.0)   # roots at the front hairline (reports only)
        b = self.v["brush"]
        self.noise = _noise_field(self.rng) if b["noise_deg"] > 0 else None
        self.flow_map = None
        if b.get("flow_map"):
            p = b["flow_map"]
            if not os.path.isabs(p):
                p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), p)
            with np.load(p) as z:
                self.flow_map = {k: np.asarray(z[k], np.float64) for k in ("y", "z", "dy", "dz", "w")}
        # the face skin: head vertices outside the hairline by more than corridor.outside_mm, within corridor.phi_deg
        # of the front, above the ears' lower end (flow.corridor_mask's definition)
        cc = dict(style.get("corridor") or {})
        self.corr = {"outside_mm": float(cc.get("outside_mm", 3.0)), "phi_deg": float(cc.get("phi_deg", 75.0))}
        V = scalp.V
        phv, _, _ = scalp.sph(V)
        z_eb = min(e["z_bot"] for e in scalp.ears.values())
        self.z_eb = z_eb
        self.face_v = (scalp.d_inside_mm(V) < -self.corr["outside_mm"]) & (np.abs(phv) < self.corr["phi_deg"]) & \
            (V[:, 2] > z_eb)
        self.tree = cKDTree(V)
        # the shoulder plane (hand): the collider's shoulder tops, |x| 110-160 mm, within 15 mm of their highest point
        self.y_sh = self.z_sh = None
        if coll_V is not None:
            ax = np.abs(coll_V[:, 0])
            band = (ax > 0.11) & (ax < 0.16) & (coll_V[:, 2] < scalp.c[2] - 0.08)
            if band.any():
                zt = coll_V[band, 2].max()
                top = band & (coll_V[:, 2] > zt - 0.015)
                self.y_sh, self.z_sh = float(np.median(coll_V[top, 1])), float(zt)
        # the shoulders' crest line (brush.over_shoulder, and the end-of-length reports)
        self.ridge = shoulder_ridge(coll_V, z_eb, scalp.c) if coll_V is not None else None
        # the back's lateral limit per height (brush.blade): where the lengths can lie on the back
        self.back = back_table(coll_V, self.ridge, scalp.c) if (coll_V is not None and self.ridge is not None and
                                                                b.get("blade", False)) else None
        h = self.v["hand"]
        self.hand_w = 1 - smoothstep(h["phi_deg"][0], h["phi_deg"][1], np.abs(phi_r))
        # the acts' state
        self.brush_on = bool(b["on"])
        self.face_on = bool(self.v["face"]["barrier"])
        self.tuck_on = bool(self.v["tuck"]["on"])
        self.hand_on = bool(h["on"])
        self.hand_left = int(h["stages"])
        self.hand_started = False
        self.released = False
        self.tousle_field = None
        self.tousle_F = None                   # lagged shake force on the nodes
        self.hand_low = None                   # lagged: nodes below the shoulders' top
        G, K = rods.G, rods.K
        self.bn = np.zeros((G, K, 3))          # lagged skin normal at each segment
        self.bb = np.zeros((G, K, 3))          # lagged stroke tangent
        self.ba = np.zeros((G, K))             # lagged reach weight
        self.face_flag = None                  # lagged: samples over the face skin
        self.tuck_j = np.full(G, -1)           # the pinned node of each tucked guide (index into P)
        self.tuck_q = np.zeros((G, 3))
        self.tuck_phase = np.zeros(G, np.int8)  # 1: over the top, 2: behind the ear
        self.tuck_e = np.zeros(G, np.int8)
        self.k_tuck = rods.Wtot / (self.v["tuck"]["sag_mm"] / 1000.0)
        self.counts = {"tucked": 0, "hand_stages": 0}
        # the stroke drawn through (brush.through): the brush's path at each node's arc, traced once
        self.through = bool(b.get("through", False))
        hold = str(b.get("through_hold", "steer"))
        self.steer = self.through and hold in ("steer", "none")          # new length at its own elevation
        self.free_pass = self.through and hold == "none"                 # the heading pull through the whole depth
        self.holding = False                   # dry.hold: the section held on its stroke after the set
        # styling in sections (DESIGN_sections): the section being worked, (G,) bool; None = every guide (as before).
        # The brush, its through hold, the face barrier and the hand act on it only; the set is taken on it only.
        self.on_g = None
        self.set_pending = False               # dry.span 'held' with dry.hold: the set waits for the hold's end
        self.P_held = None                     # node positions when the set was taken (reports: release movement)
        self._warned = set()
        self.Qp = None                         # (G, K + 1, 3) m: the brush's position at each node's arc
        self.tuck_pin = None                   # (G, K + 1) bool: the nodes the side stroke laid by an ear
        self.tuck_seg = None                   # (G, K) bool: the segments of that tucked span
        self.T_path = None                     # (G, K, 3): the brush path's unit tangent per segment
        self.k_path = None                     # (G, K) N/m per node
        # a pass on grown hair (engine/hair/passes.py, DESIGN_grow_then_style): the tool's window on each segment
        # (G, K) in place of the bristles' reach from the skin, and the drag - the bristles' friction moving with the
        # tool - on the samples in the window: {idx, a, n, C, eps} (lagged per step). None outside a pass.
        self.pass_win = None
        self.drag = None
        self.lift = None                       # the face hand's root lift (passes.sweep_pass): {idx, h, k} or None
        if self.through:
            self._trace_paths()

    # -------------------------------------------------- the stroke field
    def part_zone(self, M, th=None):
        """the part zone's weight at points M (m, 3): the front part.depth_mm of the top (polar angle under 55-88 deg),
        where the brush draws the hair away from the part line"""
        if th is None:
            _, th, _ = self.scalp.sph(M)
        return (1 - smoothstep(55.0, 88.0, th)) * \
            (1 - smoothstep(self.part_depth - 0.02, self.part_depth + 0.01, M[:, 1] - self.y_front))

    def stroke(self, M, sg, n, wf=None):
        """the brush stroke's unit tangent b at points M (m, 3) of guides on part sides sg, skin normals n; also |F_t|.
        wf (m,): the part-zone weight of each point's strand at its root (brush.part_section), None for none"""
        sc = self.scalp
        phi, th, _ = sc.sph(M)
        x, y, z = M[:, 0], M[:, 1], M[:, 2]
        fl, b = self.fl, self.v["brush"]
        F = np.zeros_like(M)
        # the part zone (the front `depth` of the top): away from the part line, by part side
        w_pz = self.part_zone(M, th)
        if wf is not None and b.get("part_section", False):
            # brush.part_section: a section parted off at the front is one section - the stylist draws it away from
            # the part along its whole stroke, down the side and back to the ear, not only while the brush is still
            # inside the part zone. The strand's own section weight (its root's part-zone weight, wf) is carried
            # along the stroke in front of the ears' back edge and above their lower end; behind the ear and below
            # it the lengths take the plain stroke (back and down), so the away term never spreads the back hair.
            y_bk = max(e["y_back"] for e in sc.ears.values())
            w_sec = wf * (1 - smoothstep(y_bk, y_bk + 0.02, y)) * smoothstep(self.z_eb, self.z_eb + 0.02, z)
            w_pz = np.maximum(w_pz, w_sec)
        F[:, 0] += fl["away_from_part"] * sg * w_pz
        # behind it, the crown: away from the follicle whorl
        if self.W is not None:
            w_cr = (1 - w_pz) * (1 - smoothstep(60.0, 100.0, th))
            F += fl["away_from_part"] * w_cr[:, None] * _unit(M - self.W)
        F[:, 1] += fl["back"]
        F[:, 0] += fl["side"]
        F[:, 2] += fl["up"]
        # the back-sweep over the front and the temples, above the ears' lower end
        z_et = max(e["z_top"] for e in sc.ears.values())
        w_fb = smoothstep(18.0, 50.0, np.abs(phi)) * (1 - smoothstep(105.0, 135.0, np.abs(phi))) * \
            smoothstep(self.z_eb, self.z_eb + 0.02, z) * (1 - smoothstep(z_et + 0.05, z_et + 0.08, z))
        F[:, 1] += fl["face_back"] * w_fb
        # the sides are brushed down
        w_sd = smoothstep(50.0, 70.0, np.abs(phi)) * (1 - smoothstep(110.0, 130.0, np.abs(phi))) * \
            smoothstep(55.0, 75.0, th)
        F[:, 2] -= b["down_sides"] * w_sd
        F[:, 2] -= b["lengths"]
        Ft = F - (F * n).sum(1, keepdims=True) * n
        mag = np.sqrt((Ft * Ft).sum(1))
        bt = Ft / np.maximum(mag, 1e-300)[:, None]
        if self.flow_map is not None:
            # the measured flow (brush.flow_map): its side-view direction (0, dy, dz) in the skin's plane, turned to
            # by the cell's weight - less where the skin faces the view's own axis less (that direction then mostly
            # leaves the skin's plane and says little)
            dyq, dzq, wq = self._flow_at(y, z)
            t = np.stack([np.zeros_like(dyq), dyq, dzq], 1)
            tt = t - (t * n).sum(1, keepdims=True) * n
            tm = np.sqrt((tt * tt).sum(1))
            wq = wq * np.clip(tm / 0.3, 0.0, 1.0)
            mix = (1.0 - wq)[:, None] * bt + wq[:, None] * (tt / np.maximum(tm, 1e-300)[:, None])
            mm = np.sqrt((mix * mix).sum(1))
            bt = np.where((mm > 1e-9)[:, None], mix / np.maximum(mm, 1e-300)[:, None], bt)
            mag = np.maximum(mag, wq)
        if self.noise is not None:
            a = np.radians(b["noise_deg"]) * self.noise(M, b["noise_mm"] / 1000.0)
            bt = np.cos(a)[:, None] * bt + np.sin(a)[:, None] * np.cross(n, bt)
        return bt, mag

    def _flow_at(self, y, z):
        """the measured flow (brush.flow_map) at points (y, z) m: bilinear dy, dz (renormalised) and weight; 0 off its
        grid"""
        fm = self.flow_map
        gy, gz = fm["y"], fm["z"]
        fy = (y - gy[0]) / (gy[1] - gy[0])
        fz = (z - gz[0]) / (gz[1] - gz[0])
        ok = (fy >= 0) & (fy <= len(gy) - 1.001) & (fz >= 0) & (fz <= len(gz) - 1.001)
        i0 = np.clip(np.floor(fy).astype(int), 0, len(gy) - 2)
        j0 = np.clip(np.floor(fz).astype(int), 0, len(gz) - 2)
        ty, tz = np.clip(fy - i0, 0, 1), np.clip(fz - j0, 0, 1)
        out = []
        for k in ("dy", "dz", "w"):
            A = fm[k] * (fm["w"] if k != "w" else 1.0)          # the directions weighted, so empty cells do not pull
            out.append((A[i0, j0] * (1 - ty) * (1 - tz) + A[i0 + 1, j0] * ty * (1 - tz) + A[i0, j0 + 1] * (1 - ty) * tz
                        + A[i0 + 1, j0 + 1] * ty * tz))
        dy, dz, w = out
        nn = np.sqrt(dy * dy + dz * dz)
        return (np.where(ok, dy / np.maximum(nn, 1e-12), 0.0), np.where(ok, dz / np.maximum(nn, 1e-12), 0.0),
                np.where(ok & (nn > 1e-9), np.clip(w, 0.0, 1.0), 0.0))

    def _skull_table(self, step=0.002, r=0.003):
        """the head's lateral skin beside each ear, ear vertices left out: for (y, z) on a step grid from front_mm in
        front of the ear to 30 mm behind it and from 30 mm below it to 60 mm above its top, the innermost lateral
        offset |x - c| of the scalp's non-ear vertices within r (m) - under the ear that is the skull in the groove
        behind the ear (not the ear's own base), in front of it the cheek skin (not the tragus). NaN where the scalp
        has no vertex."""
        sc, tk = self.scalp, self.v["tuck"]
        V = sc.V
        em = sc.ear_mask if hasattr(sc, "ear_mask") else np.zeros(len(V), bool)
        out = {}
        for e_s, e in sc.ears.items():
            lat = e_s * (V[:, 0] - sc.c[0])
            m = (~em) & (lat > 0.03)
            tr = cKDTree(V[m][:, 1:3])
            la = lat[m]
            ys = np.arange(e["y_front"] - tk["front_mm"] / 1000.0 - 0.01, e["y_back"] + 0.03, step)
            zs = np.arange(e["z_bot"] - 0.03, e["z_top"] + 0.06, step)
            Y, Z = np.meshgrid(ys, zs, indexing="ij")
            T = np.full(Y.shape, np.nan)
            for idx, nb in enumerate(tr.query_ball_point(np.stack([Y.ravel(), Z.ravel()], 1), r)):
                if nb:
                    T.flat[idx] = la[nb].min()
            out[e_s] = (ys, zs, T)
        return out

    def _skull_lat(self, e_s, y, z):
        """the skull table's lateral offset (m) at points (y, z), bilinear; NaN off the table"""
        ys, zs, T = self._skull[e_s]
        fy = (y - ys[0]) / (ys[1] - ys[0])
        fz = (z - zs[0]) / (zs[1] - zs[0])
        ok = (fy >= 0) & (fy <= len(ys) - 1.001) & (fz >= 0) & (fz <= len(zs) - 1.001)
        out = np.full(len(y), np.nan)
        if ok.any():
            i, j = np.floor(fy[ok]).astype(int), np.floor(fz[ok]).astype(int)
            u, v = fy[ok] - i, fz[ok] - j
            c = np.stack([T[i, j], T[i + 1, j], T[i, j + 1], T[i + 1, j + 1]], 1)
            w = np.stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], 1)
            good = np.isfinite(c)
            ws = (w * good).sum(1)
            val = (np.where(good, c, 0.0) * w).sum(1) / np.maximum(ws, 1e-12)
            out[np.nonzero(ok)[0]] = np.where(ws > 0.25, val, np.nan)
        return out

    def _ear_arc(self):
        """the ear's own geometry for tuck.route 'arc', per ear, from the head mesh (ear vertices, the skull table):
        - the helix's top-back arc: the ear's outline in the side view (y, z), its farthest vertex in each 5 deg bin
          from the back (0) to the top (90) about the ear's centre, fitted with a circle (algebraic least squares,
          Kasa 1976) -> centre (yc, zc), radius R_h;
        - the attachment line: per 2 mm of height the back-most ear vertex lying within tuck.gap_mm of the skull's
          skin (it sits on the skull); behind it, up to the ear's back edge at that height, the auricle stands free
          and the gap under it is where a tucked section can lie, hidden by the ear;
        - the descent line y_d(z): midway between the attachment and the edge where that free part is wider than
          gap_mm, else behind_mm behind the edge; smoothed over 5 rows.
        -> {e_s: dict}"""
        sc, tk = self.scalp, self.v["tuck"]
        V = sc.V
        out = {}
        for e_s, e in sc.ears.items():
            ev = sc.ear_mask & (e_s * (V[:, 0] - sc.c[0]) > 0)
            Y, Z = V[ev, 1], V[ev, 2]
            lat = e_s * (V[ev, 0] - sc.c[0])
            y0, z0 = 0.5 * (Y.min() + Y.max()), 0.5 * (Z.min() + Z.max())
            ang = np.degrees(np.arctan2(Z - z0, Y - y0))
            rr = np.hypot(Y - y0, Z - z0)
            py, pz = [], []
            for a0 in range(0, 90, 5):
                m = (ang >= a0) & (ang < a0 + 5)
                if m.any():
                    i = np.argmax(np.where(m, rr, -1.0))
                    py.append(Y[i])
                    pz.append(Z[i])
            py, pz = np.asarray(py), np.asarray(pz)
            A = np.stack([py, pz, np.ones_like(py)], 1)
            sol = np.linalg.lstsq(A, py * py + pz * pz, rcond=None)[0]
            yc, zc = 0.5 * sol[0], 0.5 * sol[1]
            R_h = float(np.sqrt(max(sol[2] + yc * yc + zc * zc, 1e-8)))
            fit_rms = float(np.sqrt(np.mean((np.hypot(py - yc, pz - zc) - R_h) ** 2)))
            sk = self._skull_lat(e_s, Y, Z)
            gap = lat - sk
            zs = np.arange(e["z_bot"], e["z_top"] + 1e-9, 0.002)
            yd = np.full(len(zs), np.nan)
            ya = np.full(len(zs), np.nan)
            ye = np.full(len(zs), np.nan)
            g = tk.get("gap_mm", 3.0) / 1000.0
            for k, z in enumerate(zs):
                m = np.abs(Z - z) < 0.002
                if not m.any():
                    continue
                ye[k] = Y[m].max()
                att = m & np.isfinite(gap) & (gap < g)
                if att.any():
                    ya[k] = Y[att].max()
                yd[k] = 0.5 * (ya[k] + ye[k]) if (np.isfinite(ya[k]) and ye[k] - ya[k] > g) else                     ye[k] + tk["behind_mm"] / 1000.0
            ok = np.isfinite(yd)
            yd = np.interp(zs, zs[ok], yd[ok])
            yd = np.array([np.median(yd[max(k - 2, 0):k + 3]) for k in range(len(yd))])
            free = np.isfinite(ya) & (ye - ya > g)
            out[e_s] = {"yc": float(yc), "zc": float(zc), "R_h": R_h, "fit_rms": fit_rms, "zs": zs, "yd": yd,
                        "free_rows": int(free.sum()), "rows": int(len(zs)),
                        "free_mm_p50": float(1000 * np.median((ye - ya)[free])) if free.any() else None,
                        "yd_minus_back_mm": np.round(1000 * (np.percentile(yd, [10, 50, 90]) - e["y_back"]), 1).tolist()}
        return out

    def _yd(self, e_s, z):
        """the arc route's descent line y_d at heights z (m), held flat beyond its ends"""
        a = self._arc[e_s]
        return np.interp(z, a["zs"], a["yd"])

    def _shoulder_aim(self, x, behind):
        """the stroke's end over the shoulder (brush.over_shoulder): at points x (m) the unit heading toward the
        point `behind` (m) behind the shoulders' crest at the crest's height, the descent never flatter than 10 mm
        per that run -> (aim (n, 3), ok (n,)): ok where a crest exists at that lateral offset, the point is still in
        front of the target and not more than 30 mm below the crest (lengths already past it hang as they are)"""
        yr, zr = ridge_at(self.ridge, x[:, 0])
        ty = yr + behind
        ok = np.isfinite(yr) & (x[:, 1] < ty) & (x[:, 2] > zr - 0.03)
        F = np.zeros_like(x)
        F[:, 1] = np.where(ok, ty - x[:, 1], 0.0)
        F[:, 2] = -np.maximum(np.where(ok, x[:, 2] - zr, 1.0), 0.01)
        return _unit(F), ok

    def _blade(self, x, rem):
        """brush.blade: past the shoulders' crest (behind it), the brush is drawn down the back toward the shoulder
        blade - each point aimed straight at the back's lateral limit at its height (back_table) over its remaining
        length rem (m), so a length laid at the shoulder's outer end comes onto the back by the time its end hangs,
        and one already over the back goes straight down. In front of the crest, or where there is no crest, straight
        down. -> unit heading (n, 3)"""
        zs, xbp, xbn = self.back
        c0 = self.scalp.c[0]
        xl = x[:, 0] - c0
        lim = np.where(xl >= 0, np.interp(x[:, 2], zs, xbp), np.interp(x[:, 2], zs, xbn))
        yr, _ = ridge_at(self.ridge, x[:, 0])
        behind = np.isfinite(yr) & (x[:, 1] > yr)
        F = np.zeros_like(x)
        F[:, 0] = np.where(behind, np.clip(xl, -lim, lim) - xl, 0.0)
        F[:, 2] = -np.sqrt(np.maximum(rem * rem - F[:, 0] ** 2, 1e-4))
        return _unit(F)

    def _tuck_turn(self, x, n, b, ph, eo):
        """the hand's tuck of the side section (tuck.on with brush.through), what a stylist does: every strand of the
        section from the temple (tuck.front_mm in front of the ear) down to the sideburn is carried back over the
        ear's attachment (rising, in front of the ear, to tuck.over_mm above the helix), then down BEHIND the ear into
        the groove between the ear's back and the skull (tuck.groove_mm in front of the ear's back edge; with
        tuck.route 'behind', on the skin tuck.behind_mm behind the ear's back edge) for the ear's full height, to
        10 mm below its lower end; then the stroke field (back and down the lengths) takes
        over. The ear's back and the skull hold it there after release, and its own weight keeps it down.
        ph (G,) int8, the phase of each guide's brush: 0 not tucked yet, 1 carried over the ear, 2 down the groove,
        3 past the ear (never re-enters); eo (G,) the ear. Changed in place. -> (heading b, on_skull mask): the
        points the hand lays on the skull (placed by _skull_lat, never on the ear's own surface)."""
        sc, tk = self.scalp, self.v["tuck"]
        out = b.copy()
        on = np.zeros(len(x), bool)
        arc = tk.get("route", "groove") == "arc"
        for e_s, e in sc.ears.items():
            zq = e["z_top"] + tk["over_mm"] / 1000.0
            if arc:
                # round the helix's top-back arc: centre (yc, zc), radius Ra over the top (over_mm above the ear's
                # top), easing to r_end where the arc meets the descent line at the centre's height
                A = self._arc[e_s]
                yc, zc = A["yc"], A["zc"]
                Ra = max(zq - zc, 0.004)
                r_end = max(float(self._yd(e_s, np.array([zc]))[0]) - yc, 0.004)
                yt = yc + max(Ra, r_end)
            elif tk.get("route", "groove") == "behind":
                # over the helix's top and down behind the auricle's back edge, on the mastoid skin
                yt = e["y_back"] + tk["behind_mm"] / 1000.0
            else:
                yt = e["y_back"] - tk["groove_mm"] / 1000.0
            lat = e_s * (x[:, 0] - sc.c[0])
            if tk.get("section", "ridge") == "ridge":
                # the side section: on the side of the head below the parietal ridge (the skin facing more sideways
                # than up) - one rule for every side stroke, whatever height its root has
                high = e_s * n[:, 0] > n[:, 2]
            else:
                high = x[:, 2] < zq + 0.03
            new = (ph == 0) & (lat > 0.045) & (x[:, 1] > e["y_front"] - tk["front_mm"] / 1000.0) & \
                (x[:, 1] < yt) & (x[:, 2] > e["z_bot"] - 0.01) & high
            ph[new], eo[new] = 1, e_s
            mine = eo == e_s
            if arc:
                to2 = mine & (ph == 1) & (x[:, 1] >= yc) & (x[:, 2] <= zc)
            else:
                to2 = mine & (ph == 1) & (x[:, 1] >= yt - 0.001)
            ph[to2] = 2
            to3 = mine & (ph == 2) & (x[:, 2] < e["z_bot"] - 0.01)
            ph[to3] = 3
            m1, m2 = mine & (ph == 1), mine & (ph == 2)
            if m1.any():
                # back, and up to the height over the helix: in front of the ear a straight rise to the point 12 mm in
                # front of its top front corner (the hand lifts the sideburn up and back, it does not drag it across
                # the tragus); beside the ear (never below its top in practice) a steep rise
                y, z = x[m1, 1], x[m1, 2]
                y_r = e["y_front"] - 0.012
                F = np.zeros((int(m1.sum()), 3))
                F[:, 1] = 1.0
                # tuck.fall: a section that comes to the ear from above (the front sections drawn down from the
                # part, brush.part_section) falls down the side of the head under its own weight as it is led back
                # - the hand does not hold it up against gravity above the ear while it carries it back - so in front
                # of the ear the path is the straight line to the point over the helix, as steep down as up (6:1);
                # off: no descent steeper than 45 deg, the section carried back at its height until behind the ear
                lo_f = -6.0 if tk.get("fall", False) else -1.0
                F[:, 2] = np.where(y < y_r, np.clip((zq - z) / np.maximum(y_r - y, 0.003), lo_f, 6.0),
                                   np.clip((zq - z) / 0.005, -1.0, 6.0))
                if arc:
                    # past the arc's centre: along the arc (clockwise in the side view, back and down), its radius
                    # easing from Ra at the top to r_end at the back, held to it over 4 mm - no corner to spring open
                    ry, rz = y - yc, z - zc
                    rho = np.maximum(np.hypot(ry, rz), 1e-6)
                    th = np.clip(np.arctan2(rz, ry), 0.0, 0.5 * np.pi)
                    Rt = r_end + (Ra - r_end) * th / (0.5 * np.pi)
                    c = np.clip((Rt - rho) / 0.004, -1.0, 1.0)
                    Fa = np.zeros_like(F)
                    Fa[:, 1] = rz / rho + c * ry / rho
                    Fa[:, 2] = -ry / rho + c * rz / rho
                    F = np.where((y >= yc)[:, None], Fa, F)
                out[m1] = _unit(F)
            if m2.any():
                # down the groove, held at its depth
                F = np.zeros((int(m2.sum()), 3))
                ytg = self._yd(e_s, x[m2, 2]) if arc else yt
                F[:, 1] = np.clip((ytg - x[m2, 1]) / 0.004, -1.0, 1.0)
                F[:, 2] = -1.0
                out[m2] = _unit(F)
            on |= m1 | m2
        return out, on

    def _trace_paths(self, ds=0.002):
        """the brush's path for each guide (brush.through): from the root along the stroke field b on the skin (the
        bristles reach the scalp; the hair lies within them), stepped ds and put back on the skin; where the skin
        turns under (normal pointing more than ~15 deg below horizontal: the occiput's underside, the jaw) the brush
        leaves the head and is drawn straight down off the ends, kept outside the body. Tucking turns the side strokes
        back over the ear's top and behind it (_tuck_turn). The path's position at each node's rest arc is the
        brush's position when it holds that point of the strand."""
        r = self.rods
        G = r.G
        arcs = np.concatenate([np.zeros((G, 1)), r.s + r.l], 1)          # (G, K + 1) node rest arcs
        tuck = bool(self.v["tuck"]["on"])
        path, cum, off, ph = self._trace(arcs, ds, False)
        use = np.zeros(G, bool)
        if tuck:
            self._skull = self._skull_table()
            if self.v["tuck"].get("route", "groove") == "arc":
                self._arc = self._ear_arc()
                self.counts["ear_arc"] = {str(k): {kk: (round(vv * 1000, 2) if isinstance(vv, float) and kk in (
                    "yc", "zc", "R_h", "fit_rms") else vv) for kk, vv in a.items() if kk not in ("zs", "yd")}
                    for k, a in self._arc.items()}
                self.log("  acts: ear arc (mm) %s" % self.counts["ear_arc"])
            path_t, cum_t, off_t, ph_t = self._trace(arcs, ds, True)
            # the arc where each strand's brush comes over the ear's attachment (phase 2 starts); a strand too short
            # to reach 10 mm past it stays where it grows (short sideburn hairs), on the plain stroke
            k2 = np.argmax(ph_t >= 2, 0)
            reach2 = np.where((ph_t >= 2).any(0), cum_t[k2, np.arange(G)], np.inf)
            use = (ph_t >= 1).any(0) & (arcs[:, -1] >= reach2 + 0.01)
            path[:, use], cum[:, use], off[use], ph[:, use] = path_t[:, use], cum_t[:, use], off_t[use], ph_t[:, use]
        Q = np.empty((G, arcs.shape[1], 3))
        phn = np.zeros((G, arcs.shape[1]), np.int8)
        for i in range(G):
            for c in range(3):
                Q[i, :, c] = np.interp(arcs[i], cum[:, i], path[:, i, c])
            phn[i] = ph[np.clip(np.searchsorted(cum[:, i], arcs[i]), 0, len(cum) - 1), i]
        self.Qp = Q
        # the tucked span of each strand: the nodes the hand laid over the ear and down the groove
        self.tuck_pin = (phn == 1) | (phn == 2)
        self.tuck_pin[:, 0] = False
        self.tuck_seg = self.tuck_pin[:, 1:] | self.tuck_pin[:, :-1]     # (G, K) segments touching it
        self.T_path = _unit((Q[:, 1:] - Q[:, :-1]).reshape(-1, 3)).reshape(G, -1, 3)
        # the brush's lane across the stroke at each node (through_hold 'steer'): the binormal of the path, T x n, n the
        # skin's normal at the path point (on the head the path lies on the skin; off it, the nearest skin's)
        _, gq = self.body.sample(Q[:, 1:].reshape(-1, 3), grad=True)
        nq = _unit(gq).reshape(G, -1, 3)
        self.B_path = _unit(np.cross(self.T_path, nq).reshape(-1, 3)).reshape(G, -1, 3)   # per node j + 1
        sag = self.v["brush"].get("through_sag_mm", 1.0) / 1000.0
        self.k_path = r.W / sag                                           # per node j + 1: its segment j's weight
        self.counts["through"] = {"path_off_head_share": round(float(off.mean()), 3),
                                  "path_mm_p50": round(float(1000 * np.median(cum[-1])), 1),
                                  "over_shoulder": bool(self.v["brush"].get("over_shoulder", False))}
        if self.ridge is not None:
            self.counts["through"]["path_end_vs_shoulder"] = shoulder_end(self, Q[:, -1])
        if self.back is not None:
            zs, xbp, xbn = self.back
            zt = zs[-1]
            self.counts["through"]["back_limit_mm"] = {
                "%d_below_crest" % d: [round(1000 * float(np.interp(zt - d / 1000.0, zs, xbn)), 1),
                                       round(1000 * float(np.interp(zt - d / 1000.0, zs, xbp)), 1)]
                for d in (0, 100, 200, 300, 400)}
            self.log("  acts: the back's lateral limit (-x, +x) mm %s" % self.counts["through"]["back_limit_mm"])
        if tuck:
            entered = (ph_t >= 1).any(0)
            self.counts["through"]["tuck"] = {
                "entered": int(entered.sum()), "tucked": int(use.sum()), "too_short": int((entered & ~use).sum()),
                "tucked_nodes": int(self.tuck_pin.sum()),
                "per_ear": {str(e_s): int((use & (e_s * (self.rods.X[:, 0] - self.scalp.c[0]) > 0)).sum())
                            for e_s in self.scalp.ears}}

    def _trace(self, arcs, ds, tuck):
        """one brush pass for every guide (see _trace_paths) -> path (n + 1, G, 3), its arcs (n + 1, G), off-head
        flags (G,), tuck phase per step (n + 1, G)"""
        r, body = self.rods, self.body
        G = r.G
        n_st = int(np.ceil(arcs.max() / ds)) + 2
        x = r.X.astype(np.float64).copy()
        path = np.empty((n_st + 1, G, 3))
        path[0] = x
        phs = np.zeros((n_st + 1, G), np.int8)
        ph = np.zeros(G, np.int8)
        eo = np.zeros(G, np.int8)
        off = np.zeros(G, bool)
        sg = self.side.astype(np.float64)
        dprev = None
        down = np.array([0.0, 0.0, -1.0])
        bv = self.v["brush"]
        sh_on = bool(bv.get("over_shoulder", False)) and self.ridge is not None
        sh_mm = float(bv.get("shoulder_mm", 40.0)) / 1000.0
        for k in range(n_st):
            _, g = body.sample(x, grad=True)
            n = _unit(g)
            b, mag = self.stroke(x, sg, n, self.wf_root)
            if dprev is not None:
                # a weak field (|F_t| < 0.05) gives no heading: keep the last one, in the skin's tangent plane
                keep = _unit(dprev - (dprev * n).sum(1, keepdims=True) * n)
                b = np.where((mag < 0.05)[:, None], keep, b)
            on = np.zeros(G, bool)
            if tuck:
                b, on = self._tuck_turn(x, n, b, ph, eo)
            aim = None
            if sh_on:
                # the stroke's end: below the ears (on the neck, on the skin) the brush heads back and down over the
                # shoulder toward the point shoulder_mm behind its crest, at the crest's height
                aim, ok = self._shoulder_aim(x, sh_mm)
                neck = ok & ~on & (x[:, 2] < self.z_eb - 0.01)
                if neck.any():
                    at = aim[neck] - (aim[neck] * n[neck]).sum(1, keepdims=True) * n[neck]
                    good = np.sqrt((at * at).sum(1)) > 1e-6
                    idx = np.nonzero(neck)[0][good]
                    b[idx] = _unit(at[good])
            if k * ds > 0.03:
                # a tucked strand just out of the groove is still in the hand, laid back along the neck below the
                # ear (the mastoid's underside does not drop it in front of the shoulder)
                held = on | ((ph == 3) & (x[:, 2] > self.z_eb - 0.04))
                off |= (n[:, 2] < -0.25) & ~held
            if aim is None:
                d = np.where(off[:, None], down, b)
            else:
                # off the head the brush is drawn toward behind the shoulder until it is there, then straight down -
                # with brush.blade, down the back toward the shoulder blade (_blade)
                dn = np.broadcast_to(down, x.shape) if self.back is None else self._blade(x, arcs[:, -1] - k * ds)
                d = np.where(off[:, None], np.where(ok[:, None], aim, dn), b)
            xn = x + ds * d
            sd2, g2 = body.sample(xn, grad=True)
            n2 = _unit(g2)
            corr = np.where(off, np.minimum(sd2, 0.0), sd2)               # on the skin; off it, only pushed out
            xn = xn - corr[:, None] * n2
            if on.any():
                # the hand lays the tucked span on the skull (1.5 mm off the scalp: the section's half thickness),
                # never on the ear's own face, then pushes it out of the body if the collider is fuller there
                for e_s in self.scalp.ears:
                    m = on & (eo == e_s)
                    if not m.any():
                        continue
                    la = self._skull_lat(e_s, xn[m, 1], xn[m, 2])
                    ok = np.isfinite(la)
                    idx = np.nonzero(m)[0][ok]
                    xn[idx, 0] = self.scalp.c[0] + e_s * (la[ok] + 0.0015)
                if on.any():
                    sd3, g3 = body.sample(xn[on], grad=True)
                    xn[on] -= np.minimum(sd3 - 0.001, 0.0)[:, None] * _unit(g3)
            dprev = d
            x = xn
            path[k + 1] = x
            phs[k + 1] = ph
        seg = np.sqrt(((path[1:] - path[:-1]) ** 2).sum(2))              # (n_st, G)
        cum = np.concatenate([np.zeros((1, G)), np.cumsum(seg, 0)], 0)
        return path, cum, off, phs

    def _lag_brush(self, view):
        P = view.P
        G, Ka = view.D.shape[:2]
        M = 0.5 * (P[:, :-1] + P[:, 1:]).reshape(-1, 3)
        sd, g = self.body.sample(M, grad=True)
        n = _unit(g)
        sg = np.repeat(self.side, Ka).astype(np.float64)
        bt, mag = self.stroke(M, sg, n, np.repeat(self.wf_root, Ka))
        r = self.v["brush"]["reach_mm"] / 1000.0
        # the bristles' reach from the skin; a brush or hand drawn through with no hold (through_hold 'none') is in
        # the hair at every depth, down to the scalp, so the heading pull acts wherever the strand is
        if self.pass_win is not None:
            near = self.pass_win[:, :Ka].ravel()          # a pass: the tool's window, at every depth
        else:
            near = 1.0 if self.free_pass else (1 - smoothstep(0.5 * r, r, sd))
        a = near * np.clip(mag / 0.1, 0.0, 1.0)
        self.bn[:, :Ka] = n.reshape(G, Ka, 3)
        self.bb[:, :Ka] = bt.reshape(G, Ka, 3)
        self.ba[:, :Ka] = (a * view.act.ravel()).reshape(G, Ka)
        if self.tuck_on and self.through and getattr(self, "tuck_seg", None) is not None and self.tuck_seg.any():
            # on the tucked span the brush is the hand's tuck: the bristles lead the hair along that path (over the
            # ear and down the groove), not along the side's down stroke
            ts = self.tuck_seg[:, :Ka]
            nn = self.bn[:, :Ka]
            T = self.T_path[:, :Ka]
            Tt = _unit((T - (T * nn).sum(2, keepdims=True) * nn).reshape(-1, 3)).reshape(G, Ka, 3)
            self.bb[:, :Ka][ts] = Tt[ts]
            near = (1 - smoothstep(0.5 * r, r, sd)).reshape(G, Ka)
            self.ba[:, :Ka][ts] = (near * view.act)[ts]
        if self.on_g is not None:
            self.ba[:, :Ka] *= self.on_g[:, None]          # the brush holds only the section being worked

    def _lag_face(self, view):
        Y = view.Y
        sd = self.body.sample(Y)
        near = sd < 0.04
        fl = np.zeros(len(Y), bool)
        if near.any():
            # every core (the same answers): one core spent ~0.38 s here per lag on the full head - 40 % of a styling
            # pass on the GPU settle (10-31 profile)
            _, vi = self.tree.query(Y[near], workers=-1)
            fl[np.nonzero(near)[0]] = self.face_v[vi]
        if self.on_g is not None:
            fl &= self.on_g[view.state.sq // view.state.Ka]   # the hand shields the face from the section in work
        self.face_flag = fl

    def _lag_tuck(self, view):
        """find the strands crossing an ear (first node over / in front of it) and move their pin targets"""
        P = view.P
        sc, tk = self.scalp, self.v["tuck"]
        for e_s, e in sc.ears.items():
            new = (self.tuck_j < 0)
            if not new.any():
                break
            x, y, z = P[..., 0], P[..., 1], P[..., 2]
            inb = (e_s * (x - sc.c[0]) > e["x_out"] - 0.03 if "x_out" in e else e_s * x > 0.055) & \
                (y > e["y_front"] - 0.02) & (y < e["y_back"]) & (z > e["z_bot"]) & (z < e["z_top"] + 0.015)
            inb[:, 0] = False
            hit = new & inb.any(1)
            for i in np.nonzero(hit)[0]:
                j = int(np.argmax(inb[i]))
                self.tuck_j[i], self.tuck_phase[i], self.tuck_e[i] = j, 1, e_s
        for i in np.nonzero(self.tuck_j >= 0)[0]:
            e_s = int(self.tuck_e[i])
            e = sc.ears[e_s]
            j = min(int(self.tuck_j[i]), P.shape[1] - 1)
            if self.tuck_phase[i] == 1:
                yq = 0.5 * (e["y_front"] + e["y_back"])
                zq = e["z_top"] + tk["over_mm"] / 1000 + 0.008
            else:
                yq = e["y_back"] + tk["behind_mm"] / 1000
                zq = e["z_top"] + tk["over_mm"] / 1000
            xq = self._skull_x(e_s, yq, zq) + e_s * 0.003
            q = np.array([xq, yq, zq])
            if self.tuck_phase[i] == 1 and np.linalg.norm(P[i, j] - q) < 0.006:
                self.tuck_phase[i] = 2
            self.tuck_q[i] = q
        self.counts["tucked"] = int((self.tuck_j >= 0).sum())

    def _skull_x(self, e_s, y, z):
        """the head's lateral skin at (y, z) beside the ear, ear vertices left out"""
        V = self.scalp.V
        m = (~self.scalp.ear_mask) & (np.abs(V[:, 1] - y) < 0.006) & (np.abs(V[:, 2] - z) < 0.006) & (e_s * V[:, 0] > 0)
        if not m.any():
            return e_s * 0.075
        return float(V[m, 0].max()) if e_s > 0 else float(V[m, 0].min())

    # -------------------------------------------------- the hook
    def stage(self, view, index, arc):
        if self.hand_on and self.y_sh is not None:
            below = (view.P[..., 2] < self.z_sh + 0.03).any(1) & (self.hand_w > 0)
            if below.any() and not self.hand_started:
                self.hand_started = True
            if self.hand_started:
                if self.hand_left <= 0:
                    self.hand_on = False
                    self.log("  acts: hand released")
                else:
                    self.hand_left -= 1
                    self.counts["hand_stages"] += 1
        self.lag(view)

    def lag(self, view):
        if self.hand_on and self.z_sh is not None:
            self.hand_low = view.P[..., 2] < self.z_sh + 0.03
        if self.tousle_field is not None:
            self.tousle_F = self.tousle_field(view.P)
        if self.brush_on:
            self._lag_brush(view)
        if self.face_on:
            self._lag_face(view)
        if self.tuck_on and self.through:
            pass                               # the tuck is the side stroke's end (_tuck_turn): no pin
        elif self.tuck_on:
            if self.released:
                # the first release solve is done: the ear's contact holds the hair now
                self.tuck_on = False
                self.log("  acts: tuck released (%d strands tucked)" % self.counts["tucked"])
            else:
                self._lag_tuck(view)

    def release(self):
        """every act off except the pins that stay: the tuck until the first release solve is done (tie pins would
        stay for good)"""
        self.brush_on = self.face_on = self.hand_on = False
        self.released = True

    def energy(self, view):
        D, P, Y = view.D, view.P, view.Y
        G, Ka = D.shape[:2]
        E = 0.0
        gD = gP = gY = None
        st = view.state
        if self.brush_on:
            b = self.v["brush"]
            n, bt, a = self.bn[:, :Ka], self.bb[:, :Ka], self.ba[:, :Ka]
            cb = st.cb
            kw = 0.5 * b["strength"] * cb * a
            if b["mode"] == "direction":
                c = (D * bt).sum(2)
                gD = -kw[..., None] * bt
            else:
                dt = D - (D * n).sum(2, keepdims=True) * n
                q = np.sqrt((dt * dt).sum(2) + b["eps"] ** 2)
                c = (dt * bt).sum(2) / q
                gD = -kw[..., None] * (bt / q[..., None] - (c / q ** 2)[..., None] * dt)
            E += float((kw * (1 - c)).sum())
        on_all = self.brush_on or self.holding
        pin = (not on_all) and self.released and bool(self.v["tuck"].get("hold", False)) and             getattr(self, "tuck_pin", None) is not None
        if self.through and self.Qp is not None and (on_all or pin) and not self.free_pass:
            # the bristles hold each node within reach of the brush at its arc: 1/2 k (|P - Q| - reach)_+^2; after
            # release, with tuck.hold, only the tucked nodes, within tuck.behind_mm of where the hand laid them
            r_b = (self.v["brush"]["reach_mm"] if on_all else self.v["tuck"]["behind_mm"]) / 1000.0
            if on_all and self.tuck_on and self.tuck_pin is not None:
                # the tucked span is held by the hand, within tuck.hand_mm of its path (per node)
                r_b = np.where(self.tuck_pin[:, 1:Ka + 1], self.v["tuck"]["hand_mm"] / 1000.0, r_b)
            dP = P[:, 1:] - self.Qp[:, 1:Ka + 1]
            dist = np.sqrt((dP * dP).sum(2))
            # through_hold 'steer': only the sideways offset from the brush's lane counts (the strand slides along
            # the brush, its height its own); the hand's tucked span and the post-release tuck pin stay in position
            gdir = dP / np.maximum(dist, 1e-12)[..., None]
            if self.steer and on_all:
                Bn = self.B_path[:, :Ka]
                xl = (dP * Bn).sum(2)
                lane = np.ones(dist.shape, bool) if self.tuck_pin is None else ~self.tuck_pin[:, 1:Ka + 1]
                dist = np.where(lane, np.abs(xl), dist)
                gdir = np.where(lane[..., None], np.sign(xl)[..., None] * Bn, gdir)
            ex = dist - r_b
            h = (ex > 0) & view.act
            if self.on_g is not None:
                h &= self.on_g[:, None]
            if pin:
                h &= self.tuck_pin[:, 1:Ka + 1]
            if h.any():
                k = self.k_path[:, :Ka]
                E += 0.5 * float((k[h] * ex[h] ** 2).sum())
                gP = np.zeros_like(P) if gP is None else gP
                gP[:, 1:] += np.where(h, k * ex, 0.0)[..., None] * gdir
        if self.drag is not None:
            # the tool's drag (passes.stroke_pass): each sample in the window is pulled toward where the tool carries
            # it, in the skin's tangent plane only (no normal part: it cannot press hair onto the scalp or lift it),
            # by a smoothed |x| whose force never exceeds C (the bristles' grip) and is zero once the hair has caught
            # up - friction from a moving tool, gone after the step, not a pin
            # within `reach` of the tool (the bristles' depth) the hair is not pulled at all
            dg = self.drag
            idx = dg["idx"]
            d = Y[idx] - dg["a"]
            nn = dg["n"]
            dt = d - (d * nn).sum(1, keepdims=True) * nn
            e = dg["eps"]
            gY = np.zeros_like(Y) if gY is None else gY
            if dg.get("d") is not None:
                # the bristles (passes.sweep_pass, 10-31 night): a brush's bristles sit BETWEEN the strands, so across
                # a strand the hair is carried as by a solid - the skin's own contact law, 1/2 k_c w x_p^2 on the part
                # of the lag across the strand's direction d (lagged at the step's start) - and along the strand it
                # slides through them by friction, the smoothed |x| capped at C as before. Measured before this (the
                # friction-only drag, sweep_probe on the full head): a held sample moved a median 8 % of the brush's
                # step and 98 % were dropped mid-stroke - the brush lifted the hair and let it go. Tangent plane only:
                # it cannot press hair onto the scalp or lift it off; gone after the step.
                sd_ = dg["d"]
                xa = (dt * sd_).sum(1)
                xp = dt - xa[:, None] * sd_
                q = np.sqrt(1.0 + xa * xa / (e * e))
                kp = dg["kp"]
                E += float((dg["C"] * e * (q - 1.0)).sum()) + 0.5 * float((kp * (xp * xp).sum(1)).sum())
                g_ = (dg["C"] * xa / (e * q))[:, None] * sd_ + kp[:, None] * xp
                gY[idx] += g_ - (g_ * nn).sum(1, keepdims=True) * nn
            else:
                rr = float(dg.get("reach", 0.0))
                dl = np.sqrt((dt * dt).sum(1))
                ex = np.maximum(dl - rr, 0.0)
                q = np.sqrt(1.0 + ex * ex / (e * e))
                E += float((dg["C"] * e * (q - 1.0)).sum())
                gY[idx] += (dg["C"] * ex / (e * q * np.maximum(dl, 1e-12)))[:, None] * dt
        if getattr(self, "lift", None) is not None:
            # the face hand's fingers under the roots (11-10, brush.fingers_face_lift_mm): each held sample near its root
            # is kept at least h off the skin - a one-sided push along the skin's normal, the face barrier's law; gone
            # with the hand
            lf = self.lift
            sd, g = self.body.sample(Y[lf["idx"]], grad=True)
            pen = lf["h"] - sd
            hh = pen > 0
            if hh.any():
                k = lf["k"][hh]
                E += 0.5 * float((k * pen[hh] ** 2).sum())
                gY = np.zeros_like(Y) if gY is None else gY
                gY[lf["idx"][hh]] += -(k * pen[hh])[:, None] * g[hh]
        if self.face_on and self.face_flag is not None and len(self.face_flag) == len(Y) and self.face_flag.any():
            m = self.v["face"]["margin_mm"] / 1000.0
            idx = np.nonzero(self.face_flag)[0]
            sd, g = self.body.sample(Y[idx], grad=True)
            pen = m - sd
            h = pen > 0
            if h.any():
                k = st.k_c * view.sample_w[idx[h]]
                E += 0.5 * float((k * pen[h] ** 2).sum())
                gY = np.zeros_like(Y) if gY is None else gY      # the drag may have written it
                gY[idx[h]] += -(k * pen[h])[:, None] * g[h]
        if self.tuck_on and (self.tuck_j >= 0).any():
            gP = np.zeros_like(P) if gP is None else gP
            for i in np.nonzero(self.tuck_j >= 0)[0]:
                j = min(int(self.tuck_j[i]), P.shape[1] - 1)
                d = P[i, j] - self.tuck_q[i]
                E += 0.5 * self.k_tuck[i] * float(d @ d)
                gP[i, j] += self.k_tuck[i] * d
        if self.hand_on and self.y_sh is not None:
            hp = self.v["hand"]
            yl = self.y_sh - hp["front_mm"] / 1000.0
            low = self.hand_low if (self.hand_low is not None and self.hand_low.shape == P.shape[:2]) else                 np.zeros(P.shape[:2], bool)
            pen = np.where(low, P[..., 1] - yl, 0.0) * (self.hand_w[:, None] > 0)
            if self.on_g is not None:
                pen = pen * self.on_g[:, None]
            pen[:, 0] = 0.0
            h = pen > 0
            if h.any():
                k = (self.hand_w * self.rods.Wtot / 0.002)[:, None] * np.ones_like(pen)
                E += 0.5 * float((k[h] * pen[h] ** 2).sum())
                gP = np.zeros_like(P) if gP is None else gP
                gP[..., 1] += np.where(h, k * pen, 0.0)
        if self.tousle_F is not None and self.tousle_F.shape == P.shape:
            Fn = self.tousle_F                                          # (G, Ka + 1, 3) N, lagged
            E += -float((Fn * P).sum())
            gP = np.zeros_like(P) if gP is None else gP
            gP -= Fn
        if gP is not None:
            gP[:, 0] = 0.0
        return E, {"D": gD, "P": gP, "Y": gY}

    def precond(self, view):
        """the brush's stiffness on each segment's direction (J per rad^2)"""
        if not self.brush_on:
            return None
        Ka = view.D.shape[1]
        return self.v["brush"]["strength"] * view.state.cb * self.ba[:, :Ka]

    # -------------------------------------------------- tousle
    def tousle_stroke(self, k):
        """switch on stroke k's shake: a seeded smooth random force field, tousle.force segment weights, over the crown
        (theta < 70 deg) and within 40 mm of the skin"""
        t = self.v["tousle"]
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)) + k)
        fx, fy, fz = (_noise_field(rng) for _ in range(3))
        L = t["scale_mm"] / 1000.0
        Wn = np.concatenate([self.rods.W, np.zeros((self.rods.G, 1))], 1)   # (G, K + 1)
        sc, body = self.scalp, self.body

        def field(P):
            G, n1 = P.shape[:2]
            Q = P.reshape(-1, 3)
            _, th, _ = sc.sph(Q)
            w = (1 - smoothstep(60.0, 80.0, th)) * (1 - smoothstep(0.02, 0.04, body.sample(Q)))
            F = np.stack([fx(Q, L), fy(Q, L), fz(Q, L)], 1) * w[:, None]
            return t["force"] * F.reshape(G, n1, 3) * Wn[:, :n1, None]
        self.tousle_field = field

    def tousle_off(self):
        self.tousle_field = None
        self.tousle_F = None


# ---------------------------------------------------------------- the set and the whole styled settle
def dry_set(state, acts):
    """the water set after the cut (DESIGN_v2 4.7): over s <= dry.over_mm, rho <- rho + set (dd - rho), dd = the
    current turning at each joint (d_j - d_{j-1}, d_{-1} the follicle direction); dry.lift_deg adds a rest turning
    toward the skin normal spread over the same span (round-brush lift). dry.method "blow": blow_set. Returns the
    number of joints set."""
    r = state.rods
    dv = acts.v["dry"]
    if dv.get("span", "held") == "held":
        if dv.get("hold", False) and int(dv.get("hold_evals", 0)) > 0:
            # the set is the shape the section is held in as it dries and cools: dry_hold takes it at its end
            acts.set_pending = True
            acts.counts["set"] = {"deferred_to_end_of_hold": True}
            return int(_held_mask(state, acts).sum())
        return set_held(state, acts)
    if dv.get("method", "air") == "blow":
        return blow_set(state, acts)
    Ka = state.Ka
    D = r.D[:, :Ka]
    Dp = np.concatenate([r.d0[:, None], D[:, :-1]], 1)
    dd = D - Dp
    over = dv["over_mm"] / 1000.0
    m = state.act & (r.s[:, :Ka] < over)
    s = float(dv["set"])
    rho = r.rho[:, :Ka]
    rho[m] += s * (dd[m] - rho[m])
    if dv["lift_deg"] > 0 and m.any():
        P = state.view().P
        M = 0.5 * (P[:, :-1] + P[:, 1:])
        _, g = acts.body.sample(M[m], grad=True)
        n = _unit(g)
        d = D[m]
        t = _unit(n - (n * d).sum(1, keepdims=True) * d)
        frac = np.minimum(r.l[:, :Ka][m], np.maximum(over - r.s[:, :Ka][m], 0.0)) / over
        rho[m] += (np.radians(dv["lift_deg"]) * frac)[:, None] * t
    r.rho[:, :Ka] = rho
    return int(m.sum())


def _slerp(a, b, t):
    """unit vectors a -> b, fraction t (rows)"""
    c = np.clip((a * b).sum(-1), -1.0, 1.0)
    w = np.arccos(c)
    sw = np.sin(w)
    small = sw < 1e-6
    sw = np.where(small, 1.0, sw)
    ka = np.where(small, 1 - t, np.sin((1 - t) * w) / sw)
    kb = np.where(small, t, np.sin(t * w) / sw)
    return _unit(ka[..., None] * a + kb[..., None] * b)


def blow_set(state, acts):
    """a round-brush blow-dry (hair_styling 7.1 rows 11-12, T2 round brush / base control / hairline swept back): the
    stylist holds each section at its zone's base angle e from the direction it is dried toward (the brush stroke b at
    the root, in the skin's tangent plane) - on-base 135 deg above the parietal ridge and at the front hairline (dried
    forward over the face, then rolled back: 'swept back = dried forward, then back'), half-off 90 deg from the ridge to
    the occipital, off-base 45 deg below - and winds the first dry.over_mm of it on a round brush of radius R =
    brush_mm / 2 rolled toward b, heats and cools it there. The fibre leaves its follicle along d0 and turns to the held
    direction d_h = cos e b + sin e n over root_mm, then curves on the brush at 1 / R in the plane of b and n. The
    hydrogen bonds re-form in that shape: over the wound span rho <- (1 - set) rho + set (the brush shape's turning).
    Nothing here names a height: the root lift is what the set does under the hair's weight. -> joints set
    (dry.span 'band'; span 'held' takes the same brush shape over the wound span in set_held)"""
    r = state.rods
    dv = acts.v["dry"]
    Ka = state.Ka
    d0 = r.d0
    ddb, sm, R, e, w_on, w_half, w_off, lb = _blow_shape(state, acts)
    over = dv["over_mm"] / 1000.0
    r0 = dv["root_mm"] / 1000.0
    m = state.act & (sm < over)
    s = float(dv["set"])
    rho = r.rho[:, :Ka]
    rho[m] += s * (ddb[m] - rho[m])
    hs = float(dv.get("hold_set", 0.0))
    if hs > 0:
        # the lengths dried while held on the stroke keep a share of the turning they were held in
        D = r.D[:, :Ka]
        dd = D - np.concatenate([d0[:, None], D[:, :-1]], 1)
        ml = state.act & (sm >= over)
        rho[ml] += hs * (dd[ml] - rho[ml])
    r.rho[:, :Ka] = rho
    # the set's curvature (rad / m) against the brush's 1 / R
    ks = np.sqrt((ddb * ddb).sum(2)) / lb
    kR = np.where(m, ks * R[:, None], 0.0)
    acts.counts["blow"] = {
        "joints": int(m.sum()), "set": s,
        "base_deg_p10_50_90": np.round(np.degrees(np.percentile(e, [10, 50, 90])), 1).tolist(),
        "on_base_share": round(float((w_on > 0.5).mean()), 3), "half_off_share": round(float((w_half > 0.5).mean()), 3),
        "off_base_share": round(float((w_off > 0.5).mean()), 3),
        "brush_radius_mm_p10_50_90": np.round(1000 * np.percentile(R, [10, 50, 90]), 1).tolist(),
        "turn_on_brush_deg_p50": round(float(np.degrees(np.median((over - r0) / R))), 1),
        "brush_shape_curvature_x_R_max": round(float(kR.max()), 4),
        "set_rest_curvature_x_R_max_wound": round(float(np.where(
            m, np.sqrt((rho * rho).sum(2)) / lb * R[:, None], 0.0).max()), 4),
        "hold_set": hs}
    return int(m.sum())


def _blow_shape(state, acts):
    """the round brush's shape for blow_set / set_held: each joint's turning (d_j - d_{j-1}) on the brush, capped at
    lbar_j / R. -> (ddb (G, Ka, 3), segment mid arcs sm (m), R (m), base angle e (rad), w_on, w_half, w_off, lbar)"""
    r = state.rods
    dv = acts.v["dry"]
    Ka = state.Ka
    X, d0 = r.X, r.d0
    _, g = acts.body.sample(X + 0.001 * d0, grad=True)
    n = _unit(g)
    b, _ = acts.stroke(X, acts.side.astype(np.float64), n)
    phi, th, _ = acts.scalp.sph(X)
    aphi = np.abs(phi)
    e_on, e_half, e_off = (float(x) for x in dv["base_deg"])
    w_on = np.maximum(1 - smoothstep(dv["ridge_deg"] - 5, dv["ridge_deg"] + 5, th),
                      1 - smoothstep(dv["front_phi_deg"] - 10, dv["front_phi_deg"] + 10, aphi))
    w_half = (1 - w_on) * smoothstep(105.0, 125.0, aphi) * (1 - smoothstep(dv["occiput_deg"] - 5,
                                                                            dv["occiput_deg"] + 5, th))
    w_off = 1 - w_on - w_half
    e = np.radians(w_on * e_on + w_half * e_half + w_off * e_off)
    wc = 1 - smoothstep(dv["crown_deg"] - 5, dv["crown_deg"] + 5, th)
    R = 0.5e-3 * (wc * dv["crown_brush_mm"] + (1 - wc) * dv["brush_mm"])
    d_h = np.cos(e)[:, None] * b + np.sin(e)[:, None] * n
    sm = r.s[:, :Ka] + 0.5 * r.l[:, :Ka]                       # segment mid arcs (m)
    r0 = dv["root_mm"] / 1000.0
    over = dv["over_mm"] / 1000.0
    # the directions on the brush
    t0 = np.clip(sm / r0, 0.0, 1.0)
    Dr = _slerp(np.repeat(d0[:, None], Ka, 1), np.repeat(d_h[:, None], Ka, 1), t0)
    a = e[:, None] - np.maximum(sm - r0, 0.0) / R[:, None]
    Dw = np.cos(a)[..., None] * b[:, None] + np.sin(a)[..., None] * n[:, None]
    Dt = np.where((sm <= r0)[..., None], Dr, Dw)
    # a round brush of radius R cannot bend the fibre tighter than 1 / R: each joint turns toward the brush's
    # direction by at most lbar_j / R (a 27-100 deg turn within 0.5-4 mm at the root was a 1-2 mm bend radius, gr1)
    Db = np.empty_like(Dt)
    prev = d0.astype(np.float64).copy()
    lb = r.lbar[:, :Ka]
    for j in range(Ka):
        tgt = Dt[:, j]
        ang = np.arccos(np.clip((prev * tgt).sum(1), -1.0, 1.0))
        amax = lb[:, j] / R
        t = np.where(ang > amax, amax / np.maximum(ang, 1e-12), 1.0)
        Db[:, j] = _slerp(prev, tgt, t)
        prev = Db[:, j]
    Dp = np.concatenate([d0[:, None], Db[:, :-1]], 1)
    ddb = Db - Dp
    return ddb, sm, R, e, w_on, w_half, w_off, lb


# the set's decay (hair_styling point 7, D11, 7.1 rows 20 / 25). Published: curl loss is fast at first and much slower
# after ~2 h at 90 % RH (P25, P27); a fixative benchmark keeps >= 70 % after 0.75 h at 90 % RH, a good fixative 90 %
# after 8 h, a weak one 45 % (P27, P55); a blow-dry lasts 2-4 days at room humidity (P29, P35). No per-hour curve is
# published (hair_styling's gaps), so the curve below is the simplest one that meets those numbers: two exponentials
# on an equivalent time at 90 % RH, t90 = age x (RH / 90)^RH_EXP x 10^(-2 hold).
RET_FAST, RET_TAU_FAST_H, RET_TAU_SLOW_H = 0.7, 0.5, 24.0   # unfixed at 90 % RH: 45 % at 0.75 h, 29 % at 2 h, 21 % at 8 h
RET_RH_EXP = 5.2        # room humidity (45 %) runs the clock 36x slower: 29 % left after 72 h ('lasts 2-4 days')


def retention(age_h, rh, hold=0.0):
    """the share of a fresh set left after age_h hours at rh % relative humidity under a fixative of strength hold
    (0-1): 1 at age 0. Checks against the sources: unfixed 90 % RH 0.75 h 0.45 (under the >= 0.70 a fixative must
    give); hold 0.5 at 8 h 90 % RH 0.43 (weak fixative: 0.45); hold 1 at 8 h 0.90 (good fixative); unfixed 72 h at 45 % RH
    0.29. -> float or array"""
    t = np.asarray(age_h, np.float64) * (np.asarray(rh, np.float64) / 90.0) ** RET_RH_EXP \
        * 10.0 ** (-2.0 * np.asarray(hold, np.float64))
    return RET_FAST * np.exp(-t / RET_TAU_FAST_H) + (1 - RET_FAST) * np.exp(-t / RET_TAU_SLOW_H)


def _held_mask(state, acts):
    """the joints the brush or hand held while the hair dried: with brush.through the bristles hold every active point
    of the strand (the stroke drawn through to the ends, the tuck down the groove); otherwise the brush works only near
    the skin, the first dry.over_mm"""
    r = state.rods
    Ka = state.Ka
    on = state.act if getattr(acts, "on_g", None) is None else state.act & acts.on_g[:, None]
    if acts.v["brush"].get("through", False):
        return on.copy()
    sm = r.s[:, :Ka] + 0.5 * r.l[:, :Ka]
    return on & (sm < acts.v["dry"]["over_mm"] / 1000.0)


def set_held(state, acts):
    """dry.span 'held': the set as the shape the hair dried in. Every held joint takes rest turning
    rho <- rho + r (held - rho): rho the natural turning (the follicle's wave), held the turning it is held in now (at
    the end of dry.hold), r = dry.set x retention(wear.age_h, wear.rh, finish.hold) per guide. With dry.method 'blow'
    the span wound on the round brush (dry.over_mm) takes the brush's shape instead, whose turning is capped at
    lbar / R (a brush of radius R sets no tighter curvature than 1 / R). -> joints set"""
    r = state.rods
    dv, wv, fv = acts.v["dry"], acts.v["wear"], acts.v["finish"]
    Ka = state.Ka
    d0 = r.d0
    m = _held_mask(state, acts)
    D = r.D[:, :Ka]
    held = D - np.concatenate([d0[:, None], D[:, :-1]], 1)
    sm = r.s[:, :Ka] + 0.5 * r.l[:, :Ka]
    lb = r.lbar[:, :Ka]
    wound = np.zeros_like(m)
    kR = None
    if dv.get("method", "air") == "blow":
        ddb, sm, R, e, w_on, w_half, w_off, lb = _blow_shape(state, acts)
        wound = m & (sm < dv["over_mm"] / 1000.0)
        held = np.where(wound[..., None], ddb, held)
        kR = R
    sprayed = np.ones(r.G, bool) if fv["where"] == "all" else acts.front_g.copy()
    hold_g = np.where(sprayed, float(fv["hold"]), 0.0)
    ret_g = retention(float(wv["age_h"]), float(wv["rh"]), hold_g)
    rg = float(dv["set"]) * ret_g
    rj = np.broadcast_to(rg[:, None], m.shape)
    rho = r.rho[:, :Ka]
    rho[m] += rj[m][:, None] * (held[m] - rho[m])
    over = dv["over_mm"] / 1000.0
    mb = state.act & (r.s[:, :Ka] < over)
    if getattr(acts, "on_g", None) is not None:
        mb &= acts.on_g[:, None]
    if dv.get("method", "air") != "blow" and dv["lift_deg"] > 0 and mb.any():
        # the air-dry root lift, as dry_set's band: a rest turning toward the skin normal spread over over_mm
        P = state.view().P
        M = 0.5 * (P[:, :-1] + P[:, 1:])
        _, g = acts.body.sample(M[mb], grad=True)
        n = _unit(g)
        d = D[mb]
        t = _unit(n - (n * d).sum(1, keepdims=True) * d)
        frac = np.minimum(r.l[:, :Ka][mb], np.maximum(over - r.s[:, :Ka][mb], 0.0)) / over
        rho[mb] += (np.radians(dv["lift_deg"]) * frac)[:, None] * t
    r.rho[:, :Ka] = rho
    if float(dv.get("hold_set", 0.0)) > 0 and "hold_set" not in acts._warned:
        acts._warned.add("hold_set")
        acts.warnings.append("dry.hold_set is an alias for span 'band': with span 'held' the whole held length takes "
                             "dry.set x retention; hold_set ignored")
    tdeg = np.degrees(np.sqrt((held * held).sum(2)) / np.maximum(lb, 1e-9) * 0.001)     # held curvature, deg / mm
    beyond = m & ~wound
    if getattr(acts, "on_g", None) is not None and acts.P_held is not None:
        acts.P_held[acts.on_g] = r.positions()[acts.on_g]   # each section's held shape, taken at the end of its hold
    else:
        acts.P_held = r.positions().copy()
    acts.counts["set"] = {
        "span": "held", "joints": int(m.sum()), "joints_wound": int(wound.sum()), "set": float(dv["set"]),
        "wear": {"age_h": float(wv["age_h"]), "rh": float(wv["rh"])}, "finish": dict(fv),
        "retention_unsprayed": round(float(retention(float(wv["age_h"]), float(wv["rh"]), 0.0)), 4),
        "retention_sprayed": round(float(retention(float(wv["age_h"]), float(wv["rh"]), float(fv["hold"]))), 4),
        "sprayed_guides": int(sprayed.sum()), "r_min_max": [round(float(rg.min()), 4), round(float(rg.max()), 4)],
        "held_curvature_deg_per_mm_p50_p90_beyond_wound": (np.round(np.percentile(tdeg[beyond], [50, 90]), 3).tolist()
                                                            if beyond.any() else None),
        "set_rest_curvature_x_R_max_wound": (round(float(np.where(
            wound, np.sqrt((rho * rho).sum(2)) / lb * kR[:, None], 0.0).max()), 4) if kR is not None else None)}
    return int(m.sum())


def dry_hold(st, acts, log=print):
    """dry.hold: the section stays in the brush (the stroke drawn through, the heading brush, the face hand) while it
    dries and cools, dry.hold_evals evaluations; with dry.span 'held' the set is then taken from the shape it is held
    in (set_held), and release() lets go after. generate's grow branch calls this between dry_set and release."""
    dv = acts.v["dry"]
    if not dv.get("hold", False) or int(dv.get("hold_evals", 0)) <= 0:
        return
    acts.holding = True
    try:
        acts.lag(st.view())
        st.solve(int(dv["hold_evals"]), label="hold")
    finally:
        acts.holding = False
    if getattr(acts, "set_pending", False):
        acts.set_pending = False
        n = set_held(st, acts)
        c = acts.counts["set"]
        log(" dry: set from the held shape, %d joints (%d wound), r %s (retention %.3f, sprayed %.3f)" % (
            n, c["joints_wound"], c["r_min_max"], c["retention_unsprayed"], c["retention_sprayed"]))


def first_segment_deg(rods):
    """angle of each guide's first segment off its follicle direction d0, deg (p50 / p90 / max)"""
    c = np.clip((rods.D[:, 0] * rods.d0).sum(1), -1.0, 1.0)
    a = np.degrees(np.arccos(c))
    return [round(float(np.percentile(a, 50)), 2), round(float(np.percentile(a, 90)), 2), round(float(a.max()), 2)]


def shoulder_ridge(V, z_eb, c, step=0.01):
    """the shoulders' crest line from the collider's vertices V (m): for each lateral bin (step m wide, signed x
    from the head's centre c) the highest vertex lower than 70 mm under the ears' lower end z_eb (below the jaw and
    the chin), and the median y of the vertices within 15 mm under it: the trapezius's crest out to the acromion.
    Where the body reaches the cap (the neck, and the chin's underside in front of it) there is no crest: those bins
    are left NaN and ridge_at joins the two trapezius crests straight across, behind the neck's middle.
    -> (xs, y_r, z_r) (m), NaN where a bin has no vertex or no crest"""
    V = np.asarray(V, np.float64)
    xl = V[:, 0] - c[0]
    m = V[:, 2] < z_eb - 0.07
    xs = np.arange(-0.25, 0.25 + 1e-9, step)
    yr = np.full(len(xs), np.nan)
    zr = np.full(len(xs), np.nan)
    for i, x0 in enumerate(xs):
        b = m & (np.abs(xl - x0) < step / 2)
        if b.sum() < 3:
            continue
        zt = V[b, 2].max()
        if zt > z_eb - 0.075:
            continue                                   # the neck or the chin: no crest here
        top = b & (V[:, 2] > zt - 0.015)
        yr[i], zr[i] = float(np.median(V[top, 1])), float(zt)
    return xs + c[0], yr, zr


def back_table(V, ridge, c, step=0.01, dz=0.01, depth=0.45):
    """the back's lateral limit per height (brush.blade), from the collider's vertices V (m): at each height from the
    crest's lowest point down depth (m), the back contour (the most posterior y in each lateral bin step wide) walked
    out from the spine on each side while it stays within 45 deg of facing backward (|dy| <= the bin width per bin) -
    the back over the shoulder blades, where hanging lengths rest; past it the torso's side turns forward or the arm
    stands apart. -> (zs (ascending), limit on +x, limit on -x), m from the head's centre line"""
    V = np.asarray(V, np.float64)
    xl = V[:, 0] - c[0]
    z_top = float(np.nanmin(ridge[2]))
    zs = np.arange(z_top - depth, z_top + 1e-9, dz)
    xs = np.arange(-0.25, 0.25 + 1e-9, step)
    i0 = int(np.argmin(np.abs(xs)))
    xbp = np.full(len(zs), np.nan)
    xbn = np.full(len(zs), np.nan)
    for k, z in enumerate(zs):
        m = np.abs(V[:, 2] - z) < 0.5 * dz
        yb = np.full(len(xs), np.nan)
        if m.sum() < 3:
            continue
        bi = np.clip(np.round((xl[m] - xs[0]) / step).astype(int), 0, len(xs) - 1)
        for i in np.unique(bi):
            yb[i] = V[m, 1][bi == i].max()
        fin = np.nonzero(np.isfinite(yb))[0]
        if not len(fin):
            continue
        for sgn, arr in ((1, xbp), (-1, xbn)):
            # from the bin nearest the spine that has a vertex, outward; a bin the mesh left empty is stepped over
            # (the slope taken over the two bins)
            i = i1 = int(fin[np.argmin(np.abs(fin - i0))])
            while True:
                j = i + sgn if (0 <= i + sgn < len(xs) and np.isfinite(yb[i + sgn])) else i + 2 * sgn
                if not (0 <= j < len(xs) and np.isfinite(yb[j]) and abs(yb[j] - yb[i]) <= step * abs(j - i)):
                    break
                i = j
            arr[k] = abs(xs[i]) if i != i1 else np.nan     # no contour walked from the spine: filled from its neighbours
    for arr in (xbp, xbn):
        ok = np.isfinite(arr)
        arr[:] = np.interp(zs, zs[ok], arr[ok]) if ok.any() else 0.25
    return zs, xbp, xbn


def ridge_at(ridge, x):
    """the crest line's (y, z) (m) at lateral positions x (m), linear between bins; NaN outside the bins it has"""
    xs, yr, zr = ridge
    g = np.isfinite(yr)
    if not g.any():
        return np.full(len(x), np.nan), np.full(len(x), np.nan)
    lo, hi = xs[g].min(), xs[g].max()
    y = np.interp(x, xs[g], yr[g])
    z = np.interp(x, xs[g], zr[g])
    out = (x < lo) | (x > hi)
    y[out], z[out] = np.nan, np.nan
    return y, z


def shoulder_end(acts, E, band_mm=15.0, reach_mm=20.0):
    """where each length ends against the shoulders' crest line, E (G, 3) m the end points: 'reached' if the end is
    lower than reach_mm above the crest at its lateral offset; of those, behind (y more than band_mm behind the
    crest), in front (more than band_mm in front: over the chest) or on it. The ends that did not reach it are where
    the stroke ended (short hair)."""
    yr, zr = ridge_at(acts.ridge, E[:, 0])
    known = np.isfinite(yr)
    reached = known & (E[:, 2] < zr + reach_mm / 1000.0)
    dy = E[:, 1] - yr
    beh = reached & (dy > band_mm / 1000.0)
    fr = reached & (dy < -band_mm / 1000.0)
    on = reached & ~beh & ~fr
    n = max(int(reached.sum()), 1)
    return {"guides": int(len(E)), "reached": int(reached.sum()), "no_crest": int((~known).sum()),
            "behind": int(beh.sum()), "on": int(on.sum()), "in_front": int(fr.sum()),
            "behind_share_of_reached": round(float(beh.sum()) / n, 3),
            "in_front_share_of_reached": round(float(fr.sum()) / n, 3),
            "dy_mm_p10_50_90_reached": np.round(1000 * np.percentile(dy[reached], [10, 50, 90]), 1).tolist()
            if reached.any() else None}


def behind_vertex(acts, rods, arcs_mm=(100.0, 200.0)):
    """the front-hairline guides (acts.front_g) whose point at each arc lies behind the head's vertex (y greater than
    the y of the scalp's highest skin point): the lengths carried back over the top"""
    sc = acts.scalp
    V = sc.V[~sc.ear_mask] if hasattr(sc, "ear_mask") else sc.V
    yv = float(V[np.argmax(V[:, 2]), 1])
    P = rods.positions()
    A = rods.node_arcs()
    out = {"vertex_y_mm": round(1000 * yv, 1), "front_guides": int(acts.front_g.sum())}
    for s in arcs_mm:
        q = []
        for i in np.nonzero(acts.front_g)[0]:
            if A[i, -1] < s / 1000.0:
                continue
            q.append(np.interp(s / 1000.0, A[i], P[i, :, 1]))
        q = np.asarray(q)
        out["%dmm" % s] = {"reaching": int(len(q)), "behind_share": round(float((q > yv).mean()), 3) if len(q)
                           else None, "y_mm_p10_50_90": np.round(1000 * np.percentile(q, [10, 50, 90]), 1).tolist()
                           if len(q) else None}
    return out


def ear_rim(acts, rods, Q, near_mm=4.0, top_mm=6.0, rim_mm=6.0):
    """the ear left bare (look A: the hair passes above the helix and behind the ear): the share of the ear's lateral
    rim (ear vertices within rim_mm of its outermost x, below its top top_mm band) with a hair point within near_mm
    (target < 0.2); and the side guides (|phi| 60-120 on that side) that reach the ear's mid height: the share passing
    it behind the ear (y beyond the ear's back edge where they cross that height)"""
    sc = acts.scalp
    V = sc.V
    tree = cKDTree(Q)
    P = rods.positions()
    out = {}
    for e_s, e in sc.ears.items():
        ev = sc.ear_mask & (e_s * (V[:, 0] - sc.c[0]) > 0)
        if not ev.any():
            continue
        xo = e_s * (V[ev, 0] - sc.c[0])
        rim = np.nonzero(ev)[0][(xo > xo.max() - rim_mm / 1000.0) & (V[ev, 2] < e["z_top"] - top_mm / 1000.0)]
        cov = None
        if len(rim):
            d, _ = tree.query(V[rim], distance_upper_bound=near_mm / 1000.0)
            cov = round(float(np.isfinite(d).mean()), 3)
        zm = 0.5 * (e["z_top"] + e["z_bot"])
        side = (np.abs(acts.phi_r) > 60) & (np.abs(acts.phi_r) < 120) & (e_s * (rods.X[:, 0] - sc.c[0]) > 0)
        ys, xs = [], []
        for i in np.nonzero(side)[0]:
            z = P[i, :, 2]
            j = np.nonzero((z[:-1] >= zm) & (z[1:] < zm))[0]
            if len(j) == 0:
                continue
            j = int(j[0])
            t = (z[j] - zm) / max(z[j] - z[j + 1], 1e-12)
            ys.append(P[i, j, 1] + t * (P[i, j + 1, 1] - P[i, j, 1]))
            xs.append(e_s * (P[i, j, 0] + t * (P[i, j + 1, 0] - P[i, j, 0]) - sc.c[0]))
        ys, xs = np.asarray(ys), np.asarray(xs)
        # behind the ear including its groove: past its back edge, or in its back half medial of every ear vertex
        # within 4 mm (y, z) of the crossing - between the ear's back and the skull - or where the ear is not
        grv = None
        if len(ys):
            ev_i = np.nonzero(ev)[0]
            et = cKDTree(V[ev_i][:, 1:3])
            elat = e_s * (V[ev_i, 0] - sc.c[0])
            ok = ys > e["y_back"]
            for q in np.nonzero(~ok & (ys > 0.5 * (e["y_front"] + e["y_back"])))[0]:
                nb = et.query_ball_point([ys[q], zm], 0.004)
                ok[q] = (not nb) or bool(xs[q] < elat[nb].min())
            grv = round(float(ok.mean()), 3)
        out[str(e_s)] = {"rim_vertices": int(len(rim)), "rim_cover": cov, "side_guides": int(side.sum()),
                         "crossing_mid_height": int(len(ys)),
                         "behind_share": round(float((ys > e["y_back"]).mean()), 3) if len(ys) else None,
                         "groove_or_behind_share": grv}
    return out


def lag_stick(st):
    """the friction anchors kept across a lag (Coulomb static friction): a sample that stayed within the friction's
    smoothing length eps of its anchor (in the contact's tangent plane) has stuck and keeps it; one that slid further
    gets an anchor trailing eps behind it along its slip, so it keeps ~0.7 mu N against further sliding. settle.lag
    alone re-anchors every sample where it now is, which lets a held (brushed) layer creep by up to ~eps at each lag.
    New contacts are anchored where they are. -> (stuck, slid)"""
    had = st.anc_N > 0
    old = st.anc.copy()
    st.lag()
    idx = st.sidx
    h = had[idx] & (st.anc_N[idx] > 0)
    dl = st.anc[idx] - old[idx]
    nf = st.anc_n[idx]
    tdl = dl - (dl * nf).sum(1, keepdims=True) * nf
    sl = np.sqrt((tdl * tdl).sum(1))
    stuck = h & (sl < st.eps)
    slid = h & ~stuck
    st.anc[idx[stuck]] = old[idx[stuck]]
    st.anc[idx[slid]] -= st.eps * tdl[slid] / sl[slid, None]
    st._anchors()
    return int(stuck.sum()), int(slid.sum())


def front_lift(rods, body, front, arcs_mm=(5.0, 10.0, 20.0), st=None):
    """the front hairline's lift, per stage: for the guides `front` (acts.front_g), the height of the hair above the
    skin (the body's distance, mm) at arcs_mm along the guide, the first segment's elevation off the skin's tangent
    plane (deg), and the same heights for the REST shape alone (the turning rho integrated from the follicle with no
    weight or contact, above the root's tangent plane) - what the set would stand at if nothing loaded it. With st,
    the share of the front guides' samples anchored by friction. -> dict"""
    g = np.flatnonzero(front)
    if not len(g):
        return None
    P = rods.positions()[g]
    A = rods.node_arcs()[g]
    out = {"guides": int(len(g))}
    pc = lambda v: np.round(np.percentile(v, [10, 50, 90]), 2).tolist() if len(v) else None   # noqa: E731
    # rest shape: d_j = unit(d_{j-1} + rho_j), d_{-1} = d0
    K = rods.K
    Dr = np.empty((len(g), K, 3))
    prev = rods.d0[g].copy()
    for j in range(K):
        prev = _unit(prev + rods.rho[g, j])
        Dr[:, j] = prev
    Pr = np.concatenate([rods.X[g][:, None], rods.X[g][:, None] + np.cumsum(rods.l[g][..., None] * Dr, 1)], 1)
    Ng = rods.N[g]
    for a in arcs_mm:
        s = a / 1000.0
        ok = A[:, -1] >= s
        Q = np.array([[np.interp(s, A[i], P[i, :, c]) for c in range(3)] for i in np.flatnonzero(ok)])
        Qr = np.array([[np.interp(s, A[i], Pr[i, :, c]) for c in range(3)] for i in np.flatnonzero(ok)])
        h = 1000.0 * body.sample(Q, grad=True)[0] if len(Q) else np.zeros(0)
        hr = 1000.0 * ((Qr - rods.X[g][ok]) * Ng[ok]).sum(1) if len(Qr) else np.zeros(0)
        out["height_mm_at_%gmm_p10_50_90" % a] = pc(h)
        out["rest_height_mm_at_%gmm_p10_50_90" % a] = pc(hr)
    out["first_segment_elevation_deg_p10_50_90"] = pc(np.degrees(np.arcsin(np.clip((rods.D[g, 0] * Ng).sum(1), -1, 1))))
    out["follicle_elevation_deg_p10_50_90"] = pc(np.degrees(np.arcsin(np.clip((rods.d0[g] * Ng).sum(1), -1, 1))))
    if st is not None:
        fs = front[st.sq // st.Ka]                                   # per active sample, as settle_look's fr
        if fs is not None and fs.any():
            out["front_samples_anchored_share"] = round(float((st.anc_N[st.sidx][fs] > 0).mean()), 3)
    return out


def settle_look(rods, body, acts, cfg=None, pressure=True, friction=True, splat=None, log=print, probe=None):
    """grow with the acts on, the water set, release (acts off but the pins that stay; the tuck lets go after the
    first release solve), tousle strokes, then the final gated settle. probe(name, state), if given, is called at
    'grown', 'held' (the end of the hold, the set taken), 'release0' and 'final' (reports only). -> settle.State"""
    probe = probe or (lambda name, s: None)
    st = SE.grow(rods, body, acts=acts, cfg=cfg, pressure=pressure, friction=friction, splat=splat, log=log)
    probe("grown", st)
    fr = acts.front_g[st.sq // st.Ka]                    # samples of the front-hairline guides
    sidx_g, Y_g = st.sidx.copy(), st.view().Y.copy()
    fri = {"anchored_after_growth": int((st.anc_N[st.sidx] > 0).sum()), "samples": int(len(st.sidx)), "lags": []}
    nset = dry_set(st, acts)
    log(" dry: %d joints set (set %.2f, lift %.0f deg over %.0f mm)" % (
        nset, acts.v["dry"]["set"], acts.v["dry"]["lift_deg"], acts.v["dry"]["over_mm"]))
    dry_hold(st, acts, log)
    probe("held", st)
    acts.release()
    st.stage_index += 1
    f = st.f
    nrel = max(1, int(f["release_solves"]))
    if f["evals_release"] > 0:
        per = max(2, int(round(f["evals_release"] / nrel)))
        for k in range(nrel):
            st.solve(per, label="release %d" % k)
            if k == 0:
                fri["front_slip_release0_mm_p50_p95"] = _slip(st, sidx_g, Y_g, fr)
                probe("release0", st)
            stuck, slid = lag_stick(st)
            fri["lags"].append({"stuck": stuck, "slid": slid})
            log("  friction lag: %d anchors kept (stuck), %d trailing (slid)" % (stuck, slid))
    for k in range(int(acts.v["tousle"]["strokes"])):
        acts.tousle_stroke(k)
        acts.lag(st.view())
        st.solve(int(acts.v["tousle"]["evals"]), label="tousle %d" % k)
        lag_stick(st)
    acts.tousle_off()
    keep = f["evals_release"]
    f["evals_release"] = 0
    try:
        st = SE.release(st, acts=acts)
    finally:
        f["evals_release"] = keep
    fri["front_slip_final_mm_p50_p95"] = _slip(st, sidx_g, Y_g, fr)
    acts.counts["friction"] = fri
    probe("final", st)
    return st


# ---------------------------------------------------------------- styling in sections (DESIGN_sections)
# A stylist sections the head and works from the bottom up: the nape is "the first section you will dry" (heymane.com),
# the front "always the last section to blow dry" (Monroe Brush, behindthechair.com); the four- and five-section
# partings and the horseshoe (hairfinder.com; hair_styling.md T1). Each section is brushed, held while it dries and
# cools, let go and dropped onto the ones under it, so the last-laid one lies on top by its own weight, the hair-hair
# pressure and the friction - never by a rule. Off (sections.on false) nothing changes.
SECTIONS_DEFAULTS = {
    "on": False,                                            # style in sections
    "order": ["nape", "back", "sides", "crown", "front"],   # the order they are worked in (nape first, front last)
    "below": "live",                                        # laid sections stay variables ('drop' not built)
    "drop_evals": 45,                                       # evaluations after each section is let go (2 solves)
    "path_on_layer": False,                                 # not built: the brush path riding on the laid layer
    "stagger": 0,                                           # not built: subsections staged inside a section
    "children_within": True,                                # children blend only their own section's guides
}
SECTION_NAMES = ("nape", "back", "sides", "crown", "front")


def sections_settings(st):
    """the sections block: SECTIONS_DEFAULTS <- style['sections'] -> (values, warnings). The keys not built yet are
    reported, not silently taken."""
    blk = (st or {}).get("sections") or {}
    out = copy.deepcopy(SECTIONS_DEFAULTS)
    warn = []
    for k, v in blk.items():
        if k not in out:
            warn.append("unknown key sections.%s" % k)
        else:
            out[k] = copy.deepcopy(v)
    if out["below"] != "live":
        warn.append("sections.below '%s' is not built: 'live' used" % out["below"])
        out["below"] = "live"
    for k in ("path_on_layer", "stagger"):
        if out[k]:
            warn.append("sections.%s is not built: ignored" % k)
    bad = [s for s in out["order"] if s not in SECTION_NAMES]
    if bad or len(set(out["order"])) != len(out["order"]):
        raise ValueError("sections.order: each of %s once, got %s" % (SECTION_NAMES, out["order"]))
    return out, warn


def section_of(scalp, X, dry, order=SECTION_NAMES):
    """each root's section from anatomy (DESIGN_sections 3.1), +y back: th the root's polar angle from the vertex
    (scalp.sph, the frame _blow_shape uses), dry.ridge_deg the parietal ridge, dry.occiput_deg the occiput, the ears'
    back edge y_back and the coronal plane through the ears y_c (both ears' mean).
      nape   y > y_back and th >= occiput      back  y > y_back and ridge <= th < occiput
      sides  y <= y_back and th >= ridge       crown th < ridge and y > y_c        front th < ridge and y <= y_c
    -> (n,) int, the section's place in `order` (the order it is worked in)"""
    _, th, _ = scalp.sph(X)
    ears = list(scalp.ears.values())
    y_back = float(np.mean([e["y_back"] for e in ears]))
    y_c = float(np.mean([0.5 * (e["y_front"] + e["y_back"]) for e in ears]))
    ridge, occ = float(dry["ridge_deg"]), float(dry["occiput_deg"])
    y = X[:, 1]
    name = np.where(th < ridge, np.where(y > y_c, "crown", "front"),
                    np.where(y > y_back, np.where(th >= occ, "nape", "back"), "sides"))
    idx = {s: i for i, s in enumerate(order)}
    return np.array([idx[s] for s in name], np.int64)


def section_hold(state, acts, on, log=print):
    """one section's hold and set (DESIGN_sections 3.2 c): the water set armed on that section (dry_set: with dry.span
    'held' it waits for the end of the hold), then dry_hold - the section stays in the brush while it dries and cools,
    and its set is taken from the shape it is held in, on its own joints only (acts.on_g)"""
    nset = dry_set(state, acts)
    dry_hold(state, acts, log)
    log(" section: %d guides held and set (%d joints)" % (int(on.sum()), nset))
    return nset


def _slip(st, sidx_g, Y_g, fr):
    """p50 / p95 movement (mm) of the front-hairline guides' samples since the end of growth (the brushed state)"""
    if len(st.sidx) != len(sidx_g) or not np.array_equal(st.sidx, sidx_g) or not fr.any():
        return None
    d = np.sqrt(((st.view().Y - Y_g) ** 2).sum(1))[fr] * 1000
    return [round(float(np.percentile(d, 50)), 2), round(float(np.percentile(d, 95)), 2)]


# ---------------------------------------------------------------- measures (reports, never targets)
def polylines(rods, step_mm=1.0):
    """the guides' polylines resampled every step_mm: points (m), guide index, members beyond the point"""
    P = rods.positions()
    arcs = rods.node_arcs()
    pts, gi, wn = [], [], []
    n = np.concatenate([rods.n, np.zeros((rods.G, 1))], 1)
    for i in range(rods.G):
        L = float(arcs[i, -1])
        if L <= 0:
            continue
        s = np.arange(0.0, L, step_mm / 1000.0)
        q = np.stack([np.interp(s, arcs[i], P[i, :, c]) for c in range(3)], 1)
        pts.append(q)
        gi.append(np.full(len(s), i))
        wn.append(np.interp(s, arcs[i], n[i]))
    return np.concatenate(pts), np.concatenate(gi), np.concatenate(wn)


def face_cover(acts, Q, w=None, near_mm=30.0):
    """share of the hair points (weighted by w) over the face skin: nearest head vertex on the face (corridor), within
    near_mm of the skin. Counted, nothing deleted."""
    sd = acts.body.sample(Q)
    _, vi = acts.tree.query(Q)
    m = acts.face_v[vi] & (sd < near_mm / 1000.0)
    w = np.ones(len(Q)) if w is None else w
    return float(w[m].sum() / max(w.sum(), 1e-300)), int(m.sum())


def face_split(acts, Q, gi, w=None, near_mm=(30.0, 10.0), front_cos=0.5):
    """the face cover split two ways (counted, nothing deleted): by depth (within each near_mm of the skin) and by
    place - 'any' is face_cover's measure (nearest head vertex on the face skin), 'front' only the points in FRONT of
    the face (the skin's outward direction at the point, the body's distance gradient, within 60 deg of straight
    forward), so hair hanging beside the cheeks is not counted as over the face; each also by root: the front-hairline
    guides (|phi| < 40, within 15 mm of the hairline) against the rest. Shares of all hair points (weighted by w)."""
    sd, g = acts.body.sample(Q, grad=True)
    _, vi = acts.tree.query(Q)
    onf = acts.face_v[vi]
    fwd = -_unit(g)[:, 1] > front_cos
    w = np.ones(len(Q)) if w is None else w
    W = max(float(w.sum()), 1e-300)
    fr = acts.front_g[gi]
    out = {}
    for nm in near_mm:
        m = onf & (sd < nm / 1000.0)
        o = {}
        for key, mm in (("any", m), ("front", m & fwd)):
            o[key] = round(float(w[mm].sum() / W), 5)
            o[key + "_front_hairline_roots"] = round(float(w[mm & fr].sum() / W), 5)
            o[key + "_other_roots"] = round(float(w[mm & ~fr].sum() / W), 5)
        out["%gmm" % nm] = o
    return out


def crown_layer(body, Q, centre, crown_deg=40.0):
    """settle.layers' crown measure on points Q (m): per 5 mm cell of the foot points (closest skin point) the 95th
    percentile height, p10 / p50 / p90 over the cells with >= 3 points; and the points' p90 (points within 60 mm)"""
    sd, g = body.sample(Q, grad=True)
    foot = Q - sd[:, None] * _unit(g)
    v = foot - centre
    th = np.degrees(np.arccos(np.clip(v[:, 2] / np.maximum(np.sqrt((v * v).sum(1)), 1e-12), -1, 1)))
    m = (th < crown_deg) & (sd < 0.06) & (sd > -0.01)
    if m.sum() < 10:
        return None
    cell = np.floor(foot[m] / 0.005).astype(np.int64)
    _, inv = np.unique(cell, axis=0, return_inverse=True)
    h = sd[m] * 1000
    inv = inv.ravel()
    o = np.argsort(inv, kind="stable")
    cuts = np.nonzero(np.diff(inv[o]))[0] + 1
    tops = [np.percentile(x, 95) for x in np.split(h[o], cuts) if len(x) >= 3]
    return {"layer_top_cells_p10_p50_p90_mm": np.round(np.percentile(tops, [10, 50, 90]), 2).tolist(),
            "cells": len(tops), "points_p90_mm": round(float(np.percentile(h, 90)), 2)}


def ear_top_cover(scalp, Q, radius_mm=(4.0, 8.0), top_mm=12.0):
    """per ear and radius: the share of the ear's upper vertices (within top_mm of its top) with a hair point within
    radius_mm"""
    tree = cKDTree(Q)
    out = {}
    for e_s, e in scalp.ears.items():
        V = scalp.V[scalp.ear_mask & (np.sign(scalp.V[:, 0] - scalp.c[0]) == e_s)]
        V = V[V[:, 2] > e["z_top"] - top_mm / 1000.0]
        d, _ = tree.query(V)
        out["left" if e_s > 0 else "right"] = {"%gmm" % r: round(float((d < r / 1000.0).mean()), 3) for r in radius_mm}
        out["left" if e_s > 0 else "right"]["n_vertices"] = int(len(V))
    return out


def band_layer(scalp, body, Q, din_mm=(5.0, 40.0), phi_deg=30.0, cell_mm=5.0, near_mm=60.0):
    """the front band (din_mm in from the front hairline, |phi| < phi_deg): hair height over the skin. Per 5 mm cell
    of the skin (by the nearest head vertex) the layer top (highest point); p50 / p95 over the cells, and the points'
    p95"""
    sd = body.sample(Q)
    tree = cKDTree(scalp.V)
    _, vi = tree.query(Q)
    Vn = scalp.V[vi]
    din = scalp.d_inside_mm(Vn)
    ph, _, _ = scalp.sph(Vn)
    m = (din > din_mm[0]) & (din < din_mm[1]) & (np.abs(ph) < phi_deg) & (sd < near_mm / 1000.0)
    if m.sum() < 3:
        return None
    cell = np.floor(Vn[m] / (cell_mm / 1000.0)).astype(np.int64)
    _, inv = np.unique(cell, axis=0, return_inverse=True)
    top = np.full(inv.max() + 1, -np.inf)
    np.maximum.at(top, inv.ravel(), sd[m] * 1000)
    return {"cells": int(len(top)), "points": int(m.sum()),
            "layer_top_cells_p10_p50_p95_mm": [round(float(np.percentile(top, q)), 2) for q in (10, 50, 95)],
            "points_sd_p50_p95_mm": [round(float(np.percentile(sd[m] * 1000, q)), 2) for q in (50, 95)]}
