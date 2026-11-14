# The methods we changed, and why

Field Studio is meant to be released as open source. A method covered by someone else's patent would put that release
at risk, so while we build we read the patents we find near each mechanism. When a method we planned or built turned
out to sit near one, we did not argue our way past it: we built a different mechanism, measured it, and wrote down how
it went. This page is that record, case by case: the patent as flagged in our review, what we had, what we did
instead, how the new method measured, and whether it proved better, equal or worse. Several are still open, and we say
so.

[PATENTS.md](PATENTS.md) describes how each mechanism in the released code works, so the differences are on record.
This page tells the story behind it.

**What this page is not.** These are engineering readings of public patent records by non-lawyers. They are not legal
advice and not a freedom-to-operate opinion, and no patent attorney has been consulted. We do not say here that any
code of ours falls inside or outside any claim; we describe what our mechanisms do and how they measured. Patent
statuses are as shown on public patent pages between 2026-11-11 and 2026-11-14. Our searches were of United States
records, and they were not exhaustive: finding nothing is not the same as nothing existing. If you know of a patent
we should read, please open an issue.

**How a flag is handled** (the project's standard since 2026-11-12). A search hit is a screening flag, not a finding.
A redesign is justified only with the particular claim, its jurisdiction and status, every required limitation
compared with the actual code, the strongest counterarguments, and the limitation the change removes with its cost.
An incomplete comparison is an open question, neither a prohibition nor a finding in our favour, and working
directions are not dropped just because a related patent exists. One caution shaped every case
below: a method that performs better is not, by that fact, a different method. We compare mechanisms with claims, and
we measure results separately.

**Verdicts.** Each case is judged against what we had, on the cases named, on one machine (Windows 11, 32 GB of RAM,
one 8 GB NVIDIA RTX 5050 shared with other work):

| Verdict | Meaning |
|---|---|
| Better | the new method measured better than what we had, on the named cases |
| Equal | the same result within the stated tolerance, at a similar cost |
| Worse | the new method measured worse, or we lost a capability for now |
| Open | not yet measured like for like, or the measurement is not finished |

---

## The cases in one table

| # | Area | Flagged in our review | What we had | What we do instead | Verdict |
|---|---|---|---|---|---|
| 1 | Skin weights for the rig | Autodesk US 9,836,879 B2 | weights from shortest paths through the solid's voxels | weights by heat diffusion through the solid | **Better** |
| 2 | Hair motion | US 7,616,204 (position-based dynamics) | an integrator taking velocity from corrected positions | each strand's own modes, moved in closed form, with the hard stop inside | **Better** (cost); its own reading is owed |
| 3 | Light gather | NVIDIA US 11,790,594 B2 | a gather that read an outgoing-light cache for far hits | one rule at every hit; sky visibility kept per surface cell | **Equal** (gather), **Better** (sky cells) |
| 4 | Contact | NVIDIA US 12,216,969 B2 | a barrier-based Newton contact solver | a constraint form with explicit multipliers and a hard-stop contact law | **Better** result, **Worse** iterations: **Open** |
| 5 | Pencil | MERL US 8,265,909 B2, US 8,010,328 B2 | a line built as one continuous swept field | discrete path samples, each evaluated at its own placement relative to the stroke-start surface; nothing searched along the path | **Better** accuracy; **Open** speed |
| 6 | Connect along a curve | MERL US 8,265,909 B2 | a profile carried along a curved path | held out; straight connectors only, for now | **Worse** (a capability lost for now) |
| 7 | Elastic Grab | Pixar US 10,586,401 B2 | a Grab driven by the form's linear-elastic response | moves as smooth flows; a form-aware Grab without elasticity planned | **Open** |
| 8 | Stored states for motion | Autodesk US 10,489,533 B2; Tencent US 11,823,315 B2 | a table of stored solves as the live solve's start | our own start; states computed fresh from the present (planned) | **Worse** for now |
| 9 | How the field is stored | Movidius US 11,680,803 B2, US 11,222,459 B2 (and US 12,085,392) | a band store with inside / outside tags per tile and brick | that store held out; a tile index of addresses; a hash store planned | **Open** |
| 10 | Merge's blend | Ansys US 10,083,264 B1 | a smooth union at one fixed width, offered in the window | an exact join with a blend whose width is set from both fields at the contact | **Open** (not compared head to head) |
| 11 | X-ray and the outline through covers | QuantifiCare US 11,094,139 B2 | (planned) a see-through view | real front-to-back compositing; an outline from object-ID edges | built this way from the start; nothing to compare |
| 12 | Grain-scale shadow | Adobe US 8,379,021 B1 | a fixed-step height test along the light | kept as built, the difference written down | kept |

Cases 1 to 9 are methods we changed after building or prototyping them. Cases 10 and 11 were shaped by the review
before or while they were built. Case 12 was read after it was built and kept. A shorter table at the end lists the
patents that steered designs not yet built.

---

## Methods we changed after building them

### 1. Skin weights: from shortest paths to heat diffusion (Autodesk US 9,836,879 B2, flagged 2026-11-11)

**What we had.** To make flesh follow bones, our first posing prototype computed each bone's influence from shortest
paths through the solid's interior voxels, a geodesic weighting.

**Why it changed.** The review flagged a patent on binding skin weights by shortest paths through a voxelized model.
Decided the same day: a patented method is not used, because it is a legal risk; it may be studied and learned from,
and then improved on.

**What we do instead.** Each bone's weight is the solution of one linear diffusion (heat) equation solved through the
solid itself, the bone-heat idea of Baran and Popovic (2007) moved from a surface into the volume. Heat cannot cross a
gap between fingers, because the solid's boundary insulates it. No path is traced and no shortest distance is computed.

**How it measured** (2026-11-11, CPU, a test arm bent to 90 degrees, at the same share of exactly rigid tissue):

| Weights | Rigid share | Folded tissue at the inner elbow | Volume change |
|---|---|---|---|
| Heat diffusion | 46 % | 0.018 cm3 | -0.18 % |
| Shortest paths (what we had) | 47 % | 6.7 cm3 | -1.8 % |

About 370 times less folding and ten times less volume lost. The solve took 140-160 s on the CPU for 348,000 voxels;
a whole body at 2 mm needs an iterative solver on the card, not yet built.

**Verdict: Better.** The rig design adopts influence diffused through the field's interior (never across a gap), so
nobody paints weights. The rig itself is designed, not built.

### 2. Hair motion: from a position-correcting integrator to each strand's own modes (US 7,616,204, flagged 2026-11-13)

**What we had.** A hair motion prototype that, each step, moved strand points, corrected their positions for contact
and constraints, and then took the velocity from the corrected positions.

**Why it changed.** A review of the hard-stop contact law flagged the integrator around it: its last
step, velocity read back from corrected positions, sits near the broad claim of a position-based dynamics patent that
runs to 2027-08-14.

**What we do instead.** The approach was changed to hair that responds to the forces on it now, rather than a tracked
trajectory, combined with the hard-stop contact law (both among the author's concepts). As built for research: each strand's lowest modes are computed
once from its length, weight, tension and stiffness (the lowest near 1.1-1.4 Hz, like a hanging chain); each frame,
every mode is advanced by an exact formula under the force acting now (wind drag projected on the mode, with the
fibre's own cited damping); where a strand meets skin, small contact pushes act on the modes, with skin friction so
hair can rest on the body; the finer child hairs keep a per-point stop against the body's own field. Modal dynamics
of this kind is old published work (Pentland and Williams 1989; James and Pai 2002).

**How it measured** (2026-11-13, CPU, loose locks under a fan on the human test model's head, against the prototype's
full strand simulation):

| | Tips median / p90 / max | Cost a frame | Child points inside the body |
|---|---|---|---|
| Strand modes with the hard stop | 3.7 / 10.5 / 20.0 mm | 16-20 ms | 0 |
| Full strand simulation | 3.7 / 10.9 / 20.6 mm | 256 ms | - |

About fifteen times cheaper with the motion statistics close (not identical). On the whole head, later the same day:
about ten times less wall-clock time and twenty times less thread time, again close, not identical. Left: a small tip
jitter in the tail (p99 0.41 mm against 0.24 mm), and the modes describe the gentle, nearly linear range; a flip or
hair piling on itself leaves it.

**Verdict: Better on cost, about equal on the result.** Still owed: a reading of the new modal step against the same
patent before it is chosen (it carries rates of the modal amplitudes, and whether those count is a question we have not
settled). Hair motion is research, not in the released window.

### 3. Light: one rule at every hit, and sky visibility per surface cell (NVIDIA US 11,790,594 B2, flagged 2026-11-11)

**What we had.** The look-development light gathered one bounce in two ways: hits nearer than 10 cm were lit exactly,
farther hits read a cache of light leaving surfaces. The window's sky lighting traced its own rays per pixel each
frame, with no cache.

**Why it changed.** The review found a patent on world-space irradiance caches, including reading cached outgoing
light for a hit chosen by its distance. A second reading found that our first had missed that part of the patent; the
patent was re-read and the code changed.

**What we do instead.**
- **The gather** lights every hit by one rule, never chosen by distance and never reading an outgoing-light cache
  (irradiance caching as published by Ward in 1988).
- **The window's sky** keeps, for each small surface cell (keyed by position, level and normal direction), how open
  that cell is to the sky: visibility, not light leaving a surface. A cell gets rays when the current image shows it and
  it is not yet full; no cell is ranked against another; a cell's size follows the pixel footprint; each pixel reads
  its own cell, or its own estimate while the cell fills.

**How it measured.**
- The gather (2026-11-12): one bounce matches the all-exact reference with a bias of at most 0.12 % in the human test
  model's skin folds; a still pass costs 0 to 3 ms more. The second bounce is not yet on the same rule.
- The sky cells (2026-11-13, card, 1080p, four views of the human test model, a camera-move frame):

| View | Before (ms) | After (ms) |
|---|---|---|
| Head | 39.9 | 29.2 |
| Eye | 40.0 | 30.0 |
| Pore close-up | 32.4 | 20.5 |
| Whole body | 8.3 | 7.9 |

  The converged still is the same look as before, a little closer to a 256-ray reference; the moving frame is far
  smoother. Cost: a 67 MB table and 0.24-0.32 ms to invalidate cells on an edit frame. Two earlier designs of the cell
  read were rejected on the pictures (grid-aligned noise).

**Verdict: Equal for the gather (same result, slightly slower); Better for the sky cells.** Both are in the window.

### 4. Contact: from a barrier solver to a constraint form and a hard-stop law (NVIDIA US 12,216,969 B2, flagged 2026-11-12)

**What we had.** Our first-generation contact solver: a barrier-based Newton method in the family of incremental
potential contact, used for skin-on-skin contact in posing experiments.

**Why it changed.** The review flagged a patent on contact simulation that solves an energy minimisation in a primal,
force-based formulation with a preconditioned descent. Decided that morning: build around it with custom methods,
on the expectation that better and faster ones can be found that way. The pending continuation of that patent, US 2025/0131161 A1, is watched.

**What we do instead.**
- **A constraint form.** Unknowns are velocities plus an explicit contact impulse and a two-component friction impulse
  for each contact row. Non-penetration and Coulomb friction are complementarity conditions (Fischer-Burmeister and
  Alart-Curnier forms), solved by a semismooth Newton method on the whole system with a direct sparse factorisation, a
  smoothing continuation, and a line search on the residual's norm. No energy is evaluated or minimised. The earlier
  solver stays only as the comparison baseline; it is not in the window or the released code.
- **The hard-stop law** (one of the author's concepts) on top: measurement only until surfaces touch, then whatever force touching needs,
  never a push at a distance and never a pull. Built as one row per contact pair, a projected tangent, and a release
  step (a row that would pull lets go; a row that would cross comes back).

**How it measured** (2026-11-13, CPU, the human test model's elbow crease closing, against a barrier law of the same
family as the old solver):

| | Hard-stop law | Barrier law |
|---|---|---|
| Skin crossings in the drag | 0 | 0 |
| Force on surfaces apart | exactly 0 beyond 1 um | pushes inside a 75-100 um film |
| Rows pulling at frame end | 0 | - |
| Iterations, with our own start | 1.4 to 3.4 times the barrier's | 1x |
| Seconds, one like-for-like run | 151.8 s | 330.2 s |

The deep crease frames still reach the iteration cap with our own start; the few rows crossed there are put back by a
counted clamp (at most 26 um deep, stated beside the frame time). Two other options were measured and dropped: a
penalty law crossed in every drag frame (122 to 3,216 rows) and pushed while skin was apart; an augmented Lagrangian on
the card took 574 iterations against 317 and left force on rows already apart. All the elbow runs used an elbow
skeleton later found to be wrong (a square hinge), so they will be re-run on a corrected one.

**Verdict: Better in result (exact contact, no film, no pull), Worse in iterations, Open overall.** The work is paused
by decision (tools first); contact is not in the window.

### 5. Pencil: from one swept line to discrete path samples (MERL US 8,265,909 B2 and US 8,010,328 B2, flagged 2026-11-12, re-read against the old line tool 2026-11-14)

**What we had.** A line tool that built each stroke as one continuous swept field: the tip's shape carried along the
drawn path, joined piece by piece. It also had faults of its own, measured on 2026-11-12:
- on the human test model's upper lip, 1.2 mm from the mouth's slit, it cut across the slit into the lower lip by up
  to 0.226 mm at 811 points;
- in the window it drew nothing while drawing, then ran three to four minutes of card work after each release, and the
  application grew to a 15 GB working set.

**Why it changed.** The review flagged a patent on the distance field of a shape swept along a path, found by placing
the shape at its best position along the path for each sample point. Separately, the design requires interactive
geometry feedback, the mark changing on screen beneath the tip as the cursor moves, which the old tool did not give.

**What we do instead.** Pencil represents its working tip as an implicit geometric function: a pointed conical, a
spherical, or a square, flat-ended profile. The stroke engine samples the cursor path and, for each path sample,
establishes a local tip position and orientation relative to the stroke-start surface: the cursor's surface point is
projected, in a bounded number of steps along the local surface facing, onto the surface as it was at the press, and
the local tip frame is built from that facing and the path direction. It then evaluates the tip function at affected
field nodes. Carve mode combines the object field with the negated tip field using a maximum operation; ridge mode
combines the two fields using a minimum operation; an optional smooth combination controls the junction with the
surrounding surface. The depth is limited by a configured maximum, the tip size and, in carve mode, half the part's
thickness under the point. The stroke engine therefore evaluates discrete path samples, each at its own placement
relative to the stroke-start surface, and nothing searches along the path for a nearest placement. This is a sampled
numerical implementation of an interactive line tool, not an exact continuous geometric sweep; the specified depth is
a target relative to the pass reference, and finite field resolution and edge blending also affect the realized mark.
Paint on the colour layer is a planned extension, not a shipped mode.

**How it measured** (2026-11-14):
- CPU reference: a 0.5 mm carve along a 15 mm arc reads 0.498-0.500 mm with a round tip, 0.500 mm square, 0.465-0.493
  mm with a point; no deepening where the line crosses itself within one pass; a 0.4 mm ridge 0.398-0.400 mm; the face across a 0.5 mm
  slit untouched.
- On the card: round 0.498-0.502 mm, square 0.500-0.504 mm, point 0.472-0.496 mm; the twelve other tools' results
  unchanged.
- Not yet good enough to be the window's line tool: on a surface about 45 degrees to the view, the tip placements
  wander from -0.6 to +1.4 mm, because each starts from where the cursor ray meets the groove just cut; and one plan and
  launch per path sample runs at 1.5-6 frames a second. Both fixes are designed (surface-relative tip placement taken
  from the stroke-start surface; batched path-sample processing).

**Verdict: Better on accuracy and locality; Open on speed and feel.** A note for the planned progressive on-screen
update: it must draw the pending path samples exactly as they will be written, not compute a nearest placement along
the path for each pixel.

### 6. Connect along a curved path: held out (MERL US 8,265,909 B2, flagged 2026-11-14)

**What we had.** Connect: pick two objects and a profile, and Field Studio builds a connector
between them, giving three objects that can be merged or kept apart. It could follow an arc or a drawn curve, reading
the profile in a frame carried along the path at the nearest path point.

**Why it changed.** The same review read the curved branch against the same patent: for a round profile, the nearest
path point is the shape's best placement.

**What we do instead.** The release ships straight connectors only (a circle-to-square pillar between two cubes: ends
flush within 0.004 mm, no overlap). Two redesign routes are written down and not built: a straight connector carried
along the path by the warp engine's flow (no placement solved per point; estimated one to two days), or discrete
stations of the profile, each evaluated at its own placement as Pencil evaluates its path samples (about a day, with
scallops at coarse spacing).

**Verdict: Worse for now** (a capability held out until a redesign is built).

### 7. Elastic Grab: from the form's elastic response to flows (Pixar US 10,586,401 B2, flagged 2026-11-11, re-read 2026-11-12)

**What we had.** A prototype "Elastic Grab": the pull moved the form by its own linear-elastic response, solved
numerically on a coarse lattice inside the form with a smooth load and a set Poisson ratio. Measured on 2026-11-11: a
pull on the human test model's cheek moved the back of the head 0.30 mm and the far cheek 0.34 mm, and added 37.6 cm3
of volume, against 74 cm3 for the plain Grab of the time.

**Why it changed.** A first reading compared the prototype only with the patent's narrower claims (closed-form
Kelvinlet brushes) and let it pass. A fuller reading showed that it was incomplete, because the broad claim
recites regularized solutions of linear elasticity with a specified Poisson ratio. The prototype was never offered in
the window and never merged.

**What we do instead.** Grab, Shape and the other moves are smooth flows of velocity fields, integrated forward, with
the field read back through the inverse flow and baked when the stroke ends; a curl (divergence-free) flow keeps
volume. No elasticity equation, Poisson ratio or Kelvinlet is used. Decided for the elastic feel (2026-11-12): a
Form-aware Grab, whose reach follows the form's own interior from the press point, with no material constants; an
elastic Grab is not pursued.

**How it measured.** The flows, on a sphere: the pressed point lands exactly; forward and inverse agree to 1.3e-8 m; a
pull on a sphere moves its pole 60.0 mm and its side and back 0.0 mm; the volume-keeping flow holds the Jacobian
determinant at 0.990-1.023. The form-aware Grab is not built, so the elastic prototype's head case has no like-for-like
comparison yet.

**Verdict: Open.**

### 8. Stored states for motion (Autodesk US 10,489,533 B2 and Tencent US 11,823,315 B2, flagged 2026-11-13)

**What we had.** A concept of solving movement ahead of time (one of the author's concepts), measured as a table: the elbow solved on the
live tissue law every 2.5 degrees (17 states, 3 MB), and a drag between them started from the two nearest states,
blended. At 2.5 degrees a frame it took 99 iterations in all against 351 for our usual start (3.5 times fewer), every
frame converged, no crossings, no clamp; at 5 degrees a frame, where our usual start broke (124 to 656 skin pairs
crossed a frame), the stored start held.

**Why it changed.** The review flagged two patents: one on warm-starting a finite-element solver from an estimate built
from stored data, and one on a weighted sum of stored poses chosen by their distance. The concept was then set out
more precisely: what it stores are calculations, not poses; and, after the first comparison, states are not stored
persistently: no caches or tables of historical states, each solve starting fresh with limited scope.

**What we do instead.**
- The table is not in any released code; whether it ever ships is an open decision.
- One stored state applied as it is, with no blending and no solve, was measured and failed at 2.5-degree spacing: 1.25
  degrees from the nearest entry, 73 to 600 skin pairs crossed by 0.5-1.0 mm (the blend had crossed 8 to 36 pairs by
  at most 35 um).
- The live solve starts from our own estimate (the previous live-solved frames, extrapolated and projected out of
  penetration), with no stored state in it: 1.4 to 3.4 times the barrier's iterations (case 4).
- Planned, not built: a lookahead, in which the states the body can reach next are deduced from its
  constraints, computed fresh from the present, and applied when their trigger fires; and a two-layer design, in which
  a few key points (the bones) carry the movement and creases, contact and local deformation are solved downstream.
  The first test of the bones as the only keys, at the elbow, measured No as built: the error is the bend's own
  response at the joint.

**Verdict: Worse for now.** The best-measured variant is held out, and the start we can use costs more iterations.

### 9. How the field is stored (Movidius US 11,680,803 B2 and US 11,222,459 B2, flagged 2026-11-12)

**What we had.** Two storage layouts:
- a band store built for full-body sculpting that tagged each tile and brick as inside or outside, with finer entries
  beneath (never used by the app);
- the card tracer's tile index, which skips tiles with nothing stored.

**Why it changed.** The review flagged patents on a volume held at several levels of detail, where an entry's bits
say whether each voxel is empty or at least partly occupied, with finer entries beneath, and rendering skips empty
voxels. A related patent (US 12,085,392) covers a second structure set to one of two values from the first's data.

**What we do instead.**
- The tagged band store is held out of the release.
- The tracer's index holds, per 32 mm tile and level, the address of that tile's table of bricks, or -1 where nothing
  is stored. Because only a narrow band around the surface is stored, a tile deep inside a solid and a tile of air both
  read -1: the entry says whether samples are stored, not whether space is empty or occupied. The released code's
  comments say so, and a guard comment forbids adding occupancy, inside / outside or sign tags beside the finer entries.
- The store we plan next keeps no per-tile entry of that kind: one hash maps (object, brick) to a slot in a pool of
  fixed pages; only band values change; undo copies a brick before its first change in a stroke; the sign of space it
  does not store is carried in a coarse distance value, not in tags. It was designed first for speed.

**How it measured.** A prototype of that store (2026-11-12, card, simplified brushes, no tracer):
- a brush dab cost 0.27-0.34 ms with one object loaded, and 0.21-0.30 ms with 66 objects and 25 times the bricks
  (the fastest of three runs each);
- a large Smooth writing 10,378 bricks on the human test model's head took 9-15 ms of store work and 4-7 ms to copy
  out on release (the app of the time released a similar stroke in 0.6-0.9 s, not like for like);
- undo exact on 9 of 9 strokes; at most 0.95 um from a dense reference.

The tracer reading this store, its keep-apart and re-distance passes, colour, and save and reopen are not built.

**Verdict: Open** (a promising prototype; weeks of work before it can replace today's store).

---

## Shaped before or while being built

### 10. Merge's blend (Ansys US 10,083,264 B1, flagged 2026-11-14)

**What we had.** The window's Merge box offered a smooth union at one fixed blend width, and the engine's general
smooth-union operator works the same way.

**What we do instead.** Merge was defined as an exact join with a simple setting for blend at contact.
Merge now refuses objects that do not touch and offers Group (with Ungroup); for touching objects it joins them exactly
and blends with a polynomial smooth minimum whose width is set per point from both objects' fields: full width across
the contact, tapering to nothing away from it. There is no rolling-ball fillet construction. The fixed-width box left
the window on 2026-11-14; the general smooth-union operator remains in the engine.

**A correction we made.** Our first write-up said the varying width was the difference. A further reading showed that
across the contact itself the width sits at its fixed maximum, so the varying width alone is not the distinction; the
differences we list in PATENTS.md are the ones that stand after that correction.

**How it measured** (2026-11-14, scripted on spheres): surfaces 10-100 mm from the contact move 0.0000 mm; the contact
is sealed (an internal crack where each piece kept its old face inside the solid, -0.2 mm where the body is -2 mm deep,
now reads -2.0 mm, with no line); blend at contact on two spheres went from 1.67 s to 0.23 s after an optimisation.

**Verdict: Open.** The new Merge was never measured head to head against the fixed-width union.

### 11. X-ray and the outline through covers (QuantifiCare US 11,094,139 B2, flagged 2026-11-14)

**What was planned.** Transparency that shows structural layers through objects. A common way to
draw a hidden object is a rim glow whose strength follows the angle between the surface normal and the view, in a
fixed colour over the image.

**What we built instead.** The X-ray slider and per-object opacity composite each object's own shaded surface front to
back along the ray, at its opacity, with no view-angle rim term. The selection outline comes from the edges of the
object-ID image, in one constant colour, dashed where something covers it, and reads no surface normal. Three guards
are written into the feature: the line comes only from ID or depth edges; its colour and alpha are constant along it;
the inner object is never shaded through by the outline (that stays in X-ray).

**Verdict:** built this way from the start, so there is nothing to compare against.

## Read after building, and kept

### 12. Grain-scale shadow (Adobe US 8,379,021 B1, flagged 2026-11-14)

A material's fine grain relief shadows itself with a fixed-step test: the relief is sampled at six fixed distances
toward the light, and the light ray's height above the relief at each sample is reduced to a visibility. The flagged
patent shadows a height field by finding the surface point at the greatest elevation angle toward the light. Our
routine never searches for that point. It was kept as built, with the difference written down.

Also kept as built, with how each works written in PATENTS.md: the capsule and torus primitives (standard closed-form
distances), straight Connect, and the brush's choice of which lattice cells to update (MERL US 8,265,909 B2).

The hair settle, flagged against Procter & Gamble US 10,460,054 B2 (fibre simulation from parameters that include a
shear modulus; first read 2026-11-14), carries no twist and so uses no shear modulus. It is kept as built; a reading
of that patent's full claims is still owed.

---

## Designs steered before any code

These designs are approved but not yet built, or were written with the patent in view from the first draft. Nothing
here is measured.

| Our mechanism | Flagged in our review | The route in our design |
|---|---|---|
| The tracer | MERL US 9,336,624 B2 | image-order tracing, each pixel's ray through the sparse structure; no per-cell rays with an entry and exit test |
| The lattice beside the shape tree | MERL US 9,122,270 B2 | the lattice holds samples only; the analytic tree is evaluated as a whole; finer detail subdivides on samples alone |
| Tracing hits and shadows in the lattice | NVIDIA US 11,823,319, US 12,254,556, US 11,887,245, US 12,175,588 | sphere tracing with bisection and secant refinement; one tricubic gradient; hits kept in object and world space |
| Swell / Settle to contact | Disney US 9,483,878 B2 | points stop on the other object's own field; no displayed offset shell |
| Long chains of warps | SolidWorks US 7,936,352 B2 | each move read back through its flow and baked at the stroke's end; no parametric surface |
| Settle under gravity (Place on surface) | US 7,616,204 and others | a static settle on the contact engine's route, not position correction |
| Pencil's paint layer | Adobe US 2026/0350905 A1 (pending) | paint kept as a 3-D field on the lattice, with no surface parameterisation |
| Hair shading | Pixar US 9,905,045, DreamWorks US 9,019,298 | the published Chiang et al. (2016) model; no fitted basis distributions; no Cauchy sampling |
| Skinning | Disney US 10,262,447 B2, US 9,613,456 B2; TU Delft US 10,061,871, US 9,928,663 | plain dual-quaternion or linear blending only as preview aids under the warp; no per-sample centres of rotation; no weights optimised over sampled poses |
| Fitting the skeleton | Adobe / Mixamo US 11,170,558 B2, US 10,748,325 B2 | the skeleton fitted on the body's field from landmarks and centre lines; no mesh proxy, server or learned key-point detector |
| Soft tissue over bone | Vital Mechanics US 11,675,935 | the tissue law is not tied to body points through an intermediate surface and a coupling energy |
| Learned motion | NVIDIA US 12,505,599 B2; Sony US 11,132,606 B2; Disney US 12,715,114 B2; two pending NVIDIA applications | a tracking-only imitator first: one clip, a tracking reward, no discriminator, no latent encoder, no skeleton split across clips; a fresh reading before anything more |

## Still open

- NVIDIA's pending continuation US 2025/0131161 A1, against the contact work's card path, when it is built and when the
  application grants or is abandoned.
- Disney US 9,135,738 B2 (one-point quadrature at each cell) against the tissue element's centre-point volume term,
  before that element ships on the card.
- US 7,616,204 against the modal hair step (case 2), before the hair motion is chosen.
- Procter & Gamble US 10,460,054 B2 against the hair settle: the full claims, read word for word.
- The two pending NVIDIA applications on learned motion, before adversarial rewards or multi-modal goals are added.
- The status of Movidius US 11,680,803 B2 after its first maintenance window closes (2027-06-20).

## What the changes cost, and what they gave

- Three changes measured better than what we had: heat-diffusion weights (370 times less folding), the modal hair
  response (ten to fifteen times cheaper, close in motion), and the sky visibility cells (camera-move frames 25-37 %
  faster on the head, eye and close-up views). One measured equal (the one-rule gather).
- Two cost us something for now: Connect along a curve is held out, and the stored-state start that cut iterations
  3.5 times is not released.
- The contact rebuild gave the exact hard-stop law (no film, no pull) at more iterations per frame; Pencil gave exact,
  local path-sample evaluations that are not yet fast enough for a progressive on-screen update.
- Most of the rest was decided at the design stage, before code existed, which costs nothing.

None of these readings settles a legal question, and we do not claim that any mechanism here is free of anyone's
patent. They record what we found, what we did, and how it measured, with dates, so the differences can be checked.

---

Mathew Birchard - DCV Systems, LLC - @birchard_mathew

This page is licensed under CC BY 4.0.
