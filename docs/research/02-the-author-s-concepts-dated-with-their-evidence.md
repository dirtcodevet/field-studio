[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 2. The author's concepts (dated), with their evidence

These are the concepts the author introduced, with the time each was first stated in the project record, how each
is or would be built, and what has been measured. Several have relatives in published work, named where known.

## 1. Contact read from each body's own distance field, with a short-range barrier
**First stated:** 2026-11-12, ~15:30 PT (framed as a magnetic field: a repulsion between surfaces that reaches
its limit where they touch). **Status:** built in two forms and measured on synthetic cases; promising on the tested
soft-on-soft cases, not established as generally better.

**Mechanism as built (2026-11-12, 15:50-17:15 PT; a prototype in the contact work area).**
- *Gap.* A contact sample on body A takes its gap d from body B's stored signed distance field (trilinear
  interpolation and its gradient for the direction). This replaces primitive-pair collision search for the gap
  itself; it still needs (a) a broad phase deciding which bodies and regions can interact, (b) interpolation and
  gradients, and (c) for deforming bodies, keeping B's field current or mapping into its rest frame, which can be
  substantial work.
- *Law.* The repulsion energy uses the published log barrier of IPC (Li et al., *Incremental Potential Contact*,
  SIGGRAPH 2020; IPC Toolkit, MIT licence): b(d) = -(d - d_hat)^2 ln(d / d_hat) for 0 < d < d_hat, zero beyond,
  with force kappa b'(d) along the gap gradient. d_hat = 100 um in every run. This is **regularized near-contact**:
  bodies settle at small positive gaps (49-95 um measured), not at literal zero gap; calling the film a physical
  moisture layer would need separate evidence.
- *Form N (hybrid).* The barrier is the contact law inside a Newton solve on the static force residual (no Lagrange
  multipliers, no energy line search; pseudo-transient continuation; line search on |F|^2). Each iterate is bounded by
  distance: no gap within reach may shrink by more than half its current value (linearised). A diverging force does
  not by itself prevent crossing in a discrete solve; non-crossing here rests on that step bound, the sampling of
  contact points and the field's accuracy, and was checked by an independent count of crossed sample pairs.
- *Form E (standalone).* Explicit integration (symplectic Euler) with the time step bounded by stability and by the
  same distance rule, stiffness-proportional damping, and dynamic relaxation for static rest.

**Measured (synthetic scenes; one machine, 4 threads; seconds indicative):**
| Case | Form N | IPC (same scene) | Form E | Notes |
|---|---|---|---|---|
| Two fingertip pads pressed 2 mm | 27 s | 86 s | 519 s | forces within 1 % (0.353-0.355 N), gap ~80 um, 0 crossed |
| Lips closed 0.8 mm | 16 s | 27 s | 295 s | 0.0124-0.0133 N, 0 crossed |
| Eyelid wrapping a rigid eye | 73 s | 7.4 s | 18 s | a reversal: soft-on-rigid favours IPC here |
| Hand on cheek with a sideways stick check | did not converge in 10 min | not run | not run | open; cause untraced |
| Elbow bend (stand-in arm) | converged every state to 88.75 deg, 0 crossed | 85-90 deg unconverged at its cap | not run | the lattice still folds locally at the crease; hard steps took ~4 min |
| Friction test (block stick / slide) | pass (0.292 mm, 1.68 %) | pass | fail (grip lost to contact-face vibration) | friction needs the implicit form |

**Open:** why the cheek case stalls; the reversal on soft-on-rigid; the elbow's difficulty, which measurement traced
upstream of contact (section 8); a broad phase for many bodies. **Relation to patents:** the formulation is a barrier
force inside an equilibrium Newton solve, as in IPC-family methods; renaming the line-search objective does not by
itself place it outside existing barrier-method patents, and no clearance is claimed.

## 2. A field through the environment as the contact *sensor*
**First stated:** 2026-11-12, ~15:55 PT (framed as a touch-screen analogy). **Status:** proposed; not built.

**Idea.** One field solved through the space between bodies reports where anything is approaching anything, as a
finger is detected by the disturbance it makes in a touch screen's field. Under a fixed-potential, parallel-surface
model, field energy scales roughly as area x (potential difference)^2 / gap, so a narrowing gap is detectable without
pair searches. It is a sensor only: real electrostatic forces between plates are attractive, so the analogy does not
supply the repulsive response, which stays with section 1's barrier.

**Problems to solve before it is worth building:**
- *Self-contact.* Surfaces of one body (an elbow folding onto itself) share a potential, so a potential-difference
  formulation cannot see them. A label-free formulation is needed, e.g. sensing the **local thickness of the free
  space** (the air region's own distance field or its medial-axis width), which falls toward zero between any two
  approaching surfaces regardless of which body they belong to.
- *Normalization.* Signal strength must not depend on arbitrary object numbering or potentials.
- *Resolution.* A gap narrower than one lattice cell vanishes from the grid; the sensor must hand over to per-body
  distance fields before that, or lose the signal.
- *Cost.* A global field solve replaces local candidate searches with another computation; the trade must be measured.

## 3. Persistent state, advanced when its dependencies require it
**First stated:** 2026-11-12, ~16:50 PT. **Status:** proposed; an experiment is planned.

**Idea.** The environment carries persistent fine-scale state (temperature, moisture, pressure, wear, dirt, light)
that is always present. Each region stores its state and the time it was last advanced; a quantity with a cheap
evolution law and known inputs is brought forward in one step (cooling toward a fixed ambient, for instance).
**A timestamp alone is not enough when history matters** (a surface heated, splashed, touched and sheltered while
unseen). The architecture is therefore: **store persistent state, record the disturbances that affect it, and advance
coupled processes when their dependencies require it.** Observation triggers refinement or evaluation; it never decides
whether a physical consequence exists (an unseen wet garment still wets a visible chair).

## 4. Compact encodings: authoritative data versus view-dependent caches
**First stated:** 2026-11-12, ~16:50 PT. **Status:** measured on named cases (two runs, 2026-11-12, 19:31-19:32 PT); at its pre-registered marks the verdict is NO; a render comparison is next.

**Idea.** Data is stored compactly and decoded locally on the graphics processor; on memory-bound hardware fewer bytes
plus a few decode operations can be faster, if decoding is local and random-access. Two kinds of data are kept apart:
- **Authoritative source data** (geometry, contact, editing): precision must preserve surface position and sign, thin
  gaps and connected components, reliable derivatives, and stability under repeated edits. It is never reduced
  according to the current camera, because what is unseen now can matter later (a shadow, a collision, a close-up).
- **View-dependent representations** (display caches): precision chosen by what the viewer can perceive at that place.
The test measures total edit and render cost, including decoding and writes, not only memory saved. Adaptive distance
fields have a substantial history (Frisken et al., *Adaptively Sampled Distance Fields*, SIGGRAPH 2000); any distinct
contribution would be the specific encoding, its update mechanism, and a demonstrated tradeoff.

**First measurement (2026-11-12, 19:31-19:32 PT; one machine, CPU).** 5,000 random finest-level surface bricks of the
human model's current store (each brick 729 samples, today stored at 16 bits per sample), re-encoded and decoded.
Marks written before the runs: the surface's normal displacement p99 <= 5 um and max <= 15 um; no sign change at any
sample with |F| >= 2 um; the surface normal's p99 angle change <= 1 degree; at most 50 % of 16-bit storage.

| Format | Size vs 16-bit | Normal displacement p99 / max | Sign changes | Normal angle p99 / max |
|---|---|---|---|---|
| 8 bits, uniform per brick | 50 % | 24.9 / 121 um (first run's measure) | 1,562 | 2.83 / 59 deg |
| 8 bits, companded per brick (finer near the surface) | 50 % | 2.83 / 4.44 um | 0 | 1.40 / 5.7 deg |
| 8 bits, companded, cube-root curve | 50 % | 2.57 / 3.78 um | 0 | 1.29 / 4.7 deg |
| 9 bits, companded per brick | 56 % | 1.41 / 2.31 um | 0 | 0.70 / 3.4 deg |
| 10 bits, companded | 62.5 % | 0.87 / 2.09 um | 0 | 0.42 / 2.8 deg |

What it shows: at 8 bits with companding the surface itself holds (under 5 um, no sign change); the surface normal is
what misses its 1-degree mark. 9 bits meets every mark except size (56 % against the 50 % mark), so no format passes
all marks together. A flaw in the first run's displacement measure (a crossing's position along a lattice edge is
ill-conditioned where the edge runs nearly tangent to the surface) was recorded and replaced before the second run.
Open: whether a 1-degree normal change is visible at all (a close-up render comparison decides it), and mixed
allocation (more bits only in cells the surface crosses). Not yet measured: decode cost on the graphics processor,
edit-save-reopen stability, thin gaps. Per this section's rule, any adoption starts with display caches, not the
authoritative store.

## 5. Store the recipe, keep the authored edits
**First stated:** 2026-11-12, ~16:50 PT (a reading of the DNA analogy); in use in the generators since 10-30 / 11-09.
A generated asset is **a versioned recipe (generator version, seed, settings, dependencies) + the subsequent authored
modifications + caches that can be rebuilt**, so an improved generator never silently changes an approved character.
Not claimed as an invention; recorded as the project's practice.

## 6. Realism as a bounded test
**Stated:** 2026-11-12, ~16:55 PT. The aim: no difference an observer can detect at their resolution. As an acceptance
test it is bounded explicitly: the viewing range, output resolution, lighting conditions, motions and the comparison
images are specified per test. Unresolved detail is **aggregated, not omitted**: sub-pixel fibres and pores still change
highlights, shadows and apparent roughness, so their collective effect is filtered into what is rendered (standard
sampling and filtering theory). Consistency under changes of light, view, zoom and motion is why a look is built from
the structure that produces it rather than painted.

## 7. Tools behave consistently under an explicit 3D constraint
**First stated:** 2026-11-12, ~17:25 PT. **Status:** a ten-principle framework written (Part 5); the first
form-based drag constraint built in a lane copy and measured.
A 2D mouse drag does not by itself specify a 3D stroke. The promise is: **once the stroke and its 3D constraint are
specified (the surface normal, a tangent plane, an axis, a part, a mode), changing the camera does not change the
operation.** The constraint comes from the form and is shown on screen; screen-space tools remain where intended.
Measured in the lane copy: a 30 mm pull along the normal gives 29.4-29.9 mm from the side, 30, 60 and 80 degrees and
from above (the shipped tool gives 30.1 / 17.7 / 5.7 / 0.8 mm); an out-and-back wiggle returns to 0.00 mm.

## 8. An open question: why the elbow stalls near 85-90 degrees
**Measured 2026-11-12, 17:15-17:50 PT** (synthetic stand-in arm; saved states, no new solves). Three different contact
laws stall at similar angles (85-90 degrees). That suggests a shared upstream limitation, but the three runs also share
the geometry, the material model, the discretization and parts of the solver, so it is not yet a diagnosis. Measured:
volumetric terms are under 2 % of the elastic energy (which does not by itself exclude numerical locking or poor
conditioning); the energy sits 44 % in the front crease and 22 % in a 22 mm plug of soft tissue between bone ends that
do not articulate; the real flexion axis (Visible Human Female, left elbow) lies ~36 % of the arm's depth from the back
skin. Controlled one-change-at-a-time runs from a frozen baseline followed the same evening (Part 6, section B.9):
bones that articulate at the axis did not move the stall (no on every pre-registered mark), and neither did fat at its literature
stiffness. The current plan: the skin as its own sliding sheet over the tissue, on the human model's real arm, with
a 2 mm crease resolution as the control; then the axis depth and the integration formulation, each separately.

## A note on evidence states
Proposed mechanism; implemented experiment; measured result on named cases; reproduced result under a frozen
configuration; accepted behaviour in the intended workflow. Nothing here has reached the last two states yet.

## 9. Earlier concepts in the project record (dated there)
- Rules as precomputed maps (2026-11-04): motion and light solved once into rule maps cheap to look up at run time,
  with neural refinement on top; a body as a formula plus a residual.
- Labels carry the material (2026-11-10): labelling a region applies its material, physics and baseline appearance.
- Built, not painted (2026-11-09): the look comes from building the structure that produces it.
- A sibling runtime engine (2026-11-12, a direction) built for Field Studio's fields.

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

