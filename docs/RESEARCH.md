# Field Studio: a dated research agenda with its demonstrated components

**First public research release. Prepared 2026-11-12 (Pacific time).**
**Author:** Mathew Birchard - DCV Systems, LLC - @birchard_mathew

Mathew is the author of the project and of the concepts in Part 2. The engineering was carried out with AI coding
sessions working under his direction; how that is organised is described in Part 3.

## How to read this document

**What it is.** A dated research agenda with its demonstrated components: where the project is going, what published
work each part learned from, how each mechanism we built works (written so that a skilled reader could rebuild it),
what was measured, what failed, and what comes next. It is released early and on purpose unfinished, so that the
direction, the research baseline, the current state and the future direction are on the public record with a date.

**What it is not.** It is not a claim of novelty: most ingredients are published work, credited where they are
used, and where something may be distinctive it is the specific mechanism, how the parts are combined, and the
measured trade-offs. It is not a product announcement: the tool is not finished, and the hands-on
sculpting test of the rebuilt toolkit is still to come. It is not legal advice and it claims no patent clearance:
every patent reading below is our own non-lawyer's reading of claim text, carried over as recorded.

**Dates.** Times inside the document ("first stated", "adopted", commit times) are Pacific time, 2026, and record
provenance inside the project's private working record. They are not publication dates. The date this material
became public is the date of the release that carries it: the time stamp of the public commit and of each post.
As we understand it, a dated public disclosure counts as prior art only against patent applications filed after
it; it does nothing about earlier ones.

**Evidence states.** Every mechanism carries one label:

| Label | Meaning |
|---|---|
| proposed | an idea or design; nothing built |
| implemented | built and run; not yet measured against a target |
| measured on named cases | numbers exist, for the cases named, on the machine named |
| reproduced under a frozen configuration | the same result obtained again from a fixed, recorded version of the code |
| accepted in the workflow | the project's current model or daily work is made through it |

Very little here has reached the last two states. Failures and reversals are reported beside the successes,
because a later reader needs to know which routes were tried and did not work.

**The machine.** Everything was built and measured on one desktop PC: 32 GB of RAM, one NVIDIA RTX 5050 with
8 GB (which also drives the desktop), Windows 11, shared by several jobs at once. Timings are single-machine
figures and say when other load was present. The tool's own target is 16 GB of RAM and an 8 GB card, and one of
the aims is good light on older cards without ray-tracing hardware.

**The subject.** The test subject throughout is one full-body human model ("the human model"), built from a CC0
base (Blender Studio's Human Base Meshes) and then sculpted by the author and rebuilt as fields. It is a test
body that proved the tool, not the point of the project. Images show it in segments (head, eyes, lips, hair).

**Section labels and references.** Parts 6, 7 and 8 keep their own section labels: B.n in Part 6 (contact), C.n in
Part 7 (posing, cloth, mouth) and D.n in Part 8 (hair, light). A reference such as "B.5" means section B.5 of
Part 6. Short hexadecimal strings such as `506bd914` are commit identifiers in the project's private version
history, given as provenance for a date; they will mean something to readers only once the code is released.

**Glossary.**

| Term | Meaning here |
|---|---|
| field renderer | the project's first GPU renderer of signed distance fields (from 2026-11-03), by sphere tracing |
| review renderer | the full-quality offline renderer used for the review images of skin, eyes and hair |
| card tracer | Field Studio's own viewport tracer on the graphics card ("the card" is the graphics card) |
| CPU tracer | the viewport's fallback tracer on the processor |
| look-dev light | the viewport's lighting mode with a light rig, presets and a still frame that settles |
| lane | one area of work (hair, contact, brushes ...) owned by one AI coding session |
| lead session | the session that plans, briefs and coordinates the lanes and reports to the author |
| lane copy, working copy | a copy of code in a lane's own folder where a change is built and tested before it enters the shared tree; "not landed" means it is still only there |
| sheet | a numbered image page of results shown to the author beside the reference |
| R1 | in Part 6, the constraint-based contact solver being built (early project records also used the name for the native rebuild of the human model, Part 3) |
| E1, E2 | the elbow test cases: E1 a synthetic stand-in arm, E2 the elbow region of the human model's own field |
| d1 to d4 | one-change-at-a-time experiments on the elbow stall (Part 6, section B.9) |
| n2c to n2j | successive variants of the generated head of hair (Part 8) |
| Form N, Form E | the implicit (Newton) and explicit forms of the field-read contact (Part 2, section 1) |

**Code.** The tool will be released as open source. The code is not part of this first release: parts of it are
being redesigned around patents we found while reading (listed in Part 10), and those parts stay out until the
redesign has landed. This document describes them as they are and says where the redesign stands.

## Contents

1. Direction: what Field Studio is and why
2. The author's concepts (dated), with their evidence
3. How the work is done, and the path that led here
4. The field store, its writer, the scaling study and benchmarks
5. Brushes, the toolkit framework, sculpting into contact, and the shell
6. Contact and parts
7. Posing, cloth from measurements, and the mouth library
8. Hair and light
9. The current state, measured
10. Patent readings carried over, and what is held out of the code
11. Future direction

---

# Part 1. Direction: what Field Studio is and why

**The outcome.** The project's standard: extreme, life-like realism, generally aligned with the reference, at every
level of zoom and on every part of the body, so that the result reads as an exceptionally good photograph. Section 6
of Part 2 turns that into a bounded test.

**The tool.** Field Studio is a sculpting, simulation and rendering tool in which every form is an implicit field:
a signed distance field stored on a sparse lattice at several levels of detail (0.25 mm at the finest level today).
There are no meshes anywhere: no triangles, quads or tetrahedra in the model, the tools, the renderer, the
previews or the tests (adopted as a rule 2026-11-07/08). Bodies are fields; hair is curves; cloth is yarn points on
a pattern lattice; contact, posing and light are computed on the fields directly.

**Why fields.** The industry's tools are built around meshes, and many of their tools exist to work around what a
mesh cannot do (adding surface means adding vertices; contact between two sculpted surfaces needs typed-in values;
skinning, light caches and collision all hang off mesh structure). A field has no topology to maintain, can be
cut, merged and grown freely, and gives every point its distance to the surface, which contact, light and brushes
can all read. The direction set for every part of the work (2026-11-12): the system is field-based, not mesh-based,
and its real test is whether it reaches the targeted realism and completeness with the full capability of fields,
without needing expensive hardware.

**Why a tool.** Until 2026-11-11 the project built one human model by direct engineering. On that date the aim
changed: build the tool, so that a person finishes the work by hand, quickly and intuitively. The first yardstick
is plain: a sphere to a rough head in under five minutes. The first runs did not meet it (Part 5), and on
2026-11-12 a hands-on sculpting test could not block out a head with the tools as they were. That led to a
review of the whole toolkit against an experienced sculptor's expectations (Part 5).

**Built to last.** Each subject should need less hands-on work than the one before: approved work becomes the
target that generators learn to reproduce from references (hair, skin, iris, teeth, garments, materials), stored as
a recipe (generator version, seed, settings) plus the authored edits.

**Open source.** The tool will be fully open source. No method that we read as covered by a patent is meant to
ship; where we find one, we study the published research and build a different mechanism, and record which claim
element ours lacks. Sources whose licences limit commercial use are used for study only.

---

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

# Part 3. How the work is done, and the path that led here

This part records the working practice behind Field Studio, and the dated path from a mesh pipeline in Blender to a
mesh-free field tool with its engines: the field renderer, the native rebuild of a human model, skin, eyes and the
materials library.

**Dates.** Every date and time below is an internal provenance time in Pacific time, year 2026, read from a private
local repository (commit times) or from the project's dated records. They show the order of events. They are not
publication dates. The only public date is the release date of this disclosure.

**Evidence words.** Each mechanism carries exactly one of five states:

- *proposed*: written down, not built;
- *implemented*: built and run, with no measurement we rely on;
- *measured on named cases*: numbers exist for the cases named, usually from a single run;
- *reproduced under a frozen configuration*: re-run to the same result from recorded inputs and hashes;
- *accepted in the workflow*: in daily use, by an explicit adoption decision.

Nothing here claims novelty, protection or freedom to operate. Published methods are credited where they are used.

---

## 1. How the work is done

The project is one author, Mathew, working with AI coding sessions on one desktop computer. The practices below were
adopted one at a time, usually after something went wrong. All of them carry the state *accepted in the workflow*.

**1.1 One short charter (11-02) and a short list of principles (11-06).** A single charter of at most 30 lines
states the outcome: extreme, life-like realism, generally aligned with the reference, at every level of zoom. Every
other document is a reference, a tool or a dated record, to be weighed and not obeyed. A list of principles and
machine mechanics was adopted on 11-06. Direction is read in three tiers: an observation is a hypothesis to test; an
explicit direction holds for that task only; a standing rule exists only where it is set as one.

**1.2 Outcome first (stated 11-06, the default since 11-09).** Define the target outcome from the references with
precision, so that it holds in every view, light, zoom and motion, and build it directly. Causes (anatomy, physiology,
physics) come in as drivers of variation and states, or where the outcome is itself a behaviour such as contact or
motion. Physics is never tuned until a shape appears. This replaced a cause-first default. The cost behind it: skin
colour built up from chromophore causes kept producing blotches that were hard to trace, across rounds on 11-02, 11-06
and 11-07 (hours of work, by the record). It was adopted on 11-09 with one provision: a lane whose outcome genuinely
needs causes first says so plainly and builds that way (`engine/lessons/outcome_first.md`).

**1.3 Built, not painted (11-09).** A look has to come from the structure that produces it (form, volume, translucent
layers, an optical medium, motion), so it holds when the light, view or zoom changes. Blur used to soften, baked or
painted creases and pores, painted shading and colour decals are not allowed unless that operation is what really
happens. Defects are fixed at the shape, never hidden.

**1.4 Fix a named problem at its cause (charter, 11-02).** When a problem is named, the work finds what
causes that point and resolves it there. Making the surroundings look right instead is the one thing the charter
forbids. Two recorded costs. On 11-02, eight builds tried to shape a nose and an upper lip with numeric displacement
fields fitted to 2-D outlines, each stacking an offset on the last; after about 20 hours the nose was still clearly wrong.
A profile line says nothing about width, so everything off the line was guessed by falloffs. What worked was a real
3-D form: those features were sculpted by hand. On 10-31, a one-shot hair rebuild ran ten rounds of fixes that
each made the next artefact; it was recorded as a failure and archived, not built on.

**1.5 Measure before claiming.** Numbers locate a fault and confirm a change; the rendered image, looked at beside
the reference, decides whether a form is right (`engine/lessons/measuring.md`). Pass marks are written before a run and not moved
after it. Every number is labelled measured, derived or estimated. Failures are kept as run. An instrument is proved
once on the failure it exists for: one closest-point routine with a one-token bug made 11-38 % of distances too small,
and its own self-check could not see it. In one overnight run about five results were withdrawn under these habits,
two of them over-claims in coordination summaries.

**1.6 Each round is reviewed beside the reference.** Each round is shown as a numbered sheet beside the reference:
the current model through the real renderer at full quality, looked at first by the session that made it. What enters
the current model is an explicit decision. The cost behind the real-renderer requirement: on 11-03 a session's own
render path moved the eyes of the hand-sculpted face, and the work was rejected outright. A face guard followed (a content hash over the field bricks of the
face region, checked before and after every edit). On 11-04 a review found the checksum then in use had been guarding
a superseded file outside the render path; it was replaced.

**1.7 One session per work lane, with a lead that plans and briefs (11-08).** In the overnight run of 11-07 to 11-08,
three lanes each owned by one session advanced, while the Field Studio integration, split into 30-minute hand-offs
between fresh sessions that each re-learned the plan, barely moved in four rounds
(the project's lesson record of 11-07). Since then one session owns each lane continuously; a lead
plans, briefs and arbitrates and does not carry lane work. Briefs carry the outcome, the direction as given, the machine
constants and a completion condition, not a method. Other costs from that night: flags lost in short summaries, and a stuck GPU
process nobody was permitted to stop, which held the card for three hours.

**1.8 One shared machine.** All sessions share 8 cores, 32 GB of RAM (16 GB until 11-06) and one 8 GB RTX 5050.

- *A GPU queue with lane leases* (`engine/core/gpu_lock.py`, first version 10-26). One job on the card at a time. A
  lane leases the card for up to 20 minutes and keeps it through gaps of up to 3 minutes, so a session can render,
  look, fix and render again; the next lease goes to the lane with the oldest ticket. Costs in its history: on the
  first day the card was held about 12 hours against 21.5 hours of summed waiting, and one job held it idle for 146
  minutes waiting for RAM. On 11-06 an instruction to release the card between stages made sessions trade it minute by
  minute; the lease was the answer.
- *A RAM guard* (`engine/core/mem_guard.py`, 10-30). A heavy job runs inside an operating-system job object, its peak
  is recorded, and it is ended if free RAM stays under a floor. Written after renders swamped the 16 GB machine twice
  and the watchers beside them could not stop them.
- *Task time caps.* A task (one shape, one run, one fix) stops at 30 minutes; the session looks at why it ran over and
  continues. Nothing enforces the cap, and it was broken: helpers ran 70 to 142 minutes on 11-06.

**1.9 Records.** A ledger keeps every request and run outcome and is reconciled after each run (since 10-26). Work is
committed locally; the repository has no remote. Each change of lead writes a dated hand-off: what is measured, what
is *not established*, each lane's state, and the decisions still open. Incoming sessions take the work as
it actually is, not as it is stated to be, and look at the latest images before trusting a status line.

**1.10 Sources and licences (since 10-26).** No source or asset whose terms limit commercial use is used, except to
study. Facts and numbers from a paper are used as facts; its text, figures, data and code are not copied. One ledger
(`engine/LICENSES.md`) lists each source with its licence, credit line and whether it ships.

**1.11 Mesh-free everywhere (11-07 and 11-08).** No triangles, quads, tetrahedra or element meshes in the model, the
tools, the renderer, previews, tests or scaffolding. Bodies, cloth, hair and fluids are fields, lattices, particles or
curves. The steps: vertices dropped for the model on 11-03; no meshes anywhere, as a standing rule, on 11-07; test benches
included on 11-08. What it cost: a 2-D finite-element bench used for material proofs was retired that night, and a
real-time web viewer with a mesh skin was archived.

**1.12 Patents (11-11).** The tool is intended for an open-source release. A patented method is studied, not shipped. Where
a published patent appears to cover something the tool needs, the lane designs a plainly different method and records
its reading of the claim elements one by one. These are engineering notes by non-lawyers; the searches are not
exhaustive, and no clearance is claimed.

---

## 2. The path

### 2.1 The mesh era in Blender (to 11-02), kept as record

The earliest dated build records are from 10-19. The human model was a polygon mesh in Blender, rendered with Cycles.
Its base body came from a free base model from Blender Studio (Human Base Meshes v1.4.1; every asset's licence field
says CC0, per `engine/LICENSES.md`), and the face was sculpted on it by hand. This era left procedural skin relief
driven as render-time displacement, seeded iris and hair tools, and the measuring lessons above. It also hit a wall:
at full skin detail Blender ran out of memory on every close frame on this machine (12.25 GB) and finished 31 of 120
frames in 30 minutes.
**Evidence: measured on named cases** (one model, one machine).

### 2.2 Vertices dropped (11-03)

At 00:12 the problem was recorded: treating a human form as millions of vertices with modifiers makes realism at fine
detail close to impossible. At 17:47 the decision followed: the vertex-based framework is dropped entirely. Cage, subdivision
and displacement fallbacks were set aside too. Success would be measured on the field side: the same or better look
for less time and memory on an 8 GB card.
**Evidence: accepted in the workflow.**

### 2.3 The field renderer (first run 11-03, about 18:09)

**How it works.** A form is a signed distance field F(x), negative inside, stored as sparse bricks of 9 x 9 x 9
half-float samples (8 cells per edge). The first head used one level at 0.25 mm: 136,019 bricks, 608 MB, built in
17.3 s from the mesh's render triangles as a one-time import. The renderer is a set of CUDA kernels compiled at run
time from Python. Each pixel marches a ray by sphere tracing (Hart, "Sphere tracing: a geometric method for the
antialiased ray tracing of implicit surfaces", The Visual Computer 12(10), 1996; a published method, used as
described). The field is read by trilinear interpolation; the normal is a central difference at the cell size. Three
additions:

1. *A safe march over displaced skin.* Skin micro-relief is a formula evaluated during the march, not stored geometry.
   Each 0.25 mm cell stores the minimum and maximum of that displacement. Outside the band the ray steps by the stored
   distance with no formula call; inside, the step is F / (1 + G), with G the cell's slope bound. The last point with
   F > 0 is kept, and a hit is refined by 8 bisections; the cap is 128 steps. Against a reference march with 0.5 um
   steps, overshoot went from 0.02-0.15 % of samples to 0.
2. *A detail pyramid.* Detail is evaluated only where the pixel footprint can resolve it, and folded into the shading
   below that. Macro frames went from 6.4 s to 0.33 s with no visible popping.
3. *Light as a cached response.* Sun visibility, sky occlusion and subsurface transport are solved once per field
   sample for a lighting setup and reused across frames, keyed by a content hash of the files the bake reads (906 s
   the first time for a full body, 0.4 s from the cache).

**Numbers, with their scope.** On the head, 1920 x 1080, 120 frames: the cached mode took 4.89 ms of kernel time per
frame against the same renderer's brute force at 425.5 ms, with a mean image difference of 0.96/255. Whole-shot wall
time was about 7 s against about 300 s for Blender, a figure extrapolated from 5 frames. The record lists why that is
not like for like: the field renderer computed less light transport, the materials differed, Blender was not tuned,
and the cache is valid only while light and subject stay still. Its own conclusion was a qualified yes. A later round found
the field's path tracer no faster per sample than Cycles: the speed comes from the cache, not from the field format.
**Evidence: measured on named cases** (single runs, one head, one machine).

### 2.4 Full body, and off the mesh (11-04)

By the evening of 11-04 the whole body was one tiled field on four levels (0.25, 0.5, 1 and 2 mm; fine levels only
where close shots go), each brick also carrying a material id and a 3 x 3 x 3 lattice of 32 attribute channels. A
review that night mapped every place the mesh still governed (the build, the publish gate, camera placement, the face
guard) and wrote field-only replacements: a content hash of the bricks as the model's identity, and cameras snapped to
the zero set along their own view ray. One of its tests failed as first written and is kept as a fail beside the
corrected run. It also measured that the field faithfully reproduced the old mesh's facets where triangles were coarse
(normal steps of about 1.5 degrees every 3.5 mm on a shin).
**Evidence: measured on named cases.**

### 2.5 The native rebuild (design 11-05; first native region 11-05 23:45)

**Why.** A field sampled from triangles carries their facets, and every later edit sat on them.

**How it works.** The model becomes *parts* joined by *interfaces*. Only the parts are authoritative; the brick grid
is a cache that one deterministic writer regenerates from them, bit for bit.

- A native part is a formula plus a residual. The formula is a few primitives (round cones, ellipsoids, superellipse
  sections swept along a cubic axis) joined by smooth minimums. The residual is a hierarchical cubic B-spline in
  sparse blocks at 8, 4, 2 and 1 mm, refined by exact subdivision. On the neck, one 20-parameter tube left a radial
  residual of 7.8 mm at the 95th percentile, and 231 coefficients left 0.19 mm. So the formula is the frame (rest
  coordinates, part identity, edit handles) and the residual carries the form.
- Two descriptions of the same skin are joined by a partition of unity with quintic weights over a shared base, not a
  smooth minimum, which swells where the two disagree.
- *The splice refuses instead of forcing.* The mismatch between old and new is measured across the band. The splice is
  admitted only if the band is at least 107 times the largest mismatch, which bounds the extra turn of the normal to
  1 degree (20 um needs 2.1 mm; 200 um needs 21 mm). Otherwise it refuses and returns a map of where and by how much.
  The fix is to refit the new part against the old field's values on the band, never to smooth afterwards.
- Separate tissues that touch are never blended: each part is stored whole and the visible field is their minimum.
- Every edit is an entry in a replayable operation log; undo is exact.

**Tolerance.** On 11-05 a tolerance was set for rebuilding the hand sculpts natively: about
20 um (pass mark: 1 mm-averaged offset, 99th percentile at most 20 um), in place of byte copies.

**What happened.** The neck was native by 11-05 23:45. The first full assembly, on the night of 11-06, was refused by
the writer, as designed: two joins did not fit and three had no overlap. By the 11-06 roadmap, six of ten regions were
native in one full-body cache that matched the old version on the GPU (mean difference 0.01 to 0.22 of 255); the rest
were blocked where three or more parts meet or slits close. One limit was reported plainly: where the old mesh's
stair-steps sit on a sculpted fold, holding the sculpt within 20 um means re-drawing the stair-steps. Whether every
region finally met the 20 um bar is **not established** in the records read for this part. On 11-09 the rule was extended:
no part is held as final, the hand sculpts included; flaws are fixed and small shifts are fine.
**Evidence: measured on named cases.**

**Finer than 0.25 mm.** The renderer's finest body level is 0.25 mm. Contact leaves at 0.125 mm were built for tight
folds, and the skin generator writes a 0.125 mm level where fine detail matters. A feasibility read on 11-09 found
0.0625 mm per-part lattices would fit the viewport, while a whole body at 0.01 mm would not fit this machine. Per-part
lattices finer than 0.125 mm:
**Evidence: proposed.**

### 2.6 Field Studio and the tool switch (11-05 to 11-12)

Field Studio began as a CPU-only Windows tool that sculpts directly in a field, every edit a deterministic, undoable
command (first committed 11-05). Its viewport moved to GPU tracing and sculpting on 11-08, and a product design
followed on 11-10.

On 11-11 the main effort changed. A round on the lips had come back with the lips pressed together in a way a person
would not have shaped them. The conclusion drawn: the sessions are strong at code, physics, measurement and research,
and weak in the fine visual refinement loop. So the work became the base itself (the tool, the studio and all the
components it needs) so that a person can finish the work by hand, within 16 GB of RAM and an 8 GB card for the whole
system.

**Test 1.** With front and side reference images on one centre axis, sculpt a sphere into a basic head (head shape,
eye indents, nose, cheeks, chin and jaw line) aligned with them, in under 5 minutes (first set at 2). Scripted
measures: outline mean within 3 mm, lumps at most 1.5 mm RMS, at least 30 frames per second while stroking, at most
100 ms to the next stroke.

**How the first runs went.** A scripted run in the real window on 11-11 at 10:47 made a lumpy, egg-like form with a flat face,
after 144 actions and an estimated 7.8 minutes. Four passes of Smooth took the lumps from 4.49 to 4.23 mm RMS (6 %).
Strokes ran at 6.6 to 13 frames per second. A neck pulled down with Grab tore into an open ring. 22 of 166 presses
near the silhouette turned the view instead of sculpting. The first finding was about the driver: the brush was too
small. A sculptor blocks in with a brush a third to half the size of the head, and the tool capped
brush size and ran large brushes slowest. Since the run could not tell tool limits from AI limits, each tool was then
measured on standard forms against its equivalent elsewhere. By the 11-12 morning hand-off, 30 frames per second for
small strokes was still not established, and the first hands-on run of test 1 was still open.
**Evidence: measured on named cases** (scripted runs; test 1 is not passed).

### 2.7 Skin

**Micro-relief as render-time displacement (10-23; in the field renderer since 11-03).** Forms that visibly change
shape are geometry. Fine relief (the furrow net, pores) is generated by formulas with seeded variation by region and
evaluated in the march (2.3), so it moves the silhouette and shades under any light; it is not a painted or baked map.
Porting the Blender-era displacement chain reached a height correlation of 0.998. A first round of pores generated
from published sizes was nearly invisible, and is recorded as such. The lesson kept: uniform relief reads fake;
density, size, depth and direction vary at more than one scale.
**Evidence: measured on named cases.**

**Realism targets from a reference study (11-04).** Published in-vivo numbers are turned into a band budget: colour
variation in CIELAB with shading removed, split into fine (under 1 mm), mid (1 to 4 mm) and coarse (over 4 mm) bands.
The working rule: healthy skin keeps coarse-band chroma low (a* standard deviation at most about 0.5 on the face) and
carries its texture in fine-band lightness. A total alone cannot tell blotchy from healthy. Measured on close-up
reference photos, the renderer had it inverted: fine lightness about three times too smooth, mid and coarse colour 1.5
to 4 times too strong. Sources named in the records (facts used, text and figures not copied): He, Xiao, Pointer and
Westland 2020; Schmalwieser et al. 2024; Xiao et al. 2017; Chauhan, Xiao and Wuerger 2019; Mullen 1985; Matts et al.
2007; for optics, Jacques 1998 and 2013, Donner and Jensen 2006, Prahl 1999. Body-site targets other than the face are
estimates. Cadaver data serves form and thickness only; colour comes from living references.
**Evidence: measured on named cases.**

**Colour: look first, by zone, in layers (11-05).** The first field approach solved colour spectrally from chromophore
fields. Its random walk was checked against a published Monte Carlo reference (colour difference 0.5), and the check
showed the older analytic colour model wrong by 7 to 10. An earlier tone multiplier, fitted through a film-look
display transform, had baked that transform into the material. Even with correct optics, colour from causes kept
blotching. The direction adopted: a natural base tone per zone from living references; low natural variation matched
per zone on the band ruler; sparse, faint accents; then triggered states. Causes stay for the states.
**Evidence: accepted in the workflow.**

**The seeded skin generator (started 11-10; `engine/skin/gen`).** Inputs: surface samples with normals and soft zone
weights, a few head landmarks, and about 20 settings (seed, age, sex, base tone, sun, redness, variation, freckles,
pores, texture, per-zone overrides, local refinements). The same seed and settings give the same skin; another seed
gives another skin of the same kind. Colour is a regional base smoothed along the surface, plus tan by exposure, four
seeded bands of variation, and freckles and moles as a seeded point process of irregular soft patches. Relief comes
from seeded growers scaled by age and sex. Output is evaluated at any point and written as lattices aligned with the
model's bricks. It was adopted as the model's default on 11-11. It has no panel in the tool yet and is
re-baked after a shape change.
**Evidence: accepted in the workflow.**

### 2.8 Eyes and iris

**A volume built from fibres (11-03 onward).** The iris is a heterogeneous scattering volume built from generated
fibres at their real scale, over a dark back layer, under an analytic cornea. Globe, cornea and limbus are formulas
(two spheres). The fibre volumes are sparse: one eye's arrays went from 425 MB dense to 102 MB.
**Evidence: measured on named cases.**

**Why four rounds read as a stain (analysis 11-05).** A brown inner zone had read as a circle, then a splat, a
watercolour stain and a starburst. The work paused to study reference irises: colour, then texture, then
structure. Three independent measures agreed. In real irises the amber is the bright material: brownness and lightness
correlate at about +0.5 at fibre scale. In ours it was a dark absorbing film: -0.3 to -0.7. The amber added absorption
and no scatter, and its edge was drawn across the fibres instead of being made by them, so no edge shape could fix it.
The layering was checked against anatomy literature (Yan et al. 2022; Wilkerson et al. 1996; Imesch et al. 1996; Sturm
and Larsson 2009). The reference photographs were iris macros under CC BY and CC BY-SA licences.
**Evidence: measured on named cases** (three reference irises; three others were too soft to resolve the zone).

**What the r22 draft got wrong (11-09).** It failed its own gates as run (9 of 13 texture gates, 6 of 10 geometry
gates, parallax failed). On review it read as blurred rather than natural and too regularly wavy, too symmetrical at the inner border, with
fine detail almost absent. Blur used to soften is painting (1.3).
A fresh session took the lane.

**The seeded generator (11-09 17:40; `engine/eye_lane/iris_gen.py`).** The direction: a seed and settings that give
different irises (colour variation, patterns, crypts), not one eye. Everything is drawn from (seed, settings) on a
5 um grid over the iris plane:

- a collarette as a lobed, notched crown curve with a ridge that varies along it, never a circle;
- traced strands at several depths: ciliary bundles that branch and merge along a shared flow and bend round crypts,
  fine radial fibres in the pupillary zone, and a fine layer everywhere;
- crypts as seeded lens-shaped pits floored by the deep stroma;
- broken, wavy contraction furrows, and a beaded pigment ruff at the pupil margin.

Relief and collagen density come from distance fields to the strands. There are about 80 settings and four presets.
**Evidence: measured on named cases** (two to four seeds shown beside reference photographs).

**Colour carried by the fibres (a finding of 11-09, about 21:30).** The blue irises worked because their
colour comes out of the fibre structure itself. For green, hazel and brown, a separate pigmented tissue had been laid
over the strands; it hid the fibre architecture and read as mud. That one choice drove the failure of effectively
every eye except the blue. The direction: one fibre architecture for every colour, with the pigment
carried inside the strands and tissue, per strand, layer and zone. The 11-11 status still lists this as missing, along
with a whole-eye generator.
**Evidence: proposed.**

### 2.9 The materials library (11-07 and 11-08)

**Schema and profiles** (`engine/materials/SCHEMA.md`). One JSON file per material, read by the renderer, the contact
solver and the deformation solver alike: an optics block (layers, surface, micro-relief, states), then contact
(friction, adhesion, wetting, a required solver sub-block), deformation (model, density, moduli, anisotropy) and
thermal values. Every leaf number is a small record: the value, a quality letter (M measured in the source, D derived
here, E estimated with the reason) and a source key; each source entry carries the citation and a licence note. An
unknown value is written as null with "not found", never invented. Values that exist only per pair of materials live
in pair tables; with no entry, a lookup combines the two materials' own values and labels the result an estimate.

The first commit (11-08 10:45) held 72 profiles and 7 pair tables; 75 profile files are on disk at writing. Families
include human tissues, eye, hair, fabrics, woods (maple, oak, pine, walnut), stones (granite, limestone, marble,
sandstone, slate), metals, glass, soils and fluids. Bench renders exist for fabrics, water, tissue and a skin macro,
beside licensed reference photographs. As of the 11-11 status the tool's viewport did not read the profiles, and this
part did not re-check that. A separate catalogue (11-09) plans 215 further profiles; it is planning only.
**Evidence: implemented.**

**Labels carry the material (design, 11-10).** A label is the front door to a profile: a part labelled as an
eye is known to Field Studio as an eye. Labelling a part applies its material, physics and generator. Proof sheets for
stone and wood, rendered in the field renderer beside references, are future work.
**Evidence: proposed.**

### 2.10 Published ingredients, credited

| Ingredient | Source | Terms, as recorded |
|---|---|---|
| Sphere tracing | Hart 1996, The Visual Computer 12(10) | published method; no text or code copied |
| Base body of the human model (mesh era) | Blender Studio Human Base Meshes v1.4.1 | CC0 |
| Cross-section anatomy, for study | Visible Human data, courtesy of the U.S. National Library of Medicine | NLM terms; study only, not shipped |
| Bones derived from that data | Andreassen et al. 2023, Scientific Data | CC BY 4.0; study only, not shipped |
| Skin and iris numbers | the papers named in 2.7 and 2.8 | facts used; text and figures not copied |
| Reference photographs on sheets | public-domain, CC0, CC BY and CC BY-SA images | credit, and share-alike where it applies, stay with any sheet that shows them |

### 2.11 What this part does not establish

- Test 1 has not been passed, by a script or by hand.
- Whether the whole human model meets the 20 um native bar.
- That the material profiles drive appearance in the tool.
- Any timing as a repeated measurement: nearly all are single runs on one machine with other jobs running.
- Colour carried in the fibres for non-blue irises: direction only.
- No mechanism in this part has reached *reproduced under a frozen configuration*. A freeze of the 11-04 renderer state
  exists (hashes re-verified on 11-05), but its minimum viewer build was never run and the engine has changed since.

---

# Part 4. The field store, its writer, the scaling study and benchmarks

This part describes how Field Studio stores a form, how edits are written into the store without damaging it, how the
store is checked before a version is published, how the interactive editor mirrors it on the graphics card, and what we
measured about its cost. It is written so that a skilled reader could rebuild each mechanism.

**How to read the dates.** Dates and times are Pacific time, 2026, and are internal provenance: when a mechanism was
first stated or adopted in the project record (the private version history, by commit time, or a dated lane record
where the commit came later). They are not publication dates. The public date of this text is the date of the release
that carries it.

**Evidence labels.** Each mechanism carries exactly one of: *proposed*; *implemented*; *measured on named cases*;
*reproduced under a frozen configuration*; *accepted in the workflow* (the published model is built or edited through
it).

**Machine.** Everything was measured on one Windows PC with 32 GB of RAM and one 8 GB RTX 5050 card, shared with other
jobs and with the desktop (which holds about 1.0-1.3 GB of the card). Where a number was taken beside other load, it
says so. The tool's own budget target is 16 GB of RAM and 8 GB of card memory. Unless stated, "the human model" is one
full-body figure at 0.25 mm finest spacing.

**Limits of this text.** Nothing here claims that a method is new, protected or clear of anyone's patents. Patent
readings are a non-lawyer's reading of claim text, carried over as recorded; they are not legal advice, no legal review
has been done, and no clearance is claimed. Published work we learned from is listed in the last section.

### 1. The brick lattice and its four levels

*First in the record 11-04 (the tiled full-body field); in the version history from 11-05 23:19 (`63d07d7f`).
Evidence: accepted in the workflow.*

- Space is cut into cubic **tiles** of 32 mm. Per level, a tile holds a contiguous run of **bricks**. A brick is
  8 x 8 x 8 lattice cells stored as **9 x 9 x 9 samples** (729 values); the ninth sample on each axis is the face shared
  with the next brick, so trilinear interpolation inside a brick never needs a neighbour.
- Samples are signed distances in metres (negative inside), stored as 16-bit floats (1,458 bytes a brick). Each level
  has a cap; a sample at the cap (|v| >= 0.999 cap) means "farther than the cap, no distance known", never a distance.
- Four levels, each twice as coarse as the one below:

  | level | node spacing | brick edge | band kept round the surface | cap |
  |---|---|---|---|---|
  | L0 | 0.25 mm | 2 mm | 1.5 mm | 3 mm |
  | L1 | 0.5 mm | 4 mm | 2 mm | 3 mm |
  | L2 | 1 mm | 8 mm | 4 mm | 6 mm |
  | L3 | 2 mm | 16 mm | 8 mm | 12 mm |

  L3 and L2 cover the whole body (L3 its inside too). L1 and L0 exist only in a band round the surface and only in
  regions that carry them. On the human model as of 11-12: 218,766 L0 bricks in 444 tiles, 72,955 L1, 64,609 L2,
  16,070 L3. (An 11-04 version held 145,575 L0 bricks; the count grew as regions were added.)
- Files of one version: `meta.json`, `tile_grid` (tile coordinate -> tile index), and per level `tiles`,
  `tile_table` (tile -> slice of bricks), `brick_ijk` (global brick coordinates), `sdf` (the samples), `mat` (an 8-bit
  material per sample) and `attr` (a 3 x 3 x 3 attribute lattice per brick, interpolated trilinearly).
- **One evaluation rule, used by the CPU reader, the card, the writer and every scan** (editor base node 11-08,
  `core/fv8base.py`, `dc56e0ad`). At a point, the finest level holding a brick there answers, trilinearly inside that
  brick. A level with no brick there gives only a bound (its band minus its spacing) and the next coarser level is
  read. With no level at all, the point is outside by at least the largest bound. A capped sample takes the finest
  coarser level's uncapped value at that position.
- **Contract between levels:** every L1-L3 node is also an L0 node position, and a coarse node should equal the exact
  restriction of the finest data there. Every writer below enforces this and every scan below measures it.
- **A detail-bound pyramid on L0.** Each L0 brick that holds surface has a slot of 341 cells x 12 values (fp16): a
  quadtree (1 + 4 + 16 + 64 + 256 cells) of minimum and maximum bounds and tilt for the render-time skin relief, so the
  tracer can bound the displaced surface. It depends on the samples, materials and attributes, so an edit lists the
  slots it made stale and they are re-solved. Measured 11-04 on the CPU (64 x 64 samples per brick, 488 bricks a chunk):
  5,706 slots plus 1,000 checks in 87.9 s at 2.3 GB peak; all 84,232 slots in 47.7 min at 2.35 GB peak. A first
  whole-set solve on one dense box needed about 11 GB and over 50 minutes and was replaced by the region-by-region
  solve. *Evidence: measured on named cases.*

Accuracy of the format on analytic fields (11-04, all four levels built from functions, no mesh): a 12 mm sphere's zero
set is off by at most 1.3 / 5.2 / 20.9 / 84.4 um at L0 / L1 / L2 / L3; normals jump at most 0.02-0.18 degrees across
brick faces. Two failures of thresholds written before the run are kept as failures: the per-cell gradient bound exceeds
1 by about h / r on curved surfaces (1.25 at L3 on r 12 mm; it is still a safe bound for stepping), and L3 misses a
100 um zero-set line by 2-9 % on blended forms.

### 2. Versions: copy-on-write by file, undo and replay

*In the record by 11-05 (`fv8_field.FieldStore.commit`). Evidence: accepted in the workflow.*

- A version is a folder in exactly the format the renderer reads, plus `version.json`: its parent, the op log (plain
  parameter dictionaries, replayable), the changed bricks and tiles per level, stale pyramid slots, and a SHA-256 of
  every array. Its identity is a **field content hash**: SHA-256 over (name, SHA-256 of the file) for every file the
  renderer reads.
- **Committing writes a new folder.** Arrays the edit did not change are **hard links** to the parent's files (a plain
  copy where the volume refuses links). Changed arrays are written whole as new files through a temporary name and an
  atomic replace, never in place, so a parent can never be changed through a link. Rolling back is pointing at an older
  folder. A pointer file names the published version and keeps the previous entry.
- In memory, a store opens lazily with per-brick arrays memory-mapped read-only; edits live in an overlay (brick ->
  fp16 block) with the old blocks kept for exact undo.
- Measured 11-04 on two real crops (50 seeded smooth and displace edits each): undoing all 50 returns the original hash
  bit-exactly; replaying the log twice gives the forward hash; no brick changes outside an edit's support; edit time
  p50 2.3-22 ms, p95 4-61 ms over all levels; commit 0.12-0.24 s; the renderer's own reader matches the store's
  evaluation to 1e-18 m. Brick allocation where a moved surface leaves the band (new bricks seeded from the next
  coarser level, shared faces copied, tiles re-packed at commit): 0 missing-sample hits after a +2 mm move against 82
  without it; undo including the allocation is exact.
- Measured 11-12: removing 75 superseded versions freed about 154 GB. A hard link shares bytes only with versions that
  still exist, so each version costs what it changed until it is pruned.

### 3. The writer: one transaction per edit

*First built 11-10 to 11-11 (`fv8_write.py`, op `write_txn`, `48581bfc` 11-11 01:59); the store's default writer since
11-11 13:49 (`e788e94c`); the safe sign rule 11-12 18:47 (`dd32c23c`). Evidence: accepted in the workflow (the sign
rule itself: measured on named cases).*

**Why.** Four faults were measured on 11-10 with the earlier per-slab writer: new bricks seeded from one estimate while
the edit's delta was taken against another (up to 1.44 mm off on 727,000 nodes in one edit, showing as specks and
slits); a move written in a narrow band left coarse levels at the old shape (the renderer's coarse skip then cut slits
into moved skin); a brick sharing a face sample with an edited brick kept its old copy; capped samples were
re-estimated from levels an earlier slab had already changed (about 1 mm ridges and 2,926 near-surface sign errors in
one cavity wall). All four are one fault: the edit was not evaluated against one fixed source state.

**Steps.**
1. Input is a target field T on the L0 lattice (NaN means "leave this node"), or a delta against the editor's view
   taken inside the op before anything is written. All reads come from the pre-op store; all writes are staged and
   committed at the end as one record and one undo. The op chunks its own work (4,096 bricks at a time).
2. L0: every brick holding a target node receives it, in **every copy of a shared face sample**. A brick missing where
   the target enters L0's band is allocated inside L0's existing tiles; its other nodes take the same frozen estimate,
   so estimate + delta = target.
3. L1-L3 are rebuilt from the finest geometry. A coarse node that is a target node takes the target (exact
   restriction), except a target at L0's cap (sign only) or one that overstates the distance (capped to it). Every other
   coarse node within its level's cap of the changed surface takes sign x (distance to the new zero set - eps). The new
   zero set is a **cloud of lattice-edge crossings** of the finest level present, at L0 spacing, searched with a k-d
   tree. Points only; no surface is reconstructed. eps = 0.1 h0 + 1 um (26 um), so a coarse value does not overstate
   the true distance beyond the cloud's chord error and distance bounds taken from a coarse level stay conservative.
   Nodes whose distance moved less than 10 / 20 / 40 / 80 um (L0..L3) keep their exact bytes.
4. **The safe sign rule.** The sign of a rebuilt non-target node X comes, in order, from: (a) a target node k with
   s_k > |X - X_k|, where s_k = min(|T_k|, the new cloud's distance at X_k) - eps. A distance field is 1-Lipschitz, so a
   node lying s_k from the new surface has no surface within s_k, and any X closer than that is on k's side whatever
   the surface's shape. The 8 nearest target nodes are tried; if they disagree the node is counted and passed on.
   (b) Otherwise the store's own uncapped L0 value at X. (c) Otherwise the normal at the closest cloud point. A stored
   value on the side a target node proves wrong is never kept.
5. The op is a plain dictionary (data zlib-compressed, base64, with its SHA-256); replaying it gives the same bytes.

**Reversals on the way to this rule.** An 11-05 re-distance op built its zero set with marching cubes; it was removed
on 11-11 (`40c97886`) when the project became mesh-free everywhere, and old logged ops now replay through the point
cloud. An 11-11 sign order that put the closest point's normal first (`4f4d13e9`) fixed one stale-sign case and then
failed on 11-12 at rims and cavities, which is what rule (a) replaced.

**Measured.** On the case that exposed it (removing tissue from inside a separately owned eyeball-shaped part): before,
91 / 16 / 2 / 0 wrong-way samples at L0-L3 in a ring 0.47-1.11 mm outside the ball and 154,588 nodes of tissue left
inside it; with the rule, 0 at every level. On a sculpt write-back to the human model (11-11, CPU, about 65 s, 3.7 GB
peak): surface error at most 0.075 um at 19,750 points, 0 sign flips of 58,655, 393,728 untouched bricks
byte-identical, L1-L3 with 0 overstated distances; 167 L0 samples within 0.5 mm of the surface still overstate by more
than 0.1 h (at most 0.41 mm), because the editor's field there is stretched (gradient up to 2.76) and was kept so the
surface would not move. The engine regression set (13 tests) passes.

**What the certificate assumes, and what is not established.** Rule (a) is sound only if |T_k| does not overstate the
distance to the new surface and the cloud misses no crossing near X_k. The first is guarded (s_k takes the smaller of
two independent estimates) but not verified; a per-op measurement of the target's edge Lipschitz ratio is *proposed*.
The second is measured only on the crop cases; sheets thinner than one cell, which can cross a lattice edge twice with
no sign change, are **not tested**. The chord-error allowance is argued, not separately measured. Rules (b) and (c)
carry no certificate; (b) trusts existing data, which can itself be damaged. Writer correctness and repair of existing
damage are kept as separate questions: damage is found by the scans in section 5 and repaired as its own gated edit.

### 4. Keeping brick-face samples consistent: the seam guard and the publish gate

*Publish gate by content hash and region guard 11-05 (`fv8_current.py`); seam guard at commit and seam / duplicate
checks in the gate 11-12 10:39 (`8013917f`). Evidence: accepted in the workflow.*

- **A seam** is a node whose copies in neighbouring bricks differ. On 11-12 a scan of the published version found
  108,237 / 30,263 / 7,717 / 1,082 seam nodes at L0-L3, 8,899 of the L0 ones within a cell of the surface and 169 + 33
  with opposite signs. They had been written by earlier writer paths. They were repaired at the data (a finer level's
  uncapped value at L1-L3, otherwise the copies' midrange; L0 moved at most 29.5 um), 4 + 2 enclosed air bubbles were
  filled and 7,219 unreferenced duplicate brick rows dropped. The next version scans clean and was published on the
  condition that the cause be fixed too.
- **The cause proof.** Five real strokes written back through the editor: the pre-fix writer produced 4,053 L0 and 73
  L1 seams; the fixed writer (`8f2333eb`, 11-12 09:06) produced 0 at every level. One earlier unpublished write-back
  carried 9,051 new L0 seams of up to 5.4 mm.
- **The seam guard** runs inside every commit: every changed face sample is written into every brick that holds that
  node, whatever code staged the change; stale holders it repairs are counted in the report.
- **The publish gate** refuses a version unless: (1) every file the renderer reads is present and structurally
  consistent (brick counts, tile tables in range), with no seams and no duplicate rows; (2) a per-level diff against
  the published version shows changes only where the edit was meant to act and only in the intended direction (for a
  removal: inside -> outside only; other parts unchanged); (3) a **region guard** holds: for a protected region (a
  sphere round part of the face), per level, the bricks whose box meets the sphere are sorted by global brick
  coordinate and hashed as coordinate + sample + material bytes; any difference refuses the publish and names the
  level. For write-backs from the editor a further check compares the stored sign with the editor's field round every
  edit. Measured: the gate refused a removal whose old writer flipped 614 L0 and 87 L1 samples outside -> inside in a
  ring 12.6-16.5 mm from the part's centre (11-12 17:13); that refusal led to the safe sign rule. The earlier
  published version is now refused by the gate; the repaired one passes.

### 5. Level-consistency and coincident-surface scans

*First built 11-12. Evidence: measured on named cases.*

- **Level consistency.** For every coarse node, compare the sign the coarse level stores with the sign the finest
  level gives there, and group disagreements into connected islands (phantom tissue, phantom air). Read-only, 11 s for
  the whole body. Published version: 25,421 nodes in 186 islands, all in the head, the largest two filling the eye
  interiors. An earlier relative of this fault (11-04): the original builder wrote +cap inside the body wherever no
  source surface lay within the cap, giving millions of capped samples beside uncapped negatives; a sign-repair op
  fixed those from the level with the larger cap.
- **Coincident surfaces between owned volumes.** The body still held its own copy of each eyeball: about 6.4 cm3 of
  body inside each eye part (about 85 % of it) and about 1,500 mm2 where both surfaces lie within 0.05 mm (sampled on a
  0.1 mm grid). In close renders the two surfaces trade pixels (2,000-4,700 pixels per eye view hit within 0.2 mm of
  each other), seen as stepped rings and serrated lid margins that more render passes do not remove. The fix is at the
  data: body := max(F_body, -F_part) over the part's box, written through section 3 and gated by section 4. **Not yet
  done as of this text:** the re-save with the fixed writer and a small set-off between the two surfaces (its size to
  be chosen from render tests) are *proposed*.

### 6. The interactive editor: sculpt layer, card mirror, undo and save

*First built 11-08 (`54529cc3`, `7e732504`, `d69dcb28`); band admission and stroke-own sync 11-11 to 11-12. Evidence:
accepted in the workflow.*

- **Editable copy.** Opening the published version (2.0-2.5 s) gives an object whose field is
  F = F_store + dF_sculpt: the read-only store plus a sparse delta on the object's lattice, held as 8^3-node bricks in
  a pool with one row per brick. Brushes write the delta; writing back (section 3) makes a new version and never
  changes the open one.
- **Card representation.** The card holds the store, read in place by the rule of section 1, and an **atlas** of rows
  (9^3 fp16, 1,458 B each) for every brick that holds a change plus a ring of neighbours. A row holds the object's full
  current field there. Invariant: a cell that holds a change always has its row; everywhere else the store is the
  current field. The card holds the live field during a drag; at the release the stroke's change (F_end - F_start per
  touched brick) is written into the CPU sculpt layer, which is the authority for undo and save.
- **Band admission** (11-11/12). A stroke whose dabs change only nodes near the surface admits only cells whose centre
  lies within (largest reach + the cell's half-diagonal + the batch's summed move) of the surface. Before it, a 60 mm
  Smooth and a 45 mm Inflate under mirroring admitted about 378,000 rows and ran the card out of memory. An 11-11
  attempt at a band-only write-back failed: the CPU kept stale deep values whose sign later went wrong (flakes), and it
  was withdrawn.
- **Memory.** The sculpt pool cost 5.6-6.4 KB a brick (delta 2,048 B plus mask, direction and region arrays that are
  almost always zero) and grew by doubling with a transient copy: 3.3 GB after five body strokes. On 11-12 08:44
  (`0fb997e4`) growth became copy-free (address space reserved once, committed as needed) and the side arrays cost
  nothing until written: 3,332 -> 625 MB, peak 17.7 -> 13.8 GB. The card atlas is 1.17 GB at open and 2.35 GB after
  body strokes; a dense host cell-to-row map takes 1,063 MB (step 3 of section 7 removes it). The app holds about
  5.9 GB of private memory and 2.7 GB of card memory with the model open.
- **Undo journal** (11-11 13:49, `e788e94c`). Each step's undo records go to a per-session file once final; RAM keeps
  the newest 512 MB; a save carries up to 2 GB of the newest steps; opening rebuilds the undo stack from them with no
  replay. Measured over 30.6 minutes and 544 card strokes: ten reopens of 1.4-3.2 s, hash-verified, with up to 600
  undo steps attached, undo and redo exact, no monotonic memory growth.
- **Save** (chunked save 11-08, `af613975`; interrupted-swap recovery 11-11; never cut short 11-12 19:01, `06464ae8`).
  A project is a folder of sculpt chunks (only changed chunks re-compressed, the rest hard-linked from the previous
  save), stroke results, the undo journal and `project.json` with a state hash last. It is written to `<name>.tmp` and
  swapped in by rename, retried while the operating system holds a handle, with the previous save restored if the swap
  cannot finish. Opening recomputes the hash and refuses a mismatch. **A failure seen in real use:** on 11-12 a
  save followed at once by quitting was killed with the process (the writer ran on a daemon thread), leaving no
  project file and no automatic save; the work was recovered from the temporary folder. Since then a save is counted
  as pending from request to the writer's end, quitting waits for it, and a failed write keeps the window and the
  automatic save. After: the close waited 1.4 s and the project reopened with the same hash.
- **Checks kept in the tool.** A debug check compares every card brick with the CPU field at its own nodes (tolerance
  50 um): clean after each of 12 repeated Smooth strokes on an eyelid. Those 12 strokes lower a 1.54 mm bulge to
  0.45-0.49 mm and stall; none raises it again. The report from hand use that a smoothed eyelid bulge reappeared on
  screen is therefore **not reproduced and not explained** yet; the working hypothesis (an edited brick losing its atlas row,
  so the store's original shape is drawn there) is untested. The report of lag growing after 5-10 minutes is also open:
  a 16-stroke run showed none, and the longer run was not finished.

### 7. The scaling study: what a stroke costs and where the store goes next

*Study run 11-12 07:45-09:50 (`353e664e`, `95a00abb`, `9e442f18`). Evidence: measured on named cases; steps 3-4 are
proposed.*

**Method.** Every stage of a stroke was wrapped with a timer in the real window (offscreen, card brushes and tracer,
scripted drags of 0.5-1 s, two or three runs each, other jobs beside it) and marked by what its cost grows with: the
brush's footprint, the pixels changed, or what is loaded.

**Findings.**
- The same strokes on the same 120 mm sphere, as the scene grew from 1 object to 32 objects to one more object carrying
  212,000 sculpt bricks: Draw r 6 mm fell from 25-26 fps to 11-12 to 1.6-1.9; Smooth r 40 from 17-20 to 6-11 to 0-2.
  The bricks written were identical at every load.
- At the largest load, load-bound bookkeeping took 0.6-0.9 s of each stroke and the brush's own work 0.1-0.3 s. The
  load-bound stages: a checksum of every sculpt brick to find what changed (about 280 ms a call, twice a stroke), a
  scene-wide index re-flattened, an object's whole tile tables re-uploaded after any admission (20-36 ms a call on the
  human model, 0.5-1.4 s on a body stroke), an object's whole key dictionary scanned (160-460 ms), growth by copying.
- They come from one structural fact: two authorities (the card during a drag, the CPU layer after), kept consistent
  by scanning, plus dense per-object maps.
- Footprint costs are volumetric where they could follow area: on the human model a body Flatten r 40 wrote 90,486
  bricks of which 82 % lay more than 4 mm from the surface.
- On a small stroke (eyelid Smooth r 4, about 400 bricks) the dab arithmetic is about 1 ms a frame; the rest is some
  15 host steps of 1-10 ms each.

**Prototype** (a study-only store, not the app: no tracer, guard, re-distance or colour). 8^3 fp16 bricks in fixed
16 MB pages (growth adds a page, nothing is copied); one open-addressing, linear-probing hash on the card mapping
(object, brick coordinate) -> slot; values only within 4 cells of the surface; brick borders read through the hash, so
no shared face copies exist; undo by copy-on-write per brick at its first touch in a stroke. Measured: 0.27-0.34 ms a
dab with 1 object and 31,000 bricks, 0.21-0.30 ms with 66 objects and 788,000 bricks; a forehead Smooth r 40 writing
10,378 bricks in 9-15 ms with a 4-7 ms copy to the host; undo exact on 9 of 9 strokes; Draw matches a dense reference
to one fp16 step; node-to-node steps across brick faces match those inside bricks; 75 pages added at 9 ms each. A
separate measurement indexed 424,000 bricks in 12.6 MB against the dense map's 1,065 MB, at 1.3 times the lookup time.
Known defects: it leaks the slots of undone new bricks, and it leaves open the tracer on a band store, large moves and
save.

**The four-step recommendation.**
1. Local bookkeeping in today's design: tables updated per changed entry, no whole-store sync at the press, the release
   sync from the stroke's own slots, snapshot and index only for what changed.
2. The whole turn on the card: plan, admission, dab, guard, re-distance and collision as kernels back to back with one
   host wait a turn, the brick lookup kept in one device function so step 3 can swap it.
3. The store's next shape: one card-side hash over paged pools, the card as the live copy with copy-on-write undo and
   one copy to the host per stroke, fine bricks only in the surface band and a coarse signed-distance lattice beyond
   it (the sign of unstored space comes from that distance, never from per-tile tags).
4. For worlds: chunked residency, brush levels for body-scale strokes, shared bricks for repeated objects, edit layers.

**Decision** (11-12 09:27): the full path; steps 1-2 now, step 3 right after step 2.

**Status at the time of writing.** Parts of step 1 landed 11-12 10:47 (`94ad156d`, `655c0fc2`): a body Flatten's table
time 1,359 -> 21 ms; mirror sync 266 -> 0 ms a body stroke; torso Draw 14.6 -> 23.6 fps; body Smooth 2.5 -> 5.8 fps,
with every card table equal to a full upload after every stroke. Two trial patches measured in the study: index
rebuild 31-38 -> 2 ms; body-stroke sync roughly halved. Steps 2 and 3 are not started. The study names step 3's
research risks plainly: empty-space stepping and sign consistency once deep values are not stored finely, and carrying
fine detail through body-scale moves. Neither has a prototype.

### 8. Benchmarks

**Brush scorecard** (11-11 11:54-12:06, `d7a4067a`). *Evidence: reproduced under a frozen configuration* (the tree of
commit `01bc419d` extracted to its own folder, two full runs). Fixed scripted cases, so the driver's skill is out of
it: a 500 mm slab, a bump on it, and a 240 mm bust on a 1 mm lattice; a straight drag of 4 radii in 1.0 s at 120 mouse
events a second; radii 5 / 20 / 60 mm at three strengths; each stroke undone after measuring (undo error 0.0 um in
every case). Results: 57-62 fps at r 5-20, 26-39 fps at r 60, 12-16 fps at r 80 with mirroring (Grab 5 fps). Draw,
Carve, Inflate and Grab matched their intended form; Smooth removed too little at large sizes (0.24-0.98 % a pass at
60 mm); Flatten, Clay and Pinch were scored broken (Flatten made a bump taller). Deposits repeat within about 1 % run
to run; worst frame gaps vary up to 2x with machine load. This scorecard predates the rebuild of Clay, Flatten and
Smooth that began on 11-11 at 13:30 (Part 5); it is the before-picture, not their present state.

**Full-session run on the human model** (11-12 02:44-03:21, live code frozen at one commit, `9f93e325`). *Evidence:
reproduced under a frozen configuration* for the first fault below (two fresh runs failed identically); the rest
*measured on named cases*.
- The first whole-body stroke (Smooth r 60, mirrored) ran the card out of memory at 6.9 GB and the engine then silently
  dropped the stroke's release. After three rounds of fixes, 20 of 21 body strokes complete, each with one exact undo
  step; host memory still climbed to 18-24 GB until the store change of section 6.
- Targets missed: 30 fps while a stroke moves was missed on every sculpt stroke (face-level brushes 6.7-24 fps, eyelid
  r 2.5-4 mm 2.6-14.8, lip corners 3.7-13.5, torso r 25-30 0-1.2); a next-press wait of 100 ms or less was missed on
  23 of 49.
- Targets met: a 15-minute face soak of 296 strokes kept memory flat after 4 minutes (card 6.40 GB throughout); save
  5.7 s for 339 chunks; reopen 3.0 s with an identical state hash, 4.2 s cold in a new process.
- The CPU fallback broke the 16 GB budget with one torso stroke (minutes of stall, 9-13 GB); after a band-only fix
  small strokes were 15-50x faster and a torso stroke still took about 50 s.

### 9. Parts being redesigned around named patents

On 11-12 the independent claims of the patents below were read beside our mechanisms, to the
standard that each concern name the claim element behind it and each design-around name the element that is absent.
These are flags from our own review, not conclusions that any claim is or is not met. **As of this text none of the four redesigns has landed; the
affected code is held out of any published code until it does.**

| our part | patent (status as read 11-12) | the element of concern | the planned way round, and what it removes |
|---|---|---|---|
| The capsule primitive (a sphere's distance at the sample's clamped projection onto a segment) | MERL US 8,265,909 (active, adjusted to 2031-02-23), claim 1 | the patent concerns the distance of a shape swept along a path, with an optimal path parameter found continuously per sample; flagged in our review; held out of the code pending redesign | not yet designed; brushes already apply separate fixed-centre dabs, which lack the continuous path parameter |
| Grab's brick admission (brick centres projected onto the drag segment with a clamped parameter) | same | same | not yet designed |
| The card tracer's tile skip (one entry per 32 mm tile per level saying slot or empty; empty tiles skipped to their exit) | Movidius US 11,680,803 and US 11,222,459 (active, about 2037), all seven independent claims | a first entry whose values say each coarse voxel is empty or occupied, with finer entries beneath, used in rendering; flagged in our review; held out of the code pending redesign | direction of the design-around: skip empty space by a stored distance value (the tracer already holds a distance-to-nearest-brick far field) instead of an empty tag |
| The band store (`core/bandstore.py`, 11-08), whose tile tables carry outside / inside tags | same | same | keep the sign of unstored space in a coarse signed distance, as in step 3 |

Open on our side: whether a coarse signed-distance lattice could itself be construed as the claimed set of occupancy
bits is recorded as a grey zone; validity against earlier published work was not assessed. Other readings carried from
the same check, as constraints on step 3: nothing is paged onto the card on a lookup miss and the pool stays the data's
home (NVIDIA US 9,235,512); no per-brick lists of distance functions or edit primitives (MERL US 9,122,270 and
US 8,010,328); the coarse level is its own lattice, never built by merging fine bricks into parents in the same hash
(JD.com US 10,706,611). A screening search found no patent on a hash over paged pools, copy-on-write brick undo or
discrete-dab card sculpting; that is a screen, to be repeated, and not a clearance. Two expired families (Caltech
US 7,542,036 and US 9,430,870, level-set editing) appear to be past full term; not confirmed.

### 10. Published work this part learned from

Only sources the project records name. Papers and talks were read; no code from any of them is in the store.

| source | what it taught | code licence as recorded |
|---|---|---|
| Frisken, Perry, Rockwood, Jones, *Adaptively Sampled Distance Fields*, SIGGRAPH 2000; Perry and Frisken, *Kizamu*, 2001 | resolution where detail is; local regeneration after an edit | papers only (the 2000 paper's PDF was not re-read during the study) |
| Museth, *VDB*, ACM TOG 2013; OpenVDB | shallow fixed-depth tree, constant tiles, size-independent insert and delete | Apache-2.0 (MPL-2.0 before v12) |
| Museth, *NanoVDB*, 2021; Sifakis et al. 2026; Williams et al., *fVDB*, 2024 | card read-mostly layouts; topology changes rebuild the grid | Apache-2.0 |
| Hoetzlein, *GVDB*, HPG 2016 | card brick atlas with aprons and incremental insert | Apache-2.0 |
| Niessner et al., *Voxel Hashing*, ACM TOG 2013; ASH, stdgpu, nvblox | one card hash of brick coordinates over a fixed heap | original code CC BY-NC-SA (study only); stdgpu and nvblox Apache-2.0, Open3D MIT |
| Whitaker 1998; Museth, Breen, Whitaker, Barr 2002; Adalsteinsson and Sethian 1995 | narrow-band level sets and editing operators | papers only |
| Evans, *Learning from Failure* (Dreams), 2015; Aaltonen, Claybook, GDC 2018 | edit lists culled per cell; a thin band rewritten per edit | talks only, proprietary |
| Careil, Billeter, Eisemann, *HashDAG*, 2020 | copy-on-write edits with old roots kept for undo; pages on demand | MIT |
| Blender sculpt (2024 refactor notes) | per-node undo and dirty bits; brush rules used as the benchmark's peer | GPL (read as a study; principles only) |
| Hart, *Sphere Tracing*, 1996 | stepping by the distance value | paper |

General techniques used without a single source: trilinear interpolation, k-d tree nearest-point search, SHA-256
content hashing, hard links, atomic replace-by-rename.

---

# Part 5. Brushes, the toolkit framework, sculpting into contact, and the shell

This part describes every sculpting mechanism built for Field Studio up to 2026-11-12, how each one changes the
field, what it cost per stroke, where it failed, and what replaced it. Dates are internal adoption times taken from
the project's version history (Pacific time). They record when a mechanism entered the code or the written record;
they are not the publication date of this document, which is the release date stated on the release itself.

Each mechanism carries exactly one evidence state:

- **proposed**: designed and written down, not implemented.
- **implemented**: in code (the live tool or a named working copy), not measured beyond smoke checks.
- **measured on named cases**: run on the cases named beside the numbers.
- **reproduced under a frozen configuration**: the same result obtained again from a fixed code state and inputs.
- **accepted in the workflow**: used as part of the working process.

The most important fact comes first. On 2026-11-12 at about 17:20 a hands-on sculpting test with the tool could
not block out even a character's head, and the toolkit work was stopped. The scripted head test had been
reaching all its stages, though never its marks (section 12). Much of what follows is therefore a record of mechanisms that work by their own numbers and still did not
give a sculptor a usable tool, plus the redesign that came out of that. **The hands-on sculpt test of the
rebuilt toolkit is still to come.** Nothing here should be read as a claim that the toolkit is finished.

### 1. Setting, machine and how to read the numbers

Every object is a signed-distance field F (F < 0 inside): an analytic base tree plus a sculpt layer of values added
on a sparse lattice of 8 x 8 x 8-node bricks at spacing h, stored only near surfaces. Brushes edit the sculpt layer.
Test forms use h = 1 mm (a 240 mm bust sphere); the human model uses h = 0.25 mm.

Notation: c is a dab centre on the surface, n the surface normal there, r the brush radius, s the strength,
w(u) the falloff with u = distance / r (default 1 - smoothstep: 100 / 84 / 50 / 16 / 0 % at u = 0 / 0.25 / 0.5 /
0.75 / 1), A the dab's amount, v a drag vector.

Machine: one Windows PC with 32 GB of RAM and one 8 GB RTX 5050. "The card" below means that GPU; "the CPU path" is
the fallback that runs the same rules in compiled Python (numba), 4 threads. **CPU-load caveat:** the machine was
shared by several working sessions at once, often at 90-100 % CPU. Frame rates swung by 2-3 fps between identical
runs, and some numbers were taken under that load. Where a before/after pair was interleaved in one run it is
reliable; a lone frame rate is an indication only. Card strokes are not bit-reproducible run to run (dab batching
follows frame timing); card-versus-CPU agreement is proven by a full field comparison, not by repeat runs.

### 2. The stroke pipeline

**CPU path** (in the repository's first commit, 2026-11-05; band limit 2026-11-12; measured on named cases).
1. The cursor ray is traced to the first surface. A press off the form takes the nearest surface within the brush
   radius along the ray, else the stroke starts when the drag first reaches the form. A brush press never orbits the
   view (2026-11-11).
2. The surface path is resampled to one dab every spacing x r. A frame that ends between dabs lays a partial dab whose
   share f of a spacing step scales the dab's amount linearly (2026-11-11). Before that fix the share was folded into
   strength, and two half dabs deposited 27 % more than one whole dab, so results followed frame timing.
3. Per dab: the nodes of the dab's ball are evaluated, the brush rule (section 3) gives a change per node, the
   change is multiplied by (1 - mask), and the gates run in order: the sheet gate (section 5), the keep-apart guard
   (section 4), the collider clamp (section 6). The change is written and a before-image kept for undo.
4. At release the touched bricks are re-distanced (|grad F| restored to about 1) by seeding at the surface and
   sweeping an Eikonal solve outward.

**Card path** (2026-11-08; rebuilt 2026-11-11 and 11-12; measured on named cases).
1. All dabs of one frame form a batch. A plan step admits the bricks the batch can change and uploads their rows
   (a brick plus its one-node border, 9^3 values).
2. One kernel computes every dab of the batch per node. Drag dabs of a frame are merged while they are closer than
   0.25 r.
3. The keep-apart guard runs once per batch on the net change; the collider and the sheet labels run on the card.
4. A live re-distance runs after every batch: seeds at the surface nodes, then Godunov Jacobi sweeps, with the
   surface layer (nodes whose 26-neighbourhood changes sign) frozen. The borders are refreshed from their owner
   bricks. The first version (a jump-flood nearest-foot distance that rewrote the surface nodes) left lattice-aligned
   ribs over the cranium and was replaced on 2026-11-11 16:21.
5. At release every changed brick is written back to the CPU store as one undo step. A version that skipped deep
   bricks (2026-11-11 ~15:00) left the CPU store behind the card and produced whole bricks of the wrong sign; it was
   withdrawn the same day at 17:40.

The Python interpreter's thread switch interval is set to 0.5 ms at engine start (2026-11-11 17:11). On the head
scene this alone took a mirrored Smooth stroke from 15.0 to 29.3 fps median, because the render thread had been
waiting for the interpreter lock.

### 3. The brushes: how each changes the field

All in the live tool unless stated. "Resample" means the new value at node x is the old field read at a displaced
point, which moves the surface without leaving a step in F.

| Brush | Rule at node x (w = w(|x - c| / r)) | Adopted | Evidence state |
|---|---|---|---|
| Draw | F'(x) = F(x - n A w): the surface rises A w along the single hit normal | first commit 11-05 | measured on named cases |
| Carve | F'(x) = F(x + n A w); Ctrl+Draw | 11-05 | measured on named cases |
| Inflate | F'(x) = F(x) - A w: a true offset along each point's own normal | 11-05 | measured on named cases |
| Clay | d = signed height of x above the area plane; F' = F + w (min(F, d - A) - F), at most A per dab, so a hollow fills over passes | area plane 11-11 13:30 | measured on named cases |
| Flatten | F' = F + k w (d - F), clamped to +-A per dab; two-sided, toward the area plane | 11-11 13:30 | measured on named cases |
| Smooth | section 3.1 | rebuilt 11-11 13:30 to 21:50 | measured on named cases |
| Crease | t = offset of x from the stroke line within the tangent plane; F'(x) = F(x + k w t) + A w^3: the surface is drawn toward the line and cut in | toward the line 11-11 13:59 | measured on named cases |
| Pinch | as Crease without the cut; factor bounded at 0.4 so the map's Jacobian stays above 0.5. Still tears (85 columns in the scorecard) | bound 11-11 13:30 | measured on named cases |
| Grab | F'(x) = F(x - v w), composed dab after dab. Being replaced (section 8) | 11-05 | measured on named cases |
| Snake Hook | Grab with a falloff flat to 0.8 r and a cosine edge. Being replaced | 11-11 22:59 | measured on named cases |
| Stretch | with a = v / |v| and kw = (|v| / r) w: F'(x) = F(x - a ((x - c) . a) kw / (1 + kw)), a scale along the drag axis about the pressed point. Withdrawn (section 8) | 11-05 | measured on named cases |
| Mask | paints a 0-1 mask per node; every brush's change is multiplied by (1 - mask) | 11-05; on the card 11-11 15:03 | measured on named cases |
| Region | paints a 16-bit region id per node; strokes can skip hidden regions or act in one region only | 11-11 22:27 | measured on named cases |

Amounts: displacement brushes move 0.2 x spacing x s x r per dab, so the same screen stroke gives the same-looking
shape at any radius. Smooth, Flatten, Pinch and Crease's pull use a per-dab factor k = k_max (1 - exp(-k0 s / k_max)),
monotone in strength and saturating, never clamped (2026-11-11 13:59, under the decision that every tool's
effect rise smoothly with strength). Measured on the CPU path at s = 0.1 / 0.25 / 0.5 / 1 / 1.5 / 2: Smooth removes
0.14 / 0.30 / 0.44 / 0.50 / 0.53 / 0.53 of a 12 mm bump per pass; Crease depth 0.10 / 0.24 / 0.45 / 0.82 / 1.12 /
1.34 mm. Ctrl inverts every surface brush (measured on all seven; Ctrl+Clay digs 1.44 mm).

**Area plane** (Clay, Flatten): the falloff-weighted mean position and normal of surface samples on a sunflower disc
of radius 0.8 r oriented by the hit normal, each projected onto F by one Newton step.

#### 3.1 Smooth and its fit

The first Smooth averaged F with its six neighbours at 2h whatever the radius: it removed 4 % of a 40 mm bump per
pass, and in the first head test four passes took lumps from 4.49 to 4.23 mm RMS. Rebuilt 2026-11-11:

1. Only nodes within 6h of the surface are written.
2. Each such node projects to the surface and samples two rings of 8 points in its own tangent plane, at 0.3 r and
   0.6 r. Each sample's height along the node's normal comes from one Newton step and is clamped to half its ring
   radius.
3. Opposite samples are paired. A pair's weight falls where a sample lies on a sheet facing away (the "robust2"
   ramp, 2026-11-11 21:50), so a facing surface is never drawn in.
4. A quadric is fitted to the heights. The node moves toward the fitted surface, so a sphere or cylinder at the
   brush's scale gives zero and the form's own curvature stays.
5. A lattice-scale term (the six-neighbour mean at 2h minus F) is added; without it the fit left cell-scale speckle.
6. The change is s w (fit + lattice term), capped at 2h per dab.

Measured (CPU path, one pass, r equal to the bump width W): W 4 / 12 / 40 mm removed 88 / 51 / 52 %, surrounding
form moved 0.002 mm or less. On the card at s = 1: 41-45 % per pass at r 20-60 mm; after the partial-dab fix removed
a frame-timing inflation, 34.5 / 30.4 % at r 20 / 60 and a repeat spread of 0.49 % (was 3.44 %).

Failures on record:
- **Chin bands.** Smooth at r 70, s 1 over the test head's chin left textured bands and pits. Cause: at r 70 the
  outer ring lies 42 mm out, past the chin's turn, so it is no longer a height over the tangent plane. Rings at
  0.2 / 0.4 r were clean but weaken Smooth elsewhere. An adaptive three-ring fit passed CPU invariants and on the
  card traded the pits for a new V-shaped crest across the chin. Not landed. Open.
- **Fixed points.** A form about the brush's own size grows under Smooth (a sphere of R 40 under r 40: +1.27 mm over
  8 passes), from the parabola fit's bias against a real sphere. A circle fit removed the growth and exposed the
  lattice term's own shrink (-1.29 mm). Open.
- **Volume.** Ten passes at r 30, s 0.5 on a 40 mm bump: height -53 %, volume -20.7 %. Smooth shrinks.
- **CPU-path stall.** On the CPU path a 40 mm bump stalls at about 63 % remaining, because Smooth compresses the
  field at the edge of its 6h band and the release re-distance skips it. The card path does not stall.
- Hands-on use (2026-11-12 about 18:20): Smooth worked fairly well, but a smoothed lid bulge
  came back after further strokes, and some surfaces keep small tear or grain marks.

### 4. Shared stroke rules

- **Mirror** (2026-11-11 17:46 to 19:35; measured on named cases). A dab and its mirror partner are both applied.
  Where their balls overlap, each node's weight is scaled by (w^2 + w2^2) / (w + w2)^2, so the pair sums to the
  contraharmonic mean (w^2 + w2^2) / (w + w2): smooth, no doubling on the midline, equal to w where one acts alone.
  Two earlier versions failed: snapping near-midline dabs onto the plane zeroed sideways drags (a widening Grab did
  nothing), and max(w, w2) left a sharp midline seam. As landed: a dab within 0.8 r of the plane is still snapped
  to one centred dab for the mirror-symmetric part of its effect (so a nose stays single), while a Grab or Stretch
  drag's component across the plane is carried by a feathered pair at the original centres; Smooth always uses the
  feathered pair. CPU check: sideways widening 4.47 / 4.46 mm
  on the two sides, 0.0 on the midline; midline Draw 11.44 mm against 11.18 single.
- **Keep-apart guard** (2026-11-10; measured on named cases). A node may change sign only if the inside components
  (26-connected) and outside components (6-connected) of its neighbourhood stay as they were, so a guarded stroke does not join,
  film over or tear surfaces (scope: the guard is off for Grab and Stretch, and Pinch still tears). A refused node keeps its sign at +-0.05 h. It is off for Grab and Stretch by default
  since 2026-11-11, because on large moves its refusals left specks and pockets (6 inside parts and 31 pockets in a
  neck with it on, 1 and 0 with it off).
- **Stroke-end topology check** (2026-11-11 19:35; off by default; evidence state: measured on named cases). Parts, tunnels and cavities are
  counted from node signs before and after a stroke (b0 by 26-connectivity, b2 by 6-connectivity, b1 from the Euler
  characteristic). A pocket the stroke made is filled; a closed opening is reported with "keep it / undo the
  stroke". Detection was measured on the card on a slit and a torus with the guard switched off.
- **Band-only dabs** (card 2026-11-08 / 11-11; CPU 2026-11-12 about 12:50; measured on named cases). Nodes farther from
  the surface than 1.6 x the dab's move + 6h keep their values. The CPU version first skipped on both sides and left
  a 28.6 um notch where a growing surface crossed a brick row; the fix is one-sided (a growing dab skips deep inside
  only, a cutting dab deep outside only): 5-8 um, identical to no band. On the card the same per-node skip is tied to
  flat brick-shaped plateaus, fins and grid steps when a surface grows fast (section 7). Not resolved.
- **Targeting** (2026-11-08; evidence state: measured on named cases; superseded by section 5). "Only the surface
  the brush faces" and "connected only" switches. On the thin-wall test they hold at r 6 and fail above it.

### 5. The sheet under the brush: nothing behind the worked surface moves

The requirement (2026-11-12 about 12:50): a tool acts only on the surface it is working on and on nothing behind it,
unless a setting or toggle says otherwise. The cases named: an eyelid over the eye, the two lips, a bra's two faces
over the skin beneath.

**The fault, measured** (CPU, 2026-11-12 13:00). One object: a 4 mm wall with a ball 2 mm behind it. Strokes on the
wall's front at r 6 / 12 / 20 mm moved the wall's back face by 0.30 / 2.0 / 3.9 mm (Draw), up to 3.5 mm (Clay), and
the ball behind by up to 3.5 mm (Draw) and 5.2 mm (Clay, closing the gap). Even Smooth on the flat front moved the
ball 0.52 mm at r 20. The older targeting switches held only at r 6.

**Mechanism** (landed 2026-11-12 14:40 as an opt-in switch "Protect surfaces behind"; on by default from 15:48 by
decision; evidence state: measured on named cases).
1. Surface cells are lattice cells whose corners have both signs.
2. Seeds are the surface cells under the dab centre, or the nearest surface cell along the dab's axis.
3. The seeds (taken from the 3 x 3 x 3 block of cells around the press, keeping only cells whose own normal faces
   the brush) are flooded from cell to cell across shared faces that the surface crosses, at most 1.6 (r + 2h) / h
   steps. The flood does not enter a cell whose normal has turned more than 100 degrees from the brush normal, which
   stops it rounding a lid's margin onto its back face.
4. Each node in reach takes the label of the surface cell at its own foot point x - F grad F / |grad F|; deep nodes
   take their brick's label. Labels are kept from the first time a dab sees a node, for the whole stroke.
5. A node's change is kept only if its label is the flooded sheet. An undecided band node joins the sheet unless one
   of its 26 neighbours belongs to another sheet (this rule removed specks in the test head's eye sockets).
6. The release re-distance restores every node not on the worked sheet.

Measured: thin wall, all tools, r 6-20: back face and ball 0. A lid over an eye (0.25 mm lattice, 0.5 mm gap) in the
real window on the card: without the gate Smooth r 15 / 20 moved the eye 1.7-2.4 mm and closed the gap on about 450
rays; with it 0 / 0 / 0 for Smooth r 15 / 20 and Draw / Inflate r 8. Two 3.5 mm plates 1 mm apart (lips): Draw,
Inflate, Clay, Crease, Pinch and Smooth move the facing surface 0.0 mm. Crease in a thin wall stops at the wall's
middle by design: a wall thins stroke by stroke and was not punched through in the cases tested.

Limits and failures:
- **It does not cover Grab, Stretch or Snake Hook.** Measured on the card: Grab moves the facing lip across a 1 mm
  gap by 0.52 mm and a 3 mm plate's back face by 0.72 mm; Snake Hook moves all four surfaces about 0.99 mm.
- On the human model the upper lid's field was fused with an old interior copy of the eye, so Smooth r 15 / 20 over
  it still moved that copy 0.16-0.20 mm (1.0-3.8 mm without the gate). That is a model defect, not the gate's.
- It blocks contact: a surface grown toward another surface of the same object stops at the gap's midpoint (section 7).
- Cost: Smooth r 15 fell from 23 to 15-18 fps with the labelling on the card (3-6 fps with it on the host, which was
  not shipped); a faster seed search brought it to 19.5 fps. Draw r 8 was unchanged. CPU test strokes: +21 %.

**Reach-through** (per-stroke setting implemented 2026-11-12; evidence state: implemented). Setting the cone to 180
degrees or more makes the flood cover the whole connected piece, so a lid's front and back are worked together while
the eye, a third surface, is left alone. It has no control in the window and its test cases were not run.
"This part" as a general scope for every tool is **proposed**.

### 6. Collision of brushes with locked and partner parts; the gap setting

(CPU 2026-11-09 00:19, card 00:34; partner parts 2026-11-11 18:06; evidence state: measured on named cases.)

Colliders for a stroke are: every locked protected part other than the target; any objects named as contacts; and
the target's partner parts paired with it as "touch" or "slide" (pairs marked "bond" or "pass" do not collide).
After each dab (CPU) or batch (card), on the nodes the edit changed:

    B(x)  = gap - min over the colliders of F_part(x)
    F'(x) = max(F_new(x), min(F_start(x), B(x)))     where B(x) > -2h;  F'(x) = F_new(x) elsewhere

So an edit never moves the editable surface into a part dilated by the gap; a node that already overlapped is never
pushed deeper; an untouched node keeps its bits. The rule also holds a band of two cells outside the part plus gap,
so the field next to the contact follows B and the surface lands at the gap rather than a cell short of it. The gap
is a brush setting (default 0.3 mm; 0 gives real contact, measured 0.000 mm on axis on 2026-11-09). A lid shaped
toward the eye slides over the globe at the gap instead of sinking in. Brush picks stop at locked parts instead of
passing through them (2026-11-12 15:48). Cost on the human model: 4.4-6.5 ms of a lid stroke's frame; one CPU case
where the bound was evaluated brick by brick took 10.5 s of a 13.3 s stroke and was cut to about 0.4 s by evaluating
all near bricks in one call (2026-11-12 about 01:50, field bit-identical).

### 7. Sculpting surfaces into contact

The aim: bring two surfaces to touching while sculpting (a lid onto the eye, a fold onto a neighbouring
part), smoothly, with no typed values.

**Swell to contact** (first form 2026-11-09 01:15; eased formula 2026-11-12; on the card and the default Swell from
15:48; evidence state: measured on named cases). It is Inflate times a factor k per node, then the collider clamp:

    gs   = max(T - gap + F, 0)            distance from the surface here to the contact (T = the contacts' distance)
    near = 1 within r of a contact, falling linearly to 0 at 2 r
    ease = 1 - exp(-(gs + h/2) / E)       E = max(0.25 r, 2h): growth slows over the last E of the gap
    face = (1 + cos) / 2                  cos between the surface normal and the direction to the contact
    k    = 1 + near (gain x ease x face - 1),  gain = 3

Far from any contact k = 1 (plain Inflate). The h/2 lets the collider finish the touch. Measured: a ball of r 20 mm
under a locked ball of r 4 mm, 3 mm gap, strokes of r 4, s 0.5: 2.99 -> 0.10 mm in 5 strokes, far side unmoved, no
overlap. A lid's margin came down onto the eye in 3 strokes with no outward puff. A skin fold on the human model
facing a locked part across 3.55 mm (median of 135 rays): 3.55 -> 0.3 mm in 6 strokes at r 8, 36-38 rays touching.
Rejected variants: growth toward the nearest contact point made a flat mesa with horns; a 60-90 degree facing ramp
grew a column with cliffs; a plain cosine cut growth side-on and left a shelf.

**Settle onto** (2026-11-09 01:15; evidence state: measured on named cases). Each node in the dab reads the field
from a point farther from the nearest contact along that contact's gradient g: F'(x) = F(x + g A w), with
A = s r / 2, then the clamp. One move direction per dab keeps a sheet's thickness (2026-11-09 02:14).

**Settle by column** (2026-11-12 14:17; evidence state: measured on named cases). Each column of nodes across the
move direction moves by min(A w, its own clearance to the part), so one touching point no longer stops the whole dab.

**A locked part owns its volume** (2026-11-12, working copy, not landed; evidence state: measured on named cases).
F := max(F, -P) over the part's box, where P is the part's own field (a CSG difference). On the human model this
took the lid off the interior eye copy (fused lid rays 144 -> 27 and 135 -> 19 of about 277 per eye); the written
version failed a store check for reasons in the writer and was not published.

**Findings that stopped this line** (2026-11-12, real window, card):
1. With "Protect surfaces behind" on, growth toward another surface of the same object stops at the gap's midpoint:
   3.55 -> 1.7 mm, never touching, because nodes nearer the other surface count as another sheet. With it off:
   3.55 -> 0.0 in 3 strokes. A contact tool needs the rule "the worked surface advances until it touches the other
   one, which stays unwritten". Proposed, not built.
2. The card's per-node skip and band-only admission keep values that a fast-growing surface outruns: flat
   brick-shaped plateaus at r 8, a fin, a slit and grid steps after Smooth. All vanish with the skip off, at the same
   speed, but that also removes the admission limit that keeps large strokes within card memory.
3. Settle by column roughens a lid (grainy and pitted after 3 strokes at r 4, s 0.5) and barely improves contact
   (0.65 -> 0.6 mm): each column takes its nearest sample's clearance, so moves are piecewise constant.
4. Swell as a plain balloon closes a fold but rounds the whole neighbouring form into a pillow. Swell to contact
   stalls at about 2.6 mm and corrugates when the locked part's field is not a true distance outside the body.
5. On a fold, where a dab lands and which way it pushes followed the camera, not the surface.
6. The same formula differed between card and CPU (a lid's front puffed up to 0.6 mm on the card, 0.05 mm on the
   CPU). Not explained.

In hands-on use Swell also worked fairly well. The contact tools are held for the rebuild.

### 8. Grab, Stretch and what replaces them

**What failed** (measured 2026-11-12 on a sphere of r 120 mm, h 1 mm, r 40 brush).
- *View dependence.* Grab, Stretch and Snake Hook map the drag onto a plane parallel to the screen. One intended
  pull, 30 mm out along the normal, gave 30.1 / 17.7 / 5.7 / 0.8 mm from views at 0 / 30 / 60 / 80 degrees.
- *Stretch did not work.* It scaled the field about the pressed point itself, so that point moved 0.0 mm from
  every view after a 30-80 mm drag; only the inner ring changed; a wiggle left a sharp ring around a flattened disc
  (seen in hand use as a half-circle cut out of the sphere); it compounded per mouse event (+5.19 against +4.44 mm for one
  drag at two mouse rates).
- *Path dependence.* Drag out, circle, return: Grab left 2.96 mm, Stretch 5.02 mm.
- *Folding.* F(x - v w) folds once the pull exceeds about 0.67 r; Snake Hook showed three surfaces on a probe ray.
- *The ring lied.* Grab moved 42 % at half radius and 7 % at 0.75 r against a drawn 50 % and 16 %.

**The drag constraint taken from the form** (proposed and built in a working copy 2026-11-12 about 19:00, CPU path only;
evidence state: measured on named cases). At the press, with camera forward f and normal n0: m = f - (f . n0) n0,
s = |m|. The drag plane passes through the pressed point c0 with normal m / s: it contains n0 and faces the camera
as much as a plane containing n0 can. Each cursor ray is intersected with it. Within 5 degrees of head-on
(s < sin 5) the plane normal is normalize(b m/s + (1 - b) n0), b = s / sin 5, turning into the tangent plane. With
Ctrl the target is the point on the line c0 + t n0 closest to the cursor ray; near head-on, t is the drag's
component along the camera's up axis. Measured through the real window: 29.9 / 29.9 / 29.9 / 29.4 / 29.9 mm from
0 / 30 / 60 / 80 degrees and above. A first 15-degree blend slid an 80-degree pull 32 mm sideways and was narrowed.
The silent change of operation near head-on is to become a visible switch; open.

**Anchored moves** (same copy, CPU path; evidence state: measured on named cases). The result is a function of the stroke-start field
and the current total drag. On every update each touched node is restored to its stroke-start value and one
deformation is applied. On the card the plan is first-touch brick copies written back by a row-scatter kernel, then
one dab (compiled, never run). Wiggle-and-return residual: 0.00 mm.

**Grab as a flow** (same copy, CPU path; evidence state: measured on named cases). The deformation is the time-1 flow of
V_t(x) = w(|x - c(t)| / r) v with c(t) = c0 + t v. Because w(0) = 1 the pressed point travels with the kernel; a
flow of a Lipschitz field is one-to-one, so in the continuous map nothing folds (in the built version this was measured on probe rays only). F_new(p) = F_0(Phi^-1(p)), with the inverse integrated
backward in N = clamp(ceil(|v| / 0.15 r), 4, 64) midpoint steps over a ball of radius r + |v|/2 centred at c0 + v/2.
Measured: pressed point 29.75 mm for a 30 mm drag; one surface on every probe ray.

**Stretch as an anchored elongation** (same copy; evidence state: implemented; its check was stopped before it ran).
Axis a = v / |v|; pivot P = c0 - D a with D = max(T, r), T the depth of the form behind the press. Points between
the pivot and the pressed end scale by (D + Le) / D with Le = |v| w(rho / r), rho the distance from the axis; beyond
the end the displacement falls linearly to zero over max(2|v|, r). The inverse is closed-form.

**Form-aware Grab** (decision 2026-11-12 about 09:50; evidence state: proposed;
the prototype had not been started in the records read). A Grab whose weight follows the form's own interior, for
example the distance through the material from the pressed point computed from the field itself, or a
connectivity-limited falloff, so the far side of a head or the next finger does not move. It uses no elasticity
equation and no material constants. It replaces the Elastic Grab prototype below.

**Elastic Grab prototype** (2026-11-11 20:09, test copy; parked, never in the window, NOT shipped; evidence state:
measured on named cases). The object's own linear-elastic response solved numerically on a coarse lattice inside the form,
Poisson ratio 0.4, a smooth load over 0.8 r normalised so the pressed point moves by the drag, a soft spring anchor
toward the reach, three conjugate-gradient solves at the press. On a head, a cheek pull added +38 cm3 with the back
of the head moving 0.3 mm, or +53 cm3 with a soft anchor holding the far side, against plain Grab's +74 cm3. It is withdrawn for the patent reason in
section 15. A volume-keeping variant of the flow Grab (velocity as the curl of a compact vector potential) is
proposed.

### 9. The pencil: creases, ridges and cuts drawn on the surface

**Design intent of the current Pencil** (defined 2026-11-13/14, after this write-up; in the code of the 0.2.0 release,
`3389050f`, it is not yet the window's line tool). Pencil is a surface-following precision tool for carving grooves and
building ridges. Its tip defines the local shape of the mark, with controls for width, depth or height, and the
transition into the surrounding surface. The intended interaction is direct: geometry changes beneath the tip as the
cursor moves. Each press-and-drag is one controlled pass. Retracing the same path within that pass does not accumulate
additional depth or height. Releasing and beginning another pass establishes a new surface reference, allowing
successive passes to deepen a groove or build a ridge deliberately. Pencil supports fine lines, creases, and broader
cuts through the same tool rather than requiring separate carving and creasing tools. The two earlier pencils below
are the record of how the project got there.

**The shelf's pencil** (first commit 2026-11-05; measured on named cases; found faulty). A line along the upper lip
1.2 mm above the mouth slit cut the lower lip across the slit by up to 226 um at 811 points. In the window it showed
nothing while drawing, then 3-4 minutes of work after release.

**The rebuilt pencil** (2026-11-12 about 15:50, working copy only, never landed; evidence state: measured on named
cases). Line wobble at most 0.13 mm (0.27 mm median before the correction); the line drawn with progressive on-screen updates equal to the line laid at once (0.0 um);
valley 1.497 of 1.5 mm; fine ridge 0.6 x 0.15 mm gave 0.141; a facing sphere across 0.75 mm moved 0.0 um; the lip case
0.0 um on the facing lip. Limits: a V cut closes where narrower than a cell, so cuts need a U profile; the finest line
is about 0.5 mm wide at h = 0.25 mm. **Not reached:** 4-13 fps while drawing with hitches to 527 ms, against the
brushes' 20 fps and more. The pencil work was stopped with the rest of the toolkit. The pencil in hand use that evening
was still the shelf's old one, and it fell short: it creased poorly, covered too broad an area and damaged most of
the surfaces around the mark.

### 10. The toolkit framework: ten principles and the experience harness

Written 2026-11-12 ~17:55 after a study of Blender's sculpt tools; evidence state: proposed. The build order that
follows from it (the harness first, then Grab on the common machinery) was agreed at about 18:40.

1. **Full 3D.** A tool gives the same result from any view; the camera only says where the hand points.
2. **What you see is what moves.** The ring is the true reach and its inner ring the true 50 % weight.
3. **A stroke depends only on the hand's path over the stroke-start form**, not on mouse rate, frame rate, batching,
   CPU or card.
4. **Deformations are smooth and invertible.** No folds, tears or ghost surfaces.
5. **Scope is explicit and three-dimensional**: this surface (default), this part, the connected form, the object.
6. **Size is free**, from sub-millimetre to the whole object.
7. **Big form and fine detail are separate but linked.**
8. **Strength, pressure and modifier keys mean the same in every tool.**
9. **Immediacy**: the window never stops answering.
10. **A tool is proven by the experience it gives**, not only by field numbers.

Principle 1 goes beyond Blender, whose Grab, Snake Hook, Elastic Deform, Pose and transform tools also work in the
view plane; sculptors there choose the view to choose the direction.

**Harness** (2026-11-12 18:49; measured on named cases). The real window is driven offscreen by operating-system
mouse events and results are probed in 3D on the field. T1 same intent from many views; T2 the ring; T3 wiggle and
return; T4 mouse rate 30 / 120 / 480 Hz; T5 nothing behind moves; T6 size; T7 Smooth keeps the form; T8 modifiers.
Baseline on the shipped tools: surface brushes view-independent within 3-4 %; Draw 11 % stronger at 480 Hz than at
30 Hz; window stalls up to 111 ms. The pass mark is a person blocking out a head from a sphere.

**Window freeze** (2026-11-12 ~17:49). The window in hands-on use stopped answering for about five minutes. The run's log
was overwritten by the relaunch, so the work that ran is unknown. A mechanism was measured: a compiled kernel that
keeps the interpreter lock stops the window thread for its whole run (a 93 ms kernel gave a 95 ms stall; 11 ms when
it releases the lock). Landed 18:06: a per-session log that a relaunch never erases, and a watchdog (a 250 ms timer
re-arming a native 2 s traceback dump) that writes every thread's stack while the window is stalled.

### 11. Per-stroke costs and frame rates

Targets set for the lane: 30 fps or more during every stroke at r 1-80 mm; 100 ms or less from release to the next
stroke. They are met on simple forms at small and medium radii and not met on the human model's body.

| Case (card unless said) | Result |
|---|---|
| First head test, 2026-11-11 10:47, before the rebuild | Grab median 13 fps, Smooth 6.6, Inflate 9.9; worst gap 401 ms; release to next stroke 0.29-0.38 s |
| Bust sphere, 1166 x 611 view, 2026-11-11 16:21, r 20 / 60 / 80 | Grab 53 / 33 / 27 fps, Draw 50 / 40 / 28, Smooth 53 / 38 / 28 |
| One batch at r 80 | Grab: plan 10.2 + re-distance 15.1 + borders 3.9 ms; Draw 6.8 + 7.7 + 4.0; Smooth 7.3 + 7.5 + 6.6 + dabs 5.5; render 15.5 ms |
| Bust sphere, harness, 2026-11-12 19:00, r 5 / 20 / 60 / 80 | Draw 37 / 31 / 22 / 18 fps; Smooth 29 / 27 / 19 / 16; Grab r 60 / 80: 30 / 27 (one r 20 run 5.0 fps with an 845 ms gap, unexplained); window stall 29-105 ms |
| Head scene, mirrored Smooth r 30 | 6.6 -> 21.6 fps after removing a per-batch scene sync and mask scan; in a separate A/B the thread switch interval alone gave 15.0 -> 29.3 fps |
| Human model, upper lid, small strokes | about 22-23 fps quiet; 2-5 fps under machine load. Frame at 21-22 fps: tables 1.6-8.1 ms, frame 6.7-10.1, collider 4.4-6.5, dab kernels about 1 ms |
| Human model, torso Draw | 14.6 -> 23.6 fps, release 173 -> 113 ms (2026-11-12 10:47) |
| Human model, body Smooth | 2.5 -> 5.8 fps, release 0.73 -> 0.33 s |
| Human model, body Flatten | 0 -> 1.0 fps, release 2.84 -> 2.38 s; r 40 writes 90,000 bricks, 82 % of them more than 4 mm from the surface |
| Human model, Draw r 10 on the cheek | per batch 830-1891 ms -> 6-149 ms after keeping the cell-to-row map on the card instead of uploading 332 MB per batch (2026-11-12 01:49) |
| Human model, CPU path, lid Smooth r 4, 13 dabs | 13.3 s -> 8.4 s |
| Human model, CPU path, torso Draw r 25, 2 dabs | 11.8 s and 23,972 bricks with no band; 5.3 s and 16,008 bricks with the correct one-sided band |

The cost of a ball of radius r grows as (2r / h)^3 nodes, and the radius is capped at 80h (160h on the human model).
Coarse levels for large tools, with detail stored relative to the coarse surface, are proposed. A whole-body Smooth
at r 60 once ran the card out of memory (6.52 of 6.91 GB in use) and hung the engine; strokes are now split to fit
and a card failure falls back with a notice. Hands-on use on 2026-11-12 found performance very poor: lag
set in after several minutes, and some tools took several seconds to act.

### 12. Test 1: a sphere to a head, and why its scripted runs meant little

Test 1: front and side reference photos on one axis, a sphere sculpted into a basic head (nose, eye sockets, jaw and
chin, cheeks) in under five minutes.

- 2026-11-11 10:47, first scripted run: not yet a head, a lumpy egg after 144 actions and about 7.8 minutes; Smooth
  removed 6 % of the lumps; a torn ring under the neck; rectangular seams at the mouth; 22 of 166 presses turned the
  view instead of sculpting. The brush used was too small.
- After the rebuild, scripted runs from a fresh sphere reached all stages in about 39 actions and 113-121 s of a
  person's time, 0 orbits, no holes, outline within 3.9 / 5.3 / 4.8 mm (front / face / back), lumps 2.47-2.52 mm RMS
  (target 1.5, not met). A brushes-only run was ranked best so far on 2026-11-11 18:48.
- Along the way: a nose split into two lobes (mirror), a granular jaw (midline snap stacking about 15 Smooth passes),
  jaw scratches on the card (ring samples reading unheld bricks), and the chin bands of section 3.1.
- 2026-11-12 about 17:20: a head could not be blocked out by hand. The script had pulled only from the side and
  front views, where screen-plane tools happen to do what was meant, with strokes chosen to work, and it judged the
  outline in millimetres. It measured the field, not the experience.
- The summary that evening: Smooth and Swell worked fairly well; most of the other tools were effectively unusable;
  progress had been made, with a long way still to go.

### 13. The shell

All in the live tool; each item was checked by a scripted run of the real window (evidence state: measured on named
cases) unless stated.

- **Window** (2026-11-11 11:50). Workspace tabs, a Create row of primitives, a telemetry strip (fps, frame time, RAM,
  field and undo memory, the field's Lipschitz bound), the viewport, docks for properties, references, layers,
  regions and light, and a timeline of every operation with a replay check. A status line always says what each
  mouse button does. The brush ring is drawn draped on the surface with live millimetres; Alt shows the local
  thickness and the gap to the next surface (2026-11-11 23:49).
- **Shelf, palette, keys** (2026-11-11). Tools sit on a shelf in groups (Object, Build, Refine, Move, Cut & Join,
  Protect & Select, Contact). Space opens a type-to-find palette; favourites sit in a strip; a Keys window lists and
  rebinds every key. Each brush keeps its own radius, strength, falloff and presets.
- **Glyphs** (first set 2026-11-11, drawn from the brush falloff; redrawn 2026-11-12 08:56). Each glyph is a filled
  cross-section of what the tool does to a surface. The redraw raised the closest pair's image distance at 32 px
  from 0.0039 to 0.0458. A names mode shows one tool per row with its name.
- **Grids** (2026-11-11 16:42; on by default 2026-11-12 09:17). A depth-tested floor grid under the scene's lowest
  point, fading with view distance; a view grid in axis views; spacing on a metric or imperial ladder with labels;
  optional height contours on the model. Cost: 1-2 ms per displayed frame.
- **Reference rig** (2026-11-11 11:50). Front and side photos as planes crossing on one axis, keyed out without a
  fringe, placed by crown and chin clicks or an auto-fit proposal; a true orthographic view; a silhouette check in
  millimetres. On a synthetic head with known truth, clicked landmarks put the model's outline within 0.12 / 0.17 mm
  mean of its photos. Drawing the planes cost 10.2 ms per frame until the per-view layers were cached (2.4 ms).
- **Typed transforms and masses** (2026-11-11). Move, rotate and scale with typed values, axis locks and snapping;
  primitives joined by smooth unions as live "masses"; flip and mirror on any axis; duplicate in place.
- **Regions** (2026-11-11 21:57 data, 22:33 window). A 16-bit region id per lattice node and a region table per
  object, carried through save, duplicate, mirror and merge. Regions are made by lasso, from the mask, or from a
  part. Hide, isolate, show all, and box hide. The CPU tracer clips hidden regions (the surface and the solid
  beneath, by each point's foot on the surface) and boxes (convex plane sets); with nothing hidden the image is
  bitwise unchanged. The card clips boxes (23:33). **Hidden regions on the card are not built.**
- **Section tool** (2026-11-11 19:17 CPU; card clip 23:33). A section plane on the camera cuts the view and shows
  the solid's cut face with a millimetre ruler; 24 cases are bitwise identical with no section.
- **Cut tools.** Cut (first commit 2026-11-05): a closed outline drawn on screen is punched through the object along
  the view as an exact swept-outline field. Cut with a shape (2026-11-11 16:30): any object is the cutter, with a
  side and result choice and a live preview of the volume that will go (16:48; areas within 0.5 % of geometry).
  Conform with a wall (19:21): hollow a form leaving a wall of set thickness, using a true depth computed from
  surface points because the sculpt layer is a distance only inside its band (0.043 mm maximum error on a ball).
- **Marks** (2026-11-11 19:10). A circle or note placed on the model, saved to a shared file so a second reader can
  see exactly the point meant.
- **Also**: measures (sections, girth, tape, volume), sculpt layers with a 0-1 dial, view modes on the card, a
  visible notice when an action fails or the card is lost.

### 14. Published work this learned from

- Blender's sculpt mode, versions 4.2-5.0: the manual (CC BY-SA 4.0), read for study; the source (GPL), read for
  behaviour only, no code copied; Blender Studio course pages (CC BY), descriptions only. Taken as ideas: the area
  plane under Clay and Flatten, Ctrl as each brush's inverse, autosmooth as a dial, middle-button orbit, pinch toward
  the stroke line, anchored versus incremental tools, feathered symmetry.
- Maxon ZBrush documentation, tutorials and user forums (all rights reserved): study only, paraphrased.
- F. de Goes and D. L. James, "Regularized Kelvinlets", SIGGRAPH 2017: studied; not implemented in the tool.
- W. von Funck, H. Theisel, H.-P. Seidel, "Vector field based shape deformations", 2006; A. Angelidis, M.-P. Cani,
  G. Wyvill, S. King, "Swirling-sweepers", 2004: prior art for moves as flows of divergence-free fields.
- A. Ricci 1973 and later implicit-modelling texts for CSG by max and min.
- Public forum threads recording that stopping an inflate at contact is not built into mesh sculptors.
- A 1912 anthropometric card from Wikimedia Commons (CC0), used only to test the reference rig.

### 15. Patent readings as recorded

These are a non-lawyer's readings made to decide what to build. **They are not legal advice and no clearance is
claimed for any mechanism in this document.** As we understand it, a dated publication is prior art only against later filings. The rule
followed: a concern names the claim element behind it; a design-around names the element that is absent.

- **US 10,586,401 B2 (Pixar; active, expected to 2038).** The patent concerns deformations determined from
  regularized solutions of an equation of linear elasticity with a specified Poisson ratio. The Elastic Grab prototype
  (a lattice solve of linear elasticity) was flagged in our review; it was never merged, is parked, and will not ship. An
  earlier internal note that had treated it as unaffected was withdrawn on 2026-11-12. Direction of the
  design-around: the form-aware Grab decided the same day and the flow Grab of section 8 use no elasticity equation
  and no Poisson ratio; whether that is sufficient is not asserted here.
- **US 8,265,909 B2 (MERL; active, expected to 2031).** Element of concern, claim 1: the distance field of a shape
  swept along a parametric path, found by determining an optimal path parameter per sample continuously.
  Flagged in two places: the capsule primitive (a sphere's distance at the clamped closest point of a segment) and
  the brick admission test of an unwired flow-Grab module (brick centres projected onto the drag segment with a
  clamped parameter). **Both are being redesigned; no design-around had been chosen or recorded when this was
  written**, and the full reading was still owed. Ordinary dabs are separate fixed-centre edits added in turn, with no
  per-sample search for a placement.
  The current Pencil (0.2.0) evaluates discrete path samples, each at its own placement relative to the stroke-start
  surface, and nothing searches along the path for a nearest placement; its implementation is described in
  `docs/PATENTS.md`. The records hold no reading of the rebuilt pencil of 2026-11-12 against this patent; that reading is owed before the pencil
  returns.
- **US 9,836,879 B2 and US 10,062,203 B2 (Autodesk)**, read against the sheet flood. The first requires paths
  through interior voxels from an influence to a mesh vertex, a boundary-voxel penalty, a skinning weight and an
  animation; the flood walks surface cells from the press point and yields a yes/no gate on a brush write. The second
  requires voxelising a mesh; there is no mesh.
- **US 8,988,461 B1 (Disney / ETH)**, read against Swell and Settle: each independent claim requires paint strokes
  that leave the object's shape unaltered; these tools alter the shape. Older distance-field sculpting and
  haptic-clay patents (MERL, SensAble) appear expired; their status was not confirmed at the patent office.
- **US 9,830,743 B2 (Autodesk)**, a volume-keeping smoothing brush: to be read before any keep-volume Smooth ships.
- The patent searches were web searches and partial. Finding none does not show that none exists.

### 16. What is not established

- Whether the rebuilt Grab, Stretch and drag constraint give a sculptor a usable tool. They are measured on the CPU
  path through a scripted window only. The card path of the anchored flow has never run; the new Stretch is unmeasured.
- The form-aware Grab: decided, not prototyped in the records read.
- The cause of the five-minute window freeze, and of the 5 fps Grab run at r 20.
- Why the card and CPU differ for Swell to contact; which of the card's two skip rules causes the plateaus and fins.
- A Smooth that keeps volume, has spheres of every size as fixed points, and does not band on a chin.
- Scope for move tools, reach-through as a control, the honest ring, levels, region transforms, and deformers that
  carry the sculpting with them: all proposed.
- A hands-on block-out of a head with the rebuilt toolkit, which is the pass mark, has not happened.

---

# Part 6. Contact and parts

**How to read this section.** It is a dated technical record of how Field Studio keeps separate bodies and tissues
apart, lets them touch, and solves what happens when they press on each other. Times are internal provenance from the
project's working record and commit log (Pacific time, 2026); the date this text became public is the date of the
release that carries it. Every mechanism carries one evidence state from this list: *proposed*, *implemented*,
*measured on named cases*, *reproduced under a frozen configuration*, *accepted in the workflow*. Nothing in this
section has been reproduced under a frozen configuration yet, and we say so where it matters. Patent remarks are our
own reading of claim text, not legal advice, and no clearance is claimed. Nothing here is claimed to be new to the
world; published ingredients are named with their sources.

**Machine.** One Windows PC with 32 GB of RAM and one 8 GB RTX 5050, shared with other jobs. The 11-12 solver runs are
CPU runs, one solve at a time at 4 threads (numba / OpenMP / MKL all set to 4). Because the machine was shared, seconds
are indicative; iteration counts are exact.

**What is and is not in the public code.** The IPC-style solver in `engine/contact/` (B.3) is **out of the public code
set**: it is flagged in our review; held out of the code pending redesign (B.5, B.6).

---

### B.1 Parts: separate fields that touch without fusing

**Evidence state:** measured on named cases (closed lips on the human model; a synthetic eyelid on a globe; synthetic
pressed balls). **Dates:** per-part entries in the store and renderer 11-06; pair settings 11-11 01:59; the general
design below 11-11 16:20-21:26.

**The problem.** If a body is one signed distance field, two surfaces that touch become one surface: closed lips merge,
a lid welds to the eye, pressed fingers become a mitten. The crease between them is narrower than a lattice cell, so a
single stored value per sample cannot hold both.

**The mechanism, step by step.**

1. *A part is its own field.* Each part has an id (16-bit; 0 is the body), a name, a kind (tissue, analytic, garment),
   a material profile key, and its own **band**: its own smooth signed distance on the store's lattice within a narrow
   band of its *own* zero set. The band includes the stretch of the part's surface that lies inside another part (a
   closed lip's inner face, a lid's underside on the globe, a finger's side pressed into its neighbour). Without that
   hidden stretch nothing can later open, slide or pose the pair. Storage: per level-0 brick (0.25 mm samples, 9x9x9
   half-floats), rows sorted by part.
2. *Everything else is a cache rebuilt from the parts by one call.* The union field (minimum over the parts' bands),
   the per-brick contact entries, the fine leaves, a part label per sample, and the transported surface attributes are
   rebuilt for a region by one `rebuild` call. No writer edits a cache by hand. A rebuild with no change writes nothing
   (before this rule a zero-target operation had re-distanced 32,927 samples by up to 1 mm).
3. *Contact entries ("k entries").* A brick where two or more parts come within two cells of each other stores one
   entry per part: that part's own field in that brick. The renderer takes the minimum over each part's own trilinear
   interpolant, so the crease is the true crossing of two smooth surfaces and each side keeps its own normal. Check
   "K0": the minimum over the entries equals the stored union bit for bit at every slot sample.
4. *k is soft.* Measured on synthetic adult shapes (11-11), the number of parts in one brick stays at 6 or fewer at the
   0.25 and 0.5 mm levels, but a closed mouth (28 teeth, gums, palate, tongue, cheeks, lips: 36 parts) reaches 10 at
   1 mm and 17 at 2 mm, and clasped hands with a part per phalanx reach 11. So there is no fixed limit of 8: one 16-bit
   id per entry, up to 32 per brick, refused beyond that and never silently dropped. A 12-part synthetic cluster drew
   with 0 fused points (the old limit of 8 gave 563).
5. *Leaves.* Where the gap between two parts is thin, 0.125 mm leaves refine the entries.
6. *Area is the real limit.* Memory grows with contact area times k. One forearm under four garments cost 200 MB of
   entries at 0.25 mm. So garments, held objects and furniture are separate bodies with their own fields, traced as
   their own volumes; entries are for tissue-on-tissue contact in small zones and for small-area contacts.

**Tissue-pair rules.** Every pair of parts carries one setting, and the setting means one stated thing in each
consumer:

| setting | store | renderers and Field Studio's draw | brushes | motion solve |
|---|---|---|---|---|
| touch | both bands; entries where they meet | minimum over each part's own interpolant; a crease; the visible part's own normal | a stroke on A never moves A deeper into B than it already is; B unchanged | contact with static friction |
| slide | as touch, flagged | as touch | as touch | contact with kinetic friction; tangential motion allowed |
| bond | one entry (smooth minimum, default width 0.5 mm): one tissue, no crease | no crease | bonded parts edit as one | tied nodes |
| pass | both bands kept | two surfaces drawn where they cross | no clamp | skipped |

Switching between touch, slide and pass changes a flag; switching to or from bond rebuilds that pair's entries.
Examples from the record: lips touch each other and are bonded to the face at their outer borders; a lid touches the
eye at rest and slides in a blink; a tooth and its gum are one touch pair, never a bond; a garment is its own part with
slide. A per-pair **contact record** (gap, real-contact share, fluid fill, friction, adhesion; per side roughness, film
and stiffness) has a schema and a home on a sparse 0.25 mm lattice over the pair's zone; only friction is filled today
(from the materials library's pair table), and the other slots stay empty until a measurement or cited source fills
them.

**Labels carry the material.** The part's material key selects its tissue layers, friction, thickness and damping from
the materials profiles (B.3), so labelling a region applies its physics.

**Contact that changes over time.** Rest is authoritative: every pose or edit rebuilds the caches from the parts in
the changed bricks only, never from an earlier pose, so error cannot accumulate and the return to rest is exact by
construction. A brick gets entries wherever two parts come within four cells (hysteresis), so a closing gap is drawn by
entries before it narrows under two cells. Self-contact that appears in a pose is designed as a split of the part into
sheets for that pose (entry id = part and sheet); that part is designed, not measured.

**Measured.**

| case | result |
|---|---|
| Two pressed synthetic balls above a body (11-11) | K0 0 differences at 0.25 / 0.5 / 1 mm; 0 fused points at 0.5 mm against 5,332 for the union alone; a 1 mm separation shows each hidden cap within 6.2 um (the older reader: up to 995 um) |
| Closed lips on the human model (11-11) | built as two lip parts plus the face; store cost 12.1 MB of bands, 3.6 MB of entries, 14.3 MB of leaves; opened in Field Studio as three objects in 1.0 s |
| Synthetic lid on a globe, 13 blink frames (11-11 20:50) | 0 stray points; margin gap 1.0 um typical / 6.4 um worst; 0 fused frames; return to rest byte-identical |
| Field Studio open / draw / brush / write-back on the lips | open, brush clamp and unedited write-back marks passed; one draw-agreement mark was argued rather than passed |

**Failures and corrections.** (a) Field Studio first read each part as a dense lattice box; estimated for a whole body
of about 90 parts that is 3-6 GB of RAM, over budget. Parts were changed to sparse stores read from the band rows
(11-11 19:38): about 16 MB of program data per lip measured; the whole-body figure (about +1.6 GB RAM, about 1.6 GB of
card memory) is an estimate, not a measurement. (b) A record that the lips rest closed was corrected: on the current
model they rest 0.3-1.2 mm apart across the middle and touch only near the corners; closed is a state, not the rest.
(c) The leaf writer ran out of memory on a 34-part mouth and was rebuilt to work in bounded batches (2.49 GB peak).

### B.2 Solid by default, with designed cavities

**Evidence state:** measured on named cases (the human model's store). **Date:** first stated 11-12 00:32;
built and published into the working model 11-12 07:50.

The rule: objects are solid; a designed cavity is never filled, and is made deliberately, with the cut tool or as a
labelled part, so the tool knows it is meant. Tests must show zero unintended pockets.

As built: the model's interior, which an earlier surface-band store left unknown beyond its band, is given the inside
sign at the coarsest level; enclosed air pockets of 0.01 mL or more that no design accounts for are found on a 1 mm
grid by flood fill from the exterior, given the inside sign and re-distanced inside the pocket mask only. Checks on the
published version: outside the edited masks 0 differing samples at every level; the head's outside-reachable air
(mouth behind parted lips, nostrils, ear canals) is 8,725.9 mL before and after with 0 voxels changed, and the mouth
box 268.43 mL with 0 changed; rays from front, both sides and 30 degrees below differ in 0 of 31,403 per view. The
model's inside volume on a 6 mm grid went from 27.4 L to 60.0 L. **Left open:** about ten single-sample enclosed specks
(about 0.3 uL each) and one 1.4 uL thread remain and are recorded as defects to fix at the shape. A general "no
unintended pocket" guard at every commit is not built.

### B.3 The first solver: IPC on fields (not in the public code; the measurements below compare against it)

**Evidence state:** measured on named cases (its own proof suite and the cases below). **Date:** built 11-07 to 11-08
(first commit 11-08 11:35); self-contact and cut cells 11-12 03:32-06:38. **Not in the public code** (B.5).

**Credit.** The model is the incremental potential of IPC (Li et al., *Incremental Potential Contact*, ACM TOG 39(4),
SIGGRAPH 2020; reference code and IPC Toolkit under the MIT licence). No code was copied; the equations come from the
paper. The linear solver is PARDISO through `pypardiso` (BSD-3; MKL under the Intel Simplified Software License), with
SciPy's SuperLU as fallback.

**How it works.**

1. *Bodies.* A soft body's rest shape is a signed distance field phi0(X). Its deformation is a displacement field on a
   regular lattice of spacing h (4-10 mm in these runs), trilinear cells, no mesh. Each Gauss point is weighted by the
   share of its octant inside phi0 < 0. A one-cell ghost envelope keeps the map defined just outside the skin. An
   optional two-level lattice (2h away from contact, h near it, hanging nodes tied to their parents) matched the
   all-fine indentation force within 0.72 % at about a quarter of the unknowns.
2. *Material.* Compressible neo-Hookean per layer, selective reduced integration, Poisson's ratio 0.49. Layers by depth
   below the skin from the materials profile; a Gauss point straddling layers takes their series modulus; bone is
   tied rigid. The body-skin stack used in every elbow run: epidermis 60 um at 4 MPa, dermis 1.5 mm at 40 kPa, fat
   18 mm at 26.1 kPa, muscle at 62.2 kPa. The profile cites optical coherence elastography of forearm skin for the two
   skin layers and in-vivo indentation of gluteal fat and muscle (26.1 +- 19.0 kPa at strain 0.16; 62.2 +- 10.3 kPa at
   0.19) for the deep layers, with ultrasound thickness surveys for the thicknesses. These are literature values, not
   measurements on the model (see B.9, experiment d4, for a challenge to the fat value).
3. *Gaps.* Point against a rigid field: phi_R(x) minus thickness. Point against a soft body: phi0_B(X_B(x)), by
   inverting B's deformed lattice map, with exact gradient. Self-contact: the same lookup, counting only material
   cells (not ghost cells or extrapolated maps), skipping hits whose located rest preimage lies within about 0.5 mm of
   the sample's own rest point, and dropping inverted preimages.
4. *Contact law.* The log barrier b(d) = -(d - d_hat)^2 ln(d / d_hat) for 0 < d < d_hat, zero beyond; d_hat = 0.1 mm;
   row stiffness kappa_k = E / d_hat x A_k (E the softer body's modulus, A_k the sample's area). Friction is lagged,
   smoothed Coulomb, static to kinetic, with stick width max(1e-3 m/s x dt, 1 um); coefficients per material pair.
5. *Solve.* Newton on the incremental potential with a backtracking line search; each step bounded by conservative
   (Lipschitz) continuous-collision bounds and a trust region of 0.25 h; a step is accepted only if every gap and every
   volume ratio J stays positive. Residual tolerance 1e-7 of the load; friction re-lagged up to three times.
6. *Cut cells (11-12 05:33).* A cell cut by the skin measures tissue at its in-tissue Gauss points only; the empty part
   is held by 0.1 x the cell's tissue stiffness in the rotation-invariant form mu |F - R|^2, finite for every F. Before
   this, Gauss points in a cut cell's empty part hit an invisible J > 0 wall and an elbow sweep stalled at 67.5-72.5
   degrees.

**Measured (11-08, with other jobs running; GPU figures on the 8 GB card).** A layered slab of 45.4k unknowns pressed
by a sphere: 111.7 s for 46 Newton iterations at first, 11.4 s after cached assembly, a Jacobi-preconditioned conjugate
gradient on the card and card-side tissue energy. Interactive frames, warm-started: 0.086 s per frame at 11.3k unknowns
and 0.1005 s on a two-level lattice of 10.8k unknowns (force within 0.25 % of the all-fine answer); 0.277 s at 45.4k.
Friction check "T0" (a block on a tilted plane): stick passes; slide acceleration within 3.45 % of Coulomb's law.

**Failures on record.** One of twelve regression tests fails as written (a viscoelastic swing on the two-level lattice
misses its residual on a few near-zero-load steps). A seated-load proof has one pressure mark that fails high. A fabric
band on the model's ribcage passes its volume mark and fails its pressure mark (ratio 1.30-1.65 against a smooth-curve
estimate); tracing showed the band bears on ridges and bridges hollows, obeying Laplace's law with its own curvature
within 8 %, and one residual curvature difference is still open. Cloth self-contact, fibre twist and adhesion are not
in.

**Earlier round, for the record (11-05).** Before this solver, one soft-tissue region was settled with linear hex
elements on a 2 mm lattice and frictionless contact solved as a linear complementarity problem on the Delassus operator
(one sparse factorisation, FISTA iterations, re-linearised five times), with fat 2 kPa and dermis 35 kPa after
Pailler-Mattei et al. 2008. It was a single-region tool and was superseded by B.3.

### B.4 Skin-on-skin self-contact rows

**Evidence state:** measured on named cases (the stand-in elbow); built in a working copy, **not landed**; its own pass
mark is not met. **Date:** 11-12 06:00-08:50.

At a closing crease the skin folds onto itself below cell scale, and the volume-map self-contact of B.3 first sees the
contact when it is already about 0.3 mm inside. The replacement measures skin against skin directly: each skin sample i
is paired with facing samples j in the deformed state; the gap is point-to-plane, d = (x_i - x_j) . n_j - t_i - t_j,
with n_j the rest normal carried by the cell's deformation gradient (F^-T n0, normalised); one row per sample, the
partner whose plane foot is nearest; rest neighbours within two sample spacings never pair.

Two faults were found and fixed. First, "0 pairs crossed" was a blind metric, because rows form only at positive gap;
an independent counter showed two pairs 127-217 um past each other at 82.5-90 degrees. A crossing guard now keeps a row
for every pair that was outside at the last accepted state. Second, the cause was sample spacing: at a crease narrower
than the 2 mm sample spacing the surface crosses *between* samples (at 80 degrees, 6 samples inside at 0.5 mm sampling,
0 at 2 mm). Samples went to 0.5 mm in the fine crease cells. With both, 0-70 degrees is clean at 19-33 s per step, but
from 72.5 degrees a few samples sit 0.27 mm inside tissue without any pair crossing: the 4 mm lattice folds over its
own neighbour across a cell face (B.9). One run ended in a memory error (77.7 million candidate pairs from an
over-wide search margin).

### B.5 The patent finding on the IPC-style solver

**Date:** 11-12 08:22 (the reading); 11-12 about 09:00 (a wider search); a further check of our readings the same
morning. **This is our own reading, not legal advice; no clearance is claimed.**

**The patent.** US 12,216,969 B2, "Methods of contact for simulation", NVIDIA Corporation, filed 2020-09-04, granted
2025-02-04, term adjusted by 1,033 days. Read from the USPTO document, claims on its pages 96-97. A pending
continuation, US 2025/0131161 A1 (application 18/999,835, filed 2024-12-23), was also read.

**The flag.** As we read them, the patent's independent claims concern simulation of contact in which forces
determine constraints, a preconditioner is applied to a gradient descent, and an energy minimisation is solved in a
primal formulation. Our first solver (B.3) is a barrier energy minimised by Newton steps, so it was
flagged in our review; held out of the code pending redesign. We publish no conclusion on whether any claim is or is not met;
that is a question for counsel, and so is the effect of earlier published work (IPC itself was published by other
authors before this patent's filing date). Two cautions were recorded: a rewrite was not yet
proven necessary, and a replacement was not yet proven free of the claims. Decision (11-12 about 08:35): build
around it. Direction of the design-around: the constraint solver of B.6, which solves the contact conditions as
equations with multipliers and minimises no energy. The same search also flagged US 7,616,204 (position based
dynamics, to about 2027-08) for any position-level loop.

### B.6 The constraint solver ("R1"): contact as constraints with multipliers (the rebuild, underway)

**Evidence state:** measured on named cases (the friction check and the stand-in elbow); a prototype in a working
copy of the contact code; it fails the broad soft-on-soft cases. **Date:** designed 11-12 08:45; prototype
09:30-15:21.

**What it is.** A time step solved as a nonsmooth system of equations, not as the minimum of an energy.

- *Unknowns:* velocities v+ (positions follow, x+ = x + h v+), one normal multiplier per contact row, one 2-vector
  friction multiplier per row. In quasi-static mode the unknown is the load step's displacement increment.
- *Equations F(z) = 0:*
  1. momentum: M (v+ - v) / h - f_int(x+) - f_visc - f_ext - J_n^T lambda_n / h - J_t^T lambda_t / h = 0;
  2. Signorini contact through the Fischer-Burmeister function, phi(a, b) = a + b - sqrt(a^2 + b^2) = 0 with a the
     scaled gap and b the scaled multiplier, which gives d >= 0, lambda >= 0, d lambda = 0: contact closes to zero gap,
     with no barrier offset;
  3. Coulomb friction through the Alart-Curnier projection, lambda_t = proj onto the disc of radius mu lambda_n of
     (lambda_t - r u_t), u_t the step's slip.
- *Solve:* one generalised (semismooth) Newton method on the whole residual, with an element of the B-subdifferential;
  the Jacobian is unsymmetric once friction is in; factorised directly (PARDISO unsymmetric, SuperLU fallback).
  Globalisation is a backtracking search on |F|^2 / 2, never on an energy.
- *Parameters and additions, each from a measured failure:* smoothing continuation of both nonsmooth functions (tau
  from 1e-2 driven to 0 inside each step, judged on the exact residual) after active-set chattering on a sliding face;
  a non-monotone search (memory 8); the Alart-Curnier r per row from the Delassus diagonal, times 0.3 (times 1.0 failed
  the landing 3 of 3; 0.3 passed 5 of 6); a 1e-6 term on the multiplier diagonal of the Jacobian only, for redundant
  rows; exact tangents (the baseline's projected tangents are right for a minimiser and wrong for Newton on equations:
  9e-4 relative error in cut cells, 6e-10 after); the exact curvature of each skin row (a per-row Hessian by the
  complex step), which took one crease step from 302 iterations / 521 s to 47 / 47 s; pseudo-transient continuation on
  the Jacobian for static solves; a descent test with fallback step -J^T F; an adaptive trust radius on node motion
  (0.25 mm start, 2 mm cap); row sets frozen during a line search; deterministic linear algebra (MKL reproducibility
  mode), without which the stick/slip outcome depended on thread count and summation order.
- *No pass-through:* a facing pair keeps its row at d <= 0 unless it was already clearly back to back (d < -1 um) at
  the last accepted state, so a new crossing is a violated constraint that Newton restores. Every accepted state is
  checked by an independent counter (samples inside tissue; skin pairs crossed).

**Published basis (as named in the design record):** Alart and Curnier 1991; Simo and Laursen 1992; Fischer 1992; Qi
and Sun 1993; De Luca, Facchinei and Kanzow 1996; Christensen, Klarbring, Pang and Stromberg 1998; Moreau 1988 and
Jean 1999; smoothing methods after Chen, Qi and Sun 1998 and Kanzow and Pieper 1999; non-monotone search after Grippo,
Lampariello and Lucidi 1986; pseudo-transient continuation after Kelley and Keyes 1998; complex-step differentiation
after Squire and Trapp 1998. The nearest modern work is Macklin et al. 2019, *Non-Smooth Newton Methods for Deformable
Multi-Body Dynamics*; the search found no US patent family for it.

**How it differs, element by element (our reading).**

| claim element | R1 | absent? |
|---|---|---|
| 12,216,969 (c): energy minimisation using a primal formulation | zeroes a nonsmooth complementarity residual with explicit multipliers; the specification describes "primal" as optimisation-based and usable without Lagrange multipliers, and sets it against constraint-based (dual) formulations; Coulomb with Signorini is not the minimum of any energy | absent on our reading; this is the element relied on |
| 12,216,969 (b): preconditioner applied to a gradient descent | a generalised Newton step on F from a direct factorisation; search on the residual norm | absent on our reading, but weaker alone: the specification calls Newton with a Hessian inverse a preconditioned descent, and a residual-norm search is a descent on a merit |
| 12,216,969 (a), (d) | forces are computed and states updated | present |
| dependent claim on dropped geometric stiffness | R1 keeps the constraint curvature | the opposite |
| continuation, claims 21 / 28 / 35: "a pre-conditioned force-based formulation" for at least two objects, at least partly in parallel | by the family's own specification "force-based" names the primal kind; R1 is the constraint-based kind and applies no preconditioner to any formulation | absent on the specification's own definitions |

**The stated weak point.** If "force-based formulation" were read more broadly than the specification defines it (any
method whose equations balance forces), R1's momentum equation is a force balance, and the difference would rest on
the word "pre-conditioned" alone. The design constraints kept for that reason: linear systems solved by direct
factorisation or a Krylov method on the full constraint system; no preconditioner applied to a momentum-only system as
the method of simulating the objects; the graphics-card path re-read against these claims before it exists. **The
continuation is pending**, so its claims can change; this reading is to be redone when it grants or is abandoned.

**Constraints taken from other patents the search found (our reading; each names what R1 does not do):** US 7,616,204
(no predict-then-correct positions; static mode has no velocities at all, dynamic mode is flagged under a broad
reading); US 11,270,041 (hard multipliers, no compliance tied to elastic energy); US 9,589,383 (one global solve, not
per-constraint parallel iteration); US 11,200,356 / 12,755,068 (no per-sub-interval bias accumulation); US 10,614,257
family (no mass splitting); US 9,443,341 / 10,198,844 (no projected Gauss-Seidel over separate trees); US 11,210,833
(no alternation of elastic relaxation and contact projection with an error term); US 10,198,847 (no polyline search,
no push-out pre-step); US 11,675,935 (no intermediate implicit surface tied to body vertices). Not reached by the
search: European, international and Chinese families, and US applications filed after about 2025-05.

**Measured.** Friction check: stick 0.193 mm (mark < 0.4), slide within 0.46 % of the law (baseline 3.45 %), no
pass-through, 11.4 / 7.5 Newton iterations and 1.95 / 0.84 s per step. Stand-in elbow with skin rows: every state
converged to 1e-7 up to 88.44 degrees with 0 skin pairs crossed and pairs touching at zero gap; 6-20 iterations per
step to 75 degrees, about 200 iterations and 279 s per step at the apex.

**Failures.** R1 times out on every scene with broad contact between two soft bodies (fingertips, lips, a lid on an
eye): many samples pull on near-redundant rows. Merging each pair's two one-sided rows into one converged but crossed
the dropped plane (11 pairs, 294 um), so it was reverted: where skins meet nearly at right angles both planes must
hold. The friction check's landing step sits near a convergence boundary and needs one step halving. Speed against the
baseline's interactive bench and the full proof suite are not run.

### B.7 The magnetic-field contact concept

**Evidence state:** measured on named cases (synthetic scenes). **Date:** first stated 11-12 about 15:30; patent
reading of the options 15:31; built 15:50-17:15 (in a working copy of the contact code).

**Idea.** A repulsion between surfaces that is zero until they are very near and reaches its limit where they touch,
read directly from the fields.

1. *Gap.* A contact sample on body A takes its gap d from body B's stored signed distance field (trilinear value and
   gradient), the same rows the other solvers read. This replaces primitive-pair search for the gap itself; it still
   needs a broad phase, interpolation, and for deforming bodies either a current field for B or a map into its rest
   frame.
2. *Law.* IPC's published log barrier (B.3, step 4) with d_hat = 100 um in every run and the baseline's row stiffness;
   force kappa b'(d) along the gap gradient. Bodies therefore rest at small positive gaps (49-95 um measured), not at
   zero. Calling that film a physical moisture layer would need separate evidence.
3. *Form N (hybrid).* The barrier is the contact law inside R1's Newton on the static force residual: no multipliers,
   no energy evaluated, exact tangent (bodies, kappa b'' J J^T, kappa b' times the gap's second derivative on skin
   rows, friction), pseudo-transient continuation, search on |F|^2. Each iterate is bounded by distance: no gap within
   reach may shrink by more than half its current value (linearised); a bound on step length, never a clamp on points.
   Friction is smoothed Coulomb on the step's slip with a 1 um smoothing length.
4. *Form E (standalone).* Explicit symplectic Euler; the time step bounded by stability (factor 0.5) and by the same
   half-gap rule; stiffness-proportional tissue damping (0.3 of the explicit limit) and a damped normal response per
   row (ratio 0.5); a mass floor of 5 % of the median node mass on nearly empty cut-cell nodes; static rest by dynamic
   relaxation (Gershgorin fictitious masses, kinetic damping after Cundall).

A diverging force does not by itself prevent crossing in a discrete solve; non-crossing here rests on the step bound,
the sample spacing and the field's accuracy, and was checked by the independent counter.

**Results (synthetic scenes; 4 threads, one solve at a time; seconds indicative).**

| Case | Form N | IPC baseline | Form E | R1 | Notes |
|---|---|---|---|---|---|
| Two fingertip pads pressed 2 mm | 27 s | 86 s | 519 s | timed out (> 600 s) | forces 0.353-0.355 N (within 1 %), gap about 80 um, 0 crossed, 88 samples in contact |
| Lips closed 0.8 mm | 16 s | 27 s | 295 s | timed out (> 600 s) | 0.0124-0.0133 N, gap about 92 um, 0 crossed |
| Eyelid drawn 1 mm onto a rigid eye | 73 s | 7.4 s | 18 s | timed out (> 300 s) | **a reversal:** soft-on-rigid favours the baseline; all three agree at 0.373 N, 87.8 um |
| Hand pad on cheek, 2.5 mm press then 1 mm sideways | **did not converge in 10 min** (first load step) | not run | not run | not run | open; suspects (friction smoothing, load step size) untraced |
| Elbow bend, stand-in arm | every state converged to 88.75 deg, 0 crossed | "90 deg" with 85-90 unconverged at its iteration cap | not run | converged to 88.44 deg, 0 crossed | the lattice still folds locally at the crease (0.23-0.24 mm in all three); hard steps took about 4 min |
| Friction check (block stick / slide) | pass (0.292 mm; 1.68 %) | pass (3.45 %) | **fail** (2.4-3.7 mm slip; +170 to +230 %) | pass (0.193 mm; 0.46 %) | explicit friction loses grip to contact-face vibration; stronger damping changed little |

A first attempt at the cheek and lid scenes measured nothing, because the driven regions selected no nodes; that was a
scene-construction fault, fixed before the rows above.

**Reading.** On the two soft-on-soft scenes Form N was 1.7-3.2 times faster than the baseline at the same answer; on
soft-on-rigid it was about 10 times slower; one scene did not converge. That is promising on the tested cases and not
established as generally better.

**Relation to patents (our review; no clearance claimed).** Form N is a barrier force inside an equilibrium Newton
solve, as in IPC-family methods, so it is flagged in our review under US 12,216,969 and its pending continuation;
not evaluating an energy value, or renaming the search objective, does not by itself change that. Form E is
explicit, with no minimisation, and is flagged under the pending continuation's force-based wording. Rules kept for
both because of other patents the search found: bound the time step rather than clamp points (US 7,616,204), and
stay synchronous (US 9,020,787 concerns a divergent barrier together with kinetic data structures and asynchronous
variational integrators, none of which are used). These flags were accepted for testing. Neither form is in the public code.

### B.8 A field through the environment as the contact sensor

**Evidence state:** proposed; not built. **Date:** first stated 11-12 about 15:55; prior-art check 19:56.

The idea (a touch-screen analogy): a value held in the space between bodies reports where anything
approaches anything, so candidate pairs are found without pair searches. It is a sensor only; the response stays with
a contact law (B.6, B.7). The planned build-out: one value per node of the *existing* sparse lattice in the air region,
for example the distance to the nearest other body, read from the bodies' own distance fields and refreshed only in
tiles where something moved. The pass marks were written before any run: recall of every contact row the per-pair
search finds (0 missed); sensor cost at rest no more than 10 % of the reference search; cost when one body moves no
more than the reference; memory stated. Known problems: self-contact (surfaces of one body need a label-free signal
such as the local thickness of free space); a gap narrower than a cell vanishes from the grid, so the sensor must hand
over to per-body fields before that; a global update is itself a cost to measure.

**The air-mesh check.** The closest published relative found is Air Meshes (Mueller, Chentanez, Kim and Macklin,
SIGGRAPH 2015): the air between bodies is tessellated, with a non-inversion constraint per air element. A keyword-level
search found no patent on air meshes or on tessellating empty space for contact, and no open release from its authors'
organisation. That search is **not conclusive**: two full-text patent search sites were blocked to the search, and some
related grants were seen by bibliography only. What our sensor will not do: (1) tessellate empty space into elements;
(2) use a non-inversion constraint per air element as the contact rule; (3) maintain a mesh's connectivity as bodies
move. It holds values on lattice nodes, with no elements and no connectivity. Mechanisms named as candidates to borrow
for updates are incremental distance transforms (FIESTA, VDB-EDT) and the sleeping of resting regions used by rigid
body engines.

### B.9 The elbow study: why the bend stalls near 85-90 degrees

**Evidence state:** measured on named cases (a synthetic stand-in arm); the cause is **not established**.
**Dates:** 11-12 11:00-20:00.

**The case ("E1").** A stand-in arm of cones and ellipsoids with two capsule bones (radii 12 and 11 mm) and a hinge on
the limb's centre axis; the layer stack of B.3 as one bonded solid; a 4 mm lattice refined from 8 mm near the joint;
bent 0 to 90 degrees in 2.5-5 degree steps from a dual-quaternion skinning guess, with skin-on-skin rows (B.4).

**Angles.** On the stand-in arm, angles are measured from its straight rest pose. The human model's elbow rests bent
22.2 degrees, so for it "from rest" plus about 22 degrees gives the anatomical angle. The target for a full bend is
145-155 degrees anatomical.

**What stalls.** The baseline hits its 80-iteration cap from 82.5 degrees and its 85-90 degree states are unconverged
(residual 1e-5 to 5e-5); R1 converges to 88.44 and Form N to 88.75 degrees and no further. Three contact laws stalling
at similar angles suggests a shared upstream limit, but the runs also share geometry, material, discretisation and
parts of the solver, so that is not yet a diagnosis.

**Measured.**

- *The lattice folds.* Skin samples sink into the body's own neighbouring tissue with no skin pair crossed, and the
  onset moves with cell size: from 55 degrees at 8 mm cells (0.55 mm, growing to 2.6 mm), from 80 degrees at 4 mm cells
  (16 um, then 0.12-0.24 mm). It is the same under all three laws. (B.4's earlier run, with different sampling, saw
  0.27 mm from 72.5 degrees; the records do not reconcile the two figures beyond the difference in setup.)
- *The tissue is crushed.* At 82.5 degrees 124 fold-barrier points are active, with the local volume ratio down to
  0.032.
- *Not volumetric locking.* From saved baseline states, all volumetric terms together are under 2 % of the elastic
  energy (21 of 1,136 mJ at 90 degrees), and the energy grows smoothly through the stall. An earlier statement that the
  cells lock everywhere was corrected the same hour.
- *Where the energy sits at 90 degrees:* 44 % in the front crease (505 mJ in 132 mL) and 22 % in a 22 mm plug of soft
  tissue between bone ends that do not meet (245 mJ in 112 mL).
- *A reversal.* A first finding blamed the hinge's position: rotating the forearm rigidly forces 34.5 mL of overlap at
  45 degrees and 75.7 mL at 90, and a hinge 20 mm further forward would cut that five-fold. Measurement on the Visible
  Human Female's left elbow (U.S. National Library of Medicine cryosection data, 0.33 mm in-plane, 1 mm slices; used
  for study and measurement only, not shipped) put the flexion axis, picked through the trochlea and capitulum to
  within 1-2 mm, 35.8 mm from the back skin and 63.8 mm from the front: **36 % of the depth from the back**. The
  stand-in hinges at 50 %, so it already has less tissue in front of the axis than a real arm, and the "move it
  forward" reading was withdrawn. Caveats: one specimen, heavier than the model.

**One-change experiments (marks written before each run; baseline solver, 80-iteration cap, tolerance 4e-6).**

| experiment | change | verdict |
|---|---|---|
| d1 (19:42) | bones that articulate at the axis: a 10 mm condyle along the axis, a notch sector at 11-16 mm radius that turns with the forearm | **No on every mark.** Newton reaches the cap from about 80 degrees (the baseline: 82.5) and is worse below it; the joint core's share rose from 22 % to 37 % (520 mJ). Tissue bonded to both bones is sheared over a few mm. The lesson recorded: articulation needs sliding |
| d4 (19:51) | fat modulus 26.1 -> 9 kPa (shear about 3 kPa, after a search-level reading of in-vivo elastography; a lead to verify in full text) | **No improvement at the stall.** Energy at 90 degrees 1,185 against 1,136 mJ; Newton at the cap from 75 degrees |
| d3a | a thin soft slip layer at the deep fascia | a switch is built; no result on record |
| d3b | 2 mm cells in the crease zone | marks declared (every step below the cap, no substeps); not run on record |
| d2 | the axis at the anatomical depth | not run |

After d1 and d4 the one lever measured to move the stall is cell size at the crease, so the leading candidate is the
crease's discretisation: a 4 mm trilinear cell cannot fold into a crease a few mm deep. It remains a candidate.

**On the real arm ("E2").** By decision (11-12 about 19:50) the test moves from the stand-in to an
isolated elbow region of the human model's own field, with bones placed from anatomy (committed 19:56). Measured since (commits at 20:13 and 20:17): with the
stand-in's simple bones and skin rows, the baseline converged every state from rest to 90 degrees from rest (about
112 degrees anatomical), but with Newton at its iteration cap from 67.5 degrees from rest, four substeps a step from
82.5, the predictor's starting guess infeasible from 80, and 3 skin samples 0.35 mm inside tissue at 87.5-90. So
the real arm bends further than the stand-in before stalling, and it is still far from the 145-155 degree target. A
run with anatomical, articulating bones was in progress when this was written and is not a result.

### B.10 Store seams, the seam guard and the publish gate (brief; the store has its own section)

**Evidence state:** measured on named cases; in the live writer. **Date:** 11-12 09:06-10:39 and 18:47.

Contact work exposed two store faults. *Seams:* samples on a face shared by two bricks disagreed (108,237 level-0 seam
nodes in the model, 169 with opposite signs); fixed at every level to 0. The cause was in the writer; five real brush
strokes written back gave 4,053 level-0 seams with the old writer and 0 with the fixed one. A guard at commit now
writes every changed face sample into every brick that holds it, and a gate refuses to publish a version with seams or
duplicate rows. *Sides:* after carving a cavity, a closest-point-and-normal sign test flipped air to tissue in a thin
gap beside it (91 wrong samples at level 0 in one eye test). The fix takes a rebuilt sample's side first from a
neighbouring target node whose distance exceeds their separation (a distance field cannot have a surface between
them), and only then from older rules: 0 wrong-way flips in the same test; engine tests 13 of 13.

### B.11 The persistent field, kept small

**Evidence state:** proposed. **Date:** first stated 11-12 about 16:50; scoped 11-12 about 19:50.

The larger idea (state such as temperature or moisture held in the environment and advanced when its dependencies
require it) is described in Part 2, section 3. In contact it appears only as the sensor of B.8, and its
scope is recorded as decided: the form is left open (no commitment to an air mesh, nor to static or moving
fields), and it is not to be expanded further for now, because the base has to be built before anything else is
possible.

### B.12 Summary of states

| mechanism | state | where it failed or is open |
|---|---|---|
| Parts, bands, entries, pair settings | measured on named cases | whole-body memory is an estimate; pose-time sheets not measured |
| Solid by default, designed cavities | measured on named cases | a few sub-microlitre specks remain; no general pocket guard |
| IPC-on-fields solver | measured on named cases; baseline only, not public | patent reading B.5; stalls at the elbow |
| Skin-on-skin rows | measured on named cases; not landed | own mark not met (lattice fold) |
| R1 constraint solver | measured on named cases; prototype | times out on broad soft-on-soft contact; pending continuation |
| Magnetic-field contact, Form N | measured on named cases | reversal on soft-on-rigid; cheek scene did not converge; patent flags |
| Magnetic-field contact, Form E | measured on named cases | fails friction; 2.4-11 times slower than the baseline on the scenes in the table |
| Environment-field sensor | proposed | not built |
| Elbow study | measured on named cases | cause not established |
| Seam guard, publish gate, safe sign | measured on named cases | see the store section |
| Persistent field | proposed | scope deliberately small |

---

# Part 7. Posing, cloth from measurements, and the mouth library

This part describes three subsystems of Field Studio as they stood when their work sessions were paused on the morning of 12 November 2026. Everything here works on the same representation as the rest of the tool: a form is a signed-distance field sampled on a sparse lattice with several levels (0.25 mm, 0.5 mm, 1 mm and 2 mm cells in the human model's store). There are no triangle, quad or tetrahedral meshes anywhere in these subsystems, including in tests and previews.

**How to read the dates.** Times in this part are internal provenance times, Pacific time, taken from the project's local commit history and work records (short commit ids are given so that the private history can be matched later). They are not the publication date. The publication date is the date of this release.

**Evidence states.** Each mechanism carries exactly one of: *proposed*, *implemented*, *measured on named cases*, *reproduced under a frozen configuration*, *accepted in the workflow*. Nothing in this part reached "accepted in the workflow" for the human model's published form; where a mechanism landed as a menu entry in the tool and passed its window tests, that is said in words.

**Machine.** All numbers come from one Windows PC with 32 GB of RAM and one 8 GB NVIDIA RTX 5050, shared by several work sessions at once, so CPU timings include contention.

**Patents.** Patent readings below are our own reading of the claims, not legal advice, and no clearance is claimed. Legal status, fee payments and disclaimers of the patents named were not verified.

---

## C1. Posing

### C1.1 The layers (adopted 11 Nov 2026, 16:45; design record)

A pose is always the rest form plus a deformation, rebuilt from rest every time. Rest is never written by a pose. The design has four layers:

1. **Skeleton.** Joint angles give each bone a rigid transform.
2. **Skinning map.** A smooth map from rest points to posed points, blended from the bone transforms by weights.
3. **Correctives as a rules map.** Per joint, the tissue around the joint is solved once by a contact-aware soft-body solver at sampled joint angles, stored, and looked up afterwards.
4. **Live contact** where no stored map can know the answer (a hand against a cheek). *Proposed*; not built in this subsystem.

### C1.2 Skeleton: joints and bones as data, drawn as curves (bebc4f23, 11 Nov 16:52)

*Evidence: implemented; measured on named cases (synthetic arm E1).*

A joint has a rest centre `c`, a rest frame `F` (3x3, columns are the joint's anatomical axes) and one to three rotational degrees of freedom with ranges in degrees. The bone a joint moves carries

`T_j = T_parent(j) o Rot(c_j, Q_j)`, with `Q_j = F_j R_j F_j^T` and `R_j` the product of the axis rotations.

Frames are orthonormalised with the primary axis kept exactly. This was added after a real fault: an elbow frame whose second axis was not perpendicular to the flexion axis made `F R F^T` a non-rotation, and the skinning map then disagreed with the bone's own transform by up to 10 cm.

Bones are line segments (capsules in the bench) and joints are points; in the tool they are drawn as curves and rings over the traced image. A pose is saved as angles only (a test skeleton is 769 bytes, a pose 63 bytes).

**Coupled joints** (67a598ba, 12 Nov 01:44): a driven degree of freedom follows a driver through a ratio or a piecewise-linear table. A finger's distal joint follows its middle joint at 0.67; tests K1-K6 pass. A shoulder girdle (sternoclavicular and acromioclavicular joints, with the shoulder joint expressed relative to the shoulder blade) was added to the human template (80a12cca, 12 Nov 02:15), coupled to arm elevation by straight-line fits to the start and end values reported in Ludewig et al., "Motion of the Shoulder Complex During Multiplanar Humeral Elevation", J Bone Joint Surg Am 2009;91(2):378-389 (open access, PMC2657311). Fitted slopes per degree of humerothoracic elevation: scapulothoracic upward rotation 0.31; sternoclavicular posterior rotation 0.25, retraction 0.13, elevation 0.05; acromioclavicular posterior tilt 0.15, upward rotation 0.09. These are estimates: the paper's per-angle tables were not obtained. *The girdle is implemented as data and coupling only; no girdle deformation has been solved.*

Joint ranges in the template are placeholders marked as estimates. The intended sources (published normal ranges of motion) were not logged against the values, so the ranges are **not established**.

**Pose session** (67adc08a, 11 Nov 19:33; *implemented*, its six tests pass on synthetic cases): pick a part by ray-marching the posed field, pulling the hit back to its rest point and taking that point's heaviest bone; drag rotates the part about its own joint, projected on the joint's degrees of freedom and clamped to its range; "reach" moves a grabbed point to a target by damped least squares over the chain's angles with ranges, pins and a prior that keeps other joints near their last values. None of this is wired into the Field Studio window yet.

### C1.3 Finding joints from the field (e1b47134, 11 Nov 17:20)

*Evidence: measured on named cases (the human model's left arm and hand).*

1. Start at a point and direction on a limb. Take the surface samples in a thin slab perpendicular to the current axis, within a radius of the previous section.
2. Centre the section at the midpoint of its extents along two principal axes; turn the axis toward the new centre; record centre, axis and two half-widths.
3. Stop where the section vanishes (a tip) or exceeds 1.5 times the limb's own starting width (it has merged into a larger body).
4. Joint cues are extrema of the smoothed section-size profile: narrowings between two masses (elbow, wrist) and widenings (knuckles).

Results: the forearm line is clean, with narrowings just above the elbow (37 mm section) and at the wrist (25 mm). The four finger lines are clean (71-92 mm). Knuckle widenings are only 0.4-1.3 mm over the finger's taper: the middle finger joint was found on all four fingers; the distal joint on two of four. The walk fails at the shoulder (it enters the torso), and the thumb was not done. A whole-body skeleton from the field is **not built**.

### C1.4 Skinning weights: volumetric bone heat on the lattice (902c121a, 11 Nov 17:17)

*Evidence: measured on named cases (E1; the left arm of the human model).*

**Ingredient credited:** bone heat from Baran and Popovic, "Automatic Rigging and Animation of 3D Characters" (SIGGRAPH 2007, the Pinocchio system). The method was taken from the paper; no code was copied. The original solves heat on a surface; here it is solved in the solid's volume on the lattice.

Steps:

1. Build a boolean "inside" lattice of the rest solid at spacing `g` (1-2 mm in the tests). Inside and outside come from the field's own signed samples; a region with no samples counts as outside only if it connects to a sample measured outside. Nothing is seeded from the bounding box.
2. For each inside voxel compute `d`, the straight-line distance to the nearest bone segment, and `H = c / d^2` (default `c = 1`).
3. For bone `b`, set `p_b = 1` at voxels whose nearest bone is `b` and visible from it in a straight line inside the solid, else 0.
4. Solve `(-Laplacian + H) w_b = H p_b` over inside voxels, with zero flux at the skin. The Laplacian couples inside voxels only, so weight never crosses an air gap (one finger does not pull its neighbour).
5. Drop weights under `1e-3`, renormalise, keep the top 4 per voxel (int16 ids, float32 weights). Far from a joint one weight is then exactly 1, a rigid zone.
6. Look up at a rest point by trilinear interpolation over 8 voxels, top 4, renormalised.

Solver: one sparse LU then a solve per bone up to about 400k voxels (140-160 s on the CPU for 348k voxels); conjugate gradients with a diagonal preconditioner above that (peak 3.5 GB). **Failure recorded:** a sparse LU on 811k voxels grew to 27 GB of RAM and was stopped by hand; the size switch exists because of it. A whole body at 2 mm (estimated 7.5 M voxels) has not been solved.

Measured on E1 at 90 degrees of elbow flexion, against the geodesic weights of C1.9 at about equal rigid share:

| weights | rigid share | folded tissue | volume change |
|---|---|---|---|
| bone heat, c = 1 | 46 % | 0.018 cm3 | -0.18 % |
| geodesic, zone +-1.5 limb radii | 47 % | 6.7 cm3 | -1.8 % |
| bone heat, c = 4 | 69 % | 4.6 cm3 | -0.05 % |

On the model's arm, bone heat leaves only 55 mm of the 256 mm forearm exactly rigid.

Bounded biharmonic weights (Jacobson et al. 2011) are named as a second candidate. *Proposed*; not implemented.

### C1.5 The skinning map and its inverse (bebc4f23, 11 Nov 16:52)

*Evidence: measured on named cases (E1).*

**Ingredient credited:** dual quaternion skinning, Kavan, Collins, Zara and O'Sullivan (2007/2008). Method from the papers, no code copied.

`phi(X) = DQ( sum_k w_k(X) s_k dq_k )` applied to `X`: weights are read at the rest point, each bone's unit dual quaternion is sign-aligned to the dominant one, the sum is normalised and applied. An all-identity pose returns `X` with no arithmetic, so rest is bit-exact.

A field is posed by pull-back: `F_posed(x) = min over preimages X of F_rest(X)`. The inverse is damped Newton on `phi(X) = x`, started from each candidate bone's own rigid inverse; every converged preimage is kept, so a fold (two rest points landing on one posed point) is detected rather than hidden. This CPU "twin" is the reference for every faster route.

Tests T1-T6 on E1 (two analytic capsules with a smooth elbow): rest bit-exact; rigid zones exact to 1e-16 m; inverse converged for 99.95 % of band points to 1e-10 m; 100 flex cycles bit-identical; save and reopen bit-identical. Inverse cost 0.7 ms per point in NumPy.

### C1.6 What failed first: skinning alone

*Evidence: measured on named cases (E1; the model's left arm, case E2; the hand, case E4).*

- **Zone-width sweep on E1:** with a joint zone of +-0.6 limb radii, 76 % of the arm is exactly rigid but 20.9 cm3 folds at the inner elbow at 90 degrees. At +-3 radii, 16 % is rigid, 0.2 cm3 folds, and 1.3 % of volume is lost. No width removes the fold.
- **Interactive cost:** resampling the posed band from rest each frame failed the 33 ms target written before the runs: 70-150 ms per update at UHD on the model's arm. A profile showed 8-16 ms at 10 degrees and 65-99 ms at 60-90 degrees; the cost came from samples in the compressed crease needing Newton starts from every bone.
- **The 90 degree picture was not clean** (11 Nov ~19:50): a dark crease lip and brick-shaped broken patches. The per-sample reference path showed the same, so it was the skinning fold (determinant of the map below zero), not the renderer.
- **A fist at the finest level (E4):** 1.2-3.9 s per update; zero set within tolerance only to half a curl.

Three rendering routes were benched on the same map: tracing curved rays in rest space (after Seyb et al. 2019, "Non-linear sphere tracing for rendering deformed signed distance fields"), resampling the moved band each frame, and a hybrid with rigid segments drawn as transformed volumes over the same rest bricks. Resample and hybrid matched in correctness outside the fold (zero sign errors on E1 at 0-90 degrees). **No route decision was recorded as final.**

### C1.7 Correctives: the joint zone solved once over its angle (ec35a575, 11 Nov 19:57 onward)

*Evidence: measured on named cases (E1 to 60 degrees; the model's left elbow to 55 degrees). Not established past those angles.*

The idea follows pose-space deformation (Lewis, Cordner and Fong 2000), with the corrections solved by physics instead of sculpted.

1. Cut a zone around the joint. For the model's elbow: 120 mm above and 140 mm below the joint, within 70 mm of the limb line, plus a 70 mm sphere at the joint. The sphere was added after the first zone missed tissue at the back of an elbow that is already bent 22 degrees at rest, leaving skin fragments in the picture.
2. Put a two-level displacement lattice on the zone: 4 mm cells near the crease, 8 mm elsewhere (7,254 nodes for the model's elbow).
3. Treat the zone as a soft body in the project's incremental-potential-contact solver (Part 6, section B.3), with layered tissue materials and self-contact on. Bones are capsules: nodes in the proximal bone stay at rest; nodes in the distal bone and at the distal cut follow the joint's rigid transform.
4. Gravity is on. With it off no step converged. The 0 degree state is solved too and corrections are taken relative to it, so rest stays exact.
5. Step the angle by 5 degrees to 60, then 2.5. Each step starts from the last solution plus the skinning map's increment and is solved statically. A step whose start is infeasible or that fails to converge is retried in halves. Each state is saved as it lands.
6. At draw time the correction is blended linearly between solved angles on top of skinning. A posed point is mapped to rest by a lookup in the solved lattice: deformed cell boxes are binned on a grid, an exact box test selects candidates, and Newton inverts the trilinear cell from its centre (and from the tissue centroid in folded cells). Outside the zone the proximal side is unmoved and the distal side is rigid, each taking over 10 mm past the cut.

**History of walls, kept because each changed the method:**

- First sweeps on E1 stalled near 60-75 degrees. Cause found: lattice cells cut by the skin had quadrature points in empty space, and the empty part of the cell was crushed as the crease closed (local volume ratio 0.65 at 40 degrees, 0.07 at 67.5). A local stand-in moved each cut cell's weight onto its in-tissue points and reached 70 degrees. The stand-in was then retired when the solver itself gained cut-cell quadrature (tissue points only; the empty part held by a weak barrier-free stiffness, 0.1 x mu |F - R|^2).
- A self-contact exclusion radius of 20 mm, compared between cell centres, hid 0.54-0.64 mm skin overlaps at 45-50 degrees. The radius is now 0.5 mm, judged at located rest preimages. States solved under the old radius were discarded and re-solved.

**Current numbers.**

- E1, synthetic: converged 0-60 degrees; 62.5 degrees at residual 2.3e-4; 65 degrees fails. (This is the posing work's corrective setup of 11-11 and the early hours of 11-12. The contact study in Part 6, section B.9, runs the same stand-in arm with a different lattice, skin-on-skin rows and solver settings on 11-12, and gets to 88-90 degrees; the two are not the same experiment.)
- The model's elbow: converged 0-55 degrees at residual <= 9e-7; zero contact rows and zero skin-inside-skin at 0.5 mm from 30 to 55 degrees. Judge at 55 degrees: 0 pockets; seams at the zone cuts 2.4 um (distal) and 0 um (proximal); 0.39 % of interior points claimed twice, as 0.2-1.3 mm folds 7-26 mm under the skin.
- At 60 degrees the start is infeasible: skin folds onto skin at the crease apex below cell size. This waits on a skin-on-skin self-contact piece in the solver that had not landed.
- On E1 at 70 degrees (earlier stand-in states): 0 folded pixels in the flexion-plane section with the corrective, against 325 with skinning alone.

**Known defects at 55 degrees, close view:** terraces 1 mm apart in the crease (cause: trilinear kinks of the rest grid mapped through the compressed crease; a tricubic Catmull-Rom rest read is written and CPU-tested but was never rendered); a pit with a nub beside the crease and a small bump on the silhouette, cause unknown. Only elbow flexion is solved; forearm rotation is not. The goal of a clean elbow at 90 degrees was **not reached**.

### C1.8 Drag cost through the corrective (12 Nov 01:58 and ~04:00)

*Evidence: measured on named cases (the model's left arm, 10-55 degrees, clay shading).*

Per frame, median: node update about 4 ms on the CPU; posed-band preparation on the card about 2 ms; resample 4.5-11.8 ms for about 4,000 posed bricks at 1 mm; clay trace 1.0 ms at 1080p and 3.6 ms at UHD. Frame median 16.5 ms at 1080p and 19.1 ms at UHD; 90th percentile 30.8 and 33.3 ms after buffers were made persistent (it was about 60 ms before). The card's zone samples had 0 sign mismatches against the CPU twin over 3,318 points. These timings were taken on states from the retired stand-in; they were not re-timed on the final states. A practical note: card code avoids matrix-multiply library calls because the machine's CUDA install lacked that library.

### C1.9 Geodesic voxel binding: studied, not used, and why

*Evidence for the geodesic weights: measured on named cases, bench and study only.*

A first weight module computed, for each bone, Dijkstra distances through the 26-neighbour graph of inside voxels, with a smoothstep falloff scaled by limb radius (after Dionne and de Lasa 2013, "Geodesic voxel binding for production character meshes"). On 11 Nov at about 16:55 a patent check flagged it, and it was decided that it would be studied only. The claims were then read (about 17:45) from the USPTO's public page scans, by eye.

**US 9,836,879 B2, "Mesh skinning technique" (Autodesk).** As we read them, independent claims 1, 10 and 16 each require all of:
(a) a voxelization of a 3-D model with interior voxels;
(b) traversing several paths through interior voxels from a voxel that intersects an influence (a bone) to a voxel that intersects a target vertex of the model;
(c) a distance value per path;
(d) a path through a boundary voxel has its distance increased by a multiplicative weight;
(e) the shortest path determined;
(f) a skinning weight assigned to the target vertex from that shortest distance;
(g) an animation generated from the weight.

*What our default (C1.4) lacks, on our reading:* (b), (c), (d), (e) and (f). No path is traversed and no path distance or shortest path is computed; each bone's weight is the solution of one linear diffusion equation. There is no boundary-voxel penalty and no vertex: weights live on the lattice and are read at rest points. The only distance used is the straight-line distance to the nearest bone. What it shares: a lattice over a solid (part of (a)) and animation (g).

*Our own geodesic bench module* was flagged in our review and is study-only; it is not used in the tool and is held out of the code.

**US 10,062,203 B2, "Voxelization of mesh representations" (Autodesk).** As we read them, the independent claims require voxelizing a mesh representation, choosing a voxel at the grid's exterior boundary and marking it exterior, propagating that marking, and generating a volumetric model. Our inside/outside classification starts from no mesh, and nothing is seeded at the grid border (C1.4 step 1). An earlier version that flooded from the box faces was closer to the claim's flood step, still with no mesh; it was replaced.

Searches on 11 Nov found no patents on bone heat, bounded biharmonic weights, dual quaternion skinning or Seyb's curved tracing. Those searches were not exhaustive. All of the above is our own reading, not legal advice, and no clearance is claimed.

### C1.10 Muscles and bones as field layers (plan, 11 Nov 20:30)

*Evidence: proposed.*

It was left open that any work session may build bones and muscles as field layers, whole or partial, from medical measurements, if posing tests need them (the shoulder girdle and muscles that bulge with flexion come first). The body's outer surface stays the target; the layers would sit under it and be kept consistent by contact. Nothing was built. Muscle bulge is planned as an activation state of a muscle layer inside the corrective solve.

---

## C2. Cloth from measurements

### C2.1 Measurements taken from the field (a794d4b9, 11 Nov 20:06; rules revised to 12 Nov about 05:20)

*Evidence: measured on named cases (the human model; a mannequin assembled from primitives; a dress form).*

A garment needs only a body's signed-distance field (`phi`, gradient). Measures:

- **Girths.** Horizontal section loops of the zero set at each height; the torso loop is the longest closed loop near the midline; the tape is its convex hull. Waist is the narrowest; underbust sits just below the steepest rise above the waist; bust is the fullest above that; hips the fullest below the waist.
- **Bust points.** Front-facing surface only. Apex: the forward-most point after smoothing over 21 mm. A finer tip is accepted within 30 mm if it stands at least 0.3 mm proud.
- **Neck.** Going up from the shoulder row, the first narrowest section before the tape grows 12 % (into the head).
- **Side neck point.** The concave corner of the front outline where the neck meets the shoulder slope, placed at the neck base's mid-depth.
- **Shoulder point.** Outward along the shoulder line from its flattest point, the first place the slope passes -1.
- **Arms.** Their own section loops give an axis (line fit), the armpit height (highest arm loop) and the biceps girth.
- **Taut tape between two surface points.** Start from the straight chord with inside points pushed to the skin; repeatedly move each point to its neighbours' midpoint and push any point that went inside back out. The path lies on convex skin and bridges hollows.

**Failures that shaped these rules.** The first rules put the side neck point 75 mm low and the bust points about 60 mm off at the silhouette, on both the model and the dress form, so a garment was drafted too small and sat too low. A height-window neck rule took the top of the head on a full body. The shared section code returned points hundreds of metres away where the cutting plane was nearly parallel to the surface, producing a "195 m shoulder row"; loops with points outside the body's box are now rejected, and the shared code was fixed separately. The first time the tool ran in the real window it placed the top at the neck; after the fixes two window reruns reproduced the scripted waist and bust girths to within 0.2 mm with the garment in place.

### C2.2 Cloth representation: yarn points on a pattern lattice

*Evidence: implemented; unit tests pass (energy gradient agrees with finite differences to 4e-11; hoop strain of a band 11.89 % against 11.77 % analytic).*

A panel is one pattern piece. Yarn points sit on a regular (u, v) lattice in the flat pattern: u along the course or weft, v along the wale or warp, spacing 3 mm for the swim top and 8 mm for the tee. Points inside the drafted outline are kept and the outermost row is moved onto the outline, so a cut edge follows the drafted curve.

- **Rest state** is always the flat pattern. Placing a panel on a body only gives the solver a start. The difference between the two is the garment's ease or tension.
- **Energies** (from the contact solver's cloth sheet): a two-branch J-curve stretch along each lattice direction (stiffness `k_low` up to an elastic limit, a stiffer branch beyond), shear from the angle between lattice lines, bending from second differences along lattice lines.
- **Contact** uses lattice cells as bilinear patches of four yarn points, against the body's field, with an IPC barrier and friction by material pair. No triangles.
- **Seams and ties** are stiff zero-length springs between yarn points (default 100 times the panel's segment stiffness). Two cut edges are matched by pattern arc fraction, each node tied to a point between two nodes of the other edge. Bound edges are tension-only springs with shortened rest length.
- A lined, two-layer panel is one sheet with stretch, shear and mass doubled and bending times four (an estimate).

### C2.3 Fabric profiles and their sources

- **Swim tricot** (new profile): 193.6 g/m2 and 0.593 mm, the mean of four 78-80 % polyamide swim knits from an open dataset logged as CC BY 4.0. Stretch 121/124 N/m below a 105 % elastic limit, the stiff branch 15 times that, bending and shear: all estimates scaled from a different knit, because the tricots' curves were published only as figures.
- **Cotton single jersey:** KES-FB values from Ramaiah, Radhalakshmi and Nachane (fibre2fashion article; no open licence stated, numbers used as facts): bending 0.0066 gf cm2/cm, shear 0.47 gf/(cm deg), thickness 0.919 mm, 125 g/m2. Skin friction static 0.90, dynamic 0.79 from an open CC BY 4.0 paper on jersey against chest skin. No wale/course split was found in open sources.
- **Construction rules:** FreeSewing patterns "Bee" (triangle top) and "Teagan" (tee), MIT licence, design defaults only; published size charts for tee proportions; trade sources for swimwear negative ease (15 % crosswise, 5 % lengthwise).
- **Reference photographs:** ten photos under the Pexels licence and Wikimedia Commons, each logged with author and licence.

### C2.4 Settle to rest (b84a0de0, 11 Nov 21:08)

*Evidence: implemented; measured on named cases. Reaching true rest on the tee is not established.*

1. Carry each panel from its drafted place onto the body by rays toward the interior, stopping just outside the skin. Lift any point closer than the collision offset out along the field gradient; the barrier needs every gap above zero.
2. **Close seams in the start state:** move tied nodes onto their targets and fade the move over 30-60 mm of pattern. This exists because of a failure: with stiff ties stretched across a start gap, the contact barrier was squeezed to nothing and the collision step bound held every step at zero.
3. Solve quasi-statically in short chunks of a few Newton steps. Each chunk is a new load step with friction re-lagged; state is saved after each chunk.
4. For the tee: sew with gravity off and no friction until every seam gap is under 2 mm, then run final chunks with gravity and friction. Between chunks, yarn points within 20 um of the body are lifted.
5. Stop at rest (two chunks under 2 um of movement, or energy change under 1e-5) or at a cap.

Body tissue is rigid in all cloth runs. Soft tissue yielding under a band is **not done**.

### C2.5 Garments

**Triangle halter top (swim knit).** Each cup is a flat triangle solved from three taut-tape arcs on the body (base along the underbust, two edges to the apex), each shortened by the ease. The cup's top sits at 54 % of the side-neck-to-bust line. The band and neck tie are strips; cup bases are tied to the band and cup tips to the neck tie.

- v1 settled in 464 s on the CPU with seams at 0.05 mm.
- v2 and v3 drifted loose: a 1.35x gather at the base cancelled the 15 % reduction, leaving cross tension near zero.
- v4 ("snug"): about 7 N/m tension across, median contact pressure 245-273 Pa.
- Run through the tool's own menu path on the model it reproduced v4: cup position differences median 1.3-1.6 mm, 95th percentile 4-5 mm, seams 0.16 mm.
- On a fresh mannequin built from primitives, landmarks, draft, settle and save completed, and the Garment panel passed 6 of 6 checks in the real window (one run 854 s at 99 % CPU).

*Evidence: measured on named cases; the garment panel landed in the tool.*

**Crew-neck tee (cotton jersey).** Front and back: chest = bust x (1 + ease), hem from the hips, straight side seams, shoulder seam from the side neck point to just past the shoulder point, scooped armhole, crew neckline. Short sleeves with the cap height solved so the cap curve equals the armhole length. Start: body panels on the torso's hull loops inflated by 15 mm, sleeves as open tubes about the arm axes.

Faults found and fixed in order:
- Armhole seams started 95-217 mm open and crossed. Fix: place the panels' side edges at the arm junction, and close every seam in the start state (max gap then 0-5.7 mm).
- Lumps under the bust: the start shell followed the waist and squeezed a straight tee 7-30 % across. Fix: below the bust the shell is the hull of the waist loop and the bust loop, so the front hangs from the bust.

Version 13 sewed from 5.7 mm to 0.23 mm and ran 1605 s with a 6.25 GB peak (the export). Version 14 has seams at 0.14 mm and no stretch below the bust.

**What is not established.** No tee version is at rest. After its 30 final chunks v14 still moved 2-4 mm per chunk; continued for 200 more chunks it was still moving 0.64 mm per chunk with energy falling. A variant that carries rest lengths gradually from a zero-strain start stalled with yarn points caught in narrow concave creases of the body field (residual 5.6 % of the load). A proposed remedy, colliding at the fabric's real half-thickness of 0.406 mm instead of 0.31 mm, never ran. Folding near the armholes exceeds the body's own curvature by about 26 degrees (front) and 23 degrees (back) in every version; whether that reads as natural drag lines was not judged. The tee is not available through the tool (the call raises "not implemented"). Fine moire on the jersey in renders is an aliasing fault in shading. Full-colour renders of cloth on skin were not made.

### C2.6 The garment as a field (b84a0de0; banded stores d9c866a9, 11 Nov 22:38)

*Evidence: implemented; measured on named cases.*

A settled sheet becomes a solid: `F(x) = distance to the midsurface - thickness / 2`. The midsurface is sampled about 0.15 mm apart on Catmull-Rom patches through each cell's 4x4 yarn neighbourhood (bilinear at panel edges), with normals. Distance is taken to the nearest sample; its along-normal part is exact. The result is written as a sparse store in the body's brick format at 0.25 mm, with bricks only in a band around the cloth, so it is traced, measured and cut like any other object. A midsurface check read -0.27 to -0.29 mm against the expected -0.30. The tee's export is slow and heavy: about 15 minutes, about 6 GB peak, 1.4 GB on disk for four panels.

### C2.7 Dress form from a model (first 11 Nov ~23:35; runner 48ef3234, armhole plates 26e56338, 12 Nov 07:42)

*Evidence: measured on named cases; the menu entry landed in the tool (window tests 5 of 5, failure cases 3 of 3).*

An earlier mannequin sculpted only from measurement points was set aside as likely to cause fitting faults, so the form is now derived from the model's own field:

1. Sample the body at 2 mm; inside/outside from the body's own reader.
2. Remove each arm at a flat plane through the shoulder point and the armpit.
3. Convert to a true distance by a distance transform; smooth with a Gaussian (sigma 6 mm; a sphere of radius 60 mm moves about 0.3 mm).
4. Cut flat at the neck and below the hips with a smooth maximum (3 mm rounded edge); keep one connected body.
5. Write as a store down to 1 mm.

On the human model: 310 s, 4.5 GB peak, 37.94 L. A hollow body or one with no torso is refused with a plain message. **Fit test:** the tool's top settled on the form against the same top on the model: cup median distance 1.4-1.6 mm, centroids within 1.6 mm. That test used an earlier form (cylindrical arm cut, finer levels); the armhole-plate form has not been fit-tested.

---

## C3. The mouth library

### C3.1 What it is

*Evidence: measured on named cases; the library menu landed in the tool. The seating in the human model is unpublished working versions only.*

A set of generators that build teeth, gums with alveolar bone and palate, a mandible, and a tongue as field parts, each with its own material, placed from a library menu into any head. The body is solid by default, so the mouth's air (the cavity inside the arch, the opening between the lips, the vestibule between cheeks and teeth) is designed space, and the parts sit in it touching their neighbours without being fused into them. An earlier attempt that stored teeth as material labels inside the single body field drew staircases and wrong colours; that is why they are separate parts. Library version 10 at the pause. Units are millimetres.

### C3.2 Generators (30321345, 11 Nov 20:07, through 4b171721, 12 Nov ~05:20)

- **Teeth.** 14 types. A tooth is a field in its own frame (x mesiodistal, y labiolingual, z along the axis, zero at the labial enamel-cement junction). The crown is a stack of asymmetric superellipse sections along monotone cubic profiles, with the incisal edge or cusps, marginal ridges, fossae and fissures added as structure; one to three roots; an inner dentine surface with three lobes under the incisal edge; a pulp. Smooth maxima round the junctions. The values are approximate distances (correct sign and zero set), re-distanced on a lattice afterwards. Settings: scale, wear, incisal lobes, age, seed. Only the upper central incisor's profiles were fitted to photographs (0.62 mm and 0.36 mm rms on two silhouettes); the other types use profiles drawn by eye with cited sizes.
- **Arch.** A contact curve built from its curvature (one value through the front, easing to a second behind the canine), solved so the built teeth meet the cited inter-canine and inter-molar widths. Each tooth gets inclination, angulation and a height on the occlusal curve. Contacts are closed along the arch from the midline; the lower teeth erupt until they touch. Defaults give proximal gaps 0.00 mm, occlusal gaps +-0.01 mm, overjet and overbite 1.60 mm, both measured from the build rather than imposed. This replaced a version in which the upper central incisors never touched (0.44 mm apart at best), visible as a dark slit in a smile render.
- **Gums, bone, palate.** Alveolar bone as one ridge round the sockets with its crest 1.5-2 mm below the enamel-cement junction; gingiva over it with a free margin about 0.8 mm up the enamel, scalloped margins and papillae from cited values; palatal vault with rugae. Teeth are carved out of both across a 0.2 mm ligament gap. A first version built a wall round each tapering root and left V-shaped slits; that was rebuilt.
- **Mandible.** A body swept along the basal arch with height and thickness profiles, a ramus plate from a 2-D spline outline, condyle and coronoid process, from cited adult sizes. Condyle to incisal point 102 mm (the textbook triangle is about 100). It is a first version, not checked against section images.
- **Tongue.** Free form from cited sizes with a median groove and papillae; its rest form is pressed against palate, teeth and gums so it takes their imprint.

Three-plane sections of the assembled set show 0 overlapping pixels.

### C3.3 Sampling, caching, placement

Each part is sampled on a dense lattice, evaluating fine nodes only near the surface, then converted to a true distance from the zero crossings on lattice edges. The seven-part set is 493 MB, cached by settings. Cold build 199 s (456 s before narrow-band sampling); 11.3 s from cache in the app. Placement puts the upper incisal point 2.5 mm below the meeting line of a head's lips, which are found on the head's own surface. Undo returns the memory on CPU and card (about 1 GB each; verified). While the set is placed, the app holds about 1.8 GB more than before; an attempted fix caused repeated re-uploads per brush stroke and was reverted.

### C3.4 Look

*Evidence: measured on named cases; not judged finished.*

A CPU path tracer carries three wavelengths together through nested media (pulp in dentine in enamel; gum over bone; a wet film). Scattering and absorption per medium are fitted with a plane-parallel Monte Carlo slab model to published colour measurements, including an open CC BY 4.0 paper on dentine and enamel slices (BMC Oral Health, 2021) and enamel scattering from Fried et al., Applied Optics 34:1278 (1995). Whole-tooth yellowness is a per-person setting over the cited in-vivo range (b* 10.1-17; default 12). Gingiva needed two layers to match its reference (0.2 dE, against 7 dE with one layer, which had rendered orange). A fitting error once made dentine bluish and the cervical third lavender; the fit now keeps dentine the more chromatic layer. The incisal edge band is 10.4 L* darker than the middle of the crown, against 12.5 in a living reference photo, but still reads weak by eye. A single tooth still reads uniform inside, and the root has no cementum layer.

### C3.5 Seating in the human model

*Evidence: measured on named cases (two unpublished working versions).*

The parts were written into a copy of the model's store one 24 mm region per process (peaks 2.0-3.1 GB; 68 regions), because one process for six regions had reached 12.1 GB. Checks on the second full build: 0 overlapping nodes in six sections; the model's surface more than 6 mm from every part unchanged (4,260 points, 0 sign flips, 0.0 um). A first build had tooth-bone overlaps up to 0.46 mm at the upper necks, fixed at their cause in the bone generator. Its 66 intermediate store copies used 177 GB of disk before a cap of two copies was added.

**Remaining gap.** The model's sculpted mouth cavity is wider than the dental arch, leaving air between cheeks and teeth: medians 0.3-2.9 mm by region, with a 90th percentile of 10.5 mm at the upper premolars. A prototype that fills the vestibule so that cheek and lip lie on teeth and gums as touching, unfused surfaces filled 12.8 mL on a local 0.3 mm lattice, with no sign change outside the vestibule and gap medians of 0.1-0.3 mm, but left 1.7 mL unfilled. A stricter rule measured a 47.1 mL vestibule, then failed with an out-of-memory error in a distance transform; whether all 47 mL is truly vestibule was not checked. The builder changes for this are untested, and no filled version exists.

**Data and licences.** Tooth dimensions cite a standard dental anatomy textbook's measurement table. Those values were entered from memory and are marked as estimates; they have not been checked against the book. Arch and jaw sizes come from published studies recorded with each value. Incisal lobe prevalence by age is from Alalowi and Al-Jhany (2020). Reference images are 67 Wikimedia Commons files (public domain, CC0, CC BY, CC BY-SA, each logged) and head sections from the U.S. National Library of Medicine's Visible Human Project.

---

## C4. Summary of what is not established

- Elbow correctives past 55 degrees on the human model or 60 degrees on the synthetic arm; any other joint; forearm rotation; fingers with contact; the arm raised overhead.
- Posing inside the Field Studio window, a whole-body skeleton from the field, whole-body weights, sourced joint ranges, and a final choice of rendering route.
- Bounded biharmonic weights, live contact during posing, muscle and bone layers: proposed only.
- A tee at rest; the tee through the tool; soft tissue yielding under cloth; cloth shading; a fit test of the current dress form.
- Cheeks and lips resting on the teeth in the human model; tooth types other than the upper central incisor fitted to measurements; the textbook dimensions verified.
- Legal status of the patents named; completeness of the patent searches.

---

# Part 8. Hair and light

**How to read the dates.** Times in this part are internal provenance times: Pacific time, taken from the project's local commit log and dated lane records (short commit ids are given where one exists). They say when a mechanism was first built or measured inside the project. They are not publication dates. The public date of this text is the release date of the document it sits in.

**How to read the status words.** Each mechanism carries exactly one of: *proposed* (designed, not built), *implemented* (built, runs, not yet measured against a reference), *measured on named cases* (numbers exist for the cases named), *reproduced under a frozen configuration* (the same inputs gave the same output again, bit for bit where stated), *accepted in the workflow* (in daily use and approved on review).

**Machine.** Everything was measured on one Windows PC with 32 GB of RAM and one 8 GB RTX 5050, shared between several work sessions through a queue. Timings taken while other jobs waited are marked where the record says so. "The human model" is the project's one test subject: a female head and body held as a signed-distance field on a sparse multi-level lattice (levels of 8 mm bricks down to 0.25 mm cells).

**Mesh-free.** A hair is a curve: a centreline polyline with a radius and a material frame. No triangles, ribbons, cards or tubes are built anywhere, including in previews and tests. The body the hair collides with and is shadowed by is the field itself, sampled into distance lattices.

---

## D1. Hair

### D1.1 What failed first, and why the design changed

Three earlier attempts are part of the record.

- **A rest-groom rebuild, archived 2026-10-31.** Its fixes were made mechanism by mechanism but judged only on full-head renders, and untested pieces went straight onto the head. It was reviewed as a complete failure. The lesson taken: prove each piece on one lock or one fibre before a head.
- **A built style (2026-11-10 to 11-11).** A groom generator placed locks by a specification: an envelope of layer heights, Poisson-disc lock centres, a smooth held top. It matched a reference photograph's tone and contrast measures at lock and section scale, but on review (11-11 08:14) it read as a mound at the top and front, with locks nearly all the same size and too symmetrical, and a top that looked smoothed with hair spray. It was stopped there. Status of that generator: measured on named cases, rejected.
- **The foundations audit (2026-11-11 ~00:10).** Each foundation under the look was measured or read in code. Faults found and fixed at their causes: (a) every point of a strand used its *root's* skin normal as its outward direction for sky light, so lengths hanging 30 cm below the root were lit by where they grew, not where they were; (b) the key light's size widened the skin's highlights but not the hair's; (c) one roughness value along a fibre that is about three years old at 40 cm; (d) every cut hair tapered like an uncut one; (e) 8 % of points bent tighter than a 12 mm radius; (f) thousands of strands ran through the body (2,177 and 4,961 in two sets) where the body lattice had no gradient; (g) the biggest tool fault, found ~01:10: the 3 mm body lattice was a true distance only within about 10 mm of the skin (the store clamps at ±12 mm), so it read 13 mm where the truth was over 30 mm. Every envelope number taken before that used a faulty ruler. The repair is in D1.9.

After these, the direction became: grow a natural base from causes first, and treat every style as a state applied to it (adopted 11-11 07:57).

### D1.2 Grow once, style as states

*First designed 2026-10-31 (commit 0de30c0f); the natural base on the human model 11-11 10:36. Status: measured on named cases. The base head was first reviewed as a good baseline for the hair (11-11 10:38); the current base variant still awaits acceptance (the review of the latest comparison sheet is open). The styling passes are measured on named cases in the hair work area, not in the tool's window.*

Hair is grown once from its causes: follicle position and exit angle, the fibre's inherited type, stiffness and weight, sebum creeping from the root, water, wear with age along the fibre, and a cut. The result is a rested natural head, saved. A style is a sequence of physical passes on that saved head (brush strokes, finger passes, a water set while drying, product). No pass holds anything once it ends.

**Three stages, three owners.** Hair (follicles, fibre, lengths, cut, sebum, natural wave, growth, the gated rest) produces the grown-head cache. Style (passes, water set, release, final settle, child hairs) produces a styled state. Colour (melanin, pigment along the fibre) can be redone without regrowing.

**Layer files** (`engine/hair/layers.py`). A look is five JSON files: hair, style, colour, environment, simulation. Every setting has exactly one owner layer (longest-prefix rule). A layer file that holds a key another layer owns is refused, so a style cannot re-tune a fibre or a follicle.

**The natural base as built on the human model:** 137,865 hairs at about 225 per cm² on 608 cm² of scalp; 1,023 guide hairs settled as rods; child hairs interpolated; locks formed by wet cohesion and not placed. Acts: a soft side part, one finger-comb, fingers raking wet hair off the face, air-dried. No product, no root lift. Two changes requested on review of the first base were made at their causes: front follicles turned back and to the side (fading over 70 mm inside the hairline) so the hair sweeps off the face, and the cut raised to just past the shoulders. Hairs crossing the face region fell from 2,196 to 265.

**Order of the style stage:** load the grown head (fingerprints checked) → product → brush strokes → hand passes → water set → release every act → release solves with friction anchors kept → final gated settle → child hairs → form → baby hairs → colour → a 0.2 mm push-out from the body.

### D1.3 Follicles and hair kinds by region

*Follicle exit model 2026-10-30; kinds table 2026-11-05 (commit 63d07d7f); follicle map on the field 2026-11-10 (b9ef7045). Status: scalp, brows and lashes measured on named cases; vellus implemented as data, with one test crop rendered and no review recorded.*

**Follicle map** (`engine/hair/follicle_map.py`). Samples on the field surface about 2 mm apart, each with a weight per hair kind, a density and a flow direction in its tangent plane. Follicles are drawn per sample as a Poisson count of density × weight × sample area, placed uniformly in the sample's tangent disc and re-seated on the field, so no grid shows in the roots.

**Exit angles** (angle between skin and hair): mid-scalp 35°; whorl 70° fading as exp(−d/18 mm); occiput 45°; hairline front 15°, temple 10°, nape 25°, blended inward over 25 mm. Jitter is logit-normal on [0°, 90°] with sd 4° at the mid value; azimuth jitter sd 8°. The settle clamps each guide's first segment to its follicle direction.

**Kinds** (`engine/hair/kinds.py`). Each kind is a row of 27 traits, each trait with value, range, unit, an evidence label and a source key. The kinds used on the head, face and arms, with the model's values:

| kind | root diameter | length | section a/b | shape | exit angle |
|---|---|---|---|---|---|
| scalp | 64.6 µm | cut | 1.3 | wavy type (D1.4) | by region, above |
| scalp edge (baby hairs) | ~35 µm | 38.5 mm | — | droops (longer than its 28.9 mm gravity length) | 15° |
| eyebrow | 60 µm (estimate) | 10.5 mm | 1.2 | single arc, curve diameter 30 mm | 12° |
| upper lash | 71 µm | 7.3 mm | 1.21 | arc, curve diameter 12 mm | 75° to lid skin |
| lower lash | 60 µm (estimate) | 5.6 mm | 1.21 | arc, 20 mm | 75° |
| facial vellus | 16 µm | 1.0 mm | 1.1 | straight, nearly colourless | 15° |
| arm vellus / intermediate | 25–40 µm | 2.9–4.2 mm | 1.05–1.1 | straight or shallow arc | 20° |

Every kind except scalp hair and its edge is shorter than its own gravity length, so lashes, brows and vellus keep the shape their follicle and curl give them and need no gravity solve. Pigment per kind is the scalp root's melanin times a kind factor, then scaled by diameter because absorption is stored per unit radius. The face set in use holds 150 upper and 52 lower lashes per eye and 836 hairs per brow. Weak data, stated: no healthy-adult brow fibre measurement was found; lash curvature has no published radius.

**Known gap:** the brow and lash set is a fixed strand set made on one version of the skin. A sculpt that moved the brow skin 0.65 mm buried 1,521 of 1,672 brow tips. A carry step (each hair point keeps its height over the skin, 2 Newton steps, residual p99 0.026 mm) is implemented; growing face hair from follicles on the live field is proposed.

### D1.4 The fibre: an inherited helix, and the curl index as the waviness lever

*Types 2026-10-31; the lever finding 2026-11-12 about 13:30. Status: measured on named cases (single hanging fibre); full head at higher curl index not verified (D1.7).*

In the frame carried along the strand, the rest curvature is κ0·(cos θ, flat·sin θ) with θ = τ0·(s − start) + phase, ramped in over `ramp_mm`. κ0 = 2/CD where CD is the curve diameter. τ0 comes from the curl index i (stretched length over end-to-end length): for a round helix τ0 = κ0/√(i² − 1); for a flattened one, by bisection on the measured index of the generated path. So at fixed i the shape is self-similar in CD: free amplitude ≈ (i² − 1)·CD/2, wavelength ≈ π·√(i² − 1)·CD. The classification follows de la Mettrie and Loussouarn. Design values: wavy = CD 48 mm, i 1.10, flat 0.25, a/b 1.3, lock of ~30 hairs, bond 0.1; curly = CD 10 mm, i 2.0, lock ~50, bond 0.35. Lock size and bond strength are stored as traits of the type: clumping is a property the fibre carries, not a style.

**What was wrong first.** To get a little more waviness (a review direction, 11-12 about 12:10) the first lever tried was a tighter CD (48 → 44 mm). Measured on the head: no gain. One hanging fibre, 250 mm, dry, rms offset from the straight hang: CD 48/i 1.10: 4.75 mm; CD 44/i 1.10: 4.59; CD 40/i 1.10: 4.61; CD 40/i 1.15: 5.67 (+19 %); CD 35/i 1.20: 6.32 (+33 %). A tighter CD alone makes smaller, shorter waves. The lever is the curl index. Gravity stretches the hanging wave: about 89 mm against 69 mm free.

### D1.5 The settle, the rest gate, the cached head

*Head settle 2026-10-30/31; single-rod core 2026-11-09 (3bdba016); field collider 2026-11-10. Status: measured on named cases; gate pass or fail recorded on every run.*

**Single-rod core** (`engine/hair/rod.py`). A discrete Kirchhoff rod in reduced coordinates: the unknown per joint is the Darboux vector (two curvatures and twist) in the material frame; frames and positions are integrated from the root, so length and frame orthonormality are exact. Energy = bending and twist against the rest shape plus weight. Statics by L-BFGS. Checked against Miller, Lazarus, Audoly and Reis 2014: within 1.1 mm of their elastica; the planar-to-3-D threshold matched theory to 3e-6. One of the six proofs in that first chunk failed at the time and is recorded as failed in the commit.

**Head settle** (`engine/hair/settle.py`). Per guide, one free 3-vector per segment (normalised to a direction), nodes integrated from the root. Segments graded from 0.5 mm at the root ×1.2 up to 14 mm. Energy terms:

- bending against rest turning (natural helix plus any water set);
- gravity;
- body contact ½·k·(0.5 mm − SDF)₊² on a quadratic B-spline of the body's 3 mm distance lattice, with a 1 mm lattice over the head added 11-11 after 565 of 228k sampled points were found inside the true field following a comb over the ear (0 after);
- hair-hair pressure from fibre fraction φ splatted on a 3 mm lattice, P = k·E·φ³ (van Wyk), k 1e-5, E 4 GPa dry;
- cohesion from oil and water bridges, −c·φ_w², sebum creeping 50 mm per day from the root, surface tension 30 mN/m for sebum and 72 for water;
- lagged, smoothed Coulomb friction with ε 1 mm, μ 0.5 on skin, 0.25 hair on hair wet, 0.13 dry;
- the passes' own energy through a hook.

Solver: L-BFGS-B in scaled variables. Growth is by continuation: the active arc grows through 30, 60, 100, 160, 240, 330, 430 mm, then the longest guide, 40 evaluations per stage; release 135 evaluations; the final stage is capped at 4,000. The whole evaluation also runs on the card.

**Friction anchors across a lag.** After each sub-solve, a sample that moved less than ε in its contact plane has stuck and keeps its anchor; one that slid gets an anchor trailing ε behind, so it keeps about 0.71·μN against further sliding. Without this, a held layer creeps ε per lag.

**The rest gate.** A settle is converged only when both hold: the largest unbalanced transverse force at any node, divided by the larger of the local weight and the summed term forces, is under 0.05; and the relative energy change over the last 20 evaluations is under 1e-4. No render is made of a style that fails.

**The cached head** (`engine/hair/head.py`). Saved: rod directions, rest turning, wave blocks, friction anchors. Key: SHA-256 over the sorted settings of the hair, environment and simulation layers, the collider, and hashes of the growth code. A rebuild whose key or array fingerprints differ is refused. Observed: the guard refused a run whose follicle directions differed.

### D1.6 Waves: the phase fault found and fixed

*Banding measured 2026-11-12 about 02:20; phase field 07:17 (077d0d39); jitter fix 09:26 (0df28b28); phase-aware child blend 10:05 (2ffb3d9f); wave-only shift 12:10 (b6e2d95d). Status: measured on named cases.*

**The fault.** The approved base (head "n2c") gave each 35 mm drying section one wave phase. That size came from barrel styling; applied to an air-dried head it put waves in step in horizontal bands. Measured: per 30° sector, the mean wave held 0.12–0.72 of the strands' wave variance (random phases give ~0.0002).

**The fix: a coherent phase field over the roots** (`form.phase_field`). Random Fourier features on 3-D root positions: 24 wave vectors drawn N(0,1)/scale, offsets uniform on [0, 2π), f = Σ cos(X·k + b)/√12, phase = π·f + N(0, 0.15 rad). At scale 32 mm: coherence 0.83 inside a lock, 0.41 at 8–15 mm, about 0 beyond. A first scale of 12 mm broke coherence inside a lock (0.47) and was dropped. Measured on the comparison sheet of the two heads (n2c before, n2d after): the wave correlation between sectors 90 degrees or more apart went from 0.12 to -0.01, and the even horizontal bands at the side and back no longer appear.

**Three follow-on faults, each found by measuring.**

1. *Child jitter read against the wrong reference.* Each child's own jitter was read against its old section's phase, which put a section border's random step into 51 % of children (shift p50 0.61 rad). Read against the smooth field instead: p50 0.14 rad, 0.4 % above 0.6.
2. *Blending guides that are out of step flattens the wave.* Children blend the 4 nearest guides; with guides at different phases they kept only 80–83 % of the guides' wave. Now, for the blend only: shift each guide to its section's mean phase, blend, then shift each child from the mean to its own field phase. Kept: 90–93 %.
3. *The shift moved the whole strand.* The first shift slid each segment's direction along the arc, which also moved the strand's large bends and looped crown hair up. The replacement moves only the wave: resample by arc; measure the strand's own hanging wavelength from zero crossings (clamped to 0.8–1.8× rest); take the mean line as a one-wavelength box mean (a box that wide has a zero at the wave's frequency); slide the residual wave along the arc with a smoothstep so the root's exit is untouched; put nodes back at their own arcs.

### D1.7 Head variants, the gate results, the cost of air-drying, and flyaways

*2026-11-12. Status: measured on named cases. None of the later variants has replaced the base; hair work was put on hold at about 18:40 the same day.*

"Loose stretch" below means a run of at least 5 consecutive points with fewer than 6 other-strand points within 4 mm, outside the last 15 % of the strand.

| head | change from the base | rest gate of the grown head | style gate | child wave rms by third of length (mm) | loose stretches, styled |
|---|---|---|---|---|---|
| n2c (base) | — | 0.038 pass | pass | 2.78 / 3.07 / 3.51 | 104 |
| n2d | phase field 32 mm, first shift | 0.443 **fail** | not stated in the record (it was rendered) | 2.57 / 2.77 / 3.31 | 326 |
| n2e | + wave-only shift, CD 44, exit angles 40°/50° | 0.424 **fail** | 0.047 pass | 2.78 / 2.91 / 3.40 | 168 |
| n2f | n2e, comb pull halved | (n2e's head) | 0.025 pass | 2.78 / 2.96 / 3.43 | 166 |
| n2g | n2e, water set taken after the hands | (n2e's head) | 0.038 pass | 3.08 / 3.36 / 3.82 | — |
| n2j | curl index 1.20, set after the hands | tips hanging at 0.1–0.19 | **0.992 fail** | 3.88 / 4.44 / 5.15 | 247 |

**A head failing its rest gate.** The n2d and n2e heads failed with energy flat, so it was stuck guides and not too few evaluations. A probe found 27 guides, one tip node each, resting on the front of one shoulder 0.4–12 mm off the skin: a friction stick at a hanging tip, not a strand inside the body. The style stage re-settles and its own gate decides. An earlier 12 mm version failed at 0.897. n2j failed its *style* gate at a handful of nodes (p99 0.021) and has not been rendered; a diagnostic that reports the worst nodes is implemented and has not been run.

**What air-drying costs the wave.** On the base, the untouched rested head waves at 3.42 / 3.64 / 3.95 mm; after the style, 2.78 / 3.07 / 3.51. Cause: wet hair hangs heavier and straighter, and the water set keeps a share (0.4 in this recipe) of that wet shape.

**The water set and its timing.** Over the set span, each joint's rest turning moves toward its current turning: ρ ← ρ + set·(dd − ρ). Re-wetting removes the set's share; the follicle's own curl is untouched by water. The set fades with time and humidity as R(t) = 0.7·exp(−t′/0.5 h) + 0.3·exp(−t′/24 h), t′ = t·(RH/90)^5.2·10^(−2·hold). Water softens and loads the fibre: E × (1 − 0.5·wet). The first recipe took the set at the end of the hand pass, while the hand still held the hair. Taking it in the free fall after the hands are gone (n2g) gave +11–15 % wave with the body changed by only 0.2–1.8 mm. Kept. *Tried and rejected:* re-taking friction anchors at the dry friction (0.13) after drying gave no width gain and let the front slide onto the face (529 points over the face region against 275).

**Body.** A review asked for a little more body, later clarified as mostly at the sides and as following from the texture of the waviness alone. Exit angles, comb strength and CD all measured flat after the style. On n2j the sides' p95 width rose 2.2–3.0 mm and the crown 1.4 mm with the wave, but n2j is unverified.

**Flyaways: designed against defect.** The generator marks a small fraction of strands as fly-aways that leave their lock partway along. Of the base's 104 loose stretches, 12 are on designed fly-away strands and 92 are not. On n2d: 16 designed, 310 not. On n2e: 14 and 154. On the untouched heads: 787 (base), 1,004 with the first phase shift, 740 with the wave-only shift. On review of the n2c / n2d sheet the flyaways were taken as most likely by design, to change only through styling. The counts say otherwise for n2d: its extra loose strands were a defect of the first phase shift, fixed at that cause.

**Crown loop.** The base has a loop of hair standing at the crown. Measured cause: short regrowth hairs (35–73 mm) at the whorl, where the long layer parts away and nothing lies over them; a 70 µm hair only buckles under its own weight past about 91 mm. The generator fix (2026-11-11 21:43, 7bc80fe5): short hairs take the wet set with the long hair. On a test cap: 5 of 5,331 short hairs standing before, 0 after. The base itself has not been regrown with it.

### D1.8 Passes as tools, and hair inside the tool

*Passes 2026-10-31; the Hair dock in Field Studio 2026-11-11 20:06 (91fa6b73). Status: measured on named cases.*

**A stroke as a pass.** The tool's leading edge moves down each strand's arc in graded steps (15 mm, ×1.25, capped at 80 mm). Inside its window it turns each segment's heading toward the stroke about the skin normal only, and drags the hair by bristle friction: a smoothed pull toward a point one step further on, capped at the tool's tension, with no normal component for a brush or comb.

**In the tool.** Pick a kind, paint the area, set colour, thickness, density, length, generate. Then comb, cut, puff, clump, smooth, trim, settle, water (washes the set out, cuts stay), undo per stroke, keep as a state over the natural base. Sectioned cuts: a section brush, same-length with a travelling guide, lift-and-cut at an elevation and angle, point cut.

Measured: 198,156 hairs with 1,593 guides generated in 342 s on the CPU and 195,294 in 169 s with the rods on the card; brush dabs 1.3–1.4 ms on the card; every hair drawn in 46–255 ms a frame; app 4.1 GB working set and 3.8 GB of card memory with the model open. A lift-and-cut of 40,462 hairs took 4.1 s. *Stated plainly at the time:* that first layered cut did not read in the before/after sheet, because the cut ends stayed inside the under-lengths.

### D1.9 The fibre layer: rendering, scattering, shadows

*Fibre layer 2026-11-10 (b0535200); multiple scattering 11-10; exact skin shadows 11-11; lean scalp shadow default 11-12 01:19 (1964801a). Status: accepted in the workflow for the face-hair shadow fix (approved on review) and the scalp shadow; the rest measured on named cases.*

**Data.** Per strand 24 points, radius and pigment per point, ellipticity, twist rate, roughness, root normal. Each strand is a centripetal Catmull-Rom curve. A uniform span grid (0.5 mm cells for short hair, 2 mm otherwise) lists each span in the cells its curve passes.

**Primary rays.** A 3-D DDA walk of the span grid, a conservative sphere pre-test per span, then the exact hit (curve chords plus 3 Newton steps, oval section turning with twist, sub-pixel coverage from the pixel cone's overlap). The 8 nearest hits are shaded and composited front to back, then the next 8, up to 48 fibres or until transmittance falls to 0.02.

**Scattering.** The Chiang, Bitterli, Tappan and Burley 2016 lobes (R, TT, TRT and a residual) on the Marschner 2003 fibre, with a precomputed far-field table (β_n 0.55, η 1.55), cuticle tilt 3–4°, β_m 0.10. Absorption is per unit *local* radius, so a tapered tip is lighter. Near field when a fibre is at least a pixel wide. Wear: β_m × (1 + w·min(s/0.4 m, 1.5)), 0.10 at the root toward 0.22 at 40 cm. A sized key widens β_m through Chiang's variance mapping.

**Light per hit.** (1) Body shadow: a soft SDF march toward the key. (2) Self-shadow: transmittance through every fibre crossed toward the key, each fibre counted once. (3) Sky: the body's openness times the hair's transmittance in four cosine-weighted cones, with forward attenuation a_f = TT + 0.293·R + 0.15·TRT. (4) The room per lobe. (5) Multiple scattering from the mass beneath: per fibre the back share b and forward share f, B = b/(1 − f²), added in proportion to how dense the hair is toward the skin. With it, hair-to-skin lightness went from 0.70 to 0.89 against a reference photo's 0.81.

**Body lattice repair.** For the hair's light the body lattice is made a true distance everywhere: inside ±9 mm the field's own value; beyond, the Euclidean distance to a reliable shell (5–9 mm) plus that shell's value, per side, blended over the last 1 mm.

**Hair shadowing skin: the fault and the method.** One cone of the light's size walked through the span grid visits only the cells on its centre line. Under a 12° softbox the footprint is 4–12 mm at 2–6 cm, so the cone missed most fibres and over-weighted the rest. Measured on a three-quarter frame: a lash shadow streak 1.8 mm wide at 0.40 light where a fibre-resolved reference gives 10.9 mm at 0.72. About the same total shadow, squeezed into a band 4–6 times narrower and twice as deep. Two compositing faults sat beside it: each layer's skin shadow was multiplied in after earlier layers were drawn, and the face hair was drawn over the scalp hair.

The method now: per skin pixel, exact stratified rays to a cap of the key's angular size and to six sky caps of 0.12 rad, with every fibre crossed counted; each layer's shadow is taken on bare skin first; layers are composited by first-hit depth. For scalp hair a lean version: 8 key + 6 sky pilot rays at every pixel, then up to 36 + 24 only where skin is seen through hair and the pilot rays disagree. Measured against a 144-ray reference: mean error 0.0117 (the cone: 0.0357), total shadow within 0.12 %, isolated black specks 18 against the cone's 1,066 (reference 14). *Tried and set aside:* sub-cones (5 % too dark) and the hair as a measured absorbing medium (27 % too dark: scalp hair lies in locks, and a density loses the gaps between them).

Hair in the eyes' mirror reflections: 6×6 exact sub-rays along each eye pixel's reflected ray; error against a 16×16 reference 0.0001. Two runs of the same frame first differed by noise because sub-ray seeds followed an atomic counter's order; sorting records by pixel made two runs bitwise equal. Status of that determinism: reproduced under a frozen configuration.

**Costs.** A three-quarter head frame: hair 35.9 s, of which the scalp's exact shadow 11.4 s; a hairline close-up 20.7 s; the face-hair shadow pass 0.5–1.7 s; eye reflections +0.8–1.2 s. Growing a shoulder-length head: 14 min on the CPU or about 6.5 min on the card; styling about 8–10 min on the card, 34–40 min on four CPU threads.

### D1.10 The settled hair frame in the tool's window, and the multi-light layer

*2026-11-12. Status: implemented in a working copy; not landed. Partial measurements only.*

While the view moves, the window draws hair with a fast splat. Measured against the review renderer pixel for pixel, the splat's light was 2.3–4.4 times too bright and grey. The design: when the view rests and the window's frame has converged, draw the review renderer's fibre layer over the window's own frame, depth and lights, and keep it until any input arrives. One job at a time on a worker thread, cancelled at its next launch by any mouse press, wheel or key. Two causes were fixed first: the unrepaired lattice put the hair at 0.53–0.60 of the review's radiance, and unshadowed skin under translucent blond fibres read as "ghostly". With both fixed: 0.93–1.11.

Measured in the real window: lands 9.9–11.7 s after the view stops; first layer build 27.7 s per hair version; card memory 4.25 → 4.49 GB. **It failed its stroke check:** a body stroke during a job stalled the window (0 fps, worst gap 1,504 ms, against 32.3 fps with the feature off), because the layer's bands ran on the default stream. A fix is built (its own non-blocking stream, 0.05 s bands, cancel on input) and has not been re-tested.

The multi-light fibre layer gives each extra light a direction, colour, angular size and shadow flag and sums its direct terms onto the key's. It compiles. Its three tests (bit-identity with one key, linearity, a headlight) have not been run.

---

## D2. Light

### D2.1 Tracing the field

*In the tool before this lane; measured 2026-11-11. Status: measured on named cases.*

Rays are sphere-traced through the sampled field (Hart 1996), then the hit is refined by bisection and a secant step inside the bracket. The normal is the analytic gradient of one tricubic Catmull-Rom reconstruction over the 4³ nodes around the point. The plain sculpt light is a soft-shadow march toward the sun (the min of k·f/t along the ray), 2 sky rays per pass to 0.15 m, and spherical-harmonic sky irradiance, accumulated over 16 passes. The sky is added when the frame is resolved, not in the accumulation; two measurement mistakes in this lane came from forgetting that.

Costs at 1920×1080 on the model: primary trace 1.29 ms for a full-body view, 5.00 ms for a face close-up (7.6–10.3 ns per hit pixel); a sun ray 3.0–5.3 ns; a sky ray 2.1–2.5 ns.

The tracer skips empty space with a 32 mm tile grid. The project's own reading flagged that tile skip against two patents on per-tile empty/occupied entries (Movidius US11680803, US11222459). It is being redesigned (skip by distance, not by an empty tag) and that code is held out of any published copy until it lands.

### D2.2 The rig, the panel, the presets

*Rig schema 2026-11-11 17:41 (88e5c965); presets 19:28; Light panel 20:10; in-view A|B split 20:24. Status: implemented, with CPU gates passing; the split reproduced under a frozen configuration (each half bit for bit its rig's own frame).*

A rig is a small versioned file: lights (sun, area rectangle or disk, sphere, spot, point), environment (HDRI, rotation, sky gain), exposure. Both the tool and the review renderer read it. The panel offers presets placed around the view's target, key controls, environment, save and load, A/B compare and a split view. Hair reads its lights from the same rig as irradiance before exposure.

**The Rembrandt-style preset** (2026-11-12 08:38, 03f20e3a). Key 65° round and 30° up, 4° across as seen from the subject (a 9×14 cm softbox at 2 m), fill 0.05, a dark studio with room light 0.05. It was set by measuring five traits of the pattern on the model's face (lit triangle against shadow cheek, lit against shadow side, triangle against lit side, triangle against nose shadow, and the nose shadow's edge width) against a published portrait-lighting reference that was studied and is not redistributed. Of 96 settings the chosen one was nearest (distance 0.81; the previous preset 1.72); nose shadow edge 4.0 mm against the reference's 3.9. A softer variant (10° key, fill 0.1) is kept under its own name. A 17° candidate cost the same per pass but gave an 11.6 mm edge and was not adopted.

*What light could not do:* across 288 settings the "closure" of the triangle (the region beside the nostril as dark as the reference's) never moved. On the model that region is as bright as the cheek shadow in every setting. It is the model's form there, not light, and it was not forced. Status of the preset: accepted in the workflow, conditionally (kept on the condition that it passes the legal checks and the other requirements); grain and cheek crops at the 4° key were not measured on the card.

### D2.3 The stage-1 study: what was measured before building

*2026-11-11 16:55–18:05. Status: measured on named cases (probe code, Vulkan through Slang; not the product path).*

- **Bounce by brick caches.** A radiance cache holds each surface brick's direct key light (0.18–0.23 ms for 38k bricks). An irradiance cache gathers the bounce at each brick from cosine rays that read the radiance cache (64 rays: 5.5–10.8 ms). Read per pixel by blending 8 bricks: 0.067 ms, bias −0.2 %, error below the noise of a second 256-ray reference. The earlier option (16 rays, hit lit unshadowed) was 28.7 % too bright.
- **What failed:** a cone-gathered bounce (−63 % bias); cone sky occlusion (−29 %); five distance lookups along the normal (correlation 0.54); cache points storing light by direction in 9 spherical-harmonic terms (ringing, +4.7 % bias).
- **Colour.** Bounces 2 and 3 add 7.6–9.2 % of the bounce light. In skin folds the bounced light's red/green ratio is 1.193; a grey cache is 15 % low in red there. Skin-to-skin bounce has to be carried per channel.
- **Patchiness at close zoom.** One point per 8 mm brick gave a structured error of 34 % on a face close-up against path tracing's 1.5 % floor. A 2 mm lattice: 4.5 % on the face, 2.0 % on the chest. A per-pixel gather: at the floor.
- **Sky as per-brick spherical-harmonic transfer.** 9 coefficients a brick from 256 rays to 2 m: bias −1.1 % under a smooth sky, where the short-ray method is +18 % in the most occluded tenth. Under a bright 20° soft box the 9-term transfer fails (+87 % and worse). Conclusion: smooth sky through transfer, every bright light sampled explicitly. This is a probe result. The records do not establish that per-brick transfer is in the tool's live light.
- **Caustics.** Light tracing into 2 mm floor cells through a three-mirror chain: 4.2 M photons in 5.9 ms, every caustic cell found in one batch, bias 0.1 %. Camera-side path tracing at equal time found 0.15 % of the cells. Probe only.
- **RT cores.** One box per surface brick in a hardware tree with the same in-brick hit: 1.2–1.8× faster than the software two-level DDA on primary and short sky rays, equal on sun rays. Tight boxes: up to 2.6–2.8× on sky rays.

### D2.4 The look-dev light: lattice while moving, gather when still

*Module 2026-11-11 19:33 (129d2900); lattice 19:39; in the card view 20:18 (928f3d1e); 64-pass target 11-12 04:39. Status: measured on named cases.*

**Two stores.** 8 mm brick caches keyed by (volume, brick), and a 2 mm surface lattice inside every surface brick: a 64-bit sub-cell mask and an offset per brick, valid points only, snapped to the finest surface level. On the model: 692,218 points, built in 15 ms, gathered at 128 rays in 0.50 s, all caches 38 MB on the card. The lattice covers every surface brick regardless of the camera, so orbiting needs no refresh.

**Two modes, chosen per frame by interaction.** While the view moves, pixels read the lattice (0.07–0.13 ms). When the view rests, each pass runs a per-pixel cosine gather. The key is sampled as an area light, with the plain sun term zeroed.

**The settle.** When the view rests, passes accumulate toward 64 per pixel (16 in sculpt mode). There is no denoiser: the image converges by samples. A still pixel's mix (1–3 lattice passes, then gather) measured level with path tracing at equal rays in the folds, bias −0.24 to +0.06 %. While orbiting, the lattice's error in folds was 1.1–1.9 %, fixed to the skin, with no flicker.

**Three wiring faults found by the gates and fixed:** near hits lit by the sun only, missing a sampled studio key; the key counted twice in the app (the tests had patched the light by hand, hiding it); fixed-size slices that ran to 103 ms. A fourth, found in the app only: a separately compiled module had its own unset copies of the tracer's device globals, an illegal address the moment a masked volume was sampled. A blocking-launch run named the launch.

### D2.5 The cache redesign of 2026-11-12

**What the first version did** (11-11 19:33 to 11-12 10:22). At each gather hit, if the hit was at least 10 cm away the gather read the hit brick's *outgoing* cache (albedo/π × (direct + bounce)); nearer hits were lit exactly. This came from the study, where it met the patch-error gate.

**The flag.** A further check of our patent readings found that NVIDIA US11790594 has an independent claim 16
which our earlier check had missed. As we read it, the claim concerns a ray cast from one location that meets a
second location associated with an outgoing irradiance cache, and rendering that uses the cached outgoing value
based at least on a distance between the two (claim 18: beyond a threshold). The first look-dev gather, with its
near/far switch, was flagged in our review and redesigned the same morning. Nothing carrying it was ever released.

**The redesign, landed 11-12 10:22 (506bd914).** One rule at every gather hit, never chosen by distance: the rig's lights are computed exactly at the hit (the key by a shadow ray, each area, spot or point light by one sample). The gathers read no outgoing cache. Gather rays run until they leave the scene. This is the arrangement of Ward, Rubinstein and Clear 1988. Direction of the design-around: no distance, threshold, ray-cone width or footprint decides anything at a hit. Recorded rule for future work: never add a choice between a cached value and an exact one by a hit's distance, cone width or footprint.

Status: measured on named cases. One bounce matches an all-exact reference with bias ≤ 0.12 % in the folds. A still pass at 960×540: sun, face 9.66 → 10.46 ms, chest 16.48 → 19.32; Rembrandt, face 13.48 → 13.30, chest 21.27 → 24.30. Lattice build 511 → 593 ms (sun), 628 → 716 (Rembrandt). Card memory identical (2,051.8 MB built and drawn, either way).

**What it cost, stated:** the product is now one exact bounce. The second bounce is missing: 0.5–2 % of the light in folds, about 35 % in the sun-shadowed pockets of a dark studio. A test showed the 8 mm cache's stored bounce was 1.1–2.4 times the true second bounce there (one point per brick leaks light across a fold), so it is not used. Taking the second bounce from the 2 mm lattice by the same one rule is proposed.

**Excluded prototypes.** Two study probes, `engine/light/fs_cache.cu` and `engine/light/fs_cache2.cu`, still contain the old distance switch; the second also built lattice points only for bricks visible to the camera. They were never wired into the tool. They are excluded from any published code as they stand.

**Refresh without ranking.** A light change re-gathers every valid point. An edit re-gathers only its dependency set: lattice points that are new, or lie within a fixed reach (0.15 m) of the edit's box. That is one membership test, the same for every point, walked in fixed memory order. Slice *size* is predicted from the measured cost of the fixed 4,096-element blocks it covers; it decides how many come next, never which. Claims 1 and 10 of the same patent require ranking locations against one another; no location is ever compared with another here.

### D2.6 Keeping the display alive: slices and bands

*Review renderer bands 2026-11-11 19:04 (ec99a732); hair bands 21:29 (8c3bc466); look-dev slices 11-11 19:55 to 11-12 06:03. Status: reproduced under a frozen configuration for the review renderer's bands (tiled equals untiled bitwise: image, depth, call counts); the rest measured on named cases.*

The display driver resets the card if one launch runs for about 2 s. Two resets happened on 11-11 from launches of about 10 s.

- **Review renderer:** frame passes run in row bands, alternating over two streams so one band's tail overlaps the next. Frame time unchanged (326.9 against 326.7 ms a pass at 960×540). The heaviest close-up's longest launch: 304 ms, against a 1 s cap. If a band errors, it falls back to whole-frame launches.
- **Hair kernels:** timed row bands aimed at 0.6 s (growth ≤ 1.5× per band, never under 32 rows), one round of 8 hits per launch with per-pixel state carried between launches. A whole-frame scalp launch had run 15–19 s.
- **Look-dev cache builds:** done on idle frames in slices aimed at 8 ms, into a separate set that is committed whole. A launch lasts at least one thread's serial loop, so the budget is met by chunking rays (8 a launch), never by shrinking the element count below what fills the card. Returns at once on interactive frames.

Measured after a 2 mm cheek stroke: the dependency rebuild settles in 0.86 s (a full rebuild 2.6–3.3 s), bitwise equal to a full rebuild beyond 15 cm, 0.004 % of rendered pixels differing. Edit path's longest call 12.2 ms.

**Known defects, open.** Some slices exceed the 8 ms budget in the app (begin 9–16 ms, list ~20 ms). After a stroke the view converges twice. Release to idle takes about 13 s at 1600×900 and 45 s at 2560×1440. One look-dev still pass costs 92–103 ms in the app against 15 ms in a test scene; not broken down yet. Running still passes in row bands cut the worst press-to-frame wait (219 → 152 ms) but lengthened release-to-idle (13 → 18 s); starting the edit job at release made presses worse (492 ms). None of these three patches landed. Look-dev has no hair in its light yet.

### D2.7 Skin in the light

The look-dev light treats skin as a diffuse albedo per channel. Subsurface terms (a per-channel walk table, terminator wrap, back-lit thickness) exist in the review renderer's own light code and are named in the study only as components to share later. These records do not establish them in the look-dev path. What is measured here is the colour of skin-to-skin bounce (D2.3).

### D2.8 The older-hardware aim

*Status: proposed. Estimates only; no older card was measured.*

The aim is ray-traced-quality light on cards without RT cores, such as a GTX 980 (4 GB, no ray query). The route: one Slang source compiled to Vulkan compute, the software two-level DDA for traversal, caches for bounce and sky, hair as a density field at distance. Estimate from the measured terms: a lit body at 1080p costs about 4.8 ms on the RTX 5050's software path; scaled by bandwidth and floating-point throughput, about 10–17 ms on a 980, without skin shading, eyes or hair. Running the same Slang code on a CUDA backend failed (the device could not be created), so Vulkan against CUDA on identical code is untested.

---

## D3. Published ingredients and licences

| ingredient | source | terms as recorded |
|---|---|---|
| Fibre scattering lobes, roughness mapping | Marschner et al. 2003; d'Eon et al. 2011; Chiang, Bitterli, Tappan and Burley 2016 (as in pbrt-v3) | published papers |
| Forward attenuation in a fibre mass | Zinke et al. 2008 | published paper |
| Rods | Bergou et al. 2008; Bertails et al. 2006; Miller, Lazarus, Audoly and Reis 2014 | published papers |
| Fibre mass pressure; wet cohesion | van Wyk 1946; Goldstein, Warren and Ball 2012; Bico et al. 2004 | published papers. Goldstein et al. note van Wyk's law does not fit their data: an open item |
| Hair classification and biology | de la Mettrie, Loussouarn et al. 2007; Thibaut et al. 2005; Franbourg et al. 2003 | published papers |
| Hair with water and styling | Ward, Galoppo and Lin 2004/2007; Fei et al. 2017 | published papers |
| Random Fourier features | Rahimi and Recht 2007 | published paper |
| Sphere tracing | Hart 1996 | published paper |
| Direct at every hit, indirect from a cache | Ward, Rubinstein and Clear 1988; Christensen 1999 | published papers |
| Precomputed transfer | Sloan, Kautz and Snyder 2002 | published paper |
| SDF grid tracing, studied | Hansson-Söderlund, Evans and Akenine-Möller, JCGT 2022 | CC BY-ND 3.0; studied, its exact hit not used |
| Capsule intersection (probe) | I. Quilez's articles | MIT |
| Slang / slangpy 0.43.1 (probes only) | PyPI | Apache-2.0 (Slang with LLVM exception); its bundled OptiX headers are proprietary and unused |
| Vulkan headers | Khronos | Apache-2.0 or MIT |
| Lighting and lash-shadow study photographs | public image archives | CC BY-SA and public domain; studied only, not shipped, not redistributed |

Not used: OptiX (proprietary terms, one vendor's hardware).

## D4. Patent readings carried over as recorded

**These are our own readings of the claims, made without counsel. They are not legal advice and no clearance is claimed.** A search hit is a screening flag; a claim reads only if every element of at least one claim is met. Finding none is not proof of absence. Expiry dates are estimates. Pending applications can change and need re-reading at grant.

| record | claim element of concern | what is absent in our design |
|---|---|---|
| NVIDIA US11790594, claims 16–18 | outgoing irradiance cache at a ray's hit, used based at least on a distance (18: beyond a threshold) | the first look-dev gather was flagged in our review and redesigned. Since 11-12: no outgoing cache read at hits and no distance, cone-width or footprint test. The two probe files that still hold the switch are excluded |
| same, claims 1 and 10 | ranking cache locations by characteristics (camera distance, visibility, frames since update) and casting rays from the higher-ranked | nothing is ranked; refresh is by a fixed membership test in fixed index order; the lattice is camera-independent |
| its continuation US2024/0029338 (pending) | a denoising filter mixing cache values | no denoiser; re-read at grant |
| NVIDIA US11823319 | in-voxel hit from a cubic whose coefficients come from a factored formulation | hits are sphere tracing plus bisection and secant on the sampled field; no cubic coefficients are formed. The JCGT 2022 exact hit is not used. One project record marks this reading as not verified against the full granted claims, so it is treated as open (Part 10) |
| NVIDIA US12254556 (independent claims 1, 5, 15, 22) | intermediate normals per adjacent voxel, then interpolated to the hit | the normal is the derivative of one tricubic reconstruction; no per-voxel normals exist and none are blended |
| NVIDIA US11887245 | shadow rays via the cubic's turning points | that algorithm is not used |
| NVIDIA US12175588 | hit points stored in voxel-local frames | hits are kept in world or object space |
| Pixar US9905045 | a scattering component made by *fitting* a set of basis distributions of several types, sampled from their closed forms | the Chiang 2016 lobes are derived, with one logistic per lobe whose scale comes from a closed formula; tables are numeric. Guardrail: never fit mixtures of logistic, skew-logistic, monomial or Cauchy lobes to a measured response and sample them in closed form; tabulate and sample by numeric CDF |
| DreamWorks US9019298 | importance sampling of hair lobes based on a Cauchy distribution | no Cauchy distribution anywhere in the hair code. Guardrail: keep it so |
| Disney US8674988 | per-lobe artist controls for hair shading | recorded as "expired – fee related" (lapsed 2026-03-18) as of 2026-11-11. That is a dated status, not a permanent one: such a lapse can be reinstated, and it is to be re-checked before release |
| Microsoft US7948490 (active to 2028) | transfer coefficients computed by rasterised shadow depth buffers | transfer in the probe was gathered by rays in compute; no shadow maps |
| NVIDIA US12315070 | caustics by a feedback-driven photon task buffer and splatted footprints | the probe is plain light tracing into surface cells |
| NVIDIA application US2026/0148476 (pending) | hardware linear swept spheres | measured in a probe only; hair does not depend on it |
| AMD application US2024/0331266 (pending) | capsule-chain curve intersector | the probe's box-plus-capsule test: claims to be read before it ships. Not checked |

Not found on the records searched: patents on sphere tracing, photon mapping, the Chiang 2016 or Marschner 2003 models, tight axis-aligned boxes around SDF bricks. The search was cut short by blocked queries and is incomplete.

## D5. Open items

- The curl-index head failed its style gate at a handful of nodes; no result from it is used until the gate is diagnosed.
- No hair dynamics: statics only.
- The through-fibre composite is a screen-space approximation of the TT lobe's cone.
- Friction anchors are fixed in space between lags; hair sliding on hair does not drag its neighbour.
- Section ellipticity is stored but is not yet a bending anisotropy; per-type curl indices are design values.
- Face hair does not shadow itself; brows and lashes do not follow a sculpt without the carry step.
- The settled hair frame and the multi-light fibre layer are not landed or verified.
- The second bounce is missing from the look-dev light.
- The tracer's tile skip is being redesigned.
- Nothing has been measured on a card without RT cores.

---

# Part 9. The current state, measured

As of the evening of 2026-11-12 (Pacific). One line per subsystem: what works, what is slow, what is not done.
Each line is a summary of the part named; the numbers, cases and conditions are there. Nothing in this table has
been reproduced by anyone outside the project.

**The headline, stated plainly.** The tool exists and runs: a real window, a field store with versions and undo,
brushes on the CPU and on the card, parts that touch without merging, a light rig with a settling still frame,
generated hair, teeth and garments. It does not yet pass its own first test (a sphere to a rough head in under
five minutes), it misses its frame-rate targets on the full human model, and the hands-on sculpting
test of the rebuilt toolkit has not happened yet. Several mechanisms described here are prototypes measured on
synthetic cases only.

| Subsystem | What works (evidence state) | What is slow or failing | Not done |
|---|---|---|---|
| Field store (Part 4) | Four-level sparse brick lattice, 0.25 mm finest; versions by copy-on-write; one writer transaction per edit; a seam guard and a publish gate (accepted in the workflow: the current human model is made through them; five real strokes written back gave 0 seams at every level where the earlier writer left 4,053) | Whole body at the finest level does not fit the 8 GB card at once; the full-session benchmark on the human model hit an out-of-memory fault on the card and missed the 30 frames per second and 100 ms targets; one save was interrupted by quitting (the sculpt was recovered from the temporary folder, and the save path was fixed) | The stroke's whole turn on the card; one live sparse store on the card with copy-on-write undo; per-part lattices finer than the 0.125 mm leaves that exist today |
| Brushes and toolkit (Part 5) | Draw, Carve, Inflate, Clay, Smooth, Swell to contact, Settle onto, protection of surfaces behind (implemented; measured on standard forms and on a thin-wall test). Grab is view-dependent and is being replaced; the rebuilt Pencil is faulty and has not landed | Test 1 not passed: the first scripted run made a lumpy, egg-like form with a flat face at 6.6-13 frames per second; on 11-12 a head could not be blocked out by hand; a screen-plane pull lost most of its depth off-axis (30.1 / 17.7 / 5.7 / 0.8 mm at 0 / 30 / 60 / 80 degrees) until the form-based constraint (29.4-29.9 mm at every angle, in a test copy) | The toolkit rebuilt on its ten principles; the hands-on test; form-aware Grab; design changes for two patent flags |
| Shell (Part 5) | The window, shelf, grids, section and cut tools, regions, marks (implemented; shown in the images) | | A designed product pass with the sculptor in the loop |
| Contact and parts (Part 6) | Parts as separate fields with touch / slide / bond rules; solid-by-default body with designed cavities; a barrier solver as baseline; a constraint-based prototype; the field-read barrier in two forms (all measured on named, mostly synthetic, cases) | The stand-in elbow stalls at 85-90 degrees (from a straight rest) under three different contact laws, cause not established; the human model's own elbow reached 90 degrees from rest (about 112 anatomical) with the solver at its limits, against a 145-155 degree target; soft-on-rigid favours the baseline (73 s against 7.4 s); the hand-on-cheek scene did not converge; the explicit form fails friction | A contact law of record; a broad phase for many bodies; the environment-field sensor (proposed only) |
| Posing (Part 7) | Skeleton, joint finder, volumetric bone-heat weights on the lattice, dual-quaternion map with inverse; drag at a median 16.5 ms a frame at 1080p on stored corrective states (measured on named cases) | Skinning alone folds at the elbow at every zone width; the posing corrective (an earlier, different setup from the contact study) reaches 60 degrees on the test arm and 55 degrees on the human model, with visible crease defects | 90 degrees and beyond; shoulders; bones and muscles as layers |
| Cloth (Part 7) | Body measurements and landmarks read from the field; yarn points on a pattern lattice; a swim-knit top built, settled and shown; a dress form (measured on named cases) | No tee is at rest yet | Garments as two-faced thin objects for the brushes; cloth in motion |
| Mouth library (Part 7) | Generated teeth, arch, gums, palate, jaw and tongue as field parts, placed in a head in the real window (measured on named cases) | The cheek-to-teeth space in the human model is unresolved (one run ended out of memory) | Seating published in the model; the life-like look |
| Hair (Part 8) | Grown once from follicles as curves with an inherited helix, settled to a rest gate, cached; a wave-banding fault found and fixed with measurements; fibre scattering after published models; exact shadow rays checked against a 144-ray reference (measured on named cases; the base awaits acceptance) | About 6.5 minutes to grow a head on the card; 20-36 s a frame for the full-quality hair render; one variant failed its style gate; an earlier whole rebuild was archived as failed (10-31) and a built style was stopped on review (11-11) | Dynamics (statics only today); styles as states in the tool's window (styling passes are measured in the hair work area only); brows and lashes following a sculpt; the hair in the tool's own window at full quality |
| Light (Part 8) | Sphere tracing; a rig, panel and presets; a lattice light while moving and a gather that settles over 64 passes when still, with no denoiser; the cache redesigned on 11-12 around a patent reading, one bounce within 0.12 % of the all-exact reference on the test folds (measured on named cases) | The second bounce is missing (0.5-2 % in folds, about 35 % in dark pockets); no light setting reached the studied portrait reference | Light on cards without ray-tracing hardware: nothing measured yet |
| Skin, eyes, materials (Part 3) | Render-time micro-relief generated and varied; colour by zone; a seeded iris generator (measured on named cases; see the image set for the present look); 72 sourced material profiles committed, 75 on disk (implemented; the tool's viewport did not read them as of 11-11) | The lip colour border steps in 0.3 mm cells in close-up; grain at the lid margins from a stale copy of the eye in the body field (cause measured, fix not done) | Material proof sheets (stone, wood); a second subject |

**What the images show and do not show.** The image set with this release shows the present state, flaws
included: the stepped lip border, the lash roots, the crease defects at the elbow, the dark slit between two
incisors in the earlier teeth render. They are single frames from this one machine, not comparisons against any
other tool. The earlier (2026-11-03/04) timing comparison against a path tracer is reported once, in Part 3 with its caveats (early
single runs, unequal materials); it is not a claim about the present tool.

---

# Part 10. Patent readings carried over, and what is held out of the code

**Read this first.** By decision, this release publishes flags only: it names each patent our review
flagged and says what was done about it, and it publishes no conclusion that our code does or does not meet any
claim. These are our own readings of claim text, made by non-lawyers. No patent attorney has been consulted. A search hit is a
screening flag, not proof; writing our own code or
measuring better performance is not protection; and a status such as "expired - fee related" is a dated note, not
a permanent fact. The searches were not exhaustive. Nothing here is a clearance, and nothing here is legal advice.

How a reading is recorded in this project (adopted 2026-11-12): each proposed restriction
names the claim element that causes the concern, and each design-around names the element that is absent.

## 10.1 Flags on code as it stands: held out of any public code until redesigned

| Our mechanism | Patent | What the patent concerns (as we read it) | State on 2026-11-12 |
|---|---|---|---|
| Contact solver, first generation: an IPC-style barrier energy minimised by Newton steps on the lattice (`engine/contact`) | NVIDIA US 12,216,969 | a primal energy minimisation of the contact problem with the preconditioned Newton-type update the specification describes | Flagged in our review; held out of the code pending redesign. Direction of the design-around: a constraint-based formulation that minimises no energy (Part 6, B.6), with its weak point stated there; a pending continuation is to be re-read at grant. |
| Look-dev light cache: near hits lit exactly, hits beyond 10 cm read a cached outgoing value | NVIDIA US 11,790,594, claims 16-18 | cached outgoing irradiance used according to the distance between locations (claim 18: a distance threshold) | **Redesigned 2026-11-12, 10:22 PT**: every gather hit is lit by one rule, no outgoing cache is read, no near/far switch by distance. Two earlier unwired prototypes that had the switch are held out. |
| Capsule primitive in the edit program, and the Grab brush's choice of which bricks to admit | MERL US 8,265,909 (to 2031) | distance of a shape swept along a path found through an optimal path parameter | Flagged in our review; held out of the code pending redesign. No design change chosen yet. |
| The card tracer's tile grid that skips empty tiles, and the band store | Movidius US 11,680,803 / US 11,222,459 (to about 2037) | per-tile empty / occupied entries over finer entries, with empty entries skipped | Flagged in our review; held out of the code pending redesign. Direction of the design-around: skip by stored distance rather than by an empty tag; not landed yet. |
| Elastic Grab (a prototype brush using a lattice solve of linear elasticity) | Pixar US 10,586,401 (to about 2038) | regularised solutions of linear elasticity with a specified Poisson ratio (the closed-form Kelvinlets are only dependent claims) | Flagged in our review. An earlier internal note that treated it as unaffected was withdrawn. The brush was never merged and will not ship. Direction of the design-around: a form-aware Grab with no elasticity equation (Part 5). |

Because the edit program, the tracer and the band store are central, the tool's code is not in this release.

## 10.2 Studied and set aside

| Method | Patent | What we do instead |
|---|---|---|
| Geodesic voxel binding for skinning weights | Autodesk US 9,836,879 B2, US 10,062,203 | volumetric bone heat solved on the lattice (Baran and Popović 2007), Part 7; our own geodesic bench module is study-only |
| Regularised Kelvinlet brushes | Pixar US 10,586,401 | not used; see 10.1 |
| Exact cubic in-voxel ray hit for distance grids | NVIDIA US 11,823,319 | marching with bracketing and bisection; one record names the elements as absent and another marks the reading as not verified either way, so it is treated as open |
| Cubic turning-point shadow rays; voxel-local hit storage | NVIDIA US 11,887,245; US 12,175,588 | not used; hits are kept in object or world space |
| Feedback photon splatting | NVIDIA US 12,315,070 | not used |

## 10.3 Read, and on our reading an element is absent

| Our mechanism | Patent | Absent element (our reading) |
|---|---|---|
| One tricubic gradient for normals in the tracer | NVIDIA US 12,254,556 | per-voxel stored normals and their interpolation (one record marks this reading as still to be re-checked for the review renderer's own normal path) |
| Hair fibre scattering after Chiang et al. 2016 | Pixar US 9,905,045 | fitted basis distributions sampled in closed form |
| Hair lobe sampling (logistic) | DreamWorks US 9,019,298 | Cauchy-based importance sampling |
| Per-lobe hair controls | Disney US 8,674,988 | recorded as "expired - fee related" on 2026-11-11; a dated status to re-check before any code release |

Guard rails kept in the hair code because of these readings: never fit a mixture of logistic, skew-logistic,
monomial or Cauchy distributions to a measured lobe and sample it in closed form (tabulate and sample by numeric
CDF instead), and never sample hair lobes by a Cauchy distribution.

On the records found, these published methods are **used** in what is built: sphere tracing (Hart 1996), dual
quaternion blending, bone heat, and the hair models of Marschner et al. 2003 and Chiang et al. 2016. These were
**studied, benched or run as probes only** and are not in the tool: bounded biharmonic weights (proposed, never
implemented), per-brick spherical-harmonic sky transfer and light tracing for caustics (probe results), photon
mapping, and non-linear sphere tracing (Seyb et al. 2019; benched only).

The position taken (2026-11-11 and 11-12): using a patented method is a legal risk; a patented method may be studied
and learned from, and then improved on; the work learns from existing research and builds its own mechanisms.

---

# Part 11. Future direction

Ordered roughly by what the tool needs first.

1. **The hands-on sculpting test.** A sphere to a rough head in under five minutes with the rebuilt
   toolkit, by hand. Not yet run on the rebuilt tools. It is the benchmark for the tool's first
   version.
2. **The toolkit rebuilt on its ten principles** (Part 5): every tool fully three-dimensional and independent of
   the view once its constraint is set; the constraint shown on screen; tools that touch only the sheet under the
   brush unless asked; a pencil for creases, ridges and cuts; sculpting two surfaces into contact without fusing.
3. **The store's next form** (Part 4): the stroke's whole turn on the card, then one sparse brick store live on
   the card with copy-on-write undo and a fine surface band; per-part lattices finer than today's 0.125 mm leaves where they pay.
4. **The design changes for the patent flags** in Part 10 (the capsule, Grab's admission, tile skipping, the band
   store), each recorded element by element, so the code can be released.
5. **Contact** (Part 6): finish the constraint-based solver; test the field-read barrier alone and in
   hybrids; settle why the elbow stalls near 85-90 degrees by one-change-at-a-time runs on the real arm; build the
   environment-field sensor only if its four open problems have answers; a broad phase for many bodies.
6. **Persistent state and compact encodings** (Part 2, sections 3 and 4): one small experiment each, measuring
   total cost including decode and writes.
7. **Posing** (Part 7): the elbow to 90 degrees and beyond in clay, then other joints; bones and
   muscles as field layers from medical measurements where a part needs them.
8. **Hair** (Part 8): the natural base accepted (it awaits acceptance today), then styles as states on the grown head (brush, hand, product,
   tie), in the tool's own window; brows and lashes that follow a sculpt.
9. **Light** (Part 8): the second bounce from the lattice; a portrait preset set; ray-traced-quality light on
   cards without ray-tracing hardware.
10. **Cloth** (Part 7): a tee settled to rest; garments as two-faced thin objects the brushes can tell apart from
    the body under them.
11. **Proofs on other subjects.** A materials library with examples (stone blocks, wood), from the 72 sourced material profiles already written (a further catalogue of 215 is planned only), and a native rebuild of an earlier animal model (a cat) in
    Field Studio. None of these proofs exists yet; they are the next evidence the project wants, to show the tool
    is not about one human model.
12. **A sibling runtime engine** built for these fields, and, much later, the project's larger aim: a persistent,
    causal world in which fields carry geometry, material and physics.
13. **The code release**, open source, once items 4 and the clean-up of the code set are done.

## Release record

- Prepared: 2026-11-12, evening, Pacific time.
- Public date: the time stamp of the public commit and posts that carry this document.
- Licence of this document and its images: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Author: Mathew Birchard - DCV Systems, LLC - @birchard_mathew.
