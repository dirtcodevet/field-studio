[Contents](README.md) | [Complete single file](../RESEARCH.md)

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
- *accepted in the workflow*: in daily use, by the author's decision.

Nothing here claims novelty, protection or freedom to operate. Published methods are credited where they are used.

---

## 1. How the work is done

The project is one author, Mathew, working with AI coding sessions on one desktop computer. The practices below were
adopted one at a time, usually after something went wrong. All of them carry the state *accepted in the workflow*.

**1.1 One short charter (11-02) and a short list of principles (11-06).** A single charter of at most 30 lines
states the outcome in the author's words: "extreme realism, but generally aligned with the reference ...
if I look at your end result I should be looking at it and saying, 'my god whoever took that photograph has
extraordinary talent, it's unbelievably good!' No matter what level I am zoomed to." Every other document is a
reference, a tool or a dated record, to be weighed and not obeyed. A list of principles and machine mechanics was
approved on 11-06. The author's words are read in three tiers: an observation is a hypothesis
to test; an explicit direction holds for that task only; a standing rule exists only where he says it is one.

**1.2 Outcome first (stated 11-06, the default since 11-09).** Define the target outcome from the references with
precision, so that it holds in every view, light, zoom and motion, and build it directly. Causes (anatomy, physiology,
physics) come in as drivers of variation and states, or where the outcome is itself a behaviour such as contact or
motion. Physics is never tuned until a shape appears. This replaced a cause-first default. The cost behind it: skin
colour built up from chromophore causes kept producing blotches that were hard to trace, across rounds on 11-02, 11-06
and 11-07 ("hours", by the record). The author approved it on 11-09 with his own caveat: lanes that genuinely need
causes first are to say so plainly and build that way (`engine/lessons/outcome_first.md`).

**1.3 Built, not painted (11-09).** A look has to come from the structure that produces it (form, volume, translucent
layers, an optical medium, motion), so it holds when the light, view or zoom changes. Blur used to soften, baked or
painted creases and pores, painted shading and colour decals are not allowed unless that operation is what really
happens. Defects are fixed at the shape, never hidden.

**1.4 Fix a named problem at its cause (charter, 11-02).** When the author names a problem, the session finds what
causes that point and resolves it there. Making the surroundings look right instead is the one thing the charter
forbids. Two recorded costs. On 11-02, eight builds tried to shape a nose and an upper lip with numeric displacement
fields fitted to 2-D outlines, each stacking an offset on the last; after about 20 hours the nose was "clearly wrong".
A profile line says nothing about width, so everything off the line was guessed by falloffs. What worked was a real
3-D form: the author sculpted those features himself. On 10-31, a one-shot hair rebuild ran ten rounds of fixes that
each made the next artefact; he marked it a total failure and it was archived, not built on.

**1.5 Measure before claiming.** Numbers locate a fault and confirm a change; the rendered image and the author's eye
decide whether a form is right (`engine/lessons/measuring.md`). Pass marks are written before a run and not moved
after it. Every number is labelled measured, derived or estimated. Failures are kept as run. An instrument is proved
once on the failure it exists for: one closest-point routine with a one-token bug made 11-38 % of distances too small,
and its own self-check could not see it. In one overnight run about five results were withdrawn under these habits,
two of them over-claims by the lead session.

**1.6 The author sees each round and decides.** Each round reaches him as a numbered sheet beside the reference: the
current model through the real renderer at full quality, looked at by the session first. He decides what enters the
current model. The cost behind "the real renderer": on 11-03 a session's own render path moved the eyes of his
sculpted face, and he rejected the work outright. A face guard followed (a content hash over the field bricks of the
face region, checked before and after every edit). On 11-04 a review found the checksum then in use had been guarding
a superseded file outside the render path; it was replaced.

**1.7 One session per work lane, with a lead that plans and briefs (11-08).** In the overnight run of 11-07 to 11-08,
three lanes each owned by one session advanced, while the Field Studio integration, split into 30-minute hand-offs
between fresh sessions that each re-learned the plan, barely moved in four rounds
(the project's lesson record of 11-07). Since then one session owns each lane continuously; a lead
plans, briefs and arbitrates and does not carry lane work. Briefs carry the outcome, the author's words, the machine
constants and a "done when", not a method. Other costs from that night: flags lost in short summaries, and a stuck GPU
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
is *not established*, each lane's state, and the decisions waiting on the author. Incoming sessions take the work "as
it actually is, not how it is stated to be", and look at the latest images before trusting a status line.

**1.10 Sources and licences (since 10-26).** No source or asset whose terms limit commercial use is used, except to
study. Facts and numbers from a paper are used as facts; its text, figures, data and code are not copied. One ledger
(`engine/LICENSES.md`) lists each source with its licence, credit line and whether it ships.

**1.11 Mesh-free everywhere (11-07 and 11-08).** No triangles, quads, tetrahedra or element meshes in the model, the
tools, the renderer, previews, tests or scaffolding. Bodies, cloth, hair and fluids are fields, lattices, particles or
curves. The steps: vertices dropped for the model on 11-03; "zero meshes allowed anywhere ever" on 11-07; test benches
included on 11-08. What it cost: a 2-D finite-element bench used for material proofs was retired that night, and a
real-time web viewer with a mesh skin was archived.

**1.12 Patents (11-11).** The author intends an open-source release. A patented method is studied, not shipped. Where
a published patent appears to cover something the tool needs, the lane designs a plainly different method and records
its reading of the claim elements one by one. These are engineering notes by non-lawyers; the searches are not
exhaustive, and no clearance is claimed.

---

## 2. The path

### 2.1 The mesh era in Blender (to 11-02), kept as record

The earliest dated build records are from 10-19. The human model was a polygon mesh in Blender, rendered with Cycles.
Its base body came from a free base model from Blender Studio (Human Base Meshes v1.4.1; every asset's licence field
says CC0, per `engine/LICENSES.md`), and the author sculpted the face on it. This era left procedural skin relief
driven as render-time displacement, seeded iris and hair tools, and the measuring lessons above. It also hit a wall:
at full skin detail Blender ran out of memory on every close frame on this machine (12.25 GB) and finished 31 of 120
frames in 30 minutes.
**Evidence: measured on named cases** (one model, one machine).

### 2.2 Vertices dropped (11-03)

At 00:12 the author wrote that treating a human form as millions of vertices with modifiers "makes realism detail work
almost impossible". At 17:47 he decided: "I want to get rid of the entire vertex-based framework." Cage, subdivision
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
and the cache is valid only while light and subject stay still. Its own verdict was "yes, partly". A later round found
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

**Tolerance.** On 11-05 the author allowed his sculpts to be rebuilt natively "within that small tolerance", about
20 um (pass mark: 1 mm-averaged offset, 99th percentile at most 20 um), in place of byte copies.

**What happened.** The neck was native by 11-05 23:45. The first full assembly, on the night of 11-06, was refused by
the writer, as designed: two joins did not fit and three had no overlap. By the 11-06 roadmap, six of ten regions were
native in one full-body cache that matched the old version on the GPU (mean difference 0.01 to 0.22 of 255); the rest
were blocked where three or more parts meet or slits close. One limit was reported plainly: where the old mesh's
stair-steps sit on a sculpted fold, holding the sculpt within 20 um means re-drawing the stair-steps. Whether every
region finally met the 20 um bar is **not established** in the records read for this part. On 11-09 the author added
that no part is sacred, his sculpts included: flaws are fixed and small shifts are fine.
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

On 11-11 the author changed the main effort. A round on the lips had come back with the lips pressed together, "the
sort of thing that a human would not have done". His conclusion: the sessions are strong at code, physics, measurement
and research, and weak in the fine visual refinement loop. So the work became "the actual base, the tool, the studio,
and all of the components it needs ... so that a human can finish the work", within 16 GB of RAM and an 8 GB card for
the whole system.

**Test 1.** With front and side reference images on one centre axis, sculpt a sphere into a basic head (head shape,
eye indents, nose, cheeks, chin and jaw line) aligned with them, in under 5 minutes (first said as 2). Scripted
measures: outline mean within 3 mm, lumps at most 1.5 mm RMS, at least 30 frames per second while stroking, at most
100 ms to the next stroke.

**How the first runs went.** A scripted run in the real window on 11-11 at 10:47 made "a lumpy egg with a flat face",
after 144 actions and an estimated 7.8 minutes. Four passes of Smooth took the lumps from 4.49 to 4.23 mm RMS (6 %).
Strokes ran at 6.6 to 13 frames per second. A neck pulled down with Grab tore into an open ring. 22 of 166 presses
near the silhouette turned the view instead of sculpting. The author's first comment was about the driver: "You used
too small of a brush". A sculptor blocks in with a brush a third to half the size of the head, and the tool capped
brush size and ran large brushes slowest. Since the run could not tell tool limits from AI limits, each tool was then
measured on standard forms against its equivalent elsewhere. By the 11-12 morning hand-off, 30 frames per second for
small strokes was still not established, and the author's own first hand run of test 1 was still open.
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
blotching. The author's direction: a natural base tone per zone from living references; low natural variation matched
per zone on the band ruler; sparse, faint accents; then triggered states. Causes stay for the states.
**Evidence: accepted in the workflow.**

**The seeded skin generator (started 11-10; `engine/skin/gen`).** Inputs: surface samples with normals and soft zone
weights, a few head landmarks, and about 20 settings (seed, age, sex, base tone, sun, redness, variation, freckles,
pores, texture, per-zone overrides, local refinements). The same seed and settings give the same skin; another seed
gives another skin of the same kind. Colour is a regional base smoothed along the surface, plus tan by exposure, four
seeded bands of variation, and freckles and moles as a seeded point process of irregular soft patches. Relief comes
from seeded growers scaled by age and sex. Output is evaluated at any point and written as lattices aligned with the
model's bricks. The author approved it as the model's default on 11-11. It has no panel in the tool yet and is
re-baked after a shape change.
**Evidence: accepted in the workflow.**

### 2.8 Eyes and iris

**A volume built from fibres (11-03 onward).** The iris is a heterogeneous scattering volume built from generated
fibres at their real scale, over a dark back layer, under an analytic cornea. Globe, cornea and limbus are formulas
(two spheres). The fibre volumes are sparse: one eye's arrays went from 425 MB dense to 102 MB.
**Evidence: measured on named cases.**

**Why four rounds read as a stain (analysis 11-05).** A brown inner zone had read as a circle, then a splat, a
watercolour stain and a starburst. The author asked for a pause to study reference irises: colour, then texture, then
structure. Three independent measures agreed. In real irises the amber is the bright material: brownness and lightness
correlate at about +0.5 at fibre scale. In ours it was a dark absorbing film: -0.3 to -0.7. The amber added absorption
and no scatter, and its edge was drawn across the fibres instead of being made by them, so no edge shape could fix it.
The layering was checked against anatomy literature (Yan et al. 2022; Wilkerson et al. 1996; Imesch et al. 1996; Sturm
and Larsson 2009). The reference photographs were iris macros under CC BY and CC BY-SA licences.
**Evidence: measured on named cases** (three reference irises; three others were too soft to resolve the zone).

**What the r22 draft got wrong (11-09).** It failed its own gates as run (9 of 13 texture gates, 6 of 10 geometry
gates, parallax failed). The author's read: "Ours blur, rather than being natural, and are too wavy in a consistent
way", too symmetrical at the inner border, with fine detail "all but absent". Blur used to soften is painting (1.3).
A fresh session took the lane.

**The seeded generator (11-09 17:40; `engine/eye_lane/iris_gen.py`).** His direction: a seed and settings that give
different irises (colour variation, patterns, crypts), not one eye. Everything is drawn from (seed, settings) on a
5 um grid over the iris plane:

- a collarette as a lobed, notched crown curve with a ridge that varies along it, never a circle;
- traced strands at several depths: ciliary bundles that branch and merge along a shared flow and bend round crypts,
  fine radial fibres in the pupillary zone, and a fine layer everywhere;
- crypts as seeded lens-shaped pits floored by the deep stroma;
- broken, wavy contraction furrows, and a beaded pigment ruff at the pupil margin.

Relief and collagen density come from distance fields to the strands. There are about 80 settings and four presets.
**Evidence: measured on named cases** (two to four seeds shown beside reference photographs).

**Colour carried by the fibres (the author's verdict, 11-09 about 21:30).** The blue irises worked because their
colour comes out of the fibre structure itself. For green, hazel and brown, a separate pigmented tissue had been laid
over the strands; it hid the fibre architecture and read as mud. "That was the single action that drove the failure of
effectively every eye except the blue." The direction: one fibre architecture for every colour, with the pigment
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

**Labels carry the material (design, 11-10).** A label is the front door to a profile: "I label that as an eye ...
Field Studio knows what an eye is." Labelling a part applies its material, physics and generator. Proof sheets for
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

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

