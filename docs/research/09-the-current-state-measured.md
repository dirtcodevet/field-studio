[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 9. The current state, measured

As of the evening of 2026-11-12 (Pacific). One line per subsystem: what works, what is slow, what is not done.
Each line is a summary of the part named; the numbers, cases and conditions are there. Nothing in this table has
been reproduced by anyone outside the project.

**The headline, stated plainly.** The tool exists and runs: a real window, a field store with versions and undo,
brushes on the CPU and on the card, parts that touch without merging, a light rig with a settling still frame,
generated hair, teeth and garments. It does not yet pass its own first test (a sphere to a rough head in under
five minutes), it misses its frame-rate targets on the full human model, and the author's own hands-on sculpting
test of the rebuilt toolkit has not happened yet. Several mechanisms described here are prototypes measured on
synthetic cases only.

| Subsystem | What works (evidence state) | What is slow or failing | Not done |
|---|---|---|---|
| Field store (Part 4) | Four-level sparse brick lattice, 0.25 mm finest; versions by copy-on-write; one writer transaction per edit; a seam guard and a publish gate (accepted in the workflow: the current human model is made through them; five real strokes written back gave 0 seams at every level where the earlier writer left 4,053) | Whole body at the finest level does not fit the 8 GB card at once; the full-session benchmark on the human model hit an out-of-memory fault on the card and missed the 30 frames per second and 100 ms targets; one save was interrupted by quitting (the sculpt was recovered from the temporary folder, and the save path was fixed) | The stroke's whole turn on the card; one live sparse store on the card with copy-on-write undo; per-part lattices finer than the 0.125 mm leaves that exist today |
| Brushes and toolkit (Part 5) | Draw, Carve, Inflate, Clay, Smooth, Swell to contact, Settle onto, protection of surfaces behind (implemented; measured on standard forms and on a thin-wall test). Grab is view-dependent and is being replaced; the rebuilt Pencil is faulty and has not landed | Test 1 not passed: the first scripted run made "a lumpy egg with a flat face" at 6.6-13 frames per second; on 11-12 the author could not block out a head; a screen-plane pull lost most of its depth off-axis (30.1 / 17.7 / 5.7 / 0.8 mm at 0 / 30 / 60 / 80 degrees) until the form-based constraint (29.4-29.9 mm at every angle, in a test copy) | The toolkit rebuilt on its ten principles; the author's hands-on test; form-aware Grab; design changes for two patent flags |
| Shell (Part 5) | The window, shelf, grids, section and cut tools, regions, marks (implemented; shown in the images) | | A designed product pass with the sculptor in the loop |
| Contact and parts (Part 6) | Parts as separate fields with touch / slide / bond rules; solid-by-default body with designed cavities; a barrier solver as baseline; a constraint-based prototype; the field-read barrier in two forms (all measured on named, mostly synthetic, cases) | The stand-in elbow stalls at 85-90 degrees (from a straight rest) under three different contact laws, cause not established; the human model's own elbow reached 90 degrees from rest (about 112 anatomical) with the solver at its limits, against a 145-155 degree target; soft-on-rigid favours the baseline (73 s against 7.4 s); the hand-on-cheek scene did not converge; the explicit form fails friction | A contact law of record; a broad phase for many bodies; the environment-field sensor (proposed only) |
| Posing (Part 7) | Skeleton, joint finder, volumetric bone-heat weights on the lattice, dual-quaternion map with inverse; drag at a median 16.5 ms a frame at 1080p on stored corrective states (measured on named cases) | Skinning alone folds at the elbow at every zone width; the posing corrective (an earlier, different setup from the contact study) reaches 60 degrees on the test arm and 55 degrees on the human model, with visible crease defects | 90 degrees and beyond; shoulders; bones and muscles as layers |
| Cloth (Part 7) | Body measurements and landmarks read from the field; yarn points on a pattern lattice; a swim-knit top built, settled and shown; a dress form (measured on named cases) | No tee is at rest yet | Garments as two-faced thin objects for the brushes; cloth in motion |
| Mouth library (Part 7) | Generated teeth, arch, gums, palate, jaw and tongue as field parts, placed in a head in the real window (measured on named cases) | The cheek-to-teeth space in the human model is unresolved (one run ended out of memory) | Seating published in the model; the life-like look |
| Hair (Part 8) | Grown once from follicles as curves with an inherited helix, settled to a rest gate, cached; a wave-banding fault found and fixed with measurements; fibre scattering after published models; exact shadow rays checked against a 144-ray reference (measured on named cases; the base awaits the author's acceptance) | About 6.5 minutes to grow a head on the card; 20-36 s a frame for the full-quality hair render; one variant failed its style gate; an earlier whole rebuild was archived as failed (10-31) and a built style was stopped on review (11-11) | Dynamics (statics only today); styles as states in the tool's window (styling passes are measured in the hair work area only); brows and lashes following a sculpt; the hair in the tool's own window at full quality |
| Light (Part 8) | Sphere tracing; a rig, panel and presets; a lattice light while moving and a gather that settles over 64 passes when still, with no denoiser; the cache redesigned on 11-12 around a patent reading, one bounce within 0.12 % of the all-exact reference on the test folds (measured on named cases) | The second bounce is missing (0.5-2 % in folds, about 35 % in dark pockets); no light setting reached the studied portrait reference | Light on cards without ray-tracing hardware: nothing measured yet |
| Skin, eyes, materials (Part 3) | Render-time micro-relief generated and varied; colour by zone; a seeded iris generator (measured on named cases; see the image set for the present look); 72 sourced material profiles committed, 75 on disk (implemented; the tool's viewport did not read them as of 11-11) | The lip colour border steps in 0.3 mm cells in close-up; grain at the lid margins from a stale copy of the eye in the body field (cause measured, fix not done) | Material proof sheets (stone, wood); a second subject |

**What the images show and do not show.** The image set with this release shows the present state, flaws
included: the stepped lip border, the lash roots, the crease defects at the elbow, the dark slit between two
incisors in the earlier teeth render. They are single frames from this one machine, not comparisons against any
other tool. The earlier (2026-11-03/04) timing comparison against a path tracer is reported once, in Part 3 with its caveats (early
single runs, unequal materials); it is not a claim about the present tool.

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

