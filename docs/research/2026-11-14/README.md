# Research since the first release: materials, water, fabric, contact and hair

> **Research disclosure.** This is a dated public disclosure of our own research and concepts, published while the
> work is under way. Each method was developed with checks for conflicting patents and rights; where a check found a
> conflict, we changed the method ([the methods we changed](../../PATENT_STORY.md)). This is not a claim of freedom to operate
> and not legal advice.

This section collects specific research done for Field Studio between 2026-11-07 and 2026-11-14 that the first release
did not cover, or covered only as a plan: what each test was, its numbers and the conditions they hold under, how far
the evidence goes, and what failed or is still open. It extends the first release's research write-up
([RESEARCH.md](../../RESEARCH.md)) and the status review ([STATUS_2026-11-14.md](../../STATUS_2026-11-14.md)); it does not
repeat them.

**How to read it.** Every finding carries an evidence state: *proposed* (an idea, not built), *implemented* (built, not
measured against anything), *measured on named cases* (run and measured on the scenes named), *reproduced under a frozen
configuration*, or *accepted in the workflow*. Nothing here has reached the last two. Pass marks were written before the
runs they judge, and a run that failed its mark stays a failure even when the cause was later found and fixed. Most
numbers are single runs on one desktop machine (one 8 GB graphics card) with other jobs running. Material numbers carry
a quality letter: M measured in the cited source, D derived here from measured data, E estimated. Several topics below
were measured in separate engines and test benches that Field Studio will call; each file says plainly what is and is
not in the tool's window.

## The topics

**[Materials: the library rendered, and the measurements behind it](materials.md).** Six masonry materials (sandstone,
limestone, granite, marble, slate, fired brick) now carry their structure in the field (relief, chips on the arris, a
crumbled edge, cracks with depth, iron staining in the volume, bedding) and their measured colour in the renderer. The
colours come from about 95 public reflectance spectra and from published CIELAB measurements of building sandstones.
A colour complaint was traced end to end before anything changed: the render carried the measured colour intact (R/G
1.07 against 1.08), and the cause was a grey sample, fixed by adding measured warm stones. Materials can be weathered,
soiled and rained on as states. Wood, glass, metal and fabric exist as measured profiles used by the solvers, without a
look in the tool yet. 13 images.

**[Rain and water drops on glass and stone](rain_on_glass.md).** Where a drop sits and what shape it takes on each
surface (Young-Laplace, from measured contact angles); when it slides (a matched test on PMMA: 20.5 against a measured
24.1 degrees); how a drop spreads when it hits glass (converging to within 4-8 % of a matched measurement once the
spreading sheet is resolved, after the written test failed at a coarse grid); and what ten minutes of rain does to tilted
marble, sandstone and clean glass (clean glass forms a film; marble keeps pinned drops; sandstone soaks and darkens).
Also the corrections: a published spread fit that had been measured on steel, not glass, and splash thresholds that do
not hold on smooth glass. There is no rain in the tool's window yet. 4 images.

**[Fabric under weights](fabric_under_weights.md).** A 50 mm ball of 100 g, 500 g and 2 kg resting in a clamped square
of eight fabrics: no crossing, and the sag follows the membrane law's cube root of the load (exponents 0.313-0.316 for
the wovens), ordered by measured stiffness. The same ball dropped from 10 cm (rebound 0.80-0.87, an upper bound because
fabric hysteresis is not modelled yet). The Cusick drape test against measured drape ranges (7 of 8 pass, all a little
low without self-contact), and a stretch band on a soft limb matching Laplace's law within 5.1 %. 4 images.

**[Contact and motion](contact_and_motion.md).** The elbow on the human test model's arm, where three causes of the
stall were measured and fixed and the skin's own in-plane stiffness carried the bend cleanly to 129.7 degrees
anatomical (the 145-155 degree target is not reached); sliding skin (open); the author's hard-stop contact law (zero
force while apart, never crossing, never pulling), shown to hold exactly as a law in an active-set form; warm starts
from stored solves (99 iterations against 351); a fist closed by contact; and six of the author's concepts
tested with marks declared first (the environment field as a broad phase and conservative advancement on the fields
passed; compact display encoding failed). All of it in working copies, none in the tool. 7 images.

**[Hair in motion](hair_motion.md).** The first hair dynamics: a breeze and a fan, settling after a gust (damping near
0.15, at the low edge of laboratory measurements), the finding that strands entering the skin come from how drawn strands
follow their guides rather than from the guides (fixed: 0 points inside on every frame), and a combined modal method
(about 10 times faster in wall time than full dynamics, but not identical motion). Mostly one day, on the CPU, on a patch
at the temple; nothing moves in the tool's window, and the rendered clip did not show visible motion. 6 images.

## Images

The images are Field Studio's own renders and plots, and the solvers' own sheets. Where an original sheet set a
reference photograph beside our render, the photograph is cropped out; each caption says so. The human test model
appears only as an arm, a hand, a head with hair, or sparse points, and each caption names the part.

---

Mathew Birchard - DCV Systems, LLC - @birchard_mathew
