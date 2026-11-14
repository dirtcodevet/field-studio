[Contents](README.md) | [Complete single file](../RESEARCH.md)

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

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

