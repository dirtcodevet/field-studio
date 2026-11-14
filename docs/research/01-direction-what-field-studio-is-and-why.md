[Contents](README.md) | [Complete single file](../RESEARCH.md)

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

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

