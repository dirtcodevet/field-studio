[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 1. Direction: what Field Studio is and why

**The outcome.** The author's standard, in his words: "extreme realism, but generally aligned with the reference.
Extreme, ultra realistic, life like, at every level ... if I look at your end result I should be looking at it and
saying, 'my god whoever took that photograph has extraordinary talent, it's unbelievably good!' No matter what
level I am zoomed to, no matter what body part." Section 6 of Part 2 turns that into a bounded test.

**The tool.** Field Studio is a sculpting, simulation and rendering tool in which every form is an implicit field:
a signed distance field stored on a sparse lattice at several levels of detail (0.25 mm at the finest level today).
There are no meshes anywhere: no triangles, quads or tetrahedra in the model, the tools, the renderer, the
previews or the tests (adopted as a rule 2026-11-07/08). Bodies are fields; hair is curves; cloth is yarn points on
a pattern lattice; contact, posing and light are computed on the fields directly.

**Why fields.** The industry's tools are built around meshes, and many of their tools exist to work around what a
mesh cannot do (adding surface means adding vertices; contact between two sculpted surfaces needs typed-in values;
skinning, light caches and collision all hang off mesh structure). A field has no topology to maintain, can be
cut, merged and grown freely, and gives every point its distance to the surface, which contact, light and brushes
can all read. The author's direction to every part of the work (2026-11-12): "we are a field-based system, not a
mesh ... The real test of our capability to achieve the targeted end result is whether we can truly capture the
full capability and uniqueness of what we're building, to the level of realism and completeness we've set as a
target, without having to do so on expensive hardware."

**Why a tool.** Until 2026-11-11 the project built one human model by direct engineering. On that date the aim
changed: build the tool, so that a person finishes the work by hand, quickly and intuitively. The first yardstick
is plain: a sphere to a rough head in under five minutes. The first runs did not meet it (Part 5), and on
2026-11-12 the author, sculpting himself, could not block out a head with the tools as they were. That led to a
review of the whole toolkit against an experienced sculptor's expectations (Part 5).

**Built to last.** Each subject should need less hands-on work than the one before: approved work becomes the
target that generators learn to reproduce from references (hair, skin, iris, teeth, garments, materials), stored as
a recipe (generator version, seed, settings) plus the author's edits.

**Open source.** The tool will be fully open source. No method that we read as covered by a patent is meant to
ship; where we find one, we study the published research and build a different mechanism, and record which claim
element ours lacks. Sources whose licences limit commercial use are used for study only.

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

