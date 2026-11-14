# 8. Reading patents while building in the open

*By Mathew Birchard - DCV Systems, LLC - @birchard_mathew*

Field Studio will be open source. An open licence doesn't protect anyone from a patent, so we read patents as we
build. I'm not a lawyer, no lawyer has reviewed this, and nothing here is legal advice or a clearance. These are
our own readings of claim text.

What our review flagged, and what we did about each. These are flags, not conclusions about any claim:

- Our first contact solver, a barrier energy minimised with Newton steps: NVIDIA US 12,216,969 was flagged in our
  review. The solver is held out of the code pending redesign. A constraint-based replacement is underway.
- Our light cache used a distance switch: near hits lit exactly, far hits read from a cache. A second check
  found a claim in NVIDIA US 11,790,594 that our first check had missed. We redesigned it the same morning:
  one rule at every hit, no switch.
- A prototype "elastic grab" brush: Pixar US 10,586,401 was flagged in our review. An earlier note of ours had
  treated the brush as unaffected; that note was withdrawn. The brush was never merged and won't ship.
- Two more flags sit on central pieces (a capsule primitive and how empty space is skipped). Those are being
  redesigned, and until they are, the code stays unpublished.
- For skinning we set aside geodesic voxel binding (Autodesk) and use published bone heat on the lattice.

Lessons I'd pass on: a search hit is a flag, not proof. Read every independent claim, not the first one.
Writing your own code is not a defence. And, as we understand it, a dated public release only helps against
patents filed after it.

That last point is one reason this write-up exists. Publishing it is defensive only: the methods are described in
detail, with dates, so they are on the public record.

Write-up: https://github.com/dirtcodevet/field-studio

*Prepared November 12, 2026.*

---
[All articles](README.md) | [The write-up](../docs/RESEARCH.md)
