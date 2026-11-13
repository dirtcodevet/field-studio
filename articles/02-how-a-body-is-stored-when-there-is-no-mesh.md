# 2. How a body is stored when there is no mesh

*By Mathew Birchard - DCV Systems, LLC - @birchard_mathew*

In Field Studio a form is a signed distance field: at any point in space, how far is the surface, and which side
are you on.

Storing that for a whole human figure at 0.25 mm takes care. Space is cut into 32 mm tiles. Each tile holds
bricks of 8 x 8 x 8 cells, stored as 9 x 9 x 9 samples so that neighbouring bricks share a face and
interpolation never has to look next door. There are four levels, each twice as coarse as the one below
(0.25, 0.5, 1 and 2 mm). The fine levels exist only in a thin band around the surface. The human model today is
about 219,000 bricks at the finest level.

One rule reads the field everywhere, on the CPU, on the graphics card, in the writer and in every check: the
finest level that has a brick at a point answers.

Edits are transactions. A brush stroke changes samples, and the writer then keeps every level consistent and every
shared brick face identical on both sides. That last part bit us: an earlier writer left mismatched faces
("seams"), thousands of them after a few strokes. The fix was at the cause, in the writer, plus a guard that
repairs and counts stale faces at commit and a gate that refuses to publish a version with any. Five real
strokes written back through the tool: 4,053 seams with the old writer, 0 with the new one.

Versions are folders that share unchanged files, so a new version costs only what changed, and undo is exact.

What's slow: the whole body at the finest level doesn't fit an 8 GB card at once, and the full sculpting session
on the human model missed its targets of 30 frames per second and 100 ms to the next stroke. The next step is to
move the whole stroke onto the card, then keep one sparse store live there.

Credit where due: sparse brick storage follows a long line of published work (adaptively sampled distance
fields, Frisken et al. 2000; VDB, Museth 2013).

Write-up: https://github.com/dirtcodevet/field-studio

Images: `../images/tools_only/11_grids_before_after.png`, `../images/tools_only/09_window_section_mouth_parts.png`

*Prepared November 12, 2026.*

---
[All articles](README.md) | [The write-up](../docs/RESEARCH.md)
