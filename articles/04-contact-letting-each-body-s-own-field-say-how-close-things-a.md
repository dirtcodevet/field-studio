# 4. Contact: letting each body's own field say how close things are

*By Mathew Birchard - DCV Systems, LLC - @birchard_mathew*

Sculpting two surfaces until they touch, without them merging or passing through each other, is one of the
hardest things to do in mesh tools. Eyelids on an eye. Lips. A fingertip on a cheek.

In a field-based tool every body already carries its distance field. So here is the idea I put to the project on
November 12: let a point on one body read its gap straight from the other body's stored field, and push back
with a short-range repulsion that grows as the gap closes. I think of it as a magnetic field between surfaces.

The repulsion law is not mine. It's the published log barrier from Incremental Potential Contact (Li et al.,
SIGGRAPH 2020). Our earlier solver also reads gaps from fields; what's different here is how the solve is arranged around that law. Bodies settle at
small positive gaps (49 to 95 micrometres measured), not at zero.

Results on synthetic test scenes, on one machine, beside our earlier barrier solver on the same scenes:

- Two fingertip pads pressed 2 mm: 27 s against 86 s, forces within 1 %, nothing crossed.
- Lips closed 0.8 mm: 16 s against 27 s.
- An eyelid wrapping a rigid eye: 73 s against 7.4 s. A reversal. Soft on rigid favours the older method here.
- A hand on a cheek with a sideways check: did not converge in 10 minutes. Cause not found yet.
- An explicit version of the same idea: failed the friction test.

So: promising on soft-on-soft, not established as generally better, and I'd rather show the table than the
highlight.

A second idea is only proposed: one field through the empty space between bodies, used as a sensor for anything
approaching anything, the way a touch screen senses a finger. It has four open problems listed in the write-up,
self-contact first among them.

Write-up: https://github.com/dirtcodevet/field-studio

Images: `../images/model_segments/02_eye_left_closeup.png`, `../images/model_segments/03_lips_closeup.png`

*Prepared November 12, 2026.*

---
[All articles](README.md) | [The write-up](../docs/RESEARCH.md)
