# 9. The current state, and what comes next

*By Mathew Birchard - DCV Systems, LLC - @birchard_mathew*

Where Field Studio stands on November 12, 2026, on one desktop with an 8 GB card:

Works: a real window with a field store, versions and undo. Brushes on the CPU and the card. Parts that touch
without merging. A light rig whose still frame settles over 64 passes with no denoiser. Generated hair, teeth
and gums, and a first garment built from measurements read off the body.

Doesn't yet: pass its own first test (sphere to rough head in under five minutes). Hold 30 frames per second on
the full human model. Bend an elbow anywhere near its real range (about 112 degrees so far, against 145). Move hair. Bring a tee to rest. Light a second
bounce. And my own hands-on sculpting test of the rebuilt tools is still to come.

Next, roughly in order:

1. The hands-on sculpt test with the rebuilt toolkit.
2. The stroke moved wholly onto the card, then one sparse store kept live there.
3. The redesigns that let the code be released.
4. Contact: finish the constraint solver, test the field-read barrier further, settle the elbow.
5. Hair styles as states, and hair in motion.
6. Proofs on other subjects: a materials library with stone and wood, and a native rebuild of an earlier cat
   model, to show this is a tool and not one character.
7. Good light on older cards without ray-tracing hardware.

Further out: fields that carry persistent state (temperature, moisture, wear) advanced only when something
depends on it, and a runtime engine built for these fields.

The full write-up has the methods, the numbers, the failures and the sources: https://github.com/dirtcodevet/field-studio

Images: `../images/model_segments/15_portrait_light_preset_clay.png`, `../images/tools_only/08_teeth_gums_front.png`, `../images/model_segments/06_head_side.png`

*Prepared November 12, 2026.*

---
[All articles](README.md) | [The write-up](../docs/RESEARCH.md)
