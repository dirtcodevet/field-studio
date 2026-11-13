# 7. Built, not painted

*By Mathew Birchard - DCV Systems, LLC - @birchard_mathew*

One rule runs through every part of this project: a look has to come from the structure that produces it.

Skin relief is generated and displaced at render time, not painted on. Creases are geometry. An earlier iris with colour as a layer over grey
fibres looked wrong the moment the eye wasn't blue, so the plan is for the fibres themselves to carry the colour.
That part is proposed, not built yet. Blurring something to make it look soft is painting, and it isn't allowed
unless blurring is what really happens.

The reason is practical. A painted look holds from one view, in one light, at one zoom. A built one holds when
you move the light, orbit, or come in close, and close-ups are the test here: "no matter what level I am zoomed
to".

The same idea shapes how the work is run. Define the outcome first, precisely, from references, and build it
directly. When something is wrong, fix that exact point at its cause instead of adjusting things around it until
it looks right. Measure before claiming. An approved result becomes the target a generator learns to reproduce:
a recipe of generator version, seed and settings, plus the hand edits on top, so an improved generator never
silently changes approved work.

The close-ups with this post show the present state, flaws included: the lip's colour border steps in 0.3 mm
cells, and the lash roots need work. Both are known and logged with their causes.

Write-up: https://github.com/dirtcodevet/field-studio

Images: `../images/model_segments/01_eye_right_closeup.png`, `../images/model_segments/03_lips_closeup.png`, `../images/model_segments/17_hand_back_skin.png`

*Prepared November 12, 2026.*

---
[All articles](README.md) | [The write-up](../docs/RESEARCH.md)
