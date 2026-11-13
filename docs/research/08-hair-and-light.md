[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 8. Hair and light

**How to read the dates.** Times in this part are internal provenance times: Pacific time, taken from the project's local commit log and dated lane records (short commit ids are given where one exists). They say when a mechanism was first built or measured inside the project. They are not publication dates. The public date of this text is the release date of the document it sits in.

**How to read the status words.** Each mechanism carries exactly one of: *proposed* (designed, not built), *implemented* (built, runs, not yet measured against a reference), *measured on named cases* (numbers exist for the cases named), *reproduced under a frozen configuration* (the same inputs gave the same output again, bit for bit where stated), *accepted in the workflow* (in daily use and approved in the author's review).

**Machine.** Everything was measured on one Windows PC with 32 GB of RAM and one 8 GB RTX 5050, shared between several work sessions through a queue. Timings taken while other jobs waited are marked where the record says so. "The human model" is the project's one test subject: a female head and body held as a signed-distance field on a sparse multi-level lattice (levels of 8 mm bricks down to 0.25 mm cells).

**Mesh-free.** A hair is a curve: a centreline polyline with a radius and a material frame. No triangles, ribbons, cards or tubes are built anywhere, including in previews and tests. The body the hair collides with and is shadowed by is the field itself, sampled into distance lattices.

---

## D1. Hair

### D1.1 What failed first, and why the design changed

Three earlier attempts are part of the record.

- **A rest-groom rebuild, archived 2026-10-31.** Its fixes were made mechanism by mechanism but judged only on full-head renders, and untested pieces went straight onto the head. The author's review: it "totally failed". The lesson taken: prove each piece on one lock or one fibre before a head.
- **A built style (2026-11-10 to 11-11).** A groom generator placed locks by a specification: an envelope of layer heights, Poisson-disc lock centres, a smooth held top. It matched a reference photograph's tone and contrast measures at lock and section scale, but the author's review on 11-11 08:14 was that it read "as a mound at the top and front", with locks "all basically the same size and too symmetrical" and a top that looked smoothed with hair spray. It was stopped there. Status of that generator: measured on named cases, rejected.
- **The foundations audit (2026-11-11 ~00:10).** Each foundation under the look was measured or read in code. Faults found and fixed at their causes: (a) every point of a strand used its *root's* skin normal as its outward direction for sky light, so lengths hanging 30 cm below the root were lit by where they grew, not where they were; (b) the key light's size widened the skin's highlights but not the hair's; (c) one roughness value along a fibre that is about three years old at 40 cm; (d) every cut hair tapered like an uncut one; (e) 8 % of points bent tighter than a 12 mm radius; (f) thousands of strands ran through the body (2,177 and 4,961 in two sets) where the body lattice had no gradient; (g) the biggest tool fault, found ~01:10: the 3 mm body lattice was a true distance only within about 10 mm of the skin (the store clamps at ±12 mm), so it read 13 mm where the truth was over 30 mm. Every envelope number taken before that used a faulty ruler. The repair is in D1.9.

After these, the direction became: grow a natural base from causes first, and treat every style as a state applied to it (the author's agreement, 11-11 07:57).

### D1.2 Grow once, style as states

*First designed 2026-10-31 (commit 0de30c0f); the natural base on the human model 11-11 10:36. Status: measured on named cases. The base head drew the author's first comment "This is a good baseline for the hair" (11-11 10:38); the current base variant still awaits his acceptance (his review of the latest comparison sheet is open). The styling passes are measured on named cases in the hair work area, not in the tool's window.*

Hair is grown once from its causes: follicle position and exit angle, the fibre's inherited type, stiffness and weight, sebum creeping from the root, water, wear with age along the fibre, and a cut. The result is a rested natural head, saved. A style is a sequence of physical passes on that saved head (brush strokes, finger passes, a water set while drying, product). No pass holds anything once it ends.

**Three stages, three owners.** Hair (follicles, fibre, lengths, cut, sebum, natural wave, growth, the gated rest) produces the grown-head cache. Style (passes, water set, release, final settle, child hairs) produces a styled state. Colour (melanin, pigment along the fibre) can be redone without regrowing.

**Layer files** (`engine/hair/layers.py`). A look is five JSON files: hair, style, colour, environment, simulation. Every setting has exactly one owner layer (longest-prefix rule). A layer file that holds a key another layer owns is refused, so a style cannot re-tune a fibre or a follicle.

**The natural base as built on the human model:** 137,865 hairs at about 225 per cm² on 608 cm² of scalp; 1,023 guide hairs settled as rods; child hairs interpolated; locks formed by wet cohesion and not placed. Acts: a soft side part, one finger-comb, fingers raking wet hair off the face, air-dried. No product, no root lift. Two changes requested in the author's review of the first base were made at their causes: front follicles turned back and to the side (fading over 70 mm inside the hairline) so the hair sweeps off the face, and the cut raised to just past the shoulders. Hairs crossing the face region fell from 2,196 to 265.

**Order of the style stage:** load the grown head (fingerprints checked) → product → brush strokes → hand passes → water set → release every act → release solves with friction anchors kept → final gated settle → child hairs → form → baby hairs → colour → a 0.2 mm push-out from the body.

### D1.3 Follicles and hair kinds by region

*Follicle exit model 2026-10-30; kinds table 2026-11-05 (commit 63d07d7f); follicle map on the field 2026-11-10 (b9ef7045). Status: scalp, brows and lashes measured on named cases; vellus implemented as data, with one test crop rendered and no review recorded.*

**Follicle map** (`engine/hair/follicle_map.py`). Samples on the field surface about 2 mm apart, each with a weight per hair kind, a density and a flow direction in its tangent plane. Follicles are drawn per sample as a Poisson count of density × weight × sample area, placed uniformly in the sample's tangent disc and re-seated on the field, so no grid shows in the roots.

**Exit angles** (angle between skin and hair): mid-scalp 35°; whorl 70° fading as exp(−d/18 mm); occiput 45°; hairline front 15°, temple 10°, nape 25°, blended inward over 25 mm. Jitter is logit-normal on [0°, 90°] with sd 4° at the mid value; azimuth jitter sd 8°. The settle clamps each guide's first segment to its follicle direction.

**Kinds** (`engine/hair/kinds.py`). Each kind is a row of 27 traits, each trait with value, range, unit, an evidence label and a source key. The kinds used on the head, face and arms, with the model's values:

| kind | root diameter | length | section a/b | shape | exit angle |
|---|---|---|---|---|---|
| scalp | 64.6 µm | cut | 1.3 | wavy type (D1.4) | by region, above |
| scalp edge (baby hairs) | ~35 µm | 38.5 mm | — | droops (longer than its 28.9 mm gravity length) | 15° |
| eyebrow | 60 µm (estimate) | 10.5 mm | 1.2 | single arc, curve diameter 30 mm | 12° |
| upper lash | 71 µm | 7.3 mm | 1.21 | arc, curve diameter 12 mm | 75° to lid skin |
| lower lash | 60 µm (estimate) | 5.6 mm | 1.21 | arc, 20 mm | 75° |
| facial vellus | 16 µm | 1.0 mm | 1.1 | straight, nearly colourless | 15° |
| arm vellus / intermediate | 25–40 µm | 2.9–4.2 mm | 1.05–1.1 | straight or shallow arc | 20° |

Every kind except scalp hair and its edge is shorter than its own gravity length, so lashes, brows and vellus keep the shape their follicle and curl give them and need no gravity solve. Pigment per kind is the scalp root's melanin times a kind factor, then scaled by diameter because absorption is stored per unit radius. The face set in use holds 150 upper and 52 lower lashes per eye and 836 hairs per brow. Weak data, stated: no healthy-adult brow fibre measurement was found; lash curvature has no published radius.

**Known gap:** the brow and lash set is a fixed strand set made on one version of the skin. A sculpt that moved the brow skin 0.65 mm buried 1,521 of 1,672 brow tips. A carry step (each hair point keeps its height over the skin, 2 Newton steps, residual p99 0.026 mm) is implemented; growing face hair from follicles on the live field is proposed.

### D1.4 The fibre: an inherited helix, and the curl index as the waviness lever

*Types 2026-10-31; the lever finding 2026-11-12 about 13:30. Status: measured on named cases (single hanging fibre); full head at higher curl index not verified (D1.7).*

In the frame carried along the strand, the rest curvature is κ0·(cos θ, flat·sin θ) with θ = τ0·(s − start) + phase, ramped in over `ramp_mm`. κ0 = 2/CD where CD is the curve diameter. τ0 comes from the curl index i (stretched length over end-to-end length): for a round helix τ0 = κ0/√(i² − 1); for a flattened one, by bisection on the measured index of the generated path. So at fixed i the shape is self-similar in CD: free amplitude ≈ (i² − 1)·CD/2, wavelength ≈ π·√(i² − 1)·CD. The classification follows de la Mettrie and Loussouarn. Design values: wavy = CD 48 mm, i 1.10, flat 0.25, a/b 1.3, lock of ~30 hairs, bond 0.1; curly = CD 10 mm, i 2.0, lock ~50, bond 0.35. Lock size and bond strength are stored as traits of the type: clumping is a property the fibre carries, not a style.

**What was wrong first.** To get "a bit more waviness" (author's review, 11-12 about 12:10) the first lever tried was a tighter CD (48 → 44 mm). Measured on the head: no gain. One hanging fibre, 250 mm, dry, rms offset from the straight hang: CD 48/i 1.10: 4.75 mm; CD 44/i 1.10: 4.59; CD 40/i 1.10: 4.61; CD 40/i 1.15: 5.67 (+19 %); CD 35/i 1.20: 6.32 (+33 %). A tighter CD alone makes smaller, shorter waves. The lever is the curl index. Gravity stretches the hanging wave: about 89 mm against 69 mm free.

### D1.5 The settle, the rest gate, the cached head

*Head settle 2026-10-30/31; single-rod core 2026-11-09 (3bdba016); field collider 2026-11-10. Status: measured on named cases; gate pass or fail recorded on every run.*

**Single-rod core** (`engine/hair/rod.py`). A discrete Kirchhoff rod in reduced coordinates: the unknown per joint is the Darboux vector (two curvatures and twist) in the material frame; frames and positions are integrated from the root, so length and frame orthonormality are exact. Energy = bending and twist against the rest shape plus weight. Statics by L-BFGS. Checked against Miller, Lazarus, Audoly and Reis 2014: within 1.1 mm of their elastica; the planar-to-3-D threshold matched theory to 3e-6. One of the six proofs in that first chunk failed at the time and is recorded as failed in the commit.

**Head settle** (`engine/hair/settle.py`). Per guide, one free 3-vector per segment (normalised to a direction), nodes integrated from the root. Segments graded from 0.5 mm at the root ×1.2 up to 14 mm. Energy terms:

- bending against rest turning (natural helix plus any water set);
- gravity;
- body contact ½·k·(0.5 mm − SDF)₊² on a quadratic B-spline of the body's 3 mm distance lattice, with a 1 mm lattice over the head added 11-11 after 565 of 228k sampled points were found inside the true field following a comb over the ear (0 after);
- hair-hair pressure from fibre fraction φ splatted on a 3 mm lattice, P = k·E·φ³ (van Wyk), k 1e-5, E 4 GPa dry;
- cohesion from oil and water bridges, −c·φ_w², sebum creeping 50 mm per day from the root, surface tension 30 mN/m for sebum and 72 for water;
- lagged, smoothed Coulomb friction with ε 1 mm, μ 0.5 on skin, 0.25 hair on hair wet, 0.13 dry;
- the passes' own energy through a hook.

Solver: L-BFGS-B in scaled variables. Growth is by continuation: the active arc grows through 30, 60, 100, 160, 240, 330, 430 mm, then the longest guide, 40 evaluations per stage; release 135 evaluations; the final stage is capped at 4,000. The whole evaluation also runs on the card.

**Friction anchors across a lag.** After each sub-solve, a sample that moved less than ε in its contact plane has stuck and keeps its anchor; one that slid gets an anchor trailing ε behind, so it keeps about 0.71·μN against further sliding. Without this, a held layer creeps ε per lag.

**The rest gate.** A settle is converged only when both hold: the largest unbalanced transverse force at any node, divided by the larger of the local weight and the summed term forces, is under 0.05; and the relative energy change over the last 20 evaluations is under 1e-4. No render is made of a style that fails.

**The cached head** (`engine/hair/head.py`). Saved: rod directions, rest turning, wave blocks, friction anchors. Key: SHA-256 over the sorted settings of the hair, environment and simulation layers, the collider, and hashes of the growth code. A rebuild whose key or array fingerprints differ is refused. Observed: the guard refused a run whose follicle directions differed.

### D1.6 Waves: the phase fault found and fixed

*Banding measured 2026-11-12 about 02:20; phase field 07:17 (077d0d39); jitter fix 09:26 (0df28b28); phase-aware child blend 10:05 (2ffb3d9f); wave-only shift 12:10 (b6e2d95d). Status: measured on named cases.*

**The fault.** The approved base (head "n2c") gave each 35 mm drying section one wave phase. That size came from barrel styling; applied to an air-dried head it put waves in step in horizontal bands. Measured: per 30° sector, the mean wave held 0.12–0.72 of the strands' wave variance (random phases give ~0.0002).

**The fix: a coherent phase field over the roots** (`form.phase_field`). Random Fourier features on 3-D root positions: 24 wave vectors drawn N(0,1)/scale, offsets uniform on [0, 2π), f = Σ cos(X·k + b)/√12, phase = π·f + N(0, 0.15 rad). At scale 32 mm: coherence 0.83 inside a lock, 0.41 at 8–15 mm, about 0 beyond. A first scale of 12 mm broke coherence inside a lock (0.47) and was dropped. Measured on the comparison sheet of the two heads (n2c before, n2d after): the wave correlation between sectors 90 degrees or more apart went from 0.12 to -0.01, and the even horizontal bands at the side and back no longer appear.

**Three follow-on faults, each found by measuring.**

1. *Child jitter read against the wrong reference.* Each child's own jitter was read against its old section's phase, which put a section border's random step into 51 % of children (shift p50 0.61 rad). Read against the smooth field instead: p50 0.14 rad, 0.4 % above 0.6.
2. *Blending guides that are out of step flattens the wave.* Children blend the 4 nearest guides; with guides at different phases they kept only 80–83 % of the guides' wave. Now, for the blend only: shift each guide to its section's mean phase, blend, then shift each child from the mean to its own field phase. Kept: 90–93 %.
3. *The shift moved the whole strand.* The first shift slid each segment's direction along the arc, which also moved the strand's large bends and looped crown hair up. The replacement moves only the wave: resample by arc; measure the strand's own hanging wavelength from zero crossings (clamped to 0.8–1.8× rest); take the mean line as a one-wavelength box mean (a box that wide has a zero at the wave's frequency); slide the residual wave along the arc with a smoothstep so the root's exit is untouched; put nodes back at their own arcs.

### D1.7 Head variants, the gate results, the cost of air-drying, and flyaways

*2026-11-12. Status: measured on named cases. None of the later variants has replaced the base; hair work was put on hold at about 18:40 the same day.*

"Loose stretch" below means a run of at least 5 consecutive points with fewer than 6 other-strand points within 4 mm, outside the last 15 % of the strand.

| head | change from the base | rest gate of the grown head | style gate | child wave rms by third of length (mm) | loose stretches, styled |
|---|---|---|---|---|---|
| n2c (base) | — | 0.038 pass | pass | 2.78 / 3.07 / 3.51 | 104 |
| n2d | phase field 32 mm, first shift | 0.443 **fail** | not stated in the record (it was rendered) | 2.57 / 2.77 / 3.31 | 326 |
| n2e | + wave-only shift, CD 44, exit angles 40°/50° | 0.424 **fail** | 0.047 pass | 2.78 / 2.91 / 3.40 | 168 |
| n2f | n2e, comb pull halved | (n2e's head) | 0.025 pass | 2.78 / 2.96 / 3.43 | 166 |
| n2g | n2e, water set taken after the hands | (n2e's head) | 0.038 pass | 3.08 / 3.36 / 3.82 | — |
| n2j | curl index 1.20, set after the hands | tips hanging at 0.1–0.19 | **0.992 fail** | 3.88 / 4.44 / 5.15 | 247 |

**A head failing its rest gate.** The n2d and n2e heads failed with energy flat, so it was stuck guides and not too few evaluations. A probe found 27 guides, one tip node each, resting on the front of one shoulder 0.4–12 mm off the skin: a friction stick at a hanging tip, not a strand inside the body. The style stage re-settles and its own gate decides. An earlier 12 mm version failed at 0.897. n2j failed its *style* gate at a handful of nodes (p99 0.021) and has not been rendered; a diagnostic that reports the worst nodes is implemented and has not been run.

**What air-drying costs the wave.** On the base, the untouched rested head waves at 3.42 / 3.64 / 3.95 mm; after the style, 2.78 / 3.07 / 3.51. Cause: wet hair hangs heavier and straighter, and the water set keeps a share (0.4 in this recipe) of that wet shape.

**The water set and its timing.** Over the set span, each joint's rest turning moves toward its current turning: ρ ← ρ + set·(dd − ρ). Re-wetting removes the set's share; the follicle's own curl is untouched by water. The set fades with time and humidity as R(t) = 0.7·exp(−t′/0.5 h) + 0.3·exp(−t′/24 h), t′ = t·(RH/90)^5.2·10^(−2·hold). Water softens and loads the fibre: E × (1 − 0.5·wet). The first recipe took the set at the end of the hand pass, while the hand still held the hair. Taking it in the free fall after the hands are gone (n2g) gave +11–15 % wave with the body changed by only 0.2–1.8 mm. Kept. *Tried and rejected:* re-taking friction anchors at the dry friction (0.13) after drying gave no width gain and let the front slide onto the face (529 points over the face region against 275).

**Body.** The author's review asked for a little more body, later clarified as mostly at the sides and "a derivative of just the texture of the waviness". Exit angles, comb strength and CD all measured flat after the style. On n2j the sides' p95 width rose 2.2–3.0 mm and the crown 1.4 mm with the wave, but n2j is unverified.

**Flyaways: designed against defect.** The generator marks a small fraction of strands as fly-aways that leave their lock partway along. Of the base's 104 loose stretches, 12 are on designed fly-away strands and 92 are not. On n2d: 16 designed, 310 not. On n2e: 14 and 154. On the untouched heads: 787 (base), 1,004 with the first phase shift, 740 with the wave-only shift. The author's review of the n2c / n2d sheet was that the flyaways were "most likely ... by design" and should change only through styling. The counts say otherwise for n2d: its extra loose strands were a defect of the first phase shift, fixed at that cause.

**Crown loop.** The base has a loop of hair standing at the crown. Measured cause: short regrowth hairs (35–73 mm) at the whorl, where the long layer parts away and nothing lies over them; a 70 µm hair only buckles under its own weight past about 91 mm. The generator fix (2026-11-11 21:43, 7bc80fe5): short hairs take the wet set with the long hair. On a test cap: 5 of 5,331 short hairs standing before, 0 after. The base itself has not been regrown with it.

### D1.8 Passes as tools, and hair inside the tool

*Passes 2026-10-31; the Hair dock in Field Studio 2026-11-11 20:06 (91fa6b73). Status: measured on named cases.*

**A stroke as a pass.** The tool's leading edge moves down each strand's arc in graded steps (15 mm, ×1.25, capped at 80 mm). Inside its window it turns each segment's heading toward the stroke about the skin normal only, and drags the hair by bristle friction: a smoothed pull toward a point one step further on, capped at the tool's tension, with no normal component for a brush or comb.

**In the tool.** Pick a kind, paint the area, set colour, thickness, density, length, generate. Then comb, cut, puff, clump, smooth, trim, settle, water (washes the set out, cuts stay), undo per stroke, keep as a state over the natural base. Sectioned cuts: a section brush, same-length with a travelling guide, lift-and-cut at an elevation and angle, point cut.

Measured: 198,156 hairs with 1,593 guides generated in 342 s on the CPU and 195,294 in 169 s with the rods on the card; brush dabs 1.3–1.4 ms on the card; every hair drawn in 46–255 ms a frame; app 4.1 GB working set and 3.8 GB of card memory with the model open. A lift-and-cut of 40,462 hairs took 4.1 s. *Stated plainly at the time:* that first layered cut did not read in the before/after sheet, because the cut ends stayed inside the under-lengths.

### D1.9 The fibre layer: rendering, scattering, shadows

*Fibre layer 2026-11-10 (b0535200); multiple scattering 11-10; exact skin shadows 11-11; lean scalp shadow default 11-12 01:19 (1964801a). Status: accepted in the workflow for the face-hair shadow fix (author's review: approved) and the scalp shadow; the rest measured on named cases.*

**Data.** Per strand 24 points, radius and pigment per point, ellipticity, twist rate, roughness, root normal. Each strand is a centripetal Catmull-Rom curve. A uniform span grid (0.5 mm cells for short hair, 2 mm otherwise) lists each span in the cells its curve passes.

**Primary rays.** A 3-D DDA walk of the span grid, a conservative sphere pre-test per span, then the exact hit (curve chords plus 3 Newton steps, oval section turning with twist, sub-pixel coverage from the pixel cone's overlap). The 8 nearest hits are shaded and composited front to back, then the next 8, up to 48 fibres or until transmittance falls to 0.02.

**Scattering.** The Chiang, Bitterli, Tappan and Burley 2016 lobes (R, TT, TRT and a residual) on the Marschner 2003 fibre, with a precomputed far-field table (β_n 0.55, η 1.55), cuticle tilt 3–4°, β_m 0.10. Absorption is per unit *local* radius, so a tapered tip is lighter. Near field when a fibre is at least a pixel wide. Wear: β_m × (1 + w·min(s/0.4 m, 1.5)), 0.10 at the root toward 0.22 at 40 cm. A sized key widens β_m through Chiang's variance mapping.

**Light per hit.** (1) Body shadow: a soft SDF march toward the key. (2) Self-shadow: transmittance through every fibre crossed toward the key, each fibre counted once. (3) Sky: the body's openness times the hair's transmittance in four cosine-weighted cones, with forward attenuation a_f = TT + 0.293·R + 0.15·TRT. (4) The room per lobe. (5) Multiple scattering from the mass beneath: per fibre the back share b and forward share f, B = b/(1 − f²), added in proportion to how dense the hair is toward the skin. With it, hair-to-skin lightness went from 0.70 to 0.89 against a reference photo's 0.81.

**Body lattice repair.** For the hair's light the body lattice is made a true distance everywhere: inside ±9 mm the field's own value; beyond, the Euclidean distance to a reliable shell (5–9 mm) plus that shell's value, per side, blended over the last 1 mm.

**Hair shadowing skin: the fault and the method.** One cone of the light's size walked through the span grid visits only the cells on its centre line. Under a 12° softbox the footprint is 4–12 mm at 2–6 cm, so the cone missed most fibres and over-weighted the rest. Measured on a three-quarter frame: a lash shadow streak 1.8 mm wide at 0.40 light where a fibre-resolved reference gives 10.9 mm at 0.72. About the same total shadow, squeezed into a band 4–6 times narrower and twice as deep. Two compositing faults sat beside it: each layer's skin shadow was multiplied in after earlier layers were drawn, and the face hair was drawn over the scalp hair.

The method now: per skin pixel, exact stratified rays to a cap of the key's angular size and to six sky caps of 0.12 rad, with every fibre crossed counted; each layer's shadow is taken on bare skin first; layers are composited by first-hit depth. For scalp hair a lean version: 8 key + 6 sky pilot rays at every pixel, then up to 36 + 24 only where skin is seen through hair and the pilot rays disagree. Measured against a 144-ray reference: mean error 0.0117 (the cone: 0.0357), total shadow within 0.12 %, isolated black specks 18 against the cone's 1,066 (reference 14). *Tried and set aside:* sub-cones (5 % too dark) and the hair as a measured absorbing medium (27 % too dark: scalp hair lies in locks, and a density loses the gaps between them).

Hair in the eyes' mirror reflections: 6×6 exact sub-rays along each eye pixel's reflected ray; error against a 16×16 reference 0.0001. Two runs of the same frame first differed by noise because sub-ray seeds followed an atomic counter's order; sorting records by pixel made two runs bitwise equal. Status of that determinism: reproduced under a frozen configuration.

**Costs.** A three-quarter head frame: hair 35.9 s, of which the scalp's exact shadow 11.4 s; a hairline close-up 20.7 s; the face-hair shadow pass 0.5–1.7 s; eye reflections +0.8–1.2 s. Growing a shoulder-length head: 14 min on the CPU or about 6.5 min on the card; styling about 8–10 min on the card, 34–40 min on four CPU threads.

### D1.10 The settled hair frame in the tool's window, and the multi-light layer

*2026-11-12. Status: implemented in a working copy; not landed. Partial measurements only.*

While the view moves, the window draws hair with a fast splat. Measured against the review renderer pixel for pixel, the splat's light was 2.3–4.4 times too bright and grey. The design: when the view rests and the window's frame has converged, draw the review renderer's fibre layer over the window's own frame, depth and lights, and keep it until any input arrives. One job at a time on a worker thread, cancelled at its next launch by any mouse press, wheel or key. Two causes were fixed first: the unrepaired lattice put the hair at 0.53–0.60 of the review's radiance, and unshadowed skin under translucent blond fibres read as "ghostly". With both fixed: 0.93–1.11.

Measured in the real window: lands 9.9–11.7 s after the view stops; first layer build 27.7 s per hair version; card memory 4.25 → 4.49 GB. **It failed its stroke check:** a body stroke during a job stalled the window (0 fps, worst gap 1,504 ms, against 32.3 fps with the feature off), because the layer's bands ran on the default stream. A fix is built (its own non-blocking stream, 0.05 s bands, cancel on input) and has not been re-tested.

The multi-light fibre layer gives each extra light a direction, colour, angular size and shadow flag and sums its direct terms onto the key's. It compiles. Its three tests (bit-identity with one key, linearity, a headlight) have not been run.

---

## D2. Light

### D2.1 Tracing the field

*In the tool before this lane; measured 2026-11-11. Status: measured on named cases.*

Rays are sphere-traced through the sampled field (Hart 1996), then the hit is refined by bisection and a secant step inside the bracket. The normal is the analytic gradient of one tricubic Catmull-Rom reconstruction over the 4³ nodes around the point. The plain sculpt light is a soft-shadow march toward the sun (the min of k·f/t along the ray), 2 sky rays per pass to 0.15 m, and spherical-harmonic sky irradiance, accumulated over 16 passes. The sky is added when the frame is resolved, not in the accumulation; two measurement mistakes in this lane came from forgetting that.

Costs at 1920×1080 on the model: primary trace 1.29 ms for a full-body view, 5.00 ms for a face close-up (7.6–10.3 ns per hit pixel); a sun ray 3.0–5.3 ns; a sky ray 2.1–2.5 ns.

The tracer skips empty space with a 32 mm tile grid. The project's own reading flagged that tile skip against two patents on per-tile empty/occupied entries (Movidius US11680803, US11222459). It is being redesigned (skip by distance, not by an empty tag) and that code is held out of any published copy until it lands.

### D2.2 The rig, the panel, the presets

*Rig schema 2026-11-11 17:41 (88e5c965); presets 19:28; Light panel 20:10; in-view A|B split 20:24. Status: implemented, with CPU gates passing; the split reproduced under a frozen configuration (each half bit for bit its rig's own frame).*

A rig is a small versioned file: lights (sun, area rectangle or disk, sphere, spot, point), environment (HDRI, rotation, sky gain), exposure. Both the tool and the review renderer read it. The panel offers presets placed around the view's target, key controls, environment, save and load, A/B compare and a split view. Hair reads its lights from the same rig as irradiance before exposure.

**The Rembrandt-style preset** (2026-11-12 08:38, 03f20e3a). Key 65° round and 30° up, 4° across as seen from the subject (a 9×14 cm softbox at 2 m), fill 0.05, a dark studio with room light 0.05. It was set by measuring five traits of the pattern on the model's face (lit triangle against shadow cheek, lit against shadow side, triangle against lit side, triangle against nose shadow, and the nose shadow's edge width) against a published portrait-lighting reference that was studied and is not redistributed. Of 96 settings the chosen one was nearest (distance 0.81; the previous preset 1.72); nose shadow edge 4.0 mm against the reference's 3.9. A softer variant (10° key, fill 0.1) is kept under its own name. A 17° candidate cost the same per pass but gave an 11.6 mm edge and was not adopted.

*What light could not do:* across 288 settings the "closure" of the triangle (the region beside the nostril as dark as the reference's) never moved. On the model that region is as bright as the cheek shadow in every setting. It is the model's form there, not light, and it was not forced. Status of the preset: accepted in the workflow, conditionally (the author's review: keep it if it clears the legal checks and the other requirements); grain and cheek crops at the 4° key were not measured on the card.

### D2.3 The stage-1 study: what was measured before building

*2026-11-11 16:55–18:05. Status: measured on named cases (probe code, Vulkan through Slang; not the product path).*

- **Bounce by brick caches.** A radiance cache holds each surface brick's direct key light (0.18–0.23 ms for 38k bricks). An irradiance cache gathers the bounce at each brick from cosine rays that read the radiance cache (64 rays: 5.5–10.8 ms). Read per pixel by blending 8 bricks: 0.067 ms, bias −0.2 %, error below the noise of a second 256-ray reference. The earlier option (16 rays, hit lit unshadowed) was 28.7 % too bright.
- **What failed:** a cone-gathered bounce (−63 % bias); cone sky occlusion (−29 %); five distance lookups along the normal (correlation 0.54); cache points storing light by direction in 9 spherical-harmonic terms (ringing, +4.7 % bias).
- **Colour.** Bounces 2 and 3 add 7.6–9.2 % of the bounce light. In skin folds the bounced light's red/green ratio is 1.193; a grey cache is 15 % low in red there. Skin-to-skin bounce has to be carried per channel.
- **Patchiness at close zoom.** One point per 8 mm brick gave a structured error of 34 % on a face close-up against path tracing's 1.5 % floor. A 2 mm lattice: 4.5 % on the face, 2.0 % on the chest. A per-pixel gather: at the floor.
- **Sky as per-brick spherical-harmonic transfer.** 9 coefficients a brick from 256 rays to 2 m: bias −1.1 % under a smooth sky, where the short-ray method is +18 % in the most occluded tenth. Under a bright 20° soft box the 9-term transfer fails (+87 % and worse). Conclusion: smooth sky through transfer, every bright light sampled explicitly. This is a probe result. The records do not establish that per-brick transfer is in the tool's live light.
- **Caustics.** Light tracing into 2 mm floor cells through a three-mirror chain: 4.2 M photons in 5.9 ms, every caustic cell found in one batch, bias 0.1 %. Camera-side path tracing at equal time found 0.15 % of the cells. Probe only.
- **RT cores.** One box per surface brick in a hardware tree with the same in-brick hit: 1.2–1.8× faster than the software two-level DDA on primary and short sky rays, equal on sun rays. Tight boxes: up to 2.6–2.8× on sky rays.

### D2.4 The look-dev light: lattice while moving, gather when still

*Module 2026-11-11 19:33 (129d2900); lattice 19:39; in the card view 20:18 (928f3d1e); 64-pass target 11-12 04:39. Status: measured on named cases.*

**Two stores.** 8 mm brick caches keyed by (volume, brick), and a 2 mm surface lattice inside every surface brick: a 64-bit sub-cell mask and an offset per brick, valid points only, snapped to the finest surface level. On the model: 692,218 points, built in 15 ms, gathered at 128 rays in 0.50 s, all caches 38 MB on the card. The lattice covers every surface brick regardless of the camera, so orbiting needs no refresh.

**Two modes, chosen per frame by interaction.** While the view moves, pixels read the lattice (0.07–0.13 ms). When the view rests, each pass runs a per-pixel cosine gather. The key is sampled as an area light, with the plain sun term zeroed.

**The settle.** When the view rests, passes accumulate toward 64 per pixel (16 in sculpt mode). There is no denoiser: the image converges by samples. A still pixel's mix (1–3 lattice passes, then gather) measured level with path tracing at equal rays in the folds, bias −0.24 to +0.06 %. While orbiting, the lattice's error in folds was 1.1–1.9 %, fixed to the skin, with no flicker.

**Three wiring faults found by the gates and fixed:** near hits lit by the sun only, missing a sampled studio key; the key counted twice in the app (the tests had patched the light by hand, hiding it); fixed-size slices that ran to 103 ms. A fourth, found in the app only: a separately compiled module had its own unset copies of the tracer's device globals, an illegal address the moment a masked volume was sampled. A blocking-launch run named the launch.

### D2.5 The cache redesign of 2026-11-12

**What the first version did** (11-11 19:33 to 11-12 10:22). At each gather hit, if the hit was at least 10 cm away the gather read the hit brick's *outgoing* cache (albedo/π × (direct + bounce)); nearer hits were lit exactly. This came from the study, where it met the patch-error gate.

**The flag.** An outside review of our patent checks pointed out that NVIDIA US11790594 has an independent claim 16
which our earlier check had missed. As we read it, the claim concerns a ray cast from one location that meets a
second location associated with an outgoing irradiance cache, and rendering that uses the cached outgoing value
based at least on a distance between the two (claim 18: beyond a threshold). The first look-dev gather, with its
near/far switch, was flagged in our review and redesigned the same morning. Nothing carrying it was ever released.

**The redesign, landed 11-12 10:22 (506bd914).** One rule at every gather hit, never chosen by distance: the rig's lights are computed exactly at the hit (the key by a shadow ray, each area, spot or point light by one sample). The gathers read no outgoing cache. Gather rays run until they leave the scene. This is the arrangement of Ward, Rubinstein and Clear 1988. Direction of the design-around: no distance, threshold, ray-cone width or footprint decides anything at a hit. Recorded rule for future work: never add a choice between a cached value and an exact one by a hit's distance, cone width or footprint.

Status: measured on named cases. One bounce matches an all-exact reference with bias ≤ 0.12 % in the folds. A still pass at 960×540: sun, face 9.66 → 10.46 ms, chest 16.48 → 19.32; Rembrandt, face 13.48 → 13.30, chest 21.27 → 24.30. Lattice build 511 → 593 ms (sun), 628 → 716 (Rembrandt). Card memory identical (2,051.8 MB built and drawn, either way).

**What it cost, stated:** the product is now one exact bounce. The second bounce is missing: 0.5–2 % of the light in folds, about 35 % in the sun-shadowed pockets of a dark studio. A test showed the 8 mm cache's stored bounce was 1.1–2.4 times the true second bounce there (one point per brick leaks light across a fold), so it is not used. Taking the second bounce from the 2 mm lattice by the same one rule is proposed.

**Excluded prototypes.** Two study probes, `engine/light/fs_cache.cu` and `engine/light/fs_cache2.cu`, still contain the old distance switch; the second also built lattice points only for bricks visible to the camera. They were never wired into the tool. They are excluded from any published code as they stand.

**Refresh without ranking.** A light change re-gathers every valid point. An edit re-gathers only its dependency set: lattice points that are new, or lie within a fixed reach (0.15 m) of the edit's box. That is one membership test, the same for every point, walked in fixed memory order. Slice *size* is predicted from the measured cost of the fixed 4,096-element blocks it covers; it decides how many come next, never which. Claims 1 and 10 of the same patent require ranking locations against one another; no location is ever compared with another here.

### D2.6 Keeping the display alive: slices and bands

*Review renderer bands 2026-11-11 19:04 (ec99a732); hair bands 21:29 (8c3bc466); look-dev slices 11-11 19:55 to 11-12 06:03. Status: reproduced under a frozen configuration for the review renderer's bands (tiled equals untiled bitwise: image, depth, call counts); the rest measured on named cases.*

The display driver resets the card if one launch runs for about 2 s. Two resets happened on 11-11 from launches of about 10 s.

- **Review renderer:** frame passes run in row bands, alternating over two streams so one band's tail overlaps the next. Frame time unchanged (326.9 against 326.7 ms a pass at 960×540). The heaviest close-up's longest launch: 304 ms, against a 1 s cap. If a band errors, it falls back to whole-frame launches.
- **Hair kernels:** timed row bands aimed at 0.6 s (growth ≤ 1.5× per band, never under 32 rows), one round of 8 hits per launch with per-pixel state carried between launches. A whole-frame scalp launch had run 15–19 s.
- **Look-dev cache builds:** done on idle frames in slices aimed at 8 ms, into a separate set that is committed whole. A launch lasts at least one thread's serial loop, so the budget is met by chunking rays (8 a launch), never by shrinking the element count below what fills the card. Returns at once on interactive frames.

Measured after a 2 mm cheek stroke: the dependency rebuild settles in 0.86 s (a full rebuild 2.6–3.3 s), bitwise equal to a full rebuild beyond 15 cm, 0.004 % of rendered pixels differing. Edit path's longest call 12.2 ms.

**Known defects, open.** Some slices exceed the 8 ms budget in the app (begin 9–16 ms, list ~20 ms). After a stroke the view converges twice. Release to idle takes about 13 s at 1600×900 and 45 s at 2560×1440. One look-dev still pass costs 92–103 ms in the app against 15 ms in a test scene; not broken down yet. Running still passes in row bands cut the worst press-to-frame wait (219 → 152 ms) but lengthened release-to-idle (13 → 18 s); starting the edit job at release made presses worse (492 ms). None of these three patches landed. Look-dev has no hair in its light yet.

### D2.7 Skin in the light

The look-dev light treats skin as a diffuse albedo per channel. Subsurface terms (a per-channel walk table, terminator wrap, back-lit thickness) exist in the review renderer's own light code and are named in the study only as components to share later. These records do not establish them in the look-dev path. What is measured here is the colour of skin-to-skin bounce (D2.3).

### D2.8 The older-hardware aim

*Status: proposed. Estimates only; no older card was measured.*

The aim is ray-traced-quality light on cards without RT cores, such as a GTX 980 (4 GB, no ray query). The route: one Slang source compiled to Vulkan compute, the software two-level DDA for traversal, caches for bounce and sky, hair as a density field at distance. Estimate from the measured terms: a lit body at 1080p costs about 4.8 ms on the RTX 5050's software path; scaled by bandwidth and floating-point throughput, about 10–17 ms on a 980, without skin shading, eyes or hair. Running the same Slang code on a CUDA backend failed (the device could not be created), so Vulkan against CUDA on identical code is untested.

---

## D3. Published ingredients and licences

| ingredient | source | terms as recorded |
|---|---|---|
| Fibre scattering lobes, roughness mapping | Marschner et al. 2003; d'Eon et al. 2011; Chiang, Bitterli, Tappan and Burley 2016 (as in pbrt-v3) | published papers |
| Forward attenuation in a fibre mass | Zinke et al. 2008 | published paper |
| Rods | Bergou et al. 2008; Bertails et al. 2006; Miller, Lazarus, Audoly and Reis 2014 | published papers |
| Fibre mass pressure; wet cohesion | van Wyk 1946; Goldstein, Warren and Ball 2012; Bico et al. 2004 | published papers. Goldstein et al. note van Wyk's law does not fit their data: an open item |
| Hair classification and biology | de la Mettrie, Loussouarn et al. 2007; Thibaut et al. 2005; Franbourg et al. 2003 | published papers |
| Hair with water and styling | Ward, Galoppo and Lin 2004/2007; Fei et al. 2017 | published papers |
| Random Fourier features | Rahimi and Recht 2007 | published paper |
| Sphere tracing | Hart 1996 | published paper |
| Direct at every hit, indirect from a cache | Ward, Rubinstein and Clear 1988; Christensen 1999 | published papers |
| Precomputed transfer | Sloan, Kautz and Snyder 2002 | published paper |
| SDF grid tracing, studied | Hansson-Söderlund, Evans and Akenine-Möller, JCGT 2022 | CC BY-ND 3.0; studied, its exact hit not used |
| Capsule intersection (probe) | I. Quilez's articles | MIT |
| Slang / slangpy 0.43.1 (probes only) | PyPI | Apache-2.0 (Slang with LLVM exception); its bundled OptiX headers are proprietary and unused |
| Vulkan headers | Khronos | Apache-2.0 or MIT |
| Lighting and lash-shadow study photographs | public image archives | CC BY-SA and public domain; studied only, not shipped, not redistributed |

Not used: OptiX (proprietary terms, one vendor's hardware).

## D4. Patent readings carried over as recorded

**These are our own readings of the claims, made without counsel. They are not legal advice and no clearance is claimed.** A search hit is a screening flag; a claim reads only if every element of at least one claim is met. "None found" is not proof of absence. Expiry dates are estimates. Pending applications can change and need re-reading at grant.

| record | claim element of concern | what is absent in our design |
|---|---|---|
| NVIDIA US11790594, claims 16–18 | outgoing irradiance cache at a ray's hit, used based at least on a distance (18: beyond a threshold) | the first look-dev gather was flagged in our review and redesigned. Since 11-12: no outgoing cache read at hits and no distance, cone-width or footprint test. The two probe files that still hold the switch are excluded |
| same, claims 1 and 10 | ranking cache locations by characteristics (camera distance, visibility, frames since update) and casting rays from the higher-ranked | nothing is ranked; refresh is by a fixed membership test in fixed index order; the lattice is camera-independent |
| its continuation US2024/0029338 (pending) | a denoising filter mixing cache values | no denoiser; re-read at grant |
| NVIDIA US11823319 | in-voxel hit from a cubic whose coefficients come from a factored formulation | hits are sphere tracing plus bisection and secant on the sampled field; no cubic coefficients are formed. The JCGT 2022 exact hit is not used. One project record marks this reading as not verified against the full granted claims, so it is treated as open (Part 10) |
| NVIDIA US12254556 (independent claims 1, 5, 15, 22) | intermediate normals per adjacent voxel, then interpolated to the hit | the normal is the derivative of one tricubic reconstruction; no per-voxel normals exist and none are blended |
| NVIDIA US11887245 | shadow rays via the cubic's turning points | that algorithm is not used |
| NVIDIA US12175588 | hit points stored in voxel-local frames | hits are kept in world or object space |
| Pixar US9905045 | a scattering component made by *fitting* a set of basis distributions of several types, sampled from their closed forms | the Chiang 2016 lobes are derived, with one logistic per lobe whose scale comes from a closed formula; tables are numeric. Guardrail: never fit mixtures of logistic, skew-logistic, monomial or Cauchy lobes to a measured response and sample them in closed form; tabulate and sample by numeric CDF |
| DreamWorks US9019298 | importance sampling of hair lobes based on a Cauchy distribution | no Cauchy distribution anywhere in the hair code. Guardrail: keep it so |
| Disney US8674988 | per-lobe artist controls for hair shading | recorded as "expired – fee related" (lapsed 2026-03-18) as of 2026-11-11. That is a dated status, not a permanent one: such a lapse can be reinstated, and it is to be re-checked before release |
| Microsoft US7948490 (active to 2028) | transfer coefficients computed by rasterised shadow depth buffers | transfer in the probe was gathered by rays in compute; no shadow maps |
| NVIDIA US12315070 | caustics by a feedback-driven photon task buffer and splatted footprints | the probe is plain light tracing into surface cells |
| NVIDIA application US2026/0148476 (pending) | hardware linear swept spheres | measured in a probe only; hair does not depend on it |
| AMD application US2024/0331266 (pending) | capsule-chain curve intersector | the probe's box-plus-capsule test: claims to be read before it ships. Not checked |

Not found on the records searched: patents on sphere tracing, photon mapping, the Chiang 2016 or Marschner 2003 models, tight axis-aligned boxes around SDF bricks. The search was cut short by blocked queries and is incomplete.

## D5. Open items

- The curl-index head failed its style gate at a handful of nodes; no result from it is used until the gate is diagnosed.
- No hair dynamics: statics only.
- The through-fibre composite is a screen-space approximation of the TT lobe's cone.
- Friction anchors are fixed in space between lags; hair sliding on hair does not drag its neighbour.
- Section ellipticity is stored but is not yet a bending anisotropy; per-type curl indices are design values.
- Face hair does not shadow itself; brows and lashes do not follow a sculpt without the carry step.
- The settled hair frame and the multi-light fibre layer are not landed or verified.
- The second bounce is missing from the look-dev light.
- The tracer's tile skip is being redesigned.
- Nothing has been measured on a card without RT cores.

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

