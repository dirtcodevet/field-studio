# Field Studio: hair, grown from causes and styled as states (technical disclosure)

Prepared 2026-11-12 (Pacific time) for the first public research release of Field Studio, a mesh-free sculpting,
simulation and rendering tool in which every form is an implicit field (a signed distance field on a sparse,
multi-level lattice), and of its hair engine (`engine/hair/`). Hair here is curves only: a strand is a centreline
polyline with a material frame per edge; no triangles, ribbons, cards or tubes are made anywhere. This file records
the mechanisms as built, with their parameters and the order of operations, so that the record is public prior art.
Each mechanism carries one status: **BUILT AND MEASURED** (the number is cited), **BUILT, NOT YET VERIFIED**, or
**PROPOSED**. Measurements were made on one test head unless stated. Published methods the work builds on are cited
as prior work and not claimed (see the end).

Units: millimetres (mm), metres (m), radians (rad) unless stated; "guide" = a simulated rod standing for the hairs
rooted around it; "child" = a rendered hair interpolated from guides.

---

## 1. Grow the head once from causes; styles are passes and states applied to the grown head
**First recorded:** 2026-10-31 (design), built 2026-10-31 to 2026-11-12. **Status:** BUILT; the water-set timing and
the anchor variant below are MEASURED; the photo-to-recipe reader is a BUILT prototype, NOT YET VERIFIED in a loop.

**The idea.** Hair is grown once, as in life, from its causes: follicle position and exit angle, the fibre's genetic
type (curve diameter, curl index, section), its stiffness and weight, sebum creeping from the root, water, wear and a
cut by region. The result is a *rested natural head*. A style is never part of growth: it is a sequence of physical
passes (brush strokes, finger/hand passes, a water set while air-drying, product) applied to that saved head, each of
which holds nothing once it ends. Styles can stack because every pass acts on any saved state.

**Three stages, three owners.**
| stage | owns | output |
|---|---|---|
| Hair | follicles, fibre, lengths, cut by region, sebum, natural wave; growth with no acts; release; the gated final settle | the grown-head cache |
| Style | product, the passes, the water set, release, final settle, children, form | the styled state |
| Colour | melanin, colour; re-colour without re-growing | colour arrays |

**Layer files** (`engine/hair/layers.py`). A look is composed from five JSON files — hair, style, colour,
environment, sim — and every settings leaf has exactly one owner layer (longest-prefix rule). A layer file holding a
key another layer owns is refused (validation raises, not warns), so a style can never re-tune the fibre or the
follicles. A file may `extends` another file of the same layer; `compose()` deep-merges defaults then layers.

**A stroke as a pass** (`passes.py stroke_pass`, `sweep_pass`, `hand_pass`). The tool's leading edge moves down each
strand's material arc in graded steps: 15 mm, ×1.25 per step, capped at 80 mm (≈10–13 steps on 450–650 mm hair), at
every depth. Inside its window it (a) turns each segment's heading toward the stroke about the skin normal only (it
cannot change the hair's height) and (b) drags the hair along the stroke by bristle friction: a smoothed pull toward a
point one step further along, force capped at the tool's tension, with no normal component for a brush, comb or
fingers (a lifting hand may only push outward), zero once the hair has caught up, living one step. Between steps the
hair relaxes with the friction anchors carried by `lag_stick` (§4). After the last step the tool is gone.

**The water set and its timing** (`actions.dry_set`, `dry.when`). Over the span s ≤ `over_mm` (default 40 mm; a
whole-head air-dry uses up to 300 mm) each joint's plastic rest turning ρ is moved toward the current turning
dd_j = d_j − d_{j−1} (d_{−1} = follicle direction): ρ ← ρ + set·(dd − ρ), with `set` = 0.1–0.15 air-dried (default
0.15), ~0.4 brushed while drying, ~0.7 blow-dried. An optional root lift adds `lift_deg` of turning toward the skin
normal over the same span. Re-wetting removes the set's share: ρ_set × (1 − wet) (`settle.rewet`); the follicle's own
curl is untouched by water. The set fades over time and humidity (`actions.retention`):
R(t) = 0.7·exp(−t′/0.5 h) + 0.3·exp(−t′/24 h), t′ = t·(RH/90)^5.2·10^(−2·hold). Its constants reproduce the cited
checks: unfixed at 90 % RH, 0.45 after 0.75 h; fixative hold 0.5, 0.43 after 8 h; hold 1, 0.90 after 8 h; unfixed at
45 % RH, 0.29 after 72 h. Water also softens and loads the fibre: E × (1 − 0.5·wet), weight × (1 + uptake·wet).
`dry.when` chooses when the set is taken: `with_hands` (at the end of the hand pass, while the hand still holds the
section) or `after_hands` (in the free fall after the hands are gone, as in a real air-dry).
- MEASURED (full head): `after_hands` gave +11–15 % wave in the children (rms by third of length 3.08 / 3.36 / 3.82 mm
  against 2.78 / 2.91 / 3.40) with the body (hair thickness off the scalp) changed by only +0.2–1.8 mm; settle gate
  passed at 0.038. Kept.
- MEASURED, REJECTED: re-taking the friction anchors afresh at the dry hair-hair friction (0.13 instead of the wet
  0.25) at the end of drying gave no width gain (within ~1 mm) and let the front slide onto the face (points over the
  face region 529 against 275).

**Order of the style stage** (`generate`, flow order "after"): load the grown head (fingerprints checked, §4) → product →
brush strokes → hand passes → water set (per `dry.when`) → release (every act off) → release solves with anchors kept by
`lag_stick` → tousle → the final gated settle → children (§3) → form → baby hairs → colour → a 0.2 mm push-out.

**Recipe from photos** (`recipe.py`, prototype 2026-10-30). A reader measures a draft style from reference photos in
the head's own frame (mm, deg): pose from face landmarks; scale from the visible iris diameter (11.7 ± 0.42 mm) and
the interpupillary distance (61.7 ± 3.6 mm, ÷ cos yaw), inverse-variance weighted; hair/skin/background masks by least
Gaussian cost over (L*, a*, b*, log strand-line density); each key written with its evidence, confidence and source
(`read`, `preset`, `default`, `question`), and keys no photo decides listed as questions. The draft `extends` the
head's base layer files and holds only what a look decides.

## 2. Follicle, fibre type and curl index
**Status:** BUILT AND MEASURED (single fibre); the full-head curl-index result is BUILT, NOT YET VERIFIED (gate failed).

**Follicle map** (`follicle_map.py`). Samples on the body's field surface about 2 mm apart, each with a weight per hair
kind, a density (follicles/cm²) and a flow direction in its tangent plane. Brushes: paint, erase, comb (flow projected
into each sample's tangent plane), density. Follicles are drawn per sample as a Poisson count of
density × weight × sample area, placed uniformly in the sample's tangent disc and re-seated on the field, so no grid
pattern shows in the roots.

**Exit angles by region** (`follicle.py`, angle between skin plane and hair): mid-scalp 35°; whorl 70°, fading to the
mid value as exp(−d_W / 18 mm); occiput 45° (|azimuth| ≥ 125°, polar 80–110°, 10° fade); hairline edge front 15°,
temple 10°, nape 25°, held at the hairline and blended to the inner value over 25 mm. Jitter is logit-normal on
[0°, 90°] (sd 4° at the mid value, median kept exactly) so the spread narrows toward the skin and the normal; azimuth
jitter sd 8°. Whorl at polar 32°, azimuth 180°, spiral turn 60°. The settle clamps each guide's first segment to its
follicle direction.

**The fibre's type is a helix** (`fibre.py HAIR_TYPES`, after the de la Mettrie / Loussouarn curve-diameter and
curl-index classification). In the frame parallel-transported along the strand (t, n, b) the rest curvature vector is
κ0·(cos θ, flat·sin θ), θ = τ0·(s − start) + phase, ramped in by a smoothstep over `ramp_mm`. κ0 = 2 / CD. τ0 comes
from the curl index i (stretched length / end-to-end): for a round helix τ0 = κ0 / √(i² − 1) exactly; for flat < 1 by
bisection on the measured index of the generated path. Hence, at fixed i, the shape is self-similar in CD:
free amplitude ≈ (i² − 1)·CD / 2 and wavelength ≈ π·√(i² − 1)·CD. A tighter CD alone gives smaller, shorter waves;
**the waviness lever is the curl index.** Types (design values, overridable): straight (no rest curl, a/b 1.25);
wavy (CD 48 mm, i 1.10, flat 0.25, a/b 1.3, ramp 25 mm, lock of ~30 hairs, bond 0.1); curly (CD 10 mm, i 2.0, flat 1,
a/b 1.75, lock ~50, bond 0.35). The section's ellipticity is stored but not yet used as bending anisotropy.

**Single hanging fibre, measured** (`state.single_strand` + `rod.solve_hanging`, 250 mm, dry; rms offset from the
straight hang): CD 48 / i 1.10: 4.75 mm (hanging wavelength ~89 mm against ≈69 mm free from the formula); CD 44 / i 1.10: 4.59;
CD 40 / i 1.10: 4.61; CD 40 / i 1.15: 5.67 (+19 %); CD 35 / i 1.20: 6.32 (+33 %; wet 4.66 against 3.66 for the base
fibre wet, +27 %). Gravity stretches the hanging wave (on a full head the strands' own hanging wavelength p50 85 mm
against 70 mm rest).

**Full head, curl index 1.20, water set after the hands:** children's wave +40–47 % against the base style (rms by
third 3.88 / 4.44 / 5.15 mm), sides' p95 width +2.2–3.0 mm, crown +1.4 mm. That style **failed its settle gate** (max
0.992 against 0.05, p99 0.021; a handful of nodes; energy flat), so it is not a verified result.

## 3. The wave phase field: neighbouring strands wave in step, in clumps
**First recorded:** 2026-10-29 (a phase field in the older form path); engine version 2026-11-12, children blend about 10:00
PT, wave-only shift about 12:10 PT. **Status:** BUILT AND MEASURED.

**The cause.** An air-dried natural wave is in step only inside a clump that formed while wet: neighbouring locks are
loosely together and patches of the head are apart. One random phase per strand reads as crimp; one phase per 35 mm
drying section reads as even horizontal bands.

**The field** (`form.phase_field`, key `locks.phase_field_mm`, e.g. 32 mm; 0 = off and bit-identical to before).
Random Fourier features over the 3-D root positions X (m): 24 wave vectors k_m ~ N(0, 1)/(scale_mm/1000), offsets
b_m ~ U(0, 2π), seeded (seed + 53); f(X) = Σ_m cos(X·k_m + b_m)/√12; phase = π·f(X) + N(0, jitter), jitter 0.15 rad.
Measured coherence: 0.83 inside a lock, 0.41 at 8–15 mm, ~0 beyond; the coherence length is ~0.37 × the scale. The
drying sections still set each wave train's wavelength and plane.

**Children from guides that are no longer in step** (`generate._finish`). Children blend the 4 nearest guides on their
own part side and drying section (own follicle direction blended in over the first 15 mm; hairs under 80 mm are single
rod sweeps). With guides at different phases a plain blend averages the wave flat (only 70–82 % of the guides' wave
kept). So, for the blend only: (1) each section's mean phase φ̄ = arg Σ exp(iφ_guide); (2) each guide shifted by
wrap(φ̄ − φ_guide) to the section mean; (3) blend; (4) each child shifted from φ̄ to its own field phase. A child's own
jitter is read against the smooth field at its root (the field with zero jitter, same draws), not against its old
section: reading it against the section put a border's random step into 51 % of children (|shift| p50 0.61 rad), the
fix p50 0.14 rad, 0.4 % > 0.6 rad. Measured: children kept 80–83 % of the guides' wave before, 90–93 % after.

**Wave-only phase shift** (`form.wave_shift`). The earlier shift (`wave_follow`) moved each segment's direction along
the arc by Δs = Δφ / (2π/λ), which also moved the strand's large bends and, at shifts up to π, looped crown hair (loose
stretches 1004 against 787 without the field). `wave_shift` moves only the wave: (1) resample the strand evenly by arc
(2K samples); (2) measure the strand's own hanging wavelength λ_e from the zero crossings of its wave about a 2λ arc
mean, along the wave's principal direction (SVD), over 20–90 % of the length, clamped to [0.8λ, 1.8λ] (rest λ if fewer
than 3 crossings); (3) mean line M = one-λ_e box mean along the arc (a box one wavelength wide has a zero at the wave's
frequency); wave W = R − M; (4) R′(u) = M(u) + W(q(u)), q = u + Δs·smoothstep(s0, s0 + λ_e/2, u), so the root's exit is
untouched; (5) nodes placed back at their own arcs along the new curve (length and spacing kept, root fixed).
Measured (untouched head): loose stretches 740, wave kept 92–96 % (without the field 95–97 %), 7 points inside the body
(33 without the field).

## 4. The settle: discrete elastic rods, friction anchors, the gate, the cached head
**Status:** BUILT AND MEASURED (gate pass/fail recorded on every run; cache refusal observed).

**Single-rod core** (`rod.py`, `state.py`). A discrete Kirchhoff rod in reduced coordinates (the family of
super-helices and Miller et al. 2014): unknowns are the Darboux vector Ω_e = (k1, k2, twist) per joint in the material
frame; F_e = F_{e−1}·exp(ds[Ω_e]), x_{e+1} = x_e + ds·d3(F_e), so inextensibility and frame orthonormality are exact.
E = Σ ds/2 (Ω − Ω0)ᵀ diag(B1, B2, C)(Ω − Ω0) + Σ w_e ds (m_e·up), exact gradient through the left Jacobian of SO(3);
statics by L-BFGS; stability by the Hessian's smallest eigenvalue. `solve_hanging` switches the rest curvature on in
steps (25, 50, 75, 100 %) with a small out-of-plane seed (a homotopy, not a fit). Checked against Miller et al. 2014:
within 1.1 mm of their elastica; planar-to-3-D threshold w·c = 0.2391, matching theory to 3e-6.

**Head settle** (`settle.py`). Per guide, one free 3-vector u_j per segment, d_j = u_j/|u_j|, nodes
x_{j+1} = x_j + l_j d_j (inextensible). Segments graded from 0.5 mm at the root ×1.2 to 14 mm; samples ≤ 5 mm apart.
Energy:
- bending Σ B_j/(2 l̄_j)·|d_j − d_{j−1} − ρ_j|² (d_{−1} = follicle direction; l̄ at the root joint = l_0/2; ρ = rest
  turning = natural helix + set);
- gravity Σ W_j (z(mid_j) − z_root);
- body contact Σ ½ k_c w_y (h_c − SDF(y))₊², h_c 0.5 mm, k_c 3e5 m⁻¹ × sample weight, on a quadratic B-spline of the
  body's 3 mm distance lattice;
- hair-hair pressure from a splatted fibre fraction φ on a 3 mm lattice: P = k·E·φ³ (van Wyk; k 1e-5 at κ 0.039/mm,
  scaled by (κ/κ_ref)²; E 4 GPa dry);
- oil/water bridge cohesion −c_a φ_w² (φ_w the fibre fraction weighted by wetting; sebum creeps 50 mm/day from the
  root, adhesion per length γπr·o/(o + 0.5), γ 30 mN/m for sebum, 72 for water);
- lagged smoothed Coulomb friction Σ μ N_y ε (√(1 + |Π_t(y − ȳ_y)|²/ε²) − 1), ε 1 mm, μ = 0.5 for the load the skin
  carries, 0.25 hair-on-hair (wet; 0.13 dry), ȳ_y the sample's anchor, N_y its normal load;
- the acts' energy (brush, hand) through a hook; a conditioning term keeping |u| ≈ 1.
Solver: L-BFGS-B in Jacobi (or banded) scaled variables, energy per guide in guide-weight·mm. Growth by continuation:
the active arc grows through 30, 60, 100, 160, 240, 330, 430 mm, then the longest guide (40 evaluations per stage in 2
sub-solves); release 135 evaluations in 3 solves; final gated stage capped at 4000 evaluations, scaling rebuilt every
300. Fixed evaluation caps make a seed reproduce exactly.

**Friction anchors kept across a lag** (`actions.lag_stick`). After each sub-solve, a sample whose new anchor moved
less than ε in the contact's tangent plane has stuck and keeps its old anchor; one that slid further gets an anchor
trailing ε behind it along its slip, so it keeps ≈0.71·μN (1/√2) against further sliding; new contacts are anchored
where they are. Without this a held layer creeps by up to ε per lag.

**The gate** (verified in code: `gate` 0.05, `gate_de` 1e-4). Converged when BOTH:
max over nodes |τ_{j−1} − τ_j| / max(local weight, Σ_terms |force of each term on the node|) < 0.05, where
τ_j = (I − d_j d_jᵀ) g_{d_j} / l_j is the transverse force segment j carries (the tip by τ_last; the root node is held),
the local weight is half of each adjacent segment's weight; AND the relative energy change over the last 20
evaluations < 1e-4 (energy measured from each guide's root height). The final stage stops as soon as the gate passes.
Recorded: styles passing at 0.038 and 0.047; one curl-index style failing at 0.992 (see §2).

**The cached grown head** (`head.py`). Saved (history, not recomputable): the rods' directions D, rest turning ρ, the
wave blocks, and the friction anchors (anchor, normal load, normal, index by guide/segment/sample; a cut that changes
the segment count remaps them). Rebuilt deterministically in seconds: roots, follicles, lengths, guides and members, the
body lattice. Key = SHA-256 over the sorted settings leaves of the hair, environment and sim layers (plus the few
style keys the head reads; a leaf added later at its default leaves the key unchanged), the case's collider, and code
hashes (whole growth files plus the specific functions the stage runs). Fingerprints = SHA-256 of dtype, shape and
bytes of the rebuilt arrays (roots X, guide index, member arrays; per snapshot segment lengths, arcs, stiffness,
weight). A rebuild whose key or fingerprints differ raises `HeadMismatch` and is refused; a styled state made on
another head is refused the same way. Observed: the guard refused a run whose follicle directions differed.

**Collision with the body as a field** (`field_collider.py`). The body the renderer draws is sampled into a distance
lattice (`FieldSDFGrid`); where the field store holds no brick, a node's side is found by flood fill from the lattice
boundary and its size from the distance to the nearest known node. Roots are drawn from the follicle map's samples
re-seated on the field. No mesh is built.

## 5. FibreLayer: the review renderer's hair
**First recorded:** 2026-11-10 to 11-11. **Status:** BUILT AND MEASURED (in use in the review renderer).

**Data layout** (`fibre_layer.py`, `.cuh`, CUDA via runtime compilation). Per strand: NPT = 24 points (n, 24, 3),
radius per point, pigment multiplier per point or σ_a per unit radius per point (n, 24, 3), reference radius,
ellipticity, twist rate τ (rad/m), β_m, root skin normal, seed, length. Each strand is a centripetal Catmull-Rom curve
with second-order end ghosts. A uniform span grid (0.5 mm cells when the 90th-percentile point spacing is under 1 mm,
else 2 mm) lists each span (strand × 23): its curve is sampled every 0.4 cell and listed in the cell of each sample, and
in the neighbour when the sample lies within radius + 0.1 mm of a cell face; built on the card in two passes (count,
prefix sum, write).

**Primary rays.** Per pixel, a 3-D DDA walk of the span grid with a conservative sphere pre-test per span
(radius chord + 1.2 × max radius, widened by the pixel cone), then the exact hit (curve chords + 3 Newton steps; the
oval section turning with twist; sub-pixel coverage = the cone footprint's overlap). The 8 nearest hits are kept
(`HitList8`), shaded and composited front to back; then the next 8 beyond the last, up to 6 rounds (48 fibres), until
coverage transmittance Tr ≤ 0.02. Outputs: premultiplied rgb, Tr, coloured transmittance Tc (per hit:
Tc ← Tc·((1 − a) + a·A_TT)), first-hit depth and strand id. Only fibre in front of the still's depth is drawn.

**Fibre scattering:** the Chiang et al. 2016 lobes (R, TT, TRT, residual) with a precomputed far-field table
(β_n 0.55, η 1.55); cuticle tilt α 3°, β_m 0.10 default, a/b 1.2, τ 40 rad/m. Absorption per unit *local* radius:
σ_a·r(s)/r_ref, so a tapered tip transmits more (lighter tips). Near field (true offset h) when the fibre is ≥ 1 px
wide, with per-scale cuticle tilt jitter when the pixel footprint is under 10 µm; otherwise the h-integrated far field.
Weathering: β_m × (1 + wa·min(s/0.4 m, 1.5)). A sized key widens β_m through the Chiang variance mapping:
β′ = (−0.726 + √(0.527076 + 3.248·x))/1.624, x = √(v(β_m) + ¼·r_key²), capped at 0.95.

**Light per hit** (shadow mode on; tests skipped when the hit's weight Tr·a ≤ 0.02):
1. body shadow: a soft SDF march toward the key (96 steps, start max(1.5 cells, 4 mm), step max(d, 1 mm),
   penumbra k = 8, or 1/r_key for a key wider than 0.125 rad);
2. self-shadow through the hair: `fl_Trc` toward the key from 2.5 r off the fibre, cone max(0.00465, r_key) + 2r/0.01,
   0.25 m; every fibre crossed counts once (in the cell holding its hit), each passing (1 − a) + a·a_f;
3. sky: the body's share 0.5·vis(up) + 0.5·vis(outward gradient), times the hair's transmittance along the outward
   direction N (0.35 rad cone, 0.25 m) — or four cosine-weighted cones (N and three at 60°, ¼ each) with the forward
   attenuation a_f = TT + 0.293·R + 0.15·TRT (0.293: the forward share of a smooth dielectric cylinder's R lobe);
   for long strands N blends from the root's skin normal to the body field's gradient over the first 10–40 mm;
4. the room per lobe: R and TRT each take what their centre direction sees: the environment map (or uniform sky)
   where it points away from the skin, the still behind the hair (darkened by the hair below it) where it points in,
   blended by smoothstep((cos + 0.25)/0.5);
5. multiple scattering from the mass beneath: per fibre the back share b (R + TRT + TRRT) and forward share f (TT),
   averaged over h ∈ {±0.25, ±0.75}; B = b/(1 − f²) (denominator ≥ 0.1, B ≤ 0.9); added as
   B·(key·vis·0.5 + sky·vis_sky·π/2)·dense/π, dense = 1 − mean(transmittance toward the skin side).

**Banded launches.** A whole-frame launch of the scalp layer ran 15 s, past the display watchdog (~2 s). Pixel kernels
run in bands of rows sized from the last band's timing toward BAND_S = 0.6 s (growth ≤ 1.5× per band, never under 32
rows, multiples of the 8-row block), and the layer runs one round of 8 hits per launch with each pixel's state carried
between launches (same arithmetic, same image).

**TT-spread composite** (the review's compositor). Light through the gaps between fibres keeps the exact pixel behind
(T); light through the fibres (Tc − T, the TT lobe, azimuthal width ~0.5 rad) arrives from a cone behind them:
out = rgb + T·img + max(Tc − T, 0)·spread, spread = img averaged over the cone's footprint at the gap between the hair
and what lies behind: σ_px = 0.5·gap / (metres per pixel at the hair), gap = clamp(z_behind − z_hair, 0, 0.5 m),
interpolated between Gaussian levels σ ∈ {0, 2, 6, 18, 54} px in log₃ space. This is a screen-space approximation of
the lobe's cone (adopted 2026-11-10: masses read solid, without a glass look). Several hair layers are composited back to
front per pixel by each layer's first-hit depth.

**Body lattice repair** (`repair_far`). The field store is a true distance only near the surface (bricks clamp at
±12 mm; beyond, a nearest-node fill with kinks). For the hair's light the lattice is made a signed distance everywhere:
inside the band |A| < 9 mm the field's own value; beyond it, the Euclidean distance to the nearest node of a reliable
shell (5–9 mm) plus that node's value, per side; blended over the band's last 1 mm and smoothed (σ 0.7 cells) only
outside it.

## 6. Hair occluding and occluded by skin (exact, lean shadow rays)
**First recorded:** 2026-11-11. **Status:** BUILT AND MEASURED (default for scalp hair since 2026-11-12).

**The fault it replaces.** One cone of the light's size through the span grid visits only the cells on its centre
line: once its footprint is wider than a cell it misses most fibres inside it and over-weights the rest (hard black
masses with specks under scalp hair; streaks under lashes).

**Lean exact rays** (`lean_begin` / `lean_refine`, kernels `fl_occ_pass` / `fl_occ_finish`). Per skin pixel, the
surface point (2·10⁻⁴ m short of the still's depth, jittered ±0.5 px) shoots EXACT rays (cone 1e-7 rad, 0.06 m) to:
the key, a cap of its angular radius (floor 0.02465 rad); and six sky caps of 0.12 rad (straight up, four at 50° from
up, and back toward the camera). Directions come from an R2 low-discrepancy sequence (α = 0.7548776662, 0.5698402910)
rotated per pixel by hashed offsets. Per pixel 12 floats accumulate: key rgb sum, sky rgb sum, key luminance min/max,
sky luminance min/max, key count, sky count. Pilot: 8 key + 6 sky rays at every pixel, in launches of 2 + 2 rays.
Refine: up to 36 key + 24 sky rays, only at pixels whose skin is seen through the hair (camera transmittance through
the layer > 0.02) and whose pilot rays disagree (luminance spread ≥ 0.03). Result:
Ts = w_sun·mean(key) + (1 − w_sun)·mean(sky), w_sun = L_key / (L_key + (π/2)·L_sky) by luminance. The skin under the
hair is multiplied by Ts *before* the hair is drawn over it (each layer's shadow taken on bare skin). Measured: the
stratified exact-ray method came within 0.03 % of the total shadow of a 144-ray reference; the 2 + 2-ray launches stay
under 1 s (4 + 3 rays reached 0.996 s in the densest band of a three-quarter frame).

**Hair occluded by skin** is §5 items 1 and 3 (the body's soft shadow and its share of the sky, from the repaired
lattice). Brows and lashes on their 0.5 mm grid use stratified exact rays (`fl_skin_occ3`); a measured-medium variant
(per-cell extinction, mip-mapped, read at the footprint's level beyond a near distance) is BUILT, NOT the default.

## 7. The settled hair frame in the interactive window
**First recorded:** 2026-11-12. **Status:** BUILT IN A LANE COPY, NOT LANDED, NOT VERIFIED (partial measurements).

**The idea.** While the view moves or a stroke runs, the window draws hair with its fast splat. When the view is still
and the window's own frame has converged, the review renderer's fibre layer (§5) is drawn over the window's own frame
and depth, under the window's lights, and replaces the splat until any input arrives.

**How.** One job at a time on a worker thread, keyed by (groom, version, view key including the frame below); a newer
view, a changed hair, a hidden groom or any mouse press, wheel or key in the viewport cancels it at its next kernel
launch (each launch checks a stop flag). Its own non-blocking CUDA stream; bands of ~0.05 s (the review keeps 0.6 s).
The layer stays on the card per groom version and key radius and is freed when the hair changes; the body lattice is
repaired once per grid (§5); an out-of-memory error frees the layer and retries after 60 s, the splat meanwhile. Order:
build or reuse the layer → set the key light → skin under the hair × lean occlusion (§6) → layer → TT-spread composite.
Two causes found and fixed first: the unrepaired lattice made the body shadow read nearly everywhere (hair at 0.53–0.60×
the review's radiance), and unshadowed skin under translucent fibres read as "ghostly" hair; with both fixed the layer
measured 0.93–1.11× the review's radiance at the review's own camera and light.

**Measured in the real window:** lands 9.9 s (three-quarter view) / 11.7 s (side) after the view stops; first layer
build 27.7 s per hair version; worst kernel launch 0.9 s; card memory 4.25 → 4.49 GB. A body stroke during a job
stalled the window (0 fps, worst frame gap 1504 ms; 32.3 fps / 55 ms with the feature off): the bands ran on the default
stream. The fix (non-blocking stream, 0.05 s bands, cancel on input) is built and NOT re-tested.

**Multi-light fibre layer** (lane copy). Each extra light carries direction, colour, angular size and a shadow flag;
its direct terms are summed onto the key's; with no extra light the arithmetic is unchanged. Built and compiles; its
identity (bit-for-bit against the current layer), linearity (key A + light B equals the two alone) and headlight tests
have NOT been run.

---

## Prior work built on (not claimed)
- Fibre scattering: Marschner et al. 2003; d'Eon et al. 2011; **Chiang, Bitterli, Tappan & Burley 2016** (the lobes,
  the roughness/variance mapping, the far-field integration); dual scattering's forward attenuation, Zinke et al. 2008.
- Rods: discrete elastic rods (Bergou et al. 2008); super-helices (Bertails et al. 2006); the suspended curly rod and
  its 2-D/3-D transition (Miller, Lazarus, Audoly & Reis 2014, PRL 112:068103).
- Fibre mass: van Wyk 1946 (P = kEφ³); Goldstein, Warren & Ball 2012 (the ponytail; notes that van Wyk's law does not
  fit their data — an open item here); Bico et al. 2004 (elastocapillary coalescence of wet fibres).
- Hair biology and classification: Thibaut et al. 2005 (curvature programmed in the follicle); de la Mettrie,
  Loussouarn et al. 2007 (curve diameter, curl index); Lasisi et al. 2016; Franbourg et al. 2003; Wortmann et al.
- Hair simulation with water and styling: Ward, Galoppo & Lin 2004/2007 (wetting, spray, blow-drying, cutting);
  MIRALab virtual hairdressing (2006–2010); Fei et al. 2017 (liquid in hair); the industrial patent US10460054
  (Cosserat rods, directional friction, liquid bridges, brushing). Random Fourier features: Rahimi & Recht 2007.
- A 2026-10-31 search found every ingredient somewhere and none combining follicle-grown hair with measured traits and
  styling by physical passes on one persistent state, nor drying modelled as a change of the hair's own rest shape.

## What is not claimed / open
- Not claimed: the Chiang lobes, discrete/Kirchhoff rods, L-BFGS, van Wyk's law, Coulomb friction, random Fourier
  features, R2 sequences, DDA grid walks, Catmull-Rom curves, SDF soft shadows, as such.
- Open: the curl-index full-head style failed its gate at a handful of nodes; it needs the gate diagnosed before any
  result from it is used.
- Open: the TT-spread composite is a screen-space approximation; a fibre-resolved forward-scatter of what lies behind
  would replace it.
- Open: section ellipticity is stored, not yet a bending anisotropy; per-type curl-index values are design values.
- Open: friction anchors are fixed in space between lags (hair sliding over hair does not drag its neighbour by
  friction; pressure couples them).
- Open: van Wyk's pressure law against Goldstein et al.'s fitted equation of state.
- Open: no dynamics yet (statics only); the settled window frame (§7) and the multi-light layer are not landed.
- Open: the photo-to-recipe reader has not been run in a closed loop.
