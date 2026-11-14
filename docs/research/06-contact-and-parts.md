[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 6. Contact and parts

**How to read this section.** It is a dated technical record of how Field Studio keeps separate bodies and tissues
apart, lets them touch, and solves what happens when they press on each other. Times are internal provenance from the
project's working record and commit log (Pacific time, 2026); the date this text became public is the date of the
release that carries it. Every mechanism carries one evidence state from this list: *proposed*, *implemented*,
*measured on named cases*, *reproduced under a frozen configuration*, *accepted in the workflow*. Nothing in this
section has been reproduced under a frozen configuration yet, and we say so where it matters. Patent remarks are our
own reading of claim text, not legal advice, and no clearance is claimed. Nothing here is claimed to be new to the
world; published ingredients are named with their sources.

**Machine.** One Windows PC with 32 GB of RAM and one 8 GB RTX 5050, shared with other jobs. The 11-12 solver runs are
CPU runs, one solve at a time at 4 threads (numba / OpenMP / MKL all set to 4). Because the machine was shared, seconds
are indicative; iteration counts are exact.

**What is and is not in the public code.** The IPC-style solver in `engine/contact/` (B.3) is **out of the public code
set**: it is flagged in our review; held out of the code pending redesign (B.5, B.6).

---

### B.1 Parts: separate fields that touch without fusing

**Evidence state:** measured on named cases (closed lips on the human model; a synthetic eyelid on a globe; synthetic
pressed balls). **Dates:** per-part entries in the store and renderer 11-06; pair settings 11-11 01:59; the general
design below 11-11 16:20-21:26.

**The problem.** If a body is one signed distance field, two surfaces that touch become one surface: closed lips merge,
a lid welds to the eye, pressed fingers become a mitten. The crease between them is narrower than a lattice cell, so a
single stored value per sample cannot hold both.

**The mechanism, step by step.**

1. *A part is its own field.* Each part has an id (16-bit; 0 is the body), a name, a kind (tissue, analytic, garment),
   a material profile key, and its own **band**: its own smooth signed distance on the store's lattice within a narrow
   band of its *own* zero set. The band includes the stretch of the part's surface that lies inside another part (a
   closed lip's inner face, a lid's underside on the globe, a finger's side pressed into its neighbour). Without that
   hidden stretch nothing can later open, slide or pose the pair. Storage: per level-0 brick (0.25 mm samples, 9x9x9
   half-floats), rows sorted by part.
2. *Everything else is a cache rebuilt from the parts by one call.* The union field (minimum over the parts' bands),
   the per-brick contact entries, the fine leaves, a part label per sample, and the transported surface attributes are
   rebuilt for a region by one `rebuild` call. No writer edits a cache by hand. A rebuild with no change writes nothing
   (before this rule a zero-target operation had re-distanced 32,927 samples by up to 1 mm).
3. *Contact entries ("k entries").* A brick where two or more parts come within two cells of each other stores one
   entry per part: that part's own field in that brick. The renderer takes the minimum over each part's own trilinear
   interpolant, so the crease is the true crossing of two smooth surfaces and each side keeps its own normal. Check
   "K0": the minimum over the entries equals the stored union bit for bit at every slot sample.
4. *k is soft.* Measured on synthetic adult shapes (11-11), the number of parts in one brick stays at 6 or fewer at the
   0.25 and 0.5 mm levels, but a closed mouth (28 teeth, gums, palate, tongue, cheeks, lips: 36 parts) reaches 10 at
   1 mm and 17 at 2 mm, and clasped hands with a part per phalanx reach 11. So there is no fixed limit of 8: one 16-bit
   id per entry, up to 32 per brick, refused beyond that and never silently dropped. A 12-part synthetic cluster drew
   with 0 fused points (the old limit of 8 gave 563).
5. *Leaves.* Where the gap between two parts is thin, 0.125 mm leaves refine the entries.
6. *Area is the real limit.* Memory grows with contact area times k. One forearm under four garments cost 200 MB of
   entries at 0.25 mm. So garments, held objects and furniture are separate bodies with their own fields, traced as
   their own volumes; entries are for tissue-on-tissue contact in small zones and for small-area contacts.

**Tissue-pair rules.** Every pair of parts carries one setting, and the setting means one stated thing in each
consumer:

| setting | store | renderers and Field Studio's draw | brushes | motion solve |
|---|---|---|---|---|
| touch | both bands; entries where they meet | minimum over each part's own interpolant; a crease; the visible part's own normal | a stroke on A never moves A deeper into B than it already is; B unchanged | contact with static friction |
| slide | as touch, flagged | as touch | as touch | contact with kinetic friction; tangential motion allowed |
| bond | one entry (smooth minimum, default width 0.5 mm): one tissue, no crease | no crease | bonded parts edit as one | tied nodes |
| pass | both bands kept | two surfaces drawn where they cross | no clamp | skipped |

Switching between touch, slide and pass changes a flag; switching to or from bond rebuilds that pair's entries.
Examples from the record: lips touch each other and are bonded to the face at their outer borders; a lid touches the
eye at rest and slides in a blink; a tooth and its gum are one touch pair, never a bond; a garment is its own part with
slide. A per-pair **contact record** (gap, real-contact share, fluid fill, friction, adhesion; per side roughness, film
and stiffness) has a schema and a home on a sparse 0.25 mm lattice over the pair's zone; only friction is filled today
(from the materials library's pair table), and the other slots stay empty until a measurement or cited source fills
them.

**Labels carry the material.** The part's material key selects its tissue layers, friction, thickness and damping from
the materials profiles (B.3), so labelling a region applies its physics.

**Contact that changes over time.** Rest is authoritative: every pose or edit rebuilds the caches from the parts in
the changed bricks only, never from an earlier pose, so error cannot accumulate and the return to rest is exact by
construction. A brick gets entries wherever two parts come within four cells (hysteresis), so a closing gap is drawn by
entries before it narrows under two cells. Self-contact that appears in a pose is designed as a split of the part into
sheets for that pose (entry id = part and sheet); that part is designed, not measured.

**Measured.**

| case | result |
|---|---|
| Two pressed synthetic balls above a body (11-11) | K0 0 differences at 0.25 / 0.5 / 1 mm; 0 fused points at 0.5 mm against 5,332 for the union alone; a 1 mm separation shows each hidden cap within 6.2 um (the older reader: up to 995 um) |
| Closed lips on the human model (11-11) | built as two lip parts plus the face; store cost 12.1 MB of bands, 3.6 MB of entries, 14.3 MB of leaves; opened in Field Studio as three objects in 1.0 s |
| Synthetic lid on a globe, 13 blink frames (11-11 20:50) | 0 stray points; margin gap 1.0 um typical / 6.4 um worst; 0 fused frames; return to rest byte-identical |
| Field Studio open / draw / brush / write-back on the lips | open, brush clamp and unedited write-back marks passed; one draw-agreement mark was argued rather than passed |

**Failures and corrections.** (a) Field Studio first read each part as a dense lattice box; estimated for a whole body
of about 90 parts that is 3-6 GB of RAM, over budget. Parts were changed to sparse stores read from the band rows
(11-11 19:38): about 16 MB of program data per lip measured; the whole-body figure (about +1.6 GB RAM, about 1.6 GB of
card memory) is an estimate, not a measurement. (b) A record that the lips rest closed was corrected: on the current
model they rest 0.3-1.2 mm apart across the middle and touch only near the corners; closed is a state, not the rest.
(c) The leaf writer ran out of memory on a 34-part mouth and was rebuilt to work in bounded batches (2.49 GB peak).

### B.2 Solid by default, with designed cavities

**Evidence state:** measured on named cases (the human model's store). **Date:** first stated 11-12 00:32;
built and published into the working model 11-12 07:50.

The rule: objects are solid; a designed cavity is never filled, and is made deliberately, with the cut tool or as a
labelled part, so the tool knows it is meant. Tests must show zero unintended pockets.

As built: the model's interior, which an earlier surface-band store left unknown beyond its band, is given the inside
sign at the coarsest level; enclosed air pockets of 0.01 mL or more that no design accounts for are found on a 1 mm
grid by flood fill from the exterior, given the inside sign and re-distanced inside the pocket mask only. Checks on the
published version: outside the edited masks 0 differing samples at every level; the head's outside-reachable air
(mouth behind parted lips, nostrils, ear canals) is 8,725.9 mL before and after with 0 voxels changed, and the mouth
box 268.43 mL with 0 changed; rays from front, both sides and 30 degrees below differ in 0 of 31,403 per view. The
model's inside volume on a 6 mm grid went from 27.4 L to 60.0 L. **Left open:** about ten single-sample enclosed specks
(about 0.3 uL each) and one 1.4 uL thread remain and are recorded as defects to fix at the shape. A general "no
unintended pocket" guard at every commit is not built.

### B.3 The first solver: IPC on fields (not in the public code; the measurements below compare against it)

**Evidence state:** measured on named cases (its own proof suite and the cases below). **Date:** built 11-07 to 11-08
(first commit 11-08 11:35); self-contact and cut cells 11-12 03:32-06:38. **Not in the public code** (B.5).

**Credit.** The model is the incremental potential of IPC (Li et al., *Incremental Potential Contact*, ACM TOG 39(4),
SIGGRAPH 2020; reference code and IPC Toolkit under the MIT licence). No code was copied; the equations come from the
paper. The linear solver is PARDISO through `pypardiso` (BSD-3; MKL under the Intel Simplified Software License), with
SciPy's SuperLU as fallback.

**How it works.**

1. *Bodies.* A soft body's rest shape is a signed distance field phi0(X). Its deformation is a displacement field on a
   regular lattice of spacing h (4-10 mm in these runs), trilinear cells, no mesh. Each Gauss point is weighted by the
   share of its octant inside phi0 < 0. A one-cell ghost envelope keeps the map defined just outside the skin. An
   optional two-level lattice (2h away from contact, h near it, hanging nodes tied to their parents) matched the
   all-fine indentation force within 0.72 % at about a quarter of the unknowns.
2. *Material.* Compressible neo-Hookean per layer, selective reduced integration, Poisson's ratio 0.49. Layers by depth
   below the skin from the materials profile; a Gauss point straddling layers takes their series modulus; bone is
   tied rigid. The body-skin stack used in every elbow run: epidermis 60 um at 4 MPa, dermis 1.5 mm at 40 kPa, fat
   18 mm at 26.1 kPa, muscle at 62.2 kPa. The profile cites optical coherence elastography of forearm skin for the two
   skin layers and in-vivo indentation of gluteal fat and muscle (26.1 +- 19.0 kPa at strain 0.16; 62.2 +- 10.3 kPa at
   0.19) for the deep layers, with ultrasound thickness surveys for the thicknesses. These are literature values, not
   measurements on the model (see B.9, experiment d4, for a challenge to the fat value).
3. *Gaps.* Point against a rigid field: phi_R(x) minus thickness. Point against a soft body: phi0_B(X_B(x)), by
   inverting B's deformed lattice map, with exact gradient. Self-contact: the same lookup, counting only material
   cells (not ghost cells or extrapolated maps), skipping hits whose located rest preimage lies within about 0.5 mm of
   the sample's own rest point, and dropping inverted preimages.
4. *Contact law.* The log barrier b(d) = -(d - d_hat)^2 ln(d / d_hat) for 0 < d < d_hat, zero beyond; d_hat = 0.1 mm;
   row stiffness kappa_k = E / d_hat x A_k (E the softer body's modulus, A_k the sample's area). Friction is lagged,
   smoothed Coulomb, static to kinetic, with stick width max(1e-3 m/s x dt, 1 um); coefficients per material pair.
5. *Solve.* Newton on the incremental potential with a backtracking line search; each step bounded by conservative
   (Lipschitz) continuous-collision bounds and a trust region of 0.25 h; a step is accepted only if every gap and every
   volume ratio J stays positive. Residual tolerance 1e-7 of the load; friction re-lagged up to three times.
6. *Cut cells (11-12 05:33).* A cell cut by the skin measures tissue at its in-tissue Gauss points only; the empty part
   is held by 0.1 x the cell's tissue stiffness in the rotation-invariant form mu |F - R|^2, finite for every F. Before
   this, Gauss points in a cut cell's empty part hit an invisible J > 0 wall and an elbow sweep stalled at 67.5-72.5
   degrees.

**Measured (11-08, with other jobs running; GPU figures on the 8 GB card).** A layered slab of 45.4k unknowns pressed
by a sphere: 111.7 s for 46 Newton iterations at first, 11.4 s after cached assembly, a Jacobi-preconditioned conjugate
gradient on the card and card-side tissue energy. Interactive frames, warm-started: 0.086 s per frame at 11.3k unknowns
and 0.1005 s on a two-level lattice of 10.8k unknowns (force within 0.25 % of the all-fine answer); 0.277 s at 45.4k.
Friction check "T0" (a block on a tilted plane): stick passes; slide acceleration within 3.45 % of Coulomb's law.

**Failures on record.** One of twelve regression tests fails as written (a viscoelastic swing on the two-level lattice
misses its residual on a few near-zero-load steps). A seated-load proof has one pressure mark that fails high. A fabric
band on the model's ribcage passes its volume mark and fails its pressure mark (ratio 1.30-1.65 against a smooth-curve
estimate); tracing showed the band bears on ridges and bridges hollows, obeying Laplace's law with its own curvature
within 8 %, and one residual curvature difference is still open. Cloth self-contact, fibre twist and adhesion are not
in.

**Earlier round, for the record (11-05).** Before this solver, one soft-tissue region was settled with linear hex
elements on a 2 mm lattice and frictionless contact solved as a linear complementarity problem on the Delassus operator
(one sparse factorisation, FISTA iterations, re-linearised five times), with fat 2 kPa and dermis 35 kPa after
Pailler-Mattei et al. 2008. It was a single-region tool and was superseded by B.3.

### B.4 Skin-on-skin self-contact rows

**Evidence state:** measured on named cases (the stand-in elbow); built in a working copy, **not landed**; its own pass
mark is not met. **Date:** 11-12 06:00-08:50.

At a closing crease the skin folds onto itself below cell scale, and the volume-map self-contact of B.3 first sees the
contact when it is already about 0.3 mm inside. The replacement measures skin against skin directly: each skin sample i
is paired with facing samples j in the deformed state; the gap is point-to-plane, d = (x_i - x_j) . n_j - t_i - t_j,
with n_j the rest normal carried by the cell's deformation gradient (F^-T n0, normalised); one row per sample, the
partner whose plane foot is nearest; rest neighbours within two sample spacings never pair.

Two faults were found and fixed. First, "0 pairs crossed" was a blind metric, because rows form only at positive gap;
an independent counter showed two pairs 127-217 um past each other at 82.5-90 degrees. A crossing guard now keeps a row
for every pair that was outside at the last accepted state. Second, the cause was sample spacing: at a crease narrower
than the 2 mm sample spacing the surface crosses *between* samples (at 80 degrees, 6 samples inside at 0.5 mm sampling,
0 at 2 mm). Samples went to 0.5 mm in the fine crease cells. With both, 0-70 degrees is clean at 19-33 s per step, but
from 72.5 degrees a few samples sit 0.27 mm inside tissue without any pair crossing: the 4 mm lattice folds over its
own neighbour across a cell face (B.9). One run ended in a memory error (77.7 million candidate pairs from an
over-wide search margin).

### B.5 The patent finding on the IPC-style solver

**Date:** 11-12 08:22 (the reading); 11-12 about 09:00 (a wider search); a further check of our readings the same
morning. **This is our own reading, not legal advice; no clearance is claimed.**

**The patent.** US 12,216,969 B2, "Methods of contact for simulation", NVIDIA Corporation, filed 2020-09-04, granted
2025-02-04, term adjusted by 1,033 days. Read from the USPTO document, claims on its pages 96-97. A pending
continuation, US 2025/0131161 A1 (application 18/999,835, filed 2024-12-23), was also read.

**The flag.** As we read them, the patent's independent claims concern simulation of contact in which forces
determine constraints, a preconditioner is applied to a gradient descent, and an energy minimisation is solved in a
primal formulation. Our first solver (B.3) is a barrier energy minimised by Newton steps, so it was
flagged in our review; held out of the code pending redesign. We publish no conclusion on whether any claim is or is not met;
that is a question for counsel, and so is the effect of earlier published work (IPC itself was published by other
authors before this patent's filing date). Two cautions were recorded: a rewrite was not yet
proven necessary, and a replacement was not yet proven free of the claims. Decision (11-12 about 08:35): build
around it. Direction of the design-around: the constraint solver of B.6, which solves the contact conditions as
equations with multipliers and minimises no energy. The same search also flagged US 7,616,204 (position based
dynamics, to about 2027-08) for any position-level loop.

### B.6 The constraint solver ("R1"): contact as constraints with multipliers (the rebuild, underway)

**Evidence state:** measured on named cases (the friction check and the stand-in elbow); a prototype in a working
copy of the contact code; it fails the broad soft-on-soft cases. **Date:** designed 11-12 08:45; prototype
09:30-15:21.

**What it is.** A time step solved as a nonsmooth system of equations, not as the minimum of an energy.

- *Unknowns:* velocities v+ (positions follow, x+ = x + h v+), one normal multiplier per contact row, one 2-vector
  friction multiplier per row. In quasi-static mode the unknown is the load step's displacement increment.
- *Equations F(z) = 0:*
  1. momentum: M (v+ - v) / h - f_int(x+) - f_visc - f_ext - J_n^T lambda_n / h - J_t^T lambda_t / h = 0;
  2. Signorini contact through the Fischer-Burmeister function, phi(a, b) = a + b - sqrt(a^2 + b^2) = 0 with a the
     scaled gap and b the scaled multiplier, which gives d >= 0, lambda >= 0, d lambda = 0: contact closes to zero gap,
     with no barrier offset;
  3. Coulomb friction through the Alart-Curnier projection, lambda_t = proj onto the disc of radius mu lambda_n of
     (lambda_t - r u_t), u_t the step's slip.
- *Solve:* one generalised (semismooth) Newton method on the whole residual, with an element of the B-subdifferential;
  the Jacobian is unsymmetric once friction is in; factorised directly (PARDISO unsymmetric, SuperLU fallback).
  Globalisation is a backtracking search on |F|^2 / 2, never on an energy.
- *Parameters and additions, each from a measured failure:* smoothing continuation of both nonsmooth functions (tau
  from 1e-2 driven to 0 inside each step, judged on the exact residual) after active-set chattering on a sliding face;
  a non-monotone search (memory 8); the Alart-Curnier r per row from the Delassus diagonal, times 0.3 (times 1.0 failed
  the landing 3 of 3; 0.3 passed 5 of 6); a 1e-6 term on the multiplier diagonal of the Jacobian only, for redundant
  rows; exact tangents (the baseline's projected tangents are right for a minimiser and wrong for Newton on equations:
  9e-4 relative error in cut cells, 6e-10 after); the exact curvature of each skin row (a per-row Hessian by the
  complex step), which took one crease step from 302 iterations / 521 s to 47 / 47 s; pseudo-transient continuation on
  the Jacobian for static solves; a descent test with fallback step -J^T F; an adaptive trust radius on node motion
  (0.25 mm start, 2 mm cap); row sets frozen during a line search; deterministic linear algebra (MKL reproducibility
  mode), without which the stick/slip outcome depended on thread count and summation order.
- *No pass-through:* a facing pair keeps its row at d <= 0 unless it was already clearly back to back (d < -1 um) at
  the last accepted state, so a new crossing is a violated constraint that Newton restores. Every accepted state is
  checked by an independent counter (samples inside tissue; skin pairs crossed).

**Published basis (as named in the design record):** Alart and Curnier 1991; Simo and Laursen 1992; Fischer 1992; Qi
and Sun 1993; De Luca, Facchinei and Kanzow 1996; Christensen, Klarbring, Pang and Stromberg 1998; Moreau 1988 and
Jean 1999; smoothing methods after Chen, Qi and Sun 1998 and Kanzow and Pieper 1999; non-monotone search after Grippo,
Lampariello and Lucidi 1986; pseudo-transient continuation after Kelley and Keyes 1998; complex-step differentiation
after Squire and Trapp 1998. The nearest modern work is Macklin et al. 2019, *Non-Smooth Newton Methods for Deformable
Multi-Body Dynamics*; the search found no US patent family for it.

**How it differs, element by element (our reading).**

| claim element | R1 | absent? |
|---|---|---|
| 12,216,969 (c): energy minimisation using a primal formulation | zeroes a nonsmooth complementarity residual with explicit multipliers; the specification describes "primal" as optimisation-based and usable without Lagrange multipliers, and sets it against constraint-based (dual) formulations; Coulomb with Signorini is not the minimum of any energy | absent on our reading; this is the element relied on |
| 12,216,969 (b): preconditioner applied to a gradient descent | a generalised Newton step on F from a direct factorisation; search on the residual norm | absent on our reading, but weaker alone: the specification calls Newton with a Hessian inverse a preconditioned descent, and a residual-norm search is a descent on a merit |
| 12,216,969 (a), (d) | forces are computed and states updated | present |
| dependent claim on dropped geometric stiffness | R1 keeps the constraint curvature | the opposite |
| continuation, claims 21 / 28 / 35: "a pre-conditioned force-based formulation" for at least two objects, at least partly in parallel | by the family's own specification "force-based" names the primal kind; R1 is the constraint-based kind and applies no preconditioner to any formulation | absent on the specification's own definitions |

**The stated weak point.** If "force-based formulation" were read more broadly than the specification defines it (any
method whose equations balance forces), R1's momentum equation is a force balance, and the difference would rest on
the word "pre-conditioned" alone. The design constraints kept for that reason: linear systems solved by direct
factorisation or a Krylov method on the full constraint system; no preconditioner applied to a momentum-only system as
the method of simulating the objects; the graphics-card path re-read against these claims before it exists. **The
continuation is pending**, so its claims can change; this reading is to be redone when it grants or is abandoned.

**Constraints taken from other patents the search found (our reading; each names what R1 does not do):** US 7,616,204
(no predict-then-correct positions; static mode has no velocities at all, dynamic mode is flagged under a broad
reading); US 11,270,041 (hard multipliers, no compliance tied to elastic energy); US 9,589,383 (one global solve, not
per-constraint parallel iteration); US 11,200,356 / 12,755,068 (no per-sub-interval bias accumulation); US 10,614,257
family (no mass splitting); US 9,443,341 / 10,198,844 (no projected Gauss-Seidel over separate trees); US 11,210,833
(no alternation of elastic relaxation and contact projection with an error term); US 10,198,847 (no polyline search,
no push-out pre-step); US 11,675,935 (no intermediate implicit surface tied to body vertices). Not reached by the
search: European, international and Chinese families, and US applications filed after about 2025-05.

**Measured.** Friction check: stick 0.193 mm (mark < 0.4), slide within 0.46 % of the law (baseline 3.45 %), no
pass-through, 11.4 / 7.5 Newton iterations and 1.95 / 0.84 s per step. Stand-in elbow with skin rows: every state
converged to 1e-7 up to 88.44 degrees with 0 skin pairs crossed and pairs touching at zero gap; 6-20 iterations per
step to 75 degrees, about 200 iterations and 279 s per step at the apex.

**Failures.** R1 times out on every scene with broad contact between two soft bodies (fingertips, lips, a lid on an
eye): many samples pull on near-redundant rows. Merging each pair's two one-sided rows into one converged but crossed
the dropped plane (11 pairs, 294 um), so it was reverted: where skins meet nearly at right angles both planes must
hold. The friction check's landing step sits near a convergence boundary and needs one step halving. Speed against the
baseline's interactive bench and the full proof suite are not run.

### B.7 The magnetic-field contact concept

**Evidence state:** measured on named cases (synthetic scenes). **Date:** first stated 11-12 about 15:30; patent
reading of the options 15:31; built 15:50-17:15 (in a working copy of the contact code).

**Idea.** A repulsion between surfaces that is zero until they are very near and reaches its limit where they touch,
read directly from the fields.

1. *Gap.* A contact sample on body A takes its gap d from body B's stored signed distance field (trilinear value and
   gradient), the same rows the other solvers read. This replaces primitive-pair search for the gap itself; it still
   needs a broad phase, interpolation, and for deforming bodies either a current field for B or a map into its rest
   frame.
2. *Law.* IPC's published log barrier (B.3, step 4) with d_hat = 100 um in every run and the baseline's row stiffness;
   force kappa b'(d) along the gap gradient. Bodies therefore rest at small positive gaps (49-95 um measured), not at
   zero. Calling that film a physical moisture layer would need separate evidence.
3. *Form N (hybrid).* The barrier is the contact law inside R1's Newton on the static force residual: no multipliers,
   no energy evaluated, exact tangent (bodies, kappa b'' J J^T, kappa b' times the gap's second derivative on skin
   rows, friction), pseudo-transient continuation, search on |F|^2. Each iterate is bounded by distance: no gap within
   reach may shrink by more than half its current value (linearised); a bound on step length, never a clamp on points.
   Friction is smoothed Coulomb on the step's slip with a 1 um smoothing length.
4. *Form E (standalone).* Explicit symplectic Euler; the time step bounded by stability (factor 0.5) and by the same
   half-gap rule; stiffness-proportional tissue damping (0.3 of the explicit limit) and a damped normal response per
   row (ratio 0.5); a mass floor of 5 % of the median node mass on nearly empty cut-cell nodes; static rest by dynamic
   relaxation (Gershgorin fictitious masses, kinetic damping after Cundall).

A diverging force does not by itself prevent crossing in a discrete solve; non-crossing here rests on the step bound,
the sample spacing and the field's accuracy, and was checked by the independent counter.

**Results (synthetic scenes; 4 threads, one solve at a time; seconds indicative).**

| Case | Form N | IPC baseline | Form E | R1 | Notes |
|---|---|---|---|---|---|
| Two fingertip pads pressed 2 mm | 27 s | 86 s | 519 s | timed out (> 600 s) | forces 0.353-0.355 N (within 1 %), gap about 80 um, 0 crossed, 88 samples in contact |
| Lips closed 0.8 mm | 16 s | 27 s | 295 s | timed out (> 600 s) | 0.0124-0.0133 N, gap about 92 um, 0 crossed |
| Eyelid drawn 1 mm onto a rigid eye | 73 s | 7.4 s | 18 s | timed out (> 300 s) | **a reversal:** soft-on-rigid favours the baseline; all three agree at 0.373 N, 87.8 um |
| Hand pad on cheek, 2.5 mm press then 1 mm sideways | **did not converge in 10 min** (first load step) | not run | not run | not run | open; suspects (friction smoothing, load step size) untraced |
| Elbow bend, stand-in arm | every state converged to 88.75 deg, 0 crossed | "90 deg" with 85-90 unconverged at its iteration cap | not run | converged to 88.44 deg, 0 crossed | the lattice still folds locally at the crease (0.23-0.24 mm in all three); hard steps took about 4 min |
| Friction check (block stick / slide) | pass (0.292 mm; 1.68 %) | pass (3.45 %) | **fail** (2.4-3.7 mm slip; +170 to +230 %) | pass (0.193 mm; 0.46 %) | explicit friction loses grip to contact-face vibration; stronger damping changed little |

A first attempt at the cheek and lid scenes measured nothing, because the driven regions selected no nodes; that was a
scene-construction fault, fixed before the rows above.

**Reading.** On the two soft-on-soft scenes Form N was 1.7-3.2 times faster than the baseline at the same answer; on
soft-on-rigid it was about 10 times slower; one scene did not converge. That is promising on the tested cases and not
established as generally better.

**Relation to patents (our review; no clearance claimed).** Form N is a barrier force inside an equilibrium Newton
solve, as in IPC-family methods, so it is flagged in our review under US 12,216,969 and its pending continuation;
not evaluating an energy value, or renaming the search objective, does not by itself change that. Form E is
explicit, with no minimisation, and is flagged under the pending continuation's force-based wording. Rules kept for
both because of other patents the search found: bound the time step rather than clamp points (US 7,616,204), and
stay synchronous (US 9,020,787 concerns a divergent barrier together with kinetic data structures and asynchronous
variational integrators, none of which are used). These flags were accepted for testing. Neither form is in the public code.

### B.8 A field through the environment as the contact sensor

**Evidence state:** proposed; not built. **Date:** first stated 11-12 about 15:55; prior-art check 19:56.

The idea (a touch-screen analogy): a value held in the space between bodies reports where anything
approaches anything, so candidate pairs are found without pair searches. It is a sensor only; the response stays with
a contact law (B.6, B.7). The planned build-out: one value per node of the *existing* sparse lattice in the air region,
for example the distance to the nearest other body, read from the bodies' own distance fields and refreshed only in
tiles where something moved. The pass marks were written before any run: recall of every contact row the per-pair
search finds (0 missed); sensor cost at rest no more than 10 % of the reference search; cost when one body moves no
more than the reference; memory stated. Known problems: self-contact (surfaces of one body need a label-free signal
such as the local thickness of free space); a gap narrower than a cell vanishes from the grid, so the sensor must hand
over to per-body fields before that; a global update is itself a cost to measure.

**The air-mesh check.** The closest published relative found is Air Meshes (Mueller, Chentanez, Kim and Macklin,
SIGGRAPH 2015): the air between bodies is tessellated, with a non-inversion constraint per air element. A keyword-level
search found no patent on air meshes or on tessellating empty space for contact, and no open release from its authors'
organisation. That search is **not conclusive**: two full-text patent search sites were blocked to the search, and some
related grants were seen by bibliography only. What our sensor will not do: (1) tessellate empty space into elements;
(2) use a non-inversion constraint per air element as the contact rule; (3) maintain a mesh's connectivity as bodies
move. It holds values on lattice nodes, with no elements and no connectivity. Mechanisms named as candidates to borrow
for updates are incremental distance transforms (FIESTA, VDB-EDT) and the sleeping of resting regions used by rigid
body engines.

### B.9 The elbow study: why the bend stalls near 85-90 degrees

**Evidence state:** measured on named cases (a synthetic stand-in arm); the cause is **not established**.
**Dates:** 11-12 11:00-20:00.

**The case ("E1").** A stand-in arm of cones and ellipsoids with two capsule bones (radii 12 and 11 mm) and a hinge on
the limb's centre axis; the layer stack of B.3 as one bonded solid; a 4 mm lattice refined from 8 mm near the joint;
bent 0 to 90 degrees in 2.5-5 degree steps from a dual-quaternion skinning guess, with skin-on-skin rows (B.4).

**Angles.** On the stand-in arm, angles are measured from its straight rest pose. The human model's elbow rests bent
22.2 degrees, so for it "from rest" plus about 22 degrees gives the anatomical angle. The target for a full bend is
145-155 degrees anatomical.

**What stalls.** The baseline hits its 80-iteration cap from 82.5 degrees and its 85-90 degree states are unconverged
(residual 1e-5 to 5e-5); R1 converges to 88.44 and Form N to 88.75 degrees and no further. Three contact laws stalling
at similar angles suggests a shared upstream limit, but the runs also share geometry, material, discretisation and
parts of the solver, so that is not yet a diagnosis.

**Measured.**

- *The lattice folds.* Skin samples sink into the body's own neighbouring tissue with no skin pair crossed, and the
  onset moves with cell size: from 55 degrees at 8 mm cells (0.55 mm, growing to 2.6 mm), from 80 degrees at 4 mm cells
  (16 um, then 0.12-0.24 mm). It is the same under all three laws. (B.4's earlier run, with different sampling, saw
  0.27 mm from 72.5 degrees; the records do not reconcile the two figures beyond the difference in setup.)
- *The tissue is crushed.* At 82.5 degrees 124 fold-barrier points are active, with the local volume ratio down to
  0.032.
- *Not volumetric locking.* From saved baseline states, all volumetric terms together are under 2 % of the elastic
  energy (21 of 1,136 mJ at 90 degrees), and the energy grows smoothly through the stall. An earlier statement that the
  cells lock everywhere was corrected the same hour.
- *Where the energy sits at 90 degrees:* 44 % in the front crease (505 mJ in 132 mL) and 22 % in a 22 mm plug of soft
  tissue between bone ends that do not meet (245 mJ in 112 mL).
- *A reversal.* A first finding blamed the hinge's position: rotating the forearm rigidly forces 34.5 mL of overlap at
  45 degrees and 75.7 mL at 90, and a hinge 20 mm further forward would cut that five-fold. Measurement on the Visible
  Human Female's left elbow (U.S. National Library of Medicine cryosection data, 0.33 mm in-plane, 1 mm slices; used
  for study and measurement only, not shipped) put the flexion axis, picked through the trochlea and capitulum to
  within 1-2 mm, 35.8 mm from the back skin and 63.8 mm from the front: **36 % of the depth from the back**. The
  stand-in hinges at 50 %, so it already has less tissue in front of the axis than a real arm, and the "move it
  forward" reading was withdrawn. Caveats: one specimen, heavier than the model.

**One-change experiments (marks written before each run; baseline solver, 80-iteration cap, tolerance 4e-6).**

| experiment | change | verdict |
|---|---|---|
| d1 (19:42) | bones that articulate at the axis: a 10 mm condyle along the axis, a notch sector at 11-16 mm radius that turns with the forearm | **No on every mark.** Newton reaches the cap from about 80 degrees (the baseline: 82.5) and is worse below it; the joint core's share rose from 22 % to 37 % (520 mJ). Tissue bonded to both bones is sheared over a few mm. The lesson recorded: articulation needs sliding |
| d4 (19:51) | fat modulus 26.1 -> 9 kPa (shear about 3 kPa, after a search-level reading of in-vivo elastography; a lead to verify in full text) | **No improvement at the stall.** Energy at 90 degrees 1,185 against 1,136 mJ; Newton at the cap from 75 degrees |
| d3a | a thin soft slip layer at the deep fascia | a switch is built; no result on record |
| d3b | 2 mm cells in the crease zone | marks declared (every step below the cap, no substeps); not run on record |
| d2 | the axis at the anatomical depth | not run |

After d1 and d4 the one lever measured to move the stall is cell size at the crease, so the leading candidate is the
crease's discretisation: a 4 mm trilinear cell cannot fold into a crease a few mm deep. It remains a candidate.

**On the real arm ("E2").** By decision (11-12 about 19:50) the test moves from the stand-in to an
isolated elbow region of the human model's own field, with bones placed from anatomy (committed 19:56). Measured since (commits at 20:13 and 20:17): with the
stand-in's simple bones and skin rows, the baseline converged every state from rest to 90 degrees from rest (about
112 degrees anatomical), but with Newton at its iteration cap from 67.5 degrees from rest, four substeps a step from
82.5, the predictor's starting guess infeasible from 80, and 3 skin samples 0.35 mm inside tissue at 87.5-90. So
the real arm bends further than the stand-in before stalling, and it is still far from the 145-155 degree target. A
run with anatomical, articulating bones was in progress when this was written and is not a result.

### B.10 Store seams, the seam guard and the publish gate (brief; the store has its own section)

**Evidence state:** measured on named cases; in the live writer. **Date:** 11-12 09:06-10:39 and 18:47.

Contact work exposed two store faults. *Seams:* samples on a face shared by two bricks disagreed (108,237 level-0 seam
nodes in the model, 169 with opposite signs); fixed at every level to 0. The cause was in the writer; five real brush
strokes written back gave 4,053 level-0 seams with the old writer and 0 with the fixed one. A guard at commit now
writes every changed face sample into every brick that holds it, and a gate refuses to publish a version with seams or
duplicate rows. *Sides:* after carving a cavity, a closest-point-and-normal sign test flipped air to tissue in a thin
gap beside it (91 wrong samples at level 0 in one eye test). The fix takes a rebuilt sample's side first from a
neighbouring target node whose distance exceeds their separation (a distance field cannot have a surface between
them), and only then from older rules: 0 wrong-way flips in the same test; engine tests 13 of 13.

### B.11 The persistent field, kept small

**Evidence state:** proposed. **Date:** first stated 11-12 about 16:50; scoped 11-12 about 19:50.

The larger idea (state such as temperature or moisture held in the environment and advanced when its dependencies
require it) is described in Part 2, section 3. In contact it appears only as the sensor of B.8, and its
scope is recorded as decided: the form is left open (no commitment to an air mesh, nor to static or moving
fields), and it is not to be expanded further for now, because the base has to be built before anything else is
possible.

### B.12 Summary of states

| mechanism | state | where it failed or is open |
|---|---|---|
| Parts, bands, entries, pair settings | measured on named cases | whole-body memory is an estimate; pose-time sheets not measured |
| Solid by default, designed cavities | measured on named cases | a few sub-microlitre specks remain; no general pocket guard |
| IPC-on-fields solver | measured on named cases; baseline only, not public | patent reading B.5; stalls at the elbow |
| Skin-on-skin rows | measured on named cases; not landed | own mark not met (lattice fold) |
| R1 constraint solver | measured on named cases; prototype | times out on broad soft-on-soft contact; pending continuation |
| Magnetic-field contact, Form N | measured on named cases | reversal on soft-on-rigid; cheek scene did not converge; patent flags |
| Magnetic-field contact, Form E | measured on named cases | fails friction; 2.4-11 times slower than the baseline on the scenes in the table |
| Environment-field sensor | proposed | not built |
| Elbow study | measured on named cases | cause not established |
| Seam guard, publish gate, safe sign | measured on named cases | see the store section |
| Persistent field | proposed | scope deliberately small |

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

