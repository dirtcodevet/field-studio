[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 7. Posing, cloth from measurements, and the mouth library

This part describes three subsystems of Field Studio as they stood when their work sessions were paused on the morning of 12 November 2026. Everything here works on the same representation as the rest of the tool: a form is a signed-distance field sampled on a sparse lattice with several levels (0.25 mm, 0.5 mm, 1 mm and 2 mm cells in the human model's store). There are no triangle, quad or tetrahedral meshes anywhere in these subsystems, including in tests and previews.

**How to read the dates.** Times in this part are internal provenance times, Pacific time, taken from the project's local commit history and work records (short commit ids are given so that the private history can be matched later). They are not the publication date. The publication date is the date of this release.

**Evidence states.** Each mechanism carries exactly one of: *proposed*, *implemented*, *measured on named cases*, *reproduced under a frozen configuration*, *accepted in the workflow*. Nothing in this part reached "accepted in the workflow" for the human model's published form; where a mechanism landed as a menu entry in the tool and passed its window tests, that is said in words.

**Machine.** All numbers come from one Windows PC with 32 GB of RAM and one 8 GB NVIDIA RTX 5050, shared by several work sessions at once, so CPU timings include contention.

**Patents.** Patent readings below are our own reading of the claims, not legal advice, and no clearance is claimed. Legal status, fee payments and disclaimers of the patents named were not verified.

---

## C1. Posing

### C1.1 The layers (adopted 11 Nov 2026, 16:45; design record)

A pose is always the rest form plus a deformation, rebuilt from rest every time. Rest is never written by a pose. The design has four layers:

1. **Skeleton.** Joint angles give each bone a rigid transform.
2. **Skinning map.** A smooth map from rest points to posed points, blended from the bone transforms by weights.
3. **Correctives as a rules map.** Per joint, the tissue around the joint is solved once by a contact-aware soft-body solver at sampled joint angles, stored, and looked up afterwards.
4. **Live contact** where no stored map can know the answer (a hand against a cheek). *Proposed*; not built in this subsystem.

### C1.2 Skeleton: joints and bones as data, drawn as curves (bebc4f23, 11 Nov 16:52)

*Evidence: implemented; measured on named cases (synthetic arm E1).*

A joint has a rest centre `c`, a rest frame `F` (3x3, columns are the joint's anatomical axes) and one to three rotational degrees of freedom with ranges in degrees. The bone a joint moves carries

`T_j = T_parent(j) o Rot(c_j, Q_j)`, with `Q_j = F_j R_j F_j^T` and `R_j` the product of the axis rotations.

Frames are orthonormalised with the primary axis kept exactly. This was added after a real fault: an elbow frame whose second axis was not perpendicular to the flexion axis made `F R F^T` a non-rotation, and the skinning map then disagreed with the bone's own transform by up to 10 cm.

Bones are line segments (capsules in the bench) and joints are points; in the tool they are drawn as curves and rings over the traced image. A pose is saved as angles only (a test skeleton is 769 bytes, a pose 63 bytes).

**Coupled joints** (67a598ba, 12 Nov 01:44): a driven degree of freedom follows a driver through a ratio or a piecewise-linear table. A finger's distal joint follows its middle joint at 0.67; tests K1-K6 pass. A shoulder girdle (sternoclavicular and acromioclavicular joints, with the shoulder joint expressed relative to the shoulder blade) was added to the human template (80a12cca, 12 Nov 02:15), coupled to arm elevation by straight-line fits to the start and end values reported in Ludewig et al., "Motion of the Shoulder Complex During Multiplanar Humeral Elevation", J Bone Joint Surg Am 2009;91(2):378-389 (open access, PMC2657311). Fitted slopes per degree of humerothoracic elevation: scapulothoracic upward rotation 0.31; sternoclavicular posterior rotation 0.25, retraction 0.13, elevation 0.05; acromioclavicular posterior tilt 0.15, upward rotation 0.09. These are estimates: the paper's per-angle tables were not obtained. *The girdle is implemented as data and coupling only; no girdle deformation has been solved.*

Joint ranges in the template are placeholders marked as estimates. The intended sources (published normal ranges of motion) were not logged against the values, so the ranges are **not established**.

**Pose session** (67adc08a, 11 Nov 19:33; *implemented*, its six tests pass on synthetic cases): pick a part by ray-marching the posed field, pulling the hit back to its rest point and taking that point's heaviest bone; drag rotates the part about its own joint, projected on the joint's degrees of freedom and clamped to its range; "reach" moves a grabbed point to a target by damped least squares over the chain's angles with ranges, pins and a prior that keeps other joints near their last values. None of this is wired into the Field Studio window yet.

### C1.3 Finding joints from the field (e1b47134, 11 Nov 17:20)

*Evidence: measured on named cases (the human model's left arm and hand).*

1. Start at a point and direction on a limb. Take the surface samples in a thin slab perpendicular to the current axis, within a radius of the previous section.
2. Centre the section at the midpoint of its extents along two principal axes; turn the axis toward the new centre; record centre, axis and two half-widths.
3. Stop where the section vanishes (a tip) or exceeds 1.5 times the limb's own starting width (it has merged into a larger body).
4. Joint cues are extrema of the smoothed section-size profile: narrowings between two masses (elbow, wrist) and widenings (knuckles).

Results: the forearm line is clean, with narrowings just above the elbow (37 mm section) and at the wrist (25 mm). The four finger lines are clean (71-92 mm). Knuckle widenings are only 0.4-1.3 mm over the finger's taper: the middle finger joint was found on all four fingers; the distal joint on two of four. The walk fails at the shoulder (it enters the torso), and the thumb was not done. A whole-body skeleton from the field is **not built**.

### C1.4 Skinning weights: volumetric bone heat on the lattice (902c121a, 11 Nov 17:17)

*Evidence: measured on named cases (E1; the left arm of the human model).*

**Ingredient credited:** bone heat from Baran and Popovic, "Automatic Rigging and Animation of 3D Characters" (SIGGRAPH 2007, the Pinocchio system). The method was taken from the paper; no code was copied. The original solves heat on a surface; here it is solved in the solid's volume on the lattice.

Steps:

1. Build a boolean "inside" lattice of the rest solid at spacing `g` (1-2 mm in the tests). Inside and outside come from the field's own signed samples; a region with no samples counts as outside only if it connects to a sample measured outside. Nothing is seeded from the bounding box.
2. For each inside voxel compute `d`, the straight-line distance to the nearest bone segment, and `H = c / d^2` (default `c = 1`).
3. For bone `b`, set `p_b = 1` at voxels whose nearest bone is `b` and visible from it in a straight line inside the solid, else 0.
4. Solve `(-Laplacian + H) w_b = H p_b` over inside voxels, with zero flux at the skin. The Laplacian couples inside voxels only, so weight never crosses an air gap (one finger does not pull its neighbour).
5. Drop weights under `1e-3`, renormalise, keep the top 4 per voxel (int16 ids, float32 weights). Far from a joint one weight is then exactly 1, a rigid zone.
6. Look up at a rest point by trilinear interpolation over 8 voxels, top 4, renormalised.

Solver: one sparse LU then a solve per bone up to about 400k voxels (140-160 s on the CPU for 348k voxels); conjugate gradients with a diagonal preconditioner above that (peak 3.5 GB). **Failure recorded:** a sparse LU on 811k voxels grew to 27 GB of RAM and was stopped by hand; the size switch exists because of it. A whole body at 2 mm (estimated 7.5 M voxels) has not been solved.

Measured on E1 at 90 degrees of elbow flexion, against the geodesic weights of C1.9 at about equal rigid share:

| weights | rigid share | folded tissue | volume change |
|---|---|---|---|
| bone heat, c = 1 | 46 % | 0.018 cm3 | -0.18 % |
| geodesic, zone +-1.5 limb radii | 47 % | 6.7 cm3 | -1.8 % |
| bone heat, c = 4 | 69 % | 4.6 cm3 | -0.05 % |

On the model's arm, bone heat leaves only 55 mm of the 256 mm forearm exactly rigid.

Bounded biharmonic weights (Jacobson et al. 2011) are named as a second candidate. *Proposed*; not implemented.

### C1.5 The skinning map and its inverse (bebc4f23, 11 Nov 16:52)

*Evidence: measured on named cases (E1).*

**Ingredient credited:** dual quaternion skinning, Kavan, Collins, Zara and O'Sullivan (2007/2008). Method from the papers, no code copied.

`phi(X) = DQ( sum_k w_k(X) s_k dq_k )` applied to `X`: weights are read at the rest point, each bone's unit dual quaternion is sign-aligned to the dominant one, the sum is normalised and applied. An all-identity pose returns `X` with no arithmetic, so rest is bit-exact.

A field is posed by pull-back: `F_posed(x) = min over preimages X of F_rest(X)`. The inverse is damped Newton on `phi(X) = x`, started from each candidate bone's own rigid inverse; every converged preimage is kept, so a fold (two rest points landing on one posed point) is detected rather than hidden. This CPU "twin" is the reference for every faster route.

Tests T1-T6 on E1 (two analytic capsules with a smooth elbow): rest bit-exact; rigid zones exact to 1e-16 m; inverse converged for 99.95 % of band points to 1e-10 m; 100 flex cycles bit-identical; save and reopen bit-identical. Inverse cost 0.7 ms per point in NumPy.

### C1.6 What failed first: skinning alone

*Evidence: measured on named cases (E1; the model's left arm, case E2; the hand, case E4).*

- **Zone-width sweep on E1:** with a joint zone of +-0.6 limb radii, 76 % of the arm is exactly rigid but 20.9 cm3 folds at the inner elbow at 90 degrees. At +-3 radii, 16 % is rigid, 0.2 cm3 folds, and 1.3 % of volume is lost. No width removes the fold.
- **Interactive cost:** resampling the posed band from rest each frame failed the 33 ms target written before the runs: 70-150 ms per update at UHD on the model's arm. A profile showed 8-16 ms at 10 degrees and 65-99 ms at 60-90 degrees; the cost came from samples in the compressed crease needing Newton starts from every bone.
- **The 90 degree picture was not clean** (11 Nov ~19:50): a dark crease lip and brick-shaped broken patches. The per-sample reference path showed the same, so it was the skinning fold (determinant of the map below zero), not the renderer.
- **A fist at the finest level (E4):** 1.2-3.9 s per update; zero set within tolerance only to half a curl.

Three rendering routes were benched on the same map: tracing curved rays in rest space (after Seyb et al. 2019, "Non-linear sphere tracing for rendering deformed signed distance fields"), resampling the moved band each frame, and a hybrid with rigid segments drawn as transformed volumes over the same rest bricks. Resample and hybrid matched in correctness outside the fold (zero sign errors on E1 at 0-90 degrees). **No route decision was recorded as final.**

### C1.7 Correctives: the joint zone solved once over its angle (ec35a575, 11 Nov 19:57 onward)

*Evidence: measured on named cases (E1 to 60 degrees; the model's left elbow to 55 degrees). Not established past those angles.*

The idea follows pose-space deformation (Lewis, Cordner and Fong 2000), with the corrections solved by physics instead of sculpted.

1. Cut a zone around the joint. For the model's elbow: 120 mm above and 140 mm below the joint, within 70 mm of the limb line, plus a 70 mm sphere at the joint. The sphere was added after the first zone missed tissue at the back of an elbow that is already bent 22 degrees at rest, leaving skin fragments in the picture.
2. Put a two-level displacement lattice on the zone: 4 mm cells near the crease, 8 mm elsewhere (7,254 nodes for the model's elbow).
3. Treat the zone as a soft body in the project's incremental-potential-contact solver (Part 6, section B.3), with layered tissue materials and self-contact on. Bones are capsules: nodes in the proximal bone stay at rest; nodes in the distal bone and at the distal cut follow the joint's rigid transform.
4. Gravity is on. With it off no step converged. The 0 degree state is solved too and corrections are taken relative to it, so rest stays exact.
5. Step the angle by 5 degrees to 60, then 2.5. Each step starts from the last solution plus the skinning map's increment and is solved statically. A step whose start is infeasible or that fails to converge is retried in halves. Each state is saved as it lands.
6. At draw time the correction is blended linearly between solved angles on top of skinning. A posed point is mapped to rest by a lookup in the solved lattice: deformed cell boxes are binned on a grid, an exact box test selects candidates, and Newton inverts the trilinear cell from its centre (and from the tissue centroid in folded cells). Outside the zone the proximal side is unmoved and the distal side is rigid, each taking over 10 mm past the cut.

**History of walls, kept because each changed the method:**

- First sweeps on E1 stalled near 60-75 degrees. Cause found: lattice cells cut by the skin had quadrature points in empty space, and the empty part of the cell was crushed as the crease closed (local volume ratio 0.65 at 40 degrees, 0.07 at 67.5). A local stand-in moved each cut cell's weight onto its in-tissue points and reached 70 degrees. The stand-in was then retired when the solver itself gained cut-cell quadrature (tissue points only; the empty part held by a weak barrier-free stiffness, 0.1 x mu |F - R|^2).
- A self-contact exclusion radius of 20 mm, compared between cell centres, hid 0.54-0.64 mm skin overlaps at 45-50 degrees. The radius is now 0.5 mm, judged at located rest preimages. States solved under the old radius were discarded and re-solved.

**Current numbers.**

- E1, synthetic: converged 0-60 degrees; 62.5 degrees at residual 2.3e-4; 65 degrees fails. (This is the posing work's corrective setup of 11-11 and the early hours of 11-12. The contact study in Part 6, section B.9, runs the same stand-in arm with a different lattice, skin-on-skin rows and solver settings on 11-12, and gets to 88-90 degrees; the two are not the same experiment.)
- The model's elbow: converged 0-55 degrees at residual <= 9e-7; zero contact rows and zero skin-inside-skin at 0.5 mm from 30 to 55 degrees. Judge at 55 degrees: 0 pockets; seams at the zone cuts 2.4 um (distal) and 0 um (proximal); 0.39 % of interior points claimed twice, as 0.2-1.3 mm folds 7-26 mm under the skin.
- At 60 degrees the start is infeasible: skin folds onto skin at the crease apex below cell size. This waits on a skin-on-skin self-contact piece in the solver that had not landed.
- On E1 at 70 degrees (earlier stand-in states): 0 folded pixels in the flexion-plane section with the corrective, against 325 with skinning alone.

**Known defects at 55 degrees, close view:** terraces 1 mm apart in the crease (cause: trilinear kinks of the rest grid mapped through the compressed crease; a tricubic Catmull-Rom rest read is written and CPU-tested but was never rendered); a pit with a nub beside the crease and a small bump on the silhouette, cause unknown. Only elbow flexion is solved; forearm rotation is not. The goal of a clean elbow at 90 degrees was **not reached**.

### C1.8 Drag cost through the corrective (12 Nov 01:58 and ~04:00)

*Evidence: measured on named cases (the model's left arm, 10-55 degrees, clay shading).*

Per frame, median: node update about 4 ms on the CPU; posed-band preparation on the card about 2 ms; resample 4.5-11.8 ms for about 4,000 posed bricks at 1 mm; clay trace 1.0 ms at 1080p and 3.6 ms at UHD. Frame median 16.5 ms at 1080p and 19.1 ms at UHD; 90th percentile 30.8 and 33.3 ms after buffers were made persistent (it was about 60 ms before). The card's zone samples had 0 sign mismatches against the CPU twin over 3,318 points. These timings were taken on states from the retired stand-in; they were not re-timed on the final states. A practical note: card code avoids matrix-multiply library calls because the machine's CUDA install lacked that library.

### C1.9 Geodesic voxel binding: studied, not used, and why

*Evidence for the geodesic weights: measured on named cases, bench and study only.*

A first weight module computed, for each bone, Dijkstra distances through the 26-neighbour graph of inside voxels, with a smoothstep falloff scaled by limb radius (after Dionne and de Lasa 2013, "Geodesic voxel binding for production character meshes"). On 11 Nov at about 16:55 a patent check flagged it, and the author decided it would be studied only. The claims were then read (about 17:45) from the USPTO's public page scans, by eye.

**US 9,836,879 B2, "Mesh skinning technique" (Autodesk).** As we read them, independent claims 1, 10 and 16 each require all of:
(a) a voxelization of a 3-D model with interior voxels;
(b) traversing several paths through interior voxels from a voxel that intersects an influence (a bone) to a voxel that intersects a target vertex of the model;
(c) a distance value per path;
(d) a path through a boundary voxel has its distance increased by a multiplicative weight;
(e) the shortest path determined;
(f) a skinning weight assigned to the target vertex from that shortest distance;
(g) an animation generated from the weight.

*What our default (C1.4) lacks, on our reading:* (b), (c), (d), (e) and (f). No path is traversed and no path distance or shortest path is computed; each bone's weight is the solution of one linear diffusion equation. There is no boundary-voxel penalty and no vertex: weights live on the lattice and are read at rest points. The only distance used is the straight-line distance to the nearest bone. What it shares: a lattice over a solid (part of (a)) and animation (g).

*Our own geodesic bench module* was flagged in our review and is study-only; it is not used in the tool and is held out of the code.

**US 10,062,203 B2, "Voxelization of mesh representations" (Autodesk).** As we read them, the independent claims require voxelizing a mesh representation, choosing a voxel at the grid's exterior boundary and marking it exterior, propagating that marking, and generating a volumetric model. Our inside/outside classification starts from no mesh, and nothing is seeded at the grid border (C1.4 step 1). An earlier version that flooded from the box faces was closer to the claim's flood step, still with no mesh; it was replaced.

Searches on 11 Nov found no patents on bone heat, bounded biharmonic weights, dual quaternion skinning or Seyb's curved tracing. Those searches were not exhaustive. All of the above is our own reading, not legal advice, and no clearance is claimed.

### C1.10 Muscles and bones as field layers (plan, 11 Nov 20:30)

*Evidence: proposed.*

The author left open that any work session may build bones and muscles as field layers, whole or partial, from medical measurements, if posing tests need them (the shoulder girdle and muscles that bulge with flexion come first). The body's outer surface stays the target; the layers would sit under it and be kept consistent by contact. Nothing was built. Muscle bulge is planned as an activation state of a muscle layer inside the corrective solve.

---

## C2. Cloth from measurements

### C2.1 Measurements taken from the field (a794d4b9, 11 Nov 20:06; rules revised to 12 Nov about 05:20)

*Evidence: measured on named cases (the human model; a mannequin assembled from primitives; a dress form).*

A garment needs only a body's signed-distance field (`phi`, gradient). Measures:

- **Girths.** Horizontal section loops of the zero set at each height; the torso loop is the longest closed loop near the midline; the tape is its convex hull. Waist is the narrowest; underbust sits just below the steepest rise above the waist; bust is the fullest above that; hips the fullest below the waist.
- **Bust points.** Front-facing surface only. Apex: the forward-most point after smoothing over 21 mm. A finer tip is accepted within 30 mm if it stands at least 0.3 mm proud.
- **Neck.** Going up from the shoulder row, the first narrowest section before the tape grows 12 % (into the head).
- **Side neck point.** The concave corner of the front outline where the neck meets the shoulder slope, placed at the neck base's mid-depth.
- **Shoulder point.** Outward along the shoulder line from its flattest point, the first place the slope passes -1.
- **Arms.** Their own section loops give an axis (line fit), the armpit height (highest arm loop) and the biceps girth.
- **Taut tape between two surface points.** Start from the straight chord with inside points pushed to the skin; repeatedly move each point to its neighbours' midpoint and push any point that went inside back out. The path lies on convex skin and bridges hollows.

**Failures that shaped these rules.** The first rules put the side neck point 75 mm low and the bust points about 60 mm off at the silhouette, on both the model and the dress form, so a garment was drafted too small and sat too low. A height-window neck rule took the top of the head on a full body. The shared section code returned points hundreds of metres away where the cutting plane was nearly parallel to the surface, producing a "195 m shoulder row"; loops with points outside the body's box are now rejected, and the shared code was fixed separately. The first time the tool ran in the real window it placed the top at the neck; after the fixes two window reruns reproduced the scripted waist and bust girths to within 0.2 mm with the garment in place.

### C2.2 Cloth representation: yarn points on a pattern lattice

*Evidence: implemented; unit tests pass (energy gradient agrees with finite differences to 4e-11; hoop strain of a band 11.89 % against 11.77 % analytic).*

A panel is one pattern piece. Yarn points sit on a regular (u, v) lattice in the flat pattern: u along the course or weft, v along the wale or warp, spacing 3 mm for the swim top and 8 mm for the tee. Points inside the drafted outline are kept and the outermost row is moved onto the outline, so a cut edge follows the drafted curve.

- **Rest state** is always the flat pattern. Placing a panel on a body only gives the solver a start. The difference between the two is the garment's ease or tension.
- **Energies** (from the contact solver's cloth sheet): a two-branch J-curve stretch along each lattice direction (stiffness `k_low` up to an elastic limit, a stiffer branch beyond), shear from the angle between lattice lines, bending from second differences along lattice lines.
- **Contact** uses lattice cells as bilinear patches of four yarn points, against the body's field, with an IPC barrier and friction by material pair. No triangles.
- **Seams and ties** are stiff zero-length springs between yarn points (default 100 times the panel's segment stiffness). Two cut edges are matched by pattern arc fraction, each node tied to a point between two nodes of the other edge. Bound edges are tension-only springs with shortened rest length.
- A lined, two-layer panel is one sheet with stretch, shear and mass doubled and bending times four (an estimate).

### C2.3 Fabric profiles and their sources

- **Swim tricot** (new profile): 193.6 g/m2 and 0.593 mm, the mean of four 78-80 % polyamide swim knits from an open dataset logged as CC BY 4.0. Stretch 121/124 N/m below a 105 % elastic limit, the stiff branch 15 times that, bending and shear: all estimates scaled from a different knit, because the tricots' curves were published only as figures.
- **Cotton single jersey:** KES-FB values from Ramaiah, Radhalakshmi and Nachane (fibre2fashion article; no open licence stated, numbers used as facts): bending 0.0066 gf cm2/cm, shear 0.47 gf/(cm deg), thickness 0.919 mm, 125 g/m2. Skin friction static 0.90, dynamic 0.79 from an open CC BY 4.0 paper on jersey against chest skin. No wale/course split was found in open sources.
- **Construction rules:** FreeSewing patterns "Bee" (triangle top) and "Teagan" (tee), MIT licence, design defaults only; published size charts for tee proportions; trade sources for swimwear negative ease (15 % crosswise, 5 % lengthwise).
- **Reference photographs:** ten photos under the Pexels licence and Wikimedia Commons, each logged with author and licence.

### C2.4 Settle to rest (b84a0de0, 11 Nov 21:08)

*Evidence: implemented; measured on named cases. Reaching true rest on the tee is not established.*

1. Carry each panel from its drafted place onto the body by rays toward the interior, stopping just outside the skin. Lift any point closer than the collision offset out along the field gradient; the barrier needs every gap above zero.
2. **Close seams in the start state:** move tied nodes onto their targets and fade the move over 30-60 mm of pattern. This exists because of a failure: with stiff ties stretched across a start gap, the contact barrier was squeezed to nothing and the collision step bound held every step at zero.
3. Solve quasi-statically in short chunks of a few Newton steps. Each chunk is a new load step with friction re-lagged; state is saved after each chunk.
4. For the tee: sew with gravity off and no friction until every seam gap is under 2 mm, then run final chunks with gravity and friction. Between chunks, yarn points within 20 um of the body are lifted.
5. Stop at rest (two chunks under 2 um of movement, or energy change under 1e-5) or at a cap.

Body tissue is rigid in all cloth runs. Soft tissue yielding under a band is **not done**.

### C2.5 Garments

**Triangle halter top (swim knit).** Each cup is a flat triangle solved from three taut-tape arcs on the body (base along the underbust, two edges to the apex), each shortened by the ease. The cup's top sits at 54 % of the side-neck-to-bust line. The band and neck tie are strips; cup bases are tied to the band and cup tips to the neck tie.

- v1 settled in 464 s on the CPU with seams at 0.05 mm.
- v2 and v3 drifted loose: a 1.35x gather at the base cancelled the 15 % reduction, leaving cross tension near zero.
- v4 ("snug"): about 7 N/m tension across, median contact pressure 245-273 Pa.
- Run through the tool's own menu path on the model it reproduced v4: cup position differences median 1.3-1.6 mm, 95th percentile 4-5 mm, seams 0.16 mm.
- On a fresh mannequin built from primitives, landmarks, draft, settle and save completed, and the Garment panel passed 6 of 6 checks in the real window (one run 854 s at 99 % CPU).

*Evidence: measured on named cases; the garment panel landed in the tool.*

**Crew-neck tee (cotton jersey).** Front and back: chest = bust x (1 + ease), hem from the hips, straight side seams, shoulder seam from the side neck point to just past the shoulder point, scooped armhole, crew neckline. Short sleeves with the cap height solved so the cap curve equals the armhole length. Start: body panels on the torso's hull loops inflated by 15 mm, sleeves as open tubes about the arm axes.

Faults found and fixed in order:
- Armhole seams started 95-217 mm open and crossed. Fix: place the panels' side edges at the arm junction, and close every seam in the start state (max gap then 0-5.7 mm).
- Lumps under the bust: the start shell followed the waist and squeezed a straight tee 7-30 % across. Fix: below the bust the shell is the hull of the waist loop and the bust loop, so the front hangs from the bust.

Version 13 sewed from 5.7 mm to 0.23 mm and ran 1605 s with a 6.25 GB peak (the export). Version 14 has seams at 0.14 mm and no stretch below the bust.

**What is not established.** No tee version is at rest. After its 30 final chunks v14 still moved 2-4 mm per chunk; continued for 200 more chunks it was still moving 0.64 mm per chunk with energy falling. A variant that carries rest lengths gradually from a zero-strain start stalled with yarn points caught in narrow concave creases of the body field (residual 5.6 % of the load). A proposed remedy, colliding at the fabric's real half-thickness of 0.406 mm instead of 0.31 mm, never ran. Folding near the armholes exceeds the body's own curvature by about 26 degrees (front) and 23 degrees (back) in every version; whether that reads as natural drag lines was not judged. The tee is not available through the tool (the call raises "not implemented"). Fine moire on the jersey in renders is an aliasing fault in shading. Full-colour renders of cloth on skin were not made.

### C2.6 The garment as a field (b84a0de0; banded stores d9c866a9, 11 Nov 22:38)

*Evidence: implemented; measured on named cases.*

A settled sheet becomes a solid: `F(x) = distance to the midsurface - thickness / 2`. The midsurface is sampled about 0.15 mm apart on Catmull-Rom patches through each cell's 4x4 yarn neighbourhood (bilinear at panel edges), with normals. Distance is taken to the nearest sample; its along-normal part is exact. The result is written as a sparse store in the body's brick format at 0.25 mm, with bricks only in a band around the cloth, so it is traced, measured and cut like any other object. A midsurface check read -0.27 to -0.29 mm against the expected -0.30. The tee's export is slow and heavy: about 15 minutes, about 6 GB peak, 1.4 GB on disk for four panels.

### C2.7 Dress form from a model (first 11 Nov ~23:35; runner 48ef3234, armhole plates 26e56338, 12 Nov 07:42)

*Evidence: measured on named cases; the menu entry landed in the tool (window tests 5 of 5, failure cases 3 of 3).*

An earlier mannequin sculpted only from measurement points was criticised by the author as likely to cause fitting faults, so the form is now derived from the model's own field:

1. Sample the body at 2 mm; inside/outside from the body's own reader.
2. Remove each arm at a flat plane through the shoulder point and the armpit.
3. Convert to a true distance by a distance transform; smooth with a Gaussian (sigma 6 mm; a sphere of radius 60 mm moves about 0.3 mm).
4. Cut flat at the neck and below the hips with a smooth maximum (3 mm rounded edge); keep one connected body.
5. Write as a store down to 1 mm.

On the human model: 310 s, 4.5 GB peak, 37.94 L. A hollow body or one with no torso is refused with a plain message. **Fit test:** the tool's top settled on the form against the same top on the model: cup median distance 1.4-1.6 mm, centroids within 1.6 mm. That test used an earlier form (cylindrical arm cut, finer levels); the armhole-plate form has not been fit-tested.

---

## C3. The mouth library

### C3.1 What it is

*Evidence: measured on named cases; the library menu landed in the tool. The seating in the human model is unpublished working versions only.*

A set of generators that build teeth, gums with alveolar bone and palate, a mandible, and a tongue as field parts, each with its own material, placed from a library menu into any head. The body is solid by default, so the mouth's air (the cavity inside the arch, the opening between the lips, the vestibule between cheeks and teeth) is designed space, and the parts sit in it touching their neighbours without being fused into them. An earlier attempt that stored teeth as material labels inside the single body field drew staircases and wrong colours; that is why they are separate parts. Library version 10 at the pause. Units are millimetres.

### C3.2 Generators (30321345, 11 Nov 20:07, through 4b171721, 12 Nov ~05:20)

- **Teeth.** 14 types. A tooth is a field in its own frame (x mesiodistal, y labiolingual, z along the axis, zero at the labial enamel-cement junction). The crown is a stack of asymmetric superellipse sections along monotone cubic profiles, with the incisal edge or cusps, marginal ridges, fossae and fissures added as structure; one to three roots; an inner dentine surface with three lobes under the incisal edge; a pulp. Smooth maxima round the junctions. The values are approximate distances (correct sign and zero set), re-distanced on a lattice afterwards. Settings: scale, wear, incisal lobes, age, seed. Only the upper central incisor's profiles were fitted to photographs (0.62 mm and 0.36 mm rms on two silhouettes); the other types use profiles drawn by eye with cited sizes.
- **Arch.** A contact curve built from its curvature (one value through the front, easing to a second behind the canine), solved so the built teeth meet the cited inter-canine and inter-molar widths. Each tooth gets inclination, angulation and a height on the occlusal curve. Contacts are closed along the arch from the midline; the lower teeth erupt until they touch. Defaults give proximal gaps 0.00 mm, occlusal gaps +-0.01 mm, overjet and overbite 1.60 mm, both measured from the build rather than imposed. This replaced a version in which the upper central incisors never touched (0.44 mm apart at best), visible as a dark slit in a smile render.
- **Gums, bone, palate.** Alveolar bone as one ridge round the sockets with its crest 1.5-2 mm below the enamel-cement junction; gingiva over it with a free margin about 0.8 mm up the enamel, scalloped margins and papillae from cited values; palatal vault with rugae. Teeth are carved out of both across a 0.2 mm ligament gap. A first version built a wall round each tapering root and left V-shaped slits; that was rebuilt.
- **Mandible.** A body swept along the basal arch with height and thickness profiles, a ramus plate from a 2-D spline outline, condyle and coronoid process, from cited adult sizes. Condyle to incisal point 102 mm (the textbook triangle is about 100). It is a first version, not checked against section images.
- **Tongue.** Free form from cited sizes with a median groove and papillae; its rest form is pressed against palate, teeth and gums so it takes their imprint.

Three-plane sections of the assembled set show 0 overlapping pixels.

### C3.3 Sampling, caching, placement

Each part is sampled on a dense lattice, evaluating fine nodes only near the surface, then converted to a true distance from the zero crossings on lattice edges. The seven-part set is 493 MB, cached by settings. Cold build 199 s (456 s before narrow-band sampling); 11.3 s from cache in the app. Placement puts the upper incisal point 2.5 mm below the meeting line of a head's lips, which are found on the head's own surface. Undo returns the memory on CPU and card (about 1 GB each; verified). While the set is placed, the app holds about 1.8 GB more than before; an attempted fix caused repeated re-uploads per brush stroke and was reverted.

### C3.4 Look

*Evidence: measured on named cases; not judged finished.*

A CPU path tracer carries three wavelengths together through nested media (pulp in dentine in enamel; gum over bone; a wet film). Scattering and absorption per medium are fitted with a plane-parallel Monte Carlo slab model to published colour measurements, including an open CC BY 4.0 paper on dentine and enamel slices (BMC Oral Health, 2021) and enamel scattering from Fried et al., Applied Optics 34:1278 (1995). Whole-tooth yellowness is a per-person setting over the cited in-vivo range (b* 10.1-17; default 12). Gingiva needed two layers to match its reference (0.2 dE, against 7 dE with one layer, which had rendered orange). A fitting error once made dentine bluish and the cervical third lavender; the fit now keeps dentine the more chromatic layer. The incisal edge band is 10.4 L* darker than the middle of the crown, against 12.5 in a living reference photo, but still reads weak by eye. A single tooth still reads uniform inside, and the root has no cementum layer.

### C3.5 Seating in the human model

*Evidence: measured on named cases (two unpublished working versions).*

The parts were written into a copy of the model's store one 24 mm region per process (peaks 2.0-3.1 GB; 68 regions), because one process for six regions had reached 12.1 GB. Checks on the second full build: 0 overlapping nodes in six sections; the model's surface more than 6 mm from every part unchanged (4,260 points, 0 sign flips, 0.0 um). A first build had tooth-bone overlaps up to 0.46 mm at the upper necks, fixed at their cause in the bone generator. Its 66 intermediate store copies used 177 GB of disk before a cap of two copies was added.

**Remaining gap.** The model's sculpted mouth cavity is wider than the dental arch, leaving air between cheeks and teeth: medians 0.3-2.9 mm by region, with a 90th percentile of 10.5 mm at the upper premolars. A prototype that fills the vestibule so that cheek and lip lie on teeth and gums as touching, unfused surfaces filled 12.8 mL on a local 0.3 mm lattice, with no sign change outside the vestibule and gap medians of 0.1-0.3 mm, but left 1.7 mL unfilled. A stricter rule measured a 47.1 mL vestibule, then failed with an out-of-memory error in a distance transform; whether all 47 mL is truly vestibule was not checked. The builder changes for this are untested, and no filled version exists.

**Data and licences.** Tooth dimensions cite a standard dental anatomy textbook's measurement table. Those values were entered from memory and are marked as estimates; they have not been checked against the book. Arch and jaw sizes come from published studies recorded with each value. Incisal lobe prevalence by age is from Alalowi and Al-Jhany (2020). Reference images are 67 Wikimedia Commons files (public domain, CC0, CC BY, CC BY-SA, each logged) and head sections from the U.S. National Library of Medicine's Visible Human Project.

---

## C4. Summary of what is not established

- Elbow correctives past 55 degrees on the human model or 60 degrees on the synthetic arm; any other joint; forearm rotation; fingers with contact; the arm raised overhead.
- Posing inside the Field Studio window, a whole-body skeleton from the field, whole-body weights, sourced joint ranges, and a final choice of rendering route.
- Bounded biharmonic weights, live contact during posing, muscle and bone layers: proposed only.
- A tee at rest; the tee through the tool; soft tissue yielding under cloth; cloth shading; a fit test of the current dress form.
- Cheeks and lips resting on the teeth in the human model; tooth types other than the upper central incisor fitted to measurements; the textbook dimensions verified.
- Legal status of the patents named; completeness of the patent searches.

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

