[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 10. Patent readings carried over, and what is held out of the code

**Read this first.** By decision, this release publishes flags only: it names each patent our review
flagged and says what was done about it, and it publishes no conclusion that our code does or does not meet any
claim. These are our own readings of claim text, made by non-lawyers. No patent attorney has been consulted. A search hit is a
screening flag, not proof; writing our own code or
measuring better performance is not protection; and a status such as "expired - fee related" is a dated note, not
a permanent fact. The searches were not exhaustive. Nothing here is a clearance, and nothing here is legal advice.

How a reading is recorded in this project (adopted 2026-11-12): each proposed restriction
names the claim element that causes the concern, and each design-around names the element that is absent.

## 10.1 Flags on code as it stands: held out of any public code until redesigned

| Our mechanism | Patent | What the patent concerns (as we read it) | State on 2026-11-12 |
|---|---|---|---|
| Contact solver, first generation: an IPC-style barrier energy minimised by Newton steps on the lattice (`engine/contact`) | NVIDIA US 12,216,969 | a primal energy minimisation of the contact problem with the preconditioned Newton-type update the specification describes | Flagged in our review; held out of the code pending redesign. Direction of the design-around: a constraint-based formulation that minimises no energy (Part 6, B.6), with its weak point stated there; a pending continuation is to be re-read at grant. |
| Look-dev light cache: near hits lit exactly, hits beyond 10 cm read a cached outgoing value | NVIDIA US 11,790,594, claims 16-18 | cached outgoing irradiance used according to the distance between locations (claim 18: a distance threshold) | **Redesigned 2026-11-12, 10:22 PT**: every gather hit is lit by one rule, no outgoing cache is read, no near/far switch by distance. Two earlier unwired prototypes that had the switch are held out. |
| Capsule primitive in the edit program, and the Grab brush's choice of which bricks to admit | MERL US 8,265,909 (to 2031) | distance of a shape swept along a path found through an optimal path parameter | Flagged in our review; held out of the code pending redesign. No design change chosen yet. |
| The card tracer's tile grid that skips empty tiles, and the band store | Movidius US 11,680,803 / US 11,222,459 (to about 2037) | per-tile empty / occupied entries over finer entries, with empty entries skipped | Flagged in our review; held out of the code pending redesign. Direction of the design-around: skip by stored distance rather than by an empty tag; not landed yet. |
| Elastic Grab (a prototype brush using a lattice solve of linear elasticity) | Pixar US 10,586,401 (to about 2038) | regularised solutions of linear elasticity with a specified Poisson ratio (the closed-form Kelvinlets are only dependent claims) | Flagged in our review. An earlier internal note that treated it as unaffected was withdrawn. The brush was never merged and will not ship. Direction of the design-around: a form-aware Grab with no elasticity equation (Part 5). |

Because the edit program, the tracer and the band store are central, the tool's code is not in this release.

## 10.2 Studied and set aside

| Method | Patent | What we do instead |
|---|---|---|
| Geodesic voxel binding for skinning weights | Autodesk US 9,836,879 B2, US 10,062,203 | volumetric bone heat solved on the lattice (Baran and Popović 2007), Part 7; our own geodesic bench module is study-only |
| Regularised Kelvinlet brushes | Pixar US 10,586,401 | not used; see 10.1 |
| Exact cubic in-voxel ray hit for distance grids | NVIDIA US 11,823,319 | marching with bracketing and bisection; one record names the elements as absent and another marks the reading as not verified either way, so it is treated as open |
| Cubic turning-point shadow rays; voxel-local hit storage | NVIDIA US 11,887,245; US 12,175,588 | not used; hits are kept in object or world space |
| Feedback photon splatting | NVIDIA US 12,315,070 | not used |

## 10.3 Read, and on our reading an element is absent

| Our mechanism | Patent | Absent element (our reading) |
|---|---|---|
| One tricubic gradient for normals in the tracer | NVIDIA US 12,254,556 | per-voxel stored normals and their interpolation (one record marks this reading as still to be re-checked for the review renderer's own normal path) |
| Hair fibre scattering after Chiang et al. 2016 | Pixar US 9,905,045 | fitted basis distributions sampled in closed form |
| Hair lobe sampling (logistic) | DreamWorks US 9,019,298 | Cauchy-based importance sampling |
| Per-lobe hair controls | Disney US 8,674,988 | recorded as "expired - fee related" on 2026-11-11; a dated status to re-check before any code release |

Guard rails kept in the hair code because of these readings: never fit a mixture of logistic, skew-logistic,
monomial or Cauchy distributions to a measured lobe and sample it in closed form (tabulate and sample by numeric
CDF instead), and never sample hair lobes by a Cauchy distribution.

On the records found, these published methods are **used** in what is built: sphere tracing (Hart 1996), dual
quaternion blending, bone heat, and the hair models of Marschner et al. 2003 and Chiang et al. 2016. These were
**studied, benched or run as probes only** and are not in the tool: bounded biharmonic weights (proposed, never
implemented), per-brick spherical-harmonic sky transfer and light tracing for caustics (probe results), photon
mapping, and non-linear sphere tracing (Seyb et al. 2019; benched only).

The position taken (2026-11-11 and 11-12): using a patented method is a legal risk; a patented method may be studied
and learned from, and then improved on; the work learns from existing research and builds its own mechanisms.

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

