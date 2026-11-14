# Patents near our methods

> **Research disclosure.** This is a dated public disclosure of our own research and concepts, published while the
> work is under way. Each method was developed with checks for conflicting patents and rights; where a check found a
> conflict, we changed the method ([the methods we changed](PATENT_STORY.md)). This is not a claim of freedom to operate
> and not legal advice.

Field Studio is built from our own mechanisms, but some of them sit near patented methods. While building it we read
the patents we found and compared their claims with what our code does. This page lists those patents and describes
how our mechanisms work, so anyone can see the differences for themselves.

This is an engineering note, written for transparency. It is not legal advice and not a freedom-to-operate opinion,
and we are not lawyers. Patent statuses are as shown on public patent pages on 14 November 2026. Our searches were not
exhaustive. If you know of a patent we should read, please open an issue.

## Shapes, lines and connectors

**MERL US 8,265,909 B2 (active, expected to 2031-02-23) and US 8,010,328 B2 (to 2030-04-01).** These describe the
distance field of a swept volume: the space a shape covers as it moves along a path, such as a milling tool following
a tool path. For each sample point, the shape's best placement along the path is determined, and the distance is read
from the shape's own field at that placement. The second patent applies this to milling with composite adaptively
sampled distance fields.

- *Capsule and torus.* Our capsule is a fixed primitive with a length and a radius. Its distance is the standard
  closed-form formula: the distance from the point to the capsule's axis segment, minus its radius. The torus is the
  same kind of standard closed-form primitive. We list them because a capsule is, geometrically, the volume a sphere
  covers when moved along a segment.
- *Pencil.* Pencil represents its working tip as an implicit geometric function: a pointed conical, a spherical, or a
  square, flat-ended profile. The stroke engine samples the cursor path and, for each path sample, establishes a local
  tip position and orientation relative to the stroke-start surface: the cursor's surface point is projected, in a
  bounded number of steps along the local surface facing, onto the surface as it was at the press, and the local tip
  frame is built from that facing and the path direction. It then evaluates the tip function at affected field nodes.
  Carve mode combines the object field with the negated tip field using a maximum operation; ridge mode combines the
  two fields using a minimum operation; an optional smooth combination controls the junction with the surrounding
  surface. The depth is limited by a configured maximum, the tip size and, in carve mode, half the part's thickness
  under the point. The stroke engine therefore evaluates discrete path samples, each at its own placement relative to
  the stroke-start surface, and nothing searches along the path for a nearest placement. This is a sampled numerical
  implementation of an interactive line tool, not an exact continuous geometric sweep. In this release the card
  applies one path sample per launch (batched path-sample processing is designed, not released), and Pencil is not yet
  the window's line tool. An earlier line tool, which built each stroke as one continuous swept field, was flagged in
  our review and is not part of this release.
- *Connect.* Connect builds a straight connector between two picked points. The chosen outline is extended along that
  line, so each point is measured against the outline in the plane through the point itself. The outline can change
  from one end to the other and twist along the way, and the connector is trimmed exactly by the two objects it joins.
  Curved connector paths were flagged in our review and are held out of this release pending redesign.
- *Brush reach.* When a moving brush is applied, Field Studio chooses which lattice cells to update by comparing the
  distance from each cell's centre to the segment the brush travels in that step with the brush's reach. The value
  only selects cells; it is not stored or used as the distance field of any shape.

**Ansys US 10,083,264 B1 (active, to 2035-10-13).** It describes blending implicit surfaces where they meet: distance
fields decide where to blend, the blend has a constant radius, and the space is split into cells processed in
parallel. Merge joins touching objects with a polynomial smooth minimum whose width is set from both objects' fields:
full width across the contact, tapering to nothing away from it. There is no rolling-ball fillet construction. Field
Studio also has a general smooth-union operator, the same polynomial smooth minimum at a fixed width.

## Storage and drawing

**Movidius US 11,680,803 B2 and US 11,222,459 B2 (active, expected to 2037-08-19).** These describe a volume held at
several levels of detail, where an entry holds bits saying whether each voxel is empty or at least partly occupied,
with entries for finer voxels beneath. Rendering and ray tracing skip voxels marked empty, and one of the patents loads
only some levels into on-board memory.

Field Studio keeps each object as a narrow band of distance samples around its surface, in bricks of 8 x 8 x 8 cells
grouped into 32 mm tiles. For each tile and level, the index holds the address of that tile's table of bricks, or -1
where the level stores no bricks there. Because only the band near the surface is stored, a tile deep inside a solid
and a tile of empty air both read -1: the entry records whether distance samples are stored there, not whether space
is empty or occupied. When a ray reaches a tile with no stored samples at any level, it steps to the tile's far side;
elsewhere it steps by distance, including a coarse distance to the nearest stored brick. An earlier storage layout
that tagged tiles and bricks as inside or outside was never used by the app and is not part of this release.

**MERL US 9,122,270 B2 (active, to 2034-02-02).** It describes cells that each hold both sampled distances and a set
of distance functions, typed and subdivided by both. In Field Studio the sampled lattice holds samples only. Each
object's analytic shapes form one expression tree, evaluated as a whole with bounding-box tests; no cell holds a list
of distance functions.

**MERL US 9,336,624 B2 (active, to 2035-01-11).** It describes rendering a distance field cell by cell: an object-order
pass finds each surface cell's image samples, then rays are cast inside each cell independently, with a test of
whether the ray's entry and exit points are both inside the surface. Field Studio traces each pixel's ray through the
scene in image order, carrying its state from brick to brick and refining the hit between two steps. After an edit it
re-traces only the screen rectangles that the changed regions cover.

**NVIDIA US 11,790,594 B2 (active).** It describes world-space irradiance caches, including choosing which locations
to update by ranking them, and using cached outgoing irradiance for a location chosen by the distance between
locations. The window's sky lighting keeps, for each small surface cell (keyed by position, level and normal
direction), how open that cell is to the sky: visibility, not light leaving a surface. A cell gets rays when the
current image shows it and it is not yet full; no cell is ranked against another. A cell's size follows the pixel
footprint. Each pixel reads its own cell, or its own estimate while the cell is still filling.

**Adobe US 8,379,021 B1 (active).** It describes shadowing a height field by finding the surface point at the greatest
elevation angle toward the light. A material's fine grain relief in Field Studio shadows itself with a fixed-step
clearance test: it samples the relief at six fixed distances toward the light and checks how far the light ray clears
each sample. It does not search for a maximum elevation angle or use horizon maps.

**QuantifiCare US 11,094,139 B2 (active, to 2039-06-24).** It describes drawing a second model with an intensity tied
to the angle between its surface normal and the view direction, blended over the first in a fixed colour. Field
Studio's X-ray view composites each object's own shaded surface front to back at its opacity, with no view-angle rim
term. The selection outline comes from the edges of the object-ID image, in one constant colour, dashed where
something covers it; it uses no surface normals.

## Moving and deforming

**Pixar US 10,586,401 B2 (active).** It describes sculpting brushes built on regularized solutions of linear
elasticity (Kelvinlets). Field Studio's Grab, Shape and other move tools are smooth flows of velocity fields; the field
is read back through the inverse flow. No elasticity equation, Poisson ratio or Kelvinlet is used.

**Disney US 9,483,878 B2 (active, to 2034-02-09).** It describes limiting dragged elements to a displayed offset shell.
Field Studio's moves and Swell can stop at contact, and they stop on the other object's own distance field; no offset
shell is displayed or used as a constraint surface.

**SolidWorks US 7,936,352 B2 (active).** It describes stored compositions of smooth maps applied to a parametric
surface. A Field Studio move is read back through its flow and baked into the sample lattice when the stroke ends;
there is no parametric surface and no long stored chain of maps.

## Hair

**Procter & Gamble US 10,460,054 B2 (active, expected to 2037-09-14).** It describes simulating a change in the
configuration of fibres, such as a head of hair, from a set of mechanical parameters for a fibre type and rendering
images of the result. Its claimed parameter sets include a measure of cohesion, a measure of adhesion and a shear
modulus (in one claim, a shear modulus calculated using the torsional moment of inertia of the fibre's cross-section;
in another, together with Young's and bending moduli).

Field Studio's hair generator settles each guide hair as an inextensible chain of segments clamped at its follicle,
by minimising one energy: bending against the hair's rest curl, gravity, contact with the body, hair-to-hair pressure in
the hair mass, friction, and, when an oil film is set, the oil bridges between touching hairs, which pull the mass
together and add to the load that friction carries. Bending stiffness comes from Young's modulus and the fibre's
radius. The rest curl is read in a frame carried along each hair from its follicle without twisting, so the settle has
no twist: it holds no twist energy, no torsional stiffness and no shear modulus. The hair material profiles list a
published shear modulus as reference data; the hair settle does not read it.

**Pixar US 9,905,045 B1 (active, expected to 2035-05-28) and DreamWorks US 9,019,298 B2 (active, expected to
2033-09-22).** The first describes a hair scattering model built by fitting a set of basis distributions of several
types to a scattering function, with incoming directions importance-sampled from closed-form solutions of those basis
distributions. The second describes importance sampling a hair shading function with a Cauchy distribution.

Field Studio's hair shading follows the published model of Chiang et al. (2016) as implemented in pbrt-v3. Each
scattering lobe is the product of d'Eon et al.'s longitudinal distribution and a trimmed logistic around the lobe's
azimuth, weighted by Fresnel reflection and absorption inside the fibre, with the cuticle's tilt and the published
conversions from roughness to lobe width. Nothing is fitted to a scattering function. For hairs narrower than a pixel,
the shading reads a precomputed table of the same model averaged across the fibre's width. Where directions are
sampled, a lobe is chosen by its weight and the angles come from inverting the d'Eon and trimmed-logistic
distributions; no Cauchy distribution is used.

## What this release leaves out on patent grounds

- The earlier line tool (a continuously swept stroke). The current Pencil, which evaluates discrete path samples
  against the stroke-start surface, is in.
- Connect along a curved path. Straight connectors are in.
- The earlier tagged storage layout, which the app never used.

Every other mechanism named on this page is in the release, with its working described here so the differences are
on record.

---

Mathew Birchard - DCV Systems, LLC - @birchard_mathew. This document is licensed under CC BY 4.0.
