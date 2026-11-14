# What we are building and researching

> **Research disclosure.** This is a dated public disclosure of our own research and concepts, published while the
> work is under way. Each method was developed with checks for conflicting patents and rights; where a check found a
> conflict, we changed the method ([the methods we changed](PATENT_STORY.md)). This is not a claim of freedom to operate
> and not legal advice.

This page is a dated record, as of 2026-11-14, of what Field Studio is meant to become: the tools, the interface, the
systems underneath and the outcomes intended for them, and the research behind each. Every item carries one
status. Where an idea is ours, it is described specifically enough to be recognised later, because part of this page's
purpose is to put our planned work on the public record with a date. Nothing here is a promise of a date or a
feature; plans change, and the next release will say what changed.

| Status | Meaning |
|---|---|
| **current** | being built, or researched as an active line of work (some of it paused by decision: tools first) |
| **superseded** | replaced; the row says by what |
| **ruled out** | decided against; the row says why (a patent flagged in our review, a licence, a measured failure, or a design decision) |
| **not started** | wanted and recorded, with nothing built yet |

The measured state of everything already built is in the [status review](STATUS_2026-11-14.md). The methods we
changed because of patents, and how they measured, are in [PATENT_STORY.md](PATENT_STORY.md).

## The outcome

Life-like realism at every zoom level and on every part of a form, aligned with its references, in a tool a person
uses to finish the work by hand. A look comes from building the structure that produces it (form, volume, translucent
layers, the optical medium, motion), so it holds when light, view, zoom or motion change; nothing is painted on to
imitate it. Every form is a field: no triangles, quads or tetrahedra anywhere in the model, the tools, the renderer or
the tests (a rule since 2026-11-08). The tool itself, with the operating system, should fit 16 GB of RAM, an
8 GB graphics card and a modest processor. In use: no stall over 100 ms, strokes at 30 frames a second or more, the
interactive view at 60 (120 later), and nothing changing after a release that the user did not see.

## Design philosophy

These are interaction commitments that guide implementation and testing. They are not claims that every difficult
geometric case has already been solved.

- **Begin with the action the person intends.** The interaction design begins with recognizable actions: draw a
  groove, raise a surface, flatten an area, pull out a form, place an object against another, or join two touching
  pieces. Each action needs a clear relationship between the gesture, the affected region and the resulting geometry.
  Radius describes the working extent; Hardness controls the profile within it; Smoothing controls the transition
  toward the surrounding surface; Strength describes the amount or rate of the tool's action, with units that suit
  that action (Shape strength scales displacement, a held tool's strength is a rate over time, Pencil's depth or
  height sets the extent of a line pass). Consistency means predictable meaning, not identical mathematics for
  different actions.
- **Make boundaries and consequences visible.** An edit needs to communicate which surfaces it will affect, which
  direction they will move, and what will remain protected. Near-side, whole-part and through reach express different
  scopes; protection masks, object selection and contact behavior answer other questions and stay separate controls.
  Touching, joining, grouping and deforming are distinct operations: two objects can be placed together without
  fusing, and a thin part moved without dragging unrelated geometry behind it.
- **One shared representation, each tool its own rule.** The tools operate on a shared field representation. Common
  infrastructure manages input interpretation, affected regions, field evaluation and updates, history and visual
  feedback; each tool provides its own geometric rule instead of its own modeling pipeline.
- **Bring form and material into the same workflow.** Material assignment is intended to connect an object's
  appearance, procedural variation and relevant physical properties: a stone block can have a material identity, a
  finish, variations in its composition and geometric weathering while remaining the same editable object. Each
  system consumes the properties it supports; storing density or friction does not by itself make an object part of
  an active simulation.
- **Develop through real creative work.** Establish a usable baseline, make something, observe the specific friction
  or failure, improve it, and return to the same task: from a single shaped stone block to varied copies and a wall,
  or from a primitive to a recognizable head. Numerical checks, regression cases and resource measurements support the
  question of whether a person can perform the intended action, understand the result, recover from a mistake and
  continue; direct use supplies evidence about control and predictability that isolated tests cannot. Design intent,
  implemented capability, measured behavior and user acceptance remain distinct.

---

## 1. The toolset

Designed from definitions written on 2026-11-13 and approved on 2026-11-14. The reason for the change: the earlier
tools were derived from a mesh-based system translated over, and they did not behave as intended.

| Item | Status | What it is |
|---|---|---|
| Shared controls on every tool: Radius, Strength, Hardness (softness), Smoothing | **current** | the same names and meaning on every tool; each tool adds only its own settings. Strength is a rate over time for tools that build while held. Each tool names its primary three in a fixed order, changed by keys or drags with a live cue on the view |
| Follow or flat, per tool | **current** | each tool either follows the curve of the surface or faces flat, and says which; Swell has a toggle, because different regions need it |
| Reach and Stop at contact | **current** | Reach: Near side (back and edge surfaces untouched), Whole part (both faces of a thin part move together, thickness kept), Through; Stop at contact on any tool that moves a surface. Replaces a "protect surfaces behind" check box |
| Shape library with Hollow / Shell | **current** | primitives, base clays and saved objects; Hollow is a property of a shape (an inward wall of exact thickness), live until Apply, with Re-hollow after |
| Duplicate, Mirror, Array, with Vary by seed | **current** | arrays along a line, around an axis, along a path, scattered on a surface; each copy of a library material gets its own form and look |
| Transform of objects, masses and regions; Bend, Twist, Taper, Skew | **current** | one warp engine (moves as smooth flows, forward and inverse) serves Transform, Shape, Place and later the rig |
| Shape (earlier named Stretch) | **current** | grabs every point visible to a flat lens that faces the surface pressed, and pulls; Hardness sets how sharp a point it pulls to (soft: a domed peak; hard: a pin-point); also pushes in. Not yet in the window |
| Draw out (Snake Hook) | **current** | pulls a long form along the hand's path, keeping its girth (a finger, a horn, a tail) |
| Place on surface | **current** | a new shape lands on the surface under the cursor; physics settles it |
| Mask / Protect | **current** | paint, protect a whole part, mask by thickness or distance, box, lasso, grow, shrink, invert |
| Pencil | **current** | Design intent: a surface-following precision tool for carving grooves and building ridges. Its tip defines the local shape of the mark, with controls for width, depth or height, and the transition into the surrounding surface; the intended interaction is direct: geometry changes beneath the tip as the cursor moves. Each press-and-drag is one controlled pass: retracing the same path within that pass does not accumulate additional depth or height, and releasing and beginning another pass establishes a new surface reference, so successive passes deepen a groove or build a ridge deliberately. Fine lines, creases and broader cuts come from the same tool rather than from separate carving and creasing tools. Tips point, round and square (a blunt charcoal-like block planned); taper as a setting; constant depth with a mouse, pressure later; painting on the colour layer is a planned extension. At this release (`3389050f`) the card path is built and measured (a 0.5 mm carve reads 0.498-0.502 mm with a round tip), Pencil is not yet the window's line tool, and repeat-pass placement and non-accumulation with edge blending are still being corrected and validated |
| Pencil's progressive on-screen update | **not started** | the tracer draws the pending path samples exactly as they will be written, so the on-screen update equals the result |
| Swell / Settle (one tool) | **current** | a dome under the cursor that grows while held, still or dragged, never more than a sphere's dome for the radius; Settle is the other side; with Stop at contact, a point that touches stops while the rest goes on |
| Cut | **current** | a line cut measured from the first surface (long enough and it makes two objects), a shape cut, Cut away (the cut volume becomes its own object, held like a container: delete it, move it, or put it back), and cut with an object |
| Mirrored Cut away, Shrink / Grow by a distance, exact put-back | **not started** | for the eye workflow (section 4): both eye sockets cut at once as a mirrored pair, each piece shrunk by a fraction of a millimetre, then returned exactly where it was cut from |
| Merge and Group | **current** | touching objects become one seamless object, with a "blend at contact" setting; objects that don't touch are refused and offered Group, with Ungroup; unmerged objects keep their own identity and properties (wood set in a cement wall, plumbing, wiring) |
| Connect | **current** | pick two objects and a profile (or draw it), and a connector is built between them in that shape: three objects, to merge or keep apart, each with its own properties. Straight connectors in the window |
| Connect along a curved path | **ruled out** for now | a patent flagged in our review; held out until a redesign (a straight connector carried by a flow, or discrete stations of the profile, each evaluated at its own placement) is built |
| Smooth (new) | **current** | a height field over the near sheet only; meets the held-stroke speed bar, not yet the drag bar |
| Smooth by a ring rule | **superseded** | by the new Smooth: in the hands-on test it ran 2.4 to 37 times slower than real time and moved the form 5.8 mm |
| Flatten | **current** | raises and lowers everything within the radius onto one plane through the cursor's point, oriented by the surface under it; the plane always follows the cursor; strength is a rate over time |
| Blend | **current** | rounds an edge or a crease to a radius with flat faces untouched (CPU reference only) |
| Sharpen, Restore | **not started** | Sharpen crisps an edge or crease (a modifier on Blend or Smooth); Restore paints back toward a saved snapshot locally, taking back part of a pass without undoing the rest |
| Form-aware Grab | **not started** | a Grab whose reach follows the form's own interior from the press point, with no elasticity and no material constants |
| Elastic Grab | **ruled out** | a patent flagged in our review; an elastic Grab is not pursued (decided 2026-11-12; a form-aware Grab is planned instead) |
| A line built as one continuous swept field | **superseded** | by the current Pencil; a patent flagged in our review, and the old tool cut across thin gaps and ran minutes of work after each release |
| The brush set derived from mesh sculpting (Draw, Clay, Inflate, Pinch, Crease and others) | **superseded** | as the basis of the toolset, by the design above; the older brushes stay available with the shared controls |
| Automatic finer detail | **not started** | finer lattice where the work happens, with no setting needed (0.0625 mm per part fits the window's budget on paper; 0.01 mm needs engine changes) |
| A depth handle for Whole part | **not started** | only if needed: a grip on the far-face outline that sets how deep a move reaches through a thick wall, in mm |
| The material decides whether its inner structure follows a move | **not started** | flesh and clay deform their inner layers; stone, wood and brick do not bend their grain or courses, so a push reads as carving or adding |
| Sweep and repeat along a stroke | **not started** | a profile carried along a drawn curve (braids, ropes, cables, piping and seams, welds) or a chosen object repeated along it (beads, buttons, studs, scales), with spacing, scale and twist following the stroke; to be built as discrete placements, not a continuous sweep |
| Placed generator variants | **not started** | each click-and-drag makes a new variant from a seeded generator of its kind (a clay pot, a bell, a feather, a shell) at the dragged size |
| Scatter regions | **not started** | paint a region, pick a kind (gravel, leaf litter, pine needles, grass), and a generator fills it with settled instances, using the hair engine's area-to-instances mechanism |
| Measure and grids | **current** | units (mm to inch), tape, girth along the surface, as a tape measure sits on it, kept dimensions, mass from the material; floor and view grids (a surface-contour grid is not started) |
| An Architect toolset | **not started** | a future expansion: walls drawn along a path with thickness and height, openings by type, arches, stairs, floors, repeated courses, structural checks from the materials |
| A mesh import tool | **not started** | later, for libraries: brings outside assets in as fields, with zero mesh inside the studio |

## 2. The interface

| Item | Status | What it is |
|---|---|---|
| Four categories, bold and always visible | **current** | Add Objects, Manipulate, Building Tools, Corrective (replacing the earlier category names Create matter, Move space, Add or remove matter) |
| Three panels behind toggles | **current** | Materials & Generators, Animation, Look / Render; each can pin, pop out, and dock left, right, top or bottom |
| Tool strip above the view, view bar, status bar | **current** | fixed slots so settings never hide below a scroll; nothing clipped at 1366 pixels wide |
| Keys and help on screen | **current** | quick keys and a keyboard map on the top bar; eight short menus in place of 29; 14 px text and greys at 7:1 contrast or better |
| A panel that fails to build says so | **current** | in the window, not only in a log |
| Views | **current** | Clay, Look, Silhouette; an X-ray slider and per-object opacity composited front to back; Section with the cut face; Matcap, Curvature, Normals and cavity |
| Select inside | **current** | click again to go deeper, A+click to come back; a covered object's outline drawn dashed through what covers it |
| Reference images on one centre axis | **current** | front and side images placed at the same alignment, for blocking out to references |
| Look / Render panel | **current** | designed and partly built: render always on; live quality Fast / Balanced / Full; light sets; lens, orthographic front and side views, depth of field; converged stills at size (a 360-degree turntable is not started) |
| Level of realism as a setting | **not started** | from lifelike through stylised to flat-coloured, per scene and per material, hair included |
| Animation panel | **not started** | skeleton (build from the form, fit to references, edit joints); pose by grabbing a finger, hand or arm (turn at a joint, or reach with the hand leading); mirror a pose; pin the feet; Stop at contact; saved poses; a timeline; physics (gravity, simulate, settle, reset); learned motion |
| A Windows launcher and installer | **not started** | open the tool without a script; last, by decision |
| A separate real-time web viewer for hair and body | **superseded** | by Field Studio's own window (archived 2026-11-08 under the mesh-free rule; its brush, comb, hand and fan ideas return in Field Studio) |

## 3. The field store and the renderer

| Item | Status | What it is |
|---|---|---|
| Signed distance fields on a sparse, multi-level brick lattice | **current** | finest level 0.25 mm; a narrow band of samples around each surface; every command logged, undoable and replayable to the same state hash |
| A truncated distance band | **current** | keeps stored values true distances under editing, so strokes never pile into terraces, walls or fragments |
| The tile index of addresses | **current** | per tile and level, the address of its brick table or -1 where nothing is stored; no occupancy, inside / outside or sign tags |
| A store with inside / outside tags per tile and brick | **ruled out** | a patent flagged in our review; it was never used by the app |
| The next store: one hash over a pool of pages | **not started** in the app | edits change only band values; undo copies a brick before its first change; coarse levels edited by the stroke itself, never built by merging fine bricks; nothing paged in on a miss. A prototype measured a brush dab at about 0.3 ms whether one object or 66 were loaded |
| Compact 8-bit encodings as a display cache | **ruled out** | measured: half the memory, but each pass 1.58 to 1.76 times slower against a mark of 1.05 |
| A form kept as its formula | **current** | an analytic shape tree plus the sculpted residual; the bricks are a cache that can be rebuilt |
| The tracer | **current** | image-order sphere tracing on the card, with a CPU tracer as the reference and fallback; a launch guard (no launch past 1 s, no whole-frame fallback); only changed screen regions re-traced after an edit |
| First hits at thin grazing features | **not started** | a smooth tricubic read with analytic normals, so thin rims (lid margins, lashes, a lip line) are found; today 0.25-6 % of rays there miss the first crossing |
| A live, full-detail view of a whole human at every zoom | **current** | pores, fine hair and the iris in the live window within the 8 GB card; not met today |
| A compiled core for a shipped runtime | **not started** | C++ or Rust with a portable GPU backend, once the algorithms settle; today's hot loops are already compiled |
| Light on older graphics cards without ray-tracing cores | **not started** | the aim is high-fidelity light (bounces, mirror chains) on a GTX 980-class card; today's build needs CUDA 13, which no longer supports that generation, so a portable backend is needed |

## 4. Materials and generators

| Item | Status | What it is |
|---|---|---|
| Material profiles with cited properties | **current** | density, friction, stiffness, roughness by finish, measured colour (from public spectral libraries and published measurements), each value with its source; stone, brick, wood, glass, metal, fabric, ground, water and human tissues |
| Structure built in the field, not painted | **current** | stone relief, chips that break the arris and show a paler body, cracks with depth, iron staining in the volume, weathering as a state, deposits by exposure, granite as crystals by mineral, bedding, tooled finishes |
| "This is ..." | **current** | one label binds form, look and physics ("Brick: sandstone", "eye", "cotton shirt"), then local refinement ("worn a little over here") |
| Generators as tools a person opens | **current** | pick a type and a region, set its characteristics, click, place, refine; seeded, the same seed giving the same result; realism is the floor, so any setting gives a realistic result. The iris, skin and hair engines exist; their panels in the window are not built |
| One generator per material, with sub-generators as windows | **not started** | the first: an eyeball generator (sclera, cornea dome, limbus, iris with crypts and fibres, pupil, lens) fitted to the selected form, with the existing seeded iris generator as a window inside it; a pair shares settings with slight variation |
| The eye workflow | **not started** | rough a head; a mirrored Cut away with a sphere takes both sockets; shrink each piece by about 0.3 mm; choose "eyeball" and run the generator; group and name the pair by side; drag them back into place |
| The skin generator by region | **current** | improve and extend the existing skin generator (relief, follicles, buried structure, strain): age, sex, sun exposure, freckles, pore density, calluses by occupation; fine detail carried by deformation under compression |
| A new, separate skin-detail system | **ruled out** | decided 2026-11-13: improve the existing generator only |
| Folds from a material-first layered skin | **current** (research) | the first test (2026-11-13) held volume to 0.009 % with no drift, but the folds came out as a regular corrugation like tooled leather, not skin: neither a yes nor a no by its marks |
| Painting on generator output | **not started** | edit colour, shading and fine detail on top of a generated base, by material, without breaking its tunable responses |
| Reactive states | **not started** | conditional responses by region and skin type: heat, cold, goosebumps, sweat, pressure blanching, flushing; for the eyes, pupil, redness, tears and lid aperture |
| Mouth library | **current** | teeth, gums, tongue and jaw as fields from cited measured dimensions |
| Materials realism | **current** | the stone is still cooler and flatter than its reference photographs; translucency, grainy chip edges and granite depth are next |
| Structure generators for wood, metal, glass and fabric | **not started** | their profiles exist; their built structure does not |
| Blender-era studios for the iris, hair and skin | **superseded** | by field-native generators (2026-11-05 onward); the Blender versions are kept as records |

## 5. Contact, motion and physics

| Item | Status | What it is |
|---|---|---|
| One contact engine for every part | **current** (paused) | parts touch without fusing and never pass through one another; behaviour from each material's own properties, never a rule per region |
| The hard-stop contact law | **current** (paused) | measurement only until surfaces touch, then whatever force touching needs; never a push at a distance, never a pull. Measured exact at a closing elbow crease, at 1.4 to 3.4 times a barrier law's iterations |
| A barrier-based contact solver | **superseded** | by a constraint form with explicit multipliers and the hard-stop law; a patent flagged in our review; it stays only as a comparison baseline |
| A penalty contact law | **ruled out** | measured: skin crossed in every drag frame and was pushed while apart |
| An augmented Lagrangian on the card | **ruled out** | measured: 574 iterations against 317, with force left on contacts already apart |
| A table of stored solves as the live solve's start | **ruled out** for release unless decided otherwise | measured 3.5 times fewer iterations; patents flagged in our review |
| States computed ahead (a lookahead) | **not started** | the states a body can reach next are deduced from its constraints (environment, physics, anatomy), computed fresh from the present, and applied when their trigger fires; nothing stored persistently |
| Key points carry the movement | **current** (paused) | a few key points (the bones) carry a region's movement through its structure; creases, contact and local deformation are solved downstream. The first test at the elbow measured No as built |
| Fast motion without missed contact | **current** | conservative advancement on the fields: none missed at 0.5-5 m/s |
| Physics in the window | **not started** | gravity, settling, and rigid-on-rigid contact for stacking (tier 3 below) |
| Water | **current** (paused) | a field-native free-surface solver and a surface-liquids engine (thin films of water, sweat and tears; capillary rise, wicking, transfer on contact); not in the window |
| Water shaped by its surroundings | **not started** | flow shaped by rocks, steps, falls and beds, with sediment |
| Granular media | **not started** | sand, soil, mud and gravel by a mesh-free material point method reading the profiles: angle of repose, footprint depth under load, mud flow |
| Fracture | **not started** | material on material: an axe head into a log, fabric tearing |
| Contact from imperfect surfaces | **not started** | pores, texture and moisture as the variation in contact |
| Soft-tissue secondary motion, breathing, a visible pulse | **not started** | after the rig |
| Solid by default | **current** | objects are solid; shells and cavities are designed; interiors are labelled parts |

## 6. The skeleton, the rig and learned motion

| Item | Status | What it is |
|---|---|---|
| A correct skeleton from measured references | **not started** (designed, approved 2026-11-14) | bones as rigid field parts with true lengths, joint centres and axes; joints applied from the body outward, with ranges from anatomical zero and couplings built in (the shoulder blade's rhythm, forearm rotation, the knee's screw-home, the jaw's hinge then glide); ligaments and capsules as limits |
| Flesh that follows the bones | **not started** | through the same warp engine as the sculpt tools, with influence diffused through the body's material (never across a gap), so nobody paints weights |
| Two speeds | **not started** | the warp alone as a preview while dragging, in milliseconds; the tissue law as a settle on release and at render time |
| Muscles, one at a time | **not started** | only where a motion shows the simulation can't produce what a real body does (a flexed biceps, the shoulder girdle overhead, a clenched jaw) |
| Proving poses | **not started** | an elbow bent to about 150 degrees with the forearm on the biceps, a fist, a hand to the cheek, an open mouth, a blink; then an arm overhead |
| A first animation | **current** (paused) | a short clip of the human test model's head: hair moving in air and one slow, natural blink, every frame solved and rendered; 120 frames a second, the whole clip rendered in about an hour |
| Export of the skeleton | **not started** | MJCF first (it carries field collision shapes without meshes), USD Physics second, URDF last |
| Learned motion | **not started** | one clip from free motion data (CMU, 100STYLE, ACCAD, MoCapAct), retargeted with GMR and imitated by a plain tracking learner (MimicKit on Newton or MuJoCo Warp) on this card |
| Weight painting and baked correctives | **ruled out** | a design decision |
| Weights from shortest paths through voxels | **ruled out** | a patent flagged in our review; heat diffusion measured 370 times less folding |
| Centres-of-rotation skinning; mesh-based automatic rigging | **ruled out** | patents flagged in our review, and the mesh-free rule |
| SMPL, SMPL-X, AMASS and data built on them; LAFAN1; Mixamo content for training; Isaac Gym; Isaac Sim | **ruled out** | licences (non-commercial, no training, evaluation-only, no redistribution); Isaac Sim also needs 16 GB of card memory |
| A body from a plain description | **not started** | stature, proportions and build in words become a posable, parted base model |
| Anatomy from measured data | **not started** | skeleton, muscles, fat and skin as field layers from published measurements, built where motion needs them; no organs, and not a medical reference |

## 7. Hair

| Item | Status | What it is |
|---|---|---|
| Hair as a defined thing | **current** | fibre properties (shape, stiffness, weight, oil, moisture), and kinds by region (scalp, brows, lashes, fine vellus, body hair by region), each with its own properties; types (straight, wavy, curly) from published fibre traits |
| Grow once, then style | **current** | a follicle brush (draw the area, grow it out), then styles as states made by real actions: brushing, a hand through it, products, a ponytail |
| Cutting in sections | **not started** | cut a section to varied lengths: graduation, lift-and-cut layers, point-cut softness |
| Hair motion from each strand's modes | **current** (research) | modes computed once per strand, moved in closed form under the force acting now, with the hard stop inside; about ten to fifteen times cheaper than a full strand simulation with close motion |
| A position-correcting hair integrator | **superseded** | by the modal method; a patent flagged in our review |
| Hair contact | **current** | each point stopped against the body's own field; strands don't pass through each other; hair as one density medium for both light and contact |
| A persistent hair state | **not started** | hair that stays wet and messy after a swim, lies back under a hand, changes as it dries under a brush |
| Hair accessories as real objects | **not started** | ties, scrunchies, clips, pins, a cloth strip, a hat, acting on the hair |
| Fur and grass | **not started** | the same follicle-to-strand mechanism |
| A rest-groom hair rebuild; a second hair attempt | **ruled out** | the first failed its measurements (2026-10-31); the second was judged wrong on review (2026-11-11: a mound shape and ribbon-like clumps) |

## 8. Light and rendering

| Item | Status | What it is |
|---|---|---|
| Light as rules | **current** | each object's own response to light computed once, the light reaching it (visibility, occlusion) live; a re-bake only where a better way or a real requirement says so (the eye's own response: about 150 s once, then about 4 ms a frame) |
| One rule at every gather hit | **current** | the second bounce is still to move to the same rule |
| Sky visibility per surface cell | **current** | in the window since 2026-11-13 |
| A surface light cache for still renders | **current** | off by default |
| Bounces, caustics and mirror chains in the window | **current** | the light lane's aim, owned from the toolset so the user can adjust and test light in the tool |
| A deep-zoom still viewer | **current** | zoom a 4K still down to pore and fibre level from a tile pyramid, deeper tiles rendered in idle card time |
| Hair and lashes reflected in the eye | **current** | fibre by fibre, through the cornea and tear film |
| Denoising | **ruled out** | decided 2026-11-04 |
| Neural upscaling | **not started** | kept open, to be tested against a native render |
| Neural predictive light | **not started** | an idea: learn from materials, anatomy and shadow casters to predict light and shadow under motion (an arm crossing an object) |

## 9. Cloth and garments

| Item | Status | What it is |
|---|---|---|
| Clothing from measurements | **current** | measure the body (bust, waist, hips, shoulders and so on), then make the garment a second field at an ease distance and tension from it, with a fabric profile; a T-shirt settled to rest on a dress form. Not in the window |
| Fabric behaviour | **current** | tight fabric on a limb, weighted fabric, moisture and drying; to be re-run on the mesh-free contact solver |
| Soft tissue under clothing pressure | **not started** | after garments and the contact engine |

## 10. Persistent fields, kept small

The project's long-range interest is a persistent, causal environment in which fields carry geometry, material and
physics. It stays a research line, tested one piece at a time with pass marks declared before each run (2026-11-13).

| Item | Status | What it is |
|---|---|---|
| State advanced only when its dependencies require it | **current** | correct against a fully stepped truth (a wet patch the camera had never seen was right at first sight); failed its cost marks |
| An environment field as a shared frame and broad phase | **current** | measured yes: none missed of 14.3 million reference pair-frames, at 0.2 % of a full search's cost at rest |
| Sensing contact by probes | **current** | measured yes by its marks; the saving came from caching in the object's own frame |
| The sign of the field as the wall | **current** | a surface read from the sign alone: no point may enter where another body's sign says solid, applied locally at the points (built for fine hair against the body); then each region (bodies, the air around them) solves on its own where something changed (not started) |
| A minimum-bit translation layer | **not started** | each calculation, point or tool held in as few bits as possible, with an interpreter; whether the layer speeds or slows the whole is the question |
| Movement through a pervading field | **not started** | the field as the connective frame (relative position, nearness, shared addressing), materials keeping their own solutions inside it |
| Environmental physics in simulation and render | **not started** | temperature, moisture, rain and wind in the same fields, after the base |

## 11. The tier ladder

Acceptance as scenarios: a lane passes each tier first, scripted through the real window from at least three views,
then it is run live by hand with the render on. Approved 2026-11-14.

| Tier | Scenario | Status |
|---|---|---|
| 0 | the studio baseline: a healthy field under editing, a window that answers at once | **current** |
| 1 | a stone brick with a library material | **current** (passed scripted; not yet run by hand) |
| 2 | five varied bricks, one cracked, one rougher | **current** (passed scripted; not yet run by hand) |
| 3 | stack them under physics, then nudge one | **not started** |
| 4 | a hollow sphere cut into a bowl, filled with water, then rain | **not started** |
| 5 | a head block-out from front and profile references | **current** (almost, by hand on 2026-11-14) |
| 6 | a head-and-shoulders sculpt by hand, end to end | **not started** |
| after 5 | the eye workflow (section 4) as its own scenario | **not started** (proposed) |
| later | a stepping-stone forest scene built with the tool | **not started** |
| later | benchmarks against Blender and Unreal through one-way exporters from the fields | **not started**; no mesh dependency may enter the native work |

## 12. Tooling and method

| Item | Status | What it is |
|---|---|---|
| The shared-machine guards | **current** | a GPU queue (one card job at a time, with leases), a memory guard, a launch guard and a clean exit rule, compile caches keyed on every included file |
| The landing bar | **current** | before anything lands, the real window builds every panel and a scripted open-to-close session passes |
| Test harnesses | **current** | the milestone session, a field-health regression, a safe test runner, per-tool CPU references the card versions are compared with; a 1,000-stroke bounded-error test is next |
| Records and lessons | **current** | every block of work recorded the same day with its measured cost; a ranked search over lessons, logs and readmes; an index of every module and record, so new work starts from what exists |
| A marking tool for pointing at changes in Blender files | **superseded** | by the move to fields (archived 2026-11-06) |
| Meshes or vertices anywhere | **ruled out** | a rule since 2026-11-07 and 2026-11-08: no meshes anywhere, at any stage |
| Unreal as the foundation | **ruled out** | the mesh-free rule; Unreal and Blender stay candidates to measure against (section 11) |
| Building features from numeric warps of 2-D outlines | **ruled out** | measured failure (2026-11-02): it could not make a nose or lips |
| A 2-D-to-3-D front end | **not started** | building a form from reference images; earlier image-only builds were poor and the line is dormant |
| Licensed sources whose terms limit commercial use | **ruled out** | except to study; shipped dependencies need licences that fit an open-source release |

---

Mathew Birchard - DCV Systems, LLC - @birchard_mathew

This page is licensed under CC BY 4.0.
