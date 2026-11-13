[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 5. Brushes, the toolkit framework, sculpting into contact, and the shell

This part describes every sculpting mechanism built for Field Studio up to 2026-11-12, how each one changes the
field, what it cost per stroke, where it failed, and what replaced it. Dates are internal adoption times taken from
the project's version history (Pacific time). They record when a mechanism entered the code or the written record;
they are not the publication date of this document, which is the release date stated on the release itself.

Each mechanism carries exactly one evidence state:

- **proposed**: designed and written down, not implemented.
- **implemented**: in code (the live tool or a named working copy), not measured beyond smoke checks.
- **measured on named cases**: run on the cases named beside the numbers.
- **reproduced under a frozen configuration**: the same result obtained again from a fixed code state and inputs.
- **accepted in the workflow**: used by the author as part of the working process.

The most important fact comes first. On 2026-11-12 at about 17:20 the author sculpted with the tool by hand and
stopped the toolkit: "Right now, I cannot even block out a character's head." The scripted head test had been
reaching all its stages, though never its marks (section 12). Much of what follows is therefore a record of mechanisms that work by their own numbers and still did not
give a sculptor a usable tool, plus the redesign that came out of that. **The author's hands-on sculpt test of the
rebuilt toolkit is still to come.** Nothing here should be read as a claim that the toolkit is finished.

### 1. Setting, machine and how to read the numbers

Every object is a signed-distance field F (F < 0 inside): an analytic base tree plus a sculpt layer of values added
on a sparse lattice of 8 x 8 x 8-node bricks at spacing h, stored only near surfaces. Brushes edit the sculpt layer.
Test forms use h = 1 mm (a 240 mm bust sphere); the human model uses h = 0.25 mm.

Notation: c is a dab centre on the surface, n the surface normal there, r the brush radius, s the strength,
w(u) the falloff with u = distance / r (default 1 - smoothstep: 100 / 84 / 50 / 16 / 0 % at u = 0 / 0.25 / 0.5 /
0.75 / 1), A the dab's amount, v a drag vector.

Machine: one Windows PC with 32 GB of RAM and one 8 GB RTX 5050. "The card" below means that GPU; "the CPU path" is
the fallback that runs the same rules in compiled Python (numba), 4 threads. **CPU-load caveat:** the machine was
shared by several working sessions at once, often at 90-100 % CPU. Frame rates swung by 2-3 fps between identical
runs, and some numbers were taken under that load. Where a before/after pair was interleaved in one run it is
reliable; a lone frame rate is an indication only. Card strokes are not bit-reproducible run to run (dab batching
follows frame timing); card-versus-CPU agreement is proven by a full field comparison, not by repeat runs.

### 2. The stroke pipeline

**CPU path** (in the repository's first commit, 2026-11-05; band limit 2026-11-12; measured on named cases).
1. The cursor ray is traced to the first surface. A press off the form takes the nearest surface within the brush
   radius along the ray, else the stroke starts when the drag first reaches the form. A brush press never orbits the
   view (2026-11-11).
2. The surface path is resampled to one dab every spacing x r. A frame that ends between dabs lays a partial dab whose
   share f of a spacing step scales the dab's amount linearly (2026-11-11). Before that fix the share was folded into
   strength, and two half dabs deposited 27 % more than one whole dab, so results followed frame timing.
3. Per dab: the nodes of the dab's ball are evaluated, the brush rule (section 3) gives a change per node, the
   change is multiplied by (1 - mask), and the gates run in order: the sheet gate (section 5), the keep-apart guard
   (section 4), the collider clamp (section 6). The change is written and a before-image kept for undo.
4. At release the touched bricks are re-distanced (|grad F| restored to about 1) by seeding at the surface and
   sweeping an Eikonal solve outward.

**Card path** (2026-11-08; rebuilt 2026-11-11 and 11-12; measured on named cases).
1. All dabs of one frame form a batch. A plan step admits the bricks the batch can change and uploads their rows
   (a brick plus its one-node border, 9^3 values).
2. One kernel computes every dab of the batch per node. Drag dabs of a frame are merged while they are closer than
   0.25 r.
3. The keep-apart guard runs once per batch on the net change; the collider and the sheet labels run on the card.
4. A live re-distance runs after every batch: seeds at the surface nodes, then Godunov Jacobi sweeps, with the
   surface layer (nodes whose 26-neighbourhood changes sign) frozen. The borders are refreshed from their owner
   bricks. The first version (a jump-flood nearest-foot distance that rewrote the surface nodes) left lattice-aligned
   ribs over the cranium and was replaced on 2026-11-11 16:21.
5. At release every changed brick is written back to the CPU store as one undo step. A version that skipped deep
   bricks (2026-11-11 ~15:00) left the CPU store behind the card and produced whole bricks of the wrong sign; it was
   withdrawn the same day at 17:40.

The Python interpreter's thread switch interval is set to 0.5 ms at engine start (2026-11-11 17:11). On the head
scene this alone took a mirrored Smooth stroke from 15.0 to 29.3 fps median, because the render thread had been
waiting for the interpreter lock.

### 3. The brushes: how each changes the field

All in the live tool unless stated. "Resample" means the new value at node x is the old field read at a displaced
point, which moves the surface without leaving a step in F.

| Brush | Rule at node x (w = w(|x - c| / r)) | Adopted | Evidence state |
|---|---|---|---|
| Draw | F'(x) = F(x - n A w): the surface rises A w along the single hit normal | first commit 11-05 | measured on named cases |
| Carve | F'(x) = F(x + n A w); Ctrl+Draw | 11-05 | measured on named cases |
| Inflate | F'(x) = F(x) - A w: a true offset along each point's own normal | 11-05 | measured on named cases |
| Clay | d = signed height of x above the area plane; F' = F + w (min(F, d - A) - F), at most A per dab, so a hollow fills over passes | area plane 11-11 13:30 | measured on named cases |
| Flatten | F' = F + k w (d - F), clamped to +-A per dab; two-sided, toward the area plane | 11-11 13:30 | measured on named cases |
| Smooth | section 3.1 | rebuilt 11-11 13:30 to 21:50 | measured on named cases |
| Crease | t = offset of x from the stroke line within the tangent plane; F'(x) = F(x + k w t) + A w^3: the surface is drawn toward the line and cut in | toward the line 11-11 13:59 | measured on named cases |
| Pinch | as Crease without the cut; factor bounded at 0.4 so the map's Jacobian stays above 0.5. Still tears (85 columns in the scorecard) | bound 11-11 13:30 | measured on named cases |
| Grab | F'(x) = F(x - v w), composed dab after dab. Being replaced (section 8) | 11-05 | measured on named cases |
| Snake Hook | Grab with a falloff flat to 0.8 r and a cosine edge. Being replaced | 11-11 22:59 | measured on named cases |
| Stretch | with a = v / |v| and kw = (|v| / r) w: F'(x) = F(x - a ((x - c) . a) kw / (1 + kw)), a scale along the drag axis about the pressed point. Withdrawn (section 8) | 11-05 | measured on named cases |
| Mask | paints a 0-1 mask per node; every brush's change is multiplied by (1 - mask) | 11-05; on the card 11-11 15:03 | measured on named cases |
| Region | paints a 16-bit region id per node; strokes can skip hidden regions or act in one region only | 11-11 22:27 | measured on named cases |

Amounts: displacement brushes move 0.2 x spacing x s x r per dab, so the same screen stroke gives the same-looking
shape at any radius. Smooth, Flatten, Pinch and Crease's pull use a per-dab factor k = k_max (1 - exp(-k0 s / k_max)),
monotone in strength and saturating, never clamped (2026-11-11 13:59, at the author's direction that every tool's
effect rise smoothly with strength). Measured on the CPU path at s = 0.1 / 0.25 / 0.5 / 1 / 1.5 / 2: Smooth removes
0.14 / 0.30 / 0.44 / 0.50 / 0.53 / 0.53 of a 12 mm bump per pass; Crease depth 0.10 / 0.24 / 0.45 / 0.82 / 1.12 /
1.34 mm. Ctrl inverts every surface brush (measured on all seven; Ctrl+Clay digs 1.44 mm).

**Area plane** (Clay, Flatten): the falloff-weighted mean position and normal of surface samples on a sunflower disc
of radius 0.8 r oriented by the hit normal, each projected onto F by one Newton step.

#### 3.1 Smooth and its fit

The first Smooth averaged F with its six neighbours at 2h whatever the radius: it removed 4 % of a 40 mm bump per
pass, and in the first head test four passes took lumps from 4.49 to 4.23 mm RMS. Rebuilt 2026-11-11:

1. Only nodes within 6h of the surface are written.
2. Each such node projects to the surface and samples two rings of 8 points in its own tangent plane, at 0.3 r and
   0.6 r. Each sample's height along the node's normal comes from one Newton step and is clamped to half its ring
   radius.
3. Opposite samples are paired. A pair's weight falls where a sample lies on a sheet facing away (the "robust2"
   ramp, 2026-11-11 21:50), so a facing surface is never drawn in.
4. A quadric is fitted to the heights. The node moves toward the fitted surface, so a sphere or cylinder at the
   brush's scale gives zero and the form's own curvature stays.
5. A lattice-scale term (the six-neighbour mean at 2h minus F) is added; without it the fit left cell-scale speckle.
6. The change is s w (fit + lattice term), capped at 2h per dab.

Measured (CPU path, one pass, r equal to the bump width W): W 4 / 12 / 40 mm removed 88 / 51 / 52 %, surrounding
form moved 0.002 mm or less. On the card at s = 1: 41-45 % per pass at r 20-60 mm; after the partial-dab fix removed
a frame-timing inflation, 34.5 / 30.4 % at r 20 / 60 and a repeat spread of 0.49 % (was 3.44 %).

Failures on record:
- **Chin bands.** Smooth at r 70, s 1 over the test head's chin left textured bands and pits. Cause: at r 70 the
  outer ring lies 42 mm out, past the chin's turn, so it is no longer a height over the tangent plane. Rings at
  0.2 / 0.4 r were clean but weaken Smooth elsewhere. An adaptive three-ring fit passed CPU invariants and on the
  card traded the pits for a new V-shaped crest across the chin. Not landed. Open.
- **Fixed points.** A form about the brush's own size grows under Smooth (a sphere of R 40 under r 40: +1.27 mm over
  8 passes), from the parabola fit's bias against a real sphere. A circle fit removed the growth and exposed the
  lattice term's own shrink (-1.29 mm). Open.
- **Volume.** Ten passes at r 30, s 0.5 on a 40 mm bump: height -53 %, volume -20.7 %. Smooth shrinks.
- **CPU-path stall.** On the CPU path a 40 mm bump stalls at about 63 % remaining, because Smooth compresses the
  field at the edge of its 6h band and the release re-distance skips it. The card path does not stall.
- The author's own report after sculpting (2026-11-12 18:2x): Smooth "works fairly well", but a smoothed lid bulge
  came back after further strokes, and some surfaces keep small tear or grain marks.

### 4. Shared stroke rules

- **Mirror** (2026-11-11 17:46 to 19:35; measured on named cases). A dab and its mirror partner are both applied.
  Where their balls overlap, each node's weight is scaled by (w^2 + w2^2) / (w + w2)^2, so the pair sums to the
  contraharmonic mean (w^2 + w2^2) / (w + w2): smooth, no doubling on the midline, equal to w where one acts alone.
  Two earlier versions failed: snapping near-midline dabs onto the plane zeroed sideways drags (a widening Grab did
  nothing), and max(w, w2) left a sharp midline seam. As landed: a dab within 0.8 r of the plane is still snapped
  to one centred dab for the mirror-symmetric part of its effect (so a nose stays single), while a Grab or Stretch
  drag's component across the plane is carried by a feathered pair at the original centres; Smooth always uses the
  feathered pair. CPU check: sideways widening 4.47 / 4.46 mm
  on the two sides, 0.0 on the midline; midline Draw 11.44 mm against 11.18 single.
- **Keep-apart guard** (2026-11-10; measured on named cases). A node may change sign only if the inside components
  (26-connected) and outside components (6-connected) of its neighbourhood stay as they were, so a guarded stroke does not join,
  film over or tear surfaces (scope: the guard is off for Grab and Stretch, and Pinch still tears). A refused node keeps its sign at +-0.05 h. It is off for Grab and Stretch by default
  since 2026-11-11, because on large moves its refusals left specks and pockets (6 inside parts and 31 pockets in a
  neck with it on, 1 and 0 with it off).
- **Stroke-end topology check** (2026-11-11 19:35; off by default; evidence state: measured on named cases). Parts, tunnels and cavities are
  counted from node signs before and after a stroke (b0 by 26-connectivity, b2 by 6-connectivity, b1 from the Euler
  characteristic). A pocket the stroke made is filled; a closed opening is reported with "keep it / undo the
  stroke". Detection was measured on the card on a slit and a torus with the guard switched off.
- **Band-only dabs** (card 2026-11-08 / 11-11; CPU 2026-11-12 12:5x; measured on named cases). Nodes farther from
  the surface than 1.6 x the dab's move + 6h keep their values. The CPU version first skipped on both sides and left
  a 28.6 um notch where a growing surface crossed a brick row; the fix is one-sided (a growing dab skips deep inside
  only, a cutting dab deep outside only): 5-8 um, identical to no band. On the card the same per-node skip is tied to
  flat brick-shaped plateaus, fins and grid steps when a surface grows fast (section 7). Not resolved.
- **Targeting** (2026-11-08; evidence state: measured on named cases; superseded by section 5). "Only the surface
  the brush faces" and "connected only" switches. On the thin-wall test they hold at r 6 and fail above it.

### 5. "The sheet under the brush": nothing behind the worked surface moves

The author's requirement (2026-11-12 12:5x): tools "only impact the surface that they are working on, and nothing
behind them, unless you activate a setting or a toggle". The cases he named: an eyelid over the eye, the two lips,
a bra's two faces over the skin beneath.

**The fault, measured** (CPU, 2026-11-12 13:00). One object: a 4 mm wall with a ball 2 mm behind it. Strokes on the
wall's front at r 6 / 12 / 20 mm moved the wall's back face by 0.30 / 2.0 / 3.9 mm (Draw), up to 3.5 mm (Clay), and
the ball behind by up to 3.5 mm (Draw) and 5.2 mm (Clay, closing the gap). Even Smooth on the flat front moved the
ball 0.52 mm at r 20. The older targeting switches held only at r 6.

**Mechanism** (landed 2026-11-12 14:40 as an opt-in switch "Protect surfaces behind"; on by default 15:48 at the
author's direction; evidence state: measured on named cases).
1. Surface cells are lattice cells whose corners have both signs.
2. Seeds are the surface cells under the dab centre, or the nearest surface cell along the dab's axis.
3. The seeds (taken from the 3 x 3 x 3 block of cells around the press, keeping only cells whose own normal faces
   the brush) are flooded from cell to cell across shared faces that the surface crosses, at most 1.6 (r + 2h) / h
   steps. The flood does not enter a cell whose normal has turned more than 100 degrees from the brush normal, which
   stops it rounding a lid's margin onto its back face.
4. Each node in reach takes the label of the surface cell at its own foot point x - F grad F / |grad F|; deep nodes
   take their brick's label. Labels are kept from the first time a dab sees a node, for the whole stroke.
5. A node's change is kept only if its label is the flooded sheet. An undecided band node joins the sheet unless one
   of its 26 neighbours belongs to another sheet (this rule removed specks in the test head's eye sockets).
6. The release re-distance restores every node not on the worked sheet.

Measured: thin wall, all tools, r 6-20: back face and ball 0. A lid over an eye (0.25 mm lattice, 0.5 mm gap) in the
real window on the card: without the gate Smooth r 15 / 20 moved the eye 1.7-2.4 mm and closed the gap on about 450
rays; with it 0 / 0 / 0 for Smooth r 15 / 20 and Draw / Inflate r 8. Two 3.5 mm plates 1 mm apart (lips): Draw,
Inflate, Clay, Crease, Pinch and Smooth move the facing surface 0.0 mm. Crease in a thin wall stops at the wall's
middle by design: a wall thins stroke by stroke and was not punched through in the cases tested.

Limits and failures:
- **It does not cover Grab, Stretch or Snake Hook.** Measured on the card: Grab moves the facing lip across a 1 mm
  gap by 0.52 mm and a 3 mm plate's back face by 0.72 mm; Snake Hook moves all four surfaces about 0.99 mm.
- On the human model the upper lid's field was fused with an old interior copy of the eye, so Smooth r 15 / 20 over
  it still moved that copy 0.16-0.20 mm (1.0-3.8 mm without the gate). That is a model defect, not the gate's.
- It blocks contact: a surface grown toward another surface of the same object stops at the gap's midpoint (section 7).
- Cost: Smooth r 15 fell from 23 to 15-18 fps with the labelling on the card (3-6 fps with it on the host, which was
  not shipped); a faster seed search brought it to 19.5 fps. Draw r 8 was unchanged. CPU test strokes: +21 %.

**Reach-through** (per-stroke setting implemented 2026-11-12; evidence state: implemented). Setting the cone to 180
degrees or more makes the flood cover the whole connected piece, so a lid's front and back are worked together while
the eye, a third surface, is left alone. It has no control in the window and its test cases were not run.
"This part" as a general scope for every tool is **proposed**.

### 6. Collision of brushes with locked and partner parts; the gap setting

(CPU 2026-11-09 00:19, card 00:34; partner parts 2026-11-11 18:06; evidence state: measured on named cases.)

Colliders for a stroke are: every locked protected part other than the target; any objects named as contacts; and
the target's partner parts paired with it as "touch" or "slide" (pairs marked "bond" or "pass" do not collide).
After each dab (CPU) or batch (card), on the nodes the edit changed:

    B(x)  = gap - min over the colliders of F_part(x)
    F'(x) = max(F_new(x), min(F_start(x), B(x)))     where B(x) > -2h;  F'(x) = F_new(x) elsewhere

So an edit never moves the editable surface into a part dilated by the gap; a node that already overlapped is never
pushed deeper; an untouched node keeps its bits. The rule also holds a band of two cells outside the part plus gap,
so the field next to the contact follows B and the surface lands at the gap rather than a cell short of it. The gap
is a brush setting (default 0.3 mm; 0 gives real contact, measured 0.000 mm on axis on 2026-11-09). A lid shaped
toward the eye slides over the globe at the gap instead of sinking in. Brush picks stop at locked parts instead of
passing through them (2026-11-12 15:48). Cost on the human model: 4.4-6.5 ms of a lid stroke's frame; one CPU case
where the bound was evaluated brick by brick took 10.5 s of a 13.3 s stroke and was cut to about 0.4 s by evaluating
all near bricks in one call (2026-11-12 01:5x, field bit-identical).

### 7. Sculpting surfaces into contact

The author's aim: bring two surfaces to touching while sculpting (a lid onto the eye, a fold onto a neighbouring
part), smoothly, with no typed values.

**Swell to contact** (first form 2026-11-09 01:15; eased formula 2026-11-12; on the card and the default Swell from
15:48; evidence state: measured on named cases). It is Inflate times a factor k per node, then the collider clamp:

    gs   = max(T - gap + F, 0)            distance from the surface here to the contact (T = the contacts' distance)
    near = 1 within r of a contact, falling linearly to 0 at 2 r
    ease = 1 - exp(-(gs + h/2) / E)       E = max(0.25 r, 2h): growth slows over the last E of the gap
    face = (1 + cos) / 2                  cos between the surface normal and the direction to the contact
    k    = 1 + near (gain x ease x face - 1),  gain = 3

Far from any contact k = 1 (plain Inflate). The h/2 lets the collider finish the touch. Measured: a ball of r 20 mm
under a locked ball of r 4 mm, 3 mm gap, strokes of r 4, s 0.5: 2.99 -> 0.10 mm in 5 strokes, far side unmoved, no
overlap. A lid's margin came down onto the eye in 3 strokes with no outward puff. A skin fold on the human model
facing a locked part across 3.55 mm (median of 135 rays): 3.55 -> 0.3 mm in 6 strokes at r 8, 36-38 rays touching.
Rejected variants: growth toward the nearest contact point made a flat mesa with horns; a 60-90 degree facing ramp
grew a column with cliffs; a plain cosine cut growth side-on and left a shelf.

**Settle onto** (2026-11-09 01:15; evidence state: measured on named cases). Each node in the dab reads the field
from a point farther from the nearest contact along that contact's gradient g: F'(x) = F(x + g A w), with
A = s r / 2, then the clamp. One move direction per dab keeps a sheet's thickness (2026-11-09 02:14).

**Settle by column** (2026-11-12 14:17; evidence state: measured on named cases). Each column of nodes across the
move direction moves by min(A w, its own clearance to the part), so one touching point no longer stops the whole dab.

**A locked part owns its volume** (2026-11-12, working copy, not landed; evidence state: measured on named cases).
F := max(F, -P) over the part's box, where P is the part's own field (a CSG difference). On the human model this
took the lid off the interior eye copy (fused lid rays 144 -> 27 and 135 -> 19 of about 277 per eye); the written
version failed a store check for reasons in the writer and was not published.

**Findings that stopped this line** (2026-11-12, real window, card):
1. With "Protect surfaces behind" on, growth toward another surface of the same object stops at the gap's midpoint:
   3.55 -> 1.7 mm, never touching, because nodes nearer the other surface count as another sheet. With it off:
   3.55 -> 0.0 in 3 strokes. A contact tool needs the rule "the worked surface advances until it touches the other
   one, which stays unwritten". Proposed, not built.
2. The card's per-node skip and band-only admission keep values that a fast-growing surface outruns: flat
   brick-shaped plateaus at r 8, a fin, a slit and grid steps after Smooth. All vanish with the skip off, at the same
   speed, but that also removes the admission limit that keeps large strokes within card memory.
3. Settle by column roughens a lid (grainy and pitted after 3 strokes at r 4, s 0.5) and barely improves contact
   (0.65 -> 0.6 mm): each column takes its nearest sample's clearance, so moves are piecewise constant.
4. Swell as a plain balloon closes a fold but rounds the whole neighbouring form into a pillow. Swell to contact
   stalls at about 2.6 mm and corrugates when the locked part's field is not a true distance outside the body.
5. On a fold, where a dab lands and which way it pushes followed the camera, not the surface.
6. The same formula differed between card and CPU (a lid's front puffed up to 0.6 mm on the card, 0.05 mm on the
   CPU). Not explained.

The author's report after sculpting: "Swell worked fairly well too." The contact tools are held for the rebuild.

### 8. Grab, Stretch and what replaces them

**What failed** (measured 2026-11-12 on a sphere of r 120 mm, h 1 mm, r 40 brush).
- *View dependence.* Grab, Stretch and Snake Hook map the drag onto a plane parallel to the screen. One intended
  pull, 30 mm out along the normal, gave 30.1 / 17.7 / 5.7 / 0.8 mm from views at 0 / 30 / 60 / 80 degrees.
- *"Stretch does not work."* It scaled the field about the pressed point itself, so that point moved 0.0 mm from
  every view after a 30-80 mm drag; only the inner ring changed; a wiggle left a sharp ring around a flattened disc
  (the author saw a half-circle cut out of the sphere); it compounded per mouse event (+5.19 against +4.44 mm for one
  drag at two mouse rates).
- *Path dependence.* Drag out, circle, return: Grab left 2.96 mm, Stretch 5.02 mm.
- *Folding.* F(x - v w) folds once the pull exceeds about 0.67 r; Snake Hook showed three surfaces on a probe ray.
- *The ring lied.* Grab moved 42 % at half radius and 7 % at 0.75 r against a drawn 50 % and 16 %.

**The drag constraint taken from the form** (proposed and built in a working copy 2026-11-12 19:0x, CPU path only;
evidence state: measured on named cases). At the press, with camera forward f and normal n0: m = f - (f . n0) n0,
s = |m|. The drag plane passes through the pressed point c0 with normal m / s: it contains n0 and faces the camera
as much as a plane containing n0 can. Each cursor ray is intersected with it. Within 5 degrees of head-on
(s < sin 5) the plane normal is normalize(b m/s + (1 - b) n0), b = s / sin 5, turning into the tangent plane. With
Ctrl the target is the point on the line c0 + t n0 closest to the cursor ray; near head-on, t is the drag's
component along the camera's up axis. Measured through the real window: 29.9 / 29.9 / 29.9 / 29.4 / 29.9 mm from
0 / 30 / 60 / 80 degrees and above. A first 15-degree blend slid an 80-degree pull 32 mm sideways and was narrowed.
An outside review noted that the silent change of operation near head-on should become a visible switch; open.

**Anchored moves** (same copy, CPU path; evidence state: measured on named cases). The result is a function of the stroke-start field
and the current total drag. On every update each touched node is restored to its stroke-start value and one
deformation is applied. On the card the plan is first-touch brick copies written back by a row-scatter kernel, then
one dab (compiled, never run). Wiggle-and-return residual: 0.00 mm.

**Grab as a flow** (same copy, CPU path; evidence state: measured on named cases). The deformation is the time-1 flow of
V_t(x) = w(|x - c(t)| / r) v with c(t) = c0 + t v. Because w(0) = 1 the pressed point travels with the kernel; a
flow of a Lipschitz field is one-to-one, so in the continuous map nothing folds (in the built version this was measured on probe rays only). F_new(p) = F_0(Phi^-1(p)), with the inverse integrated
backward in N = clamp(ceil(|v| / 0.15 r), 4, 64) midpoint steps over a ball of radius r + |v|/2 centred at c0 + v/2.
Measured: pressed point 29.75 mm for a 30 mm drag; one surface on every probe ray.

**Stretch as an anchored elongation** (same copy; evidence state: implemented; its check was stopped before it ran).
Axis a = v / |v|; pivot P = c0 - D a with D = max(T, r), T the depth of the form behind the press. Points between
the pivot and the pressed end scale by (D + Le) / D with Le = |v| w(rho / r), rho the distance from the axis; beyond
the end the displacement falls linearly to zero over max(2|v|, r). The inverse is closed-form.

**Form-aware Grab** (decision 2026-11-12 09:5x, in the author's words "Form-aware Grab"; evidence state: proposed;
the prototype had not been started in the records read). A Grab whose weight follows the form's own interior, for
example the distance through the material from the pressed point computed from the field itself, or a
connectivity-limited falloff, so the far side of a head or the next finger does not move. It uses no elasticity
equation and no material constants. It replaces the Elastic Grab prototype below.

**Elastic Grab prototype** (2026-11-11 20:09, test copy; parked, never in the window, NOT shipped; evidence state:
measured on named cases). The object's own linear-elastic response solved numerically on a coarse lattice inside the form,
Poisson ratio 0.4, a smooth load over 0.8 r normalised so the pressed point moves by the drag, a soft spring anchor
toward the reach, three conjugate-gradient solves at the press. On a head, a cheek pull added +38 cm3 with the back
of the head moving 0.3 mm, or +53 cm3 with a soft anchor holding the far side, against plain Grab's +74 cm3. It is withdrawn for the patent reason in
section 15. A volume-keeping variant of the flow Grab (velocity as the curl of a compact vector potential) is
proposed.

### 9. The pencil: creases, ridges and cuts drawn on the surface

**The shelf's pencil** (first commit 2026-11-05; measured on named cases; found faulty). Each stroke adds a tube
centred on the surface to the base tree (subtract for a valley, union for a ridge) with a smooth blend. Because the
tube is as deep outward as inward and the blend acts on anything in reach, a line along the upper lip 1.2 mm above
the mouth slit cut the lower lip across the slit by up to 226 um at 811 points. In the window it showed nothing
while drawing, then 3-4 minutes of work after release.

**The rebuilt pencil** (2026-11-12 15:5x, working copy only, never landed; evidence state: measured on named cases).
1. Picks define a Catmull-Rom curve sampled about every h. Each pick is moved along its view ray onto the surface as
   it was before the stroke; without that the line steered by its own fresh relief (0.27 mm median wobble: beaded
   ridges, ribbed valleys); with it, at most 0.13 mm.
2. Every value is computed from the stroke-start field F0, held per brick. Each extension recomputes only nodes in
   reach of the changed part of the curve, so drawn-live equals laid-at-once exactly (0.0 um).
3. Writes are limited to the drawn sheet by the flood of section 5, seeded by the stroke's own samples.
4. With D(lat) the profile (round, V, U or custom, softened at its shoulders) at lateral distance lat from the
   line, hgt a node's height above the drawn surface and dn its depth below it: a valley is
   F' = max(F0, min(hgt + D, 3h - dn)), removing the volume between the floor and 3h above the surface; a ridge is
   F' = F0 - D.
5. Depth is limited per sample to half the form's thickness or half the gap, measured along the normal on F0.
6. The keep-apart guard runs on every extension.

Measured: valley 1.497 of 1.5 mm; fine ridge 0.6 x 0.15 mm gave 0.141; a facing sphere across 0.75 mm moved 0.0 um;
the lip case 0.0 um on the facing lip. Limits: a V cut closes where narrower than a cell, so cuts need a U profile;
the finest line is about 0.5 mm wide at h = 0.25 mm. **Not reached:** 4-13 fps while drawing with hitches to 527 ms,
against the brushes' 20 fps and more. The pencil work was stopped with the rest of the toolkit. What the author
used by hand that evening was still the shelf's old pencil, and his verdict on it was: "pencil tool still sucks. It
doesn't crease well, it covers too broad of an area, it mangles most surfaces surrounding the mark".

### 10. The toolkit framework: ten principles and the experience harness

Written 2026-11-12 ~17:55 after a study of Blender's sculpt tools; evidence state: proposed. The build order that
follows from it (the harness first, then Grab on the common machinery) was agreed by the author at 18:4x.

1. **Full 3D.** A tool gives the same result from any view; the camera only says where the hand points.
2. **What you see is what moves.** The ring is the true reach and its inner ring the true 50 % weight.
3. **A stroke depends only on the hand's path over the stroke-start form**, not on mouse rate, frame rate, batching,
   CPU or card.
4. **Deformations are smooth and invertible.** No folds, tears or ghost surfaces.
5. **Scope is explicit and three-dimensional**: this surface (default), this part, the connected form, the object.
6. **Size is free**, from sub-millimetre to the whole object.
7. **Big form and fine detail are separate but linked.**
8. **Strength, pressure and modifier keys mean the same in every tool.**
9. **Immediacy**: the window never stops answering.
10. **A tool is proven by the experience it gives**, not only by field numbers.

Principle 1 goes beyond Blender, whose Grab, Snake Hook, Elastic Deform, Pose and transform tools also work in the
view plane; sculptors there choose the view to choose the direction.

**Harness** (2026-11-12 18:49; measured on named cases). The real window is driven offscreen by operating-system
mouse events and results are probed in 3D on the field. T1 same intent from many views; T2 the ring; T3 wiggle and
return; T4 mouse rate 30 / 120 / 480 Hz; T5 nothing behind moves; T6 size; T7 Smooth keeps the form; T8 modifiers.
Baseline on the shipped tools: surface brushes view-independent within 3-4 %; Draw 11 % stronger at 480 Hz than at
30 Hz; window stalls up to 111 ms. The pass mark is a person blocking out a head from a sphere.

**Window freeze** (2026-11-12 ~17:49). The author's window stopped answering for about five minutes. The run's log
was overwritten by the relaunch, so the work that ran is unknown. A mechanism was measured: a compiled kernel that
keeps the interpreter lock stops the window thread for its whole run (a 93 ms kernel gave a 95 ms stall; 11 ms when
it releases the lock). Landed 18:06: a per-session log that a relaunch never erases, and a watchdog (a 250 ms timer
re-arming a native 2 s traceback dump) that writes every thread's stack while the window is stalled.

### 11. Per-stroke costs and frame rates

Targets set for the lane: 30 fps or more during every stroke at r 1-80 mm; 100 ms or less from release to the next
stroke. They are met on simple forms at small and medium radii and not met on the human model's body.

| Case (card unless said) | Result |
|---|---|
| First head test, 2026-11-11 10:47, before the rebuild | Grab median 13 fps, Smooth 6.6, Inflate 9.9; worst gap 401 ms; release to next stroke 0.29-0.38 s |
| Bust sphere, 1166 x 611 view, 2026-11-11 16:21, r 20 / 60 / 80 | Grab 53 / 33 / 27 fps, Draw 50 / 40 / 28, Smooth 53 / 38 / 28 |
| One batch at r 80 | Grab: plan 10.2 + re-distance 15.1 + borders 3.9 ms; Draw 6.8 + 7.7 + 4.0; Smooth 7.3 + 7.5 + 6.6 + dabs 5.5; render 15.5 ms |
| Bust sphere, harness, 2026-11-12 19:00, r 5 / 20 / 60 / 80 | Draw 37 / 31 / 22 / 18 fps; Smooth 29 / 27 / 19 / 16; Grab r 60 / 80: 30 / 27 (one r 20 run 5.0 fps with an 845 ms gap, unexplained); window stall 29-105 ms |
| Head scene, mirrored Smooth r 30 | 6.6 -> 21.6 fps after removing a per-batch scene sync and mask scan; in a separate A/B the thread switch interval alone gave 15.0 -> 29.3 fps |
| Human model, upper lid, small strokes | about 22-23 fps quiet; 2-5 fps under machine load. Frame at 21-22 fps: tables 1.6-8.1 ms, frame 6.7-10.1, collider 4.4-6.5, dab kernels about 1 ms |
| Human model, torso Draw | 14.6 -> 23.6 fps, release 173 -> 113 ms (2026-11-12 10:47) |
| Human model, body Smooth | 2.5 -> 5.8 fps, release 0.73 -> 0.33 s |
| Human model, body Flatten | 0 -> 1.0 fps, release 2.84 -> 2.38 s; r 40 writes 90,000 bricks, 82 % of them more than 4 mm from the surface |
| Human model, Draw r 10 on the cheek | per batch 830-1891 ms -> 6-149 ms after keeping the cell-to-row map on the card instead of uploading 332 MB per batch (2026-11-12 01:49) |
| Human model, CPU path, lid Smooth r 4, 13 dabs | 13.3 s -> 8.4 s |
| Human model, CPU path, torso Draw r 25, 2 dabs | 11.8 s and 23,972 bricks with no band; 5.3 s and 16,008 bricks with the correct one-sided band |

The cost of a ball of radius r grows as (2r / h)^3 nodes, and the radius is capped at 80h (160h on the human model).
Coarse levels for large tools, with detail stored relative to the coarse surface, are proposed. A whole-body Smooth
at r 60 once ran the card out of memory (6.52 of 6.91 GB in use) and hung the engine; strokes are now split to fit
and a card failure falls back with a notice. The author's report after sculpting on 2026-11-12: "Performance is
abysmal. Lag started hitting me after several minutes, some tools take several seconds to do their job."

### 12. Test 1: a sphere to a head, and why its scripted runs meant little

Test 1: front and side reference photos on one axis, a sphere sculpted into a basic head (nose, eye sockets, jaw and
chin, cheeks) in under five minutes.

- 2026-11-11 10:47, first scripted run: "not a head": a lumpy egg after 144 actions and about 7.8 minutes; Smooth
  removed 6 % of the lumps; a torn ring under the neck; rectangular seams at the mouth; 22 of 166 presses turned the
  view instead of sculpting. The author: "You used too small of a brush."
- After the rebuild, scripted runs from a fresh sphere reached all stages in about 39 actions and 113-121 s of a
  person's time, 0 orbits, no holes, outline within 3.9 / 5.3 / 4.8 mm (front / face / back), lumps 2.47-2.52 mm RMS
  (target 1.5, not met). The author ranked a brushes-only run best so far on 2026-11-11 18:48.
- Along the way: a nose split into two lobes (mirror), a granular jaw (midline snap stacking about 15 Smooth passes),
  jaw scratches on the card (ring samples reading unheld bricks), and the chin bands of section 3.1.
- 2026-11-12 17:2x: the author could not block out a head by hand. The script had pulled only from the side and
  front views, where screen-plane tools happen to do what was meant, with strokes chosen to work, and it judged the
  outline in millimetres. It measured the field, not the experience.
- The author's summary that evening: Smooth and Swell worked fairly well; "Most of the other tools are effectively useless";
  "progress has been made, but it has a LONG way to go."

### 13. The shell

All in the live tool; each item was checked by a scripted run of the real window (evidence state: measured on named
cases) unless stated.

- **Window** (2026-11-11 11:50). Workspace tabs, a Create row of primitives, a telemetry strip (fps, frame time, RAM,
  field and undo memory, the field's Lipschitz bound), the viewport, docks for properties, references, layers,
  regions and light, and a timeline of every operation with a replay check. A status line always says what each
  mouse button does. The brush ring is drawn draped on the surface with live millimetres; Alt shows the local
  thickness and the gap to the next surface (2026-11-11 23:49).
- **Shelf, palette, keys** (2026-11-11). Tools sit on a shelf in groups (Object, Build, Refine, Move, Cut & Join,
  Protect & Select, Contact). Space opens a type-to-find palette; favourites sit in a strip; a Keys window lists and
  rebinds every key. Each brush keeps its own radius, strength, falloff and presets.
- **Glyphs** (first set 2026-11-11, drawn from the brush falloff; redrawn 2026-11-12 08:56). Each glyph is a filled
  cross-section of what the tool does to a surface. The redraw raised the closest pair's image distance at 32 px
  from 0.0039 to 0.0458. A names mode shows one tool per row with its name.
- **Grids** (2026-11-11 16:42; on by default 2026-11-12 09:17). A depth-tested floor grid under the scene's lowest
  point, fading with view distance; a view grid in axis views; spacing on a metric or imperial ladder with labels;
  optional height contours on the model. Cost: 1-2 ms per displayed frame.
- **Reference rig** (2026-11-11 11:50). Front and side photos as planes crossing on one axis, keyed out without a
  fringe, placed by crown and chin clicks or an auto-fit proposal; a true orthographic view; a silhouette check in
  millimetres. On a synthetic head with known truth, clicked landmarks put the model's outline within 0.12 / 0.17 mm
  mean of its photos. Drawing the planes cost 10.2 ms per frame until the per-view layers were cached (2.4 ms).
- **Typed transforms and masses** (2026-11-11). Move, rotate and scale with typed values, axis locks and snapping;
  primitives joined by smooth unions as live "masses"; flip and mirror on any axis; duplicate in place.
- **Regions** (2026-11-11 21:57 data, 22:33 window). A 16-bit region id per lattice node and a region table per
  object, carried through save, duplicate, mirror and merge. Regions are made by lasso, from the mask, or from a
  part. Hide, isolate, show all, and box hide. The CPU tracer clips hidden regions (the surface and the solid
  beneath, by each point's foot on the surface) and boxes (convex plane sets); with nothing hidden the image is
  bitwise unchanged. The card clips boxes (23:33). **Hidden regions on the card are not built.**
- **Section tool** (2026-11-11 19:17 CPU; card clip 23:33). A section plane on the camera cuts the view and shows
  the solid's cut face with a millimetre ruler; 24 cases are bitwise identical with no section.
- **Cut tools.** Cut (first commit 2026-11-05): a closed outline drawn on screen is punched through the object along
  the view as an exact swept-outline field. Cut with a shape (2026-11-11 16:30): any object is the cutter, with a
  side and result choice and a live preview of the volume that will go (16:48; areas within 0.5 % of geometry).
  Conform with a wall (19:21): hollow a form leaving a wall of set thickness, using a true depth computed from
  surface points because the sculpt layer is a distance only inside its band (0.043 mm maximum error on a ball).
- **Marks** (2026-11-11 19:10). A circle or note placed on the model, saved to a shared file so a second reader can
  see exactly the point meant.
- **Also**: measures (sections, girth, tape, volume), sculpt layers with a 0-1 dial, view modes on the card, a
  visible notice when an action fails or the card is lost.

### 14. Published work this learned from

- Blender's sculpt mode, versions 4.2-5.0: the manual (CC BY-SA 4.0), read for study; the source (GPL), read for
  behaviour only, no code copied; Blender Studio course pages (CC BY), descriptions only. Taken as ideas: the area
  plane under Clay and Flatten, Ctrl as each brush's inverse, autosmooth as a dial, middle-button orbit, pinch toward
  the stroke line, anchored versus incremental tools, feathered symmetry.
- Maxon ZBrush documentation, tutorials and user forums (all rights reserved): study only, paraphrased.
- F. de Goes and D. L. James, "Regularized Kelvinlets", SIGGRAPH 2017: studied; not implemented in the tool.
- W. von Funck, H. Theisel, H.-P. Seidel, "Vector field based shape deformations", 2006; A. Angelidis, M.-P. Cani,
  G. Wyvill, S. King, "Swirling-sweepers", 2004: prior art for moves as flows of divergence-free fields.
- A. Ricci 1973 and later implicit-modelling texts for CSG by max and min.
- Public forum threads recording that stopping an inflate at contact is not built into mesh sculptors.
- A 1912 anthropometric card from Wikimedia Commons (CC0), used only to test the reference rig.

### 15. Patent readings as recorded

These are a non-lawyer's readings made to decide what to build. **They are not legal advice and no clearance is
claimed for any mechanism in this document.** As we understand it, a dated publication is prior art only against later filings. The rule
followed: a concern names the claim element behind it; a design-around names the element that is absent.

- **US 10,586,401 B2 (Pixar; active, expected to 2038).** The patent concerns deformations determined from
  regularized solutions of an equation of linear elasticity with a specified Poisson ratio. The Elastic Grab prototype
  (a lattice solve of linear elasticity) was flagged in our review; it was never merged, is parked, and will not ship. An
  earlier internal note that had treated it as unaffected was withdrawn on 2026-11-12. Direction of the
  design-around: the form-aware Grab decided the same day and the flow Grab of section 8 use no elasticity equation
  and no Poisson ratio; whether that is sufficient is not asserted here.
- **US 8,265,909 B2 (MERL; active, expected to 2031).** Element of concern, claim 1: the distance field of a shape
  swept along a parametric path, found by determining "in a continuous manner" an optimal path parameter per sample.
  Flagged in two places: the capsule primitive (a sphere's distance at the clamped closest point of a segment) and
  the brick admission test of an unwired flow-Grab module (brick centres projected onto the drag segment with a
  clamped parameter). **Both are being redesigned; no design-around had been chosen or recorded when this was
  written**, and the full reading was still owed. Ordinary dabs are separate fixed-centre edits added in turn, with no
  per-sample search for a placement. The records hold no reading of the rebuilt pencil, which finds each node's
  position relative to a sampled stroke curve, against this patent; that reading is owed before the pencil returns.
- **US 9,836,879 B2 and US 10,062,203 B2 (Autodesk)**, read against the sheet flood. The first requires paths
  through interior voxels from an influence to a mesh vertex, a boundary-voxel penalty, a skinning weight and an
  animation; the flood walks surface cells from the press point and yields a yes/no gate on a brush write. The second
  requires voxelising a mesh; there is no mesh.
- **US 8,988,461 B1 (Disney / ETH)**, read against Swell and Settle: each independent claim requires paint strokes
  that "do not alter the shape" of the object; these tools alter the shape. Older distance-field sculpting and
  haptic-clay patents (MERL, SensAble) appear expired; their status was not confirmed at the patent office.
- **US 9,830,743 B2 (Autodesk)**, a volume-keeping smoothing brush: to be read before any keep-volume Smooth ships.
- The patent searches were web searches and partial. "None found" is not "none exists".

### 16. What is not established

- Whether the rebuilt Grab, Stretch and drag constraint give a sculptor a usable tool. They are measured on the CPU
  path through a scripted window only. The card path of the anchored flow has never run; the new Stretch is unmeasured.
- The form-aware Grab: decided, not prototyped in the records read.
- The cause of the five-minute window freeze, and of the 5 fps Grab run at r 20.
- Why the card and CPU differ for Swell to contact; which of the card's two skip rules causes the plateaus and fins.
- A Smooth that keeps volume, has spheres of every size as fixed points, and does not band on a chin.
- Scope for move tools, reach-through as a control, the honest ring, levels, region transforms, and deformers that
  carry the sculpting with them: all proposed.
- The author's hands-on block-out of a head with the rebuilt toolkit, which is the pass mark, has not happened.

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

