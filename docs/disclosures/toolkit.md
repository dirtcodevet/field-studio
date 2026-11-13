# Field Studio sculpting toolkit: technical disclosure of mechanisms

Dated 2026-11-12 (Pacific time). Written for public record. Each mechanism is described so that a skilled reader can
reproduce it, with the date it was adopted or proposed (from the project's version history) and its status:

- **SHIPPED**: in the tool's live code, measured.
- **BUILT**: implemented in a working copy of the tool, measured as stated, not yet shipped.
- **PROPOSED**: designed and written down on the date given, not yet implemented.

## 0. The setting

Field Studio models every object as an implicit field F (a signed distance; F < 0 inside), never as a mesh. F is an
analytic base tree (primitives, smooth booleans, deformers) plus a sculpt layer: values added to F on a sparse lattice
of bricks of 8 x 8 x 8 nodes at spacing h, stored only near surfaces. Every sculpting tool edits that sculpt layer.
After a stroke, the touched bricks are re-distanced (|grad F| restored to about 1). The surface is drawn by tracing
rays into F. "Surface" below means the zero set of F.

Notation: c0 is the point where a stroke is pressed (the first surface along the cursor ray). n0 is the surface
normal there. r is the brush radius. w(u) is the brush falloff, with w(0) = 1 and w(u) = 0 for u >= 1 (default: 1
minus smoothstep). f is the camera's forward axis and u_cam its up axis.

---

## 1. The design framework: ten principles (PROPOSED 2026-11-12, toolkit review)

These define how every tool is built and judged.

1. **Full 3D.** A tool gives the same result from any view angle.
2. **What you see is what moves.** The drawn ring is the true reach.
3. **A stroke depends only on the hand's path over the form as it was at stroke start.** It does not depend on mouse
   rate, frame rate or batching.
4. **Deformations are smooth and invertible.** No folds, tears or ghost surfaces.
5. **Scope is explicit and three-dimensional.** This surface, this part, the connected form, or the object; never
   the screen projection. By default, nothing behind the worked surface moves.
6. **Size is free.** The same behaviour holds from sub-millimetre to whole-object brushes.
7. **Big form and fine detail are separate but linked.**
8. **Strength, pressure and modifier keys mean the same in every tool.**
9. **Immediacy.** The window never stops answering.
10. **A tool is proven by the experience it gives**, not only by field numbers.

## 2. The drag constraint taken from the form (PROPOSED 2026-11-12; BUILT 2026-11-12 19:0x)

Problem: mapping a 2D drag to a 3D move on a plane parallel to the screen makes a pull along the surface normal
depend on the view. Measured on the camera-plane mapping, a 30 mm pull along n0 gave 30.1 / 17.7 / 5.7 / 0.8 mm from
views at 0 / 30 / 60 / 80 degrees between n0 and the view plane.

Mechanism:
- At the press, compute m = f - (f . n0) n0 and s = |m|.
- **Default drag plane.** It passes through c0 with normal m / s. This plane contains n0 and faces the camera as much
  as a plane containing n0 can. Each cursor position maps to the intersection of its ray with this plane. The drag
  vector is that point minus c0.
  - Seen from the side, this is the view plane.
  - From any oblique view, a cursor placed on the screen image of c0 + D n0 maps exactly to c0 + D n0.
- **Near head-on.** When s < sin(5 deg), the plane normal becomes normalize(b m/s + (1 - b) n0), with b = s / sin(5 deg).
  The plane turns continuously into the tangent plane, so the drag slides the surface.
- **Normal lock (Ctrl).** The target is the point on the line c0 + t n0 closest to the cursor ray. When the ray is
  within 5 deg of parallel to that line, t is taken as (q - c0) . u_cam instead, where q is the ray's hit on the view
  plane through c0 ("drag up = out").

Measured (working copy, real window driven by scripted mouse events, CPU path): the same 30 mm pull gave
29.9 / 29.9 / 29.9 / 29.4 / 29.9 mm from 0 / 30 / 60 / 80 degrees and from above.

## 3. Anchored moves: computed from the stroke-start field (PROPOSED 2026-11-12; BUILT 2026-11-12 19:1x)

The result of a move tool (Grab, Stretch) is a function of the field at stroke start and the drag's current total.
It is never an accumulation of per-event increments.

- On every update, each lattice node the stroke has touched so far is restored to its stroke-start value.
- One deformation is then applied for the total drag.
- On the GPU, each touched brick's stroke-start values are kept as host-side "first-touch" copies. A row-scatter
  kernel writes them back into the brick pool, the bricks' border (apron) nodes are refreshed from their owners, and
  one deformation dab is computed.
- Consequences:
  - The drawn result never depends on mouse-event rate or batching.
  - Dragging out and back to the start gives the start: measured 0.00 mm residual, against 2.96 mm for the
    incremental method.

(Mesh sculptors restore the original mesh at each step of an anchored grab. Here the same principle is carried out on
an implicit field, by first-touch brick restore and inverse-map resampling.)

## 4. Grab as the flow of a translating kernel (PROPOSED 2026-11-12; BUILT 2026-11-12 19:0x)

- **The deformation.** It is the time-1 flow of the time-dependent velocity field
  V_t(x) = w(|x - c(t)| / r) v, where c(t) = c0 + t v, t in [0, 1], and v is the total drag.
  - Because w(0) = 1, the pressed point travels exactly with the kernel centre, so it follows the drag.
  - The flow of a Lipschitz velocity field is one-to-one, so the map never folds: no ghost or inner surfaces, at any
    drag length.
- **The new field.** It is F_new(p) = F_0(Phi^-1(p)), where F_0 is the stroke-start field.
- **The inverse map.** Phi^-1(p) is evaluated per lattice node by integrating backward from t = 1 to t = 0 in N
  midpoint (second-order Runge-Kutta) steps, with N = clamp(ceil(|v| / (0.15 r)), 4, 64). Each step:
  1. w1 = w(|x - c(t1)| / r) at the step's end time t1;
  2. the midpoint x_m = x - (dt/2) w1 v at t_m = t1 - dt/2;
  3. x = x - dt w(|x_m - c(t_m)| / r) v.
- **Region.** The evaluated region is a ball centred at c0 + v/2 of radius r + |v|/2, which covers the swept capsule.
- **Measured** (working copy): the pressed point moved 29.75 mm for a 30 mm drag (probe step 0.25 mm); no probe ray
  crossed an extra surface.
- **How this differs from regularized-Kelvinlet brushes** (US 10,586,401 B2): w is a compact polynomial falloff
  transported along the drag. The mechanism uses no solution of the equations of elasticity, no Poisson ratio, no
  regularized fundamental solution and no multi-scale sums of such solutions.

**Volume-keeping variant (PROPOSED 2026-11-12).** The velocity is taken as the curl of a smooth, compactly supported
vector potential. A divergence-free flow preserves volume exactly and stays one-to-one. Prior art for vector-field
shape deformation: von Funck, Theisel and Seidel 2006; Angelidis, Cani, Wyvill and King 2004. A volume dial blends
toward the plain flow.

## 5. Stretch as an exact anchored elongation (PROPOSED 2026-11-12; BUILT, not yet measured, 2026-11-12 19:1x)

Its predecessor scaled F about the pressed point itself, along a camera-plane axis. Measured: the pressed point moved
0.0 mm from every view, and a circling drag left a sharp ring around a flattened disc.

Mechanism (one-to-one at any drag; closed-form inverse):
- **Axis and pivot.** The axis is a = v / L, with L = |v| (the drag resolved by section 2). The pivot is
  P = c0 - D a, where D = max(T, r) and T is the depth of the form behind the press point: the distance from just
  inside the surface (c0 - 2h n0) along -a to where F first turns positive. Pulled outward on a closed form, the whole
  extent of the form along the axis lengthens and its far side stays put. Pulled along the surface, a brush-sized
  piece stretches.
- **Per-node coordinates.** xa = (x - P) . a, rho = |x - P - xa a|, Le = L w(rho / r), and fade = max(2L, r) > Le.
- **The forward map along a** (rho unchanged):
  - x0 in [0, D] is scaled by (D + Le) / D, so the pressed end moves Le; on the axis that is L, and it follows the drag;
  - x0 in [D, D + fade] gets a displacement falling linearly from Le to 0;
  - everything else is unchanged.
- **The inverse used for resampling**, F_new(x) = F_0(x + (xa0 - xa) a):
  - xa0 = xa D / (D + Le) for xa in [0, D + Le];
  - xa0 = D + (xa - D - Le) fade / (fade - Le) for xa in [D + Le, D + fade];
  - xa0 = xa otherwise.
- **Region.** The evaluated region is the ball around the cylinder from P to the fade's end, of radius r.

## 6. "The sheet under the brush": nothing behind the worked surface moves (SHIPPED 2026-11-12 14:40; default on 15:48)

For each dab of a surface brush:
1. **Seeds.** The lattice's surface cells (cells whose nodes change sign) under the dab centre, or, failing that, the
   nearest surface cell along the dab normal.
2. **Flood.** The seeds are flooded across surface cells by 26-neighbour connectivity, for at most
   1.6 (r + 2h) / h steps. The flood stops where the field's gradient turns more than 100 degrees from the dab normal
   (the cone).
3. **The rule.** A node's change is kept only if its nearest surface point lies on that flooded sheet. Each node keeps
   the label from the first time a dab saw it, for the whole stroke.
4. **Undecided nodes.** An undecided band node joins the sheet unless one of its 26 neighbours belongs to another sheet.

Measured on two 3.5 mm plates 1 mm apart in one object (a lips-like slit): strokes on the upper surface moved the
upper plate's underside and the facing lower surface by 0.0 mm for Draw, Inflate, Clay, Crease, Pinch and Smooth.

## 7. Picks on the stroke-start surface (BUILT in a working copy, 2026-11-12)

A tool that picks its path while changing the surface steers by its own wake. Each pick is moved along its view ray
onto the stroke-start surface, evaluated as the current field minus the stroke's own change.
- Measured on a drawn crease: sideways wobble fell from 0.27 mm (median) to at most 0.13 mm.
- With the stroke-start values held per brick, and only nodes within reach of the changed part of the path
  recomputed, the live result equals the one-shot result exactly.
- PROPOSED for every tool.

## 8. Other mechanisms (SHIPPED unless stated)

- **Mirror feather** (2026-11-11). Where a dab and its mirror partner overlap, each weight is scaled by
  (w^2 + w2^2) / (w + w2)^2, so the pair sums to (w^2 + w2^2) / (w + w2). That sum is smooth, needs no maximum, does
  not double on the midline, and equals w where one dab acts alone.
- **Strength response** (2026-11-11). The per-dab factor is k_max (1 - exp(-k s / k_max)): monotone in strength s
  and saturating, never clamped. Displacement brushes move 0.2 x spacing x s x r per dab.
- **Area plane for Clay and Flatten** (2026-11-11). The plane is the falloff-weighted mean position and normal of
  surface samples on a sunflower disc of radius 0.8 r, oriented by the hit normal, each projected onto F by one
  Newton step.
- **Smooth at the brush's scale** (2026-11-11). For each node within the surface band:
  - a height fit in the node's own tangent plane from two rings of 8 samples at 0.3 r and 0.6 r, each projected by
    one Newton step, with pair weights and a same-sheet weight;
  - plus a lattice-scale Laplacian term;
  - the change is capped at 2h x max(1, |s|).
- **Topology guard** (2026-11-10; stroke-end pocket rule 2026-11-11):
  - a node may change sign only where the inside and outside components of its 26 / 6 neighbourhood stay as they
    were, so no stroke joins, films over or tears surfaces;
  - air pockets a stroke itself creates are filled at its end;
  - openings and cavities that existed before the stroke are never closed.
- **Band-only dabs** (2026-11-08 / 2026-11-12). Nodes farther from the surface than 1.6 x the dab's move + 6h keep
  their values, one-sided by the direction the surface moves. The re-distance at release rebuilds the band.
- **Freeze diagnostics** (2026-11-12 18:06). A per-session log that a relaunch never erases. Every 250 ms, the window
  thread re-arms a native watchdog timer (Python's faulthandler.dump_traceback_later, 2 s, repeating). While the
  window doesn't answer, every thread's stack is written to the log every 2 s, even when another thread holds the
  interpreter lock. On recovery, one line records the stall's start, end and length.
- **The honest ring** (PROPOSED 2026-11-12). The cursor's outer ring is drawn at the tool's true support, and its
  inner ring at the true 50 % contour of the weight actually applied, for every tool, moves included.
- **Levels** (PROPOSED 2026-11-12):
  - a coarse level of the field for large tools, with fine detail stored relative to the coarse surface (displacement
    in its frame);
  - big moves carry the detail with them;
  - smoothing one level leaves the other;
  - cost no longer grows as (r / h)^3.
- **Scope as a weight** (PROPOSED 2026-11-12). Each tool's falloff is multiplied by a scope weight computed from the
  field: this sheet (section 6), this part (both faces of a thin part such as an eyelid, and nothing beyond it), the
  connected form, or the object.

## 9. The experience-level test harness (BUILT 2026-11-12)

The real window is driven offscreen with operating-system-level mouse events. Results are measured in 3D on the
field: probe rays give surface heights and the number of surfaces crossed, which counts ghost surfaces. Each case is
undone and checked before the next.

- **T1 Same intent, any view.** Pull the pressed point 30 mm along its normal from 0 / 30 / 60 / 80 degrees and from
  above. Draw a 40 mm line from 30 / 60 / 90 degrees. The result must not depend on the view.
- **T2 The ring.** The change at u = d / r against the drawn falloff; 0 beyond the ring.
- **T3 Wiggle and return.** Out, around a 10 mm circle, back to the press point: the form must be back.
- **T4 Mouse rate.** The same drag at 30 / 120 / 480 events per second.
- **T5 Nothing behind moves.** A 3 mm plate and a 1 mm slit pair: the back face and the facing surface must not move.
- **T6 Size.** Frames per second, worst frame gap, the window's own stall, and release-to-responsive time, from
  r 5 mm to the cap.
- **T7 Smooth keeps the form.** Bump height and volume over 10 passes.
- **T8 Modifiers.** Ctrl's direction for every tool.
- **Pass mark.** A person blocks out a head from a sphere.

Measured on the shipped tools on 2026-11-12:
- Surface brushes are view-independent within 3-4 % and move nothing behind.
- Grab and Snake Hook fall with the view angle and move surfaces across a 1 mm gap.
- Smooth loses 21 % of a bump's volume over 10 passes.
- Draw is 11 % stronger at 480 than at 30 mouse events per second.
- Window stalls reach 111 ms.
