[Contents](README.md) | [Complete single file](../RESEARCH.md)

# Part 4. The field store, its writer, the scaling study and benchmarks

This part describes how Field Studio stores a form, how edits are written into the store without damaging it, how the
store is checked before a version is published, how the interactive editor mirrors it on the graphics card, and what we
measured about its cost. It is written so that a skilled reader could rebuild each mechanism.

**How to read the dates.** Dates and times are Pacific time, 2026, and are internal provenance: when a mechanism was
first stated or adopted in the project record (the private version history, by commit time, or a dated lane record
where the commit came later). They are not publication dates. The public date of this text is the date of the release
that carries it.

**Evidence labels.** Each mechanism carries exactly one of: *proposed*; *implemented*; *measured on named cases*;
*reproduced under a frozen configuration*; *accepted in the workflow* (the published model is built or edited through
it).

**Machine.** Everything was measured on one Windows PC with 32 GB of RAM and one 8 GB RTX 5050 card, shared with other
jobs and with the desktop (which holds about 1.0-1.3 GB of the card). Where a number was taken beside other load, it
says so. The tool's own budget target is 16 GB of RAM and 8 GB of card memory. Unless stated, "the human model" is one
full-body figure at 0.25 mm finest spacing.

**Limits of this text.** Nothing here claims that a method is new, protected or clear of anyone's patents. Patent
readings are a non-lawyer's reading of claim text, carried over as recorded; they are not legal advice, no legal review
has been done, and no clearance is claimed. Published work we learned from is listed in the last section.

### 1. The brick lattice and its four levels

*First in the record 11-04 (the tiled full-body field); in the version history from 11-05 23:19 (`63d07d7f`).
Evidence: accepted in the workflow.*

- Space is cut into cubic **tiles** of 32 mm. Per level, a tile holds a contiguous run of **bricks**. A brick is
  8 x 8 x 8 lattice cells stored as **9 x 9 x 9 samples** (729 values); the ninth sample on each axis is the face shared
  with the next brick, so trilinear interpolation inside a brick never needs a neighbour.
- Samples are signed distances in metres (negative inside), stored as 16-bit floats (1,458 bytes a brick). Each level
  has a cap; a sample at the cap (|v| >= 0.999 cap) means "farther than the cap, no distance known", never a distance.
- Four levels, each twice as coarse as the one below:

  | level | node spacing | brick edge | band kept round the surface | cap |
  |---|---|---|---|---|
  | L0 | 0.25 mm | 2 mm | 1.5 mm | 3 mm |
  | L1 | 0.5 mm | 4 mm | 2 mm | 3 mm |
  | L2 | 1 mm | 8 mm | 4 mm | 6 mm |
  | L3 | 2 mm | 16 mm | 8 mm | 12 mm |

  L3 and L2 cover the whole body (L3 its inside too). L1 and L0 exist only in a band round the surface and only in
  regions that carry them. On the human model as of 11-12: 218,766 L0 bricks in 444 tiles, 72,955 L1, 64,609 L2,
  16,070 L3. (An 11-04 version held 145,575 L0 bricks; the count grew as regions were added.)
- Files of one version: `meta.json`, `tile_grid` (tile coordinate -> tile index), and per level `tiles`,
  `tile_table` (tile -> slice of bricks), `brick_ijk` (global brick coordinates), `sdf` (the samples), `mat` (an 8-bit
  material per sample) and `attr` (a 3 x 3 x 3 attribute lattice per brick, interpolated trilinearly).
- **One evaluation rule, used by the CPU reader, the card, the writer and every scan** (editor base node 11-08,
  `core/fv8base.py`, `dc56e0ad`). At a point, the finest level holding a brick there answers, trilinearly inside that
  brick. A level with no brick there gives only a bound (its band minus its spacing) and the next coarser level is
  read. With no level at all, the point is outside by at least the largest bound. A capped sample takes the finest
  coarser level's uncapped value at that position.
- **Contract between levels:** every L1-L3 node is also an L0 node position, and a coarse node should equal the exact
  restriction of the finest data there. Every writer below enforces this and every scan below measures it.
- **A detail-bound pyramid on L0.** Each L0 brick that holds surface has a slot of 341 cells x 12 values (fp16): a
  quadtree (1 + 4 + 16 + 64 + 256 cells) of minimum and maximum bounds and tilt for the render-time skin relief, so the
  tracer can bound the displaced surface. It depends on the samples, materials and attributes, so an edit lists the
  slots it made stale and they are re-solved. Measured 11-04 on the CPU (64 x 64 samples per brick, 488 bricks a chunk):
  5,706 slots plus 1,000 checks in 87.9 s at 2.3 GB peak; all 84,232 slots in 47.7 min at 2.35 GB peak. A first
  whole-set solve on one dense box needed about 11 GB and over 50 minutes and was replaced by the region-by-region
  solve. *Evidence: measured on named cases.*

Accuracy of the format on analytic fields (11-04, all four levels built from functions, no mesh): a 12 mm sphere's zero
set is off by at most 1.3 / 5.2 / 20.9 / 84.4 um at L0 / L1 / L2 / L3; normals jump at most 0.02-0.18 degrees across
brick faces. Two failures of thresholds written before the run are kept as failures: the per-cell gradient bound exceeds
1 by about h / r on curved surfaces (1.25 at L3 on r 12 mm; it is still a safe bound for stepping), and L3 misses a
100 um zero-set line by 2-9 % on blended forms.

### 2. Versions: copy-on-write by file, undo and replay

*In the record by 11-05 (`fv8_field.FieldStore.commit`). Evidence: accepted in the workflow.*

- A version is a folder in exactly the format the renderer reads, plus `version.json`: its parent, the op log (plain
  parameter dictionaries, replayable), the changed bricks and tiles per level, stale pyramid slots, and a SHA-256 of
  every array. Its identity is a **field content hash**: SHA-256 over (name, SHA-256 of the file) for every file the
  renderer reads.
- **Committing writes a new folder.** Arrays the edit did not change are **hard links** to the parent's files (a plain
  copy where the volume refuses links). Changed arrays are written whole as new files through a temporary name and an
  atomic replace, never in place, so a parent can never be changed through a link. Rolling back is pointing at an older
  folder. A pointer file names the published version and keeps the previous entry.
- In memory, a store opens lazily with per-brick arrays memory-mapped read-only; edits live in an overlay (brick ->
  fp16 block) with the old blocks kept for exact undo.
- Measured 11-04 on two real crops (50 seeded smooth and displace edits each): undoing all 50 returns the original hash
  bit-exactly; replaying the log twice gives the forward hash; no brick changes outside an edit's support; edit time
  p50 2.3-22 ms, p95 4-61 ms over all levels; commit 0.12-0.24 s; the renderer's own reader matches the store's
  evaluation to 1e-18 m. Brick allocation where a moved surface leaves the band (new bricks seeded from the next
  coarser level, shared faces copied, tiles re-packed at commit): 0 missing-sample hits after a +2 mm move against 82
  without it; undo including the allocation is exact.
- Measured 11-12: removing 75 superseded versions freed about 154 GB. A hard link shares bytes only with versions that
  still exist, so each version costs what it changed until it is pruned.

### 3. The writer: one transaction per edit

*First built 11-10 to 11-11 (`fv8_write.py`, op `write_txn`, `48581bfc` 11-11 01:59); the store's default writer since
11-11 13:49 (`e788e94c`); the safe sign rule 11-12 18:47 (`dd32c23c`). Evidence: accepted in the workflow (the sign
rule itself: measured on named cases).*

**Why.** Four faults were measured on 11-10 with the earlier per-slab writer: new bricks seeded from one estimate while
the edit's delta was taken against another (up to 1.44 mm off on 727,000 nodes in one edit, showing as specks and
slits); a move written in a narrow band left coarse levels at the old shape (the renderer's coarse skip then cut slits
into moved skin); a brick sharing a face sample with an edited brick kept its old copy; capped samples were
re-estimated from levels an earlier slab had already changed (about 1 mm ridges and 2,926 near-surface sign errors in
one cavity wall). All four are one fault: the edit was not evaluated against one fixed source state.

**Steps.**
1. Input is a target field T on the L0 lattice (NaN means "leave this node"), or a delta against the editor's view
   taken inside the op before anything is written. All reads come from the pre-op store; all writes are staged and
   committed at the end as one record and one undo. The op chunks its own work (4,096 bricks at a time).
2. L0: every brick holding a target node receives it, in **every copy of a shared face sample**. A brick missing where
   the target enters L0's band is allocated inside L0's existing tiles; its other nodes take the same frozen estimate,
   so estimate + delta = target.
3. L1-L3 are rebuilt from the finest geometry. A coarse node that is a target node takes the target (exact
   restriction), except a target at L0's cap (sign only) or one that overstates the distance (capped to it). Every other
   coarse node within its level's cap of the changed surface takes sign x (distance to the new zero set - eps). The new
   zero set is a **cloud of lattice-edge crossings** of the finest level present, at L0 spacing, searched with a k-d
   tree. Points only; no surface is reconstructed. eps = 0.1 h0 + 1 um (26 um), so a coarse value does not overstate
   the true distance beyond the cloud's chord error and distance bounds taken from a coarse level stay conservative.
   Nodes whose distance moved less than 10 / 20 / 40 / 80 um (L0..L3) keep their exact bytes.
4. **The safe sign rule.** The sign of a rebuilt non-target node X comes, in order, from: (a) a target node k with
   s_k > |X - X_k|, where s_k = min(|T_k|, the new cloud's distance at X_k) - eps. A distance field is 1-Lipschitz, so a
   node lying s_k from the new surface has no surface within s_k, and any X closer than that is on k's side whatever
   the surface's shape. The 8 nearest target nodes are tried; if they disagree the node is counted and passed on.
   (b) Otherwise the store's own uncapped L0 value at X. (c) Otherwise the normal at the closest cloud point. A stored
   value on the side a target node proves wrong is never kept.
5. The op is a plain dictionary (data zlib-compressed, base64, with its SHA-256); replaying it gives the same bytes.

**Reversals on the way to this rule.** An 11-05 re-distance op built its zero set with marching cubes; it was removed
on 11-11 (`40c97886`) when the project became mesh-free everywhere, and old logged ops now replay through the point
cloud. An 11-11 sign order that put the closest point's normal first (`4f4d13e9`) fixed one stale-sign case and then
failed on 11-12 at rims and cavities, which is what rule (a) replaced.

**Measured.** On the case that exposed it (removing tissue from inside a separately owned eyeball-shaped part): before,
91 / 16 / 2 / 0 wrong-way samples at L0-L3 in a ring 0.47-1.11 mm outside the ball and 154,588 nodes of tissue left
inside it; with the rule, 0 at every level. On a sculpt write-back to the human model (11-11, CPU, about 65 s, 3.7 GB
peak): surface error at most 0.075 um at 19,750 points, 0 sign flips of 58,655, 393,728 untouched bricks
byte-identical, L1-L3 with 0 overstated distances; 167 L0 samples within 0.5 mm of the surface still overstate by more
than 0.1 h (at most 0.41 mm), because the editor's field there is stretched (gradient up to 2.76) and was kept so the
surface would not move. The engine regression set (13 tests) passes.

**What the certificate assumes, and what is not established.** Rule (a) is sound only if |T_k| does not overstate the
distance to the new surface and the cloud misses no crossing near X_k. The first is guarded (s_k takes the smaller of
two independent estimates) but not verified; a per-op measurement of the target's edge Lipschitz ratio is *proposed*.
The second is measured only on the crop cases; sheets thinner than one cell, which can cross a lattice edge twice with
no sign change, are **not tested**. The chord-error allowance is argued, not separately measured. Rules (b) and (c)
carry no certificate; (b) trusts existing data, which can itself be damaged. Writer correctness and repair of existing
damage are kept as separate questions: damage is found by the scans in section 5 and repaired as its own gated edit.

### 4. Keeping brick-face samples consistent: the seam guard and the publish gate

*Publish gate by content hash and region guard 11-05 (`fv8_current.py`); seam guard at commit and seam / duplicate
checks in the gate 11-12 10:39 (`8013917f`). Evidence: accepted in the workflow.*

- **A seam** is a node whose copies in neighbouring bricks differ. On 11-12 a scan of the published version found
  108,237 / 30,263 / 7,717 / 1,082 seam nodes at L0-L3, 8,899 of the L0 ones within a cell of the surface and 169 + 33
  with opposite signs. They had been written by earlier writer paths. They were repaired at the data (a finer level's
  uncapped value at L1-L3, otherwise the copies' midrange; L0 moved at most 29.5 um), 4 + 2 enclosed air bubbles were
  filled and 7,219 unreferenced duplicate brick rows dropped. The next version scans clean and was published on the
  condition that the cause be fixed too.
- **The cause proof.** Five real strokes written back through the editor: the pre-fix writer produced 4,053 L0 and 73
  L1 seams; the fixed writer (`8f2333eb`, 11-12 09:06) produced 0 at every level. One earlier unpublished write-back
  carried 9,051 new L0 seams of up to 5.4 mm.
- **The seam guard** runs inside every commit: every changed face sample is written into every brick that holds that
  node, whatever code staged the change; stale holders it repairs are counted in the report.
- **The publish gate** refuses a version unless: (1) every file the renderer reads is present and structurally
  consistent (brick counts, tile tables in range), with no seams and no duplicate rows; (2) a per-level diff against
  the published version shows changes only where the edit was meant to act and only in the intended direction (for a
  removal: inside -> outside only; other parts unchanged); (3) a **region guard** holds: for a protected region (a
  sphere round part of the face), per level, the bricks whose box meets the sphere are sorted by global brick
  coordinate and hashed as coordinate + sample + material bytes; any difference refuses the publish and names the
  level. For write-backs from the editor a further check compares the stored sign with the editor's field round every
  edit. Measured: the gate refused a removal whose old writer flipped 614 L0 and 87 L1 samples outside -> inside in a
  ring 12.6-16.5 mm from the part's centre (11-12 17:13); that refusal led to the safe sign rule. The earlier
  published version is now refused by the gate; the repaired one passes.

### 5. Level-consistency and coincident-surface scans

*First built 11-12. Evidence: measured on named cases.*

- **Level consistency.** For every coarse node, compare the sign the coarse level stores with the sign the finest
  level gives there, and group disagreements into connected islands (phantom tissue, phantom air). Read-only, 11 s for
  the whole body. Published version: 25,421 nodes in 186 islands, all in the head, the largest two filling the eye
  interiors. An earlier relative of this fault (11-04): the original builder wrote +cap inside the body wherever no
  source surface lay within the cap, giving millions of capped samples beside uncapped negatives; a sign-repair op
  fixed those from the level with the larger cap.
- **Coincident surfaces between owned volumes.** The body still held its own copy of each eyeball: about 6.4 cm3 of
  body inside each eye part (about 85 % of it) and about 1,500 mm2 where both surfaces lie within 0.05 mm (sampled on a
  0.1 mm grid). In close renders the two surfaces trade pixels (2,000-4,700 pixels per eye view hit within 0.2 mm of
  each other), seen as stepped rings and serrated lid margins that more render passes do not remove. The fix is at the
  data: body := max(F_body, -F_part) over the part's box, written through section 3 and gated by section 4. **Not yet
  done as of this text:** the re-save with the fixed writer and a small set-off between the two surfaces (its size to
  be chosen from render tests) are *proposed*.

### 6. The interactive editor: sculpt layer, card mirror, undo and save

*First built 11-08 (`54529cc3`, `7e732504`, `d69dcb28`); band admission and stroke-own sync 11-11 to 11-12. Evidence:
accepted in the workflow.*

- **Editable copy.** Opening the published version (2.0-2.5 s) gives an object whose field is
  F = F_store + dF_sculpt: the read-only store plus a sparse delta on the object's lattice, held as 8^3-node bricks in
  a pool with one row per brick. Brushes write the delta; writing back (section 3) makes a new version and never
  changes the open one.
- **Card representation.** The card holds the store, read in place by the rule of section 1, and an **atlas** of rows
  (9^3 fp16, 1,458 B each) for every brick that holds a change plus a ring of neighbours. A row holds the object's full
  current field there. Invariant: a cell that holds a change always has its row; everywhere else the store is the
  current field. The card holds the live field during a drag; at the release the stroke's change (F_end - F_start per
  touched brick) is written into the CPU sculpt layer, which is the authority for undo and save.
- **Band admission** (11-11/12). A stroke whose dabs change only nodes near the surface admits only cells whose centre
  lies within (largest reach + the cell's half-diagonal + the batch's summed move) of the surface. Before it, a 60 mm
  Smooth and a 45 mm Inflate under mirroring admitted about 378,000 rows and ran the card out of memory. An 11-11
  attempt at a band-only write-back failed: the CPU kept stale deep values whose sign later went wrong (flakes), and it
  was withdrawn.
- **Memory.** The sculpt pool cost 5.6-6.4 KB a brick (delta 2,048 B plus mask, direction and region arrays that are
  almost always zero) and grew by doubling with a transient copy: 3.3 GB after five body strokes. On 11-12 08:44
  (`0fb997e4`) growth became copy-free (address space reserved once, committed as needed) and the side arrays cost
  nothing until written: 3,332 -> 625 MB, peak 17.7 -> 13.8 GB. The card atlas is 1.17 GB at open and 2.35 GB after
  body strokes; a dense host cell-to-row map takes 1,063 MB (step 3 of section 7 removes it). The app holds about
  5.9 GB of private memory and 2.7 GB of card memory with the model open.
- **Undo journal** (11-11 13:49, `e788e94c`). Each step's undo records go to a per-session file once final; RAM keeps
  the newest 512 MB; a save carries up to 2 GB of the newest steps; opening rebuilds the undo stack from them with no
  replay. Measured over 30.6 minutes and 544 card strokes: ten reopens of 1.4-3.2 s, hash-verified, with up to 600
  undo steps attached, undo and redo exact, no monotonic memory growth.
- **Save** (chunked save 11-08, `af613975`; interrupted-swap recovery 11-11; never cut short 11-12 19:01, `06464ae8`).
  A project is a folder of sculpt chunks (only changed chunks re-compressed, the rest hard-linked from the previous
  save), stroke results, the undo journal and `project.json` with a state hash last. It is written to `<name>.tmp` and
  swapped in by rename, retried while the operating system holds a handle, with the previous save restored if the swap
  cannot finish. Opening recomputes the hash and refuses a mismatch. **A failure seen in real use:** on 11-12 a
  save followed at once by quitting was killed with the process (the writer ran on a daemon thread), leaving no
  project file and no automatic save; the work was recovered from the temporary folder. Since then a save is counted
  as pending from request to the writer's end, quitting waits for it, and a failed write keeps the window and the
  automatic save. After: the close waited 1.4 s and the project reopened with the same hash.
- **Checks kept in the tool.** A debug check compares every card brick with the CPU field at its own nodes (tolerance
  50 um): clean after each of 12 repeated Smooth strokes on an eyelid. Those 12 strokes lower a 1.54 mm bulge to
  0.45-0.49 mm and stall; none raises it again. The report from hand use that a smoothed eyelid bulge reappeared on
  screen is therefore **not reproduced and not explained** yet; the working hypothesis (an edited brick losing its atlas row,
  so the store's original shape is drawn there) is untested. The report of lag growing after 5-10 minutes is also open:
  a 16-stroke run showed none, and the longer run was not finished.

### 7. The scaling study: what a stroke costs and where the store goes next

*Study run 11-12 07:45-09:50 (`353e664e`, `95a00abb`, `9e442f18`). Evidence: measured on named cases; steps 3-4 are
proposed.*

**Method.** Every stage of a stroke was wrapped with a timer in the real window (offscreen, card brushes and tracer,
scripted drags of 0.5-1 s, two or three runs each, other jobs beside it) and marked by what its cost grows with: the
brush's footprint, the pixels changed, or what is loaded.

**Findings.**
- The same strokes on the same 120 mm sphere, as the scene grew from 1 object to 32 objects to one more object carrying
  212,000 sculpt bricks: Draw r 6 mm fell from 25-26 fps to 11-12 to 1.6-1.9; Smooth r 40 from 17-20 to 6-11 to 0-2.
  The bricks written were identical at every load.
- At the largest load, load-bound bookkeeping took 0.6-0.9 s of each stroke and the brush's own work 0.1-0.3 s. The
  load-bound stages: a checksum of every sculpt brick to find what changed (about 280 ms a call, twice a stroke), a
  scene-wide index re-flattened, an object's whole tile tables re-uploaded after any admission (20-36 ms a call on the
  human model, 0.5-1.4 s on a body stroke), an object's whole key dictionary scanned (160-460 ms), growth by copying.
- They come from one structural fact: two authorities (the card during a drag, the CPU layer after), kept consistent
  by scanning, plus dense per-object maps.
- Footprint costs are volumetric where they could follow area: on the human model a body Flatten r 40 wrote 90,486
  bricks of which 82 % lay more than 4 mm from the surface.
- On a small stroke (eyelid Smooth r 4, about 400 bricks) the dab arithmetic is about 1 ms a frame; the rest is some
  15 host steps of 1-10 ms each.

**Prototype** (a study-only store, not the app: no tracer, guard, re-distance or colour). 8^3 fp16 bricks in fixed
16 MB pages (growth adds a page, nothing is copied); one open-addressing, linear-probing hash on the card mapping
(object, brick coordinate) -> slot; values only within 4 cells of the surface; brick borders read through the hash, so
no shared face copies exist; undo by copy-on-write per brick at its first touch in a stroke. Measured: 0.27-0.34 ms a
dab with 1 object and 31,000 bricks, 0.21-0.30 ms with 66 objects and 788,000 bricks; a forehead Smooth r 40 writing
10,378 bricks in 9-15 ms with a 4-7 ms copy to the host; undo exact on 9 of 9 strokes; Draw matches a dense reference
to one fp16 step; node-to-node steps across brick faces match those inside bricks; 75 pages added at 9 ms each. A
separate measurement indexed 424,000 bricks in 12.6 MB against the dense map's 1,065 MB, at 1.3 times the lookup time.
Known defects: it leaks the slots of undone new bricks, and it leaves open the tracer on a band store, large moves and
save.

**The four-step recommendation.**
1. Local bookkeeping in today's design: tables updated per changed entry, no whole-store sync at the press, the release
   sync from the stroke's own slots, snapshot and index only for what changed.
2. The whole turn on the card: plan, admission, dab, guard, re-distance and collision as kernels back to back with one
   host wait a turn, the brick lookup kept in one device function so step 3 can swap it.
3. The store's next shape: one card-side hash over paged pools, the card as the live copy with copy-on-write undo and
   one copy to the host per stroke, fine bricks only in the surface band and a coarse signed-distance lattice beyond
   it (the sign of unstored space comes from that distance, never from per-tile tags).
4. For worlds: chunked residency, brush levels for body-scale strokes, shared bricks for repeated objects, edit layers.

**Decision** (11-12 09:27): the full path; steps 1-2 now, step 3 right after step 2.

**Status at the time of writing.** Parts of step 1 landed 11-12 10:47 (`94ad156d`, `655c0fc2`): a body Flatten's table
time 1,359 -> 21 ms; mirror sync 266 -> 0 ms a body stroke; torso Draw 14.6 -> 23.6 fps; body Smooth 2.5 -> 5.8 fps,
with every card table equal to a full upload after every stroke. Two trial patches measured in the study: index
rebuild 31-38 -> 2 ms; body-stroke sync roughly halved. Steps 2 and 3 are not started. The study names step 3's
research risks plainly: empty-space stepping and sign consistency once deep values are not stored finely, and carrying
fine detail through body-scale moves. Neither has a prototype.

### 8. Benchmarks

**Brush scorecard** (11-11 11:54-12:06, `d7a4067a`). *Evidence: reproduced under a frozen configuration* (the tree of
commit `01bc419d` extracted to its own folder, two full runs). Fixed scripted cases, so the driver's skill is out of
it: a 500 mm slab, a bump on it, and a 240 mm bust on a 1 mm lattice; a straight drag of 4 radii in 1.0 s at 120 mouse
events a second; radii 5 / 20 / 60 mm at three strengths; each stroke undone after measuring (undo error 0.0 um in
every case). Results: 57-62 fps at r 5-20, 26-39 fps at r 60, 12-16 fps at r 80 with mirroring (Grab 5 fps). Draw,
Carve, Inflate and Grab matched their intended form; Smooth removed too little at large sizes (0.24-0.98 % a pass at
60 mm); Flatten, Clay and Pinch were scored broken (Flatten made a bump taller). Deposits repeat within about 1 % run
to run; worst frame gaps vary up to 2x with machine load. This scorecard predates the rebuild of Clay, Flatten and
Smooth that began on 11-11 at 13:30 (Part 5); it is the before-picture, not their present state.

**Full-session run on the human model** (11-12 02:44-03:21, live code frozen at one commit, `9f93e325`). *Evidence:
reproduced under a frozen configuration* for the first fault below (two fresh runs failed identically); the rest
*measured on named cases*.
- The first whole-body stroke (Smooth r 60, mirrored) ran the card out of memory at 6.9 GB and the engine then silently
  dropped the stroke's release. After three rounds of fixes, 20 of 21 body strokes complete, each with one exact undo
  step; host memory still climbed to 18-24 GB until the store change of section 6.
- Targets missed: 30 fps while a stroke moves was missed on every sculpt stroke (face-level brushes 6.7-24 fps, eyelid
  r 2.5-4 mm 2.6-14.8, lip corners 3.7-13.5, torso r 25-30 0-1.2); a next-press wait of 100 ms or less was missed on
  23 of 49.
- Targets met: a 15-minute face soak of 296 strokes kept memory flat after 4 minutes (card 6.40 GB throughout); save
  5.7 s for 339 chunks; reopen 3.0 s with an identical state hash, 4.2 s cold in a new process.
- The CPU fallback broke the 16 GB budget with one torso stroke (minutes of stall, 9-13 GB); after a band-only fix
  small strokes were 15-50x faster and a torso stroke still took about 50 s.

### 9. Parts being redesigned around named patents

On 11-12 the independent claims of the patents below were read beside our mechanisms, to the
standard that each concern name the claim element behind it and each design-around name the element that is absent.
These are flags from our own review, not conclusions that any claim is or is not met. **As of this text none of the four redesigns has landed; the
affected code is held out of any published code until it does.**

| our part | patent (status as read 11-12) | the element of concern | the planned way round, and what it removes |
|---|---|---|---|
| The capsule primitive (a sphere's distance at the sample's clamped projection onto a segment) | MERL US 8,265,909 (active, adjusted to 2031-02-23), claim 1 | the patent concerns the distance of a shape swept along a path, with an optimal path parameter found continuously per sample; flagged in our review; held out of the code pending redesign | not yet designed; brushes already apply separate fixed-centre dabs, which lack the continuous path parameter |
| Grab's brick admission (brick centres projected onto the drag segment with a clamped parameter) | same | same | not yet designed |
| The card tracer's tile skip (one entry per 32 mm tile per level saying slot or empty; empty tiles skipped to their exit) | Movidius US 11,680,803 and US 11,222,459 (active, about 2037), all seven independent claims | a first entry whose values say each coarse voxel is empty or occupied, with finer entries beneath, used in rendering; flagged in our review; held out of the code pending redesign | direction of the design-around: skip empty space by a stored distance value (the tracer already holds a distance-to-nearest-brick far field) instead of an empty tag |
| The band store (`core/bandstore.py`, 11-08), whose tile tables carry outside / inside tags | same | same | keep the sign of unstored space in a coarse signed distance, as in step 3 |

Open on our side: whether a coarse signed-distance lattice could itself be construed as the claimed set of occupancy
bits is recorded as a grey zone; validity against earlier published work was not assessed. Other readings carried from
the same check, as constraints on step 3: nothing is paged onto the card on a lookup miss and the pool stays the data's
home (NVIDIA US 9,235,512); no per-brick lists of distance functions or edit primitives (MERL US 9,122,270 and
US 8,010,328); the coarse level is its own lattice, never built by merging fine bricks into parents in the same hash
(JD.com US 10,706,611). A screening search found no patent on a hash over paged pools, copy-on-write brick undo or
discrete-dab card sculpting; that is a screen, to be repeated, and not a clearance. Two expired families (Caltech
US 7,542,036 and US 9,430,870, level-set editing) appear to be past full term; not confirmed.

### 10. Published work this part learned from

Only sources the project records name. Papers and talks were read; no code from any of them is in the store.

| source | what it taught | code licence as recorded |
|---|---|---|
| Frisken, Perry, Rockwood, Jones, *Adaptively Sampled Distance Fields*, SIGGRAPH 2000; Perry and Frisken, *Kizamu*, 2001 | resolution where detail is; local regeneration after an edit | papers only (the 2000 paper's PDF was not re-read during the study) |
| Museth, *VDB*, ACM TOG 2013; OpenVDB | shallow fixed-depth tree, constant tiles, size-independent insert and delete | Apache-2.0 (MPL-2.0 before v12) |
| Museth, *NanoVDB*, 2021; Sifakis et al. 2026; Williams et al., *fVDB*, 2024 | card read-mostly layouts; topology changes rebuild the grid | Apache-2.0 |
| Hoetzlein, *GVDB*, HPG 2016 | card brick atlas with aprons and incremental insert | Apache-2.0 |
| Niessner et al., *Voxel Hashing*, ACM TOG 2013; ASH, stdgpu, nvblox | one card hash of brick coordinates over a fixed heap | original code CC BY-NC-SA (study only); stdgpu and nvblox Apache-2.0, Open3D MIT |
| Whitaker 1998; Museth, Breen, Whitaker, Barr 2002; Adalsteinsson and Sethian 1995 | narrow-band level sets and editing operators | papers only |
| Evans, *Learning from Failure* (Dreams), 2015; Aaltonen, Claybook, GDC 2018 | edit lists culled per cell; a thin band rewritten per edit | talks only, proprietary |
| Careil, Billeter, Eisemann, *HashDAG*, 2020 | copy-on-write edits with old roots kept for undo; pages on demand | MIT |
| Blender sculpt (2024 refactor notes) | per-node undo and dirty bits; brush rules used as the benchmark's peer | GPL (read as a study; principles only) |
| Hart, *Sphere Tracing*, 1996 | stepping by the distance value | paper |

General techniques used without a single source: trilinear interpolation, k-d tree nearest-point search, SHA-256
content hashing, hard links, atomic replace-by-rename.

---

---
[Contents](README.md) | [Complete single file](../RESEARCH.md)

