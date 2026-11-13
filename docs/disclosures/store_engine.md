# Field Studio: the field store and its engine (technical disclosure)

Prepared 2026-11-12 (Pacific time) for the first public research release of Field Studio, a mesh-free sculpting,
simulation and rendering tool in which every form is an implicit field: a signed distance field (negative inside,
positive outside, in metres) stored on a sparse, multi-level lattice. No triangles, quads, tetrahedra or element meshes
are made anywhere: not in the model, the editor, the writer, the renderer or the tests. This file records how the store
is laid out, how edits are written into it without breaking it, how its integrity is checked, how the interactive
engine shows and saves it, and how the levels are scanned for consistency, so that the record is public prior art.
Each mechanism carries one status: **BUILT AND MEASURED** (a number is cited), **BUILT** (in use, not separately
measured here), or **PROPOSED**. Dates are from the project's own version history (git) unless stated. Measurements
are on one full-body human model at 0.25 mm finest spacing unless stated.

Two sections touch areas where others hold patents (ray tracing of sparse distance-field grids; tile-skipping with
banded storage). They describe only what we built and say so where a part is under review and may be redesigned.
Nothing here claims a published method as ours; those are cited as prior work at the end.

---

## 1. The brick lattice and its four levels
**First built:** 2026-11-04 (the tiled full-body field; 4 levels), in the repository from 2026-11-05.
**Status:** BUILT AND MEASURED.

- Space is cut into cubic **tiles** of 32 mm. Each tile may hold, per level, a grid of **bricks**; a brick is 8 x 8 x 8
  lattice cells stored as **9 x 9 x 9 samples** (the ninth sample is the shared face with the next brick, so trilinear
  interpolation inside a brick needs no neighbour lookup).
- Four levels, each a factor 2 coarser than the one below:

  | level | node spacing h | brick edge | band kept around the surface | cap (largest stored distance) |
  |---|---|---|---|---|
  | L0 | 0.25 mm | 2 mm | 1.5 mm | 3 mm |
  | L1 | 0.5 mm | 4 mm | 2 mm | 3 mm |
  | L2 | 1 mm | 8 mm | 4 mm | 6 mm |
  | L3 | 2 mm | 16 mm | 8 mm | 12 mm |

  L3 and L2 cover the whole body (L3 its inside too); L1 and L0 are stored only in a band around the surface, in the
  regions that carry them. On the model:
  218,766 L0 bricks in 444 tiles, 72,955 L1, 64,609 L2, 16,070 L3.
- Samples are stored as 16-bit floats. A sample at the cap (|v| >= 0.999 cap) means "farther than the cap: no distance
  known here", never a distance.
- **Evaluation rule (one rule everywhere: CPU, card, writer, scans).** At a point, the finest level that holds a brick
  there answers, trilinearly inside that brick. A level whose tile holds no brick there gives only a bound (its band
  minus h) and the next coarser level is read; with no level at all the point is outside by at least the largest bound.
  A capped sample takes the finest coarser level's uncapped value at that sample (so an edit that moves the surface past
  a band edge still edits a true distance). Built 2026-11-08 as the editor's base node (`core/fv8base.py`); the same
  rule is evaluated on the CPU (numba) and on the card (CUDA).
- **Contract between levels:** a coarse node is the exact restriction of the finest data at that position (L1..L3 nodes
  are L0 nodes). Every writer and every check below enforces or measures that contract.

## 2. The sparse store and its versions (hard-linked, copy-on-write by file)
**First built:** by 2026-11-05 (`fv8_field.FieldStore.commit`). **Status:** BUILT AND MEASURED.

- A version is a folder: `meta.json`, the tile grid, and per level the sample array, brick coordinates, tile table
  (and material / attribute arrays). Its identity is a **field content hash**: sha256 over (name, sha256(file)) of every
  file the renderer reads.
- **Committing an edit writes a NEW version folder.** Arrays the edit did not change are **hard-linked** from the
  previous version (a file-level copy-on-write; a plain copy where the volume refuses links); only changed arrays are
  written. So a version costs only what it changed, every older version stays intact and openable, and rolling back is
  pointing back at an older folder. Parts of a model (separately owned volumes) are carried into the new version the
  same way. A pointer file names the published version, with the previous entry kept for rollback.
- Measured (2026-11-12): removing 75 superseded versions freed ~154 GB. A hard link shares bytes only with versions
  that still exist, so pruning old versions frees what each of them changed.

## 3. The writer: one transaction per edit, levels rebuilt, signs made safe
**First built:** 2026-11-10/11 (`fv8_write.py`, op `write_txn`); safe sign rule 2026-11-12 18:47.
**Status:** BUILT AND MEASURED.

**Why one transaction.** Four faults were measured before it (2026-11-10): new bricks started from one estimate while
the edit's delta was taken against another (up to 1.44 mm off on 727 k nodes in one edit: specks and slits); a move
written in a narrow band left the coarse levels at the old shape (the renderer's coarse skip then cut slits into moved
skin); a brick sharing a face sample with an edited brick kept its old copy (a whole node column unchanged); capped
samples re-estimated slab by slab from levels a previous slab had already changed (~1 mm ridges, 2,926 near-surface
sign errors in one cavity wall). All four are one fault: the edit was not evaluated against one fixed source state.

**The contract.**
1. Input: the TARGET field T on the L0 lattice (NaN = leave the node), or a delta against the editors' view taken inside
   the op before anything is written. Every read is from the pre-op store; every write is staged and committed at the
   end (one record, one undo). The op chunks its own work (4,096 bricks at a time), so callers never slab or tile.
2. L0: every brick holding a target node gets it, in **every copy of a shared face sample**. A brick missing where the
   target enters L0's band is allocated; its other nodes take the same frozen estimate, so estimate + delta = target.
3. **L1-L3 are rebuilt from the finest geometry.** A coarse node that is a target node takes the target (exact
   restriction). Every other coarse node within its level's cap of the changed surface takes sign x (distance to the new
   zero set - eps). The new zero set is a **cloud of lattice-edge crossings** of the finest level holding each region,
   sampled at L0 spacing, searched with a k-d tree (points only, no surface reconstruction); eps = 0.1 h0, so a coarse
   value never overstates the true distance by more than the cloud's chord error and any distance bound the renderer
   takes from a coarse level stays conservative. Nodes whose distance moved less than a per-level tolerance
   (10 / 20 / 40 / 80 um for L0..L3) keep their exact bytes. Missing coarse bricks where the new surface enters a level's
   band are allocated.
4. **The safe sign rule** (2026-11-12). The sign of a rebuilt non-target node X comes, in this order:
   (a) from a target node k with s_k > |X - X_k|, where s_k = min(|T_k|, the new cloud's distance at X_k) - eps.
   A distance field is 1-Lipschitz: a target node lying s_k from the NEW surface has no surface within s_k of it, so
   any X closer to X_k than s_k is on k's side, whatever the surface's shape (rims, cavities and edges included, where a
   closest point's normal is the wrong witness). The K = 8 nearest target nodes are tried; if their deciding answers
   disagree the node is left to the next rules and counted;
   (b) else the store's own uncapped L0 value at X;
   (c) else the normal at the closest point of the cloud.
   A stored value on the side a target node proves wrong is never kept. Measured on the case that exposed it (removing
   tissue from inside a separately owned ball-shaped part): before, 91 / 16 / 2 / 0 wrong-way stored samples at L0-L3 in
   a ring 0.47-1.11 mm outside the ball and 154,588 nodes of tissue left inside it in the editors' view; with the rule,
   0 wrong-way flips at every level and in the view.
5. The op is a plain dictionary (data zlib + base64 float32 with its sha256): replaying it gives the same bytes.

**3a. What the safe sign rule assumes, where each assumption is checked, and what would falsify it** (written
2026-11-12 after an outside review asked for exactly this). The certificate is: if no point of the new surface lies
within s_k of X_k, the open ball of radius s_k around X_k lies on one side of the surface, so every X with
|X - X_k| < s_k shares the sign of T_k. It is sound only when s_k <= the true distance from X_k to the new surface and
the sign of T_k is the true new sign at X_k.

| # | assumption | why it can fail | where it is checked now | what would falsify it | status |
|---|---|---|---|---|---|
| A1 | T_k's SIGN is the new sign at X_k (the target is authoritative there) | the editor's field and the store disagree at the node | the post-write gate compares the written store's sign with the editor's field around every edit (G2) | any node in G2 where the stored sign differs from the editor's | BUILT AND MEASURED (gate) |
| A2 | \|T_k\| does not OVERSTATE the distance to the new surface (the target is a distance or a lower bound: Lipschitz constant <= 1 in the store's metric) | a sculpt that stretches or compresses the field without re-distancing leaves \|grad T\| > 1 on its compressed side, where \|T_k\| reads larger than the distance; a non-uniform object scale breaks the metric | not by itself: guarded by A3's independent estimate (s_k takes the smaller of the two) | an edge of the target lattice with \|T_i - T_j\| > \|X_i - X_j\| (1 + tol) beside a certifying node; or a certified node whose sign G2 rejects | GUARDED, NOT VERIFIED; a per-op measurement of the target's edge Lipschitz ratio, shrinking s_k by it where it exceeds 1, is PROPOSED |
| A3 | the writer's own crossing cloud has a point within its chord allowance of every crossing of the new surface near X_k, so (cloud distance - allowance) <= the true distance | a crossing on a lattice edge between a target node and a left node can be missed; a sheet thinner than h can cross an edge twice and leave no sign change | indirectly: s_k = min(\|T_k\|, cloud estimate) - eps, so a miss matters only where A2 also overstates | the crop test (a removal inside a ball-shaped part, an addition beside it) reporting a wrong-way flip; a certified node whose sign G2 rejects | BUILT AND MEASURED on the crop cases (0 wrong-way flips at L0-L3); thin sheets NOT TESTED |
| A4 | truncation is conservative: a target value at the cap means "at least the cap", never more | a capped value treated as an exact distance would still be a lower bound; the risk is only a value ABOVE the cap | candidate search is limited to the L0 cap, so no certificate reaches past it | a stored or target magnitude above its level's cap | BUILT |
| A5 | the margin eps (0.1 h0 + 1 um = 26 um at h0 0.25 mm) covers fp32 target rounding and the cloud's crossing interpolation | a crossing interpolated linearly along an edge of a curved field lies off the true surface by up to its chord error | part of the allowance per cloud point plus eps | a measured chord error above the allowance on a curved test surface | BUILT; the bound is argued, not separately measured |
| A6 | the target and the store refer to the same parent geometry (no stale geometry) | an editor opened on one version writing into another | versions are immutable folders identified by content hash; an edit always writes a NEW version from the version it opened, and the base node refuses a store whose id changed since it was recorded | a write whose parent hash differs from the editor's recorded source id | BUILT |
| A7 | X and X_k are exact lattice positions (no interpolation in the certificate itself) | none at nodes; between nodes the drawn surface is the trilinear zero set of the stored signs and values | by construction (coarse nodes are L0 nodes) | n/a | BUILT |

When the K = 8 nearest candidates disagree, the node is not certified (it is counted and left to the next rules), so the
choice of K affects how many nodes are certified, never their correctness. The later rules (b: the stored uncapped L0
sign; c: the closest cloud point's normal) carry NO certificate: (b) trusts the existing data, which can itself be
damaged. **Writer correctness and repair of existing data are kept as separate questions:** the writer never
reconstructs a history from a damaged field; existing damage is found by the scans of section 5 and repaired at the
shape as its own, gated edit.

## 4. The seam scan and the publish gate
**First built:** publish gate by field hash and region guard 2026-11-05 (`fv8_current.py`); whole-body seam scan
2026-11-12. **Status:** BUILT AND MEASURED.

- **Seam.** A sample on a brick face is stored once per brick that holds it; a seam is a node whose copies differ. The
  scan reads every level of a version, lists every seam node, its copies, its closeness to the surface, whether the
  copies agree in sign, and the value the finer levels give there (the restriction contract). Measured on one version
  before repair: L0 108,237 / L1 30,263 / L2 7,717 / L3 1,082 seam nodes, 169 + 33 with opposite signs; repaired at the
  shape (a finer level's uncapped value at L1-L3, else the copies' midrange near the surface, L0 moved <= 29.5 um),
  plus 4 + 2 enclosed air bubbles filled and 7,219 unreferenced duplicate rows dropped; the next version scans clean.
- **The publish gate** refuses a version unless: (1) every file the renderer reads is present and structurally
  consistent (brick counts per level, tile table in range), with no seams and no duplicate rows; (2) a store diff
  against the published version at every level shows that changed samples lie only where the edit was meant to act and
  move only the intended way (for a removal: inside -> outside only, in the intended regions; other parts unchanged);
  (3) a **region guard** holds: for a protected region (a sphere around part of the face), per level, the bricks whose
  box meets the sphere, sorted by global brick coordinate (independent of storage order), hashed as coordinate + sample
  + material bytes exactly as stored, against a recorded guard; any difference refuses the publish and names the level.
  Measured: the gate refused a removal edit whose old writer flipped 614 L0 / 87 L1 samples outside -> inside in a ring
  12.6-16.5 mm from the part's centre (2026-11-12 17:13); that refusal led to the safe sign rule above.

## 5. The level-consistency scan and the coincident-surface scan
**First built:** 2026-11-12. **Status:** BUILT AND MEASURED.

- **Level consistency.** For every coarse node the scan compares the sign the coarse level stores with the sign the
  finest level holding data there gives, and groups disagreements into connected islands (phantom tissue: coarse says
  inside, fine says air; phantom air: the reverse). Measured on the published version: disagreements only around the
  eyes, 25,421 nodes in 186 islands.
- **Coincident surfaces between owned volumes.** When a body and a separately owned part (an eyeball) both describe the
  same surface, the renderer must choose between two surfaces at almost the same depth per pixel; the result is stepped
  contour rings and serrated edges that more render passes do not remove. The scan samples both fields on a 0.1 mm grid
  over the part's box and counts points where both |F| < 0.05 mm, and the volume where the body is inside the part.
  Measured: per eye, ~1,500 mm2 of coincident surface and ~6.4 cm3 of body inside the eyeball (~85 % of it); in the
  rendered frame, 2,000-4,700 pixels per close eye view where the two objects are hit within 0.2 mm of each other
  (a per-pixel object-id map compared with neighbours). The fix is at the data, not the renderer: the body gives up the
  part's volume (body := max(F_body, -P_part) over the part's box), written as a new version through the writer of
  section 3 and gated by section 4; a small set-off between the two surfaces is PROPOSED so they never coincide (its
  size to be chosen from render tests).

## 6. The interactive engine: an editable copy, a sculpt layer, and the card mirror
**First built:** 2026-11-08; band admission and stroke sync 2026-11-11/12. **Status:** BUILT AND MEASURED.

- **Editable copy.** Opening the published version gives an object whose field is
  F = F_store(q) + dF_sculpt(q): the read-only store (section 1's rule) plus a **sculpt layer**, a sparse delta on the
  object's own lattice (0.25 mm), held as 8^3-node bricks in a pool with one row per brick. Brushes write the delta;
  writing back to the store (section 3) makes a new version and never changes the open one.
- **The card mirror.** The graphics card holds (a) the store itself, read in place with the same rule, and (b) an atlas
  of rows for every brick that holds a delta, plus a ring of neighbours. A row's values are the object's full current
  field there. Invariant: a cell that holds a change always has its row; everywhere else the store is the current field.
  At a rebuild every non-zero sculpt brick (+1 cell) is admitted.
- **Band admission** (2026-11-11/12): a stroke whose dabs change only nodes within a reach of the surface admits only
  cells whose centre lies within (largest reach + the cell's half-diagonal + the batch's summed move) of the surface;
  cells beyond keep reading the store. Measured: a 60 mm Smooth and a 45 mm Inflate under mirror previously admitted
  ~378 k rows and ran the card out of memory; after, the card grows only with the band.
- **A stroke's end** writes delta += F_end - F_start for every brick it touched into the CPU sculpt layer, then marks
  exactly those bricks current on the card (no full resync). Measured on the full body: a torso Draw 14.6 -> 23.6 fps,
  a body Smooth 2.5 -> 5.8 fps, the mirror sync 266 -> 0 ms per stroke. A debug check compares every card brick with
  the CPU field at its own nodes (bricks off by > 50 um, allowing half an fp16 step); measured clean after each of 12
  repeated Smooth strokes on the eyelid (2026-11-12, this lane).
- **Repeated smoothing converges, it does not return.** Measured 2026-11-12 on the CPU path and on the card: 12
  Smooth strokes (r 6 mm) over an upper-eyelid bulge lower it from 1.54 mm to 0.45 / 0.49 mm and then stall; no stroke
  raises it again; release to the next stroke 0.45-0.6 s.

**Patent note (sparse-grid tracing and tile skip).** The card traces rays through the store's tile grid and uses each
level's band bound to skip empty space, and stores only banded bricks per tile. Related patent families exist (e.g.
ray tracing of sparse SDF grids, US 11,823,319 and family; tile skipping over a banded store). This disclosure records
what we built, as above, with dates; whether any claim reads on these parts is under review, and if one does the
affected part will be redesigned (for example, bounds taken from our own conservative coarse rebuild of section 3
rather than from a per-tile occupancy structure). Nothing here is offered as a licence or a claim of non-infringement.

## 7. The view: progressive light on the card
**First built:** 2026-11-08. **Status:** BUILT.

The viewport renders the fields directly (no mesh extraction): each pixel takes passes until it reaches 16 (the
convergence target); during interaction a preview frame is drawn and passes resume when the view rests. RGB and the
per-pixel object id are produced on the card; only what the viewport draws crosses the bus (rgb as 8-bit, ids as int16,
depth). A full-resolution, converged still with hair from the window is PROPOSED (not built yet).

## 8. Save integrity: atomic folders, a write that is never cut short, verified on open
**First built:** atomic folder swap 2026-11-08 (chunked project) and 2026-11-11 (interrupted-swap recovery); the save
is never cut short 2026-11-12 19:01. **Status:** BUILT AND MEASURED.

- A project is a folder: sculpt chunks (only changed chunks re-compressed; unchanged ones hard-linked from the previous
  save), card stroke results, undo journal, then `project.json` with the **state hash** last. It is written into
  `<name>.tmp`, then swapped in (`<name>` -> `<name>.old`, `.tmp` -> `<name>`, `.old` removed), retrying while the
  operating system holds a handle and putting the previous save back if the swap cannot finish; an interrupted swap is
  recovered at the next save or open. Opening recomputes the scene's state hash and refuses a mismatch.
- **A save is pending from the moment it is requested to the moment its writer finishes** (a counter set when the
  save is posted and cleared at the writer's end; the queue alone missed a save the engine had taken off the queue but
  not yet started). The engine counts it as work in progress, the writer thread is not a daemon (an exiting
  interpreter waits for it), quitting waits for it behind a "Saving..." dialog, and the engine's stop waits for it
  without a time limit. Every save reports its end (path, ok, hash, error); a failure is shown as SAVE FAILED, the work
  stays marked unsaved, the session's automatic save is kept and the window stays open. The automatic save is removed
  only after a confirmed save.
- Measured on a real 5-object session (2026-11-12): before, "save then close at once" left `<name>.tmp`, no
  `project.json` and no automatic save; after, the close waited 1.4 s, the project reopened with the same state hash;
  with a forced write failure on Quit, the window stayed open and the automatic save was kept.

## 9. Session logs and the freeze watchdog
**First built:** 2026-11-12 18:06. **Status:** BUILT.

Each launch writes its own log (`session_logs/fs_<date>_<time>_<pid>.log`; the newest 40 kept), so a relaunch never
erases the record of a frozen session. A watchdog in a native thread checks that the window thread answers; when it has
not for more than 2 s it writes every Python thread's stack every 2 s (it works even while a compiled kernel holds the
interpreter lock), then one line with the stall's start, end and length.

---

## Prior work used (not claimed)
Signed distance fields and their 1-Lipschitz property; trilinear interpolation on regular lattices; sparse
hierarchical voxel storage in general (e.g. octree / tiled volume layouts in the literature); k-d tree nearest-point
search; atomic replace-by-rename for files; content hashing (SHA-256); hard links as file-level copy-on-write; sphere
tracing of distance fields (Hart, 1996). What is recorded above is how these are combined and constrained in this tool.
