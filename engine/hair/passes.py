"""Hair Studio passes (numpy, Apache-2.0): the style stage's acts on hair that ALREADY EXISTS (DESIGN_grow_then_style).

Why (10-31): the hair is grown once, then styled from there; as in real life, a hand or a brush run through
the hair acts after the hair is
already there. Until then the acts ran while the hair was growing (generate._grow, flow.order 'during'): act 1 of
look A combed each new length flat as it emerged (buildHS3/README 10-31). Here a tool is drawn through the grown head.

A stroke pass (stroke_pass): the tool's leading edge moves down every strand's material arc in graded steps (15 mm,
x1.25 per step, at most 80 mm: ~11-13 steps on 450-650 mm hair) - every strand from its own root at once ('parallel').
In its window (window_mm behind the edge, soft ends) and at EVERY depth down to the scalp (a hand reaches the scalp;
nothing may break when it does) it
  - turns each segment's heading toward the stroke in the skin's tangent plane (the bristles' heading pull, actions.py,
    with acts.pass_win in place of the reach from the skin), and
  - carries the hair along with it by the bristles' friction (acts.drag, actions.py): each sample in the window is
    pulled toward where the tool is - the brush's path from the strand's root over the head along the stroke field
    (Acts._trace_paths, at the sample's own arc) - in the skin's tangent plane only, and only beyond reach_mm (the
    depth of hair the brush holds), by a smoothed |x| whose force never exceeds C (the tool's tension x the guide's
    weight, shared over the window by fibre volume). No normal part: it cannot press hair onto the scalp or lift it
    off; beyond its grip the hair slips. It lives one step. (The first form pulled each sample one step along the
    stroke where the sample was: on the F20 head a curtain over the face was moved along the face, not back over the
    head - 15 % fewer nodes in front of the face, and the crown 6.3 -> 5.4 mm.) That is friction from a moving tool, not a pin (brushed hair is not pinned to
    the scalp: it stays in place through physics and its relationship with the hair around it).
Between steps the friction anchors are re-taken where the tool left the hair (settle.lag). After the last step the
tool is gone: two relax solves with lag_stick between - only the hair's weight, bending, wave, contact, pressure, oil,
product, set and friction act.

The tools' numbers (window, tension, give) are design values with no published source (a stylist's pull ~ a few guide
weights: dry.root_mm's note). They are set once per tool and never tuned per look - tuned per look, the drag would be
the old hold under a new name.

    run_passes(state, acts, st, log, probe=None)   the style stage's hook (generate flow.order 'after')
    sweep_pass(state, acts, tool, log, ...)         a tool moved through space along paths (the brush; a hand)
    hand_pass(state, acts, guides, log, ...)        a hand through its section along the section's paths
    stroke_pass(state, acts, tool, log, ...)        one tool drawn through the hair (the window down each strand;
                                                    run_passes' hands until 10-31 night)
    schedule(L_max)                                 the leading edge's arcs (m)
"""
import time

import numpy as np
from scipy.spatial import cKDTree

from . import actions as AC

# the tools (design values, frozen: DESIGN_grow_then_style 'A stroke as a pass')
TOOLS = {
    # reach_mm: how far from the tool's place along its path the hair it holds may lie before the bristles pull it -
    # the depth of hair in the brush. 30 mm covers the hair standing off the head on the way over the top (a strand
    # h above the skin trails the skin path by ~h x the angle travelled: 15 mm x ~1.8 rad = 27 mm), so the layer is
    # carried without being pulled down onto the skull; a curtain over the face (100+ mm off) is carried back
    "brush": {"window_mm": 40.0, "tension": 3.0, "give_mm": 1.0, "reach_mm": 30.0, "evals_step": 24, "evals_relax": 90},
    "comb": {"window_mm": 10.0, "tension": 2.0, "give_mm": 1.0, "reach_mm": 10.0, "evals_step": 24, "evals_relax": 90},
    "fingers": {"window_mm": 70.0, "tension": 1.0, "give_mm": 1.0, "reach_mm": 40.0, "evals_step": 24,
                "evals_relax": 90},
}


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / max(b - a, 1e-300), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def schedule(L_max, first=0.015, grow=1.25, cap=0.08):
    """the tool's leading edge along the strands (m): first, then each step x grow (at most cap), to L_max"""
    out, s, d = [], 0.0, first
    while s < L_max - 1e-9:
        s = min(L_max, s + d)
        out.append(s)
        d = min(cap, d * grow)
    return out


def _window(arc, s_lo, s_hi, r):
    return smoothstep(s_lo - r, s_lo, arc) * (1.0 - smoothstep(s_hi, s_hi + r, arc))


def stroke_pass(state, acts, tool, log=None, label="brush", probe=None, guides=None, step_probe=None):
    """one tool drawn from the roots to the ends of every strand (see the module notes). tool: a TOOLS entry (dict).
    The brush's own keys (strength, down_sides, lengths, noise, the stroke field) come from the style through acts.
    guides (G,) bool: the strands the tool goes through (a hand around the face); None = all. step_probe: a measuring
    hook, step_probe(step, samples in the window, where they were, where the tool is at their arcs, state). -> report"""
    log = log or (lambda m: None)
    t0 = time.time()
    rods, body = state.rods, acts.body
    G, K = rods.l.shape
    win = float(tool["window_mm"]) / 1000.0
    r = min(0.005, 0.25 * win)
    eps = float(tool["give_mm"]) / 1000.0
    tens = float(tool["tension"])
    seg_mid = rods.s + 0.5 * rods.l
    edges = schedule(float(np.max(rods.L)))
    face_style = bool(acts.v["face"]["barrier"])
    reach = float(tool.get("reach_mm", 0.0)) / 1000.0
    if acts.Qp is None:
        acts._trace_paths()                    # the brush's path from each root over the head along the stroke
    Qp = acts.Qp
    state.acts = acts                          # a head's settle is built with no acts (head.instantiate): the tool enters
    acts.brush_on, acts.face_on = True, face_style
    E0 = None
    rows = []
    prev = 0.0
    for k, s_hi in enumerate(edges):
        s_lo = max(0.0, s_hi - win)
        acts.pass_win = _window(seg_mid, s_lo, s_hi, r) * (rods.l > 0)
        if guides is not None:
            acts.pass_win = acts.pass_win * guides[:, None]
        view = state.view()
        acts.drag = None
        acts.lag(view)                         # the bristles' headings in the window; the face flags
        # the drag on the samples in the window, lagged for this step
        gi = state.sq // state.Ka
        ws = _window(state.sarc, s_lo, s_hi, r)
        if guides is not None:
            ws = ws * guides[gi]
        idx = np.nonzero(ws > 1e-3)[0]
        if len(idx):
            Y0 = view.Y[idx]
            _, g = body.sample(Y0, grad=True)
            n = _unit(g)
            j = state.sq[idx] % state.Ka                 # the sample's segment and place on it
            tt = state.st[idx][:, None]
            Q = Qp[gi[idx], j] * (1.0 - tt) + Qp[gi[idx], j + 1] * tt     # where the tool is at the sample's arc
            sv = state.sv[idx] * ws[idx]
            tot = np.bincount(gi[idx], sv, minlength=G)
            C = tens * rods.Wtot[gi[idx]] * sv / np.maximum(tot[gi[idx]], 1e-300)
            acts.drag = {"idx": idx, "a": Q, "n": n, "C": C, "eps": eps, "reach": reach}
        e_before = None
        if probe is not None:
            e_before = float(state.energies().get("total", np.nan)) if hasattr(state, "energies") else None
        state.solve(int(tool["evals_step"]), label="%s %d" % (label, k))
        if step_probe is not None and len(idx):
            step_probe(k, idx, Y0, Q, state)
        state.lag()                            # the tool lifted and laid the hair: friction catches it where it is
        if E0 is None and probe is not None:
            E0 = e_before
        rows.append((round(s_hi * 1000, 1), int(len(idx))))
        prev = s_hi
    acts.drag = None
    acts.pass_win = None
    acts.brush_on = acts.face_on = False
    # the tool is gone: only the hair and what it lies on act
    nrel = 2
    for i in range(nrel):
        state.solve(int(tool["evals_relax"]) // nrel, label="%s relax %d" % (label, i))
        AC.lag_stick(state)
    rep = {"tool": dict(tool), "steps": len(edges), "edges_mm_samples": rows, "seconds": round(time.time() - t0, 1)}
    log("  pass %s: %d steps, %.1f s" % (label, len(edges), rep["seconds"]))
    return rep


def _stop_at_body(body, Y0, A, h_c, n=16):
    """the tool's targets A (m, 3) for samples at Y0, stopped where the straight move Y0 -> A would first come within
    the contact distance h_c of the skin (checked every 1/n of the move on the body's grid, fine crops blended); a
    sample already in contact keeps its target. Why (11-01): a step's pull (up to 25 mm) drew strands straight through
    the ear - a 2-4 mm flap whose contact barrier, however stiff, is far smaller than the pull (a sample can sink only
    half its thickness before it is through) - and a threaded strand cannot be pushed back out: in the pass-3 trace
    the guide nodes inside the ear went 16 (rested) -> 66 (the brush) -> 107 (the ear hand), none came out after. A
    hand does not draw hair through cartilage: the hair stops at the ear and slides round it as the hand goes on; going
    over the ear or behind it (the tuck) is untouched."""
    f = np.linspace(0.0, 1.0, n + 1)[1:]
    S = Y0[:, None, :] + f[None, :, None] * (A - Y0)[:, None, :]
    sd = body.sample(S.reshape(-1, 3)).reshape(len(Y0), n)
    free = body.sample(Y0) >= h_c
    enter = sd < h_c
    hit = free & enter.any(1)
    if not hit.any():
        return A
    first = np.argmax(enter, axis=1)
    fr = np.where(first > 0, f[np.maximum(first - 1, 0)], 0.0)
    out = A.copy()
    out[hit] = Y0[hit] + fr[hit, None] * (A[hit] - Y0[hit])
    return out


def _lane_paths(acts, lanes, lift, off, ds, z_start=None):
    """the brush's paths for a sweep: for each lane guide, from `lift` below its root and `off` out from the face (the
    brush starts under the hair hanging there) straight up to the root, then along the stroke path the guide's root
    traces over the head and down (Acts._trace_paths), resampled every ds. z_start: the tool takes the path only from
    where it first goes below that height (a hand on the lengths, not on the head). -> B (n, M, 3), T (n, M, 3),
    valid (n, M)"""
    rods, body = acts.rods, acts.body
    Qp = acts.Qp
    arcs = np.concatenate([np.zeros((rods.G, 1)), rods.s + rods.l], 1)
    out_B, out_T = [], []
    for g in lanes:
        n_end = int(np.searchsorted(arcs[g], rods.L[g] + 1e-9))
        path = Qp[g, :max(2, n_end)]
        r0 = rods.X[g]
        if z_start is not None:
            below = np.nonzero(path[:, 2] < z_start)[0]
            if not len(below) or below[0] >= len(path) - 1:
                continue
            path = path[below[0]:]
            r0 = path[0]
        _, gr = body.sample(r0[None], grad=True)
        nr = _unit(gr)[0]
        start = r0 + np.array([0.0, 0.0, -lift]) + off * nr
        pre = start + (r0 - start) * np.linspace(0.0, 1.0, 8)[:-1, None]
        P = np.concatenate([pre, path], 0)
        sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        tq = np.arange(0.0, sl[-1], ds)
        B = np.stack([np.interp(tq, sl, P[:, c]) for c in range(3)], 1)
        T = _unit(np.gradient(B, axis=0))
        out_B.append(B)
        out_T.append(T)
    M = max((len(b) for b in out_B), default=1)
    Bs = np.full((len(lanes), M, 3), np.nan)
    Ts = np.zeros((len(lanes), M, 3))
    for i, (b, t) in enumerate(zip(out_B, out_T)):
        Bs[i, :len(b)] = b
        Ts[i, :len(t)] = t
    return Bs, Ts, np.isfinite(Bs[..., 0])


def sweep_pass(state, acts, tool, log=None, label="sweep", lift_mm=150.0, off_mm=15.0, step_mm=25.0,
               radius_mm=30.0, evals_step=16, probe=None, lanes=None, z_start=None, paths=None, root_lift_mm=0.0,
               set_at_end=False):
    """a brush moved through SPACE, the way a person brushes (10-31: a brush moves a certain way, not just
    following the hair, which would leave strands where they were, only untangled and cleaner): a row of brush
    heads starts below the front hairline and temples, out in front of the face (under the hair hanging there), and
    travels up to the hairline, back over the head and down along the stroke paths (back and down), step_mm at a time.
    At each step every hair sample inside a brush head (radius_mm, at any depth) is moved the step along the brush's
    direction by the bristles (acts.drag with the strand's direction): across its strand the hair is carried as by a
    solid (the bristles sit between the strands; the skin's own contact law on the sample's weight), along its strand
    it slides through them by friction (capped at the tool's tension); tangent plane only, gone after the step. Its
    segments' headings turn toward the stroke; hair outside the brushes is not touched. Then the brush is gone and the
    hair relaxes. (10-31 night, sweep_probe on the full head: with friction alone a held sample moved a median 8 % of
    the brush's step and 98 % were dropped mid-stroke - sheet 73 showed the brush lifting the hair and
    then letting it settle; with the bristles 27 %, 87-96 % on the way up the face, and one sweep left 2922 samples
    over the face instead of 4665.) probe: a measuring hook, probe(step, held samples, where they were, the pull,
    state). lanes: the guides whose paths the tool travels (default: the front hairline and temples - the brush); a
    hand passes its section's guides with lift_mm 0 and off_mm 0 (it starts at their roots) and its span as
    radius_mm. paths: (B, T, valid) given (the lengths hand's, _length_paths) instead of the lanes'. -> report"""
    log = log or (lambda m: None)
    t0 = time.time()
    rods, body, scalp = state.rods, acts.body, acts.scalp
    G = rods.G
    state.acts = acts
    if acts.Qp is None:
        acts._trace_paths()
    ds = step_mm / 1000.0
    if paths is not None:
        B, T, ok = paths
        lanes = np.arange(len(B))
    else:
        if lanes is None:
            din = scalp.d_inside_mm(rods.X)
            phi, _, _ = scalp.sph(rods.X)
            lanes = np.nonzero((din < 12.0) & (np.abs(phi) < 110.0))[0]
        lanes = np.asarray(lanes)
        B, T, ok = _lane_paths(acts, lanes, lift_mm / 1000.0, off_mm / 1000.0, ds, z_start=z_start)
    if not len(lanes):
        return {"tool": dict(tool), "lanes": 0}
    R = radius_mm / 1000.0
    eps = float(tool["give_mm"]) / 1000.0
    tens = float(tool["tension"])
    acts.brush_on, acts.face_on = True, bool(acts.v["face"]["barrier"])
    nsteps = B.shape[1]
    moved = 0
    if root_lift_mm > 0:
        acts.lift = None
    for k in range(nsteps):
        live = ok[:, k]
        if not live.any():
            break
        Bk, Tk = B[live, k], T[live, k]
        tree = cKDTree(Bk)
        view = state.view()
        Y = view.Y
        d, li = tree.query(Y, distance_upper_bound=R, workers=-1)      # every core: the same answers
        idx = np.nonzero(np.isfinite(d))[0]
        # the segments in a brush: their headings turn toward the stroke (the bristles)
        P = view.P
        M = 0.5 * (P[:, :-1] + P[:, 1:])
        dm, _ = tree.query(M.reshape(-1, 3), distance_upper_bound=R, workers=-1)
        win = np.zeros_like(rods.l)
        win[:, :state.Ka] = np.isfinite(dm).reshape(G, -1) * (rods.l[:, :state.Ka] > 0)
        acts.pass_win = win
        acts.drag = None
        acts.lag(view)
        if len(idx):
            gi = state.sq[idx] // state.Ka
            _, g = body.sample(Y[idx], grad=True)
            n = _unit(g)
            t = Tk[li[idx]]
            t = _unit(t - (t * n).sum(1, keepdims=True) * n)
            sv = state.sv[idx]
            tot = np.bincount(gi, sv, minlength=G)
            C = tens * rods.Wtot[gi] * sv / np.maximum(tot[gi], 1e-300)
            # the bristles carry the hair across its strand (the skin's contact law on the sample's weight) and let it
            # slide along the strand by friction (C): the strand's direction at each held sample, now
            sdir = _unit(view.D.reshape(-1, 3)[state.sq[idx]])
            # a tool cannot draw hair THROUGH the body: where the straight move from a sample to where the tool takes
            # it would enter the skin, the sample is taken only to the skin (_stop_at_body)
            acts.drag = {"idx": idx, "a": _stop_at_body(body, Y[idx], Y[idx] + ds * t, state.h_c), "n": n, "C": C,
                         "eps": eps, "reach": 0.0, "d": sdir, "kp": state.k_c * state.sw[idx]}
            if root_lift_mm > 0:
                # the fingers under the roots (11-10): the section held out from the scalp - a held sample at arc s is
                # kept at least root_lift_mm x min(s / over, 1) off the skin, over_mm = dry.over_mm (0 at the follicle,
                # which sits in the skin; the full lift where the fingers hold it), out to 1.5 over_mm (the fingers'
                # width); actions.Acts.lift, a one-sided push like the face barrier. (First version 11-10 asked the
                # full lift AT the root and none at over_mm: impossible at the follicle, it held 5-6 mm.)
                over = float(acts.v["dry"]["over_mm"]) / 1000.0
                qj = state.sq[idx] % state.Ka
                sarc = rods.s[gi, qj] + 0.5 * rods.l[gi, qj]
                w = np.where(sarc <= 1.5 * over, np.clip(sarc / over, 0.0, 1.0), 0.0)
                on_ = w > 0
                # a root once lifted stays held until the hand lets go (the fingers stay under the roots while that
                # section dries): the held set accumulates over the steps
                if on_.any():
                    hnew = (root_lift_mm / 1000.0) * w[on_]
                    if acts.lift is None:
                        acts.lift = {"idx": idx[on_], "h": hnew}
                    else:
                        allx = np.concatenate([acts.lift["idx"], idx[on_]]); allh = np.concatenate([acts.lift["h"], hnew])
                        o = np.lexsort((-allh, allx)); allx, allh = allx[o], allh[o]
                        first = np.concatenate([[True], allx[1:] != allx[:-1]])
                        acts.lift = {"idx": allx[first], "h": allh[first]}
                    acts.lift["k"] = state.k_c * state.sw[acts.lift["idx"]]
            moved += len(idx)
            state.solve(int(evals_step), label="%s %d" % (label, k))
            if probe is not None:              # a measuring hook: (step, samples in a head, where they were, the pull)
                probe(k, idx, Y[idx], ds * t, state)
            state.lag()
    acts.drag = None
    if acts.lift is not None and len(acts.lift["idx"]):
        # what the fingers hold at the end of the pass (11-10): the held samples' height off the skin vs the lift
        sd_, _ = body.sample(state.view().Y[acts.lift["idx"]], grad=True)
        log("  root lift: %d samples held, off the skin p50 %.1f mm (lift p50 %.1f mm)" % (
            len(acts.lift["idx"]), 1e3 * float(np.median(sd_)), 1e3 * float(np.median(acts.lift["h"]))))
    if set_at_end:
        # dried in the hand (dry.when 'with_hands', 11-10): the set from the shape the hand holds, before it lets go
        state.dry_done = int(AC.dry_set(state, acts))
        log("  dry set in the hand: %d joints" % state.dry_done)
    acts.lift = None
    acts.pass_win = None
    acts.brush_on = acts.face_on = False
    for i in range(2):
        state.solve(int(tool["evals_relax"]) // 2, label="%s relax %d" % (label, i))
        AC.lag_stick(state)
    rep = {"tool": dict(tool), "lanes": int(len(lanes)), "steps": int(nsteps), "samples_brushed": int(moved),
           "seconds": round(time.time() - t0, 1)}
    log("  pass %s: %d lanes, %d steps, %.1f s" % (label, len(lanes), nsteps, rep["seconds"]))
    return rep


def run_passes(state, acts, st, log=None, probe=None):
    """the style stage's acts as passes, in order (generate flow.order 'after'): for now the brush (brush.on) as a
    paddle-brush stroke pass. -> report"""
    log = log or (lambda m: None)
    out = {"passes": []}
    if bool(st["brush"]["on"]):
        for k in range(int(st["brush"].get("strokes", 1))):
            # the brush moved through space along the stroke (back and down), taking the hair it meets
            out["passes"].append(sweep_pass(state, acts, TOOLS["brush"], log, label="brush %d" % k))
        if acts.v["dry"].get("when", "after_hands") == "before_hands":
            # dried and set on the brush first (dry.when, 11-10): the hands below move set hair; generate._tail then
            # skips its own set (state.dry_done)
            state.dry_done = int(AC.dry_set(state, acts))
            AC.dry_hold(state, acts, log=log)
            out["dry_before_hands"] = state.dry_done
            log("  dry set before the hands: %d joints" % state.dry_done)
        # then the hands put the hair where it goes (10-31: a person guides the hair into position with the
        # hands): each a hand drawn through its section along its own path, then gone
        if bool(st["brush"].get("fingers_shoulders", False)):
            # the lengths first (a person throws them back before sweeping the front): gathered where they hang in
            # front of the shoulders and laid back over them (lengths_pass)
            out["passes"].append(lengths_pass(state, acts, acts.rods.L > 0.25, log, label="fingers lengths"))
        if bool(st["brush"].get("fingers_face", False)):
            # the hand over the forehead: the strands rooted near the front hairline in front (|phi| < 45 deg); the
            # temples are the ear hand's (over the ear's top and behind it, the way the reference has them) - drawn
            # by this hand their paths ran down the cheek in front of the ear, where the reference has no hair to
            # measure (10-31 night: the face hand took 520 -> 996 samples over the face)
            din = acts.scalp.d_inside_mm(acts.rods.X)
            phi, _, _ = acts.scalp.sph(acts.rods.X)
            near = (din < 40.0) & (np.abs(phi) < 45.0)
            out["passes"].append(hand_pass(state, acts, near, log, label="fingers",
                                           root_lift_mm=float(acts.v["brush"].get("fingers_face_lift_mm", 0.0)),
                                           set_at_end=acts.v["dry"].get("when", "after_hands") == "with_hands"))
        if bool(st["brush"].get("fingers_ears", False)):
            _paths_with(acts, tuck=True)              # the hand's path: over the ear's top and behind it
            side = acts.tuck_pin.any(1) if acts.tuck_pin is not None else np.zeros(acts.rods.G, bool)
            if side.any():
                out["passes"].append(hand_pass(state, acts, side, log, label="fingers ears"))
        _paths_with(acts)                             # the style's own path again
    return out


def _resample(P, ds):
    """a polyline (n, 3) resampled every ds -> (points, unit tangents)"""
    sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    tq = np.arange(0.0, sl[-1] + 1e-12, ds)
    B = np.stack([np.interp(tq, sl, P[:, c]) for c in range(3)], 1)
    return B, _unit(np.gradient(B, axis=0)) if len(B) > 1 else np.zeros_like(B)


def _length_paths(acts, guides, ds, rise_mm=20.0, behind_mm=40.0, drop_mm=150.0, clear_mm=8.0):
    """the lengths hand's paths: for each guide in `guides` whose length passes the shoulders' height IN FRONT of
    their crest line (actions.ridge_at), from that point of the strand (where the hair is) up and back over the crest
    (rise_mm above it) to behind_mm behind it, then down drop_mm, kept clear_mm outside the body; resampled every ds.
    -> (B, T, valid, guide index of each lane)"""
    rods, body = acts.rods, acts.body
    P = rods.positions()
    A = rods.node_arcs()
    paths, which = [], []
    for i in np.nonzero(guides)[0]:
        live = A[i] <= rods.L[i] + 1e-9
        Pi = P[i, live]
        yr, zr = AC.ridge_at(acts.ridge, Pi[:, 0])
        k = np.nonzero(np.isfinite(zr) & (Pi[:, 2] < zr + 0.03))[0]
        if not len(k) or k[0] == 0:
            continue
        k = int(k[0])
        p0 = Pi[k]
        if p0[1] >= yr[k]:                      # already at or behind the crest: the lengths hang there
            continue
        q1 = np.array([p0[0], yr[k], zr[k] + rise_mm / 1000.0])
        q2 = np.array([p0[0], yr[k] + behind_mm / 1000.0, zr[k] + 0.01])
        q3 = q2 + np.array([0.0, 0.0, -drop_mm / 1000.0])
        Bp, _ = _resample(np.stack([p0, q1, q2, q3]), 0.005)
        for _ in range(3):                       # outside the body (the shoulder, the neck)
            sd, g = body.sample(Bp, grad=True)
            bad = sd < clear_mm / 1000.0
            if not bad.any():
                break
            Bp[bad] += (clear_mm / 1000.0 - sd[bad])[:, None] * _unit(g[bad])
        paths.append(_resample(Bp, ds))
        which.append(i)
    if not paths:
        return np.zeros((0, 1, 3)), np.zeros((0, 1, 3)), np.zeros((0, 1), bool), np.zeros(0, int)
    M = max(len(b) for b, _ in paths)
    Bs = np.full((len(paths), M, 3), np.nan)
    Ts = np.zeros((len(paths), M, 3))
    for j, (b, t) in enumerate(paths):
        Bs[j, :len(b)] = b
        Ts[j, :len(t)] = t
    return Bs, Ts, np.isfinite(Bs[..., 0]), np.asarray(which)


def lengths_pass(state, acts, guides, log=None, label="fingers lengths"):
    """a hand gathers the lengths WHERE THEY ARE and lays them back over the shoulders (10-31 night): from each long
    strand's point at the shoulders' height in front of their crest, up and back over the crest to behind it, then
    down (_length_paths) - the fingers' sweep_pass on those paths, carrying the hanging lengths across their strands.
    (Measured on pass 2: the 48 guides over the face were front-hairline strands hanging in front of the ear down to
    the chest - the brush only slides up a hanging lock, and a hand drawn from the roots or from behind the ear never
    reaches lengths 100 mm in front of its path.) -> report"""
    tool = TOOLS["fingers"]
    B, T, ok, which = _length_paths(acts, np.asarray(guides, bool), 0.025)
    rep = sweep_pass(state, acts, tool, log, label=label, radius_mm=tool["reach_mm"], paths=(B, T, ok))
    rep["guides_in_front"] = int(len(which))
    return rep


def hand_pass(state, acts, guides, log=None, label="hand", z_start=None, root_lift_mm=0.0, set_at_end=False):
    """a hand drawn through its section (guides (G,) bool) along the section's own paths - the fingers' sweep_pass: the
    hand starts at the section's roots and moves along each strand's path (Acts._trace_paths as it stands: the style's,
    the tuck's over the ear, or over the shoulder), 25 mm a step; the hair within its span (the fingers' reach_mm) is
    carried across the strands as the fingers sit between them and slides along them by the fingers' friction (their
    tension); then the hand is gone. (10-31 night, hands_probe on the full head after the brush x3: the hands as
    stroke_pass - every window sample pulled by friction toward the path at its own arc, up to 225 mm away - closed a
    median 8-30 % of the gap per step, and the face hand laid side hair over the ears: rim cover 0.49 / 0.24 -> 0.68 /
    0.77.) -> report"""
    tool = TOOLS["fingers"]
    lanes = np.nonzero(np.asarray(guides, bool))[0]
    rep = sweep_pass(state, acts, tool, log, label=label, lift_mm=0.0, off_mm=0.0, radius_mm=tool["reach_mm"],
                     lanes=lanes, z_start=z_start, root_lift_mm=root_lift_mm, set_at_end=set_at_end)
    rep["guides"] = int(len(lanes))
    return rep


def _paths_with(acts, tuck=None, over_shoulder=None):
    """retrace the tool's path (Acts._trace_paths) with the tuck's route over the ear or the stroke's end over the
    shoulder - a hand's path - and put the style's own flags back"""
    v = acts.v
    keep = (v["tuck"]["on"], v["brush"].get("over_shoulder", False))
    try:
        if tuck is not None:
            v["tuck"]["on"] = tuck
        if over_shoulder is not None:
            v["brush"]["over_shoulder"] = over_shoulder
        acts._trace_paths()
    finally:
        v["tuck"]["on"], v["brush"]["over_shoulder"] = keep
