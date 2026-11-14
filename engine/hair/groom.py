"""Outcome-first groom: a styled head of hair BUILT DIRECTLY from measured target numbers (P16). New 11-10 night (hair2 lane).

Why this exists. A day of simulated styling (hands, brush, blow-dryer, set timing, finger lift; work/hair_lane CONTINUE.md
section 4) moved her front lift from 15 to 21 mm against the TARGET's ~25-35 mm and left loops, a pale part and a face lock:
the simulated tools are not the real tools (unless the physics is perfect, a simulated blow-dryer will not shape
hair as a real one does). Per P16 the style is the outcome: its envelope, root lift, part and sweep, layered depth, lock
structure, waves and layer cut are measured on the reference and built here as such. Causes (sebum and product binding,
the set, humidity, motion) are drivers of variation on top: lock tightness, wave amplitude, frizz, flyaways. Physics keeps
contact real: nothing enters her body (her field's signed distance) or her face corridor; it is never tuned until a shape
appears. The built shape is the hair's REST shape: a later motion solve holds it as the set, the way a real set holds.

Structure of a head (what a real styled head is):
  section  : which side of the part a root falls to (or the crown's radial fall) -> the sweep direction
  lock     : ~250 hairs rooted in one ~1.5 cm2 patch that travel together (one centreline, a ribbon frame)
  sub-lock : 4-8 per lock, their own small wave phase and tightness (what the eye reads as texture inside a lock)
  strand   : its own root, offset inside its sub-lock, a slow meander, its own length (textured ends); a share crosses
             to a neighbouring lock (the crossings real heads show over a darker interior); a few fly away
Depth: hair rooted higher and nearer the part lies OUTSIDE hair rooted lower (it falls over it), so each lock holds a
depth d in [0, 1] and travels at d x the local hair thickness off her scalp: the mass fills the envelope, no scalp shows
through, and the outer layer carries the lift.

    from engine.hair import groom as G
    body = G.BodySDF(A, lo, h)                       # her field on the hair layer's 3 mm lattice (fibre_layer.body_lattice)
    out = G.groom(roots, grow_len, spec, body, seed) # roots (n, 3) m, grow_len (n,) m -> dict(P (n, npt, 3), LEN, lock, sub, d ...)
Spec: DEFAULTS below (each value names its evidence); a style layer JSON overrides any key (dotted paths)."""
from __future__ import annotations

import copy
import time

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------------------------------------------------
# The outcome, as numbers. Units: mm and degrees unless the key says otherwise. Head frame: metres, z up, -y front, +x HER
# LEFT (M8: her left is on the screen's right in a front view). Evidence: TARGET_face_grok.jpg (1408 px), scale from her
# pupil-to-menton 117 mm = 430 px (3.67 px/mm), read 11-10 ~21:30 (work/hair_lane/target/MEASURE_g1.md).
DEFAULTS = {
    "head": {"z_eye": 0.191, "z_menton": 0.0742, "z_top": 0.304},
    "part": {
        "x_mm": -30.0,          # soft side part on HER RIGHT: the hair diverges ~1-3 cm right of the forehead midline; the
                                # big section sweeps up and over to her left (TARGET top crop); hair crosses it (no line)
        "y_front": -0.105,      # from the front hairline ...
        "depth_mm": 95.0,       # ... back over the top to the crown, where the fall turns radial
        "cross_mm": 14.0,       # roots this close to the part may fall to either side ...
        "cross_share": 0.25,    # ... this share of the LOCKS crosses, and single hairs cross on their own (strand_cross at
        "strand_cross": 0.60,   #     the line, falling to 0 at cross_mm): no visible part line (none shows in the TARGET)
        "small_thick": 0.55,    # her right (small) section lies lower than the swept left (TARGET: the top is higher on
        "small_lift": 0.55,     #   the swoop side, her right side's top sits close to the head)
        "crown_thick": 0.55,    # the radial fall at the crown spreads its locks: less volume behind the part than over it
        "back_w": 0.45,         # the sweep's backward share on top (the swoop rises up-back, then over)
        "back_w_small": 0.85,   # the small (right) section goes back more than over
    },
    "thick": {                  # hair thickness off the scalp (outer layer), by height: the measured envelope
        "z":  [0.30, 0.27, 0.23, 0.19, 0.12, 0.05, -0.05],
        "mm": [47.0, 45.0, 43.0, 51.0, 52.0, 46.0, 36.0],   # set so the BUILT envelope (p95 distance from her skin on the repaired
                                                            # lattice, 11-11) hits the TARGET: top ~34, eye level ~45 each side   # top +36 mm above her vertex; ~45 mm beyond the skull each side
                                                            # at eye level (980 px wide at the pupils = 267 mm)
        "front_mm": 18.0,       # the front swoop off the hairline (projected 41 mm incl. the forehead's slope back)
        "clear_mm": 1.2,        # the innermost layer's clearance off the skin
    },
    "lift": {"deg_outer": 50.0, "deg_inner": 4.0, "jitter_deg": 6.0, "rise_mm": 16.0, "front_w": 1.0, "crown_w": 1.2,
             "small_w": 1.2, "g_jitter": 0.15, "crown_base_mm": 45.0, "small_base_mm": 8.0},   # the root's rise = atan(layer thickness / (g + rise)): the part's and the
                                                    # hairline's roots stand up (the swoop), downstream roots lie under
    "depth": {"z_lo": 0.15, "z_hi": 0.30, "pow": 0.85, "part_bonus": 0.15, "jitter": 0.10},
    "lock": {"spacing_mm": 9.5, "sub_spacing_mm": 3.6, "body_b": 0.62, "body_n": 0.35, "sub_tight": 0.42,
             "conv_mm": 28.0, "end_open": 0.75, "end_frac": 0.16, "meander_mm": 1.2, "meander_per_mm": [30.0, 90.0],
             "frizz_share": 0.04, "frizz_mm": 3.5,   # each hair its own slight wave and path: crossings of ~10-20 deg (real close-ups)
             "held_b": 1.08, "held_n": 0.80, "held_tight": 0.90, "end_wisp": 0.65},   # on the cranium the hair lies as a sheet (no gaps between
                                                                    # locks); the locks gather where it hangs free
    "root": {"emerge_deg": 30.0, "join_mm": 12.0,    # each hair leaves her skin low along its follicle, then joins its lock
             "edge_din_mm": 10.0, "edge_emerge_deg": 18.0, "edge_join_mm": 22.0},   # the hairline's hairs lie lower, longer
    "free": {"emerge_deg": [8.0, 28.0], "curl_per_m": [15.0, 70.0], "side_sd_deg": 22.0},  # baby hairs / wisps: own short path
    "flow": {"z_top_lo": 0.185, "z_top_hi": 0.275, "face_back": 0.30, "frame_fwd": 0.15, "frame_fwd_small": 0.12, "z_hold": 0.180,
             "fall_flare": 0.26, "front_bias": 0.12, "flare_y": 0.35,
             "front_split_y": 0.030, "turn_per_mm": 0.085, "turn_fall_per_mm": 0.05, "ds_mm": 2.0, "spring_mm": 7.0, "smooth_it": 60},
    "face": {"zc": 0.138, "ax": 0.064, "az": 0.100, "y_cut": -0.040, "clear_mm": 2.0,
             "lock_margin_mm": 7.0},   # whole locks keep this far outside the corridor, so their strands are not pressed onto
                                       # its edge one by one (g8: a thin veil of strands over her temple)
    # waves (comments stay ABOVE the dict: a trailing comment inside it has twice swallowed the keys after it):
    # - lam: the TARGET's dominant wavelength is ~90 mm (direction-field autocorrelation, measure_waves_ac.py, 11-11; ours at
    #   55-90 read 62-75 mm in the image);
    # - RIGIDITY: a hair of this type does not bend tighter than ~13 mm, so the amplitude follows the wavelength:
    #   A = lam^2 / (4 pi^2 R_bend), capped at amp_max (the TARGET swings 25-30 mm peak to peak);
    # - smooth from the roots, opening around cheek level; carried by sections (below), each lock a little off its section.
    "wave": {"lam_mm": [75.0, 115.0], "bend_r_mm": [13.0, 20.0], "amp_max_mm": 15.0, "plane_deg": 70.0,
             "start_mm": [90.0, 150.0], "ramp_mm": 80.0, "z_open": [0.20, 0.15], "tip_gain": 0.2, "chirp": 0.22,
             "n_share": 0.25, "head_share": 0.10, "sub_phase_sd": 0.30, "sub_amp_sd": 0.15, "lock_phase_sd": 0.45},
    "section": {"spacing_mm": 24.0, "gather": 0.40, "relief": 0.30, "crown_gather": 0.45},   # big locks: ~5 locks from a 24 mm root patch wave together and draw
                                                       # toward each other where the hair hangs (gaps between sections)
    "cut": {"z_front": 0.045, "z_back": -0.150, "pow": 1.3, "layer_mm": 360.0, "layer_d": 0.80, "texture": 0.09,
            "keep_short_mm": 140.0, "short_low_mm": 190.0, "short_low": 0.92,
            "max_mm": 470.0},   # no hair longer than hers has grown (the cut line alone let hairs down her back run to 615 mm)
    "cross": {"share": 0.09, "len_mm": 34.0},
    "fly": {"share_outer": 0.020, "wisp_share": 0.020, "halo_share": 0.10, "rate": [0.03, 0.12]},   # halo: the few short hairs that stand free on top
    "npt": 64,
}


def merged(over=None):
    sp = copy.deepcopy(DEFAULTS)
    for k, v in (over or {}).items():
        if k.startswith("_"):
            continue
        cur = sp; ks = k.split(".")
        for kk in ks[:-1]:
            cur = cur.setdefault(kk, {})
        if isinstance(v, dict) and isinstance(cur.get(ks[-1]), dict):
            cur[ks[-1]].update(v)
        else:
            cur[ks[-1]] = v
    return sp


def _unit(v, eps=1e-12):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class BodySDF:
    """her body's signed distance (m, negative inside) on a lattice (A[i, j, k] at lo + h * (i, j, k)), trilinear"""

    def __init__(self, A, lo, h):
        self.A = np.ascontiguousarray(A, np.float32); self.lo = np.asarray(lo, np.float64); self.h = float(h)
        self.sh = np.array(self.A.shape); self.flat = self.A.ravel()
        self.st = np.array([self.sh[1] * self.sh[2], self.sh[2], 1])

    def sd(self, P, chunk=2_000_000):
        P = np.asarray(P); shp = P.shape[:-1]; P = P.reshape(-1, 3); out = np.empty(len(P), np.float32)
        for a in range(0, len(P), chunk):
            Q = (P[a:a + chunk] - self.lo) / self.h
            i = np.clip(np.floor(Q).astype(np.int64), 0, self.sh - 2); f = np.clip(Q - i, 0.0, 1.0).astype(np.float32)
            b = i @ self.st; fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]; F = self.flat; s0, s1, s2 = self.st
            c00 = F[b] * (1 - fz) + F[b + s2] * fz; c01 = F[b + s1] * (1 - fz) + F[b + s1 + s2] * fz
            c10 = F[b + s0] * (1 - fz) + F[b + s0 + s2] * fz; c11 = F[b + s0 + s1] * (1 - fz) + F[b + s0 + s1 + s2] * fz
            out[a:a + chunk] = (c00 * (1 - fy) + c01 * fy) * (1 - fx) + (c10 * (1 - fy) + c11 * fy) * fx
        return out.reshape(shp)

    def grad(self, P):
        P = np.asarray(P, np.float64); e = self.h * 0.5; g = np.empty_like(P)
        for a in range(3):
            d = np.zeros(3); d[a] = e
            g[..., a] = self.sd(P + d) - self.sd(P - d)
        return _unit(g)

    def push_out(self, P, clear, it=3):
        """points closer than clear (m; scalar or per point) to her surface move out along the gradient (contact, not shape)"""
        P = np.asarray(P, np.float64).copy(); clear = np.broadcast_to(np.asarray(clear, np.float64), P.shape[:-1])
        for _ in range(it):
            s = self.sd(P); m = s < clear
            if not m.any():
                break
            P[m] += (clear[m] - s[m])[:, None] * self.grad(P[m])
        return P


def in_face(P, sp, margin=0.0):
    """points inside the face corridor (in front of her ears, inside the face ellipse, ABOVE her chin: below it hair hangs
    freely in front of her neck and shoulders, as in the TARGET; the ellipse's lower half used to gather that hair into a
    hard-edged curved wall beside her jaw, 11-11)"""
    f = sp["face"]; x, y, z = P[..., 0], P[..., 1], P[..., 2]
    u = (z - f["zc"]) / f["az"]
    xe = f["ax"] * np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0)) + f["clear_mm"] * 1e-3 + margin
    return (y < f["y_cut"]) & (np.abs(u) < 1.0) & (z > sp["head"]["z_menton"] - 0.004) & (np.abs(x) < xe)


def face_push(P, sp, margin=0.0):
    """the face corridor: points in front of her ears inside the face ellipse (x, z) move sideways to its edge (her face
    stays clear; framing locks fall at the cheek edge, as in the TARGET)"""
    f = sp["face"]; x, y, z = P[..., 0], P[..., 1], P[..., 2]
    u = (z - f["zc"]) / f["az"]; inside = (y < f["y_cut"]) & (np.abs(u) < 1.0) & (z > sp["head"]["z_menton"] - 0.004)
    xe = f["ax"] * np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0)) + f["clear_mm"] * 1e-3 + margin
    m = inside & (np.abs(x) < xe)
    if m.any():
        sgn = np.where(x >= 0, 1.0, -1.0)
        P = P.copy(); P[..., 0] = np.where(m, sgn * xe, x)
    return P, m


def thickness(P, sp):
    """the outer layer's height off the scalp at P (m): the measured envelope by height, lower over the front hairline"""
    t = sp["thick"]; z = P[..., 2]
    zz = np.asarray(t["z"])[::-1]; mm = np.asarray(t["mm"])[::-1]
    T = np.interp(z, zz, mm) * 1e-3
    front = smoothstep(-0.060, -0.090, P[..., 1]) * smoothstep(0.235, 0.265, z)
    return T * (1 - front) + t["front_mm"] * 1e-3 * front


# ---------------------------------------------------------------------------------------------------------------------
def partition(R, spacing, rng, cand=12000):
    """Poisson-disc centres over the roots (greedy over a random candidate set), every root to its nearest centre, two
    Lloyd steps snapped to roots -> (centre root index per group, group per root)"""
    idx = rng.permutation(len(R))[:min(cand, len(R))]; tree = None; keep = []
    C = R[idx]; taken = np.zeros(len(idx), bool); t = cKDTree(C)
    for j in range(len(idx)):
        if taken[j]:
            continue
        keep.append(idx[j]); taken[t.query_ball_point(C[j], spacing)] = True
    cen = np.array(keep)
    for _ in range(2):
        g = cKDTree(R[cen]).query(R)[1]
        cnt = np.bincount(g, minlength=len(cen)); M = np.zeros((len(cen), 3))
        for a in range(3):
            M[:, a] = np.bincount(g, R[:, a], minlength=len(cen)) / np.maximum(cnt, 1)
        cen = cKDTree(R).query(M)[1]
        tree = cKDTree(R[cen])
    g = tree.query(R)[1]
    return cen, g


def flow(P, L, sp):
    """the wanted direction for each lock's centreline at P (world, unit). On top: the sweep away from the part (or radial
    from the crown); lower: down, the temples a little back; blended by height."""
    fl = sp["flow"]; pt = sp["part"]
    sweep = np.zeros_like(P)
    fr = L["front"]
    sweep[fr, 0] = L["side"][fr]
    sweep[fr, 1] = np.where(L["side"][fr] > 0, pt["back_w"], pt["back_w_small"])
    rc = P[~fr] - L["whorl"]; rc[:, 2] = 0.0; rc[:, 1] = np.maximum(rc[:, 1], 0.0)   # the crown falls sideways or back, never forward
    sweep[~fr] = _unit(rc) + np.array([0.0, 0.0, -0.35])
    sweep = _unit(sweep + np.array([0.0, 0.0, -0.25]))
    down = np.tile(np.array([0.0, 0.0, -1.0]), (len(P), 1))
    # the front sections fall a little FORWARD along her face on both sides (the TARGET frames her face on her right too:
    # at eye level its hair shows ~48 mm beyond her far temple from the photo's angle, ours 24 with that side swept back,
    # 11-11); hair behind the front sections is swept back
    temple = (P[:, 1] < -0.02).astype(float) * np.where(fr, np.where(L["side"] > 0, -fl["frame_fwd"], -fl["frame_fwd_small"]), fl["face_back"])
    down[:, 1] += temple
    w = smoothstep(fl["z_top_lo"], fl["z_top_hi"], P[:, 2])[:, None]
    return _unit(w * sweep + (1 - w) * _unit(down))


def march(L, body, sp, Lmax, log=print):
    """build every lock's centreline from its root: lift, travel on its layer of the envelope, fall, flare; contact with
    her body and the face corridor kept by projection (never a shape-making force)"""
    fl = sp["flow"]; ds = fl["ds_mm"] * 1e-3; M = int(np.ceil(Lmax / ds)) + 1; nl = len(L["root"])
    clear = sp["thick"]["clear_mm"] * 1e-3
    C = np.zeros((nl, M, 3)); P = L["root"].copy(); C[:, 0] = P
    n0 = body.grad(P); f0 = flow(P, L, sp); ft = _unit(f0 - (f0 * n0).sum(1, keepdims=True) * n0)
    a = np.radians(L["lift_deg"])[:, None]; V = _unit(np.cos(a) * ft + np.sin(a) * n0)
    d = L["d"][:, None]
    radial_c = L["axis"]
    for k in range(1, M):
        phi = body.sd(P); n = body.grad(P); f = flow(P, L, sp)
        sk0 = k * ds
        lay = sk0 / (sk0 + L["g"] + L["g_eps"])          # the share of the stack below it: hair rooted downstream lies under
        tkz = 1 - (1 - L["tk"]) * smoothstep(0.22, 0.26, P[:, 2])     # her right section lies lower only on the TOP (g6: 21 mm at eye level vs 45)
        ht = clear + lay * thickness(P, sp) * tkz
        # held on the cranium by the set (above z_hold): the lock rides its layer of the envelope; below, it falls from the
        # widest point (gravity + the layers' flare), pushed off her only where it comes closer than its layer
        onhead = (n[:, 2] > -0.30) & (phi < ht + 0.018) & (P[:, 2] > fl["z_hold"])
        ftan = _unit(f - (f * n).sum(1, keepdims=True) * n)
        corr = np.clip((ht - phi) / (fl["spring_mm"] * 1e-3), -1.0, 1.0)[:, None]
        want_h = ftan                      # on the cranium the lock follows its layer; the height is set by position below
        out = P - radial_c; out[:, 2] = 0.0; out[:, 1] *= fl["flare_y"]; out = _unit(out)   # the layers spread sideways
        fb = np.where(L["root"][:, 1] < sp["flow"]["front_split_y"], -1.0, 1.0)[:, None] * np.array([0.0, 1.0, 0.0]) * fl["front_bias"]
        want_f = _unit(np.array([0.0, 0.0, -1.0]) + fl["fall_flare"] * (0.3 + d) * out + fb + 0.5 * np.clip(corr, 0, 1) * n)
        want = np.where(onhead[:, None], want_h, want_f)
        rate = np.where(onhead, fl["turn_per_mm"], fl["turn_fall_per_mm"]) * fl["ds_mm"]
        V = _unit(V + rate[:, None] * (want - V))
        Pn = P + V * ds
        # held locks reach their layer's height by position (critically damped: no ripple), once past the root's rise
        sk = k * ds; frac = 0.30 * smoothstep(0.0, 0.03, sk)
        lay1 = (sk0 + ds) / (sk0 + ds + L["g"] + L["g_eps"])
        phn = body.sd(Pn); nn = body.grad(Pn); hn = clear + lay1 * thickness(Pn, sp) * (1 - (1 - L["tk"]) * smoothstep(0.22, 0.26, Pn[:, 2]))
        rising = (phn < hn) & (sk < 0.012 + L["g_eps"])
        dz_ = np.where(onhead & ~rising, frac * (hn - phn), 0.0)
        # the correction acts square to the heading (g6: along her normal it pushed a rising lock backward on the slope
        # of her head, and the lock rolled into a crest)
        nq = nn - (nn * V).sum(1, keepdims=True) * V; nq = nq / np.maximum((nq * nn).sum(1, keepdims=True), 0.2)
        Pn = Pn + dz_[:, None] * nq
        Pc = body.push_out(Pn, clear * 0.5 + 0.4e-3 * d[:, 0], it=2)
        Pc, hit = face_push(Pc, sp, margin=sp["face"]["lock_margin_mm"] * 1e-3)
        # the heading comes from the flow steering only; a contact push removes just the part of it that went into her
        push = Pc - Pn; pl = np.linalg.norm(push, axis=1); pu = push / np.maximum(pl, 1e-12)[:, None]
        into = np.minimum((V * pu).sum(1), 0.0) * (pl > 1e-6)
        V = _unit(V - into[:, None] * pu)
        P = Pc; C[:, k] = P
    return C, ds


def _smooth_curves(C, it=4, w=0.35):
    C = C.copy()
    for _ in range(it):
        C[:, 1:-1] = (1 - w) * C[:, 1:-1] + 0.5 * w * (C[:, :-2] + C[:, 2:])
    return C


def frames(C, body):
    """a ribbon frame along each centreline: t along it; n starts as her scalp's normal at the root and is carried along the
    curve by a rotation-minimizing frame (double reflection, Wang et al. 2008), so a lock turns only where it bends; b = t x n.
    (11-10 night: n from her field's gradient at every point swung up to 50 deg per 20 mm (p90) where the hanging hair passes
    her neck and shoulder, swirling every strand about its lock centre: the fine regular crinkle seen in the g1 frames)"""
    T = _unit(np.gradient(C, axis=1))
    nl, M = C.shape[:2]
    N = np.empty_like(C); n0 = body.grad(C[:, 0]); n0 = _unit(n0 - (n0 * T[:, 0]).sum(-1, keepdims=True) * T[:, 0]); N[:, 0] = n0
    r = n0
    for i in range(M - 1):
        v1 = C[:, i + 1] - C[:, i]; c1 = np.maximum((v1 * v1).sum(-1, keepdims=True), 1e-18)
        rL = r - (2 / c1) * (v1 * r).sum(-1, keepdims=True) * v1
        tL = T[:, i] - (2 / c1) * (v1 * T[:, i]).sum(-1, keepdims=True) * v1
        v2 = T[:, i + 1] - tL; c2 = np.maximum((v2 * v2).sum(-1, keepdims=True), 1e-18)
        r = rL - (2 / c2) * (v2 * rL).sum(-1, keepdims=True) * v2
        r = _unit(r - (r * T[:, i + 1]).sum(-1, keepdims=True) * T[:, i + 1]); N[:, i + 1] = r
    B = _unit(np.cross(T, N))
    return T, N, B


def _interp_lock(A, k, u):
    """A (nl, M, ...) sampled at uniform steps; k (m,) lock index, u (m,) fractional sample index -> values"""
    M = A.shape[1]; u = np.clip(u, 0, M - 1.000001); i = np.floor(u).astype(np.int64); f = (u - i)
    f = f.reshape(f.shape + (1,) * (A.ndim - 2))
    return A[k, i] * (1 - f) + A[k, i + 1] * f


def groom(R, grow_len, sp, body, seed=1, log=print, din=None, free=None):
    """R (n, 3) roots on her scalp (m), grow_len (n,) each hair's own grown length (m; short = a new hair in its cycle);
    din (n,) mm inside the hairline (optional: the edge hairs lie lower); free (n,) bool: the hairline's baby hairs and wisps,
    too short and fine to join a lock: each its own short path along the flow, low, curling
    -> dict: P (n, npt, 3) float32, LEN (n,), lock, sub, d (per strand), report"""
    t0 = time.time(); rng = np.random.default_rng(seed); n = len(R); npt = sp["npt"]; rep = {}
    R = np.asarray(R, np.float64); Nr = body.grad(R)
    # ---- locks and sub-locks
    lc, lock = partition(R, sp["lock"]["spacing_mm"] * 1e-3, rng)
    sc, subg = partition(R, sp["lock"]["sub_spacing_mm"] * 1e-3, rng, cand=40000)
    # a sub-lock belongs to the lock of most of its roots: re-split by (sub, lock) pairs
    nl = len(lc)
    cnt = np.bincount(lock, minlength=nl); Rc = np.zeros((nl, 3))
    for a in range(3):
        Rc[:, a] = np.bincount(lock, R[:, a], minlength=nl) / np.maximum(cnt, 1)
    Rc = R[cKDTree(R).query(Rc)[1]]          # the lock's centre sits on her scalp
    # ---- sections: part side, crown, depth, lift
    pt = sp["part"]; xp = pt["x_mm"] * 1e-3; y_end = pt["y_front"] + pt["depth_mm"] * 1e-3
    whorl_y = y_end + 0.012
    front = Rc[:, 1] < whorl_y            # everything ahead of the whorl belongs to the part's sweep (no crown hair flows forward)
    side = np.where(Rc[:, 0] >= xp, 1.0, -1.0)
    near = front & (np.abs(Rc[:, 0] - xp) < pt["cross_mm"] * 1e-3)
    flip = near & (rng.random(nl) < pt["cross_share"]); side[flip] *= -1
    # the SOFT part, hair by hair: a hair rooted near the part falls to either side on its own, more often the nearer the
    # line (whole locks flipping alone left a zig-zag seam of alternating locks from above, sheet 10e)
    xs_ = np.abs(R[:, 0] - xp); cw_ = pt["cross_mm"] * 1e-3
    sw_i = np.nonzero((R[:, 1] < whorl_y) & (xs_ < cw_) & (rng.random(n) < pt["strand_cross"] * (1 - xs_ / cw_)))[0]
    for sgn in (1.0, -1.0):
        cand = np.nonzero(front & (side == -sgn))[0]; mine = sw_i[side[lock[sw_i]] == sgn]
        if len(cand) and len(mine):
            lock[mine] = cand[cKDTree(Rc[cand]).query(R[mine])[1]]
    rep["hairs_crossing_the_part"] = int(len(sw_i))
    key = subg.astype(np.int64) * (nl + 1) + lock; uk, sub = np.unique(key, return_inverse=True)
    ns = len(uk); rep.update(locks=nl, sublocks=ns)
    whorl = np.array([xp * 0.5, whorl_y, sp["head"]["z_top"]])
    dz = sp["depth"]
    d = np.clip((Rc[:, 2] - dz["z_lo"]) / (dz["z_hi"] - dz["z_lo"]), 0, 1) ** dz["pow"]
    d = np.clip(d + dz["part_bonus"] * front * np.exp(-np.abs(Rc[:, 0] - xp) / 0.03) + rng.normal(0, dz["jitter"], nl), 0.0, 1.0)
    li = sp["lift"]
    # upstream distance g: hair from the part line and the front hairline passes over hair rooted behind it; at the crown
    # the fall is radial from the whorl. The hairline's y at each x is read from the roots themselves.
    xb = np.linspace(R[:, 0].min(), R[:, 0].max(), 25); yb = np.full(len(xb) - 1, np.nan)
    for i_ in range(len(xb) - 1):
        mm_ = (R[:, 0] >= xb[i_]) & (R[:, 0] < xb[i_ + 1]) & (R[:, 2] > 0.20)
        if mm_.any():
            yb[i_] = np.percentile(R[mm_, 1], 1)
    xc_ = 0.5 * (xb[1:] + xb[:-1]); ok_ = np.isfinite(yb); yhl = np.interp(Rc[:, 0], xc_[ok_], yb[ok_])
    g_part = np.abs(Rc[:, 0] - xp); g_front = np.clip(Rc[:, 1] - yhl, 0, None) * li["front_w"]
    # the whorl's own hairs spiral out flat (crown_base); her right side of the part lies under the swept left (small_base)
    g = np.where(front, np.minimum(g_part, g_front), (np.linalg.norm((Rc - whorl)[:, :2], axis=1) + li["crown_base_mm"] * 1e-3) * li["crown_w"])
    g = np.where(front & (side < 0), np.minimum(g_part * li["small_w"] + li["small_base_mm"] * 1e-3, g_front), g)
    g = np.clip(g * (1 + rng.normal(0, li["g_jitter"], nl)), 0, None)
    g_eps = li["rise_mm"] * 1e-3
    lift = np.degrees(np.arctan(thickness(Rc, sp) / (g + g_eps))) + rng.normal(0, li["jitter_deg"], nl)
    lift = np.clip(lift, li["deg_inner"], li["deg_outer"])
    axis = np.array([0.0, float(np.median(R[:, 1])), 0.0])
    lift = np.where(front & (side < 0), lift * pt["small_lift"], lift)
    tk = np.where(front, np.where(side < 0, pt["small_thick"], 1.0), pt["crown_thick"])
    # SECTIONS (big locks, ~24 mm root patches) are set before the paths are built: each rides at its own height on the
    # cranium (relief: the TARGET's top is lifted sweeps with shadowed valleys between, not one smooth sheet)
    scn, sec = partition(Rc, sp["section"]["spacing_mm"] * 1e-3, rng, cand=nl)
    tk = tk * (1 + sp["section"]["relief"] * rng.uniform(-1.0, 1.0, len(scn)))[sec]
    L = dict(root=Rc, front=front, side=side, whorl=whorl, d=d, lift_deg=np.clip(lift, 0, 75), axis=axis, tk=tk, g=g, g_eps=g_eps)
    # the layer each lock ends in on the cranium also orders the falling layers (flare, colour depth)
    d = np.clip(0.15 / (0.15 + g + g_eps) + rng.normal(0, dz["jitter"] * 0.5, nl), 0, 1); L["d"] = d
    rep.update(front_locks=int(front.sum()), left_locks=int((front & (side > 0)).sum()), crossed=int(flip.sum()))
    # ---- centrelines
    Lmax = float(np.percentile(grow_len, 99.5)) * 1.12
    C, ds = march(L, body, sp, Lmax, log=log)
    # the march follows her lattice's gradient, which jumps at its 3 mm cells: the paths rippled (curvature sign changes every
    # 24-44 mm on the cranium, 3-6x the head's own curvature, 11-10 night). A lock is smooth over centimetres: diffuse each
    # centreline along itself (~9 mm), root fixed, then contact again
    C = _smooth_curves(C, it=sp["flow"]["smooth_it"])
    C = body.push_out(C.reshape(-1, 3), sp["thick"]["clear_mm"] * 0.5e-3).reshape(C.shape)
    M = C.shape[1]
    nsec_ = len(scn)
    cnt_s = np.bincount(sec, minlength=nsec_).astype(np.float64); Csec = np.zeros((nsec_, M, 3))
    np.add.at(Csec, sec, C); Csec /= np.maximum(cnt_s, 1)[:, None, None]
    # a lock is drawn toward its section only where it really runs with it: a section split by her shoulder or neck (one
    # part in front, one behind) has its mean INSIDE her (g13: 4,961 strands cut through her body); the pull fades with the
    # lock's distance from the mean (30 mm) and never moves a sample to where her field reads inside
    dsec = np.linalg.norm(Csec[sec] - C, axis=-1, keepdims=True)
    hang_g = smoothstep(0.235, 0.165, C[..., 2])[..., None]; cg_ = sp["section"]["crown_gather"]
    gth = sp["section"]["gather"] * (cg_ + (1 - cg_) * hang_g) * np.exp(-(dsec / 0.03) ** 2)
    Cg = C + gth * (Csec[sec] - C)
    okg = (body.sd(Cg.reshape(-1, 3)).reshape(nl, M) > sp["thick"]["clear_mm"] * 0.5e-3)[..., None]
    C = np.where(okg, Cg, C)
    C = body.push_out(C.reshape(-1, 3), sp["thick"]["clear_mm"] * 0.5e-3).reshape(C.shape)
    rep["sections"] = int(nsec_)
    log("groom: %d sections, %d locks, %d sub-locks, centrelines %d x %d (%.0f s)" % (nsec_, nl, ns, nl, M, time.time() - t0))
    # ---- the layer cut: a cut line in space (shortest face-framing at the front, longest down the back), and the outer
    #      (top) layers cut shorter (elevated layers, the TARGET's volume)
    cu = sp["cut"]; seg = np.linalg.norm(np.diff(C, axis=1), axis=2); arc = np.concatenate([np.zeros((nl, 1)), np.cumsum(seg, 1)], 1)
    th = np.arctan2(C[..., 0], -(C[..., 1] - axis[1]))
    zc = cu["z_back"] + (cu["z_front"] - cu["z_back"]) * ((1 + np.cos(th)) / 2) ** cu["pow"]
    below = (C[..., 2] < zc) & (arc > 0.06)
    first = np.where(below.any(1), below.argmax(1), M - 1)
    Lcut = arc[np.arange(nl), first]
    Lcut = np.minimum(Lcut, np.where(d > cu["layer_d"], cu["layer_mm"] * 1e-3 * (1 + 0.15 * (1 - d)), 9.0))
    Lcut = np.minimum(Lcut, cu["max_mm"] * 1e-3)
    rep["lock_len_mm_p10_50_90"] = np.percentile(Lcut * 1e3, [10, 50, 90]).round(0).tolist()
    # ---- frames and waves (per lock params; per sub-lock jitter)
    T, Nn, B = frames(C, body)
    wv = sp["wave"]
    # waves are carried by SECTIONS (big locks ~24 mm at the root, ~5 locks each): one wavelength, phase, plane and strength
    # per section, each lock a little off it; sections are independent of each other (the TARGET's big loose locks)
    nsec = int(sec.max()) + 1
    # hair3 (11-11): a whole section waving in one phase and plane drew ~5 locks side by side into one flat band that
    # bends in unison: the ribbon-like clumps seen on sheet 13. lock_lam_sd / lock_phase_sd / lock_plane_sd_deg let
    # each lock wave on its own (real wavy hair: neighbouring locks only loosely in step)
    lam = rng.uniform(*wv["lam_mm"], nsec)[sec] * (1 + rng.normal(0, wv.get("lock_lam_sd", 0.05), nl)) * 1e-3
    rb = rng.uniform(*wv["bend_r_mm"], nsec)[sec] * 1e-3
    amp = np.minimum(lam ** 2 / (4 * np.pi ** 2 * rb), wv["amp_max_mm"] * 1e-3) * np.clip(1 + rng.normal(0, 0.07, nl), 0.8, 1.15)
    s0 = rng.uniform(*wv["start_mm"], nsec)[sec] * 1e-3 + rng.normal(0, 0.008, nl); ph2 = rng.uniform(0, 2 * np.pi, nsec)[sec]
    ph0 = rng.uniform(0, 2 * np.pi, nsec)[sec] + rng.normal(0, wv["lock_phase_sd"], nl)
    sgrid = np.arange(M) * ds
    lam_s = lam[:, None] * (1 + wv["chirp"] * np.sin(2 * np.pi * sgrid[None] / (3.1 * lam[:, None]) + ph2[:, None]))
    PH = ph0[:, None] + np.cumsum(2 * np.pi * ds / lam_s, axis=1)
    hang = smoothstep(wv["z_open"][0], wv["z_open"][1], C[..., 2])   # waves open lower down, where the hair hangs free
    hangL = smoothstep(0.262, 0.222, C[..., 2])         # locks gather from the temples down; the cranium's top is a sheet
    hC = np.maximum(body.sd(C.reshape(-1, 3)).reshape(nl, M), 0.0); gC = body.grad(C.reshape(-1, 3)).reshape(nl, M, 3)
    # the amplitude from the LOCAL wavelength (one curl radius along the hair: where the wave shortens it also shallows)
    ampL = np.minimum(lam_s ** 2 / (4 * np.pi ** 2 * rb[:, None]), wv["amp_max_mm"] * 1e-3) * (amp / np.maximum(np.minimum(lam ** 2 / (4 * np.pi ** 2 * rb), wv["amp_max_mm"] * 1e-3), 1e-9))[:, None]
    AMP = ampL * smoothstep(s0[:, None], s0[:, None] + wv["ramp_mm"] * 1e-3, sgrid[None]) * (wv["head_share"] + (1 - wv["head_share"]) * hang)
    AMP = AMP * (1 + wv["tip_gain"] * np.clip(sgrid[None] / np.maximum(Lcut[:, None], 1e-3), 0, 1) ** 2)
    # hair3: an IRREGULAR wave: the bend's strength drifts along each lock (two slow beats per lock), so a lock runs nearly
    # straight for a stretch, then swings (the TARGET's loose waves), instead of one even ripple everywhere
    irr = wv.get("irreg", 0.0)
    if irr > 0:
        b1 = rng.uniform(150, 260, nl)[:, None] * 1e-3; b2 = rng.uniform(70, 130, nl)[:, None] * 1e-3
        e1 = rng.uniform(0, 2 * np.pi, nl)[:, None]; e2 = rng.uniform(0, 2 * np.pi, nl)[:, None]
        env = 1 + irr * (np.sin(2 * np.pi * sgrid[None] / b1 + e1) + 0.6 * np.sin(2 * np.pi * sgrid[None] / b2 + e2)) / 1.6
        AMP = AMP * np.clip(env, 0.15, 1.7)
    # ---- strands
    lk = sp["lock"]; sub_lock = np.zeros(ns, np.int64); sub_lock[sub] = lock
    cs = np.bincount(sub, minlength=ns); Rs = np.zeros((ns, 3))
    for a in range(3):
        Rs[:, a] = np.bincount(sub, R[:, a], minlength=ns) / np.maximum(cs, 1)
    sph = rng.normal(0, wv["sub_phase_sd"], ns); sam = np.clip(1 + rng.normal(0, wv["sub_amp_sd"], ns), 0.7, 1.25)
    # each lock waves in its own plane (within plane_deg of its layer): seen from any side some locks show their S
    om = np.radians(rng.uniform(-wv["plane_deg"], wv["plane_deg"], nsec)[sec] + rng.normal(0, wv.get("lock_plane_sd_deg", 12.0), nl)); cw, sw_ = np.cos(om)[lock], np.sin(om)[lock]
    # hair3: the wave's plane turns slowly along the lock (wavy hair is a flattened helix; a lock that keeps one plane reads
    # as a flat crimped ribbon from the side that shows its face)
    twr = np.zeros(nl)
    if wv.get("twist_deg_per_100mm", 0.0):      # (drawn only when used: the defaults keep the random stream, so g27 rebuilds as it was)
        twr = np.radians(wv["twist_deg_per_100mm"]) / 0.1 * rng.choice([-1.0, 1.0], nl) * rng.uniform(0.5, 1.5, nl)
    # strand lengths: the cut, textured ends; new short hairs keep their own length
    Ls = Lcut[lock] * (1 + rng.normal(0, cu["texture"], n).clip(-2.5, 2.5) * (1 - 0.5 * L["d"][lock]))
    short = grow_len < cu["keep_short_mm"] * 1e-3
    Ls = np.where(short, grow_len, Ls)
    Ls = np.clip(Ls, 0.004, None); rep["strand_len_mm_p10_50_90"] = np.percentile(Ls * 1e3, [10, 50, 90]).round(0).tolist()
    # root offsets in the lock's root frame: (b, n) of the strand and of its sub-lock
    F0 = np.stack([B[:, 0], Nn[:, 0], T[:, 0]], 1)            # (nl, 3 axes, 3)
    o_sub0 = np.einsum("kij,kj->ki", F0[sub_lock], Rs - C[sub_lock, 0])[:, :2]
    o_own0 = np.einsum("kij,kj->ki", F0[lock], R - C[lock, 0])[:, :2] - o_sub0[sub]
    P = np.empty((n, npt, 3), np.float32); u = np.linspace(0, 1, npt)
    rt = sp["root"]; edge = (np.asarray(din) < rt["edge_din_mm"]) if din is not None else np.zeros(n, bool)
    em_deg = np.where(edge, rt["edge_emerge_deg"], rt["emerge_deg"]); join_m = np.where(edge, rt["edge_join_mm"], rt["join_mm"]) * 1e-3
    free_ = np.zeros(n, bool) if free is None else np.asarray(free, bool); fr_ = sp["free"]
    fem = rng.uniform(*fr_["emerge_deg"], n); fcu = rng.uniform(*fr_["curl_per_m"], n); fsd = rng.normal(0, fr_["side_sd_deg"], n)
    rep["edge_strands"] = int(edge.sum()); rep["free_strands"] = int(free_.sum())
    fly_rate = np.zeros(n); fly_dir = np.zeros((n, 2))
    outer = L["d"][lock] > 0.6
    fly = (outer & (rng.random(n) < sp["fly"]["share_outer"])) | (short & (Rc[lock, 2] > 0.26) & (rng.random(n) < sp["fly"]["halo_share"]))
    fly_rate[fly] = rng.uniform(*sp["fly"]["rate"], fly.sum()); a = rng.uniform(-0.9, 0.9, fly.sum())
    fly_dir[fly] = np.stack([np.sin(a), np.abs(np.cos(a))], 1)
    # loose WISPS: a small share of whole sub-locks of the outer layers leave the mass together (the TARGET's silhouette is
    # full of them), each hair a little off its wisp's way
    wsub = (rng.random(ns) < sp["fly"]["wisp_share"]) & (L["d"][sub_lock] > 0.45)
    wm = wsub[sub] & ~short & ~free_ & ~fly
    wr_ = rng.uniform(*sp["fly"]["rate"], ns); wa_ = rng.uniform(-0.9, 0.9, ns)
    fly_rate[wm] = wr_[sub[wm]] * rng.uniform(0.8, 1.2, wm.sum()); aw = wa_[sub[wm]] + rng.normal(0, 0.15, wm.sum())
    fly_dir[wm] = np.stack([np.sin(aw), np.abs(np.cos(aw))], 1); fly = fly | wm
    fly_s0 = np.where(wm, rng.uniform(0.2, 0.6, ns)[sub], rng.uniform(0.15, 0.5, n))
    rep["flyaway_hairs"] = int(fly.sum()); rep["wisps"] = int(wsub.sum())
    cross = rng.random(n) < sp["cross"]["share"]
    # a neighbouring lock for the crossing strands
    # crossing only to a neighbour lock that travels the same way (directions over the first 60 mm agree): a strand drifting
    # to a lock leaving the other way (across the part or the whorl) would cut a shortcut arc
    nb = cKDTree(Rc).query(Rc, k=6)[1][:, 1:]
    j60 = min(M - 1, int(0.06 / ds)); dirc = _unit(C[:, j60] - C[:, 0])
    agree = (dirc[nb] * dirc[:, None]).sum(-1) > 0.85
    # ... and stay together all the way down (neighbours at the root can part at her shoulder: a strand drifting between
    # them would pass through her): the two centrelines never more than 25 mm apart
    far = np.linalg.norm(C[nb] - C[:, None], axis=-1).max(-1)        # (nl, 5)
    agree &= far < 0.025
    pick = np.where(agree, rng.random(nb.shape), -1.0).argmax(1); has = agree.any(1)
    k2l = np.where(has, nb[np.arange(nl), pick], np.arange(nl))
    k2 = k2l[lock]; cross = cross & has[lock]
    sx = rng.uniform(0.25, 0.85, n); mph = rng.uniform(0, 2 * np.pi, (n, 2))
    mfr = 0.1 / (rng.uniform(*lk["meander_per_mm"], (n, 2)) * 1e-3)        # cycles per 0.1 m
    mm = lk["meander_mm"] * 1e-3 * np.where(rng.random(n) < lk["frizz_share"], lk["frizz_mm"] / lk["meander_mm"], 1.0) * rng.uniform(0.4, 1.3, n)
    chunk = 20000; n_stop = 0
    for a0 in range(0, n, chunk):
        sl = slice(a0, min(n, a0 + chunk)); m = sl.stop - sl.start
        s = Ls[sl, None] * u[None]                       # (m, npt) arc along the strand
        q = s / ds                                       # fractional index on the lock samples
        kk = np.repeat(lock[sl], npt); qq = q.ravel()

        def at(k_, A_):
            return _interp_lock(A_, k_, qq).reshape(m, npt, *A_.shape[2:])
        Cs, Ts, Ns, Bs = at(kk, C), at(kk, T), at(kk, Nn), at(kk, B)
        hg = at(kk, hangL[..., None])[..., 0]
        ph = at(kk, PH[..., None])[..., 0]; am = at(kk, AMP[..., None])[..., 0]
        sj = sub[sl]
        frac = s / np.maximum(Ls[sl, None], 1e-6)
        conv = smoothstep(0.0, lk["conv_mm"] * 1e-3, s)
        endo = smoothstep(1 - lk["end_frac"], 1.0, frac)
        bb = lk["held_b"] + (lk["body_b"] - lk["held_b"]) * hg; bn = lk["held_n"] + (lk["body_n"] - lk["held_n"]) * hg
        bt = lk["held_tight"] + (lk["sub_tight"] - lk["held_tight"]) * hg
        spread_b = 1 + (bb - 1) * conv + lk["end_open"] * endo
        spread_n = 1 + (bn - 1) * conv + lk["end_open"] * 0.6 * endo
        tight = (1 + (bt - 1) * conv) * (1 - lk["end_wisp"] * endo)      # toward the tips each sub-lock gathers into a tapered wisp
                                                                         # while the lock opens (g8 lock end: a fuzz cloud)
        ob = o_sub0[sj, 0, None] * spread_b + o_own0[sl, 0, None] * tight * spread_b
        on = o_sub0[sj, 1, None] * spread_n + o_own0[sl, 1, None] * tight * spread_n
        # the strand's own slow meander inside its sub-lock (lateral mostly)
        mw = conv * (1 - 0.6 * endo)
        # rigidity: no hair's own wander bends tighter than the wave's limit: amplitude <= period^2 / (4 pi^2 R_min)
        capm = (0.1 / mfr[sl]) ** 2 / (4 * np.pi ** 2 * wv["bend_r_mm"][0] * 1e-3)
        ob = ob + np.minimum(mm[sl], capm[:, 0])[:, None] * mw * (np.sin(mfr[sl, 0, None] * 2 * np.pi * s / 0.1 + mph[sl, 0, None]))
        on = on + 0.5 * np.minimum(mm[sl], capm[:, 1])[:, None] * mw * (np.sin(mfr[sl, 1, None] * 2 * np.pi * s / 0.1 + mph[sl, 1, None]))
        # the wave, per sub-lock phase and amplitude
        wph = ph + sph[sj, None]; wa = am * sam[sj, None]
        w1 = wa * np.sin(wph); w2 = wv["n_share"] * wa * np.cos(wph)
        if np.any(twr):
            oms = om[lock[sl], None] + twr[lock[sl], None] * s; cs_, sn_ = np.cos(oms), np.sin(oms)
        else:
            cs_, sn_ = cw[sl, None], sw_[sl, None]
        ob = ob + cs_ * w1 - sn_ * w2; on = on + sn_ * w1 + cs_ * w2
        # fly-aways leave the mass (outward, a little sideways) from where they start
        fr_ = fly_rate[sl, None] * np.clip(s - fly_s0[sl, None] * Ls[sl, None], 0, None) * (1 + 0.3 * np.sin(2 * np.pi * s / 0.11 + mph[sl, 0, None]))
        ob = ob + fr_ * fly_dir[sl, 0, None]; on = on + fr_ * fly_dir[sl, 1, None] * 1.6
        Q = Cs + Bs * ob[..., None] + Ns * on[..., None]
        # short (new-growth) hairs lie low under the long hair along their lock's way: they cannot ride its rise and stop
        # in mid-air (g8: a crest of hair ends at her right top). The shorter the hair, the nearer her scalp
        wsh = np.clip(1.0 - Ls[sl] / (sp["cut"]["short_low_mm"] * 1e-3), 0.0, 1.0) * sp["cut"]["short_low"] * (~fly[sl])
        if wsh.any():
            hs_ = at(kk, hC[..., None])[..., 0]; gs_ = at(kk, gC)
            Q = Q - (wsh[:, None] * hs_)[..., None] * gs_
        # crossing strands drift to a neighbouring lock over len_mm from s = sx * L
        cm = cross[sl]
        if cm.any():
            idx = np.nonzero(cm)[0]; k2r = np.repeat(k2[sl][idx], npt); q2 = q[idx].ravel()
            C2 = _interp_lock(C, k2r, q2).reshape(len(idx), npt, 3); B2 = _interp_lock(B, k2r, q2).reshape(len(idx), npt, 3)
            N2 = _interp_lock(Nn, k2r, q2).reshape(len(idx), npt, 3)
            Q2 = C2 + B2 * ob[idx, :, None] * 0.6 + N2 * on[idx, :, None] * 0.6
            w = smoothstep(sx[sl][idx, None] * Ls[sl][idx, None], sx[sl][idx, None] * Ls[sl][idx, None] + sp["cross"]["len_mm"] * 1e-3, s[idx])
            Q[idx] = Q[idx] * (1 - w[..., None]) + Q2 * w[..., None]
        # the strand's own root: the along-lock residual fades out over the first ~10 mm
        res = R[sl] - Q[:, 0]
        Q = Q + res[:, None, :] * np.exp(-s / 0.008)[..., None]
        # the root: the hair leaves her skin along its follicle (emerge_deg off the skin, toward its lock's way), then joins
        nr = Nr[sl]; tl = Ts[:, 0] - (Ts[:, 0] * nr).sum(-1, keepdims=True) * nr; tl = _unit(tl)
        e_ = np.radians(em_deg[sl])[:, None]; de = np.cos(e_) * tl + np.sin(e_) * nr
        Qr = R[sl][:, None, :] + s[..., None] * de[:, None, :]
        wj = smoothstep(0.0, join_m[sl][:, None], s)[..., None]
        Q = (1 - wj) * Qr + wj * Q
        # free hairs (baby hairs, wisps): their own short path from the root along the flow, low, bending in an arc
        fm = free_[sl]
        if fm.any():
            fi = np.nonzero(fm)[0]; t_ = tl[fi]; n_ = nr[fi]; b_ = _unit(np.cross(t_, n_))
            a_ = np.radians(fsd[sl][fi])[:, None]; t2 = _unit(np.cos(a_) * t_ + np.sin(a_) * b_)
            ef = np.radians(fem[sl][fi])[:, None]; kc = fcu[sl][fi][:, None]; ss = s[fi]
            # an arc of curvature kc in the plane of (t2, n): rises at ef, bends back toward the skin (or away: sign)
            # (an arc while it bends, then it lies along the skin at th_min: it never turns back into her)
            th_min = np.radians(-2.0); s_star = (ef - th_min) / np.maximum(kc, 1e-6); sa = np.minimum(ss, s_star)
            th = ef - kc * sa
            Qf = R[sl][fi][:, None, :] + ((np.sin(ef) - np.sin(th)) / kc)[..., None] * t2[:, None, :]                  + ((np.cos(th) - np.cos(ef)) / kc)[..., None] * n_[:, None, :]
            rest = np.clip(ss - s_star, 0, None)[..., None]
            Qf = Qf + rest * (np.cos(th_min) * t2[:, None, :] + np.sin(th_min) * n_[:, None, :])
            Q[fi] = Qf
        # contact with her body, strand by strand: a point the wave or an offset carried into her stops on the way from its
        # lock's centreline (outside her by construction) to where it wanted to be: the wave flattens against her, as hair
        # resting on a shoulder or back does (her field has no gradient deep inside, so pushing out cannot do this)
        # (her face corridor is part of the same contact: snapping each point to the corridor's edge on its own rippled the
        #  framing lengths point to point, as the body stop did)
        sdq = body.sd(Q.reshape(-1, 3)).reshape(m, npt); badq = ((sdq < 0.35e-3) | in_face(Q, sp)) & ~free_[sl][:, None]; badq[:, 0] = False
        if badq.any():
            bi = np.nonzero(badq); A_ = Cs[bi]; B_ = Q[bi]; lo_ = np.zeros(len(A_)); hi_ = np.ones(len(A_))
            for _ in range(7):
                md = 0.5 * (lo_ + hi_); X_ = A_ + md[:, None] * (B_ - A_); ok_ = (body.sd(X_) >= 0.35e-3) & ~in_face(X_, sp)
                lo_ = np.where(ok_, md, lo_); hi_ = np.where(ok_, hi_, md)
            # the stop is SMOOTH along the hair: stopping each point on its own left a point-to-point sawtooth where hair
            # lies on her (a third of the points within 1 mm of her skin zig-zagged with a 12.7 mm period, 11-11). Each
            # point takes the smallest fraction among its neighbours (+-2), smoothed, never above its own
            fr_c = np.ones((m, npt)); fr_c[bi] = lo_
            fr_s = uniform_filter1d(minimum_filter1d(fr_c, 5, axis=1, mode="nearest"), 5, axis=1, mode="nearest")
            fr_s = np.minimum(fr_s, fr_c); fr_s[:, 0] = 1.0
            Q = Cs + fr_s[..., None] * (Q - Cs); n_stop += int(len(A_))
        P[sl] = Q.astype(np.float32)
    rep["points_stopped_at_her_surface"] = n_stop
    log("groom: strands built (%.0f s)" % (time.time() - t0))
    # ---- contact: her body and her face corridor (projection; the shape is the outcome above)
    rad_clear = 0.35e-3
    # hair resting on her lies smooth along itself: where a hair is within 3 mm of her skin its points relax toward their
    # neighbours' midpoint (three passes), then the push-out below restores the clearance
    P64 = P.astype(np.float64)
    for _ in range(3):
        near = body.sd(P64.reshape(-1, 3)).reshape(n, npt) < 3.0e-3; near[:, :2] = False; near[:, -1] = False
        near &= ~free_[:, None]
        mid_ = 0.5 * (P64[:, :-2] + P64[:, 2:]); w_ = 0.6 * near[:, 1:-1, None]
        P64[:, 1:-1] = (1 - w_) * P64[:, 1:-1] + w_ * mid_
    flat = P64.reshape(-1, 3)
    sdv = body.sd(flat); bad = sdv < rad_clear
    rep["points_inside_before"] = int((sdv < 0).sum())
    if bad.any():
        flat[bad] = body.push_out(flat[bad], rad_clear, it=3)
    flat, hit = face_push(flat, sp); rep["face_pushed"] = int(hit.sum())
    P = flat.reshape(n, npt, 3).astype(np.float32)
    P[:, 0] = R.astype(np.float32)
    # nothing passes through her: deep inside her band store the field is clamped and has no gradient, so a point there
    # cannot be pushed out. Such a hair is ended at its last free point (counted: it should be rare; the causes above are
    # what keep hair out of her)
    sdl = body.sd(P.reshape(-1, 3)).reshape(n, npt); inside = sdl[:, 1:] < -0.5e-3
    bad_s = np.nonzero(inside.any(1))[0]; rep["strands_ended_at_her_surface"] = int(len(bad_s))
    for i_ in bad_s:
        j0 = int(inside[i_].argmax()) + 1                       # first inside point
        keep_ = P[i_, :max(j0 - 1, 2)].astype(np.float64)
        seg_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(keep_, axis=0), axis=1))]
        tt_ = np.linspace(0.0, seg_[-1], npt); P[i_] = np.stack([np.interp(tt_, seg_, keep_[:, a_]) for a_ in range(3)], 1)
    rep["points_inside_after"] = int((body.sd(P.reshape(-1, 3)) < 0).sum())
    fz = (P[..., 1] < -0.07) & (np.abs(P[..., 0]) < 0.05) & (P[..., 2] > 0.12) & (P[..., 2] < 0.215)
    rep["strands_over_face"] = int(fz.any(1).sum())      # hair in front of her eyes, nose, mouth (lift.py's box): should be ~0
    LEN = np.linalg.norm(np.diff(P.astype(np.float64), axis=1), axis=2).sum(1)
    rep["seconds"] = round(time.time() - t0, 1)
    return dict(P=P, LEN=LEN.astype(np.float32), lock=lock.astype(np.int32), sub=sub.astype(np.int32), d=L["d"][lock].astype(np.float32),
                sec=sec[lock].astype(np.int32),
                Nr=Nr.astype(np.float32), C=C.astype(np.float32), Lcut=Lcut, report=rep, locks=L)
