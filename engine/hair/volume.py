"""Hair Studio volume (numpy + scipy, Apache-2.0): the hair's own volume - strands that cannot pass through one another.

    relax(P, r_root, r_tip, surf, cfg, log=print) -> P (n, K, 3) float32, stats

Why (10-29 night): the walk (flow.py) moves every strand on its own, so nothing stopped 145k strands lying in one sheet a
few mm thick - a helmet on top and a thin curtain down the back, however the lifts were tuned (g1-g15). Real hair takes
room: the fibre is a few per cent of the space a loose head of hair fills (the rest is air between fibres that cross,
wave and never pack). This relaxes the strands as a continuum, the volumetric method of film hair (Petrovic, Henne &
Anderson 2005, "Volumetric methods for simulation and rendering of hair", Pixar; McAdams et al. 2009). Every pass:
 1. each segment's fibre volume (length x pi r^2) goes on a voxel grid (smoothed one voxel) -> the fibre fraction rho;
 2. where rho exceeds the hair's packing (the fraction this hair holds at rest), the excess C = rho / packing - 1;
 3. every point moves down the excess's gradient by the step that would clear it (C / |grad C|, the position-based
    fluids correction), capped per pass, smoothed along the strand (it bends, it does not kink); near the skin only
    the outward part of the move is kept (combed hair is held along the scalp by its tension: it stacks, it does not
    slide over the forehead or across the part);
 4. every segment keeps its length (follow-the-leader from the fixed root);
 5. the strands stay out of the body (nearest-vertex push along the normal; the exact push comes after the form).
The result is the hair's own: a thicker layer where more hair passes (the crown, behind the head), hanging lengths
that spread to hold their amount, nothing shaped by hand. More strands or thicker strands -> more volume; looser hair
(a lower packing) -> more volume.

Packing, worked on a head (cfg note): over the crown the hair rooted upstream passes, ~2.35 roots / mm2 x ~100 mm =
235 strands per mm of width x pi (0.0275 mm)^2 = 0.56 mm2 of fibre per mm -> a layer 0.56 mm / packing thick:
14 mm at 0.04, 28 mm at 0.02.
"""
import time

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

from .scalp import smoothstep


def _ftl(P, L0):
    """follow-the-leader: each segment back to its rest length, from the root out (P float32 (n, K, 3), in place)"""
    for k in range(1, P.shape[1]):
        v = P[:, k] - P[:, k - 1]
        P[:, k] = P[:, k - 1] + v * (L0[:, k - 1] / np.maximum(np.linalg.norm(v, axis=1), 1e-9))[:, None]
    return P


def density(P, rad, vox, lo, shape, sigma_vox):
    """the fibre fraction on the grid: segment midpoints, volume len x pi r^2, nearest voxel, smoothed"""
    M = 0.5 * (P[:, 1:] + P[:, :-1])
    seg = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=2)
    v = (seg * np.pi * rad ** 2).ravel()
    q = np.floor((M.reshape(-1, 3) - lo) / vox).astype(np.int64)
    ok = np.all((q >= 0) & (q < np.array(shape)), 1)
    lin = np.ravel_multi_index(q[ok].T, shape)
    G = np.bincount(lin, weights=v[ok], minlength=int(np.prod(shape))).reshape(shape).astype(np.float32)
    return gaussian_filter(G, sigma_vox) / vox ** 3


def relax(P, r_root, r_tip, surf, cfg, log=print, key=None):
    """key (n, d): where each strand grows, for carrying the guides' moves (roots in m, plus a part-side term so the
    two sides of a part never borrow each other's move). guide_every g > 1: only every g-th strand is relaxed (radius
    x sqrt(g), so the guides carry the hair's whole volume) and the rest take the moves of their 3 nearest guides."""
    g_ = int(cfg.get("guide_every", 1))
    if g_ > 1 and key is not None:
        from scipy.spatial import cKDTree
        n = len(P)
        gi = np.arange(0, n, g_)
        Pg0 = np.asarray(P[gi], np.float32)
        Pg, stats = relax(Pg0, r_root[gi] * np.sqrt(g_), r_tip[gi] * np.sqrt(g_), surf, dict(cfg, guide_every=1), log)
        D = Pg - Pg0
        dist, nn = cKDTree(key[gi]).query(key, k=3, workers=-1)
        w = 1.0 / np.maximum(dist, 1e-4)
        w /= w.sum(1, keepdims=True)
        out = np.asarray(P, np.float32).copy()
        L0 = np.linalg.norm(out[:, 1:] - out[:, :-1], axis=2)
        for b0 in range(0, n, 20000):
            sl = slice(b0, min(n, b0 + 20000))
            out[sl] += (D[nn[sl]] * w[sl][:, :, None, None]).sum(1)
        out[gi] = Pg
        _ftl(out, L0)
        stats["guides"] = int(len(gi))
        return out, stats
    t0 = time.time()
    P = np.ascontiguousarray(P, np.float32)
    n, K = P.shape[:2]
    pk = float(cfg["packing"])
    vox = float(cfg["voxel_mm"]) / 1000
    step_max = float(cfg["step_mm"]) / 1000
    hold = float(cfg["hold_mm"]) / 1000
    clear = float(cfg["clear_mm"]) / 1000
    u = np.linspace(0, 1, K - 1, dtype=np.float32)
    rad = (r_root[:, None] * (1 - u) + r_tip[:, None] * u).astype(np.float32)          # per segment
    L0 = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=2)
    lo = P.reshape(-1, 3).min(0) - 0.06
    hi = P.reshape(-1, 3).max(0) + 0.06
    shape = tuple((np.ceil((hi - lo) / vox)).astype(int) + 1)
    sd = np.full((n, K), -1.0, np.float32)           # last known skin distance (re-checked when it may be close)
    nrm = np.zeros((n, K, 3), np.float32)
    acc = np.zeros((n, K), np.float32)               # movement since the last check
    hist = []
    for it in range(int(cfg["iters"])):
        # the skin: distance and normal for the points that may be near it
        chk = (sd - acc) < (hold * 2 + 0.004)
        chk[:, 0] = False
        if chk.any():
            Q = P[chk].astype(np.float64)
            _, j = surf.tree.query(Q, workers=-1)
            sd[chk] = np.einsum("ij,ij->i", Q - surf.V[j], surf.VN[j])
            nrm[chk] = surf.VN[j]
            acc[chk] = 0
        rho = density(P, rad, vox, lo, shape, float(cfg["smooth_vox"]))
        C = np.maximum(rho / pk - 1.0, 0.0)
        g = np.stack(np.gradient(C, vox), 0)
        f = ((P.reshape(-1, 3) - lo) / vox).T
        Cp = map_coordinates(C, f, order=1, mode="nearest").reshape(n, K)
        gp = np.stack([map_coordinates(g[a], f, order=1, mode="nearest") for a in range(3)], -1).reshape(n, K, 3)
        del g
        # the step that clears the excess, C grad C / (|grad C|^2 + (C / reach)^2): Newton's at an edge of the hair, fading
        # to nothing inside a uniform mass (where the gradient is noise), never more than reach / 2
        gnsq = (gp ** 2).sum(-1)
        reach = float(cfg["reach_vox"]) * vox
        d = -gp * (float(cfg["relax"]) * Cp / np.maximum(gnsq + (Cp / reach) ** 2, 1e-6 / vox ** 2))[..., None]
        dl = np.linalg.norm(d, axis=2, keepdims=True)
        d = d * np.minimum(1.0, step_max / np.maximum(dl, 1e-12))
        # near the skin: only the outward part (the comb's tension holds the hair along the scalp)
        near = 1.0 - smoothstep(hold, 2 * hold, sd)
        dn = (d * nrm).sum(-1, keepdims=True)
        d = d - (near[..., None] * (d - np.maximum(dn, 0) * nrm))
        d = gaussian_filter1d(d, float(cfg["along_smooth"]), axis=1)
        d[:, 0] = 0
        P += d.astype(np.float32)
        _ftl(P, L0)
        # out of the body
        nb = (sd < clear + 0.002)
        nb[:, 0] = False
        if nb.any():
            Q = P[nb].astype(np.float64)
            _, j = surf.tree.query(Q, workers=-1)
            s_ = np.einsum("ij,ij->i", Q - surf.V[j], surf.VN[j])
            push = np.maximum(clear - s_, 0)
            P[nb] = (Q + push[:, None] * surf.VN[j]).astype(np.float32)
            sd[nb] = np.maximum(s_, clear)
            nrm[nb] = surf.VN[j]
        acc += np.linalg.norm(d, axis=2)
        over = rho > pk
        ex = float(C[over].mean()) if over.any() else 0.0
        hist.append([it, round(float(over.sum() * vox ** 3 * 1e6), 1), round(ex, 3), round(float(rho.max() / pk), 2),
                     round(float(np.percentile(np.linalg.norm(d, axis=2), 99) * 1000), 3)])
        if it % 5 == 0 or it == int(cfg["iters"]) - 1:
            log("  volume pass %d: over-packed %.0f cm3, mean excess %.2f, peak rho / packing %.1f, move p99 %.2f mm  "
                "(%.0f s)" % (it, hist[-1][1], ex, hist[-1][3], hist[-1][4], time.time() - t0))
    _ftl(P, L0)
    rho = density(P, rad, vox, lo, shape, float(cfg["smooth_vox"]))
    filled = rho > 0.1 * pk
    stats = {"packing": pk, "voxel_mm": cfg["voxel_mm"], "iters": int(cfg["iters"]),
             "fibre_cm3": round(float((L0 * np.pi * rad ** 2).sum() * 1e6), 1),
             "hair_volume_cm3": round(float(filled.sum() * vox ** 3 * 1e6), 1),
             "over_packed_cm3_first_last": [hist[0][1], hist[-1][1]],
             "peak_rho_over_packing_first_last": [hist[0][3], hist[-1][3]],
             "seconds": round(time.time() - t0, 1)}
    return P, stats
