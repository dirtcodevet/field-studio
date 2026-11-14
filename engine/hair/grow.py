"""Growing short hair from follicles (hair lane, 11-10). Any kind whose hairs are shorter than their gravity length
(kinds.section(name)['gravity_shapes_it'] False: brows, lashes, vellus, body hair, stubble) holds the shape its follicle and
its own curl give it, so it is grown, not simulated: from the follicle along its emergence direction, bending by the
kind's curvature toward a bend direction, sliding along whatever surface it meets (her skin, the eye's globe). Long kinds
(scalp) go through the rod (rod.py) instead. P14: strands are centrelines; colliders are signed distance fields.

    out = grow_short(sdf, P0, D0, B0, L, r_of_s, kappa0, kappa1, seat=..., gap=...)
    P = bind(P, groups, share)            # binding: neighbours' distal parts drawn together (tear film, sebum, water)

The callers (a region's follicle map) choose P0 (roots on the skin), D0 (the emergence direction), B0 (the bend
direction), lengths from the kind's growth cycle (kinds.cycle_lengths) and the radius profile from the kind (radius()).
"""
from __future__ import annotations

import numpy as np


def grad(sdf, P, eps=2e-5):
    G = np.stack([(sdf(P + eps * e) - sdf(P - eps * e)) / (2 * eps) for e in np.eye(3)], 1)
    n = np.linalg.norm(G, axis=1, keepdims=True)
    return G / np.maximum(n, 1e-12), n[:, 0]


def radius(s, L, r0, mid_ratio=1.0, taper=3e-3, tip_ratio=0.2, spindle_until=0.5):
    """a hair's radius along its length (m): the root radius rising (spindle, mid_ratio) to its widest by spindle_until x L
    (or the taper's start, if sooner), held, then over the last `taper` (at most 65 % of a short young hair) down to
    tip_ratio x the root, (1 - x)^0.9: the natural, never-cut tip. Arrays broadcast (s (n, m), the rest (n, 1) or scalars)."""
    s = np.asarray(s, float); L = np.asarray(L, float); r0 = np.asarray(r0, float)
    tz = np.minimum(taper, 0.65 * L); st = np.maximum(L - tz, 0.35 * L); sp = np.minimum(st, spindle_until * L)
    rm = r0 * mid_ratio; rt = r0 * tip_ratio
    x1 = np.clip(s / np.maximum(sp, 1e-9), 0, 1); x2 = np.clip((s - st) / np.maximum(L - st, 1e-9), 0, 1)
    return np.where(s < st, r0 + (rm - r0) * np.sin(0.5 * np.pi * x1), rt + (rm - rt) * (1 - x2) ** 0.9)


def grow_short(sdf, P0, D0, B0, L, rfun, kappa0, kappa1, seat=0.05e-3, gap=0.02e-3, ds=0.05e-3, npt=24,
               bend_mode="fixed", sweep_axis=None, sweep_rate=None):
    """grow n hairs. sdf(P) -> signed distance (m) of everything the hair must not enter (min of her skin and any
    colliders). P0 (n, 3) follicle exits on the skin; D0 (n, 3) unit emergence directions; B0 (n, 3) bend directions
    (bend_mode 'fixed': the hair turns toward B0 in the plane of its direction and B0, e.g. a lash curling up; 'surface':
    toward -grad sdf, i.e. toward the skin it lies on, e.g. a brow). L (n,) lengths (m); rfun(s (n, m)) -> radii (n, m);
    kappa0 / kappa1 (n,) curvature (1/m) at the root / tip (linear in s / L: a lash's bend grows toward the tip).
    sweep_axis / sweep_rate (n, 3) / (n,): an in-plane turn (rad per m) about an axis (a brow's hairs turning toward the
    temple). Contact: a step that would bring the fibre's surface within `gap` of the sdf slides along it (pushed out to
    the gap, its velocity into the surface removed). The root is seated `seat` under the skin.
    -> dict(pos (n, npt, 3), rad (n, npt), len (n,), contact (n,) steps in contact, min_clear (n,) m)"""
    P0 = np.asarray(P0, float); D = np.asarray(D0, float).copy(); B0 = np.asarray(B0, float); L = np.asarray(L, float)
    n = len(P0)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    Nr, _ = grad(sdf, P0)
    R = P0 - Nr * seat
    nst = int(np.ceil(L.max() / ds)) + 2
    path = np.zeros((nst, n, 3)); path[0] = R
    s = np.zeros(n); contact = np.zeros(n, int); minc = np.full(n, 1e9)
    k0 = np.asarray(kappa0, float) * np.ones(n); k1 = np.asarray(kappa1, float) * np.ones(n)
    for k in range(1, nst):
        h = np.minimum(ds, L - s); alive = h > 1e-9
        if not alive.any():
            path[k:] = path[k - 1]; break
        idx = np.where(alive)[0]; Pk = path[k - 1, idx]; dv = D[idx]; hk = h[idx]; q = s[idx] / L[idx]
        out_ = s[idx] > seat * 2                                          # bending starts once the hair is out of the skin
        kap = k0[idx] + (k1[idx] - k0[idx]) * q
        if bend_mode == "surface":
            Nn, _ = grad(sdf, Pk); bd = -Nn
        else:
            bd = B0[idx]
        perp = bd - dv * (dv * bd).sum(1, keepdims=True); pn = np.linalg.norm(perp, axis=1, keepdims=True)
        dv2 = dv + (kap * hk * out_)[:, None] * perp / np.maximum(pn, 1e-9)
        if sweep_axis is not None:
            ax = np.asarray(sweep_axis, float)[idx]; w = np.asarray(sweep_rate, float)[idx] * hk * out_
            dv2 = dv2 + w[:, None] * np.cross(ax, dv2)
        dv2 /= np.linalg.norm(dv2, axis=1, keepdims=True)
        Pn = Pk + dv2 * hk[:, None]; s2 = s[idx] + hk
        rr = rfun_at(rfun, idx, s2)
        dist = sdf(Pn); need = rr + gap * np.clip((s2 - 2 * seat) / 0.3e-3, 0, 1)
        past = s2 > 2 * seat + 1e-6
        hit = past & (dist < need)
        if hit.any():
            j = np.where(hit)[0]; Nn, _ = grad(sdf, Pn[j])
            Pn[j] = Pn[j] + Nn * (need[j] - dist[j])[:, None]
            dn = (dv2[j] * Nn).sum(1, keepdims=True); dv2[j] = dv2[j] - Nn * np.minimum(dn, 0.0)
            dv2[j] /= np.linalg.norm(dv2[j], axis=1, keepdims=True); contact[idx[j]] += 1
        minc[idx[past]] = np.minimum(minc[idx[past]], (sdf(Pn[past]) - rr[past]))
        D[idx] = dv2; path[k] = path[k - 1]; path[k, idx] = Pn; s[idx] = s2
    pos = np.zeros((n, npt, 3)); arc = np.zeros(n); S = np.zeros((n, npt))
    for i in range(n):
        p = path[:, i]; seg = np.linalg.norm(np.diff(p, axis=0), axis=1); keep = np.r_[True, seg > 1e-12]
        p = p[keep]; cs = np.r_[0, np.cumsum(seg[seg > 1e-12])]; arc[i] = cs[-1]
        sj = np.linspace(0, cs[-1], npt); S[i] = sj
        pos[i] = np.stack([np.interp(sj, cs, p[:, a]) for a in range(3)], 1)
    rad = rfun(S)
    return {"pos": pos, "rad": rad, "len": arc, "contact": contact, "min_clear": minc}


def rfun_at(rfun, idx, s):
    """rfun (a Radius) for a subset of hairs idx (k,) at one arc each s (k,)"""
    return rfun.sub(idx, s)


class Radius:
    """per-hair radius law (radius()) usable both on the full set (n, m) and on a subset at one arc each"""
    def __init__(self, L, r0, mid_ratio, taper, tip_ratio):
        self.L = np.asarray(L, float); self.r0 = np.asarray(r0, float); self.n = len(self.L)
        self.mid = np.broadcast_to(np.asarray(mid_ratio, float), (self.n,)); self.tz = np.broadcast_to(np.asarray(taper, float), (self.n,))
        self.tip = np.broadcast_to(np.asarray(tip_ratio, float), (self.n,))

    def __call__(self, S):
        return radius(S, self.L[:, None], self.r0[:, None], self.mid[:, None], self.tz[:, None], self.tip[:, None])

    def sub(self, idx, s):
        return radius(s, self.L[idx], self.r0[idx], self.mid[idx], self.tz[idx], self.tip[idx])


def bind(pos, groups, share, start=0.35, sdf=None, rad=None, gap=0.02e-3):
    """binding (water, tear film, sebum, product drawing neighbouring fibres together): within each group (list of index
    arrays), every member's point at normalised arc q moves toward the group's mean curve at q by share x w(q), w rising
    smoothly from 0 at q = start to 1 at the tip (the root is held by its follicle). share (scalar or per group) 0 = free,
    1 = the distal parts meet. After the move, points that came within gap + r of the sdf are pushed back out."""
    pos = pos.copy(); npt = pos.shape[1]; q = np.linspace(0, 1, npt)
    x = np.clip((q - start) / max(1 - start, 1e-6), 0, 1); w = x * x * (3 - 2 * x)
    for gi, g in enumerate(groups):
        g = np.asarray(g)
        if len(g) < 2:
            continue
        c = np.asarray(share[gi] if np.ndim(share) else share, float)
        mean = pos[g].mean(0)
        pos[g] = pos[g] + (c * w)[None, :, None] * (mean[None] - pos[g])
    if sdf is not None and rad is not None:
        P = pos.reshape(-1, 3); r = rad.reshape(-1); d = sdf(P); need = r + gap
        bad = d < need; bad &= np.tile(q > 0.05, len(pos))
        if bad.any():
            Nn, _ = grad(sdf, P[bad]); P[bad] = P[bad] + Nn * (need[bad] - d[bad])[:, None]
        pos = P.reshape(pos.shape)
    return pos


def pack_bundles(pos, rad, size=(12, 60), phi=(0.30, 0.50), start=(0.08, 0.25), free_share=0.10, arcs=(0.3, 0.55, 0.8), seed=0, log=print,
                 pull=(1.0, 1.0)):
    """binding into packed bundles (the cause: sebum / water bridges and the fibres' own alignment pull neighbours into
    touching bundles; a real lock packs at phi 0.3-0.6, AUDIT 3). Neighbours along their length (their points at `arcs` of
    the point index, k-NN in that joint space) are gathered greedily into bundles of size[0]..size[1] fibres; past each
    bundle's own start (a share of the length: the root stays free in its follicle) every member moves toward the bundle's
    mean curve, to the radius its rank takes in a disc packed at phi (its own direction about the mean kept), so the bundle's
    section is the fibres' own area / phi and gaps open between bundles. free_share of the fibres stay as they were (strays,
    the halo). The strands' points are taken as evenly spaced along them (HairSet / the engine's export). -> pos, info"""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed); n, m = pos.shape[:2]
    F = np.concatenate([pos[:, int(round(a * (m - 1)))] for a in arcs], 1)
    tree = cKDTree(F); free = rng.random(n) < free_share; assigned = free.copy(); bundles = []
    for i in rng.permutation(n):
        if assigned[i]:
            continue
        k = int(rng.integers(size[0], size[1] + 1))
        _, nb = tree.query(F[i], k=min(4 * k, n)); nb = np.atleast_1d(nb); nb = nb[~assigned[nb]][:k]
        assigned[nb] = True; bundles.append(nb)
    out = pos.copy(); radii = []; qq = np.linspace(0, 1, m)
    for b in bundles:
        if len(b) < 2:
            continue
        Pb = pos[b]; mean = Pb.mean(0); off = Pb - mean[None]
        ph = rng.uniform(*phi); r_f = float(np.median(rad[b, m // 2])); Rb = np.sqrt(len(b) * r_f * r_f / ph)
        rank = np.argsort(np.argsort(np.linalg.norm(off[:, m // 2], axis=1)))
        rt = Rb * np.sqrt((rank + 0.5) / len(b))
        d = off / np.maximum(np.linalg.norm(off, axis=2, keepdims=True), 1e-9)
        st = rng.uniform(*start); x = np.clip((qq - st) / 0.15, 0, 1); w = x * x * (3 - 2 * x) * rng.uniform(*pull)
        out[b] = Pb + w[None, :, None] * (mean[None] + d * rt[:, None, None] - Pb)
        radii.append(Rb)
    info = {"bundles": len(bundles), "free": int(free.sum()), "bundle_radius_mm_p10_50_90": (np.percentile(radii, [10, 50, 90]) * 1e3).round(3).tolist()}
    log("pack_bundles: %s" % info)
    return out, info
