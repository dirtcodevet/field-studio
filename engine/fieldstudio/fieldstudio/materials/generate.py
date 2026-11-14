"""fs-materials (2026-11-14): the seeded generators. Same seed, same result; a new seed, a new stone.

Each generator turns (a library material, its finish, a seed, a few settings, the object's own form and lattice) into
an OP_MATGEN parameter block (materials/fieldops.py), so what it makes is structure in the field:
  stone_relief(look, finish, seed, h, rough)      the stone's surface: undulation + plucked-grain pits, band-limited to
                                                  what the object's lattice carries (wavelengths >= 4 h, pits >= 2 h)
  edge_chips(child, seed, h, density, ...)         conchoidal chips on the form's EDGES, found from the field itself
                                                  (high mean curvature on the surface), so any form gets them
  crack(child, seed, h, depth_frac, width, ...)    a fracture into the volume from a face, across the form's long axis
Detail finer than the lattice is the material's look (its BRDF roughness and its composition field: look.py), not
geometry; the render-time grain relief is the next step (fs-materials CONTINUE.md).

Numbers: each default names its source in STONE (materials/library.py resolves the profile's look block); E = an
estimate until the lane's research lands (fs_build/fs-materials/research/STONE_LOOK_RESEARCH.md).
"""
import math

import numpy as np
from numba import njit, prange

from . import fieldops as MF
from ..core import ops
from ..core import program as P


# ---- the child's field, sampled ---------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _eval_pts(code, prm, c1, pts, out):
    for i in prange(pts.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        out[i] = P.eval_prog(code, prm, 0, c1, pts[i, 0], pts[i, 1], pts[i, 2], vs, ps)


class Field:
    """a base tree (dict or Node) evaluated at local points"""

    def __init__(self, tree):
        node = tree if isinstance(tree, ops.Node) else ops.Node.from_dict(tree)
        self.code, self.prm, self.lo, self.hi, self.L = ops.compile_tree(node)

    def __call__(self, pts):
        pts = np.ascontiguousarray(np.asarray(pts, np.float64).reshape(-1, 3))
        out = np.empty(len(pts))
        _eval_pts(self.code, self.prm, self.code.shape[0], pts, out)
        return out

    def grad(self, pts, e=1e-4):
        pts = np.asarray(pts, float).reshape(-1, 3)
        g = np.empty_like(pts)
        for a in range(3):
            d = np.zeros(3)
            d[a] = e
            g[:, a] = (self(pts + d) - self(pts - d)) / (2 * e)
        return g


def surface_samples(F, step):
    """points on F = 0 over the form's box at ~step spacing (grid points within step of the surface, projected by two
    Newton steps), their unit normals and the field's Laplacian there (2 x mean curvature for a distance field)"""
    lo, hi = F.lo - 2 * step, F.hi + 2 * step
    ax = [np.arange(lo[k], hi[k] + 1e-12, step) for k in range(3)]
    X, Y, Z = np.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
    G = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    f = F(G)
    Q = G[np.abs(f) < 0.75 * step]
    e = 0.5 * step
    I = np.eye(3)
    for it in range(3):
        # one call for the point and its 6 neighbours: the gradient (Newton) and, at the last pass, the Laplacian
        P7 = np.concatenate([Q] + [Q + e * I[a] for a in range(3)] + [Q - e * I[a] for a in range(3)], 0)
        v = F(P7).reshape(7, -1)
        g = np.stack([(v[1 + a] - v[4 + a]) / (2 * e) for a in range(3)], 1)
        if it < 2:
            gn = np.maximum((g * g).sum(1), 1e-12)
            Q = Q - (v[0] / gn)[:, None] * g
    n = g / np.maximum(np.linalg.norm(g, axis=1), 1e-12)[:, None]
    lap = (v[1:].sum(0) - 6 * v[0]) / (e * e)
    return Q, n, lap


# ---- the stone's surface ------------------------------------------------------------------------------------------------
# finish -> relief at form scale (E until the research lands): undulation RMS (m) and its wavelength band (m), pits
# 11-14: a CUT face (sawn, rubbed, honed, droved) carries no pits: round bowls read as concrete air
# bubbles; its porosity is the grain-scale interstices of the render-time grain relief, and a dressed building stone's
# rare visible voids are flattened clay galls along the bed (to build from references). Broken faces (riven, split) keep
# plucked-grain sockets.
# (plucked grains and vugs: cell (m), chance per cell, depth (m)). The roughness slider scales the undulation and pits.
FINISH = {
    'sawn':          {'rms': 30e-6, 'band': (1e-3, 12e-3), 'hurst': 0.8, 'pit_cell': 1.6e-3, 'pit_prob': 0.0,
                      'pit_depth': 0.12e-3, 'pit_rfac': 0.35},   # Rq 30 um ~ sawn granite Ra 22-25 um (Sanmartin 2011)
    'rubbed':        {'rms': 12e-6, 'band': (1e-3, 12e-3), 'hurst': 0.8, 'pit_cell': 1.6e-3, 'pit_prob': 0.0,
                      'pit_depth': 0.12e-3, 'pit_rfac': 0.30},
    'honed':         {'rms': 5e-6, 'band': (1e-3, 12e-3), 'hurst': 0.8, 'pit_cell': 1.6e-3, 'pit_prob': 0.0,
                      'pit_depth': 0.08e-3, 'pit_rfac': 0.30},
    'polished':      {'rms': 1.5e-6, 'band': (1e-3, 12e-3), 'hurst': 0.8, 'pit_cell': 2.0e-3, 'pit_prob': 0.0,
                      'pit_depth': 0.0, 'pit_rfac': 0.30},     # polished: Ra 0.3-0.6 um (granite, Carrara: research)
    'riven_or_tooled': {'rms': 0.45e-3, 'band': (1.2e-3, 40e-3), 'hurst': 0.75, 'pit_cell': 2.0e-3, 'pit_prob': 0.14,
                        'pit_depth': 0.30e-3, 'pit_rfac': 0.38},
    # droved (dressed) face: chisel strokes ~3-4 mm apart across the face (research: the droved face with a mason's mark,
    # striations about 3-4 mm; R4 / Eggesandstein tooled blocks), 0.12 mm deep, their lines wandering (E)
    'droved':        {'rms': 40e-6, 'band': (1e-3, 12e-3), 'hurst': 0.8, 'pit_cell': 1.6e-3, 'pit_prob': 0.0,
                      'pit_depth': 0.12e-3, 'pit_rfac': 0.35,
                      'strokes': {'axis': (1.0, 0.0, 0.0), 'spacing': 3.5e-3, 'amp': 0.10e-3, 'warp': 0.03,
                                  'warp_freq': 25.0, 'passes': 2, 'jitter': 0.4, 'groove': 0.4,
                                  'blow': 10e-3, 'blow_depth': 0.3, 'blow_off': 0.0}},
    # 11-14 (update 20): the blows: a claw chisel advances ~6-10 mm a blow (E: R4's grooves change every ~5-10 mm),
    # each blow its own depth (0.85 +- 0.3, a break below zero: 0.55 broke R4's mostly continuous grooves into
    # dashes, sheet_20) and no sideways jog (one jog noise is shared by all grooves, so
    # they wiggled in step: horizontal bands; 0.12 read as wavy snakes, sheet_20)
    # 11-14: a claw chisel (the reference: tooth pitch 3.5-5 mm from its spectrum, grooves a median 1.75 mm apart from
    # 0.5 to 4.8 mm, measured): two overlapping passes at a 3.5 mm pitch, each tooth a narrow groove (0.4 of the pitch)
    # on a land, each cell's width +-20 % (the slope bound 0.7); depth matched by raking light
    'split_face':    {'rms': 1.6e-3, 'band': (1.5e-3, 60e-3), 'hurst': 0.7, 'pit_cell': 2.4e-3, 'pit_prob': 0.16,
                      'pit_depth': 0.40e-3, 'pit_rfac': 0.40},
}


def stone_relief(finish, seed, h, rough=1.0, bed_normal=(0.0, 1.0, 0.0)):
    """-> relief block. Band-limited to the lattice: no wavelength under 4 h, no pit under 2 h radius."""
    F = FINISH.get(finish, FINISH['sawn'])
    rng = np.random.default_rng([int(seed), 101])
    lam_lo = max(F['band'][0], 4.0 * h)
    lam_hi = max(F['band'][1], lam_lo)
    recs, _ = MF.octaves(rng, lam_lo, lam_hi, F['rms'] * rough, hurst=F['hurst'])
    cell = max(F['pit_cell'], 2.0 * h / max(F['pit_rfac'], 1e-3) * 1.0)
    pits = {'cell': cell if F['pit_prob'] > 0.0 else 0.0,              # no pits: the cellular loop skipped
            'prob': min(0.5, F['pit_prob'] * (0.6 + 0.4 * rough)), 'depth': F['pit_depth'] * rough,
            'rfac': F['pit_rfac'], 'seed': int(seed) * 7 + 3}
    # band = 0: F - r as it is (continuous, the zero set exact). Normalising by sqrt(1 + |grad r|^2) (band > 0) made the
    # field jump where the deepest pit changes (grad r is not continuous there): measured 11-14, max |grad| 1.8 -> 5.2
    st = F.get('strokes')
    if st:
        # tool strokes (a droved or tooled face): parallel grooves across the axis (the long faces show them; the ends,
        # constant along the axis, stay sawn), their line wandering by the warp; the seed turns them a few degrees
        ax = np.asarray(st['axis'], float)
        tw = math.radians(rng.uniform(-4.0, 4.0))
        ax = np.array([ax[0] * math.cos(tw) - ax[2] * math.sin(tw), ax[1], ax[0] * math.sin(tw) + ax[2] * math.cos(tw)])
        bed = {'normal': ax, 'spacing': st['spacing'] * rng.uniform(0.9, 1.1), 'amp': st['amp'] * rough,
               'warp': st['warp'], 'warp_freq': st['warp_freq'], 'passes': st.get('passes', 1),
               'jitter': st.get('jitter', 0.0), 'groove': st.get('groove', 1.0),
               'blow': st.get('blow', 0.0), 'blow_depth': st.get('blow_depth', 0.0), 'blow_off': st.get('blow_off', 0.0)}
    else:
        bed = {'normal': bed_normal, 'spacing': 0.01, 'amp': 0.0}
    return MF.pack_relief(recs, pits=pits, bed=bed, band=0.0)


# ---- chips on the edges -----------------------------------------------------------------------------------------------
def _edge_frame(F, p, n, e):
    """the edge's tangent at p: the Hessian's eigenvector with the smallest |curvature| across the normal"""
    H = np.zeros((3, 3))
    f0 = F(p[None])[0]
    I = np.eye(3)
    for a in range(3):
        for b in range(a, 3):
            if a == b:
                H[a, a] = (F((p + e * I[a])[None])[0] + F((p - e * I[a])[None])[0] - 2 * f0) / (e * e)
            else:
                v = F(np.array([p + e * (I[a] + I[b]), p + e * (I[a] - I[b]), p - e * (I[a] - I[b]),
                                p - e * (I[a] + I[b])]))
                H[a, b] = H[b, a] = (v[0] - v[1] - v[2] + v[3]) / (4 * e * e)
    Pn = np.eye(3) - np.outer(n, n)
    w, V = np.linalg.eigh(Pn @ H @ Pn)
    cand = [(abs(w[k]), V[:, k]) for k in range(3) if abs(V[:, k] @ n) < 0.5]
    t = min(cand, key=lambda c: c[0])[1] if cand else np.cross(n, [0.0, 1.0, 0.0])
    t = t - (t @ n) * n
    return t / max(np.linalg.norm(t), 1e-12)


def _project(F, x, n_it=4):
    """a point moved onto F = 0 along the field's gradient (Newton)"""
    x = np.asarray(x, float).copy()
    for _ in range(n_it):
        g = F.grad(x[None])[0]
        x = x - F(x[None])[0] * g / max(float(g @ g), 1e-12)
    return x


def _arris(F, p, n, t, d=6e-3):
    """the arris near the edge point p (tangent t): the faces either side (the surface projected at d across the edge
    each way), the bisector of their normals, and the arris point: the surface along the bisector from the line where
    the two face planes meet (nearest p). -> (point, bisector), or (p, n) where p is not on an arris (one face)"""
    b = np.cross(n, t)
    b /= max(np.linalg.norm(b), 1e-12)
    qa, qb = _project(F, p - b * d), _project(F, p + b * d)
    na, nb = F.grad(qa[None])[0], F.grad(qb[None])[0]
    na /= max(np.linalg.norm(na), 1e-12)
    nb /= max(np.linalg.norm(nb), 1e-12)
    if na @ nb > 0.94:                                  # under ~20 deg between them: not an arris
        return p, n
    bis = na + nb
    bis /= np.linalg.norm(bis)
    # the planes' line: x = p + a na + c nb with na.(x - qa) = 0, nb.(x - qb) = 0
    A = np.array([[1.0, na @ nb], [na @ nb, 1.0]])
    rhs = np.array([na @ (qa - p), nb @ (qb - p)])
    a, c = np.linalg.solve(A, rhs)
    x = p + a * na + c * nb
    for _ in range(8):                                  # in along the bisector to the surface (the rounded arris)
        f = F(x[None])[0]
        x = x - f * bis
        if abs(f) < 1e-7:
            break
    return x, bis


def crumble_params(seed, per_m):
    """the arris crumble's bite parameters edge_chips packs (A, lam, tau, seed, A2, lam2), for the look's run-off
    sources (look.cuh: water leaves the top where the arris is bitten); tests check they match the packed block"""
    k = min(per_m / 12.0, 1.0)
    return {'A': 0.6e-3 * k, 'lam': 12e-3, 'tau': 0.0, 'seed': int(seed) * 13 + 7, 'A2': 0.13e-3 * k, 'lam2': 4e-3}


def edge_chips(F, seed, h, per_m=12.0, s_min=None, s_max=20e-3, alpha=1.5, corner_p=0.5, step=None,
               rough_rms=250e-6, box=None, wear=None):
    """conchoidal chips on the convex edges of the form F (a Field), seeded. per_m: chips per metre of edge (edge wear);
    sizes from a power law N(> s) ~ s^-alpha over [s_min, s_max] (fragment-size practice, E). wear: weathering's worn
    arrises {R (m, the mean radius added), lam, var} about the as-made box (needs box). -> (block, info)"""
    rng = np.random.default_rng([int(seed), 202])
    wear_c = None
    if box is not None and wear and float(wear.get('R', 0.0)) > 0.0:
        half, rnd = box
        wear_c = {'half': half, 'sag': rnd * (math.sqrt(2.0) - 1.0), 'A': 0.0, 'lam': 6e-3, 'tau': 0.0,
                  'seed': int(seed) * 13 + 7,
                  'wear': {'r0': rnd, 'R': float(wear['R']), 'lam': float(wear.get('lam', 80e-3)),
                           'var': float(wear.get('var', 0.35)), 'seed': int(seed) * 17 + 9}}
    # chips of ~4-20 mm along the arris on handled bricks (11-14, measured on brick_pile_chipped_edges at ~0.31 mm/px:
    # 6-19 mm; brick_newhey_whole_chipped at ~0.21 mm/px: 4-17 mm); smaller than the arris's rounding they only dent it
    s_min = max(s_min or 0.0, 6.0 * h, 4.0e-3)
    step = step or max(1.5 * h, 1.5e-3)
    Q, N, lap = surface_samples(F, step)
    edge = lap > 0.6 / max(4.0 * step, 1e-6)           # convex, sharper than ~ a 4-step radius
    E = Q[edge]
    EN = N[edge]
    EL = lap[edge]
    info = {'surface_pts': int(len(Q)), 'edge_pts': int(len(E))}
    if len(E) == 0 or per_m <= 0.0:
        return MF.pack_chips([], (), wear_c), dict(info, chips=0, edge_m=0.0)
    # the edge's length: a greedy cover of the edge points at gap g0 (count x g0)
    g0 = 5e-3
    order = rng.permutation(len(E))
    acc = []
    for i in order:
        if not acc or np.min(np.linalg.norm(E[acc] - E[i], axis=1)) > g0:
            acc.append(i)
    L_edge = len(acc) * g0
    n_chips = int(rng.poisson(per_m * L_edge))
    chips = []
    tries = 0
    corner_thr = np.percentile(EL, 97) if len(EL) > 30 else np.inf
    while len({c['cluster'] for c in chips}) < n_chips and tries < 20 * n_chips + 20:
        tries += 1
        i = int(rng.integers(len(E)))
        if EL[i] >= corner_thr and rng.random() > corner_p:
            continue
        p, n = E[i], EN[i]
        if any(np.linalg.norm(c['centre'] - p) < 0.6 * max(c['radii']) for c in chips if c['primary']):
            continue
        u = rng.random()
        s = s_min * (1.0 - u * (1.0 - (s_min / s_max) ** alpha)) ** (-1.0 / alpha)
        t = _edge_frame(F, p, n, max(step, 2 * h))
        p, n = _arris(F, p, n, t)                     # 11-14: on the arris itself, about its true bisector
        t = t - (t @ n) * n
        t /= max(np.linalg.norm(t), 1e-12)
        b = np.cross(n, t)
        # a chip is a CLUSTER of overlapping flake scars along the arris (11-14, the references: the outline ragged and
        # angular where the scars meet), the first the largest; each scar a shallow near-flat facet (its depth axis
        # 2-3.5 x the cut), elongated along the edge, tilted to bite more into one face
        n_sc = 1 + int(rng.choice(4, p=[0.35, 0.35, 0.2, 0.1]))
        cid = len(chips)
        for j in range(n_sc):
            sj = s if j == 0 else s * rng.uniform(0.35, 0.7)
            off = 0.0 if j == 0 else rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.8) * 1.0 * (s + sj)
            pj = p + t * off
            tilt = math.radians(rng.uniform(-20.0, 20.0))  # within 20 deg of the bisector: the facet takes the arris
            nt = math.cos(tilt) * n + math.sin(tilt) * b
            bt = np.cross(nt, t)
            cut = sj * rng.uniform(0.25, 0.40)            # the scar's depth below the arris (a notch, not a dent)
            a_n = cut * rng.uniform(2.0, 3.5)
            a_t = 1.0 * sj * rng.uniform(0.8, 1.25)       # along the edge: the edge line gone along the scar
            a_b = 0.35 * sj * rng.uniform(0.8, 1.2)       # across it (a scar wider along the arris than deep)
            c = pj + nt * (a_n - cut)
            chips.append({'centre': c, 'R': np.stack([t, bt, nt]), 'radii': [a_t, a_b, a_n], 'size': sj,
                          'cluster': cid, 'primary': j == 0})
    # the scars' fracture surface: granular, rough at the mm scale (11-14: 120 um read as smooth lens-shaped dishes with
    # crisp elliptical rims; 250 um over 1.5-6 mm frays the rim and roughens the facet; the op's slope bound 1.85 -> 2.34)
    recs, _ = MF.octaves(rng, max(4 * h, 1.5e-3), 6e-3, rough_rms, hurst=0.8)
    # the arris crumble (11-14): the as-fired box's arrises ragged along their length (the same-scale references:
    # 1-5 mm bites, continuous); bite A 0.5 mm at the default wear (less below it), wavelength 6 mm. The cap is the
    # card's memory: the chip op's slope bound L sets the shell of bricks kept about the surface (gpu/mirror reach =
    # L x half a brick diagonal): + A 3.71 x 3 / 6 mm = +0.93 here (L 2.34 -> ~3.3); A 1.3 mm at 5 mm made it 5.2
    crumble = None
    if box is not None and per_m > 0.0:
        half, rnd = box
        # 11-14 (update 13): two incommensurate bite octaves, their A / lambda summing to the single 0.5 mm / 6 mm one's
        # (the same slope bound, so the same L): bites of varied size and spacing, the deepest ~0.7 mm
        k = min(per_m / 12.0, 1.0)
        crumble = {'half': half, 'sag': rnd * (math.sqrt(2.0) - 1.0), 'A': 0.6e-3 * k, 'lam': 12e-3, 'tau': 0.0,
                   'seed': int(seed) * 13 + 7, 'A2': 0.13e-3 * k, 'lam2': 4e-3}
        if wear_c:
            crumble['wear'] = wear_c['wear']
    blk = MF.pack_chips(chips, recs, crumble)
    # update 21: the top arris's chips are where water on the top collects and runs off (the look's run-off sources):
    # (x, z, along-edge radius, strength) for the largest scars near the top (the as-made box)
    src = []
    if box is not None:
        half = box[0]
        for c in sorted((c for c in chips if c['primary']), key=lambda c: -c['size']):
            ce = c['centre']
            if ce[1] > half[1] - 0.6 * c['size'] and len(src) < 8:
                src.append([round(float(ce[0]), 6), round(float(ce[2]), 6), round(float(c['radii'][0]), 6),
                            round(float(min(max(c['size'] / 10e-3, 0.4), 1.0)), 3)])
    return blk, dict(info, chips=len({c['cluster'] for c in chips}), scars=len(chips), edge_m=round(L_edge, 4),
                     sizes_mm=[round(c['size'] * 1e3, 2) for c in chips if c['primary']], top_sources=src)


# ---- a crack ----------------------------------------------------------------------------------------------------------
def crack(F, seed, h, depth_frac=0.5, width=0.7e-3, up=(0.0, 1.0, 0.0), along=None, at=None, through=True,
          rough_rms=0.35e-3, wander_rms=1.2e-3):
    """a fracture from the face most facing `up` (object frame) into the volume, across the form's long axis (bricks
    crack across their length under bending); depth_frac of the form's extent along the inward normal; width = the
    opening at the mouth (m); the fracture surface self-affine (Hurst 0.8, E: fracture-surface roughness exponent),
    with a long-wavelength wander (the crack's path across the face) and a grain-scale roughness. -> (block, info)"""
    rng = np.random.default_rng([int(seed), 303])
    Q, N, _ = surface_samples(F, max(2.0 * h, 2.0e-3))
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    C = Q.mean(0)
    X = Q - C
    w_, V = np.linalg.eigh(X.T @ X)
    long_ax = V[:, -1] if along is None else np.asarray(along, float) / np.linalg.norm(along)
    top = N @ up > 0.95
    if not top.any():
        top = N @ up > 0.7
    T = Q[top]
    s_al = (T - C) @ long_ax
    frac = float(rng.uniform(0.32, 0.68)) if at is None else float(at)
    pos = s_al.min() + frac * (s_al.max() - s_al.min())
    c_top = T[np.argmin(np.abs(s_al - pos) + 0.1 * np.linalg.norm((T - C) - np.outer((T - C) @ long_ax, long_ax), axis=1))]
    e = -up                                            # into the volume
    ang = math.radians(rng.uniform(-12.0, 12.0))       # the plane turned off square about the face normal
    side = np.cross(up, long_ax)
    side = side / np.linalg.norm(side)
    n = math.cos(ang) * long_ax + math.sin(ang) * side
    n = n - (n @ e) * e
    lean = math.radians(rng.uniform(-8.0, 8.0))        # and leaning in depth
    n = math.cos(lean) * n + math.sin(lean) * e
    n = n / np.linalg.norm(n)
    e = e - (e @ n) * n
    e = e / np.linalg.norm(e)
    t = np.cross(n, e)
    # depth: the form's thickness along e from the mouth (where the line from the mouth leaves the form)
    ss = np.arange(1, 4001) * 0.25e-3
    fl = F(c_top[None] + ss[:, None] * e[None])
    out_ = np.nonzero(fl > 0.0)[0]
    ext = float(ss[out_[0]]) if len(out_) else float(((Q - c_top) @ e).max())
    D = depth_frac * ext
    recs_w, _ = MF.octaves(rng, 12e-3, 120e-3, wander_rms, hurst=0.8, per_octave=2.0)
    recs_r, _ = MF.octaves(rng, max(4 * h, 0.8e-3), 10e-3, rough_rms, hurst=0.8, per_octave=2.0)
    blk = MF.pack_crack(c_top, n, e, t, D, width, half_len=(1e3 if through else 0.4 * ext), tip_amp=0.08,
                        tip_freq=25.0, cap=0.15, seed=int(seed) * 13 + 5, oct_recs=recs_w + recs_r)
    return blk, {'mouth': c_top.round(5).tolist(), 'normal': n.round(4).tolist(), 'depth_mm': round(D * 1e3, 2),
                 'width_mm': round(width * 1e3, 3)}


# ---- the tree --------------------------------------------------------------------------------------------------------
def wrap(tree, block, spec, nid):
    """the base tree (dict) wrapped in one matgen node -> dict"""
    return {'kind': 'matgen', 'params': {'block': [float(v) for v in block], 'spec': spec}, 'children': [tree],
            'id': int(nid), 'name': spec.get('name')}


def strip(tree):
    """the base tree without its matgen wrappers (the form a generator works on) -> (form dict, [the wrappers' specs])"""
    specs = []
    while tree.get('kind') == 'matgen':
        specs.append(tree['params'].get('spec', {}))
        tree = tree['children'][0]
    return tree, specs[::-1]
