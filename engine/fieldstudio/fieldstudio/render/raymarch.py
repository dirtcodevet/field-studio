"""CPU sphere tracer for the field scene (numba, parallel over rows; no GPU, no mesh).

Conservative stepping: at each step the tracer advances by d = min_i F_i(p) / L_i, the per-object tracked
Lipschitz bound turning F into a distance lower bound, so a step never crosses a zero set (D, given the
bounds hold). Hit when d < eps(t) = max(eps_min, t * pixel_angle * eps_scale). Rays that run out of steps are
counted as 'unresolved' (the miss census), never silently treated as background.

render(scene, camera, w, h, quality, mode) -> RenderResult(rgb uint8 (h,w,3), depth float32 (h,w) (inf = miss),
normal float32 (h,w,3), obj int32 (h,w) (object id, -1 = miss), stats dict)
modes: 'shaded' (key + fill + rim, AO from field samples), 'matcap', 'normal', 'slice' (field slice through the
camera target, perpendicular to the view: blue inside, orange outside, iso bands, the zero set in white).
quality: 0 = preview (half resolution, no AO), 1 = normal (AO 3 samples), 2 = high (AO 6, tighter eps).
"""
import math
import time
from collections import namedtuple

import numpy as np
from numba import njit, prange

from ..core import program as P
from ..core.program import STACK, eval_scene, eval_scene_f, eval_world, grid_sample, to_local

RenderResult = namedtuple('RenderResult', 'rgb depth normal obj stats')
MODES = {'shaded': 0, 'matcap': 1, 'normal': 2, 'slice': 3, 'curvature': 1}   # curvature: the card's (the CPU: matcap)
PickResult = namedtuple('PickResult', 'hit point normal object_id t')


@njit(cache=True, inline='always')
def _clip(ox, oy, oz, dx, dy, dz, lo, hi):
    t0 = 0.0
    t1 = 1e30
    o = (ox, oy, oz)
    d = (dx, dy, dz)
    for a in range(3):
        if abs(d[a]) < 1e-15:
            if o[a] < lo[a] or o[a] > hi[a]:
                return 1.0, 0.0
        else:
            ta = (lo[a] - o[a]) / d[a]
            tb = (hi[a] - o[a]) / d[a]
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
    return t0, t1


@njit(cache=True)
def ray_intervals(of_tab, ox, oy, oz, dx, dy, dz, tin, tout):
    """Per object: the ray's [t_in, t_out] through its guard box (the local box + margin, an oriented box in world
    space). Outside that box an object's zero set cannot be, so the tracer never evaluates it there (empty-space
    skipping per object). Hidden or missed objects get t_in = 1e30."""
    for oi in range(of_tab.shape[0]):
        tin[oi] = 1e30
        tout[oi] = -1e30
        if of_tab[oi, P.OF_VIS] < 0.5:
            continue
        lx, ly, lz = to_local(of_tab, oi, ox, oy, oz)
        s = of_tab[oi, P.OF_S]
        ex = (of_tab[oi, 0] * dx + of_tab[oi, 1] * dy + of_tab[oi, 2] * dz) / s
        ey = (of_tab[oi, 3] * dx + of_tab[oi, 4] * dy + of_tab[oi, 5] * dz) / s
        ez = (of_tab[oi, 6] * dx + of_tab[oi, 7] * dy + of_tab[oi, 8] * dz) / s
        a0 = 0.0
        a1 = 1e30
        ok = True
        for a in range(3):
            o = lx if a == 0 else (ly if a == 1 else lz)
            d = ex if a == 0 else (ey if a == 1 else ez)
            lo = of_tab[oi, P.OF_BMIN + a]
            hi = of_tab[oi, P.OF_BMAX + a]
            if abs(d) < 1e-15:
                if o < lo or o > hi:
                    ok = False
            else:
                ta = (lo - o) / d
                tb = (hi - o) / d
                if ta > tb:
                    ta, tb = tb, ta
                a0 = max(a0, ta)
                a1 = min(a1, tb)
        if ok and a0 <= a1:
            tin[oi] = a0
            tout[oi] = a1


@njit(cache=True, inline='always')
def local_step(oi_tab, of_tab, index_flat, Lb, lbase, oi, x, y, z, dx, dy, dz, f):
    """V1.3b per-brick L: a safe step for object oi at world point (x, y, z) along unit (dx, dy, dz) from the bound
    of the brick that holds the point, valid up to where the ray leaves that brick (-1 when it does not apply).
    Along the segment inside a cube where |grad F| <= L_loc, F(p + t d) >= F(p) - L_loc t > 0 for t < F(p) / L_loc,
    so min(F / L_loc, t_exit) never crosses the zero set (D). L_loc = max(L_base + Lb[brick], 1): an allocated
    brick's Lb covers every cell of its cube [8b, 8b + 8] h (brick_lipschitz reads cells from 8b - 1 to 8b + 7);
    in an unallocated brick the delta is exactly 0 on [8b, 8b + 7] h (all 8 nodes of each cell are its own), so
    L_loc = L_base there. Steep bricks only slow the rays that pass through them."""
    if f <= 0.0:
        return -1.0
    nx = oi_tab[oi, P.OI_NX]
    hh = of_tab[oi, P.OF_H]
    lx, ly, lz = to_local(of_tab, oi, x, y, z)
    s = of_tab[oi, P.OF_S]
    ex = (of_tab[oi, 0] * dx + of_tab[oi, 1] * dy + of_tab[oi, 2] * dz) / s
    ey = (of_tab[oi, 3] * dx + of_tab[oi, 4] * dy + of_tab[oi, 5] * dz) / s
    ez = (of_tab[oi, 6] * dx + of_tab[oi, 7] * dy + of_tab[oi, 8] * dz) / s
    bw = P.BRICK * hh
    lb0 = max(lbase[oi], 1.0)
    if lb0 >= of_tab[oi, P.OF_L]:
        return -1.0
    # outside the sculpt index box grown by one node, every node a cell reads is outside the index: delta = 0
    # there, so L_loc = L_base until the ray enters that box
    if nx == 0:
        return f / lb0
    g0 = 1e30
    g1 = -1e30
    inside = True
    for a in range(3):
        o = lx if a == 0 else (ly if a == 1 else lz)
        e = ex if a == 0 else (ey if a == 1 else ez)
        org = oi_tab[oi, P.OI_OX + a]
        cnt = oi_tab[oi, P.OI_NX + a]
        b0 = org * bw - hh
        b1 = (org + cnt) * bw + hh
        if o < b0 or o > b1:
            inside = False
        if abs(e) < 1e-15:
            if o < b0 or o > b1:
                g0 = 1e30
                g1 = -1e30
                break
            continue
        ta = (b0 - o) / e
        tb = (b1 - o) / e
        if ta > tb:
            ta, tb = tb, ta
        if g0 >= 1e29:
            g0 = ta
            g1 = tb
        else:
            g0 = max(g0, ta)
            g1 = min(g1, tb)
    if not inside:
        tent = 1e30
        if g0 < 1e29 and g0 <= g1 and g1 >= 0.0:
            tent = max(g0, 0.0)
        return min(f / lb0, tent)
    bx = int(math.floor(lx / bw))
    by = int(math.floor(ly / bw))
    bz = int(math.floor(lz / bw))
    slot = -1
    if nx > 0:
        ny = oi_tab[oi, P.OI_NY]
        nz = oi_tab[oi, P.OI_NZ]
        ix = bx - oi_tab[oi, P.OI_OX]
        iy = by - oi_tab[oi, P.OI_OY]
        iz = bz - oi_tab[oi, P.OI_OZ]
        if ix >= 0 and iy >= 0 and iz >= 0 and ix < nx and iy < ny and iz < nz:
            slot = index_flat[oi_tab[oi, P.OI_IDX] + (ix * ny + iy) * nz + iz]
    if slot >= 0:
        L = max(lbase[oi] + Lb[slot], 1.0)
        ext = bw
    else:
        L = max(lbase[oi], 1.0)
        ext = bw - hh
    if L >= of_tab[oi, P.OF_L]:
        return -1.0
    texit = 1e30
    for a in range(3):
        o = lx if a == 0 else (ly if a == 1 else lz)
        e = ex if a == 0 else (ey if a == 1 else ez)
        b0 = (bx if a == 0 else (by if a == 1 else bz)) * bw
        b1 = b0 + ext
        if o < b0 or o > b1:
            return -1.0
        if e > 1e-15:
            texit = min(texit, (b1 - o) / e)
        elif e < -1e-15:
            texit = min(texit, (b0 - o) / e)
    return min(f / L, texit)


@njit(cache=True)
def trace(code, prm, oi_tab, of_tab, index_flat, delta, Lb, lbase, ox, oy, oz, dx, dy, dz, t0, t1, eps_min, eps_k,
          max_steps, vs, ps, tin, tout):
    """-> (t, object index, steps, status) status 1 hit, 0 miss, 2 unresolved (ran out of steps).
    Only objects whose box holds the current point are evaluated, each step is min F_i / L_i over those (the local
    objects' own L), and a step never crosses into the next object's box: empty space between boxes is one step.
    V1.3b: each object's step is max(F / L_object, local_step) - the brick's own bound inside its cube."""
    ray_intervals(of_tab, ox, oy, oz, dx, dy, dz, tin, tout)
    nobj = oi_tab.shape[0]
    t = t0
    tprev = t0                              # hand2b: the ray's previous position (a bracket when a step overshoots)
    nmiss = 0
    for s in range(max_steps):
        x = ox + dx * t
        y = oy + dy * t
        z = oz + dz * t
        dbest = 1e30
        dsec = 1e30                         # hand2b: the safe step of every OTHER active object (probe limit)
        obest = -1
        tnext = 1e30
        for oi in range(nobj):
            if tin[oi] >= 1e29 or tout[oi] < t:
                continue
            if tin[oi] > t:
                if tin[oi] < tnext:
                    tnext = tin[oi]
                continue
            f = eval_world(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps)
            d = f / of_tab[oi, P.OF_L]
            dl = local_step(oi_tab, of_tab, index_flat, Lb, lbase, oi, x, y, z, dx, dy, dz, f)
            if dl > d:
                d = dl
            if d < dbest:
                if dbest < dsec:
                    dsec = dbest
                dbest = d
                obest = oi
            elif d < dsec:
                dsec = d
        if obest < 0:                       # in empty space: jump to the next box
            if tnext > t1:
                return t, -1, s + 1, 0
            t = tnext
            continue
        eps = max(eps_min, t * eps_k)
        if dbest < eps:
            # hand2 (2026-11-07): refine to the real zero crossing. dbest = F / L_loc < eps leaves the point up to
            # L_loc * eps off the surface, and L_loc jumps per 8 mm brick, which drew the sandpaper speckle, the
            # brick grid and the stair steps (M, grain diag: hit 0.44 mm off p50 -> 0.02 mm). Probe forward by F
            # (the L = 1 guess, doubling) to a sign change, then bisect; no sign change = near miss, keep marching.
            fa = eval_world(code, prm, oi_tab, of_tab, index_flat, delta, obest, x, y, z, vs, ps)
            f0 = fa
            ta = t
            tb = -1.0
            if fa <= 0.0:
                # hand2b: the last step (a near-miss step of eps is not a safe step) went past the surface: bracket
                # it between the previous position and here; inside at both ends -> the hit is here
                fp = eval_world(code, prm, oi_tab, of_tab, index_flat, delta, obest,
                                ox + dx * tprev, oy + dy * tprev, oz + dz * tprev, vs, ps)
                if fp <= 0.0 or tprev >= t:
                    return t, obest, s + 1, 1
                ta = tprev
                fa = fp
                tb = t
                fb = f0
            # hand2b: the probe evaluates obest only, so it must not pass where another object's surface could be:
            # every other active object is safe for dsec along the ray, and no new object starts before tnext
            # (verifier item 27: a probe of up to 15 F could step over a close neighbour's surface)
            # (floored at eps: the near-miss step below moves eps anyway; a probe held under eps beside a touching
            # neighbour found no crossing and sent 242 rays of the selftest fill-splice view out of steps)
            tlim = max(min(t + dsec, tnext), t + eps)
            stp = max(fa, eps_min)
            for kk in range(4 if tb < 0.0 else 0):
                tt = ta + stp
                last = False
                if tt >= tlim:
                    tt = tlim
                    last = True
                if tt <= ta:
                    break
                fb = eval_world(code, prm, oi_tab, of_tab, index_flat, delta, obest,
                                ox + dx * tt, oy + dy * tt, oz + dz * tt, vs, ps)
                if fb <= 0.0:
                    tb = tt
                    break
                ta = tt
                fa = fb
                stp *= 2.0
                if last:
                    break
            if tb < 0.0:
                # near miss (no sign change ahead): march on from t with the conservative step, never from the
                # probe's end (a probe step of F is not safe where |grad F| > 1, and skipped thin parts: core T6).
                # hand2b: after 32 near misses on one ray (a grazing ray along a silhouette, a face or a crevice) the
                # point is a hit only when it lies within a quarter of the pixel footprint (F <= eps / 4, the cone
                # rule); otherwise the ray goes on with a step of max(eps, half the point's own F) (the probe found
                # no crossing over F .. 15 F ahead), capped by the other objects' safe step. Before, the 33rd near
                # miss became a hit wherever it was (up to 0.86 mm off the surface where eps was far smaller, all on
                # silhouette pixels, M): the serrated silhouettes (verifier items 5b, 20).
                nmiss += 1
                tprev = t
                if nmiss > 32:
                    # graduated: within eps / 4 after 32 near misses, within eps after 64, and after 128 (a ray held
                    # beside a touching neighbour, whose safe step caps ours at eps) the point is taken as before
                    # hand2b, so no ray runs out of steps (selftest V5)
                    if f0 <= 0.25 * eps or (nmiss > 64 and f0 <= eps) or nmiss > 128:
                        return t, obest, s + 1, 1
                    t += min(max(dbest, eps, 0.5 * f0), max(dsec, eps))
                else:
                    t += max(dbest, eps)
                if t > t1:
                    return t, -1, s + 1, 0
                continue
            for kk in range(14):
                tm = 0.5 * (ta + tb)
                fm = eval_world(code, prm, oi_tab, of_tab, index_flat, delta, obest,
                                ox + dx * tm, oy + dy * tm, oz + dz * tm, vs, ps)
                if fm > 0.0:
                    ta = tm
                    fa = fm
                else:
                    tb = tm
                    fb = fm
                if tb - ta < 0.25 * eps:
                    break
            if fa - fb > 1e-30:                 # final secant inside the bracket
                return ta + (tb - ta) * (fa / (fa - fb)), obest, s + 1, 1
            return ta, obest, s + 1, 1
        tprev = t
        t += min(dbest, max(tnext - t, eps))
        if t > t1:
            return t, -1, s + 1, 0
    return t, -1, max_steps, 2


@njit(cache=True, inline='always')
def _f(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, vs, ps):
    f, d, o = eval_scene(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, vs, ps)
    return f


@njit(cache=True)
def normal_at(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, e, vs, ps):
    a = _f(code, prm, oi_tab, of_tab, index_flat, delta, x + e, y - e, z - e, vs, ps)
    b = _f(code, prm, oi_tab, of_tab, index_flat, delta, x - e, y - e, z + e, vs, ps)
    c = _f(code, prm, oi_tab, of_tab, index_flat, delta, x - e, y + e, z - e, vs, ps)
    d = _f(code, prm, oi_tab, of_tab, index_flat, delta, x + e, y + e, z + e, vs, ps)
    nx = a - b - c + d
    ny = -a - b + c + d
    nz = -a + b - c + d
    l = math.sqrt(nx * nx + ny * ny + nz * nz)
    if l < 1e-30:
        return 0.0, 1.0, 0.0
    return nx / l, ny / l, nz / l


@njit(cache=True)
def normal_lattice(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, H, vs, ps):
    """11-10 (the inside of a lattice store drew as blocks): on a lattice field (her store, a sculpt layer) the
    trilinear field's gradient jumps at every cell, so a point's own gradient shaded each 0.25 mm cell flat: blocks
    on the CPU drawing (the card shades from a C1 reconstruction, gpu/trace.cu r32). Here: the gradient at the 8 nodes
    of the cell holding the point (each over +-H), blended trilinearly: continuous across cells."""
    gx0 = math.floor(x / H)
    gy0 = math.floor(y / H)
    gz0 = math.floor(z / H)
    ax = x / H - gx0
    ay = y / H - gy0
    az = z / H - gz0
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = (ax if di else 1.0 - ax) * (ay if dj else 1.0 - ay) * (az if dk else 1.0 - az)
                if w <= 0.0:
                    continue
                nx, ny, nz = normal_at(code, prm, oi_tab, of_tab, index_flat, delta, (gx0 + di) * H, (gy0 + dj) * H,
                                       (gz0 + dk) * H, H, vs, ps)
                sx += w * nx
                sy += w * ny
                sz += w * nz
    l = math.sqrt(sx * sx + sy * sy + sz * sz)
    if l < 1e-30:
        return normal_at(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, H, vs, ps)
    return sx / l, sy / l, sz / l


# ---- fs-shell 11-11 (design 4.8): the view-clip list beside the section plane: box hide and hidden regions --------------
# Hidden means absent from the view. A box hides everything inside it; where the ray leaves a box inside a form, the pixel
# is the cut face (filled from the field through the exact reader, as the section's). A hidden region takes its surface
# away and, with it, the solid beneath it: an inside point belongs to the region its surface foot point (x - F n) lies
# in; the ray walks through that volume in lattice steps and shows the cut face where it meets solid of a shown region,
# or carries on tracing where it leaves the form. With no boxes and no hidden region nothing below runs (render_kernel
# calls trace() exactly as before: bitwise A/B, t27).
@njit(cache=True, inline='always')
def rid_at(index_flat, rid, oi_tab, oi, h, x, y, z):
    """the region id of the lattice node nearest local (x, y, z) (0 where no brick)"""
    nx = oi_tab[oi, P.OI_NX]
    if nx == 0:
        return 0
    ny = oi_tab[oi, P.OI_NY]
    nz = oi_tab[oi, P.OI_NZ]
    i = int(math.floor(x / h + 0.5))
    j = int(math.floor(y / h + 0.5))
    k = int(math.floor(z / h + 0.5))
    bx = (i >> P.BSH) - oi_tab[oi, P.OI_OX]
    by = (j >> P.BSH) - oi_tab[oi, P.OI_OY]
    bz = (k >> P.BSH) - oi_tab[oi, P.OI_OZ]
    if bx < 0 or by < 0 or bz < 0 or bx >= nx or by >= ny or bz >= nz:
        return 0
    slot = index_flat[oi_tab[oi, P.OI_IDX] + (bx * ny + by) * nz + bz]
    if slot < 0:
        return 0
    return int(rid[slot, i & P.BMASK, j & P.BMASK, k & P.BMASK])


@njit(cache=True, inline='always')
def _rid_node(index_flat, rid, oi_tab, oi, i, j, k):
    nx = oi_tab[oi, P.OI_NX]
    if nx == 0:
        return 0
    ny = oi_tab[oi, P.OI_NY]
    nz = oi_tab[oi, P.OI_NZ]
    bx = (i >> P.BSH) - oi_tab[oi, P.OI_OX]
    by = (j >> P.BSH) - oi_tab[oi, P.OI_OY]
    bz = (k >> P.BSH) - oi_tab[oi, P.OI_OZ]
    if bx < 0 or by < 0 or bz < 0 or bx >= nx or by >= ny or bz >= nz:
        return 0
    slot = index_flat[oi_tab[oi, P.OI_IDX] + (bx * ny + by) * nz + bz]
    if slot < 0:
        return 0
    return int(rid[slot, i & P.BMASK, j & P.BMASK, k & P.BMASK])


@njit(cache=True)
def hidden_frac(index_flat, rid, oi_tab, oi, h, x, y, z, hid):
    """how hidden local point (x, y, z) is: the 8 surrounding nodes' hidden flags blended trilinearly (0..1); the
    region's edge is where it crosses 0.5, a contour between the nodes rather than the nearest node's staircase"""
    fx = x / h
    fy = y / h
    fz = z / h
    i0 = int(math.floor(fx))
    j0 = int(math.floor(fy))
    k0 = int(math.floor(fz))
    tx = fx - i0
    ty = fy - j0
    tz = fz - k0
    acc = 0.0
    for c in range(8):
        a = c & 1
        b = (c >> 1) & 1
        e = (c >> 2) & 1
        w = (tx if a else 1.0 - tx) * (ty if b else 1.0 - ty) * (tz if e else 1.0 - tz)
        if w <= 0.0:
            continue
        r = _rid_node(index_flat, rid, oi_tab, oi, i0 + a, j0 + b, k0 + e)
        if r < hid.shape[1] and hid[oi, r] != 0:
            acc += w
    return acc


@njit(cache=True)
def hidden_frac_b3(index_flat, rid, oi_tab, oi, h, x, y, z, hid):
    """hidden_frac with a cubic B-spline over the 4 x 4 x 4 nodes around the point: the edge of a region seen on the
    surface (the hit test) is the 0.5 contour of a smooth (C2) blend of the nodes' hidden flags, so it no longer traces
    the lattice node by node (the edge read as small steps with only the region shown)"""
    fx = x / h
    fy = y / h
    fz = z / h
    i0 = int(math.floor(fx))
    j0 = int(math.floor(fy))
    k0 = int(math.floor(fz))
    wx = np.empty(4)
    wy = np.empty(4)
    wz = np.empty(4)
    for a in range(3):
        t = fx - i0 if a == 0 else (fy - j0 if a == 1 else fz - k0)
        t2 = t * t
        t3 = t2 * t
        w0 = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
        w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
        w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
        w3 = t3 / 6.0
        w = wx if a == 0 else (wy if a == 1 else wz)
        w[0] = w0
        w[1] = w1
        w[2] = w2
        w[3] = w3
    acc = 0.0
    for a in range(4):
        for b in range(4):
            wab = wx[a] * wy[b]
            for c in range(4):
                r = _rid_node(index_flat, rid, oi_tab, oi, i0 - 1 + a, j0 - 1 + b, k0 - 1 + c)
                if r < hid.shape[1] and hid[oi, r] != 0:
                    acc += wab * wz[c]
    return acc


@njit(cache=True, inline='always')
def _slab(b, ox, oy, oz, dx, dy, dz):
    """the ray's interval inside clip volume b: a convex set of 6 planes (n x, y, z, d; inside where n.p <= d; an unused
    plane n = 0, d = 1) -> (t in, t out, the plane it leaves through)"""
    ta = -1e30
    tb = 1e30
    ax = 0
    for a in range(6):
        nx = b[4 * a]
        ny = b[4 * a + 1]
        nz = b[4 * a + 2]
        s0 = nx * ox + ny * oy + nz * oz - b[4 * a + 3]
        dn = nx * dx + ny * dy + nz * dz
        if abs(dn) < 1e-30:
            if s0 > 0.0:
                return 1e30, -1e30, 0
        else:
            u = -s0 / dn
            if dn < 0.0:
                if u > ta:
                    ta = u
            elif u < tb:
                tb = u
                ax = a
    return ta, tb, ax


@njit(cache=True)
def _hidden_at(code, prm, oi_tab, of_tab, index_flat, delta, rid, hid, hasr, x, y, z, vs, ps):
    """-> (f, object, hidden): the scene's exact field at a world point and whether that point is absent (inside a
    hidden region's volume)"""
    fq = eval_scene_f(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, vs, ps, 1e9)
    f_, d_, oq = eval_scene(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, vs, ps)
    if fq >= 0.0 or oq < 0 or hasr[oq] == 0:
        return fq, oq, False
    Hl = of_tab[oq, P.OF_H] * of_tab[oq, P.OF_S]
    nx, ny, nz = normal_at(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, 0.5 * Hl, vs, ps)
    lx, ly, lz = to_local(of_tab, oq, x - fq * nx, y - fq * ny, z - fq * nz)
    return fq, oq, hidden_frac(index_flat, rid, oi_tab, oq, of_tab[oq, P.OF_H], lx, ly, lz, hid) >= 0.5


@njit(cache=True)
def _walk(code, prm, oi_tab, of_tab, index_flat, delta, rid, hid, hasr, o, ox, oy, oz, dx, dy, dz, t, t1, vs, ps):
    """through a hidden region's volume from t in lattice steps -> (t, object, f, status): status 1 the cut face (solid
    of a shown region met), 0 the ray left the forms (resume tracing at t)"""
    step = 0.75 * of_tab[o, P.OF_H] * of_tab[o, P.OF_S]
    if step <= 0.0:
        step = 1e-4
    prev = t
    for _k in range(20000):
        tt = prev + step
        if tt >= t1:
            return t1, -1, 0.0, 0
        fq, oq, hdn = _hidden_at(code, prm, oi_tab, of_tab, index_flat, delta, rid, hid, hasr,
                                 ox + dx * tt, oy + dy * tt, oz + dz * tt, vs, ps)
        if fq >= 0.0:
            return tt, -1, 0.0, 0
        if not hdn:
            a = prev
            b = tt
            fb = fq
            ob = oq
            for _i in range(8):
                m = 0.5 * (a + b)
                fm, om, hm = _hidden_at(code, prm, oi_tab, of_tab, index_flat, delta, rid, hid, hasr,
                                        ox + dx * m, oy + dy * m, oz + dz * m, vs, ps)
                if fm < 0.0 and not hm:
                    b = m
                    fb = fm
                    ob = om
                else:
                    a = m
            return b, ob, fb, 1
        prev = tt
    return prev, -1, 0.0, 0


@njit(cache=True)
def trace_clipped(code, prm, oi_tab, of_tab, index_flat, delta, Lb, lbase, rid, boxes, hid, hasr,
                  ox, oy, oz, dx, dy, dz, t0, t1, eps_min, eps_k, max_steps, vs, ps, tin, tout):
    """trace() with the view-clip list -> (t, object, steps, status, cut, cut normal x, y, z, cut f): status 1 a hit; cut
    True: the hit is a cut face (where a box or a hidden region opens a form), with its normal and the field there"""
    cur = t0
    ns_all = 0
    nb = boxes.shape[0]
    for _it in range(64):
        if cur >= t1:
            return np.inf, -1, ns_all, 0, False, 0.0, 0.0, 0.0, 0.0
        jumped = False
        cnx = 0.0
        cny = 0.0
        cnz = 0.0
        for _r in range(nb + 1):
            moved = False
            for b in range(nb):
                ta, tb, ax = _slab(boxes[b], ox, oy, oz, dx, dy, dz)
                if ta <= cur and cur < tb:
                    cur = tb
                    moved = True
                    jumped = True
                    cnx = -boxes[b, 4 * ax]               # the cut face faces back into the hidden volume
                    cny = -boxes[b, 4 * ax + 1]
                    cnz = -boxes[b, 4 * ax + 2]
            if not moved:
                break
        if cur >= t1:
            return np.inf, -1, ns_all, 0, False, 0.0, 0.0, 0.0, 0.0
        if jumped:
            qx = ox + dx * cur
            qy = oy + dy * cur
            qz = oz + dz * cur
            fq, oq, hdn = _hidden_at(code, prm, oi_tab, of_tab, index_flat, delta, rid, hid, hasr, qx, qy, qz, vs, ps)
            if fq < 0.0 and oq >= 0 and not hdn:
                cl = math.sqrt(cnx * cnx + cny * cny + cnz * cnz)
                if cl > 0.0:
                    cnx /= cl
                    cny /= cl
                    cnz /= cl
                return cur, oq, ns_all, 1, True, cnx, cny, cnz, fq
            if fq < 0.0:
                tw, ow, fw, st_ = _walk(code, prm, oi_tab, of_tab, index_flat, delta, rid, hid, hasr, oq,
                                        ox, oy, oz, dx, dy, dz, cur, t1, vs, ps)
                if st_ == 1:
                    return tw, ow, ns_all, 1, True, -dx, -dy, -dz, fw
                cur = tw
                continue
        tn = t1
        for b in range(nb):
            ta, tb, ax = _slab(boxes[b], ox, oy, oz, dx, dy, dz)
            if ta > cur and ta < tn and tb > ta:
                tn = ta
        t, o, ns, st = trace(code, prm, oi_tab, of_tab, index_flat, delta, Lb, lbase, ox, oy, oz, dx, dy, dz,
                             cur, tn, eps_min, eps_k, max_steps, vs, ps, tin, tout)
        ns_all += ns
        if st == 1:
            if hasr[o] != 0:
                lx, ly, lz = to_local(of_tab, o, ox + dx * t, oy + dy * t, oz + dz * t)
                if hidden_frac_b3(index_flat, rid, oi_tab, o, of_tab[o, P.OF_H], lx, ly, lz, hid) >= 0.5:
                    tw, ow, fw, st_ = _walk(code, prm, oi_tab, of_tab, index_flat, delta, rid, hid, hasr, o,
                                            ox, oy, oz, dx, dy, dz, t, t1, vs, ps)
                    if st_ == 1:
                        return tw, ow, ns_all, 1, True, -dx, -dy, -dz, fw
                    cur = tw
                    continue
            return t, o, ns_all, 1, False, 0.0, 0.0, 0.0, 0.0
        if st == 2:
            return t, o, ns_all, 2, False, 0.0, 0.0, 0.0, 0.0
        if tn >= t1:
            return np.inf, -1, ns_all, 0, False, 0.0, 0.0, 0.0, 0.0
        cur = tn
    return np.inf, -1, ns_all, 0, False, 0.0, 0.0, 0.0, 0.0


@njit(parallel=True, cache=True, nogil=True)
def render_kernel(code, prm, oi_tab, of_tab, index_flat, delta, mask, Lb, lbase, cam, lo, hi, w, h, mode, ao_n,
                  ao_dist,
                  eps_min, eps_k, max_steps, show_mask, slice_band, rgb, depth, nrm, obj, row_steps, row_unres,
                  rid, boxes, hid, hasr, nclip, rcol):
    cpx, cpy, cpz = cam[0], cam[1], cam[2]
    fx, fy, fz = cam[3], cam[4], cam[5]
    rx, ry, rz = cam[6], cam[7], cam[8]
    ux, uy, uz = cam[9], cam[10], cam[11]
    th = cam[12]
    tgx, tgy, tgz = cam[13], cam[14], cam[15]
    # fs-shell (11-11): the orthographic view: parallel rays along f, each starting on the plane B before the camera,
    # offset across it by the pixel's place on the target plane (D = the camera's distance to its target)
    orth = cam.shape[0] > 18 and cam[16] > 0.5
    oD = cam[17] if orth else 0.0
    oB = cam[18] if orth else 0.0
    # fs-shell 11-11 (design 4.4): the section plane (n, d): the half n.p > d is cut from the view. A ray that starts
    # on that side starts at the plane, and where the plane point is inside a form the pixel is the cut face (flat,
    # facing the cut side, with thickness bands every slice_band); a ray from the kept side stops at the plane
    sec = cam.shape[0] > 23 and cam[23] > 0.5
    snx = cam[19] if sec else 0.0
    sny = cam[20] if sec else 0.0
    snz = cam[21] if sec else 0.0
    sdd = cam[22] if sec else 0.0
    aspect = w / h
    # camera-relative lights (unit vectors towards the light)
    k1x = -0.45 * rx + 0.65 * ux - 0.6 * fx
    k1y = -0.45 * ry + 0.65 * uy - 0.6 * fy
    k1z = -0.45 * rz + 0.65 * uz - 0.6 * fz
    l = math.sqrt(k1x * k1x + k1y * k1y + k1z * k1z)
    k1x /= l
    k1y /= l
    k1z /= l
    k2x = 0.8 * rx - 0.1 * ux - 0.5 * fx
    k2y = 0.8 * ry - 0.1 * uy - 0.5 * fy
    k2z = 0.8 * rz - 0.1 * uz - 0.5 * fz
    l = math.sqrt(k2x * k2x + k2y * k2y + k2z * k2z)
    k2x /= l
    k2y /= l
    k2z /= l
    nobj = oi_tab.shape[0]
    for j in prange(h):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        tin = np.empty(nobj)
        tout = np.empty(nobj)
        steps_acc = 0
        unres = 0
        for i in range(w):
            px = (2.0 * (i + 0.5) / w - 1.0) * th * aspect
            py = (1.0 - 2.0 * (j + 0.5) / h) * th
            if orth:
                ox = cpx + (rx * px + ux * py) * oD - fx * oB
                oy = cpy + (ry * px + uy * py) * oD - fy * oB
                oz = cpz + (rz * px + uz * py) * oD - fz * oB
                dx, dy, dz = fx, fy, fz
            else:
                ox, oy, oz = cpx, cpy, cpz
                dx = fx + rx * px + ux * py
                dy = fy + ry * px + uy * py
                dz = fz + rz * px + uz * py
                dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= dl
                dy /= dl
                dz /= dl
            bg = 0.085 + 0.05 * (1.0 - (j + 0.5) / h)
            cr = bg
            cg = bg
            cb = bg * 1.08
            depth[j, i] = np.inf
            nrm[j, i, 0] = 0.0
            nrm[j, i, 1] = 0.0
            nrm[j, i, 2] = 0.0
            obj[j, i] = -1
            if mode == 3:
                den = dx * fx + dy * fy + dz * fz
                tp = ((tgx - ox) * fx + (tgy - oy) * fy + (tgz - oz) * fz) / den
                qx = ox + dx * tp
                qy = oy + dy * tp
                qz = oz + dz * tp
                f, d, o = eval_scene(code, prm, oi_tab, of_tab, index_flat, delta, qx, qy, qz, vs, ps)
                pix = tp * eps_k * 2.0
                band = f / slice_band
                fr = band - math.floor(band)
                line = 1.0 if min(fr, 1.0 - fr) < 0.04 else 0.0
                if f < 0.0:
                    cr, cg, cb = 0.12, 0.25, 0.48
                else:
                    cr, cg, cb = 0.45, 0.28, 0.12
                fade = 1.0 / (1.0 + abs(f) / (8.0 * slice_band))
                cr = cr * (0.35 + 0.65 * fade) + 0.12 * line
                cg = cg * (0.35 + 0.65 * fade) + 0.12 * line
                cb = cb * (0.35 + 0.65 * fade) + 0.12 * line
                if abs(f) < 1.5 * pix:
                    cr, cg, cb = 1.0, 1.0, 1.0
                depth[j, i] = tp
                obj[j, i] = o if f < 0.0 else -1
            else:
                t0, t1 = _clip(ox, oy, oz, dx, dy, dz, lo, hi)
                if sec:
                    s0 = snx * ox + sny * oy + snz * oz - sdd
                    dn = snx * dx + sny * dy + snz * dz
                    if s0 > 0.0:
                        if dn >= 0.0:
                            t1 = -1.0                       # the ray never reaches the kept side
                        else:
                            tp = -s0 / dn
                            qx = ox + dx * tp
                            qy = oy + dy * tp
                            qz = oz + dz * tp
                            fq0, dq, oq = eval_scene(code, prm, oi_tab, of_tab, index_flat, delta, qx, qy, qz, vs, ps)
                            # inside / outside through the exact reader (no guard bounds: the store's own coarser-level
                            # fallback, fs-pose's one inside rule), the same value the Section readout reads
                            fq = eval_scene_f(code, prm, oi_tab, of_tab, index_flat, delta, qx, qy, qz, vs, ps, 1e9)
                            if fq < 0.0 and oq >= 0:
                                # the cut face, filled from the field: the object's colour, flat lit, with a band
                                # line every slice_band of depth inside (the wall's thickness reads at a glance)
                                ar = of_tab[oq, P.OF_ALB]
                                ag = of_tab[oq, P.OF_ALB + 1]
                                ab = of_tab[oq, P.OF_ALB + 2]
                                bnd = -fq / slice_band
                                frb = bnd - math.floor(bnd)
                                line = 0.18 if min(frb, 1.0 - frb) < 0.05 else 0.0
                                pix = tp * eps_k * 2.0
                                cr = ar * 0.62 - line
                                cg = ag * 0.5 - line
                                cb = ab * 0.48 - line
                                if -fq < 1.5 * pix:
                                    cr, cg, cb = 0.95, 0.85, 0.55
                                depth[j, i] = tp
                                nrm[j, i, 0] = snx
                                nrm[j, i, 1] = sny
                                nrm[j, i, 2] = snz
                                obj[j, i] = oq
                                t1 = -1.0                   # no trace: the pixel is the cut face
                            elif tp > t0:
                                t0 = tp
                    elif dn > 0.0:
                        tp = -s0 / dn
                        if tp < t1:
                            t1 = tp
                if t0 < t1:
                    cut = False
                    cnx = 0.0
                    cny = 0.0
                    cnz = 0.0
                    cfq = 0.0
                    if nclip == 0:
                        t, o, ns, status = trace(code, prm, oi_tab, of_tab, index_flat, delta, Lb, lbase,
                                                 ox, oy, oz, dx, dy, dz,
                                                 t0, t1, eps_min, eps_k, max_steps, vs, ps, tin, tout)
                    else:
                        t, o, ns, status, cut, cnx, cny, cnz, cfq = trace_clipped(
                            code, prm, oi_tab, of_tab, index_flat, delta, Lb, lbase, rid, boxes, hid, hasr,
                            ox, oy, oz, dx, dy, dz, t0, t1, eps_min, eps_k, max_steps, vs, ps, tin, tout)
                    steps_acc += ns
                    if status == 2:
                        unres += 1
                    if status == 1 and cut:
                        # fs-shell 11-11: a cut face where a box or a hidden region opens a form (as the section's)
                        ar = of_tab[o, P.OF_ALB]
                        ag = of_tab[o, P.OF_ALB + 1]
                        ab = of_tab[o, P.OF_ALB + 2]
                        bnd = -cfq / slice_band
                        frb = bnd - math.floor(bnd)
                        line = 0.18 if min(frb, 1.0 - frb) < 0.05 else 0.0
                        cr = ar * 0.62 - line
                        cg = ag * 0.5 - line
                        cb = ab * 0.48 - line
                        depth[j, i] = t
                        nrm[j, i, 0] = cnx
                        nrm[j, i, 1] = cny
                        nrm[j, i, 2] = cnz
                        obj[j, i] = o
                    elif status == 1:
                        hx = ox + dx * t
                        hy = oy + dy * t
                        hz = oz + dz * t
                        e = max(eps_min, t * eps_k) * 0.5
                        Hl = of_tab[o, P.OF_H] * of_tab[o, P.OF_S]
                        if Hl > 0.0 and e < Hl:
                            # 11-10: every lattice object (her store with or without sculpt bricks, a sculpt layer):
                            # the cell's node gradients blended (normal_lattice); hand2 widened only sculpted objects
                            nx, ny, nz = normal_lattice(code, prm, oi_tab, of_tab, index_flat, delta, hx, hy, hz, Hl, vs, ps)
                        else:
                            nx, ny, nz = normal_at(code, prm, oi_tab, of_tab, index_flat, delta, hx, hy, hz, e, vs, ps)
                        depth[j, i] = t
                        nrm[j, i, 0] = nx
                        nrm[j, i, 1] = ny
                        nrm[j, i, 2] = nz
                        obj[j, i] = o
                        ar = of_tab[o, P.OF_ALB]
                        ag = of_tab[o, P.OF_ALB + 1]
                        ab = of_tab[o, P.OF_ALB + 2]
                        if rcol.shape[1] > 1 and rcol[o, 0, 3] > 0.0:
                            # fs-shell 11-11: the region overlay (the Regions panel's colours over the surface)
                            lx, ly, lz = to_local(of_tab, o, hx, hy, hz)
                            rr = rid_at(index_flat, rid, oi_tab, o, of_tab[o, P.OF_H], lx, ly, lz)
                            if rr > 0 and rr < rcol.shape[1] and rcol[o, rr, 3] > 0.0:
                                wa = rcol[o, rr, 3]
                                ar = ar * (1.0 - wa) + rcol[o, rr, 0] * wa
                                ag = ag * (1.0 - wa) + rcol[o, rr, 1] * wa
                                ab = ab * (1.0 - wa) + rcol[o, rr, 2] * wa
                        if show_mask:
                            lx, ly, lz = to_local(of_tab, o, hx, hy, hz)
                            mk = grid_sample(index_flat, mask, oi_tab, o, of_tab[o, P.OF_H], lx, ly, lz)
                            mk = min(max(mk, 0.0), 1.0) * 0.7
                            ar = ar * (1.0 - mk) + 0.15 * mk
                            ag = ag * (1.0 - mk) + 0.15 * mk
                            ab = ab * (1.0 - mk) + 0.18 * mk
                        vx = -dx
                        vy = -dy
                        vz = -dz
                        ndv = max(nx * vx + ny * vy + nz * vz, 0.0)
                        if mode == 2:
                            cr = nx * 0.5 + 0.5
                            cg = ny * 0.5 + 0.5
                            cb = nz * 0.5 + 0.5
                        elif mode == 1:
                            vnx = nx * rx + ny * ry + nz * rz
                            vny = nx * ux + ny * uy + nz * uz
                            vnz = ndv
                            dif = 0.5 + 0.5 * (-0.25 * vnx + 0.55 * vny + 0.8 * vnz)
                            hl = max(-0.3 * vnx + 0.45 * vny + 0.84 * vnz, 0.0)
                            sp = hl ** 40 * 0.35
                            rim = (1.0 - ndv) ** 3 * 0.25
                            cr = ar * dif * 0.95 + sp + rim
                            cg = ag * dif * 0.95 + sp + rim
                            cb = ab * dif * 0.95 + sp + rim * 1.1
                        else:
                            ao = 1.0
                            if ao_n > 0:
                                occ = 0.0
                                sca = 1.0
                                for k in range(1, ao_n + 1):
                                    # AO reads min(F, hr): evaluate with the guard threshold hr so no guard box
                                    # lower bound (a downward F jump) is read in place of the real field
                                    hr = ao_dist * k / ao_n
                                    ff = eval_scene_f(code, prm, oi_tab, of_tab, index_flat, delta,
                                                      hx + nx * hr, hy + ny * hr, hz + nz * hr, vs, ps, hr)
                                    dd = min(max(ff, 0.0), hr)
                                    occ += (hr - dd) * sca
                                    sca *= 0.8
                                ao = min(max(1.0 - 1.6 * occ / (ao_dist * ao_n * 0.5), 0.0), 1.0)
                            d1 = max(nx * k1x + ny * k1y + nz * k1z, 0.0)
                            d2 = nx * k2x + ny * k2y + nz * k2z
                            d2 = max(0.5 * d2 + 0.5, 0.0) ** 2
                            hx1 = k1x + vx
                            hy1 = k1y + vy
                            hz1 = k1z + vz
                            hl = math.sqrt(hx1 * hx1 + hy1 * hy1 + hz1 * hz1)
                            nh = max((nx * hx1 + ny * hy1 + nz * hz1) / hl, 0.0)
                            rough = of_tab[o, P.OF_ROUGH]
                            shin = 8.0 + 120.0 * (1.0 - rough) * (1.0 - rough)
                            sp = (nh ** shin) * 0.25 * (1.0 - rough * 0.7) if d1 > 0.0 else 0.0
                            rim = (1.0 - ndv) ** 3 * 0.35
                            amb = 0.16 + 0.08 * ny
                            lum_r = 1.05 * d1 + 0.28 * d2 + amb * ao
                            cr = ar * lum_r * (0.55 + 0.45 * ao) + sp + rim * ao * 0.8
                            cg = ag * lum_r * (0.55 + 0.45 * ao) + sp + rim * ao * 0.85
                            cb = ab * lum_r * (0.55 + 0.45 * ao) + sp + rim * ao
            rgb[j, i, 0] = np.uint8(min(max(cr, 0.0), 1.0) ** (1.0 / 2.2) * 255.0 + 0.5)
            rgb[j, i, 1] = np.uint8(min(max(cg, 0.0), 1.0) ** (1.0 / 2.2) * 255.0 + 0.5)
            rgb[j, i, 2] = np.uint8(min(max(cb, 0.0), 1.0) ** (1.0 / 2.2) * 255.0 + 0.5)
        row_steps[j] = steps_acc
        row_unres[j] = unres


@njit(parallel=True, cache=True, nogil=True)
def trace_many(code, prm, oi_tab, of_tab, index_flat, delta, Lb, lbase, orig, dirs, lo, hi, eps_min, eps_k, max_steps,
               out_t, out_o, out_status):
    for n in prange(orig.shape[0]):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        tin = np.empty(oi_tab.shape[0])
        tout = np.empty(oi_tab.shape[0])
        t0, t1 = _clip(orig[n, 0], orig[n, 1], orig[n, 2], dirs[n, 0], dirs[n, 1], dirs[n, 2], lo, hi)
        out_t[n] = np.inf
        out_o[n] = -1
        out_status[n] = 0
        if t0 < t1:
            t, o, ns, st = trace(code, prm, oi_tab, of_tab, index_flat, delta, Lb, lbase,
                                 orig[n, 0], orig[n, 1], orig[n, 2],
                                 dirs[n, 0], dirs[n, 1], dirs[n, 2], t0, t1, eps_min, eps_k, max_steps, vs, ps,
                                 tin, tout)
            out_status[n] = st
            if st == 1:
                out_t[n] = t
                out_o[n] = o


@njit(parallel=True, cache=True, nogil=True)
def reference_crossings(code, prm, oi_tab, of_tab, index_flat, delta, orig, dirs, lo, hi, step, out_t):
    """Dense reference: march every ray at a fixed small step; first t where F <= 0 (inf if none)."""
    for n in prange(orig.shape[0]):
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        out_t[n] = np.inf
        t0, t1 = _clip(orig[n, 0], orig[n, 1], orig[n, 2], dirs[n, 0], dirs[n, 1], dirs[n, 2], lo, hi)
        if t0 >= t1:
            continue
        t = t0
        while t <= t1:
            f, d, o = eval_scene(code, prm, oi_tab, of_tab, index_flat, delta, orig[n, 0] + dirs[n, 0] * t,
                                 orig[n, 1] + dirs[n, 1] * t, orig[n, 2] + dirs[n, 2] * t, vs, ps)
            if f <= 0.0:
                out_t[n] = t
                break
            t += step


QUALITY = {0: dict(scale=2, ao=0, eps=0.75, steps=160), 1: dict(scale=1, ao=3, eps=0.5, steps=256),
           2: dict(scale=1, ao=6, eps=0.35, steps=512)}


def _cam_array(camera):
    """[pos 0-2, f 3-5, r 6-8, u 9-11, tan(fov/2) 12, target 13-15, ortho flag 16, D 17, B 18] (fs-shell 11-11: the
    orthographic triple; see Camera.ray)"""
    f, r, u = camera.basis()
    th = math.tan(math.radians(camera.fov) * 0.5)
    ov = camera.ortho_vec() if hasattr(camera, 'ortho_vec') else np.zeros(3)
    sv = camera.section_vec() if getattr(camera, 'section', None) is not None else np.zeros(5)
    return np.array(list(camera.pos) + list(f) + list(r) + list(u) + [th] + list(camera.target) + list(ov) + list(sv),
                    np.float64)


def pixel_angle(camera, h):
    """the angle a pixel spans; orthographic: the pixel's width at the target over the ray's length there, so
    t * pixel_angle is the pixel's footprint near the target in both views"""
    a = 2.0 * math.tan(math.radians(camera.fov) * 0.5) / h
    if getattr(camera, 'ortho', False):
        D, B = camera.ortho_params()
        a *= D / max(B + D, 1e-9)
    return a


def clip_planes(v):
    """a hidden volume -> 24 floats (6 planes n, d; inside where n.p <= d): v = (lo, hi) a world box, or
    {'planes': [[nx, ny, nz, d], ...]} (up to 6: a screen box's frustum)"""
    if isinstance(v, dict):
        pl = [list(map(float, q)) for q in v['planes']][:6]
    else:
        lo, hi = (np.asarray(a, float) for a in v)
        pl = [[1.0, 0.0, 0.0, hi[0]], [-1.0, 0.0, 0.0, -lo[0]], [0.0, 1.0, 0.0, hi[1]], [0.0, -1.0, 0.0, -lo[1]],
              [0.0, 0.0, 1.0, hi[2]], [0.0, 0.0, -1.0, -lo[2]]]
    while len(pl) < 6:
        pl.append([0.0, 0.0, 0.0, 1.0])
    return [x for q in pl for x in q]


def view_clips(scene):
    """fs-shell 11-11: the view-clip list from the scene's view state (not in the hash or the log):
    scene.view_boxes [(lo, hi) world], scene.view_regions_hidden {oid: set(region ids; 0 = the unlabelled surface)},
    scene.view_region_overlay (bool: the Regions panel's colours over the surface) ->
    {'boxes' (k, 24) planes, 'hid' (nobj, K) uint8, 'hasr' (nobj,) uint8, 'n' (0: nothing clipped: trace() as before),
     'rcol' (nobj, K, 4) float (alpha 0: off; (nobj, 1, 4) when off)}"""
    objs = scene.objects
    nobj = len(objs)
    boxes = np.array([clip_planes(v) for v in (getattr(scene, 'view_boxes', None) or [])],
                     np.float64).reshape(-1, 24)
    vh = getattr(scene, 'view_regions_hidden', None) or {}
    K = 1
    for o in objs:
        ids = vh.get(o.id)
        if ids:
            K = max(K, max(ids) + 1)
    hid = np.zeros((nobj, K), np.uint8)
    hasr = np.zeros(nobj, np.uint8)
    for i, o in enumerate(objs):
        ids = vh.get(o.id)
        if ids:
            for r in ids:
                hid[i, int(r)] = 1
            hasr[i] = 1
    n = len(boxes) + int(hasr.any())
    rcol = np.zeros((nobj, 1, 4), np.float64)
    if getattr(scene, 'view_region_overlay', False):
        Kc = max([max([int(k) for k in (getattr(o, 'regions', None) or {})] + [0]) for o in objs] + [0]) + 1
        if Kc > 1:
            rcol = np.zeros((nobj, Kc, 4), np.float64)
            for i, o in enumerate(objs):
                for k, v in (getattr(o, 'regions', None) or {}).items():
                    c = v.get('colour') or (0.8, 0.5, 0.5)
                    rcol[i, int(k)] = (float(c[0]), float(c[1]), float(c[2]), 0.55)
                if getattr(o, 'regions', None):
                    rcol[i, 0, 3] = 1.0                  # the object's flag: it has an overlay (id 0 itself is not tinted)
    return {'boxes': boxes, 'hid': hid, 'hasr': hasr, 'n': n, 'rcol': rcol}


def render(scene, camera, w, h, quality=1, mode='shaded', show_mask=False, slice_band=None, scale=None, upsample=True):
    """hand1: scale overrides the quality's internal down-scale (the app picks 2-4 for the preview by its last frame
    time); upsample=False returns the internal-size arrays (the viewport scales the image itself: 14-25 ms saved per
    preview frame at 1280x800-1920x1080, M lag diagnosis)."""
    q = dict(QUALITY[int(quality)])
    if scale:
        q['scale'] = max(1, int(scale))
    t_start = time.perf_counter()
    pk = scene.pack()
    rw, rh = max(1, w // q['scale']), max(1, h // q['scale'])
    rgb = np.empty((rh, rw, 3), np.uint8)
    depth = np.empty((rh, rw), np.float32)
    nrm = np.empty((rh, rw, 3), np.float32)
    obj = np.empty((rh, rw), np.int32)
    rs = np.zeros(rh, np.int64)
    ru = np.zeros(rh, np.int64)
    lo, hi = pk.lo, pk.hi
    diag = float(np.linalg.norm(hi - lo)) if len(pk.ids) else 1.0
    eps_k = pixel_angle(camera, rh) * q['eps']
    eps_min = 1e-6
    ao_dist = 0.04 * max(diag, 1e-3)
    band = slice_band or max(diag / 40.0, 1e-4)
    if len(pk.ids):
        vc = view_clips(scene)
        render_kernel(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, pk.mask, scene.store.Lb,
                      pk.lbase, _cam_array(camera),
                      lo, hi, rw, rh, MODES[mode], q['ao'], ao_dist, eps_min, eps_k, q['steps'], show_mask, band,
                      rgb, depth, nrm, obj, rs, ru, scene.store.kview('rid'), vc['boxes'], vc['hid'], vc['hasr'], vc['n'],
                      vc['rcol'])
    else:
        for j in range(rh):
            rgb[j] = int((0.085 + 0.05 * (1 - (j + 0.5) / rh)) ** (1 / 2.2) * 255)
        depth[:] = np.inf
        nrm[:] = 0
        obj[:] = -1
    ids = np.array(pk.ids + [-1], np.int32)
    obj = ids[obj]                       # object index -> object id (index -1 -> last entry = -1)
    if q['scale'] > 1 and upsample:
        s = q['scale']
        rgb = np.repeat(np.repeat(rgb, s, 0), s, 1)[:h, :w]
        depth = np.repeat(np.repeat(depth, s, 0), s, 1)[:h, :w]
        nrm = np.repeat(np.repeat(nrm, s, 0), s, 1)[:h, :w]
        obj = np.repeat(np.repeat(obj, s, 0), s, 1)[:h, :w]
        if rgb.shape[0] < h or rgb.shape[1] < w:
            pad = ((0, h - rgb.shape[0]), (0, w - rgb.shape[1]))
            rgb = np.pad(rgb, pad + ((0, 0),), mode='edge')
            depth = np.pad(depth, pad, mode='edge')
            nrm = np.pad(nrm, pad + ((0, 0),), mode='edge')
            obj = np.pad(obj, pad, mode='edge')
    ms = (time.perf_counter() - t_start) * 1000.0
    stats = {'ms': ms, 'steps': int(rs.sum()), 'unresolved': int(ru.sum()), 'internal_res': (rw, rh),
             'lipschitz': scene.lipschitz_bound(), 'quality': int(quality), 'mode': mode}
    return RenderResult(rgb, depth, nrm, obj, stats)


def rays_for(camera, w, h, stride=1):
    """All pixel rays (every stride-th pixel) -> origins (N,3), dirs (N,3), pixel coords (N,2)."""
    f, r, u = camera.basis()
    th = math.tan(math.radians(camera.fov) * 0.5)
    xs = np.arange(0, w, stride)
    ys = np.arange(0, h, stride)
    X, Y = np.meshgrid(xs, ys)
    px = (2.0 * (X + 0.5) / w - 1.0) * th * (w / h)
    py = (1.0 - 2.0 * (Y + 0.5) / h) * th
    if getattr(camera, 'ortho', False):
        D, B = camera.ortho_params()
        o = (camera.pos[None, None] + (r[None, None] * px[..., None] + u[None, None] * py[..., None]) * D
             - f[None, None] * B).reshape(-1, 3)
        d = np.repeat(f[None], len(o), 0)
        return np.ascontiguousarray(o), np.ascontiguousarray(d), np.stack([X.ravel(), Y.ravel()], 1)
    d = f[None, None] + r[None, None] * px[..., None] + u[None, None] * py[..., None]
    d = d.reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.repeat(camera.pos[None], len(d), 0)
    return np.ascontiguousarray(o), np.ascontiguousarray(d), np.stack([X.ravel(), Y.ravel()], 1)


def trace_rays(scene, orig, dirs, eps_k=1e-4, max_steps=512):
    pk = scene.pack()
    n = len(orig)
    t = np.empty(n)
    o = np.empty(n, np.int64)
    st = np.empty(n, np.int64)
    if not len(pk.ids):
        t[:] = np.inf
        o[:] = -1
        st[:] = 0
        return t, o, st
    trace_many(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, scene.store.Lb, pk.lbase,
               np.ascontiguousarray(orig, float),
               np.ascontiguousarray(dirs, float), pk.lo, pk.hi, 1e-6, eps_k, max_steps, t, o, st)
    return t, o, st


def pick(scene, camera, x, y, w, h):
    """Pixel (x, y) of a w x h view -> PickResult(hit, point, normal, object_id, t)."""
    org, d = camera.ray(x, y, w, h)
    eps_k = pixel_angle(camera, h) * 0.25
    t_far = float('inf')
    t_add = 0.0
    if getattr(camera, 'section', None) is not None:
        # fs-shell 11-11 (design 4.4): a pick sees what the sectioned view shows
        sv = camera.section_vec()
        sn, sd = sv[:3], sv[3]
        s0, dn = float(sn @ org) - sd, float(sn @ d)
        if s0 > 0.0:
            if dn >= 0.0:
                return PickResult(False, None, None, -1, float('inf'))
            tp = -s0 / dn
            q = org + d * tp
            pk = scene.pack()
            vs = np.empty(STACK)
            ps = np.empty(4 * STACK)
            _fq, _dq, oq = eval_scene(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, q[0], q[1], q[2],
                                      vs, ps)
            fq = eval_scene_f(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, q[0], q[1], q[2], vs, ps,
                              1e9)                          # the exact reader (the one inside rule)
            if fq < 0.0 and oq >= 0:
                return PickResult(True, q, np.array(sn), pk.ids[int(oq)], float(tp))
            org, t_add = q, tp
        elif dn > 0.0:
            t_far = -s0 / dn
    vc = view_clips(scene)
    if vc['n']:
        # fs-shell 11-11 (design 4.8): with hidden boxes or regions, the pick sees what the view shows (their cut faces)
        pk = scene.pack()
        miss = PickResult(False, None, None, -1, float('inf'))
        if not len(pk.ids):
            return miss
        vs = np.empty(STACK)
        ps = np.empty(4 * STACK)
        tin = np.empty(pk.oi_tab.shape[0])
        tout = np.empty(pk.oi_tab.shape[0])
        t0, t1 = _clip(org[0], org[1], org[2], d[0], d[1], d[2], pk.lo, pk.hi)
        t1 = min(t1, t_far - t_add)
        if not t0 < t1:
            return miss
        t, o, _ns, st, cut, cnx, cny, cnz, _cf = trace_clipped(
            pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, scene.store.Lb, pk.lbase, scene.store.kview('rid'),
            vc['boxes'], vc['hid'], vc['hasr'], org[0], org[1], org[2], d[0], d[1], d[2], t0, t1, 1e-6, eps_k, 512,
            vs, ps, tin, tout)
        if st != 1:
            return miss
        p = org + d * t
        if cut:
            n = np.array([cnx, cny, cnz])
        else:
            n = np.array(normal_at(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, p[0], p[1], p[2],
                                   max(1e-6, (t + t_add) * eps_k), vs, ps))
        return PickResult(True, p, n, pk.ids[int(o)], float(t + t_add))
    t, o, st = trace_rays(scene, org[None], d[None], eps_k=eps_k)
    if st[0] != 1 or t[0] + t_add > t_far:
        return PickResult(False, None, None, -1, float('inf'))
    p = org + d * t[0]
    t[0] += t_add
    pk = scene.pack()
    vs = np.empty(STACK)
    ps = np.empty(4 * STACK)
    e = max(1e-6, t[0] * eps_k)
    n = normal_at(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, p[0], p[1], p[2], e, vs, ps)
    return PickResult(True, p, np.array(n), pk.ids[int(o[0])], float(t[0]))


def reference_hits(scene, orig, dirs, step):
    pk = scene.pack()
    out = np.empty(len(orig))
    reference_crossings(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta,
                        np.ascontiguousarray(orig, float), np.ascontiguousarray(dirs, float), pk.lo, pk.hi,
                        float(step), out)
    return out
