"""Field program: the numba stack machine that evaluates a compiled scene.

A base tree compiles to a flat postfix program (code rows [op, param_offset, jump]) plus a float64 parameter
array. Domain mutations push the current point, transform it, and pop after their subtree; value mutations
change the top of the value stack; booleans pop two values and push one. OP_GUARD skips a subtree whose
(expanded) bounding box does not contain the point and pushes a conservative distance instead.

The scene pack (see scene.Scene.pack) adds per-object tables:
  oi_tab int32 (nobj, OI_N): code start/end, sculpt index offset, index dims, index origin (brick coords)
  of_tab float64 (nobj, OF_N): inverse rotation, translation, uniform scale, local box, margin, L, lattice h,
                               albedo, visibility, roughness
No polygon mesh anywhere: every value is a field sample.
"""
import math

import numpy as np
from numba import njit

from ..materials.fieldops import matgen as _matgen   # fs-materials 11-14 (OP_MATGEN)

# ---- opcodes -------------------------------------------------------------------------------------------
OP_SPHERE, OP_BOX, OP_CYL, OP_CAPSULE, OP_CONE, OP_TORUS, OP_ELLIPSOID, OP_SLAB = 1, 2, 3, 4, 5, 6, 7, 8
OP_SAMPLED = 9      # V1.2 field region: a dense sampled lattice (trilinear) intersected with its crop box / sphere
OP_UNION, OP_SUB, OP_INTER, OP_SUNION, OP_SSUB, OP_SINTER = 20, 21, 22, 23, 24, 25
OP_XFORM, OP_TWIST, OP_BEND, OP_TAPER, OP_POP = 30, 31, 32, 33, 39
OP_SHELL, OP_OFFSET = 40, 41
OP_GUARD = 50
OP_GUARD_U = 51     # hand2b: a union operand's guard that skips only where the skip is EXACT (see eval_prog_t)
OP_OSWEEP = 52      # cutsdf (11-08): a closed screen outline swept along the view rays from the eye (see _osweep)
OP_FV8 = 53         # milestone (11-08): her FV8 brick store read natively, the finest level present (core/fv8base.py)
OP_MATGEN = 60      # fs-materials 11-14: a material generator (relief, chips, crack) on the value below, as structure
                    # in the field (materials/fieldops.py: matgen; gpu/program.cu: the card's copy)

STACK = 64          # max value / point stack depth (tree depth)
BRICK = 8           # sculpt brick edge in lattice nodes
BSH = 3             # log2(BRICK)
BMASK = 7

# oi_tab columns
OI_C0, OI_C1, OI_IDX, OI_NX, OI_NY, OI_NZ, OI_OX, OI_OY, OI_OZ = range(9)
OI_N = 9
# of_tab columns
OF_R = 0            # 9: inverse rotation (R^T), row-major
OF_T = 9            # 3: translation
OF_S = 12           # uniform scale
OF_BMIN = 13        # 3: local box min (base + sculpt region)
OF_BMAX = 16        # 3: local box max
OF_MARGIN = 19      # guard margin
OF_L = 20           # Lipschitz bound of the object's local field (base + sculpt)
OF_H = 21           # sculpt lattice spacing (m)
OF_ALB = 22         # 3: albedo
OF_VIS = 25         # 1 visible, 0 hidden
OF_ROUGH = 26
OF_MINV = 27         # hand2f: 1 = the object's mask is shown / applied inverted (Mask > Invert)
OF_N = 28


@njit(cache=True, inline='always')
def _smin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


from numba import types as _nbt, carray as _carray
from numba.extending import intrinsic as _intrinsic


@_intrinsic
def _f32_ptr(typingctx, addr):
    """r-ptr: an int64 address -> float32* (her store's shared data, core/fv8base._shared)"""
    sig = _nbt.CPointer(_nbt.float32)(_nbt.int64)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(_nbt.CPointer(_nbt.float32)))
    return sig, codegen


@njit(cache=True)
def _fv8_dat(prm, o):
    return _carray(_f32_ptr(np.int64(prm[o + 46])), (np.int64(prm[o + 47]),))


@njit(cache=True)
def _fv8_bid(prm, o, dat, l, px, py, pz):
    """level l's brick holding the point: (brick id or -1 (the tile holds none there) or -2 (no tile slot), bx, by, bz)"""
    tm = prm[o + 3]
    nx = int(prm[o + 4])
    ny = int(prm[o + 5])
    nz = int(prm[o + 6])
    tx = int(math.floor((px - prm[o]) / tm))
    ty = int(math.floor((py - prm[o + 1]) / tm))
    tz = int(math.floor((pz - prm[o + 2]) / tm))
    if tx < 0 or ty < 0 or tz < 0 or tx >= nx or ty >= ny or tz >= nz:
        return -2, 0, 0, 0
    nb = int(prm[o + 16 + l])
    if nb == 0:
        return -2, 0, 0, 0
    goff = int(prm[o + 24]) * 1048576 + int(prm[o + 25])
    slot = int(dat[goff + ((tx * ny + ty) * nz + tz) * 4 + l])
    if slot < 0:
        return -2, 0, 0, 0
    h = prm[o + 8 + l]
    bx = int(math.floor((px - prm[o]) / h * 0.125))
    by = int(math.floor((py - prm[o + 1]) / h * 0.125))
    bz = int(math.floor((pz - prm[o + 2]) / h * 0.125))
    lx = bx - tx * nb
    ly = by - ty * nb
    lz = bz - tz * nb
    if lx < 0 or ly < 0 or lz < 0 or lx >= nb or ly >= nb or lz >= nb:
        return -2, 0, 0, 0
    toff = int(prm[o + 26 + 2 * l]) * 1048576 + int(prm[o + 27 + 2 * l])
    return int(dat[toff + slot * nb * nb * nb + (lx * nb + ly) * nb + lz]), bx, by, bz


@njit(cache=True)
def _fv8_raw(prm, o, dat, l, px, py, pz):
    """level l's stored trilinear value at the point (no cap rule), NaN where level l holds no brick"""
    bid, bx, by, bz = _fv8_bid(prm, o, dat, l, px, py, pz)
    if bid < 0:
        return np.nan
    h = prm[o + 8 + l]
    fx = (px - prm[o]) / h - bx * 8.0
    fy = (py - prm[o + 1]) / h - by * 8.0
    fz = (pz - prm[o + 2]) / h - bz * 8.0
    i = min(max(int(math.floor(fx)), 0), 7)
    j = min(max(int(math.floor(fy)), 0), 7)
    k = min(max(int(math.floor(fz)), 0), 7)
    ax = fx - i
    ay = fy - j
    az = fz - k
    s = int(prm[o + 34 + 2 * l]) * 1048576 + int(prm[o + 35 + 2 * l]) + bid * 729 + (i * 9 + j) * 9 + k
    a0 = float(dat[s]) + (float(dat[s + 81]) - float(dat[s])) * ax
    a1 = float(dat[s + 9]) + (float(dat[s + 90]) - float(dat[s + 9])) * ax
    a2 = float(dat[s + 1]) + (float(dat[s + 82]) - float(dat[s + 1])) * ax
    a3 = float(dat[s + 10]) + (float(dat[s + 91]) - float(dat[s + 10])) * ax
    b0 = a0 + (a1 - a0) * ay
    b1 = a2 + (a3 - a2) * ay
    return b0 + (b1 - b0) * az


@njit(cache=True)
def _fv8_coarser(prm, o, dat, l, x, y, z, v):
    """fv8_field.pre_estimate at one capped sample of level l: the finest coarser level's uncapped value there (the
    coarsest present level's as it is), else the stored v"""
    last = -1
    for k in range(4):
        if int(prm[o + 16 + k]) > 0:
            last = k
    for k in range(l + 1, 4):
        if int(prm[o + 16 + k]) == 0:
            continue
        w = _fv8_raw(prm, o, dat, k, x, y, z)
        if w != w:
            continue
        if k != last and abs(w) >= 0.999 * prm[o + 42 + k]:
            continue
        return w
    return v


@njit(cache=True)
def _fv8_raw_node(prm, o, dat, k, gx, gy, gz):
    """program.cu fv8_raw_node: level k's trilinear at level-k node coordinates (exact for nodes of finer levels),
    not a position rebuilt from the header (a node on a coarse brick face could land in the missing neighbour)"""
    nb = int(prm[o + 16 + k])
    if nb == 0:
        return np.nan
    nx = int(prm[o + 4])
    ny = int(prm[o + 5])
    nz = int(prm[o + 6])
    tw = 8.0 * nb
    tx = int(math.floor(gx / tw))
    ty = int(math.floor(gy / tw))
    tz = int(math.floor(gz / tw))
    if tx < 0 or ty < 0 or tz < 0 or tx >= nx or ty >= ny or tz >= nz:
        return np.nan
    slot = int(dat[int(prm[o + 24]) * 1048576 + int(prm[o + 25]) + ((tx * ny + ty) * nz + tz) * 4 + k])
    if slot < 0:
        return np.nan
    bx = int(math.floor(gx / 8.0))
    by = int(math.floor(gy / 8.0))
    bz = int(math.floor(gz / 8.0))
    bid = int(dat[int(prm[o + 26 + 2 * k]) * 1048576 + int(prm[o + 27 + 2 * k]) + slot * nb * nb * nb
                  + ((bx - tx * nb) * nb + (by - ty * nb)) * nb + (bz - tz * nb)])
    if bid < 0:
        return np.nan
    fx = gx - bx * 8.0
    fy = gy - by * 8.0
    fz = gz - bz * 8.0
    i = min(max(int(math.floor(fx)), 0), 7)
    j = min(max(int(math.floor(fy)), 0), 7)
    kk = min(max(int(math.floor(fz)), 0), 7)
    ax = fx - i
    ay = fy - j
    az = fz - kk
    s = int(prm[o + 34 + 2 * k]) * 1048576 + int(prm[o + 35 + 2 * k]) + bid * 729 + (i * 9 + j) * 9 + kk
    a0 = float(dat[s]) + (float(dat[s + 81]) - float(dat[s])) * ax
    a1 = float(dat[s + 9]) + (float(dat[s + 90]) - float(dat[s + 9])) * ax
    a2 = float(dat[s + 1]) + (float(dat[s + 82]) - float(dat[s + 1])) * ax
    a3 = float(dat[s + 10]) + (float(dat[s + 91]) - float(dat[s + 10])) * ax
    b0 = a0 + (a1 - a0) * ay
    b1 = a2 + (a3 - a2) * ay
    return b0 + (b1 - b0) * az


@njit(cache=True)
def _fv8_coarser_node(prm, o, dat, l, gx, gy, gz, v):
    """program.cu fv8_coarser_node: _fv8_coarser addressed by level-l node index"""
    last = -1
    for k in range(4):
        if int(prm[o + 16 + k]) > 0:
            last = k
    for k in range(l + 1, 4):
        if int(prm[o + 16 + k]) == 0:
            continue
        sc = 1.0 / (1 << (k - l))
        w = _fv8_raw_node(prm, o, dat, k, gx * sc, gy * sc, gz * sc)
        if w != w:
            continue
        if k != last and abs(w) >= 0.999 * prm[o + 42 + k]:
            continue
        return w
    return v


@njit(cache=True)
def _fv8_below_node(prm, o, dat, l, gx, gy, gz):
    """11-09: the value the coarser levels give AT level-l node g where level l holds no brick: the first coarser
    level's uncapped value, else the first capped one (as their own path keeps a capped sample when nothing coarser
    replaces it), else level l's bound (band - h)"""
    capped = np.nan
    for k in range(l + 1, 4):
        if int(prm[o + 16 + k]) == 0:
            continue
        sc = 1.0 / (1 << (k - l))
        w = _fv8_raw_node(prm, o, dat, k, gx * sc, gy * sc, gz * sc)
        if w != w:
            continue
        if abs(w) >= 0.999 * prm[o + 42 + k]:
            if capped != capped:
                capped = w
            continue
        return w
    if capped == capped:
        return capped
    return prm[o + 20 + l] - prm[o + 8 + l]


@njit(cache=True)
def _fv8_node(prm, o, dat, l, gx, gy, gz):
    """11-09 (seam): level l's value AT its lattice node (gx, gy, gz) from any level-l brick holding it, faces
    included (the brick below holds a face node as its index 8), with the cap rule; NaN when no level-l brick holds it"""
    nb = int(prm[o + 16 + l])
    if nb == 0:
        return np.nan
    nx = int(prm[o + 4])
    ny = int(prm[o + 5])
    nz = int(prm[o + 6])
    h = prm[o + 8 + l]
    goff = int(prm[o + 24]) * 1048576 + int(prm[o + 25])
    toff = int(prm[o + 26 + 2 * l]) * 1048576 + int(prm[o + 27 + 2 * l])
    soff = int(prm[o + 34 + 2 * l]) * 1048576 + int(prm[o + 35 + 2 * l])
    b0x = gx // 8
    b0y = gy // 8
    b0z = gz // 8
    for cx in range(2):
        if cx == 1 and gx % 8 != 0:
            continue
        for cy in range(2):
            if cy == 1 and gy % 8 != 0:
                continue
            for cz in range(2):
                if cz == 1 and gz % 8 != 0:
                    continue
                bx = b0x - cx
                by = b0y - cy
                bz = b0z - cz
                tx = bx // nb
                ty = by // nb
                tz = bz // nb
                if tx < 0 or ty < 0 or tz < 0 or tx >= nx or ty >= ny or tz >= nz:
                    continue
                slot = int(dat[goff + ((tx * ny + ty) * nz + tz) * 4 + l])
                if slot < 0:
                    continue
                bid = int(dat[toff + slot * nb * nb * nb + ((bx - tx * nb) * nb + (by - ty * nb)) * nb + (bz - tz * nb)])
                if bid < 0:
                    continue
                v = float(dat[soff + bid * 729 + ((gx - bx * 8) * 9 + (gy - by * 8)) * 9 + (gz - bz * 8)])
                if abs(v) >= 0.999 * prm[o + 42 + l]:
                    v = _fv8_coarser_node(prm, o, dat, l, gx, gy, gz, v)
                return v
    return np.nan


@njit(cache=True)
def _fv8_bound_node(prm, o, dat, l0, gx, gy, gz):
    """11-09: the store's lower bound at level-l0 node g (the old per-point rule, at the node): + lb, raised to band - h
    of every level whose tile there exists but holds no brick"""
    best = prm[o + 7]
    nx = int(prm[o + 4])
    ny = int(prm[o + 5])
    nz = int(prm[o + 6])
    for k in range(4):
        nb = int(prm[o + 16 + k])
        if nb == 0 or k < l0:
            continue
        sc = 1.0 / (1 << (k - l0))
        kx = gx * sc
        ky = gy * sc
        kz = gz * sc
        tw = 8.0 * nb
        tx = int(math.floor(kx / tw))
        ty = int(math.floor(ky / tw))
        tz = int(math.floor(kz / tw))
        if tx < 0 or ty < 0 or tz < 0 or tx >= nx or ty >= ny or tz >= nz:
            continue
        slot = int(dat[int(prm[o + 24]) * 1048576 + int(prm[o + 25]) + ((tx * ny + ty) * nz + tz) * 4 + k])
        if slot < 0:
            continue
        bx = int(math.floor(kx / 8.0))
        by = int(math.floor(ky / 8.0))
        bz = int(math.floor(kz / 8.0))
        bid = int(dat[int(prm[o + 26 + 2 * k]) * 1048576 + int(prm[o + 27 + 2 * k]) + slot * nb * nb * nb
                      + ((bx - tx * nb) * nb + (by - ty * nb)) * nb + (bz - tz * nb)])
        if bid < 0:
            best = max(best, prm[o + 20 + k] - prm[o + 8 + k])
    return best


@njit(cache=True)
def _fv8_V(prm, o, dat, l0, gx, gy, gz):
    """11-09: her field's ONE value at level-l0 node g: the level-l0 sample where a brick holds the node (faces
    included, cap rule), else the first coarser level's uncapped value there, else its capped one, else the bound"""
    v = _fv8_node(prm, o, dat, l0, gx, gy, gz)
    if v == v:
        return v
    capped = np.nan
    for k in range(l0 + 1, 4):
        if int(prm[o + 16 + k]) == 0:
            continue
        sc = 1.0 / (1 << (k - l0))
        w = _fv8_raw_node(prm, o, dat, k, gx * sc, gy * sc, gz * sc)
        if w != w:
            continue
        if abs(w) >= 0.999 * prm[o + 42 + k]:
            if capped != capped:
                capped = w
            continue
        return w
    if capped == capped:
        return capped
    return _fv8_bound_node(prm, o, dat, l0, gx, gy, gz)


@njit(cache=True)
def _fv8_seam(prm, o, dat, l, px, py, pz):
    """11-09 (a card / CPU desync): a point in a level-l cell that no level-l brick holds, next to one that
    does, read the coarser level while the brick beside it read its own samples: her field stepped 130-177 um across
    the face (3 mm out, at L0's band edge; near the surface once an edit moves it there), where the card, sampling
    nodes, is continuous. Here: trilinear over the cell's 8 nodes, each the level-l value where a brick holds it,
    else the coarser levels' value at the node. NaN when no corner is held (the coarser levels as before)."""
    h = prm[o + 8 + l]
    fx = (px - prm[o]) / h
    fy = (py - prm[o + 1]) / h
    fz = (pz - prm[o + 2]) / h
    i0 = int(math.floor(fx))
    j0 = int(math.floor(fy))
    k0 = int(math.floor(fz))
    if i0 < 0 or j0 < 0 or k0 < 0:
        return np.nan
    ax = fx - i0
    ay = fy - j0
    az = fz - k0
    held = 0
    acc = 0.0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                v = _fv8_node(prm, o, dat, l, i0 + di, j0 + dj, k0 + dk)
                if v != v:
                    v = _fv8_below_node(prm, o, dat, l, i0 + di, j0 + dj, k0 + dk)
                else:
                    held += 1
                acc += (ax if di else 1.0 - ax) * (ay if dj else 1.0 - ay) * (az if dk else 1.0 - az) * v
    if held == 0:
        return np.nan
    return acc


@njit(cache=True)
def _fv8(prm, o, px, py, pz):
    """her FV8 store (core/fv8base.py header) as an editor sees it: the finest level holding a brick at the point,
    trilinear over its 8 samples, a sample stored AT the cap replaced by the coarser levels' value there (the store's
    pre_estimate / grid_L0 rule); a level whose tile holds no brick there bounds |F| below by its band - h; outside
    every brick + the largest bound (her interior is covered by L3). r-ptr: the samples are read in place from the
    store's shared float32 array (header slots 46-47)."""
    dat = _fv8_dat(prm, o)
    tm = prm[o + 3]
    nx = int(prm[o + 4])
    ny = int(prm[o + 5])
    nz = int(prm[o + 6])
    gx = (px - prm[o]) / tm
    gy = (py - prm[o + 1]) / tm
    gz = (pz - prm[o + 2]) / tm
    lb = prm[o + 7]
    if gx < 0.0 or gy < 0.0 or gz < 0.0 or gx >= nx or gy >= ny or gz >= nz:
        qx = max(max(-gx, gx - nx), 0.0) * tm
        qy = max(max(-gy, gy - ny), 0.0) * tm
        qz = max(max(-gz, gz - nz), 0.0) * tm
        return math.sqrt(qx * qx + qy * qy + qz * qz) + lb
    best = lb
    for l in range(4):
        if int(prm[o + 16 + l]) == 0:
            continue
        bid, bx, by, bz = _fv8_bid(prm, o, dat, l, px, py, pz)
        if bid < 0:
            # 11-09 (a card / CPU desync): outside the finest level's bricks her field is the trilinear of ONE
            # value per node of that level (_fv8_V): continuous everywhere, and exactly what the card samples (the
            # coarser levels' own trilinears and seams stepped 4-300 um across faces 2-8 mm out)
            h0 = prm[o + 8 + l]
            fx = (px - prm[o]) / h0
            fy = (py - prm[o + 1]) / h0
            fz = (pz - prm[o + 2]) / h0
            i0 = int(math.floor(fx))
            j0 = int(math.floor(fy))
            k0 = int(math.floor(fz))
            ax = fx - i0
            ay = fy - j0
            az = fz - k0
            acc = 0.0
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        v = _fv8_V(prm, o, dat, l, i0 + di, j0 + dj, k0 + dk)
                        acc += (ax if di else 1.0 - ax) * (ay if dj else 1.0 - ay) * (az if dk else 1.0 - az) * v
            return acc
        if bid == -2:
            continue
        h = prm[o + 8 + l]
        if bid < 0:
            best = max(best, prm[o + 20 + l] - h)
            continue
        fx = (px - prm[o]) / h - bx * 8.0
        fy = (py - prm[o + 1]) / h - by * 8.0
        fz = (pz - prm[o + 2]) / h - bz * 8.0
        i = min(max(int(math.floor(fx)), 0), 7)
        j = min(max(int(math.floor(fy)), 0), 7)
        k = min(max(int(math.floor(fz)), 0), 7)
        ax = fx - i
        ay = fy - j
        az = fz - k
        s = int(prm[o + 34 + 2 * l]) * 1048576 + int(prm[o + 35 + 2 * l]) + bid * 729 + (i * 9 + j) * 9 + k
        cap = 0.999 * prm[o + 42 + l]
        acc = 0.0
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i + di == 8 or j + dj == 8 or k + dk == 8:
                        # 11-09: a node on the brick's upper face is the neighbour's node 0 where that brick exists (one
                        # value per node, however the point's position rounds: the store's bricks disagreed there)
                        v = _fv8_node(prm, o, dat, l, bx * 8 + i + di, by * 8 + j + dj, bz * 8 + k + dk)
                    else:
                        v = float(dat[s + di * 81 + dj * 9 + dk])
                        if abs(v) >= cap:       # by node index, as program.cu (a position on a face could miss)
                            v = _fv8_coarser_node(prm, o, dat, l, bx * 8 + i + di, by * 8 + j + dj, bz * 8 + k + dk, v)
                    w = (ax if di else 1.0 - ax) * (ay if dj else 1.0 - ay) * (az if dk else 1.0 - az)
                    acc += w * v
        return acc
    return best


@njit(cache=True, inline='always')
def _sampled(prm, o, px, py, pz):
    """max(trilinear of the lattice, crop SDF). Header at prm[o:o+16]: grid lo (3), h, dims (3), crop type (0 box,
    1 sphere), crop centre (3), crop half sizes (3; sphere: radius first), -, data offset relative to o. Outside the
    lattice (which covers the crop + a 2-node apron) the crop SDF alone: a lower bound of the distance (D)."""
    cx = px - prm[o + 8]
    cy = py - prm[o + 9]
    cz = pz - prm[o + 10]
    if prm[o + 7] > 0.5:
        cs = math.sqrt(cx * cx + cy * cy + cz * cz) - prm[o + 11]
    else:
        qx = abs(cx) - prm[o + 11]
        qy = abs(cy) - prm[o + 12]
        qz = abs(cz) - prm[o + 13]
        mx = max(qx, 0.0)
        my = max(qy, 0.0)
        mz = max(qz, 0.0)
        cs = math.sqrt(mx * mx + my * my + mz * mz) + min(max(qx, max(qy, qz)), 0.0)
    h = prm[o + 3]
    nx = int(prm[o + 4])
    ny = int(prm[o + 5])
    nz = int(prm[o + 6])
    fx = (px - prm[o]) / h
    fy = (py - prm[o + 1]) / h
    fz = (pz - prm[o + 2]) / h
    i = int(math.floor(fx))
    j = int(math.floor(fy))
    k = int(math.floor(fz))
    if i < 0 or j < 0 or k < 0 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return cs + prm[o + 14] if cs > 0.0 else cs     # 11-09: + the boundary's smallest value (0 for 'sampled')
    tx = fx - i
    ty = fy - j
    tz = fz - k
    d = o + int(prm[o + 15])
    b = d + (i * ny + j) * nz + k
    syz = ny * nz
    v000 = prm[b]
    v001 = prm[b + 1]
    v010 = prm[b + nz]
    v011 = prm[b + nz + 1]
    v100 = prm[b + syz]
    v101 = prm[b + syz + 1]
    v110 = prm[b + syz + nz]
    v111 = prm[b + syz + nz + 1]
    a = v000 + (v100 - v000) * tx
    bb = v010 + (v110 - v010) * tx
    c = v001 + (v101 - v001) * tx
    dd = v011 + (v111 - v011) * tx
    e = a + (bb - a) * ty
    f = c + (dd - c) * ty
    return max(e + (f - e) * tz, cs)


@njit(cache=True, inline='always')
def _osweep(prm, o, px, py, pz):
    """Signed distance to an outline swept from the eye (OP_OSWEEP; ops.py 'outline_sweep'), clipped to a cube.

    Header at prm[o:o+15]: eye E (3), view-plane basis u (3), v (3), view axis m (3), focal distance f, cube half
    size e, n. Then n plane points (x_i, y_i) (the outline on the plane m.(p - E) = f, in (u, v) coords), then n
    unit rays A_i = unit(f m + x_i u + y_i v).
    The solid S = {p : z = m.r > 0 and (f (u.r)/z, f (v.r)/z) inside the outline (even-odd crossing test)},
    r = p - E. Its boundary is the union of the planar wedges W_i = {a A_i + b A_(i+1) : a, b >= 0} (the cone over
    each outline edge; all rays point forward, so every wedge angle is < pi). The value is EXACT: |F| = min_i
    dist(r, W_i), with dist(r, W) = |r.n| / |n| (n = A_i x A_(i+1)) when r lies over the wedge ((A_i x r).n >= 0
    and (r x A_(i+1)).n >= 0), else the distance to the nearer bounding ray (|r| when r.A <= 0, else
    sqrt(|r|^2 - (r.A)^2)); sign - inside S. The exact distance to a closed set is 1-Lipschitz, and the sign flips
    only across that set, so F is 1-Lipschitz (D) wherever it is evaluated, including behind the eye (z <= 0:
    outside, the distance to the nearest wedge, which includes the apex E). Finally max(F, cube SDF): L = 1.
    (The cheaper 'scaled 2-D distance' d2 * z / f is the in-plane distance to the cross-section at depth z, an
    UPPER bound of the 3-D distance, so it can exceed |r - surface| and break L <= 1 at oblique walls; not used.)"""
    rx = px - prm[o]
    ry = py - prm[o + 1]
    rz = pz - prm[o + 2]
    n = int(prm[o + 14])
    pb = o + 15
    ab = o + 15 + 2 * n
    rr = rx * rx + ry * ry + rz * rz
    best = 1e30
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        ax = prm[ab + 3 * i]
        ay = prm[ab + 3 * i + 1]
        az = prm[ab + 3 * i + 2]
        # ray A_i
        t = rx * ax + ry * ay + rz * az
        if t <= 0.0:
            d2 = rr
        else:
            d2 = max(rr - t * t, 0.0)
        if d2 < best:
            best = d2
        # wedge (A_i, A_j)
        bx = prm[ab + 3 * j]
        by = prm[ab + 3 * j + 1]
        bz = prm[ab + 3 * j + 2]
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx
        nn = nx * nx + ny * ny + nz * nz
        if nn > 1e-24:
            # (A x r).n and (r x B).n
            c1x = ay * rz - az * ry
            c1y = az * rx - ax * rz
            c1z = ax * ry - ay * rx
            c2x = ry * bz - rz * by
            c2y = rz * bx - rx * bz
            c2z = rx * by - ry * bx
            if c1x * nx + c1y * ny + c1z * nz >= 0.0 and c2x * nx + c2y * ny + c2z * nz >= 0.0:
                s = rx * nx + ry * ny + rz * nz
                d2 = s * s / nn
                if d2 < best:
                    best = d2
    dist = math.sqrt(best)
    z = rx * prm[o + 9] + ry * prm[o + 10] + rz * prm[o + 11]
    inside = False
    if z > 0.0:
        f = prm[o + 12]
        X = f * (rx * prm[o + 3] + ry * prm[o + 4] + rz * prm[o + 5]) / z
        Y = f * (rx * prm[o + 6] + ry * prm[o + 7] + rz * prm[o + 8]) / z
        k = n - 1
        for i in range(n):
            xi = prm[pb + 2 * i]
            yi = prm[pb + 2 * i + 1]
            xk = prm[pb + 2 * k]
            yk = prm[pb + 2 * k + 1]
            if (yi > Y) != (yk > Y):
                if X < (xk - xi) * (Y - yi) / (yk - yi) + xi:
                    inside = not inside
            k = i
    sd = -dist if inside else dist
    e = prm[o + 13]
    qx = abs(px) - e
    qy = abs(py) - e
    qz = abs(pz) - e
    mx = max(qx, 0.0)
    my = max(qy, 0.0)
    mz = max(qz, 0.0)
    bx = math.sqrt(mx * mx + my * my + mz * mz) + min(max(qx, max(qy, qz)), 0.0)
    return max(sd, bx)


@njit(cache=True)
def eval_prog(code, prm, c0, c1, px, py, pz, vs, ps):
    """Evaluate the program rows [c0, c1) at point p. vs: float64[STACK], ps: float64[4*STACK] scratch."""
    return eval_prog_t(code, prm, c0, c1, px, py, pz, vs, ps, 0.0)


@njit(cache=True)
def eval_prog_t(code, prm, c0, c1, px, py, pz, vs, ps, tau):
    """eval_prog with a guard threshold: a guard skips its subtree only when the point is outside its box AND its
    lower bound (box distance + margin, in the program's root units) is >= tau. tau = 0 is the tracer's rule (skip
    whenever outside). Shading passes tau = the largest value it reads (an AO sample height), so wherever a guard's
    lower bound would be below that height the real subtree is evaluated instead: no F jump at guard boxes inside
    the range the shading looks at (the cause of the faint rectangles near guard boxes)."""
    sp = 0
    pp = 0
    sc = 1.0            # accumulated value scale of the pushed transforms (root units = local * sc)
    i = c0
    while i < c1:
        op = code[i, 0]
        o = code[i, 1]
        if op == OP_SPHERE:
            vs[sp] = math.sqrt(px * px + py * py + pz * pz) - prm[o]
            sp += 1
        elif op == OP_BOX:
            r = prm[o + 3]
            qx = abs(px) - prm[o] + r
            qy = abs(py) - prm[o + 1] + r
            qz = abs(pz) - prm[o + 2] + r
            mx = max(qx, 0.0)
            my = max(qy, 0.0)
            mz = max(qz, 0.0)
            vs[sp] = math.sqrt(mx * mx + my * my + mz * mz) + min(max(qx, max(qy, qz)), 0.0) - r
            sp += 1
        elif op == OP_CYL:
            rr = prm[o + 2]
            dx = math.sqrt(px * px + pz * pz) - prm[o] + rr
            dy = abs(py) - prm[o + 1] + rr
            mx = max(dx, 0.0)
            my = max(dy, 0.0)
            vs[sp] = min(max(dx, dy), 0.0) + math.sqrt(mx * mx + my * my) - rr
            sp += 1
        elif op == OP_CAPSULE:
            hh = prm[o]
            qy = py - min(max(py, -hh), hh)
            vs[sp] = math.sqrt(px * px + qy * qy + pz * pz) - prm[o + 1]
            sp += 1
        elif op == OP_CONE:
            # capped cone along y in [-h, h], radius r1 at -h, r2 at +h (exact distance)
            r1 = prm[o]
            r2 = prm[o + 1]
            h = prm[o + 2]
            qx = math.sqrt(px * px + pz * pz)
            qy = py
            k2x = r2 - r1
            k2y = 2.0 * h
            rr = r1 if qy < 0.0 else r2
            cax = qx - min(qx, rr)
            cay = abs(qy) - h
            t = ((r2 - qx) * k2x + (h - qy) * k2y) / (k2x * k2x + k2y * k2y)
            t = min(max(t, 0.0), 1.0)
            cbx = qx - r2 + k2x * t
            cby = qy - h + k2y * t
            s = -1.0 if (cbx < 0.0 and cay < 0.0) else 1.0
            vs[sp] = s * math.sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby))
            sp += 1
        elif op == OP_TORUS:
            qx = math.sqrt(px * px + pz * pz) - prm[o]
            vs[sp] = math.sqrt(qx * qx + py * py) - prm[o + 1]
            sp += 1
        elif op == OP_ELLIPSOID:
            rx = prm[o]
            ry = prm[o + 1]
            rz = prm[o + 2]
            rmin = min(rx, min(ry, rz))
            ex = px / rx
            ey = py / ry
            ez = pz / rz
            vs[sp] = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * rmin
            sp += 1
        elif op == OP_SLAB:
            d = abs(px * prm[o] + py * prm[o + 1] + pz * prm[o + 2] - prm[o + 3]) - 0.5 * prm[o + 4]
            e = prm[o + 5]
            qx = abs(px) - e
            qy = abs(py) - e
            qz = abs(pz) - e
            mx = max(qx, 0.0)
            my = max(qy, 0.0)
            mz = max(qz, 0.0)
            b = math.sqrt(mx * mx + my * my + mz * mz) + min(max(qx, max(qy, qz)), 0.0)
            vs[sp] = max(d, b)
            sp += 1
        elif op == OP_SAMPLED:
            vs[sp] = _sampled(prm, o, px, py, pz)
            sp += 1
        elif op == OP_FV8:
            vs[sp] = _fv8(prm, o, px, py, pz)
            sp += 1
        elif op == OP_OSWEEP:
            vs[sp] = _osweep(prm, o, px, py, pz)
            sp += 1
        elif op >= OP_UNION and op <= OP_SINTER:
            sp -= 1
            b = vs[sp]
            a = vs[sp - 1]
            if op == OP_UNION:
                r = min(a, b)
            elif op == OP_SUB:
                r = max(a, -b)
            elif op == OP_INTER:
                r = max(a, b)
            elif op == OP_SUNION:
                r = _smin(a, b, prm[o])
            elif op == OP_SSUB:
                r = -_smin(-a, b, prm[o])
            else:
                r = -_smin(-a, -b, prm[o])
            vs[sp - 1] = r
        elif op == OP_XFORM:
            ps[4 * pp] = px
            ps[4 * pp + 1] = py
            ps[4 * pp + 2] = pz
            ps[4 * pp + 3] = prm[o + 12]
            sc *= prm[o + 12]
            pp += 1
            dx = px - prm[o + 9]
            dy = py - prm[o + 10]
            dz = pz - prm[o + 11]
            px = prm[o] * dx + prm[o + 1] * dy + prm[o + 2] * dz
            py = prm[o + 3] * dx + prm[o + 4] * dy + prm[o + 5] * dz
            pz = prm[o + 6] * dx + prm[o + 7] * dy + prm[o + 8] * dz
        elif op == OP_TWIST or op == OP_BEND or op == OP_TAPER:
            ps[4 * pp] = px
            ps[4 * pp + 1] = py
            ps[4 * pp + 2] = pz
            ps[4 * pp + 3] = 1.0
            pp += 1
            k = prm[o]
            if op == OP_TWIST:          # rotate xz by k*y
                a = k * py
                c = math.cos(a)
                s = math.sin(a)
                nx = c * px + s * pz
                pz = -s * px + c * pz
                px = nx
            elif op == OP_BEND:         # rotate xy by k*x
                a = k * px
                c = math.cos(a)
                s = math.sin(a)
                nx = c * px - s * py
                py = s * px + c * py
                px = nx
            else:                       # taper: scale xz by 1/(1 + k*y)
                g = max(1.0 + k * py, prm[o + 1])
                px = px / g
                pz = pz / g
        elif op == OP_POP:
            pp -= 1
            sc /= ps[4 * pp + 3]
            px = ps[4 * pp]
            py = ps[4 * pp + 1]
            pz = ps[4 * pp + 2]
            vs[sp - 1] *= ps[4 * pp + 3]
        elif op == OP_SHELL:
            vs[sp - 1] = abs(vs[sp - 1]) - prm[o]
        elif op == OP_OFFSET:
            vs[sp - 1] = vs[sp - 1] - prm[o]
        elif op == OP_MATGEN:
            vs[sp - 1] = _matgen(prm, o, vs[sp - 1], px, py, pz)
        elif op == OP_GUARD_U:
            # hand2b (verifier items 4, 5a): the operand B of a union / smooth union (blend k = prm[o + 7]) is
            # skipped only when its lower bound gv >= acc + k (acc = the value below it on the stack): then
            # min(acc, B) = acc and smin_k(acc, B) = acc exactly, so the guarded field equals the full field
            # EVERYWHERE, not only near the base surface. The plain guard (skip whenever outside the box) returned
            # the box bound where a merged part's sculpt had moved its surface outside the part's box (a Grab
            # pulling 40 mm), and the merged sculpt delta, computed against the full field, then put the surface
            # off by the difference: the lips, ledges and step edges along the guard box boundary.
            if sp >= 1:
                dx = max(prm[o] - px, px - prm[o + 3])
                dy = max(prm[o + 1] - py, py - prm[o + 4])
                dz = max(prm[o + 2] - pz, pz - prm[o + 5])
                if dx > 0.0 or dy > 0.0 or dz > 0.0:
                    mx = max(dx, 0.0)
                    my = max(dy, 0.0)
                    mz = max(dz, 0.0)
                    gv = math.sqrt(mx * mx + my * my + mz * mz) + prm[o + 6]
                    if gv >= vs[sp - 1] + prm[o + 7] and gv * sc >= tau:
                        vs[sp] = gv
                        sp += 1
                        i = code[i, 2]
                        continue
        elif op == OP_GUARD:
            dx = max(prm[o] - px, px - prm[o + 3])
            dy = max(prm[o + 1] - py, py - prm[o + 4])
            dz = max(prm[o + 2] - pz, pz - prm[o + 5])
            if dx > 0.0 or dy > 0.0 or dz > 0.0:
                mx = max(dx, 0.0)
                my = max(dy, 0.0)
                mz = max(dz, 0.0)
                gv = math.sqrt(mx * mx + my * my + mz * mz) + prm[o + 6]
                if gv * sc >= tau:
                    vs[sp] = gv
                    sp += 1
                    i = code[i, 2]
                    continue
        i += 1
    return vs[0]


@njit(cache=True, inline='always')
def _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i, j, k):
    bx = (i >> BSH) - ox
    by = (j >> BSH) - oy
    bz = (k >> BSH) - oz
    if bx < 0 or by < 0 or bz < 0 or bx >= nx or by >= ny or bz >= nz:
        return 0.0
    slot = index_flat[off + (bx * ny + by) * nz + bz]
    if slot < 0:
        return 0.0
    return pool[slot, i & BMASK, j & BMASK, k & BMASK]


@njit(cache=True)
def grid_sample(index_flat, pool, oi_tab, oi, h, x, y, z):
    """Trilinear sample of a sparse brick grid (delta-F or mask) at local point (x, y, z)."""
    nx = oi_tab[oi, OI_NX]
    if nx == 0:
        return 0.0
    ny = oi_tab[oi, OI_NY]
    nz = oi_tab[oi, OI_NZ]
    ox = oi_tab[oi, OI_OX]
    oy = oi_tab[oi, OI_OY]
    oz = oi_tab[oi, OI_OZ]
    off = oi_tab[oi, OI_IDX]
    fx = x / h
    fy = y / h
    fz = z / h
    i0 = int(math.floor(fx))
    j0 = int(math.floor(fy))
    k0 = int(math.floor(fz))
    tx = fx - i0
    ty = fy - j0
    tz = fz - k0
    li = i0 & BMASK
    lj = j0 & BMASK
    lk = k0 & BMASK
    if li < BMASK and lj < BMASK and lk < BMASK:
        bx = (i0 >> BSH) - ox
        by = (j0 >> BSH) - oy
        bz = (k0 >> BSH) - oz
        if bx < 0 or by < 0 or bz < 0 or bx >= nx or by >= ny or bz >= nz:
            return 0.0
        slot = index_flat[off + (bx * ny + by) * nz + bz]
        if slot < 0:
            return 0.0
        v000 = pool[slot, li, lj, lk]
        v100 = pool[slot, li + 1, lj, lk]
        v010 = pool[slot, li, lj + 1, lk]
        v110 = pool[slot, li + 1, lj + 1, lk]
        v001 = pool[slot, li, lj, lk + 1]
        v101 = pool[slot, li + 1, lj, lk + 1]
        v011 = pool[slot, li, lj + 1, lk + 1]
        v111 = pool[slot, li + 1, lj + 1, lk + 1]
    else:
        v000 = _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i0, j0, k0)
        v100 = _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i0 + 1, j0, k0)
        v010 = _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i0, j0 + 1, k0)
        v110 = _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i0 + 1, j0 + 1, k0)
        v001 = _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i0, j0, k0 + 1)
        v101 = _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i0 + 1, j0, k0 + 1)
        v011 = _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i0, j0 + 1, k0 + 1)
        v111 = _node(index_flat, pool, off, nx, ny, nz, ox, oy, oz, i0 + 1, j0 + 1, k0 + 1)
    a = v000 + (v100 - v000) * tx
    b = v010 + (v110 - v010) * tx
    c = v001 + (v101 - v001) * tx
    d = v011 + (v111 - v011) * tx
    e = a + (b - a) * ty
    f = c + (d - c) * ty
    return e + (f - e) * tz


@njit(cache=True)
def mask_sample_dir(index_flat, mask, mdir, oi_tab, oi, h, x, y, z, nx, ny, nz, cdot):
    """direction-tagged mask at local point (x, y, z) of a surface whose normal is n: trilinear over the 8 nodes, a
    node's paint counted only if its paint direction d is untagged (0) or d . n > cdot (cos of the cutoff)"""
    nxo = oi_tab[oi, OI_NX]
    if nxo == 0:
        return 0.0
    ny_ = oi_tab[oi, OI_NY]
    nz_ = oi_tab[oi, OI_NZ]
    ox = oi_tab[oi, OI_OX]
    oy = oi_tab[oi, OI_OY]
    oz = oi_tab[oi, OI_OZ]
    off = oi_tab[oi, OI_IDX]
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
    for t in range(8):
        di = t & 1
        dj = (t >> 1) & 1
        dk = (t >> 2) & 1
        i = i0 + di
        j = j0 + dj
        k = k0 + dk
        bx = (i >> BSH) - ox
        by = (j >> BSH) - oy
        bz = (k >> BSH) - oz
        if bx < 0 or by < 0 or bz < 0 or bx >= nxo or by >= ny_ or bz >= nz_:
            continue
        slot = index_flat[off + (bx * ny_ + by) * nz_ + bz]
        if slot < 0:
            continue
        m = mask[slot, i & BMASK, j & BMASK, k & BMASK]
        if m == 0.0:
            continue
        ax = float(mdir[slot, i & BMASK, j & BMASK, k & BMASK, 0])
        ay = float(mdir[slot, i & BMASK, j & BMASK, k & BMASK, 1])
        az = float(mdir[slot, i & BMASK, j & BMASK, k & BMASK, 2])
        if ax != 0.0 or ay != 0.0 or az != 0.0:
            if (ax * nx + ay * ny + az * nz) <= cdot * math.sqrt(ax * ax + ay * ay + az * az):
                continue
        w = (tx if di else 1.0 - tx) * (ty if dj else 1.0 - ty) * (tz if dk else 1.0 - tz)
        acc += w * m
    return acc


@njit(cache=True)
def eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps):
    """Object field in its local frame: base program + sculpt delta-F, guarded by the object's box."""
    return eval_local_t(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps, 0.0)


@njit(cache=True)
def eval_local_t(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps, tau):
    """eval_local with the guard threshold tau (local units; see eval_prog_t), the object box included."""
    dx = max(of_tab[oi, OF_BMIN] - x, x - of_tab[oi, OF_BMAX])
    dy = max(of_tab[oi, OF_BMIN + 1] - y, y - of_tab[oi, OF_BMAX + 1])
    dz = max(of_tab[oi, OF_BMIN + 2] - z, z - of_tab[oi, OF_BMAX + 2])
    if dx > 0.0 or dy > 0.0 or dz > 0.0:
        mx = max(dx, 0.0)
        my = max(dy, 0.0)
        mz = max(dz, 0.0)
        gv = math.sqrt(mx * mx + my * my + mz * mz) + of_tab[oi, OF_MARGIN]
        if gv >= tau:
            return gv
    f = eval_prog_t(code, prm, oi_tab[oi, OI_C0], oi_tab[oi, OI_C1], x, y, z, vs, ps, tau)
    return f + grid_sample(index_flat, delta, oi_tab, oi, of_tab[oi, OF_H], x, y, z)


@njit(cache=True, inline='always')
def to_local(of_tab, oi, x, y, z):
    s = of_tab[oi, OF_S]
    dx = x - of_tab[oi, OF_T]
    dy = y - of_tab[oi, OF_T + 1]
    dz = z - of_tab[oi, OF_T + 2]
    lx = (of_tab[oi, 0] * dx + of_tab[oi, 1] * dy + of_tab[oi, 2] * dz) / s
    ly = (of_tab[oi, 3] * dx + of_tab[oi, 4] * dy + of_tab[oi, 5] * dz) / s
    lz = (of_tab[oi, 6] * dx + of_tab[oi, 7] * dy + of_tab[oi, 8] * dz) / s
    return lx, ly, lz


@njit(cache=True)
def eval_world(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps):
    lx, ly, lz = to_local(of_tab, oi, x, y, z)
    return of_tab[oi, OF_S] * eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, lx, ly, lz, vs, ps)


@njit(cache=True)
def eval_scene(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, vs, ps):
    """Returns (F = min over visible objects, bound = min F_i / L_i (a distance lower bound), object index)."""
    fbest = 1e30
    dbest = 1e30
    obest = -1
    for oi in range(oi_tab.shape[0]):
        if of_tab[oi, OF_VIS] < 0.5:
            continue
        f = eval_world(code, prm, oi_tab, of_tab, index_flat, delta, oi, x, y, z, vs, ps)
        d = f / of_tab[oi, OF_L]
        if d < dbest:
            dbest = d
            obest = oi
        if f < fbest:
            fbest = f
    return fbest, dbest, obest


@njit(cache=True)
def eval_scene_f(code, prm, oi_tab, of_tab, index_flat, delta, x, y, z, vs, ps, tau):
    """F = min over visible objects with the guard threshold tau (world units): exact wherever F < tau, and >= tau
    elsewhere. For shading (AO), which only reads min(F, tau)."""
    fbest = 1e30
    for oi in range(oi_tab.shape[0]):
        if of_tab[oi, OF_VIS] < 0.5:
            continue
        lx, ly, lz = to_local(of_tab, oi, x, y, z)
        s = of_tab[oi, OF_S]
        f = s * eval_local_t(code, prm, oi_tab, of_tab, index_flat, delta, oi, lx, ly, lz, vs, ps, tau / s)
        if f < fbest:
            fbest = f
    return fbest


@njit(cache=True)
def eval_points(code, prm, oi_tab, of_tab, index_flat, delta, pts, out_f, out_d, out_o):
    vs = np.empty(STACK)
    ps = np.empty(4 * STACK)
    for n in range(pts.shape[0]):
        f, d, o = eval_scene(code, prm, oi_tab, of_tab, index_flat, delta, pts[n, 0], pts[n, 1], pts[n, 2], vs, ps)
        out_f[n] = f
        out_d[n] = d
        out_o[n] = o


@njit(cache=True)
def eval_points_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts, out):
    vs = np.empty(STACK)
    ps = np.empty(4 * STACK)
    for n in range(pts.shape[0]):
        out[n] = eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts[n, 0], pts[n, 1], pts[n, 2], vs, ps)
