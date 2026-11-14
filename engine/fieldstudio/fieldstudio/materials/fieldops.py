"""fs-materials (2026-11-14): material generators as STRUCTURE in the field (P16: built, not painted).

One opcode, program.OP_MATGEN: a value mutation of the subtree below it, at the current point q (the subtree's own
units, metres in the object's frame). Every generator is real geometry: the tracer marches it, the sun and sky are
shadowed by it, Section cuts it, Measure and physics read it.

  kind 1  STONE RELIEF   F' = F - r(q). r = fBm undulation (each octave turned by its own seeded rotation, so no
                         lattice axis shows) + plucked-grain pits (a cellular field: per cell a seeded chance of a pit;
                         the pits are bowls, combined by min so the field stays continuous) + bedding ridges (optional).
                         Near the surface F is a distance, so the surface moves along its normal by r.
  kind 2  CHIPS          F' = max(F, -min_k E_k(q)): conchoidal edge chips, each an oblate ellipsoid scoop (its rim
                         sharp, its scar roughened by a fracture fBm), placed by generate.py on the form's edges.
  kind 3  CRACK          F' = max(F, -d_c(q)), d_c = max(|S| - w/2, s - s_tip, |t| - half_len): a thin cavity about a
                         rough fracture surface S(q) = n.(q - c) + fBm(q) (self-affine, generate.py), opening
                         w(s) = w0 sqrt(1 - s / s_tip) from the mouth (s = 0) to a wandering tip front s_tip (the
                         linear-elastic crack-opening profile near a tip; slope capped so the field stays Lipschitz).

Parameter block at prm[o]: [kind, n_floats, ...] (layouts in the pack_* functions below). The noise is a 32-bit integer
hash with Perlin's improved gradient set, written identically in fieldops.cuh, so the card's brick evaluator
(gpu/program.cu) and this CPU evaluator give the same field to float precision.
"""
import math

import numpy as np
from numba import njit

KIND_RELIEF, KIND_CHIPS, KIND_CRACK = 1, 2, 3
OCT = 14                 # one octave record: freq, amp, R (9, row-major), offset (3)
CHIP = 15                # one chip: centre (3), R rows u, v, n (9), radii (3)
M32 = 0xFFFFFFFF


# ---- hash noise (identical in fieldops.cuh) --------------------------------------------------------------------------
@njit(cache=True, inline='always')
def h32(x):
    x = x & M32
    x = (((x >> 16) ^ x) * 0x45d9f3b) & M32
    x = (((x >> 16) ^ x) * 0x45d9f3b) & M32
    return (x >> 16) ^ x


@njit(cache=True, inline='always')
def hc(ix, iy, iz, seed):
    return h32((ix & M32) ^ h32((iy & M32) ^ h32((iz & M32) ^ h32(seed & M32))))


@njit(cache=True, inline='always')
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, inline='always')
def _grad(h, x, y, z):
    h = h & 15
    u = x if h < 8 else y
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    else:
        v = z
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


@njit(cache=True)
def noise3(x, y, z, seed):
    """improved-Perlin gradient noise, hash lattice; raw range about +-1, SD 0.27 (RAW_SD)"""
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    xf = x - fx
    yf = y - fy
    zf = z - fz
    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)
    res = 0.0
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                g = _grad(hc(ix + dx, iy + dy, iz + dz, seed), xf - dx, yf - dy, zf - dz)
                wx = u if dx else 1.0 - u
                wy = v if dy else 1.0 - v
                wz = w if dz else 1.0 - w
                res += wx * wy * wz * g
    return res


@njit(cache=True)
def fbm(prm, b, n, x, y, z):
    """sum of n octave records at prm[b:]: amp * noise(freq * R q + off); amp already holds 1 / RAW_SD"""
    s = 0.0
    for k in range(n):
        r = b + k * OCT
        f = prm[r]
        X = (prm[r + 2] * x + prm[r + 3] * y + prm[r + 4] * z) * f + prm[r + 11]
        Y = (prm[r + 5] * x + prm[r + 6] * y + prm[r + 7] * z) * f + prm[r + 12]
        Z = (prm[r + 8] * x + prm[r + 9] * y + prm[r + 10] * z) * f + prm[r + 13]
        s += prm[r + 1] * noise3(X, Y, Z, k * 7919 + 17)
    return s


@njit(cache=True)
def noise3g(x, y, z, seed):
    """noise3 and its analytic gradient -> (value, d/dx, d/dy, d/dz)"""
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    xf = x - fx
    yf = y - fy
    zf = z - fz
    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)
    du = 30.0 * xf * xf * (xf - 1.0) * (xf - 1.0)
    dv = 30.0 * yf * yf * (yf - 1.0) * (yf - 1.0)
    dw = 30.0 * zf * zf * (zf - 1.0) * (zf - 1.0)
    res = 0.0
    gx = 0.0
    gy = 0.0
    gz = 0.0
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                h = hc(ix + dx, iy + dy, iz + dz, seed) & 15
                su = 1.0 if (h & 1) == 0 else -1.0
                sv = 1.0 if (h & 2) == 0 else -1.0
                g0 = 0.0
                g1 = 0.0
                g2 = 0.0
                if h < 8:
                    g0 += su
                else:
                    g1 += su
                if h < 4:
                    g1 += sv
                elif h == 12 or h == 14:
                    g0 += sv
                else:
                    g2 += sv
                dot = g0 * (xf - dx) + g1 * (yf - dy) + g2 * (zf - dz)
                wx = u if dx else 1.0 - u
                wy = v if dy else 1.0 - v
                wz = w if dz else 1.0 - w
                dwx = du if dx else -du
                dwy = dv if dy else -dv
                dwz = dw if dz else -dw
                www = wx * wy * wz
                res += www * dot
                gx += dwx * wy * wz * dot + www * g0
                gy += wx * dwy * wz * dot + www * g1
                gz += wx * wy * dwz * dot + www * g2
    return res, gx, gy, gz


@njit(cache=True)
def fbm_g(prm, b, n, x, y, z):
    """fbm and its gradient -> (value, d/dx, d/dy, d/dz)"""
    s = 0.0
    gx = 0.0
    gy = 0.0
    gz = 0.0
    for k in range(n):
        r = b + k * OCT
        f = prm[r]
        X = (prm[r + 2] * x + prm[r + 3] * y + prm[r + 4] * z) * f + prm[r + 11]
        Y = (prm[r + 5] * x + prm[r + 6] * y + prm[r + 7] * z) * f + prm[r + 12]
        Z = (prm[r + 8] * x + prm[r + 9] * y + prm[r + 10] * z) * f + prm[r + 13]
        v, nx, ny, nz = noise3g(X, Y, Z, k * 7919 + 17)
        a = prm[r + 1]
        s += a * v
        af = a * f
        gx += af * (nx * prm[r + 2] + ny * prm[r + 5] + nz * prm[r + 8])
        gy += af * (nx * prm[r + 3] + ny * prm[r + 6] + nz * prm[r + 9])
        gz += af * (nx * prm[r + 4] + ny * prm[r + 7] + nz * prm[r + 10])
    return s, gx, gy, gz


@njit(cache=True, inline='always')
def u01(h):
    return (h & 0xFFFFFF) / 16777216.0


@njit(cache=True)
def pits(cell, prob, depth, rfac, seed, x, y, z):
    """plucked grains: the deepest bowl of the 27 cells around q (each cell holds a pit with probability prob), and its
    gradient -> (value <= 0, d/dx, d/dy, d/dz)"""
    gx = x / cell
    gy = y / cell
    gz = z / cell
    cx = int(math.floor(gx))
    cy = int(math.floor(gy))
    cz = int(math.floor(gz))
    best = 0.0
    bx = 0.0
    by = 0.0
    bz = 0.0
    for i in range(cx - 1, cx + 2):
        for j in range(cy - 1, cy + 2):
            for k in range(cz - 1, cz + 2):
                h0 = hc(i, j, k, seed)
                if u01(h0) >= prob:
                    continue
                px = i + u01(hc(i, j, k, seed + 1))
                py = j + u01(hc(i, j, k, seed + 2))
                pz = k + u01(hc(i, j, k, seed + 3))
                rad = rfac * (0.7 + 0.6 * u01(hc(i, j, k, seed + 4)))       # in cells
                sx = 0.65 + 0.8 * u01(hc(i, j, k, seed + 6))                 # each pit its own aspect
                sy = 0.65 + 0.8 * u01(hc(i, j, k, seed + 7))
                sz = 0.65 + 0.8 * u01(hc(i, j, k, seed + 8))
                dx = (gx - px) * sx
                dy = (gy - py) * sy
                dz = (gz - pz) * sz
                r2 = rad * rad
                t2 = (dx * dx + dy * dy + dz * dz) / r2
                tx = 2.0 * dx * sx / (r2 * cell)                             # d t2 / d q
                ty = 2.0 * dy * sy / (r2 * cell)
                tz = 2.0 * dz * sz / (r2 * cell)
                if t2 < 1.43:                    # an irregular outline: t2 x f, f in [0.7, 1.3]; beyond 1 / 0.7 no
                    wn, wgx, wgy, wgz = noise3g(gx * 3.0, gy * 3.0, gz * 3.0, seed + 9)   # wobble reaches 1, so
                    wb = wn * 3.7                                            # the field stays continuous
                    k3 = 3.7 * 3.0 / cell
                    if wb > 1.0 or wb < -1.0:
                        wb = min(1.0, max(-1.0, wb))
                        k3 = 0.0
                    f_ = 1.0 + 0.3 * wb
                    tx = tx * f_ + t2 * 0.3 * k3 * wgx
                    ty = ty * f_ + t2 * 0.3 * k3 * wgy
                    tz = tz * f_ + t2 * 0.3 * k3 * wgz
                    t2 = t2 * f_
                if t2 < 1.0:
                    b = (1.0 - t2)
                    dep = depth * (0.5 + u01(hc(i, j, k, seed + 5)))
                    v = -dep * b * b
                    if v < best:
                        best = v
                        bx = 2.0 * dep * b * tx
                        by = 2.0 * dep * b * ty
                        bz = 2.0 * dep * b * tz
    return best, bx, by, bz


@njit(cache=True)
def relief(prm, o, x, y, z):
    """kind 1: the outward displacement r(q) (m) and its gradient -> (r, dr/dx, dr/dy, dr/dz)"""
    n = int(prm[o + 3])
    r, gx, gy, gz = fbm_g(prm, o + 4, n, x, y, z)
    p = o + 4 + n * OCT
    if prm[p] > 0.0:
        v, ax, ay, az = pits(prm[p], prm[p + 1], prm[p + 2], prm[p + 3], int(prm[p + 4]), x, y, z)
        r += v
        gx += ax
        gy += ay
        gz += az
    if prm[p + 9] != 0.0:
        sp = prm[p + 8]
        ph = (prm[p + 5] * x + prm[p + 6] * y + prm[p + 7] * z) / sp
        hx = prm[p + 5] / sp
        hy = prm[p + 6] / sp
        hz = prm[p + 7] / sp
        if prm[p + 10] != 0.0:
            wf = prm[p + 11]
            wn, wgx, wgy, wgz = noise3g(x * wf, y * wf, z * wf, 104729)
            kw = prm[p + 10] * 5.0
            ph += kw * wn
            hx += kw * wf * wgx
            hy += kw * wf * wgy
            hz += kw * wf * wgz
        # the periodic term (tool strokes; bedding ridges): scalloped, a round-bottomed cut per period meeting the next in
        # a sharp ridge, each period its own depth (0.6-1.4 x): r = -a j (1 - (2u - 1)^2), u = frac(ph). A claw chisel
        # (11-14): PASSES rows of tooth cuts, each pass offset and its own depth (0.5-1 x), the face the deepest cut
        # (min); each cut its own width (its cell's ends moved by up to +-jitter/2 of the pitch)
        a = prm[p + 9]
        npass = 1
        jit = 0.0
        gw = 1.0
        blows = prm[o + 1] >= (p - o) + 20 and prm[p + 17] > 0.0   # update 20: the blows
        if blows:
            lb = prm[p + 17]
            n2, g2x, g2y, g2z = noise3g(x / lb + 0.83, y / lb + 0.29, z / lb + 0.57, 104779)
            ko = prm[p + 19] * 3.71
            ph += ko * n2
            hx += ko * g2x / lb
            hy += ko * g2y / lb
            hz += ko * g2z / lb
        if prm[o + 1] >= (p - o) + 16:
            npass = max(1, int(prm[p + 14]))
            jit = prm[p + 15]
        if prm[o + 1] >= (p - o) + 17 and prm[p + 16] > 0.0:
            gw = min(prm[p + 16], 1.0)                 # the tooth's cut: a fraction of its cell, a land beside it
        sd = int(prm[p + 4])
        best = 0.0
        kk = 0.0
        bmx = 0.0
        bmy = 0.0
        bmz = 0.0
        for i in range(npass):
            phi = ph
            dsc = 1.0
            if i > 0:
                phi = ph + u01(hc(i, 13, sd, 104743))
                dsc = 0.5 + 0.5 * u01(hc(i, 19, sd, 104759))
            k_ = float(math.floor(phi))
            b0 = k_
            w_ = 1.0
            if jit > 0.0:
                b0 = k_ + jit * (u01(hc(int(k_), 29 + i, sd, 104761)) - 0.5)
                if phi < b0:
                    k_ -= 1.0
                    b0 = k_ + jit * (u01(hc(int(k_), 29 + i, sd, 104761)) - 0.5)
                b1 = k_ + 1.0 + jit * (u01(hc(int(k_) + 1, 29 + i, sd, 104761)) - 0.5)
                if phi >= b1:
                    k_ += 1.0
                    b0 = b1
                    b1 = k_ + 1.0 + jit * (u01(hc(int(k_) + 1, 29 + i, sd, 104761)) - 0.5)
                w_ = b1 - b0
            s_ = (2.0 * (phi - b0) / w_ - 1.0) / gw
            j_ = 0.6 + 0.8 * u01(hc(int(k_), 7, 11, 104723 + 97 * i))
            ri = 0.0
            ki = 0.0
            if s_ * s_ < 1.0:
                ri = -a * dsc * j_ * (1.0 - s_ * s_)
                ki = a * dsc * j_ * 4.0 * s_ / (w_ * gw)
            mx = 0.0
            my = 0.0
            mz = 0.0
            if blows and ri != 0.0:
                # 11-14 (blows had lined up across grooves into horizontal bands): EACH GROOVE its own blows,
                # the blow noise read at the groove's own offset (its index and pass), so they never line up
                ho = hc(int(k_), 41 + i, sd, 104789)
                o1 = 97.0 * u01(ho)
                o2 = 97.0 * u01(h32(ho ^ 0x2545f491))
                o3 = 97.0 * u01(h32(ho ^ 0x68e31da4))
                n1, g1x, g1y, g1z = noise3g(x / lb + o1, y / lb + o2, z / lb + o3, 104773)
                m_ = 0.85 + prm[p + 18] * 3.71 * n1
                if m_ <= 0.0:
                    ri = 0.0
                    ki = 0.0
                else:
                    mm = min(m_, 1.6)
                    if m_ < 1.6:
                        kb = prm[p + 18] * 3.71 / lb
                        mx = ri * kb * g1x
                        my = ri * kb * g1y
                        mz = ri * kb * g1z
                    ri *= mm
                    ki *= mm
            if i == 0 or ri < best:
                best = ri
                kk = ki
                bmx = mx
                bmy = my
                bmz = mz
        r += best
        gx += kk * hx + bmx
        gy += kk * hy + bmy
        gz += kk * hz + bmz
    return r, gx, gy, gz


@njit(cache=True)
def relief_value(prm, o, F, x, y, z):
    """B = prm[p + 12] = 0 (the default): F - r. B > 0: F' = (F - r) / k within B of the surface, k = sqrt(1 + |grad r|^2)
    (the first-order distance to the displaced surface), continuing at slope 1 / (1 + g_max) beyond. Off by default:
    grad r jumps where the deepest pit changes, so the divided field jumps there (measured 11-14); it needs a C1 r first
    (a smooth min over the pits, a smooth wobble clamp)."""
    r, gx, gy, gz = relief(prm, o, x, y, z)
    d = F - r
    p = o + 4 + int(prm[o + 3]) * OCT
    B = prm[p + 12]
    if B <= 0.0:
        return d
    k = math.sqrt(1.0 + gx * gx + gy * gy + gz * gz)
    ad = abs(d)
    if ad <= B:
        return d / k
    v = B / k + (ad - B) / (1.0 + prm[p + 13])
    return v if d > 0.0 else -v


@njit(cache=True)
def chips_d(prm, o, x, y, z):
    """kind 2: the distance bound to the union of the chip scoops (rough scars); 1e9 when there are none"""
    K = int(prm[o + 2])
    n = int(prm[o + 3])
    c0 = o + 4 + n * OCT
    best = 1e9
    for k in range(K):
        c = c0 + k * CHIP
        dx = x - prm[c]
        dy = y - prm[c + 1]
        dz = z - prm[c + 2]
        rx = prm[c + 12]
        ry = prm[c + 13]
        rz = prm[c + 14]
        rmin = min(rx, min(ry, rz))
        rmax = max(rx, max(ry, rz))
        dd = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dd - rmax > best:                     # its lower bound (|l / r| - 1) rmin >= (dd / rmax - 1) rmin
            if (dd / rmax - 1.0) * rmin > best:
                continue
        lx = (prm[c + 3] * dx + prm[c + 4] * dy + prm[c + 5] * dz) / rx
        ly = (prm[c + 6] * dx + prm[c + 7] * dy + prm[c + 8] * dz) / ry
        lz = (prm[c + 9] * dx + prm[c + 10] * dy + prm[c + 11] * dz) / rz
        v = (math.sqrt(lx * lx + ly * ly + lz * lz) - 1.0) * rmin
        if v < best:
            best = v
    ce = c0 + K * CHIP
    vwear = 1e9
    if prm[o + 1] >= (ce - o) + 8:                # the arris crumble (11-14): a tube about the as-fired box's arrises
        ax = abs(x) - prm[ce]
        ay = abs(y) - prm[ce + 1]
        az = abs(z) - prm[ce + 2]
        e = min(math.sqrt(ax * ax + ay * ay + max(az, 0.0) ** 2),
                min(math.sqrt(ax * ax + az * az + max(ay, 0.0) ** 2), math.sqrt(ay * ay + az * az + max(ax, 0.0) ** 2)))
        sag = prm[ce + 3]
        nx = int(prm[o + 1]) - (ce - o) - 8       # extra floats: 5 wear, 2 a second bite octave, 7 both
        if nx == 5 or nx == 7:                    # WEATHERING (11-14): the arrises worn round to rw (varying along them)
            lw = prm[ce + 10]
            nw = noise3(x / lw + 0.71, y / lw + 0.43, z / lw + 0.19, int(prm[ce + 12])) * 3.71
            rw = prm[ce + 8] + prm[ce + 9] * max(1.0 + prm[ce + 11] * nw, 0.0)
            sag = rw * 0.41421356                 # the crumble rides on the worn round
            bx = ax + rw
            by = ay + rw
            bz = az + rw
            px = max(bx, 0.0)
            py = max(by, 0.0)
            pz = max(bz, 0.0)
            sd = math.sqrt(px * px + py * py + pz * pz) + min(max(bx, max(by, bz)), 0.0) - rw
            vwear = max(-sd, -0.01)               # the cavity outside the worn box (clamped: still a lower bound),
            #                                       joined after the roughness (its faces lie ON the brick's faces)
        lam = prm[ce + 5]
        nn = noise3(x / lam + 0.31, y / lam + 0.17, z / lam + 0.53, int(prm[ce + 7])) * 3.71
        w = prm[ce + 4] * max(nn - prm[ce + 6], 0.0) / (1.0 - prm[ce + 6])
        if nx == 2 or nx == 7:                    # a second bite octave (11-14 update 13): irregular bites, same L
            c2 = ce + nx + 6
            l2 = prm[c2 + 1]
            n2 = noise3(x / l2 + 0.77, y / l2 + 0.41, z / l2 + 0.23, int(prm[ce + 7]) + 101) * 3.71
            w += prm[c2] * max(n2, 0.0)
        v = e - (sag + w)
        if v < best:
            best = v
    if n > 0 and best < 0.01:
        best += fbm(prm, o + 4, n, x, y, z)
    return min(best, vwear)


@njit(cache=True)
def crack_d(prm, o, x, y, z):
    """kind 3: the distance bound to the crack's cavity"""
    dx = x - prm[o + 2]
    dy = y - prm[o + 3]
    dz = z - prm[o + 4]
    sn = prm[o + 5] * dx + prm[o + 6] * dy + prm[o + 7] * dz
    s = prm[o + 8] * dx + prm[o + 9] * dy + prm[o + 10] * dz
    t = prm[o + 11] * dx + prm[o + 12] * dy + prm[o + 13] * dz
    D = prm[o + 14]
    w0 = prm[o + 15]
    hl = prm[o + 16]
    tf = prm[o + 18]
    s_tip = D * (1.0 + prm[o + 17] * noise3(t * tf, 0.5, 0.25, int(prm[o + 20])) * 5.0)
    if hl < 1.0:                                  # a partial crack: an elliptic front
        e2 = 1.0 - (t / hl) * (t / hl)
        s_tip = s_tip * math.sqrt(e2) if e2 > 0.0 else 0.0
    n = int(prm[o + 21])
    S = sn + fbm(prm, o + 22, n, x, y, z)
    if s <= 0.0:
        w = w0
    elif s_tip <= 0.0 or s >= s_tip:
        w = 0.0
    else:
        q = 1.0 - s / s_tip
        m = w0 / (2.0 * s_tip * prm[o + 19])      # below sqrt(q) = m the opening closes linearly (slope cap)
        sq = math.sqrt(q)
        w = w0 * sq if sq >= m else w0 * q / m
    d = abs(S) - 0.5 * w
    d = max(d, s - s_tip)
    if hl < 1.0:
        d = max(d, abs(t) - hl)
    return d


@njit(cache=True)
def matgen(prm, o, F, x, y, z):
    """the op: F (the subtree's value at q) -> F'"""
    kind = int(prm[o])
    if kind == KIND_RELIEF:
        return relief_value(prm, o, F, x, y, z)
    if kind == KIND_CHIPS:
        return max(F, -chips_d(prm, o, x, y, z))
    if kind == KIND_CRACK:
        return max(F, -crack_d(prm, o, x, y, z))
    return F


# ---- blocks (Python) -------------------------------------------------------------------------------------------------
RAW_SD = 0.2695          # noise3's SD over random points (measure_noise(), 2e5 points, 11-14)
RAW_GMAX = 3.0           # |grad noise3| bound in lattice units (measured max 2.69 over 2e4 points, 11-14; margin)


def rot_from(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def octaves(rng, lam_lo, lam_hi, rms, hurst=0.8, per_octave=2.0):
    """octave records with wavelengths log-spread over [lam_lo, lam_hi] (m), a power spectrum of a self-affine
    surface (amplitude ~ lambda^hurst), scaled so the whole sum has RMS rms (m). -> (records list, slope bound)"""
    if rms <= 0.0 or lam_hi <= 0.0:
        return [], 0.0
    lam_lo = min(lam_lo, lam_hi)
    n = max(1, int(round(math.log(lam_hi / lam_lo) / math.log(per_octave))) + 1)
    lams = np.geomspace(lam_hi, lam_lo, n)
    w = lams ** hurst
    w = w / math.sqrt(float((w ** 2).sum()))
    recs, slope = [], 0.0
    for lam, wk in zip(lams, w):
        R = rot_from(rng)
        off = rng.uniform(0, 256, 3)
        a = rms * wk / RAW_SD
        recs.append([1.0 / lam, a] + list(R.ravel()) + list(off))
        slope += a * RAW_GMAX / lam
    return recs, slope


def pack_relief(oct_recs, pits=None, bed=None, band=0.0):
    """pits: {cell, prob, depth, rfac, seed} (m); bed: {normal, spacing, amp, warp, warp_freq}; band: B (m), where the
    field is normalised to a distance (relief_value; 0: F - r as it is)"""
    b = [float(KIND_RELIEF), 0.0, 0.0, float(len(oct_recs))]
    for r in oct_recs:
        b += [float(v) for v in r]
    pt = pits or {}
    b += [float(pt.get('cell', 0.0)), float(pt.get('prob', 0.0)), float(pt.get('depth', 0.0)),
          float(pt.get('rfac', 0.4)), float(pt.get('seed', 1))]
    bd = bed or {}
    nb = np.asarray(bd.get('normal', (0.0, 1.0, 0.0)), float)
    nb = nb / max(np.linalg.norm(nb), 1e-12)
    b += list(map(float, nb)) + [float(bd.get('spacing', 0.01)), float(bd.get('amp', 0.0)),
                                 float(bd.get('warp', 0.0)), float(bd.get('warp_freq', 0.0))]
    b += [0.0, 0.0]                               # B (m) and g_max: set below
    b += [float(bd.get('passes', 1)), min(float(bd.get('jitter', 0.0)), 0.9),   # a claw chisel's passes, cut widths,
          min(max(float(bd.get('groove', 1.0)), 0.05), 1.0)]                    # the tooth's groove / the pitch
    if float(bd.get('blow', 0.0)) > 0.0:          # update 20: the blows (length, depth swing, sideways offset)
        b += [float(bd['blow']), float(bd.get('blow_depth', 0.55)), float(bd.get('blow_off', 0.12))]
    b[1] = float(len(b))
    p = 4 + len(oct_recs) * OCT
    b[p + 13] = float(bounds(b, 0.0)[0])          # with B = 0, bounds(.., 0) is the slope bound of r alone
    b[p + 12] = float(band)
    return b


def pack_chips(chips, oct_recs=(), crumble=None):
    """chips: [{centre, R (3x3 rows u, v, n), radii}]; crumble: {half, sag, A, lam, tau, seed} (the arris crumble about
    an as-fired box) or None; crumble['wear']: {r0, R, lam, var, seed} (weathering: the arrises worn round from the
    as-made round r0 by R on average, varying +-var along them at wavelength lam; 5 floats after the crumble's 8)"""
    b = [float(KIND_CHIPS), 0.0, float(len(chips)), float(len(oct_recs))]
    for r in oct_recs:
        b += [float(v) for v in r]
    for c in chips:
        b += list(map(float, c['centre'])) + list(map(float, np.asarray(c['R'], float).ravel())) + \
             list(map(float, c['radii']))
    if crumble:
        b += list(map(float, crumble['half'])) + [float(crumble['sag']), float(crumble['A']), float(crumble['lam']),
                                                  min(float(crumble['tau']), 0.9), float(crumble['seed'])]
        wr = crumble.get('wear')
        if wr and float(wr.get('R', 0.0)) > 0.0:
            b += [float(wr['r0']), float(wr['R']), float(wr['lam']), min(float(wr['var']), 0.9), float(wr['seed'])]
        if float(crumble.get('A2', 0.0)) > 0.0:   # a second bite octave: A2, lam2
            b += [float(crumble['A2']), float(crumble['lam2'])]
    b[1] = float(len(b))
    return b


def pack_crack(c, n, e, t, depth, w0, half_len=1e3, tip_amp=0.08, tip_freq=40.0, cap=0.15, seed=1, oct_recs=()):
    b = [float(KIND_CRACK), 0.0] + list(map(float, c)) + list(map(float, n)) + list(map(float, e)) + \
        list(map(float, t)) + [float(depth), float(w0), float(half_len), float(tip_amp), float(tip_freq), float(cap),
                               float(seed), float(len(oct_recs))]
    for r in oct_recs:
        b += [float(v) for v in r]
    b[1] = float(len(b))
    return b


def bounds(block, L_child):
    """(L, grow): the op's Lipschitz bound given the child's, and how far its box grows (m)"""
    kind = int(block[0])
    if kind == KIND_RELIEF:
        n = int(block[3])
        slope, amp = 0.0, 0.0
        for k in range(n):
            r = 4 + k * OCT
            slope += abs(block[r + 1]) * RAW_GMAX * block[r]
            amp += abs(block[r + 1]) * 1.0
        p = 4 + n * OCT
        if block[p] > 0.0:                         # a bowl depth d over radius rfac * cell: slope <= 1.54 d / R
            dmax = 1.5 * block[p + 2]
            rmin = 0.7 * block[p + 3] * block[p]
            slope += 1.54 * dmax / max(rmin, 1e-9) * 1.45 * 2.0   # aspect up to 1.45; the outline's wobble
        if block[p + 9] != 0.0:
            jw = block[p + 15] if len(block) > p + 15 else 0.0     # the narrowest cut: 1 - jitter of the pitch
            gw = min(block[p + 16], 1.0) if len(block) > p + 16 and block[p + 16] > 0.0 else 1.0   # x the groove
            bl = len(block) > p + 19 and block[p + 17] > 0.0   # update 20: the blows
            ph_mul = 1.0 + abs(block[p + 10]) * 5.0 * RAW_GMAX * block[p + 11] * block[p + 8]
            if bl:
                ph_mul += abs(block[p + 19]) * 3.71 * RAW_GMAX * block[p + 8] / block[p + 17]
            s_cut = abs(block[p + 9]) * 4.0 * 1.4 / block[p + 8] / (1.0 - min(jw, 0.9)) / gw * ph_mul
            if bl:                                  # depth x m: m <= 1.6; plus the cut's depth x |grad m|
                s_cut = s_cut * 1.6 + abs(block[p + 9]) * 1.4 * block[p + 18] * 3.71 * RAW_GMAX / block[p + 17]
            slope += s_cut
            amp += abs(block[p + 9]) * (1.6 if bl else 1.0)
        if len(block) > p + 13 and block[p + 12] > 0.0:
            # normalised (relief_value): ~1 near the surface, 1 / (1 + g_max) x (1 + g) <= 1 beyond; 1.25 covers the
            # second-order term inside the band (measured <= 1.2: tests T2)
            return max(L_child, 1.0) * 1.25, amp
        return L_child + slope, amp
    if kind == KIND_CHIPS:
        n = int(block[3])
        slope = sum(abs(block[4 + k * OCT + 1]) * RAW_GMAX * block[4 + k * OCT] for k in range(n))
        ce = 4 + n * OCT + int(block[2]) * CHIP
        if len(block) >= ce + 8:                   # the crumble's bite: |grad w| <= A 3.71 RAW_GMAX / lam / (1 - tau)
            cr = block[ce + 4] * 3.71 * RAW_GMAX / block[ce + 5] / (1.0 - block[ce + 6])
            nx = len(block) - ce - 8
            if nx == 2 or nx == 7:                 # the second bite octave's slope
                cr += block[ce + nx + 6] * 3.71 * RAW_GMAX / block[ce + nx + 7]
            if nx == 5 or nx == 7:                 # the worn round: |grad rw| <= R var 3.71 RAW_GMAX / lam; the worn
                gr = block[ce + 9] * block[ce + 11] * 3.71 * RAW_GMAX / block[ce + 10]   # box moves <= (sqrt 3 - 1)
                cr = max(cr + 0.41421356 * gr, 0.7320508 * gr)                       # x that, the crumble's tube (sqrt 2 - 1) x
            slope += cr
        return max(L_child, 1.0 + slope), 0.0
    if kind == KIND_CRACK:
        n = int(block[21])
        slope = sum(abs(block[22 + k * OCT + 1]) * RAW_GMAX * block[22 + k * OCT] for k in range(n))
        tip = block[14] * block[17] * 5.0 * RAW_GMAX * block[18]
        return max(L_child, 1.0 + slope + block[19], 1.0 + tip), 0.0
    return L_child, 0.0


def measure_noise(n=200000, seed=3):
    """the constants RAW_SD and RAW_GMAX, measured (run once; recorded above)"""
    rng = np.random.default_rng(seed)
    P = rng.uniform(-50, 50, (n, 3))
    v = np.array([noise3(p[0], p[1], p[2], 17) for p in P])
    e = 1e-4
    g = np.array([math.sqrt(sum(((noise3(*(p + e * np.eye(3)[a]), 17) - noise3(*(p - e * np.eye(3)[a]), 17)) / (2 * e)) ** 2
                                for a in range(3))) for p in P[:20000]])
    return float(v.std()), float(g.max())
