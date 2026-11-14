"""The one warp engine's flows (TOOLSET_DESIGN section 5 item 3): smooth one-to-one warps as flows of velocity fields.

Every move tool (Shape, Transform incl. a region, Place on surface, Draw out; later posing) is the time-1 flow phi of
    v(x, t) = W(M(t)^-1 x) * (A(t) (x - p(t)) + tau(t))        [ + grad W x Psi   when Keep volume is on ]
with
- W the tool's weight, read in the BODY frame y = M(t)^-1 x: the support travels with the motion it drives (an
  anchored Shape's lens follows the hand, a region's soft edge rides with the region), so a point of weight 1 moves
  exactly with the motion: the pressed point follows the cursor exactly at Strength 1, a full-weight region moves
  rigidly, and a weight-0 point never moves;
- M(t) the motion so far (a translation for Shape, a rigid or affine motion for Transform, a path for Draw out);
- (A, tau, p) its generator (A = 0 for a translation; A skew for a rotation).

The flow of a Lipschitz field is a diffeomorphism: it cannot fold, tear or self-cross at any pull (the naive per-point
displacement folds once the pull passes about two-thirds of the radius: measured 11-12). It is integrated with RK4 in
N steps; the inverse (what the field needs: F'(x) = F(phi^-1(x))) is the same ODE integrated backward.

Weights (FP_METRIC):
- ball (0): the profile of |y - c0| (warp/profile.py);
- lens (1): the profile of the distance ACROSS the lens axis n, times a DEPTH gate from the lens's depth images taken
  at the press (warp/lens.py): 1 in front of the ramp start zs (the caught surface for Near side, the part's far face for
  Whole part), falling smoothly to 0 at the ramp end ze (world-fixed: the material behind it never moves; a push
  compresses the ramp and stops at it, the minimum wall); Through: no gate;
- grid (2): a sampled weight (a region from the mask, its soft edge = Smoothing).

Keep volume (FP_KEEPVOL, rigid generators only): v = curl(W Psi) with curl Psi = the generator,
Psi = tau x (x - p) / 2 - |x - p|^2 omega / 2. A curl has no divergence, so the flow keeps volume exactly (up to the
integration error). Patent route (LIBRARIES_AND_PATENTS 6.2): a vector-potential flow (von Funck, Theisel, Seidel
2006), never an elastic Kelvinlet.

Stop at contact (FP_CONTACT): the velocity of a point that APPROACHES another surface is scaled by a smooth ramp of
its distance to that surface (a sampled distance grid of the contact objects, minus the Gap): it slows to rest at the
surface and never crosses it; the rest keeps moving ("that point in your surface stops and the rest continues"). A
point moving away is never held. Patent route: stop on the other object's own field, no displayed offset shell
(Disney US 9,483,878).

All kernels are deterministic (no atomics, no random numbers): the same stroke gives the same field on any machine.
"""
import math

import numpy as np
from numba import njit, prange

from .profile import weight as _wprof

# ---- the flow's parameter block (float64) -------------------------------------------------------------------------
FP_METRIC, FP_R, FP_HARD, FP_SMOOTH, FP_FALL, FP_FAMILY = 0, 1, 2, 3, 4, 5
FP_C0, FP_N, FP_E1, FP_E2 = 6, 9, 12, 15
FP_REACH, FP_EPSMIN, FP_KEEPVOL = 18, 19, 20
FP_CONTACT, FP_CEPS, FP_CGAP = 21, 22, 23
FP_IMG_PIX, FP_IMG_W, FP_IMG_H, FP_IMG_U0, FP_IMG_V0 = 24, 25, 26, 27, 28
FP_G_O, FP_G_H = 29, 32
FP_C_O, FP_C_H = 33, 36
FP_NSTEP, FP_GRADEPS = 37, 38
FP_TRAVEL = 39           # 1: Whole part with Stop at contact: each column moves min(hand, its free travel) (img[2])
FP_N_ = 40

METRIC = {'ball': 0, 'lens': 1, 'grid': 2}
REACH = {'none': -1, 'near': 0, 'whole': 1, 'through': 2}

# ---- a stage row: the motion and its generator at one RK time (3 rows per step: its start, middle and end, each with
# that step's generator, so a path's piecewise velocity is exact at segment joins) -------------------------------------
ST_MINV, ST_A, ST_TAU, ST_P, ST_AX, ST_T = 0, 12, 21, 24, 27, 28
ST_N = 29


@njit(cache=True, inline='always')
def _tri(G, ox, oy, oz, h, x, y, z, outside):
    """trilinear sample of a dense grid at world (x, y, z); `outside` beyond it"""
    fx = (x - ox) / h
    fy = (y - oy) / h
    fz = (z - oz) / h
    nx, ny, nz = G.shape
    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx > nx - 1 or fy > ny - 1 or fz > nz - 1:
        return outside
    i = min(int(fx), nx - 2)
    j = min(int(fy), ny - 2)
    k = min(int(fz), nz - 2)
    tx = fx - i
    ty = fy - j
    tz = fz - k
    a = G[i, j, k] + (G[i + 1, j, k] - G[i, j, k]) * tx
    b = G[i, j + 1, k] + (G[i + 1, j + 1, k] - G[i, j + 1, k]) * tx
    c = G[i, j, k + 1] + (G[i + 1, j, k + 1] - G[i, j, k + 1]) * tx
    d = G[i, j + 1, k + 1] + (G[i + 1, j + 1, k + 1] - G[i, j + 1, k + 1]) * tx
    e = a + (b - a) * ty
    f = c + (d - c) * ty
    return e + (f - e) * tz


@njit(cache=True, inline='always')
def _img(img, c, fp, u1, u2):
    """bilinear sample of depth image c at lateral lens coords (u1, u2), clamped to the image"""
    W = int(fp[FP_IMG_W])
    H = int(fp[FP_IMG_H])
    px = (u1 - fp[FP_IMG_U0]) / fp[FP_IMG_PIX]
    py = (u2 - fp[FP_IMG_V0]) / fp[FP_IMG_PIX]
    px = min(max(px, 0.0), W - 1.000001)
    py = min(max(py, 0.0), H - 1.000001)
    i = int(px)
    j = int(py)
    tx = px - i
    ty = py - j
    a = img[c, j, i] + (img[c, j, i + 1] - img[c, j, i]) * tx
    b = img[c, j + 1, i] + (img[c, j + 1, i + 1] - img[c, j + 1, i]) * tx
    return a + (b - a) * ty


@njit(cache=True, inline='always')
def _ramp(t):
    """1 at t <= 0, 0 at t >= 1, C1 (smoothstep)"""
    if t <= 0.0:
        return 1.0
    if t >= 1.0:
        return 0.0
    return 1.0 - t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline='always')
def _travel(fp, img, st, s, x, y, z):
    """Whole part with Stop at contact: (the column's displacement so far along the motion, its rate per unit of hand
    travel, the hand's speed): a smooth min of the hand's travel and the column's free travel measured at the press
    (img[2]), so a column stops exactly where its far face meets the surface below, front and back together"""
    tx = st[s, ST_TAU]
    ty = st[s, ST_TAU + 1]
    tz = st[s, ST_TAU + 2]
    tl = math.sqrt(tx * tx + ty * ty + tz * tz)
    a = st[s, ST_T] * tl
    ex = x - fp[FP_C0]
    ey = y - fp[FP_C0 + 1]
    ez = z - fp[FP_C0 + 2]
    u1 = ex * fp[FP_E1] + ey * fp[FP_E1 + 1] + ez * fp[FP_E1 + 2]
    u2 = ex * fp[FP_E2] + ey * fp[FP_E2 + 1] + ez * fp[FP_E2 + 2]
    b = _img(img, 2, fp, u1, u2)
    k = fp[FP_EPSMIN]
    hh = max(k - abs(a - b), 0.0) / k
    if a <= b:
        return min(a, b) - hh * hh * k * 0.25, 1.0 - 0.5 * hh, tl
    return min(a, b) - hh * hh * k * 0.25, 0.5 * hh, tl


@njit(cache=True)
def body_weight(fp, img, G, st, s, x, y, z):
    """W at world point (x, y, z) at stage s (read in the body frame)"""
    if fp[FP_TRAVEL] > 0.5:
        sd, _, tl = _travel(fp, img, st, s, x, y, z)
        f = sd / tl if tl > 0 else 0.0
        yx = x - f * st[s, ST_TAU]
        yy = y - f * st[s, ST_TAU + 1]
        yz = z - f * st[s, ST_TAU + 2]
        ax = f * (st[s, ST_TAU] * fp[FP_N] + st[s, ST_TAU + 1] * fp[FP_N + 1] + st[s, ST_TAU + 2] * fp[FP_N + 2])
    else:
        yx = st[s, 0] * x + st[s, 1] * y + st[s, 2] * z + st[s, 3]
        yy = st[s, 4] * x + st[s, 5] * y + st[s, 6] * z + st[s, 7]
        yz = st[s, 8] * x + st[s, 9] * y + st[s, 10] * z + st[s, 11]
        ax = st[s, ST_AX]
    m = int(fp[FP_METRIC])
    r = fp[FP_R]
    if m == 2:
        return _tri(G, fp[FP_G_O], fp[FP_G_O + 1], fp[FP_G_O + 2], fp[FP_G_H], yx, yy, yz, 0.0)
    ex = yx - fp[FP_C0]
    ey = yy - fp[FP_C0 + 1]
    ez = yz - fp[FP_C0 + 2]
    if m == 0:
        d = math.sqrt(ex * ex + ey * ey + ez * ez)
        return _wprof(d, r, fp[FP_HARD], fp[FP_SMOOTH], int(fp[FP_FALL]), int(fp[FP_FAMILY]))
    u1 = ex * fp[FP_E1] + ey * fp[FP_E1 + 1] + ez * fp[FP_E1 + 2]
    u2 = ex * fp[FP_E2] + ey * fp[FP_E2 + 1] + ez * fp[FP_E2 + 2]
    d = math.sqrt(u1 * u1 + u2 * u2)
    w = _wprof(d, r, fp[FP_HARD], fp[FP_SMOOTH], int(fp[FP_FALL]), int(fp[FP_FAMILY]))
    if w <= 0.0:
        return 0.0
    reach = int(fp[FP_REACH])
    if reach < 0 or reach == 2:
        return w
    zb = ex * fp[FP_N] + ey * fp[FP_N + 1] + ez * fp[FP_N + 2]
    # the ramp end ze is fixed in the world (what lies behind it never moves): in the body frame it moves back by the
    # motion along n so far. The ramp start zs (the caught surface, or the part's far face) rides with the body, but
    # never closer than eps to the end: a push that reaches the end stops there (the minimum wall), it never carries it
    ze = _img(img, 1, fp, u1, u2) - ax
    zs = min(_img(img, 0, fp, u1, u2), ze - fp[FP_EPSMIN])
    if zb <= zs:
        return w
    return w * _ramp((zb - zs) / (ze - zs))


@njit(cache=True, inline='always')
def _column_point(fp, img, st, s, x, y, z):
    """for a lens with Reach Whole part: the far face of the column through (x, y, z) (where the gate is read); the
    point itself otherwise, or when it lies behind the far face"""
    if int(fp[FP_METRIC]) != 1 or int(fp[FP_REACH]) != 1:
        return x, y, z
    yx = st[s, 0] * x + st[s, 1] * y + st[s, 2] * z + st[s, 3]
    yy = st[s, 4] * x + st[s, 5] * y + st[s, 6] * z + st[s, 7]
    yz = st[s, 8] * x + st[s, 9] * y + st[s, 10] * z + st[s, 11]
    ex = yx - fp[FP_C0]
    ey = yy - fp[FP_C0 + 1]
    ez = yz - fp[FP_C0 + 2]
    u1 = ex * fp[FP_E1] + ey * fp[FP_E1 + 1] + ez * fp[FP_E1 + 2]
    u2 = ex * fp[FP_E2] + ey * fp[FP_E2 + 1] + ez * fp[FP_E2 + 2]
    zb = ex * fp[FP_N] + ey * fp[FP_N + 1] + ez * fp[FP_N + 2]
    zs = _img(img, 0, fp, u1, u2)
    if zb >= zs:
        return x, y, z
    dz = zs - zb
    return x + dz * fp[FP_N], y + dz * fp[FP_N + 1], z + dz * fp[FP_N + 2]


@njit(cache=True)
def velocity(fp, img, G, CG, st, s, x, y, z, out):
    W = body_weight(fp, img, G, st, s, x, y, z)
    keep = fp[FP_KEEPVOL] > 0.5
    if W <= 0.0 and not keep:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        return
    px = x - st[s, ST_P]
    py = y - st[s, ST_P + 1]
    pz = z - st[s, ST_P + 2]
    v0x = st[s, ST_A] * px + st[s, ST_A + 1] * py + st[s, ST_A + 2] * pz + st[s, ST_TAU]
    v0y = st[s, ST_A + 3] * px + st[s, ST_A + 4] * py + st[s, ST_A + 5] * pz + st[s, ST_TAU + 1]
    v0z = st[s, ST_A + 6] * px + st[s, ST_A + 7] * py + st[s, ST_A + 8] * pz + st[s, ST_TAU + 2]
    travel = fp[FP_TRAVEL] > 0.5
    if travel:
        _, dsda, _ = _travel(fp, img, st, s, x, y, z)
        W = W * dsda
    vx = W * v0x
    vy = W * v0y
    vz = W * v0z
    if keep:
        e = fp[FP_GRADEPS]
        gx = (body_weight(fp, img, G, st, s, x + e, y, z) - body_weight(fp, img, G, st, s, x - e, y, z)) / (2 * e)
        gy = (body_weight(fp, img, G, st, s, x, y + e, z) - body_weight(fp, img, G, st, s, x, y - e, z)) / (2 * e)
        gz = (body_weight(fp, img, G, st, s, x, y, z + e) - body_weight(fp, img, G, st, s, x, y, z - e)) / (2 * e)
        if gx != 0.0 or gy != 0.0 or gz != 0.0:
            tx = st[s, ST_TAU]
            ty = st[s, ST_TAU + 1]
            tz = st[s, ST_TAU + 2]
            wx = st[s, ST_A + 7]        # omega from the skew part of A: A = [omega]x
            wy = st[s, ST_A + 2]
            wz = st[s, ST_A + 3]
            q2 = px * px + py * py + pz * pz
            sx = 0.5 * (ty * pz - tz * py) - 0.5 * q2 * wx
            sy = 0.5 * (tz * px - tx * pz) - 0.5 * q2 * wy
            sz = 0.5 * (tx * py - ty * px) - 0.5 * q2 * wz
            vx += gy * sz - gz * sy
            vy += gz * sx - gx * sz
            vz += gx * sy - gy * sx
    if fp[FP_CONTACT] > 0.5 and not travel and (vx != 0.0 or vy != 0.0 or vz != 0.0):
        h = fp[FP_C_H]
        ox = fp[FP_C_O]
        oy = fp[FP_C_O + 1]
        oz = fp[FP_C_O + 2]
        # Whole part: the part moves as one through its thickness, so a column stops when its FAR FACE touches: the
        # gate is read at the column's far face (its depth image) and shared by every point of the column in front of
        # it (per point, the near face kept moving and crushed the part onto what it met: fs-tools t1/f1, 11-14)
        x, y, z = _column_point(fp, img, st, s, x, y, z)
        d = _tri(CG, ox, oy, oz, h, x, y, z, 1e9) - fp[FP_CGAP]
        ce = fp[FP_CEPS]
        if d < ce:
            nx = _tri(CG, ox, oy, oz, h, x + 0.5 * h, y, z, 1e9) - _tri(CG, ox, oy, oz, h, x - 0.5 * h, y, z, 1e9)
            ny = _tri(CG, ox, oy, oz, h, x, y + 0.5 * h, z, 1e9) - _tri(CG, ox, oy, oz, h, x, y - 0.5 * h, z, 1e9)
            nz = _tri(CG, ox, oy, oz, h, x, y, z + 0.5 * h, 1e9) - _tri(CG, ox, oy, oz, h, x, y, z - 0.5 * h, 1e9)
            nl = math.sqrt(nx * nx + ny * ny + nz * nz)
            vl = math.sqrt(vx * vx + vy * vy + vz * vz)
            if nl < 1e-300 or abs(nx) > 1e8 or abs(ny) > 1e8 or abs(nz) > 1e8:
                app = 1.0                          # at the grid's edge: treat as approaching (never crosses)
            else:
                app = min(max(-4.0 * (vx * nx + vy * ny + vz * nz) / (vl * nl), 0.0), 1.0)
            f = 1.0 - _ramp(max(d, 0.0) / ce)      # 0 at contact, 1 at ce and beyond
            k = 1.0 - (1.0 - f) * app
            vx *= k
            vy *= k
            vz *= k
    out[0] = vx
    out[1] = vy
    out[2] = vz


@njit(cache=True)
def _rk4(fp, img, G, CG, st, s0, s1, s2, x, y, z, hstep, buf):
    """one RK4 step from stage s0 to s2 (s1 the midpoint) with step hstep (negative = backward)"""
    k1 = buf[0]
    k2 = buf[1]
    k3 = buf[2]
    k4 = buf[3]
    velocity(fp, img, G, CG, st, s0, x, y, z, k1)
    velocity(fp, img, G, CG, st, s1, x + 0.5 * hstep * k1[0], y + 0.5 * hstep * k1[1], z + 0.5 * hstep * k1[2], k2)
    velocity(fp, img, G, CG, st, s1, x + 0.5 * hstep * k2[0], y + 0.5 * hstep * k2[1], z + 0.5 * hstep * k2[2], k3)
    velocity(fp, img, G, CG, st, s2, x + hstep * k3[0], y + hstep * k3[1], z + hstep * k3[2], k4)
    c = hstep / 6.0
    return (x + c * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), y + c * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            z + c * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))


@njit(cache=True)
def map_point(fp, img, G, CG, st, x, y, z, inverse, buf):
    n = int(fp[FP_NSTEP])
    hs = 1.0 / n
    if not inverse:
        for k in range(n):
            x, y, z = _rk4(fp, img, G, CG, st, 3 * k, 3 * k + 1, 3 * k + 2, x, y, z, hs, buf)
    else:
        for k in range(n - 1, -1, -1):
            x, y, z = _rk4(fp, img, G, CG, st, 3 * k + 2, 3 * k + 1, 3 * k, x, y, z, -hs, buf)
    return x, y, z


@njit(cache=True, parallel=True)
def map_points(fp, img, G, CG, st, pts, inverse, box, out):
    """phi (inverse False) or phi^-1 (True) of world points; points outside `box` (6,: the motion's swept support,
    lo then hi) are unmoved, exactly"""
    for i in prange(pts.shape[0]):
        x = pts[i, 0]
        y = pts[i, 1]
        z = pts[i, 2]
        if x < box[0] or y < box[1] or z < box[2] or x > box[3] or y > box[4] or z > box[5]:
            out[i, 0] = x
            out[i, 1] = y
            out[i, 2] = z
            continue
        buf = np.empty((4, 3))
        a, b, c = map_point(fp, img, G, CG, st, x, y, z, inverse, buf)
        out[i, 0] = a
        out[i, 1] = b
        out[i, 2] = c


@njit(cache=True, parallel=True)
def jacobian_det(fp, img, G, CG, st, pts, eps, inverse, out):
    """det of the map's Jacobian at points (central differences): > 0 everywhere = one-to-one and orientation kept"""
    for i in prange(pts.shape[0]):
        buf = np.empty((4, 3))
        J = np.empty((3, 3))
        for a in range(3):
            xp = pts[i, 0] + (eps if a == 0 else 0.0)
            yp = pts[i, 1] + (eps if a == 1 else 0.0)
            zp = pts[i, 2] + (eps if a == 2 else 0.0)
            xm = pts[i, 0] - (eps if a == 0 else 0.0)
            ym = pts[i, 1] - (eps if a == 1 else 0.0)
            zm = pts[i, 2] - (eps if a == 2 else 0.0)
            p1 = map_point(fp, img, G, CG, st, xp, yp, zp, inverse, buf)
            p0 = map_point(fp, img, G, CG, st, xm, ym, zm, inverse, buf)
            for b in range(3):
                J[b, a] = (p1[b] - p0[b]) / (2 * eps)
        out[i] = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
                  + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
