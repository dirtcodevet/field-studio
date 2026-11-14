"""Light through the hair mass (fs-hair, 11-11): each point's share of the key light after the hair between it and the
light, so the under-layers are darker than the top, as in real hair. Built from what the hair is, not painted.

The mass: every strand point splatted into a 2 mm lattice as fibre length; the projected fibre area per volume is
a = 2 r l / V (m^-1). A fibre crossing passes (1 - opacity) of the light (hair3 measured ~75 % through one blond
fibre; darker hair passes less: exp(-2.4 sigma) per crossing from the strand's own pigment). The extinction is then
sigma_t = a x opacity, and the key's share at a point exp(-integral of sigma_t) along the ray toward the light,
marched through the lattice (up to 60 mm). A mass of 10 % fibre by volume at 70 um gives a mean free path of ~2 mm.

The share is computed once per hair version and light (numba, CPU): about 0.3 s for 200k strands x 32 points."""
import math

import numpy as np
import numba as nb


@nb.njit(nogil=True, cache=True)
def _splat(P, RAD, SEG, lo, h, nx, ny, nz, A):
    n, k = P.shape[0], P.shape[1]
    for s in range(n):
        for i in range(k - 1):
            x = P[s, i, 0]
            if x != x or P[s, i + 1, 0] != P[s, i + 1, 0]:
                break
            cx = 0.5 * (P[s, i, 0] + P[s, i + 1, 0])
            cy = 0.5 * (P[s, i, 1] + P[s, i + 1, 1])
            cz = 0.5 * (P[s, i, 2] + P[s, i + 1, 2])
            ix = int((cx - lo[0]) / h)
            iy = int((cy - lo[1]) / h)
            iz = int((cz - lo[2]) / h)
            if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
                continue
            A[ix, iy, iz] += 2.0 * RAD[s, i] * SEG[s, i]


@nb.njit(nogil=True, cache=True, fastmath=True)
def _march(P, A, lo, h, L, opac, maxd, out):
    n, k = P.shape[0], P.shape[1]
    nx, ny, nz = A.shape
    step = h * 0.5
    ns = int(maxd / step)
    for s in nb.prange(n):
        for i in range(k):
            x0 = P[s, i, 0]
            if x0 != x0:
                out[s, i] = 1.0
                continue
            y0, z0 = P[s, i, 1], P[s, i, 2]
            tau = 0.0
            for j in range(1, ns + 1):
                d = j * step + h * 0.5          # start past the point's own voxel half
                x = x0 + L[0] * d
                y = y0 + L[1] * d
                z = z0 + L[2] * d
                ix = int((x - lo[0]) / h)
                iy = int((y - lo[1]) / h)
                iz = int((z - lo[2]) / h)
                if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
                    break
                tau += A[ix, iy, iz] * step
                if tau > 8.0:
                    break
            out[s, i] = math.exp(-tau * opac[s])


@nb.njit(nogil=True, cache=True, fastmath=True)
def _march_out(P, A, lo, h, ctr, opac, maxd, out):
    """as _march, each point toward its own outward direction from ctr (where the open sky is: the head's centre)"""
    n, k = P.shape[0], P.shape[1]
    nx, ny, nz = A.shape
    step = h * 0.5
    ns = int(maxd / step)
    for s in range(n):
        for i in range(k):
            x0 = P[s, i, 0]
            if x0 != x0:
                out[s, i] = 1.0
                continue
            y0, z0 = P[s, i, 1], P[s, i, 2]
            ox, oy, oz = x0 - ctr[0], y0 - ctr[1], z0 - ctr[2]
            on = math.sqrt(ox * ox + oy * oy + oz * oz)
            if on < 1e-9:
                out[s, i] = 1.0
                continue
            ox, oy, oz = ox / on, oy / on, oz / on
            tau = 0.0
            for j in range(1, ns + 1):
                d = j * step + h * 0.5
                ix = int((x0 + ox * d - lo[0]) / h)
                iy = int((y0 + oy * d - lo[1]) / h)
                iz = int((z0 + oz * d - lo[2]) / h)
                if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
                    break
                tau += A[ix, iy, iz] * step
                if tau > 8.0:
                    break
            out[s, i] = math.exp(-tau * opac[s])


def _mass(P, RAD, SEG, SIG, h):
    # lean (fs-hair 11-12, to save memory): the bounds by reduction (no copy of every finite point, 106 MB for
    # her), the lattice and the shares in float32 (her job's transient ~306 -> ~100 MB)
    P = np.ascontiguousarray(P, np.float32)
    Q = P.reshape(-1, 3)
    with np.errstate(all='ignore'):
        lo = np.nanmin(Q, 0).astype(np.float64)
        hi = np.nanmax(Q, 0).astype(np.float64)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return P, None, None, None, None
    lo = lo - h
    hi = hi + h
    nx, ny, nz = (np.ceil((hi - lo) / h).astype(int) + 1)
    A = np.zeros((nx, ny, nz), np.float32)
    _splat(P, np.ascontiguousarray(RAD, np.float32), np.ascontiguousarray(SEG, np.float32), lo, h, nx, ny, nz, A)
    A /= h ** 3
    if SIG is not None:
        absorbed = 1.0 - np.exp(-1.5708 * np.asarray(SIG, np.float64).mean(1))
        opac = (0.5 * absorbed + 0.04).clip(0.05, 0.9)
    else:
        opac = np.full(len(P), 0.2)
    return P, A, lo, opac, True


def shares(P, RAD, SEG, light_dir, centre=None, SIG=None, h=0.002, maxd=0.06):
    """-> (key, sky), each (n, k) float32 in (0, 1]: the key's transmittance through the hair toward the light, and the
    sky's toward each point's outward direction from centre (ones when centre is None); one splat for both"""
    P, A, lo, opac, good = _mass(P, RAD, SEG, SIG, h)
    if A is None:
        o = np.ones(P.shape[:2], np.float32)
        return o, o.copy()
    Ld = np.asarray(light_dir, np.float64)
    Ld = Ld / max(np.linalg.norm(Ld), 1e-12)
    key = np.empty(P.shape[:2], np.float32)
    _march(P, A, lo, h, Ld, opac, float(maxd), key)
    sky = np.ones(P.shape[:2], np.float32)
    if centre is not None:
        _march_out(P, A, lo, h, np.asarray(centre, np.float64), opac, float(maxd), sky)
    return key, sky


def light_share(P, RAD, SEG, light_dir, SIG=None, h=0.002, maxd=0.06):
    """-> (n, k) float32 in (0, 1]: the key's share at each point after the hair toward the light"""
    P = np.ascontiguousarray(P, np.float32)
    good = P[np.isfinite(P[..., 0])]
    if not len(good):
        return np.ones(P.shape[:2], np.float32)
    lo = good.min(0).astype(np.float64) - h
    hi = good.max(0).astype(np.float64) + h
    nx, ny, nz = (np.ceil((hi - lo) / h).astype(int) + 1)
    A = np.zeros((nx, ny, nz), np.float64)
    _splat(P, np.ascontiguousarray(RAD, np.float32), np.ascontiguousarray(SEG, np.float32), lo, h, nx, ny, nz, A)
    A /= h ** 3                                          # projected fibre area per volume (m^-1)
    Ld = np.asarray(light_dir, np.float64)
    Ld = Ld / max(np.linalg.norm(Ld), 1e-12)
    if SIG is not None:
        # what a crossing takes from the key's direction: the light absorbed along the mean chord (pi/2 radii; sigma is
        # per unit radius), halved because a light fibre sends most of what passes on forward (the TT lobe, ~10-20 deg)
        # rather than out of the beam (Zinke 2008's forward-scattering transmittance)
        absorbed = 1.0 - np.exp(-1.5708 * np.asarray(SIG, np.float64).mean(1))
        opac = (0.5 * absorbed + 0.04).clip(0.05, 0.9)
    else:
        opac = np.full(len(P), 0.2)
    out = np.empty(P.shape[:2], np.float64)
    _march(P, A, lo, h, Ld, opac, float(maxd), out)
    return out.astype(np.float32)


def shade_floor(SIG):
    """the light a point keeps however deep it lies: what scatters back out of the hair beneath (multiple scattering,
    HAIR_DEFINED 3: the bulk of light hair's brightness), larger for lighter fibres: (n,) in 0.2..0.55"""
    if SIG is None:
        return None
    alb = np.exp(-1.5708 * np.asarray(SIG, np.float64).mean(1))
    return (0.2 + 0.35 * alb).astype(np.float32)
