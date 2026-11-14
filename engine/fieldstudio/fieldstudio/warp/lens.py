"""What a lens sees (TOOLSET_DESIGN section 5 item 5): for a flat tool, the surfaces along each lens ray inside its
column, read ONCE from the field as it was at the press (1.6, "the form at the press").

A lens is a disc of radius R (Radius + Smoothing) centred on the press point, facing the way the tool looks (n, INTO
the surface). Along each of its rays, from in front of the surface:
    zf    the first surface (the caught surface)
    zfar  where the ray leaves that part again (its far face: the part's thickness along n is zfar - zf, exact)
    znext where it enters the next surface behind (across a gap), or +inf
The depth gate of warp/flow.py reads two images built from these by Reach:
    Near side   zs = zf,   ze = zf + clamp((zfar - zf) / 2, eps, Radius + Smoothing), never past zfar - wall
    Whole part  zs = zfar, ze = znext (or zfar + Radius + Smoothing when nothing is behind)
    Through     no gate
so a Near-side move never reaches the far half of a part or the surface across a gap, and a Whole-part move carries
the part's front and back together, at any thickness, and stops short of what lies behind it.

Rays that miss the form take the nearest hit ray's values (the silhouette's), and both images are smoothed over a few
pixels, so the gate is a smooth function across the lens (no tear at a rim: the sides stretch and stay attached).
"""
import numpy as np


def lens_frame(n):
    """(n, e1, e2): n unit, e1 / e2 an orthonormal pair across it (deterministic)"""
    n = np.asarray(n, float)
    n = n / max(np.linalg.norm(n), 1e-300)
    a = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


def march(field, origins, dirs, tmax, L=1.0, hmin=2e-4, hmax=None, max_steps=4000):
    """sphere-trace rays through a field (field(P) -> values, world units, |grad| <= L) and return, per ray, the first
    three sign changes along it: t_in (outside -> inside), t_out (inside -> outside), t_in2 (outside -> inside again);
    np.inf where a crossing doesn't happen before tmax. Each crossing is refined by bisection to < hmin / 64."""
    n = len(origins)
    res = np.full((n, 3), np.inf)
    t = np.zeros(n)
    f = field(origins)
    sgn = (f < 0).astype(int)                      # 1 inside at the ray's start
    stage = np.where(sgn == 1, 1, 0)               # 0: looking for t_in; 1: for t_out; 2: for t_in2; 3: done
    if (sgn == 1).any():
        res[sgn == 1, 0] = 0.0                     # a ray starting inside: its first surface is at its start
    hmax = tmax if hmax is None else hmax
    act = np.ones(n, bool)
    for _ in range(max_steps):
        act &= (stage < 3) & (t < tmax)
        if not act.any():
            break
        idx = np.nonzero(act)[0]
        step = np.clip(np.abs(f[idx]) / L, hmin, hmax)
        t1 = np.minimum(t[idx] + step, tmax)
        p1 = origins[idx] + dirs[idx] * t1[:, None]
        f1 = field(p1)
        crossed = (f1 < 0) != (f[idx] < 0)
        if crossed.any():
            ci = idx[crossed]
            a = t[ci].copy()
            b = t1[crossed].copy()
            fa = f[ci].copy()
            for _ in range(24):
                m = 0.5 * (a + b)
                fm = field(origins[ci] + dirs[ci] * m[:, None])
                same = (fm < 0) == (fa < 0)
                a = np.where(same, m, a)
                fa = np.where(same, fm, fa)
                b = np.where(same, b, m)
            tc = 0.5 * (a + b)
            st = stage[ci]
            res[ci, np.minimum(st, 2)] = tc
            stage[ci] = st + 1
        t[idx] = t1
        f[idx] = f1
    return res


def _fill_misses(img, hit):
    """nearest-hit fill (deterministic, by growing rings), so misses take the silhouette's values"""
    out = img.copy()
    have = hit.copy()
    if not have.any():
        return out, have
    H, W = img.shape
    while not have.all():
        grown = False
        new = out.copy()
        newh = have.copy()
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            sh = np.roll(np.roll(have, dy, 0), dx, 1)
            sv = np.roll(np.roll(out, dy, 0), dx, 1)
            # no wrap-around
            if dy == -1:
                sh[-1, :] = False
            if dy == 1:
                sh[0, :] = False
            if dx == -1:
                sh[:, -1] = False
            if dx == 1:
                sh[:, 0] = False
            acc += np.where(sh, sv, 0.0)
            cnt += sh
        take = (~have) & (cnt > 0)
        if take.any():
            new[take] = acc[take] / cnt[take]
            newh[take] = True
            grown = True
        out, have = new, newh
        if not grown:
            break
    return out, have


def _blur(img, sigma_px):
    if sigma_px <= 0.25:
        return img
    rad = int(np.ceil(2.5 * sigma_px))
    x = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    k /= k.sum()
    p = np.pad(img, rad, mode='edge')
    p = np.apply_along_axis(lambda v: np.convolve(v, k, mode='valid'), 0, p)
    p = np.apply_along_axis(lambda v: np.convolve(v, k, mode='valid'), 1, p)
    return p


def depth_images(field, c0, n, half, res, reach, radius_total, L=1.0, eps=None, wall=None, back=None, hmin=None):
    """The lens's two gate images (zs, ze) on a res x res grid of lateral offsets in [-half, half] (along e1, e2),
    depths measured along n from c0. field(P) -> world values of the edited object(s) at the press. reach: 'near' |
    'whole'. back: how far in front of c0 the rays start (default 2 x half). -> dict with the frame, the images and the
    raw crossings (for the cue: the far face that will move, its distance)."""
    n, e1, e2 = lens_frame(n)
    c0 = np.asarray(c0, float)
    back = 2.0 * half if back is None else float(back)
    eps = max(1e-4, 0.05 * radius_total) if eps is None else float(eps)
    wall = eps if wall is None else float(wall)
    hmin = max(half / 256.0, 5e-5) if hmin is None else float(hmin)
    u = np.linspace(-half, half, res)
    U1, U2 = np.meshgrid(u, u, indexing='xy')                 # [row j, col i]: u2 along rows, u1 along columns
    org = c0[None, :] + U1.reshape(-1, 1) * e1 + U2.reshape(-1, 1) * e2 - back * n
    dirs = np.repeat(n[None, :], len(org), 0)
    tmax = back + 8.0 * half + 1.0 * radius_total
    cr = march(field, org, dirs, tmax, L=L, hmin=hmin) - back  # depths along n from the lens plane through c0
    zf, zfar, znext = cr[:, 0].reshape(res, res), cr[:, 1].reshape(res, res), cr[:, 2].reshape(res, res)
    hit = np.isfinite(zf)
    zfar = np.where(hit & ~np.isfinite(zfar), zf + eps, zfar)
    with np.errstate(invalid='ignore'):
        if reach == 'whole':
            # the part's front and back move together; the ramp runs across the gap behind it to the next surface,
            # which never moves (a push carries the part up to it: Stop at contact or the gap holds it there)
            zs = zfar
            ze = np.where(np.isfinite(znext), np.maximum(znext, zfar + eps), zfar + radius_total)
        else:
            zs = zf
            thick = zfar - zf
            ze = zf + np.clip(0.5 * thick, eps, radius_total)
            ze = np.minimum(ze, np.maximum(zfar - wall, zf + eps))
    zs_f, have = _fill_misses(np.where(hit, zs, 0.0), hit)
    ze_f, _ = _fill_misses(np.where(hit, ze, 0.0), hit)
    if not have.any():                                         # nothing under the lens: an all-pass gate far behind
        zs_f = np.full((res, res), 4.0 * half)
        ze_f = zs_f + eps
    pix = (2.0 * half) / (res - 1)
    sig = max(1.0, 0.02 * radius_total / pix)
    zs_b = _blur(zs_f, sig)
    ze_b = np.maximum(_blur(ze_f, sig), zs_b + eps)
    return {'n': n, 'e1': e1, 'e2': e2, 'c0': c0, 'half': float(half), 'res': int(res), 'pix': pix,
            'u0': float(-half), 'v0': float(-half), 'img': np.ascontiguousarray(np.stack([zs_b, ze_b])),
            'zf': zf, 'zfar': zfar, 'znext': znext, 'hit': hit, 'eps': eps}
