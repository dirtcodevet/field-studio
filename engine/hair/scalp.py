"""Hair Studio scalp (numpy + scipy, Apache-2.0): the head's spherical frame and the hairline boundary. Mesh-free since
11-14 (fs-hairclean, P14): the triangle scalp's root sampling and cap are archived with mesh.py; the roots on a body come
from its field (field_collider.FieldScalp: the follicle map's samples, re-seated on the field).

Frame: spherical coordinates about a head centre c: phi = azimuth (0 = the face, -y; + = the subject's left, +x),
theta = polar angle from +z. The hair's root boundary is theta_b(phi) on a 1-degree grid: a point is d_inside_mm inside
the hairline (measured along the meridian; negative = outside, on the face / neck).

    boundary(sph, front_P, side_P, ...)  theta_b from a traced front hairline and a side / nape boundary, blended
    Scalp(centre, ears, ear_mask, grid, thb, z_min)   the frame (field_collider.FieldScalp adds the roots)
        .sph(P), .theta_b(phi), .d_inside_mm(P)
ears: {+1: {y_back, y_front, z_top, z_bot, x_out}, -1: {...}} (m) - the ear boxes the tuck and the corridor use.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def sph_about(c, P):
    v = np.asarray(P, np.float64) - c
    r = np.linalg.norm(v, axis=-1)
    phi = np.degrees(np.arctan2(v[..., 0], -v[..., 1]))
    th = np.degrees(np.arccos(np.clip(v[..., 2] / np.maximum(r, 1e-9), -1, 1)))
    return phi, th, r


def boundary(c, front_P, side_P, drop_mm=0.0, temple_mm=0.0, temple_phi=(25.0, 50.0), side_raise_mm=0.0, blend_deg=12.0,
             smooth_deg=1.5):
    """theta_b(phi) on a 1-degree grid (-180..180): the front from a hairline polyline on the skin (lowered drop_mm along
    the meridian, + temple_mm at the temples), the sides and nape from side_P (per phi bin its lowest point, raised
    side_raise_mm), blended over blend_deg beyond the front's span. Returns grid, thb, (front phi span)."""
    grid = np.arange(-180, 181, 1.0)
    ph, th, r = sph_about(c, side_P)
    thc = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        s = np.abs(((ph - g + 180) % 360) - 180) <= 2.0
        if s.any():
            thc[i] = th[s].max() - np.degrees(side_raise_mm / 1000 / r[s].mean())
    ok = np.isfinite(thc)
    thc = np.interp(grid, grid[ok], thc[ok], period=360)
    pt, tt, rt = sph_about(c, front_P)
    drop = drop_mm + temple_mm * smoothstep(temple_phi[0], temple_phi[1], np.abs(pt))
    tt = tt + np.degrees(drop / 1000 / rt)
    o = np.argsort(pt)
    pt, tt = pt[o], tt[o]
    tht = np.interp(grid, pt, tt)
    lo, hi = pt.min(), pt.max()
    w = np.ones(len(grid))
    w = np.where(grid < lo, smoothstep(lo - blend_deg, lo, grid), w)
    w = np.where(grid > hi, 1 - smoothstep(hi, hi + blend_deg, grid), w)
    tht = np.where(grid < lo, tt[0], np.where(grid > hi, tt[-1], tht))
    thb = w * tht + (1 - w) * thc
    thb = gaussian_filter1d(thb, smooth_deg, mode="wrap")
    return grid, thb, (float(lo), float(hi))


class Scalp:
    def __init__(self, centre, ears, ear_mask, grid, thb, z_min=-np.inf):
        self.c = np.asarray(centre, np.float64)
        self.ears = {int(k): v for k, v in ears.items()}
        self.ear_mask = ear_mask
        self.grid, self.thb = np.asarray(grid, np.float64), np.asarray(thb, np.float64)
        self.z_min = z_min

    def sph(self, P):
        return sph_about(self.c, P)

    def theta_b(self, phi):
        return np.interp(phi, self.grid, self.thb, period=360)

    def d_inside_mm(self, P):
        ph, th, r = self.sph(P)
        return np.radians(self.theta_b(ph) - th) * r * 1000
