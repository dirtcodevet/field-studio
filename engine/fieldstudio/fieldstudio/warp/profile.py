"""The shared profile (TOOLSET_DESIGN 1.3): Radius, Hardness, Smoothing and the one falloff choice, defined once.

w(d) is the weight of a tool at distance d from its centre (along the lens for a flat tool, through space for a ball):
- inside the Radius the PEAK: Hardness 0 = a soft domed peak (cos), 1 = the tool's sharpest (a cone: "stretches to a
  pin-point"); the peak always meets zero at the Radius at an angle, so Smoothing has a corner to round;
- Smoothing rounds that corner (the FOOT) with a smooth max over a band of exactly `smoothing` beyond the Radius:
  "nothing beyond radius + smoothing ever changes" (w = 0 for d >= r + s, exactly);
- the falloff choice is the shape of that rounding: soft (C1, quadratic), even (C2, cubic), sharp (C3, quartic);
- w(0) = 1 always (normalised), so Strength keeps its meaning at the centre.

Hardness shapes the peak, Smoothing shapes the foot: the two never overlap in meaning (Smoothing
addresses the transition from the outer edge of the tool radius to the surrounding area; Hardness, how sharp the point is).
"""
import math

import numpy as np
from numba import njit

FALLOFF = {'soft': 0, 'even': 1, 'sharp': 2}
PEAK_SHAPE = 0        # Shape / warps: dome (H 0) to cone (H 1)
PEAK_SWELL = 1        # Swell: soft bell (H 0) to a sphere's dome of that radius (H 1), never sharper


@njit(cache=True, inline='always')
def _peak(u, hard, family):
    """the peak on u = d / r in [0, 1]: 1 at u = 0, 0 at u = 1; and its edge slope (d/du at u = 1, < 0)"""
    if family == PEAK_SWELL:
        # soft: a bell with a long, slow taper (cos^2); hard: the sphere's dome sqrt(1 - u^2); edge slopes -0 / -inf are
        # replaced by a finite corner slope for the foot (the dome's own tangent is vertical at the edge)
        uu = min(max(u, 0.0), 1.0)
        a = math.cos(0.5 * math.pi * uu)
        bell = a * a
        dome = math.sqrt(max(1.0 - uu * uu, 0.0))
        return (1.0 - hard) * bell + hard * dome, -((1.0 - hard) * 0.25 * math.pi + hard * 2.0)
    uu = min(max(u, 0.0), 1.0)
    return (1.0 - hard) * math.cos(0.5 * math.pi * uu) + hard * (1.0 - uu), -((1.0 - hard) * 0.5 * math.pi + hard)


@njit(cache=True, inline='always')
def _smax0(a, k, fall):
    """smooth max(a, 0) over |a| < k; exactly max(a, 0) outside"""
    if k <= 0.0:
        return max(a, 0.0)
    h = k - abs(a)
    if h <= 0.0:
        return max(a, 0.0)
    h = h / k
    if fall == 0:
        return max(a, 0.0) + k * h * h * 0.25
    if fall == 1:
        return max(a, 0.0) + k * h * h * h / 6.0
    return max(a, 0.0) + k * h * h * h * h * 0.125


@njit(cache=True, inline='always')
def _raw(d, r, hard, smooth, fall, family):
    u = d / r
    pk, m = _peak(u, hard, family)
    a = pk if u <= 1.0 else m * (u - 1.0)
    k = -m * smooth / r                         # the foot reaches exactly r + smoothing (a = -k there)
    return _smax0(a, k, fall)


@njit(cache=True, inline='always')
def weight(d, r, hard, smooth, fall, family):
    """the profile w(d) in [0, 1]; w(0) = 1; w = 0 for d >= r + smooth"""
    if d >= r + smooth:
        return 0.0
    w0 = _raw(0.0, r, hard, smooth, fall, family)
    return min(_raw(d, r, hard, smooth, fall, family) / w0, 1.0)


def profile_curve(r, hardness, smoothing, falloff='soft', family=PEAK_SHAPE, n=129):
    """(d, w) samples from 0 to r + smoothing: the profile picture by the cursor (TOOLSET_DESIGN 1.3, 'Drawn')"""
    fall = FALLOFF[falloff] if isinstance(falloff, str) else int(falloff)
    d = np.linspace(0.0, r + smoothing, n)
    return d, np.array([weight(float(x), float(r), float(hardness), float(smoothing), fall, int(family)) for x in d])


LEGACY_FOOT = 0.3     # today's brushes: the outer 30 % of the radius is the foot (they have no Smoothing distance)


def legacy_curve(hardness, foot='soft', n=65):
    """Hardness (softness) for today's brushes (Grab, Snake, Elastic, Draw, Clay, Carve, Inflate, Crease, Pinch, Smooth,
    Mask; fs-shell-5, 11-14: radius, strength and softness on the move brush): their
    falloff table w(u), u = d / radius in [0, 1], n samples, w(0) = 1, w(1) = 0, for falloff='custom' + curve (core/brush
    falloff_lut; the card reads the same keys). The SAME profile as every approved tool (weight above, the Shape family):
    Hardness 0 = a soft, domed peak (a broad pull: the whole radius comes along, rounded on top); 1 = the cone that
    "stretches to a pin-point" (a tight pull: the move gathers at the centre); between, a blend. These brushes have no
    Smoothing distance, so the foot is the outer LEGACY_FOOT of the radius, rounded by the falloff choice (soft / even /
    sharp). At Hardness 0 it is within 0.054 (soft foot; even 0.106, sharp 0.133) of today's default falloff (1 -
    smoothstep) everywhere, so touching the slot doesn't jump a brush; w(0.5) is 0.455 at Hardness 0, 0.298 at 1."""
    h = min(max(float(hardness), 0.0), 1.0)
    fall = FALLOFF.get(str(foot), 0) if not isinstance(foot, (int, np.integer)) else int(foot)
    rp = 1.0 - LEGACY_FOOT
    u = np.linspace(0.0, 1.0, int(n))
    w = np.array([weight(float(x), rp, h, LEGACY_FOOT, fall, PEAK_SHAPE) for x in u])
    w[0], w[-1] = 1.0, 0.0
    return [float(v) for v in np.clip(w, 0.0, 1.0)]
