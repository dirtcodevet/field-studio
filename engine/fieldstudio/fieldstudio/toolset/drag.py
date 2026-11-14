"""Turning a 2-D drag into a 3-D move (TOOLSET_DESIGN 1.5, approved 11-14), for Shape, Transform and Place.

"A guide drawn at the press shows the plane the drag moves in: by default the plane through the tool's direction that
faces the viewer best.
- Dragging along the direction's on-screen line pulls straight out or pushes in, from any angle.
- Ctrl moves along the direction only; the drag's length drives it even head-on.
- X / Y / Z lock to an axis.
When the view makes a choice necessary (looking straight down the direction), the guide shows which plane it switched
to; it never changes silently."

guide(cam_pos, cam_fwd, cam_up, P, n) -> {'point', 'normal', 'switched', 'axis'}: the guide plane through the press
point P containing the tool's direction n, facing the viewer best (its normal is the view direction with its component
along n removed); head-on (the view within ~8 deg of n) it switches to the plane containing n and the camera's up, and
says so.
move(guide, P, n, ray, mode, ...) -> D (world m): where the cursor ray meets the guide plane, minus the press point,
times Strength (1 = the surface follows the cursor exactly); mode 'normal' (Ctrl) keeps only the part along n, and
head-on uses the drag's screen length; 'x' / 'y' / 'z' lock to a world axis.
"""
import math

import numpy as np

HEAD_ON_DEG = 8.0


def _u(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-300)


def guide(cam_pos, cam_fwd, cam_up, P, n):
    n = _u(n)
    v = _u(np.asarray(P, float) - np.asarray(cam_pos, float))       # the view ray toward the press point
    perp = v - (v @ n) * n
    switched = None
    if np.linalg.norm(perp) < math.sin(math.radians(HEAD_ON_DEG)):
        # looking (nearly) straight down the direction: the plane holding n and the camera's up (said on the guide)
        up = _u(cam_up)
        m = np.cross(n, up)
        if np.linalg.norm(m) < 1e-6:
            m = np.cross(n, _u(np.cross(cam_fwd, up)))
        switched = 'head-on: the plane through the direction and the view\'s up'
        normal = _u(m)
    else:
        normal = _u(perp)
    return {'point': np.asarray(P, float), 'normal': normal, 'switched': switched, 'axis': n}


def move(g, ray, mode='free', strength=1.0, axis=None, screen_len_px=None, px_size=None):
    """the 3-D move D for the cursor's ray (origin, direction) on the guide g"""
    o, d = np.asarray(ray[0], float), _u(ray[1])
    nrm, P, n = g['normal'], g['point'], g['axis']
    den = d @ nrm
    if abs(den) < 1e-9:
        hit = P
    else:
        hit = o + ((P - o) @ nrm) / den * d
    D = hit - P
    if mode == 'normal':
        if g['switched'] is not None and screen_len_px is not None and px_size is not None:
            D = n * screen_len_px * px_size           # head-on: the drag's length drives it
        else:
            D = (D @ n) * n
    elif mode in ('x', 'y', 'z'):
        a = np.zeros(3)
        a['xyz'.index(mode)] = 1.0
        D = (D @ a) * a
    return strength * D
