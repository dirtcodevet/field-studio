"""Protected parts in the viewport (11-09, design: hidden parts let a lid be shaped
into the eyeball unseen).

ghost(...)    'ghosted' view: the parts are left out of the frame's trace (the body under them shows) and drawn over it
              as a translucent tint with an outline, from a parts-only pick of every pixel on the card.
overlaps(...) red where an EDITABLE object's visible surface lies inside a locked part + gap and did not at her start:
              'new overlaps only' (her body still holds copies of her eyes and other parts: missing anatomy, built by
              a separate lane; until then only the overlap an edit created is shown). The start is the object's base
              field (its sculpt layer left out).
"""
import math

import numpy as np

GHOST_TINT = np.array([120.0, 200.0, 255.0])
RED = np.array([255.0, 30.0, 30.0])


def _rays(cam, W, H, step=1):
    f, r, u = cam.basis()
    th = math.tan(math.radians(cam.fov) * 0.5)
    a = W / float(H)
    xs = np.arange(0, W, step) + 0.5 * step
    ys = np.arange(0, H, step) + 0.5 * step
    px = (2.0 * xs / W - 1.0) * th * a
    py = (1.0 - 2.0 * ys / H) * th
    if getattr(cam, 'ortho', False):                # fs-shell: orthographic: parallel rays (Camera.ray)
        return np.broadcast_to(f[None, None], (len(ys), len(xs), 3)).copy()
    d = f[None, None] + r[None, None] * px[None, :, None] + u[None, None] * py[:, None, None]
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return d


def _origins(cam, W, H, step=1):
    """each pixel's ray origin (H, W, 3): the camera, or in the orthographic view its point on the rays' start plane"""
    if not getattr(cam, 'ortho', False):
        hs, ws = len(np.arange(0, H, step)), len(np.arange(0, W, step))
        return np.broadcast_to(np.asarray(cam.pos, float), (hs, ws, 3))
    f, r, u = cam.basis()
    th = math.tan(math.radians(cam.fov) * 0.5)
    a = W / float(H)
    xs = np.arange(0, W, step) + 0.5 * step
    ys = np.arange(0, H, step) + 0.5 * step
    px = (2.0 * xs / W - 1.0) * th * a
    py = (1.0 - 2.0 * ys / H) * th
    D, B = cam.ortho_params()
    return (np.asarray(cam.pos, float)[None, None] + (r[None, None] * px[None, :, None] + u[None, None] * py[:, None, None]) * D
            - f[None, None] * B)


def ghost(gscene, cam, rgb, depth, part_oids, alpha=0.35):
    """-> rgb with the parts drawn translucent + outlined (the parts must be left out of the frame's own trace)"""
    if not part_oids:
        return rgb
    H, W = depth.shape
    d = _rays(cam, W, H).reshape(-1, 3)
    o = np.ascontiguousarray(_origins(cam, W, H).reshape(-1, 3))
    t, v, n = gscene.pick(o, d, only=tuple(part_oids))
    t = t.reshape(H, W)
    m = (t > 0) & (t <= np.where(np.isfinite(depth) & (depth < 1e20), depth, np.inf) * 1.0005 + 1e-6)
    if not m.any():
        return rgb
    e = m.copy()
    e[1:, :] &= m[:-1, :]; e[:-1, :] &= m[1:, :]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
    edge = m & ~e
    out = rgb.astype(np.float32)
    out[m] = out[m] * (1 - alpha) + GHOST_TINT * alpha
    out[edge] = out[edge] * 0.1 + GHOST_TINT * 0.9
    return out.astype(np.uint8)


TOUCH_TOL_CELLS = 0.25     # sculpt-contact 11-12: a surface within a quarter of a lattice cell of a part is touching it


def overlaps(scene, cam, rgb, depth, obj, gap=0.0, max_px=400000):
    """-> (rgb with new overlaps red, pixel count). sculpt-contact 11-12 (sheets 1-3: Swell / Settle rest ON a part at
    gap 0 and the window painted every correct touch red, 954-1020 px on her lid): red only where the surface lies INSIDE
    a part by more than TOUCH_TOL_CELLS of its lattice cell (the brushes' own 0.3 mm no-new-overlap gap is a margin they
    keep, not an overlap)"""
    from ..core import collide as CL
    from ..core.scene import is_protected
    from ..tools.protected_merge import field_of, _eval_exact, _to_local, _rs
    edit = [o for o in scene.objects if not is_protected(o) and scene.shown(o)]
    if not edit or not any(is_protected(o) for o in scene.objects):
        return rgb, 0
    H, W = depth.shape
    ok = np.isfinite(depth) & (depth < 1e20)
    pk = scene.pack()
    out = rgb.copy()
    total = 0
    d = _rays(cam, W, H)
    for o in edit:
        parts = CL.colliders(scene, o.id)
        sel = ok & (obj == o.id)
        if not parts or not sel.any():
            continue
        ys, xs = np.nonzero(sel)
        if len(ys) > max_px:
            k = np.linspace(0, len(ys) - 1, max_px).astype(np.int64)
            ys, xs = ys[k], xs[k]
        P_ = _origins(cam, W, H)[ys, xs] + d[ys, xs] * depth[ys, xs][:, None]
        B = CL.bound(scene, pk, parts, P_, gap)
        c = B > TOUCH_TOL_CELLS * float(o.sculpt.h) * float(o.transform['scale'])
        if not c.any():
            continue
        ys, xs, P_, B = ys[c], xs[c], P_[c], B[c]
        # her start: the object's base field (no sculpt layer): eval with an empty delta grid
        oi = scene.index_of(o.id)
        X = np.ascontiguousarray(_to_local(o, P_))
        Fb = np.empty(len(X))
        from ..core import program as Pg
        tab = pk.oi_tab.copy()
        tab[oi, Pg.OI_NX] = 0                    # no sculpt bricks: the base field alone
        _eval_exact(pk.code, pk.prm, tab, pk.of_tab, pk.index_flat, pk.delta, oi, X, Fb)
        Fb *= _rs(o)[2]
        new = Fb >= B - 1e-6
        out[ys[new], xs[new]] = (out[ys[new], xs[new]] * 0.3 + RED * 0.7).astype(np.uint8)
        total += int(new.sum())
    return out, total
