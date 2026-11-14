"""Reference image planes (hand1, 2026-11-07): a front and a side photo shown as two planes crossing at one anchor,
composited over the traced frame (no Qt here; the UI loads the images and hands RGBA arrays in).

World: metres, Y up, the front view (key 1) looks down -Z from +Z, +X is her left (M8). The front plane is the XY
plane through the anchor (n = +Z, image +x -> world +X, image +y up -> world +Y); the side plane is the YZ plane
through the anchor (n = +X, image +x -> world -Z), which matches a left profile with the nose to the image's left as
seen from the Right preset (key 3). Each plane's centre pixel (default the image centre) sits on the anchor plus its
offset, so with zero offsets the two image centres meet dead centre on the vertical line where the planes cross.
Scale: the image height (span_px rows, default the full height) is span_mm millimetres (default 300).

Draw modes: 'depth' (the model in front draws over the plane), 'behind' (only where no model is drawn), 'xray',
'auto' (hand1b, the default: x-ray when the view looks within ~20 deg of straight at the plane, so the front photo
can be traced in the Front view and the profile in the Side view; depth-tested in every other view).
The planes are view state: they are not in the op log and do not change the state hash.
"""
import math
from dataclasses import dataclass, field, asdict

import numpy as np
from numba import njit, prange

AXES = {'front': (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        'side': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]))}
DRAW = {'depth': 0, 'behind': 1, 'xray': 2}
AUTO_ON_TOP = 0.94      # hand1b: 'auto' draws a plane on top when the view is within ~20 deg of looking straight at it
NCOL = 24


@dataclass
class RefPlane:
    name: str = 'front'                      # 'front' | 'side' (sets the axis)
    path: str = ''
    visible: bool = True
    opacity: float = 0.5
    offset_mm: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    centre_px: list = None                   # [x, y] image pixel on the anchor; None = the image centre
    span_px: list = None                     # [row0, row1] of a measured feature; None = the full image height
    span_mm: float = 300.0                   # millimetres the span covers
    flip_h: bool = False
    sides: str = 'both'                      # 'both' | 'facing'
    draw: str = 'auto'                       # 'auto' | 'depth' | 'behind' | 'xray' (hand1b: auto is the default; a
                                             # depth-tested plane through the origin is hidden inside a bust)
    key_bg: bool = True                      # hand2b: a flat background joined to the image border is transparent
    marks: dict = None                       # fs-shell rig (11-11): {'crown': [x, y], 'chin': [x, y], 'centre_x': x,
                                             # 'flip_h'} in ORIGINAL image pixels; centre_px / span_px are solved from it


@dataclass
class RefSet:
    planes: list = field(default_factory=list)
    anchor_mm: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fade_edge_on: bool = True
    show_frame: bool = True
    show_cross: bool = True
    visible: bool = True
    rig_height_mm: float = None              # fs-shell rig (11-11): the typed crown-to-chin height every plane spans
    sil_check: bool = True                   # fs-shell (11-11): the silhouette check in a reference's own view

    def get(self, name):
        for p in self.planes:
            if p.name == name:
                return p
        return None

    def to_dict(self, rel_to=None):
        import os
        d = asdict(self)
        for p in d['planes']:
            if rel_to and p['path']:
                try:
                    p['path'] = os.path.relpath(p['path'], rel_to)
                except ValueError:
                    pass
        return d

    @staticmethod
    def from_dict(d, base=None):
        import os
        rs = RefSet()
        for k in ('anchor_mm', 'fade_edge_on', 'show_frame', 'show_cross', 'visible', 'rig_height_mm', 'sil_check'):
            if k in d:
                setattr(rs, k, d[k])
        for pd in d.get('planes', []):
            p = RefPlane(**{k: v for k, v in pd.items() if k in RefPlane.__dataclass_fields__})
            if base and p.path and not os.path.isabs(p.path):
                p.path = os.path.normpath(os.path.join(base, p.path))
            rs.planes.append(p)
        return rs


def background_mask(rgba, tol=12):
    """hand2b (verifier item 11): -> bool (H, W), True on the image's flat background: the pixels within tol (0-255,
    per channel) of the border's median colour that connect to the border. Generated references stand on black, which
    darkened the model wherever the plane was drawn; keyed out, only the subject (and any matte around it) shows.
    A photo without a flat border colour keys nothing."""
    from scipy import ndimage
    a = rgba[..., :3].astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    near = np.all(np.abs(a - bg[None, None]) <= tol, axis=-1)
    if near[0].mean() < 0.5 and near[-1].mean() < 0.5 and near[:, 0].mean() < 0.5 and near[:, -1].mean() < 0.5:
        return np.zeros(near.shape, bool)       # no flat border: nothing to key
    lab, n = ndimage.label(near)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    return np.isin(lab, edge)


def key_background(rgba, mask):
    """rgba with alpha 0 on mask (a 1 px soft edge), colour unchanged"""
    from scipy import ndimage
    out = rgba.copy()
    soft = ndimage.uniform_filter(mask.astype(np.float32), 3)
    out[..., 3] = (out[..., 3].astype(np.float32) * np.where(mask, 0.0, 1.0 - 0.5 * soft)).astype(np.uint8)
    return out


def content_extent(mask, min_frac=0.003):
    """-> (top, bottom, left, right) pixel rows / columns of the non-background content (bottom, right inclusive),
    or None. A row or column counts when more than min_frac of it is content."""
    cont = ~mask
    H, W = cont.shape
    rows = np.where(cont.sum(1) > max(1, W * min_frac))[0]
    cols = np.where(cont.sum(0) > max(1, H * min_frac))[0]
    if not len(rows) or not len(cols):
        return None
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def prepare_texture(rgba, levels=4):
    """(H, W, 4) uint8 -> list of uint8 mip levels (H_i, W_i, 4), 2x2 box filtered. fs-shell (11-11): uint8, a quarter
    of the float32 levels' memory and of the bytes the kernel reads per texel (it scales by 1/255)."""
    t = np.ascontiguousarray(rgba[..., :4], dtype=np.uint8)
    out = [t]
    for _ in range(levels - 1):
        a = out[-1]
        h, w = a.shape[0] // 2, a.shape[1] // 2
        if h < 4 or w < 4:
            break
        a = a[:2 * h, :2 * w].astype(np.uint16)
        out.append(np.ascontiguousarray(((a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2] + 2) // 4)
                                        .astype(np.uint8)))
    return out


def plane_geometry(p, anchor_mm, W, H):
    """-> dict(c, u, v, n, mpp, pcx, pcy) for an image W x H px (world metres; mpp = metres per pixel)."""
    n, u, v = AXES.get(p.name, AXES['front'])
    u = -u if p.flip_h else u
    rows = abs(p.span_px[1] - p.span_px[0]) if p.span_px else H
    mpp = float(p.span_mm) / 1000.0 / max(float(rows), 1.0)
    pcx, pcy = (p.centre_px if p.centre_px else (W / 2.0, H / 2.0))
    c = (np.asarray(anchor_mm, float) + np.asarray(p.offset_mm, float)) / 1000.0
    return {'c': c, 'u': u, 'v': v, 'n': n, 'mpp': mpp, 'pcx': float(pcx), 'pcy': float(pcy)}


def outline(p, anchor_mm, W, H):
    """-> (4 world corners, world centre) of the plane's image rectangle (for the viewport overlay)."""
    g = plane_geometry(p, anchor_mm, W, H)

    def wp(ix, iy):
        return g['c'] + g['u'] * (ix - g['pcx']) * g['mpp'] - g['v'] * (iy - g['pcy']) * g['mpp']
    return [wp(0, 0), wp(W, 0), wp(W, H), wp(0, H)], g['c']


_PACKED = {'arrays': None, 'flat': None}


def pack(refs, textures, cam_pos, cam_f, frame_h, tan_half, ortho_D=None):
    """-> (tab float64 (P, NCOL), flat uint8 1-D, nplanes) for the visible planes with a texture. fs-shell (11-11): the
    flat texel buffer is built only when the drawn planes' mip levels change (it was concatenated, 16-32 MB of float32,
    on every frame: 5-9 ms of the GUI thread per frame while sculpting)."""
    rows, chunks, off = [], [], 0
    for p in refs.planes:
        tex = textures.get(p.name)
        if not p.visible or tex is None or p.opacity <= 0.0 or not refs.visible:
            continue
        H0, W0 = tex[0].shape[:2]
        W0, H0 = getattr(tex, 'W0', W0), getattr(tex, 'H0', H0)     # the original image (picks are in its pixels)
        g = plane_geometry(p, refs.anchor_mm, W0, H0)
        fade = 1.0
        if refs.fade_edge_on:
            fade = min(max((abs(float(cam_f @ g['n'])) - 0.08) / 0.22, 0.0), 1.0)
        if fade <= 0.0:
            continue
        # mip level from the plane's on-screen height (pixels) against the image height
        dist = ortho_D if ortho_D else max(float(np.linalg.norm(g['c'] - cam_pos)), 1e-3)
        plane_px = (H0 * g['mpp']) / max(2 * dist * tan_half, 1e-6) * frame_h
        H_t = tex[0].shape[0]
        lvl = int(min(max(math.floor(math.log2(max(H_t / max(plane_px, 1.0), 1.0))), 0), len(tex) - 1))
        t = tex[lvl]
        h, w = t.shape[:2]
        sc = w / float(W0)
        row = np.zeros(NCOL)
        row[0:3], row[3:6], row[6:9], row[9:12] = g['c'], g['u'], g['v'], g['n']
        row[12] = g['mpp'] / sc
        row[13], row[14] = g['pcx'] * sc, g['pcy'] * sc
        row[15], row[16] = w, h
        row[17] = float(p.opacity) * fade
        if p.draw == 'auto':
            row[18] = DRAW['xray'] if abs(float(cam_f @ g['n'])) >= AUTO_ON_TOP else DRAW['depth']
        else:
            row[18] = DRAW.get(p.draw, 0)
        row[19] = 1.0 if p.sides == 'facing' else 0.0
        row[20] = off
        rows.append(row)
        chunks.append(t)
        off += t.size
    if not rows:
        return None, None, 0
    prev = _PACKED['arrays']
    if prev is None or len(prev) != len(chunks) or any(a is not b for a, b in zip(prev, chunks)):
        _PACKED['arrays'] = list(chunks)
        _PACKED['flat'] = np.concatenate([c.reshape(-1) for c in chunks])
    return np.array(rows), _PACKED['flat'], len(rows)


@njit(parallel=True, cache=True, nogil=True)
def layers_kernel(cam, w, h, tab, flat, rx0, ry0, rx1, ry1, T, C):
    """fs-shell (11-11): each plane's layer over the rectangle rx0..rx1 x ry0..ry1 of a w x h frame: T[q] the ray
    parameter of the plane's hit (inf: none), C[q] its colour and alpha (alpha with the plane's opacity and fade).
    It depends on the camera and the planes only, so it is computed once per view and reused while a stroke runs."""
    P = tab.shape[0]
    th = cam[12]
    orth = cam.shape[0] > 18 and cam[16] > 0.5
    oD = cam[17] if orth else 0.0
    oB = cam[18] if orth else 0.0
    for jj in prange(ry1 - ry0):
        j = ry0 + jj
        for ii in range(rx1 - rx0):
            i = rx0 + ii
            px = (2.0 * (i + 0.5) / w - 1.0) * th * w / h
            py = (1.0 - 2.0 * (j + 0.5) / h) * th
            if orth:
                ox = cam[0] + (cam[6] * px + cam[9] * py) * oD - cam[3] * oB
                oy = cam[1] + (cam[7] * px + cam[10] * py) * oD - cam[4] * oB
                oz = cam[2] + (cam[8] * px + cam[11] * py) * oD - cam[5] * oB
                dx, dy, dz = cam[3], cam[4], cam[5]
            else:
                ox, oy, oz = cam[0], cam[1], cam[2]
                dx = cam[3] + cam[6] * px + cam[9] * py
                dy = cam[4] + cam[7] * px + cam[10] * py
                dz = cam[5] + cam[8] * px + cam[11] * py
                dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= dl
                dy /= dl
                dz /= dl
            for q in range(min(P, 4)):
                T[q, jj, ii] = np.inf
                nx, ny, nz = tab[q, 9], tab[q, 10], tab[q, 11]
                den = dx * nx + dy * ny + dz * nz
                if abs(den) < 1e-9:
                    continue
                if tab[q, 19] > 0.5 and den > 0.0:      # facing only: the plane is seen from its back
                    continue
                t = ((tab[q, 0] - ox) * nx + (tab[q, 1] - oy) * ny + (tab[q, 2] - oz) * nz) / den
                if t <= 0.0:
                    continue
                qx = ox + dx * t - tab[q, 0]
                qy = oy + dy * t - tab[q, 1]
                qz = oz + dz * t - tab[q, 2]
                mpp = tab[q, 12]
                ix = tab[q, 13] + (qx * tab[q, 3] + qy * tab[q, 4] + qz * tab[q, 5]) / mpp - 0.5
                iy = tab[q, 14] - (qx * tab[q, 6] + qy * tab[q, 7] + qz * tab[q, 8]) / mpp - 0.5
                W = int(tab[q, 15])
                H = int(tab[q, 16])
                if ix < -0.5 or iy < -0.5 or ix > W - 0.5 or iy > H - 0.5:
                    continue
                x0 = int(math.floor(ix))
                y0 = int(math.floor(iy))
                fx = ix - x0
                fy = iy - y0
                off = int(tab[q, 20])
                c0 = 0.0
                c1 = 0.0
                c2 = 0.0
                c3 = 0.0
                for k in range(4):
                    xx = min(max(x0 + (k & 1), 0), W - 1)
                    yy = min(max(y0 + (k >> 1), 0), H - 1)
                    wk = (fx if (k & 1) else 1.0 - fx) * (fy if (k >> 1) else 1.0 - fy)
                    b = off + (yy * W + xx) * 4
                    c0 += wk * flat[b]
                    c1 += wk * flat[b + 1]
                    c2 += wk * flat[b + 2]
                    c3 += wk * flat[b + 3]
                a = c3 * (1.0 / 255.0) * tab[q, 17]
                if a <= 0.0:
                    continue                            # keyed out: the plane leaves this pixel alone
                T[q, jj, ii] = t
                C[q, jj, ii, 0] = c0 * (1.0 / 255.0)
                C[q, jj, ii, 1] = c1 * (1.0 / 255.0)
                C[q, jj, ii, 2] = c2 * (1.0 / 255.0)
                C[q, jj, ii, 3] = a


@njit(cache=True, nogil=True)
def blend_kernel(rgb_in, depth, obj, modes, T, C, rx0, ry0, idx, rgb_out):
    """per frame: each plane's cached layer over the frame at the pixels a plane covers (idx: flat indices in the layer's
    rectangle, listed once per view), far first, tested against this frame's depth (mode 0) or hit mask (mode 1), or on
    top (mode 2). Serial on purpose: it runs in the GUI thread beside the engine's own parallel kernels, and a second
    OpenMP team spinning after every frame took the cores the engine needed (fs-shell 11-11, measured on the card)."""
    P = T.shape[0]
    rw = T.shape[2]
    ts = np.empty(4)
    qs = np.empty(4, np.int64)
    for n in range(idx.shape[0]):
        jj = idx[n] // rw
        ii = idx[n] - jj * rw
        j = ry0 + jj
        i = rx0 + ii
        nh = 0
        for q in range(P):
            t = T[q, jj, ii]
            if not (t < np.inf):
                continue
            m = modes[q]
            if m == 0 and t >= depth[j, i]:
                continue
            if m == 1 and obj[j, i] >= 0:
                continue
            ts[nh] = t
            qs[nh] = q
            nh += 1
        if nh == 0:
            continue
        r = rgb_in[j, i, 0] / 255.0
        g = rgb_in[j, i, 1] / 255.0
        bb = rgb_in[j, i, 2] / 255.0
        for _ in range(nh):                         # far first
            far = -1
            ft = -1.0
            for m in range(nh):
                if ts[m] > ft:
                    ft = ts[m]
                    far = m
            q = qs[far]
            a = C[q, jj, ii, 3]
            r = r * (1.0 - a) + C[q, jj, ii, 0] * a
            g = g * (1.0 - a) + C[q, jj, ii, 1] * a
            bb = bb * (1.0 - a) + C[q, jj, ii, 2] * a
            ts[far] = -2.0
        rgb_out[j, i, 0] = np.uint8(min(max(r * 255.0 + 0.5, 0.0), 255.0))
        rgb_out[j, i, 1] = np.uint8(min(max(g * 255.0 + 0.5, 0.0), 255.0))
        rgb_out[j, i, 2] = np.uint8(min(max(bb * 255.0 + 0.5, 0.0), 255.0))


_LAYERS = {'key': None, 'T': None, 'C': None, 'idx': None}


def screen_rect(refs, textures, cam, w, h):
    """-> (x0, y0, x1, y1): the frame pixels the drawn planes can cover (their corners projected), the whole frame when
    a corner is behind the camera"""
    xs, ys = [], []
    for p in refs.planes:
        tex = textures.get(p.name)
        if not p.visible or tex is None or p.opacity <= 0.0:
            continue
        W0 = getattr(tex, 'W0', tex[0].shape[1])
        H0 = getattr(tex, 'H0', tex[0].shape[0])
        corners, _ = outline(p, refs.anchor_mm, W0, H0)
        for c_ in corners:
            x, y, z = cam.project(c_, w, h)
            if z <= 1e-6:
                return 0, 0, w, h
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0, 0, 0, 0
    return (int(max(0, math.floor(min(xs)) - 1)), int(max(0, math.floor(min(ys)) - 1)),
            int(min(w, math.ceil(max(xs)) + 2)), int(min(h, math.ceil(max(ys)) + 2)))


def composite(frame, refs, textures):
    """frame: the engine's frame dict (rgb, depth, obj at the internal size, cam). -> rgb uint8 (h, w, 3)."""
    rgb = frame['rgb']
    if refs is None or not refs.visible or frame['stats'].get('mode') == 'slice':
        return rgb
    from .raymarch import _cam_array
    cam = frame['cam']
    h, w = rgb.shape[:2]
    ca = _cam_array(cam)
    tab, flat, n = pack(refs, textures, np.asarray(cam.pos, float), np.asarray(cam.basis()[0], float), h,
                        float(ca[12]), ortho_D=(float(ca[17]) if len(ca) > 18 and ca[16] > 0.5 else None))
    if n == 0:
        return rgb
    out = np.array(rgb, copy=True, order='C')
    depth = frame['depth']
    obj = frame['obj']
    if depth.shape != (h, w):
        depth = np.full((h, w), np.inf, np.float32)
        obj = np.full((h, w), -1, np.int32)
    x0, y0, x1, y1 = screen_rect(refs, textures, cam, w, h)
    if x1 > x0 and y1 > y0:
        # fs-shell (11-11): the planes' layer (each pixel's ray, plane hit and texel) depends on the camera and the
        # planes only: computed once per view and reused while a stroke runs; per frame only the blend against this
        # frame's depth (fs-bench: the planes took ~a third of a sculpting frame when all of it ran every frame)
        key = (tuple(np.round(ca, 12)), w, h, x0, y0, x1, y1, tab.tobytes(), id(flat))
        if _LAYERS['key'] != key:
            P_ = min(tab.shape[0], 4)
            T = np.empty((P_, y1 - y0, x1 - x0), np.float32)
            C = np.empty((P_, y1 - y0, x1 - x0, 4), np.float32)
            layers_kernel(ca, w, h, tab, flat, x0, y0, x1, y1, T, C)
            idx = np.flatnonzero(np.isfinite(T).any(0)).astype(np.int64)
            _LAYERS.update(key=key, T=T, C=C, idx=idx)
        modes = np.array([int(tab[q, 18]) for q in range(min(tab.shape[0], 4))], np.int64)
        blend_kernel(np.ascontiguousarray(rgb), np.ascontiguousarray(depth), np.ascontiguousarray(obj), modes,
                     _LAYERS['T'], _LAYERS['C'], x0, y0, _LAYERS['idx'], out)
    # fs-shell (11-11, design C6): the silhouette check, in a reference's own view, from the hit mask of this frame
    refs.sil_last = None
    if getattr(refs, 'sil_check', False) and depth.shape == (h, w):
        try:
            from . import refrig
            res = refrig.silcheck(dict(frame, obj=obj), refs, textures)
            if res is not None:
                out = refrig.draw_silcheck(out, res, thick=1 if w < 900 else 2, inplace=True)
                refs.sil_last = {k: res.get(k) for k in ('plane', 'n', 'mean', 'p90', 'max', 'within1', 'n_out',
                                                          'span_mm')}
                refs.sil_res = res
        except Exception as e:                      # the check is an aid: it must never stop the view
            refs.sil_error = repr(e)
    return out


def warmup():
    """Compile the kernel at launch (engine thread), on a tiny frame and texture."""
    from .camera import Camera
    from .raymarch import _cam_array
    cam = Camera(pos=(0.0, 0.0, 0.6), target=(0, 0, 0), fov=35.0)
    tex = {'front': prepare_texture(np.full((16, 16, 4), 200, np.uint8), 2)}
    rs = RefSet([RefPlane('front')])
    rgb = np.zeros((18, 32, 3), np.uint8)
    fr = {'rgb': rgb, 'depth': np.full((18, 32), np.inf, np.float32), 'obj': np.full((18, 32), -1, np.int32),
          'cam': cam, 'stats': {'mode': 'shaded'}}
    composite(fr, rs, tex)
