"""The Reference Rig (fs-shell, 2026-11-11; design 4.3 and appendix C6): a front and a side photo held on one centre and
one scale, set by two clicks per image (crown, chin) plus a centre column (the midline in the front, the ear canal in
the profile), or proposed by Auto-fit from each photo's own silhouette. No Qt here: the UI hands RGBA arrays in.

What this module does, each part measured on the staged bust pair and the drawn pair (fs-shell/CONTINUE.md):
- matte(rgba): the subject against its background. The background is modelled as a smooth colour field fitted to the
  image border (a quadratic per channel, robust to the subject touching the border), so a studio gradient keys
  cleanly; the alpha at the edge is the colour's position between the local subject colour and the local background
  colour (alpha = (I - B).(F - B) / |F - B|^2), and the edge colour is unmixed from the background. The old key
  (refplanes.background_mask: within 12 levels of the border median) left a speckled grey fringe where the gradient
  left that band (test-1 sheet, panel 2).
- autofit(matte): proposals for the crown (the silhouette's top), the chin (front: the under-chin shadow edge on the
  midline; profile: the chin's lowest point where its underside turns level), the centre column (front: the
  cranium's symmetry axis; profile: the ear's concha, the darkest spot in the ear zone) and the profile's facing.
- place(plane, marks, height_mm): the plane's centre_px / span_px / span_mm so the crown and chin land on the same world
  heights in every image and the centre column on the crossing line (refplanes.plane_geometry reads these as before).
- signed_distance(matte): the photo outline's signed distance in pixels (outside > 0), once per image.
- silcheck(frame, refs, textures): the silhouette check (C6): every outline pixel of the model in a reference's own
  view, looked up in that photo's signed distance, in mm; coloured green <= 1, amber <= 3, red > 3 mm.

World and image conventions are refplanes.py's: metres, Y up; front plane: image +x -> world +X; side plane: image +x
-> world -Z (a left profile, nose to the image's left), flip_h for a right-facing profile.
"""
import math

import numpy as np
from numba import njit, prange

GREEN, AMBER = 1.0, 3.0          # mm: the silhouette check's colour bands (design 4.3)
SPAN_MARGIN_MM = 12.0            # 11-11: scored from the crown to the chin plus this (the jaw's underside stays in)
HEAD_GUESS_MM = 240.0            # crown to chin when the rig's second mark is not the chin (nose base, pupils, ...)
COL_OUT = (120, 124, 132)        # outline points outside the photos' head span: drawn grey, not scored
COL_OK = (90, 220, 120)
COL_AMBER = (255, 190, 60)
COL_BAD = (255, 70, 60)


# ---- the matte --------------------------------------------------------------------------------------------------------
def _poly_terms(x, y):
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)


def background_model(rgb, border=10, iters=4):
    """-> (B (H, W, 3) float32 background colour field, sigma: the border's noise in colour levels, inlier mask of the
    border). A quadratic per channel over normalised (x, y), fitted first to the top border and the sides' upper halves
    (a subject usually stands at the bottom centre), then refitted over every border pixel that agrees with it."""
    a = rgb[..., :3].astype(np.float32)
    H, W = a.shape[:2]
    b = int(max(2, min(border, H // 8, W // 8)))
    m = np.zeros((H, W), bool)
    m[:b], m[-b:], m[:, :b], m[:, -b:] = True, True, True, True
    ys, xs = np.nonzero(m)
    step = max(1, len(ys) // 40000)
    ys, xs = ys[::step], xs[::step]
    xn, yn = xs / max(W - 1, 1) * 2 - 1, ys / max(H - 1, 1) * 2 - 1
    A = _poly_terms(xn.astype(np.float64), yn.astype(np.float64))
    c = a[ys, xs].astype(np.float64)
    use = (ys < b) | ((ys < H // 2) & ((xs < b) | (xs >= W - b)))
    if use.sum() < 50:
        use = np.ones(len(ys), bool)
    coef = None
    sig = 4.0
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[use], c[use], rcond=None)
        r = np.linalg.norm(c - A @ coef, axis=1)
        med = float(np.median(r[use])) if use.any() else 4.0
        sig = max(1.0, 1.4826 * med)
        use = r < max(4.0 * sig, 6.0)
        if use.sum() < 50:
            break
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    T = _poly_terms(xx / max(W - 1, 1) * 2 - 1, yy / max(H - 1, 1) * 2 - 1)
    B = (T @ coef.astype(np.float32)).astype(np.float32)
    inl = np.zeros((H, W), bool)
    inl[ys[use], xs[use]] = True
    return B, sig, inl


def _local_mean(img, w, size):
    """mean of img over the pixels where w > 0, in a size x size window (img (H, W, C), w (H, W) float)"""
    from scipy import ndimage as ndi
    den = ndi.uniform_filter(w, size)
    out = np.empty_like(img)
    for k in range(img.shape[-1]):
        out[..., k] = ndi.uniform_filter(img[..., k] * w, size) / np.maximum(den, 1e-6)
    return out, den


def matte(rgba, band_px=3):
    """-> dict: alpha (H, W) float32 0..1 (edge-aware), fg (H, W) bool (alpha >= 0.5, the subject only), rgb (H, W, 3)
    uint8 (edge colours unmixed from the background), keyed (bool: a background was found), sigma, ms.
    A photo with no plain background (the border is busy) keeps everything opaque: keyed False."""
    import time
    from scipy import ndimage as ndi
    t0 = time.perf_counter()
    a = rgba[..., :3].astype(np.float32)
    H, W = a.shape[:2]
    B, sig, inl = background_model(rgba)
    d = np.linalg.norm(a - B, axis=-1)
    border_ok = inl.sum() / max(1, 2 * (H + W) * 10) > 0.25
    # the subject differs from a studio ground by colour OR by texture: grey hair and a beard on a dark ground (the
    # Bertillon card) are near its colour but rough (local spread 15-19 levels against the ground's 1.5); each cue is
    # measured against the ground's own level on the border, the larger one decides. The edge band below re-decides
    # the outline by colour, so the texture window's 2 px reach does not widen it.
    L = a.mean(-1)
    m1 = ndi.uniform_filter(L, 5)
    m2 = ndi.uniform_filter(L * L, 5)
    tex = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    lo_c, hi_c = max(4.0 * sig, 10.0), max(8.0 * sig, 26.0)
    mid_c = 0.5 * (lo_c + hi_c)                        # colour: as before (the bust ground drifts up to ~8 levels)
    mid_t = max(4.0 * (float(np.median(tex[inl])) if inl.any() else 1.0), 3.0)   # texture: 4 x the ground's grain
    d = np.maximum(d / mid_c, tex / mid_t)             # >= 1: subject by colour or by texture
    lo, hi = 0.67, 1.33
    if not border_ok:
        return {'alpha': np.ones((H, W), np.float32), 'fg': np.ones((H, W), bool), 'rgb': rgba[..., :3].copy(),
                'keyed': False, 'sigma': sig, 'ms': (time.perf_counter() - t0) * 1000}
    mid = 0.5 * (lo + hi)
    near = d < mid
    lab, n = ndi.label(near)
    seeds = np.unique(lab[inl & near])
    seeds = seeds[seeds > 0]
    bg = np.isin(lab, seeds)
    fg = ~bg
    fg = ndi.binary_opening(fg, iterations=2)
    lab, n = ndi.label(fg)
    if n > 1:
        sizes = ndi.sum(fg, lab, range(1, n + 1))
        fg = lab == (1 + int(np.argmax(sizes)))
    fg = ndi.binary_fill_holes(fg)
    # edge-aware alpha in a band around the boundary
    edge = fg ^ ndi.binary_erosion(fg)
    band = ndi.binary_dilation(edge, iterations=band_px)
    alpha = fg.astype(np.float32)
    rgb = rgba[..., :3].copy()
    if band.any():
        sure_f = ndi.binary_erosion(fg, iterations=band_px + 1).astype(np.float32)
        sure_b = ndi.binary_erosion(~fg, iterations=band_px + 1).astype(np.float32)
        ys, xs = np.nonzero(band)
        y0, y1 = max(0, ys.min() - 24), min(H, ys.max() + 25)
        x0, x1 = max(0, xs.min() - 24), min(W, xs.max() + 25)
        sl = (slice(y0, y1), slice(x0, x1))
        F, wf = _local_mean(a[sl], sure_f[sl], 4 * band_px + 5)
        F2, wf2 = _local_mean(a[sl], sure_f[sl], 12 * band_px + 13)
        F = np.where((wf > 1e-3)[..., None], F, F2)
        Bl, wb = _local_mean(a[sl], sure_b[sl], 4 * band_px + 5)
        Bl = np.where((wb > 1e-3)[..., None], Bl, B[sl])
        I = a[sl]
        FB = F - Bl
        den = np.sum(FB * FB, -1)
        al = np.clip(np.sum((I - Bl) * FB, -1) / np.maximum(den, 1e-6), 0.0, 1.0)
        ok = den > 15.0 ** 2                                  # enough contrast to unmix; else keep the hard edge
        bsl = band[sl]
        alpha_sl = alpha[sl]
        alpha_sl[bsl & ok] = al[bsl & ok]
        # unmix the edge colour: C = B + (I - B) / alpha; faint pixels take the local subject colour
        C = Bl + (I - Bl) / np.maximum(al, 0.05)[..., None]
        C = np.where((al < 0.3)[..., None], F, C)
        C = np.clip(C, 0, 255)
        rs = rgb[sl]
        put = bsl & ok
        rs[put] = C[put].astype(np.uint8)
        alpha[sl] = alpha_sl
        rgb[sl] = rs
    fg = alpha >= 0.5
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.uint8)
    return {'alpha': alpha, 'fg': fg, 'rgb': rgb, 'keyed': True, 'sigma': float(sig), 'lum': lum,
            'ms': (time.perf_counter() - t0) * 1000}


def keyed_rgba(rgba, m):
    """the texture: colour unmixed at the edge, alpha = the matte's alpha times the image's own"""
    out = rgba.copy()
    if not m.get('keyed'):
        return out
    out[..., :3] = m['rgb']
    out[..., 3] = (out[..., 3].astype(np.float32) * m['alpha']).astype(np.uint8)
    return out


def signed_distance(m, open_px=0):
    """-> (H, W) float32: the distance in pixels from the photo's FORM outline, > 0 outside the subject, < 0 inside.
    The alpha's sub-pixel position is used at the edge (the outline sits where alpha = 0.5). open_px > 0 first opens
    the matte with a disc of that radius, so lashes and stray hairs (finer than ~1 mm) are not taken as the form's
    outline (the bust profile's lashes stood 5 mm proud of the eye at the pupil row)."""
    from scipy import ndimage as ndi
    fg = m['fg']
    if open_px and open_px >= 1:
        def disc(r):
            r = max(1, int(round(r)))
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            return (xx * xx + yy * yy) <= r * r
        fg = ndi.binary_opening(fg, structure=disc(open_px))
        lum = m.get('lum')
        if lum is not None:
            # lashes and hair: what a 2.5 mm opening strips AND is dark (well under the subject's own median tone);
            # thin skin-toned forms (ear rims, lips, the nose's wings) stay (the bust profile's far-eye lashes stood
            # 3.5 mm in front of the nose bridge at the pupil row)
            core = ndi.binary_opening(fg, structure=disc(2.5 * open_px))
            thin = fg & ~core
            if thin.any():
                tone = float(np.median(lum[core])) if core.any() else float(np.median(lum[fg]))
                dark = thin & (lum < 0.72 * tone)
                if dark.any():
                    fg = fg & ~ndi.binary_dilation(dark, iterations=1)
                    fg = ndi.binary_opening(fg, structure=disc(1))
        m['form_fg'] = fg                              # the rows (photo_extent_mm) read the same form outline
    dout = ndi.distance_transform_edt(~fg)
    din = ndi.distance_transform_edt(fg)
    sd = np.where(fg, -(din - 0.5), dout - 0.5).astype(np.float32)
    al = np.where(fg, m['alpha'], np.minimum(m['alpha'], 0.0)) if open_px else m['alpha']
    edge = np.abs(sd) < 0.75                       # the first ring on either side: the centre is (0.5 - alpha) px out
    sd[edge] = np.clip(0.5 - al[edge], -0.5, 0.5)
    return sd


# ---- Auto-fit ---------------------------------------------------------------------------------------------------------
def _rows_extent(fg):
    H = fg.shape[0]
    any_ = fg.any(1)
    left = np.where(any_, np.argmax(fg, 1), -1)
    right = np.where(any_, fg.shape[1] - 1 - np.argmax(fg[:, ::-1], 1), -1)
    return any_, left.astype(float), right.astype(float)


def _smooth1(v, s):
    from scipy import ndimage as ndi
    return ndi.gaussian_filter1d(v.astype(float), s, mode='nearest')


def autofit(m, rgba, kind):
    """-> dict(crown [x, y], chin [x, y], centre_x, flip_h (side), head_w_px, notes) in this image's pixels.
    kind 'front' or 'side'. The proposals are where a careful person would click; the user accepts or drags them."""
    fg = m['fg']
    H, W = fg.shape
    any_, L, R = _rows_extent(fg)
    rows = np.nonzero(any_ & ((R - L) >= 3))[0]
    if not len(rows):
        return None
    top = int(rows[0])
    width = np.where(any_, R - L + 1, 0.0)
    lum = (0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]).astype(np.float32)
    # the cranium's breadth: the widest row in the first stretch where the width still grows
    ws = _smooth1(width, 2.0)
    span = max(40, int(0.5 * W))
    seg = ws[top:min(H, top + span)]
    # head breadth: the first local maximum after the width has grown past half its eventual value
    i_max = int(np.argmax(seg[:max(5, int(0.6 * len(seg)))]))
    head_w = float(seg[i_max])
    notes = []
    out = {'kind': kind, 'head_w_px': head_w, 'notes': notes}
    if kind == 'front':
        # height estimate: a head is about 1.5 x its breadth from crown to chin (vertex-menton / eurion breadth)
        Hest = 1.5 * head_w
        r0, r1 = top + int(0.05 * Hest), top + int(0.35 * Hest)
        mids = 0.5 * (L[r0:r1] + R[r0:r1])
        cx = float(np.median(mids)) if len(mids) else 0.5 * W
        # chin: the under-chin shadow, the strongest bright-to-dark step going down the midline
        from scipy import ndimage as ndi
        half = max(3, int(0.06 * head_w))
        col = lum[:, max(0, int(cx) - half):min(W, int(cx) + half + 1)].mean(1)
        col = ndi.gaussian_filter1d(col, max(1.0, 0.006 * Hest))
        g = np.gradient(col)
        a0, a1 = top + int(0.84 * Hest), min(H - 2, top + int(1.16 * Hest))
        if a1 > a0 + 3:
            iy = a0 + int(np.argmin(g[a0:a1]))
            # the chin's edge: the dark contour line where its underside meets the neck's shadow, i.e. the darkest row
            # just below the steepest fall (the bust front: the fall steepest at 841, the contour at 853; the steepest
            # fall put the chin 3 mm high and the pupils 2 mm apart between the photos)
            b1 = min(H - 2, iy + max(4, int(0.05 * Hest)))
            seg_ = col[iy:b1 + 1]
            j = int(np.argmin(seg_))
            if 0 < j < len(seg_) - 1:
                # sub-pixel: the parabola through the minimum and its neighbours
                y0_, y1_, y2_ = seg_[j - 1], seg_[j], seg_[j + 1]
                den = y0_ - 2 * y1_ + y2_
                chin_y = float(iy + j + (0.5 * (y0_ - y2_) / den if den > 1e-6 else 0.0))
            else:
                chin_y = float(iy)
        else:
            chin_y = float(min(H - 1, top + Hest))
            notes.append('chin outside the image: estimated')
        out.update(crown=[cx, float(top)], chin=[cx, chin_y], centre_x=cx, flip_h=False)
        return out
    # ---- profile ----
    Hest = 1.5 * head_w / 1.25            # a profile's crown-to-back depth is ~1.25 x the frontal breadth
    # facing: the nose is the profile's most forward point, about 60 % down the head; the skull's back peaks higher
    r0, r1 = top, min(H, top + int(1.0 * Hest))
    iL = r0 + int(np.argmin(L[r0:r1]))
    iR = r0 + int(np.argmax(R[r0:r1]))
    # the neck sits behind the face: below the chin the outline's middle moves toward the back
    rn = min(H - 1, top + int(1.25 * Hest))
    mid_head = 0.5 * (L[top + int(0.3 * Hest)] + R[top + int(0.3 * Hest)])
    mid_neck = 0.5 * (L[rn] + R[rn]) if any_[rn] else mid_head
    vote = (1 if iL > iR else -1) + (1 if mid_neck > mid_head else -1)
    facing_left = vote >= 0
    notes.append('profile faces %s (vote %d)' % ('left' if facing_left else 'right', vote))
    if facing_left:
        front = -L                         # larger = more forward
        back = R
    else:
        front = R.copy()
        back = -L
    fs = _smooth1(np.where(any_, front, -1e4), 1.5)
    # nose tip: the most forward point between 45 % and 75 % of the head
    n0, n1 = top + int(0.45 * Hest), min(H - 1, top + int(0.78 * Hest))
    nose = n0 + int(np.argmax(fs[n0:n1]))
    # chin: below the lips the outline holds forward (the chin), then its underside turns back: the chin's lowest point
    # is the first row where the outline retreats faster than 1.2 px per row (under ~40 deg from level)
    c0, c1 = nose + int(0.18 * Hest), min(H - 2, nose + int(0.62 * Hest))
    chin_y = float(min(H - 1, top + Hest))
    if c1 > c0 + 4:
        pog = c0 + int(np.argmax(fs[c0:c1]))          # the chin's most forward point (pogonion)
        dr = -np.gradient(fs)                         # retreat per row
        k = pog
        while k < c1 and dr[k] < 1.2:
            k += 1
        chin_y = float(k)
        out['pogonion_y'] = float(pog)
    out['nose_y'] = float(nose)
    # the ear canal: the concha's shadow behind the tragus, found as the darkest blob (darker than its surroundings at
    # the ear's scale) in the ear zone: 50-72 % of the crown-to-chin height, 50-82 % of the depth from the face to the
    # back of the head. The shadow under the lobe (lower) and the hairline (an edge, not a blob) score lower.
    from scipy import ndimage as ndi
    Hh = max(10.0, chin_y - top)
    e0, e1 = top + int(0.50 * Hh), min(H - 1, top + int(0.72 * Hh))
    ec = None
    if e1 > e0 + 4:
        fr = float(np.median(np.where(facing_left, L[e0:e1], R[e0:e1])))
        bk = float(np.median(np.where(facing_left, R[e0:e1], L[e0:e1])))
        lo_, hi_ = sorted((fr + 0.50 * (bk - fr), fr + 0.82 * (bk - fr)))
        x0, x1 = int(max(0, lo_)), int(min(W - 1, hi_))
        if x1 > x0 + 4:
            s1, s2 = max(1.5, 0.012 * Hh), max(4.0, 0.045 * Hh)
            dog = ndi.gaussian_filter(lum, s2) - ndi.gaussian_filter(lum, s1)       # > 0: darker than around it
            win = dog[e0:e1, x0:x1].copy()
            win[~fg[e0:e1, x0:x1]] = -1e9
            j = np.unravel_index(int(np.argmax(win)), win.shape)
            out['ear_contrast'] = float(win[j])
            if win[j] >= 4.0:                    # a real hollow is several grey levels darker than around it
                ec = [float(x0 + j[1]), float(e0 + j[0])]
            else:
                notes.append('no ear hollow found (contrast %.1f): the canal placed at 64 %% of the depth' % win[j])
                ec = [float(fr + 0.64 * (bk - fr)), float(top + 0.61 * Hh)]
    if ec is None:
        ec = [float(0.5 * (L[top + int(0.5 * Hest)] + R[top + int(0.5 * Hest)])), float(top + 0.5 * Hest)]
        notes.append('ear canal not found: the head\'s middle column')
    out['ear'] = ec
    crown_x = float(0.5 * (L[top] + R[top]))
    chin_x = float(-fs[int(chin_y)]) if facing_left else float(fs[int(chin_y)])
    out.update(crown=[crown_x, float(top)], chin=[chin_x, chin_y], centre_x=ec[0], flip_h=not facing_left)
    return out


# ---- placing the planes -----------------------------------------------------------------------------------------------
SECOND_MARKS = ('chin', 'nose base', 'pupils', 'mouth', 'ear lobe')
CROWN_Y_MM = 115.0       # a head's crown sits at +115 mm (half of a 230 mm crown-to-chin) when the chin is not used


def place(p, marks, height_mm):
    """Set a RefPlane's centre and scale from its marks: the crown and the second mark (marks['second'], default the
    chin; the nose base, pupils, mouth or ear lobe when a beard or a collar hides the chin) are height_mm apart, the
    centre column sits on the crossing line. Height: with the chin, the crown-chin middle sits on the anchor's height
    (the bust sphere's centre); with another second mark, the crown sits at +CROWN_Y_MM. -> mm per pixel."""
    yc, yn = float(marks['crown'][1]), float(marks['chin'][1])
    if abs(yn - yc) < 2:
        raise ValueError('the crown and the second mark are less than 2 px apart')
    p.span_px = [round(yc, 2), round(yn, 2)]
    p.span_mm = float(height_mm)
    mpp = float(height_mm) / abs(yn - yc)
    if marks.get('second', 'chin') == 'chin':
        pcy = 0.5 * (yc + yn)
    else:
        pcy = yc + CROWN_Y_MM / mpp
    p.centre_px = [round(float(marks['centre_x']), 2), round(pcy, 2)]
    if 'flip_h' in marks and p.name == 'side':
        p.flip_h = bool(marks['flip_h'])
    return mpp


def world_y_mm(p, anchor_mm, iy):
    """an image row -> its world height in mm on this plane"""
    rows = abs(p.span_px[1] - p.span_px[0]) if p.span_px else None
    mpp = float(p.span_mm) / max(rows, 1.0) if rows else None
    pcy = p.centre_px[1] if p.centre_px else None
    if mpp is None or pcy is None:
        return None
    return float(anchor_mm[1]) + float(p.offset_mm[1]) + (pcy - float(iy)) * mpp


def image_row(p, anchor_mm, y_mm):
    rows = abs(p.span_px[1] - p.span_px[0]) if p.span_px else None
    if not rows or not p.centre_px:
        return None
    mpp = float(p.span_mm) / rows
    return p.centre_px[1] - (float(y_mm) - float(anchor_mm[1]) - float(p.offset_mm[1])) / mpp


# ---- the silhouette check (C6) ---------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _hit_box(obj):
    """-> (y0, y1, x0, x1) of the hit pixels (obj >= 0), y1 = 0 when none: each row scanned from both ends"""
    h, w = obj.shape
    lo = np.full(h, w, np.int64)
    hi = np.full(h, -1, np.int64)
    for j in range(h):
        for i in range(w):
            if obj[j, i] >= 0:
                lo[j] = i
                break
        if lo[j] < w:
            for i in range(w - 1, -1, -1):
                if obj[j, i] >= 0:
                    hi[j] = i
                    break
    y0, y1, x0, x1 = h, 0, w, 0
    for j in range(h):
        if hi[j] >= 0:
            y0 = min(y0, j)
            y1 = max(y1, j + 1)
            x0 = min(x0, lo[j])
            x1 = max(x1, hi[j] + 1)
    return y0, y1, x0, x1


@njit(cache=True, nogil=True)
def _boundary(obj, y0, y1, x0, x1):
    """the model's outline in the hit mask (obj >= 0): every hit pixel with an empty 4-neighbour, once per empty
    neighbour, inside rows y0..y1 and columns x0..x1 -> (x, y, dx, dy) int32 arrays. One pass in two sweeps (count, fill)
    over the model's bounding box (fs-shell 11-11: four full-frame np.nonzero passes took 6.5 of the check's 8.5 ms).
    Serial: it runs in the GUI thread beside the engine's parallel kernels (see refplanes.blend_kernel)."""
    h, w = obj.shape
    nrow = y1 - y0
    cnt = np.zeros(nrow + 1, np.int64)
    for jj in range(nrow):
        j = y0 + jj
        c = 0
        for i in range(x0, x1):
            if obj[j, i] >= 0:
                if i + 1 < w and obj[j, i + 1] < 0:
                    c += 1
                if i > 0 and obj[j, i - 1] < 0:
                    c += 1
                if j + 1 < h and obj[j + 1, i] < 0:
                    c += 1
                if j > 0 and obj[j - 1, i] < 0:
                    c += 1
        cnt[jj + 1] = c
    for jj in range(nrow):
        cnt[jj + 1] += cnt[jj]
    n = cnt[nrow]
    xs = np.empty(n, np.int32)
    ys = np.empty(n, np.int32)
    dxs = np.empty(n, np.int32)
    dys = np.empty(n, np.int32)
    for jj in range(nrow):
        j = y0 + jj
        k = cnt[jj]
        for i in range(x0, x1):
            if obj[j, i] >= 0:
                if i + 1 < w and obj[j, i + 1] < 0:
                    xs[k] = i; ys[k] = j; dxs[k] = 1; dys[k] = 0; k += 1
                if i > 0 and obj[j, i - 1] < 0:
                    xs[k] = i; ys[k] = j; dxs[k] = -1; dys[k] = 0; k += 1
                if j + 1 < h and obj[j + 1, i] < 0:
                    xs[k] = i; ys[k] = j; dxs[k] = 0; dys[k] = 1; k += 1
                if j > 0 and obj[j - 1, i] < 0:
                    xs[k] = i; ys[k] = j; dxs[k] = 0; dys[k] = -1; k += 1
    return xs, ys, dxs, dys


def _facing_plane(refs, cam_f, sizes, min_cos=0.94):
    from . import refplanes as RP
    best = None
    for p in refs.planes:
        if not p.visible or p.name not in sizes:
            continue
        n = RP.AXES.get(p.name, RP.AXES['front'])[0]
        c = abs(float(cam_f @ n))
        if c >= min_cos and (best is None or c > best[0]):
            best = (c, p)
    return None if best is None else best[1]


def silcheck(frame, refs, textures, rows_mm=None):
    """-> dict or None: for the plane the view faces (within ~20 deg), the model's outline pixels (the boundary between
    the hit mask and empty pixels), each placed half-way between the inside and the outside pixel, projected onto the
    plane, looked up in the photo's signed distance and turned into mm.
    Returns {'plane', 'px' (N, 2) frame pixel coordinates, 'mm' (N,) signed (> 0: the model is outside the photo's
    outline), 'mean', 'p90', 'max', 'n', 'rows': per height y (mm) the outer-left / outer-right errors}."""
    from . import refplanes as RP
    from .raymarch import _cam_array
    obj = frame.get('obj')
    if obj is None:
        return None
    cam = frame['cam']
    f = np.asarray(cam.basis()[0], float)
    sizes = {k: (getattr(t, 'W0', None), getattr(t, 'H0', None)) for k, t in textures.items()
             if getattr(t, 'sd', None) is not None}
    p = _facing_plane(refs, f, sizes)
    if p is None:
        return None
    tex = textures[p.name]
    sd = tex.sd
    W0, H0 = tex.W0, tex.H0
    sh, sw = sd.shape
    sx_, sy_ = sw / float(W0), sh / float(H0)        # the matte may be at the texture's (downscaled) size
    h, w = np.asarray(obj).shape[:2]
    # boundary pairs: an inside pixel with an outside 4-neighbour. Each gives one outline point: the inside pixel's hit
    # (its ray and depth) moved half a pixel outward at that depth, then projected ORTHOGRAPHICALLY onto the plane, so
    # the lens's perspective (the 5 deg lens still magnifies a point 40 mm nearer by ~1 %) does not enter the check
    depth = frame.get('depth')
    if depth is not None:
        depth = np.asarray(depth)
        if depth.shape != (h, w):
            depth = None
    objc = np.ascontiguousarray(obj)
    by0, by1, bx0, bx1 = _hit_box(objc)
    if by1 <= by0:
        return None
    xs_, ys_, dxs_, dys_ = _boundary(objc, by0, by1, bx0, bx1)
    if not len(xs_):
        return None
    Pin = np.stack([xs_, ys_], 1).astype(np.int64)
    Dir = np.stack([dxs_, dys_], 1).astype(np.float64)
    P = Pin + 0.5 * Dir                                   # the outline in frame pixels (for drawing)
    ca = _cam_array(cam)
    th = ca[12]
    g = RP.plane_geometry(p, refs.anchor_mm, W0, H0)
    if depth is not None:
        tt = depth[Pin[:, 1], Pin[:, 0]].astype(np.float64)
        okd = np.isfinite(tt) & (tt > 0)
        px = (2.0 * (Pin[:, 0] + 0.5) / w - 1.0) * th * w / h
        py = (1.0 - 2.0 * (Pin[:, 1] + 0.5) / h) * th
        if len(ca) > 18 and ca[16] > 0.5:                    # orthographic: parallel rays (Camera.ray)
            oD, oB = ca[17], ca[18]
            O = ca[0:3][None] + (ca[6:9][None] * px[:, None] + ca[9:12][None] * py[:, None]) * oD - ca[3:6][None] * oB
            X = O + ca[3:6][None] * np.where(okd, tt, 0.0)[:, None]
            pix = np.full(len(tt), 2.0 * th * oD / h)
        else:
            D = ca[3:6][None] + ca[6:9][None] * px[:, None] + ca[9:12][None] * py[:, None]
            Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
            X = ca[0:3][None] + Dn * np.where(okd, tt, 0.0)[:, None]
            pix = np.where(okd, tt, 0.0) * (D @ ca[3:6]) / np.linalg.norm(D, axis=1) * 2.0 * th / h   # a pixel there
        X = X + 0.5 * pix[:, None] * (Dir[:, :1] * ca[6:9][None] - Dir[:, 1:] * ca[9:12][None])
        Q = X - g['c'][None]
        Q = Q - (Q @ g['n'])[:, None] * g['n'][None]          # onto the plane, along its normal
        ok = okd
        t = np.ones(len(Q))
    else:
        px = (2.0 * (P[:, 0] + 0.5) / w - 1.0) * th * w / h
        py = (1.0 - 2.0 * (P[:, 1] + 0.5) / h) * th
        D = ca[3:6][None] + ca[6:9][None] * px[:, None] + ca[9:12][None] * py[:, None]
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        den = D @ g['n']
        ok = np.abs(den) > 1e-9
        t = ((g['c'] - ca[0:3]) @ g['n']) / np.where(ok, den, 1.0)
        Q = ca[0:3][None] + D * t[:, None] - g['c'][None]
    ix = g['pcx'] + (Q @ g['u']) / g['mpp'] - 0.5
    iy = g['pcy'] - (Q @ g['v']) / g['mpp'] - 0.5
    # bilinear lookup in the (possibly downscaled) signed distance
    fx = (ix + 0.5) * sx_ - 0.5
    fy = (iy + 0.5) * sy_ - 0.5
    inside = ok & (t > 0) & (fx >= 0) & (fy >= 0) & (fx <= sw - 1) & (fy <= sh - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, sw - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, sh - 2)
    ax, ay = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    v = (sd[y0, x0] * (1 - ax) * (1 - ay) + sd[y0, x0 + 1] * ax * (1 - ay) + sd[y0 + 1, x0] * (1 - ax) * ay
         + sd[y0 + 1, x0 + 1] * ax * ay)
    mm = v / sx_ * g['mpp'] * 1000.0              # signed distance in original pixels -> mm
    P, mm = P[inside], mm[inside]
    W = (g['c'][None] + Q[inside]).astype(np.float64)
    # 11-11: only the form the photos show is scored, the plane's crown to its chin plus a
    # margin; the model's own cut edge below (a neck stub) is not a disagreement with the photo
    span = head_span_mm(p, refs.anchor_mm) if getattr(refs, 'sil_span', 'head') == 'head' else None
    P_out = np.zeros((0, 2), np.float32)
    if span is not None:
        yw = W[:, 1] * 1000.0
        keep = (yw >= span[0]) & (yw <= span[1])
        P_out = P[~keep].astype(np.float32)
        P, mm, W = P[keep], mm[keep], W[keep]
    if not len(mm):
        return None
    am = np.abs(mm)
    res = {'plane': p.name, 'px': P.astype(np.float32), 'mm': mm.astype(np.float32), 'n': int(len(mm)),
           'mean': float(am.mean()), 'p90': float(np.percentile(am, 90)), 'max': float(am.max()),
           'within1': float((am <= GREEN).mean()), 'world': W, 'px_out': P_out, 'n_out': int(len(P_out)),
           'span_mm': None if span is None else [round(float(span[0]), 1), round(float(span[1]), 1)]}
    if rows_mm is not None:
        res['rows'] = rows_by_height(res, p, rows_mm)
    return res


def head_span_mm(p, anchor_mm, margin_mm=SPAN_MARGIN_MM):
    """-> (lowest, highest) world height in mm the check scores on this plane: from its chin less the margin to its
    crown plus the margin (the crown less HEAD_GUESS_MM when the second mark is not the chin); None without a rig"""
    if not p.span_px or not p.centre_px:
        return None
    yc = world_y_mm(p, anchor_mm, p.span_px[0])
    yn = world_y_mm(p, anchor_mm, p.span_px[1])
    if yc is None or yn is None:
        return None
    second = (getattr(p, 'marks', None) or {}).get('second', 'chin')
    chin = yn if second == 'chin' else yc - HEAD_GUESS_MM
    return (min(chin, yc) - margin_mm, max(chin, yc) + margin_mm)


def photo_extent_mm(p, anchor_mm, fg, alpha, y_mm, scale=1.0):
    """the photo's outline at world height y_mm on plane p, along the plane's u axis (front: world X; side: world -Z, or
    +Z mirrored): -> (lo, hi) in mm of the plane coordinate x_u (world X on the front plane, -Z on the side plane), or
    None. Sub-pixel: the outermost pixel's alpha. fg / alpha are at the matte's size (scale = matte px per image px)."""
    rows = abs(p.span_px[1] - p.span_px[0]) if p.span_px else None
    if not rows or not p.centre_px:
        return None
    mpp = float(p.span_mm) / rows
    iy = p.centre_px[1] - (float(y_mm) - float(anchor_mm[1]) - float(p.offset_mm[1])) / mpp
    j = int(round((iy + 0.5) * scale - 0.5))
    if j < 0 or j >= fg.shape[0] or not fg[j].any():
        return None
    xs = np.nonzero(fg[j])[0]
    i0, i1 = int(xs.min()), int(xs.max())
    W = fg.shape[1]
    # coverage: the edge sits at the outer pixel's far side minus the coverage of it and of its outer neighbour
    a = alpha[j]
    e0 = (i0 + 0.5) - float(a[i0]) - (float(a[i0 - 1]) if i0 > 0 else 0.0)
    e1 = (i1 - 0.5) + float(a[i1]) + (float(a[i1 + 1]) if i1 + 1 < W else 0.0)
    x0 = (e0 + 0.5) / scale - 0.5                       # matte px -> image px
    x1 = (e1 + 0.5) / scale - 0.5
    off = float(p.offset_mm[0]) if p.name == 'front' else (-float(p.offset_mm[2]) * (-1.0 if p.flip_h else 1.0))
    lo = (x0 - p.centre_px[0]) * mpp + off
    hi = (x1 - p.centre_px[0]) * mpp + off
    return lo, hi


def rows_by_height(res, p, rows_mm, tol_mm=0.6, matte=None, anchor_mm=(0.0, 0.0, 0.0), scale=1.0):
    """per height y (mm): the model's outline points within tol of that height at the plane's two extremes ('hi' =
    image right: world +X on the front plane; world -Z (the back of a left profile) on the side plane; 'lo' = image left)
    with the check's nearest-distance error (mm); with the photo's matte, the photo's outline on that row and the
    horizontal (row) error, the measure the 11-10 harness used: [{'y', 'hi_mm', 'hi_err', 'lo_mm', 'lo_err',
    'photo_lo', 'photo_hi', 'hi_row', 'lo_row'}] in the plane's own x_u coordinate (mm)."""
    Wd = res['world']
    out = []
    g_u = None
    from . import refplanes as RP
    n, u, v = RP.AXES.get(p.name, RP.AXES['front'])
    u = -u if p.flip_h else u
    xu = (Wd - np.asarray(anchor_mm, float)[None] / 1000.0) @ u * 1000.0
    for y in rows_mm:
        sel = np.abs(Wd[:, 1] * 1000.0 - y) <= tol_mm
        if not sel.any():
            out.append({'y': y})
            continue
        coord = xu[sel]
        e = res['mm'][sel]
        i_hi, i_lo = int(np.argmax(coord)), int(np.argmin(coord))
        r = {'y': y, 'hi_mm': round(float(coord[i_hi]), 2), 'hi_err': round(float(e[i_hi]), 2),
             'lo_mm': round(float(coord[i_lo]), 2), 'lo_err': round(float(e[i_lo]), 2)}
        # the outline's position AT y: a line through the extreme side's points within 2 mm of that height
        yy = Wd[:, 1] * 1000.0
        for side, ext in (('hi', coord[i_hi]), ('lo', coord[i_lo])):
            near = (np.abs(yy - y) <= 2.0) & (np.abs(xu - ext) <= 3.0)
            if near.sum() >= 3 and np.ptp(yy[near]) > 0.3:
                A = np.stack([np.ones(near.sum()), yy[near] - y], 1)
                c0, c1 = np.linalg.lstsq(A, xu[near], rcond=None)[0]
                if abs(c1) <= 1.0:                       # a steep outline: the row is well defined
                    r[side + '_mm'] = round(float(c0), 2)
                r[side + '_slope'] = round(float(c1), 2)
        if matte is not None:
            ffg = matte.get('form_fg', matte['fg'])
            ext = photo_extent_mm(p, anchor_mm, ffg, np.where(ffg, matte['alpha'], 0.0), y, scale)
            if ext is not None:
                r['photo_lo'], r['photo_hi'] = round(ext[0], 2), round(ext[1], 2)
                r['hi_row'] = round(r['hi_mm'] - ext[1], 2)
                r['lo_row'] = round(ext[0] - r['lo_mm'], 2)
        out.append(r)
    return out


def draw_silcheck(rgb, res, thick=1, inplace=False):
    """colour the model's outline in the frame by its distance from the photo's outline (green / amber / red); inplace
    draws into rgb itself (the compositor's own buffer)"""
    if res is None:
        return rgb
    out = rgb if inplace else rgb.copy()
    po = res.get('px_out')
    if po is not None and len(po):                  # outside the photos' head span: grey, not scored
        _draw_points(out, np.ascontiguousarray(po, np.float32), np.zeros(len(po), np.float32), int(thick), GREEN,
                     AMBER, np.array(COL_OUT * 3, np.uint8))
    _draw_points(out, np.ascontiguousarray(res['px'], np.float32), np.ascontiguousarray(res['mm'], np.float32),
                 int(thick), GREEN, AMBER, np.array(COL_OK + COL_AMBER + COL_BAD, np.uint8))
    return out


@njit(cache=True, nogil=True)
def _draw_points(out, P, mm, thick, green, amber, cols):
    h, w = out.shape[0], out.shape[1]
    for k in range(P.shape[0]):
        a = abs(mm[k])
        c = 0 if a <= green else (3 if a <= amber else 6)
        x = int(round(P[k, 0]))
        y = int(round(P[k, 1]))
        for dy in range(-thick + 1, thick):
            for dx in range(-thick + 1, thick):
                yy = min(max(y + dy, 0), h - 1)
                xx = min(max(x + dx, 0), w - 1)
                out[yy, xx, 0] = cols[c]
                out[yy, xx, 1] = cols[c + 1]
                out[yy, xx, 2] = cols[c + 2]
