"""On-screen cues (TOOLSET_DESIGN 1.7: "Each tool draws its true reach on the model"), as plain data the window draws.

Each function returns a list of cue dicts in WORLD coordinates (metres); the window (fs-shell's) draws them over the
frame with the style named:
    {'kind': 'polyline', 'points': [[x, y, z], ...], 'closed': bool, 'style': ...}
    {'kind': 'disc', 'centre', 'normal', 'radius', 'style'}         a ring or lens disc lying in a plane
    {'kind': 'quad', 'points': 4 corners, 'style'}                    a translucent sheet (the cut blade)
    {'kind': 'point', 'at', 'style'}                                  a contact point
    {'kind': 'text', 'at', 'text', 'style'}                           a live number by the cursor
Styles: 'ring_inner', 'ring_outer', 'lens', 'column', 'blade', 'blade_through', 'contact', 'gap', 'outline', 'path',
'far_face', 'axis', 'ghost', 'label'. Every cue is computed from the field, never guessed: what is drawn is the true
reach.
"""
import math

import numpy as np

from ..warp.lens import depth_images, lens_frame
from ..warp.bake import object_field
from . import common as C
from .contact import touching
from .connect import _frame_for, profile_sdf


def _circle(c, n, r, k=64):
    n, e1, e2 = lens_frame(n)
    t = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False)
    return (np.asarray(c, float)[None, :] + r * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)).tolist()


def ring(c, n, radius, smoothing):
    """a surface-following tool's two rings (the radius, radius + smoothing) at the press point"""
    return [{'kind': 'disc', 'centre': list(map(float, c)), 'normal': list(map(float, n)), 'radius': float(radius),
             'style': 'ring_inner'},
            {'kind': 'disc', 'centre': list(map(float, c)), 'normal': list(map(float, n)),
             'radius': float(radius + smoothing), 'style': 'ring_outer'}]


def shape_lens(scene, oid, c0, n_into, radius, smoothing, reach='near', res=32):
    """Shape's lens: the disc facing the surface, its column, and for Whole part the far face it will move (outlined
    through the form, with its distance): 'far face 600 mm'"""
    o = scene.get(oid)
    rt = radius + smoothing
    L = depth_images(object_field(scene, o), c0, n_into, 1.15 * rt, res, 'whole' if reach == 'whole' else 'near', rt)
    n = L['n']
    out = [{'kind': 'disc', 'centre': list(map(float, c0)), 'normal': list(map(float, -n)), 'radius': float(radius),
            'style': 'lens'},
           {'kind': 'disc', 'centre': list(map(float, c0)), 'normal': list(map(float, -n)), 'radius': float(rt),
            'style': 'ring_outer'}]
    hit = L['hit']
    if hit.any():
        zfar = np.where(hit, L['zfar'], np.nan)
        deep = float(np.nanmax(zfar))
        out.append({'kind': 'polyline', 'points': [list(map(float, c0)), list(map(float, np.asarray(c0) + n * deep))],
                    'closed': False, 'style': 'column'})
        mid = res // 2
        thick = float(L['zfar'][mid, mid] - L['zf'][mid, mid]) if hit[mid, mid] else float('nan')
        if reach == 'whole' and hit[mid, mid]:
            # the far face's outline: where the part's far face meets the lens's rim, drawn through the form
            u = np.linspace(-1.15 * rt, 1.15 * rt, res)
            ring_pts = []
            for k in range(48):
                a = 2 * math.pi * k / 48
                x, y = rt * math.cos(a), rt * math.sin(a)
                i = int(round((x + 1.15 * rt) / L['pix']))
                j = int(round((y + 1.15 * rt) / L['pix']))
                i, j = min(max(i, 0), res - 1), min(max(j, 0), res - 1)
                if hit[j, i]:
                    ring_pts.append(list(map(float, np.asarray(c0) + x * L['e1'] + y * L['e2'] + L['zfar'][j, i] * n)))
            if ring_pts:
                out.append({'kind': 'polyline', 'points': ring_pts, 'closed': True, 'style': 'far_face'})
            out.append({'kind': 'text', 'at': list(map(float, c0)), 'text': 'far face %.0f mm' % (thick * 1e3),
                        'style': 'label'})
        elif np.isfinite(thick):
            out.append({'kind': 'text', 'at': list(map(float, c0)),
                        'text': 'wall %.0f mm · far face stays' % (thick * 1e3), 'style': 'label'})
    return out


def cut_line(scene, oid, ray_a, ray_b, length):
    """the blade as a translucent sheet from the first surface to Length behind it, and its depth"""
    from .cut import LineCut
    lc = LineCut(oid, ray_a, ray_b, length)
    bl = lc.blade(scene)
    if bl is None:
        return [{'kind': 'text', 'at': list(map(float, ray_a[0])), 'text': 'the line misses the object',
                 'style': 'label'}]
    m, oa, ob, da, db = bl['m'], bl['oa'], bl['ob'], bl['da'], bl['db']
    d0, d1 = bl['d0'], bl['d0'] + length
    q = [oa + da * d0 / (da @ m), oa + da * d1 / (da @ m), ob + db * (d1 - (ob - oa) @ m) / (db @ m),
         ob + db * (d0 - (ob - oa) @ m) / (db @ m)]
    return [{'kind': 'quad', 'points': [list(map(float, p)) for p in q], 'style': 'blade'},
            {'kind': 'point', 'at': list(map(float, bl['first'])), 'style': 'contact'},
            {'kind': 'text', 'at': list(map(float, bl['first'])), 'text': 'cut %.1f mm from the first surface'
             % (length * 1e3), 'style': 'label'}]


def merge(scene, oids, blend=0.0):
    """the contact between each pair (a point, the gap) and whether Merge will refuse"""
    out = []
    objs = [scene.get(i) for i in oids]
    ok = False
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            m, p = touching(scene, objs[i], objs[j])
            if p is None:
                # far apart: the nearest pair of points and the line between them, with the gap
                from .contact import nearest_pair
                gap, pa, pb = nearest_pair(scene, objs[i], objs[j])
                out.append({'kind': 'polyline', 'points': [list(map(float, pa)), list(map(float, pb))], 'closed': False,
                            'style': 'gap'})
                out.append({'kind': 'point', 'at': list(map(float, pa)), 'style': 'gap'})
                out.append({'kind': 'point', 'at': list(map(float, pb)), 'style': 'gap'})
                out.append({'kind': 'text', 'at': list(map(float, 0.5 * (pa + pb))), 'style': 'label',
                            'text': 'gap %.1f mm' % (gap * 1e3)})
                continue
            touch = m <= 1e-4
            ok = ok or touch
            out.append({'kind': 'point', 'at': list(map(float, p)), 'style': 'contact' if touch else 'gap'})
            out.append({'kind': 'text', 'at': list(map(float, p)), 'style': 'label',
                        'text': ('touching' + (' · blend %.1f mm' % (blend * 1e3) if blend > 0 else ' · exact join'))
                        if touch else 'gap %.2f mm' % (2e3 * m)})
    if not ok:
        out.append({'kind': 'text', 'at': list(map(float, C.world_box(objs[0])[1])), 'style': 'label',
                    'text': 'Not touching: Group instead?'})
    return out


def connect(p_a, p_b, profile_a, profile_b=None, twist=0.0, k=96):
    """the two outlines lying across the path at each end, and the path"""
    profile_b = profile_b or profile_a
    pa, pb = np.asarray(p_a, float), np.asarray(p_b, float)
    Fr = _frame_for(pb - pa)
    out = []
    for p, prof, tw in ((pa, profile_a, 0.0), (pb, profile_b, twist)):
        # the outline: the profile's zero line found along rays from its centre
        t = np.linspace(0.0, 2 * math.pi, k, endpoint=False)
        R = 2.0 * max(float(prof.get('size', 0.01)), 1e-4) * (1.5 if prof.get('shape') != 'circle' else 1.0)
        pts = []
        for a in t:
            d = np.array([math.cos(a), math.sin(a)])
            lo, hi = 0.0, R
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if profile_sdf(prof, (mid * d)[None, :])[0] < 0:
                    lo = mid
                else:
                    hi = mid
            r = 0.5 * (lo + hi)
            th = math.radians(tw)
            u, v = r * (math.cos(th) * d[0] - math.sin(th) * d[1]), r * (math.sin(th) * d[0] + math.cos(th) * d[1])
            pts.append(list(map(float, p + u * Fr[:, 0] + v * Fr[:, 2])))
        out.append({'kind': 'polyline', 'points': pts, 'closed': True, 'style': 'outline'})
    out.append({'kind': 'polyline', 'points': [list(map(float, pa)), list(map(float, pb))], 'closed': False,
                'style': 'path'})
    return out


def deform_axis(centre, axis, extent):
    """Bend / Twist / Taper: the axis and its extent as a line with end caps"""
    c, a = np.asarray(centre, float), np.asarray(axis, float) / np.linalg.norm(axis)
    p0, p1 = c + extent[0] * a, c + extent[1] * a
    cap = 0.08 * abs(extent[1] - extent[0])
    return [{'kind': 'polyline', 'points': [list(map(float, p0)), list(map(float, p1))], 'closed': False,
             'style': 'axis'},
            {'kind': 'disc', 'centre': list(map(float, p0)), 'normal': list(map(float, a)), 'radius': cap,
             'style': 'axis'},
            {'kind': 'disc', 'centre': list(map(float, p1)), 'normal': list(map(float, a)), 'radius': cap,
             'style': 'axis'}]


def array_ghosts(scene, oid, kw):
    """where an Array's copies will land (frames: the object's box corners at each copy), before the click"""
    from .copies import ArrayCopies
    a = ArrayCopies(oid, **kw)
    o = scene.get(oid)
    lo, hi = o.local_box()
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    out = []
    for R, t, s in a.placements(scene):
        w = (corners * s) @ R.T + t
        # the box's 12 edges: its two end faces as loops and the 4 edges between them (corners: x, then y, then z)
        out.append({'kind': 'polyline', 'points': w[[0, 1, 3, 2]].tolist(), 'closed': True, 'style': 'ghost'})
        out.append({'kind': 'polyline', 'points': w[[4, 5, 7, 6]].tolist(), 'closed': True, 'style': 'ghost'})
        for i, j in ((0, 4), (1, 5), (2, 6), (3, 7)):
            out.append({'kind': 'polyline', 'points': w[[i, j]].tolist(), 'closed': False, 'style': 'ghost'})
    return out
