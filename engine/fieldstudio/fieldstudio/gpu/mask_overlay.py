"""The Mask brush's paint drawn over the card's frame (11-08 night: the GPU viewport never showed the mask, so the parts
marked with it would have been painted blind). Per pixel on an object: the hit point from the camera ray and the
frame's depth (the tracer's own ray formula), the object's sculpt-layer mask sampled there (program.grid_sample), an
orange tint by the mask value."""
import math

import numpy as np
from numba import njit, prange

from ..core import program as P
from ..core import ops
from ..core import brush as BR

TINT = np.array([255.0, 140.0, 20.0])
SERIAL_PX = 200000  # r: small rectangles run on one thread (30 ms for 17k px in parallel on a busy CPU)
LAST = {}          # diagnostics: the last overlay's rectangle and pass times


@njit(cache=True, parallel=True)
def _overlay(rgb, depth, obj, oids, Rt, T, S, H_, slots_oi, cpos, f, r, u, th, asp, index_flat, mask, oi_tab, out,
             mdir, nrm, has_n, cdot):
    Hh, Ww = depth.shape
    for y in prange(Hh):
        for x in range(Ww):
            for c in range(3):
                out[y, x, c] = rgb[y, x, c]
            o = obj[y, x]
            if o < 0:
                continue
            k = -1
            for j in range(oids.shape[0]):
                if oids[j] == o:
                    k = j
                    break
            if k < 0:
                continue
            t = depth[y, x]
            if not (t < 1e20):
                continue
            sx = (2.0 * (x + 0.5) / Ww - 1.0) * th * asp
            sy = (1.0 - 2.0 * (y + 0.5) / Hh) * th
            dx = f[0] + r[0] * sx + u[0] * sy
            dy = f[1] + r[1] * sx + u[1] * sy
            dz = f[2] + r[2] * sx + u[2] * sy
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            if cpos.shape[0] > 5 and cpos[3] > 0.5:       # fs-shell: the orthographic view (Camera.ray): parallel rays
                px = cpos[0] + (r[0] * sx + u[0] * sy) * cpos[4] - f[0] * cpos[5] + f[0] * t - T[k, 0]
                py = cpos[1] + (r[1] * sx + u[1] * sy) * cpos[4] - f[1] * cpos[5] + f[1] * t - T[k, 1]
                pz = cpos[2] + (r[2] * sx + u[2] * sy) * cpos[4] - f[2] * cpos[5] + f[2] * t - T[k, 2]
            else:
                px = cpos[0] + dx / n * t - T[k, 0]
                py = cpos[1] + dy / n * t - T[k, 1]
                pz = cpos[2] + dz / n * t - T[k, 2]
            lx = (Rt[k, 0, 0] * px + Rt[k, 0, 1] * py + Rt[k, 0, 2] * pz) / S[k]
            ly = (Rt[k, 1, 0] * px + Rt[k, 1, 1] * py + Rt[k, 1, 2] * pz) / S[k]
            lz = (Rt[k, 2, 0] * px + Rt[k, 2, 1] * py + Rt[k, 2, 2] * pz) / S[k]
            if has_n:                      # direction-tagged: the surface's own normal (the G-buffer's C1 normal)
                wx, wy, wz = nrm[y, x, 0], nrm[y, x, 1], nrm[y, x, 2]
                mx = Rt[k, 0, 0] * wx + Rt[k, 0, 1] * wy + Rt[k, 0, 2] * wz
                my = Rt[k, 1, 0] * wx + Rt[k, 1, 1] * wy + Rt[k, 1, 2] * wz
                mz = Rt[k, 2, 0] * wx + Rt[k, 2, 1] * wy + Rt[k, 2, 2] * wz
                m = P.mask_sample_dir(index_flat, mask, mdir, oi_tab, slots_oi[k], H_[k], lx, ly, lz, mx, my, mz, cdot)
            else:
                m = P.grid_sample(index_flat, mask, oi_tab, slots_oi[k], H_[k], lx, ly, lz)
            if m > 0.02:
                a = min(m * 2.0, 1.0) * 0.75          # clearly visible from light paint up (0.6 m was too faint)
                for c in range(3):
                    out[y, x, c] = np.uint8(rgb[y, x, c] * (1.0 - a) + TINT[c] * a)


def _cpos(cam):
    """the camera position for the kernels, with the orthographic triple (flag, D, B) after it (Camera.ortho_vec)"""
    ov = cam.ortho_vec() if hasattr(cam, 'ortho_vec') else np.zeros(3)
    return np.concatenate([np.asarray(cam.pos, float), ov]).astype(np.float64)


def overlay(scene, cam, rgb, depth, obj, normals=None):
    """-> rgb with the mask paint tinted (or rgb itself when no object carries mask paint)"""
    st = scene.store
    objs = [o for o in scene.objects if scene.shown(o) and any(s >= 0 and st.mask[s].any() for s in o.sculpt.slots.values())] \
        if st.has('mask') else []                 # fs-brush 11-12: no mask written anywhere
    if not objs:
        return rgb
    pk = scene.pack()
    n = len(objs)
    Rt = np.zeros((n, 3, 3)); T = np.zeros((n, 3)); S = np.zeros(n); H_ = np.zeros(n)
    oids = np.zeros(n, np.int64); soi = np.zeros(n, np.int64)
    for i, o in enumerate(objs):
        tr = o.transform
        Rt[i] = ops.rot_matrix(tr['rotate']).T
        T[i] = tr['translate']
        S[i] = max(float(tr['scale']), 1e-9)
        H_[i] = float(o.sculpt.h)
        oids[i] = o.id
        soi[i] = scene.index_of(o.id)
    f, r, u = cam.basis()
    # r (app: 130-160 ms a frame over all of her 1.4 M pixels): only the screen rectangle the painted bricks project to
    Hh, Ww = depth.shape
    rect = _paint_rect(scene, objs, cam, Ww, Hh)
    if rect is None:
        return rgb
    x0, y0, x1, y1 = rect
    LAST['rect'] = (x0, y0, x1, y1)
    if (x1 - x0) * (y1 - y0) < Ww * Hh:
        sub = overlay_rect(scene, objs, cam, rgb, depth, obj, normals, rect, pk, Rt, T, S, H_, oids, soi)
        return sub
    out = np.empty_like(rgb)
    if normals is not None and not isinstance(normals, np.ndarray):
        # r2 (app: 120-270 ms a frame with the whole normal buffer fetched): a plain pass finds the painted pixels,
        # then only their normals cross the bus (normals: a callable idx -> (n, 3))
        _overlay(np.ascontiguousarray(rgb), np.ascontiguousarray(depth, np.float32), np.ascontiguousarray(obj).astype(np.int64),
                 oids, Rt, T, S, H_, soi, _cpos(cam), np.asarray(f, float), np.asarray(r, float),
                 np.asarray(u, float), math.tan(math.radians(cam.fov) * 0.5), rgb.shape[1] / float(rgb.shape[0]),
                 pk.index_flat, pk.mask, pk.oi_tab, out, pk.mdir, np.zeros((1, 1, 3), np.float32), False, 0.0)
        hit = np.nonzero(np.any(out != rgb, 2).reshape(-1))[0]
        if not len(hit):
            return out
        nr = np.zeros((rgb.shape[0] * rgb.shape[1], 3), np.float32)
        nr[hit] = normals(hit)
        normals = nr.reshape(rgb.shape[0], rgb.shape[1], 3)
    _overlay(np.ascontiguousarray(rgb), np.ascontiguousarray(depth, np.float32), np.ascontiguousarray(obj).astype(np.int64),
             oids, Rt, T, S, H_, soi, _cpos(cam), np.asarray(f, float), np.asarray(r, float),
             np.asarray(u, float), math.tan(math.radians(cam.fov) * 0.5), rgb.shape[1] / float(rgb.shape[0]),
             pk.index_flat, pk.mask, pk.oi_tab, out, pk.mdir,
             np.ascontiguousarray(normals, np.float32) if normals is not None else np.zeros((1, 1, 3), np.float32),
             normals is not None, math.cos(math.radians(BR.MASK_DIR_CUTOFF_DEG)))
    return out


def _paint_rect(scene, objs, cam, W, H, pad=3):
    """the screen rectangle (x0, y0, x1, y1) the painted bricks of objs project into (None: none in front)"""
    from ..tools.protected_merge import _to_world
    st = scene.store
    pts = []
    for o in objs:
        g = o.sculpt
        ks = [b for b, s in g.slots.items() if s >= 0 and st.mask[s].any()]
        if not ks:
            continue
        K = np.array(ks, float)
        hh = float(g.h)
        cor = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * 8 * hh
        X = (K[:, None] * 8 * hh + cor[None]).reshape(-1, 3)
        pts.append(_to_world(o, X))
    if not pts:
        return None
    P = np.concatenate(pts)
    f, r, u = cam.basis()
    v = P - np.asarray(cam.pos, float)
    z = v @ f
    if getattr(cam, 'ortho', False):
        z = z + cam.ortho_params()[1] + 1e-3      # orthographic rays start B before the camera
    if np.all(z <= 1e-6):
        return None
    if np.any(z <= 1e-6):
        return (0, 0, W, H)                      # bricks around / behind the camera: the whole frame
    th = math.tan(math.radians(cam.fov) * 0.5)
    a = W / float(H)
    if getattr(cam, 'ortho', False):
        z = np.full(len(v), cam.ortho_params()[0])     # fs-shell: orthographic: one scale for every depth
    sx = (v @ r) / (z * th * a); sy = (v @ u) / (z * th)
    xs = (sx + 1.0) * 0.5 * W; ys = (1.0 - sy) * 0.5 * H
    x0 = int(max(0, np.floor(xs.min()) - pad)); x1 = int(min(W, np.ceil(xs.max()) + pad))
    y0 = int(max(0, np.floor(ys.min()) - pad)); y1 = int(min(H, np.ceil(ys.max()) + pad))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def overlay_rect(scene, objs, cam, rgb, depth, obj, normals, rect, pk, Rt, T, S, H_, oids, soi):
    """the overlay over a sub-rectangle: the pixel rays of the full frame, evaluated only inside rect"""
    x0, y0, x1, y1 = rect
    H, W = depth.shape
    f, r, u = cam.basis()
    th = math.tan(math.radians(cam.fov) * 0.5)
    asp = W / float(H)
    # shift the camera's principal point: the sub-image's rays are the full frame's rays at (x0 + x, y0 + y)
    sub_rgb = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    sub_dep = np.ascontiguousarray(depth[y0:y1, x0:x1], np.float32)
    sub_obj = np.ascontiguousarray(obj[y0:y1, x0:x1]).astype(np.int64)
    out_sub = np.empty_like(sub_rgb)
    nrm = np.zeros((1, 1, 3), np.float32)
    has_n = False
    if normals is not None:
        if isinstance(normals, np.ndarray):
            nrm = np.ascontiguousarray(normals[y0:y1, x0:x1], np.float32)
            has_n = True
        else:
            fn0 = _overlay_off_serial if sub_rgb.shape[0] * sub_rgb.shape[1] < SERIAL_PX else _overlay_off
            fn0(sub_rgb, sub_dep, sub_obj, oids, Rt, T, S, H_, soi, _cpos(cam), np.asarray(f, float),
                         np.asarray(r, float), np.asarray(u, float), th, asp, x0, y0, W, H, pk.index_flat, pk.mask, pk.oi_tab,
                         out_sub, pk.mdir, nrm, False, 0.0)
            hit = np.nonzero(np.any(out_sub != sub_rgb, 2))
            if len(hit[0]):
                full_idx = (hit[0] + y0) * W + (hit[1] + x0)
                nr = np.zeros(sub_rgb.shape, np.float32)
                nr[hit] = normals(full_idx)
                nrm, has_n = nr, True
            else:
                out = rgb.copy()
                return out
    import time as _t
    t0 = _t.perf_counter()
    fn = _overlay_off_serial if sub_rgb.shape[0] * sub_rgb.shape[1] < SERIAL_PX else _overlay_off
    fn(sub_rgb, sub_dep, sub_obj, oids, Rt, T, S, H_, soi, _cpos(cam), np.asarray(f, float),
                 np.asarray(r, float), np.asarray(u, float), th, asp, x0, y0, W, H, pk.index_flat, pk.mask, pk.oi_tab,
                 out_sub, pk.mdir, nrm, has_n, math.cos(math.radians(BR.MASK_DIR_CUTOFF_DEG)))
    LAST['pass2_ms'] = (_t.perf_counter() - t0) * 1000
    out = rgb.copy()
    out[y0:y1, x0:x1] = out_sub
    return out


@njit(cache=True, parallel=True)
def _overlay_off(rgb, depth, obj, oids, Rt, T, S, H_, slots_oi, cpos, f, r, u, th, asp, ox, oy, Wf, Hf, index_flat, mask,
                 oi_tab, out, mdir, nrm, has_n, cdot):
    """_overlay over a sub-rectangle at (ox, oy) of a Wf x Hf frame"""
    Hh, Ww = depth.shape
    for y in prange(Hh):
        for x in range(Ww):
            for c in range(3):
                out[y, x, c] = rgb[y, x, c]
            o = obj[y, x]
            if o < 0:
                continue
            k = -1
            for j in range(oids.shape[0]):
                if oids[j] == o:
                    k = j
                    break
            if k < 0:
                continue
            t = depth[y, x]
            if not (t < 1e20):
                continue
            sx = (2.0 * (x + ox + 0.5) / Wf - 1.0) * th * asp
            sy = (1.0 - 2.0 * (y + oy + 0.5) / Hf) * th
            dx = f[0] + r[0] * sx + u[0] * sy
            dy = f[1] + r[1] * sx + u[1] * sy
            dz = f[2] + r[2] * sx + u[2] * sy
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            if cpos.shape[0] > 5 and cpos[3] > 0.5:       # fs-shell: the orthographic view (Camera.ray): parallel rays
                px = cpos[0] + (r[0] * sx + u[0] * sy) * cpos[4] - f[0] * cpos[5] + f[0] * t - T[k, 0]
                py = cpos[1] + (r[1] * sx + u[1] * sy) * cpos[4] - f[1] * cpos[5] + f[1] * t - T[k, 1]
                pz = cpos[2] + (r[2] * sx + u[2] * sy) * cpos[4] - f[2] * cpos[5] + f[2] * t - T[k, 2]
            else:
                px = cpos[0] + dx / n * t - T[k, 0]
                py = cpos[1] + dy / n * t - T[k, 1]
                pz = cpos[2] + dz / n * t - T[k, 2]
            lx = (Rt[k, 0, 0] * px + Rt[k, 0, 1] * py + Rt[k, 0, 2] * pz) / S[k]
            ly = (Rt[k, 1, 0] * px + Rt[k, 1, 1] * py + Rt[k, 1, 2] * pz) / S[k]
            lz = (Rt[k, 2, 0] * px + Rt[k, 2, 1] * py + Rt[k, 2, 2] * pz) / S[k]
            if has_n:
                wx, wy, wz = nrm[y, x, 0], nrm[y, x, 1], nrm[y, x, 2]
                mx = Rt[k, 0, 0] * wx + Rt[k, 0, 1] * wy + Rt[k, 0, 2] * wz
                my = Rt[k, 1, 0] * wx + Rt[k, 1, 1] * wy + Rt[k, 1, 2] * wz
                mz = Rt[k, 2, 0] * wx + Rt[k, 2, 1] * wy + Rt[k, 2, 2] * wz
                m = P.mask_sample_dir(index_flat, mask, mdir, oi_tab, slots_oi[k], H_[k], lx, ly, lz, mx, my, mz, cdot)
            else:
                m = P.grid_sample(index_flat, mask, oi_tab, slots_oi[k], H_[k], lx, ly, lz)
            if m > 0.02:
                a = min(m * 2.0, 1.0) * 0.75
                for c in range(3):
                    out[y, x, c] = np.uint8(rgb[y, x, c] * (1.0 - a) + TINT[c] * a)


@njit(cache=True)
def _overlay_off_serial(rgb, depth, obj, oids, Rt, T, S, H_, slots_oi, cpos, f, r, u, th, asp, ox, oy, Wf, Hf, index_flat, mask,
                 oi_tab, out, mdir, nrm, has_n, cdot):
    """_overlay over a sub-rectangle at (ox, oy) of a Wf x Hf frame"""
    Hh, Ww = depth.shape
    for y in range(Hh):
        for x in range(Ww):
            for c in range(3):
                out[y, x, c] = rgb[y, x, c]
            o = obj[y, x]
            if o < 0:
                continue
            k = -1
            for j in range(oids.shape[0]):
                if oids[j] == o:
                    k = j
                    break
            if k < 0:
                continue
            t = depth[y, x]
            if not (t < 1e20):
                continue
            sx = (2.0 * (x + ox + 0.5) / Wf - 1.0) * th * asp
            sy = (1.0 - 2.0 * (y + oy + 0.5) / Hf) * th
            dx = f[0] + r[0] * sx + u[0] * sy
            dy = f[1] + r[1] * sx + u[1] * sy
            dz = f[2] + r[2] * sx + u[2] * sy
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            if cpos.shape[0] > 5 and cpos[3] > 0.5:       # fs-shell: the orthographic view (Camera.ray): parallel rays
                px = cpos[0] + (r[0] * sx + u[0] * sy) * cpos[4] - f[0] * cpos[5] + f[0] * t - T[k, 0]
                py = cpos[1] + (r[1] * sx + u[1] * sy) * cpos[4] - f[1] * cpos[5] + f[1] * t - T[k, 1]
                pz = cpos[2] + (r[2] * sx + u[2] * sy) * cpos[4] - f[2] * cpos[5] + f[2] * t - T[k, 2]
            else:
                px = cpos[0] + dx / n * t - T[k, 0]
                py = cpos[1] + dy / n * t - T[k, 1]
                pz = cpos[2] + dz / n * t - T[k, 2]
            lx = (Rt[k, 0, 0] * px + Rt[k, 0, 1] * py + Rt[k, 0, 2] * pz) / S[k]
            ly = (Rt[k, 1, 0] * px + Rt[k, 1, 1] * py + Rt[k, 1, 2] * pz) / S[k]
            lz = (Rt[k, 2, 0] * px + Rt[k, 2, 1] * py + Rt[k, 2, 2] * pz) / S[k]
            if has_n:
                wx, wy, wz = nrm[y, x, 0], nrm[y, x, 1], nrm[y, x, 2]
                mx = Rt[k, 0, 0] * wx + Rt[k, 0, 1] * wy + Rt[k, 0, 2] * wz
                my = Rt[k, 1, 0] * wx + Rt[k, 1, 1] * wy + Rt[k, 1, 2] * wz
                mz = Rt[k, 2, 0] * wx + Rt[k, 2, 1] * wy + Rt[k, 2, 2] * wz
                m = P.mask_sample_dir(index_flat, mask, mdir, oi_tab, slots_oi[k], H_[k], lx, ly, lz, mx, my, mz, cdot)
            else:
                m = P.grid_sample(index_flat, mask, oi_tab, slots_oi[k], H_[k], lx, ly, lz)
            if m > 0.02:
                a = min(m * 2.0, 1.0) * 0.75
                for c in range(3):
                    out[y, x, c] = np.uint8(rgb[y, x, c] * (1.0 - a) + TINT[c] * a)
