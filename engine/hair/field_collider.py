"""The body and the scalp as FIELDS for the Studio A growth (generate / head / settle / acts), so a head of hair is grown on the
body the renderer draws, with no mesh anywhere (P14). New 11-10 (hair lane); mesh-free throughout since 11-14
(fs-hairclean: the triangle Surface it once subclassed, mesh.py, is archived; push_out and report live here).

  FieldSurface  (V, VN, tree, sd, sd_exact, report, push_out, sample_area): V = points on the field's surface (a follicle
                map's samples), distances from the field itself (sdf(P) in metres), normals from its gradient.
  FieldSDFGrid  (grid.SDFGrid's: A, lo, h, hi, shape, a, fine, mm, key, report, sample, free_fraction): the lattice filled
                from the field. Where the store holds no brick (deep inside a surface-band store, or far outside) the node's
                side comes from a flood fill from the lattice's boundary (reached = outside), its size from the distance
                to the nearest known node; a solid store needs none of it.
  FieldScalp    (scalp.Scalp's frame: sph, theta_b, d_inside_mm, c, ears, ear_mask, grid, thb, z_min; and V, VN,
                sample_roots, ray_exit): roots drawn from the follicle map's scalp samples (area-weighted, jittered in each
                sample's disc, re-seated on the field), the hairline boundary (grid, thb) and ears taken from the case's
                scalp numbers; the renderer draws her skin, so there is no cap."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter1d, label
from scipy.spatial import cKDTree

from . import grid as GR
from . import scalp as SC

WORKERS = -1


def _grad(sdf, P, eps=2e-5):
    G = np.stack([(sdf(P + eps * e) - sdf(P - eps * e)) / (2 * eps) for e in np.eye(3)], -1)
    return G / np.maximum(np.linalg.norm(G, axis=-1, keepdims=True), 1e-12)


class FieldSurface:
    def __init__(self, sdf, V, VN, area=None):
        """sdf: the field (P (n, 3) m -> signed distance m); V, VN: points on its surface and their unit normals;
        area: each point's share of the surface (m2), else sample_area() estimates it from the points' spacing"""
        self.sdf_fn = sdf
        self.V = np.asarray(V, np.float64); self.VN = np.asarray(VN, np.float64)
        self.tree = cKDTree(self.V); self._area = None if area is None else np.asarray(area, np.float64)

    def _field(self, P):
        P = np.asarray(P, np.float64); return np.asarray(self.sdf_fn(P.reshape(-1, 3)), np.float64)

    def _cloud(self, P):
        """the signed offset from the nearest surface sample along its normal, and that normal (valid anywhere)"""
        _, j = self.tree.query(P, workers=WORKERS); return np.einsum("ij,ij->i", P - self.V[j], self.VN[j]), self.VN[j], j

    def _sd(self, P):
        """the field's distance where it is a distance (finite, not clamped at the band's floor), else the cloud's"""
        P = np.asarray(P, np.float64); shp = P.shape[:-1]; Q = P.reshape(-1, 3); f = self._field(Q); c, _, _ = self._cloud(Q)
        ok = np.isfinite(f) & (f > -0.011)
        return np.where(ok, f, c).reshape(shp)

    def sd(self, P):
        P = np.asarray(P, np.float64); Q = P.reshape(-1, 3)
        s = self._sd(Q); _, j = self.tree.query(Q, workers=WORKERS)
        return s.reshape(P.shape[:-1]), j.reshape(P.shape[:-1])

    def sd_exact(self, P, k=4, chunk=20000):
        """near the skin (|d| < 2 mm, a unit gradient) the field's distance and normal; deeper in the band, past its clamp or
        where a brick is missing, the nearest surface sample's offset and normal (the field's gradient is flat or turned there)"""
        P = np.asarray(P, np.float64).reshape(-1, 3); f = self._field(P); c, Nc, _ = self._cloud(P)
        eps = 2e-5; G = np.stack([(self._field(P + eps * e) - self._field(P - eps * e)) / (2 * eps) for e in np.eye(3)], 1)
        g = np.linalg.norm(G, axis=1); good = np.isfinite(f) & (np.abs(f) < 0.002) & np.isfinite(g) & (g > 0.5) & (g < 2.0)
        N = np.where(good[:, None], G / np.maximum(g, 1e-12)[:, None], Nc); s = np.where(good, f, c)
        return s, P - N * s[:, None], N

    def sample_area(self, k=6):
        """each surface point's share of the skin (m2): the area given, else a hexagonal cell of the mean spacing to
        its k nearest points, (sqrt(3) / 2) d^2"""
        if self._area is None:
            d, _ = self.tree.query(self.V, k=k + 1, workers=WORKERS)
            self._area = np.sqrt(3.0) / 2.0 * d[:, 1:].mean(1) ** 2
        return self._area

    def report(self, pts, skip_roots=True, near_mm=4.0):
        """points inside the skin and the minimum clearance (m) of strands pts (n, K, 3): the nearest surface point's
        offset for all, sd_exact for the points within near_mm of the surface or behind it"""
        n, K = pts.shape[:2]
        k0 = 1 if skip_roots else 0
        Q = np.asarray(pts[:, k0:], np.float64).reshape(-1, 3)
        dist_v, j = self.tree.query(Q, workers=WORKERS)          # one query for both (sd()'s j)
        sd_v = np.einsum("...k,...k->...", Q - self.V[j], self.VN[j])
        near = (dist_v < near_mm / 1000) | (sd_v < 0)
        sd = sd_v.copy()
        if near.any():
            sd[near] = self.sd_exact(Q[near])[0]
        inside = sd < 0
        per = inside.reshape(n, K - k0).any(1)
        return {"points": int(len(Q)), "points_inside": int(inside.sum()), "strands_with_points_inside": int(per.sum()),
                "min_clearance_mm": round(float(sd.min() * 1000), 4),
                "clearance_p01_mm": round(float(np.percentile(sd, 1) * 1000), 4),
                "exact_checked_points": int(near.sum())}, sd.reshape(n, K - k0)

    def push_out(self, P, hmin, iters=4, exact=True, smooth=1.0, fixed_root=True):
        """push the points of strands P (m, K, 3) out to at least hmin (m, scalar or (m, K)) above the surface, along the
        normal there; the push spread along the strand (Gaussian, `smooth` points) so it bends, not kinks. Returns P and
        the count of points that needed it."""
        P = np.array(P, np.float64)
        m, K = P.shape[:2]
        hm = np.broadcast_to(np.asarray(hmin, np.float64), (m, K))
        moved = 0
        for it in range(iters):
            sd_v, j = self.sd(P)
            cand = sd_v < hm + 0.002
            if fixed_root:
                cand[:, 0] = False
            if not cand.any():
                break
            need = np.zeros((m, K))
            Nn = np.zeros((m, K, 3))
            if exact:
                s, _, N = self.sd_exact(P[cand])
                need[cand] = np.maximum(hm[cand] - s, 0)
                Nn[cand] = N
            else:
                need[cand] = np.maximum(hm[cand] - sd_v[cand], 0)
                Nn[cand] = self.VN[j[cand]]
            if it == 0:
                moved = int((need > 1e-6).sum())
            if not (need > 1e-7).any():
                break
            if smooth > 0:
                sm = gaussian_filter1d(need, smooth, axis=1)
                grow = np.maximum(sm - need, 0)
                # spread the push to the neighbours along the normal of the pushed point
                Nsm = gaussian_filter1d(Nn * need[..., None], smooth, axis=1)
                Nsm /= np.maximum(np.linalg.norm(Nsm, axis=2, keepdims=True), 1e-20)
                Nn = np.where((need > 0)[..., None], Nn, Nsm)
                need = need + grow
            if fixed_root:
                need[:, 0] = 0
            P = P + (need * 1.02)[..., None] * Nn
        return P, moved


class FieldSDFGrid(GR.SDFGrid):
    def __init__(self, sdf, lo, hi, mm=3.0, fine=(), fine_mm=2.0, blend_mm=None, chunk=400000, log=print, tag=""):
        t0 = time.time(); log = log or (lambda m: None)
        self.mm = float(mm); self.h = self.mm / 1000.0; lo = np.asarray(lo, np.float64)
        shape = tuple(int(s) for s in np.ceil((np.asarray(hi, np.float64) - lo) / self.h - 1e-9).astype(int) + 1)
        A = self._fill(sdf, lo, self.h, shape, chunk)
        blend_mm = float(blend_mm) if blend_mm is not None else 2 * self.mm
        fines = []
        for b_lo, b_hi in fine:
            b_lo = np.asarray(b_lo, np.float64); hf = float(fine_mm) / 1000.0
            fshape = tuple(int(s) for s in np.ceil((np.asarray(b_hi, np.float64) - b_lo) / hf - 1e-9).astype(int) + 1)
            fines.append(GR._Fine(b_lo, hf, self._fill(sdf, b_lo, hf, fshape, chunk), blend_mm / 1000.0))
        self.A = A; self.lo = lo; self.shape = shape; self.hi = lo + self.h * (np.array(shape) - 1)
        self.a = np.clip(0.5 + A / np.float32(self.h), 0.2, 1.0).astype(np.float32); self.fine = fines
        self.key = hashlib.sha256((tag + json.dumps([lo.round(9).tolist(), list(shape), self.mm])).encode()).hexdigest()[:20]
        self.report = {"key": self.key, "mm": self.mm, "shape": list(shape), "cells": int(np.prod(shape)), "fine_boxes": len(fines),
                       "from": "field", "seconds": round(time.time() - t0, 2), "mb": round(A.nbytes * 2 / 2 ** 20, 1)}
        log("field SDF grid %s at %.1f mm, %d fine boxes, %.1f s" % (list(shape), self.mm, len(fines), self.report["seconds"]))

    @staticmethod
    def _fill(sdf, lo, h, shape, chunk, open_faces=("y-", "y+", "z+"), seal=2):
        """the field on the lattice; then every free region (no brick, or positive) that no open face of the box reaches is
        INSIDE her (a surface-band store reads hollow inside, and the box's bottom face cuts her torso, so the bottom is
        not an outside seed): those nodes get minus their distance to the outside (EDT x h)"""
        n = int(np.prod(shape)); A = np.empty(n, np.float32)
        ii = np.indices(shape).reshape(3, -1).T
        for c0 in range(0, n, chunk):
            P = lo + ii[c0:c0 + chunk] * h; A[c0:c0 + chunk] = sdf(P)
        A = A.reshape(shape); unk = ~np.isfinite(A)
        free = unk | (np.nan_to_num(A, nan=-1) > 0)
        wall = binary_dilation(~free, iterations=seal)                # the skin band thickened: seals a parted mouth, eye gaps
        lab, _ = label(free & ~wall)
        faces = {"x-": lab[0], "x+": lab[-1], "y-": lab[:, 0], "y+": lab[:, -1], "z-": lab[:, :, 0], "z+": lab[:, :, -1]}
        seeds = np.unique(np.concatenate([faces[f].ravel() for f in open_faces]))
        outside = np.isin(lab, seeds[seeds > 0])
        outside = binary_dilation(outside, iterations=seal + 1) & free   # the outside grown back up to the skin
        enclosed = free & ~outside
        dist_out = distance_transform_edt(~outside) * h                  # to the outside region
        dist, ind = distance_transform_edt(unk, return_indices=True)
        near = np.abs(np.nan_to_num(A[tuple(ind)]))
        A = np.where(unk & outside, (dist * h + near).astype(np.float32), A)
        A = np.where(enclosed, -np.maximum(dist_out, h * 0.5).astype(np.float32), A)
        return A.astype(np.float32)


class FieldScalp(SC.Scalp):
    def __init__(self, sdf, fmap, old, kinds=("scalp", "scalp_edge"), wmin=0.02):
        """sdf: the field; fmap: a FollicleMap on that field; old: the case's scalp numbers (anything with c, grid, thb,
        ears, z_min, ear_mask and V: a namespace, or a saved scalp frame) for its centre, hairline boundary (grid, thb),
        ears and z_min"""
        self.sdf_fn = sdf; self.fmap = fmap
        w = sum(fmap.W[:, fmap.k(k)].astype(np.float64) for k in kinds); sel = w > wmin
        near_head = np.linalg.norm(fmap.Q - np.asarray(old.c), axis=1) < 0.16
        self.sel = np.where(sel | (near_head & (fmap.Q[:, 2] > old.c[2] - 0.12)))[0]
        self.V = fmap.Q[self.sel]; self.VN = fmap.N[self.sel]
        self.c = np.asarray(old.c, np.float64); self.ears = dict(old.ears); self.grid = np.asarray(old.grid, np.float64)
        self.thb = np.asarray(old.thb, np.float64); self.z_min = old.z_min
        if old.ear_mask is not None and len(old.ear_mask) == len(old.V):            # the ear mask carried to the field points
            _, j = cKDTree(old.V).query(self.V, workers=WORKERS); self.ear_mask = np.asarray(old.ear_mask)[j]
        else:
            self.ear_mask = np.zeros(len(self.V), bool)
        self.area = fmap.area[self.sel].astype(np.float64) * 1e-4                    # m2 per sample

    def sample_roots(self, n, edge_mm, edge_floor, rng, band=None, outside_mm=0.3, min_cos=0.35, min_r=0.05):
        V, N = self.V, self.VN; din = self.d_inside_mm(V); v = V - self.c; rv = np.linalg.norm(v, axis=1)
        cosv = np.einsum("ij,ij->i", N, v) / np.maximum(rv, 1e-12)
        ok = (cosv > min_cos) & ~self.ear_mask & (rv > min_r) & (V[:, 2] > self.z_min)
        if band is None:
            sel = ok & (din > -outside_mm)
            w = self.area * np.where(sel, edge_floor + (1 - edge_floor) * SC.smoothstep(0.0, max(edge_mm, 1e-6), din), 0.0)
        else:
            sel = ok & (din > band[0]) & (din < band[1]); w = self.area * sel
        cnt = rng.multinomial(n, w / w.sum()); ti = np.repeat(np.arange(len(V)), cnt)
        t1 = np.cross(N[ti], np.where(np.abs(N[ti, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])); t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(N[ti], t1); r = np.sqrt(self.area[ti] / np.pi) * np.sqrt(rng.random(len(ti))); a = 2 * np.pi * rng.random(len(ti))
        X = V[ti] + (np.cos(a) * r)[:, None] * t1 + (np.sin(a) * r)[:, None] * t2
        for _ in range(3):                                                           # re-seat on the field
            G = _grad(self.sdf_fn, X); X = X - G * self.sdf_fn(X)[:, None]
        Nr = _grad(self.sdf_fn, X)
        return X, Nr, float(self.area[sel].sum() * 1e4)

    def ray_exit(self, o, d):
        """the outermost crossing of the ray o + t d out of the field and the field's normal there (field_ray_exit):
        the whorl's ray (follicle.whorl)"""
        return field_ray_exit(self.sdf_fn, o, d)


def install(sdf, lo, hi, log=print):
    """generate / head build their body grid with grid.SDFGrid(surf, mm, cache_dir=..., fine=boxes) (imported inside their
    functions): grid.SDFGrid is replaced by a factory that builds a FieldSDFGrid from the collider's field (same lattice
    spacing and fine boxes, the box lo..hi). engine/hair is mesh-free (11-14): a collider that is not a FieldSurface is
    refused"""
    def factory(surface, mm=3.0, pad_mm=30.0, cache_dir=None, fine=(), fine_mm=2.0, blend_mm=None, **kw):
        if not isinstance(surface, FieldSurface):
            raise TypeError("engine/hair is mesh-free (P14): the body must be a field (field_collider.FieldSurface), "
                            "not %s" % type(surface).__name__)
        return FieldSDFGrid(surface.sdf_fn, lo, hi, mm=mm, fine=fine, fine_mm=fine_mm, blend_mm=blend_mm, log=log)
    GR.SDFGrid = factory
    return factory


def field_ray_exit(sdf, o, d, tmax=0.3, step=0.5e-3):
    """the outermost crossing of the ray o + t d (t > 0) out of the field (inside -> outside), refined by bisection, and the
    field's normal there turned along d. None if the ray never leaves the body within tmax"""
    d = np.asarray(d, np.float64); d = d / np.linalg.norm(d); o = np.asarray(o, np.float64)
    t = np.arange(0.0, tmax, step); v = np.asarray(sdf(o[None] + t[:, None] * d[None]), np.float64); v = np.where(np.isfinite(v), v, 1.0)
    cross = np.nonzero((v[:-1] < 0) & (v[1:] >= 0))[0]
    if len(cross) == 0:
        return None, None
    k = cross[-1]; a, b = t[k], t[k + 1]
    for _ in range(30):
        m = 0.5 * (a + b); fm = sdf((o + m * d)[None])[0]
        if np.isfinite(fm) and fm < 0: a = m
        else: b = m
    P = o + 0.5 * (a + b) * d; n = _grad(sdf, P[None])[0]
    return P, (n if n @ d > 0 else -n)
