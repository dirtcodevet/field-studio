"""The follicle map (hair lane, 11-10): WHERE hair grows on a body and which way it lies. Mesh-free (P14): samples on the body's
field surface (about 2 mm apart), each carrying a weight per hair kind (engine/hair/kinds.py), follicles per cm2, and a flow
direction in its tangent plane. It is what the follicle brush paints (the area where the hair follicles
are to be is drawn, then the hair is grown out) and what the generator grows from.

    FM = FollicleMap.load(path)                       # or FollicleMap.from_regions(regions.npz, field) for a new body
    FM.paint("scalp", centre, radius, strength=1.0)   # brush: add a kind where the brush touches (falloff smooth)
    FM.erase("scalp", centre, radius)                 # brush: remove it
    FM.comb(centre, radius, direction)                # brush: the lie of the hair (projected into each sample's tangent plane)
    FM.density("scalp", centre, radius, x)            # brush: thin or thicken
    roots, normals, flow, info = FM.follicles("scalp", rng, density_x=0.8)   # -> grow
    FM.save(path)

Follicles are drawn per sample as a Poisson count of density x weight x the sample's area, placed uniformly inside the
sample's tangent disc and re-seated on the field: a weight painted at 0.5 gives half the density there (a soft edge), and no
grid or mesh pattern shows in the roots.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import kinds as HK


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class FollicleMap:
    def __init__(self, Q, N, kinds, W, dens, area, flow=None, meta=None):
        self.Q = np.asarray(Q, np.float64); self.N = np.asarray(N, np.float64); self.kinds = list(kinds)
        self.W = np.asarray(W, np.float32); self.dens = np.asarray(dens, np.float32)       # (m, K) weights, (m, K) follicles / cm2
        self.area = np.asarray(area, np.float32)                                            # (m,) cm2 per sample
        self.flow = np.zeros_like(self.Q) if flow is None else np.asarray(flow, np.float64)
        self.meta = dict(meta or {}); self._tree = None

    # ------------------------------------------------------------------ build / io
    @classmethod
    def from_regions(cls, path, sdf=None, area_cm2=None):
        """the 11-05 region map (fv8_hair_regions.npz: Q, kinds, kind_w, follicles_per_cm2) -> a follicle map; with sdf
        (a field: P (n, 3) -> m) the samples are re-seated on that surface and get its normals"""
        z = np.load(path); Q = (z["Q"] if "Q" in z.files else z["q"]).astype(np.float64); kinds = [str(k) for k in z["kinds"]]; W = z["kind_w"].astype(np.float32)
        dk = np.array([HK.follicles(k)["density"] if HK.follicles(k)["density_unit"] == "1/cm2" else 0.0 for k in kinds], np.float32)
        dens = np.tile(dk, (len(Q), 1))
        fpc = z["follicles_per_cm2"].astype(np.float32) if "follicles_per_cm2" in z.files else None
        if fpc is not None:                                                # the region's measured density where the map has one
            main = W.argmax(1); dens[np.arange(len(Q)), main] = np.where(fpc > 0, fpc, dens[np.arange(len(Q)), main])
        N = np.zeros_like(Q)
        if sdf is not None:
            Q, N = reseat(sdf, Q)
        if not area_cm2:                                                    # the region map's own area per sample (its json)
            js = os.path.splitext(path)[0] + ".json"
            area_cm2 = json.load(open(js, encoding="utf-8")).get("area_per_sample_cm2_E", 0.04) if os.path.exists(js) else 0.04
        area = np.full(len(Q), area_cm2, np.float32)
        return cls(Q, N, kinds, W, dens, area, meta={"from": os.path.basename(path)})

    def save(self, path):
        np.savez_compressed(path, Q=self.Q.astype(np.float32), N=self.N.astype(np.float32), W=self.W.astype(np.float16),
                            dens=self.dens.astype(np.float32), area=self.area, flow=self.flow.astype(np.float32),
                            kinds=np.array(self.kinds), meta=np.array(json.dumps(self.meta, default=str)))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["Q"], z["N"], [str(k) for k in z["kinds"]], z["W"], z["dens"], z["area"], z["flow"],
                   json.loads(str(z["meta"])) if "meta" in z.files else None)

    def k(self, kind):
        return self.kinds.index(kind)

    def near(self, centre, radius):
        from scipy.spatial import cKDTree
        if self._tree is None:
            self._tree = cKDTree(self.Q)
        idx = np.asarray(self._tree.query_ball_point(np.asarray(centre, float), radius), int)
        d = np.linalg.norm(self.Q[idx] - centre, axis=1) if len(idx) else np.zeros(0)
        return idx, _smooth(1.0 - d / radius)

    # ------------------------------------------------------------------ the brushes
    def paint(self, kind, centre, radius, strength=1.0):
        i, w = self.near(centre, radius); k = self.k(kind)
        self.W[i, k] = np.clip(self.W[i, k] + strength * w * (1.0 - self.W[i, k]), 0, 1)
        return len(i)

    def erase(self, kind, centre, radius, strength=1.0):
        i, w = self.near(centre, radius); k = self.k(kind)
        self.W[i, k] = np.clip(self.W[i, k] * (1.0 - strength * w), 0, 1)
        return len(i)

    def density(self, kind, centre, radius, x):
        i, w = self.near(centre, radius); k = self.k(kind)
        self.dens[i, k] = self.dens[i, k] * (1.0 + (x - 1.0) * w)
        return len(i)

    def comb(self, centre, radius, direction, strength=1.0):
        i, w = self.near(centre, radius); d = np.asarray(direction, float)
        t = d - self.N[i] * (self.N[i] @ d)[:, None]; t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        f = self.flow[i] * (1 - strength * w)[:, None] + t * (strength * w)[:, None]
        self.flow[i] = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        return len(i)

    def flow_from(self, roots, dirs, k=8):
        """the flow from grown hair (each root's first direction, projected into the tangent plane), inverse-distance over k"""
        from scipy.spatial import cKDTree
        d, j = cKDTree(np.asarray(roots, float)).query(self.Q, k=k); w = 1.0 / np.maximum(d, 1e-4)
        f = (np.asarray(dirs, float)[j] * w[..., None]).sum(1)
        f -= self.N * (self.N * f).sum(1, keepdims=True)
        self.flow = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)

    # ------------------------------------------------------------------ the follicles
    def follicles(self, kind, rng, density_x=1.0, sdf=None, wmin=0.02):
        """-> roots (n, 3), normals (n, 3), flow (n, 3), info. Count per sample ~ Poisson(dens x w x area x density_x)."""
        k = self.k(kind); w = self.W[:, k].astype(np.float64); sel = np.where(w > wmin)[0]
        lam = self.dens[sel, k] * w[sel] * self.area[sel] * density_x
        cnt = rng.poisson(lam); rep = np.repeat(sel, cnt); n = len(rep)
        Q = self.Q[rep]; N = self.N[rep]
        t1 = np.cross(N, np.where(np.abs(N[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])); t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(N, t1); r = np.sqrt(self.area[rep] / np.pi) * 1e-2 * np.sqrt(rng.random(n)); a = 2 * np.pi * rng.random(n)
        P = Q + (np.cos(a) * r)[:, None] * t1 + (np.sin(a) * r)[:, None] * t2
        if sdf is not None:
            P, N = reseat(sdf, P)
        F = self.flow[rep]
        info = {"kind": kind, "n": int(n), "samples": int(len(sel)), "expected": round(float(lam.sum()), 1),
                "area_cm2": round(float((w[sel] * self.area[sel]).sum()), 1)}
        return P, N, F, info


def reseat(sdf, P, it=4, eps=2e-5):
    """points onto the field's zero set (Newton along the gradient) + its outward normals"""
    P = np.asarray(P, np.float64).copy()
    for _ in range(it):
        G = np.stack([(sdf(P + eps * e) - sdf(P - eps * e)) / (2 * eps) for e in np.eye(3)], 1); g = np.linalg.norm(G, axis=1)
        d = sdf(P); ok = np.isfinite(d) & (g > 1e-6) & (np.abs(d) < 0.02)
        P[ok] -= (d[ok] / (g[ok] ** 2))[:, None] * G[ok]
    G = np.stack([(sdf(P + eps * e) - sdf(P - eps * e)) / (2 * eps) for e in np.eye(3)], 1)
    N = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    return P, N
