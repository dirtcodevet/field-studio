"""HairSet: grown hair, ready to draw. The generator writes it; the fibre layer (fibre_layer.py) draws it. New 11-10 (hair lane).

One set holds any mix of kinds (brows, lashes, scalp, vellus ...). Each strand is a centreline of NPT points (P14: curves,
no triangles) with, per point, its radius (taper, spindle) and a pigment multiplier along the length (root darkening, a
lighter tip, sun-bleached ends), and per strand its section (ellipticity, twist), its pigment as Chiang's absorption per unit
radius at a reference radius (the kernel scales it by the local radius: the same melanin in a thinner part absorbs less),
the cuticle's longitudinal roughness beta_m, the outward direction of the skin it grows from (what the fibre's lobes see:
the room on one side, her skin on the other), and a seed.

The appearance is DERIVED, never painted: colour = melanin concentrations through the fibre's optics (engine/hair/colour.py,
kinds.pigment); highlights = the cuticle tilt and roughness through the Chiang 2016 lobes; translucency = the same absorption
over the local radius. kinds.py holds the cited traits each value comes from.

    hs = HairSet.empty(); hs.add(kind="eyebrow", P=..., RAD=..., ...); hs.save(path); HairSet.load(path)
"""
from __future__ import annotations

import json
import os

import numpy as np

NPT = 24            # points per strand (resampled by arc length on add)
VERSION = 1

PER_POINT = ("P", "RAD", "PIG")                 # (n, NPT, 3) / (n, NPT) / (n, NPT); optional SIGP (n, NPT, 3): pigment per point
PER_STRAND = ("SIG", "RREF", "ELL", "TAU", "BM", "NRM", "SEED", "KIND", "LEN")


def resample(p, r=None, g=None, npt=NPT):
    """one centreline (m, 3) (+ per-point radius / pigment) -> npt points evenly spaced in arc length (linear between the
    given points: the given points must already be dense enough to carry the curve)"""
    p = np.asarray(p, np.float64)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    L = float(s[-1])
    t = np.linspace(0.0, L, npt)
    q = np.stack([np.interp(t, s, p[:, a]) for a in range(3)], 1)
    rr = None if r is None else np.interp(t, s, np.asarray(r, np.float64))
    gg = None if g is None else np.interp(t, s, np.asarray(g, np.float64))
    return q, rr, gg, L


class HairSet:
    def __init__(self, arrays=None, kinds=None, meta=None):
        self.a = arrays or {}
        self.kinds = list(kinds or [])
        self.meta = dict(meta or {})

    @classmethod
    def empty(cls, npt=NPT):
        a = {"P": np.zeros((0, npt, 3), np.float32), "RAD": np.zeros((0, npt), np.float32), "PIG": np.zeros((0, npt), np.float32),
             "SIG": np.zeros((0, 3), np.float32), "RREF": np.zeros(0, np.float32), "ELL": np.zeros(0, np.float32),
             "TAU": np.zeros(0, np.float32), "BM": np.zeros(0, np.float32), "NRM": np.zeros((0, 3), np.float32),
             "SEED": np.zeros(0, np.uint32), "KIND": np.zeros(0, np.int32), "LEN": np.zeros(0, np.float32)}
        return cls(a)

    @property
    def n(self):
        return int(self.a["P"].shape[0])

    @property
    def npt(self):
        return int(self.a["P"].shape[1])

    def kind_id(self, name):
        if name not in self.kinds:
            self.kinds.append(name)
        return self.kinds.index(name)

    def add(self, kind, P, RAD, SIG, RREF, ELL, TAU, BM, NRM, PIG=None, SEED=None):
        """add strands of one kind. P: list or array of centrelines (each (m, 3), m >= 2), RAD per point (same m) or per
        strand (scalar), PIG per point (default 1). SIG (n, 3) or (3,), the rest per strand or scalar. Resampled to NPT."""
        n = len(P)
        if n == 0:
            return self
        bc = lambda v, d=None: np.broadcast_to(np.asarray(v, np.float64), (n,) + (() if d is None else (d,)))  # noqa: E731
        Ps, Rs, Gs, Ls = [], [], [], []
        for i in range(n):
            p = np.asarray(P[i], np.float64)
            r = np.asarray(RAD[i], np.float64) if np.ndim(RAD[i]) else np.full(len(p), float(RAD[i]))
            g = np.ones(len(p)) if PIG is None else (np.asarray(PIG[i], np.float64) if np.ndim(PIG[i]) else np.full(len(p), float(PIG[i])))
            q, rr, gg, L = resample(p, r, g, self.npt)
            Ps.append(q); Rs.append(rr); Gs.append(gg); Ls.append(L)
        k = self.kind_id(kind)
        seed0 = int(self.a["SEED"].max()) + 1 if self.n else 1
        seed = np.asarray(SEED, np.uint64) if SEED is not None else (np.arange(seed0, seed0 + n, dtype=np.uint64) * 2654435761) % (2 ** 32)
        new = {"P": np.asarray(Ps, np.float32), "RAD": np.asarray(Rs, np.float32), "PIG": np.asarray(Gs, np.float32),
               "SIG": bc(SIG, 3).astype(np.float32), "RREF": bc(RREF).astype(np.float32), "ELL": bc(ELL).astype(np.float32),
               "TAU": bc(TAU).astype(np.float32), "BM": bc(BM).astype(np.float32), "NRM": bc(NRM, 3).astype(np.float32),
               "SEED": np.asarray(seed, np.uint32), "KIND": np.full(n, k, np.int32), "LEN": np.asarray(Ls, np.float32)}
        for key, v in new.items():
            self.a[key] = np.concatenate([self.a[key], v]) if self.n or key not in self.a else v
        return self

    def resampled(self, npt):
        """a copy with every strand resampled to npt points (arc length)"""
        out = HairSet.empty(npt); out.kinds = list(self.kinds); out.meta = dict(self.meta)
        P, R, G, Ls = [], [], [], []
        SP = [] if "SIGP" in self.a else None
        for i in range(self.n):
            q, rr, gg, L = resample(self.a["P"][i], self.a["RAD"][i], self.a["PIG"][i], npt); P.append(q); R.append(rr); G.append(gg); Ls.append(L)
            if SP is not None:
                s0 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.a["P"][i].astype(np.float64), axis=0), axis=1))]
                t = np.linspace(0, s0[-1], npt); SP.append(np.stack([np.interp(t, s0, self.a["SIGP"][i][:, c]) for c in range(3)], 1))
        a = {k: v for k, v in self.a.items() if k not in PER_POINT + ("SIGP", "LEN")}
        a.update(P=np.asarray(P, np.float32).reshape(-1, npt, 3), RAD=np.asarray(R, np.float32).reshape(-1, npt), PIG=np.asarray(G, np.float32).reshape(-1, npt),
                 LEN=np.asarray(Ls, np.float32))
        if SP is not None: a["SIGP"] = np.asarray(SP, np.float32).reshape(-1, npt, 3)
        out.a = a
        return out

    def merge(self, other):
        if other.npt != self.npt:
            other = other.resampled(self.npt) if self.n == 0 or self.npt >= other.npt else other
            if other.npt != self.npt:
                me = self.resampled(other.npt); self.a = me.a
        if ("SIGP" in self.a) != ("SIGP" in other.a):           # per-point pigment on one side only: fill the other from SIG x PIG
            for hs in (self, other):
                if "SIGP" not in hs.a:
                    hs.a["SIGP"] = (hs.a["SIG"][:, None, :] * hs.a["PIG"][:, :, None]).astype(np.float32); hs.a["PIG"] = np.ones_like(hs.a["PIG"])
        for i, nm in enumerate(other.kinds):
            sel = other.a["KIND"] == i
            if not sel.any():
                continue
            k = self.kind_id(nm)
            for key in PER_POINT + PER_STRAND + (("SIGP",) if "SIGP" in other.a else ()):
                v = other.a[key][sel]
                if key == "KIND":
                    v = np.full(int(sel.sum()), k, np.int32)
                self.a[key] = np.concatenate([self.a[key], v]) if key in self.a else v
        self.meta.setdefault("merged", []).append(other.meta)
        return self

    def select(self, kinds):
        ids = [self.kinds.index(k) for k in kinds if k in self.kinds]
        sel = np.isin(self.a["KIND"], ids)
        out = HairSet({k: v[sel] for k, v in self.a.items()}, self.kinds, dict(self.meta, selected=list(kinds)))
        return out

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        hdr = json.dumps({"version": VERSION, "npt": self.npt, "kinds": self.kinds, "meta": self.meta}, default=str)
        np.savez_compressed(path, _header=np.frombuffer(hdr.encode("utf-8"), np.uint8), **self.a)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        hdr = json.loads(bytes(z["_header"]).decode("utf-8"))
        a = {k: z[k] for k in z.files if k != "_header"}
        return cls(a, hdr["kinds"], hdr.get("meta"))

    def stats(self):
        out = {"n": self.n, "kinds": {}}
        for i, nm in enumerate(self.kinds):
            sel = self.a["KIND"] == i
            if not sel.any():
                continue
            L = self.a["LEN"][sel] * 1e3; d = 2e6 * self.a["RAD"][sel][:, 0]
            out["kinds"][nm] = {"n": int(sel.sum()), "len_mm_p10_50_90": np.percentile(L, [10, 50, 90]).round(2).tolist(),
                                "root_d_um_p50": round(float(np.median(d)), 1)}
        return out
