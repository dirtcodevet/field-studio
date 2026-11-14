"""engine/library/mouth/tongue.py: the tongue as a field (mesh-free). fs-mouth lane, 11-11.

Its free form from the cited sizes (adult women: length tip to vallecula ~68-70 mm, width ~42 mm, height ~29-37 mm,
volume ~69 mL; mouth_teeth_tongue.md D12, mouth_2026-11-11/numbers.json) with its structure: a rounded tip, the dorsum's
median sulcus, the fungiform papillae over the tip and sides (~108 / cm^2 on the tip, 0.5-1 mm domes), the
circumvallate papillae (8-9, ~1.3 mm) in a V in front of the sulcus terminalis, the base turning down toward the
vallecula. At rest in a mouth it fills the oral cavity proper: it is pressed into the palate and the lingual faces of the
teeth and gums, so it takes their imprint and touches them (contact, no overlap, no gap) -- the cause of its rest shape.
Filiform papillae (~0.2 mm high, 0.36 mm wide: finer than the part's lattice) are left to the render-time relief.

Frame (mm): the dentition frame (x her left, y backward, z up, origin at the upper incisal point).

    T = Tongue(settings); f = T.free(P)            # the free tongue
    f = T.rest(P, dentition, gums)                # pressed into the mouth
"""
import numpy as np

from teeth import smax, smin, sstep

DEFAULTS = dict(
    length=68.4,      # tip to vallecula, women [R20 Kadhum 2015: F 68.38 (1.44)]
    width=42.0,       # maximum width [mouth_2026-11-11/numbers.json: ~42 mm]
    height=31.0,      # dorsum to the floor at mid length [R20: F 28.93; numbers.json ~37 thickness incl. floor]
    tip_y=6.0,        # the tip's place behind the upper incisal point (rest: behind the incisors' lingual faces)
    tip_z=-1.5,
    sulcus=0.9,       # the median sulcus's depth (mm, E)
    fungiform=1.0,    # 0..1 their prominence
    seed=0,
    dorsum_rise=23.5, # mm the dorsum rises from the tip to the vault (at rest it meets the palate: VHF 1176-1184, E)
    press=1.0,        # mm the free form is larger than the cavity at rest (its tissue pressed into the teeth and palate)
)


class Tongue:
    def __init__(self, settings=None):
        s = dict(DEFAULTS)
        s.update(settings or {})
        self.s = s
        rng = np.random.default_rng(int(s["seed"]) + 3)
        # fungiform papillae: scattered over the anterior two thirds' tip and borders (dorsum surface coords u, v)
        n = 260
        self.fung = np.stack([rng.uniform(-1, 1, n), rng.uniform(0, 0.62, n), rng.uniform(0.6, 1.0, n)], 1)
        # circumvallate: a V of 9 at ~0.68 of the length
        k = np.arange(9) - 4
        self.cv = np.stack([k * 0.11, 0.66 + 0.035 * np.abs(k)], 1)

    def _frame(self, P):
        """the tongue's own coordinates: t along it (0 tip .. 1 vallecula), its centre line and half sizes there"""
        s = self.s
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        Lt = s["length"]
        t = np.clip((y - s["tip_y"]) / (Lt * 0.78), -0.2, 1.25)
        # the dorsum's midline: rising from the tip to the palate's height, the base curving down behind ~0.75
        zt = s["tip_z"] + s["dorsum_rise"] * sstep(0.0, 0.45, t) - 30.0 * sstep(0.72, 1.25, t) ** 1.3
        hz = s["height"] * (0.28 + 0.72 * sstep(-0.05, 0.42, t)) * (1.0 - 0.25 * sstep(0.85, 1.25, t))
        wx = 0.5 * s["width"] * (0.42 + 0.58 * sstep(-0.12, 0.38, t)) * (1.0 - 0.15 * sstep(0.8, 1.2, t))
        zc = zt - hz * 0.5
        return x, y, z, t, zt, zc, hz, wx

    def free(self, P):
        s = self.s
        x, y, z, t, zt, zc, hz, wx = self._frame(np.asarray(P, float))
        # cross-section: a flattened superellipse, its dorsum fuller than its ventral side
        u = x / wx
        v = (z - zc) / (0.5 * hz)
        p = 2.6
        r = (np.abs(u) ** p + np.abs(v) ** p) ** (1 / p)
        f = (r - 1.0) * np.minimum(wx, 0.5 * hz) * 0.9
        # the tip: rounded off in front of t = 0
        f = smax(f, (s["tip_y"] - y) - 0.0, 6.0)
        # the base: it continues down and back into the floor and the pharynx (cut far behind; a mouth part bonds there)
        f = smax(f, y - (s["tip_y"] + s["length"] * 0.98), 4.0)
        # the median sulcus on the dorsum (front two thirds)
        dorsal = sstep(0.2, 0.7, v)
        f = f + s["sulcus"] * np.exp(-(x / 2.2) ** 2) * dorsal * sstep(0.05, 0.2, t) * sstep(0.75, 0.55, t)
        # fungiform papillae: small domes on the dorsum's tip and borders; circumvallate: larger flat domes in a V
        rel = np.zeros(len(f))
        if s["fungiform"] > 0:
            for (fu, ft, fh) in self.fung:
                if abs(fu) < 0.55 and ft > 0.18:          # mostly the tip and the borders
                    continue
                d = (u - fu) ** 2 * wx ** 2 + ((t - ft) * s["length"] * 0.78) ** 2
                rel += 0.32 * fh * s["fungiform"] * np.exp(-d / 0.16) * dorsal
        for (cu, ct) in self.cv:
            d = (u - cu) ** 2 * wx ** 2 + ((t - ct) * s["length"] * 0.78) ** 2
            rel += 0.45 * np.exp(-d / 0.45) * dorsal
            rel -= 0.35 * np.exp(-((np.sqrt(d) - 0.95) / 0.22) ** 2) * dorsal     # the trench round each
        return f - rel

    def rest(self, P, dent=None, gums_field=None):
        """the tongue at rest in a mouth: its free form (a little larger than the cavity: press) carved by the teeth, the
        gums and the palate, so it touches them all (their imprint), never overlapping"""
        f = self.free(P) - self.s["press"]
        if dent is not None:
            ft, _ = dent.field(P)
            f = np.maximum(f, -ft)
        if gums_field is not None:
            g = gums_field(P)
            f = np.maximum(f, -g)
        return f
