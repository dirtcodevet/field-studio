"""engine/library/mouth/jaw.py: the mandible as a field (mesh-free), seated under the lower teeth. fs-mouth lane, 11-11.

Its form from cited adult women's sizes (reference content/mouth_2026-11-11/numbers.json): bigonial width 82-90 mm,
bicondylar 110-115, ramus height (condylion-gonion) 54.2, minimum ramus breadth 29.8, gonial angle ~121 deg, body
length gonion-pogonion 81.6, symphysis 31.8 high x 13.4 thick, the body at the first molar 26.8 high x ~14 thick,
condyle 18.5 x 8.0 mm. Built as: the body swept along the basal arch (chin to the angle) with its height and thickness
profiles; the ramus as a plate whose outline is the posterior border, the gonial angle, the sigmoid notch and the
coronoid process (a 2-D outline: a curve, not a mesh); the condyle on its neck; the mental protuberance; the alveolar
bone round the lower roots (gums.py) joined on top. The teeth are carved out across the periodontal ligament.

Frame (mm): the dentition frame (x her left, y backward, z up, origin at the upper incisal point; teeth in occlusion).
Checks: the condyle's distance from the incisal point (Bonwill's triangle, ~100 mm); sections against the VHF slices.
"""
import numpy as np

from teeth import smax, smin, sstep

DEFAULTS = dict(
    bigonial=86.0,     # [numbers.json: 82.0 (5.5) CT n=202; 89.8 (4.7) n=100] mean of the two
    bicondylar=112.5,  # [110 / 115]
    ramus_h=54.2,      # condylion - gonion
    ramus_b=29.8,      # minimum breadth
    gonial=121.0,      # deg [122.1 / 119.7]
    sym_h=31.8, sym_t=13.4,
    body_h=26.8, body_t=14.0,
    condyle_ml=18.5, condyle_ap=8.0,
    menton_z=-40.0,    # below the upper incisal point (lower incisal edge ~ +1.4; mandibular height ~41 mm bony, E)
    pog_y=-1.0,        # pogonion just ahead of the lower incisors' labial roots (E)
    seed=0,
)


def _spline_closed(V, n=24):
    """a closed Catmull-Rom curve through the control points, sampled n per span (a smooth outline)"""
    V = np.asarray(V, float)
    m = len(V)
    out = []
    for i in range(m):
        p0, p1, p2, p3 = V[i - 1], V[i], V[(i + 1) % m], V[(i + 2) % m]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def _poly_sdf(P2, V):
    """signed distance to a closed 2-D outline (vertices V (n, 2), any winding): neg inside"""
    x, y = P2[:, 0], P2[:, 1]
    d = np.full(len(x), 1e9)
    inside = np.zeros(len(x), bool)
    n = len(V)
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        e = b - a
        w = np.stack([x - a[0], y - a[1]], 1)
        t = np.clip((w @ e) / (e @ e), 0, 1)
        q = w - t[:, None] * e
        d = np.minimum(d, np.hypot(q[:, 0], q[:, 1]))
        c1 = (a[1] > y) != (b[1] > y)
        xc = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1] + 1e-12)
        inside ^= c1 & (x < xc)
    return np.where(inside, -d, d)


class Mandible:
    def __init__(self, settings=None):
        s = dict(DEFAULTS)
        s.update(settings or {})
        self.s = s
        gx = s["bigonial"] / 2
        self.cx = s["bicondylar"] / 2 - s["condyle_ml"] / 2          # the condyle's centre (lateral poles at bicondylar)
        # gonion from the body length (gonion-pogonion) with the lower border rising ~10 mm backward
        pog = np.array([0.0, s["pog_y"], s["menton_z"] + 6.0])
        gz = s["menton_z"] + 10.0
        dy = np.sqrt(max(81.6 ** 2 - gx ** 2 - (gz - pog[2]) ** 2, 100.0))
        self.gon = np.array([gx, pog[1] + dy, gz])
        # the ramus: up and back from gonion at the gonial angle to the mandibular plane
        mp = np.arctan2(self.gon[2] - s["menton_z"], self.gon[1] - (pog[1] + 4.0))      # the plane's rise backward
        ang = np.pi - np.radians(s["gonial"]) + mp                                          # posterior border's angle from +y
        self.rdir = np.array([np.cos(ang), np.sin(ang)])
        self.cond = np.array([self.cx, *(self.gon[1:] + s["ramus_h"] * self.rdir)])
        # the ramus outline in (y, z): posterior border, angle, lower border, retromolar / anterior border, coronoid,
        # sigmoid notch, condylar neck
        g = self.gon[1:]
        c = self.cond[1:]
        r = self.rdir
        nrm = np.array([-r[1], r[0]])                                 # forward (toward -y) in the ramus plane
        b = s["ramus_b"]
        L = s["ramus_h"]
        # control points in the ramus frame (a along the posterior border from gonion, q forward), mm, from the cited
        # sizes: ramus height L, minimum breadth b at mid height, the notch ~13 mm below the condyle's top, the coronoid
        # tip ~33 mm in front of the condyle and ~4 mm lower (E: anatomy texts' proportions)
        # v2 (11-11 23:45): the condylar neck ~9 mm wide (was ~4.5: a mushroom on a stalk), the notch's floor ~12 mm below
        # the condyle, the coronoid tip lower and broader, ~30 mm in front of the condyle
        A = [(-1.0, 6.0), (2.0, 0.5), (12.0, -0.8), (30.0, -0.6), (L - 10.0, -0.4), (L - 4.0, 0.8),       # posterior border
             (L - 4.5, 7.5), (L - 9.0, 10.5), (L - 12.5, 15.0), (L - 9.5, 22.0), (L - 7.5, 28.5),          # neck, notch, coronoid
             (L - 10.5, 31.5), (L - 21.0, b + 1.5), (L - 33.0, b - 0.5), (14.0, b + 3.0), (7.0, b + 9.0),  # anterior border
             (1.5, 22.0), (-2.5, 13.0)]                                                                   # lower border to the angle
        V = [g + r * a_ + nrm * q_ for a_, q_ in A]
        self.ramus_V = _spline_closed(V, 16)
        self.pog = pog

    def _basal(self, s01):
        """the basal arch (body centre line) from the chin (0) to the angle (1): quadratic Bezier in x, y; z of the
        lower border"""
        s = self.s
        P0 = np.array([0.0, s["pog_y"] + s["sym_t"] / 2])
        P3 = np.array([self.gon[0] - 3.0, self.gon[1] - 4.0])
        P1 = np.array([P3[0] * 0.62, P0[1]])                       # a broad chin (U, not V)
        P2 = np.array([P3[0] * 0.98, P0[1] + 0.45 * (P3[1] - P0[1])])
        t = s01[:, None]
        XY = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t * t * P2 + t ** 3 * P3
        return XY

    def field(self, P):
        P = np.asarray(P, float)
        s = self.s
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        ax_ = np.abs(x)
        # ---- body: nearest point on the basal arch (sampled), then a rounded cross-section there
        ts = np.linspace(0, 1, 121)
        C = self._basal(ts)
        d2 = (ax_[:, None] - C[None, :, 0]) ** 2 + (y[:, None] - C[None, :, 1]) ** 2
        k = np.argmin(d2, 1)
        t = ts[k]
        # tangent and outward normal there
        tn = np.gradient(C, axis=0)[k]
        tn /= np.linalg.norm(tn, axis=1, keepdims=True)
        nh = np.stack([tn[:, 1], -tn[:, 0]], 1)                  # outward (buccal)
        rel = np.stack([ax_ - C[k, 0], y - C[k, 1]], 1)
        u = np.sum(rel * nh, 1)                                  # across the body (+ buccal)
        zlow = s["menton_z"] + 10.0 * sstep(0.05, 1.0, t)
        hh = s["sym_h"] + (s["body_h"] - s["sym_h"]) * sstep(0.0, 0.45, t) - 3.0 * sstep(0.6, 1.0, t)
        th = s["sym_t"] + (s["body_t"] - s["sym_t"]) * sstep(0.0, 0.45, t) - 5.0 * sstep(0.65, 1.0, t)
        v = (z - zlow) / hh                                       # 0 lower border .. 1 top of the basal body
        half = 0.5 * th * (0.75 + 0.25 * np.sin(np.pi * np.clip(v * 0.9 + 0.1, 0, 1)))   # the lower border rounded
        body = np.maximum(np.abs(u) - half, np.maximum(-(z - zlow), z - (zlow + hh)))
        body = smax(np.abs(u) - half, smax(-(z - zlow), z - (zlow + hh), 3.0), 2.5)
        body = smax(body, -(t - 0.0) * 1e3 - 0.0, 0.1) if False else body
        # the end at the angle (the ramus takes over): capped by the distance past the arch's last point along its
        # tangent (11-11 21:20: a cap of (t - 1) with t clamped at 1 never cut, so the body ran on behind the angle)
        along = np.sum(rel * tn, 1)
        body = smax(body, np.where(k >= len(ts) - 1, along, -1e3), 2.0)
        # the mental protuberance: a low triangle on the chin's front, lower half
        chin = 2.2 * np.exp(-((x / 9.0) ** 2 + ((z - (s["menton_z"] + 6.0)) / 6.0) ** 2)) * sstep(0.15, -0.05, t)
        body = body - chin
        # ---- ramus: the outline in (y, z), extruded across x at the ramus's own lateral place (flaring up)
        xr = self.gon[0] - 1.5 + (self.cx - self.gon[0] + 1.5) * np.clip((z - self.gon[2]) / (self.cond[2] - self.gon[2]), 0, 1.2)
        d2r = _poly_sdf(np.stack([y, z], 1), self.ramus_V)
        thick = 3.3 + 1.2 * sstep(-6.0, 0.0, -d2r)                   # thicker inside its borders
        ram = smax(d2r, np.abs(ax_ - xr) - thick * 0.5 - 0.0 * thick, 1.6)
        half_t = 2.2 + 1.9 * np.exp(-((d2r + 2.5) / 2.6) ** 2)          # ~4.4 mm in the middle, ~8 at the borders
        # the neck thickens into the condyle (v2): within ~10 mm of the condylion
        dc = np.sqrt((y - self.cond[1]) ** 2 + (z - self.cond[2]) ** 2)
        half_t = half_t + 2.0 * sstep(12.0, 3.0, dc)
        ram = smax(d2r, np.abs(ax_ - xr) - half_t, 1.4)
        # ---- condyle: an ellipsoid (ML x AP x vertical) on the neck, its long axis turned ~15 deg (E)
        cc = self.cond
        q = np.stack([ax_ - cc[0], y - cc[1], z - cc[2]], 1)
        a, b_, c_ = s["condyle_ml"] / 2, s["condyle_ap"] / 2 + 1.0, 5.5
        th_ = np.radians(15.0)
        qx = q[:, 0] * np.cos(th_) - q[:, 1] * np.sin(th_)
        qy = q[:, 0] * np.sin(th_) + q[:, 1] * np.cos(th_)
        con = (np.sqrt((qx / a) ** 2 + (qy / b_) ** 2 + (q[:, 2] / c_) ** 2) - 1.0) * min(a, b_, c_)
        f = smin(body, ram, 4.0)
        f = smin(f, con, 3.0)
        return f

    def landmarks(self):
        return dict(gonion=self.gon.tolist(), condylion=self.cond.tolist(), pogonion=self.pog.tolist(),
                    condyle_to_incisal_mm=float(np.linalg.norm(self.cond)))
