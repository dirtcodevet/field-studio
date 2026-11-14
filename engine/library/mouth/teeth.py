"""engine/library/mouth/teeth.py: human permanent teeth as fields (mesh-free, P14). fs-mouth lane, 11-11.

Each tooth is built as its true form from measured dimensions (outcome first, P16): the crown's cross-sections from the
cemento-enamel junction (CEJ) to the incisal edge / occlusal table, the incisal edge or the cusps, ridges, fossae and
fissures as structure, the roots (one, two or three), and the inner layers that make its look: enamel over dentine (the
dentino-enamel junction, DEJ, with its mamelon lobes under the incisal edge, so the translucent edge comes from the form)
and the pulp. Nothing is painted: a tooth's colour comes from light through these layers (fs-light reads the media).

Units: millimetres. Tooth frame: x mesiodistal (+ distal), y labiolingual (+ labial / buccal), z along the long axis,
z = 0 at the labial CEJ, + toward the incisal edge / occlusal surface, - toward the apex.

    t = Tooth(TYPES["U1"], settings)          # one tooth from its type and its seeded settings
    f = t.fields(P)                           # P (N, 3) mm in the tooth frame -> {"body", "dentin", "pulp"} (mm, neg inside)

The values are approximate distances (zero set and sign right, |grad| ~ 0.6-1.4); sample.py re-distances a lattice.
Sources for every default: `teeth_types.json` beside this file (cited, with q flags M / D / E).
"""
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------------------------------- small helpers
def smax(a, b, r):
    """smooth max: rounds the crossing of two faces by r (polynomial)"""
    if r <= 0:
        return np.maximum(a, b)
    h = np.clip(0.5 + 0.5 * (a - b) / r, 0.0, 1.0)
    return b + (a - b) * h + r * h * (1.0 - h)


def smin(a, b, r):
    return -smax(-a, -b, r)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bump(x, c, w):
    """smooth compact bump: 1 at c, 0 beyond |x - c| >= w"""
    t = np.clip(1.0 - ((x - c) / w) ** 2, 0.0, 1.0)
    return t * t


class Prof:
    """a monotone cubic profile through control points, clamped outside its range"""
    def __init__(self, xs, ys):
        self.x0, self.x1 = xs[0], xs[-1]
        # C2 (no shading bands at the control points); 'natural' ends
        self.f = CubicSpline(np.asarray(xs, float), np.asarray(ys, float), bc_type="natural", extrapolate=False)
        self.y0, self.y1 = ys[0], ys[-1]

    def __call__(self, x):
        x = np.asarray(x, float)
        out = self.f(np.clip(x, self.x0, self.x1))
        return out


def _superellipse(x, y, a_m, a_d, b_f, b_l, p_f, p_l):
    """2-D asymmetric superellipse field: halves with their own half widths (mesial / distal, labial / lingual) and
    exponents (labial / lingual), so an incisor's section is broad labially and triangular lingually. The value is
    (r - 1) / |grad r| (first-order distance, continuous across the halves), mm, neg inside."""
    a = np.where(x > 0, a_d, a_m)
    b = np.where(y > 0, b_f, b_l)
    p = np.where(y > 0, p_f, p_l)
    ax, ay = np.abs(x) / a + 1e-9, np.abs(y) / b + 1e-9
    q = ax ** p + ay ** p
    r = q ** (1.0 / p)
    k = r ** (1.0 - p)
    gx = k * ax ** (p - 1.0) / a
    gy = k * ay ** (p - 1.0) / b
    g = np.sqrt(gx * gx + gy * gy)
    lo_, hi_ = 0.5 / np.maximum(a, b), 2.0 / np.minimum(a, b)
    return (r - 1.0) / np.clip(g, lo_, hi_)


# ---------------------------------------------------------------------------------------------------- the tooth types
def load_types(path=None):
    d = json.load(open(path or os.path.join(HERE, "teeth_types.json")))
    out = {}
    for k, v in d["types"].items():
        out[k] = {kk: (vv["v"] if isinstance(vv, dict) and "v" in vv else vv) for kk, vv in v.items()}
        out[k]["code"] = k
    return out


TYPES = load_types()


# ---------------------------------------------------------------------------------------------------- one tooth
class Tooth:
    """one tooth's field from its type (dimensions) and settings:
      scale       overall size factor (1 = the type's defaults)
      wear        0 = unworn (mamelons, sharp cusps) .. 1 = heavily worn (flat edge / facets); adult default ~0.3
      mamelon     0..1 how much of the three incisal lobes remains on the outer edge (None: from age, contact, wear)
      age         years (default 30). Whether an incisor keeps mamelons is drawn from its seed with the cited share of
                  incisors that have them by age (Alalowi & Al-Jhany 2020, 518 patients, full text: 6-10 y 97.5 %,
                  16-20 y 94.1, 21-25 y 93.1, 26-30 y 75.2, 31-35 y 72, 36-40 y 55, 41-45 y 31.9, 46-50 y 10.8 %);
                  when present, their depth falls with age and wear (E: no depth source found). (11-12 04:30: replaces
                  the 02:06 rule 'gone by ~16 with contact', which was not supported by this source.)
      incisal_contact  1 = the incisors meet their opposites, 0 = open bite (absence 10 % vs 24.5 % with normal contact:
                  the same source)
      carabelli   0..1 the upper first molar's lingual tubercle
      seed        small per-tooth irregularity (cusp heights, groove depths, lobe sizes)
    """

    def __init__(self, T, settings=None):
        s = dict(scale=1.0, wear=0.15, mamelon=None, age=30.0, incisal_contact=1.0, carabelli=0.0, seed=0, enamel=1.0, lobes=1.0)
        s.update(settings or {})
        self.T, self.s = T, s
        sc = s["scale"]
        rng = np.random.default_rng(s["seed"])
        self.rng = rng
        k = T["kind"]
        self.kind = k
        self.H = T["H"] * sc
        self.L = T["L"] * sc
        self.md, self.mdc = T["md"] * sc, T["md_c"] * sc
        self.ll, self.llc = T["ll"] * sc, T["ll_c"] * sc
        self.cej_m, self.cej_d = T["cej_m"] * sc, T["cej_d"] * sc
        self.wear = float(s["wear"])
        if s["mamelon"] is not None:
            self.mam = float(s["mamelon"])
        else:   # (11-12 04:30) by the cited prevalence by age; her own setting is explicit (mamelon 0)
            age = float(s["age"])
            p = float(np.interp(age, [8, 13, 18, 23, 28, 33, 38, 43, 48], [0.975, 0.967, 0.941, 0.931, 0.752, 0.72, 0.55, 0.319, 0.108]))
            p = 1.0 - (1.0 - p) * (1.0 - (1.0 - 0.10 / 0.245) * (1.0 - float(s["incisal_contact"])))
            present = np.random.default_rng([int(s["seed"]), 7177]).random() < p
            depth = float(np.clip(1.0 - (age - 8.0) / 25.0, 0.25, 1.0))
            self.mam = float(np.clip(1.0 - 2.2 * self.wear, 0, 1)) * depth if present else 0.0
        self._build_profiles()
        self._build_top()
        self._build_roots()

    # ---- cross-section profiles along the crown (s = z / H in [0, 1]) and the root (t = -z / L in [0, 1])
    def _build_profiles(self):
        T, k = self.T, self.kind
        md, mdc, ll, llc, H = self.md, self.mdc, self.ll, self.llc, self.H
        if k == "incisor":
            S = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
            r0 = mdc / md
            am = [r0 * md / 2, (r0 + (1 - r0) * 0.62) * md / 2, (r0 + (1 - r0) * 0.9) * md / 2, md / 2, md / 2, 0.99 * md / 2]
            ad = [r0 * md / 2, (r0 + (1 - r0) * 0.58) * md / 2, (r0 + (1 - r0) * 0.88) * md / 2, 0.995 * md / 2, 0.985 * md / 2, 0.955 * md / 2]
            te = T.get("t_edge", 1.1)
            # the proximal outline: the labial face gently convex (height of contour in the cervical third), the lingual
            # face with the cingulum in the cervical third and the shovel (fossa) above it, converging to the edge
            Sy = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.92, 1.0]
            r1 = 0.5 * llc / ll
            yf = np.array([r1, r1 + .03, r1 + .025, r1 + .005, r1 - .035, r1 - .09, r1 - .145, r1 - .19]) * ll
            yl = np.array([-r1, -r1 - .045, -r1 - .02, -.27, -.13, -.02, .06, r1 - .19 - te / ll]) * ll
            if T.get("arch") == "L":     # lower incisors: the edge lingual to the root's bisector, the labial face flatter
                sh = -0.07 * ll * np.array(Sy) ** 2
                yf, yl = yf + sh, yl + sh
            self.pf, self.pl = 2.7, 1.75
        elif k == "canine":
            S = [0.0, 0.2, 0.45, 0.7, 0.85, 1.0]
            am = [mdc / 2, 0.86 * md / 2, 0.97 * md / 2, md / 2, 0.97 * md / 2, 0.80 * md / 2]
            ad = [mdc / 2, 0.84 * md / 2, 0.95 * md / 2, 0.99 * md / 2, 0.93 * md / 2, 0.75 * md / 2]
            te = T.get("t_edge", 1.6)
            Sy = [0.0, 0.18, 0.25, 0.45, 0.65, 0.85, 1.0]
            yf = np.array([0.47, 0.54, 0.545, 0.52, 0.44, 0.32, 0.20]) * ll
            yl = np.array([-0.47, -0.56, -0.58, -0.42, -0.22, -0.02, 0.20 - te / ll]) * ll
            self.pf, self.pl = 2.3, 1.7
        elif k == "premolar":
            S = [0.0, 0.25, 0.55, 0.75, 0.9, 1.0]
            am = [mdc / 2, 0.86 * md / 2, 0.97 * md / 2, md / 2, 0.99 * md / 2, 0.93 * md / 2]
            ad = [mdc / 2, 0.86 * md / 2, 0.97 * md / 2, md / 2, 0.98 * md / 2, 0.92 * md / 2]
            lower = T.get("arch") == "L"
            Sy = [0.0, 0.25, 0.5, 0.8, 1.0]
            if lower:   # lower premolars: crown tilted lingually over the root, small lingual cusp
                yf = np.array([0.46, 0.52, 0.47, 0.30, 0.13]) * ll
                yl = np.array([-0.46, -0.50, -0.50, -0.42, -0.30]) * ll
            else:
                yf = np.array([0.46, 0.505, 0.49, 0.42, 0.34]) * ll
                yl = np.array([-0.46, -0.48, -0.50, -0.44, -0.33]) * ll
            self.pf, self.pl = 2.5, 2.1
        else:   # molar
            S = [0.0, 0.25, 0.55, 0.75, 0.9, 1.0]
            am = [mdc / 2, 0.88 * md / 2, 0.98 * md / 2, md / 2, 0.98 * md / 2, 0.92 * md / 2]
            ad = [mdc / 2, 0.87 * md / 2, 0.97 * md / 2, 0.99 * md / 2, 0.96 * md / 2, 0.88 * md / 2]
            lower = T.get("arch") == "L"
            Sy = [0.0, 0.25, 0.5, 0.8, 1.0]
            if lower:
                yf = np.array([0.45, 0.50, 0.465, 0.36, 0.25]) * ll
                yl = np.array([-0.45, -0.49, -0.50, -0.45, -0.34]) * ll
            else:
                yf = np.array([0.45, 0.50, 0.48, 0.41, 0.33]) * ll
                yl = np.array([-0.45, -0.48, -0.50, -0.45, -0.36]) * ll
            self.pf, self.pl = 3.0, 2.6
        if k != "incisor":
            yf = yf * (llc / ll) ** np.array([1.0] + [0.0] * (len(Sy) - 1))     # the cervical value from ll_c
            yl = yl * (llc / ll) ** np.array([1.0] + [0.0] * (len(Sy) - 1))
        # a type may carry its own measured profiles (fractions of md / ll), fitted to traced references
        if "prof_md" in T:
            S, am_f, ad_f = T["prof_md"]["s"], T["prof_md"]["m"], T["prof_md"]["d"]
            am, ad = [v * md / 2 for v in am_f], [v * md / 2 for v in ad_f]
        if "prof_ll" in T:
            Sy, yf, yl = T["prof_ll"]["s"], np.array(T["prof_ll"]["f"]) * ll, np.array(T["prof_ll"]["l"]) * ll
        self.P_am, self.P_ad = Prof(S, am), Prof(S, ad)
        self.P_yf, self.P_yl = Prof(Sy, yf), Prof(Sy, yl)
        self.shear = float(T.get("shear", 0.0))               # rhomboid occlusal outline (upper molars)
        self.ltaper = float(T.get("ltaper", 0.0))             # the lingual half narrower mesiodistally

    def section_dims(self, z):
        """crown section at height z (mm) -> (a_m, a_d, y_f, y_l) ; below the CEJ the root (trunk) section"""
        H, L = self.H, self.L
        s = np.maximum(z, 0.0) / H
        am, ad, yf, yl = self.P_am(s), self.P_ad(s), self.P_yf(s), self.P_yl(s)
        # root: taper from the cervical section to the apex (convex taper), the apex rounded
        t = np.clip(-z / L, 0.0, 1.0)
        taper = (1.0 - t) ** self.T.get("root_taper", 0.85) * (1 - self.T.get("apex_frac", 0.06)) + self.T.get("apex_frac", 0.06)
        rr = 1.0 - (1.0 - self.T.get("root_md", 0.88)) * sstep(0.0, 3.0, -z)   # the root's MD narrower, reached 3 mm below the CEJ
        ram, rad, ryf, ryl = self.P_am(0.0) * rr * taper, self.P_ad(0.0) * rr * taper, self.P_yf(0.0) * taper, self.P_yl(0.0) * taper
        w = sstep(-1.5, 1.5, z)                                # crown above, root below, blended across the CEJ (no crease)
        mix = lambda c, r_: r_ + (c - r_) * w
        return mix(am, ram), mix(ad, rad), mix(yf, ryf), mix(yl, ryl)

    # ---- the incisal edge / cusps / occlusal table
    def _build_top(self):
        T, k, rng, H = self.T, self.kind, self.rng, self.H
        jit = lambda sd: 1.0 + sd * rng.standard_normal()
        self.cusps = []
        if k in ("premolar", "molar"):
            for c in T["cusps"]:
                u, v, h = c[0], c[1], c[2]
                self.cusps.append([u + 0.03 * rng.standard_normal(), v + 0.03 * rng.standard_normal(),
                                   h * self.s["scale"] + 0.12 * rng.standard_normal(), c[3] if len(c) > 3 else 0.72,
                                   c[4] if len(c) > 4 else 1.0])
            self.ridges = [list(r) for r in T.get("ridges", [])]
            self.groove_depth = T.get("groove_depth", 0.55) * jit(0.15)
        self.lobes = [jit(0.12) for _ in range(3)]

    def top(self, x, y, am, ad, yc, bt, dentin=False):
        """the top surface height z_top(x, y) (mm): incisal edge, cusp, or occlusal table. dentin=True: the DEJ's top
        (no fissures, lowered by the enamel's thickness, mamelon lobes kept under an incisal edge)"""
        H, k, w = self.H, self.kind, self.wear
        a = np.where(x > 0, ad, am)
        u = x / a
        if k == "incisor":
            # straight edge; the mesioincisal angle sharp, the distoincisal rounded (r grows on laterals)
            # (11-12) each angle an elliptical arc, r across and r_z down (r_z = r: circular); U1's fitted to its
            # reference silhouette (fs-mouth s1_fit_corners.py), where the section had carried the corners as a chamfer
            rm, rd = self.T.get("r_mi", 0.5), self.T.get("r_di", 1.4)
            rmz, rdz = self.T.get("r_mi_z", rm), self.T.get("r_di_z", rd)
            r = np.where(x > 0, rd, rm)
            rz = np.where(x > 0, rdz, rmz)
            e = np.minimum(np.maximum(np.abs(x) - (a - r), 0.0) / r, 1.0)
            drop = rz * (1.0 - np.sqrt(np.maximum(1.0 - e * e, 0.0)))
            z = H - drop - 0.25 * (np.abs(u) ** 4) * self.T.get("edge_curve", 1.0)
            # three lobes (mamelons): on the outer edge only while young; under the edge (the DEJ) always
            lob = sum(L_ * bump(u, c, 0.36) for L_, c in zip(self.lobes, (-0.62, 0.0, 0.62)))
            if dentin:
                return z - self.T.get("t_inc", 1.7) * self.s["enamel"] + 0.65 * self.s["lobes"] * lob
            z = z - 0.25 * self.mam + 0.4 * self.mam * lob
            return np.minimum(z, H - 0.9 * w) if w > 0 else z
        if k == "canine":
            ut = self.T.get("cusp_u", -0.08)                # the cusp tip, slightly mesial on upper canines
            dm, dd = self.T.get("slope_m", 2.0), self.T.get("slope_d", 2.6)
            du = u - ut
            sl = np.where(du < 0, dm / (1 + ut), dd / (1 - ut))
            rr = 0.35
            z = H - sl * (np.sqrt(du * du + rr * rr) - rr)
            if dentin:
                return z - self.T.get("t_inc", 2.0) * self.s["enamel"]
            return np.minimum(z, H - 1.2 * w) if w > 0 else z
        # premolar / molar: cusps as rounded cones (their pointwise max leaves the grooves between them)
        v = (y - yc) / bt
        best = np.full(np.shape(x), -1e3)
        second = np.full(np.shape(x), -1e3)
        for (cu, cv, ch, slope, rad) in self.cusps:
            dx = (u - cu) * a
            dy = (v - cv) * bt
            ex, ey = -cu * a, -cv * bt                                  # toward the table's centre
            en = np.hypot(ex, ey) + 1e-9
            ex, ey = ex / en, ey / en
            dpar = dx * ex + dy * ey
            dper = -dx * ey + dy * ex
            kp = self.T.get("ridge_k", 0.62)
            rr = 0.35 * rad
            dist = np.sqrt((np.maximum(dpar, 0) * kp) ** 2 + np.minimum(dpar, 0) ** 2 + 1.25 * dper * dper + rr * rr) - rr
            c = H + ch - slope * 1.25 * dist
            if not dentin and w > 0:
                c = np.minimum(c, H + ch - 0.8 * w)              # a small wear facet on the cusp's own tip
            if dentin:
                c = H + ch - self.T.get("t_cusp", 2.0) * self.s["enamel"] - 0.85 * slope * (np.sqrt(dx * dx + dy * dy + 1.0) - 1.0)
            second = np.where(c > best, best, np.maximum(second, c))
            best = smax(best, c, 0.2) if not dentin else np.maximum(best, c)
        for (u0, v0, u1, v1, h0, wid) in self.ridges:            # oblique / transverse / marginal ridges
            px, py = u * a, v * bt
            ax_, ay_ = u0 * a, v0 * bt
            bx_, by_ = u1 * a, v1 * bt
            ex, ey = bx_ - ax_, by_ - ay_
            tt = np.clip(((px - ax_) * ex + (py - ay_) * ey) / (ex * ex + ey * ey), 0, 1)
            dd = np.hypot(px - ax_ - tt * ex, py - ay_ - tt * ey)
            c = H + h0 - (0.9 if not dentin else 0.6) * (np.sqrt(dd * dd + (0.35 * wid) ** 2) - 0.35 * wid) - (self.T.get("t_cusp", 2.0) * self.s["enamel"] if dentin else 0)
            second = np.where(c > best, best, np.maximum(second, c))
            best = smax(best, c, 0.25) if not dentin else np.maximum(best, c)
        if dentin:
            return best
        # the fissure: a narrow cut where the two highest cones meet (the groove line), inside the occlusal table
        gap = best - second
        fiss = self.groove_depth * np.exp(-(gap / 0.08) ** 2) * sstep(0.95, 0.75, np.abs(v)) * sstep(1.0, 0.8, np.abs(u))
        return best - fiss

    # ---- the roots
    def _build_roots(self):
        T = self.T
        self.roots = T.get("roots", None)          # None: one root (the section tapering to the apex)
        self.furc = T.get("furcation", 0.0) * self.s["scale"]

    def _root_field(self, x, y, z, yc0, bt0, am0, ad0):
        """multi-rooted teeth: a trunk to the furcation, then each root as a tapered, slightly curved rod"""
        L, f = self.L, self.furc
        out = np.full(np.shape(x), 1e3)
        for (ru, rv, rlen, rmd, rll, div, curv) in self.roots:
            a0 = am0 if ru < 0 else ad0
            x0, y0 = ru * a0, yc0 + rv * bt0
            Lr = rlen * self.s["scale"]
            t = np.clip((-z - f + 1.0) / (Lr - f + 1.0), 0.0, 1.0)
            cx = x0 * (1 + div * t) + curv * t * t
            cy = y0 + (y0 - yc0) * div * t
            rx = rmd / 2 * ((1 - t) ** 0.8 * 0.92 + 0.08) * (1 + 0.25 * (1 - sstep(0.0, 0.25, t)))
            ry = rll / 2 * ((1 - t) ** 0.8 * 0.92 + 0.08) * (1 + 0.25 * (1 - sstep(0.0, 0.25, t)))
            e = (np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) - 1.0) * np.minimum(rx, ry)
            e = smax(e, z - (-f + 2.5), 1.0)                   # from inside the trunk
            e = smax(e, -(z + Lr), 0.5)                        # the apex
            out = smin(out, e, 0.6)
        return out

    # ---- the fields
    def fields(self, P, want=("body", "dentin", "pulp")):
        P = np.asarray(P, float)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        H, L = self.H, self.L
        am, ad, yf, yl = self.section_dims(z)
        yc, bt = 0.5 * (yf + yl), 0.5 * (yf - yl)
        # a lingual taper (premolars / molars) and the rhomboid shear (upper molars)
        xs = x + self.shear * (y - yc)
        lw = np.clip(-(y - yc) / bt, 0, 1)
        am_e, ad_e = am * (1 - self.ltaper * lw), ad * (1 - self.ltaper * lw)
        sec = _superellipse(xs, y - yc, am_e, ad_e, bt, bt, self.pf, self.pl)
        s = z / H
        k = self.kind
        # surface structure on the crown: marginal ridges and fossa (lingual of incisors / canines), labial lobes
        if k in ("incisor", "canine"):
            a = np.where(xs > 0, ad_e, am_e)
            ua = np.abs(xs) / a
            lingual = sstep(0.0, -0.4, (y - yc) / bt)
            rid = self.T.get("marginal", 0.45) * bump(ua, 0.86, 0.22) * bump(s, 0.62, 0.36)
            fos = self.T.get("fossa", 0.25) * bump(ua, 0.0, 0.62) * bump(s, 0.6, 0.3)
            if k == "canine":   # the lingual ridge from the cingulum to the cusp tip, the two fossae beside it
                rid = rid + self.T.get("lingual_ridge", 0.45) * bump(xs / a - self.T.get("cusp_u", -0.08), 0, 0.28) * bump(s, 0.62, 0.42)
            # (11-11 23:40) the incisal ridge between the marginal ridges is thinner than the corners' silhouette: the
            # lingual fossa runs up to it, so the mid edge is ~0.8-1.0 mm thick 1 mm from the edge (where the enamel alone
            # makes the edge translucent: slab 0.8 mm over black L* 52, b* -2; at 1.5 mm L* 62, b* +4)
            thin = self.T.get("incisal_thin", 0.0) * bump(ua, 0.0, 0.8) * sstep(0.7, 0.96, s)
            sec = sec - lingual * (rid - fos - thin)
            labial = sstep(0.0, 0.4, (y - yc) / bt)
            dep = self.T.get("labial_dep", 0.07) * (bump(ua, 0.36, 0.13)) * bump(s, 0.7, 0.35)
            if k == "canine":
                dep = dep - self.T.get("labial_ridge", 0.12) * bump(xs / a - self.T.get("cusp_u", -0.08), 0, 0.3) * bump(s, 0.65, 0.4)
            sec = sec + labial * dep
        # the top
        if self.roots is None or True:
            ztop = self.top(xs, y, am_e, ad_e, yc, bt)
        crown = smax(sec, z - ztop, self.T.get("r_rim", 0.5) if k in ("premolar", "molar") else 0.35)
        out = {}
        if self.roots is None:
            body = smax(crown, -(z + L), 0.6)
        else:
            trunk = smax(crown, -(z + self.furc + 0.5), 1.2)
            body = smin(trunk, self._root_field(x, y, z, *self._cerv()), 1.4)
        out["body"] = body
        if "dentin" in want or "pulp" in want:
            # the enamel's thickness: zero at the CEJ (curved: higher on the proximal faces), full by a third of the crown
            zc = self.cej_height(xs, y, am_e, ad_e, yc, bt)
            # the enamel's thickness up the crown: incisors / canines from the measured labial thirds (U1 0.41 / 0.62 /
            # 0.945 mm cervical / middle / incisal; t_side scales it), posteriors reaching t_side by 40 % of the crown
            sr = np.clip((z - zc) / H, 0.0, 1.0)
            if k in ("incisor", "canine"):
                prof = np.interp(sr, [0.0, 0.06, 0.17, 0.5, 0.83, 1.0], [0.0, 0.22, 0.434, 0.656, 1.0, 1.0])
            else:
                prof = sstep(0.0, 0.4, sr)
            tside = self.T.get("t_side", 1.1) * self.s["enamel"] * prof
            dz = self.top(xs, y, am_e, ad_e, yc, bt, dentin=True)
            den_crown = smax(sec + tside, z - dz, 0.9)
            den = np.where(z - zc > 0, den_crown, body)
            den = np.maximum(den, body)
            out["dentin"] = den
            if "pulp" in want:
                pul = den + self.T.get("pulp_wall", 2.3) * self.s["scale"]
                pul = smax(pul, -(z + 0.65 * L), 0.5)
                # the canal(s): a thin rod to the apex inside each root
                if self.roots is None:
                    r = 0.42 * (1 - np.clip(-z / L, 0, 1)) + 0.12
                    can = smax(np.hypot(x - 0.0, y - self._yc0()) - r, -(z + L - 0.6), 0.2)
                    can = smax(can, z - 0.5, 0.3)
                else:
                    can = self._root_field(x, y, z, *self._cerv()) + 0.0
                    can = can + 1.2   # inside each root by ~1.2 mm
                out["pulp"] = smin(pul, can, 0.6)
        return out

    def _yc0(self):
        return float(0.5 * (self.P_yf(0.0) + self.P_yl(0.0)))

    def _cerv(self):
        yf0, yl0 = float(self.P_yf(0.0)), float(self.P_yl(0.0))
        return 0.5 * (yf0 + yl0), 0.5 * (yf0 - yl0), float(self.P_am(0.0)), float(self.P_ad(0.0))

    def cej_height(self, x, y, am, ad, yc, bt):
        """the CEJ's height z_cej(x, y): 0 on the labial and lingual faces, rising by cej_m / cej_d on the proximal faces"""
        a = np.where(x > 0, ad, am)
        ux, vy = (x / a) ** 2, ((y - yc) / bt) ** 2
        w = ux / (ux + vy + 1e-9)
        w = w ** 1.6
        return np.where(x > 0, self.cej_d, self.cej_m) * w

    def bbox(self, pad=1.0):
        a = max(self.md, self.ll) / 2 + 1.5
        lo = np.array([-a - 2.5, -a - 2.5, -self.L - pad])
        hi = np.array([a + 2.5, a + 2.5, self.H + pad + 0.8])
        return lo, hi
