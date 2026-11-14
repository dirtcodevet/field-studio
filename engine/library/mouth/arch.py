"""engine/library/mouth/arch.py: the upper and lower dental arches placed in occlusion (fields, mesh-free). fs-mouth, 11-11.

The arch is the curve through the teeth's contact points: every tooth sits on it at its own arc length, so neighbours
touch at their contacts. Its shape (a 4th-order arch form, y = a2 x^2 + a4 x^4) is solved so the built dentition meets
the measured widths at their own landmarks (the canines' cusp tips, the first molars' mesiobuccal cusp tips). Each tooth
is turned by its crown inclination (torque) and angulation (tip), lifted to the occlusal curve (the curve of Spee), and
the lower arch is closed onto the upper until the teeth touch (contact, no overlap): the overjet and overbite are then
measured, not imposed.

Frame (mm): the dentition frame of the library item: x her left, y backward (+ = posterior), z up; origin at the upper
incisal point (the mid point of the two upper central incisors' incisal edges, on their labial face). Upper crowns point
down (-z), lower crowns up.

    D = Dentition(settings)                    # settings: see DEFAULTS
    f, which = D.field(P)                      # union of every tooth (body), and the index of the nearest tooth
    D.teeth: list of placed teeth (code, side, Tooth, R, t)  -> tooth frame p_t = R^T (p - t)
"""
import json
import os

import numpy as np
from scipy.optimize import brentq, least_squares

from teeth import Tooth, TYPES, smin

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["1", "2", "3", "4", "5", "6", "7"]

# cited defaults (women): arch widths at their landmarks, depth, overjet / overbite targets (checks), inclinations
DEFAULTS = dict(
    seed=0,
    scale=1.0,                 # all teeth
    wear=0.15,
    age=30.0,                  # years: the incisal mamelons wear off with incisal contact (teeth.Tooth)
    incisal_contact=1.0,       # 1 = the incisors meet; 0 = open bite (mamelons stay more often)
    mamelon=None,              # None: by age and contact (teeth.Tooth); a number sets every incisor's (her: 0)
    ic_u=33.4, im_u=50.2,      # upper intercanine (cusp tips), intermolar (M1 mesiobuccal cusp tips)  [mouth_spec_research / teeth_spec]
    ic_l=25.2, im_l=42.8,      # lower
    spee=1.8,                  # curve of Spee depth (mm) [R7 2.1 (0.6)]
    occl_tilt=0.0,             # the occlusal plane's tilt in this frame (deg)
    overjet_t=2.0, overbite_t=2.3,   # targets the closing is checked against [R7 2.0 (0.5), 2.3 (0.9)]
    crowding=0.0,              # 0..1: per-tooth rotation / displacement noise (alignment)
    spacing=0.0,               # mm added between neighbours (diastema-free default)
    third_molars=False,
)
# long-axis inclination (deg, + = crown labial / buccal) and mesiodistal angulation (deg, + = crown mesial): U1 from Saadeh
# 2025 (U1 to palatal plane 113.8 in 106 women -> 23.8); L1 from IMPA 92 with a 25 deg mandibular plane (~27); the rest
# estimated from Andrews' normal pattern (E)
INCL = {"U": [23.8, 20.0, 12.0, 4.0, 4.0, -2.0, -6.0], "L": [24.0, 22.0, 14.0, 2.0, -4.0, -12.0, -18.0]}
TIP = {"U": [4.0, 8.0, 9.0, 2.5, 2.5, 5.0, 0.5], "L": [0.5, 0.5, 2.5, 1.3, 1.5, 2.0, 3.0]}


def _rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class ArchCurve:
    """the contact arch from its curvature along the arc: kappa1 through the anterior segment (an ovoid front), easing to
    kappa2 behind the canine (straighter flanks running back): heading theta(s) = int kappa, x = int cos, y = int sin.
    y backward (+), x lateral (her left for the + side). s_c: the canine's middle on the arc."""
    def __init__(self, k1, k2, s_c):
        self.k1, self.k2, self.s_c = k1, k2, s_c
        ss = np.linspace(0, 70, 7001)
        e = np.clip((ss - (s_c - 3.0)) / 9.0, 0, 1)
        kap = k1 + (k2 - k1) * e * e * (3 - 2 * e)
        th = np.concatenate([[0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * np.diff(ss))])
        ds = np.diff(ss)
        x = np.concatenate([[0], np.cumsum(np.cos(0.5 * (th[1:] + th[:-1])) * ds)])
        y = np.concatenate([[0], np.cumsum(np.sin(0.5 * (th[1:] + th[:-1])) * ds)])
        self.ss, self.th, self.x, self.y = ss, th, x, y

    def at(self, s):
        x = float(np.interp(s, self.ss, self.x))
        y = float(np.interp(s, self.ss, self.y))
        th = float(np.interp(s, self.ss, self.th))
        t = np.array([np.cos(th), np.sin(th), 0.0])
        n = np.array([t[1], -t[0], 0.0])              # outward (labial / buccal): forward and lateral
        return np.array([x, y, 0.0]), t, n


class Dentition:
    def __init__(self, settings=None):
        s = dict(DEFAULTS)
        s.update(settings or {})
        self.s = s
        self.rng = np.random.default_rng(s["seed"])
        self.lower_t = None
        codes = ORDER + (["8"] if s["third_molars"] else [])
        self.types = {a: [TYPES[a + c] for c in codes if a + c in TYPES] for a in ("U", "L")}
        self.teeth = []
        self.arch = {}
        for a in ("U", "L"):
            self._place_arch(a)
        self._close()
        # (11-12) eruption moves each lower tooth along its own (tilted) axis: its contacts with its neighbours re-closed
        # along the arch (the eruption kept), then it erupts again to its antagonists
        for _ in range(2):                                          # two cycles settle it (a tilted molar moves on)
            self.contacts["L"] = self._close_contacts([p for p in self.teeth if p["arch"] == "L"], "L", iters=3)
            self._erupt_lower()

    # ---- one arch
    def _tooth_tip_local(self, t):
        """the tooth's highest point in its own frame (incisal edge mid, canine cusp, buccal cusp tip)"""
        w = t.wear                                    # (11-11 00:30) the worn edge / cusp, as teeth.py top() wears it
        if t.kind == "incisor":
            return np.array([0.0, float(0.5 * (t.P_yf(1.0) + t.P_yl(1.0))), t.H - 0.9 * w])
        if t.kind == "canine":
            u = t.T.get("cusp_u", 0.0)
            a = float(t.P_am(1.0) if u < 0 else t.P_ad(1.0))
            return np.array([u * a, float(0.5 * (t.P_yf(1.0) + t.P_yl(1.0))), t.H - 1.2 * w])
        c = max(t.cusps, key=lambda c: c[1])        # the most buccal cusp ~ the buccal cusp tip (mesiobuccal on molars)
        if t.kind == "molar":
            c = [cc for cc in t.cusps if cc[1] > 0][0]
        yc, bt = 0.5 * (t.P_yf(1.0) + t.P_yl(1.0)), 0.5 * (t.P_yf(1.0) - t.P_yl(1.0))
        a = float(t.P_am(1.0) if c[0] < 0 else t.P_ad(1.0))
        return np.array([c[0] * a, float(yc + c[1] * bt), t.H + c[2] - 0.8 * w])

    def _build(self, a, a2, a4):
        """place one arch's teeth on the curve (curvatures a2 front, a4 flanks; 1/mm); returns the placed list"""
        s = self.s
        up = a == "U"
        out = []
        tps = self.types[a]
        seeds = self.rng.integers(0, 1 << 30, size=2 * len(tps)) if not hasattr(self, "_seeds_" + a) else getattr(self, "_seeds_" + a)
        setattr(self, "_seeds_" + a, seeds)
        widths = [Tooth(T, dict(scale=s["scale"])).md + s["spacing"] for T in tps]
        wadj = getattr(self, "_wadj", {}).get(a)                    # (11-12) the spacing contact needs (pass 2)
        if wadj is not None:
            widths = [w + d for w, d in zip(widths, wadj)]
        cum = np.concatenate([[0], np.cumsum(widths)])
        curve = ArchCurve(a2, a4, cum[2] + widths[2] / 2)
        for side in (+1, -1):
            for i, T in enumerate(tps):
                tooth = Tooth(T, dict(scale=s["scale"], wear=s["wear"], age=s["age"], incisal_contact=s["incisal_contact"], mamelon=s["mamelon"],
                                      seed=int(seeds[i + (len(tps) if side < 0 else 0)])))
                sm = cum[i] + widths[i] / 2
                P, tg, nm = curve.at(sm)
                P = P * np.array([side, 1, 1])
                tg = tg * np.array([side, 1, 1])
                nm = nm * np.array([side, 1, 1])
                # the tooth's frame: x distal = along the tangent away from the midline, y labial = outward, z toward the edge
                ez = np.array([0, 0, -1.0]) if up else np.array([0, 0, 1.0])
                ex = tg / np.linalg.norm(tg)
                ey = nm
                # columns: tooth x, y, z in the dentition frame; on her right the frame is a mirror (det -1), so the
                # right teeth are the mirror images of the left
                R = np.stack([ex, ey, ez], 1)
                # inclination: about the tooth's x (crown labial / buccal = +); angulation: about its y (crown mesial = +)
                inc, tip = INCL[a][i], TIP[a][i]
                Rinc = _rot(ex, (-inc if up else inc) * side)
                Rtip = _rot(ey, (-tip if up else tip) * side)
                # alignment noise (crowding): rotation about the long axis and a small labiolingual shift
                cr = s["crowding"]
                rr = np.random.default_rng(int(seeds[i]) + 7 * side)
                Rrot = _rot(R[:, 2], cr * rr.normal(0, 8.0 if tooth.kind in ("incisor", "canine") else 3.0))
                Rfull = Rrot @ Rtip @ Rinc @ R
                shv = cr * rr.normal(0, 0.4)
                out.append(dict(code=T["code"], arch=a, side=side, i=i, tooth=tooth, R=Rfull, t=None, contact=P,
                                ey=nm, shift=shv * nm, s_mid=sm, s_end=cum[-1], sh=shv, _seed=int(seeds[i])))
        self.arch[a] = curve
        return out

    def _seat(self, placed, a):
        """translate each tooth so its contact level lies on the arch line and its tip on the occlusal curve"""
        up = a == "U"
        spee = self.s["spee"]
        for p in placed:
            t, R = p["tooth"], p["R"]
            # the contact point in the tooth frame: the widest point (proximal) at the contact height; use the labial
            # contact line: the tooth's centre at contact height projected on the arch line from the labial face
            tip = self._tooth_tip_local(t)
            # horizontal: the arch line passes through the tooth's MD centre at its contact height, on the line between
            # labial and lingual faces' mid (the contact points lie near the middle third of the BL width)
            sc = 0.8 if t.kind in ("incisor", "canine") else 0.7
            zc = sc * t.H
            yc = float(0.5 * (t.P_yf(zc / t.H) + t.P_yl(zc / t.H)))
            cl = np.array([0.0, yc, zc])
            # vertical: the occlusal curves. Lower cusp tips: z_l = OB + rise * f - spee * sin(pi f) (f = arc fraction to
            # the last molar's middle); upper tips = z_l - the vertical overlap (the overbite at the incisors, ~1.6 mm
            # of buccal-cusp overlap behind the canines). The closing below settles the true contacts.
            f = np.clip(p["s_mid"] / max(p["s_end"] - 5.0, 1.0), 0, 1)
            OB = self.s["overbite_t"]
            zl = OB + 1.6 * f - spee * np.sin(np.pi * f)
            overlap = OB + (1.6 - OB) * np.clip((p["s_mid"] - 12.0) / 8.0, 0, 1)
            zocc = zl if not up else zl - overlap
            P0 = p["contact"] + p["shift"]
            # position: R @ cl + t = P0 horizontally; R @ tip + t has z = zocc
            tvec = P0 - R @ cl
            tz = zocc - (R @ tip)[2]
            tvec[2] = tz
            p["t"] = tvec
        return placed

    def _replace(self, p, a, s_new):
        """the same tooth (its rotations) moved along its arch to arc position s_new, re-seated"""
        curve = self.arch[a]
        P, tg, nm = curve.at(s_new)
        side = p["side"]
        P = P * np.array([side, 1, 1]); tg = tg * np.array([side, 1, 1]); nm = nm * np.array([side, 1, 1])
        ez = np.array([0, 0, -1.0]) if a == "U" else np.array([0, 0, 1.0])
        ex = tg / np.linalg.norm(tg)
        R = np.stack([ex, nm, ez], 1)
        inc, tip = INCL[a][p["i"]], TIP[a][p["i"]]
        up = a == "U"
        Rinc = _rot(ex, (-inc if up else inc) * side)
        Rtip = _rot(nm, (-tip if up else tip) * side)
        Rrot = _rot(R[:, 2], self.s["crowding"] * np.random.default_rng(int(p["_seed"]) + 7 * side).normal(0, 8.0 if p["tooth"].kind in ("incisor", "canine") else 3.0))
        p.update(R=Rrot @ Rtip @ Rinc @ R, contact=P, ey=nm, shift=p["sh"] * nm, s_mid=s_new)
        self._seat([p], a)
        if a == "L" and self.lower_t is not None:                  # after the closing: the arch's move and its eruption kept
            p["t0"] = p["t"].copy()
            p["t"] = p["t0"] + self.lower_t + p["R"][:, 2] * p.get("_erupt", 0.0)

    def pair_gap(self, p, q, n=40000, seed=1):
        """the smallest distance from tooth p's crown surface to tooth q (mm; negative = overlap)"""
        t = p["tooth"]
        rng = np.random.default_rng(seed)
        Q = np.stack([rng.uniform(-t.md / 2 - 0.6, t.md / 2 + 0.6, n), rng.uniform(-t.ll / 2 - 0.6, t.ll / 2 + 0.6, n),
                      rng.uniform(0.1 * t.H, t.H + 0.6, n)], 1)
        f = t.fields(Q, want=("body",))["body"]
        Q = Q[np.abs(f) < 0.06]
        return float(self.tooth_field(q, Q @ p["R"].T + p["t"])["body"].min())

    def _close_contacts(self, pl, a, iters=3):
        """(11-12) neighbours brought into contact along the arch, from the midline outward, as mesial drift brings real
        teeth together: the central pair closes on each other, each next tooth slides mesially (or back, where crowns
        overlap) until it touches its mesial neighbour. Placement by nominal widths had left gaps up to 0.44 mm (1.6 mm at
        the centrals' edge) and overlaps to 0.31 mm. -> {pair: final gap mm}"""
        by = {(p["code"], p["side"]): p for p in pl}
        codes = sorted({p["code"] for p in pl}, key=lambda c: int(c[1:]))
        res = {}
        c1 = codes[0]
        sp = float(self.s["spacing"])                               # the setting's gap between neighbours (0: touching)
        for _ in range(iters):
            g = self.pair_gap(by[(c1, 1)], by[(c1, -1)]) - sp
            if abs(g) > 5.0:                                         # not a contact reading: never move on it
                break
            g = float(np.clip(g, -2.0, 2.0))
            for sd in (1, -1):
                self._replace(by[(c1, sd)], a, by[(c1, sd)]["s_mid"] - 0.5 * g)
        res[c1 + "L-R"] = self.pair_gap(by[(c1, 1)], by[(c1, -1)])
        for sd in (1, -1):
            for c0, c in zip(codes[:-1], codes[1:]):
                p, q = by[(c, sd)], by[(c0, sd)]
                d0 = q["s_mid"] - q.get("_s0", q["s_mid"])                 # the mesial neighbour's own move, carried first
                p["_s0"] = p["s_mid"]
                s_ = p["s_mid"] + d0
                self._replace(p, a, s_)
                for _ in range(iters):
                    g = self.pair_gap(p, q) - sp
                    if abs(g) > 5.0:
                        break
                    self._replace(p, a, p["s_mid"] - float(np.clip(g, -2.0, 2.0)))
                res["%s-%s%s" % (c0, c, "L" if sd > 0 else "R")] = self.pair_gap(p, q)
        return res

    def _place_arch(self, a):
        s = self.s
        ic, im = (s["ic_u"], s["im_u"]) if a == "U" else (s["ic_l"], s["im_l"])

        def measure(a2, a4):
            pl = self._seat(self._build(a, a2, a4), a)
            pts = {}
            for p in pl:
                tip = self._tooth_tip_local(p["tooth"])
                pts[(p["code"], p["side"])] = p["R"] @ tip + p["t"]
            c3 = a + "3"
            c6 = a + "6"
            return pl, abs(pts[(c3, 1)][0] - pts[(c3, -1)][0]), abs(pts[(c6, 1)][0] - pts[(c6, -1)][0])

        def resid(v):
            _, w3, w6 = measure(v[0], v[1])
            return [w3 - ic, w6 - im]
        sol = least_squares(resid, [0.05, 0.01], x_scale=[0.01, 0.01], bounds=([0.0, -0.05], [0.2, 0.2]))
        pl, w3, w6 = measure(*sol.x)
        # (11-12) pass 1: close the contacts; the arch then fitted again with each tooth's spacing as contact needs it
        # (the cited intercanine / intermolar widths are between teeth in contact), and closed once more
        s0 = {(p["code"], p["side"]): p["s_mid"] for p in pl}
        self._close_contacts(pl, a)
        n_ = len(self.types[a])
        adj = np.zeros(n_)
        for i in range(n_):
            ds = [pp["s_mid"] - s0[(pp["code"], pp["side"])] for pp in pl if pp["i"] == i]
            adj[i] = float(np.mean(ds))
        # tooth i's axis sits at sum(w_<i) + w_i / 2: moving it by ds_i takes dw_i = 2 (ds_i - sum(dw_<i))
        dw = np.zeros(n_)
        for i in range(n_):
            dw[i] = 2.0 * (adj[i] - dw[:i].sum())
        self._wadj = getattr(self, "_wadj", {})
        self._wadj[a] = list(dw)
        sol = least_squares(resid, sol.x, x_scale=[0.01, 0.01], bounds=([0.0, -0.05], [0.2, 0.2]))
        pl, w3, w6 = measure(*sol.x)
        self.contacts = getattr(self, "contacts", {})
        self.contacts[a] = self._close_contacts(pl, a, iters=4)
        pts = {(p["code"], p["side"]): p["R"] @ self._tooth_tip_local(p["tooth"]) + p["t"] for p in pl}
        w3 = abs(pts[(a + "3", 1)][0] - pts[(a + "3", -1)][0]); w6 = abs(pts[(a + "6", 1)][0] - pts[(a + "6", -1)][0])
        self.fit = getattr(self, "fit", {})
        curve = self.arch[a]
        # the arch depth: the incisors' contact (0) to the line through the first molars' mesial contacts
        tps = self.types[a]
        widths = [Tooth(T, dict(scale=s["scale"])).md + s["spacing"] for T in tps]
        sm1 = float(np.sum(widths[:5]))
        depth = float(curve.at(sm1)[0][1])
        self.fit[a] = dict(k_front=float(sol.x[0]), k_flank=float(sol.x[1]), ic=float(w3), im=float(w6), ic_t=ic, im_t=im, depth_to_M1_mesial=depth)
        self.teeth += pl

    # ---- the lower arch closed onto the upper
    def _close(self):
        """the lower arch: horizontally its incisors' labial face overjet_t behind the uppers' labial incisal edge (at
        the edge's height); vertically raised or lowered until the arches just touch (no overlap). The overjet, overbite
        and the contacts by region are measured after."""
        self.lower_t = np.zeros(3)
        self._apply_lower(self.lower_t)
        # (11-11 00:25) the probe at the upper incisors' OWN edge height (a fixed 0.15 mm missed worn incisors: NaN)
        zp = 0.5 * (self.landmarks()["U1L"][2] + self.landmarks()["U1R"][2]) + 0.5
        yu = self._labial_y("U", 2.4, zp)
        yl = self._labial_y("L", 2.4, zp)
        if not (np.isfinite(yu) and np.isfinite(yl)):
            raise ValueError("arch closing: the incisors' labial faces not found at z %.2f (yu %s, yl %s)" % (zp, yu, yl))
        self.lower_t = np.array([0.0, (yu + self.s["overjet_t"]) - yl, 0.0])
        self._apply_lower(self.lower_t)
        for it in range(8):
            g, _ = self.gap()
            if g is None or not np.isfinite(g) or g > 50.0:      # no contact found: never move on that
                break
            self.lower_t = self.lower_t + np.array([0, 0, g])
            self._apply_lower(self.lower_t)
        self.closed_gap = float(self.gap()[0])
        self._erupt_lower()

    def _erupt_lower(self):
        # each lower tooth erupts along its own axis until it meets the upper arch (eruption stops at occlusal contact:
        # the cause of the occlusal heights), so every tooth touches its antagonists, none overlaps
        self.erupt = {}
        for p in self.teeth:
            if p["arch"] != "L":
                continue
            ax = p["R"][:, 2]
            moved = 0.0
            g_prev = None
            for it in range(14):
                g = self.tooth_gap(p)
                if g is None or not np.isfinite(g) or g > 50.0 or abs(g) < 0.01:
                    break
                # (11-12) eruption stops at occlusal contact, and where rising no longer brings an antagonist closer
                # (none within reach: a crowded lower second molar slid past the upper arch's end had risen 28 mm while
                # its reading stayed ~3 mm); 6 mm in all as a backstop
                if g_prev is not None and g > 0 and g > 0.8 * g_prev and moved > 1.0:
                    break
                g_prev = g
                step = g / max(abs(ax[2]), 0.5)
                step = float(np.clip(step, -1.5, 3.0))
                tot = p.get("_erupt", 0.0) + moved
                step = float(np.clip(step, -6.0 - tot, 6.0 - tot))
                if abs(step) < 1e-6:
                    break
                p["t"] = p["t"] + ax * step
                moved += step
            p["_erupt"] = p.get("_erupt", 0.0) + moved
            self.erupt["%s%s" % (p["code"], "L" if p["side"] > 0 else "R")] = dict(moved=moved, gap=self.tooth_gap(p))

    def _labial_y(self, arch, x, z):
        """the most labial (smallest y) surface point of an arch along the line (x, ., z)"""
        ys = np.arange(-15.0, 15.0, 0.02)
        P = np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], 1)
        f, _ = self.field(P, arch=arch)
        k = np.where(f < 0)[0]
        if not len(k):
            return np.nan
        j = k[0]
        return float(ys[j - 1] + (ys[j] - ys[j - 1]) * f[j - 1] / (f[j - 1] - f[j])) if j > 0 else float(ys[j])

    def _apply_lower(self, d):
        for p in self.teeth:
            if p["arch"] == "L":
                if "t0" not in p:
                    p["t0"] = p["t"].copy()
                p["t"] = p["t0"] + d

    def tooth_field(self, p, P, want=("body",)):
        Q = (P - p["t"]) @ p["R"]                    # p_t = R^T (P - t)
        if p["side"] < 0:
            pass
        return p["tooth"].fields(Q, want=want)

    def field(self, P, arch=None, want="body"):
        P = np.asarray(P, float)
        out = np.full(len(P), 1e3)
        idx = np.full(len(P), -1)
        for j, p in enumerate(self.teeth):
            if arch and p["arch"] != arch:
                continue
            lo, hi = self.tooth_box(p)
            m = np.all((P >= lo) & (P <= hi), axis=1)
            if not m.any():
                continue
            f = self.tooth_field(p, P[m], want=(want,))[want]
            sub = out[m]
            better = f < sub
            sub[better] = f[better]
            out[m] = sub
            ii = idx[m]
            ii[better] = j
            idx[m] = ii
        return out, idx

    def tooth_box(self, p, pad=1.5):
        lo, hi = p["tooth"].bbox(pad=pad)
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        W = C @ p["R"].T + p["t"]
        return W.min(0), W.max(0)

    def gap(self, h=0.15):
        """the smallest distance between the upper and lower crowns (mm; negative = overlap), sampled on the lower crowns'
        occlusal halves: lower crown surface points evaluated against the upper field"""
        best = 1e9
        where = None
        rng = np.random.default_rng(0)
        for p in self.teeth:
            if p["arch"] != "L":
                continue
            t = p["tooth"]
            # sample the lower crown's upper part in its frame, keep near-surface points
            n = 6000
            Q = np.stack([rng.uniform(-t.md / 2 - 0.5, t.md / 2 + 0.5, n), rng.uniform(-t.ll / 2 - 0.5, t.ll / 2 + 0.5, n),
                          rng.uniform(0.4 * t.H, t.H + 0.6, n)], 1)
            f = t.fields(Q, want=("body",))["body"]
            Q = Q[np.abs(f) < 0.12]
            if not len(Q):
                continue
            W = Q @ p["R"].T + p["t"]
            fu, _ = self.field(W, arch="U")
            k = int(np.argmin(fu))
            if fu[k] < best:
                best, where = float(fu[k]), W[k]
        return best, where

    def cusp_points(self, p):
        """world positions of a posterior tooth's cusp tips and its occlusal table's centre"""
        t = p["tooth"]
        yc, bt = 0.5 * (t.P_yf(1.0) + t.P_yl(1.0)), 0.5 * (t.P_yf(1.0) - t.P_yl(1.0))
        out = []
        for c in t.cusps:
            a = float(t.P_am(1.0) if c[0] < 0 else t.P_ad(1.0))
            out.append(p["R"] @ np.array([c[0] * a, float(yc + c[1] * bt), t.H + c[2]]) + p["t"])
        cen = p["R"] @ np.array([0.0, float(yc), t.H - 2.0]) + p["t"]
        return np.array(out), cen

    def tooth_gap(self, p, n=7000, seed=0):
        """the smallest distance from one lower tooth's crown (its occlusal half) to the upper arch (mm, neg = overlap)"""
        t = p["tooth"]
        rng = np.random.default_rng(seed)
        Q = np.stack([rng.uniform(-t.md / 2 - 0.6, t.md / 2 + 0.6, n), rng.uniform(-t.ll / 2 - 0.6, t.ll / 2 + 0.6, n),
                      rng.uniform(0.35 * t.H, t.H + 0.8, n)], 1)
        f = t.fields(Q, want=("body",))["body"]
        Q = Q[np.abs(f) < 0.1]
        if not len(Q):
            return None
        W = Q @ p["R"].T + p["t"]
        fu, _ = self.field(W, arch="U")
        return float(fu.min()) if np.isfinite(fu.min()) and fu.min() < 50 else None

    def landmarks(self):
        out = {}
        for p in self.teeth:
            tip = self._tooth_tip_local(p["tooth"])
            out["%s%s" % (p["code"], "L" if p["side"] > 0 else "R")] = (p["R"] @ tip + p["t"]).tolist()
        return out

    def measures(self):
        """overjet (upper labial incisal edge to the lower labial face, horizontal) and overbite (vertical overlap of the
        incisal edges) at x = 2.4 mm (the left central incisors), the arch widths, the contact gap after closing"""
        L = self.landmarks()
        zu = 0.5 * (L["U1L"][2] + L["U1R"][2])
        zl = 0.5 * (L["L1L"][2] + L["L1R"][2])
        yu = self._labial_y("U", 2.4, zu + 0.5)
        yl = self._labial_y("L", 2.4, zu + 0.5)
        return dict(overjet=float(yl - yu), overbite=float(zl - zu), fit=self.fit, closed_gap=getattr(self, "closed_gap", None),
                    lower_t=self.lower_t.tolist())
