"""engine/library/mouth/gums.py: the gingiva and the alveolar bone around a placed dentition, and the palate's mucosa
(fields, mesh-free). fs-mouth lane, 11-11.

Built from what makes them (outcome first, P16):
  - the alveolar bone follows each root (a wall ~1 mm thick round the root: the labial root eminences, the juga, come
    from the roots themselves), its crest ~1.5-2 mm below the CEJ, scalloped as the CEJ is;
  - the gingiva covers the bone: its free margin rides ~0.8 mm up the enamel past the CEJ (the sulcus), its zenith on
    the labial face, the interdental papillae filling the embrasures up toward the contact points (the neighbouring
    collars meet there); attached gingiva, then alveolar mucosa up to the vestibular fold;
  - teeth are carved out of both: gingiva and tooth touch at the sulcus (contact) and bond below it (the junctional
    epithelium), bone and root meet across the periodontal ligament (0.2 mm), so nothing is fused into a tooth.
Frame and units as arch.py (mm, x her left, y backward, z up).

    G = Gums(dentition, settings)
    G.field(P) -> {"gum": .., "bone": ..}  (neg inside)
"""
import numpy as np

from teeth import smax, smin, sstep, _superellipse

DEFAULTS = dict(
    margin=0.8,          # free gingival margin above the CEJ on the labial face (mm) [GING: 0.5-1 coronal in health]
    papilla_gap=1.6,     # (unused since 11-11 20:55: the papilla's height is the cited scallop below)
    gum_thick=1.1,       # the free gingiva's thickness at the margin (mm) [thin-thick phenotype 0.8 / 1.7, numbers.json]
    mucosa=1.2,          # attached gingiva / alveolar mucosa over the bone (mm) [E, between the phenotypes]
    kg_width=4.5,        # the attached (keratinised) gingiva's reach from the tooth (mm) [~5 mm keratinised width, numbers.json]
    bone_wall=1.0,       # bone round the root (mm)
    crest=1.7,           # bone crest below the CEJ (mm) [biologic width ~2 mm]
    pdl=0.2,             # periodontal ligament (mm)
    fold=11.0,           # depth of the vestibular fold below (lower) / above (upper) the gingival margin (mm)
    recession=0.0,       # mm the margin moves apically (seedable state: age, brushing)
    palate_height=20.7,  # palatal vault above the upper occlusal plane at the first molars (mm) [R7 20.7 (2.4) at 31 y; numbers.json 20.7]
    palate_width=39.1,   # palatal width at the first molars (mm) [mouth_2026-11-11/numbers.json, women]
    rugae=1.0,           # 0..1 the transverse palatal ridges' height (seedable: 3-5 ridges, ~1-1.5 mm)
    seed=0,
)


class Gums:
    def __init__(self, dent, settings=None):
        s = dict(DEFAULTS)
        s.update(settings or {})
        self.s, self.D = s, dent
        rng = np.random.default_rng(s["seed"])
        self.jit = {id(p): 1.0 + 0.08 * rng.standard_normal() for p in dent.teeth}

    def _tooth_parts(self, p, P):
        """one tooth's contribution in its frame: (collar field for the gingiva, bone field, tooth body)"""
        t = p["tooth"]
        Q = (P - p["t"]) @ p["R"]
        x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
        F = t.fields(Q, want=("body",))
        body = F["body"]
        am, ad, yf, yl = t.section_dims(np.zeros(len(z)) + 0.0)
        yc, bt = 0.5 * (yf + yl), 0.5 * (yf - yl)
        zc = t.cej_height(x, y, am, ad, yc, bt)                # CEJ height at this angle round the tooth
        a = np.where(x > 0, ad, am)
        ux, vy = (x / a) ** 2, ((y - yc) / bt) ** 2
        wprox = (ux / (ux + vy + 1e-9)) ** 2                    # 1 at the proximal faces, 0 labial / lingual
        s = self.s
        # the gingival margin: labial / lingual the CEJ + margin, its zenith shifted distally (U1 1.1, U2 0.4, U3 0 mm);
        # proximally it rises by the scallop to the papilla's tip (U1 4.3, U2 3.7, U3 4.4 mm; mouth_2026-11-11/numbers.json)
        code = t.T.get("code", "")
        zsh = {"U1": 1.1, "U2": 0.4}.get(code, 0.0)
        scal = {"U1": 4.3, "U2": 3.7, "U3": 4.4, "L1": 3.4, "L2": 3.5, "L3": 3.8}.get(code, 2.8)   # L / posteriors E
        xs_ = x - zsh
        a2 = np.where(xs_ > 0, ad, am)
        ux2 = (xs_ / a2) ** 2
        wl = (ux2 / (ux2 + vy + 1e-9)) ** 1.5
        zm0 = zc * 0 + s["margin"] - s["recession"]
        zm = zm0 + (scal + zc.max() * 0) * wl * 1.0 + 0.0 * wprox
        zm = zm * self.jit[id(p)] + zc * (1 - wl) * 0.0
        # bone: round the root (body + pdl .. + wall), its crest below the CEJ (scalloped like the CEJ, higher between)
        zcr = zc - s["crest"] + 0.0 * wprox
        # (11-11 21:50) the alveolar process is a continuous ridge, not a wall round each root: the bone round a tooth is
        # its cervical cross-section (widened by the wall) carried down its own axis to past the apex, so neighbours
        # overlap; the root is carved out as its socket (field()). A wall round the tapering root left V-valleys between
        # diverging roots that the gum laid over them showed as slits.
        xs_c = x + t.shear * (y - yc)
        sec0 = _superellipse(xs_c, y - yc, am, ad, bt, bt, 2.2, 2.0)            # rounder than the crown (no boxy corners)
        bw = s["pdl"] + s["bone_wall"] + 0.5 * sstep(1.0, 8.0, zcr - z)
        bone = smax(sec0 - bw, z - zcr, 0.6)
        # the root eminences (juga): a smooth swell ADDED to the ridge over the root's labial / buccal side (summed in
        # field(), never a union, so no crease can form between roots; 11-11 22:35). Height by type: the canine
        # eminence the largest (E: anatomy texts; ~1 mm), incisors ~0.5, premolars ~0.4, molars ~0.3
        amp = {"canine": 1.0, "incisor": 0.5, "premolar": 0.4, "molar": 0.3}[t.kind]
        lab_side = sstep(0.0, 2.5, y - yc)
        along = sstep(0.5, 3.0, zcr - z) * (1.0 - sstep(0.55 * t.L, 0.95 * t.L, zcr - z))
        swell = amp * np.exp(-(xs_c / (0.55 * (am + ad) * 0.5 + 0.8)) ** 2) * lab_side * along
        bone = smax(bone, -(z + t.L + 2.5), 1.2)
        # the free gingiva (11-11 21:40, restructured): a collar ~gum_thick round the crown's neck from the margin down to
        # just below the bone crest, where the mucosa over the bone (field()) takes over; neighbouring collars fill the
        # embrasure up to the papilla. (Before, each collar ran ~14 mm down the root with a tilted end: fins between roots.)
        collar = smax(body - s["gum_thick"], z - zm, 0.45)
        collar = smax(collar, (zcr - 2.5) - z, 1.0)
        # the socket: the root and its ligament below the crest (neither bone nor gum there); above the crest the gum
        # touches the tooth (the sulcus: contact, no gap)
        socket = smax(body - s["pdl"], z - zcr, 0.3)
        # the attached gingiva's reach (keratinised): within ~4.5 mm of the neck below the margin; beyond it, past the
        # mucogingival junction, the alveolar mucosa (redder, non-keratinised) [keratinised width ~5 mm, numbers.json]
        kg = smax(body - s["kg_width"], z - (zm + 0.5), 0.8)
        return collar, bone, body, socket, swell, kg

    def roof(self, P):
        """the oral cavity's roof height z at (x, y): the palatal mucosa's air side (relief included), mm"""
        P = np.asarray(P, float)
        x, y = P[:, 0], P[:, 1]
        zs = np.linspace(0.0, 34.0, 69)
        best = np.full(len(P), -1e3)                          # no palate above (x, y): no roof, so no oral air there
        for zz in zs[::-1]:
            Q = np.stack([x, y, np.full(len(P), zz)], 1)
            d = self.palate(Q)
            best = np.where(d < 0, zz, best)
        return best

    def palate(self, P, bone=False):
        """the palatal mucosa's solid (mm, neg inside): a vault from behind the incisors to the soft palate's start,
        bounded by the lingual gingiva (joined by a smooth union in field()); the incisive papilla, the rugae and the
        median raphe as relief. The vault's air side is the oral cavity proper's roof (the tongue's dorsum rests on it)."""
        s = self.s
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        L = self.D.landmarks()
        y6 = 0.5 * (L["U6L"][1] + L["U6R"][1])                     # the first molars' mesiobuccal cusps
        zoc = 0.5 * (L["U6L"][2] + L["U6R"][2])
        # midline height: low behind the incisors (their palatal gingiva), rising to the vault by the premolars
        ymid = np.clip(y, 4.0, 52.0)
        zmid = 9.5 + (zoc + s["palate_height"] - 9.5) * sstep(5.0, y6 - 6.0, ymid)
        # half width of the vault (at its foot on the lingual gingiva): narrow in front, the cited width at the molars
        xw = 0.5 * s["palate_width"] * (0.55 + 0.45 * sstep(2.0, y6, ymid)) + 1.5
        zfoot = 8.5 + 1.5 * sstep(0.0, y6, ymid)
        u = np.clip(np.abs(x) / xw, 0, 1.4)
        roof = zmid - (zmid - zfoot) * np.abs(u) ** 2.6
        # relief: the median raphe (a low ridge), the incisive papilla (a pad behind the centrals), the rugae (waves
        # running out from the raphe, behind the canines)
        rng = np.random.default_rng(int(s["seed"]) + 17)
        rel = 0.25 * np.exp(-(x / 1.2) ** 2) * sstep(8.0, 14.0, y)
        rel += 1.6 * np.exp(-((x / 2.6) ** 2 + ((y - 6.5) / 3.0) ** 2))
        for k in range(4):
            y0 = 9.5 + 3.4 * k + rng.normal(0, 0.4)
            curve = y0 + 0.06 * x ** 2 * (0.6 + 0.2 * k) + 0.8 * np.sin(0.45 * np.abs(x) + k)
            rel += s["rugae"] * (1.2 - 0.2 * k) * np.exp(-((y - curve) / 0.9) ** 2) * sstep(1.5, 3.0, np.abs(x)) * sstep(14.0 + 1.5 * k, 8.0, np.abs(x))
        # solid above the roof (mucosa ~3 mm, then bone: the gum part keeps the mucosa only), from y 3 to 50
        d = -(z - (roof - rel))                                      # neg above the roof surface
        d = np.maximum(d, (roof + 3.2) - z) * 0 + d                  # (the bone above: bone part)
        d = smax(d, z - (roof + 3.2), 0.8)                           # mucosa thickness ~3 mm
        d = smax(d, -(y - 2.5), 1.0)
        d = smax(d, y - (y6 + 22.0), 2.0)
        d = smax(d, np.abs(x) - (xw + 4.0), 1.5)
        # (11-11 21:30) only lingual of the dental arch: the palate's slab ran out laterally through the labial gum
        cv = self.D.arch["U"]
        k = int(np.argmax(cv.x))
        yarch = np.interp(np.abs(x), cv.x[:k + 1], cv.y[:k + 1], right=1e3)
        lingual = (yarch + 1.0) - y
        d = smax(d, lingual, 1.2)
        if bone:                                                     # the hard palate above the mucosa (~5 mm)
            b = smax(-(z - (roof + 3.2)), z - (roof + 8.5), 1.0)
            b = smax(b, -(y - 4.0), 1.5)
            b = smax(b, y - (y6 + 14.0), 2.0)
            b = smax(b, np.abs(x) - (xw + 5.0), 1.5)
            b = smax(b, lingual + 1.0, 1.2)
            return d, b
        return d

    def field(self, P, arch=None):
        """arch: 'U' or 'L' for one arch's gum and bone (its own part); the teeth of both arches are carved out"""
        P = np.asarray(P, float)
        gum = np.full(len(P), 1e3)
        bone = np.full(len(P), 1e3)
        teeth = np.full(len(P), 1e3)
        sock = np.full(len(P), 1e3)
        swell = np.zeros(len(P))
        kg = np.full(len(P), 1e3)
        for p in self.D.teeth:
            if arch and p["arch"] != arch:
                lo, hi = self.D.tooth_box(p, pad=0.5)
                m = np.all((P >= lo) & (P <= hi), axis=1)
                if m.any():
                    teeth[m] = np.minimum(teeth[m], self.D.tooth_field(p, P[m])["body"])
                continue
            lo, hi = self.D.tooth_box(p, pad=6.0)        # bone wall ~2 + its blend 3 + margin (11-11 21:30: 4.5 left seams)
            m = np.all((P >= lo) & (P <= hi), axis=1)
            if not m.any():
                continue
            c, b, tb, so, sw, kgv = self._tooth_parts(p, P[m])
            gum[m] = smin(gum[m], c, 1.2)                      # the free gingiva collars; neighbours fill the embrasure
            bone[m] = smin(bone[m], b, 4.0)                    # one alveolar process, not separate pegs
            teeth[m] = np.minimum(teeth[m], tb)
            sock[m] = np.minimum(sock[m], so)
            swell[m] = swell[m] + sw
            kg[m] = smin(kg[m], kgv, 1.5)
        bone = bone - swell                                         # the juga (smooth, summed)
        gum = smin(gum, bone - self.s["mucosa"], 2.2)              # the mucosa over the bone (attached gingiva, alveolar mucosa);
                                                                    # a wide blend: no crease where the free gingiva meets it
        # the vestibular fold: the gum ends in a rounded fold, deeper over the roots (upper: above, lower: below)
        # (the fold joins the lip / cheek mucosa in a head: a bond pair at the fold)
        if arch in (None, "U"):                                     # the palatal mucosa joins the upper gum, the
            pal, palb = self.palate(P, bone=True)                   # hard palate the upper bone
            gum = smin(gum, pal, 2.5)
            bone = smin(bone, palb, 2.5)
        if arch:                                                    # the fold: a rounded end ~11 mm past the margins
            zf = 19.0 if arch == "U" else -17.5
            cut = (P[:, 2] - zf) if arch == "U" else (zf - P[:, 2])
            if arch == "U":                                         # labial / buccal only: the palate continues inside
                cv = self.D.arch["U"]
                k = np.argmax(cv.x)
                yarch = np.interp(np.abs(P[:, 0]), cv.x[:k + 1], cv.y[:k + 1])
                w_out = sstep(1.0, -2.0, P[:, 1] - yarch)
                cut = np.where(w_out > 0.5, cut, -1e3) * 1.0 + 0.0
                cut = cut * w_out + (-6.0) * (1 - w_out)
            gum = smax(gum, cut, 1.5)                               # (the bone runs on past the fold: it holds the roots)
            zb = 27.0 if arch == "U" else -25.0                     # the bone part ends past the apices, rounded
            bone = smax(bone, (P[:, 2] - zb) if arch == "U" else (zb - P[:, 2]), 2.0)
        gum = np.maximum(gum, -teeth)                               # teeth carved out: contact, not fusion
        gum = np.maximum(gum, -sock)                                # nor in the sockets
        bone = np.maximum(bone, -sock)                              # the ligament between root and bone
        # (11-12 04:05) and the bone never inside a tooth at any height: the smooth union (blend 4 mm) that makes one
        # alveolar process swells it ~1 mm past each tooth's own crest, where the socket (below the crest) did not carve
        # it: the maxilla overlapped the upper teeth's necks by up to 0.46 mm (s5c on mouth2). Carved like the mandible:
        # by the tooth bodies plus the ligament.
        bone = np.maximum(bone, -(teeth - self.s["pdl"]))
        gum = np.maximum(gum, -(bone + 0.0))                        # gum on the bone (its inner face on the bone's)
        if arch in (None, "U"):
            kg = smin(kg, self.palate(P), 1.0)                      # the palatal mucosa is masticatory (keratinised) too
        return {"gum": gum, "bone": bone, "teeth": teeth, "kg": kg}
