"""engine/library/mouth/library_items.py: the mouth's library items built as lattices Field Studio can place (its
'lattice' node: a dense true-distance grid in metres). fs-mouth lane, 11-11. No Qt here: Field Studio's
ui/library_ui.py calls build() in a worker and adds the returned specs.

Items: a dentition (upper + lower teeth in occlusion), each arch alone, the gums (upper + lower, bone inside), the
mouth set. Settings (seedable): seed, size, wear, alignment (crowding), spacing, gum recession, arch width, overjet.
Each built part is cached by its settings (engine/library/_cache, git-ignored), so a second pick is instant.

Frame: the parts are written in the dentition frame (x her left, y backward, z up; metres) and placed with rotate
[-90, 0, 0] (Field Studio is y-up with the front at +z), exactly as her store is.
"""
import hashlib
import json
import os
import time

import numpy as np

import arch as AR
import gums as GU
import fieldgrid as FG
import tongue as TG
import jaw as JW

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "_cache", "mouth")
VERSION = 10                               # bump when the generators change (old cache entries are ignored)

SETTINGS = [  # key, label, default, min, max, step (the dialog's controls)
    ("seed", "Seed", 0, 0, 9999, 1),
    ("scale", "Tooth size", 1.0, 0.85, 1.15, 0.01),
    ("wear", "Wear", 0.15, 0.0, 1.0, 0.05),
    ("age", "Age (years; incisal mamelons by age, Alalowi 2020)", 30, 8, 80, 1),
    ("crowding", "Alignment (0 = even)", 0.0, 0.0, 1.0, 0.05),
    ("spacing", "Spacing (mm)", 0.0, 0.0, 1.0, 0.05),
    ("recession", "Gum recession (mm)", 0.0, 0.0, 3.0, 0.1),
    ("ic_u", "Upper canine width (mm)", 33.4, 29.0, 38.0, 0.1),
    ("im_u", "Upper molar width (mm)", 50.2, 45.0, 56.0, 0.1),
    ("overjet_t", "Overjet (mm)", 2.0, 0.5, 5.0, 0.1),
]
DEFAULTS = {k: d for k, _, d, _, _, _ in SETTINGS}

# one albedo per object until the per-part optics are read by the renderers (fs-light): the media fitted to the
# measured colours give these (enamel + dentine 3 mm over dark ~ L* 75; buccal gingiva L* 53 a* 23 b* 15)
ALBEDO = {"teeth": [0.80, 0.74, 0.60], "gum": [0.62, 0.30, 0.28], "bone": [0.86, 0.80, 0.66], "tongue": [0.62, 0.32, 0.32]}

ITEMS = {
    "mouth.dentition": dict(name="Dentition (upper + lower teeth)", parts=["teeth_U", "teeth_L"]),
    "mouth.teeth_upper": dict(name="Upper teeth", parts=["teeth_U"]),
    "mouth.teeth_lower": dict(name="Lower teeth", parts=["teeth_L"]),
    "mouth.gums": dict(name="Gums (upper + lower, with the palate)", parts=["gum_U", "gum_L"]),
    "mouth.tongue": dict(name="Tongue (at rest, shaped by the teeth and palate)", parts=["tongue"]),
    "mouth.jaw": dict(name="Jaw (mandible + maxillary alveolus and hard palate)", parts=["mandible", "maxilla"]),
    "mouth.set": dict(name="Mouth set (teeth + gums + tongue + jaw)", parts=["teeth_U", "teeth_L", "gum_U", "gum_L", "tongue", "mandible", "maxilla"]),
}
PART_INFO = {  # pid, label, albedo key, contact pairs (fs-contact's modes: touch / slide / bond / pass)
    "teeth_U": (1, "Upper teeth", "teeth", {"2": "touch", "3": "touch"}),
    "teeth_L": (2, "Lower teeth", "teeth", {"1": "touch", "4": "touch"}),
    "gum_U": (3, "Upper gums", "gum", {"1": "touch"}),
    "gum_L": (4, "Lower gums", "gum", {"2": "touch", "6": "touch"}),
    "tongue": (5, "Tongue", "tongue", {"1": "slide", "2": "slide", "3": "slide", "4": "slide"}),
    "mandible": (6, "Mandible", "bone", {"2": "touch", "4": "touch"}),
    "maxilla": (7, "Maxilla (alveolus + hard palate)", "bone", {"1": "touch", "3": "touch"}),
}


# (11-12 04:30) the parts are sampled near their surfaces only (fieldgrid.sample_band: a coarse pass, then the fine
# nodes within band + the coarse cell's diagonal; identical after redistance on the mandible's full box and the tongue,
# 0.0000 mm, 0 flips; the mandible 54 s vs 203 s dense), in 300 k-point chunks (1.5 M made a 1.34 GiB array per call
# inside the app's own process)
BAND = dict(coarse=4, margin=1.0, chunk=300_000)


def _key(part, settings, h):
    d = {k: settings.get(k, DEFAULTS[k]) for k in DEFAULTS}
    d.update(part=part, h=h, v=VERSION)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _teeth_lattice(D, arch, h, band=1.6, margin=2.2):
    """one arch's teeth: each tooth sampled in its own box, re-distanced, clamped at +-band, min into the arch lattice
    (the union of non-overlapping teeth; band <= margin, so every face of the box is positive and no value exceeds
    the true distance)"""
    boxes = [D.tooth_box(p, pad=margin) for p in D.teeth if p["arch"] == arch]
    lo = np.min([b[0] for b in boxes], 0)
    hi = np.max([b[1] for b in boxes], 0)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    L = np.full(n, band, np.float32)
    for p in D.teeth:
        if p["arch"] != arch:
            continue
        blo, bhi = D.tooth_box(p, pad=margin)
        i0 = np.floor((blo - lo) / h).astype(int)
        i1 = np.ceil((bhi - lo) / h).astype(int) + 1
        i1 = np.minimum(i1, n)
        plo = lo + i0 * h
        G, _ = FG.sample_band(lambda P: D.tooth_field(p, P)["body"], plo, lo + (i1 - 1) * h, h, band, **BAND)
        Dt = FG.redistance(G, h, plo, band=band)
        Dt = np.clip(Dt, -band, band)
        sl = tuple(slice(i0[k], i0[k] + Dt.shape[k]) for k in range(3))
        L[sl] = np.minimum(L[sl], Dt[:L[sl].shape[0], :L[sl].shape[1], :L[sl].shape[2]])
    return L, lo


def _gum_lattice(D, arch, h, band=1.6, margin=2.5, settings=None):
    G = GU.Gums(D, dict(recession=settings.get("recession", 0.0), seed=settings.get("seed", 0)))
    boxes = [D.tooth_box(p, pad=4.0) for p in D.teeth if p["arch"] == arch]
    lo = np.min([b[0] for b in boxes], 0) - margin
    hi = np.max([b[1] for b in boxes], 0) + margin
    # the vestibular fold: the gum part ends ~11 mm past its margin (here a rounded end by height; in a head it bonds
    # to the lip and cheek mucosa there)
    zf = 19.0 if arch == "U" else -17.5

    def fn(P):
        g = G.field(P, arch=arch)["gum"]
        cut = (P[:, 2] - zf) if arch == "U" else (zf - P[:, 2])
        from teeth import smax
        return smax(g, cut, 1.5)
    Gg, lo = FG.sample_band(fn, lo, hi, h, band, **BAND)
    Dg = FG.redistance(Gg, h, lo, band=band)
    return np.clip(Dg, -band, band), lo


def _tongue_lattice(D, h, settings, band=1.6):
    G = GU.Gums(D, dict(recession=settings.get("recession", 0.0), seed=settings.get("seed", 0)))
    T = TG.Tongue(dict(seed=settings.get("seed", 0)))
    lo, hi = np.array([-26.0, -2.0, -18.0]), np.array([26.0, 62.0, 28.0])

    def fn(P):
        f = T.free(P) - T.s["press"]
        ft, _ = D.field(P)
        g = G.field(P)
        return np.maximum.reduce([f, -ft, -g["gum"], -g["bone"]])
    Gt, lo = FG.sample_band(fn, lo, hi, h, band, **BAND)
    return np.clip(FG.redistance(Gt, h, lo, band=band), -band, band), lo


def _bone_lattice(D, part, h, settings, band=2.0):
    if part == "mandible":
        J = JW.Mandible(dict(seed=settings.get("seed", 0)))
        G = GU.Gums(D)
        lo, hi = np.array([-60.0, -14.0, -46.0]), np.array([60.0, 96.0, 30.0])

        def fn(P):
            m = J.field(P)
            b = G.field(P, arch="L")["bone"]
            ft, _ = D.field(P)
            return np.maximum(np.minimum(m, b), -(ft - 0.2))        # the teeth carved across the ligament
    else:
        G = GU.Gums(D)
        lo, hi = np.array([-34.0, -12.0, -4.0]), np.array([34.0, 56.0, 36.0])

        def fn(P):
            return G.field(P, arch="U")["bone"]
    Gb, lo = FG.sample_band(fn, lo, hi, h, band, **BAND)
    return np.clip(FG.redistance(Gb, h, lo, band=band), -band, band), lo


def build(item, settings=None, h_teeth=0.18, h_gum=0.25, progress=None):
    """-> list of part dicts {key, name, file, sha, h_m, albedo, pid, pairs} (files written in CACHE)"""
    from importlib import import_module
    fv8base = None
    try:
        fv8base = import_module("fieldstudio.core.fv8base")
    except Exception:
        pass
    s = dict(DEFAULTS)
    s.update(settings or {})
    os.makedirs(CACHE, exist_ok=True)
    out = []
    D = None
    for part in ITEMS[item]["parts"]:
        h = h_teeth if part.startswith("teeth") else (0.4 if part in ("mandible",) else h_gum)
        key = _key(part, s, h)
        f = os.path.join(CACHE, "%s_%s.npz" % (part, key))
        if not os.path.exists(f):
            t0 = time.time()
            if D is None:
                if progress:
                    progress("building the dentition (seed %d) ..." % s["seed"])
                D = AR.Dentition({k: s[k] for k in ("seed", "scale", "wear", "age", "crowding", "spacing", "ic_u", "im_u", "overjet_t")})
            if progress:
                progress("sampling %s ..." % part)
            if part.startswith("teeth"):
                L, lo = _teeth_lattice(D, part[-1], h)
            elif part.startswith("gum"):
                L, lo = _gum_lattice(D, part[-1], h, settings=s)
            elif part == "tongue":
                L, lo = _tongue_lattice(D, h, s)
            else:
                L, lo = _bone_lattice(D, part, h, s)
            tmp = f + ".tmp.npz"
            np.savez(tmp, data=(L * 1e-3).astype(np.float32), lo=(lo * 1e-3).astype(np.float64), h=np.float64(h * 1e-3))
            os.replace(tmp, f)
            meta = dict(part=part, settings=s, h_mm=h, lo_mm=list(map(float, lo)), shape=list(L.shape), seconds=time.time() - t0)
            if D is not None:
                meta["measures"] = D.measures()
            json.dump(meta, open(f[:-4] + ".json", "w"), indent=1, default=float)
        pid, label, alb, pairs = PART_INFO[part]
        sha = fv8base.file_sha(f) if fv8base else hashlib.sha256(open(f, "rb").read()).hexdigest()[:24]
        out.append(dict(key=part, name=label, file=os.path.abspath(f), sha=sha, h_m=h * 1e-3, albedo=ALBEDO[alb], pid=pid, pairs=pairs))
    return out


def anchor_from_lips(lips, settings=None):
    """the upper incisal point's place (Field Studio world: y up, z front, metres) from a head's lips found on its
    surface (library_ui.head_lips: {'stomion': (x, y, z), 'labrale_sup': (x, y, z)}), as fs-mouth's s5 seated the set
    in her mouth: the incisal edge 2.5 mm below the stomion (E: the rest display), the upper lip 12.6 mm thick at
    labrale superius over the upper central incisor's labial face (Arnett 1999 women)"""
    s = dict(DEFAULTS)
    s.update(settings or {})
    D = AR.Dentition({k: s[k] for k in ("seed", "scale", "wear", "age", "crowding", "spacing", "ic_u", "im_u", "overjet_t")})
    xs, ys, zs = (float(v) for v in lips["stomion"])
    _, yl, zl = (float(v) for v in lips["labrale_sup"])
    yE = ys - 0.0025
    zr = (yl - yE) * 1e3                                   # labrale superius's height above the incisal edge (mm)
    lab = D._labial_y("U", 2.4, zr)                        # the labial face there, dentition frame (y + backward, mm)
    if not np.isfinite(lab):
        lab = D._labial_y("U", 2.4, 0.4)
    zE = zl - 0.0126 + lab * 1e-3
    return [xs, yE, zE]


def specs(parts, anchor_m=(0.0, 0.0, 0.0), instance=None):
    """Field Studio AddObject specs for built parts: lattice base, rotate [-90, 0, 0], translate = anchor (the upper
    incisal point's place in the world, metres), a part record so the pieces touch without fusing (fs-contact's pairs)"""
    inst = instance or ("library:mouth:%d" % int(time.time() * 1000))
    sp = []
    for k, p in enumerate(parts):
        base = {"id": 7000 + k * 10, "kind": "lattice", "name": p["name"], "params": {"file": p["file"], "sha": p["sha"]}, "children": []}
        sp.append({"name": p["name"], "h": float(p["h_m"]), "base": base, "translate": list(map(float, anchor_m)),
                   "rotate": [-90.0, 0.0, 0.0], "scale": 1.0,
                   "colour": {"albedo": list(p["albedo"]), "variation": 0.0}, "material": {"roughness": 0.25, "specular": 0.5},
                   "part": {"source": inst, "pid": int(p["pid"]), "name": p["name"], "pairs": dict(p["pairs"])}})
    return sp
