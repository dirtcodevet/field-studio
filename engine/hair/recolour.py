"""Hair Studio recolour (numpy, Apache-2.0): a grown npz -> the same hair in another colour, in seconds, no re-growing.

Why (10-31): the studio in parts, the hair set once per person and colour addressed on its
own. Colour is a stage of its own (colour.colourize): it reads only what the npz saves, so the colour can change while
every point, radius, kind, lock and guide stays byte for byte as grown.

    python -m engine.hair.recolour IN.npz [--colour LAYER [--layers DIR] [--from-style]] [--out OUT.npz] [--seed N]
                                   [--case-loader LOADER.py] [--uncompressed]

    --colour LAYER       a colour layer (engine/hair/layers.py: layers/colour/<name>.json, a path or a name): a json
                         object with a "colour" and / or a "material" block (plus name / notes / description /
                         extends within the colour layer); each block given REPLACES the run's - the schema's defaults
                         under the layer's keys, as layers.compose builds a look - validated against style.S (an
                         unknown key refused, a value out of range clipped with a warning). Any other block is
                         refused: a colour file cannot change the hair or the style. Without --colour the run's own
                         colour is drawn again (with --seed: another realisation).
       --layers DIR      where a layer NAME is looked for (DIR/colour/, DIR); repeatable
       --from-style      LAYER is a whole style file (its "extends" chain resolved): only its colour and material
                         are taken, everything else in it is ignored, by request
    --out OUT.npz        default IN_<layer name>.npz next to IN; never IN itself
    --seed N             the colour seed (default: the npz's colour_seed, else the style's seed)
    --case-loader PATH   only for an npz from before the colour stage (no ear_z_top): a python file with
                         load(style) -> (scalp, ...) that rebuilds the run's scalp, e.g. the case's hs3_scalp.py
    --uncompressed       np.savez instead of np.savez_compressed (larger file, written several times faster)
-> OUT.npz: every array of IN byte-identical except melp, mel, red, mel_root, sigma, material_json, style_json (its
   colour / material blocks replaced, so the file says what colour it is) and colour_seed (+ ear_z_top when the case
   loader supplied it); OUT.json: IN.json's report with "colour" replaced and a "recolour" record.

An npz from before the colour stage has no bun (the bundle id per strand): each strand is then its own bundle (said in
the report), and its colour realisation is the stage's, not the one it was grown with.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

from . import colour as CO
from . import scalp as SC
from . import style as SY
try:                                     # the layers' guard (engine/hair/layers.py) when it is there
    from . import layers as LY
except ImportError:
    LY = None

COLOUR_ARRAYS = ("melp", "mel", "red", "mel_root", "sigma", "material_json")
LAYER_BLOCKS = ("colour", "material")
LAYER_META = ("name", "notes", "description")


class SavedScalp(SC.Scalp):
    """the scalp frame the colour reads (sph, d_inside_mm, ears' z_top), from the arrays an npz saves - the same
    methods as scalp.Scalp on the same numbers, so its colours equal the generator's; no surface"""

    def __init__(self, centre, grid, thb, ear_z_top):
        self.c = np.asarray(centre, np.float64)
        self.grid, self.thb = np.asarray(grid, np.float64), np.asarray(thb, np.float64)
        self.ears = {i: {"z_top": float(z)} for i, z in enumerate(np.asarray(ear_z_top, np.float64))}


def read_layer(path, from_style=False, dirs=()):
    """-> ({block: dict}, warnings): the colour layer's colour / material blocks, each the schema's defaults under the
    file's keys (as layers.compose builds a look), validated; refuses any other block (or, with from_style, takes only
    those two from a whole style). The colour layer is read by layers.load_layer when engine/hair/layers.py is there
    (its guard and its 'extends' within the colour layer; a name looked for in <dir>/colour/ and <dir> of dirs), else
    by the same rule here: only colour, material and name / notes / description."""
    if from_style:
        st, _ = SY.load(path, dirs)
        raw = {k: st[k] for k in LAYER_BLOCKS}
    elif LY is not None:
        try:
            raw, _, _ = LY.load_layer("colour", path, dirs)
        except LY.LayerError as e:
            raise SystemExit("colour layer refused - %s (a whole style file: --from-style)" % e)
    else:
        raw = json.load(open(path))
        if not isinstance(raw, dict):
            raise SystemExit("colour layer %s: not a json object" % path)
        other = sorted(k for k in raw if k not in LAYER_BLOCKS and k not in LAYER_META)
        if other:
            raise SystemExit("colour layer %s: refused - only %s (and %s) belong in a colour layer, it sets %s; a "
                             "colour file cannot change the hair or the style (a whole style file: --from-style)"
                             % (path, list(LAYER_BLOCKS), list(LAYER_META), other))
    if not any(k in raw for k in LAYER_BLOCKS):
        raise SystemExit("colour layer %s: no colour or material block" % path)
    out, warn = {}, []
    for b in LAYER_BLOCKS:
        if b not in raw:
            continue
        if not isinstance(raw[b], dict):
            raise SystemExit("colour layer %s: %s should be an object" % (path, b))
        blk = SY.deep_merge(SY.DEFAULTS[b], raw[b])
        w = SY.validate(blk, SY.S[b], b + ".")
        bad = [x for x in w if x.startswith("unknown key")]
        if bad:
            raise SystemExit("colour layer %s: refused - %s" % (path, "; ".join(bad)))
        warn += w
        out[b] = blk
    return out, warn


def _load_case_scalp(loader, style):
    spec = importlib.util.spec_from_file_location("case_scalp_loader", os.path.abspath(loader))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(loader)))
    spec.loader.exec_module(mod)
    got = mod.load(style, log=lambda m: None) if "log" in mod.load.__code__.co_varnames else mod.load(style)
    return got[0] if isinstance(got, tuple) else got


def scalp_for(Z, style, loader=None):
    """-> (scalp, how, extra arrays): the saved frame when the npz has ear_z_top, else the case loader's scalp (its
    frame checked against the npz's centre / grid / thb)"""
    if "ear_z_top" in Z and all(k in Z for k in ("centre", "grid", "thb")):
        return SavedScalp(Z["centre"], Z["grid"], Z["thb"], Z["ear_z_top"]), "saved (centre, grid, thb, ear_z_top)", {}
    if not loader:
        raise SystemExit("this npz predates the colour stage (no ear_z_top): give --case-loader, the case's scalp "
                         "loader (e.g. <case folder>/hs3_scalp.py)")
    sc = _load_case_scalp(loader, style)
    for k, v in (("centre", sc.c), ("grid", sc.grid), ("thb", sc.thb)):
        if k in Z and not np.array_equal(np.asarray(Z[k], np.float64), np.asarray(v, np.float64)):
            raise SystemExit("the case loader's scalp differs from the npz's %s: not the scalp this hair grew on" % k)
    ez = np.array([float(e["z_top"]) for e in sc.ears.values()], np.float64)
    return sc, "case loader %s" % loader, {"ear_z_top": ez}


def recolour(Z, layer=None, seed=None, loader=None):
    """Z: the npz's arrays (dict). layer: {block: dict} from read_layer (None: the run's own colour). -> (new colour
    arrays incl. style_json and colour_seed, report)"""
    style = json.loads(str(Z["style_json"]))
    for b, blk in (layer or {}).items():
        style[b] = blk
    st = SY.effective(style)
    if seed is None:
        seed = int(Z["colour_seed"]) if "colour_seed" in Z else int(style["seed"])
    sc, how, extra = scalp_for(Z, style, loader)
    bun = Z["bun"] if "bun" in Z else None
    col = CO.colourize(Z["pts"], Z["kind"], Z["din"], Z["lock"], Z["sub"], bun, sc, st, seed)
    rep = col.pop("report")
    rep["scalp"] = how
    rep["ash"] = st["colour"]["ash"]
    if bun is None:
        rep["bun"] = "absent (npz from before the colour stage): each strand its own bundle"
    new = dict(col, style_json=json.dumps(style), colour_seed=np.int64(seed), **extra)
    return new, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--colour")
    ap.add_argument("--layers", action="append", default=[])
    ap.add_argument("--from-style", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--case-loader")
    ap.add_argument("--uncompressed", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    src = os.path.abspath(a.npz)
    layer, warn = (read_layer(a.colour, a.from_style, a.layers) if a.colour else (None, []))
    for w in warn:
        print("COLOUR WARNING", w, flush=True)
    tag = (os.path.splitext(os.path.basename(a.colour))[0] if a.colour
           else "seed%d" % a.seed if a.seed is not None else "recolour")
    out = os.path.abspath(a.out or src[:-4] + "_" + tag + ".npz")
    if not out.endswith(".npz"):        # numpy appends it: the guard below must see the file actually written
        out += ".npz"
    if os.path.normcase(out) == os.path.normcase(src):
        raise SystemExit("--out is the input: the grown file is kept as it is (write the recolour beside it)")
    with np.load(src) as f:
        Z = {k: f[k] for k in f.files}
    t_load = time.time() - t0
    t1 = time.time()
    new, rep = recolour(Z, layer, a.seed, a.case_loader)
    t_colour = time.time() - t1
    Z.update(new)
    t2 = time.time()
    (np.savez if a.uncompressed else np.savez_compressed)(out, **Z)
    t_write = time.time() - t2
    rec = {"from": src, "colour_file": os.path.abspath(a.colour) if a.colour else None, "from_style": a.from_style,
           "seed": int(new["colour_seed"]), "warnings": warn, "replaced": list(COLOUR_ARRAYS) + ["style_json",
           "colour_seed"] + [k for k in ("ear_z_top",) if k in new],
           "seconds": {"load": round(t_load, 2), "colour": round(t_colour, 2), "write": round(t_write, 2),
                       "total": round(time.time() - t0, 2)},
           "npz_mb": round(os.path.getsize(out) / 2 ** 20, 1)}
    jin = src[:-4] + ".json"
    full = json.load(open(jin)) if os.path.exists(jin) else {}
    full.update(colour=rep, recolour=rec, out=out, material=json.loads(new["material_json"]), ash=rep["ash"])
    json.dump(full, open(out[:-4] + ".json", "w"), indent=1, default=float)
    print(json.dumps({"out": out, "colour": {k: rep[k] for k in ("seed", "strands", "melanin_point_p5_50_95")
                                             if k in rep}, "recolour": rec["seconds"]}, indent=1))
    return out


if __name__ == "__main__":
    main()
