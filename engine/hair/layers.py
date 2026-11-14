"""Hair Studio LAYERS (Apache-2.0): a look composed from five files, each the only owner of its own settings, so the
hair is set once per person and no style can change it.

Why: a monolithic style file (style.py) lets a look reach into the fibre, the follicles and the root count - a
'style' round could quietly re-tune the hair it was meant only to style. Design (10-31): the studio has one layer for the
hair (type, length, characteristics, thickness), one for the style, and one for colour; a style
round styles the hair and does not re-tune its physics or characteristics. So
every leaf of the schema style.S has exactly one OWNER layer (OWNER below, longest prefix wins), and a layer file
that holds a key another layer owns is refused: validate_layer RAISES, it does not warn.

    hair         the person's hair, set once: roots, hairline (baby hairs too), follicle, fibre (not its water),
                 radius, waves, clump, frizz, locks (not the hand's split, the brush lanes or the drying section),
                 oil (the sebum, not the product)
    style        one look: part, flow, the acts (brush, dry, tuck, hand, tousle, finish, face), gather, cut, length,
                 ears, corridor, lift, the hours worn, the product, the split / lanes / sections, the children's mode
                 and the styled ringlets, the sideburns
    colour       colour, material - changes without re-growing
    environment  the air the look is worn in: wear.rh and the fibre's water (fibre.wet)
    sim          the solver's resolution and the case's body: grow (the rest of it), points, body, volume, and the
                 engine's switches flow.source (comb / grow) and flow.order (grow then style, or the legacy order)

A layer FILE is json with only its own keys, nested as in style.S, plus name / notes / description / extends.
"extends" names another file of the SAME layer: it is looked for beside the file, then in <dir>/<layer>/ and <dir>
for each of the given dirs; a file in another layer's directory, or one holding another layer's keys, is refused.
compose() deep-merges DEFAULTS with the layers into exactly style.load's form (plus _layers, _layer_names), so
generate takes it unchanged; split() turns a monolithic style into the five minimal layer files.

    python -m engine.hair.layers table                      every schema leaf with its layer
    python -m engine.hair.layers split STYLE --out DIR [--styles D] [--names hair=..,style=..,colour=..,environment=..,sim=..]
                                                            -> DIR/<layer>/<name>.json (only the keys off DEFAULTS)
    python -m engine.hair.layers check HAIR STYLE COLOUR [--env E] [--sim X] [--root DIR]
                                                            the resolved summary, or the guard's error (exit 2)
"""
import copy
import json
import os
import sys

from .style import S, DEFAULTS, deep_merge, validate, load as load_style

LAYERS = ("hair", "style", "colour", "environment", "sim")
META = ("name", "notes", "description", "extends")     # allowed at the top of every layer file

# the table (10-31): block -> layer, with the keys that leave their block's layer named in full. The longest
# matching entry owns a key; "x.y_*" matches every key of x starting with y_. A key no entry matches has no layer.
OWNER = {
    # hair: the person's hair, set once
    "roots": "hair", "hairline": "hair", "follicle": "hair", "fibre": "hair", "radius": "hair", "waves": "hair",
    "clump": "hair", "frizz": "hair", "locks": "hair", "oil": "hair",
    # her length and haircut are part of the head grown once (10-31: one layer for the hair: type, length,
    # characteristics, thickness); a new haircut is a new hair file. The seed is this head's draw of locks
    # and waves; the follicular units and the natural wave in the rods describe the hair
    "length": "hair", "cut": "hair", "grow.sideburn": "hair", "seed": "hair",
    "grow.fu_*": "hair", "grow.rod_waves": "hair", "grow.wave_random_mm": "hair",
    # style: one look
    "part": "style", "flow": "style", "brush": "style", "dry": "style", "tuck": "style", "hand": "style",
    "tousle": "style", "finish": "style", "face": "style", "gather": "style",
    "ears": "style", "corridor": "style", "lift": "style",
    "wear": "style", "oil.product": "style",
    "locks.split": "style", "locks.brush_lane_mm": "style", "locks.section_mm": "style", "locks.phase_field_mm": "style",
    "grow.children_mode": "style", "grow.ringlet_*": "style",
    # the short hairs' set and laid route and the growth along the stroke follow the acts (dry, tuck, brush)
    "grow.short_set": "style", "grow.short_laid": "style", "grow.short_laid_share": "style",
    "grow.grow_along_stroke": "style",
    # colour: re-coloured without re-growing
    "colour": "colour", "material": "colour",
    # environment: the air it is worn in
    "wear.rh": "environment", "fibre.wet": "environment",
    # sim: the solver and the case's body
    "grow": "sim", "points": "sim", "body": "sim", "volume": "sim", "flow.source": "sim",
    # grow once then style, or the legacy order (generate flow.order): how the engine runs the look, not the look -
    # so a style file cannot switch it (a sim file extends the solver's file with it: layers/sim/hs3_full_after)
    "flow.order": "sim",
}


class LayerError(ValueError):
    """a layer file holds a key it does not own, or extends a file of another layer"""


def _hit(entry, path):
    if entry.endswith("*"):
        return path.startswith(entry[:-1])
    return path == entry or path.startswith(entry + ".")


def matches(path):
    """every OWNER entry matching a dotted key, longest first"""
    return sorted((e for e in OWNER if _hit(e, path)), key=len, reverse=True)


def owner(path):
    """the layer owning a dotted key ('fibre.wet' -> 'environment'), 'meta' for name / notes / description /
    extends, None when no entry matches"""
    if path in META:
        return "meta"
    m = matches(path)
    return OWNER[m[0]] if m else None


def leaves(node=S, pre=""):
    """every leaf of the schema as a dotted key"""
    out = []
    for k, v in node.items():
        if isinstance(v, dict):
            out += leaves(v, pre + k + ".")
        else:
            out.append(pre + k)
    return out


def _walk(raw, node=S, pre=""):
    """(dotted key, value) for each leaf a raw or resolved style dict sets, following style.S: a dict under a schema
    block is walked into, anything else (a schema leaf, an unknown key, a block given as a non-dict) is one leaf"""
    out = []
    for k, v in raw.items():
        if not pre and (k.startswith("_") or k in META):
            continue
        spec = node.get(k) if isinstance(node, dict) else None
        if isinstance(spec, dict) and isinstance(v, dict):
            out += _walk(v, spec, pre + k + ".")
        else:
            out.append((pre + k, v))
    return out


def table():
    """[(dotted key, layer, default)] for every schema leaf"""
    rows = []
    for p in leaves():
        d = DEFAULTS
        for k in p.split("."):
            d = d[k]
        rows.append((p, owner(p), d))
    return rows


def validate_layer(layer, raw, where=""):
    """RAISES LayerError when the layer file `raw` holds a key another layer owns (or one no layer owns), naming each
    key and its owner; returns the number of keys it sets"""
    if layer not in LAYERS:
        raise LayerError("unknown layer %r (one of %s)" % (layer, ", ".join(LAYERS)))
    bad = []
    pairs = _walk(raw)
    for path, v in pairs:
        node = S
        for k in path.split("."):
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict) and not isinstance(v, dict):
            # a whole block replaced by a non-object would wipe keys other layers own (review 10-31: "grow": null
            # in a sim file emptied the style's grow.children_mode)
            bad.append("%s is a block: give it as an object of its keys" % path)
            continue
        own = owner(path)
        if own is None:
            bad.append("%s: no layer owns this key (it is not in style.S and no block of it is assigned)" % path)
        elif own != layer:
            bad.append("%s belongs to the %s layer: a %s file cannot change %s settings; put it in the %s file"
                       % (path, own, layer, own, own))
    if bad:
        raise LayerError("%s%s file refused:\n  %s" % ((where + ": ") if where else "", layer, "\n  ".join(bad)))
    return len(pairs)


def _layer_of_dir(path):
    """the layer a file's directory is named for (layers/<layer>/x.json), else None"""
    d = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return d if d in LAYERS else None


def _find(name, where):
    if os.path.isfile(name):
        return name
    for d in where:
        for c in (os.path.join(d, name), os.path.join(d, name + ".json")):
            if os.path.isfile(c):
                return c
    raise FileNotFoundError("layer file %r not found in %s" % (name, where))


def _search(layer, dirs):
    out = []
    for d in dirs:
        out += [os.path.join(d, layer), d]
    return out


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _nm(path, raw):
    """a layer's name: its own, else its file's base name, else 'unnamed' (a name ends up in the run's file name)"""
    return raw.get("name") or (os.path.splitext(os.path.basename(path))[0] if path else "unnamed")


def load_layer(layer, src, dirs=()):
    """one layer resolved through its extends chain -> (raw dict without extends, [paths], [names]); src is a name
    (looked for in <dir>/<layer>/ and <dir>), a path, or a dict (a layer held in memory)"""
    if isinstance(src, dict):
        path, raw = None, copy.deepcopy(src)
    else:
        path = os.path.abspath(_find(src, _search(layer, dirs)))
        other = _layer_of_dir(path)
        if other is not None and other != layer:
            raise LayerError("%s: a %s file given as the %s layer (it lies in layers/%s/): a %s file cannot change %s "
                             "settings; put it in the %s file" % (path, other, layer, other, layer, other, other))
        raw = _read(path)
    validate_layer(layer, raw, path or "<%s dict>" % layer)
    chain = [(path, raw)]
    seen = {path} if path else set()
    while chain[-1][1].get("extends"):
        here, ext = chain[-1]
        p = os.path.abspath(_find(ext["extends"], ([os.path.dirname(here)] if here else []) + _search(layer, dirs)))
        if p in seen:
            raise LayerError("extends loop at %s" % p)
        seen.add(p)
        other = _layer_of_dir(p)
        if other is not None and other != layer:
            raise LayerError("%s extends %s, a %s file: extends stays within one layer (a %s file cannot change %s "
                             "settings; put it in the %s file)" % (here or "<%s dict>" % layer, p, other, layer,
                                                                   other, other))
        r = _read(p)
        validate_layer(layer, r, p)
        chain.append((p, r))
    merged = {}
    for _, c in reversed(chain):
        merged = deep_merge(merged, {k: v for k, v in c.items() if k != "extends"})
    return merged, [p for p, _ in chain], [_nm(p, c) for p, c in chain]


def compose(hair, style, colour, environment=None, sim=None, dirs=()):
    """the resolved style from the layers -> (style, warnings), in style.load's form: DEFAULTS deep-merged with
    hair, style, colour, environment, sim (a missing environment / sim = DEFAULTS'), name =
    <style>__<hair>__<colour>[__<environment>][__<sim>] (each layer's name, else its file's base name),
    the style layer's notes / description, _file = the style layer's file, _chain = ['layer:name', ...] through each
    layer's extends, _layers = {layer: abs path} (None for a dict), _layer_names = {layer: name}; then style.validate
    (unknown keys, ranges clipped - warnings, as style.load)"""
    given = dict(hair=hair, style=style, colour=colour, environment=environment, sim=sim)
    st = copy.deepcopy(DEFAULTS)
    paths, names, chain, raws = {}, {}, [], {}
    for L in LAYERS:
        if given[L] is None:
            continue
        raw, ps, ns = load_layer(L, given[L], dirs)
        st = deep_merge(st, {k: v for k, v in raw.items() if k not in META})
        raws[L] = raw
        paths[L] = ps[0]
        names[L] = raw.get("name") or ns[0]
        chain += ["%s:%s" % (L, n) for n in ns]
    st["name"] = "__".join(names[L] for L in ("style", "hair", "colour", "environment", "sim") if L in names)
    for k in ("notes", "description"):
        if k in raws["style"]:
            st[k] = raws["style"][k]
    st["_file"] = paths["style"]
    st["_chain"] = chain
    st["_layers"] = paths
    st["_layer_names"] = names
    return st, validate(st)


def _same(a, b):
    """equal in value AND json type (3 != 3.0: the file says which)"""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return False, None
        d = d[k]
    return True, d


def _put(d, path, v):
    ks = path.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = copy.deepcopy(v)


def split(style, dirs=(), names=None):
    """a monolithic style (a name in dirs or a path) -> {layer: raw dict}: style.load's resolved style cut by OWNER,
    each layer holding only the leaves that differ from DEFAULTS (minimal files) plus a name (names[layer], else
    <style name>_<layer>) and a note of where it came from. compose(*split(x)) == style.load(x) but for the
    provenance (_file, _chain, _layers, name, notes)."""
    st, _ = load_style(style, list(dirs))
    names = names or {}
    out = {L: {} for L in LAYERS}
    for path, v in _walk(st):
        has, d = _get(DEFAULTS, path)
        if has and _same(v, d):
            continue
        own = owner(path)
        if own in (None, "meta"):
            raise LayerError("%s: %s has no layer; it cannot be split" % (st["_file"], path))
        _put(out[own], path, v)
    base = st.get("name", "unnamed")
    res = {}
    for L in LAYERS:
        res[L] = {"name": names.get(L) or "%s_%s" % (base, L),
                  "notes": "split from %s (%s) by engine/hair/layers.py: the %s keys that differ from DEFAULTS"
                           % (base, os.path.basename(st["_file"]), L)}
        res[L].update(out[L])
    return res


def _summary(st, warn):
    lines = ["name      %s" % st["name"]]
    for L in LAYERS:
        if L in st["_layers"]:
            lines.append("%-11s %s  (%s)" % (L, st["_layer_names"][L], st["_layers"][L]))
        else:
            lines.append("%-11s DEFAULTS" % L)
    lines.append("chain     %s" % " + ".join(st["_chain"]))
    f, g = st["fibre"], st["grow"]
    lines.append("hair      fibre.type %r, roots.count %s, radius.root_mm %s, oil.amount %s"
                 % (f["type"], st["roots"]["count"], st["radius"]["root_mm"], st["oil"]["amount"]))
    lines.append("style     flow.source %r, part.type %r, brush.on %s, tuck.on %s, cut.type %r"
                 % (st["flow"]["source"], st["part"]["type"], st["brush"]["on"], st["tuck"]["on"], st["cut"]["type"]))
    lines.append("colour    melanin %s, redness %s, ash %s" % (st["colour"]["melanin"], st["colour"]["redness"],
                                                              st["colour"]["ash"]))
    lines.append("env       wear.rh %s, fibre.wet %s" % (st["wear"]["rh"], f["wet"]))
    lines.append("sim       guides %s, seg_max_mm %s, points %s, seed %s, body %s"
                 % (g["guides"], g["seg_max_mm"], st["points"], st["seed"], st["body"]))
    lines += ["WARNING %s" % w for w in warn]
    return "\n".join(lines)


def main(a):
    if not a or a[0] == "table":
        rows = table()
        for p, L, d in rows:
            print("| `%s` | %s | %s |" % (p, L, json.dumps(d)))
        n = {}
        for _, L, _ in rows:
            n[L] = n.get(L, 0) + 1
        print("\n%d leaves: %s" % (len(rows), ", ".join("%s %d" % (L, n.get(L, 0)) for L in LAYERS + ("meta",))))
        return 0

    def opt(name, default=None, many=False):
        vals = [a[i + 1] for i in range(len(a) - 1) if a[i] == name]
        return vals if many else (vals[-1] if vals else default)

    if a[0] == "split":
        dirs = opt("--styles", many=True)
        names = dict(kv.split("=", 1) for kv in opt("--names", "").split(",") if kv)
        out = opt("--out")
        res = split(a[1], dirs, names)
        for L in LAYERS:
            n = len(_walk(res[L]))
            if out:
                os.makedirs(os.path.join(out, L), exist_ok=True)
                p = os.path.join(out, L, res[L]["name"] + ".json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(res[L], f, indent=1)
                    f.write("\n")
                print("%-11s %3d keys -> %s" % (L, n, p))
            else:
                print("%-11s %3d keys %s" % (L, n, json.dumps(res[L])))
        return 0
    if a[0] == "check":
        roots = opt("--root", many=True) or [os.path.join(os.getcwd(), "layers")]
        try:
            st, warn = compose(a[1], a[2], a[3], environment=opt("--env"), sim=opt("--sim"), dirs=roots)
        except LayerError as e:
            print("REFUSED", e)
            return 2
        print(_summary(st, warn))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
