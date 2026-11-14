"""The grown head (numpy, Apache-2.0): the Hair stage of DESIGN_grow_then_style.md - her hair grown ONCE with no act,
cut on its natural fall, rested, and cached, so a style starts from hair that is already there.

Why (10-31): the hair is grown once, then styled from there; as in real life, a hand run through the hair,
a ponytail, a brush run through it or product added all happen after the hair is
already there. Until now generate._grow ran the acts during growth (each new length
combed as it emerged); here the growth has no act at all, and nothing in it is tuned: the same prelude, the same settle,
the same cut as generate's cut.taken 'fall'.

    head = grow_head(scalp, collider, style, menton_z, case=None)      -> GrownHead, two snapshots:
        'grown'   SE.grow(rods, body, acts=None) - exactly the natural fall generate grows for cut.taken 'fall' (rods_f)
        'rested'  her haircut read on that fall (generate._cut_plan, _cut, rods.members(cut)), the friction anchors
                  carried over the cut by (guide, segment, sample in segment), the release solves with
                  actions.lag_stick, the gated final settle - all with no act
    save(head, path) -> path.npz + path.json;   head = load(path)
    rods, state = instantiate(head, scalp, collider, snapshot='rested')

instantiate is the ONLY way to a live head, for the in-memory head and the loaded one alike: it re-runs the cheap,
deterministic part (generate._roots_part_kinds and _prelude: roots, follicles, fibre, lengths, guides and members, the
body grid, fresh rods), refuses a rebuild whose fingerprints (X, gi, mg, mL, ms; per snapshot l, s, B, W) or settings
differ from the head's, then writes the saved history into the fresh rods and State. It never changes the head.

What is saved, per snapshot (history: it cannot be recomputed): the rods' directions D and rest turning rho, the wave's
edits wave_x / wave_rho_w and the wave block (A, lam, s0, phase, rnd, hx) with wave_n0; the cut lengths mc; the State's
act_full, Ka, the static-friction anchors anc / anc_N / anc_M / anc_n (fixed points in space with their load: re-taking
them where the hair lies would let a rested head slide with no cause), the aligned-count alignment sal (set per stage),
stage_index and evals. A few MB at 500 guides.

The key (meta['key']): sha256 over the resolved hair + environment + sim settings (layers.OWNER), the style-owned keys
the prelude reads until they move (HEAD_STYLE_KEYS: the part's partition of the guides, flow.behind_ears, the drying
section locks.section_mm), the case (its ids, the scalp's and the collider's geometry, menton_z), the code that grows
the head (CODE_FILES whole; generate's and actions' functions the stage runs) and FORMAT. oil.product is forced to 0:
product is a style pass, put on hair that is already there. meta['key_parts'] keeps the parts apart, so a caller can
tell a head grown by older code from a head of other hair.
"""
import copy
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np

from . import actions as AC
from . import generate as GEN
from . import layers as LY
from . import settle as SE
from . import style as SY
from .ram import peak_mb

FORMAT = 2       # 2 (10-31 review): the follicle directions / root normals and the scalp's normals in the checks
SNAPSHOTS = ("grown", "rested")
KEY_LAYERS = ("hair", "environment", "sim")
# style-owned keys the grown head reads (generate._roots_part_kinds / _prelude): the part splits the guides (a guide
# never straddles it), flow.behind_ears picks the sideburn kinds, locks.section_mm sets the wave's drying sections
HEAD_STYLE_KEYS = ("part", "flow.behind_ears", "locks.section_mm", "locks.phase_field_mm")
CODE_FILES = ("settle", "fibre", "follicle", "children", "grid", "form", "scalp")
CODE_FUNCS = {"generate": ("_roots_part_kinds", "_ear_boxes", "_prelude", "_tip_rule", "_cut_plan", "_cut"),
              "actions": ("part_side", "lag_stick"),
              "head": ("grow_head", "_rest", "_sample_table", "_carry_anchors")}
_HERE = os.path.dirname(os.path.abspath(__file__))


class HeadMismatch(ValueError):
    """the rebuild does not match the head: other roots, guides, members or segments (fingerprints), other hair /
    environment / sim settings, or another case"""


class GrownHead:
    """a grown head: meta (json: key, key_parts, settings, menton_z, case, fingerprints, report) and snaps
    {'grown' | 'rested': {name: array}}. Use it through instantiate(); its arrays are never written to."""

    def __init__(self, meta, snaps):
        self.meta, self.snaps = meta, snaps

    @property
    def key(self):
        return self.meta["key"]

    def __repr__(self):
        return "GrownHead(%s, %s, %s)" % (self.key[:16], self.meta.get("name"), "/".join(self.snaps))


# ---------------------------------------------------------------- settings, key, fingerprints
def head_settings(style):
    """the settings the Hair stage grows from: a copy of the style with oil.product 0 (the product is a style pass) and
    without flow.order (it says how a run uses the head - grow then style, or the acts while it grows - not what the
    head is; review 10-31: kept, it made the head API's key differ from generate's for the same hair)"""
    s = copy.deepcopy(style)
    s["oil"] = dict(s.get("oil") or {}, product=0.0)
    if isinstance(s.get("flow"), dict):
        s["flow"] = {k: v for k, v in s["flow"].items() if k != "order"}
    return s


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError("not json: %r" % type(o))


def _public(st):
    """the style without its private keys (_file, _chain, _layers ...), as plain json values"""
    return json.loads(json.dumps({k: v for k, v in st.items() if not k.startswith("_")}, default=_jsonable))


# leaves added to the settings after heads were keyed: at their default they leave the key as it was (a head grown
# before the leaf existed is that same head - 11-01: settle's contact_ear_k turned every saved head into a mismatch);
# set to anything else they key the head apart
ADDED_AT_DEFAULT = {"grow.contact_ear_k": 1.0, "locks.phase_field_mm": 0.0}


def settings_of(st):
    """the dotted leaves the key covers: every leaf of the hair, environment and sim layers, and HEAD_STYLE_KEYS (a leaf
    of ADDED_AT_DEFAULT at its default left out)"""
    out = {}
    for path, v in LY._walk(_public(st)):
        if path in ADDED_AT_DEFAULT and v == ADDED_AT_DEFAULT[path]:
            continue
        if LY.owner(path) in KEY_LAYERS or any(path == k or path.startswith(k + ".") for k in HEAD_STYLE_KEYS):
            out[path] = v
    return dict(sorted(out.items()))


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def fingerprint(a):
    """sha256 of an array's dtype, shape and bytes"""
    a = np.ascontiguousarray(a)
    return _sha(("%s%s" % (a.dtype.str, a.shape)).encode() + a.tobytes())


def code_hashes():
    """the code that grows the head: whole files (CODE_FILES) and the functions of generate / actions / head the stage
    runs (CODE_FUNCS: an edit elsewhere in those files, the style branch or an act, leaves the head as grown)"""
    out = {}
    for m in CODE_FILES:
        with open(os.path.join(_HERE, m + ".py"), "rb") as fh:
            out[m] = _sha(fh.read())
    mods = {"generate": GEN, "actions": AC, "head": sys.modules[__name__]}
    for m, names in CODE_FUNCS.items():
        out[m] = _sha("".join(inspect.getsource(getattr(mods[m], n)) for n in names).encode())
    # with the GPU core on (HAIR_GPU=1, gpu.CoreGPU) the head's whole settle runs in gpu.py: its code is the head's
    # code, and a GPU-grown head (the same physics, another rounding path) is keyed apart from a CPU-grown one
    # (review wf_dccedefb-dc2, 10-31: a head grown by a kernel that was later fixed would otherwise stay cached)
    if os.environ.get("HAIR_GPU", "0") in ("1", "true", "yes"):
        from . import gpu as _gpu
        if _gpu.enabled():
            with open(os.path.join(_HERE, "gpu.py"), "rb") as fh:
                out["gpu"] = _sha(fh.read())
    return out


# fingerprints of the triangle colliders' faces (a field collider's were empty arrays): retired with the meshes (P14,
# fs-hairclean 11-14); a head grown before carries them, and rebuild() leaves them out of the comparison
RETIRED_GEOMETRY = ("collider_F", "scalp_F")


def case_geometry(scalp, collider):
    """the case's geometry as fingerprints: the scalp's samples and hairline, the collider's surface points"""
    out = {"scalp_V": fingerprint(np.asarray(scalp.V, np.float64)),
           "hairline": fingerprint(np.concatenate([np.ravel(scalp.grid), np.ravel(scalp.thb)]).astype(np.float64)),
           "collider_V": fingerprint(np.asarray(collider.V, np.float64)),
           "scalp_centre": fingerprint(np.asarray(scalp.c, np.float64))}
    # the root normals and the body's sign come from the normals (review 10-31: a changed VN gave a head new d0 / N
    # that no fingerprint saw)
    for nm, obj in (("scalp", scalp), ("collider", collider)):
        k = "%s_VN" % nm
        if getattr(obj, "VN", None) is not None:
            out[k] = fingerprint(np.asarray(obj.VN, np.float64))
    ears = getattr(scalp, "ears", None)
    if ears:
        out["ears"] = _sha(json.dumps({str(e): {k: float(v) for k, v in d.items() if np.isscalar(v)}
                                       for e, d in ears.items()}, sort_keys=True).encode())
    return out


def head_key(style, scalp, collider, menton_z, case=None):
    """-> (key, parts): parts = settings (sha over settings_of), case (ids, geometry, menton_z), code, format"""
    parts = {"settings": _sha(json.dumps(settings_of(head_settings(style)), sort_keys=True,
                                         separators=(",", ":")).encode()),
             "case": {"ids": dict(case or {}), "geometry": case_geometry(scalp, collider),
                      "menton_z": repr(float(menton_z))},
             "code": code_hashes(), "format": FORMAT}
    return _sha(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()), parts


def _fp_common(RP, pre):
    # d0 (the follicle directions: the bending energy's root clamp) and NR (the root normals) too - a follicle edit
    # leaves X, gi, mg, mL, ms as they were (review 10-31)
    return {"X": fingerprint(RP["X"]), "gi": fingerprint(pre["gi"]), "mg": fingerprint(pre["mg"]),
            "mL": fingerprint(pre["mL"]), "ms": fingerprint(pre["ms"]), "d0": fingerprint(pre["d0"]),
            "NR": fingerprint(RP["NR"])}


def _fp_rods(rods):
    return {"l": fingerprint(rods.l), "s": fingerprint(rods.s), "B": fingerprint(rods.B), "W": fingerprint(rods.W)}


# ---------------------------------------------------------------- snapshots
def _snapshot(rods, state):
    """a copy of the history of a settle (see the module notes)"""
    s = dict(D=rods.D.copy(), rho=rods.rho.copy(), mc=rods.mc.copy(), act_full=state.act_full.copy(),
             Ka=np.int64(state.Ka), anc=state.anc.copy(), anc_N=state.anc_N.copy(), anc_M=state.anc_M.copy(),
             anc_n=state.anc_n.copy(), stage_index=np.int64(state.stage_index), evals=np.int64(state.evals))
    if getattr(state, "sal", None) is not None:
        s["sal"] = np.array(state.sal, copy=True)
    w = getattr(rods, "wave", None)
    if w is not None:
        s["wave_n0"] = rods.wave_n0.copy()
        for k in ("wave_x", "wave_rho_w"):
            if getattr(rods, k, None) is not None:
                s[k] = getattr(rods, k).copy()
        for k in ("A", "lam", "s0", "phase"):
            s["wave." + k] = np.array(w[k], np.float64, copy=True)
        if w.get("rnd") is not None:
            s["wave.rnd.a"] = np.float64(w["rnd"]["a"])
            s["wave.rnd.k"], s["wave.rnd.p"] = w["rnd"]["k"].copy(), w["rnd"]["p"].copy()
        if w.get("hx") is not None:
            for k in ("kappa", "tau"):
                s["wave.hx." + k] = np.array(w["hx"][k], np.float64, copy=True)
            for k in ("flat", "s0", "ramp"):
                s["wave.hx." + k] = np.float64(w["hx"][k])
    return s


def _wave_of(s):
    """the rods' wave dict from a snapshot (fresh copies; the scalars as python floats, as rest_waves has them)"""
    w = {k: s["wave." + k].copy() for k in ("A", "lam", "s0", "phase")}
    w["rnd"] = None if "wave.rnd.a" not in s else dict(a=float(s["wave.rnd.a"]), k=s["wave.rnd.k"].copy(),
                                                       p=s["wave.rnd.p"].copy())
    if "wave.hx.kappa" in s:
        w["hx"] = dict(kappa=s["wave.hx.kappa"].copy(), tau=s["wave.hx.tau"].copy(),
                       **{k: float(s["wave.hx." + k]) for k in ("flat", "s0", "ramp")})
    return w


def _put_turning(rods, s):
    """the rest turning and the wave's edits as saved (copies)"""
    rods.rho = s["rho"].copy()
    if "wave.A" in s:
        rods.wave_x = s["wave_x"].copy() if "wave_x" in s else None
        rods.wave_rho_w = s["wave_rho_w"].copy() if "wave_rho_w" in s else None


# ---------------------------------------------------------------- the cut's anchors
def _sample_table(rods):
    """every sample by (guide g, segment j, k-th in its segment), its segment's length and where it lies now"""
    K = rods.K
    q = rods.q_all
    ns = np.bincount(q, minlength=rods.G * K)
    k = np.arange(len(q)) - (np.cumsum(ns) - ns)[q]
    g, j = q // K, q % K
    P = rods.positions()
    Y = P[g, j] + rods.t_all[:, None] * (P[g, j + 1] - P[g, j])
    return dict(K=K, g=g, j=j, k=k, l=rods.l[g, j].copy(), Y=Y)


def _carry_anchors(old, anc, new_state):
    """the friction anchors across a cut that changed the segments (rods.members(cut): K, the flat index g K + j and
    the last segments' lengths change). A sample keeps its anchor by (guide, segment, sample in segment): exactly
    where its segment kept its length (it has not moved); where the cut changed the segment's length (its samples
    moved along it), the anchor keeps its offset from the sample, so the static friction's memory - how far the hair
    has slipped from where it stuck - survives; samples past the cut are gone with the hair; a segment that grew (the
    cut's short tail joined to it) gets new samples with no anchor (anchored at the next lag). old: _sample_table of
    the rods before the cut; anc: the old state's (anc, anc_N, anc_M, anc_n). -> counts"""
    r = new_state.rods
    K1 = r.K
    ns1 = np.bincount(r.q_all, minlength=r.G * K1)
    start1 = np.cumsum(ns1) - ns1
    jj = np.minimum(old["j"], K1 - 1)
    q1 = old["g"] * K1 + jj
    ok = (old["j"] < K1) & (old["k"] < ns1[q1])
    src = np.nonzero(ok)[0]
    dst = start1[q1[src]] + old["k"][src]
    a, aN, aM, an = anc
    same = old["l"][src] == r.l[old["g"][src], old["j"][src]]
    Y1 = _sample_table(r)["Y"][dst]
    held = aN[src] > 0
    moved = ~same & held
    new_anc = a[src].copy()
    new_anc[moved] = Y1[moved] + (a[src][moved] - old["Y"][src][moved])
    new_state.anc[dst] = new_anc
    new_state.anc_N[dst] = aN[src]
    new_state.anc_M[dst] = aM[src]
    new_state.anc_n[dst] = an[src]
    return {"anchored_before": int((aN > 0).sum()), "kept_exact": int((same & held).sum()),
            "kept_offset": int(moved.sum()), "cut_off": int(((~ok) & (aN > 0)).sum()),
            "new_samples_unanchored": int(len(r.q_all) - len(dst))}


# ---------------------------------------------------------------- the Hair stage
def _rest(state, f):
    """the head comes to rest with no act: the release solves (friction lagged by actions.lag_stick, the static
    friction that keeps an anchor where the hair stuck), then settle.release's final gated settle (its own release
    solves off: they re-anchor every sample, as generate's grow branch has it)"""
    state.stage_index += 1
    nrel = max(1, int(f["release_solves"]))
    if f["evals_release"] > 0:
        per = max(2, int(round(f["evals_release"] / nrel)))
        for k in range(nrel):
            state.solve(per, label="release %d" % k)
            AC.lag_stick(state)
    keep = state.f["evals_release"]
    state.f["evals_release"] = 0
    try:
        state = SE.release(state, acts=None)
    finally:
        state.f["evals_release"] = keep
    return state


def grow_head(scalp, collider, style, menton_z, case=None, cache_dir=None, log=print, _live=None):
    """the Hair stage: the style's hair (oil.product 0) grown with no act, cut on its natural fall, rested.
    style: a resolved style (style.load / layers.compose, as generate takes it); case: ids for the key (e.g. the case
    name); cache_dir: the body grid's ($HAIR_GRID_CACHE when None). _live (tests): a dict that receives the live rods
    and state at the end. -> GrownHead"""
    t0 = time.time()
    log = log or (lambda m: None)
    tick = lambda m: log("%s  (%.1f s)" % (m, time.time() - t0))  # noqa: E731
    slog = lambda m: tick("  " + m)  # noqa: E731
    sth = head_settings(style)
    st = SY.effective(sth)                       # what generate hands its grow branch
    seed = int(st["seed"])
    key, parts = head_key(style, scalp, collider, menton_z, case)
    rep, T = {}, {}
    RP = GEN._roots_part_kinds(scalp, st, np.random.default_rng(seed), rep, tick)
    pre = GEN._prelude(scalp, collider, st, RP, rep, tick, cache_dir=cache_dir)
    rods, body, f, mL = pre["rods"], pre["body"], pre["f"], pre["mL"]
    T["prelude"] = time.time() - t0

    # grown: the natural fall, every act off (generate's rods_f for cut.taken 'fall')
    t = time.time()
    state = SE.grow(rods, body, acts=None, cfg=st, pressure=True, friction=True, log=slog)
    T["grow"] = time.time() - t
    n_hist = len(state.hist)
    snaps = {"grown": _snapshot(rods, state)}
    fps = {"common": _fp_common(RP, pre), "grown": _fp_rods(rods)}
    tick("head grown: %d guides, K %d, %d evaluations" % (rods.G, rods.K, state.evals))

    # her haircut, read on that fall as generate's cut.taken 'fall' reads it; the anchors carried over the cut
    t = time.time()
    ztip, layer = GEN._cut_plan(st, menton_z, RP["phi_r"], RP["th_r"], RP["din"], RP["main"], seed)
    Lcut, reached = GEN._cut(rods, ztip, layer, mL)
    Lcut = np.where(RP["kind"] == 2, mL, Lcut)
    old = _sample_table(rods)
    anc = (state.anc, state.anc_N, state.anc_M, state.anc_n)
    tip_z = np.round(1000 * np.percentile(rods.positions()[:, -1, 2] - menton_z, [10, 50, 90]), 1).tolist()
    rods.members(cut=Lcut)
    st2 = SE.State(rods, body, st, True, True, None, state.splat, slog)
    carried = _carry_anchors(old, anc, st2)
    st2.act_full = rods.l > 0
    st2.activate(float(rods.L.max()) + 1e-9)        # nothing switches on: every surviving segment stays as it lies
    st2.evals, st2.eval_s, st2.hist, st2.stage_index, st2.t0 = state.evals, state.eval_s, state.hist, \
        state.stage_index, state.t0
    state = st2
    T["cut"] = time.time() - t
    cut = {"reached_plane": int(reached.sum()), "members": int(len(Lcut)),
           "length_mm_p10_50_90": np.percentile(Lcut * 1000, [10, 50, 90]).round(1).tolist(),
           "grown_mm_p10_50_90": np.percentile(mL * 1000, [10, 50, 90]).round(1).tolist(),
           "held_at_grown_length": int((Lcut >= mL - 1e-9).sum()), "taken": "fall",
           "guide_tip_z_mm_p10_50_90_grown": tip_z, "anchors": carried}
    tick("head cut on its natural fall: %d of %d hairs reached their plane; lengths p10/50/90 %s mm; anchors %s" % (
        reached.sum(), len(Lcut), cut["length_mm_p10_50_90"], carried))

    # rested: release with static friction, the gated final, no act
    t = time.time()
    state = _rest(state, f)
    T["rest"] = time.time() - t
    snaps["rested"] = _snapshot(rods, state)
    fps["rested"] = _fp_rods(rods)
    srep = SE.report(state, {"centre": scalp.c, "chin_z": menton_z})
    T["all"] = time.time() - t0
    rested = {k: srep[k] for k in ("guides", "members", "energies_per_guide_weight_mm", "gate",
                                   "clamp_first_segment_vs_d0_deg_p50_p90_max", "fibre_fraction", "layers", "evals",
                                   "seconds_per_eval") if k in srep}
    tick("head rested: gate %s (max %.3f, p99 %.4f), %d evaluations" % (
        state.gate["passed"], state.gate["max_force_ratio"], state.gate["p99_force_ratio"], state.evals))
    meta = {"format": FORMAT, "key": key, "key_parts": parts, "name": style.get("name"),
            "settings": _public(sth), "product_in_style": (style.get("oil") or {}).get("product"),
            "menton_z": float(menton_z), "case": dict(case or {}), "head_style_keys": list(HEAD_STYLE_KEYS),
            "fingerprints": fps,
            "report": {"grown": {"evals": int(snaps["grown"]["evals"]), "K": int(snaps["grown"]["D"].shape[1])},
                       "cut": cut, "rested": rested, "warnings": list(pre["swarn"]) + list(pre["gwarn"]),
                       "prelude": {k: v for k, v in rep.items() if k in ("roots", "follicle")}},
            "state": {sn: {"hist": json.loads(json.dumps(h, default=_jsonable)), "gate": g}
                      for sn, h, g in (("grown", state.hist[:n_hist], None), ("rested", state.hist, state.gate))},
            "seconds": {k: round(v, 1) for k, v in T.items()}, "peak_ram_mb": peak_mb(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if _live is not None:
        _live.update(rods=rods, state=state, pre=pre, RP=RP)
    return GrownHead(meta, snaps)


# ---------------------------------------------------------------- save / load
def _arrays_sha(snaps):
    h = hashlib.sha256()
    for sn in sorted(snaps):
        for k in sorted(snaps[sn]):
            h.update(("%s__%s" % (sn, k)).encode())
            h.update(fingerprint(snaps[sn][k]).encode())
    return h.hexdigest()


def save(head, path):
    """path(.npz) + the same name .json: the snapshots' arrays (uncompressed: float64 history compresses little) and
    the meta with a sha over the arrays; both written to a temporary name and moved into place. -> the npz path"""
    base = path[:-4] if path.endswith(".npz") else path
    d = os.path.dirname(os.path.abspath(base))
    os.makedirs(d, exist_ok=True)
    arrays = {"%s__%s" % (sn, k): v for sn, s in head.snaps.items() for k, v in s.items()}
    meta = dict(head.meta, arrays_sha=_arrays_sha(head.snaps))
    tmp = base + ".tmp%d" % os.getpid()
    np.savez(tmp + ".npz", **arrays)
    with open(tmp + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=_jsonable)
    os.replace(tmp + ".npz", base + ".npz")
    os.replace(tmp + ".json", base + ".json")
    return base + ".npz"


def load(path):
    """-> GrownHead from save()'s pair; refuses another FORMAT or arrays that are not the ones the meta was saved with"""
    base = path[:-4] if path.endswith(".npz") else path
    with open(base + ".json", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("format") != FORMAT:
        raise HeadMismatch("%s: head format %r, this code reads %d" % (base, meta.get("format"), FORMAT))
    snaps = {}
    with np.load(base + ".npz") as z:
        for name in z.files:
            sn, k = name.split("__", 1)
            snaps.setdefault(sn, {})[k] = z[name]
    if _arrays_sha(snaps) != meta.get("arrays_sha"):
        raise HeadMismatch("%s: the arrays are not the ones its meta was saved with" % base)
    meta.pop("arrays_sha", None)
    return GrownHead(meta, snaps)


# ---------------------------------------------------------------- instantiate
def rebuild(head, scalp, collider, st=None, cache_dir=None, log=None):
    """the deterministic part of the head (generate._roots_part_kinds, _prelude) rebuilt for the case and checked:
    the case's geometry, the fingerprints, then the settings (st: the run's resolved style, default the head's own;
    its oil.product is set to 0 as the head's was). -> ctx {st, RP, pre} for instantiate (reusable: instantiate
    builds fresh rods from it each time). Raises HeadMismatch."""
    geo = case_geometry(scalp, collider)
    hg = {k: v for k, v in head.meta["key_parts"]["case"]["geometry"].items() if k not in RETIRED_GEOMETRY}
    bad = [k for k in sorted(set(hg) | set(geo)) if geo.get(k) != hg.get(k)]   # a key missing on either side counts
    if bad:
        raise HeadMismatch("the case differs from the one the head was grown on: %s" % ", ".join(bad))
    sth = head_settings(head.meta["settings"] if st is None else st)
    stE = SY.effective(sth)
    rep = {}
    tick = log or (lambda m: None)
    RP = GEN._roots_part_kinds(scalp, stE, np.random.default_rng(int(stE["seed"])), rep, tick)
    pre = GEN._prelude(scalp, collider, stE, RP, rep, tick, cache_dir=cache_dir)
    fp = _fp_common(RP, pre)
    hf = head.meta["fingerprints"]["common"]
    bad = [k for k in sorted(set(hf) | set(fp)) if fp.get(k) != hf.get(k)]
    if bad:
        raise HeadMismatch("the rebuilt %s differ from the head's (%s guides rebuilt): the head was grown from other "
                           "roots, guides or members - grow a new head" % (", ".join(bad), len(pre["gi"])))
    if st is not None:
        sk = _sha(json.dumps(settings_of(sth), sort_keys=True, separators=(",", ":")).encode())
        if sk != head.meta["key_parts"]["settings"]:
            a, b = settings_of(head_settings(head.meta["settings"])), settings_of(sth)
            diff = sorted(k for k in set(a) | set(b) if json.dumps(a.get(k)) != json.dumps(b.get(k)))
            raise HeadMismatch("the head was grown for other hair / environment / sim settings: %s" % ", ".join(diff))
    # the code that grew the head: a change is reported, not refused - the saved state is history, and the follicles,
    # roots and members it depends on are checked above; the physics now in the code settles it from here
    now = code_hashes()
    was = head.meta["key_parts"]["code"]
    drift = sorted(k for k in set(was) | set(now) if now.get(k) != was.get(k))   # 'gpu': grown on one core, now the other
    if drift and log:
        log("head: grown by other code in %s (the saved state is kept; regrow to grow it with this code)"
            % ", ".join(drift))
    return {"st": stE, "RP": RP, "pre": pre, "key": head.key, "code_drift": drift, "prelude_report": rep}


def instantiate(head, scalp, collider, snapshot="rested", st=None, ctx=None, cache_dir=None, log=None):
    """the live rods and settle of a snapshot of the head: fresh rods from the rebuilt prelude (ctx from rebuild(), or
    rebuilt here), cut as the snapshot was, checked against the snapshot's fingerprints (l, s, B, W), then the saved
    history written into them and into a fresh State (no acts; its splat built as any State builds it). The ONLY entry
    to a head, in memory or loaded; the head is never written to. -> (rods, state)"""
    if snapshot not in head.snaps:
        raise KeyError("snapshot %r: the head has %s" % (snapshot, sorted(head.snaps)))
    if ctx is not None and st is not None:
        raise ValueError("pass st to rebuild(), not with a ctx (the ctx already holds the settings it was checked with)")
    if ctx is not None and ctx.get("key") != head.key:
        raise HeadMismatch("the ctx was rebuilt for another head")
    ctx = ctx or rebuild(head, scalp, collider, st=st, cache_dir=cache_dir, log=log)
    s = head.snaps[snapshot]
    rods = ctx["pre"]["fresh_rods"]()
    if not np.array_equal(s["mc"], rods.mc):
        rods.members(cut=s["mc"].copy())
    fp = _fp_rods(rods)
    bad = [k for k, v in head.meta["fingerprints"][snapshot].items() if fp.get(k) != v]
    if bad:
        raise HeadMismatch("the rebuilt rods' %s differ from the '%s' snapshot's" % (", ".join(bad), snapshot))
    if s["D"].shape != rods.D.shape:
        raise HeadMismatch("the '%s' snapshot's D is %s, the rebuilt rods' %s" % (snapshot, s["D"].shape,
                                                                                  rods.D.shape))
    rods.D = s["D"].copy()
    if "wave.A" in s:
        rods.wave = _wave_of(s)
        rods.wave_n0 = s["wave_n0"].copy()
    _put_turning(rods, s)
    state = SE.State(rods, ctx["pre"]["body"], ctx["st"], True, True, None, None, log or (lambda m: None))
    # a helix fibre's State reads the rest turning in the frame of the saved D on construction (wave_frame); its loc
    # is the natural turning + wave_x either way. Put back rho / wave_x / wave_rho_w exactly as saved.
    _put_turning(rods, s)
    state.act_full = s["act_full"].copy()
    for k in ("anc", "anc_N", "anc_M", "anc_n"):
        setattr(state, k, s[k].copy())
    state._setup(int(s["Ka"]))
    if "sal" in s:                                   # the alignment as its stage set it (not re-read from D)
        state.sal = s["sal"].copy()
    state.stage_index, state.evals = int(s["stage_index"]), int(s["evals"])
    sm = (head.meta.get("state") or {}).get(snapshot) or {}
    state.hist = copy.deepcopy(sm.get("hist") or [])
    state.gate = copy.deepcopy(sm.get("gate"))
    return rods, state
