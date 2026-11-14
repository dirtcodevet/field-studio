"""The fibre layer on the GPU (hair lane, 11-10): one CuPy module draws a HairSet (engine/hair/hairset.py) of any kinds, cut
against a still's depth and composited over it (fibre_layer.cuh says what is drawn and how).

    FL = FibreLayer(cp, hairset)                 # compile + upload + span grid
    FL.light(dir_to_sun, sun_irradiance_rgb, sky_radiance_rgb)
    rgb, T, Tc, depth, ids, ms = FL.layer(cam, zbuf, bg)     # cam: {"pos","target","up","hfov","w","h"}
    img2 = FL.composite(img, cam, zbuf)          # rgb + Tc x img
    Ts = FL.skin_shade(cam, zbuf)                # the key's transmittance through the hair at each still pixel

The fibre code it builds on (Chiang 2016 lobes, Table A far field, the oval cone-footprint hit) is
fibre_scatter.cuh beside this file, proven on the 11-05 brows and lashes (T1 GPU vs CPU twin)."""
from __future__ import annotations

import glob
import math
import os
import site
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FV = os.environ.get("FIELDSTUDIO_FV8_TOOLS", "")                       # the store reader's folder (fv8_field.py) for body_lattice; unset: none
FIBRE_CUH = os.path.join(HERE, "fibre_scatter.cuh")
LAYER_CUH = os.path.join(HERE, "fibre_layer.cuh")
TABLE_A = os.environ.get("FIELDSTUDIO_HAIR_TABLE_A", "")    # Chiang far-field table (far32; beta_n 0.55, eta 1.55); not shipped: without it FibreLayer stops and the Hair panel uses its lattice light
R_EFF = 3.224027e-05                                                  # fv8_hair/tl1_light.npz r_eff (only the bake path reads it)
GRID_H = 0.5e-3
BAND_S = float(os.environ.get("FL_BAND_S", "0.15"))                    # target seconds per pixel-kernel launch (FibreLayer._bands); fs-store-2 11-12: 0.6 -> 0.15
BAND_ROWS_MIN = 8                                                      # fs-store-2 11-12: was 32 (a 32-row band of fl_skin_occ3 at 2000 px ran 1.13 s)
NPT = 24
FL_ABORT_S = float(os.environ.get("FL_ABORT_MS", "1000")) / 1000.0     # fs-shell 11-14: a band past this STOPS the call
_PLANS = {}                                                            # fs-store-5 (lane copy): band plans kept across FibreLayer instances
# keratin's optics (kinds.py _keratin: eta 1.55 M; the far table's beta_n 0.55; alpha the cuticle tilt 3 deg M (hair.md T7))
HT_DEFAULT = {"beta_m": 0.10, "beta_n": 0.55, "eta": 1.55, "alpha_deg": 3.0, "ell": 1.2, "tau_per_m": 40.0,
              "s_len_m": 50e-6 * (1 - 5.0 / 6.0), "step_m": 0.5e-6, "random_roughness": 0.0}


def cuda_include():
    c = glob.glob(os.path.join(site.getusersitepackages(), "nvidia", "cu*", "include", "cuda_fp16.h"))
    c += glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cu*", "include", "cuda_fp16.h"))
    if os.environ.get("CUDA_PATH"):
        c += glob.glob(os.path.join(os.environ["CUDA_PATH"], "include", "cuda_fp16.h"))
    return os.path.dirname(c[0]) if c else None


def source(npt=24):
    return open(FIBRE_CUH, encoding="utf-8").read() + "\n#define FNPT %d\n" % int(npt) + open(LAYER_CUH, encoding="utf-8").read()


def span_grid(P, RAD, h=GRID_H, pad=1e-4, chunk=4000):
    """each span k of strand i (the bounding box of its Catmull-Rom points k..k+1, widened by a quarter of its chord for the
    curve's bulge, by its largest radius and by pad) -> the cells it touches; vectorised in chunks of strands.
    -> org (3,), dim (3,), off (cells+1,) uint32, idx int32 (strand * (NPT-1) + k)"""
    n, npt = P.shape[:2]
    lo = P.reshape(-1, 3).min(0).astype(np.float64) - 1e-3; hi = P.reshape(-1, 3).max(0).astype(np.float64) + 1e-3
    dim = np.maximum(np.ceil((hi - lo) / h).astype(np.int64), 1)
    cells, spans = [], []
    for c0 in range(0, n, chunk):
        Pc = P[c0:c0 + chunk].astype(np.float32); Rc = RAD[c0:c0 + chunk].astype(np.float32); m_ = len(Pc)
        a = Pc[:, :-1]; b = Pc[:, 1:]; ch = np.linalg.norm(b - a, axis=2, keepdims=True)
        r = np.maximum(Rc[:, :-1], Rc[:, 1:])[..., None] + 0.25 * ch + pad
        b0 = np.clip(np.floor((np.minimum(a, b) - r - lo) / h).astype(np.int64), 0, dim - 1).reshape(-1, 3)
        b1 = np.clip(np.floor((np.maximum(a, b) + r - lo) / h).astype(np.int64), 0, dim - 1).reshape(-1, 3)
        sid = (np.arange(c0, c0 + m_, dtype=np.int64)[:, None] * (npt - 1) + np.arange(npt - 1)[None]).reshape(-1)
        ext = b1 - b0 + 1; E = ext.max(0)
        for ox in range(E[0]):
            mx = ext[:, 0] > ox
            for oy in range(E[1]):
                my = mx & (ext[:, 1] > oy)
                for oz in range(E[2]):
                    m = my & (ext[:, 2] > oz)
                    if not m.any():
                        continue
                    cells.append(((b0[m, 0] + ox) * dim[1] + (b0[m, 1] + oy)) * dim[2] + (b0[m, 2] + oz)); spans.append(sid[m])
    cells = np.concatenate(cells); spans = np.concatenate(spans).astype(np.int32)
    o = np.argsort(cells, kind="stable"); cells = cells[o]; spans = spans[o]
    nc = int(np.prod(dim))
    cnt = np.bincount(cells, minlength=nc)
    off = np.zeros(nc + 1, np.int64); off[1:] = np.cumsum(cnt)
    if off[-1] >= 2 ** 32:
        raise ValueError("span grid too large")
    return lo.astype(np.float32), dim.astype(np.int32), off.astype(np.uint32), spans


def cam_frame(cam):
    pos = np.asarray(cam["pos"], float); fw = np.asarray(cam["target"], float) - pos; fw /= np.linalg.norm(fw)
    rt = np.cross(fw, cam["up"]); rt /= np.linalg.norm(rt); upv = np.cross(rt, fw)
    pixang = 2 * math.tan(math.radians(cam["hfov"]) / 2) / int(cam["w"])
    return pos, fw, rt, upv, pixang


def _f3(v):
    return np.asarray(v, np.float32).reshape(3).view(np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)]))[0]


def repair_far(A, h, band=0.009, shell=0.004):
    """the lattice as a signed distance EVERYWHERE. The field store is a true distance only in a band near her surface: its
    bricks clamp at +-12 mm (a plateau: 13 mm read where the true distance is 30) and beyond the bricks the fill adds the
    distance to the nearest brick node (kinks, flat gradients). Near contact never noticed; anything that reads distances or
    directions further out does (the groom's layers up to 48 mm off her scalp; the renderer's outward direction: blocky dark
    patches on the crown, 11-11). Repair: inside the band (|A| < band) the field's own value; beyond it, the Euclidean
    distance to the nearest node of a reliable shell (band - shell < |A| < band) plus that node's own value, by side;
    blended over the last millimetre of the band and lightly smoothed outside it."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    A = np.asarray(A, np.float32); out = A.copy()
    for sgn in (1.0, -1.0):
        v = A * sgn
        sh = (v > band - shell) & (v < band)                     # reliable shell on this side
        if not sh.any():
            continue
        dist, ind = distance_transform_edt(~sh, return_indices=True)
        far = (dist * h + v[tuple(ind)]).astype(np.float32)      # distance through the shell node
        w = np.clip((v - (band - 0.001)) / 0.001, 0.0, 1.0)      # 0 inside the band, 1 beyond it
        side = v > 0
        out = np.where(side, sgn * ((1 - w) * v + w * np.maximum(far, v)), out)
    sm = gaussian_filter(out, 0.7)
    wf = np.clip((np.abs(out) - band) / 0.003, 0.0, 1.0)         # smooth only outside the band (the facets of the shell's nodes)
    return ((1 - wf) * out + wf * sm).astype(np.float32)


def body_lattice(field_dir, cache_dir, lo=(-0.28, -0.32, -0.36), hi=(0.28, 0.34, 0.40), mm=3.0, log=print, far=True):
    """her body's signed distance on a lattice from the field the renderer draws (fv8_field.FieldStore), cached per field
    (the store's own file stamp) -> (A, lo, h). No brick: the side by flood fill from the box (engine/hair/field_collider)"""
    import hashlib, json as _j
    sys.path.insert(0, FV)
    import fv8_field as FF
    from .field_collider import FieldSDFGrid
    key = hashlib.sha1(_j.dumps([os.path.abspath(field_dir), list(lo), list(hi), mm,
                                 sorted((f, os.path.getmtime(os.path.join(field_dir, f))) for f in os.listdir(field_dir))[:50]]).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, "body_%s.npz" % key)
    pfar = os.path.join(cache_dir, "body_%s_far.npz" % key)      # the same lattice with its far field repaired (repair_far)
    if far and os.path.exists(pfar):
        z = np.load(pfar); return z["A"].astype(np.float32), z["lo"], float(z["h"])
    if os.path.exists(path):
        z = np.load(path)
        if not far:
            return z["A"], z["lo"], float(z["h"])
        Af = repair_far(z["A"].astype(np.float32), float(z["h"])); np.savez_compressed(pfar, A=Af, lo=z["lo"], h=z["h"])
        return Af, z["lo"], float(z["h"])
    S = FF.FieldStore(field_dir)
    g = FieldSDFGrid(lambda P: S.evaluate_F(np.asarray(P, np.float64), level=None), lo, hi, mm=mm, log=log)
    os.makedirs(cache_dir, exist_ok=True); np.savez_compressed(path, A=g.A.astype(np.float16), lo=g.lo, h=g.h)
    if far:
        Af = repair_far(g.A.astype(np.float32), float(g.h)); np.savez_compressed(pfar, A=Af, lo=g.lo, h=g.h); return Af, g.lo, g.h
    return g.A, g.lo, g.h


class FLGuardStop(RuntimeError):
    """fs-shell 11-14: a FibreLayer call stopped by its band guard (a launch past FL_ABORT_MS); nothing more was launched"""


class FibreLayer:
    def __init__(self, cp, hs, ht=None, amb=1.0, tag="", grid_h=None, shadow=None, body=None, env=None, key_r=0.0, nloc=1, skyrays=0):
        npt = hs.a["P"].shape[1]; self.cp = cp; self.hs = hs; self.keep = []; self.n = hs.n; self.tag = tag
        L90 = float(np.percentile(hs.a["LEN"], 90)) if hs.n else 0.0
        self.grid_h = float(grid_h or (GRID_H if L90 / max(npt - 1, 1) < 1.0e-3 else 2.0e-3))
        self.shadow = int(shadow if shadow is not None else (1 if hs.n > 20000 else 0))
        inc = cuda_include(); t = time.time()
        # 11-10 (found): the frame's tag in the source made every frame miss the compile cache (~12 s per layer);
        # the source differs only by the set's point count, and the hook draws its layers one at a time
        self.mod = cp.RawModule(code=source(npt),
                                options=("--use_fast_math", "-std=c++17") + (("-I" + inc,) if inc else ()))
        self.mod.compile(); self.compile_s = round(time.time() - t, 1)
        self.k_layer = self.mod.get_function("fl_layer"); self.k_shade = self.mod.get_function("fl_skin_shade")
        self.k_occ = self.mod.get_function("fl_skin_occ2")
        a = hs.a
        up = lambda x: self.keep.append(cp.asarray(np.ascontiguousarray(x))) or self.keep[-1]  # noqa: E731
        ptr = lambda x: np.array([x.data.ptr], np.uint64)  # noqa: E731
        h = dict(HT_DEFAULT, **(ht or {}))
        HT = np.array([h["beta_m"], h["beta_n"], h["eta"], math.radians(h["alpha_deg"]), h["ell"], h["tau_per_m"], h["s_len_m"],
                       h["step_m"], R_EFF, 1.0, h["random_roughness"], 0.0], np.float32)
        TA = np.ascontiguousarray(np.load(TABLE_A)["far32"], np.float32)
        # fd: n, pad0 = sky over the hemisphere (4 cones; 0 = one ray along N), 0, 0 | amb, weathering, the key's radius, local room direction
        # (11-11 04:40: a comment placed INSIDE this line had cut off the float half, so amb / weathering / key size / local direction were
        #  all 0 for every frame rendered 04:20-04:37: g27s, g27f and all of g28 are invalid)
        fd = np.zeros(8, np.float32); fd[:4] = np.array([hs.n, int(skyrays), 0, 0], np.int32).view(np.float32)
        fd[4:] = [amb, float(h.get("weather_x", 0.0)), float(key_r), float(nloc)]
        lo_ = a["P"].reshape(-1, 3).min(0).astype(np.float64) - 1e-3; hi_ = a["P"].reshape(-1, 3).max(0).astype(np.float64) + 1e-3
        dim = np.maximum(np.ceil((hi_ - lo_) / self.grid_h).astype(np.int64), 1); org = lo_.astype(np.float32)
        self.setc(FP=ptr(up(a["P"].astype(np.float32).reshape(-1))), FRAD=ptr(up(a["RAD"].astype(np.float32))), FPIG=ptr(up(a["PIG"].astype(np.float32))),
                  FSIG=ptr(up(a["SIG"].astype(np.float32))), FRREF=ptr(up(a["RREF"].astype(np.float32))), FELL=ptr(up(a["ELL"].astype(np.float32))),
                  FTAU=ptr(up(a["TAU"].astype(np.float32))), FBM=ptr(up(a["BM"].astype(np.float32))), FNRM=ptr(up(a["NRM"].astype(np.float32))),
                  FSEED=ptr(up(a["SEED"].astype(np.uint32))), FLEN=ptr(up(a["LEN"].astype(np.float32))),
                  FGORG=org, FGDIM=dim.astype(np.int32), FGH=np.array([self.grid_h], np.float32),
                  HT=HT, TA=ptr(up(TA)), FD=fd, FSHADOW=np.array([self.shadow], np.int32),
                  FSIGP=(ptr(up(a["SIGP"].astype(np.float32))) if "SIGP" in a else np.array([0], np.uint64)),
                  FSKYMED=np.array([0], np.int32))
        if body is not None:                    # (A (nx, ny, nz) m, lo (3,), h): her body in the hair's light
            BA, blo, bh = body
            self.setc(FBODY=ptr(up(np.ascontiguousarray(BA, np.float32).reshape(-1))), FBORG=np.asarray(blo, np.float32),
                      FBDIM=np.asarray(BA.shape, np.int32), FBH=np.array([bh], np.float32))
        else:
            self.setc(FBODY=np.array([0], np.uint64))
        if env is not None:                     # (map (h, w, 3) radiance, cos rot, sin rot): the renderer's environment
            E_, cr_, sr_ = env; E_ = np.ascontiguousarray(E_, np.float32)
            self.setc(FENV=ptr(up(E_.reshape(-1))), FENVR=np.array([cr_, sr_, E_.shape[1], E_.shape[0]], np.float32))
        else:
            self.setc(FENV=np.array([0], np.uint64))
        t = time.time(); nspan = hs.n * (npt - 1); nc = int(np.prod(dim)); kg = self.mod.get_function("fl_grid")
        cnt = cp.zeros(nc, cp.uint32); blk = ((nspan + 255) // 256,)
        kg(blk, (256,), (np.int32(nspan), np.float32(1e-4), cnt, cnt, cp.zeros(1, cp.int32), np.int32(0)))
        off = cp.zeros(nc + 1, cp.uint64); off[1:] = cp.cumsum(cnt.astype(cp.uint64)); tot = int(off[-1])
        if tot >= 2 ** 32: raise ValueError("span grid too large: %d" % tot)
        off = off.astype(cp.uint32); idx = cp.zeros(max(tot, 1), cp.int32); cur = off[:-1].copy(); del cnt
        kg(blk, (256,), (np.int32(nspan), np.float32(1e-4), cur, cur, idx, np.int32(1))); cp.cuda.Device().synchronize(); del cur
        if os.environ.get('FL_GRID_SORT', '1') == '1' and tot > 1:
            # fs-store-6 11-13: the cells' span lists in a FIXED order (span id within each cell). fl_grid's pass 1 places spans with
            # atomics, so the order varied build to build and with it the fibres' product order in fl_Trc and ties in the 8-nearest
            # hit lists: 89 / 25 pixels of a 1080p frame differed by ~1 fp16 step between two frames of one scene (hs2).
            cell = cp.searchsorted(off, cp.arange(tot, dtype=cp.uint32), side='right').astype(cp.int64) - 1
            key = cp.sort((cell << 32) | idx.astype(cp.int64)); idx = (key & 0xFFFFFFFF).astype(cp.int32); del cell, key
        self.keep += [off, idx]; self.setc(FGOFF=ptr(off), FGIDX=ptr(idx)); self.grid_s = round(time.time() - t, 2)
        self.grid = {"dim": dim.tolist(), "h_mm": self.grid_h * 1e3, "spans_listed": tot, "cells": nc, "shadow": self.shadow, "spans": nspan}
        self.resident_mb = round(sum(int(x.nbytes) for x in self.keep) / 1e6, 2)

    def _bands(self, kern, args, W, H, info=None, tail=(), pk=None):
        """launch a pixel kernel over the frame in bands of rows (hair-shadow lane 11-11), each band timed. One launch over a
        whole frame ran 15 s for the scalp layer; the display watchdog resets the card past ~2 s.
        fs-store-2 11-12 (a 32-row band of fl_skin_occ3 at 2000 px ran 1.13 s): the target is BAND_S (0.15 s) and a band may
        be as thin as BAND_ROWS_MIN (8) rows. A kernel's FIRST call at a size walks down the frame from 8 rows, each next
        band sized from the last one's time (growth at most 1.5x a band); every call records each 8-row unit's measured time
        and the NEXT call of that kernel at that size plans its bands from it (units grouped while their sum stays under
        BAND_S), so repeated calls (the lean refinement's passes) are not stuck re-growing from 8 rows. Whole multiples of the
        8-row block: no row runs twice; the pixels and the image are what one launch gives.
        fs-store-4 11-13: pk (e.g. layer()'s round) is part of the plan's key: calls whose cost differs (round 0 vs round 3 of
        fl_layer_rnd, 9.0 s launches when they shared one plan) learn separate plans. A planned call probes the plan's heaviest 8-row unit
        and scales the plan by max(1, measured / planned) before grouping; a planned band past 3 x BAND_S returns the rest of
        the call to walking from 8 rows."""
        cp = self.cp; e0, e1 = cp.cuda.Event(), cp.cuda.Event(); worst = 0.0; n = 0
        nu = (H + 7) // 8; key = (id(kern), int(W), int(H), pk); profs = self.__dict__.setdefault("_band_prof", {})
        # fs-store-5 11-13 (lane copy): a frame SEQUENCE builds a new FibreLayer every frame, so each frame re-walked every band
        # from 8 rows (the scalp layer 25-27 s a frame vs 13.5 s planned). The plans are kept across instances (module level),
        # keyed by the set's tag, the kernel's name, the size, pk and the sky switch; the probe still guards a slower frame.
        kname = next((k_ for k_, v_ in self.__dict__.items() if k_.startswith("k_") and v_ is kern), "id%d" % id(kern))   # the attribute's name (k_rnd, k_pass ...): the same in every instance
        gkey = (getattr(self, "tag", ""), kname, int(W), int(H), pk, getattr(self, "skymed", 0))
        if os.environ.get("FL_PLAN_SHARE", "1") == "1" and key not in profs and gkey in _PLANS:
            profs[key] = _PLANS[gkey]
        prof = profs.get(key); cost = np.zeros(nu)

        def run(y0, r):
            e0.record()
            kern(((W + 15) // 16, (r + 7) // 8, 1), (16, 8, 1), tuple(args) + (np.int32(y0),) + tuple(tail))
            e1.record(); e1.synchronize(); t = cp.cuda.get_elapsed_time(e0, e1) * 1e-3
            u0 = y0 // 8; u1 = min(nu, (y0 + r + 7) // 8); cost[u0:u1] = t / max(u1 - u0, 1)
            if FL_ABORT_S > 0 and t > FL_ABORT_S:
                # fs-shell 11-14 (Tier 0 part 1, L-1142: the plan reused from the last call had no stop): a band past
                # FL_ABORT_MS (1 s, half the display watchdog) stops the call: the card is synchronised, nothing more is
                # launched, this kernel's plan is dropped (the next call walks from 8 rows), FLGuardStop to the caller
                cp.cuda.Device().synchronize()
                profs.pop(key, None); _PLANS.pop(gkey, None)
                raise FLGuardStop("FibreLayer: a band took %.0f ms > %.0f ms (rows %d-%d of %d, width %d): the call stops, "
                                  "nothing more is launched" % (t * 1e3, FL_ABORT_S * 1e3, y0, y0 + r, H, W))
            return t
        done = np.zeros(nu, bool)
        if prof is not None:                                     # planned from this kernel's last call at this size (and pk)
            prof = np.array(prof, dtype=np.float64); pu = int(np.argmax(prof))
            t = run(pu * 8, min(8, H - pu * 8)); worst = max(worst, t); n += 1; done[pu] = True
            if prof[pu] > 0.0:
                prof *= max(1.0, t / prof[pu])                   # the probe (the plan's heaviest unit): a plan slower than measured is scaled up
            u = 0
            while u < nu:
                if done[u]:
                    u += 1; continue
                v = u + 1; s = prof[u]
                while v < nu and not done[v] and s + prof[v] <= BAND_S: s += prof[v]; v += 1
                y0 = u * 8; r = min(v * 8, H) - y0; t = run(y0, r); worst = max(worst, t); n += 1; done[u:v] = True; u = v
                if t > 3.0 * BAND_S:                             # the plan was wrong here: walk the rest
                    break
        u = 0; rows = BAND_ROWS_MIN                              # walk every unit not yet drawn, from 8 rows, never across a drawn one
        while u < nu:
            if done[u]:
                u += 1; continue
            v = u
            while v < nu and v - u < max(1, rows // 8) and not done[v]: v += 1
            y0 = u * 8; r = min(v * 8, H) - y0; t = run(y0, r); worst = max(worst, t); n += 1; done[u:v] = True; u = v
            rows = max(BAND_ROWS_MIN, int(min(1.5 * rows, rows * BAND_S / max(t, 1e-4)) // 8 * 8))
        profs[key] = cost
        if os.environ.get("FL_PLAN_SHARE", "1") == "1": _PLANS[gkey] = cost
        self.last_bands = {"n": n, "worst_s": round(worst, 3)}
        if info is not None:
            info.update(self.last_bands)

    def setc(self, **kw):
        for kname, arr in kw.items():
            g = self.mod.get_global(kname); arr = np.ascontiguousarray(arr); g.copy_from_host(arr.ctypes.data, arr.nbytes)

    def light(self, sun_dir, sun_rgb, sky_rgb):
        s = np.asarray(sun_dir, np.float32); s = s / np.linalg.norm(s)
        self.setc(HSUN=s, HSUNRGB=np.asarray(sun_rgb, np.float32), HSKY=np.asarray(sky_rgb, np.float32))
        self._hsun = s
        # fs-store-5 11-13 (lane copy): FL_SKYMED = bits (1 the sky, 2 the skin side) through the hair's measured medium, built
        # here from the fibres for this frame (after HSUN: set 0 is along the key); unset / 0 = today's exact wide cones
        sm = int(os.environ.get("FL_SKYMED", "3") or 0) if self.shadow else 0   # 11-13: the medium is the default (accepted on the hair); FL_SKYMED=0 = the exact wide cones
        if sm:
            self.build_medium(near=float(os.environ.get("FL_SKYMED_NEAR", "0.5")), levels=int(os.environ.get("FL_SKYMED_LEVELS", "5")))
        self.setc(FSKYMED=np.array([sm], np.int32)); self.skymed = sm
        # fs-store-6 11-13 (lane copy): FL_KEYDOM=1 -> the key through the hair read from a map built here once a frame in the
        # key's own view (build_dom); unset / 0 = today's exact rays
        kd = int(os.environ.get('FL_KEYDOM', '0') or 0) if self.shadow else 0
        if kd:
            self.build_dom(texel=float(os.environ.get('FL_KEYDOM_TEXEL', '0.35e-3')), max_cols=int(float(os.environ.get('FL_KEYDOM_MAXCOLS', '1.5e6'))))
        self.setc(FDOM=np.array([kd], np.int32))
        self.setc(FREF=np.array([int(os.environ.get('FL_REF', '0') or 0)], np.int32))   # fs-store-6: the reference (tests only)

    def layer(self, cam, zbuf=None, bg=None):
        cp = self.cp; pos, fw, rt, upv, pixang = cam_frame(cam); W, H = int(cam["w"]), int(cam["h"]); n_ = W * H
        z = cp.full(n_, 1e3, cp.float32) if zbuf is None else cp.ascontiguousarray(zbuf.reshape(-1).astype(cp.float32))
        b = 0 if bg is None else cp.ascontiguousarray(bg.reshape(-1).astype(cp.float32))
        rgb = cp.zeros(n_ * 3, cp.float32); T = cp.zeros(n_, cp.float32); Tc = cp.zeros(n_ * 3, cp.float32)
        dep = cp.zeros(n_, cp.float32); ids = cp.zeros(n_, cp.int32)
        t0 = time.time()
        # (hair-shadow 11-11) one launch per round of 8 hits (fl_layer_rnd: the same arithmetic as fl_layer, so the same image), each
        # round in timed bands: a dense pixel's 6 rounds never sit in one launch (a 32-row band of the scalp took 1.5 s whole)
        if not hasattr(self, "k_rnd"):
            self.k_rnd = self.mod.get_function("fl_layer_rnd")
        st = cp.zeros(n_, cp.float32); worst = 0.0; nl = 0
        head = (_f3(pos), _f3(fw), _f3(rt), _f3(upv), np.float32(pixang), np.int32(W), np.int32(H), z, (np.uint64(0) if bg is None else b), rgb, T, Tc, dep, ids, st)
        for rnd in range(6):
            if rnd > 0 and not bool((st >= 0).any()):
                break
            # fs-store-4 11-13: each round its own band plan (pk = rnd): round 0 planned from round 3's times ran 9.0 s launches
            self._bands(self.k_rnd, head + (np.int32(rnd),), W, H, pk=rnd); worst = max(worst, self.last_bands["worst_s"]); nl += self.last_bands["n"]
        self.last_bands = {"n": nl, "worst_s": round(worst, 3), "rounds": rnd + 1}
        del st
        return (rgb.reshape(H, W, 3), T.reshape(H, W), Tc.reshape(H, W, 3), dep.reshape(H, W), ids.reshape(H, W),
                round((time.time() - t0) * 1e3, 1))

    def skin_shade(self, cam, zbuf, pen_min=None):
        cp = self.cp; pos, fw, rt, upv, pixang = cam_frame(cam); W, H = int(cam["w"]), int(cam["h"])
        pen = float(pen_min if pen_min is not None else 0.00465)        # the sun's angular radius (rad)
        Ts = cp.zeros(W * H, cp.float32)
        self._bands(self.k_shade, (_f3(pos), _f3(fw), _f3(rt), _f3(upv), np.float32(pixang), np.int32(W), np.int32(H),
                                   cp.ascontiguousarray(zbuf.reshape(-1).astype(cp.float32)), np.float32(pen), Ts), W, H)
        return Ts.reshape(H, W)

    def skin_occ(self, cam, zbuf, wsun, pen_sun=0.02465):
        """the hair's coloured occlusion of her skin at each still pixel (H, W, 3): key share wsun, sky the rest (hair2 11-10)"""
        cp = self.cp; pos, fw, rt, upv, pixang = cam_frame(cam); W, H = int(cam["w"]), int(cam["h"])
        Ts = cp.ones(W * H * 3, cp.float32)
        self._bands(self.k_occ, (_f3(pos), _f3(fw), _f3(rt), _f3(upv), np.float32(pixang), np.int32(W), np.int32(H),
                                 cp.ascontiguousarray(zbuf.reshape(-1).astype(cp.float32)), np.float32(wsun), np.float32(pen_sun), Ts), W, H)
        return Ts.reshape(H, W, 3)

    def skin_occ_exact(self, cam, zbuf, wsun, key_r, ns_key=8, sky_r=0.12, ns_sky=2, key_spen=1e-7, sky_spen=1e-7):
        """as skin_occ, each light integrated by stratified EXACT rays (fl_skin_occ3, hair-shadow lane 11-11): the key a cap of
        angular radius key_r (ns_key^2 rays), each of the six sky directions a cap of sky_r (ns_sky^2 rays), the point jittered
        over the pixel. For short hair on the 0.5 mm grid (brows, lashes); fibre_layer.cuh says why one cone was wrong."""
        cp = self.cp; pos, fw, rt, upv, pixang = cam_frame(cam); W, H = int(cam["w"]), int(cam["h"])
        if not hasattr(self, "k_occ3"):
            self.k_occ3 = self.mod.get_function("fl_skin_occ3")
        Ts = cp.ones(W * H * 3, cp.float32)
        self._bands(self.k_occ3, (_f3(pos), _f3(fw), _f3(rt), _f3(upv), np.float32(pixang), np.int32(W), np.int32(H),
                                  cp.ascontiguousarray(zbuf.reshape(-1).astype(cp.float32)), np.float32(wsun), np.float32(key_r), np.int32(ns_key),
                                  np.float32(sky_r), np.int32(ns_sky), Ts), W, H, tail=(np.float32(key_spen), np.float32(sky_spen)))
        return Ts.reshape(H, W, 3)

    def build_medium(self, near=0.5, levels=5):
        """the hair as a measured absorbing medium (fl_med_build / fl_med_down, hair-shadow lane 11-11): per span-grid cell the
        fibres' extinction along the light (call light() first) and orientation-averaged, with levels - 1 mip levels; near: the
        exact fibres serve the shadow ray until its footprint radius reaches near x the cell (fl_Tmed). -> MB on the card"""
        cp = self.cp; t = time.time(); dims = [tuple(int(x) for x in self.grid["dim"])]
        for _ in range(levels - 1):
            dims.append(tuple(max(1, (x + 1) // 2) for x in dims[-1]))
        n = [int(np.prod(d)) for d in dims]; offs = np.zeros(18, np.int64); o = 0
        offs[0] = 0; offs[6] = 3 * n[0]; offs[12] = 6 * n[0]; o = 9 * n[0]   # level 0: the key set, the isotropic set, the forward set (fl_med_build)
        for l in range(1, levels):
            offs[l] = o; o += 3 * n[l]; offs[6 + l] = o; o += 3 * n[l]; offs[12 + l] = o; o += 3 * n[l]
        if getattr(self, "_med", None) is not None and self._med.size == o:
            med = self._med; med.fill(0)                                  # a rebuild (light() again): the same buffer
        else:
            med = cp.zeros(o, cp.float32); self.keep.append(med); self._med = med
        kb = self.mod.get_function("fl_med_build"); nspan = self.hs.n * (self.hs.a["P"].shape[1] - 1)
        kb(((nspan + 255) // 256,), (256,), (np.int32(nspan), med))
        n0f = 9 * n[0]; self.mod.get_function("fl_med_fix")(((n0f + 255) // 256,), (256,), (np.int64(n0f), med))   # fs-store-6: fixed point -> float (level 0)
        kd = self.mod.get_function("fl_med_down")
        for l in range(1, levels):
            for sset in (0, 6, 12):
                src = med[int(offs[sset + l - 1]):]; dst = med[int(offs[sset + l]):]
                kd(((n[l] + 255) // 256,), (256,), (src, np.array(dims[l - 1], np.int32).view(np.dtype([("x", np.int32), ("y", np.int32), ("z", np.int32)]))[0],
                                                     dst, np.array(dims[l], np.int32).view(np.dtype([("x", np.int32), ("y", np.int32), ("z", np.int32)]))[0]))
        cp.cuda.Device().synchronize()
        dd = np.zeros((6, 3), np.int32); dd[:levels] = np.array(dims, np.int32)
        self.setc(FMED=np.array([med.data.ptr], np.uint64), FMEDL=np.array([levels], np.int32), FMEDDIM=dd, FMEDOFF=offs,
                  FMED_NEAR=np.array([near], np.float32))
        self.medium = {"levels": levels, "near": near, "mb": round(med.nbytes / 1e6, 1), "s": round(time.time() - t, 2),
                       "max0_per_m": round(float(med[:9 * n[0]].max()), 1)}   # fs-store-6: the fixed point holds up to 262,143 /m a cell
        return self.medium["mb"]

    DOM_Z = (0.0, 0.25e-3, 0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3, 16e-3, 32e-3, 64e-3, 128e-3, 512e-3)   # the layers after d0 (m)

    def build_dom(self, texel=0.35e-3, max_cols=1500000):
        """fs-store-6 11-13: the key's light through the hair as a map in the key's own view, built from the fibres (fl_dom_build
        pass 0 / 1, fl_dom_scan; call light() first): columns texel wide (wider when the hair's extent across the key needs more
        than max_cols), each with its first fibre depth and the optical depth summed in layers after it (DOM_Z). -> self.dom"""
        cp = self.cp; t = time.time(); s = np.asarray(self._hsun, np.float64)
        a = np.cross(s, [0.0, 0.0, 1.0] if abs(s[2]) < 0.9 else [1.0, 0.0, 0.0]); a /= np.linalg.norm(a); b_ = np.cross(s, a)
        # hs3: float64 host copies of every strand point (8.8 M x 3) hit the job's memory cap: the extent on the card, in chunks
        Pa = self.hs.a['P']; pad = 2e-3; ax = cp.asarray(np.stack([a, b_, s], 1).astype(np.float32)); lo3 = np.full(3, 1e30); hi3 = np.full(3, -1e30)
        for c0 in range(0, Pa.shape[0], 16384):
            P_ = cp.asarray(np.ascontiguousarray(Pa[c0:c0 + 16384], np.float32).reshape(-1, 3))   # no matmul: cuBLAS isn't installed
            q = cp.stack([P_[:, 0] * ax[0, j] + P_[:, 1] * ax[1, j] + P_[:, 2] * ax[2, j] for j in range(3)], 1); del P_
            lo3 = np.minimum(lo3, q.min(0).get()); hi3 = np.maximum(hi3, q.max(0).get()); del q
        u0, v0 = float(lo3[0]) - pad, float(lo3[1]) - pad; ext = (float(hi3[0]) + pad - u0, float(hi3[1]) + pad - v0)
        ts = max(float(texel), float(np.sqrt(ext[0] * ext[1] / max_cols))); nu, nv = int(np.ceil(ext[0] / ts)), int(np.ceil(ext[1] / ts))
        s_off = float(hi3[2]) + pad; ncol = nu * nv; KL = len(self.DOM_Z)
        if getattr(self, '_dom_buf', None) is None or self._dom_buf[0] != ncol:
            self._dom_buf = (ncol, cp.empty(ncol, cp.float32), cp.empty(ncol * (KL + 2), cp.uint32), cp.empty(ncol * KL, cp.float32), cp.empty(ncol * 4, cp.float32))
        _, d0, acc, T, C = self._dom_buf; d0.fill(1e30); acc.fill(0)
        ptr = lambda x: np.array([x.data.ptr], np.uint64)
        self.setc(FDOMD0=ptr(d0), FDOMACC=ptr(acc), FDOMT=ptr(T), FDOMC=ptr(C), FDOMA=a.astype(np.float32), FDOMB=b_.astype(np.float32),
                  FDOMO=np.array([u0, v0, ts, s_off], np.float32), FDOMN=np.array([nu, nv], np.int32), FDOMZ=np.array(self.DOM_Z, np.float32))
        kb = self.mod.get_function('fl_dom_build'); nspan = self.hs.n * (self.hs.a['P'].shape[1] - 1); CH = 1 << 19
        e0, e1 = cp.cuda.Event(), cp.cuda.Event(); worst = 0.0; nl = 0
        for ps in (0, 1):
            for s0 in range(0, nspan, CH):
                e0.record(); kb(((min(CH, nspan - s0) + 255) // 256,), (256,), (np.int32(s0), np.int32(nspan), np.int32(ps))); e1.record(); e1.synchronize()
                worst = max(worst, cp.cuda.get_elapsed_time(e0, e1)); nl += 1
        self.mod.get_function('fl_dom_scan')(((ncol + 255) // 256,), (256,), (np.int32(ncol),)); cp.cuda.Device().synchronize()
        self.dom = {'texel_mm': round(ts * 1e3, 3), 'nu': nu, 'nv': nv, 'mb': round((d0.nbytes + acc.nbytes + T.nbytes + C.nbytes) / 1e6, 1),
                    'launches': nl, 'worst_ms': round(worst, 1), 'cols_with_hair': int((d0 < 1e29).sum()), 'tau_max': round(float(T.reshape(-1, KL)[:, -1].max()), 2),
                    's': round(time.time() - t, 3)}
        print('fl dom:', self.dom, flush=True)
        return self.dom

    def skin_occ_med(self, cam, zbuf, wsun, pen_sun):
        """as skin_occ, the shadow ray's far part through the measured medium (fl_skin_occ4; build_medium() first)"""
        cp = self.cp; pos, fw, rt, upv, pixang = cam_frame(cam); W, H = int(cam["w"]), int(cam["h"])
        if not hasattr(self, "k_occ4"):
            self.k_occ4 = self.mod.get_function("fl_skin_occ4")
        Ts = cp.ones(W * H * 3, cp.float32)
        self._bands(self.k_occ4, (_f3(pos), _f3(fw), _f3(rt), _f3(upv), np.float32(pixang), np.int32(W), np.int32(H),
                                  cp.ascontiguousarray(zbuf.reshape(-1).astype(cp.float32)), np.float32(wsun), np.float32(pen_sun), Ts), W, H)
        return Ts.reshape(H, W, 3)

    def lean_begin(self, cam, zbuf, wsun, key_r, sky_r=0.12, pilot=(8, 6), chunk=(2, 2)):
        """the exact, adaptive skin shadow (fl_occ_pass, hair-shadow lane 11-11; the scalp): the PILOT rays per pixel (key, sky),
        in passes of chunk rays, over every pixel of zbuf. -> the occlusion from the pilot (H, W, 3); lean_refine() continues"""
        cp = self.cp; pos, fw, rt, upv, pixang = cam_frame(cam); W, H = int(cam["w"]), int(cam["h"]); t0 = time.time()
        if not hasattr(self, "k_pass"):
            self.k_pass = self.mod.get_function("fl_occ_pass"); self.k_fin = self.mod.get_function("fl_occ_finish")
        z = cp.ascontiguousarray(zbuf.reshape(-1).astype(cp.float32))
        acc = cp.zeros((W * H, 12), cp.float32); acc[:, 6] = 1e9; acc[:, 8] = 1e9; acc[:, 7] = -1e9; acc[:, 9] = -1e9
        self._lean = {"cam": (pos, fw, rt, upv, pixang, W, H), "z": z, "acc": acc, "wsun": wsun, "key_r": key_r, "sky_r": sky_r,
                      "k": 0, "q": 0, "pilot": pilot, "worst": 0.0, "passes": 0, "t": time.time() - t0}
        self._lean_passes(z, pilot, chunk, -1.0)
        return self._lean_finish()

    def lean_refine(self, zmask, total=(36, 24), chunk=(2, 2), conv=0.03):
        """after lean_begin: the remaining rays up to total, only at the pixels of zmask (< 1e3) whose pilot rays disagreed
        (conv: the spread of the rays' light share below which a pixel is done; < 0: every pixel). -> (H, W, 3)"""
        cp = self.cp; t0 = time.time()
        zm = cp.ascontiguousarray(zmask.reshape(-1).astype(cp.float32))
        self._lean_passes(zm, total, chunk, conv)
        L = self._lean; acc = L["acc"]
        refined = float(((acc[:, 10] > L["pilot"][0]) | (acc[:, 11] > L["pilot"][1])).sum()) / max(float((L["z"] < 1e3).sum()), 1.0)
        out = self._lean_finish(); L["t"] += time.time() - t0
        self.lean = {"passes": L["passes"], "refined": round(refined, 4), "worst_s": round(L["worst"], 3), "s": round(L["t"], 2)}
        self.last_bands = {"worst_s": round(L["worst"], 3)}
        return out

    def _lean_passes(self, z, upto, chunk, conv):
        L = self._lean; pos, fw, rt, upv, pixang, W, H = L["cam"]; t0 = time.time()
        head = (_f3(pos), _f3(fw), _f3(rt), _f3(upv), np.float32(pixang), np.int32(W), np.int32(H), z, np.float32(L["key_r"]), np.float32(L["sky_r"]))
        while L["k"] < upto[0] or L["q"] < upto[1]:
            k1, q1 = min(upto[0], L["k"] + chunk[0]), min(upto[1], L["q"] + chunk[1])
            self._bands(self.k_pass, head + (np.int32(L["k"]), np.int32(k1), np.int32(L["q"]), np.int32(q1), np.float32(conv),
                                             np.int32(L["pilot"][0]), L["acc"]), W, H)
            L["worst"] = max(L["worst"], self.last_bands["worst_s"]); L["passes"] += 1; L["k"], L["q"] = k1, q1
        L["t"] += time.time() - t0

    def _lean_finish(self):
        cp = self.cp; L = self._lean; W, H = L["cam"][5], L["cam"][6]; Ts = cp.ones(W * H * 3, cp.float32)
        self.k_fin(((W * H + 255) // 256,), (256,), (np.int32(W * H), L["z"], L["acc"], np.float32(L["wsun"]), Ts))
        self.last_bands = {"worst_s": round(L["worst"], 3)}
        return Ts.reshape(H, W, 3)

    def skin_occ_lean(self, cam, zbuf, wsun, key_r, sky_r=0.12, pilot=(8, 6), total=(36, 24), chunk=(2, 2), conv=0.03):
        """lean_begin + lean_refine over the same pixels (the probe's form). self.lean logs the passes, the share of pixels
        refined, the longest launch and the time"""
        self.lean_begin(cam, zbuf, wsun, key_r, sky_r, pilot, chunk)
        return self.lean_refine(zbuf, total, chunk, conv)

    def trace_rays(self, rays, ns=1):
        """the hair along given rays (fl_rays, hair-shadow lane 11-11): rays (n, 11) float32 cupy = origin 3, unit direction 3,
        cone half-angle (rad), far limit (m), the radiance behind 3 -> (n, 8): premultiplied rgb 3, coloured transmittance 3,
        coverage transmittance, first-hit distance. ns > 1: ns x ns stratified exact sub-rays over the cone (the reference).
        Launched in chunks of rays, each timed (~BAND_S)."""
        cp = self.cp; rays = cp.ascontiguousarray(rays, dtype=cp.float32); n = int(rays.shape[0]); out = cp.zeros((max(n, 1), 8), cp.float32)
        if ns > 1:
            out[:, 7] = 1e30
        if not hasattr(self, "k_rays"):
            self.k_rays = self.mod.get_function("fl_rays")
        # (11-12) ns > 1: at most 36 sub-rays a ray per launch (16 x 16 in one launch ran 1.7-2.1 s), the rays in timed chunks
        worst = 0.0; e0, e1 = cp.cuda.Event(), cp.cuda.Event(); tot = ns * ns if ns > 1 else 1
        for a0 in range(0, tot, 36):
            a1 = min(tot, a0 + 36); i0, chunk = 0, 4096
            while i0 < n:
                m = min(chunk, n - i0); e0.record()
                self.k_rays(((m + 127) // 128,), (128,), (np.int32(n), rays, np.int32(ns), out, np.int32(i0), np.int32(a0), np.int32(a1)))
                e1.record(); e1.synchronize(); t = cp.cuda.get_elapsed_time(e0, e1) * 1e-3; worst = max(worst, t); i0 += m
                chunk = int(min(2 * chunk, max(128, chunk * BAND_S / max(t, 1e-4))))
        if ns > 1:
            out[:, :7] *= 1.0 / (ns * ns)
        self.last_bands = {"rays": n, "worst_s": round(worst, 3)}
        return out[:n]

    def composite(self, img, cam, zbuf):
        rgb, T, Tc, dep, ids, ms = self.layer(cam, zbuf, img)
        out = rgb + Tc * img
        nan = int((~self.cp.isfinite(out)).any(-1).sum())
        st = {"ms": ms, "nan": nan, "px_cov_gt_0.5": int((T < 0.5).sum()), "strands_seen": int(len(self.cp.unique(ids))) - 1}
        return self.cp.nan_to_num(out), st
