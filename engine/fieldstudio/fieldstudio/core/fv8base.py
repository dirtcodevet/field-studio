"""Her FV8 brick store as an editable base node (milestone item 2, 11-08 night): the primitive 'fv8store'.

The node's params name the store (source folder + source_id = the sha of its meta / tile grid), so describe(), the
state hash and the op log stay small; compile() packs the store into the program's data (after the params, as the
'sampled' region does): the tile grid's slots, each level's tile table and every brick's 9^3 samples, as float64
values (exact copies of the stored fp16; the card's float32 copy is exact too). The evaluator (program._fv8 and
program.cu's fv8_d) is the store's own evaluate_F: per point the finest level holding a brick there, trilinear inside
it; a level whose tile holds no brick there gives a lower bound (its band - h) and the next coarser level is read; no
level at all: + the larger of those bounds (outside her; L3 covers her whole interior: meta 'coverage whole body').

Header (48 floats at the op's parameter offset o; slot 15 = the data's offset from o, patched by compile_tree):
  0-2 tile grid origin (store frame), 3 tile size, 4-6 grid dims, 7 the bound outside every brick (L3 band - h),
  8-11 h per level, 16-19 bricks per tile axis per level (0: level absent), 20-23 band per level,
  24-25 grid offset in the data (hi, lo: off = hi * 2^20 + lo, each exact in float32), 26-33 tile-table offsets,
  34-41 sample offsets (per level, hi / lo), 42-45 cap per level, 46 the data's address (the CPU reads the shared
  float32 array in place), 47 its size. The card's evaluator (gpu/geval.py) appends the data to its own copy of the
  program and sets slot 15 to its offset.
Capped samples (|v| >= 0.999 cap: no distance known there) take the store's editors' rule (fv8_field.pre_estimate /
grid_L0): the finest coarser level's uncapped value at that sample (the coarsest level present as it is), so a stroke
that moves the surface past a band edits a true field, as the write-back's delta_grid (capped='coarser') expects.
"""
import hashlib
import json
import os
import time

import numpy as np

SPLIT = 1 << 20
_CACHE = {}
NLEV = 4


def source_id(field_dir):
    h = hashlib.sha256()
    for f in ('meta.json', 'tile_grid.npy'):
        with open(os.path.join(field_dir, f), 'rb') as fh:
            h.update(fh.read())
    for l in range(NLEV):
        p = os.path.join(field_dir, 'L%d_sdf.npy' % l)
        if os.path.exists(p):
            h.update(str(os.path.getsize(p)).encode())
            with open(p, 'rb') as fh:            # head and tail of the samples: a changed file changes the id
                h.update(fh.read(1 << 20))
                fh.seek(max(os.path.getsize(p) - (1 << 20), 0))
                h.update(fh.read())
    return h.hexdigest()[:24]


class LazyBlob:
    """the store's data, written straight into the compiled program's array by ops.compile_tree (no 2 GB staging copy:
    the first trials peaked at 2x her data while concatenating)"""
    big = True

    def __init__(self, size, pieces):
        self.size = int(size)
        self.shape = (self.size,)
        self.pieces = pieces

    def fill(self, out):
        o = 0
        for pc in self.pieces:
            if pc[0] == 'arr':
                a = pc[1]
                out[o:o + a.size] = a
                o += a.size
            else:
                m = np.load(pc[1], mmap_mode='r').reshape(-1)
                for k in range(0, pc[2], 1 << 24):               # 16 M samples at a time (fp16 -> float64)
                    out[o + k:o + min(k + (1 << 24), pc[2])] = m[k:k + (1 << 24)]
                o += pc[2]
        assert o == self.size


_DATA = {}          # source folder -> float32 data (kept alive: compiled programs hold its address)


def _shared(field_dir, size, pieces):
    """r-ptr (11-08 night): her store's data as ONE float32 array per store per process (fp16 samples and table indices
    are exact in float32), read in place by the program through its address; programs and scene packs hold only the
    header (before: 2 GB float64 inside every compiled program and every pack; Open's replay reached 22 GB)"""
    key = os.path.abspath(field_dir)
    d = _DATA.get(key)
    if d is None or d.size != size:
        d = np.empty(size, np.float32)
        o = 0
        for pc in pieces:
            if pc[0] == 'arr':
                d[o:o + pc[1].size] = pc[1]
                o += pc[1].size
            else:
                m = np.load(pc[1], mmap_mode='r').reshape(-1)
                for k in range(0, pc[2], 1 << 24):
                    d[o + k:o + min(k + (1 << 24), pc[2])] = m[k:k + (1 << 24)]
                o += pc[2]
        d.setflags(write=False)
        _DATA[key] = d
    return d


def data_of(field_dir):
    return _DATA.get(os.path.abspath(field_dir))


def _split(off):
    return [float(off // SPLIT), float(off % SPLIT)]


def load(field_dir, sid=None):
    """-> {'hdr': 48 floats (slot 15 = 0), 'data': float64, 'lo', 'hi' (store frame), 'report'} (cached)"""
    key = (os.path.abspath(field_dir), sid)
    if key in _CACHE:
        return _CACHE[key]
    t0 = time.perf_counter()
    if sid is not None and source_id(field_dir) != sid:
        raise ValueError('fv8store: %s changed since it was recorded (%s)' % (field_dir, sid))
    meta = json.load(open(os.path.join(field_dir, 'meta.json')))
    org = np.asarray(meta['tile_grid_origin'], float)
    tm = float(meta['tile_m'])
    tg = np.load(os.path.join(field_dir, 'tile_grid.npy'))
    dims = tg.shape[:3]
    hdr = np.zeros(48)
    hdr[0:3] = org
    hdr[3] = tm
    hdr[4:7] = dims
    grid = np.ascontiguousarray(tg.astype(np.float64)).ravel()
    pieces = [('arr', grid)]
    off = grid.size
    hdr[24:26] = _split(0)
    rep = {'levels': {}}
    bound = 0.0
    for l in range(NLEV):
        key_l = 'L%d' % l
        if key_l not in meta['levels']:
            continue
        lv = meta['levels'][key_l]
        h = float(lv['h_m'])
        nb = int(lv['bricks_per_tile_axis'])
        band = float(lv['band_m'])
        tt = np.load(os.path.join(field_dir, 'L%d_tile_table.npy' % l)).astype(np.float64).ravel()
        sdf_path = os.path.join(field_dir, 'L%d_sdf.npy' % l)
        n_sdf = int(np.prod(np.load(sdf_path, mmap_mode='r').shape))
        hdr[8 + l] = h
        hdr[16 + l] = nb
        hdr[20 + l] = band
        hdr[42 + l] = float(lv.get('cap_m', 1.0))     # capped samples: the editors' rule (fv8_field.pre_estimate)
        hdr[26 + 2 * l: 28 + 2 * l] = _split(off)
        pieces.append(('arr', tt))
        off += tt.size
        hdr[34 + 2 * l: 36 + 2 * l] = _split(off)
        pieces.append(('sdf', sdf_path, n_sdf))
        off += n_sdf
        bound = band - h
        rep['levels'][key_l] = {'h_m': h, 'bricks': n_sdf // 729, 'band_m': band}
    hdr[7] = max(bound, 0.0)
    data = _shared(field_dir, off, pieces)                # one float32 copy per store per process (r-ptr)
    hdr[46] = float(data.ctypes.data)                    # its address and size: program._fv8 reads it in place
    hdr[47] = float(data.size)
    lo = org
    hi = org + np.asarray(dims, float) * tm
    rep.update({'MB': round(off * 8 / 2 ** 20, 1), 'load_s': round(time.perf_counter() - t0, 2)})
    out = {'hdr': hdr, 'data': data, 'lo': lo, 'hi': hi, 'report': rep}
    # not cached: the compiled program holds its own copy (2 GB each; the first trial's peak working set was 11.7 GB
    # with a cached copy beside it); a recompile reloads in ~1.3 s
    return out


def params_for(field_dir):
    return {'source': os.path.abspath(field_dir), 'source_id': source_id(field_dir)}


# her placement in Field Studio (as the viewport's reference store: FV8 local is z up, feet at z = -1.424; feet on y = 0)
TRANSFORM = {'translate': [0.0, 1.424, 0.0], 'rotate': [-90.0, 0.0, 0.0], 'scale': 1.0}


# fs-store-3 11-13 (decided: one eye per side, the eye engine's, used everywhere): the eye objects are the
# surface the real renderer DRAWS (fv8_eyes.cuh eye_d0: the coat sphere U the cornea sphere, polynomial smooth union of
# k 0.15 mm, EYE_LIMB_K_MM). The window's smooth union (OP_SUNION, smin_d) is the same polynomial, so with blend = k the
# two surfaces are one. Before: the 12.17 mm eyeball sphere U the cornea sphere, hard (the old stand-in, 50 um inside the
# drawn coat and without the limbus fill).
EYE_LIMB_K_M = 1.5e-4


def _eye_params_path(field_dir):
    import os as _os
    return _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(field_dir))), 'eye_params.json')


def eye_base(e):
    """one eye's base tree (AddObject spec form, her store's frame, metres) from its eye_params.json entry"""
    def sph(c, r):
        return {'kind': 'transform', 'params': {'translate': [float(v) for v in c], 'rotate': [0.0, 0.0, 0.0],
                                                'scale': [1.0, 1.0, 1.0]},
                'children': [{'kind': 'sphere', 'params': {'radius': float(r)}, 'children': []}]}
    return {'kind': 'union', 'params': {'blend': EYE_LIMB_K_M},
            'children': [sph(e['eyeball_center'], e.get('coat_radius', e['eyeball_radius'])),
                         sph(e['cornea_center'], e['cornea_radius'])]}


def eye_specs(field_dir):
    """her eyes as separate protected objects (AddObject specs), from work/eye_params.json (her own fit, in her store's
    frame): the drawn eye's surface (eye_base: coat sphere U cornea sphere, smooth union k 0.15 mm)"""
    import os as _os
    p = _eye_params_path(field_dir)
    if not _os.path.exists(p):
        return []
    E = json.load(open(p))
    out = []
    for side in ('L', 'R'):
        e = E.get(side)
        if not e:
            continue
        out.append({'name': 'eye %s (protected)' % side, 'base': eye_base(e), 'translate': TRANSFORM['translate'],
                    'rotate': TRANSFORM['rotate'], 'h': 0.00025})
    return out


def eye_upgrades(scene, field_dir=None):
    """fs-store-3 11-13: the eye objects of an older project that are still the old stand-in (a hard union of the eyeball
    sphere and the cornea sphere, eye_params.json's radii) -> [(oid, new base tree, side)]: the same tree with the coat
    radius and the smooth union of the drawn eye. The caller pushes SetBase for each (one undo step each, logged)."""
    import os as _os
    p = _eye_params_path(field_dir) if field_dir else None
    if p is None or not _os.path.exists(p):
        p = _os.environ.get('FIELDSTUDIO_EYE_PARAMS', '')
    if not _os.path.exists(p):
        return []
    E = json.load(open(p))
    out = []
    for o in scene.objects:
        if not (o.name.startswith('eye ') and '(protected)' in o.name):
            continue
        side = o.name[4:5]
        e = E.get(side)
        if not e:
            continue
        t = o.base.to_dict()
        try:
            kids = t['children']
            r0 = float(kids[0]['children'][0]['params']['radius'])
            r1 = float(kids[1]['children'][0]['params']['radius'])
            c0 = [float(v) for v in kids[0]['params']['translate']]
        except (KeyError, IndexError, TypeError):
            continue
        if t.get('kind') != 'union' or float(t['params'].get('blend', 0.0)) != 0.0:
            continue
        if abs(r0 - float(e['eyeball_radius'])) > 1e-7 or abs(r1 - float(e['cornea_radius'])) > 1e-7:
            continue
        if max(abs(a - b) for a, b in zip(c0, e['eyeball_center'])) > 1e-7:
            continue
        t['params']['blend'] = EYE_LIMB_K_M
        kids[0]['children'][0]['params']['radius'] = float(e.get('coat_radius', e['eyeball_radius']))
        out.append((o.id, t, side))
    return out


_LAT = {}


def lattice_load(path, sha=None):
    """milestone: a dense field file (npz: data (nx, ny, nz) float32 field values, lo (3,), h) -> {'data' float64 flat,
    'lo', 'h', 'dims'}, cached; sha (of the file) checked when given"""
    key = (os.path.abspath(path), sha)
    if key in _LAT:
        return _LAT[key]
    # fs-mouth 11-12: only small files are kept (< 32 MB): the compiled program copies the data it needs, so a large
    # lattice cached here was a second float64 copy kept forever (0.92 GB for the mouth set, after its undo too)
    small = os.path.getsize(path) < 32 * 2 ** 20
    if sha:
        hh = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:24]
        if hh != sha:
            raise ValueError('lattice %s changed since it was recorded' % path)
    with np.load(path) as z:
        D = np.asarray(z['data'], np.float64)
        out = {'data': np.ascontiguousarray(D).ravel(), 'lo': np.asarray(z['lo'], float), 'h': float(z['h']),
               'dims': D.shape}
    if small:
        _LAT[key] = out
    return out


def file_sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:24]
