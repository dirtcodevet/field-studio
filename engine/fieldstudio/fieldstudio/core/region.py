"""V1.2 field regions: a box / sphere crop of a native brick-field version (the FV8 store format) as a Field Studio
object, and the write-back of its sculpt layer as a NEW version of that store.

The region object's base node is kind 'sampled': the crop's stored samples on a dense lattice (the finest level that
holds bricks in the crop, L0 0.25 mm or L1 0.5 mm; points that level does not hold come from the next level that does,
through FieldStore.evaluate_F; points no level holds take +-clamp by the sign of their nearest known sample), evaluated
trilinear by the numba program (program.OP_SAMPLED) and intersected with the crop shape, so the region is a closed solid
with flat (box) or round (sphere) cut faces. Its Lipschitz bound is the measured per-cell bound of the trilinear lattice
(rigorous for trilinear, D), max'd with the crop shape's 1. Material and colour stay neutral: only F is read.

The store is read ONLY through its own module (fv8_field.FieldStore: bricks_in_box / gather / evaluate_F) and written
ONLY through its ops: write-back builds the sculpt delta-F on its lattice in the store frame and applies
{"op": "delta_grid"} (every level that holds bricks in the support, brick allocation where the moved surface leaves a
band), then FieldStore.commit(new dir) - a new version folder; the source version is never modified. Publishing is
never done here (publishing is a separate step, after checks); the face shape guard (fv8_current.py --check) runs on the
new version and its result is returned.

Node params (all plain JSON, so the op log replays): source (absolute version dir), source_id (the version's identity at
load: sha256 of its version.json / meta.json), crop 'box' | 'sphere', centre (store frame, m), size ([hx, hy, hz] half
extents for a box; [r, r, r] for a sphere), anchor (store-frame point that is the object's local origin, snapped to the
grid level's lattice so the sculpt lattice (0.5 mm) lies on the store's L1 samples), clamp (m).
Data stays where it is: nothing here uploads or copies a field anywhere but the new version folder.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

_CACHE = {}          # canonical params json -> dict(data float64 flat, grid_lo local (3,), h, dims, L, report)
_FV8 = {}


def find_tools(field_dir):
    """the directory holding fv8_field.py for a store version: $FIELDSTUDIO_FV8_TOOLS, else the nearest parent of the
    version that holds it (versions live in <tools>/work/fv8_store/<version>)."""
    env = os.environ.get('FIELDSTUDIO_FV8_TOOLS')
    if env and os.path.exists(os.path.join(env, 'fv8_field.py')):
        return env
    d = os.path.abspath(field_dir)
    for _ in range(6):
        d = os.path.dirname(d)
        if os.path.exists(os.path.join(d, 'fv8_field.py')):
            return d
    raise FileNotFoundError('fv8_field.py not found above %s (set FIELDSTUDIO_FV8_TOOLS)' % field_dir)


def fv8(field_dir):
    """the store's own module (imported from its tools directory, not re-implemented)"""
    tools = find_tools(field_dir)
    if tools not in _FV8:
        spec = importlib.util.spec_from_file_location('fv8_field', os.path.join(tools, 'fv8_field.py'))
        m = importlib.util.module_from_spec(spec)
        sys.modules.setdefault('fv8_field', m)
        spec.loader.exec_module(m)
        _FV8[tools] = m
    return _FV8[tools]


def source_id(field_dir):
    for fn in ('version.json', 'meta.json'):
        p = os.path.join(field_dir, fn)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return '%s:%s' % (fn, hashlib.sha256(f.read()).hexdigest()[:24])
    return ''


def current_version(pointer):
    """the version a FIELD_CURRENT.json pointer names (relative paths resolve from the repo root above it)"""
    d = json.load(open(pointer))['field_dir']
    if os.path.isabs(d):
        return d
    p = os.path.abspath(os.path.dirname(pointer))
    for _ in range(8):
        if os.path.isdir(os.path.join(p, d)):
            return os.path.normpath(os.path.join(p, d))
        p = os.path.dirname(p)
    raise FileNotFoundError(d)


def region_params(field_dir, centre, size, crop='box', clamp=None):
    """node params for a crop (centre store frame, size: half extent(s) for a box, radius for a sphere). clamp: None =
    the grid level's cap (M, a 6 cm crop: clamp 6 mm gave a cell bound of 20.8 - L0's +-3 mm cap beside coarser
    levels' larger values; clamp 3 mm gave 1.57)"""
    field_dir = os.path.abspath(field_dir)
    F8 = fv8(field_dir)
    S = F8.FieldStore(field_dir)
    sz = np.broadcast_to(np.asarray(size, float), (3,)).astype(float)
    if crop == 'sphere':
        sz = np.full(3, float(np.max(sz)))
    c = np.asarray(centre, float)
    lv = _grid_level(S, F8, c - sz, c + sz)
    h = F8.LEVEL_H[lv]
    anchor = S.org + np.round((c - S.org) / (2 * h if lv == 0 else h)) * (2 * h if lv == 0 else h)
    if clamp is None:
        clamp = float(S.lv[lv].get('cap_m', 12 * h))
    return {'source': field_dir, 'source_id': source_id(field_dir), 'crop': crop,
            'centre': [float(v) for v in c], 'size': [float(v) for v in sz], 'anchor': [float(v) for v in anchor],
            'clamp': float(clamp)}


def _grid_level(S, F8, lo, hi):
    for l in sorted(S.levels):
        if len(S.bricks_in_box(l, lo, hi)):
            return l
    return max(S.levels)


def load(params):
    """-> cached dict: data (float64, C order (nx,ny,nz) flat), grid_lo (local), h, dims, L, report"""
    key = json.dumps(params, sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]
    t0 = time.perf_counter()
    src = params['source']
    if params.get('source_id') and source_id(src) != params['source_id']:
        raise ValueError('field region: %s changed since it was loaded (%s)' % (src, params['source_id']))
    F8 = fv8(src)
    S = F8.FieldStore(src)
    c = np.asarray(params['centre'], float)
    sz = np.asarray(params['size'], float)
    anchor = np.asarray(params['anchor'], float)
    lo, hi = c - sz, c + sz
    lv = _grid_level(S, F8, lo, hi)
    h = F8.LEVEL_H[lv]
    i0 = np.floor((lo - S.org) / h).astype(np.int64) - 2           # a 2-node apron round the crop
    i1 = np.ceil((hi - S.org) / h).astype(np.int64) + 2
    dims = tuple(int(v) for v in (i1 - i0 + 1))
    V = np.full(dims, np.nan, np.float32)
    # 1. the grid level's own samples (exact copies of the stored fp16 values)
    ids = S.bricks_in_box(lv, S.org + i0 * h, S.org + i1 * h)
    n_own = 0
    if len(ids):
        W, g0, _I, _dup = S.gather(lv, ids)
        a0 = np.maximum(g0, i0)
        a1 = np.minimum(g0 + np.array(W.shape), i1 + 1)
        if np.all(a1 > a0):
            V[tuple(slice(a0[k] - i0[k], a1[k] - i0[k]) for k in range(3))] = \
                W[tuple(slice(a0[k] - g0[k], a1[k] - g0[k]) for k in range(3))]
        n_own = int(np.isfinite(V).sum())
        del W
    # 2. points the grid level does not hold: the finest level that does (the store's evaluate_F, trilinear)
    miss = np.argwhere(np.isnan(V))
    n_coarse = 0
    for k0 in range(0, len(miss), 1 << 20):
        q = miss[k0:k0 + (1 << 20)]
        X = S.org + (q + i0) * h
        f = S.evaluate_F(X)
        V[q[:, 0], q[:, 1], q[:, 2]] = f
        n_coarse += int(np.isfinite(f).sum())
    # 3. no level holds them: +-clamp by the nearest known sample's sign; everything clamped to +-clamp (keeps the
    #    zero set; |clamp(F)| <= |F| keeps the distance bound; clamp is 1-Lipschitz, so L does not grow)
    cl = float(params.get('clamp', 0.006))
    nan = np.isnan(V)
    n_fill = int(nan.sum())
    if n_fill and n_fill < V.size:
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
        V[nan] = np.sign(V[tuple(ix[nan] for ix in idx)]) * cl
        del idx
    elif n_fill:
        V[:] = cl
    np.clip(V, -cl, cl, out=V)
    Lmax, Lp999, ncell = F8._lip_cells(V, h)
    L = max(1.0, float(Lmax or 1.0))
    grid_lo = S.org + i0 * h - anchor
    rep = {'level': int(lv), 'h_m': h, 'dims': list(dims), 'samples': int(V.size), 'from_grid_level': n_own,
           'from_coarser_levels': n_coarse, 'filled_by_sign': n_fill, 'L_cell_max': round(L, 4),
           'L_cell_p999': None if Lp999 is None else round(Lp999, 4), 'load_s': round(time.perf_counter() - t0, 2),
           'MB': round(V.size * 8 / 1e6, 1)}
    out = {'data': np.ascontiguousarray(V, np.float64).ravel(), 'grid_lo': grid_lo, 'h': h, 'dims': dims, 'L': L,
           'report': rep}
    if len(_CACHE) > 4:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def compile_params(params):
    """-> (OP_SAMPLED header (16 floats; slot 15 = data offset, patched by compile_tree), data, lo, hi, L), local frame"""
    R = load(params)
    anchor = np.asarray(params['anchor'], float)
    c = np.asarray(params['centre'], float) - anchor
    sz = np.asarray(params['size'], float)
    g = R['grid_lo']
    hdr = [g[0], g[1], g[2], R['h'], R['dims'][0], R['dims'][1], R['dims'][2],
           1.0 if params.get('crop') == 'sphere' else 0.0, c[0], c[1], c[2], sz[0], sz[1], sz[2], 0.0, 0.0]
    return hdr, R['data'], c - sz, c + sz, R['L']


# ---------------------------------------------------------------- write-back
def find_region_node(base):
    """the 'sampled' node and whether its frame is the object's local frame (no domain mutation above it)"""
    path = []

    def walk(n, above):
        if n.kind == 'sampled':
            path.append((n, above))
            return
        for ch in n.children:
            walk(ch, above or n.kind in ('transform', 'twist', 'bend', 'taper'))
    walk(base, False)
    return path[0] if path else (None, False)


def sculpt_delta(scene, obj):
    """the object's sculpt layer as a dense delta-F lattice: (origin local (3,), h, D float32), D zero where no brick"""
    g = obj.sculpt
    g.region_box()                       # 11-09: refresh the allocated range after frees (scene.SculptGrid)
    if g.bmin is None:
        return None
    B = 8
    bmin, bmax = g.bmin, g.bmax
    dims = (bmax - bmin + 1) * B + 1
    D = np.zeros(tuple(int(v) for v in dims), np.float32)
    for b, s in g.slots.items():
        o = (np.asarray(b) - bmin) * B
        D[o[0]:o[0] + B, o[1]:o[1] + B, o[2]:o[2] + B] = scene.store.delta[s]
    return bmin * B * g.h, g.h, D


def prepare_write_back(scene, oid):
    """read what write-back needs from the scene (call under the scene lock; fast): the region node's params and the
    object's sculpt delta on its lattice in the store frame, cut to the crop shape"""
    obj = scene.get(oid)
    node, moved = find_region_node(obj.base)
    if node is None:
        raise ValueError('object %s has no field region node' % obj.name)
    if moved:
        raise ValueError('the region node sits under a domain mutation (transform / twist / bend / taper): its sculpt '
                         'frame is not the store frame, so it cannot be written back as is')
    p = dict(node.params)
    sd = sculpt_delta(scene, obj)
    if sd is None or not np.any(sd[2]):
        raise ValueError('the region has no sculpt to write back')
    o_loc, h, D = sd
    anchor = np.asarray(p['anchor'], float)
    # the delta outside the crop shape is dropped (the store beyond the crop was never shown)
    ax = [o_loc[k] + anchor[k] + np.arange(D.shape[k]) * h for k in range(3)]
    c = np.asarray(p['centre'], float)
    sz = np.asarray(p['size'], float)
    if p.get('crop') == 'sphere':
        r2 = sum((ax[k].reshape([-1 if j == k else 1 for j in range(3)]) - c[k]) ** 2 for k in range(3))
        inside = r2 <= sz[0] ** 2
    else:
        inside = np.ones(D.shape, bool)
        for k in range(3):
            m = (ax[k] >= c[k] - sz[k]) & (ax[k] <= c[k] + sz[k])
            inside &= m.reshape([-1 if j == k else 1 for j in range(3)])
    dropped = int(((D != 0) & ~inside).sum())
    D = np.where(inside, D, 0).astype(np.float32)
    return {'params': p, 'name': obj.name, 'origin': o_loc + anchor, 'h': float(h), 'D': D, 'dropped': dropped}


def write_back(scene, oid, out_dir, note='', allocate=True, check=True, timeout=900):
    """apply object oid's sculpt delta to its source version as a NEW version (out_dir must not exist).
    -> report dict (op report, commit, guard check). Never publishes."""
    return write_back_prepared(prepare_write_back(scene, oid), out_dir, note, allocate, check, timeout)


def write_back_prepared(prep, out_dir, note='', allocate=True, check=True, timeout=900):
    t0 = time.perf_counter()
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError('%s exists: write-back makes a NEW version folder only' % out_dir)
    p = prep['params']
    src = p['source']
    if p.get('source_id') and source_id(src) != p['source_id']:
        raise ValueError('the source version changed since the region was loaded')
    D = prep['D']
    F8 = fv8(src)
    S = F8.FieldStore(src)
    # fs-trust 11-11: the store module builds the op: fv8_write's write_txn by default (since 11-11), with D on the
    # region's own lattice (0.25 or 0.5 mm) carried to L0's nodes; FV8_WRITER=legacy: the old delta_grid op. (The earlier
    # txn branch here passed a 0.5 mm region's D as if it lay on L0's lattice.)
    op = F8.delta_grid_op(prep['origin'], prep['h'], D, allocate=allocate, source='Field Studio region write-back',
                          object=prep['name'], region={k: p[k] for k in ('crop', 'centre', 'size', 'anchor')})
    rep_op = S.apply(op)
    t1 = time.perf_counter()
    S2 = S.commit(out_dir, note=note or 'Field Studio region write-back (%s); NOT published' % prep['name'])
    t2 = time.perf_counter()
    rep = {'source': src, 'new_version': out_dir, 'delta_nodes_nonzero': int((D != 0).sum()),
           'delta_dropped_outside_crop': prep['dropped'], 'delta_max_abs_um': rep_op.get('delta_max_abs_um'),
           'op': {k: v for k, v in op.items() if k != 'data_zlib_b64'}, 'op_report': rep_op,
           'apply_s': round(t1 - t0, 2), 'commit_s': round(t2 - t1, 2), 'published': False}
    del S, S2
    if check:
        rep['guard'] = guard_check(out_dir, timeout=timeout)
    return rep


def guard_check(version_dir, timeout=900):
    """python fv8_current.py --check <abs dir> (the face shape guard + structure + content hash) -> dict"""
    tools = find_tools(version_dir)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, os.path.join(tools, 'fv8_current.py'), '--check', os.path.abspath(version_dir)],
                       cwd=tools, capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    out = {'returncode': r.returncode, 's': round(time.perf_counter() - t0, 1)}
    try:
        out['report'] = json.loads(r.stdout)
    except ValueError:
        out['stdout_tail'] = r.stdout[-2000:]
        out['stderr_tail'] = r.stderr[-2000:]
    out['ok'] = r.returncode == 0 and out.get('report', {}).get('why') == 'ok'
    return out
