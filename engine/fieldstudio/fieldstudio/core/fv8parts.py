"""Her store's PARTS in Field Studio (fs-contact S3, 11-11).

A store version written by fv8_parts (parts.json + parts/L0_band_*: each part's own field, the
authority; the union, k entries and leaves are caches) opens as ONE OBJECT PER PART, every one an 'fv8store' node of its own
store, built once per source version into the cache (cache version 2, 11-11 evening; version 1 used dense lattices over each
part's box and a dense cut over the parts' box: 155 MB for her lips, 3-6 GB estimated for her whole body, memory_mark.md):
  * part 0 (her body) is part0/: her store with part 0's own field written at every node of every other part's brick
    (fv8_write targets, L0 exactly as given, L1-L3 rebuilt from it), so it draws her body without the other parts' tissue
    (the face continued under the lips, cut away beneath them); the union's k files and leaves are not kept there;
  * every other part is part_<pid>/, a small store of its band rows (RowSampler: its rows bit for bit at every L0 node, a
    distance lower bound beyond them), with a part record {'source', 'pid', 'name', 'pairs': {other pid: touch | slide |
    bond | pass}}. RAM and the card follow the band rows, not the boxes.
Each object draws its own field (the tracer's nearest hit over objects = the k rule's visible part). Brushes edit one
part; touch / slide partners collide (core/collide.part_partners, gap 0, existing overlap kept). Write-back (writeback())
evaluates every part object's field on the store's L0 lattice over the parts' box (her own edits outside it go to her
REAL source, not the cache) and calls fv8_parts.rebuild: the union,
the bands, the k entries, the leaves all rebuilt from the parts, never written by hand.
"""
import json
import os
import sys

import numpy as np

from . import fv8base, region

H0 = 0.00025


def _fp(field_dir):
    """the store's own fv8_parts module (beside its fv8_field), imported once"""
    F8 = region.fv8(field_dir); d = os.path.dirname(os.path.abspath(F8.__file__))
    if d not in sys.path:
        sys.path.insert(0, d)
    import fv8_parts                                                   # noqa: E402
    return fv8_parts, F8


def has_parts(field_dir):
    return os.path.exists(os.path.join(field_dir, 'parts.json')) and os.path.exists(os.path.join(field_dir, 'parts', 'L0_band_pid.npy'))


def _cache_dir(field_dir):
    alt = os.environ.get('FIELDSTUDIO_PARTS_DIR')
    base = os.path.join(alt, os.path.basename(os.path.abspath(field_dir))) if alt else os.path.abspath(field_dir)
    return base + '_fsparts'


def _pairs(graph):
    """parts.json pairs {"a-b": {"contact": mode}} -> {a: {b: mode}, b: {a: mode}} (str keys, as saved in projects)"""
    out = {}
    for k, v in (graph.get('pairs') or {}).items():
        a, b = k.split('-'); m = v.get('contact') if isinstance(v, dict) else v
        out.setdefault(a, {})[b] = m; out.setdefault(b, {})[a] = m
    return out


def parts_box(B, pid=None, margin_bricks=2):
    """the L0 node box (g0, g1 inclusive) of a part's band bricks (all non-zero parts when pid is None), + margin"""
    sel = (B['pid'] != 0) if pid is None else (B['pid'] == pid)
    ijk = np.concatenate([B['ijk'][sel], B['inside'][(B['inside'][:, 3] != 0) if pid is None else (B['inside'][:, 3] == pid)][:, :3]])
    g0 = (ijk.min(0) - margin_bricks) * 8; g1 = (ijk.max(0) + 1 + margin_bricks) * 8
    return g0.astype(np.int64), g1.astype(np.int64)


CACHE_VERSION = 7          # 7 (19:58): L3 solid over each part's box (the inside rule); 6 (19:36): = 4 again (5 widened the zone: worse); 4: part 0's coarse = its own signed distance at the parts' bricks; 3: max(union, part 0); 2 (11-11 evening): sparse part stores (memory mark: the dense boxes of version 1 cost 3-6 GB for her whole body)


def _zero_points(ijk, vals):
    """lattice-edge zero crossings of rows (n, 9, 9, 9) -> (m, 3) in L0 node units of the store (not metres)"""
    out = []
    for ax in range(3):
        A = np.moveaxis(vals, ax + 1, 1); va = A[:, :-1]; vb = A[:, 1:]; m = (va < 0) != (vb < 0)
        if not m.any():
            continue
        b, i, j, k = np.nonzero(m); t = va[b, i, j, k] / (va[b, i, j, k] - vb[b, i, j, k])
        if ax == 0: loc = np.stack([i + t, j.astype(float), k.astype(float)], 1)
        elif ax == 1: loc = np.stack([j.astype(float), i + t, k.astype(float)], 1)
        else: loc = np.stack([j.astype(float), k.astype(float), i + t], 1)
        out.append(ijk[b] * 8 + loc)
    return np.concatenate(out) if out else np.zeros((0, 3))


class RowSampler:
    """a part's own field from its band rows, sparse (no box array): fn(X (n, 3) m, store frame) -> m.
    In a row brick: trilinear in the row (a node on a brick face reads a row that holds it, so every L0 node gives its row's own
    value bit for bit); elsewhere: sign x (distance to the part's zero points - 0.2 mm), minus inside the part's inside bricks.
    A non-row brick lies >= band - half a cell diagonal (~1.28 mm) from the surface, so the 0.2 mm keeps the far value a lower bound
    of the distance (the zero points are 0.25 mm apart) and a sphere step from it never passes the surface"""
    MARGIN = 0.2e-3

    def __init__(self, org, ijk, vals, inside, cap):
        from scipy.spatial import cKDTree
        self.org = np.asarray(org, float); self.cap = float(cap)
        self.vals = np.asarray(vals, np.float32).reshape(-1, 9, 9, 9)
        ijk = np.asarray(ijk, np.int64); inside = np.asarray(inside, np.int64).reshape(-1, 3)
        allb = np.concatenate([ijk, inside]) if len(inside) else ijk
        self.base = allb.min(0) - 2; self.K = int((allb.max(0) - self.base).max() + 4)
        rk = self._key(ijk); o = np.argsort(rk); self.rk = rk[o]; self.ri = o
        self.ik = np.sort(self._key(inside)) if len(inside) else np.zeros(0, np.int64)
        Z = _zero_points(ijk, self.vals)
        self.tree = cKDTree(Z * H0 + self.org) if len(Z) else None

    def _key(self, b):
        b = np.asarray(b, np.int64) - self.base
        ok = np.all((b >= 0) & (b < self.K), axis=-1)
        return np.where(ok, (b[..., 0] * self.K + b[..., 1]) * self.K + b[..., 2], -1)

    @staticmethod
    def _find(sk, k):
        if not len(sk):
            return np.full(len(k), -1, np.int64)
        i = np.minimum(np.searchsorted(sk, k), len(sk) - 1)
        return np.where((k >= 0) & (sk[i] == k), i, -1)

    def _rows(self, X):
        X = np.asarray(X, float); g = (X - self.org) / H0
        gr = np.round(g); g = np.where(np.abs(g - gr) < 1e-6, gr, g)       # a lattice node is a node (float error put 0.7 % of them
        b0 = np.floor(g / 8.0).astype(np.int64); loc0 = g - 8.0 * b0     # in the brick below, outside every row: run 1 of the check)
        on = loc0 < 1e-6                                              # on a brick's low face: the brick below holds it too (at 8)
        row = np.full(len(X), -1, np.int64); loc = loc0.copy()
        for s in range(8):                                            # the brick itself first, then the ones below along face axes
            sh = np.array([(s >> 2) & 1, (s >> 1) & 1, s & 1], np.int64)
            need = (row < 0) & np.all(on | (sh[None] == 0), axis=1)
            if not need.any():
                continue
            r = self._find(self.rk, self._key(b0[need] - sh[None])); got = r >= 0
            idx = np.nonzero(need)[0][got]; row[idx] = self.ri[r[got]]; loc[idx] = loc0[idx] + 8.0 * sh[None]
        return X, row, loc, b0

    def _tri(self, row, loc):
        c = np.clip(np.floor(loc).astype(np.int64), 0, 7); t = loc - c; acc = np.zeros(len(loc))
        for dx in (0, 1):                                             # the 8 corners gathered directly (run 1 copied each point's
            for dy in (0, 1):                                         # whole 9^3 row: 2.9 GB per million points)
                for dz in (0, 1):
                    w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                    acc += w * self.vals[row, c[:, 0] + dx, c[:, 1] + dy, c[:, 2] + dz]
        return acc

    def _far(self, X):
        return self.tree.query(X, workers=-1)[0] if self.tree is not None else np.full(len(X), 10 * self.cap)

    def __call__(self, X):
        """L0: the rows themselves; beyond them a distance lower bound (not clipped at the cap: a coarse level keeps only the bricks
        near the surface; run 1 clipped it at 3 mm and every L2 / L3 brick of the box passed the band test: 150 s a part)"""
        X, row, loc, b0 = self._rows(X); out = np.full(len(X), np.nan); hit = row >= 0
        if hit.any():
            out[hit] = self._tri(row[hit], loc[hit])
        rest = ~hit
        if rest.any():
            ins = self._find(self.ik, self._key(b0[rest])) >= 0
            out[rest] = np.where(ins, -1.0, 1.0) * np.maximum(self._far(X[rest]) - self.MARGIN, 0.0)
        return out

    def coarse(self, X):
        """L1-L3: a SIGNED DISTANCE everywhere (the sign from the row where there is one, else inside / outside by brick; the
        magnitude from the zero points). The reader takes a coarser level wherever an L0 sample is capped: interpolating capped rows
        there flipped signs 3 mm outside the upper lip (s3 F2a 19:20, 754 nodes: a phantom surface); a distance cannot"""
        X, row, loc, b0 = self._rows(X); sg = np.ones(len(X)); hit = row >= 0
        if hit.any():
            sg[hit] = np.where(self._tri(row[hit], loc[hit]) < 0, -1.0, 1.0)
        rest = ~hit
        if rest.any():
            sg[rest] = np.where(self._find(self.ik, self._key(b0[rest])) >= 0, -1.0, 1.0)
        return sg * self._far(X)


def _key_of(field_dir):
    st = [fv8base.source_id(field_dir)]
    for f in ('ijk', 'pid', 'sdf', 'inside'):
        p = os.path.join(field_dir, 'parts', 'L0_band_%s.npy' % f); s_ = os.stat(p); st.append('%s:%d:%d' % (f, s_.st_size, int(s_.st_mtime)))
    return {'cache_version': CACHE_VERSION, 'source': os.path.abspath(field_dir), 'stamp': st}


def _part_store(F8, S, B, pid, cap, out_dir):
    """part pid as its own small fv8 store (the renderer's format; build_from_fn over its own tile-aligned box, L3..L0)"""
    sel = B['pid'] == pid; ijk = B['ijk'][sel].astype(np.int64); vals = B['sdf'][sel]
    inside = B['inside'][B['inside'][:, 3] == pid][:, :3].astype(np.int64)
    fn = RowSampler(S.org, ijk, vals, inside, cap)
    g0, g1 = parts_box(B, pid, margin_bricks=4); A = 128                       # nodes; 128 = one 32 mm tile, aligned for L0-L3
    g0 = (np.floor(g0 / A) * A).astype(np.int64); g1 = (np.ceil(g1 / A) * A).astype(np.int64)
    import shutil
    lo, hi = S.org + g0 * H0, S.org + g1 * H0; tmp = out_dir + '_L0'
    # L3 keeps EVERY brick of the part's box (the shared inside rule, 11-11 19:11: the coarsest level is solid; a palm-thick part's
    # centre lies beyond L3's 8 mm band and read as outside); L2 / L1 keep their bands
    _S2, rep = F8.build_from_fn(fn.coarse, lo, hi, out_dir, levels=(3, 2, 1), band={3: 1.0, 2: 0.004, 1: 0.002})
    _S3, rep0 = F8.build_from_fn(fn, lo, hi, tmp, levels=(0,), band={0: cap}, cap={0: cap})      # every row is an L0 brick
    import gc
    del _S2, _S3; gc.collect()                                                  # their maps hold the files (Windows: no move while open)
    for f in os.listdir(tmp):
        if f.startswith('L0_'):
            os.replace(os.path.join(tmp, f), os.path.join(out_dir, f))
    tg = np.load(os.path.join(out_dir, 'tile_grid.npy')); tg[..., 0] = np.load(os.path.join(tmp, 'tile_grid.npy'))[..., 0]
    np.save(os.path.join(out_dir, 'tile_grid.npy'), tg)
    m = json.load(open(os.path.join(out_dir, 'meta.json'))); m['levels']['L0'] = json.load(open(os.path.join(tmp, 'meta.json')))['levels']['L0']
    m['_'] = 'fs-contact: part %d of %s as its own store (fv8parts RowSampler: L0 its rows, L1-L3 its signed distance)' % (pid, S.d)
    json.dump(m, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=1); shutil.rmtree(tmp)
    rep.update(rep0); return rep, int(sel.sum())


def _raise_coarse(vd, oth, g0, E0, cap0):
    """in the committed store vd: every L1-L3 node inside one of the bricks `oth` (L0 brick ijk) takes part 0's own signed distance:
    the sign of E0 there (part 0's own field on the L0 lattice from node g0), the distance to E0's zero set, at most the level's cap. The level files are hard links to her source: each changed one is
    written as a NEW file (temp + os.replace), never in place"""
    G9 = np.stack(np.meshgrid(*[np.arange(9)] * 3, indexing='ij'), -1).reshape(-1, 3)
    K = np.int64(1 << 20)

    def key(b):
        return ((b[..., 0] + K // 2) * K + (b[..., 1] + K // 2)) * K + (b[..., 2] + K // 2)
    # only the other parts' bricks: outside them part 0 IS the union (its coarse levels included). Widening by one brick was tried
    # (s3 19:33): 209 sign disagreements instead of 94, all on the union's own coarse cavity (the mouth bag: L1-L3 hold a cavity
    # its L0 does not; 4,196 such nodes disagree in the union's own editors' view), so the edge only moved
    ok_ = np.unique(key(np.asarray(oth, np.int64))); rep = {}
    # part 0's own surface: the zero crossings of E0 along the lattice's edges (m from the box's node 0). A coarse node at the other
    # parts' bricks takes sign(E0) x the distance to it: max(union's coarse, E0) was tried (s3 19:26): next to the cut under the
    # lips E0 goes from -3 mm (capped) to +3 mm within a brick (not a distance), and the coarse interpolation crossed zero 3 mm
    # inside part 0 at 975 nodes (a hidden phantom cavity under the editors' rule)
    from scipy.spatial import cKDTree
    Zp = []
    for ax in range(3):
        a_ = np.moveaxis(E0, ax, 0); va = a_[:-1]; vb = a_[1:]; mz = (va < 0) != (vb < 0)
        if not mz.any():
            continue
        i_, j_, k_ = np.nonzero(mz); t_ = va[i_, j_, k_] / (va[i_, j_, k_] - vb[i_, j_, k_])
        loc = np.stack([i_ + t_, j_.astype(float), k_.astype(float)], 1)
        if ax == 1: loc = np.stack([j_.astype(float), i_ + t_, k_.astype(float)], 1)
        if ax == 2: loc = np.stack([j_.astype(float), k_.astype(float), i_ + t_], 1)
        Zp.append(loc)
    tree = cKDTree(np.concatenate(Zp) * H0) if Zp else None
    rep['part0_zero_points'] = int(tree.n) if tree is not None else 0
    meta = json.load(open(os.path.join(vd, 'meta.json')))
    for l in (1, 2, 3):
        if 'L%d' % l not in meta['levels']:
            continue
        p_s = os.path.join(vd, 'L%d_sdf.npy' % l); bijk = np.load(os.path.join(vd, 'L%d_brick_ijk.npy' % l)).astype(np.int64)
        st = 1 << l; lo_b = (g0 // (8 * st)) - 1; hi_b = ((g0 + np.array(E0.shape)) // (8 * st)) + 1
        cand = np.nonzero(np.all((bijk >= lo_b) & (bijk <= hi_b), 1))[0]
        if not len(cand):
            continue
        A = np.load(p_s); shp = A.shape; R = A.reshape(len(A), -1); ch = 0
        n = (bijk[cand][:, None, :] * 8 + G9[None]) * st                  # L0 node index of every node of the candidate bricks
        inb = np.zeros(n.shape[:2], bool)
        for s_ in range(8):                                               # the node's L0 brick, or the one below on a face
            sh = np.array([(s_ >> 2) & 1, (s_ >> 1) & 1, s_ & 1], np.int64)
            onf = np.all(((n % 8) == 0) | (sh == 0), axis=-1)
            inb |= np.isin(key(n // 8 - sh), ok_) & onf
        li = n - g0; inbox = np.all((li >= 0) & (li < np.array(E0.shape)), axis=-1); m = inb & inbox
        if m.any():
            bi, ni = np.nonzero(m); e = E0[li[bi, ni, 0], li[bi, ni, 1], li[bi, ni, 2]]
            capl = float(meta['levels']['L%d' % l].get('cap_m', cap0))
            d = tree.query(li[bi, ni].astype(np.float64) * H0, workers=-1)[0] if tree is not None else np.full(len(e), capl)
            new = (np.where(e < 0, -1.0, 1.0) * np.minimum(d, capl)).astype(np.float16); chg = new != R[cand[bi], ni]; ch = int(chg.sum())
            if ch:
                R[cand[bi[chg]], ni[chg]] = new[chg]; tmp = p_s + '.tmp.npy'; np.save(tmp, R.reshape(shp)); os.replace(tmp, p_s)
        rep['L%d' % l] = {'nodes_in_other_parts_bricks': int(m.sum()), 'raised': ch}
    return rep


def _body_store(FP, F8, S, B, out_dir):
    """part 0 (her body without the other parts' tissue) as a whole store: her store with part 0's own field written as targets at
    every node of every other part's brick (fv8_write: L0 exactly as given), committed to out_dir; L1-L3 = max(union's, part 0's band)
    there (_raise_coarse). The source's
    k files and leaves (the union's) are not kept: this store draws part 0 alone"""
    import fv8_write as FW                                                      # beside fv8_field (_fp put it on the path)
    g0, g1 = parts_box(B, None, margin_bricks=3); origin = S.org + g0 * H0; shape = tuple(int(v) for v in (g1 - g0 + 1))
    E0 = FP.band_field(S, S.d, origin, shape, pids=[0])[0]
    T = np.full(shape, np.nan, np.float32)
    oth = np.concatenate([B['ijk'][B['pid'] != 0], B['inside'][B['inside'][:, 3] != 0][:, :3]]).astype(np.int64)
    for b in oth:
        a = b * 8 - g0; s0 = np.maximum(a, 0); s1 = np.minimum(a + 9, np.array(shape))
        if np.all(s1 > s0):
            T[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]] = E0[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
    op = FW.target_op(origin, T, coarse=False, l0_rebuild=False, crop=True, note='fs-contact fv8parts: part 0 alone (Field Studio cache)')
    r = S.apply(op); S.commit(out_dir, note='Field Studio cache: part 0 of %s alone; NOT a version of her, never published' % S.d)
    del T
    # L1-L3: the union's own levels, except at the other parts' bricks: part 0's own signed distance there (_raise_coarse).
    # fv8_write's coarse rebuild was tried (s3 19:20): it left outside-signs inside part 0 at 492 capped nodes
    r['coarse'] = _raise_coarse(out_dir, oth, g0, E0, float(S.lv[0]['cap_m']))
    del E0
    gone = []
    for f in os.listdir(out_dir):
        if 'kslot' in f or 'kleaf' in f:
            os.remove(os.path.join(out_dir, f)); gone.append(f)                  # hard links in the cache folder only
    return {k: r.get(k) for k in ('target_nodes', 'surface_nodes_changed', 'write_s', 'coarse')}, gone


def build_cache(field_dir):
    """-> (cache dir, info), SPARSE (cache version 2): part_<pid>/ for every part != 0, its own small fv8 store (its band rows: RAM and
    card follow its band, not its box), and part0/ = her store with the other parts' tissue taken out (part 0's own field at their
    bricks). Reused while the source version and its bands are unchanged"""
    import shutil
    FP, F8 = _fp(field_dir); cd = _cache_dir(field_dir); key = _key_of(field_dir); cj = os.path.join(cd, 'cache.json')
    if os.path.exists(cj):
        try:
            info = json.load(open(cj))
            if info.get('key') == key and all(os.path.exists(os.path.join(d['file'], 'meta.json')) for d in list(info['parts'].values()) + [info['part0']]):
                info['parts'] = {int(k): v for k, v in info['parts'].items()}; info['reused'] = True
                return cd, info
        except Exception:
            pass
    if os.path.isdir(cd) and os.path.basename(cd).endswith('_fsparts'):
        shutil.rmtree(cd)                                                       # this module's own cache folder only
    os.makedirs(cd, exist_ok=True)
    S = F8.FieldStore(field_dir); cap = float(S.lv[0]['cap_m']); B = FP.read_bands(field_dir)
    graph = json.load(open(os.path.join(field_dir, 'parts.json'))); info = {'key': key, 'parts': {}, 'cap_m': cap}
    for pid in sorted(set(int(p) for p in B['pid']) | set(int(p) for p in B['inside'][:, 3])):
        if pid == 0:
            continue
        d = os.path.join(cd, 'part_%d' % pid); rep, nrows = _part_store(F8, S, B, pid, cap, d)
        gp = (graph.get('parts', {}).get(str(pid)) or {})
        info['parts'][pid] = {'file': d, 'name': gp.get('name', 'part %d' % pid), 'material': gp.get('material'), 'parent': gp.get('parent'), 'band_rows': nrows,
                              'bricks': {k: v['bricks'] for k, v in rep.items() if k.startswith('L')}, 'build_s': rep.get('build_s')}
    d0 = os.path.join(cd, 'part0'); r0, gone = _body_store(FP, F8, F8.FieldStore(field_dir), B, d0)
    info['part0'] = {'file': d0, 'write': r0, 'k_files_not_kept': len(gone)}
    info['pairs'] = _pairs(graph); info['graph_parts'] = graph.get('parts', {})
    json.dump(info, open(cj, 'w'), indent=1, default=str)
    return cd, info


def open_specs(field_dir, h=H0):
    """-> (her base node dict, [AddObject spec per part], her part record, info) for a parts version: every part an 'fv8store'
    node of its own sparse store (cache version 2)"""
    from . import ops
    cd, info = build_cache(field_dir); src = os.path.abspath(field_dir); pairs = info['pairs']
    her = ops.Node('fv8store', fv8base.params_for(info['part0']['file'])).to_dict()
    specs = []
    for pid, d in sorted(info['parts'].items()):
        specs.append({'name': '%s (part %d)' % (d['name'], pid), 'base': ops.Node('fv8store', fv8base.params_for(d['file'])).to_dict(),
                      'translate': fv8base.TRANSFORM['translate'], 'rotate': fv8base.TRANSFORM['rotate'], 'h': float(h),
                      'part': {'source': src, 'pid': int(pid), 'name': d['name'], 'pairs': dict(pairs.get(str(pid), {})),
                               'parent': d.get('parent', (info['graph_parts'].get(str(pid)) or {}).get('parent'))}})   # the builder's own nesting (fs-shell's parts tree)
    her_part = {'source': src, 'pid': 0, 'name': (info['graph_parts'].get('0') or {}).get('name', 'body'), 'pairs': dict(pairs.get('0', {}))}
    return her, specs, her_part, info


def part_objects(scene, source):
    src = os.path.abspath(source)
    return [o for o in scene.objects if getattr(o, 'part', None) and os.path.abspath(o.part.get('source', '')) == src]


def prepare_writeback(scene, her_oid, levels=(0, 1, 2)):
    """(under the scene lock; seconds) -> what write_parts needs: every part object's field on the store's L0 lattice over the
    parts' box (fv8_parts.rebuild_box), the part graph, and her own write-back plan (fv8writeback.prepare) with its tiles that
    meet the box dropped (her object outside the box is store + sculpt: the cut lattice lies inside the box)"""
    from ..tools.protected_merge import field_of
    from . import fv8writeback as WB
    her = scene.get(her_oid); src = os.path.abspath(her.part['source']); FP, F8 = _fp(src)
    S = F8.FieldStore(src); B = FP.read_bands(src); g0, g1 = parts_box(B, None)
    lo = S.org + g0 * H0; hi = S.org + g1 * H0; origin, shape = FP.rebuild_box(S, lo, hi, levels)
    prep = WB.prepare(scene, her_oid); tl = WB.T_NODES * prep['h']; box_lo = origin; box_hi = origin + (np.array(shape) - 1) * H0
    # her edits outside the box are her SCULPT only: the cut lattice in her base is the parts' (inside the box), so the tiles
    # WB.prepare adds for it change nothing outside (s3 F4 11-11: they were written anyway, 11.9 GB in the dense cloud)
    hs = float(her.sculpt.h); m = 0.004 + 2 * max(prep['h'], hs); Bk = 8 * hs; st = scene.store; sculpt_tiles = set()
    for bk, sl in her.sculpt.slots.items():
        if sl >= 0 and st.delta[sl].any():
            a0 = np.floor((np.asarray(bk, float) * Bk - m - prep['org']) / tl).astype(int); a1 = np.floor((np.asarray(bk, float) * Bk + Bk + m - prep['org']) / tl).astype(int)
            for i in range(a0[0], a1[0] + 1):
                for j in range(a0[1], a1[1] + 1):
                    for k in range(a0[2], a1[2] + 1):
                        sculpt_tiles.add((i, j, k))
    keep = []
    for t in sorted(sculpt_tiles):
        a = prep['org'] + np.asarray(t, float) * tl; b = a + tl
        if np.any(b < box_lo) or np.any(a > box_hi):
            keep.append(t)
    prep = dict(prep); prep['tiles'] = keep
    # her object reads part0/ (the cache); its write-back delta (object - that store) is her sculpt alone outside the box, applied to
    # her REAL source version (cache v2)
    prep['source'] = src; prep['source_id'] = fv8base.source_id(src)
    objs = {int(o.part['pid']): o for o in part_objects(scene, src)}; pk = scene.pack()
    ax = [origin[k] + np.arange(shape[k]) * H0 for k in range(3)]; X = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    # every part object's field through field_of (world in, its own transform back to the store frame); the base trees below are
    # read at the same store-frame nodes. The two paths agree to ~1e-12 m on an unedited part (s3 F4, 11-11), so a change under a
    # nanometre is not an edit (below). eval_points_local was tried and is not this quantity (5.3 M targets on an unedited scene).
    R = np.asarray(_rot(), float); W = X @ R.T + np.asarray(fv8base.TRANSFORM['translate'], float); del X
    ent = {pid: np.asarray(field_of(scene, pk, o, W), np.float32).reshape(shape) for pid, o in sorted(objs.items())}
    # the parts AS OPENED, through the same reader (their base trees fresh from the cache, no sculpt): the edited nodes are
    # exactly where ent differs from these (s3 F4 11-11: compared with fv8_field.grid_L0, a different reader of the same store,
    # an unedited write-back re-targeted 21,928 nodes)
    from . import ops
    from .fv8writeback import _eval_many
    her_base, specs, _hp, _info = open_specs(src)
    trees = {0: ops.Node.from_dict(her_base)}
    for sp in specs:
        trees[int(sp['part']['pid'])] = ops.Node.from_dict(sp['base'])
    Xl = np.ascontiguousarray(np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)); base = {}
    for pid in ent:
        code, prm, *_ = ops.compile_tree(trees[pid]); out = np.empty(len(Xl)); _eval_many(code, prm, Xl, out)
        base[pid] = out.astype(np.float32).reshape(shape)
    # entries = each part's BAND (the authority) + exactly what was changed (current - as opened, one reader): an unedited part writes
    # its band back bit for bit, whatever the two store readers do at nodes L0 does not hold
    bands = FP.band_field(S, src, origin, shape)
    for pid in list(ent):
        d = ent[pid] - base[pid]; bk = bands.get(pid, base[pid])
        ent[pid] = np.where(np.abs(d) < 1e-9, bk, bk + d).astype(np.float32)       # under a nanometre is not an edit
    base = {pid: bands.get(pid, base[pid]) for pid in ent}
    graph = json.load(open(os.path.join(src, 'parts.json')))
    return {'source': src, 'lo': lo, 'hi': hi, 'levels': tuple(levels), 'entries': ent, 'base': base, 'her_prep': prep, 'graph': graph,
            'objects': {str(k): o.name for k, o in objs.items()}}


def write_parts(pp, out_dir, note='', leaves=None):
    """(no lock; the rebuild) her outside-the-box edits through the normal write-back into <out_dir>_outside when there are any,
    then fv8_parts.rebuild from that (or her source) into out_dir. -> report"""
    from . import fv8writeback as WB
    src = pp['source']; FP, F8 = _fp(src); base_dir = src; rep0 = None
    if pp['her_prep']['tiles']:
        mid = os.path.abspath(out_dir) + '_outside'
        rep0 = WB.write(pp['her_prep'], mid, note=(note or 'Field Studio write-back') + ' (her edits outside the parts box)', check=True)
        base_dir = mid
    S = F8.FieldStore(base_dir); g = pp['graph']
    pairs = {k: (v.get('contact') if isinstance(v, dict) else v) for k, v in (g.get('pairs') or {}).items()}
    parts = {int(k): v for k, v in (g.get('parts') or {}).items()}
    if leaves is None:
        leaves = 1 if os.path.exists(os.path.join(src, 'L0_kleaf_index.npy')) else 0
    S2, rep = FP.rebuild(S, out_dir, pp['lo'], pp['hi'], entries=pp['entries'], parts=parts, pairs=pairs, levels=pp['levels'], leaves=leaves,
                         note=note or 'Field Studio write-back of her parts (fv8parts)', base_entries=pp.get('base') if base_dir == src else None)
    rep['objects'] = pp['objects']; rep['outside_box_writeback'] = rep0; rep['source_for_rebuild'] = base_dir
    return rep


def writeback(scene, her_oid, out_dir, note='', leaves=None, levels=(0, 1, 2)):
    """both steps at once (scripts and tests; the app runs prepare under its lock and write_parts in a thread)"""
    return write_parts(prepare_writeback(scene, her_oid, levels), out_dir, note=note, leaves=leaves)


def _rot():
    from . import ops
    return ops.rot_matrix(fv8base.TRANSFORM['rotate'])
