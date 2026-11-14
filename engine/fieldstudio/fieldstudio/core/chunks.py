"""Incremental sculpt saves (READY item 1, 11-08): the sculpt layer saved as chunk files with a content digest each.

Before: every save compressed the whole layer into sculpt.npz (154 MB after a 300-stroke session: 40-72 s, and the
explicit Save held the engine lock for all of it). Now a project folder holds sculpt/<name>.npz, one file per
(object, 16^3-brick chunk), and project.json's 'sculpt' entry lists each chunk's digest. A save compresses only the
chunks whose digest is new; an unchanged chunk is kept (same folder) or hard-linked / copied from the previous save
(another folder, e.g. an autosave written into a fresh temporary folder).

Projects with sculpt.npz (format 1 without the 'sculpt' entry) still load.
"""
import hashlib
import os
import shutil

import numpy as np

CHUNK = 8                  # bricks per chunk side (r48 first run: 16 dirtied most of the layer per stroke)
KIND = 'chunks/1'


def chunk_arrays(sc, copy_=True):
    """-> {name: (keys (n,3) int64, delta (n,8,8,8) f32, mask (n,8,8,8) f32)} of the non-zero bricks"""
    st = sc.store
    out = {}
    for o in sc.objects:
        items = [(b, s) for b, s in o.sculpt.slots.items() if s >= 0]
        if not items:
            continue
        keys = np.array([b for b, _ in items], np.int64).reshape(-1, 3)
        slots = np.array([s for _, s in items], np.int64)
        d = st.delta[slots]
        m = st.mask[slots]
        md = st.mdir[slots]
        rd = st.rid[slots]                       # fs-shell 11-11: region ids (design 4.8)
        nz = d.reshape(len(slots), -1).any(1) | m.reshape(len(slots), -1).any(1) | rd.reshape(len(slots), -1).any(1)
        keys, d, m, md, rd = keys[nz], d[nz], m[nz], md[nz], rd[nz]
        if not len(keys):
            continue
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        keys, d, m, md, rd = keys[order], d[order], m[order], md[order], rd[order]
        cid = keys // CHUNK
        u, inv = np.unique(cid, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        for j, c in enumerate(u.tolist()):
            sel = inv == j
            name = 'o%d_%d_%d_%d' % (o.id, c[0], c[1], c[2])
            e = (keys[sel], d[sel] if not copy_ else d[sel].copy(), m[sel] if not copy_ else m[sel].copy())
            if md[sel].any() or rd[sel].any():   # tagged mask paint: a 4th array (older chunks and digests unchanged)
                e = e + (md[sel].copy(),)
            if rd[sel].any():               # fs-shell 11-11: region ids, a 5th array only where any
                e = e + (rd[sel].copy(),)
            out[name] = e
    return out


def digest(entry):
    h = hashlib.blake2b(digest_size=16)
    for a in entry:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write(path, chunks, prev=None):
    """write <path>/sculpt/<name>.npz for every chunk; reuse a chunk file whose digest is unchanged (in path itself, or
    linked from prev = (folder, manifest)). -> (manifest {name: digest}, stats)"""
    d = os.path.join(path, 'sculpt')
    os.makedirs(d, exist_ok=True)
    old = {}
    mf = os.path.join(d, '_manifest.txt')
    if os.path.exists(mf):
        for line in open(mf, encoding='utf-8'):
            p = line.split()
            if len(p) == 2:
                old[p[0]] = p[1]
    pdir, pman = (prev if prev else (None, {}))
    manifest, written, kept, linked = {}, 0, 0, 0
    for name, entry in chunks.items():
        dg = digest(entry)
        manifest[name] = dg
        dst = os.path.join(d, name + '.npz')
        if old.get(name) == dg and os.path.exists(dst):
            kept += 1
            continue
        src = os.path.join(pdir, 'sculpt', name + '.npz') if pdir else None
        if src and pman.get(name) == dg and os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            tmp = dst + '.tmp'
            if os.path.exists(tmp):
                os.remove(tmp)
            try:
                os.link(src, tmp)
            except OSError:
                shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            linked += 1
            continue
        k, de, m = entry[:3]
        tmp = dst + '.tmp.npz'
        if len(entry) > 4:
            np.savez_compressed(tmp, keys=k, delta=de, mask=m, mdir=entry[3], rid=entry[4])
        elif len(entry) > 3:
            np.savez_compressed(tmp, keys=k, delta=de, mask=m, mdir=entry[3])
        elif m.any():
            np.savez_compressed(tmp, keys=k, delta=de, mask=m)
        else:
            np.savez_compressed(tmp, keys=k, delta=de)
        os.replace(tmp, dst)
        written += 1
    for f in os.listdir(d):
        if f.endswith('.npz') and f[:-4] not in manifest:
            os.remove(os.path.join(d, f))
    with open(mf + '.tmp', 'w', encoding='utf-8') as f:
        for n in sorted(manifest):
            f.write('%s %s\n' % (n, manifest[n]))
    os.replace(mf + '.tmp', mf)
    old_npz = os.path.join(path, 'sculpt.npz')
    if os.path.exists(old_npz):
        os.remove(old_npz)
    return manifest, {'chunks': len(manifest), 'written': written, 'kept': kept, 'linked': linked}


def _read_one(d, name):
    with np.load(os.path.join(d, name + '.npz')) as z:
        k, de = z['keys'], z['delta']
        m = z['mask'] if 'mask' in z.files else np.zeros_like(de)
        md = z['mdir'] if 'mdir' in z.files else None
        rd = z['rid'] if 'rid' in z.files else None
    if rd is not None:
        return name, (k, de, m, md if md is not None else np.zeros(de.shape + (3,), np.int8), rd)
    return name, ((k, de, m) if md is None else (k, de, m, md))


def read(path, manifest):
    """-> {'o<id>_bricks', 'o<id>_delta', 'o<id>_mask'} concatenated per object (the sculpt.npz layout). r48b: the
    files are read on 8 threads (zlib releases the interpreter lock; 1065 files read one by one took 16.3 s to open)"""
    from concurrent.futures import ThreadPoolExecutor
    per = {}
    d = os.path.join(path, 'sculpt')
    names = sorted(manifest)
    with ThreadPoolExecutor(8) as ex:
        got = dict(ex.map(lambda n: _read_one(d, n), names))
    for name in names:
        oid = int(name.split('_')[0][1:])
        per.setdefault(oid, []).append(got[name])
    out = {}
    for oid, parts in per.items():
        out['o%d_bricks' % oid] = np.concatenate([p[0] for p in parts])
        out['o%d_delta' % oid] = np.concatenate([p[1] for p in parts])
        out['o%d_mask' % oid] = np.concatenate([p[2] for p in parts])
        if any(len(p) > 3 for p in parts):
            out['o%d_mdir' % oid] = np.concatenate([p[3] if len(p) > 3 else np.zeros(p[2].shape + (3,), np.int8)
                                                    for p in parts])
        if any(len(p) > 4 for p in parts):     # fs-shell 11-11: region ids
            out['o%d_rid' % oid] = np.concatenate([p[4] if len(p) > 4 else np.zeros(p[1].shape, np.uint16)
                                                   for p in parts])
    return out


def manifest_of(path):
    """the manifest a saved folder holds (for linking from it), {} if none"""
    mf = os.path.join(path, 'sculpt', '_manifest.txt')
    out = {}
    if os.path.exists(mf):
        for line in open(mf, encoding='utf-8'):
            p = line.split()
            if len(p) == 2:
                out[p[0]] = p[1]
    return out


def state_hash(describe, chunks, object_ids):
    """r48b: Scene.state_hash computed from a snapshot's description and chunks (off the engine lock): the same bytes in
    the same order (objects in scene order, each object's non-zero bricks sorted by key)"""
    import json
    h = hashlib.sha256()
    h.update(json.dumps(describe, sort_keys=True, separators=(',', ':')).encode())
    per = {}
    for name, e in chunks.items():
        per.setdefault(int(name.split('_')[0][1:]), []).append(e)
    for oid in object_ids:
        parts = per.get(oid)
        if not parts:
            continue
        k = np.concatenate([p[0] for p in parts])
        de = np.concatenate([p[1] for p in parts])
        m = np.concatenate([p[2] for p in parts])
        md = np.concatenate([p[3] if len(p) > 3 else np.zeros(p[2].shape + (3,), np.int8) for p in parts])
        rd = (np.concatenate([p[4] if len(p) > 4 else np.zeros(p[1].shape, np.uint16) for p in parts])
              if any(len(p) > 4 for p in parts) else None)       # fs-shell 11-11: region ids, as Scene.state_hash
        order = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))
        for i in order.tolist():
            h.update(np.array([oid, k[i, 0], k[i, 1], k[i, 2]], np.int64).tobytes())
            h.update(np.ascontiguousarray(de[i]).tobytes())
            h.update(np.ascontiguousarray(m[i]).tobytes())
            if md[i].any():
                h.update(np.ascontiguousarray(md[i]).tobytes())
            if rd is not None and rd[i].any():
                h.update(b'rid' + np.ascontiguousarray(rd[i]).tobytes())
    return h.hexdigest()
