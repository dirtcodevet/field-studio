"""Baked fields (fs-shell, 2026-11-11; the sculpt-exact wall for Cut > Conform). Design: <project>.fsproj/baked/<sha>.npy
on save, and ~/FieldStudio/baked/ until the first save, so projects stay self-contained: a project moved or reopened
elsewhere finds its bakes, and autosave and Save As carry them.

A bake is a dense field on a lattice (the 'lattice' node's file: npz with data (nx, ny, nz) float32, lo (3,), h),
named by the sha of its bytes. New bakes go to BAKE_HOME; a save copies the bakes its objects use into
<project>/baked/; a 'lattice' node whose recorded file is gone is found by its sha in the open project's baked folder,
then BAKE_HOME (core/ops.py asks resolve() for any 'lattice' file that is not where it was recorded)."""
import hashlib
import os
import shutil
import tempfile

import numpy as np

BAKE_HOME = os.environ.get('FIELDSTUDIO_BAKE_HOME') or os.path.join(os.path.expanduser('~'), 'FieldStudio', 'baked')
PROJECT_DIRS = []                     # the open project's baked folder first (the window sets it on open / save)


def is_bake(path):
    """a bake: under BAKE_HOME or in a project's baked/ folder"""
    if not path:
        return False
    d = os.path.abspath(os.path.dirname(str(path)))
    return os.path.basename(d) == 'baked' or d == os.path.abspath(BAKE_HOME)


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:24]


def write(data, lo, h):
    """a new bake -> {'file': its path under BAKE_HOME, 'sha'}"""
    os.makedirs(BAKE_HOME, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix='.npz', dir=BAKE_HOME)
    os.close(fd)
    np.savez(tmp, data=np.asarray(data, np.float32), lo=np.asarray(lo, float), h=float(h))
    sha = _sha(tmp)
    dst = os.path.join(BAKE_HOME, '%s.npz' % sha)
    if os.path.exists(dst):
        os.remove(tmp)
    else:
        os.replace(tmp, dst)
    return {'file': dst, 'sha': sha}


def write_arrays(**arrays):
    """fs-shell 11-11 (sculpt layers, snapshots): named arrays stored by the sha of their own bytes (names, dtypes, shapes,
    data; a replay writes the same sha) -> {'file', 'sha'}"""
    import hashlib
    hs = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        hs.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    sha = hs.hexdigest()
    os.makedirs(BAKE_HOME, exist_ok=True)
    dst = os.path.join(BAKE_HOME, '%s.npz' % sha)
    if not os.path.exists(dst):
        fd, tmp = tempfile.mkstemp(suffix='.npz', dir=BAKE_HOME)
        os.close(fd)
        np.savez(tmp, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
        os.replace(tmp, dst)
    _ARR[sha] = {k: np.ascontiguousarray(v).copy() for k, v in arrays.items()}
    return {'file': dst, 'sha': sha}


_ARR = {}


def read_arrays(path, sha):
    """the arrays a write_arrays stored (found by sha as resolve finds bakes), cached"""
    if sha in _ARR:
        return _ARR[sha]
    p = resolve(path, sha)
    if not p:
        raise FileNotFoundError('stored arrays %s are missing (%s)' % (sha, path))
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    _ARR[sha] = d
    return d


def resolve(path, sha):
    """the recorded file if it is there, else <sha>.npz in the open project's baked folder, then BAKE_HOME"""
    if path and os.path.exists(path):
        return path
    if sha:
        for d in list(PROJECT_DIRS) + [BAKE_HOME]:
            c = os.path.join(d, '%s.npz' % sha)
            if os.path.exists(c):
                return c
    return path


def used(scene):
    """[(file, sha)] of the bakes the scene's objects use ('lattice' nodes whose file is a bake: is_bake)"""
    out = []
    for o in scene.objects:
        stack = [o.base]
        while stack:
            n = stack.pop()
            f = str(n.params.get('file') or '') if n.kind == 'lattice' else ''
            if is_bake(f):
                out.append((f, n.params.get('sha')))
            stack.extend(n.children)
        for rec in list(getattr(o, 'layers', None) or []) + list(getattr(o, 'snapshots', None) or []):   # fs-shell 11-11
            if rec.get('sha') and is_bake(str(rec.get('file') or '')):
                out.append((rec.get('file'), rec.get('sha')))
    return out


def carry(scene, project_dir):
    """copy the scene's bakes into <project_dir>/baked/ (Save, Save As, autosave) -> the number copied"""
    dest = os.path.join(project_dir, 'baked')
    n = 0
    for f, sha in used(scene):
        src = resolve(f, sha)
        if not src or not os.path.exists(src):
            continue
        os.makedirs(dest, exist_ok=True)
        dst = os.path.join(dest, '%s.npz' % sha)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            n += 1
    return n


def set_project(project_dir):
    """the open project's baked folder goes first in the search"""
    PROJECT_DIRS[:] = [os.path.join(project_dir, 'baked')] if project_dir else []


def used_in_snapshot(snap):
    """[(file, sha)] of the bakes in a window snapshot's object trees (dicts), for a save posted from the UI thread"""
    out = []
    for o in (snap or {}).get('objects', []):
        stack = [o.get('base') or {}]
        while stack:
            n = stack.pop()
            f = str((n.get('params') or {}).get('file') or '') if n.get('kind') == 'lattice' else ''
            if is_bake(f):
                out.append((f, (n.get('params') or {}).get('sha')))
            stack.extend(n.get('children') or [])
        for rec in list(o.get('layers') or []) + list(o.get('snapshots') or []):   # fs-shell 11-11
            if rec.get('sha') and is_bake(str(rec.get('file') or '')):
                out.append((rec.get('file'), rec.get('sha')))
    return out


def carry_files(pairs, project_dir):
    """copy the given bakes into <project_dir>/baked/ (before a save: its folder swap carries them over) -> copied"""
    dest = os.path.join(project_dir, 'baked')
    n = 0
    for f, sha in pairs:
        src = resolve(f, sha)
        if not src or not os.path.exists(src):
            continue
        dst = os.path.join(dest, '%s.npz' % sha)
        if os.path.abspath(src) == os.path.abspath(dst) or os.path.exists(dst):
            continue
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n
