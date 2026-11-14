"""The undo journal (fs-trust, 2026-11-11): each undo step's own records on disk, so Open never replays the log and a
long session's undo history does not have to live in RAM.

Why (M, the 11-10 stab audit on a test project): Open rebuilt the undo history by replaying the whole op log into a second
scene on a worker thread: 121 s after 4 rounds, 397 s after 8, RAM +4.7 GB while it ran (a second copy of the sculpt
layer), and the rebuilt history was dropped whenever the user edited before it finished. The undo data the session already
held (each stroke's StrokeRecords: the bricks' contents before it, the bricks it created) is exactly what Open needs.

- A command that can describe its undo state (Command.undo_state() -> (meta dict, arrays dict), history.py) gets a uid
  at push. Once it is final (the next command is pushed) its state is written to <session>/undo/<uid>.npz by the one
  spill worker (gpu_results.submit). Old steps past the RAM budget drop their records from RAM (History._cap) and read
  them back from the file when undone.
- save_into(project, uids) puts the files of the project's undoable steps in <project>/undo/ (hard links when the
  volume allows; written when not spilled yet); project.json's 'undo' lists the uids of the done steps, oldest first.
- load_from(project) registers the files; history.load_project rebuilds the done stack as commands made from their log
  entries whose records are read only when undone. Nothing is replayed; the scene is the saved arrays (hash-checked).
Bit-exact: the records are the arrays themselves (float32 / int8 / float64), written with np.lib.format, deflate 1.
"""
import json
import os
import threading
import uuid
import zipfile

import numpy as np

from . import gpu_results as GR

_lock = threading.Lock()
_paths = {}          # uid -> [file paths], newest first
_dir = [None]
RAM_BUDGET = int(float(os.environ.get('FIELDSTUDIO_UNDO_RAM_MB', '512')) * 2 ** 20)
DISK_CAP = int(float(os.environ.get('FIELDSTUDIO_UNDO_DISK_GB', '4')) * 2 ** 30)
SAVE_QUOTA = int(float(os.environ.get('FIELDSTUDIO_UNDO_SAVE_GB', '2')) * 2 ** 30)   # undo files a saved project carries
KEEP_RESIDENT = 2    # the newest steps keep their records in RAM whatever the budget (live merges, idle passes)


def new_uid():
    return 'u' + uuid.uuid4().hex[:15]


def _session_dir():
    if _dir[0] is None:
        d = os.path.join(GR._spill_dir(), 'undo')
        os.makedirs(d, exist_ok=True)
        _dir[0] = d
    return _dir[0]


def _add(uid, p):
    with _lock:
        L = _paths.setdefault(uid, [])
        if p in L:
            L.remove(p)
        L.insert(0, p)


def path_of(uid):
    with _lock:
        ps = list(_paths.get(uid, ()))
    for p in ps:
        if os.path.exists(p):
            return p
    return None


def forget(uid):
    with _lock:
        _paths.pop(uid, None)


# ---- records <-> arrays ----------------------------------------------------------------------------------------
def _keys4(keys):
    return np.array([(int(o), int(b[0]), int(b[1]), int(b[2])) for o, b in keys], np.int64).reshape(-1, 4)


def _unkeys4(a):
    return [(int(r[0]), (int(r[1]), int(r[2]), int(r[3]))) for r in np.asarray(a).reshape(-1, 4).tolist()]


def record_arrays(pfx, rec, out):
    """a brush.StrokeRecord -> arrays named pfx_* in out"""
    sk = list(rec.snap.keys())
    out[pfx + '_snap_k'] = _keys4(sk)
    if sk:
        out[pfx + '_snap_d'] = np.stack([rec.snap[k][0] for k in sk]).astype(np.float32, copy=False)
        out[pfx + '_snap_m'] = np.stack([rec.snap[k][1] for k in sk]).astype(np.float32, copy=False)
        out[pfx + '_snap_L'] = np.array([rec.snap[k][2] for k in sk], np.float64)
    out[pfx + '_new_k'] = _keys4(rec.new)
    dk = list(getattr(rec, 'dsnap', {}).keys())
    out[pfx + '_dsnap_k'] = _keys4(dk)
    if dk:
        out[pfx + '_dsnap_v'] = np.stack([rec.dsnap[k] for k in dk]).astype(np.int8, copy=False)
    rk = list(getattr(rec, 'rsnap', {}).keys())             # fs-brush 11-11 (design 4.8): region ids, only when any
    if rk:
        out[pfx + '_rsnap_k'] = _keys4(rk)
        out[pfx + '_rsnap_v'] = np.stack([rec.rsnap[k] for k in rk]).astype(np.uint16, copy=False)
    lk = list(rec.Lsnap.keys())
    out[pfx + '_Lsnap_k'] = _keys4(lk)
    out[pfx + '_Lsnap_v'] = np.array([rec.Lsnap[k] for k in lk], np.float64)
    return out


def record_from(pfx, A):
    from .brush import StrokeRecord
    r = StrokeRecord()
    sk = _unkeys4(A[pfx + '_snap_k'])
    if sk:
        d, m, L = A[pfx + '_snap_d'], A[pfx + '_snap_m'], A[pfx + '_snap_L']
        for i, k in enumerate(sk):
            r.snap[k] = (np.array(d[i]), np.array(m[i]), float(L[i]))
    r.new = _unkeys4(A[pfx + '_new_k'])
    r.new_set = set(r.new)
    dk = _unkeys4(A[pfx + '_dsnap_k'])
    if dk:
        v = A[pfx + '_dsnap_v']
        for i, k in enumerate(dk):
            r.dsnap[k] = np.array(v[i])
    if (pfx + '_rsnap_k') in A:                                # fs-brush 11-11 (design 4.8)
        rv = A[pfx + '_rsnap_v']
        for i, k in enumerate(_unkeys4(A[pfx + '_rsnap_k'])):
            r.rsnap[k] = np.array(rv[i])
    lk = _unkeys4(A[pfx + '_Lsnap_k'])
    lv = A[pfx + '_Lsnap_v']
    for i, k in enumerate(lk):
        r.Lsnap[k] = float(lv[i])
    return r


def record_keys(rec):
    """every (oid, brick) a record restores (snap, dsnap, Lsnap): the bricks an undo writes into"""
    return set(rec.snap) | set(getattr(rec, 'dsnap', {})) | set(getattr(rec, 'rsnap', {})) | set(rec.Lsnap)


# ---- files -------------------------------------------------------------------------------------------------------
def write_file(p, meta, arrays):
    tmp = p + '.tmp.npz'
    allm = dict(arrays)
    allm['_meta'] = np.frombuffer(json.dumps(meta, sort_keys=True).encode('utf-8'), np.uint8)
    GR.savez_fast(tmp, allm)
    os.replace(tmp, p)
    return os.path.getsize(p)


def read_file(p):
    with np.load(p) as z:
        meta = json.loads(bytes(z['_meta']).decode('utf-8'))
        arrays = {k: np.asarray(z[k]) for k in z.files if k != '_meta'}
    return meta, arrays


def spill(uid, meta, arrays, on_done=None):
    """queue the write of uid's state to the session folder (the one spill worker); on_done(path, size) after"""
    def job():
        p = os.path.join(_session_dir(), uid + '.npz')
        n = write_file(p, meta, arrays)
        _add(uid, p)
        if on_done is not None:
            on_done(p, n)
    GR.submit(job)


def flush(timeout=None):
    return GR.flush(timeout)


def save_into(path, uids, prev=None, pending=None):
    """<path>/undo/<uid>.npz for each uid (linked from the session / the previous folder, else written from
    pending[uid] = (meta, arrays)); other files there are removed. -> files kept"""
    d = os.path.join(path, 'undo')
    want = [u for u in dict.fromkeys(uids) if u]
    if not want:
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
            os.rmdir(d)
        return 0
    os.makedirs(d, exist_ok=True)
    n = 0
    for uid in want:
        dst = os.path.join(d, uid + '.npz')
        if os.path.exists(dst):
            n += 1
            continue
        srcs = []
        if prev:
            srcs.append(os.path.join(prev, 'undo', uid + '.npz'))
        with _lock:
            srcs += list(_paths.get(uid, ()))
        for s in srcs:
            if s and os.path.exists(s):
                GR._link_or_copy(s, dst)
                break
        if not os.path.exists(dst) and pending and uid in pending:
            write_file(dst, *pending[uid])
        if os.path.exists(dst):
            n += 1
    keep = {u + '.npz' for u in want}
    for f in os.listdir(d):
        if f not in keep:
            os.remove(os.path.join(d, f))
    return n


def load_from(path):
    d = os.path.join(path, 'undo')
    n = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.npz') and '.tmp' not in f:
                _add(f[:-4], os.path.join(d, f))
                n += 1
    return n


def repoint(path):
    """files saved in <path>/undo are read from there from now on (a save swapped the folder)"""
    load_from(path)
