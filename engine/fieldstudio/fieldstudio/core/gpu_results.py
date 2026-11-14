"""Results of GPU brush strokes, so the op log replays to the same state hash (option A of the 11-08 replay decision).

A stroke applied on the card (gpu/sculpt.py) leaves its result in the sculpt layer through its StrokeRecord; the CPU
brushes would reproduce it only to a few um, so the stroke's log entry names a result id instead, and replay applies
the recorded result exactly: per object, the changed sculpt bricks' keys and their delta after the stroke.

- put(rid, data) keeps the result in memory and spills it to a session folder on a worker thread (so a long session's
  results do not stay in RAM); get(rid) returns it from memory or the spill.
- save_into(project, rids) writes the results a log names into <project>/gpu_strokes.npz; load_from(project) reads
  them back (Open), so Verify replay and the undo history rebuilt on Open work without a GPU.
- r45: a result is stored as the bitwise XOR of each brick's delta after the stroke with its delta before it
  (encode(data, rec); uint32, exact; zero wherever a node did not change, so it compresses ~10x better than the
  delta itself: 2.0 MB a stroke in r44's 113-stroke run), and saved as <project>/gpu_strokes/<rid>.npz, one file per
  stroke written once: a save copies only the strokes it does not have yet (r44: 63 s to re-compress 227 MB on
  every save). Open registers the files and reads each only when a replay needs it.
- fs-trust 11-11 (reconciliation 1.7 / item 9: results sat outside every cap, 32 kept by count, a thread per stroke,
  eviction only inside a spill thread): ONE spill worker drains a queue; a result leaves RAM as soon as its file is
  written; the results waiting in RAM are counted in bytes (ram_bytes(), shown with the undo memory) and kept under
  RAM_BUDGET: a put() that would pass it writes its own file at once (back-pressure, never an unbounded backlog).
  Spills are deflate level 1 (2.6x faster than level 6 on XOR data, 9 % larger: M 11-11).
"""
import os
import queue
import threading
import uuid
import zipfile

import numpy as np

_mem = {}            # rid -> {oid: (keys (N, 3) int64, after (N, 8, 8, 8) float32)}: results not yet spilled
_spilled = {}        # rid -> spill file path
_lock = threading.Lock()
_dir = None
KEEP_IN_RAM = 0      # fs-trust 11-11: none kept once spilled (was 32 by count)
RAM_BUDGET = int(float(os.environ.get('FIELDSTUDIO_RESULTS_RAM_MB', '256')) * 2 ** 20)   # bytes waiting to spill
_pending = [0]       # bytes in _mem
_inflight = set()    # rids being written
_q = queue.Queue()
_worker = [None]
_done_cv = threading.Condition(_lock)


def _spill_dir():
    global _dir
    if _dir is None:
        env = os.environ.get('FIELDSTUDIO_GPU_RESULTS_DIR')
        root = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'FieldStudio', 'gpu_results')
        base = env or os.path.join(root, uuid.uuid4().hex[:12])
        os.makedirs(base, exist_ok=True)
        _dir = base
        if not env:
            # fs-trust 11-11: a session's spill folder (results, undo journal, checkpoints) is its own scratch: removed at
            # exit, and folders of sessions whose process has ended are removed at the next start. A saved project
            # keeps its own copies (hard links or copies), so nothing saved depends on these folders.
            try:
                with open(os.path.join(base, 'owner.pid'), 'w') as f:
                    f.write(str(os.getpid()))
                import atexit
                import shutil
                atexit.register(shutil.rmtree, base, True)
                _sweep_dead(root, base)
            except Exception:
                pass
    return _dir


def _pid_alive(pid):
    import ctypes
    k32 = ctypes.WinDLL('kernel32')
    h = k32.OpenProcess(0x1000, False, int(pid))         # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        return bool(ok) and code.value == 259             # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def _sweep_dead(root, mine):
    import shutil
    import time as _t
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if p == mine or not os.path.isdir(p):
            continue
        f = os.path.join(p, 'owner.pid')
        try:
            if os.path.exists(f):
                dead = not _pid_alive(int(open(f).read().strip() or 0))
            else:
                dead = _t.time() - os.path.getmtime(p) > 24 * 3600     # folders from before 11-11: a day old
            if dead:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def new_id():
    return uuid.uuid4().hex[:16]


def _flat(data):
    out = {}
    for oid, (keys, after) in data.items():
        out['o%d_keys' % oid] = np.asarray(keys, np.int64)
        out['o%d_after' % oid] = after if after.dtype == np.uint32 else np.asarray(after, np.float32)   # r45
    return out


def _unflat(z):
    data = {}
    for k in z.files:
        if k.endswith('_keys'):
            oid = int(k[1:-5])
            data[oid] = (np.asarray(z[k]), np.asarray(z['o%d_after' % oid]))
    return data


def encode(data, rec):
    """r45: {oid: (keys, after float32)} -> {oid: (keys, after XOR before as uint32)}; before = the brick's delta at
    the stroke's start (rec.snap) or zero (a brick the stroke allocated, rec.new_set)"""
    out = {}
    for oid, (keys, after) in data.items():
        after = np.ascontiguousarray(after, np.float32)
        before = np.zeros_like(after)
        for i, k in enumerate(map(tuple, np.asarray(keys).reshape(-1, 3).tolist())):
            sn = rec.snap.get((oid, k))
            if sn is not None:
                before[i] = sn[0]
            elif (oid, k) not in rec.new_set:
                raise ValueError('GPU stroke result: brick %r of object %d has no snapshot' % (k, oid))
        out[oid] = (np.asarray(keys, np.int64), after.view(np.uint32) ^ before.view(np.uint32))
    return out


def _data_bytes(data):
    return int(sum(np.asarray(k).nbytes + np.asarray(a).nbytes for k, a in data.values()))


def savez_fast(path, arrays, level=1):
    """np.savez_compressed's format (np.load reads it) at deflate level `level`"""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        for k, a in arrays.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(a), allow_pickle=False)


def _spill_one(rid):
    """write rid's file (once) and drop it from RAM; safe from the worker and from put()'s back-pressure path"""
    with _lock:
        data = _mem.get(rid)
        if data is None or rid in _spilled or rid in _inflight:
            return
        _inflight.add(rid)
    try:
        p = os.path.join(_spill_dir(), rid + '.npz')
        tmp = p + '.tmp.npz'
        savez_fast(tmp, _flat(data))                 # r45: compressed once here; a save copies the file
        os.replace(tmp, p)
        with _lock:
            _spilled[rid] = p
            d = _mem.pop(rid, None)
            if d is not None:
                _pending[0] -= _data_bytes(d)
    finally:
        with _lock:
            _inflight.discard(rid)
            _done_cv.notify_all()


def _work():
    while True:
        job = _q.get()
        try:
            if callable(job):
                job()                                # fs-trust: the undo journal's writes share this one worker
            else:
                _spill_one(job)
        except Exception as e:                       # the data stays in RAM (get() still serves it); say why once
            print('gpu_results: spill of %s failed: %s' % (job, e), flush=True)
        finally:
            _q.task_done()


def submit(fn):
    """run fn() on the one spill worker (after the writes queued before it)"""
    _ensure_worker()
    _q.put(fn)


def _ensure_worker():
    with _lock:
        if _worker[0] is None or not _worker[0].is_alive():
            _worker[0] = threading.Thread(target=_work, name='gpu-results-spill', daemon=True)
            _worker[0].start()


def put(rid, data):
    nb = _data_bytes(data)
    with _lock:
        _mem[rid] = data
        _pending[0] += nb
        over = _pending[0] > RAM_BUDGET
    _ensure_worker()
    if over:
        _spill_one(rid)                              # back-pressure: this caller writes its own result now
    else:
        _q.put(rid)


def ram_bytes():
    """bytes of results held in RAM (waiting for, or being written to, their spill file)"""
    with _lock:
        return int(_pending[0])


def flush(timeout=None):
    """wait until every queued result is written (tests; a save copies files instead of re-writing) -> True when done"""
    import time as _t
    end = None if timeout is None else _t.perf_counter() + timeout
    while True:
        with _lock:
            busy = bool(_inflight) or _q.unfinished_tasks > 0
        if not busy:
            return True
        if end is not None and _t.perf_counter() > end:
            return False
        _t.sleep(0.005)


def get(rid):
    with _lock:
        d = _mem.get(rid)
        p = _spilled.get(rid)
    if d is not None:
        return d
    if p is None or not os.path.exists(p):
        return None
    with np.load(p) as z:
        return _unflat(z)


def nbytes(rid):
    with _lock:
        d = _mem.get(rid)
    if d is None:
        return 0
    return sum(k.nbytes + a.nbytes for k, a in d.values())


def save_into(path, rids, prev=None):
    """r45: <path>/gpu_strokes/<rid>.npz for each named result (copied from the spill or an opened project's file
    when there is one, else written); files the log no longer names are removed. -> results saved"""
    import shutil
    d = os.path.join(path, 'gpu_strokes')
    want = [r for r in dict.fromkeys(rids) if r]
    if not want and not os.path.isdir(d):
        _legacy_remove(path)
        return 0
    os.makedirs(d, exist_ok=True)
    n = 0
    for rid in want:
        dst = os.path.join(d, rid + '.npz')
        if os.path.exists(dst):
            n += 1
            continue
        with _lock:
            src = _spilled.get(rid)
        if prev and not (src and os.path.exists(src)):
            q = os.path.join(prev, 'gpu_strokes', rid + '.npz')
            src = q if os.path.exists(q) else src
        if src and os.path.exists(src) and src.endswith('.npz') and _is_compressed_file(src):
            _link_or_copy(src, dst)
            n += 1
            continue
        data = get(rid)
        if data is None:
            continue
        np.savez_compressed(dst + '.tmp.npz', **_flat(data))
        os.replace(dst + '.tmp.npz', dst)
        n += 1
    keep = {r + '.npz' for r in want}
    for f in os.listdir(d):
        if f not in keep:
            os.remove(os.path.join(d, f))
    _legacy_remove(path)
    return n


def _link_or_copy(src, dst):
    """r48: a hard link when the volume allows (a save no longer copies hundreds of MB of results)"""
    import shutil
    tmp = dst + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def repoint(path):
    """r48: results registered from a folder that a save swapped out now point into its replacement"""
    d = os.path.join(path, 'gpu_strokes')
    if not os.path.isdir(d):
        return
    with _lock:
        for f in os.listdir(d):
            if f.endswith('.npz') and '.tmp' not in f:
                rid = f[:-4]
                p = _spilled.get(rid)
                if p is None or not os.path.exists(p):
                    _spilled[rid] = os.path.join(d, f)


def ck_repoint(path):
    """fs-trust 11-11: checkpoints saved in `path` are read from there from now on; their RAM copies are dropped"""
    for f in os.listdir(path):
        if f.startswith('checkpoint_') and f.endswith('.npz') and not f.endswith('.tmp.npz'):
            cid = f[len('checkpoint_'):-4]
            _ck_add_path(cid, os.path.join(path, f))
            with _lock:
                _ck.pop(cid, None)


def _is_compressed_file(p):
    return True          # every spill since r45 is written compressed; an opened project's files are too


def _legacy_remove(path):
    p = os.path.join(path, 'gpu_strokes.npz')
    if os.path.exists(p):
        os.remove(p)


def _save_into_r42(path, rids):
    arrays = {}
    for rid in rids:
        d = get(rid)
        if d is None:
            continue
        for k, v in _flat(d).items():
            arrays['r%s_%s' % (rid, k)] = v
    p = os.path.join(path, 'gpu_strokes.npz')
    if arrays:
        tmp = p + '.tmp.npz'
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, p)
    elif os.path.exists(p):
        os.remove(p)
    return len(arrays) // 2


def load_from(path):
    """r45: register <path>/gpu_strokes/<rid>.npz (read when a replay needs one); r42's single gpu_strokes.npz is
    read whole"""
    n = 0
    d = os.path.join(path, 'gpu_strokes')
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.npz') and '.tmp' not in f:
                with _lock:
                    _spilled[f[:-4]] = os.path.join(d, f)
                n += 1
    p = os.path.join(path, 'gpu_strokes.npz')
    if not os.path.exists(p):
        return n
    with np.load(p) as z:
        by = {}
        for k in z.files:
            rid, rest = k[1:].split('_', 1)
            by.setdefault(rid, {})[rest] = np.asarray(z[k])
    for rid, parts in by.items():
        data = {}
        for k, v in parts.items():
            if k.endswith('_keys'):
                oid = int(k[1:-5])
                data[oid] = (v, parts['o%d_after' % oid])
        with _lock:
            if rid not in _mem:
                _pending[0] += _data_bytes(data)
            _mem[rid] = data
        n += 1
    return n


# ---- r44: checkpoints (consolidated history): cid -> {'scene': describe(), 'hash': state hash, 'arrays': {...}}
# fs-trust 11-11: a checkpoint's arrays (a whole scene's sculpt layer) stayed in RAM for the rest of the session; once
# its file exists (a save, or Open) only the path is kept and ck_get reads the file when a replay needs it
_ck = {}
_ck_path = {}


def ck_new_id():
    return 'ck' + uuid.uuid4().hex[:14]


def _ck_write(p, d):
    import json
    meta = json.dumps({'scene': d['scene'], 'hash': d['hash']}, sort_keys=True).encode('utf-8')
    tmp = p + '.tmp.npz'
    np.savez_compressed(tmp, _meta=np.frombuffer(meta, np.uint8), **d['arrays'])
    os.replace(tmp, p)


def _ck_add_path(cid, p):
    with _lock:
        L = _ck_path.setdefault(cid, [])
        if p in L:
            L.remove(p)
        L.insert(0, p)                       # newest first


def ck_put(cid, data):
    """fs-trust 11-11: written to the session's spill folder at once; RAM keeps it only if that write fails"""
    try:
        p = os.path.join(_spill_dir(), 'checkpoint_%s.npz' % cid)
        _ck_write(p, data)
        _ck_add_path(cid, p)
    except Exception:
        with _lock:
            _ck[cid] = data


def _ck_read(p):
    import json
    with np.load(p) as z:
        meta = json.loads(bytes(z['_meta']).decode('utf-8'))
        arrays = {k: np.asarray(z[k]) for k in z.files if k != '_meta'}
    return {'scene': meta['scene'], 'hash': meta['hash'], 'arrays': arrays}


def ck_get(cid):
    with _lock:
        d = _ck.get(cid)
        ps = list(_ck_path.get(cid, ()))
    if d is not None:
        return d
    for p in ps:
        if os.path.exists(p):
            return _ck_read(p)
    return None


def ck_forget(cid):
    with _lock:
        _ck.pop(cid, None)
        _ck_path.pop(cid, None)


def ck_ram_bytes():
    with _lock:
        return int(sum(a.nbytes for d in _ck.values() for a in d['arrays'].values()))


def ck_ids_in(log):
    return [e.get('cid') for e in log if isinstance(e, dict) and e.get('type') == 'Checkpoint']


def ck_save_into(path, cids, prev=None):
    import json
    keep = set()
    for cid in cids:
        p = os.path.join(path, 'checkpoint_%s.npz' % cid)
        if os.path.exists(p):
            keep.add(os.path.basename(p))
            continue                  # a checkpoint never changes once written
        q = os.path.join(prev, 'checkpoint_%s.npz' % cid) if prev else None
        with _lock:
            q2 = list(_ck_path.get(cid, ()))
        for src in [q] + q2:
            if src and os.path.exists(src):
                _link_or_copy(src, p)   # r48
                keep.add(os.path.basename(p))
                break
        if os.path.exists(p):
            continue
        with _lock:
            d = _ck.get(cid)
        if d is None:
            continue
        keep.add(os.path.basename(p))
        _ck_write(p, d)
    for f in os.listdir(path):
        if f.startswith('checkpoint_') and f.endswith('.npz') and f not in keep:
            os.remove(os.path.join(path, f))
    return len(keep)


def ck_load_from(path):
    """fs-trust 11-11: registers the checkpoint files (read by ck_get when a replay needs one), as load_from does
    for results; before, every checkpoint's arrays were read into RAM at Open"""
    n = 0
    for f in os.listdir(path):
        if f.startswith('checkpoint_') and f.endswith('.npz') and not f.endswith('.tmp.npz'):
            _ck_add_path(f[len('checkpoint_'):-4], os.path.join(path, f))
            n += 1
    return n


def rids_in(log):
    return [e.get('gpu_result') for e in log if isinstance(e, dict) and e.get('gpu_result')]
