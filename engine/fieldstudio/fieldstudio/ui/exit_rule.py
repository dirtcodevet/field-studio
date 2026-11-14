"""The exit rule for Field Studio and its harnesses (fs-shell, 2026-11-13; ledger L-1142): STOP LAUNCHING, SYNC, THEN EXIT.

Why. A process that exits (or os._exit / TerminateProcess) while one of its threads still has card work queued or
running can reset the display driver; six resets on 11-13 across the lanes. Before this, Field Studio's own close joined
the engine thread for 5 s, never synchronised the card, and never stopped the hair threads.

How.
  1. stop launching: the launch guard refuses every new pixel pass (gpu/launch_guard.request_stop); the engine thread
     is told to stop and joined (it ends the job it is in: a stroke, a frame, a call), with no new job started;
  2. the threads that may use the card outside the engine (the hair generator, its shadow and its rest-after-comb,
     started through card_thread() below) are joined;
  3. sync: the device is synchronised, so every launch already queued has ended;
  4. then the caller exits (the window closes; a harness calls hard_exit()).
A thread still alive after its wait is named in the record (never killed: the sync still runs, and the process
exits after it). The record of the last close goes to ~/FieldStudio/last_exit.json (times and what was waited for).

Env: FS_EXIT_WAIT_S (120): the longest wait for the engine thread and for each card thread."""
import json
import os
import threading
import time

WAIT_S = float(os.environ.get('FS_EXIT_WAIT_S', '120'))
RECORD = os.path.join(os.path.expanduser('~'), 'FieldStudio', 'last_exit.json')

_threads = set()
_lock = threading.Lock()


def card_thread(target, name='card-work', args=(), kwargs=None, daemon=True):
    """start a thread that may launch card work; the exit rule joins it before the card is synchronised"""
    def run():
        try:
            target(*args, **(kwargs or {}))
        finally:
            with _lock:
                _threads.discard(th)
    th = threading.Thread(target=run, name=name, daemon=daemon)
    with _lock:
        _threads.add(th)
    th.start()
    return th


def live_card_threads():
    with _lock:
        return [t for t in _threads if t.is_alive()]


def _sync_card():
    """synchronise the device if cupy has been loaded in this process (never imports it otherwise)"""
    import sys
    cp = sys.modules.get('cupy')
    if cp is None:
        return 'no card in this process'
    try:
        t = time.perf_counter()
        cp.cuda.Device().synchronize()
        return round((time.perf_counter() - t) * 1000, 1)
    except Exception as e:
        return 'sync failed: %r' % (e,)


def stop_card_work(engine=None, wait_s=None, record=True):
    """1-3 of the exit rule -> the record (dict). engine: a ui.engine.Engine (or None). Safe to call twice."""
    wait_s = WAIT_S if wait_s is None else float(wait_s)
    rec = {'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': []}
    t0 = time.perf_counter()
    try:
        from ..gpu import launch_guard
        launch_guard.request_stop()
        rec['steps'].append(('guard_stop', 0.0))
    except Exception as e:
        rec['steps'].append(('guard_stop', 'failed: %r' % (e,)))
    if engine is not None:
        t = time.perf_counter()
        try:
            alive = engine.stop_thread(wait_s)
        except AttributeError:            # an older Engine without stop_thread: its own stop
            engine.stop()
            alive = engine.thread.is_alive()
        rec['steps'].append(('engine_thread', round((time.perf_counter() - t) * 1000, 1)))
        if alive:
            rec['engine_thread_still_alive'] = True
    for th in live_card_threads():
        t = time.perf_counter()
        th.join(wait_s)
        rec['steps'].append(('join %s' % th.name, round((time.perf_counter() - t) * 1000, 1)))
        if th.is_alive():
            rec.setdefault('still_alive', []).append(th.name)
    rec['steps'].append(('sync_ms', _sync_card()))
    rec['total_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    if record:
        try:
            os.makedirs(os.path.dirname(RECORD), exist_ok=True)
            with open(RECORD, 'w', encoding='utf-8') as f:
                json.dump(rec, f, indent=1, default=str)
        except Exception:
            pass
    return rec


def hard_exit(code=0, engine=None, wait_s=None):
    """for harnesses that end by os._exit: the exit rule first (stop launching, sync), then the exit.
    fs-brush's lessons (bench/test1.py, 11-12 / 11-13): the card's pools given back after the sync, then TerminateProcess
    on Windows (os._exit hung 8 min in a DLL shutdown hook holding the card lock, twice)"""
    import sys
    try:
        stop_card_work(engine, wait_s=wait_s)
        cp = sys.modules.get('cupy')
        if cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
    finally:
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        if sys.platform == 'win32':
            try:
                import ctypes
                k_ = ctypes.WinDLL('kernel32')
                k_.GetCurrentProcess.restype = ctypes.c_void_p
                k_.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
                k_.TerminateProcess(k_.GetCurrentProcess(), int(code))
            except Exception:
                pass
        os._exit(int(code))
