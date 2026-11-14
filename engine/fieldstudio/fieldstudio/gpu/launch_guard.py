"""Field Studio's card launch guard (fs-shell, 2026-11-13): no single card launch long enough to reset the display driver.

Why. Windows resets the display driver when one GPU job runs past ~2 s (TDR). Field Studio's renderer launched each
pass over a whole rectangle in one go, and the converged still (up to 3840 px, 8 passes a call) far longer. Six resets
on 11-13 across the lanes (ledger L-1031 / L-1142).

What. A pixel kernel's work over a rectangle is cut into tiles predicted to take TARGET_MS (100 ms) each. Every tile
is timed on the card (events); at most INFLIGHT (2) are queued, so a surprise is seen one launch later at most.
The prediction is deliberately coarse and conservative: every pixel is assumed as costly as the COSTLIEST tile (ms a
pixel) measured in this call or the kind's previous call. A per-pixel cost map was tried first and dropped before use:
a map from the last frame says "cheap" wherever there was sky, and the first frame after her field appears there
would have launched tiles sized for sky over her (~100x costlier a pixel).
  - A kind (call x light kernel) never measured, or the scene's volume set changed since its last call (an object or
    her field added, removed, shown or hidden): the call MEASURES: its first tile is one wave (2 blocks a
    multiprocessor), each next tile at most 2x the last and at most 16 waves, one in flight.
  - Always: no tile larger than CAP_MS at the worst cost a pixel this kind has EVER shown on this machine (kept across
    sessions in ~/FieldStudio/launch_guard_v3.json). Only what the CARD took enters that store: a launch whose host
    issue filled more than HOST_SHARE of its measured span (a lazy module load, the GIL, a busy CPU) is not kept (11-14;
    the old launch_guard.json kept such stalls as card time and is no longer read), from tiles of STORE_WAVES (4) waves
    or more (a smaller tile is mostly the fixed cost a launch). And never below WORST_FLOOR
    (2e-4 ms a px a pass, ~8x her head): a store with no record of costly content (a fresh machine) still bounds
    every tile.

Stop. A launch past ABORT_MS (1000 ms, half the watchdog) STOPS the call: the card is synchronised, nothing more is
launched, GuardStop goes to the caller (the viewport keeps its pending pixels; the next frame plans from what this one
measured). Nothing is ever relaunched larger, and there is no whole-frame fallback (fs-store-7's lesson, 11-13 18:38:
a fallback answered a tiler's exception with whole-frame launches and reset the driver). request_stop() (the app's
exit) makes every later call raise GuardStop before launching anything.

Pixels never change: every tile keeps each pixel's true index (the kernels take x0, y0, w, h) and each pixel runs its
passes in order, so the picture is the one a single launch gives (the n3 sky cells are shared between pixels and fill
in launch order, as they did between passes).

Env: FS_LAUNCH_MS (100), FS_LAUNCH_CAP_MS (250), FS_LAUNCH_ABORT_MS (1000), FS_LAUNCH_WARN_MS (400),
FS_LAUNCH_INFLIGHT (2), FS_LAUNCH_GUARD=0: one launch per call as before (A/B only; still timed, still stopped)."""
import json
import os
import threading
import time
from collections import deque

TARGET_MS = float(os.environ.get('FS_LAUNCH_MS', '100'))
CAP_MS = float(os.environ.get('FS_LAUNCH_CAP_MS', '250'))
ABORT_MS = float(os.environ.get('FS_LAUNCH_ABORT_MS', '1000'))
WARN_MS = float(os.environ.get('FS_LAUNCH_WARN_MS', '400'))
INFLIGHT = max(1, int(os.environ.get('FS_LAUNCH_INFLIGHT', '2')))
ON = os.environ.get('FS_LAUNCH_GUARD', '1') != '0'
FLOOR_KEEP = 0.8              # a kind's floor (ms a px) carried to its next call: max(this call's worst, 0.8 x the last)
WORST_FLOOR = float(os.environ.get('FS_LAUNCH_WORST_FLOOR', '2e-4'))  # fs-shell-4 11-14: the cap plans every pixel at
#                                least this costly (ms a px a pass; her head measured over whole frames: 2-3e-5)
STORE_WAVES = int(os.environ.get('FS_LAUNCH_STORE_WAVES', '4'))     # fs-shell-4 11-14: the smallest tile (in waves) whose
#                                cost a pixel may enter the store (fixed per-launch time amortised)
HOST_SHARE = float(os.environ.get('FS_LAUNCH_HOST_SHARE', '0.25'))   # fs-shell-4 11-14: a launch whose host issue took
#                                more than this share of its measured span is not the card's cost (not kept)
# fs-shell-4 11-14: a new file for the costs the card took (the old launch_guard.json holds host stalls as card time,
# 2,856x / 225x the kernels' cost; sessions still running the old guard keep writing it, so this one never reads it)
STORE = os.path.join(os.path.expanduser('~'), 'FieldStudio', 'launch_guard_v3.json')   # v3: + STORE_WAVES (v2 kept a
#                                measuring tile's fixed cost a launch as cost a pixel)


class GuardStop(RuntimeError):
    """a call stopped by the launch guard (a launch past ABORT_MS, or the app is closing): nothing more was launched"""


_STOP = threading.Event()     # process-wide: the app's exit sets it (every guard stops launching)


def request_stop():
    _STOP.set()


def stop_requested():
    return _STOP.is_set()


def watch_stop_file(path, every_s=0.2):
    """stop launching when `path` appears (tests/run_tests.py's time limit: the exit rule from outside the process)"""
    def loop():
        while not _STOP.is_set():
            if os.path.exists(path):
                print('fieldstudio: the stop file %s appeared: no more card launches' % path, flush=True)
                _STOP.set()
                return
            time.sleep(every_s)
    threading.Thread(target=loop, name='launch-guard-stop-file', daemon=True).start()


if os.environ.get('FS_STOP_FILE'):
    watch_stop_file(os.environ['FS_STOP_FILE'])


def _load_store():
    try:
        with open(STORE, encoding='utf-8') as f:
            d = json.load(f)
        return {str(k): float(v) for k, v in d.get('worst_ms_px', {}).items()}
    except Exception:
        return {}


class LaunchGuard:
    """one per GpuScene (one CUDA context, the null stream); run() is called from the thread that owns the scene"""

    def __init__(self, cp):
        self.cp = cp
        try:
            sms = int(cp.cuda.Device().attributes['MultiProcessorCount'])
        except Exception:
            sms = 20
        self.wave_px = 256 * sms                      # 2 blocks of 16 x 8 a multiprocessor (5,120 px on the RTX 5050)
        self.kinds = {}                               # kind -> {'gen', 'cpp' (floor, ms / px / unit)}
        self.ever = _load_store()                     # kind -> the worst ms / px / unit ever measured (all sessions)
        self._ever_saved = dict(self.ever)
        self.log = deque(maxlen=2000)                 # (t, kind, x0, y0, w, h, units, ms) of recent launches
        self.stats = {'calls': 0, 'launches': 0, 'max_ms': 0.0, 'warn': 0, 'stops': 0, 'measured_calls': 0}
        self.last = {}

    def _tile(self, x, y, x0, y0, w, h, px):
        """the next tile from (x, y) in row order: whole-width bands of 8-row multiples, or 8-row stripes cut in x"""
        if x == x0 and px >= 8 * w:
            r = min(y0 + h - y, max(8, (px // w) // 8 * 8))
            return (x0, y, w, r)
        r = min(8, y0 + h - y)
        cw = max(16, (px // r) // 16 * 16)
        return (x, y, min(cw, x0 + w - x), r)

    def run(self, kind, W, H, rect, issue, units=1, gen=None):
        """issue(x0, y0, w, h) enqueues `units` launches for that sub-rect on the current stream (not waited for).
        gen: the scene's volume-set key (a change re-measures). Returns the call's stats. Raises GuardStop (nothing more
        launched) on a launch past ABORT_MS or after request_stop()."""
        if _STOP.is_set():
            raise GuardStop('launch guard: the app is closing; nothing launched')
        cp = self.cp
        x0, y0, w, h = (int(v) for v in rect)
        res = {'kind': kind, 'launches': 0, 'max_ms': 0.0, 'ms': 0.0, 'measured': False}
        if w <= 0 or h <= 0:
            return res
        self.stats['calls'] += 1
        st = self.kinds.get(kind)
        known = st is not None and st['gen'] == gen
        floor = st['cpp'] if known else 0.0
        worst = self.ever.get(kind, 0.0)
        cur = [0.0]                                    # the costliest tile of this call (ms / px / unit)
        inflight = []                                  # (rect, e0, e1)
        measuring = not known
        res['measured'] = measuring
        if measuring:
            self.stats['measured_calls'] += 1

        def settle():
            (tx, ty, tw, th), e0, e1, host_ms = inflight.pop(0)
            e1.synchronize()
            ms = cp.cuda.get_elapsed_time(e0, e1)
            # a tile under one wave still occupies the card like a wave: its time a pixel is counted over a wave (a
            # 16-px edge tile would otherwise read as a 40x costlier pixel and shrink every later tile)
            cpp = ms / (max(tw * th, self.wave_px) * units)
            cur[0] = max(cur[0], cpp)
            # fs-shell-4 11-14 (the time inventory: her still 1.0 -> 5.7 s, 7,935 launches under 2 ms): the events
            # bracket the HOST's issue as well as the kernel, so on an idle stream a stall between e0 and the launch
            # (CUDA's lazy module load on a kernel's first launch after a recompile, the GIL, the CPU taken by other
            # jobs) read as card time: the store held prog_p0:review at 2,856x and prog_pn:review at 225x what the
            # kernels take (t64/guard_ab.py), and every later tile of every session was cut to match. A measurement
            # whose span was mostly the host's still sizes this call (cur, conservative) but never enters the store
            # kept across sessions, which holds only what the card took.
            # and only a tile of STORE_WAVES waves or more: a smaller one's time is mostly the card's fixed cost a launch
            # (counted over a wave), so its ms a px read up to 30x the kernel's (prog_pn:review 7.8e-4 kept from the
            # measuring tiles vs 2-3e-5 measured over whole frames, t64/guard_ab.py); WORST_FLOOR bounds what is not kept
            if cpp > self.ever.get(kind, 0.0) and host_ms <= HOST_SHARE * ms and tw * th >= STORE_WAVES * self.wave_px:
                self.ever[kind] = cpp
            self.log.append((round(time.perf_counter(), 3), kind, tx, ty, tw, th, units, round(ms, 2)))
            res['launches'] += 1
            res['ms'] += ms
            res['max_ms'] = max(res['max_ms'], ms)
            self.stats['launches'] += 1
            self.stats['max_ms'] = max(self.stats['max_ms'], ms)
            if ms > WARN_MS:
                self.stats['warn'] += 1
                print('fieldstudio: LAUNCH GUARD a %s launch took %.0f ms (%d units over x %d y %d w %d h %d, %.5f ms a px '
                      'a unit)' % (kind, ms, units, tx, ty, tw, th, cpp), flush=True)
            if ABORT_MS > 0 and ms > ABORT_MS:
                del inflight[:]
                cp.cuda.Device().synchronize()           # what was already queued ends; nothing new is issued
                self.stats['stops'] += 1
                self.kinds.pop(kind, None)               # the next call of this kind measures again
                self.save()
                raise GuardStop('launch guard STOP: a %s launch took %.0f ms > %.0f ms (x %d y %d w %d h %d): the call '
                                'stops, nothing more is launched' % (kind, ms, ABORT_MS, tx, ty, tw, th))
            return tw * th

        last_px = 0
        y, x = y0, x0
        while y < y0 + h:
            if _STOP.is_set():
                while inflight:
                    settle()
                raise GuardStop('launch guard: the app is closing; the call stops')
            if not ON:
                t = (x0, y0, w, h)
            else:
                est = max(floor, cur[0])
                px = int(TARGET_MS / (est * units)) if est > 0 else 1 << 30
                # fs-shell-4 11-14 (a safety floor, t64/sky_to_her.py): the cap never plans below
                # WORST_FLOOR. With no record of her cost (a fresh machine, or the new store before she was seen), the
                # estimate after looking at sky is sky's, and a 3840 x 2160 still of her went out as ONE 8-pass launch
                # of 965 ms (the abort is 1000, the watchdog ~2 s). The floor bounds every tile of content up to that
                # cost a pixel to CAP_MS whatever the history holds
                px = min(px, int(CAP_MS / (max(worst, WORST_FLOOR) * units)))
                if measuring:
                    px = min(px, self.wave_px if last_px == 0 else 2 * last_px, 16 * self.wave_px)
                px = max(px, 128)
                t = self._tile(x, y, x0, y0, w, h, px)
            while len(inflight) >= (1 if measuring else INFLIGHT):
                settle()
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            th0 = time.perf_counter()
            e0.record()
            issue(*t)
            e1.record()
            inflight.append((t, e0, e1, (time.perf_counter() - th0) * 1000.0))
            last_px = t[2] * t[3]
            if measuring:
                settle()
                worst = self.ever.get(kind, 0.0)
            # advance
            if t[0] + t[2] >= x0 + w:
                x, y = x0, t[1] + t[3]
            else:
                x = t[0] + t[2]
            if not ON:
                break
        while inflight:
            settle()
        self.kinds[kind] = {'gen': gen, 'cpp': max(cur[0], FLOOR_KEEP * floor)}
        self.last = res
        if measuring:
            self.save()
        return res

    def save(self):
        """the worst costs ever measured, kept as the next session's cap"""
        if self.ever == self._ever_saved:
            return
        try:
            os.makedirs(os.path.dirname(STORE), exist_ok=True)
            d = {'note': 'Field Studio launch guard (gpu/launch_guard.py): the worst card time a pixel ever took, per kernel '
                         'kind (ms a px a pass); a cap on tile sizes only', 'worst_ms_px': self.ever}
            tmp = STORE + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(d, f, indent=1)
            os.replace(tmp, STORE)
            self._ever_saved = dict(self.ever)
        except Exception:
            pass
