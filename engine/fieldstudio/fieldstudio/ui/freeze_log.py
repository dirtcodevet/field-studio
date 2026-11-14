"""The freeze record (fs-shell-5, 2026-11-14; the first hands-on test: freezes were felt in a short session
with little done, yet its log held only the 3.4 s start-up
stall, because the watchdog in run_field_studio.py wrote only stalls over 2 s and every freeze felt was shorter).

A freeze a user can feel is a few hundred ms. Two kinds are recorded, each with what ran:
  - WINDOW: the window thread did not answer (its 50 ms heartbeat was late by more than FREEZE_MS): menus, panels, keys
    and the view all stop together;
  - VIEW: the user was working in the view (a stroke, an orbit, a pan, a transform drag) and no new frame came for more than
    FREEZE_MS after the view asked for one, while the window itself answered: the model stops following the hand.
What ran: a sampler thread (Python) wakes every SAMPLE_MS; while the heartbeat is late, or a frame is overdue, it takes
the window thread's and the engine thread's Python stacks (sys._current_frames). The record names the frames inside
Field Studio that the samples were in most often, the tool, what the view was doing and the engine's queued jobs. When
the window thread is inside C code holding Python's lock (a Qt or card call that does not release it), no Python thread
can run, so there is no sample: the record says so (the 2 s faulthandler dump in run_field_studio.py still writes every
thread's C-safe stack for those).
Each freeze is one line in the session log ('[freeze ...]', its stacks indented under it) and one JSON line in
<session log>.freezes.jsonl beside it, for a harness or a session to read. FIELDSTUDIO_FREEZE_MS sets the threshold
(default 250 ms); 0 turns the record off. Cost while nothing is frozen: one heartbeat a 50 ms (a float store) and the
sampler's check every SAMPLE_MS (a few microseconds)."""
import collections
import json
import os
import sys
import threading
import time

FREEZE_MS = float(os.environ.get('FIELDSTUDIO_FREEZE_MS', '250'))
SAMPLE_MS = 20.0
BEAT_MS = 50
LATE_MS = 100.0                     # the sampler starts taking stacks once the heartbeat is this late
VIEW_DRAGS = ('stroke', 'orbit', 'pan', 'xform', 'mode')
OURS = ('fieldstudio', 'run_field_studio')
SESSION = None                      # (the session log's open file, its path), set by run_field_studio.py


def _stamp(t):
    return time.strftime('%H:%M:%S', time.localtime(t)) + '.%03d' % int((t % 1) * 1000)


def _frames(frame, limit=40):
    """[(short file, line, function)] innermost first"""
    out = []
    while frame is not None and len(out) < limit:
        co = frame.f_code
        out.append((co.co_filename, frame.f_lineno, co.co_name))
        frame = frame.f_back
    return out


def _short(fn):
    fn = fn.replace('\\', '/')
    i = fn.rfind('/fieldstudio/')
    return fn[i + len('/fieldstudio/'):] if i >= 0 else fn.rsplit('/', 1)[-1]


def _ours(stack, n=4):
    """the innermost n frames inside Field Studio, with the innermost frame overall when it is outside (numpy, Qt ...)"""
    mine = [f for f in stack if any(o in f[0].replace('\\', '/') for o in OURS)]
    head = []
    if stack and stack[0] not in mine[:1]:
        head = ['%s:%d %s' % (_short(stack[0][0]), stack[0][1], stack[0][2])]
    return head + ['%s:%d %s' % (_short(f[0]), f[1], f[2]) for f in mine[:n]]


class FreezeLog:
    def __init__(self, win=None, fh=None, freeze_ms=None, jsonl=None, engine=None):
        self.win = win
        self.engine = engine if engine is not None else getattr(win, 'engine', None)
        self.fh = fh
        self.freeze_s = (FREEZE_MS if freeze_ms is None else float(freeze_ms)) / 1000.0
        self.jsonl = jsonl
        self.records = []                       # every freeze this session (a harness reads them)
        self.main_id = threading.main_thread().ident
        self.beat = time.perf_counter()
        self.samples = collections.deque(maxlen=400)    # (t, 'window' | 'engine', stack)
        self.view_wait = None                   # (t the view asked, seq) while a frame is owed during a drag
        self.view_seq_done = -1
        self._stop = False
        self._lock = threading.Lock()
        self.timer = None
        self.thread = None

    # ---- on the window thread ---------------------------------------------------------------------------------------
    def install(self):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        from PySide6.QtCore import Qt
        self.timer = QTimer(app)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(BEAT_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        if self.engine is not None:
            try:
                self.engine.frameReady.connect(self._frame)
            except Exception:
                pass
        vp = getattr(self.win, 'viewport', None)
        if vp is not None and not getattr(vp, '_freeze_wrapped', False):
            req = vp.request

            def request(interactive=True, _req=req):
                _req(interactive)
                if getattr(vp, 'drag', None) in VIEW_DRAGS and self.view_wait is None:
                    self.view_wait = (time.perf_counter(), vp.seq)
            vp.request = request
            vp._freeze_wrapped = True
            eng = self.engine
            if eng is not None:                 # a stroke's moves ask the engine for its frames, not the view
                post = eng.post

                def post_(*job, _post=post):
                    _post(*job)
                    if job and job[0] in ('stroke_begin', 'stroke_move') and self.view_wait is None:
                        self.view_wait = (time.perf_counter(), vp.seq)
                eng.post = post_
        app.aboutToQuit.connect(self.stop)
        self.thread = threading.Thread(target=self._sample_loop, name='freeze-sampler', daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self._stop = True
        if self.timer is not None:
            self.timer.stop()

    def _tick(self):
        now = time.perf_counter()
        gap = now - self.beat
        self.beat = now
        if gap > self.freeze_s + BEAT_MS / 1000.0:
            self._record('window', now - gap, now)
        vw = self.view_wait
        if vw is not None and now - vw[0] > 0.1:
            eng = self.engine
            try:                                # the engine has nothing left to do: no frame was owed (a held tool
                idle = not eng._jobs and not eng._busy and not eng._render_due   # that changed nothing)
            except Exception:
                idle = False
            if idle:
                self.view_wait = None

    def _frame(self, fr):
        vw = self.view_wait
        if vw is None:
            return
        seq = fr.get('seq', 0) if isinstance(fr, dict) else 0
        if seq < vw[1]:
            return                              # an older frame: the one asked for is still owed
        self.view_wait = None
        now = time.perf_counter()
        if now - vw[0] > self.freeze_s:
            self._record('view', vw[0], now)

    # ---- the sampler (its own thread) -----------------------------------------------------------------------------
    def _sample_loop(self):
        while not self._stop:
            time.sleep(SAMPLE_MS / 1000.0)
            now = time.perf_counter()
            late = (now - self.beat) * 1000.0 > LATE_MS
            vw = self.view_wait
            owed = vw is not None and (now - vw[0]) * 1000.0 > LATE_MS
            if not (late or owed):
                continue
            try:
                fr = sys._current_frames()
            except Exception:
                continue
            eng_t = getattr(self.engine, 'thread', None)
            with self._lock:
                if late and self.main_id in fr:
                    self.samples.append((now, 'window', _frames(fr[self.main_id])))
                if eng_t is not None and eng_t.ident in fr:
                    self.samples.append((now, 'engine', _frames(fr[eng_t.ident])))

    # ---- the record ---------------------------------------------------------------------------------------------------
    def _context(self):
        c = {}
        win, eng = self.win, self.engine
        try:
            tp = getattr(win, 'tools', None)
            c['tool'] = getattr(tp, 'tool', None)
            reg = getattr(win, 'reg_tool', None)
            if reg is not None:
                c['tool'] = 'reg:%s' % (reg[0],)
            vp = getattr(win, 'viewport', None)
            c['view'] = getattr(vp, 'drag', None)
            pm = getattr(win, 'panels', None)
            if pm is not None:
                c['panels'] = sorted(n for n, d in pm.docks.items() if d.isVisible())
        except Exception:
            pass
        try:
            c['engine_jobs'] = [j[0] for j in list(eng._jobs)[:8]] if eng is not None else None
            c['engine_busy'] = bool(getattr(eng, '_busy', False))
            c['card'] = getattr(eng, 'gpu_view', None) is not None
        except Exception:
            pass
        return c

    def _record(self, kind, t0, t1):
        with self._lock:
            got = [s for s in self.samples if t0 - 0.05 <= s[0] <= t1 + 0.05]
        rec = {'kind': kind, 'at': time.time() - (time.perf_counter() - t0), 'ms': round((t1 - t0) * 1000.0, 0)}
        rec.update(self._context())
        for who in ('window', 'engine'):
            st = [s[2] for s in got if s[1] == who]
            rec['%s_samples' % who] = len(st)
            cnt = collections.Counter(tuple(_ours(s)) for s in st)
            rec[who] = [[n, list(k)] for k, n in cnt.most_common(3)]
        if kind == 'window' and rec['window_samples'] == 0:
            rec['note'] = ('no window sample: the window thread was inside C code holding Python\'s lock (a Qt or card '
                           'call); the engine thread shows what else ran')
        self.records.append(rec)
        self._write(rec)

    def _write(self, rec):
        if self.fh is not None:
            try:
                head = '[freeze %s] %s %.2f s; tool %s, view %s, panels %s; engine %s%s%s\n' % (
                    _stamp(rec['at']), rec['kind'].upper(), rec['ms'] / 1000.0, rec.get('tool'), rec.get('view'),
                    ','.join(rec.get('panels') or []) or '-',
                    'busy' if rec.get('engine_busy') else 'idle',
                    (' (queued: %s)' % ', '.join(rec['engine_jobs'])) if rec.get('engine_jobs') else '',
                    '' if rec.get('card', True) else ', CPU (no card)')
                lines = [head]
                for who in ('window', 'engine'):
                    for n, fr in rec.get(who) or []:
                        lines.append('    %s thread, %d of %d samples: %s\n' % (
                            who, n, rec['%s_samples' % who], ' <- '.join(fr) or '(no Field Studio frame)'))
                if rec.get('note'):
                    lines.append('    %s\n' % rec['note'])
                self.fh.write(''.join(lines))
            except Exception:
                pass
        if self.jsonl:
            try:
                with open(self.jsonl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, default=str) + '\n')
            except Exception:
                pass


def install(win, fh=None, jsonl=None, freeze_ms=None):
    """start the record on the window's thread -> the FreezeLog (win.freeze_log), or None when FIELDSTUDIO_FREEZE_MS=0"""
    ms = FREEZE_MS if freeze_ms is None else float(freeze_ms)
    if ms <= 0:
        return None
    fl = FreezeLog(win, fh, ms, jsonl).install()
    win.freeze_log = fl
    if fh is not None:
        try:
            fh.write('[freeze] on: the window or the view stopped for over %.0f ms is written here with what ran%s\n'
                     % (ms, (' (and %s)' % jsonl) if jsonl else ''))
        except Exception:
            pass
    return fl
