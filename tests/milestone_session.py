"""The milestone session: one modest, real Field Studio session end to end, in the real window (offscreen Qt).

  1 open Field Studio                 (process start -> the window warm and idle)
  2 a sphere                          (the Shapes row, a click in the view)
  3 sculpt with the live tools        (Swell held + dragged, Settle held, Flatten dragged, a Mask stroke then a region
                                       Transform, a Cut through the top: cap + body)
  4 a library material                (the Materials panel: the body made sandstone)
  5 a limited motion                  (Field Studio has no rig and no keyframes yet: the cut cap is swung 30 deg open
                                       and back in 12 + 12 Transform frames, each frame rendered; one undo step)
  6 a Merge                           (the cap back onto the body)
  7 save, then reopen                 (File > Save to a new project; File > Open it; the state hash compared)
  8 exit cleanly                      (the window's close, then the interpreter's own exit)

Per step: wall time to idle, UI-thread stalls over 100 ms (a 10 ms heartbeat), card memory when a card is used,
engine failures, undo steps, and a window shot when the frame is final. Results: tests/out/milestone_<tag>.json and
tests/out/shots_<tag>/.

CPU only (no graphics card needed; about 5 minutes on an 8-core desktop):
    FIELDSTUDIO_GPU=0 MILE_TAG=cpu_smoke python tests/milestone_session.py
With the card (CuPy + an NVIDIA GPU): leave FIELDSTUDIO_GPU unset.
The run passes when every step reports no failures, step 7's reopened hash equals the saved one, and the process
exits 0."""
import json
import os
import subprocess
import sys
import tempfile
import time

T_START = time.perf_counter()
FS = os.environ.get('FS_TREE') or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine', 'fieldstudio'))
sys.path.insert(0, FS)
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get('MILE_TAG', 'run')
SHOTS = os.path.join(HERE, 'out', 'shots_' + TAG)
os.makedirs(SHOTS, exist_ok=True)
TMP = tempfile.mkdtemp(prefix='fsshell_mile_')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QT_QPA_FONTDIR', os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'))
os.environ['FIELDSTUDIO_SETTINGS'] = os.path.join(TMP, 'settings.ini')
os.environ['FIELDSTUDIO_AUTOSAVE_DIR'] = os.path.join(TMP, 'autosave')
os.makedirs(os.environ['FIELDSTUDIO_AUTOSAVE_DIR'], exist_ok=True)
os.environ.pop('FS_FT_ROWS', None)
import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox
app = QApplication.instance() or QApplication(sys.argv[:1])
from fieldstudio.ui import theme
from fieldstudio.ui.engine import Engine
from fieldstudio.ui.main_window import MainWindow
from fieldstudio.ui import toolreg as TR
import fieldstudio

OUT = {'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': TAG, 'tree': os.path.dirname(os.path.dirname(fieldstudio.__file__)),
       'steps': {}, 'notes': []}
theme.apply(app)
eng = Engine()
win = MainWindow(eng)
win.ask_auto = {}
win.no_prompt = True
eng.start()
win.resize(1600, 900)
win.show()
win.activateWindow()
vp = win.viewport
FRAMES, FAILS, MSGS = [], [], []
eng.frameReady.connect(lambda fr: FRAMES.append(time.perf_counter()))
eng.failed.connect(lambda m: FAILS.append((time.perf_counter(), str(m)[:300])))
eng.message.connect(lambda m: MSGS.append((time.perf_counter(), str(m)[:200])))
import threading
def _watchdog():                                       # the WindowsApps stub: shell timeouts can't kill this process
    OUT['error'] = 'WATCHDOG: the session passed 1200 s'
    json.dump(OUT, open(os.path.join(HERE, 'out', 'milestone_%s.json' % TAG), 'w'), indent=1, default=str)
    print('WATCHDOG EXIT', flush=True)
    from fieldstudio.ui.exit_rule import hard_exit
    hard_exit(4, eng)
_wd = threading.Timer(1200, _watchdog)
_wd.daemon = True
_wd.start()
BEATS = []
_hb = QTimer()
_hb.timeout.connect(lambda: BEATS.append(time.perf_counter()))
_hb.start(10)


def pump(sec=0.05):
    t = time.perf_counter() + sec
    while time.perf_counter() < t:
        app.processEvents(); time.sleep(0.002)


def wait_idle(timeout=300):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        if eng.idle():
            pump(0.03)
            if eng.idle():
                return time.perf_counter() - t0
        time.sleep(0.002)
    return None


def final_frame(timeout=30.0):
    seq = vp.seq
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        fr = vp.frame
        if fr is not None and fr.get('final') and fr.get('seq', seq) >= seq and eng.idle():
            pump(0.15)
            return True
        time.sleep(0.01)
    return False


def card_mem():
    d = {}
    try:
        import cupy as cp
        mp = cp.get_default_memory_pool()
        d['pool_used_mb'] = round(mp.used_bytes() / 2 ** 20, 1)
        d['pool_total_mb'] = round(mp.total_bytes() / 2 ** 20, 1)
    except Exception as e:
        d['pool'] = repr(e)[:80]
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5)
        u, t = (int(x) for x in r.stdout.strip().split(','))
        d['card_used_mb'], d['card_total_mb'] = u, t
    except Exception as e:
        d['smi'] = repr(e)[:80]
    return d


def steps_n():
    g = list((win.snap or {}).get('log_groups') or [])
    i = n = 0
    while i < len(g):
        i += max(int(g[i] or 0), 1)
        n += 1
    return n


class Step:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.b0, self.f0, self.x0, self.n0 = len(BEATS), len(FAILS), len(FRAMES), steps_n()
        print('STEP', self.name, 'begins', flush=True)
        return self

    def __exit__(self, et, ev, tb):
        idle = wait_idle(300)
        t1 = time.perf_counter()
        b = BEATS[self.b0:]
        gaps = np.diff([self.t0] + b + [t1]) * 1000.0 if b else np.array([(t1 - self.t0) * 1000.0])
        rec = {'wall_s': round(t1 - self.t0, 2), 'idle_wait_s': None if idle is None else round(idle, 2),
               'ui_stalls_over_100ms': [round(float(g), 0) for g in gaps if g > 100.0],
               'ui_max_gap_ms': round(float(gaps.max()), 0), 'frames': len(FRAMES) - self.x0,
               'undo_steps': steps_n() - self.n0, 'failures': [m for _t, m in FAILS[self.f0:]],
               'card': card_mem(), 'error': None if ev is None else repr(ev)[:400]}
        rec.update(getattr(self, 'extra', {}))
        OUT['steps'][self.name] = rec
        print('STEP', self.name, json.dumps(rec, default=str)[:600], flush=True)
        json.dump(OUT, open(os.path.join(HERE, 'out', 'milestone_%s.json' % TAG), 'w'), indent=1, default=str)
        return True                                     # record the error and carry on to the next step


def shot(name, preset=None):
    if preset is not None:
        vp.cam.preset(preset); vp.request(False)
    ok = final_frame()
    win.grab().save(os.path.join(SHOTS, '%s.png' % name))
    print('shot', name, 'final' if ok else 'NOT FINAL', flush=True)


R = 0.05


def scr(p):
    x, y, _ = vp.cam.project(np.asarray(p, float), vp.width(), vp.height())
    return QPoint(int(round(x)), int(round(y)))


def front(x, y):
    return scr([x, y, float(np.sqrt(max(R * R - x * x - y * y, 0.0)))])


def key(k):
    win.activateWindow(); vp.setFocus(); pump(0.03)
    QTest.mouseMove(vp, QPoint(vp.width() // 2, vp.height() // 2)); pump(0.03)
    QTest.keyClick(vp, k); pump(0.4)


def stroke(p0, p1=None, hold=0.0, drag_s=0.8, hz=120):
    win.activateWindow(); vp.setFocus(); pump(0.02)
    QTest.mouseMove(vp, p0); pump(0.03)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, p0)
    pump(hold)
    if p1 is not None:
        n = max(2, int(drag_s * hz))
        ts = time.perf_counter()
        for k in range(1, n + 1):
            a = k / n
            QTest.mouseMove(vp, QPoint(int(round(p0.x() + (p1.x() - p0.x()) * a)), int(round(p0.y() + (p1.y() - p0.y()) * a))))
            while time.perf_counter() - ts < a * drag_s:
                app.processEvents(); time.sleep(0.001)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, p1 if p1 is not None else p0)
    pump(0.05)
    wait_idle()


def objs():
    return list((win.snap or {}).get('objects') or [])


try:
    # ---- 1 open --------------------------------------------------------------------------------------------------------
    with Step('1_open') as s:
        while not vp.warm and time.perf_counter() - T_START < 600:
            pump(0.1)
        s.extra = {'process_start_to_warm_s': round(time.perf_counter() - T_START, 2),
                   'renderer': 'card' if getattr(eng, 'gpu_view', None) is not None else 'cpu'}
    shot('1_open')

    # ---- 2 a sphere: the Shapes row, a click in the view ---------------------------------------------------------------
    with Step('2_sphere') as s:
        n_obj = len(objs())
        win.categories.run('reg:add'); pump(0.4)
        vals = dict(getattr(win, 'reg_values', {}) or {})
        vals[('add', 'kind')] = 'sphere'
        win.reg_values = vals
        win.tool_strip._sig = None; pump(0.3)
        c = QPoint(vp.width() // 2, vp.height() // 2)
        win.activateWindow(); vp.setFocus(); QTest.mouseMove(vp, c); pump(0.05)
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, c); pump(0.3)
        wait_idle()
        how = 'Shapes row, a click'
        if len(objs()) == n_obj:                        # fallback, said so
            eng.post('add', {'kind': 'sphere', 'params': {'radius': R}, 'name': 'Sphere', 'h': 0.0005})
            wait_idle()
            how = 'the Shapes row placed nothing: Add > sphere'
        TR.clear(win)
        win.tools.set_tool('select'); pump(0.2)
        o = objs()[-1]
        s.extra = {'how': how, 'object': {k: o.get(k) for k in ('name', 'id', 'transform')}}
        oid = o['id']
        win.props.set_selection([oid]); pump(0.2)
        vp.cam.preset(1); eng.post('frame', [oid]); wait_idle(); pump(0.3)
    # the sphere's real centre and radius (the Shapes row's size), read back for the strokes
    def geom(e):
        from fieldstudio.warp.bake import object_field
        sc = e.hist.scene
        o_ = sc.get(oid)
        f = object_field(sc, o_)
        t = np.asarray((o_.transform or {}).get('translate', [0, 0, 0]), float)
        zs = np.linspace(0.0, 0.5, 20001)
        v = f(np.stack([np.full_like(zs, t[0]), np.full_like(zs, t[1]), t[2] + zs], 1))
        k = np.nonzero(v > 0)[0]
        return t.tolist(), float(zs[k[0]]) if len(k) else None
    box = {}
    tok = object()
    eng.callDone.connect(lambda r: box.setdefault('r', r[1]) if r[0] is tok else None)
    eng.post('call', geom, tok)
    t_ = time.perf_counter()
    while 'r' not in box and time.perf_counter() - t_ < 60:
        app.processEvents(); time.sleep(0.005)
    C0, Rr = box.get('r', ([0, 0, 0], R))
    R = float(Rr or R)
    C0 = np.asarray(C0, float)
    OUT['sphere'] = {'centre_m': C0.tolist(), 'radius_m': R}

    def front(x, y):                                   # noqa: F811 (the measured sphere)
        return scr(C0 + np.array([x * R, y * R, float(np.sqrt(max(1.0 - x * x - y * y, 0.0))) * R]))
    shot('2_sphere')

    # ---- 3 sculpt ------------------------------------------------------------------------------------------------------
    with Step('3a_swell') as s:
        key(Qt.Key_I)
        s.extra = {'tool': win.tools.tool, 'row': TR.current(win)}
        stroke(front(0.0, 0.0), hold=0.8)                         # held still
        stroke(front(-0.45, -0.35), front(0.45, -0.35), hold=0.2, drag_s=1.0)   # dragged
    shot('3a_swell', preset=5)
    vp.cam.preset(1); vp.request(False); wait_idle()
    with Step('3b_settle') as s:
        side = next((c_ for c_ in win.tool_strip.own.findChildren(QComboBox) if c_.findData('settle') >= 0), None)
        if side is not None:
            side.setCurrentIndex(side.findData('settle')); pump(0.4)
        s.extra = {'tool': win.tools.tool}
        stroke(front(0.35, 0.35), hold=0.6)
    with Step('3c_flatten') as s:
        key(Qt.Key_T)
        s.extra = {'tool': win.tools.tool, 'row': TR.current(win)}
        stroke(front(-0.5, 0.35), front(-0.1, 0.35), hold=0.2, drag_s=0.8)
    shot('3c_settle_flatten', preset=5)
    vp.cam.preset(1); vp.request(False); wait_idle()
    with Step('3d_region_transform') as s:
        key(Qt.Key_B)                                   # Mask / Protect: paint the left half held
        s.extra = {'mask_tool': win.tools.tool}
        stroke(front(-0.7, 0.5), front(-0.7, -0.5), hold=0.1, drag_s=0.7)
        stroke(front(-0.4, 0.6), front(-0.4, -0.6), hold=0.1, drag_s=0.7)
        win.tools.set_tool('move'); pump(0.3)
        rx = getattr(win, 'region_xform', None)
        t_ = time.perf_counter()
        while rx is not None and rx.state() != 'partial' and time.perf_counter() - t_ < 20:
            pump(0.1)
        s.extra['region_state'] = rx.state() if rx is not None else None
        pump(0.4)
        g = vp.gizmo.geo()
        dv, L = g['axes'][0]
        p0 = g['sc'] + dv * (L * 0.75)
        p1 = p0 + dv * 40.0
        win.activateWindow(); vp.setFocus()
        QTest.mouseMove(vp, QPoint(int(p0[0]), int(p0[1]))); pump(0.03)
        QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(int(p0[0]), int(p0[1]))); pump(0.03)
        for k in range(1, 13):
            QTest.mouseMove(vp, QPoint(int(p0[0] + (p1[0] - p0[0]) * k / 12), int(p0[1] + (p1[1] - p0[1]) * k / 12))); pump(0.02)
        QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(int(p1[0]), int(p1[1]))); pump(0.1)
        wait_idle()
        eng.post('mask', oid, 'clear'); wait_idle()     # Mask > Clear (the cut and merge below act on the whole form)
        win.tools.set_tool('select'); pump(0.2)
    shot('3d_region_transform', preset=5)
    vp.cam.preset(1); vp.request(False); wait_idle()
    with Step('3e_cut') as s:
        win.props.set_selection([oid]); pump(0.2)
        win.categories.run('reg:cut'); pump(0.4)
        vals = dict(getattr(win, 'reg_values', {}) or {})
        vals[('cut', 'type')] = 'line'
        vals[('cut', 'length')] = 0.3
        win.reg_values = vals
        win.props.set_selection([oid]); wait_idle()
        before = {o['id'] for o in objs()}
        yc = C0[1] + 0.55 * R
        a, b = scr([C0[0] - 1.4 * R, yc, C0[2] + R]), scr([C0[0] + 1.4 * R, yc, C0[2] + R])
        win.activateWindow(); vp.setFocus(); QTest.mouseMove(vp, a); pump(0.03)
        QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, a); pump(0.05)
        for k in range(1, 13):
            QTest.mouseMove(vp, QPoint(int(a.x() + (b.x() - a.x()) * k / 12), int(a.y() + (b.y() - a.y()) * k / 12))); pump(0.03)
        wait_idle()
        QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, b); pump(0.1)
        wait_idle()
        new = [o for o in objs() if o['id'] not in before]
        s.extra = {'new_objects': [o['name'] for o in new], 'objects': [o['name'] for o in objs()]}
        TR.clear(win)
        win.tools.set_tool('select'); pump(0.2)
    shot('3e_cut', preset=5)
    vp.cam.preset(1); vp.request(False); wait_idle()

    # which piece is the cap (above the line) and which the body
    def pieces(e):
        from fieldstudio.tools import measure as MS
        out = []
        for o_ in e.hist.scene.objects:
            try:
                lo, hi = MS.extents(e.hist.scene, o_.id)
                out.append((o_.id, o_.name, float(hi[1]), float(lo[1])))
            except Exception:
                pass
        return out
    box = {}
    tok = object()
    eng.callDone.connect(lambda r: box.setdefault('r', r[1]) if r[0] is tok else None)
    eng.post('call', pieces, tok)
    t_ = time.perf_counter()
    while 'r' not in box and time.perf_counter() - t_ < 60:
        app.processEvents(); time.sleep(0.005)
    pcs = box.get('r') or []
    OUT['pieces'] = pcs
    pcs = sorted(pcs, key=lambda p: p[3])               # by the lowest point: the body first, the cap (highest low) last
    body_id = pcs[0][0] if pcs else oid
    cap_id = pcs[-1][0] if len(pcs) > 1 else None

    # ---- 4 a library material: the Materials panel -----------------------------------------------------------------------
    with Step('4_material') as s:
        P = getattr(win, 'materials_panel', None)
        win.props.set_selection([body_id]); pump(0.5)
        found = None
        if P is not None:
            fam_ids = [b_ for b_ in (P.fam_group.buttons() if hasattr(P, 'fam_group') else [])]
            for bt in [None] + fam_ids:
                if bt is not None:
                    bt.click(); pump(0.2)
                for i in range(P.list.count()):
                    if str(P.list.item(i).data(Qt.UserRole)).startswith('stone.sandstone'):
                        found = (i, P.list.item(i).data(Qt.UserRole)); break
                if found:
                    break
            if found:
                P.list.setCurrentRow(found[0]); pump(0.4)
                P.apply_btn.click(); pump(0.3)
                wait_idle()
        mat = next((o.get('material') for o in objs() if o['id'] == body_id), None)
        s.extra = {'panel': P is not None, 'picked': found and found[1], 'body_material': mat}
    shot('4_material', preset=5)

    # ---- 5 a limited motion: the cap swung open and back (Transform frames; no rig, no keyframes) ---------------------
    with Step('5_motion') as s:
        frames_t = []
        if cap_id is not None:
            tr0 = dict(next(o for o in objs() if o['id'] == cap_id).get('transform') or {})
            rot0 = list(tr0.get('rotate', [0.0, 0.0, 0.0]))
            drag_id = 'milestone-motion'
            vp.cam.preset(5); vp.request(False); wait_idle()
            seq = [30.0 * k / 12 for k in range(1, 13)] + [30.0 * (12 - k) / 12 for k in range(1, 13)]
            for i, ang in enumerate(seq):
                tr = dict(tr0)
                tr['rotate'] = [rot0[0] - ang, rot0[1], rot0[2]]
                tf = time.perf_counter()
                eng.post('props', cap_id, {'transform': tr}, drag_id)
                vp.request(True)
                wait_idle()
                frames_t.append(round((time.perf_counter() - tf) * 1000, 1))
                if i in (5, 11, 17):
                    win.grab().save(os.path.join(SHOTS, '5_motion_f%02d.png' % i))
            s.extra = {'cap': cap_id, 'frames': len(seq), 'frame_ms_p50': float(np.median(frames_t)),
                       'frame_ms_max': float(max(frames_t)), 'note': 'Field Studio has no rig and no keyframes: the '
                       'motion is the cap turned by Transform frames (30 deg open, then closed), one undo step'}
        else:
            s.extra = {'note': 'the cut made no separate cap: no part to move'}
    shot('5_motion_closed', preset=5)

    # ---- 6 a Merge: the cap back onto the body ---------------------------------------------------------------------------
    with Step('6_merge') as s:
        if cap_id is not None:
            win.props.set_selection([body_id, cap_id]); pump(0.3)
            win.categories.run('reg:merge'); pump(0.4)
            btn = getattr(win.tool_strip, 'reg_act_btn', None)
            if btn is not None and btn.isVisible():
                btn.click(); pump(0.3)
            wait_idle()
        TR.clear(win)
        win.tools.set_tool('select'); pump(0.2)
        s.extra = {'objects': [o['name'] for o in objs()]}
    shot('6_merge', preset=5)

    # ---- 7 save, reopen -----------------------------------------------------------------------------------------------
    path = os.path.join(TMP, 'milestone.fsproj')

    def state_hash(e):
        return e.hist.scene.state_hash()
    with Step('7a_save') as s:
        win.project_path = path
        win.save_project()
        wait_idle()
        eng.wait_saves(600)
        box = {}; tok = object()
        eng.callDone.connect(lambda r: box.setdefault('r', r[1]) if r[0] is tok else None)
        eng.post('call', state_hash, tok)
        t_ = time.perf_counter()
        while 'r' not in box and time.perf_counter() - t_ < 60:
            app.processEvents(); time.sleep(0.005)
        h_saved = box.get('r')
        s.extra = {'path': path, 'exists': os.path.isdir(path), 'hash': str(h_saved)[:16]}
    with Step('7b_reopen') as s:
        win.set_tool_mode(None)                         # File > Open, the folder chosen (open_project's own steps)
        win.project_path = win.project_target(path)
        try:
            from fieldstudio.tools import bakes as _BK
            _BK.set_project(path)
        except Exception:
            pass
        eng.post('open', path)
        wait_idle(600)
        box = {}; tok = object()
        eng.callDone.connect(lambda r: box.setdefault('r', r[1]) if r[0] is tok else None)
        eng.post('call', state_hash, tok)
        t_ = time.perf_counter()
        while 'r' not in box and time.perf_counter() - t_ < 60:
            app.processEvents(); time.sleep(0.005)
        h_open = box.get('r')
        s.extra = {'hash': str(h_open)[:16], 'same_as_saved': h_open == h_saved, 'objects': [o['name'] for o in objs()]}
    shot('7_reopened', preset=5)
except Exception:
    import traceback
    traceback.print_exc()
    OUT['error'] = traceback.format_exc()[-2500:]

# ---- 8 exit cleanly: his close ------------------------------------------------------------------------------------------
OUT['failures_all'] = [m for _t, m in FAILS]
OUT['messages_tail'] = [m for _t, m in MSGS[-12:]]
OUT['card_before_close'] = card_mem()
t_close = time.perf_counter()
OUT['busy_at_close'] = not eng.idle()
_hb.stop()
win.close()                                             # closeEvent -> Engine.stop -> the exit rule
OUT['close_s'] = round(time.perf_counter() - t_close, 3)
OUT['last_exit'] = getattr(eng, 'last_exit', None)
OUT['session_s'] = round(time.perf_counter() - T_START, 1)
json.dump(OUT, open(os.path.join(HERE, 'out', 'milestone_%s.json' % TAG), 'w'), indent=1, default=str)
print('CLOSED', json.dumps({k: OUT[k] for k in ('close_s', 'session_s', 'busy_at_close')}), flush=True)
_wd.cancel()
app.quit()
# the interpreter's own exit (no os._exit): the exit rule already stopped launching and synchronised
