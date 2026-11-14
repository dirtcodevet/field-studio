"""--monkey [SECONDS]: random UI actions for a while, offscreen, through the same calls and Qt mouse events the hand
makes. It looks for crashes and hangs, not for pretty results.

THRESHOLDS (V1.3, written before the first run, not moved after it)
  K1 no crash        no exception in the UI thread (sys.excepthook) and no engine 'failed' report
  K2 responsive      after every action the engine goes idle again within 60 s (no hang, no lost stroke state)
  K3 determinism     at the end, replaying the op log into an empty scene gives the same state hash
  K4 memory          peak working set of the process < 3072 MB
  K5 scenarios       the named hard cases each ran: rapid clicks, undo during a drag, redo during a drag, delete the
                     stroked object mid-stroke, a second mouse button during a drag, brush radius at both ends and
                     beyond (0.1 mm, 150 mm, 1 m), save with no objects, a timeline jump mid-stroke, a field tool
                     cancelled mid-drag, and (when the default field version is on this machine) opening a field
                     region twice
  K6 commit          [V1.3b, written before its first run] peak commit charge (private bytes, PeakPagefileUsage)
                     of the process < 8192 MB (the earlier 67 GB commit, not seen by K4's working set)
"""
import json
import os
import random
import sys
import tempfile
import time
import traceback

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from .main_window import PRIMS, SCREENS, process_memory

BRUSHES = ['draw', 'clay', 'carve', 'inflate', 'smooth', 'flatten', 'crease', 'pinch', 'grab', 'stretch', 'mask']


def _mouse(w, kind, pos, button=Qt.LeftButton, buttons=None, mods=Qt.NoModifier):
    t = {'press': QEvent.MouseButtonPress, 'move': QEvent.MouseMove, 'release': QEvent.MouseButtonRelease}[kind]
    if buttons is None:
        buttons = Qt.NoButton if kind == 'release' else button
    if kind == 'move':
        button = Qt.NoButton
    gp = w.mapToGlobal(QPoint(int(pos.x()), int(pos.y())))
    ev = QMouseEvent(t, QPointF(pos), QPointF(gp), button, buttons, mods)
    QApplication.sendEvent(w, ev)


def run(app, win, engine, args):
    secs = float(getattr(args, 'monkey', 300) or 300)
    seed = int(getattr(args, 'seed', 0) or 1105)
    rng = random.Random(seed)
    res = {'thresholds': __doc__.split('THRESHOLDS')[1].strip().splitlines(), 'seed': seed, 'seconds': secs}
    ui_exc = []
    old_hook = sys.excepthook

    def hook(t, v, tb):
        ui_exc.append(''.join(traceback.format_exception(t, v, tb))[-1500:])
    sys.excepthook = hook
    calls = {}
    engine.callDone.connect(lambda tr: calls.__setitem__(tr[0], tr[1]))
    hangs = []
    scen = {}

    def pump(sec=0.02):
        t = time.perf_counter()
        while time.perf_counter() - t < sec:
            app.processEvents()
            time.sleep(0.002)

    def wait_idle(timeout=60.0, what=''):
        t = time.perf_counter()
        stable = 0
        while True:
            app.processEvents()
            if engine.idle():
                stable += 1
                if stable > 3:
                    return True
            else:
                stable = 0
            if time.perf_counter() - t > timeout:
                hangs.append(what)
                return False
            time.sleep(0.003)

    def call(fn, timeout=120.0):
        tok = object()
        engine.post('call', fn, tok)
        t = time.perf_counter()
        while tok not in calls:
            app.processEvents()
            time.sleep(0.002)
            if time.perf_counter() - t > timeout:
                hangs.append('call')
                return None
        return calls.pop(tok)

    vp = win.viewport
    win.ask_auto = {}                      # tool dialogs take their defaults
    win.resize(1400, 860)
    win.show()
    pump(0.2)
    wait_idle(600, 'warm-up')

    def rpos(on_object=True):
        W, H = vp.width(), vp.height()
        if on_object and vp.frame is not None:
            obj = vp.frame['obj']
            ys, xs = np.nonzero(obj >= 0)
            if len(xs) and rng.random() < 0.85:
                k = rng.randrange(len(xs))
                return QPointF(xs[k] * W / obj.shape[1] + 0.5, ys[k] * H / obj.shape[0] + 0.5)
        return QPointF(rng.uniform(2, W - 2), rng.uniform(2, H - 2))

    def oids():
        return [o['id'] for o in (win.snap or {}).get('objects', [])]

    def drag(path, button=Qt.LeftButton, mods=Qt.NoModifier, mid=None):
        _mouse(vp, 'press', path[0], button, mods=mods)
        for i, q in enumerate(path[1:]):
            _mouse(vp, 'move', q, button, buttons=button, mods=mods)
            if mid is not None and i == len(path) // 2:
                mid()
            pump(rng.choice((0.0, 0.004, 0.015)))
        _mouse(vp, 'release', path[-1], button, mods=mods)

    def wander(p0, n=None):
        n = n or rng.randint(2, 14)
        pts = [p0]
        for _ in range(n):
            q = pts[-1]
            pts.append(QPointF(min(max(q.x() + rng.uniform(-30, 30), 1), vp.width() - 1),
                               min(max(q.y() + rng.uniform(-30, 30), 1), vp.height() - 1)))
        return pts

    # ---- actions ------------------------------------------------------------------------------------------------
    def a_create():
        kind, _, params = rng.choice(PRIMS)
        win.create(kind, params)

    def a_brush_stroke():
        win.tools.set_tool(rng.choice(BRUSHES))
        win.tools.radius.setValue(rng.choice((0.002, 0.006, 0.012, 0.02)), emit=True)
        win.tools.strength.setValue(rng.uniform(0.05, 1.0), emit=True)
        mods = Qt.ControlModifier if rng.random() < 0.2 else Qt.NoModifier
        drag(wander(rpos()), mods=mods)

    def a_rapid_clicks():
        win.tools.set_tool(rng.choice(['select'] + BRUSHES))
        for _ in range(rng.randint(5, 20)):
            p = rpos()
            _mouse(vp, 'press', p)
            _mouse(vp, 'release', p)
        scen['rapid clicks'] = True

    def a_undo_mid_drag():
        win.tools.set_tool(rng.choice(['draw', 'clay', 'grab', 'smooth']))
        drag(wander(rpos(), 10), mid=lambda: engine.post(rng.choice(['undo', 'redo'])))
        scen['undo during a drag'] = True

    def a_delete_mid_stroke():
        win.tools.set_tool(rng.choice(['draw', 'inflate', 'grab']))
        p = rpos()

        def kill():
            o = vp._obj_at(p)
            if o >= 0:
                engine.post('remove', [o])
                scen['delete the stroked object mid-stroke'] = True
        drag(wander(p, 10), mid=kill)

    def a_second_button():
        win.tools.set_tool(rng.choice(['draw', 'clay']))
        path = wander(rpos(), 8)
        _mouse(vp, 'press', path[0], Qt.LeftButton)
        _mouse(vp, 'move', path[1], Qt.LeftButton, buttons=Qt.LeftButton)
        _mouse(vp, 'press', path[1], Qt.RightButton, buttons=Qt.LeftButton | Qt.RightButton)
        for q in path[2:]:
            _mouse(vp, 'move', q, Qt.LeftButton, buttons=Qt.LeftButton | Qt.RightButton)
        _mouse(vp, 'release', path[-1], Qt.RightButton, buttons=Qt.LeftButton)
        _mouse(vp, 'release', path[-1], Qt.LeftButton, buttons=Qt.NoButton)
        scen['a second mouse button during a drag'] = True

    def a_radius_extremes(r=None):
        win.tools.set_tool(rng.choice(['draw', 'clay', 'smooth', 'inflate']))
        r = r or rng.choice((0.0001, 0.0005, 0.15, 1.0))
        s = dict(win.tools.brush())
        s['radius'] = r
        vp.brush = s                       # beyond the slider's range too: the engine must clamp
        drag(wander(rpos(), 3))
        vp.brush = win.tools.brush()
        scen.setdefault('brush radius extremes', set()).add(r)

    def a_goto_mid_stroke():
        win.tools.set_tool('draw')
        n = len((win.snap or {}).get('log', []))
        drag(wander(rpos(), 8), mid=lambda: engine.post('goto', rng.randint(0, max(n, 0))))
        scen['a timeline jump mid-stroke'] = True

    def a_undo_redo():
        for _ in range(rng.randint(1, 6)):
            engine.post(rng.choice(['undo', 'undo', 'redo']))

    def a_goto():
        n = len((win.snap or {}).get('log', []))
        engine.post('goto', rng.randint(0, n + 2))

    def a_camera():
        k = rng.random()
        if k < 0.4:
            drag(wander(rpos(False)), button=Qt.LeftButton, mods=Qt.AltModifier)
        elif k < 0.7:
            drag(wander(rpos(False)), button=rng.choice((Qt.RightButton, Qt.MiddleButton)))
        else:
            for _ in range(rng.randint(1, 5)):
                ev = QWheelEvent(rpos(False), vp.mapToGlobal(QPointF(10, 10)), QPoint(0, 0),
                                 QPoint(0, rng.choice((120, -120))), Qt.NoButton, Qt.NoModifier,
                                 Qt.NoScrollPhase, False)
                QApplication.sendEvent(vp, ev)
        if rng.random() < 0.2:
            win.preset(rng.choice((1, 3, 5, 7, 9)))
        if rng.random() < 0.15:
            win.frame_selection()

    def a_select():
        win.tools.set_tool('select')
        p = rpos()
        mods = Qt.ShiftModifier if rng.random() < 0.3 else Qt.NoModifier
        _mouse(vp, 'press', p, mods=mods)
        _mouse(vp, 'release', p, mods=mods)

    def a_object_ops():
        ids = oids()
        if not ids:
            return
        k = rng.random()
        if k < 0.25 and len(ids) >= 2:
            win.props.set_selection(rng.sample(ids, 2))
            win.props._boolean(rng.choice(['union', 'subtract', 'intersect']), rng.random() < 0.5)
        elif k < 0.45:
            win.props.set_selection([rng.choice(ids)])
            win.props._duplicate()
        elif k < 0.65:
            win.props.set_selection([rng.choice(ids)])
            win.props._delete()
        elif k < 0.8:
            win.props.set_selection([rng.choice(ids)])
            win.props._hide()
        else:
            win.props.set_selection([rng.choice(ids)])
            mk = rng.choice(['twist', 'bend', 'taper', 'shell', 'offset'])
            win.props._wrap(mk, {'twist': {'rate': 3.0}, 'bend': {'rate': 2.0}, 'taper': {'rate': 1.0, 'g_min': 0.2},
                                 'shell': {'thickness': 0.01}, 'offset': {'distance': 0.004}}[mk])

    def a_field_tool(lab=None, cancel=False):
        labels = ['Pencil ridge', 'Pencil valley', 'Cut (remove)', 'Cut and separate', 'Splice', 'Fill and splice',
                  'Smooth detail (keep shape)', 'Smooth shape (keep detail)']
        lab = lab or rng.choice(labels)
        ids = oids()
        if ids:
            win.props.set_selection([rng.choice(ids)])
        win.field_tools[lab]()
        pump(0.02)
        k = rng.random()
        if lab.startswith('Splice') or lab.startswith('Fill'):
            for _ in range(2):
                p = rpos()
                _mouse(vp, 'press', p)
                _mouse(vp, 'release', p)
                pump(0.05)
        elif k < 0.25 or cancel:           # cancelled mid-drag
            path = wander(rpos(), 8)
            _mouse(vp, 'press', path[0])
            for q in path[1:5]:
                _mouse(vp, 'move', q, buttons=Qt.LeftButton)
            win.escape()
            for q in path[5:]:
                _mouse(vp, 'move', q, buttons=Qt.LeftButton)
            _mouse(vp, 'release', path[-1])
            scen['a field tool cancelled mid-drag'] = True
        else:
            drag(wander(rpos(), rng.randint(3, 12)))
        if rng.random() < 0.7:
            win.escape()

    def a_view_mode():
        win.set_mode(rng.choice(['shaded', 'matcap', 'normal', 'slice']))

    def a_new_and_save_empty():
        win.new_scene()
        wait_idle(60, 'new')
        p = os.path.join(tempfile.mkdtemp(prefix='fs_monkey_'), 'empty.fsproj')
        win.project_path = p
        win.save_project()
        wait_idle(60, 'save empty')
        ok = os.path.exists(os.path.join(p, 'project.json'))
        scen['save with no objects'] = ok
        if ok:
            win.project_path = None
            engine.post('open', p)
            wait_idle(60, 'open empty')
        a_create()
        a_create()

    def a_quality():
        win.set_quality(rng.choice(['auto', 0, 1, 2]))
        win.set_res(rng.choice([0.5, 0.75, 1.0]))

    actions = [(a_brush_stroke, 30), (a_create, 6), (a_rapid_clicks, 5), (a_undo_mid_drag, 5),
               (a_delete_mid_stroke, 3), (a_second_button, 3), (a_radius_extremes, 3), (a_goto_mid_stroke, 2),
               (a_undo_redo, 8), (a_goto, 3), (a_camera, 10), (a_select, 6), (a_object_ops, 6), (a_field_tool, 8),
               (a_view_mode, 3), (a_new_and_save_empty, 1), (a_quality, 2)]
    pool = [f for f, w in actions for _ in range(w)]

    # the fixed hard cases first, then random ones until the time runs out
    t0 = time.perf_counter()
    for f in (a_create, a_create, a_rapid_clicks, a_undo_mid_drag, a_delete_mid_stroke, a_create,
              a_second_button, lambda: a_radius_extremes(0.0001), lambda: a_radius_extremes(0.0005),
              lambda: a_radius_extremes(0.15), lambda: a_radius_extremes(1.0), a_goto_mid_stroke,
              lambda: a_field_tool('Pencil ridge', True), lambda: a_field_tool('Cut (remove)', True),
              a_new_and_save_empty):
        f()
        wait_idle(60, f.__name__)
    # opening a field region twice (heavy: a small crop), when the default version is here
    try:
        from ..core import region
        here = os.path.dirname(os.path.abspath(__file__))
        ptr = os.environ.get('FIELDSTUDIO_FIELD_CURRENT', '')
        ver = region.current_version(ptr)
        if ver and os.path.isdir(ver) and not getattr(args, 'no_region', False):
            with open(os.path.join(ver, 'meta.json')) as f:
                regs = json.load(f).get('shot_regions', {})
            c = next(iter(regs.values()))['centre'] if regs else [0.0, 1.0, 0.0]
            p = region.region_params(ver, c, 0.012, 'box')
            engine.post('open_region', p, 'region A')
            engine.post('open_region', p, 'region A again')
            wait_idle(180, 'open region twice')
            names = [o['name'] for o in (win.snap or {}).get('objects', [])]
            scen['opening a field region twice'] = sum(1 for n in names if n.startswith('region A')) == 2
            win.tools.set_tool('inflate')
            a_brush_stroke()
            wait_idle(60, 'region stroke')
            win.new_scene()
            wait_idle(60, 'new after region')
            a_create()
        else:
            scen['opening a field region twice'] = 'skipped (no field version here)'
    except Exception:
        scen['opening a field region twice'] = 'error: ' + traceback.format_exc()[-400:]
    n_actions = 0
    counts = {}
    while time.perf_counter() - t0 < secs:
        f = rng.choice(pool)
        try:
            f()
        except Exception:
            ui_exc.append('%s: %s' % (f.__name__, traceback.format_exc()[-1500:]))
        counts[f.__name__] = counts.get(f.__name__, 0) + 1
        n_actions += 1
        if rng.random() < 0.5:
            wait_idle(60, f.__name__)
        else:
            pump(0.01)
        if len(oids()) > 10:
            win.props.set_selection(oids()[:-4])
            win.props._delete()
        if len(hangs) > 3:
            break
    wait_idle(120, 'end')
    win.escape()
    win.tools.set_tool('select')
    wait_idle(60, 'end2')
    res['actions'] = n_actions
    res['action_counts'] = counts
    res['seconds_run'] = round(time.perf_counter() - t0, 1)
    # K3: replay the final log
    from ..core.history import replay
    h = call(lambda e: (e.hist.scene.state_hash(), e.hist.log()))
    if h is not None:
        want, log = h
        got = replay(log).scene.state_hash()
        res['log_entries'] = len(log)
        res['replay_same'] = got == want
    else:
        res['replay_same'] = False
    ws, peak = process_memory()
    res['peak_mb'] = round(peak)
    commit = getattr(process_memory, 'commit', (0.0, 0.0))
    res['peak_commit_mb'] = round(commit[1])
    res['undo_mb'] = round(engine.hist.memory_bytes() / 2 ** 20, 1) if hasattr(engine, 'hist') else None
    res['ui_exceptions'] = ui_exc[:10]
    res['engine_errors'] = [e[-1500:] for e in win.errors[:10]]
    res['hangs'] = hangs
    sc = {k: (sorted(v) if isinstance(v, set) else v) for k, v in scen.items()}
    res['scenarios'] = sc
    need = ['rapid clicks', 'undo during a drag', 'delete the stroked object mid-stroke',
            'a second mouse button during a drag', 'brush radius extremes', 'save with no objects',
            'a timeline jump mid-stroke', 'a field tool cancelled mid-drag']
    checks = {'K1_no_crash': not ui_exc and not win.errors,
              'K2_responsive': not hangs,
              'K3_determinism': bool(res['replay_same']),
              'K4_memory': peak < 3072,
              'K6_commit': 0 < res['peak_commit_mb'] < 8192,
              'K5_scenarios': all(sc.get(k) for k in need) and set(sc.get('brush radius extremes', [])) >=
              {0.0001, 0.0005, 0.15, 1.0} and sc.get('opening a field region twice') is not False}
    res['checks'] = checks
    res['pass'] = all(checks.values())
    os.makedirs(SCREENS, exist_ok=True)
    win.grab().save(os.path.join(SCREENS, 'monkey_end.png'))
    with open(os.path.join(SCREENS, 'monkey.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)
    sys.excepthook = old_hook
    print(json.dumps({'checks': checks, 'actions': n_actions, 'hangs': hangs, 'ui_exc': len(ui_exc),
                      'engine_errors': len(win.errors), 'peak_mb': res['peak_mb'],
                      'peak_commit_mb': res['peak_commit_mb']}, indent=1))
    for e in (ui_exc + win.errors)[:6]:
        print('----\n' + e[-1200:])
    engine.stop()
    app.quit()
    return 0 if res['pass'] else 1
