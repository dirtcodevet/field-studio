"""--selftest: build a demo scene through the same engine jobs and viewport stroke calls the mouse uses, render,
check, grab the WINDOW (QWidget.grab) to screens/selftest.png and exit.

THRESHOLDS (written before the first run, not moved after it)
  S1 grab       screens/selftest.png written, at least 1000x600 px
  S2 frame      the final frame is quality 2; objects cover > 5 % of its pixels; 0 unresolved rays
  S3 scene      cube + sphere + cylinder were added; after the boolean 2 objects remain; one base tree holds a union
                node with blend > 0 (smooth) and another a twist node
  S4 strokes    10 strokes (10 tools) through Viewport.begin/move/end_stroke -> 10 BrushStroke log entries and a
                changed state hash
  S5 undo/redo  undo x10 -> the pre-stroke hash; redo x10 -> the post-stroke hash
  S6 replay     replaying the op log into an empty scene -> the same hash
  S7 save/open  save .fsproj -> open -> the same hash
  S8 UI         the UI thread's longest event-loop gap while the engine renders and sculpts (after warm-up) < 100 ms
  S9 memory     peak working set of the process < 3072 MB
  S10 launch    [V1.1, written before the V1.1 run] the warm-up shows progress in the viewport through at least 7
                steps ending at 'ready' (renderer, brushes and the field tools' kernels, so the first tool action
                does not stall on a compile); Help > Quick start opens as a panel that lists the controls (it
                names Orbit, Pencil, Cut, Splice and Undo); its grab is saved to screens/fs_quickstart.png
  V1-V8         the stage-3 demo scene and Field Tools: see ui/demo.py (its thresholds were written first)
  R1-R6         [V1.2] the field-region test (ui/region_test.py; its thresholds were written first), when the
                default field version exists; --selftest-region runs it alone
"""
import json
import os
import tempfile
import time

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPointF, QTimer

from .main_window import SCREENS, process_memory


class GapMeter:
    def __init__(self):
        self.t = QElapsedTimer()
        self.t.start()
        self.last = None
        self.max_gap = 0.0
        self.on = False
        self.timer = QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def tick(self):
        now = self.t.nsecsElapsed() / 1e6
        if self.on and self.last is not None:
            self.max_gap = max(self.max_gap, now - self.last)
        self.last = now


def run(app, win, engine, args):
    res = {'thresholds': __doc__.split('THRESHOLDS')[1].strip().splitlines()}
    checks = {}
    calls = {}
    engine.callDone.connect(lambda tr: calls.__setitem__(tr[0], tr[1]))
    gap = GapMeter()

    def pump(sec=0.05):
        t = time.perf_counter()
        while time.perf_counter() - t < sec:
            app.processEvents()
            time.sleep(0.002)

    def wait_idle(timeout=180.0):
        t = time.perf_counter()
        stable = 0
        while True:
            app.processEvents()
            if engine.idle():
                stable += 1
                if stable > 5:
                    break
            else:
                stable = 0
            if time.perf_counter() - t > timeout:
                raise TimeoutError('engine not idle after %.0f s' % timeout)
            time.sleep(0.003)
        pump(0.05)

    def call(fn):
        tok = object()
        engine.post('call', fn, tok)
        t = time.perf_counter()
        while tok not in calls:
            app.processEvents()
            time.sleep(0.002)
            if time.perf_counter() - t > 120:
                raise TimeoutError('call')
        return calls.pop(tok)

    def state_hash():
        return call(lambda e: e.hist.scene.state_hash())

    t0 = time.perf_counter()
    win.resize(1600, 960)
    win.show()
    pump(0.2)
    wait_idle(600)
    res['warmup_s'] = round(time.perf_counter() - t0, 1)
    vp = win.viewport
    if getattr(args, 'selftest_region', None) is not None:      # V1.2: only the field-region test
        raise SystemExit('--selftest-region is not part of this release')
        rchecks, rres = region_test.run(app, win, engine, {'pump': pump, 'wait_idle': wait_idle, 'call': call},
                                        args.selftest_region or None)
        rres['warmup_s'] = res['warmup_s']
        rres['total_s'] = round(time.perf_counter() - t0, 1)
        rres['engine_errors'] = win.errors
        rres['peak_working_set_MB'] = round(process_memory()[1], 0)
        rres['pass'] = rres['pass'] and not win.errors
        with open(os.path.join(SCREENS, 'fs_region.json'), 'w', encoding='utf-8') as f:
            json.dump(rres, f, indent=1, default=str)
        print(json.dumps({k: rres[k] for k in ('checks', 'pass', 'total_s', 'peak_working_set_MB', 'engine_errors')},
                         indent=1, default=str))
        engine.stop()
        win.close()
        return 0 if rres['pass'] else 1
    gap.on = True

    # -- S3 the demo scene: cube, sphere, cylinder; a smooth union; a twist ----------------------------------------
    engine.post('new')
    engine.post('add', {'kind': 'box', 'name': 'cube', 'params': {'size': [0.14, 0.14, 0.14], 'round': 0.008},
                        'translate': [-0.13, 0.0, 0.0], 'colour': {'albedo': [0.62, 0.52, 0.45], 'variation': 0.0}})
    engine.post('add', {'kind': 'sphere', 'name': 'sphere', 'params': {'radius': 0.075},
                        'translate': [-0.13, 0.09, 0.02], 'colour': {'albedo': [0.45, 0.52, 0.62], 'variation': 0.0}})
    engine.post('add', {'kind': 'cylinder', 'name': 'cylinder', 'params': {'radius': 0.05, 'height': 0.22,
                                                                           'round': 0.004},
                        'translate': [0.13, 0.0, 0.0], 'colour': {'albedo': [0.55, 0.6, 0.45], 'variation': 0.0}})
    wait_idle()
    objs = {o['name']: o for o in win.snap['objects']}
    n_added = len(objs)
    cube, sph, cyl = objs['cube']['id'], objs['sphere']['id'], objs['cylinder']['id']
    engine.post('boolean', cube, sph, 'union', 0.03)                      # Splice: smooth union, k = 30 mm
    wait_idle()
    cyl_root = [o for o in win.snap['objects'] if o['id'] == cyl][0]['base']['id']
    # a box beside the cylinder becomes part of its tree too? keep it simple: twist the cylinder's root
    engine.post('wrap', cyl, cyl_root, 'twist', {'rate': 9.0})
    # make the twist visible: a cylinder is round, so stretch its section first (non-uniform scale under the twist)
    wait_idle()
    engine.post('wrap', cyl, cyl_root, 'transform', {'scale': [1.0, 1.0, 0.45]})
    wait_idle()
    engine.post('frame', [])
    wait_idle()
    vp.cam.orbit(-18, 14)
    vp.request(False)
    wait_idle()
    snap = win.snap

    def kinds(d):
        out = [(d['kind'], d['params'])]
        for c in d.get('children', []):
            out += kinds(c)
        return out
    all_nodes = [k for o in snap['objects'] for k in kinds(o['base'])]
    checks['S3_scene'] = (n_added == 3 and len(snap['objects']) == 2
                          and any(k == 'union' and p['blend'] > 0 for k, p in all_nodes)
                          and any(k == 'twist' for k, p in all_nodes))
    res['objects_after_boolean'] = [o['name'] for o in snap['objects']]

    # -- S4 ten strokes through the viewport's stroke path ---------------------------------------------------------
    h_pre = state_hash()
    tools = ['draw', 'clay', 'carve', 'inflate', 'smooth', 'flatten', 'crease', 'pinch', 'grab', 'stretch']
    W, H = vp.width(), vp.height()
    centres = {o['id']: np.asarray(o['transform']['translate'], float) for o in snap['objects']}
    targets = [(cube, np.array([0.0, -0.02, 0.07])), (cyl, np.array([0.0, 0.0, 0.03]))]
    stroke_ms = []
    stroke_L = []
    for i, tool in enumerate(tools):
        oid, off = targets[i % 2]
        c = centres[oid] + off + np.array([0.0, 0.03 * ((i // 2) % 3 - 1), 0.0])
        x, y, z = vp.cam.project(c, W, H)
        win.tools.set_tool(tool)
        win.tools.radius.setValue(0.016, emit=True)
        win.tools.strength.setValue(0.6, emit=True)
        if i == 4:
            win.tools.sym[0].setChecked(True)          # one stroke with X symmetry
        elif i == 5:
            win.tools.sym[0].setChecked(False)
        vp.brush = win.tools.brush()
        ts = time.perf_counter()
        path = [QPointF(x - 28 + 56 * k / 12.0, y - 6 + 12 * k / 12.0) for k in range(13)]
        start = next((k for k, q in enumerate(path) if vp._obj_at(q) == oid), 6)   # press ON the form
        vp.begin_stroke(path[start])
        for q in path[start + 1:]:
            vp.move_stroke(q)
            pump(0.012)
        vp.end_stroke()
        wait_idle()
        stroke_ms.append(round((time.perf_counter() - ts) * 1000))
        stroke_L.append(round(win.snap['L'], 3))
    win.tools.set_tool('select')
    h_post = state_hash()
    log = call(lambda e: [c.to_log()['type'] for c in e.hist.done])
    n_strokes = sum(1 for t in log if t == 'BrushStroke')
    checks['S4_strokes'] = n_strokes == 10 and h_post != h_pre
    res['stroke_ms_incl_waits'] = stroke_ms
    res['scene_L_after_each_stroke'] = stroke_L
    res['log_labels'] = list(win.snap['log'])

    # -- S5 undo / redo ----------------------------------------------------------------------------------------------
    for _ in range(10):
        engine.post('undo')
    wait_idle()
    h_undo = state_hash()
    for _ in range(10):
        engine.post('redo')
    wait_idle()
    h_redo = state_hash()
    checks['S5_undo_redo'] = h_undo == h_pre and h_redo == h_post

    # -- S6 replay ---------------------------------------------------------------------------------------------------
    def rep(e):
        from ..core import replay
        t = time.perf_counter()
        h = replay(e.hist.log()).scene.state_hash()
        return h, time.perf_counter() - t
    h_rep, rep_s = call(rep)
    checks['S6_replay'] = h_rep == h_post
    res['replay_s'] = round(rep_s, 2)

    # -- S7 save / open ---------------------------------------------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix='fieldstudio_selftest_')
    path = os.path.join(tmp, 'selftest.fsproj')
    engine.post('save', path, {'camera': vp.cam.to_dict()})
    wait_idle()
    engine.post('open', path)
    wait_idle()
    h_open = state_hash()
    checks['S7_save_open'] = h_open == h_post
    res['undo_after_open'] = call(lambda e: e.hist.can_undo())

    # -- S2 final frame + S1 grabs -----------------------------------------------------------------------------------
    win.props.set_selection([cube])
    vp.request(False)
    wait_idle()
    fr = vp.frame
    obj_frac = float((fr['obj'] >= 0).mean())
    checks['S2_frame'] = fr['final'] and fr['quality'] == 2 and obj_frac > 0.05 and fr['stats']['unresolved'] == 0
    res['final_frame'] = {'res': list(fr['stats']['internal_res']), 'ms': round(fr['stats']['ms'], 1),
                          'quality': fr['quality'], 'object_fraction': round(obj_frac, 3),
                          'unresolved': fr['stats']['unresolved'], 'L': round(fr['stats']['lipschitz'], 3)}
    os.makedirs(SCREENS, exist_ok=True)
    pump(0.2)
    gap.on = False
    grab_path = os.path.join(SCREENS, 'selftest.png')
    pm = win.grab()
    pm.save(grab_path)
    vp.grab().save(os.path.join(SCREENS, 'selftest_viewport.png'))
    checks['S1_grab'] = os.path.exists(grab_path) and pm.width() >= 1000 and pm.height() >= 600
    res['grab_px'] = [pm.width(), pm.height()]
    # an extra look at the field itself (not a check)
    win.set_mode('slice')
    wait_idle()
    pump(0.1)
    vp.grab().save(os.path.join(SCREENS, 'selftest_slice.png'))
    win.set_mode('shaded')

    checks['S8_ui_gap'] = gap.max_gap < 100.0
    res['ui_max_gap_ms'] = round(gap.max_gap, 1)
    # S10: the launch progress and the Quick start panel
    steps = list(getattr(win.viewport, 'warm_steps', []))
    qs = win.show_quick_start()
    pump(0.3)
    qs_path = os.path.join(SCREENS, 'fs_quickstart.png')
    qs_ok = bool(qs is not None and qs.grab().save(qs_path))
    txt = qs.findChild(__import__('PySide6.QtWidgets', fromlist=['QTextBrowser']).QTextBrowser).toPlainText() if qs else ''
    qs.close()
    names = all(k in txt for k in ('Orbit', 'Pencil', 'Cut', 'Splice', 'Undo'))
    checks['S10_launch'] = len(steps) >= 8 and steps[-1] == 'ready' and qs_ok and names
    res['launch'] = {'progress_steps': steps, 'quick_start_grab': qs_ok, 'quick_start_names_controls': names}
    ws, peak = process_memory()
    checks['S9_memory'] = peak < 3072
    res['peak_working_set_MB'] = round(peak, 0)
    # stage 3: the demo scene with every capability, the Field Tools through the viewport, grabs, latency
    from . import demo
    dchecks, dres = demo.run(app, win, engine, {'pump': pump, 'wait_idle': wait_idle, 'call': call, 'gap': gap,
                                                'screens': SCREENS})
    checks.update(dchecks)
    res['demo'] = dres
    res['region'] = 'not in this release'
    res['hashes'] = {'pre_strokes': h_pre[:16], 'post_strokes': h_post[:16], 'after_undo10': h_undo[:16],
                     'after_redo10': h_redo[:16], 'replay': h_rep[:16], 'opened': h_open[:16]}
    res['engine_errors'] = win.errors
    res['numba_threads'] = engine.threads
    res['checks'] = {k: bool(v) for k, v in sorted(checks.items())}
    res['pass'] = all(checks.values()) and not win.errors
    res['total_s'] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(SCREENS, 'selftest.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    engine.stop()
    win.close()
    return 0 if res['pass'] else 1
