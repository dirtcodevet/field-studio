"""Stage 3 self-test (part of --selftest): the demo scene built through the app's own paths - engine jobs for
create / mutate / combine, Viewport.begin/move/end_stroke for brushes, and the Field Tools through the menu
callbacks of tools.register plus Viewport.mode_press/move/release (the same calls the mouse makes) - then window
grabs, the end-to-end determinism checks and the latency numbers.

Demo scene (a 4 x 2 grid, metres):
  top row     twisted cube | bent cylinder | sphere sculpted with Clay, Draw, Smooth and a detail-band smooth |
              smooth union of a capsule and a torus
  bottom row  plate with a pencil ridge and a pencil valley | cylinder cut and separated | block with a ball
              spliced on (SPLICE) | two blocks joined by a lofted pillar (FILL AND SPLICE)

THRESHOLDS (written before the first run, not moved after it)
  V1 scene     base trees hold a twist, a bend, a smooth union (blend > 0), a pencil ridge (union root) and a pencil
               valley (subtract root), a Splice node and a fill-and-splice node; the cut made exactly 1 new object;
               the sculpted sphere has sculpt bricks > 0
  V2 tool path the op log holds Cut x1, Splice x1, FillSplice x1, LayerSmooth x1 and >= 3
               BrushStroke, every one entered through the viewport's mode / stroke calls
  V3 undo/redo goto 0 -> the empty-scene hash; goto N -> the full-demo hash
  V4 replay    the op log replays into an empty scene to the same hash; save .fsproj -> open -> the same hash
  V5 frames    every grabbed frame is final (idle refine), has 0 unresolved rays and objects on > 5 % of pixels
  V6 latency   640x360 preview (q0) median < 100 ms; 1280x720 q1 median < 1500 ms; brush dab apply (radius 12 mm,
               Clay / Draw / Smooth) median < 60 ms per dab; UI event-loop gap during the demo < 100 ms
  V7 memory    peak working set of the process < 3072 MB
  V8 grabs     6 window PNGs (>= 1000 x 600) in screens/
"""
import os
import statistics
import time

import numpy as np
from PySide6.QtCore import QPointF

from ..render import Camera

TOP, BOT = 0.12, -0.14
XS = (-0.36, -0.12, 0.12, 0.36)


def run(app, win, engine, ctx):
    pump, wait_idle, call, gap = ctx['pump'], ctx['wait_idle'], ctx['call'], ctx['gap']
    SCREENS = ctx['screens']
    vp = win.viewport
    res, checks = {}, {}
    op_ms = {}
    win.ask_auto = {}

    def state_hash():
        return call(lambda e: e.hist.scene.state_hash())

    def objects():
        return {o['name']: o for o in win.snap['objects']}

    def root_id(name):
        return objects()[name]['base']['id']

    def add(name, kind, params, t, col, rot=(0.0, 0.0, 0.0)):
        engine.post('add', {'kind': kind, 'name': name, 'params': params, 'translate': list(t), 'rotate': list(rot),
                            'colour': {'albedo': col, 'variation': 0.0}})

    def set_cam(pos, target, fov=35.0):
        vp.cam = Camera(pos=pos, target=target, fov=fov)
        vp.request(False)
        wait_idle()

    def scr(p):
        x, y, z = vp.cam.project(np.asarray(p, float), vp.width(), vp.height())
        return QPointF(x, y)

    def timed(label, fn):
        t = time.perf_counter()
        fn()
        wait_idle()
        op_ms[label] = round((time.perf_counter() - t) * 1000)

    def drag_mode(points):
        vp.mode_press(scr(points[0]))
        pump(0.01)
        for p in points[1:]:
            vp.mode_move(scr(p))
            pump(0.01)
        vp.mode_release(scr(points[-1]))

    def click_mode(p):
        q = scr(p)
        vp.mode_press(q)
        vp.mode_release(q)
        wait_idle()

    def stroke(tool, pts, radius, strength, layer='sculpt'):
        win.tools.set_tool(tool)
        win.tools.set_layer(layer)
        win.tools.radius.setValue(radius, emit=True)
        win.tools.strength.setValue(strength, emit=True)
        vp.brush = win.tools.brush()
        vp.begin_stroke(scr(pts[0]))
        for p in pts[1:]:
            vp.move_stroke(scr(p))
            pump(0.012)
        vp.end_stroke()
        wait_idle()

    gap.max_gap = 0.0
    gap.on = True
    engine.post('new')
    wait_idle()
    h_empty = state_hash()

    # ---- top row: mutations, sculpt, smooth union -------------------------------------------------------------
    add('twisted cube', 'box', {'size': [0.08, 0.17, 0.08], 'round': 0.004}, (XS[0], TOP, 0), [0.62, 0.52, 0.45])
    add('bent cylinder', 'cylinder', {'radius': 0.026, 'height': 0.2, 'round': 0.003}, (XS[1], TOP, 0),
        [0.45, 0.52, 0.62])
    add('sculpted sphere', 'sphere', {'radius': 0.07}, (XS[2], TOP, 0), [0.62, 0.45, 0.5])
    add('capsule', 'capsule', {'radius': 0.03, 'height': 0.11}, (XS[3], TOP, 0), [0.55, 0.6, 0.45])
    add('torus', 'torus', {'major': 0.055, 'minor': 0.016}, (XS[3], TOP, 0), [0.55, 0.6, 0.45], rot=(90, 0, 0))
    wait_idle()
    engine.post('wrap', objects()['twisted cube']['id'], root_id('twisted cube'), 'twist', {'rate': 8.0})
    wait_idle()
    cyl = objects()['bent cylinder']['id']
    engine.post('wrap', cyl, root_id('bent cylinder'), 'transform',
                {'translate': [0.0, 0.0, 0.0], 'rotate': [0.0, 0.0, 90.0], 'scale': [1.0, 1.0, 1.0]})
    wait_idle()
    engine.post('wrap', cyl, root_id('bent cylinder'), 'bend', {'rate': 6.0})
    wait_idle()
    ob = objects()
    engine.post('boolean', ob['capsule']['id'], ob['torus']['id'], 'union', 0.02)
    wait_idle()

    # ---- bottom row: the tool objects --------------------------------------------------------------------------
    add('pencil plate', 'box', {'size': [0.17, 0.12, 0.04], 'round': 0.004}, (XS[0], BOT, 0), [0.58, 0.55, 0.42])
    add('cut cylinder', 'cylinder', {'radius': 0.05, 'height': 0.14, 'round': 0.004}, (XS[1], BOT, 0),
        [0.45, 0.58, 0.58])
    add('splice block', 'box', {'size': [0.12, 0.06, 0.08], 'round': 0.004}, (XS[2], BOT - 0.03, 0),
        [0.62, 0.52, 0.45])
    add('ball', 'sphere', {'radius': 0.033}, (XS[2], BOT + 0.10, 0.06), [0.45, 0.52, 0.62])
    add('base block', 'box', {'size': [0.12, 0.04, 0.08], 'round': 0.004}, (XS[3], BOT - 0.06, 0),
        [0.55, 0.6, 0.45])
    add('top block', 'box', {'size': [0.12, 0.04, 0.08], 'round': 0.004}, (XS[3], BOT + 0.06, 0),
        [0.62, 0.45, 0.5])
    wait_idle()
    ob = objects()

    # sculpt: Clay, Draw (fine line), Smooth (sculpt layer), Smooth on the detail band - front view of the sphere
    c = np.array([XS[2], TOP, 0.0])
    set_cam(c + [0.0, 0.0, 0.42], c)

    def on_sphere(u, v):          # a point on the sphere's front from (u, v) in its radius units
        r = 0.07
        z = np.sqrt(max(r * r - (u * r) ** 2 - (v * r) ** 2, 0.0))
        return c + [u * r, v * r, z]
    clay = [on_sphere(-0.55 + 1.1 * k / 12, 0.25 + 0.15 * np.sin(k / 2.0)) for k in range(13)]
    timed('clay stroke', lambda: stroke('clay', clay, 0.016, 0.7))
    line = [on_sphere(-0.5 + 1.0 * k / 12, -0.3) for k in range(13)]
    timed('draw stroke (fine)', lambda: stroke('draw', line, 0.005, 0.8))
    half = [on_sphere(0.05 + 0.5 * k / 8, 0.28) for k in range(9)]
    timed('smooth stroke', lambda: stroke('smooth', half, 0.018, 0.8))
    half2 = [on_sphere(0.05 + 0.45 * k / 8, -0.3) for k in range(9)]
    timed('smooth detail-band stroke', lambda: stroke('smooth', half2, 0.012, 1.0, layer='detail'))
    win.tools.set_layer('sculpt')
    win.tools.set_tool('select')

    # PENCIL ridge and valley across the plate's front face
    pc = np.array([XS[0], BOT, 0.0])
    set_cam(pc + [0.0, 0.05, 0.42], pc)
    spec = {'width': 0.008, 'depth': 0.004, 'softness': 0.002, 'profile': 'round'}
    win.ask_auto = {'Pencil (ridge)': spec, 'Pencil (valley)': dict(spec, profile='V')}
    ridge = [pc + [-0.065 + 0.13 * k / 16, 0.025 + 0.012 * np.sin(k / 2.5), 0.02] for k in range(17)]
    valley = [pc + [-0.065 + 0.13 * k / 16, -0.028 + 0.010 * np.cos(k / 2.0), 0.02] for k in range(17)]
    win.escape()

    # CUT AND SEPARATE: an outline over the cylinder's top with a slanted lower edge, drawn from the left-front so
    # the cap moves left and forward (toward that eye); the piece is then recoloured (an ordinary object edit)
    cc = np.array([XS[1], BOT, 0.0])
    set_cam(cc + [-0.30, 0.06, 0.32], cc)
    win.props.set_selection([ob['cut cylinder']['id']])
    pump(0.05)
    n_before = len(win.snap['objects'])
    win.ask_auto['Cut (separate)'] = {'distance': 0.06, 'kerf': 0.0, 'blend': 0.0}
    f_, r_, u_ = vp.cam.basis()
    quad = [cc + r_ * a + u_ * b for a, b in ((-0.10, 0.035), (0.10, 0.0), (0.10, 0.13), (-0.10, 0.13))]
    outline = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        outline += [a + (b - a) * k / 8 for k in range(8)]
    win.field_tools['Cut and separate']()
    timed('cut and separate', lambda: drag_mode(outline))
    win.escape()
    n_cut_new = len(win.snap['objects']) - n_before
    piece = [o for o in win.snap['objects'] if o['name'].endswith('piece')]
    if piece:
        engine.post('props', piece[0]['id'], {'colour': {'albedo': [0.70, 0.62, 0.40], 'variation': 0.0}})
        wait_idle()
    win.props.set_selection([])

    # SPLICE: the block's top face (A), then the ball's front (B)
    sc = np.array([XS[2], BOT, 0.0])
    set_cam(sc + [0.0, 0.29, 0.40], sc)
    win.ask_auto['Splice'] = {'blend': 0.008}
    win.field_tools['Splice']()
    t = time.perf_counter()
    click_mode([XS[2], BOT, 0.0])                                   # block top centre
    click_mode([XS[2], BOT + 0.10, 0.06 + 0.033])                   # ball front
    op_ms['splice (2 clicks)'] = round((time.perf_counter() - t) * 1000)
    win.escape()

    # FILL AND SPLICE: the base block's top face (A, seen from above) and the top block's underside (B, from below)
    fc = np.array([XS[3], BOT, 0.0])
    win.ask_auto['Fill and splice'] = {'blend': 0.004, 'radius_a': 0.018, 'radius_b': 0.011}
    win.field_tools['Fill and splice']()
    a_pt = np.array([XS[3], BOT - 0.04, 0.0])
    b_pt = np.array([XS[3], BOT + 0.04, 0.0])
    t = time.perf_counter()
    set_cam(a_pt + [0.0, 0.12, 0.45], a_pt)
    click_mode(a_pt)
    set_cam(b_pt + [0.0, -0.12, 0.45], b_pt)
    click_mode(b_pt)
    op_ms['fill and splice (2 clicks)'] = round((time.perf_counter() - t) * 1000)
    win.escape()
    wait_idle()
    gap.on = False

    # ---- V1 / V2 ------------------------------------------------------------------------------------------------
    h_full = state_hash()
    snap = win.snap

    def nodes(d):
        out = [d]
        for ch in d.get('children', []):
            out += nodes(ch)
        return out
    alln = [n for o in snap['objects'] for n in nodes(o['base'])]
    names = [str(n.get('name') or '') for n in alln]
    kinds = [n['kind'] for n in alln]
    ridge_ok = any(n['kind'] == 'union' and str(n.get('name', '')).startswith('Pencil') for n in alln)
    valley_ok = any(n['kind'] == 'subtract' and str(n.get('name', '')).startswith('Pencil') for n in alln)
    sphere = [o for o in snap['objects'] if o['name'] == 'sculpted sphere'][0]
    checks['V1_scene'] = ('twist' in kinds and 'bend' in kinds
                          and any(n['kind'] == 'union' and n['params'].get('blend', 0) > 0 for n in alln)
                          and any(s.startswith('Splice') for s in names)
                          and any('fill' in s.lower() for s in names) and n_cut_new == 1
                          and sphere['bricks'] > 0)
    res['scene_objects'] = [o['name'] for o in snap['objects']]
    res['tool_node_names'] = sorted({s.split(':')[0] if s.startswith('tooldata') else s for s in names
                                     if s and not s.startswith('from ') and s not in kinds})[:40]
    res['cut_new_objects'] = n_cut_new
    res['sphere_bricks'] = sphere['bricks']
    log_types = call(lambda e: [c.to_log()['type'] for c in e.hist.done])
    cnt = {k: log_types.count(k) for k in sorted(set(log_types))}
    res['log_counts'] = cnt
    checks['V2_tool_path'] = (cnt.get('Cut', 0) == 1
                              and cnt.get('Splice', 0) == 1 and cnt.get('FillSplice', 0) == 1
                              and cnt.get('LayerSmooth', 0) == 1 and cnt.get('BrushStroke', 0) >= 3)
    res['op_ms_incl_render'] = op_ms

    # ---- grabs (the camera set, then the window) ----------------------------------------------------------------
    grabs, frames = [], []

    def grab(name):
        vp.request(False)
        wait_idle()
        pump(0.15)
        fr = vp.frame
        frames.append({'name': name, 'final': bool(fr['final']), 'quality': fr['quality'],
                       'res': list(fr['stats']['internal_res']), 'ms': round(fr['stats']['ms'], 1),
                       'unresolved': int(fr['stats']['unresolved']),
                       'object_fraction': round(float((fr['obj'] >= 0).mean()), 3),
                       'L': round(fr['stats']['lipschitz'], 3)})
        path = os.path.join(SCREENS, name)
        pm = win.grab()
        pm.save(path)
        grabs.append({'path': path, 'px': [pm.width(), pm.height()]})

    win.set_mode('shaded')
    win.ws.setCurrentIndex(0)
    win.props.set_selection([])
    engine.post('frame', [])
    wait_idle()
    vp.cam.orbit(-14, 10)
    vp.cam.zoom(0.92)
    grab('fs_overview.png')

    win.props.set_selection([objects()['pencil plate']['id']])
    set_cam(pc + [0.10, 0.07, 0.30], pc)
    grab('fs_pencil.png')

    win.props.set_selection([objects()['sculpted sphere']['id']])
    mid = np.array([(XS[1] + XS[2]) / 2 - 0.06, TOP, 0.0])
    set_cam(mid + [0.06, 0.08, 0.62], mid)
    win.tools.set_tool('clay')
    win.tools.radius.setValue(0.014, emit=True)
    vp.brush = win.tools.brush()
    hp = on_sphere(-0.2, 0.0)
    q = scr(hp)
    x, y, v = vp.to_render(q)
    engine.request_hover(v, x, y)
    wait_idle()
    grab('fs_sculpt.png')
    win.tools.set_tool('select')
    vp.hover = None

    win.props.set_selection([])
    mid = np.array([(XS[1] + XS[2]) / 2 - 0.02, BOT, 0.0])
    set_cam(mid + [0.10, 0.16, 0.55], mid)
    grab('fs_cut_splice.png')

    fill_obj = [o for o in win.snap['objects'] if o['name'] in ('base block', 'top block')]
    win.props.set_selection([fill_obj[0]['id']] if fill_obj else [])
    set_cam(fc + [0.17, 0.08, 0.36], fc)
    grab('fs_fill_splice.png')

    win.props.set_selection([])
    set_cam(fc + [0.0, 0.0, 0.45], fc)
    win.set_mode('slice')
    grab('fs_slice.png')
    win.set_mode('shaded')
    res['grabs'] = grabs
    res['frames'] = frames
    checks['V5_frames'] = all(f['final'] and f['unresolved'] == 0 and f['object_fraction'] > 0.05
                              for f in frames if f['name'] != 'fs_slice.png') and bool(frames)
    checks['V8_grabs'] = len(grabs) == 6 and all(os.path.exists(g['path']) and g['px'][0] >= 1000 and
                                                 g['px'][1] >= 600 for g in grabs)

    # ---- V3 undo / redo, V4 replay + save / open -----------------------------------------------------------------
    n_done = call(lambda e: len(e.hist.done))
    engine.post('goto', 0)
    wait_idle()
    h0 = state_hash()
    engine.post('goto', n_done)
    wait_idle()
    h1 = state_hash()
    checks['V3_undo_redo'] = h0 == h_empty and h1 == h_full
    res['commands'] = n_done

    def rep(e):
        from ..core import replay
        t = time.perf_counter()
        h = replay(e.hist.log()).scene.state_hash()
        return h, time.perf_counter() - t
    h_rep, rep_s = call(rep)
    import tempfile
    path = os.path.join(tempfile.mkdtemp(prefix='fieldstudio_demo_'), 'demo.fsproj')
    engine.post('save', path, {'camera': vp.cam.to_dict(), 'app': 'Field Studio V1 demo'})
    wait_idle()
    engine.post('open', path)
    wait_idle()
    h_open = state_hash()
    checks['V4_replay_save_open'] = h_rep == h_full and h_open == h_full
    res['replay_s'] = round(rep_s, 2)
    res['hashes'] = {'empty': h_empty[:16], 'full': h_full[:16], 'goto0': h0[:16], 'gotoN': h1[:16],
                     'replay': h_rep[:16], 'opened': h_open[:16]}
    res['demo_project'] = path

    # ---- V6 latency (engine thread, the demo scene, the overview camera) ----------------------------------------
    engine.post('frame', [])
    wait_idle()
    cam = vp.cam.copy()
    cam.orbit(-14, 10)

    def lat(e):
        from ..render import render
        out = {}
        sc = e.hist.scene
        for w, h, q, n in ((640, 360, 0, 9), (640, 360, 1, 5), (640, 360, 2, 3), (1280, 720, 1, 3),
                           (1280, 720, 2, 3)):
            wall, st = [], None
            for _ in range(n):
                t = time.perf_counter()
                r = render(sc, cam, w, h, quality=q)
                wall.append((time.perf_counter() - t) * 1000)
                st = r.stats
            out['%dx%d_q%d' % (w, h, q)] = {'median_ms': round(statistics.median(wall), 1),
                                            'min_ms': round(min(wall), 1), 'max_ms': round(max(wall), 1),
                                            'internal_res': list(st['internal_res']), 'unresolved':
                                                int(st['unresolved']), 'n': n}
        return out
    res['latency_frames'] = call(lat)

    def dabs(e):
        from ..core import History, AddObject, BrushStroke
        hh = History()
        o = hh.push(AddObject({'kind': 'sphere', 'params': {'radius': 0.07}}))
        out = {}
        for tool in ('clay', 'draw', 'smooth', 'inflate', 'crease'):
            per = []
            for rep_i in range(3):
                ang = np.linspace(-0.5, 0.5, 12) + 0.15 * rep_i
                smp = [{'p': [0.07 * np.sin(a), 0.02 * rep_i - 0.02, 0.07 * np.cos(a)],
                        'n': [np.sin(a), 0.0, np.cos(a)]} for a in ang]
                t = time.perf_counter()
                hh.push(BrushStroke(o.id, {'tool': tool, 'radius': 0.012, 'strength': 0.5}, smp))
                per.append((time.perf_counter() - t) * 1000 / len(smp))
            out[tool] = {'median_ms_per_dab': round(statistics.median(per), 2), 'max_ms_per_dab': round(max(per), 2)}
        return out
    res['latency_dabs'] = call(dabs)

    def tool_cmds(e):
        from ..core import History, Command
        hh = History()
        out = {}
        for d in e.hist.log():
            cmd = Command.from_log(d)
            t = time.perf_counter()
            hh.push(cmd)
            ms = (time.perf_counter() - t) * 1000
            out.setdefault(d['type'], []).append(round(ms, 1))
        return out
    res['command_apply_ms'] = call(tool_cmds)
    lf = res['latency_frames']
    dab_meds = [res['latency_dabs'][k]['median_ms_per_dab'] for k in ('clay', 'draw', 'smooth')]
    res['ui_max_gap_ms_demo'] = round(gap.max_gap, 1)
    checks['V6_latency'] = (lf['640x360_q0']['median_ms'] < 100 and lf['1280x720_q1']['median_ms'] < 1500
                            and statistics.median(dab_meds) < 60 and gap.max_gap < 100)
    from .main_window import process_memory
    ws, peak = process_memory()
    res['peak_working_set_MB'] = round(peak, 0)
    checks['V7_memory'] = peak < 3072
    win.ask_auto = None
    return checks, res
