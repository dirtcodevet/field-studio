"""Create > Mass > Soften the joins (fs-shell, 2026-11-11; test 1 with the block-in kit).

Masses meet in creases: the smooth union rounds each join over the blend, and a larger blend bulges the form (it adds
volume at every join: kit_2's sweep, face blend 15 -> 40 mm moved the face outline 1.9 -> 7.5 mm and filled the eye
sockets). A sculptor smooths along the join instead. Here the joins are found exactly from the masses' own fields
(tools/masses.find_joins: the surface points where the two nearest forms are equally near) and stroked with the
Smooth brush along them, at the radius and strength set. Each run of a join is stroked from a camera facing it
(its points' mean normal), so joins behind or under the head are reached without turning the view; strokes go through
the engine's normal stroke path (the card brushes, replay), grouped as one undo step."""
import numpy as np


def stroke_runs(win, oid, runs, radius, strength, passes=1):
    """post one Smooth stroke per run (passes times, back and forth as a sculptor rubs a join), each seen from a camera
    facing it -> the number of strokes posted"""
    from ..render.camera import Camera
    vp = win.viewport
    eng = win.engine
    base = vp.view()
    W, H = base['w'], base['h']
    s = dict(vp.brush or {})
    s.update({'tool': 'smooth', 'radius': float(radius), 'strength': float(strength), 'symmetry': [False] * 3,
              'layer': 'sculpt'})
    s.pop('invert', None)
    n = 0
    for run in runs:
        P = np.asarray(run['pts'], float)
        nrm = np.asarray(run['normal'], float)
        c = P.mean(0)
        ext = float(np.max(np.linalg.norm(P - c, axis=1))) if len(P) > 1 else radius
        D = max(0.25, 3.0 * ext + 4.0 * radius)
        up = np.array([0.0, 1.0, 0.0]) if abs(nrm[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        cam = Camera(c + nrm * D, c, up, vp.cam.fov, False)
        v = dict(base, cam=cam, selected=[oid], tool='smooth')
        px = [cam.project(p, W, H)[:2] for p in P]
        for k in range(max(1, int(passes))):
            seq = px if k % 2 == 0 else px[::-1]
            eng.post('stroke_begin', v, float(seq[0][0]), float(seq[0][1]), s)
            for x, y in seq[1:]:
                eng.post('stroke_move', v, float(x), float(y))
            eng.post('stroke_end')
            n += 1
    return n


def soften(win, oid=None, radius=None, strength=0.6, done=None, passes=3):
    """find the selected object's joins on the engine thread, then stroke them; done(report) when the strokes are
    posted"""
    from ..tools.masses import find_joins
    vp = win.viewport
    oid = oid if oid is not None else (vp.selected[0] if vp.selected else None)
    if oid is None:
        win.status.showMessage('Soften the joins: select a massed object first.', 6000)
        return None
    eng = win.engine
    tok = object()
    r = 0.02 if radius is None else float(radius)

    def fn(e):
        res = find_joins(e.hist.scene, oid, step=0.002, radius=r)
        res['n0'] = len(e.hist.done)
        return res

    def group(e, n0):
        """the strokes as ONE undo step (the engine's grouped undo: group_n on the last, group_first_n on the first)"""
        cmds = list(e.hist.done[n0:])
        for c in cmds:
            c.ui_label = 'Soften the joins'
        if len(cmds) > 1:
            cmds[-1].group_n = len(cmds)
            cmds[0].group_first_n = len(cmds)
        return len(cmds)

    def got(rr, tk=tok):
        if rr[0] is not tk:
            return
        eng.callDone.disconnect(got)
        res = rr[1] or {}
        runs = res.get('runs') or []
        if not runs:
            win.status.showMessage('Soften the joins: no joins found (an object with masses has them).', 8000)
            if done:
                done(dict(res, strokes=0))
            return
        n = stroke_runs(win, oid, runs, r, strength, passes)
        pairs = sorted({tuple(x['pair']) for x in runs})
        tok2 = object()

        def grouped(rr2, tk2=tok2):
            if rr2[0] is not tk2:
                return
            eng.callDone.disconnect(grouped)
            win.status.showMessage('Softened %d join%s between masses: %d Smooth strokes (%d passes), radius %.0f mm; one '
                                   'undo step.' % (len(pairs), '' if len(pairs) == 1 else 's', n, passes, r * 1000),
                                   12000)
            if done:
                done(dict(res, strokes=n, pairs=pairs, applied=rr2[1]))
        eng.callDone.connect(grouped)
        eng.post('call', lambda e: group(e, res['n0']), tok2)    # runs after the strokes (jobs run in order)
    eng.callDone.connect(got)
    eng.post('call', fn, tok)
    return tok


def ask(win):
    from PySide6.QtWidgets import QInputDialog
    from ..tools.masses import mass_blend
    vp = win.viewport
    if not vp.selected:
        win.status.showMessage('Soften the joins: select a massed object first.', 6000)
        return
    objs = {o['id']: o for o in (win.snap or {}).get('objects', [])}
    o = objs.get(vp.selected[0])
    k = mass_blend(o['base']) if o is not None and o.get('base') else None
    r0 = round((k or 0.015) * 1500.0)            # 1.5 x the blend, in mm
    r, ok = QInputDialog.getInt(win, 'Soften the joins', 'Smooth radius along the joins (mm):', r0, 2, 200)
    if not ok:
        return
    s, ok = QInputDialog.getDouble(win, 'Soften the joins', 'Strength (0-1):', 0.6, 0.05, 1.0, 2)
    if not ok:
        return
    soften(win, radius=r / 1000.0, strength=s)


def install(win):
    mb = win.menuBar()
    cm = next((a.menu() for a in mb.actions() if a.menu() and 'Create' in a.menu().title()), None)
    mm = None
    if cm is not None:
        mm = next((a.menu() for a in cm.actions() if a.menu() and a.menu().title().replace('&', '').startswith('Mass')), None)
    target = mm or cm
    if target is None:
        return None
    target.addSeparator()
    a = target.addAction('Soften the joins (Smooth along where masses meet)...')
    a.triggered.connect(lambda: ask(win))
    win.joins_action = a
    return a
