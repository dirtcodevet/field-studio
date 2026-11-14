"""The cutter's live preview (fs-shell, 2026-11-11; design 4.18: "A live preview tints the volume that will go").

While a cutter (Create > Cutter) is selected, the volume the cut would take away (target AND cutter for 'cut away:
inside', target AND NOT cutter for 'outside') is traced exactly from the fields (tools.shapecut.preview_mask) on the
engine thread, whenever the cutter, the scene, the view or the side changes (debounced), and drawn over the frame as a
red tint with an outline, through the model (x-ray: the part that goes is usually inside the target). A stale preview
(the view moved since) is not drawn. CPU or card: the trace reads the fields, not the tracer."""
import numpy as np
from PySide6.QtCore import QObject, QTimer

TINT = np.array([235.0, 60.0, 50.0])


class CutPreview:
    def __init__(self, win):
        self.win = win
        self.vp = win.viewport
        self.res = None
        self.res_key = None
        self.want_key = None
        self.busy = False
        self.on = True
        self.timer = QTimer(win)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self._run)
        win.engine.callDone.connect(self._done)
        self.stats = []

    # ---- what to preview ----------------------------------------------------------------------------------------------
    def target(self):
        """-> (cutter oid, [target oids], side) or None"""
        from ..tools.shapecut import is_cutter
        if not self.on:
            return None
        objs = {o['id']: o for o in (getattr(self.win, 'snap', None) or {}).get('objects', [])}
        sel = [o for o in self.vp.selected if o in objs]
        if not sel or not is_cutter(objs[sel[0]]):
            return None
        tg = [o for o in getattr(self.win, 'cut_targets', []) if o in objs and not is_cutter(objs[o])]
        if not tg:
            tg = [o for o in objs if not is_cutter(objs[o])]
        if not tg:
            return None
        cb = getattr(self.win, 'context_bar', None)
        side = cb.cut_side.currentData() if cb is not None else 'inside'
        return sel[0], tg, side

    def active(self):
        return self.target() is not None

    def _key(self, cam, W, H, tgt):
        from ..render.raymarch import _cam_array
        snap = getattr(self.win, 'snap', None) or {}
        return (tuple(np.round(_cam_array(cam), 9)), int(W), int(H), tgt[0], tuple(tgt[1]), tgt[2],
                snap.get('version'), len(snap.get('log', [])))

    def schedule(self, *_):
        if self.active():
            self.timer.start()
        elif self.res is not None:
            self.res = None
            self.vp.refs_changed()

    # ---- the trace on the engine thread -----------------------------------------------------------------------------
    def _run(self):
        tgt = self.target()
        fr = self.vp.frame
        if tgt is None or fr is None:
            return
        if self.busy:
            self.timer.start()
            return
        cam = fr['cam'].copy()
        H, W = np.asarray(fr['rgb']).shape[:2]
        key = self._key(cam, W, H, tgt)
        if key == self.res_key:
            return
        self.want_key = key
        self.busy = True
        cut, tg, side = tgt

        def fn(e):
            from ..tools.shapecut import preview_mask
            return preview_mask(e.hist.scene, cam, W, H, cut, tg, side)
        self.token = object()
        self.win.engine.post('call', fn, self.token)

    def _done(self, r):
        if not isinstance(r, tuple) or len(r) < 2 or r[0] is not getattr(self, 'token', None):
            return
        self.busy = False
        self.res = None if getattr(r[1], 'failed', False) else r[1]     # a failed call answers with a falsy CallFailed (fs-brush cba5ef61); the engine reports the failure itself;
        self.res_key = self.want_key                                    # no tint, and this view is not traced again
        if self.res is not None:
            self.stats.append(round(float(self.res.get('ms', 0.0)), 1))
        self.vp.refs_changed()

    # ---- drawing ---------------------------------------------------------------------------------------------------------
    def composite(self, fr, rgb):
        tgt = self.target()
        if tgt is None:
            return rgb
        H, W = rgb.shape[:2]
        key = self._key(fr['cam'], W, H, tgt)
        if key != self.res_key:
            self.schedule()                    # the view, the cutter or the scene changed: trace again
            return rgb
        r = self.res
        if not r or r['mask'].size == 0 or not r['mask'].any():
            return rgb
        s = r['step']
        mk = r['sd_px'] if r.get('sd_px') is not None else (1.0 - 2.0 * r['mask'].astype(np.float32))
        hh, ww = mk.shape[0] * s, mk.shape[1] * s
        # bilinear upsample of the traced mask: a soft, even edge (the trace is one ray per s pixels)
        fy = np.clip((np.arange(hh) + 0.5) / s - 0.5, 0, mk.shape[0] - 1)
        fx = np.clip((np.arange(ww) + 0.5) / s - 0.5, 0, mk.shape[1] - 1)
        y0_, x0_ = np.floor(fy).astype(int), np.floor(fx).astype(int)
        y1_, x1_ = np.minimum(y0_ + 1, mk.shape[0] - 1), np.minimum(x0_ + 1, mk.shape[1] - 1)
        ay, ax = (fy - y0_)[:, None], (fx - x0_)[None, :]
        m = (mk[y0_][:, x0_] * (1 - ay) * (1 - ax) + mk[y0_][:, x1_] * (1 - ay) * ax + mk[y1_][:, x0_] * ay * (1 - ax)
             + mk[y1_][:, x1_] * ay * ax)
        y0, x0 = r['y0'], r['x0']                 # ray k covers pixels [x0 + k s, x0 + (k + 1) s)
        ya, xa = max(0, y0), max(0, x0)
        yb, xb = min(H, y0 + hh), min(W, x0 + ww)
        if yb <= ya or xb <= xa:
            return rgb
        sd = m[ya - y0:yb - y0, xa - x0:xb - x0]
        cov = np.clip(0.5 - sd, 0.0, 1.0)                       # pixel coverage from the signed distance
        a_ = cov[..., None] * 0.4
        edge = np.clip(1.0 - np.abs(sd + 0.5) / 1.2, 0.0, 1.0)[..., None]   # a ~2 px line on the silhouette
        a_ = np.maximum(a_, edge * 0.85)
        out = rgb.copy()
        blk = out[ya:yb, xa:xb].astype(np.float32)
        blk = blk * (1.0 - a_) + TINT[None, None] * a_
        out[ya:yb, xa:xb] = blk.astype(np.uint8)
        return out


def conform(win, wall=None, target=None, done=None, open_margin=0.0):
    """Create > Cutter > Conform to the selected object: a cutter that is its inward offset at the wall (design 4.18:
    "hollow out a skull ... a 4mm wall"); then the wall the cut would leave, read from the field (wall_report)"""
    from ..tools.shapecut import ConformCutter, is_cutter, wall_report
    from . import units as U
    vp = win.viewport
    objs = {o['id']: o for o in (win.snap or {}).get('objects', [])}
    if target is None:
        sel = [o for o in vp.selected if o in objs and not is_cutter(objs[o])]
        target = sel[0] if sel else None
    if target is None:
        win.status.showMessage('Conform: select the object (or part) to hollow first.', 8000)
        return None
    unit = getattr(win, 'units', 'mm')
    if wall is None:
        from PySide6.QtWidgets import QInputDialog
        txt, ok = QInputDialog.getText(win, 'Conform with a wall', 'Wall thickness to keep (%s). Where you painted the '
                                       'Mask the wall is left open, its rim on your mask line; add a second length '
                                       '("4 mm, 2 mm") to open it that much wider:' % unit,
                                       text=U.format_length(0.004, unit))
        if not ok:
            return None
        parts = [t_ for t_ in txt.split(',') if t_.strip()]
        wall = U.parse_length(parts[0], unit) if parts else None
        if len(parts) > 1:
            open_margin = U.parse_length(parts[1], unit) or 0.0
        if wall is None or wall <= 0:
            win.status.showMessage('Conform: a wall thickness, e.g. 4 mm.', 8000)
            return None
    eng = win.engine
    win.cut_targets = [target]
    tok = object()

    def report(e):
        """after the push (jobs run in order): the new cutter is the scene's last object"""
        sc = e.hist.scene
        cut = sc.objects[-1] if sc.objects else None
        if cut is None or not str(cut.name).startswith('Cutter (conform'):
            return None
        rep = wall_report(sc, target, cut.id) or {}
        last = e.hist.done[-1] if e.hist.done else None
        bake = getattr(last, 'bake', None) or {}
        rep['openings'] = int(bake.get('openings', 0) or 0)       # masked surface points that open the wall
        rep['mode'] = getattr(last, 'mode', None)
        return cut.id, rep

    def reported(rr, tk=tok):
        if rr[0] is not tk:
            return
        eng.callDone.disconnect(reported)
        if getattr(rr[1], 'failed', False):      # a failed call answers with a falsy CallFailed (fs-brush cba5ef61); the engine reports the failure itself
            win.status.showMessage('Conform: the wall could not be read (the notice over the view says why); Undo '
                                   'takes the cutter back.', 12000)
            return
        if rr[1] is None:
            win.status.showMessage('Conform: the cutter was not made.', 8000)
            return
        cid, rep = rr[1]
        rep = rep or {}
        win.conform_report = rep
        if rep.get('n'):
            thin = rep.get('thin', 0)
            msg = ('Conform: a %s wall; from the surface it reads %s to %s (median %s) over %d points%s%s. Apply cut '
                   '(side inside) hollows it; Lift out keeps the inside as an object.'
                   % (U.format_length(wall, unit), U.format_length(rep['min'], unit),
                      U.format_length(rep['max'], unit), U.format_length(rep['median'], unit), rep['n'],
                      ('; THINNER than set at %d points (a feature)' % thin) if thin else '',
                      ('; the masked area is left open' if rep.get('openings') else
                       '; paint the Mask where it should stay open, then Conform again' if rep.get('mode') == 'true'
                       else '')))
        else:
            msg = 'Conform: the region is thinner than a %s wall everywhere: nothing to hollow.' % (
                U.format_length(wall, unit))
        win.status.showMessage(msg, 20000)
        if done:
            done({'cutter': cid, 'target': target, 'wall': wall, 'report': rep})
    eng.callDone.connect(reported)
    eng.post('push', ConformCutter(target, wall, mode='auto', open_margin=float(open_margin or 0.0)))   # one undo step; the cutter becomes the selection
    eng.post('call', report, tok)
    return tok


class ConformFollow(QObject):
    """a live sculpt-exact Conform cutter follows the sculpting (design: while the cutter is live, before Apply, the
    wall follows the surface as the sculpting goes on, refreshed after a stroke is released, never during one):
    when a stroke on its target has ended, the cutter is re-baked from the surface as it is now and swapped in,
    in the same undo step as the stroke"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.seen = None
        self.busy = False
        self.again = False
        self.last = None                       # the last refresh's report (tests read it)

    def live(self, snap):
        """{target id: [(cutter id, wall)]} of the live sculpt-exact Conform cutters in a snapshot"""
        import json
        from ..tools.common import TOOLDATA
        out = {}
        for o in (snap or {}).get('objects', []):
            b = o.get('base') or {}
            nm = str(b.get('name') or '')
            if b.get('kind') == 'lattice' and nm.startswith(TOOLDATA):
                try:
                    d = json.loads(nm[len(TOOLDATA):])
                except Exception:
                    continue
                if d.get('tool') == 'conform' and d.get('target_id') is not None:
                    out.setdefault(int(d['target_id']), []).append((o['id'], float(d['wall'])))
        return out

    def scene_changed(self, snap):
        ids = (snap or {}).get('log_ids') or []
        last = ids[-1] if ids else None
        if last == self.seen:
            return
        self.seen = last
        if not self.live(snap):
            return
        if self.busy:
            self.again = True
            return
        self.check()

    def check(self):
        eng = self.win.engine
        if getattr(eng, '_stroke', None) is not None:      # never during a stroke
            QTimer.singleShot(150, self.check)
            return
        live = self.live(self.win.snap)
        tok = object()
        self.busy = True

        def fn(e):
            """on the engine thread: the last step a finished stroke on a live target? -> re-bake its cutters"""
            import time as _t
            from ..core.history import BrushStroke
            from ..tools import truedist as TD, bakes as BK
            done = e.hist.done
            if not done or not isinstance(done[-1], BrushStroke) or int(done[-1].oid) not in live:
                return None
            stroke = done[-1]
            sc = e.hist.scene
            out = []
            for cid, wall in live[int(stroke.oid)]:
                t0 = _t.perf_counter()
                o = sc.get(int(stroke.oid))
                G = TD.conform_local(sc, int(stroke.oid), wall)
                bake = BK.write(G['cutter'], G['lo'], G['h'])
                out.append((cid, bake, (_t.perf_counter() - t0) * 1000))
            return {'stroke': getattr(stroke, 'serial', None), 'bakes': out}

        def done(rr, tk=tok):
            if rr[0] is not tk:
                return
            eng.callDone.disconnect(done)
            res = rr[1]
            if getattr(res, 'failed', False):    # a failed call answers with a falsy CallFailed (fs-brush cba5ef61); the engine reports the failure itself;
                res = None                         # the wall stays as it was, and the next stroke tries again
                self.win.status.showMessage('Conform: the wall could not follow this stroke (the notice over the view '
                                            'says why); it is as it was.', 12000)
            if res is not None:
                from ..tools.shapecut import RebakeConform
                for cid, bake, ms in res['bakes']:
                    eng.post('push', RebakeConform(cid, bake))
                n = len(res['bakes'])

                def group(e, serial=res['stroke'], n=n):
                    """the stroke and its re-bakes: one undo step"""
                    d = e.hist.done
                    if len(d) < n + 1 or getattr(d[-n - 1], 'serial', None) != serial:
                        return False
                    d[-1].group_n = n + 1
                    d[-n - 1].group_first_n = n + 1
                    return True
                eng.post('call', group, object())
                self.last = {'stroke': res['stroke'], 'ms': [round(b[2]) for b in res['bakes']],
                             'cutters': [b[0] for b in res['bakes']]}
                self.win.status.showMessage('Conform: the wall follows the sculpt (re-baked in %.1f s).' % (
                    sum(b[2] for b in res['bakes']) / 1000.0), 6000)
            self.busy = False
            if self.again:
                self.again = False
                QTimer.singleShot(0, self.check)
        eng.callDone.connect(done)
        eng.post('call', fn, tok)


def install(win):
    win.conform_follow = ConformFollow(win)
    win.engine.sceneChanged.connect(win.conform_follow.scene_changed)
    mb = win.menuBar()
    cm = next((a.menu() for a in mb.actions() if a.menu() and 'Create' in a.menu().title()), None)
    cut_menu = None
    if cm is not None:
        cut_menu = next((a.menu() for a in cm.actions() if a.menu() and a.menu().title().startswith('Cutter')), None)
    if cut_menu is not None:
        cut_menu.addSeparator()
        a = cut_menu.addAction('Conform to the selected object, keeping a wall (hollow it)...')
        a.triggered.connect(lambda: conform(win))
        win.conform_action = a
    cp = CutPreview(win)
    win.cut_preview = cp
    vp = win.viewport
    if not hasattr(vp, 'overlays'):
        vp.overlays = []
    vp.overlays.append(cp)
    win.engine.sceneChanged.connect(cp.schedule)
    cb = getattr(win, 'context_bar', None)
    if cb is not None:
        cb.cut_side.currentIndexChanged.connect(lambda _i: cp.schedule())
    return cp
