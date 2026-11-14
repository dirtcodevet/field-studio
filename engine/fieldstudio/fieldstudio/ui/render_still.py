"""Render > This view, converged at full resolution: the window's own look (the card view) converged to a
fixed number of passes per pixel at up to 3840 px, opened in the full-resolution viewer (ui/still_viewer.py) and
saved as a PNG in the stills folder (FIELDSTUDIO_STILLS, default ~/FieldStudio/stills)."""
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STILLS = os.environ.get('FIELDSTUDIO_STILLS') or os.path.join(os.path.expanduser('~'), 'FieldStudio', 'stills')


VIEW_PASSES = 64                    # passes per pixel (the window's view converges at 16)


def converge_view(e, cam, W, H, passes=VIEW_PASSES):
    """on the engine thread (Engine.post('call', ...)): the card's view of the scene at W x H, converged to `passes`
    per pixel with the window's own light; -> (H, W, 3) uint8, or None without the card. The viewport restarts its
    own accumulation afterwards (its key is cleared)."""
    gv = getattr(e, 'gpu_view', None)
    if gv is None:
        return None
    from ..gpu.viewport import _with_clips
    G = gv.gpu
    with e.lock:
        sc = e.hist.scene
        gv.sync_if_changed(sc)
        cam2 = _with_clips(cam, sc)
        light = dict(gv.light or {})
        old = G.TARGET
        try:
            G.TARGET = int(passes)
            G.restart(cam2, W, H, light=light)
            left, n = 0, 0
            while left < G.TARGET and n < 4 * passes:
                left = G.progress(cam2, W, H, rect=(0, 0, W, H), passes=8, light=light)
                n += 8
            rgb = G.cp.ascontiguousarray(G.d_out.view(G.cp.uint8).reshape(H, W, 4)[:, :, 2::-1]).get()
        finally:
            G.TARGET = old
            gv.key = None
            gv.pending = None
    return np.ascontiguousarray(rgb)


def converge_window_view(win, scale=2.0, max_side=3840):
    """the window's side: View converged at full resolution -> the full-resolution viewer (and a PNG beside the stills)"""
    from PySide6.QtGui import QImage
    vp = win.viewport
    dpr = float(getattr(vp, 'devicePixelRatioF', lambda: 1.0)())
    W, H = int(vp.width() * dpr * scale), int(vp.height() * dpr * scale)
    k = min(1.0, max_side / float(max(W, H)))
    W, H = max(64, int(W * k)), max(64, int(H * k))
    cam = vp.cam.copy()
    try:
        win.status.showMessage('Converging this view at %d x %d px (%d passes) ...' % (W, H, VIEW_PASSES), 20000)
    except Exception:
        pass
    box = {}
    tok = object()

    def done(r):
        if r[0] is not tok:
            return
        try:
            win.engine.callDone.disconnect(done)
        except Exception:
            pass
        rgb = r[1]
        if rgb is None or not hasattr(rgb, 'shape'):
            win.status.showMessage('Converged view: needs the card view (FIELDSTUDIO_GPU on): %s' % (rgb,), 20000)
            return
        os.makedirs(STILLS, exist_ok=True)
        p = os.path.join(STILLS, 'window_view_%s.png' % time.strftime('%Y%m%d_%H%M%S'))
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
        img.save(p)
        from .still_viewer import StillViewer
        v = StillViewer(win, p, title='This view, converged (%d x %d)' % (rgb.shape[1], rgb.shape[0]))
        v.show()
        win._still_viewers = getattr(win, '_still_viewers', []) + [v]
        win.status.showMessage('Converged view saved: %s' % p, 30000)
        vp.request(False)
    win.engine.callDone.connect(done)
    win.engine.post('call', lambda e: converge_view(e, cam, W, H), tok)
    return True
