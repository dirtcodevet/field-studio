"""Library > Mouth: pick a pre-built item (dentition, one arch, gums, the mouth set), set its seedable settings, and it is
placed in the scene as field objects you can move and refine by hand (fs-mouth lane, 11-11; design: a
long library of pre-built items, teeth, jaw and tongue first: a click gives the item, then it is placed by eye).

The items are built by engine/library/mouth/library_items.py on a worker thread (cached by their settings, so a second
pick is instant) and added through the ordinary AddObject spec: a 'lattice' base (a true-distance grid), rotate
[-90, 0, 0] (the item's z-up frame into Field Studio's y-up, front +z), a part record so the pieces touch without fusing
(fs-contact's touch pairs: a stroke on the gums never pushes into the teeth). Placement: the upper incisal point goes to
the mouth's usual place in the scene's head (from the scene's bounds: ~155 mm below the top, ~13 mm behind the front
surface), then Move (W) places it exactly.

Installed from main_window.py's plug-in block (a guarded `install(self)`, like the other *_ui modules).
"""
import os
import sys
import threading
import traceback

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_MOUTH = os.path.join(_ROOT, 'engine', 'library', 'mouth')


def _lib():
    if _MOUTH not in sys.path:
        sys.path.insert(0, _MOUTH)
    import library_items
    return library_items


class _Relay(QObject):
    done = Signal(object)
    progress = Signal(str)


def mouth_anchor(win):
    """the upper incisal point's place (world, metres) in the scene's head, probed on its surface: the top (vertex)
    along +y through the scene's middle, the mouth ~155 mm below it (an adult woman's vertex-to-stomion, E), the
    incisal point ~13 mm behind the front surface there (the upper lip's thickness, Arnett 1999 women 12.6); the empty
    scene: the origin"""
    import numpy as np
    try:
        sc = win.engine.hist.scene
        if not sc.objects:
            return [0.0, 0.0, 0.0]
        lo, hi = sc.bounds()
        cx, cz = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2])
        ys = np.arange(lo[1], hi[1], 0.0005)
        f = sc.evaluate(np.stack([np.full_like(ys, cx), ys, np.full_like(ys, cz)], 1))
        ins = np.where(f < 0)[0]
        ytop = float(ys[ins[-1]]) if len(ins) else float(hi[1])
        ym = ytop - 0.155
        zs = np.arange(lo[2], hi[2], 0.0005)
        f = sc.evaluate(np.stack([np.full_like(zs, cx), np.full_like(zs, ym), zs], 1))
        ins = np.where(f < 0)[0]
        zf = float(zs[ins[-1]]) if len(ins) else float(hi[2])
        return [float(cx), float(ym), float(zf - 0.013)]
    except Exception:
        return [0.0, 0.0, 0.0]


def head_lips(evaluate, lo, hi):
    """the lips on a head's front profile (world: y up, z front, metres): {'stomion', 'labrale_sup', 'pronasale',
    'vertex'} as (x, y, z), or None where the head has no lips to find (a sphere, a rough head). The vertex is the
    highest inside point; the midline the middle of the head's width below it; the profile the frontmost inside point
    per height; the nose tip its most anterior point 90-170 mm below the vertex; labrale superius the first bulge
    10-32 mm below the nose tip, the stomion the deepest notch within 16 mm below that (fs-mouth 11-12: the mouth set
    had landed in front of her lips by a fixed 155 mm / 13 mm rule)."""
    import numpy as np
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ev = lambda P: np.nan_to_num(np.asarray(evaluate(np.asarray(P, float)), float), nan=1.0)
    xs = np.arange(lo[0], hi[0], 0.004)
    zs = np.arange(lo[2], hi[2], 0.004)
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    top = None
    for y in np.arange(hi[1], lo[1], -0.002):
        f = ev(np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], 1))
        if (f < 0).any():
            k = int(np.argmin(f))
            top = (float(X.ravel()[k]), float(y), float(Z.ravel()[k]))
            break
    if top is None:
        return None
    xv, yt, zv = top
    yy = np.arange(yt + 0.004, yt - 0.02, -0.0005)
    f = ev(np.stack([np.full_like(yy, xv), yy, np.full_like(yy, zv)], 1))
    ins = np.where(f < 0)[0]
    yt = float(yy[ins[0]]) if len(ins) else yt
    # the midline: the middle of the head's width 80 mm below the vertex
    xx = np.arange(xv - 0.13, xv + 0.13, 0.0005)
    f = ev(np.stack([xx, np.full_like(xx, yt - 0.08), np.full_like(xx, zv)], 1))
    ins = np.where(f < 0)[0]
    if len(ins) < 20:
        return None
    xc = float(0.5 * (xx[ins[0]] + xx[ins[-1]]))
    # the front profile on the midline
    ys = np.arange(yt - 0.06, yt - 0.27, -0.0005)
    zz = np.arange(zv - 0.06, zv + 0.22, 0.00025)
    Y, Zq = np.meshgrid(ys, zz, indexing='ij')
    F = ev(np.stack([np.full(Y.size, xc), Y.ravel(), Zq.ravel()], 1)).reshape(Y.shape)
    inside = F < 0
    front = np.where(inside.any(1), zz[inside.shape[1] - 1 - np.argmax(inside[:, ::-1], 1)], np.nan)
    if np.isnan(front).mean() > 0.3:
        return None
    fr = np.copy(front)
    ok = np.isfinite(fr)
    fr[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], fr[ok]) if ok.sum() > 2 else 0.0
    k3 = np.ones(3) / 3.0
    fr = np.convolve(np.pad(fr, 1, mode='edge'), k3, mode='valid')
    dy = yt - ys                                                  # depth below the vertex (m)
    nose = np.where((dy > 0.09) & (dy < 0.17))[0]
    jn = int(nose[np.argmax(fr[nose])])
    def band(j0, a, b):
        return np.where((dy - dy[j0] > a) & (dy - dy[j0] < b))[0]
    # the profile's sequence below the nose tip: subnasale (the hollow under the nose), labrale superius (the upper
    # lip's bulge), the stomion (the notch between the lips), the lower lip in front again; each must stand out by
    # ~1 mm (11-12: a flat hollow's equal neighbours had passed a plain local-maximum test)
    sn = band(jn, 0.005, 0.025)
    if not len(sn):
        return None
    jsn = int(sn[np.argmin(fr[sn])])
    ul = band(jsn, 0.002, 0.016)
    if not len(ul):
        return None
    jl = int(ul[np.argmax(fr[ul])])
    lip = band(jl, 0.0015, 0.012)
    if not len(lip):
        return None
    js = int(lip[np.argmin(fr[lip])])
    after = band(js, 0.001, 0.014)
    if (fr[jl] - fr[jsn] < 0.0008 or fr[jl] - fr[js] < 0.001 or not len(after)
            or fr[after].max() - fr[js] < 0.0008):
        return None
    return {'stomion': (xc, float(ys[js]), float(fr[js])), 'labrale_sup': (xc, float(ys[jl]), float(fr[jl])),
            'pronasale': (xc, float(ys[jn]), float(fr[jn])), 'vertex': (xc, yt, zv)}


class LibraryDialog(QDialog):
    def __init__(self, win, item):
        super().__init__(win)
        LI = _lib()
        self.win, self.item, self.LI = win, item, LI
        self.setWindowTitle('Library: %s' % LI.ITEMS[item]['name'])
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.ctl = {}
        for k, label, d, lo, hi, st in LI.SETTINGS:
            if isinstance(d, int):
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setValue(int(d))
            else:
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(st)
                w.setDecimals(2)
                w.setValue(d)
            self.ctl[k] = w
            form.addRow(label, w)
        lay.addLayout(form)
        row = QHBoxLayout()
        rnd = QPushButton('Random seed')
        rnd.clicked.connect(self._random)
        row.addWidget(rnd)
        row.addStretch(1)
        lay.addLayout(row)
        self.note = QLabel('Built on the CPU and cached: the first pick of a new setting takes ~1-3 min.')
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('Place')
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _random(self):
        import random
        self.ctl['seed'].setValue(random.randint(0, 9999))

    def settings(self):
        return {k: w.value() for k, w in self.ctl.items()}


def place(win, item, settings=None, anchor=None, wait=False):
    """build (worker thread) and add the item's parts; returns the thread (wait=True joins it, for scripts)"""
    LI = _lib()
    relay = getattr(win, '_library_relay', None)
    if relay is None:
        relay = _Relay()
        win._library_relay = relay

        def _done(res):
            if isinstance(res, Exception):
                win.status.showMessage('Library: the build failed: %r' % (res,), 15000)
                return
            item_, parts, anc = res
            for sp in LI.specs(parts, anc):
                win.engine.post('add', sp)
            win.status.showMessage('Library: placed %s (%d parts). Move (W) to place it exactly; the parts touch '
                                   'without fusing.' % (LI.ITEMS[item_]['name'], len(parts)), 15000)
        relay.done.connect(_done)
        relay.progress.connect(lambda m: win.status.showMessage('Library: ' + m, 60000))
    lips = None
    if anchor is None:
        try:
            sc = win.engine.hist.scene
            if sc.objects:
                lo_, hi_ = sc.bounds()
                lips = head_lips(sc.evaluate, lo_, hi_)
        except Exception:
            traceback.print_exc()
            lips = None
    anc = list(anchor) if anchor is not None else (None if lips else mouth_anchor(win))

    def work():
        try:
            parts = LI.build(item, settings or {}, progress=relay.progress.emit)
            a_ = anc if anc is not None else LI.anchor_from_lips(lips, settings or {})
            relay.done.emit((item, parts, a_))
        except Exception as e:
            traceback.print_exc()
            relay.done.emit(e)
    th = threading.Thread(target=work, daemon=True, name='library-build')
    th.start()
    if wait:
        th.join()
    return th


def install(win):
    LI = _lib()
    menu = None
    for a in win.menuBar().actions():
        if a.text().replace('&', '') == 'Library':
            menu = a.menu()
    # (11-12) 'Li&brary': Alt+L is View > Reference lens (fs-bench-full F2 found the clash); Alt+B is free
    if menu is None:
        menu = win.menuBar().addMenu('Li&brary')
    menu.setTitle('Li&brary')
    mouth = menu.addMenu('Mouth')
    for item, it in LI.ITEMS.items():
        act = mouth.addAction(it['name'] + ' ...')

        def run(_=False, item=item):
            d = LibraryDialog(win, item)
            if d.exec():
                place(win, item, d.settings())
        act.triggered.connect(run)
    win.library_menu = menu
