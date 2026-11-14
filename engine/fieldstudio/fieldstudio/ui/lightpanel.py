"""The Light panel (fs-light, 11-11; design: the lighting is owned from the
tool set, with the tools to adjust, refine and test the shared lighting within the tool).

View > Light: a dock beside Properties. Nothing changes until "Rig" is picked at the top: the view keeps its sculpting
key (the light that follows the camera) and its headlight / cavity aid.
  - Light from: View key (sculpting) | Rig
  - Quality: Sculpt (the plain light, fast) | Look-dev (bounce and the studio's area lights, built on idle frames;
    live when the viewport carries the light runtime hook)
  - Presets (Rembrandt, classic Rembrandt, loop, butterfly, clamshell, split, three-point, rim, high / low key),
    placed around what the view looks at
  - The key: direction (azimuth, elevation about her), colour temperature, strength, size (the sun disc's diameter)
  - The environment: the HDRI, its rotation, the background's brightness; exposure in stops
  - The rig's lights: on / off, solo
  - A / B: store two rigs, flip between them, or Compare: both rendered at this view, side by side
  - Save / Load setups (JSON)
The model under it is engine/light/rig_controller.py (no Qt there; its tests run headless). Card work goes through the
engine's 'call' jobs (a rule: never from the UI thread)."""
import importlib
import importlib.util
import math
import os
import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDockWidget, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QPushButton, QRadioButton, QScrollArea, QVBoxLayout,
                               QWidget)

from .widgets import FloatField

HERE = os.path.dirname(os.path.abspath(__file__))
LIGHT_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'light'))
HDRI_ROOT = os.environ.get('FIELDSTUDIO_HDRI_DIR') or os.path.join(os.path.expanduser('~'), 'FieldStudio', 'hdri')
RIG_DIR = os.path.join(os.path.expanduser('~'), 'FieldStudio', 'light rigs')
# decided 11-12: if the look-dev card fault (app4) is not named and fixed by the morning,
# Look-dev is greyed with a plain reason so it cannot crash the session. Off (False) until that decision.
LOOKDEV_UNDER_REPAIR = False
LOOKDEV_REPAIR_NOTE = ('Look-dev is under repair (a card fault found overnight is being fixed); '
                       'the plain light and every other light tool work as usual.')
PRESET_NAMES = [('rembrandt', 'Rembrandt'), ('rembrandt_classic', 'Rembrandt, classic'), ('rembrandt_soft', 'Rembrandt, soft key'),
                ('loop', 'Loop'), ('butterfly', 'Butterfly'), ('clamshell', 'Clamshell'), ('split', 'Split'),
                ('three_point', 'Three-point'), ('rim_hair', 'Rim and hair lights'), ('high_key', 'High key'),
                ('low_key', 'Low key')]


def light_pkg(sub):
    """engine/light loaded once as the package `fs_light` (unique: no clash with any other `rig` module)"""
    if 'fs_light' not in sys.modules:
        spec = importlib.util.spec_from_file_location('fs_light', os.path.join(LIGHT_DIR, '__init__.py'),
                                                      submodule_search_locations=[LIGHT_DIR])
        m = importlib.util.module_from_spec(spec)
        sys.modules['fs_light'] = m
        spec.loader.exec_module(m)
    return importlib.import_module('fs_light.' + sub)


_KTAB = None


def kelvin_of(rgb):
    """the colour temperature nearest a light's rgb (its chromaticity against rig.kelvin_rgb's table, 1500-15000 K)"""
    global _KTAB
    RG = light_pkg('rig')
    if _KTAB is None:
        ks = np.arange(1500.0, 15001.0, 100.0)
        t = np.array([RG.kelvin_rgb(k) for k in ks])
        _KTAB = (ks, t / t.sum(1, keepdims=True))
    c = np.asarray(rgb, float)
    c = c / max(c.sum(), 1e-12)
    ks, t = _KTAB
    return float(ks[int(np.argmin(((t - c) ** 2).sum(1)))])


class LightPanel(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        RC = light_pkg('rig_controller')
        self.RG = light_pkg('rig')
        hd = self._hdris()
        self.C = RC.RigController(hd[0][1] if hd else HDRI_ROOT)
        self.mode = 'view'
        self.split_on = False
        self._seq = 0
        self._sel = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._post)
        self._rt_timer = QTimer(self)               # while the look-dev module compiles (background thread): poll, then
        self._rt_timer.setInterval(300)             # ask for one frame so the swap happens without the view being touched
        self._rt_timer.timeout.connect(self._poll_rt)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        outer.addWidget(sc)
        body = QWidget()
        sc.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel('Light from'))
        self.r_view, self.r_rig = QRadioButton('View key (sculpting)'), QRadioButton('Rig')
        self.r_view.setToolTip('The sculpting key that follows the camera, with the View menu\'s headlight and cavity')
        self.r_rig.setToolTip('The rig below: your own lights, placed in the scene')
        self.r_view.setChecked(True)
        g = QButtonGroup(self)
        g.addButton(self.r_view)
        g.addButton(self.r_rig)
        self.r_rig.toggled.connect(self._mode)
        row.addWidget(self.r_view)
        row.addWidget(self.r_rig)
        row.addStretch(1)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Quality'))
        self.quality = QComboBox()
        self.quality.addItems(['Sculpt (plain light, fast)', 'Look-dev (bounce, area lights)'])
        self.quality.currentIndexChanged.connect(self._quality)
        if LOOKDEV_UNDER_REPAIR:
            it = self.quality.model().item(1)
            if it is not None:
                it.setEnabled(False)
            self.quality.setToolTip(LOOKDEV_REPAIR_NOTE)
        row.addWidget(self.quality, 1)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Preset'))
        self.preset = QComboBox()
        self.preset.addItem('Field Studio default (the HDRI\'s sun and sky)', None)
        for k, t in PRESET_NAMES:
            if k in self.RG.PRESETS:
                self.preset.addItem(t, k)
        self.preset.activated.connect(self._preset)
        row.addWidget(self.preset, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('Place around'))
        self.around = QComboBox()
        self.around.addItems(['her front (+Z)', 'the camera'])
        self.around.setToolTip('A preset\'s angles are measured from her front, or from the camera\'s direction')
        row.addWidget(self.around, 1)
        lay.addLayout(row)

        gk = QGroupBox('Key light')
        kl = QVBoxLayout(gk)
        self.az = FloatField('Azimuth', -180.0, 180.0, decimals=0, suffix=' deg')
        self.az.setToolTip('0 = from her front; + toward her right (the screen\'s left, viewed from the front)')
        self.el = FloatField('Elevation', -10.0, 90.0, decimals=0, suffix=' deg')
        self.kel = FloatField('Colour', 1800.0, 12000.0, decimals=0, suffix=' K')
        self.kel.setToolTip('Colour temperature (a blackbody): candle ~1900, tungsten 3200, daylight 5600, shade 7500+')
        self.stren = FloatField('Strength', 0.0, 4.0, decimals=2)
        self.size = FloatField('Size', 0.1, 90.0, decimals=2, log=True, suffix=' deg')
        self.size.setToolTip('How big the key looks from her (the real sun: 0.53 deg; a 60x90 softbox at 1.2 m: ~41 deg).'
                             ' Larger = softer shadows. Sculpt quality draws at most 2 deg; look-dev samples the full size')
        for w in (self.az, self.el, self.kel, self.stren, self.size):
            kl.addWidget(w)
        self.az.valueChanged.connect(lambda v: self._edit(lambda: self.C.set_sun(azimuth_deg=v)))
        self.el.valueChanged.connect(lambda v: self._edit(lambda: self.C.set_sun(elevation_deg=v)))
        self.kel.valueChanged.connect(lambda v: self._edit(lambda: self.C.set_temperature('sun', v)))
        self.stren.valueChanged.connect(lambda v: self._edit(lambda: self.C.set_strength('sun', v)))
        self.size.valueChanged.connect(lambda v: self._edit(lambda: self.C.set_sun_size(v)))
        lay.addWidget(gk)

        ge = QGroupBox('Environment')
        el_ = QVBoxLayout(ge)
        row = QHBoxLayout()
        row.addWidget(QLabel('HDRI'))
        self.hdri = QComboBox()
        for h, path in hd:
            self.hdri.addItem(h, path)
        self.hdri.setToolTip('The environment map: the neutral studio (the default) or a CC0 HDRI in work/hdri (more can be '
                             'added there)')
        row.addWidget(self.hdri, 1)
        el_.addLayout(row)
        self.sky = FloatField('Sky light', 0.0, 2.0, decimals=2)
        self.sky.setToolTip("The environment's light on her (0 = a dark studio: the key and the rig's lights only)")
        self.rot = FloatField('Rotation', 0.0, 360.0, decimals=0, suffix=' deg')
        self.bg = FloatField('Background', 0.0, 2.0, decimals=2)
        self.bg.setToolTip('The environment as seen behind her (0 = a black studio surround); her light is unchanged')
        self.ev = FloatField('Exposure', -4.0, 4.0, decimals=1, suffix=' EV')
        for w in (self.sky, self.rot, self.bg, self.ev):
            el_.addWidget(w)
        self.sky.valueChanged.connect(lambda v: self._edit(lambda: self.C.set_environment(sky_gain=v)))
        self.hdri.activated.connect(lambda i: self._edit(lambda: self.C.set_environment(hdri=self.hdri.itemData(i))))
        self.rot.valueChanged.connect(lambda v: self._edit(lambda: self.C.set_environment(rotation_deg=v)))
        self.bg.valueChanged.connect(lambda v: self._edit(lambda: self.C.set_environment(background_gain=v)))
        self.ev.valueChanged.connect(lambda v: self._edit(lambda: self._set_ev(v)))
        lay.addWidget(ge)

        gl = QGroupBox('Lights in the rig')
        ll = QVBoxLayout(gl)
        self.lights = QListWidget()
        self.lights.setMaximumHeight(120)
        self.lights.itemChanged.connect(self._toggle_light)
        ll.addWidget(self.lights)
        row = QHBoxLayout()
        self.b_solo = QPushButton('Solo')
        self.b_solo.setCheckable(True)
        self.b_solo.setToolTip('Only the selected light (the others off while held)')
        self.b_solo.toggled.connect(self._solo)
        row.addWidget(self.b_solo)
        row.addStretch(1)
        ll.addLayout(row)
        lay.addWidget(gl)

        ga = QGroupBox('Compare and keep')
        al = QVBoxLayout(ga)
        row = QHBoxLayout()
        for s in ('A', 'B'):
            b = QPushButton('Store ' + s)
            b.clicked.connect(lambda _=False, s=s: self._store(s))
            row.addWidget(b)
        for s in ('A', 'B'):
            b = QPushButton('Show ' + s)
            b.clicked.connect(lambda _=False, s=s: self._recall(s))
            row.addWidget(b)
        al.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton('Compare A | B')
        b.setToolTip('Both stored rigs rendered at this view, side by side')
        b.clicked.connect(self.compare)
        row.addWidget(b)
        self.b_split = QPushButton('Split A | B in the view')
        self.b_split.setCheckable(True)
        self.b_split.setToolTip('The view in two halves: left the live rig (A, edit it here), right the stored B; '
                                'strokes and orbiting keep both live (plain light)')
        self.b_split.toggled.connect(self._split)
        row.addWidget(self.b_split)
        b = QPushButton('Save...')
        b.clicked.connect(self._save)
        row.addWidget(b)
        b = QPushButton('Load...')
        b.clicked.connect(self._load)
        row.addWidget(b)
        al.addLayout(row)
        lay.addWidget(ga)

        self.status = QLabel('')
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch(1)
        win.engine.callDone.connect(self._done)
        self.refresh()

    # ---- the model -> the widgets ----
    def _hdris(self):
        """-> [(name, folder)]: the default environment first (fs-shell 11-13: the neutral studio, gpu/studio_env.py),
        then every HDRI folder in work/hdri"""
        out = []
        try:
            from ..gpu.scene import HDRI_DIR
            nm = os.path.basename(HDRI_DIR)
            out.append(('studio (neutral)' if nm == 'studio_neutral' else nm, HDRI_DIR))
        except Exception:
            HDRI_DIR = None
        try:
            for d in sorted(os.listdir(HDRI_ROOT)):
                p_ = os.path.join(HDRI_ROOT, d)
                if os.path.isfile(os.path.join(p_, 'env.json')) and (HDRI_DIR is None or os.path.normpath(p_) != os.path.normpath(HDRI_DIR)):
                    out.append((d, p_))
        except OSError:
            pass
        return out

    def refresh(self):
        C = self.C
        az, el = C.sun_angles()
        s = C.rig.sun()
        for w, v in ((self.az, az), (self.el, el), (self.stren, s.strength), (self.size, C.key_size_deg()),
                     (self.rot, C.rig.environment.rotation_deg), (self.bg, C.rig.environment.background_gain),
                     (self.sky, getattr(C.rig.environment, 'sky_gain', 1.0)),
                     (self.ev, C.rig.exposure.ev)):
            if not w.busy():
                w.setValue(v)
        if not self.kel.busy():
            self.kel.setValue(kelvin_of(s.rgb))
        self.quality.blockSignals(True)
        self.quality.setCurrentIndex(1 if C.rig.quality in ('lookdev', 'final') else 0)
        self.quality.blockSignals(False)
        self.lights.blockSignals(True)
        self.lights.clear()
        for l in C.rig.lights:
            if l.kind in ('env', 'sky_uniform') or (l.kind == 'sun' and C.key_twin() is not None):
                continue                           # the key's plain-light twin is shown as the key itself
            it = QListWidgetItem('%s  (%s)' % (l.name or l.kind, l.kind))
            it.setData(Qt.UserRole, l.name or l.kind)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if l.reaches else Qt.Unchecked)
            self.lights.addItem(it)
        self.lights.blockSignals(False)
        self._say()

    def _say(self, extra=''):
        C = self.C
        local = [l for l in C.rig.lights if l.kind not in ('sun', 'env', 'sky_uniform')]
        msg = []
        if self.split_on:
            msg.append("Split view: left = the live rig (A), right = B (plain light in both; look-dev's bounce in a split "
                       "needs its caches twice: a later step).")
        if self.mode == 'view':
            msg.append('The view\'s sculpting key is lit; pick Rig to light her with this rig.')
        elif local and C.rig.quality not in ('lookdev', 'final'):
            msg.append('Sculpt quality: the key renders as a sharp sun; the softboxes, fills and rims render in Look-dev.')
        if C.rig.quality in ('lookdev', 'final') and not self._has_runtime():
            msg.append('Look-dev\'s bounce and area lights wait for the viewport\'s light-runtime hook (proposed 19:55);'
                       ' until then the view shows the plain light.')
        if extra:
            msg.append(extra)
        self.status.setText(' '.join(msg))

    def _has_runtime(self):
        gv = getattr(self.win.engine, 'gpu_view', None)
        return gv is not None and hasattr(gv, 'light_rt')

    # ---- edits ----
    def _edit(self, fn):
        fn()
        self._apply()

    def _set_ev(self, v):
        self.C.rig.exposure.ev = float(v)
        self.C._changed('exposure')

    def _mode(self, rig_on):
        if not rig_on and self.split_on:
            self.b_split.setChecked(False)          # the view key has no A | B
        self.mode = 'rig' if rig_on else 'view'
        self._apply()
        self._say()

    def _quality(self, i):
        self.C.set_quality('lookdev' if i == 1 else 'sculpt')
        self._apply()
        self._say()

    def _subject(self):
        cam = self.win.viewport.cam
        return np.asarray(cam.target, float), np.asarray(cam.pos, float)

    def _preset(self, i):
        key = self.preset.itemData(i)
        tgt, pos = self._subject()
        if key is None:
            q = self.C.rig.quality
            self.C.rig = self.RG.fieldstudio_default(self.C.hdri_dir)
            self.C.rig.quality = q
        else:
            front = None
            if self.around.currentIndex() == 1:
                f = pos - tgt
                f[1] = 0.0
                if np.linalg.norm(f) > 1e-6:
                    front = f / np.linalg.norm(f)
            self.C.apply_preset(key, tgt, front=front)    # the rig records tgt as its subject
        if self.mode != 'rig':
            self.r_rig.setChecked(True)            # a preset is asked for: show it
        self.refresh()
        self._apply()

    def _toggle_light(self, it):
        self.C.mute(it.data(Qt.UserRole), on=it.checkState() != Qt.Checked)
        self._apply()

    def _solo(self, on):
        it = self.lights.currentItem()
        self.C.solo(it.data(Qt.UserRole) if (on and it is not None) else None)
        if on and it is None:
            self.b_solo.blockSignals(True)
            self.b_solo.setChecked(False)
            self.b_solo.blockSignals(False)
            self._say('Select a light first, then Solo.')
            return
        self._apply()

    def _store(self, s):
        self.C.store(s)
        if s == 'B' and self.split_on:
            self._install_split()
        self._say('Stored as %s.' % s)

    # ---- the in-view split (engine/light/split_view.py: one scene, rig B over the right half's pixels) ----
    def _split(self, on):
        C = self.C
        if on and ('A' not in C.slots or 'B' not in C.slots):
            self.b_split.blockSignals(True)
            self.b_split.setChecked(False)
            self.b_split.blockSignals(False)
            self._say('Store A and B first, then Split.')
            return
        if on:
            self.split_on = True
            if self.mode != 'rig':
                self.r_rig.setChecked(True)
            C.recall('A')
            C.set_quality('sculpt')
            self.quality.setEnabled(False)
            self.refresh()
            self._install_split()
        else:
            self.split_on = False
            self.quality.setEnabled(True)
            self._remove_split()
        self._apply()
        self._say()

    def _install_split(self):
        C = self.C
        keep = C.rig
        C.rig = C.slots['B']
        rig_b = C.effective_rig()
        C.rig = keep
        rig_b.quality = 'sculpt'
        SVm, RGm, FL = light_pkg('split_view'), light_pkg('rig'), light_pkg('fs_lightcache')
        env = RGm.Env(rig_b.environment.hdri or C.hdri_dir)

        def fn(engine):
            gv = engine.gpu_view
            if gv is None:
                return 'no card view'
            G = gv.gpu
            rt = getattr(gv, 'light_rt', None)
            if rt is not None and rt.state != 'off':
                rt._off(G)                          # plain light in both halves
            base = FL.shaded_kernel(G)              # the Shaded kernel (a view mode may hold the slot: fs-shell)
            if isinstance(base, SVm.SplitKernel):
                base.set_rig_b(rig_b, env)
            else:
                FL.set_shaded(G, SVm.SplitKernel(G, base, rig_b, env, 0.5))
            gv.key = None
            return 'ok'
        self._seq += 1
        self.win.engine.post('call', fn, ('lightpanel', self._seq))

    def _remove_split(self):
        SVm, FL = light_pkg('split_view'), light_pkg('fs_lightcache')

        def fn(engine):
            gv = engine.gpu_view
            if gv is None:
                return 'no card view'
            base = FL.shaded_kernel(gv.gpu)
            if isinstance(base, SVm.SplitKernel):
                FL.set_shaded(gv.gpu, base.inner)
            gv.key = None
            return 'ok'
        self._seq += 1
        self.win.engine.post('call', fn, ('lightpanel', self._seq))

    def _recall(self, s):
        if s not in self.C.slots:
            self._say('Nothing stored as %s yet.' % s)
            return
        self.C.recall(s)
        self.refresh()
        self._apply()
        self._say('Showing %s.' % s)

    def _save(self):
        os.makedirs(RIG_DIR, exist_ok=True)
        p, _ = QFileDialog.getSaveFileName(self, 'Save light rig', os.path.join(RIG_DIR, (self.C.rig.name or 'rig') + '.json'),
                                           'Light rig (*.json)')
        if p:
            self.C.rig.name = os.path.splitext(os.path.basename(p))[0]
            self.C.save(p)
            self._say('Saved %s.' % p)

    def _load(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Load light rig', RIG_DIR if os.path.isdir(RIG_DIR) else '', 'Light rig (*.json)')
        if p:
            self.C.load(p)
            self.refresh()
            self._apply()
            self._say('Loaded %s.' % os.path.basename(p))

    # ---- to the card (the engine thread) ----
    def _apply(self):
        self._timer.start(30)                       # a slider drag posts once per 30 ms, not once per tick

    def _post(self):
        eng = self.win.engine
        if getattr(eng, 'gpu_view', None) is None:
            self._say('The card view is off: the rig applies when it is on.')
            return
        rig = self.C.effective_rig() if self.mode == 'rig' else None
        if rig is not None and self.split_on:
            rig.quality = 'sculpt'                  # the split is plain light (one kernel wrapped, no caches)
        look = rig is not None and rig.quality in ('lookdev', 'final')
        LR = light_pkg('light_runtime') if look else None

        def fn(engine):
            gv = engine.gpu_view
            if gv is None:
                return 'no card view'
            base = {k: v for k, v in gv.light.items() if k != 'rig'}
            gv.light = base if rig is None else dict(base, rig=rig)
            if look and hasattr(gv, 'light_rt') and gv.light_rt is None:
                gv.light_rt = LR.LightRuntime()
            gv.key = None                           # the rig is not in the view's frame key: restart once
            return 'ok'
        self._seq += 1
        eng.post('call', fn, ('lightpanel', self._seq))
        self.win.viewport.request(False)
        if look:
            self._rt_timer.start()

    def _poll_rt(self):
        gv = getattr(self.win.engine, 'gpu_view', None)
        rt = getattr(gv, 'light_rt', None) if gv is not None else None
        if rt is None:
            return
        if getattr(rt, 'compiling', False):
            self._say('Preparing the look-dev light (compiled once after an update; the view keeps the plain light meanwhile)...')
            return
        self._rt_timer.stop()
        if getattr(rt, 'error', None):
            self._say('The look-dev light could not start: %s' % rt.error)
        else:
            self._say()
        self.win.viewport.request(False)

    def compare(self):
        C = self.C
        if 'A' not in C.slots or 'B' not in C.slots:
            self._say('Store A and B first (set a light, Store A; change it, Store B).')
            return
        rigs = []
        keep = C.rig
        for s in ('A', 'B'):
            C.rig = C.slots[s]
            rigs.append(C.effective_rig())
        C.rig = keep
        v = self.win.viewport.view()
        cam, w, h = v['cam'], v['w'], v['h']

        def fn(engine):
            import cupy as cp
            gv = engine.gpu_view
            if gv is None:
                return None
            G = gv.gpu
            k0 = G.k_light
            G.k_light = light_pkg('fs_lightcache').plain_kernel(G)   # the comparison is the plain light (any slot holder put back after)
            out = []
            try:
                for r in rigs:
                    G.render_rect(cam, w, h, passes=16, light={'rig': r})
                    out.append(cp.ascontiguousarray(G.d_out.view(cp.uint8).reshape(h, w, 4)[:, :, 2::-1]).get())
            finally:
                G.k_light = k0
                gv.key = None
            return out
        self._seq += 1
        self._cmp_token = ('lightpanel-compare', self._seq)
        self.win.engine.post('call', fn, self._cmp_token)
        self._say('Rendering A and B at this view...')

    def _done(self, tr):
        token, res = tr
        if token == getattr(self, '_cmp_token', None):
            self.win.viewport.request(False)
            if not res:
                self._say('Compare needs the card view.')
                return
            CompareDialog(self, res[0], res[1]).show()
            self._say('')
        elif isinstance(token, tuple) and token and token[0] == 'lightpanel':
            self.win.viewport.request(False)


class CompareDialog(QDialog):
    def __init__(self, parent, a, b):
        super().__init__(parent)
        self.setWindowTitle('Light test: A | B')
        lay = QHBoxLayout(self)
        for name, im in (('A', a), ('B', b)):
            col = QVBoxLayout()
            col.addWidget(QLabel(name))
            im = np.ascontiguousarray(im)
            q = QImage(im.data, im.shape[1], im.shape[0], 3 * im.shape[1], QImage.Format_RGB888).copy()
            lb = QLabel()
            lb.setPixmap(QPixmap.fromImage(q).scaledToWidth(min(760, im.shape[1]), Qt.SmoothTransformation))
            col.addWidget(lb)
            lay.addLayout(col)


def install(win):
    panel = LightPanel(win)
    dock = QDockWidget('Light', win)
    dock.setObjectName('light_panel')
    dock.setWidget(panel)
    dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
    dock.setMinimumWidth(340)
    win.addDockWidget(Qt.RightDockWidgetArea, dock)
    docks = getattr(win, 'docks', None)
    if isinstance(docks, dict) and 'Properties' in docks:
        win.tabifyDockWidget(docks['Properties'], dock)
        docks['Properties'].raise_()
        docks['Light'] = dock
    vm = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'View' in a.menu().title()), None)
    if vm is not None:
        a = dock.toggleViewAction()
        a.setText('Light (rigs, presets, the key, the environment, A / B)')
        vm.addAction(a)
    win.light_panel, win.light_dock = panel, dock
    return panel
