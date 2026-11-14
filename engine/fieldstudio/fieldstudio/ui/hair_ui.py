"""The Hair workspace (fs-hair, 11-11; design 4.13, chunk 11).

The workflow (11-11): choose the type of hair and the body region, choose the area with a tool, refine the details
(colour, thickness, density, etc.), run the generator, then shorten some of it, move it and style it a bit.

The panel (Hair, a dock on the right; Tools > Hair... opens it):
- KIND: what grows there (scalp, hairline, brow, body, underarm, lower body); its cited defaults fill the details.
- AREA: a brush that paints where the hair grows on a form (drag; Ctrl erases); the area in cm2 is shown.
- DETAILS: colour (pigment: melanin, redness, ash, or a preset), thickness (um), density (/cm2), length (mm), seed.
- GENERATE: the hair engine grows it (engine/hair: roots by density, follicles at the kind's angle along the lie, the
  growth cycle's lengths, long hair settled as rods with weight, stiffness, curl, contact and friction).
- STYLE: Comb (drag: the hair under the brush follows, roots stay, lengths kept, never through the form), Cut (drag: each
  hair under the scissors ends there, blunt), Trim (a length for all of it), Settle (the hair comes to rest as it lies),
  Water (the set washes out: back to the natural fall at the current lengths).
- STATES: a styled hair kept by name over the natural base (styles as states).
- HER NATURAL BASE: her approved base (hair3's n2c, approved as the hair's baseline) placed on her.

Drawing: every strand drawn into the frame (hair/draw.py: on the card when the window's renderer is, else the CPU),
depth-tested against the form, cached per view; the overlay is the viewport's own `overlays` hook (as the grids).
Middle / right drags still orbit and pan; Esc leaves a hair tool."""
import math
import os
import threading
import time
import zlib

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from ..hair import draw as HD
from ..hair import generate as HG
from ..hair.groom import Frame, HairGroom
from .exit_rule import card_thread   # fs-shell 11-13: the hair's card threads, joined at the exit

HAIR_COL = QColor(214, 170, 96)
AREA_CELL = 0.0015            # m: the area's samples are one per 1.5 mm cell of the surface (object frame)
COLOURS = {                   # pigment presets (eumelanin concentration as melanin 0-1, redness, ash): working starts
    'Golden blond (her base)': (0.45, 0.045, 0.05),
    'Platinum blond': (0.18, 0.02, 0.15),
    'Light brown': (0.65, 0.06, 0.03),
    'Brown': (0.80, 0.07, 0.02),
    'Dark brown': (0.90, 0.05, 0.0),
    'Black': (0.97, 0.03, 0.0),
    'Auburn': (0.70, 0.45, 0.0),
    'Red': (0.55, 0.85, 0.0),
    'Grey': (0.30, 0.0, 0.8),
}
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_UI_DIR) != 'ui':     # a work-in-progress copy (fs-hair/wip): the live package's folder
    import fieldstudio.ui as _fu
    _UI_DIR = os.path.dirname(os.path.abspath(_fu.__file__))
HER_BASE = os.environ.get('FIELDSTUDIO_HAIR_BASE', '')


# ---- frame helpers (the viewport's last frame) ------------------------------------------------------------------------
def _frame(win):
    fr = getattr(win.viewport, 'frame', None)
    if fr is None or fr.get('depth') is None or fr.get('obj') is None:
        return None
    return fr


def _frame_px(win, pos):
    fr = _frame(win)
    vp = win.viewport
    h, w = np.asarray(fr['obj']).shape
    return pos.x() * w / max(vp.width(), 1), pos.y() * h / max(vp.height(), 1), w, h


def _rays(cam, ii, jj, w, h):
    from .marks_ui import frame_rays
    return frame_rays(cam, ii, jj, w, h)


def _report(win, what, why, todo=None, details=None):
    """a failure over the viewport (fs-shell's failure_ui, ce629430); never raises"""
    try:
        from .failure_ui import report
        kw = {'details': details} if details else {}
        if todo:
            kw['todo'] = todo
        report(win, what, why, **kw)
    except Exception:
        pass


def sphere_centre(R, sub=20000):
    """the centre of the sphere best fitting the roots (the head's centre, for the sky's outward direction); a flat patch
    of roots gives a far centre, which still points outward"""
    R = np.asarray(R, np.float64)
    R = R[np.isfinite(R[:, 0])][:: max(1, len(R) // sub)]
    if len(R) < 8:
        return None
    A = np.c_[2 * R, np.ones(len(R))]
    b = (R * R).sum(1)
    try:
        c = np.linalg.lstsq(A, b, rcond=None)[0][:3]
    except Exception:
        return None
    return c if np.all(np.isfinite(c)) else None


def stats_words(st):
    """the groom's checks in words for the status line"""
    if not st:
        return ''
    out = ['%s hairs' % format(int(st.get('strands', 0)), ',')]
    L = st.get('length_mm_p10_50_90')
    if L:
        out.append('lengths %.0f / %.0f / %.0f mm (p10 / 50 / 90)' % tuple(L))
    if 'points_inside_form' in st:
        k = int(st['points_inside_form'])
        out.append('none inside the form' if k == 0 else '%d points inside the form' % k)
    return '; '.join(out)


class _Bridge(QObject):
    done = Signal(object)
    progress = Signal(str)
    shade = Signal(object)
    rest = Signal(object)


class HairOverlay:
    """the viewport overlay: each visible groom drawn into the frame, cached per (view, groom version, depth)"""

    def __init__(self, layer):
        self.L = layer
        self.cache = {}
        self.bufs = {}
        self.stats = {}
        self.interactive = False

    def active(self):
        return any(g.visible and g.cur is not None for g in self.L.grooms.values()) or bool(self.L.area_on())

    def composite(self, fr, rgb):
        L = self.L
        cam = fr['cam']
        depth = np.asarray(fr['depth'], np.float32)
        H, W = rgb.shape[:2]
        if depth.shape != (H, W):
            return rgb
        # the depth's exact signature (face_b 11-12: a sum over every 17th row / 13th column missed an undo's change, so
        # the final frame kept a stale hair layer): a CRC over the whole buffer, ~2-3 ms for her 2.6 MB
        dkey = (depth.shape, zlib.crc32(np.ascontiguousarray(depth).view(np.uint8)))
        out = rgb
        t_all = time.perf_counter()
        for oid, g in list(L.grooms.items()):
            if not g.visible or g.cur is None:
                continue
            fm = Frame(L.win.viewport.transforms.get(oid))
            if getattr(g, '_centre_v', None) != g.version:
                g._centre_w = fm.to_world(np.nanmean(g.cur['P'][:, 0], 0)[None])[0]
                g._centre_l = sphere_centre(g.cur['P'][:, 0])
                g._centre_v = g.version
            L._hair_centre_w = g._centre_w
            L._hair_centre_local = g._centre_l
            gv_ = getattr(L.win.engine, 'gpu_view', None)
            lkey = repr(sorted((k, repr(v)) for k, v in (getattr(gv_, 'light', None) or {}).items()))
            ck = (oid, tuple(np.round(HD.cam_vec(cam, W, H), 7)), W, H, g.version, fm.key, self.interactive, lkey,
                  getattr(g, 'section_version', 0), getattr(L, 'others', 'dim'))
            # (fs-bench run 3, 11-12: face strokes 3x slower with her hair shown) a body stroke moves the skin, so the depth
            # changed every frame and the whole hair was redrawn at full quality each frame (~200 ms). The hair has not
            # changed: while frames are not final its last layer is laid over the new frame; the final frame redraws it
            # in full where the depth or the quality differ. A moved camera redraws at the interactive budget till final.
            final = bool(fr.get('final', True))
            want_full = final and not self.interactive
            hit = self.cache.get(oid)
            reuse = (hit is not None and hit[0] == ck and (hit[1] == dkey or not final) and (hit[2] or not want_full))
            if not reuse:
                cam_l = fm.camera(cam)
                dep_l = depth / fm.s
                st = {}
                lights = L.hair_lights(fm, cam) if os.environ.get('FS_HAIR_RIG_LIGHT', '1') != '0' else None
                if lights is not None:
                    light = np.asarray(lights['dirs'][lights.get('key', 0)], float)   # shadowed through the hair
                else:
                    light = L.light_dir(fm)
                COL = L.shaded(g, light, allow=not self.interactive)
                RADd = g.cur['RAD']
                sec = getattr(g, 'section', None)
                mode = getattr(L, 'others', 'dim')
                if sec is None or mode == 'show':
                    g._sec_draw = None                     # no section: its dimmed / hidden copy goes (it leaked 106 MB)
                if sec is not None and len(sec) == g.n and mode != 'show':
                    sk = (g.version, getattr(g, 'section_version', 0), mode, id(COL))
                    sc = getattr(g, '_sec_draw', None)
                    if sc is None or sc[0] != sk:
                        off = ~sec.astype(bool)
                        if mode == 'hide':
                            R2 = g.cur['RAD'].copy()
                            R2[off] = 0.0                      # not drawn
                            sc = (sk, COL, R2)
                        else:
                            C2 = COL.copy()
                            C2[off, :, :3] *= 0.35             # dimmed: pinned away, still seen
                            sc = (sk, C2, g.cur['RAD'])
                        g._sec_draw = sc
                    COL, RADd = sc[1], sc[2]
                layer = None
                try:
                    if L.use_card() and not getattr(L, 'card_failed', None):
                        try:
                            buf = self.bufs.setdefault(oid, HD.StrandBuffers())
                            rows = None
                            if self.interactive and L.touched is not None and getattr(L, 'cur_oid', None) == oid                                     and len(L.touched) == g.n:
                                rows = np.nonzero(L.touched)[0]          # the stroke's strands only
                            buf.set((oid, g.version, id(COL), id(RADd)), g.cur['P'], RADd, COL, rows=rows,
                                    col_key=id(COL))
                            fb = getattr(g, '_fibre', None)     # run_hair's light (the last one while a new one computes)
                            if fb is not None and len(fb[1]) == g.n and COL.shape[2] == 3:
                                buf.set_light(id(fb[1]), fb[1], fb[2])
                            else:
                                buf.clear_light()
                            layer = HD.draw(buf, cam_l, W, H, dep_l, light_dir=light,
                                            budget=12_000_000 if want_full else 1_500_000, stats=st,
                                            cov_mult=g.cov_mult(), keep_on_card=True, lights=lights)
                        except Exception:                # the card path failed: the CPU draws from now on (logged)
                            import traceback
                            L.card_failed = traceback.format_exc()[-1500:]
                            print('hair: card draw failed, CPU from now on:', L.card_failed, flush=True)
                            _report(L.win, 'Hair drawn on the card', L.card_failed.strip().splitlines()[-1],
                                    'the hair is drawn on the CPU from now on', L.card_failed)
                            layer = None
                    if layer is None:
                        fb = getattr(g, '_fibre', None)
                        okl = fb is not None and len(fb[1]) == g.n and COL.shape[2] == 3
                        layer = HD.draw_cpu(g.cur['P'], RADd, COL, cam_l, W, H, dep_l, light_dir=light,
                                            budget=900_000 if want_full else 120_000, stats=st,
                                            cov_mult=g.cov_mult(), lights=lights,
                                            flb=fb[1] if okl else None, flk=fb[2] if okl else 0)
                        layer = (layer[0], layer[1], lights is not None)
                except Exception as e:                       # the hair must never stop the view
                    L.win.status.showMessage('Hair draw: %r' % (e,), 8000)
                    continue
                hit = (ck, dkey, want_full, layer)
                self.cache[oid] = hit
                self.stats[oid] = st
                self.redraws = getattr(self, 'redraws', 0) + 1
            else:
                self.reuses = getattr(self, 'reuses', 0) + 1
            layer = hit[3]
            if isinstance(layer, HD.CardLayer):
                out = layer.resolve(out)
            else:
                out = (HD.resolve_tm if len(layer) > 2 and layer[2] else HD.resolve)(out, layer[0], layer[1])
        if L.area_on():
            out = L.draw_area(fr, out)
        self.compose_ms = (time.perf_counter() - t_all) * 1000.0
        if self.interactive:
            L.compose_log.append(self.compose_ms)
        return out


class HairLayer(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.grooms = {}                     # oid -> HairGroom
        self.tool = None                     # None | 'area' | 'comb' | 'cut'
        self.radius_px = 60.0
        self.others = 'dim'                  # the hair pinned out of the section: 'show' | 'dim' | 'hide'
        self.strength = 0.7
        self.area = {}                       # oid -> {cell key: local point}
        self.area_target = None
        self.press = None
        self.last = None
        self.touched = None
        self.busy = False
        self.bridge = _Bridge()
        self.bridge.done.connect(self._generated)
        self.bridge.progress.connect(lambda m: self._say(m))
        self.bridge.shade.connect(self._shade_done)
        self.bridge.rest.connect(self._rest_done)
        self.overlay = HairOverlay(self)
        self.dab_ms = []
        self.compose_log = []
        self.timings = []

    # ---- environment ---------------------------------------------------------------------------------------------------
    def use_card(self):
        eng = self.win.engine
        return bool(getattr(eng, 'use_gpu', False)) and os.environ.get('FS_HAIR_CPU', '0') != '1'

    def hair_lights(self, fm, cam):
        """the window's light for the hair, one light system with the skin: fs-light's rig.hair_lights on what the view
        renders (gv.light: the Light panel's effective rig, or the view's own sculpting key, headlight and ev), with the
        tracer's quality rule deciding which lights exist (agreed with fs-light 11-11 ~22:15). -> {'dirs', 'rgb',
        'expo', 'sky', 'key'} in the object's frame (expo already divided by pi: radiance = irradiance / pi x the
        fibre's terms), or None (the old working light)"""
        try:
            gv = getattr(self.win.engine, 'gpu_view', None)
            if gv is None or getattr(gv, 'gpu', None) is None:
                return None
            from .lightpanel import light_pkg
            R = light_pkg('rig')
            light = dict(getattr(gv, 'light', {}) or {})
            rig = light.get('rig')
            hd = (getattr(getattr(rig, 'environment', None), 'hdri', None) if rig is not None else None)
            if not hd:
                from ..gpu.scene import HDRI_DIR
                hd = HDRI_DIR
            envs = getattr(self, '_envs', None)
            if envs is None:
                envs = self._envs = {}
            if hd not in envs:
                envs[hd] = R.Env(hd)
            centre = getattr(self, '_hair_centre_w', None)
            if centre is None:
                centre = np.asarray(cam.target, float)
            r = R.hair_lights(light, centre, cam.pos, envs[hd])
            ls = r.get('lights') or []
            if not ls:
                return None
            dirs = np.array([np.asarray(l['dir'], float) for l in ls])
            rgb = np.array([np.asarray(l['rgb'], float) for l in ls])
            key = next((i for i, l in enumerate(ls) if l.get('key')), 0)
            sky = math.pi * np.asarray(r.get('sky_mean_radiance', (0.0, 0.0, 0.0)), float)   # radiance -> irradiance
            out = {'dirs': fm.dir_local(dirs), 'rgb': rgb, 'expo': float(r['exposure']) / math.pi, 'sky': sky,
                   'key': int(key), 'quality': r.get('quality')}
            # the sky as the skin sees it (fs-light): the tracer's SH along each strand's outward direction
            # from the head's centre (a sphere fitted to the roots), in world through the object's rotation
            if r.get('sky_sh') is not None and getattr(self, '_hair_centre_local', None) is not None:
                out.update(sky_sh=np.asarray(r['sky_sh'], float), centre=self._hair_centre_local, rot=fm.R)
            return out
        except Exception as e:
            if not getattr(self, '_lights_warned', False):
                print('hair: the window light could not be read (%r); the working light is used' % (e,), flush=True)
                self._lights_warned = True
            return None

    def light_dir(self, fm):
        try:
            gv = getattr(self.win.engine, 'gview', None) or getattr(self.win.engine, 'gpu_view', None)
            L = np.asarray(gv.gpu.sun_world()[0], float) if gv is not None else None
        except Exception:
            L = None
        if L is None or not np.all(np.isfinite(L)):
            L = np.array([0.35, 0.8, 0.45])
        return fm.dir_local(L[None])[0]

    def _say(self, m, ms=8000):
        self.win.status.showMessage('Hair: ' + m, ms)
        p = getattr(self, 'panel', None)
        if p is not None:
            p.info.setText(m)

    def refresh(self):
        self.win.viewport.refs_changed()

    def refresh_soon(self, ms=40):
        """many brush events, one redraw: the view is redrawn at most every ms while a stroke runs"""
        t = getattr(self, '_rtimer', None)
        if t is None:
            t = self._rtimer = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(self.refresh)
        if not t.isActive():
            t.start(ms)

    def shaded(self, g, light, allow=True):
        """the hair's colour with the key's share through the hair mass (hair/shadow.py), per version and light. It is
        computed on a worker thread (about 1 s for 200k hairs): the last shade (or the plain colour) is drawn until it
        lands, then the view redraws; while a stroke runs nothing is recomputed"""
        key = (g.version, tuple(np.round(np.asarray(light, float), 3)))
        sh = getattr(g, '_shade', None)
        if sh is not None and sh[0] == key and sh[1].shape[:2] == g.cur['COL'].shape[:2]:
            return sh[1]
        if allow and getattr(g, '_shade_job', None) != key:
            g._shade_job = key
            # the current arrays, not copies (282 MB for her): a stroke bumps the version, and a result computed while
            # the hair changed is dropped by its key
            P, RAD, SEG, COL0 = g.cur['P'], g.cur['RAD'], g.cur['SEG'], g.cur['COL']
            SIG = (g.base or {}).get('SIG')
            # on the card: run_hair's own light at the strand points (hair-shadow's method, hair/fibre_light.py: the
            # key's and the sky's coloured transmittance through the fibres and the backscatter), one answer with the
            # review render; else (the CPU view, or a failure) the lattice light below
            use_fibre = (self.use_card() and not getattr(self, '_fibre_failed', False)
                         and os.environ.get('FS_HAIR_FIBRE_LIGHT', '0') == '1')   # opt-in until herL4b proves it on her

            def work():
                # one shade job at a time (two at once doubled the memory and raced the fibre layer's source); a job
                # superseded while it waited is skipped
                lock = self.__dict__.setdefault('_shade_lock', threading.Lock())
                with lock:
                    if getattr(g, '_shade_job', None) != key:
                        return
                    _work()

            def _work():
                try:
                    t = time.perf_counter()
                    if use_fibre:
                        try:
                            from ..hair import fibre_light as FLT
                            inf = {}
                            FLB, K = FLT.window_light(g, light, info=inf)
                            # the light in its own small buffer (every K-th point, uint16, ~55 MB for her), beside the
                            # plain base colour: the drawer interpolates it along each strand
                            g._fibre = (key, FLB, K)
                            self.timings.append(('fibre_light_ms', round((time.perf_counter() - t) * 1000, 1), inf))
                            self.bridge.shade.emit((g, key, COL0))
                            return
                        except Exception as e:
                            self._fibre_failed = True
                            print('hair: the fibre light failed, the lattice light from now on: %r' % (e,), flush=True)
                            _report(self.win, 'Hair light (run_hair\'s method)', repr(e), 'the lattice light is used')
                            t = time.perf_counter()
                    from ..hair.shadow import shares, shade_floor
                    # the shadow scales the LIGHT, not the albedo (00:2x): the base stays as grown; channel 3 is the key's
                    # transmittance through the hair, channel 4 the sky's (outward from the head's centre, where the SH
                    # sky is read), each with the light that scatters back out of the hair beneath (the floor)
                    ctr = sphere_centre(P[:, 0])
                    if ctr is None:
                        ctr = np.nanmean(P[:, 0], 0)
                    ks, ss = shares(P, RAD, SEG, light, centre=ctr, SIG=SIG)
                    fl = shade_floor(SIG)
                    fl = np.full(len(P), 0.3, np.float32) if fl is None else fl
                    f2 = fl[:, None]
                    # the same small light buffer as run_hair's light (every 3rd point, uint16): the key's and the sky's
                    # share with the light scattered back from beneath (the floor) folded in; no backscatter term
                    from ..hair.fibre_light import sample_ks
                    K = 3
                    kq = sample_ks(P.shape[1], K)
                    FLB = np.zeros((len(P), len(kq), 9), np.uint16)
                    a_ = np.rint(np.clip(f2 + (1.0 - f2) * ks[:, kq], 0.0, 1.0) * 65535.0).astype(np.uint16)
                    b_ = np.rint(np.clip(f2 + (1.0 - f2) * ss[:, kq], 0.0, 1.0) * 65535.0).astype(np.uint16)
                    del ks, ss
                    FLB[..., 0:3] = a_[..., None]
                    FLB[..., 3:6] = b_[..., None]
                    g._fibre = (key, FLB, K)
                    self.timings.append(('shadow_ms', round((time.perf_counter() - t) * 1000, 1)))
                    self.bridge.shade.emit((g, key, COL0))
                except Exception as e:
                    print('hair shadow failed: %r' % (e,), flush=True)
                    _report(self.win, 'Hair light', repr(e), 'the hair is drawn unshadowed')
            card_thread(work, name='hair-shadow')       # fs-shell 11-13: joined by the exit rule (ui/exit_rule.py)
        if sh is not None and sh[1].shape[:2] == g.cur['COL'].shape[:2]:
            return sh[1]
        return g.cur['COL']

    def _rest_done(self, r):
        g, (v0, sel, P), ms = r
        ok = g.apply_rows(v0, sel, P)
        self.timings.append(('settle_after_comb_ms', ms, 'applied' if ok else 'dropped (the hair changed meanwhile)'))
        if ok:
            self.refresh()

    def _shade_done(self, r):
        g, key, COL = r
        if g.cur is None or COL.shape[:2] != g.cur['COL'].shape[:2]:
            return
        g._shade = (key, COL)
        if key[0] == g.version:
            self.overlay.cache.pop(g.oid, None)
            self.refresh()

    def target(self):
        """the object the hair tool works on: the selected one, else the one under the area brush"""
        sel = list(getattr(self.win.viewport, 'selected', []) or [])
        if sel:
            return int(sel[0])
        return self.area_target

    # ---- the area brush --------------------------------------------------------------------------------------------------
    def area_on(self):
        """the painted area shows while painting, and until hair grows on it"""
        if self.tool == 'area':
            return True
        return any(A and oid not in self.grooms for oid, A in self.area.items())

    def paint_area(self, pos, erase=False):
        fr = _frame(self.win)
        if fr is None:
            return 0
        fx, fy, w, h = _frame_px(self.win, pos)
        obj, dep = np.asarray(fr['obj']), np.asarray(fr['depth'])
        sc = w / max(self.win.viewport.width(), 1)
        r = max(1.0, self.radius_px * sc)
        x0, x1 = int(max(0, fx - r)), int(min(w - 1, fx + r))
        y0, y1 = int(max(0, fy - r)), int(min(h - 1, fy + r))
        if x1 < x0 or y1 < y0:
            return 0
        J, I = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        I, J = I.ravel(), J.ravel()
        inside = (I - fx) ** 2 + (J - fy) ** 2 <= r * r
        I, J = I[inside], J[inside]
        ids = obj[J, I]
        if self.area_target is None:
            ix, iy = int(min(max(fx, 0), w - 1)), int(min(max(fy, 0), h - 1))
            if obj[iy, ix] < 0:
                return 0
            self.area_target = int(obj[iy, ix])
        keep = ids == self.area_target
        I, J = I[keep], J[keep]
        if not len(I):
            return 0
        O, D = _rays(fr.get('cam') or self.win.viewport.cam, I.astype(float), J.astype(float), w, h)
        P = O + D * dep[J, I].astype(float)[:, None]
        fm = Frame(self.win.viewport.transforms.get(self.area_target))
        Q = fm.to_local(P)
        keys = np.floor(Q / AREA_CELL).astype(np.int64)
        A = self.area.setdefault(self.area_target, {})
        n0 = len(A)
        if erase:
            for kk in map(tuple, keys):
                A.pop(kk, None)
        else:
            for kk, q in zip(map(tuple, keys), Q):
                if kk not in A:
                    A[kk] = q
        return len(A) - n0

    def area_cm2(self, oid=None):
        oid = self.area_target if oid is None else oid
        A = self.area.get(oid) or {}
        return len(A) * (AREA_CELL * 100) ** 2 / 1.5          # one sample per occupied cell: ~cell^2 / 1.5 a sample

    def draw_area(self, fr, rgb):
        """the painted area tinted on the form (its samples in front of the form, a small disc each)"""
        out = rgb
        cam = fr['cam']
        depth = np.asarray(fr['depth'])
        H, W = rgb.shape[:2]
        for oid, A in self.area.items():
            if not A or (oid in self.grooms and self.tool != 'area'):
                continue
            fm = Frame(self.win.viewport.transforms.get(oid))
            Pw = fm.to_world(np.array(list(A.values())))
            f, r, u = cam.basis()
            q = Pw - cam.pos
            zf = q @ f
            ok = zf > 1e-4
            th = math.tan(math.radians(cam.fov) * 0.5)
            px = (q @ r) / np.maximum(zf, 1e-9)
            py = (q @ u) / np.maximum(zf, 1e-9)
            sx = ((px / (th * W / H) + 1) * 0.5 * W).astype(int)
            sy = ((1 - py / th) * 0.5 * H).astype(int)
            t = np.linalg.norm(q, axis=1)
            ok &= (sx >= 0) & (sy >= 0) & (sx < W) & (sy < H)
            sx, sy, t = sx[ok], sy[ok], t[ok]
            vis = t <= depth[sy, sx] + 0.002
            sx, sy = sx[vis], sy[vis]
            if out is rgb:
                out = rgb.copy()
            m = np.zeros((H, W), bool)
            # each sample covers its 1.5 mm cell: a disc as wide as the cell is on screen (no gaps up close)
            from ..render.raymarch import pixel_angle
            tv = t[vis] if len(t[vis]) else np.array([1.0])
            rpx = int(np.clip(np.ceil(0.6 * AREA_CELL * fm.s / max(float(np.median(tv)) * pixel_angle(cam, H), 1e-9)), 1, 12))
            for dy in range(-rpx, rpx + 1):
                for dx in range(-rpx, rpx + 1):
                    if dx * dx + dy * dy <= rpx * rpx + 1:
                        m[np.clip(sy + dy, 0, H - 1), np.clip(sx + dx, 0, W - 1)] = True
            blk = out[m].astype(np.float32)
            out[m] = (blk * 0.55 + np.array([214, 150, 60], np.float32) * 0.45).astype(np.uint8)
        return out

    def clear_area(self):
        self.area.pop(self.area_target, None)
        self.refresh()

    # ---- generate --------------------------------------------------------------------------------------------------------
    def generate(self, settings):
        oid = self.area_target if self.area_target is not None else self.target()
        A = self.area.get(oid) or {}
        if oid is None or len(A) < 3:
            self._say('paint an area first (Area, then drag on the form).')
            return False
        if self.busy:
            self._say('the generator is still running.')
            return False
        self.busy = True
        fm = Frame(self.win.viewport.transforms.get(oid))
        Ql = np.array(list(A.values()))
        Qw = fm.to_world(Ql)
        Lm = float(settings.get('length_mm', 300)) / 1000.0
        lo, hi = HG.body_box(Qw, Lm)
        h = 0.003
        shape = tuple(int(s) for s in np.ceil((hi - lo) / h).astype(int) + 1)
        n = int(np.prod(shape))
        ii = np.indices(shape).reshape(3, -1).T
        vals = np.empty(n, np.float32)
        t0 = time.perf_counter()
        chunk = 1 << 20
        self._say('sampling the form for the hair (%d points)...' % n, 60000)
        state = {'c0': 0, 'token': object()}
        eng = self.win.engine

        def post_next():
            c0 = state['c0']
            if c0 >= n:
                eng.callDone.disconnect(got)
                A_ = np.nan_to_num(vals.reshape(shape), nan=1.0, posinf=1.0, neginf=-1.0)
                from ..hair.style import BodyGrid
                grid = BodyGrid(lo, h, A_)
                self.timings.append(('body_field_s', round(time.perf_counter() - t0, 2)))
                # fs-shell 11-13: the generator may use the card: joined by the exit rule (ui/exit_rule.py)
                th = card_thread(self._gen_thread, name='hair-generate', args=(oid, fm, Ql, Qw, settings, grid))
                return
            P = lo + ii[c0:c0 + chunk] * h

            def fn(e, P=P):
                return e.hist.scene.evaluate(P)
            state['token'] = object()
            eng.post('call', fn, state['token'])

        def got(r):
            if not isinstance(r, tuple) or r[0] is not state['token']:
                return
            if getattr(r[1], 'failed', False):          # fs-brush cba5ef61: a failed call still answers (CallFailed)
                eng.callDone.disconnect(got)
                self._say('the form could not be sampled for the hair: %s' % getattr(r[1], 'error', r[1]), 20000)
                _report(self.win, 'Sampling the form for the hair', str(getattr(r[1], 'error', r[1])),
                        details=getattr(r[1], 'trace', None))
                return
            c0 = state['c0']
            vals[c0:c0 + chunk] = np.asarray(r[1], np.float32)
            state['c0'] = c0 + chunk
            if (state['c0'] // chunk) % 4 == 0:
                self._say('sampling the form %d%%' % int(100 * min(state['c0'], n) / n), 60000)
            post_next()
        eng.callDone.connect(got)
        post_next()
        return True

    def _gen_thread(self, oid, fm, Ql, Qw, settings, grid):
        try:
            t0 = time.perf_counter()
            # normals and the area per sample from the field
            f, g = grid.sample(Qw)
            Nw = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            area = (AREA_CELL * 100) ** 2 / np.maximum(np.abs(fm.dir_local(Nw)).sum(1), 1.0)
            res = HG.generate(Qw, Nw, area, settings, grid, log=lambda m: self.bridge.progress.emit(m),
                              card=self.use_card())
            Pl = fm.to_local(res['P'].reshape(-1, 3)).reshape(res['P'].shape).astype(np.float32)
            RAD = res['RAD'] / fm.s
            # the body grid in the object's frame (resampled from world)
            lgrid = local_grid(grid, fm, Pl)
            from ..hair import style as ST
            from ..hair.groom import HairGroom
            gr = self.grooms.get(oid) or HairGroom(oid, 'hair on object %d' % oid)
            gr.settings = dict(settings)
            gr.set_base(Pl, RAD, res['COL'], grid=lgrid, report=res['report'],
                        extra={'SIG': res['SIG'], 'SIGP': res['SIGP']})
            A, lo, h = gr._grid_args()
            ST.ftl_pushout(gr.cur['P'], gr.cur['SEG'], np.arange(gr.n), A, lo, h, gr.clearance())
            gr.base['P'] = gr.cur['P'].copy()
            gr.base['SEG'] = gr.cur['SEG'].copy()
            gr.report['total_with_pushout_s'] = round(time.perf_counter() - t0, 1)
            self.bridge.done.emit((oid, gr, None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.bridge.done.emit((oid, None, e))

    def _generated(self, r):
        oid, gr, err = r
        self.busy = False
        if err is not None:
            self._say('the generator failed: %r' % (err,), 20000)
            return
        self.grooms[oid] = gr
        rep = gr.report
        self.timings.append(('generate_s', rep.get('total_with_pushout_s')))

        def pushout_fine(g):
            # the grown hair kept clear of the form by the fine lattice too (where the coarse one misplaces thin forms)
            from ..hair import style as ST
            t = time.perf_counter()
            A, meta, hh = g._grid_args()
            ST.ftl_pushout(g.cur['P'], g.cur['SEG'], np.arange(g.n, dtype=np.int64), A, meta, hh, g.clearance())
            g.base['P'] = g.cur['P'].copy()
            g.version += 1
            self.timings.append(('pushout_fine_s', round(time.perf_counter() - t, 2)))
            self.refresh()
        self._add_fine(oid, gr, then=pushout_fine)
        self._say('grown: %d hairs (%d guides) in %.0f s; lengths p10/50/90 %s mm' % (
            rep.get('roots', 0), rep.get('guides', 0), rep.get('total_with_pushout_s', 0),
            rep.get('length_mm_p10_50_90')), 30000)
        if getattr(self, 'panel', None) is not None:
            self.panel.sync()
        self.refresh()

    # ---- her natural base --------------------------------------------------------------------------------------------------
    def her_oid(self):
        for o in (self.win.snap or {}).get('objects', []):
            b = o.get('base') or {}
            if b.get('kind') == 'fv8store':
                return int(o['id'])
        return None

    def load_her_base(self, path=HER_BASE):
        oid = self.her_oid()
        if oid is None:
            self._say('open her first (File > Open her): her natural base grows on her.')
            return False
        if not os.path.exists(path):
            self._say('her base set is missing: %s' % path)
            _report(self.win, 'Her natural base', 'her base set is missing: %s' % path)
            return False
        t0 = time.perf_counter()
        z = np.load(path, allow_pickle=True)
        P = z['P'].astype(np.float32)                 # her store frame = her object's own frame
        RAD = z['RAD'].astype(np.float32)
        SIGP = z['SIGP'] if 'SIGP' in z.files else None
        COL = HG.base_colour(SIGP.astype(np.float32)) if SIGP is not None else np.full(P.shape, 0.6, np.float32)
        gr = HairGroom(oid, 'her natural base (n2c)')
        gr.set_base(P, RAD, COL, grid=None, report={'source': path, 'strands': int(len(P))},
                    extra=dict({'SIG': z['SIG'], 'SIGP': SIGP},
                               **{k_: z[k_] for k_ in ('ELL', 'TAU', 'BM', 'PIG', 'SEED', 'KIND') if k_ in z.files
                                  and not (k_ == 'PIG' and bool(np.all(z[k_] == 1)))}))     # all-ones PIG: not kept
        self.grooms[oid] = gr
        self.timings.append(('her_base_load_s', round(time.perf_counter() - t0, 2)))
        self._say('her natural base: %d hairs (hair3 n2c). Building her field for the comb...' % len(P), 20000)
        self._her_grid(oid, gr)
        if getattr(self, 'panel', None) is not None:
            self.panel.sync()
        self.refresh()
        return True

    def _sample_lattice(self, oid, lo, hi, h, done, chunk=1 << 20):
        """the form's field on a lattice in the object's frame (lo, hi, h, metres there), sampled on the engine thread
        in chunks (the view keeps drawing between them); done(A) on the UI thread"""
        fm = Frame(self.win.viewport.transforms.get(oid))
        lo = np.asarray(lo, float)
        shape = tuple(int(s) for s in np.ceil((np.asarray(hi, float) - lo) / h).astype(int) + 1)
        n = int(np.prod(shape))
        vals = np.empty(n, np.float32)
        state = {'c0': 0, 'token': None}
        eng = self.win.engine

        def post_next():
            c0 = state['c0']
            if c0 >= n:
                eng.callDone.disconnect(got)
                done(np.nan_to_num(vals.reshape(shape), nan=1.0, posinf=1.0, neginf=-1.0))
                return
            idx = np.arange(c0, min(n, c0 + chunk))
            ii = np.stack(np.unravel_index(idx, shape), 1)
            Pw = fm.to_world(lo + ii * h)

            def fn(e, Pw=Pw):
                return e.hist.scene.evaluate(Pw)
            state['token'] = object()
            eng.post('call', fn, state['token'])

        def got(r):
            if not isinstance(r, tuple) or r[0] is not state['token']:
                return
            if getattr(r[1], 'failed', False):          # fs-brush cba5ef61: a failed call still answers (CallFailed)
                eng.callDone.disconnect(got)
                self._say('her field could not be sampled for the hair: %s' % getattr(r[1], 'error', r[1]), 20000)
                _report(self.win, 'Sampling her field for the hair', str(getattr(r[1], 'error', r[1])),
                        details=getattr(r[1], 'trace', None))
                return
            c0 = state['c0']
            vals[c0:c0 + chunk] = np.asarray(r[1], np.float32) / fm.s
            state['c0'] = c0 + chunk
            post_next()
        eng.callDone.connect(got)
        post_next()

    def _fine_box(self, gr, pad=0.025, h=0.001, cap=24_000_000):
        """the fine lattice's box over the roots (where hair leaves the skin and the brushes work), and its step: 1 mm,
        coarser only if the box would pass cap cells"""
        R = gr.cur['P'][:, 0].astype(np.float64)
        lo = R.min(0) - pad
        hi = R.max(0) + pad
        cells = np.prod(np.ceil((hi - lo) / h) + 1)
        if cells > cap:
            h = h * float((cells / cap) ** (1.0 / 3.0))
        return lo, hi, h

    def _add_fine(self, oid, gr, then=None):
        """the fine lattice over the roots added to the groom's body; then(gr) once it is in"""
        lo, hi, h = self._fine_box(gr)
        t0 = time.perf_counter()

        def done(A):
            if gr.grid is not None:
                gr.grid.set_fine(lo, h, A)
                gr._card_grid = None
            self.timings.append(('fine_field_s', round(time.perf_counter() - t0, 2), round(h * 1000, 2)))
            if then is not None:
                then(gr)
        self._sample_lattice(oid, lo, hi, h, done)

    def _her_grid(self, oid, gr):
        """her body as lattices in her frame for the brushes' contact: 3 mm around all her hair, then 1 mm over her
        head where the hair leaves the skin (engine thread, in chunks)"""
        P = gr.cur['P']
        lo = np.nanmin(P.reshape(-1, 3), 0) - 0.02
        hi = np.nanmax(P.reshape(-1, 3), 0) + 0.02
        h = 0.003
        t0 = time.perf_counter()

        def coarse_done(A):
            from ..hair.style import BodyGrid
            gr.grid = BodyGrid(lo, h, A)

            def fine_done(gr_):
                self.timings.append(('her_field_s', round(time.perf_counter() - t0, 2)))
                self._say('her field ready for the brushes (%.0f s; 1 mm over her head).' % (time.perf_counter() - t0))
            self._add_fine(oid, gr, then=fine_done)
        self._sample_lattice(oid, lo, hi, h, coarse_done)

    # ---- style brushes -------------------------------------------------------------------------------------------------------
    def _groom_here(self):
        oid = self.target()
        if oid is None or oid not in self.grooms:
            if len(self.grooms) == 1:
                return next(iter(self.grooms.items()))
            return None, None
        return oid, self.grooms[oid]

    def stroke_begin(self, pos):
        oid, g = self._groom_here()
        if g is None:
            self._say('no hair on this object yet: paint an area and Generate.')
            return False
        self.cur_oid = oid
        self.touched = np.zeros(g.n, np.uint8)
        self.card_stroke = None
        if self.use_card() and not getattr(self, 'card_failed', None) and not getattr(self, 'card_brush_failed', None)                 and os.environ.get('FS_HAIR_CARD_BRUSH', '1') != '0':
            try:
                from ..hair.card_style import CardStroke
                fr, fm, cam_l, W, H, dep_l = self._view(oid)
                buf = self.overlay.bufs.setdefault(oid, HD.StrandBuffers())
                COL = self.shaded(g, self.light_dir(fm), allow=False)
                buf.set((oid, g.version, id(COL)), g.cur['P'], g.cur['RAD'], COL, col_key=id(COL))
                self.card_stroke = CardStroke(g, buf, cam_l, W, H, dep_l, self.tool)
            except Exception:
                import traceback
                print('hair: card brush unavailable, the CPU brushes:', traceback.format_exc()[-800:], flush=True)
                self.card_stroke = None
        if self.card_stroke is None:
            g.begin()
        self.overlay.interactive = True
        self.last = QPointF(pos)
        self.stroke_t0 = time.perf_counter()
        self.dab_ms = []
        self.compose_log = []
        if self.tool == 'cut':
            self._cut_at(pos)
        return True

    def _view(self, oid):
        """the frame, the object's frame, the camera and depth in it: cached per frame (a stroke reads it every mouse
        event; converting a 1M-pixel depth each time cost several ms an event)"""
        fr = _frame(self.win)
        tr = self.win.viewport.transforms.get(oid)
        key = (id(fr['depth']), oid, repr(tr), id(fr.get('cam')))
        c = getattr(self, '_view_cache', None)
        if c is not None and c[0] == key:
            return c[1]
        fm = Frame(tr)
        cam = fr['cam']
        H, W = np.asarray(fr['depth']).shape
        v = (fr, fm, fm.camera(cam), W, H, np.asarray(fr['depth'], np.float64) / fm.s)
        self._view_cache = (key, v)
        return v

    def paint_section(self, pos, add=True):
        oid, g = self._groom_here()
        if g is None:
            self._say('no hair here yet.')
            return 0
        fr, fm, cam_l, W, H, dep_l = self._view(oid)
        fx, fy, w, h = _frame_px(self.win, pos)
        rpx = self.radius_px * w / max(self.win.viewport.width(), 1)
        k = g.section_dab(cam_l, W, H, dep_l, fx, fy, rpx, add=add)
        if k:
            self.refresh_soon()
        return k

    def cut_section(self, mode, length_mm, to_mm=0.0, elev_deg=0.0, angle_deg=0.0, point_mm=0.0):
        """the stylist's cuts on the section (groom.cut_same_length / cut_lift). -> hairs cut"""
        oid, g = self._groom_here()
        if g is None:
            self._say('no hair here yet.')
            return 0
        fm = Frame(self.win.viewport.transforms.get(oid))
        up_l = fm.dir_local(np.array([[0, 1.0, 0]]))[0]
        t = time.perf_counter()
        if mode == 'same':
            ax = None
            if to_mm and abs(to_mm - length_mm) > 1e-6:
                # the graded length runs across the view: left (the from length) to right (the to length)
                f, r, _ = self.win.viewport.cam.basis()
                ax = fm.dir_local(np.asarray(r, float)[None])[0]
            m = g.cut_same_length(length_mm, to_mm if ax is not None else None, ax, point_mm=point_mm)
        else:
            m = g.cut_lift(elev_deg, length_mm, angle_deg=angle_deg, point_mm=point_mm, up_local=up_l)
        self.timings.append(('cut_section', mode, round((time.perf_counter() - t) * 1000, 1), m))
        self._say('%s: %d hairs cut; %s' % ('same length' if mode == 'same' else 'lift and cut', m, stats_words(g.stats())))
        self.refresh()
        return m

    def _card_dab(self, *a, **k):
        """a card dab; on any failure the stroke goes on with the CPU brushes (the card's edits so far are dropped,
        the CPU copy is as it was before the stroke) and the card brushes stay off for the session (logged)"""
        try:
            self.card_stroke.dab(*a, **k)
            return True
        except Exception:
            import traceback
            print('hair: card brush failed, the CPU brushes from now on:', traceback.format_exc()[-800:], flush=True)
            cs = self.card_stroke
            self.card_stroke = None
            self.card_brush_failed = True
            cs.buf.hold = False
            cs.buf.key = None
            g = self.grooms[self.cur_oid]
            g.begin()
            return False

    def _cut_at(self, pos):
        g = self.grooms[self.cur_oid]
        fr, fm, cam_l, W, H, dep_l = self._view(self.cur_oid)
        fx, fy, w, h = _frame_px(self.win, pos)
        rpx = self.radius_px * w / max(self.win.viewport.width(), 1)
        t = time.perf_counter()
        if self.card_stroke is not None and self._card_dab(fx, fy, rpx):
            g.version += 1
            self.dab_ms.append((time.perf_counter() - t) * 1000)
            return
        idx = g.cut_dab(cam_l, W, H, dep_l, fx, fy, rpx)    # the hair in front, off the skin (scissors)
        self.touched[idx] = 1
        self.dab_ms.append((time.perf_counter() - t) * 1000)

    def stroke_move(self, pos):
        oid = getattr(self, 'cur_oid', None)
        if oid is None or self.last is None:
            return
        g = self.grooms[oid]
        if self.tool == 'cut':
            # dabs along the drag every third of a radius
            d = math.hypot(pos.x() - self.last.x(), pos.y() - self.last.y())
            steps = max(1, int(d / max(self.radius_px / 3.0, 1.0)))
            for i in range(1, steps + 1):
                q = QPointF(self.last.x() + (pos.x() - self.last.x()) * i / steps,
                            self.last.y() + (pos.y() - self.last.y()) * i / steps)
                self._cut_at(q)
            self.last = QPointF(pos)
            self.refresh_soon()
            return
        fr, fm, cam_l, W, H, dep_l = self._view(oid)
        fx0, fy0, w, h = _frame_px(self.win, self.last)
        fx1, fy1, _, _ = _frame_px(self.win, pos)
        rpx = self.radius_px * w / max(self.win.viewport.width(), 1)
        if self.tool in ('puff', 'clump', 'smooth'):
            d = math.hypot(fx1 - fx0, fy1 - fy0)
            if d < max(1.0, rpx / 6.0):
                return
            t = time.perf_counter()
            if self.card_stroke is not None and self._card_dab(fx1, fy1, rpx, strength=self.strength):
                g.version += 1
            else:
                g.style_dab(self.tool, cam_l, W, H, dep_l, fx1, fy1, rpx, self.strength, self.touched)
            self.dab_ms.append((time.perf_counter() - t) * 1000)
            self.last = QPointF(pos)
            self.refresh_soon()
            return
        dpx, dpy = fx1 - fx0, fy1 - fy0
        if abs(dpx) + abs(dpy) < 0.5:
            return
        # the drag in the object's frame at the hair's depth under the brush
        ix, iy = int(min(max(fx0, 0), W - 1)), int(min(max(fy0, 0), H - 1))
        t_ = float(dep_l[iy, ix]) if np.isfinite(dep_l[iy, ix]) and dep_l[iy, ix] < 1e3 else getattr(self, '_last_t', None)
        if t_ is None:
            P = g.cur['P']
            t_ = float(np.linalg.norm(np.nanmean(P[:, -1], 0) - cam_l.pos))
        self._last_t = t_
        px_m = t_ * HD.pixel_angle(cam_l, H)
        f, r, u = cam_l.basis()
        drag = (r * dpx - u * dpy) * px_m
        t = time.perf_counter()
        if self.card_stroke is not None and self._card_dab((fx0 + fx1) * 0.5, (fy0 + fy1) * 0.5, rpx,
                                                           strength=self.strength, drag=drag):
            g.version += 1
        else:
            g.comb_dab(cam_l, W, H, dep_l, (fx0 + fx1) * 0.5, (fy0 + fy1) * 0.5, rpx, drag, self.strength,
                       self.touched)
        self.dab_ms.append((time.perf_counter() - t) * 1000)
        self.last = QPointF(pos)
        self.refresh_soon()

    def stroke_end(self):
        oid = getattr(self, 'cur_oid', None)
        if oid is None:
            return
        g = self.grooms[oid]
        cs = getattr(self, 'card_stroke', None)
        if cs is not None:
            t = time.perf_counter()
            try:
                idx, old = cs.finish()
            except Exception:
                import traceback
                print('hair: card stroke could not come back; its edits are dropped:', traceback.format_exc()[-800:],
                      flush=True)
                self.card_brush_failed = True
                cs.buf.hold = False
                cs.buf.key = None
                idx, old = np.zeros(0, np.int64), {kk: g.cur[kk][:0].copy() for kk in ('P', 'RAD', 'COL', 'SEG')}
            self.touched = np.zeros(g.n, np.uint8)
            self.touched[idx] = 1
            if len(idx):
                g.undo_stack.append(dict(idx=idx, label=self.tool or '', **old))
                g.undo_stack = g.undo_stack[-64:]
                g.redo_stack = []
            g.version += 1
            g._box_version = getattr(g, '_box_version', 0) + 1
            self.timings.append(('card_finish_ms', round((time.perf_counter() - t) * 1000, 1)))
            self.card_stroke = None
            g._pending = None
        n_t = int(self.touched.sum()) if self.touched is not None else 0
        if self.tool == 'comb' and n_t and getattr(self.panel, 'settle_after', None) is not None \
                and self.panel.settle_after.isChecked():
            if cs is None:
                g.commit(self.touched, 'comb')
            # the combed hair comes to rest on a worker thread (2.6 s on her 26k combed hairs: no freeze after the
            # release); it lands as its own undo step if nothing changed the hair meanwhile
            sel = np.where(self.touched)[0]
            up_l = Frame(self.win.viewport.transforms.get(oid)).dir_local(np.array([[0, 1.0, 0]]))[0]

            taken = g.take_rows(sel)

            def rest(g=g, taken=taken, up_l=up_l):
                t = time.perf_counter()
                try:
                    r = g.settle_copy(taken, up_local=up_l)
                    self.bridge.rest.emit((g, r, round((time.perf_counter() - t) * 1000, 1)))
                except Exception as e:
                    print('hair: rest after comb failed: %r' % (e,), flush=True)
            card_thread(rest, name='hair-rest')     # fs-shell 11-13: joined by the exit rule (ui/exit_rule.py)
        elif cs is None:
            g.commit(self.touched, self.tool or '')
        g._log(self.tool or 'stroke', strands=n_t, dabs=len(self.dab_ms), card=cs is not None,
               dab_ms_p50=round(float(np.median(self.dab_ms)), 1) if self.dab_ms else None,
               dab_ms_max=round(float(np.max(self.dab_ms)), 1) if self.dab_ms else None)
        cl = self.compose_log
        self.timings.append((self.tool + '_stroke', {
            'strands': n_t, 'dabs': len(self.dab_ms),
            'dab_ms_p50': round(float(np.median(self.dab_ms)), 1) if self.dab_ms else None,
            'redraws': len(cl), 'redraw_ms_p50': round(float(np.median(cl)), 1) if cl else None,
            'redraw_ms_p95': round(float(np.percentile(cl, 95)), 1) if cl else None,
            'stroke_s': round(time.perf_counter() - self.stroke_t0, 2)}))
        self._say('%s: %d hairs; %s' % (self.tool, n_t, stats_words(g.stats())), 15000)
        self.cur_oid = None
        self.last = None
        self.overlay.interactive = False
        self.refresh()

    def clear_all(self, why=None):
        """every groom, its card buffers and the area paint gone (a new scene: the ids restart, so nothing carries over)"""
        for b in list(getattr(self.overlay, 'bufs', {}).values()):
            try:
                b.free()
            except Exception:
                pass
        self.overlay.bufs = {}
        self.overlay.cache = {}
        n = len(self.grooms)
        self.grooms = {}
        self.area = {}
        self.card_stroke = None
        self.cur_oid = None
        self.press = None
        if n and why:
            self._say('hair: %d groom(s) of the previous scene released (%s)' % (n, why))
        return n

    def on_scene(self, snap):
        # F4 (fs-bench 11-12): a new history (open her, new, open a region, open a project) restarts the ids, so a groom
        # keyed by id would be drawn on whatever takes its id: on a new scene generation every groom goes
        gen = (snap or {}).get('scene_gen')
        if gen is not None and gen != getattr(self, '_scene_gen', None):
            first = getattr(self, '_scene_gen', None) is None
            self._scene_gen = gen
            if not first and self.clear_all('a new scene'):
                if getattr(self, 'panel', None) is not None:
                    self.panel.sync()
                self.refresh()
        ids = {int(o['id']) for o in (snap or {}).get('objects', [])}
        gone = [oid for oid in self.grooms if oid not in ids]
        for oid in gone:
            self.grooms.pop(oid, None)
            self.overlay.cache.pop(oid, None)
            self.overlay.bufs.pop(oid, None)
        for oid in [o for o in self.area if o not in ids]:
            self.area.pop(oid, None)
        if gone:
            self.refresh()

    # ---- the project: hair beside it (<project>.hair/: one npz + json per hair; outside the save worker's folder swap) --
    def on_engine_message(self, m):
        m = str(m)
        if not m.startswith('Saved '):
            return
        path = m[6:].split(' (state hash')[0].strip()
        if not path or 'autosave' in path.lower():
            return
        try:
            self.save_to(path)
        except Exception as e:
            self._say('hair not saved: %r' % (e,), 20000)

    def save_to(self, project_path):
        d = os.path.normpath(project_path) + '.hair'
        saved = getattr(self, '_saved', {})
        n = 0
        for oid, g in self.grooms.items():
            if saved.get((d, oid)) == g.version:
                continue
            g.save(d)
            saved[(d, oid)] = g.version
            n += 1
        self._saved = saved
        if n:
            self._say('hair saved beside the project (%d): %s' % (n, d))
        return n

    def on_project_loaded(self, extra):
        p = getattr(self.win, 'project_path', None)
        self.clear_all()
        # this scene's generation is the loaded project's: its snapshot (after this) must not clear the hair loaded here
        self._scene_gen = getattr(getattr(getattr(self.win, 'engine', None), 'hist', None), 'gen', None)
        if not p:
            return
        d = os.path.normpath(p) + '.hair'
        if not os.path.isdir(d):
            return
        from ..hair.groom import HairGroom
        for f in sorted(os.listdir(d)):
            if f.startswith('hair_') and f.endswith('.json'):
                try:
                    g = HairGroom.load(os.path.join(d, f[:-5]))
                    self.grooms[g.oid] = g
                    getattr(self, '_saved', {})[(d, g.oid)] = g.version
                except Exception as e:
                    self._say('hair %s not loaded: %r' % (f, e), 20000)
        if self.grooms:
            self._say('hair loaded: %d' % len(self.grooms))
        if getattr(self, 'panel', None) is not None:
            self.panel.sync()
        self.refresh()

    # ---- the brush circle (a transparent child over the viewport) --------------------------------------------------------
    def draw_badge(self, p):
        if self.tool is None:
            return
        text = {'area': 'Hair area  -  drag on the form to paint where hair grows; Ctrl erases; Esc leaves',
                'comb': 'Comb  -  drag through the hair; roots stay, lengths kept; Esc leaves',
                'cut': 'Cut  -  drag across the hair: each hair ends where the scissors meet it; Esc leaves',
                'puff': 'Puff  -  drag over the roots to lift them (volume); Esc leaves',
                'clump': 'Clump  -  drag along the hair to draw it into a lock; Esc leaves',
                'smooth': 'Smooth  -  drag along the hair to relax its kinks; Esc leaves',
                'section': 'Section  -  drag over the roots to take them into the section (Ctrl takes them out); the rest is '
                           'pinned away; Esc leaves'}[self.tool]
        p.setFont(QFont('Segoe UI', 10, QFont.DemiBold))
        r = QRectF(12, 44, 9 * len(text) + 24, 26)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(20, 22, 26, 235))
        p.drawRoundedRect(r, 5, 5)
        p.setPen(HAIR_COL)
        p.drawText(r.adjusted(10, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        mp = getattr(self, 'mouse', None)
        if mp is not None:
            p.setPen(QPen(HAIR_COL, 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(mp, self.radius_px, self.radius_px)


class _Glass(QWidget):
    """a transparent child over the viewport for the hair tool's badge and brush circle"""

    def __init__(self, layer, vp):
        super().__init__(vp)
        self.L = layer
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(vp.size())
        self.show()

    def paintEvent(self, e):
        if self.L.tool is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        self.L.draw_badge(p)
        p.end()


class _HairFilter(QObject):
    def __init__(self, layer):
        super().__init__(layer)
        self.L = layer

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        L = self.L
        t = ev.type()
        if t == QEvent.Resize and getattr(L, 'glass', None) is not None:
            L.glass.resize(L.win.viewport.size())
            return False
        if L.tool is None:
            return False
        if t == QEvent.MouseMove:
            L.mouse = QPointF(ev.position())
            L.glass.update()
        if t in (QEvent.ShortcutOverride, QEvent.KeyPress) and ev.key() == Qt.Key_Escape:
            if t == QEvent.ShortcutOverride:
                ev.accept()
                return True
            L.panel.set_tool(None)
            return True
        if t == QEvent.Wheel and ev.modifiers() & Qt.ControlModifier:
            L.radius_px = float(np.clip(L.radius_px * (1.1 if ev.angleDelta().y() > 0 else 0.9), 4, 600))
            L.glass.update()
            return True
        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            L.press = QPointF(ev.position())
            if L.tool == 'area':
                L.paint_area(ev.position(), erase=bool(ev.modifiers() & Qt.ControlModifier))
                L.panel.sync()
                L.refresh()
            elif L.tool == 'section':
                L.paint_section(ev.position(), add=not bool(ev.modifiers() & Qt.ControlModifier))
            else:
                if not L.stroke_begin(ev.position()):
                    L.press = None
            return True
        if t == QEvent.MouseMove and L.press is not None:
            if L.tool == 'area':
                L.paint_area(ev.position(), erase=bool(ev.modifiers() & Qt.ControlModifier))
                L.panel.sync()
                L.refresh_soon()
            elif L.tool == 'section':
                L.paint_section(ev.position(), add=not bool(ev.modifiers() & Qt.ControlModifier))
            else:
                L.stroke_move(ev.position())
            return True
        if t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and L.press is not None:
            L.press = None
            if L.tool == 'section':
                L.panel.sync()
                L.refresh()
            elif L.tool != 'area':
                L.stroke_end()
            L.panel.sync()
            return True
        return False


def local_grid(grid, fm, Pl, h=None, pad=0.03):
    """the world body lattice resampled in the object's frame over the hair's box"""
    from ..hair.style import BodyGrid
    h = grid.h / fm.s if h is None else h
    pts = Pl.reshape(-1, 3)
    pts = pts[np.isfinite(pts[:, 0])]
    lo = pts.min(0) - pad / fm.s
    hi = pts.max(0) + pad / fm.s
    shape = tuple(int(s) for s in np.ceil((hi - lo) / h).astype(int) + 1)
    ii = np.indices(shape).reshape(3, -1).T
    out = np.empty(len(ii), np.float32)
    for c0 in range(0, len(ii), 1 << 21):
        Pw = fm.to_world(lo + ii[c0:c0 + (1 << 21)] * h)
        f, _ = grid.sample(Pw)
        out[c0:c0 + (1 << 21)] = f / fm.s
    return BodyGrid(lo, h, out.reshape(shape))


class HairPanel(QWidget):
    def __init__(self, layer):
        super().__init__()
        self.L = layer
        layer.panel = self
        v = QVBoxLayout(self)
        # kind and area
        g1 = QGroupBox('1  What grows, and where')
        f1 = QFormLayout(g1)
        self.kind = QComboBox()
        for k, (lab, *_r) in HG.KINDS.items():
            self.kind.addItem(lab, k)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        f1.addRow('Kind', self.kind)
        row = QHBoxLayout()
        self.b_area = QPushButton('Area brush')
        self.b_area.setCheckable(True)
        self.b_area.clicked.connect(lambda on: self.set_tool('area' if on else None))
        row.addWidget(self.b_area)
        b = QPushButton('Clear area')
        b.clicked.connect(lambda: (self.L.clear_area(), self.sync()))
        row.addWidget(b)
        f1.addRow(row)
        self.area_lab = QLabel('area 0 cm2')
        f1.addRow(self.area_lab)
        v.addWidget(g1)
        # details
        g2 = QGroupBox('2  Details')
        f2 = QFormLayout(g2)
        self.colour = QComboBox()
        for k in COLOURS:
            self.colour.addItem(k)
        self.colour.currentIndexChanged.connect(self._colour_changed)
        f2.addRow('Colour', self.colour)
        self.melanin = QDoubleSpinBox()
        self.melanin.setRange(0.01, 0.99)
        self.melanin.setSingleStep(0.02)
        self.redness = QDoubleSpinBox()
        self.redness.setRange(0.0, 1.0)
        self.redness.setSingleStep(0.02)
        self.ash = QDoubleSpinBox()
        self.ash.setRange(0.0, 1.0)
        self.ash.setSingleStep(0.02)
        f2.addRow('Melanin', self.melanin)
        f2.addRow('Redness', self.redness)
        f2.addRow('Ash', self.ash)
        self.thick = QDoubleSpinBox()
        self.thick.setRange(10, 200)
        self.thick.setSuffix(' um')
        f2.addRow('Thickness', self.thick)
        self.density = QDoubleSpinBox()
        self.density.setRange(1, 600)
        self.density.setSuffix(' /cm2')
        f2.addRow('Density', self.density)
        self.length = QDoubleSpinBox()
        self.length.setRange(0.5, 1200)
        self.length.setSuffix(' mm')
        f2.addRow('Length', self.length)
        self.seed = QSpinBox()
        self.seed.setRange(1, 99999)
        f2.addRow('Seed', self.seed)
        self.preview = QCheckBox('Preview (1 hair in 10, about 10x faster)')
        self.preview.setToolTip('grows a tenth of the hairs (and guides), each drawn covering ten, so the density and '
                                'colour read the same; untick and Generate again for every hair')
        f2.addRow(self.preview)
        self.b_gen = QPushButton('Generate')
        self.b_gen.setStyleSheet('font-weight: 600;')
        self.b_gen.clicked.connect(self.generate)
        f2.addRow(self.b_gen)
        v.addWidget(g2)
        # style
        g3 = QGroupBox('3  Shorten, move and style')
        f3 = QFormLayout(g3)
        row = QHBoxLayout()
        self.b_comb = QPushButton('Comb')
        self.b_comb.setCheckable(True)
        self.b_comb.clicked.connect(lambda on: self.set_tool('comb' if on else None))
        self.b_cut = QPushButton('Cut')
        self.b_cut.setCheckable(True)
        self.b_cut.clicked.connect(lambda on: self.set_tool('cut' if on else None))
        row.addWidget(self.b_comb)
        row.addWidget(self.b_cut)
        f3.addRow(row)
        row = QHBoxLayout()
        self.b_more = {}
        for t_, lab, tip in (('puff', 'Puff', 'lift the roots under the brush toward the skin normal: volume'),
                             ('clump', 'Clump', 'draw the hairs under the brush into a lock (as water, oil or product '
                                                'binds them)'),
                             ('smooth', 'Smooth', 'relax kinks along the hairs under the brush')):
            b = QPushButton(lab)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda on, t_=t_: self.set_tool(t_ if on else None))
            row.addWidget(b)
            self.b_more[t_] = b
        f3.addRow(row)
        self.rad = QSpinBox()
        self.rad.setRange(4, 600)
        self.rad.setValue(int(self.L.radius_px))
        self.rad.setSuffix(' px')
        self.rad.valueChanged.connect(lambda x: setattr(self.L, 'radius_px', float(x)))
        f3.addRow('Brush (Ctrl+wheel)', self.rad)
        self.stren = QDoubleSpinBox()
        self.stren.setRange(0.05, 1.0)
        self.stren.setSingleStep(0.05)
        self.stren.setValue(self.L.strength)
        self.stren.valueChanged.connect(lambda x: setattr(self.L, 'strength', float(x)))
        f3.addRow('Comb strength', self.stren)
        self.settle_after = QCheckBox('Let combed hair come to rest')
        self.settle_after.setChecked(True)
        f3.addRow(self.settle_after)
        row = QHBoxLayout()
        self.trim_mm = QDoubleSpinBox()
        self.trim_mm.setRange(1, 1200)
        self.trim_mm.setValue(150)
        self.trim_mm.setSuffix(' mm')
        bt = QPushButton('Trim all to')
        bt.clicked.connect(self.trim)
        row.addWidget(bt)
        row.addWidget(self.trim_mm)
        f3.addRow(row)
        row = QHBoxLayout()
        for lab, fn in (('Settle', self.settle), ('Water (wash out the set)', self.wash), ('Undo', self.undo)):
            b = QPushButton(lab)
            b.clicked.connect(fn)
            row.addWidget(b)
        f3.addRow(row)
        v.addWidget(g3)
        # sections and stylist cuts (11-11)
        g35 = QGroupBox('Sections and cuts (a stylist\'s way)')
        f35 = QFormLayout(g35)
        row = QHBoxLayout()
        self.b_section = QPushButton('Section brush')
        self.b_section.setCheckable(True)
        self.b_section.setToolTip('drag over the roots to take them into the section (Ctrl takes them out); the brushes '
                                  'and cuts act on the section only')
        self.b_section.clicked.connect(lambda on: self.set_tool('section' if on else None))
        row.addWidget(self.b_section)
        for lab, fn in (('All hair', self.section_all), ('Invert', self.section_invert)):
            b = QPushButton(lab)
            b.clicked.connect(fn)
            row.addWidget(b)
        f35.addRow(row)
        self.others = QComboBox()
        for k_, lab in (('dim', 'Dimmed'), ('hide', 'Hidden'), ('show', 'Shown')):
            self.others.addItem(lab, k_)
        self.others.currentIndexChanged.connect(self._others)
        f35.addRow('The rest of the hair', self.others)
        self.cut_mode = QComboBox()
        self.cut_mode.addItem('Same length (travelling guide)', 'same')
        self.cut_mode.addItem('Lift and cut (stationary guide)', 'lift')
        f35.addRow('Cut', self.cut_mode)
        self.cut_len = QDoubleSpinBox()
        self.cut_len.setRange(2, 1200)
        self.cut_len.setValue(150)
        self.cut_len.setSuffix(' mm')
        f35.addRow('Length (guide)', self.cut_len)
        self.cut_to = QDoubleSpinBox()
        self.cut_to.setRange(0, 1200)
        self.cut_to.setSuffix(' mm')
        self.cut_to.setToolTip('same length only: the length runs from Length (screen left) to this (screen right); 0 = none')
        f35.addRow('To (graded)', self.cut_to)
        self.cut_elev = QDoubleSpinBox()
        self.cut_elev.setRange(0, 180)
        self.cut_elev.setSuffix(' deg')
        self.cut_elev.setToolTip('lift and cut: 0 one length, 90 layers, 180 long layers (shorter on top)')
        f35.addRow('Elevation', self.cut_elev)
        self.cut_angle = QDoubleSpinBox()
        self.cut_angle.setRange(-80, 80)
        self.cut_angle.setSuffix(' deg')
        self.cut_angle.setToolTip('lift and cut: the cutting line tilted (an A-line, a face frame)')
        f35.addRow('Angle', self.cut_angle)
        self.cut_point = QDoubleSpinBox()
        self.cut_point.setRange(0, 40)
        self.cut_point.setValue(6)
        self.cut_point.setSuffix(' mm')
        self.cut_point.setToolTip('point cut: each end shortened by a random 0 .. this, a soft edge instead of a blunt line')
        f35.addRow('Point cut', self.cut_point)
        bcs = QPushButton('Cut the section')
        bcs.clicked.connect(self.cut_section)
        f35.addRow(bcs)
        v.addWidget(g35)
        # states
        g4 = QGroupBox('4  States (styles on the natural base)')
        f4 = QVBoxLayout(g4)
        self.states = QListWidget()
        self.states.setMaximumHeight(90)
        f4.addWidget(self.states)
        row = QHBoxLayout()
        bs = QPushButton('Keep as state')
        bs.clicked.connect(self.save_state)
        bsh = QPushButton('Show state')
        bsh.clicked.connect(self.show_state)
        row.addWidget(bs)
        row.addWidget(bsh)
        f4.addLayout(row)
        v.addWidget(g4)
        g5 = QGroupBox('Her')
        f5 = QHBoxLayout(g5)
        bh = QPushButton('Her natural base')
        bh.setToolTip('hair3 n2c: grown by physics, part + fingers + air-dried; two changes (11-11)')
        bh.clicked.connect(lambda: self.L.load_her_base())
        f5.addWidget(bh)
        be = QPushButton('Export for the review render')
        be.setToolTip('the current hair as a HairSet (.npz) in the object frame, for the review renderer (HAIR_SET)')
        be.clicked.connect(self.export)
        f5.addWidget(be)
        self.vis = QCheckBox('Show hair')
        self.vis.setChecked(True)
        self.vis.toggled.connect(self._vis)
        f5.addWidget(self.vis)
        v.addWidget(g5)
        self.info = QLabel('')
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        v.addStretch(1)
        self._kind_changed()
        self._colour_changed()

    def _kind_changed(self, *_):
        d = HG.defaults(self.kind.currentData())
        self.thick.setValue(d['diameter_um'])
        self.density.setValue(d['density'])
        self.length.setValue(d['length_mm'])

    def _colour_changed(self, *_):
        m, r, a = COLOURS[self.colour.currentText()]
        self.melanin.setValue(m)
        self.redness.setValue(r)
        self.ash.setValue(a)

    def settings(self):
        return {'kind': self.kind.currentData(), 'density': self.density.value(), 'length_mm': self.length.value(),
                'diameter_um': self.thick.value(), 'melanin': self.melanin.value(), 'redness': self.redness.value(),
                'ash': self.ash.value(), 'seed': self.seed.value(),
                'preview': 0.1 if self.preview.isChecked() else 1.0}

    def set_tool(self, tool):
        self.L.tool = tool
        for b, t in ((self.b_area, 'area'), (self.b_comb, 'comb'), (self.b_cut, 'cut')) + tuple(
                (b_, t_) for t_, b_ in getattr(self, 'b_more', {}).items()) + tuple(
                [(self.b_section, 'section')] if hasattr(self, 'b_section') else []):
            b.setChecked(tool == t)
        self.L.glass.update()
        self.L.refresh()

    def sync(self):
        self.area_lab.setText('area %.1f cm2 on object %s' % (self.L.area_cm2(), self.L.area_target))
        oid, g = self.L._groom_here()
        self.states.clear()
        if g is not None:
            for nm in g.states:
                self.states.addItem(nm)

    def generate(self):
        self.L.generate(self.settings())

    def _g(self):
        oid, g = self.L._groom_here()
        if g is None:
            self.L._say('no hair here yet.')
        return g

    def trim(self):
        g = self._g()
        if g is not None:
            m = g.trim(self.trim_mm.value() / 1000.0)
            self.L._say('trimmed %d hairs to %.0f mm; %s' % (m, self.trim_mm.value(), stats_words(g.stats())))
            self.L.refresh()

    def settle(self):
        g = self._g()
        if g is not None:
            t = time.perf_counter()
            g.settle(iters=30, set_frac=0.85)
            self.L._say('settled in %.1f s; %s' % (time.perf_counter() - t, stats_words(g.stats())))
            self.L.refresh()

    def wash(self):
        g = self._g()
        if g is not None:
            g.wash()
            self.L._say('washed: back to the natural fall (cuts kept).')
            self.L.refresh()

    def undo(self):
        g = self._g()
        if g is not None and g.undo():
            self.L.refresh()

    def save_state(self):
        g = self._g()
        if g is not None:
            nm = 'style %d' % (len(g.states) + 1)
            k = g.save_state(nm)
            self.sync()
            self.L._say('kept "%s" (%d hairs changed from the natural base)' % (nm, k))

    def show_state(self):
        g = self._g()
        it = self.states.currentItem()
        if g is not None and it is not None:
            g.show_state(it.text())
            self.L.refresh()

    def export(self, path=None):
        oid, g = self.L._groom_here()
        if g is None:
            self.L._say('no hair to export.')
            return None
        if path is None:
            d = os.environ.get('FS_HAIR_EXPORT_DIR') or os.path.join(os.path.expanduser('~'), 'FieldStudio', 'hair')
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, 'hs_fs_%d_%s.npz' % (oid, time.strftime('%Y%m%d_%H%M%S')))
        hs = g.to_hairset()
        hs.save(path)
        self.L._say('exported %d hairs: %s' % (g.n, path), 20000)
        return path

    def section_all(self):
        oid, g = self.L._groom_here()
        if g is not None:
            g.set_section(None)
            self.L.refresh()

    def section_invert(self):
        oid, g = self.L._groom_here()
        if g is not None:
            sec = getattr(g, 'section', None)
            g.set_section(np.zeros(g.n, np.uint8) if sec is None else (1 - sec.astype(np.uint8)))
            self.L.refresh()

    def _others(self, *_):
        self.L.others = self.others.currentData()
        self.L.refresh()

    def cut_section(self):
        self.L.cut_section(self.cut_mode.currentData(), self.cut_len.value(), self.cut_to.value(), self.cut_elev.value(),
                           self.cut_angle.value(), self.cut_point.value())

    def _vis(self, on):
        for g in self.L.grooms.values():
            g.visible = bool(on)
            g.version += 1
        self.L.refresh()


def install(win):
    L = HairLayer(win)
    win.hair = L
    vp = win.viewport
    if not hasattr(vp, 'overlays'):
        vp.overlays = []
    vp.overlays.append(L.overlay)
    L.glass = _Glass(L, vp)
    L.filter = _HairFilter(L)
    vp.installEventFilter(L.filter)
    panel = HairPanel(L)
    d = QDockWidget('Hair')
    # the panel scrolls inside its dock (fs-shell 22:5x: unscrolled, its 1385 px minimum height forced the whole window
    # to >= 1633 px: larger than a 1080-line screen, and every window test's viewport grew to 1070 x 1339)
    from PySide6.QtWidgets import QFrame, QScrollArea
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setWidget(panel)
    d.setWidget(sa)
    d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
    d.setMinimumWidth(320)
    win.addDockWidget(Qt.RightDockWidgetArea, d)
    docks = getattr(win, 'docks', {})
    if 'Properties' in docks:
        win.tabifyDockWidget(docks['Properties'], d)
    docks['Hair'] = d
    L.dock = d
    win.engine.sceneChanged.connect(L.on_scene)              # an object deleted (or a new scene): its hair goes
    win.engine.message.connect(L.on_engine_message)          # hair saved beside the project after a Save
    win.engine.projectLoaded.connect(L.on_project_loaded)
    tm = getattr(win, 'menu_tools', None)
    if tm is not None:
        a = tm.addAction('Hair: grow, cut and comb...')
        a.triggered.connect(lambda: (d.show(), d.raise_()))
    return L
