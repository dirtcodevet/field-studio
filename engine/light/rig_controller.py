"""The light tools' model (fs-light, 11-11; STUDY section e): what a Light panel in Field Studio drives. No Qt here:
the panel (with fs-shell's install hook) calls these and hands `light_dict()` to the viewport's renders
(GpuScene.light_params(rig=...), fcff006d). A / B slots carry the side-by-side light test; save / load keep rigs next
to the project.

    C = RigController(hdri_dir)                    # starts from Field Studio's own light, bit for bit
    C.set_sun(azimuth_deg=40, elevation_deg=35)    # degrees about her (y up): 0 = from her front, + toward her right
    C.set_temperature('sun', 4300); C.set_strength('sun', 1.2); C.set_sun_size(0.53)
    C.set_environment(rotation_deg=90)            # the HDRI's rotation; or C.set_environment(hdri=<folder>)
    C.apply_preset('rembrandt', subject, front)    # studio rigs (rig.PRESETS)
    C.solo('key softbox 60x90'); C.mute('fill reflector'); C.set_quality('lookdev'); C.set_bounce(True)
    C.store('A'); ... C.store('B'); C.recall('A')  # the side-by-side test's two rigs
    C.save(path); C.load(path)
    light = C.light_dict()                         # -> GpuScene render / progress `light=` argument
"""
import copy
import math
import os

import numpy as np

try:
    from . import rig as RG          # noqa
except ImportError:
    import rig as RG


class RigController:
    def __init__(self, hdri_dir, front=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)):
        self.hdri_dir = hdri_dir
        self.rig = RG.fieldstudio_default(hdri_dir)
        self.front = np.asarray(front, float) / np.linalg.norm(front)
        self.up = np.asarray(up, float) / np.linalg.norm(up)
        self.slots = {}
        self.bounce = False
        self.solo_name = None
        self.listeners = []

    # ---- change notification (the panel and the viewport subscribe) ----
    def on_change(self, fn):
        self.listeners.append(fn)

    def _changed(self, what):
        for fn in list(self.listeners):
            fn(what)

    # ---- lights ----
    def _light(self, name):
        if name == 'sun':
            return self.rig.sun()
        for l in self.rig.lights:
            if l.name == name:
                return l
        raise KeyError(name)

    def names(self):
        return [l.name or l.kind for l in self.rig.lights]

    def key_twin(self):
        """a studio preset's key is two lights: the softbox (sampled in look-dev) and its twin sun (what the plain light
        draws: rig.studio_preset). The key controls move, colour, dim and size both; None when the sun is its own light"""
        s = self.rig.sun()
        if s is None or not (s.name or '').startswith('key as sun'):
            return None
        return next((l for l in self.rig.lights if l.kind != 'sun' and l.group == 'studio'), None)

    def _twin_dist(self, t):
        if self.rig.subject is None or t.position is None:
            return None
        return float(np.linalg.norm(np.asarray(t.position, float) - np.asarray(self.rig.subject, float)))

    def key_size_deg(self):
        """the key's angular diameter as seen from her (the softbox's longest side, or the sun disc)"""
        t = self.key_twin()
        dist = self._twin_dist(t) if t is not None else None
        if t is not None and dist and t.size:
            return 2.0 * math.degrees(math.atan2(0.5 * max(t.size), dist))
        return 2.0 * self.rig.sun().angular_radius_deg

    def sun_angles(self):
        d = np.asarray(self.rig.sun().direction, float)
        right = np.cross(self.front, self.up)
        el = math.degrees(math.asin(max(-1.0, min(1.0, float(d @ self.up)))))
        az = math.degrees(math.atan2(float(d @ right), float(d @ self.front)))
        return az, el

    def set_sun(self, azimuth_deg=None, elevation_deg=None):
        az0, el0 = self.sun_angles()
        az = az0 if azimuth_deg is None else float(azimuth_deg)
        el = el0 if elevation_deg is None else float(elevation_deg)
        _p, d = RG._place(np.zeros(3), self.front, self.up, az, el, 1.0)
        self.rig.sun().direction = [float(x) for x in d]
        t = self.key_twin()
        dist = self._twin_dist(t) if t is not None else None
        if dist:                                     # the softbox moves with it, at its own distance from her
            t.position = [float(x) for x in np.asarray(self.rig.subject, float) + d * dist]
            t.direction = [float(x) for x in d]
        self._changed('sun')

    def set_temperature(self, name, kelvin):
        l = self._light(name)
        s = float(np.mean(l.rgb)) if l.rgb else 1.0
        k = RG.kelvin_rgb(kelvin)
        m = float(np.mean(k))                    # the colour only: its mean held (a slider drag must not dim the light)
        l.rgb = [c / m * s for c in k]
        l.note = '%d K' % int(kelvin)
        t = self.key_twin()
        if t is not None and (l is t or l is self.rig.sun()):        # the key pair: both take the colour
            for o in (t, self.rig.sun()):
                if o is not l:
                    so = float(np.mean(o.rgb)) if o.rgb else 1.0
                    o.rgb = [c / m * so for c in k]
        self._changed(name)

    def set_strength(self, name, strength):
        l = self._light(name)
        l.strength = float(strength)
        t = self.key_twin()
        if t is not None and (l is t or l is self.rig.sun()):
            t.strength = float(strength); self.rig.sun().strength = float(strength)
        self._changed(name)

    def set_sun_size(self, angular_diameter_deg):
        """the key's angular DIAMETER as seen from her (the real sun: 0.53 deg). The sun disc the plain light draws is
        capped at 2 deg (the SDF penumbra estimate needs distances inside the stored band); a studio key's softbox takes
        the full size (sampled in look-dev, its longest side scaled, its shape kept)"""
        D = float(angular_diameter_deg)
        self.rig.sun().angular_radius_deg = min(1.0, 0.5 * D)
        t = self.key_twin()
        dist = self._twin_dist(t) if t is not None else None
        if dist and t.size:
            f = 2.0 * dist * math.tan(math.radians(0.5 * min(D, 170.0))) / max(t.size)
            t.size = [float(x) * f for x in t.size]
        self._changed('sun')

    def set_environment(self, hdri=None, rotation_deg=None, background_gain=None, sky_gain=None):
        e = self.rig.environment
        if sky_gain is not None: e.sky_gain = max(0.0, float(sky_gain))
        if hdri is not None: e.hdri = hdri
        if rotation_deg is not None: e.rotation_deg = float(rotation_deg) % 360.0
        if background_gain is not None: e.background_gain = float(background_gain)
        s = self.rig.sun()
        if (hdri is not None or rotation_deg is not None) and e.extract_sun and e.hdri and s is not None and s.name == 'hdri sun':
            d, _, _ = RG.Env(e.hdri).sun_world(e.rotation_deg)    # the map's own sun turns with the map
            s.direction = [float(x) for x in d]
        self._changed('environment')

    def mute(self, name, on=True):
        r = [] if on else ['diffuse', 'specular', 'bounce', 'caustics']
        l = self._light(name)
        l.reaches = list(r)
        t = self.key_twin()
        if t is not None and (l is t or l is self.rig.sun()):        # the key pair is one light to the user
            t.reaches = list(r); self.rig.sun().reaches = list(r)
        self._changed(name)

    def solo(self, name=None):
        self.solo_name = name; self._changed('solo')

    def set_quality(self, q):
        assert q in ('sculpt', 'lookdev', 'final', 'game'); self.rig.quality = q; self._changed('quality')

    def set_bounce(self, on):
        self.bounce = bool(on); self._changed('bounce')

    def apply_preset(self, name, subject, front=None):
        base = self.rig
        r = RG.studio_preset(name, subject, self.front if front is None else front, self.up, base=base)
        r.quality = base.quality if base.quality != 'sculpt' else 'lookdev'
        self.rig = r
        self._changed('preset')

    # ---- slots, files ----
    def store(self, slot):
        self.slots[slot] = copy.deepcopy(self.rig)

    def recall(self, slot):
        self.rig = copy.deepcopy(self.slots[slot]); self._changed('recall')

    def save(self, path):
        self.rig.save(path)

    def load(self, path):
        self.rig = RG.Rig.load(path); self._changed('load')

    # ---- what the renderers get ----
    def effective_rig(self):
        """the rig as rendered: muted lights dropped; with a solo, only that light (the sun kept as the key if it is the solo)"""
        r = copy.deepcopy(self.rig)
        t = self.key_twin()
        solo_sun = self.solo_name == 'sun' or (t is not None and self.solo_name in ((t.name or t.kind), (self.rig.sun().name)))
        keep = []
        for l in r.lights:
            if l.kind in ('env', 'sky_uniform'):
                keep.append(l); continue
            if not l.reaches:
                continue
            if self.solo_name is not None and (l.name or l.kind) != self.solo_name and not (l.kind == 'sun' and solo_sun):
                if l.kind == 'sun':
                    l.strength = 0.0; keep.append(l)
                continue
            keep.append(l)
        if not any(l.kind == 'sun' for l in keep):          # light_pass needs a key: a dark one when muted
            s = copy.deepcopy(self.rig.sun()); s.strength = 0.0; keep.insert(0, s)
        r.lights = keep
        return r

    def light_dict(self):
        return {'rig': self.effective_rig(), 'bounce': self.bounce}
