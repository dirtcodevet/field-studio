"""The light rig: ONE description of a scene's light for Field Studio, the review renderer and a game runtime
(fs-light L1, 11-11; STUDY_light_stage1.md section f). Data only: no renderer imports this yet.

A rig is lights + an environment + exposure + a quality mode. Directions are in Field Studio's world frame (y up,
toward the light); the review renderer's store frame (z up) is reached by R_FV, a signed permutation, so conversions
are exact in floating point.

Converters (each with an exact round-trip gate in rig_test.py):
- fieldstudio_lp(rig, cam_pos, env)  -> the 24-float L array that gpu/scene.py GpuScene.light_params builds today
- review_light(rig)                  -> the {dir, S, sun_rgb, sky_rgb, spd} entry fvsk_spec_r5.lights builds today
- from_review(name, entry)           -> a rig holding that named light

Methods carried by a rig must clear the patent and licence check before they ship (LANES shared rule 11-11 16:58).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

SCHEMA = 1
R_FV = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])      # store (z up) -> world (y up)
SUN_RADIUS_DEG = 0.267                                          # Field Studio's gpu/scene.py constant
KINDS = ('sun', 'sky_uniform', 'env', 'area_rect', 'area_disk', 'sphere', 'spot', 'point', 'emissive')


@dataclass
class Light:
    kind: str                                   # one of KINDS
    name: str = ''
    direction: Optional[List[float]] = None     # sun: unit vector toward the light (world, y up)
    position: Optional[List[float]] = None      # local lights (world, m)
    rgb: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    strength: float = 1.0                        # sun: the multiplier S on rgb (irradiance units of the renderer)
    angular_radius_deg: float = SUN_RADIUS_DEG   # sun disk: sets the penumbra
    size: Optional[List[float]] = None          # area lights (m)
    spot: Optional[List[float]] = None          # [cone deg, blend 0..1]
    spd: Optional[str] = None                   # named illuminant for spectral paths ('D65', 'sun_20deg', ...)
    shadows: bool = True
    reaches: List[str] = field(default_factory=lambda: ['diffuse', 'specular', 'bounce', 'caustics'])
    include_labels: List[str] = field(default_factory=list)   # light linking by material label (empty: all)
    exclude_labels: List[str] = field(default_factory=list)
    group: str = ''
    note: str = ''


@dataclass
class Environment:
    hdri: Optional[str] = None                  # folder with env_2k.npy + env.json (None: no map)
    rotation_deg: float = 0.0
    sky_rgb: Optional[List[float]] = None       # a uniform sky radiance when no map (review renderer's sky_rgb)
    background_gain: float = 1.0
    extract_sun: bool = True                    # the map's sun disk becomes the rig's analytic sun
    sky_gain: float = 1.0                       # the sky's LIGHT on her (0 = a dark studio); background_gain is only the view behind


@dataclass
class Exposure:
    rule: str = 'card'                          # 'card': k = pi / E (FV8's rule), or 'fixed'
    value: Optional[float] = None               # 'fixed': the multiplier
    ev: float = 0.0


@dataclass
class Shading:                                  # Field Studio's present light terms (gpu/scene.py light_params)
    wrap: float = 0.15
    sun_reach_m: float = 10.0
    sky_reach_m: float = 0.15
    sky_rays: int = 2
    spec_level: float = 0.06
    spec_exponent: float = 24.0
    sky_step_cap: int = 64
    sun_step_cap: int = 256
    headlight: float = 0.0
    cavity: float = 1.0


@dataclass
class Rig:
    name: str
    lights: List[Light] = field(default_factory=list)
    environment: Environment = field(default_factory=Environment)
    exposure: Exposure = field(default_factory=Exposure)
    shading: Shading = field(default_factory=Shading)
    quality: str = 'sculpt'                     # 'sculpt' | 'lookdev' | 'final' | 'game'
    tonemap: str = 'aces_fit'
    schema: int = SCHEMA
    note: str = ''
    subject: Optional[List[float]] = None       # the point a studio rig's lights aim at (world, m); None: no studio lights

    def sun(self) -> Optional[Light]:
        return next((l for l in self.lights if l.kind == 'sun'), None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_json(text: str) -> 'Rig':
        d = json.loads(text)
        if d.get('schema', 0) > SCHEMA:
            raise ValueError('rig schema %s is newer than this reader (%s)' % (d.get('schema'), SCHEMA))
        return Rig(name=d['name'], lights=[Light(**l) for l in d.get('lights', [])],
                   environment=Environment(**d.get('environment', {})), exposure=Exposure(**d.get('exposure', {})),
                   shading=Shading(**d.get('shading', {})), quality=d.get('quality', 'sculpt'),
                   tonemap=d.get('tonemap', 'aces_fit'), schema=d.get('schema', SCHEMA), note=d.get('note', ''),
                   subject=d.get('subject'))

    def save(self, path):
        open(path, 'w', encoding='utf-8').write(self.to_json())

    @staticmethod
    def load(path) -> 'Rig':
        return Rig.from_json(open(path, encoding='utf-8').read())


# ---------------- the environment (the same reading as gpu/scene.py, numpy only) ----------------
class Env:
    """env_2k.npy + env.json of an HDRI folder; the sun direction and the 9-term SH irradiance exactly as GpuScene
    computes them (sun_world, sky_sh), for a rotation"""
    def __init__(self, folder):
        self.folder = folder
        self.envj = json.load(open(os.path.join(folder, 'env.json')))
        self._env = None
        sb = np.array(self.envj['sun_dir_hdri'], float)
        self.sun_b = sb / np.linalg.norm(sb)
        self.sun_rgb = np.array(self.envj['sun_irradiance_rgb'], float)

    def env(self):
        if self._env is None:
            self._env = np.load(os.path.join(self.folder, 'env_2k.npy')).astype(np.float32)
        return self._env

    def sun_world(self, rot_deg):
        r = math.radians(rot_deg)
        cr, sr = math.cos(r), math.sin(r)
        b = self.sun_b
        bx, by, bz = cr * b[0] + sr * b[1], -sr * b[0] + cr * b[1], b[2]
        w = np.array([bx, bz, -by])
        return w / np.linalg.norm(w), cr, sr

    def sky_sh(self, rot_deg):
        """the 27 SH irradiance coefficients GpuScene.sky_sh uploads for this rotation (cached per rotation)"""
        c = self.__dict__.setdefault('_sh', {})
        if rot_deg not in c:
            c[rot_deg] = sky_sh_np(self.env(), rot_deg)
        return c[rot_deg]


def sky_sh_np(env_np, rot_deg):
    """GpuScene.sky_sh's arithmetic for a given rotation (the same lines, so a rotation equal to the scene's gives its
    coefficients bit for bit): 9 cosine-convolved SH terms per channel (Ramamoorthi-Hanrahan), world frame, y up"""
    env = env_np
    eh0, ew0 = env.shape[:2]
    env = env.reshape(eh0 // 4, 4, ew0 // 4, 4, 3).mean((1, 3))
    eh, ew = env.shape[:2]
    v = (np.arange(eh) + 0.5) / eh * np.pi
    u = (np.arange(ew) + 0.5) / ew
    V, U = np.meshgrid(v, u, indexing='ij')
    bz = np.cos(V)
    phi = (U - 0.5) * 2 * np.pi
    rx, ry = np.sin(V) * np.cos(phi), np.sin(V) * np.sin(phi)
    r = math.radians(rot_deg)
    cr, sr = math.cos(r), math.sin(r)
    bx, by = cr * rx + sr * ry, -sr * rx + cr * ry
    x, y, z = bx, bz, -by
    dO = (2 * np.pi / ew) * (np.pi / eh) * np.sin(V)
    Y = np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y,
                  1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)], 0)
    A = np.array([np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4,
                  np.pi / 4])
    sh = np.zeros((3, 9), np.float32)
    for c in range(3):
        sh[c] = A * (Y * (env[..., c] * dO)[None]).reshape(9, -1).sum(1)
    return sh.reshape(-1)


def fieldstudio_default(hdri_dir) -> Rig:
    """the rig that reproduces Field Studio's review light today (light_params with its defaults)"""
    e = Env(hdri_dir)
    d, _, _ = e.sun_world(0.0)
    return Rig(name='fieldstudio_default', quality='sculpt',
               lights=[Light(kind='sun', name='hdri sun', direction=[float(x) for x in d],
                             rgb=[float(x) for x in e.sun_rgb], strength=1.0, spd='D65',
                             note='the noon_grass HDRI sun split out (env.json)'),
                       Light(kind='env', name='noon_grass sky', note='the HDRI as sky light (SH irradiance x sky rays)')],
               environment=Environment(hdri=hdri_dir, rotation_deg=0.0), exposure=Exposure(rule='card'),
               shading=Shading(), note="gpu/scene.py light_params defaults, 11-11")


def fieldstudio_lp(rig: Rig, cam_pos, env: Env) -> np.ndarray:
    """the 24-float L array of gpu/scene.py GpuScene.light_params, from a rig (same order, same arithmetic)"""
    sh = rig.shading
    rot = rig.environment.rotation_deg
    L_, cr, sr = env.sun_world(rot)
    sun = rig.sun()
    if sun is not None and sun.direction is not None:
        L_ = np.asarray(sun.direction, float) / np.linalg.norm(sun.direction)
    srgb = env.sun_rgb if sun is None else np.asarray(sun.rgb, float) * float(sun.strength)
    skyE = float(np.mean(env.envj['sky_E_horizontal_rgb']))
    g = float(getattr(rig.environment, 'sky_gain', 1.0))
    if g != 1.0:                                 # the card meter reads the sky the rig lets in (1: as before, bit for bit)
        skyE *= g
    E = float(np.mean(srgb)) + skyE
    expo = rig.exposure.value if rig.exposure.rule == 'fixed' else math.pi / max(E, 1e-6)
    expo *= 2.0 ** float(rig.exposure.ev)
    rad = sun.angular_radius_deg if sun is not None else SUN_RADIUS_DEG
    Lp = np.zeros(24, np.float32)
    Lp[0:3] = L_
    Lp[3:6] = srgb
    Lp[6], Lp[7] = cr, sr
    Lp[8] = expo
    Lp[9] = rig.environment.background_gain
    Lp[10] = sh.wrap
    Lp[11] = sh.sun_reach_m
    Lp[12] = 1.0 / math.tan(math.radians(max(rad, 1e-3)))
    Lp[13] = sh.sky_reach_m
    Lp[14] = sh.sky_rays
    Lp[15] = sh.spec_level
    Lp[16] = sh.spec_exponent
    Lp[17:20] = cam_pos
    Lp[20] = sh.sky_step_cap
    Lp[21] = sh.sun_step_cap
    Lp[22] = float(sh.headlight) * float(np.mean(srgb))
    return Lp


def light_params_host(light, cam_pos, env: Env) -> np.ndarray:
    """the L array GpuScene.light_params gives for a card view's light dict (gv.light), on the host (no card): with a
    'rig', fieldstudio_lp; without, the view's own light: its 'sun_dir' (the sculpting key), 'headlight', 'ev' with the
    HDRI's sun and sky (light_params' defaults = fieldstudio_default: rig_test G5). The environment's rotation is the
    rig's (0 without one)."""
    light = dict(light or {})
    rig = light.get('rig')
    if rig is None:
        rig = fieldstudio_default(env.folder)
        if light.get('sun_dir') is not None:
            d = np.asarray(light['sun_dir'], float)
            rig.sun().direction = [float(x) for x in d / np.linalg.norm(d)]
        rig.exposure.ev = float(light.get('ev', 0.0))
        rig.shading.headlight = float(light.get('headlight', 0.0))
        if light.get('sky_n') is not None:
            rig.shading.sky_rays = int(light['sky_n'])
    return fieldstudio_lp(rig, cam_pos, env)


STUDIO_KINDS = ('area_rect', 'area_disk', 'sphere', 'spot', 'point')


def hair_lights(light, point, cam_pos, env: Env, max_lights=8):
    """The lights a drawer outside the tracer (fs-hair's strands) uses at `point` so it agrees with the skin, from the
    card view's light dict (gv.light: what the view renders; with a rig it is the panel's effective rig, mutes and solo
    applied). The tracer's quality rule decides which lights exist:
      - no rig / a 'sculpt' or 'game' rig: the key as the sun (a studio key's plain-light twin, or the HDRI's sun);
        studio fills and rims are not drawn at this quality, so they are not returned;
      - a 'lookdev' / 'final' rig: the studio lights themselves (area / disk / sphere / spot / point; the key sampled,
        its sun twin dropped), each as a direction from `point` and its irradiance there: the rig's calibration (rgb x
        strength, the sun twin's units, on axis at the rig's subject) with distance (d_subject / d_point)^2, the
        emitter's own facing and a spot's cone; an HDRI sun that is not a twin stays.
    Plus the sculpting headlight (toward the camera) when the view has one.
    -> {'lights': [{'name', 'kind', 'dir' (unit, toward the light), 'rgb' (irradiance, renderer units before exposure),
        'key' (the one to shadow through the hair mass), 'shadow'}], 'exposure' (the frame's k), 'sky_sh' (27: the
        tracer's SH sky irradiance), 'sky_mean_radiance' (rgb), 'quality'}"""
    light = dict(light or {})
    rig = light.get('rig')
    Lp = light_params_host(light, cam_pos, env)
    p = np.asarray(point, float)
    look = rig is not None and getattr(rig, 'quality', 'sculpt') in ('lookdev', 'final')
    s = rig.sun() if rig is not None else None
    twin = s is not None and (s.name or '').startswith('key as sun')
    studio = [l for l in (rig.lights if rig is not None else []) if l.kind in STUDIO_KINDS and l.reaches and l.position is not None]
    out = []
    srgb = np.asarray(Lp[3:6], float)
    if not (look and twin and studio) and srgb.max() > 0:
        out.append({'name': (s.name if s is not None else 'hdri sun'), 'kind': 'sun', 'dir': np.asarray(Lp[0:3], float),
                    'rgb': srgb, 'key': True, 'shadow': True})
    if look and studio:
        scale = 1.0          # rgb x strength: the units of the key's sun twin (rt2 R6, 11-11 ~22:00)
        subj = np.asarray(rig.subject, float) if getattr(rig, 'subject', None) is not None else None
        first = True
        for l in studio:
            pos = np.asarray(l.position, float)
            v = pos - p
            d = float(np.linalg.norm(v)) or 1e-6
            w = v / d
            E = np.asarray(l.rgb, float) * float(l.strength) * scale
            if subj is not None:
                E = E * (float(np.linalg.norm(pos - subj)) / d) ** 2
            ax = -np.asarray(l.direction if l.direction is not None else (0.0, -1.0, 0.0), float)
            ax = ax / np.linalg.norm(ax)                     # the emitter's axis (toward the subject), as set_rig
            ca = -float(ax @ w)
            if l.kind in ('area_rect', 'area_disk'):
                E = E * max(ca, 0.0)                         # its face turned from the point
            elif l.kind == 'spot':
                cone = l.spot or [30.0, 0.3]
                c0, c1 = math.cos(math.radians(cone[0])), math.cos(math.radians(cone[0] * (1 - cone[1])))
                E = E * (0.0 if ca <= c0 else min(1.0, (ca - c0) / max(c1 - c0, 1e-4)))
            if E.max() <= 0:
                continue
            out.append({'name': l.name, 'kind': l.kind, 'dir': w, 'rgb': E, 'key': first, 'shadow': first})
            first = False
    if Lp[22] > 0:
        hv = np.asarray(cam_pos, float) - p
        out.append({'name': 'headlight', 'kind': 'headlight', 'dir': hv / (np.linalg.norm(hv) or 1e-6),
                    'rgb': np.full(3, float(Lp[22])), 'key': False, 'shadow': False})
    if not any(o['key'] for o in out) and out:
        out[0]['key'] = out[0]['shadow'] = True
    rot = float(rig.environment.rotation_deg) if rig is not None else 0.0
    g = float(getattr(rig.environment, 'sky_gain', 1.0)) if rig is not None else 1.0
    return {'lights': out[:max_lights], 'exposure': float(Lp[8]), 'sky_sh': env.sky_sh(rot) * np.float32(g),
            'sky_mean_radiance': np.asarray(env.envj.get('env_mean', [0.0, 0.0, 0.0]), float) * g,
            'quality': getattr(rig, 'quality', 'sculpt') if rig is not None else 'view'}


# ---------------- colour temperature (physics: Planck's law through an analytic CIE 1931 fit) ----------------
def _cie_xyz_bar(lam_nm):
    """CIE 1931 2-deg colour matching functions, the multi-lobe Gaussian fit of Wyman, Sloan, Shirley (JCGT 2013)"""
    l = np.asarray(lam_nm, float)
    def g(x, mu, s1, s2):
        return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)
    xb = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2)
    yb = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1)
    zb = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8)
    return xb, yb, zb


def kelvin_rgb(K):
    """linear-sRGB colour of a blackbody at K kelvin relative to a 6504 K blackbody (which reads (1, 1, 1)): the
    convention for light colours here (the fit's own error cancels at the reference)"""
    a = _kelvin_rgb_raw(K); w = _kelvin_rgb_raw(6504.0)
    return [float(x / max(y, 1e-12)) for x, y in zip(a, w)]


def _kelvin_rgb_raw(K):
    lam = np.arange(380.0, 781.0, 5.0)
    h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lm = lam * 1e-9
    B = 1.0 / (lm ** 5 * (np.exp(h * c / (lm * kB * float(K))) - 1.0))
    xb, yb, zb = _cie_xyz_bar(lam)
    X, Y, Z = (B * xb).sum(), (B * yb).sum(), (B * zb).sum()
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = M @ np.array([X, Y, Z]) / max(Y, 1e-30)
    w = M @ np.array([0.95047, 1.0, 1.08883])                   # D65 white -> divide so 6504 K reads near neutral
    return [float(x) for x in np.clip(rgb / w, 0.0, None)]


# ---------------- studio rigs (photographic practice; sizes and distances of real modifiers) ----------------
# (kind, name, azimuth deg from her front, + toward HER right; elevation deg; distance m; size m; strength; kelvin; note)
PRESETS = {
    'rembrandt': [('area_rect', 'key softbox 60x90', 45, 40, 1.2, (0.6, 0.9), 1.0, 5600, 'the cheek triangle on her shadow side'),
                  ('area_rect', 'fill reflector', -40, 10, 1.0, (0.8, 0.8), 0.25, 5600, 'opens the shadow side a little')],
    'rembrandt_classic': [('area_rect', 'key softbox 9x14', 65, 30, 2.0, (0.093, 0.1397), 1.0, 5600,
                           'the key 65 deg to her side, 30 up, 4 deg across from her (11-12, rb7: of 96 settings the nearest '
                           'to a Rembrandt-lit photograph on five traits: its three brightness ratios, the triangle against '
                           'the region beside the nose, and the nose shadow edge 4.0 mm against its 3.9; a dark studio with '
                           'a little room light, PRESET_ENV)'),
                          ('area_rect', 'fill reflector', -40, 10, 1.2, (0.8, 0.8), 0.05, 5600, '')],
    'rembrandt_soft': [('area_rect', 'key softbox 25x35', 65, 30, 2.0, (0.233, 0.35), 1.0, 5600,
                        'the classic preset until 11-12 (10 deg across from her): the same pattern with a softer nose '
                        'shadow edge (rb7 9.4 mm; cmp2 key alone 6.8 mm)'),
                       ('area_rect', 'fill reflector', -40, 10, 1.2, (0.8, 0.8), 0.1, 5600, '')],
    'loop': [('area_rect', 'key softbox 60x90', 30, 30, 1.2, (0.6, 0.9), 1.0, 5600, 'the nose shadow loops toward the cheek'),
             ('area_rect', 'fill', -30, 5, 1.4, (0.9, 0.9), 0.3, 5600, '')],
    'butterfly': [('area_disk', 'beauty dish 55', 0, 45, 1.0, (0.55, 0.55), 1.0, 5600, 'above the camera: the butterfly under the nose')],
    'clamshell': [('area_disk', 'beauty dish 55', 0, 35, 1.0, (0.55, 0.55), 1.0, 5600, ''),
                  ('area_rect', 'fill strip below', 0, -20, 0.8, (0.3, 1.2), 0.5, 5600, 'under the chin: shadows opened')],
    'split': [('area_rect', 'key strip 30x120', 90, 10, 1.0, (0.3, 1.2), 1.0, 5600, 'half the face lit')],
    'three_point': [('area_rect', 'key softbox 60x90', 35, 30, 1.2, (0.6, 0.9), 1.0, 5600, ''),
                    ('area_rect', 'fill octa 120', -45, 10, 1.8, (1.2, 1.2), 0.35, 5600, ''),
                    ('area_rect', 'rim strip 30x120', 160, 40, 1.5, (0.3, 1.2), 0.8, 5600, 'separates her from the background')],
    'rim_hair': [('area_rect', 'rim strip left', 135, 20, 1.5, (0.3, 1.2), 0.7, 5600, ''),
                 ('area_rect', 'rim strip right', -135, 20, 1.5, (0.3, 1.2), 0.7, 5600, ''),
                 ('spot', 'hair light', 180, 70, 1.5, (0.2, 0.2), 0.5, 5600, 'cone 30 deg'),
                 ('area_rect', 'soft front fill', 0, 5, 2.0, (1.2, 1.2), 0.2, 5600, '')],
    'high_key': [('area_disk', 'octa 150 front', 0, 20, 1.6, (1.5, 1.5), 1.0, 5600, ''),
                 ('area_rect', 'fill left', 60, 10, 1.6, (1.0, 1.4), 0.6, 5600, ''),
                 ('area_rect', 'fill right', -60, 10, 1.6, (1.0, 1.4), 0.6, 5600, '')],
    'low_key': [('area_rect', 'key softbox with grid', 70, 30, 1.2, (0.6, 0.9), 1.0, 5600, 'narrow spread, no fill')],
}


def _place(subject, front, up, az, el, dist):
    f = np.asarray(front, float); f = f / np.linalg.norm(f)
    u = np.asarray(up, float); u = u - f * float(f @ u); u = u / np.linalg.norm(u)
    right_of_her = np.cross(f, u)                       # she faces f: her right = front x up (M8: screen left from the front)
    a, e = math.radians(az), math.radians(el)
    d = f * math.cos(a) * math.cos(e) + right_of_her * math.sin(a) * math.cos(e) + u * math.sin(e)
    d = d / np.linalg.norm(d)
    return np.asarray(subject, float) + d * dist, d


# a studio preset's own studio (11-12): the Rembrandt presets were measured in a dark studio (no backdrop, the room light
# as given); without it the default outdoor sky fills the shadow side and the pattern is gone. Other presets keep the
# environment they are applied over.
PRESET_ENV = {'rembrandt_classic': {'sky_gain': 0.05, 'background_gain': 0.0},
              'rembrandt_soft': {'sky_gain': 0.0, 'background_gain': 0.0}}


def studio_preset(name, subject, front, up=(0.0, 1.0, 0.0), base: Optional[Rig] = None, key_strength=None) -> Rig:
    """a studio rig around `subject` (world point, e.g. her face centre) facing `front` (world). The KEY (first light)
    is also the rig's sun direction and strength, so today's renderers (one key + sky) show it; the other lights are
    carried in the rig for the cache light pass's area / spot lights (fs-light L2/L5)."""
    spec = PRESETS[name]
    r = Rig(name='studio_' + name, quality='lookdev', note='studio preset %s around %s' % (name, list(np.round(subject, 3))),
            subject=[float(x) for x in subject])
    if base is not None:
        r.environment = Environment(**asdict(base.environment)); r.exposure = Exposure(**asdict(base.exposure))
        r.shading = Shading(**asdict(base.shading))
    for k, v in PRESET_ENV.get(name, {}).items():
        setattr(r.environment, k, v)
    ks = key_strength if key_strength is not None else 1.0
    lights = []
    for i, (kind, nm, az, el, dist, size, strength, K, note) in enumerate(spec):
        pos, d = _place(subject, front, up, az, el, dist)
        lights.append(Light(kind=kind, name=nm, position=[float(x) for x in pos], direction=[float(x) for x in d],
                            rgb=kelvin_rgb(K), strength=float(strength) * ks, size=list(size),
                            spot=[30.0, 0.3] if kind == 'spot' else None, group='studio', note=note))
    key = lights[0]
    sun = Light(kind='sun', name='key as sun (today\'s renderers)', direction=list(key.direction), rgb=list(key.rgb),
                strength=key.strength, angular_radius_deg=min(1.0, math.degrees(math.atan2(0.5 * max(key.size), max(1e-3, spec[0][4])))),
                note=('the key seen from her, as a sun; its angular size capped at 1 deg: the SDF penumbra estimate needs '
                      'distances inside the stored band (gpu/scene.py r8), so a softbox\'s true softness waits for area-light '
                      'sampling (L5)'))
    r.lights = [sun] + lights
    return r


def from_review(name, entry, spd_name=None) -> Rig:
    """a rig holding one of the review renderer's named lights (fvsk_spec_r5.lights: dir in the store frame, z up)"""
    d_world = R_FV @ np.asarray(entry['dir'], float)
    return Rig(name='review_' + name, quality='final',
               lights=[Light(kind='sun', name=name, direction=[float(x) for x in d_world], rgb=[float(x) for x in entry['sun_rgb']],
                             strength=float(entry['S']), spd=spd_name or name, note=entry.get('_', ''))],
               environment=Environment(sky_rgb=[float(x) for x in entry['sky_rgb']]))


def review_light(rig: Rig) -> dict:
    """the review renderer's light entry (dir in the store frame) from a rig"""
    sun = rig.sun()
    return {'dir': R_FV.T @ np.asarray(sun.direction, float), 'S': sun.strength, 'sun_rgb': list(sun.rgb),
            'sky_rgb': list(rig.environment.sky_rgb or [0.0, 0.0, 0.0]), 'spd_name': sun.spd}
