"""The neutral studio: Field Studio's default environment (fs-shell, 2026-11-13; Tier 0 part 3).

Decided (11-13): a grass and trees scene as the default environment distracts and is not needed there; it stays
choosable (the Light panel), and the default is this neutral studio.

What it is: an environment map made here, from a formula (no image, no licence: P13), in the same form as an HDRI
folder (env_2k.npy: 1024 x 2048 x 3 fp16 radiance, Blender equirect, z up; env.json: the key's direction and
irradiance, the sky's horizontal irradiance), so every reader of an HDRI folder takes it unchanged (gpu/scene.py,
engine/light/rig.py, the hair's light). Neutral grey everywhere (equal R, G, B): a soft sky brighter overhead, a darker
floor below the horizon, no horizon line (a smooth blend over +-12 deg), no sun disk in the map (the key is the
analytic light: in the window's 'key' mode it follows the view, as before). The map's values are set so the
background reads as a mid-dark plain grey behind a form under the card's exposure rule (k = pi / E).

Other maps stay available: the Light panel's environment list keeps every HDRI folder (noon_grass among them).
The folder is generated once, at first use, into ~/FieldStudio/env/studio_neutral (VERSION below; a new version
regenerates it); nothing binary is kept in the repository."""
import json
import math
import os

import numpy as np

VERSION = 1
NAME = 'studio_neutral'
ROOT = os.path.join(os.path.expanduser('~'), 'FieldStudio', 'env')

# radiance (linear, equal RGB): overhead, at the horizon from above, at the horizon from below, straight down
SKY_ZENITH = 0.50
SKY_HORIZON = 0.36
FLOOR_HORIZON = 0.24
FLOOR_NADIR = 0.18
BLEND_DEG = 12.0          # the horizon blend's half width
KEY_IRRADIANCE = 2.6      # the key's irradiance on a surface facing it (the 'sun' of an HDRI folder; neutral)
KEY_ELEV_DEG, KEY_AZ_DEG = 50.0, 135.0     # the key's direction in the map's frame (used only in 'hdri' light mode)


def radiance(elev):
    """elev (radians, -pi/2 .. pi/2) -> radiance (one channel)"""
    e = np.asarray(elev, np.float64)
    s = np.sin(np.clip(e, 0.0, None))
    sky = SKY_HORIZON + (SKY_ZENITH - SKY_HORIZON) * s ** 0.8
    d = np.sin(np.clip(-e, 0.0, None))
    floor = FLOOR_HORIZON + (FLOOR_NADIR - FLOOR_HORIZON) * d
    t = np.clip((np.degrees(e) + BLEND_DEG) / (2 * BLEND_DEG), 0.0, 1.0)
    t = t * t * (3 - 2 * t)                 # smoothstep: no horizon line
    return floor + (sky - floor) * t


def build(folder):
    os.makedirs(folder, exist_ok=True)
    H, W = 1024, 2048
    v = (np.arange(H) + 0.5) / H * np.pi                # polar angle from +z (Blender up)
    elev = np.pi / 2 - v
    col = radiance(elev).astype(np.float32)               # (H,)
    env = np.repeat(np.repeat(col[:, None, None], W, 1), 3, 2)
    np.save(os.path.join(folder, 'env_2k.npy'), env.astype(np.float16))
    # the horizontal irradiance of the upper hemisphere: E = int L cos(theta) dOmega
    th = v[v < np.pi / 2]
    dO = (2 * np.pi) * (np.pi / H) * np.sin(th)
    E = float((radiance(np.pi / 2 - th) * np.cos(th) * dO).sum())
    el, az = math.radians(KEY_ELEV_DEG), math.radians(KEY_AZ_DEG)
    sun = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    j = {'source': 'Field Studio neutral studio (gpu/studio_env.py v%d): made from a formula, no image, no licence' % VERSION,
         'version': VERSION, 'frame': 'Blender equirect, z up', 'neutral': True,
         'sun_dir_hdri': [round(x, 5) for x in sun], 'sun_elevation_deg': KEY_ELEV_DEG, 'sun_azimuth_deg_hdri': KEY_AZ_DEG,
         'sun_irradiance_rgb': [KEY_IRRADIANCE] * 3, 'sky_E_horizontal_rgb': [round(E, 4)] * 3,
         'env_mean': [round(float(env[..., 0].mean()), 4)] * 3,
         'radiance': {'zenith': SKY_ZENITH, 'sky_horizon': SKY_HORIZON, 'floor_horizon': FLOOR_HORIZON,
                      'nadir': FLOOR_NADIR, 'blend_deg': BLEND_DEG}}
    with open(os.path.join(folder, 'env.json'), 'w', encoding='utf-8') as f:
        json.dump(j, f, indent=1)
    try:
        from PIL import Image
        pv = np.clip(env[::8, ::8] / max(SKY_ZENITH, 1e-6), 0, 1) ** (1 / 2.2)
        Image.fromarray((pv * 255).astype(np.uint8)).save(os.path.join(folder, 'env_preview.png'))
    except Exception:
        pass
    return folder


def ensure(root=None):
    """-> the neutral studio's folder (made now if missing or of an older version)"""
    folder = os.path.join(root or ROOT, NAME)
    try:
        with open(os.path.join(folder, 'env.json'), encoding='utf-8') as f:
            if json.load(f).get('version') == VERSION and os.path.isfile(os.path.join(folder, 'env_2k.npy')):
                return folder
    except Exception:
        pass
    return build(folder)
