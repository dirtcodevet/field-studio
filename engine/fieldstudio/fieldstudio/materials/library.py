"""fs-materials (2026-11-14): the library as Field Studio's Materials & Generators panel reads it.

The profiles are engine/materials/profiles/*.json (format engine/materials/SCHEMA.md: every number {v, q, src} with its
source). This module turns one into what the panel shows and what an object carries:
  families()                  the panel's chips -> [material ids]
  card(mid)                   the material's card: name, plain words ("heavy . grippy . brittle"), finishes, the
                              physics numbers with their quality (M measured / D derived / E estimated) and sources
  physics(mid, volume_m3)     density, mass, friction (dry / wet), porosity, absorption, strength, toughness, hardness
  resolve_look(mid, finish, seed, rough)   the look an object carries (albedo drawn for this seed inside the stone's
                              measured spread; grain, mottle, bedding; GGX alpha by finish; F0 from n)
  look_ready(mid)             True when the profile has a look block (colour as numbers)
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'materials', 'profiles'))

# the panel's families (INTERFACE_DESIGN 5.1) -> the profiles' family names
CHIPS = [('Stone', ('stone',)), ('Brick & clay', ('ceramic',)), ('Wood', ('wood',)), ('Metal', ('metal',)),
         ('Glass', ('glass',)), ('Ground', ('ground',)), ('Water', ('fluid',)), ('Fabric', ('fabric',)),
         ('Polymer', ('polymer',))]
FINISH_WORDS = {'droved': 'Droved (tooled)', 'sawn': 'Sawn', 'rubbed': 'Rubbed', 'honed': 'Honed', 'polished': 'Polished', 'flamed': 'Flamed',
                'riven_or_tooled': 'Riven / tooled', 'split_face': 'Split face', 'riven': 'Riven',
                'pressed': 'Pressed', 'wirecut': 'Wire-cut', 'sandfaced': 'Sand-faced'}
_CACHE = {}


def _v(node, default=None):
    if isinstance(node, dict) and 'v' in node:
        return node['v']
    return default if node is None else node


def get(mid):
    p = os.path.join(PROF, mid + '.json')
    st = os.stat(p).st_mtime
    hit = _CACHE.get(mid)
    if hit and hit[0] == st:
        return hit[1]
    P = json.load(open(p, encoding='utf-8'))
    _CACHE[mid] = (st, P)
    return P


def ids():
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PROF, '*.json')))


def value(P, path, default=None):
    node = P
    for k in path.split('.'):
        if not isinstance(node, dict):
            return default
        node = node.get(k)
        if node is None:
            return default
    return _v(node, default)


def quality(P, path):
    node = P
    for k in path.split('.'):
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node.get('q') if isinstance(node, dict) else None


def families():
    out = []
    all_ids = ids()
    for name, fams in CHIPS:
        ms = []
        for mid in all_ids:
            try:
                P = get(mid)
            except Exception:
                continue
            if P.get('family') in fams and P.get('kind') not in ('light_source', 'constants'):
                ms.append(mid)
        out.append((name, ms))
    return out


def look_ready(mid):
    try:
        return isinstance(get(mid).get('look'), dict)
    except Exception:
        return False


def display_name(mid):
    base = mid.split('.')[-1].replace('_', ' ')
    return base[:1].upper() + base[1:]


def _mid(v):
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return 0.5 * (v[0] + v[1])
    if isinstance(v, dict):
        if 'typical' in v:
            return v['typical']
        if 'dry' in v:
            return _mid(v['dry'])
    return v


def physics(mid, volume_m3=None):
    """the numbers that travel with the material (SI), each with its quality"""
    P = get(mid)
    rho = _v(P.get('deform', {}).get('density_kg_m3'))
    fr = P.get('contact', {}).get('friction', {})
    d = {
        'density_kg_m3': rho, 'density_q': quality(P, 'deform.density_kg_m3'),
        'friction_static': _v(fr.get('static')), 'friction_kinetic': _v(fr.get('kinetic')),
        'friction_q': (fr.get('static') or {}).get('q'), 'friction_counterface': fr.get('counterface'),
        'porosity': value(P, 'contact.porosity'), 'water_absorption_pct': value(P, 'contact.water_absorption'),
        'restitution': value(P, 'contact.restitution'),
        'model': P.get('deform', {}).get('model'),
        'E_GPa': value(P, 'deform.E_GPa'), 'poisson': value(P, 'deform.poisson'),
        'compressive_MPa': value(P, 'deform.fracture.strength_compressive_MPa'),
        'tensile_MPa': value(P, 'deform.fracture.strength_tensile_MPa'),
        'toughness_MPa_m05': value(P, 'deform.fracture.toughness_KIC_MPa_m05'),
        'hardness': value(P, 'contact.hardness'),
        'conductivity_W_mK': value(P, 'thermal.conductivity_W_mK'),
        'feels_C_at_20': value(P, 'thermal.contact_temp_with_skin_C'),
    }
    if volume_m3 is not None and rho is not None:
        d['volume_L'] = volume_m3 * 1e3
        d['mass_kg'] = volume_m3 * float(_mid(rho))
    return d


def words(mid):
    """plain words for the card: heavy / light, grippy / slippery, brittle / tough / soft, porous"""
    ph = physics(mid)
    w = []
    rho = _mid(ph['density_kg_m3'])
    if isinstance(rho, (int, float)):
        w.append('heavy' if rho >= 1500 else 'light' if rho < 800 else 'mid-weight')   # in the hand (water 1000)
    mu = _mid(ph['friction_static'])
    if isinstance(mu, (int, float)):
        w.append('grippy' if mu >= 0.55 else 'slippery' if mu < 0.3 else 'medium grip')
    m = ph.get('model')
    if m == 'brittle':
        w.append('brittle')
    elif m in ('elastoplastic', 'rigid') and get(mid).get('family') == 'metal':
        w.append('tough')
    por = _mid(ph['porosity'])
    if isinstance(por, (int, float)) and por >= 0.05:
        w.append('porous')
    return ' · '.join(w)


def finishes(mid):
    P = get(mid)
    lk = P.get('look', {})
    fin = _v(lk.get('finishes')) if lk else None
    if isinstance(fin, dict):
        return list(fin.keys())
    a = value(P, 'surface.alpha_by_finish')
    return list(a.keys()) if isinstance(a, dict) else []


def card(mid):
    P = get(mid)
    return {'id': mid, 'name': display_name(mid), 'family': P.get('family'), 'words': words(mid),
            'about': P.get('_', ''), 'finishes': finishes(mid), 'look_ready': look_ready(mid),
            'physics': physics(mid), 'sources': {k: {'cite': s.get('cite'), 'lic': s.get('lic')}
                                                 for k, s in P.get('sources', {}).items()}}


# ---- the look an object carries -------------------------------------------------------------------------------------
def varieties(mid):
    """the measured colour varieties of a material (look.varieties) -> {name: {albedo_lin, sample}}"""
    v = _v((get(mid).get('look') or {}).get('varieties'))
    return v if isinstance(v, dict) else {}


def _stains(st, rng):
    """the iron staining for one object: patches and Liesegang bands, strengths and coverage drawn for its seed (some
    blocks of a bed are plain, some banded)"""
    if not st:
        return {}
    j = lambda x, r=0.25: float(x) * float(np.exp(rng.normal(0.0, r)))
    return {'dir': list(st.get('dir', [0.0, 0.0, 0.0])),
            'patch_m': j(st.get('patch_mm', 0.0) * 1e-3, 0.2), 'patch_log': j(st.get('patch_log', 0.0), 0.35),
            'patch_th': float(st.get('patch_th', 0.5)) + float(rng.normal(0.0, 0.3)),
            'band_m': j(st.get('band_mm', 0.0) * 1e-3, 0.2), 'band_log': j(st.get('band_log', 0.0), 0.35),
            'band_w': float(st.get('band_w', 0.3)), 'ring_zone': j(st.get('ring_zone', 0.0), 0.4)}


def _skin(sk, rng):
    """a fired body's skin for one object: its depth and the body's colour under it (the form is added by the
    command that knows the object: commands.with_skin)"""
    if not sk:
        return {}
    return {'depth_m': float(sk.get('depth_mm', 1.5)) * 1e-3 * float(np.exp(rng.normal(0.0, 0.25))),
            'width_m': float(sk.get('width_mm', 0.6)) * 1e-3,
            'body_log': [float(x) * float(np.exp(rng.normal(0.0, 0.2))) for x in sk.get('body_log', (0.0, 0.0, 0.0))]}


def _rings(s):
    """rings per unit of the patch field for the band spacing: the field's in-plane gradient has a median of 7.1 / its
    wavelength (measured 11-14 on 6000 points), so the median spacing is band_m"""
    if s.get('band_m', 0.0) > 0.0 and s.get('patch_m', 0.0) > 0.0:
        s['ring_k'] = s['patch_m'] / (7.1 * s['band_m'])
    return s


def _weather(wt, w, rng, shift=None):
    """weathering w (0 as made .. 1 an old exposed piece): the face colour's log shift (measured where the profile has a
    weathered / bare pair) x w, uneven over the object (patches); the arrises' wear radius x w (this seed's own +-20 %);
    the grains' relief gain. {} when w is 0 or the profile has no weathering block."""
    if not wt or w <= 0.0:
        return {}
    w = min(max(float(w), 0.0), 1.0)
    return {'w': w, 'log_shift': [float(x) * w for x in (shift or (0.0, 0.0, 0.0))],
            'patch_m': float(wt.get('patch_mm', 0.0)) * 1e-3, 'patch_frac': float(wt.get('patch_frac', 0.0)),
            'wear_m': float(wt.get('wear_mm', 0.0)) * 1e-3 * w * float(np.exp(rng.normal(0.0, 0.2))),
            'wear_lam_m': float(wt.get('wear_lam_mm', 80.0)) * 1e-3, 'wear_var': float(wt.get('wear_var', 0.35)),
            'micro_gain': float(wt.get('micro_gain', 0.0)) * w}


# 11-14 (update 14): the deposit kinds (a material on the surface; ESTIMATES until measured spectra: a runner ask):
# dust = a light yellowish-brown soil fines film, lime = mortar residue / efflorescence (calcite / salts, near white),
# soot = an urban black crust (carbon in gypsum)
DEPOSITS = {'dust': [0.33, 0.27, 0.19], 'lime': [0.78, 0.77, 0.73], 'soot': [0.035, 0.033, 0.030]}


# 11-14 (update 24): how the block was cut from its bed, as masons say it, in the block's own
# frame (y up, z its front face, x along it): NATURAL BED (beds horizontal as quarried: walling's default),
# FACE-BEDDED (beds parallel to the front face), EDGE-BEDDED (beds vertical, across the face)
BEDDING = {'natural': (0.0, 1.0, 0.0), 'face': (0.0, 0.0, 1.0), 'edge': (1.0, 0.0, 0.0)}


def resolve_look(mid, finish=None, seed=1, rough=1.0, variety=None, bed_normal=(0.0, 1.0, 0.0), weather=0.0,
                 deposit=0.0, deposit_kind='dust', exposure=0.0, bedding=None):
    """the look for one object: the variety's MEASURED mean albedo moved by this seed within the copy-to-copy spread
    (lightness, and along the iron axis), its grains / interstices / laminae / spots / mottle with seeded strengths
    inside their ranges, and the finish's GGX alpha (rougher -> alpha up). Deterministic in its arguments."""
    P = get(mid)
    L = P.get('look')
    if not isinstance(L, dict):
        return None
    if bedding in BEDDING:
        bed_normal = BEDDING[bedding]
    fins = _v(L.get('finishes')) or {}
    finish = finish if finish in fins else (next(iter(fins)) if fins else None)
    fv = fins.get(finish, {}) if finish else {}
    vars_ = _v(L.get('varieties')) or {}
    variety = variety if variety in vars_ else _v(L.get('default_variety'), next(iter(vars_), None))
    if variety in vars_:
        base = np.asarray(vars_[variety]['albedo_lin'], float)
    else:
        base = np.asarray(_v(L['albedo_lin']), float)
    rng = np.random.default_rng([int(seed), 404])
    iron = np.asarray(_v(L.get('iron_dir'), [0.0, 0.0, 0.0]), float)
    if variety in vars_ and vars_[variety].get('iron_dir'):          # a variety's own colouring mineral (red: hematite)
        iron = np.asarray(vars_[variety]['iron_dir'], float)
    alb = base * np.exp(rng.normal(0.0, float(_v(L.get('copy_log_sd'), 0.0))) +
                        rng.normal(0.0, float(_v(L.get('copy_iron_sd'), 0.0))) * iron)
    g = _v(L.get('grain')) or {}
    m = _v(L.get('mottle')) or {}
    b = _v(L.get('laminae')) or {}
    sp = _v(L.get('spots')) or {}
    jit = lambda x, r=0.12: float(x) * float(np.exp(rng.normal(0.0, r)))
    alpha = float(fv.get('alpha', value(P, 'surface.alpha', 0.5)))
    # the bed: near planar, dipping a few degrees (natural bed; a face parallel to it shows few broad bands)
    dip = np.radians(float(rng.uniform(*b.get('dip_deg', (0.0, 0.0)))))
    az = float(rng.uniform(0.0, 2.0 * np.pi))
    bn = np.asarray(bed_normal, float)
    bn = bn / max(np.linalg.norm(bn), 1e-12)
    tx = np.cross(bn, [1.0, 0.0, 0.0] if abs(bn[0]) < 0.9 else [0.0, 0.0, 1.0])
    tx /= np.linalg.norm(tx)
    ty = np.cross(bn, tx)
    bed_n = np.cos(dip) * bn + np.sin(dip) * (np.cos(az) * tx + np.sin(az) * ty)
    alpha = min(0.98, alpha * (1.0 + 0.35 * (rough - 1.0)))
    wt = _weather(_v(L.get('weather')) or {}, weather, np.random.default_rng([int(seed), 505]),
                  _v(L.get('weather_colour')))
    fr = _v(L.get('fracture')) or {}
    out = {
        'material': mid, 'finish': finish, 'variety': variety, 'seed': int(seed), 'rough': float(rough),
        'albedo': [round(float(x), 5) for x in np.clip(alb, 0.01, 0.95)],
        'alpha': round(alpha, 4), 'F0': float(value(P, 'surface.F0', 0.04)),
        'on_sigma': float(fv.get('on_sigma', 0.3)) * (1.0 + 0.3 * (rough - 1.0)),
        'iron_dir': [float(x) for x in iron],
        'grain': {'size_m': jit(g.get('size_mm', 0.0) * 1e-3, 0.08), 'log_sd': jit(g.get('log_sd', 0.0)),
                  'dir': list(g.get('dir', [1.0, 1.0, 1.0])),
                  'fine_m': float(g.get('fine_mm', 0.0)) * 1e-3, 'fine_log_sd': jit(g.get('fine_log_sd', 0.0)),
                  'interstice_log': float(g.get('interstice_log', 0.0)),
                  'interstice_r0': float(g.get('interstice_r0', 0.6)),
                  'hue_log_sd': float(g.get('hue_log_sd', 0.0)),
                  'clear_p': float(g.get('clear_p', 0.0)),
                  'hue_dir': list(g.get('hue_dir', [0.0, 0.0, 0.0])),
                  'cluster_m': float(g.get('cluster_mm', 0.0)) * 1e-3,
                  'cluster_log_sd': jit(g.get('cluster_log_sd', 0.0)),
                  'dark_p': jit(g.get('dark_p', 0.0), 0.25), 'dark_log': float(g.get('dark_log', 0.0))},
        'mottle': {'lam_m': jit(m.get('lam_mm', 0.0) * 1e-3, 0.2), 'log_sd': jit(m.get('log_sd', 0.0), 0.25)},
        'laminae': {'spacing_m': jit(b.get('spacing_mm', 0.0) * 1e-3, 0.2),
                    'iron_log_sd': jit(b.get('iron_log_sd', 0.0), 0.3),
                    'grey_log_sd': jit(b.get('grey_log_sd', 0.0), 0.3),
                    'normal': [float(x) for x in bed_n], 'warp_m': float(b.get('warp_mm', 0.0)) * 1e-3},
        'spots': {'cell_m': float(sp.get('cell_mm', 0.0)) * 1e-3, 'share': jit(sp.get('share', 0.0), 0.35),
                  'radius_cells': float(sp.get('radius_cells', 0.0)), 'log': float(sp.get('log', 0.0))},
        'micro': {'amp_m': float(_v(L.get('micro'), {}).get('amp_um', 0.0)) * 1e-6 * (1.0 + 0.5 * (rough - 1.0)),
                  'radius_cells': float(_v(L.get('micro'), {}).get('radius_cells', 0.62)),
                  'clip': float(fv.get('grain_clip', 1.0))},
        'stains': _rings(_stains(_v(L.get('stains')) or {}, rng)),
        'skin': _skin(_v(L.get('skin')) or {}, rng),
        'relief': fv.get('relief', finish), 'bedding': bedding if bedding in BEDDING else 'natural',
    }
    if fr.get('amp_um', 0.0) > 0.0:                 # broken / worn surfaces granular (look.cuh mat_micro)
        out['fracture'] = {'amp_m': float(fr['amp_um']) * 1e-6,
                           'cell_m': float(fr.get('cell_mm', 0.6)) * 1e-3}
    cr = _v(L.get('crystals')) or {}
    if cr.get('cell_mm', 0.0) > 0.0:                # granite: crystals by mineral (look.cuh lk_crystal)
        out['crystals'] = {'cell_m': jit(cr['cell_mm'] * 1e-3, 0.08), 'spread': float(cr.get('spread', 0.0)),
                           'warp': float(cr.get('warp', 0.0)), 'flake_m': float(cr.get('flake_mm', 0.0)) * 1e-3,
                           'flake_share': float(cr.get('flake_share', 0.0)),
                           'clot_m': float(cr.get('clot_mm', 6.0)) * 1e-3, 'band': float(cr.get('boundary_band', 0.0)),
                           'flake_aspect': float(cr.get('flake_aspect', 1.0)), 'shares': list(cr['shares']),
                           'log': {k: list(v) for k, v in cr['log'].items()},
                           'crystal_log_sd': float(cr.get('crystal_log_sd', 0.0)),
                           'quartz_alpha': float(cr.get('quartz_alpha', 1.0)),
                           'biotite_alpha': float(cr.get('biotite_alpha', 1.0)),
                           'optics': dict(cr.get('optics') or {}), 'jag': float(cr.get('jag', 0.0)),
                           'fray': float(cr.get('fray', 0.0))}
    if deposit and deposit > 0.0:                   # the deposit state: amount 0..1 of a kind (look.cuh: coverage rules)
        dr = np.random.default_rng([int(seed), 606])
        out['deposit'] = {'amount': min(max(float(deposit), 0.0), 1.0), 'kind': deposit_kind,
                          'albedo': list(DEPOSITS.get(deposit_kind, DEPOSITS['dust'])),
                          'up_bias': 0.6, 'pore_fill': 1.0, 'patch_m': 0.06 * float(np.exp(dr.normal(0.0, 0.2))),
                          'patch_frac': 0.3, 'clear_m': 2.5e-3, 'exposure': min(max(float(exposure), 0.0), 1.0)}
    if wt:
        out['weather'] = wt
        out['micro']['amp_m'] *= 1.0 + wt['micro_gain']
    return out
