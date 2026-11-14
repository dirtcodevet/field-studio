"""fs-materials (2026-11-14): the card look of library materials (the device side is materials/look.cuh, inserted into
gpu/trace.cu; light_pass calls mat_shade for every volume whose object carries a material).

Materials.before_frame(G, mirror, scene, cam, w, h) runs once a frame (gpu/viewport.py GpuView.frame, like fs-light's
light_rt hook): it writes the view's pixel angle (the colour field is filtered by the pixel's footprint) and, when any
object's material changed, the per-volume table g_mat; True asks the view for a full restart.

table_row(look) -> the 48 floats of one volume (the layout is look.cuh's comment; MAT_* below name the slots).
"""
import math

import numpy as np

MATF = 192
MAT_NV = 64
MAT_ON, MAT_RGB, MAT_ALPHA, MAT_F0, MAT_SIGMA = 0, 1, 4, 5, 6
MAT_GRAIN, MAT_GRAIN_SD, MAT_GDIR, MAT_FINE, MAT_FINE_SD, MAT_INT_K, MAT_INT_R0 = 7, 8, 9, 12, 13, 14, 15
MAT_MOT_L, MAT_MOT_SD, MAT_IRON, MAT_DARK_P, MAT_DARK_K = 16, 17, 18, 21, 22
MAT_LAM_S, MAT_LAM_SD, MAT_BED_N, MAT_LAM_WARP, MAT_SEED = 23, 24, 25, 28, 29
MAT_SPOT_CELL, MAT_SPOT_P, MAT_SPOT_R, MAT_SPOT_K, MAT_LAM_GREY, MAT_INT_PHI, MAT_INT_MEAN = 30, 31, 32, 33, 34, 35, 36
MAT_CLU_L, MAT_CLU_SD = 37, 38
MAT_MIC_A, MAT_MIC_R, MAT_MIC_CLIP, MAT_MIC_MEAN = 39, 40, 41, 42
MAT_GRAIN_HUE, MAT_GRAIN_HDIR, MAT_CLEAR = 43, 44, 47
MAT_STAIN_DIR, MAT_PATCH = 48, 51
MAT_BAND = 54

# the nearest-grain distance d1 (cells) of look.cuh's cellular grains, measured 11-14 over 3e5 random points:
# percentiles 10 / 25 / 50 / 75 / 90 / 99 % = 0.290 / 0.400 / 0.524 / 0.643 / 0.742 / 0.895
_D1_Q = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
_D1_V = np.array([0.0, 0.290, 0.400, 0.524, 0.643, 0.742, 0.895, 1.10])


def interstice_stats(r0, width=0.22, n=4001):
    """(P(d1 > r0), the mean smoothstep ramp over those points) from the measured d1 distribution"""
    u = np.linspace(0.0, 1.0, n)
    d = np.interp(u, _D1_Q, _D1_V)
    t = np.clip((d - r0) / width, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    inside = d > r0
    phi = float(inside.mean())
    return phi, float(t[inside].mean()) if inside.any() else 0.0


def micro_mean(radius, clip, n=4001):
    """the mean grain-relief height (look.cuh mat_micro, before the amplitude) over the measured d1 distribution: the
    dome t = 1 - (d1 / R)^2 inside R, clipped at clip, / clip (the per-grain height factor's mean is 1)"""
    u = np.linspace(0.0, 1.0, n)
    d = np.interp(u, _D1_Q, _D1_V)
    t = 1.0 - (d / radius) ** 2
    h = np.where(t > 0.0, np.minimum(t, clip) / clip, 0.0)
    return float(h.mean())


def fracture_mean(n=4001):
    """the fracture facets' mean height (look.cuh lk_facet_h): the level's mean 0.4, the tilt's 0, - 0.6 E[d1^2]
    over the measured d1 distribution"""
    u = np.linspace(0.0, 1.0, n)
    d = np.interp(u, _D1_Q, _D1_V)
    return float(0.4 - 0.6 * (d * d).mean())


# ---- the crystals' area shares: a numpy port of look.cuh lk_crystal (the same hashes and noise) -------------------
_M32 = np.uint64(0xFFFFFFFF)


def _h32(x):
    x = np.asarray(x).astype(np.uint64) & _M32
    x = (((x >> np.uint64(16)) ^ x) * np.uint64(0x45d9f3b)) & _M32
    x = (((x >> np.uint64(16)) ^ x) * np.uint64(0x45d9f3b)) & _M32
    return ((x >> np.uint64(16)) ^ x) & _M32


def _u32(i):
    return np.asarray(i).astype(np.int64).astype(np.uint64) & _M32


def _hc(ix, iy, iz, seed):
    return _h32(_u32(ix) ^ _h32(_u32(iy) ^ _h32(_u32(iz) ^ _h32(_u32(seed)))))


def _u01(h):
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / 16777216.0


def _grad(h, x, y, z):
    h = h & np.uint64(15)
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where((h & np.uint64(1)) == 0, u, -u) + np.where((h & np.uint64(2)) == 0, v, -v)


def _noise3(x, y, z, seed):
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    ix, iy, iz = fx.astype(np.int64), fy.astype(np.int64), fz.astype(np.int64)
    xf, yf, zf = x - fx, y - fy, z - fz
    fade = lambda t: t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    u, v, w = fade(xf), fade(yf), fade(zf)
    res = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                g = _grad(_hc(ix + dx, iy + dy, iz + dz, seed), xf - dx, yf - dy, zf - dz)
                res = res + (u if dx else 1.0 - u) * (v if dy else 1.0 - v) * (w if dz else 1.0 - w) * g
    return res


def crystal_kinds(q, r, seed):
    """the mineral kind (0 white feldspar, 1 K-feldspar, 2 quartz, 3 biotite) at local points q (n, 3) for table row r
    and the object seed: look.cuh lk_crystal, in numpy (float64; the card's float32 can differ at a few outline points)"""
    s = int(seed) * 13 + 1301
    c = r[96]
    g = q / c
    if r[98] > 0.0:
        sc = 1.25
        w = [_noise3(g[:, 0] * sc + a, g[:, 1] * sc + b, g[:, 2] * sc + cc, s + k) * 3.71
             for a, b, cc, k in ((0.13, 0.71, 0.37, 11), (0.53, 0.29, 0.91, 12), (0.83, 0.47, 0.19, 13))]
        g = g + r[98] * np.stack(w, 1)
    if r[148] > 0.0:                               # the fine warp octave (update 18): jagged edges
        s2 = 4.0
        j = [_noise3(g[:, 0] * s2 + a, g[:, 1] * s2 + b, g[:, 2] * s2 + cc, s + k) * 3.71
             for a, b, cc, k in ((0.61, 0.17, 0.83, 14), (0.29, 0.93, 0.41, 15), (0.77, 0.53, 0.07, 16))]
        g = g + r[148] * np.stack(j, 1)
    cxyz = np.floor(g).astype(np.int64)
    best = np.full(len(q), 1e9)
    best2 = np.full(len(q), 1e9)
    bh = np.zeros(len(q), np.uint64)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                I, J, K = cxyz[:, 0] + i, cxyz[:, 1] + j, cxyz[:, 2] + k
                p = np.stack([I + _u01(_hc(I, J, K, s + 1)), J + _u01(_hc(I, J, K, s + 2)), K + _u01(_hc(I, J, K, s + 3))], 1)
                d = ((g - p) ** 2).sum(1) - r[97] * (_u01(_hc(I, J, K, s + 5)) - 0.5)
                m = d < best
                best2 = np.where(m, best, np.minimum(best2, d))
                best = np.where(m, d, best)
                bh = np.where(m, _hc(I, J, K, s + 7), bh)
    u = _u01(bh)
    kind = np.where(u < r[104], 0, np.where(u < r[104] + r[105], 1, np.where(u < r[104] + r[105] + r[106], 2, 3)))
    bnd = np.maximum(1.0 - (best2 - best) / r[102], 0.0) if r[102] > 0.0 else np.zeros(len(q))
    if r[99] > 0.0:
        cl = _noise3(q[:, 0] / r[101] + 0.5, q[:, 1] / r[101] + 0.25, q[:, 2] / r[101] + 0.75, s + 21) * 3.71
        pf = r[100] * bnd * np.clip(0.5 + 0.35 * cl, 0.0, 1.0)
        f = q / r[99]
        fi = np.floor(f).astype(np.int64)
        e = r[103] * r[103] - 1.0
        flake = np.zeros(len(q), bool)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    I, J, K = fi[:, 0] + i, fi[:, 1] + j, fi[:, 2] + k
                    hf = _hc(I, J, K, s + 31)
                    on = _u01(hf) < pf
                    dx = f[:, 0] - (I + _u01(_hc(I, J, K, s + 32)))
                    dy = f[:, 1] - (J + _u01(_hc(I, J, K, s + 33)))
                    dz = f[:, 2] - (K + _u01(_hc(I, J, K, s + 34)))
                    th = 2 * np.pi * _u01(_h32(hf ^ np.uint64(0x3c6ef372)))
                    ct = 2.0 * _u01(_h32(hf ^ np.uint64(0xa54ff53a))) - 1.0
                    st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
                    ra = dx * st * np.cos(th) + dy * st * np.sin(th) + dz * ct
                    d2 = dx * dx + dy * dy + dz * dz + e * ra * ra
                    rr2 = np.full(len(q), 0.36)
                    if r[156] > 0.0:                  # update 23: a frayed edge (look.cuh)
                        fr = _noise3(f[:, 0] * 3.0 + 0.3, f[:, 1] * 3.0 + 0.7, f[:, 2] * 3.0 + 0.1, s + 35) * 3.71
                        near = d2 < 0.36 * (1.0 + r[156]) ** 2
                        rr2 = np.where(near, 0.36 * np.maximum(1.0 + r[156] * fr, 0.2), rr2)
                    flake |= on & (d2 < rr2)
        kind = np.where(flake & (bnd > 0.0), 3, kind)
    return kind


_EFF = {}


def crystal_shares(r, n=60000, seed=1):
    """the kinds' effective area (= volume, Delesse) shares of table row r: random points in a 20-cell cube"""
    rng = np.random.default_rng(97)
    q = rng.uniform(-10.0, 10.0, (n, 3)) * r[96]
    k = crystal_kinds(q, r, seed)
    return [float((k == i).mean()) for i in range(4)]


def ring_share(th, k, zone, w, n=4000):
    """the Liesegang rings' mean over the volume (look.cuh): E[b] with the patch field v ~ N(0, 1), dr = (v - th) k
    rings behind the edge, b = (the ring profile at dr) x (1 - dr / zone) inside 0 < dr < zone, integrated exactly
    over dr (the profile and the inward fade correlate); the rings' strengths average 1. (The small warp of dr is
    left out: Monte Carlo of look.cuh's formula agrees within ~1 %.)"""
    if k <= 0.0 or zone <= 0.0:
        return 0.0
    wf, wb = 0.5 * w, 2.0 * w
    e = math.exp(-(1.0 - wf) / wb)
    dr = (np.arange(n) + 0.5) / n * zone
    kr = np.floor(dr)
    u = dr - kr
    s = np.clip(u / wf, 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    jp = np.where(kr > 0.5, e, 0.0)
    prof = np.where(u < wf, jp + (1.0 - jp) * s, np.exp(-(u - wf) / wb))
    v = th + dr / k
    pdf = np.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi)
    return float(np.sum(pdf * prof * (1.0 - dr / zone)) * zone / k / n)


def table_row(look):
    """look: the resolved look of one object (library.resolve_look) -> float32 (48,)"""
    r = np.zeros(MATF, np.float32)
    if not look:
        return r
    r[MAT_ON] = 1.0
    r[MAT_RGB:MAT_RGB + 3] = look['albedo']
    r[MAT_ALPHA] = look['alpha']
    r[MAT_F0] = look['F0']
    r[MAT_SIGMA] = look.get('on_sigma', 0.0)
    g = look.get('grain', {})
    r[MAT_GRAIN] = g.get('size_m', 0.0)
    r[MAT_GRAIN_SD] = g.get('log_sd', 0.0)
    r[MAT_GDIR:MAT_GDIR + 3] = g.get('dir', (1.0, 1.0, 1.0))
    r[MAT_FINE] = g.get('fine_m', 0.0)
    r[MAT_FINE_SD] = g.get('fine_log_sd', 0.0)
    r[MAT_INT_K] = g.get('interstice_log', 0.0)
    r0 = float(g.get('interstice_r0', 0.6))
    r[MAT_INT_R0] = r0
    phi, mean = interstice_stats(r0)
    r[MAT_INT_PHI], r[MAT_INT_MEAN] = phi, mean
    r[MAT_GRAIN_HUE] = g.get('hue_log_sd', 0.0)
    r[MAT_CLEAR] = g.get('clear_p', 0.0)
    r[MAT_GRAIN_HDIR:MAT_GRAIN_HDIR + 3] = g.get('hue_dir', (0.0, 0.0, 0.0))
    r[MAT_CLU_L] = g.get('cluster_m', 0.0)
    r[MAT_CLU_SD] = g.get('cluster_log_sd', 0.0)
    r[MAT_DARK_P] = g.get('dark_p', 0.0)
    r[MAT_DARK_K] = g.get('dark_log', 0.0)
    r[MAT_IRON:MAT_IRON + 3] = look.get('iron_dir', (0.0, 0.0, 0.0))
    m = look.get('mottle', {})
    r[MAT_MOT_L] = m.get('lam_m', 0.0)
    r[MAT_MOT_SD] = m.get('log_sd', 0.0)
    b = look.get('laminae', {})
    r[MAT_LAM_S] = b.get('spacing_m', 0.0)
    r[MAT_LAM_SD] = b.get('iron_log_sd', 0.0)
    r[MAT_LAM_GREY] = b.get('grey_log_sd', 0.0)
    nb = np.asarray(b.get('normal', (0.0, 1.0, 0.0)), float)
    r[MAT_BED_N:MAT_BED_N + 3] = nb / max(np.linalg.norm(nb), 1e-12)
    r[MAT_LAM_WARP] = b.get('warp_m', 0.0)
    sp = look.get('spots', {})
    r[MAT_SPOT_CELL] = sp.get('cell_m', 0.0)
    r[MAT_SPOT_P] = sp.get('share', 0.0)
    r[MAT_SPOT_R] = sp.get('radius_cells', 0.0)
    r[MAT_SPOT_K] = sp.get('log', 0.0)
    mc = look.get('micro', {})
    if mc.get('amp_m', 0.0) > 0.0:
        r[MAT_MIC_A] = mc['amp_m']
        r[MAT_MIC_R] = mc.get('radius_cells', 0.62)
        r[MAT_MIC_CLIP] = mc.get('clip', 1.0)
        r[MAT_MIC_MEAN] = micro_mean(r[MAT_MIC_R], r[MAT_MIC_CLIP])
    st = look.get('stains', {})
    r[MAT_STAIN_DIR:MAT_STAIN_DIR + 3] = st.get('dir', (0.0, 0.0, 0.0))
    r[MAT_PATCH:MAT_PATCH + 3] = (st.get('patch_m', 0.0), st.get('patch_log', 0.0), st.get('patch_th', 0.5))
    zr, kr = st.get('ring_zone', 0.0), st.get('ring_k', 0.0)
    r[MAT_BAND:MAT_BAND + 6] = (st.get('band_m', 0.0), st.get('band_log', 0.0), st.get('band_w', 0.3), zr, kr,
                                ring_share(st.get('patch_th', 0.5), kr, zr, st.get('band_w', 0.3)))
    sk = look.get('skin') or {}
    if sk.get('form'):                             # the fired skin over its as-fired form (a rounded box)
        fm = sk['form']
        r[64] = 1.0
        r[65:68] = fm['half']
        r[68] = fm['round']
        r[69:71] = (sk['depth_m'], sk['width_m'])
        r[71:74] = sk['body_log']
    fr = look.get('fracture') or {}
    if look.get('form') and fr.get('amp_m', 0.0) > 0.0:   # 11-14: broken / worn surfaces granular (below the as-made box)
        r[65:68] = look['form']['half']
        r[68] = look['form']['round']
        r[74], r[75], r[76] = fr['amp_m'], fr['cell_m'], fr.get('start_m', 0.15e-3)
        r[82] = fracture_mean()
    cr = look.get('crystals') or {}
    if cr.get('cell_m', 0.0) > 0.0:               # 11-14 (update 12): granite as crystals by mineral
        r[96:104] = (cr['cell_m'], cr['spread'], cr['warp'], cr['flake_m'], cr['flake_share'], cr['clot_m'],
                     cr['band'], cr['flake_aspect'])
        r[104:107] = cr['shares'][:3]
        for i, kname in enumerate(('feldspar', 'kfeldspar', 'quartz', 'biotite')):
            r[107 + 3 * i:110 + 3 * i] = cr['log'][kname]
        r[148] = cr.get('jag', 0.0)
        r[156] = cr.get('fray', 0.0)
        key = tuple(round(float(x), 6) for x in r[97:119]) + (round(float(r[148]), 6), round(float(r[156]), 6))
        eff = cr.get('eff') or _EFF.get(key)
        if eff is None:
            eff = _EFF[key] = crystal_shares(r, seed=int(look.get('seed', 1)) % 1000003)
        r[119:123] = eff
        r[123], r[124], r[125] = cr['crystal_log_sd'], cr['quartz_alpha'], cr['biotite_alpha']
        op = cr.get('optics') or {}                # 11-14 (update 15): glints, milky feldspar, quartz depth
        r[138], r[139], r[140] = op.get('feldspar_glint', 0.0), op.get('biotite_sheen', 0.0), op.get('glint_alpha', 0.08)
        r[141], r[142] = op.get('quartz_depth_mm', 0.0) * 1e-3, op.get('quartz_inclusion', 0.0)
        r[143], r[144] = op.get('feldspar_milky', 0.0), op.get('milky_mm', 0.6) * 1e-3
        r[145], r[146], r[147] = op.get('zoning_log', 0.0), op.get('seam_log', 0.0), op.get('seam_gap', 0.0)
    dp = look.get('deposit') or {}
    if dp.get('amount', 0.0) > 0.0:                # 11-14 (update 14): the deposit state
        r[128] = dp['amount']
        r[129:132] = dp['albedo']
        r[132], r[133] = dp.get('up_bias', 0.6), dp.get('pore_fill', 1.0)
        r[134], r[135] = dp.get('patch_m', 0.03), dp.get('patch_frac', 0.6)
        r[137] = dp.get('exposure', 0.0)
        srcs = dp.get('sources') or []             # update 21: the top arris's chip scars as run-off sources
        r[155] = float(min(len(srcs), 8))
        for i, sv in enumerate(srcs[:8]):
            r[160 + 4 * i:164 + 4 * i] = sv
        cb = dp.get('crumble') or {}               # update 19: run-off leaves the top at the crumble's bites
        if cb.get('A', 0.0) > 0.0:
            r[149:155] = (cb['A'], cb['lam'], min(cb['tau'], 0.9), cb['seed'], cb.get('A2', 0.0), cb.get('lam2', 1.0))
        r[136] = dp.get('clear_m', 2.5e-3) if look.get('form') else 0.0
        if dp.get('form') and not look.get('form'):
            r[65:68] = dp['form']['half']
            r[68] = dp['form']['round']
            r[136] = dp.get('clear_m', 2.5e-3)
    wt = look.get('weather') or {}
    if wt:                                         # 11-14: weathering's colour (measured shift x w, patchy, mean kept)
        r[77:80] = wt.get('log_shift', (0.0, 0.0, 0.0))
        r[80], r[81] = wt.get('patch_m', 0.0), wt.get('patch_frac', 0.0)
    r[MAT_SEED] = float(int(look.get('seed', 1)) % 1000003)
    return r


def pixel_angle(cam, h):
    a = 2.0 * math.tan(math.radians(float(cam.fov)) * 0.5) / max(float(h), 1.0)
    if getattr(cam, 'ortho', False):
        D, B = cam.ortho_params()
        a *= D / max(B + D, 1e-9)
    return a


class Materials:
    """the per-frame hook (one per GpuView)"""

    def __init__(self):
        self._sig = None
        self._pa = None
        self._mod = None
        self._rows = {}

    def _globals(self, G):
        cp = G.cp
        if self._mod is not G.mod:
            self._mod = G.mod
            self._sig = None
            self._pa = None
        try:
            gm = G.mod.get_global('g_mat')
            gp = G.mod.get_global('g_mat_pa')
        except Exception:
            return None, None                    # a trace.cu without the material block: nothing to do
        return cp.ndarray((MAT_NV * MATF,), cp.float32, gm), cp.ndarray((1,), cp.float32, gp)

    def before_frame(self, G, mirror, scene, cam, w, h):
        tab, pa_g = self._globals(G)
        if tab is None:
            return False
        pa = pixel_angle(cam, h)
        if pa != self._pa:
            pa_g[...] = np.float32(pa)
            self._pa = pa
        T = np.zeros((MAT_NV, MATF), np.float32)
        vix = {id(v): i for i, v in enumerate(G.vols)}
        for o in scene.objects:
            look = (getattr(o, 'material', None) or {}).get('look')
            if not look:
                continue
            ov = mirror.objs.get(o.id)
            v = getattr(ov, 'vol', None) if ov is not None else None
            i = vix.get(id(v), -1) if v is not None else -1
            if 0 <= i < MAT_NV:
                key = id(look)
                row = self._rows.get(key)
                if row is None or row[0] is not look:
                    row = (look, table_row(look))
                    self._rows[key] = row
                T[i] = row[1]
        sig = T.tobytes()
        if sig == self._sig:
            return False
        tab[...] = G.cp.asarray(T.ravel())
        self._sig = sig
        if len(self._rows) > 256:
            self._rows.clear()
        return True
