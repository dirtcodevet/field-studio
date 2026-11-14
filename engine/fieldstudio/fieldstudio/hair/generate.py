"""Grow hair on a region of any form in Field Studio (fs-hair, 11-11), with the hair engine's own physics.

The workflow (11-11): choose the type of hair and the body region, choose the area with a tool, refine the details
(colour, thickness, density), run the generator. This module is the generator step: a region's
surface samples (painted in the window) and the settings in, strands out.

The path is the one that grew her approved natural base (engine/hair; hair3's n2c, the accepted baseline):
- roots: Poisson per surface sample at the set density (per cm2), jittered in the sample's tangent disc, re-seated on
  the field;
- follicles: each hair leaves the skin at the kind's emergence angle along the region's flow (the lie: down the form
  under gravity unless combed);
- lengths: the growth cycle (engine/hair/fibre.cycle_lengths), capped at the set length, so ends never sit on one line;
- long hair: guides (about 1 in 130) grown and settled as rods with the fibre's stiffness, rest curl, weight, contact,
  hair-hair pressure and friction (engine/hair/settle.grow, then release), children grown from the guides
  (children.interpolate);
- every point pushed clear of the form (a fibre radius plus 0.2 mm), lengths kept from the root out;
- colour from the pigment (engine/hair/colour.sigma: eumelanin, pheomelanin, ash) per strand with a strand spread.

The engine works z up (gravity -z); Field Studio's world is y up. The mapping is a rotation about x: engine = (x, -z, y),
world = (x, ez, -ey), so the engine's front (-y) is Field Studio's +z, as for her store.

Measured (fs-hair probe, CPU, a 100 mm sphere's cap, 250 mm hair): 3,000 roots / 41 guides ~10 s;
20,000 roots / 153 guides ~24 s (grow 11.6, release 8.5, children 1.5)."""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# the layers the generator composes (hair, style, colour, environment, sim): engine/hair/presets, generic copies of the
# values that grew her natural base (checked 11-11: fibre, radius, oil, waves, locks, grow and the settle's and the
# children's settings resolve identically), so the tool needs no case folder; the panel sets density, length,
# thickness and pigment on top
LAYER_DIRS = [os.path.join(ROOT, 'engine', 'hair', 'presets')]
SHORT_REPORT = {}             # the last generate's short hairs (children.short_hairs: canopy, balance)
DEFAULT_LAYERS = dict(hair='natural_long_wavy', style='natural_airdry', colour='golden_blond',
                      environment='indoor_45rh', sim='full_after_g1000')

# what the user sets in the panel (the kind's cited defaults fill the rest: engine/hair/kinds.py)
KINDS = {
    # name: (label, density /cm2, length mm, diameter um, emergence deg, long)
    'scalp': ('Scalp (long)', 225.0, 300.0, 70.0, 40.0, True),
    'scalp_edge': ('Hairline / baby hairs', 120.0, 60.0, 45.0, 20.0, False),
    'eyebrow': ('Eyebrow', 100.0, 10.0, 75.0, 15.0, False),
    'vellus_body': ('Body vellus', 40.0, 3.0, 25.0, 15.0, False),
    'terminal_limb': ('Limb (terminal)', 30.0, 12.0, 60.0, 15.0, False),
    'axillary': ('Underarm', 60.0, 25.0, 90.0, 30.0, False),
    'lower_body': ('Lower body', 70.0, 30.0, 110.0, 30.0, False),
}


def defaults(kind='scalp'):
    lab, dens, L, d, em, long_ = KINDS[kind]
    return {'kind': kind, 'density': dens, 'length_mm': L, 'diameter_um': d, 'emergence_deg': em,
            'melanin': 0.45, 'redness': 0.045, 'ash': 0.05, 'strand_sd': 0.15, 'guides_per': 130, 'seed': 1}


def to_engine(P):
    P = np.asarray(P, np.float64)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


def to_world(E):
    E = np.asarray(E, np.float64)
    return np.stack([E[..., 0], E[..., 2], -E[..., 1]], -1)


# ---- the body as a lattice (one evaluation of the scene, then everything samples it) ---------------------------------
def body_box(Q, length_m, up=(0.0, 1.0, 0.0), pad=0.03):
    Q = np.asarray(Q, float)
    lo = Q.min(0) - (length_m * 0.75 + pad)
    hi = Q.max(0) + (length_m * 0.75 + pad)
    upv = np.asarray(up, float)
    # hair falls: reach the full length below the region
    lo = np.minimum(lo, Q.min(0) - upv * (length_m + pad) - pad)
    return lo, hi


def body_grid(sdf_world, lo, hi, h=0.003, chunk=1 << 20, log=None):
    """a dense F lattice over [lo, hi] in world (y up): (BodyGrid). sdf_world: (N, 3) -> (N,)"""
    from .style import BodyGrid
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    shape = tuple(int(s) for s in np.ceil((hi - lo) / h).astype(int) + 1)
    n = int(np.prod(shape))
    A = np.empty(n, np.float32)
    ii = np.indices(shape).reshape(3, -1).T
    t0 = time.time()
    for c0 in range(0, n, chunk):
        P = lo + ii[c0:c0 + chunk] * h
        A[c0:c0 + chunk] = sdf_world(P)
        if log and c0 and (c0 // chunk) % 8 == 0:
            log('body field %d%%' % int(100 * c0 / n))
    A = np.nan_to_num(A.reshape(shape), nan=1.0, posinf=1.0, neginf=-1.0)
    if log:
        log('body field lattice %s at %.1f mm in %.1f s' % (list(shape), h * 1000, time.time() - t0))
    return BodyGrid(lo, h, A)


# ---- roots from the region --------------------------------------------------------------------------------------------
def region_roots(Q, N, area_cm2, density, rng, grid=None, spacing=None):
    """Q, N (m, 3) surface samples (world), area_cm2 (m,) -> roots X, normals NR (n, 3), the sample each came from"""
    Q = np.asarray(Q, float)
    N = np.asarray(N, float)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    lam = np.asarray(area_cm2, float) * float(density)
    cnt = rng.poisson(lam)
    src = np.repeat(np.arange(len(Q)), cnt)
    n = len(src)
    if spacing is None:
        spacing = float(np.sqrt(np.median(area_cm2) * 1e-4))
    # jitter in each sample's tangent disc (the samples tile the surface at `spacing`)
    a = np.where(np.abs(N[src, 0]) < 0.9, 1.0, 0.0)
    t1 = np.cross(N[src], np.stack([a, 1 - a, np.zeros(n)], 1))
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(N[src], t1)
    r = np.sqrt(rng.random(n)) * spacing * 0.56
    th = rng.random(n) * 2 * np.pi
    X = Q[src] + t1 * (r * np.cos(th))[:, None] + t2 * (r * np.sin(th))[:, None]
    NR = N[src].copy()
    if grid is not None:
        for _ in range(2):                      # re-seat on the field (Newton along the gradient)
            f, g = grid.sample(X)
            gn = np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            ok = (np.abs(f) < 0.01)[:, None]
            X = np.where(ok, X - g / gn * f[:, None], X)
            NR = np.where(ok, g / gn, NR)
    return X, NR, src


def flow_down(X, NR, up=(0.0, 1.0, 0.0), crown=None, swirl_mm=25.0, turn=1.0):
    """the default lie: down the form under gravity; where the surface faces up (the crown), a WHORL: the hair turns
    round the crown point (clockwise seen from above, as 80-84 % of whorls are: NATURAL_BASE_SOURCES 2, F34-F35) and
    opens to radial over swirl_mm, so the hair near the centre lies round and over it instead of leaving it bare (a
    radial lie left 52 % of a 40 mm crown disc bare, fs-hair probe 21:30). turn: +1 clockwise, -1 counter-clockwise,
    0 radial (the old lie)"""
    upv = np.asarray(up, float)
    fl = -upv[None] - (NR @ -upv)[:, None] * NR
    fn = np.linalg.norm(fl, axis=1)
    if crown is None:
        top = NR @ upv
        crown = X[np.argmax(top)] if len(X) else np.zeros(3)
    aw = X - np.asarray(crown)[None]
    dist = np.linalg.norm(aw, axis=1)
    aw -= (aw * NR).sum(1, keepdims=True) * NR
    aw /= np.maximum(np.linalg.norm(aw, axis=1, keepdims=True), 1e-9)
    if turn and swirl_mm > 0:
        tw = np.cross(NR, aw) * float(np.sign(turn))       # round the crown (NR x radial: counter-clockwise from above)
        tw = -tw                                           # clockwise seen from outside (above the crown)
        s = np.exp(-dist / (swirl_mm / 1000.0))[:, None]
        aw = aw * (1.0 - 0.85 * s) + tw * (0.85 * s + 0.15)
        aw /= np.maximum(np.linalg.norm(aw, axis=1, keepdims=True), 1e-9)
    w = np.clip(fn / 0.35, 0, 1)[:, None]                   # flat (crown) -> the whorl; sloped -> downhill
    fl = fl / np.maximum(fn, 1e-9)[:, None] * w + aw * (1 - w)
    return fl / np.maximum(np.linalg.norm(fl, axis=1, keepdims=True), 1e-9)


# ---- the generator ----------------------------------------------------------------------------------------------------
def generate(Q, N, area_cm2, settings, grid, up=(0.0, 1.0, 0.0), flow=None, log=print, npt=32, card=False):
    """card: the rods' settle on the graphics card (engine/hair/gpu.py, HAIR_GPU=1 for this call: a settle evaluation
    0.248 -> 0.028 s on 514 guides, each launch far under the display watchdog); the CPU stays the reference"""
    prev = os.environ.get('HAIR_GPU')
    if card:
        os.environ['HAIR_GPU'] = '1'
    try:
        out = _generate(Q, N, area_cm2, settings, grid, up, flow, log, npt)
        out['report']['card_settle'] = bool(card)
        return out
    finally:
        if card:
            if prev is None:
                os.environ.pop('HAIR_GPU', None)
            else:
                os.environ['HAIR_GPU'] = prev


def _generate(Q, N, area_cm2, settings, grid, up=(0.0, 1.0, 0.0), flow=None, log=print, npt=32):
    """-> dict of arrays in WORLD (y up): P (n, npt, 3) float32, RAD (n, npt), SIG (n, 3), SIGP (n, npt, 3) f16,
    COL (n, npt, 3) the viewport's base colour, LEN (n,), X roots, NR, report"""
    from engine.hair import layers as LY, settle as SE, children as CH, fibre as FB, field_collider as FC
    from engine.hair import colour as HC
    T = {}
    t_all = time.time()
    s = dict(defaults(settings.get('kind', 'scalp')), **settings)
    kind = s['kind']
    pv = float(s.get('preview', 1.0) or 1.0)           # preview: 1 in 1/pv hairs grown (each drawn covering 1/pv)
    if pv < 1.0:
        s['density'] = float(s['density']) * pv           # the same hairs per guide: a tenth of the guides too
    long_ = KINDS.get(kind, KINDS['scalp'])[5]
    rng = np.random.default_rng(int(s['seed']))
    t = time.time()
    X, NR, src = region_roots(Q, N, area_cm2, float(s['density']), rng, grid=grid)
    n = len(X)
    if n == 0:
        raise ValueError('the region holds no roots: paint a larger area or raise the density')
    if flow is None:
        fl = flow_down(X, NR, up, swirl_mm=float(s.get('whorl_mm', 25.0)), turn=float(s.get('whorl_turn', 1.0)))
    else:
        fl = np.asarray(flow, float)[src]
        fl -= (fl * NR).sum(1, keepdims=True) * NR
        fl /= np.maximum(np.linalg.norm(fl, axis=1, keepdims=True), 1e-9)
    em = np.radians(float(s['emergence_deg']))
    em = em * rng.uniform(0.75, 1.25, n)
    d0 = np.cos(em)[:, None] * fl + np.sin(em)[:, None] * NR
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    T['roots'] = time.time() - t
    log('%d roots on %.1f cm2 (%.0f /cm2)' % (n, float(np.sum(area_cm2)), n / max(float(np.sum(area_cm2)), 1e-9)))

    st, warn = LY.compose(DEFAULT_LAYERS['hair'], DEFAULT_LAYERS['style'], DEFAULT_LAYERS['colour'],
                          environment=DEFAULT_LAYERS['environment'], sim=DEFAULT_LAYERS['sim'], dirs=LAYER_DIRS)
    st['radius'] = dict(st['radius'], root_mm=float(s['diameter_um']) / 2000.0)
    max_mm = float(s['length_mm'])
    Xe, Ne, de = to_engine(X), to_engine(NR), to_engine(d0)

    def sdf_e(P):
        f, _ = grid.sample(to_world(P))
        return f

    t = time.time()
    fb = FB.props(st.get('fibre'), st['radius'], dict(st['oil']))
    mL = FB.cycle_lengths(n, st.get('fibre'), np.random.default_rng(int(s['seed']) + 223), max_mm=max_mm)
    ms = np.random.default_rng(int(s['seed']) + 227).uniform(0.85, 1.15, n)
    T['fibre'] = time.time() - t
    if long_ and max_mm > 60:
        t = time.time()
        key = CH.part_key(Xe, np.zeros(n), np.zeros(n), 60.0)
        ng = max(8, int(round(n / float(s.get('guides_per', 130)))))
        gi, _ = CH.guides(Xe, key, ng, np.random.default_rng(int(s['seed']) + 211), log=None)
        mg, counts, _ = CH.members(key[gi], key)
        f, swarn = SE.settings(st)
        lo_w, hi_w = grid.lo, grid.lo + grid.h * (np.array(grid.A.shape) - 1)
        corners = to_engine(np.array([[x, y, z] for x in (lo_w[0], hi_w[0]) for y in (lo_w[1], hi_w[1])
                                      for z in (lo_w[2], hi_w[2])]))
        body = FC.FieldSDFGrid(sdf_e, corners.min(0), corners.max(0), mm=max(3.0, grid.h * 1000), log=None)
        T['body'] = time.time() - t
        log('%d guides; body lattice %s' % (len(gi), body.report['shape']))
        t = time.time()
        rods = SE.Rods(Xe[gi], Ne[gi], de[gi], fb, mg, mL, ms, seg_root_mm=f['seg_root_mm'],
                       seg_growth=f['seg_growth'], seg_max_mm=f['seg_max_mm'], sample_mm=f['sample_mm'],
                       min_tail_mm=f['min_tail_mm'])
        # the natural wave (as generate._prelude): one drying-section phase field over all the roots, each section its
        # own wavelength and plane, so neighbours share a wave and the head's sections do not (no rings in phase)
        from engine.hair import form as FO
        seed = int(s['seed'])
        wv_rods = bool(f.get('rod_waves', False)) or SE.helix_on(fb)
        sec_all = None
        if wv_rods:
            lcs_w = FO._cfg(st, 'locks', FO.LOCK_CAUSE_DEFAULTS)
            # the wave's phase (fs-hair 11-12, seen on sheet 20, measured: n2c's and this generator's waves
            # ran in step over 35 mm drying sections, a curling barrel's width, so a natural base read as even horizontal
            # bands): for an air-dried natural base the waves are in step only inside a clump formed while wet. The phase
            # is the engine's coherent field (form.py's 10-29 phase field: a smooth random field over the roots plus each
            # hair's own jitter) at about two lock widths, so neighbouring locks wave loosely together (fully random locks
            # read as crimp) and the head's patches do not; the sections (grouping, wavelength, tilt) at the same scale
            wv = dict(st.get('waves') or {}, **dict(s.get('waves') or {}))    # the preset's, then the call's overrides
            pscale = float(wv.get('phase_scale_mm', 15.0))
            sec_mm = float(wv.get('section_mm', pscale if pscale > 0 else lcs_w['section_mm']))
            ph_all, sec_all, ph_sec = FO.section_phase(Xe, sec_mm, seed, lcs_w['phase_jitter'], parts=True)
            if pscale > 0:
                rp = np.random.default_rng(seed + 53)
                kv = rp.normal(0, 1, (24, 3)) / pscale
                bv = rp.uniform(0, 2 * np.pi, 24)
                fld = np.cos((np.asarray(Xe, np.float64) * 1000.0) @ kv.T + bv).sum(1) / np.sqrt(12)
                ph_all = np.pi * fld + rp.normal(0, float(wv.get('lock_jitter', 0.4)), len(Xe))
            lam_sec, tilt_sec = SE.section_wave(len(ph_sec), st.get('waves') or {}, seed)
            SE.rest_waves(rods, st.get('waves') or {}, seed, phase=ph_all[gi], random_mm=float(f['wave_random_mm']),
                          lam=lam_sec[sec_all[gi]], tilt=tilt_sec[sec_all[gi]])
        log('growing %d guides as rods (weight, stiffness, curl, contact, pressure, friction)...' % len(gi))
        state = SE.grow(rods, body, acts=None, cfg=st, pressure=True, friction=True, log=lambda m: None)
        T['grow'] = time.time() - t
        # air-dried, no product (the natural base, as her n2c was): the hair takes the water set in the shape it dried
        # in (engine/hair/actions.set_held: rest turning <- rest + set x retention x (held - rest), dry.set 0.4 over the
        # first dry.over_mm), then lets go; settings.air_dry False leaves it untouched
        P_wet = rods.positions().copy()        # the wet fall: what the short hairs are laid in as the hair dries
        if s.get('air_dry', True):
            t = time.time()
            try:
                from types import SimpleNamespace
                from engine.hair import actions as AC
                av, _ = AC.settings(st)
                dv = av['dry']
                acts_dry = SimpleNamespace(v=av, on_g=None, front_g=np.zeros(rods.G, bool), body=body, counts={},
                                           P_held=None, _warned=set(), warnings=[])
                if dv.get('method', 'air') == 'air':
                    nset = AC.set_held(state, acts_dry) if dv.get('span', 'held') == 'held' else                         AC.dry_set(state, acts_dry)
                    log('air-dried: the set taken at %d joints' % int(nset))
            except Exception as e:
                log('air-dry skipped: %r' % (e,))
            T['air_dry'] = time.time() - t
        t = time.time()
        log('releasing (the hair settles as it lies)...')
        state = SE.release(state, acts=None)
        T['release'] = time.time() - t
        t = time.time()
        gcfg = CH.settings(st)
        gcfg = gcfg[0] if isinstance(gcfg, tuple) else gcfg
        gcfg = dict(gcfg, children_points=npt)
        Pg = rods.positions()
        long_m = mL >= float(gcfg.get('short_mm', 80.0)) / 1000.0
        il, ish = np.nonzero(long_m)[0], np.nonzero(~long_m)[0]
        Pe = np.zeros((n, npt, 3))
        if len(il):
            Pe[il], _ = CH.interpolate(Pg, rods.L, rods.X, Xe[il], de[il], key[il], mL[il], key_g=key[gi],
                                       d0_g=rods.d0, arcs=rods.node_arcs(), cfg=gcfg,
                                       sec_c=sec_all[il] if sec_all is not None else None,
                                       sec_g=sec_all[gi] if sec_all is not None else None)
        T['children'] = time.time() - t
        if len(ish):
            # the short (still growing) hairs: each its own rod under the long hair's layer lying over it (canopy)
            t = time.time()
            log('%d short hairs under the long layer...' % len(ish))
            cano = None
            if bool(st['grow'].get('canopy', True)):
                cano = CH.build_canopy((Pg, rods.W, rods.V, rods.l), Xe[ish], mL[ish], CH._distance(body), gcfg)
            # a washed head's short regrowth hairs are laid down wet with the long hair (water binds them to it) and
            # keep that set as they air-dry (engine/hair children.short_hairs: held + set_r, with static friction
            # under the long layer); without it they stood at their follicle angle where no layer lay over them
            # (the crown spikes: fs-hair probe 21:4x, 4-5 standing up to 57 mm)
            held = r_set = None
            if s.get('air_dry', True) and bool(st['grow'].get('short_set', True)):
                try:
                    from engine.hair import actions as AC
                    av, _ = AC.settings(st)
                    held, _ = CH.interpolate(P_wet, rods.L, rods.X, Xe[ish], de[ish], key[ish], mL[ish],
                                             key_g=key[gi], d0_g=rods.d0, arcs=rods.node_arcs(), cfg=gcfg)
                    r_set = CH.set_share(av['dry'], av['wear'], av['finish'], np.zeros(len(ish)),
                                         np.full(len(ish), 100.0))
                except Exception as e:
                    log('short hairs: no wet set (%r)' % (e,))
                    held = r_set = None
            Ps, shrep = CH.short_hairs(Xe[ish], de[ish], mL[ish], fb, body, scale=ms[ish], cfg=gcfg, log=None,
                                       canopy=cano, held=held, set_r=r_set)
            Pe[ish] = Ps
            T['short'] = time.time() - t
            SHORT_REPORT.clear()
            SHORT_REPORT.update(n=int(len(ish)), canopy=(shrep or {}).get('canopy'),
                                balanced=(shrep or {}).get('balanced'), stalled=(shrep or {}).get('stalled'))
        P = to_world(Pe).astype(np.float32)
        nguides = int(len(gi))
    else:
        t = time.time()
        from engine.hair import grow as GW, kinds as HK
        kn = kind if kind in HK.HAIR_KINDS else 'vellus_body'
        L_ = HK.cycle_lengths(kn, n, rng, {'mode': 'trim', 'max_mm': max_mm})[0] * 1e-3
        L_ = np.maximum(L_, 0.1e-3)
        mL = L_
        rf = GW.Radius(L_, float(s['diameter_um']) * 0.5e-6 * ms, 1.0, HK.v(kn, 'taper_mm') * 1e-3,
                       HK.v(kn, 'tip_ratio'))
        cd = HK.v(kn, 'cd_mm')
        kap = (2.0 / (cd * 1e-3) if cd else 30.0) * np.clip(rng.normal(1, 0.3, n), 0.3, 2.0)
        r = GW.grow_short(sdf_e, Xe, de, -Ne, L_, rf, kap, kap, seat=0.04e-3, gap=0.01e-3,
                          ds=min(0.05e-3, float(np.percentile(L_, 50)) / 6), npt=npt, bend_mode='surface')
        P = to_world(r['pos']).astype(np.float32)
        T['grow_short'] = time.time() - t
        nguides = 0
    # radius along each hair: the shaft, a natural taper over the last few mm (an uncut tip)
    t = time.time()
    seg = np.linalg.norm(np.diff(P.astype(np.float64), axis=1), axis=2)
    L = seg.sum(1)
    arc = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, 1)], 1)
    r0 = float(s['diameter_um']) / 2e6 * ms
    tap = np.clip((L[:, None] - arc) / 0.004, 0.25, 1.0)
    RAD = (r0[:, None] * tap).astype(np.float32)
    # colour: pigment per strand (a spread about the set melanin), its absorption along the hair
    srng = np.random.default_rng(int(s['seed']) + 331)
    m = np.clip(float(s['melanin']) * np.exp(srng.normal(0, float(s['strand_sd']), n)), 0.01, 0.995)
    SIG = HC.sigma(m, float(s['redness']), float(s['ash'])).astype(np.float32)        # per unit radius
    lighter = 1.0 - 0.25 * np.clip(arc / np.maximum(L[:, None], 1e-9), 0, 1)              # lengths lighten (sun, wear)
    SIGP = (SIG[:, None, :] * lighter[..., None]).astype(np.float16)
    COL = base_colour(SIGP.astype(np.float32))
    T['radius_colour'] = time.time() - t
    rep = {'roots': n, 'guides': nguides, 'seconds': {k: round(v, 2) for k, v in T.items()},
           'total_s': round(time.time() - t_all, 1), 'settings': s,
           'length_mm_p10_50_90': np.percentile(L * 1000, [10, 50, 90]).round(1).tolist()}
    return {'P': P, 'RAD': RAD, 'SIG': SIG, 'SIGP': SIGP, 'COL': COL, 'LEN': L.astype(np.float32), 'X': X, 'NR': NR,
            'report': rep}


def base_colour(SIGP):
    """the viewport's working colour of a fibre from its absorption per unit radius (linear): light that crosses a
    fibre and comes back out of the mass (blond hair is bright from the light scattered between fibres, HAIR_DEFINED
    3): a fibre's two-way transmission over ~1.2 radii, lifted by the mass's multiple scattering"""
    tr = np.exp(-np.asarray(SIGP, np.float32) * 1.2)
    return np.clip(0.04 + 0.62 * tr, 0, 1).astype(np.float32)
