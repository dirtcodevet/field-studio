"""fs-materials (2026-11-14): the undoable edits behind the Materials & Generators panel.

Every edit is one history command with its resolved result in the log (formula is the record: a replay needs no
generator run, and the same seed gives the same numbers anyway):
  AssignMaterial   an object becomes a library material: its look (o.material['look']), its colour (the look's mean
                   albedo, for every path that reads colour only), its generators on the form (stone relief, edge
                   chips; a crack kept if it had one), and a lattice fine enough for that detail when it holds no
                   sculpt yet. One undo step.
  MaterialBatch    several commands as one undo step: kept ONLY so logs written 11-14 before the one Array still
                   replay (the app's Array is fs-tools' ArrayCopies; its Vary calls reseed)
Builders (pure: they return the commands' arguments):
  assign_args(scene, oid, mid, finish, seed, rough, chips_per_m)
  crack_args(scene, oid, seed, depth_frac, width_mm)          -> a SetBase tree (the crack is geometry only)
  reseed(scene, oid, seed)                                     -> a copy's own material, colour, generator tree
                                                                  (fs-tools' Array calls it for a seeded Vary)
  regenerate(form, gens, h)                                    -> the base tree from the form and its generator specs
Generator specs live on each matgen node's params['spec']: {'gen': 'relief' | 'chips' | 'crack', 'seed', ...}.
"""
import copy
import math

import numpy as np

from ..core import history as HIST
from ..core import ops
from ..core.scene import SculptGrid
from . import generate as G
from . import library as LIB

DETAIL_H = 0.0002          # the lattice a stone's generators need near brick scale (a 0.7 mm crack mouth = 3.5 cells)
H_FRAC = 1.0 / 1000.0      # larger forms: h = size / 1000 (a 2 m slab: 2 mm), never coarser than the object had


def _size(tree):
    F = G.Field(tree)
    return float(np.max(F.hi - F.lo))


def target_h(form, h_now):
    return min(float(h_now), max(DETAIL_H, _size(form) * H_FRAC))


def regenerate(form, gens, h, nid0):
    """the form wrapped in its generators (in order), each block resolved from its spec at lattice h -> (tree, next id)"""
    tree = copy.deepcopy(form)
    nid = int(nid0)
    for sp in gens:
        g = sp.get('gen')
        if g == 'relief':
            blk = G.stone_relief(sp.get('relief', 'sawn'), sp['seed'], h, rough=sp.get('rough', 1.0))
        elif g == 'chips':
            blk, info = G.edge_chips(G.Field(form), sp['seed'], h, per_m=sp.get('per_m', 12.0), box=_box_of(form),
                                     wear=sp.get('wear'))
            sp = dict(sp, info={'chips': info.get('chips'), 'edge_m': info.get('edge_m'),
                                'top_sources': info.get('top_sources', [])})
        elif g == 'crack':
            blk, info = G.crack(G.Field(tree), sp['seed'], h, depth_frac=sp.get('depth_frac', 0.5),
                                width=sp.get('width_mm', 0.7) * 1e-3, at=sp.get('at'), through=sp.get('through', True))
            sp = dict(sp, info=info)
        else:
            continue
        tree = G.wrap(tree, blk, sp, nid)
        tree['name'] = {'relief': 'Stone surface', 'chips': 'Chips', 'crack': 'Crack'}[g]
        nid += 1
    return tree, nid


def _gens_of(o):
    form, specs = G.strip(o.base.to_dict())
    return form, [copy.deepcopy(s) for s in specs]


def _box_of(form):
    """(half size, round) when the as-fired form is a rounded box, else None"""
    if form.get('kind') != 'box':
        return None
    p = form.get('params', {})
    hs = [max(float(v) * 0.5, 1e-6) for v in p.get('size', (0.2, 0.2, 0.2))]
    return hs, min(max(float(p.get('round', 0.0)), 0.0), min(hs))


def with_skin(look, form):
    """a fired skin (a brick's) lies over the AS-FIRED form: the object's base primitive before its generators. A
    rounded box is carried to the look (its half size and round, in the object's frame); another form: no skin.
    11-14 (update 11): every box form is carried as look['form'] (the as-made surface: below it by more than the
    finish's own relief a surface is broken or worn, and granular: look.cuh mat_micro), with the depth where that starts
    (the finish's relief: 2.5 x its RMS + 2 x its tool-stroke depth + 0.1 mm)."""
    if look is None:
        return None
    if form.get('kind') == 'box' and look.get('deposit') and look['deposit'].get('exposure', 0.0) > 0.0 and             look.get('_chips'):                       # run-off leaves the top at the crumble's bites (look.cuh)
        look['deposit']['crumble'] = G.crumble_params(*look['_chips'])
    look.pop('_chips', None)
    if form.get('kind') == 'box' and look.get('deposit'):     # the deposit's arris clearing needs the as-made box
        p = form.get('params', {})
        hs = [max(float(v) * 0.5, 1e-6) for v in p.get('size', (0.2, 0.2, 0.2))]
        look['deposit']['form'] = {'half': hs, 'round': min(max(float(p.get('round', 0.0)), 0.0), min(hs))}
    if form.get('kind') == 'box' and look.get('fracture'):
        p = form.get('params', {})
        hs = [max(float(v) * 0.5, 1e-6) for v in p.get('size', (0.2, 0.2, 0.2))]
        look['form'] = {'half': hs, 'round': min(max(float(p.get('round', 0.0)), 0.0), min(hs))}
        fn = G.FINISH.get(look.get('relief') or 'sawn', G.FINISH['sawn'])
        st = fn.get('strokes') or {}
        look['fracture']['start_m'] = 2.5 * fn['rms'] * float(look.get('rough', 1.0)) + \
            2.0 * float(st.get('amp', 0.0)) * float(look.get('rough', 1.0)) + 0.1e-3
    else:
        look.pop('form', None)
    sk = look.get('skin')
    if sk and form.get('kind') == 'box':
        p = form.get('params', {})
        hs = [max(float(v) * 0.5, 1e-6) for v in p.get('size', (0.2, 0.2, 0.2))]
        sk['form'] = {'half': hs, 'round': min(max(float(p.get('round', 0.0)), 0.0), min(hs))}
    elif sk:
        sk.pop('form', None)
    return look


def _sources_to_look(look, tree):
    """update 21: an exposed deposit's run-off sources from the chips generator's top-arris scars (its spec info)"""
    dp = (look or {}).get('deposit') or {}
    if dp.get('exposure', 0.0) <= 0.0:
        return
    _, specs = G.strip(tree)
    for sp in specs:
        if sp.get('gen') == 'chips':
            dp['sources'] = list((sp.get('info') or {}).get('top_sources') or [])


def _wear_spec(look):
    wt = (look or {}).get('weather') or {}
    return {'R': wt['wear_m'], 'lam': wt['wear_lam_m'], 'var': wt['wear_var']} if wt.get('wear_m', 0.0) > 0.0 else None


def assign_args(scene, oid, mid, finish=None, seed=1, rough=1.0, chips_per_m=12.0, keep_crack=True, variety=None,
                weather=0.0, deposit=0.0, deposit_kind='dust', exposure=0.0, bedding='natural'):
    o = scene.get(oid)
    form, old = _gens_of(o)
    look = LIB.resolve_look(mid, finish, seed, rough, variety, weather=weather, deposit=deposit,
                            deposit_kind=deposit_kind, exposure=exposure, bedding=bedding)
    if look is not None:
        if float(chips_per_m) > 0.0:
            look['_chips'] = (int(seed), float(chips_per_m))
        with_skin(look, form)
    if look is None:
        raise ValueError('%s has no look yet (colour as numbers): physics only' % mid)
    has_sculpt = bool(o.sculpt.slots)
    h = float(o.sculpt.h) if has_sculpt else target_h(form, o.sculpt.h)
    gens = [{'gen': 'relief', 'seed': int(seed), 'relief': look.get('relief') or 'sawn', 'rough': float(rough)},
            {'gen': 'chips', 'seed': int(seed), 'per_m': float(chips_per_m)}]
    if _wear_spec(look):
        gens[1]['wear'] = _wear_spec(look)
    if keep_crack:
        gens += [dict(s) for s in old if s.get('gen') == 'crack']
    nid0 = max(scene.next_id, max(n.id for n in o.base.walk()) + 1)
    tree, nid = regenerate(form, gens, h, nid0)
    _sources_to_look(look, tree)
    mat = dict(copy.deepcopy(o.material) or {}, roughness=look['alpha'], specular=look['F0'], lib=mid,
               finish=look['finish'], variety=look.get('variety'), seed=int(seed), rough=float(rough), look=look,
               weather=float(weather), deposit=float(deposit), deposit_kind=deposit_kind, exposure=float(exposure),
               bedding=bedding)
    col = dict(copy.deepcopy(o.colour) or {}, albedo=list(look['albedo']))
    return {'oid': int(oid), 'material': mat, 'colour': col, 'tree': tree, 'h': h, 'next_id': nid}


def crack_args(scene, oid, seed=1, depth_frac=0.5, width_mm=0.7, at=None, through=True):
    o = scene.get(oid)
    form, gens = _gens_of(o)
    gens = [g for g in gens if g.get('gen') != 'crack'] + [
        {'gen': 'crack', 'seed': int(seed), 'depth_frac': float(depth_frac), 'width_mm': float(width_mm),
         'at': at, 'through': bool(through)}]
    nid0 = max(scene.next_id, max(n.id for n in o.base.walk()) + 1)
    tree, nid = regenerate(form, gens, float(o.sculpt.h), nid0)
    return {'oid': int(oid), 'tree': tree, 'next_id': nid}


def reseed(scene, oid, seed):
    """for any copying tool's Vary (fs-tools' ArrayCopies): what a copy of oid carries with its own seed - its material
    (the look drawn for that seed), its colour and its base tree (the same form, its generators re-run with the seed;
    a crack is the original's own and is not copied). None when oid has no library material.
    -> {'material', 'colour', 'tree', 'next_id'}"""
    o = scene.get(oid)
    mat = o.material or {}
    if not mat.get('lib'):
        return None
    form, gens = _gens_of(o)
    chips_ = [g for g in gens if g.get('gen') == 'chips']
    look_ = LIB.resolve_look(mat['lib'], mat.get('finish'), int(seed), mat.get('rough', 1.0), mat.get('variety'),
                             weather=float(mat.get('weather', 0.0)), deposit=float(mat.get('deposit', 0.0)),
                             deposit_kind=mat.get('deposit_kind', 'dust'), exposure=float(mat.get('exposure', 0.0)),
                             bedding=mat.get('bedding', 'natural'))
    if chips_ and float(chips_[0].get('per_m', 12.0)) > 0.0:
        look_['_chips'] = (int(seed), float(chips_[0].get('per_m', 12.0)))
    look = with_skin(look_, form)
    gs = [dict(g, seed=int(seed)) for g in gens if g.get('gen') != 'crack']
    for g in gs:                                   # the copy's own wear (its seed), the same weathering state
        if g.get('gen') == 'chips' and _wear_spec(look):
            g['wear'] = _wear_spec(look)
    nid0 = max(scene.next_id, max(n.id for n in o.base.walk()) + 1)
    tree, nid = regenerate(form, gs, float(o.sculpt.h), nid0)
    _sources_to_look(look, tree)
    return {'material': dict(mat, seed=int(seed), look=look, roughness=look['alpha'], specular=look['F0']),
            'colour': dict(o.colour, albedo=list(look['albedo'])), 'tree': tree, 'next_id': nid}


@HIST.command
class AssignMaterial(HIST.Command):
    label = 'Material'

    def __init__(self, oid, material, colour, tree, h, next_id=None):
        self.oid = int(oid)
        self.material = copy.deepcopy(material)
        self.colour = copy.deepcopy(colour)
        self.tree = copy.deepcopy(tree)
        self.h = float(h)
        self.next_id = next_id

    def args(self):
        return {'oid': self.oid, 'material': self.material, 'colour': self.colour, 'tree': self.tree, 'h': self.h,
                'next_id': self.next_id}

    def do(self, scene):
        o = scene.get(self.oid)
        self.old = {'material': copy.deepcopy(o.material), 'colour': copy.deepcopy(o.colour), 'base': o.base.to_dict(),
                    'h': float(o.sculpt.h), 'next_id': int(scene.next_id)}
        base = ops.Node.from_dict(self.tree)
        ops.compile_tree(base)                        # validate before anything changes
        relattice = abs(self.h - o.sculpt.h) > 1e-12
        if relattice and o.sculpt.slots:
            raise ValueError('the object holds sculpt data: its lattice stays (%.3f mm)' % (o.sculpt.h * 1e3))
        o.material = copy.deepcopy(self.material)
        o.colour = copy.deepcopy(self.colour)
        o.base = base
        if relattice:
            o.sculpt = SculptGrid(self.h)
        if self.next_id is not None:
            scene.next_id = max(scene.next_id, int(self.next_id))
        o.invalidate()
        scene.touch(structural=True, index=relattice)

    def undo(self, scene):
        o = scene.get(self.oid)
        o.material = copy.deepcopy(self.old['material'])
        o.colour = copy.deepcopy(self.old['colour'])
        o.base = ops.Node.from_dict(self.old['base'])
        relattice = abs(self.old['h'] - o.sculpt.h) > 1e-12
        if relattice:
            o.sculpt = SculptGrid(self.old['h'])
        scene.next_id = self.old['next_id']
        o.invalidate()
        scene.touch(structural=True, index=relattice)

    def undo_state(self):
        st = HIST._json_exact({'old': self.old})
        return None if st is None else (lambda: (st, {}))

    def restore_undo_state(self, scene, meta, arrays):
        self.old = meta['old']


@HIST.command
class MaterialBatch(HIST.Command):
    """sub-commands (log dicts) done in order and undone in reverse, as one step"""
    label = 'Array'

    def __init__(self, steps, label=None):
        self.steps = copy.deepcopy(steps)
        self._cmds = None
        if label:
            self.label = label

    def args(self):
        return {'steps': self.steps, 'label': self.label}

    def do(self, scene):
        if self._cmds is None:
            self._cmds = [HIST.Command.from_log(d) for d in self.steps]
        for c in self._cmds:
            c.do(scene)

    def undo(self, scene):
        for c in reversed(self._cmds):
            c.undo(scene)


def assign(hist, scene, oid, mid, finish=None, seed=1, rough=1.0, chips_per_m=12.0, variety=None, weather=0.0,
           deposit=0.0, deposit_kind='dust', exposure=0.0, bedding='natural'):
    return hist.push(AssignMaterial(**assign_args(scene, oid, mid, finish, seed, rough, chips_per_m, variety=variety,
                                                  weather=weather, deposit=deposit, deposit_kind=deposit_kind,
                                                  exposure=exposure, bedding=bedding)))


def add_crack(hist, scene, oid, seed=1, depth_frac=0.5, width_mm=0.7, at=None, through=True):
    a = crack_args(scene, oid, seed, depth_frac, width_mm, at, through)
    return hist.push(HIST.SetBase(a['oid'], a['tree'], a['next_id']))


def fast_props(scene, oid):
    """the as-made volume (m^3) and own-axes size (m, 3) of a box or sphere form without sculpt data, analytic, or None.
    A rounded box is its inner box grown by the round r (Steiner): V = L1 L2 L3 + 2 (L1 L2 + L2 L3 + L3 L1) r +
    pi (L1 + L2 + L3) r^2 + 4/3 pi r^3, L the inner sides. Generators (chips, a crack) take well under 1 % off; not
    counted (the 'as made' volume)."""
    o = scene.get(oid)
    if o.sculpt.slots:
        return None
    form, _ = G.strip(o.base.to_dict())
    if form.get('children'):
        return None
    sc = o.transform.get('scale', 1.0)
    if isinstance(sc, (list, tuple)):
        if max(sc) - min(sc) > 1e-12:
            return None
        sc = sc[0]
    s = float(sc)
    p = form.get('params', {})
    if form.get('kind') == 'box':
        hs = [max(float(v) * 0.5, 1e-6) for v in p.get('size', (0.2, 0.2, 0.2))]
        r = min(max(float(p.get('round', 0.0)), 0.0), min(hs))
        L1, L2, L3 = (2.0 * (h - r) for h in hs)
        V = L1 * L2 * L3 + 2.0 * (L1 * L2 + L2 * L3 + L3 * L1) * r + math.pi * (L1 + L2 + L3) * r * r + \
            4.0 / 3.0 * math.pi * r ** 3
        return V * s ** 3, [2.0 * h * s for h in hs]
    if form.get('kind') == 'sphere':
        r = max(float(p.get('radius', 0.1)), 1e-6)
        return 4.0 / 3.0 * math.pi * r ** 3 * s ** 3, [2.0 * r * s] * 3
    return None


_PROPS = {}


def properties(scene, oid, n=128):
    """what the panel shows for an object with a material: volume, mass and the material's physics (11-14: analytic for
    a box or sphere form; else sampled, cached per object and scene version)"""
    from ..tools import measure as MS
    o = scene.get(oid)
    mid = (o.material or {}).get('lib')
    fp = fast_props(scene, oid)
    if fp is not None:
        V = fp[0]
    else:
        key = (int(oid), int(scene.version), int(n))
        V = _PROPS.get(key)
        if V is None:
            if len(_PROPS) > 64:
                _PROPS.clear()
            V = _PROPS[key] = MS.volume(scene, oid, n=n)
    d = {'volume_L': V * 1e3}
    if fp is not None:
        d['dims_m'] = fp[1]
    if mid:
        d.update(LIB.physics(mid, V))
        d['words'] = LIB.words(mid)
        d['material'] = mid
        d['finish'] = (o.material or {}).get('finish')
        d['seed'] = (o.material or {}).get('seed')
    return d
