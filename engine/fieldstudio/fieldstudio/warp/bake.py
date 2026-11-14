"""A warp applied to an object (TOOLSET_DESIGN section 5 item 3): F'(x) = F(phi^-1(x)), written into its sculpt layer.

The object's field is its base program plus its sculpt lattice (core/scene.py). A warp of the object reads the field as
it was at the press through the warp's inverse map, at every lattice node the motion can reach, and writes the
difference from the base into the lattice:
    delta'(x) = F0(psi(x)) / max(1, |J_psi(x)^T grad F0(psi(x))|) - F_base(x)        psi = phi^-1
- nodes the warp leaves in place keep their value exactly (bit for bit);
- the division keeps the value a lower bound of the distance where the map compresses (the tracer's safety); where it
  stretches the raw value already is one; the release re-distance (core/brush.redistance, fs-core's) then makes the
  band a true distance again;
- bricks are added where the new surface or its band arrives (sign change, a value within the band, or a value lower
  than the old: the surface came closer), so no stale value outside them can overstate the distance;
- the mask (and region ids) ride with the surface: mask'(x) = mask(psi(x)).
Undo is exact (a brush.StrokeRecord: the bricks' prior values, the bricks added).

WarpObject(oid, spec) is the Command: spec is a JSON description of the warp (warp_from_spec), so the op log replays it
to the same state hash: the depth images and the contact grid are read again from the scene as it was before it.
"""
import copy
import math

import numpy as np
from numba import njit, prange

from ..core import program as P
from ..core import brush as BR
from ..core.history import Command, command
from . import build as WB
from . import flow as FL
from .lens import depth_images

B = P.BRICK
BAND_BRICKS = 2.0        # the band (in bricks) around the new surface that every warp writes in full
W_BAND = 12.0            # fs-core's truncated distance band (02ab2cfa): stored values within +-12 h, clamped beyond
CARD = {'on': False, 'error': None}   # set on by toolset/jobs.register when the window has the card (its own process)


def _rs(o):
    from ..core import ops
    t = o.transform
    return ops.rot_matrix(t['rotate']), np.asarray(t['translate'], float), max(float(t['scale']), 1e-9)


def to_world(o, X):
    R, t, s = _rs(o)
    return (np.asarray(X, float) * s) @ R.T + t


def to_local(o, W):
    R, t, s = _rs(o)
    return ((np.asarray(W, float) - t) @ R) / s


@njit(cache=True, parallel=True)
def _eval_local(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts, out):
    for n in prange(pts.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        out[n] = P.eval_local_t(code, prm, oi_tab, of_tab, index_flat, delta, oi, pts[n, 0], pts[n, 1], pts[n, 2],
                                vs, ps, 1e9)


@njit(cache=True, parallel=True)
def _eval_base(code, prm, oi_tab, oi, pts, out):
    for n in prange(pts.shape[0]):
        vs = np.empty(P.STACK)
        ps = np.empty(4 * P.STACK)
        out[n] = P.eval_prog_t(code, prm, oi_tab[oi, P.OI_C0], oi_tab[oi, P.OI_C1], pts[n, 0], pts[n, 1], pts[n, 2],
                               vs, ps, 1e9)


@njit(cache=True, parallel=True)
def _mask_at(index_flat, mask, oi_tab, oi, h, pts, out):
    for n in prange(pts.shape[0]):
        out[n] = P.grid_sample(index_flat, mask, oi_tab, oi, h, pts[n, 0], pts[n, 1], pts[n, 2])


def object_field(scene, o, pk=None):
    """world field of object o (as the scene is now): f(Wpts) -> values (world units)"""
    pk = pk or scene.pack()
    oi = scene.index_of(o.id)
    R, t, s = _rs(o)

    def f(Wp):
        X = np.ascontiguousarray(((np.asarray(Wp, float) - t) @ R) / s)
        out = np.empty(len(X))
        _eval_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, X, out)
        return out * s
    return f


def contact_grid(scene, others, box, h, pad):
    """a sampled distance (world) to the union of the objects `others` over box (lo, hi) + pad, spacing h"""
    pk = scene.pack()
    lo = np.asarray(box[0], float) - pad
    hi = np.asarray(box[1], float) + pad
    n = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    cap = 160
    if n.max() > cap:                            # bounded cost: a coarser grid for a big box
        h = float((hi - lo).max() / (cap - 1))
        n = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    ax = [lo[a] + h * np.arange(n[a]) for a in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    d = np.full(len(G), 1e9)
    for o in others:
        d = np.minimum(d, object_field(scene, o, pk)(G))
    return {'grid': d.reshape(tuple(n)), 'origin': lo, 'h': h}


def column_travel(lens, contact, D):
    """Whole part with Stop at contact: for each lens column, how far its far face can travel along the motion before it
    meets the contact surface (minus the Gap), read from the press; columns that miss the part take their neighbours'"""
    from .flow import _tri
    from .lens import _fill_misses
    D = np.asarray(D, float)
    L = float(np.linalg.norm(D))
    hit = lens['hit']
    if L < 1e-12:
        return np.zeros(hit.shape)
    dh = D / L
    res = lens['res']
    u = np.linspace(-lens['half'], lens['half'], res)
    U1, U2 = np.meshgrid(u, u, indexing='xy')
    zfar = np.where(hit, lens['zfar'], 0.0)
    P = lens['c0'][None, :] + U1.reshape(-1, 1) * lens['e1'] + U2.reshape(-1, 1) * lens['e2'] + \
        zfar.reshape(-1, 1) * lens['n']
    G, o, h = contact['grid'], contact['origin'], contact['h']
    gap = contact.get('gap', 0.0)
    hf = hit.reshape(-1)

    def dist(p):
        return _tri(G, o[0], o[1], o[2], h, p[0], p[1], p[2], 1e9) - gap
    T = np.full(len(P), L + 1.0)
    for i in range(len(P)):
        if not hf[i]:
            continue
        t = 0.0
        p = P[i]
        d = dist(p)
        while t < L + 1e-3 and d > 1e-7:
            t += max(d, 1e-6) if d < 1e8 else L + 1.0
            d = dist(p + t * dh)
        if t < L + 1e-3:
            a_, b_ = max(t - 2.0 * h, 0.0), t
            for _ in range(30):
                m_ = 0.5 * (a_ + b_)
                if dist(p + m_ * dh) > 0:
                    a_ = m_
                else:
                    b_ = m_
            T[i] = 0.5 * (a_ + b_)
    T, _ = _fill_misses(np.where(hit, T.reshape(res, res), 0.0), hit)
    return T


# ---- the warp from its JSON spec -------------------------------------------------------------------------------------
def warp_from_spec(scene, oid, spec):
    """spec (JSON): {'motion': 'anchored' | 'rigid' | 'affine' | 'path', 'support': {...}, 'D' | 'R','t','pivot' |
    'A','tau','pivot' | 'points', 'keep_volume', 'contact': {'objects': [ids] | 'others', 'gap', 'eps'}}
    support: {'metric': 'ball' | 'lens', 'c0', 'radius', 'hardness', 'smoothing', 'falloff', 'n' (lens, into the
    surface), 'reach' ('near' | 'whole' | 'through'), 'lens_res'}"""
    o = scene.get(oid)
    s = spec['support']
    metric = s.get('metric', 'ball')
    if spec['motion'] == 'deform':
        # bend / twist / taper of the whole object (toolset/moves.ClosedWarp: closed form, exact inverse); its box is
        # the object's box and where the deformation carries it
        from ..toolset.moves import ClosedWarp
        from ..toolset.common import world_box
        w = ClosedWarp(spec['kind'], spec['centre'], spec['axis'], spec['extent'], spec['amount'],
                       spec.get('direction'))
        lo, hi = world_box(o)
        g = np.stack(np.meshgrid(*[np.linspace(lo[a], hi[a], 7) for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
        allp = np.concatenate([g, w.forward(g)])
        pad = BAND_BRICKS * B * o.sculpt.h * _rs(o)[2] + 0.1 * float(np.linalg.norm(hi - lo))
        w.box = np.concatenate([allp.min(0) - pad, allp.max(0) + pad])
        w.local = False                       # a whole-object deformation: never the value blend (it tears)
        if spec.get('mask_motion') and scene.store.has('mask'):
            # a masked patch stays: the motion itself weighted by the mask (toolset/moves.MaskedDeform); specs written
            # before 'mask_motion' replay as they were (the mask not read)
            from ..toolset.moves import MaskedDeform, deform_weight
            sm = float(spec.get('smoothing', 0.0))
            einfo = {}
            gw = deform_weight(scene, o, sm, cw=w, info=einfo)
            if gw is not None:
                w = MaskedDeform(w, gw[0], gw[1], gw[2], smoothing=sm)
                w.edge = einfo
        return w
    if metric == 'region':
        # a painted region or the unmasked part, carried into the material behind its surface, soft edge = Smoothing
        from ..toolset.moves import region_grid
        G, glo, gh = region_grid(scene, o, s.get('source', 'unmasked'), float(s.get('smoothing', 0.0)))
        sup = WB.Support('grid', c0=s.get('c0', [0.0, 0.0, 0.0]), radius=1.0, grid=G, grid_origin=glo, grid_h=gh)
        metric = 'grid'
    lens = None
    reach = s.get('reach', 'near')
    rt = float(s['radius']) + float(s.get('smoothing', 0.0))
    if metric == 'lens' and reach in ('near', 'whole'):
        f = object_field(scene, o)
        lens = depth_images(f, s['c0'], s['n'], half=1.15 * rt, res=int(s.get('lens_res', 48)), reach=reach,
                            radius_total=rt, eps=s.get('eps'), wall=s.get('wall'))
    elif metric == 'lens':
        from .lens import lens_frame
        n, e1, e2 = lens_frame(s['n'])
        lens = {'n': n, 'e1': e1, 'e2': e2, 'c0': np.asarray(s['c0'], float), 'half': 1.15 * rt, 'res': 2,
                'pix': 1.0, 'u0': 0.0, 'v0': 0.0, 'img': np.zeros((2, 2, 2)), 'eps': max(1e-4, 0.05 * rt),
                'zfar': np.array([[4.0 * rt]])}
    if metric != 'grid':
        sup = WB.Support(metric, c0=s['c0'], radius=s['radius'], hardness=s.get('hardness', 0.0),
                         smoothing=s.get('smoothing', 0.0), falloff=s.get('falloff', 'soft'),
                         lens=lens, reach=reach if metric == 'lens' else 'none')
    sup.front_pad = BAND_BRICKS * B * o.sculpt.h * _rs(o)[2] + 1e-3
    contact = None
    c = spec.get('contact')
    if c:
        ids = c.get('objects', 'others')
        if ids == 'others':
            others = [x for x in scene.objects if x.id != oid and scene.shown(x)]
        else:
            others = [scene.get(int(i)) for i in ids]
        if others:
            lo, hi = sup.box()
            mv = spec.get('D') or spec.get('t') or [0.0, 0.0, 0.0]
            mv = np.abs(np.asarray(mv, float))
            hc = float(c.get('h', max(o.sculpt.h * _rs(o)[2], 2.5e-4)))
            contact = contact_grid(scene, others, (lo - mv, hi + mv), hc, pad=2 * hc + 1e-3)
            contact['gap'] = float(c.get('gap', 0.0))
            contact['eps'] = float(c.get('eps', 1.5 * contact['h']))
    if contact is not None and metric == 'lens' and reach == 'whole' and spec['motion'] == 'anchored':
        lens['travel'] = column_travel(lens, contact, spec['D'])
    kv = bool(spec.get('keep_volume', False))
    m = spec['motion']
    if m == 'anchored':
        return WB.anchored(sup, spec['D'], keep_volume=kv, contact=contact)
    if m == 'rigid':
        return WB.rigid(sup, np.asarray(spec['R'], float), spec['t'], spec['pivot'], keep_volume=kv, contact=contact)
    if m == 'affine':
        return WB.affine(sup, spec['A'], spec['tau'], spec['pivot'], keep_volume=kv, contact=contact)
    if m == 'path':
        return WB.along_path(sup, spec['points'], keep_volume=kv, contact=contact)
    raise ValueError(m)


# ---- the bake --------------------------------------------------------------------------------------------------------
def _brick_nodes(keys, h):
    """local node coordinates of bricks (n,3 int keys) -> (n*512, 3), brick-major, i/j/k order as the pool stores"""
    o = np.arange(B)
    I, J, K = np.meshgrid(o, o, o, indexing='ij')
    off = np.stack([I, J, K], -1).reshape(-1, 3)
    nodes = (keys[:, None, :] * B + off[None, :, :]).reshape(-1, 3)
    return nodes.astype(np.float64) * h


def candidate_bricks(o, box_world):
    """the object's bricks (local keys) whose node box meets the warp's swept box"""
    lo, hi = np.asarray(box_world[:3]), np.asarray(box_world[3:])
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    L = to_local(o, c)
    h = o.sculpt.h
    blo = np.floor(L.min(0) / (B * h)).astype(np.int64) - 1
    bhi = np.floor(L.max(0) / (B * h)).astype(np.int64) + 1
    rng = [np.arange(blo[a], bhi[a] + 1) for a in range(3)]
    K = np.stack(np.meshgrid(*rng, indexing='ij'), -1).reshape(-1, 3)
    return K


def bake(scene, oid, w, rec, band_bricks=BAND_BRICKS, tol=2e-6, max_bricks=60000, use_mask=False):
    """apply warp w to object oid's field, recording undo data in rec. -> stats dict. use_mask: masked areas change
    0.0 mm (design 3.2: "Masked areas change 0.0 mm under every tool"): a node's new value is blended back to its current
    one by the mask there (1 keeps it exactly), the soft mask's edge a soft edge; new commands set it (older logs replay
    without)"""
    o = scene.get(oid)
    oi = scene.index_of(oid)
    g = o.sculpt
    st = scene.store
    h = g.h
    _, _, s = _rs(o)
    pk = scene.pack()
    keys = candidate_bricks(o, w.box)
    if len(keys) > max_bricks:
        raise ValueError('the warp reaches %d bricks of this object (cap %d): use a smaller move or a coarser level'
                         % (len(keys), max_bricks))
    X = _brick_nodes(keys, h)                                   # local
    Wp = to_world(o, X)
    psi = None
    if CARD['on'] and hasattr(w, 'fp'):
        # the window owns the card: the map through warp.cuh (float; parity with the CPU kernels: fs-tools t1/g1)
        try:
            from . import card as WC
            psi = WC.map_points(w, Wp, True)
        except Exception as ex:
            CARD['error'] = repr(ex)
            psi = None
    if psi is None:
        psi = w.inverse(Wp)
    moved = np.any(np.abs(psi - Wp) > 1e-12, axis=1).reshape(len(keys), -1)
    bm = moved.any(1)
    keys, X, Wp, psi, moved = keys[bm], X.reshape(len(bm), -1, 3)[bm], Wp.reshape(len(bm), -1, 3)[bm], \
        psi.reshape(len(bm), -1, 3)[bm], moved[bm]
    nb = len(keys)
    if nb == 0:
        return {'bricks': 0, 'new': 0, 'nodes': 0}
    Xf = np.ascontiguousarray(X.reshape(-1, 3))
    Yl = np.ascontiguousarray(to_local(o, psi.reshape(-1, 3)))
    # the field as it is (before) at the nodes, and at the pulled-back points
    Fcur = np.empty(len(Xf))
    _eval_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, Xf, Fcur)
    Fb = np.empty(len(Xf))
    _eval_base(pk.code, pk.prm, pk.oi_tab, oi, Xf, Fb)
    F0 = np.empty(len(Yl))
    _eval_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, Yl, F0)
    Mx = None
    if use_mask and st.has('mask') and getattr(w, 'local', True):
        # the mask at each node: masked material stays (the blend, then the gradient's bound below as for any pulled-back
        # value; Mask > Invert reads the shown mask). Local tools only (Shape, a ball or lens support): a whole-object
        # move with a masked patch would tear the material where staying meets moving: its motion is weighted by the
        # mask instead (toolset/moves.MaskedDeform; w.local False)
        Mx = np.empty(len(Xf))
        _mask_at(pk.index_flat, st.kview('mask'), pk.oi_tab, oi, h, Xf, Mx)
        Mx = np.clip(Mx, 0.0, 1.0)
        if getattr(o, 'mask_inv', False) or pk.of_tab[oi, P.OF_MINV] > 0.5:
            Mx = 1.0 - Mx
        F0 = F0 + (Fcur - F0) * Mx
    # |grad_x F0(psi(x))| from the brick's own node grid (central differences inside a brick, one-sided at its faces):
    # the chain rule's J_psi^T grad F0 without evaluating the field again (it was 6 more evaluations a node: 7x the cost)
    Fb4 = F0.reshape(nb, B, B, B)
    g2 = np.zeros((nb, B, B, B))
    for a in range(3):
        g2 += np.gradient(Fb4, h, axis=1 + a) ** 2
    gn = np.sqrt(g2).reshape(-1)
    Fnew = np.where(gn > 1.02, F0 / np.maximum(gn, 1.0), F0)      # compressed: a lower bound kept; 2 % is the FD's noise
    mv = moved.reshape(-1)
    Fnew = np.clip(Fnew, -W_BAND * h, W_BAND * h)            # fs-core's band: a true distance within W, clamped beyond
    Fnew = np.where(mv, Fnew, Fcur)                             # unmoved nodes keep their value exactly
    if Mx is not None:
        Fnew = np.where(Mx >= 1.0 - 1e-6, Fcur, Fnew)          # fully masked nodes keep their value exactly
    band = band_bricks * B * h
    Fn_b = Fnew.reshape(nb, -1)
    Fc_b = Fcur.reshape(nb, -1)
    # a brick is added where the new surface or its band lies, and outside the form wherever the old value would
    # overstate the new distance (the tracer steps by it there); inside, a stale value is never stepped on
    need = (np.any(np.sign(Fn_b) != np.sign(Fc_b), 1) | np.any(np.minimum(np.abs(Fn_b), np.abs(Fc_b)) < band, 1)
            | np.any((Fn_b > 0.0) & (Fn_b < Fc_b - tol - 1e-3 * np.abs(Fc_b)), 1))
    # the mask rides with the surface
    has_mask = st.has('mask')
    if has_mask:
        Mv = np.empty(len(Yl))
        _mask_at(pk.index_flat, st.kview('mask'), pk.oi_tab, oi, h, Yl, Mv)
        if Mx is not None:                                  # masked material keeps its own mask (it did not move)
            Mv = Mv + (Mx - Mv) * Mx
    nodes = 0
    new_keys = []
    slots = []
    for q in range(nb):
        b = tuple(int(v) for v in keys[q])
        sl = g.get_slot(b)
        if sl < 0:
            if not need[q]:
                continue
            new_keys.append(b)
            slots.append(-1)
        else:
            slots.append(sl)
    # allocate the new bricks in one go
    if new_keys:
        ns = st.alloc_many(len(new_keys))
        st.delta[ns] = 0.0
        st.Lb[ns] = 0.0
        if st.has('mask'):
            st.mask[ns] = 0.0
        g.set_slots(new_keys, ns)
        for b in new_keys:
            rec.new.append((o.id, b))
            rec.new_set.add((o.id, b))
    touched = []
    for q in range(nb):
        b = tuple(int(v) for v in keys[q])
        sl = g.get_slot(b)
        if sl < 0:
            continue
        key = (o.id, b)
        if key not in rec.snap and key not in rec.new_set:
            rec.snap[key] = (st.delta[sl].copy(), st.mask[sl].copy(), float(st.Lb[sl]))
        dnew = (Fn_b[q] - Fb.reshape(nb, -1)[q]).reshape(B, B, B)
        old = st.delta[sl].astype(np.float64)
        m_ = moved[q].reshape(B, B, B)
        st.delta[sl] = np.where(m_, dnew, old).astype(np.float32)
        if has_mask:
            mo = st.mask[sl]
            st.mask[sl] = np.where(m_, Mv.reshape(nb, B, B, B)[q], mo).astype(st.mask.dtype)
        touched.append(b)
        nodes += int(m_.sum())
    if new_keys:
        scene.touch(index=True)
    else:
        scene.touch()
    oi = scene.index_of(oid)
    BR.measure_bricks(scene, oi, set(touched) | {(b[0] + dx, b[1] + dy, b[2] + dz) for b in touched
                                                 for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)})
    scene.refresh_object(oi)
    return {'bricks': len(touched), 'new': len(new_keys), 'nodes': nodes, 'steps': w.steps,
            'touched': touched, 'gmax': float(gn[mv].max()) if mv.any() else 1.0}


class RecUndo:
    """the undo journal for commands whose undo is one brush.StrokeRecord in self.rec (as BrushStroke's): its arrays go
    to disk with the project, so an Open rebuilds the undo history through them (core/undojournal.py)"""

    def undo_state(self):
        rec = getattr(self, 'rec', None)
        if rec is None:
            return None
        from ..core import undojournal as UJ

        def make():
            A = {}
            UJ.record_arrays('rec', rec, A)
            return {'kind': type(self).__name__}, A
        return make

    def restore_undo_state(self, scene, meta, A):
        from ..core import undojournal as UJ
        self.rec = UJ.record_from('rec', A)

    def evict_undo(self):
        return False


@command
class WarpObject(RecUndo, Command):
    """A warp (Shape, a region Transform, Draw out, Place's settle ...) applied to one object's field. spec: see
    warp_from_spec. redistance: run the release re-distance on the touched bricks (as a brush stroke's release)."""
    label = 'Warp'

    def __init__(self, oid, spec, redistance=True):
        self.oid = int(oid)
        self.spec = copy.deepcopy(spec)
        self.redistance = bool(redistance)
        self.rec = None
        self.info = None

    def args(self):
        return {'oid': self.oid, 'spec': self.spec, 'redistance': self.redistance}

    def do(self, scene):
        from ..toolset.common import refuse_protected
        refuse_protected(scene, [self.oid], {'deform': 'Deform', 'rigid': 'Move region'}.get(self.spec.get('motion'), 'Shape'))
        w = warp_from_spec(scene, self.oid, self.spec)
        rec = BR.StrokeRecord()
        info = bake(scene, self.oid, w, rec, use_mask=bool(self.spec.get('use_mask', False)))
        if getattr(w, 'edge', None):
            # a masked deform: how far the moving part's edge eases around the held patch (the window says it, so the
            # user sees why a big deform needs room: "the edge eases over N mm"; toolset/moves.deform_weight)
            info['edge_mm'] = w.edge.get('edge_mm')
            info['slide_mm'] = w.edge.get('slide_mm')
        if self.redistance and info['touched']:
            try:
                BR.redistance(scene, self.oid, info['touched'], rec, force=True)
            except Exception as ex:                       # the field stays as baked (a lower bound): never fatal
                info['redist_error'] = repr(ex)
        info.pop('touched', None)
        self.rec = rec
        self.info = info
        return info

    def undo(self, scene):
        if self.rec is not None:
            BR.undo_record(scene, self.rec)
            self.rec = None

    def nbytes(self):
        return 256 if self.rec is None else self.rec.nbytes()
