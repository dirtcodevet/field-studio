"""Attached protected parts (11-09, design: soft parts such as a lid are flesh ATTACHED at their
roots, not floating objects; the eyeball has no root).

A part's ROOT is the region where it is fused to the body: painted on the part with the Mask brush, then stored as a
root-weight lattice w (0..1: 1 within R0 of the painted root surface, a smoothstep to 0 at R1) in the part's frame, a
sha-named file referenced by the part's 'root' property (logged with SetObjectProps; undo / save / replay like any
property).

FuseRoot(body, part) bakes the fusion into the BODY's sculpt layer (as the protected merge does):
    F_body' = F_body + w (min(F_body, F_part) - F_body)    over the root's footprint (whole bricks)
a union: where the part stands out of the body the body follows it (seamless), the ramp blends it in, the body is never
carved (r2: F + w (F_part - F) carved grooves where the ramp passed the part's region), and outside the footprint
nothing changes (w = 0: bit for bit). Unedited parts are copies of her body there: the fusion changes nothing.
Collider: nodes where w > 0 are exempt (collide.bound), so the root never pushes the body away from its own part.
"""
import hashlib
import os

import numpy as np

from .history import Command, command
from . import brush as BR

R0, R1 = 0.0005, 0.0015


def root_lattice(points, h, r0=R0, r1=R1):
    """points (n, 3) part-local root surface points -> (w (nx, ny, nz) float32, lo (3,), h)"""
    from scipy.ndimage import distance_transform_edt
    lo = np.floor((points.min(0) - r1 - 2 * h) / h) * h
    hi = np.ceil((points.max(0) + r1 + 2 * h) / h) * h
    dims = (np.round((hi - lo) / h).astype(int) + 1)
    M = np.zeros(tuple(dims), bool)
    idx = np.round((points - lo) / h).astype(int)
    idx = idx[np.all((idx >= 0) & (idx < dims), 1)]
    M[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    d = distance_transform_edt(~M) * h
    t = np.clip((r1 - d) / max(r1 - r0, 1e-9), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32), lo, float(h)


def save_root(w, lo, h, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, '_tmp_root.npz')
    np.savez_compressed(tmp, data=w, lo=lo, h=np.float64(h))
    sha = hashlib.sha256(open(tmp, 'rb').read()).hexdigest()[:16]
    path = os.path.join(out_dir, 'root_%s_%s.npz' % (name.replace(' ', '_'), sha))
    os.replace(tmp, path)
    return path, sha


_CACHE = {}


def load_root(root):
    key = (root['file'], root.get('sha'))
    if key not in _CACHE:
        z = np.load(root['file'])
        _CACHE[key] = (z['data'].astype(np.float32), z['lo'].astype(float), float(z['h']))
    return _CACHE[key]


def root_weight(part, Xp):
    """root weight at part-local points Xp (n, 3); 0 for a part without a root"""
    r = getattr(part, 'root', None)
    if not r:
        return np.zeros(len(Xp))
    w, lo, h = load_root(r)
    from scipy.ndimage import map_coordinates
    q = ((np.asarray(Xp, float) - lo) / h).T
    return map_coordinates(w, q, order=1, mode='constant', cval=0.0)


@command
class FuseRoot(Command):
    """bake a part's root fusion into the body's sculpt layer (see the module text); undo / redo exact"""
    label = 'Fuse part root'

    def __init__(self, body, part):
        self.body = int(body)
        self.part = int(part)
        self.target = self.body                    # MergeProtected's writers read self.target / self.rec
        self.rec = None
        self._vals = None
        self.info = {}

    def args(self):
        return {'body': self.body, 'part': self.part}

    def nbytes(self):
        return 256 + (self.rec.nbytes() if self.rec is not None else 0)

    def do(self, scene):
        from ..tools.protected_merge import MergeProtected, field_of, _to_local, _to_world, _rs
        if self._vals is not None:
            MergeProtected._apply_values(self, scene, self._vals)
            return self.info
        body, part = scene.get(self.body), scene.get(self.part)
        r = getattr(part, 'root', None)
        if not r:
            raise ValueError('%s has no root: paint it on the part with the Mask brush and set it first' % part.name)
        w_l, lo, hw = load_root(r)
        hi = lo + (np.array(w_l.shape) - 1) * hw
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        cb = _to_local(body, _to_world(part, c))
        h = float(body.sculpt.h)
        i0 = (np.floor(cb.min(0) / h).astype(np.int64) // 8) * 8
        i1 = (np.ceil(cb.max(0) / h).astype(np.int64) // 8) * 8 + 7
        ax = [np.arange(i0[a], i1[a] + 1) * h for a in range(3)]
        Xb = np.ascontiguousarray(np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3))
        W = _to_world(body, Xb)
        w = root_weight(part, _to_local(part, W))
        dF = np.zeros(len(Xb))
        act = w > 0.0
        if act.any():
            pk = scene.pack()
            Fb = field_of(scene, pk, body, W[act])
            Fp = field_of(scene, pk, part, W[act])
            # r2 (11-09, dark curved lines seen beside a part): F + w (F_part - F) carved grooves wherever the root's
            # ramp reached past the part's own region (the part's field reads 'outside' there: 1,117 of 2,123 nodes
            # raised, up to 0.665 mm). A part grows OUT of the body: the root is a union, F + w (min(F, F_part) - F),
            # which can only add
            dF[act] = w[act] * (np.minimum(Fb, Fp) - Fb) / _rs(body)[2]
        dF = dF.reshape(tuple(int(v) for v in (i1 - i0 + 1)))
        self.rec = BR.StrokeRecord()
        vals = MergeProtected._write(self, scene, body, i0, dF)
        self._vals = vals
        self.info = {'nodes_in_root': int(act.sum()), 'changed': int((dF != 0).sum()), 'bricks': len(vals)}
        return self.info

    def undo(self, scene):
        BR.undo_record(scene, self.rec)
