"""fs-brush 11-11 (design D4): fill a cavity solid, only as an explicit act ("Fill it solid" in the window's prompt, or the
hidden-voids view). The cavity is found again on the object's own lattice (local multiples of its sculpt h, so rotated
objects such as her store copy work): the outside nodes (6-connected) enclosed within the cavity's box + 3 nodes, the
component at (or nearest to) its centroid; those nodes are set inside, half a cell deep, through tools.common.write_nodes
(one undo step; replay recomputes the same nodes from the same field). Pockets a stroke makes by accident are filled at
the stroke's end by gpu/sculpt.py _stroke_topo; existing cavities are only filled here."""
import copy
import numpy as np

from ..core import brush as BR
from ..core.history import BrushStroke, command
from . import common as C


def fill_cavity(scene, oid, cav, rec):
    """-> the number of nodes filled. cav: {'centroid_m': world point, 'bbox_m': (lo, hi) world}"""
    from scipy import ndimage as ndi
    from ..core.program import eval_points_local
    from .protected_merge import _to_local
    oi = scene.index_of(oid)
    obj = scene.objects[oi]
    h = float(obj.sculpt.h)
    lo_w, hi_w = (np.asarray(v, float) for v in cav['bbox_m'])
    C8 = np.array([[x, y, z] for x in (lo_w[0], hi_w[0]) for y in (lo_w[1], hi_w[1]) for z in (lo_w[2], hi_w[2])])
    L = _to_local(obj, C8)
    I0 = np.floor(L.min(0) / h).astype(np.int64) - 3
    I1 = np.ceil(L.max(0) / h).astype(np.int64) + 3
    shp = I1 - I0 + 1
    if int(np.prod(shp)) > 8_000_000:
        raise ValueError('the cavity box is too large to fill here (%d nodes)' % int(np.prod(shp)))
    ax = [np.arange(I0[a], I1[a] + 1) * h for a in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    pk = scene.pack()
    F = np.empty(len(P))
    eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, np.ascontiguousarray(P), F)
    F = F.reshape(tuple(shp))
    lab, n = ndi.label(~np.pad(F < 0.0, 1))
    border = set(np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                           lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))) - {0}
    lab = lab[1:-1, 1:-1, 1:-1]
    cands = [c for c in range(1, n + 1) if c not in border]
    if not cands:
        return 0
    cl = _to_local(obj, np.asarray(cav['centroid_m'], float)[None])[0]
    ci = np.round(cl / h).astype(np.int64) - I0
    best, bd = None, None
    for c in cands:
        w = np.argwhere(lab == c)
        d = float(np.min(np.sum((w - ci) ** 2, 1)))
        if bd is None or d < bd:
            best, bd = c, d
    m = lab == best
    df = np.zeros(tuple(shp))
    df[m] = -0.5 * h - F[m]
    C.write_nodes(scene, oi, I0, df, rec)
    return int(m.sum())


@command
class FillCavity(BrushStroke):
    """Fill one enclosed cavity solid (design D4: the user's explicit act). cavity: {'centroid_m', 'bbox_m'} (world)."""
    label = 'Fill a cavity'

    def __init__(self, oid, cavity):
        self.oid = int(oid)
        self.cavity = {'centroid_m': [float(v) for v in cavity['centroid_m']],
                       'bbox_m': [[float(v) for v in cavity['bbox_m'][0]], [float(v) for v in cavity['bbox_m'][1]]]}
        self.settings = {}
        self.samples = []
        self.rec = None
        self.filled = 0

    def args(self):
        return {'oid': self.oid, 'cavity': copy.deepcopy(self.cavity)}

    def do(self, scene):
        self.rec = BR.StrokeRecord()
        self.filled = fill_cavity(scene, self.oid, self.cavity, self.rec)
        scene.touch()

    def extend(self, scene, samples):
        return None

    def finish(self, scene):
        return None
