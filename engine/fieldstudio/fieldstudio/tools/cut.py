"""CUT: a closed outline drawn on the screen, swept through the object(s) along the view (DESIGN_NOTES 10d).

The swept outline is exact for a perspective camera and mesh-free (P14; cutsdf 11-08, replacing the ear-clipped
triangle cones): the cutter is ONE core 'outline_sweep' node, the outline (any closed polygon, concave included)
on the view plane at focal distance 1 along the mean ray, swept along the rays from the eye, clipped to a cube
that holds the object. Its value is the exact signed 3-D distance to the swept surface (the union of the planar
wedges eye + {a d_i + b d_(i+1)}), signed by projecting the point through the eye and a crossing test on the
outline; points behind the eye are outside. L = 1 (D; program._osweep has the derivation).

mode 'remove'   : object := object - cutter (a punched hole, optional smooth blend).
mode 'separate' : object := object - grow(cutter, kerf/2); a new object := object AND shrink(cutter, kerf/2),
                  moved by `distance` toward the eye (or along `offset_dir`), with the colour, material and a copy
                  of the sculpt layer. A piece with no material inside (coarse 24^3 census) is not created.
"""
import copy

import numpy as np

from ..core import ops
from ..core.history import Command, command
from ..core.scene import SceneObject
from . import common as C


def outline_from_screen(camera, pts_px, w, h):
    """Screen outline (pixel coords) -> (eye, ray directions) in world space."""
    dirs = []
    eye = None
    for x, y in pts_px:
        o, d = camera.ray(float(x), float(y), w, h)
        eye = o
        dirs.append([float(v) for v in d])
    return [float(v) for v in eye], dirs


def _decimate(dirs, nmax=32):
    if len(dirs) <= nmax:
        return dirs
    idx = np.round(np.linspace(0, len(dirs), nmax, endpoint=False)).astype(int)
    return [dirs[i] for i in idx]


def build_cutter(eye_l, dirs_l, extent, ids):
    """Local-frame cutter for an outline (eye, ray directions): one 'outline_sweep' node (no triangles)."""
    D = np.array([C.unit(d) for d in dirs_l])
    m = C.unit(D.mean(0))
    F = C.frame_from(m, [0.0, 1.0, 0.0] if abs(m[1]) < 0.9 else [1.0, 0.0, 0.0])
    u, v = F[:, 0], F[:, 2]
    dm = D @ m
    if np.any(dm <= 1e-6):
        raise ValueError('outline rays must all point forward')
    poly = np.column_stack([(D @ u) / dm, (D @ v) / dm])
    # drop repeated points (a zero-length edge adds nothing; the wedge code skips them anyway)
    keep = [0]
    for i in range(1, len(poly)):
        if np.linalg.norm(poly[i] - poly[keep[-1]]) > 1e-9:
            keep.append(i)
    if len(keep) > 1 and np.linalg.norm(poly[keep[-1]] - poly[keep[0]]) <= 1e-9:
        keep.pop()
    poly = poly[keep]
    if len(poly) < 3:
        raise ValueError('a cut outline needs at least 3 distinct points')
    node = ops.Node('outline_sweep', {'eye': [float(x) for x in eye_l], 'axis': [float(x) for x in m],
                                      'u': [float(x) for x in u], 'v': [float(x) for x in v], 'focal': 1.0,
                                      'points': [[float(a), float(b)] for a, b in poly],
                                      'extent': float(extent)}, nid=ids())
    n = len(poly)
    convex = all(((poly[(i + 1) % n] - poly[i])[0] * (poly[(i + 2) % n] - poly[i])[1] -
                  (poly[(i + 1) % n] - poly[i])[1] * (poly[(i + 2) % n] - poly[i])[0]) * np.sign(_area2(poly))
                 >= -1e-15 for i in range(n))
    return node, {'convex': bool(convex), 'points': n, 'kind': 'outline_sweep'}


def _area2(poly):
    x, y = poly[:, 0], poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _extent(obj):
    lo, hi = obj.local_box()
    return float(np.max(np.abs(np.concatenate([lo, hi])))) * 1.2 + 0.005


@command
class Cut(Command):
    """Cut objects oids with an outline (eye + ray directions, world). mode 'remove' | 'separate'."""
    label = 'Cut'

    def __init__(self, oids, eye, dirs, mode='remove', distance=0.02, kerf=0.0, blend=0.0, offset_dir=None):
        self.oids = [int(o) for o in oids]
        self.eye = [float(v) for v in eye]
        self.dirs = [[float(v) for v in d] for d in _decimate(dirs)]
        self.mode = mode
        self.distance = float(distance)
        self.kerf = float(kerf)
        self.blend = float(blend)
        self.offset_dir = None if offset_dir is None else [float(v) for v in offset_dir]

    def args(self):
        return {'oids': self.oids, 'eye': self.eye, 'dirs': self.dirs, 'mode': self.mode,
                'distance': self.distance, 'kerf': self.kerf, 'blend': self.blend, 'offset_dir': self.offset_dir}

    def do(self, scene):
        if self.mode not in ('remove', 'separate'):
            raise ValueError('mode must be remove or separate')
        self._prev_next = scene.next_id
        self._old = {}
        self._pieces = []
        ids = C.Ids(scene.next_id)
        made = []
        mean_dir = C.unit(np.mean(np.array([C.unit(d) for d in self.dirs]), axis=0))
        off = np.asarray(self.offset_dir, float) if self.offset_dir is not None else -mean_dir
        off = C.unit(off) * self.distance
        new_trees = {}
        for oid in self.oids:
            o = scene.get(oid)
            self._old[oid] = o.base.to_dict()
            eye_l = C.to_local(o, self.eye)
            dirs_l = [C.dir_to_local(o, d) for d in self.dirs]
            ext = _extent(o)
            cutter, info = build_cutter(eye_l, dirs_l, ext, ids)
            data = {'tool': 'cut', 'mode': self.mode, 'eye': C.r7(self.eye), 'kerf': self.kerf,
                    'info': info}
            if self.mode == 'remove':
                cut_node = cutter
                C.set_data(cut_node, data)
                new_trees[oid] = ops.Node('subtract', {'blend': self.blend}, [o.base, cut_node], nid=ids(),
                                          name='Cut')
                continue
            grow = ops.Node('offset', {'distance': 0.5 * self.kerf}, [cutter], nid=ids()) if self.kerf else cutter
            C.set_data(grow, data)
            new_trees[oid] = ops.Node('subtract', {'blend': self.blend}, [o.base, grow], nid=ids(), name='Cut')
            cutter2, _ = build_cutter(eye_l, dirs_l, ext, ids)
            shrink = (ops.Node('offset', {'distance': -0.5 * self.kerf}, [cutter2], nid=ids())
                      if self.kerf else cutter2)
            C.set_data(shrink, dict(data, part='piece'))
            piece_tree = ops.Node('intersect', {'blend': self.blend}, [C.renumber(o.base, ids), shrink],
                                  nid=ids(), name='Cut piece')
            made.append((o, piece_tree))
        # pieces: census first (on the unmoved piece), then create
        for o, piece_tree in made:
            if not _has_material(o, piece_tree):
                continue
            pid = ids()
            t = copy.deepcopy(o.transform)
            t['translate'] = [float(a + b) for a, b in zip(t['translate'], off)]
            p = SceneObject(pid, '%s piece' % o.name, piece_tree, transform=t, colour=copy.deepcopy(o.colour),
                            material=copy.deepcopy(o.material), h=o.sculpt.h)
            p.detail = copy.deepcopy(o.detail)
            slots = C.copy_sculpt(scene, o, p)
            self._pieces.append((p, slots))
        for oid, tree in new_trees.items():
            o = scene.get(oid)
            o.base = tree
            o.invalidate()
            o.compiled()
        for p, slots in self._pieces:
            scene.objects.append(p)
        scene.next_id = max(scene.next_id, ids.next_id)
        scene.touch(structural=True, index=True)
        return [p.id for p, s in self._pieces]

    def undo(self, scene):
        for p, slots in reversed(self._pieces):
            scene.objects.remove(p)
            C.release_sculpt(scene, p, set(slots))
        for oid, tree in self._old.items():
            o = scene.get(oid)
            o.base = ops.Node.from_dict(tree)
            o.invalidate()
        scene.next_id = self._prev_next
        self._pieces = []
        scene.touch(structural=True, index=True)


def _has_material(obj, tree, n=24):
    code, prm, lo, hi, L = ops.compile_tree(tree)
    ax = [np.linspace(lo[a], hi[a], n) for a in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    out = np.empty(len(g))
    C._eval_tree(code, prm, np.ascontiguousarray(g), out)
    return bool((out < 0).any())
