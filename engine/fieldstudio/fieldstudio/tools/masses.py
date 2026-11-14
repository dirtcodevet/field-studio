"""Masses (fs-shell, 2026-11-11; design 4.5 "Primitive block-in: Masses", chunk 6, on test 1's
path).

The yardstick (6.1): a head from a cranium ellipsoid, a jaw box, a neck capsule, a nose wedge and ear slabs in under a
minute, already smooth where they join. A mass is a primitive that stays live inside an object: a 'transform' node
(named 'mass <kind>') over the primitive, joined to the rest of the object's base by a smooth union (named 'mass
union') whose blend is the object's one blend dial. Masses are placed and shaped with the gizmo (its jobs become the
mass's transform; ui/gizmo.py), and the object's sculpt layer stays a delta on top, so masses can be moved after
sculpting. Every edit is one undoable, replayable command; the tree stays exact (min / smooth-min of distances, no
triangles: P14).
Tree shape: [object scale / gizmo deformers ...] -> mass union -> (mass union -> (... base ...), mass), mass) ..."""
import copy

from ..core import ops
from ..core.history import Command, command

ROOT_KEEP = ('object scale',)          # nodes kept above the masses (the axis stretch, the gizmo's deformers)


def _keep(node):
    nm = str(node.get('name', ''))
    return nm in ROOT_KEEP or nm.startswith('gizmo ')


def _split(tree):
    """-> (the chain of kept root nodes (outermost first), the subtree under them)"""
    keep = []
    node = tree
    while _keep(node) and node.get('children'):
        keep.append(node)
        node = node['children'][0]
    return keep, node


def _rejoin(keep, sub):
    if not keep:
        return sub
    out = copy.deepcopy(keep[0])
    cur = out
    for k in keep[1:]:
        k2 = copy.deepcopy(k)
        cur['children'] = [k2]
        cur = k2
    cur['children'] = [sub]
    return out


def mass_nodes(tree):
    """-> [mass transform node dicts] in the order they were added"""
    out = []
    stack = [tree]
    while stack:
        d = stack.pop()
        if str(d.get('name', '')).startswith('mass ') and d.get('kind') == 'transform':
            out.append(d)
        stack.extend(d.get('children', []))
    return sorted(out, key=lambda d: d['id'])


def mass_blend(tree):
    """-> the masses' blend (m) or None when the object has no masses"""
    stack = [tree]
    while stack:
        d = stack.pop()
        if d.get('name') == 'mass union':
            return float(d['params'].get('blend', 0.0))
        stack.extend(d.get('children', []))
    return None


class _TreeEdit(Command):
    """a base-tree edit computed on the engine thread from the tree as it stands (ids from the scene)"""

    def _apply(self, scene, tree, next_id=None):
        o = scene.get(self.oid)
        self.old = o.base.to_dict()
        self.prev_next = scene.next_id
        o.base = ops.Node.from_dict(tree)
        o.compiled()
        if next_id is not None:
            scene.next_id = max(scene.next_id, int(next_id))
        o.invalidate()
        scene.touch(structural=True)

    def undo(self, scene):
        o = scene.get(self.oid)
        o.base = ops.Node.from_dict(self.old)
        o.invalidate()
        scene.next_id = self.prev_next
        scene.touch(structural=True)


@command
class AddMass(_TreeEdit):
    """add a live mass to object oid: kind / params of the primitive; translate (object local, m), rotate (deg),
    scale [sx, sy, sz]; blend (m, the object's mass blend; None keeps it, default 10 mm for the first)"""
    label = 'Add mass'

    def __init__(self, oid, kind, params, translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
                 blend=None):
        self.oid = int(oid)
        self.kind = str(kind)
        self.params = copy.deepcopy(dict(params))
        self.translate = [float(v) for v in translate]
        self.rotate = [float(v) for v in rotate]
        self.scale = [float(v) for v in scale]
        self.blend = None if blend is None else float(blend)

    def args(self):
        return {'oid': self.oid, 'kind': self.kind, 'params': self.params, 'translate': self.translate,
                'rotate': self.rotate, 'scale': self.scale, 'blend': self.blend}

    def do(self, scene):
        o = scene.get(self.oid)
        tree = o.base.to_dict()
        keep, sub = _split(tree)
        k = self.blend if self.blend is not None else (mass_blend(tree) if mass_blend(tree) is not None else 0.01)
        nid = scene.next_id
        n = len(mass_nodes(tree)) + 1
        mass = {'id': nid, 'kind': 'transform', 'name': 'mass %s %d' % (self.kind, n),
                'params': {'translate': list(self.translate), 'rotate': list(self.rotate), 'scale': list(self.scale)},
                'children': [{'id': nid + 1, 'kind': self.kind, 'name': self.kind, 'params': copy.deepcopy(self.params),
                              'children': []}]}
        uni = {'id': nid + 2, 'kind': 'union', 'name': 'mass union', 'params': {'blend': float(k)},
               'children': [sub, mass]}
        self._apply(scene, _rejoin(keep, uni), nid + 3)
        self.mass_id = nid
        return {'select': [self.oid], 'node': nid}


@command
class SetMassBlend(_TreeEdit):
    """the object's blend dial: every mass union's blend (m)"""
    label = 'Mass blend'

    def __init__(self, oid, blend):
        self.oid = int(oid)
        self.blend = float(blend)

    def args(self):
        return {'oid': self.oid, 'blend': self.blend}

    def do(self, scene):
        tree = scene.get(self.oid).base.to_dict()
        stack = [tree]
        while stack:
            d = stack.pop()
            if d.get('name') == 'mass union':
                d['params']['blend'] = max(0.0, self.blend)
            stack.extend(d.get('children', []))
        self._apply(scene, tree)


@command
class RemoveMass(_TreeEdit):
    """take a mass out (its union collapses to the other side)"""
    label = 'Remove mass'

    def __init__(self, oid, mass_id):
        self.oid, self.mass_id = int(oid), int(mass_id)

    def args(self):
        return {'oid': self.oid, 'mass_id': self.mass_id}

    def do(self, scene):
        tree = scene.get(self.oid).base.to_dict()

        def walk(d):
            if d.get('name') == 'mass union' and len(d.get('children', [])) == 2:
                a, b = d['children']
                if b.get('id') == self.mass_id:
                    return walk(a)
                if a.get('id') == self.mass_id:
                    return walk(b)
            d = dict(d)
            d['children'] = [walk(c) for c in d.get('children', [])]
            return d
        self._apply(scene, walk(tree))


# ---- the joins: where two forms meet (fs-shell 11-11, test 1 with the kit: the masses' seams) -----------------------
def forms(tree):
    """-> [(name, tree)] each form of a massed object alone under the object's kept root nodes (its axis stretch and
    gizmo deformers): the base (what the masses were added to) first, then each mass"""
    keep, sub = _split(tree)
    out, node = [], sub
    ms = []
    while node.get('name') == 'mass union' and len(node.get('children', [])) == 2:
        ms.append(node['children'][1])
        node = node['children'][0]
    out.append(('base', _rejoin(keep, copy.deepcopy(node))))
    for m in reversed(ms):
        out.append((str(m.get('name', 'mass')), _rejoin(keep, copy.deepcopy(m))))
    return out


def _local(o, P):
    from .protected_merge import _to_local
    return _to_local(o, P)


def find_joins(scene, oid, step=0.002, radius=0.02, tol=None):
    """the lines where two of a massed object's forms meet on its surface, as stroke paths: each join is the surface
    points where the two nearest forms are equally near (f_i = f_j, both within the blend), ordered into polylines with
    points about radius / 2 apart and cut into runs whose surface normals stay within 45 deg of each other (each run can
    be stroked from one camera facing it). -> {'runs': [{'pair', 'pts' (n, 3), 'normal' (3,)}], 'n_surface', 'ms'}"""
    import time as _t
    import numpy as np
    from .protected_merge import field_of, world_box
    t0 = _t.perf_counter()
    o = scene.get(oid)
    tree = o.base.to_dict() if hasattr(o.base, 'to_dict') else o.base
    fs = forms(tree)
    if len(fs) < 2:
        return {'runs': [], 'n_surface': 0, 'ms': 0.0, 'forms': [n for n, _ in fs]}
    k = mass_blend(tree) or 0.01
    tol = 0.3 * k if tol is None else tol
    pk = scene.pack()
    lo, hi = world_box(o)
    ax = [np.arange(lo[a] - step, hi[a] + 2 * step, step) for a in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1)
    F = field_of(scene, pk, o, G.reshape(-1, 3)).reshape(G.shape[:3])
    pts = []
    for a in range(3):                          # sign changes along each axis: points on the surface
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a], s1[a] = slice(0, -1), slice(1, None)
        f0, f1 = F[tuple(s0)], F[tuple(s1)]
        m = np.isfinite(f0) & np.isfinite(f1) & ((f0 < 0) != (f1 < 0))
        if not m.any():
            continue
        t = (f0[m] / (f0[m] - f1[m]))[:, None]
        pts.append(G[tuple(s0)][m] + t * (G[tuple(s1)][m] - G[tuple(s0)][m]))
    if not pts:
        return {'runs': [], 'n_surface': 0, 'ms': (_t.perf_counter() - t0) * 1000, 'forms': [n for n, _ in fs]}
    P = np.vstack(pts)
    q = np.ascontiguousarray(_local(o, P))
    sc = float(np.mean(np.asarray(o.transform.get('scale', 1.0), float)))
    from .common import _eval_tree
    V = np.empty((len(P), len(fs)))
    for i, (_, t_) in enumerate(fs):
        code, prm, _lo, _hi, _L = ops.compile_tree(ops.Node.from_dict(t_))
        out = np.empty(len(q))
        _eval_tree(code, prm, q, out)
        V[:, i] = out * sc
    order = np.argsort(V, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    f1, f2 = V[np.arange(len(P)), i1], V[np.arange(len(P)), i2]
    on = (f2 - f1 < tol) & (f2 < k)
    runs = []
    cell = max(radius * 0.5, step)
    for a_, b_ in sorted({(int(min(x, y)), int(max(x, y))) for x, y in zip(i1[on], i2[on])}):
        sel = on & (((i1 == a_) & (i2 == b_)) | ((i1 == b_) & (i2 == a_)))
        Q, d = P[sel], (f2 - f1)[sel]
        if len(Q) < 3:
            continue
        # one point per cell of radius / 2: the one nearest the crease (f_i = f_j)
        key = np.floor(Q / cell).astype(np.int64)
        idx = {}
        for j, kk in enumerate(map(tuple, key)):
            if kk not in idx or d[j] < d[idx[kk]]:
                idx[kk] = j
        Q = Q[sorted(idx.values())]
        for line in _chain(Q, 1.9 * cell):
            if len(line) < 2:
                continue
            N = _normals(scene, pk, o, line)
            for run, nrm in _runs(line, N):
                runs.append({'pair': (fs[a_][0], fs[b_][0]), 'pts': run, 'normal': nrm})
    return {'runs': runs, 'n_surface': int(len(P)), 'n_join': int(on.sum()), 'blend': k,
            'ms': (_t.perf_counter() - t0) * 1000, 'forms': [n for n, _ in fs]}


def _chain(Q, link):
    """order points into polylines: start at an end (the point with the fewest neighbours), walk to the nearest
    unvisited point within link; a gap starts a new line"""
    import numpy as np
    left = set(range(len(Q)))
    D = np.linalg.norm(Q[:, None] - Q[None], axis=2) if len(Q) <= 4000 else None
    lines = []
    while left:
        L = np.array(sorted(left))
        if D is not None:
            nb = (D[np.ix_(L, L)] < link).sum(1)
        else:
            nb = np.zeros(len(L))
        cur = int(L[int(np.argmin(nb))])
        line = [cur]
        left.discard(cur)
        while left:
            L = np.array(sorted(left))
            dd = D[cur, L] if D is not None else np.linalg.norm(Q[L] - Q[cur], axis=1)
            j = int(np.argmin(dd))
            if dd[j] > link:
                break
            cur = int(L[j])
            line.append(cur)
            left.discard(cur)
        lines.append(Q[line])
    return lines


def _normals(scene, pk, o, P, h=2e-4):
    import numpy as np
    from .protected_merge import field_of
    g = np.empty_like(P)
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        g[:, a] = field_of(scene, pk, o, P + e) - field_of(scene, pk, o, P - e)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-12)


def _runs(line, N, max_deg=45.0):
    """cut a polyline where its normals turn more than max_deg from the run's mean"""
    import numpy as np
    c = np.cos(np.radians(max_deg))
    out, a = [], 0
    m = N[0].copy()
    for j in range(1, len(line)):
        mm = m / max(np.linalg.norm(m), 1e-12)
        if float(N[j] @ mm) < c:
            out.append((line[a:j], mm))
            a, m = j, N[j].copy()
        else:
            m += N[j]
    mm = m / max(np.linalg.norm(m), 1e-12)
    out.append((line[a:], mm))
    return out
