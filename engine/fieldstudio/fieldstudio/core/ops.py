"""Base-layer nodes: analytic primitives, booleans (sharp or smooth), mutations.

A base tree is a tree of Node objects. Every node keeps its parameters editable after creation; the tree is
recompiled into a program (program.py) on change. compile_tree returns the program, the node's bounding box in
its output space, and its tracked Lipschitz bound L (|F(p)| <= L * distance(p, zero set) wherever F is
evaluated; D = derived per node below).

Lipschitz per node (D):
  primitives: exact or conservative distance, L = 1 (ellipsoid: (|p/r| - 1) * r_min has |grad| <= 1)
  union / subtract / intersect, sharp or polynomial-smooth: L = max(L_a, L_b) (the smooth-min gradient is a
     convex combination of the operands' gradients)
  transform (translate, rotate, scale/stretch): value * min(scale), L = L_child
  twist (xz rotated by rate*y): L = L_child * (k R / 2 + sqrt(1 + (k R / 2)^2)), R = max xz radius of the guard box
  bend (xy rotated by rate*x): L = L_child * (1 + k rho), rho = max xy radius of the guard box
  taper (xz scaled by 1/g, g = 1 + rate*y >= g_min): L = L_child * (max(1, 1/g_min) + |rate| R / g_min^2)
  shell, offset: L = L_child
Deformations carry a guard: outside their box + margin they return the box distance + margin (a lower bound of
the true distance), so the bound only has to hold over the guarded region.
"""
import copy
import os
import math

import numpy as np

from . import program as P

PRIMITIVES = {
    'sphere': {'radius': 0.1},
    'box': {'size': [0.2, 0.2, 0.2], 'round': 0.0},
    'cylinder': {'radius': 0.08, 'height': 0.2, 'round': 0.0},
    'capsule': {'radius': 0.06, 'height': 0.2},
    'cone': {'radius_bottom': 0.1, 'radius_top': 0.0, 'height': 0.2},
    'torus': {'major': 0.1, 'minor': 0.03},
    'ellipsoid': {'radii': [0.12, 0.08, 0.06]},
    'slab': {'normal': [0.0, 1.0, 0.0], 'offset': 0.0, 'thickness': 0.02, 'extent': 0.2},
    # V1.2 field region: a crop of a brick-field version, sampled (core/region.py); L measured from its lattice
    'sampled': {'source': '', 'source_id': '', 'crop': 'box', 'centre': [0.0, 0.0, 0.0], 'size': [0.03, 0.03, 0.03],
                'anchor': [0.0, 0.0, 0.0], 'clamp': 0.006},
    # milestone (11-08): her whole FV8 store as a base, read natively at every level (core/fv8base.py)
    'fv8store': {'source': '', 'source_id': ''},
    # milestone: a dense field file (protected parts: core/masked_parts.py)
    'lattice': {'file': '', 'sha': ''},
    # cutsdf (11-08): a closed outline on the view plane m.(p - eye) = focal (points in (u, v) coords) swept along
    # the view rays from the eye, clipped to a cube of half size extent; exact signed distance, L = 1
    # (program._osweep). The CUT tool's cutter; mesh-free (no triangulation).
    'outline_sweep': {'eye': [0.0, 0.0, 1.0], 'axis': [0.0, 0.0, -1.0], 'u': [1.0, 0.0, 0.0], 'v': [0.0, 1.0, 0.0],
                      'focal': 1.0, 'points': [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
                      'extent': 0.2},
}
BOOLEANS = {'union': {'blend': 0.0}, 'subtract': {'blend': 0.0}, 'intersect': {'blend': 0.0}}
MUTATIONS = {
    'transform': {'translate': [0.0, 0.0, 0.0], 'rotate': [0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0]},
    'twist': {'rate': 5.0},          # rad per metre along y
    'bend': {'rate': 3.0},           # rad per metre along x
    'taper': {'rate': 2.0, 'g_min': 0.2},
    'shell': {'thickness': 0.01},    # hollow: |F| - thickness/2
    'offset': {'distance': 0.005},   # round / grow: F - distance
    'cull': {'pad': 0.0},            # hand2f: a box guard only (child box + margin + pad): merged parts far from a point
                                     # cost one box test (L unchanged; a lower bound of the distance outside the box).
                                     # pad = 2 x the blend it is joined with, so a smooth join never sees the guard
    # fs-materials 11-14: a material generator on its child (materials/fieldops.py): block = the resolved parameter
    # block (relief / chips / crack, program.OP_MATGEN); spec = the generator's own settings (seed, sliders) for the
    # panel. Relief moves the surface along its normal; chips and cracks remove material. L and box from fieldops.bounds
    'matgen': {'block': [0.0, 2.0], 'spec': {}},
}
KINDS = {**PRIMITIVES, **BOOLEANS, **MUTATIONS}


# hand2b: parameters a node takes only when given (absent from older trees, so their describe() and hash stay the same)
OPTIONAL = {'cull': ('union_k',),      # union_k >= 0: the exact union-operand guard (program.OP_GUARD_U)
            'transform': ('shear',)}  # 11-10: a skew (ops._compile 'transform'); absent = none, old trees unchanged


class Node:
    """One node of a base tree. kind in KINDS; params a dict (lists for vectors); children Nodes."""

    def __init__(self, kind, params=None, children=None, nid=0, name=None):
        if kind not in KINDS:
            raise ValueError('unknown node kind %r' % kind)
        p = copy.deepcopy(KINDS[kind])
        if params:
            for k, v in params.items():
                if k not in p and k not in OPTIONAL.get(kind, ()):
                    raise ValueError('node %s has no parameter %r' % (kind, k))
                p[k] = copy.deepcopy(v)
        self.kind = kind
        self.params = p
        self.children = list(children or [])
        self.id = int(nid)
        self.name = name or kind

    def to_dict(self):
        return {'id': self.id, 'kind': self.kind, 'name': self.name, 'params': copy.deepcopy(self.params),
                'children': [c.to_dict() for c in self.children]}

    @staticmethod
    def from_dict(d):
        return Node(d['kind'], d['params'], [Node.from_dict(c) for c in d.get('children', [])], d.get('id', 0),
                    d.get('name'))

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, nid):
        for n in self.walk():
            if n.id == nid:
                return n
        return None

    def parent_of(self, nid):
        for n in self.walk():
            for i, c in enumerate(n.children):
                if c.id == nid:
                    return n, i
        return None, -1


def rot_matrix(deg):
    """Euler XYZ (degrees) -> 3x3 rotation, R = Rz @ Ry @ Rx."""
    ax, ay, az = [math.radians(float(a)) for a in deg]
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _box_corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def _margin(lo, hi):
    return max(0.05 * float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))), 0.002)


class Compiled:
    def __init__(self):
        self.code = []     # rows [op, param_offset, jump]
        self.prm = []
        self.blobs = []    # (row, float64 array): sampled-field data appended after prm (offset patched in compile_tree)

    def emit(self, op, params=(), jump=0):
        self.code.append([op, len(self.prm), jump])
        self.prm.extend(float(v) for v in params)
        return len(self.code) - 1


def _prim(node, out):
    p = node.params
    k = node.kind
    if k == 'sphere':
        r = max(float(p['radius']), 1e-6)
        out.emit(P.OP_SPHERE, [r])
        return np.array([-r] * 3), np.array([r] * 3)
    if k == 'box':
        hs = np.maximum(np.asarray(p['size'], float) * 0.5, 1e-6)
        rr = min(max(float(p['round']), 0.0), float(hs.min()))
        out.emit(P.OP_BOX, [hs[0], hs[1], hs[2], rr])
        return -hs, hs
    if k == 'cylinder':
        r = max(float(p['radius']), 1e-6)
        hh = max(float(p['height']) * 0.5, 1e-6)
        rr = min(max(float(p['round']), 0.0), r, hh)
        out.emit(P.OP_CYL, [r, hh, rr])
        return np.array([-r, -hh, -r]), np.array([r, hh, r])
    if k == 'capsule':
        r = max(float(p['radius']), 1e-6)
        hh = max(float(p['height']) * 0.5, 0.0)
        out.emit(P.OP_CAPSULE, [hh, r])
        return np.array([-r, -hh - r, -r]), np.array([r, hh + r, r])
    if k == 'cone':
        r1 = max(float(p['radius_bottom']), 0.0)
        r2 = max(float(p['radius_top']), 0.0)
        if r1 + r2 < 1e-6:
            r1 = 1e-6
        hh = max(float(p['height']) * 0.5, 1e-6)
        rm = max(r1, r2)
        out.emit(P.OP_CONE, [r1, r2, hh])
        return np.array([-rm, -hh, -rm]), np.array([rm, hh, rm])
    if k == 'torus':
        R = max(float(p['major']), 1e-6)
        r = max(float(p['minor']), 1e-6)
        out.emit(P.OP_TORUS, [R, r])
        return np.array([-R - r, -r, -R - r]), np.array([R + r, r, R + r])
    if k == 'ellipsoid':
        rr = np.maximum(np.asarray(p['radii'], float), 1e-6)
        out.emit(P.OP_ELLIPSOID, list(rr))
        return -rr, rr
    if k == 'sampled':
        from . import region
        hdr, data, lo, hi, L = region.compile_params(p)
        row = out.emit(P.OP_SAMPLED, hdr)
        out.blobs.append((row, data))
        out.sampled_L = max(getattr(out, 'sampled_L', 1.0), L)
        node._L = L
        return lo, hi
    if k == 'lattice':
        # milestone (11-08 night): a dense field saved beside her store (protected parts): trilinear on its
        # own lattice inside its box, the box's distance outside (program OP_SAMPLED's layout, crop = the box)
        from . import fv8base
        f_ = p['file']
        if p.get('sha') and not os.path.exists(f_):
            # fs-shell 11-11: a baked field (tools/bakes.py) that moved with its project: found by its sha (a file that
            # is where it was recorded is read as before)
            from ..tools.bakes import resolve
            f_ = resolve(f_, p.get('sha'))
        L = fv8base.lattice_load(f_, p.get('sha') or None)
        lo = np.asarray(L['lo'], float)
        dims = np.asarray(L['dims'], int)
        hh = float(L['h'])
        hi = lo + (dims - 1) * hh
        c, sz = 0.5 * (lo + hi), 0.5 * (hi - lo) - 1e-9
        # 11-09: outside the box the field is the box distance PLUS the smallest field value on the box's boundary (any
        # path to the part's surface crosses the boundary, and is at least that value from there). r: the box distance
        # alone read ~0 at the box faces, which shadow / sky rays took for a surface: planar shadow slabs on her thigh
        D = np.asarray(L['data'], float).reshape(tuple(int(v) for v in dims))
        pad = max(0.0, float(min(D[0].min(), D[-1].min(), D[:, 0].min(), D[:, -1].min(), D[:, :, 0].min(), D[:, :, -1].min())))
        hdr = [lo[0], lo[1], lo[2], hh, dims[0], dims[1], dims[2], 0.0, c[0], c[1], c[2], sz[0], sz[1], sz[2], pad, 0.0]
        row = out.emit(P.OP_SAMPLED, hdr)
        out.blobs.append((row, L['data']))
        return lo, hi
    if k == 'fv8store':
        from . import fv8base
        S = fv8base.load(p['source'], p.get('source_id') or None)
        out.emit(P.OP_FV8, list(S['hdr']))            # r-ptr: the data is read in place (header slots 46-47)
        out.fv8_data = getattr(out, 'fv8_data', []) + [S['data']]   # kept alive by the compiled result
        return np.asarray(S['lo'], float), np.asarray(S['hi'], float)
    if k == 'slab':
        n = np.asarray(p['normal'], float)
        n = n / max(np.linalg.norm(n), 1e-12)
        e = max(float(p['extent']), 1e-6)
        out.emit(P.OP_SLAB, [n[0], n[1], n[2], float(p['offset']), max(float(p['thickness']), 1e-6), e])
        return np.array([-e] * 3), np.array([e] * 3)
    if k == 'outline_sweep':
        eye = np.asarray(p['eye'], float)
        m = np.asarray(p['axis'], float)
        m = m / max(np.linalg.norm(m), 1e-12)
        u = np.asarray(p['u'], float)
        u = u - (u @ m) * m
        u = u / max(np.linalg.norm(u), 1e-12)
        v = np.cross(m, u) if float(np.cross(m, u) @ np.asarray(p['v'], float)) >= 0 else -np.cross(m, u)
        f = max(float(p['focal']), 1e-9)
        pts = np.asarray(p['points'], float).reshape(-1, 2)
        if len(pts) < 3:
            raise ValueError('outline_sweep needs at least 3 points')
        rays = f * m[None, :] + pts[:, :1] * u[None, :] + pts[:, 1:2] * v[None, :]
        rays = rays / np.linalg.norm(rays, axis=1)[:, None]
        e = max(float(p['extent']), 1e-6)
        out.emit(P.OP_OSWEEP, list(eye) + list(u) + list(v) + list(m) + [f, e, float(len(pts))]
                 + list(pts.ravel()) + list(rays.ravel()))
        return np.array([-e] * 3), np.array([e] * 3)
    raise ValueError(k)


def _compile(node, out):
    """Emit node; return (lo, hi, L)."""
    k = node.kind
    if k in PRIMITIVES:
        lo, hi = _prim(node, out)
        return lo, hi, (getattr(node, '_L', 1.0) if k == 'sampled' else 1.0)
    if k in BOOLEANS:
        if len(node.children) != 2:
            raise ValueError('%s needs 2 children' % k)
        kb = max(float(node.params['blend']), 0.0)
        # r50: guards below a smooth boolean are padded by 2 x the largest enclosing blend (in their own units), so a
        # skipped part's box bound can never enter the blend (the pencil leak when a root's blend was edited live:
        # 1513 surface points moved up to 162 um; the pencil's own pad only knew the blend it was built with)
        k_prev = getattr(out, 'kenc', 0.0)
        out.kenc = max(k_prev, kb)
        la, ha, La = _compile(node.children[0], out)
        lb, hb, Lb = _compile(node.children[1], out)
        out.kenc = k_prev
        smooth = kb > 1e-9
        op = {'union': (P.OP_UNION, P.OP_SUNION), 'subtract': (P.OP_SUB, P.OP_SSUB),
              'intersect': (P.OP_INTER, P.OP_SINTER)}[k][1 if smooth else 0]
        out.emit(op, [kb] if smooth else [0.0])
        grow = 0.25 * kb if smooth else 0.0   # smooth-min lowers F by at most k/4
        if k == 'union':
            lo, hi = np.minimum(la, lb) - grow, np.maximum(ha, hb) + grow
        elif k == 'subtract':
            lo, hi = la, ha
        else:
            lo, hi = np.maximum(la, lb), np.minimum(ha, hb)
            hi = np.maximum(hi, lo)
        return lo, hi, max(La, Lb)
    # mutations: one child
    if len(node.children) != 1:
        raise ValueError('%s needs 1 child' % k)
    p = node.params
    child = node.children[0]
    if k in ('shell', 'offset'):
        lo, hi, L = _compile(child, out)
        if k == 'shell':
            t = max(float(p['thickness']), 1e-6) * 0.5
            out.emit(P.OP_SHELL, [t])
            return lo - t, hi + t, L
        d = float(p['distance'])
        out.emit(P.OP_OFFSET, [d])
        g = max(d, 0.0)
        return lo - g, hi + g, L
    if k == 'matgen':
        from ..materials import fieldops as _MF
        lo, hi, L = _compile(child, out)
        blk = [float(v) for v in p['block']]
        out.emit(P.OP_MATGEN, blk)
        L2, g = _MF.bounds(blk, L)
        return lo - g, hi + g, L2
    if k == 'cull':
        # hand2b: 'union_k' >= 0 (a merge's union operand after the first): the exact guard OP_GUARD_U (skip only
        # where min / smin_k with the value below it is exact); absent or < 0: the plain box guard (as hand2f)
        uk = float(p.get('union_k', -1.0))
        if uk >= 0.0:
            guard_row = out.emit(P.OP_GUARD_U, [0.0] * 8)
        else:
            guard_row = out.emit(P.OP_GUARD, [0.0] * 7)
        lo, hi, L = _compile(child, out)
        m = _margin(lo, hi) + max(float(p['pad']), 2.0 * getattr(out, 'kenc', 0.0))   # r50
        glo, ghi = lo - m, hi + m
        out.prm[out.code[guard_row][1]: out.code[guard_row][1] + 7] = [glo[0], glo[1], glo[2], ghi[0], ghi[1], ghi[2], m]
        if uk >= 0.0:
            out.prm[out.code[guard_row][1] + 7] = uk
        out.code[guard_row][2] = len(out.code)
        return lo, hi, L
    if k == 'transform':
        # hand2f: the scale keeps its sign (a mirror: scale [-1, 1, 1] reflects the child, Mirror-duplicate); the value
        # scale is min |s| (a reflection is an isometry). Positive scales compile to the same numbers as before.
        s0 = np.asarray(p['scale'], float)
        sa = np.maximum(np.abs(s0), 1e-6)
        s = np.where(s0 < 0, -sa, sa)
        R = rot_matrix(p['rotate'])
        t = np.asarray(p['translate'], float)
        sh = p.get('shear')
        if sh is not None and np.any(np.asarray(sh, float) != 0.0):
            # 11-10 (there was no intuitive way to skew): shear [k_xy, k_xz, k_yx, k_yz, k_zx, k_zy], k_ab moving
            # the child along a by k times its b coordinate; A = R diag(s) Sh, q = A^-1 (p - t), the value scaled by
            # A's smallest singular value (the field stays a lower bound of the distance)
            k = np.asarray(sh, float).reshape(6)
            Sh = np.array([[1.0, k[0], k[1]], [k[2], 1.0, k[3]], [k[4], k[5], 1.0]])
            A = R @ np.diag(s) @ Sh
            M = np.linalg.inv(A)
            smin = float(np.linalg.svd(A, compute_uv=False).min())
        else:
            A = R @ np.diag(s)
            M = np.diag(1.0 / s) @ R.T                    # q = M (p - t)
            smin = float(sa.min())
        row = out.emit(P.OP_XFORM, list(M.ravel()) + list(t) + [smin])
        k_prev = getattr(out, 'kenc', 0.0)
        out.kenc = k_prev / smin                          # r50: the enclosing blend in the child's units
        lo, hi, L = _compile(child, out)
        out.kenc = k_prev
        out.emit(P.OP_POP)
        c = _box_corners(lo, hi)
        w = (A @ c.T).T + t
        return w.min(0), w.max(0), L
    # deformations with a guard
    guard_row = out.emit(P.OP_GUARD, [0.0] * 7)
    push_row = out.emit({'twist': P.OP_TWIST, 'bend': P.OP_BEND, 'taper': P.OP_TAPER}[k], [0.0, 0.0])
    lo, hi, L = _compile(child, out)
    out.emit(P.OP_POP)
    end = len(out.code)
    kk = float(p['rate'])
    if k == 'twist':
        Rc = math.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[2]), abs(hi[2])))
        olo = np.array([-Rc, lo[1], -Rc])
        ohi = np.array([Rc, hi[1], Rc])
    elif k == 'bend':
        rc = math.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[1]), abs(hi[1])))
        olo = np.array([-rc, -rc, lo[2]])
        ohi = np.array([rc, rc, hi[2]])
    else:
        gmin_p = max(float(p['g_min']), 1e-3)
        gmax = max(1.0 + kk * lo[1], 1.0 + kk * hi[1], gmin_p)
        olo = np.array([min(lo[0] * gmax, lo[0]), lo[1], min(lo[2] * gmax, lo[2])])
        ohi = np.array([max(hi[0] * gmax, hi[0]), hi[1], max(hi[2] * gmax, hi[2])])
    m = _margin(olo, ohi) + 2.0 * getattr(out, 'kenc', 0.0)   # r50
    glo, ghi = olo - m, ohi + m
    if k == 'twist':
        R = math.hypot(max(abs(glo[0]), abs(ghi[0])), max(abs(glo[2]), abs(ghi[2])))
        a = abs(kk) * R * 0.5
        Lk = L * (a + math.sqrt(1.0 + a * a))
        extra = 0.0
    elif k == 'bend':
        rho = math.hypot(max(abs(glo[0]), abs(ghi[0])), max(abs(glo[1]), abs(ghi[1])))
        Lk = L * (1.0 + abs(kk) * rho)
        extra = 0.0
    else:
        gmin_p = max(float(p['g_min']), 1e-3)
        gmin = max(min(1.0 + kk * glo[1], 1.0 + kk * ghi[1]), gmin_p)
        R = math.hypot(max(abs(glo[0]), abs(ghi[0])), max(abs(glo[2]), abs(ghi[2])))
        Lk = L * (max(1.0, 1.0 / gmin) + abs(kk) * R / (gmin * gmin))
        extra = gmin_p
    out.prm[out.code[guard_row][1]: out.code[guard_row][1] + 7] = [glo[0], glo[1], glo[2], ghi[0], ghi[1], ghi[2], m]
    out.code[guard_row][2] = end
    out.prm[out.code[push_row][1]] = kk
    out.prm[out.code[push_row][1] + 1] = extra
    return olo, ohi, Lk


_BIG = {}     # milestone: compiled programs holding her store (2 GB), shared by exact tree (read-only), newest 2 kept


def compile_tree(root):
    """-> (code int32 (n,3), prm float64 (m,), lo (3,), hi (3,), L)."""
    big_key = None
    if any(n.kind == 'fv8store' for n in root.walk()):
        import json as _json
        big_key = _json.dumps(root.to_dict(), sort_keys=True)
        hit = _BIG.get(big_key)
        if hit is not None:
            return hit
    out = Compiled()
    lo, hi, L = _compile(root, out)
    code = np.array(out.code, dtype=np.int32).reshape(-1, 3)
    prm = np.array(out.prm if out.prm else [0.0], dtype=np.float64)
    if out.blobs:                                   # sampled fields: data after the params, offset relative to the row's
        n = prm.shape[0]
        total = n + sum(int(d.size) for _r, d in out.blobs)
        full = np.empty(total, np.float64)
        full[:n] = prm
        for row, data in out.blobs:
            o = out.code[row][1]
            full[o + 15] = float(n - o)
            if getattr(data, 'big', False):
                data.fill(full[n:n + data.size])    # her store: written in place (fv8base.LazyBlob)
            else:
                full[n:n + data.shape[0]] = data
            n += int(data.size)
        prm = full
    res = (code, prm, np.asarray(lo, float), np.asarray(hi, float), float(L))
    if big_key is not None:
        prm.setflags(write=False)
        while len(_BIG) >= 2:
            _BIG.pop(next(iter(_BIG)))
        _BIG[big_key] = res
    return res


# ---- tree editing helpers (pure: return new trees; commands make them undoable) -------------------------

def primitive(kind, nid, **params):
    return Node(kind, params, nid=nid)


def boolean(kind, a, b, nid, blend=0.0):
    return Node(kind, {'blend': blend}, [a, b], nid=nid)


def mutate(kind, child, nid, **params):
    return Node(kind, params, [child], nid=nid)
