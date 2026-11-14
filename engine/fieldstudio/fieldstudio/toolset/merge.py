"""Merge (Join), Group and Ungroup (TOOLSET_DESIGN 3.3, approved 11-14).

Design: two separate objects in contact become one seamless object on 'merge'. Unmerged, they keep their own
properties and ID and can each be edited individually. The seam: an exact join, with one easy setting, 'blend at
contact', for a taper that works like the softness settings (how smooth the connection's gradient is). Not
touching (Q4): Merge refuses and offers to group them, with an 'ungroup' option.

JoinObjects(oids, blend)
- refuses (NotTouching) unless the objects' touch graph is connected (each pair's contact read from their exact
  fields: toolset/contact.py); the window then offers Group;
- the join is EXACT: core/combine.CombineObjects with a sharp union (the sources' sculpt layers carried exactly);
- blend > 0: a fillet only where they touch, baked into the merged object's lattice. Its width FOLLOWS THE CONTACT:
      k(x) = blend * ramp((F_a(x) + F_b(x)) / (2 blend))      F_a + F_b = the local gap between the two surfaces
      F'   = smin_k(x)(F_a, F_b)  where it lowers the union, the union elsewhere (bit for bit)
  widest where the objects meet most closely (gap 0 or overlap: k = blend), tapering to nothing where the gap opens
  to twice the blend. Patent route (LIBRARIES_AND_PATENTS 6.1, Ansys US 10,083,264): a variable-width, range-controlled
  blend (Pasko / Gourmel), never a constant-radius rolling-ball fillet.
GroupObjects(oids) / UngroupObjects(oids): a group keeps every object's identity and editability; it selects, moves
and turns as one (the window's selection reads o.group).
"""
import copy
import math

import numpy as np

from ..core.history import Command, command
from ..core.combine import CombineObjects, _frame
from ..core import brush as BR
from . import common as C
from .contact import touch_graph, TOUCH_TOL


class NotTouching(ValueError):
    """Merge refused: the objects do not touch. offer = 'group'"""
    offer = 'group'


def _ramp(t):
    t = np.clip(t, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


@command
class GroupObjects(Command):
    label = 'Group'

    def __init__(self, oids):
        self.oids = [int(o) for o in oids]

    def args(self):
        return {'oids': self.oids}

    def do(self, scene):
        if len(self.oids) < 2:
            raise ValueError('Group needs two or more objects')
        objs = [scene.get(o) for o in self.oids]
        self.old = {o.id: getattr(o, 'group', None) for o in scene.objects}
        # one group: the selection plus every member of a group it touches (groups merge)
        olds = {g for g in (getattr(o, 'group', None) for o in objs) if g is not None}
        used = [g for g in (getattr(o, 'group', None) for o in scene.objects) if g is not None]
        gid = min(olds) if olds else (max(used) + 1 if used else 1)
        for o in scene.objects:
            if o.id in self.oids or (getattr(o, 'group', None) in olds and olds):
                o.group = gid
        scene.touch(structural=True)
        return gid

    def undo(self, scene):
        for o in scene.objects:
            if o.id in self.old:
                o.group = self.old[o.id]
        scene.touch(structural=True)

    def undo_state(self):
        st = {'old': {str(k): v for k, v in self.old.items()}}
        return lambda: (st, {})

    def restore_undo_state(self, scene, meta, arrays):
        self.old = {int(k): v for k, v in meta['old'].items()}


@command
class UngroupObjects(Command):
    label = 'Ungroup'

    def __init__(self, oids):
        self.oids = [int(o) for o in oids]

    def args(self):
        return {'oids': self.oids}

    def do(self, scene):
        gs = {getattr(scene.get(o), 'group', None) for o in self.oids} - {None}
        self.old = {o.id: o.group for o in scene.objects if getattr(o, 'group', None) in gs}
        for o in scene.objects:
            if o.id in self.old:
                o.group = None
        scene.touch(structural=True)
        return sorted(self.old)

    def undo(self, scene):
        for o in scene.objects:
            if o.id in self.old:
                o.group = self.old[o.id]
        scene.touch(structural=True)

    def undo_state(self):
        st = {'old': {str(k): v for k, v in self.old.items()}}
        return lambda: (st, {})

    def restore_undo_state(self, scene, meta, arrays):
        self.old = {int(k): v for k, v in meta['old'].items()}


def group_members(scene, oid):
    """the ids that select / move with oid (its group, or itself)"""
    g = getattr(scene.get(oid), 'group', None)
    if g is None:
        return [int(oid)]
    return [o.id for o in scene.objects if getattr(o, 'group', None) == g]


SEAL_BAND = 12.0          # fs-core's band (lattice steps): within it the field stays a true distance (02ab2cfa)


@command
class JoinObjects(Command):
    """Merge: an exact join of touching objects, with an optional blend at contact (m). Refuses objects that do not
    touch (NotTouching: offer Group)."""
    label = 'Merge'

    def __init__(self, oids, blend=0.0, tol=TOUCH_TOL, name=None, seal=False):
        self.oids = [int(o) for o in oids]
        self.blend = float(blend)
        self.seal = bool(seal)                 # new Merges seal the contact; logs written before replay as they were
        self.tol = float(tol)
        self.name = name
        self.combine = None
        self.rec = None
        self.info = {}
        self._plan = None

    def args(self):
        d = {'oids': self.oids, 'blend': self.blend, 'tol': self.tol}
        if self.name is not None:
            d['name'] = self.name
        if self.seal:
            d['seal'] = True
        return d

    def do(self, scene):
        C.refuse_protected(scene, self.oids, 'Merge')
        if self.combine is not None and self.rec is None:
            # redo: the same merged object back (CombineObjects' own redo), the same blend written again from the
            # plan read at the first do (deterministic: the same values)
            if (self.blend > 0 and self._plan is None) or (self.seal and getattr(self, '_seal', None) is None):
                # restored from the undo journal (a reopen): the plans read again from the sources, which the undo put
                # back exactly (the same values as the first do's)
                _, pairs, _ = touch_graph(scene, self.oids, self.tol)
                srcs_ = [scene.get(o) for o in self.oids]
                if self.blend > 0 and self._plan is None:
                    self._plan = self._blend_plan(scene, srcs_, pairs)
                self._seal = self._seal_plan(scene, srcs_, pairs) if self.seal else None
            M = self.combine.do(scene)
            self.rec = BR.StrokeRecord()
            self._seal_write(scene, M, self._seal)
            if self._plan is not None:
                self._blend_write(scene, M, self._plan)
            return M
        if len(self.oids) < 2 or len(set(self.oids)) != len(self.oids):
            raise ValueError('Merge needs two or more different objects')
        connected, pairs, vals = touch_graph(scene, self.oids, self.tol)
        gaps = {}
        for k, v in vals.items():
            if np.isfinite(v):
                gaps['%d-%d' % k] = round(2e3 * v, 3)
            else:                                   # far apart: the nearest pair's gap (the refusal says how far)
                from .contact import nearest_pair
                gaps['%d-%d' % k] = round(1e3 * nearest_pair(scene, scene.get(k[0]), scene.get(k[1]))[0], 1)
        self.info = {'pairs': [(a, b, round(m * 1e3, 4)) for a, b, m in pairs], 'gaps_mm': gaps}
        if not connected:
            raise NotTouching('Not touching: Group instead? (gaps %s mm)' % ', '.join(
                '%s' % v for v in self.info['gaps_mm'].values()))
        srcs = [scene.get(o) for o in self.oids]
        # the blend's nodes, read from the SOURCES before they leave the scene (M's frame and lattice as CombineObjects
        # makes them: the first source's frame, the finest sculpted world spacing)
        plan = self._blend_plan(scene, srcs, pairs) if self.blend > 0 else None
        self._plan = plan
        self._seal = self._seal_plan(scene, srcs, pairs) if self.seal else None
        self.combine = CombineObjects(self.oids, 'union', 0.0, 'first', name=self.name)
        M = self.combine.do(scene)
        self.rec = BR.StrokeRecord()
        self.info.update(self._seal_write(scene, M, self._seal))
        if plan is not None:
            self.info.update(self._blend_write(scene, M, plan))
        self.info['bricks'] = len(M.sculpt.slots)
        return M

    # ---- the seal: the contact made solid ---------------------------------------------------------------------------
    # Merge makes one seamless object. The exact union min(fa, fb) of two pieces that only touch keeps each
    # piece's own face at the contact: inside the merged solid F climbs to 0 on the old contact face (a box cut in two and
    # merged back read -0.2 mm at the cut where the body is -2 mm under its top and -20 mm in its middle: fs-tools
    # u3/seam), an internal crack the tracer draws where it meets the outside (fs-shell-4's t66 seam). At an inside node
    # the union's distance is the deciding piece's own (|min(fa, fb)|) exactly when that piece's nearest surface point is
    # a REAL face (outside the other piece); otherwise the node is deeper than both read. Those nodes, and the thin air
    # between faces that touch within the tolerance, are set to the band's deep value, and the release re-distance
    # (fs-core's) rebuilds true distances from the real surface alone.
    def _seal_plan(self, scene, srcs, pairs):
        from scipy.spatial import cKDTree
        first = srcs[0]
        RM, tM, sM = _frame(first.transform)
        hw = [o.sculpt.h * _frame(o.transform)[2] for o in srcs]
        hws = [x for o, x in zip(srcs, hw) if o.sculpt.slots] or hw
        hM = min(hws) / sM
        hW = hM * sM
        band = SEAL_BAND * hW
        tolc = max(2.0 * self.tol, 0.25 * hW)
        pk = scene.pack()
        from ..warp.bake import object_field
        fields = {o.id: object_field(scene, o, pk) for o in srcs}
        byid = {o.id: o for o in srcs}
        nodes = {}
        for a, b, _ in pairs:
            la, ha = C.world_box(byid[a])
            lb, hb = C.world_box(byid[b])
            lo = np.maximum(la, lb) - band - 2.0 * hW
            hi = np.minimum(ha, hb) + band + 2.0 * hW
            if np.any(lo > hi):
                continue
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            L = ((c - tM) @ RM) / sM
            keys = C.bricks_in_box(L.min(0), L.max(0), hM, pad=0)
            if not len(keys):
                continue
            # the bricks' nodes as one dense grid (M's lattice, local), the two pieces' fields on it (world units)
            k0 = keys.min(0)
            nb = keys.max(0) - k0 + 1
            n = nb * C.B
            ax = [(k0[i] * C.B + np.arange(n[i])) * hM for i in range(3)]
            X = np.stack(np.meshgrid(*ax, indexing='ij'), -1)
            W = (X.reshape(-1, 3) * sM) @ RM.T + tM
            fa = fields[a](W).reshape(tuple(n))
            fb = fields[b](W).reshape(tuple(n))
            # the REAL surface: each piece's zero crossings on the grid's edges, sub-cell by linear interpolation (as
            # redistance_exact), less the contact faces: a crossing within tolc of the other piece whose face there faces
            # it (opposite normals). A rim point (the top face beside a cut) also lies within tolc of the other piece, but
            # its face is across the other's: keeping only "outside the other by tolc" dropped the rim on both sides and
            # raised a 0.5 mm ridge along the old cut (fs-tools u3/seam)
            ga = np.stack(np.gradient(fa), -1)
            gb = np.stack(np.gradient(fb), -1)
            pts = []
            for F1, F2, G1, G2 in ((fa, fb, ga, gb), (fb, fa, gb, ga)):
                for ax_ in range(3):
                    s0 = [slice(None)] * 3
                    s1 = [slice(None)] * 3
                    s0[ax_] = slice(0, -1)
                    s1[ax_] = slice(1, None)
                    f0, f1 = F1[tuple(s0)], F1[tuple(s1)]
                    m = (f0 <= 0) != (f1 <= 0)
                    if not m.any():
                        continue
                    t = f0[m] / (f0[m] - f1[m])
                    g0, g1 = F2[tuple(s0)][m], F2[tuple(s1)][m]
                    n1 = G1[tuple(s0)][m] + t[:, None] * (G1[tuple(s1)][m] - G1[tuple(s0)][m])
                    n2 = G2[tuple(s0)][m] + t[:, None] * (G2[tuple(s1)][m] - G2[tuple(s0)][m])
                    cs = (n1 * n2).sum(1) / np.maximum(np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1), 1e-30)
                    keep = ((g0 + t * (g1 - g0)) > tolc) | (cs > -0.5)
                    idx = np.argwhere(m)[keep].astype(float)
                    idx[:, ax_] += t[keep]
                    pts.append(idx)
            if not pts or not sum(len(p_) for p_ in pts):
                continue
            P = (np.concatenate(pts) + k0 * C.B) * hM
            fm = np.minimum(fa, fb)
            gap = (fa > 0) & (fb > 0) & (fa + fb <= 2.0 * self.tol)          # the air between faces that touch
            inside = (fm <= 0) | gap
            cand = inside & (fm > -band)
            if not cand.any():
                continue
            # only the nodes the contact decides: the deciding piece's nearest surface point lies inside the other piece,
            # or on a contact face (within tolc of the other, the two faces facing); elsewhere the union's value is
            # already the distance to a real face and stays bit for bit (the k-d tree's sampled surface would move it by
            # its sampling error: 0.057 mm on intersecting spheres' far surface, fs-tools m3)
            from scipy.ndimage import map_coordinates
            ci = np.argwhere(cand)
            fa_c, fb_c = fa[cand], fb[cand]
            usea = fa_c <= fb_c
            fs_ = np.where(usea, fa_c, fb_c)
            gs = np.where(usea[:, None], ga[cand], gb[cand])
            gs = gs / np.maximum(np.linalg.norm(gs, axis=1), 1e-30)[:, None]
            Pg = ci - (fs_ / hW)[:, None] * gs                                 # grid coordinates of the nearest point
            fo = np.where(usea, map_coordinates(fb, Pg.T, order=1, mode='nearest'),
                          map_coordinates(fa, Pg.T, order=1, mode='nearest'))
            go = np.stack([np.where(usea, map_coordinates(gb[..., k], Pg.T, order=1, mode='nearest'),
                                    map_coordinates(ga[..., k], Pg.T, order=1, mode='nearest')) for k in range(3)], 1)
            cs = (gs * go).sum(1) / np.maximum(np.linalg.norm(go, axis=1), 1e-30)
            decided = (fo < -tolc) | ((fo <= tolc) & (cs < -0.5)) | gap[cand]
            # the distance to the real surface: to its nearest sample (exact on a face's flat run; at a sharp concave
            # crease of overlapping pieces with no blend the crease moves by up to the sampling error, 0.09 mm at h 0.5 mm:
            # m3. The tangent planes of the nearest samples did worse at a cut's rim, where the normals tilt: -1.84 for -2)
            d, _ = cKDTree(P).query(X[cand][decided], workers=-1)
            true = np.full(len(fa_c), np.inf)
            true[decided] = -np.minimum(d * sM, band)
            deeper = true < fm[cand] - 1e-6 * hW
            if not deeper.any():
                continue
            V = np.full(fm.shape, np.nan)
            cw = ci[deeper]
            V[cw[:, 0], cw[:, 1], cw[:, 2]] = true[deeper] / sM
            Vb = V.reshape(nb[0], C.B, nb[1], C.B, nb[2], C.B).transpose(0, 2, 4, 1, 3, 5).reshape(-1, C.B ** 3)
            for q, kk in enumerate(np.stack(np.meshgrid(*[k0[i] + np.arange(nb[i]) for i in range(3)], indexing='ij'),
                                            -1).reshape(-1, 3)):
                if np.isnan(Vb[q]).all():
                    continue
                kk = tuple(int(v) for v in kk)
                nodes[kk] = np.fmin(nodes[kk], Vb[q]) if kk in nodes else Vb[q]
        return {'nodes': nodes, 'hM': hM}

    def _seal_write(self, scene, M, plan):
        if not plan or not plan['nodes']:
            return {'seal_bricks': 0}
        keys = sorted(plan['nodes'])
        X = C.brick_nodes(keys, M.sculpt.h).reshape(-1, 3)
        Fm = C.field_local(scene, M, X)
        Vs = np.concatenate([plan['nodes'][k] for k in keys])
        Fn = np.where(np.isnan(Vs), Fm, np.minimum(Fm, Vs))
        st = scene.store
        g = M.sculpt
        cur = []
        for k in keys:
            sl = g.get_slot(k)
            cur.append(st.delta[sl].astype(np.float64) if sl >= 0 else np.zeros((8, 8, 8)))
        cur = np.stack(cur)
        nd = cur + (Fn - Fm).reshape(cur.shape)
        changed = np.abs(Fn - Fm).reshape(len(keys), -1).max(1) > 0
        sel = [i for i in range(len(keys)) if changed[i]]
        done = C.write_bricks(scene, M, [keys[i] for i in sel], nd[sel], self.rec)
        # the values are distances to the real surface already: the bricks' bounds re-measured, no re-distance
        C.finish_bricks(scene, M, done, self.rec, redistance=False)
        return {'seal_bricks': len(done), 'seal_nodes': int((~np.isnan(Vs)).sum())}

    # ---- the contact-following blend --------------------------------------------------------------------------------
    def _blend_plan(self, scene, srcs, pairs):
        first = srcs[0]
        RM, tM, sM = _frame(first.transform)
        hw = [o.sculpt.h * _frame(o.transform)[2] for o in srcs]
        hws = [x for o, x in zip(srcs, hw) if o.sculpt.slots] or hw
        hM = min(hws) / sM
        K = self.blend
        G = 2.0 * K
        pk = scene.pack()
        from ..warp.bake import object_field
        fields = {o.id: object_field(scene, o, pk) for o in srcs}
        byid = {o.id: o for o in srcs}
        nodes = {}
        for a, b, _ in pairs:
            la, ha = C.world_box(byid[a])
            lb, hb = C.world_box(byid[b])
            lo = np.maximum(la, lb) - 2.0 * G
            hi = np.minimum(ha, hb) + 2.0 * G
            if np.any(lo > hi):
                continue
            # bricks of M's lattice (local = first's frame, spacing hM) over the pair's contact box
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            L = ((c - tM) @ RM) / sM
            keys = C.bricks_in_box(L.min(0), L.max(0), hM, pad=1)
            # first the bricks' centres: only bricks where the blend can act (both fields within reach of each other and
            # of the contact) are read in full (Lipschitz 1 fields: a node is within half a brick diagonal of its centre)
            Xc = (keys + 0.5) * C.B * hM
            Wc = (Xc * sM) @ RM.T + tM
            fac, fbc = fields[a](Wc), fields[b](Wc)
            r_b = 0.5 * math.sqrt(3.0) * C.B * hM * sM
            act = (np.abs(fac - fbc) < K + 2 * r_b) & (np.maximum(fac + fbc, 0.0) < G + 2 * r_b)
            keys = keys[act]
            if not len(keys):
                continue
            X = C.brick_nodes(keys, hM).reshape(-1, 3)
            W = (X * sM) @ RM.T + tM
            fa = fields[a](W)
            fb = fields[b](W)
            k = K * _ramp(np.maximum(fa + fb, 0.0) / G)
            with np.errstate(divide='ignore', invalid='ignore'):
                hh = np.where(k > 0, np.maximum(k - np.abs(fa - fb), 0.0) / np.where(k > 0, k, 1.0), 0.0)
            fs = np.minimum(fa, fb) - hh * hh * k * 0.25
            low = (np.minimum(fa, fb) - fs).reshape(len(keys), -1)
            keep = low.max(1) > 1e-9
            for q in np.nonzero(keep)[0]:
                kk = tuple(int(v) for v in keys[q])
                v = fs.reshape(len(keys), -1)[q] / sM
                nodes[kk] = np.minimum(nodes[kk], v) if kk in nodes else v
        return {'nodes': nodes, 'hM': hM}

    def _blend_write(self, scene, M, plan):
        if not plan['nodes']:
            return {'blend_bricks': 0}
        keys = sorted(plan['nodes'])
        X = C.brick_nodes(keys, M.sculpt.h).reshape(-1, 3)
        Fm = C.field_local(scene, M, X)                     # the exact union, as merged
        Fb = C.field_local(scene, M, X, base_only=True)
        Fs = np.concatenate([plan['nodes'][k] for k in keys])
        Fn = np.minimum(Fm, Fs)
        st = scene.store
        g = M.sculpt
        cur = []
        for k in keys:
            sl = g.get_slot(k)
            cur.append(st.delta[sl].astype(np.float64) if sl >= 0 else np.zeros((8, 8, 8)))
        cur = np.stack(cur)
        nd = cur + (Fn - Fm).reshape(cur.shape)
        changed = np.abs(Fn - Fm).reshape(len(keys), -1).max(1) > 0
        sel = [i for i in range(len(keys)) if changed[i]]
        done = C.write_bricks(scene, M, [keys[i] for i in sel], nd[sel], self.rec)
        info = C.finish_bricks(scene, M, done, self.rec, redistance=True)
        return {'blend_bricks': len(done), 'blend_mm': round(self.blend * 1e3, 3),
                'max_fill_mm': round(float((Fm - Fn).max()) * _frame(M.transform)[2] * 1e3, 3), **info}

    def undo(self, scene):
        if self.rec is not None:
            BR.undo_record(scene, self.rec)
            self.rec = None
        self.combine.undo(scene)

    def discard(self, scene, state):
        if self.combine is not None:
            self.combine.discard(scene, state)

    # ---- the undo journal (fs-tools-2 11-14): the merge's own (CombineObjects: the sources in full)
    # and the blend's brick record, so a save ending with a Merge carries the whole history and a reopen needs no replay
    def undo_state(self):
        if self.combine is None or self.rec is None:
            return None
        mk = self.combine.undo_state()
        if mk is None:
            return None
        rec = self.rec
        from ..core import undojournal as UJ

        def make():
            meta, A = mk()
            UJ.record_arrays('jrec', rec, A)
            return {'kind': 'JoinObjects', 'combine': meta}, A
        return make

    def restore_undo_state(self, scene, meta, A):
        from ..core import undojournal as UJ
        self.combine = CombineObjects(self.oids, 'union', 0.0, 'first', name=self.name)
        self.combine.restore_undo_state(scene, meta['combine'], A)
        self.rec = UJ.record_from('jrec', A)

    def evict_undo(self):
        return False

    def nbytes(self):
        n = self.combine.nbytes() if self.combine is not None else 256
        return n + (self.rec.nbytes() if self.rec is not None else 0)
