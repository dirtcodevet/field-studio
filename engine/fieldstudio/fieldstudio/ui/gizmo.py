"""The transform gizmo (fs-shell, 2026-11-11; design 4.5 "Gizmo deformers", chunk 6).

The handles on the selected object, in its own axes, as Blender's and ZBrush's transform gizmos (moved here from
ui/viewport.py, where they were added 11-10: the primitive shapes lacked an intuitive
way to stretch, shape, sculpt form or skew). Move: an arrow per axis. Scale: a box per axis stretches that axis
(Shift: skews it toward the other axis the drag runs along); the centre disc scales evenly. Rotate: a ring per axis.
Away from the handles the drag works as before (move in the view plane, turn, scale evenly).

Added for chunk 6 (design 4.5):
- Deformers on the gizmo: a small ring of icons at its base (bend, twist, taper, inflate); a drag left / right dials
  the amount live; one drag is one log step (the engine's 'deform' job).
- Numeric entry: typing during a drag sets the value exactly, as Blender's G X 0.64; Backspace edits, Enter ends the
  drag with that value, Esc puts the object back. Units (design D1): a length in any unit ("34 in", "1.5 m", "5' 6\"",
  a bare number in the scene's unit); Rotate: degrees; Scale: a factor, or a length / size that sets the object's real
  extent ("120 mm" along the locked axis, "2x4x8ft" for a box: lumber from the stock list). X / Y / Z lock an axis
  before typing; once a value has begun every key is part of it.
- Axis lock: X / Y / Z during a drag lock the move / turn / stretch to the object's own axis (again: free).
- Snapping: Ctrl held during a drag: 1 mm steps (move), 5 deg (rotate), 5 % (scale) (design 4.5). The Scale handles'
  skew moved from Ctrl to Shift for this.
- Pivot: the object's centre, or a picked surface point (Pivot > Pick a surface point, then a click on the form);
  rotations and even scaling turn and grow about it.
The viewport keeps the mouse; it calls begin / job / end / draw / hit here. Jobs are the engine's own tuples
('props', oid, {...}, drag), ('oscale', oid, [sx, sy, sz], drag[, shear]), ('deform', oid, kind, params, drag)."""
import math

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen

AXIS_COL = (QColor(235, 80, 80), QColor(110, 210, 90), QColor(80, 140, 245))
HANDLE_PX = 90
# the deformer ring: kind, label, the engine's mutation and its amount parameter, units per screen pixel, unit text
DEFORMERS = [('bend', 'Bend', 'bend', 'rate', 0.05, 'rad/m'), ('twist', 'Twist', 'twist', 'rate', 0.1, 'rad/m'),
             ('taper', 'Taper', 'taper', 'rate', 0.03, '/m'), ('inflate', 'Inflate', 'offset', 'distance', 0.0001, 'm')]
SNAP_MM, SNAP_DEG, SNAP_SCALE = 1.0, 5.0, 0.05      # SNAP_MM: the default; the scene's unit and grid set the step


def _rodrigues(ax, ang):
    ax = np.asarray(ax, float)
    ax = ax / max(np.linalg.norm(ax), 1e-12)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _snap(v, step):
    return round(v / step) * step


class Gizmo:
    """one per viewport; state of the current gizmo drag (vp.xform holds the press state as before)"""

    def __init__(self, vp):
        self.vp = vp
        self.pivot_mode = 'centre'        # 'centre' | 'point'
        self.pivot_point = None           # world point (m) when pivot_mode == 'point'
        self.picking_pivot = False        # the next left click on the form sets the pivot point
        self.typed = ''                   # numeric entry during a drag
        self.lock = None                  # axis index locked by X / Y / Z during a drag
        self.last_value = None            # (what, value, unit) of the drag so far, for the readout
        self.deform_hover = None

    # ---- geometry ---------------------------------------------------------------------------------------------------
    def active(self):
        vp = self.vp
        return vp.tool in ('move', 'rotate', 'scale') and bool(vp.selected) and vp.drag in (None, 'xform')

    def geo(self):
        vp = self.vp
        if not self.active():
            return None
        x0 = vp.xform
        if x0 is not None and x0.get('geo') is not None:
            return x0['geo']
        oid = vp.selected[0]
        tr = vp.transforms.get(oid)
        if tr is None:
            return None
        from ..core.ops import rot_matrix
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        c = np.asarray(tr['translate'], float)
        R = rot_matrix(tr['rotate'])
        mc = self.mass_ctx()
        if mc is not None:                      # design 4.5: the gizmo on the selected mass, in its own axes
            c, R = mc['c'], mc['R']
        else:
            rc = self.region_ctx(oid)           # fs-shell-3 11-14: a masked object: the gizmo at the free part's pivot
            if rc is not None:
                c = np.asarray(rc['pivot'], float)
        sx, sy, z = vp.cam.project(c, W, H)
        if z <= 1e-6:
            return None
        k = HANDLE_PX * 2.0 * z * math.tan(math.radians(vp.cam.fov) * 0.5) / H   # world length of a handle
        axes = []
        for i in range(3):
            ex, ey, _ = vp.cam.project(c + R[:, i] * k, W, H)
            d = np.array([ex - sx, ey - sy])
            L = float(np.linalg.norm(d))
            axes.append((d / L if L > 1e-6 else np.zeros(2), L))
        rings = []
        if vp.tool == 'rotate':
            for i in range(3):
                j, l = (i + 1) % 3, (i + 2) % 3
                pts = [vp.cam.project(c + 0.85 * k * (math.cos(a) * R[:, j] + math.sin(a) * R[:, l]), W, H)[:2]
                       for a in np.linspace(0.0, 2 * math.pi, 49)]
                rings.append(np.array(pts))
        # the deformer ring: four icons in an arc below the gizmo's centre
        dfs = []
        for n, d in enumerate(DEFORMERS):
            a = math.radians(90 + (n - 1.5) * 26)
            dfs.append((d[0], np.array([sx + math.cos(a) * (HANDLE_PX + 26), sy + math.sin(a) * (HANDLE_PX + 26)])))
        return {'oid': oid, 'c': c, 'R': R, 'k': k, 'sc': np.array([sx, sy]), 'axes': axes, 'rings': rings,
                'deformers': dfs}

    def hit(self, pos):
        g = self.geo()
        if g is None:
            return None
        q = np.array([pos.x(), pos.y()])
        for kind, cpt in g['deformers']:
            if np.linalg.norm(q - cpt) <= 12:
                return ('deform', kind)
        if np.linalg.norm(q - g['sc']) <= 10:
            return ('centre', -1)
        vp = self.vp
        if vp.tool == 'rotate':
            best = None
            f = vp.cam.basis()[0]
            for i, ring in enumerate(g['rings']):
                d = float(np.min(np.linalg.norm(ring - q, axis=1)))
                score = d + 4.0 * (1.0 - abs(float(g['R'][:, i] @ f)))     # where rings cross: the one facing you
                if d <= 7 and (best is None or score < best[0]):
                    best = (score, i)
            return ('ring', best[1]) if best else None
        best = None
        for i, (dv, L) in enumerate(g['axes']):
            if L < 8:
                continue
            v = q - g['sc']
            t = float(v @ dv)
            if not (0.2 * L <= t <= L + 9):
                continue
            d = float(np.linalg.norm(v - t * dv))
            if d <= 8 and (best is None or d < best[0]):
                best = (d, i)
        return ('axis', best[1]) if best else None

    # ---- the drag ---------------------------------------------------------------------------------------------------
    # ---- masses (design 4.5): the gizmo moves, turns and stretches a mass like a small object ---------------------
    def _snap_obj(self, oid):
        try:
            win = self.vp.window()
        except Exception:
            return None
        return next((o for o in (getattr(win, 'snap', None) or {}).get('objects', []) if o.get('id') == oid), None)

    def mass_ctx(self, oid=None, nid=None):
        """-> the selected mass's frame {oid, nid, node, c (world), R (world), S, To (object R, t, s)} or None"""
        from ..core.ops import rot_matrix
        vp = self.vp
        if oid is None:
            if not vp.selected:
                return None
            oid = vp.selected[0]
        if nid is None:
            try:
                props = vp.window().props
                nid = props.node_id if props.order and props.order[0] == oid else None
            except Exception:
                nid = None
        if nid is None:
            return None
        o = self._snap_obj(oid)
        if o is None:
            return None
        node = None
        stack = [o['base']]
        while stack:
            d = stack.pop()
            if d['id'] == nid:
                node = d
                break
            stack.extend(d.get('children', []))
        if node is None or node.get('kind') != 'transform' or not str(node.get('name', '')).startswith('mass '):
            return None
        tr = o['transform']
        Ro, to, so = rot_matrix(tr['rotate']), np.asarray(tr['translate'], float), float(tr['scale'])
        p = node['params']
        Rn = rot_matrix(p.get('rotate', [0.0, 0.0, 0.0]))
        c = to + Ro @ (np.asarray(p.get('translate', [0.0, 0.0, 0.0]), float) * so)
        return {'oid': oid, 'nid': nid, 'node': node, 'c': c, 'R': Ro @ Rn, 'S': list(p.get('scale', [1.0, 1.0, 1.0])),
                'To': (Ro, to, so)}

    def mass_at(self, pos):
        """the mass of the selected object under the screen point (the one whose surface is nearest the hit), or None"""
        vp = self.vp
        if not vp.selected:
            return None
        oid = vp.selected[0]
        o = self._snap_obj(oid)
        if o is None:
            return None
        from ..tools.masses import mass_nodes
        ms = mass_nodes(o['base'])
        if not ms:
            return None
        fr = getattr(vp, 'frame', None)
        if fr is None or fr.get('depth') is None:
            return None
        obj = fr['obj']
        h, w = obj.shape
        ix = int(pos.x() * w / max(vp.width(), 1))
        iy = int(pos.y() * h / max(vp.height(), 1))
        if not (0 <= ix < w and 0 <= iy < h) or obj[iy, ix] != oid:
            return None
        cam = fr.get('cam') or vp.cam
        org, dvec = cam.ray(ix, iy, w, h)
        P = np.asarray(org, float) + np.asarray(dvec, float) * float(fr['depth'][iy, ix])
        from ..core import ops
        from ..core.ops import rot_matrix
        from ..tools.common import _eval_tree
        tr = o['transform']
        Ro, to, so = rot_matrix(tr['rotate']), np.asarray(tr['translate'], float), float(tr['scale'])
        q = ((P - to) @ Ro) / so
        best = None
        for m in ms:
            try:
                code, prm, lo, hi, L = ops.compile_tree(ops.Node.from_dict(m))
                out = np.empty(1)
                _eval_tree(code, prm, np.ascontiguousarray(q[None]), out)
                f = abs(float(out[0])) * so
            except Exception:
                continue
            if best is None or f < best[0]:
                best = (f, m['id'])
        # a mass only when its surface is near the hit (a blend can round the join: 1.5 x the blend + 2 mm)
        from ..tools.masses import mass_blend
        tol = 1.5 * float(mass_blend(o['base']) or 0.0) * so + 0.002
        return best[1] if best is not None and best[0] <= tol else None

    def begin(self, x, pos):
        """called by the viewport's begin_xform with its press state x: adds the handle hit and its geometry"""
        self.typed, self.lock, self.last_value = '', None, None
        hit = self.hit(pos) if (self.vp.selected and x['oid'] == self.vp.selected[0]) else None
        # design 4.5: a press on a mass (not on a handle) takes that mass; the panel shows it selected
        if hit is None and x['oid'] == (self.vp.selected[0] if self.vp.selected else None):
            nid = self.mass_at(pos)
            if nid is not None:
                try:
                    props = self.vp.window().props
                    props.node_id = nid
                    props.refresh()
                except Exception:
                    pass
        mc = self.mass_ctx(x['oid'])
        if mc is not None:
            from .engine import euler_from_matrix
            from ..core import ops
            x['mass'] = mc
            x['t0'] = {'translate': [float(v) for v in mc['c']], 'rotate': euler_from_matrix(mc['R']), 'scale': 1.0}
            x['stretch0'] = list(mc['S'])
            x['shear0'] = None
            try:
                _c, _p, lo, hi, _L = ops.compile_tree(ops.Node.from_dict(mc['node']))
                x['ext0'] = (np.asarray(hi, float) - np.asarray(lo, float)) * mc['To'][2]
            except Exception:
                x['ext0'] = None
        if hit is not None and hit[0] != 'centre':
            geo = self.geo()
            v = np.array([pos.x(), pos.y()]) - geo['sc']
            x['gz'] = {'hit': hit, 'geo': geo, 'a0': math.atan2(v[1], v[0])}
            x['geo'] = geo
        else:
            x['gz'] = None
        if hit is not None and hit[0] == 'deform':
            x['deform0'] = self.deform_amounts(x['oid'])
        elif mc is None:
            # fs-shell-3 11-14 (TOOLSET_DESIGN 3.2): the object's mask protects part of it: Move / Rotate move the free
            # part (fs-tools' tools.region_move, posted once at the release); the job is computed as for an object
            # standing at the part's pivot, and the drag itself changes nothing in the scene
            rc = self.region_ctx(x['oid'])
            if rc is not None:
                x['region'] = rc
                x['t0'] = {'translate': [float(v) for v in rc['pivot']], 'rotate': list(x['t0']['rotate']),
                           'scale': x['t0']['scale']}
        return x

    # ---- fs-shell-3 11-14: a masked region moved by the gizmo (ui/region_xform.py) --------------------------------------
    def _rx(self):
        try:
            return getattr(self.vp.window(), 'region_xform', None)
        except Exception:
            return None

    def region_ctx(self, oid=None):
        rx = self._rx()
        return rx.ctx(oid) if rx is not None else None

    def finish(self, x):
        """the drag ends (release, Enter, a click of a typed transform): a region drag posts its one move now; a
        duplicate's placing move joins the Duplicate as one undo step (fs-shell-3 11-14, (g) option A)"""
        if x is not None and x.get('region') is not None:
            rx = self._rx()
            if rx is not None:
                rx.commit(x)
        if x is not None and x.get('dup_group') is not None and x.get('moved'):
            self.vp.engine.post('shell.group_last', int(x['dup_group']), 'Duplicate')

    def deform_amounts(self, oid):
        """-> {gizmo kind: its amount now} from the scene snapshot's base tree ('gizmo <mutation>' nodes)"""
        try:
            win = self.vp.window()
        except Exception:
            win = None
        snap = getattr(win, 'snap', None) or {}
        o = next((ob for ob in snap.get('objects', []) if ob.get('id') == oid), None)
        out = {}
        node = (o or {}).get('base')
        while node is not None:
            nm = str(node.get('name', ''))
            if nm.startswith('gizmo '):
                for kind, _l, mk, par, _s, _u in DEFORMERS:
                    if nm == 'gizmo ' + mk:
                        out[kind] = float(node.get('params', {}).get(par, 0.0))
            elif nm != 'object scale':
                break
            node = node['children'][0] if node.get('children') else None
        return out

    def apply_key(self, r):
        """a key during a drag: redo the job with the typed value / lock, end it (Enter) or put it back (Esc)"""
        vp = self.vp
        x = vp.xform
        if x is None:
            return
        if r == 'update':
            pos = x['p0'] if x.get('kb') or vp.mouse_pos is None else vp.mouse_pos
            job = self.job(x, pos, getattr(vp, '_mods', Qt.NoModifier) | x.get('mods', Qt.NoModifier))
            if job is not None:
                vp.engine.post(*job)
        elif r == 'cancel' and x.get('moved'):
            vp.engine.post('undo')                 # Esc: the drag's own step (one per drag, merged by its id) goes
        if r == 'end':
            self.finish(x)                         # fs-shell-3 11-14: a region drag's one move
        if r in ('end', 'cancel'):
            vp.drag = None
            vp.xform = None
            self.typed, self.lock = '', None
            vp.request(False)
        vp.update()

    def unit(self):
        try:
            return getattr(self.vp.window(), 'units', 'mm')
        except Exception:
            return 'mm'

    def snap_step(self):
        """metres: the grid in use sets it (design D2), else the scene's unit (1 mm, or 1/16 in)"""
        from . import units as U
        st = getattr(self.vp, 'grid_step', None)
        g = st() if callable(st) else None
        return g if g else U.SNAP_STEP.get(self.unit(), 0.001)

    def _len(self, m):
        from . import units as U
        return U.format_length(m, self.unit())

    def _extents(self, x):
        """the object's local extents (m) at the press, its axis stretch and scale included (from its base tree)"""
        if 'ext0' not in x:
            from ..core import ops
            try:
                win = self.vp.window()
            except Exception:
                win = None
            o = next((ob for ob in (getattr(win, 'snap', None) or {}).get('objects', []) if ob.get('id') == x['oid']),
                     None)
            ext = None
            if o is not None:
                try:
                    _c, _p, lo, hi, _L = ops.compile_tree(ops.Node.from_dict(o['base']))
                    ext = (np.asarray(hi, float) - np.asarray(lo, float)) * float(x['t0']['scale'])
                except Exception:
                    ext = None
            x['ext0'] = ext
        return x['ext0']

    def pivot(self, x):
        if self.pivot_mode == 'point' and self.pivot_point is not None:
            return np.asarray(self.pivot_point, float)
        return np.asarray(x['t0']['translate'], float)

    def _typed_value(self):
        """a plain number typed (degrees, a factor, a rate), or None"""
        if not self.typed or self.typed in ('-', '.', '-.'):
            return None
        import re
        t = re.sub(r'\s*(deg|\u00b0|x)\s*$', '', self.typed.strip().lower())
        try:
            return float(t)
        except ValueError:
            return None

    def _typed_length(self):
        """a length typed in any unit (m), or None"""
        from . import units as U
        if not self.typed or self.typed in ('-', '.', '-.'):
            return None
        return U.parse_length(self.typed, self.unit())

    def _typed_has_unit(self):
        import re
        return re.search(r'[a-z"\'\u2032\u2033]', self.typed.lower()) is not None and \
            not re.fullmatch(r'\s*[-+.\d]+\s*(deg|x)?\s*', self.typed.lower())

    def job(self, x, pos, mods):
        """-> the engine job for the drag so far (snapping, the typed value, the axis lock, the pivot). The caller
        posts it. One drag stays one undo step: when the job changes kind mid-drag (a free Scale, then "2x4x8ft"
        typed: a uniform scale, then an axis stretch), the earlier kind's step is undone first."""
        j = self._job(x, pos, mods)
        if j is not None and x.get('region') is not None:
            # fs-shell-3 11-14: a region drag posts nothing until its release; the ghost follows this motion
            from .region_xform import RegionXform
            x['region_motion'] = RegionXform.motion(x, j)
            if self.last_value is not None:
                self.last_value = ('Unmasked part: ' + str(self.last_value[0]),) + tuple(self.last_value[1:])
            return None
        if j is not None and x.get('mass') is not None:
            j = self._mass_job(x, j)
        if j is not None and x.get('party') and hasattr(self.vp, 'with_party'):
            j = self.vp.with_party(x, j)          # fs-shell-6 11-14: the rest of the selection moves with it
        if j is not None:
            lk = x.get('last_kind')
            if lk is not None and lk != j[0]:
                try:
                    self.vp.engine.post('undo')
                except Exception:
                    pass
            x['last_kind'] = j[0]
            x['moved'] = True                      # the drag now has its one undo step
        return j

    def _mass_job(self, x, j):
        """an object-style job computed for the mass -> the mass's own transform ('param' on its node)"""
        from ..core.ops import rot_matrix
        from .engine import euler_from_matrix
        mc = x['mass']
        Ro, to, so = mc['To']
        p0 = mc['node']['params']
        params = {}
        if j[0] == 'props':
            t = j[2].get('transform', {})
            if 'translate' in t:
                params['translate'] = [round(float(v), 6) for v in (Ro.T @ (np.asarray(t['translate'], float) - to)) / so]
            if 'rotate' in t:
                params['rotate'] = [round(float(v), 4) for v in euler_from_matrix(Ro.T @ rot_matrix(t['rotate']))]
            if 'scale' in t:
                f = float(t['scale'])
                params['scale'] = [round(float(v) * f, 6) for v in p0.get('scale', [1.0, 1.0, 1.0])]
        elif j[0] == 'oscale':
            params['scale'] = [round(float(v), 6) for v in j[2]]
        else:
            return None                             # deformers act on whole objects
        return ('param', mc['oid'], mc['nid'], params) if params else None

    def _job(self, x, pos, mods):
        from ..core.ops import rot_matrix
        from .engine import euler_from_matrix
        vp = self.vp
        snap = bool(mods & Qt.ControlModifier)
        typed = self._typed_value()
        gz = x.get('gz')
        hit = gz['hit'] if gz else None
        t0 = x['t0']
        c = np.asarray(t0['translate'], float)
        R0 = rot_matrix(t0['rotate'])
        cam = x['cam']
        f, r, u = cam.basis()
        d = np.array([pos.x() - x['p0'].x(), pos.y() - x['p0'].y()])
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        D = float(np.linalg.norm(c - cam.pos))
        mpp = 2.0 * D * math.tan(math.radians(cam.fov) * 0.5) / H          # metres per screen pixel at the object
        lock = self.lock if self.lock is not None else (hit[1] if hit and hit[0] in ('axis', 'ring') else None)

        if hit is not None and hit[0] == 'deform':
            kind = hit[1]
            spec = next(dd for dd in DEFORMERS if dd[0] == kind)
            base = float((x.get('deform0') or {}).get(kind, 0.0))
            amt = typed if typed is not None else base + d[0] * spec[4]
            if kind == 'inflate' and self.typed:
                tl = self._typed_length()                  # a length in any unit (bare: the scene's)
                amt = tl if tl is not None else amt
            if snap:
                amt = _snap(amt, self.snap_step() if kind == 'inflate' else 0.5)
            self.last_value = ((spec[1], self._len(amt), '') if kind == 'inflate' else (spec[1], amt, spec[5]))
            params = {spec[3]: round(float(amt), 6)}
            if spec[2] == 'taper':
                params['g_min'] = 0.2
            return ('deform', x['oid'], spec[2], params, x['drag'])

        if vp.tool == 'move':
            if lock is not None:
                ax = R0[:, lock]
                if self._typed_length() is not None:
                    w = self._typed_length()
                else:
                    dv, L = (x['geo']['axes'][lock] if x.get('geo') else (None, 0))
                    if dv is None or L < 1e-6:
                        # the axis on screen: project it now
                        ex, ey, _ = cam.project(c + ax * 0.01, W, H)
                        sx, sy, _ = cam.project(c, W, H)
                        sv = np.array([ex - sx, ey - sy])
                        Ls = float(np.linalg.norm(sv))
                        dv, L = (sv / Ls if Ls > 1e-9 else np.zeros(2)), Ls / 0.01 * x['geo']['k'] if x.get('geo') else Ls
                        w = float(d @ dv) * 0.01 / max(Ls, 1e-9)
                    else:
                        w = float(d @ dv) / max(L, 1e-6) * x['geo']['k']
                if snap:
                    w = _snap(w, self.snap_step())
                t = c + ax * w
                self.last_value = ('Move %s' % 'XYZ'[lock], self._len(w), '')
            else:
                t = c + r * d[0] * mpp - u * d[1] * mpp
                if self._typed_length() is not None:       # a typed length without an axis: along the object's X
                    t = c + R0[:, 0] * self._typed_length()
                if snap:
                    t = c + np.array([_snap(v, self.snap_step()) for v in (t - c)])
                self.last_value = ('Move', self._len(float(np.linalg.norm(t - c))), '')
            return ('props', x['oid'], {'transform': {'translate': [round(float(v), 6) for v in t]}}, x['drag'])

        if vp.tool == 'rotate':
            if lock is not None:
                ax = R0[:, lock]
                if gz is not None and gz['hit'][0] == 'ring' and gz['hit'][1] == lock:
                    v = np.array([pos.x(), pos.y()]) - gz['geo']['sc']
                    ang = math.atan2(v[1], v[0]) - gz['a0']
                    ang = ang if float(ax @ f) > 0 else -ang
                else:
                    ang = math.radians(d[0] * 0.5)
                if typed is not None:
                    ang = math.radians(typed)
                if snap:
                    ang = math.radians(_snap(math.degrees(ang), SNAP_DEG))
                Rr = _rodrigues(ax, ang)
                self.last_value = ('Rotate %s' % 'XYZ'[lock], math.degrees(ang), 'deg')
            else:
                a = math.radians(d[0] * 0.5)
                b = math.radians(d[1] * 0.5)
                if typed is not None:
                    a, b = math.radians(typed), 0.0         # a typed value turns about the view's up
                if snap:
                    a = math.radians(_snap(math.degrees(a), SNAP_DEG))
                    b = math.radians(_snap(math.degrees(b), SNAP_DEG))
                Rr = _rodrigues(r, b) @ _rodrigues(np.array([0.0, 1.0, 0.0]), a)
                self.last_value = ('Rotate', math.degrees(a), 'deg')
            Rn = Rr @ R0
            out = {'rotate': [round(float(v_), 4) for v_ in euler_from_matrix(Rn)]}
            p = self.pivot(x)
            if float(np.linalg.norm(p - c)) > 1e-9:
                out['translate'] = [round(float(v_), 6) for v_ in p + Rr @ (c - p)]
            return ('props', x['oid'], {'transform': out}, x['drag'])

        # scale: a typed size ("2x4x8ft", "30 x 40 x 50 mm") or a length with its unit sets the real extent (D1)
        if self.typed:
            from . import units as U
            ext = self._extents(x)
            size = U.parse_size(self.typed, self.unit()) if ('x' in self.typed.lower() or
                                                             self.typed.strip().lower() in
                                                             {k.lower() for k in U.stock()}) else None
            if size is not None and ext is not None:
                st = list(x['stretch0'])
                for i, L_ in enumerate(size[:3]):
                    if ext[i] > 1e-9:
                        st[i] = round(max(st[i] * L_ / ext[i], 0.01), 6)
                self.last_value = ('Size', U.format_size(size, self.unit()), '')
                sh = x.get('shear0')
                return ('oscale', x['oid'], st, x['drag'], sh if sh and any(sh) else None)
            if self._typed_has_unit() and self._typed_length() is not None and ext is not None:
                L_ = abs(self._typed_length())
                i = lock if lock is not None else (None if vp.xform_axis == 'all' else 'xyz'.index(vp.xform_axis))
                if i is not None and ext[i] > 1e-9:
                    st = list(x['stretch0'])
                    st[i] = round(max(st[i] * L_ / ext[i], 0.01), 6)
                    self.last_value = ('Size %s' % 'XYZ'[i], self._len(L_), '')
                    sh = x.get('shear0')
                    return ('oscale', x['oid'], st, x['drag'], sh if sh and any(sh) else None)
                if i is None and max(ext) > 1e-9:
                    fac = L_ / float(max(ext))
                    out = {'scale': round(max(float(t0['scale']) * fac, 1e-3), 6)}
                    self.last_value = ('Size (largest)', self._len(L_), '')
                    return ('props', x['oid'], {'transform': out}, x['drag'])
        if gz is not None and gz['hit'][0] == 'axis' and (mods & Qt.ShiftModifier):
            # skew: the axis-i tip slides toward the other axis the drag runs along most (was Ctrl before 11-11)
            i = gz['hit'][1]
            g = gz['geo']
            cand = [a for a in range(3) if a != i and g['axes'][a][1] >= 8]
            if not cand:
                return None
            j = max(cand, key=lambda a: abs(float(d @ g['axes'][a][0])))
            dj, Lj = g['axes'][j]
            amt = float(d @ dj) / max(Lj, 1e-6)
            if snap:
                amt = _snap(amt, SNAP_SCALE)
            sh = list(x.get('shear0') or [0.0] * 6)
            slot = {(0, 1): 0, (0, 2): 1, (1, 0): 2, (1, 2): 3, (2, 0): 4, (2, 1): 5}[(j, i)]
            sh[slot] = round(max(min(sh[slot] + amt, 3.0), -3.0), 4)
            self.last_value = ('Skew %s' % 'XYZ'[i], amt, '')
            return ('oscale', x['oid'], list(x['stretch0']), x['drag'], sh)
        axis = lock
        if axis is None and gz is None and vp.xform_axis != 'all':
            axis = 'xyz'.index(vp.xform_axis)
        if axis is not None:
            if gz is not None and gz['hit'][0] == 'axis' and gz['hit'][1] == axis:
                dv, L = gz['geo']['axes'][axis]
                fac = math.exp(float(d @ dv) / 150.0)
            else:
                fac = math.exp((d[0] - d[1]) / 300.0)
            if typed is not None:
                fac = typed
            if snap:
                fac = max(_snap(fac, SNAP_SCALE), SNAP_SCALE)
            st = list(x['stretch0'])
            st[axis] = round(max(st[axis] * fac, 0.05), 5)
            self.last_value = ('Stretch %s' % 'XYZ'[axis], fac, 'x')
            sh = x.get('shear0')
            return ('oscale', x['oid'], st, x['drag'], sh if sh and any(sh) else None)
        fac = math.exp((d[0] - d[1]) / 300.0)
        if typed is not None:
            fac = typed
        if snap:
            fac = max(_snap(fac, SNAP_SCALE), SNAP_SCALE)
        out = {'scale': round(max(float(t0['scale']) * fac, 1e-3), 6)}
        p = self.pivot(x)
        if float(np.linalg.norm(p - c)) > 1e-9:
            out['translate'] = [round(float(v_), 6) for v_ in p + fac * (c - p)]
        self.last_value = ('Scale', fac, 'x')
        return ('props', x['oid'], {'transform': out}, x['drag'])

    def cancel_job(self, x):
        """Esc during a drag: the object as it was at the press (one more merge into the same drag step)"""
        t0 = x['t0']
        gz = x.get('gz')
        if gz is not None and gz['hit'][0] == 'deform':
            kind = gz['hit'][1]
            spec = next(dd for dd in DEFORMERS if dd[0] == kind)
            return ('deform', x['oid'], spec[2], {spec[3]: float((x.get('deform0') or {}).get(kind, 0.0))}, x['drag'])
        if self.vp.tool == 'scale' and (x.get('stretch0') is not None) and (self.lock is not None or gz is not None
                                                                              or self.vp.xform_axis != 'all'):
            return ('oscale', x['oid'], list(x['stretch0']), x['drag'], x.get('shear0'))
        return ('props', x['oid'], {'transform': {'translate': list(t0['translate']), 'rotate': list(t0['rotate']),
                                                  'scale': t0['scale']}}, x['drag'])

    # ---- keys during a drag -----------------------------------------------------------------------------------------
    def key(self, ev):
        """-> 'update' (redo the job), 'end' (Enter), 'cancel' (Esc) or None (not ours)"""
        k = ev.key()
        t = ev.text()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            return 'end'
        if k == Qt.Key_Escape:
            return 'cancel'
        if k == Qt.Key_Backspace:
            self.typed = self.typed[:-1]
            return 'update'
        if t == '-':                           # the sign toggles at any point (Blender)
            self.typed = self.typed[1:] if self.typed.startswith('-') else '-' + self.typed
            return 'update'
        if (k in (Qt.Key_X, Qt.Key_Y, Qt.Key_Z) and self.typed and self._axis_after_value()
                and not (ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier))):
            i = {Qt.Key_X: 0, Qt.Key_Y: 1, Qt.Key_Z: 2}[k]
            self.lock = None if self.lock == i else i     # 11-11: "90 X" as well as "X 90" (Blender takes both)
            return 'update'
        if self.typed and t and t.isprintable() and not (ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
            self.typed += t                    # a value has begun: units, 'x' of a size, spaces are part of it
            return 'update'
        if k in (Qt.Key_X, Qt.Key_Y, Qt.Key_Z) and not (ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
            i = {Qt.Key_X: 0, Qt.Key_Y: 1, Qt.Key_Z: 2}[k]
            self.lock = None if self.lock == i else i
            return 'update'
        if t and (t.isdigit() or t in '.-'):
            if t == '-':
                self.typed = self.typed[1:] if self.typed.startswith('-') else '-' + self.typed
            elif t == '.' and '.' in self.typed:
                pass
            else:
                self.typed += t
            return 'update'
        return None

    def _axis_after_value(self):
        """may X / Y / Z after a typed value pick the axis? Rotate: always (an angle has no letters); Move: while the
        value is a bare number (after a unit's letters they are part of it); Scale: never ('x' joins a size)"""
        import re
        tool = getattr(self.vp, 'tool', None)
        if tool == 'rotate':
            return True
        if tool == 'move':
            return re.fullmatch(r'[-+.\d\s]*', self.typed or '') is not None
        return False

    # ---- typed transforms without a drag (11-11, the copy / scale / rotate / place flow) ------------------------------
    def start_typed(self):
        """the active object tool's key pressed again (W W, R R, Y Y): a transform driven by typed values, no drag
        needed; X / Y / Z pick the axis, Enter (or a click) applies, Esc (or a right-click) puts it back"""
        vp = self.vp
        if vp.drag is not None or vp.xform is not None or not vp.selected:
            return False
        if getattr(vp, 'tool', None) not in ('move', 'rotate', 'scale'):
            return False
        g = self.geo()
        if g is None:
            return False
        c = QPointF(float(g['sc'][0]), float(g['sc'][1]))
        if not vp.begin_xform(c):
            return False
        vp.drag = 'xform'
        vp.xform['kb'] = True
        mp = getattr(vp, 'mouse_pos', None)        # the mouse moves it from where the cursor is now (no jump)
        vp.xform['p0'] = QPointF(mp) if (mp is not None and vp.rect().contains(mp.toPoint())) else c
        self.typed, self.lock = '', None
        try:
            vp.window().status.showMessage(
                {'move': 'Move: type a distance (X / Y / Z picks the axis; units welcome, e.g. -120 mm)',
                 'rotate': 'Rotate: type an angle in degrees (X / Y / Z picks the axis)',
                 'scale': 'Scale: type a factor (0.5), a size with a unit, or W x H x D'}[vp.tool]
                + '. Enter or a click applies; Esc puts it back.', 15000)
        except Exception:
            pass
        vp.update()
        return True

    # ---- drawing ----------------------------------------------------------------------------------------------------
    def draw(self, p):
        g = self.geo()
        vp = self.vp
        rx = self._rx()
        if rx is not None:
            try:
                rx.draw(p)                     # fs-shell-3 11-14: the free part tinted (its soft edge), its ghost
            except Exception as e:
                print('region view: %r' % (e,))
        if g is not None:
            hov = getattr(vp, 'handle_hover', None)
            act = vp.xform['gz']['hit'] if (vp.xform is not None and vp.xform.get('gz')) else None
            sc = QPointF(float(g['sc'][0]), float(g['sc'][1]))
            f = vp.cam.basis()[0]
            order = sorted(range(3), key=lambda i: -float(g['R'][:, i] @ f))    # far axes first
            for i in order:
                col = QColor(AXIS_COL[i])
                lit = hov in (('axis', i), ('ring', i)) or act in (('axis', i), ('ring', i)) or self.lock == i
                if lit:
                    col = col.lighter(150)
                if vp.tool == 'rotate':
                    ring = g['rings'][i]
                    path = QPainterPath(QPointF(float(ring[0][0]), float(ring[0][1])))
                    for pt in ring[1:]:
                        path.lineTo(QPointF(float(pt[0]), float(pt[1])))
                    p.setPen(QPen(col, 3.2 if lit else 2.0))
                    p.setBrush(Qt.NoBrush)
                    p.drawPath(path)
                    continue
                dv, L = g['axes'][i]
                if L < 8:
                    continue
                tp = g['sc'] + dv * L
                tip = QPointF(float(tp[0]), float(tp[1]))
                p.setPen(QPen(col, 3.0 if lit else 2.0))
                p.drawLine(sc, tip)
                p.setBrush(col)
                p.setPen(Qt.NoPen)
                if vp.tool == 'move':
                    n = np.array([-dv[1], dv[0]])
                    a = g['sc'] + dv * (L + 10)
                    b = g['sc'] + dv * (L - 4)
                    path = QPainterPath(QPointF(float(a[0]), float(a[1])))
                    path.lineTo(QPointF(float(b[0] + n[0] * 5), float(b[1] + n[1] * 5)))
                    path.lineTo(QPointF(float(b[0] - n[0] * 5), float(b[1] - n[1] * 5)))
                    path.closeSubpath()
                    p.drawPath(path)
                else:
                    p.drawRect(QRectF(tip.x() - 5, tip.y() - 5, 10, 10))
            p.setPen(QPen(QColor(235, 235, 235, 220), 2.0))
            p.setBrush(QColor(255, 255, 255, 60) if hov == ('centre', -1) else Qt.NoBrush)
            p.drawEllipse(sc, 7, 7)
            p.setBrush(Qt.NoBrush)
            self._draw_deformers(p, g, hov, act)
            if self.lock is not None and vp.xform is not None:
                # the locked axis as a long line through the object
                ax = g['R'][:, self.lock]
                W, H = max(vp.width(), 1), max(vp.height(), 1)
                a = vp.cam.project(g['c'] - ax * g['k'] * 8, W, H)
                b = vp.cam.project(g['c'] + ax * g['k'] * 8, W, H)
                p.setPen(QPen(AXIS_COL[self.lock], 1.2, Qt.DashLine))
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
        self._draw_pivot(p)
        self._draw_readout(p)
        gr = getattr(vp, 'grid', None)
        if gr is not None and gr.active():
            gr.draw_labels(p)                  # design D2: the major lines' values
        ms = getattr(vp.window(), 'measures', None)
        if ms is not None:
            ms.draw(p)                         # queue item 5: the pinned measures
        sc_ = getattr(vp.window(), 'section', None)
        if sc_ is not None:
            try:
                sc_.draw(p)                    # design 4.4: the section plane, thickness / gap, the ruler
            except Exception as e:
                print('section view: %r' % (e,))
        mk = getattr(vp.window(), 'marks', None)
        if mk is not None:
            try:
                mk.draw(p)                     # design 4.14: the user's marks (pins, circled regions), the loop being drawn
            except Exception as e:
                print('marks view: %r' % (e,))
        tp = getattr(vp.window(), 'topo', None)
        if tp is not None:
            try:
                tp.draw(p)                     # design D4: the hidden-void view, the prompt bar's place
            except Exception as e:
                print('topology view: %r' % (e,))
        cu = getattr(vp.window(), 'surface_cursor', None)
        if cu is not None:
            try:
                cu.draw(p)                     # design 4.6: the brush ring on the surface, live mm
            except Exception as e:
                print('surface cursor: %r' % (e,))
        rg = getattr(vp.window(), 'regions_ui', None)
        if rg is not None:
            try:
                rg.draw(p)                     # design 4.8: the region lasso / box hide being drawn
            except Exception as e:
                print('regions view: %r' % (e,))

    def _draw_deformers(self, p, g, hov, act):
        from . import glyphs as GL
        p.setFont(QFont('Segoe UI', 7))
        for kind, cpt in g['deformers']:
            lit = hov == ('deform', kind) or act == ('deform', kind)
            c = QPointF(float(cpt[0]), float(cpt[1]))
            p.setPen(QPen(QColor(226, 190, 70) if lit else QColor(150, 158, 170), 1.4))
            p.setBrush(QColor(29, 32, 38, 220))
            p.drawEllipse(c, 11, 11)
            # the deformer's own little picture: a bar bent, twisted, tapered or swollen
            p.setPen(QPen(QColor(235, 235, 235) if lit else QColor(200, 204, 210), 1.5))
            x, y = c.x(), c.y()
            if kind == 'bend':
                path = QPainterPath(QPointF(x - 6, y + 3))
                path.quadTo(QPointF(x, y - 7), QPointF(x + 6, y + 3))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
            elif kind == 'twist':
                for s in (-1, 1):
                    path = QPainterPath(QPointF(x - 6, y + 3 * s))
                    path.cubicTo(QPointF(x - 2, y + 3 * s), QPointF(x + 2, y - 3 * s), QPointF(x + 6, y - 3 * s))
                    p.setBrush(Qt.NoBrush)
                    p.drawPath(path)
            elif kind == 'taper':
                p.drawLine(QPointF(x - 6, y - 5), QPointF(x + 6, y - 2))
                p.drawLine(QPointF(x - 6, y + 5), QPointF(x + 6, y + 2))
            else:
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, 4, 4)
                p.drawEllipse(c, 7, 7)
            if lit:
                p.setPen(QColor(235, 235, 235))
                p.drawText(QPointF(x - 14, y + 22), next(dd[1] for dd in DEFORMERS if dd[0] == kind))

    def _draw_pivot(self, p):
        if self.pivot_mode != 'point' or self.pivot_point is None:
            return
        vp = self.vp
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        sx, sy, z = vp.cam.project(np.asarray(self.pivot_point, float), W, H)
        if z <= 1e-6:
            return
        p.setPen(QPen(QColor(250, 204, 21), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(sx, sy), 6, 6)
        p.drawLine(QPointF(sx - 10, sy), QPointF(sx + 10, sy))
        p.drawLine(QPointF(sx, sy - 10), QPointF(sx, sy + 10))

    def _draw_readout(self, p):
        vp = self.vp
        if vp.xform is None or self.last_value is None:
            if self.picking_pivot:
                text = 'Click on the form to set the pivot (Esc: keep the centre)'
            else:
                return
        else:
            what, v, unit = self.last_value
            vs = v if isinstance(v, str) else '%.2f' % v
            text = '%s  %s%s' % (what, vs if not self.typed else '%s   (typed: %s)' % (vs, self.typed),
                                 (' ' + unit) if unit else '')
            text += '   X / Y / Z: axis   Ctrl: snap   type a number   Enter / Esc'
        p.setFont(QFont('Segoe UI', 9))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 16
        r = QRectF((vp.width() - w) / 2.0, vp.height() - 64, w, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(16, 18, 22, 200))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(QColor(235, 235, 235))
        p.drawText(r, Qt.AlignCenter, text)

    # ---- pivot --------------------------------------------------------------------------------------------------------
    def set_pivot_mode(self, mode):
        """'centre', or 'pick' (the next left click on the form sets the point; the mode becomes 'point')"""
        if mode == 'pick':
            self.picking_pivot = True
        else:
            self.picking_pivot = False
            self.pivot_mode = 'centre'
            self.pivot_point = None
        self.vp.update()

    def pick_pivot(self, pos):
        """the surface point under pos from the frame's depth (the ray t), or None off the form"""
        vp = self.vp
        fr = getattr(vp, 'frame', None)
        if fr is None or fr.get('depth') is None:
            return None
        dep, obj = fr['depth'], fr['obj']
        h, w = obj.shape
        ix = int(pos.x() * w / max(vp.width(), 1))
        iy = int(pos.y() * h / max(vp.height(), 1))
        if not (0 <= ix < w and 0 <= iy < h) or obj[iy, ix] < 0:
            return None
        cam = fr.get('cam') or vp.cam
        o, dvec = cam.ray(ix, iy, w, h)
        P = np.asarray(o, float) + np.asarray(dvec, float) * float(dep[iy, ix])
        self.pivot_point = [float(v) for v in P]
        self.pivot_mode = 'point'
        self.picking_pivot = False
        vp.update()
        cb = getattr(vp.window(), 'context_bar', None)
        if cb is not None:
            cb.sync()
        return self.pivot_point


class DragKeys(QObject):
    """during a gizmo drag the number keys, X / Y / Z, Backspace, Enter and Esc belong to the drag (the app's
    shortcuts would otherwise take X: Cut, Y: Scale, Z ...); installed on the application by install()"""

    def __init__(self, vp):
        super().__init__(vp)
        self.vp = vp

    def _ours(self, ev):
        k = ev.key()
        t = ev.text()
        g = getattr(self.vp, 'gizmo', None)
        if g is not None and g.typed and t and t.isprintable():
            return True                        # a value has begun: every printable key is part of it
        return (k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape, Qt.Key_Backspace, Qt.Key_X, Qt.Key_Y, Qt.Key_Z)
                or (t and (t.isdigit() or t in '.-')))

    def eventFilter(self, obj, ev):
        vp = self.vp
        t = ev.type()
        if t in (QEvent.ShortcutOverride, QEvent.KeyPress) and not ev.isAutoRepeat():
            import time as _time
            vp._last_key_t = _time.perf_counter()      # install_window: only a KEY choosing the tool again starts typing
        if t == QEvent.MouseButtonPress and obj is vp and vp.xform is not None and vp.xform.get('kb'):
            g = getattr(vp, 'gizmo', None)
            if g is not None:                      # a typed transform: a click applies it, a right-click puts it back
                g.apply_key('cancel' if ev.button() == Qt.RightButton else 'end')
                return True
        if t not in (QEvent.ShortcutOverride, QEvent.KeyPress):
            return False
        g = getattr(vp, 'gizmo', None)
        if g is None:
            return False
        if g.picking_pivot and t == QEvent.KeyPress and ev.key() == Qt.Key_Escape:
            g.set_pivot_mode('centre')
            return True
        if vp.drag != 'xform' or vp.xform is None or not self._ours(ev):
            return False
        if t == QEvent.ShortcutOverride:
            ev.accept()                    # the key comes to us as a KeyPress, not to a shortcut
            return True
        r = g.key(ev)
        if r is None:
            return False
        g.apply_key(r)
        return True


def install_window(win):
    """the window half (after its panels exist): the active object tool chosen again starts a typed transform"""
    vp = win.viewport
    g = getattr(vp, 'gizmo', None)
    tools = getattr(win, 'tools', None)
    if g is None or tools is None or not hasattr(tools, 'toolChanged'):
        return False
    st = {'last': getattr(tools, 'tool', None)}

    def changed(key):
        import time as _time
        if key != st['last'] and vp.xform is not None and vp.xform.get('kb'):
            # fs-shell-3 11-14: another tool chosen while a typed transform runs (a copy following the mouse, Ctrl+D or
            # the Duplicate row) places it where it is, as Enter would; it no longer follows the cursor under that tool
            g.apply_key('end')
        by_key = _time.perf_counter() - getattr(vp, '_last_key_t', -1e9) < 0.3
        if key == st['last'] and key in ('move', 'rotate', 'scale') and by_key:
            # the tool's KEY pressed again (not a script's or a button's set_tool: test 1 re-sets the tool before
            # every handle drag, and a typed transform must not start under it)
            QTimer.singleShot(0, g.start_typed)    # after the viewport has taken the tool
        st['last'] = key
    tools.toolChanged.connect(changed)
    return True


def install(vp):
    from PySide6.QtWidgets import QApplication
    from ..render import grids as GR
    vp.grid = GR.Grid(vp)                      # design D2: floor / view grids (render/grids.py); snapping uses its step
    vp.grid_step = vp.grid.step
    vp.contours = GR.Contours(vp, vp.grid)     # design D2's third grid: contour lines on the model
    vp.overlays = [vp.grid, vp.contours]       # composited in viewport._compose, under the reference planes
    vp.gizmo = Gizmo(vp)
    vp.gizmo_keys = DragKeys(vp)
    QApplication.instance().installEventFilter(vp.gizmo_keys)
    return vp.gizmo
