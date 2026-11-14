"""Marks in the window (fs-shell, 2026-11-11; design 4.14, chunk 9; data and the shared folder: tools/marks.py).

Tools > Mark (or the Marks button on the status bar): a click on a form drops a numbered pin; a drag draws a loop that is
projected onto the surface as a region (every pixel of the form inside the loop). The user names it and adds a note (one
dialog, Enter to skip). The mark sits in its object's own frame, so it follows the object; it is drawn with its number
and name (dashed where the form hides it), saved in the project, and written at once to the shared view folder with a
still of what was in view, where the sessions read it (tools/marks.read_shared). Right-click a mark: rename / note, look from
where it was marked, delete. Esc leaves the tool. Middle / right drags still orbit and pan while marking."""
import math
import os
import time

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen
from PySide6.QtWidgets import QToolButton, QMenu

MARK_COL = QColor(255, 196, 64)


class _Obj:
    """the transform an object has in the window's snapshot, for tools.protected_merge's frame helpers"""

    def __init__(self, tr):
        self.transform = {'translate': list(tr.get('translate', (0, 0, 0))), 'rotate': list(tr.get('rotate', (0, 0, 0))),
                          'scale': float(tr.get('scale', 1.0))}


def _local(tr, P):
    from ..tools.protected_merge import _to_local
    return _to_local(_Obj(tr), np.asarray(P, float).reshape(-1, 3))


def _world(tr, Q):
    from ..tools.protected_merge import _to_world
    return _to_world(_Obj(tr), np.asarray(Q, float).reshape(-1, 3))


def frame_rays(cam, ii, jj, w, h):
    """vectorised Camera.ray for frame pixels (ii, jj) -> (origins (n, 3), unit directions (n, 3))"""
    f, r, u = cam.basis()
    th = math.tan(math.radians(cam.fov) * 0.5)
    a = w / float(h)
    px = (2.0 * (ii + 0.5) / w - 1.0) * th * a
    py = (1.0 - 2.0 * (jj + 0.5) / h) * th
    if getattr(cam, 'ortho', False):
        D, B = cam.ortho_params()
        O = cam.pos[None] + (r[None] * px[:, None] + u[None] * py[:, None]) * D - f[None] * B
        return O, np.repeat(f[None], len(ii), 0)
    d = f[None] + r[None] * px[:, None] + u[None] * py[:, None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.repeat(np.asarray(cam.pos, float)[None], len(ii), 0), d


class MarkLayer(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.items = []
        self.active = False
        self.path = None                 # the loop being drawn (widget points)
        self.label_rects = []
        self.next_n = 1
        self.button = QToolButton()
        self.button.setText('Marks 0')
        self.button.setToolTip('Marks: pins and circled regions you made, read by the sessions (Tools > Mark)')
        self.button.setPopupMode(QToolButton.InstantPopup)
        self.menu = QMenu(self.button)
        self.button.setMenu(self.menu)
        self.menu.aboutToShow.connect(self._fill_menu)
        win.status.addPermanentWidget(self.button)

    # ---- the frame's hits ----------------------------------------------------------------------------------------------
    def _frame(self):
        fr = getattr(self.win.viewport, 'frame', None)
        if fr is None or fr.get('obj') is None or fr.get('depth') is None:
            return None
        return fr

    def hit(self, pos):
        """-> (world point, object id) under a widget point, from the last frame, or None"""
        fr = self._frame()
        vp = self.win.viewport
        if fr is None:
            return None
        obj, dep = np.asarray(fr['obj']), np.asarray(fr['depth'])
        h, w = obj.shape
        ix = int(pos.x() * w / max(vp.width(), 1))
        iy = int(pos.y() * h / max(vp.height(), 1))
        if not (0 <= ix < w and 0 <= iy < h) or obj[iy, ix] < 0:
            return None
        O, D = frame_rays(fr.get('cam') or vp.cam, np.array([ix]), np.array([iy]), w, h)
        return O[0] + D[0] * float(dep[iy, ix]), int(obj[iy, ix])

    def region(self, pts):
        """the form's pixels inside a loop (widget points) -> (world points, object id) of the object most covered"""
        from matplotlib.path import Path
        fr = self._frame()
        vp = self.win.viewport
        if fr is None or len(pts) < 3:
            return None
        obj, dep = np.asarray(fr['obj']), np.asarray(fr['depth'])
        h, w = obj.shape
        sx, sy = w / max(vp.width(), 1), h / max(vp.height(), 1)
        poly = np.array([[p.x() * sx, p.y() * sy] for p in pts])
        x0, y0 = np.floor(poly.min(0)).astype(int)
        x1, y1 = np.ceil(poly.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x1 <= x0 or y1 <= y0:
            return None
        J, I = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        I, J = I.ravel(), J.ravel()
        inside = Path(poly).contains_points(np.stack([I + 0.5, J + 0.5], 1))
        I, J = I[inside], J[inside]
        ids = obj[J, I]
        keep = ids >= 0
        if not keep.any():
            return None
        oid = int(np.bincount(ids[keep]).argmax())
        sel = ids == oid
        I, J = I[sel], J[sel]
        O, D = frame_rays(fr.get('cam') or vp.cam, I.astype(float), J.astype(float), w, h)
        P = O + D * dep[J, I].astype(float)[:, None]
        return P, oid, int(sel.sum())

    # ---- making marks --------------------------------------------------------------------------------------------------
    def add_pin(self, pos):
        r = self.hit(pos)
        if r is None:
            self.win.status.showMessage('Mark: click on a form.', 5000)
            return None
        P, oid = r
        return self._finish({'kind': 'pin', 'oid': oid, 'p': P, 'P': P[None]}, pos)

    def add_region(self, pts):
        r = self.region(pts)
        if r is None:
            self.win.status.showMessage('Mark: circle part of a form.', 5000)
            return None
        P, oid, npx = r
        # the outline: each loop point's own hit (points off the form are left out), in order
        out = []
        for q in pts[::max(1, len(pts) // 120)]:
            h_ = self.hit(q)
            if h_ is not None and h_[1] == oid:
                out.append(h_[0])
        c = P.mean(0)
        m = {'kind': 'region', 'oid': oid, 'p': c, 'P': P, 'outline': np.array(out) if out else P[:1],
             'pixels': npx, 'size_mm': ((P.max(0) - P.min(0)) * 1000.0).round(1).tolist()}
        return self._finish(m, pts[0])

    def _finish(self, m, pos):
        vp = self.win.viewport
        tr = vp.transforms.get(m['oid']) or {}
        objs = {o['id']: o for o in (self.win.snap or {}).get('objects', [])}
        m['object'] = objs.get(m['oid'], {}).get('name', 'object %d' % m['oid'])
        m['n'] = self.next_n
        self.next_n += 1
        m['q'] = _local(tr, m['p'])[0].tolist()
        if m['kind'] == 'region':
            m['outline_q'] = _local(tr, m['outline']).tolist()
            S = m['P'][np.linspace(0, len(m['P']) - 1, min(400, len(m['P']))).astype(int)]
            m['samples_q'] = _local(tr, S).tolist()
        m['centre_mm'] = (np.asarray(m['p']) * 1000.0).round(2).tolist()
        m['camera'] = vp.cam.to_dict()
        m['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        name, note = '', ''
        if not getattr(self.win, 'no_prompt', False):
            got = self._ask(m['n'], m['kind'])
            if got is None:                       # Esc on the dialog: no mark
                self.next_n -= 1
                return None
            name, note = got
        m['name'], m['note'] = name, note
        for k in ('p', 'P', 'outline'):
            m.pop(k, None)
        self.items.append(m)
        self._sync_button()
        vp.update()
        self.share(m)
        return m

    def _ask(self, n, kind, name='', note=''):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit
        d = QDialog(self.win)
        d.setWindowTitle('Mark %d' % n)
        lay = QFormLayout(d)
        a, b = QLineEdit(name), QLineEdit(note)
        a.setPlaceholderText('what it is (e.g. this crease, the lid edge)')
        b.setPlaceholderText('what is wrong or wanted (optional)')
        lay.addRow('Name', a)
        lay.addRow('Note', b)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lay.addRow(bb)
        if d.exec() != QDialog.Accepted:
            return None
        return a.text().strip(), b.text().strip()

    # ---- the shared view ----------------------------------------------------------------------------------------------
    def share(self, m=None):
        """marks.json in the shared folder (and, for a new mark, a still of the view with the marks drawn)"""
        from ..tools import marks as MK
        from .main_window import app_settings as _s
        try:
            setting = str(_s().value('share_dir', '') or '') or None
        except Exception:
            setting = None
        d = MK.share_dir(setting)
        try:
            os.makedirs(d, exist_ok=True)
            if m is not None:
                self.win.viewport.update()
                img = self.win.viewport.grab()
                p = os.path.join(d, 'mark_%d.png' % m['n'])
                img.save(p)
                m['still'] = p
            MK.write_shared(self.items, camera=self.win.viewport.cam.to_dict(),
                            project=getattr(self.win, 'project_path', None), directory=setting)
            self.shared_dir = d
        except Exception as e:
            self.win.status.showMessage('Marks: the shared view could not be written (%s).' % e, 10000)

    # ---- the list ----------------------------------------------------------------------------------------------------
    def _sync_button(self):
        self.button.setText('Marks %d' % len(self.items))

    def _fill_menu(self):
        self.menu.clear()
        a = self.menu.addAction('Mark a point or circle an area (Tools > Mark)')
        a.triggered.connect(lambda: start(self.win))
        if self.items:
            self.menu.addSeparator()
            for k, m in enumerate(self.items):
                t = '%d  %s%s' % (m['n'], m.get('name') or ('pin' if m['kind'] == 'pin' else 'region'),
                                  (' - ' + m['note']) if m.get('note') else '')
                act = self.menu.addAction(t[:80])
                act.triggered.connect(lambda _=False, k=k: self.look(k))
            self.menu.addSeparator()
            s = self.menu.addAction('Share the view again (marks, camera, a still)')
            s.triggered.connect(lambda: self.share())
            c = self.menu.addAction('Clear all marks')
            c.triggered.connect(self.clear)

    def look(self, k):
        """look from where the mark was made"""
        from ..render.camera import Camera
        m = self.items[k]
        if m.get('camera'):
            self.win.viewport.cam = Camera.from_dict(m['camera'])
            self.win.viewport.request(False)
        self.win.viewport.update()

    def clear(self):
        self.items = []
        self.next_n = 1
        self._sync_button()
        self.share()
        self.win.viewport.update()

    def to_list(self):
        from ..tools.marks import clean
        return [clean(m) for m in self.items]

    def load(self, lst):
        self.items = [dict(m) for m in (lst or [])]
        self.next_n = max([int(m.get('n', 0)) for m in self.items] + [0]) + 1
        self._sync_button()
        self.win.viewport.update()

    # ---- drawing (the gizmo's overlay painter) ---------------------------------------------------------------------------
    def draw(self, p):
        vp = self.win.viewport
        W, H = max(vp.width(), 1), max(vp.height(), 1)
        cam = vp.cam
        self.label_rects = []
        if self.active:                    # the tool's badge, over the viewport's own (it shows the object tool)
            text = 'Mark  -  click a point, or drag a loop around an area; Esc leaves'
            p.setFont(QFont('Segoe UI', 10, QFont.DemiBold))
            r = QRectF(12, 10, min(W - 140, max(9 * len(text) + 24, 520)), 26)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(20, 22, 26, 255))
            p.drawRoundedRect(r, 5, 5)
            p.setPen(MARK_COL)
            p.drawText(r.adjusted(10, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        if self.path is not None and len(self.path) > 1:          # the loop being drawn
            p.setPen(QPen(MARK_COL, 2.0, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            path = QPainterPath(self.path[0])
            for q in self.path[1:]:
                path.lineTo(q)
            path.closeSubpath()
            p.drawPath(path)
        if not self.items:
            return
        ms = getattr(self.win, 'measures', None)
        p.setFont(QFont('Segoe UI', 9, QFont.Bold))
        fm = p.fontMetrics()
        placed = []
        # the Section tool (design 4.4): a mark on the cut-away side (n.p > d) is faded and dashed, "(cut away)"
        sec = getattr(cam, 'section', None)
        sv = cam.section_vec() if sec is not None else None
        self.last_states = {}                      # n -> 'shown' / 'hidden' (behind a form) / 'cut away' (a section)

        def cut_away(P):
            if sv is None:
                return np.zeros(len(P), bool)
            return (np.asarray(P, float) @ sv[:3] - sv[3]) > 1e-6
        for k, m in enumerate(self.items):
            tr = vp.transforms.get(m['oid'])
            if tr is None:
                continue
            c = _world(tr, m['q'])[0]
            cx, cy, cd = cam.project(c, W, H)
            if cd <= 0:
                continue
            if m['kind'] == 'region' and m.get('outline_q'):
                Pw = _world(tr, m['outline_q'])
                pr = np.array([cam.project(q, W, H) for q in Pw])
                hid = ms._hidden(Pw, pr, W, H) if ms is not None else np.zeros(len(Pw), bool)
                gone = cut_away(Pw)
                faded = QColor(MARK_COL)
                faded.setAlpha(70)
                for i in range(len(pr)):
                    j = (i + 1) % len(pr)
                    if pr[i, 2] <= 0 or pr[j, 2] <= 0:
                        continue
                    if gone[i] or gone[j]:
                        p.setPen(QPen(faded, 1.2, Qt.DotLine))
                    else:
                        p.setPen(QPen(MARK_COL, 1.4 if (hid[i] or hid[j]) else 2.2,
                                      Qt.DashLine if (hid[i] or hid[j]) else Qt.SolidLine))
                    p.drawLine(QPointF(pr[i, 0], pr[i, 1]), QPointF(pr[j, 0], pr[j, 1]))
            hidden = bool(ms._hidden(c[None], np.array([[cx, cy, cd]]), W, H)[0]) if ms is not None else False
            if bool(cut_away(c[None])[0]):
                hidden = True
                self.last_states[m['n']] = 'cut away'
            else:
                self.last_states[m['n']] = 'hidden' if hidden else 'shown'
            p.setPen(QPen(QColor(20, 20, 20), 1.2))
            col = QColor(MARK_COL)
            if hidden:
                col.setAlpha(120)
            p.setBrush(col)
            p.drawEllipse(QPointF(cx, cy), 9, 9)
            p.setPen(QColor(20, 20, 20))
            p.drawText(QRectF(cx - 9, cy - 9, 18, 18), Qt.AlignCenter, str(m['n']))
            text = (m.get('name') or '') + ((': ' + m['note']) if m.get('note') else '')
            if bool(cut_away(c[None])[0]):
                text = (text + '  (cut away)') if text else '(cut away)'
            if text:
                tw = fm.horizontalAdvance(text[:60]) + 12
                r = QRectF(cx + 14, cy - 10, tw, fm.height() + 6)
                while any(r.intersects(q) for q in placed):
                    r.translate(0, fm.height() + 8)
                placed.append(r)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(16, 18, 22, 215))
                p.drawRoundedRect(r, 4, 4)
                p.setPen(MARK_COL)
                p.drawText(r, Qt.AlignCenter, text[:60])
                self.label_rects.append((r, k))
            self.label_rects.append((QRectF(cx - 10, cy - 10, 20, 20), k))

    def mark_at(self, pos):
        for r, k in self.label_rects:
            if r.contains(pos):
                return k
        return None


class _MarkFilter(QObject):
    """on the viewport: with the Mark tool, a left click pins, a left drag circles; right-click on a mark: its menu"""

    def __init__(self, layer):
        super().__init__(layer)
        self.layer = layer
        self.press = None

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        L = self.layer
        t = ev.type()
        if t == QEvent.MouseButtonPress and ev.button() == Qt.RightButton:
            k = L.mark_at(ev.position())
            if k is not None and k < len(L.items):
                self._menu(k)
                return True
        if not L.active:
            return False
        if t in (QEvent.ShortcutOverride, QEvent.KeyPress) and ev.key() == Qt.Key_Escape:
            if t == QEvent.ShortcutOverride:
                ev.accept()                    # Esc comes here as a key press, not to the window's Esc shortcut
                return True
            stop(L.win)
            return True
        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            self.press = QPointF(ev.position())
            L.path = [QPointF(ev.position())]
            return True
        if t == QEvent.MouseMove and self.press is not None:
            L.path.append(QPointF(ev.position()))
            L.win.viewport.update()
            return True
        if t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and self.press is not None:
            pts, self.press = L.path, None
            L.path = None
            span = max((math.hypot(q.x() - pts[0].x(), q.y() - pts[0].y()) for q in pts), default=0.0)
            if span < 6:
                L.add_pin(pts[0])
            else:
                L.add_region(pts)
            L.win.viewport.update()
            return True
        return False

    def _menu(self, k):
        from PySide6.QtGui import QCursor
        L = self.layer
        m = L.items[k]
        menu = QMenu(L.win)
        ed = menu.addAction('Name and note...')
        lk = menu.addAction('Look from where it was marked')
        dl = menu.addAction('Delete mark %d' % m['n'])
        got = menu.exec(QCursor.pos())
        if got is ed:
            r = L._ask(m['n'], m['kind'], m.get('name', ''), m.get('note', ''))
            if r is not None:
                m['name'], m['note'] = r
                L.share()
        elif got is lk:
            L.look(k)
        elif got is dl:
            del L.items[k]
            L._sync_button()
            L.share()
        L.win.viewport.update()


def start(win):
    L = win.marks
    L.active = True
    win.status.showMessage('Mark: click a point on a form, or circle an area. The sessions read your marks.', 10000)
    win.viewport.update()


def stop(win):
    L = win.marks
    L.active = False
    L.path = None
    win.viewport.update()


def install(win):
    L = MarkLayer(win)
    win.marks = L
    L.filter = _MarkFilter(L)
    win.viewport.installEventFilter(L.filter)
    tm = getattr(win, 'menu_tools', None)
    if tm is not None:
        a = tm.addAction('Mark a point or circle an area (for the sessions)')
        a.triggered.connect(lambda: start(win))
        L.action = a
    return L
