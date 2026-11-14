"""Selecting an object inside another (fs-shell, 2026-11-14; design: an object inside another
is selected (to move or change it) by a click again on the same spot; a held key goes back a level; what counts
as seen follows the X-ray slider).

- A click takes the first object along the ray that is drawn solid enough to see: an object whose drawn opacity
  ((1 - X-ray) x its own opacity, the same rule the X-ray composite draws with) is under 50 % is passed through; if
  everything under the cursor is see-through, the front one.
- A plain click again on the same spot (within 4 px, the view unmoved) goes one object deeper along that ray, the last
  wrapping to the front; A held while clicking goes one level back out (A: no tool or action uses it; Shift and Ctrl
  add to the selection, Alt orbits).
- A line under the cursor says where it is: "Inner · 2 of 3 under the cursor   click again: Bone   A+click: back to
  Cube   Esc: nothing".
The objects along the ray come from the card's own pick (gpu/viewport.py pick(..., exclude)), one ray per object; on
the CPU view, the CPU pick the same way."""
import time

from PySide6.QtCore import QObject, QEvent, QPointF, Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel

BACK_KEY = Qt.Key_A
BACK_NAME = 'A'
NEAR_PX = 4.0
SEE_THROUGH = 0.5


class _HeldKey(QObject):
    """records whether the back key is held (never consumes it)"""

    def __init__(self):
        super().__init__()
        self.held = False

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.KeyPress, QEvent.KeyRelease) and getattr(ev, 'key', lambda: None)() == BACK_KEY \
                and not ev.isAutoRepeat():
            self.held = t == QEvent.KeyPress
        elif t == QEvent.WindowDeactivate:
            self.held = False
        return False


class SelectInside:
    def __init__(self, win):
        self.win = win
        self.state = None                  # (x, y, cam key, chain [oids front -> inner], index)
        self.key = _HeldKey()
        QApplication.instance().installEventFilter(self.key)
        self.tip = QLabel(win.viewport)
        self.tip.setStyleSheet('background: rgba(20, 22, 26, 225); color: #ffd278; padding: 5px 9px; border-radius: 5px;')
        self.tip.hide()
        self._hide = QTimer(self.tip)
        self._hide.setSingleShot(True)
        self._hide.timeout.connect(self.tip.hide)

    # ---- the objects along the ray ---------------------------------------------------------------------------------------
    def chain(self, pos):
        """-> [object ids] front to back under the view point pos (widget coordinates)"""
        vp = self.win.viewport
        eng = self.win.engine
        view = vp.view()
        x = pos.x() * view['w'] / max(vp.width(), 1)
        y = pos.y() * view['h'] / max(vp.height(), 1)
        out = []
        with eng.lock:
            sc = eng.hist.scene
            for _ in range(12):
                try:
                    if eng.gpu_view is not None:
                        hit, p, n, oid, t = eng.gpu_view.pick(sc, view['cam'], x, y, view['w'], view['h'], tuple(out))
                    else:
                        from ..render.raymarch import pick as _cpu_pick
                        r = _cpu_pick(sc, view['cam'], x, y, view['w'], view['h'], exclude=tuple(out)) \
                            if 'exclude' in _cpu_pick.__code__.co_varnames else None
                        if r is None:
                            if not out:
                                oid0 = vp._obj_at(pos)
                                if oid0 >= 0:
                                    out.append(oid0)
                            break
                        hit, oid = r.hit, r.oid
                except Exception:
                    break
                if not hit or oid is None or oid < 0 or oid in out:
                    break
                out.append(int(oid))
        return out

    def drawn_alpha(self, oid):
        vp = self.win.viewport
        xr = float(getattr(vp, 'xray', 0.0) or 0.0) if vp.mode in ('shaded', 'matcap') else 0.0
        return (1.0 - min(xr, 0.95)) * float(vp.obj_opacity.get(oid, 1.0))

    # ---- the click -------------------------------------------------------------------------------------------------------
    def click(self, pos):
        """-> the object id a plain click selects (-1: nothing)"""
        vp = self.win.viewport
        ck = tuple(round(float(v), 6) for v in list(vp.cam.pos) + list(vp.cam.target)) + (vp.width(), vp.height())
        st = self.state
        same = (st is not None and abs(st[0] - pos.x()) <= NEAR_PX and abs(st[1] - pos.y()) <= NEAR_PX and st[2] == ck
                and time.perf_counter() - st[5] < 30.0)
        back = self.key.held
        if same and st[3]:
            ch, i = st[3], st[4]
            i = (i - 1) % len(ch) if back else (i + 1) % len(ch)
        else:
            ch = self.chain(pos)
            if not ch:
                self.state = None
                self.tip.hide()
                return -1
            i = next((k for k, o in enumerate(ch) if self.drawn_alpha(o) >= SEE_THROUGH), 0)
        self.state = (pos.x(), pos.y(), ck, ch, i, time.perf_counter())
        self._say(pos, ch, i)
        return ch[i]

    def _name(self, oid):
        objs = {o['id']: o for o in (getattr(self.win, 'snap', None) or {}).get('objects', [])}
        return objs[oid]['name'] if oid in objs else 'object %d' % oid

    def _say(self, pos, ch, i):
        if len(ch) < 2:
            self.tip.hide()
            return
        nxt, prv = ch[(i + 1) % len(ch)], ch[(i - 1) % len(ch)]
        text = '%s  ·  %d of %d under the cursor     click again: %s     %s+click: back to %s     Esc: nothing' % (
            self._name(ch[i]), i + 1, len(ch), self._name(nxt), BACK_NAME, self._name(prv))
        self.tip.setText(text)
        self.tip.adjustSize()
        vp = self.win.viewport
        x = min(int(pos.x()) + 18, max(0, vp.width() - self.tip.width() - 6))
        y = min(int(pos.y()) + 22, max(0, vp.height() - self.tip.height() - 6))
        self.tip.move(x, y)
        self.tip.show()
        self.tip.raise_()
        self._hide.start(6000)
        try:
            self.win.status.showMessage(text, 6000)
        except Exception:
            pass


class _ListHover(QObject):
    """part 4 (fs-shell-2 11-14, cleared with part 3's guards): hovering a name in the Scene list outlines
    that object in the view through whatever covers it (solid where seen, dashed where covered, HOVER_COLOUR); the
    outline is the edge of the object's own ID image (gpu/viewmodes.vol_mask, ui/viewport._id_outline: the guards are
    written there). On the card only (the CPU view has no per-object ID image)."""

    def __init__(self, win, lst):
        super().__init__(lst)
        self.win, self.lst, self.cur = win, lst, None

    def set(self, oid):
        if oid == self.cur:
            return
        self.cur = oid
        vp = self.win.viewport
        vp.outline_hover = oid
        gv = getattr(self.win.engine, 'gpu_view', None)
        if gv is not None:
            gv.outline_hover = oid             # read by GpuView.frame on the engine thread (a plain attribute)
        vp.request(False)

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.MouseMove:
            it = self.lst.itemAt(ev.position().toPoint())
            self.set(int(it.data(Qt.UserRole)) if it is not None and it.data(Qt.UserRole) is not None else None)
        elif t in (QEvent.Leave, QEvent.HoverLeave, QEvent.Hide):
            self.set(None)
        return False


def install(win):
    si = SelectInside(win)
    win.select_inside = si
    lst = getattr(getattr(win, 'props', None), 'list', None)
    if lst is not None:
        lst.setMouseTracking(True)
        lst.viewport().setMouseTracking(True)
        si.list_hover = _ListHover(win, lst)
        lst.viewport().installEventFilter(si.list_hover)
    return si
