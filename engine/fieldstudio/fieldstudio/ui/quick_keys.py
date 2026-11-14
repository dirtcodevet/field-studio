"""The quick keys on the view (fs-shell-5, 2026-11-14; the first hands-on test: an on-screen
quick reference for the keys of common actions (hardness, copy, mirror, tools, etc.), and the keyboard map one button
click away on the top bar).

A card in the view's lower-left corner, shown while the user works and hidden with one click (the Quick keys button on the
top bar, the x on the card, or Panels / Help > Quick keys); remembered. It lists the common keys in plain words:
  the brush (size, strength, hardness, smooth while held, invert) while a brush is in hand,
  the tools for roughing in and the object tools,
  copy, mirror, undo, delete, hide,
  the view (orbit, pan, zoom, frame, front / side / top, X-ray) and search / all keys.
Every key is read from the keymap as it is now (a rebound key shows its new key; one with no key is left out).
Drawn once into a picture when the tool or the keys change; the view's frames only copy it (no text layout per frame).
The card itself lets the mouse through to the view; only its x takes a click."""
import time

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QToolButton, QWidget

MARGIN = 12
W = 318
TOP = 86             # fs-shell-6 11-14: the tool's badge (y 10, 26 px) and the transform strip under it (y 42, ~36 px):
                     # at 1366 x 768 the card covered the strip's Move / Rotate / Scale; a section that does not fit
                     # below it is left out whole (all keys: the Keyboard map, F1)
# (label, how the key is found): ('act', action id) | ('tool', tool key, default key) | ('text', fixed keys)
BRUSH = [('Size', [('act', 'Sculpt/Brush radius smaller'), ('act', 'Sculpt/Brush radius larger'), ('text', 'F-drag')]),
         ('Strength', [('act', 'Sculpt/Brush strength lower'), ('act', 'Sculpt/Brush strength higher'),
                       ('text', 'Shift+F-drag')]),
         ('Softness', [('act', 'Sculpt/Brush softer (softness / hardness)'), ('act', 'Sculpt/Brush harder (softness / hardness)'),
                       ('text', 'Ctrl+F-drag')]),
         ('Smooth while held', [('text', 'Shift')]), ('Invert (carve, dig)', [('text', 'Ctrl')]),
         ('Mirror X on / off', [('act', 'Sculpt/Mirror X on / off')])]
# the keys are the menu actions' (rebindable) keys; a registry row's live key first where it has one
TOOLS = [('Select', 'select'), ('Move', 'move'), ('Rotate', 'rotate'), ('Scale', 'scale'), ('Draw', 'draw'),
         ('Clay', 'clay'), ('Swell', 'act:Tools/The toolset/Swell / Settle'), ('Smooth', 'smooth'),
         ('Flatten', 'act:Tools/The toolset/Flatten'), ('Grab', 'grab'), ('Shape', 'stretch'),
         ('Pencil', 'act:Tools/Pencil ridge'), ('Mask', 'mask'), ('Cut', 'act:Tools/The toolset/Cut (a line, or a shape)'),
         ('Shapes', 'act:Tools/The toolset/Shape library (Add)'), ('Mirror copies', 'act:Tools/The toolset/Mirror (the copies tool)')]
EDIT = [('Undo', 'Edit/Undo'), ('Redo', 'Edit/Redo'), ('Copy', 'Edit/Duplicate object (with its sculpt)'),
        ('Mirror copy', 'Edit/Mirror-duplicate across X (world, with its sculpt)'), ('Delete', 'Edit/Delete object'),
        ('Hide', 'Edit/Hide / show selected (view only)'), ('Isolate', 'Edit/Isolate selected (again: show all)'),
        # fs-shell-6 11-14 (the eye workflow: group, label, drag back with keys and quick tools)
        ('Group', 'Tools/Group the selected (they select and move together)'), ('Ungroup', 'Tools/Ungroup the selected')]
VIEW = [('Orbit', [('text', 'Middle-drag')]), ('Pan', [('text', 'Shift+Middle')]), ('Zoom', [('text', 'Wheel')]),
        ('Frame', [('act', 'View/Frame selection')]), ('Front', [('act', 'View/Front view')]),
        ('Side', [('act', 'View/Right view')]), ('Top', [('act', 'View/Top view')]),
        ('X-ray', [('act', 'View/X-ray on / off (see through to the layers inside; Clay and Look)')]),
        ('Search', [('text', 'Space')]),
        ('All keys', [('act', 'Help/Keys window (every key, the current tool, a keyboard map)')])]
FLOW = ('Tools', 'Edit', 'View')     # key-word pairs that flow along the line (rows for the brush)
OBJECT_TOOLS = ('select', 'move', 'rotate', 'scale')


def _settings():
    from .main_window import app_settings
    return app_settings()


class QuickKeys(QWidget):
    def __init__(self, win):
        super().__init__(win.viewport)
        self.win = win
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._pm = None
        self._sig = None
        self._keys_t = 0.0
        self._acts = {}
        self.close_btn = QToolButton(win.viewport)
        self.close_btn.setText('x')
        self.close_btn.setToolTip('Hide the quick keys (the Quick keys button on the top bar shows them again)')
        self.close_btn.setStyleSheet('QToolButton { color: #e6e9ee; background: #2a303a; border: 1px solid #56606e; '
                                     'border-radius: 4px; font-weight: 700; padding: 0px 6px; }'
                                     'QToolButton:hover { background: #3b6fd8; color: white; }')
        self.close_btn.setFixedSize(QSize(22, 22))
        self.close_btn.clicked.connect(lambda: self.set_shown(False))
        try:
            self.shown = str(_settings().value('quick_keys_shown', 'true')).lower() == 'true'
        except Exception:
            self.shown = True
        win.viewport.installEventFilter(self)
        win.tools.toolChanged.connect(lambda _k: self.refresh())
        self._reg_seen = None
        from PySide6.QtCore import QTimer
        self._poll = QTimer(self)                      # a registry row changes win.reg_tool, not the brush: follow it
        self._poll.timeout.connect(self._follow_reg)
        self._poll.start(300)
        self.refresh()

    # ---- shown / hidden ------------------------------------------------------------------------------------------------
    def set_shown(self, on):
        self.shown = bool(on)
        try:
            _settings().setValue('quick_keys_shown', 'true' if self.shown else 'false')
        except Exception:
            pass
        for a in (getattr(self.win, 'act_quick_keys', None),):
            if a is not None and a.isChecked() != self.shown:
                a.blockSignals(True)
                a.setChecked(self.shown)
                a.blockSignals(False)
        b = getattr(getattr(self.win, 'topbar', None), 'quick_btn', None)
        if b is not None and b.isChecked() != self.shown:
            b.blockSignals(True)
            b.setChecked(self.shown)
            b.blockSignals(False)
        self.refresh()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize and self.shown:
            self.refresh()                     # its height follows the view's (a short view drops the last sections)
        return False

    def _follow_reg(self):
        r = getattr(self.win, 'reg_tool', None)
        if r != self._reg_seen:
            self._reg_seen = r
            if self.shown:
                self.refresh()

    # ---- the keys as the keymap has them now ------------------------------------------------------------------------
    def _key_of_act(self, aid):
        a = self._acts.get(aid)
        if a is None or a.shortcut().isEmpty():
            return ''
        from . import keymap as KM
        return KM.native(KM.norm(a.shortcut()))

    def _tool_key(self, ref):
        win = self.win
        try:
            if ref.startswith('act:'):
                return self._key_of_act(ref[4:])
            if ref.startswith('reg:'):
                rk = (getattr(win, 'live_reg_keys', {}) or {}).get(ref)
                if rk:
                    return rk[1]
                from . import toolreg as TR
                R = TR.registry()
                sp = R.BY_KEY.get(ref[4:]) if R is not None else None
                return getattr(sp, 'key_hint', '') or ''
            from . import keymap as KM
            ta = (getattr(win, 'tool_actions', {}) or {}).get(ref)
            if ta is not None and not ta.shortcut().isEmpty():
                return KM.native(KM.norm(ta.shortcut()))
            return KM.tool_key(win, ref, '') or ''
        except Exception:
            return ''

    def _rows(self):
        now = time.perf_counter()
        if not self._acts or now - self._keys_t > 30.0:
            from . import keymap as KM
            self._acts = KM.actions(self.win)
            self._keys_t = now
        tool = self.win.tools.tool
        brush = tool not in OBJECT_TOOLS and getattr(self.win.viewport, 'tool_mode', None) is None
        sections = []

        def keys(spec):
            ks = []
            for kind, v in spec:
                k = self._key_of_act(v) if kind == 'act' else v
                if k:
                    ks.append(k)
            return ks
        try:
            from . import quick_settings as QS
            tname, slots = QS.describe(self.win)
        except Exception:
            tname, slots = None, []
        if tname is not None and slots:                # fs-shell-5 11-14: a registry tool's own three, same keys
            sections.append((tname, [(lab, keys(BRUSH[i][1])) for lab, i in slots]))
        elif brush:
            sections.append(('Brush', [(lab, keys(sp)) for lab, sp in BRUSH]))
        tl = []
        for lab, ref in TOOLS:
            k = self._tool_key(ref)
            if k:
                tl.append((k, lab))
        sections.append(('Tools', tl))
        sections.append(('Edit', [(self._key_of_act(aid), lab) for lab, aid in EDIT if self._key_of_act(aid)]))
        sections.append(('View', [(keys(sp)[0], lab) for lab, sp in VIEW if keys(sp)]))
        return sections

    # ---- the picture --------------------------------------------------------------------------------------------------
    def refresh(self):
        if not self.shown:
            self.hide()
            self.close_btn.hide()
            return
        try:
            secs = self._rows()
        except Exception as e:
            print('quick keys: %r' % (e,))
            return
        avail_h = max(160, self.win.viewport.height() - TOP - MARGIN - 34)
        sig = (repr(secs), avail_h)
        if sig != self._sig or self._pm is None:
            self._sig = sig
            self._pm = self._draw(secs, avail_h)
        self.resize(self._pm.size())
        self.place()
        self.show()
        self.raise_()
        self.close_btn.show()
        self.close_btn.raise_()
        self.update()

    def _layout(self, secs, fm_r, fm_k, limit_h, pad, row_h, head_h):
        """-> [(kind, payload, x, y)] and the height used; a section that does not fit is left out whole"""
        items, y = [], pad + 26
        for title, rows in secs:
            sec, yy = [('head', title, pad, y)], y + head_h
            if title in FLOW:
                x = pad
                for k, lab in rows:
                    w_ = fm_k.horizontalAdvance(k) + 12 + 4 + fm_r.horizontalAdvance(lab) + 10
                    if x + w_ > W - pad and x > pad:
                        x = pad
                        yy += row_h
                    sec.append(('pair', (k, lab), x, yy))
                    x += w_
                yy += row_h
            else:
                for lab, ks in rows:
                    sec.append(('row', (lab, ks), pad, yy))
                    yy += row_h
            if yy + pad > limit_h and items:
                break
            items += sec
            y = yy + 2
        return items, y + pad

    def _draw(self, secs, avail_h):
        f_head = QFont('Segoe UI')
        f_head.setPixelSize(14)
        f_head.setBold(True)
        f_row = QFont('Segoe UI')
        f_row.setPixelSize(13)
        f_key = QFont('Segoe UI')
        f_key.setPixelSize(12)
        f_key.setBold(True)
        fm_r, fm_k = QFontMetrics(f_row), QFontMetrics(f_key)
        row_h, head_h, pad = 21, 22, 10
        items, h = self._layout(secs, fm_r, fm_k, avail_h, pad, row_h, head_h)
        h = int(min(h, avail_h))
        pm = QPixmap(W, h)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(86, 96, 110), 1))
        p.setBrush(QColor(18, 20, 25, 228))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, h - 1), 7, 7)
        p.setFont(f_head)
        p.setPen(QColor(244, 246, 249))
        p.drawText(QPointF(pad, pad + 15), 'Quick keys')
        for kind, pl, x, y in items:
            if kind == 'head':
                p.setFont(f_head)
                p.setPen(QColor(255, 210, 120))
                p.drawText(QPointF(x, y + 16), pl)
            elif kind == 'pair':
                k, lab = pl
                kw = self._cap(p, f_key, fm_k, k, x, y)
                p.setFont(f_row)
                p.setPen(QColor(232, 236, 241))
                p.drawText(QPointF(x + kw + 4, y + 14), lab)
            else:
                lab, ks = pl
                p.setFont(f_row)
                p.setPen(QColor(232, 236, 241))
                p.drawText(QPointF(x, y + 14), lab)
                xx = 150
                for k in ks:
                    if xx + fm_k.horizontalAdvance(k) + 12 > W - pad:
                        break
                    xx += self._cap(p, f_key, fm_k, k, xx, y) + 5
        p.end()
        return pm

    @staticmethod
    def _cap(p, f_key, fm_k, k, x, y):
        w = fm_k.horizontalAdvance(k) + 12
        p.setPen(QPen(QColor(120, 132, 150), 1))
        p.setBrush(QColor(44, 50, 60))
        p.drawRoundedRect(QRectF(x, y + 1, w, 18), 4, 4)
        p.setFont(f_key)
        p.setPen(QColor(156, 206, 255))
        p.drawText(QRectF(x, y + 1, w, 18), Qt.AlignCenter, k)
        return w

    def place(self):
        vp = self.win.viewport
        x, y = MARGIN, max(TOP, vp.height() - self.height() - MARGIN - 34)
        self.move(x, y)
        self.close_btn.move(x + self.width() - self.close_btn.width() - 6, y + 6)

    def paintEvent(self, e):
        if self._pm is None:
            return
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pm)
        p.end()


def install(win):
    qk = QuickKeys(win)
    win.quick_keys = qk
    return qk
