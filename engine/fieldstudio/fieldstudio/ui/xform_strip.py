"""The transform strip (fs-shell, 2026-11-14; Tier 0 part 5).

Found 11-13: how to rotate or manipulate an object in FS was not clear. The handles (ui/gizmo.py) appear only
with the Move / Rotate / Scale tools (W / R / Y), and the window opens on Select (Q): a click on a form selected it and
nothing on the screen said what to do next.

What it is: whenever a form is selected (with Select, Move, Rotate or Scale), a strip at the top left of the view,
under the tool's badge, names it and offers Move, Rotate and Scale as labelled buttons with their keys; the one in use
is lit. A click picks that tool, and the gizmo's handles appear on the form. Nothing else changes: the drags, keys,
typed values and the handles are the gizmo's own.
"""
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton

from . import glyphs as GL

TOOLS = (('move', 'Move', 'W'), ('rotate', 'Rotate', 'R'), ('scale', 'Scale', 'Y'))


class XformStrip(QFrame):
    def __init__(self, win):
        super().__init__(win.viewport)
        self.win = win
        self.setObjectName('xformStrip')
        self.setStyleSheet('#xformStrip { background: rgba(20, 22, 26, 215); border-radius: 5px; }'
                           'QToolButton { color: #e8ecf2; padding: 3px 8px; border-radius: 4px; }'
                           'QToolButton:hover { background: #2e3440; }'
                           'QToolButton:checked { background: #3b6fd8; color: white; }')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 6, 3)
        lay.setSpacing(4)
        self.name = QLabel('')
        self.name.setStyleSheet('color: #c8ced6;')
        lay.addWidget(self.name)
        self.btn = {}
        for key, label, sc in TOOLS:
            b = QToolButton()
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            try:
                b.setIcon(GL.glyph_icon(key, 18))
                b.setIconSize(QSize(18, 18))
            except Exception:
                pass
            b.setToolTip({'move': 'Move the selected form: drag an arrow to move along it, the centre or the form to '
                                  'move in the view plane',
                          'rotate': 'Rotate the selected form: drag a ring to turn it about that axis',
                          'scale': 'Scale the selected form: drag a box to stretch one axis, the centre to scale evenly'}[key]
                         + '. While dragging: X / Y / Z lock an axis, type a number for an exact value, Ctrl snaps.')
            b.clicked.connect(lambda _=False, k=key: self._pick(k))
            lay.addWidget(b)
            self.btn[key] = (b, label, sc)
        self._sig = None
        self.hide()
        self._t = QTimer(self)
        self._t.timeout.connect(self.refresh)
        self._t.start(150)

    def _pick(self, key):
        tp = self.win.tools
        tp.set_tool('select' if tp.tool == key else key)     # a second click on the lit one goes back to Select
        self.refresh()
        self.win.viewport.setFocus()

    def _names(self, oids):
        objs = {o['id']: o for o in (getattr(self.win, 'snap', None) or {}).get('objects', [])}
        n = [objs[o]['name'] for o in oids if o in objs]
        if not n:
            return ''
        return n[0] if len(n) == 1 else '%s +%d' % (n[0], len(n) - 1)

    def refresh(self):
        vp = self.win.viewport
        try:
            from .keymap import tool_key
            keys = tuple(tool_key(self.win, k, sc) for k, _l, sc in TOOLS)
        except Exception:
            keys = tuple(sc for _k, _l, sc in TOOLS)
        show = (bool(vp.selected) and vp.tool in ('select', 'move', 'rotate', 'scale') and vp.tool_mode is None
                and getattr(vp, 'warm', True))
        sig = (show, tuple(vp.selected), vp.tool, keys, self._names(vp.selected) if show else '')
        if sig == self._sig:
            return
        self._sig = sig
        if not show:
            self.hide()
            return
        self.name.setText(sig[4] + ':')
        for (key, (b, label, _sc)), k in zip(self.btn.items(), keys):
            b.setText('%s  %s' % (label, k) if k else label)
            b.setChecked(vp.tool == key)
        self.adjustSize()
        self.move(12, 42)                # under the tool's badge (ui/viewport.py _draw_tool_badge: y 10, height 26)
        self.show()
        self.raise_()


def install(win):
    win.xform_strip = XformStrip(win)
    return win.xform_strip
