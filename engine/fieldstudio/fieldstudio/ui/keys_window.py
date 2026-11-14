"""The Keys window (fs-shell, 2026-11-11; design 4.2: a pop-out window that can be
dragged around or minimized).

F1 or the Keys button on the status bar opens it. A frameless tool window dragged by its title strip anywhere (a
second monitor too), resizes from its corner, collapses to the strip (_) or closes (x); dragged near an edge of the main
window it snaps to that edge. Its place, size and state are remembered.
Inside:
- Now: what each key and modifier does with the current tool; it follows the tool.
- A search box over every key.
- List: every key by category; keys already used get a tick (Figma's learning cue).
- Keyboard: a keyboard map, keys coloured by category, free keys grey; hover names the action.
Holding ? shows the same sheet as a translucent overlay over the view while held (PowerToys' peek).
Rebinding (design 4.2): right-click a key in the list or on the keyboard map: Rebind..., Add another key..., Remove this
key, Restore default, Restore all keys (ui/keymap.py; conflicts shown before anything changes; saved in the settings)."""
import json

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QSizeGrip,
                               QStackedWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QButtonGroup)

from . import glyphs as GL
from . import keymap as KM

CAT_ORDER = ['Now', 'Brushes', 'Object', 'Field tools', 'View', 'Edit', 'File', 'Sculpt', 'Tools', 'Mouse', 'Help']
CAT_COLOUR = {'Brushes': (232, 140, 60), 'Object': (150, 162, 180), 'Field tools': (226, 84, 84),
              'View': (64, 186, 170), 'Edit': (80, 140, 245), 'File': (160, 120, 230), 'Sculpt': (226, 190, 70),
              'Tools': (110, 200, 100), 'Help': (140, 150, 160), 'Mouse': (200, 200, 200)}
MOUSE = [('Left-drag', 'use the tool (with a brush it starts on the form and never orbits)'),
         ('Middle-drag', 'orbit'), ('Alt+Left-drag', 'orbit'), ('Shift+Middle-drag', 'pan'),
         ('Right-drag', 'pan'), ('Right-click', 'action menu'), ('Wheel', 'zoom toward the cursor'),
         ('F-drag', 'brush radius'), ('Shift+F-drag', 'brush strength'), ('Ctrl+F-drag', 'brush hardness'),
         ('Space', 'search tools and actions'),
         ('Hold ?', 'peek at this sheet')]
NOW = {
    '_brush': [('Left-drag', '{name} (starts on the form, never orbits)'), ('Shift', 'smooth while held'),
               ('Ctrl', 'invert'), ('F-drag / [ ]', 'radius'), ('Shift+F-drag / - =', 'strength, 0 to 2'),
               ('Ctrl+F-drag / Shift+[ ]', 'hardness, soft to hard'),
               ('Middle / Alt+Left-drag', 'orbit'), ('Shift+Middle-drag', 'pan'), ('Q', 'back to Select')],
    'draw': [('Ctrl', 'carve instead')], 'carve': [('Ctrl', 'draw instead')], 'mask': [('Ctrl', 'erase the mask')],
    'inflate': [('Ctrl', 'deflate')], 'clay': [('Ctrl', 'dig')],
    'select': [('Left-click', 'select (Shift adds)'), ('Left-drag on empty space', 'orbit'),
               ('H / Shift+H', 'hide / show all'),
               ('/', 'isolate'), ('Ctrl+D', 'duplicate'), ('Delete', 'delete')],
    'move': [('Left-drag', 'move the object (an arrow: along that axis)'), ('X / Y / Z (dragging)', 'lock to that axis'),
             ('Type a number (dragging)', 'exact mm; Enter ends, Esc puts it back'), ('Ctrl (dragging)', 'snap 1 mm'),
             ('Ring icons below', 'bend, twist, taper, inflate: drag left / right'), ('Alt+Left-drag', 'orbit'),
             ('Q', 'back to Select')],
    'rotate': [('Left-drag', 'turn the object (a ring: about that axis)'), ('X / Y / Z (dragging)', 'about that axis'),
               ('Type a number (dragging)', 'exact degrees; Enter / Esc'), ('Ctrl (dragging)', 'snap 5 deg'),
               ('Pivot', 'object centre or a picked surface point (context bar)'), ('Alt+Left-drag', 'orbit'),
               ('Q', 'back to Select')],
    'scale': [('Left-drag', 'grow / shrink (a box: stretch that axis)'), ('Shift on a box', 'skew'),
              ('X / Y / Z (dragging)', 'stretch that axis'), ('Type a number (dragging)', 'exact factor; Enter / Esc'),
              ('Ctrl (dragging)', 'snap 5 %'), ('Q', 'back to Select')],
}
KEYBOARD = [['Esc', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12'],
            ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Backspace'],
            ['Tab', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
            ['Caps', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", 'Return'],
            ['Shift', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'Shift'],
            ['Ctrl', 'Alt', 'Space', 'Alt', 'Menu', 'Ctrl', 'Home', 'Delete']]
WIDE = {'Backspace': 2.0, 'Tab': 1.5, '\\': 1.5, 'Caps': 1.75, 'Return': 2.25, 'Shift': 2.25, 'Ctrl': 1.5, 'Alt': 1.25,
        'Space': 6.0, 'Menu': 1.25, 'Home': 1.25, 'Delete': 1.25}


def collect(win):
    """-> [(category, key text, action name, action id or None)] from the menus (every QAction with a shortcut) and
    the tools; the tools' keys are read from their actions, so a rebinding shows at once"""
    from .panels import TOOLS
    from .shell import menu_actions, _clean, action_id
    out = []
    obj = {'select', 'move', 'rotate', 'scale'}
    tool_names = [n for _, n, _ in TOOLS]
    for key, name, sc in TOOLS:
        k = KM.tool_key(win, key, sc)
        if k:
            out.append(('Object' if key in obj else 'Brushes', k, name, 'Sculpt/' + name))
    seen = set((c, k) for c, k, _, _ in out)
    for path, a in menu_actions(win):
        top = path.split('/')[0]
        aid = action_id(path, a)                   # fs-shell-5 11-14: stable across ui/menus.py's renames
        for ks in a.shortcuts():
            k = ks.toString(QKeySequence.NativeText)
            if not k:
                continue
            cat = 'Brushes' if top == 'Sculpt' and _clean(a.text()) in tool_names else top
            if top == 'Tools' and any(w in _clean(a.text()) for w in ('Pencil', 'Cut', 'Splice', 'Fill')):
                cat = 'Field tools'
            if (cat, k) in seen:
                continue
            seen.add((cat, k))
            out.append((cat, k, _clean(a.text()), aid))
    for k, what in MOUSE:
        out.append(('Mouse', k, what, None))
    return out


class _Strip(QWidget):
    """the title strip: drag moves the window, double-click collapses"""

    def __init__(self, kw):
        super().__init__(kw)
        self.kw = kw
        self._off = None
        self.setFixedHeight(26)
        self.setCursor(Qt.SizeAllCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._off = e.globalPosition().toPoint() - self.kw.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._off is not None:
            self.kw.move(e.globalPosition().toPoint() - self._off)

    def mouseReleaseEvent(self, e):
        if self._off is not None:
            self._off = None
            self.kw.snap()
            self.kw.save_state()

    def mouseDoubleClickEvent(self, e):
        self.kw.toggle_collapse()


class KeyMap(QWidget):
    """the keyboard map: keys coloured by category, free keys grey, hover names the action"""

    def __init__(self, kw):
        super().__init__(kw)
        self.kw = kw
        self.setMouseTracking(True)
        self.setMinimumHeight(190)
        self.rects = []

    def bound(self):
        m = {}
        for cat, k, name, aid in self.kw.entries:
            if cat == 'Mouse':
                continue
            base = k.split('+')[-1] if '+' in k and len(k.split('+')[-1]) > 0 else k
            mods = k[:-len(base)].rstrip('+') if base != k else ''
            m.setdefault(base.upper() if len(base) == 1 else base, []).append((cat, (mods + '+' if mods else '') + base,
                                                                              name, aid))
        return m

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(29, 32, 38))
        bound = self.bound()
        unit = (self.width() - 16) / 15.5
        kh = min(30.0, (self.height() - 16) / 6.4)
        self.rects = []
        f = QFont('Segoe UI', 7)
        p.setFont(f)
        q = self.kw.search.text().strip().lower()
        for r, row in enumerate(KEYBOARD):
            x = 8.0
            y = 8.0 + r * (kh + 4)
            for k in row:
                w = WIDE.get(k, 1.0) * unit - 3
                rc = QRectF(x, y, w, kh)
                acts = bound.get(k.upper() if len(k) == 1 else k, [])
                if acts:
                    c = QColor(*CAT_COLOUR.get(acts[0][0], (140, 150, 160)))
                    if q and not any(q in n.lower() or q in kk.lower() for _, kk, n, _a in acts):
                        c.setAlpha(60)
                    p.setBrush(c.darker(160))
                    p.setPen(QPen(c, 1.2))
                else:
                    p.setBrush(QColor(44, 47, 54))
                    p.setPen(QPen(QColor(60, 64, 72), 1))
                p.drawRoundedRect(rc, 3, 3)
                p.setPen(QColor(235, 235, 235) if acts else QColor(110, 115, 125))
                p.drawText(rc.adjusted(3, 1, -2, -1), Qt.AlignLeft | Qt.AlignTop, k)
                if acts:
                    p.setPen(QColor(200, 200, 200))
                    p.drawText(rc.adjusted(3, 0, -2, -2), Qt.AlignLeft | Qt.AlignBottom,
                               acts[0][2][:max(3, int(w / 5.5))])
                self.rects.append((rc, k, acts))
                x += w + 3
        p.end()

    def mouseMoveEvent(self, e):
        for rc, k, acts in self.rects:
            if rc.contains(e.position()):
                self.setToolTip('<b>%s</b><br>%s<br><i>right-click to rebind</i>' % (
                    k, '<br>'.join('%s: %s' % (kk, n) for _, kk, n, _a in acts) or 'free'))
                return
        self.setToolTip('')

    def contextMenuEvent(self, e):
        """right-click a key: rebind what it runs, or bind a free key to an action"""
        for rc, k, acts in self.rects:
            if rc.contains(QPointF(e.pos())):
                m = QMenu(self)
                for _c, kk, n, aid in acts:
                    if aid:
                        m.addAction('Rebind %s (%s)...' % (n, kk), lambda a=aid: self.kw.rebind(a))
                if k in ('Shift', 'Ctrl', 'Alt', 'Menu', 'Caps', 'Space', 'F'):
                    m.addAction('%s is fixed (%s)' % (k, KM.RESERVED.get(k, 'a modifier'))).setEnabled(False)
                else:
                    m.addAction('Bind %s to an action...' % k, lambda kk=k: self.kw.bind_free(kk))
                m.addSeparator()
                m.addAction('Restore all keys to their defaults', self.kw.restore_all)
                m.exec(e.globalPos())
                return


class KeysWindow(QWidget):
    def __init__(self, win):
        super().__init__(win, Qt.Tool | Qt.FramelessWindowHint)
        self.win = win
        self.setObjectName('keyswin')
        self.setStyleSheet('#keyswin { background: #1d2026; border: 1px solid #3b82f6; }')
        self.used = set(json.loads(_settings().value('keys_used', '[]') or '[]'))
        self.collapsed = False
        self.full_h = 560
        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)
        self.strip = _Strip(self)
        sl = QHBoxLayout(self.strip)
        sl.setContentsMargins(8, 0, 2, 0)
        t = QLabel('Keys')
        t.setStyleSheet('font-weight: 600;')
        sl.addWidget(t)
        self.now_short = QLabel('')
        self.now_short.setObjectName('dim')
        sl.addWidget(self.now_short, 1)
        for txt, tip, fn in (('_', 'Collapse to this strip (double-click the strip too)', self.toggle_collapse),
                             ('x', 'Close (F1 opens it again)', self.close)):
            b = QPushButton(txt)
            b.setFixedSize(24, 22)
            b.setStyleSheet('padding: 0px; font-weight: 600;')
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(fn)
            sl.addWidget(b)
        lay.addWidget(self.strip)
        self.body = QWidget()
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(6)
        self.now = QLabel('')
        self.now.setWordWrap(True)
        self.now.setTextFormat(Qt.RichText)
        self.now.setStyleSheet('background: #23272e; border-radius: 4px; padding: 6px;')
        bl.addWidget(self.now)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search keys ("smooth", "mirror", "Ctrl")')
        self.search.textChanged.connect(self.refill)
        row.addWidget(self.search, 1)
        self.views = QButtonGroup(self)
        for i, name in enumerate(('List', 'Keyboard')):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setFocusPolicy(Qt.NoFocus)
            self.views.addButton(b, i)
            row.addWidget(b)
        self.views.idClicked.connect(lambda i: (self.stack.setCurrentIndex(i), self.save_state()))
        bl.addLayout(row)
        self.stack = QStackedWidget()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Key', 'Does', ''])
        self.tree.setColumnWidth(0, 120)
        self.tree.setColumnWidth(1, 250)
        self.tree.setRootIsDecorated(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.setToolTip('Right-click a key to rebind it')
        self.stack.addWidget(self.tree)
        self.map = KeyMap(self)
        self.stack.addWidget(self.map)
        bl.addWidget(self.stack, 1)
        grip = QSizeGrip(self)
        gl = QHBoxLayout()
        gl.addStretch(1)
        gl.addWidget(grip)
        bl.addLayout(gl)
        lay.addWidget(self.body, 1)
        self.entries = []
        self.resize(460, self.full_h)
        win.tools.toolChanged.connect(lambda _k: self.update_now())
        self.refresh()
        self.restore_state()

    # ---- content ---------------------------------------------------------------------------------------------------
    def refresh(self):
        self.entries = collect(self.win)
        self.update_now()
        self.refill()

    def update_now(self):
        tp = self.win.tools
        key = tp.tool
        from .panels import TOOLS
        name = dict((k, n) for k, n, _ in TOOLS).get(key, key)
        brush = key not in ('select', 'move', 'rotate', 'scale')
        own = NOW.get(key, [])
        own_d = dict(own)
        rows = [(k, own_d.pop(k) if k in own_d else w.format(name=name))       # a tool's own Ctrl takes 'invert's place
                for k, w in (NOW['_brush'] if brush else [])] + [(k, w) for k, w in own if k in own_d]
        if tp.field_active:
            rows = [('Left-drag', 'use %s' % tp.field_active), ('Esc', 'leave the field tool'),
                    ('Alt+Left-drag', 'orbit')]
            name = tp.field_active
        col = QColor(*GL.CAT_COL.get(GL.category_of(key), (140, 150, 160))).name()
        self.now.setText('<b style="color:%s">Now: %s</b><br>%s' % (
            col, name, '<br>'.join('<span style="color:#e5e7eb">%s</span> &nbsp; %s' % (k, w) for k, w in rows)))
        self.now_short.setText('  Now: %s' % name)

    def refill(self, *_):
        q = self.search.text().strip().lower()
        self.tree.clear()
        cats = {}
        for cat, k, name, aid in self.entries:
            if q and q not in k.lower() and q not in name.lower() and q not in cat.lower():
                continue
            cats.setdefault(cat, []).append((k, name, aid))
        # fs-shell-5 11-14 (t68: opening this window froze the window 0.53 s): KM.actions() walks every menu
        # action; it was called twice for every row (n x 240). Once per fill now
        acts = KM.actions(self.win)
        self.tree.setUpdatesEnabled(False)
        for cat in sorted(cats, key=lambda c: CAT_ORDER.index(c) if c in CAT_ORDER else 99):
            top = QTreeWidgetItem([cat, '', ''])
            c = QColor(*CAT_COLOUR.get(cat, (140, 150, 160)))
            top.setForeground(0, c)
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self.tree.addTopLevelItem(top)
            for k, name, aid in cats[cat]:
                a_ = acts.get(aid) if aid else None
                note = a_.property('keys_note') if a_ is not None else None   # fs-shell-3 11-14: a key's why, when
                it = QTreeWidgetItem([k, ('%s  ·  %s' % (name, note)) if note else name, '✓' if k in self.used else ''])
                if note:
                    it.setToolTip(1, str(note))
                it.setData(0, Qt.UserRole, aid)
                it.setData(1, Qt.UserRole, k)
                if k in self.used:
                    it.setForeground(0, QColor(176, 184, 195))          # fs-shell-5: was (140, 145, 155), 5:1
                if aid:
                    a = a_
                    if a is not None and KM.current(a) != KM.default(a):
                        it.setForeground(1, QColor(251, 191, 36))      # changed from the default: amber
                        it.setToolTip(1, 'Rebound (default: %s)' % (
                            ', '.join(KM.native(x) for x in KM.default(a)) or 'none'))
                top.addChild(it)
            top.setExpanded(True)
        self.tree.setUpdatesEnabled(True)
        self.map.update()

    # ---- rebinding (design 4.2) -------------------------------------------------------------------------------------
    def _tree_menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None or it.parent() is None:
            return
        aid, k = it.data(0, Qt.UserRole), it.data(1, Qt.UserRole)
        m = QMenu(self)
        if not aid:
            m.addAction('Mouse buttons and held keys are fixed').setEnabled(False)
        else:
            a = KM.actions(self.win).get(aid)
            m.addAction('Rebind...', lambda: self.rebind(aid))
            m.addAction('Add another key...', lambda: self.rebind(aid, add=True))
            if a is not None and len(KM.current(a)) > 1:
                m.addAction('Remove %s' % k, lambda: KM.remove_key(self.win, aid, QKeySequence(k).toString(KM.PT)))
            ra = m.addAction('Restore default (%s)' % (', '.join(KM.native(x) for x in KM.default(a)) or 'none'),
                             lambda: KM.restore(self.win, aid))
            ra.setEnabled(a is not None and KM.current(a) != KM.default(a))
        m.addSeparator()
        m.addAction('Restore all keys to their defaults', self.restore_all)
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def rebind(self, aid, add=False):
        d = KM.RebindDialog(self.win, aid, self, add=add)
        self.rebind_dialog = d
        return d.exec()

    def bind_free(self, key):
        d = KM.ActionPicker(self.win, key, self)
        self.rebind_dialog = d
        return d.exec()

    def restore_all(self):
        KM.restore_all(self.win)

    def keys_changed(self):
        self.refresh()

    def mark_used(self, key_text):
        if key_text and key_text not in self.used:
            self.used.add(key_text)
            _settings().setValue('keys_used', json.dumps(sorted(self.used)))
            if self.isVisible():
                self.refill()

    # ---- window behaviour ----------------------------------------------------------------------------------------
    def toggle_collapse(self):
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.full_h = max(self.height(), 200)
            self.body.hide()
            self.setFixedHeight(self.strip.height() + 2)
        else:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.body.show()
            self.resize(self.width(), self.full_h)
        self.save_state()

    def snap(self, reach=36):
        """dragged near an edge of the main window: snap to it (docked to that edge)"""
        g = self.win.frameGeometry()
        r = self.frameGeometry()
        x, y = r.x(), r.y()
        if abs(r.left() - g.left()) < reach:
            x = g.left()
        elif abs(r.right() - g.right()) < reach:
            x = g.right() - r.width() + 1
        if abs(r.top() - g.top()) < reach:
            y = g.top()
        elif abs(r.bottom() - g.bottom()) < reach:
            y = g.bottom() - r.height() + 1
        self.move(x, y)
        self.docked = (x in (g.left(), g.right() - r.width() + 1)) or (y in (g.top(), g.bottom() - r.height() + 1))

    def save_state(self):
        s = _settings()
        s.setValue('keys_window/geometry', json.dumps([self.x(), self.y(), self.width(),
                                                       self.full_h if self.collapsed else self.height()]))
        s.setValue('keys_window/collapsed', 'true' if self.collapsed else 'false')
        s.setValue('keys_window/view', str(self.stack.currentIndex()))

    def restore_state(self):
        s = _settings()
        try:
            x, y, w, h = json.loads(s.value('keys_window/geometry', '') or 'null') or [None] * 4
        except Exception:
            x = None
        if x is not None:
            self.move(int(x), int(y))
            self.resize(int(w), int(h))
            self.full_h = int(h)
        else:
            g = self.win.frameGeometry()
            self.move(g.right() - self.width() - 380, g.top() + 120)
        v = int(s.value('keys_window/view', '0') or 0)
        self.stack.setCurrentIndex(v)
        self.views.button(v).setChecked(True)
        if str(s.value('keys_window/collapsed', 'false')).lower() == 'true':
            self.toggle_collapse()

    def moveEvent(self, e):
        super().moveEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self.collapsed and self.isVisible():
            self.save_state()

    def closeEvent(self, e):
        self.save_state()
        super().closeEvent(e)

    def toggle(self):
        if self.isVisible():
            self.close()
        else:
            self.refresh()
            self.show()
            self.raise_()


class Peek(QWidget):
    """the hold-? overlay: the Now section and the list, translucent over the view while ? is held"""

    def __init__(self, win, kw):
        super().__init__(win.viewport)
        self.win, self.kw = win, kw
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.hide()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 18, 22, 205))
        p.setPen(QColor(235, 235, 235))
        f = QFont('Segoe UI', 9)
        p.setFont(f)
        cols = 3
        cw = (self.width() - 40) / cols
        x0, y = 20.0, 24.0
        p.drawText(QPointF(x0, y), 'Keys (release ? to close; F1 keeps the window open)')
        y += 22
        cats = {}
        for cat, k, name, _aid in self.kw.entries:
            cats.setdefault(cat, []).append((k, name))
        col, yy = 0, y
        for cat in sorted(cats, key=lambda c: CAT_ORDER.index(c) if c in CAT_ORDER else 99):
            rows = cats[cat]
            need = 18 * (len(rows) + 1)
            if yy + need > self.height() - 10 and yy > y:
                col += 1
                yy = y
            if col >= cols:
                break
            x = x0 + col * cw
            p.setPen(QColor(*CAT_COLOUR.get(cat, (140, 150, 160))))
            p.drawText(QPointF(x, yy), cat)
            yy += 18
            for k, name in rows:
                p.setPen(QColor(240, 240, 240))
                p.drawText(QPointF(x, yy), k)
                p.setPen(QColor(170, 175, 185))
                p.drawText(QPointF(x + 92, yy), name[:38])
                yy += 16
            yy += 6
        p.end()


class KeysFilter(QObject):
    """F1 opens / closes the Keys window (the menu action does it); hold ? peeks; every shortcut used is ticked"""

    def __init__(self, win, kw, peek):
        super().__init__(win)
        self.win, self.kw, self.peek = win, kw, peek

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.KeyPress, QEvent.KeyRelease) and ev.key() == Qt.Key_Question and not ev.isAutoRepeat():
            if QApplication.activeWindow() is not self.win:
                return False
            if t == QEvent.KeyPress:
                self.kw.entries = self.kw.entries or collect(self.win)
                self.peek.setGeometry(self.win.viewport.rect())
                self.peek.show()
                self.peek.raise_()
            else:
                self.peek.hide()
            return True
        return False


def _settings():
    from .main_window import app_settings
    return app_settings()


def install(win):
    kw = KeysWindow(win)
    win.keys_window = kw
    peek = Peek(win, kw)
    win.keys_peek = peek
    win.keys_filter = KeysFilter(win, kw, peek)
    QApplication.instance().installEventFilter(win.keys_filter)
    # every menu action with a key ticks its key when used (the key it has then, so rebinding keeps this right); a
    # tool pick ticks the tool's key
    from .shell import menu_actions
    for _path, a in menu_actions(win):
        a.triggered.connect(lambda _=False, act=a: kw.mark_used(act.shortcut().toString(QKeySequence.NativeText)))
    win.tools.toolChanged.connect(lambda k: kw.mark_used(KM.tool_key(win, k)))
    b = QPushButton('Keys (F1)')
    b.setToolTip('The Keys window: what every key does, what the current tool does, a keyboard map')
    b.setFocusPolicy(Qt.NoFocus)
    b.clicked.connect(kw.toggle)
    win.statusBar().addPermanentWidget(b)
    win.keys_button = b
    return kw
