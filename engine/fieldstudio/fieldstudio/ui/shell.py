"""The shell (fs-shell, 2026-11-11; design 4.1 and 5.1): every tool findable.

- One registry (items()): the brushes and object tools, the field tools and every menu action, each with its glyph or
  category colour, key, one-line "what it does" and search words.
- ToolShelf: the tools by category (Object, Build, Refine, Move, Cut & Join, Protect & Select, Contact) as 32 px glyphs
  in a narrow column on the left; a category header folds; hover gives the name, key and what it does; a button can
  be dragged onto the favourites strip.
- ContextBar: the active tool's own settings above the viewport (glyph, name, radius in mm, strength, falloff, mirror);
  the gear opens the full Tool settings panel.
- FavouritesStrip: any tool or action dragged from the shelf or the palette; reorder by dragging, right-click removes;
  remembered.
- Palette: Space at the cursor: one search over tools AND actions ("crease", "sharp", "front", "rig", "hide"); Enter picks.
- Presets (11-11, design 4.1): "Save as preset..." in a brush's right-click menu (shelf, context bar, favourites,
  palette) or the context bar's Presets button; presets list under their brush in the palette and can go on the
  favourites strip.
install(win) wires them into the main window."""
import json

from PySide6.QtCore import QEvent, QMimeData, QObject, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QDrag, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMenu, QPushButton, QScrollArea, QSizePolicy,
                               QToolBar, QToolButton, QVBoxLayout, QWidget, QAbstractSpinBox, QTextEdit,
                               QPlainTextEdit, QInputDialog)

from . import glyphs as GL
from . import keymap as KM

MIME = 'application/x-fieldstudio-item'
SHELF_CATS = ['Object', 'Build', 'Refine', 'Move', 'Cut & Join', 'Protect & Select', 'Contact']
FIELD_GLYPH = {'Pencil ridge': 'pencil ridge', 'Pencil valley': 'pencil valley', 'Cut (remove)': 'cut',
               'Cut and separate': 'cut + separate', 'Splice': 'splice', 'Fill and splice': 'fill + splice'}
FIELD_KEYS = {'Pencil ridge': 'L', 'Pencil valley': 'Shift+L', 'Cut (remove)': 'X', 'Cut and separate': 'Shift+X',
              'Splice': 'J', 'Fill and splice': 'Shift+J'}
if __import__('os').environ.get('FS_LAYOUT', '') != 'classic':   # fs-shell-3 11-14: X is the approved Cut's
    FIELD_KEYS['Cut (remove)'] = ''                                 # (main_window.REG_KEYS)
DEFAULT_FAVS = ['tool:grab', 'tool:clay', 'tool:smooth', 'tool:inflate', 'tool:crease', 'tool:mask',
                'action:View/Front view', 'action:View/Right view', 'action:View/Top view',
                'action:View/Three-quarter view', 'action:View/Frame selection',
                'action:File/Reference Rig (align front and side)...']
SHORT_NAMES = {'Front view': 'Front', 'Right view': 'Right', 'Top view': 'Top', 'Three-quarter view': '3/4',
               'Frame selection': 'Frame', 'Reference Rig (align front and side)...': 'Rig'}


class Item:
    __slots__ = ('key', 'kind', 'name', 'cat', 'shortcut', 'desc', 'words', 'run', 'glyph')

    def __init__(self, key, kind, name, cat, shortcut, desc, words, run, glyph=None):
        self.key, self.kind, self.name, self.cat, self.shortcut = key, kind, name, cat, shortcut
        self.desc, self.words, self.run, self.glyph = desc, words, run, glyph

    def icon(self, size=32):
        if self.glyph:
            return GL.glyph_icon(self.glyph, size)
        return category_icon(self.cat, size)


_cat_icons = {}


def category_icon(cat, size=32):
    k = (cat, size)
    if k not in _cat_icons:
        pm = QPixmap(size, size)
        pm.fill(GL.BG)
        p = QPainter(pm)
        p.fillRect(0, 0, max(2, size // 16), size, QColor(*GL.CAT_COL.get(cat, (140, 150, 160))))
        p.setPen(QColor(200, 204, 210))
        p.drawText(pm.rect(), Qt.AlignCenter, cat[:1])
        p.end()
        _cat_icons[k] = QIcon(pm)
    return _cat_icons[k]


def _clean(text):
    return text.replace('&', '').split('\t')[0].strip()


def atext(a):
    """an action's name as its id reads it: the name it was made with ('fs_text', set when ui/menus.py renames it into
    plainer words), else its text"""
    t = a.property('fs_text')
    return _clean(t) if t else _clean(a.text())


def action_id(path, a):
    return '%s/%s' % (path, atext(a))


def menu_actions(win):
    """-> [(menu path, QAction)] of every leaf action in the menu bar. fs-shell-5 11-14: the path is the one the action
    was made under ('fs_path', stamped by ui/menus.py before it regroups the menus), so ids stay what the rebound
    keys, the column, the view bar and the search know"""
    out = []

    def walk(menu, path):
        for a in menu.actions():
            if a.isSeparator():
                continue
            if a.menu() is not None:
                root = a.menu().property('fs_menu_root')   # fs-shell-2 11-14: a menu the top bar folded keeps its
                walk(a.menu(), [str(root)] if root else path + [_clean(a.menu().title())])   # old name as its path
            elif a.text():
                fp = a.property('fs_path')
                out.append((str(fp) if fp else '/'.join(path), a))
    for a in win.menuBar().actions():
        if a.menu() is not None:
            walk(a.menu(), [_clean(a.menu().title())])
    return out


def items(win):
    """the registry: tools first (by shelf category), then the field tools, then every menu action"""
    from .panels import TOOLS
    out = []
    tp = win.tools
    for key, name, sc in TOOLS:
        cat, desc, words = GL.TOOL_INFO.get(key, ('Build', '', ''))
        out.append(Item('tool:' + key, 'tool', name, cat, KM.tool_key(win, key, sc), desc, words,
                        (lambda k=key: tp.set_tool(k)), key))
        for pr in (tp.presets(key) if hasattr(tp, 'presets') else []):
            out.append(Item('preset:%s:%s' % (key, pr['name']), 'preset', '%s: %s' % (name, pr['name']), cat, '',
                            'Preset of %s: r %g mm, strength %.2f, %s falloff' % (
                                name, round(pr.get('radius', 0) * 1000.0, 1), pr.get('strength', 0),
                                pr.get('falloff', 'smooth')),
                            '%s preset %s' % (words, pr['name'].lower()), (lambda q=pr: tp.apply_preset(q)), key))
    for label, cb in (getattr(win, 'field_tools', None) or {}).items():
        g = FIELD_GLYPH.get(label)
        if g is None:
            continue
        cat, desc, words = GL.TOOL_INFO.get(g, ('Cut & Join', '', ''))
        out.append(Item('field:' + label, 'field', label, cat, FIELD_KEYS.get(label, ''), desc, words,
                        (lambda f=cb: f()), g))
    seen = set()
    tool_names = set(n for _, n, _ in TOOLS)
    for path, a in menu_actions(win):
        name = atext(a)
        key = 'action:%s/%s' % (path.split('/')[0], name)
        name = _clean(a.text())                        # fs-shell-5 11-14: shown by its words now (ui/menus.py)
        if key in seen or (path.startswith('Sculpt') and name in tool_names):
            continue                                   # the Sculpt menu's tool entries are the tools themselves
        seen.add(key)
        sc = a.shortcut().toString(QKeySequence.NativeText) if not a.shortcut().isEmpty() else ''
        cat = {'File': 'File', 'View': 'View'}.get(path.split('/')[0], path.split('/')[0])
        root, rest = path.split('/')[0], path.split('/')[1:]          # fs-shell-2 11-14: shown where it is now
        where = (getattr(getattr(win, 'menu_folder', None), 'where', None) or {}).get(root)
        shown = ' > '.join([where] + rest) if where else path.replace('/', ' > ')
        if a.property('fs_where'):                    # fs-shell-5 11-14: the menu it sits in after ui/menus.py
            shown = str(a.property('fs_where'))
        out.append(Item(key, 'action', name, cat, sc, shown, path.lower() + ' ' + shown.lower(),
                        (lambda act=a: act.trigger()), None))
    return out


def search(its, q):
    q = q.strip().lower()
    if not q:
        return list(its)
    toks = q.split()
    scored = []
    for it in its:
        name = it.name.lower()
        hay = ' '.join((name, it.words or '', (it.desc or '').lower(), it.cat.lower()))
        if not all(t in hay for t in toks):
            continue
        wn = name.replace('/', ' ').replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
        s = 5
        if it.shortcut and q == it.shortcut.lower():
            s = -1                                     # the tool's own key letter: that tool first
        elif toks[0] in wn:
            s = -0.5                                   # a whole word ("rig" -> Reference Rig before Right view)
        elif name.startswith(q):
            s = 0
        elif any(w.startswith(toks[0]) for w in wn):
            s = 1
        elif toks[0] in (it.words or '').split():
            s = 1.5                                    # an intent word ("fold" -> Crease)
        elif toks[0] in name:
            s = 2
        elif any(w.startswith(toks[0]) for w in (it.words or '').split()):
            s = 3
        s += 0.0 if it.kind == 'tool' else (0.2 if it.kind == 'field' else 0.4)
        scored.append((s, len(it.name), it))
    scored.sort(key=lambda x: (x[0], x[1]))
    res = [x[2] for x in scored]
    # presets under their brush (design 4.1): a brush and its matching presets stay together, at the place of
    # whichever of them ranks first (a preset found by its own name brings its brush up just above it)
    byk = {it.key: it for it in res}
    under = {}
    for it in res:
        if it.kind == 'preset' and 'tool:' + it.key.split(':')[1] in byk:
            under.setdefault('tool:' + it.key.split(':')[1], []).append(it)
    out, done = [], set()
    for it in res:
        head = ('tool:' + it.key.split(':')[1]) if (it.kind == 'preset' and it.key.split(':')[0] == 'preset'
                                                     and 'tool:' + it.key.split(':')[1] in byk) else it.key
        if head in done:
            continue
        done.add(head)
        if head in under:
            out.append(byk[head])
            out.extend(under[head])
        else:
            out.append(it)
    return out


def save_preset_dialog(win, tool):
    """asks a name (a default from the settings) and saves the brush's settings under it"""
    from .panels import TOOLS
    tp = win.tools
    name = dict((k, n) for k, n, _ in TOOLS).get(tool, tool)
    text, ok = QInputDialog.getText(win, 'Save as preset', 'A name for this %s preset:' % name,
                                    text=tp.preset_default_name(tool))
    if ok:
        return tp.save_preset(text, tool)
    return None


def brush_menu(win, tool, m=None, parent=None):
    """a brush's right-click menu: Save as preset..., its presets, delete one, the shared-size lock"""
    tp = win.tools
    m = m or QMenu(parent or win)
    if not tp._is_brush(tool):
        return m
    m.addAction('Save as preset...', lambda: save_preset_dialog(win, tool))
    ps = tp.presets(tool)
    if ps:
        sub = m.addMenu('Presets')
        for pr in ps:
            sub.addAction(pr['name'], lambda q=pr: tp.apply_preset(q))
        dm = m.addMenu('Delete preset')
        for pr in ps:
            dm.addAction(pr['name'], lambda q=pr: tp.delete_preset(tool, q['name']))
    a = m.addAction('Share size and strength across brushes')
    a.setCheckable(True)
    a.setChecked(tp.share_rs)
    a.toggled.connect(tp.share.setChecked)
    return m


# ---- the shelf ------------------------------------------------------------------------------------------------------
class _DragButton(QToolButton):
    def __init__(self, item_key, parent=None):
        super().__init__(parent)
        self.item_key = item_key
        self._press = None

    def mousePressEvent(self, e):
        self._press = e.position()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press is not None and (e.buttons() & Qt.LeftButton) and \
                (e.position() - self._press).manhattanLength() > 10:
            d = QDrag(self)
            md = QMimeData()
            md.setData(MIME, self.item_key.encode('utf-8'))
            d.setMimeData(md)
            d.setPixmap(self.icon().pixmap(32, 32))
            self._press = None
            self.setDown(False)
            d.exec(Qt.CopyAction)
            return
        super().mouseMoveEvent(e)


class ToolShelf(QWidget):
    """the tools by category in a narrow column (two 32 px glyphs per row); with names on (View > Tool names on the
    shelf, or right-click the shelf), one tool a row with its name beside its glyph (11-11: icons that were all
    circles were hard to sort through; Blender shows the toolbar's names when it is widened)"""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.buttons = {}
        self.headers = {}
        self.bodies = {}
        self.grids = {}
        self.order = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        its = [i for i in items(win) if i.kind in ('tool', 'field')]
        cfg = _settings()
        folded = set(json.loads(cfg.value('shelf_folded', '[]') or '[]'))
        self.names = str(cfg.value('shelf_names', 'false')).lower() in ('true', '1')
        for cat in SHELF_CATS:
            group = [i for i in its if i.cat == cat]
            if not group:
                continue
            h = QPushButton(cat.replace('&', '&&'))
            h.setFlat(True)
            col = QColor(*GL.CAT_COL[cat])
            h.setStyleSheet('QPushButton { text-align: left; font-size: 7pt; color: %s; padding: 1px 2px; '
                            'border: none; border-top: 2px solid %s; background: transparent; }' % (col.name(), col.name()))
            h.setToolTip('%s: click to fold or unfold' % cat)
            h.clicked.connect(lambda _=False, c=cat: self.fold(c))
            lay.addWidget(h)
            body = QWidget()
            g = QGridLayout(body)
            g.setContentsMargins(0, 0, 0, 0)
            g.setSpacing(2)
            for n, it in enumerate(group):
                b = _DragButton(it.key)
                b.setIcon(it.icon(32))
                b.setIconSize(QSize(32, 32))
                b.setCheckable(True)
                b.setAutoRaise(True)
                b.setFocusPolicy(Qt.NoFocus)
                b.setText(it.name)
                b.setToolTip('<b>%s</b>%s<br>%s' % (it.name, ('  [%s]' % it.shortcut) if it.shortcut else '', it.desc))
                b.clicked.connect(lambda _=False, i=it: self._pick(i))
                if it.kind == 'tool' and win.tools._is_brush(it.key[5:]):
                    b.setContextMenuPolicy(Qt.CustomContextMenu)
                    b.customContextMenuRequested.connect(
                        lambda _p, k=it.key[5:], bb=b: self._menu(k, bb))
                self.buttons[it.key] = b
                self.order.setdefault(cat, []).append(b)
            self.grids[cat] = g
            lay.addWidget(body)
            self.headers[cat], self.bodies[cat] = h, body
            body.setVisible(cat not in folded)
        lay.addStretch(1)
        self._lay_out()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._shelf_menu)
        win.tools.toolChanged.connect(lambda _k: self.sync())
        self.sync()

    def _lay_out(self):
        """glyphs only: two 38 px tiles a row; names: one tool a row, its name beside its glyph, as wide as the longest"""
        fm = self.fontMetrics()
        wide = 38 + 8 + max([fm.horizontalAdvance(b.text()) for bs in self.order.values() for b in bs] or [0]) + 10
        for cat, bs in self.order.items():
            g = self.grids[cat]
            for b in bs:
                g.removeWidget(b)
            for n, b in enumerate(bs):
                if self.names:
                    b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                    b.setFixedSize(wide, 38)
                    g.addWidget(b, n, 0)
                else:
                    b.setToolButtonStyle(Qt.ToolButtonIconOnly)
                    b.setFixedSize(38, 38)
                    g.addWidget(b, n // 2, n % 2)
        self.adjustSize()

    def set_names(self, on):
        """names beside the glyphs on or off; kept for the next session; the shelf's column follows its width"""
        self.names = bool(on)
        _settings().setValue('shelf_names', 'true' if self.names else 'false')
        self._lay_out()
        sa = getattr(self, 'scroll', None)
        if sa is not None:
            sa.setFixedWidth(self.sizeHint().width() + 14)
        a = getattr(self.win, 'shelf_names_action', None)
        if a is not None and a.isChecked() != self.names:
            a.setChecked(self.names)

    def _shelf_menu(self, pos):
        m = QMenu(self)
        a = m.addAction('Tool names on the shelf')
        a.setCheckable(True)
        a.setChecked(self.names)
        a.toggled.connect(self.set_names)
        self.last_menu = m
        m.exec(self.mapToGlobal(pos))

    def _menu(self, tool, b):
        m = brush_menu(self.win, tool, parent=self)
        m.addSeparator()
        fs = getattr(self.win, 'fav_strip', None)
        if fs is not None:
            m.addAction('Add to favourites', lambda: fs.add('tool:' + tool))
        self.last_menu = m
        m.exec(QCursor.pos())

    def keys_changed(self):
        for it in items(self.win):
            b = self.buttons.get(it.key)
            if b is not None:
                b.setToolTip('<b>%s</b>%s<br>%s' % (it.name, ('  [%s]' % it.shortcut) if it.shortcut else '', it.desc))

    def fold(self, cat):
        b = self.bodies[cat]
        b.setVisible(not b.isVisible())
        _settings().setValue('shelf_folded', json.dumps([c for c, w in self.bodies.items() if not w.isVisible()]))

    def _pick(self, it):
        it.run()
        self.sync()

    def sync(self):
        tp = self.win.tools
        fa = getattr(tp, 'field_active', None)
        for key, b in self.buttons.items():
            on = (key == 'field:%s' % fa) if fa else (key == 'tool:%s' % tp.tool)
            if b.isChecked() != on:
                b.setChecked(on)


# ---- the context bar ------------------------------------------------------------------------------------------------
class ContextBar(QWidget):
    """the active tool's own settings, one row above the viewport"""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        tp = win.tools
        self.setStyleSheet('background: #1a1d22;')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.setSpacing(8)
        self.glyph = QLabel()
        self.glyph.setFixedSize(26, 26)
        lay.addWidget(self.glyph)
        self.name = QLabel('')
        self.name.setStyleSheet('font-weight: 600;')
        self.name.setMinimumWidth(110)
        for w in (self.glyph, self.name):
            w.setContextMenuPolicy(Qt.CustomContextMenu)
            w.customContextMenuRequested.connect(lambda _p: self._menu())
            w.setToolTip('Right-click: save this brush as a preset, its presets')
        lay.addWidget(self.name)
        self.r = QDoubleSpinBox()
        self.r.setDecimals(1)
        self.r.setRange(0.5, 120.0)
        self.r.setSuffix(' mm')
        self.r.setPrefix('r ')
        self.r.setKeyboardTracking(False)
        self.r.setToolTip('Brush radius ([ ] or F-drag in the view)')
        self.r.valueChanged.connect(lambda v: tp.set_radius(v / 1000.0))
        lay.addWidget(self.r)
        self.s = QDoubleSpinBox()
        self.s.setDecimals(2)
        self.s.setRange(0.0, 2.0)
        self.s.setSingleStep(0.05)
        self.s.setPrefix('s ')
        self.s.setKeyboardTracking(False)
        self.s.setToolTip('Strength 0-2 (- = or Shift+F-drag); past 1 every brush keeps rising')
        self.s.valueChanged.connect(lambda v: tp.set_strength(v))
        lay.addWidget(self.s)
        self.falloff = QComboBox()
        for k in ('smooth', 'linear', 'sharp', 'constant', 'custom'):
            self.falloff.addItem(k, k)
        self.falloff.setToolTip('Falloff')
        self.falloff.currentIndexChanged.connect(self._falloff)
        lay.addWidget(self.falloff)
        self.presets = QToolButton()
        self.presets.setText('Presets')
        self.presets.setPopupMode(QToolButton.InstantPopup)
        self.presets.setFocusPolicy(Qt.NoFocus)
        self.presets.setToolTip('This brush\'s presets; Save as preset... keeps its radius, strength, falloff, autosmooth '
                                'and targeting under a name')
        self.presets_menu = QMenu(self.presets)
        self.presets_menu.aboutToShow.connect(self._fill_presets)
        self.presets.setMenu(self.presets_menu)
        lay.addWidget(self.presets)
        # fs-shell 11-11 (design 4.5): the pivot of the object tools' turns and even scaling
        self.pivot = QComboBox()
        self.pivot.addItem('Pivot: object centre', 'centre')
        self.pivot.addItem('Pivot: pick a surface point...', 'pick')
        self.pivot.setToolTip('Rotate and Scale turn and grow about this point. "Pick": the next click on the form '
                              'sets it (Esc keeps the centre).')
        self.pivot.activated.connect(self._pivot)
        lay.addWidget(self.pivot)
        # fs-shell 11-11 (design D1): the scene's unit, beside the gizmo's tools
        self.units = QComboBox()
        from .units import SCENE_UNITS
        for u_ in SCENE_UNITS:
            self.units.addItem(u_, u_)
        self.units.setToolTip('The scene\'s unit: typed values during a drag, snapping and read-outs. Any unit can '
                              'always be typed ("34 in", "1.5 m", "2x4x8ft").')
        self.units.activated.connect(lambda i: win.set_units(self.units.itemData(i)))
        lay.addWidget(self.units)
        # fs-shell 11-11 (design 4.5): an object with masses: its blend dial (how smoothly the masses join)
        self.blend_lab = QLabel('Blend')
        self.blend = QDoubleSpinBox()
        self.blend.setDecimals(1)
        self.blend.setRange(0.0, 200.0)
        self.blend.setSuffix(' mm')
        self.blend.setKeyboardTracking(False)
        self.blend.setToolTip('How smoothly the masses join (0: a sharp crease). One undo step per change.')
        self.blend.valueChanged.connect(self._blend)
        lay.addWidget(self.blend_lab)
        lay.addWidget(self.blend)
        self.whole = QPushButton('Whole object')
        self.whole.setFocusPolicy(Qt.NoFocus)
        self.whole.setToolTip('The gizmo back on the whole object (a click on a mass takes that mass)')
        self.whole.clicked.connect(self._whole)
        lay.addWidget(self.whole)
        # fs-shell 11-11 (design 4.18): with a cutter selected, its cut: what goes and what becomes of it
        self.cut_side = QComboBox()
        self.cut_side.addItem('Cut away: inside', 'inside')
        self.cut_side.addItem('Cut away: outside', 'outside')
        self.cut_side.setToolTip('Inside: a hole where the cutter is. Outside: keep only what is inside the cutter.')
        self.cut_result = QComboBox()
        self.cut_result.addItem('Delete it', 'delete')
        self.cut_result.addItem('Lift out (new object)', 'lift')
        self.cut_result.addItem('Keep both (touching)', 'keep')
        self.cut_result.setToolTip('What becomes of the part that is cut away')
        self.cut_apply = QPushButton('Apply cut')
        self.cut_apply.setToolTip('One undo step; the cutter is used up')
        self.cut_apply.clicked.connect(lambda: win.apply_cut(self.cut_side.currentData(), self.cut_result.currentData()))
        for w in (self.cut_side, self.cut_result, self.cut_apply):
            w.setFocusPolicy(Qt.NoFocus)
            lay.addWidget(w)
        lay.addWidget(QLabel('Mirror'))
        self.mir = []
        for i, ax in enumerate('XYZ'):
            b = QPushButton(ax)
            b.setCheckable(True)
            b.setFixedWidth(26)
            b.setFocusPolicy(Qt.NoFocus)
            b.toggled.connect(lambda on, j=i: tp.sym[j].setChecked(on))
            lay.addWidget(b)
            self.mir.append(b)
        from .widgets import ElideLabel
        self.hint = ElideLabel('', min_w=40)
        self.hint.setObjectName('dim')
        lay.addWidget(self.hint, 1)
        g = QPushButton('Settings')
        g.setToolTip('All of this tool\'s settings (the Tool settings panel)')
        g.setFocusPolicy(Qt.NoFocus)
        g.clicked.connect(self._gear)
        lay.addWidget(g)
        tp.settingsChanged.connect(lambda _s: self.sync())
        tp.toolChanged.connect(lambda _k: self.sync())
        win.engine.sceneChanged.connect(lambda *_a: self.sync())     # the selection (a cutter shows its cut)
        for b in tp.sym:
            b.toggled.connect(lambda _on: self.sync())
        self.sync()

    def _falloff(self, i):
        k = self.falloff.itemData(i)
        tp = self.win.tools
        if hasattr(tp, 'falloff'):
            j = tp.falloff.findData(k) if hasattr(tp.falloff, 'findData') else -1
            if j >= 0:
                tp.falloff.setCurrentIndex(j)
                return
        tp._falloff_changed(k)

    def _whole(self):
        pr = self.win.props
        objs = {o['id']: o for o in (getattr(self.win, 'snap', None) or {}).get('objects', [])}
        sel = [o for o in self.win.viewport.selected if o in objs]
        if sel:
            pr.node_id = objs[sel[0]]['base']['id']
            pr.refresh()
        self.sync()
        self.win.viewport.update()

    def _blend(self, v):
        from ..tools.masses import SetMassBlend
        sel = list(self.win.viewport.selected)
        if sel:
            self.win.engine.post('push', SetMassBlend(sel[0], float(v) / 1000.0))

    def _pivot(self, i):
        g = getattr(self.win.viewport, 'gizmo', None)
        if g is not None:
            g.set_pivot_mode(self.pivot.itemData(i))
        self.sync()

    def _fill_presets(self):
        self.presets_menu.clear()
        brush_menu(self.win, self.win.tools.tool, self.presets_menu)

    def _menu(self):
        tool = self.win.tools.tool
        if self.win.tools._is_brush(tool):
            m = brush_menu(self.win, tool, parent=self)
            self.last_menu = m
            m.exec(QCursor.pos())

    def keys_changed(self):
        self.sync()

    def _gear(self):
        d = self.win.docks.get('Tools')
        if d is not None:
            d.setVisible(not d.isVisible())
            if d.isVisible():
                d.raise_()

    def sync(self):
        tp = self.win.tools
        key = tp.tool
        info = GL.TOOL_INFO.get(key, ('Build', '', ''))
        self.glyph.setPixmap(GL.glyph_pixmap(key, 26))
        from .panels import TOOLS
        name = dict((k, n) for k, n, _ in TOOLS).get(key, key)
        sc = KM.tool_key(self.win, key, dict((k, s) for k, _, s in TOOLS).get(key, ''))
        self.name.setText('%s  %s' % (name, sc))
        brush = key not in ('select', 'move', 'rotate', 'scale')
        for w in (self.r, self.s, self.falloff, self.presets):
            w.setEnabled(brush)
            w.setVisible(brush)
        own = key in ('snake', 'elastic')               # fs-shell 11-12: these carry their own falloff (fs-brush)
        self.falloff.setEnabled(brush and not own)
        self.falloff.setToolTip('Falloff (%s uses its own)' % name
                                if own else 'Falloff')
        objt = key in ('move', 'rotate', 'scale')
        self.pivot.setVisible(objt)
        self.units.setVisible(objt)
        self.units.blockSignals(True)
        self.units.setCurrentIndex(max(0, self.units.findData(getattr(self.win, 'units', 'mm'))))
        self.units.blockSignals(False)
        from ..tools.shapecut import is_cutter
        objs = {o['id']: o for o in (getattr(self.win, 'snap', None) or {}).get('objects', [])}
        sel = [o for o in self.win.viewport.selected if o in objs]
        cut = bool(sel) and is_cutter(objs[sel[0]])
        for w in (self.cut_side, self.cut_result, self.cut_apply):
            w.setVisible(cut)
        from ..tools.masses import mass_blend
        mb = mass_blend(objs[sel[0]]['base']) if sel else None
        for w in (self.blend_lab, self.blend):
            w.setVisible(mb is not None)
        g_ = getattr(self.win.viewport, 'gizmo', None)
        on_mass = bool(objt and g_ is not None and g_.mass_ctx() is not None)
        self.whole.setVisible(on_mass)
        if mb is not None and abs(self.blend.value() - mb * 1000.0) > 1e-6:
            self.blend.blockSignals(True)
            self.blend.setValue(mb * 1000.0)
            self.blend.blockSignals(False)
        g = getattr(self.win.viewport, 'gizmo', None)
        if g is not None and objt:
            self.pivot.blockSignals(True)
            if self.pivot.count() > 2:
                self.pivot.removeItem(2)
            if g.pivot_mode == 'point' and g.pivot_point is not None:
                self.pivot.addItem('Pivot: surface point (%.0f, %.0f, %.0f mm)' % tuple(v * 1000.0 for v in g.pivot_point),
                                   'point')
                self.pivot.setCurrentIndex(2)
            else:
                self.pivot.setCurrentIndex(0)
            self.pivot.blockSignals(False)
        st = tp.settings
        for w, v in ((self.r, st.get('radius', 0.01) * 1000.0), (self.s, st.get('strength', 0.5))):
            w.blockSignals(True)
            w.setValue(float(v))
            w.blockSignals(False)
        self.falloff.blockSignals(True)
        i = self.falloff.findData(st.get('falloff', 'smooth'))
        self.falloff.setCurrentIndex(max(0, i))
        self.falloff.blockSignals(False)
        for b, src in zip(self.mir, tp.sym):
            b.blockSignals(True)
            b.setChecked(src.isChecked())
            b.blockSignals(False)
        inv = {'draw': 'Ctrl: carve', 'carve': 'Ctrl: draw', 'mask': 'Ctrl: erase', 'inflate': 'Ctrl: deflate',
               'clay': 'Ctrl: dig'}.get(key, 'Ctrl: invert')
        self.hint.setText(('   '.join(x for x in ('Shift: smooth' if brush else '', inv, 'F: size' if brush else '',
                                                    'Middle: orbit' if brush else '')
                                     if x)) if brush else info[1])
        if self.whole.isVisible():                 # design 4.5: the gizmo is on a mass
            self.hint.setText('The gizmo is on a mass: drag, type a value, Ctrl snaps; a click on another mass takes it')


# ---- the favourites strip ---------------------------------------------------------------------------------------------
class FavouritesStrip(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.setAcceptDrops(True)
        self.setStyleSheet('background: #181a1f;')
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(8, 2, 8, 2)
        self.lay.setSpacing(3)
        self.label = QLabel('Favourites')
        self.label.setObjectName('dim')
        self.label.setToolTip('Drag any tool from the shelf or the Space palette here; drag to reorder; right-click '
                              'to remove')
        self.keys = []
        self.reg = {}
        try:
            self.keys = json.loads(_settings().value('favourites', '') or '[]') or list(DEFAULT_FAVS)
        except Exception:
            self.keys = list(DEFAULT_FAVS)
        self.rebuild()
        win.tools.toolChanged.connect(lambda _k: self.sync())

    def rebuild(self):
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w is not None and w is not self.label:
                w.hide()                         # fs-shell-2 11-14: a removed button leaves the screen now
                w.deleteLater()
        self.lay.addWidget(self.label)
        self.reg = {i.key: i for i in items(self.win)}
        self.buttons = {}
        for k in self.keys:
            it = self.reg.get(k)
            if it is None:
                continue
            b = _DragButton(k)
            if it.kind != 'action':
                b.setIcon(it.icon(24))
            b.setIconSize(QSize(24, 24))
            b.setText(SHORT_NAMES.get(it.name, it.name) if it.kind == 'action' else '')
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon if it.kind == 'action' else Qt.ToolButtonIconOnly)
            b.setAutoRaise(True)
            b.setCheckable(it.kind == 'tool')
            if it.kind == 'preset':
                b.setText('%s' % it.name.split(': ', 1)[-1][:14])
                b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip('<b>%s</b>%s<br>%s' % (it.name, ('  [%s]' % it.shortcut) if it.shortcut else '', it.desc))
            b.clicked.connect(lambda _=False, i=it: (i.run(), self.sync()))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, kk=k: self._menu(kk))
            self.lay.addWidget(b)
            self.buttons[k] = b
        self.lay.addStretch(1)
        # fs-shell 11-11: favourites that do not fit go to a >> menu at the end (the strip held the window at 1632 px)
        self.more = QToolButton()
        self.more.setText('»')
        self.more.setAutoRaise(True)
        self.more.setPopupMode(QToolButton.InstantPopup)
        self.more.setFocusPolicy(Qt.NoFocus)
        self.more.setToolTip('More favourites (the strip is narrower than they are)')
        self.more_menu = QMenu(self.more)
        self.more.setMenu(self.more_menu)
        self.lay.addWidget(self.more)
        self.more.hide()
        self.sync()
        self._fit()

    def minimumSizeHint(self):
        return QSize(self.label.sizeHint().width() + 60, super().minimumSizeHint().height())

    def sizeHint(self):
        w = 16 + self.label.sizeHint().width() + sum(b.sizeHint().width() + 3 for b in self.buttons.values())
        return QSize(w, super().sizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        """show the favourites that fit, in order; the rest go to the >> menu"""
        if not hasattr(self, 'more'):
            return
        btns = [(k, self.buttons[k]) for k in self.keys if k in self.buttons]
        avail = self.width() - 16 - self.label.sizeHint().width() - 3
        need = sum(b.sizeHint().width() + 3 for _, b in btns)
        lim = avail - (self.more.sizeHint().width() + 3 if need > avail else 0)
        used, hidden = 0, []
        for k, b in btns:
            w = b.sizeHint().width() + 3
            if used + w <= lim and not hidden:
                b.setVisible(True)
                used += w
            else:
                b.setVisible(False)
                hidden.append((k, b))
        self.more_menu.clear()
        for k, b in hidden:
            it = self.reg.get(k)
            a = self.more_menu.addAction(b.icon(), it.name if it is not None else k)
            a.triggered.connect(lambda _=False, bb=b: bb.click())
        self.more.setVisible(bool(hidden))
        self.hidden_keys = [k for k, _ in hidden]

    def sync(self):
        tool = 'tool:%s' % self.win.tools.tool
        for k, b in self.buttons.items():
            if b.isCheckable():
                b.setChecked(k == tool)

    def _menu(self, k):
        m = QMenu(self)
        a = m.addAction('Remove from favourites')
        it = self.reg.get(k)
        if it is not None and it.kind in ('tool', 'preset') and self.win.tools._is_brush(it.glyph):
            m.addSeparator()
            brush_menu(self.win, it.glyph, m)
            if it.kind == 'preset':
                tool, name = k.split(':', 2)[1:]
                m.addAction('Delete this preset', lambda: self.win.tools.delete_preset(tool, name))
        self.last_menu = m
        if m.exec(QCursor.pos()) is a:
            self.remove(k)

    def keys_changed(self):
        self.rebuild()

    def add(self, k, at=None):
        if k in self.keys:
            self.keys.remove(k)
        self.keys.insert(len(self.keys) if at is None else at, k)
        self._save()
        self.rebuild()

    def remove(self, k):
        if k in self.keys:
            self.keys.remove(k)
            self._save()
            self.rebuild()

    def _save(self):
        _settings().setValue('favourites', json.dumps(self.keys))

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()

    def dropEvent(self, e):
        k = bytes(e.mimeData().data(MIME)).decode('utf-8')
        x = e.position().x()
        at = 0
        for kk in self.keys:
            b = self.buttons.get(kk)
            if b is not None and b.geometry().center().x() < x:
                at += 1
        if k in self.keys and self.keys.index(k) < at:
            at -= 1
        self.add(k, at)
        e.acceptProposedAction()


# ---- the Space palette ----------------------------------------------------------------------------------------------
class Palette(QFrame):
    picked = Signal(str)

    def __init__(self, win):
        super().__init__(win, Qt.Popup | Qt.FramelessWindowHint)
        self.win = win
        self.setObjectName('palette')
        self.setStyleSheet('#palette { background: #1d2026; border: 1px solid #3b82f6; border-radius: 6px; }')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText('Search tools and actions ("crease", "smooth", "front", "rig", "hide")')
        self.edit.textChanged.connect(self.refill)
        self.edit.installEventFilter(self)
        lay.addWidget(self.edit)
        self.list = _PList()
        self.list.setIconSize(QSize(32, 32))
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        self.list.setDragEnabled(True)
        self.list.itemActivated.connect(self._activate)
        self.list.itemClicked.connect(self._activate)
        lay.addWidget(self.list)
        self.resize(460, 440)
        self.its = []

    def open_at(self, gpos=None):
        self.its = items(self.win)
        self.edit.blockSignals(True)
        self.edit.clear()
        self.edit.blockSignals(False)
        self.refill('')
        gpos = gpos or QCursor.pos()
        scr = QApplication.screenAt(gpos) or QApplication.primaryScreen()
        g = scr.availableGeometry()
        x = min(max(g.left(), gpos.x() - 40), g.right() - self.width())
        y = min(max(g.top(), gpos.y() - 20), g.bottom() - self.height())
        self.move(x, y)
        self.show()
        self.raise_()
        self.activateWindow()
        self.edit.setFocus()
        self.opened_at = __import__('time').perf_counter()

    def refill(self, q):
        self.list.clear()
        res = search(self.its, q)
        for it in res[:60]:
            nm = ('      \u21b3 %s' % it.name.split(': ', 1)[-1]) if it.kind == 'preset' else it.name
            li = QListWidgetItem(it.icon(24 if it.kind == 'preset' else 32),
                                 '%s%s   %s' % (nm, ('   [%s]' % it.shortcut) if it.shortcut else '',
                                                '\u00b7 ' + ('preset' if it.kind == 'preset' else it.cat)))
            li.setToolTip(it.desc)
            li.setData(Qt.UserRole, it.key)
            self.list.addItem(li)
        if self.list.count():
            self.list.setCurrentRow(0)
        self.results = res

    def eventFilter(self, obj, e):
        if obj is self.edit and e.type() == QEvent.KeyPress:
            k = e.key()
            if k in (Qt.Key_Down, Qt.Key_Up):
                r = self.list.currentRow() + (1 if k == Qt.Key_Down else -1)
                self.list.setCurrentRow(min(max(r, 0), self.list.count() - 1))
                return True
            if k in (Qt.Key_Return, Qt.Key_Enter):
                it = self.list.currentItem()
                if it is not None:
                    self._activate(it)
                return True
            if k == Qt.Key_Escape:
                self.hide()
                return True
        return False

    def _menu(self, pos):
        li = self.list.itemAt(pos)
        if li is None:
            return
        key = li.data(Qt.UserRole)
        it = next((i for i in self.its if i.key == key), None)
        if it is None:
            return
        m = QMenu(self)
        fs = getattr(self.win, 'fav_strip', None)
        if fs is not None:
            m.addAction('Add to favourites', lambda: fs.add(key))
        if it.kind in ('tool', 'preset') and self.win.tools._is_brush(it.glyph):
            m.addSeparator()
            brush_menu(self.win, it.glyph, m)
            if it.kind == 'preset':
                tool, name = key.split(':', 2)[1:]
                m.addAction('Delete this preset', lambda: self.win.tools.delete_preset(tool, name))
        self.last_menu = m
        m.exec(self.list.viewport().mapToGlobal(pos))
        if self.isVisible():
            self.its = items(self.win)
            self.refill(self.edit.text())

    def _activate(self, li):
        key = li.data(Qt.UserRole)
        it = next((i for i in self.its if i.key == key), None)
        self.hide()
        if it is not None:
            it.run()
            self.picked.emit(key)
            self.last_pick = (key, __import__('time').perf_counter() - getattr(self, 'opened_at', 0.0))
        self.win.viewport.setFocus()


class _PList(QListWidget):
    """the palette's list: a dragged row carries its item key, so it can be dropped on the favourites strip"""

    def mimeData(self, its):
        md = super().mimeData(its)
        if its:
            md.setData(MIME, str(its[0].data(Qt.UserRole)).encode('utf-8'))
        return md


class SpaceFilter(QObject):
    """Space (no modifier, not in a text or number box) opens the palette at the mouse"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress and ev.key() == Qt.Key_Space and not ev.isAutoRepeat() \
                and not (ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.ShiftModifier | Qt.MetaModifier)):
            fw = QApplication.focusWidget()
            if isinstance(fw, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)):
                return False
            aw = QApplication.activeWindow()
            if aw is not self.win:
                return False
            if self.win.viewport.drag is not None:
                return False
            self.win.palette.open_at()
            return True
        return False


def _settings():
    from .main_window import app_settings
    return app_settings()


def install(win):
    """the shelf (left), the context bar and the favourites strip (above the view), the palette (Space); the full Tool
    settings panel stays one click away (the context bar's Settings, View > Tool settings panel)"""
    tb = QToolBar('Tool shelf')
    tb.setObjectName('toolshelf')
    tb.setMovable(False)
    tb.setOrientation(Qt.Vertical)
    shelf = ToolShelf(win)
    sa = QScrollArea()
    sa.setWidget(shelf)
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setFixedWidth(shelf.sizeHint().width() + 14)
    shelf.scroll = sa
    tb.addWidget(sa)
    win.addToolBar(Qt.LeftToolBarArea, tb)
    win.shelf, win.shelf_bar = shelf, tb
    vm = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'View' in a.menu().title()), None)
    if vm is not None:                         # View > Tool names on the shelf (also on the shelf's right-click)
        a = QAction('Tool names on the shelf', win)
        a.setCheckable(True)
        a.setChecked(shelf.names)
        a.toggled.connect(shelf.set_names)
        vm.addAction(a)
        win.shelf_names_action = a
    cl = win.centralWidget().layout()
    win.context_bar = ContextBar(win)
    win.fav_strip = FavouritesStrip(win)
    cl.insertWidget(1, win.context_bar)
    slot = getattr(win, 'fav_slot', None)
    if slot is not None:
        slot.addWidget(win.fav_strip, 1)
        win.fav_strip.setStyleSheet('background: transparent;')
    else:
        cl.insertWidget(2, win.fav_strip)
    win.palette = Palette(win)
    if hasattr(win.tools, 'presetsChanged'):
        win.tools.presetsChanged.connect(win.fav_strip.rebuild)
    win.space_filter = SpaceFilter(win)
    QApplication.instance().installEventFilter(win.space_filter)
    d = win.docks.get('Tools')
    if d is not None:
        d.setWindowTitle('Tool settings')
        d.hide()
        d.toggleViewAction().setText('Tool settings panel')
    return shelf
