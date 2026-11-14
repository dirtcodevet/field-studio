"""Left: sculpt tools + Brush Controls. Right: scene objects, combine (booleans / splice), the selected object's
layers (base tree with live node parameters, sculpt, detail, colour, material), mutations, object properties.
Panels read the engine's scene snapshots and post jobs; they never touch the scene."""
import copy
import json

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QColorDialog, QComboBox, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                               QScrollArea, QSizePolicy, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from ..core import brush as BR


def _registry_cut(order, kind):
    """fs-shell-2 11-14 (fs-tools' routing): Subtract and Intersect are Cut with an object
    (tools.cut_object(target, cutter, 'rest' | 'piece'); the cutter is used up, undo puts it back): the first selected
    is the target, each other one cuts it in turn. core/combine.CombineObjects keeps no window entry point. True when
    it ran"""
    import os
    if os.environ.get('FS_LAYOUT', '') == 'classic':
        return False
    from PySide6.QtWidgets import QApplication
    win = next((t for t in QApplication.topLevelWidgets() if hasattr(t, 'tool_strip') and hasattr(t, 'engine')), None)
    h = getattr(getattr(win, 'engine', None), '_job_handlers', None) or {}
    if win is None or 'tools.cut_object' not in h:
        return False
    keep = 'rest' if kind == 'subtract' else 'piece'
    for c in list(order)[1:]:
        win.engine.post('tools.cut_object', int(order[0]), int(c), keep)
    try:
        win.status.showMessage('%s: %d object(s) cut the first selected (each one undo step)' % (
            'Subtract' if keep == 'rest' else 'Intersect', len(order) - 1), 5000)
    except Exception:
        pass
    return True


def _registry_merge():
    """fs-shell-2 11-14 (decided: ONE Merge in the app): every union merge (the Scene panel, Ctrl+J, the right-click
    menu, the menus, the search) runs fs-tools' Merge (tools.merge: an exact join, blend at contact, refused with
    'Group instead' when they do not touch) through the strip's Merge tool. True when it did. FS_LAYOUT=classic, or the
    registry not running, keeps the old merge (core/combine.py)"""
    import os
    if os.environ.get('FS_LAYOUT', '') == 'classic':
        return False
    from PySide6.QtWidgets import QApplication
    win = next((t for t in QApplication.topLevelWidgets() if hasattr(t, 'tool_strip') and hasattr(t, 'engine')), None)
    if win is None:
        return False
    try:
        from . import toolreg as TR
        if not TR.runnable(win, 'merge'):
            return False
        TR.merge_now(win)
        return True
    except Exception as e_:
        print('registry merge not run: %r' % (e_,))
        return False
from ..core import ops
from .widgets import FalloffCurve, FloatField, VecField, mutation_icon, tool_icon

TOOLS = [('select', 'Select / orbit', 'Q'), ('move', 'Move object', 'W'), ('rotate', 'Rotate object', 'R'),
         ('scale', 'Scale object', 'Y'),
         ('grab', 'Move / Grab', 'G'), ('inflate', 'Inflate', 'I'), ('smooth', 'Smooth', 'S'),
         ('clay', 'Clay', 'C'), ('draw', 'Draw', 'D'), ('carve', 'Carve', 'V'), ('flatten', 'Flatten', 'T'),
         ('crease', 'Crease', 'K'), ('pinch', 'Pinch', 'P'), ('stretch', 'Stretch', 'E'),
         ('snake', 'Snake Hook', 'N'), ('mask', 'Mask', 'B'),
         ('swell', 'Swell to contact', 'Shift+I'), ('settle', 'Settle onto', 'Shift+G')]
try:                     # fs-shell 11-12 (reachable before the test): Elastic Grab appears here (key U) the
    from ..core import brush as _BRt                    # moment fs-brush's core knows the tool 'elastic' (not before:
    if 'elastic' in getattr(_BRt, 'TOOLS', ()):         # its card path is still being built)
        TOOLS.insert([k for k, _, _ in TOOLS].index('snake') + 1, ('elastic', 'Elastic Grab', 'U'))
except Exception:
    pass
SHORT = {'select': 'Select', 'move': 'Move', 'rotate': 'Rotate', 'scale': 'Scale', 'grab': 'Grab'}
MAX_RADIUS_UI = 0.12      # hand1: m (was 0.04); the engine clamps per object lattice (80 h, at most 120 mm)
OBJECT_TOOLS = ('move', 'rotate', 'scale')
TOOL_TIPS = {
    'select': 'Click selects (Shift or Ctrl+click adds); drag orbits.',
    'move': 'Drag the whole object in the view plane (one undo step per drag).',
    'rotate': 'Drag left / right to turn the object about the vertical, up / down to tilt it.',
    'scale': 'Drag right or up to grow the object, left or down to shrink it. Scale axis X / Y / Z stretches '
             'the base form along one axis.',
    'draw': 'Pushes the surface out along its normal (Ctrl inverts: carve).',
    'clay': 'Builds up towards a plane above the surface, like laying clay.',
    'carve': 'Pushes the surface in along its normal (Ctrl inverts: draw).',
    'inflate': 'Grows the form uniformly (F - A w).',
    'smooth': 'Low-pass of F inside the brush support.',
    'flatten': 'Pulls the field towards the tangent plane at the dab.',
    'crease': 'Pinches towards the stroke and cuts a groove.',
    'pinch': 'Draws the surface together towards the stroke.',
    'grab': 'Drags the field under the brush with the mouse.',
    'stretch': 'Stretches the field along the drag direction around the press point.',
    'snake': 'Snake Hook - pull a form out along the drag, keeping its thickness (a Grab whose pull is even across the '
             'brush and fades only at the rim).',
    'elastic': 'Elastic Grab - pulls like a lump of clay: the form around the brush follows elastically and the far '
               'side stays put (reach: short or wide).',
    'mask': 'Paints a mask that protects the field from other brushes (Ctrl erases).',
    'swell': 'Grows the surface until it meets a protected part (or the selected objects), then stops at the gap and '
             'conforms to it (a lid swelling to the eye).',
    'settle': 'Moves the surface toward the nearest protected part (or selected object) until it rests on it at the '
              'gap (a lid settling onto the eyeball).',
}

# name -> (lo, hi, factor, decimals, suffix, log)
PARAM_SPEC = {
    'radius': (0.0005, 0.5, 1000, 1, ' mm', True), 'size': (0.001, 1.0, 1000, 1, '', False),
    'round': (0.0, 0.1, 1000, 1, ' mm', False), 'height': (0.0005, 1.0, 1000, 1, ' mm', True),
    'radius_bottom': (0.0, 0.5, 1000, 1, ' mm', False), 'radius_top': (0.0, 0.5, 1000, 1, ' mm', False),
    'major': (0.001, 0.5, 1000, 1, ' mm', True), 'minor': (0.0005, 0.2, 1000, 1, ' mm', True),
    'radii': (0.001, 1.0, 1000, 1, '', False), 'normal': (-1.0, 1.0, 1, 3, '', False),
    'offset': (-0.5, 0.5, 1000, 1, ' mm', False), 'thickness': (0.0002, 0.2, 1000, 2, ' mm', True),
    'extent': (0.001, 1.0, 1000, 1, ' mm', True), 'blend': (0.0, 0.1, 1000, 1, ' mm', False),
    'translate': (-2.0, 2.0, 1000, 1, '', False), 'rotate': (-360.0, 360.0, 1, 1, '', False),
    'scale': (0.05, 20.0, 1, 3, '', False), 'rate': (-60.0, 60.0, 1, 2, ' rad/m', False),
    'g_min': (0.01, 1.0, 1, 2, '', False), 'distance': (-0.05, 0.05, 1000, 2, ' mm', False),
}
VEC_UNITS = {'size': 'mm', 'radii': 'mm', 'translate': 'mm', 'rotate': 'deg', 'scale': 'x', 'normal': ''}
FIELD_TOOLS = [('Pencil ridge', 'Pencil ridge', 'L: drag across a form; the stroke stays live and editable'),
               ('Pencil valley', 'Pencil valley', 'Shift+L: the same stroke cut in as a groove'),
               ('Cut (remove)', 'Cut', 'X: draw a closed outline; it is swept through and removed'),
               ('Cut and separate', 'Cut + separate', 'Shift+X: the part inside the outline becomes its own object, '
                                                       'moved apart by the chosen distance'),
               ('Splice', 'Splice', 'J: click face A, then face B: B turns and joins A with a smooth union'),
               ('Fill and splice', 'Fill + splice', 'Shift+J: click face A, then face B: a pillar of the chosen '
                                                    'profiles fills between them, matched on both faces'),
               ('Edit last pencil...', 'Edit pencil...', 'Width, depth, softness, profile and mode of the '
                                                         'selected object (its last pencil stroke)')]
MUTATION_BUTTONS = [('twist', 'Twist', {'rate': 8.0}), ('bend', 'Bend', {'rate': 4.0}),
                    ('taper', 'Taper', {'rate': 2.5, 'g_min': 0.2}),
                    ('stretch', 'Stretch', {'scale': [1.0, 1.4, 1.0]}), ('shell', 'Shell', {'thickness': 0.008}),
                    ('offset', 'Round', {'distance': 0.004}), ('transform', 'Move/Rotate', {})]


def _group(title):
    g = QGroupBox(title)
    v = QVBoxLayout(g)
    v.setContentsMargins(8, 10, 8, 8)
    v.setSpacing(5)
    return g, v


def _scroll(inner):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return sa


def lock_icon(size=16):
    """a padlock drawn in code (P13: no licensed art)"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    c = QColor(205, 210, 220)
    p.setPen(QPen(c, max(1.2, size / 11.0)))
    p.setBrush(Qt.NoBrush)
    w = size * 0.42
    p.drawArc(QRectF(size / 2 - w / 2, size * 0.1, w, size * 0.62), 0, 180 * 16)
    p.drawLine(QPointF(size / 2 - w / 2, size * 0.41), QPointF(size / 2 - w / 2, size * 0.5))
    p.drawLine(QPointF(size / 2 + w / 2, size * 0.41), QPointF(size / 2 + w / 2, size * 0.5))
    p.setBrush(c)
    p.drawRoundedRect(QRectF(size * 0.2, size * 0.48, size * 0.6, size * 0.44), size * 0.08, size * 0.08)
    p.end()
    return QIcon(pm)


class ToolPanel(QWidget):
    """hand1 (2026-11-07): Brush Controls (radius, strength) and Mirror first, then every tool as a compact grid that
    never needs scrolling, then the object transform options, the field tools and the finer brush settings."""
    toolChanged = Signal(str)
    settingsChanged = Signal(dict)
    showMaskChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = copy.deepcopy(BR.DEFAULT_SETTINGS)
        self.settings['radius'] = 0.012
        self.settings['strength'] = 0.5
        self.settings['curve'] = [1.0, 0.95, 0.8, 0.5, 0.2, 0.0]
        self.settings['layer'] = 'sculpt'         # the Smooth brush's layer (engine: LayerSmooth for detail/shape)
        self.settings['detail_scale'] = 0.002
        self.tool = 'select'
        self.last_sculpt = 'draw'
        self.field_active = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # hand1b: Brush and Tools sit in a fixed top part that never scrolls (the real window opened with them
        # scrolled out of sight); the field tools and the finer settings scroll below them
        top = QWidget()
        lay = QVBoxLayout(top)
        lay.setContentsMargins(6, 6, 6, 2)
        lay.setSpacing(8)
        inner = QWidget()
        rest = QVBoxLayout(inner)
        rest.setContentsMargins(6, 2, 6, 6)
        rest.setSpacing(8)

        g, v = _group('Brush')
        self.radius = FloatField('Radius', 0.0005, MAX_RADIUS_UI, 1000, 1, log=True, suffix=' mm', label_w=56)
        self.radius.setValue(self.settings['radius'])
        self.radius.valueChanged.connect(lambda x: self._set('radius', x))
        self.radius.setToolTip('[ ] or F-drag in the view. Up to 40 mm on a 0.5 mm lattice, 80 mm on a 1 mm lattice '
                               '(File > New bust). The smallest brush is 2 lattice cells (0.5 mm on her 0.25 mm lattice); '
                               'at that size a dab spreads to about 1.3x its radius.')
        v.addWidget(self.radius)
        # 11-10 (Blender sizes the brush in screen pixels: zoomed in on a lid it gets finer by itself): an option to keep
        # the brush's size on the screen as the view zooms (the radius in mm scales with each zoom step)
        self.radius_follow = QCheckBox('Radius follows the zoom (keeps its size on screen)')
        self.radius_follow.setToolTip('Zooming in makes the brush finer and zooming out larger, as in Blender (screen-space '
                                      'radius); the radius box shows the size in mm as it changes. Off: the radius stays '
                                      'in mm whatever the zoom.')
        self.radius_follow.toggled.connect(lambda on: self._set('radius_follow', bool(on)))
        v.addWidget(self.radius_follow)
        self.settings['radius_follow'] = False
        self.strength = FloatField('Strength', 0.0, 2.0, 1, 2, label_w=56)     # design 4.7: strength reaches 2
        self.strength.setValue(self.settings['strength'])
        self.strength.valueChanged.connect(lambda x: self._set('strength', x))
        self.strength.setToolTip('- and = keys, or Shift+F-drag in the view')
        v.addWidget(self.strength)
        # fs-shell 11-11 (design 4.7 "Autosmooth: a per-brush 0-1 dial"): the rebuilt Smooth over each dab's
        # footprint (fs-brush's kernels read settings['autosmooth']; without it each brush takes its own default)
        self.autosmooth = FloatField('Autosmooth', 0.0, 1.0, 1, 2, label_w=56)
        self.autosmooth.setValue(0.0)
        self.autosmooth.setToolTip('Smooths each dab\'s footprint as the brush goes (0 = off). Remembered per brush.')
        self.autosmooth.valueChanged.connect(lambda x: self._set('autosmooth', float(x)))
        v.addWidget(self.autosmooth)
        # fs-shell 11-11 (design 4.1 "Per-brush settings"): each brush remembers its radius, strength, falloff,
        # autosmooth and targeting; the lock shares radius and strength across brushes (the old behaviour)
        row = QHBoxLayout()
        self.share = QToolButton()
        self.share.setCheckable(True)
        self.share.setIcon(lock_icon(16))
        self.share.setText('Share size and strength')
        self.share.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.share.setFocusPolicy(Qt.NoFocus)
        self.share.setToolTip('Off (default): every brush keeps its own radius and strength, as in Blender.\n'
                              'On: one radius and strength shared by all brushes.')
        self.share.toggled.connect(self._share_changed)
        row.addWidget(self.share)
        self.mem_label = QLabel('')
        self.mem_label.setObjectName('dim')
        row.addWidget(self.mem_label, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('Mirror'))
        self.sym = []
        for i, ax in enumerate('XYZ'):
            b = QPushButton(ax)
            b.setCheckable(True)
            b.setFixedWidth(30)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip('Mirror strokes across the object\'s local %s = 0 plane' % ax)
            b.toggled.connect(self._sym_changed)
            row.addWidget(b)
            self.sym.append(b)
        # hand2f: world symmetry: mirror about the world planes through the origin (x = 0 is her midline); the
        # mirrored dab goes to whichever object's surface is nearest, so a left-limb stroke lands on the right limb
        self.sym_world = QPushButton('World')
        self.sym_world.setCheckable(True)
        self.sym_world.setFocusPolicy(Qt.NoFocus)
        self.sym_world.setToolTip('On: mirror across the WORLD x = 0 plane (her midline) onto whichever object is '
                                  'there (Alt+W). Off: across the object\'s own local plane.')
        self.sym_world.toggled.connect(lambda on: self._set('sym_world', bool(on)))
        row.addWidget(self.sym_world)
        row.addStretch(1)
        v.addLayout(row)
        lay.addWidget(g)

        g, v = _group('Tools (click or key; always switchable)')
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        grid = QGridLayout()
        grid.setSpacing(3)
        for i, (key, name, sc) in enumerate(TOOLS):
            b = QToolButton()
            b.setIcon(tool_icon(key, 20))
            b.setText('%s  %s' % (SHORT.get(key, name), sc))
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setToolTip('%s  [%s]\n%s' % (name, sc, TOOL_TIPS[key]))
            b.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            self.group.addButton(b)
            self.buttons[key] = b
            grid.addWidget(b, i // 3, i % 3)
        v.addLayout(grid)
        row = QHBoxLayout()
        lb = QLabel('Scale axis')
        row.addWidget(lb)
        self.axis = QComboBox()
        for text, key in (('All (uniform)', 'all'), ('X only', 'x'), ('Y only', 'y'), ('Z only', 'z')):
            self.axis.addItem(text, key)
        self.axis.setToolTip('Scale object (Y): uniform, or one axis (a stretch of the base form; '
                             'the sculpt layer keeps its shape)')
        row.addWidget(self.axis, 1)
        v.addLayout(row)
        hint = QLabel('Shift-drag: smooth.  Ctrl-drag: invert.  F-drag: radius.  Shift+F-drag: strength.  '
                      'Esc: back to the brush.')
        hint.setObjectName('dim')
        hint.setWordWrap(True)
        v.addWidget(hint)
        lay.addWidget(g)
        lay = rest

        g, v = _group('Field Tools (pencil, cut, splice, fill)')
        grid = QGridLayout()
        grid.setSpacing(4)
        self.field_buttons = {}
        for i, (label, text, tip) in enumerate(FIELD_TOOLS):
            b = QPushButton(text)
            b.setCheckable(label != 'Edit last pencil...')
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(tip + '\nThe settings dialog opens on first use; Ctrl+click opens it again.')
            b.setEnabled(False)                    # enabled when the app wires the callbacks (tools.register)
            grid.addWidget(b, i // 2, i % 2)
            self.field_buttons[label] = b
        v.addLayout(grid)
        self.field_hint = QLabel('A field tool takes the left mouse button until you pick another tool or press Esc.')
        self.field_hint.setObjectName('dim')
        self.field_hint.setWordWrap(True)
        v.addWidget(self.field_hint)
        lay.addWidget(g)

        g, v = _group('Brush settings')
        row = QHBoxLayout()
        lb = QLabel('Falloff')
        lb.setFixedWidth(78)
        row.addWidget(lb)
        self.falloff = QComboBox()
        self.falloff.addItems(['smooth', 'linear', 'sharp', 'constant', 'custom'])
        self.falloff.currentTextChanged.connect(self._falloff_changed)
        row.addWidget(self.falloff, 1)
        v.addLayout(row)
        self.curve = FalloffCurve()
        self.curve.curveChanged.connect(self._curve_changed)
        v.addWidget(self.curve)
        row = QHBoxLayout()
        lb = QLabel('Smooth on')
        lb.setFixedWidth(78)
        row.addWidget(lb)
        self.layer = QComboBox()
        for text, key in (('Sculpt layer (all)', 'sculpt'), ('Detail band, keep shape', 'detail'),
                          ('Shape band, keep detail', 'shape')):
            self.layer.addItem(text, key)
        self.layer.setToolTip('Edit by layer: the Smooth brush smooths the whole sculpt layer, or only its fine '
                              'detail band (shape kept), or only its shape band (detail kept).')
        self.layer.currentIndexChanged.connect(lambda i: self._set('layer', self.layer.itemData(i)))
        row.addWidget(self.layer, 1)
        v.addLayout(row)
        self.detail_scale = FloatField('Detail scale', 0.0005, 0.02, 1000, 1, log=True, suffix=' mm')
        self.detail_scale.setValue(0.002)
        self.detail_scale.setToolTip('The band split: features finer than this are "detail".')
        self.detail_scale.valueChanged.connect(lambda x: self._set('detail_scale', x))
        v.addWidget(self.detail_scale)
        self.spacing = FloatField('Spacing', 0.05, 1.0, 1, 2)
        self.spacing.setValue(self.settings['spacing'])
        self.spacing.valueChanged.connect(lambda x: self._set('spacing', x))
        v.addWidget(self.spacing)
        # 11-10: the stroke stabilizer (Blender's Stabilize Stroke): the brush follows the pointer on a string of this many
        # screen pixels (0 = off); a steady line for lid creases and lip borders
        self.lazy = FloatField('Stabilizer', 0.0, 120.0, 1, 0, suffix=' px')
        self.lazy.setValue(0.0)
        self.lazy.setToolTip('The brush follows the pointer on a string this long and moves only when it is pulled '
                             'taut: a steady stroke for fine lines (lid creases, lip borders). 0 = off.')
        self.lazy.valueChanged.connect(lambda x: self._set('lazy', float(x)))
        v.addWidget(self.lazy)
        self.settings['lazy'] = 0.0
        # brush targeting (11-08 night: a paint on one fold also painted the facing fold across a narrow
        # gap). The brush keeps its tilt to the surface under the pointer and its contour-following; these only choose
        # WHICH surface points receive the effect. Per tool; on by default for the Mask brush.
        self.face_only = QCheckBox('Only the surface the brush faces')
        self.face_only.setToolTip('A point receives the effect only if its surface turns the same way as the brush '
                                  '(within the facing cone). The far face across a gap or the back of a thin flap '
                                  'turns the other way, so it is left alone.')
        self.face_only.toggled.connect(lambda on: self._set_target('face_only', on))
        v.addWidget(self.face_only)
        self.face_cone = FloatField('Facing cone', 20.0, 170.0, 1, 0, suffix=' deg')
        self.face_cone.setValue(80.0)
        self.face_cone.setToolTip('How far a surface may turn from the brush and still be affected.')
        self.face_cone.valueChanged.connect(lambda x: self._set('face_cone', float(x)))
        v.addWidget(self.face_cone)
        self.connected = QCheckBox('Only the connected surface (not across gaps)')
        self.connected.setToolTip('A point receives the effect only if the surface runs to it from the brush centre '
                                  'without crossing a gap or a fold.')
        self.connected.toggled.connect(lambda on: self._set_target('connected', on))
        v.addWidget(self.connected)
        # fs-brush 11-12 (design: a tool changes only the surface it works on, nothing behind it, unless a setting or
        # a toggle says otherwise): the sheet under the brush
        # (core/brush.py sheet_gate; gpu/sculpt.py on the card). Opt-in per tool for now (it slows big brushes near thin
        # parts); to become the default once it runs on the card at full speed.
        self.sheet_only = QCheckBox('Protect surfaces behind')
        self.sheet_only.setToolTip('The brush changes only the surface it is on, the one connected to the point under the '
                                   'cursor: the back of a thin part (a lid, a lip, a fold), another surface across a gap '
                                   '(the eye under a lid, the other lip) and anything behind stay exactly as they were, '
                                   'with no labels needed. Slower with big brushes near thin parts.')
        self.sheet_only.toggled.connect(self._set_sheet)
        v.addWidget(self.sheet_only)
        # fs-brush 11-12, decided: the screen plane by default; the form rule on a visible switch
        self.drag_form = QCheckBox('Drag along the form')
        self.drag_form.setToolTip('Grab, Stretch and Snake Hook: off (default) the surface follows the cursor in the screen '
                                  'plane, as in Blender and ZBrush. On: the drag works on a plane through the surface '
                                  'normal, so pulling a point straight out along its normal works from any view; '
                                  'Ctrl pulls along the normal only.')
        self.drag_form.toggled.connect(lambda on: self._set('drag_form', bool(on)))
        v.addWidget(self.drag_form)
        self.mask_cutoff = FloatField('Mask: other side', 90.0, 175.0, 1, 0, suffix=' deg')
        self.mask_cutoff.setValue(BR.MASK_DIR_CUTOFF_DEG)
        self.mask_cutoff.setToolTip('Mask paint is not seen by a surface turned more than this from the side it was '
                                    'painted on (the facing sheet across a gap, the back of a flap). Rough or jagged '
                                    'parts of the painted surface itself stay painted. Default 110 deg.')
        self.mask_cutoff.valueChanged.connect(lambda x: setattr(BR, 'MASK_DIR_CUTOFF_DEG', float(x)))
        v.addWidget(self.mask_cutoff)
        self.collide = QCheckBox('Protected parts collide (no new overlap)')
        self.collide.setChecked(True)
        self.collide.setToolTip('Every brush keeps the surface out of locked protected parts (eyes and other parts) '
                                'by the gap: a lid pushed toward the eye slides over it. Where her body already overlaps a '
                                'part, edits never push it deeper. Unprotected parts do not collide.')
        self.collide.toggled.connect(lambda on: self._set('collide', bool(on)))
        v.addWidget(self.collide)
        self.settings['collide'] = True
        self.settings['collide_gap'] = 0.0003
        self.contact_gap = FloatField('Swell / Settle gap', 0.0, 2.0, 1000, 2, suffix=' mm')
        self.contact_gap.setValue(0.0)
        self.contact_gap.setToolTip('How far Swell to contact and Settle onto stop from the part: 0 = touching (the '
                                    'default). Other brushes keep a no-new-overlap gap of 0.3 mm from protected parts.')
        self.contact_gap.valueChanged.connect(lambda x: self._set('contact_gap', float(x)))
        v.addWidget(self.contact_gap)
        self.settings['contact_gap'] = 0.0
        self.use_mask = QCheckBox('Mask protects the field')
        self.use_mask.setChecked(True)
        self.use_mask.toggled.connect(lambda on: self._set('use_mask', on))
        v.addWidget(self.use_mask)
        self.show_mask = QCheckBox('Show mask in the viewport')
        self.show_mask.toggled.connect(self.showMaskChanged)
        v.addWidget(self.show_mask)
        hint = QLabel('Radius is in world millimetres on the surface. Spacing is the distance between dabs as a '
                      'fraction of the radius.')
        hint.setWordWrap(True)
        hint.setObjectName('dim')
        v.addWidget(hint)
        lay.addWidget(g)

        g, v = _group('Navigation')
        tips = QLabel('Sculpt: L-drag on a form\nOrbit: L-drag on empty space or Alt+L-drag\nPan: R- or M-drag\n'
                      'Zoom: wheel   Frame: . or Home   Views: 1 3 7 5 9\nUndo / redo: Ctrl+Z / Ctrl+Y\n'
                      'Viewport PNG: Ctrl+S')
        tips.setObjectName('dim')
        v.addWidget(tips)
        lay.addWidget(g)
        lay.addStretch(1)
        outer.addWidget(top)
        self.scroll = _scroll(inner)
        outer.addWidget(self.scroll, 1)
        self._mem_init()
        self.set_tool('select')

    # ---- per-brush settings and presets (fs-shell 11-11, design 4.1) -----------------------------------------------
    # Each brush remembers these (Blender 5.3 stores size and strength per brush); targeting lives in self._targets.
    MEM_KEYS = ('radius', 'strength', 'falloff', 'curve', 'autosmooth', 'hardness', 'foot')   # fs-shell-5: + hardness, foot
    SHARED_KEYS = ('radius', 'strength')
    DEFAULT_CURVE = (1.0, 0.95, 0.8, 0.5, 0.2, 0.0)
    presetsChanged = Signal()

    @staticmethod
    def _cfg():
        from .main_window import app_settings
        return app_settings()

    def _mem_init(self):
        self._mem_tool = None
        self._mem = {}
        self.share_rs = False
        try:
            cfg = self._cfg()
            self._mem = json.loads(cfg.value('brush_memory', '') or '{}') or {}
            self.share_rs = str(cfg.value('brush_share', 'false')).lower() == 'true'
        except Exception:
            self._mem = {}
        if not hasattr(self, '_targets'):
            self._targets = {}
        if not hasattr(self, '_sheet'):
            self._sheet = {}
        for tool, m in self._mem.items():
            if 'face_only' in m and 'connected' in m:
                self._targets[tool] = (bool(m['face_only']), bool(m['connected']))
            if 'sheet_only' in m:
                self._sheet[tool] = bool(m['sheet_only'])
        sh = self._mem.get('_shared') or {}
        if self.share_rs and sh:                    # the shared size as it was left
            self.settings['radius'] = min(max(float(sh.get('radius', 0.012)), 0.0005), MAX_RADIUS_UI)
            self.settings['strength'] = min(max(float(sh.get('strength', 0.5)), 0.0), 2.0)
            self.radius.setValue(self.settings['radius'])
            self.strength.setValue(self.settings['strength'])
        self.share.blockSignals(True)
        self.share.setChecked(self.share_rs)
        self.share.blockSignals(False)
        self._mem_timer = QTimer(self)
        self._mem_timer.setSingleShot(True)
        self._mem_timer.setInterval(1200)
        self._mem_timer.timeout.connect(self.flush_memory)

    def _is_brush(self, key):
        return key not in (None, 'select', 'move', 'rotate', 'scale')

    def _remember(self, k, v):
        """a change of a remembered setting while a brush is current goes into that brush's memory"""
        t = self._mem_tool
        if t is None or k not in self.MEM_KEYS:
            return
        if self.share_rs and k in self.SHARED_KEYS:
            t = '_shared'                      # the lock: one shared size; each brush's own waits for the lock to open
        self._mem.setdefault(t, {})[k] = list(v) if isinstance(v, (list, tuple)) else v
        self._mem_timer.start()

    def _snapshot(self, tool):
        s = self.settings
        m = {'radius': float(s['radius']), 'strength': float(s['strength']),
             'falloff': s.get('falloff', 'smooth'), 'curve': list(s.get('curve') or [])}
        if 'autosmooth' in s:
            m['autosmooth'] = float(s['autosmooth'])
        if 'hardness' in s:                            # fs-shell-5 11-14: each brush keeps its own hardness
            m['hardness'] = float(s['hardness'])
        if s.get('foot'):
            m['foot'] = str(s['foot'])
        return m

    def _recall(self, tool):
        """switching to a brush: its own settings come back (a brush used for the first time keeps the current ones
        and remembers them from now on)"""
        m = self._mem.get(tool)
        if m is None:
            # first use: the working size carries over; falloff and autosmooth start at the brush's own defaults
            # (fs-brush's per-tool AUTOSMOOTH applies while 'autosmooth' is unset)
            st_ = float(self.settings['strength'])
            from .toolreg import FT_DEFAULT_RATE, strength_unit
            if strength_unit(getattr(self, '_recall_prev', None)) != strength_unit(tool):
                # fs-shell-3 11-14: a rate (Swell / Settle m/s, Flatten a fraction a second) and today's strength are
                # different quantities: a first use across them starts from its own default, never the other's number
                st_ = FT_DEFAULT_RATE.get(tool, 0.5)
            m = self._mem[tool] = {'radius': float(self.settings['radius']),
                                   'strength': st_,
                                   'falloff': 'smooth', 'curve': list(self.DEFAULT_CURVE)}
        s = self.settings
        if not self.share_rs:
            if 'radius' in m:
                s['radius'] = min(max(float(m['radius']), 0.0005), MAX_RADIUS_UI)
                self.radius.setValue(s['radius'])
            if 'strength' in m:
                s['strength'] = min(max(float(m['strength']), 0.0), 2.0)
                self.strength.setValue(s['strength'])
        if 'hardness' in m:                            # fs-shell-5 11-14: the brush's own hardness comes back
            s['hardness'] = min(max(float(m['hardness']), 0.0), 1.0)
        elif not str(tool).startswith('ft_'):          # today's brush never given one: its falloff is the soft default
            s['hardness'] = 0.0
        if m.get('foot'):
            s['foot'] = str(m['foot'])
        if m.get('curve'):
            s['curve'] = list(m['curve'])
        if m.get('falloff'):
            s['falloff'] = m['falloff']
            self.falloff.blockSignals(True)
            self.falloff.setCurrentText(s['falloff'])
            self.falloff.blockSignals(False)
            self.curve.set_kind(s['falloff'], s.get('curve'))
        if 'autosmooth' in m:
            s['autosmooth'] = float(m['autosmooth'])
        else:
            s.pop('autosmooth', None)            # the engine applies the brush's own default (BR.AUTOSMOOTH)
        self.autosmooth.setValue(float(s.get('autosmooth', BR.AUTOSMOOTH.get(tool, 0.0))))
        self.mem_label.setText(('%s: shared size' if self.share_rs else '%s: its own settings') % tool)

    def _share_changed(self, on):
        self.share_rs = bool(on)
        try:
            self._cfg().setValue('brush_share', 'true' if on else 'false')
        except Exception:
            pass
        t = self._mem_tool
        if on:
            self._mem['_shared'] = {'radius': float(self.settings['radius']),
                                    'strength': float(self.settings['strength'])}
            self._mem_timer.start()
        elif self._is_brush(t):
            self._mem_tool = None              # the lock opened: this brush's own size comes back
            self._recall(t)
            self._mem_tool = t
            self.settingsChanged.emit(self.settings)
        if self._is_brush(t):
            self.mem_label.setText(('%s: shared size' if on else '%s: its own settings') % t)

    def flush_memory(self):
        """write the brushes' memory (debounced; the main window calls it on close)"""
        try:
            out = {}
            for tool, m in self._mem.items():
                d = dict(m)
                if tool in self._targets:
                    d['face_only'], d['connected'] = (bool(x) for x in self._targets[tool])
                if tool in getattr(self, '_sheet', {}):
                    d['sheet_only'] = bool(self._sheet[tool])
                out[tool] = d
            self._cfg().setValue('brush_memory', json.dumps(out))
        except Exception:
            pass

    def brush_memory(self, tool):
        return dict(self._mem.get(tool) or {})

    # presets: "Save as preset" in a brush's context menu; listed under their brush in the Space palette
    def presets(self, tool=None):
        try:
            ps = json.loads(self._cfg().value('brush_presets', '') or '[]') or []
        except Exception:
            ps = []
        return [p for p in ps if tool is None or p.get('tool') == tool]

    def _write_presets(self, ps):
        self._cfg().setValue('brush_presets', json.dumps(ps))
        self.presetsChanged.emit()

    def preset_default_name(self, tool=None):
        tool = tool or self.tool
        m = self._snapshot(tool) if tool == self._mem_tool else (self._mem.get(tool) or self._snapshot(tool))
        return '%g mm, s %.2f, %s' % (round(m['radius'] * 1000.0, 1), m['strength'], m.get('falloff', 'smooth'))

    def save_preset(self, name, tool=None):
        """the brush's current settings (its memory when it is not the current brush) under a name; same name
        replaces"""
        tool = tool or self.tool
        if not self._is_brush(tool):
            return None
        m = self._snapshot(tool) if tool == self._mem_tool else dict(self._mem.get(tool) or self._snapshot(tool))
        fo, co = self._target_for(tool)
        p = dict(m, tool=tool, name=str(name).strip() or self.preset_default_name(tool), face_only=bool(fo),
                 connected=bool(co), sheet_only=bool(getattr(self, '_sheet', {}).get(tool, BR.SHEET_DEFAULT)))
        ps = [q for q in self.presets() if not (q.get('tool') == tool and q.get('name') == p['name'])]
        ps.append(p)
        self._write_presets(ps)
        return p

    def delete_preset(self, tool, name):
        ps = [q for q in self.presets() if not (q.get('tool') == tool and q.get('name') == name)]
        self._write_presets(ps)

    def apply_preset(self, p):
        """pick the preset's brush and set its settings (radius and strength too, shared or not)"""
        tool = p.get('tool')
        if not self._is_brush(tool):
            return
        self.set_tool(tool)
        self._targets[tool] = (bool(p.get('face_only', False)), bool(p.get('connected', False)))
        if not hasattr(self, '_sheet'):
            self._sheet = {}
        self._sheet[tool] = bool(p.get('sheet_only', False))
        self._show_target(tool)
        s = self.settings
        if p.get('curve'):
            s['curve'] = list(p['curve'])
        if p.get('falloff'):
            s['falloff'] = p['falloff']
            self.falloff.blockSignals(True)
            self.falloff.setCurrentText(s['falloff'])
            self.falloff.blockSignals(False)
            self.curve.set_kind(s['falloff'], s.get('curve'))
        if 'autosmooth' in p:
            s['autosmooth'] = float(p['autosmooth'])
            self._remember('autosmooth', s['autosmooth'])
        else:
            s.pop('autosmooth', None)
            self._mem.get(tool, {}).pop('autosmooth', None)
        self.autosmooth.setValue(float(s.get('autosmooth', BR.AUTOSMOOTH.get(tool, 0.0))))
        for k in ('falloff', 'curve'):
            self._remember(k, s.get(k))
        self.set_radius(p.get('radius', s['radius']))         # these two emit the settings
        self.set_strength(p.get('strength', s['strength']))
        self.last_preset = p

    def relabel_tools(self, keys):
        """fs-shell 11-11: a rebinding in the Keys window shows on the tool buttons"""
        for key, name, sc in TOOLS:
            b = self.buttons.get(key)
            if b is not None:
                k = keys.get(key, sc)
                b.setText('%s  %s' % (SHORT.get(key, name), k))
                b.setToolTip('%s  [%s]\n%s' % (name, k or 'no key', TOOL_TIPS[key]))

    def set_field_tools(self, callbacks):
        """callbacks: {menu label: callback} from tools.register. After the callback (which may open a dialog that
        is cancelled) the buttons are re-synced from the real active field tool, so the panel never lies."""
        for label, b in self.field_buttons.items():
            cb = callbacks.get(label)
            if cb is not None:
                b.setEnabled(True)
                b.clicked.connect(lambda _=False, f=cb: (f(), self.set_field_tool_active(self.field_active)))

    def set_field_tool_active(self, label):
        """label = the active field tool's button label or None. Exactly one tool in the panel shows as active:
        a field tool unchecks the brush buttons; leaving it re-checks the brush."""
        self.field_active = label
        for k, b in self.field_buttons.items():
            b.blockSignals(True)
            b.setChecked(k == label)
            b.blockSignals(False)
        self.group.setExclusive(False)
        for k, b in self.buttons.items():
            b.setChecked(label is None and k == self.tool)
        self.group.setExclusive(True)

    def set_layer(self, key):
        i = self.layer.findData(key)
        if i >= 0:
            self.layer.setCurrentIndex(i)

    # (face_only, connected); Mask: direction-tagged + connected. (11-09 night: Smooth facing + connected by default tore
    # thin folds into spikes; Smooth keeps surfaces apart by topology instead)
    TARGET_DEFAULTS = {'mask': (False, True)}

    def _target_for(self, tool):
        if not hasattr(self, '_targets'):
            self._targets = {}
        return self._targets.setdefault(tool, self.TARGET_DEFAULTS.get(tool, (False, False)))

    def _set_target(self, k, on):
        tool = self.settings.get('tool')
        fo, co = self._target_for(tool)
        self._targets[tool] = (bool(on), co) if k == 'face_only' else (fo, bool(on))
        self._set(k, bool(on))

    def _set_sheet(self, on):
        """fs-brush 11-12: 'Protect surfaces behind' for the current brush (remembered per brush)"""
        tool = self.settings.get('tool')
        if not hasattr(self, '_sheet'):
            self._sheet = {}
        self._sheet[tool] = bool(on)
        self._set('sheet_only', bool(on))

    def _show_target(self, tool):
        fo, co = self._target_for(tool)
        so = bool(getattr(self, '_sheet', {}).get(tool, BR.SHEET_DEFAULT))   # a brush never set: the gate's default
        self.settings['sheet_only'] = so
        if hasattr(self, 'sheet_only'):
            self.sheet_only.blockSignals(True)
            self.sheet_only.setChecked(so)
            self.sheet_only.blockSignals(False)
        self.settings['face_only'], self.settings['connected'] = fo, co
        # 11-09: the Mask brush tags its paint with the painted sheet's direction (a facing sheet across any gap never
        # receives it; program.mask_sample_dir); its facing cone is not needed then
        self.settings['paint_dir'] = tool == 'mask'
        if tool == 'mask':
            self.settings['face_only'] = False
        self.settings.setdefault('face_cone', 80.0)
        for w, val in ((self.face_only, fo), (self.connected, co)):
            w.blockSignals(True)
            w.setChecked(val)
            w.blockSignals(False)

    def _set(self, k, v):
        self.settings[k] = v
        if hasattr(self, '_mem'):
            self._remember(k, v)
        self.settingsChanged.emit(self.settings)

    def _sym_changed(self, _=None):
        self._set('symmetry', [b.isChecked() for b in self.sym])

    def set_mirror(self, xyz, world=None):
        for b, on in zip(self.sym, xyz):
            b.setChecked(bool(on))
        if world is not None:
            self.sym_world.setChecked(bool(world))

    def _falloff_changed(self, kind):
        self.curve.set_kind(kind, self.settings.get('curve'))
        self._set('falloff', kind)

    def _curve_changed(self, curve):
        self._set('curve', list(curve))

    def set_tool(self, key):
        """Always emits (a click on the active tool still takes the mouse back from a field tool)."""
        self.tool = key
        if key not in ('select', 'move', 'rotate', 'scale'):
            self.last_sculpt = key
        self.group.setExclusive(False)
        for k, b in self.buttons.items():
            b.setChecked(k == key)
        self.group.setExclusive(True)
        if key not in ('select', 'move', 'rotate', 'scale'):
            self._recall_prev = self.settings.get('tool')
            self.settings['tool'] = key
            try:                                   # fs-shell-3 11-14: the engine's timed brushes take a RATE (Swell 0.004
                sp_ = self.strength.spin           # m/s): 2 decimals read 0.00; the box's rounding is never written
                sp_.blockSignals(True)
                sp_.setDecimals(4 if str(key).startswith('ft_') else 2)
                sp_.blockSignals(False)
            except Exception:
                pass
            if hasattr(self, '_mem') and key != self._mem_tool:
                self._mem_tool = None                  # nothing is remembered while the widgets change over
                self._recall(key)
                self._mem_tool = key
            if hasattr(self, 'face_only'):
                self._show_target(key)
        self.toolChanged.emit(key)
        self.settingsChanged.emit(self.settings)

    def set_radius(self, r):
        self.radius.setValue(min(max(float(r), 0.0005), MAX_RADIUS_UI), emit=True)

    def set_strength(self, s):
        self.strength.setValue(min(max(float(s), 0.0), 2.0), emit=True)

    def scale_radius(self, f):
        self.set_radius(self.settings['radius'] * f)

    def step_strength(self, d):
        self.set_strength(self.settings['strength'] + d)

    # fs-shell-5 11-14 (the first hands-on test: no way was found to adjust a tool's hardness /
    # softness): Hardness 0-1 on every brush, by that name; keys Shift+[ / Shift+] and Ctrl+F-drag (main_window,
    # viewport); the strip's Hardness slot shows it. What it does on a brush is that brush's definition (fs-tools:
    # toolset/ftools for the timed brushes; HARDNESS_CURVE below for today's brushes once fs-tools names it)
    def set_hardness(self, h):
        h = min(max(float(h), 0.0), 1.0)
        self._set('hardness', h)
        self._hardness_curve(h)

    def step_hardness(self, d):
        self.set_hardness(float(self.settings.get('hardness', 0.0)) + d)

    def _hardness_curve(self, h):
        """today's brushes read their falloff LUT ('custom' + 'curve', core/brush.falloff_lut): the hardness becomes
        that curve through fs-tools' definition (warp/profile.legacy_curve) when it exists"""
        tool = str(self.settings.get('tool', ''))
        if tool.startswith('ft_') or self.tool in ('select', 'move', 'rotate', 'scale'):
            return False
        try:
            from ..warp.profile import legacy_curve
        except Exception:
            return False
        try:
            foot = self.legacy_foot()
            from ..core.brush import LUT_N
            curve = [float(x) for x in legacy_curve(h, foot, n=LUT_N)]
        except Exception as e:
            print('hardness curve: %r' % (e,))
            return False
        self.settings['foot'] = foot
        self.settings['curve'] = curve
        self.settings['falloff'] = 'custom'
        self._remember('falloff', 'custom')
        self._set('curve', curve)
        return True

    FOOT_OF = {'smooth': 'soft', 'linear': 'even', 'sharp': 'sharp'}

    def legacy_foot(self):
        """the foot (soft / even / sharp) of today's brush: its falloff kind, or, once its Hardness made the falloff
        a curve, the foot that curve was made with"""
        fo = self.settings.get('falloff', 'smooth')
        if fo == 'custom':
            return self.settings.get('foot') or 'soft'
        return self.FOOT_OF.get(fo, 'soft')

    def set_legacy_foot(self, kind):
        """a foot button on today's brush: with a Hardness curve in use the curve is made again with this foot (the
        hardness kept); otherwise the falloff kind, as before"""
        if self.settings.get('falloff') == 'custom' and 'hardness' in self.settings:
            self.settings['foot'] = self.FOOT_OF.get(kind, 'soft')
            return self._hardness_curve(float(self.settings.get('hardness', 0.0)))
        return False

    def brush(self):
        s = copy.deepcopy(self.settings)
        if self.tool not in ('select', 'move', 'rotate', 'scale'):
            s['tool'] = self.tool
        return s


class PropertiesPanel(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.snap = None
        self.order = []
        self.node_id = None
        self._list_sig = None
        self._tree_sig = None
        self._form_sig = None
        self.fields = {}
        self.obj_fields = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        g, v = _group('Scene')
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setMaximumHeight(130)
        self.list.itemSelectionChanged.connect(self._list_selection)
        v.addWidget(self.list)
        row = QHBoxLayout()
        for name, fn, tip in (('Duplicate', self._duplicate, 'Ctrl+D: a copy in place that follows the mouse (type a distance; Enter or a click places it), sculpt included'),
                              ('Mirror X', self._mirror_dup, 'Ctrl+Shift+D: a mirrored copy across x = 0 (her left '
                                                             'built, her right made from it), sculpt included'),
                              ('Delete', self._delete, 'Delete'),
                              ('Hide / show', self._hide, 'H: hide in the view only (not an undo step); '
                                                          'Shift+H shows all, / isolates')):
            b = QPushButton(name)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)

        # hand2f: Merge (2+ objects, sculpt layers carried exactly) and Subtract / Intersect through the same command
        g, v = _group('Merge / combine (first selected = A)')
        row = QHBoxLayout()
        self.combine_buttons = []
        for kind, name, tip in (('union', 'Merge', 'Ctrl+J: the selected objects become ONE object, so strokes cross '
                                                   'the joints; every sculpt layer is carried exactly'),
                                ('subtract', 'Subtract', 'A minus the others (sculpt kept)'),
                                ('intersect', 'Intersect', 'Only where all overlap (sculpt kept)')):
            b = QPushButton(name)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=kind: self._boolean(k))
            row.addWidget(b)
            self.combine_buttons.append(b)
        v.addLayout(row)
        row = QHBoxLayout()
        self.smooth_merge = QCheckBox('Smooth joint')
        self.smooth_merge.setToolTip('Off: a sharp join (min). On: a smooth blend of radius k at every joint.')
        row.addWidget(self.smooth_merge)
        self.blend = FloatField('k', 0.0, 0.06, 1000, 1, suffix=' mm')
        row.addWidget(self.blend, 1)
        v.addLayout(row)
        self.world_frame = QCheckBox('Merged object in the world frame (local Mirror X = the midline x = 0)')
        self.world_frame.setToolTip('Off: the merged object takes the first-selected object\'s frame.')
        v.addWidget(self.world_frame)
        self._load_merge_settings()
        for w in (self.smooth_merge, self.world_frame):
            w.toggled.connect(lambda _=False: self._save_merge_settings())
        self.blend.valueChanged.connect(lambda _=None: self._save_merge_settings())
        self.combine_hint = QLabel('Select two or more objects (Shift+click) to merge them.')
        self.combine_hint.setObjectName('dim')
        self.combine_hint.setWordWrap(True)
        v.addWidget(self.combine_hint)
        lay.addWidget(g)
        # fs-shell-2 11-14 (decided: ONE Merge in the app): in the approved layout Merge is the strip's tool (fs-tools'
        # job); Subtract and Intersect stay under Edit > Object tools > Combine selected and the right-click menu
        self.combine_group = g
        if __import__('os').environ.get('FS_LAYOUT', '') != 'classic':
            g.hide()

        g, v = _group('Layers')
        self.title = QLabel('No object selected')
        self.title.setStyleSheet('font-weight: 600;')
        v.addWidget(self.title)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(['Layer / node', 'Value'])
        self.tree.setColumnWidth(0, 170)
        self.tree.setMinimumHeight(170)
        self.tree.itemSelectionChanged.connect(self._tree_selection)
        v.addWidget(self.tree)
        lay.addWidget(g)

        g, v = _group('Node parameters (live)')
        self.node_title = QLabel('-')
        self.node_title.setObjectName('dim')
        v.addWidget(self.node_title)
        self.form_box = QWidget()
        self.form = QVBoxLayout(self.form_box)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(4)
        v.addWidget(self.form_box)
        v.addWidget(QLabel('Add a mutation above the selected node:'))
        grid = QGridLayout()
        grid.setSpacing(4)
        self.mut_buttons = []
        for i, (kind, name, params) in enumerate(MUTATION_BUTTONS):
            b = QPushButton(mutation_icon(kind), name)
            b.clicked.connect(lambda _=False, k=kind, p=params: self._wrap(k, p))
            grid.addWidget(b, i // 3, i % 3)
            self.mut_buttons.append(b)
        v.addLayout(grid)
        self.remove_node = QPushButton('Remove selected node')
        self.remove_node.clicked.connect(self._unwrap)
        v.addWidget(self.remove_node)
        lay.addWidget(g)

        g, v = _group('Object')
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._rename)
        row = QHBoxLayout()
        lb = QLabel('Name')
        lb.setFixedWidth(78)
        row.addWidget(lb)
        row.addWidget(self.name_edit)
        v.addLayout(row)
        self.obj_fields['translate'] = VecField('Move (mm)', -2.0, 2.0, 1000, 1, step=1.0)
        self.obj_fields['rotate'] = VecField('Rotate (deg)', -360, 360, 1, 1, step=5.0)
        self.obj_fields['scale'] = FloatField('Scale', 0.05, 10.0, 1, 3, log=True)
        for k in ('translate', 'rotate', 'scale'):
            self.obj_fields[k].valueChanged.connect(lambda val, key=k: self._transform(key, val))
            v.addWidget(self.obj_fields[k])
        row = QHBoxLayout()
        lb = QLabel('Albedo')
        lb.setFixedWidth(78)
        row.addWidget(lb)
        self.colour_btn = QPushButton('')
        self.colour_btn.clicked.connect(self._pick_colour)
        row.addWidget(self.colour_btn, 1)
        v.addLayout(row)
        self.obj_fields['roughness'] = FloatField('Roughness', 0.0, 1.0, 1, 2)
        self.obj_fields['roughness'].valueChanged.connect(
            lambda val: self._post_props({'material': {'roughness': float(val)}}))
        v.addWidget(self.obj_fields['roughness'])
        # fs-shell 11-13 (Tier 0 part 4, X-ray): how see-through this object is in Clay and Look (view only:
        # not saved, no undo step); the View bar's X-ray slider makes every object see-through at once
        self.obj_fields['opacity'] = FloatField('Opacity', 0.0, 1.0, 1, 2)
        self.obj_fields['opacity'].setToolTip('How see-through this object is in Clay and Look (1 = solid). A view '
                                              'setting: not saved, no undo step. The X-ray slider above the view does '
                                              'every object at once.')
        self.obj_fields['opacity'].valueChanged.connect(self._opacity)
        v.addWidget(self.obj_fields['opacity'])
        self.viewport = None                 # set by the window (the X-ray's per-object opacity is view state)
        lay.addWidget(g)
        lay.addStretch(1)
        outer.addWidget(_scroll(inner))
        self._enable()

    # ---- helpers -----------------------------------------------------------------------------------------------
    def obj(self, oid=None):
        if self.snap is None:
            return None
        oid = self.active() if oid is None else oid
        for o in self.snap['objects']:
            if o['id'] == oid:
                return o
        return None

    def active(self):
        return self.order[0] if self.order else None

    def _enable(self):
        has = self.obj() is not None
        for b in self.mut_buttons + [self.remove_node, self.colour_btn, self.name_edit]:
            b.setEnabled(has)
        two = len(self.order) >= 2
        for b in self.combine_buttons:
            b.setEnabled(two)
        if two and self.snap:
            names = [self.obj(o)['name'] for o in self.order if self.obj(o)]
            self.combine_hint.setText('%d selected (A = %s). Merge makes them one object named %s; the sculpt layers '
                                      'come along; one undo step.' % (len(names), names[0], names[0]))
        else:
            self.combine_hint.setText('Select two or more objects (Shift+click in the view, Ctrl+click here).')

    # ---- snapshot ----------------------------------------------------------------------------------------------
    def update_snapshot(self, snap):
        self.snap = snap
        ids = [o['id'] for o in snap['objects']]
        self.order = [o for o in self.order if o in ids]
        sig = [(o['id'], o['name'], o['visible'], o.get('hidden', False)) for o in snap['objects']]
        if sig != self._list_sig:
            self._list_sig = sig
            self.list.blockSignals(True)
            self.list.clear()
            for o in snap['objects']:
                shown = o['visible'] and not o.get('hidden', False)
                it = QListWidgetItem(('%s' if shown else '%s  (hidden)') % o['name'])
                it.setData(Qt.UserRole, o['id'])
                if not shown:
                    it.setForeground(QColor('#8a919b'))
                self.list.addItem(it)
            self.list.blockSignals(False)
        if snap.get('select') is not None:
            if snap.get('select_node') is not None:
                self.node_id = snap['select_node']
            self.set_selection(snap['select'])
        else:
            if snap.get('select_node') is not None:
                self.node_id = snap['select_node']
            self._sync_list()
            self.refresh()

    def set_selection(self, oids):
        old = list(self.order)
        self.order = [o for o in oids if self.obj(o) is not None]
        if old[:1] != self.order[:1]:
            o = self.obj()
            if o is not None and (self.node_id is None or not self._node_exists(o, self.node_id)):
                self.node_id = o['base']['id']
        self._sync_list()
        self.refresh()
        self.selectionChanged.emit(list(self.order))

    def _sync_list(self):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setSelected(it.data(Qt.UserRole) in self.order)
        self.list.blockSignals(False)

    def _list_selection(self):
        cur = [it.data(Qt.UserRole) for it in self.list.selectedItems()]
        order = [o for o in self.order if o in cur] + [o for o in cur if o not in self.order]
        self.set_selection(order)

    @staticmethod
    def _node_exists(o, nid):
        stack = [o['base']]
        while stack:
            d = stack.pop()
            if d['id'] == nid:
                return True
            stack.extend(d.get('children', []))
        return False

    def _find_node(self, o, nid):
        stack = [o['base']]
        while stack:
            d = stack.pop()
            if d['id'] == nid:
                return d
            stack.extend(d.get('children', []))
        return None

    # ---- the layers tree + forms --------------------------------------------------------------------------------
    def refresh(self):
        o = self.obj()
        self._enable()
        if o is None:
            self.title.setText('No object selected')
            self.tree.clear()
            self._tree_sig = None
            self._clear_form()
            self._form_sig = None
            self.node_title.setText('-')
            return
        if self.node_id is None or not self._node_exists(o, self.node_id):
            self.node_id = o['base']['id']
        self.title.setText('%s   (object %d)' % (o['name'], o['id']))

        def sig_of(d):
            return (d['id'], d['kind'], tuple(sig_of(c) for c in d.get('children', [])))
        tsig = (o['id'], sig_of(o['base']))
        if tsig != self._tree_sig:
            self._tree_sig = tsig
            self._build_tree(o)
        else:
            self._update_tree(o)
        self._select_tree_node()
        node = self._find_node(o, self.node_id)
        fsig = (o['id'], node['id'], node['kind'])
        if fsig != self._form_sig:
            self._form_sig = fsig
            self._build_form(o, node)
        else:
            for k, w in self.fields.items():
                if not w.busy():
                    w.setValue(node['params'][k])
        self._update_object_fields(o)

    def _summary(self, d):
        p = d['params']
        k = d['kind']
        mm = lambda v: '%.1f' % (1000 * float(v))  # noqa
        if k == 'sphere':
            return 'r %s mm' % mm(p['radius'])
        if k == 'box':
            return '%s x %s x %s mm' % tuple(mm(v) for v in p['size'])
        if k == 'cylinder':
            return 'r %s, h %s mm' % (mm(p['radius']), mm(p['height']))
        if k == 'capsule':
            return 'r %s, h %s mm' % (mm(p['radius']), mm(p['height']))
        if k == 'cone':
            return 'r %s/%s, h %s mm' % (mm(p['radius_bottom']), mm(p['radius_top']), mm(p['height']))
        if k == 'torus':
            return 'R %s, r %s mm' % (mm(p['major']), mm(p['minor']))
        if k == 'ellipsoid':
            return '%s x %s x %s mm' % tuple(mm(v) for v in p['radii'])
        if k in ops.BOOLEANS:
            return 'sharp' if p['blend'] <= 0 else 'blend k %s mm' % mm(p['blend'])
        if k in ('twist', 'bend', 'taper'):
            return '%.2f rad/m' % p['rate']
        if k == 'shell':
            return 't %s mm' % mm(p['thickness'])
        if k == 'offset':
            return 'd %s mm' % mm(p['distance'])
        if k == 'transform':
            return 'scale %s' % ', '.join('%.2f' % v for v in p['scale'])
        return ''

    def _build_tree(self, o):
        self.tree.blockSignals(True)
        self.tree.clear()
        base = QTreeWidgetItem(['Base (analytic tree)', '%d nodes, L %.3f' % (o['nodes'], o['L_base'])])
        base.setData(0, Qt.UserRole, ('layer', 'base'))
        self.tree.addTopLevelItem(base)
        self._node_items = {}

        def add(parent, d):
            nm = d.get('name')
            is_data = bool(nm and nm.startswith('tooldata:'))
            if is_data:
                try:
                    import json
                    nm = '%s data' % json.loads(nm[len('tooldata:'):]).get('tool', 'tool')
                except Exception:
                    nm = 'tool data'
            it = QTreeWidgetItem([d['kind'] if nm in (None, d['kind']) else '%s (%s)' % (d['kind'], nm),
                                  self._summary(d)])
            it.setData(0, Qt.UserRole, ('node', d['id']))
            parent.addChild(it)
            self._node_items[d['id']] = it
            if is_data:
                # V1.1: a tool's generated subtree (hundreds of pieces for a long pencil) is edited through its tool
                # (Tools > Edit pencil), so it is not listed node by node; that kept the UI thread responsive
                n = sum(1 for _ in _walk(d)) - 1
                it.setText(1, '%d generated nodes' % n)
                return
            for c in d.get('children', []):
                add(it, c)

        def _walk(d):
            yield d
            for c in d.get('children', []):
                yield from _walk(c)
        add(base, o['base'])
        self._layer_items = {}
        for key, name in (('sculpt', 'Sculpt (delta-F grid)'), ('detail', 'Detail (placeholder)'),
                          ('colour', 'Colour (albedo)'), ('material', 'Material')):
            it = QTreeWidgetItem([name, ''])
            it.setData(0, Qt.UserRole, ('layer', key))
            self.tree.addTopLevelItem(it)
            self._layer_items[key] = it
        self._update_tree(o)
        self.tree.expandAll()
        self.tree.blockSignals(False)

    def _update_tree(self, o):
        stack = [o['base']]
        while stack:
            d = stack.pop()
            it = self._node_items.get(d['id'])
            if it is not None:
                it.setText(1, self._summary(d))
            stack.extend(d.get('children', []))
        self.tree.topLevelItem(0).setText(1, '%d nodes, L %.3f' % (o['nodes'], o['L_base']))
        self._layer_items['sculpt'].setText(1, '%d bricks, h %.1f mm, L %.3f' % (
            o['bricks'], o['sculpt_h'] * 1000, o['L_sculpt']))
        self._layer_items['detail'].setText(1, 'off (V1 placeholder)')
        a = o['colour']['albedo']
        self._layer_items['colour'].setText(1, 'albedo %.2f %.2f %.2f' % tuple(a))
        self._layer_items['material'].setText(1, 'roughness %.2f' % o['material'].get('roughness', 0.5))

    def _select_tree_node(self):
        it = self._node_items.get(self.node_id) if hasattr(self, '_node_items') else None
        if it is not None and not it.isSelected():
            self.tree.blockSignals(True)
            self.tree.clearSelection()
            it.setSelected(True)
            self.tree.blockSignals(False)

    def _tree_selection(self):
        its = self.tree.selectedItems()
        if not its:
            return
        tag = its[0].data(0, Qt.UserRole)
        if tag and tag[0] == 'node':
            self.node_id = tag[1]
            self.refresh()

    def _clear_form(self):
        while self.form.count():
            w = self.form.takeAt(0).widget()
            if w is not None:
                w.hide()                         # fs-shell-2 11-14: out of sight now; deleteLater alone left the
                w.deleteLater()                  # old row drawn under the new one until its deferred delete ran
        self.fields = {}

    def _build_form(self, o, node):
        self._clear_form()
        kind = node['kind']
        group = 'primitive' if kind in ops.PRIMITIVES else 'boolean' if kind in ops.BOOLEANS else 'mutation'
        self.node_title.setText('%s  (%s, node %d)' % (kind, group, node['id']))
        if kind in ('fv8store', 'lattice'):   # milestone: her store / a protected part's field file: read-only here
            for k, v in node['params'].items():
                lb = QLabel('%s: %s' % (k, v))
                lb.setWordWrap(True)
                lb.setObjectName('dim')
                self.form.addWidget(lb)
            lb = QLabel('Her published field, read natively at every level. Edit it with the brushes, Merge with '
                        'protected parts and Cut; File > Write her edits... makes a NEW version (never the published one).')
            lb.setWordWrap(True)
            self.form.addWidget(lb)
            return
        if kind == 'sampled':                  # V1.2 field region: read-only (its data comes from the store version)
            p = node['params']
            for k in ('source', 'crop', 'centre', 'size', 'clamp'):
                v = p.get(k)
                if isinstance(v, list):
                    v = ', '.join('%.1f' % (1000 * float(x)) for x in v) + ' mm'
                elif k == 'clamp':
                    v = '%.1f mm' % (1000 * float(v))
                lb = QLabel('%s: %s' % (k, v))
                lb.setWordWrap(True)
                lb.setObjectName('dim')
                self.form.addWidget(lb)
            return
        for k, val in node['params'].items():
            lo, hi, fac, dec, suf, log = PARAM_SPEC.get(k, (-10.0, 10.0, 1, 3, '', False))
            if isinstance(val, str) or (isinstance(val, list) and any(isinstance(x, str) for x in val)):
                lb = QLabel('%s: %s' % (k, val))   # 11-09: text params (a file, a sha) shown, never as numbers
                lb.setWordWrap(True)
                lb.setObjectName('dim')
                self.form.addWidget(lb)
                continue
            if isinstance(val, list):
                unit = VEC_UNITS.get(k, '')
                if k == 'scale':
                    # hand2b (verifier item 17): a vector scale may be negative (Mirror-duplicate's 'mirror x' node
                    # is [-1, 1, 1]); the 0.05..20 range clamped -1 to 0.05 and an edit of y or z posted the whole
                    # vector, un-mirroring the part. Range -20..20; a component dragged inside +-lo keeps its sign.
                    w = VecField('%s%s' % (k, ' (%s)' % unit if unit else ''), -hi, hi, fac, dec)
                    w.setValue(val)

                    def _scale_vec(v, key=k, old=[float(x) for x in val], wid=w, lo=lo):
                        out = []
                        for x, o in zip(v, old):
                            x = float(x)
                            if abs(x) < lo:
                                x = lo if (o >= 0 if x == 0.0 else x > 0) else -lo
                            out.append(x)
                        old[:] = out
                        wid.setValue(out)
                        self._param(key, out)
                    w.valueChanged.connect(_scale_vec)
                    self.form.addWidget(w)
                    self.fields[k] = w
                    continue
                w = VecField('%s%s' % (k, ' (%s)' % unit if unit else ''), lo, hi, fac, dec)
                w.setValue(val)
                w.valueChanged.connect(lambda v, key=k: self._param(key, [float(x) for x in v]))
            else:
                w = FloatField(k, lo, hi, fac, dec, log=log, suffix=suf)
                w.setValue(float(val))
                w.valueChanged.connect(lambda v, key=k: self._param(key, float(v)))
            self.form.addWidget(w)
            self.fields[k] = w
        if not node['params']:
            lb = QLabel('(no parameters)')
            lb.setObjectName('dim')
            self.form.addWidget(lb)

    def _update_object_fields(self, o):
        t = o['transform']
        for k in ('translate', 'rotate'):
            if not self.obj_fields[k].busy():
                self.obj_fields[k].setValue(t[k])
        if not self.obj_fields['scale'].busy():
            self.obj_fields['scale'].setValue(float(t['scale']))
        if not self.obj_fields['roughness'].busy():
            self.obj_fields['roughness'].setValue(float(o['material'].get('roughness', 0.5)))
        if not self.obj_fields['opacity'].busy() and self.viewport is not None:
            self.obj_fields['opacity'].setValue(float(self.viewport.obj_opacity.get(o['id'], 1.0)))
        if not self.name_edit.hasFocus():
            self.name_edit.setText(o['name'])
        a = o['colour']['albedo']
        disp = [int(round(255 * min(max(c, 0.0), 1.0) ** (1 / 2.2))) for c in a]
        self.colour_btn.setStyleSheet('QPushButton { background: rgb(%d,%d,%d); }' % tuple(disp))

    # ---- actions ------------------------------------------------------------------------------------------------
    def _opacity(self, val):
        """fs-shell 11-13: the selected objects' opacity (view state on the viewport; the card composites it)"""
        vp = self.viewport
        if vp is None:
            return
        for oid in (self.order or []):
            if float(val) >= 0.999:
                vp.obj_opacity.pop(oid, None)
            else:
                vp.obj_opacity[oid] = float(val)
        if vp.mode not in ('shaded', 'matcap'):
            w = vp.window()
            if hasattr(w, 'set_mode'):
                w.set_mode('matcap')
        vp.request(False)

    def _param(self, key, value):
        o = self.obj()
        if o is not None and self.node_id is not None:
            self.engine.post('param', o['id'], self.node_id, {key: value})

    def _post_props(self, props):
        o = self.obj()
        if o is not None:
            self.engine.post('props', o['id'], props)

    def _transform(self, key, val):
        self._post_props({'transform': {key: float(val) if key == 'scale' else [float(x) for x in val]}})

    def _rename(self):
        o = self.obj()
        name = self.name_edit.text().strip()
        if o is not None and name and name != o['name']:
            self.engine.post('props', o['id'], {'name': name})

    def _pick_colour(self):
        o = self.obj()
        if o is None:
            return
        a = o['colour']['albedo']
        c = QColorDialog.getColor(QColor(*[int(round(255 * min(max(x, 0.0), 1.0) ** (1 / 2.2))) for x in a]),
                                  self, 'Albedo')
        if c.isValid():
            lin = [round((v / 255.0) ** 2.2, 4) for v in (c.red(), c.green(), c.blue())]
            self.engine.post('props', o['id'], {'colour': {'albedo': lin}})

    def _wrap(self, kind, params):
        o = self.obj()
        if o is None:
            return
        mk = 'transform' if kind == 'stretch' else kind
        self.engine.post('wrap', o['id'], self.node_id, mk, copy.deepcopy(params))

    def _unwrap(self):
        o = self.obj()
        if o is not None and self.node_id is not None:
            self.engine.post('unwrap', o['id'], self.node_id)

    # ---- hand2f: merge settings (remembered), merge, mirror-duplicate, mask, view hide --------------------------------
    def _load_merge_settings(self):
        try:
            from .main_window import app_settings
            cfg = app_settings()
            self.blend.setValue(float(cfg.value('merge_k_m', 0.010)))
            self.smooth_merge.setChecked(str(cfg.value('merge_smooth', 'true')).lower() == 'true')
            self.world_frame.setChecked(str(cfg.value('merge_world', 'false')).lower() == 'true')
        except Exception:
            self.blend.setValue(0.010)
            self.smooth_merge.setChecked(True)

    def _save_merge_settings(self):
        try:
            from .main_window import app_settings
            cfg = app_settings()
            cfg.setValue('merge_k_m', float(self.blend.value()))
            cfg.setValue('merge_smooth', 'true' if self.smooth_merge.isChecked() else 'false')
            cfg.setValue('merge_world', 'true' if self.world_frame.isChecked() else 'false')
        except Exception:
            pass

    def merge_k(self, smooth=None):
        smooth = self.smooth_merge.isChecked() if smooth is None else smooth
        k = float(self.blend.value())
        if smooth and k <= 0:
            k = 0.010
        return k if smooth else 0.0

    def merge(self, kind='union', k=None):
        """Merge / Subtract / Intersect the selection (2+), the first-selected object first."""
        if len(self.order) < 2:
            self.engine.message.emit('Merge: select two or more objects first (Shift+click adds to the selection).')
            return
        if kind == 'union' and _registry_merge():
            return
        if kind in ('subtract', 'intersect') and _registry_cut(self.order, kind):
            return
        k = self.merge_k() if k is None else float(k)
        self.engine.post('merge', list(self.order), kind, k, 'world' if self.world_frame.isChecked() else 'first')

    def _boolean(self, kind, force_smooth=False):
        if len(self.order) < 2:
            return
        if kind in ('subtract', 'intersect') and _registry_cut(self.order, kind):
            return
        self.merge(kind, self.merge_k(True) if force_smooth else None)

    def _duplicate(self):
        # fs-shell 11-11 (as Blender's Shift+D): a copy IN PLACE that follows the mouse (a typed move:
        # type a distance, X / Y / Z for the axis; Enter or a click places it, Esc leaves it where the original is);
        # copies named 'A copy', 'A copy 2' ... (ui/flip_ui.duplicate_in_place)
        if self.active() is not None:
            from .flip_ui import duplicate_in_place
            duplicate_in_place(self.window(), self.active())

    def _mirror_dup(self):
        # fs-shell-3 11-14 ((g), ONE mirror): Ctrl+Shift+D / Mirror X run fs-tools' tools.mirror across the world X plane
        # (toolset/copies.MirrorAcross, exact); the engine's own mirror_dup stays for the classic layout and old logs
        h = getattr(self.engine, '_job_handlers', None) or {}
        for oid in list(self.order):
            if 'tools.mirror' in h:
                self.engine.post('tools.mirror', int(oid), 'world_x')
            else:
                self.engine.post('mirror_dup', oid)

    def _mask(self, op):
        for oid in list(self.order):
            self.engine.post('mask', oid, op)

    def _delete(self):
        if self.order:
            self.engine.post('remove', list(self.order))

    def _hide(self):
        """hand2f: hide / show in the VIEW (not a logged property, not an undo step; the hash ignores it)"""
        if self.order:
            self.engine.post('view', 'toggle', list(self.order))
