"""The top menus, short and grouped by what the user is doing (fs-shell-5, 2026-11-14).

The first hands-on test: the menus at the top (Edit, View, etc.) opened very long, unsorted lists of unclear options,
and the other menus (assigning materials, animation and rigging) could not be found. The cause: the approved top bar's
four menus (design 3.1) took every other menu folded in, so View held 60 rows in two columns and Edit four nested lists;
the panels sat three levels down (View > Panels).

Now eight menus, each a screen's height or less, named by the job:
  File    projects, reference photos, pictures and stills, her field
  Edit    undo, copy, delete, select, hide
  Add     shapes, masses, cutters, garments, the mouth library
  Sculpt  the brushes by what they do, brush size / strength / hardness, mirror, mask
  Object  select and transform, merge and combine, change its shape, the tools on the form, measure, check, protect
  View    the look (Clay / Look / Silhouette / X-ray), the camera, grid, light, section
  Panels  every panel in one place (Materials, Scene, Look / Render, References, Layers, Regions, Hair, Timeline, the
          keyboard map, the quick keys); Animation and rigging named as not built yet
  Help    quick start, what to try, the keyboard map, the quick keys, search, failures; the sessions' checks last
Nothing is lost: every action the window and the lanes made keeps its id (the path and name the rebound keys, the
categories column, the view bar and the search use: stamped as 'fs_path' before anything moves; shell.menu_actions
reads it), its key and its place in the search, which now names where it is ('fs_where'). An action no entry below
places goes to 'More' at the end of its menu, so a lane's new action still shows. LAYOUT is the whole map: edit it to
move something; tests/test_window_builds.py opens every menu."""
import re

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from .shell import _clean


def norm(s):
    return re.sub(r'\s+', ' ', str(s).replace('&', '')).strip().lower()


# where an action that no entry places goes ('More' at the end of that menu), by the menu it was first made in
HOME = {'File': 'File', 'Edit': 'Edit', 'Create': 'Add', 'Library': 'Add', 'Sculpt': 'Sculpt', 'Tools': 'Object',
        'View': 'View', 'Render': 'File', 'Help': 'Help'}
# old routes to the same panels (the panel tabs and the Panels menu open them now): kept for their ids, not shown
HIDE = ['View/Look / Render', 'View/Layers (record a pass, dial it 0-1, snapshots)',
        'View/Regions (paint, lasso, from mask or part; hide, isolate, box hide)', 'Tools/Hair: grow, cut and comb...',
        'Tools/Materials Generators...', 'Tools/Materials & Generators...', 'View/Tool shelf', 'View/Tool settings panel',
        'View/Tool names on the shelf', 'Help/Keys window (every key, the current tool, a keyboard map)',
        'Tools/Combine selected/Merge smooth (blend k)', 'View/Panels/*']
# plainer words for a few (the id keeps the name it was made with: 'fs_text', read by shell.atext)
RENAME = {'Sculpt/Stretch': 'Shape (pulls the surface as far as you drag)',
          'Edit/Duplicate object (with its sculpt)': 'Copy (duplicate, with its sculpt)',
          'Edit/Mirror-duplicate across X (world, with its sculpt)': 'Mirror copy across X (with its sculpt)',
          'File/Save viewport PNG (screens/)': 'Save a picture of the view (PNG)',
          'File/Save window PNG (screens/)': 'Save a picture of the whole window (PNG)',
          'Tools/The toolset/Shape library (Add)': 'Shapes (the library: primitives, your saved forms)',
          'Sculpt/Move / Grab': 'Grab (move the surface under the brush)'}

S = '-'
LAYOUT = [
    ('File', [
        'File/New...',
        ('sub', 'New from a start (head, figure, hand, bust, empty)', [
            'File/New head (sphere + neck, 1 mm lattice, Mirror X; then photos)',
            'File/New figure (a 1680 mm standing blockout, 1 mm lattice)',
            'File/New hand (palm and fingers, 0.5 mm lattice)',
            'File/New bust (a plain sphere, 1 mm lattice)',
            'File/New empty scene']),
        'File/Open project (.fsproj)...', 'File/Save project', 'File/Save project as...', S,
        ('sub', 'Reference photos (add, line up, show)', [
            'File/Add reference images (front, side)...', 'File/Reference Rig (align front and side)...', S,
            'View/Reference planes visible', 'View/Reference lens (5 deg, near orthographic)',
            'View/Orthographic (parallel rays)']),
        ('sub', 'Pictures and stills', [
            'File/Save viewport PNG (screens/)', 'File/Save window PNG (screens/)', S,
            'Render/Full-detail still of this view (skin, eyes, hair)...',
            "Render/This view, converged at full resolution (the window's own look)", 'Render/Open a still...']),
        ('sub', 'Her field (the published figure)', [
            'File/Open her field (editable copy, whole body)', 'File/Write her edits to a new version...', S,
            'File/Open field region...', 'File/Write region back (new version)...', S,
            'View/Published field beside the figure (view only)']),
        S, 'File/Quit']),
    ('Edit', [
        'Edit/Undo', 'Edit/Redo', S,
        'Edit/Duplicate object (with its sculpt)', 'Edit/Mirror-duplicate across X (world, with its sculpt)',
        'Edit/Delete object', 'Edit/Rename object...', S,
        'Edit/Select all', 'Edit/Invert selection', 'Edit/Exit tool / select none', S,
        'Edit/Hide / show selected (view only)', 'Edit/Isolate selected (again: show all)', 'Edit/Show all',
        'View/Show all regions, boxes and objects']),
    ('Add', [
        'Tools/The toolset/Shape library (Add)', S,
        'Create/Sphere', 'Create/Cube', 'Create/Cylinder', 'Create/Capsule', 'Create/Cone', 'Create/Torus',
        'Create/Ellipsoid', 'Create/Slab', S,
        ('menu', 'Create/Mass (add to the selected object)', 'Mass (grows out of the selected object)'),
        ('menu', 'Create/Cutter (cut with a shape)', 'Cutter (cut with a shape)'), S,
        'Create/Garment... (dress the selected body from its own measurements)',
        'Create/Dress form from the selected model...',
        ('menu', 'Library/Mouth', 'Mouth (teeth, gums, tongue, jaw)')]),
    ('Sculpt', [
        ('sub', 'Build up (Swell, Draw, Clay, Inflate, Pencil)', [
            'Tools/The toolset/Swell / Settle', 'Sculpt/Draw', 'Sculpt/Clay', 'Sculpt/Inflate', S,
            'Tools/Pencil ridge', 'Tools/Pencil valley', 'Tools/Edit last pencil...']),
        ('sub', 'Carve and crease', ['Sculpt/Carve', 'Sculpt/Crease', 'Sculpt/Pinch']),
        ('sub', 'Move the surface (Grab, Shape, Snake Hook)', ['Sculpt/Move / Grab', 'Sculpt/Stretch',
                                                              'Sculpt/Snake Hook']),
        ('sub', 'Smooth and flatten', [
            'Sculpt/Smooth', 'Tools/The toolset/Flatten', S, 'Tools/Smooth detail (keep shape)',
            'Tools/Smooth shape (keep detail)']),
        ('sub', "Today's other brushes", ['Sculpt/Swell to contact', 'Sculpt/Settle onto', 'Sculpt/Flatten']),
        S,
        'Sculpt/Brush radius smaller', 'Sculpt/Brush radius larger', 'Sculpt/Brush strength lower',
        'Sculpt/Brush strength higher', 'Sculpt/Brush softer (softness / hardness)', 'Sculpt/Brush harder (softness / hardness)', S,
        'Sculpt/Mirror X on / off', 'Sculpt/Mirror Y on / off', 'Sculpt/Mirror Z on / off',
        'Sculpt/Mirror in the world frame (x = 0 midline) on / off', S,
        ('sub', 'Mask (protect while sculpting)', ['Sculpt/Mask', 'Sculpt/Clear mask (selected)',
                                                  'Sculpt/Invert mask (selected)'])]),
    ('Object', [
        'Sculpt/Select / orbit', 'Sculpt/Move object', 'Sculpt/Rotate object', 'Sculpt/Scale object', S,
        'Tools/Merge selected (sculpt kept; last settings)', 'Tools/Merge selected...',
        'Tools/Group the selected (they select and move together)', 'Tools/Ungroup the selected',
        ('menu', 'Tools/Combine selected', 'Combine (merge, subtract, intersect)'),
        ('menu', 'Edit/Flip and mirror', 'Flip and mirror'),
        ('menu', 'Tools/Mutate selected node', 'Change its shape (twist, bend, taper, shell, round)'), S,
        ('sub', 'Tools on the form (place, cut, splice, connect, copies)', [
            'Tools/The toolset/Place on surface', S,
            'Tools/The toolset/Cut (a line, or a shape)', 'Tools/Cut (remove)', 'Tools/Cut and separate',
            'Tools/Splice', 'Tools/Fill and splice', S,
            'Tools/The toolset/Connect', 'Tools/The toolset/Mirror (the copies tool)', S, 'Tools/Exit tool']),
        ('sub', 'Measure (size, girth, tape, weight, units)', [
            'View/Dimensions of the selected object (W x H x D)',
            'View/Dimensions: world extents (the space it takes up in the scene)', S,
            'Tools/Measure/Measure girth (click a form; Shift: vertical, Ctrl: facing the view)',
            'Tools/Measure/Measure with a tape (two clicks: straight and along the surface)',
            'Tools/Measure/Volume and weight of the selected object...',
            'Tools/Measure/Girth profile of the selected object (along an axis; landmarks)...',
            'Tools/Measure/Clear all measures', S,
            ('menu', 'View/Units', 'Units')]),
        ('sub', 'Check the form (topology, hidden voids, section)', [
            'Tools/Check topology of the selected object (parts, tunnels, pockets)',
            'View/Hidden voids (sealed pockets inside the selected object)',
            'Tools/Section tool (cut the view with a plane)']),
        ('sub', 'Protected parts (her locked parts)', [
            'Tools/How to mark her protected parts', S,
            'Tools/Protect the masked part as...', 'Tools/Unprotect a part to sculpt it...',
            'Tools/Protect a part again (relock)...', "Tools/Set a part's root from its mask paint...",
            'Tools/Merge with protected parts...', S,
            'Tools/Check overlaps with protected parts (new overlaps only)',
            'View/Show new overlaps in red (new overlaps only)',
            ('menu', 'View/Protected parts', 'Show them (shown, ghosted, hidden)')]),
        'Tools/Mark a point or circle an area (for the sessions)']),
    ('View', [
        'View/Clay (sculpting: matcap and cavity)', 'View/Look (the real material and light, refining as you work)',
        'View/Silhouette (flat, to check against the references)',
        'View/X-ray on / off (see through to the layers inside; Clay and Look)', S,
        'View/Frame selection', 'View/Frame all',
        ('sub', 'Camera (front, side, top ...)', [
            'View/Front view', 'View/Right view', 'View/Left view', 'View/Back view', 'View/Top view',
            'View/Bottom view', 'View/Three-quarter view', S,
            'View/Orbit around the point under the cursor', 'View/Zoom toward the cursor']),
        ('menu', 'View/Grid', 'Grid'),
        ('sub', 'Light and matcap', [
            'View/Brighter view (cycle +0 / +1 / +2 stops)',
            'View/Sculpting key light (follows the view; off = the HDRI sun)',
            'View/Sculpt headlight (lights recesses from the view; no shadows)',
            'View/Sculpt view light (headlight, cavity, exposure)...', S,
            ('menu', 'View/Matcap and cavity', 'Matcap and cavity')]),
        'View/Section (cut the view with a plane; thickness, gap, mm ruler)', S,
        'View/Action menu at the mouse (also right-click)']),
    ('Panels', [
        ('new', 'panels'), S,
        ('new', 'keys_map'), ('new', 'quick_keys'), ('new', 'timeline'), S,
        ('new', 'animation')]),
    ('Help', [
        'Help/Quick start', 'Help/What to try', 'Help/Controls (short list)', S,
        ('new', 'keys_map'), ('new', 'quick_keys'), 'Help/Search tools and actions (Space)', S,
        'Help/Failures this session*',
        ('sub', 'For the sessions (checks and engine views)', [
            'Tools/Verify op-log replay (hash)', 'Tools/Consolidate history (keep the last 50 undo steps)', S,
            ('menu', 'View/Debug views', 'Engine views (normals, field slice, curvature)')])]),
]
TOP = [t for t, _e in LAYOUT]
MNEMONIC = {'Panels': 'Pa&nels', 'Object': 'O&bject', 'Help': 'Hel&p'}


def _leaves(menu):
    out = []
    for a in menu.actions():
        if a.menu() is not None:
            out += _leaves(a.menu())
        elif not a.isSeparator():
            out.append(a)
    return out


def _detach(a, keep=None):
    """take an action out of every menu it sits in (it stays on the window, so its key still works)"""
    for w in list(a.associatedObjects()):
        if isinstance(w, QMenu) and w is not keep:
            w.removeAction(a)


def _new_actions(win):
    """the actions this module adds: the panels, the keyboard map, the quick keys, the timeline, Animation (not built)"""
    out = {}
    pm = getattr(win, 'panels', None)
    acts = []
    for n in (list(pm.docks) if pm is not None else []):
        if n.startswith('Animation'):                  # not built: named once, below, as not built yet
            continue
        a = QAction(n.replace('&', '&&'), win)
        a.setProperty('fs_path', 'Panels')
        a.triggered.connect(lambda _=False, nn=n: win.panels.open(nn))
        acts.append(a)
    out['panels'] = acts
    a = QAction('Keyboard map (every key; F1)', win)
    a.setProperty('fs_path', 'Panels')
    a.triggered.connect(lambda: _keys_map(win))
    out['keys_map'] = [a]
    win.act_keys_map = a
    a = QAction('Quick keys on the view (show / hide)', win)
    a.setCheckable(True)
    a.setProperty('fs_path', 'Panels')
    qk = getattr(win, 'quick_keys', None)
    a.setChecked(bool(qk is not None and qk.shown))
    a.toggled.connect(lambda on: getattr(win, 'quick_keys', None) is not None and win.quick_keys.set_shown(on))
    out['quick_keys'] = [a]
    win.act_quick_keys = a
    tl = next((d for d in win.findChildren(__import__('PySide6.QtWidgets', fromlist=['QDockWidget']).QDockWidget)
               if d.windowTitle() == 'Timeline'), None)
    if tl is not None:
        a = tl.toggleViewAction()
        a.setText('Timeline (the steps, to step back to one)')
        a.setProperty('fs_path', 'Panels')
        out['timeline'] = [a]
    a = QAction('Animation and rigging: not built yet', win)
    a.setEnabled(False)
    a.setToolTip('There is no rig or keyframe system yet. Today: Transform (W / R / Y) moves, turns and scales a form '
                 'or a masked part of it, one undo step at a time.')
    a.setProperty('fs_path', 'Panels')
    out['animation'] = [a]
    return out


def _keys_map(win):
    kw = getattr(win, 'keys_window', None)
    if kw is not None:
        kw.toggle()


def build(win):
    """move every action into LAYOUT's menus; -> {'placed': n, 'more': [ids], 'hidden': [ids]}"""
    from .shell import menu_actions
    mb = win.menuBar()
    # 1. stamp every action's id (its path as made) before anything moves
    leaves = {}
    for path, a in menu_actions(win):
        if a.property('fs_path') is None:
            a.setProperty('fs_path', path)
        aid = '%s/%s' % (a.property('fs_path'), _clean(a.text()))
        leaves.setdefault(norm(aid), a)
    subs = {}

    def walk(menu, path):
        for a in menu.actions():
            if a.menu() is not None:
                p = path + [_clean(a.menu().title())]
                root = a.menu().property('fs_menu_root')
                if root:
                    p = [str(root)]
                subs.setdefault(norm('/'.join(p)), (a.menu(), menu))
                walk(a.menu(), p)
    old_tops = [a for a in mb.actions() if a.menu() is not None]
    for a in old_tops:
        walk(a.menu(), [_clean(a.menu().title())])
    new = _new_actions(win)
    placed = set()
    report = {'more': [], 'hidden': [], 'missing': []}

    def place(target, entries, where):
        last_sep = True
        for e in entries:
            if e == S:
                if not last_sep:
                    target.addSeparator()
                    last_sep = True
                continue
            if isinstance(e, tuple) and e[0] == 'sub':
                m = QMenu(e[1], target)
                place(m, e[2], where + [e[1]])
                if m.actions():
                    target.addMenu(m)
                    last_sep = False
                continue
            if isinstance(e, tuple) and e[0] == 'menu':
                hit = subs.get(norm(e[1]))
                if hit is None:
                    report['missing'].append(e[1])
                    continue
                sub, parent = hit
                if parent is not None:
                    parent.removeAction(sub.menuAction())
                sub.setTitle(e[2])
                target.addMenu(sub)
                for a in _leaves(sub):
                    placed.add(id(a))
                    a.setProperty('fs_where', ' > '.join(where + [e[2]]))
                last_sep = False
                continue
            if isinstance(e, tuple) and e[0] == 'new':
                for a in new.get(e[1], []):
                    target.addAction(a)
                    placed.add(id(a))
                    if a.property('fs_where') is None:
                        a.setProperty('fs_where', ' > '.join(where))
                    last_sep = False
                continue
            key = norm(e)
            hits = ([a for k, a in leaves.items() if k.startswith(key[:-1])] if key.endswith('*')
                    else [leaves[key]] if key in leaves else [])
            if not hits:
                report['missing'].append(e)
            for a in hits:
                if id(a) in placed:
                    continue
                _detach(a)
                target.addAction(a)
                placed.add(id(a))
                a.setProperty('fs_where', ' > '.join(where))
                last_sep = False
    menus = {}
    for title, entries in LAYOUT:
        # the Alt letters: none may be a key already (Alt+H shows all regions, Alt+O is orthographic): Object Alt+B,
        # Help Alt+P (ui2/check_topbar T8)
        m = QMenu(MNEMONIC.get(title, '&' + title), mb)
        m.setProperty('fs_top', title)
        place(m, entries, [title])
        menus[title] = m
    # 2. what no entry placed: the hidden routes in a hidden holder (their keys and ids live on), the rest in 'More'
    hold = QMenu('(old routes)', menus['Help'])
    hide = set(norm(h) for h in HIDE if not h.endswith('*'))
    hide_pre = [norm(h[:-1]) for h in HIDE if h.endswith('*')]
    more = {}
    for k, a in leaves.items():
        if id(a) in placed:
            continue
        root = str(a.property('fs_path') or '').split('/')[0]
        if k in hide or any(k.startswith(h_) for h_ in hide_pre):
            _detach(a)
            hold.addAction(a)
            report['hidden'].append(k)
            continue
        home = HOME.get(root, 'Object')
        if home not in more:
            more[home] = QMenu('More', menus[home])
        _detach(a)
        more[home].addAction(a)
        a.setProperty('fs_where', '%s > More' % home)
        report['more'].append(k)
    for home, m in more.items():
        menus[home].addSeparator()
        menus[home].addMenu(m)
    menus['Help'].addMenu(hold).setVisible(False)
    # 3. the bar: the new menus in place of the old
    for a in old_tops:
        mb.removeAction(a)
    for title, _e in LAYOUT:
        mb.addMenu(menus[title])
    # 4. plainer words (the id keeps the old name)
    for aid, words in RENAME.items():
        a = leaves.get(norm(aid))
        if a is not None and a.property('fs_text') is None:
            a.setProperty('fs_text', _clean(a.text()))
            a.setText(words)
    win.top_menus = menus
    report['placed'] = len(placed)
    win.menu_report = report
    return report
