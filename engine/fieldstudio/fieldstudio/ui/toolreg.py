"""The window's reading of fs-tools' tool registry (fs-shell-2, 2026-11-14; the window / tools
contract; tools/registry.py is fs-tools' and data only).

- ROWS: the approved interface's sixteen rows (design 3.2). The registry's one 'copies' tool shows as Duplicate,
  Mirror and Array, each presetting its mode.
- Each row's name and status come from the registry (a 'planned' tool with no way to run today is greyed 'coming'); its
  key from the window's keymap (rebindable), never from the registry's key hint.
- How a row runs:
  - through fs-tools' own jobs, once the window has wired that tool's runtime (OWN_RUNTIME) and its jobs are
    registered on the engine (toolset.jobs.register, at the window's start): the row selects the tool, the strip
    shows its own settings and its action button, the action posts the job with the selection;
  - otherwise today's way (categories.TOOL_MAP's runner), with the registry's wording, until it is wired.
FS_LAYOUT=classic keeps today's column and strip."""
import math

ROWS = [('add', 'Shapes', 'add', None), ('add', 'Duplicate', 'copies', 'duplicate'), ('add', 'Mirror', 'copies', 'mirror'),
        ('add', 'Array', 'copies', 'array'),
        ('manipulate', 'Transform', 'transform', None), ('manipulate', 'Shape', 'shape', None),
        ('manipulate', 'Place on surface', 'place', None), ('manipulate', 'Mask / Protect', 'mask', None),
        ('build', 'Pencil', 'pencil', None), ('build', 'Swell / Settle', 'swell', None), ('build', 'Cut', 'cut', None),
        ('build', 'Merge (Join)', 'merge', None), ('build', 'Connect', 'connect', None),
        ('corrective', 'Smooth', 'smooth', None), ('corrective', 'Flatten', 'flatten', None),
        ('corrective', 'Blend', 'blend', None)]
OWN_RUNTIME = {'merge', 'copies', 'place', 'add', 'cut', 'connect'}  # the window runs their own jobs (2b: fs-shell-3)
# fs-shell-3 11-14: what the strip's line says for a tool that acts at the pointer (ui/reg_mouse.py)
HINT = {'place': 'Press on an object and drag it over another: it sits on the surface under the cursor, turned to it; '
                 'release drops it. Wheel: spin. Pick up: the selected one follows the cursor, a click drops it.',
        'add': 'Click on a form: the shape lands on the surface there, sitting on it (nothing under the cursor: at the '
               'view\'s centre). Hollow: a shell, its wall measured inward; the outside stays.',
        'cut': 'Line: drag a line across the form (or click its two ends); the blade shows, Length deep from the first '
               'surface. Shape: click on the form; its volume goes to the container (Cut away deletes it).',
        'connect': 'Click a surface, then a surface of another object: the connector runs between them in your '
                   'profile, three objects in contact. Esc: start again.'}


def _extra():
    """fs-shell-3 11-14: settings the window needs that the registry does not list yet;
    a registry Setting with the same key replaces each"""
    R = registry()
    if R is None:
        return {}
    from .reg_mouse import SHAPES
    return {'add': (R.Setting('kind', 'Shape', 'choice', choices=SHAPES, default='sphere',
                              says='The shape to add (heads, masses and cutters: More shapes)'),),
            'cut': (R.Setting('shape_size', 'Shape size', 'number', range=(0.001, 1.0), unit='m', default=0.02,
                              says='Cut with a shape: the shape\'s largest extent'),),
            'connect': (R.Setting('size', 'Profile size', 'number', range=(0.0005, 0.5), unit='m', default=0.005,
                                  says='The profile\'s radius (circle) or half side (square)'),)}
ROW_SAYS = {'duplicate': 'One copy of the selection, its sculpting and all',
            'mirror': 'A mirrored copy of the selection across the middle',
            'array': 'Copies in a line, each a little different with Vary'}
COPY_JOB = {'duplicate': 'tools.duplicate', 'mirror': 'tools.mirror', 'array': 'tools.array'}
# the settings each copies row shows (its mode is the row)
COPY_OWN = {'duplicate': (), 'mirror': (), 'array': ('count', 'vary')}


def registry():
    try:
        from ..tools import registry as REG
        return REG
    except Exception:
        return None


def spec(key):
    R = registry()
    return R.BY_KEY.get(key) if R is not None else None


def handlers(win):
    return getattr(getattr(win, 'engine', None), '_job_handlers', None) or {}


# ---- fs-shell-3 11-14: the registry's held brushes on the engine's timed brushes (core/brush.FT_TOOLS, ee9e0f9b) --------
FT_DEFAULT_RATE = {'ft_swell': 0.004, 'ft_settle': 0.004, 'ft_flatten': 0.5}   # 4 mm/s; 50 % a second (first use)


def strength_unit(tool):
    """what a brush's Strength means: 'm/s' (Swell / Settle), '1/s' (Flatten), '' (today's brushes)"""
    return {'ft_swell': 'm/s', 'ft_settle': 'm/s', 'ft_flatten': '1/s'}.get(str(tool or ''), '')
STRENGTH_SHOW = {'ft_swell': (1000.0, ' mm/s', 0.0, 50.0, 1), 'ft_settle': (1000.0, ' mm/s', 0.0, 50.0, 1),
                 'ft_flatten': (100.0, ' %/s', 0.0, 200.0, 0)}                  # factor, suffix, lo, hi, decimals
FT_NAMES = {'ft_swell': 'Swell', 'ft_settle': 'Settle', 'ft_flatten': 'Flatten'}


def strength_text(tool, value):
    """fs-shell-4 11-14: a timed brush's Strength as the strip shows it ('4.0 mm/s', '50 %/s'); None for today's brushes
    (the viewport's tool badge and its F-drag read-out showed the raw rate, 'strength 0.00' for 4 mm/s)"""
    sh = STRENGTH_SHOW.get(str(tool or ''))
    if sh is None:
        return None
    fac, suf, _lo, _hi, dec = sh
    return '%.*f%s' % (dec, float(value) * fac, suf)


def brush_live(key):
    """-> the registry brush tool's engine names when the window runs it on the engine's timed brushes now: the
    registry no longer marks it 'planned' (FS_FT_ROWS=1: a harness before that) and the engine times every name"""
    sp = spec(key)
    names = tuple(getattr(sp, 'brush', None) or ()) if sp is not None else ()
    if not names or sp.run != 'brush':
        return ()
    import os
    if sp.status == 'planned' and os.environ.get('FS_FT_ROWS', '') != '1':
        return ()
    try:
        from ..core import brush as _BRft
        ft = set(getattr(_BRft, 'FT_TOOLS', ()))
    except Exception:
        return ()
    return names if all(n in ft for n in names) else ()


def brush_name(win, key):
    """the engine brush for the row now: Swell / Settle by its own 'side' setting, else the registry's first"""
    names = brush_live(key)
    if not names:
        return None
    if key == 'swell' and len(names) > 1:
        sp = spec(key)
        side = (getattr(win, 'reg_values', {}) or {}).get((key, 'side'),
                                                         next((s.default for s in sp.own if s.key == 'side'), 'swell'))
        return names[1] if side == 'settle' else names[0]
    return names[0]


def apply_brush(win, key):
    """the brush in hand = the row's engine brush with the row's own settings (Follow / Flat ...), a sane first rate"""
    name = brush_name(win, key)
    if name is None:
        return None
    tp = win.tools
    tp.set_tool(name)                                   # (this clears reg_tool through toolChanged: set it after)
    sp = spec(key)
    vals = getattr(win, 'reg_values', {}) or {}
    for s in sp.own:
        if s.key in ('follow',):
            # fs-shell-4 11-14: through _set, which emits: the viewport's brush is a COPY taken at each emit
            # (ToolPanel.brush), and set_tool above emitted before this line, so a direct write here never reached the
            # press (t62: the strip's Flat went out as 'follow' until another setting changed)
            v_ = vals.get((key, s.key), s.default)
            if tp.settings.get(s.key) != v_:
                tp._set(s.key, v_)
    # (the first-use rate: the panel's own per-tool memory, panels.ToolPanel._recall with FT_DEFAULT_RATE)
    for k_, v_ in (('hardness', 0.0), ('smoothing', 0.0), ('foot', 'soft')):
        if k_ not in tp.settings:
            tp._set(k_, v_)
    return name


def runnable(win, key, mode=None):
    """True when the window runs this registry tool through its own jobs now"""
    if brush_live(key):
        return True                                     # fs-shell-3 11-14: a held brush on the engine's timed path
    sp = spec(key)
    if sp is None or key not in OWN_RUNTIME or sp.status == 'planned':
        return False
    need = [COPY_JOB[mode]] if key == 'copies' and mode else [j for j in sp.jobs if j in ('tools.merge',)] or list(sp.jobs)
    h = handlers(win)
    return all(j in h for j in need)


def rows(win, today):
    """-> the column's rows in categories.TOOL_MAP's shape (category, name, run, light keys, sentence, picture key):
    run 'reg:<key>[:<mode>]' for a tool the window runs through its own jobs, else today's runner (None: coming)"""
    old = {r[1]: r for r in today}
    out = []
    for cat, name, key, mode in ROWS:
        sp = spec(key)
        o = old.get(name)
        run, light, says, pk = (o[2], o[3], o[4], o[5]) if o is not None else (None, (), '', 'p:' + key)
        if sp is not None:
            says = ROW_SAYS.get(mode) or sp.says or says
            if runnable(win, key, mode):
                run, light = 'reg:%s%s' % (key, ':' + mode if mode else ''), ()
            elif sp.status == 'planned' and run is None:
                says = says + ' (coming from fs-tools)'
        out.append((cat, name, run, light, says, pk))
    return out


# ---- the selected registry tool ----------------------------------------------------------------------------------------
def select(win, run):
    """a 'reg:' row: the viewport's tool goes to Select (clicks select objects), the strip shows this tool"""
    parts = run.split(':')
    key, mode = parts[1], (parts[2] if len(parts) > 2 else None)
    if brush_live(key):
        apply_brush(win, key)             # fs-shell-3 11-14: the engine's timed brush is the tool in hand
    else:
        win.tools.set_tool('select')      # leaves a field tool too; the selection stays (the tool acts on it)
    win.reg_tool = (key, mode)
    win.reg_values = dict(getattr(win, 'reg_values', {}) or {})
    win.engine.tool_result = None


def clear(win):
    win.reg_tool = None


def current(win):
    return getattr(win, 'reg_tool', None)


def card(win):
    """-> (name, kind, says) of the selected registry tool"""
    key, mode = current(win)
    sp = spec(key)
    name = {'duplicate': 'Duplicate', 'mirror': 'Mirror', 'array': 'Array'}.get(mode, sp.name if sp else key)
    return name, (sp.kind if sp else ''), ROW_SAYS.get(mode) or (sp.says if sp else '')


def own_settings(win):
    """-> the Setting specs the strip shows for the selected tool (the copies row's mode is the row itself)"""
    key, mode = current(win)
    sp = spec(key)
    if sp is None:
        return ()
    if key == 'copies':
        return tuple(s for s in sp.own if s.key in COPY_OWN.get(mode, ()))
    have = {s.key for s in sp.own}
    ext = tuple(s for s in _extra().get(key, ()) if s.key not in have)
    # the library's own choice (Add's Shape) leads; a size the registry lacks (Cut's shape, Connect's profile) follows
    return ext + tuple(sp.own) if key == 'add' else tuple(sp.own) + ext


def value(win, s):
    key, mode = current(win)
    return (getattr(win, 'reg_values', {}) or {}).get((key, s.key), s.default)


def extras(win):
    """fs-shell-3 11-14: the strip's further buttons for the selected tool: (text, tip, fn)"""
    key, _mode = current(win)
    rm = getattr(win, 'reg_mouse', None)
    if key == 'add' and rm is not None:
        def more():
            cat = getattr(win, 'categories', None)
            return cat._shapes_menu() if cat is not None else None
        return [('Hollow the selected', 'Hollows the selected object inward to the Wall; its outside stays (one undo '
                 'step; again = Re-hollow)', rm.hollow_selected),
                ('More shapes ...', 'New head, new bust, masses (block-in), cutters', more)]
    if key == 'cut' and rm is not None:
        return [('Place the cut piece', 'The piece a shape cut took out: it follows the cursor over the surfaces; a '
                 'click sets it down (Place on surface)', rm.place_piece)]
    return []


def set_value(win, s, v):
    key, _mode = current(win)
    vals = dict(getattr(win, 'reg_values', {}) or {})
    vals[(key, s.key)] = v
    win.reg_values = vals
    if brush_live(key):                       # fs-shell-3 11-14: Swell / Settle, Follow / Flat change the brush in hand
        cur = (key, _mode)
        apply_brush(win, key)
        win.reg_tool = cur


def action_text(win):
    key, mode = current(win)
    if key == 'copies':
        return {'duplicate': 'Duplicate', 'mirror': 'Mirror', 'array': 'Make the copies'}.get(mode, 'Copy')
    if key == 'place':
        return 'Pick up'
    sp = spec(key)
    return (sp.action[0] if sp and sp.action and sp.action[0] else 'Apply')


def selection(win):
    vp = win.viewport
    return [int(o) for o in (getattr(vp, 'selected', None) or [])]


def act(win):
    """the action button: post the selected tool's job with the selection and its settings; -> a line for the status"""
    key, mode = current(win)
    oids = selection(win)
    eng = win.engine
    prev = getattr(eng, 'tool_result', None)   # fs-shell-3 11-14: Cut away reads the piece the last cut holds
    eng.tool_result = None
    if key == 'merge':
        if len(oids) < 2:
            return 'Merge: select two or more objects (Shift+click adds).'
        blend = float(value(win, next(s for s in spec('merge').own if s.key == 'blend')) or 0.0)
        eng.post('tools.merge', oids, blend)
        return 'Merging %d objects%s ...' % (len(oids), ' (blend at contact %.1f mm)' % (blend * 1e3) if blend else '')
    if key == 'copies':
        if len(oids) != 1:
            return '%s: select one object.' % card(win)[0]
        oid = oids[0]
        if mode == 'duplicate':
            # fs-shell-3 11-14 (option A, decided): the row behaves as Ctrl+D: the copy in place follows the
            # mouse (type a distance; a click places it), Duplicate + its move one undo step
            from .flip_ui import duplicate_in_place
            duplicate_in_place(win, oid)
        elif mode == 'mirror':
            eng.post('tools.mirror', oid, 'world_x')
        else:
            sp = spec('copies')
            n = int(value(win, next(s for s in sp.own if s.key == 'count')) or 5)
            seed = int(value(win, next(s for s in sp.own if s.key == 'vary')) or 0)
            kw = {'mode': 'line', 'count': n, 'step': [step_x(win, oid), 0.0, 0.0]}
            if seed:
                kw['vary'] = {'seed': seed, 'size': 0.15, 'turn': 15.0}
            eng.post('tools.array', oid, kw)
        return '%s ...' % card(win)[0]
    if key == 'place':
        rm = getattr(win, 'reg_mouse', None)
        return rm.pick_up() if rm is not None else ''
    if key == 'add':
        rm = getattr(win, 'reg_mouse', None)
        return rm.add_at() if rm is not None else ''
    if key == 'cut':
        r = prev or {}                         # the cut's own result (cleared above for the next action)
        pid = r.get('held') if r.get('tool') == 'cut' else None
        if pid is None:
            return 'Cut: nothing is held. Drag a cut line across the form, or click it with a shape.'
        eng.post('tools.container_delete', int(pid))
        return 'Cut away: the held piece is deleted (one undo step).'
    if key == 'connect':
        return 'Connect: click a surface in the view, then a surface of another object.'
    return ''


def hint(win):
    """the strip's line for the selected registry tool"""
    key, _mode = current(win)
    if brush_live(key):
        return {'swell': 'Press and hold on the form: it swells (or settles) while held, at the rate set; drag to move '
                         'it. Follow: each point along its own facing; Flat: one direction, the press facing.',
                'flatten': 'Press and hold: the surface comes onto the plane through the cursor, at the rate set; '
                           'drag to move the plane with it.'}.get(key, 'Press and hold on the form.')
    return HINT.get(key) or 'Select the objects in the view, then %s.' % action_text(win).lower()


def is_brush(win):
    cur = current(win)
    return bool(cur) and bool(brush_live(cur[0]))


def step_x(win, oid):
    """an array's spacing: the object's own width along x, plus 15 % (the registry's default 100 mm overlaps a
    larger form)"""
    eng = win.engine
    try:
        with eng.lock:
            o = eng.hist.scene.get(oid)
            lo, hi = o.local_box()
            sc = o.transform.get('scale', 1.0)
        sx = float(sc[0]) if isinstance(sc, (list, tuple)) else float(sc)
        w = float(hi[0] - lo[0]) * abs(sx)
        return round(w * 1.15, 6) if math.isfinite(w) and w > 0 else 0.1
    except Exception:
        return 0.1


def offer(win):
    """-> (text, job, args) when the last action was refused with an offer (Merge: 'Not touching: Group instead?')"""
    r = getattr(win.engine, 'tool_result', None) or {}
    if r.get('tool') == 'merge' and r.get('offer') == 'group':
        gaps = r.get('gaps_mm')
        if isinstance(gaps, (list, tuple)):
            gaps = min(gaps) if gaps else None
        g = (' (gap %.1f mm)' % float(gaps)) if isinstance(gaps, (int, float)) else ''
        return 'Not touching%s: Group instead' % g, 'tools.group', [list(r.get('oids') or [])]
    return None


# ---- the tool's cues (toolset/cues.py: data; ui/cues_draw.py draws them) ------------------------------------------------
def _cues_job(engine, key, mode, oids, vals, sig):
    """on the engine thread (under its lock): the selected tool's cues, measured from the field; no scene change"""
    from ..toolset import cues as CU
    sc = engine.hist.scene
    out = []
    try:
        if key == 'merge' and len(oids) >= 2:
            out = CU.merge(sc, oids, float(vals.get('blend') or 0.0))
        elif key == 'copies' and mode == 'array' and len(oids) == 1:
            kw = {'mode': 'line', 'count': int(vals.get('count') or 5), 'step': list(vals.get('step') or [0.1, 0, 0])}
            out = CU.array_ghosts(sc, oids[0], kw)
    except Exception as e:                    # a cue that cannot be measured is not drawn (and says why once)
        out = []
        engine.tool_cues_error = repr(e)
    engine.tool_cues = (sig, out)
    return False


def install_cues(win):
    try:
        win.engine.register_job('shell.cues', _cues_job)
    except ValueError:
        pass
    win._cue_sig = None


def update_cues(win):
    """called by the strip's sync: post the cue job when the tool, the selection, its settings or the scene changed;
    hand finished cues to the view"""
    vp = win.viewport
    cur = current(win)
    if cur is None:
        if getattr(vp, 'cues', None):
            vp.cues = None
            vp.update()
        win._cue_sig = None
        return
    key, mode = cur
    oids = selection(win)
    sp = spec(key)
    vals = {s.key: value(win, s) for s in (sp.own if sp else ())}
    if key == 'copies' and mode == 'array' and len(oids) == 1:
        vals['step'] = [step_x(win, oids[0]), 0.0, 0.0]
    snap = getattr(win, 'snap', None) or {}
    ver = (len(snap.get('log') or ()), len(snap.get('redo_log') or ()))
    sig = (key, mode, tuple(oids), tuple(sorted((k, str(v)) for k, v in vals.items())), ver)
    if sig != getattr(win, '_cue_sig', None):
        win._cue_sig = sig
        win.engine.post('shell.cues', key, mode, list(oids), vals, sig)
        if getattr(vp, 'cues', None):             # never the last selection's cues over the new one
            vp.cues = None
            vp.update()
    tc = getattr(win.engine, 'tool_cues', None)
    if tc is not None and tc[0] == sig and getattr(vp, 'cues', None) is not tc[1]:
        vp.cues = tc[1]
        vp.update()


def merge_now(win):
    """fs-shell-2 11-14 (ONE Merge): the strip's Merge tool on the current selection, at once (Ctrl+J, the Scene panel,
    the menus): fs-tools' tools.merge with the strip's Blend at contact; a refusal shows 'Group instead' in the strip"""
    select(win, 'reg:merge')
    line = act(win)
    try:
        win.status.showMessage(line, 5000)
    except Exception:
        pass
    return line


def hide_old_merges(win):
    """the old fixed-width smooth union's own menu entry (core/combine.py; the 'possibly' under Ansys US 10,083,264,
    LIBRARIES_AND_PATENTS 6.4) leaves the menus in the approved layout; Merge is fs-tools' join"""
    from .shell import menu_actions
    for _p, a in menu_actions(win):
        if a.text().replace('&', '') in ('Merge smooth (blend k)',):
            a.setVisible(False)
