"""Flip and mirror from the gizmo (fs-shell, 2026-11-11; design: the gizmo drags to rotate,
scale and flip, with a quick button to mirror or copy). Commands: tools/flip.py.

On the context bar while Move / Rotate / Scale is up: a Flip / Mirror button: Flip X / Y / Z (in place, on the object's
own axes, the gizmo's), Mirror copy across the world X / Y / Z plane (X: the body's midline), Mirror copy on its own
X / Y / Z (through its centre). Keys: Ctrl+Alt+X / Y / Z flip; Ctrl+Shift+D stays the mirror copy across X. The same
actions sit in Edit > Flip and mirror. One undo step each; the copy is selected."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMenu, QToolButton


def _selected(win):
    from ..tools.shapecut import is_cutter
    objs = {o['id']: o for o in (getattr(win, 'snap', None) or {}).get('objects', [])}
    return [o for o in win.viewport.selected if o in objs]


def flip(win, axis):
    from ..tools.flip import FlipObject
    sel = _selected(win)
    if not sel:
        win.status.showMessage('Flip: select an object first.', 6000)
        return False
    for oid in sel:
        win.engine.post('push', FlipObject(oid, axis))
    win.status.showMessage('Flipped on its own %s (Ctrl+Z puts it back).' % axis.upper(), 6000)
    return True


def mirror_copy(win, axis, through='world'):
    """fs-shell-3 11-14 (registry stage 2b (g), ONE mirror): across a world plane every entry (this menu, Ctrl+Shift+D,
    the Scene panel's Mirror X, the Mirror row) runs fs-tools' tools.mirror (toolset/copies.MirrorAcross, exact);
    through the object's own centre stays tools/flip.MirrorCopy (fs-tools' object planes pass through its origin)"""
    from ..tools.flip import MirrorCopy
    sel = _selected(win)
    if not sel:
        win.status.showMessage('Mirror copy: select an object first.', 6000)
        return False
    h = (getattr(win.engine, '_job_handlers', None) or {})
    for oid in sel:
        if through == 'world' and 'tools.mirror' in h:
            win.engine.post('tools.mirror', int(oid), 'world_%s' % axis.lower())
        else:
            win.engine.post('push', MirrorCopy(oid, axis, through))
    win.status.showMessage('Mirror copy across %s (the copy is selected).' % (
        'the world %s plane' % axis.upper() if through == 'world' else 'its own %s, through its centre' % axis.upper()),
        8000)
    return True


def copy_name(win, name):
    """'A copy', then 'A copy 2', 'A copy 3' ... (names that do not collide)"""
    names = {o.get('name') for o in (getattr(win, 'snap', None) or {}).get('objects', [])}
    base = '%s copy' % name
    if base not in names:
        return base
    k = 2
    while '%s %d' % (base, k) in names:
        k += 1
    return '%s %d' % (base, k)


def duplicate_in_place(win, oid, move=True):
    """Ctrl+D (as Blender's Shift+D): a copy in place (DuplicateObject with offset 0: the command as it
    was, so older logs replay unchanged), selected, then a typed move that follows the mouse (type a distance, X / Y / Z
    for the axis; Enter or a click places it, Esc leaves it on the original)"""
    from ..core.combine import DuplicateObject
    objs = {o['id']: o for o in (getattr(win, 'snap', None) or {}).get('objects', [])}
    if oid not in objs:
        return None
    nm = copy_name(win, objs[oid]['name'])
    eng = win.engine
    tok = object()

    def find(e):
        sc = e.hist.scene
        for o in reversed(sc.objects):
            if o.name == nm:
                return o.id
        return None

    def got(rr, tk=tok):
        if rr[0] is not tk:
            return
        eng.callDone.disconnect(got)
        cid = rr[1]
        if cid is None or getattr(cid, 'failed', False):      # a failed call answers with a falsy CallFailed (fs-brush cba5ef61); the engine reports the failure itself
            return
        win.duplicate_last = cid

        def go(tries=[0]):
            vp = win.viewport
            known = {o['id'] for o in (getattr(win, 'snap', None) or {}).get('objects', [])}
            if cid not in known and tries[0] < 100:          # the panel takes ids only once its snapshot has them
                tries[0] += 1
                QTimer.singleShot(20, go)
                return
            vp.selected = [cid]
            win.props.set_selection([cid])
            if not move:
                return
            if win.tools.tool != 'move':
                win.tools.set_tool('move')
            g = getattr(vp, 'gizmo', None)
            if g is not None:
                def typed(gz=g):
                    # fs-shell-3 11-14 (option A, decided): the copy follows the mouse as approved; its
                    # placing move and the Duplicate become ONE undo step at the click (gizmo.finish -> shell.group_last)
                    if gz.start_typed() and vp.xform is not None:
                        vp.xform['dup_group'] = cid
                QTimer.singleShot(0, typed)
            win.status.showMessage('Copied in place as "%s": it follows the mouse; type a distance (X / Y / Z for the '
                                   'axis), Enter or a click places it, Esc leaves it on the original.' % nm, 12000)
        QTimer.singleShot(0, go)
    eng.callDone.connect(got)
    # fs-shell-3 11-14: fs-tools' tools.duplicate, once it takes the copy's name (pending);
    # until then the same command it pushes (DuplicateObject), with the non-colliding name
    h = (getattr(eng, '_job_handlers', None) or {}).get('tools.duplicate')
    takes_name = False
    if h is not None:
        try:
            import inspect
            takes_name = 'name' in inspect.signature(h).parameters
        except (TypeError, ValueError):
            takes_name = False
    if takes_name:
        eng.post('tools.duplicate', int(oid), [0.0, 0.0, 0.0], nm)
    else:
        eng.post('push', DuplicateObject(oid, offset=[0.0, 0.0, 0.0], name=nm))
    eng.post('call', find, tok)
    return tok


def _group_last(engine, oid, label='Duplicate'):
    """engine thread: the copy's Duplicate and the move that placed it (the last two steps, both its own) become one
    undo step named label; anything else in between leaves them as they are"""
    from ..core.combine import DuplicateObject
    from ..core.history import SetObjectProps
    done = engine.hist.done
    if len(done) < 2:
        return False
    d, m = done[-2], done[-1]
    if not (isinstance(d, DuplicateObject) and getattr(getattr(d, 'obj', None), 'id', None) == int(oid)
            and isinstance(m, SetObjectProps) and m.oid == int(oid)):
        return False
    if any(getattr(c, 'group_n', 0) or getattr(c, 'group_first_n', 0) for c in (d, m)):
        return False
    d.group_first_n = 2
    m.group_n = 2
    d.ui_label = m.ui_label = label
    return True


def _fill(menu, win):
    menu.clear()
    for ax in 'xyz':
        a = menu.addAction('Flip %s (its own axis)	Ctrl+Alt+%s' % (ax.upper(), ax.upper()))   # the key shown, not bound twice
        a.triggered.connect(lambda _=False, ax=ax: flip(win, ax))
    menu.addSeparator()
    for ax, note in (('x', ' (the midline)'), ('y', ''), ('z', '')):
        a = menu.addAction('Mirror copy across the world %s plane%s%s' % (ax.upper(), note,
                                                                           '	Ctrl+Shift+D' if ax == 'x' else ''))
        a.triggered.connect(lambda _=False, ax=ax: mirror_copy(win, ax, 'world'))
    menu.addSeparator()
    for ax in 'xyz':
        a = menu.addAction('Mirror copy on its own %s (through its centre)' % ax.upper())
        a.triggered.connect(lambda _=False, ax=ax: mirror_copy(win, ax, 'centre'))


def install(win):
    try:                                   # fs-shell-3 11-14: Duplicate + its placing move = one undo step
        win.engine.register_job('shell.group_last', _group_last)
    except ValueError:
        pass
    cb = getattr(win, 'context_bar', None)
    b = None
    if cb is not None:
        b = QToolButton()
        b.setText('Flip / Mirror')
        b.setFocusPolicy(Qt.NoFocus)
        b.setPopupMode(QToolButton.InstantPopup)
        b.setToolTip('Flip in place on X / Y / Z (Ctrl+Alt+X / Y / Z), or a mirrored copy across any axis, through '
                     'the world origin or the object\'s centre')
        m = QMenu(b)
        m.aboutToShow.connect(lambda: _fill(m, win))
        b.setMenu(m)
        lay = cb.layout()
        i = lay.indexOf(cb.units) if hasattr(cb, 'units') else -1
        lay.insertWidget(i + 1 if i >= 0 else lay.count(), b)
        b.hide()
        win.flip_button = b

        def sync(*_):
            tool = getattr(win.tools, 'tool', None)
            b.setVisible(tool in ('move', 'rotate', 'scale') and bool(_selected(win)))
        win.tools.toolChanged.connect(lambda _k: QTimer.singleShot(0, sync))
        win.props.selectionChanged.connect(lambda _l: QTimer.singleShot(0, sync))
        win.engine.sceneChanged.connect(lambda _s: sync())
    # the window's actions (keys and the Edit menu)
    em = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'Edit' in a.menu().title()), None)
    if em is not None:
        sub = em.addMenu('Flip and mirror')
        for ax in 'xyz':
            win._act(sub, 'Flip %s (its own axis)' % ax.upper(), lambda _=False, ax=ax: flip(win, ax),
                     'Ctrl+Alt+%s' % ax.upper())
        sub.addSeparator()
        for ax in 'xyz':
            win._act(sub, 'Mirror copy across the world %s plane' % ax.upper(),
                     lambda _=False, ax=ax: mirror_copy(win, ax, 'world'))
        for ax in 'xyz':
            win._act(sub, 'Mirror copy on its own %s (through its centre)' % ax.upper(),
                     lambda _=False, ax=ax: mirror_copy(win, ax, 'centre'))
    return b
