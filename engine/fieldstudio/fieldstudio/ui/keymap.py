"""Rebinding keys (fs-shell, 2026-11-11; design 4.2 "Rebinding": right-click a key in the Keys window and choose
"Rebind...". Conflicts are shown; defaults are restorable. Blender's keymap editor and ZBrush's Ctrl+Alt-click are the
peers).

Every key in Field Studio is a menu action's shortcut, so a binding is (action id -> key sequences), the id being the
action's menu path ("Sculpt/Smooth", "View/Front view"). install() records each action's default keys once, then
applies the overrides saved in the app settings ('keymap': {id: [portable key text, ...]}; only changed actions are
stored, an empty list means unbound). A few keys are not actions and stay fixed (RESERVED): F (radius / strength
drag), Space (the palette), ? (peek), and the mouse."""
import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QKeySequenceEdit, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout)

PT = QKeySequence.PortableText
NT = QKeySequence.NativeText
RESERVED = {'F': 'hold F over the view: brush radius (Shift+F: strength)', 'Shift+F': 'brush strength drag',
            'Ctrl+F': 'brush hardness drag (fs-shell-5 11-14)',
            'Space': 'the search palette', '?': 'peek at the keys (hold)', 'Shift+?': 'peek at the keys (hold)'}


def _clean(text):
    return text.replace('&', '').split('\t')[0].strip()


def norm(seq):
    """-> portable text of a key sequence or text ('' for none)"""
    ks = seq if isinstance(seq, QKeySequence) else QKeySequence(str(seq))
    return ks.toString(PT)


def native(text):
    return QKeySequence(text).toString(NT) if text else ''


def actions(win):
    """-> {id: QAction} for every leaf menu action (first path wins when an action sits in two menus)"""
    from .shell import menu_actions, action_id
    out = {}
    for path, a in menu_actions(win):
        aid = action_id(path, a)                   # fs-shell-5 11-14: the name it was made with (ui/menus.py renames)
        out.setdefault(aid, a)
    return out


def current(a):
    return [norm(s) for s in a.shortcuts() if not s.isEmpty()]


def default(a):
    d = a.property('fs_default_keys')
    return json.loads(d) if d else current(a)


def _settings():
    from .main_window import app_settings
    return app_settings()


def overrides():
    try:
        return json.loads(_settings().value('keymap', '') or '{}') or {}
    except Exception:
        return {}


def _save(win):
    try:                                           # fs-shell-5 11-14: the strip's and quick keys' key labels follow
        from . import toolstrip as _TS
        _TS._KEYS_CACHE.clear()
    except Exception:
        pass
    out = {}
    for aid, a in actions(win).items():
        if current(a) != default(a):
            out[aid] = current(a)
    _settings().setValue('keymap', json.dumps(out))


def install(win):
    """record the defaults, apply the saved keys"""
    acts = actions(win)
    for a in acts.values():
        if a.property('fs_default_keys') is None:
            a.setProperty('fs_default_keys', json.dumps(current(a)))
    for aid, keys in overrides().items():
        a = acts.get(aid)
        if a is not None:
            a.setShortcuts([QKeySequence(k) for k in keys])
    win.keymap_acts = acts


def holders(win, seq, exclude=None):
    """-> [(id, QAction)] that hold this key now (the action being rebound left out)"""
    k = norm(seq)
    if not k:
        return []
    out, seen = [], set()
    for aid, a in actions(win).items():
        if a is exclude or id(a) in seen:
            continue
        if k in current(a):
            out.append((aid, a))
            seen.add(id(a))
    return out


def reserved(seq):
    return RESERVED.get(native(norm(seq))) or RESERVED.get(norm(seq))


def assign(win, aid, keys, add=False):
    """give the action these keys (add=True keeps its others); any other action holding one loses it"""
    acts = actions(win)
    a = acts[aid]
    keys = [norm(k) for k in keys if norm(k)]
    for k in keys:
        for _oid, o in holders(win, k, exclude=a):
            o.setShortcuts([QKeySequence(x) for x in current(o) if x != k])
    new = (current(a) + [k for k in keys if k not in current(a)]) if add else keys
    a.setShortcuts([QKeySequence(k) for k in new])
    _save(win)
    changed(win)


def remove_key(win, aid, key):
    a = actions(win)[aid]
    a.setShortcuts([QKeySequence(k) for k in current(a) if k != norm(key)])
    _save(win)
    changed(win)


def restore(win, aid):
    a = actions(win)[aid]
    keys = default(a)
    for k in keys:                               # a default taken back from whoever holds it now
        for _oid, o in holders(win, k, exclude=a):
            o.setShortcuts([QKeySequence(x) for x in current(o) if x != k])
    a.setShortcuts([QKeySequence(k) for k in keys])
    _save(win)
    changed(win)


def restore_all(win):
    for a in actions(win).values():
        a.setShortcuts([QKeySequence(k) for k in default(a)])
    _settings().setValue('keymap', '{}')
    changed(win)


def tool_key(win, key, fallback=''):
    """-> the native key text a tool is bound to now (the Sculpt menu's action), for labels"""
    a = (getattr(win, 'tool_actions', None) or {}).get(key)
    if a is None:
        return fallback
    return a.shortcut().toString(NT) if not a.shortcut().isEmpty() else ''


def tool_of_action(win, a):
    for k, ta in (getattr(win, 'tool_actions', None) or {}).items():
        if ta is a:
            return k
    return None


def changed(win):
    """every label that shows a key follows a rebinding"""
    for fn in ('relabel_tools',):
        f = getattr(win.tools, fn, None)
        if f is not None:
            f({k: tool_key(win, k) for k in (getattr(win, 'tool_actions', None) or {})})
    for name in ('shelf', 'fav_strip', 'context_bar', 'keys_window'):
        w = getattr(win, name, None)
        f = getattr(w, 'keys_changed', None) if w is not None else None
        if f is not None:
            f()
    try:
        win.update_hint()
    except Exception:
        pass


class RebindDialog(QDialog):
    """press the new key; who holds it now is shown before anything changes"""

    def __init__(self, win, aid, parent=None, add=False):
        super().__init__(parent or win)
        self.win, self.aid, self.add = win, aid, add
        a = actions(win)[aid]
        self.a = a
        self.setWindowTitle('Rebind: %s' % _clean(a.text()))
        lay = QVBoxLayout(self)
        cur = ', '.join(native(k) for k in current(a)) or 'none'
        dflt = ', '.join(native(k) for k in default(a)) or 'none'
        info = QLabel('<b>%s</b> (%s)<br>Now: %s &nbsp; Default: %s<br>%s' % (
            _clean(a.text()), aid.rsplit('/', 1)[0].replace('/', ' > '), cur, dflt,
            'Press the key to ADD (the current keys stay).' if add else 'Press the new key (it replaces the current one).'))
        info.setTextFormat(Qt.RichText)
        lay.addWidget(info)
        self.edit = QKeySequenceEdit()
        self.edit.setMaximumSequenceLength(1)
        self.edit.setClearButtonEnabled(True)
        self.edit.keySequenceChanged.connect(self._check)
        lay.addWidget(self.edit)
        self.conflict = QLabel('')
        self.conflict.setWordWrap(True)
        lay.addWidget(self.conflict)
        row = QHBoxLayout()
        self.ok = QPushButton('Assign')
        self.ok.setEnabled(False)
        self.ok.clicked.connect(self._assign)
        row.addWidget(self.ok)
        b = QPushButton('Unbind')
        b.setToolTip('Remove every key from this action')
        b.clicked.connect(self._unbind)
        row.addWidget(b)
        b = QPushButton('Restore default')
        b.clicked.connect(self._restore)
        row.addWidget(b)
        b = QPushButton('Cancel')
        b.clicked.connect(self.reject)
        row.addWidget(b)
        lay.addLayout(row)
        self.edit.setFocus()
        self.result_keys = None

    def _check(self, ks):
        k = norm(ks)
        self.key = k
        if not k:
            self.ok.setEnabled(False)
            self.conflict.setText('')
            return
        r = reserved(k)
        if r:
            self.ok.setEnabled(False)
            self.conflict.setText('<span style="color:#f87171">%s is fixed: %s.</span>' % (native(k), r))
            return
        hs = holders(self.win, k, exclude=self.a)
        self.ok.setEnabled(True)
        if hs:
            self.conflict.setText('<span style="color:#fbbf24">%s is used by %s. Assign takes it from %s.</span>' % (
                native(k), ', '.join('<b>%s</b> (%s)' % (_clean(o.text()), oid.split('/')[0]) for oid, o in hs),
                'it' if len(hs) == 1 else 'them'))
        else:
            self.conflict.setText('<span style="color:#86efac">%s is free.</span>' % native(k))

    def _assign(self):
        assign(self.win, self.aid, [self.key], add=self.add)
        self.result_keys = current(self.a)
        self.accept()

    def _unbind(self):
        assign(self.win, self.aid, [])
        self.result_keys = []
        self.accept()

    def _restore(self):
        restore(self.win, self.aid)
        self.result_keys = current(self.a)
        self.accept()


class ActionPicker(QDialog):
    """a free key on the keyboard map: pick the action it should run"""

    def __init__(self, win, key, parent=None):
        super().__init__(parent or win)
        self.win, self.key = win, norm(key)
        self.setWindowTitle('Bind %s' % native(self.key))
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Pick the action %s should run (its current key is replaced):' % native(self.key)))
        self.f = QLineEdit()
        self.f.setPlaceholderText('Search actions')
        self.f.textChanged.connect(self.refill)
        lay.addWidget(self.f)
        self.list = QListWidget()
        self.list.itemActivated.connect(self._pick)
        self.list.itemDoubleClicked.connect(self._pick)
        lay.addWidget(self.list)
        self.resize(420, 420)
        self.refill('')
        self.f.setFocus()

    def refill(self, q):
        q = q.strip().lower()
        self.list.clear()
        for aid, a in sorted(actions(self.win).items()):
            name = _clean(a.text())
            if q and q not in name.lower() and q not in aid.lower():
                continue
            li = QListWidgetItem('%s   [%s]   %s' % (name, ', '.join(native(k) for k in current(a)) or 'none',
                                                   aid.rsplit('/', 1)[0].replace('/', ' > ')))
            li.setData(Qt.UserRole, aid)
            self.list.addItem(li)

    def _pick(self, li):
        assign(self.win, li.data(Qt.UserRole), [self.key])
        self.accept()
