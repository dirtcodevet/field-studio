"""A failed action, shown where the user is looking (fs-shell 11-12; design: when an action a person started fails,
it shows where they look, with what failed and what to do, and nothing is lost silently). Tonight F5's card out-of-memory
and "Her natural base" failed with only a status-bar line, and a harness waited 15 minutes on one).

A failure notice over the top of the viewport: what failed (the action in plain words), why (the error, said plainly for
the kinds we know: the card's memory, the machine's memory, a missing or locked file), what to do, and Details (the
technical text) / Copy / Dismiss. It stays until dismissed; failures that come while it shows are counted on it. Every
failure is kept: win.failures (this session), Help > 'Failures this session', and <autosave dir>/failures.log.

Fed by: the engine's failed signal (a job that raised; the status bar still gets its line), engine and panel status
messages that say an action failed ('... failed: ...', 'FAILED'), and report(win, what, why, todo, details) for a panel
that catches its own error."""
import os
import re
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QVBoxLayout)

# the engine's job kinds, as the action the user started
JOBS = {
    'stroke_begin': 'The brush stroke', 'stroke_flush': 'The brush stroke', 'stroke_end': 'The brush stroke',
    'stroke_move': 'The brush stroke', 'push': 'The command', 'cmd': 'The command', 'add': 'Adding the object',
    'param': 'Changing the shape', 'props': 'Changing the properties', 'oscale': 'Scaling the object',
    'deform': 'The deform', 'wrap': 'The wrap', 'unwrap': 'The unwrap', 'boolean': 'The boolean', 'merge': 'The merge',
    'mirror_dup': 'Mirror-duplicate', 'mask': 'The mask', 'view': 'Changing the view', 'remove': 'Deleting',
    'duplicate': 'Duplicating', 'undo': 'Undo', 'redo': 'Redo', 'goto': 'Stepping the history',
    'open_her': 'Opening her field', 'fill_cavity': 'Fill a cavity', 'her_writeback': 'Writing her field back',
    'set_root': 'Setting the root', 'set_locked': 'Locking', 'protect_masked': 'Protecting the masked part',
    'merge_protected': 'Merging with the protected parts', 'new': 'Starting a new scene', 'open': 'Opening the project',
    'save': 'Saving the project', 'autosave': 'The autosave', 'frame': 'Framing the view',
    'consolidate': 'Consolidating', 'verify_replay': 'Verify replay', 'open_region': 'Opening the field region',
    'write_region': 'Writing the region back', 'gpu_light': 'The light on the card', 'gpu_aid': 'The card view',
    'gpu_ref_store': 'Loading the reference on the card', 'check_overlaps': 'Checking overlaps',
    'show_overlaps': 'Showing overlaps', 'parts_view': 'The parts view', 'adopt_replay': 'Adopting the history',
    'call': 'A tool step', 'idle refinement': "The brush's refinement", 'warm-up': 'Starting the view'}
# a tool step's owner, from the module named in its traceback
OWNERS = (('hair_ui', 'A hair step'), ('hair', 'A hair step'), ('measure', 'A measure'), ('regions', 'A region step'),
          ('cloth', 'A garment step'), ('garment', 'A garment step'), ('library', 'A library step'),
          ('sculpt_layers', 'A layer step'), ('lightpanel', 'A light step'), ('light', 'A light step'),
          ('contact', 'A contact step'), ('pose', 'A pose step'), ('shapecut', 'The cut'), ('cutprev', 'The cut'))


def plain(err, details=''):
    """(why, what to do) for an error text, in plain words where the kind is known"""
    t = '%s\n%s' % (err, details)
    low = t.lower()
    if ('outofmemory' in low.replace(' ', '') or 'out of memory' in low) and ('cuda' in low or 'cupy' in low
                                                                              or 'allocating' in low or 'gpu' in low):
        m = re.search(r'allocating ([\d,]+) bytes', t)
        need = (' (it needed %.1f GB more)' % (int(m.group(1).replace(',', '')) / 1e9)) if m else ''
        return ('the graphics card ran out of memory%s; the card is shared with the other work running' % need,
                'wait for the other card work to finish (or close it), then do it again; a smaller brush or a coarser '
                'level needs less card memory')
    if 'memoryerror' in low or 'unable to allocate' in low or 'not enough memory' in low:
        m = re.search(r'allocate ([\d.]+ ?[KMGT]i?B)', t)
        return ('the computer ran out of memory%s' % ((' (it needed %s)' % m.group(1)) if m else ''),
                'close other programs, or save and restart Field Studio, then do it again')
    m = re.search(r"(FileNotFoundError|No such file or directory)[^\n]*?['\"]([^'\"]+)['\"]", t)
    if m:
        return ('a file it needs is missing: %s' % m.group(2), 'check that the file or folder is there, then do it again')
    if 'permissionerror' in low or 'permission denied' in low:
        return ('a file is locked or read-only', 'close the program holding it (or pick another folder), then do it '
                'again')
    first = next((ln.strip() for ln in str(err).splitlines() if ln.strip()), 'an unexpected error')
    return (first[:240], 'do it again; if it fails again, press Copy and paste the details to the session working on '
            'this part')


def from_engine(text):
    """the engine's failed text ('<job> failed: <error>\\n<traceback>', 'warm-up failed:...', a bare traceback) ->
    (what, err, details)"""
    text = str(text or '')
    head, _, rest = text.partition('\n')
    m = re.match(r'\s*([A-Za-z_ -]+?) (?:failed|stopped):\s*(.*)$', head)
    if m:
        job = m.group(1).strip()
        err = m.group(2).strip() or (rest.strip().splitlines() or ['an error'])[-1]
        what = JOBS.get(job, "The '%s' step" % job)
        if job == 'call':
            for key, name in OWNERS:
                if re.search(r'[\\/]%s[^\\/]*\.py' % key, rest):
                    what = name
                    break
        return what, str(err or 'an error'), text
    last = [ln for ln in text.strip().splitlines() if ln.strip()]
    return 'A step', (last[-1] if last else 'an error'), text


FAILED = re.compile(r'\bfailed\b', re.I)


class FailureNotice(QFrame):
    """over the top of the viewport, until dismissed"""
    shown_failure = Signal(object)

    def __init__(self, win):
        super().__init__(win.viewport)
        self.win = win
        self.setObjectName('failureNotice')
        self.setStyleSheet(
            '#failureNotice { background: #3b1014; border: 2px solid #f0525f; border-radius: 6px; }'
            '#failureNotice QLabel { color: #fde8ea; background: transparent; }'
            '#failureNotice QLabel#title { font-weight: 700; font-size: 14px; color: #ffffff; }'
            '#failureNotice QPushButton { color: #fde8ea; background: #5a1a20; border: 1px solid #f0525f; '
            'border-radius: 3px; padding: 2px 10px; }'
            '#failureNotice QPlainTextEdit { color: #fde8ea; background: #240a0d; border: 1px solid #7a2a31; }')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.title = QLabel(objectName='title')
        self.title.setWordWrap(True)
        self.more = QLabel()
        top.addWidget(self.title, 1)
        top.addWidget(self.more)
        lay.addLayout(top)
        self.why = QLabel()
        self.why.setWordWrap(True)
        self.why.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.todo = QLabel()
        self.todo.setWordWrap(True)
        lay.addWidget(self.why)
        lay.addWidget(self.todo)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(160)
        self.details.hide()
        lay.addWidget(self.details)
        row = QHBoxLayout()
        row.addStretch(1)
        self.b_details = QPushButton('Details')
        self.b_copy = QPushButton('Copy')
        self.b_all = QPushButton('All failures')
        self.b_close = QPushButton('Dismiss')
        for b in (self.b_details, self.b_copy, self.b_all, self.b_close):
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)
            row.addWidget(b)
        lay.addLayout(row)
        self.b_details.clicked.connect(self.toggle_details)
        self.b_copy.clicked.connect(self.copy)
        self.b_all.clicked.connect(lambda: show_all(self.win))
        self.b_close.clicked.connect(self.dismiss)
        self.current = None
        self.unseen = 0
        self.hide()
        win.viewport.installEventFilter(self)

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if ev.type() == QEvent.Resize and self.isVisible():
            self.place()
        return False

    def place(self):
        vp = self.win.viewport
        w = max(260, min(620, vp.width() - 32))
        self.setFixedWidth(w)
        self.adjustSize()
        self.move(max(8, (vp.width() - w) // 2), 12)
        self.raise_()

    def show_failure(self, f):
        if self.isVisible() and self.current is not None:
            self.unseen += 1
        self.current = f
        self.title.setText('%s failed' % f['what'])
        self.why.setText('Why: %s' % f['why'])
        self.todo.setText('What to do: %s' % f['todo'])
        self.more.setText(('+%d more' % self.unseen) if self.unseen else '')
        self.details.setPlainText(f.get('details') or '')
        self.b_details.setVisible(bool(f.get('details')))
        self.place()
        self.show()
        self.raise_()
        self.shown_failure.emit(f)

    def toggle_details(self):
        self.details.setVisible(not self.details.isVisible())
        self.b_details.setText('Hide details' if self.details.isVisible() else 'Details')
        self.place()

    def copy(self):
        f = self.current or {}
        QApplication.clipboard().setText('%s failed (%s)\nWhy: %s\nWhat to do: %s\n\n%s' % (
            f.get('what'), f.get('time'), f.get('why'), f.get('todo'), f.get('details') or ''))
        self.b_copy.setText('Copied')
        QTimer.singleShot(1500, lambda: self.b_copy.setText('Copy'))

    def dismiss(self):
        self.unseen = 0
        self.details.hide()
        self.b_details.setText('Details')
        self.hide()


class _Hub(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.recent = []                      # (text key, t): the same failure said twice (engine + status) shows once

    def seen(self, key):
        now = time.time()
        self.recent = [(k, t) for k, t in self.recent if now - t < 3.0]
        if any(k == key for k, _ in self.recent):
            return True
        self.recent.append((key, now))
        return False


def _log_path(win):
    d = os.environ.get('FIELDSTUDIO_AUTOSAVE_DIR') or os.path.join(os.path.expanduser('~'), '.fieldstudio', 'autosave')
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return os.path.join(d, 'failures.log')


def report(win, what, why, todo=None, details=''):
    """a failure of an action the user started: shown over the view, kept in win.failures and the failures log"""
    hub = getattr(win, '_failure_hub', None)
    if hub is not None and hub.seen((str(what), str(why)[:120])):
        return None
    if todo is None:
        why, todo = plain(why, details)
    f = {'time': time.strftime('%H:%M:%S'), 'what': str(what), 'why': str(why), 'todo': str(todo),
         'details': str(details or '')}
    win.failures.append(f)
    p = _log_path(win)
    if p:
        try:
            with open(p, 'a', encoding='utf-8') as fh:
                fh.write('%s  %s failed\n  Why: %s\n  What to do: %s\n%s\n\n' % (
                    time.strftime('%Y-%m-%d %H:%M:%S'), f['what'], f['why'], f['todo'],
                    '\n'.join('  ' + ln for ln in f['details'].splitlines()[-12:])))
        except OSError:
            pass
    n = getattr(win, 'failure_notice', None)
    if n is not None:
        n.show_failure(f)
    a = getattr(win, 'failures_act', None)
    if a is not None:
        a.setText('Failures this session (%d)...' % len(win.failures))
    return f


def show_all(win):
    dlg = QDialog(win)
    dlg.setWindowTitle('Failures this session')
    lay = QVBoxLayout(dlg)
    t = QPlainTextEdit()
    t.setReadOnly(True)
    t.setPlainText('\n\n'.join('%s  %s failed\nWhy: %s\nWhat to do: %s' % (f['time'], f['what'], f['why'], f['todo'])
                               for f in reversed(win.failures)) or 'No failures this session.')
    lay.addWidget(t)
    dlg.resize(620, 420)
    win.failures_dialog = dlg
    dlg.show()
    return dlg


def install(win):
    win.failures = []
    win._failure_hub = _Hub(win)
    win.failure_notice = FailureNotice(win)
    eng = win.engine

    def on_failed(text):
        win._failure_hub.seen(str(text).splitlines()[0][:200] if str(text) else '')   # its status line (main_window)
        what, err, details = from_engine(text)
        why, todo = plain(err, details)
        report(win, what, why, todo, details)
    # ahead of the window's own handler (its status line would otherwise be read as a second, cruder failure)
    own = getattr(win, 'on_failed', None)
    if own is not None:
        try:
            eng.failed.disconnect(own)
        except (RuntimeError, TypeError):
            own = None
    eng.failed.connect(on_failed)
    if own is not None:
        eng.failed.connect(own)

    def on_message(m):
        m = str(m or '')
        if FAILED.search(m) and not win._failure_hub.seen(m[:200]):
            what, _, err = m.partition(' failed')
            if not err:
                what, _, err = m.partition(' FAILED')
            what = (what.strip(' :') or 'A step')[:80]
            err = err.strip(' :') or m
            why, todo = plain(err)
            report(win, what[:1].upper() + what[1:], why, todo, m)
    eng.message.connect(on_message)
    sb = win.statusBar()
    sb.messageChanged.connect(lambda m: on_message(m) if m else None)
    hm = next((a.menu() for a in win.menuBar().actions() if a.menu() and 'Hel' in a.menu().title().replace('&', '')),
              None)
    if hm is not None:
        a = QAction('Failures this session (0)...', win)
        a.triggered.connect(lambda: show_all(win))
        hm.addAction(a)
        win.failures_act = a
    return win.failure_notice
