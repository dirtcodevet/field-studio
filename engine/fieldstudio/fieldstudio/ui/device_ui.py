"""The render and sculpt device, said plainly (fs-shell 11-11, after a card-view fault: twice tonight
her store failed to load on the card under RAM pressure and the view quietly drew on the CPU behind a 15 s status message).

While the card is expected (FIELDSTUDIO_GPU not 0) but the view is drawing on the CPU, the status bar keeps an amber
notice, "CPU view - the card failed: <the error>", until the card is back (the engine retries it after memory-pressure
failures; its snapshot carries render_device, card_error and card_expected: fs-brush 6617a483). The full error is the
notice's tooltip.

fs-shell-5 11-14 (design: the window must SAY when it sculpts on the CPU; a card harness had reported CPU
numbers as card ones after the engine's warm-up failed to commit a 3.1 GB sculpt pool): the notice also covers
  - the card that never started (the warm-up failed: engine.gpu_view stays None): "The card could not start: the view
    and sculpting run on the CPU (slower)" and why;
  - sculpting on the CPU while the view is on the card (engine.gpu_sculpt missing): "Sculpting on the CPU: ..." and why
    (FIELDSTUDIO_GPU_SCULPT=0 is said as the user's choice, not as a failure).
It is checked on every scene change and twice a second, so it never waits for a snapshot."""
import os

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel


def card_state(win, snap=None):
    """-> None while the card draws and sculpts as expected, else (short text, full reason)"""
    eng = getattr(win, 'engine', None)
    if os.environ.get('FIELDSTUDIO_GPU', '1') == '0' or eng is None or not getattr(eng, 'warm', False):
        return None
    snap = snap if snap is not None else (getattr(win, 'snap', None) or {})
    gv = getattr(eng, 'gpu_view', None)
    if gv is None:
        why = (str(getattr(eng, 'gpu_error', '') or '').strip().splitlines() or ['unknown error'])[-1]
        return ('The card could not start: the view and sculpting run on the CPU (slower)',
                'The graphics card could not start, so the view draws and the brushes sculpt on the CPU (slower):\n%s'
                '\n\nRestart Field Studio when the card is free (other card work closed) to sculpt on the card again.' % why)
    if snap.get('render_device') == 'cpu':
        err = str(snap.get('card_error') or 'unknown error')
        short = err if len(err) <= 70 else err[:67] + '...'
        return ('CPU view - the card failed: %s' % short,
                'The view is drawing on the CPU because the card failed:\n%s\n\nThe engine retries the card after '
                'memory-pressure failures; this notice goes when the card is back.' % err)
    if getattr(eng, 'gpu_sculpt', None) is None:
        if os.environ.get('FIELDSTUDIO_GPU_SCULPT', '1') == '0':
            return ('Sculpting on the CPU (FIELDSTUDIO_GPU_SCULPT=0)',
                    'The brushes sculpt on the CPU because FIELDSTUDIO_GPU_SCULPT=0 is set; the view is on the card.')
        return ('Sculpting on the CPU: the card\'s sculpting could not start',
                'The view is on the card, but the brushes sculpt on the CPU (slower): the card\'s sculpting did not '
                'start.\n%s' % (str(getattr(eng, 'gpu_error', '') or '') or 'No error was recorded.'))
    return None


class DeviceNotice(QLabel):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setStyleSheet('QLabel { color: #1b1300; background: #f0b429; border-radius: 3px; padding: 1px 8px; '
                           'font-weight: 600; }')
        self.hide()
        self._t = QTimer(self)
        self._t.timeout.connect(lambda: self.on_snapshot(None))
        self._t.start(500)

    def on_snapshot(self, snap):
        st = card_state(self.win, snap)
        if st is None:
            if self.isVisible():
                self.hide()
            return
        if self.text() != st[0]:
            self.setText(st[0])
            self.setToolTip(st[1])
        if not self.isVisible():
            self.show()


def install(win):
    n = DeviceNotice(win)
    win.statusBar().addPermanentWidget(n)
    win.engine.sceneChanged.connect(n.on_snapshot)
    win.device_notice = n
    return n
