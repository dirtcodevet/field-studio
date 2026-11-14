"""The Field Studio main window: the concept art's layout (menus, workspace tabs, telemetry strip, left tools,
centre viewport, right properties, bottom timeline), with field modelling in place of its mesh modes."""
import ctypes
import json
import os
import sys
import time

from PySide6.QtCore import Qt, QTimer, QObject, QEvent
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QAbstractSpinBox, QLineEdit)
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QTabBar,
                               QToolBar, QVBoxLayout, QWidget, QListView)

from ..core import ops
from ..render import Camera
from .panels import TOOLS, PropertiesPanel, ToolPanel
from .viewport import Viewport
from .widgets import prim_icon
from PySide6.QtCore import QPointF as QPointFCopy
from .. import tools as FT

# tools.register asks for P / K / Esc, which the sculpt brushes (P pinch, K crease) and Select none already use:
# the field tools take L (line / pencil) and X (cut); Esc exits a tool before it clears the selection.
TOOL_KEYS = {'P': 'L', 'Shift+P': 'Shift+L', 'K': 'X', 'Shift+K': 'Shift+X', 'Esc': None}
# fs-shell-3 11-14 (registry stage 2b, the keys): the approved interface's keys for the tools the window runs through
# fs-tools' own jobs (the approved mock: Place P, Cut X, Connect K, Mirror M, Swell I; Shapes A
# goes to Shift+A, Blender's Add, because A held + click is select-inside's way back out). Today's brushes that
# held those letters (Inflate I, Crease K, Pinch P) and the outline Cut (X) keep their menu entries, unbound, rebindable
# in the Keys window. FS_LAYOUT=classic keeps today's keys.
APPROVED_LAYOUT = os.environ.get('FS_LAYOUT', '') != 'classic'
APPROVED_TOOL_KEYS = {'inflate': None, 'crease': None, 'pinch': None, 'swell': 'I'}
REG_KEYS = [('Shape library (Add)', 'reg:add', 'Shift+A'), ('Place on surface', 'reg:place', 'P'),
            ('Cut (a line, or a shape)', 'reg:cut', 'X'), ('Connect', 'reg:connect', 'K'),
            ('Mirror (the copies tool)', 'reg:copies:mirror', 'M')]
LENGTH_FIELDS = ('width', 'depth', 'softness', 'distance', 'kerf', 'blend', 'radius', 'radius_a', 'radius_b',
                 'detail_scale')
CHOICES = {'profile': ['round', 'V', 'U', 'custom'], 'mode': ['ridge', 'valley']}


class ToolHooks:
    """The ui_hooks of tools.register. Tool modes run on the ENGINE thread (engine job 'mode'), so their picks
    and pushes keep the threading rule; camera and size come from the view the event was posted with."""

    def __init__(self, win):
        self.win = win
        self.engine = win.engine

    @property
    def history(self):
        return self.engine.hist

    def add_action(self, menu, label, callback, shortcut=None):
        return self.win.add_tool_action(label, callback, shortcut)

    def get_selection(self):
        order = list(self.win.props.order)
        return order[0] if order else None

    def _view(self):
        return self.engine._mode_view or self.win.viewport.view()

    def get_camera(self):
        return self._view()['cam']

    def viewport_size(self):
        v = self._view()
        return v['w'], v['h']

    def set_tool_mode(self, mode):
        self.win.set_tool_mode(mode)

    def pick(self, x, y):
        return self.engine._pick(self._view(), x, y)

    def ask(self, title, fields):
        return self.win.ask(title, fields)

    def status(self, msg):
        self.engine.message.emit(msg)

    def refresh(self):
        pass

    def push(self, cmd):
        self.engine.post('cmd', cmd, None)

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCREENS = os.path.join(HERE, 'screens')

PRIMS = [('sphere', 'Sphere', {'radius': 0.08}),
         ('box', 'Cube', {'size': [0.14, 0.14, 0.14], 'round': 0.006}),
         ('cylinder', 'Cylinder', {'radius': 0.06, 'height': 0.16, 'round': 0.004}),
         ('capsule', 'Capsule', {'radius': 0.05, 'height': 0.12}),
         ('cone', 'Cone', {'radius_bottom': 0.08, 'radius_top': 0.0, 'height': 0.16}),
         ('torus', 'Torus', {'major': 0.08, 'minor': 0.025}),
         ('ellipsoid', 'Ellipsoid', {'radii': [0.1, 0.07, 0.06]}),
         ('slab', 'Slab', {'normal': [0.0, 1.0, 0.0], 'offset': 0.0, 'thickness': 0.02, 'extent': 0.12})]
def prim_size(kind, params):
    """fs-shell: a primitive's size (m): its radius or its largest half extent"""
    p = params or {}
    if kind == 'sphere':
        return float(p.get('radius', 0.05))
    if kind == 'box':
        return 0.5 * max(float(v) for v in p.get('size', [0.1]))
    if kind in ('cylinder', 'capsule'):
        return max(float(p.get('radius', 0.05)), 0.5 * float(p.get('height', 0.1)) + (float(p.get('radius', 0.05))
                                                                                      if kind == 'capsule' else 0.0))
    if kind == 'cone':
        return max(float(p.get('radius_bottom', 0.05)), float(p.get('radius_top', 0.0)), 0.5 * float(p.get('height', 0.1)))
    if kind == 'torus':
        return float(p.get('major', 0.05)) + float(p.get('minor', 0.01))
    if kind == 'ellipsoid':
        return max(float(v) for v in p.get('radii', [0.05]))
    if kind == 'slab':
        return float(p.get('extent', 0.1))
    return 0.06


PALETTE = [[0.62, 0.52, 0.45], [0.45, 0.52, 0.62], [0.55, 0.6, 0.45], [0.62, 0.45, 0.5], [0.58, 0.55, 0.42],
           [0.45, 0.58, 0.58]]


def process_memory():
    """-> (working set MB, peak working set MB) of this process (Windows: GetProcessMemoryInfo)."""
    try:
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError('GetProcessMemoryInfo failed')
        process_memory.commit = (pmc.PagefileUsage / 2 ** 20, pmc.PeakPagefileUsage / 2 ** 20)
        return pmc.WorkingSetSize / 2 ** 20, pmc.PeakWorkingSetSize / 2 ** 20
    except Exception:
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            return r, r
        except Exception:
            return 0.0, 0.0


class Telemetry(QWidget):
    """The concept art's telemetry strip: FPS / frame ms / RAM / field memory / undo memory / L / CPU threads."""

    def __init__(self):
        super().__init__()
        self.setObjectName('telemetry')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 3, 10, 3)
        lay.setSpacing(16)
        self.values = {}
        for key, title in (('fps', 'FPS'), ('ms', 'Frame'), ('ram', 'RAM'), ('field', 'Field'), ('undo', 'Undo'),
                           ('L', 'Lipschitz L'), ('cpu', 'Render')):
            box = QVBoxLayout()
            box.setSpacing(0)
            t = QLabel(title)
            t.setObjectName('telemetryLabel')
            v = QLabel('-')
            v.setObjectName('telemetryValue')
            box.addWidget(t)
            box.addWidget(v)
            lay.addLayout(box)
            self.values[key] = v

    def set(self, key, text):
        self.values[key].setText(text)


class OpLog(QWidget):
    """Bottom strip: the op log as a timeline (done entries, then undone ones dimmed). Click = go to that step."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        head = QHBoxLayout()
        self.info = QLabel('Op log - every model change is a command; the log replays to the same state hash.')
        self.info.setObjectName('dim')
        head.addWidget(self.info, 1)
        b = QPushButton('Verify replay')
        b.setToolTip('Replay the op log into an empty scene and compare state hashes')
        b.clicked.connect(lambda: engine.post('verify_replay'))
        head.addWidget(b)
        # fs-shell-2 11-14 (step 6, the view's area: at 1366 x 768 the 110 px Timeline was the view's biggest loss): in
        # the approved layout the steps fold away to this one line by default (not in the design; rarely used);
        # 'Steps' opens them, remembered per person. FS_LAYOUT=classic keeps them open
        self.steps_btn = QPushButton('Steps \u25b8')
        self.steps_btn.setCheckable(True)
        self.steps_btn.setToolTip('Show the undo steps as a timeline (click one to step there)')
        self.steps_btn.toggled.connect(self.set_open)
        head.insertWidget(0, self.steps_btn)
        lay.addLayout(head)
        self.list = QListWidget()
        self.list.setFlow(QListView.LeftToRight)
        self.list.setWrapping(False)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list.setFixedHeight(46)
        self.list.itemClicked.connect(self._clicked)
        lay.addWidget(self.list)
        classic = os.environ.get('FS_LAYOUT', '') == 'classic'
        try:
            opened = classic or str(app_settings().value('timeline_open', 'false')).lower() == 'true'
        except Exception:
            opened = True
        self.steps_btn.setChecked(opened)
        self.set_open(opened)

    def set_open(self, on):
        self.list.setVisible(bool(on))
        self.steps_btn.setText('Steps \u25be' if on else 'Steps \u25b8')
        if self.steps_btn.isChecked() != bool(on):
            self.steps_btn.setChecked(bool(on))
        try:
            app_settings().setValue('timeline_open', 'true' if on else 'false')
        except Exception:
            pass

    def update_snapshot(self, snap):
        self.list.blockSignals(True)
        self.list.clear()
        base = snap.get('base_log', 0)
        if base:
            it = QListWidgetItem('  %d loaded  ' % base)
            it.setData(Qt.UserRole, None)
            self.list.addItem(it)
        n = len(snap['log'])
        # fs-shell 11-11: one entry per undo step; a group (several commands undone as one, e.g.
        # Soften the joins) shows once, named for what was done, and a click steps to its end
        steps = self._steps(snap['log'], snap.get('log_groups'), 0)
        rsteps = self._steps(snap['redo_log'], snap.get('redo_groups'), n)
        for k, (lab, end, cnt) in enumerate(steps):
            it = QListWidgetItem('  %d  %s  ' % (k + 1, lab if cnt == 1 else '%s (%d)' % (lab, cnt)))
            it.setData(Qt.UserRole, end)
            self.list.addItem(it)
        for j, (lab, end, cnt) in enumerate(rsteps):
            it = QListWidgetItem('  %d  %s  ' % (len(steps) + j + 1, lab if cnt == 1 else '%s (%d)' % (lab, cnt)))
            it.setData(Qt.UserRole, end)
            it.setForeground(Qt.gray)
            self.list.addItem(it)
        if steps:
            cur = self.list.item(len(steps) - 1 + (1 if base else 0))
            cur.setSelected(True)
            self.list.scrollToItem(cur)
        self.list.blockSignals(False)
        self.info.setText('Op log: %d steps done, %d undone. Click an entry to step there. Undo history %.1f MB.' % (
            len(steps), len(rsteps), snap['hist_mem'] / 2 ** 20))

    @staticmethod
    def _steps(labels, groups, offset):
        """[(label, the log index to step to (its last command), commands in it)]: a group starts where groups[i] > 1"""
        out, i = [], 0
        groups = list(groups or [])
        while i < len(labels):
            g = groups[i] if i < len(groups) else 0
            cnt = g if g > 1 and i + g <= len(labels) else 1
            out.append((labels[i], offset + i + cnt, cnt))
            i += cnt
        return out

    def _clicked(self, it):
        k = it.data(Qt.UserRole)
        if k is not None:
            self.engine.post('goto', int(k))


def app_settings():
    """hand1b: the app's QSettings. FIELDSTUDIO_SETTINGS=<file.ini> points tests at a scratch file, so a test never
    writes the user's registry (test_hand1 once set quick_start_seen there and used up the one-time Quick start)."""
    from PySide6.QtCore import QSettings
    path = os.environ.get('FIELDSTUDIO_SETTINGS')
    if path:
        return QSettings(path, QSettings.IniFormat)
    return QSettings('FieldStudio', 'FieldStudio')


class KeyFilter(QObject):
    """hand1: an application-wide key filter.
    1. A number box (spin box) with focus no longer swallows the tool letter keys: Qt asks the focus widget first
       (ShortcutOverride) and a spin box's line edit accepts every printable key, so before, D / C / S ... did nothing
       after typing a radius (diagnosis root cause 3). Letters and [ ] go to the shortcuts; digits, '.', '-' stay
       in the box. Text fields (the Name box) keep every key.
    2. F held over the view: drag to set the brush radius; Shift+F: strength (release F or click to end)."""

    LETTERS = set(range(Qt.Key_A, Qt.Key_Z + 1)) | {Qt.Key_BracketLeft, Qt.Key_BracketRight, Qt.Key_BraceLeft,
                                                     Qt.Key_BraceRight}    # fs-shell-5: + Shift+[ ] (hardness)
    # hand1b: '=' never belongs in a number; '-' and '+' only in a box that takes negative numbers (the reference
    # offsets); Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z are the app's undo / redo (a box's value change is itself an undo step,
    # so the box's own text undo only hid the app's undo)
    SIGNS = {Qt.Key_Equal, Qt.Key_Plus, Qt.Key_Minus}
    UNDO = {Qt.Key_Z, Qt.Key_Y}

    def __init__(self, win):
        super().__init__(win)
        self.win = win

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.ShortcutOverride:
            mods = ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
            spin = self._in_spin()
            if spin is None:
                return False
            k = ev.key()
            give = ((not mods and k in self.LETTERS)
                    or (not mods and k in self.SIGNS and (k == Qt.Key_Equal or getattr(spin, 'minimum', lambda: 0)() >= 0))
                    or (mods == Qt.ControlModifier and k in self.UNDO))
            if give:
                ev.ignore()                    # not overridden: the application shortcut gets the key
                return True
            return False
        if t in (QEvent.KeyPress, QEvent.KeyRelease) and ev.key() == Qt.Key_F and not ev.isAutoRepeat():
            vp = self.win.viewport
            if t == QEvent.KeyRelease:
                if vp.adjust is not None:
                    vp.end_adjust()
                    return True
                return False
            mods = ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
            if mods in (Qt.NoModifier, Qt.ControlModifier) and vp.underMouse() and not self._typing():
                # fs-shell-5 11-14: Ctrl+F-drag sets the hardness
                return vp.begin_adjust('hardness' if mods == Qt.ControlModifier else
                                       'strength' if ev.modifiers() & Qt.ShiftModifier else 'radius')
        return False

    @staticmethod
    def _in_spin():
        """-> the number box (spin box) that has the keyboard focus, or None"""
        fw = QApplication.focusWidget()
        if isinstance(fw, QAbstractSpinBox):
            return fw
        if isinstance(fw, QLineEdit) and isinstance(fw.parent(), QAbstractSpinBox):
            return fw.parent()
        return None

    @staticmethod
    def _typing():
        fw = QApplication.focusWidget()
        return isinstance(fw, QLineEdit) and not isinstance(fw.parent(), QAbstractSpinBox)


def _session_alive(session):
    """hand2b: a session name ends in _<pid> (MainWindow.session); True when that process is still running, so its
    autosave belongs to an open window and is not a crash leftover (verifier item 25)."""
    try:
        pid = int(session.rsplit('_', 1)[1])
    except (IndexError, ValueError):
        return False
    if pid == os.getpid():
        return True
    if os.name == 'nt':
        import ctypes
        k32 = ctypes.windll.kernel32
        hp = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not hp:
            return False
        try:
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(hp, ctypes.byref(code))
            return bool(ok) and code.value == 259          # STILL_ACTIVE
        finally:
            k32.CloseHandle(hp)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.project_path = None
        self.snap = None
        self.errors = []
        self.colour_i = 0
        self.setWindowTitle('Field Studio V1 - implicit-field modelling (CPU)')
        self.resize(1600, 960)

        self.viewport = Viewport(engine)
        self.tools = ToolPanel()
        self.props = PropertiesPanel(engine)
        self.props.viewport = self.viewport        # fs-shell 11-13: the X-ray's per-object opacity (view state)
        self.oplog = OpLog(engine)
        self.telemetry = Telemetry()

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self._viewport_bar())
        cl.addWidget(self.viewport, 1)
        self.setCentralWidget(central)

        from .refpanel import RefPanel
        self.refpanel = RefPanel(self.viewport)
        self.refpanel.lensChanged.connect(lambda on: self._sync_action(self.act_lens, on))
        self.refpanel.orthoChanged.connect(lambda on: self._sync_action(self.act_ortho, on))
        self.refpanel.visibleChanged.connect(lambda on: self._sync_action(self.act_refs, on))
        self.viewport.panel = self.tools
        self.tools.axis.currentIndexChanged.connect(self.on_axis)
        docks = {}
        for title, w, area, width in (('Tools', self.tools, Qt.LeftDockWidgetArea, 290),
                                      ('Properties', self.props, Qt.RightDockWidgetArea, 360),
                                      ('References', self.refpanel, Qt.RightDockWidgetArea, 360)):
            d = QDockWidget(title)
            d.setWidget(w)
            d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
            d.setMinimumWidth(width)
            self.addDockWidget(area, d)
            docks[title] = d
        self.tabifyDockWidget(docks['Properties'], docks['References'])
        docks['Properties'].raise_()
        self.docks = docks
        d = QDockWidget('Timeline')
        d.setWidget(self.oplog)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, d)

        self._menus()
        self.ask_auto = None                  # self-test: {dialog title: values} answers the tool dialogs
        self.tool_hooks = ToolHooks(self)
        self.field_tools = FT.register(self.tool_hooks)
        self.tools.set_field_tools(self.field_tools)
        self._top_bars()
        self.status = self.statusBar()
        self.status.showMessage('Compiling field kernels on the CPU...')
        from .widgets import ElideLabel
        self.hint = ElideLabel('', min_w=160)      # fs-shell 11-11: cut with ... when narrow (it held the window at 1632 px)
        self.hint.setObjectName('dim')
        self.status.addPermanentWidget(self.hint)
        # a message (left) and the hint (right) share the bar: while a message shows, the hint gives it the room its
        # text needs (fs-shell-6 11-14: at half the bar, 'Grouped 2 objects, named ... (Ctrl+Shift+' was cut mid-word
        # where the hint began), down to the hint's own 160 px
        self.status.messageChanged.connect(self._fit_hint)
        self.update_hint()

        # wiring
        engine.sceneChanged.connect(self.on_snapshot)
        engine.message.connect(lambda m: self.status.showMessage(m, 12000))
        engine.failed.connect(self.on_failed)
        engine.saveDone.connect(self.on_save_done)       # fs-store 11-12: a failed save is said, and the work stays unsaved
        engine.regionDone.connect(self.on_region_done)
        self.region_report = None
        engine.ready.connect(self.on_ready)
        engine.progress.connect(self.viewport.set_warm_progress)
        engine.boundsReady.connect(self.on_bounds)
        engine.projectLoaded.connect(self.on_project_loaded)
        self.viewport.selectRequest.connect(self.on_viewport_select)
        self.viewport.contextRequest.connect(self.on_context)              # hand2f: the right-click action menu
        self.viewport.statsChanged.connect(self.on_stats)
        self.props.selectionChanged.connect(self.viewport.set_selected)
        self.tools.toolChanged.connect(self.on_tool)
        self.tools.settingsChanged.connect(self.on_brush)
        self.tools.showMaskChanged.connect(self.on_show_mask)
        self.viewport.brush = self.tools.brush()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)
        self.tick()
        self.ask_memory = {}                  # hand1: a field tool's settings, asked on first use (Ctrl+click: again)
        self.key_filter = KeyFilter(self)
        from PySide6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self.key_filter)
        # hand2f: save safety. No prompt and no autosave in automated runs (offscreen / --selftest / --monkey) unless
        # FIELDSTUDIO_AUTOSAVE_DIR is set (tests point it at a scratch folder)
        self.no_prompt = (os.environ.get('QT_QPA_PLATFORM', '') in ('offscreen', 'minimal')
                          or bool(os.environ.get('FIELDSTUDIO_NO_PROMPT')))
        self.autosave_dir = os.environ.get('FIELDSTUDIO_AUTOSAVE_DIR') or (
            None if self.no_prompt else os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'),
                                                     'FieldStudio', 'autosave'))
        self.session = time.strftime('%Y%m%d_%H%M%S') + '_%d' % os.getpid()
        self.saved_marker = ([], 0)           # (log_ids, committed) at the last save / open / new scene
        self._mark_clean_at = 0          # hand2f: the snapshot with this many done steps is 'clean'
        self.autosave_marker = ([], 0)
        self.autosave_t = time.perf_counter()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave_tick)
        self.autosave_timer.start(10000)
        # fs-shell (design 4.1 / 4.2 / 5.1): the tool shelf, the context bar, the favourites strip, the Space palette
        # and the Keys window
        cfg_ = app_settings()                  # fs-shell 11-11 (design D2): the grids as the user left them; 11-12 (t51): on
        sp_ = str(cfg_.value('grid_spacing', '') or '')   # until turned off, as Blender opens
        self.set_grid(floor=str(cfg_.value('grid_floor', 'true')).lower() == 'true',
                      view=str(cfg_.value('grid_view', 'true')).lower() == 'true',
                      spacing=float(sp_) if sp_ else None, remember=False,
                      contours=str(cfg_.value('grid_contours', 'false')).lower() == 'true')
        from . import shell as SH, keys_window as KW, keymap as KM
        KM.install(self)                       # the user's rebound keys (design 4.2), before anything reads a key
        SH.install(self)
        KW.install(self)
        from .cutprev import install as _cut_preview
        _cut_preview(self)                     # fs-shell 11-11 (design 4.18): the cutter's live preview
        from .measure_ui import install as _measures
        _measures(self)                        # fs-shell 11-11 (queue item 5): pinned measures
        from .profile_ui import install as _profile
        _profile(self)                         # fs-shell 11-11: girth profiles, landmarks
        try:                                   # fs-shell 11-11 (design D4, the UI half): the topology count,
            from .topo_ui import install as _topo   # the closed-opening prompt, the hidden-void view
            _topo(self)
        except Exception as e_:
            self._install_failed('topology UI', e_)
        try:                                   # fs-shell 11-11 (test 1 with the kit): Create > Mass > Soften
            from .joins_ui import install as _joins   # the joins (Smooth along where masses meet)
            _joins(self)
        except Exception as e_:
            self._install_failed('Soften the joins', e_)
        try:                                   # fs-shell 11-11 (the copy / scale / rotate / place flow): W W, R R,
            from .gizmo import install_window as _gz_win   # Y Y start a typed transform (no drag needed)
            _gz_win(self)
        except Exception as e_:
            self._install_failed('typed transforms', e_)
        try:                                   # fs-shell-3 11-14 (TOOLSET_DESIGN 3.2, registry stage 2b): Move / Rotate
            from .region_xform import install as _rx   # on a masked object move the part the Mask leaves free (fs-tools'
            _rx(self)                                   # tools.region_move): its soft edge tinted, one undo step
        except Exception as e_:
            self._install_failed('region transform', e_)
        try:                                   # fs-shell 11-14 (Tier 0 part 5: how to rotate or
            from .xform_strip import install as _xs   # manipulate an object was not findable): Move / Rotate / Scale named on the
            _xs(self)                                 # view whenever a form is selected
        except Exception as e_:
            self._install_failed('transform strip', e_)
        try:                                   # fs-shell-2 11-14 (the window / tools contract):
            from ..toolset.jobs import register as _tool_jobs   # fs-tools' jobs on this engine, before the column
            self.tool_jobs = _tool_jobs(self.engine)            # reads which of its tools can run
        except Exception as e_:
            self.tool_jobs = []
            if not (isinstance(e_, ModuleNotFoundError) and 'toolset' in str(getattr(e_, 'name', ''))):
                self._install_failed('toolset jobs', e_)
        try:                                   # fs-shell-3 11-14 (registry stage 2b): the view's presses for the tools
            from .reg_mouse import install as _rm     # that act where the cursor points (Place on surface; Add, Cut, Connect)
            _rm(self)
        except Exception as e_:
            self._install_failed('registry tools in the view', e_)
        try:                                   # fs-shell 11-14 (the approved interface, 3.2): the four categories,
            from .categories import install as _cats  # bold, on the left (FS_LAYOUT=classic: the old shelf)
            _cats(self)
        except Exception as e_:
            self._install_failed('categories column', e_)
        try:                                   # fs-shell 11-14 (the approved interface, 3.3): the tool strip above
            from .toolstrip import install as _strip  # the view: the four shared controls in fixed places
            _strip(self)
        except Exception as e_:
            self._install_failed('tool strip', e_)
        try:                                   # fs-shell 11-14 (the approved interface, 3.4): the view bar (the
            from .viewbar import install as _vbar     # object's opacity; Clay / Look / Silhouette, X-ray, Section,
            _vbar(self)                               # Measure, Refs, Mirror X)
        except Exception as e_:
            self._install_failed('view bar', e_)
        try:                                   # fs-shell 11-11 (design 4.14, chunk 9): Marks, a spot circled
            from .marks_ui import install as _marks   # on the model and named, saved with the project
            _marks(self)
        except Exception as e_:
            self._install_failed('Marks', e_)
        try:                                   # fs-shell 11-11 (design 4.4): the Section tool, a plane that
            from .section_ui import install as _sect   # cuts the view; thickness, gap, mm ruler
            _sect(self)
        except Exception as e_:
            self._install_failed('Section tool', e_)
        try:                                   # fs-shell 11-11: Flip X / Y / Z in place, a
            from .flip_ui import install as _flip      # mirror copy across any axis
            _flip(self)
        except Exception as e_:
            self._install_failed('Flip / mirror', e_)
        try:                                   # fs-shell 11-11 (chunk 9): matcap / curvature / normals / cavity drawn
            from ..gpu.viewmodes import install as _vmodes     # on the card (a drop-in for the light pass)
            _vmodes(self.engine)
        except Exception as e_:
            self._install_failed('Card view modes', e_)
        try:                                   # fs-shell 11-11 (design 4.10, chunk 9): sculpt layers (record a pass,
            from .sculpt_layers_ui import install as _layers   # dial it -1..2, solo, merge) and snapshots
            _layers(self)
        except Exception as e_:
            self._install_failed('Layers', e_)
        try:                                   # fs-shell 11-11 (design 4.6, chunk 9): the brush ring ON the surface,
            from .cursor_ui import install as _cursor      # the mirror ring, live mm (Alt: thickness and gap)
            _cursor(self)
        except Exception as e_:
            self._install_failed('Surface cursor', e_)
        try:                                   # fs-shell 11-11: a lasting notice while the view
            from .device_ui import install as _device      # draws on the CPU because the card failed
            _device(self)
        except Exception as e_:
            self._install_failed('Device notice', e_)
        try:                                   # fs-shell 11-12: a failed action shown where the user looks:
            from .failure_ui import install as _failures   # what failed, why, what to do; kept, logged
            _failures(self)
        except Exception as e_:
            self._install_failed('Failure notice', e_)
        try:                                   # fs-shell 11-11 (design 4.8): Regions - lasso, from mask / part,
            from .regions_ui import install as _regions   # H / Shift+H / Alt+H, box hide, the panel
            _regions(self)
        except Exception as e_:
            self._install_failed('Regions', e_)
        try:                                   # fs-light's Light panel (ui/lightpanel.py, its file; hook by
            from .lightpanel import install as _light   # fs-shell 11-11): inert until it exists
            _light(self)
        except Exception as e_:
            if not (isinstance(e_, ModuleNotFoundError) and str(getattr(e_, 'name', '')).endswith('lightpanel')):
                self._install_failed('Light panel', e_)
        try:                                   # fs-hair 11-11 (design 4.13, the hair workflow): the Hair dock -
            from .hair_ui import install as _hair      # area, kind, details, Generate, Comb, Cut, Trim, states
            _hair(self)
        except Exception as e_:
            self._install_failed('Hair', e_)
        try:                                   # fs-cloth 11-11 (design D3; ui/garment_ui.py, its file; hook by fs-shell):
            from .garment_ui import install as _garment   # Create > Garment... dresses the selected body
            _garment(self)
        except Exception as e_:
            if not (isinstance(e_, ModuleNotFoundError) and str(getattr(e_, 'name', '')).endswith('garment_ui')):
                self._install_failed('Garment panel', e_)
        try:                                   # fs-mouth 11-11 (ui/library_ui.py, its file; hook by fs-shell): Library >
            from .library_ui import install as _library   # Mouth > teeth, gums, the set, with their settings
            _library(self)
        except Exception as e_:
            if not (isinstance(e_, ModuleNotFoundError) and str(getattr(e_, 'name', '')).endswith('library_ui')):
                self._install_failed('Library', e_)
        try:                                   # fs-materials 11-14 (ui/materials_ui.py, its file; hook by fs-shell): the
            from .materials_ui import install as _materials   # Materials & Generators dock (design 5.1)
            _materials(self)
        except Exception as e_:
            if not (isinstance(e_, ModuleNotFoundError) and str(getattr(e_, 'name', '')).endswith('materials_ui')):
                self._install_failed('Materials panel', e_)
        if 'Properties' in self.docks:         # fs-shell 11-12: the window opens on Properties, whichever panel was
            self.docks['Properties'].raise_()  # installed last (the Hair panel, tabified last, was left on top)
        try:                                   # fs-shell 11-14 (the approved interface, rule 5): the panels as tabs on
            from .panels2 import install as _panels   # an edge: one open at a time, pin, pop out, dock to any edge,
            _panels(self)                             # remembered per person (after every dock exists)
        except Exception as e_:
            self._install_failed('panels', e_)
        try:                                   # fs-shell 11-14 (X-ray): click again to select the object
            from .select_inside import install as _si   # inside, A+click to go back out
            _si(self)
        except Exception as e_:
            self._install_failed('select inside', e_)
        vm = next((a.menu() for a in self.menuBar().actions() if a.menu() and 'View' in a.menu().title()), None)
        if vm is not None:
            vm.addSeparator()
            vm.addAction(self.shelf_bar.toggleViewAction())
            vm.addAction(self.docks['Tools'].toggleViewAction())
        try:                                   # fs-shell-5 11-14: the quick keys on the view, before the
            from .quick_keys import install as _qk     # top bar (its Quick keys button follows it)
            _qk(self)
        except Exception as e_:
            self._install_failed('the quick keys', e_)
        try:                                   # fs-shell-2 11-14 (the approved interface, step 5): one top row,
            from .topbar import install as _topbar   # Field Studio, File / Edit / View / Help (the other menus
            _topbar(self)                             # folded in), the search box, live state; last, so it
        except Exception as e_:                       # folds every menu the installs above made
            self._install_failed('top bar', e_)
        QTimer.singleShot(0, self._show_install_failures)   # fs-shell-5 11-14: shown once the window is up

    # ---- fs-shell-5 11-14: a part of the window that fails to build is SHOWN to the user ------------------------------
    # (11-14: the Materials panel failed to build, the window wrote one log line and went on without it; the
    # user saw only that Materials would not open). Every install below records its failure here; once the window is up,
    # each is shown over the view (failure_ui.report: what, why, what to do) and a lasting mark stays on the status line
    # until it is read ('1 part of the window failed to build'); tests/test_window_builds.py fails on any of them.
    def _install_failed(self, what, e):
        import traceback
        tb = traceback.format_exc()
        if not hasattr(self, 'install_failures'):
            self.install_failures = []
        self.install_failures.append({'what': what, 'error': repr(e), 'trace': tb})
        print('%s not installed: %r' % (what, e))
        if getattr(self, '_fail_mark', None) is not None:
            self._show_install_failures()

    def _show_install_failures(self):
        fails = list(getattr(self, 'install_failures', []) or [])
        if not fails:
            return
        from PySide6.QtWidgets import QPushButton
        b = getattr(self, '_fail_mark', None)
        if b is None:
            b = QPushButton()
            b.setObjectName('failMark')
            b.setStyleSheet('#failMark { color: #1a1300; background: #ffcc4d; border: none; border-radius: 4px; '
                            'padding: 2px 10px; font-weight: 600; }')
            b.setToolTip('Show what failed to build, why, and what to do')
            b.clicked.connect(self._open_install_failures)
            self.status.addPermanentWidget(b)
            self._fail_mark = b
        n = len(fails)
        b.setText('%d part%s of the window failed to build: %s' % (
            n, '' if n == 1 else 's', ', '.join(f['what'] for f in fails[:3]) + (' ...' if n > 3 else '')))
        b.show()
        try:
            from .failure_ui import report
            shown = getattr(self, '_fails_reported', 0)
            for f in fails[shown:]:
                report(self, '%s could not be built' % f['what'],
                       '%s; the rest of the window works without it' % f['error'],
                       'Restart Field Studio after the next update; if it stays, the details are kept '
                       '(Copy, or the session log)', f['trace'])
            self._fails_reported = len(fails)
        except Exception as e_:
            print('install failures not shown: %r' % (e_,))

    def _open_install_failures(self):
        try:
            from .failure_ui import show_all
            show_all(self)
        except Exception:
            pass

    # ---- building ------------------------------------------------------------------------------------------------
    def _act(self, menu, text, fn, shortcut=None, checkable=False):
        a = QAction(text, self)
        if shortcut:
            a.setShortcuts([QKeySequence(s) for s in (shortcut if isinstance(shortcut, (list, tuple))
                                                      else [shortcut])])
            # hand1: application-wide, so the keys work with focus anywhere (the Quick start panel, a dock, a dialog
            # left open); number boxes give the letter keys up through the app key filter (KeyFilter below)
            a.setShortcutContext(Qt.ApplicationShortcut)
        a.setCheckable(checkable)
        a.triggered.connect(fn)
        menu.addAction(a)
        self.addAction(a)
        return a

    @staticmethod
    def _sync_action(act, on):
        """hand1b: a View menu check mark follows the References panel (one state, shown in two places)"""
        if act.isChecked() != bool(on):
            act.blockSignals(True)
            act.setChecked(bool(on))
            act.blockSignals(False)

    def _menus(self):
        mb = self.menuBar()
        m = mb.addMenu('&File')
        # fs-shell (design 4.5): File > New opens the start page (Head, Figure, Hand, Empty); the starts are also here
        self._act(m, 'New...', self.new_start, 'Ctrl+N')
        self._act(m, 'New head (sphere + neck, 1 mm lattice, Mirror X; then photos)', lambda: self.new_head(True),
                  'Ctrl+Shift+N')
        self._act(m, 'New figure (a 1680 mm standing blockout, 1 mm lattice)', self.new_figure, 'Ctrl+Alt+N')
        self._act(m, 'New hand (palm and fingers, 0.5 mm lattice)', self.new_hand)
        self._act(m, 'New bust (a plain sphere, 1 mm lattice)', self.new_bust)
        self._act(m, 'New empty scene', self.new_scene)
        self._act(m, 'Add reference images (front, side)...', self.add_reference_images, 'Ctrl+R')
        self._act(m, 'Reference Rig (align front and side)...', lambda: self.refpanel.open_rig(), 'Ctrl+Shift+R')
        self._act(m, 'Open project (.fsproj)...', self.open_project, 'Ctrl+O')
        self._act(m, 'Save project', self.save_project, 'Ctrl+Shift+S')
        self._act(m, 'Save project as...', lambda: self.save_project(ask=True))
        m.addSeparator()
        self._act(m, 'Open her field (editable copy, whole body)', self.open_her_editable)
        self._act(m, 'Write her edits to a new version...', self.write_her_back)
        self._act(m, 'Open field region...', self.open_field_region)
        self._act(m, 'Write region back (new version)...', self.write_region_back)
        m.addSeparator()
        self._act(m, 'Save viewport PNG (screens/)', self.screenshot, 'Ctrl+S')
        self._act(m, 'Save window PNG (screens/)', lambda: self.screenshot(window=True))
        m.addSeparator()
        self._act(m, 'Quit', self.close, 'Ctrl+Q')

        m = mb.addMenu('&Edit')
        self._act(m, 'Undo', lambda: self.engine.post('undo'), 'Ctrl+Z')
        self._act(m, 'Redo', lambda: self.engine.post('redo'), ['Ctrl+Y', 'Ctrl+Shift+Z'])
        m.addSeparator()
        self._act(m, 'Duplicate object (with its sculpt)', self.props._duplicate, 'Ctrl+D')
        self._act(m, 'Mirror-duplicate across X (world, with its sculpt)', self.props._mirror_dup, 'Ctrl+Shift+D')
        self._act(m, 'Delete object', self.props._delete, 'Delete')
        self._act(m, 'Rename object...', self.rename_object, 'F2')
        m.addSeparator()
        self._act(m, 'Hide / show selected (view only)', self.props._hide, 'H')
        self._act(m, 'Isolate selected (again: show all)', self.isolate, '/')
        self._act(m, 'Show all', lambda: self.engine.post('view', 'show_all', []), 'Shift+H')
        m.addSeparator()
        self._act(m, 'Select all', self.select_all, 'Ctrl+A')
        self._act(m, 'Invert selection', self.select_invert, 'Ctrl+Shift+I')
        self._act(m, 'Exit tool / select none', self.escape, 'Escape')

        m = mb.addMenu('&Create')
        for kind, name, params in PRIMS:
            a = self._act(m, name, lambda _=False, k=kind, p=params: self.create(k, p))
            a.setIcon(prim_icon(kind))
        # fs-shell 11-11 (design 4.18, the cut tool): a cutter of any primitive shape, placed with the gizmo, then
        # Apply cut on the context bar: inside / outside, delete / lift out / keep both, one undo step
        # fs-shell 11-11 (design 4.5): masses: live primitives smooth-joined into the selected object (its blend dial
        # on the context bar), placed and shaped with the gizmo
        mma = m.addMenu('Mass (add to the selected object)')
        for kind, name, params in PRIMS:
            a = self._act(mma, name + ' mass', lambda _=False, k=kind, p=params: self.add_mass(k, p))
            a.setIcon(prim_icon(kind))
        mma.addSeparator()
        self._act(mma, 'Chain (capsules laid click by click; Esc ends)...', self._chain)
        mc = m.addMenu('Cutter (cut with a shape)')
        for kind, name, params in PRIMS:
            if kind == 'ellipsoid':
                continue
            a = self._act(mc, name + ' cutter', lambda _=False, k=kind, p=params: self.create_cutter(k, p))
            a.setIcon(prim_icon(kind))

        m = mb.addMenu('&Sculpt')
        self.tool_actions = {}               # fs-shell 11-11: the tools' key actions (rebindable in the Keys window)
        # fs-shell-4 11-14 (Swell / Settle's row showed NO key): a registry brush row that runs the
        # engine's timed brushes (toolreg.brush_live) takes the registry's own key (Swell / Settle I, Flatten T; Smooth S,
        # Pencil L, Mask B, Blend Shift+B as they go live); the legacy brush that held the letter (Swell to contact,
        # Flatten) keeps its Sculpt menu entry, unbound (rebindable in the Keys window). Before, I and T still chose the
        # legacy brushes after the rows went live
        self.live_reg_keys = {}
        if APPROVED_LAYOUT:
            from . import toolreg as _TRk
            _R = _TRk.registry()
            for sp_ in (list(_R.BY_KEY.values()) if _R is not None else []):
                if getattr(sp_, 'run', '') == 'brush' and sp_.key_hint and _TRk.brush_live(sp_.key):
                    self.live_reg_keys['reg:' + sp_.key] = (sp_.name, sp_.key_hint)
        _taken = {k_ for _n, k_ in self.live_reg_keys.values()}
        for key, name, sc in TOOLS:
            if APPROVED_LAYOUT and key in APPROVED_TOOL_KEYS:     # fs-shell-3 11-14: the approved keys (REG_KEYS)
                sc = APPROVED_TOOL_KEYS[key]
            if sc in _taken:
                sc = None
            self.tool_actions[key] = self._act(m, name, lambda _=False, k=key: self.tools.set_tool(k), sc)
        m.addSeparator()
        # fs-shell-5 11-14: the same keys step the tool-in-hand's three primary settings (a brush's
        # radius / strength / softness; another tool's own, e.g. Cut's length): ui/quick_settings.py, shown on the cursor
        from . import quick_settings as _QS
        self._act(m, 'Brush radius smaller', lambda: _QS.step(self, 0, -1), '[')
        self._act(m, 'Brush radius larger', lambda: _QS.step(self, 0, 1), ']')
        self._act(m, 'Brush strength lower', lambda: _QS.step(self, 1, -1), '-')
        self._act(m, 'Brush strength higher', lambda: _QS.step(self, 1, 1), ['=', '+'])
        # fs-shell-5 11-14 (hardness / softness were not findable): Shift+[ softer, Shift+] harder, 10 %
        self._act(m, 'Brush softer (softness / hardness)', lambda: _QS.step(self, 2, -1), ['{', 'Shift+{', 'Shift+['])
        self._act(m, 'Brush harder (softness / hardness)', lambda: _QS.step(self, 2, 1), ['}', 'Shift+}', 'Shift+]'])
        m.addSeparator()
        for ax in ('x', 'y', 'z'):
            self._act(m, 'Mirror %s on / off' % ax.upper(),
                      lambda _=False, i='xyz'.index(ax): self.tools.sym[i].toggle(), 'Alt+%s' % ax.upper())
        self._act(m, 'Mirror in the world frame (x = 0 midline) on / off', lambda: self.tools.sym_world.toggle(),
                  'Alt+W')
        m.addSeparator()
        self._act(m, 'Clear mask (selected)', lambda: self.props._mask('clear'), 'Alt+M')
        self._act(m, 'Invert mask (selected)', lambda: self.props._mask('invert'), 'Ctrl+I')

        m = mb.addMenu('&Tools')
        mm = m.addMenu('Mutate selected node')
        from .panels import MUTATION_BUTTONS
        for kind, name, params in MUTATION_BUTTONS:
            self._act(mm, name, lambda _=False, k=kind, p=params: self.props._wrap(k, p))
        self._act(m, 'Merge selected (sculpt kept; last settings)', lambda: self.props.merge('union'), 'Ctrl+J')
        # fs-shell-6 11-14 (the eye workflow; from fs-shell-5's t73): a group selects and moves as one;
        # a pair across her midline is named by her side as it is grouped
        self._act(m, 'Group the selected (they select and move together)', self.group_selected, 'Ctrl+G')
        self._act(m, 'Ungroup the selected', self.ungroup_selected, 'Ctrl+Shift+G')
        self._act(m, 'Merge selected...', self.merge_ask, 'Ctrl+Shift+J')
        mm = m.addMenu('Combine selected')
        for kind, name in (('union', 'Merge'), ('subtract', 'Subtract'), ('intersect', 'Intersect')):
            self._act(mm, name, lambda _=False, k=kind: self.props._boolean(k))
        self._act(mm, 'Merge smooth (blend k)', lambda: self.props._boolean('union', force_smooth=True))
        m.addSeparator()
        self._act(m, 'Merge with protected parts...', self.merge_protected_ask, 'Ctrl+Alt+J')
        self._act(m, 'Protect the masked part as...', self.protect_masked_ask)
        self._act(m, 'How to mark her protected parts', self.protect_hint)
        self._act(m, 'Unprotect a part to sculpt it...', lambda: self.lock_part_ask(False))
        self._act(m, 'Protect a part again (relock)...', lambda: self.lock_part_ask(True))
        self._act(m, 'Check overlaps with protected parts (new overlaps only)', lambda: self.engine.post('check_overlaps'))
        self._act(m, "Set a part's root from its mask paint...", self.set_root_ask)
        self._act(m, 'Verify op-log replay (hash)', lambda: self.engine.post('verify_replay'))
        self._act(m, 'Consolidate history (keep the last 50 undo steps)', self.consolidate_ask)
        m.addSeparator()
        # fs-shell 11-11 (design D1, queue item 5): measures read exactly off the field, pinned in the scene
        mm_ = m.addMenu('Measure')
        self._act(mm_, 'Measure girth (click a form; Shift: vertical, Ctrl: facing the view)',
                  lambda: self._measure('girth'))
        self._act(mm_, 'Measure with a tape (two clicks: straight and along the surface)', lambda: self._measure('tape'))
        self._act(mm_, 'Volume and weight of the selected object...', self._measure_volume)
        self._act(mm_, 'Girth profile of the selected object (along an axis; landmarks)...',
                  lambda: __import__('fieldstudio.ui.profile_ui', fromlist=['x']).open_panel(self))
        self._act(mm_, 'Clear all measures', lambda: self.measures.clear())
        m.addSeparator()
        self.reg_key_actions = {}             # fs-shell-3 11-14: the registry tools' keys (the column shows them)
        if APPROVED_LAYOUT:
            mt_ = m.addMenu('The toolset')
            for name_, run_, key_ in REG_KEYS:
                self.reg_key_actions[run_] = self._act(mt_, name_, lambda _=False, r=run_: self._reg_key(r), key_)
            for run_, (name_, key_) in self.live_reg_keys.items():          # fs-shell-4 11-14: the live brush rows
                self.reg_key_actions[run_] = self._act(mt_, name_, lambda _=False, r=run_: self._reg_key(r), key_)
            # decided 11-14, open to change: the Keys window says so beside the key
            self.reg_key_actions['reg:add'].setProperty(
                'keys_note', 'Shift+A, not the mock\'s A: A held + a click is select-inside\'s way back out. '
                             'Chosen 11-14; yours to change (right-click: Rebind)')
            m.addSeparator()
        self.menu_tools = m                   # tools.register adds Pencil / Cut / Splice / Fill / layer smooth here

        m = mb.addMenu('&View')
        # fs-shell 11-11 (design D1): the scene's unit for typed values, snapping and read-outs (saved with the scene)
        from PySide6.QtGui import QActionGroup as _QAG
        from .units import SCENE_UNITS
        mu = m.addMenu('Units')
        self.units_group = _QAG(self)
        self.units_actions = {}
        for u_ in SCENE_UNITS:
            a_ = mu.addAction({'mm': 'Millimetres (mm)', 'cm': 'Centimetres (cm)', 'm': 'Metres (m)',
                               'in': 'Inches (in)', 'ft-in': "Feet and inches (5' 6\")"}[u_])
            a_.setCheckable(True)
            a_.triggered.connect(lambda _=False, uu=u_: self.set_units(uu))
            self.units_group.addAction(a_)
            self.units_actions[u_] = a_
        self.units = str(app_settings().value('units_default', 'mm') or 'mm')
        if self.units not in SCENE_UNITS:
            self.units = 'mm'
        self.units_actions[self.units].setChecked(True)
        # fs-shell 11-11 (design D2): the grids, each a toggle, in the scene's unit
        self.dims_act = m.addAction('Dimensions of the selected object (W x H x D)')
        self.dims_act.setCheckable(True)
        self.dims_act.toggled.connect(lambda on: self.measures.set_dims(on))
        self.dims_world_act = m.addAction('Dimensions: world extents (the space it takes up in the scene)')
        self.dims_world_act.setCheckable(True)           # fs-shell 11-11: own size by default
        self.dims_world_act.toggled.connect(lambda on: self.measures.set_dims_world(on))
        mg = m.addMenu('Grid')
        self._act(mg, 'Show grid (on / off)', self.toggle_grid, 'Alt+G')
        self.grid_floor_act = mg.addAction('Floor grid (perspective views)')
        self.grid_floor_act.setCheckable(True)
        self.grid_floor_act.toggled.connect(lambda on: self.set_grid(floor=on))
        self.grid_view_act = mg.addAction('View grid (front / side / top views)')
        self.grid_view_act.setCheckable(True)
        self.grid_view_act.toggled.connect(lambda on: self.set_grid(view=on))
        self.grid_contour_act = mg.addAction('Surface contours on the model (height)')
        self.grid_contour_act.setCheckable(True)
        self.grid_contour_act.toggled.connect(lambda on: self.set_grid(contours=on))
        self.grid_spacing_menu = mg.addMenu('Spacing')
        self.grid_spacing_menu.aboutToShow.connect(self._fill_grid_spacing)
        m.addSeparator()
        # fs-shell 11-13 (Tier 0 part 4, what makes the work effective): the views needed (Clay, the
        # sculpting view; Look, real material and light; Silhouette, flat, against the references; X-ray, see-through
        # layers in Clay and Look); the debug views in their own submenu, out of the way
        for mode, name, keys in (('matcap', 'Clay (sculpting: matcap and cavity)', 'Shift+1'),
                                 ('shaded', 'Look (the real material and light, refining as you work)', 'Shift+2'),
                                 ('silhouette', 'Silhouette (flat, to check against the references)', 'Shift+3')):
            self._act(m, name, lambda _=False, md=mode: self.set_mode(md), keys)
        self._act(m, 'X-ray on / off (see through to the layers inside; Clay and Look)', self.toggle_xray, 'Alt+Q')
        md_ = m.addMenu('Debug views')
        for mode, name in (('normal', 'Normals'), ('slice', 'Field slice'),
                           ('curvature', 'Curvature (convex warm, concave cool)')):
            self._act(md_, name, lambda _=False, md=mode: self.set_mode(md))
        mm = m.addMenu('Matcap and cavity')                # fs-shell 11-11 (chunk 9): the card's view modes
        self.vm_settings = {'matcap': 'clay', 'cavity': 0.6}
        for kind in ('clay', 'red wax', 'grey'):
            a_ = mm.addAction('Matcap: %s' % kind)
            a_.triggered.connect(lambda _=False, k=kind: self._vm_set(matcap=k))
        self.cavity_act = mm.addAction('Cavity shading (darken concave forms)')
        self.cavity_act.setCheckable(True)
        self.cavity_act.setChecked(True)
        self.cavity_act.toggled.connect(lambda on: self._vm_set(cavity=0.6 if on else 0.0))
        m.addSeparator()
        for key, name in ((1, 'Front'), (3, 'Right'), (7, 'Top'), (5, 'Three-quarter'), (9, 'Back'), (4, 'Left'),
                          (2, 'Bottom')):
            self._act(m, '%s view' % name, lambda _=False, k=key: self.preset(k), str(key))
        self._act(m, 'Frame selection', self.frame_selection, ['.', 'Home'])   # hand1: F is the radius drag now
        self._act(m, 'Frame all', lambda: self.engine.post('frame', []))
        self._act(m, 'Action menu at the mouse (also right-click)', self.context_at_mouse, ['Shift+F10', 'Menu'])
        m.addSeparator()
        # 11-10: navigation for close work (on by default, remembered): orbit about the surface under the cursor,
        # zoom toward the cursor
        cfg = app_settings()
        oc = str(cfg.value('orbit_cursor', 'true')).lower() != 'false'
        zc = str(cfg.value('zoom_cursor', 'true')).lower() != 'false'
        self.viewport.orbit_cursor, self.viewport.zoom_cursor = oc, zc

        def _nav(key, on):
            setattr(self.viewport, key, bool(on))
            app_settings().setValue(key, 'true' if on else 'false')
        self.act_orbit_cursor = self._act(m, 'Orbit around the point under the cursor', lambda on: _nav('orbit_cursor', on),
                                          checkable=True)
        self.act_orbit_cursor.setChecked(oc)
        self.act_zoom_cursor = self._act(m, 'Zoom toward the cursor', lambda on: _nav('zoom_cursor', on), checkable=True)
        self.act_zoom_cursor.setChecked(zc)
        m.addSeparator()
        self.act_refs = self._act(m, 'Reference planes visible', lambda on: self.refpanel.all_visible.setChecked(on),
                                  'Alt+R', checkable=True)
        self.act_refs.setChecked(True)
        self.act_lens = self._act(m, 'Reference lens (5 deg, near orthographic)',
                                  lambda on: self.refpanel.set_lens(on), 'Alt+L', checkable=True)
        # fs-shell (design 4.3): the true orthographic view (parallel rays); with a rig, axis views use it by themselves
        self.act_ortho = self._act(m, 'Orthographic (parallel rays)', lambda on: self._set_ortho(on), 'Alt+O',
                                   checkable=True)
        m.addSeparator()
        # r26: the GPU viewport's light: a sculpting key that follows the view (default) or the HDRI's own sun
        self.act_keylight = self._act(m, 'Sculpting key light (follows the view; off = the HDRI sun)',
                                      lambda on: self.engine.post('gpu_light', 'key' if on else 'hdri'),
                                      checkable=True)
        self.act_keylight.setChecked(os.environ.get('FIELDSTUDIO_LIGHT', 'key') == 'key')
        # 11-08 night (deep recesses too dark to see): a sculpting visibility aid; review renders keep
        # the plain light
        self.act_headlight = self._act(m, 'Sculpt headlight (lights recesses from the view; no shadows)',
                                       lambda on: self.engine.post('gpu_aid', {'headlight': self._aid_level if on else 0.0}),
                                       checkable=True)
        self._aid_level = 0.6
        self.act_headlight.setChecked(True)
        self._act(m, 'Sculpt view light (headlight, cavity, exposure)...', self.view_light_dialog)
        pv = m.addMenu('Protected parts')
        from PySide6.QtGui import QActionGroup
        grp = QActionGroup(self)
        for key, text in (('shown', 'Shown'), ('ghost', 'Ghosted (see through, outlined)'), ('hidden', 'Hidden')):
            a = self._act(pv, text, lambda _=False, k=key: self.engine.post('parts_view', k), checkable=True)
            grp.addAction(a)
            a.setChecked(key == 'shown')
        self.act_overlaps = self._act(m, 'Show new overlaps in red (new overlaps only)',
                                      lambda on: self.engine.post('show_overlaps', on), checkable=True)
        self.act_overlaps.setChecked(True)
        self._ev = 0.0
        self._act(m, 'Brighter view (cycle +0 / +1 / +2 stops)', self.cycle_exposure, 'Shift+E')   # E is Stretch
        self.act_refstore = self._act(m, 'Published field beside the figure (view only)',
                                      lambda on: self.engine.post('gpu_ref_store', self.published_field(), on),
                                      checkable=True)

        # fs-store 11-12 (there was no findable way to render hair or view full-resolution renders): a still of
        # the view being looked at, through the real renderer with skin, iris and hair, in its own full-resolution viewer
        m = mb.addMenu('&Render')
        self._act(m, "This view, converged at full resolution (the window's own look)", self._converge_view, 'Shift+F12')
        self._act(m, 'Open a still...', self.open_still)
        m = mb.addMenu('Hel&p')   # fs-shell 11-12 (fs-bench-full F1): Alt+H is Show all, so Help's mnemonic is P
        self._act(m, 'Keys window (every key, the current tool, a keyboard map)',
                  lambda: self.keys_window.toggle(), 'F1')
        self._act(m, 'Search tools and actions (Space)', lambda: self.palette.open_at())
        self._act(m, 'Quick start', self.show_quick_start)
        self._act(m, 'What to try', self.show_what_to_try, 'Shift+F1')     # hand2f: F2 is Rename now
        self._act(m, 'Controls (short list)', self.show_help)

    def cycle_exposure(self):
        """Shift+E: the view's exposure +0 -> +1 -> +2 stops -> +0 (the sculpting aid; review renders unchanged)"""
        self._ev = {0.0: 1.0, 1.0: 2.0}.get(self._ev, 0.0)
        self.engine.post('gpu_aid', {'ev': self._ev})
        self.statusBar().showMessage('View exposure %+d stop(s) (Shift+E cycles)' % int(self._ev), 3000)

    def view_light_dialog(self):
        from PySide6.QtWidgets import QDialog, QFormLayout, QSlider, QDialogButtonBox
        d = QDialog(self)
        d.setWindowTitle('Sculpt view light')
        f = QFormLayout(d)
        sl = {}
        for key, label, lo, hi, val in (('headlight', 'Headlight', 0, 150, int(round(self._aid_level * 100))),
                                        ('cavity', 'Cavity / occlusion', 0, 100, int(round(getattr(self, '_cav', 1.0) * 100))),
                                        ('ev', 'Exposure (stops x 10)', -20, 30, int(round(self._ev * 10)))):
            s_ = QSlider(Qt.Horizontal)
            s_.setRange(lo, hi)
            s_.setValue(val)
            f.addRow(label, s_)
            sl[key] = s_

        def apply(_=None):
            self._aid_level = sl['headlight'].value() / 100.0
            self._cav = sl['cavity'].value() / 100.0
            self._ev = sl['ev'].value() / 10.0
            self.act_headlight.blockSignals(True)
            self.act_headlight.setChecked(self._aid_level > 0)
            self.act_headlight.blockSignals(False)
            self.engine.post('gpu_aid', {'headlight': self._aid_level, 'cavity': self._cav, 'ev': self._ev})
        for s_ in sl.values():
            s_.valueChanged.connect(apply)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        d.show()

    def _top_bars(self):
        tb = QToolBar('Workspace')
        tb.setMovable(False)
        self.ws = QTabBar()
        for name in ('Model', 'Sculpt', 'Inspect'):
            self.ws.addTab(name)
        self.ws.setToolTip('Model: select and build from primitives, booleans and mutations.\n'
                           'Sculpt: brushes on the sculpt layer.\nInspect: the field slice (F < 0 inside).')
        self.ws.currentChanged.connect(self.on_workspace)
        # fs-shell 11-12 ('Inspect' clipped to 'Ir' on the card, only two tabs on the CPU): the tabs keep their
        # full width; when the row is short the Create buttons drop their names (icons + tooltips) instead
        self.ws.setUsesScrollButtons(False)
        self.ws.setElideMode(Qt.ElideNone)
        self.ws.setExpanding(False)
        self.ws.setMinimumWidth(self.ws.sizeHint().width())
        tb.addWidget(self.ws)
        # fs-shell: the primitives sit in this row (one row fewer above the view)
        lb = QLabel('    Create ')
        lb.setObjectName('dim')
        tb.addWidget(lb)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for kind, name, params in PRIMS:
            a = QAction(prim_icon(kind), name, self)
            a.setToolTip('Add a %s at the view centre (an analytic primitive node, editable later; its lattice from its '
                         'size, Mirror X on the midline)' % name.lower())
            a.triggered.connect(lambda _=False, k=kind, p=params: self.create(k, p))
            tb.addAction(a)
        sp = QWidget()
        sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(sp)
        tb.addWidget(self.telemetry)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.top_bar = tb
        self._fit_timer = QTimer(self)
        self._fit_timer.setInterval(1500)
        self._fit_timer.timeout.connect(self._fit_top_bar)        # the telemetry's width changes with the device
        self._fit_timer.start()

    def _fit_top_bar(self):
        """fs-shell 11-12: the Create buttons with their names when the row has room, icons only when it has not, so
        the workspace tabs and the telemetry always show whole"""
        tb = getattr(self, 'top_bar', None)
        if tb is None:
            return
        # the tabs' own size once the style has padded them (set at build, before the theme's padding, it was 175 px
        # for 207 px of tabs: the row kept the Create names and squeezed 'Inspect' to 'Inspec')
        need = self.ws.sizeHint().width()
        if self.ws.minimumWidth() != need:
            self.ws.setMinimumWidth(need)
        fm = tb.fontMetrics()
        names = sum(fm.horizontalAdvance(n) + 8 for _k, n, _p in PRIMS)
        icons = 30 * len(PRIMS)
        tel = max(self.telemetry.sizeHint().width(), self.telemetry.minimumWidth())
        fixed = self.ws.minimumWidth() + tel + 120
        want = Qt.ToolButtonTextBesideIcon if fixed + icons + names <= tb.width() else Qt.ToolButtonIconOnly
        if tb.toolButtonStyle() != want:
            tb.setToolButtonStyle(want)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        try:
            self._fit_top_bar()
        except Exception:
            pass

    def _viewport_bar(self):
        bar = QWidget()
        bar.setStyleSheet('background: #1d2026;')
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(6)

        def combo(label, items, fn, tip):
            lb = QLabel(label)
            lb.setObjectName('dim')
            c = QComboBox()
            for text, data in items:
                c.addItem(text, data)
            c.currentIndexChanged.connect(lambda i: fn(c.itemData(i)))
            c.setToolTip(tip)
            lay.addWidget(lb)
            lay.addWidget(c)
            return c
        self.mode_combo = combo('View', [('Clay', 'matcap'), ('Look', 'shaded'), ('Silhouette', 'silhouette')],
                                lambda md: md and self.set_mode(md),
                                'Clay: the sculpting view (matcap and cavity, Shift+1). Look: the real material and '
                                'light, refining as you work (Shift+2). Silhouette: flat, to check the outline against '
                                'the references (Shift+3). The debug views (Normals, Field slice, Curvature) are in '
                                'View > Debug views.')
        self.mode_combo.setPlaceholderText('Debug view')
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1)              # Look, as the viewport starts
        self.mode_combo.blockSignals(False)
        # fs-shell 11-13: X-ray: how see-through the forms are in Clay and Look (0 = solid); Alt+Q toggles it
        from PySide6.QtWidgets import QSlider
        xl = QLabel('X-ray')
        xl.setObjectName('dim')
        self.xray_slider = QSlider(Qt.Horizontal)
        self.xray_slider.setRange(0, 100)
        self.xray_slider.setFixedWidth(110)
        self.xray_slider.setToolTip('X-ray: see through the forms to the layers inside (Clay and Look; Alt+Q '
                                    'toggles). Per-object opacity: Properties > Opacity.')
        self.xray_val = QLabel('off')
        self.xray_val.setObjectName('dim')
        self.xray_val.setFixedWidth(34)
        self.xray_slider.valueChanged.connect(self.set_xray)
        self._xray_last = 50
        lay.addWidget(xl)
        lay.addWidget(self.xray_slider)
        lay.addWidget(self.xray_val)
        self.q_combo = combo('Quality', [('Auto', 'auto'), ('Preview', 0), ('Normal', 1),
                                         ('High', 2)], self.set_quality,
                             'Auto renders half resolution while you move, then the high quality frame when idle')
        self.res_combo = combo('Resolution', [('100 %', 1.0), ('75 %', 0.75), ('50 %', 0.5)], self.set_res,
                               'Internal render resolution (lower = faster on a busy CPU)')
        lay.addSpacing(12)
        # fs-shell: the favourites strip (tools and actions dragged here; Front / Right / Top / 3/4 / Frame by default)
        self.fav_slot = QHBoxLayout()
        self.fav_slot.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self.fav_slot, 1)
        b = QPushButton('Screenshot')
        b.setToolTip('Ctrl+S: save the viewport PNG under engine/fieldstudio/screens/')
        b.clicked.connect(lambda: self.screenshot())
        lay.addWidget(b)
        return bar

    # ---- engine events ----------------------------------------------------------------------------------------------
    def on_ready(self, secs):
        self.viewport.warm = True
        from PySide6.QtCore import QTimer as _QT
        _QT.singleShot(500, self.offer_recovery)         # hand2f: crash recovery (interactive runs only)
        self.status.showMessage('Field kernels ready in %.1f s (CPU, %d threads). Create a primitive or sculpt.'
                                % (secs, self.engine.threads), 10000)
        self.viewport.request(False)

    def on_failed(self, text):
        self.errors.append(text)
        print('[engine] ' + text, file=sys.stderr)
        self.status.showMessage(text.splitlines()[0][:200], 15000)

    def on_snapshot(self, snap):
        prev = self.snap
        self.snap = snap
        g_ = getattr(self.viewport, 'grid', None)       # fs-shell 11-11 (design D2): the floor under the scene
        if g_ is not None and g_.floor and g_.scene_changed(snap):
            self.viewport.refs_changed()
        if snap.get('adopted') and prev is not None:
            # hand2b: an opened project's undo history arrived (the same state, new step identities): a window that
            # was clean (or autosaved) at that state stays so
            if self._marker(prev) == self.saved_marker:
                self.saved_marker = self._marker(snap)
            if self._marker(prev) == self.autosave_marker:
                self.autosave_marker = self._marker(snap)
        if self._mark_clean_at is not None and len(snap.get('log_ids', [])) >= self._mark_clean_at:
            self._mark_clean_at = None     # hand2f: a new scene / preset / opened project starts clean
            self.saved_marker = self._marker(snap)
            self.autosave_marker = self.saved_marker
        self.props.update_snapshot(snap)
        self.oplog.update_snapshot(snap)
        objs = snap['objects']
        self.viewport.objects_info = {'objects': len(objs), 'nodes': sum(o['nodes'] for o in objs),
                                      'bricks': sum(o['bricks'] for o in objs),
                                      'h': (objs[0]['sculpt_h'] * 1000) if objs else 0.5}
        self.viewport.transforms = {o['id']: o['transform'] for o in objs}
        self.viewport.groups = {o['id']: o.get('group') for o in objs}        # fs-shell-6: a group drags as one
        self.viewport.stretches = {o['id']: list(o['base']['params'].get('scale', [1.0, 1.0, 1.0]))
                                   for o in objs if o['base'].get('kind') == 'transform'
                                   and o['base'].get('name') == 'object scale'}
        self.viewport.shears = {o['id']: list(o['base']['params'].get('shear', [0.0] * 6))
                                for o in objs if o['base'].get('kind') == 'transform'
                                and o['base'].get('name') == 'object scale'}
        self.telemetry.set('field', '%.0f MB' % (snap['field_mem'] / 2 ** 20))
        self.telemetry.set('undo', '%.1f MB' % (snap['hist_mem'] / 2 ** 20))
        self.telemetry.set('L', '%.3f' % snap['L'])
        self.viewport.update()

    def on_stats(self, s):
        self.refpanel.check_auto_lens()
        self.telemetry.set('fps', '%d' % s['fps'])
        self.telemetry.set('ms', '%.0f ms' % s['ms'])
        st = s['stats']
        self.telemetry.set('cpu', 'q%d %dx%d' % (st['quality'], st['internal_res'][0], st['internal_res'][1]))

    def on_bounds(self, b):
        lo, hi = b
        # hand2b: an exact fit of the box corners at the viewport's aspect (the box already carries the 5 % guard
        # margins on every side, so margin 1.0 leaves a small border)
        vp = self.viewport
        self.viewport.cam.frame(lo, hi, margin=1.0, aspect=max(vp.width(), 1) / float(max(vp.height(), 1)),
                                sphere_margin=0.8)
        self.viewport.request(False)

    def toggle_grid(self):
        """Show grid (Alt+G): every grid off, or back to the last set shown (the floor and view grids at first)"""
        g = self.viewport.grid
        c = getattr(self.viewport, 'contours', None)
        now = (g.floor, g.view, bool(c and c.on))
        if any(now):
            self._grid_last = now
            self.set_grid(floor=False, view=False, contours=False)
        else:
            f_, v_, c_ = getattr(self, '_grid_last', (True, True, False))
            self.set_grid(floor=f_, view=v_, contours=c_)

    def set_grid(self, floor=None, view=None, spacing='keep', remember=True, contours=None):
        """the grids (design D2): floor / view toggles; spacing None = adaptive, else metres"""
        g = getattr(self.viewport, 'grid', None)
        if g is None:
            return
        if floor is not None:
            g.floor = bool(floor)
            if g.floor and self.snap is not None:
                g.scene_changed(self.snap)
        if view is not None:
            g.view = bool(view)
        if spacing != 'keep':
            g.spacing = None if spacing is None else float(spacing)
        c_ = getattr(self.viewport, 'contours', None)
        if contours is not None and c_ is not None:
            c_.on = bool(contours)
            a_ = getattr(self, 'grid_contour_act', None)
            if a_ is not None and a_.isChecked() != c_.on:
                a_.blockSignals(True)
                a_.setChecked(c_.on)
                a_.blockSignals(False)
        for a_, on in ((getattr(self, 'grid_floor_act', None), g.floor), (getattr(self, 'grid_view_act', None), g.view)):
            if a_ is not None and a_.isChecked() != on:
                a_.blockSignals(True)
                a_.setChecked(on)
                a_.blockSignals(False)
        if remember:
            cfg = app_settings()
            cfg.setValue('grid_floor', 'true' if g.floor else 'false')
            cfg.setValue('grid_view', 'true' if g.view else 'false')
            cfg.setValue('grid_spacing', '' if g.spacing is None else repr(g.spacing))
            if c_ is not None:
                cfg.setValue('grid_contours', 'true' if c_.on else 'false')
        self.viewport.refs_changed()

    def _fill_grid_spacing(self):
        from ..render import grids as GR
        from .units import format_length
        mm = self.grid_spacing_menu
        mm.clear()
        g = self.viewport.grid
        a_ = mm.addAction('Adaptive (finer as you zoom)')
        a_.setCheckable(True)
        a_.setChecked(g.spacing is None)
        a_.triggered.connect(lambda _=False: self.set_grid(spacing=None))
        for s_ in GR.ladder(self.units):
            a_ = mm.addAction(format_length(s_, self.units).replace('.0 mm', ' mm'))
            a_.setCheckable(True)
            a_.setChecked(g.spacing is not None and abs(g.spacing - s_) < 1e-12)
            a_.triggered.connect(lambda _=False, ss=s_: self.set_grid(spacing=ss))

    def set_units(self, u, remember=True):
        """the scene's unit (design D1): typed values, snapping, read-outs; new scenes start in the last one chosen"""
        from .units import SCENE_UNITS
        if u not in SCENE_UNITS:
            return
        self.units = u
        g = getattr(self.viewport, 'grid', None)
        if g is not None and g.spacing is not None:
            from ..render import grids as GR
            if not any(abs(g.spacing - s_) < 1e-12 for s_ in GR.ladder(u)):
                g.spacing = None                    # a chosen spacing of the other system: adaptive again
        a_ = getattr(self, 'units_actions', {}).get(u)
        if a_ is not None and not a_.isChecked():
            a_.setChecked(True)
        if remember:
            app_settings().setValue('units_default', u)
        cb = getattr(self, 'context_bar', None)
        if cb is not None:
            cb.sync()
        self.viewport.update()

    def on_project_loaded(self, extra):
        self._mark_clean_at = 0
        if (extra or {}).get('units'):
            self.set_units(extra['units'], remember=False)
        if getattr(self, 'measures', None) is not None:
            self.measures.load((extra or {}).get('measures'))
        if getattr(self, 'marks', None) is not None:
            self.marks.load((extra or {}).get('marks'))
        cam = (extra or {}).get('camera')
        if cam:
            self.viewport.cam = Camera.from_dict(cam)
        refs = (extra or {}).get('refs')
        # hand1b: the project's references replace the ones on screen; a project without references clears them
        missing = self.refpanel.apply_dict(refs or {}, base=self.project_path)
        if missing:
            self.status.showMessage('Reference image(s) not found: %s' % ', '.join(missing), 15000)
        self.refpanel.restore_lens((extra or {}).get('view', {}))
        pend = getattr(self, 'pending_refs', None)
        if pend:                               # the command line's --ref-front / --ref-side override the project's
            self.pending_refs = None
            self.load_refs(*pend)
        self.viewport.request(False)

    def tick(self):
        ws, peak = process_memory()
        self.telemetry.set('ram', '%.0f MB' % ws)
        self.peak_mb = max(getattr(self, 'peak_mb', 0.0), peak)
        if self.viewport.frame_times and time.perf_counter() - self.viewport.frame_times[-1] > 1.0:
            self.telemetry.set('fps', '0 (idle)')

    # ---- UI actions ----------------------------------------------------------------------------------------------------
    def on_viewport_select(self, oid, additive):
        order = list(self.props.order)
        # fs-shell-6 11-14 (fs-shell-5's t73): a click in the view on a member of a group brings its whole group (the
        # Scene list still selects one object alone, to edit it inside the group)
        objs = (self.snap or {}).get('objects', [])
        g = next((o.get('group') for o in objs if o.get('id') == oid), None)
        mem = [o['id'] for o in objs if g is not None and o.get('group') == g] if oid >= 0 else []
        mem = mem or ([oid] if oid >= 0 else [])
        if oid < 0:
            if not additive:
                order = []
        elif additive:
            order = [o for o in order if o not in mem] if oid in order else order + [o for o in mem if o not in order]
        else:
            order = [oid] + [o for o in mem if o != oid]
        self.props.set_selection(order)

    def update_hint(self):
        """V1.3: the status bar's right side says what each mouse button does NOW (tool, field tool or select)."""
        vp = self.viewport
        nav = 'Middle or Alt+Left-drag: orbit   Shift+Middle / Right-drag: pan   Right-click: actions   Wheel: zoom   .: frame'
        m = vp.tool_mode
        if m is not None:
            k = type(m).__name__
            left = {'PencilMode': 'Left-drag across a form: draw the pencil stroke',
                    'CutMode': 'Left-drag: draw the cut outline (it closes itself)',
                    'PickPairMode': 'Left-click: face A, then face B'}.get(k, 'Left-drag on a form: smooth the band')
            text = '%s   Alt+Left-drag: orbit   %s   Esc: leave the tool' % (left, nav)
        elif vp.tool == 'select':
            text = ('Left-click: select (Shift adds)   %s / %s / %s: move / rotate / scale the selection   Left-drag: orbit   '
                    '%s   D C V ...: pick a brush' % (self._key_of('move', 'W'), self._key_of('rotate', 'R'),
                                                      self._key_of('scale', 'Y'), nav))
        elif vp.tool in ('move', 'rotate', 'scale'):
            text = ('Left-drag: %s the object (handles: one axis; rings below: bend / twist / taper / inflate)   '
                    'while dragging: X Y Z lock, type a number, Ctrl snaps, Enter / Esc   the tool key again: type a value, '
                    'no drag   Alt+Left: orbit   %s'
                    % ({'move': 'move', 'rotate': 'rotate', 'scale': 'scale'}[vp.tool], nav))
        else:
            from . import toolreg as _TRn                 # fs-shell-4 11-14: Swell / Settle / Flatten, not 'ft_swell'
            name = vp.TOOL_NAMES.get(vp.tool) or _TRn.FT_NAMES.get(vp.tool, vp.tool)
            inv = {'draw': ' (Ctrl: carve)', 'carve': ' (Ctrl: draw)', 'mask': ' (Ctrl: erase)', 'inflate': ' (Ctrl: deflate)',
                   'clay': ' (Ctrl: dig)'}.get(vp.tool, ' (Ctrl: invert)')
            text = ('Left-drag: %s%s (starts on the form, never orbits)   Shift: smooth   %s   '
                    'F-drag / [ ]: radius   Shift+F-drag / - =: strength (to 2)   %s: select'
                    % (name, inv, nav, self._key_of('select', 'Q')))
        if os.environ.get('FS_LAYOUT', '') != 'classic':   # fs-shell-2 11-14 (design 3.5): the keys read as a list
            text = '  ·  '.join(p.strip() for p in text.split('   ') if p.strip())
        self.hint.setText(text)

    def _key_of(self, tool, fallback=''):
        from .keymap import tool_key
        return tool_key(self, tool, fallback) or 'no key'

    def on_axis(self, i):
        self.viewport.xform_axis = self.tools.axis.itemData(i)
        self.viewport.update()

    def on_tool(self, key):
        # hand1: any tool pick (button, key, menu or workspace tab) leaves a field tool first; before, the field tool
        # kept the left mouse button while the panel showed the new brush (diagnosis root cause 1)
        if self.viewport.tool_mode is not None:
            self.set_tool_mode(None)
        if self.viewport.adjust is not None:
            self.viewport.end_adjust()
        if key != 'select' and self.viewport.mode == 'slice':
            self.set_mode('shaded')              # a brush on the field slice finds no surface: back to shaded
        self.viewport.tool = key
        self.viewport.brush = self.tools.brush()
        self.update_hint()
        if self.viewport.frame is not None:
            self.viewport._compose()           # 11-10: the selection outline shows only with Select / Move / Rotate / Scale
        self.viewport.update()
        self.ws.blockSignals(True)
        if key in ('select', 'move', 'rotate', 'scale'):
            if self.ws.currentIndex() == 1:
                self.ws.setCurrentIndex(0)
        else:
            self.ws.setCurrentIndex(1)
        self.ws.blockSignals(False)

    def on_brush(self, settings):
        self.viewport.brush = self.tools.brush()
        self.viewport.update()

    def on_show_mask(self, on):
        self.viewport.show_mask = on
        self.viewport.request(False)

    def on_workspace(self, i):
        if i == 0:
            self.tools.set_tool('select')
            if self.viewport.mode == 'slice':
                self.set_mode('shaded')
        elif i == 1:
            self.tools.set_tool(self.tools.last_sculpt)
            if self.viewport.mode == 'slice':
                self.set_mode('shaded')
        else:
            self.set_mode('slice')

    def _vm_set(self, **kw):
        """fs-shell 11-11: the matcap kind / cavity strength the card's view modes use (gpu/viewmodes.py)"""
        self.vm_settings.update(kw)
        self.engine.vm_settings = dict(self.vm_settings)
        if self.viewport.mode not in ('matcap', 'curvature'):
            self.set_mode('matcap')
        self.viewport.request(False)

    def set_mode(self, mode):
        """fs-shell 11-13: the views (Clay = 'matcap', Look = 'shaded', Silhouette) in the View list; a debug view
        (Normals, Field slice, Curvature) leaves the list blank ('Debug view')"""
        self.viewport.mode = mode
        idx = self.mode_combo.findData(mode)
        if self.mode_combo.currentIndex() != idx:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)
        if hasattr(self, 'xray_slider'):
            self.xray_slider.setEnabled(mode in ('shaded', 'matcap'))
        self.viewport.request(False)

    def set_xray(self, pct):
        """fs-shell 11-13: X-ray amount (0-100 %; the card composite stops at 95 %) for Clay and Look"""
        self.viewport.xray = float(pct) / 100.0
        self.xray_val.setText('off' if pct == 0 else '%d%%' % pct)
        if pct > 0:
            self._xray_last = pct
            if self.viewport.mode not in ('shaded', 'matcap'):
                self.set_mode('matcap')
        self.viewport.request(False)

    def toggle_xray(self):
        self.xray_slider.setValue(0 if self.xray_slider.value() > 0 else (self._xray_last or 50))

    def set_quality(self, q):
        self.viewport.quality = q
        self.viewport.request(False)

    def set_res(self, s):
        self.viewport.res_scale = float(s)
        self.viewport.request(False)

    AXIS_KEYS = (1, 2, 3, 4, 7, 9)

    def preset(self, key):
        self.viewport.cam.preset(key)
        # fs-shell (design 4.3 / 5.3): with a Reference Rig loaded, the axis views go to the near-orthographic lens
        # (a side press at 35 deg landed 48.7 mm off the midline); orbiting off the axis switches it back
        rp = self.refpanel
        if key in self.AXIS_KEYS and rp.rig_active():
            rp.axis_view()                    # true orthographic where the renderer draws it, else the 5 deg lens
        self.viewport.request(False)

    def _set_ortho(self, on):
        got = self.refpanel.set_ortho(on)
        self.refpanel.ortho_auto = False
        if on and not got:
            self._sync_action(self.act_ortho, False)
            self.status.showMessage('The orthographic view needs the card tracer\'s parallel rays (pending in '
                                    'gpu/trace.cu); the 5 deg lens (Alt+L) is the nearest until then.', 10000)

    def frame_selection(self):
        self.engine.post('frame', list(self.props.order))

    def create(self, kind, params):
        t = [round(float(v), 4) for v in self.viewport.cam.target]
        col = PALETTE[self.colour_i % len(PALETTE)]
        self.colour_i += 1
        # fs-shell (design 4.5 / 5.3): the lattice from the object's size (h = size / 120, at least 0.25 mm), so its
        # brushes reach 80 h = two thirds of its size (Add > Sphere capped brushes at 40 mm on a 0.5 mm lattice), and
        # Mirror X on for an object centred on the midline
        h = max(0.00025, round(prim_size(kind, params) / 120.0, 6))
        self.engine.post('add', {'kind': kind, 'params': dict(params), 'translate': t, 'h': h,
                                 'colour': {'albedo': col, 'variation': 0.0}})
        if abs(t[0]) < 0.0005:
            self.tools.set_mirror([True, False, False])
        if self.tools.tool != 'select':
            self.tools.set_tool('select')

    # ---- fs-shell 11-11 (design 4.18): cut with a shape -------------------------------------------------------
    CUTTER_COLOUR = [0.86, 0.24, 0.20]

    def _chain(self):
        from .chain_ui import start
        start(self)

    def add_mass(self, kind, params, scale=0.6):
        """Create > Mass: a live primitive (scaled to the job) smooth-joined into the selected object at the view's
        centre (an object of its own when nothing is selected); the Move tool comes up on it"""
        import copy as _copy
        import numpy as np
        from ..tools.masses import AddMass
        from ..core.ops import rot_matrix
        p = _copy.deepcopy(params)
        for k_, v_ in list(p.items()):
            if k_ in ('radius', 'height', 'major', 'minor', 'radius_bottom', 'radius_top', 'extent', 'thickness'):
                p[k_] = float(v_) * scale
            elif k_ in ('size', 'radii'):
                p[k_] = [float(x) * scale for x in v_]
        objs = {o['id']: o for o in (self.snap or {}).get('objects', [])}
        sel = [o for o in self.viewport.selected if o in objs]
        if not sel:
            self.create(kind, p)                  # the first mass is the object itself
            self.status.showMessage('A new object; Create > Mass again adds masses that blend into it.', 10000)
            self.tools.set_tool('move')
            return
        oid = sel[0]
        tr = objs[oid]['transform']
        Ro, to, so = rot_matrix(tr['rotate']), np.asarray(tr['translate'], float), float(tr['scale'])
        c = np.asarray(self.viewport.cam.target, float)
        local = (Ro.T @ (c - to)) / so
        self.engine.post('push', AddMass(oid, kind, p, translate=[float(v) for v in local]))
        self.tools.set_tool('move')
        self.status.showMessage('Mass added: drag it (the gizmo is on it), type a value, or set the blend on the bar '
                                'above the view. Click another mass to take it.', 12000)

    def create_cutter(self, kind, params, scale=0.5):
        """a cutter (a red primitive named 'Cutter (<kind>)') at the selected object's centre, half the primitive's
        default size; the objects selected now are what it will cut. The Move tool comes up so the gizmo places it."""
        import copy as _copy
        from ..tools.shapecut import is_cutter
        objs = {o['id']: o for o in (self.snap or {}).get('objects', [])}
        self.cut_targets = [o for o in self.viewport.selected if o in objs and not is_cutter(objs[o])]
        c = (self.viewport.transforms.get(self.cut_targets[0], {}).get('translate') if self.cut_targets else None)
        t = [round(float(v), 4) for v in (c if c is not None else self.viewport.cam.target)]
        p = _copy.deepcopy(params)
        for k_, v_ in list(p.items()):
            if k_ in ('radius', 'height', 'major', 'minor', 'radius_bottom', 'radius_top', 'extent', 'thickness'):
                p[k_] = float(v_) * scale
            elif k_ == 'size':
                p[k_] = [float(x) * scale for x in v_]
        h = max(0.00025, round(prim_size(kind, p) / 120.0, 6))
        self.engine.post('add', {'kind': kind, 'params': p, 'translate': t, 'h': h, 'name': 'Cutter (%s)' % kind,
                                 'colour': {'albedo': list(self.CUTTER_COLOUR), 'variation': 0.0}})
        self.tools.set_tool('move')
        self.status.showMessage('Cutter: place it with the gizmo (arrows, rings below to bend / twist), then Apply cut '
                                'on the bar above the view. It cuts %s.' % (
                                    ', '.join(objs[o]['name'] for o in self.cut_targets) or 'every object it touches'),
                                15000)

    def apply_cut(self, side='inside', result='delete', distance=0.03):
        """the selected cutter cuts its targets (those selected when it was made, else every other object), one step"""
        from ..tools.shapecut import ShapeCut, is_cutter
        objs = {o['id']: o for o in (self.snap or {}).get('objects', [])}
        cutters = [o for o in self.viewport.selected if o in objs and is_cutter(objs[o])]
        if not cutters:
            self.status.showMessage('Select a cutter first (Create > Cutter).', 8000)
            return False
        targets = [o for o in getattr(self, 'cut_targets', []) if o in objs and not is_cutter(objs[o])]
        if not targets:
            targets = [o for o in objs if not is_cutter(objs[o])]
        if not targets:
            self.status.showMessage('Nothing to cut.', 8000)
            return False
        self.engine.post('push', ShapeCut(targets, cutters[0], side, result, distance))
        self.cut_targets = []
        return True

    # ---- field tools (tools.register) -------------------------------------------------------------------------
    def _reg_key(self, run):
        """fs-shell-3 11-14: a registry tool's key: the column's row (the strip shows it); Place's key again with an
        object selected picks that object up (it follows the cursor; a click drops it: TOOLSET_DESIGN 3.2 "a key while
        it is selected")"""
        from . import toolreg as TR
        cur = TR.current(self)
        rm = getattr(self, 'reg_mouse', None)
        if run == 'reg:place' and cur == ('place', None) and rm is not None and len(self.viewport.selected or []) == 1:
            line = rm.pick_up()
            try:
                self.status.showMessage(line, 6000)
            except Exception:
                pass
            return
        cat = getattr(self, 'categories', None)
        if cat is not None:
            cat.run(run)
        else:
            TR.select(self, run)

    def add_tool_action(self, label, callback, shortcut=None):
        sc = TOOL_KEYS.get(shortcut, shortcut) if shortcut else None
        if APPROVED_LAYOUT and shortcut == 'K':
            sc = None                          # fs-shell-3 11-14: X is the approved Cut's (REG_KEYS); the outline cut is in Tools
        if label == 'Exit tool':
            sc = None                          # Esc: self.escape exits the tool first
        return self._act(self.menu_tools, label, lambda _=False, f=callback: f(), sc)

    def set_tool_mode(self, mode):
        vp = self.viewport
        vp.tool_mode = mode
        vp.mode_path = []
        label, text = None, ''
        if mode is not None:
            k = type(mode).__name__
            if k == 'PencilMode':
                label = 'Pencil ' + mode.spec.get('mode', 'ridge')
                text = '%s: drag across a form (Esc exits)' % label
            elif k == 'CutMode':
                label = 'Cut and separate' if mode.mode == 'separate' else 'Cut (remove)'
                text = '%s: draw a closed outline over the selected object (Esc exits)' % label
            elif k == 'PickPairMode':
                label = 'Splice' if mode.kind == 'splice' else 'Fill and splice'
                text = '%s: click face A, then face B (Esc exits)' % label
            elif k == 'ChainMode':                   # fs-shell 11-11 (design 4.5)
                text = 'Chain: click a form, then each next point (off the form: in the view plane); Esc ends'
            elif k == 'MeasureMode':                 # fs-shell 11-11 (queue item 5)
                text = ('Measure girth: click a form (Shift: vertical across the screen, Ctrl: facing the view); Esc '
                        'exits' if mode.kind == 'girth' else
                        'Measure with a tape: click two points on a form; Esc exits')
            else:
                text = 'Smooth %s: drag on a form (Esc exits)' % mode.settings.get('layer', '')
        vp.mode_label = text
        self.tools.set_field_tool_active(label)
        self.update_hint()
        if mode is not None:
            self.status.showMessage(text, 8000)
        vp.update()

    def _measure(self, kind):
        from .measure_ui import start
        start(self, kind)

    def _measure_volume(self, material=None):
        """the selected object's volume, and its weight from a material's density (asked unless given)"""
        from ..tools.measure import DENSITY
        if not self.viewport.selected:
            self.status.showMessage('Select an object first.', 6000)
            return
        if material is None:
            from PySide6.QtWidgets import QInputDialog
            names = ['none (volume only)'] + list(DENSITY)
            got, ok = QInputDialog.getItem(self, 'Volume and weight', 'Material (density) for the weight:', names, 1,
                                           False)
            if not ok:
                return
            material = None if got.startswith('none') else got
        self.measures.add_volume(self.viewport.selected[0], material)

    def escape(self):
        if self.viewport.adjust is not None:
            self.viewport.end_adjust()
        elif self.viewport.tool_mode is not None:
            self.set_tool_mode(None)
        else:
            self.props.set_selection([])

    def ask(self, title, fields):
        """A small dialog for a tool's settings (lengths in mm). The self-test answers through ask_auto.
        hand1: asked on the first use of each tool; later picks reuse those values at once (Ctrl held: ask again)."""
        if self.ask_auto is not None:
            out = dict(fields)
            out.update(self.ask_auto.get(title, {}))
            return out
        from PySide6.QtWidgets import QApplication
        remembered = self.ask_memory.get(title)
        if remembered is not None and not (QApplication.keyboardModifiers() & Qt.ControlModifier):
            return dict(remembered)
        out = self._ask_dialog(title, dict(remembered or fields))
        if out is not None:
            self.ask_memory[title] = dict(out)
        # hand1b: the keyboard goes back to the view (the field tool's next input is a drag there). Without this,
        # after the dialog closed there could be no focus widget and no active window at all (the offscreen
        # platform never reactivates the parent; on Windows the focus went back to whatever box had it before)
        self.activateWindow()
        self.viewport.setFocus()
        return out

    def _ask_dialog(self, title, fields):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        form = QFormLayout(dlg)
        ed = {}
        from PySide6.QtWidgets import QCheckBox
        for k, v in fields.items():
            if k in CHOICES:
                w = QComboBox()
                w.addItems(CHOICES[k])
                w.setCurrentText(str(v))
            elif isinstance(v, bool):            # hand2f: on / off settings (Merge: smooth, world frame)
                w = QCheckBox()
                w.setChecked(v)
            else:
                w = QDoubleSpinBox()
                w.setDecimals(2)
                if k in LENGTH_FIELDS:
                    w.setRange(0.0, 500.0)
                    w.setSuffix(' mm')
                    w.setValue(float(v) * 1000.0)
                else:
                    w.setRange(0.0, 10.0)
                    w.setValue(float(v))
            form.addRow(k.replace('_', ' '), w)
            ed[k] = w
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        ok = dlg.exec() == QDialog.Accepted
        out = {}
        for k, w in ed.items():
            if k in CHOICES:
                out[k] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[k] = w.isChecked()
            else:
                out[k] = w.value() / 1000.0 if k in LENGTH_FIELDS else w.value()
        dlg.deleteLater()                      # hand1b: not kept as a hidden child of the window
        return out if ok else None

    def new_scene(self):
        if not self.maybe_save('Start a new scene'):
            return False
        self.set_tool_mode(None)
        self.project_path = None
        self.refpanel.default_span_mm = None      # hand2b (verifier item 26): New figure's 1680 mm span ends here
        self._mark_clean_at = 0
        self.engine.post('new')
        return True

    def new_figure(self):
        """hand2: File > New figure: a standing 1680 mm figure-scale start (head, neck, torso, pelvis and the limb
        pairs as separate objects on a 1 mm lattice; core/figure.py), Mirror X on, a 40 mm Grab brush, framed."""
        from ..core.figure import figure_specs
        if self.new_scene() is False:
            return
        for sp in figure_specs():
            self.engine.post('add', sp)
        self._mark_clean_at = len(figure_specs())   # the preset itself is not unsaved work
        self.tools.set_mirror([True, False, False], world=True)
        self.refpanel.figure_scale(1680.0)    # hand2f: reference planes at figure scale, feet on y = 0
        self.tools.set_tool('grab')
        self.tools.set_radius(0.04)            # after the pick: each brush remembers its own size
        self.viewport.cam.preset(1)
        self.engine.post('frame', [])
        self.status.showMessage('New figure: 1680 mm, %d parts on a 1 mm lattice, Mirror X on. Shape each part with '
                                'Grab / Inflate / Clay (10-80 mm), Move/Rotate/Scale with W/R/Y.' % len(figure_specs()),
                                15000)

    def new_start(self):
        """fs-shell: File > New: the start page; one click (or 1-4, Enter) picks Head, Figure, Hand or Empty"""
        from .start_page import StartPage
        d = StartPage(self)
        self.start_page = d
        ok = d.exec()
        self.start_page = None
        if not ok or not d.choice:
            return
        if d.choice == 'head':
            self.new_head(d.with_refs)
        elif d.choice == 'figure':
            self.new_figure()
        elif d.choice == 'hand':
            self.new_hand()
        else:
            self.new_scene()

    def new_head(self, with_refs=False):
        """fs-shell (design 4.5): the Head start: the 240 mm bust sphere with a neck stub blended in, one object on a
        1 mm lattice (brushes to 80 mm), Mirror X, a 50 mm Grab (11-11: brushes sized to the form), framed
        from the front; the References panel up, and with_refs the photo picker and the Reference Rig straight after"""
        from .start_page import head_spec
        if self.new_scene() is False:
            return
        self._mark_clean_at = 1
        self.engine.post('add', head_spec())
        self.tools.set_mirror([True, False, False])
        self.tools.set_tool('grab')
        self.tools.set_radius(0.05)            # after the pick: each brush remembers its own size
        self.viewport.cam.preset(1)
        self.engine.post('frame', [])
        self.docks['References'].raise_()
        self.status.showMessage('New head: a 240 mm sphere with a neck on a 1 mm lattice, Mirror X on, Grab 50 mm. '
                                'Ctrl+R loads front + side photos into the Reference Rig.', 15000)
        if with_refs and not self.no_prompt:
            from PySide6.QtCore import QTimer as _QT
            _QT.singleShot(0, self.refpanel.load_and_rig)

    def new_hand(self):
        """fs-shell: the Hand start: palm, four fingers and a thumb as one object on a 0.5 mm lattice, Mirror off"""
        from .start_page import hand_spec
        if self.new_scene() is False:
            return
        self._mark_clean_at = 1
        self.engine.post('add', hand_spec())
        self.tools.set_mirror([False, False, False])
        self.tools.set_tool('grab')
        self.tools.set_radius(0.012)            # after the pick: each brush remembers its own size
        self.viewport.cam.preset(1)
        self.engine.post('frame', [])
        self.status.showMessage('New hand: palm, fingers and thumb on a 0.5 mm lattice (brushes to 40 mm).', 15000)

    def new_bust(self):
        """hand1: File > New bust: one sphere of 120 mm radius on a 1 mm sculpt lattice (brushes up to 80 mm stay
        bounded: 160^3 nodes a dab), Mirror X on, a 25 mm Grab brush, framed from the front."""
        if self.new_scene() is False:
            return
        self._mark_clean_at = 1
        self.engine.post('add', {'kind': 'sphere', 'params': {'radius': 0.12}, 'name': 'Bust', 'h': 0.001,
                                 'colour': {'albedo': [0.66, 0.56, 0.49], 'variation': 0.0}})
        self.tools.set_mirror([True, False, False])
        self.tools.set_tool('grab')
        self.tools.set_radius(0.025)            # after the pick: each brush remembers its own size
        self.viewport.cam.preset(1)
        self.engine.post('frame', [])
        self.status.showMessage('New bust: a 240 mm sphere on a 1 mm lattice, Mirror X on. G grab, I inflate, '
                                'S smooth, C clay; W / R / Y move, rotate, scale the object.', 15000)

    # ---- hand1: reference images ---------------------------------------------------------------------------------
    def add_reference_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Reference images: the front and the side (profile) photo',
                                                self.refpanel.last_dir, 'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not paths:
            return
        self.refpanel.last_dir = os.path.dirname(paths[0])
        from .refpanel import sort_front_side
        front, side = sort_front_side(paths)
        self.load_refs(front, side)
        if not self.no_prompt:
            self.refpanel.open_rig()          # fs-shell: straight on to the rig (Auto-fit proposed, Enter applies)

    def load_refs(self, front=None, side=None, span_mm=None, lens=False):
        """Load the front and / or side plane (the command line and the menu); frames the view on them."""
        done = []
        for name, path in (('front', front), ('side', side)):
            if not path:
                continue
            try:
                self.refpanel.load(name, path, span_mm)
                done.append('%s %s' % (name, os.path.basename(path)))
            except Exception as e:
                self.status.showMessage('Reference %s: %s' % (name, e), 15000)
        if done:
            self.docks['References'].raise_()
            if lens:
                self.refpanel.set_lens(True)
            self.status.showMessage('References: %s. They cross at the world origin; 1 = front view, 3 = side view.'
                                    % ', '.join(done), 15000)
        self.viewport.refs_changed()
        return done

    def open_her_editable(self):
        """milestone: her published field as ONE editable object (her store read natively at every level; edits
        written back only as new versions), with her eyeballs as separate protected objects"""
        src = self.published_field()
        if not src:
            QMessageBox.warning(self, 'Her field', 'FIELD_CURRENT.json names no field version on this machine.')
            return False
        if not self.maybe_save('Open her field'):
            return False
        self.set_tool_mode(None)
        self.project_path = None
        self._mark_clean_at = 0
        self.engine.post('open_her', src, 0.00025, True)
        return True

    def _converge_view(self):
        from .render_still import converge_window_view
        return converge_window_view(self)

    def open_still(self):
        """fs-store 11-12: a still (PNG) in the full-resolution viewer (fit, 1:1, zoom, pan)"""
        from .render_still import STILLS
        from .still_viewer import StillViewer
        p, _ = QFileDialog.getOpenFileName(self, 'Open a still', STILLS if os.path.isdir(STILLS) else '', 'Images (*.png *.jpg)')
        if p:
            v = StillViewer(self, p)
            v.show()
            self._still_viewers = getattr(self, '_still_viewers', []) + [v]

    def write_her_back(self):
        import time as _t
        snap = self.snap or {}
        objs = [o for o in snap.get('objects', []) if 'editable copy' in o.get('name', '')]
        if not objs:
            QMessageBox.information(self, 'Write her edits', 'No editable copy of her is open (File > Open her field).')
            return False
        src = self.published_field()
        out = os.path.join(os.path.dirname(src), 'fs_edit_%s' % _t.strftime('%Y%m%d_%H%M%S'))
        r = QMessageBox.question(self, 'Write her edits',
                                 'Write the edits of "%s" into a NEW store version:\n%s\n\nThe published version and '
                                 'FIELD_CURRENT.json are not touched. It runs in the background.' % (objs[0]['name'], out))
        if r == QMessageBox.Yes:
            self.engine.post('her_writeback', objs[0]['id'], out)
        return True

    PROTECT_HINT = '\n'.join([
        'Marking protected parts (once per part):',
        '',
        '1. Select the object (click it).',
        '2. Mask brush (B). Paint the whole visible surface of ONE part, both its faces as far as you can see '
        'them; a small radius for the edges. Ctrl-drag erases. The paint stays on the surface you paint: a facing '
        'surface across a gap never takes it, however narrow the gap (Brush settings: "Mask: other side", 110 deg).',
        '3. Edit > Protect the masked part as... and give it a name.',
        '   The part is built from your marks; the mask clears.',
        '4. Repeat for the next part. Marking a part again replaces it.',
        '5. File > Save. The parts are kept as protected objects (shown; View > Protected parts ghosts or hides '
        'them). Every brush and Merge with protected parts keeps the body a gap away from them.',
        '6. To reshape a part later: Edit > Unprotect a part; sculpt it; Edit > Protect a part again.',
        '',
        'A limit to know: two surfaces closer than about two lattice cells (0.5 mm at a 0.25 mm lattice) cannot be '
        'SCULPTED apart while they are one field: a brush moves both. Marking them works at any gap; to sculpt one '
        'apart from the other, protect it as a part and Unprotect it when you shape it.',
        '',
        'Contact: Swell to contact (Shift+I) and Settle onto (Shift+G) rest ON a protected part (or the selected '
        'object) at gap 0, touching. Inside one field two sheets cannot come closer than ~2 cells without fusing, so '
        'things that touch (a lid and an eye) must stay separate objects.'])

    def protect_hint(self):
        QMessageBox.information(self, 'Marking her protected parts', self.PROTECT_HINT)

    @staticmethod
    def _is_protected_doc(o):
        lk = o.get('locked')
        return bool(lk) if lk is not None else '(protected)' in o.get('name', '')

    def set_root_ask(self):
        """11-09 attached parts: Unprotect the part, paint where it grows from her body with the Mask brush, then this"""
        from PySide6.QtWidgets import QInputDialog
        objs = (self.snap or {}).get('objects', [])
        cand = [o for o in objs if o.get('locked') is False or self._is_protected_doc(o)]
        if not cand:
            QMessageBox.information(self, 'Set root', 'No protected parts in the scene.')
            return
        names = [o['name'] for o in cand]
        name, ok = QInputDialog.getItem(self, "Set a part's root", 'Part (Unprotect it and paint its root with the '
                                        'Mask brush first):', names, 0, False)
        if ok:
            self.engine.post('set_root', cand[names.index(name)]['id'])

    def lock_part_ask(self, lock):
        """11-08 night (design: unlock protected parts and eyes at will, sculpt them, relock). Unprotect: the part
        becomes visible and editable with every brush; Protect relocks it (brushes pass through it again, merges
        keep it apart). Both are undoable steps."""
        from PySide6.QtWidgets import QInputDialog
        objs = (self.snap or {}).get('objects', [])
        if lock:
            cand = [o for o in objs if o.get('locked') is False]
        else:
            cand = [o for o in objs if self._is_protected_doc(o)]
        title = 'Protect a part again' if lock else 'Unprotect a part'
        if not cand:
            QMessageBox.information(self, title, 'No unprotected parts.' if lock else 'No protected parts in the scene.')
            return
        names = [o['name'] for o in cand]
        name, ok = QInputDialog.getItem(self, title, 'Part:', names, 0, False)
        if ok:
            self.engine.post('set_locked', cand[names.index(name)]['id'], bool(lock))

    def protect_masked_ask(self):
        from PySide6.QtWidgets import QInputDialog
        snap = self.snap or {}
        objs = snap.get('objects', [])
        sel = list(self.viewport.selected)
        tgt = [o for o in objs if o['id'] in sel] or [o for o in objs if 'editable copy' in o.get('name', '')]
        if not tgt:
            QMessageBox.information(self, 'Protect the masked part', 'Select her editable copy first.\n\n' + self.PROTECT_HINT)
            return
        name, ok = QInputDialog.getItem(self, 'Protect the masked part as', 'The part you painted with the Mask brush:',
                                        ['part 1', 'part 2', 'part 3', 'part 4'], 0, True)
        if ok:
            self.engine.post('protect_masked', tgt[0]['id'], name)

    def merge_protected_ask(self):
        """milestone: Merge with protected parts: the selected shapes into the target (her), never fused to the
        protected objects (her eyeballs ...) or to the surface painted with the Mask brush"""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QComboBox, QListWidget, QListWidgetItem, \
            QDoubleSpinBox
        from PySide6.QtCore import Qt as _Qt
        snap = self.snap or {}
        objs = snap.get('objects', [])
        sel = list(self.viewport.selected)
        if len(objs) < 2 or not sel:
            QMessageBox.information(self, 'Merge with protected parts', 'Select the shape(s) to merge first.')
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Merge with protected parts')
        lay = QFormLayout(dlg)
        tgt = QComboBox()
        cands = [o for o in objs if o['id'] not in sel]
        for o in cands:
            tgt.addItem(o['name'], o['id'])
        for i, o in enumerate(cands):
            if 'editable copy' in o['name']:
                tgt.setCurrentIndex(i)
        lay.addRow('Merge into', tgt)
        prot = QListWidget()
        for o in cands:
            it = QListWidgetItem(o['name'])
            it.setData(_Qt.UserRole, o['id'])
            it.setFlags(it.flags() | _Qt.ItemIsUserCheckable)
            it.setCheckState(_Qt.Checked if self._is_protected_doc(o) else _Qt.Unchecked)
            prot.addItem(it)
        lay.addRow('Keep separate (protected)', prot)
        mask = QCheckBox('Also keep away from the surface painted with the Mask brush')
        lay.addRow(mask)
        k = QDoubleSpinBox(); k.setRange(0.1, 30.0); k.setDecimals(1); k.setValue(3.0); k.setSuffix(' mm')
        gap = QDoubleSpinBox(); gap.setRange(0.1, 5.0); gap.setDecimals(2); gap.setValue(0.5); gap.setSuffix(' mm')
        lay.addRow('Blend (smooth union)', k)
        lay.addRow('Gap to protected parts', gap)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        target = tgt.currentData()
        protected = [prot.item(i).data(_Qt.UserRole) for i in range(prot.count())
                     if prot.item(i).checkState() == _Qt.Checked and prot.item(i).data(_Qt.UserRole) != target]
        self.engine.post('merge_protected', target, sel, protected, mask.isChecked(), k.value() / 1000.0,
                         gap.value() / 1000.0)

    def published_field(self):
        """r33: the field version work/FIELD_CURRENT.json names (her published model), or None."""
        root = os.path.abspath(os.path.join(HERE, '..', '..'))   # HERE = engine/fieldstudio
        ptr = os.environ.get('FIELDSTUDIO_FIELD_CURRENT', '')   # a pointer file {'field_dir': ...}; unset: none
        try:
            d = json.load(open(ptr, encoding='utf-8'))
            p = os.path.join(root, d['field_dir'])
            return p if os.path.exists(os.path.join(p, 'meta.json')) else None
        except Exception:
            return None

    def project_target(self, path):
        """r27: where saves of an opened project go. A folder named *_staged.fsproj is a staged starting point (a
        template): it opens with no save target, so the first save asks where and offers a dated copy beside it; the
        staged original is never written."""
        name = os.path.basename(os.path.normpath(path or '')).lower()
        if name.endswith('_staged.fsproj'):
            self.template_from = path
            self.status.showMessage('Opened %s as a template: Save asks where (the staged original is never written).'
                                    % os.path.basename(os.path.normpath(path)), 20000)
            return None
        self.template_from = None
        return path

    def open_project(self):
        if not self.maybe_save('Open another project'):
            return
        path = QFileDialog.getExistingDirectory(self, 'Open a Field Studio project folder (.fsproj)')
        if path:
            self.set_tool_mode(None)
            self.project_path = self.project_target(path)
            self.refpanel.default_span_mm = None  # hand2b: the opened project's planes carry their own spans
            try:                                  # fs-shell 11-11: its baked fields are found in its baked/ folder
                from ..tools import bakes as _BK
                _BK.set_project(path)
            except Exception:
                pass
            self.engine.post('open', path)

    # ---- V1.2 field regions -------------------------------------------------------------------------------------------
    def open_field_region(self):
        """File > Open field region: a box / sphere crop of a brick-field version (FV8 store format) as a new object"""
        from ..core import region
        start = ''
        ptr = os.environ.get('FIELDSTUDIO_FIELD_CURRENT', '')
        try:
            start = region.current_version(ptr)
        except Exception:
            pass
        d = QFileDialog.getExistingDirectory(self, 'Open a field version (a folder with meta.json + L*_sdf.npy)', start)
        if not d:
            return
        try:
            with open(os.path.join(d, 'meta.json')) as f:
                meta = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, 'Field region', 'Not a field version: %s' % e)
            return
        regs = meta.get('shot_regions', {})
        CHOICES['region'] = list(regs) + ['view centre']
        CHOICES['shape'] = ['box', 'sphere']
        a = self.ask('Field region', {'region': CHOICES['region'][0], 'shape': 'box', 'size': 0.03})
        if not a:
            return
        c = regs[a['region']]['centre'] if a['region'] in regs else [float(v) for v in self.viewport.cam.target]
        try:
            p = region.region_params(d, c, float(a['size']), a['shape'])
        except Exception as e:
            QMessageBox.warning(self, 'Field region', str(e))
            return
        self.set_tool_mode(None)
        self.engine.post('open_region', p, 'region %s %s' % (os.path.basename(d), a['region']))

    def write_region_back(self):
        """File > Write region back: the selected region's sculpt -> a NEW version folder (never published)"""
        from PySide6.QtWidgets import QInputDialog
        objs = {o['id']: o for o in (self.snap or {}).get('objects', [])}
        oid = next((i for i in self.props.order if objs.get(i, {}).get('base', {}).get('kind') == 'sampled'), None)
        if oid is None:
            oid = next((i for i, o in objs.items() if o['base']['kind'] == 'sampled'), None)
        if oid is None:
            QMessageBox.information(self, 'Write region back', 'Select a field region object first.')
            return
        src = objs[oid]['base']['params']['source']
        default = os.path.join(os.path.dirname(src), '%s_fs_%s' % (os.path.basename(src), time.strftime('%Y%m%d_%H%M%S')))
        path, ok = QInputDialog.getText(self, 'Write region back',
                                        'New version folder (must not exist; it is never published):', text=default)
        if ok and path:
            self.engine.post('write_region', oid, path)

    def on_region_done(self, rep):
        self.region_report = rep
        if 'error' in rep:
            self.status.showMessage('Write region back failed: %s' % rep['error'], 20000)
            if self.ask_auto is None:
                QMessageBox.warning(self, 'Write region back', rep['error'])
            return
        g = rep.get('guard', {})
        gr = g.get('report', {})
        txt = ('New version: %s\n(NOT published)\n\nGuard (fv8_current.py --check): %s - %s\n'
               'Face guard sha256: %s\nDelta: %d nodes, max %.1f um; apply %.1f s, commit %.1f s, check %.1f s' % (
                   rep['new_version'], 'OK' if g.get('ok') else 'FAILED', gr.get('why', g.get('stdout_tail', '')),
                   (gr.get('face_guard_sha256') or '')[:16], rep['delta_nodes_nonzero'], rep['delta_max_abs_um'] or 0,
                   rep['apply_s'], rep['commit_s'], g.get('s', 0)))
        self.status.showMessage('Region written: %s (guard %s)' % (rep['new_version'], 'ok' if g.get('ok') else 'FAILED'),
                                20000)
        if self.ask_auto is None:
            QMessageBox.information(self, 'Write region back', txt)

    def save_project(self, ask=False):
        path = self.project_path
        if ask or not path:
            start = 'untitled.fsproj'
            tf = getattr(self, 'template_from', None)
            if tf:                                # r27: a dated working copy beside the staged original
                base = os.path.basename(os.path.normpath(tf))
                start = os.path.join(os.path.dirname(os.path.normpath(tf)),
                                     base[:-len('_staged.fsproj')] + time.strftime('_%Y-%m-%d_%H%M.fsproj'))
            path, _ = QFileDialog.getSaveFileName(self, 'Save project (a .fsproj folder)', start,
                                                  'Field Studio project (*.fsproj)')
            if not path:
                return
            if not path.endswith('.fsproj'):
                path += '.fsproj'
            tf = getattr(self, 'template_from', None)
            if tf and os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(tf)):
                QMessageBox.warning(self, 'Field Studio', 'That is the staged original; choose another name.')
                return
            self.project_path = path
            self.template_from = None
        extra = {'camera': self.viewport.cam.to_dict(), 'app': 'Field Studio V1',
                 'view': self.refpanel.lens_state(), 'units': getattr(self, 'units', 'mm')}
        if getattr(self, 'measures', None) is not None and self.measures.items:
            extra['measures'] = self.measures.to_list()
        if getattr(self, 'marks', None) is not None and self.marks.items:
            extra['marks'] = self.marks.to_list()     # fs-shell 11-11: the marks (design 4.14)
        if self.snap is not None:
            self.saved_marker = self._marker(self.snap)        # hand2f: what this save holds (the unsaved check)
        ref_copies = []
        if self.viewport.refs.planes:
            # hand1: the reference images go into <project>/refs/ (local only; never part of a public copy).
            # r49: the SAVE WORKER copies them into its own new folder before the swap (the window copied them into the
            # live folder before posting, so two quick saves could lose or mix them up while one was being swapped)
            rd = os.path.join(path, 'refs')
            for p in self.viewport.refs.planes:
                if p.path and os.path.exists(p.path):
                    # hand1b: one file per plane (<plane>_<name>), so two images with the same file name from
                    # different folders no longer overwrite each other
                    base = os.path.basename(p.path)
                    if not base.startswith(p.name + '_'):
                        base = '%s_%s' % (p.name, base)
                    ref_copies.append((os.path.abspath(p.path), base))
                    p.path = os.path.join(rd, base)
            extra['refs'] = self.viewport.refs.to_dict(rel_to=path)
        self._carry_bakes(path)                # fs-shell 11-11: baked fields (Conform) into <project>/baked/ first
        self.engine.post('save', path, extra, ref_copies)
        return path

    def _carry_bakes(self, path):
        """fs-shell 11-11 (projects stay self-contained; Save As and autosave carry their bakes): the baked
        fields the scene uses are copied into <path>/baked/ before the save (its folder swap carries them over)"""
        try:
            from ..tools import bakes as BK
            BK.carry_files(BK.used_in_snapshot(self.snap), path)
            if path == self.project_path:
                BK.set_project(path)           # the saved project's baked/ folder is searched first from now on
        except Exception as e:
            print('bakes not carried into %s: %r' % (path, e))

    def screenshot(self, window=False, path=None):
        os.makedirs(SCREENS, exist_ok=True)
        if path is None:
            path = os.path.join(SCREENS, '%s_%s.png' % ('window' if window else 'viewport',
                                                       time.strftime('%Y%m%d_%H%M%S')))
        pm = (self if window else self.viewport).grab()
        ok = pm.save(path)
        self.status.showMessage(('Saved %s' if ok else 'Could not save %s') % path, 8000)
        return path if ok else None

    def show_help(self):
        QMessageBox.information(self, 'Field Studio controls', (
            'The model is implicit fields only: F(X) = 0 is the surface, F < 0 inside. No mesh anywhere.\n\n'
            'Create: toolbar or Create menu (analytic primitives at the view centre).\n'
            'Select (Q): click selects, Shift+click adds; Ctrl+click in the Scene list.\n'
            'Merge: select two or more objects (Shift+click), Ctrl+J (last settings) or Ctrl+Shift+J (asks): one object, '
            'strokes cross the joints, every sculpt layer kept. Subtract / Intersect in Properties or the right-click '
            'menu.\n'
            'Right-click (no drag): the action menu (merge, duplicate, Ctrl+Shift+D mirror-duplicate across x = 0, F2 '
            'rename, H hide, / isolate, Shift+H show all, Alt+M / Ctrl+I mask clear / invert, Ctrl+A select all).\n'
            'Mutate: pick a node in Layers, then Twist / Bend / Taper / Stretch / Shell / Round / Move-Rotate.\n'
            'Node parameters edit live; every change is one undo step (slider drags merge).\n\n'
            'Sculpt: D draw, C clay, V carve, I inflate, S smooth, T flatten, K crease, P pinch, G grab, E stretch, '
            'B mask. Ctrl while sculpting inverts (draw/carve, mask erase), Shift smooths. [ ] or F-drag radius, - = or Shift+F-drag strength. Object: W move, R rotate, Y scale.\n'
            'Mirror: Alt+X / Y / Z; Alt+W mirrors about the WORLD x = 0 plane onto whichever object is there.\n'
            'Navigate: L-drag on empty space or Alt+L-drag orbit, R/M-drag pan, wheel zoom, . frame, '
            '1 3 7 5 9 views.\n\n'
            'Field tools (Tools menu or the left panel): L pencil ridge, Shift+L valley, X cut, Shift+X cut and '
            'separate, J splice, Shift+J fill and splice; Esc exits a tool. Smooth on: sculpt / detail band / '
            'shape band.\n\n'
            'Ctrl+Z / Ctrl+Y undo / redo. Ctrl+S viewport PNG to screens/. Ctrl+Shift+S save the project. Quitting with '
            'unsaved changes asks first; an autosave runs every 3 minutes or 20 steps.'))

    def show_what_to_try(self):
        """V1.3: Help > What to try: a short card of first things to do with the default scene."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout
        dlg = getattr(self, '_try', None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle('What to try')
            dlg.resize(620, 560)
            lay = QVBoxLayout(dlg)
            tb = QTextBrowser(dlg)
            tb.setHtml(WHAT_TO_TRY_HTML)
            lay.addWidget(tb)
            bb = QDialogButtonBox(QDialogButtonBox.Close, dlg)
            bb.rejected.connect(dlg.close)
            lay.addWidget(bb)
            self._try = dlg
        dlg.show()
        dlg.raise_()
        return dlg

    def show_quick_start(self, at_launch=False):
        """Help > Quick start (F1): a non-modal 'Getting started' panel listing every control (V1.1)."""
        from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout
        cfg = app_settings()
        if at_launch:
            # hand1: shown at launch only until it has been seen once (it took the keyboard from the main window)
            if (str(cfg.value('quick_start_at_launch', 'true')).lower() != 'true'
                    or str(cfg.value('quick_start_seen', 'false')).lower() == 'true'):
                return None
            cfg.setValue('quick_start_seen', 'true')
        dlg = getattr(self, '_quick', None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle('Getting started with Field Studio')
            dlg.resize(760, 680)
            lay = QVBoxLayout(dlg)
            tb = QTextBrowser(dlg)
            tb.setOpenExternalLinks(False)
            tb.setHtml(QUICK_START_HTML)
            lay.addWidget(tb)
            cb = QCheckBox('Show this at launch', dlg)
            cb.setChecked(str(cfg.value('quick_start_at_launch', 'true')).lower() == 'true')
            cb.toggled.connect(lambda on: cfg.setValue('quick_start_at_launch', 'true' if on else 'false'))
            lay.addWidget(cb)
            bb = QDialogButtonBox(QDialogButtonBox.Close, dlg)
            bb.rejected.connect(dlg.close)
            lay.addWidget(bb)
            self._quick = dlg
        dlg.show()
        dlg.raise_()
        if at_launch:
            self.activateWindow()             # the main window keeps the keyboard
            self.viewport.setFocus()
        return dlg

    def closeEvent(self, e):
        if not self.maybe_save('Quit Field Studio'):
            e.ignore()
            return
        # fs-store 11-12: a save still being written (Ctrl+S just before Quit, or the one maybe_save started) finishes
        # first; if it failed, the window stays open with the work in it and the session autosave is kept
        if not self.wait_for_save('Quit Field Studio'):
            e.ignore()
            return
        self._drop_session_autosave()
        try:
            self.tools.flush_memory()        # fs-shell 11-11: the brushes' own settings
        except Exception:
            pass
        self.engine.stop()
        super().closeEvent(e)

    # ---- hand2f: save safety (unsaved prompt, autosave, recovery) ------------------------------------------------------
    @staticmethod
    def _marker(snap):
        return (list(snap.get('log_ids', [])), int(snap.get('committed', 0)))

    def is_dirty(self):
        sn = self.snap
        if sn is None:
            return False
        return self._marker(sn) != self.saved_marker

    def maybe_save(self, what):
        """-> True to go on (saved, discarded or nothing to save), False to stay (Cancel)."""
        if self.no_prompt or not self.is_dirty():
            return True
        r = QMessageBox.question(self, what, 'The scene has changes that are not saved (%d steps in the timeline).\n'
                                 'Save them first?' % len((self.snap or {}).get('log', [])),
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Discard:
            return True
        path = self.save_project()
        if not path:
            return False
        # fs-store 11-12: wait for the save itself, not engine.idle() (idle came ~0.1 s after the post, while the
        # writer thread was still writing: Quit then killed it, an 18:24 save). False (stay) when it failed.
        return self.wait_for_save(what, posted=True)

    def wait_for_save(self, what, posted=False):
        """fs-store 11-12: wait until no project save is queued or being written, with a 'Saving...' dialog after a
        moment. -> True when there was none or the last one is complete; False when it failed (SAVE FAILED is shown by
        on_save_done) or the user chose to keep the window open while it goes on."""
        eng = self.engine
        if not eng.save_pending():
            # nothing in flight: the save this call posted (if any) has already ended; an older failure was said
            # when it happened and leaves the work marked unsaved (a Discard at Quit then closes as chosen)
            return self._last_save_ok() if posted else True
        from PySide6.QtWidgets import QProgressDialog
        dlg = None
        t0 = time.perf_counter()
        asked = t0
        while eng.save_pending():
            QApplication.processEvents()
            el = time.perf_counter() - t0
            if dlg is None and el > 0.4:
                dlg = QProgressDialog('Saving %s ...\nField Studio waits until it is written.' % (
                    os.path.basename(str(getattr(self, 'project_path', '') or 'the project'))), None, 0, 0, self)
                dlg.setWindowTitle(what)
                dlg.setWindowModality(Qt.WindowModal)
                dlg.setMinimumDuration(0)
                dlg.setCancelButton(None)
                dlg.show()
            if dlg is not None:
                dlg.setLabelText('Saving %s ... (%.0f s)\nField Studio waits until it is written.' % (
                    os.path.basename(str(getattr(self, 'project_path', '') or 'the project')), el))
            if time.perf_counter() - asked > 120.0:
                asked = time.perf_counter()
                if self.no_prompt:
                    if el > 1800.0:
                        break
                else:
                    r = QMessageBox.question(self, what, 'The save is still being written (%.0f s).\nKeep waiting? '
                                             '(No keeps Field Studio open; the save goes on.)' % el,
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    if r == QMessageBox.No:
                        break
            time.sleep(0.03)
        if dlg is not None:
            dlg.close()
        QApplication.processEvents()                # the save's own end (on_save_done) is handled first
        if eng.save_pending():
            return False
        return self._last_save_ok()

    def _last_save_ok(self):
        ls = getattr(self.engine, 'last_save', None)
        return ls is None or ls[1] is not None

    def on_save_done(self, res):
        """fs-store 11-12: the end of every project save. A failure keeps the work marked unsaved (Quit asks again)
        and says so in a dialog; the session autosave is untouched."""
        if res.get('ok'):
            return
        self.saved_marker = ([], -1)                # matches no state: the work counts as unsaved
        txt = 'SAVE FAILED for %s\n\n%s\n\nYour work is still open and is not lost; the session autosave is kept. ' \
              'Try File > Save As... to another folder.' % (res.get('path'), res.get('error'))
        self.status.showMessage(txt.replace('\n', ' '), 60000)
        if not self.no_prompt:
            QMessageBox.critical(self, 'SAVE FAILED', txt)

    def _autosave_paths(self):
        out = []
        if self.autosave_dir:
            out.append(os.path.join(self.autosave_dir, '%s.fsproj' % self.session))
        if self.project_path and not self.no_prompt:
            out.append(self.project_path.rstrip('/\\') + '.autosave')
        return out

    def autosave_tick(self, force=False):
        """every 10 s: autosave when the work changed and 3 minutes have passed or 20 steps were added, and nothing
        is being dragged (the snapshot is taken on the engine thread, the write runs on a worker thread)"""
        if not self.autosave_dir or self.snap is None:
            return False
        mk = self._marker(self.snap)
        if mk == self.autosave_marker:
            return False
        changed = len(set(mk[0]) ^ set(self.autosave_marker[0])) + abs(mk[1] - self.autosave_marker[1])
        due = force or changed >= 20 or time.perf_counter() - self.autosave_t >= 180.0
        if not due or self.viewport.drag is not None or self.engine._stroke is not None or self.engine.autosave_busy:
            return False
        extra = {'camera': self.viewport.cam.to_dict(), 'app': 'Field Studio V1 (autosave)',
                 'view': self.refpanel.lens_state(), 'autosave_of': self.project_path or '',
                 'units': getattr(self, 'units', 'mm')}
        if self.viewport.refs.planes:
            extra['refs'] = self.viewport.refs.to_dict()
        for ap_ in self._autosave_paths():
            self._carry_bakes(ap_)            # fs-shell 11-11: the autosave carries its bakes too
        self.engine.post('autosave', self._autosave_paths(), extra)
        self.autosave_marker = mk
        self.autosave_t = time.perf_counter()
        return True

    def _drop_session_autosave(self):
        import shutil
        if self.autosave_dir:
            shutil.rmtree(os.path.join(self.autosave_dir, '%s.fsproj' % self.session), ignore_errors=True)

    def offer_recovery(self):
        """at launch: an autosave left by a session that did not close cleanly (a crash, a kill) is offered once;
        it is renamed to .recovered either way, so it is not offered again (kept as a backup, newest 5)."""
        d = self.autosave_dir
        if self.no_prompt or not d or not os.path.isdir(d):
            return None
        cands = []
        for n in os.listdir(d):
            fp = os.path.join(d, n)
            if n.endswith('.fsproj') and not n.startswith(self.session) and os.path.isfile(
                    os.path.join(fp, 'project.json')):
                if _session_alive(n[:-len('.fsproj')]):
                    continue                  # hand2b (verifier item 25): another open window's live autosave
                cands.append((os.path.getmtime(os.path.join(fp, 'project.json')), fp))
        if not cands:
            return None
        cands.sort()
        mt, fp = cands[-1]
        try:
            with open(os.path.join(fp, 'project.json'), encoding='utf-8') as f:
                doc = json.load(f)
            info = '%d objects, %d timeline steps' % (len(doc['scene']['objects']), len(doc.get('log', [])))
            src = doc.get('extra', {}).get('autosave_of') or 'an unsaved scene'
        except Exception:
            info, src = 'unreadable', '?'
        when = time.strftime('%Y-%m-%d %H:%M', time.localtime(mt))
        rec = fp[:-len('.fsproj')] + '.recovered'
        import shutil
        shutil.rmtree(rec, ignore_errors=True)
        os.replace(fp, rec)
        for _, old in cands[:-1]:
            tgt = old[:-len('.fsproj')] + '.recovered'
            try:                              # hand2b: os.replace onto an existing folder raises on Windows
                shutil.rmtree(tgt, ignore_errors=True)
                os.replace(old, tgt)
            except OSError:
                pass
        recs = sorted((os.path.getmtime(os.path.join(d, n)), os.path.join(d, n)) for n in os.listdir(d)
                      if n.endswith('.recovered'))
        for _, old in recs[:-5]:
            shutil.rmtree(old, ignore_errors=True)
        r = QMessageBox.question(self, 'Recover autosaved work?',
                                 'Field Studio was not closed cleanly last time. An autosave from %s was found (%s; '
                                 'of %s).\n\nOpen it now? (It is kept either way in %s.)' % (when, info, src, rec),
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r == QMessageBox.Yes:
            self.project_path = None          # Save asks where (the autosave folder is not the user's project)
            self.engine.post('open', rec)
            return rec
        return None

    # ---- hand2f: the right-click action menu, selection, rename, merge settings -----------------------------------------
    def context_at_mouse(self):
        vp = self.viewport
        from PySide6.QtCore import QPointF
        pos = vp.mouse_pos or QPointF(vp.width() / 2, vp.height() / 2)
        self.on_context(vp._obj_at(pos), QPointF(pos))

    def on_context(self, oid, pos):
        """Right-click (no drag), the Menu key or Shift+F10: the actions for what is under the cursor. The object
        joins the selection unless it is already in it (a multi-selection stays when the click is on a member)."""
        from PySide6.QtWidgets import QMenu
        if oid >= 0 and oid not in self.props.order:
            self.on_viewport_select(oid, False)
        sel = list(self.props.order)
        menu = QMenu(self)
        n = len(sel)

        def add(m, text, fn, key=None, enabled=True):
            a = m.addAction(text + (('\t' + key) if key else ''))
            a.triggered.connect(fn)
            a.setEnabled(enabled)
            return a
        if oid >= 0 or n:
            o = self.props.obj(sel[0]) if sel else None
            head = menu.addAction('%s%s' % (o['name'] if o else '?', (' + %d more' % (n - 1)) if n > 1 else ''))
            head.setEnabled(False)
            # fs-shell 11-14 (Tier 0 part 5: how to rotate or manipulate an object was not findable): the
            # transforms first, where a right-click looks for them
            for tk, tn in (('move', 'Move'), ('rotate', 'Rotate'), ('scale', 'Scale')):
                add(menu, tn, lambda _=False, k=tk: self.tools.set_tool(k), self._key_of(tk, {'move': 'W', 'rotate': 'R',
                                                                                              'scale': 'Y'}[tk]))
            menu.addSeparator()
            if os.environ.get('FS_LAYOUT', '') != 'classic':   # fs-shell-2 11-14: ONE Merge (fs-tools' job)
                add(menu, 'Merge %d selected (one object; blend at contact in the strip)' % n if n > 1
                    else 'Merge (select 2+ first)', lambda: self.props.merge('union'), 'Ctrl+J')
            else:
                mm = menu.addMenu('Merge %d selected' % n if n > 1 else 'Merge (select 2+ first)')
                mm.setEnabled(n >= 2)
                k_mm = self.props.blend.value() * 1000
                add(mm, 'Merge with the last settings (%s)' % ('smooth %.0f mm' % k_mm if self.props.smooth_merge.isChecked()
                                                              else 'sharp'), lambda: self.props.merge('union'), 'Ctrl+J')
                add(mm, 'Merge sharp', lambda: self.props.merge('union', 0.0))
                add(mm, 'Merge smooth, k = %.0f mm' % max(k_mm, 10.0), lambda: self.props.merge('union', self.props.merge_k(True)))
                add(mm, 'Merge smooth...', self.merge_ask, 'Ctrl+Shift+J')
            cm = menu.addMenu('Combine')
            cm.setEnabled(n >= 2)
            add(cm, 'Subtract the others from %s' % (o['name'] if o else 'A'), lambda: self.props._boolean('subtract'))
            add(cm, 'Intersect', lambda: self.props._boolean('intersect'))
            menu.addSeparator()
            add(menu, 'Duplicate (with sculpt)', self.props._duplicate, 'Ctrl+D')
            add(menu, 'Mirror-duplicate across X (world)', self.props._mirror_dup, 'Ctrl+Shift+D')
            add(menu, 'Rename...', self.rename_object, 'F2', enabled=n == 1)
            add(menu, 'Delete', self.props._delete, 'Del')
            menu.addSeparator()
            add(menu, 'Hide', lambda: self.engine.post('view', 'hide', list(self.props.order)), 'H')
            add(menu, 'Isolate (again: show all)', self.isolate, '/')
            add(menu, 'Show all', lambda: self.engine.post('view', 'show_all', []), 'Shift+H')
            add(menu, 'Frame selected', self.frame_selection, '.')
            menu.addSeparator()
            km = menu.addMenu('Mask')
            add(km, 'Clear mask', lambda: self.props._mask('clear'), 'Alt+M')
            add(km, 'Invert mask', lambda: self.props._mask('invert'), 'Ctrl+I')
            sm = menu.addMenu('Select')
        else:
            am = menu.addMenu('Add primitive here')
            for kind, name, params in PRIMS:
                a = am.addAction(prim_icon(kind), name)
                a.triggered.connect(lambda _=False, k=kind, p=params, q=QPointFCopy(pos): self.add_primitive_at(k, p, q))
            add(menu, 'Show all', lambda: self.engine.post('view', 'show_all', []), 'Shift+H')
            add(menu, 'Frame all', lambda: self.engine.post('frame', []))
            vm = menu.addMenu('View')
            for key, name in ((1, 'Front'), (3, 'Right side'), (9, 'Back'), (7, 'Top')):
                add(vm, name, lambda _=False, k=key: self.preset(k), str(key))
            sm = menu.addMenu('Select')
        add(sm, 'All', self.select_all, 'Ctrl+A')
        add(sm, 'None', lambda: self.props.set_selection([]), 'Esc')
        add(sm, 'Invert', self.select_invert, 'Ctrl+Shift+I')
        self._last_menu = menu
        if getattr(self, 'menu_auto', None) is not None:     # tests: run an item by its text instead of showing it
            return menu
        menu.popup(self.viewport.mapToGlobal(pos.toPoint()))
        return menu

    def add_primitive_at(self, kind, params, pos):
        """Add a primitive where the right-click was: on the view plane through the camera target, sized for the
        scene (figure scale when the scene is a figure), on the lattice the scene uses."""
        import numpy as np
        vp = self.viewport
        x, y, v = vp.to_render(pos)
        cam = v['cam']
        o, d = cam.ray(x, y, v['w'], v['h'])
        f, r, u = cam.basis()
        den = float(d @ f)
        t = float((cam.target - o) @ f) / den if abs(den) > 1e-9 else float(np.linalg.norm(cam.target - o))
        pt = o + d * t
        objs = (self.snap or {}).get('objects', [])
        h = min([ob['sculpt_h'] for ob in objs] or [0.0005])
        col = PALETTE[self.colour_i % len(PALETTE)]
        self.colour_i += 1
        self.engine.post('add', {'kind': kind, 'params': dict(params), 'translate': [round(float(c), 4) for c in pt],
                                 'colour': {'albedo': col, 'variation': 0.0}, 'h': h})

    def _fit_hint(self, m):
        """fs-shell-6 11-14: while a status message shows, the keys hint narrows to leave the message its whole text"""
        if not m:
            self.hint.setMaximumWidth(16777215)
            return
        from PySide6.QtWidgets import QWidget as _W
        others = sum(w.width() + 6 for w in self.status.findChildren(_W, options=Qt.FindDirectChildrenOnly)
                     if w.isVisible() and w is not self.hint and not w.isWindow())
        need = self.status.fontMetrics().horizontalAdvance(m) + 30
        self.hint.setMaximumWidth(max(160, self.status.width() - others - need))

    def group_selected(self):
        """fs-shell-6 11-14: Group (fs-tools' GroupObjects); two objects on both sides of her midline are named by her
        side in the same undo step: x < 0 is her RIGHT (on the screen's left from the front), x > 0 her LEFT"""
        oids = [int(o) for o in (self.props.order or self.viewport.selected or [])]
        if len(oids) < 2:
            self.status.showMessage('Group: select two or more objects first (Shift+click adds).', 6000)
            return
        objs = {o['id']: o for o in (self.snap or {}).get('objects', [])}
        names = side_names(oids, objs)

        # a job of the window's own (not a 'call': a call changes the scene without a new snapshot, so the Scene list
        # showed the names and the group only at the next change)
        if 'shell.group' not in (getattr(self.engine, '_job_handlers', None) or {}):
            self.engine.register_job('shell.group', _group_job)
        self.engine.post('shell.group', oids, names)
        self.status.showMessage('Grouped %d objects%s: they select and move together (Ctrl+Shift+G ungroups).' % (
            len(oids), (', named ' + ' and '.join(sorted(names.values()))) if names else ''), 8000)

    def ungroup_selected(self):
        oids = [int(o) for o in (self.props.order or self.viewport.selected or [])]
        if oids:
            self.engine.post('tools.ungroup', oids)
            self.status.showMessage('Ungrouped: each object selects and moves on its own again.', 6000)

    def rename_object(self):
        from PySide6.QtWidgets import QInputDialog
        o = self.props.obj()
        if o is None:
            return
        name, ok = QInputDialog.getText(self, 'Rename', 'Name of %s:' % o['name'], text=o['name'])
        if ok and name.strip() and name.strip() != o['name']:
            self.engine.post('props', o['id'], {'name': name.strip()})

    def select_all(self):
        self.props.set_selection([o['id'] for o in (self.snap or {}).get('objects', []) if not o.get('hidden')])

    def select_invert(self):
        cur = set(self.props.order)
        self.props.set_selection([o['id'] for o in (self.snap or {}).get('objects', [])
                                  if o['id'] not in cur and not o.get('hidden')])

    def isolate(self):
        if self.props.order:
            self.engine.post('view', 'isolate', list(self.props.order))

    def consolidate_ask(self):
        """r44: explicit: folds the op log's old entries into one checkpoint (smaller projects); the last 50 steps stay
        undoable and the state hash is checked before and after"""
        r = QMessageBox.question(
            self, 'Consolidate history',
            'Fold the op log\'s older entries into one checkpoint of the current history?\n\n'
            'The last 50 steps stay undoable; older steps can no longer be undone or stepped to in the Timeline. '
            'The model does not change: the new log is replayed and must give the same state hash, or nothing changes.\n\n'
            'The project gets smaller when it is next saved.')
        if r == QMessageBox.Yes:
            self.engine.post('consolidate', 50)

    def merge_ask(self):
        if len(self.props.order) < 2:
            self.status.showMessage('Merge: select two or more objects first (Shift+click adds).', 8000)
            return
        if os.environ.get('FS_LAYOUT', '') != 'classic':   # fs-shell-2 11-14 (ONE Merge): its settings are the strip's
            try:
                from . import toolreg as TR
                if TR.runnable(self, 'merge'):
                    TR.select(self, 'reg:merge')
                    self.status.showMessage('Merge: set Blend at contact in the strip, then Merge.', 8000)
                    return
            except Exception as e_:
                print('registry merge not selected: %r' % (e_,))
        fields = {'smooth': self.props.smooth_merge.isChecked(), 'blend': float(self.props.blend.value()),
                  'world_frame': self.props.world_frame.isChecked()}
        if self.ask_auto is not None:
            v = self.ask('Merge selected', fields)
        else:                                  # always asks (the panel remembers the answer for Ctrl+J)
            v = self._ask_dialog('Merge selected', fields)
            self.activateWindow()
            self.viewport.setFocus()
        if v is None:
            return
        self.props.smooth_merge.setChecked(bool(v['smooth']))
        self.props.blend.setValue(float(v['blend']))
        self.props.world_frame.setChecked(bool(v['world_frame']))
        self.props._save_merge_settings()
        self.props.merge('union')


WHAT_TO_TRY_HTML = """
<h2>What to try (10 minutes)</h2>
<p>The window opens with a sphere and a cube. The status bar's right side always says what each mouse button does
right now; the badge at the top left of the view names the active tool.</p>
<ol>
<li><b>Look around.</b> Drag on empty space to orbit, right-drag to pan, wheel to zoom, . (period) to frame.</li>
<li><b>Sculpt the sphere.</b> Press C (Clay) and drag over the sphere. Then D (Draw), V (Carve), S (Smooth).
[ and ] change the radius; hold Ctrl to invert Draw / Carve. Each drag is one undo step (Ctrl+Z).</li>
<li><b>Grab and pull.</b> G, then drag from the sphere's side: the form follows the mouse.</li>
<li><b>A pencil ridge.</b> Press L and drag across the cube's front face. Shift+L draws a valley.
Tools &gt; Edit last pencil... changes its width, depth and profile afterwards.</li>
<li><b>Cut a piece off.</b> Select the cube (Q, click), press Shift+X and draw a loop over a corner:
the corner becomes its own object, moved apart.</li>
<li><b>Join two forms.</b> Press J, click a spot on the sphere (face A), then on the cube (face B):
the cube lands on the sphere with a smooth join.</li>
<li><b>Combine.</b> Select both (click, Shift+click), then Subtract with Blend k &gt; 0 in Properties.</li>
<li><b>Mutate.</b> Pick a node in Layers, then Twist or Bend; drag its rate - it stays live.</li>
<li><b>Look inside.</b> The view bar's Display: field slice shows F on a plane (white = the surface).</li>
<li><b>Step back in time.</b> Click any entry in the Timeline; Verify replay checks the state hash.</li>
</ol>
<p><b>Things to tell us about:</b> anything that feels slow (the telemetry strip shows ms per frame and L, the
field's steepness), a stroke that lands somewhere odd, a crash or a freeze. Ctrl+S saves the view as a PNG.</p>
<p>Known in V1.3: the first launch after an update compiles the kernels (about a minute, progress shown);
frames are slower right after big strokes until you release the mouse; booleans and Duplicate drop the sculpt.</p>
"""


QUICK_START_HTML = """
<h2>Field Studio: getting started</h2>
<p>The model is fields only: F(X) = 0 is the surface, F &lt; 0 inside. There is no mesh. Everything runs on the CPU.
Every change is one undo step (Ctrl+Z / Ctrl+Y), and the timeline at the bottom steps to any point.</p>
<h3>1. Look around</h3>
<table cellpadding="3">
<tr><td><b>Orbit</b></td><td>L-drag on empty space, or Alt+L-drag anywhere</td></tr>
<tr><td><b>Pan</b></td><td>R-drag or M-drag</td></tr>
<tr><td><b>Zoom</b></td><td>mouse wheel</td></tr>
<tr><td><b>Frame</b></td><td>. or Home (the selection, or everything)</td></tr>
<tr><td><b>Views</b></td><td>1 front, 3 right, 7 top, 5 three-quarter, 9 back</td></tr>
<tr><td><b>Display</b></td><td>the view bar: shaded, matcap, normals, field slice (white = the surface)</td></tr>
</table>
<h3>2. Make and combine forms</h3>
<table cellpadding="3">
<tr><td><b>Create</b></td><td>toolbar or Create menu: sphere, cube, cylinder, capsule, cone, torus, ellipsoid, slab</td></tr>
<tr><td><b>Select</b></td><td>Q then click; Shift+click adds; Esc clears</td></tr>
<tr><td><b>Combine</b></td><td>select two objects: Union / Subtract / Intersect (Blend k &gt; 0 = smooth), or Splice</td></tr>
<tr><td><b>Mutate</b></td><td>pick a node in Layers: Twist, Bend, Taper, Stretch, Shell, Round, Move/Rotate</td></tr>
<tr><td><b>Edit later</b></td><td>every node's parameters stay live in Properties (drags merge into one undo step)</td></tr>
</table>
<h3>3. Sculpt</h3>
<table cellpadding="3">
<tr><td><b>Brush</b></td><td>L-drag on a form with a brush active. D draw, C clay, V carve, I inflate, S smooth, T flatten,
K crease, P pinch, G grab, E stretch, B mask</td></tr>
<tr><td><b>Radius / strength</b></td><td>[ ] or hold F and drag; - = or hold Shift+F and drag (also the Brush panel)</td></tr>
<tr><td><b>Smooth</b></td><td>hold Shift while sculpting (any brush)</td></tr>
<tr><td><b>Whole object</b></td><td>W move, R rotate, Y scale. Handles on the selected object: arrows move along an axis, rings turn about one, boxes stretch one axis (Ctrl+drag a box: skew); the centre or the form: as before</td></tr>
<tr><td><b>Mirror</b></td><td>Alt+X / Alt+Y / Alt+Z, or the Brush panel</td></tr>
<tr><td><b>Invert</b></td><td>hold Ctrl while sculpting (draw becomes carve, mask erases)</td></tr>
<tr><td><b>Smooth on</b></td><td>Brush Controls: the whole sculpt, the detail band only (shape kept) or the shape band only
(detail kept)</td></tr>
</table>
<h3>4. Field tools (left panel or Tools menu)</h3>
<table cellpadding="3">
<tr><td><b>Pencil ridge / valley</b></td><td>L / Shift+L, then drag across a form. The stroke is one live swept field:
Tools &gt; Edit pencil changes width, depth, profile (round, V, U), ridge/valley and softness afterwards</td></tr>
<tr><td><b>Cut / Cut and separate</b></td><td>X / Shift+X, then draw an outline over the object: the inside is swept
through and removed, or becomes its own object moved apart</td></tr>
<tr><td><b>Splice / Fill and splice</b></td><td>J / Shift+J, then click face A, then face B: B lands on A with a smooth
join, or a pillar fills the gap lofted from A's profile to B's</td></tr>
<tr><td><b>Leave a tool</b></td><td>pick any other tool (button or key), or Esc (a second Esc clears the selection). A field tool's settings dialog opens on first use; Ctrl+click asks again</td></tr>
</table>
<h3>5. Reference images</h3>
<table cellpadding="3">
<tr><td><b>Add</b></td><td>File &gt; Add reference images... (Ctrl+R): a front and a side photo as two planes crossing at the origin; the References panel sets opacity, offset, height, centre, flip and draw behind the model; View &gt; Reference lens (Alt+L) for a near-orthographic match</td></tr>
<tr><td><b>New figure</b></td><td>File &gt; New figure (Ctrl+Alt+N): a standing 1680 mm blockout, head to feet as separate parts (limb pairs mirrored), 1 mm lattice, Mirror X in the world frame, reference planes at 1680 mm with the feet on the floor</td></tr>
<tr><td><b>Merge</b></td><td>select the parts (Shift+click), Ctrl+J: they become one object, strokes cross the joints, every sculpt layer is kept (Ctrl+Shift+J asks sharp / smooth k). Right-click &gt; Merge does the same</td></tr>
<tr><td><b>Right-click</b></td><td>a click (no drag) opens the action menu: merge, subtract / intersect, duplicate (Ctrl+D), mirror-duplicate across x = 0 (Ctrl+Shift+D), rename (F2), hide (H), isolate (/), show all (Shift+H), mask clear (Alt+M) / invert (Ctrl+I), add a primitive here</td></tr>
<tr><td><b>New bust</b></td><td>File &gt; New bust (Ctrl+Shift+N): a 240 mm sphere on a 1 mm lattice, Mirror X, brushes up to 80 mm</td></tr>
</table>
<h3>6. Keep your work</h3>
<table cellpadding="3">
<tr><td><b>Undo / redo</b></td><td>Ctrl+Z / Ctrl+Y; click a timeline entry to step to it; Verify replay checks the hash</td></tr>
<tr><td><b>Save project</b></td><td>Ctrl+Shift+S (a .fsproj folder; File &gt; Open checks its hash). Quit / New / Open ask first when there are unsaved changes; an autosave runs every 3 minutes or 20 steps and is offered back after a crash</td></tr>
<tr><td><b>Screenshot</b></td><td>Ctrl+S saves the viewport as a PNG in engine/fieldstudio/screens/</td></tr>
</table>
<p><b>New here?</b> Help &gt; What to try (Shift+F1) is a ten-minute tour of the default sphere and cube.</p>
<p>The very first launch after an update compiles the field kernels (about a minute, shown by the progress bar in the
viewport); after that, start-up takes a few seconds. This panel is Help &gt; Quick start (F1).</p>
"""


def _group_job(eng, oids, names):
    """fs-shell-6 11-14: Group (+ the pair's names by her side) as ONE undo step; -> True: the scene changed"""
    from ..core import SetObjectProps
    from ..toolset.merge import GroupObjects
    cmds = [SetObjectProps(int(i), {'name': n}) for i, n in (names or {}).items()] + [GroupObjects(oids)]
    eng._push_group(cmds, 'Group' + (' as a pair' if names else ''))
    eng._select = [int(o) for o in oids]
    return True


def side_names(oids, objs):
    """fs-shell-6 11-14 (the eye workflow): two objects on opposite sides of her midline (world x = 0) -> {oid:
    '<name> (her right)' / '(her left)'}; her right is x < 0, on the screen's left in the Front view (which looks down
    -z with +x to the right). None when the pair does not straddle the midline (names are left as they are)"""
    import re
    if len(oids) != 2 or not all(i in objs for i in oids):
        return None
    xs = [float(((objs[i].get('transform') or {}).get('translate') or [0, 0, 0])[0]) for i in oids]
    if xs[0] * xs[1] >= 0:
        return None
    stem = re.sub(r'\s*(\(her (left|right)\)|copy|mirror|\.?\d+)$', '', str(objs[oids[0]].get('name') or 'Part'),
                  flags=re.I).strip() or 'Part'
    return {i: '%s (her %s)' % (stem, 'right' if x < 0 else 'left') for i, x in zip(oids, xs)}
