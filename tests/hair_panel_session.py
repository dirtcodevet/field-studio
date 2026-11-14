"""The Hair panel session: the real window (offscreen Qt) grows hair on a sphere, end to end.

  1 open Field Studio                 (process start -> the window warm and idle; the Hair panel installed)
  2 a sphere                          (a 50 mm sphere at a 0.5 mm lattice)
  3 an area                           (the Hair panel's area on the sphere's top-back cap, one sample per area cell,
                                       as the area brush keeps them)
  4 grow                              (the panel's own Generate with fixed settings: the scalp kind, density 60 /cm2,
                                       length 200 mm, seed 4; the engine samples the scene's field, the hair engine
                                       grows and settles the guides on the generator thread, the children follow, the
                                       strands are pushed out of the surface and the groom comes back on the UI thread)
  5 exit cleanly                      (the exit rule, then the process exits with the result's code)

Results: tests/out/hair_<tag>.json (the strand count, the generator's report and timings, hashes of the grown arrays,
engine failures, the panel's install failures) and tests/out/hair_<tag>.npz (the grown strands).

CPU only (no graphics card needed; FIELDSTUDIO_GPU defaults to 0 here):
    FIELDSTUDIO_GPU=0 HAIR_TAG=cpu_smoke python tests/hair_panel_session.py
The run passes when the Hair panel is installed, the groom holds strands (more than 0), the engine reports no failure,
no panel failed to install, and the process exits 0. With the same settings, two runs on one machine grow the same
arrays (the hashes in the JSON)."""
import hashlib
import json
import os
import sys
import tempfile
import time

T_START = time.perf_counter()
FS = os.environ.get('FS_TREE') or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine', 'fieldstudio'))
ROOT = os.path.normpath(os.path.join(FS, '..', '..'))      # the repository: engine/hair is read from here
sys.path.insert(0, FS)
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get('HAIR_TAG', 'run')
OUTDIR = os.path.join(HERE, 'out')
os.makedirs(OUTDIR, exist_ok=True)
TMP = tempfile.mkdtemp(prefix='fs_hair_session_')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QT_QPA_FONTDIR', os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'))
os.environ.setdefault('FIELDSTUDIO_GPU', '0')                # the CPU unless FIELDSTUDIO_GPU is set
os.environ['FS_HAIR_CPU'] = '1'                            # the hair's settle and drawing on the CPU in this test
os.environ['FIELDSTUDIO_SETTINGS'] = os.path.join(TMP, 'settings.ini')
os.environ['FIELDSTUDIO_AUTOSAVE_DIR'] = os.path.join(TMP, 'autosave')
os.environ['FIELDSTUDIO_NO_PROMPT'] = '1'
os.makedirs(os.environ['FIELDSTUDIO_AUTOSAVE_DIR'], exist_ok=True)
import numpy as np
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv[:1])
from fieldstudio.ui import theme
from fieldstudio.ui.engine import Engine
from fieldstudio.ui.main_window import MainWindow
from fieldstudio.ui import hair_ui as HU
from fieldstudio.hair import generate as HG
import fieldstudio
import engine.hair

SETTINGS = dict(HG.defaults('scalp'), density=60.0, length_mm=200.0, seed=4)
OUT = {'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': TAG, 'tree': os.path.dirname(os.path.dirname(fieldstudio.__file__)),
       'engine_hair': os.path.dirname(os.path.abspath(engine.hair.__file__)), 'gpu_env': os.environ.get('FIELDSTUDIO_GPU'),
       'settings': SETTINGS, 'steps': {}, 'checks': {}}
theme.apply(app)
eng = Engine()
win = MainWindow(eng)
win.ask_auto = {}
win.no_prompt = True
eng.start()
win.resize(1400, 900)
win.show()
FAILS, MSGS = [], []
eng.failed.connect(lambda m: FAILS.append(str(m)[:300]))
eng.message.connect(lambda m: MSGS.append(str(m)[:200]))


def pump(sec=0.05):
    t = time.perf_counter() + sec
    while time.perf_counter() < t:
        app.processEvents()
        time.sleep(0.002)


def wait_idle(timeout=300):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        if eng.idle():
            pump(0.03)
            if eng.idle():
                return True
        time.sleep(0.002)
    return False


def step(name, t0, **extra):
    OUT['steps'][name] = dict({'wall_s': round(time.perf_counter() - t0, 2), 'failures_so_far': len(FAILS)}, **extra)


def finish():
    OUT['install_failures'] = [str(x.get('what', x) if isinstance(x, dict) else x)[:200] + ': '
                               + str(x.get('error', '') if isinstance(x, dict) else '')[:200]
                               for x in (getattr(win, 'install_failures', None) or [])]
    OUT['engine_failed'] = FAILS
    OUT['messages_tail'] = MSGS[-12:]
    c = OUT['checks']
    c['hair_panel_installed'] = bool(OUT.get('hair_panel'))
    c['strands_over_0'] = int(OUT.get('strands') or 0) > 0
    c['no_engine_failure'] = not FAILS
    c['no_install_failure'] = not OUT['install_failures']
    c['no_error'] = not OUT.get('error')
    OUT['passed'] = all(c.values())
    OUT['session_s'] = round(time.perf_counter() - T_START, 1)
    json.dump(OUT, open(os.path.join(OUTDIR, 'hair_%s.json' % TAG), 'w'), indent=1, default=str)
    print(json.dumps({k: OUT.get(k) for k in ('passed', 'strands', 'grow_s', 'session_s', 'checks', 'error')}, default=str),
          flush=True)
    from fieldstudio.ui.exit_rule import hard_exit
    hard_exit(0 if OUT['passed'] else 1, eng)


try:
    # ---- 1 open --------------------------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    wait_idle()
    OUT['hair_panel'] = isinstance(getattr(win, 'hair', None), HU.HairLayer)
    step('1_open', t0, hair_panel=OUT['hair_panel'], renderer='card' if getattr(eng, 'use_gpu', False) else 'cpu')
    if not OUT['hair_panel']:
        OUT['error'] = 'the Hair panel is not installed'
        finish()
    # ---- 2 a sphere ----------------------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    R = 0.05
    eng.post('add', {'kind': 'sphere', 'params': {'radius': R}, 'name': 'Sphere', 'h': 0.0005})
    wait_idle()
    objs = list((win.snap or {}).get('objects') or [])
    oid = int(objs[-1]['id'])
    step('2_sphere', t0, objects=len(objs))
    # ---- 3 the area: the sphere's top-back cap in the object's frame (y up), one sample per area cell --------------------
    t0 = time.perf_counter()
    n = 60000
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    th = np.pi * (1 + 5 ** 0.5) * i
    U = np.stack([np.sqrt(1 - z * z) * np.cos(th), z, np.sqrt(1 - z * z) * np.sin(th)], 1)
    U = U[(U[:, 1] > 0.35) & (U[:, 2] < 0.4)]
    Q = U * R
    A = {}
    for kk, q in zip(map(tuple, np.floor(Q / HU.AREA_CELL).astype(np.int64)), Q):
        A.setdefault(kk, q)
    L = win.hair
    L.area[oid] = A
    L.area_target = oid
    step('3_area', t0, samples=len(A))
    # ---- 4 grow --------------------------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    started = bool(L.generate(SETTINGS))
    while time.perf_counter() - t0 < 900:
        app.processEvents()
        if oid in L.grooms or (not L.busy and time.perf_counter() - t0 > 5):
            break
        time.sleep(0.01)
    pump(0.5)
    OUT['grow_s'] = round(time.perf_counter() - t0, 1)
    gr = L.grooms.get(oid)
    step('4_grow', t0, started=started, grown=gr is not None)
    if gr is None:
        OUT['error'] = 'no groom after Generate (busy %s)' % L.busy
        finish()
    arrays = {'P': gr.base['P'], 'RAD': gr.cur['RAD'], 'COL': gr.cur['COL'], 'SEG': gr.cur['SEG']}
    for k in ('SIG', 'SIGP'):
        if k in gr.base:
            arrays[k] = gr.base[k]
    OUT['strands'] = int(gr.n)
    OUT['finite'] = bool(np.isfinite(np.asarray(arrays['P'], np.float64)).all())
    OUT['checks']['points_finite'] = OUT['finite']
    OUT['report'] = {k: v for k, v in (gr.report or {}).items() if k != 'settings'}
    OUT['sha256'] = {k: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16] for k, a in arrays.items()}
    np.savez(os.path.join(OUTDIR, 'hair_%s.npz' % TAG), **{k: np.asarray(a) for k, a in arrays.items()})
except Exception:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    OUT['error'] = traceback.format_exc()[-2500:]
# ---- 5 exit: the exit rule (stop launching, synchronise), then the process ends with the result's code ------------------
finish()
