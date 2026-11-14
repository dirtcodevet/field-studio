"""Field Studio V1: field-based 3-D modelling (implicit fields, no polygon mesh), CPU only.

  python run_field_studio.py                 open the window (a sphere to start)
  python run_field_studio.py --open X.fsproj open a project folder (--project is the same)
  --bust                                     start with a fresh bust sphere (1 mm lattice, Mirror X)
  --ref-front F.jpg --ref-side S.jpg         reference planes crossing at the origin (--ref-height-mm, --ref-lens)
  --rig-auto                                 fs-shell: align them with the Reference Rig's Auto-fit
  python run_field_studio.py --selftest      build a demo scene, check it, grab the window to screens/selftest.png
                                             (offscreen unless --visible)
  --selftest-region [DIR]                    V1.2: only the field-region test (a crop of a field version;
                                             not part of this release) - offscreen
  --monkey [SECONDS]                         V1.3: random UI actions offscreen (default 300 s) -> screens/monkey.json
  --threads N                                CPU threads for the field kernels (default: logical CPUs - 2)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# fs-brush 11-12 (a window froze ~17:49-17:54 and its only log, fieldstudio_last_run.log, was emptied by the relaunch
# at 17:56, so the cause could not be named): the FREEZE DIAGNOSTICS. Wherever the log is on (pythonw launches, or
# FIELDSTUDIO_LOG set):
# (1) a per-session log, session_logs/fs_<date>_<time>_<pid>.log beside this file (FIELDSTUDIO_SESSION_LOG_DIR moves
#     it), that a relaunch never erases; the newest SESSION_KEEP are kept; fieldstudio_last_run.log names it;
# (2) a watchdog: the window thread re-arms faulthandler's timer every 250 ms; when it doesn't answer for STALL_S, the
#     timer (a C thread: it runs even while a kernel holds Python's lock) writes every Python thread's stack to the
#     session log, every STALL_S until the window answers; then one line gives the stall's start, end and length.
#     An '[alive]' line each minute gives the clock and the threads' names with the ids the stack dumps print.
# FIELDSTUDIO_WATCHDOG=0 turns (2) off; FIELDSTUDIO_STALL_S sets the threshold (default 2 s).
SESSION_KEEP = 40
STALL_S = float(os.environ.get('FIELDSTUDIO_STALL_S', '2.0'))


def _stamp(t):
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t)) + '.%03d' % int((t % 1) * 1000)


def _open_session_log(last_run, last_run_path):
    """-> the open per-session log file (line-buffered, kept for the whole run), or None if it can't be made"""
    import glob
    import time
    try:
        d = os.environ.get('FIELDSTUDIO_SESSION_LOG_DIR') or os.path.join(HERE, 'session_logs')
        os.makedirs(d, exist_ok=True)
        old = sorted(glob.glob(os.path.join(d, 'fs_*.log')), key=os.path.getmtime)
        for p in old[:max(0, len(old) - (SESSION_KEEP - 1))]:
            try:
                os.remove(p)
            except OSError:
                pass
        path = os.path.join(d, 'fs_%s_%d.log' % (time.strftime('%Y%m%d_%H%M%S'), os.getpid()))
        g = open(path, 'a', encoding='utf-8', buffering=1)
        g.write('[session %s] Field Studio, pid %d, %s\n[session] args: %s\n[session] also: %s\n' % (
            _stamp(time.time()), os.getpid(), sys.version.split()[0], ' '.join(sys.argv[1:]) or '(none)',
            last_run_path))
        last_run.write('[session log] %s\n' % path)
        return g
    except Exception as e:
        try:
            last_run.write('[session log] could not be made: %s\n' % e)
        except Exception:
            pass
        return None


def _install_watchdog(fh):
    """the window thread's heartbeat re-arms faulthandler's timer; see the note at the top of this file"""
    import faulthandler
    import threading
    import time
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    st = {'last': time.time(), 'alive': 0.0}

    def tick():
        now = time.time()
        gap = now - st['last']
        if gap > STALL_S:
            fh.write('[watchdog %s] the window did not answer for %.1f s, from %s to %s (the stacks above, written '
                     'every %.0f s while it was stalled, show what ran)\n' % (
                         _stamp(now), gap, _stamp(st['last']), _stamp(now), STALL_S))
        st['last'] = now
        faulthandler.dump_traceback_later(STALL_S, repeat=True, file=fh)
        ths = tuple(sorted((t.name, t.ident) for t in threading.enumerate() if t.ident))
        if now - st['alive'] >= 60.0 or ths != st.get('ths'):
            st['alive'], st['ths'] = now, ths
            fh.write('[alive %s] threads: %s\n' % (_stamp(now), ', '.join('%s=0x%08x' % t for t in ths)))

    def off():
        faulthandler.cancel_dump_traceback_later()
        fh.write('[watchdog %s] off: the window is closing\n' % _stamp(time.time()))

    timer = QTimer(app)
    timer.setInterval(250)
    timer.timeout.connect(tick)
    timer.start()
    app.aboutToQuit.connect(off)
    st['timer'] = timer
    _install_watchdog.state = st
    fh.write('[watchdog] on: a window stall over %.1f s writes every thread\'s stack here\n' % STALL_S)
    tick()


def main():
    ap = argparse.ArgumentParser(description='Field Studio V1')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--selftest-region', nargs='?', const='', default=None, dest='selftest_region',
                    help='V1.2: run only the field-region test (optional field version dir)')
    ap.add_argument('--visible', action='store_true', help='show the window during --selftest')
    ap.add_argument('--open', '--project', default=None, dest='open', help='open a project folder (.fsproj)')
    ap.add_argument('--bust', action='store_true', help='hand1: start with File > New bust (a sphere, 1 mm lattice)')
    ap.add_argument('--ref-front', default=None, dest='ref_front', help='hand1: front reference image')
    ap.add_argument('--ref-side', default=None, dest='ref_side', help='hand1: side (profile) reference image')
    ap.add_argument('--ref-height-mm', type=float, default=None, dest='ref_height_mm',
                    help='hand1: the full image height in mm (default 300)')
    ap.add_argument('--ref-lens', action='store_true', dest='ref_lens', help='hand1: start with the 5 deg lens')
    ap.add_argument('--rig-auto', action='store_true', dest='rig_auto',
                    help='fs-shell: Auto-fit the Reference Rig on --ref-front / --ref-side (crown-chin = --ref-height-mm '
                         'or 230 mm)')
    ap.add_argument('--refs-json', default=None, dest='refs_json',
                    help='hand1: a staged reference set (the References panel saved as JSON: paths, centres, spans)')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--monkey', nargs='?', type=float, const=300.0, default=None,
                    help='V1.3: random UI actions for SECONDS (default 300), offscreen; screens/monkey.json')
    ap.add_argument('--seed', type=int, default=0, help='--monkey: random seed')
    ap.add_argument('--no-region', action='store_true', dest='no_region', help='--monkey: skip the field region case')
    args = ap.parse_args()
    no_console = sys.stdout is None or sys.stderr is None      # hand1b: pythonw (the desktop shortcut)
    session = None
    if no_console or os.environ.get('FIELDSTUDIO_LOG'):
        # hand1: started by pythonw (no console): errors go to a log file next to the launcher
        log = os.environ.get('FIELDSTUDIO_LOG') or os.path.join(HERE, 'fieldstudio_last_run.log')
        f = open(log, 'w', encoding='utf-8', buffering=1)
        session = _open_session_log(f, log)

        class _Capped:
            """11-09 night: the log stops at LOG_CAP bytes (a failure repeated every idle turn wrote ~100 MB a minute,
            250+ MB on one run); the cap is said once in the log. fs-brush 11-12: everything also goes to the session
            log (g), which a relaunch does not erase"""
            LOG_CAP = int(os.environ.get('FIELDSTUDIO_LOG_CAP_MB', '20')) * 1024 * 1024

            def __init__(self, f, g=None):
                self.f, self.g, self.n, self.full = f, g, 0, False

            def write(self, s):
                if self.full:
                    return len(s)
                self.n += len(s)
                if self.n > self.LOG_CAP:
                    self.full = True
                    s = '\n[log] capped at %d MB; later output dropped\n' % (self.LOG_CAP >> 20)
                if self.g is not None:
                    try:
                        self.g.write(s)
                    except Exception:
                        pass
                self.f.write(s)
                return len(s)

            def flush(self):
                self.f.flush()
                if self.g is not None:
                    try:
                        self.g.flush()
                    except Exception:
                        pass

            def __getattr__(self, k):
                return getattr(self.f, k)
        sys.stdout = sys.stderr = _Capped(f, session)
        import faulthandler
        faulthandler.enable(session or f)       # fs-brush 11-12: a crash's stacks land where a relaunch keeps them
        import traceback

        def hook(t, v, tb):
            traceback.print_exception(t, v, tb, file=sys.stderr)
            sys.stderr.flush()
        sys.excepthook = hook
    if args.threads:
        os.environ['FIELDSTUDIO_THREADS'] = str(args.threads)
    if args.selftest_region is not None:
        args.selftest = True
    if args.monkey is not None:
        args.selftest = True
    if args.selftest and not args.visible:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        if os.name == 'nt':        # the offscreen platform has no font database of its own on Windows
            os.environ.setdefault('QT_QPA_FONTDIR', os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'))
    if session is not None and os.environ.get('FIELDSTUDIO_WATCHDOG', '1') != '0':
        try:
            _install_watchdog(session)
        except Exception as e:                  # the diagnostics never stop the app from opening
            print('[watchdog] not installed: %s' % e, file=sys.stderr)
        # fs-shell-5 11-14 (a test froze with little done; every one was under the 2 s above):
        # the freeze record (ui/freeze_log.py), installed by the app once the window exists: the window or the view
        # stopped for over FIELDSTUDIO_FREEZE_MS (250) is written to this log with what ran
        try:
            from fieldstudio.ui import freeze_log as _FL
            _FL.SESSION = (session, getattr(session, 'name', None))
        except Exception as e:
            print('[freeze] not prepared: %s' % e, file=sys.stderr)
    try:
        from fieldstudio.ui.app import run
        return run(args)
    except Exception:
        import traceback
        msg = traceback.format_exc()
        print(msg, file=sys.stderr)
        # hand1b: the message box only for the desktop shortcut (no console, a real screen); a headless or console
        # run (tests, --monkey, QT_QPA_PLATFORM=offscreen) prints the error and exits, never waiting on a dialog
        if (os.name == 'nt' and no_console and not args.selftest
                and os.environ.get('QT_QPA_PLATFORM', '') not in ('offscreen', 'minimal')):
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, msg[-1800:], 'Field Studio could not start', 0x10)
            except Exception:
                pass
        return 1


if __name__ == '__main__':
    sys.exit(main())
