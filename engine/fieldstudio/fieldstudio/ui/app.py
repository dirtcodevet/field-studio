"""Start the Field Studio window (or the self-test)."""
import sys

from PySide6.QtWidgets import QApplication

from . import theme
from .engine import Engine
from .main_window import MainWindow


def default_scene(engine):
    """V1.3: something to play with at launch: a sphere and a rounded cube side by side, framed."""
    engine.post('add', {'kind': 'sphere', 'params': {'radius': 0.07}, 'name': 'Sphere', 'translate': [-0.09, 0.0, 0.0],
                        'colour': {'albedo': [0.62, 0.52, 0.45], 'variation': 0.0}})
    engine.post('add', {'kind': 'box', 'params': {'size': [0.12, 0.12, 0.12], 'round': 0.008}, 'name': 'Cube',
                        'translate': [0.09, 0.0, 0.0], 'colour': {'albedo': [0.45, 0.52, 0.62], 'variation': 0.0}})
    engine.post('frame', [])


def run(args):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName('Field Studio')
    theme.apply(app)
    engine = Engine()
    win = MainWindow(engine)
    engine.start()
    try:                                       # fs-shell-5 11-14: the freeze record into the session log (run_field_studio)
        from . import freeze_log as FL
        if FL.SESSION is not None and FL.SESSION[0] is not None:
            FL.install(win, FL.SESSION[0], jsonl=(FL.SESSION[1] + '.freezes.jsonl') if FL.SESSION[1] else None)
    except Exception as e:
        print('[freeze] not installed: %r' % (e,))
    if getattr(args, 'monkey', None) is not None:
        from . import monkey
        return monkey.run(app, win, engine, args)
    if getattr(args, 'selftest', False):
        from . import selftest
        return selftest.run(app, win, engine, args)
    win.show()
    win.activateWindow()
    win.viewport.setFocus()                    # hand1b: the keys go to the view at launch (no panel box has focus)
    from PySide6.QtCore import QTimer
    QTimer.singleShot(300, lambda: win.show_quick_start(at_launch=True))
    refs = (getattr(args, 'ref_front', None), getattr(args, 'ref_side', None), getattr(args, 'ref_height_mm', None),
            bool(getattr(args, 'ref_lens', False)))
    if getattr(args, 'open', None):
        win.project_path = win.project_target(args.open)
        if refs[0] or refs[1]:
            win.pending_refs = refs            # applied after the project's own references (on_project_loaded)
        engine.post('open', args.open)
    else:
        if getattr(args, 'bust', False):
            win.new_bust()
        else:
            win._mark_clean_at = 2             # hand2f: the start scene is not unsaved work
            default_scene(engine)
        if refs[0] or refs[1]:
            QTimer.singleShot(400, lambda: win.load_refs(*refs))
            if getattr(args, 'rig_auto', False):
                # fs-shell: Auto-fit the rig on the loaded photos (crown, chin, centre), the typed height or 230 mm
                QTimer.singleShot(700, lambda: win.refpanel.rig_autofit_apply(refs[2]))
    if getattr(args, 'refs_json', None):
        import json
        import os
        with open(args.refs_json, encoding='utf-8') as f:
            staged = json.load(f)

        def stage():
            missing = win.refpanel.apply_dict(staged, base=os.path.dirname(os.path.abspath(args.refs_json)))
            win.docks['References'].raise_()
            if staged.get('lens'):
                win.refpanel.set_lens(True)
            win.status.showMessage('Staged references loaded from %s%s' % (
                args.refs_json, ('; missing: %s' % missing) if missing else ''), 15000)
        QTimer.singleShot(600, stage)
    rc = app.exec()
    engine.stop()
    return rc
