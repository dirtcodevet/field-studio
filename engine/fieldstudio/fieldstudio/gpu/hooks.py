"""11-09 (READY 1): the GPU stroke hooks (history.GPU_STROKE / GPU_STROKE_END) as one implementation for the app's
engine (ui/engine.py _install_gpu_sculpt) and the script API (script.py). Moved verbatim from ui/engine.py.

install(gv, scene_ok, emit) -> the GpuSculpt. scene_ok(scene): the hooks act only on that scene (replays, worker
scenes and tests run the CPU path); emit(str): user-facing messages."""
from ..core import history as HIST


def install(gv, scene_ok, emit=lambda m: None):
    from .sculpt import GpuSculpt
    gs = GpuSculpt(gv)

    def hook(scene, cmd, samples):
        if not scene_ok(scene):
            return False                 # replays, worker scenes, tests: the CPU path
        if not gs.accepts(scene, cmd.oid, cmd.settings):
            return False
        if not samples:
            return True                  # a grab press: the stroke is claimed, nothing to apply yet
        return _run(scene, cmd, samples)

    def _run(scene, cmd, samples, flush=False):
        # fs-brush 11-13: flush = the release of an anchored stroke shown through the tracer (gpu/sculpt.py MOVE_VIEW):
        # the lattice is written once, here, with the same grid-overflow handling as any batch
        first = not getattr(cmd, '_gpu_applied', False)
        for attempt in (0, 1):
            try:
                boxes = gs.apply(scene, cmd.oid, cmd.settings, samples, cmd.rec, flush=flush)
                gv.note_dirty(boxes)
                cmd._gpu_applied = True
                return True
            except Exception as e:
                if not isinstance(e, RuntimeError) or 'past the volume grid' not in str(e) or attempt:
                    if first:
                        # fs-brush 11-12: the stroke's first batch failed: it never enters the history, so whatever that
                        # batch already did on the card is undone here (probe5b: two strokes with no history step)
                        try:
                            n_ = gs.abort()
                            gv.note_change()
                            emit('GPU sculpt: the stroke could not start (%s); %d card rows put back'
                                 % (str(e).splitlines()[0][:120], n_))
                        except Exception:
                            pass
                    raise
                st_ = cmd.settings
                anchored = ((st_.get('tool') == 'grab' and st_.get('grab_flow', False)) or
                            (st_.get('tool') == 'stretch' and st_.get('stretch_flow', False)))
                if attempt == 0 and 'past the volume grid' in str(e) and anchored:
                    # fs-brush 11-12 (tk4 card: a block-out's skull-base Stretch overflowed the grid mid-stroke; the
                    # rebuild baked the half-done stretch and the anchored stroke lost its stroke-start field): an
                    # anchored stroke depends only on its total, so the card's work is put back (the stroke-start
                    # field), the grid rebuilt around the reach, and the WHOLE stroke so far applied again at once
                    gs.abort()
                    oid_ = getattr(e, 'oid', None)
                    if oid_ is not None and getattr(e, 'box', None) is not None:
                        gv.mirror.grow_hint[oid_] = e.box
                    for k_, ov in gv.mirror.objs.items():
                        if oid_ is None or k_ == oid_:
                            ov.base_sig = None
                    gv._sync(scene)
                    gs.rebuilds = getattr(gs, 'rebuilds', 0) + 1
                    samples = list(cmd.samples) + list(samples)
                    emit('GPU sculpt: the form grew past its grid; rebuilt, the stroke reapplied (%s)' % e)
                    continue
                if attempt == 0 and 'past the volume grid' in str(e):
                    # r19: the live stroke's changes go into the sculpt layer FIRST (its rows are about to be
                    # freed), then the volumes are rebuilt around the grown form, then the batch again
                    gs.finish(scene, cmd.rec)
                    oid_ = getattr(e, 'oid', None)
                    if oid_ is not None and getattr(e, 'box', None) is not None:
                        gv.mirror.grow_hint[oid_] = e.box
                    for k_, ov in gv.mirror.objs.items():
                        if oid_ is None or k_ == oid_:   # r40: only the object whose grid overflowed
                            ov.base_sig = None
                    gv._sync(scene)
                    gs.rebuilds = getattr(gs, 'rebuilds', 0) + 1
                    emit('GPU sculpt: the form grew past its grid; rebuilt (%s)' % e)
                    continue
                raise
        return False
    HIST.GPU_STROKE = hook

    def end(scene, cmd):
        # 11-09 (the script API's replay check): once per stroke. History.push calls end_stroke() on the previous
        # command ("safe to call any time"), so a GPU stroke's release ran again at the NEXT command's push: a second
        # CPU re-distance of its bricks (beside a protected part: up to 155 um) whose record replaced the first (undo lost
        # one), and replay ran it once - the live and replayed states differed.
        if getattr(cmd, '_gpu_ended', False):
            return
        if scene_ok(scene) and cmd.rec is not None:
            cmd._gpu_ended = True
            if gs.mv_pending(cmd.rec):
                if getattr(gs, 'pw_host', False) and gs.pw_ok(cmd.rec):
                    # fs-brush-2 11-13 (design: the release must never be a pause): the stroke stays in
                    # the view as its formula and its write runs behind the release, a slice per engine turn
                    # (ui/engine.py); _tail stores its result when the write is done
                    gs.pw_begin(scene, cmd, lambda: _tail(scene, cmd))
                    return
                try:
                    _run(scene, cmd, [], flush=True)
                except Exception as e:
                    n_ = gs.abort()
                    gv.note_change()
                    emit('GPU sculpt: the move could not be written (%s); %d card rows put back'
                         % (str(e).splitlines()[0][:120], n_))
            gs.finish(scene, cmd.rec)
            _tail(scene, cmd)

    def _tail(scene, cmd):
        if True:
            res = getattr(gs, '_result', None)
            if res:                       # r42: the log names this result; replay applies it exactly
                from ..core import gpu_results as GR
                rid = GR.new_id()
                GR.put(rid, GR.encode(res, cmd.rec))     # r45: stored as XOR against the stroke's start
                cmd.gpu_result = rid
                gs._result = {}
            # fs-brush 11-11: a stroke whose footprint was re-distanced live on the card (gpu/sculpt.py) needs no release
            # pass; that pass left brick seams at its region's edge and took ~0.4 s of the release wait (M)
            if HIST.GPU_END_REDIST and cmd.settings.get('tool') != 'mask' and not gs.live_redist:
                cmd.cpu_redist_after_gpu(scene)
                gv.note_change()
    HIST.GPU_STROKE_END = end
    from . import scene as _GSC
    HIST.POOL_TRIM = _GSC.trim          # fs-trust 11-11: the card pool's unused blocks back at a stroke's end and a save
    return gs
