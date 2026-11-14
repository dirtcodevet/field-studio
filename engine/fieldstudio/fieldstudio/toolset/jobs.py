"""The toolset's engine jobs (the window contract, 11-14).

register(engine) registers every job through Engine.register_job (ui/engine.py, fs-shell's); the window posts them as
engine.post(name, *args) from the tool strip (tools/registry.py lists each tool's jobs). Each job runs on the engine
thread under its lock, pushes one undoable command (or one group of them, one undo step), and returns True when the
scene changed. A refusal is ui.engine.Refused (a status line, not a failure); its details stay in engine.tool_result for
the window to act on (Merge -> "Not touching: Group instead?").

Arguments are SI (m, world coordinates), lists rather than arrays.
"""
from ..core.history import AddObject, RemoveObject, SetObjectProps
from ..core.combine import DuplicateObject
from .merge import JoinObjects, GroupObjects, UngroupObjects, NotTouching, group_members
from .common import Protected as _Protected
from .cut import LineCut, CutAway
from .hollow import HollowObject
from .copies import MirrorAcross, ArrayCopies
from .connect import ConnectObjects


def _refused(msg):
    from ..ui.engine import Refused
    return Refused(msg)


def _result(engine, **kw):
    engine.tool_result = kw
    return kw


def merge(engine, oids, blend=0.0):
    cmd = JoinObjects(oids, blend, seal=True)       # fs-tools-2: the contact sealed (one seamless object)
    try:
        engine._push(cmd, 'Merge')
    except NotTouching as e:
        _result(engine, tool='merge', refused='not_touching', offer='group', oids=list(oids), gaps_mm=cmd.info.get('gaps_mm'))
        raise _refused(str(e))
    _result(engine, tool='merge', oid=int(oids[0]), info=cmd.info)
    engine._select = [int(oids[0])]
    return True


def group(engine, oids):
    gid = engine._push(GroupObjects(oids), 'Group')
    _result(engine, tool='group', group=gid, oids=list(oids))
    return True


def ungroup(engine, oids):
    engine._push(UngroupObjects(oids), 'Ungroup')
    _result(engine, tool='ungroup', oids=list(oids))
    return True


def members(engine, oid):
    """not a change: the ids that select and move with oid (its group)"""
    _result(engine, tool='members', oids=group_members(engine.hist.scene, oid))
    return False


def cut_line(engine, oid, ray_a, ray_b, length, kerf=None, scope='this'):
    """scope 'this' (the default: the object pressed) or 'all' (every object the blade crosses; one undo step)"""
    if scope != 'all':
        cmd = LineCut(oid, ray_a, ray_b, length, kerf)
        try:
            info = engine._push(cmd, 'Cut')
        except _Protected:
            raise                        # the guard says it with its offer (Unprotect)
        except ValueError as e:
            raise _refused(str(e))
        _result(engine, tool='cut', **{k: v for k, v in info.items() if k != 'quad'})
        return True
    done, infos = [], {}
    with engine.lock:
        for o in list(engine.hist.scene.objects):
            if not engine.hist.scene.shown(o):
                continue
            c = LineCut(o.id, ray_a, ray_b, length, kerf)
            if c.blade(engine.hist.scene) is None:
                continue
            try:
                infos[o.id] = engine._push_nolock(c, 'Cut')
                c.ui_label = 'Cut (all objects)'
                done.append(c)
            except ValueError:               # a protected part or one the blade misses: the others are cut
                continue
        if done:
            done[-1].group_n = len(done)
            done[0].group_first_n = len(done)
    if not done:
        raise _refused('The cut line does not cross any object')
    _result(engine, tool='cut', objects={k: {kk: vv for kk, vv in v.items() if kk != 'quad'} for k, v in infos.items()})
    return True


def cut_shape(engine, oid, shape, params, at):
    cmd = CutAway(oid, shape, params, at)
    try:
        pid = engine._push(cmd, 'Cut away')
    except _Protected:
        raise                        # the guard says it with its offer (Unprotect)
    except ValueError as e:
        raise _refused(str(e))
    _result(engine, tool='cut', held=pid, from_oid=int(oid))        # the container holds this piece
    return True


def cut_object(engine, oid, cutter, keep='both'):
    """Cut with another object (it is used up): keep 'both' (the piece to the container), 'rest' = Subtract,
    'piece' = Intersect"""
    from .cut import CutWithObject
    try:
        r = engine._push(CutWithObject(oid, cutter, keep), {'both': 'Cut with object', 'rest': 'Subtract',
                                                           'piece': 'Intersect'}.get(keep, 'Cut'))
    except _Protected:
        raise                        # the guard says it with its offer (Unprotect)
    except ValueError as e:
        raise _refused(str(e))
    _result(engine, tool='cut', held=(r if keep == 'both' else None), oid=int(oid), keep=keep)
    return True


def container_delete(engine, pid):
    engine._push(RemoveObject(pid), 'Delete cut piece')
    _result(engine, tool='cut', held=None)
    return True


def container_place(engine, pid, translate, rotate=None):
    o = engine.hist.scene.get(pid)
    t = dict(o.transform)
    t['translate'] = [float(v) for v in translate]
    if rotate is not None:
        t['rotate'] = [float(v) for v in rotate]
    engine._push(SetObjectProps(pid, {'transform': t}), 'Place cut piece')
    _result(engine, tool='cut', held=pid, placed=True)
    return True


def hollow(engine, oid, wall):
    try:
        info = engine._push(HollowObject(oid, wall), 'Hollow')
    except _Protected:
        raise                        # the guard says it with its offer (Unprotect)
    except ValueError as e:
        raise _refused(str(e))
    _result(engine, tool='hollow', **info)
    return True


rehollow = hollow          # the same command on a hollow object: its cavity filled, then derived again


def add(engine, kind, params, translate, hollow_wall=None, name=None, h=None):
    spec = {'kind': kind, 'params': params, 'translate': [float(v) for v in translate]}
    if name:
        spec['name'] = name
    if h:
        spec['h'] = float(h)
    cmds = [AddObject(spec)]
    with engine.lock:
        o = engine._push_nolock(cmds[0], 'Add %s' % kind)
        if hollow_wall:
            c = HollowObject(o.id, hollow_wall)
            engine._push_nolock(c, 'Hollow')
            cmds.append(c)
            cmds[-1].group_n = len(cmds)
            cmds[0].group_first_n = len(cmds)
    engine._select = [o.id]
    _result(engine, tool='add', oid=o.id)
    return True


def add_at(engine, kind, params, ray, fallback, hollow_wall=None, align='surface', offset=0.0, h=None):
    """Add a shape that lands ON the surface under the cursor (TOOLSET_DESIGN 3.1: "A new shape lands on the surface
    under the cursor, sitting on it, or at the reference centre when nothing is under the cursor"): one undo step"""
    from .place import Placer
    spec = {'kind': kind, 'params': params, 'translate': [float(v) for v in fallback]}
    if h:
        spec['h'] = float(h)
    with engine.lock:
        a = AddObject(spec)
        o = engine._push_nolock(a, 'Add %s' % kind)
        cmds = [a]
        pl = Placer(engine.hist.scene, o.id)
        hit = pl.hit(ray)
        if hit is not None:
            p, n = hit
            R, t = pl.frame(p, n, align, offset)
            c = SetObjectProps(o.id, {'transform': pl.drop(R, t, n, offset)})
            engine._push_nolock(c, 'Place on surface')
            cmds.append(c)
        if hollow_wall:
            c = HollowObject(o.id, hollow_wall)
            engine._push_nolock(c, 'Hollow')
            cmds.append(c)
        for c in cmds:
            c.ui_label = 'Add %s' % kind
        cmds[-1].group_n = len(cmds)
        cmds[0].group_first_n = len(cmds)
    engine._select = [o.id]
    _result(engine, tool='add', oid=o.id, on_surface=hit is not None)
    return True


def mask_by(engine, oid, by='thickness', value=0.002, op='set', other=None):
    from .mask import MaskBy
    info = engine._push(MaskBy(oid, by, value, op, other), 'Mask by %s' % by)
    _result(engine, tool='mask', **info)
    return True


def mask_op(engine, oid, op, amount=0.001):
    from .mask import MaskOp
    engine._push(MaskOp(oid, op, amount), 'Mask %s' % op)
    _result(engine, tool='mask', op=op)
    return True


def duplicate(engine, oid, offset=None):
    M = engine._push(DuplicateObject(oid, offset), 'Duplicate')
    engine._select = [M.id]
    _result(engine, tool='copies', oids=[M.id])
    return True


def mirror(engine, oid, plane='world_x', offset=0.0):
    ids = engine._push(MirrorAcross(oid, plane, offset), 'Mirror')
    _result(engine, tool='copies', oids=ids)
    return True


def flip(engine, oid, axis='x'):
    """Flip in place on the object's own axis through its centre (fs-shell's tools/flip.FlipObject: exact)"""
    from ..tools.flip import FlipObject
    engine._push(FlipObject(oid, axis), 'Flip %s' % axis.upper())
    _result(engine, tool='copies', oids=[int(oid)], flipped=axis)
    return True


def array(engine, oid, kw):
    ids = engine._push(ArrayCopies(oid, **kw), 'Array')
    _result(engine, tool='copies', oids=ids)
    return True


def connect(engine, a, b, p_a, p_b, profile_a, profile_b=None, twist=0.0):
    cmd = ConnectObjects(a, b, p_a, p_b, profile_a, profile_b, twist)
    try:
        cid = engine._push(cmd, 'Connect')
    except _Protected:
        raise                        # the guard says it with its offer (Unprotect)
    except ValueError as e:
        raise _refused(str(e))
    engine._select = [cid]
    _result(engine, tool='connect', oid=cid, info=cmd.info)
    return True


JOBS = {
    'tools.merge': merge, 'tools.group': group, 'tools.ungroup': ungroup, 'tools.members': members,
    'tools.cut_line': cut_line, 'tools.cut_shape': cut_shape, 'tools.container_delete': container_delete,
    'tools.container_place': container_place, 'tools.hollow': hollow, 'tools.rehollow': rehollow, 'tools.add': add,
    'tools.duplicate': duplicate, 'tools.mirror': mirror, 'tools.array': array, 'tools.connect': connect, 'tools.add_at': add_at, 'tools.flip': flip, 'tools.mask_by': mask_by,
    'tools.mask_op': mask_op,
    'tools.cut_object': cut_object,
}


def _guarded(name, fn):
    """a job whose tool meets a protected part says so (Refused, a status message) with the offer to Unprotect, in
    tool_result: refused 'protected', offer 'unprotect' (every tool that changes a part refuses it: common.Protected)"""
    import functools
    from .common import Protected

    @functools.wraps(fn)
    def run(engine, *a, **k):
        try:
            return fn(engine, *a, **k)
        except Protected as e:
            _result(engine, tool=name.split('.')[-1], refused='protected', offer='unprotect')
            raise _refused(str(e))
    return run


def register(engine):
    """register every toolset job on the window's engine (call once at start; fs-shell's line)"""
    from . import moves, place
    jobs = dict(JOBS)
    jobs.update(moves.JOBS)
    jobs.update(place.JOBS)
    for name, fn in jobs.items():
        try:
            engine.register_job(name, _guarded(name, fn))
        except _Protected:
            raise                        # the guard says it with its offer (Unprotect)
        except ValueError:
            pass                      # already registered (a second call)
    engine.tool_result = None
    if getattr(engine, 'gpu_view', None) is not None or getattr(engine, 'use_gpu', False):
        from ..warp import bake as WBK
        WBK.CARD['on'] = True            # warps map their points on the card (the window's own process and card)
    return sorted(jobs)
