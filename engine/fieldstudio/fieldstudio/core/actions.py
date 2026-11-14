"""11-09 (READY 1, the scripted sculpt API): the app's part actions as ONE implementation, called by the UI engine
(ui/engine.py jobs 'protect_masked', 'set_locked', 'set_root', 'check_overlaps') and by the script API (script.py),
so a script does exactly what the window's buttons do. Moved verbatim from ui/engine.py (r55-r63 behaviour unchanged).

Each takes hist (core.history.History) and push(cmd, label) -> the command's result, and returns the message the app
shows (str), plus what it changed where a caller needs it."""
import hashlib
import os
import time

import numpy as np

from .history import AddObject, RemoveObject, SetObjectProps
from .combine import MaskEdit


def _default_push(hist):
    def push(cmd, label=None):
        if label:
            cmd.ui_label = label
        return hist.push(cmd)
    return push


def check_overlaps(hist):
    """Edit > Check overlaps: new surface nodes of every editable object inside a protected part + gap.
    -> (message, total)"""
    from . import collide as CL
    from .scene import is_protected
    t0 = time.perf_counter()
    sc = hist.scene
    res = [(o.name,) + CL.new_overlaps(sc, o) for o in sc.objects if not is_protected(o)]
    tot = sum(r[1] for r in res)
    msg = ('Overlap check (%s; her body still holds copies of her parts until the anatomy is '
           'built): %d surface node(s) inside a protected part + gap%s (%.1f s).' % (
               res[0][2] if res else 'new overlaps only', tot,
               (': ' + ', '.join('%s %d' % (r[0], r[1]) for r in res if r[1])) if tot else '',
               time.perf_counter() - t0))
    return msg, tot


def set_root(hist, pid, push=None):
    """Edit > Set root: the root painted (Mask brush) on the unprotected part pid becomes its root.
    -> (message, ok)"""
    from . import attach as AT, masked_parts as GP
    push = push or _default_push(hist)
    sc = hist.scene
    part = sc.get(pid)
    pts = GP.masked_surface(sc, part)
    if len(pts) < 10:
        return ('Set root refused: paint the root on %s with the Mask brush first (Unprotect it '
                'so the brush reaches it).' % part.name), False
    w, lo, hh = AT.root_lattice(pts, float(part.sculpt.h))
    d = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'FieldStudio', 'part_roots')
    path, sha = AT.save_root(w, lo, hh, d, part.name)
    push(SetObjectProps(pid, {'root': {'file': path, 'sha': sha}}), 'Set root of %s' % part.name)
    hist.push(MaskEdit(pid, 'clear'))
    return ('%s: root set from %d painted surface points. Protect it again to fuse the root into '
            'her body (the rest stays a separate part in contact).' % (part.name, len(pts))), True


def set_locked(hist, oid, lock, push=None):
    """Edit > Unprotect (lock False: editable, shown) / Protect again (lock True: locked; an attached part's root is
    fused into the editable object(s) it touches). -> (message, unprotected: bool). The caller runs check_overlaps
    after a relock (the app posts it as its own job)."""
    push = push or _default_push(hist)
    o = hist.scene.get(oid)
    props = {'locked': bool(lock)}
    if not lock and not o.visible:
        props['visible'] = True
    push(SetObjectProps(oid, props), ('Protect %s' if lock else 'Unprotect %s') % o.name)
    if lock and getattr(o, 'root', None):
        # 11-09: relocking an attached part fuses its root into the editable object(s) it touches (skipped when the
        # part is as it was at its last fusion: the bake is not idempotent in the ramp)
        from . import attach as AT
        from .scene import is_protected
        from ..tools.protected_merge import world_box
        _h = hashlib.sha256(repr(o.base.to_dict()).encode())
        for b_, s_ in sorted(o.sculpt.slots.items()):
            if s_ >= 0:
                _h.update(repr(b_).encode())
                _h.update(hist.scene.store.delta[s_].tobytes())
        psig = _h.hexdigest()[:16]
        if o.root.get('fused_sig') != psig:
            a, b = world_box(o)
            for body in [x for x in hist.scene.objects if not is_protected(x) and x.id != o.id]:
                a2, b2 = world_box(body)
                if np.all(a <= b2) and np.all(a2 <= b):
                    push(AT.FuseRoot(body.id, o.id), 'Fuse root of %s into %s' % (o.name, body.name))
            push(SetObjectProps(oid, {'root': dict(o.root, fused_sig=psig)}), 'Root fused')
    msg = (('%s is protected again: brushes pass through it, merges keep it apart.' if lock else
            '%s is unprotected: sculpt it with any brush; Edit > Protect a part again relocks it.') % o.name)
    return msg, not lock


def protect_masked(hist, oid, name):
    """Edit > Protect the masked part: the surface painted with the Mask brush becomes a protected part (a lattice
    object), the mask cleared. -> (message, changed part names)"""
    from . import masked_parts as GP, fv8base
    t0 = time.perf_counter()
    sc = hist.scene
    o = sc.get(oid)
    node = next((n for n in o.base.walk() if n.kind == 'fv8store'), None)
    if node is not None:
        out_dir = GP.out_dir_for(node.params['source'])
        org = fv8base.load(node.params['source'], node.params.get('source_id') or None)['hdr'][0:3]
    else:
        out_dir = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'FieldStudio',
                               'protected_parts', 'obj%d' % oid)
        org = None
    mine, info, n = GP.protect_masked(sc, o, name, out_dir, org)
    changed, kept = [], []
    for part, d in info.items():
        nm = '%s (protected)' % part
        ex = [x for x in sc.objects if x.name == nm]
        if ex and ex[0].base.params.get('sha') == d['sha']:
            continue
        if ex and (ex[0].locked is False or ex[0].sculpt.slots):
            kept.append(part)
            continue
        for x in ex:
            hist.push(RemoveObject(x.id))
        spec = {'name': nm, 'base': {'kind': 'lattice', 'params': {'file': d['file'], 'sha': d['sha']},
                                     'children': []},
                'translate': list(o.transform['translate']), 'rotate': list(o.transform['rotate']),
                'scale': o.transform['scale'], 'h': float(o.sculpt.h)}
        hist.push(AddObject(spec))
        changed.append(part)
    hist.push(MaskEdit(oid, 'clear'))
    msg = ('Protected part "%s" made from %d marked surface points (%.1f s); parts rebuilt: %s%s. '
           'The mask is cleared: paint the next part, or File > Save.' % (
               name, n, time.perf_counter() - t0, ', '.join(changed) or 'none',
               ('; kept as you edited them: ' + ', '.join(kept)) if kept else ''))
    return msg, changed
