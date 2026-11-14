"""One pattern for every tool's settings (fs-shell-5, 2026-11-14; design: the three primary
settings, radius, strength and softness, quick and easy to adjust on any tool, the Pencil included).

The same keys and drags, whatever the tool in hand, step its three primary settings in order (fs-tools' registry names
them: tools/registry.py ToolSpec.quick, size-like, amount-like, softness-like):
    slot 0   [  ]          or hold F over the view and drag
    slot 1   -  =          or Shift+F drag
    slot 2   Shift+[  ]    or Ctrl+F drag
A brush (today's or a registry brush row) steps Radius / Strength / Softness-Hardness on the brush in hand (ToolPanel);
another registry tool steps its own settings (ui/toolreg.py reg_values, remembered per tool): Cut its length, shape size,
type; Shapes its size, kind, wall; Copies its count, mode, vary; Connect its size, twist, profile; and so on. Each change
shows on the cursor (viewport.flash_adjust / the drag read-out) and in the strip; a tool with fewer than three leaves the
other keys doing nothing for it. describe(win) gives the quick keys card and the strip what each slot is now."""
import math

SLOT_KEYS = (('Sculpt/Brush radius smaller', 'Sculpt/Brush radius larger', 'F'),
             ('Sculpt/Brush strength lower', 'Sculpt/Brush strength higher', 'Shift+F'),
             ('Sculpt/Brush softer (softness / hardness)', 'Sculpt/Brush harder (softness / hardness)', 'Ctrl+F'))
SHARED = ('radius', 'strength', 'hardness', 'smoothing')
BRUSH_LABEL = {'radius': 'Size', 'strength': 'Strength', 'hardness': 'Softness', 'smoothing': 'Smoothing'}


def _reg(win):
    from . import toolreg as TR
    cur = TR.current(win)
    if cur is None:
        return None, None
    return cur, TR.spec(cur[0])


def own_slots(win):
    """-> [(Quick, Setting)] for a registry tool that is not a brush in hand, else None (the brush path)"""
    from . import toolreg as TR
    cur, sp = _reg(win)
    if cur is None or sp is None or TR.is_brush(win):
        return None
    qs = tuple(getattr(sp, 'quick', ()) or ())
    if not qs:
        return []
    own = {s.key: s for s in TR.own_settings(win)}
    out = []
    for q in qs:
        s = own.get(q.key)
        if s is not None:
            out.append((q, s))
        else:
            out.append((q, None))           # a shared key on a non-brush tool (e.g. Transform's smoothing)
    return out


def _fmt(s, v):
    if s is None:
        return ''
    if s.kind == 'choice' or s.choices:
        return str(v)
    if s.kind == 'toggle':
        return 'on' if v else 'off'
    if isinstance(s.default, int) and not isinstance(s.default, bool):
        return '%d' % int(v)
    if s.unit == 'm':
        return '%.1f mm' % (float(v) * 1000.0)
    if s.unit in ('deg', '°'):
        return '%.0f deg' % float(v)
    return ('%.2f %s' % (float(v), s.unit)).strip()


def _stepped(q, s, v, d):
    """the next value of setting s from v, one step in direction d (+1 / -1)"""
    lo, hi = (tuple(s.range) if s.range else (None, None))
    if q.mode == 'choice' or s.choices:
        ch = list(s.choices or ())
        if not ch:
            return v
        i = ch.index(v) if v in ch else 0
        return ch[(i + d) % len(ch)]
    if s.kind == 'toggle':
        return not bool(v)
    v = float(v if v is not None else (s.default or 0.0))
    if q.mode == 'relative':
        nv = v * (1.0 + q.step) if d > 0 else v / (1.0 + q.step)
        if v == 0.0 and lo is not None and hi is not None:
            nv = (hi - lo) * 0.05 * (1 if d > 0 else 0)
    elif q.mode == 'range' and lo is not None:
        nv = v + d * q.step * (hi - lo)
    elif q.mode == 'count':
        nv = v + d * (q.step or 1)
    else:                                     # 'add'
        nv = v + d * q.step
    if lo is not None:
        nv = min(max(nv, lo), hi)
    if isinstance(s.default, int) and not isinstance(s.default, bool):
        nv = int(round(nv))
    return nv


def step(win, slot, d):
    """the key path: slot 0-2, d +1 / -1. -> True when a registry tool's own setting changed (False: the brush path ran
    or nothing to do)"""
    sl = own_slots(win)
    if sl is None:
        tp = win.tools
        if slot == 0:
            tp.scale_radius(1.15 if d > 0 else 1 / 1.15)
        elif slot == 1:
            tp.step_strength(0.05 * d)
        else:
            tp.step_hardness(0.1 * d)
        _flash(win, ('radius', 'strength', 'hardness')[slot])
        return False
    if slot >= len(sl) or sl[slot][1] is None:
        return False
    q, s = sl[slot]
    from . import toolreg as TR
    TR.set_value(win, s, _stepped(q, s, TR.value(win, s), d))
    _refresh_strip(win)
    _flash(win, ('quick', slot))
    return True


def drag_value(win, slot, v0, dx):
    """the drag path for a registry tool's own setting: the value dx px from where the drag began"""
    sl = own_slots(win)
    if not sl or slot >= len(sl) or sl[slot][1] is None:
        return None
    q, s = sl[slot]
    if q.mode == 'choice' or s.choices or s.kind == 'toggle':
        v = v0
        n = int(dx / 40.0)
        for _ in range(abs(n)):
            v = _stepped(q, s, v, 1 if n > 0 else -1)
        return v
    lo, hi = (tuple(s.range) if s.range else (None, None))
    v0 = float(v0)
    if q.mode == 'relative':
        nv = v0 * math.exp(dx / 200.0) if v0 > 0 else ((hi - lo) * dx / 600.0 if hi is not None else dx / 600.0)
    elif q.mode == 'range' and lo is not None:
        nv = v0 + dx / 300.0 * (hi - lo)
    elif q.mode == 'count':
        nv = v0 + dx / 25.0
    else:
        nv = v0 + dx / 20.0 * q.step
    if lo is not None:
        nv = min(max(nv, lo), hi)
    if isinstance(s.default, int) and not isinstance(s.default, bool):
        nv = int(round(nv))
    return nv


def set_slot(win, slot, v):
    sl = own_slots(win)
    if not sl or slot >= len(sl) or sl[slot][1] is None:
        return
    from . import toolreg as TR
    TR.set_value(win, sl[slot][1], v)
    _refresh_strip(win)


def slot_value(win, slot):
    sl = own_slots(win)
    if not sl or slot >= len(sl) or sl[slot][1] is None:
        return None
    from . import toolreg as TR
    return TR.value(win, sl[slot][1])


def readout(win, slot):
    """'Length 120.0 mm' for the cursor"""
    sl = own_slots(win)
    if not sl or slot >= len(sl) or sl[slot][1] is None:
        return ''
    from . import toolreg as TR
    s = sl[slot][1]
    return '%s %s' % (s.label, _fmt(s, TR.value(win, s)))


def describe(win):
    """-> (tool name or None for a brush, [(label, slot)]) for the quick keys card and the strip"""
    sl = own_slots(win)
    if sl is None:
        return None, [(BRUSH_LABEL[k], i) for i, k in enumerate(('radius', 'strength', 'hardness'))]
    from . import toolreg as TR
    name = TR.card(win)[0]
    return name, [((s.label if s is not None else BRUSH_LABEL.get(q.key, q.key.capitalize())), i)
                  for i, (q, s) in enumerate(sl)]


def _flash(win, kind):
    f = getattr(win.viewport, 'flash_adjust', None)
    if f is not None:
        f(kind)


def _refresh_strip(win):
    ts = getattr(win, 'tool_strip', None)
    if ts is not None:
        ts._sig = None                    # its own boxes show the new value at the next sync
        ts.sync()
