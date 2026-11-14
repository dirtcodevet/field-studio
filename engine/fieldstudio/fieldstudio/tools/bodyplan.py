"""Body-plan templates for measuring (fs-shell, 2026-11-11; design: girths are found automatically in the tool,
regions are not, since automatic regions would not suit animals, dragons, blocks or logs).

The templates are DATA, shared with fs-pose (agreed 11-11): engine/pose/templates/<plan>.json, one per body plan, keyed
by named regions. A region carries fs-pose's joints (their rules in engine/pose/finder.py) and fs-shell's measured
sections; each section names a rule here and its parameters, so either lane edits its own entries without touching the
other's code. A named landmark can anchor a joint, a measure, or both.
Forms without a template keep the profile's own features (measure.profile_features) and the user's own named sections.

Section rules (run on a girth profile whose s rises from the feet toward the head; earlier sections may be named in
later ones' params):
  narrowest_in_span    the narrowest section within params.span (fractions of the profile's length)
  first_wider_than     the first section above params.above whose loop is wider than params.width_m (x scale)
  below_steepest_step  the section just below the steepest girth step between params.from and params.to
  widest_between       the fullest section between params.from and params.to (None: the profile's end)
Each rule also says how its landmark is re-measured at 1 mm (refine): the narrowest, the fullest, or the fold."""
import glob
import json
import os

import numpy as np

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pose', 'templates'))


def _idx(named, key, default):
    return named.get(key, default) if key is not None else default


def r_narrowest_in_span(prof, named, params, scale):
    s, g = np.asarray(prof['s']), np.asarray(prof['tape'])
    a, b = params.get('span', [0.1, 0.75])
    span = s.max() - s.min()
    m = (s > s.min() + a * span) & (s < s.min() + b * span)
    return int(np.argmin(np.where(m, g, np.inf))) if m.any() else None


def r_first_wider_than(prof, named, params, scale):
    s, w = np.asarray(prof['s']), np.asarray(prof['width'])
    i0 = _idx(named, params.get('above'), 0)
    above = np.where((np.arange(len(s)) > i0) & (w > float(params.get('width_m', 0.4)) * scale))[0]
    return int(above[0]) if len(above) else len(s) - 1


def r_below_steepest_step(prof, named, params, scale):
    g = np.asarray(prof['tape'])
    i0 = _idx(named, params.get('from'), 0)
    i1 = _idx(named, params.get('to'), len(g) - 1)
    seg = np.arange(i0, max(i0, i1 - 1))
    if len(seg) < 2:
        return None
    return int(seg[int(np.argmax(np.diff(g)[seg]))])


def r_widest_between(prof, named, params, scale):
    g = np.asarray(prof['tape'])
    i0 = _idx(named, params.get('from'), -1)
    i1 = _idx(named, params.get('to'), len(g))
    idx = np.arange(i0 + 1, i1)
    return int(idx[int(np.argmax(g[idx]))]) if len(idx) else None


RULES = {'narrowest_in_span': (r_narrowest_in_span, 'min'), 'first_wider_than': (r_first_wider_than, None),
         'below_steepest_step': (r_below_steepest_step, 'fold'), 'widest_between': (r_widest_between, 'max')}


def load_templates(directory=TEMPLATE_DIR):
    """-> {'<plan>: <region>': (template, region name)} for every region that has measured sections"""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.json'))):
        try:
            t = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue
        for rname, reg in (t.get('regions') or {}).items():
            secs = [s for s in reg.get('sections', []) if not s.get('hidden')]
            if secs:
                label = '%s %s (%s)' % (t.get('plan', os.path.basename(path)[:-5]), rname,
                                        ', '.join(s['name'] for s in secs))
                out[label] = (t, rname)
    return out


def apply(template, region, prof, scale=1.0):
    """run a region's section rules on a profile -> ({name: profile index}, {name: refine kind})"""
    named, kinds = {}, {}
    for sec in template['regions'][region].get('sections', []):
        fn, kind = RULES[sec['rule']]
        i = fn(prof, named, sec.get('params') or {}, scale)
        if i is None:
            continue
        named[sec['name']] = i
        if not sec.get('hidden'):
            kinds[sec['name']] = kind
    return {k: v for k, v in named.items() if k in kinds}, kinds


def _make(t, rname):
    def run(prof, scale=1.0):
        marks, kinds = apply(t, rname, prof, scale)
        run.kinds = kinds
        return marks
    run.kinds = {}
    return run


TEMPLATES = {label: _make(t, r) for label, (t, r) in load_templates().items()}


def human_torso(prof, scale=1.0):
    """the human template's torso sections (kept for callers of the first version)"""
    t = json.load(open(os.path.join(TEMPLATE_DIR, 'human.json'), encoding='utf-8'))
    return apply(t, 'torso', prof, scale)[0]


def refine(scene, oid, prof, marks, half=0.004, step=0.001, kinds=None):
    """each landmark re-measured at 1 mm around its profile section, as its rule says: 'min' the narrowest, 'max' the
    fullest, 'fold' the section just below the steepest 1 mm step -> {name: girth}"""
    from . import measure as M
    s = np.asarray(prof['s'])
    if kinds is None:
        kinds = {'waist': 'min', 'underbust': 'fold', 'bust': 'max', 'hips': 'max'}
    out = {}
    for name, i in marks.items():
        kind = kinds.get(name, 'max')
        if kind == 'fold' and i + 1 < len(s):
            zs = np.arange(s[i], s[i + 1] + 1e-9, step)
        else:
            zs = np.arange(s[i] - half, s[i] + half + 1e-9, step)
        gs = [g for g in (M.section_at(scene, oid, prof, float(z)) for z in zs) if g is not None]
        if not gs:
            continue
        if kind == 'min':
            out[name] = min(gs, key=lambda g: g['tape'])
        elif kind == 'fold':
            d = [gs[k + 1]['tape'] - gs[k]['tape'] for k in range(len(gs) - 1)]
            out[name] = gs[int(np.argmax(d))] if d else gs[0]
        else:
            out[name] = max(gs, key=lambda g: g['tape'])
    return out
