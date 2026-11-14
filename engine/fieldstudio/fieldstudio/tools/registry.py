"""The tool registry (fs-tools, 11-14): the one list the window builds its four columns and its tool strip from.

The contract with the window (agreed 11-14): the window (ui/*) hard-wires no tool; it
reads TOOLS. Each ToolSpec says what the tool is, as the design describes it
(engine/fieldstudio/design/TOOLSET_DESIGN_2026-11-13.md, approved 11-14), which shared controls it shows, its own
settings (at most three), how it runs, and its jobs. Settings the window sends are SI: radius (m), strength (the tool's
unit as a float), hardness (0-1), smoothing (m), foot ('soft'|'even'|'sharp'), reach ('near'|'whole'|'through'),
stop_contact (bool) + gap (m), plus the tool's own keys.

run:
- 'object'  a click tool: the window posts engine.post(job, *args) (toolset/jobs.py registers the jobs);
- 'move'    an anchored or held move on the surface: press / drag / release jobs (job_press, job_drag, job_release),
            with live cues from cues(engine, state);
- 'brush'   today's stroke_begin / stroke_move / stroke_end with the settings dict (fs-core's brush path).
status: 'ready' (proven), 'building' (usable, its proof not all in), 'planned' (shown greyed, says when).

key_hint: the approved mock's keys (INTERFACE_DESIGN; fs-shell-2's keymap), the window's keymap wins.
Data only: importing this file imports nothing heavy.
"""
from dataclasses import dataclass, field, replace as _replace
from typing import Callable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    kind: str                         # 'choice' | 'number' | 'toggle'
    choices: Tuple = ()
    range: Tuple = ()                 # (lo, hi) for a number
    unit: str = ''
    default: object = None
    says: str = ''


@dataclass(frozen=True)
class Quick:
    """one of a tool's primary settings for the window's quick pattern (design 11-14: any tool, the Pencil
    included, has its settings adjusted quickly and easily; the keys [ ] , - = , Shift+[ ] and
    F / Shift+F / Ctrl+F drags step a tool's quick settings in order: size-like, amount-like, softness-like)"""
    key: str                          # a shared key ('radius' | 'strength' | 'hardness' | 'smoothing') or one of `own`
    step: float = 0.1                 # by mode (below)
    mode: str = 'relative'            # 'relative': x (1 +- step) of the current value (sizes, depths, rates);
                                      # 'range': +- step x (hi - lo) of the setting's range (0-1 values: hardness);
                                      # 'add': +- step in its unit (values that start at 0: offsets, blends, twist);
                                      # 'count': +- step (whole numbers); 'choice': the next / previous choice


Q_SIZE, Q_AMOUNT, Q_SOFT = Quick('radius'), Quick('strength'), Quick('hardness', 0.1, 'range')
BRUSH_QUICK = (Q_SIZE, Q_AMOUNT, Q_SOFT)   # every brush: Radius, Strength, Hardness (shown as softness where a tool says so)


@dataclass(frozen=True)
class ToolSpec:
    key: str
    category: str                     # 'add' | 'manipulate' | 'build' | 'corrective'
    name: str
    key_hint: str
    kind: str                         # e.g. 'FLAT · anchored', 'FOLLOWS THE SURFACE · held', 'OBJECTS · click'
    says: str
    glyph: str
    shared: Tuple[str, ...] = ()      # subset of ('radius', 'strength', 'hardness', 'smoothing')
    strength_unit: Optional[Callable] = None
    meanings: Tuple[Tuple[str, str], ...] = ()
    reach: Optional[str] = None       # None, or the default 'near' | 'whole' | 'through'
    stop_contact: bool = False
    own: Tuple[Setting, ...] = ()
    run: str = 'object'               # 'object' | 'move' | 'brush'
    action: Tuple[str, str] = ('', '')   # object tools: (button text, key)
    jobs: Tuple[str, ...] = ()        # the engine jobs it posts (toolset/jobs.py)
    status: str = 'planned'
    profile: Optional[Callable] = None   # (values dict) -> (x, y) samples, when not the standard profile
    brush: Tuple[str, ...] = ()       # held brush tools: the engine's stroke tool ids (core/brush.FT_TOOLS), by 'side'
    quick: Tuple[Quick, ...] = ()     # the primary settings, in order (size-like, amount-like, softness-like); see Quick


def _mm(v):
    return '%.1f mm' % (v * 1e3)


def _per_drag(v):
    return '1 mm drag = %.1f mm' % v


def _mm_s(v):
    return '%.1f mm/s' % (v * 1e3)


def _pct_s(v):
    return '%d %% a second' % round(v * 100)


def _shape_profile(values):
    from ..warp.profile import profile_curve, PEAK_SHAPE
    return profile_curve(values.get('radius', 0.01), values.get('hardness', 0.0), values.get('smoothing', 0.0),
                         values.get('foot', 'soft'), PEAK_SHAPE)


def _swell_profile(values):
    from ..warp.profile import profile_curve, PEAK_SWELL
    return profile_curve(values.get('radius', 0.01), values.get('hardness', 0.0), values.get('smoothing', 0.0),
                         values.get('foot', 'soft'), PEAK_SWELL)


SHARED_MEANING = {
    'radius': 'how wide the full effect is',
    'strength': 'how much, in the unit shown',
    'hardness': 'the shape of the peak: 0 soft and domed, 100 the sharpest',
    'smoothing': 'how the change meets the untouched surface beyond the radius',
}

TOOLS = [
    # ---- Add Objects --------------------------------------------------------------------------------------------------
    ToolSpec('add', 'add', 'Shape library', 'A', 'OBJECTS · click',
             'Adds a solid shape on the surface under the cursor; Hollow makes it a shell',
             'add', own=(Setting('hollow', 'Hollow / Shell', 'toggle', default=False,
                                 says='Hollows the whole shape inward to the wall you set; the outside stays'),
                         Setting('wall', 'Wall', 'number', range=(0.0002, 0.2), unit='m', default=0.005,
                                 says='The wall thickness, measured inward from the outside'),
                         Setting('size', 'Size', 'number', range=(0.001, 5.0), unit='m', default=0.05),
                         Setting('kind', 'Shape', 'choice',
                                 choices=('sphere', 'box', 'cylinder', 'capsule', 'cone', 'torus', 'ellipsoid'),
                                 default='sphere', says='The library shape; Size scales its proportions')),
             run='object', action=('Add', 'A'), jobs=('tools.add_at', 'tools.add', 'tools.hollow', 'tools.rehollow'),
             status='building'),
    ToolSpec('copies', 'add', 'Duplicate / Mirror / Array', 'Ctrl+D', 'OBJECTS · click',
             'Copies the selection: once, mirrored, or many along a line, axis or path',
             'copies', own=(Setting('mode', 'Mode', 'choice', choices=('duplicate', 'mirror', 'array'),
                                    default='duplicate'),
                            Setting('count', 'Count', 'number', range=(2, 200), default=5),
                            Setting('vary', 'Vary (seed)', 'number', range=(0, 9999), default=0,
                                    says='Each copy a little different: size and turn from the seed (0 = identical)')),
             run='object', action=('Copy', 'Ctrl+D'), jobs=('tools.duplicate', 'tools.mirror', 'tools.array'),
             status='building'),
    # ---- Manipulate ---------------------------------------------------------------------------------------------------
    ToolSpec('transform', 'manipulate', 'Transform', 'W', 'OBJECTS, MASSES, REGIONS · anchored',
             'Moves, turns, scales or bends an object or a masked region, its sculpting with it',
             'transform', shared=('smoothing',), reach='whole', stop_contact=True,
             own=(Setting('pivot', 'Pivot', 'choice', choices=('centre', 'surface point', 'border', 'joint'),
                          default='centre'),
                  Setting('deform', 'Deform', 'choice', choices=('none', 'bend', 'twist', 'taper', 'skew'), default='none')),
             run='move', jobs=('tools.region_move', 'tools.deform'), status='building'),
    ToolSpec('shape', 'manipulate', 'Shape', 'E', 'FLAT · anchored',
             'Grabs everything the lens sees and pulls it, or pushes it in; drag back and it returns',
             'shape', shared=('radius', 'strength', 'hardness', 'smoothing'), strength_unit=_per_drag,
             meanings=(('strength', 'how far each mm of drag moves the surface (1 = it follows the cursor)'),
                       ('hardness', 'how sharp a point it stretches to: soft = a dome, hard = a pin-point')),
             reach='near', stop_contact=True,
             own=(Setting('direction', 'Direction', 'choice', choices=('surface', 'view', 'axis', 'picked'),
                          default='surface', says='The lens faces the surface where you press'),
                  Setting('keep_volume', 'Keep volume', 'toggle', default=False,
                          says='On: it narrows as it lengthens, like clay')),
             run='move', jobs=('tools.shape_press', 'tools.shape_drag', 'tools.shape_release'),
             status='planned', profile=_shape_profile),         # the warp engine is in; live needs the card path (fs-core)
    ToolSpec('place', 'manipulate', 'Place on surface', 'P', 'FOLLOWS THE SURFACE · anchored',
             'The object rides the surface under the cursor, sitting on it; click drops it',
             'place', own=(Setting('align', 'Align', 'choice', choices=('surface', 'upright', 'keep'),
                                   default='surface'),
                           Setting('offset', 'Offset', 'number', range=(-0.05, 0.05), unit='m', default=0.0),
                           Setting('settle', 'Settle', 'choice', choices=('off', 'once', 'simulate'), default='off',
                                   says='Physics settle needs rigid contact: coming')),
             run='move', jobs=('tools.place_press', 'tools.place_drag', 'tools.place_release'), status='building'),
    ToolSpec('mask', 'manipulate', 'Mask / Protect', 'B', 'FOLLOWS THE SURFACE · held',
             'Says what tools may not change: paint, protect a part, or by thickness',
             'mask', shared=('radius', 'strength', 'hardness', 'smoothing'), strength_unit=_pct_s, reach='near',
             own=(Setting('mode', 'Mode', 'choice', choices=('paint', 'protect part', 'by thickness', 'by distance', 'box',
                                                              'lasso'), default='paint'),
                  Setting('below', 'Thinner than', 'number', range=(0.0001, 0.1), unit='m', default=0.002,
                          says='By thickness: every surface thinner than this is masked')),
             run='brush', jobs=('tools.mask_by', 'tools.mask_op'), status='building'),   # by thickness / distance and
             # Grow / Shrink / Soften / Invert / Clear are in (toolset/mask.py); the paint brush is fs-core's path
    # ---- Building Tools -----------------------------------------------------------------------------------------------
    ToolSpec('pencil', 'build', 'Pencil', 'L', 'FOLLOWS THE SURFACE · line',
             'Carves a precise line, raises a ridge, or draws colour, exactly under the tip',
             'pencil', shared=('radius', 'strength', 'hardness', 'smoothing'), strength_unit=_mm, reach='near',
             own=(Setting('mode', 'Mode', 'choice', choices=('carve', 'ridge', 'draw'), default='carve'),
                  Setting('tip', 'Tip', 'choice', choices=('point', 'round', 'square'), default='point')),
             run='brush', status='planned'),
    ToolSpec('swell', 'build', 'Swell / Settle', 'I', 'FOLLOW or FLAT · held',
             'Swells or settles the surface under the cursor while held; it can stop at contact',
             'swell', shared=('radius', 'strength', 'hardness', 'smoothing'), strength_unit=_mm_s, reach='near',
             stop_contact=True,
             own=(Setting('follow', 'Follow / Flat', 'choice', choices=('follow', 'flat'), default='follow'),
                  Setting('side', 'Swell / Settle', 'choice', choices=('swell', 'settle'), default='swell')),
             run='brush', status='building', profile=_swell_profile, brush=('ft_swell', 'ft_settle')),   # live on the
             # card since ee9e0f9b / 5d7a938a / f51a573e (gpu/sculpt.cu ft_dab), its four gates in (11-14)
    ToolSpec('cut', 'build', 'Cut', 'X', 'FLAT (line) · click',
             "Cuts a straight line away from your view, or takes a shape's volume out",
             'cut', own=(Setting('type', 'Type', 'choice', choices=('line', 'shape'), default='line'),
                         Setting('length', 'Length', 'number', range=(0.0001, 5.0), unit='m', default=0.05,
                                 says='How deep the line cut goes, from the first surface it meets'),
                         Setting('shape', 'Shape', 'choice',
                                 choices=('box', 'sphere', 'cylinder', 'cone', 'capsule', 'torus'), default='box'),
                         Setting('shape_size', 'Shape size', 'number', range=(0.001, 1.0), unit='m', default=0.02,
                                 says="The cutting shape's size")),
             run='object', action=('Cut away', 'Enter'),
             jobs=('tools.cut_line', 'tools.cut_shape', 'tools.container_delete', 'tools.container_place'),
             status='building'),
    ToolSpec('merge', 'build', 'Merge', 'Ctrl+J', 'OBJECTS · click',
             'Joins two touching objects into one; objects apart can be grouped instead',
             'merge', own=(Setting('blend', 'Blend at contact', 'number', range=(0.0, 0.05), unit='m', default=0.0,
                                   says='0 = an exact join; more = a smooth fillet up to that size where they touch'),),
             run='object', action=('Merge', 'Ctrl+J'), jobs=('tools.merge', 'tools.group', 'tools.ungroup', 'tools.members'),
             status='building'),
    ToolSpec('connect', 'build', 'Connect', 'K', 'FLAT (bridge) · click',
             'Builds a connector between two surfaces, in the profile you choose',
             'connect', own=(Setting('profile', 'Profile', 'choice', choices=('circle', 'square', 'rounded'),
                                     default='circle'),
                             Setting('twist', 'Twist', 'number', range=(-720.0, 720.0), unit='deg', default=0.0),
                             Setting('size', 'Profile size', 'number', range=(0.0005, 0.5), unit='m', default=0.005,
                                     says="The profile's radius or half side")),
             run='object', action=('Connect', 'K'), jobs=('tools.connect',), status='building'),
    # ---- Corrective ---------------------------------------------------------------------------------------------------
    ToolSpec('smooth', 'corrective', 'Smooth', 'S', 'FOLLOWS THE SURFACE · held',
             'Takes bumps out without changing the form', 'smooth',
             shared=('radius', 'strength', 'hardness', 'smoothing'), strength_unit=_pct_s, reach='near',
             own=(Setting('what', 'Form / Detail', 'choice', choices=('form', 'detail', 'both'), default='both'),),
             run='brush', status='planned'),
    ToolSpec('flatten', 'corrective', 'Flatten', 'T', 'FLAT · held',
             'Raises and lowers the surface onto the plane under the cursor while held', 'flatten',
             shared=('radius', 'strength', 'hardness', 'smoothing'), strength_unit=_pct_s, reach='near',
             run='brush', status='building', brush=('ft_flatten',)),
    ToolSpec('blend', 'corrective', 'Blend', 'Shift+B', 'FOLLOWS THE SURFACE · held',
             'Rounds a crease, an edge or a seam into a smooth fillet where you brush', 'blend',
             shared=('radius', 'strength', 'hardness', 'smoothing'), strength_unit=_pct_s, reach='near',
             run='brush', status='planned'),
]

# the primary settings of each tool (11-14; the window's quick pattern): size-like, amount-like, softness-like
QUICK = {
    'add': (Quick('size'), Quick('kind', 0, 'choice'), Quick('wall')),
    'copies': (Quick('count', 1, 'count'), Quick('mode', 0, 'choice'), Quick('vary', 1, 'count')),
    'transform': (Quick('smoothing'), Quick('deform', 0, 'choice'), Quick('pivot', 0, 'choice')),
    'shape': BRUSH_QUICK,
    'place': (Quick('offset', 0.0005, 'add'), Quick('align', 0, 'choice'), Quick('settle', 0, 'choice')),
    'mask': BRUSH_QUICK,
    'pencil': BRUSH_QUICK,            # width (the tip), depth / height, the edge's softness; the tip is its 4th, on the strip
    'swell': BRUSH_QUICK,
    'cut': (Quick('length'), Quick('shape_size'), Quick('type', 0, 'choice')),
    'merge': (Quick('blend', 0.0005, 'add'),),
    'connect': (Quick('size'), Quick('twist', 15.0, 'add'), Quick('profile', 0, 'choice')),
    'smooth': BRUSH_QUICK,
    'flatten': BRUSH_QUICK,
    'blend': BRUSH_QUICK,
}
TOOLS = [_replace(t, quick=QUICK.get(t.key, ())) for t in TOOLS]

BY_KEY = {t.key: t for t in TOOLS}
CATEGORIES = (('add', 'Add Objects'), ('manipulate', 'Manipulate'), ('build', 'Building Tools'),
              ('corrective', 'Corrective'))
