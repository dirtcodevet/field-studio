"""Units of the job (fs-shell, 2026-11-11; design Appendix D1: the sizes of what is being
worked on easy to see, from boards and a house to stone bricks, 2x4s and blocks).

The field stays in metres. A scene has one display unit (mm, cm, m, in, ft-in), saved with the scene; typed entry
accepts any unit whatever the scene's ("34 in", "1.5 m", "5' 6\"", "1 1/2 in", "-2.5cm"); a bare number is in the
scene's unit. Sizes: "30 x 40 x 50 mm", "3in x 2m", and stock by name from an editable preset list ("2x4" = a nominal
2x4, actually 1.5 x 3.5 in; "2x4x8ft" adds its length; "brick (UK)").

Stock sizes are public standards (common facts, no licence involved, P13): US softwood dimensional lumber (American
Softwood Lumber Standard PS 20: a nominal 1 is 3/4 in, 2 is 1-1/2 in; nominal widths 2 3 4 6 8 10 12 are 1-1/2 2-1/2
3-1/2 5-1/2 7-1/4 9-1/4 11-1/4 in), the US modular brick (7-5/8 x 3-5/8 x 2-1/4 in), the UK brick (215 x 102.5 x 65 mm,
BS EN 771-1 work size), the US concrete block (nominal 8 x 8 x 16, actual 7-5/8 x 7-5/8 x 15-5/8 in)."""
import json
import re

UNITS = {'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'in': 0.0254, 'ft': 0.3048}
SCENE_UNITS = ('mm', 'cm', 'm', 'in', 'ft-in')
ALIASES = {'mm': 'mm', 'millimetre': 'mm', 'millimetres': 'mm', 'millimeter': 'mm', 'millimeters': 'mm',
           'cm': 'cm', 'centimetre': 'cm', 'centimetres': 'cm', 'centimeter': 'cm', 'centimeters': 'cm',
           'm': 'm', 'metre': 'm', 'metres': 'm', 'meter': 'm', 'meters': 'm',
           'in': 'in', 'inch': 'in', 'inches': 'in', '"': 'in', '″': 'in',
           'ft': 'ft', 'foot': 'ft', 'feet': 'ft', "'": 'ft', '′': 'ft'}
# snapping step (m) per scene unit when no grid sets it: a millimetre, or a sixteenth of an inch
SNAP_STEP = {'mm': 0.001, 'cm': 0.001, 'm': 0.001, 'in': 0.0254 / 16, 'ft-in': 0.0254 / 16}

_IN = 0.0254
LUMBER_NOMINAL = {1: 0.75, 2: 1.5, 3: 2.5, 4: 3.5, 5: 4.5, 6: 5.5, 8: 7.25, 10: 9.25, 12: 11.25}   # in, PS 20 dry


def default_stock():
    """the editable stock list: name -> [width, height, (depth)] in metres"""
    out = {}
    for a in (1, 2, 4):
        for b in (2, 3, 4, 6, 8, 10, 12):
            if b < a:
                continue
            out['%dx%d' % (a, b)] = [LUMBER_NOMINAL[a] * _IN, LUMBER_NOMINAL[b] * _IN]
    out['brick (US modular)'] = [7.625 * _IN, 2.25 * _IN, 3.625 * _IN]
    out['brick (UK)'] = [0.215, 0.065, 0.1025]
    out['block 8x8x16'] = [15.625 * _IN, 7.625 * _IN, 7.625 * _IN]
    return out


def stock():
    """the user's stock list (app settings 'stock_presets', JSON name -> metres), else the defaults"""
    try:
        from .main_window import app_settings
        d = json.loads(app_settings().value('stock_presets', '') or '{}')
        if d:
            return {str(k): [float(x) for x in v] for k, v in d.items()}
    except Exception:
        pass
    return default_stock()


_NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s+\d+/\d+|/\d+)?'
_UNIT = r'(?:mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m(?:et(?:er|re)s?)?|in(?:ch(?:es)?)?|ft|foot|feet|["\'′″])'


def _num(s):
    s = s.strip()
    m = re.fullmatch(r'([-+]?)(\d+)\s+(\d+)/(\d+)', s)          # "1 1/2"
    if m:
        v = int(m.group(2)) + int(m.group(3)) / float(m.group(4))
        return -v if m.group(1) == '-' else v
    m = re.fullmatch(r'([-+]?)(\d+)/(\d+)', s)                  # "3/4"
    if m:
        v = int(m.group(2)) / float(m.group(3))
        return -v if m.group(1) == '-' else v
    return float(s)


def parse_length(text, unit='mm'):
    """-> metres, or None. A bare number is in `unit` (a scene unit; 'ft-in' reads a bare number as inches)."""
    t = str(text).strip().lower().replace(',', '.')
    if not t:
        return None
    # feet and inches: 5' 6", 5'6, 5 ft 6 in, 5ft6in, 5 ft
    m = re.fullmatch(r'([-+]?)\s*(\d+(?:\.\d+)?)\s*(?:\'|′|ft|foot|feet)\s*(?:(\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)'
                     r'\s*(?:"|″|in|inch|inches)?)?', t)
    if m:
        v = float(m.group(2)) * UNITS['ft'] + (_num(m.group(3)) * _IN if m.group(3) else 0.0)
        return -v if m.group(1) == '-' else v
    m = re.fullmatch(r'(%s)\s*(%s)?' % (_NUM, _UNIT), t)
    if not m:
        return None
    try:
        v = _num(m.group(1))
    except ValueError:
        return None
    u = ALIASES.get(m.group(2)) if m.group(2) else None
    if u is None:
        u = 'in' if unit == 'ft-in' else unit
    return v * UNITS[u]


def parse_size(text, unit='mm'):
    """-> [metres, ...] (2 or 3 extents) or None: "30 x 40 x 50 mm", "3in x 2m", stock names ("2x4", "brick (UK)"),
    and stock with a length ("2x4x8ft", "2x4 x 96 in")."""
    t = str(text).strip()
    if not t:
        return None
    st = stock()
    low = {k.lower(): v for k, v in st.items()}
    if t.lower() in low:
        return list(low[t.lower()])
    parts = [p.strip() for p in re.split(r'\s*[x×*]\s*', t.lower()) if p.strip()]
    if len(parts) < 2:
        return None
    # a stock section ("2x4") followed by a length
    head = '%sx%s' % (parts[0], parts[1])
    if head in low and len(parts) <= 3:
        out = list(low[head])[:2]
        if len(parts) == 3:
            ln = parse_length(parts[2], 'ft-in' if unit in ('in', 'ft-in') else unit)
            if ln is None:
                return None
            out.append(ln)
        return out
    # plain extents: a unit on the last applies to bare numbers before it ("30 x 40 x 50 mm")
    tail = re.search(r'(%s)\s*$' % _UNIT, parts[-1])
    shared = ALIASES.get(tail.group(1)) if tail else None
    out = []
    for p in parts:
        has_unit = re.search(r'(%s)\s*$' % _UNIT, p) is not None
        v = parse_length(p, unit) if has_unit or shared is None else parse_length(p + ' ' + shared, unit)
        if v is None:
            return None
        out.append(v)
    return out if 2 <= len(out) <= 3 else None


def format_length(m, unit='mm', digits=None):
    """metres -> text in the scene's unit ('ft-in' as 5' 6-1/2")"""
    if m is None:
        return ''
    if unit == 'ft-in':
        neg = m < 0
        inches = abs(m) / _IN
        ft = int(inches // 12)
        rest = inches - 12 * ft
        sixteenths = int(round(rest * 16))
        if sixteenths == 192:
            ft, sixteenths = ft + 1, 0
        whole, frac = divmod(sixteenths, 16)
        fs = ''
        if frac:
            g = 16
            n = frac
            while n % 2 == 0:
                n //= 2
                g //= 2
            fs = '-%d/%d' % (n, g)
        if ft:
            s = "%d' %d%s\"" % (ft, whole, fs)
        else:
            s = ('%d%s"' % (whole, fs)) if whole or not fs else ('%s"' % fs[1:])
        return ('-' if neg else '') + s
    v = m / UNITS[unit]
    if digits is None:
        digits = {'mm': 1, 'cm': 2, 'm': 4, 'in': 3}[unit]
    return '%.*f %s' % (digits, v, unit)


def format_size(ms, unit='mm'):
    return ' x '.join(format_length(v, unit) for v in ms)
