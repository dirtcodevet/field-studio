"""Tool glyphs (fs-shell, 2026-11-11; design 4.1 / 5.2; redrawn 11-12, t50, as icons that were all circles
were hard to sort through): every tool's icon is a cross-section of the form AFTER the tool, drawn as a filled
silhouette with a bold white surface (a silhouette reads at 32 px where a thin line did not), what it was dashed where
that tells the story, its motion in the category colour, and the category as a band on the left edge. The shapes come
from the brush's real falloff (core/brush.falloff_lut, the same LUT the dabs use). Each tool that read like its
neighbour has its own cue: Clay a flat-topped layer with strata, Inflate a fan of normals, Draw one arrow, Elastic a
spring, Grab a sideways drag, Crease a groove cut down, Pinch inward arrows on a ridge, Flatten a plane through a cut
cap. Drawn by our own code (P13). Categories and colours follow the design mockup (design/glyphs/make_glyphs.py)."""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from ..core import brush as BR

CATS = [('Object', (150, 162, 180)), ('Build', (232, 140, 60)), ('Refine', (64, 186, 170)), ('Move', (160, 120, 230)),
        ('Cut & Join', (226, 84, 84)), ('Protect & Select', (80, 140, 245)), ('Contact', (110, 200, 100)),
        ('Paint & Label', (226, 190, 70)), ('Hair', (170, 120, 80)), ('View', (140, 150, 160)), ('File', (140, 150, 160))]
CAT_COL = dict(CATS)
BG = QColor(38, 40, 45)
GREY = QColor(120, 124, 132)
WHITE = QColor(238, 238, 240)

# tool key -> (category, one-line "what it does to the form", search words)
TOOL_INFO = {
    'select': ('Object', 'Click to select; drag on empty space to orbit', 'pick choose orbit'),
    'move': ('Object', 'Moves the whole object', 'translate position place'),
    'rotate': ('Object', 'Turns the whole object', 'turn spin orient'),
    'scale': ('Object', 'Grows or shrinks the whole object (or one axis)', 'size resize stretch axis'),
    'draw': ('Build', 'Raises a soft hill along the normal', 'add raise build bump hill'),
    'clay': ('Build', 'Builds up toward a plane: a flat-topped layer', 'build add plateau layer mass block'),
    'carve': ('Build', 'Sinks a soft hollow along the normal', 'dig cut hollow subtract dent'),
    'inflate': ('Build', 'Swells the form outward evenly', 'grow swell puff fatten volume'),
    'smooth': ('Refine', 'Evens out lumps and noise', 'relax soften blend even lumps'),
    'flatten': ('Refine', 'Pulls the surface to a plane', 'plane level flat planar'),
    'crease': ('Refine', 'Cuts a sharp groove and pinches its edges', 'fold sharp line groove wrinkle crease'),
    'pinch': ('Refine', 'Draws the surface together toward the stroke', 'sharpen tighten ridge edge'),
    'grab': ('Move', 'Drags the form with the mouse', 'move pull push shape block mass'),
    'stretch': ('Move', 'Stretches the form along the drag', 'elongate lengthen pull'),
    'snake': ('Move', 'Pulls a form out along the drag, keeping its thickness', 'snake hook pull tentacle horn arm '
              'extrude drag out'),
    'elastic': ('Move', 'Pulls the form elastically; the far side stays put', 'elastic grab soft pull clay kelvin'),
    'mask': ('Protect & Select', 'Paints a mask that protects the surface', 'protect freeze lock hold'),
    'swell': ('Contact', 'Grows until it touches a protected part', 'contact fill touch grow'),
    'settle': ('Contact', 'Moves the surface onto a part below it', 'contact rest drape onto'),
    'pencil ridge': ('Cut & Join', 'Draws a thin raised line', 'line ridge pencil detail'),
    'pencil valley': ('Cut & Join', 'Draws a thin groove line', 'line groove pencil valley detail'),
    'cut': ('Cut & Join', 'Removes what an outline encloses', 'delete remove trim boolean lasso'),
    'cut + separate': ('Cut & Join', 'Cuts an outline into its own object', 'split separate detach'),
    'splice': ('Cut & Join', 'Joins two faces', 'join weld connect merge'),
    'fill + splice': ('Cut & Join', 'Fills the gap between two faces and joins them', 'fill bridge join'),
}

_cache = {}


def _lut():
    return np.asarray(BR.falloff_lut('smooth'))


def _w(x, r=0.62):
    """the brush falloff at signed x (glyph units -1..1) for a brush of radius r"""
    lut = _lut()
    u = np.clip(np.abs(x) / r, 0.0, 1.0)
    return np.interp(u, np.linspace(0, 1, len(lut)), lut)


XS = np.linspace(-1.0, 1.0, 121)


FORM = QColor(112, 117, 128)      # the form's cut face (clay grey); its surface is the white edge on top
FORM_DARK = QColor(78, 82, 92)    # a part below / the strata in a clay layer
GHOST = QColor(150, 154, 162)     # what the form was (dashed)


class _G:
    """a glyph painter in tile units, designed at 32 px (fs-shell 11-12, t50): x -1..1 across (right of the band), y up
    from the form's baseline; widths in pixels at 32 px, scaled with the size"""

    def __init__(self, p, size, cat, base=0.70):
        self.p, self.s, self.col = p, float(size), QColor(*CAT_COL[cat])
        self.base = base

    def P(self, x, y):
        return QPointF(self.s * (0.13 + 0.415 * (x + 1.0)), self.s * (self.base - 0.62 * y))

    def pw(self, w):
        return max(1.0, w * self.s / 32.0)

    def _path(self, ys, xs):
        path = QPainterPath()
        first = True
        for x, y in zip(xs, ys):
            if not np.isfinite(y):
                first = True
                continue
            q = self.P(x, y)
            if first:
                path.moveTo(q)
                first = False
            else:
                path.lineTo(q)
        return path

    def line(self, ys, col, w, xs=None, dash=False):
        xs = XS if xs is None else np.asarray(xs, float)
        pen = QPen(col, self.pw(w))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        if dash:
            pen.setStyle(Qt.DashLine)
        self.p.setPen(pen)
        self.p.setBrush(Qt.NoBrush)
        self.p.drawPath(self._path(ys, xs))

    def form(self, ys, xs=None, col=None, edge=True):
        """the form's cross-section: filled from its surface down to the tile's foot, the surface a bold white edge;
        NaN gaps split it into pieces (a cut)"""
        xs = XS if xs is None else np.asarray(xs, float)
        ys = np.asarray(ys, float)
        ok = np.isfinite(ys)
        k = 0
        while k < len(xs):
            if not ok[k]:
                k += 1
                continue
            j = k
            while j < len(xs) and ok[j]:
                j += 1
            poly = QPolygonF([self.P(x, y) for x, y in zip(xs[k:j], ys[k:j])])
            poly.append(QPointF(self.P(xs[j - 1], 0).x(), self.s))
            poly.append(QPointF(self.P(xs[k], 0).x(), self.s))
            self.p.setPen(Qt.NoPen)
            self.p.setBrush(col or FORM)
            self.p.drawPolygon(poly)
            k = j
        if edge:
            self.line(ys, WHITE, 2.2, xs)

    def ghost(self, ys, xs=None):
        """what the form was, dashed"""
        self.line(ys, GHOST, 1.3, xs, dash=True)

    def arrow(self, a, b, col=None, w=2.4, head=0.17):
        col = col or self.col
        p, q = self.P(*a), self.P(*b)
        ang = math.atan2(q.y() - p.y(), q.x() - p.x())
        L, Wd = head * self.s, 0.58 * head * self.s
        base = QPointF(q.x() - L * math.cos(ang), q.y() - L * math.sin(ang))
        pen = QPen(col, self.pw(w))
        pen.setCapStyle(Qt.FlatCap)
        self.p.setPen(pen)
        self.p.drawLine(p, base)
        nx, ny = -math.sin(ang), math.cos(ang)
        self.p.setPen(Qt.NoPen)
        self.p.setBrush(col)
        self.p.drawPolygon(QPolygonF([q, QPointF(base.x() + Wd * nx, base.y() + Wd * ny),
                                      QPointF(base.x() - Wd * nx, base.y() - Wd * ny)]))

    def plane(self, y, x0=-1.0, x1=1.0, col=None, w=1.6):
        self.line(np.full(2, y), col or self.col, w, xs=[x0, x1], dash=True)

    def dot(self, x, y, r=2.6, col=None):
        self.p.setPen(Qt.NoPen)
        self.p.setBrush(col or self.col)
        self.p.drawEllipse(self.P(x, y), r * self.s / 32.0, r * self.s / 32.0)


def _arm(g, t0, t1, c, thick):
    """a pulled-out arm of even thickness along a quadratic curve (snake hook), as a filled band"""
    ts = np.linspace(0.0, 1.0, 24)
    P0, P1, P2 = (np.asarray(v, float) for v in (t0, c, t1))
    C = ((1 - ts) ** 2)[:, None] * P0 + (2 * (1 - ts) * ts)[:, None] * P1 + (ts ** 2)[:, None] * P2
    D = np.gradient(C, axis=0)
    N = np.stack([-D[:, 1], D[:, 0]], 1)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    hw = 0.5 * thick * (1.0 - 0.25 * ts)
    L, R = C + N * hw[:, None], C - N * hw[:, None]
    poly = QPolygonF([g.P(*p) for p in L] + [g.P(*p) for p in R[::-1]])
    g.p.setPen(QPen(WHITE, g.pw(2.0)))
    g.p.setBrush(FORM)
    g.p.drawPolygon(poly)


def _spring(g, a, b, turns=4, amp=0.14):
    """a zigzag spring from a to b in the category colour, ending in an arrowhead"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)
    pts = [a]
    for k in range(1, 2 * turns):
        pts.append(a + d * (k / (2 * turns)) * 0.8 + n * amp * (1 if k % 2 else -1))
    pts.append(a + d * 0.8)
    path = QPainterPath()
    path.moveTo(g.P(*pts[0]))
    for q in pts[1:]:
        path.lineTo(g.P(*q))
    pen = QPen(g.col, g.pw(2.0))
    pen.setJoinStyle(Qt.MiterJoin)
    g.p.setPen(pen)
    g.p.setBrush(Qt.NoBrush)
    g.p.drawPath(path)
    g.arrow(tuple(a + d * 0.78), tuple(b))


def _draw(g, key):
    Z = np.zeros_like(XS)
    if key == 'draw':                               # one soft hill, raised along the normal
        g.form(0.78 * _w(XS, 0.5)); g.arrow((0.0, 0.05), (0.0, 1.02))
    elif key == 'carve':                            # one soft hollow (the baseline high in the tile)
        g.base = 0.38
        g.form(-0.78 * _w(XS, 0.5)); g.arrow((0.0, 0.92), (0.0, -0.08))
    elif key == 'clay':                             # a flat-topped layer laid on: square shoulders, strata, no arrow
        top = 0.62
        y = np.minimum(2.4 * _w(XS, 0.62), top)
        g.form(y)
        for h in (0.2, 0.41):
            on = np.where(y > h + 0.05, h, np.nan)
            g.line(on, FORM_DARK, 1.4)
        g.plane(top + 0.16, -0.5, 0.5)
    elif key == 'inflate':                          # the whole form swells outward: a fan of normals
        b = 0.62 * _w(XS, 0.95)
        g.ghost(0.42 * _w(XS, 0.8))
        g.form(b)
        for x in (-0.5, 0.0, 0.5):
            y0 = 0.62 * float(_w(np.array([x]), 0.95)[0])
            g.arrow((x * 0.9, y0 - 0.05), (x * 1.75, y0 + 0.48), w=2.2, head=0.15)
    elif key == 'smooth':                           # lumps on the left, evened on the right, swept by the stroke
        rng = np.random.default_rng(3)
        n = 0.16 * np.sin(XS * 21.0) + 0.06 * rng.standard_normal(XS.size)
        t = np.clip((XS + 0.15) / 0.45, 0.0, 1.0)
        g.form(0.3 + n * (1.0 - t * t * (3 - 2 * t)))
        g.arrow((-0.75, 0.86), (0.85, 0.86), w=2.2)
    elif key == 'flatten':                          # a rounded hill cut by a plane: the cap gone (dashed)
        hill = 0.95 * _w(XS, 0.75)
        cut = 0.42
        g.ghost(np.where(hill > cut, hill, np.nan))
        g.form(np.minimum(hill, cut))
        g.plane(cut, -1.0, 1.0, w=1.8)
    elif key == 'crease':                           # a sharp groove cut down
        g.base = 0.52
        g.form(-0.85 * _w(XS, 0.16)); g.arrow((0.0, 0.85), (0.0, 0.12))
    elif key == 'pinch':                            # the sides drawn together into a sharp ridge
        g.form(0.62 * _w(XS, 0.16))
        g.arrow((-0.95, 0.38), (-0.3, 0.38), w=2.2); g.arrow((0.95, 0.38), (0.3, 0.38), w=2.2)
    elif key == 'grab':                             # the form dragged sideways with the mouse
        b = 0.72 * _w(XS, 0.42)
        xs2 = XS + 0.42 * _w(XS + 0.0, 0.7)
        g.ghost(b)
        g.form(b, xs=xs2); g.arrow((-0.35, 0.95), (0.55, 0.95))
    elif key == 'stretch':                          # drawn out along the drag both ways
        b = 0.62 * _w(XS, 0.3)
        xs2 = np.clip(XS * (1.0 + 1.1 * _w(XS, 0.95)), -1, 1)
        g.ghost(b)
        g.form(b, xs=xs2)
        g.arrow((-0.1, 0.92), (-0.95, 0.92), w=2.2); g.arrow((0.1, 0.92), (0.95, 0.92), w=2.2)
    elif key == 'snake':                            # an arm of even thickness pulled out of a flat surface
        g.form(0.12 * _w(XS, 0.5))
        _arm(g, (-0.25, 0.0), (0.7, 0.98), (-0.2, 0.75), 0.42)
    elif key == 'elastic':                          # a broad elastic pull: a spring, the far side stays
        b = 0.55 * _w(XS, 0.5)
        xs2 = XS + 0.28 * _w(XS, 1.6)
        g.ghost(b)
        g.form(b, xs=xs2)
        _spring(g, (-0.85, 0.92), (0.75, 0.92))
    elif key == 'mask':                             # paint over the surface that protects it
        y = 0.45 * _w(XS, 0.9)
        g.form(y, edge=False)
        sel = np.abs(XS) <= 0.6                     # the painted area: the form's top under the stroke, held in colour
        xs_, ys_ = XS[sel], y[sel]
        c = QColor(g.col); c.setAlpha(210)
        poly = QPolygonF([g.P(x, v) for x, v in zip(xs_, ys_)] + [g.P(x, v - 0.32) for x, v in zip(xs_[::-1], ys_[::-1])])
        g.p.setPen(Qt.NoPen); g.p.setBrush(c); g.p.drawPolygon(poly)
        g.line(y, WHITE, 2.2)
    elif key == 'swell':                            # grows until it touches a part above
        g.p.setPen(Qt.NoPen); g.p.setBrush(FORM_DARK)
        g.p.drawRect(QRectF(g.P(-1.0, 1.04), g.P(1.0, 0.84)))
        g.plane(0.84, -1.0, 1.0, w=1.8)
        g.ghost(0.25 * _w(XS, 0.45))
        g.form(np.minimum(0.25 * _w(XS, 0.45) + 0.7 * _w(XS, 0.42), 0.82))
        g.arrow((0.6, 0.02), (0.6, 0.68), w=2.2, head=0.14)
    elif key == 'settle':                           # comes down onto the part below it
        g.form(0.18 + 0.3 * _w(XS, 0.6), col=FORM_DARK, edge=False)
        g.ghost(0.85 + 0.0 * XS)
        g.line(0.24 + 0.3 * _w(XS, 0.6), WHITE, 2.2)
        g.arrow((0.0, 1.05), (0.0, 0.66), w=2.2, head=0.14)
    elif key in ('pencil ridge', 'pencil valley'):  # a thin line drawn with a pencil
        up = key == 'pencil ridge'
        g.form(0.22 + (0.32 if up else -0.32) * _w(XS, 0.07))
        tip = (0.04, 0.6 if up else 0.0)
        g.p.setPen(QPen(g.col, g.pw(4.4), Qt.SolidLine, Qt.FlatCap))
        g.p.drawLine(g.P(tip[0] + 0.5, tip[1] + 0.62), g.P(tip[0] + 0.12, tip[1] + 0.14))
        g.p.setPen(Qt.NoPen); g.p.setBrush(WHITE)
        a, b = g.P(tip[0] + 0.12, tip[1] + 0.14), g.P(*tip)
        g.p.drawPolygon(QPolygonF([QPointF(a.x() - 0.05 * g.s, a.y() - 0.03 * g.s),
                                   QPointF(a.x() + 0.05 * g.s, a.y() + 0.03 * g.s), b]))
    elif key in ('cut', 'cut + separate'):          # the outline's inside removed (or lifted out as its own object)
        y = np.full_like(XS, 0.55)
        g.form(np.where(np.abs(XS) < 0.32, np.nan, y))
        pen = QPen(g.col, g.pw(1.8), Qt.DashLine)
        g.p.setPen(pen)
        g.p.drawLine(g.P(-0.32, -0.6), g.P(-0.32, 1.1)); g.p.drawLine(g.P(0.32, -0.6), g.P(0.32, 1.1))
        if key == 'cut + separate':
            g.p.setPen(QPen(WHITE, g.pw(2.0))); g.p.setBrush(FORM)
            g.p.drawRect(QRectF(g.P(-0.24, 1.12), g.P(0.24, 0.62)))
        else:
            g.p.setPen(QPen(g.col, g.pw(2.4)))
            g.p.drawLine(g.P(-0.16, 0.5), g.P(0.16, 0.0)); g.p.drawLine(g.P(-0.16, 0.0), g.P(0.16, 0.5))
    elif key in ('splice', 'fill + splice'):        # two faces joined (the gap filled first)
        gap = 0.18 if key == 'splice' else 0.42
        lo, hi = 0.15, 0.55
        a = np.where(XS < -gap, lo, np.nan)
        b = np.where(XS > gap, hi, np.nan)
        j = lo + (hi - lo) * 0.5 * (1.0 + np.tanh(XS / (0.45 * gap + 0.02)))
        jb = np.where(np.abs(XS) <= gap + 0.06, j, np.nan)
        c = QColor(g.col); c.setAlpha(255)
        g.form(jb, col=c, edge=False)
        g.form(a); g.form(b)
        g.line(jb, WHITE, 2.2)
        if key == 'fill + splice':
            g.p.setPen(QPen(WHITE, g.pw(2.4)))
            m = g.P(0.0, 0.85); r = 0.11 * g.s
            g.p.drawLine(QPointF(m.x() - r, m.y()), QPointF(m.x() + r, m.y()))
            g.p.drawLine(QPointF(m.x(), m.y() - r), QPointF(m.x(), m.y() + r))
    elif key == 'select':
        g.base = 0.55
        poly = QPolygonF([g.P(-0.3, 0.75), g.P(-0.3, -0.6), g.P(-0.02, -0.28), g.P(0.2, -0.72), g.P(0.38, -0.62),
                          g.P(0.16, -0.2), g.P(0.52, -0.2)])
        g.p.setPen(QPen(WHITE, g.pw(1.6))); g.p.setBrush(QColor(238, 238, 240, 90)); g.p.drawPolygon(poly)
    elif key == 'move':
        g.base = 0.55
        for a, b in (((0, 0), (0, 0.75)), ((0, 0), (0, -0.7)), ((0, 0), (0.85, 0)), ((0, 0), (-0.85, 0))):
            g.arrow(a, b, WHITE, 2.0, head=0.15)
    elif key == 'rotate':
        g.base = 0.55
        path = QPainterPath()
        c = g.P(0, 0.0)
        R = 0.3 * g.s
        path.arcMoveTo(QRectF(c.x() - R, c.y() - R, 2 * R, 2 * R), 40)
        path.arcTo(QRectF(c.x() - R, c.y() - R, 2 * R, 2 * R), 40, 280)
        g.p.setPen(QPen(WHITE, g.pw(2.2))); g.p.setBrush(Qt.NoBrush); g.p.drawPath(path)
        th = math.radians(40 + 280)                 # the arc's end, the head along its counter-clockwise tangent
        e = QPointF(c.x() + R * math.cos(th), c.y() - R * math.sin(th))
        ang = math.atan2(-math.cos(th), -math.sin(th))
        L, Wd = 0.2 * g.s, 0.12 * g.s
        tip = QPointF(e.x() + 0.55 * L * math.cos(ang), e.y() + 0.55 * L * math.sin(ang))
        bx, by = tip.x() - L * math.cos(ang), tip.y() - L * math.sin(ang)
        nx, ny = -math.sin(ang), math.cos(ang)
        g.p.setPen(Qt.NoPen); g.p.setBrush(WHITE)
        g.p.drawPolygon(QPolygonF([tip, QPointF(bx + Wd * nx, by + Wd * ny), QPointF(bx - Wd * nx, by - Wd * ny)]))
    elif key == 'scale':
        g.base = 0.55
        g.p.setPen(QPen(GREY, g.pw(1.4))); g.p.setBrush(Qt.NoBrush)
        g.p.drawRect(QRectF(g.P(-0.6, 0.0), g.P(0.0, -0.7)))
        g.p.setPen(QPen(WHITE, g.pw(2.0)))
        g.p.drawRect(QRectF(g.P(-0.6, 0.8), g.P(0.75, -0.7)))
        g.arrow((-0.05, 0.05), (0.62, 0.68), WHITE, 2.0, head=0.15)
    else:
        g.form(0.4 * _w(XS))


def glyph_pixmap(key, size=32, band=True):
    k = (key, size, band)
    if k in _cache:
        return _cache[k]
    cat = TOOL_INFO.get(key, ('Build',))[0]
    pm = QPixmap(size, size)
    pm.fill(BG)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    _draw(_G(p, size, cat), key)
    if band:
        p.fillRect(QRectF(0, 0, max(2.0, size / 16.0), size), QColor(*CAT_COL[cat]))
    p.end()
    _cache[k] = pm
    return pm


def glyph_icon(key, size=32):
    ic = QIcon()
    ic.addPixmap(glyph_pixmap(key, size))
    if size != 48:
        ic.addPixmap(glyph_pixmap(key, 48))
    return ic


def category_of(key):
    return TOOL_INFO.get(key, ('Build',))[0]
