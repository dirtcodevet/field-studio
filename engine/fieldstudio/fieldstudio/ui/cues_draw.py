"""Drawing fs-tools' on-screen cues (fs-shell-2, 2026-11-14; toolset/cues.py gives them as data in world metres:
"Each tool draws its true reach on the model", TOOLSET_DESIGN 1.7).

The meaning colours (interface design 3.2), the same in every tool:
  orange  the tool (rings, lens, its column, the cut blade, an axis, a path, an outline)
  yellow  what it caught (the far face Whole part will move)
  green   contact;  amber dashed  a gap (not touching)
  light dashed  a ghost (where copies will land) or a back surface seen through
  labels  light text on a dark backing, by the point they name
Discs are drawn as their rim (64 points) projected through the view's camera; anything behind the camera is skipped.
Nothing here computes a cue: what is drawn is what toolset/cues.py measured from the field."""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen

ORANGE = QColor(245, 150, 60)
YELLOW = QColor(250, 214, 80)
GREEN = QColor(90, 210, 110)
AMBER = QColor(236, 178, 70)
LIGHT = QColor(225, 230, 238)
STYLE = {  # style -> (colour, width, dashed, fill alpha)
    'ring_inner': (ORANGE, 2.0, False, 0), 'ring_outer': (ORANGE, 1.2, True, 0), 'lens': (ORANGE, 2.0, False, 40),
    'column': (ORANGE, 1.2, True, 0), 'blade': (ORANGE, 1.6, False, 60), 'blade_through': (ORANGE, 1.2, True, 30),
    'axis': (ORANGE, 1.6, False, 0), 'path': (ORANGE, 1.6, True, 0), 'outline': (ORANGE, 1.8, False, 0),
    'far_face': (YELLOW, 1.8, True, 0), 'contact': (GREEN, 2.0, False, 255), 'gap': (AMBER, 2.0, True, 0),
    'ghost': (LIGHT, 1.2, True, 0), 'label': (LIGHT, 1.0, False, 0)}


def _proj(cam, w, h, pt):
    x, y, z = cam.project(np.asarray(pt, float), w, h)
    return (QPointF(x, y) if z > 1e-6 and math.isfinite(x) and math.isfinite(y) else None)


def _rim(c, n, r, k=64):
    n = np.asarray(n, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    t = np.linspace(0.0, 2.0 * math.pi, k + 1)
    return np.asarray(c, float)[None, :] + r * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)


def _pen(style):
    col, wd, dashed, _f = STYLE.get(style, (ORANGE, 1.5, False, 0))
    pen = QPen(col, wd)
    if dashed:
        pen.setStyle(Qt.DashLine)
    pen.setCosmetic(True)
    return pen


def _path(cam, w, h, pts, closed):
    path, drawing = QPainterPath(), False
    for q in pts:
        s = _proj(cam, w, h, q)
        if s is None:
            drawing = False
            continue
        if drawing:
            path.lineTo(s)
        else:
            path.moveTo(s)
            drawing = True
    if closed and len(pts) > 2 and drawing:
        s0 = _proj(cam, w, h, pts[0])
        if s0 is not None:
            path.lineTo(s0)
    return path


def draw(p, cam, w, h, cues):
    """draw cue dicts with the painter p over a w x h view; -> how many were drawn"""
    n = 0
    labels = []
    for c in cues or ():
        k, st = c.get('kind'), c.get('style', '')
        if k == 'text':
            labels.append(c)
            continue
        p.setPen(_pen(st))
        p.setBrush(Qt.NoBrush)
        if k == 'polyline':
            path = _path(cam, w, h, c.get('points') or [], bool(c.get('closed')))
            p.drawPath(path)
        elif k == 'disc':
            path = _path(cam, w, h, _rim(c['centre'], c['normal'], float(c['radius'])), True)
            fa = STYLE.get(st, (ORANGE, 1, False, 0))[3]
            if fa:
                col = QColor(STYLE[st][0])
                col.setAlpha(fa)
                p.fillPath(path, col)
            p.drawPath(path)
        elif k == 'quad':
            path = _path(cam, w, h, c.get('points') or [], True)
            fa = STYLE.get(st, (ORANGE, 1, False, 0))[3]
            if fa:
                col = QColor(STYLE[st][0])
                col.setAlpha(fa)
                p.fillPath(path, col)
            p.drawPath(path)
        elif k == 'point':
            s = _proj(cam, w, h, c['at'])
            if s is None:
                continue
            if st == 'contact':
                p.setBrush(GREEN)
            p.drawEllipse(s, 5.0, 5.0)
            p.setBrush(Qt.NoBrush)
        else:
            continue
        n += 1
    f = QFont('Segoe UI', 9)
    f.setBold(True)
    p.setFont(f)
    fm = p.fontMetrics()
    for c in labels:
        s = _proj(cam, w, h, c['at'])
        if s is None:
            continue
        t = str(c.get('text', ''))
        r = QRectF(s.x() + 10, s.y() - 22, fm.horizontalAdvance(t) + 12, fm.height() + 6)
        r.moveLeft(min(r.left(), w - r.width() - 4))
        r.moveTop(max(4.0, r.top()))
        p.fillRect(r, QColor(16, 18, 22, 215))
        p.setPen(QColor(255, 214, 120) if 'Not touching' in t or 'gap' in t else LIGHT)
        p.drawText(r, int(Qt.AlignCenter), t)
        n += 1
    return n
