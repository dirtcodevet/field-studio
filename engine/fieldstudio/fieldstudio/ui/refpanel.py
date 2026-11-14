"""hand1 (2026-11-07): the References panel: a front and a side photo as two planes crossing at one anchor (see
render/refplanes.py for the geometry). Every change here recomposites the last frame; nothing goes to the engine,
the op log or the state hash. The planes are saved in the project's 'extra' (paths relative to the project, the
images copied into <project>/refs/)."""
import math
import os

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QPushButton, QScrollArea, QSlider, QVBoxLayout,
                               QWidget)

from ..render import refplanes as RP

MAX_SIDE = 2048


def sort_front_side(paths):
    """-> (front, side) from picked file names: 'front' / 'side' or 'profile' in the name, else in order"""
    front = next((p for p in paths if 'front' in os.path.basename(p).lower()), None)
    side = next((p for p in paths if any(k in os.path.basename(p).lower() for k in ('profile', 'side'))), None)
    rest = [p for p in paths if p not in (front, side)]
    if front is None and rest:
        front = rest.pop(0)
    if side is None and rest:
        side = rest.pop(0)
    return front, side


def load_rgba(path):
    """-> (H, W, 4) uint8 array (downscaled so the longer side <= 2048) and the ORIGINAL (W, H)."""
    img = QImage(path)
    if img.isNull():
        raise IOError('cannot read image %s' % path)
    W0, H0 = img.width(), img.height()
    if max(W0, H0) > MAX_SIDE:
        img = img.scaled(MAX_SIDE, MAX_SIDE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    img = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
    arr = arr[:, :w * 4].reshape(h, w, 4).copy()
    return arr, (W0, H0)


class RefImageDialog(QDialog):
    """Click on the image: 'centre' mode -> one point; 'measure' mode -> two points (rows used)."""

    def __init__(self, path, mode, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Click the centre point' if mode == 'centre' else
                            'Click the top, then the bottom of a feature you know the height of')
        self.mode = mode
        self.pix = QPixmap(path)
        self.points = []
        self.lab = QLabel()
        self.lab.setCursor(Qt.CrossCursor)
        lay = QVBoxLayout(self)
        hint = QLabel('Centre: click the point that should sit on the planes\' crossing line (for example the ear canal '
                      'in the side view, the midline between the pupils in the front view).' if mode == 'centre' else
                      'Click two points one above the other (for example the crown and the chin); you then type their '
                      'distance in millimetres.')
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addWidget(self.lab)
        self.sc = min(1.0, 860.0 / max(self.pix.height(), 1), 1100.0 / max(self.pix.width(), 1))
        self._show()
        self.lab.mousePressEvent = self._click

    def _show(self):
        pm = self.pix.scaled(int(self.pix.width() * self.sc), int(self.pix.height() * self.sc), Qt.KeepAspectRatio,
                             Qt.SmoothTransformation)
        p = QPainter(pm)
        p.setPen(QPen(QColor(255, 60, 60), 2))
        for x, y in self.points:
            p.drawLine(QPointF(x * self.sc - 8, y * self.sc), QPointF(x * self.sc + 8, y * self.sc))
            p.drawLine(QPointF(x * self.sc, y * self.sc - 8), QPointF(x * self.sc, y * self.sc + 8))
        p.end()
        self.lab.setPixmap(pm)

    def _click(self, e):
        x, y = e.position().x() / self.sc, e.position().y() / self.sc
        self.points.append((x, y))
        self._show()
        if (self.mode == 'centre' and len(self.points) == 1) or len(self.points) == 2:
            self.accept()


class _PlaneBox(QGroupBox):
    def __init__(self, panel, name, title):
        super().__init__(title)
        self.panel, self.name = panel, name
        g = QGridLayout(self)
        g.setSpacing(4)
        self.path_lab = QLabel('(no image)')
        self.path_lab.setObjectName('dim')
        self.path_lab.setWordWrap(True)
        g.addWidget(self.path_lab, 0, 0, 1, 4)
        b = QPushButton('Load...')
        b.clicked.connect(self._load)
        g.addWidget(b, 1, 0)
        self.visible = QCheckBox('Visible')
        self.visible.setChecked(True)
        self.visible.toggled.connect(lambda on: self._set('visible', on))
        g.addWidget(self.visible, 1, 1)
        self.flip = QCheckBox('Flip H')
        self.flip.toggled.connect(lambda on: self._set('flip_h', on))
        g.addWidget(self.flip, 1, 2)
        self.facing = QCheckBox('Facing only')
        self.facing.setToolTip('Draw the plane only from its front side (not mirrored from behind)')
        self.facing.toggled.connect(lambda on: self._set('sides', 'facing' if on else 'both'))
        g.addWidget(self.facing, 1, 3)
        g.addWidget(QLabel('Opacity'), 2, 0)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.valueChanged.connect(lambda v: self._set('opacity', v / 100.0))
        g.addWidget(self.opacity, 2, 1, 1, 3)
        g.addWidget(QLabel('Draw'), 3, 0)
        self.draw = QComboBox()
        for text, key in (('Auto: on top in its own view', 'auto'), ('Depth-tested', 'depth'),
                          ('Behind the model only', 'behind'), ('On top (x-ray)', 'xray')):
            self.draw.addItem(text, key)
        self.draw.currentIndexChanged.connect(lambda i: self._set('draw', self.draw.itemData(i)))
        self.draw.setToolTip('Auto (default): drawn over the model when you look straight at the plane (the front '
                             'plane in the Front view, the side plane in the Side view), so you can trace it; '
                             'depth-tested in every other view, where the planes cross inside the model')
        g.addWidget(self.draw, 3, 1, 1, 3)
        g.addWidget(QLabel('Height mm'), 4, 0)
        self.span = QDoubleSpinBox()
        self.span.setRange(10, 3000)
        self.span.setDecimals(1)
        self.span.setValue(300.0)
        self.span.setToolTip('The full image height in mm, or the measured feature\'s height when one is measured')
        self.span.valueChanged.connect(lambda v: self._set('span_mm', float(v)))
        g.addWidget(self.span, 4, 1)
        b = QPushButton('Measure...')
        b.setToolTip('Click two points on the image (e.g. crown and chin) and type their distance in mm')
        b.clicked.connect(self._measure)
        g.addWidget(b, 4, 2)
        b = QPushButton('Full height')
        b.clicked.connect(lambda: self._set('span_px', None, refresh=True))
        g.addWidget(b, 4, 3)
        g.addWidget(QLabel('Offset mm'), 5, 0)
        self.off = []
        for i in range(3):
            sp = QDoubleSpinBox()
            sp.setRange(-500, 500)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setPrefix('XYZ'[i] + ' ')
            sp.valueChanged.connect(self._offset)
            g.addWidget(sp, 5, 1 + i)
            self.off.append(sp)
        b = QPushButton('Pick centre...')
        b.setToolTip('Click the image point that should sit on the crossing line')
        b.clicked.connect(self._centre)
        g.addWidget(b, 6, 0, 1, 2)
        b = QPushButton('Image centre')
        b.clicked.connect(lambda: self._set('centre_px', None, refresh=True))
        g.addWidget(b, 6, 2, 1, 2)
        self.key = QCheckBox('Key out flat background')
        self.key.setChecked(True)
        self.key.setToolTip('A flat background colour joined to the image border (the black around generated '
                            'references) is drawn transparent, so it does not darken the model')
        self.key.toggled.connect(self._key)
        g.addWidget(self.key, 7, 0, 1, 4)
        self.info = QLabel('')
        self.info.setObjectName('dim')
        self.info.setWordWrap(True)
        g.addWidget(self.info, 8, 0, 1, 4)

    def plane(self):
        return self.panel.vp.refs.get(self.name)

    def _set(self, k, v, refresh=False):
        p = self.plane()
        if p is None:
            return
        setattr(p, k, v)
        if refresh:
            self.refresh()
        self.panel.changed_now()

    def _key(self, on):
        p = self.plane()
        if p is None:
            return
        p.key_bg = bool(on)
        if p.path and os.path.exists(p.path):
            self.panel.load(self.name, p.path)            # re-prepare the texture with / without the key
        self.panel.changed_now()

    def _offset(self, _=None):
        self._set('offset_mm', [s.value() for s in self.off])

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Reference image (%s)' % self.name, self.panel.last_dir,
                                              'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if path:
            self.panel.last_dir = os.path.dirname(path)
            self.panel.load(self.name, path)

    def _centre(self):
        p = self.plane()
        if p is None or not p.path:
            return
        d = RefImageDialog(p.path, 'centre', self)
        if d.exec() == QDialog.Accepted and d.points:
            self._set('centre_px', [round(d.points[0][0], 1), round(d.points[0][1], 1)], refresh=True)

    def _measure(self):
        p = self.plane()
        if p is None or not p.path:
            return
        d = RefImageDialog(p.path, 'measure', self)
        if d.exec() == QDialog.Accepted and len(d.points) == 2:
            mm, ok = QInputDialog.getDouble(self, 'Measured height', 'Distance between the two points (mm):',
                                            230.0, 1.0, 3000.0, 1)
            if ok:
                p.span_px = [round(d.points[0][1], 1), round(d.points[1][1], 1)]
                p.span_mm = float(mm)
                self.refresh()
                self.panel.changed_now()

    def refresh(self):
        p = self.plane()
        widgets = [self.visible, self.flip, self.facing, self.opacity, self.draw, self.span, self.key] + self.off
        for w in widgets:
            w.blockSignals(True)
        if p is None:
            self.path_lab.setText('(no image)')
            self.info.setText('')
        else:
            self.path_lab.setText(os.path.basename(p.path) or '(no image)')
            self.path_lab.setToolTip(p.path)
            self.visible.setChecked(p.visible)
            self.flip.setChecked(p.flip_h)
            self.facing.setChecked(p.sides == 'facing')
            self.opacity.setValue(int(round(p.opacity * 100)))
            self.draw.setCurrentIndex(max(0, self.draw.findData(p.draw)))
            self.span.setValue(float(p.span_mm))
            self.key.setChecked(bool(getattr(p, 'key_bg', True)))
            for s, v in zip(self.off, p.offset_mm):
                s.setValue(float(v))
            W, H = self.panel.vp.ref_sizes.get(self.name, (0, 0))
            c = p.centre_px or [W / 2.0, H / 2.0]
            self.info.setText('%d x %d px; centre at (%.0f, %.0f) px; %s = %.1f mm (%.3f mm per px)' % (
                W, H, c[0], c[1], 'rows %.0f-%.0f' % tuple(p.span_px) if p.span_px else 'full height',
                p.span_mm, p.span_mm / max(abs(p.span_px[1] - p.span_px[0]) if p.span_px else H, 1)))
        for w in widgets:
            w.blockSignals(False)


class RefPanel(QWidget):
    changed = Signal()
    lensChanged = Signal(bool)          # hand1b: the View menu's 'Reference lens' follows this
    visibleChanged = Signal(bool)       # hand1b: the View menu's 'Reference planes visible' follows this

    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self.vp = viewport
        self.last_dir = ''
        self.lens_saved = None
        self.lens_auto = False              # fs-shell: the lens was switched on by an axis view with a rig loaded
        self.mattes = {}                    # fs-shell: name -> the photo's matte (render/refrig.matte)
        self.rgba = {}                      # fs-shell: name -> the loaded (unkeyed) image, for Auto-fit and the rig
        self.rig_height_mm = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        hint = QLabel('Two photos as planes crossing at one point: the front plane faces the Front view (key 1), the '
                      'side plane the Right view (key 3, her left profile). Their centre pixels meet on the vertical '
                      'crossing line. File > Add reference images... loads both.')
        hint.setWordWrap(True)
        hint.setObjectName('dim')
        lay.addWidget(self._rig_group())
        lay.addWidget(hint)
        self.boxes = {'front': _PlaneBox(self, 'front', 'Front'), 'side': _PlaneBox(self, 'side', 'Side (profile)')}
        for b in self.boxes.values():
            lay.addWidget(b)
        g = QGroupBox('Both planes')
        gl = QGridLayout(g)
        self.all_visible = QCheckBox('References visible')
        self.all_visible.setChecked(True)
        self.all_visible.toggled.connect(self._all)
        gl.addWidget(self.all_visible, 0, 0, 1, 2)
        self.fade = QCheckBox('Fade when edge-on')
        self.fade.setChecked(True)
        self.fade.toggled.connect(lambda on: self._gset('fade_edge_on', on))
        gl.addWidget(self.fade, 0, 2, 1, 2)
        self.frames = QCheckBox('Frames and centre marks')
        self.frames.setChecked(True)
        self.frames.toggled.connect(lambda on: (self._gset('show_frame', on), self._gset('show_cross', on)))
        gl.addWidget(self.frames, 1, 0, 1, 2)
        self.lens = QCheckBox('Reference lens (5 deg)')
        self.lens.setToolTip('A long lens so the model lines up with the near-orthographic photos (about 2 % '
                             'parallax left instead of about 18 % at 35 deg)')
        self.lens.toggled.connect(self.set_lens)
        gl.addWidget(self.lens, 1, 2, 1, 2)
        gl.addWidget(QLabel('Anchor mm'), 2, 0)
        self.anchor = []
        for i in range(3):
            sp = QDoubleSpinBox()
            sp.setRange(-1000, 1000)
            sp.setDecimals(1)
            sp.setPrefix('XYZ'[i] + ' ')
            sp.valueChanged.connect(self._anchor)
            gl.addWidget(sp, 2, 1 + i)
            self.anchor.append(sp)
        lay.addWidget(g)
        lay.addStretch(1)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(inner)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(sa)

    # ---- state ------------------------------------------------------------------------------------------------------
    def changed_now(self):
        self.vp.refs_changed()
        self.changed.emit()

    def _gset(self, k, v):
        setattr(self.vp.refs, k, v)
        self.changed_now()

    def _all(self, on):
        self.vp.refs.visible = on
        self.visibleChanged.emit(bool(on))
        self.changed_now()

    def _refresh_global(self):
        """hand1b: the 'Both planes' boxes show the RefSet's state (after a project or staged set is applied)"""
        rs = self.vp.refs
        for w, on in ((self.all_visible, rs.visible), (self.fade, rs.fade_edge_on),
                      (self.frames, rs.show_frame or rs.show_cross)):
            w.blockSignals(True)
            w.setChecked(bool(on))
            w.blockSignals(False)
        self.visibleChanged.emit(bool(rs.visible))

    def _anchor(self, _=None):
        self.vp.refs.anchor_mm = [s.value() for s in self.anchor]
        self.changed_now()

    # ---- fs-shell (11-11): the Reference Rig ------------------------------------------------------------------------
    def _rig_group(self):
        g = QGroupBox('Reference Rig: front and side on one centre')
        gl = QGridLayout(g)
        gl.setSpacing(4)
        b = QPushButton('Load front + side...')
        b.setToolTip('Pick both photos (names with "front" and "side" / "profile" are sorted for you); the rig opens')
        b.clicked.connect(self.load_and_rig)
        gl.addWidget(b, 0, 0, 1, 2)
        self.b_rig = QPushButton('Set up rig...')
        self.b_rig.setToolTip('Two clicks per photo (crown, chin) plus the centre (midline / ear canal), or Auto-fit')
        self.b_rig.clicked.connect(self.open_rig)
        gl.addWidget(self.b_rig, 0, 2, 1, 2)
        self.b_af = QPushButton('Auto-fit now')
        self.b_af.setToolTip("Find each head's outline and place both photos at once (no dialog)")
        self.b_af.clicked.connect(lambda: self.rig_autofit_apply())
        gl.addWidget(self.b_af, 1, 0, 1, 2)
        self.rig_h = QDoubleSpinBox()
        self.rig_h.setRange(20.0, 3000.0)
        self.rig_h.setDecimals(1)
        self.rig_h.setSuffix(' mm')
        self.rig_h.setPrefix('crown-chin ')
        self.rig_h.setValue(230.0)
        self.rig_h.setToolTip('The real crown-to-chin height both photos span')
        self.rig_h.editingFinished.connect(self._rig_height_changed)
        gl.addWidget(self.rig_h, 1, 2, 1, 2)
        self.sil = QCheckBox('Silhouette check (mm)')
        self.sil.setChecked(True)
        self.sil.setToolTip("In the Front and Side views: the model's outline coloured by its distance from the "
                            "photo's outline (green <= 1 mm, amber <= 3, red > 3)")
        self.sil.toggled.connect(self._sil_toggle)
        gl.addWidget(self.sil, 2, 0, 1, 2)
        self.sil_lab = QLabel('')
        self.sil_lab.setTextFormat(Qt.PlainText)
        self.sil_lab.setMinimumWidth(10)
        from PySide6.QtWidgets import QSizePolicy as _SP
        self.sil_lab.setSizePolicy(_SP.Ignored, _SP.Preferred)     # a changing text never resizes the panel
        gl.addWidget(self.sil_lab, 2, 2, 1, 2)
        gl.addWidget(QLabel('Nudge'), 3, 0)
        self.nudge_which = QComboBox()
        self.nudge_which.addItem('Front', 'front')
        self.nudge_which.addItem('Side', 'side')
        gl.addWidget(self.nudge_which, 3, 1)
        nb = QHBoxLayout()
        nb.setContentsMargins(0, 0, 0, 0)
        for txt, dx, dy in (('<', -1, 0), ('>', 1, 0), ('^', 0, 1), ('v', 0, -1)):
            b = QPushButton(txt)
            b.setFixedWidth(28)
            b.setToolTip('Move the selected photo 0.1 mm (Shift: 1 mm); arrow keys do the same with this panel focused')
            b.clicked.connect(lambda _=False, a=dx, c=dy: self.nudge(self.nudge_which.currentData(), a, c))
            nb.addWidget(b)
        nb.addStretch(1)
        w = QWidget()
        w.setLayout(nb)
        gl.addWidget(w, 3, 2, 1, 2)
        self.rig_lab = QLabel('')
        self.rig_lab.setObjectName('dim')
        self.rig_lab.setWordWrap(True)
        gl.addWidget(self.rig_lab, 4, 0, 1, 4)
        self.vp.statsChanged.connect(self._sil_update)
        return g

    def keyPressEvent(self, e):
        step = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, 1), Qt.Key_Down: (0, -1)}.get(e.key())
        if step is not None and self.rig_active():
            self.nudge(self.nudge_which.currentData(), *step)
            e.accept()
            return
        super().keyPressEvent(e)

    def rig_active(self):
        return any(getattr(p, 'marks', None) for p in self.vp.refs.planes)

    def _rig_refresh(self):
        if not hasattr(self, 'rig_lab'):
            return
        rs = self.vp.refs
        if rs.rig_height_mm:
            self.rig_h.blockSignals(True)
            self.rig_h.setValue(float(rs.rig_height_mm))
            self.rig_h.blockSignals(False)
        sec = next((p.marks.get('second', 'chin') for p in rs.planes if getattr(p, 'marks', None)), 'chin')
        self.rig_h.setPrefix('crown-%s ' % sec)
        parts = []
        for p in rs.planes:
            mk = getattr(p, 'marks', None)
            if mk:
                rows = abs(mk['chin'][1] - mk['crown'][1])
                parts.append('%s: crown %.0f, %s %.0f px (%.3f mm/px), centre x %.0f%s' % (
                    p.name, mk['crown'][1], mk.get('second', 'chin'), mk['chin'][1], p.span_mm / max(rows, 1), mk['centre_x'],
                    ', mirrored' if p.flip_h else ''))
            elif p.path:
                parts.append('%s: loaded, not set' % p.name)
        self.rig_lab.setText('\n'.join(parts) if parts else 'No photos yet: Load front + side...')
        self.sil.blockSignals(True)
        self.sil.setChecked(bool(getattr(rs, 'sil_check', True)))
        self.sil.blockSignals(False)

    def _arr_scale(self, name):
        """ORIGINAL image px per loaded-array px (1 unless the photo was larger than 2048 px)"""
        arr = self.rgba.get(name)
        W0, H0 = self.vp.ref_sizes.get(name, (None, None))
        if arr is None or not W0:
            return 1.0
        return W0 / float(arr.shape[1])

    @staticmethod
    def _marks_scaled(mk, k):
        out = dict(mk)
        for key in ('crown', 'chin', 'centre'):
            if key in out:
                out[key] = [float(out[key][0]) * k, float(out[key][1]) * k]
        if 'centre_x' in out:
            out['centre_x'] = float(out['centre_x']) * k
        return out

    def form_outline(self, name):
        """the silhouette check's FORM outline for one photo, once its scale is known: the matte opened at 1 mm, so
        lashes and stray hairs are not taken for the form (the bust profile's lashes stood 5 mm proud at the pupils)"""
        from ..render import refrig as RR
        m = self.mattes.get(name)
        tex = self.vp.ref_tex.get(name)
        p = self.vp.refs.get(name)
        if m is None or not m.get('keyed') or tex is None or p is None:
            return
        W0, H0 = self.vp.ref_sizes[name]
        rows = abs(p.span_px[1] - p.span_px[0]) if p.span_px else H0
        mpp = float(p.span_mm) / max(rows, 1.0)
        k = m['fg'].shape[1] / float(W0)
        tex.sd = RR.signed_distance(m, open_px=k * 1.0 / max(mpp, 1e-6))

    def autofit_marks(self):
        """-> {name: marks in ORIGINAL image px} proposed by Auto-fit for every loaded photo with a plain background"""
        from ..render import refrig as RR
        out = {}
        for name, m in self.mattes.items():
            if not m.get('keyed') or name not in self.rgba:
                continue
            af = RR.autofit(m, self.rgba[name], name)
            if af is None:
                continue
            cy = af['ear'][1] if 'ear' in af else 0.5 * (af['crown'][1] + af['chin'][1])
            mk = {'crown': af['crown'], 'chin': af['chin'], 'centre_x': af['centre_x'], 'centre': [af['centre_x'], cy]}
            if name == 'side':
                mk['flip_h'] = bool(af.get('flip_h', False))
            out[name] = self._marks_scaled(mk, self._arr_scale(name))
        return out

    def apply_rig(self, marks_by_plane, height_mm=None, view=True):
        """The scripted rig (fs-bench, tests, the dialog): marks per plane in ORIGINAL image px ({'crown': [x, y],
        'chin': [x, y], 'centre_x': x, optional 'flip_h'}), the crown-to-chin height in mm. Every plane with marks spans
        that height crown to chin, its centre column on the crossing line, its offsets cleared; with view=True the lens
        goes on and the view frames the head from the front. -> {name: mm per px}."""
        from ..render import refrig as RR
        h = float(height_mm or self.rig_h.value())
        out = {}
        for name, mk in marks_by_plane.items():
            p = self.vp.refs.get(name)
            if p is None:
                continue
            out[name] = RR.place(p, mk, h)
            p.marks = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in mk.items()}
            p.offset_mm = [0.0, 0.0, 0.0]
            self.form_outline(name)
        self.vp.refs.rig_height_mm = h
        self.rig_height_mm = h
        for b in self.boxes.values():
            b.refresh()
        self._rig_refresh()
        if view and out:
            self.frame_rig(1)
        self.changed_now()
        return out

    def frame_rig(self, key=1):
        """an axis view of the rig's head on the near-orthographic lens"""
        cam = self.vp.cam
        if self.lens_on():
            self.set_lens(False)
        cam.preset(key)
        H = float(self.vp.refs.rig_height_mm or 230.0) / 1000.0
        a = np.asarray(self.vp.refs.anchor_mm, float) / 1000.0
        lo, hi = a - 0.62 * H, a + 0.62 * H
        cam.frame(lo, hi, margin=1.05, aspect=max(self.vp.width(), 1) / max(self.vp.height(), 1))
        self.axis_view()
        self.vp.request(False)

    def rig_autofit_apply(self, height_mm=None, view=True):
        """Auto-fit and apply in one step (the panel's button, --rig-auto, fs-bench). -> the marks used"""
        mk = self.autofit_marks()
        if mk:
            self.apply_rig(mk, height_mm, view=view)
        return mk

    def open_rig(self):
        if not self.rgba:
            return self.load_and_rig()
        from .refrig_dialog import RigDialog
        d = RigDialog(self, self.window())
        for name, c in d.canvases.items():                 # the dialog works in the loaded arrays' pixels
            p = self.vp.refs.get(name)
            if p is not None and getattr(p, 'marks', None):
                mk = self._marks_scaled(p.marks, 1.0 / self._arr_scale(name))
                c.marks = {'crown': mk['crown'], 'chin': mk['chin'],
                           'centre': mk.get('centre', [mk['centre_x'], mk['crown'][1]])}
                if 'flip_h' in mk:
                    c.marks['flip_h'] = mk['flip_h']
        sec = next((p.marks.get('second') for p in self.vp.refs.planes if getattr(p, 'marks', None)), None)
        if sec:
            i = d.second_box.findData(sec)
            if i >= 0:
                d.second_box.setCurrentIndex(i)
        d._update_info()
        self.rig_dialog = d
        if d.exec() == QDialog.Accepted:
            mk = {n: self._marks_scaled(m, self._arr_scale(n)) for n, m in d.marks().items()}
            self.rig_h.setValue(d.height_mm.value())
            self.apply_rig(mk, d.height_mm.value())
        self.rig_dialog = None

    def load_and_rig(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Reference photos: the front and the side (profile)',
                                                self.last_dir, 'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not paths:
            return
        self.last_dir = os.path.dirname(paths[0])
        front, side = sort_front_side(paths)
        for name, path in (('front', front), ('side', side)):
            if path:
                self.load(name, path)
        self.open_rig()

    def _rig_height_changed(self):
        h = float(self.rig_h.value())
        for p in self.vp.refs.planes:
            if getattr(p, 'marks', None):
                p.span_mm = h
        if self.rig_active():
            self.vp.refs.rig_height_mm = h
        for p in self.vp.refs.planes:
            self.form_outline(p.name)
        for b in self.boxes.values():
            b.refresh()
        self._rig_refresh()
        self.changed_now()

    def nudge(self, name, dx, dy):
        """move one photo 0.1 mm per step (Shift: 1 mm) in its own image directions: dx along the image's x (front:
        world X; side: world -Z, or +Z when mirrored), dy up"""
        from PySide6.QtWidgets import QApplication
        p = self.vp.refs.get(name)
        if p is None:
            return
        step = 1.0 if QApplication.keyboardModifiers() & Qt.ShiftModifier else 0.1
        off = list(p.offset_mm)
        if name == 'front':
            off[0] += dx * step
        else:
            off[2] += (dx * step) * (1.0 if p.flip_h else -1.0)
        off[1] += dy * step
        p.offset_mm = [round(v, 3) for v in off]
        self.boxes[name].refresh()
        self.rig_lab.setText('%s offset: x %.1f, y %.1f, z %.1f mm' % ((name,) + tuple(p.offset_mm)))
        self.changed_now()

    def _sil_toggle(self, on):
        self.vp.refs.sil_check = bool(on)
        self.changed_now()

    def _sil_update(self, *_):
        """the check's readout, at most ~4 times a second and only when it changes: a rich, word-wrapped label set on
        every frame re-laid the panel out during strokes (fs-shell 11-11, the references' per-frame cost)"""
        import time as _time
        now = _time.perf_counter()
        if now - getattr(self, '_sil_t', 0.0) < 0.25:
            if not getattr(self, '_sil_timer', None):
                from PySide6.QtCore import QTimer
                self._sil_timer = QTimer(self)
                self._sil_timer.setSingleShot(True)
                self._sil_timer.timeout.connect(self._sil_update)
            if not self._sil_timer.isActive():
                self._sil_timer.start(260)
            return
        self._sil_t = now
        if not getattr(self.vp.refs, 'sil_check', False) or not self.vp.refs.visible:
            text, col = '', '#8a919b'
        else:
            r = getattr(self.vp.refs, 'sil_last', None)
            if not r:
                text, col = 'look from the Front (1) or Side (3)', '#8a919b'
            else:
                col = '#4ade80' if r['mean'] <= 1.0 else ('#fbbf24' if r['mean'] <= 3.0 else '#f87171')
                text = '%s: mean %.1f, 90%% %.1f, max %.0f mm, %d%% within 1 mm' % (
                    r['plane'], r['mean'], r['p90'], r['max'], int(round(100 * r['within1'])))
                if r.get('n_out'):
                    text += ' (crown to chin; grey: not scored)'
        if text != self.sil_lab.text():
            self.sil_lab.setText(text)
        if col != getattr(self, '_sil_col', None):
            self._sil_col = col
            self.sil_lab.setStyleSheet('color: %s;' % col)

    def check_auto_lens(self):
        """with the orthographic view or the lens switched on by an axis view, orbiting off the axis switches it back
        (Blender's auto perspective)"""
        auto_o = getattr(self, 'ortho_auto', False) and getattr(self.vp.cam, 'ortho', False)
        if not ((self.lens_auto and self.lens_on()) or auto_o):
            return
        f = np.asarray(self.vp.cam.basis()[0], float)
        if np.max(np.abs(f)) < math.cos(math.radians(1.5)):
            if auto_o:
                self.set_ortho(False)
                self.ortho_auto = False
            if self.lens_auto and self.lens_on():
                self.set_lens(False)
                self.lens_auto = False

    # ---- fs-shell (11-11): the orthographic view (parallel rays; render/camera.py) ------------------------------------
    orthoChanged = Signal(bool)

    def ortho_supported(self):
        """the CPU renderer draws parallel rays; the card's tracer does once gpu/scene.py says ORTHO = True"""
        eng = self.vp.engine
        if not getattr(eng, 'use_gpu', False) or getattr(eng, 'gpu_view', None) is None:
            return True
        try:
            from ..gpu import scene as GS
            return bool(getattr(GS, 'ORTHO', False))
        except Exception:
            return False

    def set_ortho(self, on):
        on = bool(on) and self.ortho_supported()
        if on and self.lens_on():
            self.set_lens(False)                 # the same view scale at the target, now with parallel rays
            self.lens_auto = False
        self.vp.cam.ortho = on
        self.orthoChanged.emit(on)
        self.vp.request(False)
        return on

    def axis_view(self):
        """an axis view with a rig loaded: the orthographic view where the renderer draws it, else the 5 deg lens"""
        if self.ortho_supported():
            if not getattr(self.vp.cam, 'ortho', False):
                self.set_ortho(True)
                self.ortho_auto = True
        elif not self.lens_on():
            self.set_lens(True)
            self.lens_auto = True

    def figure_scale(self, height_mm=1680.0):
        """hand2f: New figure: every plane spans the figure's height (the image's full height = height_mm) and the
        anchor sits at mid-height, so an image cropped head to feet stands with its feet on y = 0; planes loaded
        later in this session take the same span."""
        self.default_span_mm = float(height_mm)
        self.vp.refs.anchor_mm = [0.0, float(height_mm) / 2.0, 0.0]
        for s_, v in zip(getattr(self, 'anchor', []), self.vp.refs.anchor_mm):
            s_.blockSignals(True)
            s_.setValue(v)
            s_.blockSignals(False)
        for p in self.vp.refs.planes:
            self._fit_figure(p)
        for b in self.boxes.values():
            b.refresh()
        self.changed_now()

    def _fit_figure(self, p):
        """hand2b (verifier item 10): New figure: the SUBJECT (the image's non-background rows, not the full image
        height) spans the figure's height and its middle sits on the anchor (mid-height), so a full-body reference
        stands with its crown at the figure's top and its feet on y = 0. Before, the full image height was 1680 mm
        and the generated bodies (about 205 px of margin above the head, 230 px below the feet of 1728) stood ~25 %
        small. An image with no flat background keeps its full height."""
        span = getattr(self, 'default_span_mm', None)
        if not span:
            return
        p.span_mm = float(span)
        ext = getattr(self, 'extent', {}).get(p.name)
        if ext:
            t, b, l, r = ext
            p.span_px = [float(t), float(b + 1)]
            p.centre_px = [0.5 * (l + r + 1), 0.5 * (t + b + 1)]
        else:
            p.span_px = None
            p.centre_px = None

    def load(self, name, path, span_mm=None):
        arr, (W0, H0) = load_rgba(path)
        refs = self.vp.refs
        p = refs.get(name)
        fresh = p is None
        if p is None:
            p = RP.RefPlane(name=name)
            if getattr(self, 'default_span_mm', None):
                p.span_mm = float(self.default_span_mm)        # hand2f: a New figure session's height
            refs.planes.append(p)
        if p.path and os.path.normcase(os.path.abspath(p.path)) != os.path.normcase(os.path.abspath(path)):
            p.centre_px, p.span_px = None, None          # a new image: its own centre and full height
            fresh = True
        p.path = os.path.abspath(path)
        if span_mm:
            p.span_mm = float(span_mm)
        # fs-shell (11-11): the edge-aware matte (render/refrig.py) replaces hand2b's flat-colour key: the background is
        # a smooth colour field fitted to the border, the edge alpha unmixed (no ragged fringe); the subject's extent
        # in ORIGINAL pixels, the matte and the image kept for Auto-fit, and the outline's signed distance for the
        # silhouette check
        if not hasattr(self, 'extent'):
            self.extent = {}
        sd = None
        try:
            from ..render import refrig as RR
            m = RR.matte(arr)
            self.rgba[name] = arr
            self.mattes[name] = m
            ext = RP.content_extent(~m['fg']) if m.get('keyed') and m['fg'].any() else None
            if ext:
                sy, sx = H0 / float(arr.shape[0]), W0 / float(arr.shape[1])
                ext = (ext[0] * sy, (ext[1] + 1) * sy - 1, ext[2] * sx, (ext[3] + 1) * sx - 1)
            self.extent[name] = ext
            if m.get('keyed'):
                sd = RR.signed_distance(m)
                if getattr(p, 'key_bg', True):
                    arr = RR.keyed_rgba(arr, m)
            m.pop('rgb', None)                           # the keyed texture holds it now
        except Exception as e:                           # the key is a convenience: never block a load
            self.extent[name] = None
            self.load_error = repr(e)
        if fresh and not span_mm and getattr(self, 'default_span_mm', None):
            self._fit_figure(p)
        # the plane works in the ORIGINAL image's pixels (centre and span picks); the texture may be downscaled
        tex = RP.prepare_texture(arr)
        self.vp.ref_tex[name] = [t for t in tex]
        self.vp.ref_sizes[name] = (W0, H0)
        self._tex_scale_fix(name, W0, H0, arr.shape[1], arr.shape[0])
        self.vp.ref_tex[name].sd = sd
        self.boxes[name].refresh()
        self._rig_refresh()
        self.changed_now()
        return p

    def _tex_scale_fix(self, name, W0, H0, w, h):
        """the texture is in downscaled pixels; refplanes.pack maps pixel coords by the mip width / W0, so keep W0
        on the plane's geometry by storing the full-size W0 as level 0's logical width (handled in pack via sizes)"""
        self.vp.ref_tex[name] = _Tex(self.vp.ref_tex[name], W0, H0)

    def to_dict(self, rel_to=None):
        return self.vp.refs.to_dict(rel_to)

    def apply_dict(self, d, base=None):
        rs = RP.RefSet.from_dict(d or {}, base)
        missing = []
        self.vp.refs = RP.RefSet([], rs.anchor_mm, rs.fade_edge_on, rs.show_frame, rs.show_cross, rs.visible,
                                 rs.rig_height_mm, rs.sil_check)
        self.vp.ref_tex.clear()
        self.vp.ref_sizes.clear()
        for p in rs.planes:
            if p.path and os.path.exists(p.path):
                q = self.load(p.name, p.path)
                for k in ('visible', 'opacity', 'offset_mm', 'centre_px', 'span_px', 'span_mm', 'flip_h', 'sides',
                          'draw', 'key_bg', 'marks'):
                    setattr(q, k, getattr(p, k))
                if not q.key_bg:                         # hand2b: the texture was keyed at load; reload it plain
                    self.load(p.name, p.path)
            else:
                missing.append(p.path)
        for s, v in zip(self.anchor, self.vp.refs.anchor_mm):
            s.blockSignals(True)
            s.setValue(float(v))
            s.blockSignals(False)
        for p in self.vp.refs.planes:
            self.form_outline(p.name)
        for b in self.boxes.values():
            b.refresh()
        self._refresh_global()
        self._rig_refresh()
        self.changed_now()
        return missing

    LENS_FOV = 5.0
    NORMAL_FOV = 35.0

    def lens_on(self):
        """hand1b: the lens state is read from the camera itself (a 5 deg field of view), so the panel, the menu and
        a reopened project can never disagree about it"""
        return abs(float(self.vp.cam.fov) - self.LENS_FOV) < 0.01

    def set_lens(self, on):
        cam = self.vp.cam
        import math
        on = bool(on)
        if on and not self.lens_on():
            self.lens_saved = float(cam.fov)
            k = math.tan(math.radians(cam.fov) / 2) / math.tan(math.radians(self.LENS_FOV / 2))
            cam.pos = cam.target + (cam.pos - cam.target) * k
            cam.fov = self.LENS_FOV
        elif not on and self.lens_on():
            back = self.lens_saved if self.lens_saved and self.lens_saved > self.LENS_FOV + 0.01 else self.NORMAL_FOV
            k = math.tan(math.radians(self.LENS_FOV / 2)) / math.tan(math.radians(back) / 2)
            cam.pos = cam.target + (cam.pos - cam.target) * k
            cam.fov = back
            self.lens_saved = None
        self._show_lens()
        self.vp.request(False)

    def _show_lens(self):
        on = self.lens_on()
        if self.lens.isChecked() != on:
            self.lens.blockSignals(True)
            self.lens.setChecked(on)
            self.lens.blockSignals(False)
        self.lensChanged.emit(on)

    def lens_state(self):
        """-> the view state a project saves: {'lens': bool, 'fov_normal': deg}"""
        return {'lens': self.lens_on(), 'fov_normal': float(self.lens_saved or self.NORMAL_FOV)
                if self.lens_on() else float(self.vp.cam.fov)}

    def restore_lens(self, view):
        """after a project's camera is restored: the lens box and menu show it, and unticking returns to the saved
        normal field of view (35 deg for a project saved before hand1b)"""
        fov_n = float((view or {}).get('fov_normal', self.NORMAL_FOV))
        self.lens_saved = fov_n if self.lens_on() else None
        self._show_lens()


class _Tex(list):
    """A list of mip levels that also knows the original image size (pack uses W0 for pixel-coordinate mapping)."""

    def __init__(self, levels, W0, H0):
        super().__init__(levels)
        self.W0, self.H0 = W0, H0
