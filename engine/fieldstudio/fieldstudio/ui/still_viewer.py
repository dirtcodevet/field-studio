"""fs-store 11-12 (item 5: a way to render hair and to view renders at full resolution).
A still at its full resolution, in its own window beside Field Studio: opens fitted to the window; 1:1 shows every
pixel; the wheel zooms about the cursor (1/8x - 16x); a left drag pans; F fits, 1 shows 1:1; Save as... writes a PNG.
The window is not modal, so sculpting goes on beside it.
    StillViewer(parent, path_or_qimage, title=None).show()"""
import os

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPainter, QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class _Canvas(QWidget):
    def __init__(self, img, parent=None):
        super().__init__(parent)
        self.img = img
        self.scale = 1.0
        self.off = QPointF(0, 0)        # the image's top-left in widget pixels
        self._drag = None
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.fitted = True

    def fit(self):
        if self.img.isNull():
            return
        s = min(self.width() / self.img.width(), self.height() / self.img.height())
        self.scale = s
        self.off = QPointF((self.width() - self.img.width() * s) / 2, (self.height() - self.img.height() * s) / 2)
        self.fitted = True
        self.update()

    def one_to_one(self, about=None):
        self._zoom_to(1.0, about)

    def _zoom_to(self, s, about=None):
        about = about if about is not None else QPointF(self.width() / 2, self.height() / 2)
        s = max(1.0 / 8, min(16.0, s))
        p = (about - self.off) / self.scale            # the image point under the cursor stays there
        self.scale = s
        self.off = about - p * s
        self.fitted = False
        self.update()

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self._zoom_to(self.scale * f, e.position())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = (e.position(), QPointF(self.off))

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            p0, o0 = self._drag
            self.off = o0 + (e.position() - p0)
            self.fitted = False
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def resizeEvent(self, e):
        if self.fitted:
            self.fit()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self.img.isNull():
            return
        # nearest pixels from 1:1 up (every pixel as it is), smooth below 1:1
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 1.0)
        p.translate(self.off)
        p.scale(self.scale, self.scale)
        p.drawImage(0, 0, self.img)
        p.end()


class StillViewer(QDialog):
    def __init__(self, parent, src, title=None):
        super().__init__(parent)
        self.path = src if isinstance(src, str) else None
        img = QImage(src) if isinstance(src, str) else src
        self.setWindowTitle(title or ('Still - %s' % (os.path.basename(self.path) if self.path else 'render')))
        self.setModal(False)
        self.canvas = _Canvas(img, self)
        self.info = QLabel('%d x %d px' % (img.width(), img.height()))
        b_fit, b_one, b_save = QPushButton('Fit (F)'), QPushButton('1:1 (1)'), QPushButton('Save as...')
        b_fit.clicked.connect(self.canvas.fit)
        b_one.clicked.connect(lambda: self.canvas.one_to_one())
        b_save.clicked.connect(self.save_as)
        row = QHBoxLayout()
        for w in (b_fit, b_one, b_save):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(self.info)
        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(row)
        QShortcut(QKeySequence('F'), self, self.canvas.fit)
        QShortcut(QKeySequence('1'), self, lambda: self.canvas.one_to_one())
        self.resize(min(1600, img.width() + 40) if not img.isNull() else 800,
                    min(1000, img.height() + 80) if not img.isNull() else 600)

    def showEvent(self, e):
        super().showEvent(e)
        self.canvas.fit()

    def save_as(self):
        start = self.path or 'render.png'
        p, _ = QFileDialog.getSaveFileName(self, 'Save the still', start, 'PNG image (*.png)')
        if p:
            if not p.lower().endswith('.png'):
                p += '.png'
            self.canvas.img.save(p)
            self.info.setText('saved %s' % p)
