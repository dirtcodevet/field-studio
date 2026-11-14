"""Dark theme (palette + stylesheet) after the concept art's layout: charcoal panels, blue accent.

fs-shell-5 11-14 (the first hands-on test: too little contrast on several things, and some
words and icons still too small to read, the words most), measured with fs-shell/t69/audit_contrast.py:
- the global rule set font-size 9pt on QWidget, and a style sheet's font wins over setFont: EVERY text rendered at
  12 px, the approved 21 px category names and the strip's 22 px tool name included. The size is now the application
  font (FONT_PX, 14 px), so a widget's own size holds;
- the dim greys were 5-6:1 on the panels at 12 px; text is now >= 7:1 on its panel (DIM), disabled text >= 4.5:1
  (OFF), borders a step lighter (LINE) so the panels' edges read."""
from PySide6.QtGui import QColor, QFont, QPalette

FONT_PX = 14            # the body text (was 9 pt = 12 px everywhere, the style sheet overriding every widget's size)
BG = '#16181c'          # window
PANEL = '#1d2026'       # panels
PANEL2 = '#23272e'      # group headers, inputs
LINE = '#3d4450'        # was #2e333b
TEXT = '#e8ebef'        # was #d7dbe0
DIM = '#b4bcc7'         # was #8a919b (5.6:1 on the window, 4.9:1 on an input): now 9:1 / 7.6:1
OFF = '#8f98a4'         # disabled text: 5.9:1 on the window (was DIM)
ACCENT = '#3b82f6'
ACCENT2 = '#2563eb'
GOOD = '#4ade80'
WARN = '#fbbf24'

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI'; }}
QMainWindow::separator {{ background: {LINE}; width: 2px; height: 2px; }}
QMenuBar {{ background: {BG}; border-bottom: 1px solid {LINE}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {PANEL2}; }}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; }}
QMenu::item {{ padding: 6px 26px 6px 16px; }}
QMenu::item:selected {{ background: {ACCENT2}; }}
QMenu::item:disabled {{ color: {OFF}; }}
QToolBar {{ background: {BG}; border: none; spacing: 4px; padding: 3px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px 8px; }}
QToolButton:hover {{ background: {PANEL2}; border-color: {LINE}; }}
QToolButton:checked {{ background: #1e3a5f; border-color: {ACCENT}; }}
QToolButton:disabled {{ color: {OFF}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 4px; padding: 4px 10px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT2}; }}
QPushButton:checked {{ background: #1e3a5f; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {OFF}; border-color: {PANEL2}; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px; margin-top: 16px; padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {TEXT}; font-weight: 600; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {PANEL}; padding: 4px 8px; border-bottom: 1px solid {LINE}; }}
QScrollArea {{ border: none; background: {PANEL}; }}
QScrollArea > QWidget > QWidget {{ background: {PANEL}; }}
QLabel {{ background: transparent; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 3px;
    padding: 2px 4px; selection-background-color: {ACCENT2}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {LINE}; selection-background-color: {ACCENT2}; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #e5e7eb; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {LINE}; border-radius: 3px; background: {PANEL2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QTreeWidget, QListWidget {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 4px; outline: none; }}
QTreeWidget::item, QListWidget::item {{ padding: 2px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: #1e3a5f; color: white; }}
QHeaderView::section {{ background: {PANEL2}; border: none; padding: 3px; }}
QTabBar::tab {{ background: transparent; padding: 6px 14px; border: 1px solid transparent; border-radius: 4px; margin: 2px; }}
QTabBar::tab:selected {{ background: #1e3a5f; border-color: {ACCENT}; }}
QTabBar::tab:hover {{ background: {PANEL2}; }}
QStatusBar {{ background: {BG}; border-top: 1px solid {LINE}; color: {DIM}; font-size: 13px; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {LINE}; }}
QScrollBar:vertical {{ background: {PANEL}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 20px; }}
QScrollBar:horizontal {{ background: {PANEL}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 4px; min-width: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
#telemetry {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px; }}
#telemetryLabel {{ color: {GOOD}; font-weight: 600; font-size: 8pt; }}
#telemetryValue {{ color: {TEXT}; font-size: 10pt; }}
#dim {{ color: {DIM}; }}
"""


def apply(app):
    app.setStyle('Fusion')
    f = QFont('Segoe UI')
    f.setPixelSize(FONT_PX)
    app.setFont(f)
    p = QPalette()
    for role, c in ((QPalette.Window, BG), (QPalette.Base, PANEL2), (QPalette.AlternateBase, PANEL),
                    (QPalette.Text, TEXT), (QPalette.WindowText, TEXT), (QPalette.Button, PANEL2),
                    (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT2), (QPalette.HighlightedText, '#ffffff'),
                    (QPalette.ToolTipBase, PANEL2), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, DIM)):
        p.setColor(role, QColor(c))
    app.setPalette(p)
    app.setStyleSheet(STYLE)
