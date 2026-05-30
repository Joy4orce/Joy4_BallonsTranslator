"""Full-screen translucent overlay for dragging a region rectangle.

Covers the entire virtual desktop (all monitors). Press-drag-release picks
a rect; ESC or right-click cancels. Emits region_selected(x, y, w, h) where
coordinates are in the virtual desktop's pixel space (top-left = virtual
geometry origin, which may be negative on multi-monitor setups).
"""

from qtpy.QtCore import Qt, QRect, QPoint, Signal
from qtpy.QtGui import QPainter, QColor, QPen, QGuiApplication
from qtpy.QtWidgets import QWidget


class RegionSelector(QWidget):

    region_selected = Signal(int, int, int, int)  # x, y, w, h (virtual coords)
    cancelled = Signal()

    def __init__(self):
        super().__init__()
        # Frameless, always-on-top, no taskbar entry. Translucent so we can
        # see the screen behind the selection dim layer.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._origin = QPoint()
        self._current = QPoint()
        self._dragging = False

    def show_on_virtual_desktop(self):
        """Resize to cover ALL connected screens (virtual desktop), then show.

        Two important notes:

        1) We compute the union of every screen's geometry ourselves rather
           than relying solely on primaryScreen().virtualGeometry(). On
           multi-GPU setups the latter sometimes only reports the screens
           sharing the primary's display chain — i.e., a single monitor.

        2) We deliberately use plain show(), NOT showFullScreen().
           showFullScreen() snaps the widget to one screen and ignores the
           geometry we set, which is exactly the dual-monitor bug we're
           fixing. Plain show() honors setGeometry, so the frameless +
           always-on-top window genuinely spans the virtual desktop.
        """
        screens = QGuiApplication.screens()
        if not screens:
            geom = QRect(0, 0, 1920, 1080)
        else:
            geom = QGuiApplication.primaryScreen().virtualGeometry()
            for s in screens:
                geom = geom.united(s.geometry())

        self._virtual_origin = geom.topLeft()  # remember for coord mapping
        self.setGeometry(geom)
        self.show()
        self.raise_()
        self.activateWindow()

    # ── Mouse / keyboard ────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.pos()
            self._current = event.pos()
            self._dragging = True
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self._cancel()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        rect = QRect(self._origin, self._current).normalized()
        # Reject single-click / tiny drags (likely accidental).
        if rect.width() < 5 or rect.height() < 5:
            self._cancel()
            return
        # Map widget-local coords back to virtual desktop coords by adding
        # the widget's screen origin.
        x = rect.x() + self._virtual_origin.x()
        y = rect.y() + self._virtual_origin.y()
        self.hide()
        self.region_selected.emit(x, y, rect.width(), rect.height())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()

    def _cancel(self):
        self._dragging = False
        self.hide()
        self.cancelled.emit()

    # ── Paint ───────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        # Dim the whole screen with a semi-transparent black layer so the
        # selection box stands out and the user knows the overlay is active.
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))

        if self._dragging:
            rect = QRect(self._origin, self._current).normalized()
            # Cut the selection area back out by repainting it transparent.
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Selection border.
            pen = QPen(QColor(64, 196, 255), 2)
            painter.setPen(pen)
            painter.drawRect(rect)

            # Size label.
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                rect.bottomRight() + QPoint(6, 16),
                f'{rect.width()} × {rect.height()}'
            )
