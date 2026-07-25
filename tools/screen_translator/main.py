"""Screen translator — Yomichan-style overlay translation for images on screen.

Workflow:
  1. App stays resident in the system tray, optionally bound to a global hotkey.
  2. Trigger -> full-screen translucent overlay; user drags a rectangle.
  3. The selected screen region is captured to a numpy array.
  4. Pipeline (text detect -> OCR -> translate) runs on a worker thread.
  5. A frameless always-on-top overlay paints translations at each balloon's
     bbox, positioned to match the original capture region.

This file is the glue — every subsystem lives in its own module.
"""

from __future__ import annotations

import os
import sys
import traceback

# Add project root to sys.path before any sibling imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Force qtpy to pick PyQt6 (BallonsTranslator's default).
os.environ.setdefault('QT_API', 'pyqt6')

import numpy as np
from qtpy.QtCore import Qt, QObject, QRect, QThread, Signal
from qtpy.QtGui import QAction, QGuiApplication, QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication, QMenu, QMessageBox, QSystemTrayIcon, QWidget
)

from tools.screen_translator import config as cfg
from tools.screen_translator.pipeline import TranslationPipeline, TranslatedBlock
from tools.screen_translator.region_selector import RegionSelector
from tools.screen_translator.result_overlay import ResultOverlay


# ── Screen capture ──────────────────────────────────────────────────────────

def capture_region(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Grab pixels from the virtual desktop. Returns an HxWx3 uint8 RGB array.

    Uses Qt's QScreen.grabWindow which handles HiDPI correctly and works on
    every Qt-supported platform without extra deps. On multi-monitor we walk
    each screen and grab from the one that contains the rect's top-left.
    """
    screen = None
    for s in QGuiApplication.screens():
        if s.geometry().contains(x, y):
            screen = s
            break
    if screen is None:
        screen = QGuiApplication.primaryScreen()

    geom = screen.geometry()
    # Convert virtual-desktop coords to local-screen coords for grabWindow.
    local_x = x - geom.x()
    local_y = y - geom.y()

    pixmap: QPixmap = screen.grabWindow(0, local_x, local_y, w, h)
    image: QImage = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)

    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    # Qt rows are padded to 4 bytes; bytesPerLine() tells us the stride.
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
        (height, image.bytesPerLine())
    )[:, : width * 3].reshape((height, width, 3))
    return arr.copy()


# ── Worker thread ───────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    """Runs detect/OCR/translate/inpaint on a background thread so the UI
    stays responsive (the first run can take 5-15s for module + model loading)."""

    progress = Signal(str)
    # (blocks, inpainted_img-or-None). object lets us carry the np.ndarray
    # across the queued signal without Qt trying to wrap it.
    finished_ok = Signal(object, object)
    finished_err = Signal(str, str)    # message, traceback

    def __init__(self, pipeline: TranslationPipeline, settings, img: np.ndarray):
        super().__init__()
        self.pipeline = pipeline
        self.settings = settings
        self.img = img

    def run(self):
        try:
            # Re-apply settings on every capture — if the user changed the
            # translator/OCR in BT's UI since last capture, this is where we
            # pick that up. Unchanged modules are no-ops.
            self.pipeline.apply_settings(self.settings, progress_cb=self.progress.emit)
            # translate_image emits its own per-phase progress
            # ("Detecting text...", "OCR (N blocks)...", "Translating (N blocks)...",
            #  "Inpainting (N blocks)...") via the same signal.
            blocks, inpainted_img = self.pipeline.translate_image(
                self.img, progress_cb=self.progress.emit,
            )
            self.finished_ok.emit(blocks, inpainted_img)
        except Exception as e:
            self.finished_err.emit(str(e), traceback.format_exc())


# ── Status indicator (small floating label during pipeline run) ─────────────

class StatusIndicator(QWidget):
    """Tiny corner toast that mirrors PipelineWorker.progress messages plus
    a simple progress bar at the bottom."""

    # Toast dimensions. Slight height bump over the original (50→78) makes
    # room for the progress bar without crowding the message line.
    TOAST_W = 300
    TOAST_H = 78
    BAR_HEIGHT = 6     # progress bar thickness
    BAR_MARGIN_X = 14
    BAR_MARGIN_BOTTOM = 12

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._message = ''
        # 0.0 - 1.0 explicit progress; -1 means "indeterminate" (no bar shown).
        self._progress = -1.0

    def set_message(self, msg: str):
        self._message = msg
        self.update()

    def set_progress(self, frac: float):
        """Update the progress bar. Pass 0.0 to clear, 1.0 for complete,
        or -1 to hide the bar entirely (indeterminate / non-progress toast)."""
        if frac < 0:
            self._progress = -1.0
        else:
            self._progress = max(0.0, min(1.0, frac))
        self.update()

    def reset(self):
        """Reset to fresh state — call before starting a new task."""
        self._message = ''
        self._progress = 0.0
        self.update()

    def show_near_cursor(self):
        cursor_pos = QGuiApplication.primaryScreen().geometry().bottomRight()
        # Place toast near bottom-right of primary screen.
        self.setGeometry(
            cursor_pos.x() - self.TOAST_W - 20,
            cursor_pos.y() - self.TOAST_H - 30,
            self.TOAST_W, self.TOAST_H,
        )
        self.show()
        self.raise_()

    def paintEvent(self, event):
        from qtpy.QtGui import QPainter, QColor, QFont
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background panel.
        painter.setBrush(QColor(20, 20, 20, 220))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 6, 6)

        # Reserve space for the progress bar at the bottom; message text
        # lives above it. When the bar is hidden (progress < 0), let the
        # message use the full height.
        has_bar = self._progress >= 0
        text_bottom = (
            self.height() - self.BAR_HEIGHT - self.BAR_MARGIN_BOTTOM - 4
            if has_bar else self.height()
        )

        # Message — top region.
        painter.setPen(QColor(230, 230, 230))
        font = QFont('Malgun Gothic', 11)
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(14, 8, -14, -(self.height() - text_bottom)),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self._message,
        )

        if not has_bar:
            return

        # Progress bar — full-width track + filled portion.
        bar_y = self.height() - self.BAR_HEIGHT - self.BAR_MARGIN_BOTTOM
        bar_w = self.width() - 2 * self.BAR_MARGIN_X
        # Track (unfilled background).
        painter.setBrush(QColor(70, 70, 70, 200))
        painter.drawRoundedRect(
            self.BAR_MARGIN_X, bar_y, bar_w, self.BAR_HEIGHT,
            self.BAR_HEIGHT / 2, self.BAR_HEIGHT / 2,
        )
        # Fill — accent blue, scales 0..1 of bar width.
        filled_w = int(bar_w * self._progress)
        if filled_w > 0:
            painter.setBrush(QColor(64, 168, 255, 240))
            painter.drawRoundedRect(
                self.BAR_MARGIN_X, bar_y, filled_w, self.BAR_HEIGHT,
                self.BAR_HEIGHT / 2, self.BAR_HEIGHT / 2,
            )

        # Tiny percentage label at the right end of the bar (small, dim).
        painter.setPen(QColor(180, 180, 180))
        pct_font = QFont('Malgun Gothic', 8)
        painter.setFont(pct_font)
        painter.drawText(
            self.rect().adjusted(
                self.BAR_MARGIN_X, bar_y - 14,
                -self.BAR_MARGIN_X, -(self.BAR_HEIGHT + self.BAR_MARGIN_BOTTOM),
            ),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            f'{int(self._progress * 100)}%',
        )


# ── Controller ──────────────────────────────────────────────────────────────

class ScreenTranslatorApp(QObject):
    """Top-level glue. Owns the pipeline (single instance, shared across
    captures), the tray icon, and chains region_selector -> capture ->
    worker -> result_overlay for each trigger."""

    # Hotkey thread emits this; the Qt main thread handles the slot. This is
    # how we marshal off the `keyboard` package's daemon thread back into Qt.
    _hotkey_pressed = Signal()

    def __init__(self):
        super().__init__()
        self.pipeline = TranslationPipeline()
        # Resolve once at startup just to pick up the hotkey / UI options.
        # Module-level settings are re-resolved fresh on each capture.
        self._startup_settings = cfg.resolve()

        self._selector: RegionSelector | None = None
        self._overlay: ResultOverlay | None = None
        self._worker: PipelineWorker | None = None
        self._status = StatusIndicator()
        self._busy = False

        self._build_tray()
        self._install_hotkey()
        self._hotkey_pressed.connect(self.start_capture)

    # ── Setup ───────────────────────────────────────────────────────────────

    def _build_tray(self):
        # Use BallonsTranslator's icon if available; otherwise fall back to
        # a built-in Qt icon so the tray still has a visible glyph.
        icon_path = os.path.join(_ROOT, 'icons', 'BallonsTranslator.ico')
        if not os.path.exists(icon_path):
            icon_path = os.path.join(_ROOT, 'icons', 'icon.ico')
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            icon = QApplication.style().standardIcon(
                QApplication.style().StandardPixmap.SP_DesktopIcon
            )

        self.tray = QSystemTrayIcon(icon)
        self.tray.setToolTip(
            'Screen Translator\n'
            f'Hotkey: {self._startup_settings.global_hotkey or "(disabled)"}'
        )

        menu = QMenu()
        capture_action = QAction('Capture && Translate', self)
        capture_action.triggered.connect(self.start_capture)
        if self._startup_settings.global_hotkey:
            capture_action.setShortcut(self._startup_settings.global_hotkey)
        menu.addAction(capture_action)

        menu.addSeparator()

        about_action = QAction('About', self)
        about_action.triggered.connect(self._show_about)
        menu.addAction(about_action)

        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        # Left-click also triggers capture (mirrors hotkey for mouse users).
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason):
        # Reason.Trigger is left-click; Reason.Context handled by setContextMenu.
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.start_capture()

    def _install_hotkey(self):
        hotkey = self._startup_settings.global_hotkey
        if not hotkey:
            return
        try:
            import keyboard
        except ImportError:
            print('[screen_translator] keyboard pkg not installed; '
                  'global hotkey disabled')
            return
        try:
            keyboard.add_hotkey(hotkey, lambda: self._hotkey_pressed.emit())
            print(f'[screen_translator] global hotkey registered: {hotkey}')
        except Exception as e:
            # On Windows non-admin sometimes can't bind certain combos.
            print(f'[screen_translator] failed to register hotkey: {e}')

    # ── Capture flow ────────────────────────────────────────────────────────

    def start_capture(self):
        if self._busy:
            print('[screen_translator] busy, ignoring trigger')
            return
        # Dismiss any prior overlay so it doesn't sit on top of the new
        # region selector.
        if self._overlay is not None:
            self._overlay.close()
            self._overlay = None

        self._selector = RegionSelector()
        self._selector.region_selected.connect(self._on_region_selected)
        self._selector.cancelled.connect(self._on_capture_cancelled)
        self._selector.show_on_virtual_desktop()

    def _on_capture_cancelled(self):
        self._selector = None

    def _on_region_selected(self, x: int, y: int, w: int, h: int):
        self._selector = None
        try:
            img = capture_region(x, y, w, h)
        except Exception as e:
            self._show_error('Capture failed', str(e), traceback.format_exc())
            return

        print(f'[screen_translator] capture: virtual=({x},{y}) {w}x{h}  '
              f'image={img.shape}')

        # grabWindow() returns PHYSICAL pixels for a LOGICAL rect, so on a
        # HiDPI screen this is larger than (w, h). The overlay needs it to map
        # pipeline block coords back into its own logical geometry.
        self._capture_img_size = (img.shape[1], img.shape[0])
        # Kept so the overlay's clipboard copy can composite onto the page even
        # when inpainting produced nothing to paint.
        self._capture_img = img

        # Re-read BT's config.json on every capture so settings changed in the
        # main app (translator switch, language pair, API key, etc.) take
        # effect immediately without restarting this process. Resolve happens
        # on the main thread (cheap JSON read); module rebuilds happen on the
        # worker thread.
        settings = cfg.resolve()
        print(f'[screen_translator] settings: detector={settings.text_detector}  '
              f'ocr={settings.ocr_module}  translator={settings.translator_module}  '
              f'lang={settings.source_lang}->{settings.target_lang}')

        self._busy = True
        self._status.reset()
        self._status.set_message('Loading pipeline...')
        self._status.show_near_cursor()

        self._worker = PipelineWorker(self.pipeline, settings, img)
        # Route progress through our own handler so we can map the worker's
        # phase-name strings to an estimated fraction for the progress bar
        # before forwarding to the toast.
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished_ok.connect(
            lambda blocks, inpainted: self._on_translation_done(
                blocks, inpainted, (x, y, w, h), settings,
            )
        )
        self._worker.finished_err.connect(self._on_translation_err)
        self._worker.start()

    # ── Progress mapping ────────────────────────────────────────────────────

    # Pipeline-phase keyword → fractional progress. Higher than 0.95 is
    # reserved for completion handlers so the bar visibly snaps to 100%
    # only when finished_ok fires.
    _PROGRESS_MAP = (
        ('text detector', 0.05),
        ('loading ocr',   0.15),
        ('translator',    0.25),
        ('inpainter',     0.35),   # apply_settings: "Loading inpainter: ..."
        ('ready',         0.40),
        ('detecting',     0.50),   # translate_image phase 1
        ('ocr (',         0.60),   # translate_image phase 2
        ('translating',   0.75),   # translate_image phase 3
        ('inpainting',    0.90),   # translate_image phase 4
    )

    def _on_worker_progress(self, msg: str):
        """Forward the message to the toast AND advance the progress bar
        based on which phase the message indicates."""
        self._status.set_message(msg)
        low = msg.lower()
        for keyword, frac in self._PROGRESS_MAP:
            if keyword in low:
                self._status.set_progress(frac)
                return
        # Unknown message — leave progress bar at whatever it currently is.

    def _on_translation_done(self, blocks, inpainted_img, capture_xywh, settings):
        self._busy = False
        self._worker = None

        # Categorize the outcome up front so we can give clear feedback for
        # each failure mode (silent fallback to empty translations is the
        # most user-confusing case — looks like a black hole if we don't
        # call it out).
        detected_count = len(blocks)
        translated_count = sum(1 for b in blocks if b.translation.strip())
        print(f'[screen_translator] result: detected={detected_count}  '
              f'translated={translated_count}')

        # Case 1: nothing detected — drag missed the text or detector failed.
        if detected_count == 0:
            self._status.hide()
            self._toast_briefly('텍스트 없음 (감지된 풍선 0개)', 2000)
            return

        # Case 2: text detected but ALL translations are empty — the OCR
        # or translator silently returned empty strings. Common culprits:
        # Local LLM server not running at the configured base_url, network
        # block on a cloud translator, or OCR failed to read the crops.
        if translated_count == 0:
            self._status.hide()
            self._toast_briefly(
                f'번역 실패: {detected_count}개 검출, 번역 0개\n'
                f'[{settings.translator_module}] 응답이 비어 있습니다.\n'
                f'런처 콘솔 로그를 확인하세요.',
                4000,
            )
            return

        # Case 3: at least one translation — open the overlay.
        x, y, w, h = capture_xywh
        print(f'[screen_translator] overlay: at virtual=({x},{y}) {w}x{h} '
              f'with {translated_count}/{detected_count} translated blocks')

        # Snap progress to 100% briefly so the user sees the bar complete,
        # then hide the toast just before showing the overlay.
        self._status.set_message('완료')
        self._status.set_progress(1.0)
        from qtpy.QtCore import QTimer
        QTimer.singleShot(350, self._status.hide)

        self._overlay = ResultOverlay(
            blocks=blocks,
            capture_xywh=capture_xywh,
            inpainted_img=inpainted_img,
            raw_img=getattr(self, '_capture_img', None),
            source_size=getattr(self, '_capture_img_size', None),
            auto_dismiss_ms=settings.overlay_auto_dismiss_ms,
            box_opacity=settings.overlay_box_opacity,
            font_family=settings.overlay_font_family,
            font_size=settings.overlay_font_size,
        )
        self._overlay.closed.connect(self._on_overlay_closed)
        self._overlay.show_overlay()
        # Sanity-log where Qt actually placed the overlay window — diverges
        # from the requested geometry on some multi-monitor configs.
        g = self._overlay.geometry()
        print(f'[screen_translator] overlay shown: actual geometry='
              f'({g.x()},{g.y()}) {g.width()}x{g.height()}  '
              f'visible={self._overlay.isVisible()}')

    def _toast_briefly(self, message: str, duration_ms: int):
        """Show a status toast for the given duration. Used for short-lived
        feedback (text-not-found, translation-failed) where opening a full
        overlay would be misleading or pointless. The progress bar is
        hidden — these aren't progress events."""
        from qtpy.QtCore import QTimer
        self._status.set_message(message)
        self._status.set_progress(-1)  # hide bar — not a progress event
        self._status.show_near_cursor()
        QTimer.singleShot(duration_ms, self._status.hide)

    def _on_overlay_closed(self):
        self._overlay = None

    def _on_translation_err(self, msg, tb):
        self._busy = False
        self._status.hide()
        self._worker = None
        self._show_error('Translation failed', msg, tb)

    # ── Misc ────────────────────────────────────────────────────────────────

    def _show_about(self):
        # Re-resolve so the dialog reflects current BT settings, not the ones
        # at app startup. Helpful for confirming a settings change took.
        s = cfg.resolve()
        QMessageBox.information(
            None, 'Screen Translator',
            'BallonsTranslator screen capture mode\n\n'
            f'Detector: {s.text_detector}\n'
            f'OCR: {s.ocr_module}\n'
            f'Translator: {s.translator_module}\n'
            f'Language: {s.source_lang} → {s.target_lang}\n'
            f'Device: {s.device}\n'
            f'Hotkey: {s.global_hotkey or "(disabled)"}\n\n'
            'Settings inherit from BallonsTranslator\'s main UI.\n'
            'Drag a rectangle around manga panels / images with text.'
        )

    def _show_error(self, title, msg, tb):
        print(f'[screen_translator] ERROR {title}: {msg}')
        print(tb)
        QMessageBox.critical(None, title, f'{msg}\n\n{tb[-800:]}')


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # keep running after overlays close

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(
            None, 'Screen Translator',
            'System tray is not available on this system.'
        )
        sys.exit(1)

    controller = ScreenTranslatorApp()
    # Keep `controller` alive for the app's lifetime by stashing on the app.
    app._screen_translator_controller = controller

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
