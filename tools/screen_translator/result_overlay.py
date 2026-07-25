"""Floating always-on-top overlay that draws translations at each detected
balloon's bbox.

Sized and positioned to match the captured screen region. Block coordinates
from the pipeline are in capture-IMAGE space, which is physical pixels, while
the widget's geometry is logical — the two differ by the screen's
devicePixelRatio, so `source_size` is used to rescale them. Click or ESC
dismisses; optional auto-timeout.
"""

from dataclasses import replace
from typing import List, Optional

import numpy as np
from qtpy.QtCore import Qt, QRect, QTimer, Signal
from qtpy.QtGui import (
    QPainter, QColor, QImage, QPen, QFont, QFontMetrics, QPalette, QPixmap,
    QGuiApplication,
)
from qtpy.QtWidgets import QWidget

from .pipeline import TranslatedBlock


class ResultOverlay(QWidget):

    closed = Signal()

    # Padding inside each translation box (used by both layout & paint).
    BOX_PADDING_X = 6
    BOX_PADDING_Y = 4
    # Smallest font we'll shrink to. Below this Korean glyphs are unreadable
    # even on a high-DPI monitor. If shrinking to this minimum still doesn't
    # fit, the box height grows as a last-resort fallback.
    MIN_FONT_PT = 8
    # Bitmask of QPainter::drawText flags used for both measurement and paint.
    TEXT_FLAGS = int(Qt.AlignmentFlag.AlignTop
                     | Qt.AlignmentFlag.AlignLeft
                     | Qt.TextFlag.TextWordWrap)

    def __init__(self, blocks: List[TranslatedBlock], capture_xywh: tuple,
                 *, inpainted_img: Optional[np.ndarray] = None,
                 raw_img: Optional[np.ndarray] = None,
                 source_size: Optional[tuple] = None,
                 auto_dismiss_ms: int = 0, box_opacity: float = 0.85,
                 font_family: str = 'Malgun Gothic', font_size: int = 14):
        super().__init__()
        self.blocks = [b for b in blocks if b.translation.strip()]
        self.box_opacity = box_opacity
        self.font_family = font_family
        self.max_font_pt = font_size
        # `self.font` is the *configured* font, used for things outside the
        # per-block boxes (the corner hint). Each block picks its own size.
        self.font = QFont(font_family, font_size)
        self.font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)

        # Inpainted background: when present, the overlay paints this image
        # at 1:1 over the captured region and renders translation text
        # directly on top (no white boxes — the original text is already
        # erased from the background). When None, falls back to translucent
        # white boxes over the unmodified screen content.
        self._background_pixmap = self._to_pixmap(inpainted_img) if inpainted_img is not None else None

        # The unmodified capture. Not painted on screen — in translucent mode
        # the live screen showing through IS the page. It exists so the
        # clipboard copy has something opaque to composite onto; see
        # _copy_to_clipboard.
        self._raw_pixmap = self._to_pixmap(raw_img) if raw_img is not None else None

        # Frameless, always-on-top, transparent, no taskbar entry. Cursor
        # stays normal so users can move their mouse off to the side; the
        # whole widget swallows clicks to close.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        # Only request a translucent background when we DON'T have an
        # inpainted background. With the inpainted image, the whole overlay
        # is opaque (the image fills it), and TranslucentBackground would
        # cost an unnecessary compositing layer on top.
        if self._background_pixmap is None:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Stash the original capture dimensions BEFORE layout — _compute_layout
        # needs them as the soft upper bound for box expansion (boxes can grow
        # to the edge of the captured region but not beyond, so we don't
        # bleed outside the user's chosen area).
        _, _, orig_cap_w, orig_cap_h = capture_xywh
        self._orig_capture_size = (orig_cap_w, orig_cap_h)

        # Block coords arrive in capture-image (physical) pixels; the widget is
        # laid out in logical ones. Convert before laying anything out so the
        # font-fitting search also works against logical box sizes.
        self.blocks = self._to_overlay_space(self.blocks, source_size)

        # For each block, decide the largest font size that fits, expanding
        # the box into whatever room is available between neighboring blocks.
        # See _compute_layout's docstring for the full algorithm.
        self._layout = self._compute_layout()

        # The overlay is exactly the region the user dragged. Boxes are kept
        # inside it by _clamp_to_region, so there is nothing hanging past the
        # edges to make room for. Growing the window instead (what this used
        # to do) added blank strips on the right/bottom and drew the text out
        # there, away from the balloon it belongs to.
        x, y, w, h = capture_xywh

        self._capture_xywh = (x, y, w, h)
        self.setGeometry(x, y, w, h)

        # Right-click copy state. `_suppress_hint` blanks the corner dismiss
        # hint during a grab() so the copied image is clean. `_show_copy_hint`
        # briefly replaces the dismiss text with a "복사됨" confirmation after
        # the copy lands on the clipboard.
        self._suppress_hint = False
        self._show_copy_hint = False

        if auto_dismiss_ms > 0:
            QTimer.singleShot(auto_dismiss_ms, self.close)

    # ── Layout ──────────────────────────────────────────────────────────────

    # Alignment for balloon-filled text: centered both ways + word wrap.
    ALIGN_CENTER = int(Qt.AlignmentFlag.AlignHCenter
                       | Qt.AlignmentFlag.AlignVCenter
                       | Qt.TextFlag.TextWordWrap)

    def _compute_layout(self):
        """For each block, decide the font size, final box rect, and text
        alignment.

        Returns a list of dicts (one per block, same order as self.blocks):
            {'rect': (x1, y1, x2, y2), 'font_pt': int, 'align': int}

        Two layout modes:
          - Balloon mode (block has balloon_xyxy): lay the translation out to
            FILL the whole balloon, centered. This is the normal case once
            balloon detection + inpaint are on, and it's what makes vertical-
            Japanese balloons read correctly after translation to horizontal
            Korean.
          - Text-box mode (no balloon): the original behavior — start at the
            text bbox, top-left aligned, and expand into free space.
        """
        out = []
        for idx, blk in enumerate(self.blocks):
            balloon = getattr(blk, 'balloon_xyxy', None)
            if balloon is not None:
                out.append(self._layout_in_balloon(blk, balloon))
            else:
                out.append(self._layout_in_textbox(idx, blk))
        return out

    def _to_overlay_space(self, blocks, source_size):
        """Rescale block coords from capture-image pixels to overlay coords.

        QScreen.grabWindow() takes a LOGICAL rect but hands back a pixmap in
        PHYSICAL pixels, so on a HiDPI screen the captured image — and with it
        every coordinate the pipeline derives from it — is devicePixelRatio
        times larger than this widget, whose geometry is logical. Left
        unscaled, a block's text lands further down-right the further it sits
        from the top-left corner, drawn over the artwork instead of its
        balloon. The inpainted background hid the mismatch because it is
        explicitly scaled into the region when painted.

        No-ops at ratio 1 (no HiDPI scaling) and when `source_size` is unknown.
        """
        if not source_size:
            return blocks
        src_w, src_h = source_size
        if src_w <= 0 or src_h <= 0:
            return blocks
        cap_w, cap_h = self._orig_capture_size
        sx, sy = cap_w / src_w, cap_h / src_h
        if abs(sx - 1) < 1e-3 and abs(sy - 1) < 1e-3:
            return blocks

        def rescale(box):
            if box is None:
                return None
            x1, y1, x2, y2 = box
            return (int(round(x1 * sx)), int(round(y1 * sy)),
                    int(round(x2 * sx)), int(round(y2 * sy)))

        return [replace(b, xyxy=rescale(b.xyxy),
                        balloon_xyxy=rescale(b.balloon_xyxy)) for b in blocks]

    def _clamp_to_region(self, rect):
        """Slide `rect` (x1, y1, x2, y2) back inside the captured region.

        The overlay widget is exactly the captured region, so anything outside
        it is clipped by the window surface and simply never seen. Moving a box
        back in keeps every glyph visible; the text may then sit off its
        balloon, which is the lesser evil — before this, an overflowing box
        made the widget grow and the text was drawn in the blank strip that
        appeared on the right/bottom, nowhere near its balloon.

        Size is preserved. A box genuinely larger than the region is pinned to
        the near edge and trimmed to the region, since there is nowhere else
        for it to go.
        """
        cap_w, cap_h = self._orig_capture_size
        x1, y1, x2, y2 = (int(v) for v in rect)
        w, h = max(1, x2 - x1), max(1, y2 - y1)

        if w >= cap_w:
            x1, w = 0, cap_w
        else:
            x1 = min(max(x1, 0), cap_w - w)
        if h >= cap_h:
            y1, h = 0, cap_h
        else:
            y1 = min(max(y1, 0), cap_h - h)

        return (x1, y1, x1 + w, y1 + h)

    def _layout_in_balloon(self, blk, balloon):
        """Fit the translation to the balloon, centered.

        Readability order: fill the balloon at the largest font that fits →
        spill outside the balloon rather than shrink further → shrink only when
        even the region has no room. Showing all of the text matters more than
        respecting the balloon outline, so the box is allowed to grow past it;
        what is never acceptable is text clipped away where the balloon still
        looks half empty.
        """
        bx1, by1, bx2, by2 = [int(v) for v in balloon]
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        cap_w, cap_h = self._orig_capture_size

        # 1) Largest font whose wrapped text fits the balloon untouched.
        for pt in range(self.max_font_pt, self.MIN_FONT_PT - 1, -1):
            nw, nh = self._needed_size(blk.translation, pt, bw)
            if nw <= bw and nh <= bh:
                return {
                    'rect': self._clamp_to_region((bx1, by1, bx2, by2)),
                    'font_pt': pt,
                    'align': self.ALIGN_CENTER,
                }

        # 2) Too small at every font size. Keep the font as large as possible
        #    and let the box spill outside the balloon instead, bounded by the
        #    captured region, centered so the text stays visually attached.
        for pt in range(self.max_font_pt, self.MIN_FONT_PT - 1, -1):
            fitted = self._fit_in_expanded_box(
                blk.translation, pt, bw, cap_w, cap_h,
            )
            if fitted is not None:
                w, h = fitted
                return {
                    'rect': self._clamp_to_region(
                        self._centered_on(balloon, max(bw, w), max(bh, h))),
                    'font_pt': pt,
                    'align': self.ALIGN_CENTER,
                }

        # 3) Even the minimum font needs more than the whole region. Take its
        #    natural size so as much as possible shows; the clamp trims the
        #    rest because there is nowhere left to grow.
        nw, nh = self._needed_size(blk.translation, self.MIN_FONT_PT, cap_w)
        return {
            'rect': self._clamp_to_region(
                self._centered_on(balloon, max(bw, nw), max(bh, nh))),
            'font_pt': self.MIN_FONT_PT,
            'align': self.ALIGN_CENTER,
        }

    def _fit_in_expanded_box(self, text, pt, min_w, max_w, max_h):
        """Smallest (w, h) that holds `text` at font `pt`, searching widths in
        [min_w, max_w] and requiring height <= max_h. None if it never fits."""
        fm = QFontMetrics(QFont(self.font_family, pt))
        pad_x = 2 * self.BOX_PADDING_X
        pad_y = 2 * self.BOX_PADDING_Y

        def needed_at(w):
            inner_w = max(1, w - pad_x)
            bbox = fm.boundingRect(0, 0, inner_w, 0, self.TEXT_FLAGS, text)
            return (bbox.width() + pad_x, bbox.height() + pad_y)

        return self._smallest_fitting_width(
            needed_at, max(1, min_w), max_w, max_h,
        )

    @staticmethod
    def _centered_on(box, w, h):
        """A (x1, y1, x2, y2) rect of size w x h sharing `box`'s center."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx, ny = cx - w // 2, cy - h // 2
        return (nx, ny, nx + w, ny + h)

    def _layout_in_textbox(self, idx, blk):
        """Original text-bbox layout: top-left aligned, expand into free
        space between neighbors. Used when no balloon was detected."""
        x1, y1, x2, y2 = blk.xyxy
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        max_right, max_bottom = self._expansion_bounds(idx)
        max_w = max(bw, max_right - x1)
        max_h = max(bh, max_bottom - y1)

        chosen_pt = self.MIN_FONT_PT
        chosen_w, chosen_h = bw, bh
        found = False
        for pt in range(self.max_font_pt, self.MIN_FONT_PT - 1, -1):
            result = self._try_fit_at_font(
                blk.translation, pt, bw, bh, max_w, max_h,
            )
            if result is not None:
                chosen_pt = pt
                chosen_w, chosen_h = result
                found = True
                break

        if not found:
            # The room between neighbours isn't enough at any font size. Spill
            # past them — bounded by the captured region — instead of shrinking
            # to the minimum and clipping. Overlapping a neighbour still reads;
            # truncated text does not.
            cap_w, cap_h = self._orig_capture_size
            for pt in range(self.max_font_pt, self.MIN_FONT_PT - 1, -1):
                fitted = self._fit_in_expanded_box(
                    blk.translation, pt, bw, cap_w, cap_h,
                )
                if fitted is not None:
                    chosen_pt = pt
                    chosen_w, chosen_h = fitted
                    found = True
                    break

        if not found:
            # Needs more than the whole region even at the minimum font.
            chosen_pt = self.MIN_FONT_PT
            nw, nh = self._needed_size(blk.translation, self.MIN_FONT_PT, cap_w)
            chosen_w = max(bw, nw)
            chosen_h = max(bh, nh)

        return {
            'rect': self._clamp_to_region((x1, y1, x1 + chosen_w, y1 + chosen_h)),
            'font_pt': chosen_pt,
            'align': self.TEXT_FLAGS,
        }

    def _needed_size(self, text, pt, box_w):
        """(outer_w, outer_h) the wrapped `text` needs in a box of width
        `box_w` at font size `pt`, including padding. outer_w can exceed
        box_w if an unbreakable token doesn't fit (horizontal overflow)."""
        fm = QFontMetrics(QFont(self.font_family, pt))
        inner_w = max(1, box_w - 2 * self.BOX_PADDING_X)
        bbox = fm.boundingRect(0, 0, inner_w, 0, self.TEXT_FLAGS, text)
        return (bbox.width() + 2 * self.BOX_PADDING_X,
                bbox.height() + 2 * self.BOX_PADDING_Y)

    def _expansion_bounds(self, idx):
        """For block at `idx`, return (max_right_x, max_bottom_y) — the
        furthest right/bottom edge the block can grow to without crossing
        any other block's left/top edge.

        Two-sided check: the other block must (a) actually be to the right /
        below us AND (b) overlap our perpendicular range. A block far above
        and to the right doesn't constrain rightward expansion of our row.

        Bounds also clamp to the original captured region — boxes don't
        bleed past the user's drag.
        """
        x1, y1, x2, y2 = self.blocks[idx].xyxy
        orig_cap_w, orig_cap_h = self._orig_capture_size
        max_right = orig_cap_w
        max_bottom = orig_cap_h
        for i, other in enumerate(self.blocks):
            if i == idx:
                continue
            ox1, oy1, ox2, oy2 = other.xyxy
            # Other to the right AND vertically overlapping → constrains width.
            if ox1 >= x2 and not (oy2 <= y1 or oy1 >= y2):
                max_right = min(max_right, ox1)
            # Other below AND horizontally overlapping → constrains height.
            if oy1 >= y2 and not (ox2 <= x1 or ox1 >= x2):
                max_bottom = min(max_bottom, oy1)
        return max_right, max_bottom

    def _try_fit_at_font(self, text, pt, orig_w, orig_h, max_w, max_h):
        """Find the smallest (w, h) ∈ [orig_w..max_w] × [orig_h..max_h] in
        which `text` fits at font size `pt` with word wrap, or None if it
        doesn't fit even at max expansion.

        "Fits" means both: the rendered text doesn't overflow the box's
        horizontal bounds (bbox.width() <= box_inner_w) AND it fits in the
        vertical bounds (bbox.height() <= box_inner_h). Both checks matter
        because Korean words have no inter-syllable break points — wrap a
        long word at too-narrow a width and Qt renders it overflowing past
        the right edge while still reporting a tall bbox.

        Priority within a font size:
          a) Original size — no expansion.
          b) Expand in the direction matching original aspect (narrow-tall
             balloons grow wider; wide-short balloons grow taller).
          c) Expand both if (b) isn't enough.
        """
        fm = QFontMetrics(QFont(self.font_family, pt))
        pad_x = 2 * self.BOX_PADDING_X
        pad_y = 2 * self.BOX_PADDING_Y

        def needed_at(w):
            """Return (needed_outer_w, needed_outer_h) for a box of width `w`.

            Both include padding. needed_outer_w may exceed `w` when a
            single word can't fit in `w - pad_x` — that's the horizontal
            overflow case we have to detect.
            """
            inner_w = max(1, w - pad_x)
            bbox = fm.boundingRect(0, 0, inner_w, 0, self.TEXT_FLAGS, text)
            return (bbox.width() + pad_x, bbox.height() + pad_y)

        # a) Original size — both axes must fit.
        nw, nh = needed_at(orig_w)
        if nw <= orig_w and nh <= orig_h:
            return (orig_w, orig_h)

        # b) Aspect-driven preference: a balloon that's taller than wide
        # (typical vertical Japanese) prefers width expansion; a balloon
        # that's wider than tall prefers height expansion.
        prefer_width_expansion = orig_h > orig_w

        if prefer_width_expansion:
            # Find smallest width that lets text fit in original height.
            result = self._smallest_fitting_width(
                needed_at, orig_w + 1, max_w, orig_h,
            )
            if result is not None:
                return result
            # Otherwise allow height to grow up to max_h.
            return self._smallest_fitting_width(
                needed_at, orig_w + 1, max_w, max_h,
            )
        else:
            # Height-first: keep original width, grow height.
            if nw <= orig_w and nh <= max_h:
                return (orig_w, nh)
            # Otherwise grow width too.
            return self._smallest_fitting_width(
                needed_at, orig_w + 1, max_w, max_h,
            )

    @staticmethod
    def _smallest_fitting_width(needed_at, w_lo, w_hi, h_limit):
        """Binary-search the smallest integer w in [w_lo, w_hi] for which
        the text fits without horizontal overflow AND within height h_limit.

        The "fits" predicate is monotonically True past some critical w:
        as w grows, bbox.width() either stays at the natural min (for an
        unbreakable word) or rises toward w, while bbox.height() falls.
        So once both pass, they stay passing — making binary search valid.

        Returns (w, h_needed) or None if even w_hi doesn't fit.
        """
        if w_lo > w_hi:
            return None
        nw, nh = needed_at(w_hi)
        if nw > w_hi or nh > h_limit:
            return None
        best = None
        lo, hi = w_lo, w_hi
        while lo <= hi:
            mid = (lo + hi) // 2
            nw, nh = needed_at(mid)
            if nw <= mid and nh <= h_limit:
                best = (mid, nh)
                hi = mid - 1
            else:
                lo = mid + 1
        return best

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def show_overlay(self):
        self.show()
        # Re-assert position after show() — on Windows multi-monitor, a
        # frameless + translucent + Tool window's pre-show setGeometry()
        # call is sometimes ignored and the window pops up on the primary
        # screen instead of where it was placed. move() after show() forces
        # the window back to the captured region's actual coordinates.
        x, y, _w, _h = self._capture_xywh
        self.move(x, y)
        self.raise_()

    def mousePressEvent(self, event):
        # Right-click copies the rendered overlay (inpainted background +
        # translation text) to the clipboard without dismissing — the user
        # often wants to keep reading after grabbing the image. Any other
        # button dismisses, preserving the original click-to-close UX.
        if event.button() == Qt.MouseButton.RightButton:
            self._copy_to_clipboard()
            return
        self.close()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Space):
            self.close()

    def _copy_to_clipboard(self):
        """Grab the overlay's current rendering and push it to the system
        clipboard as a QPixmap.

        The corner dismiss hint is suppressed during the grab so it doesn't
        leak into the copied image — users pasting into a doc/chat want the
        translated panel, not our UI chrome. After the copy, the hint flips
        to a "복사됨 ✓" confirmation for 1.5 s so the user can tell it
        worked (the overlay stays open).

        Without an inpainted background the widget is translucent and paints
        only the boxes and text, so grab() comes back ~99% transparent — the
        page itself is the live screen underneath, which a widget grab cannot
        see. Pasted anywhere with a white background that reads as a blank,
        washed-out page. Compositing the grab over the captured pixels gives
        the copy the artwork it is supposed to carry.
        """
        self._suppress_hint = True
        try:
            pixmap = self.grab()
        finally:
            self._suppress_hint = False

        base = self._background_pixmap or self._raw_pixmap
        if base is not None and pixmap.hasAlphaChannel():
            flat = QPixmap(pixmap.size())
            flat.setDevicePixelRatio(pixmap.devicePixelRatio())
            painter = QPainter(flat)
            cap_w, cap_h = self._orig_capture_size
            painter.drawPixmap(0, 0, cap_w, cap_h, base)
            painter.drawPixmap(0, 0, cap_w, cap_h, pixmap)
            painter.end()
            pixmap = flat

        QGuiApplication.clipboard().setPixmap(pixmap)

        self._show_copy_hint = True
        self.update()
        QTimer.singleShot(1500, self._clear_copy_hint)

    def _clear_copy_hint(self):
        # The overlay may already be closed by the time the timer fires
        # (user dismissed before 1.5 s elapsed). Guard against painting a
        # destroyed widget.
        if not self.isVisible():
            return
        self._show_copy_hint = False
        self.update()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    # ── Paint ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pixmap(img: np.ndarray) -> QPixmap:
        """Convert an HxWx3 uint8 RGB numpy array to a QPixmap.

        The inpainter returns the image in the same channel order it was
        given (we feed it RGB from capture_region), so no swap is needed.
        Padded rows in Qt are byte-aligned; we make a contiguous copy to
        guarantee a clean reshape.
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        if img.ndim == 2:
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
        # QImage shares memory with the numpy buffer; copy() detaches so
        # the pixmap survives the source array going out of scope.
        return QPixmap.fromImage(qimg.copy())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Mode A — inpaint background: paint the cleaned image first,
        # then render translations directly on top (no white boxes
        # because the original text is already gone from the pixels).
        # Mode B — translucent: keep the white-box-with-text style on top
        # of the unmodified screen content.
        inpaint_mode = self._background_pixmap is not None
        if inpaint_mode:
            # The inpainted image matches the captured region, which is also
            # the widget size, so this covers the whole surface exactly.
            orig_w, orig_h = self._orig_capture_size
            painter.drawPixmap(0, 0, orig_w, orig_h, self._background_pixmap)

        for blk, item in zip(self.blocks, self._layout):
            x1, y1, x2, y2 = item['rect']
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            rect = QRect(x1, y1, w, h)

            if not inpaint_mode:
                # Mode B: translucent white box over the unmodified region
                # so the original text doesn't bleed through.
                painter.setBrush(QColor(255, 255, 255,
                                        int(255 * self.box_opacity)))
                painter.setPen(QPen(QColor(64, 96, 128, 200), 1))
                painter.drawRoundedRect(rect, 4, 4)

            # Per-block font: _compute_layout picked the largest size where
            # the wrapped translation fits its box. Balloon-filled blocks are
            # centered (align == ALIGN_CENTER); text-box blocks are top-left.
            f = QFont(self.font_family, item['font_pt'])
            f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
            painter.setFont(f)
            painter.setPen(QColor(20, 20, 20))
            painter.drawText(
                rect.adjusted(self.BOX_PADDING_X, self.BOX_PADDING_Y,
                              -self.BOX_PADDING_X, -self.BOX_PADDING_Y),
                item.get('align', self.TEXT_FLAGS),
                blk.translation,
            )

        # Tiny dismiss hint in the corner so first-time users know how to
        # close it (and that right-click copies). Skipped entirely when a
        # clipboard grab is in progress so our UI chrome doesn't end up in
        # the copied image. Uses the configured max font, not any per-block
        # adjusted size — the hint should be consistent regardless of how
        # blocks shrunk.
        if not self._suppress_hint:
            painter.setFont(self.font)
            fm = QFontMetrics(self.font)
            hint = '복사됨 ✓' if self._show_copy_hint \
                else '좌클릭/ESC 닫기 · 우클릭 복사'
            hint_rect = fm.boundingRect(hint)
            bg = QRect(
                self.width() - hint_rect.width() - 16,
                self.height() - hint_rect.height() - 12,
                hint_rect.width() + 12,
                hint_rect.height() + 8,
            )
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bg, 4, 4)
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(bg, Qt.AlignmentFlag.AlignCenter, hint)
