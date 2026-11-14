"""Lightweight wrapper around BallonsTranslator's detect/OCR/translate modules.

Each module loads lazily on first use (the first capture is therefore slow —
several seconds — but subsequent ones are fast). All three modules live in
the same Python process, so we share the same torch/CUDA context.

apply_settings() lets the caller reconfigure between captures: only modules
whose name/params actually changed get rebuilt, so changing the translator
in BallonsTranslator's UI is essentially free, while changing the OCR pays
a one-time reload cost.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import numpy as np


# Make BallonsTranslator's modules/utils importable. tools/screen_translator/
# is two levels below the project root.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@dataclass
class TranslatedBlock:
    """Plain-data view of a TextBlock — what the overlay actually consumes.

    xyxy is in the captured image's pixel space (origin = capture top-left).
    The overlay maps these to screen coordinates by adding the capture origin.

    balloon_xyxy, when set, is the bounding box of the speech balloon that
    encloses this text (also in capture-image pixel space). The overlay uses
    it as the text layout area so the translation fills the whole balloon
    instead of starting at the narrow original-text bbox. None when no balloon
    was detected (e.g. SFX, open-air text) → overlay falls back to xyxy.
    """
    xyxy: tuple                   # (x1, y1, x2, y2)
    angle: float                  # rotation degrees
    source_text: str
    translation: str
    balloon_xyxy: tuple = None    # (x1, y1, x2, y2) or None


class TranslationPipeline:
    """Holds the three module instances and orchestrates one image at a time.

    State is driven by apply_settings(SimpleNamespace) — the caller passes a
    fresh resolved-config object before each capture; we diff against the
    last-applied config and rebuild only what changed.
    """

    def __init__(self):
        self._current: Optional[SimpleNamespace] = None
        self._detector = None
        self._ocr = None
        self._translator = None
        self._inpainter = None
        self._translators_imported = False

    # ── Module registration (imports trans_*.py once) ───────────────────────

    def _import_translators(self):
        if self._translators_imported:
            return
        import importlib, re
        translator_dir = os.path.join(_ROOT, 'modules', 'translators')
        pattern = re.compile(r'trans_(.*?).py$')
        for f in os.listdir(translator_dir):
            if pattern.match(f):
                importlib.import_module('modules.translators.' + f[:-3])
        self._translators_imported = True

    # ── Settings application ────────────────────────────────────────────────

    def apply_settings(self, settings: SimpleNamespace, progress_cb=None):
        """Apply a resolved settings namespace, rebuilding modules as needed.

        Diff strategy: each module is rebuilt only when its name OR its
        merged params dict differ from what we last constructed it with.
        Translator additionally rebuilds on source/target language changes
        (BaseTranslator binds those at __init__ time).

        Call this on a worker thread — first invocation triggers model
        downloads, and OCR model load on swap costs ~1s.
        """
        def _emit(msg):
            if progress_cb:
                progress_cb(msg)

        self._import_translators()
        from modules import TEXTDETECTORS, OCR, TRANSLATORS, INPAINTERS

        prev = self._current  # may be None on first call

        det_params = settings.detector_params.get(settings.text_detector)
        if (prev is None
                or prev.text_detector != settings.text_detector
                or self._params_or_device_changed(
                    prev.detector_params.get(prev.text_detector),
                    det_params, prev.device, settings.device)):
            _emit(f'Loading text detector: {settings.text_detector}...')
            det_cls = TEXTDETECTORS.module_dict[settings.text_detector]
            self._detector = det_cls(
                **self._build_module_params(det_cls, det_params, settings.device)
            )
            self._detector.load_model()

        ocr_params = settings.ocr_params.get(settings.ocr_module)
        if (prev is None
                or prev.ocr_module != settings.ocr_module
                or self._params_or_device_changed(
                    prev.ocr_params.get(prev.ocr_module),
                    ocr_params, prev.device, settings.device)):
            _emit(f'Loading OCR: {settings.ocr_module}...')
            ocr_cls = OCR.module_dict[settings.ocr_module]
            self._ocr = ocr_cls(
                **self._build_module_params(ocr_cls, ocr_params, settings.device)
            )
            self._ocr.load_model()

        tr_params = settings.translator_params.get(settings.translator_module)
        if (prev is None
                or prev.translator_module != settings.translator_module
                or prev.source_lang != settings.source_lang
                or prev.target_lang != settings.target_lang
                or self._params_or_device_changed(
                    prev.translator_params.get(prev.translator_module),
                    tr_params, None, None)):
            _emit(f'Loading translator: {settings.translator_module}...')
            tr_cls = TRANSLATORS.module_dict[settings.translator_module]
            merged = self._merge_dict(tr_cls.params, tr_params)
            self._translator = tr_cls(
                settings.source_lang, settings.target_lang,
                raise_unsupported_lang=False,
                **merged,
            )

        # Inpainter — only loaded when enabled (skips download/init cost
        # for users running with ENABLE_INPAINT=False).
        if settings.enable_inpaint:
            ip_params = settings.inpainter_params.get(settings.inpainter_module)
            prev_enabled = prev is not None and prev.enable_inpaint
            if (not prev_enabled
                    or prev.inpainter_module != settings.inpainter_module
                    or self._params_or_device_changed(
                        prev.inpainter_params.get(prev.inpainter_module)
                            if prev_enabled else None,
                        ip_params, prev.device if prev_enabled else None,
                        settings.device)):
                _emit(f'Loading inpainter: {settings.inpainter_module}...')
                ip_cls = INPAINTERS.module_dict[settings.inpainter_module]
                self._inpainter = ip_cls(
                    **self._build_module_params(ip_cls, ip_params, settings.device)
                )
                self._inpainter.load_model()
        else:
            # User disabled inpaint — drop any previously-loaded model so
            # the VRAM/RAM gets reclaimed on the next torch GC.
            self._inpainter = None

        self._current = copy.deepcopy(settings)
        _emit('Pipeline ready.')

    # ── Convenience: original "ensure loaded" entry point ──────────────────

    def ensure_loaded(self, settings: SimpleNamespace, progress_cb=None):
        """First-time load (or no-op if already loaded with these settings)."""
        self.apply_settings(settings, progress_cb=progress_cb)

    # ── Params plumbing ────────────────────────────────────────────────────

    @staticmethod
    def _params_or_device_changed(a: Optional[dict], b: Optional[dict],
                                   dev_a, dev_b) -> bool:
        # Both None → no change. One None one not → changed.
        if (a is None) != (b is None):
            return True
        if a != b:
            return True
        if dev_a != dev_b:
            return True
        return False

    def _build_module_params(self, cls, user_params: Optional[dict],
                              device: str) -> dict:
        """Construct the kwargs to pass to an OCR / detector class __init__.

        Start from the class's advertised default `params` shape, deep-merge
        the user's saved per-module params (preserving selector/editor inner
        shape), then force-set the device selector to our chosen device.

        The device override is intentional — the screen translator may want
        a different device than the main app (e.g., main on CPU for desktop
        recording, screen translator on GPU for snappy captures).
        """
        params = copy.deepcopy(cls.params) if cls.params else {}
        params = self._merge_dict(params, user_params)
        if 'device' in params and isinstance(params['device'], dict):
            params['device']['select'] = device
        return params

    @staticmethod
    def _merge_dict(defaults: Optional[dict], overrides: Optional[dict]) -> dict:
        """Shallow-merge `overrides` on top of `defaults`, preserving nested
        dicts (selector / editor entries) by updating their inner keys
        instead of replacing the whole dict."""
        merged = copy.deepcopy(defaults) if defaults else {}
        if not overrides:
            return merged
        for k, v in overrides.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k].update(v)
            else:
                merged[k] = v
        return merged

    # ── Per-image run ──────────────────────────────────────────────────────

    def translate_image(self, img: np.ndarray, progress_cb=None):
        """Run detect → OCR → translate → (optional) inpaint on one image.

        Returns (blocks, inpainted_img):
          - blocks: List[TranslatedBlock] in the image's pixel space.
                    Empty list = no text detected (inpainted_img is then None).
          - inpainted_img: np.ndarray with the original text erased — same
                    shape as `img`. None if inpainting is disabled OR the
                    inpaint step failed.

        progress_cb: optional callable(str) invoked at each phase boundary
        (Detecting, OCR, Translating, Inpainting). The caller (PipelineWorker)
        plumbs this to the status toast.
        """
        if self._current is None:
            raise RuntimeError('apply_settings() must be called before translate_image()')

        def _emit(msg):
            if progress_cb:
                progress_cb(msg)

        _emit('Detecting text...')
        t0 = time.time()
        mask, blk_list = self._detector.detect(img)
        t_det = time.time() - t0

        # NOTE: reading-order sorting for the translation batch now lives in
        # BaseTranslator.translate_textblk_lst (shared with the main app), so
        # we no longer sort here. Each block keeps its detection order in the
        # list; only the internal translation request is reordered.

        if not blk_list:
            return [], None

        _emit(f'OCR ({len(blk_list)} blocks)...')
        t0 = time.time()
        self._ocr.run_ocr(img, blk_list)
        t_ocr = time.time() - t0

        _emit(f'Translating ({len(blk_list)} blocks)...')
        t0 = time.time()
        self._translator.translate_textblk_lst(blk_list)
        t_tr = time.time() - t0

        inpainted_img = None
        t_ip = 0.0
        if self._current.enable_inpaint and self._inpainter is not None and mask is not None:
            _emit(f'Inpainting ({len(blk_list)} blocks)...')
            t0 = time.time()
            try:
                # Dilate the text mask slightly so anti-aliased glyph edges
                # are fully covered — without this, the inpainter often
                # leaves faint outlines of the original characters behind.
                import cv2
                ksize = 3
                element = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),
                )
                inpaint_mask = cv2.dilate(mask, element)

                # InpainterBase.inpaint() ZEROES the mask it is given, block by
                # block, as it works (modules/inpaint/base.py:109 — the
                # inpaint_by_block path, which is the default). Anything that
                # needs to know what was masked has to snapshot it first, or it
                # reads back a mask with exactly the text areas erased.
                mask_px = int(np.count_nonzero(inpaint_mask >= 127))

                inpainted_img = self._inpainter.inpaint(img, inpaint_mask, blk_list)

                # How much of the page the inpainter actually rewrote. The
                # detector can produce a very broad mask on sketchy or
                # low-contrast art, and then the model repaints most of the
                # image — which reads as a washed-out, over-bright result
                # rather than as an obvious failure. Logging coverage makes
                # that visible instead of leaving it to guesswork.
                if inpainted_img is not None:
                    total = img.shape[0] * img.shape[1]
                    changed = int(np.count_nonzero(
                        np.any(inpainted_img != img, axis=2)))
                    print(f'[screen_translator] inpaint: mask covers '
                          f'{mask_px / total:.1%}, {changed / total:.1%} of '
                          f'pixels rewritten, mean brightness '
                          f'{img.mean():.1f} -> {inpainted_img.mean():.1f}')
            except Exception as e:
                # If inpaint fails for any reason, fall back to no-inpaint
                # mode: the overlay will just paint translation boxes on the
                # raw capture (the previous behavior). Don't kill the whole
                # translation.
                print(f'[screen_translator] inpaint failed ({type(e).__name__}: {e}); '
                      f'falling back to no-inpaint overlay')
                inpainted_img = None
            t_ip = time.time() - t0

        print(f'[screen_translator] detect={t_det:.2f}s  '
              f'ocr={t_ocr:.2f}s ({len(blk_list)} blocks)  '
              f'translate={t_tr:.2f}s  '
              f'inpaint={t_ip:.2f}s')

        out = []
        for blk in blk_list:
            x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
            src = blk.get_text() if hasattr(blk, 'get_text') else ''
            if isinstance(src, list):
                src = '\n'.join(src)
            tr = getattr(blk, 'translation', '') or ''
            angle = float(getattr(blk, 'angle', 0) or 0)

            balloon_xyxy = None
            if self._current.fill_balloon:
                # Use the ORIGINAL img (not inpainted) — the balloon outline
                # is what we trace, and inpaint doesn't touch it anyway.
                balloon_xyxy = self._compute_balloon_xyxy(img, (x1, y1, x2, y2))

            out.append(TranslatedBlock(
                xyxy=(x1, y1, x2, y2),
                angle=angle,
                source_text=src,
                translation=tr,
                balloon_xyxy=balloon_xyxy,
            ))
        return out, inpainted_img

    # Balloon search tuning. See _compute_balloon_xyxy.
    _BALLOON_PADS = (1.0, 2.0)       # crop margin, in multiples of the text's long side
    _BALLOON_TOL = 32                # grey levels a pixel may differ from the paper colour
    _BALLOON_MAX_AREA_RATIO = 12     # balloon/text bbox area, above which...
    _BALLOON_MAX_PAGE_SHARE = 0.10   # ...a region this much of the capture is a panel
    _BALLOON_HARD_AREA_RATIO = 40    # never a balloon, however small on the page
    _BALLOON_MIN_FILL = 0.4          # paper pixels / balloon bbox area

    @staticmethod
    def _compute_balloon_xyxy(img, text_xyxy):
        """Find the balloon or caption frame enclosing `text_xyxy` and return
        its bounding box in image coords, or None if there isn't a clean one.

        The balloon is taken to be the connected patch of "paper" the text sits
        on: every pixel within a tolerance of the background colour between the
        glyphs, reachable from there without crossing an outline. That makes no
        assumption about the balloon's shape or how big it is relative to the
        text, which is the point — it has to handle a wide caption frame around
        a narrow column of vertical text and a tall thin balloon around a single
        column equally.

        utils.imgproc_utils.extract_ballon_region is not used here on purpose.
        It only keeps contours covering at least 40% of the crop it is given, so
        it works when the crop is a little larger than the balloon and silently
        finds nothing, or the wrong thing, otherwise; with the balloon's size
        unknown there is no crop that satisfies it for both shapes above.

        A result is trusted only when its extent is known and plausible:
          - it must not run into an edge of the search crop (the patch continues
            beyond what was looked at; retried once with a bigger crop). An
            edge that is also the image edge is a real boundary and is allowed.
          - it must not reach three or more image edges — that is the paper
            colour flooding the whole capture, not a balloon.
          - it must contain the text's center and be at least the text's size.
          - it must not be a whole panel: far larger than the text AND a
            sizeable share of the capture. Size relative to the text alone is
            not enough to tell — a lone "!" in a big balloon is as lopsided as
            narration loose in a panel, but the balloon is small on the page.
          - it must be reasonably solid rather than a thin leak along gutters.
        Any miss returns None and the caller falls back to the text bbox, so a
        miss never produces a worse layout than no detection at all.
        """
        try:
            import cv2

            ih, iw = img.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in text_xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            tw, th = x2 - x1, y2 - y1
            if tw < 2 or th < 2:
                return None
            tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2

            def grey(a):
                return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

            # Paper colour: the most common grey level in a thin ring just
            # outside the text bbox. Text never touches its outline, so that
            # ring is almost all paper, whichever way round the polarity is.
            # (Sampling inside the bbox instead picks the INK colour for a
            # dense bold column, where ink is the majority of the box.)
            ring = max(2, min(tw, th) // 6)
            rx1, ry1 = max(0, x1 - ring), max(0, y1 - ring)
            rx2, ry2 = min(iw, x2 + ring), min(ih, y2 + ring)
            around = grey(img[ry1:ry2, rx1:rx2])
            outside = np.ones(around.shape, bool)
            outside[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
            samples = around[outside] if outside.any() else around.ravel()
            hist = np.bincount((samples >> 3).ravel(), minlength=32)
            paper_level = int(hist.argmax()) * 8 + 4

            cls = TranslationPipeline
            prev_crop = None
            for pad_mul in cls._BALLOON_PADS:
                pad = int(max(tw, th) * pad_mul)
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(iw, x2 + pad), min(ih, y2 + pad)
                if (cx1, cy1, cx2, cy2) == prev_crop:
                    break       # already bounded by the image; nothing new to see
                prev_crop = (cx1, cy1, cx2, cy2)
                crop_w, crop_h = cx2 - cx1, cy2 - cy1

                crop = grey(img[cy1:cy2, cx1:cx2]).astype(np.int16)
                paper = (np.abs(crop - paper_level) <= cls._BALLOON_TOL).astype(np.uint8)
                # 4-connectivity: a one-pixel diagonal outline still separates.
                n, labels, stats, _ = cv2.connectedComponentsWithStats(
                    paper, connectivity=4)
                if n < 2:
                    return None

                # The paper patch that owns most of the text bbox and its
                # ring. Counters enclosed by glyphs are separate, much smaller
                # components.
                inside = labels[ry1 - cy1:ry2 - cy1, rx1 - cx1:rx2 - cx1]
                owned = np.bincount(inside.ravel(), minlength=n)
                owned[0] = 0
                label = int(owned.argmax())
                if owned[label] == 0:
                    return None
                lx, ly, lw, lh, area = (int(v) for v in stats[label])

                at_l, at_t = lx == 0, ly == 0
                at_r, at_b = lx + lw == crop_w, ly + lh == crop_h
                cut_off = ((at_l and cx1 > 0) or (at_t and cy1 > 0)
                           or (at_r and cx2 < iw) or (at_b and cy2 < ih))
                if cut_off:
                    continue    # extent unknown at this crop size

                # Extent is real from here on, so a failed check is final.
                if at_l + at_t + at_r + at_b >= 3:
                    return None
                bx1, by1 = cx1 + lx, cy1 + ly
                bx2, by2 = bx1 + lw, by1 + lh
                if not (bx1 <= tcx <= bx2 and by1 <= tcy <= by2):
                    return None
                if lw * lh < tw * th:
                    return None
                area_ratio = (lw * lh) / (tw * th)
                if area_ratio > cls._BALLOON_HARD_AREA_RATIO:
                    return None
                if (area_ratio > cls._BALLOON_MAX_AREA_RATIO
                        and lw * lh > cls._BALLOON_MAX_PAGE_SHARE * iw * ih):
                    return None
                if area < cls._BALLOON_MIN_FILL * lw * lh:
                    return None
                return (bx1, by1, bx2, by2)

            return None
        except Exception as e:
            print(f'[screen_translator] balloon detect failed '
                  f'({type(e).__name__}: {e}); using text bbox')
            return None

    # ── Introspection for UI tooltips / About dialog ───────────────────────

    @property
    def current(self) -> Optional[SimpleNamespace]:
        return self._current
