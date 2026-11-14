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
import re
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
    # Size of the original glyphs in capture-image pixels, as the detector
    # measured it; 0 when unknown. The overlay keeps the translation from being
    # set much larger than this — a timestamp nine pixels high drawn at the
    # full font size lands on top of everything around it.
    font_px: float = 0.0


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
        # Chinese captures (see config.AUTO_DETECT_CHINESE): a fast OCR that
        # only tells the language, and the OCR that reads Chinese, loaded the
        # first time one turns up.
        self._probe_ocr = None
        self._zh_ocr = None

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

        # The language probe is loaded up front so the first capture does not
        # wait for it; the Chinese OCR only when a Chinese capture arrives.
        if not self._auto_chinese(settings):
            self._probe_ocr = self._zh_ocr = None
        else:
            stale = (prev is None or prev.device != settings.device
                     or getattr(prev, 'lang_probe_ocr', None) != settings.lang_probe_ocr
                     or getattr(prev, 'chinese_ocr_module', None) != settings.chinese_ocr_module)
            if stale:
                self._zh_ocr = None
            if stale or self._probe_ocr is None:
                _emit(f'Loading language probe: {settings.lang_probe_ocr}...')
                try:
                    self._probe_ocr = self._load_ocr(settings.lang_probe_ocr, settings)
                except Exception as e:
                    # Without it every capture is simply taken as the configured
                    # language, which is how things were before.
                    self._probe_ocr = None
                    print(f'[screen_translator] language probe unavailable '
                          f'({type(e).__name__}: {e}); captures are read as '
                          f'{settings.source_lang}')

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

    # ── Chinese captures ───────────────────────────────────────────────────

    @staticmethod
    def _auto_chinese(settings) -> bool:
        """Whether captures are checked for Chinese: only when asked for, and
        only when the configured source is Japanese — any other source language
        was chosen on purpose and is left alone."""
        return bool(getattr(settings, 'auto_detect_chinese', False)
                    and settings.source_lang == '日本語'
                    and getattr(settings, 'lang_probe_ocr', None)
                    and getattr(settings, 'chinese_ocr_module', None))

    def _load_ocr(self, name, settings):
        from modules import OCR
        cls = OCR.module_dict[name]
        ocr = cls(**self._build_module_params(
            cls, settings.ocr_params.get(name), settings.device))
        ocr.load_model()
        return ocr

    @staticmethod
    def _looks_chinese(texts):
        """(is_chinese, kana, hanzi) for what the probe read off a capture.

        Japanese cannot be written without kana: particles, verb endings and
        loanwords are all kana, and a page of it reads as more kana than kanji.
        Chinese has none. So: a fair number of Chinese characters and next to
        no kana. "Next to no" rather than none because the probe is a rough
        reader and will now and then see a kana in a Chinese character; a few
        characters at least, because two kanji on their own (a sign, a label)
        say nothing either way and are left as the configured language.
        """
        joined = ''.join(texts)
        kana = sum(1 for c in joined if '぀' <= c <= 'ヿ' and c not in '・ー')
        hanzi = sum(1 for c in joined if '一' <= c <= '鿿')
        return (hanzi >= 4 and kana <= 0.05 * hanzi), kana, hanzi

    def _pick_reader(self, img, blk_list, emit):
        """The OCR and source language for this capture: (ocr, source_lang).

        Runs the probe over the detected blocks, and switches to the Chinese
        OCR when what it read is Chinese. The probe's reading is thrown away —
        whichever OCR is picked reads every block again.
        """
        cur = self._current
        if self._probe_ocr is None or not self._auto_chinese(cur):
            return self._ocr, cur.source_lang
        t0 = time.time()
        try:
            self._probe_ocr.run_ocr(img, blk_list)
            chinese, kana, hanzi = self._looks_chinese(
                [blk.get_text() for blk in blk_list])
        except Exception as e:
            print(f'[screen_translator] language probe failed '
                  f'({type(e).__name__}: {e}); reading as {cur.source_lang}')
            return self._ocr, cur.source_lang
        lang = cur.chinese_source_lang if chinese else cur.source_lang
        print(f'[screen_translator] language: {lang} '
              f'(probe read {kana} kana, {hanzi} hanzi, {time.time() - t0:.2f}s)')
        if not chinese:
            return self._ocr, cur.source_lang
        if self._zh_ocr is None:
            emit(f'Loading OCR: {cur.chinese_ocr_module}...')
            try:
                self._zh_ocr = self._load_ocr(cur.chinese_ocr_module, cur)
            except Exception as e:
                print(f'[screen_translator] Chinese OCR unavailable '
                      f'({type(e).__name__}: {e}); reading as {cur.source_lang}')
                return self._ocr, cur.source_lang
        return self._zh_ocr, cur.chinese_source_lang

    @classmethod
    def _keep_list_lines(cls, blk_list):
        """For an OCR that reads a block line by line: keep the lines of a list
        apart. Joined end to end, as a paragraph's lines rightly are, the items
        of a list run together and cannot be told apart again."""
        for blk in blk_list:
            lines = blk.text if isinstance(blk.text, list) else None
            if not lines or getattr(blk, 'vertical', False):
                continue
            lines = [t.strip() for t in lines if t and t.strip()]
            if len(lines) >= 2 and any(cls._is_list_item(t) for t in lines):
                blk.text = '\n'.join(lines)

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

        self._clamp_blocks_to_image(img, blk_list)

        _emit(f'OCR ({len(blk_list)} blocks)...')
        t0 = time.time()
        ocr, source_lang = self._pick_reader(img, blk_list, _emit)
        ocr.run_ocr(img, blk_list)
        if ocr is not self._ocr:
            # The probe is a rough reader. The Chinese OCR is a careful one
            # that also knows kana, so its reading settles it: kana in what it
            # read means the capture was Japanese after all, and Japanese is
            # read by the Japanese OCR.
            chinese, kana, hanzi = self._looks_chinese(
                [blk.get_text() for blk in blk_list])
            if not chinese:
                print(f'[screen_translator] language: {self._current.source_lang} '
                      f'after all (read {kana} kana, {hanzi} hanzi)')
                ocr, source_lang = self._ocr, self._current.source_lang
                ocr.run_ocr(img, blk_list)
        if ocr is self._ocr:
            try:
                self._reread_lists(img, blk_list, mask)
            except Exception as e:
                # Purely an improvement over the block-level read; never let it
                # take the capture down.
                print(f'[screen_translator] list re-read skipped '
                      f'({type(e).__name__}: {e})')
        else:
            # The Chinese OCR reads line by line already; the row-by-row
            # re-read exists for one that reads a block as a single picture.
            self._keep_list_lines(blk_list)
        t_ocr = time.time() - t0

        _emit(f'Translating ({len(blk_list)} blocks)...')
        t0 = time.time()
        # The translator is built for the configured source language; for a
        # capture in another one it is told so for this call only.
        configured_lang = self._translator.lang_source
        self._translator.lang_source = source_lang
        try:
            self._translator.translate_textblk_lst(blk_list)
        finally:
            self._translator.lang_source = configured_lang
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

                # Erase only what there is a translation to put back. A block
                # that came through with none (nothing read, or the translator
                # dropped it) would otherwise simply vanish from the page.
                to_erase, spared = self._spare_untranslated(inpaint_mask, blk_list)
                if spared:
                    print(f'[screen_translator] {spared} block(s) have no '
                          f'translation; their text is left as it is')

                # InpainterBase.inpaint() ZEROES the mask it is given, block by
                # block, as it works (modules/inpaint/base.py:109 — the
                # inpaint_by_block path, which is the default). Anything that
                # needs to know what was masked has to snapshot it first, or it
                # reads back a mask with exactly the text areas erased.
                mask_px = int(np.count_nonzero(inpaint_mask >= 127))

                inpainted_img = self._inpainter.inpaint(img, inpaint_mask, to_erase)

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

        boxes = [tuple(int(v) for v in blk.xyxy) for blk in blk_list]
        if self._current.fill_balloon:
            # Use the ORIGINAL img (not inpainted) — the balloon outline is
            # what we trace, and inpaint doesn't touch it anyway.
            reasons = []
            balloons = self._assign_balloons(img, boxes, reasons)
            # One line per block: which way it will be laid out, and why. A
            # block with no balloon is drawn at its own text box, which is the
            # layout that goes wrong when the decision was mistaken.
            for box, balloon, reason in zip(boxes, balloons, reasons):
                print(f'[screen_translator] block {box}: '
                      + (f'balloon {balloon}' if balloon is not None
                         else f'no balloon ({reason})'))
        else:
            balloons = [None] * len(boxes)

        out = []
        for blk, (x1, y1, x2, y2), balloon_xyxy in zip(blk_list, boxes, balloons):
            src = blk.get_text() if hasattr(blk, 'get_text') else ''
            if isinstance(src, list):
                src = '\n'.join(src)
            tr = getattr(blk, 'translation', '') or ''
            angle = float(getattr(blk, 'angle', 0) or 0)

            out.append(TranslatedBlock(
                xyxy=(x1, y1, x2, y2),
                angle=angle,
                source_text=src,
                translation=tr,
                balloon_xyxy=balloon_xyxy,
                font_px=max(0.0, float(getattr(blk, 'font_size', 0) or 0)),
            ))

        if self._current.fill_balloon:
            from modules.translators.base import RTL_SOURCE_LANGS
            out = self._merge_blocks_sharing_balloon(
                out, rtl=source_lang in RTL_SOURCE_LANGS)
        return out, inpainted_img

    @staticmethod
    def _clamp_blocks_to_image(img, blk_list):
        """Pull every block's box inside the image, a pixel clear of its edges.

        The detector returns boxes that overhang the capture when text runs up
        to its edge — x1 of -8 for a list set against the left side. The OCR
        modules skip any box that is not strictly inside the image
        (`x1 > 0 and y1 > 0 and x2 < w and y2 < h`, modules/ocr/__init__.py),
        log 'invalid textbbox' and leave the text empty. Nothing is translated
        for that block, the inpainter still erases it, and the text is simply
        gone from the result. Hence one pixel in from the edge, not onto it.
        """
        ih, iw = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
            cx1, cy1 = max(1, x1), max(1, y1)
            cx2, cy2 = min(iw - 1, x2), min(ih - 1, y2)
            if cx1 >= cx2 or cy1 >= cy2:
                continue        # nothing of it is on the image; leave it be
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                blk.xyxy = [cx1, cy1, cx2, cy2]

    @staticmethod
    def _spare_untranslated(mask, blk_list, pad=5):
        """Take the blocks that have no translation out of the inpaint mask.

        Returns (blocks to erase, number spared). `mask` is edited in place.
        Erasing text is only right when there is something to draw in its
        place; with no translation the original is the better thing to show.
        Where a spared block's box overlaps a translated one the translated
        block wins, so its text is still fully erased.
        """
        def translated(blk):
            return bool((getattr(blk, 'translation', '') or '').strip())

        keep = [blk for blk in blk_list if translated(blk)]
        lost = [blk for blk in blk_list if not translated(blk)]
        if not lost:
            return keep, 0

        ih, iw = mask.shape[:2]

        def area(blk, grow):
            x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
            return (slice(max(0, y1 - grow), min(ih, y2 + grow)),
                    slice(max(0, x1 - grow), min(iw, x2 + grow)))

        spare = np.zeros((ih, iw), bool)
        for blk in lost:
            spare[area(blk, pad)] = True
        for blk in keep:
            spare[area(blk, 0)] = False
        mask[spare] = 0
        return keep, len(lost)

    @staticmethod
    def _iou(a, b):
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        if ix <= 0 or iy <= 0:
            return 0.0
        inter = ix * iy
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / union if union > 0 else 0.0

    @classmethod
    def _text_is_centered(cls, balloon, text_box):
        """Whether `text_box` sits in the middle of `balloon` on both axes."""
        lo, hi = cls._BALLOON_CENTER_BAND
        bw, bh = balloon[2] - balloon[0], balloon[3] - balloon[1]
        if bw <= 0 or bh <= 0:
            return False
        rx = ((text_box[0] + text_box[2]) / 2 - balloon[0]) / bw
        ry = ((text_box[1] + text_box[3]) / 2 - balloon[1]) / bh
        return lo <= rx <= hi and lo <= ry <= hi

    @classmethod
    def _size_reason(cls, balloon, text_box, iw, ih):
        """Why `balloon` is too big to be the balloon of `text_box`, or None.

        Far larger than its text AND a sizeable share of the capture is a
        panel, not a balloon. Size relative to the text alone is not enough to
        tell — a lone "!" in a big balloon is as lopsided as narration loose in
        a panel — but the balloon is small on the page.
        """
        area = (balloon[2] - balloon[0]) * (balloon[3] - balloon[1])
        text = max(1, (text_box[2] - text_box[0]) * (text_box[3] - text_box[1]))
        ratio, share = area / text, area / (iw * ih)
        if ratio > cls._BALLOON_HARD_AREA_RATIO:
            return f'patch is {ratio:.0f}x the text'
        if ratio > cls._BALLOON_MAX_AREA_RATIO and share > cls._BALLOON_MAX_PAGE_SHARE:
            return (f'patch is {ratio:.0f}x the text and {share:.0%} of the '
                    f'capture (a panel)')
        return None

    @classmethod
    def _assign_balloons(cls, img, boxes, reasons=None):
        """The balloon for each text box, or None — one entry per box. A list
        passed as `reasons` is filled with, per box, why it got no balloon
        (None where it got one).

        Filling a balloon re-centres the text in it. That is only right when
        the original text was centred there to begin with; otherwise the patch
        of paper under the text is not a balloon drawn around it, it is a
        background that happens to have text on it, and "filling" it moves the
        translation away from where the original was — into the middle of the
        artwork, in the cases that prompted this.

        Centring is judged per balloon, not per block: two paragraphs side by
        side in one balloon are each off-centre and centred as a pair. So
        patches are found first, boxes sharing one are grouped, and the group's
        combined text has to sit in the middle of it.
        """
        found, masks, why_not = [], [], []
        for box in boxes:
            info, why = {}, []
            found.append(cls._compute_balloon_xyxy(
                img, box, check_center=False, check_fill=False,
                check_size=False, all_text=boxes, out=info, why=why))
            masks.append(info.get('mask'))
            why_not.append(why[0] if why else None)

        groups = []          # each: indices of boxes sharing one patch
        for i, balloon in enumerate(found):
            if balloon is None:
                continue
            home = next((g for g in groups
                         if cls._iou(found[g[0]], balloon) >= 0.8), None)
            if home is not None:
                home.append(i)
            else:
                groups.append([i])

        def union(idx):
            return (min(boxes[i][0] for i in idx), min(boxes[i][1] for i in idx),
                    max(boxes[i][2] for i in idx), max(boxes[i][3] for i in idx))

        ih, iw = img.shape[:2]

        def verdict(rect, idx):
            """None if `rect` can be the balloon of blocks `idx`, else why not."""
            text = union(idx)
            if not cls._text_is_centered(rect, text):
                return 'text is not in the middle of the patch'
            return cls._size_reason(rect, text, iw, ih)

        out = [None] * len(boxes)
        for group in groups:
            balloon, mask = found[group[0]], masks[group[0]]

            # As one balloon first. That is the simplest reading and the one
            # that has been giving good results: text split across two blocks,
            # or across two balloons drawn well into each other, is joined and
            # set once in the middle.
            fill = float(mask.mean()) if mask is not None else 1.0
            if fill >= cls._BALLOON_MIN_FILL:
                problem = verdict(balloon, group)
                for i in group:
                    if problem is None:
                        out[i] = balloon
                    else:
                        why_not[i] = problem
                continue

            # Too sparse to be one balloon. With text in more than one place
            # it may be two balloons joined at an angle — little of the box
            # around the pair is paper, though each is solid enough on its own.
            reason = f'patch fills only {fill:.0%} of its box'
            lobes = (cls._split_into_lobes(balloon, mask, [boxes[i] for i in group])
                     if len(group) > 1 else [])
            for i in group:
                why_not[i] = reason
            if len(lobes) < 2:
                continue
            for members, rect, lobe_fill in lobes:
                idx = [group[k] for k in members]
                if lobe_fill < cls._BALLOON_MIN_FILL:
                    continue
                problem = verdict(rect, idx)
                for i in idx:
                    if problem is None:
                        out[i] = rect
                    else:
                        why_not[i] = problem + ' (its part of a joined pair)'
        if reasons is not None:
            reasons[:] = [None if out[i] is not None else why_not[i]
                          for i in range(len(boxes))]
        return out

    @staticmethod
    def _split_into_lobes(balloon, mask, text_boxes):
        """Divide one patch among the text on it:
        [(member indices, rect, share of rect that is paper), ...].

        Two balloons drawn overlapping are a single patch of paper, but they are
        still two balloons, each with its own text. Laid out as one, the two
        utterances would be run together in the middle of the pair.

        Text boxes belong to the same lobe when what lies between them is
        paper: the part of their joint convex hull outside the boxes themselves
        is mostly patch. That holds for paragraphs side by side in one balloon
        and fails across the notch where two balloons meet. Each lobe then
        takes the part of the patch nearest its own text. A patch with a single
        lobe comes back whole, as before.

        It has to be actual paper, measured between the boxes only. Filling in
        the patch's outline to get rid of the glyph holes also fills in any
        artwork the balloons wrap around — hair across a balloon, say — and
        then everything looks connected.
        """
        import cv2

        n = len(text_boxes)
        if mask is None:
            return [(list(range(n)), balloon, 1.0)]
        whole = [(list(range(n)), balloon, float(mask.mean()))]
        if n < 2:
            return whole
        bx1, by1 = balloon[0], balloon[1]
        h, w = mask.shape

        paper = mask.astype(np.uint8)

        def local(box):
            x1, y1, x2, y2 = box
            return (max(0, x1 - bx1), max(0, y1 - by1),
                    min(w, x2 - bx1), min(h, y2 - by1))

        def corners(box):
            x1, y1, x2, y2 = box
            return [(x1 - bx1, y1 - by1), (x2 - bx1, y1 - by1),
                    (x2 - bx1, y2 - by1), (x1 - bx1, y2 - by1)]

        parent = list(range(n))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                pts = np.array(corners(text_boxes[i]) + corners(text_boxes[j]), np.int32)
                hull = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
                hull_area = int(hull.sum())
                between = hull.copy()
                for k in (i, j):
                    ax1, ay1, ax2, ay2 = local(text_boxes[k])
                    between[ay1:ay2, ax1:ax2] = 0
                gap = int(between.sum())
                # Boxes that all but touch have nothing between them to judge.
                if gap < 0.05 * hull_area or int((between & paper).sum()) >= 0.8 * gap:
                    parent[root(i)] = root(j)

        clusters = {}
        for i in range(n):
            clusters.setdefault(root(i), []).append(i)
        clusters = list(clusters.values())
        if len(clusters) < 2:
            return whole

        # Each lobe: the patch pixels nearer its text than any other lobe's.
        ys, xs = np.mgrid[0:h, 0:w]
        xs = xs + bx1
        ys = ys + by1
        dist = []
        for members in clusters:
            best = np.full((h, w), np.inf)
            for k in members:
                x1, y1, x2, y2 = text_boxes[k]
                dx = np.maximum(np.maximum(x1 - xs, xs - x2), 0)
                dy = np.maximum(np.maximum(y1 - ys, ys - y2), 0)
                best = np.minimum(best, np.hypot(dx, dy))
            dist.append(best)
        nearest = np.argmin(np.stack(dist), axis=0)

        lobes = []
        for c, members in enumerate(clusters):
            cell = (nearest == c) & mask
            rows = np.flatnonzero(cell.any(axis=1))
            cols = np.flatnonzero(cell.any(axis=0))
            if rows.size == 0 or cols.size == 0:
                continue
            r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
            # Its own text counts as filled: the ink is not paper, but it is
            # not a gap in the balloon either.
            filled = cell.copy()
            for k in members:
                ax1, ay1, ax2, ay2 = local(text_boxes[k])
                filled[ay1:ay2, ax1:ax2] = True
            lobes.append((members, (bx1 + c0, by1 + r0, bx1 + c1, by1 + r1),
                          float(filled[r0:r1, c0:c1].mean())))
        return lobes

    @staticmethod
    def _merge_blocks_sharing_balloon(blocks, rtl):
        """Fold blocks that landed in the same balloon into one.

        The detector often splits one balloon's text into several blocks — two
        paragraphs side by side, a line set apart from the rest. Each of them
        is then handed the same balloon, and the overlay fills that balloon
        with each translation separately, centered: every one drawn straight
        over the others. A balloon is one flow of text, so its blocks are
        joined in reading order and laid out once.

        Balloons count as the same above 0.8 IoU rather than only when
        identical, since two blocks can sample slightly different paper levels
        and come back a pixel or two apart. Blocks without a balloon are left
        alone; order of the list is otherwise preserved.
        """
        from modules.translators.base import reading_order_indices

        def iou(a, b):
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix <= 0 or iy <= 0:
                return 0.0
            inter = ix * iy
            union = ((a[2] - a[0]) * (a[3] - a[1])
                     + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            return inter / union if union > 0 else 0.0

        groups = []
        for blk in blocks:
            if blk.balloon_xyxy is not None:
                home = next((g for g in groups
                             if g[0].balloon_xyxy is not None
                             and iou(g[0].balloon_xyxy, blk.balloon_xyxy) >= 0.8),
                            None)
                if home is not None:
                    home.append(blk)
                    continue
            groups.append([blk])

        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
            members = [group[k] for k in reading_order_indices(group, rtl)]
            merged.append(TranslatedBlock(
                xyxy=(min(m.xyxy[0] for m in members), min(m.xyxy[1] for m in members),
                      max(m.xyxy[2] for m in members), max(m.xyxy[3] for m in members)),
                angle=members[0].angle,
                source_text='\n'.join(m.source_text for m in members if m.source_text),
                translation='\n'.join(m.translation for m in members
                                      if m.translation.strip()),
                balloon_xyxy=(min(m.balloon_xyxy[0] for m in members),
                              min(m.balloon_xyxy[1] for m in members),
                              max(m.balloon_xyxy[2] for m in members),
                              max(m.balloon_xyxy[3] for m in members)),
                font_px=max(m.font_px for m in members),
            ))
            print(f'[screen_translator] merged {len(members)} blocks sharing '
                  f'one balloon')
        return merged

    # Characters that open a list item. A horizontal block with a row starting
    # on one of these is a list: its rows are content, not line wrapping.
    _LIST_MARKERS = '・･•●○◆◇■□▶▷※＊*-‐–'
    _RE_NUMBERED_ITEM = re.compile(r'^(?:[①-⑳]|[(（]?\d{1,2}[.)）])')

    @classmethod
    def _is_list_item(cls, text):
        text = text.lstrip()
        return bool(text) and (text[0] in cls._LIST_MARKERS
                               or cls._RE_NUMBERED_ITEM.match(text) is not None)

    @staticmethod
    def _split_rows(img, box, text_mask=None):
        """Split a horizontal text block into its rows by where the ink is.

        Returns one (x1, y1, x2, y2) per row, top to bottom, or None when the
        block does not separate cleanly into two or more rows. See _find_rows,
        which also says what the rows were found standing on.
        """
        return TranslationPipeline._find_rows(img, box, text_mask)[0]

    @staticmethod
    def _find_rows(img, box, text_mask=None):
        """Rows of a horizontal text block, and the grey level of the ground
        they stand on: (rows, paper), or (None, None).

        The detector's own line polygons are not usable for this: on a profile
        card they came back as one polygon covering most of the block plus a few
        partial ones, with a whole row missing. A projection of the ink onto the
        vertical axis is crude but dependable for what it is needed for — text
        set in rows on a plain ground.

        Which ground: the level just outside the block is tried first (a list
        on a flat page), then the commonest level inside it (a caption laid
        across a picture on its own strip of white, where outside the block is
        the picture). Either has to make up a fair part of the block to count
        as its ground at all. Looking inside first would be wrong the other
        way: in a tight box round heavy type the commonest level is the ink.

        `text_mask` is the detector's text mask for the whole image. Given, ink
        only counts where the detector also saw text, which keeps the edge of a
        white strip and the picture showing between two strips out of it.

        Text drawn straight onto artwork has no ground to speak of; its rows
        come out irregular, which is caught and refused.
        """
        cls = TranslationPipeline
        ih, iw = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None, None

        outer, _ = cls._paper_level(img, (x1, y1, x2, y2))
        inner = cls._text_paper_level(img, (x1, y1, x2, y2))
        levels = [outer]
        if abs(inner - outer) > cls._BALLOON_TOL:
            levels.append(inner)

        inside = cls._grey(img[y1:y2, x1:x2]).astype(np.int16)
        # A little wider than the block, so a glyph (or a bullet) hard against
        # the block's side still has paper on both sides of it.
        gx1, gx2 = max(0, x1 - 4), min(iw, x2 + 4)
        crop = cls._grey(img[y1:y2, gx1:gx2]).astype(np.int16)
        seen = None if text_mask is None else text_mask[y1:y2, gx1:gx2] >= 127

        for paper in levels:
            if (np.abs(inside - paper) <= cls._BALLOON_TOL).mean() < 1 / 3:
                continue
            ink = cls._ink_on_paper(np.abs(crop - paper) <= cls._BALLOON_TOL)
            if seen is not None:
                ink &= seen
            rows = cls._rows_from_ink(ink)
            if rows:
                return ([(max(0, gx1 + a), max(0, y1 + b), min(iw, gx1 + c), min(ih, y1 + d))
                         for a, b, c, d in rows], paper)
        return None, None

    @staticmethod
    def _rows_from_ink(ink):
        """Row boxes (x1, y1, x2, y2), in the mask's own coordinates and padded
        a little, from a boolean ink mask; None when it is not rows of text."""
        h = ink.shape[0]
        # A row of text against the gap between two rows, judged relative to
        # how much ink a text row carries here. The gap is not reliably empty:
        # with tight leading a bracket or a dakuten reaches into it, and a
        # capture scaled by the display blurs the edges across it. Asking for
        # zero ink welds the whole block into one row.
        profile = ink.sum(axis=1)
        busy = float(np.percentile(profile, 90))
        if busy < 2:
            return None
        has_ink = profile >= max(2.0, 0.15 * busy)

        runs, start = [], None
        for y, flag in enumerate(has_ink):
            if flag and start is None:
                start = y
            elif not flag and start is not None:
                runs.append([start, y])
                start = None
        if start is not None:
            runs.append([start, h])
        if len(runs) < 2:
            return None

        heights = sorted(e - s for s, e in runs)
        typical = heights[len(heights) // 2]

        # A piece of a row belongs to its neighbour: a short run next to a full
        # one (dakuten that cleared the line, a ruby line), or two parts of one
        # row with a sliver of a gap between them. Two FULL rows stay apart
        # however thin the gap — with tight leading one or two pixels is all
        # there is between them.
        rows = [runs[0]]
        for s, e in runs[1:]:
            gap = s - rows[-1][1]
            least = min(e - s, rows[-1][1] - rows[-1][0])
            if ((least < 0.4 * typical and gap <= 0.5 * typical)
                    or (least < 0.7 * typical and gap <= max(1, 0.15 * typical))):
                rows[-1][1] = e
            else:
                rows.append([s, e])
        # Whatever is still much shorter than a row is a stray mark, not text.
        rows = [r for r in rows if r[1] - r[0] >= 0.35 * typical]
        if len(rows) < 2:
            return None
        if max(e - s for s, e in rows) > 3 * typical:
            return None         # not rows of text

        pad = 3
        boxes = []
        for s, e in rows:
            cols = np.flatnonzero(ink[s:e].any(axis=0))
            if cols.size == 0:
                continue
            boxes.append((int(cols[0]) - pad, s - pad, int(cols[-1]) + 1 + pad, e + pad))
        return boxes if len(boxes) >= 2 else None

    def _reread_lists(self, img, blk_list, text_mask=None):
        """Re-read horizontal list blocks one row at a time, keeping the rows.

        The OCR reads a block as a single image and returns a single string. For
        a balloon that is right — its lines are just wrapping. For a list (a
        profile card, a spec sheet) the rows ARE the structure: read as one
        image they come back run together with no breaks, so the translation is
        one paragraph and nothing downstream can put the items back on their own
        lines. A six-row block is also well outside what the model reads
        reliably in one go.

        So each horizontal block is split into rows and the rows read
        separately; if any row opens with a list marker, the block's text
        becomes those rows joined by newlines. The translator passes the
        newlines through and the overlay draws them. A block that is not a list
        keeps its original reading untouched.
        """
        read_row = getattr(self._ocr, 'ocr_img', None)
        if read_row is None:
            return
        lists = []          # (block, rows) for every block rewritten as a list
        for blk in blk_list:
            if getattr(blk, 'vertical', False):
                continue
            rows, paper = self._find_rows(img, blk.xyxy, text_mask)
            if not rows or len(rows) > 30:
                continue
            right_edge = max(r[2] for r in rows)

            texts = []
            prev = None         # (right edge, height) of the last row read
            for rx1, ry1, rx2, ry2 in rows:
                # A bullet is taken off by shape and the rest read without it.
                # Left in, the OCR tends to fuse it into the first character
                # (・シ read as ジ), which loses both the marker and the word.
                cut = self._leading_bullet_width(
                    img[ry1:ry2, rx1:rx2], paper,
                    None if text_mask is None else text_mask[ry1:ry2, rx1:rx2])
                t = read_row(self._pad_row(img[ry1:ry2, rx1 + cut:rx2], paper))
                t = (t if isinstance(t, str) else '').strip()
                if not t:
                    continue
                if cut and not self._is_list_item(t):
                    t = '・' + t
                # An item too long for one row wraps onto the next, which then
                # has no bullet. It is the same sentence: kept as a row of its
                # own it is translated as two halves. A wrapped row is one that
                # ran to the block's right side — to within a couple of
                # characters, since a line may break early to keep a word whole.
                wrapped = (prev is not None and texts
                           and self._is_list_item(texts[-1])
                           and not self._is_list_item(t)
                           and prev[0] >= right_edge - 2 * prev[1])
                if wrapped:
                    texts[-1] += t
                else:
                    texts.append(t)
                prev = (rx2, ry2 - ry1)
            if len(texts) < 2 or not any(self._is_list_item(t) for t in texts):
                continue
            blk.text = '\n'.join(texts)
            lists.append((blk, len(rows)))
            print(f'[screen_translator] list block re-read as {len(texts)} items '
                  f'({len(rows)} rows)')

        for lst, n_rows in lists:
            self._absorb_list_title(blk_list, lst, n_rows)

    @staticmethod
    def _absorb_list_title(blk_list, lst, n_rows):
        """Fold a one-row block sitting directly on top of a list into the list
        as its first row.

        The detector sometimes hands back a list's heading (the name on a
        profile card) as a block of its own. Left separate, its translation is
        laid out in its own box, wraps when it is longer than the original, and
        runs down over the list's first row. It is the list's title; as the
        first row of the same block it is set in the same size and cannot
        collide with what it heads.
        """
        lx1, ly1, lx2, ly2 = [int(v) for v in lst.xyxy]
        row_h = (ly2 - ly1) / max(1, n_rows)
        for other in blk_list:
            if other is lst or getattr(other, 'vertical', False):
                continue
            ox1, oy1, ox2, oy2 = [int(v) for v in other.xyxy]
            one_row = (oy2 - oy1) <= 1.6 * row_h
            sits_on_top = ly1 - row_h <= oy2 <= ly1 + 0.5 * row_h and oy1 < ly1
            overlap = min(lx2, ox2) - max(lx1, ox1)
            lined_up = overlap >= 0.5 * (ox2 - ox1)
            title = other.get_text().strip()
            if not (one_row and sits_on_top and lined_up and title):
                continue
            lst.text = title + '\n' + lst.get_text()
            lst.xyxy = [min(lx1, ox1), min(ly1, oy1), max(lx2, ox2), max(ly2, oy2)]
            # By identity, not list.remove(): TextBlock's == compares fields and
            # raises on some of them, and remove() compares with every block it
            # passes on the way to the one it wants.
            blk_list[:] = [b for b in blk_list if b is not other]
            print('[screen_translator] list title folded into the list')
            return

    # Widest a row may be, relative to its height, when handed to the OCR.
    _ROW_MAX_ASPECT = 4.0

    @staticmethod
    def _pad_row(row_img, paper):
        """Pad a wide row above and below with paper colour before OCR.

        manga-ocr squashes its input into a square, so a single long row
        arrives with its characters crushed sideways. Padding towards a squarer
        shape eases that, up to a point: measured on ten rows of a profile card
        at two capture scales, mean similarity to the true text was 0.90 as
        cropped, 0.96 padded to 4:1, 0.80 at 2:1 and 0.50 at 1:1 — past 4:1 the
        text becomes too small a part of the picture.
        """
        import cv2
        h, w = row_img.shape[:2]
        want_h = int(round(w / TranslationPipeline._ROW_MAX_ASPECT))
        if want_h <= h:
            return row_img
        top = (want_h - h) // 2
        fill = paper if row_img.ndim == 2 else (paper,) * row_img.shape[2]
        return cv2.copyMakeBorder(row_img, top, want_h - h - top, 0, 0,
                                  cv2.BORDER_CONSTANT, value=fill)

    @staticmethod
    def _leading_bullet_width(row_img, paper, seen=None):
        """Width, in pixels from the left of a row crop, taken up by a leading
        bullet and the gap after it; 0 when the row does not start with one.

        A bullet is recognised by shape: the first blob of ink is small in both
        directions next to the row (a dot or a dash, not a character), stands
        clear of what follows, and something does follow.
        """
        cls = TranslationPipeline
        ink = cls._ink_on_paper(
            np.abs(cls._grey(row_img).astype(np.int16) - paper) <= cls._BALLOON_TOL)
        if seen is not None and seen.shape == ink.shape:
            # Only where the detector saw text, give or take a few pixels
            # sideways — a row cut from a strip of white laid over a picture
            # has the picture along its top and bottom, and that is not ink.
            import cv2
            near = cv2.dilate((seen >= 127).astype(np.uint8), np.ones((1, 9), np.uint8))
            ink &= near > 0
        # A speck is not a mark. One stray pixel under a bullet (the corner of
        # the strip the row is set on) is enough to make the bullet read as
        # tall as a character.
        import cv2
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] <= 2:
                ink[labels == i] = False
        # The row's height is that of its text, not of every line with a mark
        # on it: the padding above and below the row picks up the odd run of
        # pixels from whatever the row is set on.
        profile = ink.sum(axis=1)
        rows_with_ink = np.flatnonzero(
            profile >= max(1.0, 0.15 * float(np.percentile(profile, 90))))
        if rows_with_ink.size == 0:
            return 0
        row_h = int(rows_with_ink[-1] - rows_with_ink[0] + 1)
        cols = ink.any(axis=0)

        x, n = 0, cols.size
        while x < n and not cols[x]:
            x += 1
        start = x
        while x < n and cols[x]:
            x += 1
        end = x                         # first blob is [start, end)
        while x < n and not cols[x]:
            x += 1
        nxt = x                         # the next blob starts here
        if nxt >= n:
            return 0                    # nothing after it: not a bullet

        blob_rows = np.flatnonzero(ink[:, start:end].any(axis=1))
        blob_h = int(blob_rows[-1] - blob_rows[0] + 1)
        is_bullet = (end - start <= 0.6 * row_h
                     and blob_h <= 0.55 * row_h
                     and nxt - end >= 0.1 * row_h)
        return nxt if is_bullet else 0

    # Balloon search tuning. See _compute_balloon_xyxy.
    _BALLOON_PADS = (1.0, 2.0, 4.0)  # crop margin, in multiples of the text's long side
    _BALLOON_TOL = 32                # grey levels a pixel may differ from the paper colour
    _BALLOON_MAX_AREA_RATIO = 12     # balloon/text bbox area, above which...
    _BALLOON_MAX_PAGE_SHARE = 0.10   # ...a region this much of the capture is a panel
    _BALLOON_HARD_AREA_RATIO = 40    # never a balloon, however small on the page
    _BALLOON_ABS_PAGE_SHARE = 0.5    # over this much of the capture: the page itself
    _BALLOON_CENTER_BAND = (0.3, 0.7)  # where the text's center must sit in the balloon
    _BALLOON_MAX_ENCLOSED = 0.04     # largest foreign object inside, / balloon bbox area
    _BALLOON_MIN_FILL = 0.4          # paper pixels / balloon bbox area

    @staticmethod
    def _grey(a):
        import cv2
        return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _text_paper_level(img, box):
        """Grey level of the ground a block of horizontal rows is set on: the
        most common level INSIDE the box.

        _paper_level looks just outside the box, which is right for a balloon
        and wrong for a caption laid across a picture on its own strip of
        white — outside the box is the picture. Inside, rows of text with
        leading between them are mostly ground whatever surrounds them. (The
        dense bold column that makes _paper_level look outside is vertical
        text, which is never split into rows.)
        """
        x1, y1, x2, y2 = box
        grey = TranslationPipeline._grey(img[y1:y2, x1:x2])
        hist = np.bincount((grey >> 3).ravel(), minlength=32)
        return int(hist.argmax()) * 8 + 4

    @staticmethod
    def _ink_on_paper(is_paper):
        """Pixels that are not paper but have paper on both sides of them in
        their row — marks standing on the paper, as opposed to whatever lies
        beyond the paper's edge."""
        w = is_paper.shape[1]
        has = is_paper.any(axis=1)
        first = is_paper.argmax(axis=1)
        last = w - 1 - is_paper[:, ::-1].argmax(axis=1)
        xs = np.arange(w)[None, :]
        within = (xs >= first[:, None]) & (xs <= last[:, None]) & has[:, None]
        return ~is_paper & within

    @staticmethod
    def _paper_level(img, box):
        """Grey level of the paper a text box sits on, and the box grown by the
        ring it was sampled from: (level, (rx1, ry1, rx2, ry2)).

        The most common level in a thin ring just outside the box. Text never
        touches its outline, so that ring is almost all paper, whichever way
        round the polarity is. Sampling inside the box instead picks the INK
        colour for a dense bold column, where ink is the majority of the box.
        """
        ih, iw = img.shape[:2]
        x1, y1, x2, y2 = box
        ring = max(2, min(x2 - x1, y2 - y1) // 6)
        rx1, ry1 = max(0, x1 - ring), max(0, y1 - ring)
        rx2, ry2 = min(iw, x2 + ring), min(ih, y2 + ring)
        around = TranslationPipeline._grey(img[ry1:ry2, rx1:rx2])
        outside = np.ones(around.shape, bool)
        outside[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
        samples = around[outside] if outside.any() else around.ravel()
        hist = np.bincount((samples >> 3).ravel(), minlength=32)
        return int(hist.argmax()) * 8 + 4, (rx1, ry1, rx2, ry2)

    @staticmethod
    def _compute_balloon_xyxy(img, text_xyxy, check_center=True, why=None,
                              all_text=None, out=None, check_fill=True,
                              check_size=True):
        """Find the balloon or caption frame enclosing `text_xyxy` and return
        its bounding box in image coords, or None if there isn't a clean one.

        `check_center=False` skips the test that the text sits in the middle of
        the patch, for _assign_balloons, which applies it to all the text in a
        balloon together rather than block by block. Pass a list as `why` to
        have the reason for a None appended to it. `all_text` is every text box
        on the page, so that other text sitting on the same patch is not
        mistaken for something foreign inside it. A dict passed as `out` gets
        the patch itself under 'mask', as a boolean array over the returned box.
        `check_fill=False` leaves the solidity test to the caller, which needs
        to apply it per lobe: two balloons joined at an angle fill little of
        the box around the pair, and a good deal of the box around each.
        `check_size=False` likewise leaves the patch-versus-text size test to
        the caller, which has to weigh the patch against ALL the text on it.

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
          - it must not be the page. A flat background around a figure is one
            connected patch of paper too, and when the capture includes a
            sliver of whatever surrounds the picture that patch is neatly
            enclosed — the whole background, or just the piece of it on one
            side of the figure. Three things tell it from a balloon: it is over
            half the capture; or it has something large sitting inside it that
            is not this text (another balloon, typically — a balloon holds
            glyphs and nothing else); or the text is not in the middle of it.
          - it must be reasonably solid rather than a thin leak along gutters.

        The checks lean towards None on purpose. Calling a background a balloon
        throws the translation into the middle of the artwork; missing a real
        balloon only falls back to drawing it where the original text was.
        Any miss returns None and the caller falls back to the text bbox, so a
        miss never produces a worse layout than no detection at all.
        """
        def no(reason):
            if why is not None:
                why.append(reason)
            return None

        last = 'no search was possible'
        try:
            import cv2

            ih, iw = img.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in text_xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            tw, th = x2 - x1, y2 - y1
            if tw < 2 or th < 2:
                return no('text box is empty')
            tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2

            cls = TranslationPipeline
            grey = cls._grey
            paper_level, (rx1, ry1, rx2, ry2) = cls._paper_level(img, (x1, y1, x2, y2))

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
                    return no('no paper-coloured pixels around the text')

                # The paper patch that owns most of the text bbox and its
                # ring. Counters enclosed by glyphs are separate, much smaller
                # components.
                inside = labels[ry1 - cy1:ry2 - cy1, rx1 - cx1:rx2 - cx1]
                owned = np.bincount(inside.ravel(), minlength=n)
                owned[0] = 0
                label = int(owned.argmax())
                if owned[label] == 0:
                    return no('no paper patch under the text')
                lx, ly, lw, lh, area = (int(v) for v in stats[label])

                at_l, at_t = lx == 0, ly == 0
                at_r, at_b = lx + lw == crop_w, ly + lh == crop_h
                cut_off = ((at_l and cx1 > 0) or (at_t and cy1 > 0)
                           or (at_r and cx2 < iw) or (at_b and cy2 < ih))
                if cut_off:
                    last = 'patch runs past the search area (open, or very large)'
                    continue    # extent unknown at this crop size

                # Extent is real from here on, so a failed check is final.
                if at_l + at_t + at_r + at_b >= 3:
                    return no('patch reaches three or more edges of the capture')
                bx1, by1 = cx1 + lx, cy1 + ly
                bx2, by2 = bx1 + lw, by1 + lh
                if not (bx1 <= tcx <= bx2 and by1 <= tcy <= by2):
                    return no('patch does not contain the text centre')
                if lw * lh < tw * th:
                    return no('patch is smaller than the text')
                share = lw * lh / (iw * ih)
                if share > cls._BALLOON_ABS_PAGE_SHARE:
                    return no(f'patch is {share:.0%} of the capture (the page)')
                if check_center and not cls._text_is_centered(
                        (bx1, by1, bx2, by2), (x1, y1, x2, y2)):
                    return no('text is not in the middle of the patch')

                # Anything large enclosed by the patch that is not text sitting
                # on it. This block's glyphs are blanked first, and so is any
                # other block whose own paper is this same patch: a second
                # paragraph, or the text of a balloon joined onto this one.
                # Dense glyphs run together into one blob easily big enough to
                # look like an object. Text standing on a DIFFERENT patch is
                # left in — that is another balloon lying inside a background.
                foreign = labels[ly:ly + lh, lx:lx + lw] != label

                def blank(box):
                    ring = max(2, min(box[2] - box[0], box[3] - box[1]) // 6)
                    ax1 = max(0, box[0] - ring - cx1)
                    ay1 = max(0, box[1] - ring - cy1)
                    ax2 = min(crop_w, box[2] + ring - cx1)
                    ay2 = min(crop_h, box[3] + ring - cy1)
                    if ax2 <= ax1 or ay2 <= ay1:
                        return
                    counts = np.bincount(labels[ay1:ay2, ax1:ax2].ravel(), minlength=n)
                    counts[0] = 0
                    if int(counts.argmax()) != label or counts[label] == 0:
                        return          # stands on some other patch
                    foreign[max(0, ay1 - ly):max(0, ay2 - ly),
                            max(0, ax1 - lx):max(0, ax2 - lx)] = False

                blank((x1, y1, x2, y2))
                for other in all_text or ():
                    blank(tuple(int(v) for v in other))
                fn, _fl, fstats, _ = cv2.connectedComponentsWithStats(
                    foreign.astype(np.uint8), connectivity=8)
                for k in range(1, fn):
                    ox, oy, ow, oh, oarea = (int(v) for v in fstats[k])
                    enclosed = ox > 0 and oy > 0 and ox + ow < lw and oy + oh < lh
                    if enclosed and oarea > cls._BALLOON_MAX_ENCLOSED * lw * lh:
                        return no(f'patch encloses a foreign object '
                                  f'({oarea / (lw * lh):.0%} of it)')

                if check_size:
                    too_big = cls._size_reason(
                        (bx1, by1, bx2, by2), (x1, y1, x2, y2), iw, ih)
                    if too_big:
                        return no(too_big)
                if check_fill and area < cls._BALLOON_MIN_FILL * lw * lh:
                    return no(f'patch fills only {area / (lw * lh):.0%} of its box')
                if out is not None:
                    out['mask'] = labels[ly:ly + lh, lx:lx + lw] == label
                return (bx1, by1, bx2, by2)

            return no(last)
        except Exception as e:
            print(f'[screen_translator] balloon detect failed '
                  f'({type(e).__name__}: {e}); using text bbox')
            return None

    # ── Introspection for UI tooltips / About dialog ───────────────────────

    @property
    def current(self) -> Optional[SimpleNamespace]:
        return self._current
