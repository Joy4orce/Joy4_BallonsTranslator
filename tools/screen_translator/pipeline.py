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
        try:
            self._reread_lists(img, blk_list)
        except Exception as e:
            # Purely an improvement over the block-level read; never let it
            # take the capture down.
            print(f'[screen_translator] list re-read skipped '
                  f'({type(e).__name__}: {e})')
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

        boxes = [tuple(int(v) for v in blk.xyxy) for blk in blk_list]
        if self._current.fill_balloon:
            # Use the ORIGINAL img (not inpainted) — the balloon outline is
            # what we trace, and inpaint doesn't touch it anyway.
            balloons = self._assign_balloons(img, boxes)
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
            ))

        if self._current.fill_balloon:
            from modules.translators.base import RTL_SOURCE_LANGS
            out = self._merge_blocks_sharing_balloon(
                out, rtl=self._current.source_lang in RTL_SOURCE_LANGS)
        return out, inpainted_img

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
    def _assign_balloons(cls, img, boxes):
        """The balloon for each text box, or None — one entry per box.

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
        found = [cls._compute_balloon_xyxy(img, box, check_center=False)
                 for box in boxes]
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

        out = [None] * len(boxes)
        for group in groups:
            text = (min(boxes[i][0] for i in group), min(boxes[i][1] for i in group),
                    max(boxes[i][2] for i in group), max(boxes[i][3] for i in group))
            balloon = found[group[0]]
            if cls._text_is_centered(balloon, text):
                for i in group:
                    out[i] = found[i]
        return out

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
    def _split_rows(img, box):
        """Split a horizontal text block into its rows by where the ink is.

        Returns one (x1, y1, x2, y2) per row, top to bottom, or None when the
        block does not separate cleanly into two or more rows.

        The detector's own line polygons are not usable for this: on a profile
        card they came back as one polygon covering most of the block plus a few
        partial ones, with a whole row missing. A projection of the ink onto the
        vertical axis is crude but dependable for what it is needed for — text
        set in rows on a plain ground. Over artwork the "ink" is everything and
        the rows come out irregular, which is caught and refused.
        """
        cls = TranslationPipeline
        ih, iw = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 8:
            return None

        paper, _ = cls._paper_level(img, (x1, y1, x2, y2))
        crop = cls._grey(img[y1:y2, x1:x2]).astype(np.int16)
        ink = np.abs(crop - paper) > cls._BALLOON_TOL
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

        # A sliver of a gap, or a short run next to a full one (dakuten that
        # cleared the line, a ruby line), belongs to its neighbour.
        rows = [runs[0]]
        for s, e in runs[1:]:
            gap = s - rows[-1][1]
            short = min(e - s, rows[-1][1] - rows[-1][0]) < 0.4 * typical
            if gap <= max(1, 0.15 * typical) or (short and gap <= 0.5 * typical):
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
            boxes.append((max(0, x1 + int(cols[0]) - pad), max(0, y1 + s - pad),
                          min(iw, x1 + int(cols[-1]) + 1 + pad), min(ih, y1 + e + pad)))
        return boxes if len(boxes) >= 2 else None

    def _reread_lists(self, img, blk_list):
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
        ih, iw = img.shape[:2]
        lists = []          # (block, rows) for every block rewritten as a list
        for blk in blk_list:
            if getattr(blk, 'vertical', False):
                continue
            rows = self._split_rows(img, blk.xyxy)
            if not rows or len(rows) > 30:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in blk.xyxy]
            paper, _ = self._paper_level(
                img, (max(0, bx1), max(0, by1), min(iw, bx2), min(ih, by2)))

            texts = []
            for rx1, ry1, rx2, ry2 in rows:
                # A bullet is taken off by shape and the rest read without it.
                # Left in, the OCR tends to fuse it into the first character
                # (・シ read as ジ), which loses both the marker and the word.
                cut = self._leading_bullet_width(img[ry1:ry2, rx1:rx2], paper)
                t = read_row(self._pad_row(img[ry1:ry2, rx1 + cut:rx2], paper))
                t = (t if isinstance(t, str) else '').strip()
                if not t:
                    continue
                if cut and not self._is_list_item(t):
                    t = '・' + t
                texts.append(t)
            if len(texts) < 2 or not any(self._is_list_item(t) for t in texts):
                continue
            blk.text = '\n'.join(texts)
            lists.append((blk, len(texts)))
            print(f'[screen_translator] list block re-read as {len(texts)} rows')

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
    def _leading_bullet_width(row_img, paper):
        """Width, in pixels from the left of a row crop, taken up by a leading
        bullet and the gap after it; 0 when the row does not start with one.

        A bullet is recognised by shape: the first blob of ink is small in both
        directions next to the row (a dot or a dash, not a character), stands
        clear of what follows, and something does follow.
        """
        cls = TranslationPipeline
        ink = np.abs(cls._grey(row_img).astype(np.int16) - paper) > cls._BALLOON_TOL
        rows_with_ink = np.flatnonzero(ink.any(axis=1))
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
                     and nxt - end >= 0.12 * row_h)
        return nxt if is_bullet else 0

    # Balloon search tuning. See _compute_balloon_xyxy.
    _BALLOON_PADS = (1.0, 2.0)       # crop margin, in multiples of the text's long side
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
    def _compute_balloon_xyxy(img, text_xyxy, check_center=True):
        """Find the balloon or caption frame enclosing `text_xyxy` and return
        its bounding box in image coords, or None if there isn't a clean one.

        `check_center=False` skips the test that the text sits in the middle of
        the patch, for _assign_balloons, which applies it to all the text in a
        balloon together rather than block by block.

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
                if lw * lh > cls._BALLOON_ABS_PAGE_SHARE * iw * ih:
                    return None
                if check_center and not cls._text_is_centered(
                        (bx1, by1, bx2, by2), (x1, y1, x2, y2)):
                    return None

                # Anything large enclosed by the patch that is not this text.
                # Own glyphs are blanked first; another paragraph's glyphs in
                # the same balloon are each far too small to count.
                foreign = labels[ly:ly + lh, lx:lx + lw] != label
                fx1, fy1 = max(0, rx1 - cx1 - lx), max(0, ry1 - cy1 - ly)
                fx2, fy2 = min(lw, rx2 - cx1 - lx), min(lh, ry2 - cy1 - ly)
                if fx2 > fx1 and fy2 > fy1:
                    foreign[fy1:fy2, fx1:fx2] = False
                fn, _fl, fstats, _ = cv2.connectedComponentsWithStats(
                    foreign.astype(np.uint8), connectivity=8)
                for k in range(1, fn):
                    ox, oy, ow, oh, oarea = (int(v) for v in fstats[k])
                    enclosed = ox > 0 and oy > 0 and ox + ow < lw and oy + oh < lh
                    if enclosed and oarea > cls._BALLOON_MAX_ENCLOSED * lw * lh:
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
