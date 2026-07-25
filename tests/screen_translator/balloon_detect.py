"""Balloon detection for the screen translator, on synthetic pages.

The overlay lays a translation out to fill the balloon it was given; when
detection misses, it falls back to the narrow OCR text bbox and the result reads
as misaligned with the frame it belongs to. These cases cover the shapes that
actually appear in the comics this tool gets pointed at.

Assertions are on properties rather than exact pixel counts, so the test stays
meaningful when the segmentation is tuned:

  - a rectangular caption frame must be recovered nearly in full (the reported
    bug: a tall column of vertical text yielded a crop too narrow to contain the
    frame, so only a sliver around the ink came back)
  - the same frame flush against the panel edge must still be recovered
  - a round balloon must at least be found and be clearly larger than the ink
  - text on flat background with no enclosing shape must return None, rather
    than a bogus region invented by a flood fill that ran to the crop border

Run: python tests/screen_translator/balloon_detect.py
"""
import os
import os.path as osp
import sys

import cv2
import numpy as np

APP_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(APP_ROOT)

PAGE_W, PAGE_H = 724, 1288      # a real capture size from the screen translator


def new_page():
    """White page with busy artwork, so crops are never trivially flat."""
    page = np.full((PAGE_H, PAGE_W, 3), 255, np.uint8)
    rng = np.random.default_rng(1)
    page[600:, :] = rng.integers(120, 250, (PAGE_H - 600, PAGE_W, 3),
                                 dtype=np.uint8)
    for x in range(0, PAGE_W, 17):
        cv2.line(page, (x, 300), (x + 60, 590), (60, 60, 60), 1)
    return page


def draw_vertical_text(page, frame, columns=3):
    """Fake columns of vertical glyphs inside `frame`. Returns the ink bbox."""
    fx1, fy1, fx2, fy2 = frame
    pad, col_w, gap = 12, 16, 14
    xs, x = [], fx2 - pad - col_w
    for _ in range(columns):
        xs.append(x)
        x -= col_w + gap
    y0, y1 = fy1 + pad, fy2 - pad
    for cx in xs:
        for gy in range(y0, y1 - 14, 22):
            cv2.rectangle(page, (cx, gy), (cx + col_w, gy + 14), (20, 20, 20), -1)
    return (min(xs), y0, max(xs) + col_w, y1)


def rect_frame(page, frame):
    cv2.rectangle(page, frame[:2], frame[2:], (255, 255, 255), -1)
    cv2.rectangle(page, frame[:2], frame[2:], (0, 0, 0), 2)


def build_cases():
    cases = []

    page = new_page()
    frame = (60, 90, 260, 400)
    rect_frame(page, frame)
    cases.append(dict(
        name='rect caption frame, vertical text',
        page=page, text=draw_vertical_text(page, frame, 4),
        want='cover', frame=frame, min_cover=0.85,
    ))

    page = new_page()
    frame = (0, 90, 200, 380)
    rect_frame(page, frame)
    cases.append(dict(
        name='rect caption frame at panel edge',
        page=page, text=draw_vertical_text(page, frame, 3),
        want='cover', frame=frame, min_cover=0.85,
    ))

    page = new_page()
    cv2.ellipse(page, (500, 200), (110, 80), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (500, 200), (110, 80), 0, 0, 360, (0, 0, 0), 2)
    cases.append(dict(
        name='round speech balloon',
        page=page, text=draw_vertical_text(page, (420, 140, 570, 260), 2),
        want='bigger_than_text', min_area_ratio=1.5,
    ))

    page = new_page()
    cases.append(dict(
        name='no enclosing shape, text on background',
        page=page, text=draw_vertical_text(page, (300, 700, 460, 980), 3),
        want='none',
    ))

    return cases


def main():
    from tools.screen_translator.pipeline import TranslationPipeline
    detect = TranslationPipeline._compute_balloon_xyxy

    failures = 0
    for case in build_cases():
        text = case['text']
        got = detect(case['page'], text)
        tw, th = text[2] - text[0], text[3] - text[1]
        want = case['want']

        if want == 'none':
            ok = got is None
            detail = 'None as required' if ok else f'invented {got}'
        elif got is None:
            ok = False
            detail = 'no balloon found; would fall back to the text bbox'
        elif want == 'cover':
            fx1, fy1, fx2, fy2 = case['frame']
            fw, fh = fx2 - fx1, fy2 - fy1
            gw, gh = got[2] - got[0], got[3] - got[1]
            cover_w, cover_h = gw / fw, gh / fh
            ok = cover_w >= case['min_cover'] and cover_h >= case['min_cover']
            detail = (f'{gw}x{gh} of frame {fw}x{fh} '
                      f'({cover_w:.0%} x {cover_h:.0%} covered)')
        else:   # bigger_than_text
            gw, gh = got[2] - got[0], got[3] - got[1]
            ratio = (gw * gh) / max(1, tw * th)
            ok = ratio >= case['min_area_ratio']
            detail = (f'{gw}x{gh}, {ratio:.1f}x the ink area '
                      f'(need {case["min_area_ratio"]}x)')

        failures += 0 if ok else 1
        print(f'{"PASS" if ok else "FAIL"}  {case["name"]:36} {detail}')

    print()
    print('all cases passed' if not failures else f'{failures} case(s) failed')
    return 1 if failures else 0


if __name__ == '__main__':
    os.chdir(APP_ROOT)
    sys.exit(main())
