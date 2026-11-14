"""Balloon detection for the screen translator, on synthetic pages.

The overlay lays a translation out to fill the balloon it was given; when
detection misses, it falls back to the narrow OCR text bbox, and when detection
is wrong the text lands somewhere it does not belong. These cases cover the
shapes that actually appear in the comics this tool gets pointed at, and the
ways an earlier version of the detector got them wrong.

A hit is checked against the true outline on every side (within TOL pixels), not
just for being "big enough" — a region truncated on one side passes any area
test and still centers the text in the wrong place. Cases that must not produce
a balloon are checked for None.

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
TOL = 6                         # px a detected edge may sit from the true outline
INK = (20, 20, 20)


def new_page(w=PAGE_W, h=PAGE_H, busy=True, paper=255):
    """A page with busy artwork, so crops are never trivially flat."""
    page = np.full((h, w, 3), paper, np.uint8)
    if busy:
        rng = np.random.default_rng(1)
        page[h // 2:, :] = rng.integers(120, 250, (h - h // 2, w, 3), dtype=np.uint8)
        for x in range(0, w, 17):
            cv2.line(page, (x, h // 4), (x + 60, h // 2 - 10), (60, 60, 60), 1)
    return page


def vertical_text(page, box, columns=3, ink=INK):
    """Fake columns of vertical glyphs inside `box`. Returns the ink bbox."""
    bx1, by1, bx2, by2 = box
    pad, col_w, gap = 12, 16, 14
    xs, x = [], bx2 - pad - col_w
    for _ in range(columns):
        xs.append(x)
        x -= col_w + gap
    y0, y1 = by1 + pad, by2 - pad
    for cx in xs:
        for gy in range(y0, y1 - 14, 22):
            cv2.rectangle(page, (cx, gy), (cx + col_w, gy + 14), ink, -1)
    return (min(xs), y0, max(xs) + col_w, y1)


def rect_frame(page, frame, fill=(255, 255, 255), line=(0, 0, 0)):
    cv2.rectangle(page, frame[:2], frame[2:], fill, -1)
    cv2.rectangle(page, frame[:2], frame[2:], line, 2)


def ellipse(page, center, axes, fill=(255, 255, 255), line=(0, 0, 0)):
    cv2.ellipse(page, center, axes, 0, 0, 360, fill, -1)
    cv2.ellipse(page, center, axes, 0, 0, 360, line, 2)
    cx, cy = center
    ax, ay = axes
    return (cx - ax, cy - ay, cx + ax, cy + ay)


def build_cases():
    cases = []

    def hit(name, page, text, outline):
        cases.append(dict(name=name, page=page, text=text, outline=outline))

    def miss(name, page, text):
        cases.append(dict(name=name, page=page, text=text, outline=None))

    # --- shapes that must be recovered, edge for edge -------------------
    page = new_page()
    frame = (60, 90, 260, 400)
    rect_frame(page, frame)
    hit('rect caption frame, vertical text', page,
        vertical_text(page, frame, 4), frame)

    page = new_page()
    frame = (0, 90, 200, 380)
    rect_frame(page, frame)
    hit('rect caption frame at page edge', page,
        vertical_text(page, frame, 3), frame)

    # A wide frame around ONE narrow column — the shape a crop sized from the
    # text's own aspect ratio can never contain.
    page = new_page()
    frame = (300, 80, 560, 420)
    rect_frame(page, frame)
    hit('wide frame, single narrow column', page,
        vertical_text(page, (500, 80, 560, 420), 1), frame)

    page = new_page()
    outline = ellipse(page, (500, 200), (110, 80))
    hit('round speech balloon over artwork', page,
        vertical_text(page, (420, 140, 570, 260), 2), outline)

    # Tall thin balloon, one column, nothing else around: its bbox is a small
    # share of any generous crop.
    page = new_page(700, 900, busy=False)
    outline = ellipse(page, (350, 400), (45, 190))
    hit('isolated tall balloon, one column', page,
        vertical_text(page, (322, 250, 378, 550), 1), outline)

    page = new_page()
    outline = ellipse(page, (360, 220), (120, 90), fill=(200, 200, 200))
    hit('grey-filled balloon', page,
        vertical_text(page, (290, 160, 430, 280), 2), outline)

    page = new_page()
    frame = (80, 100, 300, 380)
    rect_frame(page, frame, fill=(25, 25, 25), line=(255, 255, 255))
    hit('dark frame, light text', page,
        vertical_text(page, frame, 3, ink=(235, 235, 235)), frame)

    # --- situations that must NOT produce a balloon ----------------------
    page = new_page()
    miss('text straight on artwork, no shape', page,
         vertical_text(page, (300, 700, 460, 980), 3))

    # Small capture of borderless text: the paper colour floods the whole image.
    page = new_page(260, 100, busy=False, paper=235)
    cv2.putText(page, 'HELLO WORLD', (34, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, INK, 2)
    miss('small flat capture, borderless text', page, (30, 36, 230, 64))

    # Narration sitting directly in a big empty panel: the enclosed region is
    # the panel, not a balloon.
    page = new_page(busy=False)
    cv2.rectangle(page, (40, 100), (680, 900), (0, 0, 0), 3)
    miss('text loose inside a large panel', page,
         vertical_text(page, (330, 400, 390, 620), 1))

    return cases


def main():
    from tools.screen_translator.pipeline import TranslationPipeline
    detect = TranslationPipeline._compute_balloon_xyxy

    failures = 0
    for case in build_cases():
        got = detect(case['page'], case['text'])
        outline = case['outline']

        if outline is None:
            ok = got is None
            detail = 'None as required' if ok else f'invented {got}'
        elif got is None:
            ok = False
            detail = 'no balloon found; would fall back to the text bbox'
        else:
            off = [g - o for g, o in zip(got, outline)]
            ok = all(abs(d) <= TOL for d in off)
            detail = (f'{got} vs outline {outline}, '
                      f'edge offsets L{off[0]:+d} T{off[1]:+d} R{off[2]:+d} B{off[3]:+d}')

        failures += 0 if ok else 1
        print(f'{"PASS" if ok else "FAIL"}  {case["name"]:36} {detail}')

    print()
    print('all cases passed' if not failures else f'{failures} case(s) failed')
    return 1 if failures else 0


if __name__ == '__main__':
    os.chdir(APP_ROOT)
    sys.exit(main())
