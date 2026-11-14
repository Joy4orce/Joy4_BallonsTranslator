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
    """Fake columns of vertical glyphs centred in `box`, the way text sits in a
    balloon or a caption frame. Returns the ink bbox."""
    bx1, by1, bx2, by2 = box
    pad, col_w, gap = 12, 16, 14
    total = columns * col_w + (columns - 1) * gap
    rightmost = (bx1 + bx2 + total) // 2 - col_w
    xs = [rightmost - i * (col_w + gap) for i in range(columns)]
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

    # A wide frame around a narrow run of text — the shape a crop sized from
    # the text's own aspect ratio can never contain.
    page = new_page()
    frame = (300, 80, 560, 420)
    rect_frame(page, frame)
    hit('wide frame, narrow text in the middle', page,
        vertical_text(page, (378, 80, 482, 420), 2), frame)

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

    # A flat page background with text in one corner, captured with a sliver of
    # the dark viewer around the picture. The background is then one neatly
    # enclosed patch of paper; called a balloon, the text is thrown into the
    # middle of the picture. (Seen on a real character sheet.)
    page, text = character_sheet()
    miss('flat page background, dark surround', page, text)

    # Same sheet, but the figure reaches the top of the picture and cuts the
    # background in two. The piece on the list's side is under half the
    # capture, so size alone does not give it away; the list sitting in its
    # corner does. (This is the variant that got through first time.)
    page, text = character_sheet(figure_to_top=True)
    miss('background split by the figure', page, text)

    # And with the list in the MIDDLE of its piece of background, so position
    # does not give it away either — but that piece also holds a speech
    # balloon, and a balloon does not contain another balloon.
    page, text = character_sheet(figure_to_top=True, balloon_on_list_side=True)
    miss('background piece holding a balloon', page, text)

    # Deliberately conservative: text hard against one side of a much wider
    # frame is NOT taken for that frame's text, because the same geometry is
    # what a background with text in its corner looks like. It falls back to
    # being drawn where the original text was.
    page = new_page()
    frame = (300, 80, 560, 420)
    rect_frame(page, frame)
    miss('text against one side of a wide frame', page,
         vertical_text(page, (500, 80, 560, 420), 1))

    return cases


def list_rows(page, origin, rows, glyph=18, leading=5, ink=INK):
    """Draw rows of fake horizontal glyphs. `rows` is a list of (bullet, n):
    whether the row opens with a bullet dot, and how many glyphs it has.
    Returns the ink bbox of the whole list."""
    x0, y = origin
    right = x0
    for bullet, n in rows:
        x = x0
        if bullet:
            cy = y + glyph // 2
            cv2.rectangle(page, (x, cy - 3), (x + 6, cy + 3), ink, -1)
            x += 6 + glyph // 2
        for _ in range(n):
            cv2.rectangle(page, (x, y), (x + glyph - 4, y + glyph - 1), ink, -1)
            x += glyph
        right = max(right, x - 4)
        y += glyph + leading
    return (x0, origin[1], right, y - leading)


def character_sheet(margin=20, figure_to_top=False, balloon_on_list_side=False):
    """A profile-card page: flat grey ground, a figure rising from the bottom
    edge, a speech balloon, a list on the left — inside a dark surround, as
    when the capture is dragged a little wider than the picture.

    `figure_to_top` runs the figure up to the top edge, splitting the ground
    into a left and a right piece. `balloon_on_list_side` puts the balloon in
    the left piece and the list in the middle of that piece instead of its
    corner. Returns (page, list text bbox)."""
    w, h = 720, 500
    page = np.full((h, w, 3), 222, np.uint8)
    if figure_to_top:
        figure = np.array([[300, h], [320, 0], [480, 0], [520, h]], np.int32)
    else:
        figure = np.array([[300, h], [320, 180], [400, 90], [480, 180], [520, h]], np.int32)
    cv2.fillPoly(page, [figure], (60, 60, 70))
    spec = [(False, 4), (False, 9), (True, 6), (True, 10), (True, 8)]
    if balloon_on_list_side:
        ellipse(page, (150, 425), (50, 60))
        text = list_rows(page, (55, 195), spec)
    else:
        ellipse(page, (610, 200), (70, 120))
        text = list_rows(page, (40, 50), spec)
    page = cv2.copyMakeBorder(page, margin, margin, margin, margin,
                              cv2.BORDER_CONSTANT, value=(30, 30, 30))
    return page, tuple(v + margin for v in text)


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

    failures += check_shared_balloon(TranslationPipeline)
    failures += check_lists(TranslationPipeline)

    print()
    print('all cases passed' if not failures else f'{failures} case(s) failed')
    return 1 if failures else 0


def check_lists(pipeline_cls):
    """A horizontal list block must be read row by row and keep its rows.

    Read as one image the rows come back run together, so the translation is a
    single paragraph and the items cannot be put back on their own lines. The
    OCR is stubbed out here: what is under test is the splitting, the bullet
    handling and the reassembly, not the model.
    """
    from utils.textblock import TextBlock

    fails = 0

    def expect(label, cond, detail):
        nonlocal fails
        fails += 0 if cond else 1
        print(f'{"PASS" if cond else "FAIL"}  {label:36} {detail}')

    spec = [(False, 9), (True, 6), (True, 10), (True, 8), (True, 7)]
    page = np.full((400, 600, 3), 222, np.uint8)
    title_box = list_rows(page, (40, 40), [(False, 4)])
    box = list_rows(page, (40, 40 + 23), spec)

    # Row splitting, including on a capture scaled the way a 125% display
    # scales it — tight leading plus interpolation leaves no empty line between
    # rows, which is what welded the whole block into one row at first.
    for scale in (1.0, 1.25):
        img = page if scale == 1.0 else cv2.resize(
            page, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        sbox = tuple(int(round(v * scale)) for v in box)
        rows = pipeline_cls._split_rows(img, sbox)
        expect(f'list rows found at {scale}x', rows is not None and len(rows) == len(spec),
               f'{None if rows is None else len(rows)} of {len(spec)}')
        if rows is None or len(rows) != len(spec):
            continue
        paper, _ = pipeline_cls._paper_level(img, sbox)
        cuts = [pipeline_cls._leading_bullet_width(img[y1:y2, x1:x2], paper) > 0
                for x1, y1, x2, y2 in rows]
        expect(f'bullets told from text at {scale}x', cuts == [b for b, _ in spec], str(cuts))

    rows = pipeline_cls._split_rows(page, title_box)
    expect('a single row is not a list', rows is None, str(rows))

    class StubOCR:
        """Stands in for the model: every row reads as plain text, no marker."""
        def ocr_img(self, img):
            return 'テキスト'

    def block(xyxy, text):
        b = TextBlock()
        x1, y1, x2, y2 = xyxy
        b.set_lines_by_xywh([x1, y1, x2 - x1, y2 - y1], adjust_bbox=True)
        b.vertical = False
        b.text = text
        return b

    pipe = pipeline_cls.__new__(pipeline_cls)
    pipe._ocr = StubOCR()

    title, lst = block(title_box, '見出し'), block(box, '一塊に読まれた文字列')
    balloon_text = block((300, 300, 340, 380), 'たてがき')
    balloon_text.vertical = True
    # The title is deliberately NOT first: removing it must not depend on
    # comparing blocks with ==, which TextBlock does not support safely.
    blocks = [balloon_text, lst, title]
    pipe._reread_lists(page, blocks)

    lines = lst.get_text().split('\n')
    expect('title folded in, rows kept', len(lines) == 1 + len(spec) and lines[0] == '見出し',
           repr(lines))
    marks = [ln.startswith('・') for ln in lines[1:]]
    expect('markers restored on bullet rows', marks == [b for b, _ in spec], str(marks))
    expect('title block removed from the list',
           len(blocks) == 2 and not any(b is title for b in blocks),
           f'{len(blocks)} blocks left')
    expect('vertical block left untouched', balloon_text.get_text() == 'たてがき',
           repr(balloon_text.get_text()))

    # No bullets anywhere: rows are wrapping, the original reading must stand.
    page2 = np.full((300, 600, 3), 222, np.uint8)
    para = block(list_rows(page2, (40, 40), [(False, 12), (False, 11), (False, 7)]),
                 '段落として読まれた文字列')
    pipe._reread_lists(page2, [para])
    expect('plain paragraph keeps its reading', para.get_text() == '段落として読まれた文字列',
           repr(para.get_text()))
    return fails


def check_shared_balloon(pipeline_cls):
    """Two text blocks inside one balloon must come out as ONE block, joined in
    reading order. Left as two, each is laid out across the whole balloon and
    the translations are drawn on top of each other."""
    from tools.screen_translator.pipeline import TranslatedBlock

    page = new_page(500, 500, busy=False)
    ellipse(page, (250, 250), (150, 110))
    right = vertical_text(page, (258, 178, 328, 322), 2)
    left = vertical_text(page, (158, 178, 228, 322), 2)
    other_frame = (20, 20, 120, 120)
    rect_frame(page, other_frame)
    other = vertical_text(page, other_frame, 1)

    # Through _assign_balloons, as the pipeline does: each paragraph on its own
    # is off-centre in the balloon, and it is the two together that are centred.
    # Detection order deliberately not reading order: left paragraph first.
    loose = (400, 400, 440, 480)
    balloons = pipeline_cls._assign_balloons(page, [left, other, right, loose])
    blocks = [
        TranslatedBlock(left, 0.0, 'L', 'second', balloons[0]),
        TranslatedBlock(other, 0.0, 'O', 'elsewhere', balloons[1]),
        TranslatedBlock(right, 0.0, 'R', 'first', balloons[2]),
        TranslatedBlock(loose, 0.0, 'N', 'no balloon', balloons[3]),
    ]

    fails = 0

    def expect(label, cond, detail):
        nonlocal fails
        fails += 0 if cond else 1
        print(f'{"PASS" if cond else "FAIL"}  {label:36} {detail}')

    same = blocks[0].balloon_xyxy is not None and blocks[0].balloon_xyxy == blocks[2].balloon_xyxy
    expect('two blocks, one balloon: same rect', same,
           f'{blocks[0].balloon_xyxy} / {blocks[2].balloon_xyxy}')

    rtl = pipeline_cls._merge_blocks_sharing_balloon(blocks, rtl=True)
    texts = [b.translation for b in rtl]
    expect('merged into one, right-to-left',
           texts == ['first\nsecond', 'elsewhere', 'no balloon'], repr(texts))

    ltr = pipeline_cls._merge_blocks_sharing_balloon(blocks, rtl=False)
    expect('merged into one, left-to-right',
           ltr[0].translation == 'second\nfirst', repr(ltr[0].translation))

    m = rtl[0]
    expect('merged text bbox spans both',
           m.xyxy == (left[0], min(left[1], right[1]), right[2], max(left[3], right[3])),
           str(m.xyxy))
    return fails


if __name__ == '__main__':
    os.chdir(APP_ROOT)
    sys.exit(main())
