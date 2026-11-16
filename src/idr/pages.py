"""Bước 1: phân tích cấu trúc PDF International Debt Report 2026 và render trang cần vision.

Chạy: .venv/Scripts/python -m src.idr.pages
Output:
  data/idr/pages.json   mỗi trang: pdf_page, printed_page (label), section, kind, captions, bbox figure/bảng
  data/idr/pages/pNNN.png  ảnh trang 150 dpi (không commit; tái tạo bằng lệnh trên)
  data/idr/crops/<id>.png  ảnh crop vùng figure/bảng (từ dòng caption tới hết dòng Source/Note)
"""
from __future__ import annotations

import json
import re

import pymupdf

from .. import config

PDF = config.DATA_DIR / '9781464823558.pdf'
OUT = config.DATA_DIR / 'idr'
DPI = 150
CAPTION = re.compile(r'^(Figure|Table|Box) ([A-Z]?[0-9]+[A-Z]?\.[0-9]+(?:\.[0-9]+)?)\b')


def section_of(pdf_page: int) -> str:
    if pdf_page <= 24:
        return 'front_matter'
    if pdf_page <= 74:
        return 'chapter_1'
    if pdf_page <= 106:
        return 'chapter_2'
    if pdf_page <= 134:
        return 'chapter_3'
    if pdf_page <= 263:
        return 'part2_profiles'
    return 'appendix'


def caption_regions(page: pymupdf.Page) -> list[dict]:
    """Vùng của mỗi Figure/Table: từ block caption tới block 'Source(s)/Note' kế tiếp (hoặc caption kế/cuối trang)."""
    blocks = sorted(page.get_text('blocks'), key=lambda b: (round(b[1]), b[0]))
    caps = []
    for b in blocks:
        m = CAPTION.match(b[4].strip())
        if m and m.group(1) != 'Box':
            caps.append((f'{m.group(1)} {m.group(2)}', b, b[4].strip().replace('\n', ' ')[:160]))
    regions = []
    for i, (cid, b, title) in enumerate(caps):
        y0 = b[1]
        y_next = caps[i + 1][1][1] if i + 1 < len(caps) else page.rect.y1
        y1 = y_next
        for s in blocks:
            if y0 < s[1] < y_next and re.match(r'^(Sources?|Note)s?:', s[4].strip()):
                y1 = max(s[3], y0)
        # Cột trái/phải: figure có thể chỉ chiếm một cột -> lấy hết bề rộng trang cho an toàn.
        rect = pymupdf.Rect(page.rect.x0 + 20, y0 - 4, page.rect.x1 - 20, min(y1 + 6, page.rect.y1))
        regions.append({'id': cid, 'title': title, 'bbox': [round(v, 1) for v in rect]})
    return regions


def main() -> None:
    doc = pymupdf.open(PDF)
    (OUT / 'pages').mkdir(parents=True, exist_ok=True)
    (OUT / 'crops').mkdir(parents=True, exist_ok=True)
    pages = []
    for i, page in enumerate(doc):
        n = i + 1
        text = page.get_text()
        sec = section_of(n)
        regions = caption_regions(page) if sec.startswith('chapter') or sec == 'front_matter' else []
        n_draw = len(page.get_drawings())
        if sec == 'part2_profiles' and len(text) > 200:
            kind = 'profile_howto' if n == 136 else 'profile'
        elif regions:
            kind = 'visual'
        elif len(text.strip()) < 40:
            kind = 'blank_or_cover'
        else:
            kind = 'text'
        needs_vision = kind in ('visual', 'profile', 'profile_howto') or (n_draw > 150 and kind == 'text')
        rec = {'pdf_page': n, 'printed_page': page.get_label(), 'section': sec, 'kind': kind,
               'needs_vision': needs_vision, 'n_drawings': n_draw, 'text_chars': len(text), 'regions': regions}
        if needs_vision:
            pix = page.get_pixmap(dpi=DPI)
            rec['image'] = f'data/idr/pages/p{n:03d}.png'
            pix.save(config.ROOT / rec['image'])
            for r in regions:
                slug = r['id'].replace(' ', '_')
                r['crop'] = f'data/idr/crops/{slug}_p{n:03d}.png'
                page.get_pixmap(dpi=DPI, clip=pymupdf.Rect(r['bbox'])).save(config.ROOT / r['crop'])
        pages.append(rec)
    (OUT / 'pages.json').write_text(json.dumps(pages, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    from collections import Counter
    print('pages', len(pages), dict(Counter(p['kind'] for p in pages)))
    print('needs vision', sum(p['needs_vision'] for p in pages),
          '| figure/table regions', sum(len(p['regions']) for p in pages))
    print('text-kind pages with many drawings:', [p['pdf_page'] for p in pages if p['kind'] == 'text' and p['needs_vision']])


if __name__ == '__main__':
    main()
