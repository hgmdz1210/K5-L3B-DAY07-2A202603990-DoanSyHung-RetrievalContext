"""Bước 2b: kiểm chứng mô tả vision với text layer gốc của PDF.

Mọi con số model ghi KHÔNG kèm dấu "≈" phải xuất hiện nguyên văn trong text layer của trang
(hoặc là năm / số thứ tự figure/panel). Số không tìm thấy = nghi bịa -> liệt kê để đọc tay.
Chạy: .venv/Scripts/python -m src.idr.verify_vision --model gpt-5.4-2026-03-05
Output: results/idr/vision_check.json
"""
from __future__ import annotations

import argparse
import json
import re

import pymupdf

from .. import config
from .pages import OUT, PDF

NUM = re.compile(r'(?<![\w.])(≈\s*)?[-–+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])(≈\s*)?[-–+]?\d+(?:\.\d+)?%?')


def norm(tok: str) -> str:
    return tok.replace('–', '-').lstrip('+').rstrip('%')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-5.4-2026-03-05')
    a = ap.parse_args()
    doc = pymupdf.open(PDF)
    pages = {p['pdf_page']: p for p in json.loads((OUT / 'pages.json').read_text(encoding='utf-8'))}
    rows, tot_checked, tot_bad = [], 0, 0
    for f in sorted((OUT / 'vision' / a.model).glob('p*.json')):
        rec = json.loads(f.read_text(encoding='utf-8'))
        n = rec['pdf_page']
        layer = doc[n - 1].get_text().replace('–', '-').replace('−', '-')
        layer_nums = {norm(m.group(0)) for m in NUM.finditer(layer)}
        layer_nums |= {x.lstrip('-') for x in layer_nums}
        bad, checked = [], 0
        md = rec['markdown']
        for m in NUM.finditer(md):
            tok = m.group(0)
            before = md[max(0, m.start() - 16):m.start()]
            if '≈' in tok or re.search(r'(≈|~|about|around|roughly|near|approximately|nearly)\W*$', before, re.I):
                continue  # model đã đánh dấu là ước lượng từ hình
            if re.search(r'(19|20)\d\d\s*[–-]\s*$', before):
                continue  # năm viết tắt, vd 2018–19
            if re.search(r'(≈|~|about|around)\W*[-–]?[\d.,]+%?\s*(–|-|to)\s*$', before, re.I):
                continue  # vế sau của khoảng ước lượng, vd ≈40–44%
            v = norm(tok)
            core = v.lstrip('-')
            if re.fullmatch(r'(19|20)\d\d', core) or len(core) == 1 or core in ('10', '0'):
                continue  # năm, nhãn panel/figure, mốc trục đơn giản
            checked += 1
            if v not in layer_nums and core not in layer_nums:
                bad.append(tok)
        tot_checked += checked
        tot_bad += len(bad)
        rows.append({'pdf_page': n, 'kind': pages[n]['kind'], 'numbers_checked': checked,
                     'not_in_text_layer': sorted(set(bad))})
    out = config.RESULTS_DIR / 'idr' / 'vision_check.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {'model': a.model, 'pages': len(rows), 'numbers_checked': tot_checked,
               'numbers_not_in_text_layer': tot_bad,
               'pages_with_issues': [r['pdf_page'] for r in rows if r['not_in_text_layer']]}
    out.write_text(json.dumps({'summary': summary, 'pages': rows}, ensure_ascii=False, indent=1) + '\n',
                   encoding='utf-8')
    print(json.dumps(summary))
    for r in rows:
        if r['not_in_text_layer']:
            print(r['pdf_page'], r['kind'], r['not_in_text_layer'][:15])


if __name__ == '__main__':
    main()
