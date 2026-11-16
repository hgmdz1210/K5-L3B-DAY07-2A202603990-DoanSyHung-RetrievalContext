"""Bước 3: tạo chunk đa phương thức cho IDR 2026, embedding và lưu vào pgvector (bảng idr_chunks).

Chạy: .venv/Scripts/python -m src.idr.build_index [--vision-model gpt-5.4-2026-03-05]
Chunk (schema Day 7: id, text, source_id, source, page, metadata):
- text    : đoạn văn theo trang (gom block <= 1200 ký tự), BỎ block nằm trong vùng figure/bảng và
            header/footer lặp lại. Trang hồ sơ quốc gia không lấy text layer (thứ tự số bị xáo trộn).
- figure / table / diagram : mỗi Figure/Table một chunk = mô tả vision (đã kiểm chứng số với text layer),
            metadata giữ figure_id, bbox vùng trang, ảnh crop.
- profile : mỗi trang hồ sơ quốc gia/vùng một chunk = mô tả vision có cấu trúc.
Mỗi chunk có header ngữ cảnh (tài liệu, phần, trang in, figure) để retrieval biết vị trí.
`page` = số trang PDF (1-280); `metadata.printed_page` = số trang in trên giấy.
"""
from __future__ import annotations

import argparse
import json
import re

import psycopg
import pymupdf

from .. import config
from ..day6_store import QueryEmbedder
from .pages import OUT, PDF

SOURCE_ID = 'idr-2026'
SOURCE = 'World Bank, International Debt Report 2026 (CC BY 3.0 IGO), 9781464823558.pdf'
SECTION_NAME = {
    'front_matter': 'Front matter (Foreword, Key Takeaways, Abbreviations)',
    'chapter_1': 'Chapter 1. Analysis of External Debt Stocks and Debt Flows as of End-2025',
    'chapter_2': 'Chapter 2. Debt Dynamics: Policy Implications',
    'chapter_3': 'Chapter 3. Advancing Debt Transparency',
    'part2_profiles': 'Part 2. Aggregate and Country Profiles',
    'appendix': 'Appendix (Data Sources, Country Groups, Glossary)',
}
DIAGRAMS = {'Figure 3B.1', 'Figure 3.2'}  # sơ đồ/timeline, không phải biểu đồ số liệu
BOILER = re.compile(r'^(INTERNATIONAL DEBT REPORT 2026|\d{1,3}|[0-9]\. [A-Z][A-Za-z ,:\-–]+|Glossary|PART \d)$')


def header(p: dict, extra: str = '') -> str:
    return (f"[International Debt Report 2026 | {SECTION_NAME[p['section']]} | printed page {p['printed_page']} "
            f"(PDF page {p['pdf_page']}){' | ' + extra if extra else ''}]\n")


def inside(block, regions) -> bool:
    x0, y0, x1, y1 = block[:4]
    cy = (y0 + y1) / 2
    return any(r['bbox'][1] <= cy <= r['bbox'][3] for r in regions)


def text_chunks(page: pymupdf.Page, p: dict, max_chars: int = 1200) -> list[dict]:
    blocks = [b for b in page.get_text('blocks') if b[6] == 0]
    paras = []
    for b in sorted(blocks, key=lambda b: (b[1], b[0])):
        t = re.sub(r'\s*\n\s*', ' ', b[4]).strip()
        if not t or BOILER.match(t) or inside(b, p['regions']):
            continue
        paras.append(t)
    chunks, cur = [], ''
    for t in paras:
        if cur and len(cur) + len(t) + 1 > max_chars:
            chunks.append(cur)
            cur = ''
        cur = f'{cur}\n{t}' if cur else t
    if cur:
        chunks.append(cur)
    out = []
    for k, body in enumerate(chunks):
        if len(body) < 40:
            continue
        out.append({'id': f"idr:p{p['pdf_page']:03d}:text:{k}", 'text': header(p) + body,
                    'source_id': SOURCE_ID, 'source': SOURCE, 'page': p['pdf_page'],
                    'metadata': {'kind': 'text', 'section': p['section'], 'printed_page': p['printed_page']}})
    return out


def split_vision(md: str) -> list[tuple[str | None, str]]:
    """Tách mô tả vision theo heading '## Figure x.y' / '## Table x.y'."""
    parts = re.split(r'(?m)^(?=#{1,3} *(?:\*\*)?(?:Figure|Table) [A-Z]?[0-9]+[A-Z]?\.[0-9]+(?:\.[0-9]+)?)', md)
    out = []
    for part in parts:
        m = re.match(r'#{1,3} *(?:\*\*)?((?:Figure|Table) [A-Z]?[0-9]+[A-Z]?\.[0-9]+(?:\.[0-9]+)?)', part)
        if part.strip():
            out.append((m.group(1) if m else None, part.strip()))
    return out


def visual_chunks(p: dict, md: str) -> list[dict]:
    out = []
    if p['kind'] in ('profile', 'profile_howto'):
        name = md.split('\n', 1)[0].lstrip('# ').split('—')[0].strip()
        kind = 'profile' if p['kind'] == 'profile' else 'diagram'
        out.append({'id': f"idr:p{p['pdf_page']:03d}:{kind}", 'text': header(p, f'{kind}: {name}') + md,
                    'source_id': SOURCE_ID, 'source': SOURCE, 'page': p['pdf_page'],
                    'metadata': {'kind': kind, 'section': p['section'], 'printed_page': p['printed_page'],
                                 'profile_name': name, 'bbox': 'full page', 'image': p['image']}})
        return out
    regions = {r['id']: r for r in p['regions']}
    for fid, body in split_vision(md):
        if fid is None or fid not in regions:
            if out:  # đoạn thừa (vd. phần mở đầu) -> nối vào chunk trước
                out[-1]['text'] += '\n' + body
            continue
        r = regions[fid]
        kind = 'table' if fid.startswith('Table') else ('diagram' if fid in DIAGRAMS else 'figure')
        out.append({'id': f"idr:p{p['pdf_page']:03d}:{fid.replace(' ', '_')}",
                    'text': header(p, f"{kind}: {r['title']}") + body,
                    'source_id': SOURCE_ID, 'source': SOURCE, 'page': p['pdf_page'],
                    'metadata': {'kind': kind, 'section': p['section'], 'printed_page': p['printed_page'],
                                 'figure_id': fid, 'bbox': r['bbox'], 'crop': r['crop']}})
    missing = set(regions) - {c['metadata'].get('figure_id') for c in out}
    if missing:
        raise RuntimeError(f"PDF page {p['pdf_page']}: vision thiếu mô tả cho {sorted(missing)}")
    return out


def store(chunks: list[dict], vectors: list[list[float]]) -> None:
    with psycopg.connect(config.DATABASE_URL) as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        conn.execute(f'''CREATE TABLE IF NOT EXISTS idr_chunks (
            chunk_id text PRIMARY KEY, document_id text NOT NULL, source text NOT NULL, page integer NOT NULL,
            kind text NOT NULL, text text NOT NULL, metadata jsonb NOT NULL,
            embedding vector({config.EMBEDDING_DIM}) NOT NULL)''')
        conn.execute('TRUNCATE idr_chunks')
        with conn.cursor() as cur:
            cur.executemany(
                'INSERT INTO idr_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)',
                [(c['id'], c['source_id'], c['source'], c['page'], c['metadata']['kind'], c['text'],
                  json.dumps(c['metadata'], ensure_ascii=False), '[' + ','.join(f'{x:.8g}' for x in v) + ']')
                 for c, v in zip(chunks, vectors)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--vision-model', default='gpt-5.4-2026-03-05')
    a = ap.parse_args()
    doc = pymupdf.open(PDF)
    pages = json.loads((OUT / 'pages.json').read_text(encoding='utf-8'))
    vdir = OUT / 'vision' / a.vision_model
    chunks, skipped = [], []
    for p in pages:
        if p['kind'] == 'blank_or_cover':
            skipped.append(p['pdf_page'])
            continue
        if p['kind'] not in ('profile',):
            chunks += text_chunks(doc[p['pdf_page'] - 1], p)
        if p['needs_vision']:
            vf = vdir / f"p{p['pdf_page']:03d}.json"
            if not vf.exists():
                raise RuntimeError(f'Thiếu mô tả vision cho PDF page {p["pdf_page"]}; chạy src.idr.vision trước')
            if p['regions'] or p['kind'] in ('profile', 'profile_howto'):
                chunks += visual_chunks(p, json.loads(vf.read_text(encoding='utf-8'))['markdown'])
    ids = [c['id'] for c in chunks]
    assert len(ids) == len(set(ids)), 'chunk ID trùng'
    emb = QueryEmbedder()
    vectors = []
    for i in range(0, len(chunks), 96):
        batch = [c['text'] for c in chunks[i:i + 96]]
        resp = emb.client.embeddings.create(model=emb.model, input=batch)
        emb.tokens += resp.usage.total_tokens
        vectors += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
    store(chunks, vectors)
    (OUT / 'chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    from collections import Counter
    stats = {'chunks': len(chunks), 'by_kind': dict(Counter(c['metadata']['kind'] for c in chunks)),
             'pages_indexed': len({c['page'] for c in chunks}), 'pages_skipped_blank_or_cover': skipped,
             'embedding_model': emb.model, 'embedding_tokens': emb.tokens, 'embedding_cost_usd': round(emb.cost_usd, 4),
             'vision_model': a.vision_model,
             'max_chunk_chars': max(len(c['text']) for c in chunks)}
    (OUT / 'index_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
