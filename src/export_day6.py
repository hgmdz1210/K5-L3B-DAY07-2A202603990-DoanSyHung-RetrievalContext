"""Kiểm tra kết nối Day 6 và xuất chunk sang data/day6/chunks.json (schema Day 7).

Chạy: .venv/Scripts/python -m src.export_day6
File JSON dùng cho lexical retrieval (BM25/TF-IDF) và kiểm chứng evidence; dense retrieval vẫn
truy vấn trực tiếp pgvector. Corpus Pride and Prejudice (Project Gutenberg #1342) thuộc phạm vi
công cộng tại Mỹ nên được phép chia sẻ.
"""
import json

from . import config
from .day6_store import Day6Store, QueryEmbedder


def main() -> None:
    store = Day6Store()
    info = store.check()
    print('Vector store:', json.dumps(info))
    vec = QueryEmbedder().embed('Who took Netherfield Park?')
    top = store.dense_search(vec, top_k=3)
    print('Smoke test dense:', [(c['id'], c['page'], round(c['score'], 3)) for c in top])
    chunks = store.all_chunks()
    out = config.DATA_DIR / 'day6' / 'chunks.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Đã xuất {len(chunks)} chunk -> {out.relative_to(config.ROOT)}')


if __name__ == '__main__':
    main()
