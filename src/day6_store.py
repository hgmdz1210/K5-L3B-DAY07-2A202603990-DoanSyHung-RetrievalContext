"""Kết nối vector store Day 6 (PostgreSQL + pgvector) và embedding model tương thích.

Index: bảng `chunks` của Day 6, strategy B (paragraph chunks, <=1200 ký tự), OpenAI
text-embedding-3-small 1536 chiều, cosine, exact search (không ANN). Query embedding dùng
đúng model này; mọi lỗi kết nối được ném ra, không có fallback giả.
"""
from __future__ import annotations

import time
from typing import Any

import psycopg
from openai import OpenAI

from . import config

FILTER_COLUMNS = {'chapter': 'chapter', 'page': 'page'}


def to_chunk(row: dict[str, Any]) -> dict[str, Any]:
    """Chuẩn hóa một dòng Day 6 sang schema Day 7 (data/README.md)."""
    meta = dict(row.get('metadata') or {})
    meta.update({'chapter': row['chapter'], 'char_start': row['char_start'], 'char_end': row['char_end'],
                 'day6_strategy': config.DAY6_STRATEGY})
    return {'id': row['chunk_id'], 'text': row['text'], 'source_id': row['document_id'],
            'source': row['source'], 'page': row['page'], 'metadata': meta}


class QueryEmbedder:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=config.require_api_key(), max_retries=3, timeout=60)
        self.model = config.EMBEDDING_MODEL
        self.tokens = 0
        self.calls = 0
        self._cache: dict[str, list[float]] = {}

    @property
    def cost_usd(self) -> float:
        return self.tokens / 1e6 * config.EMBEDDING_PRICE_PER_1M

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        resp = self.client.embeddings.create(model=self.model, input=[text])
        self.calls += 1
        self.tokens += resp.usage.total_tokens
        vec = resp.data[0].embedding
        if len(vec) != config.EMBEDDING_DIM:
            raise RuntimeError(f'Embedding {len(vec)} chiều, index cần {config.EMBEDDING_DIM}')
        self._cache[text] = vec
        return vec


class Day6Store:
    def __init__(self, database_url: str | None = None, strategy: str | None = None) -> None:
        self.database_url = database_url or config.DATABASE_URL
        self.strategy = strategy or config.DAY6_STRATEGY

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, connect_timeout=10)

    def check(self) -> dict[str, Any]:
        with self._connect() as conn:
            dims = conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid='chunks'::regclass "
                                "AND attname='embedding'").fetchone()[0]
            n, pages = conn.execute('SELECT count(*), count(DISTINCT page) FROM chunks WHERE strategy=%s',
                                    (self.strategy,)).fetchone()
            ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
        if dims != config.EMBEDDING_DIM:
            raise RuntimeError(f'Index lưu vector {dims} chiều nhưng embedding model cho {config.EMBEDDING_DIM}')
        return {'strategy': self.strategy, 'chunks': n, 'pages': pages, 'dims': dims, 'pgvector': ext}

    def all_chunks(self) -> list[dict[str, Any]]:
        sql = ('SELECT chunk_id, document_id, source, page, chapter, char_start, char_end, text, metadata '
               'FROM chunks WHERE strategy=%s ORDER BY page, char_start')
        with self._connect() as conn:
            cur = conn.execute(sql, (self.strategy,))
            cols = [d.name for d in cur.description]
            return [to_chunk(dict(zip(cols, r))) for r in cur.fetchall()]

    def dense_search(self, vector: list[float], top_k: int, filters: dict[str, Any] | None = None
                     ) -> list[dict[str, Any]]:
        where, params = ['strategy = %s'], [self.strategy]
        for key, value in (filters or {}).items():
            if key not in FILTER_COLUMNS:
                raise ValueError(f'Filter {key!r} không hỗ trợ; chỉ có {sorted(FILTER_COLUMNS)}')
            where.append(f'{FILTER_COLUMNS[key]} = %s')
            params.append(int(value))
        qv = '[' + ','.join(f'{x:.8g}' for x in vector) + ']'
        sql = (f'SELECT chunk_id, document_id, source, page, chapter, char_start, char_end, text, metadata, '
               f'1 - (embedding <=> %s::vector) AS score FROM chunks WHERE {" AND ".join(where)} '
               f'ORDER BY embedding <=> %s::vector LIMIT %s')
        with self._connect() as conn:
            cur = conn.execute(sql, [qv, *params, qv, top_k])
            cols = [d.name for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return [{**to_chunk(r), 'score': float(r['score'])} for r in rows]
