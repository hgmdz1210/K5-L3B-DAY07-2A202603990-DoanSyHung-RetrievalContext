"""Dense search trên bảng idr_chunks (cùng interface với Day6Store để dùng lại các retriever)."""
from __future__ import annotations

import json
from typing import Any

import psycopg

from .. import config

FILTER_COLUMNS = {'kind': 'kind', 'page': 'page'}


def load_chunks() -> list[dict[str, Any]]:
    return json.loads((config.DATA_DIR / 'idr' / 'chunks.json').read_text(encoding='utf-8'))


class IdrStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or config.DATABASE_URL

    def dense_search(self, vector: list[float], top_k: int, filters: dict[str, Any] | None = None
                     ) -> list[dict[str, Any]]:
        where, params = ['TRUE'], []
        for key, value in (filters or {}).items():
            if key not in FILTER_COLUMNS:
                raise ValueError(f'Filter {key!r} không hỗ trợ; chỉ có {sorted(FILTER_COLUMNS)}')
            where.append(f'{FILTER_COLUMNS[key]} = %s')
            params.append(value)
        qv = '[' + ','.join(f'{x:.8g}' for x in vector) + ']'
        sql = (f'SELECT chunk_id, document_id, source, page, text, metadata, 1 - (embedding <=> %s::vector) AS score '
               f'FROM idr_chunks WHERE {" AND ".join(where)} ORDER BY embedding <=> %s::vector LIMIT %s')
        with psycopg.connect(self.database_url, connect_timeout=10) as conn:
            rows = conn.execute(sql, [qv, *params, qv, top_k]).fetchall()
        return [{'id': r[0], 'source_id': r[1], 'source': r[2], 'page': r[3], 'text': r[4], 'metadata': r[5],
                 'score': float(r[6])} for r in rows]
