"""Đánh giá retrieval + context trên một split (không gọi agent).

Ví dụ:
  .venv/Scripts/python -m src.eval_retrieval --split dev --method dense --top-k 3 --budget 4000
  .venv/Scripts/python -m src.eval_retrieval --split dev --method hybrid --use-filter --name hybrid-filter

Metric (mẫu số = câu có đáp án; câu thiếu evidence chỉ ghi trace):
- Hit@1 / Evidence Hit@3: có chunk chứa evidence hợp lệ (đúng nguồn, trang, chứa nguyên văn một gold)
  ở hạng 1 / trong top-3. Luôn xuất top-3 kể cả khi top-k khác 3.
- Context Evidence Rate: chunk chứa evidence nằm trong context cuối (sau khi pack theo budget).
Gold chỉ dùng sau khi retrieval xong để chấm.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from . import config
from .llm import ChatModel
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever, pack_context


def hit(chunk, gold):
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page'] and g['evidence'] in chunk['text']
               for g in gold)


def build(method, **kw):
    if method == 'bm25':
        return BM25Retriever()
    dense = DenseRetriever()
    if method == 'dense':
        return dense
    if method == 'hybrid':
        return HybridRetriever(dense, BM25Retriever(), pool=kw.get('pool', 20))
    if method == 'rerank':
        return RerankRetriever(dense, BM25Retriever(), ChatModel(), pool=kw.get('pool', 20))
    raise ValueError(method)


def slim(c, gold):
    out = {k: c[k] for k in ('rank', 'id', 'page', 'score')}
    out['chapter'] = c['metadata'].get('chapter')
    if 'ranks' in c:
        out['ranks'] = c['ranks']
    if gold is not None:
        out['hit'] = hit(c, gold)
    return out


def run(split, method, top_k, budget, use_filter, name, pool=20):
    questions = [q for q in json.loads((config.DATA_DIR / 'benchmark' / 'questions.json').read_text(encoding='utf-8'))
                 if q['split'] == split]
    retriever = build(method, pool=pool)
    traces = []
    for q in questions:
        filters = q.get('filter') if use_filter else None
        t0 = time.perf_counter()
        # Retrieval chỉ nhận câu hỏi gốc + filter người dùng; gold KHÔNG được truyền vào.
        results = retriever.search(q['question'], top_k=max(top_k, 3), filters=filters)
        ms = (time.perf_counter() - t0) * 1000
        context, ids = pack_context(results[:top_k], budget)
        gold = q.get('gold') if q['answerable'] else None
        tr = {'query_id': q['id'], 'split': split, 'answerable': q['answerable'], 'question': q['question'],
              'filter': filters, 'method': method, 'top_k': top_k, 'budget_chars': budget,
              'results_top3': [slim(c, gold) for c in results[:3]],
              'context_ids': ids, 'context_chars': len(context), 'elapsed_ms': round(ms, 1)}
        if gold is not None:
            tr['hit1'] = bool(results and hit(results[0], gold))
            tr['hit3'] = any(hit(c, gold) for c in results[:3])
            tr['context_hit'] = any(hit(c, gold) for c in results[:top_k] if c['id'] in ids)
        traces.append(tr)

    ans = [t for t in traces if t['answerable']]
    n = len(ans)
    emb = getattr(retriever, 'embedder', None) or getattr(getattr(retriever, 'dense', None), 'embedder', None)
    summary = {
        'name': name, 'split': split, 'method': method, 'top_k': top_k, 'budget_chars': budget,
        'use_filter': use_filter, 'rrf_pool': pool if method == 'hybrid' else None,
        'n_answerable': n, 'n_unanswerable': len(traces) - n,
        'hit1': f"{sum(t['hit1'] for t in ans)}/{n}", 'hit1_rate': round(sum(t['hit1'] for t in ans) / n, 3),
        'hit3': f"{sum(t['hit3'] for t in ans)}/{n}", 'hit3_rate': round(sum(t['hit3'] for t in ans) / n, 3),
        'context_evidence': f"{sum(t['context_hit'] for t in ans)}/{n}",
        'context_evidence_rate': round(sum(t['context_hit'] for t in ans) / n, 3),
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces)),
        'median_ms': round(statistics.median(t['elapsed_ms'] for t in traces), 1),
        'embedding_tokens': emb.tokens if emb else 0,
        'llm_usage': retriever.llm.usage() if hasattr(retriever, 'llm') else None,
        'cost_usd': round((emb.cost_usd if emb else 0.0)
                          + (retriever.llm.cost_usd if hasattr(retriever, 'llm') else 0.0), 6),
        'misses_hit3': [t['query_id'] for t in ans if not t['hit3']],
    }
    out = config.RESULTS_DIR / 'retrieval' / f'{name}-{split}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--method', required=True, choices=('dense', 'bm25', 'hybrid', 'rerank'))
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--budget', type=int, default=4000, help='context budget, đơn vị ký tự')
    ap.add_argument('--pool', type=int, default=20, help='số candidate mỗi retriever cho RRF')
    ap.add_argument('--use-filter', action='store_true', help='dùng filter chapter người dùng nêu trong câu hỏi')
    ap.add_argument('--name')
    a = ap.parse_args()
    s = run(a.split, a.method, a.top_k, a.budget, a.use_filter, a.name or a.method, a.pool)
    print(json.dumps(s, ensure_ascii=False))


if __name__ == '__main__':
    main()
