"""Chạy agent trên một split và lưu trace + số đo tự động (chưa phải nhãn chất lượng câu trả lời).

Ví dụ:
  .venv/Scripts/python -m src.run_agent --split dev --retriever dense  --name agent-dense
  .venv/Scripts/python -m src.run_agent --split dev --retriever rerank --use-filter --name agent-rerank-filter

Output results/agent/<name>-<split>/:
  traces.json   từng câu: status, answer, citations, events (mọi tool call), context cuối, token, thời gian
  summary.json  status, phân bố số lượt search, Context Evidence Rate, Hit của lượt search đầu, chi phí
  cases.json + reviews_template.json  để chấm tay rồi tổng hợp bằng starter/agent_metrics.py
Gold chỉ dùng sau khi agent chạy xong để tính Context Evidence Rate.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from . import config
from .agent import run_agent
from .day6_store import QueryEmbedder
from .llm import ChatModel
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever


def hit(chunk, gold):
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page'] and g['evidence'] in chunk['text']
               for g in gold)


def build_retriever(kind: str, pool: int):
    if kind == 'bm25':
        return BM25Retriever(), None
    dense = DenseRetriever(embedder=QueryEmbedder())
    if kind == 'dense':
        return dense, None
    if kind == 'hybrid':
        return HybridRetriever(dense, BM25Retriever(), pool=pool), None
    rerank_llm = ChatModel()
    return RerankRetriever(dense, BM25Retriever(), rerank_llm, pool=pool), rerank_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--retriever', required=True, choices=('dense', 'bm25', 'hybrid', 'rerank'))
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--max-searches', type=int, default=4)
    ap.add_argument('--budget', type=int, default=12000, help='context budget cộng dồn, đơn vị ký tự')
    ap.add_argument('--pool', type=int, default=20)
    ap.add_argument('--use-filter', action='store_true')
    ap.add_argument('--only', nargs='*', help='chỉ chạy các query id này (debug trên dev)')
    ap.add_argument('--name', required=True)
    a = ap.parse_args()

    questions = [q for q in json.loads((config.DATA_DIR / 'benchmark' / 'questions.json').read_text(encoding='utf-8'))
                 if q['split'] == a.split and (not a.only or q['id'] in a.only)]
    retriever, rerank_llm = build_retriever(a.retriever, a.pool)
    agent_llm = ChatModel()
    chunks = {c['id']: c for c in BM25Retriever().chunks}
    traces = []
    for q in questions:
        filters = q.get('filter') if a.use_filter else None
        rerank_before = rerank_llm.cost_usd if rerank_llm else 0.0
        # Agent chỉ nhận câu hỏi + filter người dùng nêu; KHÔNG nhận gold/expected_answer.
        r = run_agent(q['question'], agent_llm, retriever, filters=filters, top_k=a.top_k,
                      max_searches=a.max_searches, budget=a.budget)
        r = {'query_id': q['id'], 'split': a.split, 'answerable': q['answerable'], 'question': q['question'], **r}
        r['rerank_cost_usd'] = round((rerank_llm.cost_usd if rerank_llm else 0.0) - rerank_before, 6)
        if q['answerable']:
            searches = [e for e in r['events'] if e['event'] == 'search']
            first = [chunks[x['id']] for x in searches[0]['results'][:3]] if searches else []
            r['first_search_hit1'] = bool(first and hit(first[0], q['gold']))
            r['first_search_hit3'] = any(hit(c, q['gold']) for c in first)
            r['context_hit'] = any(hit(chunks[i], q['gold']) for i in r['context_ids'])
            r['cited_gold'] = any(hit(chunks[i], q['gold']) for i in r['citations'] if i in chunks)
        traces.append(r)
        print(f"{q['id']:4} {r['status']:8} searches={r['n_search']} "
              f"ctx_hit={r.get('context_hit', '-')} {r['elapsed_s']}s | {r['answer'][:110]}")

    ok = [t for t in traces if t['status'] != 'error']
    ans = [t for t in ok if t['answerable']]
    una = [t for t in ok if not t['answerable']]
    frac = lambda n, d: f'{n}/{d}'
    emb = getattr(retriever, 'embedder', None) or getattr(getattr(retriever, 'dense', None), 'embedder', None)
    cost = agent_llm.cost_usd + (rerank_llm.cost_usd if rerank_llm else 0) + (emb.cost_usd if emb else 0)
    summary = {
        'name': a.name, 'split': a.split, 'agent_model': agent_llm.model, 'retriever': a.retriever,
        'top_k': a.top_k, 'max_searches': a.max_searches, 'budget_chars': a.budget, 'use_filter': a.use_filter,
        'rrf_or_rerank_pool': a.pool if a.retriever in ('hybrid', 'rerank') else None,
        'n': len(traces), 'errors': len(traces) - len(ok),
        'status': dict(Counter(t['status'] for t in traces)),
        'tool_use': frac(sum(t['n_search'] > 0 for t in ok), len(ok)),
        'searches_distribution': dict(sorted(Counter(t['n_search'] for t in ok).items())),
        'mean_searches': round(statistics.mean(t['n_search'] for t in ok), 2) if ok else None,
        'first_search_hit1': frac(sum(t['first_search_hit1'] for t in ans), len(ans)),
        'first_search_hit3': frac(sum(t['first_search_hit3'] for t in ans), len(ans)),
        'context_evidence': frac(sum(t['context_hit'] for t in ans), len(ans)),
        'answered_with_gold_citation': frac(sum(t['cited_gold'] for t in ans), len(ans)),
        'answerable_abstained': frac(sum(t['status'] == 'abstain' for t in ans), len(ans)),
        'unanswerable_abstained': frac(sum(t['status'] == 'abstain' for t in una), len(una)),
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in ok)) if ok else None,
        'median_latency_s': round(statistics.median(t['elapsed_s'] for t in traces), 2),
        'agent_llm': agent_llm.usage(), 'rerank_llm': rerank_llm.usage() if rerank_llm else None,
        'embedding_tokens': emb.tokens if emb else 0,
        'total_cost_usd': round(cost, 4),
        'price_assumption_usd_per_1M': {'agent_in': config.AGENT_PRICE_IN_PER_1M,
                                        'agent_out': config.AGENT_PRICE_OUT_PER_1M,
                                        'embedding': config.EMBEDDING_PRICE_PER_1M},
        'note': 'Số đo tự động; chất lượng câu trả lời/citation phải chấm tay trong reviews.json.',
    }
    out = config.RESULTS_DIR / 'agent' / f'{a.name}-{a.split}'
    out.mkdir(parents=True, exist_ok=True)
    dump = lambda name, obj: (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n',
                                                     encoding='utf-8')
    dump('traces.json', traces)
    dump('summary.json', summary)
    dump('cases.json', [{'id': q['id'], 'split': q['split'], 'answerable': q['answerable']} for q in questions])
    dump('reviews_template.json', [
        {'query_id': t['query_id'], 'answer_grade': (0 if t['status'] == 'abstain' else None) if t['answerable'] else None,
         'citation_support': [None] * len(t['citations']),
         **({} if t['answerable'] else {'abstention_correct': None}), 'reason': ''}
        for t in ok])
    print(json.dumps({k: v for k, v in summary.items() if k not in ('agent_llm', 'rerank_llm', 'price_assumption_usd_per_1M')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
