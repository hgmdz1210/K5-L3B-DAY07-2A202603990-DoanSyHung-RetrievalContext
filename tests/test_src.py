"""Test offline cho code bài làm (src/): không gọi API/DB, chỉ kiểm tra logic.

Mock chỉ dùng để test code; mọi kết quả benchmark trong báo cáo đến từ model và vector store thật.
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import run_agent  # noqa: E402
from src.retrieval import BM25Retriever, HybridRetriever, pack_context  # noqa: E402

CHUNKS = [
    {'id': 'c1', 'text': 'Netherfield Park is let at last to a young man of large fortune.', 'source_id': 'd',
     'source': 's', 'page': 1, 'metadata': {'chapter': 1}},
    {'id': 'c2', 'text': 'Mr. Collins proposed to Elizabeth and she refused him.', 'source_id': 'd',
     'source': 's', 'page': 2, 'metadata': {'chapter': 19}},
]


def tool_call(name, args):
    call = SimpleNamespace(id='x', function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    msg = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


class FakeLLM:
    """Trả lần lượt các action đã định; đếm token như ChatModel."""
    def __init__(self, actions):
        self.actions, self.prompt_tokens, self.completion_tokens = list(actions), 0, 0

    def chat(self, messages, **kw):
        if kw.get('response_format'):  # verifier
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.actions.pop(0)))])
        return tool_call(*self.actions.pop(0))


class TestRetrieval(unittest.TestCase):
    def test_bm25_ranks_keyword_match_first(self):
        r = BM25Retriever(chunks=CHUNKS).search('Who proposed to Elizabeth?', top_k=2)
        self.assertEqual(r[0]['id'], 'c2')
        self.assertEqual([x['rank'] for x in r], list(range(1, len(r) + 1)))

    def test_bm25_filter(self):
        r = BM25Retriever(chunks=CHUNKS).search('Elizabeth Netherfield', top_k=5, filters={'chapter': 1})
        self.assertEqual({x['id'] for x in r}, {'c1'})

    def test_rrf_fuses_two_rankings(self):
        bm = BM25Retriever(chunks=CHUNKS)
        other = SimpleNamespace(name='dense', search=lambda q, top_k, filters=None: [
            {**CHUNKS[0], 'score': 0.9, 'rank': 1}, {**CHUNKS[1], 'score': 0.1, 'rank': 2}][:top_k])
        r = HybridRetriever(other, bm, pool=2).search('Collins proposed', top_k=2)
        self.assertEqual({x['id'] for x in r}, {'c1', 'c2'})
        self.assertTrue(all('ranks' in x for x in r))

    def test_pack_context_budget_and_dedup(self):
        ctx, ids = pack_context([CHUNKS[0], CHUNKS[0], CHUNKS[1]], budget=120)
        self.assertEqual(ids, ['c1'])
        self.assertIn('page 1', ctx)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.ret = BM25Retriever(chunks=CHUNKS)

    def test_answer_with_valid_citation(self):
        llm = FakeLLM([('search', {'query': 'Collins proposed'}),
                       ('final_answer', {'answer': 'She refused him [c2].', 'citations': ['c2']})])
        r = run_agent('Did Elizabeth accept Collins?', llm, self.ret, top_k=2)
        self.assertEqual(r['status'], 'answered')
        self.assertEqual(r['n_search'], 1)

    def test_invalid_citation_twice_abstains(self):
        llm = FakeLLM([('search', {'query': 'Collins'}),
                       ('final_answer', {'answer': 'x [zz]', 'citations': ['zz']}),
                       ('final_answer', {'answer': 'x [zz]', 'citations': ['zz']})])
        r = run_agent('q', llm, self.ret, top_k=2)
        self.assertEqual(r['status'], 'abstain')
        self.assertIn('abstain_invalid_citations', [e['event'] for e in r['events']])

    def test_search_limit_forces_decision(self):
        llm = FakeLLM([('search', {'query': 'a'}), ('search', {'query': 'b'}), ('abstain', {'reason': 'none'})])
        r = run_agent('q', llm, self.ret, top_k=1, max_searches=1)
        self.assertEqual(r['n_search'], 1)
        self.assertIn('search_rejected', [e['event'] for e in r['events']])
        self.assertEqual(r['status'], 'abstain')

    def test_verifier_rejects_unsupported_answer(self):
        llm = FakeLLM([('search', {'query': 'Netherfield'}),
                       ('final_answer', {'answer': 'In 2015 [c1].', 'citations': ['c1']})])
        verifier = FakeLLM(['{"supported": true, "answers_question": false, "reason": "wrong year"}'])
        r = run_agent('When?', llm, self.ret, top_k=2, verifier=verifier)
        self.assertEqual(r['status'], 'abstain')
        self.assertIn('verifier_rejected', [e['event'] for e in r['events']])

    def test_errors_are_reported_not_faked(self):
        class Boom:
            name = 'boom'
            def search(self, *a, **k):
                raise RuntimeError('db down')
        r = run_agent('q', FakeLLM([('search', {'query': 'x'})]), Boom(), top_k=1)
        self.assertEqual(r['status'], 'error')


class TestBenchmarks(unittest.TestCase):
    def test_basic_benchmark_counts(self):
        q = json.loads((Path(__file__).resolve().parents[1] / 'data/benchmark/questions.json').read_text(encoding='utf-8'))
        for split, need_ans, need_una in (('dev', 15, 2), ('test', 5, 2)):
            s = [x for x in q if x['split'] == split]
            self.assertGreaterEqual(sum(x['answerable'] for x in s), need_ans)
            self.assertGreaterEqual(sum(not x['answerable'] for x in s), need_una)

    def test_advanced_benchmark_coverage(self):
        q = json.loads((Path(__file__).resolve().parents[1] / 'data/idr/benchmark.json').read_text(encoding='utf-8'))
        test = [x for x in q if x['split'] == 'test']
        need = {'table': 6, 'image_diagram': 5, 'chart': 5, 'multihop': 4, 'text': 4, 'unanswerable': 6}
        for g, n in need.items():
            self.assertGreaterEqual(sum(x['group'] == g for x in test), n, g)
        self.assertGreaterEqual(len(test), 30)
        self.assertFalse({x['question'] for x in test} & {x['question'] for x in q if x['split'] == 'dev'})


if __name__ == '__main__':
    unittest.main()
