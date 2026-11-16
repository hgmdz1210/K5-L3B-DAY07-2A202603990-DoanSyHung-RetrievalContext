"""Tạo benchmark Day 7 từ benchmark Day 6 + 4 câu thiếu bằng chứng, rồi kiểm chứng evidence.

Chạy: .venv/Scripts/python -m src.build_benchmark --day6 <Day6>/benchmark/questions.json

- Giữ nguyên 24 câu và cách chia Day 6 (mỗi câu thứ 4 là test): 18 dev / 6 test có đáp án.
- Thêm 2 dev + 2 test thiếu bằng chứng; mỗi câu ghi cách kiểm tra absence trong corpus.
- `gold` = evidence chính + evidence thay thế (alt) của Day 6; mỗi evidence phải nằm trọn trong một
  chunk đúng nguồn/trang của index B. `filter` (chapter) là phạm vi người dùng nêu trong câu hỏi.
- Gold chỉ dùng để chấm, không bao giờ đưa vào agent hoặc retrieval.
"""
import argparse
import json
import re

from . import config

UNANSWERABLE = [
    {'id': 'u01', 'split': 'dev', 'type': 'unanswerable',
     'question': 'How much yearly rent did Mr. Bingley pay to lease Netherfield Park?',
     'absence_reason': "Corpus không có số tiền thuê Netherfield: regex '\\brent\\b' 0 kết quả; không có 'pounds'/'a year' "
                       "trong vòng 80 ký tự quanh 'Netherfield'. Gần chủ đề: thu nhập 4-5 nghìn bảng/năm của Bingley "
                       "(mồi nhử, không phải tiền thuê)."},
    {'id': 'u02', 'split': 'dev', 'type': 'unanswerable',
     'question': 'Which university did Mr. Bennet attend?',
     'absence_reason': "Chỉ có 3 kết quả 'Oxford|Cambridge|universit': Mr. Collins 'one of the universities' (tr.89), "
                       "Wickham học Cambridge (tr.249), Oxford là địa danh trên đường đi (tr.299). Không nói Mr. Bennet học đâu."},
    {'id': 'u03', 'split': 'test', 'type': 'unanswerable',
     'question': 'What did Jane and Mr. Bingley name their first child?',
     'absence_reason': "Chương 61 (tr.472-473) chỉ nói Bingley và Jane ở Netherfield 12 tháng rồi mua đất gần Derbyshire; "
                       "không có đoạn nào nói về con của họ ('child|baby|infant|christen' không có kết quả liên quan)."},
    {'id': 'u04', 'split': 'test', 'type': 'unanswerable',
     'question': 'On what date did Elizabeth Bennet marry Mr. Darcy?',
     'absence_reason': "Không có ngày cưới: tên tháng/lễ ở chương 55-61 chỉ gồm November (tr.429), April (tr.450), "
                       "'Christmas' là lời mời tới Pemberley (tr.470, mồi nhử), 'May' là động từ (tr.433)."},
]


def hit(chunk, gold):
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page'] and g['evidence'] in chunk['text']
               for g in gold)


QUOTES = str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"})


def exact_quote(evidence, page, chunks):
    """Day 6 có quote gõ nháy thẳng; corpus dùng nháy cong. Thay 1:1 ký tự nên vị trí giữ nguyên,
    trả về đúng đoạn nguyên văn trong corpus để giữ phép so khớp chính xác."""
    target = evidence.translate(QUOTES)
    for c in chunks:
        if c['page'] == page:
            i = c['text'].translate(QUOTES).find(target)
            if i >= 0:
                return c['text'][i:i + len(evidence)]
    return evidence


def convert(q, chunks):
    gold = [{'source_id': q['gold_source_id'], 'page': q['gold_page'], 'chapter': q['gold_chapter'],
             'evidence': exact_quote(q['evidence_quote'], q['gold_page'], chunks)}]
    gold += [{'source_id': q['gold_source_id'], 'page': a['gold_page'], 'chapter': a['gold_chapter'],
              'evidence': exact_quote(a['evidence_quote'], a['gold_page'], chunks)}
             for a in q.get('alt_evidence') or []]
    out = {'id': q['id'], 'split': q['split'], 'type': q['type'], 'answerable': True,
           'expected_behavior': 'search_and_answer', 'question': q['question'],
           'expected_answer': q['expected_answer'], 'gold': gold}
    if q.get('filter'):
        out['filter'] = q['filter']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day6', required=True, help='Đường dẫn benchmark/questions.json của Day 6')
    args = ap.parse_args()
    day6 = json.load(open(args.day6, encoding='utf-8'))
    chunks = json.loads((config.DATA_DIR / 'day6' / 'chunks.json').read_text(encoding='utf-8'))
    questions = [convert(q, chunks) for q in day6['questions']]
    questions += [{**u, 'answerable': False, 'expected_behavior': 'search_and_abstain'} for u in UNANSWERABLE]

    problems = []
    for q in questions:
        if not q['answerable']:
            continue
        for g in q['gold']:
            if not any(hit(c, [g]) for c in chunks):
                problems.append(f"{q['id']}: evidence trang {g['page']} không nằm trọn trong chunk nào")
        if 'filter' in q and not re.search(rf"\bchapter {q['filter']['chapter']}\b", q['question'], re.I):
            problems.append(f"{q['id']}: filter không được nêu trong câu hỏi")
    if problems:
        raise SystemExit('Benchmark lỗi:\n' + '\n'.join(problems))

    out = config.DATA_DIR / 'benchmark' / 'questions.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for split in ('dev', 'test'):
        sel = [q for q in questions if q['split'] == split]
        print(f"{split}: {sum(q['answerable'] for q in sel)} có đáp án, "
              f"{sum(not q['answerable'] for q in sel)} thiếu bằng chứng, "
              f"{sum('filter' in q for q in sel)} có filter")
    print(f'OK -> {out.relative_to(config.ROOT)}; mọi evidence nằm trọn trong một chunk B đúng trang.')


if __name__ == '__main__':
    main()
