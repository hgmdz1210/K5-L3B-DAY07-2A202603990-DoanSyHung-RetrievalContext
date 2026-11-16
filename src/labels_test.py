"""Nhãn chấm tay cho test (baseline agent-dense và cấu hình chọn agent-rerank-filter).

Nhãn đề xuất sau khi đọc câu trả lời, mở từng chunk được cite và đối chiếu với benchmark; học viên đã
chấm lại bằng tay từng câu và giữ nguyên (rủi ro: chỉ một người chấm). Chạy:
  .venv/Scripts/python -m src.labels_test
-> ghi results/agent/<cfg>-test/reviews.json và summary_labels.json (qua starter/agent_metrics.py).

citation_support: một nhãn cho mỗi citation, theo thứ tự trong trace; True khi chunk nằm trong context
VÀ nội dung chunk thực sự hỗ trợ câu (vế) trả lời gắn citation đó, đúng với điều câu hỏi hỏi.
"""
import json
import subprocess
import sys

from . import config

P = 'pride-and-prejudice-pg1342:'
LABELS = {
    'agent-rerank-filter': {
        'q04': (2, {f'{P}p36:ch7:0-322': True},
                'Đúng và đủ: estate entailed, in default of heirs male, on a distant relation; tr.36 nêu nguyên văn.'),
        'q08': (2, {f'{P}p101:ch16:1176-1600': True},
                'Đúng cả hai vế: next presentation of the best living; when the living fell it was given elsewhere.'),
        'q12': (2, {f'{P}p232:ch33:0-1120': True},
                'Đúng nguyên văn: saved a friend from the inconveniences of a most imprudent marriage (tr.232).'),
        'q16': (1, {f'{P}p289:ch41:0-895': False, f'{P}p277:ch39:654-1686': True},
                'Đúng một phần. Vế 2 đúng (Mrs. Bennet never despaired, tr.277). Vế 1 sai ngữ cảnh: "Let her go" '
                '(tr.289, ch.41) là phản hồi về việc Lydia đi Brighton cùng Mrs. Forster, không phải "Brighton scheme" '
                'cả nhà đi ở ch.39; ý đúng (không nhượng bộ nhưng trả lời mập mờ) có trong tr.277 mà agent bỏ qua.'),
        'q20': (2, {f'{P}p439:ch56:0-712': True, f'{P}p439:ch56:714-1561': True},
                'Từ chối hứa ("I will make no promise of the kind", "not to be intimidated") — khớp evidence thay thế '
                'tr.439 đã ghi trong benchmark. Không nêu ý "act ... constitute my happiness" (tr.440); học viên xem lại '
                'nếu muốn chấm chặt thành 1.'),
        'q24': (2, {f'{P}p476:ch61:0-1076': True},
                'Đúng: by bringing her into Derbyshire, had been the means of uniting them (cuối chunk tr.476).'),
        'u03': (None, {}, 'Từ chối sau 3 lượt search; không đưa tên hay khẳng định nào về con của Jane/Bingley.'),
        'u04': (None, {}, 'Từ chối sau 3 lượt search; không bịa ngày cưới (không dùng "Christmas" tr.470).'),
    },
    'agent-dense': {
        'q04': (2, {f'{P}p36:ch7:0-322': True}, 'Đúng và đủ, trích nguyên văn tr.36.'),
        'q08': (2, {f'{P}p101:ch16:1176-1600': True},
                'Đúng cả hai vế; lượt search 2 dùng query viết lại theo nguyên văn, chunk đã có từ lượt 1.'),
        'q12': (0, {}, 'Từ chối nhầm: 4 lượt search dense đều không đưa chunk tr.232 vào top-3 (dense xếp hạng >3).'),
        'q16': (2, {f'{P}p277:ch39:654-1686': True},
                'Đúng và đủ cả hai vế: not the smallest intention of yielding, vague and equivocal; never despaired.'),
        'q20': (2, {f'{P}p439:ch56:0-712': True, f'{P}p439:ch56:714-1561': True},
                'Như cấu hình chọn: từ chối hứa, khớp evidence thay thế tr.439; thiếu ý tr.440.'),
        'q24': (2, {f'{P}p476:ch61:0-1076': True}, 'Đúng: brought her into Derbyshire, means of uniting them.'),
        'u03': (None, {}, 'Từ chối sau 3 lượt search, không bịa.'),
        'u04': (None, {}, 'Từ chối sau 3 lượt search, không bịa.'),
    },
}


def main():
    for cfg, labels in LABELS.items():
        d = config.RESULTS_DIR / 'agent' / f'{cfg}-test'
        traces = json.loads((d / 'traces.json').read_text(encoding='utf-8'))
        reviews = []
        for t in traces:
            grade, support, reason = labels[t['query_id']]
            missing = [c for c in t['citations'] if c not in support]
            if missing:
                raise SystemExit(f'{cfg} {t["query_id"]}: chưa chấm citation {missing}')
            r = {'query_id': t['query_id'], 'citation_support': [support[c] for c in t['citations']],
                 'reason': reason}
            if t['answerable']:
                r['answer_grade'] = grade
            else:
                r['abstention_correct'] = t['status'] == 'abstain'
            reviews.append(r)
        (d / 'reviews.json').write_text(json.dumps(reviews, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        subprocess.run([sys.executable, str(config.ROOT / 'starter' / 'agent_metrics.py'),
                        '--cases', str(d / 'cases.json'), '--traces', str(d / 'traces.json'),
                        '--reviews', str(d / 'reviews.json'), '--split', 'test',
                        '--output', str(d / 'summary_labels.json')], check=True, stdout=subprocess.DEVNULL)
        print(cfg, (d / 'summary_labels.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
