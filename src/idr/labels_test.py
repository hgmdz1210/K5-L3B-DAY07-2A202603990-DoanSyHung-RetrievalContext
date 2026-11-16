"""Nhãn chấm tay ĐẠT/KHÔNG ĐẠT cho toàn bộ test nâng cao (ADVANCED.md §3), kèm lý do.

Đã đọc từng câu trả lời, mở từng chunk được cite và đối chiếu với trang gốc (text layer / ảnh crop).
Chấm chặt: đúng một phần, sai citation, từ chối câu có đáp án, hay dùng giá trị ước lượng thay cho số
báo cáo nêu rõ đều KHÔNG ĐẠT. Lỗi hệ thống tính không đạt (không có lỗi nào).
Chạy: .venv/Scripts/python -m src.idr.labels_test -> results/idr/<cfg>-test/labels.json + labels_summary.json
"""
import json
from collections import Counter

from .. import config

OK = True
LABELS = {
    'agent-rerank-verify': {
        't01': (OK, 'US$386.4 billion = Table 1.1 hàng IMF, cột 2021 (PDF p.28) — cite đúng bảng.'),
        't02': (OK, '1,242.2 -> 2,364.4, chênh 1,122.2 tỷ, đúng phép tính; cite Table 1.1.'),
        't03': (OK, '2023, -11.0 tỷ = giá trị nhỏ nhất hàng Short-term Table 1.2 (p.63).'),
        't04': (OK, '2024, 18.3 tỷ = max hàng World Bank Table 1.2.'),
        't05': (OK, '1,493.7 / 9,445.0 = 15.8%, đúng ô Table 1.1 năm 2025.'),
        't06': (False, 'Từ chối nhầm: bản nháp tự mâu thuẫn ("larger by 0.6" rồi "smaller by 6.4"); verifier bác bỏ -> abstain.'),
        't07': (OK, 'Tier 3 = IDR, "Actionable knowledge" (Figure 3B.1, p.132); trích thêm p.133 có nguyên văn.'),
        't08': (OK, 'China 144 tỷ (Japan 112, Other 244) — khớp ảnh sơ đồ All LMICs (text layer xếp sai thứ tự nhãn).'),
        't09': (OK, 'Private creditors 1,263 triệu; Russian Federation 1,392 triệu — khớp legend sơ đồ Viet Nam (p.260).'),
        't10': (OK, 'Belize, 1,609, +2.6% from 2024 — chunk text p.136 (trang hướng dẫn) chứa nguyên văn ví dụ.'),
        't11': (OK, 'China 8,766 > ADB 7,379 triệu, chênh 1,387; cite hồ sơ Sri Lanka.'),
        't12': (OK, 'March cao nhất 2025 (~56%); late reporters 2020 ~13% > 2025 ~6% — khớp ảnh Figure 3.2.'),
        't13': (OK, 'Türkiye, private nonguaranteed ~45 tỷ là thành phần lớn nhất — khớp ảnh Figure 1.6.'),
        't14': (OK, '22 downgrades, 15 upgrades, cao nhất 2025 (6) — nhãn in trên Figure 2.10.'),
        't15': (OK, 'South Asia lớn nhất 2025 (~205); SSA giảm ~62 -> ~51 — đúng xu hướng, giá trị ước lượng hợp lý.'),
        't16': (OK, '4,875 báo cáo, 3,762 khớp — nhãn Figure 3.3 và text p.112.'),
        't17': (False, 'Từ chối nhầm: bản nháp mở đầu "Kenya had the larger" mâu thuẫn với phép tính; verifier bác bỏ -> abstain.'),
        't18': (OK, '15.2 tỷ (Table 1.2, IMF 2020) và 214.6 tỷ (Table 1.1, IMF 2020); cite cả hai bảng.'),
        't19': (False, 'Chấm chặt: dùng long-term ≈226 tỷ (ước lượng từ Figure B1.4.1) thay cho 227 tỷ báo cáo nêu (p.38) -> 90.3% '
                       'thay vì 90.7%. Tổng 250,257 triệu (hồ sơ Ukraine p.257) đúng.'),
        't20': (OK, '131,777 triệu = 131.777 tỷ / 3,313 tỷ = 3.98% ≈ 4.0%; đổi đơn vị đúng, cite 2 hồ sơ.'),
        't21': (OK, 'Angola, Argentina, Colombia, Nigeria, Senegal; 133-198% — Box 1.2 p.32.'),
        't22': (OK, '8.2% (p.65: lãi suất từ chủ nợ tư nhân; Key Takeaways gọi là average rate), gap 3.4 -> 5.5 điểm % (p.20).'),
        't23': (OK, 'Trả nợ từ doanh thu tài sản Nga bị phong tỏa qua Ukraine Loan Cooperation Mechanism; vẫn là nợ do nghĩa vụ hợp đồng — p.39.'),
        't24': (OK, '1.5 tỷ, 2 tranche 750 triệu: 7.875% đến 2033, 8.80% đến 2038 (p.73); nêu thêm yield 8.2/9.2% (p.52) đúng.'),
        'tu1': (OK, 'Từ chối, không đưa số 2026.'), 'tu2': (OK, 'Từ chối, không đưa số cho Germany.'),
        'tu3': (OK, 'Từ chối, không bịa lãi suất ERA.'), 'tu4': (OK, 'Từ chối, không đưa số 2010.'),
        'tu5': (OK, 'Từ chối; không lấy "Other bilateral" thay cho China.'), 'tu6': (OK, 'Từ chối, không bịa tăng trưởng GDP.'),
    },
    'agent-dense': {
        't01': (False, 'Từ chối nhầm: 4 lượt dense không đưa Table 1.1 vào context.'),
        't02': (OK, 'Đúng: 1,122.2 tỷ, cite Table 1.1.'), 't03': (OK, 'Đúng: 2023, -11.0 tỷ.'), 't04': (OK, 'Đúng: 2024, 18.3 tỷ.'),
        't05': (False, 'Dùng số làm tròn trong text (1.5 / 9.4 nghìn tỷ) -> 15.9%, không phải Table 1.1 (15.8%); cite sai nguồn.'),
        't06': (False, 'Tự mâu thuẫn: "private ... larger" rồi "actually smaller by 6.4".'),
        't07': (OK, 'Đúng: IDR, Actionable knowledge.'), 't08': (OK, 'Đúng: China 144 tỷ.'),
        't09': (OK, 'Đúng: 1,263 triệu; Russian Federation 1,392 triệu.'), 't10': (OK, 'Đúng: Belize, 1,609, +2.6%.'),
        't11': (OK, 'Đúng: China 8,766 > ADB 7,379.'), 't12': (OK, 'Đúng, thêm text p.110 (13% và 5.8%).'),
        't13': (OK, 'Đúng: Türkiye, PNG.'), 't14': (OK, 'Đúng: 22, 15, 2025 (6).'), 't15': (OK, 'Đúng xu hướng và vùng.'),
        't16': (OK, 'Đúng: 4,875 và 3,762.'),
        't17': (False, 'Kết luận "Kenya had the larger" sai (Sri Lanka lớn hơn), dù phép tính đúng.'),
        't18': (False, 'Sai: IMF 2020 = 5.7 tỷ (đọc nhầm), không tìm được 214.6 tỷ Table 1.1.'),
        't19': (False, 'Sai mẫu số: lấy 227 tỷ làm tổng nợ -> 99.6% (đúng là 227/250.3 = 90.7%).'),
        't20': (OK, 'Đúng: 3.98% ≈ 4.0%, đổi đơn vị đúng.'), 't21': (OK, 'Đúng.'), 't22': (OK, 'Đúng: 8.2%, 3.4 -> 5.5.'),
        't23': (OK, 'Đúng.'), 't24': (OK, 'Đúng.'),
        'tu1': (OK, 'Từ chối.'), 'tu2': (OK, 'Từ chối.'), 'tu3': (OK, 'Từ chối.'), 'tu4': (OK, 'Từ chối.'),
        'tu5': (OK, 'Từ chối.'), 'tu6': (OK, 'Từ chối.'),
    },
}


def main():
    bench = {q['id']: q for q in json.loads((config.DATA_DIR / 'idr' / 'benchmark.json').read_text(encoding='utf-8'))}
    for cfg, labels in LABELS.items():
        d = config.RESULTS_DIR / 'idr' / f'{cfg}-test'
        traces = {t['query_id']: t for t in json.loads((d / 'traces.json').read_text(encoding='utf-8'))}
        test_ids = [q for q in bench if bench[q]['split'] == 'test']
        assert set(test_ids) == set(labels) == set(traces), 'phải chấm đủ mọi câu test'
        rows = [{'query_id': i, 'group': bench[i]['group'], 'status': traces[i]['status'], 'pass': labels[i][0],
                 'reason': labels[i][1], 'citations': traces[i].get('citations', [])} for i in test_ids]
        groups = {}
        for g in ['table', 'image_diagram', 'chart', 'multihop', 'text', 'unanswerable']:
            rs = [r for r in rows if r['group'] == g]
            groups[g] = f"{sum(r['pass'] for r in rs)}/{len(rs)}"
        n = sum(r['pass'] for r in rows)
        summ = {'config': cfg, 'pass': f'{n}/{len(rows)}', 'pass_rate': n / len(rows), 'meets_90': n / len(rows) >= 0.9,
                'by_group': groups, 'system_errors': sum(r['status'] == 'error' for r in rows),
                'status': dict(Counter(r['status'] for r in rows))}
        (d / 'labels.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        (d / 'labels_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
