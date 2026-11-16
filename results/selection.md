# Chốt cấu hình trước khi chạy test

Ngày chốt: 2026-11-16 — **ghi trước khi chạy bất kỳ cấu hình nào trên test.** Mọi số liệu dưới đây là dev
(18 câu có đáp án + 2 câu thiếu bằng chứng).

## Tiêu chí (theo thứ tự ưu tiên, chỉ dùng dev)

1. Context Evidence Rate và số câu citation trỏ đúng chunk evidence — cao nhất.
2. Từ chối nhầm câu có đáp án ít nhất, với điều kiện từ chối đúng 2/2 câu thiếu bằng chứng.
3. Hit@1/Hit@3 của lượt search đầu (query gốc).
4. Thời gian và chi phí: chấp nhận chi phí cao hơn nếu (1) tăng ít nhất 2 câu; ngang nhau thì chọn rẻ hơn.

## Kết quả dev

| Cấu hình | Lượt đầu Hit@1/Hit@3 | Context Evidence | Citation đúng evidence | Abstain đúng | Từ chối nhầm | Trung vị | Chi phí |
|---|---|---|---|---|---|---|---|
| agent-dense (baseline) | 8/18 · 12/18 | 12/18 | 12/18 | 2/2 | 2/18 | 3.9 s | ~$0.059 |
| agent-rerank | 14/18 · 16/18 | 16/18 | 16/18 | 2/2 | 0/18 | 5.0 s | ~$0.190 |
| agent-rerank-filter | 15/18 · 17/18 | 17/18 | 17/18 | 2/2 | 0/18 | 4.9 s | ~$0.159 |

## Quyết định

- **Cấu hình chọn:** `agent-rerank-filter` — agent `gpt-5.4-mini-2026-03-17`, tool `search` dùng
  dense (pgvector Day 6, top-20) ∪ BM25 (top-20) → LLM rerank → top-3; filter chapter khi người dùng nêu
  trong câu hỏi; tối đa 4 lượt search; context budget 12000 ký tự cộng dồn; citation bắt buộc.
- **Baseline so sánh:** `agent-dense` — cùng agent, tool `search` chỉ dense top-3, không filter.
- Lệnh chạy test (mỗi cấu hình đúng một lần, không chỉnh sau khi xem):

```powershell
.venv\Scripts\python -m src.run_agent --split test --retriever rerank --use-filter --name agent-rerank-filter
.venv\Scripts\python -m src.run_agent --split test --retriever dense --name agent-dense
.venv\Scripts\python -m src.eval_retrieval --split test --method rerank --use-filter --name rerank-filter
.venv\Scripts\python -m src.eval_retrieval --split test --method dense --name dense
```

Lưu ý: 6 câu test có đáp án được kế thừa từ Day 6 và đã được xem ở Day 6 (không dùng để chỉnh Day 7).
