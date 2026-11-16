# Thử thách nâng cao — chốt cấu hình trước khi chạy test

Ngày chốt: 2026-11-16, **trước khi chạy bất kỳ cấu hình nào trên test**. Số liệu dưới đây là dev
(15 câu có đáp án + 3 câu thiếu bằng chứng). "Đạt" = chấm tay theo ADVANCED.md §3 (đúng và đủ, citation đúng
trang/vùng, hoặc từ chối không thêm khẳng định vô căn cứ).

## Tiêu chí (thứ tự ưu tiên)

1. Số câu dev đạt (chấm tay).
2. Câu thiếu bằng chứng từ chối đúng và không từ chối nhầm câu có đáp án.
3. Context đủ evidence, Hit@3.
4. Thời gian và chi phí.

## Retrieval trên dev (lượt tìm với câu hỏi gốc, top-3)

| Phương pháp | Hit@1 | Hit@3 | Context đủ evidence |
|---|---|---|---|
| Dense (pgvector `idr_chunks`) | 9/15 | 10/15 | 10/15 |
| BM25 | 9/15 | 13/15 | 11/15 |
| Hybrid RRF | 8/15 | 11/15 | 11/15 |
| Rerank (dense 20 ∪ BM25 20 → LLM) | 15/15 | 15/15 | 13/15 |

## Agent trên dev

| Cấu hình | Context đủ | Abstain đúng | Từ chối nhầm | Đạt (tay) | Trung vị | Chi phí ước tính |
|---|---|---|---|---|---|---|
| agent-dense (baseline) | 13/15 | 2/3 | 0/15 | 17/18 | 3.5 s | $0.07 |
| agent-rerank | 14/15 | 2/3 | 0/15 | 17/18 | 4.5 s | $0.26 |
| agent-rerank-k5 | 14/15 | 1/3 | 0/15 | 16/18 | 4.5 s | $0.25 |
| agent-rerank-v2 (prompt kiểm tra chặt) | 15/15 | 2/3 | 1/15 | 16/18 | 4.4 s | $0.29 |
| agent-rerank-v3 (prompt hẹp) | 14/15 | 1/3 | 0/15 | 16/18 | 4.9 s | $0.25 |
| **agent-rerank-verify** | **15/15** | **3/3** | **0/15** | **18/18** | 6.2 s | $0.30 |

## Quyết định

- **Cấu hình chọn:** `agent-rerank-verify` — agent `gpt-5.4-mini-2026-03-17`, prompt v1, tool `search` =
  dense top-20 ∪ BM25 top-20 → LLM rerank (mỗi candidate cắt 2000 ký tự) → top-3; tối đa 5 lượt search;
  context budget 30000 ký tự; citation bắt buộc; verifier (`gpt-5.4-mini`) kiểm tra câu trả lời với các đoạn
  được cite, không đạt → từ chối.
- **Baseline:** `agent-dense` — cùng agent/prompt, tool `search` chỉ dense top-3, không verifier.
- Lệnh test (mỗi cấu hình chạy đúng một lần, không chỉnh sau khi xem):

```powershell
.venv\Scripts\python -m src.idr.run --split test --retriever rerank --verify --name agent-rerank-verify
.venv\Scripts\python -m src.idr.run --split test --retriever dense --name agent-dense
.venv\Scripts\python -m src.idr.run --split test --retriever rerank --retrieval-only --name ret-rerank
.venv\Scripts\python -m src.idr.run --split test --retriever dense --retrieval-only --name ret-dense
```

Ghi chú: gold d12 được bổ sung evidence thay thế (PDF p.34, p.62) sau khi đọc dev — chỉ làm đầy đủ gold,
không thay đổi pipeline. Thay đổi code trong A4: thông báo lỗi citation liệt kê id hợp lệ; thêm verifier.
