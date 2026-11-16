# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

## Bài làm — Đoàn Sỹ Hưng (2A202603990)

Báo cáo: [report/individual/2A202603990_DoanSyHung.md](report/individual/2A202603990_DoanSyHung.md) · Chốt cấu hình: [results/selection.md](results/selection.md)

**Cần có:** container pgvector của Day 6 đang chạy (`docker compose -p day6_student_starter up -d --wait` trong `<Day6>/starter`) và OpenAI API key.
Không có volume Day 6: clone repo Day 6 (https://github.com/hgmdz1210/K5-L3B-DAY06-DoanSyHung-2A202603990-DataIngestion), làm theo README của nó (`python -m src.run --split dev` nạp index A/B bằng `text-embedding-3-small`), rồi chạy các lệnh dưới. Chunk đã xuất sẵn ở `data/day6/chunks.json`; mô tả vision của IDR đã cache ở `data/idr/vision/` nên dựng lại index nâng cao không cần gọi vision lại.

```powershell
python -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env          # điền OPENAI_API_KEY; không commit .env
.venv\Scripts\python -m src.export_day6                                   # kiểm tra kết nối, xuất chunk
.venv\Scripts\python -m src.build_benchmark --day6 <Day6>\benchmark\questions.json
# Retrieval trên dev: dense | bm25 | hybrid | rerank  (thêm --use-filter để dùng filter chapter)
.venv\Scripts\python -m src.eval_retrieval --split dev --method rerank --use-filter --name rerank-filter
# Agent trên dev/test
.venv\Scripts\python -m src.run_agent --split dev --retriever rerank --use-filter --name agent-rerank-filter
.venv\Scripts\python -m src.labels_test                                   # nhãn test + agent_metrics
.venv\Scripts\python -m unittest discover -s tests -v
```

| Thư mục | Nội dung |
|---|---|
| `src/` | `day6_store.py` (pgvector + embedding), `retrieval.py` (dense, BM25, hybrid RRF, rerank), `agent.py`, `run_agent.py`, `eval_retrieval.py`, `labels_test.py` |
| `data/` | `day6/chunks.json` (879 chunk), `benchmark/questions.json` (28 câu) |
| `results/` | trace + summary từng cấu hình (`retrieval/`, `agent/`), log, `selection.md` |

**Thử thách nâng cao** (IDR 2026, `data/9781464823558.pdf`; kết quả test 27/30 = 90.0%):

```powershell
# Nếu Windows chặn cổng 55433: dùng docker-compose.override.yml (cổng 55533) và sửa DATABASE_URL trong .env
.venv\Scripts\python -m src.idr.pages            # phân loại trang, render, crop figure/bảng
.venv\Scripts\python -m src.idr.vision --model gpt-5.4-2026-03-05   # mô tả bằng vision (cache data/idr/vision/)
.venv\Scripts\python -m src.idr.verify_vision    # đối chiếu số với text layer gốc
.venv\Scripts\python -m src.idr.build_index      # chunk + embedding -> pgvector idr_chunks
.venv\Scripts\python -m src.idr.benchmark        # kiểm chứng benchmark với PDF gốc
.venv\Scripts\python -m src.idr.run --split test --retriever rerank --verify --name agent-rerank-verify
.venv\Scripts\python -m src.idr.labels_test      # nhãn đạt/không đạt + tổng hợp theo nhóm
```

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
