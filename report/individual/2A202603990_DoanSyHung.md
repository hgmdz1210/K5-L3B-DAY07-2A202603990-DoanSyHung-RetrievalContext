# Báo cáo Day 7 — Đoàn Sỹ Hưng (2A202603990)

## Dữ liệu và benchmark

- **Nguồn:** *Pride and Prejudice* (Project Gutenberg #1342, bản HTML), phạm vi công cộng tại Mỹ. Corpus có 475 trang, 61 chương. Dùng lại index B của Day 6: 879 chunk theo đoạn, mỗi chunk tối đa 1200 ký tự. Chunk được xuất ra `data/day6/chunks.json`.
- **Vector store:** PostgreSQL 16 + pgvector 0.8.7 (Docker trong repo Day 6), cosine, exact search. Embedding OpenAI `text-embedding-3-small`, 1536 chiều; query embedding cùng model với index, code tự kiểm tra số chiều.
- **Agent:** `gpt-5.4-mini-2026-03-17`, tool calling, `reasoning_effort=none`, `temperature=0`.
- **Chia dev/test:** giữ cách chia của Day 6 (mỗi câu thứ 4 vào test).
  - Dev: 18 câu có đáp án + 2 câu thiếu bằng chứng.
  - Test: 6 câu có đáp án + 2 câu thiếu bằng chứng.
  - 5 câu có filter chapter do người dùng nêu trong câu hỏi.
- **Câu có đáp án:** gold gồm nguồn, trang và evidence nguyên văn; script kiểm tra mỗi evidence nằm trọn trong một chunk đúng trang.
- **Câu thiếu bằng chứng (u01–u04):** chọn gần chủ đề và có mồi nhử. Đã grep toàn corpus để xác nhận không có đáp án; cách kiểm tra ghi trong `absence_reason`.
- **Tiêu chí chấm:** theo BENCHMARK.md.
- **Hạn chế:** 6 câu test có đáp án đã được xem ở Day 6. Day 7 không dùng chúng để chỉnh, nhưng test không hoàn toàn độc lập.

## Retrieval và thử nghiệm

Điều kiện chung: top-k=3, context 4000 ký tự, query gốc. Hit tính khi chunk đúng nguồn, đúng trang và chứa nguyên văn một gold. Chi phí tính cho cả split; giá token là ước tính, cần đối chiếu bảng giá OpenAI.

| Cấu hình | Phương pháp | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence | Thời gian (trung vị) / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense pgvector, không filter | Index Day 6 | dev | 8/18 (44%) | 12/18 (67%) | 12/18 | 360 ms / ~$0.00001 |
| Phương pháp 2 | BM25 (k1=1.5, b=0.75) | Khớp từ khóa tốt cho câu hỏi gần nguyên văn | dev | 7/18 (39%) | 8/18 (44%) | 8/18 | 3 ms / $0 |
| Phương pháp 3 | Hybrid RRF (k=60, mỗi nhánh 20 candidate) | Dense và BM25 bổ sung cho nhau | dev | 7/18 (39%) | 10/18 (56%) | 10/18 | 389 ms / ~$0.00001 |
| Cải thiện 1 | Dense top-20 ∪ BM25 top-20, LLM rerank, lấy top-3 | Evidence có trong pool nhưng ở hạng thấp; rerank đẩy lên | dev | 14/18 (78%) | 16/18 (89%) | 16/18 | 2.1 s / ~$0.10 |
| Cải thiện 2 | Cải thiện 1 + filter chapter | Phân biệt cảnh giống nhau ở các chương khác nhau | dev | 15/18 (83%) | 17/18 (94%) | 17/18 | 2.0 s / ~$0.10 |
| Cấu hình đã chọn | Cải thiện 2 | Chốt trên dev | test | 5/6 (83%) | 6/6 (100%) | 6/6 | 2.0 s / ~$0.04 |
| Baseline | Dense | — | test | 3/6 (50%) | 5/6 (83%) | 5/6 | 457 ms / ~$0.00001 |

Rerank (Cải thiện 1) vừa là phương pháp thứ 4 vừa là cải thiện. Dense + filter cho kết quả bằng dense (12/18), vì filter chỉ đưa q22 lên hạng 10.

**Lỗi dev 1 (q11, q18, q19, q21):** dense xếp evidence ở hạng 13 đến ngoài top 50, trong khi BM25 xếp ở hạng 1–10. Nguyên nhân là câu hỏi chứa từ khóa đặc thù ("neither sensible nor agreeable", "Brighton", "neighbours") mà embedding làm loãng.
- Thử hybrid RRF trước: chỉ đạt 10/18, vì k=60 làm phẳng thứ hạng. Đây là kết quả không cải thiện.
- Sau đó thử pool dense ∪ BM25 + rerank: đạt 16/18.

**Lỗi dev 2 (q22):** chương 34 và chương 58 đều có cảnh Darcy cầu hôn, nên dense lấy nhầm chương. Khi thêm filter `chapter=58` mà người dùng nêu trong câu hỏi, đạt 17/18.

**Lỗi chưa giải quyết (q02):** câu hỏi diễn đạt lại ("dismiss her looks") nên mọi cấu hình đều trượt. Agent trả lời bằng đoạn hồi tưởng ở trang 30 thay vì câu gốc ở trang 15.

**Lý do chọn:** tiêu chí được công bố trước khi chạy test trong `results/selection.md`, theo thứ tự: Context Evidence, từ chối nhầm, Hit, rồi thời gian/chi phí. Cải thiện 2 dẫn đầu ở mọi tiêu chí chất lượng. Đổi lại, chi phí khoảng gấp 3 lần và chậm hơn khoảng 1 giây mỗi câu so với baseline.

## Agent và đánh giá

- **Tool:** `search(query)`, `final_answer(answer, citations)`, `abstain(reason)`. Model bắt buộc gọi tool ở mỗi lượt, mỗi lượt 1 tool.
- **Truy xuất:** lượt đầu dùng câu hỏi gốc; các lượt sau model tự viết lại query.
- **Điều kiện dừng:** dừng khi model gọi `final_answer` hoặc `abstain`. Tối đa 4 lượt search, quá giới hạn thì buộc kết luận.
- **Context:** cộng dồn qua các lượt search, mỗi lượt thêm top-3 chunk chưa thấy, tổng budget 12000 ký tự, bỏ chunk trùng.
- **Citation:** phải nằm trong context và xuất hiện dạng `[id]` trong câu trả lời. Sai thì model được sửa 1 lần, sai tiếp thì từ chối.
- **Lỗi API/DB:** ghi `status=error`, không thay bằng kết quả giả. Nội dung chunk được xem là dữ liệu, không phải chỉ dẫn.

| Cấu hình / split | Tool use | Số lượt search (phân bố; trung bình) | Grounded 0/1/2 (trung bình) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Dense / dev | 20/20 | 1:13, 2:2, 3:4, 4:1; 1.65 | chưa chấm tay | — | 16/16 | 2/2 | 2/18 |
| Rerank / dev | 20/20 | 1:17, 3:2, 4:1; 1.35 | chưa chấm tay | — | 18/18 | 2/2 | 0/18 |
| Rerank + filter / dev | 20/20 | 1:17, 2:2, 3:1; 1.20 | chưa chấm tay | — | 18/18 | 2/2 | 0/18 |
| **Dense / test (baseline)** | 8/8 | 1:4, 2:1, 3:2, 4:1; 2.0 | 1/0/5 (1.67) | 6/6 | 5/5 | 2/2 | 1/6 |
| **Rerank + filter / test (đã chọn)** | 8/8 | 1:6, 3:2; 1.5 | 0/1/5 (1.83) | 7/8 | 6/6 | 2/2 | 0/6 |

Chấm context trên 6 câu test có đáp án (thang 0/1/2):
- Baseline: 5 câu điểm 2, 1 câu điểm 0 (q12).
- Cấu hình đã chọn: 6 câu điểm 2.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q12 (cấu hình chọn) | Có bằng chứng | 1 search, chunk tr.232 ở hạng 1 | "saved a friend from the inconveniences of a most imprudent marriage" [p232] | 2: đúng nguyên văn |
| q12 (baseline) | Có bằng chứng, lỗi | 4 search dense, tr.232 không lọt top-3 | Từ chối | 0: từ chối nhầm, lỗi do retrieval |
| q16 (cấu hình chọn) | Có bằng chứng, lỗi | 1 search, context gồm tr.277, 286, 289 | Vế 1 dùng "Let her go" [p289]; vế 2 "never despaired" [p277] | 1: vế 1 lấy nhầm cảnh ch.41 (Lydia đi Brighton), đáp án đúng nằm ở tr.277; citation p289 không hỗ trợ |
| q20 (cả hai) | Có bằng chứng | 1 search | "I will make no promise of the kind", "not to be intimidated" [p439 ×2] | 2: khớp evidence thay thế tr.439; thiếu ý ở tr.440 |
| u03, u04 (cả hai) | Thiếu bằng chứng | 3 search, các query được viết lại | Từ chối | Đúng: không bịa tên con hay ngày cưới |

- **Nhãn test:** AI soạn đề xuất sau khi đọc từng chunk được cite; học viên đã tự chấm lại bằng tay từng câu, đối chiếu chunk gốc và giữ nguyên nhãn. Rủi ro còn lại: chỉ có một người chấm.
- **Trace và nhãn:**
  - Retrieval: `results/retrieval/<cfg>-<split>/`.
  - Agent: `results/agent/<cfg>-<split>/` (gồm `traces.json`, `summary.json`; riêng test có thêm `reviews.json`, `summary_labels.json`).
  - Nhãn test và lý do: `src/labels_test.py`.
  - Lệnh chạy: xem README.
- **Chi phí agent (ước tính):**
  - Dev: dense ~$0.06, rerank ~$0.19, rerank + filter ~$0.16.
  - Test: ~$0.03 (baseline) và ~$0.08 (cấu hình đã chọn).
  - Token thật được ghi trong `summary.json`.
- **Trước/sau trên test:**
  - Grounded answer tăng từ 1.67 lên 1.83.
  - Từ chối nhầm giảm từ 1/6 xuống 0/6.
  - Context Evidence tăng từ 5/6 lên 6/6.
- **Đánh đổi:**
  - Chi phí khoảng gấp 2.7 lần (thêm 1 lần gọi LLM để rerank cho mỗi lượt search).
  - Agent ít khi search lại khi đã có bằng chứng gián tiếp (q02, q16). Rerank tốt hơn không ngăn được lỗi suy luận nhầm cảnh.
- **Giới hạn:**
  - Test nhỏ: 1 câu bằng 17 điểm phần trăm.
  - Chỉ có một tài liệu và một người chấm.
  - Grounded answer trên dev chưa chấm tay, chỉ có số đo tự động.
  - Giá token là ước tính.

## Thử thách nâng cao (nếu thực hiện)

- **Tài liệu:** World Bank, *International Debt Report 2026*.
  - 280 trang PDF: xxii trang đầu, 254 trang nội dung.
  - Giấy phép CC BY 3.0 IGO. Bản gốc: https://hdl.handle.net/10986/45762 (DOI 10.1596/978-1-4648-2355-8). File: `data/9781464823558.pdf`.
  - Nội dung: text, 2 bảng lớn, 47 biểu đồ, 3 sơ đồ, 127 trang hồ sơ quốc gia/vùng dạng infographic.
- **Xử lý (code trong `src/idr/`):**
  - **Text:** PyMuPDF tách theo block, gom ≤1200 ký tự, bỏ phần nằm trong vùng hình và header/footer.
  - **Bảng, biểu đồ, sơ đồ:** cắt vùng hình từ caption tới dòng Source/Note, render 150 dpi. `gpt-5.4-2026-03-05` vision mô tả trục, chú giải, đơn vị, giá trị (≈ nếu chỉ ước lượng được từ hình) và chép bảng đủ mọi ô.
  - **Hồ sơ quốc gia:** dùng vision theo trang. Lý do: text layer có đủ số nhưng sai thứ tự nhãn; ví dụ trang All LMICs ghi "Japan 144 112", nhưng hình là China 144, Japan 112.
- **Kiểm chứng trích xuất** (`results/idr/vision_check.json`): so 5050 con số do vision ghi ra với text layer gốc.
  - Bảng: khớp 100%.
  - Hồ sơ: 7/4164 số không khớp, đều là ước lượng.
  - Biểu đồ không in nhãn giá trị: 81/818 số không khớp, là ước lượng mà model không đánh dấu ≈.
- **Phạm vi index:**
  - 511 chunk trên 265 trang: 332 text, 2 bảng, 47 biểu đồ, 3 sơ đồ, 127 hồ sơ.
  - Lưu trong bảng pgvector `idr_chunks`, embedding `text-embedding-3-small`.
  - Mỗi chunk giữ trang PDF, trang in, id figure, bbox và ảnh crop.
  - **Không index 14 trang** (bìa, trang trắng, trang tách phần). Không ảnh hưởng nội dung vì các trang này không có dữ liệu.
  - **Giới hạn:** số trên biểu đồ không in nhãn chỉ là ước lượng; 4 figure nằm trong box ban đầu bị bỏ sót và đã được bổ sung.
- **Benchmark** (`data/idr/benchmark.json`, soạn trong `src/idr/benchmark.py`):
  - Chia dev/test trước khi chạy bất kỳ thử nghiệm nào.
  - Dev: 15 câu có đáp án + 3 câu thiếu bằng chứng.
  - Test: 24 câu có đáp án (bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4) + 6 câu thiếu bằng chứng.
  - Evidence ghi theo trang và vùng. Mọi số và trích dẫn được kiểm tự động với **text layer gốc**; giá trị chỉ đọc bằng mắt được đối chiếu với ảnh crop.
  - Gold d12 được bổ sung evidence thay thế sau khi đọc kết quả dev; việc này không thay đổi pipeline.
- **Chọn cấu hình trên dev** (`results/idr/selection.md`), số câu đạt khi chấm tay:

  | Cấu hình | Đạt |
  |---|---|
  | Dense | 17/18 |
  | Rerank | 17/18 |
  | Rerank top-5 | 16/18 |
  | Prompt v2 / v3 (không cải thiện) | 16/18 |
  | **Rerank + verifier** | **18/18** |

  Cấu hình chốt gồm:
  - Agent `gpt-5.4-mini`, tối đa 5 lượt search, context 30000 ký tự.
  - Tool `search`: dense top-20 ∪ BM25 top-20, cho LLM rerank, lấy top-3.
  - Verifier kiểm tra câu trả lời với đúng các đoạn được cite; không đạt thì từ chối.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 5/6 | 83% | t06: agent tự mâu thuẫn ở câu so sánh, verifier đổi thành từ chối |
| Ảnh, sơ đồ | 5/5 | 100% | Đọc đúng sơ đồ phân rã chủ nợ, kể cả chỗ text layer xếp sai nhãn (t08) |
| Biểu đồ | 5/5 | 100% | Giá trị ước lượng lệch ±2 tỷ, xu hướng và thứ hạng đúng |
| Multi-hop | 2/4 | 50% | t17: tự mâu thuẫn rồi bị từ chối. t19: dùng 226 tỷ ước lượng từ biểu đồ thay vì 227 tỷ báo cáo ghi rõ (90.3% so với 90.7%, chấm chặt) |
| Text | 4/4 | 100% | |
| Thiếu bằng chứng | 6/6 | 100% | Từ chối, không thêm khẳng định |
| **Toàn bộ** | **27/30** | **90.0%** | Đạt ngưỡng ≥90% (không làm tròn). Baseline agent + dense: 24/30 (80%) |

- **Trace, nhãn và evidence từng câu:**
  - Trace: `results/idr/agent-rerank-verify-test/traces.json`.
  - Nhãn và lý do: `labels.json`, `labels_summary.json`, soạn trong `src/idr/labels_test.py`.
  - Học viên đã đọc lại benchmark trước khi chạy và đã đọc, duyệt nhãn đạt/không đạt của toàn bộ 30 câu test.
- **Lỗi hệ thống:** 0.
- **Thời gian:** trung vị 6.4 giây/câu.
- **Token test (cấu hình chốt):** agent 106k đầu vào / 5.6k đầu ra, rerank 433k đầu vào, verifier 37k đầu vào. Chi phí ước tính khoảng $0.48 với giá giả định cho `gpt-5.4-mini`.
- **Token vision cho index:** 507k đầu vào / 154k đầu ra (`gpt-5.4`), tốn **$3.59 thực tế**.
- **Chi phí thực tế** (dashboard OpenAI, Analytics > Spend, 10–16/11/2026):
  - `gpt-5.4` $3.59: vision.
  - `gpt-5.4-mini` $2.08: mọi lần chạy agent, rerank và verifier của Day 7, cả phần cơ bản và nâng cao.
  - `text-embedding-3-small` $0.02: gồm cả Day 6.
  - **Tổng Day 7 khoảng $5.7.**
  - Chi phí theo từng cấu hình trong các bảng trên là ước tính từ số token đã log, theo giá giả định $0.75/$4.5 cho mỗi 1M token của `gpt-5.4-mini`. Tổng ước tính phần mini là $3.26, cao hơn thực tế ($2.08) vì OpenAI cache prompt nên token đầu vào lặp lại được tính rẻ hơn. Dùng các số ước tính này để so sánh tương đối giữa các cấu hình.
- **Giới hạn:**
  - Kết quả chạm đúng ngưỡng. Lỗi còn lại tập trung ở câu so sánh: agent viết kết luận trước khi tính. Verifier chặn được câu trả lời sai nhưng đổi chúng thành từ chối.
  - Rerank tốn khoảng 15k token/lượt.
  - Chỉ một người chấm.
- **Tái lập:** xem phần "Thử thách nâng cao" trong README.
- **Thay đổi sau khi chạy phần cơ bản:**
  - Thông báo lỗi citation trong `src/agent.py` giờ liệt kê id hợp lệ, và agent có thêm verifier tùy chọn. Kết quả phần cơ bản được chạy trước thay đổi này; thay đổi chỉ tác động khi agent cite sai.
  - pgvector chạy ở cổng 55533 (`docker-compose.override.yml`) vì Windows chặn dải cổng chứa 55433. Dữ liệu dùng chung volume của Day 6, không đổi.
