# Checkpoints — Day 22: LangSmith + Prompt Versioning

Lab gồm 1 checkpoint chuẩn bị + 4 checkpoint ứng với 4 nhiệm vụ. Hoàn thành checkpoint trước rồi mới sang checkpoint sau. Hướng dẫn chi tiết từng bước ở [Guide.md](Guide.md), tiêu chí chấm ở [RUBRIC.md](RUBRIC.md).

---

## Checkpoint 0 — Chuẩn bị môi trường (~30 phút)

**Cần làm**
- Tạo virtualenv, `pip install -r requirements.txt`.
- Tạo tài khoản + API key LangSmith, tạo project (ví dụ `day22-lab`).
- `cp .env.example .env`, điền `LANGSMITH_*`, `LANGCHAIN_TRACING_V2=true`, `PROVIDER` và key của provider đã chọn.

**Sản phẩm**: file `.env` (chỉ ở máy local, không commit).

**Cần hiểu**
- Vì sao biến `LANGCHAIN_TRACING_V2` / `LANGSMITH_API_KEY` phải được đặt **trước** khi import LangChain.
- Vai trò của `PROVIDER` và `utils/llm_factory.py`.

**Tự kiểm tra**
```bash
cd src && python config.py      # không có lỗi
git status                      # KHÔNG thấy .env
```

---

## Checkpoint 1 — RAG Pipeline + LangSmith tracing (25đ, 25–45 phút)

**Cần làm**: hoàn thành `src/01_langsmith_rag_pipeline.py` — `setup_vectorstore()`, `RAG_PROMPT`, `build_rag_chain()` (retriever → prompt → LLM → parser), `ask()` có `@traceable`, chạy 50 câu hỏi.

**Sản phẩm**: `evidence/01_langsmith_traces.png`.

**Cần hiểu**
- Luồng chunk → embed → FAISS → retrieve → generate.
- `@traceable` tạo trace thế nào; một trace gồm những run con nào.

**Tự kiểm tra**
- Script in đủ 50 cặp Q/A, không lỗi.
- LangSmith project hiển thị **≥ 50 traces**; mở 1 trace thấy câu hỏi, context được truy xuất và câu trả lời.

---

## Checkpoint 2 — Prompt Hub & A/B Routing (25đ, 20–30 phút)

**Cần làm**: trong `src/02_prompt_hub_ab_routing.py` đổi `PROMPT_V1_NAME`/`PROMPT_V2_NAME` theo tên mình, viết 2 system prompt khác nhau rõ rệt, push lên Hub, pull về khi chạy, routing tất định bằng hash `request_id`.

**Sản phẩm**: `evidence/02_prompt_hub.png`, `evidence/02_ab_routing_log.txt`.

**Cần hiểu**
- Vì sao cần versioning prompt và tách prompt khỏi code.
- Vì sao routing bằng hash là tất định còn `random` thì không.

**Tự kiểm tra**
```bash
python 02_prompt_hub_ab_routing.py | tee ../evidence/02_ab_routing_log.txt
grep -c "v1" ../evidence/02_ab_routing_log.txt; grep -c "v2" ../evidence/02_ab_routing_log.txt   # cả hai > 0
```
- Chạy lại lần 2: cùng `request_id` → cùng phiên bản.
- Log in "Đã pull ... từ Hub" (không phải "local fallback").
- Prompt Hub hiển thị cả 2 prompt.

---

## Checkpoint 3 — RAGAS Evaluation (25đ, 45–75 phút — bắt đầu sớm)

**Cần làm**: trong `src/03_ragas_evaluation.py` dùng lại đúng 2 system prompt ở bước 2, chạy 50 cặp QA qua cả V1 và V2, dựng `EvaluationDataset` từ `SingleTurnSample`, tính 4 chỉ số, lưu report.

**Sản phẩm**: `data/ragas_report.json`, `evidence/03_ragas_report.json`, `evidence/03_ragas_scores.png`.

**Cần hiểu**
- Ý nghĩa của `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`.
- Vì sao prompt khác nhau cho ra điểm khác nhau.

**Tự kiểm tra**
```bash
cp ../data/ragas_report.json ../evidence/03_ragas_report.json
python -m json.tool ../evidence/03_ragas_report.json
```
- Report có điểm của **cả V1 và V2**, đủ 4 chỉ số, 50 mẫu mỗi phiên bản.
- Faithfulness ≥ 0.8 ở ít nhất 1 phiên bản.

---

## Checkpoint 4 — Guardrails AI Validators (25đ, 20–30 phút)

**Cần làm**: trong `src/04_guardrails_validator.py` tự viết `PIIDetector` (regex, ≥ 3 loại PII) và `JSONFormatter` (tự sửa ≥ 2 kiểu lỗi, có JSON dự phòng), dùng `@register_validator`, `on_fail=OnFailAction.FIX` truyền vào **constructor**.

**Sản phẩm**: `evidence/04_pii_demo_log.txt`, `evidence/04_json_demo_log.txt`.

**Cần hiểu**
- Vòng đời validate → fail → fix trong Guardrails.
- Khác biệt giữa truyền `on_fail` vào constructor và vào `Guard.use()`.

**Tự kiểm tra**
- PII log: ≥ 5 test case, có đầu vào sạch, PII bị thay bằng chuỗi an toàn.
- JSON log: ≥ 4 test case (hợp lệ, có fences, nháy đơn, sai hoàn toàn → JSON dự phòng).

---

## Checkpoint cuối — Nộp bài

Chạy đủ phần "Kiểm tra trước khi nộp" trong [SUBMISSION.md](SUBMISSION.md), push repo đặt đúng tên và nộp link trước deadline.
