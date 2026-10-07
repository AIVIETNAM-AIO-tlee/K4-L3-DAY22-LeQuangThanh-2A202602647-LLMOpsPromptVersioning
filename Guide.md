# Hướng dẫn thực hành — Day 22: LangSmith + Prompt Versioning

Tài liệu này hướng dẫn từng bước chi tiết để hoàn thành lab. Đọc kỹ từng phần trước khi bắt tay vào code.

---

## Chuẩn bị (30 phút)

### Bước 1 — Tạo virtual environment

Luôn dùng virtual environment để tránh xung đột gói giữa các dự án:

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

Sau khi kích hoạt, dấu nhắc terminal sẽ hiển thị `(venv)` ở đầu dòng.

### Bước 2 — Cài thư viện

```bash
pip install -r requirements.txt
```

> Quá trình này có thể mất 5–10 phút lần đầu. Trong lúc chờ, hãy thực hiện bước tiếp theo.

### Bước 3 — Tạo tài khoản LangSmith và lấy API key

1. Truy cập [smith.langchain.com](https://smith.langchain.com) và đăng ký tài khoản miễn phí.
2. Vào **Settings** → **API Keys** → nhấn **Create API Key**.
3. Sao chép key (bắt đầu bằng `lsv2_`) — bạn sẽ cần dùng ở bước tiếp theo.
4. Tạo project mới tại **Projects** → **New Project**, đặt tên ví dụ `day22-lab`.

### Bước 4 — Cấu hình tệp `.env`

Sao chép tệp mẫu:

```bash
cp .env.example .env
```

Mở `.env` bằng bất kỳ editor nào và điền thông tin:

```env
# ─── LangSmith (bắt buộc) ─────────────────────────────────────────────────────
LANGCHAIN_API_KEY=lsv2_pt_...        # API key vừa lấy ở trên
LANGCHAIN_PROJECT=day22-lab          # Tên project trên LangSmith
LANGCHAIN_TRACING_V2=true            # Bật tracing — không thay đổi giá trị này

# ─── Chọn provider LLM ────────────────────────────────────────────────────────
PROVIDER=openai                      # openai | gemini | anthropic | ollama | openrouter

# ─── OpenAI (nếu PROVIDER=openai) ────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# ─── Google Gemini (nếu PROVIDER=gemini) ─────────────────────────────────────
GOOGLE_API_KEY=AIza...

# ─── Anthropic (nếu PROVIDER=anthropic) ──────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ─── OpenRouter (nếu PROVIDER=openrouter) ────────────────────────────────────
OPENROUTER_API_KEY=sk-or-...
```

Giải thích các biến quan trọng:
- `LANGCHAIN_API_KEY`: Xác thực với LangSmith, không bao giờ chia sẻ công khai.
- `LANGCHAIN_PROJECT`: Tên project để nhóm các traces lại, dễ tìm trên dashboard.
- `LANGCHAIN_TRACING_V2`: Phải là `true` để bật tracing — đặt sai sẽ mất toàn bộ traces.
- `PROVIDER`: Xác định LLM và embedding model nào được dùng trong toàn bộ lab.

### Bước 5 — Chọn LLM provider và điền thông tin

Chỉ cần điền thông tin cho provider bạn chọn, bỏ trống các provider còn lại:

- **OpenAI**: Ổn định nhất, khuyến nghị nếu có key. Dùng `gpt-4o-mini` để tiết kiệm chi phí.
- **Gemini**: Miễn phí với quota 15 request/phút — phù hợp nhưng có thể chậm hơn ở bước RAGAS.
- **Anthropic**: Chất lượng rất cao, chi phí trung bình.
- **Ollama**: Chạy hoàn toàn offline. Cần cài [ollama.ai](https://ollama.ai) và pull model trước (`ollama pull llama3.2`).
- **OpenRouter**: Tổng hợp nhiều model, có free tier cho một số model.

### Bước 6 — Xác minh cài đặt

```bash
cd src && python config.py
```

Nếu thấy thông báo xác nhận (không có lỗi đỏ), bạn đã sẵn sàng. Nếu có lỗi, kiểm tra lại các giá trị trong `.env`.

---

## Bước 1: RAG Pipeline với LangSmith (25–45 phút)

### Mục tiêu của bước này

Bạn sẽ xây dựng một RAG pipeline hoàn chỉnh: load dữ liệu → chunk → embed → index vào FAISS → tạo chain hỏi đáp → gắn decorator `@traceable` để mỗi câu hỏi tạo ra một trace trên LangSmith.

### Hướng dẫn từng bước

**1. Mở file `src/01_langsmith_rag_pipeline.py`**

Đọc qua toàn bộ file để hiểu cấu trúc trước khi bắt đầu viết code.

**2. Implement hàm `setup_vectorstore()`**

Hàm này cần thực hiện 4 việc theo thứ tự:

```python
def setup_vectorstore():
    embeddings = get_embeddings()                  # Lấy embedding model từ factory
    docs = load_knowledge_base()                   # Load text từ data/knowledge_base.txt
    chunks = split_text(docs)                      # Chia thành chunks nhỏ hơn
    vectorstore = build_vectorstore(chunks, embeddings)  # Tạo FAISS index
    return vectorstore
```

Gợi ý: hàm `get_embeddings()`, `load_knowledge_base()`, `split_text()`, và `build_vectorstore()` đã được implement trong `utils/`. Chỉ cần gọi đúng thứ tự.

**3. Định nghĩa `RAG_PROMPT`**

Tạo prompt template hướng dẫn LLM trả lời dựa trên context:

```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý hữu ích. Chỉ trả lời dựa trên context được cung cấp. "
               "Nếu không tìm thấy thông tin, hãy nói 'Tôi không tìm thấy thông tin này'.\n\n"
               "Context:\n{context}"),
    ("human", "{question}"),
])
```

**4. Implement hàm `build_rag_chain()`**

Xây dựng LCEL chain nối retriever → prompt → LLM → parser:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever   # main() cần cả 2
```

**5. Implement hàm `ask()` với decorator `@traceable`**

Đây là bước quan trọng nhất — decorator `@traceable` giúp mỗi lần gọi `ask()` tạo ra một trace trên LangSmith:

```python
from langsmith import traceable

@traceable(name="rag-query")
def ask(chain, question: str) -> str:
    return chain.invoke(question)
```

Lưu ý: `@traceable` phải được viết **ngay trên** định nghĩa hàm, không có dòng nào ở giữa.

**6. Hoàn thiện hàm `main()`**

```python
def main():
    vectorstore = setup_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    # SAMPLE_QUESTIONS là list[str] — dùng cho Bước 1 và 2
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {str(answer)[:100]}\n")
```

**7. Chạy bước 1**

```bash
python 01_langsmith_rag_pipeline.py
```

**8. Xác minh trên LangSmith**

Mở [smith.langchain.com](https://smith.langchain.com) → chọn project `day22-lab` → vào tab **Runs**. Bạn sẽ thấy ít nhất 50 traces, mỗi trace chứa câu hỏi, context được truy xuất và câu trả lời.

**9. Chụp ảnh màn hình bằng chứng**

Chụp màn hình giao diện LangSmith đang hiển thị danh sách traces → lưu vào `evidence/01_langsmith_traces.png`.

---

## Bước 2: Prompt Hub & A/B Routing (20–30 phút)

### Mục tiêu của bước này

Bạn sẽ tạo 2 system prompt có phong cách khác nhau, đẩy lên LangSmith Prompt Hub, pull về khi chạy, và định tuyến câu hỏi một cách tất định (cùng câu hỏi → luôn cùng prompt).

### Hướng dẫn từng bước

**1. Mở file `src/02_prompt_hub_ab_routing.py`**

**2. Đổi tên prompt thành tên của bạn**

Tìm các biến `PROMPT_V1_NAME` và `PROMPT_V2_NAME`, đổi thành tên duy nhất để tránh trùng với bạn khác:

```python
PROMPT_V1_NAME = "nguyen-van-a-rag-prompt-v1"   # Thay bằng tên của bạn
PROMPT_V2_NAME = "nguyen-van-a-rag-prompt-v2"
```

**3. Viết 2 system prompt với phong cách khác nhau**

Hai prompt phải có ngữ nghĩa rõ ràng khác nhau (không chỉ khác vài từ). **Bắt buộc giữ `{context}` trong system prompt** — nếu thiếu, LLM sẽ không nhận được tài liệu truy xuất mà chương trình vẫn chạy bình thường, khiến điểm faithfulness ở Bước 3 tụt mạnh.

```python
# V1: Ngắn gọn, thân thiện
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn (2-4 câu), chỉ dựa trên context. "
    "Nếu không có thông tin, hãy nói thẳng là không biết.\n\n"
    "Context:\n{context}"
)

# V2: Chuyên nghiệp, có cấu trúc
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Đọc kỹ context, xác định các facts liên quan, "
    "rồi viết câu trả lời rõ ràng, có tổ chức (3-5 câu). Không suy đoán ngoài context.\n\n"
    "Context:\n{context}"
)
```

`PROMPT_V1` / `PROMPT_V2` đã được tạo sẵn từ 2 biến này ngay bên dưới trong file.

**4. Implement hàm `push_prompts_to_hub(client)`**

```python
url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – ngắn gọn")
url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – có cấu trúc")
```

> Chạy lại Bước 2 lần thứ hai mà không đổi prompt sẽ thấy lỗi `409 Conflict ... Nothing to commit`. Đây **không phải lỗi** — Hub chỉ báo prompt chưa thay đổi nên không tạo phiên bản mới.

**5. Implement hàm `pull_prompts_from_hub(client)`**

```python
prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
```

> Nếu log in `ℹ️ Dùng local fallback ...` nghĩa là pull từ Hub **thất bại** (thường do sai API key) và bạn sẽ mất điểm tiêu chí 2.2/2.3. Phải thấy `↓ Đã pull ... từ Hub` cho cả 2 prompt.

**6. Implement hàm `get_prompt_version()` — định tuyến bằng MD5 hash**

Hash MD5 của `request_id` đảm bảo cùng input luôn cho cùng output (tất định). Hàm trả về **tên prompt** (để tra trong dict `prompts`):

```python
def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME
```

**7. Implement hàm `ask_ab()`**

```python
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs    = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer  = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}
```

**8. Hoàn thiện hàm `main()`**

Các dòng cần điền trong `main()`:

```python
client      = Client(api_key=config.LANGSMITH_API_KEY)
prompts     = pull_prompts_from_hub(client)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
# trong vòng lặp:
version_key = get_prompt_version(request_id)
result      = ask_ab(retriever, llm, prompt, question, version_tag)
```

**9. Chạy và lưu log**

```bash
python 02_prompt_hub_ab_routing.py | tee ../evidence/02_ab_routing_log.txt
```

> **Windows:** dùng **Git Bash** để chạy lệnh `tee`, và chạy `export PYTHONUTF8=1` một lần trước đó (PowerShell: `$env:PYTHONUTF8=1`). Nếu không, log có emoji sẽ lỗi `UnicodeEncodeError` khi ghi ra file.

**10. Chụp ảnh Prompt Hub**

Vào [smith.langchain.com](https://smith.langchain.com) → **Prompt Hub** → tìm 2 prompt vừa push → chụp màn hình → lưu vào `evidence/02_prompt_hub.png`.

---

## Bước 3: RAGAS Evaluation (30–45 phút cài đặt + 15–30 phút chạy)

> **Lưu ý quan trọng:** Bước này cần 15–30 phút để chạy xong (đôi khi lâu hơn nếu dùng Gemini free tier). Hãy bắt đầu sớm và không đóng terminal trong lúc chờ.

### Mục tiêu của bước này

Đánh giá cả 2 phiên bản prompt trên 50 cặp QA bằng 4 chỉ số RAGAS và lưu kết quả vào JSON.

### Hướng dẫn từng bước

**1. Mở file `src/03_ragas_evaluation.py`**

**2. Sao chép `SYSTEM_V1` và `SYSTEM_V2` từ bước 2**

Đảm bảo 2 system prompt giống hệt bước 2 để kết quả có thể so sánh được.

**3. Implement hàm `run_rag(retriever, llm, prompt, question)`**

Điểm quan trọng nhất: trường `contexts` trong RAGAS phải là `list[str]` (danh sách chuỗi), không phải một chuỗi ghép lại:

```python
def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)

    # QUAN TRỌNG: contexts phải là list[str], không ghép thành một string
    contexts = [doc.page_content for doc in docs]
    ctx_str  = "\n\n".join(contexts)          # chỉ dùng để truyền vào prompt

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })
    return {"answer": answer, "contexts": contexts}
```

**4. Hoàn thiện hàm `collect_rag_outputs()`**

`QA_PAIRS` là list các dict có khóa `question` và `reference`:

```python
out = run_rag(retriever, llm, prompt, qa["question"])
results.append({
    "question":  qa["question"],
    "reference": qa["reference"],
    "answer":    out["answer"],
    "contexts":  out["contexts"],   # list[str]
})
```

**5. Implement hàm `build_ragas_dataset()`**

`SingleTurnSample` yêu cầu đúng 4 trường này:

```python
samples = [
    SingleTurnSample(
        user_input=r["question"],           # Câu hỏi
        response=r["answer"],               # Câu trả lời của LLM
        retrieved_contexts=r["contexts"],   # list[str] — contexts truy xuất được
        reference=r["reference"],           # Đáp án chuẩn từ qa_pairs.py
    )
    for r in rag_results
]
return EvaluationDataset(samples=samples)
```

**6. Hoàn thiện hàm `run_ragas_eval()`**

```python
dataset = build_ragas_dataset(rag_results)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=llm_eval,
    embeddings=emb_eval,
)
```

Phần tính trung bình điểm và in kết quả đã có sẵn trong file.

**7. Hoàn thiện hàm `main()`**

```python
vectorstore = setup_vectorstore()
# ... (phần chạy V1/V2 và in bảng so sánh đã có sẵn)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
```

**8. Chạy bước 3**

```bash
python 03_ragas_evaluation.py
```

Quá trình sẽ mất 15–30 phút. Đừng đóng terminal.

**9. Chụp ảnh terminal**

Khi thấy bảng so sánh điểm xuất hiện → chụp màn hình → lưu vào `evidence/03_ragas_scores.png`.

**10. Sao chép báo cáo JSON vào thư mục evidence**

```bash
cp ../data/ragas_report.json ../evidence/03_ragas_report.json
```

---

## Bước 4: Guardrails AI Validators (20–30 phút)

### Mục tiêu của bước này

Xây dựng 2 validator tùy chỉnh: một để phát hiện và che thông tin cá nhân (PII), một để kiểm tra và sửa JSON lỗi từ đầu ra của LLM.

### Hướng dẫn từng bước

**1. Mở file `src/04_guardrails_validator.py`**

**2. Implement hàm `PIIDetector.validate()`**

Logic: duyệt qua `self.PII_PATTERNS` (đã cho sẵn), tìm các match, thay bằng placeholder.

> **Quan trọng:** khi phát hiện PII phải trả về `FailResult(..., fix_value=...)`. Với `on_fail=OnFailAction.FIX`, Guardrails sẽ thay output bằng `fix_value`. Nếu trả về `PassResult(...)`, Guardrails giữ nguyên input — PII **không** bị che dù log vẫn in "Đã redact".

```python
def validate(self, value: str, metadata: dict):
    redacted_text = value
    found_pii     = []

    for pii_type, pattern in self.PII_PATTERNS.items():
        for match in re.findall(pattern, value):
            redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
            found_pii.append((pii_type, match))

    if found_pii:
        return FailResult(
            error_message=f"Phát hiện PII: {[p[0] for p in found_pii]}",
            fix_value=redacted_text,
        )
    return PassResult()
```

**3. Implement hàm `JSONFormatter._repair()`**

Sửa các lỗi JSON phổ biến từ đầu ra LLM (phần gỡ markdown fences đã có sẵn):

```python
text = text.replace("'", '"')                     # nháy đơn → nháy đôi
text = re.sub(r',\s*([}\]])', r'\1', text)        # xóa dấu phẩy thừa trước } hoặc ]
```

**4. Implement hàm `JSONFormatter.validate()`**

```python
def validate(self, value: str, metadata: dict):
    # 1) JSON hợp lệ sẵn → pass
    try:
        json.loads(value)
        return PassResult()
    except json.JSONDecodeError:
        pass

    # 2) Sửa được → FailResult kèm fix_value là JSON đã chuẩn hóa
    try:
        parsed = json.loads(self._repair(value))
        return FailResult(error_message="JSON lỗi, đã tự sửa",
                          fix_value=json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        pass

    # 3) Không sửa được → trả về JSON dự phòng (tiêu chí 4.7)
    fallback = json.dumps({"error": "Không thể phân tích JSON", "raw": value[:200]},
                          ensure_ascii=False)
    return FailResult(error_message="Không thể sửa JSON", fix_value=fallback)
```

**5. Hoàn thiện hàm `demo_pii_guard()` và `demo_json_guard()`**

`on_fail` phải truyền vào **constructor của validator**, không phải `Guard.use()`:

```python
guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
# trong vòng lặp test case:
result = guard.validate(text)
```

**6. Kiểm tra kết quả**

- PII: dòng `Output:` phải chứa `[EMAIL_REDACTED]`, `[PHONE_REDACTED]`, ... — nếu Output giống hệt Input là bạn đang trả về `PassResult` thay vì `FailResult(fix_value=...)`.
- JSON: các case sửa được phải in ra JSON đã format lại; case không sửa được phải in ra JSON dự phòng `{"error": ...}`.

**7. Chạy và lưu log**

Script in cả 2 demo (PII + JSON), nên ghi ra **cả 2 tệp evidence** trong 1 lần chạy:

```bash
python 04_guardrails_validator.py | tee ../evidence/04_pii_demo_log.txt ../evidence/04_json_demo_log.txt
```

> **Windows:** dùng **Git Bash** để chạy lệnh `tee`, và chạy `export PYTHONUTF8=1` một lần trước đó (PowerShell: `$env:PYTHONUTF8=1`). Nếu không, log có emoji sẽ lỗi `UnicodeEncodeError` khi ghi ra file.

---

## Kiểm tra trước khi nộp

Hãy đi qua danh sách này trước khi push lên GitHub:

**Mã nguồn:**
- [ ] `src/01_langsmith_rag_pipeline.py` chạy không có lỗi, in ra 50 câu hỏi/đáp
- [ ] `src/02_prompt_hub_ab_routing.py` chạy không có lỗi, log hiển thị nhãn v1/v2
- [ ] `src/03_ragas_evaluation.py` hoàn thành và in bảng điểm so sánh
- [ ] `src/04_guardrails_validator.py` chạy không có lỗi, hiển thị kết quả các test case
- [ ] `data/ragas_report.json` tồn tại và chứa điểm của cả V1 lẫn V2

**Bằng chứng (evidence):**
- [ ] `evidence/01_langsmith_traces.png` — Ảnh chụp LangSmith, thấy rõ ít nhất 50 traces
- [ ] `evidence/02_prompt_hub.png` — Ảnh chụp Prompt Hub, thấy rõ 2 prompt được đặt tên
- [ ] `evidence/02_ab_routing_log.txt` — File log có nội dung, hiển thị cả v1 lẫn v2
- [ ] `evidence/03_ragas_scores.png` — Ảnh chụp bảng điểm RAGAS trên terminal
- [ ] `evidence/03_ragas_report.json` — File JSON hợp lệ (sao chép từ `data/`)
- [ ] `evidence/04_pii_demo_log.txt` — File log có ít nhất 5 test case
- [ ] `evidence/04_json_demo_log.txt` — File log có ít nhất 4 test case

**Bảo mật:**
- [ ] `.gitignore` có dòng `.env`
- [ ] Không có API key nào xuất hiện trong bất kỳ tệp `.py` nào
- [ ] Lệnh `git diff --cached` không hiển thị nội dung `.env`

---

## Nộp bài

### 1. Tạo GitHub repository public

Truy cập [github.com/new](https://github.com/new), tạo repository mới:
- Tên: **theo quy ước** trong [SUBMISSION.md](SUBMISSION.md), ví dụ `K4-L3-DAY22-NguyenVanA-2A20260000-LLMOpsPromptVersioning`
- Visibility: **Public** (bắt buộc để chấm điểm)
- Không tích "Initialize with README" (vì bạn đã có code)

### 2. Push code lên GitHub

```bash
# Tạo .gitignore nếu chưa có
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "venv/" >> .gitignore

git init
git add .
git status   # Kiểm tra lần cuối — đảm bảo KHÔNG thấy .env trong danh sách

git commit -m "Day 22: LangSmith + Prompt Versioning lab"
git remote add origin https://github.com/<tên-của-bạn>/<tên-repo-theo-SUBMISSION.md>.git
git push -u origin main
```

### 3. Nộp thông tin

Nộp qua cổng của khóa học:

1. **URL GitHub repository** — ví dụ: `https://github.com/nguyen-van-a/K4-L3-DAY22-NguyenVanA-2A20260000-LLMOpsPromptVersioning`
2. **URL LangSmith project** — ví dụ: `https://smith.langchain.com/o/<org-id>/projects/p/<project-id>`
3. Xác nhận thư mục `evidence/` đã có đầy đủ 7 tệp

> **Nhắc lại:** Không bao giờ commit tệp `.env` hoặc dán API key vào mã nguồn. Đây là lỗi vi phạm bảo mật bị trừ 10 điểm tự động.
