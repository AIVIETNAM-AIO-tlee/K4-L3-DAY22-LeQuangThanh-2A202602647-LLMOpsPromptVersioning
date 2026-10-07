# Yêu cầu cài đặt — Day 22: LangSmith + Prompt Versioning

## Phiên bản Python
Python 3.10 trở lên

## Cài đặt toàn bộ thư viện

```bash
pip install -r requirements.txt
```

## requirements.txt

```
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.3.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langsmith>=0.2.0
openai>=1.0.0
faiss-cpu>=1.7.0
ragas>=0.4.0
guardrails-ai>=0.5.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
datasets>=2.0.0
numpy>=1.25.0
```

## Công dụng của từng thư viện

| Thư viện | Dùng để |
|---------|---------|
| `langchain` | Framework LLM cốt lõi |
| `langchain-openai` | `ChatOpenAI`, `OpenAIEmbeddings` |
| `langchain-community` | Tích hợp vector store FAISS |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `langsmith` | LangSmith tracing, client Prompt Hub |
| `openai` | Gọi trực tiếp OpenAI API |
| `faiss-cpu` | Chỉ mục tìm kiếm tương đồng (similarity search) |
| `ragas` | Các chỉ số đánh giá RAG |
| `guardrails-ai` | Framework kiểm định đầu ra (output validation) |
| `python-dotenv` | Đọc tệp `.env` |
| `tiktoken` | Đếm token cho text splitter |
| `datasets` | RAGAS cần dùng bên trong |
| `numpy` | Tính trung bình danh sách điểm RAGAS |

## Lưu ý quan trọng về phiên bản

### RAGAS 0.4.x
- Dùng `from ragas.metrics import faithfulness, answer_relevancy, ...` (KHÔNG import từ `ragas.metrics.collections`)
- `result[metric_name]` trả về một **list** các số float khi có nhiều sample — dùng `numpy.mean()` để tính trung bình
- Truyền `llm=` và `embeddings=` vào hàm `evaluate()`, không truyền vào constructor của metric

### Guardrails AI 0.10.x
- Tham số `on_fail` thuộc về **constructor của validator**: `MyValidator(on_fail=OnFailAction.FIX)`
- `Guard.use()` nhận **instance** của validator, không nhận class
- `Guard.validate(text)` là hàm gọi chính
- Với `OnFailAction.FIX`, validator phải trả về `FailResult(error_message=..., fix_value=...)` — Guardrails thay output bằng `fix_value`. `PassResult(value_override=...)` **không** thay đổi output

### LangChain 0.3.x
- Dùng `ChatOpenAI(api_key=..., base_url=..., model=...)` cho endpoint tùy chỉnh
- Dùng `OpenAIEmbeddings(api_key=..., base_url=..., model=...)` cho endpoint embedding tùy chỉnh

## Biến môi trường

Sao chép `.env.example` thành `.env` rồi điền giá trị. Các biến tối thiểu:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=day22-lab
PROVIDER=openai
OPENAI_API_KEY=sk-...
```

> ⚠️ **Không bao giờ commit `.env` lên git.** Hãy thêm nó vào `.gitignore`.

## Kiểm tra cài đặt

Chạy lệnh kiểm tra cấu hình:
```bash
python config.py
```

Kết quả mong đợi:
```
✅ Config loaded successfully
   LangSmith project : your-project-name
   OpenAI endpoint   : https://...
   Default LLM model : gpt-5.4-mini
   Embedding model   : text-embedding-3-small
```
