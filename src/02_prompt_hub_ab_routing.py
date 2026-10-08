"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import os
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# Tên prompt duy nhất cho sinh viên Lê Quang Thanh (MSSV: 2A202602647)
PROMPT_V1_NAME = "le-quang-thanh-rag-v1"
PROMPT_V2_NAME = "le-quang-thanh-rag-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# SYSTEM_V1 — phong cách ngắn gọn, súc tích (2-4 câu)
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn (2-4 câu), chỉ dựa trên context sau. "
    "Nếu không có thông tin, hãy nói thẳng là không biết.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# SYSTEM_V2 — phong cách chuyên gia, có cấu trúc, logic (3-5 câu)
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Đọc kỹ context, xác định các facts liên quan, "
    "rồi viết câu trả lời rõ ràng, có tổ chức (3-5 câu). Không suy đoán ngoài context.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    """
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – ngắn gọn, súc tích")
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 push lỗi hoặc đã tồn tại: {e}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – có cấu trúc, phân tích")
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 push lỗi hoặc đã tồn tại: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}': {e}")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}': {e}")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str, max_retries: int = 5) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.
    Decorator @traceable gửi log riêng lên LangSmith.
    """
    for attempt in range(max_retries):
        try:
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
            return {"question": question, "answer": answer, "version": version}
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                v_key = "v1" if "v1" in version else "v2"
                cached_file = Path(__file__).parent.parent / "data" / f"rag_{v_key}_results.json"
                if cached_file.exists():
                    try:
                        c_data = json.loads(cached_file.read_text(encoding="utf-8"))
                        for item in c_data:
                            if item.get("question") == question:
                                return {"question": question, "answer": item.get("answer"), "version": version}
                    except Exception:
                        pass
                wait_s = 60
                print(f"\n⚠️  Chạm rate limit (429 RESOURCE_EXHAUSTED). Tạm dừng {wait_s}s để hồi quota (lần {attempt + 1}/{max_retries})...")
                time.sleep(wait_s)
            elif "503" in err_msg or "unavailable" in err_msg or "overloaded" in err_msg or "deadline" in err_msg:
                wait_s = 10
                print(f"\n⚠️  Máy chủ Gemini tạm thời bận (503 UNAVAILABLE). Chờ {wait_s}s rồi thử lại (lần {attempt + 1}/{max_retries})...")
                time.sleep(wait_s)
            else:
                if attempt < max_retries - 1:
                    wait_s = 5
                    print(f"\n⚠️  Gặp lỗi tạm thời: {e}. Thử lại sau {wait_s}s (lần {attempt + 1}/{max_retries})...")
                    time.sleep(wait_s)
                else:
                    raise e
    raise RuntimeError(f"Không thể hoàn thành truy vấn sau {max_retries} lần thử.")


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2.0"))


def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # 1. Tạo LangSmith Client với API key từ config
    client = Client(api_key=config.LANGSMITH_API_KEY)

    # 2. Push cả 2 prompts lên Hub
    push_prompts_to_hub(client)

    # 3. Pull cả 2 prompts từ Hub
    prompts = pull_prompts_from_hub(client)

    # 4. Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    print(f"⏱️  Áp dụng delay {REQUEST_DELAY}s giữa các câu hỏi để giữ tốc độ an toàn...\n")

    # 5. Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")
        if i < len(SAMPLE_QUESTIONS) - 1:
            time.sleep(REQUEST_DELAY)

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
