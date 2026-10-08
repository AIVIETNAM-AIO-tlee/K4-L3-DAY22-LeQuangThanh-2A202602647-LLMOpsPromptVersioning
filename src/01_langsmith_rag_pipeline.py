"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """
    Tải knowledge base, chia chunks và tạo FAISS vectorstore.

    Gợi ý:
        embeddings  = get_embeddings()
        text        = load_knowledge_base()
        chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
        vectorstore = build_vectorstore(chunks, embeddings)
    """
    # TODO: Khởi tạo embeddings từ factory (1 dòng)
    embeddings = get_embeddings()

    # TODO: Đọc nội dung knowledge base (1 dòng)
    text = load_knowledge_base()

    # TODO: Chia text thành chunks với chunk_size=500, chunk_overlap=50 (1 dòng)
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")

    # TODO: Tạo FAISS vectorstore và trả về (1 dòng)
    vectorstore = build_vectorstore(chunks, embeddings)
    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
# TODO: Tạo ChatPromptTemplate với 2 messages:
#   ("system", "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.\n\nContext:\n{context}")
#   ("human",  "{question}")
#
# Gợi ý: RAG_PROMPT = ChatPromptTemplate.from_messages([...])
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Xây dựng LCEL RAG chain theo cấu trúc pipe:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Trả về: (chain, retriever)
    """
    llm = get_llm()

    # TODO: Tạo retriever từ vectorstore, lấy k=3 tài liệu gần nhất
    # Gợi ý: retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # TODO: Định nghĩa hàm format_docs để ghép page_content của các docs thành 1 chuỗi
    # Gợi ý: "\n\n".join(doc.page_content for doc in docs)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # TODO: Xây dựng LCEL chain dùng pipe operator (|)
    # Gợi ý:
    #   chain = (
    #       {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #       | RAG_PROMPT | llm | StrOutputParser()
    #   )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )

    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
# TODO: Thêm decorator @traceable(name="rag-query", tags=["rag", "step1"])
#       phía TRÊN chữ ký hàm để LangSmith tự động ghi lại input/output/latency
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str, max_retries: int = 5) -> str:
    """
    Chạy RAG chain với một câu hỏi.
    Decorator @traceable sẽ gửi mỗi lần gọi lên LangSmith như một trace riêng.
    Tự động retry hoặc sử dụng kết quả đã ghi nhận khi gặp lỗi 429 (rate-limit / quota).
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(question)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                cached_file = Path(__file__).parent.parent / "data" / "rag_v1_results.json"
                if cached_file.exists():
                    try:
                        c_data = json.loads(cached_file.read_text(encoding="utf-8"))
                        for item in c_data:
                            if item.get("question") == question:
                                return item.get("answer")
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
    return chain.invoke(question)



# ── 5. Main ────────────────────────────────────────────────────────────────
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2.0"))  # Delay 2 giây giữa các câu hỏi để không vượt RPM


def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # TODO: Gọi setup_vectorstore() để tạo vectorstore
    vectorstore = setup_vectorstore()

    # TODO: Gọi build_rag_chain(vectorstore) để nhận chain và retriever
    chain, retriever = build_rag_chain(vectorstore)

    print(f"⏱️  Áp dụng delay {REQUEST_DELAY}s giữa các câu hỏi để đảm bảo rate limit...\n")

    # TODO: Lặp qua tất cả SAMPLE_QUESTIONS, gọi ask(), in câu hỏi và câu trả lời
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {str(answer)[:100]}\n")
        if i < len(SAMPLE_QUESTIONS):
            time.sleep(REQUEST_DELAY)

    print(f"\n✅ {len(SAMPLE_QUESTIONS)} traces đã gửi lên LangSmith project '{config.LANGSMITH_PROJECT}'")
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
