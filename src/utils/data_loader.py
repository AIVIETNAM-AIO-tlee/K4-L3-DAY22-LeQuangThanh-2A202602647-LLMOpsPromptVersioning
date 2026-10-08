"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings, cache_dir: str = None, force_rebuild: bool = False):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.
    Hỗ trợ cache vào đĩa để tránh tốn quota embed lại nhiều lần,
    chia batch nhỏ và tự động retry khi chạm rate-limit (429).

    Args:
        chunks       : list[str] — danh sách text chunks đã chia
        embeddings   : Embeddings instance (từ get_embeddings())
        cache_dir    : đường dẫn thư mục lưu/tải cache FAISS (mặc định: data/faiss_index)
        force_rebuild: nếu True, ép tạo lại thay vì đọc từ cache

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    import time
    from langchain_community.vectorstores import FAISS

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "faiss_index"
    else:
        cache_dir = Path(cache_dir)

    # 1. Kiểm tra cache đã tồn tại chưa
    if not force_rebuild and (cache_dir / "index.faiss").exists() and (cache_dir / "index.pkl").exists():
        print(f"📦 Đang tải FAISS vectorstore từ cache ({cache_dir}) ...")
        vectorstore = FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)
        print("✅ Đã tải FAISS vectorstore từ cache thành công.")
        return vectorstore

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")

    # 2. Chia nhỏ thành các batch (50 chunks/batch) để kiểm soát quota RPM
    batch_size = 50
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"   Đang xử lý batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(batch, embeddings)
                else:
                    vectorstore.add_texts(batch)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    wait_s = 60
                    print(f"\n⚠️  Chạm rate limit (429 RESOURCE_EXHAUSTED). Tạm dừng {wait_s}s để quota reset (lần {attempt + 1}/{max_retries})...")
                    time.sleep(wait_s)
                else:
                    raise e
        else:
            raise RuntimeError(f"Không thể index batch {batch_num} sau {max_retries} lần thử.")

        # Delay nhỏ giữa các batch để giảm áp lực lên API
        if i + batch_size < len(chunks):
            time.sleep(2)

    # 3. Lưu index vào cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))
        print(f"💾 Đã lưu FAISS index vào cache: {cache_dir}")
    except Exception as e:
        print(f"⚠️  Không thể lưu FAISS cache: {e}")

    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore

