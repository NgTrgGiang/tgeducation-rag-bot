"""
ingest.py - Đọc knowledge base JSON → tạo embeddings → lưu vào ChromaDB

Chạy: python ingest.py
"""
import json
import time
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, COLLECTION_NAME, KB_FILE


def load_knowledge_base(filepath: str) -> list[dict]:
    """Đọc file JSON knowledge base."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📄 Đã đọc {len(data)} entries từ {filepath}")
    return data


def build_document_text(entry: dict) -> str:
    """
    Tạo text tối ưu cho embedding từ một entry.
    Kết hợp nhiều trường để tăng khả năng retrieval.
    """
    parts = []

    # Title - quan trọng nhất cho semantic search
    parts.append(f"Tiêu đề: {entry['title']}")

    # Content - nội dung chính
    parts.append(f"Nội dung: {entry['content']}")

    # Summary - tóm tắt giúp embedding hiểu tổng quan
    parts.append(f"Tóm tắt: {entry['summary']}")

    # Typical questions - CỰC KỲ QUAN TRỌNG cho RAG
    # Giúp match câu hỏi user với câu hỏi mẫu
    if entry.get("typical_questions"):
        questions_text = " | ".join(entry["typical_questions"])
        parts.append(f"Câu hỏi thường gặp: {questions_text}")

    # Tags - từ khóa bổ sung
    if entry.get("tags"):
        parts.append(f"Từ khóa: {', '.join(entry['tags'])}")

    return "\n".join(parts)


def build_metadata(entry: dict) -> dict:
    """
    Tạo metadata cho ChromaDB filtering.
    ChromaDB chỉ hỗ trợ str, int, float, bool.
    """
    return {
        "id": entry["id"],
        "title": entry["title"],
        "category": entry["category"],
        "service": entry["service"],
        "student_level": entry["student_level"],
        "subject": entry["subject"],
        "intent": entry["intent"],
        "audience": entry["audience"],
        "priority": entry["priority"],
        "sensitivity": entry["sensitivity"],
        "source_type": entry["source_type"],
        "locale": entry["locale"],
        "escalation_required": entry["escalation_required"],
        "human_handoff_hint": entry.get("human_handoff_hint", ""),
        # Lưu summary riêng để hiển thị nhanh
        "summary": entry["summary"],
        # Lưu content gốc để trả về cho LLM
        "content": entry["content"],
    }


def ingest():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("🚀 TG Education RAG - Knowledge Base Ingestion")
    print("=" * 60)

    # 1. Load knowledge base
    entries = load_knowledge_base(KB_FILE)

    # 2. Initialize embedding model
    print(f"\n⏳ Đang tải embedding model: {EMBEDDING_MODEL}...")
    start = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Model loaded trong {time.time()-start:.1f}s")

    # 3. Build documents for embedding
    print("\n📝 Đang xây dựng documents cho embedding...")
    documents = []
    metadatas = []
    ids = []

    for entry in entries:
        doc_text = build_document_text(entry)
        metadata = build_metadata(entry)
        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(entry["id"])

    # 4. Generate embeddings
    print(f"\n⏳ Đang tạo embeddings cho {len(documents)} documents...")
    start = time.time()
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=8)
    embeddings_list = embeddings.tolist()
    print(f"✅ Embeddings created trong {time.time()-start:.1f}s")
    print(f"   Embedding dimension: {len(embeddings_list[0])}")

    # 5. Store in ChromaDB
    print(f"\n💾 Đang lưu vào ChromaDB tại {CHROMA_PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Xóa collection cũ nếu tồn tại (re-ingest)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Đã xóa collection cũ '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "TG Education K12 Customer Support Knowledge Base"}
    )

    # Add documents in batches
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings_list[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"   Đã thêm batch {i//batch_size + 1}: entries {i+1}-{end}")

    # 6. Verify
    count = collection.count()
    print(f"\n{'=' * 60}")
    print(f"✅ HOÀN TẤT! Đã ingest {count} documents vào ChromaDB")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Persist dir: {CHROMA_PERSIST_DIR}")
    print(f"{'=' * 60}")

    # Quick test
    print("\n🔍 Quick test - tìm kiếm 'học phí bao nhiêu'...")
    test_query = "học phí bao nhiêu"
    test_embedding = model.encode([test_query]).tolist()
    results = collection.query(
        query_embeddings=test_embedding,
        n_results=3,
    )
    print(f"   Top 3 kết quả:")
    for i, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
        meta = results["metadatas"][0][i]
        print(f"   {i+1}. [{doc_id}] {meta['title']} (distance: {distance:.4f})")


if __name__ == "__main__":
    ingest()
