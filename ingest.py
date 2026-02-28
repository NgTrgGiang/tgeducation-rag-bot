"""
ingest.py - Đọc knowledge base JSON → tạo embeddings → lưu vào ChromaDB

Dùng ChromaDB default embedding (onnxruntime - nhẹ, không cần PyTorch)

Chạy: python ingest.py
"""
import json
import time
import chromadb
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, KB_FILE


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

    # Title - quan trọng nhất
    parts.append(f"Tiêu đề: {entry['title']}")

    # Content - nội dung chính
    parts.append(f"Nội dung: {entry['content']}")

    # Summary
    parts.append(f"Tóm tắt: {entry['summary']}")

    # Typical questions - giúp match câu hỏi user
    if entry.get("typical_questions"):
        questions_text = " | ".join(entry["typical_questions"])
        parts.append(f"Câu hỏi thường gặp: {questions_text}")

    # Tags
    if entry.get("tags"):
        parts.append(f"Từ khóa: {', '.join(entry['tags'])}")

    return "\n".join(parts)


def build_metadata(entry: dict) -> dict:
    """Tạo metadata cho ChromaDB filtering."""
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
        "summary": entry["summary"],
        "content": entry["content"],
    }


def ingest():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("🚀 TG Education RAG - Knowledge Base Ingestion")
    print("=" * 60)

    # 1. Load knowledge base
    entries = load_knowledge_base(KB_FILE)

    # 2. Build documents
    print("\n📝 Đang xây dựng documents...")
    documents = []
    metadatas = []
    ids = []

    for entry in entries:
        doc_text = build_document_text(entry)
        metadata = build_metadata(entry)
        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(entry["id"])

    # 3. Store in ChromaDB (ChromaDB tự tạo embedding bằng default model)
    print(f"\n💾 Đang lưu vào ChromaDB tại {CHROMA_PERSIST_DIR}...")
    print("   (Sử dụng ChromaDB default embedding - onnxruntime)")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Xóa collection cũ nếu tồn tại
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Đã xóa collection cũ '{COLLECTION_NAME}'")
    except Exception:
        pass

    # Tạo collection MỚI - ChromaDB sẽ tự dùng default embedding function
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "TG Education K12 Customer Support Knowledge Base"}
    )

    # Add documents (ChromaDB tự tạo embeddings)
    start = time.time()
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"   Đã thêm batch {i//batch_size + 1}: entries {i+1}-{end}")

    print(f"   Embeddings created trong {time.time()-start:.1f}s")

    # 4. Verify
    count = collection.count()
    print(f"\n{'=' * 60}")
    print(f"✅ HOÀN TẤT! Đã ingest {count} documents vào ChromaDB")
    print(f"{'=' * 60}")

    # Quick test
    print("\n🔍 Quick test - tìm kiếm 'học phí bao nhiêu'...")
    results = collection.query(
        query_texts=["học phí bao nhiêu"],
        n_results=3,
    )
    print(f"   Top 3 kết quả:")
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"   {i+1}. [{doc_id}] {meta['title']} (distance: {dist:.4f})")


if __name__ == "__main__":
    ingest()
