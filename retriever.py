"""
retriever.py - Tìm kiếm knowledge chunks liên quan từ ChromaDB

Dùng ChromaDB default embedding (nhẹ, không cần PyTorch)
"""
import chromadb
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K


class Retriever:
    """Knowledge base retriever using ChromaDB."""

    def __init__(self):
        print("⏳ Đang khởi tạo Retriever...")
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"✅ Retriever sẵn sàng! ({self.collection.count()} documents)")

    def search(
        self,
        query: str,
        top_k: int = None,
        category: str = None,
        service: str = None,
        student_level: str = None,
        subject: str = None,
        audience: str = None,
    ) -> list[dict]:
        """
        Tìm kiếm knowledge chunks phù hợp nhất.

        Args:
            query: Câu hỏi của người dùng
            top_k: Số kết quả trả về

        Returns:
            List[dict] với keys: id, title, content, summary, metadata, distance
        """
        if top_k is None:
            top_k = TOP_K

        # Build metadata filter
        where_filter = self._build_filter(category, service, student_level, subject, audience)

        # Query ChromaDB (tự tạo embedding cho query)
        kwargs = {
            "query_texts": [query],
            "n_results": top_k,
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            formatted.append({
                "id": results["ids"][0][i],
                "title": meta.get("title", ""),
                "content": meta.get("content", ""),
                "summary": meta.get("summary", ""),
                "category": meta.get("category", ""),
                "priority": meta.get("priority", ""),
                "intent": meta.get("intent", ""),
                "escalation_required": meta.get("escalation_required", False),
                "human_handoff_hint": meta.get("human_handoff_hint", ""),
                "distance": results["distances"][0][i],
                "document": results["documents"][0][i],
            })

        return formatted

    def _build_filter(self, category, service, student_level, subject, audience) -> dict | None:
        """Build ChromaDB where filter."""
        conditions = []
        if category:
            conditions.append({"category": category})
        if service:
            conditions.append({"service": service})
        if student_level:
            conditions.append({"student_level": student_level})
        if subject:
            conditions.append({"subject": subject})
        if audience:
            conditions.append({"audience": audience})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def format_context(self, results: list[dict]) -> str:
        """Format kết quả thành context string cho LLM."""
        if not results:
            return "Không tìm thấy thông tin liên quan."

        context_parts = []
        for i, r in enumerate(results, 1):
            part = f"""
--- Tài liệu {i} [{r['id']}] ---
Tiêu đề: {r['title']}
Danh mục: {r['category']}
Mức ưu tiên: {r['priority']}
Nội dung: {r['content']}
"""
            if r.get("escalation_required"):
                part += f"⚠️ Cần chuyển nhân viên: {r['human_handoff_hint']}\n"
            context_parts.append(part.strip())

        return "\n\n".join(context_parts)


# === CLI test ===
if __name__ == "__main__":
    retriever = Retriever()

    test_queries = [
        "Học phí bao nhiêu?",
        "Muốn đổi giáo viên thì sao?",
        "Tôi muốn hoàn tiền",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"📝 Query: {query}")
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"  [{r['id']}] {r['title']} (dist: {r['distance']:.4f})")
