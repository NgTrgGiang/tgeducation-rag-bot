"""
chatbot.py - RAG Chatbot kết hợp Retriever + LLM (OpenRouter hoặc Ollama local)

Flow:
1. Nhận câu hỏi từ user
2. Retriever tìm top-K chunks liên quan từ ChromaDB
3. Xây dựng prompt với context
4. Gửi cho LLM API (OpenAI-compatible) để sinh câu trả lời
5. Trả về câu trả lời + sources
"""
from openai import OpenAI
from retriever import Retriever
from config import OPENROUTER_API_KEY, LLM_BASE_URL, LLM_MODEL, SYSTEM_PROMPT, TOP_K


class RAGChatbot:
    """RAG-powered chatbot for TG Education customer support."""

    def __init__(self):
        print("⏳ Đang khởi tạo RAG Chatbot...")

        # Init retriever
        self.retriever = Retriever()

        # Detect mode: local (Ollama) or cloud (OpenRouter)
        self.is_local = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL

        if self.is_local:
            # Ollama - không cần API key
            self.client = OpenAI(
                base_url=LLM_BASE_URL,
                api_key="ollama",  # Ollama không kiểm tra key
            )
            provider = "Ollama (local)"
        else:
            # OpenRouter - cần API key
            if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
                raise ValueError(
                    "❌ Chưa cấu hình OPENROUTER_API_KEY!\n"
                    "👉 Lấy API key tại: https://openrouter.ai/keys\n"
                    "👉 Hoặc dùng Ollama local: LLM_BASE_URL=http://localhost:11434/v1"
                )
            self.client = OpenAI(
                base_url=LLM_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
            provider = "OpenRouter"

        self.model = LLM_MODEL
        print(f"✅ RAG Chatbot sẵn sàng! (Model: {self.model} via {provider})")

    def chat(self, user_message: str, chat_history: list = None) -> dict:
        """
        Xử lý câu hỏi từ user.

        Args:
            user_message: Câu hỏi của khách hàng
            chat_history: Lịch sử chat (optional)

        Returns:
            dict với keys: answer, sources, escalation_needed, handoff_hint
        """
        # 1. Retrieve relevant documents
        results = self.retriever.search(user_message, top_k=TOP_K)

        # 2. Build context from retrieved documents
        context = self.retriever.format_context(results)

        # 3. Check if escalation is needed
        escalation_needed = any(r.get("escalation_required") for r in results)
        handoff_hints = [
            r["human_handoff_hint"]
            for r in results
            if r.get("escalation_required") and r.get("human_handoff_hint")
        ]

        # 4. Build messages for OpenAI-compatible API
        messages = self._build_messages(user_message, context, chat_history)

        # 5. Call OpenRouter
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại sau.\n(Lỗi: {str(e)})"

        # 6. Build sources list
        sources = [
            {"id": r["id"], "title": r["title"], "category": r["category"]}
            for r in results[:3]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "escalation_needed": escalation_needed,
            "handoff_hint": handoff_hints[0] if handoff_hints else "",
        }

    def _build_messages(self, question: str, context: str, chat_history: list = None) -> list:
        """Xây dựng messages array cho OpenAI-compatible API."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add chat history
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add context + current question
        user_content = f"""CONTEXT (Thông tin từ knowledge base):
{context}

CÂU HỎI CỦA KHÁCH HÀNG:
{question}"""

        messages.append({"role": "user", "content": user_content})

        return messages


# === CLI test mode ===
if __name__ == "__main__":
    bot = RAGChatbot()

    print("\n" + "=" * 60)
    print("🤖 TG Education RAG Chatbot - CLI Mode")
    print(f"   Model: {bot.model} (via OpenRouter)")
    print("   Gõ 'quit' để thoát")
    print("=" * 60)

    history = []
    while True:
        question = input("\n👤 Bạn: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("👋 Tạm biệt!")
            break
        if not question:
            continue

        result = bot.chat(question, history)

        print(f"\n🤖 Trợ lý: {result['answer']}")

        if result["sources"]:
            print(f"\n📚 Nguồn tham khảo:")
            for s in result["sources"]:
                print(f"   - [{s['id']}] {s['title']}")

        if result["escalation_needed"]:
            print(f"\n⚠️ Lưu ý: {result['handoff_hint']}")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})
