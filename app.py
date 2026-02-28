"""
app.py - Giao diện chat Gradio cho TG Education RAG Chatbot
Chạy: python app.py
"""
import gradio as gr
from chatbot import RAGChatbot

# Global chatbot instance
bot = None


def initialize():
    """Khởi tạo chatbot."""
    global bot
    bot = RAGChatbot()


def respond(message: str, chat_history: list):
    """Xử lý tin nhắn từ user."""
    if not message.strip():
        return "", chat_history

    # Convert Gradio history format to our format
    history = []
    for user_msg, bot_msg in chat_history:
        history.append({"role": "user", "content": user_msg})
        if bot_msg:
            history.append({"role": "assistant", "content": bot_msg})

    # Get response from chatbot
    result = bot.chat(message, history)

    # Build response text
    response = result["answer"]

    # Add sources
    if result["sources"]:
        response += "\n\n📚 **Nguồn tham khảo:**"
        for s in result["sources"]:
            response += f"\n- `{s['id']}` {s['title']}"

    # Add escalation warning
    if result["escalation_needed"]:
        response += f"\n\n⚠️ **Lưu ý:** {result['handoff_hint']}"

    chat_history.append((message, response))
    return "", chat_history


def create_app():
    """Tạo Gradio app."""
    with gr.Blocks(
        title="TG Education - Trợ lý AI",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
        ),
        css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        .header { text-align: center; padding: 20px 0; }
        .header h1 { color: #1e40af; margin-bottom: 5px; }
        .header p { color: #6b7280; font-size: 14px; }
        footer { display: none !important; }
        """
    ) as app:
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎓 TG Education - Trợ lý AI</h1>
            <p>Hỗ trợ tư vấn về đăng ký, học phí, lịch học, giáo viên và các dịch vụ tại TG Education</p>
        </div>
        """)

        # Chat interface
        chatbot_ui = gr.Chatbot(
            label="Chat",
            height=500,
            show_label=False,
            avatar_images=(None, "https://em-content.zobj.net/source/apple/391/robot_1f916.png"),
            bubble_full_width=False,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Nhập câu hỏi... (VD: Học phí bao nhiêu?)",
                show_label=False,
                scale=9,
                container=False,
            )
            send_btn = gr.Button("Gửi", variant="primary", scale=1)

        # Example questions
        gr.Examples(
            examples=[
                "Học phí bao nhiêu?",
                "Làm sao đăng ký học thử?",
                "Muốn đổi giáo viên thì sao?",
                "Chính sách hoàn tiền như thế nào?",
                "Con tôi muốn nghỉ 1 buổi, báo trước bao lâu?",
                "Có khóa luyện thi vào lớp 10 không?",
                "Không vào được Zoom, phải làm sao?",
                "Địa chỉ trung tâm ở đâu?",
            ],
            inputs=msg_input,
            label="💡 Câu hỏi mẫu",
        )

        # Clear button
        clear_btn = gr.Button("🗑️ Xóa lịch sử chat", variant="secondary", size="sm")

        # Event handlers
        msg_input.submit(respond, [msg_input, chatbot_ui], [msg_input, chatbot_ui])
        send_btn.click(respond, [msg_input, chatbot_ui], [msg_input, chatbot_ui])
        clear_btn.click(lambda: (None, []), outputs=[msg_input, chatbot_ui])

        # Footer info
        gr.HTML("""
        <div style="text-align: center; padding: 15px; color: #9ca3af; font-size: 12px;">
            Powered by RAG (Retrieval-Augmented Generation) | ChromaDB + Gemini AI<br>
            ⚠️ Thông tin chỉ mang tính tham khảo. Liên hệ hotline 1900-xxxx để được hỗ trợ chính thức.
        </div>
        """)

    return app


if __name__ == "__main__":
    initialize()
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
