"""
messenger_bot.py - Facebook Messenger Webhook cho TG Education RAG Chatbot

Chạy local:  python messenger_bot.py
Test:        ngrok http 5000

Flow:
  Messenger → Facebook Server → Webhook (file này) → RAG Chatbot → Messenger
"""
import os
import json
import hashlib
import hmac
import logging
from flask import Flask, request, jsonify
import requests
from chatbot import RAGChatbot
from config import OPENROUTER_API_KEY

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# === Flask App ===
app = Flask(__name__)

# === Facebook Config ===
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "")
VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "tgeducation_verify_2026")
APP_SECRET = os.getenv("FB_APP_SECRET", "")

# === Messenger API ===
FB_API_URL = "https://graph.facebook.com/v21.0/me/messages"

# === In-memory chat history (per user) ===
# Production nên dùng Redis hoặc database
chat_histories: dict[str, list] = {}
MAX_HISTORY = 6  # Giữ 6 tin nhắn gần nhất

# === RAG Chatbot (lazy init) ===
bot: RAGChatbot = None


def get_bot() -> RAGChatbot:
    """Lazy initialization của chatbot."""
    global bot
    if bot is None:
        logger.info("Đang khởi tạo RAG Chatbot...")
        bot = RAGChatbot()
        logger.info("RAG Chatbot sẵn sàng!")
    return bot


# =============================================
# WEBHOOK VERIFICATION
# Facebook gửi GET request để xác minh webhook
# =============================================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Xác minh webhook với Facebook."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("✅ Webhook verified successfully!")
        return challenge, 200
    else:
        logger.warning("❌ Webhook verification failed!")
        return "Forbidden", 403


# =============================================
# RECEIVE MESSAGES
# Facebook gửi POST request khi có tin nhắn mới
# =============================================
@app.route("/webhook", methods=["POST"])
def receive_message():
    """Nhận và xử lý tin nhắn từ Messenger."""
    body = request.get_json()

    if body.get("object") != "page":
        return "Not Found", 404

    # Xử lý từng entry (có thể có nhiều events cùng lúc)
    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")

            if not sender_id:
                continue

            # Xử lý tin nhắn text
            if "message" in event and "text" in event["message"]:
                message_text = event["message"]["text"]
                logger.info(f"📩 Nhận tin nhắn từ {sender_id}: {message_text}")

                # Gửi typing indicator
                send_typing(sender_id, "typing_on")

                # Xử lý bằng RAG chatbot
                handle_message(sender_id, message_text)

                # Tắt typing indicator
                send_typing(sender_id, "typing_off")

            # Xử lý postback (nút bấm)
            elif "postback" in event:
                payload = event["postback"].get("payload", "")
                logger.info(f"🔘 Postback từ {sender_id}: {payload}")
                handle_postback(sender_id, payload)

    return "OK", 200


# =============================================
# MESSAGE HANDLER
# =============================================
def handle_message(sender_id: str, message_text: str):
    """Xử lý tin nhắn bằng RAG chatbot."""
    # Kiểm tra lệnh đặc biệt
    lower_text = message_text.lower().strip()

    if lower_text in ["hi", "hello", "xin chào", "chào"]:
        send_welcome(sender_id)
        return

    if lower_text in ["menu", "help", "trợ giúp"]:
        send_menu(sender_id)
        return

    if lower_text in ["reset", "xóa", "làm mới"]:
        chat_histories.pop(sender_id, None)
        send_text(sender_id, "🔄 Đã xóa lịch sử chat. Bạn có thể đặt câu hỏi mới!")
        return

    # Lấy chat history
    history = chat_histories.get(sender_id, [])

    # Gọi RAG chatbot
    try:
        chatbot = get_bot()
        result = chatbot.chat(message_text, history)

        # Xây dựng câu trả lời (bỏ markdown cho Messenger)
        answer = result["answer"]
        answer = answer.replace("**", "").replace("##", "").replace("# ", "")

        # Gửi trả lời (chia nhỏ nếu quá dài)
        send_long_text(sender_id, answer)

        # Lưu history
        history.append({"role": "user", "content": message_text})
        history.append({"role": "assistant", "content": result["answer"]})
        # Giữ tối đa MAX_HISTORY messages
        chat_histories[sender_id] = history[-MAX_HISTORY:]

    except Exception as e:
        logger.error(f"Lỗi xử lý tin nhắn: {e}", exc_info=True)
        send_text(
            sender_id,
            "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau hoặc liên hệ hotline 1900-xxxx."
        )


def handle_postback(sender_id: str, payload: str):
    """Xử lý nút bấm."""
    responses = {
        "GET_STARTED": lambda: send_welcome(sender_id),
        "MENU_PRICING": lambda: handle_message(sender_id, "Học phí bao nhiêu?"),
        "MENU_TRIAL": lambda: handle_message(sender_id, "Đặt lịch học thử"),
        "MENU_SCHEDULE": lambda: handle_message(sender_id, "Đổi lịch học"),
        "MENU_CONTACT": lambda: send_text(
            sender_id,
            "📞 Hotline: 1900-xxxx\n📧 Email: support@tgeducation.vn\n💬 Zalo OA: TG Education\n\n🏢 Hà Nội: 123 Nguyễn Trãi, Thanh Xuân\n🏢 TP.HCM: 456 Lê Văn Sỹ, Quận 3"
        ),
    }
    action = responses.get(payload)
    if action:
        action()
    else:
        send_text(sender_id, "Xin lỗi, tôi chưa hiểu yêu cầu. Bạn có thể gõ câu hỏi trực tiếp.")


# =============================================
# SEND FUNCTIONS
# =============================================
def send_text(recipient_id: str, text: str):
    """Gửi tin nhắn text đơn giản."""
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text},
        "messaging_type": "RESPONSE",
    }
    _call_send_api(payload)


def send_long_text(recipient_id: str, text: str, max_len: int = 2000):
    """Gửi text dài, chia thành nhiều tin nhắn nếu cần."""
    if len(text) <= max_len:
        send_text(recipient_id, text)
        return

    # Chia theo dòng, không cắt giữa chừng
    parts = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            parts.append(current.strip())
            current = line
        else:
            current += "\n" + line if current else line
    if current:
        parts.append(current.strip())

    for part in parts:
        send_text(recipient_id, part)


def send_typing(recipient_id: str, action: str):
    """Gửi typing indicator (typing_on / typing_off)."""
    payload = {
        "recipient": {"id": recipient_id},
        "sender_action": action,
    }
    _call_send_api(payload)


def send_welcome(sender_id: str):
    """Gửi tin nhắn chào mừng với quick replies."""
    payload = {
        "recipient": {"id": sender_id},
        "message": {
            "text": "Xin chào! 👋 Tôi là trợ lý AI của TG Education.\n\nTôi có thể giúp bạn về:\n📚 Học phí & ưu đãi\n📝 Đăng ký & học thử\n📅 Lịch học & nghỉ phép\n👨‍🏫 Giáo viên & chất lượng\n💻 Hỗ trợ kỹ thuật\n\nHãy đặt câu hỏi hoặc chọn chủ đề bên dưới!",
            "quick_replies": [
                {"content_type": "text", "title": "💰 Học phí", "payload": "ask_pricing"},
                {"content_type": "text", "title": "📝 Học thử", "payload": "ask_trial"},
                {"content_type": "text", "title": "📅 Lịch học", "payload": "ask_schedule"},
                {"content_type": "text", "title": "📞 Liên hệ", "payload": "ask_contact"},
            ],
        },
        "messaging_type": "RESPONSE",
    }
    _call_send_api(payload)


def send_menu(sender_id: str):
    """Gửi menu dạng buttons."""
    payload = {
        "recipient": {"id": sender_id},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "button",
                    "text": "📋 Menu chính - Chọn chủ đề bạn cần hỗ trợ:",
                    "buttons": [
                        {"type": "postback", "title": "💰 Xem học phí", "payload": "MENU_PRICING"},
                        {"type": "postback", "title": "📝 Đặt lịch học thử", "payload": "MENU_TRIAL"},
                        {"type": "postback", "title": "📞 Liên hệ", "payload": "MENU_CONTACT"},
                    ],
                },
            }
        },
        "messaging_type": "RESPONSE",
    }
    _call_send_api(payload)


def _call_send_api(payload: dict):
    """Gọi Facebook Send API."""
    if not PAGE_ACCESS_TOKEN:
        logger.warning("⚠️ FB_PAGE_ACCESS_TOKEN chưa được cấu hình!")
        return

    headers = {"Content-Type": "application/json"}
    params = {"access_token": PAGE_ACCESS_TOKEN}

    try:
        resp = requests.post(FB_API_URL, params=params, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Facebook API error: {resp.status_code} - {resp.text}")
        else:
            logger.debug(f"Message sent successfully")
    except Exception as e:
        logger.error(f"Send API error: {e}")


# =============================================
# SETUP PERSISTENT MENU & GET STARTED
# Chạy 1 lần để cấu hình trên Facebook
# =============================================
def setup_messenger_profile():
    """Cấu hình Persistent Menu và Get Started button."""
    if not PAGE_ACCESS_TOKEN:
        print("❌ Cần FB_PAGE_ACCESS_TOKEN để setup!")
        return

    url = "https://graph.facebook.com/v21.0/me/messenger_profile"
    headers = {"Content-Type": "application/json"}
    params = {"access_token": PAGE_ACCESS_TOKEN}

    profile = {
        "get_started": {"payload": "GET_STARTED"},
        "greeting": [
            {
                "locale": "default",
                "text": "Xin chào {{user_full_name}}! 👋\nTôi là trợ lý AI của TG Education. Nhấn 'Bắt đầu' để tôi hỗ trợ bạn!"
            }
        ],
        "persistent_menu": [
            {
                "locale": "default",
                "composer_input_disabled": False,
                "call_to_actions": [
                    {"type": "postback", "title": "💰 Xem học phí", "payload": "MENU_PRICING"},
                    {"type": "postback", "title": "📝 Đặt lịch học thử", "payload": "MENU_TRIAL"},
                    {"type": "postback", "title": "📅 Đổi lịch học", "payload": "MENU_SCHEDULE"},
                    {"type": "postback", "title": "📞 Liên hệ", "payload": "MENU_CONTACT"},
                    {
                        "type": "web_url",
                        "title": "🌐 Website",
                        "url": "https://tgeducation.vn",
                    },
                ],
            }
        ],
    }

    resp = requests.post(url, params=params, headers=headers, json=profile, timeout=30)
    if resp.status_code == 200:
        print("✅ Messenger Profile đã được cấu hình!")
    else:
        print(f"❌ Lỗi: {resp.status_code} - {resp.text}")


# =============================================
# HEALTH CHECK
# =============================================
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "TG Education RAG Chatbot",
        "messenger": "active",
    })


# =============================================
# AUTO INGEST (for fresh deploy)
# =============================================
def auto_ingest_if_needed():
    """Tự động chạy ingestion nếu ChromaDB chưa có data."""
    from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
    import chromadb

    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            logger.info(f"✅ ChromaDB đã có {collection.count()} documents, bỏ qua ingestion.")
            return
    except Exception:
        pass

    logger.info("⚠️ ChromaDB trống, đang chạy ingestion tự động...")
    from ingest import ingest
    ingest()
    logger.info("✅ Ingestion hoàn tất!")


# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_messenger_profile()
    else:
        logger.info("=" * 50)
        logger.info("🚀 TG Education Messenger Bot")
        logger.info("=" * 50)

        # Auto ingest if needed (first deploy)
        auto_ingest_if_needed()

        # Pre-load chatbot
        get_bot()

        # Run Flask server
        port = int(os.getenv("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)
