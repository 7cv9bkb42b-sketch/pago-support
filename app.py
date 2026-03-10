import os
import re
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
HELPSCOUT_APP_ID   = os.environ.get("HELPSCOUT_APP_ID", "")
HELPSCOUT_APP_SECRET = os.environ.get("HELPSCOUT_APP_SECRET", "")
QDRANT_URL         = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY     = os.environ.get("QDRANT_API_KEY", "")
COLLECTION         = "pago_conversations"

SYSTEM_PROMPT = """You are a customer support agent for Pago International, an Australian ergonomic chair company that sells primarily through Officeworks stores across Australia.

## About Pago International
- We sell ergonomic office chairs (Radar III, Electra, Matrix Advance, Nest, Zeke, Flash)
- Our chairs are sold through Officeworks retail stores across Australia
- Website: www.pagointernational.com.au
- Tagline: "Your comfort for life"
- We are based in Australia and ship via Australia Post

## Your Tone and Style
- Always greet the customer by their first name: "Hi [Name],"
- Be friendly, concise, and professional
- Always apologise: "Apologies for the inconvenience."
- Sign off every reply with: "Best Regards"
- Never be defensive, take ownership of problems quickly
- Keep replies short and actionable

## How to Handle Common Issues
1. FAULTY PARTS - Ask for photo of issue AND photo of label under seat. Issue RA for Officeworks exchange OR send replacement parts.
2. SPARE PARTS - Ask for: shipping address + mobile number + photo of chair label.
3. DELIVERY - We ship via Australia Post. Tracking: https://auspost.com.au/mypost/track/details/
4. WARRANTY - Ask for Officeworks receipt OR photo of sticker under seat.
5. ASSEMBLY - Gas lift removal: firm force, friction only. Wobbling: tighten all screws.
6. AFTER RESOLUTION - Always ask customer to leave an Officeworks review.

Only write the reply, nothing else."""


def get_embedding(text):
    response = requests.post(
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        json={"inputs": text[:512]},
        timeout=15,
    )
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and isinstance(result[0], list):
            return result[0]
        elif isinstance(result, list) and isinstance(result[0], float):
            return result
    return None


def search_similar(message, n=3):
    embedding = get_embedding(message)
    if not embedding:
        return []
    response = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={"vector": embedding, "limit": n, "with_payload": True},
        timeout=10,
    )
    if response.status_code != 200:
        return []
    results = response.json().get("result", [])
    return [r["payload"] for r in results if r.get("score", 0) > 0.3]


def draft_reply(customer_message, customer_name=""):
    similar = search_similar(customer_message)
    examples = ""
    if similar:
        examples = "\n\nHere are similar past conversations:\n"
        for i, conv in enumerate(similar, 1):
            examples += f"\nExample {i}:\nCustomer: {conv['customer_message'][:300]}\nAgent: {conv['agent_reply'][:300]}\n"
    name_hint = f"The customer name is {customer_name}. " if customer_name else ""
    prompt = f"{name_hint}{examples}\n\nDraft a reply to:\n\n{customer_message}"
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": "claude-sonnet-4-20250514", "max_tokens": 1000, "system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def get_helpscout_token():
    response = requests.post(
        "https://api.helpscout.net/v2/auth/token",
        data={"grant_type": "client_credentials", "client_id": HELPSCOUT_APP_ID, "client_secret": HELPSCOUT_APP_SECRET},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def post_note(conversation_id, text):
    token = get_helpscout_token()
    response = requests.post(
        f"https://api.helpscout.net/v2/conversations/{conversation_id}/notes",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"text": text},
        timeout=15,
    )
    response.raise_for_status()


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json or {}
    event_type = request.headers.get("X-HelpScout-Event", "")
    print(f"Event: {event_type}")
    if event_type not in ("convo.created", "convo.customer.reply.created"):
        return jsonify({"status": "ignored"}), 200
    try:
        conv = data.get("conversation", {})
        conversation_id = conv.get("id")
        customer = conv.get("customer", {})
        full_name = customer.get("fullName", "") or ""
        customer_name = full_name.split()[0] if full_name else customer.get("fname", "")
        threads = conv.get("_embedded", {}).get("threads", [])
        customer_message = ""
        for thread in reversed(threads):
            if thread.get("type") == "customer":
                body = re.sub(r"<[^>]+>", " ", thread.get("body", ""))
                body = re.sub(r"\s+", " ", body).strip()
                if len(body) > 30:
                    customer_message = body
                    break
        if not customer_message:
            customer_message = conv.get("preview", "")
        if not customer_message:
            return jsonify({"status": "no message"}), 200
        draft = draft_reply(customer_message, customer_name)
        post_note(conversation_id, f"AI Draft Reply:\n\n{draft}")
        print(f"Note posted to {conversation_id}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
