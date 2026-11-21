
import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import errors

load_dotenv()

app = Flask(__name__)

SUPPORT_INSTRUCTIONS = """You are a helpful AI customer support assistant for an online store.
Your responsibilities:
1. Answer customer questions politely and clearly.
2. Help with product enquiries, delivery, returns and refunds using only known information.
3. Keep answers simple and friendly.
4. If you do not know something, do not invent information.
5. Never claim to have checked an order or Shopify data.
6. Explain that live order tracking will be available once the store integration is connected."""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON request body is required"}), 400

    message = data.get("message")
    if not isinstance(message, str):
        return jsonify({"error": "Message is required"}), 400

    message = message.strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if len(message) > 500:
        return jsonify({"error": "Message is too long"}), 400

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return jsonify({"error": "Gemini is not configured. Set GEMINI_API_KEY in .env."}), 503

    try:
        client = genai.Client(api_key=api_key)
        configured_model = os.getenv("GEMINI_MODEL")
        model_names = [
            configured_model or "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
        ]
        response = None

        for model_name in dict.fromkeys(model_names):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"{SUPPORT_INSTRUCTIONS}\n\nCustomer message: {message}",
                )
                break
            except errors.APIError as error:
                app.logger.warning("Gemini model %s failed: %s", model_name, error)
                continue
    except errors.APIError:
        app.logger.exception("Gemini API request failed")
        return jsonify({"error": "Gemini could not process your message. Please try again."}), 502

    reply = getattr(response, "text", None)
    if not reply or not reply.strip():
        app.logger.error("Gemini returned an empty response")
        return jsonify({"error": "Gemini returned an empty response. Please try again."}), 502

    return jsonify({"reply": reply.strip()})


@app.errorhandler(500)
def handle_internal_server_error(_error):
    return jsonify({"error": "An unexpected server error occurred. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
