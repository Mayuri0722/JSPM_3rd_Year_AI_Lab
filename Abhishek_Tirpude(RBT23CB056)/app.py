import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# --- Gemini API Initialization ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("⚠️ GEMINI_API_KEY not found in .env file!")

    # Initialize Gemini client (new official SDK)
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Recommended stable fast model
    MODEL_NAME = "gemini-2.0-flash"

except Exception as e:
    print(f"❌ Error initializing Gemini client: {e}")
    client = None
    MODEL_NAME = None


# --- Chatbot Core Function ---
def get_chatbot_response(user_message):
    """
    Generates a response using Gemini with built-in sentiment analysis.
    """

    if not client:
        return "⚠️ Gemini client not initialized properly."

    system_prompt = (
        "You are a friendly and intelligent assistant. "
        "Analyze the user's message sentiment (Positive, Negative, Neutral, or Mixed) "
        "and then provide a helpful response. "
        "Respond strictly in JSON format with two keys: 'sentiment' and 'response'. "
        "Example: {'sentiment': 'Positive', 'response': 'That sounds wonderful!'}"
    )

    # Combine system prompt and user input
    final_prompt = f"{system_prompt}\n\nUser message: {user_message}"

    try:
        # Generate structured response
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=final_prompt
        )

        raw_text = response.text.strip()

        # Parse model JSON output
        try:
            data = json.loads(raw_text)
            sentiment = data.get("sentiment", "Unknown")
            answer = data.get("response", "Sorry, I couldn’t process that.")
            return f"🧭 **Sentiment:** {sentiment}\n\n💬 **Chatbot:** {answer}"

        except json.JSONDecodeError:
            # Handle plain-text fallback
            return f"⚠️ Model returned non-JSON response:\n{raw_text}"

    except Exception as e:
        return f"❌ Error during API call: {str(e)}"


# --- Flask Routes ---
@app.route('/')
def index():
    """Main chat page."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handles user input from frontend."""
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    bot_response = get_chatbot_response(user_message)
    return jsonify({"response": bot_response})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
