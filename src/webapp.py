import re
from flask import Flask, render_template, request, jsonify

from src.vanilla_rag import vanilla_csv
from src.agentic_rag import agentic_csv

CSV_PATH = "data/Surgical_Wait_Times_20260207.csv"

app = Flask(__name__)


def _basic_sanitize(html: str) -> str:
    """
    Minimal safety: remove <script> blocks.
    (We still use innerHTML on the client.)
    """
    if not html:
        return ""
    html = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    return html


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        vanilla_html = vanilla_csv(question, CSV_PATH)
        agentic_html = agentic_csv(question, CSV_PATH)

        vanilla_html = _basic_sanitize(vanilla_html)
        agentic_html = _basic_sanitize(agentic_html)

        return jsonify({
            "vanilla_html": vanilla_html,
            "agentic_html": agentic_html,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # run: python -m src.webapp
    app.run(host="127.0.0.1", port=5000, debug=True)
