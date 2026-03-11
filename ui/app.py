from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Backend API URL - can be configured via environment variable
API_URL = os.environ.get("API_URL", "http://localhost:8000")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.form.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        response = requests.post(f"{API_URL}/ask", json={"query": user_query})
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
