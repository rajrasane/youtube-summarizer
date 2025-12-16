from flask import Flask, request, jsonify
from flask_cors import CORS

from pipeline import pipeline  # your existing pipeline.py

app = Flask(__name__)
CORS(app)  # allow frontend JS calls

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    youtube_url = data.get("url")

    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400

    try:
        summary = pipeline(youtube_url)
        return jsonify({
            "success": True,
            "summary": summary
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
