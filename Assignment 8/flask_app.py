"""
Assignment 08: Flask REST API Server for Conversational AI Chatbots & Prompt Engineering
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon

Exposes multi-turn chatbot endpoints and prompt testing services.
"""

import os
import sys
from flask import Flask, request, jsonify
from chatbot_engine import chatbot_engine

app = Flask(__name__)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "service": "Conversational AI & Prompt Engineering REST API",
        "institution": "Sanjivani College of Engineering, Kopargaon",
        "practical": "Practical No. 08 - Conversational AI & Prompt Engineering",
        "endpoints": {
            "GET /": "Service Info",
            "GET /health": "Health Check",
            "POST /chat/college": "College Inquiry Multi-Turn Chatbot",
            "POST /chat/healthcare": "Healthcare Assistant Chatbot",
            "POST /prompt/test": "Prompt Engineering Playground API"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "chatbot-engine"})

@app.route('/chat/college', methods=['POST'])
def chat_college():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
        session_id = data.get("session_id", "default_college")

        if not message:
            return jsonify({"success": False, "error": "Message parameter is required."}), 400

        result = chatbot_engine.college_chat(message, session_id=session_id)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/chat/healthcare', methods=['POST'])
def chat_healthcare():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
        session_id = data.get("session_id", "default_health")

        if not message:
            return jsonify({"success": False, "error": "Message parameter is required."}), 400

        result = chatbot_engine.healthcare_chat(message, session_id=session_id)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/prompt/test', methods=['POST'])
def prompt_test():
    try:
        data = request.get_json(force=True)
        technique = data.get("technique", "zero-shot")
        prompt = data.get("prompt", "What are the B.Tech course fees?")

        result = chatbot_engine.execute_prompt_technique(technique, prompt)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask Chatbot API Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
