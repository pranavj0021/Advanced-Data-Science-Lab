"""
Assignment 07/08: Flask REST API Server for Transformer Models
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon

Exposes RESTful endpoints for:
1. Text Summarization (/api/summarize)
2. Question Answering (/api/qa)
3. Automated Resume Screening (/api/screen_resume)
"""

import os
import sys
from flask import Flask, request, jsonify
from transformer_engine import nlp_engine

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
        "service": "HuggingFace Transformer NLP REST API",
        "institution": "Sanjivani College of Engineering, Kopargaon",
        "department": "Department of Information Technology",
        "practical": "Practical 07/08 - Transformer Summarization & Question Answering",
        "endpoints": {
            "GET /": "Service Information",
            "GET /health": "Health Check",
            "POST /api/summarize": "Text Summarization Endpoint",
            "POST /api/qa": "Question Answering Endpoint",
            "POST /api/screen_resume": "Resume Screening & Evaluation Endpoint"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "transformers_available": nlp_engine.models_loaded
    })

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        max_length = int(data.get("max_length", 130))
        min_length = int(data.get("min_length", 30))
        
        if not text:
            return jsonify({"success": False, "error": "Field 'text' is required."}), 400
            
        result = nlp_engine.summarize_text(text, max_length=max_length, min_length=min_length)
        return jsonify({"success": True, "data": result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/qa', methods=['POST'])
def api_qa():
    try:
        data = request.get_json(force=True)
        context = data.get("context", "")
        question = data.get("question", "")
        
        if not context or not question:
            return jsonify({"success": False, "error": "Both 'context' and 'question' are required."}), 400
            
        result = nlp_engine.answer_question(context, question)
        return jsonify({"success": True, "data": result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/screen_resume', methods=['POST'])
def api_screen_resume():
    try:
        data = request.get_json(force=True)
        resume_text = data.get("resume_text", "")
        
        if not resume_text:
            return jsonify({"success": False, "error": "Field 'resume_text' is required."}), 400
            
        result = nlp_engine.screen_resume(resume_text)
        return jsonify({"success": True, "data": result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask Transformer API Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
