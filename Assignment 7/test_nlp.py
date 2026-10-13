"""
Assignment 07/08: Automated NLP Integration Test Suite
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon

Validates HuggingFace Transformer summarization, QA, and REST endpoints.
"""

import sys
import json
import requests
from transformer_engine import nlp_engine

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_engine_directly():
    print("=" * 60)
    print("Testing Transformer Engine Direct Module Execution")
    print("=" * 60)
    
    doc = """
    The Transformer architecture was introduced by Vaswani et al. in the landmark 2017 paper titled Attention Is All You Need.
    It replaces recurrent neural networks with self-attention mechanisms, enabling parallelization and long-range contextual modeling.
    """
    
    # 1. Summarization
    summ = nlp_engine.summarize_text(doc)
    print(f"\n[Summarization Test]")
    print(f"Summary: {summ['summary_text']}")
    print(f"Method: {summ['method']}")
    
    # 2. Question Answering
    qa = nlp_engine.answer_question(context=doc, question="When was Transformer introduced?")
    print(f"\n[Question Answering Test]")
    print(f"Question: {qa['question']}")
    print(f"Answer: {qa['answer']}")
    print(f"Score: {qa['score']}")
    
    return True

def test_api_servers(base_url="http://127.0.0.1:5000"):
    print("\n" + "=" * 60)
    print(f"Testing REST API Endpoint Server at {base_url}")
    print("=" * 60)
    
    try:
        r_health = requests.get(f"{base_url}/health", timeout=3)
        print(f"[GET /health] Status: {r_health.status_code}")
        print(f"Response: {r_health.json()}")
    except Exception as e:
        print(f"❌ Server connection failed at {base_url}: {e}")
        return False
        
    # Summarize API
    r_sum = requests.post(f"{base_url}/api/summarize", json={"text": "Transformers process text in parallel.", "max_length": 50, "min_length": 10})
    print(f"\n[POST /api/summarize] Status: {r_sum.status_code}")
    print(f"Response: {json.dumps(r_sum.json(), indent=2)}")
    
    # QA API
    r_qa = requests.post(f"{base_url}/api/qa", json={"context": "Aarav has 6 years software experience.", "question": "How many years experience does Aarav have?"})
    print(f"\n[POST /api/qa] Status: {r_qa.status_code}")
    print(f"Response: {json.dumps(r_qa.json(), indent=2)}")
    return True

if __name__ == "__main__":
    test_engine_directly()
    test_api_servers("http://127.0.0.1:5000")
