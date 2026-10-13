"""
Assignment 08: Automated Chatbot Integration & Prompt Test Suite
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon

Validates multi-turn session tracking, intent classification, prompt techniques, and API endpoints.
"""

import sys
import json
import requests
from chatbot_engine import chatbot_engine

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_engine_chat():
    print("=" * 60)
    print("Testing Multi-Turn Chatbot Engine Direct Calls")
    print("=" * 60)
    
    # 1. College Chat Multi-turn session
    session_id = "unit_test_college"
    r1 = chatbot_engine.college_chat("What are the eligibility criteria for B.Tech admission?", session_id=session_id)
    print(f"\n[Turn 1 - College Admission]")
    print(f"User: What are the eligibility criteria for B.Tech admission?")
    print(f"Bot Response: {r1['response']}")
    print(f"Intent: {r1['intent']} | History Length: {r1['history_length']}")
    
    r2 = chatbot_engine.college_chat("And what about the hostel fees?", session_id=session_id)
    print(f"\n[Turn 2 - Hostel Inquiry]")
    print(f"User: And what about the hostel fees?")
    print(f"Bot Response: {r2['response']}")
    print(f"Intent: {r2['intent']} | History Length: {r2['history_length']}")
    
    # 2. Healthcare Chat Emergency Check
    r_emerg = chatbot_engine.healthcare_chat("I am suffering from severe chest pain and breathlessness!")
    print(f"\n[Healthcare Emergency Escalation Check]")
    print(f"Bot Response: {r_emerg['response']}")
    print(f"Is Emergency: {r_emerg.get('is_emergency')}")
    
    return True

def test_api_servers(base_url="http://127.0.0.1:5000"):
    print("\n" + "=" * 60)
    print(f"Testing Chatbot REST API Endpoints at {base_url}")
    print("=" * 60)
    
    try:
        r_health = requests.get(f"{base_url}/health", timeout=3)
        print(f"[GET /health] Status: {r_health.status_code}")
        print(f"Response: {r_health.json()}")
    except Exception as e:
        print(f"❌ Could not connect to API server at {base_url}: {e}")
        return False
        
    r_chat = requests.post(f"{base_url}/chat/college", json={"message": "What is the fee structure?", "session_id": "test_api_1"})
    print(f"\n[POST /chat/college] Status: {r_chat.status_code}")
    print(f"Response: {json.dumps(r_chat.json(), indent=2)}")
    return True

if __name__ == "__main__":
    test_engine_chat()
    test_api_servers("http://127.0.0.1:5000")
