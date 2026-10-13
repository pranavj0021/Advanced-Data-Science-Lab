"""
Assignment 06: Integration Testing for Containerized AI Services
Subject: Advanced Data Science Laboratory (PEIT412B) - Practical 06
Sanjivani College of Engineering, Kopargaon

Validates endpoints for:
1. AI Attendance Monitoring Service (Port 5000)
2. Smart Retail AI Service (Port 5000 / 8000)
"""

import sys
import json
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_attendance_service(base_url="http://127.0.0.1:5000"):
    print("=" * 60)
    print(f"Testing AI Attendance System at {base_url}")
    print("=" * 60)
    
    try:
        r_health = requests.get(f"{base_url}/health", timeout=3)
        print(f"[GET /health] Status: {r_health.status_code}")
        print(f"Response: {r_health.json()}")
    except Exception as e:
        print(f"❌ Could not connect to Attendance Service: {e}")
        return False
        
    payload = {"face_feature_vector": [0.45, 0.82, 0.19, 0.64]}
    r_mark = requests.post(f"{base_url}/mark_attendance", json=payload)
    print(f"\n[POST /mark_attendance] Status: {r_mark.status_code}")
    print(f"Response: {json.dumps(r_mark.json(), indent=2)}")
    return True

def test_retail_service(base_url="http://127.0.0.1:5000"):
    print("\n" + "=" * 60)
    print(f"Testing Smart Retail AI Service at {base_url}")
    print("=" * 60)
    
    try:
        r_health = requests.get(f"{base_url}/health", timeout=3)
        print(f"[GET /health] Status: {r_health.status_code}")
        print(f"Response: {r_health.json()}")
    except Exception as e:
        print(f"❌ Could not connect to Retail Service: {e}")
        return False
        
    footfall_payload = {"footfall_count": 210, "store_capacity": 250}
    r_foot = requests.post(f"{base_url}/predict/footfall", json=footfall_payload)
    print(f"\n[POST /predict/footfall] Status: {r_foot.status_code}")
    print(f"Response: {json.dumps(r_foot.json(), indent=2)}")
    
    stock_payload = {"shelf_stock_pct": 18.5, "category": "Dairy & Eggs"}
    r_stock = requests.post(f"{base_url}/predict/stock", json=stock_payload)
    print(f"\n[POST /predict/stock] Status: {r_stock.status_code}")
    print(f"Response: {json.dumps(r_stock.json(), indent=2)}")
    return True

if __name__ == "__main__":
    test_attendance_service()
    test_retail_service()
