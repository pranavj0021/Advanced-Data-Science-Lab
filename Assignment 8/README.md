# Practical No. 08: Designing Prompts for LLM Interaction & Chatbot Workflows

**Sanjivani Rural Education Society’s**  
**Sanjivani College of Engineering, Kopargaon**  
**Department of Information Technology**  
*Class: B. Tech. Semester: VII | Subject: Advanced Data Science Laboratory (PEIT412B)*  

---

## 📌 Project Overview
This repository contains the complete implementation for **Practical No. 08**, covering Prompt Engineering techniques (**Zero-Shot**, **Few-Shot**, **Role-Based System Prompting**, **Chain-of-Thought**), multi-turn conversation context management, dialogue intent classification, emergency escalation protocols, and real-world conversational AI applications.

### 💬 Case Studies Implemented
1. **Sanjivani College Inquiry Chatbot**: Answers student queries regarding B.Tech/M.Tech admissions, fee structures, hostel accommodation, and placement statistics using role-based system prompts.
2. **Healthcare Information Assistant**: Helps users with symptom-based guidance, clinic appointment scheduling, mandatory medical disclaimers, and automated emergency escalation for severe symptoms.

---

## 📁 Repository Structure
```
Assignment_8/
├── chatbot_engine.py          # Multi-Turn Session Engine & Prompt Strategies
├── flask_app.py               # Flask REST API (:5000)
├── fastapi_app.py             # FastAPI REST API (:8000 + Swagger UI)
├── app.py                     # Streamlit Interactive Web Application
├── test_chatbot.py            # Automated Integration Test Suite
├── postman_collection.json    # Postman API Test Collection
├── Assignment_8.ipynb         # Full Jupyter Notebook
└── README.md
```

---

## 🚀 How to Run the Applications

### 1. Test Chatbot Engine & Session Context Directly
```bash
python test_chatbot.py
```

### 2. Run Flask REST API (Port 5000)
```bash
python flask_app.py
```
*Exposed Endpoints:*
- `GET http://127.0.0.1:5000/health`
- `POST http://127.0.0.1:5000/chat/college`
- `POST http://127.0.0.1:5000/chat/healthcare`
- `POST http://127.0.0.1:5000/prompt/test`

### 3. Run FastAPI REST API (Port 8000 + Swagger UI)
```bash
uvicorn fastapi_app:app --port 8000 --reload
```
- Interactive Documentation: `http://127.0.0.1:8000/docs`

### 4. Run Streamlit Interactive Web Application
```bash
streamlit run app.py
```

---

## 📊 Summary of Implemented Features
- 🎓 **College Inquiry Chatbot**: Multi-turn dialogue UI with quick query buttons, intent detector, and history reset.
- 🏥 **Healthcare Assistant**: Medical disclaimer enforcement, symptom checker, and emergency hotline escalation hook.
- 🧪 **Prompt Engineering Lab**: Interactive playground testing Zero-Shot, Few-Shot, Role-Based, and Chain-of-Thought techniques.
- 🗺️ **Dialogue State Diagram**: Visual state machine mapping intents, context, and fallback logic.
- 🔌 **REST API & Postman Emulator**: Built-in REST client testing Flask/FastAPI servers live.
