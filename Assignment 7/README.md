# Practical No. 07/08: Understanding Transformer Architecture & HuggingFace Models

**Sanjivani Rural Education Society’s**  
**Sanjivani College of Engineering, Kopargaon**  
**Department of Information Technology**  
*Class: B. Tech. Semester: VII | Subject: Advanced Data Science Laboratory (PEIT412B)*  

---

## 📌 Project Overview
This repository contains the complete implementation for **Practical No. 07 / 08**, covering the Transformer architecture, HuggingFace model deployment (`pipeline`, `AutoTokenizer`, `AutoModel`), **Text Summarization** (BART / T5), **Extractive Question Answering** (DistilBERT SQuAD), and real-world applications: **Resume Screening System** and **AI Document Summarizer**.

---

## 📁 Repository Structure
```
Assignment_7/ (and Assignment_8/)
├── datasets/
│   ├── resumes/
│   │   ├── software_engineer_resume.txt
│   │   └── data_scientist_resume.txt
│   └── documents/
│       └── ai_engineering_report.txt
├── transformer_engine.py      # HuggingFace Summarization & QA Engine
├── flask_app.py               # Flask REST API (:5000)
├── fastapi_app.py             # FastAPI REST API (:8000 + Swagger UI)
├── app.py                     # Streamlit Interactive Web Application
├── test_nlp.py                # Automated Integration Test Suite
├── postman_collection.json    # Postman API Test Collection
├── Assignment_7.ipynb         # Full Jupyter Notebook
└── README.md
```

---

## 🚀 How to Run the Applications

### 1. Test NLP Engine Directly
```bash
python test_nlp.py
```

### 2. Run Flask REST API (Port 5000)
```bash
python flask_app.py
```
*Endpoints:*
- `GET http://127.0.0.1:5000/health`
- `POST http://127.0.0.1:5000/api/summarize`
- `POST http://127.0.0.1:5000/api/qa`
- `POST http://127.0.0.1:5000/api/screen_resume`

### 3. Run FastAPI REST API (Port 8000 + Swagger UI)
```bash
uvicorn fastapi_app:app --port 8000 --reload
```
- Interactive Documentation: `http://127.0.0.1:8000/docs`

### 4. Run Streamlit Interactive Web Portal
```bash
streamlit run app.py
```

---

## 📊 Summary of Implemented Features
- 📄 **Resume Screening System**: Automatically extracts candidate skills, years of experience, degree, and answers recruiter custom queries.
- 📚 **AI Document Summarizer**: Generates abstractive summaries using HuggingFace BART/T5 models with word reduction metrics.
- 🔌 **REST API & Postman Emulator**: Built-in REST endpoint testing inside Streamlit.
- 🧠 **Self-Attention Visualizer**: Interactive Query-Key-Value attention matrix matrix simulation.
