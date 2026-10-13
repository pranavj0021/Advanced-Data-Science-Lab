"""
Assignment 07/08: Streamlit Interactive Web Application & Transformer Portal
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from transformer_engine import nlp_engine

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Page Configuration
st.set_page_config(
    page_title="Transformer NLP & Resume Screening Portal - Practical 07/08",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RESUMES_DIR = os.path.join(DATASETS_DIR, "resumes")
DOCS_DIR = os.path.join(DATASETS_DIR, "documents")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 15px;
        color: #3b82f6;
        margin-bottom: 20px;
    }
    .card-profile {
        background-color: #eff6ff;
        border-left: 6px solid #2563eb;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .card-summary {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .card-qa {
        background-color: #fef3c7;
        border-left: 6px solid #d97706;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .skill-badge {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 13px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/illustrations/100/null/brain.png", width=80)
    st.title("Sanjivani COE - IT Dept")
    st.markdown("**Advanced Data Science Lab (PEIT412B)**")
    st.markdown("**Practical 07/08**: HuggingFace Transformers")
    st.divider()
    st.markdown("### Model Architecture")
    st.write("• **Summarization**: BART / DistilBART / T5")
    st.write("• **Question Answering**: DistilBERT SQuAD")
    st.write("• **API Frameworks**: Flask (`:5000`) & FastAPI (`:8000`)")
    st.divider()
    st.caption("Class: B.Tech Sem VII | Autonomous SPPU")

# Title Header
st.markdown('<div class="main-header">🤗 Transformer Architecture & NLP Model Deployment Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Practical 07/08: Resume Screening System & AI Document Summarizer via HuggingFace</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Resume Screening System", 
    "📚 AI Document Summarizer", 
    "🔌 REST API & Postman Emulator", 
    "🧠 Self-Attention & Architecture", 
    "📚 Theory & FAQ Guide"
])

# Helper to load sample files
def load_sample_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# ---------------------------------------------------------
# TAB 1: RESUME SCREENING SYSTEM
# ---------------------------------------------------------
with tab1:
    st.header("📄 Resume Screening & Candidate Evaluation System")
    st.write("Automated NLP pipeline using Transformer Summarization & Extractive Question Answering to screen candidate profiles and answer recruiter queries.")
    
    col_sel1, col_sel2 = st.columns([1, 1])
    with col_sel1:
        sample_resume_choice = st.selectbox("Select Sample Resume Profile", [
            "Software Engineer (Aarav Sharma)", 
            "Data Scientist (Ananya Patel)", 
            "Custom Resume (Paste Text Below)"
        ])
        
    if "Software Engineer" in sample_resume_choice:
        default_resume = load_sample_file(os.path.join(RESUMES_DIR, "software_engineer_resume.txt"))
    elif "Data Scientist" in sample_resume_choice:
        default_resume = load_sample_file(os.path.join(RESUMES_DIR, "data_scientist_resume.txt"))
    else:
        default_resume = ""
        
    resume_input = st.text_area("Candidate Resume Content", value=default_resume, height=220, placeholder="Paste resume text here...")
    
    if st.button("🚀 Screen Candidate Profile", type="primary", use_container_width=True):
        if not resume_input.strip():
            st.error("Please paste or select candidate resume text first.")
        else:
            with st.spinner("Analyzing resume using HuggingFace Transformers..."):
                screening_res = nlp_engine.screen_resume(resume_input)
                
            r_col1, r_col2 = st.columns([1, 1])
            
            with r_col1:
                st.markdown(f"""
                <div class="card-profile">
                    <h3 style="margin:0; color:#1e3a8a;">👤 Candidate Profile</h3>
                    <p style="font-size:16px; margin-top:8px;"><b>Name:</b> {screening_res['candidate_name']}</p>
                    <p style="font-size:16px;"><b>Target Role:</b> {screening_res['target_role']}</p>
                    <p style="font-size:16px;"><b>Experience:</b> {screening_res['years_experience']}</p>
                    <p style="font-size:16px;"><b>Education:</b> {screening_res['education']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🛠️ Extracted Technical Skills")
                skills_html = "".join([f'<span class="skill-badge">{s}</span>' for s in screening_res['detected_skills']])
                st.markdown(skills_html, unsafe_allow_html=True)
                
            with r_col2:
                st.markdown(f"""
                <div class="card-summary">
                    <h3 style="margin:0; color:#16a34a;">📝 Candidate Executive Summary</h3>
                    <p style="font-size:15px; margin-top:10px; line-height:1.6;">{screening_res['executive_summary']}</p>
                    <p style="font-size:12px; color:#6b7280; margin-top:10px;">Word Reduction: {screening_res['word_count_reduction']}%</p>
                </div>
                """, unsafe_allow_html=True)
                
    st.divider()
    st.subheader("❓ Custom Recruiter Query Answering (Extractive QA)")
    st.write("Ask any question regarding the candidate's resume context:")
    
    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        recruiter_question = st.text_input("Enter Recruiter Question", value="What programming languages or frameworks does the candidate know?", placeholder="e.g., How many years of experience does the candidate have?")
    with q_col2:
        st.write("")
        st.write("")
        btn_qa = st.button("🔍 Answer Query")
        
    if btn_qa and resume_input.strip():
        qa_res = nlp_engine.answer_question(resume_input, recruiter_question)
        st.markdown(f"""
        <div class="card-qa">
            <h4 style="margin:0; color:#92400e;">Question: {qa_res['question']}</h4>
            <p style="font-size:18px; font-weight:700; color:#b45309; margin-top:8px;">Answer: "{qa_res['answer']}"</p>
            <p style="font-size:12px; color:#78350f;">Confidence Score: <b>{qa_res['score']}</b> | Model: {qa_res['method']}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: AI DOCUMENT SUMMARIZER
# ---------------------------------------------------------
with tab2:
    st.header("📚 AI Document Summarizer")
    st.write("Generates concise, coherent abstractive summaries for lengthy technical reports, legal articles, and research documents using HuggingFace BART / T5.")
    
    doc_choice = st.selectbox("Select Sample Document", [
        "Technical Report: Evolution of Transformer Architectures",
        "Custom Document (Paste Text Below)"
    ])
    
    if "Technical Report" in doc_choice:
        default_doc = load_sample_file(os.path.join(DOCS_DIR, "ai_engineering_report.txt"))
    else:
        default_doc = ""
        
    doc_input = st.text_area("Document Content", value=default_doc, height=250)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        max_len = st.slider("Maximum Summary Token Length", 50, 250, 130)
    with col_p2:
        min_len = st.slider("Minimum Summary Token Length", 20, 100, 30)
        
    if st.button("✨ Generate AI Document Summary", type="primary", use_container_width=True):
        if not doc_input.strip():
            st.error("Please enter or select document text.")
        else:
            with st.spinner("Summarizing document using HuggingFace Transformer..."):
                summ = nlp_engine.summarize_text(doc_input, max_length=max_len, min_length=min_len)
                
            st.markdown(f"""
            <div class="card-summary">
                <h3 style="margin:0; color:#16a34a;">📄 Generated Summary</h3>
                <p style="font-size:16px; margin-top:12px; line-height:1.7;">{summ['summary_text']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Original Words", f"{summ['word_count_original']} words")
            s_col2.metric("Summary Words", f"{summ['word_count_summary']} words")
            s_col3.metric("Length Reduction", f"{summ['reduction_percentage']}%")

# ---------------------------------------------------------
# TAB 3: REST API TESTER & POSTMAN EMULATOR
# ---------------------------------------------------------
with tab3:
    st.header("🔌 REST API Endpoint Tester & Postman Emulator")
    st.write("Test deployed Flask (`port 5000`) or FastAPI (`port 8000`) transformer endpoints live from Streamlit.")
    
    api_framework = st.radio("Select Framework", ["Flask API (Port 5000)", "FastAPI (Port 8000)"], horizontal=True)
    endpoint = st.selectbox("Select Endpoint", ["/api/summarize", "/api/qa", "/api/screen_resume", "/health"])
    
    port = 5000 if "Flask" in api_framework else 8000
    full_url = f"http://127.0.0.1:{port}{endpoint}"
    st.code(f"HTTP Request to: {full_url}", language="http")
    
    if endpoint == "/api/summarize":
        payload = {"text": doc_input if doc_input else "The Transformer architecture relies on self-attention.", "max_length": 100, "min_length": 25}
    elif endpoint == "/api/qa":
        payload = {"context": resume_input if resume_input else "Aarav has 6 years experience.", "question": "How many years experience does Aarav have?"}
    elif endpoint == "/api/screen_resume":
        payload = {"resume_text": resume_input if resume_input else "Aarav Sharma - Senior Engineer with 6 years experience."}
    else:
        payload = {}
        
    payload_json = st.text_area("JSON Payload", value=json.dumps(payload, indent=2), height=160)
    
    if st.button("🚀 Send HTTP Request"):
        try:
            if endpoint == "/health":
                res = requests.get(full_url, timeout=3)
            else:
                res = requests.post(full_url, json=json.loads(payload_json), timeout=3)
                
            st.subheader(f"Response (Status: {res.status_code})")
            if res.status_code == 200:
                st.success("API Response Success!")
                st.json(res.json())
            else:
                st.error(f"Error {res.status_code}")
                st.text(res.text)
        except requests.exceptions.ConnectionError:
            st.warning(f"Could not connect to http://127.0.0.1:{port}. Run flask_app.py or fastapi_app.py locally.")

# ---------------------------------------------------------
# TAB 4: SELF-ATTENTION & ARCHITECTURE VISUALIZATION
# ---------------------------------------------------------
with tab4:
    st.header("🧠 Transformer Architecture & Self-Attention Mechanics")
    
    st.markdown("""
    ### Scaled Dot-Product Attention Equation
    $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\\frac{Q K^T}{\sqrt{d_k}}\\right) V$$
    """)
    
    st.subheader("💡 Interactive Query-Key Matrix Simulation")
    words = ["The", "Transformer", "uses", "self-attention"]
    num_words = len(words)
    
    # Simulate Attention Weights Matrix
    np.random.seed(42)
    raw_attn = np.random.uniform(0.1, 1.0, (num_words, num_words))
    attn_matrix = raw_attn / raw_attn.sum(axis=-1, keepdims=True)
    
    df_attn = pd.DataFrame(attn_matrix, index=words, columns=words)
    st.dataframe(df_attn.style.background_gradient(cmap="Blues"), use_container_width=True)
    st.caption("Attention weights heatmap representing token-to-token contextual relevance.")

# ---------------------------------------------------------
# TAB 5: THEORY & LAB FAQ GUIDE
# ---------------------------------------------------------
with tab5:
    st.header("📚 Practical 07/08 Theory & FAQ Guide")
    
    faqs = [
        ("What is the Transformer architecture, and how does it differ from RNNs/LSTMs?",
         "Introduced in 'Attention Is All You Need' (Vaswani et al., 2017), the Transformer relies entirely on attention mechanisms to model token relationships across sequences in parallel, eliminating the sequential recurrent bottlenecks and vanishing gradients inherent in RNNs/LSTMs."),
        
        ("Explain self-attention and multi-head attention.",
         "• **Self-Attention**: Computes token relevance across an entire sequence by transforming input embeddings into Query ($Q$), Key ($K$), and Value ($V$) vectors.\n• **Multi-Head Attention**: Runs multiple self-attention operations in parallel, allowing the model to focus on diverse syntactic and semantic subspaces simultaneously."),
        
        ("What is the difference between encoder-only, decoder-only, and encoder-decoder models?",
         "• **Encoder-Only** (e.g. BERT): Bidirectional contextual processing; best for classification & QA.\n• **Decoder-Only** (e.g. GPT, LLaMA): Autoregressive processing; best for text generation.\n• **Encoder-Decoder** (e.g. BART, T5): Sequence-to-sequence transformation; best for translation & abstractive summarization."),
        
        ("What is the HuggingFace pipeline() API?",
         "An abstracted, high-level HuggingFace utility that automatically handles tokenization, model inference, and output decoding for tasks such as `summarization`, `question-answering`, `text-classification`, and `translation`."),
        
        ("What is the difference between extractive and abstractive summarization?",
         "• **Extractive**: Selects exact key sentences directly from source document.\n• **Abstractive**: Generates novel, rephrased sentences capturing core meaning using encoder-decoder models like BART or T5.")
    ]
    
    for q, a in faqs:
        with st.expander(f"❓ {q}"):
            st.write(a)

st.markdown("---")
st.caption("© 2026 Department of Information Technology | Sanjivani College of Engineering, Kopargaon")
