"""
Assignment 08: Streamlit Interactive Web Application & Conversational AI Portal
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon
"""

import os
import sys
import json
import requests
import streamlit as st
from chatbot_engine import chatbot_engine, COLLEGE_SYSTEM_PROMPT, HEALTHCARE_SYSTEM_PROMPT

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Page Config
st.set_page_config(
    page_title="Conversational AI & Prompt Engineering Portal - Practical 08",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: 800;
        color: #065f46;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 15px;
        color: #059669;
        margin-bottom: 20px;
    }
    .card-college {
        background-color: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .card-health {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .badge-intent {
        display: inline-block;
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/illustrations/100/null/chat.png", width=80)
    st.title("Sanjivani COE - IT Dept")
    st.markdown("**Advanced Data Science Lab (PEIT412B)**")
    st.markdown("**Practical 08**: Conversational AI & LLM Prompts")
    st.divider()
    st.markdown("### Systems & Frameworks")
    st.write("• **LLM Engine**: OpenAI / LLM Simulator")
    st.write("• **Techniques**: Zero-Shot, Few-Shot, System, CoT")
    st.write("• **APIs**: Flask (`:5000`) & FastAPI (`:8000`)")
    st.divider()
    st.caption("Class: B.Tech Sem VII | Autonomous SPPU")

# Header
st.markdown('<div class="main-header">💬 Conversational AI & Prompt Engineering Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Practical No. 08: Multi-Turn Chatbot Workflows & Prompt Engineering Strategies</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎓 College Inquiry Chatbot",
    "🏥 Healthcare Assistant Chatbot",
    "🧪 Prompt Engineering Lab",
    "🗺️ Dialogue Workflow & States",
    "🔌 REST API & Postman Emulator",
    "📚 Theory & FAQ Guide"
])

# Initialize Session States for Chat Histories
if "college_messages" not in st.session_state:
    st.session_state.college_messages = [
        {"role": "assistant", "content": "Hello! Welcome to Sanjivani College of Engineering, Kopargaon. How can I assist you with admissions, courses, fees, hostels, or placements today?"}
    ]

if "health_messages" not in st.session_state:
    st.session_state.health_messages = [
        {"role": "assistant", "content": "Hello! I am your virtual Healthcare Information Assistant. How can I help with general symptom guidance or appointment booking today?\n\n⚠️ *Disclaimer: I am an AI assistant, not a doctor. Consult a certified medical professional for diagnosis or emergency.*"}
    ]

# ---------------------------------------------------------
# TAB 1: COLLEGE INQUIRY CHATBOT
# ---------------------------------------------------------
with tab1:
    st.header("🎓 Sanjivani College Inquiry Assistant")
    st.write("Multi-turn conversational chatbot answering student queries on admissions, fees, engineering branches, and campus facilities.")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    if col_btn1.button("📋 Admission Criteria"):
        st.session_state.college_input_preset = "What are the eligibility criteria for B.Tech admission?"
    if col_btn2.button("💰 Fee Structure"):
        st.session_state.college_input_preset = "What is the fee structure for B.Tech open category?"
    if col_btn3.button("🏠 Hostel Facilities"):
        st.session_state.college_input_preset = "Tell me about the hostel facilities and mess."
    if col_btn4.button("💼 Placement Packages"):
        st.session_state.college_input_preset = "What were the highest placement packages last year?"

    st.divider()

    # Render Chat History
    for msg in st.session_state.college_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Box
    user_input = st.chat_input("Ask Sanjivani College Assistant...")
    preset = getattr(st.session_state, "college_input_preset", None)
    if preset:
        user_input = preset
        st.session_state.college_input_preset = None

    if user_input:
        st.session_state.college_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            res = chatbot_engine.college_chat(user_input, session_id="streamlit_college")
            reply = res["response"]
            st.markdown(reply)
            st.caption(f"Detected Intent: {res['intent']} | Provider: {res['provider']}")
            st.session_state.college_messages.append({"role": "assistant", "content": reply})

    if st.button("🔄 Reset College Chat History"):
        st.session_state.college_messages = [
            {"role": "assistant", "content": "Hello! Welcome to Sanjivani College of Engineering, Kopargaon. How can I assist you today?"}
        ]
        chatbot_engine.session_manager.clear_session("streamlit_college", COLLEGE_SYSTEM_PROMPT)
        st.rerun()

# ---------------------------------------------------------
# TAB 2: HEALTHCARE ASSISTANT CHATBOT
# ---------------------------------------------------------
with tab2:
    st.header("🏥 Healthcare Information Assistant")
    st.write("Conversational assistant providing symptom guidance, appointment scheduling, safety disclaimers, and emergency escalation.")

    st.markdown("""
    <div class="card-health">
        <h4 style="margin:0; color:#b91c1c;">⚠️ SAFETY PROTOCOL ACTIVE</h4>
        <p style="margin-top:5px; font-size:13px; color:#7f1d1d;">This assistant enforces mandatory safety disclaimers and immediate emergency hotline escalation for severe symptoms.</p>
    </div>
    """, unsafe_allow_html=True)

    h_btn1, h_btn2, h_btn3 = st.columns(3)
    if h_btn1.button("🤒 Mild Fever & Cough"):
        st.session_state.health_input_preset = "I have a mild fever and cough since yesterday. What should I do?"
    if h_btn2.button("📅 Schedule Appointment"):
        st.session_state.health_input_preset = "I want to schedule an appointment with a doctor."
    if h_btn3.button("🚨 Emergency Check (Severe Chest Pain)"):
        st.session_state.health_input_preset = "I am experiencing severe chest pain and difficulty breathing!"

    st.divider()

    for msg in st.session_state.health_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_h_input = st.chat_input("Ask Healthcare Assistant...")
    h_preset = getattr(st.session_state, "health_input_preset", None)
    if h_preset:
        user_h_input = h_preset
        st.session_state.health_input_preset = None

    if user_h_input:
        st.session_state.health_messages.append({"role": "user", "content": user_h_input})
        with st.chat_message("user"):
            st.markdown(user_h_input)

        with st.chat_message("assistant"):
            h_res = chatbot_engine.healthcare_chat(user_h_input, session_id="streamlit_health")
            h_reply = h_res["response"]
            st.markdown(h_reply)
            if h_res.get("is_emergency"):
                st.error("🚨 EMERGENCY ESCALATION ACTIVATED! Directing to 108 / 911 Hotline.")
            st.session_state.health_messages.append({"role": "assistant", "content": h_reply})

    if st.button("🔄 Reset Healthcare Chat History"):
        st.session_state.health_messages = [
            {"role": "assistant", "content": "Hello! I am your virtual Healthcare Information Assistant. How can I help you today?"}
        ]
        chatbot_engine.session_manager.clear_session("streamlit_health", HEALTHCARE_SYSTEM_PROMPT)
        st.rerun()

# ---------------------------------------------------------
# TAB 3: PROMPT ENGINEERING LABORATORY
# ---------------------------------------------------------
with tab3:
    st.header("🧪 Prompt Engineering Strategy Laboratory")
    st.write("Compare prompt design techniques: **Zero-Shot**, **Few-Shot**, **Role-Based (System Prompting)**, and **Chain-of-Thought (CoT)**.")

    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        tech_choice = st.selectbox("Select Prompting Strategy", [
            "Zero-Shot Prompting",
            "Few-Shot Prompting",
            "Role-Based / System Prompting",
            "Chain-of-Thought (CoT) Prompting"
        ])
    with col_p2:
        task_input = st.text_input("Enter Task Prompt", value="Explain B.Tech admission eligibility and fee structure for Open category.")

    if st.button("⚡ Execute Prompt Technique", type="primary"):
        res_prompt = chatbot_engine.execute_prompt_technique(tech_choice, task_input)

        st.subheader(f"Strategy: {res_prompt['technique']}")
        st.info(res_prompt["explanation"])

        st.markdown("### Formatted Prompt Construction sent to LLM:")
        st.code(res_prompt["formatted_prompt"], language="text")

        st.markdown("### Model Generated Output:")
        st.success(res_prompt["output_response"])

# ---------------------------------------------------------
# TAB 4: DIALOGUE WORKFLOW & STATE DIAGRAM
# ---------------------------------------------------------
with tab4:
    st.header("🗺️ Chatbot Dialogue Workflow & State Machine")

    st.markdown("""
    ```
    +-----------------------------------------------------------------------------------+
    |                              USER INPUT MESSAGE                                   |
    +-----------------------------------------------------------------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    INTENT CLASSIFIER ENGINE   |
                              +-------------------------------+
                                  /           |           \\
                                 /            |            \\
    +------------------------------+   +---------------+   +----------------------------------+
    | COLLEGE INQUIRY INTENT       |   | HEALTHCARE    |   | EMERGENCY ESCALATION             |
    | (Admissions/Fees/Hostel)     |   | INTENT        |   | (Chest Pain / Severe Symptoms)   |
    +------------------------------+   +---------------+   +----------------------------------+
                   |                          |                             |
                   v                          v                             v
    +------------------------------+   +---------------+   +----------------------------------+
    | Multi-Turn Context Manager   |   | Multi-Turn    |   | 🚨 TRIGGER EMERGENCY HOTLINE     |
    | (Role-Based System Prompt)   |   | Context +     |   | (Call 108 / Visit Hospital ER)   |
    +------------------------------+   | Safety        |   +----------------------------------+
                   |                   | Disclaimer    |
                   v                   +---------------+
    +------------------------------+          |
    | GENERATE ASSISTANT REPLY     | <--------+
    +------------------------------+
    ```
    """)

# ---------------------------------------------------------
# TAB 5: REST API TESTER & POSTMAN EMULATOR
# ---------------------------------------------------------
with tab5:
    st.header("🔌 REST API Endpoint Tester & Postman Emulator")
    st.write("Test deployed Flask (`port 5000`) or FastAPI (`port 8000`) chatbot endpoints live.")

    framework = st.radio("Select Backend Server", ["Flask API (Port 5000)", "FastAPI (Port 8000)"], horizontal=True)
    api_ep = st.selectbox("Select Endpoint", ["/chat/college", "/chat/healthcare", "/prompt/test", "/health"])

    port = 5000 if "Flask" in framework else 8000
    url = f"http://127.0.0.1:{port}{api_ep}"
    st.code(f"HTTP POST Request: {url}", language="http")

    if api_ep == "/chat/college":
        payload = {"message": "What is the fee structure for B.Tech open category?", "session_id": "api_session_1"}
    elif api_ep == "/chat/healthcare":
        payload = {"message": "I have mild cough and fever. What should I do?", "session_id": "api_health_1"}
    elif api_ep == "/prompt/test":
        payload = {"technique": "few-shot", "prompt": "What are the B.Tech course fees?"}
    else:
        payload = {}

    payload_str = st.text_area("Request JSON Payload", value=json.dumps(payload, indent=2), height=150)

    if st.button("🚀 Send HTTP Request"):
        try:
            if api_ep == "/health":
                res = requests.get(url, timeout=3)
            else:
                res = requests.post(url, json=json.loads(payload_str), timeout=3)

            st.subheader(f"Response (Status Code: {res.status_code})")
            if res.status_code == 200:
                st.success("API Request Succeeded!")
                st.json(res.json())
            else:
                st.error(f"API Error {res.status_code}")
                st.text(res.text)
        except requests.exceptions.ConnectionError:
            st.warning(f"Could not connect to {url}. Make sure flask_app.py or fastapi_app.py is running.")

# ---------------------------------------------------------
# TAB 6: THEORY & FAQ GUIDE
# ---------------------------------------------------------
with tab6:
    st.header("📚 Practical 08 Theory & FAQ Guide")

    faqs = [
        ("What is prompt engineering, and why is it important for LLM-based applications?",
         "Prompt engineering is the practice of crafting clear, structured input instructions (prompts) to guide a Large Language Model toward producing accurate, contextual, and desired responses. It controls tone, boundary rules, output formatting, and reduces hallucinations."),

        ("Differentiate between zero-shot, few-shot, and chain-of-thought prompting.",
         "• **Zero-Shot**: Asks the LLM to perform a task with zero input-output examples.\n• **Few-Shot**: Provides a small number of example input-output pairs so the LLM infers expected pattern and formatting.\n• **Chain-of-Thought (CoT)**: Instructs the model to reason step-by-step before producing the final answer, improving performance on multi-step logic."),

        ("What is a system prompt, and how does it influence chatbot behavior?",
         "A system prompt defines the top-level persona, role instructions, domain boundary constraints, and response rules that govern the LLM's entire conversational behavior across all turns."),

        ("How is context/chat history managed in a multi-turn conversational AI system?",
         "Context is managed by storing previous user and assistant message pairs in a structured session array (`chat_history`) and passing the accumulated message history back to the LLM API during each subsequent turn."),

        ("What safety measures should be built into a healthcare or sensitive-domain chatbot?",
         "1. Mandatory Safety Disclaimers (AI assistant, not licensed doctor).\n2. Automated Emergency Escalation Protocols (immediate referral to 108/911 for severe symptoms like chest pain).\n3. Prohibition of prescription dosages or definitive medical diagnosis.")
    ]

    for q, a in faqs:
        with st.expander(f"❓ {q}"):
            st.write(a)

st.markdown("---")
st.caption("© 2026 Department of Information Technology | Sanjivani College of Engineering, Kopargaon")
