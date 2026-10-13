"""
Assignment 08: Conversational AI Engine & Prompt Engineering Framework
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon

Provides multi-turn conversational AI engines for:
1. College Inquiry Assistant (Admissions, Courses, Fees, Facilities)
2. Healthcare Assistant (Symptom Guidance, Appointments, Safety Disclaimers)
3. Prompt Engineering Playground (Zero-Shot, Few-Shot, Role-Based, Chain-of-Thought)
"""

import os
import re
import sys
import uuid
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatbotEngine")

# Try loading OpenAI API client if key is configured, else use LLM Simulator Engine
HAS_OPENAI = False
try:
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
        HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ==============================================================================
# SANJIVANI COLLEGE KNOWLEDGE BASE & SYSTEM PROMPT
# ==============================================================================
COLLEGE_SYSTEM_PROMPT = """
You are the official Conversational AI Assistant for Sanjivani College of Engineering, Kopargaon (Autonomous SPPU).
Your goal is to provide accurate, polite, and clear information regarding admissions, engineering courses, fees, hostels, and placements.

Rules:
1. Maintain a professional, welcoming tone.
2. Answer queries based on Sanjivani College facts (B.Tech programs in IT, Computer, Data Science, AI&DS, Civil, Mech, Electrical).
3. Fees: B.Tech Open category ~ ₹1,05,000/year; OBC/EBC ~ ₹60,000/year; SC/ST ~ ₹10,000/year.
4. Eligibility: 10+2 with Physics, Mathematics, and valid MHT-CET / JEE Main score (min 45% for open, 40% reserved).
5. If the query is outside college topics, politely state your scope and direct them to admissions@sanjivani.org.in or +91-2423-222862.
"""

COLLEGE_FEW_SHOT_PROMPT = """
Example 1:
User: What are the eligibility criteria for B.Tech admission?
Assistant: For B.Tech admission at Sanjivani COE, candidate must have passed 10+2 with Physics & Mathematics as compulsory subjects with at least 45% marks (40% for reserved category) and a valid MHT-CET or JEE Main score.

Example 2:
User: How is the hostel facility and food quality?
Assistant: Sanjivani COE provides separate high-tech hostels for boys and girls with 24/7 Wi-Fi, security, clean mess providing hygienic meals, gym, and sports facilities.

Example 3:
User: What was the highest placement package last year?
Assistant: The highest placement package achieved by Sanjivani COE students was ₹20 LPA, with top recruiters including TCS, Infosys, Cognizant, Wipro, Capgemini, and Hexaware.
"""

# ==============================================================================
# HEALTHCARE ASSISTANT SYSTEM PROMPT & SAFETY DISCLAIMER
# ==============================================================================
HEALTHCARE_SYSTEM_PROMPT = """
You are a virtual Healthcare Information Assistant. Your role is to provide GENERAL symptom guidance, health wellness FAQs, and appointment scheduling help.

CRITICAL SAFETY RULES:
1. DISCLAIMER REQUIRED: Always state that you are an AI assistant, NOT a licensed doctor, and that your output is for informational purposes only.
2. MEDICAL EMERGENCY ESCALATION: If the user mentions chest pain, difficulty breathing, severe bleeding, sudden paralysis, or loss of consciousness, IMMEDIATELY instruct them to call Emergency (108/911) or visit the nearest hospital emergency room.
3. NEVER provide prescription medication dosages or definitive medical diagnosis.
"""

# ==============================================================================
# MULTI-TURN SESSION CONTEXT MANAGER
# ==============================================================================
class SessionContextManager:
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str, system_prompt: str) -> List[Dict[str, str]]:
        if session_id not in self.sessions:
            self.sessions[session_id] = [{"role": "system", "content": system_prompt}]
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        if session_id in self.sessions:
            self.sessions[session_id].append({"role": role, "content": content})

    def clear_session(self, session_id: str, system_prompt: str):
        self.sessions[session_id] = [{"role": "system", "content": system_prompt}]

session_manager = SessionContextManager()

# ==============================================================================
# CONVERSATIONAL AI ENGINE CLASS
# ==============================================================================
class ConversationalAIEngine:
    def __init__(self):
        self.session_manager = session_manager

    def classify_college_intent(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ['admission', 'apply', 'eligibility', 'cutoff', 'cet', 'jee']):
            return 'admissions_inquiry'
        elif any(k in t for k in ['fee', 'fees', 'cost', 'tuition', 'scholarship']):
            return 'fee_structure'
        elif any(k in t for k in ['course', 'branch', 'department', 'b.tech', 'm.tech', 'it', 'computer', 'data science']):
            return 'course_details'
        elif any(k in t for k in ['hostel', 'mess', 'room', 'accommodation', 'stay']):
            return 'hostel_facility'
        elif any(k in t for k in ['placement', 'package', 'salary', 'company', 'recruiter', 'job']):
            return 'placement_info'
        else:
            return 'general_college_query'

    def classify_healthcare_intent(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ['chest pain', 'breath', 'bleeding', 'heart attack', 'emergency', 'unconscious', 'paralysis']):
            return 'EMERGENCY_ESCALATION'
        elif any(k in t for k in ['fever', 'cough', 'headache', 'cold', 'pain', 'stomach', 'symptom', 'sick']):
            return 'symptom_check'
        elif any(k in t for k in ['appointment', 'book', 'doctor', 'schedule', 'consult', 'visit']):
            return 'appointment_booking'
        else:
            return 'general_health_faq'

    def college_chat(self, user_message: str, session_id: str = "default_college") -> Dict[str, Any]:
        """Handles multi-turn dialogue for Sanjivani College Inquiry Chatbot."""
        history = self.session_manager.get_history(session_id, COLLEGE_SYSTEM_PROMPT)
        intent = self.classify_college_intent(user_message)

        bot_reply = ""
        # 1. OpenAI API Execution if key is configured
        if HAS_OPENAI:
            try:
                history.append({"role": "user", "content": user_message})
                res = client.chat.completions.create(model="gpt-4o-mini", messages=history, temperature=0.7)
                bot_reply = res.choices[0].message.content
                history.append({"role": "assistant", "content": bot_reply})
                return {
                    "response": bot_reply,
                    "session_id": session_id,
                    "intent": intent,
                    "provider": "OpenAI LLM API (gpt-4o-mini)",
                    "history_length": len(history)
                }
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}. Falling back to Rule-based LLM Simulator.")

        # 2. Rule-based LLM Simulator with Context
        t = user_message.lower()
        if intent == 'admissions_inquiry':
            bot_reply = "For B.Tech admission at Sanjivani College of Engineering, candidates must have passed 10+2 with Physics & Math (min 45% open, 40% reserved) and have a valid MHT-CET or JEE Main score. Admissions follow the DTE Maharashtra CAP rounds."
        elif intent == 'fee_structure':
            bot_reply = "Sanjivani COE Annual Fee Structure: Open Category: ~₹1,05,000/yr | OBC/EBC: ~₹60,000/yr | SC/ST/VJNT: ~₹10,000/yr. Scholarships like TFWS and EBC are available for eligible students."
        elif intent == 'course_details':
            bot_reply = "Sanjivani COE offers autonomous B.Tech programs in Information Technology (IT), Computer Engineering, AI & Data Science, Civil, Mechanical, and Electrical Engineering. All programs are NAAC 'A' Grade & NBA accredited."
        elif intent == 'hostel_facility':
            bot_reply = "Sanjivani COE provides separate high-tech boys and girls hostels on campus with 24/7 Wi-Fi, solar hot water, sports facilities, digital security, and clean mess facilities."
        elif intent == 'placement_info':
            bot_reply = "Sanjivani Training & Placement Cell recorded top packages up to ₹20 LPA last year. Major recruiters include TCS, Infosys, Cognizant, Wipro, Capgemini, and Hexaware."
        else:
            bot_reply = "Thank you for reaching out to Sanjivani College of Engineering! For specific queries, please email admissions@sanjivani.org.in or call +91-2423-222862."

        self.session_manager.add_message(session_id, "user", user_message)
        self.session_manager.add_message(session_id, "assistant", bot_reply)

        return {
            "response": bot_reply,
            "session_id": session_id,
            "intent": intent,
            "provider": "Sanjivani LLM Simulator Engine",
            "history_length": len(history)
        }

    def healthcare_chat(self, user_message: str, session_id: str = "default_health") -> Dict[str, Any]:
        """Handles multi-turn dialogue for Healthcare Assistant with safety disclaimers."""
        history = self.session_manager.get_history(session_id, HEALTHCARE_SYSTEM_PROMPT)
        intent = self.classify_healthcare_intent(user_message)

        disclaimer = "\n\n⚠️ *Medical Disclaimer: I am an AI virtual assistant, not a doctor. For professional diagnosis or emergency, consult a certified healthcare professional.*"

        if intent == 'EMERGENCY_ESCALATION':
            emergency_reply = "🚨 **MEDICAL EMERGENCY DETECTED!** 🚨\nPlease immediately call Emergency Medical Services (108 / 911) or proceed to the nearest hospital emergency department! Do not wait for online medical advice."
            return {
                "response": emergency_reply + disclaimer,
                "session_id": session_id,
                "intent": intent,
                "is_emergency": True,
                "provider": "Healthcare Safety Escalation Protocol"
            }

        bot_reply = ""
        if HAS_OPENAI:
            try:
                history.append({"role": "user", "content": user_message})
                res = client.chat.completions.create(model="gpt-4o-mini", messages=history, temperature=0.5)
                bot_reply = res.choices[0].message.content + disclaimer
                history.append({"role": "assistant", "content": bot_reply})
                return {
                    "response": bot_reply,
                    "session_id": session_id,
                    "intent": intent,
                    "is_emergency": False,
                    "provider": "OpenAI LLM API"
                }
            except Exception as e:
                logger.warning(f"OpenAI error: {e}")

        if intent == 'symptom_check':
            bot_reply = "For mild symptoms like fever, cough, or fatigue: ensure adequate rest, stay hydrated with fluids, and monitor your temperature. If fever exceeds 102°F or symptoms persist for over 3 days, consult a physician."
        elif intent == 'appointment_booking':
            bot_reply = "To schedule a doctor consultation: Please specify your preferred specialty (e.g., General Medicine, Pediatrics, Cardiology) and preferred date. Our clinic desk will confirm availability."
        else:
            bot_reply = "I can assist with general health queries, wellness tips, and appointment booking. What health concern can I help you with today?"

        full_response = bot_reply + disclaimer
        self.session_manager.add_message(session_id, "user", user_message)
        self.session_manager.add_message(session_id, "assistant", full_response)

        return {
            "response": full_response,
            "session_id": session_id,
            "intent": intent,
            "is_emergency": False,
            "provider": "Healthcare AI Engine"
        }

    def execute_prompt_technique(self, technique: str, task_prompt: str) -> Dict[str, Any]:
        """Executes & compares Zero-Shot, Few-Shot, Role-Play, and Chain-of-Thought prompting strategies."""
        tech_lower = technique.lower()
        formatted_prompt = ""
        explanation = ""

        if "zero" in tech_lower:
            formatted_prompt = f"Task: {task_prompt}\n\nProvide direct answer without examples."
            explanation = "Zero-Shot Prompting: Requests output directly based solely on instructions without prior context examples."
            sample_output = f"Direct Zero-Shot Answer to '{task_prompt}': Evaluated based on raw model instruction."

        elif "few" in tech_lower:
            formatted_prompt = f"{COLLEGE_FEW_SHOT_PROMPT}\n\nTask: {task_prompt}\nAssistant:"
            explanation = "Few-Shot Prompting: Demonstrates input-output example pairs first so the model infers desired formatting."
            sample_output = f"Few-Shot Contextual Answer for '{task_prompt}' following exemplar structure."

        elif "role" in tech_lower or "system" in tech_lower:
            formatted_prompt = f"System Persona: {COLLEGE_SYSTEM_PROMPT}\n\nUser Query: {task_prompt}"
            explanation = "Role-Based / System Prompting: Assigns a specific persona and behavioral rules to constrain output tone."
            sample_output = f"Official Representative Answer for '{task_prompt}' in Sanjivani COE tone."

        elif "chain" in tech_lower or "cot" in tech_lower:
            formatted_prompt = f"User Query: {task_prompt}\n\nInstruction: Let's think step-by-step before arriving at the final response.\nStep 1: Identify key intent.\nStep 2: Retrieve facts.\nStep 3: Formulate answer."
            explanation = "Chain-of-Thought (CoT) Prompting: Encourages step-by-step reasoning before generating final response."
            sample_output = f"Chain-of-Thought Reasoning:\n1. Query analyzes '{task_prompt}'.\n2. Key factors verified.\n3. Final Answer formulated clearly."

        else:
            formatted_prompt = task_prompt
            explanation = "Standard Prompting Strategy."
            sample_output = f"Processed prompt: {task_prompt}"

        return {
            "technique": technique,
            "formatted_prompt": formatted_prompt,
            "output_response": sample_output,
            "explanation": explanation
        }

# Global Singleton
chatbot_engine = ConversationalAIEngine()
