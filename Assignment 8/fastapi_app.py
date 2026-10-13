"""
Assignment 08: FastAPI Chatbot Server with Pydantic Validation & Swagger UI
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon

Provides high-performance chatbot endpoints and interactive Swagger UI at http://127.0.0.1:8000/docs
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from chatbot_engine import chatbot_engine
import uvicorn

app = FastAPI(
    title="Conversational AI & Prompt Engineering API - Sanjivani IT Dept",
    description="FastAPI service for College Inquiry Chatbot, Healthcare Assistant, and Prompting Playground",
    version="1.0.0"
)

class ChatInput(BaseModel):
    message: str = Field(..., example="What are the eligibility criteria for B.Tech admission?", description="User query message")
    session_id: Optional[str] = Field("session_001", example="session_001", description="Multi-turn session tracking ID")

class PromptInput(BaseModel):
    technique: str = Field("few-shot", example="few-shot", description="Prompting strategy: zero-shot, few-shot, role-play, chain-of-thought")
    prompt: str = Field(..., example="What are the B.Tech course fees?", description="Task prompt instruction")

@app.get('/')
def root():
    return {
        "status": "online",
        "framework": "FastAPI",
        "service": "Conversational AI Engine",
        "documentation": "/docs",
        "institution": "Sanjivani College of Engineering, Kopargaon"
    }

@app.get('/health')
def health():
    return {"status": "healthy", "service": "fastapi-chatbot-engine"}

@app.post('/chat/college')
def college_chat_endpoint(item: ChatInput):
    if not item.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    result = chatbot_engine.college_chat(item.message, session_id=item.session_id)
    return {"success": True, "data": result}

@app.post('/chat/healthcare')
def healthcare_chat_endpoint(item: ChatInput):
    if not item.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    result = chatbot_engine.healthcare_chat(item.message, session_id=item.session_id)
    return {"success": True, "data": result}

@app.post('/prompt/test')
def prompt_test_endpoint(item: PromptInput):
    if not item.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    result = chatbot_engine.execute_prompt_technique(item.technique, item.prompt)
    return {"success": True, "data": result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
