"""
Assignment 07/08: FastAPI Transformer Server with Pydantic Validation & Swagger UI
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon

Provides high-performance API endpoints and automatic Swagger UI documentation at http://127.0.0.1:8000/docs
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from transformer_engine import nlp_engine
import uvicorn

app = FastAPI(
    title="HuggingFace Transformer NLP API - Sanjivani IT Dept",
    description="FastAPI service for Text Summarization, Extractive QA, and Resume Screening",
    version="1.0.0"
)

class SummarizeInput(BaseModel):
    text: str = Field(..., description="Document text to be summarized")
    max_length: int = Field(130, example=120, description="Maximum summary token length")
    min_length: int = Field(30, example=30, description="Minimum summary token length")

class QAInput(BaseModel):
    context: str = Field(..., description="Background text document context")
    question: str = Field(..., example="How many years of experience does the candidate have?", description="Question to answer")

class ResumeInput(BaseModel):
    resume_text: str = Field(..., description="Candidate resume text payload")

@app.get('/')
def root():
    return {
        "status": "online",
        "framework": "FastAPI",
        "service": "HuggingFace Transformer API",
        "documentation": "/docs",
        "institution": "Sanjivani College of Engineering, Kopargaon"
    }

@app.get('/health')
def health():
    return {
        "status": "healthy",
        "transformers_ready": nlp_engine.models_loaded
    }

@app.post('/api/summarize')
def summarize_endpoint(item: SummarizeInput):
    if not item.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
    result = nlp_engine.summarize_text(item.text, max_length=item.max_length, min_length=item.min_length)
    return {"success": True, "data": result}

@app.post('/api/qa')
def qa_endpoint(item: QAInput):
    if not item.context.strip() or not item.question.strip():
        raise HTTPException(status_code=400, detail="Context and Question cannot be empty.")
    result = nlp_engine.answer_question(item.context, item.question)
    return {"success": True, "data": result}

@app.post('/api/screen_resume')
def screen_resume_endpoint(item: ResumeInput):
    if not item.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")
    result = nlp_engine.screen_resume(item.resume_text)
    return {"success": True, "data": result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
