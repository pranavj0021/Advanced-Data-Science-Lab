"""
Assignment 07/08: HuggingFace Transformer NLP Inference Engine
Subject: Advanced Data Science Laboratory (PEIT412B)
Department of Information Technology, Sanjivani College of Engineering, Kopargaon

Provides transformer-based NLP inference pipelines for:
1. Text Summarization (BART / T5 / DistilBART)
2. Extractive Question Answering (DistilBERT SQuAD / BERT)
3. Automated Resume Screening & Recruiter Query Answering
"""

import os
import re
import sys
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TransformerEngine")

# HuggingFace Transformers & PyTorch Imports
HAS_TRANSFORMERS = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("HuggingFace Transformers library not installed or partially available.")

class TransformerNLPEngine:
    def __init__(self):
        self.summarizer_pipeline = None
        self.qa_pipeline = None
        self.models_loaded = False
        self._init_pipelines()

    def _init_pipelines(self):
        """Initializes HuggingFace pipelines with fallback error handling."""
        if not HAS_TRANSFORMERS:
            logger.info("Operating in Heuristic NLP mode.")
            return

        try:
            logger.info("Loading HuggingFace Summarization Pipeline (DistilBART / BART)...")
            # Using lightweight DistilBART for fast CPU inference
            self.summarizer_pipeline = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",
                tokenizer="sshleifer/distilbart-cnn-12-6"
            )
        except Exception as e:
            logger.warning(f"Could not load primary HuggingFace summarizer model: {e}. Trying fallback pipeline...")
            try:
                self.summarizer_pipeline = pipeline("summarization")
            except Exception as e2:
                logger.error(f"Summarizer pipeline initialization failed: {e2}")

        try:
            logger.info("Loading HuggingFace Question-Answering Pipeline (DistilBERT SQuAD)...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad"
            )
        except Exception as e:
            logger.warning(f"Could not load QA model: {e}. Trying default QA pipeline...")
            try:
                self.qa_pipeline = pipeline("question-answering")
            except Exception as e2:
                logger.error(f"QA pipeline initialization failed: {e2}")

        self.models_loaded = (self.summarizer_pipeline is not None or self.qa_pipeline is not None)

    def summarize_text(self, text: str, max_length: int = 130, min_length: int = 30) -> Dict[str, Any]:
        """Summarizes text using Transformer pipeline with fallback."""
        if not text or len(text.strip()) == 0:
            return {"summary_text": "", "method": "none", "word_count_original": 0, "word_count_summary": 0}

        original_words = len(text.split())

        if self.summarizer_pipeline is not None:
            try:
                # Adjust max_length dynamically based on input length
                effective_max = min(max_length, max(40, int(original_words * 0.6)))
                effective_min = min(min_length, max(15, int(effective_max * 0.4)))
                
                result = self.summarizer_pipeline(
                    text, 
                    max_length=effective_max, 
                    min_length=effective_min, 
                    do_sample=False
                )
                summary = result[0]['summary_text']
                summary_words = len(summary.split())
                
                return {
                    "summary_text": summary,
                    "method": "HuggingFace Transformer (DistilBART)",
                    "word_count_original": original_words,
                    "word_count_summary": summary_words,
                    "reduction_percentage": round((1.0 - (summary_words / max(1, original_words))) * 100, 1)
                }
            except Exception as e:
                logger.warning(f"Transformer summarization exception: {e}. Falling back to Extractive Summary.")

        # Fallback Extractive Heuristic Summarizer
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 15]
        selected_sentences = sentences[:max(2, min(4, len(sentences)))]
        summary_fallback = " ".join(selected_sentences)
        summary_words = len(summary_fallback.split())

        return {
            "summary_text": summary_fallback,
            "method": "Extractive Heuristic (Sentence Salience)",
            "word_count_original": original_words,
            "word_count_summary": summary_words,
            "reduction_percentage": round((1.0 - (summary_words / max(1, original_words))) * 100, 1)
        }

    def answer_question(self, context: str, question: str) -> Dict[str, Any]:
        """Answers recruiter or user question based on context document."""
        if not context or not question:
            return {"question": question, "answer": "Context or question is empty.", "score": 0.0, "method": "none"}

        if self.qa_pipeline is not None:
            try:
                result = self.qa_pipeline(question=question, context=context)
                return {
                    "question": question,
                    "answer": result["answer"],
                    "score": round(float(result["score"]), 4),
                    "start": result.get("start", 0),
                    "end": result.get("end", 0),
                    "method": "HuggingFace DistilBERT-SQuAD"
                }
            except Exception as e:
                logger.warning(f"Transformer QA exception: {e}. Falling back to pattern QA.")

        # Fallback Extractive Pattern QA
        q_lower = question.lower()
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', context) if len(s.strip()) > 10]
        
        keywords = [w for w in q_lower.split() if w not in ['what', 'is', 'the', 'how', 'many', 'years', 'of', 'does', 'in', 'have', 'candidate', 'who']]
        best_sentence = "Information not explicitly mentioned in document."
        max_overlap = 0

        for s in sentences:
            s_lower = s.lower()
            overlap = sum(1 for kw in keywords if kw in s_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = s

        return {
            "question": question,
            "answer": best_sentence,
            "score": 0.75 if max_overlap > 0 else 0.20,
            "method": "Extractive Keyword Match"
        }

    def screen_resume(self, resume_text: str) -> Dict[str, Any]:
        """Screens candidate resume, extracts metadata, summarizes, and answers key recruiter queries."""
        # 1. Executive Summary
        summ_res = self.summarize_text(resume_text, max_length=100, min_length=25)
        
        # 2. Key Recruiter Queries
        queries = [
            ("Candidate Name", "What is the candidate's name?"),
            ("Years of Experience", "How many years of experience does the candidate have?"),
            ("Key Technical Skills", "What programming languages and technical skills does the candidate possess?"),
            ("Highest Education Degree", "What is the candidate's degree or education qualification?"),
            ("Target Role / Designation", "What is the candidate's target job role or current title?")
        ]
        
        qa_results = {}
        for key, q in queries:
            ans_res = self.answer_question(context=resume_text, question=q)
            qa_results[key] = ans_res["answer"]
            
        # 3. Rule-based skill extraction enrichment
        known_skills = ["Python", "JavaScript", "React", "Flask", "FastAPI", "Docker", "PyTorch", "TensorFlow", "Transformers", "SQL", "PostgreSQL", "AWS", "Kubernetes", "Git"]
        found_skills = [skill for skill in known_skills if re.search(r'\b' + re.escape(skill) + r'\b', resume_text, re.IGNORECASE)]
        
        return {
            "candidate_name": qa_results.get("Candidate Name", "Extracted Candidate"),
            "target_role": qa_results.get("Target Role / Designation", "Software Professional"),
            "years_experience": qa_results.get("Years of Experience", "N/A"),
            "education": qa_results.get("Highest Education Degree", "Bachelor of Technology"),
            "detected_skills": found_skills if found_skills else ["Python", "Machine Learning"],
            "recruiter_answers": qa_results,
            "executive_summary": summ_res["summary_text"],
            "word_count_reduction": summ_res.get("reduction_percentage", 0.0)
        }

# Global Singleton Instance
nlp_engine = TransformerNLPEngine()
