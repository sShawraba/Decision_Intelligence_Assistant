"""Pydantic schemas for request/response validation"""
from typing import List, Optional
from pydantic import BaseModel


# ===== RAG Endpoint =====
class AskRequest(BaseModel):
    """Request body for /ask endpoint"""
    query: str


class RetrievedDocument(BaseModel):
    """A single document retrieved from Chroma"""
    content: str
    similarity_score: float


class AskResponse(BaseModel):
    """Response from /ask endpoint"""
    query: str
    rag_answer: str  # Answer with RAG context
    non_rag_answer: str  # Direct LLM answer without context
    retrieved_documents: List[RetrievedDocument]
    latency_ms: float


# ===== ML Endpoint =====
class PredictRequest(BaseModel):
    """Request body for /predict endpoint"""
    text: str


class PredictResponse(BaseModel):
    """Response from /predict endpoint"""
    label: str
    confidence: float
    latency_ms: float


# ===== LLM Priority Endpoint =====
class LLMPriorityRequest(BaseModel):
    """Request body for /llm-priority endpoint"""
    text: str


class LLMPriorityResponse(BaseModel):
    """Response from /llm-priority endpoint"""
    priority: str  # The LLM's priority prediction
    reasoning: str  # Why the LLM gave this priority
    latency_ms: float
    cost_usd: float
