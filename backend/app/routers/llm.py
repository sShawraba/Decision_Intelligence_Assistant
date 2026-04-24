"""LLM priority prediction endpoint router"""
import time
from fastapi import APIRouter
from app.schemas.models import LLMPriorityRequest, LLMPriorityResponse
from app.services.llm_service import llm_service
from app.utils.logger import logger

router = APIRouter(prefix="/api", tags=["LLM"])


@router.post("/llm-priority", response_model=LLMPriorityResponse)
async def llm_priority(request: LLMPriorityRequest):
    """
    LLM-based priority prediction: Use zero-shot LLM to classify priority.
    
    Returns:
    - priority: HIGH, MEDIUM, or LOW
    - reasoning: Explanation for the prediction
    - latency_ms: Processing time
    - cost_usd: API cost for this request
    """
    start_time = time.time()
    
    # Get priority prediction
    result = llm_service.predict_priority(request.text)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the query
    logger.log_query(
        query=request.text,
        query_type="llm",
        response={
            "priority": result['priority'],
            "reasoning": result['reasoning']
        },
        latency=time.time() - start_time,
        cost=result['cost']
    )
    
    return LLMPriorityResponse(
        priority=result['priority'],
        reasoning=result['reasoning'],
        latency_ms=round(latency_ms, 2),
        cost_usd=result['cost']
    )
