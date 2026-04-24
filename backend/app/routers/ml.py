"""ML prediction endpoint router"""
import time
from fastapi import APIRouter
from app.schemas.models import PredictRequest, PredictResponse
from app.services.ml_service import ml_service
from app.utils.logger import logger

router = APIRouter(prefix="/api", tags=["ML"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    ML prediction endpoint: Use trained sklearn model to classify text.
    
    Returns:
    - label: Predicted class
    - confidence: Confidence score
    - latency_ms: Processing time
    """
    start_time = time.time()
    
    # Make prediction
    prediction = ml_service.predict(request.text)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the query
    logger.log_query(
        query=request.text,
        query_type="ml",
        response={
            "label": prediction['label'],
            "confidence": prediction['confidence']
        },
        latency=time.time() - start_time,
        cost=0.0  # ML models don't have API costs
    )
    
    return PredictResponse(
        label=prediction['label'],
        confidence=prediction['confidence'],
        latency_ms=round(latency_ms, 2)
    )
