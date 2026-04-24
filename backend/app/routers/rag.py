"""RAG endpoint router"""
import time
from fastapi import APIRouter
from app.schemas.models import AskRequest, AskResponse, RetrievedDocument
from app.services.chroma_service import chroma_service
from app.services.llm_service import llm_service
from app.utils.logger import logger

router = APIRouter(prefix="/api", tags=["RAG"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    RAG endpoint: Retrieve context from Chroma and answer with/without context.
    
    Returns:
    - RAG answer: Uses retrieved documents as context
    - Non-RAG answer: Direct LLM response without context
    - Retrieved documents: Top-k similar documents
    """
    start_time = time.time()
    query = request.query
    
    # Retrieve similar documents from Chroma
    retrieved = chroma_service.search(query, k=3)
    retrieved_documents = [
        RetrievedDocument(**doc) for doc in retrieved
    ]
    
    # Build context from retrieved documents
    context = "\n\n".join([
        f"Document {i+1} (similarity: {doc.similarity_score}):\n{doc.content}"
        for i, doc in enumerate(retrieved_documents)
    ])
    
    # Get RAG answer (with context)
    rag_result = llm_service.ask_with_context(query, context)
    rag_answer = rag_result['answer']
    rag_cost = rag_result['cost']
    
    # Get non-RAG answer (without context)
    non_rag_result = llm_service.ask_without_context(query)
    non_rag_answer = non_rag_result['answer']
    non_rag_cost = non_rag_result['cost']
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the query
    logger.log_query(
        query=query,
        query_type="rag",
        response={
            "rag_answer": rag_answer,
            "non_rag_answer": non_rag_answer,
            "retrieved_count": len(retrieved_documents)
        },
        latency=time.time() - start_time,
        cost=rag_cost + non_rag_cost
    )
    
    return AskResponse(
        query=query,
        rag_answer=rag_answer,
        non_rag_answer=non_rag_answer,
        retrieved_documents=retrieved_documents,
        latency_ms=round(latency_ms, 2)
    )
