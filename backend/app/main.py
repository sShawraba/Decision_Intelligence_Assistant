"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import rag, ml, llm

# Create FastAPI app
app = FastAPI(
    title="Decision Intelligence Assistant",
    description="RAG + ML + LLM decision support system",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite + production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag.router)
app.include_router(ml.router)
app.include_router(llm.router)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Decision Intelligence Assistant API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check for container orchestration"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
