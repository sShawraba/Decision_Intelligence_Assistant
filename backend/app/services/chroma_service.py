"""Chroma vector store service for RAG"""
import chromadb
from app.utils.config import CHROMA_PATH, TOP_K


class ChromaService:
    """Service for interacting with Chroma vector store"""

    def __init__(self):
        """Initialize Chroma client with persistent storage"""
        # Use persistent client for local file storage
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Get or create a collection
        # Note: Chroma automatically handles embeddings with default model
        self.collection = self.client.get_or_create_collection(
            name="tweets",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: list[str], ids: list[str] = None):
        """
        Add documents to the Chroma collection.
        
        Args:
            documents: List of document texts to add
            ids: List of document IDs (optional, auto-generated if not provided)
        """
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            ids=ids
        )

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query text
            k: Number of top results to return
            
        Returns:
            List of dictionaries with 'content' and 'similarity_score'
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # Chroma returns distances, convert to similarity scores (1 - distance for cosine)
        formatted_results = [
            {
                "content": doc,
                "similarity_score": round(1 - distance, 3)  # Convert distance to similarity
            }
            for doc, distance in zip(documents, distances)
        ]
        
        return formatted_results

    def clear(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(name="tweets")
        self.collection = self.client.get_or_create_collection(
            name="tweets",
            metadata={"hnsw:space": "cosine"}
        )


# Global instance
chroma_service = ChromaService()
