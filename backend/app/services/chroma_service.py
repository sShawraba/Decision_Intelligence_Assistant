"""Chroma vector store service for RAG"""
import chromadb
from app.utils.config import CHROMA_PATH, TOP_K
#from app.services.embedding_service import get_embedding #coupled to openai



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

        if ids is None:
            ids = [str(i) for i in range(len(documents))]

        # embeddings = [get_embedding(doc) for doc in documents]

        # self.collection.add(
        #     documents=documents,
        #     embeddings=embeddings,
        #     ids=ids
        # )

        self.collection.add( 
            documents=documents,
            ids=ids
        )   #chromadb handles embeddings to work on gemini

    def search(self, query: str, k: int = TOP_K):

        # query_embedding = get_embedding(query)

        # results = self.collection.query(
        #     query_embeddings=[query_embedding],
        #     n_results=k
        # )
        
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
