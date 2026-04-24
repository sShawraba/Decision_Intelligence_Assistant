"""Logging utilities for tracking queries, outputs, and costs"""
import json
import time
from datetime import datetime
from pathlib import Path
from app.utils.config import LOGS_PATH


class Logger:
    """Simple logger for tracking queries and responses"""

    def __init__(self):
        self.log_file = Path(LOGS_PATH) / "queries.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_query(self, query: str, query_type: str, response: dict, latency: float, cost: float = 0.0):
        """
        Log a query and its response to a JSONL file.
        
        Args:
            query: The input query text
            query_type: Type of query (rag, ml, llm)
            response: The response dictionary
            latency: Time taken to process (seconds)
            cost: Cost of the request (for LLM API calls)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_type": query_type,
            "latency_seconds": round(latency, 3),
            "cost_usd": round(cost, 6),
            "response": response,
        }

        # Append to JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# Global logger instance
logger = Logger()
