"""LLM service for calling OpenAI API"""
from openai import OpenAI
from app.utils.config import OPENAI_API_KEY, MODEL_NAME


class LLMService:
    """Service for calling OpenAI API"""

    def __init__(self):
        """Initialize OpenAI client"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = MODEL_NAME

    def ask_with_context(self, query: str, context: str) -> dict:
        """
        Ask LLM a question with RAG context.
        
        Args:
            query: The user's question
            context: Retrieved context from Chroma
            
        Returns:
            Dictionary with 'answer' and 'cost'
        """
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        return self._call_llm(prompt)

    def ask_without_context(self, query: str) -> dict:
        """
        Ask LLM a question without any context.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary with 'answer' and 'cost'
        """
        prompt = f"Question: {query}\n\nAnswer:"
        return self._call_llm(prompt)

    def predict_priority(self, text: str) -> dict:
        """
        Use LLM to predict priority in a zero-shot manner.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with 'priority' (HIGH/MEDIUM/LOW) and 'reasoning'
        """
        prompt = f"""Classify the following text into one of these priority levels: HIGH, MEDIUM, LOW.
Respond with ONLY two lines:
Line 1: The priority level (HIGH/MEDIUM/LOW)
Line 2: Brief reasoning (1-2 sentences)

Text: {text}

Response:"""

        result = self._call_llm(prompt)
        
        # Parse the response
        lines = result['answer'].strip().split('\n')
        priority = lines[0].strip().upper() if lines else "MEDIUM"
        reasoning = lines[1].strip() if len(lines) > 1 else ""
        
        return {
            "priority": priority,
            "reasoning": reasoning,
            "cost": result['cost']
        }

    def _call_llm(self, prompt: str) -> dict:
        """
        Internal method to call OpenAI API.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Dictionary with 'answer' and 'cost'
        """
        if not OPENAI_API_KEY:
            return {
                "answer": "Error: OPENAI_API_KEY not configured",
                "cost": 0.0
            }
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Calculate cost (rough estimation for GPT-3.5)
            # Actual pricing: https://openai.com/pricing
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # GPT-3.5-turbo pricing: $0.0005 per 1K input tokens, $0.0015 per 1K output tokens
            cost = (input_tokens * 0.0005 + output_tokens * 0.0015) / 1000
            
            return {
                "answer": answer,
                "cost": round(cost, 6)
            }
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "cost": 0.0
            }


# Global instance
llm_service = LLMService()
