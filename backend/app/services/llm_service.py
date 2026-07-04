"""LLM service for calling OpenAI API"""
from openai import OpenAI
from app.utils.config import OPENAI_API_KEY, MODEL_NAME, GEMINI_API_KEY
import google.generativeai as genai

class LLMService:
    """Service for calling OpenAI API"""

    def __init__(self):
        """Initialize OpenAI client"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = MODEL_NAME

        # Gemini setup
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.gemini_model = None

    def ask_with_context(self, query: str, context: str) -> dict:
        """
        Ask LLM a question with RAG context.
        """
        prompt = f"""
You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question:
{query}

Answer:
"""
        return self._call_llm(prompt)

    def ask_without_context(self, query: str) -> dict:
        """
        Ask LLM a question without any context.
        """
        prompt = f"""
Question:
{query}

Answer:
"""
        return self._call_llm(prompt)

    def predict_priority(self, text: str) -> dict:
        """
        Use LLM to predict priority in a zero-shot manner.
        """
        prompt = f"""
Classify the following text into one of these priority levels: HIGH, MEDIUM, LOW.

Respond with ONLY two lines:
Line 1: The priority level (HIGH/MEDIUM/LOW)
Line 2: Brief reasoning (1-2 sentences)

Text:
{text}

Response:
"""

        result = self._call_llm(prompt)

        lines = result["answer"].strip().split("\n")
        priority = lines[0].strip().upper() if lines else "MEDIUM"
        reasoning = lines[1].strip() if len(lines) > 1 else ""

        return {
            "priority": priority,
            "reasoning": reasoning,
            "cost": result["cost"]
        }

    def _call_llm(self, prompt: str) -> dict:

        # ===== TRY OPENAI FIRST =====
        if OPENAI_API_KEY:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )

                answer = response.choices[0].message.content

                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                cost = (input_tokens * 0.0005 + output_tokens * 0.0015) / 1000

                return {
                    "answer": answer,
                    "cost": round(cost, 6),
                    "provider": "openai"
                }

            except Exception as e:
                print(f"OpenAI failed → switching to Gemini: {e}")

        # ===== FALLBACK TO GEMINI =====
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)

                return {
                    "answer": response.text,
                    "cost": 0.0,  # Gemini free tier or unknown cost
                    "provider": "gemini"
                }

            except Exception as e:
                print(f"Gemini also failed: {e}")

        # ===== TOTAL FAILURE =====
        return {
            "answer": "Error: All LLM providers failed",
            "cost": 0.0,
            "provider": "none"
        }


# Global instance
llm_service = LLMService()