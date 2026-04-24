/**
 * API client for communicating with the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Ask a question and get RAG answers
 */
export async function askQuestion(query) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error asking question:", error);
    throw error;
  }
}

/**
 * Get ML model prediction
 */
export async function predictWithML(text) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error predicting with ML:", error);
    throw error;
  }
}

/**
 * Get LLM priority prediction
 */
export async function predictWithLLM(text) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/llm-priority`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error predicting with LLM:", error);
    throw error;
  }
}
