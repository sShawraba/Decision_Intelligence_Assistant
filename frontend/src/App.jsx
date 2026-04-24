import { useState } from "react";
import "./App.css";
import { askQuestion, predictWithML, predictWithLLM } from "./services/api";

function App() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // RAG results
  const [ragAnswer, setRagAnswer] = useState("");
  const [nonRagAnswer, setNonRagAnswer] = useState("");
  const [retrievedDocs, setRetrievedDocs] = useState([]);
  const [ragLatency, setRagLatency] = useState(0);

  // ML results
  const [mlLabel, setMlLabel] = useState("");
  const [mlConfidence, setMlConfidence] = useState(0);
  const [mlLatency, setMlLatency] = useState(0);

  // LLM results
  const [llmPriority, setLlmPriority] = useState("");
  const [llmReasoning, setLlmReasoning] = useState("");
  const [llmLatency, setLlmLatency] = useState(0);
  const [llmCost, setLlmCost] = useState(0);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setLoading(true);
    setError("");
    setRagAnswer("");
    setNonRagAnswer("");
    setRetrievedDocs([]);
    setMlLabel("");
    setLlmPriority("");

    try {
      // Call all three endpoints in parallel
      const [ragResult, mlResult, llmResult] = await Promise.all([
        askQuestion(input),
        predictWithML(input),
        predictWithLLM(input),
      ]);

      // Set RAG results
      setRagAnswer(ragResult.rag_answer);
      setNonRagAnswer(ragResult.non_rag_answer);
      setRetrievedDocs(ragResult.retrieved_documents);
      setRagLatency(ragResult.latency_ms);

      // Set ML results
      setMlLabel(mlResult.label);
      setMlConfidence(mlResult.confidence);
      setMlLatency(mlResult.latency_ms);

      // Set LLM results
      setLlmPriority(llmResult.priority);
      setLlmReasoning(llmResult.reasoning);
      setLlmLatency(llmResult.latency_ms);
      setLlmCost(llmResult.cost_usd);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getPriorityClass = (priority) => {
    return priority.toLowerCase();
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Decision Intelligence Assistant</h1>
        <p>
          Ask a question to get RAG-powered answers, ML predictions, and LLM
          insights
        </p>
      </div>

      {/* Input Section */}
      <div className="input-section">
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Enter your query or text here..."
              disabled={loading}
            />
            <button type="submit" disabled={loading}>
              {loading ? "Processing..." : "Submit"}
            </button>
          </div>
        </form>
      </div>

      {/* Error Display */}
      {error && <div className="error">{error}</div>}

      {/* Loading Indicator */}
      {loading && <div className="loading">Processing your request...</div>}

      {/* Results Section */}
      {(ragAnswer || nonRagAnswer || mlLabel || llmPriority) && !loading && (
        <>
          <div className="results-section">
            {/* RAG Answer */}
            {ragAnswer && (
              <div className="result-card">
                <h3>RAG Answer</h3>
                <div className="answer-text">{ragAnswer}</div>
                <div className="metadata">
                  <span>
                    <strong>Latency:</strong> {ragLatency}ms
                  </span>
                </div>
              </div>
            )}

            {/* Non-RAG Answer */}
            {nonRagAnswer && (
              <div className="result-card">
                <h3>Direct LLM Answer</h3>
                <div className="answer-text">{nonRagAnswer}</div>
                <div className="metadata">
                  <span>
                    <strong>No context used</strong>
                  </span>
                </div>
              </div>
            )}

            {/* Retrieved Documents */}
            {retrievedDocs.length > 0 && (
              <div className="result-card retrieved-section">
                <h3>Retrieved Documents ({retrievedDocs.length})</h3>
                {retrievedDocs.map((doc, idx) => (
                  <div key={idx} className="document-item">
                    <p>{doc.content}</p>
                    <div className="similarity-score">
                      Similarity: {(doc.similarity_score * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Comparison Section */}
          <div className="comparison-section">
            <h2>Comparison: ML vs LLM Priority Prediction</h2>
            <div className="comparison-grid">
              {/* ML Prediction */}
              {mlLabel && (
                <div className="comparison-item">
                  <h4>ML Model</h4>
                  <div className="label">{mlLabel}</div>
                  <p>
                    <strong>Confidence:</strong> {(mlConfidence * 100).toFixed(1)}%
                  </p>
                  <p>
                    <strong>Latency:</strong> {mlLatency}ms
                  </p>
                </div>
              )}

              {/* LLM Prediction */}
              {llmPriority && (
                <div className="comparison-item">
                  <h4>LLM (GPT)</h4>
                  <div className={`label ${getPriorityClass(llmPriority)}`}>
                    {llmPriority}
                  </div>
                  <p>
                    <strong>Reasoning:</strong> {llmReasoning}
                  </p>
                  <p>
                    <strong>Latency:</strong> {llmLatency}ms
                  </p>
                  <p>
                    <strong>Cost:</strong> ${llmCost.toFixed(6)}
                  </p>
                </div>
              )}

              {/* Summary */}
              <div className="comparison-item">
                <h4>Summary</h4>
                <p>
                  <strong>ML Speed:</strong> Instant (local)
                </p>
                <p>
                  <strong>LLM Quality:</strong> Better reasoning
                </p>
                <p>
                  <strong>Total Cost:</strong> ${llmCost.toFixed(6)}
                </p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
