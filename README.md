# Decision Intelligence Assistant

A full-stack AI application combining **RAG** (Retrieval-Augmented Generation), **ML** classification, and **LLM** reasoning for intelligent decision support.

## Tech Stack

- **Backend**: FastAPI + Python
- **Frontend**: React + Vite
- **Vector Store**: Chroma (local, persistent)
- **ML**: scikit-learn (offline training, loaded at runtime)
- **LLM**: OpenAI API (with cost tracking)
- **Environment**: Python with uv
- **Containerization**: Docker + docker-compose

## Project Structure

```
decision-intelligence-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app entry point
│   │   ├── routers/                # API endpoints
│   │   │   ├── rag.py              # RAG endpoint
│   │   │   ├── ml.py               # ML prediction endpoint
│   │   │   └── llm.py              # LLM priority endpoint
│   │   ├── schemas/                # Pydantic models
│   │   │   └── models.py           # Request/response schemas
│   │   ├── services/               # Business logic
│   │   │   ├── chroma_service.py   # Vector store operations
│   │   │   ├── ml_service.py       # Model loading/inference
│   │   │   ├── llm_service.py      # OpenAI API calls
│   │   │   └── feature_extraction.py  # Feature engineering
│   │   └── utils/
│   │       ├── config.py           # Configuration
│   │       └── logger.py           # Query logging
│   ├── Dockerfile
│   ├── pyproject.toml              # Python dependencies
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── main.jsx                # React entry point
│   │   ├── App.jsx                 # Main component
│   │   ├── App.css                 # Styles
│   │   ├── index.css               # Global styles
│   │   └── services/
│   │       └── api.js              # API client
│   ├── Dockerfile
│   ├── vite.config.js              # Vite configuration
│   ├── package.json
│   ├── index.html
│   └── .env.example
├── docker-compose.yml
├── .gitignore
├── .env.example                    # Root env file
└── README.md
```

## Quick Start

### Prerequisites

- Docker & Docker Compose (recommended)
- OR: Python 3.11+, Node.js 18+, OpenAI API key

### Using Docker Compose (Recommended)

1. **Clone and setup**:
   ```bash
   cd decision-intelligence-assistant
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Place your trained model**:
   ```bash
   cp /path/to/your/model.pkl ./model.pkl
   ```

3. **Start services**:
   ```bash
   docker-compose up --build
   ```

4. **Access**:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Local Development (Without Docker)

#### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env with your settings
   
   # Using uv (recommended)
   uv sync
   ```

2. **Prepare your model**:
   ```bash
   cp /path/to/your/model.pkl ./
   ```

3. **Run backend**:
   ```bash
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup

1. **Install Node dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Run dev server**:
   ```bash
   npm run dev
   ```

3. **Access**: http://localhost:5173

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
# Required
OPENAI_API_KEY=sk-...your-key-here...

# Optional
MODEL_NAME=gpt-3.5-turbo          # OpenAI model (default: gpt-3.5-turbo)
CHROMA_PATH=./chroma_data         # Chroma storage path
MODEL_PATH=./model.pkl            # Path to your trained model
LOGS_PATH=./logs                  # Query logs path
```

## Logging & Monitoring

All queries are logged to `logs/queries.jsonl` with:
- Timestamp
- Query text
- Query type (rag, ml, llm)
- Response
- Latency
- Cost (for LLM calls)

View logs:
```bash
tail -f logs/queries.jsonl | jq
```

## Development Tips

1. **API Documentation**: Visit http://localhost:8000/docs for interactive Swagger UI
2. **Hot Reload**: Both backend (with `--reload`) and frontend (Vite) support hot reload
3. **Cost Tracking**: Check the frontend UI for real-time cost of LLM calls
4. **Adding Features**:
   - New endpoints: Create a router in `backend/app/routers/`
   - New services: Add to `backend/app/services/`
   - New frontend components: Add to `frontend/src/`

## Common Issues

### "OPENAI_API_KEY not set"
- Make sure `.env` file exists in the root directory
- Add your OpenAI API key to `.env`

### "Model file not found"
- Place your trained `model.pkl` in the project root
- Update `MODEL_PATH` in `.env` if needed

### Port already in use
- Backend: Change port in docker-compose.yml or via `--port` flag
- Frontend: Change port in vite.config.js

### CORS errors
- Backend already allows frontend origin in `main.py`
- If using different port, update the CORS config

## Next Steps

1. **Add your data**: Create and populate Chroma with your documents
2. **Train your model**: Prepare and save your scikit-learn model
3. **Test endpoints**: Use the Swagger UI at `/docs`
4. **Deploy**: Use Docker Compose or adapt to your cloud platform
5. **Customize**: Modify prompts, features, and UI to fit your use case

## License

MIT - Feel free to use and modify for your bootcamp project!

## Support

For issues, check:
1. Logs in `logs/queries.jsonl`
2. Backend Swagger UI: http://localhost:8000/docs
3. Browser console for frontend errors
4. Docker logs: `docker-compose logs -f`
