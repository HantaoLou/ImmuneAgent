# Bio-Agent Backend

FastAPI backend for Bio-Agent Demo System.

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Running

```bash
python -m uvicorn main:app --reload --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `POST /api/chat` - Chat endpoint with SSE streaming

## Configuration

Environment variables:
- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8000)
- `API_RELOAD` - Enable auto-reload (default: true)
- `SANDBOX_BASE_DIR` - Sandbox directory (default: ./sandbox)
- `AGENT_EXECUTOR_MODE` - Agent execution mode (default: iterative)
