# ALF Uniforms AI Agent Backend

FastAPI backend for the ALF Shopify website assistant using Groq.

## Render settings

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## Environment variables

- `GROQ_API_KEY` — your Groq API key. Keep it in Render only; never put it in Shopify theme code.
- `GROQ_MODEL` — default `openai/gpt-oss-20b`. You can later switch to `openai/gpt-oss-120b` without changing code.
- `ALLOWED_ORIGINS` — comma-separated storefront origins. Start with `https://alf-uniforms.myshopify.com`; add the final custom domain when connected.
- `RATE_LIMIT_PER_MINUTE` — default 30.

## Endpoints

- `GET /health`
- `POST /api/chat`

The backend is stateless. The Shopify theme keeps the active-session conversation and old-chat history in the browser, and sends the relevant conversation history to this backend each turn.
