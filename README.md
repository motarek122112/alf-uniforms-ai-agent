# ALF Groq AI Agent Render Backend V4 — GPT Style

Compatible with ALF Uniforms Shopify Theme V22.

## Render environment variables
Required:
- `GROQ_API_KEY` = your Groq API key

Recommended:
- `GROQ_MODEL` = `openai/gpt-oss-20b` (or your existing working Groq model)
- `GROQ_FALLBACK_MODEL` = optional second Groq model
- `ALLOWED_ORIGINS` = comma-separated storefront origins, for example `https://alf-uniforms-kw.myshopify.com`

## Render settings
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

The service exposes:
- `GET /health`
- `POST /api/chat`

The chat contract is JSON:
`{ reply, actions, auto_action, context }`
