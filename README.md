# ALF Groq AI Agent Backend V4.1 — Compatibility Fix

## Required Render environment
Set ONE of these to your Groq key (GROQ_API_KEY is recommended):
- GROQ_API_KEY
- GROQ_KEY
- GROQ_API_TOKEN
- GROQ_TOKEN

Optional:
- GROQ_MODEL (default: openai/gpt-oss-20b)
- GROQ_FALLBACK_MODEL

The service automatically falls back through:
- openai/gpt-oss-20b
- openai/gpt-oss-120b
- qwen/qwen3.8-27b

## Render
Build:
`pip install -r requirements.txt`

Start (recommended):
`uvicorn main:app --host 0.0.0.0 --port $PORT`

Also compatible with:
`python app.py`
`python server.py`

## Diagnostics
- GET /health
- GET /health/groq
- POST /api/chat

CORS is intentionally open to storefront browser origins; no credentials are accepted and the Groq key remains server-side.


## v4.2 quote bridge fix
- Canonical Get a Quote patch schema.
- Dedicated confirmed-detail extractor for quote fill requests.
- Merges all confirmed details into one quote-update patch.
- Normalizes uniforms, industries, sizes, branding and aliases.

Validated with canonical security quote example:
- Security Uniforms — 50
- Industry: Security
- Color: Blue
- Sizes: S 20 / M 20 / L 10
- Logo placement: chest and back
