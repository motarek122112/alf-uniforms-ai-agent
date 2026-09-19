# ALF Groq AI Backend V4.2 — Rate Limit / 502 Resilience

This build keeps the same `/api/chat` contract used by the Shopify theme.

Changes:
- compact prompt and only last 10 chat messages are sent to Groq to reduce TPM usage
- primary model remains `openai/gpt-oss-20b` by default
- automatic fallback to `groq/compound-mini` on provider 429/5xx/capacity/timeout errors
- lower max completion tokens and low reasoning effort for GPT-OSS
- clearer Render logs showing provider status/model

Environment variables:
- `GROQ_API_KEY` required
- `GROQ_MODEL` optional, default `openai/gpt-oss-20b`
- `GROQ_FALLBACK_MODEL` optional, default `groq/compound-mini`
- `ALLOWED_ORIGINS` optional, default `https://alf-uniforms.myshopify.com`
- `RATE_LIMIT_PER_MINUTE` optional, default `30`

Deploy by replacing the backend repository files and triggering a Render deploy. No Shopify theme change is required if it already points to the same Render URL.
