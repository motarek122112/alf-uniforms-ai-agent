# ALF Uniforms AI Agent Backend V2

FastAPI backend for the ALF Shopify website assistant using Groq.

## What V2 adds

The AI can now understand real Get a Quote details from normal conversation and prepare a structured form update for:

- Uniform type + quantity
- Company and industry
- Project/team description
- Color and deadline
- Male/female counts
- Size breakdown (S / M / L / XL / XXL / Other)
- Branding method
- Logo placement and logo readiness
- Branding notes
- Contact name, phone, email and Kuwait area
- Follow-up method and best contact time
- Additional notes

Nothing is written into the quotation silently. The AI first summarizes what it understood and returns a **Confirm & fill my quote** action. The Shopify theme applies the fields only after the visitor clicks that confirmation.

## Render update

Replace the previous backend files with this V2 package and redeploy the same Render service. Keep your existing environment variables; you do not need a new URL.

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## Environment variables

- `GROQ_API_KEY` — your Groq API key. Keep it in Render only.
- `GROQ_MODEL` — default `openai/gpt-oss-20b`.
- `ALLOWED_ORIGINS` — e.g. `https://alf-uniforms.myshopify.com`
- `RATE_LIMIT_PER_MINUTE` — default 30.

## Endpoints

- `GET /health`
- `POST /api/chat`

The backend remains stateless. Chat history, active-session memory, the saved Enquiry List, and confirmed quote patches are handled by the Shopify theme/browser.


## V3 language / conversation update
The agent now keeps one primary language per reply, follows the customer's current language instead of the site UI language, uses cleaner line-broken summaries, preserves prior answers, and keeps action labels in the same language as the reply. No environment-variable changes are required from V2.
