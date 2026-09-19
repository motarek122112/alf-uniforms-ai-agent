# ALF Uniforms AI Agent Backend V4 — Human Bilingual Sales Concierge

FastAPI + Groq backend for the ALF Shopify assistant.

## What changed in V4

- Arabic/English conversation language is persistent and controlled by the storefront.
- A number, phone, email, date, size code, product name, or short value such as `Printing`, `call`, or `WhatsApp` does not switch the conversation language.
- Explicit language requests such as `عربي`, `بالعربي`, `English`, or a clearly different-language sentence switch the assistant cleanly.
- Language-control messages are never stored as enquiry data.
- The agent sounds like a human sales adviser rather than a numbered 1–20 form.
- The 20 Business Enquiry fields are still enforced internally before final confirmation, but counters and step wording are never shown to the customer.
- Visitors can interrupt with normal questions or small talk; the assistant answers naturally and then returns to the missing requirement.
- Volunteered information is stored in the correct field even if it arrives out of order. Example: `اسمي محمد` is stored as the contact name, never as the company name.
- Uniform type can be remembered before quantity. A later standalone quantity can complete that selected uniform instead of making the customer repeat the product name.
- Refusal is handled politely: the assistant explains why a useful detail helps ALF, while genuinely unknown/not-applicable information can be recorded and left for follow-up.
- Server-side confirmation guard prevents a `quote-update` action until every required field is resolved.

## GitHub / Render update

Replace the backend files in the existing GitHub repository with the files in this package, commit/push, and let the same Render service redeploy. Keep the existing Render URL and environment variables.

- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## Environment variables

- `GROQ_API_KEY` — keep it in Render only.
- `GROQ_MODEL` — defaults to `openai/gpt-oss-20b`.
- `ALLOWED_ORIGINS` — e.g. `https://alf-uniforms.myshopify.com`.
- `RATE_LIMIT_PER_MINUTE` — defaults to 30.

## Endpoints

- `GET /health`
- `POST /api/chat`

The backend is stateless. The Shopify theme keeps the conversation, selected language, structured enquiry draft, and resolved-field state in the browser and sends the current state with every AI request.
