# ALF Uniforms AI Agent Backend V3 — Sales Concierge

FastAPI backend for the ALF Shopify website assistant using Groq.

## What V3 adds

- Sales-concierge personality: proactive, concise, human-like and focused on resolving the customer's requirement instead of ending the chat quickly.
- Structured quote-draft memory across the conversation.
- The agent collects the requirement in chat before sending the visitor to Get a Quote.
- It avoids re-asking information already provided.
- It collects uniform + quantity, team/project details, branding and contact/follow-up details.
- It summarizes the complete requirement and asks for explicit confirmation before the Shopify form is changed.
- The confirmed quote-update carries the complete draft so all applicable fields across the 4 Get a Quote steps can be filled at once.
- Navigation actions can include a same-language follow-up question that the theme displays as a temporary notification after the destination page loads.
- The agent is instructed to continue helping after a rejected recommendation instead of closing the conversation.

## Render update

Replace the previous backend files in the same GitHub repository with this V3 package and redeploy the same Render service. Keep the existing Render URL and environment variables.

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## Environment variables

- `GROQ_API_KEY` — keep it in Render only.
- `GROQ_MODEL` — default `openai/gpt-oss-20b`.
- `ALLOWED_ORIGINS` — e.g. `https://alf-uniforms.myshopify.com`
- `RATE_LIMIT_PER_MINUTE` — default 30.

## Endpoints

- `GET /health`
- `POST /api/chat`

The backend remains stateless. The browser/theme stores the active conversation, old chats and structured quote draft; the backend receives that state on each request and returns the merged draft.


## V4 conversational sales-concierge changes
- Persistent Arabic/English conversation language; numbers, sizes, emails, product names, and short option words do not flip the language.
- Active collection state is now sent to the AI so it understands what is already known, unknown, or still missing.
- The 20-field business-enquiry checklist is internal only; the visitor no longer sees progress counts or robotic field numbering.
- Semantic extraction is emphasized (for example, “my name is Mohamed” is a contact name, not a company name).
- Light conversation and side questions are answered naturally before returning to the enquiry.
- Optional `collection_updates` lets the model explicitly mark fields as genuinely unknown or not applicable without inventing values.
