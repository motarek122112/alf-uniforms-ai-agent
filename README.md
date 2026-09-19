# ALF Groq AI Agent Backend V9 — Semantic State / Customer First

This version fixes the remaining conversation loop where the visitor could clarify a role, ask for options, or provide a quantity and the assistant would fall back to the same uniform-type question.

## Main changes
- State is semantic rather than tied to one rigid expected field.
- Visitor corrections are authoritative: `ارضي` changes pilot context to airport ground crew; later explicit pilot wording can change it back.
- `Custom Uniform` is a valid structured enquiry item for formal pilot-uniform requirements.
- Multi-fact turns can capture several details at once, including total quantity and male/female counts.
- Company extraction stops before the sentence moves into a uniform request.
- Industry/project details can be captured whenever the visitor volunteers them, not only when that exact field is pending.
- The local fail-safe always returns a useful contextual reply; it no longer returns an empty response that causes the storefront to repeat a fixed collector question.
- Side questions and recommendation requests are answered first; enquiry completion remains a background goal.

## Deploy
Replace the files in the existing GitHub backend repository, commit/push, and let the same Render service redeploy. Keep the existing `GROQ_API_KEY` and other environment variables.

After deployment, `/health` should report:
- `version`: `1.9.0`
- `architecture`: `semantic-state-customer-first`
- `ai_configured`: `true`
