# ALF Groq AI Agent Backend V10 — Natural Dialogue + Deterministic Memory

This version separates **conversation quality** from **enquiry memory**.

The AI is free to answer naturally in Arabic or English, while the storefront/backend keeps the structured enquiry state separately. The model is no longer forced to generate a JSON object containing both dialogue and state on every message.

## Why this fixes the bad conversation loop
- The model returns normal conversational text only.
- Obvious facts are captured deterministically before the AI call, so `100`, `نعم`, `لا`, company name, project context, etc. do not disappear.
- `collection.current_field` is only a hint to the assistant, not a hard scripted step.
- Arabic remains the active language until the visitor clearly switches language.
- Repeated company / industry / quantity questions are blocked by persistent structured state.
- Formal pilot requirements are handled as `Custom Uniform`, not incorrectly presented as standard Polo/Workwear pilot uniforms.
- Provider failure falls back to `openai/gpt-oss-20b`; if both fail, the local state-aware assistant still keeps the conversation moving.

## Recommended Render environment

```text
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
```

Keep your existing `GROQ_API_KEY`.

## Deploy
Replace the backend repository files with this package, commit/push, and let the same Render service redeploy.

After deployment `/health` should report version `2.0.0` and architecture `natural-dialogue-deterministic-memory`.
