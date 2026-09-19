# ALF Groq AI Agent Backend V11 — Agent Orchestrator

This build keeps the AI conversational while the Shopify theme owns the reliable agent behavior.

## What changed
- `openai/gpt-oss-20b` is the default production model; `qwen/qwen3.6-27b` is fallback.
- The model does **not** control navigation, quote submission, form filling, or the next required field.
- The model answers naturally in the active language; the storefront appends exactly one next enquiry question when required.
- The model is explicitly forbidden from inventing URLs, quote IDs, order IDs, delivery guarantees, or pretending a page/form action already happened.
- Urgent dates such as “tomorrow” are treated as requested targets only; ALF still has to confirm feasibility.
- Improved Arabic context extraction for workers, engineering, company naming, girls/women and men counts.
- Provider replies are cleaned before reaching the storefront.

## Recommended Render environment

```text
GROQ_MODEL=openai/gpt-oss-20b
GROQ_FALLBACK_MODEL=qwen/qwen3.6-27b
```

Keep the existing `GROQ_API_KEY` and Render URL.

## Deploy
Replace the current backend repository files with this package, commit/push, and let the same Render service redeploy.

After deployment `/health` should report:

```json
{"version":"2.1.0","architecture":"agent-orchestrated-natural-dialogue"}
```
