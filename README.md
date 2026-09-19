ALF Groq AI Agent Backend V8 — state-first anti-repeat fix

# ALF Uniforms AI Agent Backend V7 — Customer-First + Rate-Limit Fix

This version fixes the main cause of the repetitive fallback behavior seen after the first successful AI reply.

## What changed
- The system prompt was reduced from roughly 16k characters to about 4.5k characters.
- Only the latest 8 conversation messages are sent to Groq instead of 24.
- The backend sends one Groq request per visitor turn instead of automatically retrying a failed request and potentially doubling token usage.
- `reasoning_effort=low` and a smaller completion budget reduce latency/token use.
- The runtime context sent to the model is compact and does not include the large frontend collector rule text.
- The 20-field enquiry is still enforced before final confirmation, but it stays in the background.
- The local fail-safe is much more conversational and understands Arabic/Gulf phrases such as `هلااا`, `شركتي`, `اسم شركتي الأحمر`, `أبي أزياء`, and requests involving pilots/workers.
- A formal pilot uniform is not falsely presented as an existing ready category; it is handled as a custom-uniform requirement.
- `/health` now shows whether an AI key is configured using `ai_configured` without exposing the key.

## Deployment
Replace the backend files in the same GitHub repository and deploy the existing Render service. Keep `GROQ_API_KEY`. The default model remains `openai/gpt-oss-20b`.

After deployment, open `/health` on the Render service and confirm `status` is `ok` and `ai_configured` is `true`.
