ALF Backend V4.1 - Render 502 fix

Changes:
- Explicit include_reasoning=False for Groq GPT-OSS JSON responses.
- One safe fallback retry without response_format if Groq rejects JSON/reasoning request formatting.
- Exact upstream exception is printed to Render logs for diagnosis.
- Storefront still receives a safe generic error; API keys are never exposed.
- Same GROQ_API_KEY, GROQ_MODEL and ALLOWED_ORIGINS environment variables can be kept.

Deploy:
1. Replace the repository files with this package.
2. Commit/push to the branch connected to Render.
3. Wait for Deploy succeeded / Live.
4. Open Render Logs, then send one chatbot message.
5. If it still fails, search logs for "Groq primary request failed" or "Groq final failure".
