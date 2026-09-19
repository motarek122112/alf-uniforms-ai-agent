# ALF Uniforms AI Agent Backend V6 — Customer-First Human Sales Concierge

This version makes the visitor's latest intent the first priority. The 20-field enquiry remains enforced before final confirmation, but it is now a quiet background goal rather than the visible conversation script. Recommendation requests, comparisons, normal small talk, and clarifications are answered first.

It also improves the local fail-safe so provider downtime does not turn the bot into a repetitive form: Arabic greetings such as elongated "هلاااا" are understood, "اقترح انت" produces a contextual recommendation, and direct field answers advance to the next missing detail instead of repeating the previous question.

Deployment: replace the backend files in the same GitHub repository, keep the existing GROQ_API_KEY / Render environment variables, and redeploy.
