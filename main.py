import json
import os
import re
import time
from collections import defaultdict, deque
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

APP_NAME = "ALF Uniforms AI Agent"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

DEFAULT_ORIGINS = "https://alf-uniforms.myshopify.com"
ALLOWED_ORIGINS = [
    x.strip().rstrip("/")
    for x in re.split(r"[,;\n]", os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS))
    if x.strip()
]

UNIFORMS = [
    "Polo Shirts & T-Shirts",
    "Chef Uniforms & Aprons",
    "Chef Caps & Accessories",
    "Cargo Pants & Workwear",
    "Security Uniforms",
    "Event & Promo Team Apparel",
]

ROUTES = {
    "Polo Shirts & T-Shirts": "/pages/polo-t-shirts",
    "Chef Uniforms & Aprons": "/pages/chef-uniforms",
    "Chef Caps & Accessories": "/pages/chef-uniforms",
    "Cargo Pants & Workwear": "/pages/workwear",
    "Security Uniforms": "/pages/security-uniforms",
    "Event & Promo Team Apparel": "/pages/event-uniforms",
}

STATIC_ROUTES = {
    "/",
    "/#real-work",
    "/pages/custom-uniforms",
    "/pages/embroidery-printing",
    "/pages/bulk-orders",
    "/pages/ready-stock",
    "/pages/how-it-works",
    "/pages/polo-t-shirts",
    "/pages/chef-uniforms",
    "/pages/workwear",
    "/pages/security-uniforms",
    "/pages/event-uniforms",
}

SYSTEM_PROMPT = f"""
You are the ALF Uniforms Website Assistant for a Kuwait uniform supplier.
Your job is to help a website visitor make a useful buying decision and move through the actual ALF enquiry journey. You are not a generic chatbot.

BUSINESS RULES
- ALF sells custom uniforms for organizations and teams in Kuwait.
- The website is enquiry/quotation based, not fixed unit-price ecommerce.
- Minimum order starts from 12 pieces per uniform type. Never invent a quantity for the visitor.
- Never invent a unit price, final quotation, delivery promise, stock status, client name, completed project, testimonial, or production time.
- ALF confirms the final quotation after reviewing uniform type, quantity, sizes and branding.
- For large/bulk requirements, guide the visitor through the bulk/enquiry flow.
- Real ALF Work means only the real-work section populated by the ALF team. Do not invent project proof.
- When a human is needed, offer WhatsApp/human help.

UNIFORM CATEGORIES
{json.dumps(UNIFORMS, ensure_ascii=False)}

CATEGORY ROUTES
{json.dumps(ROUTES, ensure_ascii=False)}

WEBSITE ROUTES
- Uniform range: /pages/custom-uniforms
- Branding: /pages/embroidery-printing
- Bulk orders: /pages/bulk-orders
- Ready stock: /pages/ready-stock
- How it works: /pages/how-it-works
- Real ALF Work: /#real-work
- Fresh quote: /pages/get-a-quote?mode=fresh&intent=general
- Continue saved Enquiry List: /pages/get-a-quote?mode=selected&intent=general
- Branding quote: /pages/get-a-quote?mode=fresh&intent=branding
- Bulk quote: /pages/get-a-quote?mode=fresh&intent=bulk
- Similar real project: /pages/get-a-quote?mode=fresh&intent=project

IMPORTANT JOURNEY DIFFERENCES
1. "Add to Enquiry List" saves only a uniform type. It does NOT assume quantity. The visitor can collect multiple uniform types while browsing.
2. "Continue with my selection" opens the quotation with those exact saved uniforms already selected. The visitor then adds quantity for each, sizes, branding and contact details.
3. "Get a Quote" / Fresh quote intentionally starts from zero when the visitor did not come from a saved Enquiry List.
4. "Quote this uniform now" starts a quote with the current uniform already selected.
5. Branding, Bulk Orders, Real Work and Human Help should each lead to their relevant flow, not all to the same generic result.

AGENT ACTIONS
You may return UI actions. Allowed action types:
- navigate: value is one of the ALF internal routes above.
- add: value must be exactly one uniform category from UNIFORM CATEGORIES.
- enquiry-list: opens the saved enquiry list; no value required.
- whatsapp: value is a short message to ALF staff.
- prompt: value is a suggested user message for the chat.

If the user explicitly asks you to open/go to a page, add a uniform, open the enquiry list, or contact WhatsApp, you may set ONE auto_action matching that explicit request. Do not auto-execute a purchase-like commitment. Adding to an enquiry list is allowed because it is only a shortlist.

RECOMMENDATION BEHAVIOR
- Ask one short clarifying question when the team/use case is unclear.
- For restaurant/cafe/kitchen: start with Chef Uniforms & Aprons; front-of-house can also use Polo Shirts & T-Shirts.
- Corporate/office/reception/sales: Polo Shirts & T-Shirts.
- Warehouse/maintenance/logistics/operations: Cargo Pants & Workwear.
- Security/guards: Security Uniforms.
- Exhibitions/promotional/event staff: Event & Promo Team Apparel; polos may also fit a cleaner corporate style.
- Be concise and commercial, but never pressure the visitor or make unsupported claims.
- Reply in the user's language. If they use Arabic, use clear conversational Arabic suitable for Kuwait; if English, use concise professional English.

OUTPUT
Return ONLY a valid JSON object with this exact top-level structure:
{{
  "reply": "short helpful answer",
  "actions": [{{"label":"...","type":"navigate|add|enquiry-list|whatsapp|prompt","value":"..."}}],
  "auto_action": null OR {{"label":"...","type":"navigate|add|enquiry-list|whatsapp","value":"..."}},
  "context": {{"lastUniform":"","industry":"","quantity":0}}
}}
Keep actions to 0-3 useful choices. Do not output markdown.
""".strip()

app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Lightweight abuse protection. Render instances are ephemeral, so this is intentionally simple.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
_hits: dict[str, deque] = defaultdict(deque)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2500)


class PageState(BaseModel):
    path: str = Field(default="/", max_length=300)
    label: str = Field(default="Home", max_length=160)
    title: str = Field(default="ALF Uniforms", max_length=220)


class EnquiryItem(BaseModel):
    name: str = Field(max_length=120)


class AgentContext(BaseModel):
    lastUniform: str = Field(default="", max_length=120)
    industry: str = Field(default="", max_length=120)
    quantity: int = Field(default=0, ge=0, le=100000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)
    page: PageState = Field(default_factory=PageState)
    enquiry: list[EnquiryItem] = Field(default_factory=list, max_length=20)
    context: AgentContext = Field(default_factory=AgentContext)
    locale: str = Field(default="en", max_length=20)


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again shortly.")
    q.append(now)


def _safe_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("Model did not return JSON")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object")
    return data


def _valid_navigation(value: str) -> bool:
    if value in STATIC_ROUTES:
        return True
    if value.startswith("/pages/get-a-quote?"):
        return True
    return value in ROUTES.values()


def _clean_action(raw: Any, allow_prompt: bool = True) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    atype = str(raw.get("type", "")).strip()
    label = str(raw.get("label", "")).strip()[:80]
    value = str(raw.get("value", "")).strip()[:500]
    allowed = {"navigate", "add", "enquiry-list", "whatsapp"}
    if allow_prompt:
        allowed.add("prompt")
    if atype not in allowed:
        return None
    if not label:
        label = {
            "navigate": "Open page",
            "add": "Add to Enquiry List",
            "enquiry-list": "My enquiry list",
            "whatsapp": "Human help",
            "prompt": "Continue",
        }[atype]
    if atype == "navigate" and not _valid_navigation(value):
        return None
    if atype == "add" and value not in UNIFORMS:
        return None
    if atype == "prompt" and not value:
        return None
    if atype == "whatsapp" and not value:
        value = "Hello ALF Uniforms, I need help with a uniform requirement."
    if atype == "enquiry-list":
        value = ""
    return {"label": label, "type": atype, "value": value}


def _clean_context(raw: Any, fallback: AgentContext) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    last_uniform = str(raw.get("lastUniform", fallback.lastUniform) or "")
    if last_uniform and last_uniform not in UNIFORMS:
        last_uniform = fallback.lastUniform if fallback.lastUniform in UNIFORMS else ""
    industry = str(raw.get("industry", fallback.industry) or "")[:120]
    try:
        quantity = int(raw.get("quantity", fallback.quantity) or 0)
    except Exception:
        quantity = fallback.quantity
    quantity = max(0, min(quantity, 100000))
    return {"lastUniform": last_uniform, "industry": industry, "quantity": quantity}


@app.get("/")
def root() -> dict[str, str]:
    return {"service": APP_NAME, "status": "online", "model": MODEL}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL}


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    _rate_limit(request)
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on the server.")

    enquiry_names = [item.name for item in payload.enquiry if item.name in UNIFORMS]
    runtime_context = {
        "current_page": payload.page.model_dump(),
        "saved_enquiry_uniforms": enquiry_names,
        "session_context": payload.context.model_dump(),
        "locale": payload.locale,
    }

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "CURRENT WEBSITE STATE:\n" + json.dumps(runtime_context, ensure_ascii=False),
        },
    ]
    messages.extend({"role": m.role, "content": m.content} for m in payload.messages[-24:])

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.25,
            max_completion_tokens=900,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or ""
        data = _safe_json(content)
    except HTTPException:
        raise
    except Exception as exc:
        # Do not leak keys or provider internals to the storefront.
        raise HTTPException(status_code=502, detail="The AI service is temporarily unavailable.") from exc

    reply = str(data.get("reply", "")).strip()[:3000]
    if not reply:
        reply = "I can help you choose the right ALF uniform and continue the correct enquiry flow."

    actions = []
    for raw in (data.get("actions") or [])[:3]:
        cleaned = _clean_action(raw, allow_prompt=True)
        if cleaned:
            actions.append(cleaned)

    auto_action = _clean_action(data.get("auto_action"), allow_prompt=False) if data.get("auto_action") else None
    context = _clean_context(data.get("context"), payload.context)

    return {
        "reply": reply,
        "actions": actions,
        "auto_action": auto_action,
        "context": context,
        "model": MODEL,
    }
