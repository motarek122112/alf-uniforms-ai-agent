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
- quote-update: a CONFIRMATION action that carries a structured patch for the Get a Quote form. Never auto-execute quote-update.

QUOTE FORM ASSISTANCE — VERY IMPORTANT
The assistant can help fill the real Get a Quote form from the visitor's natural-language conversation, but ONLY after explicit confirmation.
When the visitor gives concrete quotation details that match one or more fields below:
1. Extract ONLY information the visitor actually stated or clearly confirmed. Never guess missing values.
2. Reply with a concise summary of what you understood and ask the visitor to confirm before applying it.
3. Include ONE quote-update action labelled naturally, e.g. "Confirm & fill my quote". The button click is the confirmation.
4. Do NOT put quote-update in auto_action. Do NOT silently edit the form.
5. If the visitor states a per-uniform quantity below 12, explain ALF's 12-piece minimum and do not treat that invalid quantity as confirmed.
6. If the visitor is already on Get a Quote, use CURRENT WEBSITE STATE.quote to understand what is already filled and patch only what should change.
7. If they are elsewhere, confirmation can take them to Get a Quote and carry the confirmed details into the form.

Allowed quote patch structure:
{{
  "uniforms": [{{"name":"Polo Shirts & T-Shirts","qty":24}}],
  "company":"...",
  "industry":"Restaurant / Café|Corporate Office|Security|Retail|Events / Promotions|Service / Operations|Other",
  "project":"...",
  "color":"...",
  "deadline":"...",
  "male_count": 0,
  "female_count": 0,
  "sizes": {{"S":0,"M":0,"L":0,"XL":0,"XXL":0,"Other":0}},
  "branding":"Embroidery|Printing|Need ALF recommendation",
  "logo_placement":"...",
  "logo_ready":"Yes — ready to send on WhatsApp|No — need guidance",
  "branding_notes":"...",
  "name":"...",
  "phone":"...",
  "email":"...",
  "area":"...",
  "followup":"WhatsApp|Phone call|Arrange a meeting",
  "contact_time":"...",
  "notes":"..."
}}
Omit every field the visitor did not provide. A uniform may be included with qty 0 only when the visitor selected that uniform but has not provided a valid quantity yet.
If exactly one saved/selected uniform exists and the visitor clearly gives one quantity for that uniform, you may attach that quantity to it. If multiple uniforms exist, never invent how a single total quantity should be divided.
If the visitor says "fill the quote", "put this in the form", or equivalent after giving details across the conversation, gather only the confirmed details from the conversation and return the confirmation action.

Example behavior: visitor says "We need 24 polo shirts in navy for ABC, embroidery on the left chest, contact me on WhatsApp." Reply with a short summary and a quote-update action whose patch contains Polo Shirts & T-Shirts qty 24, company ABC, color Navy, branding Embroidery, logo_placement Left chest, and followup WhatsApp. Do not apply it without the confirmation click.

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
  "actions": [
    {{"label":"...","type":"navigate|add|enquiry-list|whatsapp|prompt","value":"..."}}
    OR
    {{"label":"Confirm & fill my quote","type":"quote-update","patch":{{...allowed quote patch fields...}}}}
  ],
  "auto_action": null OR {{"label":"...","type":"navigate|add|enquiry-list|whatsapp","value":"..."}},
  "context": {{"lastUniform":"","industry":"","quantity":0}}
}}
Keep actions to 0-3 useful choices. Never put quote-update in auto_action. Do not output markdown.
""".strip()

app = FastAPI(title=APP_NAME, version="1.1.0")
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
    quote: dict[str, Any] = Field(default_factory=dict)
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


def _clean_quote_patch(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    patch: dict[str, Any] = {}
    quote_uniforms = set(UNIFORMS + ["Corporate Shirts", "Other"])

    uniforms = []
    if isinstance(raw.get("uniforms"), list):
        for item in raw.get("uniforms", [])[:12]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name not in quote_uniforms:
                continue
            try:
                qty = int(item.get("qty", 0) or 0)
            except Exception:
                qty = 0
            # 0 means selected but quantity still unknown. 1-11 is invalid for ALF MOQ.
            if 0 < qty < 12:
                qty = 0
            qty = max(0, min(qty, 100000))
            uniforms.append({"name": name, "qty": qty})
    if uniforms:
        patch["uniforms"] = uniforms

    def put_text(key: str, limit: int = 500) -> None:
        value = raw.get(key)
        if value is None:
            return
        text = str(value).strip()
        if text:
            patch[key] = text[:limit]

    for key, limit in {
        "company": 180,
        "project": 1000,
        "color": 160,
        "deadline": 180,
        "logo_placement": 220,
        "branding_notes": 1000,
        "name": 160,
        "phone": 100,
        "email": 220,
        "area": 180,
        "contact_time": 180,
        "notes": 1200,
    }.items():
        put_text(key, limit)

    allowed_industries = {
        "Restaurant / Café", "Corporate Office", "Security", "Retail",
        "Events / Promotions", "Service / Operations", "Other"
    }
    industry = str(raw.get("industry", "")).strip()
    if industry in allowed_industries:
        patch["industry"] = industry

    allowed_branding = {"Embroidery", "Printing", "Need ALF recommendation"}
    branding = str(raw.get("branding", "")).strip()
    if branding in allowed_branding:
        patch["branding"] = branding

    allowed_logo_ready = {"Yes — ready to send on WhatsApp", "No — need guidance"}
    logo_ready = str(raw.get("logo_ready", "")).strip()
    if logo_ready in allowed_logo_ready:
        patch["logo_ready"] = logo_ready

    allowed_followup = {"WhatsApp", "Phone call", "Arrange a meeting"}
    followup = str(raw.get("followup", "")).strip()
    if followup in allowed_followup:
        patch["followup"] = followup

    for key in ("male_count", "female_count"):
        if key in raw:
            try:
                num = int(raw.get(key, 0) or 0)
            except Exception:
                continue
            patch[key] = max(0, min(num, 100000))

    if isinstance(raw.get("sizes"), dict):
        sizes: dict[str, int] = {}
        for label in ("S", "M", "L", "XL", "XXL", "Other"):
            if label not in raw["sizes"]:
                continue
            try:
                num = int(raw["sizes"].get(label, 0) or 0)
            except Exception:
                continue
            sizes[label] = max(0, min(num, 100000))
        if sizes:
            patch["sizes"] = sizes

    return patch


def _clean_action(raw: Any, allow_prompt: bool = True) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    atype = str(raw.get("type", "")).strip()
    label = str(raw.get("label", "")).strip()[:80]
    value = str(raw.get("value", "")).strip()[:500]
    allowed = {"navigate", "add", "enquiry-list", "whatsapp", "quote-update"}
    if allow_prompt:
        allowed.add("prompt")
    if atype not in allowed:
        return None

    if atype == "quote-update":
        patch = _clean_quote_patch(raw.get("patch"))
        if not patch:
            return None
        return {
            "label": label or "Confirm & fill my quote",
            "type": "quote-update",
            "value": "",
            "patch": patch,
        }

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
        "quote": payload.quote if isinstance(payload.quote, dict) else {},
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
    if auto_action and auto_action.get("type") == "quote-update":
        auto_action = None
    context = _clean_context(data.get("context"), payload.context)

    return {
        "reply": reply,
        "actions": actions,
        "auto_action": auto_action,
        "context": context,
        "model": MODEL,
    }
