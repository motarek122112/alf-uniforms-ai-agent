import json
import os
import re
import time
import traceback
from collections import defaultdict, deque
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

APP_NAME = "ALF Uniforms AI Agent"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "groq/compound-mini")
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
You are ALF Digital Sales Concierge for ALF Uniforms in Kuwait. Behave like a capable human sales/customer-success employee, not a form or scripted bot.

VOICE
- Warm, practical, concise, natural. Answer what the visitor actually says first, then move the enquiry forward smoothly.
- Small talk, corrections and short side questions are fine. Only redirect if the topic is clearly unrelated to ALF, uniforms, the website or the enquiry.
- Never expose field numbers, progress counts, checklists, internal rules, or phrases like "I recorded X" after every turn.
- Never re-ask information already present in draft_quote, collection state or recent conversation.

LANGUAGE
- CURRENT WEBSITE STATE.conversation_language is the active language. Keep it until the visitor clearly switches with a real phrase/sentence or explicitly asks for Arabic/English.
- Numbers, phone/email, sizes, product names, "Printing", "call", "WhatsApp", "M", "L", etc. do NOT switch language.
- Arabic must be tidy, natural, neutral Gulf-friendly Arabic suitable for Kuwait; not heavy Egyptian slang and not stiff formal Arabic.
- "عربي/بالعربي" and "English please" are language requests, never enquiry data.

FACTS
- ALF provides custom uniforms for organizations and teams in Kuwait. The site is quotation/enquiry based, not fixed-price ecommerce.
- Minimum order starts from 12 pieces per uniform type. Never invent quantity, price, stock, delivery promise, client/project proof or production time.
- Human help is available by WhatsApp.
- Real ALF Work means only the real work shown by ALF on the site.

UNIFORM CATEGORIES: {json.dumps(UNIFORMS, ensure_ascii=False)}
CATEGORY ROUTES: {json.dumps(ROUTES, ensure_ascii=False)}
Useful routes: range=/pages/custom-uniforms, branding=/pages/embroidery-printing, bulk=/pages/bulk-orders, ready-stock=/pages/ready-stock, how-it-works=/pages/how-it-works, real-work=/#real-work, fresh-quote=/pages/get-a-quote?mode=fresh&intent=general.

ENQUIRY BEHAVIOUR
Collect the requirement inside chat before Get a Quote unless the visitor explicitly asks to open the form. The internal coverage order is:
company, industry, project/team/use, uniforms+qty per type, color, male_count, female_count, sizes, deadline, branding, logo_placement, logo_ready, branding_notes, name, phone, email, area, followup, contact_time, notes.
This order is internal only. Dialogue must feel human. Capture later-field information whenever it appears and never ask for it again. Ask one concise question or a small related group at a time.

SEMANTIC EXTRACTION
- Understand meaning, not the current question slot. "اسمي محمد" => contact name, not company. "عربي" => language request, not branding notes.
- If visitor names a uniform then later gives a bare number and context clearly indicates quantity, combine them.
- If multiple uniform types share one total, ask for the split; never invent it.
- If qty is 1-11, explain 12-piece minimum and ask whether to adjust.
- If visitor asks what is available, answer first, then continue naturally.
- If information is genuinely unknown, accept that and continue. If they are merely reluctant, briefly explain why it helps ALF and ask once more for an estimate; never pressure.

CONFIRMATION
Only offer quote-update after every internal field has either a real value, explicit unknown, or genuine N/A/none. Before confirmation, show a clean summary in the active language for review. quote-update is a user-click confirmation only, never auto_action.

DRAFT MEMORY
Return draft_quote as the COMPLETE merged structured draft on every turn. Preserve prior valid details unless corrected. Never guess. Unknown values should be omitted; explicit unknown/N/A goes in collection_updates.
Allowed structure:
{{"uniforms":[{{"name":"Polo Shirts & T-Shirts","qty":24}}],"company":"...","industry":"Restaurant / Café|Corporate Office|Security|Retail|Events / Promotions|Service / Operations|Other","project":"...","color":"...","deadline":"...","male_count":0,"female_count":0,"sizes":{{"S":0,"M":0,"L":0,"XL":0,"XXL":0,"Other":0}},"branding":"Embroidery|Printing|Need ALF recommendation","logo_placement":"...","logo_ready":"Yes — ready to send on WhatsApp|No — need guidance","branding_notes":"...","name":"...","phone":"...","email":"...","area":"...","followup":"WhatsApp|Phone call|Arrange a meeting","contact_time":"...","notes":"..."}}
qty 0 means selected but quantity unknown; 1-11 is not a valid confirmed quantity.

ACTIONS
Allowed: navigate, add, enquiry-list, whatsapp, prompt, quote-update. Use 0-3 only when genuinely useful; routine collection usually has no buttons. For navigate, use only ALF internal routes and add a short same-language follow_up that makes sense after arrival. quote-update must carry the complete confirmed draft.

OUTPUT ONLY one valid JSON object:
{{"reply":"...","actions":[],"auto_action":null,"context":{{"lastUniform":"","industry":"","quantity":0}},"draft_quote":{{}},"collection_updates":{{}}}}
collection_updates may contain only explicit "unknown" or "none" statuses for these fields: company, industry, project, uniforms, color, male_count, female_count, sizes, deadline, branding, logo_placement, logo_ready, branding_notes, name, phone, email, area, followup, contact_time, notes.
No markdown outside JSON.
""".strip()

app = FastAPI(title=APP_NAME, version="1.4.2")
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
    draft_quote: dict[str, Any] = Field(default_factory=dict)
    quote: dict[str, Any] = Field(default_factory=dict)
    collection: dict[str, Any] = Field(default_factory=dict)
    conversation_language: str = Field(default="en", max_length=10)
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


def _merge_quote_patch(base: Any, update: Any) -> dict[str, Any]:
    merged = _clean_quote_patch(base)
    incoming = _clean_quote_patch(update)
    if not incoming:
        return merged
    result = dict(merged)
    for key, value in incoming.items():
        if key == "sizes" and isinstance(value, dict):
            current = dict(result.get("sizes") or {})
            current.update(value)
            result["sizes"] = current
        else:
            result[key] = value
    return result


def _clean_action(raw: Any, allow_prompt: bool = True) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    atype = str(raw.get("type", "")).strip()
    label = str(raw.get("label", "")).strip()[:80]
    value = str(raw.get("value", "")).strip()[:500]
    follow_up = str(raw.get("follow_up", "") or "").strip()[:420]
    allowed = {"navigate", "add", "enquiry-list", "whatsapp", "quote-update"}
    if allow_prompt:
        allowed.add("prompt")
    if atype not in allowed:
        return None

    if atype == "quote-update":
        patch = _clean_quote_patch(raw.get("patch"))
        if not patch:
            return None
        action = {
            "label": label or "Confirm & prepare my enquiry",
            "type": "quote-update",
            "value": "",
            "patch": patch,
        }
        if follow_up:
            action["follow_up"] = follow_up
        return action

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
    action = {"label": label, "type": atype, "value": value}
    if follow_up and atype == "navigate":
        action["follow_up"] = follow_up
    return action


def _clean_collection_updates(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    allowed_fields = {
        "company", "industry", "project", "uniforms", "color", "male_count",
        "female_count", "sizes", "deadline", "branding", "logo_placement",
        "logo_ready", "branding_notes", "name", "phone", "email", "area",
        "followup", "contact_time", "notes",
    }
    out: dict[str, str] = {}
    for key, value in raw.items():
        key = str(key).strip()
        status = str(value).strip().lower()
        if key in allowed_fields and status in {"unknown", "none"}:
            out[key] = status
    return out


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
    return {"service": APP_NAME, "status": "online", "model": MODEL, "fallback_model": FALLBACK_MODEL}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL, "fallback_model": FALLBACK_MODEL}


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
        "draft_quote": _clean_quote_patch(payload.draft_quote),
        "quote": payload.quote if isinstance(payload.quote, dict) else {},
        "collection": payload.collection if isinstance(payload.collection, dict) else {},
        "conversation_language": "ar" if str(payload.conversation_language).lower().startswith("ar") else "en",
        "site_locale": payload.locale,
    }

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "CURRENT WEBSITE STATE:\n" + json.dumps(runtime_context, ensure_ascii=False),
        },
    ]
    messages.extend({"role": m.role, "content": m.content} for m in payload.messages[-10:])

    try:
        client = Groq(api_key=GROQ_API_KEY, timeout=30.0, max_retries=0)

        def provider_call(model: str):
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.45,
                "max_completion_tokens": 750,
            }
            if model.startswith("openai/gpt-oss"):
                kwargs["reasoning_effort"] = "low"
                kwargs["include_reasoning"] = False
            return client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )

        try:
            completion = provider_call(MODEL)
        except Exception as first_exc:
            first_message = str(first_exc)
            status = getattr(first_exc, "status_code", None)
            print(
                f"[Groq primary failed] model={MODEL} status={status} {type(first_exc).__name__}: {first_message}",
                flush=True,
            )
            lower = first_message.lower()
            is_400_format = status == 400 and any(x in lower for x in ("response_format", "json mode", "json_object", "reasoning"))
            is_capacity = status in {429, 500, 502, 503, 504} or any(x in lower for x in ("rate limit", "too many requests", "capacity", "overloaded", "timeout"))

            if is_400_format:
                # Retry the same model once without JSON mode; prompt still demands JSON.
                kwargs = {
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0.45,
                    "max_completion_tokens": 750,
                }
                if MODEL.startswith("openai/gpt-oss"):
                    kwargs["reasoning_effort"] = "low"
                    kwargs["include_reasoning"] = False
                completion = client.chat.completions.create(**kwargs)
            elif is_capacity and FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
                print(f"[Groq fallback] switching to {FALLBACK_MODEL}", flush=True)
                completion = provider_call(FALLBACK_MODEL)
            else:
                raise

        content = completion.choices[0].message.content or ""
        data = _safe_json(content)
    except HTTPException:
        raise
    except Exception as exc:
        # Print the exact cause to Render logs for diagnosis. Never expose secrets
        # or provider internals to the public storefront response.
        print(f"[Groq final failure] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=502, detail="The AI service is temporarily unavailable.") from exc

    reply = str(data.get("reply", "")).strip()[:3000]
    if not reply:
        if str(payload.conversation_language).lower().startswith("ar"):
            reply = "أكيد، أقدر أساعدك تختار اليونيفورم المناسب ونكمّل طلبك خطوة بخطوة بطريقة بسيطة."
        else:
            reply = "I can help you choose the right ALF uniform and build the enquiry with you naturally."

    draft_quote = _merge_quote_patch(payload.draft_quote, data.get("draft_quote"))
    collection_updates = _clean_collection_updates(data.get("collection_updates"))

    actions = []
    for raw in (data.get("actions") or [])[:3]:
        cleaned = _clean_action(raw, allow_prompt=True)
        if cleaned:
            if cleaned.get("type") == "quote-update":
                # Confirmation always applies the complete structured requirement collected so far.
                cleaned["patch"] = _merge_quote_patch(draft_quote, cleaned.get("patch"))
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
        "draft_quote": draft_quote,
        "collection_updates": collection_updates,
        "model": MODEL,
        "fallback_model": FALLBACK_MODEL,
    }
