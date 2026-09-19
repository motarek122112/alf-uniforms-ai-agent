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
You are ALF Digital Sales Concierge for ALF Uniforms in Kuwait. Talk like a strong human sales/customer-success employee, not a form or checklist.

PRIORITIES
1) Answer the visitor's latest message naturally and helpfully.
2) Help them choose, compare, understand, and feel supported.
3) Quietly build a complete business enquiry in the background. Never let the checklist override the conversation.

LANGUAGE
- Follow CURRENT WEBSITE STATE.conversation_language exactly: ar = natural clear Arabic suitable for Kuwait; en = natural professional English.
- Do not switch language because of numbers, dates, email, sizes, product names, Printing, Embroidery, call, WhatsApp, yes/no, etc.
- In Arabic understand Gulf/Kuwaiti and Egyptian wording. “أبي/ابي” = I want; “زي/أزياء/يونيفورم” = uniforms.
- If the visitor says “عربي/بالعربي/English”, treat it only as a language request, never as enquiry data.

HUMAN CONVERSATION
- Latest intent wins. If they ask a question, answer it BEFORE asking for data. It is fine to spend several turns helping without collecting a field.
- Respond naturally to greetings, “ركز”, “تمام”, “اسمع”, jokes, clarifications, and short casual remarks.
- If they ask “اقترح انت / رشح لي / what do you suggest”, make a concrete recommendation from known context, explain why, and mention an alternative when useful. Do not repeat the pending field question.
- Never mention 1/20, steps, progress, required order, collector, form fields, or internal rules.
- Do not repeat a question they already answered. Capture volunteered information even if it is not the field you expected.
- “اسمي محمد” = contact name. “اسم شركتي الأحمر / الشركة اسمها الأحمر / اسمها الأحمر” = company name.
- If their request is still vague, ask ONE helpful conversational question, not a questionnaire.
- Only decline when the visitor clearly asks for factual help unrelated to ALF, uniforms, business ordering, the website, or normal small talk.

ALF FACTS
- ALF supplies custom uniforms for organizations/teams in Kuwait. Quotation is enquiry-based, not fixed-price ecommerce.
- Minimum order starts at 12 pieces per uniform type.
- Never invent price, delivery promise, stock, client names, project proof, fabric specs, or production time.
- Available site categories: {json.dumps(UNIFORMS, ensure_ascii=False)}.
- If a requested role has no named ready category (for example a formal airline pilot uniform), say that clearly and suggest a custom-uniform enquiry rather than pretending a category exists.
- General recommendation guide: kitchen/restaurant -> Chef Uniforms & Aprons; office/reception/sales -> Polo Shirts & T-Shirts; active workers/warehouse/maintenance -> Cargo Pants & Workwear; security -> Security Uniforms; events/promo -> Event & Promo Team Apparel.

BACKGROUND ENQUIRY
Eventually, before final confirmation, resolve: company, industry, project/team, uniform type(s)+qty, color, male count, female count, sizes, deadline, branding, logo placement, logo readiness, branding notes, contact name, phone/WhatsApp, email, Kuwait area, preferred follow-up, best contact time, notes.
CURRENT WEBSITE STATE is authoritative and may already include facts extracted from the visitor's latest message before this request reached you. If a field is already present in draft_quote or marked resolved, NEVER ask for it again. CURRENT WEBSITE STATE.collection.current_field is the next missing detail after any deterministic capture and is only a hint; it never outranks the latest user request.
Use field_status: known only when draft_quote contains the value; unknown only when the user truly does not know; none only when not applicable/there is nothing to add. If they refuse a useful detail, briefly explain why it helps and ask for an estimate, but do not pressure them repeatedly.
Do not offer quote-update until every required field is resolved and each chosen uniform has qty >= 12.

ACTIONS
Allowed: navigate, add, enquiry-list, whatsapp, prompt, quote-update. Use 0-3 only when genuinely useful. quote-update is explicit confirmation only, never auto_action.
Routes: {json.dumps(ROUTES, ensure_ascii=False)}; range=/pages/custom-uniforms; branding=/pages/embroidery-printing; bulk=/pages/bulk-orders; real work=/#real-work; quote=/pages/get-a-quote?mode=fresh&intent=general.

OUTPUT
Return ONLY one JSON object with keys: reply, actions, auto_action, context, draft_quote, field_status.
- reply: natural answer in current language.
- context: object with lastUniform, industry, quantity.
- draft_quote: complete merged draft so far, not just changes.
- field_status: complete statuses you can safely support.
- auto_action: null unless a non-destructive navigation/add action is clearly appropriate.
No markdown outside reply text.
""".strip()

app = FastAPI(title=APP_NAME, version="1.8.0")
# Shopify can serve the same uploaded theme from the myshopify domain, a custom
# storefront domain, and preview/editor hosts. CORS is not authentication here;
# the API is already public, while the Groq key remains server-side. Allow HTTPS
# storefront origins so a domain change never silently forces the theme into its
# local fallback mode.
ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", r"https://.*").strip() or r"https://.*"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
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


class CollectionState(BaseModel):
    active: bool = False
    current_field: str = Field(default="", max_length=80)
    required_order: list[str] = Field(default_factory=list, max_length=40)
    resolved: dict[str, str] = Field(default_factory=dict)
    rule: str = Field(default="", max_length=1600)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)
    page: PageState = Field(default_factory=PageState)
    enquiry: list[EnquiryItem] = Field(default_factory=list, max_length=20)
    context: AgentContext = Field(default_factory=AgentContext)
    draft_quote: dict[str, Any] = Field(default_factory=dict)
    quote: dict[str, Any] = Field(default_factory=dict)
    collection: CollectionState = Field(default_factory=CollectionState)
    conversation_language: Literal["ar", "en"] = "en"
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


def _language_only_message(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.fullmatch(
        r"(?:عربي|العربي|بالعربي|باللغة العربية|تكلم عربي|اتكلم عربي|arabic|arabic please|english|english please|انجليزي|إنجليزي|بالانجليزي|بالإنجليزي|تكلم انجليزي|اتكلم انجليزي)",
        t,
        flags=re.I,
    ))


def _extract_person_name(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    patterns = [
        r"^(?:أنا\s+)?اسمي\s+(.+)$",
        r"^(?:أنا|انا)\s+(.+?)\s+(?:وعندي|عندي|ولدي|لدي)\s+(?:شركة|شركه|مؤسسة|موسسة)\b",
        r"^my\s+name\s+is\s+(.+)$",
        r"^i(?:'m|\s+am)\s+([A-Za-z][A-Za-z .'-]{1,80})$",
    ]
    for pattern in patterns:
        m = re.match(pattern, t, flags=re.I)
        if m:
            name = re.sub(r"\s+", " ", m.group(1)).strip(" .,-")
            return name[:160]
    return ""


def _extract_company_name(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    patterns = [
        r"^(?:اسم\s+شركتي|اسم\s+(?:الشركة|الشركه)|(?:الشركة|الشركه)\s+اسمها|شركتي\s+اسمها|اسمها)\s*[:\-]?\s*(.+)$",
        r"^(?:عندي|لدي)\s+(?:شركة|شركه|مؤسسة|موسسة)(?:\s+اسمها)?\s+(.+)$",
        r"^(?:أنا|انا)\s+.+?\s+(?:وعندي|عندي|ولدي|لدي)\s+(?:شركة|شركه|مؤسسة|موسسة)(?:\s+اسمها)?\s+(.+)$",
        r"^(?:my\s+company(?:\s+name)?\s+is|company(?:\s+name)?\s+is|business(?:\s+name)?\s+is)\s+(.+)$",
    ]
    for pattern in patterns:
        m = re.match(pattern, t, flags=re.I)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip(" .,-")
            value = re.sub(r"^هي\s+", "", value, flags=re.I)
            if value:
                return value[:180]
    return ""


def _normalize_industry_text(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    if re.search(r"airport|airline|aviation|air transport|مطار|مطارات|طيران|شركة طيران|طيران مدني|hospital|healthcare|clinic|مستشفى|مستوصف|عيادة|school|university|education|مدرسة|جامعة|تعليم|construction|contracting|مقاولات|إنشاءات|انشاءات|hotel|فندق|other|أخرى|اخرى", t, flags=re.I):
        return "Other"
    if re.search(r"restaurant|cafe|café|food|hospitality|مطعم|كافيه|ضيافة", t, flags=re.I):
        return "Restaurant / Café"
    if re.search(r"security|guard|امن|أمن|حراسة", t, flags=re.I):
        return "Security"
    if re.search(r"retail|shop|store|commerce|trading|تجزئة|متجر|تجارة|تجاره|تجاري|تجارية", t, flags=re.I):
        return "Retail"
    if re.search(r"event|promotion|promo|exhibition|فعاليات|ترويج|معرض", t, flags=re.I):
        return "Events / Promotions"
    if re.search(r"service|operations|warehouse|maintenance|logistics|خدمات|تشغيل|مخزن|صيانة|لوجست", t, flags=re.I):
        return "Service / Operations"
    if re.search(r"corporate|office|company|مكتب|شركة", t, flags=re.I):
        return "Corporate Office"
    return ""


def _extract_project_hint(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    # Explicit use/team wording first.
    patterns = [
        r"(?:ل|لل)\s*(طيارين|الطيارين|عمال|العمال|موظفين|الموظفين|شيفات|الشيفات|امن|أمن|حراس|فريق مبيعات|استقبال)(?:\s|$)",
        r"(?:for|for my|for our)\s+([A-Za-z][A-Za-z \/&-]{2,80})(?:\?|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, t, flags=re.I)
        if m:
            value = re.sub(r"^ال", "", m.group(1).strip(), flags=re.I)
            return value[:160]
    return ""

def _merge_model_draft(base: Any, update: Any, latest_user_text: str, current_field: str = "") -> dict[str, Any]:
    """Merge model extraction with deterministic guards for known failure modes."""
    base_clean = _clean_quote_patch(base)
    if _language_only_message(latest_user_text):
        # A language-control message is never enquiry data.
        return base_clean

    incoming = _clean_quote_patch(update)
    person_name = _extract_person_name(latest_user_text)
    company_name = _extract_company_name(latest_user_text)
    project_hint = _extract_project_hint(latest_user_text)
    if person_name:
        incoming["name"] = person_name
        if "company" not in base_clean and "company" in incoming:
            company = str(incoming.get("company", "")).strip().lower()
            raw = latest_user_text.strip().lower()
            if company in {person_name.lower(), raw} or raw.endswith(company):
                incoming.pop("company", None)
    if company_name:
        explicit_rename = bool(re.search(r"اسم\s+شركتي|اسم\s+(?:الشركة|الشركه)|company\s+name|business\s+name", latest_user_text, flags=re.I))
        if "company" not in base_clean or explicit_rename:
            incoming["company"] = company_name
    if project_hint and "project" not in base_clean and not str(incoming.get("project", "") or "").strip():
        incoming["project"] = project_hint

    if current_field == "industry" and "industry" not in base_clean:
        normalized_industry = _normalize_industry_text(latest_user_text)
        if normalized_industry:
            incoming["industry"] = normalized_industry

    # Phrases such as "أبي زي" / "I need a uniform" are user intent, not a
    # company name. A model extraction mistake here would poison the entire
    # strict collector, so reject that one known bad mapping deterministically.
    if "company" not in base_clean and _looks_like_uniform_intent(latest_user_text):
        incoming.pop("company", None)

    return _merge_quote_patch(base_clean, incoming)


COLLECTION_FIELD_IDS = (
    "company", "industry", "project", "uniforms", "color", "male_count",
    "female_count", "sizes", "deadline", "branding", "logo_placement",
    "logo_ready", "branding_notes", "name", "phone", "email", "area",
    "followup", "contact_time", "notes",
)


def _draft_has_field(draft: dict[str, Any], field_id: str) -> bool:
    if field_id == "uniforms":
        items = draft.get("uniforms") if isinstance(draft.get("uniforms"), list) else []
        return bool(items) and all(str(x.get("name", "")).strip() and int(x.get("qty", 0) or 0) >= 12 for x in items if isinstance(x, dict))
    if field_id == "sizes":
        sizes = draft.get("sizes") if isinstance(draft.get("sizes"), dict) else {}
        return any(int(v or 0) > 0 for v in sizes.values())
    if field_id in ("male_count", "female_count"):
        return field_id in draft and isinstance(draft.get(field_id), int) and draft[field_id] >= 0
    value = draft.get(field_id)
    return value is not None and str(value).strip() != ""


def _clean_field_status(raw: Any, draft: dict[str, Any], fallback: Any = None) -> dict[str, str]:
    result: dict[str, str] = {}
    fallback = fallback if isinstance(fallback, dict) else {}
    for key, value in fallback.items():
        if key in COLLECTION_FIELD_IDS and value in {"known", "unknown", "none"}:
            if value != "known" or _draft_has_field(draft, key):
                result[key] = value
    raw = raw if isinstance(raw, dict) else {}
    for key, value in raw.items():
        if key not in COLLECTION_FIELD_IDS or value not in {"known", "unknown", "none"}:
            continue
        if value == "known" and not _draft_has_field(draft, key):
            continue
        result[key] = value
    # Any valid concrete draft value is known, even if the model omitted the status.
    for key in COLLECTION_FIELD_IDS:
        if _draft_has_field(draft, key):
            result[key] = "known"
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




def _looks_like_uniform_intent(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not t:
        return False
    if t in {"زي", "ازياء", "أزياء", "يونيفورم", "يونيفورمات", "uniform", "uniforms"}:
        return True
    return bool(re.search(
        r"(?:^|\s)(?:ابي|أبي|ابغى|أبغى|اريد|أريد|عايز|محتاج|احتاج|أحتاج|need|want)\s+(?:لي\s+)?(?:زي|ازياء|أزياء|يونيفورم|يونيفورمات|uniform|uniforms|ملابس\s+عمل)(?:\s|$)",
        t,
        flags=re.I,
    ))


def _is_greeting(text: str) -> bool:
    t = re.sub(r"[!؟?.,،]+$", "", (text or "").strip().lower())
    if re.fullmatch(r"هلا+", t):
        return True
    return t in {"هلا", "هلا والله", "مرحبا", "مرحبًا", "السلام عليكم", "اهلين", "أهلين", "hi", "hello", "hey"}


def _asks_uniform_types(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.search(r"(?:الأنواع|الانواع|انواع|أنواع|المتاح|متوفر|متاحة|available|categories|types)", t, flags=re.I))


def _asks_for_recommendation(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return bool(re.search(r"(?:اقترح|رشح|اختار|اختيار|انسب|أنسب|تنصح|مناسب|suggest|recommend|what do you suggest|which.*best|what.*best)", t, flags=re.I))


def _is_casual_nudge(text: str) -> bool:
    t = re.sub(r"[!؟?.,،]+$", "", (text or "").strip().lower())
    return t in {"ركز", "اسمع", "اسمعني", "تمام", "اوكي", "أوكي", "طيب", "معاي", "معايا", "focus", "listen", "اسألني انت", "اسالني انت", "اسأل انت", "اسال انت", "ابدأ اسأل", "ابدأ اسال", "كمل انت", "كمّل انت", "ask me", "you ask", "lead me"}


def _local_recommendation(payload: "ChatRequest") -> tuple[str, list[dict[str, Any]]]:
    latest = payload.messages[-1].content if payload.messages else ""
    draft = payload.draft_quote if isinstance(payload.draft_quote, dict) else {}
    project = str(draft.get("project", "") or "")
    industry = str(draft.get("industry", "") or "")
    combined = f"{latest} {project} {industry}".lower()
    ar = payload.conversation_language == "ar"

    if re.search(r"(طيار|طيارين|pilot|pilots|airline|aviation)", combined, flags=re.I):
        if ar:
            return (
                "لو تقصد طيارين رسميين، ما راح أقول لك إن في فئة جاهزة باسم طيارين وهي مو موجودة في الكتالوج الحالي. الأفضل نمشيها كـ Custom Uniform ونحدد الستايل الرسمي والهوية. وإذا المقصود طاقم أرضي بالمطار فممكن نختار Workwear أو Polo حسب طبيعة دورهم. تقصد الطيارين أنفسهم ولا الطاقم الأرضي؟",
                [{"label":"عرض الزي المخصص","type":"navigate","value":"/pages/custom-uniforms"}],
            )
        return (
            "If you mean formal pilots, I would treat it as a custom-uniform request because the current catalog does not list a dedicated pilot category. If you mean airport ground crew, Workwear or Polos may fit depending on the role. Do you mean the pilots themselves or ground crew?",
            [{"label":"View custom uniforms","type":"navigate","value":"/pages/custom-uniforms"}],
        )

    if re.search(r"(عمال|عامل|مخزن|تحميل|تشغيل|صيانة|لوجست|warehouse|worker|workers|labou?r|operations|maintenance|logistics)", combined, flags=re.I):
        if ar:
            return (
                "بما إن الزي للعمال، أنا أميل أبدأ معك بـ Cargo Pants & Workwear لأنه عملي أكثر للحركة والشغل اليومي. إذا العمال أغلب وقتهم قدام العملاء أو داخل معرض/محل، البولو والتي‑شيرت ممكن يطلع أرتب. لو شغلهم تشغيل وحركة، فالـWorkwear هو اختياري الأول. طبيعة شغلهم أقرب لأي واحد فيهم؟",
                [
                    {"label":"عرض الـWorkwear","type":"navigate","value":"/pages/workwear"},
                    {"label":"عرض البولو والتي‑شيرت","type":"navigate","value":"/pages/polo-t-shirts"},
                ],
            )
        return (
            "For a general workers team, I’d start with Cargo Pants & Workwear because it suits active day-to-day work better. If they are mostly customer-facing retail staff, polos and T-shirts can look cleaner. If the role is operational, Workwear would be my first pick. Which setting is closer?",
            [
                {"label":"View Workwear","type":"navigate","value":"/pages/workwear"},
                {"label":"View Polo & T-Shirts","type":"navigate","value":"/pages/polo-t-shirts"},
            ],
        )
    if re.search(r"(مطعم|كافيه|مطبخ|شيف|restaurant|cafe|kitchen|chef|hospitality)", combined, flags=re.I):
        return (("لو الفريق مطعم أو مطبخ، أبدأ بزي الشيف والمرايل، ولو عندك فريق استقبال أو خدمة عملاء أضيف لهم بولو موحد." if ar else "For a restaurant or kitchen team, I’d start with Chef Uniforms & Aprons, with polos as a clean option for front-of-house staff."), [{"label":"عرض زي الشيف","type":"navigate","value":"/pages/chef-uniforms"}] if ar else [{"label":"View Chef Uniforms","type":"navigate","value":"/pages/chef-uniforms"}])
    if re.search(r"(أمن|حراسة|security|guard)", combined, flags=re.I):
        return (("لو الفريق أمن أو حراسة، الزي الأمني هو الاختيار الطبيعي كبداية، وبعدها نضبط اللون والبراندنج حسب الجهة." if ar else "For a security team, Security Uniforms are the natural starting point, then we can tailor color and branding to the organization."), [{"label":"عرض الزي الأمني","type":"navigate","value":"/pages/security-uniforms"}] if ar else [{"label":"View Security Uniforms","type":"navigate","value":"/pages/security-uniforms"}])
    if re.search(r"(فعالية|فعاليات|معرض|ترويج|event|promo|promotion|exhibition)", combined, flags=re.I):
        return (("للفعاليات والترويج، Event & Promo Team Apparel هو الأقرب، والبولو بديل ممتاز لو تبي شكل أبسط وأكثر رسميّة." if ar else "For events and promotional teams, Event & Promo Team Apparel is the closest match; polos are a good alternative for a cleaner corporate look."), [{"label":"عرض زي الفعاليات","type":"navigate","value":"/pages/event-uniforms"}] if ar else [{"label":"View Event Apparel","type":"navigate","value":"/pages/event-uniforms"}])
    if re.search(r"(شركة|مكتب|استقبال|مبيعات|corporate|office|reception|sales)", combined, flags=re.I):
        return (("لفريق مكتب أو استقبال أو مبيعات، البولو والتي‑شيرت غالبًا أفضل بداية: شكله مرتب وسهل نطابقه مع ألوان وهوية الشركة." if ar else "For an office, reception or sales team, polos and T-shirts are usually the best starting point: clean, versatile and easy to match to the brand."), [{"label":"عرض البولو والتي‑شيرت","type":"navigate","value":"/pages/polo-t-shirts"}] if ar else [{"label":"View Polo & T-Shirts","type":"navigate","value":"/pages/polo-t-shirts"}])

    return (("أكيد أرشح لك، بس ما أبي أعطيك اختيار عشوائي. طبيعة شغل الفريق أكثر حركة وتشغيل، تعامل مباشر مع العملاء، مطبخ، أمن، ولا فعاليات؟ على أساسها أعطيك اختياري الأول وبديله." if ar else "Absolutely — I can recommend it, but I don’t want to throw out a random option. Is the team mainly active/operational, customer-facing, kitchen, security, or events? I’ll give you a first choice and a backup."), [])


def _natural_local_reply(payload: "ChatRequest", reason: str = "") -> dict[str, Any]:
    """Useful conversational fail-safe for temporary provider/rate-limit failures."""
    latest = payload.messages[-1].content if payload.messages else ""
    ar = payload.conversation_language == "ar"
    draft = _merge_model_draft(payload.draft_quote, {}, latest, payload.collection.current_field)
    field_status = _clean_field_status({}, draft, payload.collection.resolved)
    actions: list[dict[str, Any]] = []
    t = (latest or "").strip().lower()
    company_name = _extract_company_name(latest)
    project_hint = _extract_project_hint(latest)
    project = str(draft.get("project", "") or project_hint or "")

    if _is_greeting(latest):
        reply = "هلا 👋 حياك الله. قل لي شنو محتاج بالضبط وأنا أرتبه معك." if ar else "Hi 👋 Welcome. Tell me what you need and I’ll work it through with you."
    elif _is_casual_nudge(latest):
        if re.search(r"اسألني انت|اسالني انت|اسأل انت|اسال انت|ابدأ اسأل|ابدأ اسال|كمل انت|كمّل انت|ask me|you ask|lead me", latest, flags=re.I):
            current = payload.collection.current_field or "company"
            prompts_ar = {
                "company": "تمام، نبدأ من الأساس: شنو اسم الشركة أو الجهة؟",
                "industry": "تمام، ومجال نشاط الشركة شنو تقريبًا؟",
                "project": "حلو، والزي هذا لأي فريق أو استخدام بالتحديد؟",
                "uniforms": "خلنا نختار الأنسب. شنو نوع الزي اللي في بالك، وإذا محتار أقدر أرشح لك؟",
            }
            prompts_en = {
                "company": "Sure — let’s start with the basics. What is the company or organization name?",
                "industry": "Sure — what industry is the company in?",
                "project": "Great — which team or use is the uniform for?",
                "uniforms": "Let’s narrow it down. What uniform type do you have in mind, or would you like me to recommend one?",
            }
            reply = (prompts_ar if ar else prompts_en).get(current, "تمام، خلنا نكمل من أهم تفصيلة ناقصة عندي." if ar else "Sure — let’s continue with the most useful missing detail.")
        else:
            reply = "معك ومركز 👌 كمل، شنو في بالك؟" if ar else "I’m with you 👍 Go ahead — what’s on your mind?"
    elif re.fullmatch(r"(?:شركتي|الشركة|الشركه|اسم الشركة|اسم الشركه|my company|company)", t, flags=re.I):
        reply = "تمام، شنو اسم الشركة؟" if ar else "Sure — what’s the company name?"
    elif company_name:
        reply = (f"تمام، شركة {company_name} 👍 وش نوع الفريق أو الاستخدام اللي تبون الزي له؟" if ar else f"Got it — {company_name}. What team or use is the uniform for?")
    elif _asks_for_recommendation(latest):
        reply, actions = _local_recommendation(payload)
    elif _asks_uniform_types(latest) or re.search(r"(?:ازياء|أزياء|يونيفورمات|ملابس\\s+عمل).*(?:عندكم|متوفر|متاحة|available|types)|(?:ايه|إيه|وش|شنو|ما).*(?:الازياء|الأزياء|الأنواع|انواع)", t, flags=re.I):
        if re.search(r"طيار|pilot|airline|aviation", f"{latest} {project}", flags=re.I):
            reply = ("إذا تقصد زي طيارين رسمي، ما عندنا داخل الموقع فئة جاهزة باسم «زي طيارين» أقول لك إنها موجودة وهي مو موجودة. نقدر نمشيه كـ Custom Uniform ونحدد الشكل والهوية المطلوبة، أما لو المقصود طاقم أرضي أو فريق تشغيلي فأقدر أرشح من الأنواع الموجودة حسب طبيعة شغلهم. تحب الستايل يكون رسمي جدًا ولا عملي أكثر؟" if ar else "If you mean a formal airline pilot uniform, the site does not list a ready category specifically called pilot uniforms, so I won’t pretend it does. We can treat it as a custom-uniform enquiry and define the look and branding. If you mean ground or operational crew, I can recommend from the existing categories. Do you want a very formal look or something more practical?")
            actions = [{"label":"الزي المخصص" if ar else "Custom uniforms","type":"navigate","value":"/pages/custom-uniforms"}]
        else:
            reply = ("أكيد. الموجود عند ALF يشمل بولو وتي‑شيرت، زي شيف ومرايل، قبعات وإكسسوارات شيف، كارجو وملابس عمل، زي أمني، وملابس فرق الفعاليات والترويج. قل لي الفريق بيشتغل شنو وأنا أضيقها لك بدل ما أخليك تختار من قائمة طويلة." if ar else "ALF offers polos and T-shirts, chef uniforms and aprons, chef caps/accessories, cargo/workwear, security uniforms, and event/promo apparel. Tell me what the team actually does and I’ll narrow it down for you.")
            actions = [{"label":"عرض كل الأنواع" if ar else "Browse all uniforms","type":"navigate","value":"/pages/custom-uniforms"}]
    elif _looks_like_uniform_intent(latest) or re.search(r"(?:ازياء|أزياء|يونيفورمات|ملابس\\s+عمل)", t, flags=re.I):
        if re.search(r"طيار|pilot|airline|aviation", f"{latest} {project}", flags=re.I):
            reply = ("أكيد. لو الزي لطيارين رسميين فالأفضل نتعامل معه كطلب Custom Uniform لأن الموقع ما يعرض فئة طيارين جاهزة بالاسم. نقدر نحدد معك الستايل الرسمي، الألوان، الشعار والكمية وبعدها فريق ALF يراجع الطلب. تقصد طيارين فعليًا ولا طاقم أرضي بالمطار؟" if ar else "Absolutely. For formal pilots, I’d treat this as a custom-uniform requirement because the site does not list a dedicated pilot category. We can define the formal look, colors, logo and quantity for ALF to review. Do you mean pilots themselves or airport ground crew?")
            actions = [{"label":"عرض الزي المخصص" if ar else "View custom uniforms","type":"navigate","value":"/pages/custom-uniforms"}]
        elif project:
            reply, actions = _local_recommendation(payload)
        elif str(draft.get("company", "") or "").strip():
            reply = "أكيد. الزي لأي فريق داخل الشركة بالضبط؟ عمال، مبيعات، استقبال، مطبخ، أمن، فعاليات…؟" if ar else "Absolutely. Which team inside the company is this for — workers, sales, reception, kitchen, security, events, or something else?"
        else:
            reply = "أكيد نساعدك. الزي لمين بالضبط أو لطبيعة شغل شنو؟ على أساسها أرشح لك بدل ما أعطيك اختيار عشوائي." if ar else "Absolutely. Who is the uniform for, or what kind of work do they do? I’ll recommend from there rather than guessing."
    elif re.search(r"(price|pricing|cost|kwd|دينار|سعر|تكلفة)", t):
        reply = "السعر يتحدد حسب النوع والكمية والمقاسات والبراندنج، وفريق ALF يأكد عرض السعر بعد مراجعة الطلب. إذا تحب نقدر نضبط المواصفات أول بحيث يوصلهم Brief واضح." if ar else "Pricing depends on the uniform type, quantity, sizes and branding, and ALF confirms the quotation after reviewing the requirement. I can help you shape a clear brief first."
    elif re.search(r"(minimum|moq|اقل كمية|أقل كمية)", t):
        reply = "الحد الأدنى يبدأ من 12 قطعة لكل نوع يونيفورم." if ar else "The minimum starts from 12 pieces per uniform type."
    elif re.search(r"(logo|brand|branding|embroider|embroidery|print|printing|طباعة|تطريز|لوجو|شعار)", t):
        reply = "متوفر تطريز وطباعة. الأنسب يعتمد على نوع الزي وشكل الشعار والاستخدام؛ إذا قلت لي القطعة اللي اخترتها أرشح لك بين الاثنين." if ar else "ALF offers embroidery and printing. The better option depends on the garment, logo and use; tell me the item and I’ll help you choose."
        actions = [{"label":"خيارات البراندنج" if ar else "Branding options","type":"navigate","value":"/pages/embroidery-printing"}]
    else:
        current = (payload.collection.current_field or "").strip()
        stripped = re.sub(r"[!؟?.,،]+$", "", latest.strip().lower())
        if current == "company" and stripped in {"شركتي", "الشركة", "اسم الشركة", "my company", "company"}:
            reply = "تمام، شنو اسم الشركة؟" if ar else "Sure — what’s the company name?"
        elif current:
            # Let the storefront parse an answer and ask the genuinely next field.
            reply = ""
        else:
            reply = ("أكيد، خذ راحتك. قل لي اللي تحتاجه عن الزي أو الاختيارات وأنا أرد عليك مباشرة؛ وإذا احتجنا تفاصيل للطلب نجمعها بهدوء أثناء الكلام." if ar else "Of course — talk to me normally. Ask me anything about the uniforms or options and I’ll answer directly; if we need order details, we can collect them naturally as we go.")

    return {
        "reply": reply,
        "actions": actions,
        "auto_action": None,
        "context": _clean_context({}, payload.context),
        "draft_quote": draft,
        "field_status": field_status,
        "conversation_language": payload.conversation_language,
        "model": MODEL,
        "provider": "local-failsafe",
        "fallback_reason": reason[:120],
    }

@app.get("/")
def root() -> dict[str, str]:
    return {"service": APP_NAME, "status": "online", "model": MODEL}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": "1.8.0", "architecture": "state-first-anti-repeat", "model": MODEL, "ai_configured": bool(GROQ_API_KEY)}


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    _rate_limit(request)
    if not GROQ_API_KEY:
        return _natural_local_reply(payload, reason="missing_api_key")

    enquiry_names = [item.name for item in payload.enquiry if item.name in UNIFORMS]
    runtime_context = {
        "page": {"path": payload.page.path, "label": payload.page.label},
        "saved_uniforms": enquiry_names,
        "context": payload.context.model_dump(),
        "draft_quote": _clean_quote_patch(payload.draft_quote),
        "collection": {
            "active": payload.collection.active,
            "current_field": payload.collection.current_field,
            "resolved": payload.collection.resolved,
        },
        "conversation_language": payload.conversation_language,
    }

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "CURRENT WEBSITE STATE:\n" + json.dumps(runtime_context, ensure_ascii=False),
        },
    ]
    messages.extend({"role": m.role, "content": m.content} for m in payload.messages[-8:])

    try:
        # Keep each turn compact. The Groq developer tier can have a low TPM
        # allowance, so one concise request is safer than retrying the same turn
        # and accidentally doubling token usage on a 429.
        client = Groq(api_key=GROQ_API_KEY, timeout=18.0, max_retries=0)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.55,
            max_completion_tokens=600,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or ""
        data = _safe_json(content)
    except Exception as exc:
        # Keep the storefront useful during a temporary provider/rate-limit
        # failure. The next turn will try Groq again; the local reply is not a
        # dead-end canned message.
        reason = exc.__class__.__name__
        status = getattr(exc, "status_code", None)
        if status:
            reason = f"{reason}:{status}"
        return _natural_local_reply(payload, reason=reason)

    reply = str(data.get("reply", "")).strip()[:3000]
    if not reply:
        reply = (
            "أقدر أساعدك تختار اليونيفورم المناسب ونكمّل تفاصيل الطلب بشكل واضح."
            if payload.conversation_language == "ar"
            else "I can help you choose the right ALF uniform and continue the enquiry naturally."
        )

    latest_user_text = payload.messages[-1].content if payload.messages else ""
    draft_quote = _merge_model_draft(payload.draft_quote, data.get("draft_quote"), latest_user_text, payload.collection.current_field)
    field_status = _clean_field_status(
        data.get("field_status"),
        draft_quote,
        payload.collection.resolved,
    )
    ready_for_confirmation = all(field_id in field_status for field_id in COLLECTION_FIELD_IDS)

    actions = []
    for raw in (data.get("actions") or [])[:3]:
        cleaned = _clean_action(raw, allow_prompt=True)
        if cleaned:
            if cleaned.get("type") == "quote-update":
                if not ready_for_confirmation:
                    continue
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
        "field_status": field_status,
        "conversation_language": payload.conversation_language,
        "model": MODEL,
        "provider": "groq",
    }
