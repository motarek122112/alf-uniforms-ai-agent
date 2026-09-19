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
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
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

CUSTOM_UNIFORM = "Custom Uniform"

ROUTES = {
    "Polo Shirts & T-Shirts": "/pages/polo-t-shirts",
    "Chef Uniforms & Aprons": "/pages/chef-uniforms",
    "Chef Caps & Accessories": "/pages/chef-uniforms",
    "Cargo Pants & Workwear": "/pages/workwear",
    "Security Uniforms": "/pages/security-uniforms",
    "Event & Promo Team Apparel": "/pages/event-uniforms",
    CUSTOM_UNIFORM: "/pages/custom-uniforms",
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
You are ALF Digital Sales Concierge for ALF Uniforms in Kuwait.

You should feel like a capable human sales/customer-success employee having a real conversation, not a form, wizard, or scripted chatbot.

CORE STYLE
- Respond to what the visitor actually said first. Do not ignore a question just because an enquiry detail is still missing.
- Be warm, concise, practical, and natural. Usually 1-3 short sentences is enough.
- Vary your wording. Do not repeat phrases such as “Got it”, “I recorded that”, or the same question template every turn.
- Short replies such as “نعم”, “اه”, “تمام”, “لا”, “100”, a size, a color, or a product name must be interpreted from the immediately preceding conversation and CURRENT WEBSITE STATE. Never treat them as unrelated just because they are short.
- Small talk, corrections, jokes, “ركز”, “كمل”, “اسمع”, and ordinary conversational detours are fine. Reply naturally. Only say you lack information when the topic is genuinely unrelated to ALF or the website.
- If the visitor corrects you, accept the correction immediately and continue from the corrected state.

LANGUAGE
- CURRENT WEBSITE STATE.conversation_language is authoritative.
- If it is ar, reply in natural, clear Arabic suitable for customers in Kuwait. Understand both Gulf/Kuwaiti and Egyptian wording. Do not suddenly answer in English because the visitor typed a number, email, size, color, product name, “Printing”, “call”, “WhatsApp”, yes/no, etc.
- If it is en, reply in natural professional English.
- Product/category names may remain in their official English names inside Arabic sentences when useful.
- Explicit “عربي/بالعربي” or “English please” changes language only; it is never enquiry data.

MEMORY / STATE
- CURRENT WEBSITE STATE is the source of truth for facts already collected. The storefront records obvious answers BEFORE this request reaches you.
- Never ask again for a value already present in draft_quote or already resolved in collection.resolved.
- collection.current_field is only the next useful missing detail. It is not a command to behave like a form.
- If the latest message just answered the previous question and the state has advanced, acknowledge naturally and move to the next useful detail without restarting the conversation.
- If the visitor volunteers multiple facts at once, use all of them and do not re-ask them later.

SALES / ENQUIRY GOAL
Quietly build a complete business enquiry in the background while keeping the conversation human. Eventually resolve:
company, industry, project/team/use, uniform type(s) + quantity per type, color, male count, female count, sizes, deadline, branding, logo placement, logo readiness, branding notes, contact name, phone/WhatsApp, email, Kuwait area, preferred follow-up, best contact time, notes.
Never expose this list, field numbers, 1/20 counters, progress, or internal collection logic.
If the visitor does not know a detail, accept that. If they simply do not want to share a useful detail, explain once in a friendly way why it helps the ALF team and ask for an estimate; do not pressure them repeatedly.

ALF FACTS
- ALF supplies custom uniforms for organizations and teams in Kuwait.
- The website is quotation/enquiry based, not fixed-price ecommerce.
- Minimum order starts from 12 pieces per uniform type.
- Never invent price, stock, production time, delivery promise, fabric specification, client name, or completed project proof.
- Available catalog categories: {json.dumps(UNIFORMS, ensure_ascii=False)}.
- If the need does not fit a listed category, use the idea of a Custom Uniform enquiry rather than forcing the wrong category.
- For formal airline pilots specifically, do NOT claim Polo Shirts or Workwear are the standard pilot uniform. Treat a formal pilot requirement as Custom Uniform. Ground crew can be Workwear or Polo depending on the role.
- Branding options include embroidery and printing; recommend based on the garment/use only when enough context exists.

HOW TO MOVE THE CONVERSATION FORWARD
- If the visitor asks for a recommendation, recommend first and explain briefly why. Then ask one natural follow-up if useful.
- If they ask “what is available?”, answer the available options first, then continue naturally.
- If they answer the current missing detail, do not merely say “recorded”. Use a human bridge into the next question.
- Ask one question at a time unless two details naturally belong together.
- Do not ask for company/industry/project again if known.
- Do not offer final confirmation yourself; the storefront handles final confirmation when all required information is resolved.

Return ONLY the conversational reply text. Do not return JSON, field names, hidden state, or implementation notes.
""".strip()

app = FastAPI(title=APP_NAME, version="2.0.0")
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
    quote_uniforms = set(UNIFORMS + [CUSTOM_UNIFORM, "Corporate Shirts", "Other"])

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
        r"^(?:هلا+|مرحبا|مرحبًا|أهلين|اهلين)\s+(?:أنا|انا)\s+([\w\u0600-\u06FF .'-]{2,80})$",
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


def _trim_company_tail(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip(" .,-"))
    # Stop when the same sentence moves from naming the company into the uniform request.
    parts = re.split(
        r"\s+(?=(?:وابي|وأبي|وابغى|وأبغى|وعايز|ومحتاج|واحتاج|وأحتاج|واريد|وأريد|ابي|أبي|ابغى|أبغى|عايز|محتاج|احتاج|أحتاج|اريد|أريد|need|want)\b)",
        value,
        maxsplit=1,
        flags=re.I,
    )
    return parts[0].strip(" .,-")[:180]


def _extract_company_name(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    patterns = [
        r"^(?:اسم\s+شركتي|اسم\s+(?:الشركة|الشركه)|(?:الشركة|الشركه)\s+اسمها|شركتي\s+اسمها|اسمها)\s*[:\-]?\s*(.+)$",
        r"^(?:عندي|لدي)\s+(?:شركة|شركه|مؤسسة|موسسة)(?:\s+اسمها)?\s+(.+)$",
        r"^(?:شركة|شركه|مؤسسة|موسسة)\s+(.+)$",
        r"^(?:أنا|انا)\s+.+?\s+(?:وعندي|عندي|ولدي|لدي)\s+(?:شركة|شركه|مؤسسة|موسسة)(?:\s+اسمها)?\s+(.+)$",
        r"^(?:my\s+company(?:\s+name)?\s+is|company(?:\s+name)?\s+is|business(?:\s+name)?\s+is)\s+(.+)$",
    ]
    for pattern in patterns:
        m = re.match(pattern, t, flags=re.I)
        if m:
            value = re.sub(r"^(?:هي|هو)\s+", "", m.group(1), flags=re.I)
            value = _trim_company_tail(value)
            if value:
                return value
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
    if re.search(r"corporate|office|مكتب|إداري|اداري", t, flags=re.I):
        return "Corporate Office"
    return ""


def _extract_project_hint(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    low = t.lower()
    # Explicit corrections should override old context.
    if re.search(r"(?:طاقم\s*)?أرضي|(?:طاقم\s*)?ارضي|ground\s*crew|ground\s*staff", low, flags=re.I):
        return "طاقم أرضي بالمطار" if re.search(r"[\u0600-\u06FF]", t) else "Airport ground crew"
    if re.search(r"طيارين?\s*(?:فعلي|فعليا|فعليًا|رسمي|رسميين)|formal\s+pilots?|actual\s+pilots?", low, flags=re.I):
        return "طيارين رسميين" if re.search(r"[\u0600-\u06FF]", t) else "Formal pilots"
    if re.search(r"(?:يونيفورمات?|يونيفرمات?|ازياء|أزياء|زي)\s+(?:ل)?(?:طيارين|طيار)|(?:فريق|الفريق)\s+(?:بيشتغل|يشتغل|هو)\s+(?:طيارين|طيار)", t, flags=re.I):
        return "طيارين" if re.search(r"[\u0600-\u06FF]", t) else "Pilots"
    patterns = [
        r"(?:ل|لل)\s*(طيارين|الطيارين|عمال|العمال|موظفين|الموظفين|شيفات|الشيفات|امن|أمن|حراس|فريق مبيعات|استقبال)(?:\s|$)",
        r"(?:فريق|الفريق)\s+(?:بيشتغل|يشتغل|هو)\s+(طيارين|عمال|موظفين|شيفات|امن|أمن|حراس)(?:\s|$)",
        r"(?:for|for my|for our)\s+([A-Za-z][A-Za-z \/&-]{2,80})(?:\?|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, t, flags=re.I)
        if m:
            value = re.sub(r"^ال", "", m.group(1).strip(), flags=re.I)
            return value[:160]
    return ""

def _extract_team_counts(text: str) -> dict[str, int]:
    t = re.sub(r"[,،]", " ", (text or "").lower())
    out: dict[str, int] = {}
    male_patterns = [r"(\d{1,5})\s*(?:ذكر|ذكور|رجل|رجال|male|men)", r"(?:ذكر|ذكور|رجل|رجال|male|men)\s*(\d{1,5})"]
    female_patterns = [r"(\d{1,5})\s*(?:انثى|أنثى|انثي|أنثي|إناث|نساء|female|women)", r"(?:انثى|أنثى|انثي|أنثي|إناث|نساء|female|women)\s*(\d{1,5})"]
    for pat in male_patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            out["male_count"] = int(m.group(1)); break
    for pat in female_patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            out["female_count"] = int(m.group(1)); break
    return out


def _extract_total_quantity(text: str) -> int:
    t = (text or "").strip().lower()
    explicit = re.search(r"(?:اجمالي|إجمالي|المجموع|total|quantity|كمية)\s*[:\-]?\s*(\d{1,6})", t, flags=re.I)
    if explicit:
        return int(explicit.group(1))
    piece = re.search(r"\b(\d{2,6})\s*(?:قطعة|قطعه|قطع|pcs?|pieces?)\b", t, flags=re.I)
    if piece:
        return int(piece.group(1))
    counts = _extract_team_counts(text)
    nums = [int(x) for x in re.findall(r"\b\d{1,6}\b", t)]
    if len(nums) >= 3 and counts.get("male_count") is not None and counts.get("female_count") is not None:
        expected = counts["male_count"] + counts["female_count"]
        if nums[0] == expected:
            return nums[0]
    return 0


def _explicit_custom_uniform(text: str, project: str = "") -> bool:
    t = f"{text or ''} {project or ''}".lower()
    return bool(re.search(r"(?:زي|يونيفورم|uniform)\s*(?:ل)?\s*(?:طيار|طيارين)|pilot\s+uniform|formal\s+pilot", t, flags=re.I))


def _semantic_capture(base: Any, latest_user_text: str, current_field: str = "") -> dict[str, Any]:
    draft = _clean_quote_patch(base)
    if _language_only_message(latest_user_text):
        return draft

    person_name = _extract_person_name(latest_user_text)
    if person_name:
        draft["name"] = person_name

    company_name = _extract_company_name(latest_user_text)
    if company_name:
        explicit_rename = bool(re.search(r"اسم\s+شركتي|اسم\s+(?:الشركة|الشركه)|company\s+name|business\s+name", latest_user_text, flags=re.I))
        if "company" not in draft or explicit_rename:
            draft["company"] = company_name

    # Industry can be volunteered at any time; it must not depend on current_field.
    industry = _normalize_industry_text(latest_user_text)
    if industry and ("industry" not in draft or current_field == "industry"):
        draft["industry"] = industry

    project_hint = _extract_project_hint(latest_user_text)
    if project_hint:
        # Ground/pilot clarifications are authoritative corrections, not merely hints.
        if re.search(r"أرضي|ارضي|ground|طيارين?\s*(?:فعلي|رسمي)|formal\s+pilot|actual\s+pilot|(?:فريق|الفريق).*طيارين|(?:يونيفورمات?|يونيفرمات?|ازياء|أزياء|زي)\s+(?:ل)?طيار", latest_user_text, flags=re.I) or "project" not in draft:
            draft["project"] = project_hint
        if re.search(r"أرضي|ارضي|ground", latest_user_text, flags=re.I):
            kept = [x for x in list(draft.get("uniforms") or []) if x.get("name") != CUSTOM_UNIFORM]
            if kept:
                draft["uniforms"] = kept
            else:
                draft.pop("uniforms", None)

    for key, value in _extract_team_counts(latest_user_text).items():
        draft[key] = value

    uniforms = list(draft.get("uniforms") or [])
    qty = _extract_total_quantity(latest_user_text)
    if qty < 12 and current_field == "uniforms":
        bare = re.fullmatch(r"\s*(\d{1,6})\s*(?:قطعة|قطعه|قطع|pcs?|pieces?)?\s*", latest_user_text or "", flags=re.I)
        if bare:
            qty = int(bare.group(1))
    project_text = str(draft.get("project", "") or "")
    if _explicit_custom_uniform(latest_user_text, project_text) or (qty >= 12 and not uniforms and re.search(r"طيار|pilot", project_text, flags=re.I)):
        found = next((x for x in uniforms if x.get("name") == CUSTOM_UNIFORM), None)
        if not found:
            found = {"name": CUSTOM_UNIFORM, "qty": 0}
            uniforms.append(found)
        if qty >= 12:
            found["qty"] = qty
        draft["uniforms"] = uniforms
    else:
        pending = [x for x in uniforms if int(x.get("qty", 0) or 0) == 0]
        if qty >= 12 and len(pending) == 1:
            pending[0]["qty"] = qty
            draft["uniforms"] = uniforms

    return _clean_quote_patch(draft)


def _merge_model_draft(base: Any, update: Any, latest_user_text: str, current_field: str = "") -> dict[str, Any]:
    """Merge deterministic semantic capture first, then safe model extraction."""
    base_clean = _semantic_capture(base, latest_user_text, current_field)
    if _language_only_message(latest_user_text):
        return base_clean

    incoming = _clean_quote_patch(update)
    # Deterministic captures from the current user turn are authoritative.
    merged = _merge_quote_patch(base_clean, incoming)
    protected = _semantic_capture(base_clean, latest_user_text, current_field)
    merged = _merge_quote_patch(merged, protected)

    if "company" not in base_clean and _looks_like_uniform_intent(latest_user_text):
        # Never let generic intent like "أبي زي" become the company name.
        candidate = str(merged.get("company", "") or "").lower()
        if candidate and candidate in latest_user_text.strip().lower():
            merged.pop("company", None)
    return merged

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
    allowed_uniforms = set(UNIFORMS + [CUSTOM_UNIFORM])
    if last_uniform and last_uniform not in allowed_uniforms:
        last_uniform = fallback.lastUniform if fallback.lastUniform in allowed_uniforms else ""
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
    draft = _semantic_capture(payload.draft_quote, latest, payload.collection.current_field)
    project = str(draft.get("project", "") or "")
    industry = str(draft.get("industry", "") or "")
    combined = f"{latest} {project} {industry}".lower()
    ar = payload.conversation_language == "ar"

    # Ground crew is intentionally checked before pilot/aviation. A visitor may
    # start by saying pilots and then clarify "أرضي"; the clarification wins.
    if re.search(r"أرضي|ارضي|ground\s*crew|ground\s*staff", combined, flags=re.I):
        if ar:
            return (
                "للطاقم الأرضي عندك خيارين منطقيين أكثر من غيرهم: لو الشغل فيه حركة وتشغيل وتحميل فـ Cargo Pants & Workwear أنسب؛ ولو الفريق يتعامل مع المسافرين أو واقف في كاونترات وخدمة عملاء فالبولو أو التي‑شيرت يطلع أرتب. لو وصفت لي دورهم اليومي بجملة، أختار لك واحد منهم بشكل أدق.",
                [
                    {"label":"عرض الـWorkwear","type":"navigate","value":"/pages/workwear"},
                    {"label":"عرض البولو والتي‑شيرت","type":"navigate","value":"/pages/polo-t-shirts"},
                ],
            )
        return (
            "For airport ground crew, I’d narrow it to two sensible routes: Cargo Pants & Workwear for active operational roles, or Polos/T-shirts for customer-facing counters and service teams. Tell me what they do day to day and I’ll choose between them.",
            [
                {"label":"View Workwear","type":"navigate","value":"/pages/workwear"},
                {"label":"View Polo & T-Shirts","type":"navigate","value":"/pages/polo-t-shirts"},
            ],
        )

    if re.search(r"طيار|طيارين|pilot|pilots|airline|aviation", combined, flags=re.I):
        if ar:
            return (
                "لو المقصود الطيارين أنفسهم، الأفضل نمشيه كـ Custom Uniform لأن الموقع ما عنده فئة جاهزة باسم زي طيارين. نقدر نبني الطلب على الستايل الرسمي، اللون، الشعار والكمية، وبعدها فريق ALF يراجعه معك. إذا تبي، أبدأ معك بالشكل العام أو الكمية.",
                [{"label":"عرض الزي المخصص","type":"navigate","value":"/pages/custom-uniforms"}],
            )
        return (
            "For the pilots themselves, I’d treat it as a Custom Uniform requirement because the site does not list a ready pilot-uniform category. We can define the formal look, color, branding and quantity for ALF to review. We can start with the look or the quantity.",
            [{"label":"View custom uniforms","type":"navigate","value":"/pages/custom-uniforms"}],
        )

    if re.search(r"عمال|عامل|مخزن|تحميل|تشغيل|صيانة|لوجست|warehouse|worker|workers|labou?r|operations|maintenance|logistics", combined, flags=re.I):
        if ar:
            return (
                "بما إن الزي للعمال، أنا أميل أبدأ معك بـ Cargo Pants & Workwear لأنه عملي للحركة والشغل اليومي. لو الفريق قدام العملاء أكثر من التشغيل، البولو والتي‑شيرت يطلع أنظف. لو شغلهم تشغيل وحركة، فالـWorkwear هو اختياري الأول.",
                [
                    {"label":"عرض الـWorkwear","type":"navigate","value":"/pages/workwear"},
                    {"label":"عرض البولو والتي‑شيرت","type":"navigate","value":"/pages/polo-t-shirts"},
                ],
            )
        return (
            "For a workers team, I’d start with Cargo Pants & Workwear for active day-to-day work. If they are mainly customer-facing, polos and T-shirts can look cleaner.",
            [
                {"label":"View Workwear","type":"navigate","value":"/pages/workwear"},
                {"label":"View Polo & T-Shirts","type":"navigate","value":"/pages/polo-t-shirts"},
            ],
        )
    if re.search(r"مطعم|كافيه|مطبخ|شيف|restaurant|cafe|kitchen|chef|hospitality", combined, flags=re.I):
        return (("لو الفريق مطعم أو مطبخ، أبدأ بزي الشيف والمرايل، ولو عندك استقبال أو خدمة عملاء أضيف لهم بولو موحد." if ar else "For a restaurant or kitchen team, I’d start with Chef Uniforms & Aprons, with polos as a clean option for front-of-house staff."), [{"label":"عرض زي الشيف","type":"navigate","value":"/pages/chef-uniforms"}] if ar else [{"label":"View Chef Uniforms","type":"navigate","value":"/pages/chef-uniforms"}])
    if re.search(r"أمن|حراسة|security|guard", combined, flags=re.I):
        return (("لو الفريق أمن أو حراسة، الزي الأمني هو الاختيار الطبيعي كبداية، وبعدها نضبط اللون والبراندنج حسب الجهة." if ar else "For a security team, Security Uniforms are the natural starting point, then we can tailor color and branding to the organization."), [{"label":"عرض الزي الأمني","type":"navigate","value":"/pages/security-uniforms"}] if ar else [{"label":"View Security Uniforms","type":"navigate","value":"/pages/security-uniforms"}])
    if re.search(r"فعالية|فعاليات|معرض|ترويج|event|promo|promotion|exhibition", combined, flags=re.I):
        return (("للفعاليات والترويج، Event & Promo Team Apparel هو الأقرب، والبولو بديل ممتاز لو تبي شكل أبسط وأكثر رسميّة." if ar else "For events and promotional teams, Event & Promo Team Apparel is the closest match; polos are a good alternative for a cleaner corporate look."), [{"label":"عرض زي الفعاليات","type":"navigate","value":"/pages/event-uniforms"}] if ar else [{"label":"View Event Apparel","type":"navigate","value":"/pages/event-uniforms"}])
    if re.search(r"شركة|مكتب|استقبال|مبيعات|corporate|office|reception|sales", combined, flags=re.I):
        return (("لفريق مكتب أو استقبال أو مبيعات، البولو والتي‑شيرت غالبًا أفضل بداية: شكله مرتب وسهل نطابقه مع ألوان وهوية الشركة." if ar else "For an office, reception or sales team, polos and T-shirts are usually the best starting point: clean, versatile and easy to match to the brand."), [{"label":"عرض البولو والتي‑شيرت","type":"navigate","value":"/pages/polo-t-shirts"}] if ar else [{"label":"View Polo & T-Shirts","type":"navigate","value":"/pages/polo-t-shirts"}])

    return (("أكيد أرشح لك، بس ما أبي أعطيك اختيار عشوائي. وصف لي طبيعة شغل الفريق بجملة واحدة وأنا أعطيك اختياري الأول وبديله." if ar else "Absolutely — tell me in one line what the team does day to day and I’ll give you a first choice plus an alternative."), [])

def _natural_local_reply(payload: "ChatRequest", reason: str = "") -> dict[str, Any]:
    """Customer-first fail-safe that remains useful even when the model is unavailable."""
    latest = payload.messages[-1].content if payload.messages else ""
    ar = payload.conversation_language == "ar"
    draft = _semantic_capture(payload.draft_quote, latest, payload.collection.current_field)
    field_status = _clean_field_status({}, draft, payload.collection.resolved)
    actions: list[dict[str, Any]] = []
    t = (latest or "").strip().lower()
    company_name = _extract_company_name(latest)
    project = str(draft.get("project", "") or "")
    uniforms = list(draft.get("uniforms") or [])

    def pack(reply: str, acts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "reply": reply,
            "actions": acts or [],
            "auto_action": None,
            "context": _clean_context({}, payload.context),
            "draft_quote": draft,
            "field_status": field_status,
            "conversation_language": payload.conversation_language,
            "model": MODEL,
            "provider": "local-failsafe",
            "fallback_reason": reason[:120],
        }

    if _is_greeting(latest) or re.match(r"^(?:هلا+|مرحبا|مرحبًا|أهلين|اهلين)\s+(?:أنا|انا)\s+", latest.strip(), flags=re.I):
        name = str(draft.get("name", "") or "").strip()
        if ar:
            return pack(f"هلا {name} 👋 تشرفنا. قل لي شنو تحتاج وأنا أمشي معك." if name else "هلا 👋 حياك الله. قل لي شنو تحتاج وأنا أمشي معك.")
        return pack(f"Hi {name} 👋 Nice to meet you. Tell me what you need and I’ll work through it with you." if name else "Hi 👋 Welcome. Tell me what you need and I’ll work through it with you.")

    if _is_casual_nudge(latest):
        if re.search(r"اسألني انت|اسالني انت|اسأل انت|اسال انت|ابدأ اسأل|ابدأ اسال|كمل انت|كمّل انت|ask me|you ask|lead me", latest, flags=re.I):
            current = payload.collection.current_field or ""
            prompts_ar = {
                "company": "تمام، نبدأ من عندك: شنو اسم الشركة أو الجهة؟",
                "industry": "تمام. طبيعة نشاط الشركة شنو تقريبًا؟",
                "project": "حلو. الزي هذا لأي فريق أو وظيفة بالتحديد؟",
                "uniforms": "أقدر أرشح لك بدل ما أخليك تختار عشوائي. شنو طبيعة شغل الفريق اليومية؟",
                "color": "تمام. في لون معين في بالك أو لون مرتبط بهوية الشركة؟",
            }
            prompts_en = {
                "company": "Sure — what is the company or organization name?",
                "industry": "Sure — what kind of business is it?",
                "project": "Great — which team or job role is the uniform for?",
                "uniforms": "I can recommend instead of making you pick blindly. What does the team do day to day?",
                "color": "Great. Do you already have a preferred color or brand color?",
            }
            return pack((prompts_ar if ar else prompts_en).get(current, "تمام، اسألني أو خلّيني أقودك من اللي ناقص عندي." if ar else "Sure — I can lead from the most useful missing detail."))
        return pack("معك ومركز 👌 كمل، شنو في بالك؟" if ar else "I’m with you 👍 Go ahead — what’s on your mind?")

    # Short clarification after a pilot/ground-crew disambiguation.
    if re.fullmatch(r"(?:ارضي|أرضي|طاقم ارضي|طاقم أرضي|ground|ground crew|ground staff)", t, flags=re.I):
        reply, actions = _local_recommendation(payload)
        return pack(reply, actions)
    if re.fullmatch(r"(?:طيارين? فعليا|طيارين? فعليًا|طيارين? رسميين?|pilots?|formal pilots?)", t, flags=re.I):
        reply, actions = _local_recommendation(payload)
        return pack(reply, actions)

    if re.fullmatch(r"(?:شركتي|الشركة|الشركه|اسم الشركة|اسم الشركه|my company|company)", t, flags=re.I):
        return pack("تمام، شنو اسم الشركة؟" if ar else "Sure — what’s the company name?")

    if company_name:
        return pack((f"تمام، سجلت اسم الشركة: {company_name}. قل لي شنو تحتاج لها بالضبط وأنا أرتب الخيارات معك." if ar else f"Got it — I have the company as {company_name}. Tell me what you need for the team and I’ll help narrow the options."))

    # Multi-fact team split / total quantity.
    counts = _extract_team_counts(latest)
    total_qty = _extract_total_quantity(latest)
    if counts:
        bits = []
        if total_qty:
            bits.append((f"الإجمالي {total_qty}" if ar else f"total {total_qty}"))
        if "male_count" in counts:
            bits.append((f"{counts['male_count']} رجال" if ar else f"{counts['male_count']} men"))
        if "female_count" in counts:
            bits.append((f"{counts['female_count']} سيدات" if ar else f"{counts['female_count']} women"))
        joined = "، ".join(bits)
        if re.search(r"طيار|pilot", project, flags=re.I):
            return pack((f"تمام، ثبت عندي {joined}. وبما إن الطلب للطيارين، نقدر نمشيه كـ Custom Uniform بدل ما نختار فئة غير مناسبة من الكتالوج. عندكم لون أو هوية معينة للزي؟" if ar else f"Got it — I have {joined}. Since this is for pilots, we can treat it as a Custom Uniform requirement instead of forcing a mismatched catalog category. Do you have a preferred color or brand look?"), [{"label":"عرض الزي المخصص" if ar else "View custom uniforms","type":"navigate","value":"/pages/custom-uniforms"}])
        return pack((f"تمام، ثبت عندي {joined}." if ar else f"Got it — I have {joined}."))

    if _asks_for_recommendation(latest):
        reply, actions = _local_recommendation(payload)
        return pack(reply, actions)

    asks_types = _asks_uniform_types(latest) or bool(re.search(r"(?:ازياء|أزياء|يونيفورمات|ملابس\s+عمل).*(?:عندكم|متوفر|متاحة|available|types)|(?:ايه|إيه|وش|شنو|ما).*(?:الازياء|الأزياء|الأنواع|انواع|الخيارات|خيارات)", t, flags=re.I))
    if asks_types:
        if re.search(r"أرضي|ارضي|ground", project, flags=re.I):
            return pack(("للطاقم الأرضي، أقرب خيارين فعليًا هم: Cargo Pants & Workwear لو الشغل تشغيلي وفيه حركة، أو Polo Shirts & T‑Shirts لو الفريق يتعامل مع المسافرين وواجهة الخدمة. إذا قلت لي دورهم اليومي أقول لك أي واحد أختار." if ar else "For ground crew, the two most relevant options are Cargo Pants & Workwear for active operational roles, or Polo Shirts & T-Shirts for customer-facing service teams. Tell me what they do day to day and I’ll choose between them."), [
                {"label":"عرض الـWorkwear" if ar else "View Workwear","type":"navigate","value":"/pages/workwear"},
                {"label":"عرض البولو" if ar else "View Polo","type":"navigate","value":"/pages/polo-t-shirts"},
            ])
        if re.search(r"طيار|pilot|airline|aviation", f"{latest} {project}", flags=re.I):
            return pack(("للطيارين الرسميين ما عندنا فئة جاهزة باسم «زي طيارين» داخل الكتالوج، فالأصح نخليه Custom Uniform. أما لو كنت تقصد الطاقم الأرضي فالأقرب Workwear أو Polo حسب طبيعة الدور." if ar else "For formal pilots, the current catalog does not list a dedicated pilot category, so the right route is Custom Uniform. For ground crew, Workwear or Polos are the closer options depending on the role."), [{"label":"عرض الزي المخصص" if ar else "View custom uniforms","type":"navigate","value":"/pages/custom-uniforms"}])
        return pack(("أكيد. الموجود عند ALF يشمل بولو وتي‑شيرت، زي شيف ومرايل، قبعات وإكسسوارات شيف، كارجو وملابس عمل، زي أمني، وملابس فرق الفعاليات والترويج. وإذا احتياجك خارج الفئات دي نقدر نمشيه كـ Custom Uniform." if ar else "ALF offers polos and T-shirts, chef uniforms and aprons, chef caps/accessories, cargo/workwear, security uniforms, and event/promo apparel. If your requirement sits outside those categories, we can handle it as a Custom Uniform enquiry."), [{"label":"عرض كل الأنواع" if ar else "Browse all uniforms","type":"navigate","value":"/pages/custom-uniforms"}])

    if _looks_like_uniform_intent(latest) or re.search(r"(?:ازياء|أزياء|يونيفورمات|ملابس\s+عمل|زي\s+طيار)", t, flags=re.I):
        reply, actions = _local_recommendation(payload)
        return pack(reply, actions)

    # A direct quantity can complete the one selected/pending uniform.
    if total_qty >= 12:
        if uniforms:
            item = uniforms[0]
            return pack((f"تمام، ثبت عندي {total_qty} قطعة من {item.get('name','الزي')}. بالنسبة للشكل، في لون معين أو لون من هوية الشركة تحب نمشي عليه؟" if ar else f"Great — I have {total_qty} pieces for {item.get('name','the uniform')}. Do you have a preferred color or brand color for the look?"))
        if re.search(r"طيار|pilot", project, flags=re.I):
            return pack((f"تمام، العدد {total_qty} قطعة. وبما إن الطلب للطيارين، أسجله كـ Custom Uniform بدل ما أختار لك فئة غلط. في لون أو ستايل رسمي معين في بالك؟" if ar else f"Got it — {total_qty} pieces. Since this is for pilots, I’ll treat it as a Custom Uniform requirement rather than force the wrong category. Do you have a preferred color or formal style?"), [{"label":"عرض الزي المخصص" if ar else "View custom uniforms","type":"navigate","value":"/pages/custom-uniforms"}])

    if re.search(r"(price|pricing|cost|kwd|دينار|سعر|تكلفة)", t):
        return pack("السعر يتحدد حسب النوع والكمية والمقاسات والبراندنج، وفريق ALF يأكد عرض السعر بعد مراجعة الطلب. نقدر نضبط المواصفات هنا أول عشان يوصلهم Brief واضح." if ar else "Pricing depends on uniform type, quantity, sizes and branding, and ALF confirms the quotation after reviewing the requirement. I can help you shape a clear brief first.")
    if re.search(r"(minimum|moq|اقل كمية|أقل كمية)", t):
        return pack("الحد الأدنى يبدأ من 12 قطعة لكل نوع يونيفورم." if ar else "The minimum starts from 12 pieces per uniform type.")
    if re.search(r"(logo|brand|branding|embroider|embroidery|print|printing|طباعة|تطريز|لوجو|شعار)", t):
        return pack("متوفر تطريز وطباعة. الأنسب يعتمد على نوع الزي وشكل الشعار والاستخدام؛ إذا قلت لي القطعة اللي اخترتها أرشح لك بين الاثنين." if ar else "ALF offers embroidery and printing. The better option depends on the garment, logo and use; tell me the item and I’ll help you choose.", [{"label":"خيارات البراندنج" if ar else "Branding options","type":"navigate","value":"/pages/embroidery-printing"}])

    # Never return an empty reply. If the model is down, keep the conversation alive
    # instead of forcing the storefront into a repetitive field prompt.
    current = (payload.collection.current_field or "").strip()
    if current == "color":
        return pack("تمام. وبالنسبة للشكل العام، في لون معين في بالك أو لون مرتبط بهوية الشركة؟" if ar else "Great. Do you have a preferred color or a brand color for the overall look?")
    if current == "uniforms":
        reply, actions = _local_recommendation(payload)
        return pack(reply, actions)
    return pack("فاهمك. كمل لي الفكرة براحتك وأنا أرد عليك على نفس النقطة، ونرتب تفاصيل الطلب أثناء الكلام بدون ما نحولها لاستبيان." if ar else "I’m with you. Keep going and I’ll respond to the point you’re making; we can collect the order details naturally along the way.")

@app.get("/")
def root() -> dict[str, str]:
    return {"service": APP_NAME, "status": "online", "model": MODEL}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": "2.0.0", "architecture": "natural-dialogue-deterministic-memory", "model": MODEL, "ai_configured": bool(GROQ_API_KEY)}


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    _rate_limit(request)
    latest_user_text = payload.messages[-1].content if payload.messages else ""

    # Deterministic state comes first. This prevents a language model from
    # forgetting, overwriting, or re-asking facts that the storefront already captured.
    draft_quote = _semantic_capture(payload.draft_quote, latest_user_text, payload.collection.current_field)
    field_status = _clean_field_status({}, draft_quote, payload.collection.resolved)

    if not GROQ_API_KEY:
        local = _natural_local_reply(payload, reason="missing_api_key")
        local["draft_quote"] = draft_quote
        local["field_status"] = field_status
        return local

    runtime_context = {
        "page": {"path": payload.page.path, "label": payload.page.label},
        "draft_quote": draft_quote,
        "collection": {
            "active": payload.collection.active,
            "current_field": payload.collection.current_field,
            "resolved": field_status,
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
    # Enough history for conversational continuity without burning the daily token budget.
    messages.extend({"role": m.role, "content": m.content} for m in payload.messages[-12:])

    def run_model(model: str) -> str:
        client = Groq(api_key=GROQ_API_KEY, timeout=20.0, max_retries=0)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.72,
            "max_completion_tokens": 480,
        }
        # Qwen's non-thinking mode is better suited to a quick natural sales conversation.
        if model.startswith("qwen/"):
            kwargs.update({"reasoning_effort": "none", "top_p": 0.8})
        elif model.startswith("openai/gpt-oss"):
            kwargs.update({"reasoning_effort": "low"})
        completion = client.chat.completions.create(**kwargs)
        return (completion.choices[0].message.content or "").strip()

    provider_model = MODEL
    try:
        reply = run_model(MODEL)
        if not reply:
            raise ValueError("empty_model_reply")
    except Exception as first_exc:
        first_status = getattr(first_exc, "status_code", None)
        print(f"[ALF primary model failed] model={MODEL} status={first_status} error={first_exc}")
        try:
            provider_model = FALLBACK_MODEL
            reply = run_model(FALLBACK_MODEL)
            if not reply:
                raise ValueError("empty_fallback_reply")
        except Exception as second_exc:
            second_status = getattr(second_exc, "status_code", None)
            print(f"[ALF fallback model failed] model={FALLBACK_MODEL} status={second_status} error={second_exc}")
            local = _natural_local_reply(payload, reason=f"provider_failure:{second_status or first_status or 'unknown'}")
            local["draft_quote"] = draft_quote
            local["field_status"] = field_status
            return local

    # Keep provider output clean; accidental surrounding quotes are common with some models.
    reply = reply.strip()
    if len(reply) >= 2 and reply[0] == reply[-1] == '"':
        reply = reply[1:-1].strip()
    reply = reply[:3000]

    return {
        "reply": reply,
        "actions": [],
        "auto_action": None,
        "context": _clean_context({}, payload.context),
        "draft_quote": draft_quote,
        "field_status": field_status,
        "conversation_language": payload.conversation_language,
        "model": provider_model,
        "provider": "groq",
    }

