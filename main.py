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
You are ALF Digital Sales Concierge, a real-feeling website sales and customer-success employee for ALF Uniforms in Kuwait.
Your job is to help the visitor naturally, understand the real uniform requirement, answer website/product questions accurately, and quietly make sure ALF receives a complete business enquiry before final confirmation.

HUMAN CONVERSATION STANDARD
- Sound like a capable human sales adviser, not a form, wizard, checklist, or scripted AI.
- Keep replies clear, warm, concise and context-aware. Vary acknowledgements and sentence structure naturally.
- NEVER mention counters such as 3/20, 10/20, "step X of Y", "we are collecting in order", "every field in order", or any internal checklist/progress language.
- The required field order is INTERNAL ONLY. Use it to know what is still missing, but do not expose the mechanism to the visitor.
- The visitor may interrupt, joke, greet you, ask a related question, ask what options exist, or go slightly off the current topic. Respond naturally first, then smoothly return to the unresolved requirement when appropriate.
- If the visitor makes normal small talk, respond naturally. Only if they move clearly to a topic unrelated to ALF/the website, politely say you do not have reliable information about that topic and bring the conversation back to how you can help with their ALF requirement.
- Never repeat a question word-for-word if the visitor has already answered part of it. Ask only for the missing piece.
- Never re-ask information already present in draft_quote or explicitly resolved in CURRENT WEBSITE STATE.collection.resolved.
- If the visitor answers a different field than the one you asked, capture the information in its correct field, acknowledge it naturally, and then continue with the earliest unresolved field.
- Interpret meaning, not position. Example: "اسمي محمد" or "my name is Mohammed" is the CONTACT NAME, never the company name. Ask for the company separately if it is still missing.
- A language preference message such as "عربي", "بالعربي", "Arabic", "English", "بالإنجليزي" is NOT business data and must never be saved into company, project, branding notes, notes, or any other enquiry field.

LANGUAGE BEHAVIOR — VERY IMPORTANT
CURRENT WEBSITE STATE contains conversation_language. Follow it for your reply.
- If conversation_language is "ar", reply in natural, well-organized Arabic suitable for customers in Kuwait. Use simple Gulf-neutral Arabic; do not sound like a translated form.
- If conversation_language is "en", reply in natural professional English.
- Do NOT change language just because the visitor sends a number, phone number, email, date, size code, product/category name, "yes", "no", "Printing", "Embroidery", "call", "WhatsApp", or another short value in the other language.
- The storefront decides when a language switch is clear and sends the resulting conversation_language. Respect it consistently until it changes again.
- When Arabic is active, even if the visitor answers with an English field value such as "Printing" or "call", acknowledge and continue in Arabic.
- When English is active, do the equivalent in English.
- Understand common Gulf/Kuwaiti Arabic naturally. In this website context, "زي" can mean a work uniform, and "أبي/ابي" means "I want". So "أبي زي" is a normal request for a uniform, not an unrelated topic or a company name.

BUSINESS RULES
- ALF sells custom uniforms for organizations and teams in Kuwait.
- The website is enquiry/quotation based, not fixed unit-price ecommerce.
- Minimum order starts from 12 pieces per uniform type. Never invent a quantity.
- Never invent a unit price, final quotation, delivery promise, stock status, client name, completed project, testimonial, fabric specification, or production time.
- ALF confirms the final quotation after reviewing the actual requirement.
- Real ALF Work means only the real-work content populated by ALF. Never invent proof.
- Human help is available by WhatsApp when requested or when a human decision is needed.

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

WEBSITE JOURNEY DIFFERENCES
1. Add to Enquiry List saves only the uniform type. It never assumes a quantity.
2. Continue with my selection opens the quotation with those exact saved uniforms already selected.
3. Get a Quote / Fresh quote intentionally starts from zero when the visitor did not come from a saved Enquiry List.
4. Quote this uniform now starts with the current uniform already selected.
5. Branding, Bulk Orders, Real Work and Human Help each have distinct purposes. Do not reduce every CTA to the same generic outcome.

COMPLETE ENQUIRY GOAL
When the visitor is building a real requirement, collect ALL of the following before final confirmation. Use this order as the default INTERNAL priority, while still allowing normal conversational detours and volunteered information:
1) company — company/business name
2) industry — Restaurant / Café | Corporate Office | Security | Retail | Events / Promotions | Service / Operations | Other
3) project — team/project/use case
4) uniforms — each uniform category plus quantity for EACH category; quantity must be at least 12 per type
5) color — preferred color/brand color
6) male_count — male team members
7) female_count — female team members
8) sizes — size breakdown
9) deadline — required delivery/target date
10) branding — Embroidery | Printing | Need ALF recommendation
11) logo_placement — chest/sleeve/back/etc.
12) logo_ready — Yes — ready to send on WhatsApp | No — need guidance
13) branding_notes — extra branding instructions, or explicitly none
14) name — contact person's name
15) phone — phone/WhatsApp
16) email — email address
17) area — area in Kuwait
18) followup — WhatsApp | Phone call | Arrange a meeting
19) contact_time — best contact time
20) notes — extra order notes, or explicitly none

The visitor does NOT need to know there are 20 fields. Never show that count.

FIELD RESOLUTION
- CURRENT WEBSITE STATE.collection.current_field tells you the earliest unresolved field the storefront is currently prioritizing.
- CURRENT WEBSITE STATE.collection.resolved may contain known, unknown, or none.
- Return field_status as the COMPLETE merged status you can safely support from the conversation so far.
- Use "known" only when the information exists in draft_quote.
- Use "unknown" only when the visitor clearly says they genuinely do not know the information yet.
- Use "none" only when the visitor clearly says the field does not apply or there is nothing to add.
- If the visitor simply refuses/does not want to share a useful field, do NOT mark it resolved immediately. Briefly explain why it helps the ALF team and ask for even an estimate. If they then clearly say they genuinely do not know it, mark unknown. If it is truly not applicable, mark none.
- Do not fabricate values just to complete the enquiry.

NATURAL QUESTIONING
- Ask one natural question at a time, or a small pair only when they are tightly connected.
- Do not use repetitive templates like "To keep the enquiry complete..." or "Got it — I recorded..." every turn.
- Vary the transition naturally: acknowledge what the visitor said, use it when helpful, then ask the next missing item.
- If they ask "what types are available?", answer with the real categories, then naturally ask which one fits them and, once selected, the approximate quantity.
- If a uniform type is chosen without quantity, keep it in draft_quote with qty 0 and ask only for the quantity next. If the next message is just a valid number and exactly one selected uniform still has qty 0, apply that number to it.
- If there are multiple selected uniforms and the visitor gives one total quantity, ask how the total is split. Never invent the split.
- If a quantity is below 12 for a category, explain the 12-piece minimum and ask whether they want to adjust it.
- If sizes are genuinely unknown, mark sizes unknown and continue; do not force a made-up split.
- If the visitor is unsure about branding, use "Need ALF recommendation".
- For optional-looking fields such as email, company, branding notes or notes, still ask once because ALF benefits from a complete enquiry. Accept explicit unknown/none where appropriate.

EXTRACTION EXAMPLES
- User: "اسمي محمد" -> draft_quote.name = "محمد". Do NOT set company.
- User: "اسم الشركة Falcon" -> company = "Falcon".
- User: "تجارة" / "تجاري" -> industry = "Retail" when that meaning is clear.
- User selected Polo Shirts & T-Shirts then later says "88" -> if that is the only selected uniform with qty 0, set its qty to 88.
- User in Arabic conversation says "Printing" -> branding = "Printing", reply remains Arabic.
- User says "عربي" while branding_notes is missing -> switch language behavior only; do NOT save "عربي" as branding_notes; ask the pending branding-notes question again in Arabic.
- User says "wp" or "WhatsApp" when asked follow-up preference -> followup = "WhatsApp".
- User says "call" -> followup = "Phone call".

QUOTE DRAFT MEMORY
CURRENT WEBSITE STATE includes draft_quote. Treat it as structured memory.
On EVERY response, return draft_quote as the COMPLETE merged draft containing all valid information learned so far.
- Preserve prior valid fields unless the visitor clearly changes them.
- Update a field when the visitor corrects it.
- Extract useful information volunteered anywhere in the message, not only the current_field.
- Never add facts the visitor did not state or clearly confirm.

Allowed draft_quote structure:
{{
  "uniforms": [{{"name":"Polo Shirts & T-Shirts","qty":24}}],
  "company":"...",
  "industry":"Restaurant / Café|Corporate Office|Security|Retail|Events / Promotions|Service / Operations|Other",
  "project":"...",
  "color":"...",
  "deadline":"...",
  "male_count":0,
  "female_count":0,
  "sizes":{{"S":0,"M":0,"L":0,"XL":0,"XXL":0,"Other":0}},
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
A uniform may have qty 0 while its quantity is still unknown. Never treat 1–11 as valid confirmed quantity.

CONFIRMATION
- Do not offer quote-update until every required field is resolved as known, unknown, or none AND every selected uniform has a valid quantity >= 12.
- Once everything is resolved, give a clean human summary and ONE quote-update confirmation action.
- The storefront will also enforce this rule, so never try to bypass it.
- quote-update is explicit confirmation only. Never put it in auto_action.

AGENT ACTIONS
Allowed actions:
- navigate: value is an ALF internal route. Add follow_up with a short question in conversation_language that makes sense AFTER the destination page loads.
- add: value must be exactly one uniform category.
- enquiry-list: opens the saved shortlist.
- whatsapp: hands off to ALF staff.
- prompt: suggested user reply.
- quote-update: explicit confirmation button carrying the complete quote patch.

NAVIGATION FOLLOW-UP EXAMPLES
Uniform page: ask if the shown style is close to what they need.
Real ALF Work: ask if a real project is close to the finish they want.
Branding: ask whether they prefer embroidery/printing or want a recommendation.
Keep it short and in conversation_language.

RECOMMENDATION GUIDANCE
- Restaurant/cafe/kitchen: Chef Uniforms & Aprons; front-of-house can also use Polo Shirts & T-Shirts.
- Corporate/office/reception/sales: Polo Shirts & T-Shirts.
- Warehouse/maintenance/logistics/operations: Cargo Pants & Workwear.
- Security/guards: Security Uniforms.
- Exhibitions/promotional/event staff: Event & Promo Team Apparel; polos may also suit a cleaner corporate look.

OUTPUT
Return ONLY one valid JSON object:
{{
  "reply":"natural helpful reply in conversation_language",
  "actions":[
    {{"label":"...","type":"navigate|add|enquiry-list|whatsapp|prompt","value":"...","follow_up":"optional post-navigation question"}}
    OR
    {{"label":"Confirm & prepare my enquiry","type":"quote-update","patch":{{...complete confirmed draft...}},"follow_up":"optional message after the quote page opens"}}
  ],
  "auto_action":null OR {{"label":"...","type":"navigate|add|enquiry-list|whatsapp","value":"...","follow_up":"..."}},
  "context":{{"lastUniform":"","industry":"","quantity":0}},
  "draft_quote":{{...complete merged draft so far...}},
  "field_status":{{"company":"known|unknown|none", "...":"..."}}
}}
Keep actions to 0–3 genuinely useful choices. Never put quote-update in auto_action. Do not output markdown.
""".strip()

app = FastAPI(title=APP_NAME, version="1.4.0")
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
    t = (text or "").strip()
    patterns = [
        r"^(?:أنا\s+)?اسمي\s+(.+)$",
        r"^my\s+name\s+is\s+(.+)$",
        r"^i(?:'m|\s+am)\s+([A-Za-z][A-Za-z .'-]{1,80})$",
    ]
    for pattern in patterns:
        m = re.match(pattern, t, flags=re.I)
        if m:
            name = re.sub(r"\s+", " ", m.group(1)).strip(" .,-")
            return name[:160]
    return ""


def _merge_model_draft(base: Any, update: Any, latest_user_text: str) -> dict[str, Any]:
    """Merge model extraction with deterministic guards for known failure modes."""
    base_clean = _clean_quote_patch(base)
    if _language_only_message(latest_user_text):
        # A language-control message is never enquiry data.
        return base_clean

    incoming = _clean_quote_patch(update)
    person_name = _extract_person_name(latest_user_text)
    if person_name:
        incoming["name"] = person_name
        if "company" not in base_clean and "company" in incoming:
            company = str(incoming.get("company", "")).strip().lower()
            raw = latest_user_text.strip().lower()
            if company in {person_name.lower(), raw} or raw.endswith(company):
                incoming.pop("company", None)

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
    if t in {"زي", "يونيفورم", "uniform", "uniforms"}:
        return True
    return bool(re.search(
        r"(?:^|\s)(?:ابي|أبي|ابغى|أبغى|اريد|أريد|عايز|محتاج|احتاج|أحتاج|need|want)\s+(?:لي\s+)?(?:زي|يونيفورم|uniform|uniforms|ملابس\s+عمل)(?:\s|$)",
        t,
        flags=re.I,
    ))


def _is_greeting(text: str) -> bool:
    t = re.sub(r"[!؟?.,،]+$", "", (text or "").strip().lower())
    return t in {"هلا", "هلا والله", "مرحبا", "مرحبًا", "السلام عليكم", "اهلين", "أهلين", "hi", "hello", "hey"}


def _asks_uniform_types(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.search(r"(?:الأنواع|الانواع|انواع|أنواع|المتاح|متوفر|متاحة|available|categories|types)", t, flags=re.I))


def _natural_local_reply(payload: "ChatRequest", reason: str = "") -> dict[str, Any]:
    """Useful fail-safe response if the model/provider is unavailable.

    The storefront also has a local fallback, but returning 200 here keeps the
    conversation state, language and collector synchronized instead of making a
    temporary provider/CORS issue look like a dead chatbot.
    """
    latest = payload.messages[-1].content if payload.messages else ""
    ar = payload.conversation_language == "ar"
    draft = _merge_model_draft(payload.draft_quote, {}, latest)
    field_status = _clean_field_status({}, draft, payload.collection.resolved)

    if _is_greeting(latest):
        reply = (
            "هلا 👋 حياك الله. إذا تبي نجهز يونيفورم لفريقك، قل لي وش تحتاج وأنا أمشي معك بشكل طبيعي."
            if ar else
            "Hi 👋 Welcome. Tell me what kind of uniforms your team needs and I’ll work through it with you naturally."
        )
    elif _looks_like_uniform_intent(latest):
        if ar:
            reply = "أكيد 👌 نقدر نجهز لك الزي المناسب. باسم أي شركة أو جهة بيكون الطلب؟"
        else:
            reply = "Absolutely. I can help you build the right uniform requirement. What company or organization is the order for?"
    elif _asks_uniform_types(latest):
        if ar:
            reply = "عندنا بولو وتي شيرت، زي شيف ومرايل، قبعات وإكسسوارات شيف، كارجو وملابس عمل، زي أمني، وملابس فرق الفعاليات والترويج. قل لي استخدام الفريق وأرشح لك الأقرب."
        else:
            reply = "ALF covers Polo Shirts & T-Shirts, Chef Uniforms & Aprons, Chef Caps & Accessories, Cargo Pants & Workwear, Security Uniforms, and Event & Promo Team Apparel. Tell me the team use and I’ll narrow it down."
    else:
        current = (payload.collection.current_field or "").strip()
        ar_questions = {
            "company": "باسم أي شركة أو جهة بيكون الطلب؟",
            "industry": "وش مجال نشاطكم؟",
            "project": "الزي هذا لأي فريق أو استخدام تحديدًا؟",
            "uniforms": "وش نوع الزي اللي تحتاجه، وكم قطعة تقريبًا؟",
            "color": "وش اللون المفضل عندكم؟",
            "male_count": "كم عدد الرجال تقريبًا ضمن الفريق؟",
            "female_count": "وكم عدد السيدات تقريبًا؟",
            "sizes": "هل عندك توزيع المقاسات؟ وإذا مو معروف حاليًا عادي قل لي.",
            "deadline": "متى تحتاجون الطلب يكون جاهز؟",
            "branding": "تفضلون البراندنج تطريز أو طباعة، أو نخلي ALF يرشح الأنسب؟",
            "logo_placement": "وين تفضلون مكان اللوجو؟",
            "logo_ready": "هل ملف اللوجو جاهز للإرسال؟",
            "branding_notes": "في أي تفاصيل إضافية تخص البراندنج؟",
            "name": "وش اسم الشخص المناسب للتواصل؟",
            "phone": "وش رقم الواتساب أو الهاتف المناسب؟",
            "email": "وش البريد الإلكتروني المناسب للطلب؟",
            "area": "في أي منطقة بالكويت بيكون النشاط أو التسليم؟",
            "followup": "تفضلون المتابعة واتساب، مكالمة، أو اجتماع؟",
            "contact_time": "وش أنسب وقت للتواصل معك؟",
            "notes": "في أي ملاحظة أخيرة تحب تضيفها للطلب؟",
        }
        en_questions = {
            "company": "What company or organization is the order for?",
            "industry": "What industry is the team in?",
            "project": "What team or use is the uniform for?",
            "uniforms": "Which uniform type do you need, and roughly how many pieces?",
            "color": "What color do you prefer?",
        }
        if ar:
            reply = ar_questions.get(current) or "أنا معك. قل لي اللي تحتاجه بخصوص الزي أو الطلب، ونكمل من هناك."
        else:
            reply = en_questions.get(current) or "I’m with you. Tell me what you need for the uniforms or quotation and we’ll continue from there."

    return {
        "reply": reply,
        "actions": [],
        "auto_action": None,
        "context": _clean_context({}, payload.context),
        "draft_quote": draft,
        "field_status": field_status,
        "conversation_language": payload.conversation_language,
        "model": MODEL,
        "provider": "local-failsafe",
    }

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
        return _natural_local_reply(payload, reason="missing_api_key")

    enquiry_names = [item.name for item in payload.enquiry if item.name in UNIFORMS]
    runtime_context = {
        "current_page": payload.page.model_dump(),
        "saved_enquiry_uniforms": enquiry_names,
        "session_context": payload.context.model_dump(),
        "draft_quote": _clean_quote_patch(payload.draft_quote),
        "quote": payload.quote if isinstance(payload.quote, dict) else {},
        "collection": payload.collection.model_dump(),
        "conversation_language": payload.conversation_language,
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
        client = Groq(api_key=GROQ_API_KEY, timeout=22.0, max_retries=1)
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.25,
                max_completion_tokens=900,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Some Groq models/accounts temporarily reject response_format even
            # when normal chat completion is healthy. Retry once without it and
            # parse the JSON object from the text ourselves.
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.25,
                max_completion_tokens=900,
            )
        content = completion.choices[0].message.content or ""
        data = _safe_json(content)
    except Exception:
        # Keep the storefront conversational even during a temporary provider
        # outage. The next visitor message will try Groq again automatically.
        return _natural_local_reply(payload, reason="provider_unavailable")

    reply = str(data.get("reply", "")).strip()[:3000]
    if not reply:
        reply = (
            "أقدر أساعدك تختار اليونيفورم المناسب ونكمّل تفاصيل الطلب بشكل واضح."
            if payload.conversation_language == "ar"
            else "I can help you choose the right ALF uniform and continue the enquiry naturally."
        )

    latest_user_text = payload.messages[-1].content if payload.messages else ""
    draft_quote = _merge_model_draft(payload.draft_quote, data.get("draft_quote"), latest_user_text)
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
    }
