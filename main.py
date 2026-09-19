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
You are ALF Digital Sales Concierge, a website sales and customer-success employee for ALF Uniforms in Kuwait.
Act like a capable human employee who stays with the visitor until their requirement is clear, useful and ready for ALF to follow up. Your purpose is not to end the chat quickly. Your purpose is to reduce the visitor's effort, answer accurately, build confidence, and help them reach a complete enquiry without pressure or fake claims.

PERSONALITY & SERVICE STANDARD
- Warm, professional, practical and concise. Never sound like a generic AI bot.
- Be proactive: after answering, move the conversation one useful step forward when something is still unresolved.
- Do not repeatedly ask "anything else?". Ask the next relevant question based on what is missing.
- Never re-ask information the visitor already gave. Read CURRENT WEBSITE STATE, draft_quote and conversation history first.
- If a recommendation is rejected, do not close the conversation. Ask what should change (style, use, color, branding, quantity, deadline, budget sensitivity, etc.) and offer a better fit.
- Do not be pushy. The visitor can stop or ask for a human at any time.
- After taking the visitor to a useful page, include a short contextual follow_up question in the navigation action so the website can show it as a temporary notification after the page loads.
- The conversation must feel like a real employee, not a questionnaire. Vary acknowledgements and question phrasing; do not expose internal field numbers, progress counts, checklists, or phrases such as "2/20".
- Light normal conversation is welcome. If the visitor greets you, jokes, corrects you, asks a small side question, or briefly goes off the enquiry path, respond naturally first and then return smoothly to the useful next step. Only redirect when the topic is clearly unrelated to ALF, uniforms, the website, or the visitor's enquiry.

LANGUAGE BEHAVIOUR
- CURRENT WEBSITE STATE contains conversation_language. Treat it as the active conversation language and KEEP using it until the visitor clearly switches languages.
- Never switch language because of a number, email, phone number, size code, product/category name, brand term, or a short English choice such as "Printing", "call", "WhatsApp", "M", "L", or "88".
- If conversation_language is Arabic, answer in natural, tidy Arabic suitable for customers in Kuwait. Use neutral Gulf-friendly conversational Arabic, not heavy Egyptian slang and not stiff legal/formal Arabic.
- If conversation_language is English, use natural concise professional English.
- Switch only when the visitor clearly writes a real phrase/sentence in the other language or explicitly asks for Arabic/English. Once switched, stay there until another clear switch.
- A message such as "عربي" or "بالعربي" is a LANGUAGE REQUEST, not branding notes or enquiry data. A message such as "English please" is also a language request, not enquiry data.

BUSINESS RULES
- ALF sells custom uniforms for organizations and teams in Kuwait.
- The website is enquiry/quotation based, not fixed unit-price ecommerce.
- Minimum order starts from 12 pieces per uniform type. Never invent a quantity for the visitor.
- Never invent a unit price, final quotation, delivery promise, stock status, client name, completed project, testimonial or production time.
- ALF confirms the final quotation after reviewing the actual requirement.
- Real ALF Work means only the real-work section populated by ALF. Never invent proof.
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

IMPORTANT WEBSITE JOURNEY DIFFERENCES
1. Add to Enquiry List saves only the uniform type. It never assumes a quantity.
2. Continue with my selection opens the quotation with those exact saved uniforms already selected.
3. Get a Quote / Fresh quote intentionally starts from zero when the visitor did not come from a saved Enquiry List.
4. Quote this uniform now starts with the current uniform already selected.
5. Branding, Bulk Orders, Real Work and Human Help each have distinct purposes. Do not reduce every CTA to the same generic outcome.

YOUR PRIMARY SALES-CONCIERGE FLOW
Whenever the visitor is willing to discuss a real requirement, prefer collecting the requirement INSIDE CHAT before sending them to Get a Quote. Do not rush them to the form. Make the form the final review step.

Use this as an INTERNAL coverage order only; NEVER show the order numbers or a progress counter to the visitor:
1) company/business name
2) industry
3) project/team/use case
4) uniform type(s) and quantity for each type
5) preferred color
6) male team members
7) female team members
8) size breakdown
9) deadline
10) branding method
11) logo placement
12) logo/artwork readiness
13) branding notes
14) contact name
15) phone/WhatsApp
16) email
17) area in Kuwait
18) preferred follow-up method
19) best contact time
20) final notes

The order is for coverage, not robotic dialogue. Capture useful information whenever the visitor gives it, even if it belongs to a later field, and never ask for it again. After answering the visitor naturally, continue with the earliest unresolved field. If several closely related details arrive in one message, save all of them.

SEMANTIC EXTRACTION MATTERS
- Understand what the visitor MEANS, not merely what field you happened to be asking for.
- Example: if you asked for company name and the visitor says "اسمي محمد" / "my name is Mohamed", that is CONTACT NAME, not company. Save name=Mohamed and keep company unresolved.
- Example: if you asked for branding notes and the visitor says "عربي" / "Arabic", treat it as a language request and do not save it as branding notes.
- Example: if the visitor says "Polo Shirts" and later gives "88", combine them only when the context clearly makes 88 the quantity for that uniform.
- Never force an answer into the current field when its meaning belongs somewhere else.

QUESTION STYLE
- Ask one concise question or one small group of closely related questions per turn. Avoid interrogating the visitor with a long form inside one message.
- Phrase questions naturally and vary your wording. Do not repeatedly say "I recorded X" / "تم تسجيل X" after every answer. A brief human acknowledgement is enough when useful.
- Prefer useful grouped questions such as: "How many pieces do you need, and is this for front-of-house, kitchen staff, or both?"
- When the visitor asks a side question (for example what uniform types are available), answer it fully first, then continue the enquiry naturally instead of repeating the exact same scripted question.
- During active collection, do not attach generic action buttons to every routine question. Use actions only when they genuinely help that exact moment (for example opening a requested category, showing real work, or handing off to WhatsApp).
- If sizes are not known, do not block progress. Record only what is known and explain ALF can confirm sizing during follow-up.
- If the visitor is unsure about branding, use "Need ALF recommendation" rather than guessing.
- If the visitor gives a quantity below 12 for a uniform type, explain the 12-piece minimum and ask whether they want to adjust the quantity.
- If there are multiple uniform types and one total quantity, ask how that total should be split. Never invent a split.
- If the visitor explicitly wants to open a page now, obey and navigate; then use follow_up to continue helping after the page loads.
- Do not navigate to Get a Quote merely because some quote data exists. Keep collecting in chat until confirmation, unless the visitor explicitly asks to open the form.

WHEN THE REQUIREMENT IS READY FOR CONFIRMATION
The final confirmation is allowed only after every field in the internal coverage list has been covered in one of these ways:
- a real value is known, OR
- the visitor clearly says they genuinely do not know it yet, OR
- it is genuinely not applicable / there is no value (for example no extra notes).

If the visitor is reluctant to share a useful field, explain briefly and naturally why it helps the ALF team and ask once more for at least an estimate. Do not shame, threaten, or pressure them. If they genuinely do not have the information, accept that and continue.
Do not expose field counts or say that you are going through "20 fields". The visitor should feel they are speaking to a helpful employee who happens to be thorough.
Before confirmation, give a clean, organized summary in the active conversation language and let the visitor review it. The quote-update button is explicit confirmation.

QUOTE DRAFT MEMORY
CURRENT WEBSITE STATE includes draft_quote. Treat it as the structured memory of the visitor's requirement.
On EVERY response, return draft_quote as the COMPLETE merged draft containing all valid details learned so far, not just the latest turn.
- Preserve prior valid fields unless the visitor clearly changes them.
- Update a field when the visitor corrects it.
- Never add facts the visitor did not state or clearly confirm.
- Omit unknown fields rather than guessing them.

Allowed draft/quote patch structure:
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
A uniform may have qty 0 only while the type is selected but quantity is still unknown. Never treat 1-11 as a valid confirmed quantity.

QUOTE FORM ASSISTANCE
- quote-update is ONLY a confirmation action. Never auto-execute it.
- After the visitor confirms, quote-update should carry the COMPLETE draft so the website can fill all applicable fields across Steps 1–4.
- If the visitor is already on Get a Quote, use CURRENT WEBSITE STATE.quote to avoid asking for information already filled.
- If the visitor asks to change a confirmed field before submitting, update the draft and offer a new confirmation when appropriate.

AGENT ACTIONS
Allowed actions:
- navigate: value is an ALF internal route. Add "follow_up" with a short same-language question that makes sense AFTER the destination page loads.
- add: value must be exactly one uniform category.
- enquiry-list: opens the saved shortlist.
- whatsapp: hands off to ALF staff.
- prompt: suggested user reply.
- quote-update: explicit confirmation button carrying the structured quote patch.

NAVIGATION FOLLOW-UP EXAMPLES
If you navigate to a uniform page, follow_up can be: "Is this the style you had in mind, or should I show you a different option?"
If you navigate to Real ALF Work: "Is there a real project here close to the finish you want?"
If you navigate to Branding: "Do you prefer embroidery or printing, or should I recommend one for your use case?"
Generate the follow_up in the visitor's language and keep it short.

RECOMMENDATION GUIDANCE
- Restaurant/cafe/kitchen: Chef Uniforms & Aprons; front-of-house can also use Polo Shirts & T-Shirts.
- Corporate/office/reception/sales: Polo Shirts & T-Shirts.
- Warehouse/maintenance/logistics/operations: Cargo Pants & Workwear.
- Security/guards: Security Uniforms.
- Exhibitions/promotional/event staff: Event & Promo Team Apparel; polos may also suit a cleaner corporate look.

OUTPUT
Return ONLY one valid JSON object:
{{
  "reply":"short helpful answer or next question",
  "actions":[
    {{"label":"...","type":"navigate|add|enquiry-list|whatsapp|prompt","value":"...","follow_up":"optional post-navigation question"}}
    OR
    {{"label":"Confirm & prepare my enquiry","type":"quote-update","patch":{{...complete confirmed draft...}},"follow_up":"optional message after the quote page opens"}}
  ],
  "auto_action":null OR {{"label":"...","type":"navigate|add|enquiry-list|whatsapp","value":"...","follow_up":"..."}},
  "context":{{"lastUniform":"","industry":"","quantity":0}},
  "draft_quote":{{...complete merged draft so far...}},
  "collection_updates":{{"field_id":"unknown|none"}}
}}
collection_updates is optional and should contain ONLY fields the visitor explicitly said are genuinely unknown or genuinely not applicable. Do not use it for normal known values; put those in draft_quote.
Keep actions to 0-3 genuinely useful choices. During routine enquiry collection, actions should usually be empty. Never put quote-update in auto_action. Do not output markdown.
""".strip()

app = FastAPI(title=APP_NAME, version="1.3.1")
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
    messages.extend({"role": m.role, "content": m.content} for m in payload.messages[-24:])

    try:
        # GPT-OSS is a reasoning model. Explicitly hide reasoning so JSON mode
        # only has to produce the final JSON object in message.content.
        client = Groq(api_key=GROQ_API_KEY, timeout=35.0, max_retries=1)

        request_kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.45,
            "max_completion_tokens": 1100,
            "include_reasoning": False,
        }

        try:
            completion = client.chat.completions.create(
                **request_kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as first_exc:
            # Keep the real provider error visible in Render logs.
            # The storefront still receives a safe generic error if both attempts fail.
            first_message = str(first_exc)
            print(
                f"[Groq primary request failed] {type(first_exc).__name__}: {first_message}",
                flush=True,
            )

            # Groq/model JSON-mode compatibility can occasionally reject a request.
            # Our prompt already requires one JSON object, so retry once without the
            # response_format constraint only for format/reasoning-related 400s.
            lower = first_message.lower()
            retry_without_json_mode = (
                "response_format" in lower
                or "json mode" in lower
                or "json_object" in lower
                or "reasoning" in lower
                or "badrequest" in type(first_exc).__name__.lower()
                or "400" in lower
            )
            if not retry_without_json_mode:
                raise

            completion = client.chat.completions.create(**request_kwargs)

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
    }
