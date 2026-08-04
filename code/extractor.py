"""extractor.py — manager email -> structured request fields.

Three interchangeable extractors, all returning the same dict:
    {is_request, item_id, store_id, event_name, requested_pct, urgency}

    extract_gemini(...)  Google Gemini      (pip install google-genai;  GEMINI_API_KEY)
    extract_claude(...)  Anthropic Claude   (pip install anthropic;     ANTHROPIC_API_KEY)
    extract_rules(...)   regex baseline, no key and no network

The LLM only translates language into fields. It never produces a quantity
decision — that is the engine's job.
"""
import json
import re

GEMINI_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-haiku-4-5"

EVENT_VOCAB = ["Thanksgiving", "SuperBowl", "IndependenceDay", "LaborDay",
               "Christmas", "Easter", "ValentinesDay", "MemorialDay",
               "NewYear", "Halloween", "MothersDay", "FathersDay"]


def build_system_prompt(products, event_vocab=EVENT_VOCAB):
    return f"""You extract structured data from store manager emails about inventory orders.

Allowed event names (return EXACTLY one of these strings, or null if no clear named event):
{json.dumps(event_vocab)}

Known products (display name -> ids). Emails may use the display name, a partial
name, or the raw item_id:
{json.dumps(products, indent=1)}

Return ONLY a JSON object with exactly these fields:
- "is_request": true only if the email asks to change an order quantity; complaints and FYIs are false
- "item_id": string or null
- "store_id": string or null (usually in the From line)
- "event_name": one of the allowed names, or null. Map informal references
  ("the big game" -> SuperBowl, "turkey week" -> Thanksgiving). Use null if no
  named event is identifiable, even when a reason for demand is given.
- "requested_pct": number or null. A fraction: "+50%" -> 0.5, "double our usual" -> 1.0,
  "half as much again" -> 0.5. If no explicit amount is stated ("a lot more",
  "stock up big"), use null. Never guess a number.
- "urgency": "low", "normal", or "high"

Examples:

Email: "From: Store Manager, CA_1\\nSubject: game day\\nBig game coming - take the Hillside trail mix tub up 40% for that week please. Deb"
{{"is_request": true, "item_id": "FOODS_3_252", "store_id": "CA_1", "event_name": "SuperBowl", "requested_pct": 0.4, "urgency": "normal"}}

Email: "From: Store Manager, CA_3\\nSubject: FYI\\nHeads up that the pancake mix pallets arrived damp. Photos attached. Not asking for order changes, just documenting. P."
{{"is_request": false, "item_id": null, "store_id": "CA_3", "event_name": null, "requested_pct": null, "urgency": "normal"}}"""


def _parse_json_reply(text):
    """Strip any markdown fences and parse the JSON body."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def _extract_text_from_message_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # common keys that may hold the text
        for k in ("text", "content", "message", "completion"):
            if k in content and isinstance(content[k], str):
                return content[k]
            if k in content and isinstance(content[k], dict) and "text" in content[k]:
                return content[k]["text"]
    if hasattr(content, "text"):
        return getattr(content, "text")
    if hasattr(content, "content"):
        return getattr(content, "content")
    return str(content)


def _anthropic_text_from_message(message):
    """Robustly extract text from various shapes of Anthropic client responses."""
    if message is None:
        return ""
    # string-like response
    if isinstance(message, str):
        return message
    # dict-like
    if isinstance(message, dict):
        # try a few common keys
        for key in ("content", "completion", "text", "output", "response"):
            if key in message:
                val = message[key]
                if isinstance(val, list) and val:
                    return _extract_text_from_message_content(val[0])
                return _extract_text_from_message_content(val)
    # object with attributes
    for attr in ("content", "completion", "text", "output", "response"):
        if hasattr(message, attr):
            val = getattr(message, attr)
            if isinstance(val, list) and val:
                return _extract_text_from_message_content(val[0])
            return _extract_text_from_message_content(val)
    # last resort
    return str(message)


def _normalise(fields):
    """Coerce the model's output into the exact schema the engine expects."""
    out = {
        "is_request": bool(fields.get("is_request", False)),
        "item_id": fields.get("item_id") or None,
        "store_id": fields.get("store_id") or None,
        "event_name": fields.get("event_name") or None,
        "requested_pct": fields.get("requested_pct"),
        "urgency": fields.get("urgency") or "normal",
    }
    if out["requested_pct"] is not None:
        pct = float(out["requested_pct"])
        # tolerate a model that answers 50 when it means 0.5
        out["requested_pct"] = pct / 100 if pct > 5 else pct
    return out


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

def extract_gemini(email_text, products, event_vocab=EVENT_VOCAB,
                   model=GEMINI_MODEL, client=None):
    # If a client is provided, avoid importing google.genai so tests or
    # environments without the package can still call this function.
    config = None
    if client is None:
        from google import genai
        from google.genai import types

        client = genai.Client()          # reads GEMINI_API_KEY
        config = types.GenerateContentConfig(
            system_instruction=build_system_prompt(products, event_vocab),
            response_mime_type="application/json",
            temperature=0,
        )

    response = client.models.generate_content(
        model=model,
        contents=email_text,
        config=config,
    )
    return _normalise(_parse_json_reply(response.text))


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------

def extract_claude(email_text, products, event_vocab=EVENT_VOCAB,
                   model=CLAUDE_MODEL, client=None):
    import anthropic

    client = client or anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    message = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0,
        system=build_system_prompt(products, event_vocab),
        messages=[{"role": "user", "content": email_text}],
    )
    text = _anthropic_text_from_message(message)
    try:
        parsed = _parse_json_reply(text)
    except Exception:
        raise ValueError(f"Failed to parse JSON from Anthropic response. Raw text:\n{text!r}")
    return _normalise(parsed)


# --------------------------------------------------------------------------
# rule-based baseline (no key, no network)
# --------------------------------------------------------------------------

_EVENT_KEYWORDS = [
    (r"thanksgiv|turkey week", "Thanksgiving"),
    (r"super\s*bowl|big game|game day", "SuperBowl"),
    (r"independence day|4th of july|july 4|fourth of july", "IndependenceDay"),
    (r"labor day", "LaborDay"),
    (r"christmas|xmas", "Christmas"),
    (r"easter", "Easter"),
    (r"halloween", "Halloween"),
    (r"valentine", "ValentinesDay"),
    (r"memorial day", "MemorialDay"),
    (r"new year", "NewYear"),
]
_NEGATION = r"not (?:changing|asking for|requesting)|no change(?:s)? to (?:any )?order"
_REQUEST_VERBS = r"\bincrease\b|\bbump\b|\braise\b|\bmore of\b|% more|\bup \d|\bstock up\b|\bdouble\b|\+\s*\d+\s*%"
_URGENT = r"\bURGENT\b|\basap\b|!!|\bNEED\b"


def _match_product(text, products):
    code = re.search(r"[A-Z]+_\d+_\d+", text)
    if code:
        for ids in products.values():
            if ids["item_id"] == code.group(0):
                return ids["item_id"], ids["store_id"]
        return code.group(0), None

    low = text.lower()
    best, best_hits = None, 0
    for name, ids in products.items():
        tokens = [t.rstrip("s") for t in name.lower().split() if len(t) >= 3]
        hits = sum(1 for t in tokens if t in low)
        if hits >= 2 and hits > best_hits:
            best, best_hits = ids, hits
    return (best["item_id"], best["store_id"]) if best else (None, None)


def extract_rules(email_text, products, event_vocab=EVENT_VOCAB):
    low = email_text.lower()
    store_match = re.search(r"[A-Z]{2}_\d", email_text)
    store = store_match.group(0) if store_match else None
    urgency = "high" if re.search(_URGENT, email_text) else "normal"

    if re.search(_NEGATION, low):
        return _normalise({"is_request": False, "store_id": store, "urgency": urgency})

    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    if pct_match:
        pct = float(pct_match.group(1)) / 100
    elif re.search(r"\bdouble\b", low):
        pct = 1.0
    else:
        pct = None

    event = next((name for pattern, name in _EVENT_KEYWORDS if re.search(pattern, low)), None)
    item, store_from_product = _match_product(email_text, products)

    return _normalise({
        "is_request": bool(pct is not None or re.search(_REQUEST_VERBS, low)),
        "item_id": item,
        "store_id": store or store_from_product,
        "event_name": event,
        "requested_pct": pct,
        "urgency": urgency,
    })


EXTRACTORS = {"gemini": extract_gemini, "claude": extract_claude, "rules": extract_rules}