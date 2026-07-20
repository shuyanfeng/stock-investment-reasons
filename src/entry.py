"""
stock-investment-reasons — Cloudflare Python Worker
GET  /?stock=<ticker or name>
POST /  body: {"stock": "<ticker or name>"}

Ported from the original Vercel Python serverless function
(api/stock-investment-reasons.py). Uses OpenRouter (Perplexity Sonar)
with live web search to return the top 5 reasons to invest in a given
stock as compact JSON.

Notes on the port:
- Cloudflare Workers has a single entry point (no Vercel-style
  api/*.py file routing), so this class is the whole route.
- urllib.request can't make real network calls inside the Workers
  sandbox, so the OpenRouter call goes through Python Workers' FFI
  bridge to the native JS fetch() — no pip packages required.
- OPENROUTER_API_KEY must be set as an encrypted secret (Cloudflare
  dashboard > Settings > Variables and secrets, or `wrangler secret put`),
  never committed to wrangler.jsonc or source.
"""

import json
import re
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

from workers import WorkerEntrypoint, Response
from js import fetch, Object
from pyodide.ffi import to_js

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# perplexity/sonar: lightweight online model with live web search
MODEL = "perplexity/sonar"

SYSTEM_PROMPT = """\
You are a financial analyst. Given a stock ticker or company name, return ONLY a
valid JSON object — no markdown, no extra text.

Required structure (5 reasons, each explanation MAX 1 sentence with one data point):
{
  "ticker": "<UPPERCASE>",
  "company_name": "<Full Name>",
  "reasons": [
    {"rank": 1, "title": "<\u22648 words>", "explanation": "<1 sentence, 1 data point>"},
    {"rank": 2, "title": "<\u22648 words>", "explanation": "<1 sentence, 1 data point>"},
    {"rank": 3, "title": "<\u22648 words>", "explanation": "<1 sentence, 1 data point>"},
    {"rank": 4, "title": "<\u22648 words>", "explanation": "<1 sentence, 1 data point>"},
    {"rank": 5, "title": "<\u22648 words>", "explanation": "<1 sentence, 1 data point>"}
  ]
}

Rules: exactly 5 reasons. No markdown fences. No text outside the JSON object."""


class OpenRouterError(Exception):
    """Raised when the OpenRouter HTTP call itself fails (non-2xx response)."""
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _to_js_obj(d: dict):
    """Convert a (possibly nested) Python dict into a JS object for fetch()."""
    return to_js(d, dict_converter=Object.fromEntries)


def _json_response(status: int, body: dict) -> Response:
    payload = json.dumps(body, separators=(",", ":"))
    return Response(
        payload,
        status=status,
        headers={"Content-Type": "application/json"},
    )


async def call_openrouter(stock: str, api_key: str) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Stock: {stock}"},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    })

    options = _to_js_obj({
        "method": "POST",
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://xinfinite.io",
            "X-Title": "Stock Investment Reasons",
        },
        "body": payload,
    })

    js_resp = await fetch(OPENROUTER_URL, options)

    if not js_resp.ok:
        detail = await js_resp.text()
        raise OpenRouterError(js_resp.status, detail)

    raw_data = await js_resp.json()
    data = raw_data.to_py() if hasattr(raw_data, "to_py") else raw_data

    raw = data["choices"][0]["message"]["content"].strip()
    # Strip markdown code fences if the model wraps its output
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)
    if "reasons" not in result or len(result["reasons"]) == 0:
        raise ValueError("Model response missing 'reasons' field")

    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["disclaimer"] = "Not financial advice. Do your own research."
    return result


async def _parse_stock(request) -> str:
    """Extract the 'stock' param from query string (GET) or JSON body (POST)."""
    if request.method == "POST":
        raw_body = await request.json()
        data = raw_body.to_py() if hasattr(raw_body, "to_py") else raw_body
        return str(data.get("stock", "")).strip()
    else:
        parsed = urlparse(request.url)
        params = parse_qs(parsed.query)
        return params.get("stock", [""])[0].strip()


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        try:
            stock = await _parse_stock(request)
        except (json.JSONDecodeError, ValueError):
            return _json_response(400, {"error": "Invalid JSON body"})

        if not stock:
            return _json_response(400, {"error": "Missing required parameter: stock"})

        api_key = getattr(self.env, "OPENROUTER_API_KEY", None)
        if not api_key:
            return _json_response(500, {"error": "OPENROUTER_API_KEY is not configured"})

        try:
            result = await call_openrouter(stock, str(api_key))
            return _json_response(200, result)
        except OpenRouterError as e:
            return _json_response(502, {"error": "OpenRouter request failed", "detail": e.detail})
        except ValueError as e:
            return _json_response(400, {"error": str(e)})
        except json.JSONDecodeError as e:
            return _json_response(502, {"error": "Failed to parse model response as JSON", "detail": str(e)})
        except Exception as e:
            return _json_response(500, {"error": str(e)})
