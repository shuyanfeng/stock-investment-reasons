"""
stock-investment-reasons — Vercel Python serverless function
GET  /api/stock-investment-reasons?stock=<ticker or name>
POST /api/stock-investment-reasons  body: {"stock": "<ticker or name>"}

Uses OpenRouter (Perplexity Sonar) with live web search to return
the top 5 reasons to invest in a given stock as compact JSON.
Designed to respond within 10 s and fit within a 2,000-char response limit.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# perplexity/sonar: lightweight online model with live web search, faster than sonar-pro
MODEL = "perplexity/sonar"
# Hard timeout: stay under the 10 s caller limit
REQUEST_TIMEOUT = 8

SYSTEM_PROMPT = """\
You are a financial analyst. Given a stock ticker or company name, return ONLY a
valid JSON object — no markdown, no extra text.

Required structure (5 reasons, each explanation MAX 1 sentence with one data point):
{
  "ticker": "<UPPERCASE>",
  "company_name": "<Full Name>",
  "reasons": [
    {"rank": 1, "title": "<≤8 words>", "explanation": "<1 sentence, 1 data point>"},
    {"rank": 2, "title": "<≤8 words>", "explanation": "<1 sentence, 1 data point>"},
    {"rank": 3, "title": "<≤8 words>", "explanation": "<1 sentence, 1 data point>"},
    {"rank": 4, "title": "<≤8 words>", "explanation": "<1 sentence, 1 data point>"},
    {"rank": 5, "title": "<≤8 words>", "explanation": "<1 sentence, 1 data point>"}
  ]
}

Rules: exactly 5 reasons. No markdown fences. No text outside the JSON object."""


def call_openrouter(stock: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Stock: {stock}"},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://stock-investment-reasons-2mp2.vercel.app",
            "X-Title": "Stock Investment Reasons",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read())

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


def parse_stock(path: str, method: str, body: bytes) -> str:
    """Extract the 'stock' param from query string (GET) or JSON body (POST)."""
    if method == "POST":
        data = json.loads(body or b"{}")
        stock = data.get("stock", "")
    else:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
        stock = params.get("stock", [""])[0]
    return stock.strip()


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self._handle(body=b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._handle(body=body)

    def _handle(self, body: bytes):
        try:
            stock = parse_stock(self.path, self.command, body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "Invalid JSON body"})
            return

        if not stock:
            self._json(400, {"error": "Missing required parameter: stock"})
            return

        try:
            result = call_openrouter(stock)
            self._json(200, result)
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            self._json(502, {"error": "OpenRouter request failed", "detail": detail})
        except TimeoutError:
            self._json(504, {"error": "Request timed out"})
        except json.JSONDecodeError as e:
            self._json(502, {"error": "Failed to parse model response as JSON", "detail": str(e)})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _json(self, status: int, body: dict):
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass  # suppress default request logging
