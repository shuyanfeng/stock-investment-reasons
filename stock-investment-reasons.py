"""
stock-investment-reasons — Vercel Python serverless function
GET /api/stock-investment-reasons?stock=<ticker or company name>

Uses OpenRouter (Perplexity Sonar Pro) with live web search to return
the top 5 reasons to invest in a given stock as structured JSON.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Perplexity Sonar Pro: online model with built-in web search
MODEL = "perplexity/sonar-pro"

SYSTEM_PROMPT = """\
You are a financial research analyst. When given a stock ticker or company name,
search the web for the latest information and return ONLY a valid JSON object —
no markdown, no commentary, no extra text.

Required JSON structure:
{
  "ticker": "<uppercase ticker symbol>",
  "company_name": "<full legal company name>",
  "reasons": [
    {
      "rank": 1,
      "title": "<short, specific title>",
      "explanation": "<2–3 sentences with concrete data points, figures, or trends>"
    }
  ]
}

Rules:
- Exactly 5 reasons, ranked 1–5.
- Each explanation must cite at least one specific figure (revenue, growth %, P/E, etc.).
- If the input is ambiguous, resolve to the most commonly traded public company.
- Return ONLY the JSON object. No markdown fences. No preamble."""


def call_openrouter(stock: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Stock to analyze: {stock}"},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://stock-investment-reasons.vercel.app",
            "X-Title": "Stock Investment Reasons",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read())

    raw = data["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if the model wraps its output
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    # Validate minimum shape
    if "reasons" not in result or len(result["reasons"]) == 0:
        raise ValueError("LLM response missing 'reasons' field")

    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["disclaimer"] = (
        "For informational purposes only. Not financial advice. "
        "Always conduct your own due diligence."
    )
    return result


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        stock = params.get("stock", [None])[0]

        if not stock or not stock.strip():
            self._json(400, {"error": "Missing required query parameter: stock"})
            return

        try:
            result = call_openrouter(stock.strip())
            self._json(200, result)
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            self._json(502, {"error": "OpenRouter request failed", "detail": body})
        except json.JSONDecodeError as e:
            self._json(502, {"error": "Failed to parse model response as JSON", "detail": str(e)})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def do_OPTIONS(self):
        # CORS preflight
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _json(self, status: int, body: dict):
        payload = json.dumps(body, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        pass  # suppress default request logging
