#!/usr/bin/env python3
"""
CATALYST — macro_agent.py v3.0
================================
מערכת מודיעין קטליסטית למניות.
סורקת רשימה קבועה של טיקרים ומנתחת:
  1. דוח רבעוני אחרון + אירועים עסקיים מהותיים
  2. קטליזטורים צפויים — רגולטוריים, מאקרו, עסקיים, סקטוריאליים
  3. שאלות ממוקדות שמשקפות עלייה/ירידה צפויה

Usage:
  python macro_agent.py                    # Run all tickers
  python macro_agent.py --ticker ASML,PLTR # Run specific tickers
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Tuple, Dict, List

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================
# LOGGING
# ==============================================================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"catalyst_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("CATALYST")

# ==============================================================
# GEMINI SDK (new → legacy fallback)
# ==============================================================
try:
    from google import genai as genai_sdk
    from google.genai import types as genai_types
    GENAI_NEW_SDK = True
    log.info("Using google.genai (new SDK)")
except ImportError:
    try:
        import google.generativeai as genai_sdk
        GENAI_NEW_SDK = False
        log.info("Using google.generativeai (legacy SDK)")
    except ImportError:
        log.critical("No Gemini SDK found. Install: pip install google-genai")
        sys.exit(1)

import yfinance as yf


# ==============================================================
# FIXED TICKER LIST
# ==============================================================
WATCHLIST = [
    "ASML", "PLTR", "MELI", "URNM", "FCX",
    "ISRG", "PWR", "XYL", "STNG", "SMH",
    "LNG", "SOXX", "QQQ", "SPY", "MU",
]


# ==============================================================
# PYDANTIC MODELS
# ==============================================================
class CatalystItem(BaseModel):
    """Single catalyst event."""
    type: str = Field(description="סוג: earnings / regulatory / macro / business / sector")
    title: str = Field(description="כותרת קצרה בעברית — 3-6 מילים")
    description: str = Field(description="תיאור בעברית — 2-3 משפטים")
    impact: str = Field(description="חיובי / שלילי / לא ברור")
    timeframe: str = Field(description="מסגרת זמן: מיידי / ימים / שבועות / חודשים")

class QuestionAnswer(BaseModel):
    """Targeted Q&A about the stock."""
    question: str = Field(description="שאלה ממוקדת בעברית")
    answer: str = Field(description="תשובה בעברית — 2-4 משפטים")
    implication: str = Field(description="חיובי / שלילי / ניטרלי")

class StockAnalysis(BaseModel):
    """Complete analysis for a single stock."""
    ticker: str
    company_name: str = Field(description="שם החברה בעברית")
    sector: str = Field(description="סקטור בעברית")
    signal: str = Field(description="סיגנל פעולה: 2-3 מילים בעברית")
    direction: str = Field(description="bullish / bearish / neutral")
    confidence: int = Field(ge=0, le=100, description="רמת ביטחון 0-100")
    earnings_insight: str = Field(description="ניתוח דוח רבעוני אחרון — 3-5 משפטים בעברית")
    bottom_line: str = Field(description="שורה תחתונה — משפט אחד חד")
    catalysts: List[CatalystItem] = Field(min_length=1, max_length=5)
    questions: List[QuestionAnswer] = Field(min_length=2, max_length=3)

class BatchResult(BaseModel):
    """Result for a batch of stocks."""
    stocks: List[StockAnalysis]


# ==============================================================
# CONFIG
# ==============================================================
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"
    OUTPUT_FILE = Path(__file__).parent.parent / "docs" / "master_data.json"
    HISTORY_DIR = Path(__file__).parent.parent / "docs" / "history"
    CONFIG_FILE = Path(__file__).parent.parent / "docs" / "config.json"
    MAX_RETRIES = 5
    RETRY_DELAY = 5
    RATE_LIMIT_DELAY = 65
    BATCH_SIZE = 5  # tickers per Gemini call


# ==============================================================
# DATA FETCHER
# ==============================================================
class DataFetcher:
    """Fetches market data for the watchlist."""

    def fetch_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch comprehensive data for all tickers."""
        log.info(f"Fetching data for {len(tickers)} tickers...")
        results = {}

        # Quick price data via batch download
        try:
            data = yf.download(tickers, period="5d", group_by="ticker", threads=True, progress=False)
        except Exception as e:
            log.error(f"Batch download failed: {e}")
            data = None

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}

                # Price from batch data
                price = 0
                change_pct = 0
                if data is not None:
                    try:
                        if len(tickers) == 1:
                            df = data
                        else:
                            df = data[ticker] if ticker in data.columns.get_level_values(0) else None
                        if df is not None and not df.empty:
                            df = df.dropna()
                            if len(df) >= 2:
                                price = round(float(df["Close"].iloc[-1]), 2)
                                prev = float(df["Close"].iloc[-2])
                                change_pct = round(((price - prev) / prev) * 100, 2)
                    except Exception:
                        pass

                if price == 0:
                    price = info.get("currentPrice", info.get("regularMarketPrice", 0))

                result = {
                    "ticker": ticker,
                    "company_name": info.get("longName", info.get("shortName", ticker)),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "price": price,
                    "change_pct": change_pct,
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", None),
                    "forward_pe": info.get("forwardPE", None),
                    "dividend_yield": info.get("dividendYield", None),
                    "52w_high": info.get("fiftyTwoWeekHigh", 0),
                    "52w_low": info.get("fiftyTwoWeekLow", 0),
                    "target_price": info.get("targetMeanPrice", 0),
                    "revenue": info.get("totalRevenue", None),
                    "profit_margin": info.get("profitMargins", None),
                    "free_cash_flow": info.get("freeCashflow", None),
                    "debt_to_equity": info.get("debtToEquity", None),
                    "description": info.get("longBusinessSummary", ""),
                    "earnings_date": "Unknown",
                    "analyst_ratings": {},
                }

                # Analyst ratings
                try:
                    rec_summary = stock.recommendations_summary
                    if rec_summary is not None:
                        result["analyst_ratings"] = {
                            "strongBuy": int(rec_summary.get("strongBuy", 0)),
                            "buy": int(rec_summary.get("buy", 0)),
                            "hold": int(rec_summary.get("hold", 0)),
                            "sell": int(rec_summary.get("sell", 0)),
                            "strongSell": int(rec_summary.get("strongSell", 0)),
                        }
                except Exception:
                    pass

                # Earnings date
                try:
                    cal = stock.calendar
                    if cal and isinstance(cal, dict) and "Earnings Date" in cal:
                        result["earnings_date"] = str(cal["Earnings Date"][0])
                except Exception:
                    pass

                results[ticker] = result
                log.info(f"  ✓ {ticker}: ${price} ({change_pct:+.2f}%)")

            except Exception as e:
                log.warning(f"  ✗ {ticker} fetch failed: {e}")
                results[ticker] = {"ticker": ticker, "error": str(e)}

            time.sleep(0.2)

        return results


# ==============================================================
# CATALYST AI ENGINE
# ==============================================================
class CatalystEngine:
    """Gemini-powered catalyst analysis."""

    def __init__(self):
        if not Config.GEMINI_API_KEY:
            log.critical("GEMINI_API_KEY not set!")
            sys.exit(1)
        if GENAI_NEW_SDK:
            self.client = genai_sdk.Client(api_key=Config.GEMINI_API_KEY)
        else:
            genai_sdk.configure(api_key=Config.GEMINI_API_KEY)
            self.client = None

    def call_gemini(self, prompt: str, temperature: float = 0.7) -> str:
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                if GENAI_NEW_SDK:
                    response = self.client.models.generate_content(
                        model=Config.GEMINI_MODEL,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=16000,
                        ),
                    )
                    return response.text or ""
                else:
                    model = genai_sdk.GenerativeModel(Config.GEMINI_MODEL)
                    response = model.generate_content(
                        prompt,
                        generation_config={"temperature": temperature, "max_output_tokens": 16000},
                    )
                    return response.text or ""
            except Exception as e:
                err_str = str(e)
                log.warning(f"Gemini attempt {attempt}/{Config.MAX_RETRIES}: {err_str[:200]}")
                if attempt < Config.MAX_RETRIES:
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait = Config.RATE_LIMIT_DELAY
                        log.info(f"Rate limited — waiting {wait}s...")
                    else:
                        wait = Config.RETRY_DELAY * attempt
                    time.sleep(wait)
                else:
                    raise

    def build_analysis_prompt(self, stocks_data: List[Dict]) -> str:
        data_str = json.dumps(stocks_data, ensure_ascii=False, default=str)

        return f"""# תפקיד: אנליסט מודיעין שוק ההון — מערכת CATALYST

אתה אנליסט מודיעין שוק ההון. התפקיד שלך: לזהות **קטליזטורים** — אירועים, נתונים, או שינויים שצפויים להזיז מניות בטווח הקרוב.

## עקרונות ברזל:
1. **קטליזטור = אירוע שמשנה מחיר.** לא סקירה כללית. לא "החברה טובה". רק: מה הולך לקרות, ולמה זה ישפיע.
2. **דוח רבעוני אחרון**: מה הודיעה ההנהלה? מה הפתיע? מה השתנה בתוכנית העסקית? (למשל: הפסקת השקעה בתשתיות, שינוי guidance, רכישה/מכירה, קיצוץ עובדים)
3. **קטליזטורים קדימה**: רגולציה, מאקרו (ריבית, מכסים, גיאו-פוליטיקה), עסקי (חוזים, שותפויות, מוצרים), סקטוריאלי.
4. **שאלות ממוקדות**: לכל מניה 2-3 שאלות שמשקיע צריך לשאול את עצמו עכשיו, עם תשובות מבוססות.
5. **אם אין קטליזטור ברור — אמור את זה.** עדיף "אין קטליזטור ברור" מאשר לייצר אחד מלאכותי.

## נתוני שוק:
{data_str}

## המשימה:
לכל מניה, ייצר ניתוח מלא. **כל הטקסט בעברית. מפתחות JSON באנגלית.**

## פורמט פלט — JSON בלבד:
```json
{{
  "stocks": [
    {{
      "ticker": "XXX",
      "company_name": "שם בעברית",
      "sector": "סקטור בעברית",
      "signal": "סיגנל קצר 2-3 מילים",
      "direction": "bullish/bearish/neutral",
      "confidence": 75,
      "earnings_insight": "ניתוח דוח רבעוני אחרון — 3-5 משפטים. מה הפתיע? מה השתנה? מה ההנהלה אמרה?",
      "bottom_line": "שורה תחתונה אחת — למה לשים לב עכשיו",
      "catalysts": [
        {{
          "type": "earnings/regulatory/macro/business/sector",
          "title": "כותרת קצרה",
          "description": "2-3 משפטים",
          "impact": "חיובי/שלילי/לא ברור",
          "timeframe": "מיידי/ימים/שבועות/חודשים"
        }}
      ],
      "questions": [
        {{
          "question": "שאלה ממוקדת שמשקיע צריך לשאול",
          "answer": "תשובה מבוססת — 2-4 משפטים",
          "implication": "חיובי/שלילי/ניטרלי"
        }}
      ]
    }}
  ]
}}
```

## כללים:
- signal: 2-3 מילים שמסכמות את המצב (למשל: "מומנטום חזק", "סיכון רגולטורי", "הזדמנות בתיקון", "ללא קטליזטור ברור")
- direction: bullish אם הקטליזטורים חיוביים, bearish אם שליליים, neutral אם מעורב
- confidence: 0-100 — כמה בטוח אתה בכיוון
- catalysts: 1-5 קטליזטורים. סוגים: earnings, regulatory, macro, business, sector
- questions: 2-3 שאלות. כל שאלה עם implication: חיובי/שלילי/ניטרלי
- ETFs (SMH, SOXX, QQQ, SPY): נתח ברמת האינדקס — מה מניע את הסקטור/שוק

**חשוב: החזר JSON בלבד. ללא טקסט נוסף.**"""


# ==============================================================
# JSON EXTRACTION (robust — handles various Gemini output quirks)
# ==============================================================
def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from Gemini response. Handles:
    - Clean JSON
    - Markdown code fences
    - Partial responses (array without wrapper)
    - Single objects without array
    """
    if not text or not text.strip():
        return None

    # Step 1: Strip markdown fences
    cleaned = text.strip()
    for fence in ["```json", "```JSON", "```"]:
        if fence in cleaned:
            parts = cleaned.split(fence, 1)
            if len(parts) > 1:
                cleaned = parts[1]
                break
    if "```" in cleaned:
        cleaned = cleaned.split("```")[0]
    cleaned = cleaned.strip()

    # Step 2: Try direct parse
    try:
        result = json.loads(cleaned)
        return _normalize_stocks_result(result)
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 3: Try wrapping in {"stocks": [...]} if it looks like an array
    if cleaned.startswith("["):
        try:
            arr = json.loads(cleaned)
            return {"stocks": arr}
        except (json.JSONDecodeError, ValueError):
            pass

    # Step 4: Try wrapping bare object in array — {"ticker":...} → {"stocks": [{...}]}
    if cleaned.startswith("{") and '"ticker"' in cleaned[:100]:
        try:
            # Might be a single stock object
            obj = json.loads(cleaned)
            if "ticker" in obj:
                return {"stocks": [obj]}
        except (json.JSONDecodeError, ValueError):
            pass

    # Step 5: Find the outermost JSON structure (object or array)
    # Find first { or [
    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")

    # Try array first if it appears before object (likely stocks array)
    if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        arr_end = _find_matching_bracket(cleaned, arr_start, "[", "]")
        if arr_end > arr_start:
            try:
                arr = json.loads(cleaned[arr_start:arr_end + 1])
                if isinstance(arr, list):
                    return {"stocks": arr}
            except (json.JSONDecodeError, ValueError):
                pass

    # Try object
    if obj_start != -1:
        obj_end = _find_matching_bracket(cleaned, obj_start, "{", "}")
        if obj_end > obj_start:
            try:
                result = json.loads(cleaned[obj_start:obj_end + 1])
                return _normalize_stocks_result(result)
            except (json.JSONDecodeError, ValueError):
                pass

    # Step 6: Last resort — find ALL top-level JSON objects and collect them
    objects = _extract_all_objects(cleaned)
    if objects:
        stocks = []
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and "ticker" in obj:
                    stocks.append(obj)
                elif isinstance(obj, dict) and "stocks" in obj:
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue
        if stocks:
            return {"stocks": stocks}

    log.error(f"JSON extraction failed. Preview: {cleaned[:300]}")
    return None


def _normalize_stocks_result(result) -> Optional[Dict]:
    """Ensure result has the expected {"stocks": [...]} structure."""
    if isinstance(result, dict):
        if "stocks" in result:
            return result
        if "ticker" in result:
            return {"stocks": [result]}
        # Maybe top_picks from old format?
        if "top_picks" in result:
            return {"stocks": result["top_picks"]}
    if isinstance(result, list):
        return {"stocks": result}
    return result


def _find_matching_bracket(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Find the matching closing bracket."""
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_all_objects(text: str) -> List[str]:
    """Extract all top-level {...} objects from text."""
    objects = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            end = _find_matching_bracket(text, i, "{", "}")
            if end > i:
                objects.append(text[i:end + 1])
                i = end + 1
                continue
        i += 1
    return objects


# ==============================================================
# PYDANTIC VALIDATION
# ==============================================================
def validate_stock(raw: Dict) -> Dict:
    """Validate a single stock analysis against the Pydantic model."""
    try:
        obj = StockAnalysis(**raw)
        return obj.model_dump()
    except Exception as e:
        log.warning(f"  Validation issue for {raw.get('ticker', '?')}: {e}")
        return raw  # Return raw data, still usable


# ==============================================================
# MARKET STATUS
# ==============================================================
def get_market_status() -> str:
    now_et = datetime.now(timezone(timedelta(hours=-4)))
    weekday = now_et.weekday()
    hour = now_et.hour
    minute = now_et.minute

    if weekday >= 5:
        return "weekend"
    current_minutes = hour * 60 + minute
    if current_minutes < 4 * 60:
        return "closed"
    elif current_minutes < 9 * 60 + 30:
        return "pre_market"
    elif current_minutes < 16 * 60:
        return "open"
    elif current_minutes < 20 * 60:
        return "after_hours"
    else:
        return "closed"


# ==============================================================
# MAIN PIPELINE
# ==============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CATALYST — Stock Catalyst Intelligence")
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        default="",
        help="Run specific ticker(s). Comma-separated. Example: ASML,PLTR"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which tickers to run
    if args.ticker:
        run_tickers = [t.strip().upper() for t in args.ticker.split(",")]
        partial_run = True
    else:
        run_tickers = list(WATCHLIST)
        partial_run = False

    log.info("=" * 60)
    log.info("CATALYST v3.0 — Stock Catalyst Intelligence")
    if partial_run:
        log.info(f"PARTIAL RUN — tickers: {', '.join(run_tickers)}")
    else:
        log.info(f"FULL RUN — {len(run_tickers)} tickers")
    log.info("=" * 60)
    start_time = time.time()

    # ─── Phase 1: DATA FETCH ───
    log.info("PHASE 1: DATA FETCH")
    fetcher = DataFetcher()
    stock_data = fetcher.fetch_batch(run_tickers)

    valid_data = {k: v for k, v in stock_data.items() if "error" not in v}
    log.info(f"Data fetched: {len(valid_data)}/{len(run_tickers)} tickers")

    if not valid_data:
        log.error("No valid data. Aborting.")
        sys.exit(1)

    # ─── Phase 2: AI ANALYSIS (batched) ───
    log.info("PHASE 2: AI CATALYST ANALYSIS")
    engine = CatalystEngine()

    all_analyses = []
    tickers_list = list(valid_data.keys())
    batch_size = Config.BATCH_SIZE

    for i in range(0, len(tickers_list), batch_size):
        batch_tickers = tickers_list[i:i + batch_size]
        batch_data = [valid_data[t] for t in batch_tickers]

        log.info(f"  ── Batch {i // batch_size + 1}: {', '.join(batch_tickers)} ──")

        prompt = engine.build_analysis_prompt(batch_data)
        response = engine.call_gemini(prompt, temperature=0.6)
        result = extract_json(response)

        batch_results = {}
        if result and "stocks" in result:
            for stock_raw in result["stocks"]:
                validated = validate_stock(stock_raw)
                ticker = validated.get("ticker", "")
                if ticker in valid_data:
                    validated["price"] = valid_data[ticker].get("price", 0)
                    validated["change_pct"] = valid_data[ticker].get("change_pct", 0)
                batch_results[ticker] = validated
                log.info(f"    ✓ {ticker}: {validated.get('signal', '?')} ({validated.get('direction', '?')})")
        else:
            log.error(f"  ✗ Failed to parse batch. Raw: {response[:300]}")

        # Check for missing tickers — retry individually
        missing = [t for t in batch_tickers if t not in batch_results]
        if missing:
            log.warning(f"  ⚠ Missing from batch: {', '.join(missing)} — retrying individually")
            for mt in missing:
                try:
                    time.sleep(2)
                    retry_prompt = engine.build_analysis_prompt([valid_data[mt]])
                    retry_resp = engine.call_gemini(retry_prompt, temperature=0.6)
                    retry_result = extract_json(retry_resp)
                    if retry_result and "stocks" in retry_result and retry_result["stocks"]:
                        validated = validate_stock(retry_result["stocks"][0])
                        validated["price"] = valid_data[mt].get("price", 0)
                        validated["change_pct"] = valid_data[mt].get("change_pct", 0)
                        batch_results[mt] = validated
                        log.info(f"    ✓ {mt} (retry): {validated.get('signal', '?')}")
                    else:
                        log.error(f"    ✗ {mt} retry failed")
                except Exception as e:
                    log.error(f"    ✗ {mt} retry error: {e}")

        # Add results or fallback
        for t in batch_tickers:
            if t in batch_results:
                all_analyses.append(batch_results[t])
            else:
                all_analyses.append({
                    "ticker": t,
                    "company_name": valid_data[t].get("company_name", t),
                    "sector": valid_data[t].get("sector", ""),
                    "signal": "ניתוח נכשל",
                    "direction": "neutral",
                    "confidence": 0,
                    "price": valid_data[t].get("price", 0),
                    "change_pct": valid_data[t].get("change_pct", 0),
                    "earnings_insight": "ניתוח לא זמין — נסה שנית",
                    "bottom_line": "ניתוח לא זמין",
                    "catalysts": [{"type": "unknown", "title": "לא זמין", "description": "הניתוח נכשל", "impact": "לא ברור", "timeframe": "לא ידוע"}],
                    "questions": [{"question": "?", "answer": "ניתוח לא זמין", "implication": "ניטרלי"}],
                })

        if i + batch_size < len(tickers_list):
            time.sleep(2)  # Rate limit between batches

    # Sort by confidence (highest first)
    all_analyses.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    # ─── Phase 3: MERGE (for partial runs) ───
    existing_data = None
    if partial_run and Config.OUTPUT_FILE.exists():
        try:
            with open(Config.OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            log.info("Loaded existing data for merge")
        except Exception:
            pass

    if partial_run and existing_data and "stocks" in existing_data:
        # Merge: replace updated tickers, keep rest
        existing_map = {s["ticker"]: s for s in existing_data["stocks"]}
        for analysis in all_analyses:
            existing_map[analysis["ticker"]] = analysis
        all_analyses = sorted(existing_map.values(), key=lambda x: x.get("confidence", 0), reverse=True)

    # ─── Phase 4: SAVE ───
    log.info("PHASE 3: VALIDATE & SAVE")

    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "market_status": get_market_status(),
            "pipeline_version": "3.0",
            "total_tickers": len(all_analyses),
            "run_mode": "partial" if partial_run else "full",
            "tickers_updated": run_tickers,
        },
        "stocks": all_analyses,
    }

    # Save
    Config.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # History
    Config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history_file = Config.HISTORY_DIR / f"{date_str}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info(f"Archived to {history_file}")

    # History index
    history_index = []
    for hf in sorted(Config.HISTORY_DIR.glob("*.json"), reverse=True):
        if hf.name == "index.json":
            continue
        history_index.append({"date": hf.stem, "file": f"history/{hf.name}"})
    with open(Config.HISTORY_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(history_index, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    log.info(f"Saved {len(all_analyses)} analyses to {Config.OUTPUT_FILE}")
    log.info(f"Total time: {elapsed:.1f}s")
    log.info("=" * 60)
    log.info("CATALYST v3.0 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
