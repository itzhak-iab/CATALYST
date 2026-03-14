#!/usr/bin/env python3
"""
Q5 Command Matrix — macro_agent.py (PRD v1.0)
==============================================
מערכת מודיעין פיננסית קונטרריאנית.

Pipeline:
  Phase 1: RADAR — סריקה דינמית של 50+ מניות
  Phase 2: AI TRIAGE — Gemini בוחר Top 3 לכל עמודה
  Phase 3: DEEP FETCH — נתונים מעמיקים ל-12 מניות נבחרות
  Phase 4: AI X-RAY — ניתוח 8 פרמטרים לכל מניה (בעברית)
  Phase 5: VALIDATE & SAVE — master_data.json

כל הפלט בעברית. מפתחות JSON באנגלית.
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Tuple, Dict, List

# Load .env for local runs; GitHub Actions injects secrets via environment
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

log_filename = LOG_DIR / f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("Q5")

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
# CONFIG
# ==============================================================
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"
    OUTPUT_FILE = Path(__file__).parent.parent / "docs" / "master_data.json"
    HISTORY_DIR = Path(__file__).parent.parent / "docs" / "history"
    CONFIG_FILE = Path(__file__).parent.parent / "docs" / "config.json"
    MAX_RETRIES = 3
    RETRY_DELAY = 5

    # 8 X-Ray parameters per PRD
    XRAY_KEYS = [
        "earnings_guidance",
        "regulation",
        "esg_ethics",
        "analyst_peak",
        "boring_premium",
        "debt_asymmetry",
        "hostage_power",
        "capital_iq",
    ]


# ==============================================================
# PHASE 1: RADAR SCANNER
# ==============================================================
class RadarScanner:
    """Dynamically discovers ~50 interesting stocks from contrarian sectors + screeners."""

    CONTRARIAN_UNIVERSE = {
        "shipping": ["STNG", "EGLE", "SBLK", "ZIM", "DAC", "GSL"],
        "coal_energy": ["BTU", "AMR", "HCC", "ARLP", "CTRA"],
        "tobacco_sin": ["MO", "PM", "BTI", "IMBBY"],
        "defense": ["LMT", "RTX", "NOC", "GD", "HII"],
        "nuclear_uranium": ["CCJ", "UEC", "LEU", "NNE", "SMR"],
        "oil_gas": ["ET", "EPD", "MPLX", "PBF", "DK"],
        "infrastructure": ["URI", "FLR", "PWR", "EME", "STRL"],
        "rare_earth_mining": ["MP", "LAC", "ALB", "UUUU"],
        "reits_value": ["STWD", "ABR", "BXMT", "NLY"],
        "pharma_value": ["PFE", "BMY", "TEVA", "VTRS"],
        "telecoms_boring": ["T", "VZ", "LUMN"],
        "ag_commodities": ["ADM", "BG", "CTVA", "MOS", "NTR"],
        "banks_value": ["C", "WFC", "USB", "KEY", "RF"],
        "auto_old_economy": ["F", "GM", "STLA"],
        "industrial_boring": ["CMI", "CAT", "DE", "PCAR"],
    }

    # Additional trending tickers to add diversity beyond the static universe
    EXTRA_TICKERS = [
        "INTC", "BA", "PYPL", "DIS", "NCLH", "CCL",  # beaten-down large caps
        "CLF", "X", "AA", "FCX",                       # metals/materials
        "OXY", "DVN", "HAL", "SLB",                    # energy services
        "KMI", "WMB", "OKE",                            # midstream
    ]

    def scan(self) -> List[str]:
        """Returns a deduplicated list of tickers from universe + extras."""
        all_tickers = set()

        # Static universe
        for sector, tickers in self.CONTRARIAN_UNIVERSE.items():
            for t in tickers:
                all_tickers.add(t)

        # Extra tickers for diversity
        for t in self.EXTRA_TICKERS:
            all_tickers.add(t)

        log.info(f"Total universe tickers: {len(all_tickers)}")
        return sorted(all_tickers)

    def fetch_light_data(self, tickers: List[str]) -> List[Dict]:
        """Fetch lightweight data (price, volume, change) for triage."""
        log.info(f"Fetching light data for {len(tickers)} tickers...")
        results = []

        # Batch download
        try:
            data = yf.download(
                tickers,
                period="5d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            log.error(f"Batch download failed: {e}")
            return results

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data
                else:
                    df = data[ticker] if ticker in data.columns.get_level_values(0) else None

                if df is None or df.empty:
                    continue

                df = df.dropna()
                if len(df) < 2:
                    continue

                last_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                change_pct = ((last_close - prev_close) / prev_close) * 100
                volume = int(df["Volume"].iloc[-1])
                avg_volume = int(df["Volume"].mean())

                results.append({
                    "ticker": ticker,
                    "price": round(last_close, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "vol_ratio": round(volume / max(avg_volume, 1), 2),
                })
            except Exception:
                continue

        log.info(f"Light data fetched for {len(results)} tickers")
        return results


# ==============================================================
# PHASE 3: DEEP DATA FETCHER
# ==============================================================
class DeepDataFetcher:
    """Fetch in-depth data for selected ~12 stocks."""

    def fetch_deep(self, ticker: str) -> Dict:
        """Full financial profile for a single stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            result = {
                "ticker": ticker,
                "company_name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "price": {
                    "current": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                    "target_mean": info.get("targetMeanPrice", 0),
                    "52w_high": info.get("fiftyTwoWeekHigh", 0),
                    "52w_low": info.get("fiftyTwoWeekLow", 0),
                },
                "fundamentals": {
                    "pe_ratio": info.get("trailingPE", None),
                    "forward_pe": info.get("forwardPE", None),
                    "pb_ratio": info.get("priceToBook", None),
                    "dividend_yield": info.get("dividendYield", None),
                    "payout_ratio": info.get("payoutRatio", None),
                    "free_cash_flow": info.get("freeCashflow", None),
                    "revenue": info.get("totalRevenue", None),
                    "ebitda": info.get("ebitda", None),
                    "profit_margin": info.get("profitMargins", None),
                    "roe": info.get("returnOnEquity", None),
                    "debt_to_equity": info.get("debtToEquity", None),
                    "total_debt": info.get("totalDebt", None),
                    "total_cash": info.get("totalCash", None),
                },
                "analyst": {},
                "business_description": info.get("longBusinessSummary", ""),
                "analyst_ratings": {},
                "insiders": [],
            }

            # Analyst recommendations
            try:
                recs = stock.recommendations
                if recs is not None and not recs.empty:
                    latest = recs.tail(5).to_dict("records")
                    result["analyst"]["recent_recs"] = latest
            except Exception:
                pass

            # Analyst ratings summary
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

            # Insider transactions
            try:
                insiders = stock.insider_transactions
                if insiders is not None and not insiders.empty:
                    recent = insiders.head(10).to_dict("records")
                    result["insiders"] = recent
            except Exception:
                pass

            # Earnings dates
            try:
                cal = stock.calendar
                if cal is not None:
                    if isinstance(cal, dict):
                        result["earnings_date"] = str(cal.get("Earnings Date", ["Unknown"])[0]) if "Earnings Date" in cal else "Unknown"
                    else:
                        result["earnings_date"] = "Unknown"
            except Exception:
                result["earnings_date"] = "Unknown"

            return result

        except Exception as e:
            log.warning(f"Deep fetch failed for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}


# ==============================================================
# CONTRARIAN AI ENGINE
# ==============================================================
class ContrarianAIEngine:
    """Interfaces with Gemini for triage + full X-Ray analysis."""

    # ───── STRATEGY DEFINITIONS (per PRD Section 3) ─────
    STRATEGY_DEFINITIONS = """
## 4 זירות הפעולה:

### 1. day_trading — מסחר יומי: ארביטראז' סמנטי
זיהוי חברה שחווה פער מחיר שלילי (Gap Down) כתוצאה מ"רעש" (למשל ציוץ או עדכון מדד) ולא מפגיעה פונדמנטלית בעסק. רווח מהיר מפאניקה.

### 2. swing — סווינג: רכיבה על קטליזטור
זיהוי חברה המתקרבת לאירוע מכונן (דו"ח כספי, החלטת ריבית, אישור רגולטורי) אשר השוק מתמחר אותו כרגע בחסר.

### 3. position — פוזיציה: שבירת מגמה בצווארי בקבוק
זיהוי סקטור שבו מתחיל להיכנס "כסף חכם" (מחזורי מסחר חריגים) עקב חוסר מהותי ובלתי נראה בתשתית או חומר גלם.

### 4. investment — השקעה ארוכת טווח: מונופול וחפיר פיזי
חברה בעלת חפיר כלכלי שלא ניתן לשכפול (זיכיונות, כורים), מאזן חסין אינפלציה, ויכולת ייצור תזרים מזומנים גם במיתון עולמי.
"""

    # ───── X-RAY DEFINITIONS (per PRD Section 4) ─────
    XRAY_DEFINITIONS = """
## 8 פרמטרי רנטגן — לכל מניה שנבחרה:

1. **earnings_guidance** (מעקב דוחות וצפי): תאריך הדו"ח הקרוב, צפי האנליסטים מול התזה ההפוכה/נסתרת שלנו.
2. **regulation** (שדה מוקשים רגולטורי/משפטי): מהו האיום הממשלתי/משפטי המרכזי? האם הוא חוסם מתחרים או סכנה קיומית?
3. **esg_ethics** (פרדוקס האתיקה וה-ESG): מי מנסה להרוס את החברה? האם החרם חוסם מתחרים ושומר על היצע נמוך?
4. **analyst_peak** (מדד מיצוי קונצנזוס): האם ההמונים כבר כאן? אם כולם ממליצים "קנייה חזקה" — סימן שלילי. אנו מחפשים שווקים ריקים.
5. **boring_premium** (ארביטראז' השעמום): יחס Hype-to-FCF. חברות אפורות שמייצרות הררי מזומנים אך ללא סיקור תקשורתי.
6. **debt_asymmetry** (אסימטריית חוב אינפלציונית): חברות עם חוב ארוך בריבית אפסית קבועה, שמרוויחות משחיקת החוב באינפלציה.
7. **hostage_power** (רדיוס הפיצוץ ותופס ערובה): האם החברה חוליה קריטית בשרשרת האספקה? האם הלקוח יכול לעבור למתחרה?
8. **capital_iq** (מנת משכל של הקצאת הון): האם המנכ"ל בונה אימפריות מיותרות או שמרן שמכווץ מניות ומשלם דיבידנדים?
"""

    # ───── JSON SCHEMA (per PRD Section 5) ─────
    JSON_SCHEMA = """{
  "meta": {
    "generated_at": "ISO_TIMESTAMP"
  },
  "matrix": {
    "day_trading": {
      "top_picks": [
        {
          "ticker": "XXX",
          "company_name": "שם החברה בעברית",
          "sector": "סקטור בעברית",
          "price": { "current": 0.0 },
          "composite_score": { "total": 0 },
          "thesis_summary": "תקציר התזה בעברית — 2-3 משפטים",
          "xray": {
            "earnings_guidance": { "score": 0, "analysis": "ניתוח בעברית 2-3 משפטים" },
            "regulation": { "score": 0, "analysis": "..." },
            "esg_ethics": { "score": 0, "analysis": "..." },
            "analyst_peak": { "score": 0, "analysis": "..." },
            "boring_premium": { "score": 0, "analysis": "..." },
            "debt_asymmetry": { "score": 0, "analysis": "..." },
            "hostage_power": { "score": 0, "analysis": "..." },
            "capital_iq": { "score": 0, "analysis": "..." }
          }
        }
      ]
    },
    "swing": { "top_picks": ["3 מניות במבנה זהה"] },
    "position": { "top_picks": ["3 מניות במבנה זהה"] },
    "investment": { "top_picks": ["3 מניות במבנה זהה"] }
  }
}"""

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
        """Call Gemini with retry logic."""
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
                log.warning(f"Gemini attempt {attempt}/{Config.MAX_RETRIES} failed: {e}")
                if attempt < Config.MAX_RETRIES:
                    time.sleep(Config.RETRY_DELAY * attempt)
                else:
                    raise

    def build_triage_prompt(self, radar_data: List[Dict]) -> str:
        """Phase 2: Ask Gemini to pick Top 3 per column from radar hits."""
        compact = json.dumps(radar_data, ensure_ascii=False, indent=None)

        return f"""אתה אנליסט השקעות קונטרריאני (Contrarian).

{self.STRATEGY_DEFINITIONS}

## נתוני הראדר (מחירים, נפחים, שינויים):
{compact}

## המשימה שלך:
מתוך רשימת המניות, בחר בדיוק 3 מניות לכל אחת מ-4 הזירות.
מניה יכולה להופיע ביותר מזירה אחת אם היא מתאימה.

## פורמט הפלט — JSON בלבד:
```json
{{
  "day_trading": ["TICK1", "TICK2", "TICK3"],
  "swing": ["TICK1", "TICK2", "TICK3"],
  "position": ["TICK1", "TICK2", "TICK3"],
  "investment": ["TICK1", "TICK2", "TICK3"]
}}
```

החזר JSON בלבד. ללא טקסט נוסף."""

    def build_xray_prompt(self, column_key: str, column_desc: str, deep_data: List[Dict]) -> str:
        """Phase 4: Full X-Ray for 3 stocks in one column."""
        data_str = json.dumps(deep_data, ensure_ascii=False, default=str)

        return f"""אתה אנליסט קונטרריאני. נתונים פיננסיים מפורטים של 3 מניות לזירת **{column_desc}**.

{self.XRAY_DEFINITIONS}

## הנתונים:
{data_str}

## המשימה:
ייצר ניתוח X-Ray מלא עבור כל אחת מ-3 המניות.

### כללים קריטיים:
1. **כל הטקסט בעברית בלבד** — תקציר, ניתוח, שמות סקטורים, תיאור החברה, דעת האנליסטים.
2. **מפתחות JSON באנגלית בלבד** — בדיוק כפי שמופיע בסכמה.
3. **ציון 1-100** לכל פרמטר — מבוסס על הפילוסופיה הקונטרריאנית (ציון גבוה = הזדמנות קונטרריאנית חזקה).
4. **ניתוח 2-3 משפטים** לכל פרמטר — ספציפי, עם נתונים.
5. **composite_score** — ממוצע משוקלל של 8 הפרמטרים.
6. **company_description** — תיאור של 3-4 משפטים בעברית על מה החברה עושה.
7. **analyst_ratings** — סיכום דעת האנליסטים בעברית, כולל consensus (קנייה/החזק/מכירה), summary (2-3 משפטים), bull_case (משפט אחד), bear_case (משפט אחד).

## פורמט פלט — JSON בלבד:
```json
{{
  "top_picks": [
    {{
      "ticker": "XXX",
      "company_name": "שם בעברית",
      "sector": "סקטור בעברית",
      "price": {{ "current": 0.0 }},
      "composite_score": {{ "total": 0 }},
      "thesis_summary": "תקציר קונטרריאני 2-3 משפטים בעברית",
      "company_description": "תיאור החברה בעברית — 3-4 משפטים על מה החברה עושה",
      "analyst_ratings": {{
        "consensus": "קנייה/החזק/מכירה",
        "summary": "סיכום דעת האנליסטים בעברית — 2-3 משפטים",
        "bull_case": "התזה החיובית — משפט אחד",
        "bear_case": "התזה השלילית — משפט אחד"
      }},
      "xray": {{
        "earnings_guidance": {{ "score": 0, "analysis": "ניתוח בעברית" }},
        "regulation": {{ "score": 0, "analysis": "..." }},
        "esg_ethics": {{ "score": 0, "analysis": "..." }},
        "analyst_peak": {{ "score": 0, "analysis": "..." }},
        "boring_premium": {{ "score": 0, "analysis": "..." }},
        "debt_asymmetry": {{ "score": 0, "analysis": "..." }},
        "hostage_power": {{ "score": 0, "analysis": "..." }},
        "capital_iq": {{ "score": 0, "analysis": "..." }}
      }}
    }}
  ]
}}
```

החזר JSON בלבד. ללא טקסט נוסף."""


# ==============================================================
# JSON EXTRACTION
# ==============================================================
def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from Gemini response, handling markdown fences and edge cases."""
    if not text or not text.strip():
        return None

    # Method 1: Direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Method 2: Strip markdown code fences (multiple patterns)
    cleaned = text.strip()
    for fence in ["```json", "```JSON", "```"]:
        if fence in cleaned:
            parts = cleaned.split(fence, 1)
            if len(parts) > 1:
                cleaned = parts[1]
                break
    if "```" in cleaned:
        cleaned = cleaned.split("```")[0]

    try:
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Method 3: Find outermost { } with brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = -1

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
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    # Method 4: Last resort — find last }
    last_brace = text.rfind("}")
    if last_brace > start:
        try:
            return json.loads(text[start:last_brace + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    log.error(f"All JSON extraction methods failed. Text preview: {text[:200]}")
    return None


# ==============================================================
# VALIDATOR
# ==============================================================
class OutputValidator:
    COLUMNS = ["day_trading", "swing", "position", "investment"]

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        errors = []

        if "meta" not in data:
            errors.append("Missing 'meta'")
        if "matrix" not in data:
            errors.append("Missing 'matrix'")
            return False, errors

        matrix = data["matrix"]
        for col in self.COLUMNS:
            if col not in matrix:
                errors.append(f"Missing column '{col}'")
                continue
            picks = matrix[col].get("top_picks", [])
            if len(picks) != 3:
                errors.append(f"'{col}' has {len(picks)} picks (expected 3)")

            for i, pick in enumerate(picks):
                if "ticker" not in pick:
                    errors.append(f"'{col}' pick {i}: missing ticker")
                if "xray" not in pick:
                    errors.append(f"'{col}' pick {i}: missing xray")
                    continue
                xray = pick["xray"]
                for key in Config.XRAY_KEYS:
                    if key not in xray:
                        errors.append(f"'{col}' {pick.get('ticker','?')}: missing xray.{key}")
                    elif "score" not in xray[key]:
                        errors.append(f"'{col}' {pick.get('ticker','?')}: xray.{key} missing score")

        return len(errors) == 0, errors


# ==============================================================
# MARKET STATUS HELPER
# ==============================================================
def get_market_status() -> str:
    """Determine current US market status."""
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
def main():
    log.info("=" * 60)
    log.info("Q5 COMMAND MATRIX — macro_agent.py")
    log.info("=" * 60)
    start_time = time.time()

    # ─── Phase 1: RADAR ───
    log.info("PHASE 1: RADAR SCAN")
    radar = RadarScanner()
    tickers = radar.scan()
    light_data = radar.fetch_light_data(tickers)

    if not light_data:
        log.error("Radar returned no data. Aborting.")
        sys.exit(1)

    log.info(f"Radar complete: {len(light_data)} stocks with data")

    # ─── Phase 2: AI TRIAGE ───
    log.info("PHASE 2: AI TRIAGE")
    ai = ContrarianAIEngine()

    triage_prompt = ai.build_triage_prompt(light_data)
    triage_response = ai.call_gemini(triage_prompt, temperature=0.5)
    triage_result = extract_json(triage_response)

    if not triage_result:
        log.error("Failed to parse triage response. Raw output saved.")
        log.error(triage_response[:500])
        sys.exit(1)

    # Collect all unique tickers selected
    selected_tickers = set()
    column_tickers = {}  # type: Dict[str, List[str]]
    for col in OutputValidator.COLUMNS:
        picks = triage_result.get(col, [])
        column_tickers[col] = picks[:3]  # Max 3 per column
        for t in picks[:3]:
            selected_tickers.add(t)

    log.info(f"Triage selected {len(selected_tickers)} unique tickers: {selected_tickers}")

    # ─── Phase 3: DEEP FETCH ───
    log.info("PHASE 3: DEEP DATA FETCH")
    deep_fetcher = DeepDataFetcher()
    deep_data_map = {}  # type: Dict[str, Dict]

    for ticker in selected_tickers:
        log.info(f"  Deep fetch: {ticker}")
        deep_data_map[ticker] = deep_fetcher.fetch_deep(ticker)
        time.sleep(0.3)  # Rate limiting

    # ─── Phase 4: AI X-RAY ───
    log.info("PHASE 4: AI X-RAY ANALYSIS")
    column_descriptions = {
        "day_trading": "מסחר יומי — ארביטראז' סמנטי",
        "swing": "סווינג — רכיבה על קטליזטור",
        "position": "פוזיציה — שבירת מגמה בצווארי בקבוק",
        "investment": "השקעה ארוכת טווח — מונופול וחפיר פיזי",
    }

    final_matrix = {}

    for col in OutputValidator.COLUMNS:
        log.info(f"  X-Ray for column: {col}")
        tickers_for_col = column_tickers.get(col, [])
        deep_for_col = [deep_data_map.get(t, {"ticker": t}) for t in tickers_for_col]

        xray_prompt = ai.build_xray_prompt(col, column_descriptions[col], deep_for_col)
        xray_response = ai.call_gemini(xray_prompt, temperature=0.6)
        xray_result = extract_json(xray_response)

        if xray_result and "top_picks" in xray_result:
            final_matrix[col] = {"top_picks": xray_result["top_picks"]}
        else:
            log.error(f"  Failed to parse X-Ray for {col}")
            log.error(f"  Raw: {xray_response[:300]}")
            # Fallback: create minimal entries
            final_matrix[col] = {
                "top_picks": [
                    {
                        "ticker": t,
                        "company_name": deep_data_map.get(t, {}).get("company_name", t),
                        "sector": deep_data_map.get(t, {}).get("sector", ""),
                        "price": {"current": deep_data_map.get(t, {}).get("price", {}).get("current", 0)},
                        "composite_score": {"total": 0},
                        "thesis_summary": "ניתוח לא זמין — נסה שנית",
                        "xray": {k: {"score": 0, "analysis": "לא זמין"} for k in Config.XRAY_KEYS},
                    }
                    for t in tickers_for_col
                ]
            }

        time.sleep(1)  # Rate limiting between columns

    # ─── Phase 5: VALIDATE & SAVE ───
    log.info("PHASE 5: VALIDATE & SAVE")

    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "market_status": get_market_status(),
            "radar_stats": {
                "total_scanned": len(tickers),
                "with_data": len(light_data),
                "selected_for_analysis": len(selected_tickers),
            },
        },
        "matrix": final_matrix,
    }

    validator = OutputValidator()
    is_valid, errors = validator.validate(output)

    if not is_valid:
        log.warning(f"Validation warnings ({len(errors)}):")
        for err in errors:
            log.warning(f"  - {err}")
    else:
        log.info("Validation PASSED — all 4 columns × 3 picks × 8 X-Ray params")

    # Save
    Config.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Archive to history
    Config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history_file = Config.HISTORY_DIR / f"{date_str}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info(f"Archived to {history_file}")

    # Update history index
    history_index = []
    for hf in sorted(Config.HISTORY_DIR.glob("*.json"), reverse=True):
        if hf.name == "index.json":
            continue
        history_index.append({
            "date": hf.stem,
            "file": f"history/{hf.name}",
        })
    with open(Config.HISTORY_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(history_index, f, ensure_ascii=False, indent=2)
    log.info(f"History index updated: {len(history_index)} entries")

    elapsed = time.time() - start_time
    log.info(f"Saved to {Config.OUTPUT_FILE}")
    log.info(f"Total time: {elapsed:.1f}s")
    log.info("=" * 60)
    log.info("Q5 MATRIX COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
