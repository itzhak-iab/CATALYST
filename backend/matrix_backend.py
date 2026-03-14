#!/usr/bin/env python3
"""
Command & Control Matrix — Backend Engine v3.0 (Dynamic Radar)
================================================================
NEW ARCHITECTURE (v3):
  Phase 1: RADAR SCAN — dynamically find 30-50 interesting stocks
  Phase 2: AI TRIAGE — Gemini picks Top 3 per column from radar hits
  Phase 3: DEEP FETCH — full options/insiders/debt for selected ~12 stocks
  Phase 4: AI X-RAY — full 8-parameter contrarian analysis
  Phase 5: VALIDATE & SAVE — master_data.json for the Frontend

No more hardcoded TARGETS. The system finds its own prey every day.

IMPORTANT: The Contrarian Philosophy prompts and JSON Schema
are IMMUTABLE. This upgrade is purely architectural.
"""

import os
import sys
import json

# Load .env file if running locally (GitHub Actions uses Secrets instead)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required in CI — secrets come from environment
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Tuple, Dict, List

# ==============================================================
# LOGGING SETUP
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
logger = logging.getLogger("matrix_engine")

# ==============================================================
# DEPENDENCY CHECK & IMPORTS
# ==============================================================
def _check_dependency(module_name: str, pip_name: Optional[str] = None):
    """Import a module or exit with a clear message."""
    try:
        return __import__(module_name)
    except ImportError:
        pip_name = pip_name or module_name
        logger.critical(f"Missing dependency: {module_name}. Install with: pip install {pip_name}")
        sys.exit(1)

yf = _check_dependency("yfinance")

# Use new google.genai SDK (the old google.generativeai is deprecated)
try:
    from google import genai as genai_sdk
    from google.genai import types as genai_types
    GENAI_NEW_SDK = True
    logger.info("Using google.genai (new SDK)")
except ImportError:
    try:
        import google.generativeai as genai_sdk
        GENAI_NEW_SDK = False
        logger.info("Using google.generativeai (legacy SDK)")
    except ImportError:
        logger.critical("Missing dependency: google-genai. Install with: pip install google-genai")
        sys.exit(1)

# ==============================================================
# CONFIGURATION
# ==============================================================
class Config:
    """Central configuration — all tunables in one place."""

    # --- API ---
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_MAX_RETRIES = 3
    GEMINI_RETRY_DELAY_SEC = 5
    GEMINI_TEMPERATURE = 0.4

    # --- yfinance ---
    YF_MAX_RETRIES = 2
    YF_RETRY_DELAY_SEC = 2

    # --- Radar ---
    RADAR_TARGET_COUNT = 40       # How many stocks the radar aims to surface
    RADAR_LIGHT_FIELDS_ONLY = True  # Phase 1 fetches light data only
    TOP_PICKS_PER_COLUMN = 3      # AI selects this many per strategy column

    # --- Output ---
    # JSON goes to docs/ so GitHub Pages can serve it
    OUTPUT_FILE = Path(__file__).parent.parent / "docs" / "master_data.json"
    SCHEMA_FILE = Path(__file__).parent.parent / "docs" / "master_data_schema.json"


# ==============================================================
# 1. RADAR SCANNER (The Hunter)
# ==============================================================
class RadarScanner:
    """
    Dynamically discovers 30-50 stocks worth analyzing TODAY.

    Sources:
    1. Contrarian Universe — curated sectors we care about
       (commodities, energy, shipping, mining, defense, infrastructure)
    2. Yahoo Finance screeners — most active, top gainers/losers,
       unusual volume (anomalies = our bread and butter)
    3. Sector ETF holdings — catches what we might miss

    The Radar does NOT do deep analysis. It fetches LIGHT data
    (price, volume, change%) to feed the AI Triage step.
    """

    # ---- CONTRARIAN UNIVERSE (sectors the philosophy targets) ---- #
    # These are NOT hardcoded "picks" — they're the HUNTING GROUND.
    # The AI decides which ones matter today.
    CONTRARIAN_UNIVERSE = {
        # Uranium & Nuclear
        "uranium": ["CCJ", "UEC", "DNN", "NXE", "LEU", "UUUU"],
        # Shipping & Maritime
        "shipping": ["STNG", "INSW", "FRO", "GOGL", "EGLE", "GNK", "ZIM", "DAC", "SBLK"],
        # Mining & Metals
        "mining": ["VALE", "RIO", "BHP", "FCX", "TECK", "HBM", "SCCO"],
        # Oil & Gas (traditional energy)
        "oil_gas": ["OXY", "DVN", "PXD", "COP", "EOG", "MPC", "VLO", "PSX"],
        # Agriculture & Fertilizers
        "agriculture": ["MOS", "NTR", "CF", "FMC", "ADM", "BG"],
        # Defense & Aerospace
        "defense": ["LMT", "RTX", "NOC", "GD", "HII", "LHX"],
        # Infrastructure & Industrials (boring = beautiful)
        "infrastructure": ["CAT", "DE", "URI", "VMC", "MLM", "EMR"],
        # Coal & Thermal (ESG pariah = sin premium)
        "coal_thermal": ["BTU", "ARCH", "CEIX", "HCC", "AMR"],
        # Rare Earths & Critical Minerals
        "rare_earths": ["MP", "UAMY"],
        # Tech Bottlenecks (not tech stocks — supply chain chokepoints)
        "tech_bottleneck": ["ASML", "AMAT", "LRCX", "KLAC", "ONTO"],
        # Water & Utilities (anti-fragile)
        "water_utilities": ["AWK", "WM", "RSG", "WTRG"],
    }

    # ---- YAHOO SCREENER QUERIES ---- #
    # These catch daily anomalies outside our curated universe
    SCREENER_QUERIES = [
        "most_actives",
        "day_losers",
        "day_gainers",
        "undervalued_large_caps",
    ]

    @classmethod
    def scan(cls) -> List[Dict]:
        """
        Run the full radar scan. Returns a list of dicts with light data.
        Each dict has: ticker, company_name, sector, price, change_pct, volume, market_cap
        """
        logger.info("📡 RADAR: Starting dynamic stock scan...")
        all_tickers = set()
        radar_results = []

        # ---- Source 1: Contrarian Universe ---- #
        logger.info("   🎯 Source 1: Contrarian Universe sectors...")
        universe_tickers = []
        for sector, tickers in cls.CONTRARIAN_UNIVERSE.items():
            universe_tickers.extend(tickers)
        all_tickers.update(universe_tickers)
        logger.info(f"      {len(universe_tickers)} tickers from {len(cls.CONTRARIAN_UNIVERSE)} sectors")

        # ---- Source 2: Yahoo Screeners (dynamic daily anomalies) ---- #
        logger.info("   🔍 Source 2: Yahoo Finance screeners...")
        screener_tickers = cls._fetch_screener_tickers()
        new_from_screener = screener_tickers - all_tickers
        all_tickers.update(screener_tickers)
        logger.info(f"      {len(screener_tickers)} from screeners, {len(new_from_screener)} new additions")

        logger.info(f"   📊 Total unique tickers to scan: {len(all_tickers)}")

        # ---- Light Fetch: basic data for all tickers ---- #
        logger.info("   ⚡ Fetching light data (price, volume, change%)...")
        radar_results = cls._light_fetch_batch(list(all_tickers))

        # ---- Filter & Rank: keep the most interesting ---- #
        logger.info(f"   🧮 Filtering {len(radar_results)} stocks for anomalies...")
        filtered = cls._filter_anomalies(radar_results)

        logger.info(f"   ✅ RADAR complete: {len(filtered)} stocks passed anomaly filter")
        return filtered

    @classmethod
    def _fetch_screener_tickers(cls) -> set:
        """Fetch tickers from Yahoo Finance screeners."""
        tickers = set()
        for query in cls.SCREENER_QUERIES:
            try:
                screener = yf.Screener()
                screener.set_default_body(query)
                response = screener.response
                quotes = response.get("quotes", [])
                for quote in quotes[:15]:  # Top 15 from each screener
                    symbol = quote.get("symbol", "")
                    if symbol and "." not in symbol and len(symbol) <= 5:
                        tickers.add(symbol)
                logger.info(f"      ✓ {query}: {min(len(quotes), 15)} tickers")
            except Exception as e:
                logger.warning(f"      ✗ {query} failed: {e}")
                # Try alternative method
                try:
                    screener_alt = yf.screen(query)
                    if screener_alt and "quotes" in screener_alt:
                        for quote in screener_alt["quotes"][:15]:
                            symbol = quote.get("symbol", "")
                            if symbol and "." not in symbol:
                                tickers.add(symbol)
                except Exception:
                    pass
        return tickers

    @classmethod
    def _light_fetch_batch(cls, tickers: List[str]) -> List[Dict]:
        """
        Fetch lightweight data for all tickers at once using yf.download().
        Much faster than individual Ticker() calls for 50+ stocks.
        """
        results = []

        # Batch download — one API call for all tickers
        try:
            batch_data = yf.download(
                tickers,
                period="5d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.error(f"   Batch download failed: {e}. Falling back to individual fetches.")
            batch_data = None

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}

                current = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                if current == 0:
                    continue  # Skip invalid tickers

                prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or current
                change_pct = round(((current - prev_close) / prev_close) * 100, 2) if prev_close else 0
                volume = info.get("regularMarketVolume") or info.get("volume") or 0
                avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day") or 1
                volume_ratio = round(volume / max(avg_volume, 1), 2)

                high_52w = info.get("fiftyTwoWeekHigh") or 1
                distance_pct = round(((current - high_52w) / high_52w) * 100, 2) if high_52w else 0

                results.append({
                    "ticker": ticker,
                    "company_name": info.get("shortName", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "sub_sector": info.get("industry", "Unknown"),
                    "exchange": info.get("exchange", ""),
                    "currency": info.get("currency", "USD"),
                    "current_price": current,
                    "change_pct_1d": change_pct,
                    "volume": volume,
                    "avg_volume_10d": avg_volume,
                    "volume_ratio": volume_ratio,
                    "market_cap": info.get("marketCap", 0),
                    "high_52w": high_52w,
                    "low_52w": info.get("fiftyTwoWeekLow", 0),
                    "distance_from_52w_high_pct": distance_pct,
                    "free_cash_flow": info.get("freeCashflow", 0),
                    "debt_to_equity": info.get("debtToEquity", 0),
                    "beta": info.get("beta", 0),
                    "short_pct_of_float": info.get("shortPercentOfFloat", 0),
                    "recommendation_key": info.get("recommendationKey", "none"),
                    "held_pct_insiders": info.get("heldPercentInsiders", 0),
                })
            except Exception as e:
                logger.debug(f"      Light fetch failed for {ticker}: {e}")
                continue

        return results

    @classmethod
    def _filter_anomalies(cls, stocks: List[Dict]) -> List[Dict]:
        """
        Filter and rank stocks by 'interestingness' score.
        We want: unusual volume, big price moves, high short interest,
        insider activity signals, and proximity to 52w extremes.
        """
        scored = []
        for stock in stocks:
            score = 0

            # Volume anomaly (most important for day trading)
            vr = stock.get("volume_ratio", 1)
            if vr > 3.0:
                score += 30
            elif vr > 2.0:
                score += 20
            elif vr > 1.5:
                score += 10

            # Price move (absolute value — both gaps up and gaps down matter)
            change = abs(stock.get("change_pct_1d", 0))
            if change > 5:
                score += 25
            elif change > 3:
                score += 15
            elif change > 1.5:
                score += 8

            # Near 52-week low (contrarian loves beaten-down)
            dist = stock.get("distance_from_52w_high_pct", 0)
            if dist < -40:
                score += 20
            elif dist < -25:
                score += 12
            elif dist < -15:
                score += 5

            # High short interest (potential squeeze)
            short_pct = stock.get("short_pct_of_float", 0) or 0
            if short_pct > 0.15:
                score += 15
            elif short_pct > 0.08:
                score += 8

            # Known contrarian sectors get a small boost
            sector = (stock.get("sector", "") or "").lower()
            contrarian_sectors = ["energy", "basic materials", "industrials", "utilities"]
            if any(cs in sector for cs in contrarian_sectors):
                score += 5

            # Market cap filter — we prefer mid/large cap (above $500M)
            mcap = stock.get("market_cap", 0) or 0
            if mcap < 500_000_000:
                score -= 10  # Penalize micro-caps

            stock["radar_score"] = score
            scored.append(stock)

        # Sort by score, take top N
        scored.sort(key=lambda x: x["radar_score"], reverse=True)
        target = Config.RADAR_TARGET_COUNT
        result = scored[:target]

        # Log top 10
        logger.info(f"   🏆 Top 10 radar hits:")
        for s in result[:10]:
            logger.info(
                f"      {s['ticker']:6s} | ${s['current_price']:>8.2f} | "
                f"Δ{s['change_pct_1d']:>+6.1f}% | Vol×{s['volume_ratio']:.1f} | "
                f"Score: {s['radar_score']}"
            )

        return result


# ==============================================================
# 2. DEEP DATA FETCHER (The Surgeon)
# ==============================================================
class DeepDataFetcher:
    """
    Fetches FULL enriched data for selected stocks only.
    Called AFTER the AI triage — only for the ~12 stocks that made the cut.

    Includes: options chain, insider transactions, balance sheet, analyst recs.
    """

    @classmethod
    def fetch_deep(cls, ticker: str) -> dict:
        """Full deep fetch for a single ticker."""
        logger.info(f"   🔬 Deep fetching {ticker}...")
        result = {"ticker": ticker, "fetch_errors": []}

        for attempt in range(1, Config.YF_MAX_RETRIES + 1):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
                    raise ValueError(f"Empty info for {ticker}")
                break
            except Exception as e:
                logger.warning(f"      Attempt {attempt}/{Config.YF_MAX_RETRIES} failed: {e}")
                if attempt < Config.YF_MAX_RETRIES:
                    time.sleep(Config.YF_RETRY_DELAY_SEC)
                else:
                    logger.error(f"      ❌ All attempts failed for {ticker}")
                    result["fetch_errors"].append(f"info: {str(e)}")
                    return result

        current = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        high_52w = info.get("fiftyTwoWeekHigh") or 1
        distance_pct = round(((current - high_52w) / high_52w) * 100, 2) if high_52w else 0

        result.update({
            "company_name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "sub_sector": info.get("industry", "Unknown"),
            "exchange": info.get("exchange", "Unknown"),
            "currency": info.get("currency", "USD"),
            "current_price": current,
            "change_pct_1d": round(info.get("regularMarketChangePercent", 0) or 0, 2),
            "high_52w": high_52w,
            "low_52w": info.get("fiftyTwoWeekLow", 0),
            "distance_from_52w_high_pct": distance_pct,
            "market_cap": info.get("marketCap", 0),
            "free_cash_flow": info.get("freeCashflow", 0),
            "fcf_yield_pct": cls._calc_fcf_yield(info),
            "debt_to_equity": info.get("debtToEquity", 0),
            "total_debt": info.get("totalDebt", 0),
            "total_cash": info.get("totalCash", 0),
            "ebitda": info.get("ebitda", 0),
            "trailing_pe": info.get("trailingPE", 0),
            "forward_pe": info.get("forwardPE", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "held_pct_insiders": info.get("heldPercentInsiders", 0),
            "held_pct_institutions": info.get("heldPercentInstitutions", 0),
            "short_pct_of_float": info.get("shortPercentOfFloat", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "beta": info.get("beta", 0),
            "recommendation_key": info.get("recommendationKey", "none"),
            "target_mean_price": info.get("targetMeanPrice", 0),
            "number_of_analyst_opinions": info.get("numberOfAnalystOpinions", 0),
        })

        # --- Deep data layers --- #
        result["options_summary"] = cls._fetch_options(stock, ticker, result)
        result["insider_transactions"] = cls._fetch_insiders(stock, ticker, result)
        result["analyst_recommendations"] = cls._fetch_analyst_recs(stock, ticker, result)
        result["balance_sheet_debt"] = cls._fetch_debt(stock, ticker, result)

        logger.info(
            f"      ✅ {ticker} deep fetch complete | "
            f"${current} | FCF Yield: {result.get('fcf_yield_pct', 0)}% | "
            f"Errors: {len(result['fetch_errors'])}"
        )
        return result

    @staticmethod
    def _calc_fcf_yield(info: dict) -> float:
        fcf = info.get("freeCashflow", 0) or 0
        mcap = info.get("marketCap", 0) or 0
        return round((fcf / mcap) * 100, 2) if mcap > 0 and fcf != 0 else 0.0

    @classmethod
    def _fetch_options(cls, stock, ticker: str, result: dict) -> dict:
        try:
            expirations = stock.options
            if not expirations:
                return {"available": False}

            nearest = expirations[0]
            chain = stock.option_chain(nearest)
            calls, puts = chain.calls, chain.puts

            total_call_vol = int(calls["volume"].sum()) if "volume" in calls.columns else 0
            total_put_vol = int(puts["volume"].sum()) if "volume" in puts.columns else 0
            put_call_ratio = round(total_put_vol / max(total_call_vol, 1), 2)

            avg_call_iv = round(float(calls["impliedVolatility"].mean()) * 100, 1) if "impliedVolatility" in calls.columns else 0
            avg_put_iv = round(float(puts["impliedVolatility"].mean()) * 100, 1) if "impliedVolatility" in puts.columns else 0

            top_calls = []
            if not calls.empty and "volume" in calls.columns:
                for _, row in calls.nlargest(3, "volume").iterrows():
                    top_calls.append({
                        "strike": float(row.get("strike", 0)),
                        "volume": int(row.get("volume", 0)),
                        "open_interest": int(row.get("openInterest", 0)),
                        "implied_volatility": round(float(row.get("impliedVolatility", 0)) * 100, 1),
                    })

            top_puts = []
            if not puts.empty and "volume" in puts.columns:
                for _, row in puts.nlargest(3, "volume").iterrows():
                    top_puts.append({
                        "strike": float(row.get("strike", 0)),
                        "volume": int(row.get("volume", 0)),
                        "open_interest": int(row.get("openInterest", 0)),
                        "implied_volatility": round(float(row.get("impliedVolatility", 0)) * 100, 1),
                    })

            return {
                "available": True,
                "nearest_expiry": nearest,
                "put_call_volume_ratio": put_call_ratio,
                "avg_call_iv_pct": avg_call_iv,
                "avg_put_iv_pct": avg_put_iv,
                "total_call_volume": total_call_vol,
                "total_put_volume": total_put_vol,
                "top_volume_calls": top_calls,
                "top_volume_puts": top_puts,
            }
        except Exception as e:
            result["fetch_errors"].append(f"options: {str(e)}")
            return {"available": False, "error": str(e)}

    @classmethod
    def _fetch_insiders(cls, stock, ticker: str, result: dict) -> dict:
        try:
            insiders = stock.insider_transactions
            if insiders is None or insiders.empty:
                return {"available": False}

            recent = []
            for _, row in insiders.head(10).iterrows():
                recent.append({
                    "insider": str(row.get("Insider", "Unknown")),
                    "relation": str(row.get("Insider Relation", "")),
                    "transaction": str(row.get("Transaction", "")),
                    "shares": int(row.get("Shares", 0)) if row.get("Shares") else 0,
                    "date": str(row.get("Start Date", "")),
                })

            buy_shares = sum(r["shares"] for r in recent if "purchase" in r.get("transaction", "").lower() or "buy" in r.get("transaction", "").lower())
            sell_shares = sum(r["shares"] for r in recent if "sale" in r.get("transaction", "").lower() or "sell" in r.get("transaction", "").lower())

            return {
                "available": True,
                "recent_transactions": recent,
                "net_insider_shares": buy_shares - sell_shares,
                "signal": "net_buying" if buy_shares > sell_shares else ("net_selling" if sell_shares > buy_shares else "neutral"),
            }
        except Exception as e:
            result["fetch_errors"].append(f"insiders: {str(e)}")
            return {"available": False, "error": str(e)}

    @classmethod
    def _fetch_analyst_recs(cls, stock, ticker: str, result: dict) -> dict:
        try:
            recs = stock.recommendations
            if recs is None or recs.empty:
                return {"available": False}

            entries = []
            for _, row in recs.tail(5).iterrows():
                entries.append({
                    "firm": str(row.get("Firm", "Unknown")),
                    "to_grade": str(row.get("To Grade", "")),
                    "from_grade": str(row.get("From Grade", "")),
                    "action": str(row.get("Action", "")),
                })
            return {"available": True, "recent": entries}
        except Exception as e:
            result["fetch_errors"].append(f"analyst_recs: {str(e)}")
            return {"available": False, "error": str(e)}

    @classmethod
    def _fetch_debt(cls, stock, ticker: str, result: dict) -> dict:
        try:
            bs = stock.balance_sheet
            if bs is None or bs.empty:
                return {"available": False}

            latest = bs.iloc[:, 0]
            return {
                "available": True,
                "total_debt": float(latest.get("Total Debt", 0) or 0),
                "total_equity": float(latest.get("Total Stockholders Equity", latest.get("Stockholders Equity", 0)) or 0),
                "total_cash_and_equivalents": float(latest.get("Cash And Cash Equivalents", 0) or 0),
                "net_debt": float((latest.get("Total Debt", 0) or 0) - (latest.get("Cash And Cash Equivalents", 0) or 0)),
            }
        except Exception as e:
            result["fetch_errors"].append(f"balance_sheet: {str(e)}")
            return {"available": False, "error": str(e)}


# ==============================================================
# 3. AI ENGINE (The Contrarian Brain)
# ==============================================================
class ContrarianAIEngine:
    """
    Two-stage AI pipeline:
    1. TRIAGE: Pick Top 3 per column from 40 radar hits (light data)
    2. X-RAY: Full 8-parameter analysis for the ~12 selected stocks (deep data)

    ⚠️  THE PROMPTS AND PHILOSOPHY BELOW ARE IMMUTABLE.
    ⚠️  DO NOT MODIFY THE AI INSTRUCTIONS OR SCHEMA EXPECTATIONS.
    """

    # ---- STRATEGY DEFINITIONS (IMMUTABLE) ---- #
    STRATEGIES = {
        "day_trading": {
            "name_he": "מארב נזילות",
            "name_en": "Liquidity Ambush",
            "objective": "ניצול מכירה מכנית/כפויה לרווח מהיר תוך-יומי, ללא חשיפת לילה",
            "edge_thesis": (
                "אנחנו לא קונים 'חדשות רעות'. אנחנו קונים וואקום נזילות שנוצר ממכירה "
                "אלגוריתמית עיוורת (ETF rebalancing, stop-loss cascades) — בידיעה שהמחיר "
                "יתוקן ברגע שהמכירה המכנית תסתיים."
            ),
            "ai_prompt": (
                "מתוך רשימת המניות שלנו, איזו חברה חווה הבוקר פער מחיר שלילי (Gap Down) "
                "שנובע ממכירה כפויה (ETF rebalancing, stop-loss cascade, margin call) ולא "
                "מפגיעה פיזית בעסק? חפש אנומליות נפח מול ממוצע 20 יום."
            ),
            "refresh_cadence": "intraday",
        },
        "swing": {
            "name_he": "עיוורון קטליזטורים",
            "name_en": "Blind Catalysts",
            "objective": "ניצול אירוע נקודתי 'משעמם' שהשוק מסנן — לפני שהתקשורת מגלה אותו",
            "edge_thesis": (
                'השוק מתמחר דו"חות שבועות מראש, אבל מתעלם מקטליזטורים "משעממים": '
                "סיום פריסת חוב, Lock-up expirations, פשיטת רגל של מתחרים, או שינויי "
                "רגולציה שקטים שמשנים את כלכלת התעשייה."
            ),
            "ai_prompt": (
                "איזו חברה מתקרבת לאירוע מכונן 'שקט' (סיום חוב, lock-up expiration, "
                "אישור רגולטורי, פשיטת רגל של מתחרה) שהשוק מתמחר בחסר כי הוא 'משעמם "
                "מדי' לכותרות?"
            ),
            "refresh_cadence": "daily_evening",
        },
        "position": {
            "name_he": "הנגזרת השנייה של צוואר הבקבוק",
            "name_en": "Second Derivative Bottleneck",
            "objective": "תפיסת החוליה הנסתרת בשרשרת האספקה — שלב אחד עמוק מהטרנד שכולם רואים",
            "edge_thesis": (
                "כולם קונים את הטרנד הישיר (שבבי AI, ליתיום לרכב חשמלי). אנחנו קונים את "
                "ה'את החפירה בבהלה לזהב' — החברה שמייצרת את קירור השרתים, את הנחושת "
                "לשנאים, את הדלק לספינות שמובילות את הכל."
            ),
            "ai_prompt": (
                "באיזה סקטור הכסף המוסדי מתחיל להבין שיש חוסר מהותי? איזו חוליה "
                "בשרשרת האספקה (שלב שני/שלישי מהטרנד הראשי) מציגה פריצה במחזור מסחר "
                "שמעידה על כניסת 'כסף חכם' — בעוד התקשורת עדיין לא גילתה את הסיפור?"
            ),
            "refresh_cadence": "daily_evening",
        },
        "investment": {
            "name_he": "אנטי-שבירות ומונופול תשתיתי",
            "name_en": "Anti-Fragility & Infrastructure Monopoly",
            "objective": (
                "גידור התיק מפני אינפלציה, ריביות גבוהות ומשברים גיאופוליטיים — "
                "בעלות על נכסים שהעולם לא יכול לחיות בלעדיהם"
            ),
            "edge_thesis": (
                'טכנולוגיה משתבשת. אנחנו משקיעים בנכסים "משעממים" שהביקוש אליהם קשיח '
                "לחלוטין: אנרגיה, חקלאות בסיסית, ספנות קשיחה, כרייה. חברות שמייצרות "
                "מיליארדים גם אם הריבית תעלה ל-10% או תפרוץ מלחמת עולם."
            ),
            "ai_prompt": (
                "לאיזו חברה יש את החפיר הפיזי הרחב ביותר (כורים, זיכיונות, נתיבי סחר) "
                "שאי אפשר לשכפל, מאזן חסין-כדורים, ויכולת לייצר תזרים מזומנים חיובי "
                "גם בתרחיש של מיתון + ריבית 10% + מלחמה?"
            ),
            "refresh_cadence": "weekly",
        },
    }

    # ---- 8 X-RAY PARAMETER DESCRIPTIONS (IMMUTABLE) ---- #
    XRAY_INSTRUCTIONS = """
    FOR EACH STOCK, you MUST populate ALL 8 X-Ray parameters inside the "xray" object:

    1. "options_mispricing" — פערי תמחור אופציות סמויים
       Required fields: score (0-100), signal (surprise_up|surprise_down|neutral|apathy), analysis (string)
       Optional: implied_volatility_percentile, put_call_ratio, unusual_activity[], next_earnings_date, sources[]
       Philosophy: Ignore analyst EPS estimates. Focus on what the OPTIONS MARKET is pricing.
       If IV Percentile < 20, insurance is cheap = opportunity. If market is APATHETIC to upcoming earnings = potential surprise.

    2. "regulatory_moat" — רגולציה כחומת מגן
       Required: score (0-100), moat_type (license_monopoly|environmental_barrier|government_concession|capital_barrier|patent_wall|none), analysis
       Optional: barrier_cost_estimate, pending_legislation[], threat_level (1-10), sources[]
       Philosophy: Regulation is NOT a risk — it's a BARRIER TO ENTRY for competitors.
       If government demands $3B and 10 years to open a new mine, the existing company becomes a LEGAL MONOPOLY.

    3. "esg_sin_premium" — פרמיית החטא
       Required: score (0-100), boycott_intensity (extreme|high|moderate|low|none), analysis
       Optional: excluded_from_indices[], institutional_ownership_trend, competitor_funding_difficulty, new_capacity_pipeline_years, sources[]
       Philosophy: The more HATED an industry is by ESG funds, the harder it is for competitors to raise capital.
       Result: zero new supply + massive demand = windfall profits for incumbents.

    4. "crowd_exhaustion" — מיצוי כבשים מול איסוף זאבים
       Required: score (0-100), phase (wolf_accumulation|early_crowd|peak_euphoria|distribution|capitulation), analysis
       Optional: insider_activity{}, buyback_status{}, analyst_consensus{}, media_sentiment{}, ripe_for_exit (bool), sources[]
       Philosophy: If INSIDERS are buying while the PUBLIC is indifferent = BUY (wolf_accumulation).
       If PUBLIC is euphoric while INSIDERS sell = SELL (peak_euphoria). Compare earnings transcripts to Twitter headlines.

    5. "boring_premium" — ארביטראז' השעמום
       Required: score (0-100), hype_to_fcf_ratio (number), analysis
       Optional: media_mentions_weekly, social_media_mentions_weekly, free_cash_flow_annual_m, fcf_yield_pct, google_trends_score, business_description, sources[]
       Philosophy: Hype-to-FCF Ratio = (weekly_mentions / annual_FCF_in_millions). LOWER = better.
       Boring companies making billions that nobody talks about at dinner parties = ideal.

    6. "debt_asymmetry" — אסימטריה של מבנה החוב
       Required: score (0-100), verdict (inflation_asset|neutral|ticking_bomb), analysis
       Optional: total_debt_m, debt_to_equity, weighted_avg_rate, fixed_rate_pct, maturity_wall[], refinancing_risk, interest_coverage_ratio, sources[]
       Philosophy: Debt locked at 2-3% fixed rate while market rate is 5.5% = the debt is an INFLATION ASSET.
       Debt that needs refinancing soon at high rates = TICKING BOMB. Analyze the MATURITY WALL, not just total debt.

    7. "hostage_power" — תופס ערובה או בן ערובה
       Required: score (0-100), role (hostage_taker|mutual_dependency|hostage|replaceable), analysis
       Optional: market_share_pct, substitution_difficulty, pricing_power_evidence, geopolitical_leverage{}, supply_chain_position, sources[]
       Philosophy: We ONLY want "hostage_taker" companies — the missing link in the supply chain that cannot be bypassed.
       If a company can raise prices 20% and NO customer can leave, that's absolute pricing power.

    8. "capital_allocation_iq" — מנת המשכל של הקצאת ההון
       Required: score (0-100), ceo_type (stealth_compounder|disciplined_allocator|neutral|empire_builder|value_destroyer), analysis
       Optional: ceo_name, ceo_tenure_years, fcf_deployment{}, shares_outstanding_trend{}, roic_5y_avg, ma_track_record, skin_in_the_game, sources[]
       Philosophy: Maximum score for CEOs who are QUIET and AGGRESSIVE with buybacks (stealth_compounder).
       Shrinking the share count = increasing our ownership without investing another dollar.
       Empire builders who make headline acquisitions at premium prices = value_destroyer.
    """

    # ================================================================
    # STAGE 1: TRIAGE — Pick Top 3 per column from radar data
    # ================================================================
    @classmethod
    def build_triage_prompt(cls, radar_data: List[Dict]) -> str:
        """
        Build a LIGHT prompt for the AI to select candidates.
        Only sends basic price/volume/sector data — NOT deep data.
        """
        # Compact the radar data to reduce token usage
        compact = []
        for s in radar_data:
            compact.append({
                "ticker": s["ticker"],
                "name": s.get("company_name", s["ticker"]),
                "sector": s.get("sector", "?"),
                "industry": s.get("sub_sector", "?"),
                "price": s.get("current_price", 0),
                "change_1d": s.get("change_pct_1d", 0),
                "vol_ratio": s.get("volume_ratio", 1),
                "dist_52w_high": s.get("distance_from_52w_high_pct", 0),
                "mcap_m": round((s.get("market_cap", 0) or 0) / 1_000_000),
                "fcf": s.get("free_cash_flow", 0),
                "d_e": s.get("debt_to_equity", 0),
                "beta": s.get("beta", 0),
                "short_float": s.get("short_pct_of_float", 0),
                "insider_pct": s.get("held_pct_insiders", 0),
                "radar_score": s.get("radar_score", 0),
            })

        return f"""You are the Chief Investment Officer of an elite Contrarian Hedge Fund.

TODAY'S RADAR SCAN surfaced {len(compact)} stocks with unusual activity.
Your job: select the TOP 3 stocks for EACH of our 4 strategy columns.
A stock MAY appear in multiple columns if it fits multiple strategies.

RADAR DATA (light scan — price, volume, sector only):
{json.dumps(compact, indent=1)}

OUR 4 STRATEGIES:
{json.dumps(cls.STRATEGIES, indent=2, ensure_ascii=False)}

SELECTION CRITERIA (Contrarian Philosophy):
- Day Trading: Look for MECHANICAL selling (high volume_ratio + negative change = possible ETF dump / stop-loss cascade). NOT fundamental bad news.
- Swing: Look for BORING catalysts the market ignores (debt refinancing, lock-up expiry, quiet regulatory approval). Near-term trigger.
- Position: Look for SECOND DERIVATIVE bottleneck plays — supply chain chokepoints one level deeper than the obvious trend.
- Investment: Look for ANTI-FRAGILE monopolies — physical moats, inelastic demand, sin premium sectors, boring cash machines.

IMPORTANT: Pick stocks that BEST FIT each strategy's contrarian thesis.
Prefer stocks from sectors like: energy, shipping, mining, agriculture, defense, infrastructure.
Avoid FAANG/big-tech unless they are specifically a supply chain chokepoint.

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "triage": {{
    "day_trading": ["TICK1", "TICK2", "TICK3"],
    "swing": ["TICK1", "TICK2", "TICK3"],
    "position": ["TICK1", "TICK2", "TICK3"],
    "investment": ["TICK1", "TICK2", "TICK3"]
  }},
  "reasoning": {{
    "day_trading": "One sentence why these 3",
    "swing": "One sentence why these 3",
    "position": "One sentence why these 3",
    "investment": "One sentence why these 3"
  }}
}}

Return ONLY the JSON. No markdown, no backticks.
"""

    # ================================================================
    # STAGE 2: FULL X-RAY — Deep analysis for selected stocks
    # ================================================================
    @classmethod
    def build_xray_prompt(cls, deep_data: Dict[str, List[dict]]) -> str:
        """
        Build the FULL prompt with deep data for the selected ~12 stocks.
        This is the same quality prompt as v2.0 but fed with richer data.
        """
        return f"""You are the Chief Investment Officer of an elite Contrarian Hedge Fund.
Your mission: produce the FULL Command & Control Matrix with 8 X-Ray parameters per stock.

═══════════════════════════════════════════
DEEP MARKET DATA (options, insiders, debt, analyst recs included)
═══════════════════════════════════════════
{json.dumps(deep_data, indent=2, default=str)}

═══════════════════════════════════════════
THE 4 STRATEGY COLUMNS
═══════════════════════════════════════════
{json.dumps(cls.STRATEGIES, indent=2, ensure_ascii=False)}

═══════════════════════════════════════════
THE 8 X-RAY PARAMETERS (MANDATORY FOR EACH STOCK)
═══════════════════════════════════════════
{cls.XRAY_INSTRUCTIONS}

═══════════════════════════════════════════
EXACT OUTPUT STRUCTURE
═══════════════════════════════════════════
Return a JSON object with this structure:
{{
  "matrix": {{
    "day_trading": {{
      "strategy": {{ "name_he": "...", "name_en": "...", "objective": "...", "edge_thesis": "..." }},
      "ai_prompt": "...",
      "refresh_cadence": "intraday",
      "top_picks": [
        {{
          "ticker": "...", "company_name": "...", "sector": "...", "sub_sector": "...", "exchange": "...",
          "price": {{ "current": 0, "currency": "USD", "change_pct_1d": 0, "change_pct_1w": 0, "change_pct_1m": 0, "high_52w": 0, "low_52w": 0, "distance_from_52w_high_pct": 0 }},
          "composite_score": {{ "total": 0, "confidence": "high|medium|low", "breakdown": {{ "options_mispricing": 0, "regulatory_moat": 0, "esg_sin_premium": 0, "crowd_exhaustion": 0, "boring_premium": 0, "debt_asymmetry": 0, "hostage_power": 0, "capital_allocation_iq": 0 }} }},
          "thesis_summary": "2 sentences max",
          "xray": {{ ALL 8 PARAMETERS with score + analysis + required fields }},
          "action": {{ "recommendation": "...", "entry_zone": "...", "stop_loss": "...", "target_1": "...", "target_2": "...", "time_horizon": "...", "key_trigger": "..." }}
        }}
      ]
    }},
    "swing": {{ ... same ... }},
    "position": {{ ... same ... }},
    "investment": {{ ... same ... }}
  }}
}}

CRITICAL RULES:
- Use REAL price data from the deep market data. Do NOT invent prices.
- Every stock MUST have ALL 8 xray parameters filled with score + analysis.
- Use the OPTIONS data, INSIDER data, DEBT data provided to enrich your analysis.
- Source types: news|sec_filing|earnings_transcript|analyst_report|data_provider|social|government|research_paper
- Source credibility: verified|reliable|unverified
- composite_score.total = weighted average of the 8 breakdown scores.
- DO NOT include the "meta" key — backend injects it.
- Return ONLY raw JSON. No markdown, no backticks.
"""

    # ================================================================
    # GEMINI API CALLER (shared by both stages)
    # ================================================================
    @classmethod
    def call_gemini(cls, prompt: str, expect_key: str = "matrix") -> Optional[dict]:
        """
        Call Gemini API with retries and JSON extraction.
        expect_key: the top-level key to validate ("matrix" or "triage").
        """
        if not Config.GEMINI_API_KEY:
            logger.critical("❌ GEMINI_API_KEY is not set!")
            return None

        if GENAI_NEW_SDK:
            client = genai_sdk.Client(api_key=Config.GEMINI_API_KEY)
        else:
            genai_sdk.configure(api_key=Config.GEMINI_API_KEY)
            model = genai_sdk.GenerativeModel(
                Config.GEMINI_MODEL,
                generation_config=genai_sdk.GenerationConfig(
                    temperature=Config.GEMINI_TEMPERATURE,
                    response_mime_type="application/json",
                ),
            )

        for attempt in range(1, Config.GEMINI_MAX_RETRIES + 1):
            logger.info(f"   🧠 Gemini call — attempt {attempt}/{Config.GEMINI_MAX_RETRIES}...")
            try:
                if GENAI_NEW_SDK:
                    response = client.models.generate_content(
                        model=Config.GEMINI_MODEL,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=Config.GEMINI_TEMPERATURE,
                            response_mime_type="application/json",
                        ),
                    )
                else:
                    response = model.generate_content(prompt)

                if not response or not response.text:
                    logger.warning(f"      Empty response on attempt {attempt}")
                    continue

                parsed = cls._extract_json(response.text.strip())
                if parsed is None:
                    logger.warning(f"      JSON parse failed on attempt {attempt}")
                    if attempt < Config.GEMINI_MAX_RETRIES:
                        time.sleep(Config.GEMINI_RETRY_DELAY_SEC)
                    continue

                if expect_key not in parsed:
                    logger.warning(f"      Missing '{expect_key}' key on attempt {attempt}")
                    if attempt < Config.GEMINI_MAX_RETRIES:
                        time.sleep(Config.GEMINI_RETRY_DELAY_SEC)
                    continue

                logger.info(f"   ✅ Gemini returned valid JSON with '{expect_key}' key")
                return parsed

            except Exception as e:
                logger.error(f"      Gemini error on attempt {attempt}: {e}")
                if attempt < Config.GEMINI_MAX_RETRIES:
                    time.sleep(Config.GEMINI_RETRY_DELAY_SEC * attempt)

        logger.critical("❌ All Gemini attempts exhausted.")
        return None

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[dict]:
        """Aggressively extract JSON from AI output."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        cleaned = raw_text
        for prefix in ("```json", "```JSON", "```"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass

        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(raw_text[first:last + 1])
            except json.JSONDecodeError:
                pass

        logger.error("   Could not extract valid JSON from AI output")
        return None


# ==============================================================
# 4. VALIDATOR (The Quality Gate)
# ==============================================================
class OutputValidator:
    """Validates the final JSON. Does NOT modify content."""

    REQUIRED_COLUMNS = ["day_trading", "swing", "position", "investment"]
    REQUIRED_XRAY_KEYS = [
        "options_mispricing", "regulatory_moat", "esg_sin_premium",
        "crowd_exhaustion", "boring_premium", "debt_asymmetry",
        "hostage_power", "capital_allocation_iq",
    ]

    @classmethod
    def validate(cls, data: dict) -> Tuple[bool, List[str]]:
        warnings = []
        if "matrix" not in data:
            return False, ["FATAL: Missing 'matrix' key"]

        matrix = data["matrix"]
        for col in cls.REQUIRED_COLUMNS:
            if col not in matrix:
                warnings.append(f"Missing column: {col}")
                continue

            column = matrix[col]
            if "strategy" not in column:
                warnings.append(f"{col}: Missing 'strategy'")

            picks = column.get("top_picks", [])
            if not isinstance(picks, list):
                warnings.append(f"{col}: 'top_picks' is not a list")
                continue
            if len(picks) > 3:
                column["top_picks"] = picks[:3]

            for i, pick in enumerate(picks):
                prefix = f"{col}[{i}]({pick.get('ticker', '?')})"
                for field in ["ticker", "company_name", "thesis_summary"]:
                    if field not in pick:
                        warnings.append(f"{prefix}: Missing '{field}'")

                xray = pick.get("xray", {})
                if not xray:
                    warnings.append(f"{prefix}: Missing 'xray'")
                else:
                    for key in cls.REQUIRED_XRAY_KEYS:
                        if key not in xray:
                            warnings.append(f"{prefix}: Missing xray.{key}")
                        elif isinstance(xray[key], dict):
                            if "score" not in xray[key]:
                                warnings.append(f"{prefix}: xray.{key} missing 'score'")
                            if "analysis" not in xray[key]:
                                warnings.append(f"{prefix}: xray.{key} missing 'analysis'")

                cs = pick.get("composite_score", {})
                if not cs or "total" not in cs:
                    warnings.append(f"{prefix}: Missing composite_score.total")

        return not any("FATAL" in w for w in warnings), warnings


# ==============================================================
# 5. ORCHESTRATOR (The Conductor) — NEW 5-PHASE PIPELINE
# ==============================================================
def determine_market_status() -> str:
    from datetime import timezone as tz
    now_utc = datetime.now(tz.utc)
    et_hour = (now_utc.hour - 5) % 24
    weekday = now_utc.weekday()
    if weekday >= 5:
        return "weekend"
    if et_hour < 4:
        return "closed"
    if et_hour < 9 or (et_hour == 9 and now_utc.minute < 30):
        return "pre_market"
    if et_hour < 16:
        return "open"
    if et_hour < 20:
        return "after_hours"
    return "closed"


def main():
    """
    NEW PIPELINE:
    Phase 1: RADAR SCAN → find 30-50 interesting stocks dynamically
    Phase 2: AI TRIAGE → Gemini picks Top 3 per column (light data)
    Phase 3: DEEP FETCH → full data for ~12 selected stocks only
    Phase 4: AI X-RAY → full 8-parameter contrarian analysis
    Phase 5: VALIDATE & SAVE → master_data.json
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("🚀 COMMAND & CONTROL MATRIX v3.0 — Dynamic Radar Engine")
    logger.info(f"   Timestamp: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"   Model: {Config.GEMINI_MODEL}")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════
    # PHASE 1: RADAR SCAN
    # ═══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("📡 PHASE 1: RADAR SCAN — Hunting for anomalies...")
    logger.info("=" * 60)

    radar_results = RadarScanner.scan()

    if len(radar_results) < 4:
        logger.critical(f"❌ Radar returned only {len(radar_results)} stocks. Need at least 4. Aborting.")
        sys.exit(1)

    logger.info(f"\n📊 Phase 1 complete: {len(radar_results)} stocks on the radar")

    # ═══════════════════════════════════════════
    # PHASE 2: AI TRIAGE
    # ═══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("🧠 PHASE 2: AI TRIAGE — Gemini selects Top 3 per column...")
    logger.info("=" * 60)

    triage_prompt = ContrarianAIEngine.build_triage_prompt(radar_results)
    logger.info(f"   Triage prompt size: {len(triage_prompt):,} characters")

    triage_result = ContrarianAIEngine.call_gemini(triage_prompt, expect_key="triage")

    if triage_result is None:
        logger.critical("❌ AI triage failed. Aborting.")
        sys.exit(1)

    triage = triage_result["triage"]
    selected_tickers = set()
    for col, tickers in triage.items():
        if isinstance(tickers, list):
            selected_tickers.update(tickers)
            logger.info(f"   {col}: {tickers}")

    # Log reasoning if available
    reasoning = triage_result.get("reasoning", {})
    for col, reason in reasoning.items():
        logger.info(f"   💡 {col}: {reason}")

    logger.info(f"\n📊 Phase 2 complete: {len(selected_tickers)} unique stocks selected for deep analysis")

    # ═══════════════════════════════════════════
    # PHASE 3: DEEP FETCH
    # ═══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("🔬 PHASE 3: DEEP FETCH — Full data for selected stocks...")
    logger.info("=" * 60)

    deep_data: Dict[str, List[dict]] = {
        "day_trading": [],
        "swing": [],
        "position": [],
        "investment": [],
    }

    # Fetch deep data for each selected ticker
    fetched_cache: Dict[str, dict] = {}
    for ticker in selected_tickers:
        if ticker not in fetched_cache:
            fetched_cache[ticker] = DeepDataFetcher.fetch_deep(ticker)

    # Organize into columns
    for col, tickers in triage.items():
        if col in deep_data and isinstance(tickers, list):
            for ticker in tickers:
                if ticker in fetched_cache:
                    deep_data[col].append(fetched_cache[ticker])

    total_errors = sum(len(d.get("fetch_errors", [])) for d in fetched_cache.values())
    logger.info(f"\n📊 Phase 3 complete: {len(fetched_cache)} stocks deep-fetched, {total_errors} non-fatal errors")

    # ═══════════════════════════════════════════
    # PHASE 4: AI X-RAY (Full 8-parameter analysis)
    # ═══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("🔍 PHASE 4: AI X-RAY — Full contrarian analysis...")
    logger.info("=" * 60)

    xray_prompt = ContrarianAIEngine.build_xray_prompt(deep_data)
    logger.info(f"   X-Ray prompt size: {len(xray_prompt):,} characters")

    ai_result = ContrarianAIEngine.call_gemini(xray_prompt, expect_key="matrix")

    if ai_result is None:
        logger.critical("❌ AI X-Ray failed. Aborting.")
        sys.exit(1)

    # ═══════════════════════════════════════════
    # PHASE 4.5: INJECT METADATA
    # ═══════════════════════════════════════════
    logger.info("\n🔧 Injecting metadata...")
    market_status = determine_market_status()
    ai_result["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": f"matrix-engine-v3.0-radar/{Config.GEMINI_MODEL}",
        "data_sources": [
            "yfinance_api",
            "yahoo_screeners",
            "cboe_options_chain",
            "sec_insider_transactions",
            "analyst_recommendations",
            f"gemini_{Config.GEMINI_MODEL}",
        ],
        "market_status": market_status,
        "next_refresh": (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
        "radar_stats": {
            "total_scanned": len(radar_results),
            "selected_for_deep_analysis": len(selected_tickers),
            "triage_reasoning": reasoning,
        },
    }

    # Inject immutable strategy definitions
    for col_name, strategy_def in ContrarianAIEngine.STRATEGIES.items():
        if col_name in ai_result.get("matrix", {}):
            col = ai_result["matrix"][col_name]
            col["strategy"] = strategy_def
            col["ai_prompt"] = strategy_def["ai_prompt"]
            col["refresh_cadence"] = strategy_def["refresh_cadence"]

    # ═══════════════════════════════════════════
    # PHASE 5: VALIDATE & SAVE
    # ═══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("✅ PHASE 5: VALIDATE & SAVE")
    logger.info("=" * 60)

    is_valid, warnings = OutputValidator.validate(ai_result)
    if warnings:
        logger.warning(f"   ⚠️  {len(warnings)} validation warnings:")
        for w in warnings:
            logger.warning(f"      - {w}")
    else:
        logger.info("   ✅ Zero validation warnings")

    if not is_valid:
        logger.critical("❌ Validation FAILED. Output not saved.")
        sys.exit(1)

    output_path = Config.OUTPUT_FILE
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ai_result, f, ensure_ascii=False, indent=2)
        file_size_kb = output_path.stat().st_size / 1024
        logger.info(f"   ✅ Saved to: {output_path}")
        logger.info(f"   📦 File size: {file_size_kb:.1f} KB")
    except Exception as e:
        logger.critical(f"❌ Failed to save: {e}")
        sys.exit(1)

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    elapsed = round(time.time() - start_time, 1)
    pick_counts = {}
    for col in ["day_trading", "swing", "position", "investment"]:
        pick_counts[col] = len(ai_result.get("matrix", {}).get(col, {}).get("top_picks", []))

    logger.info("\n" + "=" * 60)
    logger.info("🏁 ENGINE COMPLETE")
    logger.info(f"   Total time: {elapsed}s")
    logger.info(f"   Market status: {market_status}")
    logger.info(f"   Radar scanned: {len(radar_results)} stocks")
    logger.info(f"   Deep analyzed: {len(selected_tickers)} stocks")
    logger.info(f"   Final picks: {pick_counts}")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Log: {log_filename}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
