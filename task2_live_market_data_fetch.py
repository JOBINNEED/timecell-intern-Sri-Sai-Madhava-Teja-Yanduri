#!/usr/bin/env python3
"""
TASK 2: Live Market Data Fetch
Timecell.AI Summer Internship 2025 - Technical Assessment

Fetches real-time asset prices from free public APIs concurrently.
- CoinGecko for crypto (BTC, ETH) — returns INR + USD natively, no FX math
- CoinGecko for Gold (pax-gold) — troy oz price, converted to per-gram in a dedicated transform
- Yahoo Finance for NIFTY50 (Indian index)

Design decisions:
- httpx.AsyncClient: fetches all sources simultaneously instead of sequentially
- tenacity: exponential backoff retries on transient failures, no pre-flight ping anti-pattern
- Pydantic: typed AssetPrice model, no raw dicts
- Market timestamps: sourced from API responses, not local clock
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import httpx
import pytz
from pydantic import BaseModel
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# Logging — clean, structured output for errors
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
TROY_OZ_TO_GRAMS = 31.1035


# ---------------------------------------------------------------------------
# Data model — typed, no raw dicts
# ---------------------------------------------------------------------------
class AssetPrice(BaseModel):
    name: str
    price: float
    currency: str
    price_timestamp: datetime  # when the market last priced this asset


# ---------------------------------------------------------------------------
# Asset config — each asset knows its own source + any post-fetch transform
# ---------------------------------------------------------------------------
@dataclass
class AssetConfig:
    display_name: str
    source: str                          # "coingecko" | "yahoo"
    source_id: str                       # CoinGecko ID or Yahoo ticker
    currency: str                        # display currency label
    transform: Callable[[float], float]  # price transformation (identity by default)


def _identity(x: float) -> float:
    return x


def _troy_oz_to_per_gram(price_inr: float) -> float:
    """Convert INR-per-troy-oz → INR-per-gram."""
    return price_inr / TROY_OZ_TO_GRAMS


# Assets to fetch — easy to extend (add Silver, Platinum, etc.)
ASSETS: list[AssetConfig] = [
    AssetConfig("BTC",    "coingecko", "bitcoin",  "USD",   _identity),
    AssetConfig("ETH",    "coingecko", "ethereum", "USD",   _identity),
    AssetConfig("GOLD",   "coingecko", "pax-gold", "INR/g", _troy_oz_to_per_gram),
    AssetConfig("NIFTY50","yahoo",     "^NSEI",    "INR",   _identity),
]


# ---------------------------------------------------------------------------
# Retry decorator — exponential backoff, no pre-flight ping
# ---------------------------------------------------------------------------
def _with_retries(func):
    return retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )(func)


# ---------------------------------------------------------------------------
# Fetchers — single responsibility: fetch raw data, nothing else
# ---------------------------------------------------------------------------
@_with_retries
async def _fetch_coingecko(
    client: httpx.AsyncClient,
    ids: list[str],
) -> dict:
    """
    Fetch prices for a list of CoinGecko IDs.
    Requests both USD and INR so no manual FX conversion is needed.
    """
    response = await client.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": ",".join(ids),
            "vs_currencies": "usd,inr",
            "include_last_updated_at": "true",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@_with_retries
async def _fetch_yahoo(client: httpx.AsyncClient, ticker: str) -> dict:
    """Fetch a single Yahoo Finance chart response."""
    response = await client.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={"interval": "1d", "range": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Parsers — extract AssetPrice from raw API payloads
# ---------------------------------------------------------------------------
def _parse_coingecko(asset: AssetConfig, raw: dict) -> AssetPrice:
    """
    Pull price + market timestamp out of a CoinGecko /simple/price response.
    Uses INR directly for gold (no hardcoded FX rate), USD for everything else.
    """
    entry = raw[asset.source_id]

    # Gold is displayed in INR/g — use the INR price from the API
    raw_price = entry["inr"] if asset.currency.startswith("INR") else entry["usd"]
    price = asset.transform(raw_price)

    # Market timestamp comes from the API, not the local clock
    market_ts = datetime.fromtimestamp(entry["last_updated_at"], tz=IST)

    return AssetPrice(
        name=asset.display_name,
        price=price,
        currency=asset.currency,
        price_timestamp=market_ts,
    )


def _parse_yahoo(asset: AssetConfig, raw: dict) -> AssetPrice:
    """Pull price + market timestamp out of a Yahoo Finance chart response."""
    meta = raw["chart"]["result"][0]["meta"]
    price = asset.transform(meta["regularMarketPrice"])
    currency = meta.get("currency", asset.currency)

    # regularMarketTime is a Unix timestamp — use it, not datetime.now()
    market_ts = datetime.fromtimestamp(meta["regularMarketTime"], tz=IST)

    return AssetPrice(
        name=asset.display_name,
        price=price,
        currency=currency,
        price_timestamp=market_ts,
    )


# ---------------------------------------------------------------------------
# Orchestrator — fires all fetches concurrently, collects results + errors
# ---------------------------------------------------------------------------
async def fetch_all_assets(
    assets: list[AssetConfig],
) -> tuple[list[AssetPrice], list[str]]:
    """
    Fetch all assets concurrently using a single shared httpx client.
    Returns (successful_results, error_messages).
    """
    results: list[AssetPrice] = []
    errors: list[str] = []

    # Group CoinGecko assets so we hit the API once for all of them
    cg_assets = [a for a in assets if a.source == "coingecko"]
    yahoo_assets = [a for a in assets if a.source == "yahoo"]

    async with httpx.AsyncClient() as client:
        # Build coroutines
        cg_task = _fetch_coingecko(client, [a.source_id for a in cg_assets])
        yahoo_tasks = [_fetch_yahoo(client, a.source_id) for a in yahoo_assets]

        # Fire everything simultaneously
        all_tasks = [cg_task] + yahoo_tasks
        raw_responses = await asyncio.gather(*all_tasks, return_exceptions=True)

    # --- Parse CoinGecko batch ---
    cg_raw = raw_responses[0]
    if isinstance(cg_raw, Exception):
        errors.append(f"CoinGecko batch failed: {cg_raw}")
        log.error("✗ CoinGecko batch failed: %s", cg_raw)
    else:
        for asset in cg_assets:
            try:
                ap = _parse_coingecko(asset, cg_raw)
                results.append(ap)
                log.info("✓ %s @ %s", asset.display_name, ap.price_timestamp.strftime("%H:%M:%S %Z"))
            except (KeyError, TypeError) as exc:
                errors.append(f"{asset.display_name}: parse error — {exc}")
                log.error("✗ %s parse error: %s", asset.display_name, exc)

    # --- Parse Yahoo responses ---
    for asset, raw in zip(yahoo_assets, raw_responses[1:]):
        if isinstance(raw, Exception):
            errors.append(f"{asset.display_name} failed: {raw}")
            log.error("✗ %s failed: %s", asset.display_name, raw)
        else:
            try:
                ap = _parse_yahoo(asset, raw)
                results.append(ap)
                log.info("✓ %s @ %s", asset.display_name, ap.price_timestamp.strftime("%H:%M:%S %Z"))
            except (KeyError, TypeError) as exc:
                errors.append(f"{asset.display_name}: parse error — {exc}")
                log.error("✗ %s parse error: %s", asset.display_name, exc)

    return results, errors


# ---------------------------------------------------------------------------
# Display — formats AssetPrice objects into a clean terminal table
# ---------------------------------------------------------------------------
def display_results(results: list[AssetPrice], errors: list[str]) -> None:
    fetch_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")

    print("\n" + "=" * 70)
    print(f"  Asset Prices — script ran at {fetch_time}")
    print("=" * 70 + "\n")

    if results:
        headers = ["Asset", "Price", "Currency", "Price Timestamp (IST)"]
        rows = [
            [
                ap.name,
                f"{ap.price:>14,.2f}",
                ap.currency,
                ap.price_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ]
            for ap in results
        ]
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("  No asset prices were successfully fetched.")

    if errors:
        print("\n" + "─" * 70)
        print("  ERRORS")
        print("─" * 70)
        for err in errors:
            print(f"  • {err}")

    total = len(results) + len(errors)
    print("\n" + "─" * 70)
    print(f"  Fetched {len(results)}/{total} assets successfully.")
    print("─" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 70)
    print("  TASK 2: LIVE MARKET DATA FETCH")
    print("=" * 70)

    results, errors = asyncio.run(fetch_all_assets(ASSETS))
    display_results(results, errors)


if __name__ == "__main__":
    main()
