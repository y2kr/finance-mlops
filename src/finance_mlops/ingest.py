from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import psycopg

log = logging.getLogger("ingest")

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id           bigserial PRIMARY KEY,
    dedup_key    text NOT NULL UNIQUE,        -- sha256(ticker + normalized_title)
    ticker       text NOT NULL,               -- watchlist symbol from the feed
    source       text,                        -- outlet, e.g. "Reuters" (from RSS)
    headline     text NOT NULL,
    summary      text,                        -- RSS <description>, often empty
    url          text,                        -- raw, for clicking (NOT the dedup key)
    published_at timestamptz,                 -- feed pubDate; nullable if missing
    ingested_at  timestamptz NOT NULL DEFAULT now()
);
"""

INSERT = """
INSERT INTO news (dedup_key, ticker, source, headline, summary, url, published_at)
VALUES (%(dedup_key)s, %(ticker)s, %(source)s, %(headline)s, %(summary)s,
        %(url)s, %(published_at)s)
ON CONFLICT (dedup_key) DO NOTHING
"""

DB_URL = os.environ.get("DATABASE_URL", "postgresql:///finance_mlops")
TICKERS_FILE = Path(os.environ.get("TICKERS_FILE", "tickers.txt"))


def feed_url(ticker: str) -> str:
    q = quote_plus(f"{ticker} stock")
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def read_tickers(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.upper())
    return out


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def dedup_key(ticker: str, title: str) -> str:
    raw = f"{ticker}\x00{normalize_title(title)}".encode()
    return hashlib.sha256(raw).hexdigest()


def _published_at(entry) -> datetime | None:
    t = entry.get("published_parsed")
    return datetime.fromtimestamp(time.mktime(t), tz=UTC) if t else None


def rows_for_ticker(ticker: str) -> list[dict]:
    feed = feedparser.parse(feed_url(ticker))
    rows = []
    for e in feed.entries:
        title = (e.get("title") or "").strip()
        if not title:
            log.warning("skip %s: feed entry missing title", ticker)
            continue
        source = e.get("source") or {}
        rows.append(
            {
                "dedup_key": dedup_key(ticker, title),
                "ticker": ticker,
                "source": source.get("title"),
                "headline": title,
                "summary": e.get("summary"),
                "url": e.get("link"),
                "published_at": _published_at(e),
            }
        )
    return rows


def ingest(conn) -> int:
    conn.execute(SCHEMA)
    inserted = 0
    for ticker in read_tickers(TICKERS_FILE):
        for row in rows_for_ticker(ticker):
            inserted += conn.execute(INSERT, row).rowcount
    conn.commit()
    return inserted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with psycopg.connect(DB_URL) as conn:
        n = ingest(conn)
    log.info("ingested %d new rows", n)


if __name__ == "__main__":
    main()
