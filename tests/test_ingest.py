"""Pure-function checks for the ingester (no DB, no network).

The DB insert path is declarative SQL (ON CONFLICT DO NOTHING) and not worth a
live Postgres in unit tests; the parsing + dedup logic is what can break.
"""

from finance_mlops import ingest


def test_read_tickers_strips_comments_blanks_and_uppercases(tmp_path):
    f = tmp_path / "tickers.txt"
    f.write_text("# header\naapl\n\n  msft  # inline comment\nNVDA\n")
    assert ingest.read_tickers(f) == ["AAPL", "MSFT", "NVDA"]


def test_dedup_key_ignores_case_and_whitespace():
    a = ingest.dedup_key("AAPL", "Apple   beats   earnings")
    b = ingest.dedup_key("AAPL", "  apple beats EARNINGS  ")
    assert a == b


def test_dedup_key_differs_by_ticker():
    # Same headline in two feeds must produce two rows (per-ticker grain).
    assert ingest.dedup_key("AAPL", "Big merger news") != ingest.dedup_key(
        "MSFT", "Big merger news"
    )


def test_rows_for_ticker_maps_fields_and_skips_titleless(monkeypatch):
    fake = type(
        "Feed",
        (),
        {
            "entries": [
                {
                    "title": "Apple soars",
                    "link": "https://news.example/aapl?utm=x",
                    "summary": "<p>up</p>",
                    "source": {"title": "Reuters"},
                    "published_parsed": (2026, 6, 17, 12, 0, 0, 0, 0, 0),
                },
                {"title": "   ", "link": "https://news.example/empty"},  # skipped
            ]
        },
    )()
    monkeypatch.setattr(ingest.feedparser, "parse", lambda _url: fake)

    rows = ingest.rows_for_ticker("AAPL")
    assert len(rows) == 1  # title-less entry dropped
    row = rows[0]
    assert row["ticker"] == "AAPL"
    assert row["headline"] == "Apple soars"
    assert row["source"] == "Reuters"
    assert row["url"] == "https://news.example/aapl?utm=x"
    assert row["dedup_key"] == ingest.dedup_key("AAPL", "Apple soars")
    assert row["published_at"] is not None
