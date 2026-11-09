"""News ingestion from JSON, CSV and RSS sources into :class:`NewsItem` objects."""

from __future__ import annotations

import csv
import json
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from news_sentiment_trader.models import NewsItem, make_id, to_utc

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
USER_AGENT = "news-sentiment-trader/0.1 (+research)"


def parse_timestamp(value: str | int | float | datetime) -> datetime:
    """Parse ISO-8601, RFC-822 (RSS) or epoch-seconds timestamps into UTC."""
    if isinstance(value, datetime):
        return to_utc(value)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        return to_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        pass
    try:
        return to_utc(parsedate_to_datetime(text))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unrecognized timestamp: {value!r}") from exc


def _split_tickers(raw: Any) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        parts = raw.replace(";", ",").replace("|", ",").split(",")
    else:
        parts = list(raw)
    return tuple(p.strip().upper() for p in parts if str(p).strip())


def item_from_record(record: Mapping[str, Any], default_source: str = "unknown") -> NewsItem:
    """Build a :class:`NewsItem` from a loosely-structured mapping.

    Accepts common field aliases (``published``/``date`` for the timestamp,
    ``headline`` for the title, ``description`` for the summary).
    """
    title = str(record.get("title") or record.get("headline") or "").strip()
    if not title:
        raise ValueError("record has no title")
    ts_raw = record.get("timestamp") or record.get("published") or record.get("date")
    if ts_raw is None:
        raise ValueError(f"record has no timestamp: {title!r}")
    ts = parse_timestamp(ts_raw)
    source = str(record.get("source") or default_source)
    summary = str(record.get("summary") or record.get("description") or "").strip()
    item_id = str(record.get("id") or make_id(source, title, ts))
    return NewsItem(
        id=item_id,
        timestamp=ts,
        source=source,
        title=title,
        summary=summary,
        tickers=_split_tickers(record.get("tickers")),
    )


def load_json(path: str | Path) -> list[NewsItem]:
    """Load news from a JSON file containing a list or ``{"items": [...]}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("items", []) if isinstance(data, dict) else data
    return [item_from_record(r) for r in records]


def load_csv(path: str | Path) -> list[NewsItem]:
    """Load news from a CSV file with at least ``timestamp`` and ``title`` columns."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [item_from_record(row) for row in csv.DictReader(fh)]


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def parse_rss(xml_text: str, source: str | None = None) -> list[NewsItem]:
    """Parse an RSS 2.0 or Atom document into news items.

    Items without a parseable date are skipped rather than guessed, because a
    wrong timestamp would silently introduce look-ahead bias downstream.
    """
    root = ET.fromstring(xml_text)
    items: list[NewsItem] = []
    channel = root.find("channel")
    if channel is not None:  # RSS 2.0
        feed_name = source or _text(channel.find("title")) or "rss"
        for node in channel.findall("item"):
            date = _text(node.find("pubDate"))
            title = _text(node.find("title"))
            if not date or not title:
                continue
            record = {
                "id": _text(node.find("guid")) or None,
                "title": title,
                "summary": _text(node.find("description")),
                "timestamp": date,
                "source": feed_name,
            }
            items.append(item_from_record(record))
        return items
    if root.tag == f"{_ATOM_NS}feed":
        feed_name = source or _text(root.find(f"{_ATOM_NS}title")) or "atom"
        for node in root.findall(f"{_ATOM_NS}entry"):
            date = _text(node.find(f"{_ATOM_NS}published")) or _text(
                node.find(f"{_ATOM_NS}updated")
            )
            title = _text(node.find(f"{_ATOM_NS}title"))
            if not date or not title:
                continue
            record = {
                "id": _text(node.find(f"{_ATOM_NS}id")) or None,
                "title": title,
                "summary": _text(node.find(f"{_ATOM_NS}summary")),
                "timestamp": date,
                "source": feed_name,
            }
            items.append(item_from_record(record))
        return items
    raise ValueError(f"unsupported feed root element: {root.tag}")


def fetch_rss(url: str, timeout: float = 15.0, source: str | None = None) -> list[NewsItem]:
    """Download and parse an RSS/Atom feed (network access required)."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http(s) feed URLs are supported")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return parse_rss(body, source=source)


def load_news(path_or_url: str | Path) -> list[NewsItem]:
    """Load news from a ``.json``/``.csv``/``.xml`` file or an http(s) RSS URL."""
    text = str(path_or_url)
    if text.startswith(("http://", "https://")):
        return fetch_rss(text)
    path = Path(text)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix == ".csv":
        return load_csv(path)
    if suffix in {".xml", ".rss", ".atom"}:
        return parse_rss(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported news file type: {suffix!r}")


def deduplicate(
    items: Iterable[NewsItem], window: timedelta = timedelta(hours=24)
) -> list[NewsItem]:
    """Drop repeated items, keeping the earliest copy.

    An item is a duplicate if its id was already seen, or if the same source
    published the same title (case-insensitive) less than ``window`` earlier.
    Recurring headlines (e.g. a daily "what to watch" column) are kept.
    """
    seen_ids: set[str] = set()
    last_seen: dict[tuple[str, str], datetime] = {}
    out: list[NewsItem] = []
    for item in sorted(items, key=lambda i: i.timestamp):
        key = (item.source.lower(), item.title.lower())
        prev = last_seen.get(key)
        if item.id in seen_ids or (prev is not None and item.timestamp - prev < window):
            continue
        seen_ids.add(item.id)
        last_seen[key] = item.timestamp
        out.append(item)
    return out


def save_json(items: Iterable[NewsItem], path: str | Path) -> None:
    """Write news items to a JSON file (list of records)."""
    payload = [i.to_dict() for i in items]
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
