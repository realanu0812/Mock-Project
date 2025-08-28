import os
import re
import json
import argparse
from urllib.parse import urlparse, parse_qs, urlsplit
from datetime import datetime, timedelta, timezone

import requests
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparse
from pathlib import Path

# =========================
# CONFIG (defaults; CLI can override some)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "blogs.json"
STATE_FILE = DATA_DIR / "state_blogs.json"

# Sites to track (Google News RSS per site)
SITES = [
    "healthline.com",
    "medicalnewstoday.com",
    "bodybuilding.com",
    "muscleandstrength.com",
    "barbend.com",
    "precisionnutrition.com",
    "eatthis.com",
    "examine.com",
    "menshealth.com",
    "verywellfit.com",
    "self.com",
    "shape.com",
    "livestrong.com",
    "myfitnesspal.com/blog",
    "t-nation.com",
    "nutrition.org",
    "authoritynutrition.com",
    "foodnetwork.com/healthyeats",
]

# Broad, protein-focused query for Google News
QUERY = '(protein OR whey OR casein OR "protein powder" OR "protein shake" OR "protein bar" OR collagen OR "amino acid" OR BCAA OR EAA OR leucine OR "high-protein")'

# Filters
APPLY_KEYWORD_FILTER = True
APPLY_DATE_FILTER = True          # applies iff date exists
KEEP_UNDATED = True               # keep items without date (flag)

# Defaults (can be overridden via CLI)
DEFAULT_DAYS_BACK = int(os.getenv("BLOG_DAYS_BACK", "90"))
DEFAULT_TOTAL_CAP = int(os.getenv("BLOG_TOTAL_CAP", "50"))
DEFAULT_PER_DOMAIN_CAP = int(os.getenv("BLOG_PER_DOMAIN_CAP", "15"))

# Keyword list (used on title/summary)
KEYWORDS = [
    "protein","whey","casein","pea protein","soy protein","plant protein",
    "collagen","amino acid","amino acids","bcaa","eaa","leucine",
    "high-protein","protein powder","protein shake","protein bar","muscle"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NutritionBot/1.1; +https://example.com/bot)"}
# =========================


# ---------- state / io ----------
def load_json_list(p: Path) -> list:
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_json_list(p: Path, data: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_state(obj: dict) -> None:
    # Save state in simple format: last_run_iso, added, total
    state = {
        "last_run_iso": obj.get("last_run_iso", iso_now()),
        "added": obj.get("added", 0),
        "total": obj.get("total", 0)
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- time helpers ----------
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def parse_dt(dt_str):
    if not dt_str:
        return None
    try:
        dt = dateparse.parse(dt_str)
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


# ---------- scraping helpers ----------
def google_news_feed(site: str) -> str:
    # Example: https://news.google.com/rss/search?q=(protein...)+site:healthline.com&hl=en&gl=US&ceid=US:en
    query = f"{QUERY} site:{site}"
    return (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(query)}"
        "&hl=en&gl=US&ceid=US:en"
    )

def is_google_news_url(url: str) -> bool:
    return urlparse(url).netloc.endswith("news.google.com")

def extract_original_from_gnews(link: str) -> str:
    """
    Google News RSS sometimes gives links like:
    https://news.google.com/rss/articles/...?...&url=https://example.com/article&...
    We try to pull the 'url' param; otherwise return original.
    """
    try:
        qs = parse_qs(urlsplit(link).query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]
        m = re.search(r"[?&]url=(https?://[^&]+)", link)
        if m:
            return requests.utils.unquote(m.group(1))
    except Exception:
        pass
    return link

def contains_keywords(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in KEYWORDS)

def fetch_feed(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return feedparser.parse(r.content)
    except Exception as e:
        print(f"[FEEDERR] {url} → {e}")
        return feedparser.parse(b"")


# ---------- core processing ----------
def process_feed(
    feed_url: str,
    global_bucket: list,
    totals: dict,
    per_domain_counts: dict,
    cutoff_dt: datetime | None,
    total_cap: int,
    per_domain_cap: int,
):
    fp = fetch_feed(feed_url)
    entries = fp.entries or []
    kept = 0
    print(f"[FEED] {feed_url} → {len(entries)} entries")

    for idx, e in enumerate(entries, start=1):
        if len(global_bucket) >= total_cap:
            break

        raw_link = (e.get("link") or "").strip()
        if not raw_link:
            # print(f"[DROP] #{idx}: no link")
            continue

        link = extract_original_from_gnews(raw_link) if is_google_news_url(raw_link) else raw_link
        domain = urlparse(link).netloc

        title = (e.get("title") or "").strip()
        summary = BeautifulSoup(e.get("summary", ""), "html.parser").get_text(" ", strip=True)

        pub_dt = None
        for k in ("published", "updated", "pubDate"):
            pub_dt = parse_dt(e.get(k))
            if pub_dt:
                break

        # keyword filter (on title/summary)
        if APPLY_KEYWORD_FILTER and not (contains_keywords(title) or contains_keywords(summary)):
            continue

        # incremental date cutoff if we have dates
        if cutoff_dt is not None and pub_dt is not None:
            if pub_dt <= cutoff_dt:
                # older or equal to last run → skip
                continue

        # date filter window only when we don't have a cutoff (i.e., first run)
        if cutoff_dt is None and APPLY_DATE_FILTER:
            cutoff_window = datetime.now(timezone.utc) - timedelta(days=DEFAULT_DAYS_BACK)
            if pub_dt is not None and pub_dt < cutoff_window:
                continue
            if pub_dt is None and not KEEP_UNDATED:
                continue

        # per-domain cap for THIS run
        per_domain_counts.setdefault(domain, 0)
        if per_domain_counts[domain] >= per_domain_cap:
            continue

        item = {
            "title": title,
            "url": link,
            "domain": domain,
            "published_at": pub_dt.isoformat() if pub_dt else None,
            "date_status": "known" if pub_dt else "unknown",
            "combined_text": f"{title}\n\n{summary}" if summary else title,
            "source": "blogs",
            "source_type": "blog_article",
            "is_verified": False
        }

        global_bucket.append(item)
        per_domain_counts[domain] += 1
        kept += 1

        if len(global_bucket) >= total_cap:
            break

    totals[feed_url] = {"entries": len(entries), "kept": kept}


# ---------- merge / dedupe ----------
def dedupe_merge(existing: list, new_items: list) -> list:
    """
    Dedupe by URL (stable: first occurrence wins).
    """
    seen = set()
    out = []
    for it in existing + new_items:
        url = (it.get("url") or "").strip()
        key = ("url", url)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=str, default=None, help="Incremental cutoff ISO-8601 (e.g., 2025-08-01T00:00:00Z). If absent, uses state; else falls back to --days.")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS_BACK, help="Days back for first run (no state & no --since).")
    ap.add_argument("--total-cap", type=int, default=DEFAULT_TOTAL_CAP, help="Global cap per run.")
    ap.add_argument("--per-domain-cap", type=int, default=DEFAULT_PER_DOMAIN_CAP, help="Per-domain cap per run.")
    ap.add_argument("--sites", nargs="*", default=SITES, help="Override sites list.")
    ap.add_argument("--full", action="store_true", help="Ignore state & since; run full (uses --days window).")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine incremental cutoff
    cutoff_dt = None
    state = load_state()
    if not args.full:
        since = args.since or state.get("last_run_iso")
        cutoff_dt = parse_iso(since)

    print("=== BLOGS RUN (Google News RSS + Incremental) ===")
    if cutoff_dt:
        print(f"Incremental cutoff: {cutoff_dt.isoformat()}")
    else:
        print(f"Window: last {args.days} days (no cutoff/state)")
    print(f"TOTAL_CAP={args.total_cap} | PER_DOMAIN_CAP={args.per_domain_cap}")
    print("Sites:", ", ".join(args.sites))

    totals = {}
    run_bucket = []
    per_domain_counts = {}

    # Pull per-site Google News feeds
    feed_urls = [google_news_feed(site) for site in args.sites]
    for feed_url in feed_urls:
        if len(run_bucket) >= args.total_cap:
            break
        process_feed(
            feed_url=feed_url,
            global_bucket=run_bucket,
            totals=totals,
            per_domain_counts=per_domain_counts,
            cutoff_dt=cutoff_dt,
            total_cap=args.total_cap,
            per_domain_cap=args.per_domain_cap,
        )

    # Append + dedupe with existing file
    existing = load_json_list(OUTPUT_FILE)
    merged = dedupe_merge(existing, run_bucket)
    save_json_list(OUTPUT_FILE, merged)

    # State update
    save_state({
        "last_run_iso": iso_now(),
        "added": len(run_bucket),
        "total": len(merged),
        "sites": args.sites,
        "mode": "incremental" if cutoff_dt else f"window_{args.days}_days"
    })

    # summary
    print("\n=== SUMMARY ===")
    for k, v in totals.items():
        print(f"{k}: {v['kept']} kept / {v.get('entries', 0)} entries")
    print(f"TOTAL NEW THIS RUN: {len(run_bucket)}")
    print(f"TOTAL AFTER MERGE: {len(merged)} → saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()