import os
import re
import json
from urllib.parse import urlparse, parse_qs, urlsplit
from datetime import datetime, timedelta, timezone

import requests
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparse

# =========================
# CONFIG
# =========================
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "blogs.json")

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

# Query terms (broad protein-related)
QUERY = "(protein OR whey OR casein OR \"protein powder\" OR \"protein shake\" OR \"protein bar\" OR collagen OR \"amino acid\" OR BCAA OR EAA OR leucine OR \"high-protein\")"

# Filters
APPLY_KEYWORD_FILTER = True
APPLY_DATE_FILTER = True          # applies iff date exists
KEEP_UNDATED = True               # keep items without date (flag)
DAYS_BACK = 90

# Caps
TOTAL_CAP = 50
PER_DOMAIN_CAP = 15

# Keyword list (used on title/summary)
KEYWORDS = [
    "protein","whey","casein","pea protein","soy protein","plant protein",
    "collagen","amino acid","amino acids","bcaa","eaa","leucine",
    "high-protein","protein powder","protein shake","protein bar","muscle"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NutritionBot/1.0)"}
# =========================


def google_news_feed(site: str) -> str:
    # Example: https://news.google.com/rss/search?q=(protein...)+site:healthline.com&hl=en&gl=US&ceid=US:en
    return (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(QUERY + ' site:' + site)}"
        "&hl=en&gl=US&ceid=US:en"
    )

def is_google_news_url(url: str) -> bool:
    return urlparse(url).netloc.endswith("news.google.com")

def extract_original_from_gnews(link: str) -> str:
    """
    Google News RSS sometimes gives links like:
    https://news.google.com/rss/articles/CBMi...?...&url=https://example.com/article&...
    We try to pull the 'url' param; otherwise return original.
    """
    try:
        # Some links are direct; some carry url= param.
        qs = parse_qs(urlsplit(link).query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]
        # Some variants embed the target within the path; attempt a regex:
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

def in_window_if_dated(pub_dt):
    if pub_dt is None:
        return KEEP_UNDATED
    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    return pub_dt >= cutoff

def fetch_feed(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        return feedparser.parse(r.content)
    except Exception as e:
        print(f"[FEEDERR] {url} → {e}")
        return feedparser.parse(b"")

def process_feed(feed_url: str, global_bucket: list, totals: dict):
    fp = fetch_feed(feed_url)
    entries = fp.entries or []
    kept = 0
    print(f"[FEED] {feed_url} → {len(entries)} entries")

    per_domain_counts = {}

    for idx, e in enumerate(entries, start=1):
        if len(global_bucket) >= TOTAL_CAP:
            break

        raw_link = (e.get("link") or "").strip()
        if not raw_link:
            print(f"[DROP] #{idx}: no link")
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
            # Debug line to see what's getting dropped:
            # print(f"[DROP] #{idx}: keyword fail — title='{title[:60]}'")
            continue

        # date filter (only if date present)
        if APPLY_DATE_FILTER:
            if pub_dt is not None and not in_window_if_dated(pub_dt):
                # print(f"[DROP] #{idx}: out of window — {pub_dt.isoformat()}")
                continue
            if pub_dt is None and not KEEP_UNDATED:
                # print(f"[DROP] #{idx}: no date and KEEP_UNDATED=False")
                continue

        # per-domain cap
        per_domain_counts.setdefault(domain, 0)
        if per_domain_counts[domain] >= PER_DOMAIN_CAP:
            # print(f"[DROP] #{idx}: per-domain cap {domain}")
            continue

        item = {
            "title": title,
            "url": link,
            "domain": domain,
            "published_at": pub_dt.isoformat() if pub_dt else None,
            "date_status": "known" if pub_dt else "unknown",
            "combined_text": f"{title}\n\n{summary}" if summary else title,
            "source_type": "blog_article",
            "is_verified": False
        }

        global_bucket.append(item)
        per_domain_counts[domain] += 1
        kept += 1

        if len(global_bucket) >= TOTAL_CAP:
            break

    totals[feed_url] = {"entries": len(entries), "kept": kept}

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    totals = {}
    all_items = []

    print("=== GOOGLE NEWS RSS RUN ===")
    print(f"DAYS_BACK={DAYS_BACK}, TOTAL_CAP={TOTAL_CAP}, PER_DOMAIN_CAP={PER_DOMAIN_CAP}")
    print("Sites:", *SITES, sep="\n  - ")

    feed_urls = [google_news_feed(site) for site in SITES]
    for feed_url in feed_urls:
        if len(all_items) >= TOTAL_CAP:
            break
        process_feed(feed_url, all_items, totals)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in totals.items():
        print(f"{k}: {v['kept']} kept / {v['entries']} entries")
    print(f"TOTAL KEPT: {len(all_items)} → saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()