# nutrition_insights/scrappers/reddit_scrapper.py
"""
Reddit scraper with append + incremental mode.

Features
- Uses --since ISO-8601 (preferred) or --days (default 30) to fetch only fresh posts
- Appends to data/reddit.json instead of replacing, with robust dedupe
- Stores last_run_iso in data/state_reddit.json for the next incremental run
- Wider protein-focused keywords; multiple relevant subreddits
- Captures top comments (unverified) and builds combined_text for RAG

Credentials (ENV)
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT   (e.g., "nutrition-insights/1.0 by yourname")

Example:
  python scrappers/reddit_scrapper.py --since 2025-08-01T00:00:00Z --limit 60 --comments 8
  python scrappers/reddit_scrapper.py --days 30
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import praw

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_FILE = DATA_DIR / "reddit.json"
STATE_FILE = DATA_DIR / "state_reddit.json"

# ---------- Defaults ----------
DEFAULT_DAYS = int(os.getenv("REDDIT_DAYS_BACK", "30"))
POST_LIMIT = int(os.getenv("REDDIT_POST_CAP", "50"))
COMMENTS_LIMIT = int(os.getenv("REDDIT_COMMENTS_LIMIT", "10"))

SUBREDDITS = [
    "nutrition", "fitness", "supplements", "bodybuilding",
    "veganfitness", "xxfitness"
]

KEYWORDS = [
    # core
    "protein", "whey", "casein", "collagen",
    "pea protein", "soy protein", "plant protein",
    "rice protein", "hemp protein",
    # products/uses
    "protein powder", "protein shake", "protein bar", "clear whey",
    # science/quality
    "amino acid", "amino acids", "eaa", "bcaa", "leucine",
    "diaas", "pdcaas", "bioavailability", "digestibility",
    # outcomes/contexts
    "high-protein", "high protein", "muscle recovery", "hypertrophy",
    "muscle protein synthesis", "mps", "athletes", "sarcopenia"
]

# ---------- Utilities ----------
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_json(path: Path, arr: list):
    path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_state(obj: dict):
    save_json(STATE_FILE, obj)

def parse_since(since: str | None, days: int) -> datetime:
    """Return a timezone-aware UTC cutoff datetime."""
    if since:
        try:
            s = since.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc) - timedelta(days=days)

def contains_keyword(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in KEYWORDS)

def dedupe_key(item: dict):
    # Prefer unique Reddit permalink if available; else URL fallback
    if item.get("permalink"):
        return ("permalink", item["permalink"])
    return ("url", item.get("url", ""))

def merge_append(existing: list, new_items: list) -> list:
    """Append and dedupe by permalink/url, preserving first occurrence."""
    seen = set()
    out = []
    for it in existing + new_items:
        k = dedupe_key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

# ---------- Scraper ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=str, default=None, help="ISO-8601 UTC time (e.g., 2025-08-01T00:00:00Z)")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Fallback window (days) if --since missing")
    ap.add_argument("--limit", type=int, default=POST_LIMIT, help="Max posts to collect (global across subs)")
    ap.add_argument("--comments", type=int, default=COMMENTS_LIMIT, help="Top comments per post to include")
    ap.add_argument("--subs", type=str, nargs="*", default=SUBREDDITS, help="Override subreddits list")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine cutoff (prefer CLI --since, else last state, else --days)
    state = load_state()
    since_iso = args.since or state.get("last_run_iso")
    cutoff_dt = parse_since(since_iso, args.days)
    cutoff_ts = cutoff_dt.timestamp()

    # Credentials via ENV (remove any hard-coded secrets)
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    uagent = os.getenv("REDDIT_USER_AGENT", "nutrition-insights/1.0 (reddit scraper)")

    if not cid or not csec:
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")

    reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=uagent)

    collected = []
    for sub in args.subs:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.new(limit=800):  # scan new; break early by cutoff
            if post.created_utc < cutoff_ts:
                break  # subreddit.new returns newest first

            title = post.title or ""
            body = post.selftext or ""
            if not (contains_keyword(title) or contains_keyword(body)):
                continue

            item = {
                "title": title,
                "url": getattr(post, "url", ""),
                "domain": "reddit.com",
                "permalink": f"https://www.reddit.com{post.permalink}",
                "source_type": "reddit_post",
                "is_verified": False,  # Phase 1 = unverified
                "score": int(getattr(post, "score", 0)),
                "created_utc": float(post.created_utc),
                "published_at": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
            }

            # Top comments (by score)
            if args.comments > 0:
                try:
                    post.comments.replace_more(limit=0)
                    comments = sorted(post.comments, key=lambda c: getattr(c, "score", 0), reverse=True)[: args.comments]
                    top_texts = []
                    for c in comments:
                        if not getattr(c, "body", None):
                            continue
                        txt = c.body.strip()
                        # Keep comments regardless of keyword; they are context
                        top_texts.append(txt)
                    if top_texts:
                        item["top_comments"] = top_texts
                except Exception:
                    # Ignore comment fetch issues
                    pass

            # Combined text for RAG
            combined_segments = [f"Title: {title}"]
            if body:
                combined_segments.append(f"Post: {body}")
            if item.get("top_comments"):
                combined_segments.append("Comments:\n" + "\n\n".join(item["top_comments"]))
            item["combined_text"] = "\n\n".join(combined_segments).strip()

            collected.append(item)
            if len(collected) >= args.limit:
                break

        if len(collected) >= args.limit:
            break

    existing = load_json(OUT_FILE)
    merged = merge_append(existing, collected)
    save_json(OUT_FILE, merged)
    print(f"✅ Reddit: +{len(collected)} new | total={len(merged)} → {OUT_FILE}")

    save_state({"last_run_iso": iso_now(), "added": len(collected), "total": len(merged)})

if __name__ == "__main__":
    main()