import praw
from datetime import datetime, timedelta, timezone
import json
import os

# ====== CONFIG ======
CLIENT_ID = "frUhUqOJkJTZrnhQvtlV_w"
CLIENT_SECRET = "feaKgAXQ2hIWh7AfIi67SO5FpFCSyg"
USER_AGENT = "protein_scrapper"

KEYWORDS = [
    "protein", "whey", "casein", "plant protein",
    "protein powder", "protein shake", "protein bar"
]
DAYS_BACK = 30
POST_LIMIT = 50
COMMENTS_LIMIT = 10
DATA_FOLDER = "data"
OUTPUT_FILE = os.path.join(DATA_FOLDER, "reddit.json")
# ====================


def contains_keyword(text: str) -> bool:
    """Check if text contains any keyword (case-insensitive)."""
    text_lower = text.lower()
    return any(k in text_lower for k in KEYWORDS)


def scrape_reddit(topic="nutrition", days_back=DAYS_BACK, limit=POST_LIMIT):
    # Make sure data directory exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

    subreddit = reddit.subreddit(topic)
    cutoff_timestamp = (datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp()

    posts_data = []
    for post in subreddit.new(limit=500):
        if post.created_utc < cutoff_timestamp:
            continue

        # Filter by keywords in title or body
        if not (contains_keyword(post.title) or contains_keyword(post.selftext)):
            continue

        # Collect top comments that match keywords
        post.comments.replace_more(limit=0)
        top_comments = []
        for c in sorted(post.comments, key=lambda x: x.score, reverse=True)[:COMMENTS_LIMIT]:
            if contains_keyword(c.body):
                top_comments.append({
                    "text": c.body.strip(),
                    "source_type": "reddit_comment",
                    "is_verified": False  # unverified at this stage
                })

        combined_text = f"Title: {post.title}\n\nPost: {post.selftext}\n\nComments:\n" + \
                        "\n\n".join([c["text"] for c in top_comments])

        posts_data.append({
            "title": post.title,
            "url": post.url,
            "selftext": post.selftext,
            "source_type": "reddit_post",  # type of data
            "is_verified": False,          # unverified at this stage
            "comments": top_comments,
            "combined_text": combined_text,
            "score": post.score,
            "created_utc": post.created_utc,
            "created_at": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat()
        })

        if len(posts_data) >= limit:
            break

    # Save as JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(posts_data, f, indent=2, ensure_ascii=False)

    print(f"[Reddit] Saved {len(posts_data)} protein-related posts to {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape_reddit()