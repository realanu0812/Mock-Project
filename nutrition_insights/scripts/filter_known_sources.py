import json
import sys
from pathlib import Path

# Usage: python filter_known_sources.py <input_json> <output_json>
if len(sys.argv) < 3:
    print("Usage: python filter_known_sources.py <input_json> <output_json>")
    sys.exit(1)

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Only keep records with recognized sources/types
known = []
for r in data:
    src = str(r.get("source", "")).strip().lower()
    src_type = str(r.get("source_type", "")).strip().lower()
    url = str(r.get("url", "")).lower()
    is_blog = src in ["blogs", "blog", "blogger"] or "blog" in url or src_type == "blog_article"
    is_reddit = src == "reddit" and src_type == "reddit_post" or "reddit.com" in url
    is_journal = src == "journals" and src_type == "journal_article" or any(x in url for x in ["pubmed", "journal"])
    if is_blog or is_reddit or is_journal:
        known.append(r)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(known, f, ensure_ascii=False, indent=2)

print(f"Filtered {len(known)} known records out of {len(data)} total.")
