# nutrition_insights/scrappers/journals_scrapper.py
"""
PubMed journals scraper (protein-focused) with append + incremental mode.

- Appends to data/journals.json
- Dedupes by PMID
- Incremental:
    * --since ISO-8601 (preferred)
    * else data/state_journals.json:last_run_iso
    * else --months N (default 6)

Env (optional):
  NCBI_API_KEY, NCBI_EMAIL
  PUBMED_MONTHS_BACK (default 6)
  PUBMED_RETMAX (default 500)
  PUBMED_PAGE_SIZE (default 200)

Examples:
  python scrappers/journals_scrapper.py --since 2025-06-01T00:00:00Z --retmax 400
  python scrappers/journals_scrapper.py --months 6
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import requests
from xml.etree import ElementTree as ET
from pathlib import Path

# -----------------------------
# Config / Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = DATA_DIR / "journals.json"
STATE_PATH = DATA_DIR / "state_journals.json"

# Defaults (can be overridden by ENV or CLI)
DEFAULT_MONTHS_BACK = int(os.getenv("PUBMED_MONTHS_BACK", "6"))
DEFAULT_RETMAX = int(os.getenv("PUBMED_RETMAX", "500"))
DEFAULT_PAGE_SIZE = int(os.getenv("PUBMED_PAGE_SIZE", "200"))
SLEEP_SEC = 0.2  # polite pause between efetch pages

# Query: protein-focused (keep yours)
QUERY = 'protein AND (nutrition OR diet OR supplement OR "protein powder" OR whey OR casein OR leucine OR "amino acid")'

# Trusted source flags
SOURCE_TYPE = "pubmed_article"
IS_VERIFIED = True
MIN_ABSTRACT_CHARS = 200

# PubMed E-utilities
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Optional: set your NCBI API key & contact email (recommended)
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")   # export NCBI_API_KEY=your_key
NCBI_EMAIL   = os.getenv("NCBI_EMAIL", "")     # export NCBI_EMAIL=you@domain.com

# -----------------------------
# Helpers
# -----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def iso_now() -> str:
    return utc_now().isoformat()

def to_date_str(d: datetime) -> str:
    """PubMed E-utilities expects YYYY/MM/DD for mindate/maxdate."""
    return d.strftime("%Y/%m/%d")

def parse_iso(s: str) -> Optional[datetime]:
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def load_json_list(p: Path) -> list:
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_state(obj: dict) -> None:
    save_json(STATE_PATH, obj)

def _base_params():
    p = {}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    return p

# -----------------------------
# PubMed calls
# -----------------------------
def esearch_pmids(term: str, start_date: str, end_date: str, retmax: int) -> List[str]:
    params = {
        **_base_params(),
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "mindate": start_date,
        "maxdate": end_date,
        "datetype": "pdat",   # publication date
        "sort": "most+recent"
    }
    r = requests.get(ESEARCH, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    idlist = data.get("esearchresult", {}).get("idlist", [])
    return idlist

def efetch_xml(pmids: List[str]) -> str:
    params = {
        **_base_params(),
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
    }
    r = requests.get(EFETCH, params=params, timeout=60)
    r.raise_for_status()
    return r.text

def text_or_none(el):
    return el.text.strip() if el is not None and el.text else None

def _parse_pubdate(article) -> Optional[str]:
    # Try Journal pub date first
    pub = article.find("Journal/JournalIssue/PubDate") if article is not None else None
    year = text_or_none(pub.find("Year")) if pub is not None else None
    month = text_or_none(pub.find("Month")) if pub is not None else None
    day = text_or_none(pub.find("Day")) if pub is not None else None
    if not year:
        return None
    m = month or "01"
    d = day or "01"
    dt = None
    # common formats (e.g., "2024-Aug-15" or "2024-08-15")
    for fmt in ("%Y-%b-%d", "%Y-%m-%d", "%Y-%b", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(f"{year}-{m}-{d}", fmt if "%d" in fmt else fmt)
            break
        except Exception:
            continue
    if not dt:
        # last resort: just year-month-day normalized
        try:
            dt = datetime.strptime(f"{year}-{m}-{d}", "%Y-%m-%d")
        except Exception:
            return None
    return dt.replace(tzinfo=timezone.utc).isoformat()

def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    records = []

    for node in root.findall(".//PubmedArticle"):
        medline = node.find("MedlineCitation")
        article = medline.find("Article") if medline is not None else None

        pmid_el = medline.find("PMID") if medline is not None else None
        pmid = text_or_none(pmid_el)

        title = text_or_none(article.find("ArticleTitle")) if article is not None else None
        journal = text_or_none(article.find("Journal/Title")) if article is not None else None

        # Abstract
        abstract_parts = []
        if article is not None:
            for ab in article.findall("Abstract/AbstractText"):
                if ab.text:
                    abstract_parts.append(ab.text.strip())
        abstract = "\n\n".join(abstract_parts).strip() if abstract_parts else ""

        pub_iso = _parse_pubdate(article)

        combined_text = ((title or "").strip() + "\n\n" + abstract).strip()

        # Protein filter + minimal content
        if "protein" not in combined_text.lower():
            continue
        if len(abstract) < MIN_ABSTRACT_CHARS:
            if not (title and "protein" in title.lower()):
                continue

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

        records.append({
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "url": url,
            "domain": "pubmed.ncbi.nlm.nih.gov",
            "published_at": pub_iso,                   # may be None
            "date_status": "known" if pub_iso else "unknown",
            "combined_text": combined_text,
            "source_type": SOURCE_TYPE,
            "is_verified": IS_VERIFIED
        })

    return records

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=str, default=None, help="ISO-8601 UTC (e.g., 2025-06-01T00:00:00Z)")
    ap.add_argument("--months", type=int, default=DEFAULT_MONTHS_BACK, help="Window size if no --since/state")
    ap.add_argument("--retmax", type=int, default=DEFAULT_RETMAX, help="Max PMIDs to fetch from esearch")
    ap.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="IDs per efetch request (100-200 good)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Decide start/end window
    state = load_state()
    since_iso = args.since or state.get("last_run_iso")
    if since_iso:
        since_dt = parse_iso(since_iso)
        if not since_dt:
            since_dt = utc_now() - timedelta(days=args.months * 30)
    else:
        since_dt = utc_now() - timedelta(days=args.months * 30)

    end_dt = utc_now()

    start_date = to_date_str(since_dt)
    end_date   = to_date_str(end_dt)
    print(f"[PubMed] Window: {start_date} → {end_date}  ({args.months} months back fallback)")

    # Existing data for append + dedupe
    existing = load_json_list(OUT_PATH)
    existing_by_pmid = {r.get("pmid"): r for r in existing if r.get("pmid")}

    # Search IDs
    pmids = esearch_pmids(QUERY, start_date, end_date, args.retmax)
    print(f"[PubMed] Found {len(pmids)} PMIDs (cap {args.retmax})")

    # Remove PMIDs we already have to avoid refetch
    pmids_to_fetch = [p for p in pmids if p not in existing_by_pmid]
    print(f"[PubMed] New PMIDs to fetch: {len(pmids_to_fetch)} (skipped {len(pmids) - len(pmids_to_fetch)} known)")

    # Fetch in batches
    all_new: List[Dict[str, Any]] = []
    for i in range(0, len(pmids_to_fetch), args.page_size):
        batch = pmids_to_fetch[i:i+args.page_size]
        if not batch:
            break
        xml_text = efetch_xml(batch)
        recs = parse_pubmed_xml(xml_text)
        all_new.extend(recs)
        print(f"[PubMed] Parsed {len(recs)} records (new total {len(all_new)})")
        time.sleep(SLEEP_SEC)
        # Hard stop if we already reached retmax worth of *new* records
        if len(all_new) >= args.retmax:
            break

    # Merge + dedupe (prefer existing, then add new)
    merged = {**existing_by_pmid}
    for r in all_new:
        pmid = r.get("pmid")
        if not pmid:
            continue
        if pmid not in merged:
            merged[pmid] = r

    # Sort (newest first when available)
    def sort_key(x):
        d = x[1].get("published_at")
        try:
            return parse_iso(d).timestamp() if d else 0.0
        except Exception:
            return 0.0

    final_list = [v for _, v in sorted(merged.items(), key=sort_key, reverse=True)]

    save_json(OUT_PATH, final_list)
    print(f"✅ Saved {len(all_new)} new | total={len(final_list)} → {OUT_PATH}")

    # Update state (use now, not last pub date)
    save_state({
        "last_run_iso": iso_now(),
        "last_window": {"start": start_date, "end": end_date},
        "added": len(all_new),
        "total": len(final_list)
    })

if __name__ == "__main__":
    main()