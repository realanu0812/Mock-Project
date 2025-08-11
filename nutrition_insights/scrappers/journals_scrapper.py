import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import requests
from xml.etree import ElementTree as ET

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "journals.json")

# Window (months back)
MONTHS_BACK = 6  # was 4

# Query: protein-focused
QUERY = 'protein AND (nutrition OR diet OR supplement OR "protein powder" OR whey OR casein OR leucine OR "amino acid")'

# How many total to pull (IDs) from PubMed search
RETMAX = 500  # increase if you want more than 500

# Batch size for efetch (IDs per request). 100–200 is safe & fast.
PAGE_SIZE = 200  # was 100

# Rate limits:
# - Without API key: ~3 req/sec total across E-utilities
# - With API key: up to 10 req/sec
# We'll be nice and sleep a little between requests.
SLEEP_SEC = 0.2

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


def utc_now():
    return datetime.now(timezone.utc)

def date_range_months_back(months: int):
    end = utc_now()
    start = end - timedelta(days=months * 30)
    return start.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d")

def _base_params():
    p = {}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    return p

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
    r = requests.get(ESEARCH, params=params, timeout=20)
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
    r = requests.get(EFETCH, params=params, timeout=40)
    r.raise_for_status()
    return r.text

def text_or_none(el):
    return el.text.strip() if el is not None and el.text else None

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

        # Pub date (best-effort)
        pub = article.find("Journal/JournalIssue/PubDate") if article is not None else None
        year = text_or_none(pub.find("Year")) if pub is not None else None
        month = text_or_none(pub.find("Month")) if pub is not None else None
        day = text_or_none(pub.find("Day")) if pub is not None else None

        pub_iso = None
        if year:
            m = month if month else "01"
            d = day if day else "01"
            # Try common formats (e.g., "Jan" or "01")
            dt = None
            for fmt in ("%Y-%b-%d", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(f"{year}-{m}-{d}", fmt)
                    break
                except Exception:
                    pass
            if dt:
                pub_iso = dt.replace(tzinfo=timezone.utc).isoformat()

        combined_text = ((title or "").strip() + "\n\n" + abstract).strip()

        # Protein filter + substance filter
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

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    start_date, end_date = date_range_months_back(MONTHS_BACK)
    print(f"[PubMed] Window: {start_date} → {end_date}")

    # 1) Search IDs
    pmids = esearch_pmids(QUERY, start_date, end_date, RETMAX)
    print(f"[PubMed] Found {len(pmids)} PMIDs (cap {RETMAX})")

    # 2) Fetch in PAGE_SIZE batches
    all_records: List[Dict[str, Any]] = []
    for i in range(0, len(pmids), PAGE_SIZE):
        batch = pmids[i:i+PAGE_SIZE]
        if not batch:
            break
        xml_text = efetch_xml(batch)
        recs = parse_pubmed_xml(xml_text)
        all_records.extend(recs)
        print(f"[PubMed] Parsed {len(recs)} records (total {len(all_records)})")
        time.sleep(SLEEP_SEC)  # be polite (lower if you set API key)

        if len(all_records) >= RETMAX:
            break

    # 3) Dedup by PMID
    uniq = {r["pmid"]: r for r in all_records if r.get("pmid")}
    final = list(uniq.values())
    print(f"[PubMed] Final unique records: {len(final)}")

    # 4) Save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(final)} PubMed articles to {OUT_PATH}")

if __name__ == "__main__":
    main()