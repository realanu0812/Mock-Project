# nutrition_insights/scrappers/journals_scrapper.py
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

"""
PubMed scraper (robust):
- ESearch (JSON) to find PMIDs, with retries + date-window splitting on timeouts
- ESummary (JSON) to get title, pubdate, etc.
- EFetch (XML) ONLY for abstract text (JSON isn't supported for PubMed abstracts)
- Incremental: --since ISO; appends and de-dupes by id=pmid:<id>
- Polite rate limiting + optional NCBI_EMAIL / NCBI_API_KEY
- is_verified=True for journals
"""

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTFILE = DATA_DIR / "journals.json"

ESEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

QUERY = (
    '((protein[tiab] OR "protein intake"[tiab] OR "dietary protein"[tiab] OR high-protein[tiab] '
    'OR "protein supplementation"[tiab] OR "protein supplement"[tiab] OR whey[tiab] OR casein[tiab] '
    'OR "milk protein"[tiab] OR collagen[tiab] OR "plant protein"[tiab] OR "soy protein"[tiab] '
    'OR "pea protein"[tiab] OR "rice protein"[tiab] OR "hemp protein"[tiab] OR "amino acid"[tiab] '
    'OR "essential amino acid"[tiab] OR EAA[tiab] OR BCAA[tiab] OR leucine[tiab]) '
    'OR Dietary Proteins[mh] OR Protein Supplementation[mh] OR Whey Proteins[mh] OR Caseins[mh] '
    'OR Amino Acids, Essential[mh]) '
    'AND (nutrition[tiab] OR diet[tiab] OR dietary[tiab] OR "sports nutrition"[tiab] OR exercise[tiab] '
    'OR training[tiab] OR "resistance training"[tiab] OR muscle[tiab] OR "muscle protein synthesis"[tiab] '
    'OR MPS[tiab] OR hypertrophy[tiab] OR strength[tiab] OR recovery[tiab] OR performance[tiab] '
    'OR satiety[tiab] OR "weight loss"[tiab] OR "body composition"[tiab] OR "fat loss"[tiab] '
    'OR sarcopenia[tiab] OR "older adults"[tiab] OR aging[tiab] OR athletes[tiab] OR bioavailability[tiab] '
    'OR digestibility[tiab] OR DIAAS[tiab] OR PDCAAS[tiab] OR "protein quality"[tiab] OR "protein timing"[tiab] '
    'OR "protein distribution"[tiab] OR "per-meal protein"[tiab] OR "gut health"[tiab] OR microbiome[tiab] '
    'OR kidney[tiab] OR renal[tiab] OR safety[tiab] OR tolerability[tiab]) '
    'AND Humans[mh] NOT ("protein kinase"[tiab] OR kinase[tiab] OR phosphorylation[tiab] OR proteomics[tiab] '
    'OR "protein folding"[tiab] OR amyloid[tiab] OR prion[tiab] OR ubiquitin[tiab] OR "transcription factor"[tiab] '
    'OR "signal transduction"[tiab] OR oncogene[tiab])'
)

# ---------------- HTTP session with retries ----------------

def build_session(timeout_s: int = 45) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.timeout = timeout_s
    return s

def eutils_params(base: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(base)
    email = os.environ.get("NCBI_EMAIL")
    api_key = os.environ.get("NCBI_API_KEY")
    if email:   p["email"] = email
    if api_key: p["api_key"] = api_key
    return p

def polite_delay():
    # NCBI suggests ≤3 req/sec (higher with api_key). We stay conservative.
    time.sleep(0.35)

# ---------------- ESearch with window splitting ------------

def esearch_pmids(session: requests.Session, query: str, start: datetime, end: datetime, retmax: int) -> List[str]:
    params = eutils_params({
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "pub_date",
        "datetype": "pdat",
        "mindate": start.strftime("%Y/%m/%d"),
        "maxdate": end.strftime("%Y/%m/%d"),
    })
    polite_delay()
    try:
        r = session.get(ESEARCH, params=params, timeout=session.timeout)
        r.raise_for_status()
        data = r.json()
        idlist = data.get("esearchresult", {}).get("idlist", [])
        return list(dict.fromkeys(idlist))
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
        if (end - start).days <= 1:
            raise
        mid = start + (end - start) / 2
        mid = datetime.fromtimestamp(mid.timestamp(), tz=start.tzinfo)
        left  = esearch_pmids(session, query, start, mid, retmax)
        right = esearch_pmids(session, query, mid + timedelta(days=1), end, retmax)
        return list(dict.fromkeys(left + right))
    except requests.HTTPError:
        # If rate limited, back off and split once
        if r is not None and r.status_code in (429, 500, 502, 503, 504):
            time.sleep(3)
            if (end - start).days <= 1:
                raise
            mid = start + (end - start) / 2
            mid = datetime.fromtimestamp(mid.timestamp(), tz=start.tzinfo)
            left  = esearch_pmids(session, query, start, mid, retmax)
            right = esearch_pmids(session, query, mid + timedelta(days=1), end, retmax)
            return list(dict.fromkeys(left + right))
        raise

# ---------------- ESummary (JSON) --------------------------

def esummary_meta(session: requests.Session, pmids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Return {pmid: {title, pubdate, doi?}} via ESummary JSON.
    """
    meta: Dict[str, Dict[str, Any]] = {}
    if not pmids:
        return meta
    BATCH = 200
    for i in range(0, len(pmids), BATCH):
        chunk = pmids[i:i+BATCH]
        params = eutils_params({
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(chunk),
        })
        polite_delay()
        r = session.get(ESUMMARY, params=params, timeout=session.timeout)
        r.raise_for_status()
        data = r.json()
        # ESummary JSON: data["result"] has "uids" and uid objects
        res = data.get("result", {})
        for uid in res.get("uids", []):
            rec = res.get(uid, {})
            title = (rec.get("title") or "").strip()
            pubdate = (rec.get("pubdate") or rec.get("sortpubdate") or "").strip()
            eloc = rec.get("elocationid") or ""
            articleids = rec.get("articleids") or []
            doi = ""
            for a in articleids:
                if a.get("idtype") == "doi":
                    doi = a.get("value") or ""
                    break
            meta[uid] = {"title": title, "date": pubdate, "doi": doi, "elocationid": eloc}
    return meta

# ---------------- EFetch (XML) for abstracts --------------

def efetch_abstracts(session: requests.Session, pmids: List[str]) -> Dict[str, str]:
    """
    Return {pmid: abstract_text} by parsing PubmedArticleSet XML.
    """
    out: Dict[str, str] = {}
    if not pmids:
        return out
    BATCH = 200
    for i in range(0, len(pmids), BATCH):
        chunk = pmids[i:i+BATCH]
        params = eutils_params({
            "db": "pubmed",
            "retmode": "xml",          # XML is the stable format for PubMed abstracts
            "rettype": "abstract",
            "id": ",".join(chunk),
        })
        polite_delay()
        r = session.get(EFETCH, params=params, timeout=session.timeout)
        r.raise_for_status()
        xml_text = r.text

        # Parse XML defensively
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            # If something odd returned, skip this batch gracefully
            continue

        # Find abstracts:
        # /PubmedArticleSet/PubmedArticle/MedlineCitation/Article/Abstract/AbstractText
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//MedlineCitation/PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
            if not pmid:
                continue
            # An abstract can have multiple AbstractText nodes (with Label attributes)
            parts: List[str] = []
            for ab in art.findall(".//MedlineCitation/Article/Abstract/AbstractText"):
                txt = (ab.text or "").strip()
                label = ab.attrib.get("Label")
                if txt:
                    parts.append(f"{label}: {txt}" if label else txt)
            out[pmid] = "\n\n".join([p for p in parts if p])
    return out

# ---------------- IO helpers ------------------------------

def load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        arr = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(arr, list):
            return {str(x.get("id") or x.get("pmid")): x for x in arr if isinstance(x, dict)}
        return {}
    except Exception:
        return {}

def save_all(path: Path, records: List[Dict[str, Any]]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

def save_state(path: Path, last_run_iso: str, pmids: list):
    state = {
        "last_run_iso": last_run_iso,
        "added": len(pmids),
        "total": len(pmids)
    }
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Scrape PubMed protein-in-nutrition (incremental, robust).")
    ap.add_argument("--since", help="ISO timestamp for incremental cutoff (exclusive). If omitted, uses 6 months back.")
    ap.add_argument("--months-back", type=int, default=6, help="Fallback lookback if --since missing.")
    ap.add_argument("--retmax", type=int, default=500, help="ESearch max PMIDs to return.")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    if args.since:
        try:
            start = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        except Exception:
            print(f"[WARN] Bad --since format; using {args.months_back} months back.", file=sys.stderr)
            start = now - timedelta(days=30 * args.months_back)
    else:
        start = now - timedelta(days=30 * args.months_back)
    end = now

    print(f"[PubMed] Window: {start.strftime('%Y/%m/%d')} → {end.strftime('%Y/%m/%d')}  (fallback {args.months_back} months)")
    session = build_session(timeout_s=45)

    # 1) PMIDs
    try:
        pmids = esearch_pmids(session, QUERY, start, end, args.retmax)
    except Exception as e:
        print(f"[ERROR] ESearch failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[PubMed] Found {len(pmids)} PMIDs (cap {args.retmax})")

    # 2) Incremental skip
    existing = load_existing(OUTFILE)
    new_pmids = [p for p in pmids if f"pmid:{p}" not in existing]
    print(f"[PubMed] New PMIDs to fetch: {len(new_pmids)} (skipped {len(pmids) - len(new_pmids)} known)")

    # If nothing new, exit cleanly
    if not new_pmids:
        print(f"✅ Saved 0 new | total={len(existing)} → {OUTFILE}")
        return

    # 3) ESummary for metadata
    meta = esummary_meta(session, new_pmids)

    # 4) EFetch for abstracts (XML)
    abstracts = efetch_abstracts(session, new_pmids)

    # 5) Build records
    to_add: List[Dict[str, Any]] = []
    for p in new_pmids:
        m = meta.get(p, {})
        title = (m.get("title") or "").strip()
        date  = (m.get("date")  or "").strip()
        text  = abstracts.get(p, "").strip()
        url   = f"https://pubmed.ncbi.nlm.nih.gov/{p}/"
        rec = {
            "id": f"pmid:{p}",
            "pmid": p,
            "title": title,
            "date": date,
            "url": url,
            "source": "journals",
            "source_type": "journal_article",
            "is_verified": True,
            "text": text,
        }
        to_add.append(rec)

    # 6) Merge & save
    merged: Dict[str, Dict[str, Any]] = dict(existing)
    for r in to_add:
        merged[r["id"]] = r
    merged_list = list(merged.values())
    save_all(OUTFILE, merged_list)
    # Save state
    state_path = DATA_DIR / "state_journals.json"
    last_run_iso = datetime.now(timezone.utc).isoformat()
    all_pmids = [r["pmid"] for r in merged_list]
    save_state(state_path, last_run_iso, all_pmids)
    print(f"✅ Saved {len(to_add)} new | total={len(merged_list)} → {OUTFILE}")

if __name__ == "__main__":
    main()