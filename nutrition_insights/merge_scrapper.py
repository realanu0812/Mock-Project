# nutrition_insights/merge_scrapper.py
import json
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
from datetime import datetime, timezone

# ---- Paths ----
ROOT = Path(__file__).resolve().parent                  # .../nutrition_insights
PROJECT_ROOT = ROOT.parent                              # repo root
SCRAPERS_DIR = ROOT / "scrappers"
DATA_DIR = ROOT / "data"                                # canonical data dir
DATA_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_FILE = DATA_DIR / "combined.json"
STATE_FILE = DATA_DIR / "merge_state.json"

SCRAPER_FILES = {
    "blogs": SCRAPERS_DIR / "blogs_scrapper.py",
    "reddit": SCRAPERS_DIR / "reddit_scrapper.py",
    "journals": SCRAPERS_DIR / "journals_scrapper.py",
}

# Where to look for data emitted by scrapers (will be synced into DATA_DIR)
CANDIDATE_DATA_DIRS = [
    DATA_DIR,                   # nutrition_insights/data (canonical)
    PROJECT_ROOT / "data",      # repo_root/data (fallback)
]

DATA_FILENAMES = {
    "blogs": "blogs.json",
    "reddit": "reddit.json",
    "journals": "journals.json",
}

# ----------------- utils -----------------
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(s: str | None):
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def load_json(path: Path | None):
    if not path or not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return []

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_state(state: dict) -> None:
    save_json(STATE_FILE, state)

def run_scraper(script_path: Path, since_iso: str | None):
    """Run a scraper with cwd=ROOT so relative 'data/' writes land in nutrition_insights/data.
       Pass --since if provided.
    """
    if not script_path.exists():
        print(f"[SKIP] Scraper not found: {script_path}")
        return 0

    cmd = [sys.executable, str(script_path)]
    if since_iso:
        cmd += ["--since", since_iso]

    print(f"\n=== Running {script_path.relative_to(ROOT)} (cwd={ROOT}) ===")
    if since_iso:
        print(f"[INCREMENTAL] since={since_iso}")

    # Load .env and pass env vars to subprocess
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT.parent / ".env")
    except Exception:
        pass
    import os
    env = os.environ.copy()
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    return 1

def find_or_recover(filename: str) -> Path | None:
    """Find file in known data dirs; if found outside canonical DATA_DIR, copy it in."""
    found_path = None
    for base in CANDIDATE_DATA_DIRS:
        p = base / filename
        if p.exists():
            found_path = p
            break

    if not found_path:
        return None

    canonical = DATA_DIR / filename
    if found_path.resolve() != canonical.resolve():
        canonical.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(found_path, canonical)
        print(f"[SYNC] Copied {found_path} -> {canonical}")
        return canonical
    else:
        return canonical

def dedupe_by_url(items: list) -> list:
    seen = set()
    out = []
    for it in items:
        url = (it.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(it)
    return out

# --------------- main orchestration ---------------
def main():
    ap = argparse.ArgumentParser(description="Run scrapers and merge outputs (append + incremental).")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip running scrapers; just merge existing JSON files.")
    ap.add_argument("--full", action="store_true",
                    help="Ignore saved state and do a full scrape (scraper defaults).")
    ap.add_argument("--since", type=str, default=None,
                    help="Force incremental cutoff (ISO-8601 UTC). Overrides saved state.")
    ap.add_argument("--replace", action="store_true",
                    help="Replace combined.json instead of append-merge.")
    args = ap.parse_args()

    # Determine since cutoff for this orchestration
    state = load_state()
    since_iso: str | None = None
    if args.full:
        print("[MODE] Full run requested (no --since passed to scrapers).")
    else:
        since_iso = args.since or state.get("last_run_iso")
        if since_iso:
            # normalize
            dt = parse_iso(since_iso)
            if dt:
                since_iso = dt.isoformat()
            print(f"[MODE] Incremental run since {since_iso}")
        else:
            print("[MODE] No previous run or --since provided; scrapers will use their own defaults.")

    # Run scrapers (unless merge-only)
    if not args.merge_only:
        ran = 0
        for name, script in SCRAPER_FILES.items():
            # Pass the *same* since to each scraper so they're aligned
            run_scraper(script, since_iso=None if args.full else since_iso)
            ran += 1
        if ran == 0:
            print("[WARN] No scrapers executed.")
    else:
        print("[MODE] --merge-only: skipping scraper execution.")

    # Inspect available outputs
    print(f"\n[INFO] Looking for outputs in:")
    for d in CANDIDATE_DATA_DIRS:
        print(" -", d.resolve())
        if d.exists():
            for f in sorted(d.glob("*.json")):
                print("    •", f.name)

    # Find & sync source files into canonical data dir
    sources: dict[str, Path | None] = {}
    for key, fname in DATA_FILENAMES.items():
        p = find_or_recover(fname)
        if not p:
            print(f"[WARN] Missing data file: {fname}")
        sources[key] = p

    # Load new source data
    new_batches = {}
    total_new_items = 0
    for key, p in sources.items():
        items = load_json(p)
        new_batches[key] = items
        print(f"[LOAD] {key}: {len(items)} (from {p if p else 'N/A'})")
        total_new_items += len(items)

    # Merge into combined.json (append or replace)
    if args.replace:
        merged = []
    else:
        merged = load_json(OUTPUT_FILE)  # start from previous combined

    # Track counts before adding
    before = len(merged)

    # Append all new data
    for key in ("blogs", "reddit", "journals"):
        merged.extend(new_batches.get(key, []))

    # Dedupe by URL
    deduped = dedupe_by_url(merged)

    # Save combined
    save_json(OUTPUT_FILE, deduped)

    # Build per-source counts in the final combined (rough, by URL set overlap)
    final_urls = { (it.get("url") or "").strip() for it in deduped if it.get("url") }
    src_counts = {}
    for key, items in new_batches.items():
        src_counts[key] = sum(1 for it in items if (it.get("url") or "").strip() in final_urls)

    added = len(deduped) - before

    print(f"\n✅ Combined dataset {'replaced' if args.replace else 'appended'} at {OUTPUT_FILE.resolve()}")
    print(f"   Added this run (after dedupe): {added}")
    print(f"   Total records (deduped): {len(deduped)}")
    print("\n--- Source counts (loaded this run) ---")
    for k, cnt in src_counts.items():
        print(f"{k}: {cnt}")
    print("--------------------------------------")

    # Update orchestration state
    new_state = {
        "last_run_iso": iso_now(),
        "used_since": since_iso if (since_iso and not args.full) else None,
        "mode": "full" if args.full else ("incremental" if since_iso else "default"),
        "files": {
            "output": str(OUTPUT_FILE),
            "blogs": str(sources["blogs"]) if sources["blogs"] else None,
            "reddit": str(sources["reddit"]) if sources["reddit"] else None,
            "journals": str(sources["journals"]) if sources["journals"] else None,
        },
        "counts": {
            "combined_total": len(deduped),
            "added_after_dedupe": added,
            "loaded_blogs": len(new_batches.get("blogs", [])),
            "loaded_reddit": len(new_batches.get("reddit", [])),
            "loaded_journals": len(new_batches.get("journals", [])),
        },
    }
    save_state(new_state)
    print(f"[STATE] Updated {STATE_FILE}")

if __name__ == "__main__":
    main()