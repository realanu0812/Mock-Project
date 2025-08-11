# nutrition_insights/merge_scrapper.py
import json
import sys
import argparse
import subprocess
from pathlib import Path
import shutil

# ---- Paths ----
ROOT = Path(__file__).resolve().parent                  # .../nutrition_insights
PROJECT_ROOT = ROOT.parent                              # repo root
SCRAPERS_DIR = ROOT / "scrappers"
DATA_DIR = ROOT / "data"                                # canonical data dir
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "combined.json"

SCRAPER_FILES = [
    SCRAPERS_DIR / "blogs_scrapper.py",
    SCRAPERS_DIR / "reddit_scrapper.py",
    SCRAPERS_DIR / "journals_scrapper.py",
]

# places we’ll search for outputs if missing
CANDIDATE_DATA_DIRS = [
    DATA_DIR,                          # nutrition_insights/data
    PROJECT_ROOT / "data",             # repo_root/data (common if run from root)
]

DATA_FILENAMES = ["blogs.json", "reddit.json", "journals.json"]


def run_scraper(script_path: Path):
    """Run a scraper with cwd=ROOT so relative 'data/' writes land in nutrition_insights/data."""
    if not script_path.exists():
        print(f"[SKIP] Scraper not found: {script_path}")
        return
    print(f"\n=== Running {script_path.relative_to(ROOT)} (cwd={ROOT}) ===")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(ROOT))


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
        # copy into canonical location so everything is under nutrition_insights/data
        canonical.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(found_path, canonical)
        print(f"[SYNC] Copied {found_path} -> {canonical}")
        return canonical
    else:
        return canonical


def load_json(path: Path | None):
    if not path or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Run scrapers and merge outputs.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip running scrapers; just merge existing JSON files.")
    args = parser.parse_args()

    if not args.merge_only:
        for script in SCRAPER_FILES:
            run_scraper(script)

    # Ensure we can see what’s actually there
    print(f"\n[INFO] Looking for outputs in:")
    for d in CANDIDATE_DATA_DIRS:
        print(" -", d.resolve())
        if d.exists():
            for f in sorted(d.glob("*.json")):
                print("    •", f.name)

    # Find each data file, recover into canonical dir if needed
    sources = {}
    for fname in DATA_FILENAMES:
        p = find_or_recover(fname)
        if not p:
            print(f"[WARN] Missing data file after scrape: {fname}")
        sources[fname] = p

    # Load & merge
    combined = []
    for fname in DATA_FILENAMES:
        p = sources.get(fname)
        data = load_json(p)
        print(f"[LOAD] {fname}: {len(data)} (from {p if p else 'N/A'})")
        combined.extend(data)

    # Deduplicate by URL
    seen = set()
    deduped = []
    for item in combined:
        url = (item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(item)

    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Combined dataset saved to {OUTPUT_FILE.resolve()}")
    print(f"   Total records (deduped): {len(deduped)}")


if __name__ == "__main__":
    main()