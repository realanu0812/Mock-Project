# nutrition_insights/run_all.py
from __future__ import annotations
import os, sys, json, subprocess, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

# ---------------- paths ----------------
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parent                        # .../nutrition_insights
REPO_ROOT = PKG_ROOT.parent
DATA = PKG_ROOT / "data"

SCR_REDDIT  = PKG_ROOT / "scrappers" / "reddit_scrapper.py"
SCR_BLOGS   = PKG_ROOT / "scrappers" / "blogs_scrapper.py"
SCR_JRNLS   = PKG_ROOT / "scrappers" / "journals_scrapper.py"
MERGER      = PKG_ROOT / "merge_scrapper.py"
FILTER      = PKG_ROOT / "rag" / "filter_corpus.py"
INDEX       = PKG_ROOT / "rag" / "build_index.py"
QUERY       = PKG_ROOT / "rag" / "query_cli.py"

# Make package importable in child processes if needed
for p in (str(REPO_ROOT), str(PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Ensure __init__.py exists so imports like "nutrition_insights.*" work
for d in (PKG_ROOT, PKG_ROOT / "rag"):
    f = d / "__init__.py"
    if not f.exists():
        try:
            f.write_text("", encoding="utf-8")
        except Exception:
            pass

# ---------------- tiny utils ----------------
def read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def run(cmd, cwd: Path | None = None, env: dict | None = None, title: str | None = None):
    if title:
        print(title)
    subprocess.run(cmd, check=True, cwd=str(cwd or PKG_ROOT), env=env)

def load_env():
    """
    Load .env if present. Prefer python-dotenv; fall back to manual parse.
    """
    env_path = REPO_ROOT / ".env"
    if "python-dotenv" in {pkg.key for pkg in __import__("pkg_resources").working_set}:
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)  # silently no-op if missing
        except Exception:
            pass
    else:
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if not line or line.strip().startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            except Exception:
                pass

def freshness():
    """
    Read each scraper's state file and decide if we need to run.
    If no state found, we run.
    """
    state = {
        "reddit":   read_json(DATA / "state_reddit.json", {}),
        "blogs":    read_json(DATA / "state_blogs.json", {}),
        "journals": read_json(DATA / "state_journals.json", {}),
    }
    now = datetime.now(timezone.utc)

    def need(skey, max_age_hours):
        last = state[skey].get("last_run_iso")
        if not last:
            return True, None, None
        try:
            last_dt = datetime.fromisoformat(last)
        except Exception:
            return True, None, None
        age_h = (now - last_dt).total_seconds() / 3600.0
        return age_h > max_age_hours, last_dt, age_h

    # Default cadence: reddit 6h, blogs 24h, journals 24h
    need_reddit, last_reddit, age_reddit = need("reddit", 6)
    need_blogs,  last_blogs,  age_blogs  = need("blogs", 24)
    need_jrnls,  last_jrnls,  age_jrnls  = need("journals", 24)

    print("\n[Freshness check]")
    print(f"  reddit:   last={last_reddit}  age_h={None if age_reddit is None else round(age_reddit,2)}  need={need_reddit}")
    print(f"  blogs:    last={last_blogs}   age_h={None if age_blogs  is None else round(age_blogs,2)}   need={need_blogs}")
    print(f"  journals: last={last_jrnls} age_h={None if age_jrnls is None else round(age_jrnls,2)} need={need_jrnls}")

    return {
        "reddit": need_reddit,
        "blogs": need_blogs,
        "journals": need_jrnls,
        "last": {
            "reddit": last_reddit,
            "blogs": last_blogs,
            "journals": last_jrnls,
        }
    }

def ensure_reddit_env():
    req = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
    missing = [k for k in req if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    print(f"[ENV] Reddit creds loaded (CLIENT_ID={os.environ['REDDIT_CLIENT_ID'][:4]}..., USER_AGENT set)")

# ---------------- main pipeline ----------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python -m nutrition_insights.run_all \"<your protein question>\"")
        sys.exit(1)

    question = sys.argv[1]

    # Load .env and verify required bits
    load_env()
    ensure_reddit_env()

    # Optional: allow overriding the LLM model via env
    os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

    # Freshness & decide incremental boundaries
    fr = freshness()

    # --- Scrape: BLOGS ---
    if fr["blogs"]:
        # blogs supports incremental by internal state; no args required
        run([sys.executable, str(SCR_BLOGS)], cwd=PKG_ROOT,
            title=f"\n>>> {sys.executable} {SCR_BLOGS}  (cwd={PKG_ROOT})")

    # --- Scrape: REDDIT ---
    if fr["reddit"]:
        run([sys.executable, str(SCR_REDDIT)], cwd=PKG_ROOT,
            title=f"\n>>> {sys.executable} {SCR_REDDIT}  (cwd={PKG_ROOT})")

    # --- Scrape: JOURNALS ---
    if fr["journals"]:
        run([sys.executable, str(SCR_JRNLS)], cwd=PKG_ROOT,
            title=f"\n>>> {sys.executable} {SCR_JRNLS}  (cwd={PKG_ROOT})")

    # --- Merge to combined.json (merge only; scrapers already ran or were skipped) ---
    run([sys.executable, str(MERGER), "--merge-only"], cwd=PKG_ROOT,
        title=f"\n>>> {sys.executable} {MERGER} --merge-only  (cwd={PKG_ROOT})")

    # --- Filter corpus (defaults tuned for protein) ---
    run([
        sys.executable, str(FILTER),
        "--window-days", "180",
        "--protein-min-relevance", "0.30",
        "--near-dupe-thresh", "0.93"
    ], cwd=PKG_ROOT,
       title=f"\n>>> {sys.executable} {FILTER} --window-days 180 --protein-min-relevance 0.30 --near-dupe-thresh 0.93  (cwd={PKG_ROOT})")

    # --- Build / refresh FAISS ---
    run([
        sys.executable, str(INDEX),
        "--min-quality", "0.15"
    ], cwd=PKG_ROOT,
       title=f"\n>>> {sys.executable} {INDEX} --min-quality 0.15  (cwd={PKG_ROOT})")

    # --- Query (no flags needed beyond -q; the rest have strong defaults in query_cli.py) ---
    run([
        sys.executable, str(QUERY),
        "-q", question
    ], cwd=PKG_ROOT,
       title=f"\n>>> {sys.executable} {QUERY} -q {question}  (cwd={PKG_ROOT})")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step failed: {e.args}\nCommand: {' '.join(e.cmd)}\nReturn code: {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)