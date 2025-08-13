# nutrition_insights/build_all.py
import argparse
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent          # .../nutrition_insights
PROJECT_ROOT = PKG.parent                      # .../Mock Project

def run(cmd, cwd=None):
    where = cwd or PROJECT_ROOT
    print(f"\n>>> {' '.join(cmd)}  (cwd={where})")
    subprocess.run(cmd, check=True, cwd=str(where))

def main():
    p = argparse.ArgumentParser(description="End-to-end: merge -> corpus -> index -> (optional) QA")
    p.add_argument("--scrape", action="store_true",
                   help="Run scrapers via merge_scrapper.py before building.")
    p.add_argument("--merge-only", action="store_true",
                   help="Only merge existing JSONs (skip running scrapers). Implies --scrape.")
    p.add_argument("--skip-qa", action="store_true",
                   help="Build everything but do not ask a question.")
    p.add_argument("--question", "-q", default="What are the top protein trends this week?",
                   help="Question to ask the QA CLI at the end.")
    args = p.parse_args()

    # 1) Scrape + merge (optional) — script lives inside the package folder
    if args.scrape or args.merge_only:
        merge_args = ["--merge-only"] if args.merge_only else []
        run([sys.executable, str(PKG / "merge_scrapper.py"), *merge_args], cwd=PROJECT_ROOT)

    # 2) Build corpus (module import needs PROJECT_ROOT as CWD)
    run([sys.executable, "-m", "nutrition_insights.rag.build_corpus"], cwd=PROJECT_ROOT)

    # 3) Build indexes
    run([sys.executable, "-m", "nutrition_insights.rag.build_index"], cwd=PROJECT_ROOT)

    # 4) Ask a question (optional)
    if not args.skip_qa:
        run([sys.executable, "-m", "nutrition_insights.rag.qa_cli", args.question], cwd=PROJECT_ROOT)
        print("\n✅ Done.")

if __name__ == "__main__":
    main()