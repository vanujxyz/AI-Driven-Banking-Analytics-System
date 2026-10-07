"""End-to-end pipeline: raw data -> blended data -> 5 modules -> ranked action list -> tests.
Usage:  python run_pipeline.py [--skip-tests]
Requires: conda env 'bankml' (see requirements.txt; JDK needed for H2O) and the raw datasets
(src/download_data.py fetches them if KAGGLE_API_TOKEN is set)."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["download_data", "data_blend", "churn", "fraud", "clv", "survival", "recommender", "integrate", "export_dashboard"]


def main():
    for s in STEPS:
        t0 = time.time()
        print(f"\n=== {s} ===", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "src" / f"{s}.py")], cwd=ROOT)
        if r.returncode:
            sys.exit(f"step '{s}' failed (exit {r.returncode})")
        print(f"--- {s} done in {time.time() - t0:.0f}s")
    if "--skip-tests" not in sys.argv:
        sys.exit(subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
