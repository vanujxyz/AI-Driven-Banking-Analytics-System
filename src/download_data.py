"""Download the two Kaggle datasets via the REST API. Token is read from the KAGGLE_API_TOKEN env var
(never stored in the repo). Skips files already present."""
import os
import sys
import zipfile
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATASETS = {"churn": "radheshyamkollipara/bank-customer-churn", "paysim": "ealaxi/paysim1"}
EXPECTED = {"churn": "Customer-Churn-Records.csv", "paysim": "PS_20174392719_1491204439457_log.csv"}


def main():
    for name, ref in DATASETS.items():
        if (RAW / name / EXPECTED[name]).exists():
            print(f"{name}: present, skipping"); continue
        tok = os.environ.get("KAGGLE_API_TOKEN")
        if not tok:
            sys.exit("Set KAGGLE_API_TOKEN to download datasets")
        RAW.mkdir(parents=True, exist_ok=True)
        z = RAW / f"{name}.zip"
        with requests.get(f"https://www.kaggle.com/api/v1/datasets/download/{ref}",
                          headers={"Authorization": f"Bearer {tok}"}, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(z, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        zipfile.ZipFile(z).extractall(RAW / name)
        print(f"{name}: downloaded")


if __name__ == "__main__":
    main()
