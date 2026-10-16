"""
Download the UCI Heart Disease dataset (all four collection sites) and
save a combined raw CSV plus a manifest with checksums for reproducibility.

Usage:
    python data/download_data.py            # writes to data/raw/
    python data/download_data.py --out data/raw
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"

# The four "processed" files share the same 14-column layout.
SITES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, manifest = [], {"downloaded_at": datetime.now(timezone.utc).isoformat(), "files": {}}

    for site, filename in SITES.items():
        dest = out_dir / filename
        try:
            download(f"{BASE_URL}/{filename}", dest)
        except Exception as exc:  # noqa: BLE001 - surface any network failure clearly
            print(f"ERROR: could not download {filename}: {exc}", file=sys.stderr)
            return 1

        manifest["files"][filename] = {"site": site, "sha256": sha256_of(dest)}

        # '?' marks missing values in the source files
        df = pd.read_csv(dest, header=None, names=COLUMNS, na_values="?")
        df["site"] = site
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Original target is 0 (no disease) to 4 (severity); collapse to binary.
    combined["target"] = (combined["num"] > 0).astype(int)

    combined_path = out_dir / "heart_disease_combined_raw.csv"
    combined.to_csv(combined_path, index=False)
    manifest["combined"] = {
        "file": combined_path.name,
        "rows": len(combined),
        "sha256": sha256_of(combined_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved {len(combined)} rows to {combined_path}")
    print("\nRows per site:")
    print(combined["site"].value_counts().to_string())
    print("\nMissing values (%) by site:")
    print((combined.drop(columns=["site"]).isna().groupby(combined["site"]).mean() * 100)
          .round(1).loc[:, lambda d: d.sum() > 0].to_string())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "raw")
    sys.exit(main(parser.parse_args().out))
