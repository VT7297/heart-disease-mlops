"""
Command-line inference with the trained model.

Usage:
    python src/predict.py samples/patient_high_risk.json
    python src/predict.py --json '{"age": 63, "sex": 1, "cp": 4, "thalach": 108, "exang": 1}'
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import DEFAULT_MODEL_PATH, load_model  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", type=Path, help="JSON file: one patient object or a list")
    src.add_argument("--json", help="inline JSON patient object or list")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    a = p.parse_args()

    payload = json.loads(a.json) if a.json else json.loads(a.file.read_text())
    records = payload if isinstance(payload, list) else [payload]
    model = load_model(a.model)
    for rec, res in zip(records, model.predict(records), strict=True):
        print(json.dumps({"input": rec, **res, "model_version": model.version}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
