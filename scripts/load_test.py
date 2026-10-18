"""
Send realistic traffic to the API so the monitoring dashboard has data.

Patients are sampled from the cleaned dataset (missing values are simply omitted,
like a real caller would). A small share of requests is deliberately invalid.
With --drift-after, the population shifts to older, higher-risk patients so the
"mean predicted P(disease)" panel and the drift alert can be demonstrated.

Usage:
    python scripts/load_test.py --url http://$(minikube ip) --host heart.local --duration 300
    python scripts/load_test.py --url http://127.0.0.1:8000 --duration 60          # local uvicorn
    python scripts/load_test.py --url http://$(minikube ip) --host heart.local \
        --duration 480 --drift-after 240
"""
import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]
INTS = {"age", "sex", "cp", "fbs", "restecg", "exang", "slope"}


def load_patients(path: Path) -> list[dict]:
    patients = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            p = {}
            for k in FIELDS:
                v = row.get(k, "")
                if v not in ("", "nan", "NaN", "<NA>"):
                    p[k] = int(float(v)) if k in INTS else float(v)
            if {"age", "sex", "cp"} <= p.keys():
                patients.append(p)
    return patients


def drifted(p: dict) -> dict:
    """Older patients, more asymptomatic chest pain and exercise angina."""
    q = dict(p)
    q["age"] = min(int(q["age"]) + 12, 90)
    if random.random() < 0.6:
        q["cp"] = 4
    if random.random() < 0.5:
        q["exang"] = 1
    return q


def invalid() -> dict:
    return random.choice([
        {"age": 63, "sex": 1, "cp": 9},                 # bad category
        {"age": 4, "sex": 1, "cp": 2},                  # out of range
        {"age": 50, "sex": 1},                          # missing required field
        {"age": 50, "sex": 1, "cp": 2, "site": "x"},    # unknown field
    ])


def post(url: str, host: str | None, body: dict) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, None  # connection failure: counted as status 0, traffic continues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="base URL, e.g. http://192.168.64.2")
    ap.add_argument("--host", default=None, help="Host header for the Ingress, e.g. heart.local")
    ap.add_argument("--duration", type=int, default=300, help="seconds to run")
    ap.add_argument("--rps", type=float, default=4.0, help="requests per second")
    ap.add_argument("--invalid-rate", type=float, default=0.05)
    ap.add_argument("--drift-after", type=int, default=None, help="seconds before switching to drifted data")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "heart_disease_clean.csv")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    random.seed(a.seed)
    patients = load_patients(a.data)
    endpoint = a.url.rstrip("/") + "/predict"
    print(f"{len(patients)} patients loaded; sending {a.rps}/s to {endpoint} for {a.duration}s")

    codes, labels, probs = Counter(), Counter(), []
    start = time.time()
    n = 0
    while (elapsed := time.time() - start) < a.duration:
        drift = a.drift_after is not None and elapsed >= a.drift_after
        if random.random() < a.invalid_rate:
            body = invalid()
        else:
            body = random.choice(patients)
            body = drifted(body) if drift else body
        code, resp = post(endpoint, a.host, body)
        codes[code] += 1
        if resp:
            labels[resp["label"]] += 1
            probs.append(resp["probability_disease"])
        n += 1
        if n % 50 == 0:
            mean = sum(probs[-50:]) / max(len(probs[-50:]), 1)
            print(f"  t={elapsed:5.0f}s  sent={n:5d}  status={dict(codes)}  "
                  f"mean P(disease) last 50={mean:.2f}{'  [DRIFT]' if drift else ''}")
        time.sleep(max(0.0, (n / a.rps) - (time.time() - start)))

    print(f"\ndone: {n} requests  status={dict(codes)}  outcomes={dict(labels)}")


if __name__ == "__main__":
    main()
