"""
Generate the Grafana dashboard JSON (k8s/monitoring/dashboards/heart-api.json).
Kept as code so the dashboard is reviewable and reproducible.

Usage: python scripts/build_dashboard.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "k8s" / "monitoring" / "dashboards" / "heart-api.json"
DS = {"type": "prometheus", "uid": "prometheus"}
JOB = 'job="heart-api"'

_ids = iter(range(1, 100))


def target(expr, legend="", instant=False, fmt="time_series", ref="A"):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "instant": instant,
            "range": not instant, "format": fmt, "refId": ref}


def panel(kind, title, grid, targets, unit=None, description="", options=None, field_extra=None):
    defaults = {"unit": unit} if unit else {}
    defaults.update(field_extra or {})
    return {"id": next(_ids), "type": kind, "title": title, "description": description,
            "datasource": DS, "gridPos": dict(zip("xywh", grid, strict=True)),
            "targets": targets, "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": options or {}}


def row(title, y):
    return {"id": next(_ids), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


stat_opts = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
             "colorMode": "value", "graphMode": "area", "textMode": "auto"}

panels = [
    row("Service health", 0),
    panel("stat", "Request rate (req/s)", (0, 1, 5, 4),
          [target(f'sum(rate(http_requests_total{{{JOB}}}[1m]))')], "reqps",
          "All API requests per second across pods", stat_opts),
    panel("stat", "Error rate (5xx)", (5, 1, 5, 4),
          [target(f'100 * sum(rate(http_requests_total{{{JOB}, status=~"5.."}}[5m])) '
                  f'/ clamp_min(sum(rate(http_requests_total{{{JOB}}}[5m])), 1e-9)')], "percent",
          "Share of requests failing with a server error", stat_opts,
          {"thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None},
                                                         {"color": "red", "value": 5}]}}),
    panel("stat", "Rejected inputs (422)", (10, 1, 5, 4),
          [target(f'sum(increase(http_requests_total{{{JOB}, status="422"}}[$__range]))')], "short",
          "Requests rejected by input validation in the selected time range", stat_opts),
    panel("stat", "p95 /predict latency", (15, 1, 5, 4),
          [target(f'histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket'
                  f'{{{JOB}, path="/predict"}}[5m])))')], "s", "", stat_opts),
    panel("stat", "Pods up", (20, 1, 4, 4),
          [target(f'sum(up{{{JOB}}})')], "short", "Scrape targets currently healthy", stat_opts),

    row("Traffic & latency", 5),
    panel("timeseries", "Requests per second by path and status", (0, 6, 12, 8),
          [target(f'sum by (path, status) (rate(http_requests_total{{{JOB}}}[1m]))', "{{path}} {{status}}")],
          "reqps"),
    panel("timeseries", "/predict latency percentiles", (12, 6, 12, 8),
          [target(f'histogram_quantile({q}, sum by (le) (rate(http_request_duration_seconds_bucket'
                  f'{{{JOB}, path="/predict"}}[5m])))', f"p{int(q * 100)}", ref=r)
           for q, r in ((0.5, "A"), (0.95, "B"), (0.99, "C"))], "s"),
    panel("timeseries", "Requests per pod (load balancing)", (0, 14, 12, 7),
          [target(f'sum by (pod) (rate(http_requests_total{{{JOB}}}[1m]))', "{{pod}}")], "reqps"),
    panel("stat", "Model version serving", (12, 14, 12, 7),
          [target('max by (version, model) (model_info)', "{{model}} · v{{version}}", instant=True)],
          None, "From the model_info gauge exposed by each pod",
          {**stat_opts, "textMode": "name", "colorMode": "none", "graphMode": "none"}),

    row("Model behaviour", 21),
    panel("piechart", "Prediction outcomes (selected range)", (0, 22, 8, 9),
          [target(f'sum by (label) (increase(predictions_total{{{JOB}}}[$__range]))', "{{label}}",
                  instant=True)],
          "short", "", {"reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                        "pieType": "donut", "legend": {"displayMode": "table", "placement": "right",
                                                       "values": ["value", "percent"]}}),
    panel("timeseries", "Mean predicted P(disease) — drift signal", (8, 22, 8, 9),
          [target(f'sum(rate(prediction_probability_disease_sum{{{JOB}}}[5m])) '
                  f'/ clamp_min(sum(rate(prediction_probability_disease_count{{{JOB}}}[5m])), 1e-9)',
                  "mean P(disease)")],
          "percentunit", "A sustained shift suggests the incoming patient population has changed",
          None, {"min": 0, "max": 1}),
    panel("bargauge", "Predicted probability distribution", (16, 22, 8, 9),
          [target(f'sum by (le) (increase(prediction_probability_disease_bucket{{{JOB}}}[$__range]))',
                  "{{le}}", instant=True, fmt="heatmap")],
          "short", "Histogram of predicted P(disease), bucket upper bounds",
          {"orientation": "vertical", "displayMode": "gradient",
           "reduceOptions": {"calcs": ["lastNotNull"], "values": False}}),
    panel("timeseries", "Missing inputs imputed, by feature", (0, 31, 24, 7),
          [target(f'sum by (feature) (rate(prediction_missing_feature_total{{{JOB}}}[5m]))', "{{feature}}")],
          "reqps", "Data-quality signal: optional fields omitted by callers, filled by the model pipeline"),
]

dashboard = {
    "uid": "heart-api",
    "title": "Heart Disease API — Service & Model Monitoring",
    "tags": ["mlops", "heart-disease"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"},
    "editable": False,
    "panels": panels,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dashboard, indent=2))
print(f"wrote {OUT} ({len(panels)} panels)")
