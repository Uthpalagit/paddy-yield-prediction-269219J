"""Small prediction API for the paddy yield model (optional MLOps / deployment part).
Run locally:  uvicorn api:app --port 8000
"""
import os
import sys
import json
import time
import pickle
import logging
from datetime import datetime, timezone
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from paddy_helpers import register_for_unpickling, build_features, explain_row  # noqa: E402

register_for_unpickling()
with open(os.path.join(BASE_DIR, "paddy_model.pkl"), "rb") as f:
    BUNDLE = pickle.load(f)

app = FastAPI(title="Sri Lanka paddy yield estimator", version="1.3")

# every prediction is written to stdout as one JSON line; in Azure Container Apps stdout goes to Log Analytics.
# Set PREDICTION_LOG to also append the lines to a file, which monitor.py can read.
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("paddy-api")
LOG_FILE = os.environ.get("PREDICTION_LOG")


class SeasonInput(BaseModel):
    district: str = Field(examples=["POLONNARUWA"])
    season: Literal["Maha", "Yala"]
    sown_ha: float = Field(gt=0)
    share_major: float = Field(ge=0, le=1)
    share_minor: float = Field(ge=0, le=1)
    share_rainfed: float = Field(ge=0, le=1)
    yield_same_season_lag1: float
    yield_same_season_roll3: float
    yield_prev_season: float
    national_prev_season: float
    rain_season: float = Field(ge=0)
    rain_wettest_month: float = Field(ge=0)
    rain_driest_month: float = Field(ge=0)
    rain_before: float = Field(ge=0)
    dry_spell_days: float = Field(ge=0, le=200)
    heavy_rain_days: float = Field(ge=0, le=200)
    nitrogen_import_ratio: Optional[float] = Field(default=None, gt=0, le=5)
    tmean: Optional[float] = None
    tmax: float
    tmin: Optional[float] = None
    rh: float = Field(ge=0, le=100)
    solar: Optional[float] = None
    wind: float = Field(ge=0)
    arima_forecast: Optional[float] = Field(default=None, description="Trend forecast; computed from the saved yield history if left out")
    sown_vs_prev3: Optional[float] = Field(default=None, gt=0, description="Sown extent vs the previous 3 same seasons; computed if left out")
    fertilizer_shortage: bool = Field(default=False, description="Policy input: true for a season under a fertilizer ban or shortage like 2021/22")


@app.get("/health")
def health():
    return {"status": "healthy", "model": BUNDLE["model_name"], "trained_on": BUNDLE["trained_on"]}


def harvest_class(estimate, recent_average):
    # the same rule as the classification task in the notebook (Section 16): more than 10% below or above the 3-season average
    if recent_average is None or pd.isna(recent_average) or recent_average <= 0:
        return None
    ratio = estimate / recent_average
    return "poor" if ratio < 0.9 else "good" if ratio > 1.1 else "normal"


def check(item):
    if item.district not in BUNDLE["districts"]:
        raise HTTPException(status_code=422, detail=f"Unknown district. Use one of: {BUNDLE['districts']}")
    if abs(item.share_major + item.share_minor + item.share_rainfed - 1) > 0.05:
        raise HTTPException(status_code=422, detail="Irrigation shares must add up to 1")


@app.post("/predict/batch")
def predict_batch(items: List[SeasonInput]):
    # many seasons in one call, used by the app's what-if curves and CSV upload
    if not 1 <= len(items) <= 500:
        raise HTTPException(status_code=422, detail="Send between 1 and 500 seasons")
    for item in items:
        check(item)
    rows = pd.concat([build_features(item.model_dump(), BUNDLE) for item in items], ignore_index=True)
    estimates = BUNDLE["model"].predict(rows)
    half = BUNDLE.get("range_80", BUNDLE["test_scores"]["Test MAE"])
    log.info(json.dumps({"event": "batch_prediction", "timestamp": datetime.now(timezone.utc).isoformat(), "model": BUNDLE["model_name"],
                         "rows": len(items)}))
    return [{"district": item.district, "season": item.season, "estimated_yield_kg_ha": round(float(e), 1),
             "likely_range_kg_ha": [round(float(e) - half, 1), round(float(e) + half, 1)],
             "harvest_class": harvest_class(float(e), item.yield_same_season_roll3)} for item, e in zip(items, estimates)]


@app.post("/predict")
def predict(item: SeasonInput):
    start = time.perf_counter()
    check(item)
    row = build_features(item.model_dump(), BUNDLE)
    estimate = float(BUNDLE["model"].predict(row)[0])
    mae = BUNDLE["test_scores"]["Test MAE"]
    half = BUNDLE.get("range_80", mae)  # 80% of the validation-year errors were within this
    interval = (estimate - half, estimate + half)
    contributions = explain_row(BUNDLE["model"], row).round(1)
    # the same season under the other fertilizer policy, so a client can show the scenario without a second call
    flipped = row.copy()
    flipped["fertilizer_shortage"] = int(not item.fertilizer_shortage)
    other_policy = float(BUNDLE["model"].predict(flipped)[0])
    derived = {c: (None if pd.isna(row[c].iloc[0]) else round(float(row[c].iloc[0]), 3)) for c in ("rain_anomaly_pct", "sown_vs_prev3", "arima_forecast")}
    record = {"event": "prediction", "timestamp": datetime.now(timezone.utc).isoformat(), "model": BUNDLE["model_name"],
              "trained_on": BUNDLE["trained_on"], "inputs": item.model_dump(), "estimate": round(estimate, 1),
              "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
    log.info(json.dumps(record))
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    return {"estimated_yield_kg_ha": round(estimate, 1),
            "likely_range_kg_ha": [round(interval[0], 1), round(interval[1], 1)],
            "harvest_class": harvest_class(estimate, item.yield_same_season_roll3),
            "fertilizer_shortage": item.fertilizer_shortage,
            "other_policy_estimate_kg_ha": round(other_policy, 1),
            "top_contributions_kg_ha": contributions.head(5).to_dict(),
            "contributions_kg_ha": contributions.to_dict(),
            "derived_features": derived}
