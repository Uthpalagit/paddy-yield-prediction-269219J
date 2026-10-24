"""Tests for the prediction API, run by the CI workflow on every push (pytest tests)."""
import os
import sys

import pandas as pd
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from api import app, BUNDLE, SeasonInput  # noqa: E402

client = TestClient(app)


def held_out_season():
    # the held-out Polonnaruwa Maha 2025/26 season, with only the fields the API accepts
    row = pd.read_csv(os.path.join(BASE_DIR, "holdout_sample.csv")).iloc[0]
    return {k: (None if pd.isna(row[k]) else row[k].item() if hasattr(row[k], "item") else row[k])
            for k in SeasonInput.model_fields if k in row.index and k != "fertilizer_shortage"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["model"] == BUNDLE["model_name"]


def test_predict_held_out_season():
    r = client.post("/predict", json=held_out_season())
    assert r.status_code == 200
    body = r.json()
    assert 2000 < body["estimated_yield_kg_ha"] < 8000
    low, high = body["likely_range_kg_ha"]
    assert abs((high - low) - 2 * BUNDLE["range_80"]) < 1
    assert len(body["top_contributions_kg_ha"]) == 5


def test_shortage_lowers_the_estimate():
    normal = client.post("/predict", json=held_out_season()).json()["estimated_yield_kg_ha"]
    shortage = client.post("/predict", json={**held_out_season(), "fertilizer_shortage": True}).json()["estimated_yield_kg_ha"]
    assert shortage < normal


def test_unknown_district_rejected():
    assert client.post("/predict", json={**held_out_season(), "district": "LONDON"}).status_code == 422


def test_shares_must_add_up():
    assert client.post("/predict", json={**held_out_season(), "share_major": 0.9, "share_minor": 0.5}).status_code == 422


def test_negative_rainfall_rejected():
    assert client.post("/predict", json={**held_out_season(), "rain_season": -10}).status_code == 422


def test_harvest_class_returned():
    assert client.post("/predict", json=held_out_season()).json()["harvest_class"] in ("poor", "normal", "good")


def test_batch_matches_single():
    single = client.post("/predict", json=held_out_season()).json()["estimated_yield_kg_ha"]
    batch = client.post("/predict/batch", json=[held_out_season(), {**held_out_season(), "fertilizer_shortage": True}])
    assert batch.status_code == 200
    body = batch.json()
    assert len(body) == 2 and abs(body[0]["estimated_yield_kg_ha"] - single) < 0.1
    assert body[1]["estimated_yield_kg_ha"] < body[0]["estimated_yield_kg_ha"]


def test_batch_rejects_bad_rows():
    assert client.post("/predict/batch", json=[]).status_code == 422
    assert client.post("/predict/batch", json=[{**held_out_season(), "district": "LONDON"}]).status_code == 422
