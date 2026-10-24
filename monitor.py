"""Drift and performance check for the deployed paddy yield model, meant to run on a schedule (as in the MLOps workshop).

Two ways to feed it:
  python monitor.py --log predictions_log.jsonl [--actuals actuals.csv]
      inputs logged by api.py (PREDICTION_LOG), plus the official DCS yields once they are released
      (actuals.csv with columns district, season, year, yield_kg_ha)
  python monitor.py --season 2025.75
      replays one past season from Raw Data/cleaned_paddy_data.csv (2025.75 = Maha 2025/26, 2021.75 = Maha 2021/22).
      The deployed model was trained on every season up to Maha 2025/26, so a replayed season is one it has already seen:
      the replay shows the checks working, while notebook section 13.1 tests the rules on seasons the model had not seen.

It compares the new inputs with the training inputs (Population Stability Index per feature), and the estimates with the
official yields when they are available. The rules are the ones in notebook section 13.1. It writes monitoring_report.json,
and exits with code 1 when retraining is recommended, so that a scheduled workflow fails visibly and someone retrains.
"""
import os
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from paddy_helpers import register_for_unpickling, build_features, drift_report, out_of_range  # noqa: E402

SHORTAGE_SEASONS = (2021.75, 2022.25)  # Maha 2021/22 (import ban) and Yala 2022 (shortage)


def load_bundle():
    register_for_unpickling()
    with open(os.path.join(BASE_DIR, "paddy_model.pkl"), "rb") as f:
        return pickle.load(f)


def from_log(path, actuals_path, bundle):
    records = [json.loads(line) for line in open(path) if line.strip()]
    inputs = pd.DataFrame([r["inputs"] for r in records])
    rows = pd.concat([build_features(r, bundle) for r in inputs.to_dict("records")], ignore_index=True)
    frame = rows.assign(estimate=[r["estimate"] for r in records])
    if actuals_path:
        actuals = pd.read_csv(actuals_path)
        frame = frame.merge(actuals[["district", "season", "yield_kg_ha"]], on=["district", "season"], how="left")
    return frame, f"{len(records)} logged predictions"


def from_season(t, bundle):
    data = pd.read_csv(os.path.join(BASE_DIR, "Raw Data", "cleaned_paddy_data.csv"))
    data["t"] = data.year + np.where(data.season == "Maha", 0.75, 0.25)
    season = data[np.isclose(data.t, t)].copy()
    if season.empty:
        raise SystemExit(f"No season {t} in the cleaned dataset")
    season["fertilizer_shortage"] = int(t in SHORTAGE_SEASONS)  # the policy in force is known for past seasons
    for col in bundle["feature_columns"]:
        if col not in season:
            season[col] = np.nan
    season["estimate"] = bundle["model"].predict(season[bundle["feature_columns"]])
    label = f"{'Maha' if t % 1 == 0.75 else 'Yala'} {int(t)}{'/' + str(int(t) + 1)[2:] if t % 1 == 0.75 else ''}"
    return season, f"replay of {label}, {len(season)} districts (a season the deployed model was trained on)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", help="JSON-lines file written by api.py")
    parser.add_argument("--actuals", help="CSV with the official yields for the logged seasons")
    parser.add_argument("--season", type=float, help="replay one past season, e.g. 2021.75")
    parser.add_argument("--report", default="monitoring_report.json")
    args = parser.parse_args()

    bundle = load_bundle()
    if args.log:
        frame, source = from_log(args.log, args.actuals, bundle)
    elif args.season:
        frame, source = from_season(args.season, bundle)
    else:
        parser.error("give --log or --season")

    reference = bundle["reference_inputs"]
    seasons = frame["season"].dropna().unique()
    if len(seasons) == 1:  # compare Maha with Maha and Yala with Yala
        reference = reference[reference.season == seasons[0]]
    features = [c for c in reference.columns if c not in ("district", "season", "fertilizer_shortage", "national_prev_season")]
    history = [c for c in ["yield_same_season_lag1", "yield_same_season_roll3", "yield_prev_season", "arima_forecast"] if c in features]
    outside = out_of_range(reference, frame, features)
    drift = drift_report(reference[features], frame)

    report = {"source": source, "model": bundle["model_name"], "trained_on": bundle["trained_on"],
              "history_out_of_range_pct": round(float(outside[history].max() * 100), 1),
              "features_out_of_range_pct": (outside[outside > 0.2] * 100).round(1).to_dict(),
              "psi_largest": drift.psi.head(3).round(2).to_dict(),
              "note_on_psi": "one season has about 25 districts sharing the same weather, so PSI is noisy; see notebook 13.1"}
    extrapolating = report["history_out_of_range_pct"] > 50
    retrain = False
    if "yield_kg_ha" in frame and frame["yield_kg_ha"].notna().any():
        known = frame.dropna(subset=["yield_kg_ha"])
        mae = float(np.mean(np.abs(known.estimate - known.yield_kg_ha)))
        val = bundle["validation_mae"]
        alert_level = val["CV MAE"] + 2 * val["CV std"]
        retrain = mae > alert_level
        report.update({"labelled_rows": len(known), "mae": round(mae, 1), "alert_level": round(alert_level, 1), "error_alert": retrain})
    else:
        report.update({"labelled_rows": 0, "note": "no official yields yet: input drift only"})
    report["retrain_recommended"] = bool(retrain)
    report["action"] = ("retrain on all seasons including this one (rerun the notebook), then promote only if it beats the current model"
                        if retrain else ("most districts are beyond their training range: expect the estimates to lag, check them against field reports"
                                         if extrapolating else "no action"))

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(1 if retrain else 0)


if __name__ == "__main__":
    main()
