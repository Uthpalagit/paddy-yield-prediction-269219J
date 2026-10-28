"""Helpers shared by the Streamlit app and the API: the blend class (needed to unpickle the model),
and the feature builder that turns user inputs into the row the model expects."""
import warnings

import numpy as np
import pandas as pd


class RidgeCatBoostBlend:
    # same class as in the notebook (Section 8.4); pickle looks it up by name when the model is loaded
    def __init__(self, weight_ridge=0.5, alpha=10.0, cat_kwargs=None):
        self.weight_ridge, self.alpha, self.cat_kwargs = weight_ridge, alpha, cat_kwargs or {}

    def predict(self, X):
        return self.weight_ridge * self.ridge_.predict(X) + (1 - self.weight_ridge) * self.cat_.predict(X)


def register_for_unpickling():
    # the notebook defined the class in __main__, so make it available there before pickle.load
    import __main__
    __main__.RidgeCatBoostBlend = RidgeCatBoostBlend


def arima_forecast(history, order=(1, 1, 1), min_history=8):
    from statsmodels.tsa.arima.model import ARIMA
    values = np.asarray([v for v in history if pd.notna(v)], dtype=float)
    if len(values) < min_history:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(ARIMA(values, order=order).fit().forecast(1)[0])
        except Exception:
            return np.nan


def build_features(inputs, bundle):
    district, season = inputs["district"], inputs["season"]
    key = f"{district}|{season}"
    row = dict(inputs)

    normal = bundle["rain_normals"].get((district, season), np.nan)
    row["rain_anomaly_pct"] = (row["rain_season"] / normal - 1) * 100 if normal and pd.notna(normal) else np.nan

    if pd.isna(row.get("sown_vs_prev3", np.nan)):
        recent_sown = bundle["sown_history"].get(key, [])
        row["sown_vs_prev3"] = row["sown_ha"] / np.mean(recent_sown) if recent_sown else np.nan

    if pd.isna(row.get("arima_forecast", np.nan)):
        row["arima_forecast"] = arima_forecast(bundle["yield_history"].get(key, []))

    row["fertilizer_shortage"] = int(bool(row.get("fertilizer_shortage", 0)))  # policy input: 1 under a ban or shortage like 2021/22

    frame = pd.DataFrame([row])
    for col in bundle["feature_columns"]:
        if col not in frame:
            frame[col] = np.nan
    return frame[bundle["feature_columns"]]


def explain_row(model, row):
    # exact contributions for one input row: tree SHAP for CatBoost, the blend or a tree pipeline; the terms of an EBM
    import shap
    from catboost import CatBoostRegressor
    if hasattr(model, "cat_"):
        model = model.cat_
    if isinstance(model, CatBoostRegressor):
        values = pd.Series(shap.TreeExplainer(model)(row).values[0], index=row.columns)
    elif hasattr(model, "eval_terms"):
        # Explainable Boosting Machine: additive, so its terms are the exact contributions (a pair is split between its two features)
        terms = model.eval_terms(row)[0]
        values = pd.Series(0.0, index=row.columns)
        for j, features in enumerate(model.term_features_):
            for f in features:
                values[model.feature_names_in_[f]] += terms[j] / len(features)
    else:
        prep, tree = model.named_steps["prep"], model.named_steps["model"]
        names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
        raw = pd.Series(shap.TreeExplainer(tree)(prep.transform(row)).values[0], index=names)
        values = pd.Series({col: raw[[n for n in names if n.startswith(col + "_")]].sum() if col in ("district", "season") else raw[col]
                            for col in row.columns})
    return values.reindex(values.abs().sort_values(ascending=False).index)



def psi(reference, current, bins=None):
    # Population Stability Index: how far the distribution of `current` has moved from `reference`
    # (cut points are quantiles of the reference; below 0.1 stable, 0.1-0.25 moderate shift, above 0.25 major shift).
    # One season has only about 25 districts, so the number of bins follows the sample size (about 5 values per bin, 4 to 10 bins)
    reference = pd.Series(reference, dtype=float).dropna()
    current = pd.Series(current, dtype=float).dropna()
    if bins is None:
        bins = int(np.clip(len(current) // 5, 4, 10))
    if len(reference) < bins or len(current) == 0:
        return np.nan
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share, cur_share = np.clip(ref_share, 1e-4, None), np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_report(reference, current, major=0.5, moderate=0.25):
    # PSI for every numeric column the two frames share, largest first. The usual cut-offs (0.25 and 0.1) assume large samples;
    # with one season's 25 districts, random noise alone reaches about 0.45 (notebook 13.1), so the defaults are doubled
    # a national feature has one value per season, the same for every district, so it has no distribution to compare and is skipped
    cols = [c for c in reference.columns if c in current.columns and pd.Series(current[c]).nunique() > 2]
    out = pd.DataFrame({"psi": [psi(reference[c], current[c]) for c in cols]}, index=cols)
    out["status"] = np.select([out.psi > major, out.psi > moderate], ["major shift", "moderate shift"], "stable")
    return out.sort_values("psi", ascending=False)


def out_of_range(reference, current, features, keys=("district", "season")):
    # share of the new rows whose value lies outside the range the same district and season had in the reference (training) data;
    # tree-based models can't follow values beyond what they have seen, so this is where estimates start to lag
    keys = list(keys)
    grouped = reference.groupby(keys)[features]
    low_by, high_by = grouped.min().add_suffix("__min"), grouped.max().add_suffix("__max")
    merged = current[keys + features].merge(low_by, left_on=keys, right_index=True, how="left").merge(high_by, left_on=keys, right_index=True, how="left")
    shares = {}
    for f in features:
        value, low, high = merged[f], merged[f + "__min"], merged[f + "__max"]
        known = value.notna() & low.notna()
        shares[f] = float(((value < low) | (value > high))[known].mean()) if known.any() else np.nan
    return pd.Series(shares).sort_values(ascending=False)
