import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from paddy_helpers import RidgeCatBoostBlend, register_for_unpickling, build_features, explain_row  # noqa: E402

MODEL_PATH = os.path.join(BASE_DIR, "paddy_model.pkl")
EXAMPLE_PATH = os.path.join(BASE_DIR, "holdout_sample.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "model_results.csv")
MAP_PATH = os.path.join(BASE_DIR, "Raw Data", "district_map.geojson")  # official district boundaries (geoBoundaries), used only to draw the map
REPO_URL = "https://github.com/Uthpalagit/paddy-yield-prediction-269219J"
AZURE_APP_URL = "https://paddy-yield-app.grayflower-d27356cf.malaysiawest.azurecontainerapps.io"
AZURE_API_DOCS = "https://paddy-yield-api.grayflower-d27356cf.malaysiawest.azurecontainerapps.io/docs"
BACKUP_URL = "https://paddy-yield-prediction-269219j-2wfd9iajbb7ebf2jpuzzgb.streamlit.app"  # the same app on Streamlit Community Cloud
# the deployed prediction API; the app sends every estimate there and only uses its own copy of the model if the API is unreachable
API_URL = os.environ.get("PADDY_API_URL", "").rstrip("/")

GREEN, DARK, GOLD, RED, GREY = "#2D6A4F", "#1B4332", "#D89A3D", "#C1432C", "#6D7976"
FEATURE_NAMES = {
    "yield_same_season_roll3": "Average of last 3 seasons", "yield_same_season_lag1": "Same season last year",
    "arima_forecast": "Trend forecast (ARIMA)", "yield_prev_season": "Previous season", "national_prev_season": "National previous season",
    "share_rainfed": "Rainfed share", "share_major": "Major irrigation share", "share_minor": "Minor irrigation share",
    "nitrogen_import_ratio": "Nitrogen fertilizer imports", "district": "District", "season": "Season", "sown_ha": "Sown extent",
    "sown_vs_prev3": "Sown extent vs. recent seasons", "rain_season": "Season rainfall", "rain_anomaly_pct": "Rainfall vs. normal",
    "rain_wettest_month": "Wettest month", "rain_driest_month": "Driest month", "rain_before": "Rain before the season",
    "dry_spell_days": "Longest dry spell", "heavy_rain_days": "Heavy-rain days", "tmean": "Mean temperature", "tmax": "Maximum temperature",
    "tmin": "Minimum temperature", "rh": "Humidity", "solar": "Solar radiation", "wind": "Wind",
    "fertilizer_shortage": "Fertilizer shortage (policy)",
}

st.set_page_config(page_title="Paddy Yield Estimator | Sri Lanka", page_icon="🌾", layout="wide", initial_sidebar_state="auto")
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');
  .block-container {{padding: 1rem 2rem .5rem; max-width: 1560px;}}
  header[data-testid="stHeader"] {{background: transparent; height: 2.5rem;}}
  [data-testid="stSidebar"] {{background: #FFFFFF; border-right: 1px solid #DDE4DE;}}
  [data-testid="stSidebar"][aria-expanded="true"] {{min-width: 310px;}}
  [data-testid="stSidebarUserContent"] {{padding-top: 0;}}
  [data-testid="stSidebarHeader"] {{height: .75rem; min-height: .75rem; padding: 0;}}
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap: .4rem;}}
  .brand {{font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: {DARK};}}
  .brand small {{display: block; font-family: sans-serif; font-size: .78rem; font-weight: 400; color: {GREY};
                 line-height: 1.35; margin: .2rem 0 .75rem;}}
  .side-h {{font-size: .7rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: {GREEN};
            border-top: 1px solid #E4EAE5; padding-top: .6rem; margin: .35rem 0 .3rem;}}
  .topbar {{display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;
            background: linear-gradient(120deg, {DARK} 0%, {GREEN} 100%); color: white; border-radius: 14px;
            padding: .8rem 1.3rem; box-shadow: 0 6px 20px rgba(27,67,50,.15);}}
  .topbar .title {{font-family: 'Source Serif 4', Georgia, serif; font-size: 1.5rem; font-weight: 700; line-height: 1.2;}}
  .topbar .sub {{color: #CDEBD8; font-size: .85rem;}}
  .chips {{display: flex; flex-wrap: wrap; gap: .45rem;}}
  .chip {{background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); border-radius: 999px;
          padding: .25rem .75rem; font-size: .8rem; white-space: nowrap;}}
  .chip b {{color: #F2C879;}}
  .card {{background: white; border: 1px solid #DDE4DE; border-radius: 14px; padding: .8rem 1.1rem; min-height: 112px; margin-bottom: .75rem;
          box-shadow: 0 2px 8px rgba(27,67,50,.05);}}
  .card .label {{color: {GREY}; font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;}}
  .card .value {{color: {DARK}; font-family: 'Source Serif 4', Georgia, serif; font-size: 1.75rem; font-weight: 600; line-height: 1.3;}}
  .card .note {{color: {GREY}; font-size: .8rem;}}
  .card.main {{background: linear-gradient(160deg, #FFFFFF 0%, #E6F4EC 100%); border-color: #95D5B2;}}
  .card.main .value {{font-size: 2.2rem; color: {GREEN}; line-height: 1.15;}}
  .card.warn {{background: linear-gradient(160deg, #FFFFFF 0%, #FBEFE1 100%); border-color: #EBC48F;}}
  .card.actual {{background: linear-gradient(160deg, #FFFFFF 0%, #EEF1FA 100%); border-color: #B9C3E6;}}
  [data-testid="stTabs"] button p {{font-size: .95rem; font-weight: 600;}}
  .st-key-view {{margin-top: .6rem;}}
  .st-key-view [data-testid="stButtonGroup"] {{border-bottom: 1px solid #DDE4DE; width: 100%;}}
  .st-key-view button {{background: transparent !important; border: none !important; border-radius: 0 !important; box-shadow: none !important;
                        border-bottom: 3px solid transparent !important; padding: .35rem .9rem !important; margin-bottom: -1px;}}
  .st-key-view button p {{font-size: .95rem; font-weight: 600; color: {GREY};}}
  .st-key-view button[kind="segmented_controlActive"] {{border-bottom-color: {GREEN} !important;}}
  .st-key-view button[kind="segmented_controlActive"] p {{color: {GREEN};}}
  [data-testid="stCaptionContainer"] {{margin-top: -.4rem;}}
  [data-testid="stMain"] [data-testid="stCaptionContainer"] {{margin-top: .2rem;}}
  .outlook {{background: white; border: 1px solid #DDE4DE; border-radius: 12px; padding: .55rem 1rem; color: {DARK}; font-size: .92rem;}}
  .outlook .dot {{display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .55rem; vertical-align: -1px;}}
  /* scale with the screen: shrink the input panel on shorter screens, tighten the layout on narrow ones */
  @media (max-height: 880px) and (min-width: 768px) {{[data-testid="stSidebarContent"] {{zoom: .94;}}}}
  @media (max-height: 800px) and (min-width: 768px) {{[data-testid="stSidebarContent"] {{zoom: .86;}}}}
  @media (max-height: 680px) and (min-width: 768px) {{[data-testid="stSidebarContent"] {{zoom: .78;}}}}
  @media (max-width: 1100px) {{.block-container {{padding: 1rem 1rem .5rem;}} .card .value {{font-size: 1.45rem;}}
                               .card.main .value {{font-size: 1.8rem;}} .topbar .title {{font-size: 1.25rem;}}}}
  @media (max-width: 1100px) {{[data-testid="stMain"] [data-testid="stHorizontalBlock"] {{flex-wrap: wrap;}}
                               [data-testid="stMain"] [data-testid="stColumn"] {{min-width: 100%;}}
                               [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4))
                                 > [data-testid="stColumn"] {{min-width: calc(50% - .6rem);}}}}
  @media (max-width: 640px) {{.block-container {{padding: .75rem 1rem .5rem;}} .card {{min-height: 0;}}
                              .chip {{white-space: normal;}}}}
  @media (min-width: 1800px) {{html {{font-size: 17px;}} [data-testid="stSidebar"][aria-expanded="true"] {{min-width: 360px; max-width: 360px;}}}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    register_for_unpickling()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_map():
    if not os.path.exists(MAP_PATH):
        return None
    with open(MAP_PATH) as f:
        return json.load(f)


def card(label, value, note="", kind=""):
    st.markdown(f'<div class="card {kind}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)


@st.cache_data(ttl=600, show_spinner=False)
def estimate_with_api(payload_json):
    r = requests.post(f"{API_URL}/predict", data=payload_json, headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def batch_with_api(payloads_json):
    r = requests.post(f"{API_URL}/predict/batch", data=payloads_json, headers={"Content-Type": "application/json"}, timeout=120)
    if r.status_code == 422:
        raise ValueError(r.json().get("detail"))
    r.raise_for_status()
    return r.json()


def to_payload(d):
    out = {k: (None if isinstance(v, float) and np.isnan(v) else v.item() if hasattr(v, "item") else v) for k, v in d.items()}
    out["fertilizer_shortage"] = bool(out.get("fertilizer_shortage") or False)
    return out


def harvest_class(estimate, recent_average):
    # the rule of the classification task in the notebook, more than 10% below or above the 3-season average
    if recent_average is None or pd.isna(recent_average) or recent_average <= 0:
        return None
    ratio = estimate / recent_average
    return "poor" if ratio < 0.9 else "good" if ratio > 1.1 else "normal"


def estimate_many(rows):
    # many seasons at once, through the API's batch endpoint, or with the local model if the API cannot be reached
    payloads = [to_payload(r) for r in rows]
    if API_URL:
        try:
            return [r["estimated_yield_kg_ha"] for r in batch_with_api(json.dumps(payloads, default=float))]
        except ValueError:
            raise
        except Exception:
            pass
    feats = pd.concat([build_features({**r, "fertilizer_shortage": int(r["fertilizer_shortage"])}, bundle) for r in payloads],
                      ignore_index=True)
    return [float(v) for v in bundle["model"].predict(feats)]


def typical_inputs(district, season):
    # the same pre-filled values the input panel shows for a district, used for the example batch file
    k = f"{district}|{season}"
    hist, sown_hist = bundle["yield_history"].get(k, []), bundle["sown_history"].get(k, [])
    if not hist or not sown_hist:
        return None
    m, w = bundle["irrigation_mix"].get(district, {}), bundle["weather_medians"].get((district, season), {})
    get = lambda name, default: float(w[name]) if pd.notna(w.get(name, np.nan)) else float(default)
    major, minor = float(np.nan_to_num(m.get("share_major", 0.3))), float(np.nan_to_num(m.get("share_minor", 0.3)))
    if major + minor > 1:  # a few districts record slightly more than their sown extent under irrigation
        major, minor = major / (major + minor), minor / (major + minor)
    roll3 = float(np.mean(hist[-3:]))
    return dict(district=district, season=season, sown_ha=float(sown_hist[-1]), share_major=major, share_minor=minor,
                share_rainfed=max(0.0, 1 - major - minor), yield_same_season_lag1=float(hist[-1]), yield_same_season_roll3=roll3,
                yield_prev_season=roll3, national_prev_season=float(bundle["latest_national"]["national_yield"]),
                rain_season=get("rain_season", 800), rain_wettest_month=get("rain_wettest_month", 300),
                rain_driest_month=get("rain_driest_month", 40), rain_before=get("rain_before", 300),
                dry_spell_days=get("dry_spell_days", 15), heavy_rain_days=get("heavy_rain_days", 2), tmean=get("tmean", 27.0),
                tmax=get("tmax", 31.0), tmin=get("tmin", 23.0), rh=get("rh", 80.0), solar=get("solar", 18.0), wind=get("wind", 2.5),
                fertilizer_shortage=False)


@st.cache_data(ttl=60, show_spinner=False)
def api_is_up():
    try:
        return requests.get(f"{API_URL}/health", timeout=5).ok
    except Exception:
        return False


def side_heading(text):
    st.markdown(f'<div class="side-h">{text}</div>', unsafe_allow_html=True)


def style(fig, height, title=None):
    # one look for every chart: transparent background, light grid, the same fonts as the page
    fig.update_layout(height=height, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=DARK, size=12), margin=dict(l=10, r=30, t=40 if title else 10, b=10),
                      title=dict(text=title, x=0.01, font=dict(family="Georgia, serif", size=15, color=DARK)) if title else None,
                      hoverlabel=dict(bgcolor="white", font_color=DARK))
    fig.update_xaxes(showgrid=False, linecolor="#C9D3CC")
    fig.update_yaxes(gridcolor="#E4EAE5", zeroline=False)
    return fig


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("paddy_model.pkl not found. Run the notebook first.")
    st.stop()

mae = bundle["test_scores"]["Test MAE"]
uses = set(bundle["feature_columns"])  # only ask for inputs the saved model actually uses
example = pd.read_csv(EXAMPLE_PATH).iloc[0].to_dict() if os.path.exists(EXAMPLE_PATH) else None

# ---------- inputs: a fixed panel on the left ----------
with st.sidebar:
    st.markdown('<div class="brand">🌾 Season inputs<small>Typical values for the district are pre-filled.</small></div>',
                unsafe_allow_html=True)
    if example is not None:
        b1, b2 = st.columns([1.7, 1])
        if b1.button("Load test season", width="stretch", type="primary",
                     help=f"{example['district'].title()}, {example['season']} {int(example['year'])}/{int(example['year']) + 1}: "
                          "a season the model never saw in training"):
            st.session_state["example"] = example
        if b2.button("Reset", width="stretch"):
            st.session_state.pop("example", None)
    ex = st.session_state.get("example")

    side_heading("Season")
    districts = bundle["districts"]
    district = st.selectbox("District", districts, index=districts.index(ex["district"]) if ex else districts.index("POLONNARUWA"),
                            format_func=lambda d: d.title())
    season = st.segmented_control("Season", ["Maha", "Yala"], default=ex["season"] if ex else "Maha", width="stretch",
                                  help="Maha: about October-March (north-east monsoon). Yala: about April-September.") or "Maha"
    key = f"{district}|{season}"
    use_ex = bool(ex) and district == ex["district"] and season == ex["season"]
    history = bundle["yield_history"].get(key, [])
    mix = bundle["irrigation_mix"].get(district, {})
    wx = bundle["weather_medians"].get((district, season), {})

    def val(name, default):
        if use_ex and pd.notna(ex.get(name)):
            return float(ex[name])
        return float(wx[name]) if pd.notna(wx.get(name, np.nan)) else float(default)

    side_heading("Planting and policy")
    recent_sown = bundle["sown_history"].get(key, [])
    sown = st.number_input("Sown extent (ha)", min_value=0.0, step=500.0, format="%.0f",
                           value=float(ex["sown_ha"]) if use_ex else float(recent_sown[-1] if recent_sown else 10000))
    c1, c2 = st.columns(2)
    share_major = c1.number_input("Major irrigation share", 0.0, 1.0, float(ex["share_major"]) if use_ex and pd.notna(ex["share_major"])
                                  else float(np.nan_to_num(mix.get("share_major", 0.3))), 0.05, format="%.2f")
    share_minor = c2.number_input("Minor irrigation share", 0.0, 1.0, float(ex["share_minor"]) if use_ex and pd.notna(ex["share_minor"])
                                  else float(np.nan_to_num(mix.get("share_minor", 0.3))), 0.05, format="%.2f")
    if share_major + share_minor > 1:  # the three shares must add up to 1, so scale the two irrigation shares down
        st.caption(f"The two shares add up to {share_major + share_minor:.2f}, so they are scaled to add up to 1.")
        share_major, share_minor = share_major / (share_major + share_minor), share_minor / (share_major + share_minor)
    share_rainfed = max(0.0, 1.0 - share_major - share_minor)
    st.caption(f"Rainfed share (the remainder) **{share_rainfed:.2f}**")
    n_ratio = np.nan
    if "nitrogen_import_ratio" in uses:
        latest_n = bundle.get("latest_nitrogen_ratio", {"ratio": 1.0})
        n_default = ex.get("nitrogen_import_ratio") if use_ex else np.nan
        n_ratio = st.slider("Nitrogen fertilizer imports vs. the previous 3 years", 0.2, 2.0,
                            float(n_default) if pd.notna(n_default) else float(round(latest_n["ratio"], 2)), 0.01,
                            help="1.0 means a normal year. In 2021, the year of the fertilizer ban, it was about 0.44.")
    shortage = False
    if "fertilizer_shortage" in uses:
        shortage = st.toggle("Fertilizer ban / shortage", value=False,
                             help="Government policy input, as in 2021/22. The model learned its effect from the two shortage seasons, "
                                  "Maha 2021/22 (import ban) and Yala 2022 (shortage in the economic crisis).")

    side_heading("Weather")
    c1, c2 = st.columns(2)
    rain_season = c1.number_input("Season rain (mm)", min_value=0.0, step=25.0, format="%.0f", value=val("rain_season", 800))
    rain_before = c2.number_input("Rain before (mm)", min_value=0.0, step=25.0, format="%.0f", value=val("rain_before", 300),
                                  help="Maha: August-September. Yala: the previous October-January, which fills the tanks.")
    dry_spell = c1.number_input("Dry spell (days)", min_value=0.0, step=1.0, format="%.0f", value=val("dry_spell_days", 15),
                                help="Longest run of days without rain in the season")
    heavy_days = c2.number_input("Heavy-rain days", min_value=0.0, step=1.0, format="%.0f", value=val("heavy_rain_days", 2),
                                 help="Days with 50 mm of rain or more")
    with st.expander("More weather inputs"):
        c1, c2 = st.columns(2)
        rain_wettest = c1.number_input("Wettest month (mm)", min_value=0.0, format="%.0f", value=val("rain_wettest_month", 300))
        rain_driest = c2.number_input("Driest month (mm)", min_value=0.0, format="%.0f", value=val("rain_driest_month", 40))
        tmax = c1.number_input("Max temp (°C)", format="%.1f", value=val("tmax", 31.0))
        tmean = c2.number_input("Mean temp (°C)", format="%.1f", value=val("tmean", 27.0)) if "tmean" in uses else np.nan
        tmin = c1.number_input("Min temp (°C)", format="%.1f", value=val("tmin", 23.0)) if "tmin" in uses else np.nan
        rh = c2.number_input("Humidity (%)", min_value=0.0, max_value=100.0, format="%.0f", value=val("rh", 80.0))
        solar = c1.number_input("Solar (MJ/m²/day)", min_value=0.0, format="%.1f", value=val("solar", 18.0))
        wind = c2.number_input("Wind at 2 m (m/s)", min_value=0.0, format="%.1f", value=val("wind", 2.5))
    with st.expander("Recent yields (DCS records)"):
        if use_ex:
            lag1, roll3, prev = ex["yield_same_season_lag1"], ex["yield_same_season_roll3"], ex["yield_prev_season"]
        else:
            lag1 = history[-1] if history else np.nan
            roll3 = float(np.mean(history[-3:])) if history else np.nan
            prev = np.nan
        lag1 = st.number_input("Same season last year (kg/ha)", value=float(lag1), format="%.0f")
        roll3 = st.number_input(f"Average of the last 3 {season} seasons", value=float(roll3), format="%.0f")
        prev = st.number_input("The season just before (kg/ha)", value=float(prev) if pd.notna(prev) else float(roll3), format="%.0f")
        national_prev = st.number_input("National yield, previous season", format="%.0f",
                                        value=float(ex["national_prev_season"]) if use_ex else float(bundle["latest_national"]["national_yield"]))

# ---------- header ----------
st.markdown(f"""
<div class="topbar">
  <div><div class="title">Sri Lanka Paddy Yield Estimator</div>
  <div class="sub">District yield for a Maha or Yala season as soon as it ends, months before the official survey</div></div>
  <div class="chips">
    <div class="chip">Model <b>{bundle['model_name']}</b></div>
    <div class="chip">Test error <b>{mae:,.0f} kg/ha</b></div>
    <div class="chip"><b>25</b> districts + <b>2</b> irrigation areas</div>
    <div class="chip">{"Estimates from the <b>live prediction API</b>" if API_URL and api_is_up() else "Estimates from the <b>local model</b>"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

VIEWS = ["Estimate", "What-if", "District map", "Model comparison", "Batch upload", "About"]
view = st.segmented_control("Page", VIEWS, default="Estimate", key="view", label_visibility="collapsed") or "Estimate"

inputs = dict(district=district, season=season, sown_ha=sown,
              # for the held-out example, use its stored trend features (the saved history already contains that season)
              arima_forecast=ex["arima_forecast"] if use_ex else np.nan, sown_vs_prev3=ex["sown_vs_prev3"] if use_ex else np.nan,
              share_major=share_major, share_minor=share_minor, share_rainfed=share_rainfed, nitrogen_import_ratio=n_ratio,
              yield_same_season_lag1=lag1, yield_same_season_roll3=roll3, yield_prev_season=prev, national_prev_season=national_prev,
              rain_season=rain_season, rain_wettest_month=rain_wettest, rain_driest_month=rain_driest, rain_before=rain_before,
              dry_spell_days=dry_spell, heavy_rain_days=heavy_days, tmean=tmean, tmax=tmax, tmin=tmin, rh=rh, solar=solar, wind=wind,
              fertilizer_shortage=int(shortage))
result, source = None, "the model loaded in this app (the API was not reachable)"
if API_URL:
    payload = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in inputs.items()}
    payload["fertilizer_shortage"] = bool(shortage)
    try:
        result, source = estimate_with_api(json.dumps(payload, default=float)), "the live prediction API"
    except Exception:
        result = None
if result:
    estimate = result["estimated_yield_kg_ha"]
    low, high = result["likely_range_kg_ha"]
    other = result["other_policy_estimate_kg_ha"]
    contrib = pd.Series(result["contributions_kg_ha"], dtype=float)
    rain_anomaly = result["derived_features"]["rain_anomaly_pct"]
    trend = result["derived_features"]["arima_forecast"]
else:
    row = build_features(inputs, bundle)
    estimate = float(bundle["model"].predict(row)[0])
    half = bundle.get("range_80", mae)  # 80% of the model's errors on the validation years were within this
    low, high = estimate - half, estimate + half
    flipped = row.copy()
    flipped["fertilizer_shortage"] = int(not shortage)
    other = float(bundle["model"].predict(flipped)[0])
    contrib = explain_row(bundle["model"], row)
    rain_anomaly = row["rain_anomaly_pct"].iloc[0]
    trend = row["arima_forecast"].iloc[0]
contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)

if view == "Estimate":

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        card("Estimated yield", f"{estimate:,.0f}", f"kg/ha · production ≈ {estimate * sown / 1e3:,.0f} t", kind="main")
    with k2:
        card("Likely range", f"{low:,.0f} – {high:,.0f}", "80% of past errors fell inside this range")
    with k3:
        if use_ex:
            err = estimate - ex["yield_kg_ha"]
            card("Actual (DCS)", f"{ex['yield_kg_ha']:,.0f}", f"estimate off by {err:+,.0f} kg/ha ({abs(err) / ex['yield_kg_ha'] * 100:.0f}%)",
                 kind="actual")
        else:
            card("vs. last 3 seasons", f"{estimate - roll3:+,.0f}", f"kg/ha against the {season} average of {roll3:,.0f}")
    with k4:
        if "fertilizer_shortage" in uses:  # the same season under the other policy, as a scenario
            card("With normal fertilizer" if shortage else "If fertilizer is short", f"{other:,.0f}",
                 f"{other - estimate:+,.0f} kg/ha · {(other - estimate) * sown / 1e3:+,.0f} t for this district", kind="warn")
        else:
            card("Rainfall vs. normal", f"{rain_anomaly:+.0f}%" if rain_anomaly is not None else "n/a", "against the 1981-2010 average")

    left, right = st.columns([1.1, 1])
    series = pd.DataFrame(bundle.get("yield_series", {}).get(key, []), columns=["year", "yield"])
    est_year = int(ex["year"]) if use_ex else (int(series.year.max()) + 1 if len(series) else 2026)
    hist = series[series.year < est_year]
    with left, st.container(border=True):
        fig = go.Figure()
        if len(hist):
            fig.add_trace(go.Scatter(x=hist.year, y=hist["yield"], mode="lines+markers", name="official yield",
                                     line=dict(color=GREEN, width=2.2), marker=dict(size=4),
                                     hovertemplate="%{x}: %{y:,.0f} kg/ha<extra></extra>"))
        fig.add_trace(go.Scatter(x=[est_year, est_year], y=[low, high], mode="lines", name="likely range",
                                 line=dict(color=GOLD, width=10), opacity=.35, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[est_year], y=[estimate], mode="markers", name="estimate",
                                 marker=dict(color=GOLD, size=15, symbol="star", line=dict(color=DARK, width=1)),
                                 hovertemplate="estimate: %{y:,.0f} kg/ha<extra></extra>"))
        if use_ex:
            fig.add_trace(go.Scatter(x=[est_year], y=[ex["yield_kg_ha"]], mode="markers", name="actual",
                                     marker=dict(color="#4C5FA8", size=11, symbol="diamond"),
                                     hovertemplate="actual: %{y:,.0f} kg/ha<extra></extra>"))
        style(fig, 360, f"{district.title()} {season}: history and estimate")
        fig.update_layout(yaxis_title="kg per hectare", legend=dict(orientation="h", y=-0.08, x=0, xanchor="left", yanchor="top",
                                                                    font=dict(size=11)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right, st.container(border=True):
        contrib = contrib.head(8)[::-1]
        fig = go.Figure(go.Bar(x=contrib.values, y=[FEATURE_NAMES.get(c, c) for c in contrib.index], orientation="h",
                               marker_color=[GREEN if v > 0 else RED for v in contrib.values],
                               text=[f"{v:+,.0f}" for v in contrib.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:+,.0f} kg/ha<extra></extra>"))
        style(fig, 360, "What drove this estimate (kg/ha)")
        fig.update_layout(margin=dict(r=50))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    outlook = harvest_class(estimate, roll3)
    o1, o2 = st.columns([3, 1], vertical_alignment="center")
    with o1:
        colour = {"poor": RED, "normal": GOLD, "good": GREEN}.get(outlook, GREY)
        text = {"poor": "a poor harvest, more than 10% below", "normal": "a normal harvest, within 10% of",
                "good": "a good harvest, more than 10% above"}.get(outlook, "no outlook (no recent average)")
        st.markdown(f'<div class="outlook"><span class="dot" style="background:{colour}"></span>Harvest outlook '
                    f'<b>{(outlook or "n/a").title()}</b>, the estimate points to {text} the recent {season} average</div>',
                    unsafe_allow_html=True)
    with o2:
        record = {**to_payload(inputs), "estimated_yield_kg_ha": round(estimate, 1), "likely_low": round(low, 1), "likely_high": round(high, 1),
                  "harvest_outlook": outlook, "estimate_source": source}
        st.download_button("Download this estimate", pd.DataFrame([record]).to_csv(index=False).encode(),
                           file_name=f"estimate_{district.lower().replace(' ', '_')}_{season.lower()}.csv", mime="text/csv",
                           width="stretch", icon=":material/download:")
    st.caption(f"Estimate from {source}. Recent yields and irrigation usually matter most. The model learns from past seasons, so a shock "
               "it has never seen can push the estimate off, and a large gap from early field reports is a warning sign.")

base_inputs = {**inputs, "arima_forecast": trend if trend is not None else np.nan}

if view == "What-if":
    options = {
        "Season rainfall (mm)": ("rain_season", np.linspace(0.4, 1.6, 13) * max(rain_season, 100)),
        "Rain before the season (mm)": ("rain_before", np.linspace(0, 2, 13) * max(rain_before, 50)),
        "Longest dry spell (days)": ("dry_spell_days", np.linspace(0, 60, 13)),
        "Heavy-rain days": ("heavy_rain_days", np.linspace(0, 15, 16)),
        "Major irrigation share": ("share_major", np.linspace(0, 1 - share_minor, 13)),
        "Sown extent (ha)": ("sown_ha", np.linspace(0.5, 1.5, 11) * max(sown, 100)),
        "Maximum temperature (°C)": ("tmax", np.linspace(tmax - 2, tmax + 2, 9)),
    }
    w1, w2 = st.columns([1, 2.6])
    with w1:
        choice = st.selectbox("Change one input", list(options))
        st.caption("Every other input stays as set in the panel on the left. Both lines come from the same model, "
                   "one with normal fertilizer supply and one with a shortage like 2021/22.")
    field, grid = options[choice]
    rows = []
    for policy in (False, True):
        for v in grid:
            r = {**base_inputs, field: float(v), "fertilizer_shortage": policy}
            if field == "share_major":
                r["share_rainfed"] = max(0.0, 1 - float(v) - share_minor)
            rows.append(r)
    try:
        values = estimate_many(rows)
    except ValueError as e:
        values = None
        w2.error(f"The API rejected these inputs. {e}")
    if values:
        n = len(grid)
        curve = pd.DataFrame({"x": list(grid) * 2, "estimate": values,
                              "policy": ["Normal fertilizer"] * n + ["Fertilizer shortage"] * n})
        current = base_inputs[field]
        with w2, st.container(border=True):
            fig = go.Figure()
            for name, colour in [("Normal fertilizer", GREEN), ("Fertilizer shortage", GOLD)]:
                d = curve[curve.policy == name]
                fig.add_trace(go.Scatter(x=d.x, y=d.estimate, mode="lines+markers", name=name, line=dict(color=colour, width=2.5),
                                         marker=dict(size=6), hovertemplate="%{x:,.1f} gives %{y:,.0f} kg/ha<extra></extra>"))
            fig.add_vline(x=current, line=dict(color=GREY, dash="dot"), annotation_text="current value", annotation_position="top")
            style(fig, 400, f"Yield as {choice.split(' (')[0].lower()} changes")
            fig.update_layout(xaxis_title=choice, yaxis_title="kg per hectare",
                              legend=dict(orientation="h", y=-0.2, x=0, xanchor="left", yanchor="top"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        normal = curve[curve.policy == "Normal fertilizer"].estimate
        with w1:
            fmt = (lambda v: f"{v:.2f}") if field == "share_major" else (lambda v: f"{v:,.1f}" if field == "tmax" else f"{v:,.0f}")
            card("Lowest across the range", f"{normal.min():,.0f}", f"kg/ha at {fmt(grid[int(normal.values.argmin())])}")
            card("Highest across the range", f"{normal.max():,.0f}", f"kg/ha at {fmt(grid[int(normal.values.argmax())])}", kind="main")
        st.caption("The model was trained on the conditions seen between 1981 and 2026. Values far outside that range are extrapolation "
                   "and the curve can flatten there, because a tree-based model cannot predict beyond what it has seen.")

if view == "District map":
    geo = load_map()
    series_all = bundle.get("yield_series", {})
    m1, m2, m3 = st.columns([1, 1, 1.4])
    map_season = m1.segmented_control("Season", ["Maha", "Yala"], default="Maha", key="map_season", width="stretch") or "Maha"
    measure = m2.segmented_control("Show", ["Yield", "Change"], default="Yield", key="map_measure", width="stretch",
                                   help="Change is the percentage change from the same season of the year before") or "Yield"
    years = sorted({int(y) for k, v in series_all.items() if k.endswith(map_season) for y, _ in v})
    map_year = m3.select_slider("Year", options=years, value=max(years)) if years else None
    lookup = {k.split("|")[0]: dict((int(y), yl) for y, yl in v) for k, v in series_all.items() if k.endswith(map_season)}
    map_df = pd.DataFrame([{"district": d, "yield": v.get(map_year), "before": v.get(map_year - 1) if map_year else None}
                           for d, v in lookup.items()]).dropna(subset=["yield"])
    if map_df.empty:
        st.warning("No data for this selection.")
    else:
        map_df["change"] = (map_df["yield"] / map_df["before"] - 1) * 100
        col = "yield" if measure == "Yield" else "change"
        shown = map_df.dropna(subset=[col])
        label = f"{map_season} {map_year}{'/' + str(map_year + 1)[2:] if map_season == 'Maha' else ''}"
        g1, g2 = st.columns([1.15, 1])
        with g1, st.container(border=True):
            if geo is None:
                st.warning("The district boundary file is missing.")
            else:
                on_map = shown[shown.district.isin({f["id"] for f in geo["features"]})]
                scale = dict(colorscale="YlGn", zmin=2000, zmax=6500) if col == "yield" else dict(colorscale="RdYlGn", zmid=0, zmin=-40, zmax=40)
                fig = go.Figure(go.Choropleth(geojson=geo, featureidkey="id", locations=on_map.district, z=on_map[col],
                                              text=on_map.district.str.title(), marker_line_color="white", marker_line_width=0.8,
                                              colorbar=dict(title="kg/ha" if col == "yield" else "%", thickness=12, len=0.75),
                                              hovertemplate="%{text}<br>%{z:,.0f}" + (" kg/ha" if col == "yield" else "%") + "<extra></extra>",
                                              **scale))
                if district in set(on_map.district):  # outline the district chosen in the input panel
                    fig.add_trace(go.Choropleth(geojson=geo, featureidkey="id", locations=[district], z=[0], showscale=False,
                                                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                                marker_line_color=DARK, marker_line_width=2.5, hoverinfo="skip"))
                fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
                fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                  title=dict(text=f"{'Yield' if col == 'yield' else 'Change from the year before'}, {label}", x=0.01,
                                             font=dict(family="Georgia, serif", size=15, color=DARK)))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with g2, st.container(border=True):
            rank = shown.sort_values(col)
            colours = [DARK if d == district else (GREEN if (v >= 0 or col == "yield") else RED) for d, v in zip(rank.district, rank[col])]
            fig = go.Figure(go.Bar(x=rank[col], y=rank.district.str.title(), orientation="h", marker_color=colours,
                                   text=[f"{v:,.0f}" if col == "yield" else f"{v:+.0f}%" for v in rank[col]], textposition="outside",
                                   cliponaxis=False, hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
            style(fig, 560, "District ranking" + (" (kg/ha)" if col == "yield" else " (% change)"))
            fig.update_layout(margin=dict(r=50), yaxis=dict(tickfont=dict(size=10.5)), bargap=0.25)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        top = map_df.sort_values("yield", ascending=False)
        n1, n2, n3 = st.columns(3)
        with n1:
            card(f"Highest · {label}", f"{top.iloc[0]['yield']:,.0f}", f"kg/ha · {top.iloc[0]['district'].title()}", kind="main")
        with n2:
            card(f"Lowest · {label}", f"{top.iloc[-1]['yield']:,.0f}", f"kg/ha · {top.iloc[-1]['district'].title()}", kind="warn")
        with n3:
            card(f"Median district · {label}", f"{map_df['yield'].median():,.0f}", f"kg/ha across {len(map_df)} reporting units")
        st.caption("Official DCS district yields. Choose Change and move the slider to 2021 to see the fertilizer-ban season. The district chosen "
                   "in the input panel is outlined. Mahaweli 'H' and Udawalawe are irrigation areas without their own boundary, so they "
                   "appear in the ranking only (reported until 2018). Boundaries from geoBoundaries.")

if view == "Model comparison":
    if os.path.exists(RESULTS_PATH):
        res = pd.read_csv(RESULTS_PATH)
        res["Model"] = res["Model"].str.replace(r" \{.*\}", "", regex=True)
        res["Model"] = res["Model"].replace({  # shorter names so the chart labels fit on a phone
            "Existing stacked XGBoost (Senevirathne & Premasiri, 2025)": "Existing stacked XGBoost (2025)",
            "Stacking Ridge + RF + LightGBM + CatBoost -> Ridge": "Stacking (4 models -> Ridge)"})
        res["Model"] = res["Model"].str.replace("Existing ", "Published ", regex=False)  # comparison methods, not candidates
        # three separate views, selection among the 20 candidates on validation, their test score, and the benchmark against published methods
        VIEWS_MC = ["Validation 2010-2016 (used to choose)", "Test 2017-2026 (final check)", "Published methods (benchmark)"]
        stage = st.segmented_control("Compare", VIEWS_MC, default=VIEWS_MC[0], key="rank_by") or VIEWS_MC[0]
        candidates = res[~res.Model.str.startswith("Published")]
        table_cfg = {"Model": st.column_config.TextColumn("Model", width="medium"),
                     "CV MAE": st.column_config.NumberColumn("Val. MAE", format="%.0f"),
                     "Test MAE": st.column_config.NumberColumn("Test MAE", format="%.0f"),
                     "Test R2": st.column_config.NumberColumn("Test R²", format="%.2f")}
        c1, c2 = st.columns([1.25, 1])
        if stage == VIEWS_MC[2]:
            bench = res[res.Model.str.startswith("Published") | res.Model.isin([bundle["model_name"], "Naive average of last 3 same seasons"])]
            bench = bench.set_index("Model").loc[[bundle["model_name"], "Published SVR, RBF kernel (CJS 2021)", "Published stacked XGBoost (2025)",
                                                  "Naive average of last 3 same seasons"]].reset_index()
            with c1, st.container(border=True):
                fig = go.Figure()
                for metric, label, colour in [("CV MAE", "validation 2010-2016", "#A9C5B4"), ("Test MAE", "test 2017-2026", DARK)]:
                    fig.add_trace(go.Bar(y=bench.Model, x=bench[metric], orientation="h", name=label, marker_color=colour,
                                         text=[f"{v:,.0f}" for v in bench[metric]], textposition="outside", cliponaxis=False))
                style(fig, 420, "Deployed model against published methods (MAE, kg/ha)")
                fig.update_layout(barmode="group", margin=dict(r=50), yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(size=11)),
                                  xaxis=dict(range=[300, bench["Test MAE"].max() * 1.1]),
                                  legend=dict(orientation="h", y=-0.15, x=0, xanchor="left", yanchor="top"))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            with c2:
                st.markdown("The chosen model is compared with existing approaches to the same problem. Two published Sri Lankan paddy "
                            "methods were rebuilt on the same data and the same time split, a stacked XGBoost (Senevirathne and "
                            "Premasiri, 2025) and a support vector regression (Ekanayake et al., 2021). They are a benchmark, not candidates "
                            "for selection. The deployed EBM matches them on the test seasons (446 kg/ha against 443 and 451) and all three "
                            "clearly beat the naive three-season average (532). The scores in the original papers used other districts, years "
                            "and inputs, so only this rebuilt comparison is like for like.")
                st.dataframe(bench[["Model", "CV MAE", "Test MAE", "Test R2"]], hide_index=True, width="stretch", column_config=table_cfg)
        else:
            col, title = ("CV MAE", "Validation error 2010-2016 (kg/ha)") if stage == VIEWS_MC[0] else ("Test MAE", "Test error 2017-2026 (kg/ha)")
            plot = candidates.dropna(subset=["CV MAE"]).sort_values(col, ascending=False)
            colours = [GOLD if m == bundle["model_name"] else (GREY if m.startswith(("Naive", "ARIMA(")) else GREEN) for m in plot.Model]
            with c1, st.container(border=True):
                fig = go.Figure(go.Bar(x=plot[col], y=plot["Model"], orientation="h", marker_color=colours,
                                       text=[f"{v:,.0f}" for v in plot[col]], textposition="outside", cliponaxis=False))
                style(fig, 520, title)
                fig.update_layout(margin=dict(r=50), yaxis=dict(tickfont=dict(size=11), automargin=True),
                                  xaxis=dict(range=[plot[col].min() * 0.95, plot[col].max() * 1.06]))  # start near the best score so the gaps are visible
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            with c2:
                if col == "CV MAE":
                    st.markdown("**Gold** is the deployed model, **grey** the simple baselines. The choice was made on these seven "
                                "validation years only. Among the 20 candidate models, NGBoost (343) and the Explainable Boosting Machine (347) "
                                "were within 1% of each other, so the one whose error varied least from year to year, the EBM, was deployed.")
                else:
                    st.markdown("**Gold** is the deployed model. The test seasons were used once, after the choice was made. Smooth models "
                                "such as the Gaussian Process and Ridge did best here because 2017-2026 brought record yields and then a "
                                "crash, but they were not chosen, as choosing on the test seasons would make this score optimistic. The "
                                "difference between the EBM and Ridge is not significant (p = 0.14), and the EBM beat NGBoost, the "
                                "validation leader (446 against 459).")
                st.dataframe(candidates[["Model", "CV MAE", "Test MAE", "Test R2"]].sort_values(col), hide_index=True, width="stretch",
                             height=420, column_config=table_cfg)

if view == "Batch upload":
    st.markdown("Estimate many district-seasons at once. Download the example file, which holds every district with its typical values, "
                "change any values, and upload it again. Each row is sent to the prediction API, which checks it in the same way as a "
                "single estimate.")
    examples = [r for r in (typical_inputs(d, s_) for d in bundle["districts"] for s_ in ("Maha", "Yala")) if r]
    template = pd.DataFrame(examples)
    b1, b2 = st.columns([1, 2.2])
    with b1:
        st.download_button("Download example file", template.to_csv(index=False).encode(), file_name="paddy_batch_example.csv",
                           mime="text/csv", width="stretch", icon=":material/download:")
        uploaded = st.file_uploader("Upload a CSV file", type="csv")
        use_example = st.checkbox("Use the example file", value=uploaded is None)
    table = pd.read_csv(uploaded) if uploaded is not None else (template if use_example else None)
    if table is not None:
        missing = [c for c in template.columns if c not in table.columns and c not in ("tmean", "tmin", "solar")]
        if missing:
            b2.error("The file is missing these columns " + ", ".join(missing))
        elif len(table) > 500:
            b2.error("Please upload at most 500 rows.")
        else:
            table["fertilizer_shortage"] = table.get("fertilizer_shortage", False)
            table["fertilizer_shortage"] = table["fertilizer_shortage"].astype(str).str.lower().isin(["true", "1", "yes"])
            try:
                table["estimated_yield_kg_ha"] = np.round(estimate_many(table.to_dict("records")), 1)
                table["harvest_outlook"] = [harvest_class(e, r) for e, r in zip(table.estimated_yield_kg_ha, table.yield_same_season_roll3)]
                table["production_t"] = (table.estimated_yield_kg_ha * table.sown_ha / 1e3).round(0)
                with b2:
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        card("Seasons estimated", f"{len(table)}", "rows in the file")
                    with k2:
                        w = table.sown_ha / table.sown_ha.sum()
                        card("Area-weighted yield", f"{(table.estimated_yield_kg_ha * w).sum():,.0f}", "kg/ha across all rows", kind="main")
                    with k3:
                        card("Total production", f"{table.production_t.sum() / 1e6:,.2f} Mt",
                             f"{(table.harvest_outlook == 'poor').sum()} rows with a poor outlook", kind="warn")
                show_cols = ["district", "season", "sown_ha", "estimated_yield_kg_ha", "harvest_outlook", "production_t", "fertilizer_shortage"]
                view_table = table[show_cols].assign(district=table.district.str.title())
                st.dataframe(view_table, hide_index=True, width="stretch", height=380,
                             column_config={"district": "District", "season": "Season",
                                            "sown_ha": st.column_config.NumberColumn("Sown (ha)", format="%.0f"),
                                            "estimated_yield_kg_ha": st.column_config.NumberColumn("Estimate (kg/ha)", format="%.0f"),
                                            "harvest_outlook": "Outlook",
                                            "production_t": st.column_config.NumberColumn("Production (t)", format="%.0f"),
                                            "fertilizer_shortage": "Shortage"})
                st.download_button("Download the results", table.to_csv(index=False).encode(), file_name="paddy_batch_estimates.csv",
                                   mime="text/csv", icon=":material/download:")
                st.caption("The example file uses each district's most recent sown extent, recent yields and median weather, so its "
                           "totals are an illustration and not a forecast for a particular season.")
            except ValueError as e:
                b2.error(f"The API rejected the file. {e}")

if view == "About":
    a1, a2 = st.columns([1.4, 1], gap="large")
    with a1:
        st.markdown(f"""
##### What the tool does
It estimates the average paddy yield (kg/ha) of a Sri Lankan district for a Maha or Yala season as soon as the season ends, \
months before the Department of Census and Statistics publishes the official figure.

##### How it was built
The model is an {bundle['model_name']} trained on {bundle['trained_on']}. It was chosen from twenty approaches, including \
Random Forest, XGBoost, LightGBM, CatBoost, stacking and ARIMA hybrids, using yearly validation folds from 2010 to 2016. \
On the unseen 2017-2026 test seasons its mean absolute error was {mae:,.0f} kg/ha, against 532 kg/ha for the best naive baseline.

##### Data sources
Eleven datasets were tested and the model uses five of them, namely the Department of Census and Statistics paddy statistics, \
CHIRPS satellite rainfall, NASA POWER weather, the fertilizer policy record and OpenStreetMap district locations. \
The map uses district boundaries from geoBoundaries.

##### Limitations
The model learns from past seasons, so a shock it has never seen can push the estimate off. The fertilizer-shortage effect was \
learned from only two seasons and should be read as a scenario. Mahaweli 'H' and Udawalawe were reported only until 2018.
""")
    with a2:
        card("Test error (MAE)", f"{mae:,.0f} kg/ha", "2017-2026 seasons, never used for training or selection", kind="main")
        st.write("")
        card("Accuracy", "87.6%", "100% minus the mean absolute percentage error on the test seasons")
        st.write("")
        st.link_button("Source code on GitHub", REPO_URL, width="stretch", icon=":material/code:")
        if API_URL:  # the main deployment on Azure, which also links to the backup copy
            st.link_button("Prediction API documentation", f"{API_URL}/docs", width="stretch", icon=":material/api:")
            st.link_button("Backup copy on Streamlit Community Cloud", BACKUP_URL, width="stretch", icon=":material/backup:")
        else:  # the backup copy, which runs the bundled model and links back to the Azure deployment
            st.link_button("Main version on Azure", AZURE_APP_URL, width="stretch", icon=":material/cloud:")
            st.link_button("Prediction API documentation", AZURE_API_DOCS, width="stretch", icon=":material/api:")
