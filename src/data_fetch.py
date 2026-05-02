from datetime import date
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import requests
from config import SENTRY_URL, CAD_URL, MAX_OBJECTS, PARAMETER_SETS, ACTIVE_PARAMETER_SET


def to_float(value):
    try:
        if value is None:
            return np.nan
        return float(str(value).replace(",", "").strip())
    except Exception:
        return np.nan


def parse_days(value):
    if value is None:
        return np.nan
    
    match = re.search(r"[-+]?\d*\.?\d+", str(value))

    if match is None:
        return np.nan
    
    return float(match.group(0))


def get_json(url, params=None):
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "error" in payload and payload.get("error"):
        raise RuntimeError(str(payload.get("error")))
    
    return payload


def fetch_sentry_summary():
    payload = get_json(SENTRY_URL)
    records = payload.get("data", [])

    if not records:
        raise RuntimeError("Sentry returned no objects.")
    
    df = pd.DataFrame(records)
    numeric_columns = ["diameter", "h", "ip", "n_imp", "ps_cum", "ps_max", "ts_max", "v_inf"]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = df[column].map(to_float)
    df = df.rename(columns={
        "des": "designation",
        "fullname": "full_name",
        "ip": "impact_probability",
        "ps_cum": "palermo_scale",
        "ps_max": "palermo_scale_max",
        "v_inf": "velocity_km_s",
        "diameter": "diameter_km",
        "n_imp": "potential_impacts"
    })
    df = df.sort_values("palermo_scale", ascending=False).head(MAX_OBJECTS).reset_index(drop=True)
    return df


def fetch_sentry_details(designation):
    try:
        payload = get_json(SENTRY_URL, {"des": designation})
        summary = payload.get("summary", {})
        return {
            "observations": to_float(summary.get("nobs")),
            "arc_days": parse_days(summary.get("darc")),
            "first_obs": summary.get("first_obs"),
            "last_obs": summary.get("last_obs"),
            "impact_velocity_km_s": to_float(summary.get("v_imp"))
        }
    except Exception:
        return {
            "observations": np.nan,
            "arc_days": np.nan,
            "first_obs": None,
            "last_obs": None,
            "impact_velocity_km_s": np.nan
        }


def fetch_closest_approach(designation):
    params = {
        "des": designation,
        "date-min": str(date.today()),
        "date-max": "+36525",
        "dist-max": "0.2",
        "sort": "dist",
        "limit": "1",
        "diameter": "true",
        "fullname": "true"
    }
    try:
        payload = get_json(CAD_URL, params)
        fields = payload.get("fields", [])
        rows = payload.get("data", [])

        if not rows:
            return {
                "close_approach_date": None,
                "miss_distance_au": np.nan,
                "cad_velocity_km_s": np.nan,
                "cad_diameter_km": np.nan
            }
        row = dict(zip(fields, rows[0]))

        return {
            "close_approach_date": row.get("cd"),
            "miss_distance_au": to_float(row.get("dist")),
            "cad_velocity_km_s": to_float(row.get("v_rel")),
            "cad_diameter_km": to_float(row.get("diameter"))
        }
    except Exception:
        return {
            "close_approach_date": None,
            "miss_distance_au": np.nan,
            "cad_velocity_km_s": np.nan,
            "cad_diameter_km": np.nan
        }


def robust_norm(series, lower=0.05, upper=0.95):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    
    if values.notna().sum() == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    
    fill_value = values.median()
    values = values.fillna(fill_value)
    lo = values.quantile(lower)
    hi = values.quantile(upper)

    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        return pd.Series(np.zeros(len(values)), index=values.index)
    
    clipped = values.clip(lo, hi)

    return ((clipped - lo) / (hi - lo)).clip(0, 1)


def rev_robust_norm(series, lower=0.05, upper=0.95):
    return 1 - robust_norm(series, lower, upper)


def add_details(df):
    detail_rows = [fetch_sentry_details(des) for des in df["designation"]]
    ca_rows = [fetch_closest_approach(des) for des in df["designation"]]
    details = pd.DataFrame(detail_rows)
    close_approaches = pd.DataFrame(ca_rows)

    combined = pd.concat([df.reset_index(drop=True), details, close_approaches], axis=1)
    combined["diameter_km"] = combined["diameter_km"].fillna(combined["cad_diameter_km"])
    combined["velocity_km_s"] = combined["cad_velocity_km_s"].fillna(combined["velocity_km_s"])
    combined["observations"] = combined["observations"].fillna(combined["potential_impacts"])
    combined["arc_days"] = combined["arc_days"].fillna(combined["observations"].median())
    combined["miss_distance_au"] = combined["miss_distance_au"].fillna(combined["miss_distance_au"].median())
    combined["diameter_km"] = combined["diameter_km"].fillna(combined["diameter_km"].median())
    combined["velocity_km_s"] = combined["velocity_km_s"].fillna(combined["velocity_km_s"].median())
    return combined


def add_priority_features(df, parameter_set=None):
    params = PARAMETER_SETS[parameter_set or ACTIVE_PARAMETER_SET]

    result = df.copy()
    result["risk_score"] = robust_norm(result["palermo_scale"])
    result["size_score"] = robust_norm(np.log1p(result["diameter_km"]))
    result["speed_score"] = robust_norm(result["velocity_km_s"])
    result["uncertainty_need"] = 0.55 * rev_robust_norm(np.log1p(result["observations"])) + 0.45 * rev_robust_norm(np.log1p(result["arc_days"]))
    result["close_approach_score"] = rev_robust_norm(result["miss_distance_au"])
    result["risk_weight"] = result["risk_score"].clip(0, 1)
    result["information_rate"] = params["a0"] + params["alpha_q"] * result["uncertainty_need"] + params["alpha_c"] * result["close_approach_score"]
    result["max_useful_hours"] = params["u0"] + params["beta_q"] * result["uncertainty_need"] + params["beta_r"] * result["risk_score"]
    result["max_useful_hours"] = result["max_useful_hours"].clip(0.5, 8.0)
    result["parameter_set"] = parameter_set or ACTIVE_PARAMETER_SET
    return result


def load_or_fetch_data(cache_path="outputs/processed_neo_priority.csv", refresh=False, parameter_set=None):
    path = Path(cache_path)

    if path.exists() and not refresh:
        return pd.read_csv(path)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_sentry_summary()
    df = add_details(df)
    df = add_priority_features(df, parameter_set=parameter_set)
    df.to_csv(path, index=False)
    return df
