#!/usr/bin/env python3
"""
Comprehensive client analytics pipeline for Retail contract data.

This script reproduces the full analysis that was executed interactively:
1. Data quality profiling
2. Feature engineering
3. Global metrics
4. Temporal trends
5. Cohort & retention analytics
6. Location deep-dives
7. Nationality segmentation
8. City-level insights
9. Payment behaviour
10. Anomaly detection
11. Missing data forensics

Outputs are written as CSV/JSON files beside the input dataset.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Retail contract IDs - Sheet1.csv"
ENRICHED_CSV = BASE_DIR / "contracts_enriched.csv"
KIOSK_CLEAN_CSV = BASE_DIR / "contracts_kiosk_clean.csv"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist dict payload as pretty-printed JSON."""
    path.write_text(json.dumps(payload, indent=2))


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def convert_series_to_dict(series: pd.Series, *, to_int: bool = False) -> Dict[str, Any]:
    if to_int:
        return {str(k): int(v) for k, v in series.items()}
    return {str(k): safe_float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in series.items()}


def render_figure_to_data_uri(fig: Figure) -> str:
    """Convert a Matplotlib figure to a base64 data URI string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def plot_horizontal_bar(data: Dict[str, float], title: str, xlabel: str) -> str | None:
    """Render a horizontal bar chart and return a data URI."""
    if not data:
        return None

    labels = list(data.keys())
    values = [float(v) for v in data.values()]
    height = max(4.0, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(10, height))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color="#4c72b0")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return render_figure_to_data_uri(fig)


def plot_line_chart(data: Dict[str, float], title: str, xlabel: str, ylabel: str) -> str | None:
    """Render a line chart and return a data URI."""
    if not data:
        return None

    items = sorted(((str(k), float(v)) for k, v in data.items()), key=lambda pair: pair[0])
    labels = [label for label, _ in items]
    values = [value for _, value in items]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(labels, values, marker="o", color="#2a9d8f")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return render_figure_to_data_uri(fig)


def series_to_html_table(
    data: Dict[str, Any],
    columns: Tuple[str, str],
    *,
    sort_desc: bool = False,
    limit: int | None = None,
) -> str:
    """Convert a dictionary to an HTML table."""
    if not data:
        return "<p>No data available.</p>"

    items = list(data.items())
    if sort_desc:
        items.sort(key=lambda item: item[1], reverse=True)
    if limit is not None:
        items = items[:limit]

    df = pd.DataFrame(items, columns=list(columns))
    return df.to_html(index=False, classes="data-table", border=0)


def records_to_html_table(
    records: List[Dict[str, Any]],
    columns: Iterable[str] | None = None,
    *,
    rename: Dict[str, str] | None = None,
    limit: int | None = None,
) -> str:
    """Convert a list of dictionaries to an HTML table."""
    if not records:
        return "<p>No records available.</p>"

    df = pd.DataFrame(records)
    if rename:
        df = df.rename(columns=rename)
    if columns is not None:
        selected = [col for col in columns if col in df.columns]
        df = df[selected]
    if limit is not None:
        df = df.head(limit)

    return df.to_html(index=False, classes="data-table", border=0)


# --------------------------------------------------------------------------- #
# Step 1: Load & audit
# --------------------------------------------------------------------------- #

@dataclass
class LoadResult:
    raw_df: pd.DataFrame
    column_types: Dict[str, str]
    duplicates: Dict[str, int]


def load_and_audit() -> LoadResult:
    df = pd.read_csv(INPUT_FILE)
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    duplicate_metrics = {}
    for col in ["Contract ID", "Interview ID", "Contract Client ID"]:
        if col not in df.columns:
            continue
        duplicate_metrics[col] = int(df.duplicated(subset=[col]).sum())
    if {"Contract ID", "Interview ID"}.issubset(df.columns):
        duplicate_metrics["Contract+Interview"] = int(df.duplicated(subset=["Contract ID", "Interview ID"]).sum())
    duplicate_metrics["Full Record"] = int(df.duplicated().sum())

    return LoadResult(raw_df=df, column_types=column_types, duplicates=duplicate_metrics)


# --------------------------------------------------------------------------- #
# Step 2: Data quality
# --------------------------------------------------------------------------- #

def run_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    blank_counts = {}
    for col in df.columns:
        if df[col].dtype == "object":
            blank_counts[col] = int((df[col].astype(str).str.strip() == "").sum())

    datetime_parse = {}
    for col in ["Contract Creation Date", "Paid Date", "Termination Date"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        non_null = int(df[col].notna().sum())
        invalid = int(((parsed.isna()) & df[col].notna()).sum())
        datetime_parse[col] = {"non_null": non_null, "invalid": invalid}

    string_whitespace = {}
    for col in df.select_dtypes(include="object").columns:
        stripped = df[col].astype(str).str.strip()
        diff = (df[col] != stripped) & df[col].notna()
        string_whitespace[col] = int(diff.sum())

    return {
        "missing": missing_counts.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "blank_counts": blank_counts,
        "duplicates": load_and_audit().duplicates,  # recompute for completeness
        "datetime_parse": datetime_parse,
        "string_whitespace": string_whitespace,
    }


# --------------------------------------------------------------------------- #
# Step 3: Feature engineering
# --------------------------------------------------------------------------- #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols = ["Contract Creation Date", "Paid Date", "Termination Date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    creation = df["Contract Creation Date"]
    termination = df["Termination Date"]
    paid = df["Paid Date"]

    contract_duration = (termination - creation).dt.total_seconds() / 86400
    contract_duration = contract_duration.where(termination.notna())
    df["contract_duration_days"] = contract_duration

    analysis_date = pd.Timestamp.today().normalize()
    active_duration = (analysis_date - creation).dt.total_seconds() / 86400
    active_duration = active_duration.where(termination.isna())
    df["active_duration_days"] = active_duration

    time_to_payment = (paid - creation).dt.total_seconds() / 86400
    df["time_to_payment_days"] = time_to_payment

    df["creation_year"] = creation.dt.year
    df["creation_quarter"] = creation.dt.to_period("Q").astype(str)
    df["creation_month"] = creation.dt.to_period("M").astype(str)
    df["creation_week"] = creation.dt.isocalendar().week.astype(int)
    df["creation_day_of_week"] = creation.dt.day_name()
    df["creation_hour"] = creation.dt.hour

    df["is_terminated"] = termination.notna()
    df["is_paid"] = paid.notna()
    df["is_termination_missing"] = termination.isna()
    df["contract_status"] = np.where(df["is_terminated"], "Terminated", "Active")

    df.to_csv(ENRICHED_CSV, index=False)
    return df


# --------------------------------------------------------------------------- #
# Step 4: Global metrics
# --------------------------------------------------------------------------- #

def compute_global_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    status_counts = df["contract_status"].value_counts()
    location_counts = df.groupby("Interview Location").size().sort_values(ascending=False)
    location_share = (location_counts / total * 100).round(2)

    monthly = df.copy()
    monthly["creation_month"] = monthly["Contract Creation Date"].dt.to_period("M")
    month_counts = monthly.groupby("creation_month").size().sort_index()

    location_growth = {}
    for loc, group in monthly.groupby("Interview Location"):
        series = group.groupby("creation_month").size().sort_index()
        first_month = series.index[0]
        last_month = series.index[-1]
        first_count = int(series.iloc[0])
        last_count = int(series.iloc[-1])
        growth_rate = None
        if first_count != 0:
            growth_rate = round((last_count - first_count) / first_count, 3)
        location_growth[loc] = {
            "first_month": str(first_month),
            "first_count": first_count,
            "last_month": str(last_month),
            "last_count": last_count,
            "growth_rate": growth_rate,
        }

    nationality_counts = df["Client Nationality"].value_counts(dropna=False).sort_values(ascending=False)
    nationality_share = (nationality_counts / total * 100).round(2)

    city_counts = df["City"].value_counts(dropna=False).sort_values(ascending=False)
    city_share = (city_counts / total * 100).round(2)

    return {
        "total_contracts": total,
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "termination_rate": round(status_counts.get("Terminated", 0) / total, 4),
        "active_rate": round(status_counts.get("Active", 0) / total, 4),
        "location_counts": {k: int(v) for k, v in location_counts.head(20).items()},
        "location_share": {k: float(v) for k, v in location_share.head(20).items()},
        "location_bottom_counts": {k: int(v) for k, v in location_counts.tail(20).items()},
        "location_growth": location_growth,
        "nationality_top10": {k: int(v) for k, v in nationality_counts.head(10).items()},
        "nationality_share_top10": {k: float(v) for k, v in nationality_share.head(10).items()},
        "nationality_bottom10": {k: int(v) for k, v in nationality_counts.tail(10).items()},
        "city_counts": {k: int(v) for k, v in city_counts.head(20).items()},
        "city_share": {k: float(v) for k, v in city_share.head(20).items()},
        "cities_missing": int(df["City"].isna().sum()),
    }


# --------------------------------------------------------------------------- #
# Step 5: Temporal analysis
# --------------------------------------------------------------------------- #

def analyze_temporal(df: pd.DataFrame) -> Dict[str, Any]:
    creation = df["Contract Creation Date"]
    df = df.copy()
    df["creation_month"] = creation.dt.to_period("M")

    monthly_counts = df.groupby("creation_month").size().sort_index()
    monthly_growth = monthly_counts.pct_change().round(3)
    monthly_ma = monthly_counts.rolling(3).mean().round(1)

    weekday_counts = df.groupby(df["creation_day_of_week"]).size()
    weekday_counts = weekday_counts.reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    weekday_share = (weekday_counts / len(df) * 100).round(2)

    df["is_weekend"] = df["creation_day_of_week"].isin(["Friday", "Saturday", "Sunday"])
    weekend_split = df.groupby("is_weekend").size()

    hour_counts = df.groupby("creation_hour").size().sort_index()
    quarter_counts = df.groupby(creation.dt.to_period("Q")).size().sort_index()

    payment_lag = df.groupby(df["creation_month"])["time_to_payment_days"].agg(["median", "mean", "count"]).round(
        {"median": 2, "mean": 2}
    )

    return {
        "monthly_counts": {str(k): int(v) for k, v in monthly_counts.items()},
        "monthly_growth": {str(k): (None if pd.isna(v) else float(v)) for k, v in monthly_growth.items()},
        "monthly_moving_avg": {str(k): (None if pd.isna(v) else float(v)) for k, v in monthly_ma.items()},
        "weekday_counts": {k: int(v) for k, v in weekday_counts.dropna().items()},
        "weekday_share": {k: float(v) for k, v in weekday_share.dropna().items()},
        "weekend_split": {str(k): int(v) for k, v in weekend_split.items()},
        "hour_counts": {int(k): int(v) for k, v in hour_counts.items()},
        "quarter_counts": {str(k): int(v) for k, v in quarter_counts.items()},
        "payment_lag_by_month": {
            str(idx): {metric: safe_float(val) for metric, val in row.dropna().items()}
            for idx, row in payment_lag.iterrows()
        },
        "time_range": {"min": str(creation.min()), "max": str(creation.max())},
    }


# --------------------------------------------------------------------------- #
# Step 6: Cohort & retention
# --------------------------------------------------------------------------- #

def cohort_and_retention(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    analysis_date = pd.Timestamp.today().normalize()
    df["cohort_month"] = df["Contract Creation Date"].dt.to_period("M")
    df["end_date"] = df["Termination Date"].fillna(analysis_date)

    months_active = (
        (df["end_date"].dt.year - df["Contract Creation Date"].dt.year) * 12
        + (df["end_date"].dt.month - df["Contract Creation Date"].dt.month)
    ).clip(lower=0)
    df["months_active"] = months_active

    cohort_sizes = df.groupby("cohort_month").size()
    max_age = int(months_active.max())

    retention_table = {}
    for cohort, cohort_df in df.groupby("cohort_month"):
        size = len(cohort_df)
        retention = []
        for age in range(max_age + 1):
            active = (cohort_df["months_active"] >= age).sum()
            retention.append(round(active / size, 4))
        retention_table[str(cohort)] = {"size": int(size), "retention": retention}

    is_event = df["is_terminated"]
    durations = np.where(
        is_event,
        (df["Termination Date"] - df["Contract Creation Date"]).dt.days,
        (analysis_date - df["Contract Creation Date"]).dt.days,
    )
    durations = np.maximum(durations, 0)

    km_df = pd.DataFrame({"duration": durations, "event": is_event.astype(int)}).sort_values("duration")
    n_at_risk = len(km_df)
    survival, hazard, timeline = [], [], []
    prob = 1.0
    for duration, group in km_df.groupby("duration"):
        d = int(group["event"].sum())
        timeline.append(int(duration))
        hazard_rate = d / n_at_risk if n_at_risk else 0
        hazard.append(round(hazard_rate, 5))
        if n_at_risk:
            prob *= (1 - hazard_rate)
        survival.append(round(prob, 5))
        n_at_risk -= len(group)

    duration_series = pd.Series(durations)
    duration_stats = {
        "terminated_mean_days": float(duration_series[is_event].mean()),
        "terminated_median_days": float(duration_series[is_event].median()),
        "active_mean_days": float(duration_series[~is_event].mean()),
        "overall_mean_days": float(duration_series.mean()),
        "overall_median_days": float(duration_series.median()),
    }

    termination_month = df.loc[is_event, "Termination Date"].dt.to_period("M")
    churn_counts = termination_month.value_counts().sort_index()
    churn_rate_by_month = {}
    for month, churn in churn_counts.items():
        base = len(df[df["cohort_month"] <= month])
        churn_rate_by_month[str(month)] = round(churn / base, 4) if base else None

    return {
        "cohort_sizes": {str(k): int(v) for k, v in cohort_sizes.items()},
        "cohort_retention_table": retention_table,
        "max_age_months": max_age,
        "kaplan_meier": {
            "timeline_days": timeline,
            "survival": survival,
            "hazard": hazard,
        },
        "duration_stats": duration_stats,
        "churn_counts_by_month": {str(k): int(v) for k, v in churn_counts.items()},
        "churn_rate_by_month": churn_rate_by_month,
    }


# --------------------------------------------------------------------------- #
# Step 7: Location deep dive
# --------------------------------------------------------------------------- #

def location_deep_dive(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    location_counts = df["Interview Location"].value_counts()
    top_locations = location_counts.head(10).index.tolist()

    results = {}
    monthly = df.copy()
    monthly["creation_month"] = monthly["Contract Creation Date"].dt.to_period("M")
    all_months = sorted(monthly["creation_month"].unique())
    last_six = all_months[-6:] if len(all_months) >= 6 else all_months
    prev_six = all_months[-12:-6] if len(all_months) >= 12 else []

    for loc in top_locations:
        loc_df = df[df["Interview Location"] == loc].copy()
        loc_total = len(loc_df)
        status_counts = loc_df["contract_status"].value_counts()

        terminated = loc_df[loc_df["contract_status"] == "Terminated"]
        duration_stats = {
            "mean_days": safe_float(terminated["contract_duration_days"].mean()),
            "median_days": safe_float(terminated["contract_duration_days"].median()),
        }

        payment_stats = {
            "median_time_to_payment_days": safe_float(loc_df["time_to_payment_days"].median()),
            "p75_time_to_payment_days": safe_float(loc_df["time_to_payment_days"].quantile(0.75)),
        }

        nat_counts = loc_df["Client Nationality"].value_counts(dropna=False).head(10)
        nat_share = (nat_counts / loc_total * 100).round(2)

        loc_monthly = monthly[monthly["Interview Location"] == loc]
        month_counts = loc_monthly.groupby("creation_month").size().sort_index()
        last_counts = {str(m): int(month_counts.get(m, 0)) for m in last_six}
        prev_counts = {str(m): int(month_counts.get(m, 0)) for m in prev_six}
        last_avg = sum(last_counts.values()) / len(last_counts) if last_counts else 0
        prev_avg = sum(prev_counts.values()) / len(prev_counts) if prev_counts else 0
        growth = None
        if prev_avg > 0:
            growth = round((last_avg - prev_avg) / prev_avg, 3)

        peak_day = loc_df["creation_day_of_week"].value_counts().idxmax()
        peak_hour = int(loc_df["creation_hour"].value_counts().idxmax())

        results[loc] = {
            "total_contracts": int(loc_total),
            "share_pct": round(loc_total / total * 100, 2),
            "status_counts": {k: int(v) for k, v in status_counts.items()},
            "termination_rate": round(status_counts.get("Terminated", 0) / loc_total, 4),
            "duration_stats": duration_stats,
            "payment_stats": payment_stats,
            "top_nationalities": {
                str(k): {"count": int(v), "share_pct": float(nat_share.loc[k])}
                for k, v in nat_counts.items()
            },
            "monthly_last6": last_counts,
            "monthly_prev6": prev_counts,
            "growth_last6_vs_prev6": growth,
            "peak_day": peak_day,
            "peak_hour": peak_hour,
        }

    benchmark = pd.DataFrame({
        "total_contracts": location_counts,
        "termination_rate": df.groupby("Interview Location")["is_terminated"].mean(),
        "median_duration_days": df[df["contract_status"] == "Terminated"]
        .groupby("Interview Location")["contract_duration_days"]
        .median(),
        "median_time_to_payment_days": df.groupby("Interview Location")["time_to_payment_days"].median(),
    }).fillna(0)

    return {
        **results,
        "benchmark_metrics": benchmark.head(20).round(3).to_dict(orient="index"),
    }


# --------------------------------------------------------------------------- #
# Step 8: Nationality segmentation
# --------------------------------------------------------------------------- #

def nationality_segmentation(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df["Client Nationality"] = df["Client Nationality"].fillna("Unknown")
    total = len(df)

    nat_counts = df["Client Nationality"].value_counts()
    top_nats = nat_counts.head(10).index.tolist()

    pivot = (
        df[df["Client Nationality"].isin(top_nats)]
        .pivot_table(index="Interview Location", columns="Client Nationality", values="Contract ID", aggfunc="count", fill_value=0)
    )

    result_matrix = {}
    for loc in pivot.index:
        loc_values = {}
        for nat in pivot.columns:
            val = int(pivot.loc[loc, nat])
            if val:
                loc_values[str(nat)] = val
        result_matrix[loc] = loc_values

    nat_metrics = {}
    for nat in top_nats:
        nat_df = df[df["Client Nationality"] == nat].copy()
        count = len(nat_df)
        top_locations = nat_df["Interview Location"].value_counts().head(5)
        loc_share = (top_locations / count * 100).round(2)

        terminated = nat_df[nat_df["is_terminated"]]
        nat_metrics[nat] = {
            "count": int(count),
            "share_pct": round(count / total * 100, 2),
            "top_locations": {
                loc: {"count": int(top_locations.loc[loc]), "share_pct": float(loc_share.loc[loc])}
                for loc in top_locations.index
            },
            "termination_rate": round(nat_df["is_terminated"].mean(), 4),
            "terminated_mean_duration_days": safe_float(terminated["contract_duration_days"].mean()),
            "terminated_median_duration_days": safe_float(terminated["contract_duration_days"].median()),
            "median_time_to_payment_days": safe_float(nat_df["time_to_payment_days"].median()),
        }

    diversity = {}
    for loc, group in df.groupby("Interview Location"):
        counts = group["Client Nationality"].value_counts()
        total_loc = counts.sum()
        probs = counts / total_loc
        shannon = -float(sum(p * log(p) for p in probs if p > 0))
        diversity[loc] = {
            "unique_nationalities": int(len(counts)),
            "shannon_index": shannon,
            "top_nationality_share": round(float(probs.max()), 4),
        }

    return {
        "top_nationalities_overall": {k: int(v) for k, v in nat_counts.head(20).items()},
        "location_nationality_matrix": result_matrix,
        "top_nationality_metrics": nat_metrics,
        "location_diversity": diversity,
    }


# --------------------------------------------------------------------------- #
# Step 9: City insights
# --------------------------------------------------------------------------- #

def city_insights(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df["City"] = df["City"].fillna("Unknown")
    city_counts = df["City"].value_counts()
    top_cities = city_counts.head(10).index.tolist()

    city_metrics = {}
    for city in top_cities:
        city_df = df[df["City"] == city].copy()
        total = len(city_df)
        terminated = city_df[city_df["is_terminated"]]
        top_kiosks = city_df["Interview Location"].value_counts().head(5)
        kiosk_share = (top_kiosks / total * 100).round(2)
        top_nats = city_df["Client Nationality"].value_counts().head(5)
        nat_share = (top_nats / total * 100).round(2)

        city_metrics[city] = {
            "total_contracts": int(total),
            "termination_rate": round(city_df["is_terminated"].mean(), 4),
            "avg_duration_days_terminated": safe_float(terminated["contract_duration_days"].mean()),
            "median_time_to_payment_days": safe_float(city_df["time_to_payment_days"].median()),
            "top_kiosks": {
                loc: {"count": int(top_kiosks.loc[loc]), "share_pct": float(kiosk_share.loc[loc])}
                for loc in top_kiosks.index
            },
            "top_nationalities": {
                nat: {"count": int(top_nats.loc[nat]), "share_pct": float(nat_share.loc[nat])}
                for nat in top_nats.index
            },
        }

    return {
        "city_counts": {k: int(v) for k, v in city_counts.items()},
        "top_city_metrics": city_metrics,
    }


# --------------------------------------------------------------------------- #
# Step 10: Payment behaviour
# --------------------------------------------------------------------------- #

def payment_behaviour(df: pd.DataFrame) -> Dict[str, Any]:
    payment = df["time_to_payment_days"]
    paid_mask = df["Paid Date"].notna()

    summary = {
        "paid_ratio": round(paid_mask.mean(), 4),
        "missing_paid_count": int((~paid_mask).sum()),
        "time_to_payment_stats": {
            "mean_days": safe_float(payment.mean()),
            "median_days": safe_float(payment.median()),
            "std_days": safe_float(payment.std()),
            "p90_days": safe_float(payment.quantile(0.9)),
            "p99_days": safe_float(payment.quantile(0.99)),
            "min_days": safe_float(payment.min()),
            "max_days": safe_float(payment.max()),
        },
        "negative_payment_count": int((payment < 0).sum()),
        "negative_payment_examples": df.loc[
            payment.sort_values().head(5).index,
            ["Contract ID", "Interview Location", "Contract Creation Date", "Paid Date", "time_to_payment_days"],
        ]
        .assign(
            **{
                "Contract Creation Date": lambda x: x["Contract Creation Date"].astype(str),
                "Paid Date": lambda x: x["Paid Date"].astype(str),
            }
        )
        .to_dict(orient="records"),
    }

    summary["location_payment_stats"] = (
        df.groupby("Interview Location")["time_to_payment_days"]
        .agg(["median", "mean", "count"])
        .round(3)
        .to_dict(orient="index")
    )

    nat_counts = df["Client Nationality"].fillna("Unknown").value_counts()
    top_nats = nat_counts.head(10).index.tolist()
    summary["nationality_payment_stats"] = (
        df[df["Client Nationality"].isin(top_nats)]
        .groupby("Client Nationality")["time_to_payment_days"]
        .agg(["median", "mean", "count"])
        .round(3)
        .to_dict(orient="index")
    )

    summary["slowest_payments"] = (
        df.nlargest(20, "time_to_payment_days")[
            ["Contract ID", "Interview Location", "Client Nationality", "time_to_payment_days", "Contract Creation Date", "Paid Date"]
        ]
        .assign(
            **{
                "Contract Creation Date": lambda x: x["Contract Creation Date"].astype(str),
                "Paid Date": lambda x: x["Paid Date"].astype(str),
            }
        )
        .to_dict(orient="records")
    )
    summary["fastest_payments"] = (
        df.nsmallest(20, "time_to_payment_days")[
            ["Contract ID", "Interview Location", "Client Nationality", "time_to_payment_days", "Contract Creation Date", "Paid Date"]
        ]
        .assign(
            **{
                "Contract Creation Date": lambda x: x["Contract Creation Date"].astype(str),
                "Paid Date": lambda x: x["Paid Date"].astype(str),
            }
        )
        .to_dict(orient="records")
    )

    return summary


# --------------------------------------------------------------------------- #
# Step 11: Anomalies
# --------------------------------------------------------------------------- #

def anomaly_detection(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    analysis_date = pd.Timestamp.today().normalize()
    df["Client Nationality"] = df["Client Nationality"].fillna("Unknown")

    terminated = df[df["is_terminated"]].copy()
    terminated["duration_days_calc"] = (terminated["Termination Date"] - terminated["Contract Creation Date"]).dt.days
    q1 = terminated["duration_days_calc"].quantile(0.25)
    q3 = terminated["duration_days_calc"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    long_outliers = terminated[terminated["duration_days_calc"] > upper].sort_values(
        "duration_days_calc", ascending=False
    ).head(20)

    short_outliers = terminated[terminated["duration_days_calc"] < 1].sort_values("duration_days_calc").head(20)

    active = df[~df["is_terminated"]].copy()
    active["duration_days_calc"] = (analysis_date - active["Contract Creation Date"]).dt.days
    active_outliers = active[active["duration_days_calc"] > 600].sort_values(
        "duration_days_calc", ascending=False
    ).head(20)

    payment = df["time_to_payment_days"]
    payment_neg = df[(payment > 14) | (payment < -7)].nsmallest(20, "time_to_payment_days")
    payment_high = df[payment > 14].nlargest(20, "time_to_payment_days")

    future_records = df[df["Contract Creation Date"] > analysis_date][
        ["Contract ID", "Interview Location", "Contract Creation Date"]
    ].assign(**{"Contract Creation Date": lambda x: x["Contract Creation Date"].astype(str)})
    pre_2023 = df[df["Contract Creation Date"] < pd.Timestamp("2023-01-01")][
        ["Contract ID", "Interview Location", "Contract Creation Date"]
    ].assign(**{"Contract Creation Date": lambda x: x["Contract Creation Date"].astype(str)})

    loc_term = df.groupby("Interview Location")["is_terminated"].mean()
    loc_counts = df["Interview Location"].value_counts()
    loc_term_df = pd.DataFrame({"termination_rate": loc_term, "count": loc_counts})
    loc_term_df["zscore"] = (
        (loc_term_df["termination_rate"] - loc_term_df["termination_rate"].mean())
        / loc_term_df["termination_rate"].std(ddof=0)
    )
    loc_term_outliers = loc_term_df[loc_term_df["zscore"].abs() > 2].round(3).to_dict(orient="index")

    return {
        "termination_duration_outliers": long_outliers[
            ["Contract ID", "Interview Location", "Client Nationality", "duration_days_calc"]
        ].to_dict(orient="records"),
        "short_duration_outliers": short_outliers[
            ["Contract ID", "Interview Location", "Client Nationality", "duration_days_calc"]
        ].to_dict(orient="records"),
        "active_duration_outliers": active_outliers[
            ["Contract ID", "Interview Location", "Client Nationality", "duration_days_calc"]
        ].to_dict(orient="records"),
        "payment_negative_outliers": payment_neg[
            ["Contract ID", "Interview Location", "Client Nationality", "time_to_payment_days"]
        ].to_dict(orient="records"),
        "payment_slow_outliers": payment_high[
            ["Contract ID", "Interview Location", "Client Nationality", "time_to_payment_days"]
        ].to_dict(orient="records"),
        "future_creation_records": future_records.to_dict(orient="records"),
        "pre_2023_records": pre_2023.to_dict(orient="records"),
        "location_termination_zscore": loc_term_outliers,
    }


# --------------------------------------------------------------------------- #
# Step 12: Missing data analysis
# --------------------------------------------------------------------------- #

def missing_data_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    for col in ["Client Nationality", "City", "Paid Date", "Termination Date"]:
        missing = df[col].isna()
        summary[col] = {
            "missing_count": int(missing.sum()),
            "missing_pct": round(missing.mean() * 100, 2),
        }

    nat_missing = df[df["Client Nationality"].isna()]
    nat_missing_by_loc = nat_missing["Interview Location"].value_counts().head(15)
    nat_missing_by_city = nat_missing["City"].fillna("Unknown").value_counts().head(10)

    city_missing = df[df["City"].isna()]["Interview Location"].value_counts().head(15)
    paid_missing = df[df["Paid Date"].isna()]["Interview Location"].value_counts().head(15)
    term_missing = df[df["Termination Date"].isna()]["Interview Location"].value_counts().head(15)

    return {
        "missing_summary": summary,
        "missing_nationality_top_locations": {k: int(v) for k, v in nat_missing_by_loc.items()},
        "missing_nationality_top_cities": {k: int(v) for k, v in nat_missing_by_city.items()},
        "nat_missing_termination_rate": round(float(nat_missing["is_terminated"].mean()), 4) if len(nat_missing) else None,
        "nat_present_termination_rate": round(
            float(df[df["Client Nationality"].notna()]["is_terminated"].mean()), 4
        ),
        "missing_city_top_locations": {k: int(v) for k, v in city_missing.items()},
        "missing_paid_top_locations": {k: int(v) for k, v in paid_missing.items()},
        "termination_missing_top_locations": {k: int(v) for k, v in term_missing.items()},
    }


def export_kiosk_nationality_metrics(_: pd.DataFrame) -> None:
    """Build kiosk × nationality outputs from the cleaned per-contract dataset."""

    if not KIOSK_CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean kiosk file missing: {KIOSK_CLEAN_CSV}")

    clean_df = pd.read_csv(
        KIOSK_CLEAN_CSV,
        parse_dates=["Start Month", "End Month"],
    )
    if clean_df.empty:
        return

    clean_df = clean_df.rename(
        columns={
            "Contract ID": "contract_id",
            "Start Month": "start_month",
            "End Month": "end_month",
            "Client Nationality Clean": "client_nationality_clean",
            "Is Active Latest": "is_active_latest",
        }
    )
    clean_df["kiosk"] = (
        clean_df["Kiosk"]
        .fillna(clean_df["Interview Location"])
        .fillna("Unknown Location")
    )
    clean_df["client_nationality"] = (
        clean_df["client_nationality_clean"]
        .fillna(clean_df["Client Nationality"])
        .fillna("Unknown")
    )

    clean_df["start_period"] = clean_df["start_month"].dt.to_period("M")
    clean_df["end_period"] = clean_df["end_month"].dt.to_period("M")
    clean_df["is_active_latest"] = clean_df["is_active_latest"].fillna(False).astype(bool)
    latest_period = clean_df["end_period"].dropna().max()
    clean_df.loc[clean_df["end_period"].isna(), "end_period"] = latest_period

    monthly_records: List[Dict[str, Any]] = []
    for row in clean_df.itertuples(index=False):
        start = row.start_period
        end = row.end_period
        if pd.isna(start) or pd.isna(end):
            continue
        periods = pd.period_range(start, end, freq="M")
        for period in periods:
            monthly_records.append(
                {
                    "Contract ID": row.contract_id,
                    "Kiosk": row.kiosk,
                    "Client Nationality": row.client_nationality,
                    "Month": period.to_timestamp(),
                    "Active": 1,
                    "Is Start Month": int(period == start),
                    "Is End Month": int((period == end) and bool(not row.is_active_latest)),
                }
            )

    monthly_active = pd.DataFrame(monthly_records)
    if monthly_active.empty:
        return
    monthly_active = monthly_active.sort_values(["Kiosk", "Client Nationality", "Month"])
    monthly_active.to_csv(BASE_DIR / "kiosk_nationality_monthly_active.csv", index=False)

    summary = (
        monthly_active.groupby(["Kiosk", "Client Nationality", "Month"])
        .agg(
            active_contracts=("Active", "sum"),
            new_contracts=("Is Start Month", "sum"),
            terminations=("Is End Month", "sum"),
        )
        .reset_index()
        .sort_values(["Kiosk", "Client Nationality", "Month"])
    )

    summary["prev_active"] = (
        summary.groupby(["Kiosk", "Client Nationality"])["active_contracts"]
        .shift(1)
        .fillna(0)
    )
    prev_safe = summary["prev_active"].replace({0: pd.NA})
    summary["churn_rate"] = summary["terminations"] / prev_safe
    summary["churn_rate"] = pd.to_numeric(summary["churn_rate"], errors="coerce")
    summary.loc[~np.isfinite(summary["churn_rate"].fillna(np.nan)), "churn_rate"] = pd.NA
    summary["retention_rate"] = 1 - summary["churn_rate"]
    summary["net_change"] = summary["active_contracts"] - summary["prev_active"]
    summary.to_csv(BASE_DIR / "kiosk_nationality_time_series.csv", index=False)

    latest_month = summary["Month"].max()
    latest_df = summary[summary["Month"] == latest_month].copy()
    kiosk_totals = latest_df.groupby("Kiosk")["active_contracts"].sum()
    latest_df["kiosk_total"] = latest_df["Kiosk"].map(kiosk_totals)
    latest_df["share_of_kiosk"] = latest_df["active_contracts"] / latest_df["kiosk_total"]
    latest_df.to_csv(BASE_DIR / "kiosk_nationality_latest_distribution.csv", index=False)

    window_months = (
        summary["Month"].drop_duplicates().sort_values().tail(3)
    )
    recent_df = summary[summary["Month"].isin(window_months)]
    recent_summary = (
        recent_df.groupby(["Kiosk", "Client Nationality"])
        .agg(
            avg_churn_rate=("churn_rate", "mean"),
            avg_retention_rate=("retention_rate", "mean"),
            mean_active=("active_contracts", "mean"),
            net_change_total=("net_change", "sum"),
            latest_active=("active_contracts", "last"),
        )
        .reset_index()
    )
    recent_summary.to_csv(BASE_DIR / "kiosk_nationality_recent_summary.csv", index=False)

def generate_html_report(
    path: Path,
    load_result: LoadResult,
    data_quality: Dict[str, Any],
    global_metrics: Dict[str, Any],
    temporal: Dict[str, Any],
    retention: Dict[str, Any],
    locations: Dict[str, Any],
    nationality: Dict[str, Any],
    city: Dict[str, Any],
    payment: Dict[str, Any],
    anomalies: Dict[str, Any],
    missing: Dict[str, Any],
) -> None:
    report_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    raw_df = load_result.raw_df

    def fmt_number(value: Any, decimals: int = 0) -> str:
        if value is None:
            return "—"
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return "—"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):,.{decimals}f}"
        return str(value)

    def fmt_ratio(value: Any, decimals: int = 2) -> str:
        if value is None:
            return "—"
        return f"{float(value) * 100:.{decimals}f}%"

    total_contracts = global_metrics.get("total_contracts", len(raw_df))
    status_counts = {str(k): v for k, v in global_metrics.get("status_counts", {}).items()}
    active_count = status_counts.get("Active", 0)
    terminated_count = status_counts.get("Terminated", 0)
    termination_rate = global_metrics.get("termination_rate")
    active_rate = global_metrics.get("active_rate")

    unique_locations = (
        int(raw_df["Interview Location"].nunique(dropna=False)) if "Interview Location" in raw_df.columns else 0
    )
    unique_cities = int(raw_df["City"].nunique(dropna=False)) if "City" in raw_df.columns else 0
    total_nationalities = (
        int(raw_df["Client Nationality"].nunique(dropna=False)) if "Client Nationality" in raw_df.columns else 0
    )
    known_nationalities = (
        int(raw_df["Client Nationality"].dropna().nunique()) if "Client Nationality" in raw_df.columns else 0
    )

    payment_stats = payment.get("time_to_payment_stats", {})
    median_payment_days = payment_stats.get("median_days")
    mean_payment_days = payment_stats.get("mean_days")

    duration_stats = retention.get("duration_stats", {})
    terminated_median_duration = duration_stats.get("terminated_median_days")
    overall_mean_duration = duration_stats.get("overall_mean_days")

    cohort_table = retention.get("cohort_retention_table", {})
    cohort_count = len(cohort_table)
    max_age_months = retention.get("max_age_months")
    survival = retention.get("kaplan_meier", {}).get("survival", [])
    latest_survival = survival[-1] if survival else None

    summary_cards = [
        {
            "title": "Total Contracts",
            "value": fmt_number(total_contracts),
            "meta": f"{fmt_number(unique_locations)} locations • {fmt_number(unique_cities)} cities",
        },
        {
            "title": "Active Contracts",
            "value": fmt_number(active_count),
            "meta": f"{fmt_ratio(active_rate)} of portfolio",
        },
        {
            "title": "Terminated Contracts",
            "value": fmt_number(terminated_count),
            "meta": f"{fmt_ratio(termination_rate)} of portfolio",
        },
        {
            "title": "Known Nationalities",
            "value": fmt_number(known_nationalities),
            "meta": f"{fmt_number(total_nationalities)} incl. missing",
        },
        {
            "title": "Median Payment Time",
            "value": f"{fmt_number(median_payment_days, 2)} days",
            "meta": f"Mean {fmt_number(mean_payment_days, 2)} days",
        },
        {
            "title": "Median Terminated Duration",
            "value": f"{fmt_number(terminated_median_duration, 2)} days",
            "meta": f"Overall mean {fmt_number(overall_mean_duration, 1)} days",
        },
    ]

    # Data quality tables
    missing_summary = missing.get("missing_summary", {})
    if missing_summary:
        missing_df = pd.DataFrame.from_dict(missing_summary, orient="index")
        missing_df = missing_df.reset_index().rename(
            columns={"index": "Field", "missing_count": "Missing Count", "missing_pct": "Missing %"}
        )
        missing_df["Missing Count"] = missing_df["Missing Count"].astype(int)
        missing_df["Missing %"] = missing_df["Missing %"].map(
            lambda val: "—" if pd.isna(val) else f"{float(val):.2f}%"
        )
        missing_table = missing_df.to_html(index=False, classes="data-table", border=0)
    else:
        missing_table = "<p>No missing data detected.</p>"

    duplicates = data_quality.get("duplicates", {})
    duplicates_table = (
        series_to_html_table({k: int(v) for k, v in duplicates.items()}, ("Field", "Duplicate Records"), sort_desc=True)
        if duplicates
        else "<p>No duplicate records detected.</p>"
    )

    datetime_parse = data_quality.get("datetime_parse")
    if datetime_parse is None:
        datetime_parse = {
            key.replace("_parse", ""): value
            for key, value in data_quality.items()
            if key.endswith("_parse") and isinstance(value, dict)
        }
    if datetime_parse:
        parse_df = pd.DataFrame.from_dict(datetime_parse, orient="index").reset_index()
        parse_df = parse_df.rename(columns={"index": "Field", "non_null": "Non-null", "invalid": "Invalid"})
        parse_df["Non-null"] = parse_df["Non-null"].astype(int)
        parse_df["Invalid"] = parse_df["Invalid"].astype(int)
        datetime_table = parse_df.to_html(index=False, classes="data-table", border=0)
    else:
        datetime_table = "<p>No datetime parsing issues detected.</p>"

    # Global metrics charts and tables
    top_locations_dict = {
        name: int(value)
        for name, value in list(global_metrics.get("location_counts", {}).items())[:10]
    }
    location_chart_uri = plot_horizontal_bar(
        top_locations_dict,
        "Top Interview Locations (Contracts)",
        "Contracts",
    )
    location_table_html = series_to_html_table(
        top_locations_dict,
        ("Location", "Contracts"),
        sort_desc=True,
    )

    top_nationalities_dict = {
        name: int(value)
        for name, value in list(nationality.get("top_nationalities_overall", {}).items())[:10]
    }
    nationality_table_html = series_to_html_table(
        top_nationalities_dict,
        ("Nationality", "Contracts"),
        sort_desc=True,
    )

    top_cities_dict = {
        name: int(value) if pd.notna(value) else 0
        for name, value in list(city.get("city_counts", {}).items())[:10]
    }
    city_table_html = series_to_html_table(
        top_cities_dict,
        ("City", "Contracts"),
        sort_desc=True,
    )

    monthly_counts_dict = {
        str(period): int(value)
        for period, value in temporal.get("monthly_counts", {}).items()
    }
    monthly_chart_uri = plot_line_chart(
        monthly_counts_dict,
        "Contracts Created Per Month",
        "Month",
        "Contracts",
    )

    weekday_share = temporal.get("weekday_share", {})
    if weekday_share:
        weekday_df = pd.DataFrame(list(weekday_share.items()), columns=["Weekday", "Share (%)"])
        weekday_df["Share (%)"] = weekday_df["Share (%)"].map(lambda val: round(float(val), 2))
        weekday_table_html = weekday_df.to_html(index=False, classes="data-table", border=0)
    else:
        weekday_table_html = "<p>No weekday distribution available.</p>"

    weekend_split = temporal.get("weekend_split", {})
    weekend_table_html = series_to_html_table(
        {("Weekend" if str(k) == "True" else "Weekday"): v for k, v in weekend_split.items()},
        ("Bucket", "Contracts"),
        sort_desc=True,
    ) if weekend_split else "<p>No weekend split available.</p>"

    payment_stats_df = pd.DataFrame(payment_stats, index=[0]).T.reset_index()
    if not payment_stats_df.empty:
        payment_stats_df = payment_stats_df.rename(columns={"index": "Metric", 0: "Days"})
        payment_stats_df["Days"] = payment_stats_df["Days"].map(
            lambda val: "—" if pd.isna(val) else round(float(val), 2)
        )
        payment_stats_table = payment_stats_df.to_html(index=False, classes="data-table", border=0)
    else:
        payment_stats_table = "<p>No payment statistics available.</p>"

    def round_record_field(records: List[Dict[str, Any]], field: str, decimals: int = 2) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for record in records:
            new_record = dict(record)
            if field in new_record and new_record[field] is not None and not pd.isna(new_record[field]):
                value = float(new_record[field])
                if decimals == 0:
                    new_record[field] = int(round(value))
                else:
                    new_record[field] = round(value, decimals)
            processed.append(new_record)
        return processed

    slowest_payments = round_record_field(payment.get("slowest_payments", []), "time_to_payment_days", 2)
    fastest_payments = round_record_field(payment.get("fastest_payments", []), "time_to_payment_days", 2)

    slowest_payments_table = records_to_html_table(
        slowest_payments,
        columns=[
            "Contract ID",
            "Interview Location",
            "Client Nationality",
            "time_to_payment_days",
            "Contract Creation Date",
            "Paid Date",
        ],
        rename={"time_to_payment_days": "Time to Payment (days)", "Interview Location": "Location"},
        limit=5,
    )
    fastest_payments_table = records_to_html_table(
        fastest_payments,
        columns=[
            "Contract ID",
            "Interview Location",
            "Client Nationality",
            "time_to_payment_days",
            "Contract Creation Date",
            "Paid Date",
        ],
        rename={"time_to_payment_days": "Time to Payment (days)", "Interview Location": "Location"},
        limit=5,
    )

    cohort_sizes_items = sorted(retention.get("cohort_sizes", {}).items(), key=lambda item: item[0])
    cohort_sizes_df = pd.DataFrame(cohort_sizes_items, columns=["Cohort Month", "Contracts"]).head(12)
    cohort_sizes_table = (
        cohort_sizes_df.to_html(index=False, classes="data-table", border=0)
        if not cohort_sizes_df.empty
        else "<p>No cohort data available.</p>"
    )

    benchmark_metrics = locations.get("benchmark_metrics", {})
    if benchmark_metrics:
        benchmark_df = pd.DataFrame.from_dict(benchmark_metrics, orient="index").reset_index()
        benchmark_df = benchmark_df.rename(
            columns={
                "index": "Location",
                "total_contracts": "Contracts",
                "termination_rate": "Termination Rate",
                "median_duration_days": "Median Duration (days)",
                "median_time_to_payment_days": "Median Payment (days)",
            }
        )
        benchmark_df = benchmark_df.sort_values("Contracts", ascending=False).head(10)
        benchmark_df["Termination Rate"] = benchmark_df["Termination Rate"].map(
            lambda val: fmt_ratio(val, 2) if val is not None else "—"
        )
        benchmark_df["Median Duration (days)"] = benchmark_df["Median Duration (days)"].map(
            lambda val: "—" if pd.isna(val) else round(float(val), 3)
        )
        benchmark_df["Median Payment (days)"] = benchmark_df["Median Payment (days)"].map(
            lambda val: "—" if pd.isna(val) else round(float(val), 3)
        )
        benchmark_table = benchmark_df.to_html(index=False, classes="data-table", border=0)
    else:
        benchmark_table = "<p>No benchmark metrics available.</p>"

    termination_duration_outliers = round_record_field(
        anomalies.get("termination_duration_outliers", []),
        "duration_days_calc",
        0,
    )
    termination_outliers_table = records_to_html_table(
        termination_duration_outliers,
        columns=["Contract ID", "Interview Location", "Client Nationality", "duration_days_calc"],
        rename={"Interview Location": "Location", "duration_days_calc": "Duration (days)"},
        limit=10,
    )

    active_duration_outliers = round_record_field(
        anomalies.get("active_duration_outliers", []),
        "duration_days_calc",
        0,
    )
    active_outliers_table = records_to_html_table(
        active_duration_outliers,
        columns=["Contract ID", "Interview Location", "Client Nationality", "duration_days_calc"],
        rename={"Interview Location": "Location", "duration_days_calc": "Duration (days)"},
        limit=10,
    )

    location_zscores = anomalies.get("location_termination_zscore", {})
    if location_zscores:
        loc_z_df = pd.DataFrame.from_dict(location_zscores, orient="index").reset_index()
        loc_z_df = loc_z_df.rename(columns={"index": "Location", "termination_rate": "Termination Rate", "count": "Contracts", "zscore": "Z-score"})
        loc_z_df["Termination Rate"] = loc_z_df["Termination Rate"].map(
            lambda val: fmt_ratio(val, 2) if val is not None else "—"
        )
        loc_z_df["Z-score"] = loc_z_df["Z-score"].map(lambda val: "—" if pd.isna(val) else round(float(val), 3))
        loc_z_df = loc_z_df.sort_values("Z-score").head(10)
        loc_z_table = loc_z_df.to_html(index=False, classes="data-table", border=0)
    else:
        loc_z_table = "<p>No location-level anomalies detected.</p>"

    missing_nat_locations = series_to_html_table(
        missing.get("missing_nationality_top_locations", {}),
        ("Location", "Missing Nationalities"),
        sort_desc=True,
    ) if missing.get("missing_nationality_top_locations") else "<p>No missing nationality hotspots.</p>"

    missing_paid_locations = series_to_html_table(
        missing.get("missing_paid_top_locations", {}),
        ("Location", "Missing Paid Date"),
        sort_desc=True,
    ) if missing.get("missing_paid_top_locations") else "<p>No paid date gaps detected.</p>"

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8' />",
        f"<title>Retail Contracts Analysis Report — {report_date}</title>",
        "<style>",
        "body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; background: #f5f7fb; color: #1f2937; }",
        ".header { background: #111827; color: #f9fafb; padding: 32px 48px; }",
        ".header h1 { margin: 0 0 8px; font-size: 28px; }",
        ".header p { margin: 0; color: #d1d5db; }",
        ".section { padding: 32px 48px; }",
        ".section h2 { margin-top: 0; color: #111827; }",
        ".cards { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 20px; }",
        ".card { background: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08); }",
        ".card h3 { margin: 0 0 12px; font-size: 16px; color: #4b5563; text-transform: uppercase; letter-spacing: 0.04em; }",
        ".card-value { margin: 0; font-size: 28px; font-weight: 600; color: #111827; }",
        ".card-meta { margin: 6px 0 0; font-size: 13px; color: #6b7280; }",
        ".panels { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }",
        ".panel { background: #ffffff; border-radius: 10px; padding: 20px 24px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06); }",
        ".panel h3 { margin-top: 0; color: #1f2937; }",
        ".panel p { color: #4b5563; }",
        ".chart { margin: 12px 0 8px; text-align: center; }",
        ".chart img { max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08); }",
        ".data-table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 14px; }",
        ".data-table th, .data-table td { border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }",
        ".data-table th { background: #f3f4f6; color: #111827; font-weight: 600; }",
        ".data-table tr:nth-child(even) { background: #f9fafb; }",
        ".note { font-size: 13px; color: #6b7280; margin-top: 12px; }",
        ".grid-two { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }",
        "footer { padding: 24px 48px 48px; color: #6b7280; font-size: 14px; }",
        "</style>",
        "</head>",
        "<body>",
        "<header class='header'>",
        "<h1>Retail Contracts Intelligence Report</h1>",
        f"<p>Generated on {report_date}</p>",
        "</header>",
        "<section class='section'>",
        "<h2>Executive Overview</h2>",
        "<div class='cards'>",
    ]

    for card in summary_cards:
        html_parts.extend(
            [
                "<div class='card'>",
                f"<h3>{card['title']}</h3>",
                f"<p class='card-value'>{card['value']}</p>",
                f"<p class='card-meta'>{card['meta']}</p>" if card.get("meta") else "",
                "</div>",
            ]
        )

    html_parts.extend(
        [
            "</div>",
            f"<p class='note'>Cohort span: {fmt_number(cohort_count)} monthly cohorts • Retention tracked for up to {fmt_number(max_age_months)} months • Latest survival probability {fmt_number(latest_survival, 3)}</p>",
            "</section>",
            "<section class='section'>",
            "<h2>Data Quality</h2>",
            "<div class='panels'>",
            "<div class='panel'>",
            "<h3>Missing Value Summary</h3>",
            missing_table,
            "</div>",
            "<div class='panel'>",
            "<h3>Duplicate Checks</h3>",
            duplicates_table,
            "</div>",
            "<div class='panel'>",
            "<h3>Date Parsing Validation</h3>",
            datetime_table,
            "</div>",
            "</div>",
            "</section>",
            "<section class='section'>",
            "<h2>Global Metrics</h2>",
            "<div class='grid-two'>",
            "<div class='panel'>",
            "<h3>Top Interview Locations</h3>",
        ]
    )

    if location_chart_uri:
        html_parts.append(f"<div class='chart'><img src='{location_chart_uri}' alt='Top interview locations chart' /></div>")
    html_parts.append(location_table_html)
    html_parts.extend(
        [
            "</div>",
            "<div class='panel'>",
            "<h3>Top Nationalities</h3>",
            nationality_table_html,
            "</div>",
            "</div>",
            "<div class='grid-two' style='margin-top:20px;'>",
            "<div class='panel'>",
            "<h3>Top Cities</h3>",
            city_table_html,
            "</div>",
            "<div class='panel'>",
            "<h3>Location Benchmark (Top 10)</h3>",
            benchmark_table,
            "</div>",
            "</div>",
            "</section>",
            "<section class='section'>",
            "<h2>Temporal Trends</h2>",
            "<div class='grid-two'>",
            "<div class='panel'>",
            "<h3>Monthly Contract Velocity</h3>",
        ]
    )

    if monthly_chart_uri:
        html_parts.append(f"<div class='chart'><img src='{monthly_chart_uri}' alt='Monthly contract trend chart' /></div>")
    else:
        html_parts.append("<p>No monthly trend data available.</p>")
    html_parts.extend(
        [
            "</div>",
            "<div class='panel'>",
            "<h3>Weekday Distribution</h3>",
            weekday_table_html,
            "<h3>Weekend vs Weekday</h3>",
            weekend_table_html,
            "</div>",
            "</div>",
            "</section>",
            "<section class='section'>",
            "<h2>Payment Behaviour</h2>",
            "<div class='grid-two'>",
            "<div class='panel'>",
            "<h3>Payment Lag Statistics</h3>",
            payment_stats_table,
            "</div>",
            "<div class='panel'>",
            "<h3>Slowest Payments (Top 5)</h3>",
            slowest_payments_table,
            "<h3>Fastest Payments (Top 5)</h3>",
            fastest_payments_table,
            "</div>",
            "</div>",
            "</section>",
            "<section class='section'>",
            "<h2>Retention & Cohorts</h2>",
            "<div class='grid-two'>",
            "<div class='panel'>",
            "<h3>Cohort Sizes (Recent 12)</h3>",
            cohort_sizes_table,
            "</div>",
            "<div class='panel'>",
            "<h3>Key Retention Metrics</h3>",
            f"<p>Max tracked age: {fmt_number(max_age_months)} months</p>",
            f"<p>Latest survival probability: {fmt_number(latest_survival, 3)}</p>",
            f"<p>Median terminated duration: {fmt_number(terminated_median_duration, 2)} days</p>",
            f"<p>Overall mean duration: {fmt_number(overall_mean_duration, 1)} days</p>",
            "</div>",
            "</div>",
            "</section>",
            "<section class='section'>",
            "<h2>Anomaly Detection</h2>",
            "<div class='grid-two'>",
            "<div class='panel'>",
            "<h3>Longest Terminated Contracts</h3>",
            termination_outliers_table,
            "</div>",
            "<div class='panel'>",
            "<h3>Longest Active Contracts</h3>",
            active_outliers_table,
            "</div>",
            "</div>",
            "<div class='panel' style='margin-top:20px;'>",
            "<h3>Location Termination Z-scores</h3>",
            loc_z_table,
            "</div>",
            "</section>",
            "<section class='section'>",
            "<h2>Missing Data Hotspots</h2>",
            "<div class='grid-two'>",
            "<div class='panel'>",
            "<h3>Nationality Missing by Location</h3>",
            missing_nat_locations,
            "</div>",
            "<div class='panel'>",
            "<h3>Paid Date Missing by Location</h3>",
            missing_paid_locations,
            "</div>",
            "</div>",
            "</section>",
            "<footer>",
            "<p>Report generated by the automated analytics pipeline. Visuals rendered with Matplotlib.</p>",
            "</footer>",
            "</body>",
            "</html>",
        ]
    )

    path.write_text("\n".join(html_parts), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    load_result = load_and_audit()
    raw_df = load_result.raw_df

    data_quality = run_data_quality(raw_df)
    write_json(BASE_DIR / "data_quality_summary.json", data_quality)

    enriched_df = engineer_features(raw_df)

    global_metrics = compute_global_metrics(enriched_df)
    write_json(BASE_DIR / "analysis_global_metrics.json", global_metrics)

    temporal_metrics = analyze_temporal(enriched_df)
    write_json(BASE_DIR / "analysis_temporal.json", temporal_metrics)

    retention_metrics = cohort_and_retention(enriched_df)
    write_json(BASE_DIR / "analysis_retention.json", retention_metrics)

    location_metrics = location_deep_dive(enriched_df)
    write_json(BASE_DIR / "analysis_locations.json", location_metrics)

    nationality_metrics = nationality_segmentation(enriched_df)
    write_json(BASE_DIR / "analysis_nationality.json", nationality_metrics)

    city_metrics = city_insights(enriched_df)
    write_json(BASE_DIR / "analysis_city.json", city_metrics)

    payment_metrics = payment_behaviour(enriched_df)
    write_json(BASE_DIR / "analysis_payment.json", payment_metrics)

    anomaly_metrics = anomaly_detection(enriched_df)
    write_json(BASE_DIR / "analysis_anomalies.json", anomaly_metrics)

    missing_metrics = missing_data_analysis(enriched_df)
    write_json(BASE_DIR / "analysis_missing.json", missing_metrics)

    export_kiosk_nationality_metrics(enriched_df)

    report_path = BASE_DIR / "analysis_report.html"
    generate_html_report(
        report_path,
        load_result,
        data_quality,
        global_metrics,
        temporal_metrics,
        retention_metrics,
        location_metrics,
        nationality_metrics,
        city_metrics,
        payment_metrics,
        anomaly_metrics,
        missing_metrics,
    )

    # Optional console summary
    print("Analysis complete.")
    print(f"- Enriched dataset: {ENRICHED_CSV.name}")
    print("- JSON outputs:")
    for name in [
        "data_quality_summary.json",
        "analysis_global_metrics.json",
        "analysis_temporal.json",
        "analysis_retention.json",
        "analysis_locations.json",
        "analysis_nationality.json",
        "analysis_city.json",
        "analysis_payment.json",
        "analysis_anomalies.json",
        "analysis_missing.json",
    ]:
        print(f"  • {name}")
    print(f"- HTML report: {report_path.name}")


if __name__ == "__main__":
    pd.set_option("mode.copy_on_write", True)
    main()

