#!/usr/bin/env python3
"""
Generate a standalone interactive HTML report for retail sales & churn analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Retail contract IDs - Sheet1.csv"
OUTPUT_PATH = BASE_DIR / "retail_nationality_kiosk_report.html"


def safe_json(data) -> str:
    """Dump JSON safely for embedding into <script> tags."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def round_float(value, digits: int = 2):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return round(float(value), digits)


def mean_without_na(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    return float(cleaned.mean())


def median_without_na(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    return float(cleaned.median())


def compute_cancellation_insights(df_slice: pd.DataFrame) -> Dict:
    if df_slice.empty:
        return {"rateSummary": [], "timeToCancel": []}

    df_local = df_slice.copy()
    df_local["base_date"] = df_local["Paid Date"].where(df_local["Paid Date"].notna(), df_local["Contract Creation Date"])
    df_local["base_date"] = pd.to_datetime(df_local["base_date"], errors="coerce")
    term_mask = df_local["contract_status"] == "Terminated"

    def cohort_stats(label: str, mask: pd.Series) -> Dict:
        subset = df_local[mask]
        total = int(len(subset))
        terminations = int((subset["contract_status"] == "Terminated").sum())
        rate = terminations / total if total else 0.0
        return {
            "cohort": label,
            "contracts": total,
            "terminations": terminations,
            "cancellation_rate": round_float(rate, 3),
        }

    df_local["nat_lower"] = df_local["Client Nationality"].fillna("Unknown").str.lower()
    rate_summary = [
        cohort_stats("Retail (all)", df_local.index == df_local.index),
        cohort_stats("Other nationalities", df_local["nat_lower"] != "filipina"),
        cohort_stats("Filipina", df_local["nat_lower"] == "filipina"),
    ]

    base_df = df_local[df_local["base_date"].notna()].copy()
    term_df = base_df[
        term_mask
        & base_df["Termination Date"].notna()
    ].copy()
    term_df["tto_days"] = (term_df["Termination Date"] - term_df["base_date"]).dt.days
    term_df = term_df[term_df["tto_days"] >= 0]

    pre_start = pd.Timestamp("2024-01-01")
    pre_end = pd.Timestamp("2024-12-01")

    term_df["nat_lower"] = term_df["Client Nationality"].fillna("Unknown").str.lower()

    def segment_stats(
        period_label: str,
        contract_label: str,
        cohort_label: str,
        base_mask: pd.Series,
        term_mask_local: pd.Series,
    ) -> Dict:
        contracts_total = int(base_df[base_mask].shape[0])
        subset = term_df[term_mask_local]
        terminated = int(len(subset))
        cancel_7d = int((subset["tto_days"] <= 7).sum())
        cancel_30d = int((subset["tto_days"] <= 30).sum())
        rate_7d = cancel_7d / contracts_total if contracts_total else 0.0
        rate_30d = cancel_30d / contracts_total if contracts_total else 0.0
        return {
            "period": period_label,
            "contract_type": contract_label,
            "cohort": cohort_label,
            "terminated": terminated,
            "cancel_7d": cancel_7d,
            "cancel_30d": cancel_30d,
            "pct_7d": round_float(rate_7d, 3),
            "pct_30d": round_float(rate_30d, 3),
        }

    period_masks = [
        ("Pre-promoters", lambda df: (df["base_date"] >= pre_start) & (df["base_date"] < pre_end)),
        ("Post-promoters", lambda df: df["base_date"] > pre_end),
    ]
    contract_masks = [
        ("CC", lambda df: df["Contract Type"].str.upper() == "CC"),
        ("MV", lambda df: df["Contract Type"].str.upper() == "MV"),
        ("All", lambda df: df.index == df.index),
    ]
    cohort_masks = [
        ("Retail (all)", lambda df: df.index == df.index),
        ("Filipina", lambda df: df["nat_lower"] == "filipina"),
        ("Other nationalities", lambda df: df["nat_lower"] != "filipina"),
    ]

    time_to_cancel: List[Dict] = []
    for period_label, p_mask_fn in period_masks:
        for contract_label, c_mask_fn in contract_masks:
            for cohort_label, n_mask_fn in cohort_masks:
                base_mask = p_mask_fn(base_df) & c_mask_fn(base_df) & n_mask_fn(base_df)
                term_mask_local = p_mask_fn(term_df) & c_mask_fn(term_df) & n_mask_fn(term_df)
                time_to_cancel.append(
                    segment_stats(period_label, contract_label, cohort_label, base_mask, term_mask_local)
                )

    return {
        "rateSummary": rate_summary,
        "timeToCancel": time_to_cancel,
    }


def _legacy_prepare_data() -> Dict:
    df = pd.read_csv(DATA_PATH)

    # Track raw missing before cleaning
    nat_missing_mask = df["Client Nationality"].isna() | df["Client Nationality"].astype(str).str.strip().eq("")
    nat_missing_count = int(nat_missing_mask.sum())

    city_missing_mask = df["City"].isna() | df["City"].astype(str).str.strip().eq("")
    city_missing_count = int(city_missing_mask.sum())

    paid_missing_count = int(df["Paid Date"].isna().sum())
    client_name_missing_count = int(df["Client Name"].isna().sum())
    termination_missing_count = int(df["Termination Date"].isna().sum())

    # Convert identifiers to string to avoid scientific notation or truncation
    for col in ["Contract ID", "Interview ID", "Contract Client ID"]:
        df[col] = df[col].astype(str)

    # Parse dates
    for col in ["Contract Creation Date", "Paid Date", "Termination Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Clean strings
    df["Interview Location"] = df["Interview Location"].fillna("Unknown Location").astype(str).str.strip()

    def clean_nationality(value) -> str:
        if pd.isna(value):
            return "Unknown nationality"
        stripped = str(value).strip()
        return stripped if stripped else "Unknown nationality"

    df["Client Nationality"] = df["Client Nationality"].apply(clean_nationality)

    invalid_city_counts: Dict[str, int] = {}

    def clean_city(value) -> str:
        if pd.isna(value):
            return "Unknown"
        stripped = str(value).strip()
        if not stripped:
            return "Unknown"
        if stripped.isdigit():
            invalid_city_counts[stripped] = invalid_city_counts.get(stripped, 0) + 1
            return "Unknown"
        return stripped

    df["City_clean"] = df["City"].apply(clean_city)

    # Derived columns
    terminated_mask = df["Termination Date"].notna()
    df["contract_status"] = np.where(terminated_mask, "Terminated", "Active")

    df["contract_duration_days"] = np.where(
        terminated_mask,
        (df["Termination Date"] - df["Contract Creation Date"]).dt.days,
        np.nan,
    )

    negative_duration_mask = df["contract_duration_days"] < 0
    negative_duration_count = int(negative_duration_mask.sum())
    if negative_duration_count:
        df.loc[negative_duration_mask, "contract_duration_days"] = np.nan

    df["payment_lag_days"] = (df["Paid Date"] - df["Contract Creation Date"]).dt.days
    payment_lag_negative_count = int(df["payment_lag_days"].dropna().lt(0).sum())

    df["contract_year"] = df["Contract Creation Date"].dt.year
    df["contract_month"] = df["Contract Creation Date"].dt.to_period("M").astype(str)
    df["termination_month"] = df["Termination Date"].dt.to_period("M").astype(str)

    total_contracts = int(len(df))
    unique_clients = int(df["Contract Client ID"].nunique())
    terminated_contracts = int((df["contract_status"] == "Terminated").sum())
    active_contracts = total_contracts - terminated_contracts
    termination_rate = terminated_contracts / total_contracts if total_contracts else 0.0

    contract_date_min = df["Contract Creation Date"].min()
    contract_date_max = df["Contract Creation Date"].max()
    contract_month_period_series = df["Contract Creation Date"].dt.to_period("M")
    max_contract_period = contract_month_period_series.max()
    min_contract_period = contract_month_period_series.min()
    if (
        pd.isna(max_contract_period)
        or pd.isna(contract_date_max)
    ):
        last_full_month_period = None
    elif contract_date_max.is_month_end:
        last_full_month_period = max_contract_period
    else:
        last_full_month_period = max_contract_period - 1

    overall_kpis = {
        "total_contracts": total_contracts,
        "unique_clients": unique_clients,
        "active_contracts": active_contracts,
        "terminated_contracts": terminated_contracts,
        "termination_rate": round_float(termination_rate, 3),
        "contract_date_min": contract_date_min.strftime("%Y-%m-%d") if pd.notna(contract_date_min) else None,
        "contract_date_max": contract_date_max.strftime("%Y-%m-%d") if pd.notna(contract_date_max) else None,
    }

    # Nationality aggregation
    nationality_group = (
        df.groupby("Client Nationality")
        .agg(
            contracts_count=("Contract ID", "count"),
            terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            active_count=("contract_status", lambda s: int((s == "Active").sum())),
        )
        .reset_index()
    )
    nationality_group["termination_rate"] = nationality_group["terminated_count"] / nationality_group["contracts_count"]
    nationality_group["contracts_share"] = nationality_group["contracts_count"] / total_contracts
    nationality_stats: List[Dict] = []
    for row in nationality_group.sort_values("contracts_count", ascending=False).to_dict("records"):
        nationality_stats.append(
            {
                "nationality": row["Client Nationality"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
                "contracts_share": round_float(row["contracts_share"]),
            }
        )

    # Highlight segments
    nationality_threshold = 30
    eligible_nationalities = [n for n in nationality_stats if n["contracts_count"] >= nationality_threshold]
    low_churn_nationalities = sorted(eligible_nationalities, key=lambda x: (x["termination_rate"], -x["contracts_count"]))[:5]
    high_churn_nationalities = sorted(
        eligible_nationalities,
        key=lambda x: (-(x["termination_rate"] if x["termination_rate"] is not None else -1), -x["contracts_count"]),
    )[:5]

    # Kiosk aggregation
    kiosk_group = (
        df.groupby("Interview Location")
        .agg(
            contracts_count=("Contract ID", "count"),
            terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            active_count=("contract_status", lambda s: int((s == "Active").sum())),
        )
        .reset_index()
    )
    kiosk_group["termination_rate"] = kiosk_group["terminated_count"] / kiosk_group["contracts_count"]
    kiosk_stats: List[Dict] = []
    for row in kiosk_group.sort_values("contracts_count", ascending=False).to_dict("records"):
        kiosk_stats.append(
            {
                "kiosk": row["Interview Location"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
            }
        )

    kiosk_threshold = 50
    eligible_kiosks = [k for k in kiosk_stats if k["contracts_count"] >= kiosk_threshold]
    best_kiosks = sorted(eligible_kiosks, key=lambda x: (x["termination_rate"], -x["contracts_count"]))[:5]
    worst_kiosks = sorted(
        eligible_kiosks,
        key=lambda x: (-(x["termination_rate"] if x["termination_rate"] is not None else -1), -x["contracts_count"]),
    )[:5]

    # Nationality x Kiosk aggregation
    nk_group = (
        df.groupby(["Interview Location", "Client Nationality"])
        .agg(
            contracts_count=("Contract ID", "count"),
            terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            active_count=("contract_status", lambda s: int((s == "Active").sum())),
        )
        .reset_index()
    )
    nk_group["termination_rate"] = nk_group["terminated_count"] / nk_group["contracts_count"]

    combination_stats: List[Dict] = []
    for row in nk_group.to_dict("records"):
        combination_stats.append(
            {
                "kiosk": row["Interview Location"],
                "nationality": row["Client Nationality"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
            }
        )

    risk_threshold_contracts = 10
    risk_threshold_rate = 0.6
    risk_segments = [
        combo
        for combo in combination_stats
        if combo["contracts_count"] >= risk_threshold_contracts
        and (combo["termination_rate"] or 0) >= risk_threshold_rate
    ]
    risk_segments = sorted(
        risk_segments,
        key=lambda x: (-(x["termination_rate"] if x["termination_rate"] is not None else -1), -x["contracts_count"]),
    )

    # Heatmap data (limit to top 12 each)
    top_nationalities = [n["nationality"] for n in nationality_stats[:12]]
    top_kiosks = [k["kiosk"] for k in kiosk_stats[:12]]
    heatmap_nationalities = list(top_nationalities)
    heatmap_kiosks = list(top_kiosks)

    data_map = {
        (row["kiosk"], row["nationality"]): row for row in combination_stats if row["contracts_count"] > 0
    }
    termination_rate_matrix: List[List[float | None]] = []
    contracts_matrix: List[List[int | None]] = []
    tooltip_matrix: List[List[str]] = []

    for kiosk in top_kiosks:
        rate_row: List[float | None] = []
        count_row: List[int | None] = []
        tooltip_row: List[str] = []
        for nationality in top_nationalities:
            combo = data_map.get((kiosk, nationality))
            if combo:
                rate_row.append(combo["termination_rate"])
                count_row.append(combo["contracts_count"])
                tooltip_row.append(
                    f"{kiosk} / {nationality}<br>Contracts: {combo['contracts_count']}<br>"
                    f"Terminated: {combo['terminated_count']}<br>"
                    f"Termination rate: {round_float(combo['termination_rate'], 3) if combo['termination_rate'] is not None else 'N/A'}"
                )
            else:
                rate_row.append(None)
                count_row.append(None)
                tooltip_row.append(f"{kiosk} / {nationality}<br>No contracts")
        termination_rate_matrix.append(rate_row)
        contracts_matrix.append(count_row)
        tooltip_matrix.append(tooltip_row)

    # Drill-down mappings
    nationality_drilldown: Dict[str, List[Dict]] = {}
    kiosk_drilldown: Dict[str, List[Dict]] = {}

    for nationality, group in nk_group.groupby("Client Nationality"):
        sorted_rows = group.sort_values("contracts_count", ascending=False).to_dict("records")
        nationality_drilldown[nationality] = [
            {
                "kiosk": row["Interview Location"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
            }
            for row in sorted_rows
        ]

    for kiosk, group in nk_group.groupby("Interview Location"):
        sorted_rows = group.sort_values("contracts_count", ascending=False).to_dict("records")
        kiosk_drilldown[kiosk] = [
            {
                "nationality": row["Client Nationality"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
            }
            for row in sorted_rows
        ]

    # Client-level aggregations
    client_group = (
        df.groupby("Contract Client ID")
        .agg(
            client_contracts_count=("Contract ID", "count"),
            client_terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            client_active_count=("contract_status", lambda s: int((s == "Active").sum())),
            first_contract_date=("Contract Creation Date", "min"),
            last_contract_date=("Contract Creation Date", "max"),
        )
        .reset_index()
    )

    client_group["client_term_rate"] = (
        client_group["client_terminated_count"] / client_group["client_contracts_count"]
    ).replace([np.inf, -np.inf], np.nan)

    clients_with_one = client_group[client_group["client_contracts_count"] == 1]
    clients_with_multi = client_group[client_group["client_contracts_count"] > 1]

    clients_with_one_contract = int(len(clients_with_one))
    clients_with_multiple_contracts = int(len(clients_with_multi))

    repeat_rate = clients_with_multiple_contracts / unique_clients if unique_clients else 0
    single_client_termination_rate = float(
        (clients_with_one["client_terminated_count"] > 0).mean()
    ) if clients_with_one_contract else 0.0
    multi_client_termination_rate = float(
        (clients_with_multi["client_terminated_count"] > 0).mean()
    ) if clients_with_multiple_contracts else 0.0

    def bucket_contracts(count: int) -> str:
        return str(count) if count < 5 else "5+"

    client_group["contracts_bucket"] = client_group["client_contracts_count"].apply(bucket_contracts)

    distribution = (
        client_group.groupby("contracts_bucket").size().reindex(["1", "2", "3", "4", "5+"], fill_value=0)
    )
    distribution_records = [
        {"contracts": bucket, "clients": int(distribution.loc[bucket])} for bucket in ["1", "2", "3", "4", "5+"]
    ]

    bucket_summary = []
    for bucket, bucket_df in client_group.groupby("contracts_bucket"):
        avg_term_rate = bucket_df["client_term_rate"].mean()
        bucket_summary.append(
            {
                "bucket": bucket,
                "clients": int(len(bucket_df)),
                "avg_client_termination_rate": round_float(avg_term_rate),
            }
        )
    bucket_summary = sorted(bucket_summary, key=lambda x: ["1", "2", "3", "4", "5+"].index(x["bucket"]))

    client_metrics = {
        "total_unique_clients": unique_clients,
        "clients_with_one_contract": clients_with_one_contract,
        "clients_with_multiple_contracts": clients_with_multiple_contracts,
        "repeat_rate": round_float(repeat_rate),
        "single_client_termination_rate": round_float(single_client_termination_rate),
        "multi_client_termination_rate": round_float(multi_client_termination_rate),
        "distribution": distribution_records,
        "bucket_summary": bucket_summary,
    }

    # Time series
    contracts_by_month = df.groupby("contract_month").size()
    terminations_by_month = df[df["contract_status"] == "Terminated"].groupby("termination_month").size()
    all_months = sorted(set(contracts_by_month.index) | set(terminations_by_month.index))
    time_series = [
        {
            "month": month,
            "contracts": int(contracts_by_month.get(month, 0)),
            "terminations": int(terminations_by_month.get(month, 0)),
        }
        for month in all_months
        if isinstance(month, str) and month != "NaT"
    ]

    nationality_monthly_contracts = (
        df.groupby(["Client Nationality", "contract_month"]).size()
    )
    nationality_monthly_terminations = (
        df[df["contract_status"] == "Terminated"].groupby(["Client Nationality", "termination_month"]).size()
    )
    unique_nationalities = sorted(df["Client Nationality"].unique())
    time_series_by_nationality: Dict[str, List[Dict[str, int]]] = {}
    for nationality in unique_nationalities:
        series_entries = [
            {
                "month": month,
                "contracts": int(nationality_monthly_contracts.get((nationality, month), 0)),
                "terminations": int(nationality_monthly_terminations.get((nationality, month), 0)),
            }
            for month in all_months
            if isinstance(month, str) and month != "NaT"
        ]
        time_series_by_nationality[nationality] = series_entries
    time_series_by_nationality["All nationalities"] = time_series

    if last_full_month_period is None or pd.isna(min_contract_period) or min_contract_period > last_full_month_period:
        contract_months_sorted: List[str] = []
    else:
        contract_months_sorted = [
            str(period) for period in pd.period_range(min_contract_period, last_full_month_period, freq="M")
        ]
    kiosk_monthly_counts = (
        df.groupby(["Interview Location", "Client Nationality", "contract_month"])
        .size()
        .reset_index(name="contracts_count")
    )
    monthly_lookup = {
        (row["Interview Location"], row["Client Nationality"], row["contract_month"]): int(row["contracts_count"])
        for row in kiosk_monthly_counts.to_dict("records")
        if isinstance(row["contract_month"], str) and row["contract_month"] != "NaT"
    }
    status_monthly_counts = (
        df.groupby(["Interview Location", "Client Nationality", "contract_month", "contract_status"])
        .size()
        .reset_index(name="contracts_count")
    )
    status_lookup = {
        (
            row["Interview Location"],
            row["Client Nationality"],
            row["contract_month"],
            row["contract_status"],
        ): int(row["contracts_count"])
        for row in status_monthly_counts.to_dict("records")
        if isinstance(row["contract_month"], str) and row["contract_month"] != "NaT"
    }

    kiosk_top_timeseries: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    for kiosk, entries in kiosk_drilldown.items():
        timeseries_nationalities = [entry["nationality"] for entry in entries[:5]]
        if not timeseries_nationalities:
            continue
        kiosk_top_timeseries[kiosk] = {
            "nationalities": timeseries_nationalities,
            "all": {},
            "Active": {},
            "Terminated": {},
        }
        for nationality in timeseries_nationalities:
            all_counts: List[int] = []
            active_counts: List[int] = []
            terminated_counts: List[int] = []
            for month in contract_months_sorted:
                all_counts.append(monthly_lookup.get((kiosk, nationality, month), 0))
                active_counts.append(status_lookup.get((kiosk, nationality, month, "Active"), 0))
                terminated_counts.append(status_lookup.get((kiosk, nationality, month, "Terminated"), 0))
            kiosk_top_timeseries[kiosk]["all"][nationality] = all_counts
            kiosk_top_timeseries[kiosk]["Active"][nationality] = active_counts
            kiosk_top_timeseries[kiosk]["Terminated"][nationality] = terminated_counts

    # City stats
    city_group = (
        df.groupby("City_clean")
        .agg(
            contracts_count=("Contract ID", "count"),
            terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            active_count=("contract_status", lambda s: int((s == "Active").sum())),
        )
        .reset_index()
    )
    city_group["termination_rate"] = city_group["terminated_count"] / city_group["contracts_count"]
    city_stats = [
        {
            "city": row["City_clean"],
            "contracts_count": int(row["contracts_count"]),
            "terminated_count": int(row["terminated_count"]),
            "active_count": int(row["active_count"]),
            "termination_rate": round_float(row["termination_rate"]),
        }
        for row in city_group.sort_values("contracts_count", ascending=False).to_dict("records")
    ]

    # Payment lag summary
    valid_payment_lag = df["payment_lag_days"].dropna()
    payment_lag_summary = {
        "count": int(valid_payment_lag.count()),
        "min": round_float(valid_payment_lag.min(), 2) if not valid_payment_lag.empty else None,
        "q1": round_float(valid_payment_lag.quantile(0.25), 2) if not valid_payment_lag.empty else None,
        "median": round_float(valid_payment_lag.median(), 2) if not valid_payment_lag.empty else None,
        "q3": round_float(valid_payment_lag.quantile(0.75), 2) if not valid_payment_lag.empty else None,
        "max": round_float(valid_payment_lag.max(), 2) if not valid_payment_lag.empty else None,
    }

    missing_summary = [
        {
            "field": "Paid Date",
            "missing_count": paid_missing_count,
            "missing_pct": round_float(paid_missing_count / total_contracts if total_contracts else 0, 3),
        },
        {
            "field": "Client Name",
            "missing_count": client_name_missing_count,
            "missing_pct": round_float(client_name_missing_count / total_contracts if total_contracts else 0, 3),
        },
        {
            "field": "City",
            "missing_count": city_missing_count,
            "missing_pct": round_float(city_missing_count / total_contracts if total_contracts else 0, 3),
        },
        {
            "field": "Termination Date",
            "missing_count": termination_missing_count,
            "missing_pct": round_float(termination_missing_count / total_contracts if total_contracts else 0, 3),
        },
        {
            "field": "Client Nationality",
            "missing_count": nat_missing_count,
            "missing_pct": round_float(nat_missing_count / total_contracts if total_contracts else 0, 3),
        },
    ]

    data_quality = {
        "missing_summary": missing_summary,
        "invalid_city_values": [{"value": key, "count": count} for key, count in invalid_city_counts.items()],
        "payment_lag_summary": payment_lag_summary,
        "negative_duration_count": negative_duration_count,
        "payment_lag_negative_count": payment_lag_negative_count,
        "notes": [
            "Contracts without a termination date are treated as Active.",
            "Numeric city labels are recoded to Unknown for city-level reporting.",
            "Missing client nationality is consolidated under 'Unknown nationality'.",
            "Termination counts by month are based on termination dates.",
        ],
    }

    sanity_checks = {
        "nationality_total": sum(item["contracts_count"] for item in nationality_stats),
        "kiosk_total": sum(item["contracts_count"] for item in kiosk_stats),
        "combination_total": sum(item["contracts_count"] for item in combination_stats),
        "overall_total": total_contracts,
    }

    data_bundle = {
        "overallKpis": overall_kpis,
        "nationalityStats": nationality_stats,
        "lowChurnNationalities": low_churn_nationalities,
        "highChurnNationalities": high_churn_nationalities,
        "kioskStats": kiosk_stats,
        "bestKiosks": best_kiosks,
        "worstKiosks": worst_kiosks,
        "combinationStats": combination_stats,
        "riskSegments": risk_segments,
        "heatmapData": {
            "kiosks": heatmap_kiosks,
            "nationalities": heatmap_nationalities,
            "terminationRateMatrix": termination_rate_matrix,
            "contractsMatrix": contracts_matrix,
            "tooltipMatrix": tooltip_matrix,
        },
        "nationalityDrilldown": nationality_drilldown,
        "kioskDrilldown": kiosk_drilldown,
        "kioskTopNationalityTimeseries": {
            "months": contract_months_sorted,
            "series": kiosk_top_timeseries,
        },
        "clientMetrics": client_metrics,
        "timeSeriesByNationality": time_series_by_nationality,
        "timeSeries": time_series,
        "cityStats": city_stats,
        "dataQuality": data_quality,
        "sanityChecks": sanity_checks,
    }

    return data_bundle


def compute_dataset(df_slice: pd.DataFrame) -> Dict:
    total_contracts = int(len(df_slice))
    unique_clients = int(df_slice["Contract Client ID"].nunique()) if total_contracts else 0
    terminated_contracts = int((df_slice["contract_status"] == "Terminated").sum())
    active_contracts = total_contracts - terminated_contracts
    termination_rate = terminated_contracts / total_contracts if total_contracts else 0.0

    contract_date_min = df_slice["Contract Creation Date"].min()
    contract_date_max = df_slice["Contract Creation Date"].max()

    contract_month_period_series = df_slice["contract_month_period"].dropna()
    max_contract_period = contract_month_period_series.max() if not contract_month_period_series.empty else pd.NaT
    min_contract_period = contract_month_period_series.min() if not contract_month_period_series.empty else pd.NaT
    if (
        pd.isna(max_contract_period)
        or pd.isna(contract_date_max)
        or pd.isna(min_contract_period)
    ):
        last_full_month_period = None
    elif contract_date_max.is_month_end:
        last_full_month_period = max_contract_period
    else:
        last_full_month_period = max_contract_period - 1

    if last_full_month_period is None or pd.isna(min_contract_period) or min_contract_period > last_full_month_period:
        timeline_months: List[str] = []
    else:
        timeline_months = [
            str(period) for period in pd.period_range(min_contract_period, last_full_month_period, freq="M")
        ]

    overall_kpis = {
        "total_contracts": total_contracts,
        "unique_clients": unique_clients,
        "active_contracts": active_contracts,
        "terminated_contracts": terminated_contracts,
        "termination_rate": round_float(termination_rate, 3),
        "contract_date_min": contract_date_min.strftime("%Y-%m-%d") if pd.notna(contract_date_min) else None,
        "contract_date_max": contract_date_max.strftime("%Y-%m-%d") if pd.notna(contract_date_max) else None,
    }

    if total_contracts == 0:
        empty_dataset = {
            "overallKpis": overall_kpis,
            "nationalityStats": [],
            "lowChurnNationalities": [],
            "highChurnNationalities": [],
            "kioskStats": [],
            "bestKiosks": [],
            "worstKiosks": [],
            "combinationStats": [],
            "riskSegments": [],
            "heatmapData": {
                "kiosks": [],
                "nationalities": [],
                "terminationRateMatrix": [],
                "contractsMatrix": [],
                "tooltipMatrix": [],
            },
            "nationalityDrilldown": {},
            "kioskDrilldown": {},
            "kioskTopNationalityTimeseries": {
                "months": timeline_months,
                "series": {},
            },
            "cancellationInsights": {
                "rateSummary": [],
                "timeToCancel": [],
            },
            "clientMetrics": {
                "total_unique_clients": 0,
                "clients_with_one_contract": 0,
                "clients_with_multiple_contracts": 0,
                "repeat_rate": 0,
                "single_client_termination_rate": 0,
                "multi_client_termination_rate": 0,
                "distribution": [
                    {"contracts": bucket, "clients": 0} for bucket in ["1", "2", "3", "4", "5+"]
                ],
                "bucket_summary": [],
            },
            "timeSeries": [
                {"month": month, "contracts": 0, "terminations": 0} for month in timeline_months
            ],
            "cityStats": [],
            "dataQuality": {
                "missing_summary": [
                    {"field": "Paid Date", "missing_count": 0, "missing_pct": 0},
                    {"field": "Client Name", "missing_count": 0, "missing_pct": 0},
                    {"field": "City", "missing_count": 0, "missing_pct": 0},
                    {"field": "Termination Date", "missing_count": 0, "missing_pct": 0},
                    {"field": "Client Nationality", "missing_count": 0, "missing_pct": 0},
                ],
                "invalid_city_values": [],
                "payment_lag_summary": {
                    "count": 0,
                    "min": None,
                    "q1": None,
                    "median": None,
                    "q3": None,
                    "max": None,
                },
                "negative_duration_count": 0,
                "payment_lag_negative_count": 0,
                "notes": [
                    "Contracts without a termination date are treated as Active.",
                    "Numeric city labels are recoded to Unknown for city-level reporting.",
                    "Missing client nationality is consolidated under 'Unknown nationality'.",
                    "Termination counts by month are based on termination dates.",
                ],
            },
            "sanityChecks": {
                "nationality_total": 0,
                "kiosk_total": 0,
                "combination_total": 0,
                "overall_total": 0,
            },
        }
        return empty_dataset

    nationality_group = (
        df_slice.groupby("Client Nationality")
        .agg(
            contracts_count=("Contract ID", "count"),
            terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            active_count=("contract_status", lambda s: int((s == "Active").sum())),
            avg_duration_days=("contract_duration_days", mean_without_na),
            median_duration_days=("contract_duration_days", median_without_na),
        )
        .reset_index()
    )
    nationality_group["termination_rate"] = nationality_group["terminated_count"] / nationality_group["contracts_count"]
    nationality_group["contracts_share"] = nationality_group["contracts_count"] / total_contracts
    nationality_stats: List[Dict] = []
    for row in nationality_group.sort_values("contracts_count", ascending=False).to_dict("records"):
        nationality_stats.append(
            {
                "nationality": row["Client Nationality"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
                "contracts_share": round_float(row["contracts_share"]),
                "avg_duration_days": round_float(row["avg_duration_days"]),
                "median_duration_days": round_float(row["median_duration_days"]),
            }
        )

    nationality_threshold = 30
    eligible_nationalities = [n for n in nationality_stats if n["contracts_count"] >= nationality_threshold]
    low_churn_nationalities = sorted(
        eligible_nationalities,
        key=lambda x: (x["termination_rate"] if x["termination_rate"] is not None else 0, -x["contracts_count"]),
    )[:5]
    high_churn_nationalities = sorted(
        eligible_nationalities,
        key=lambda x: (-(x["termination_rate"] if x["termination_rate"] is not None else -1), -x["contracts_count"]),
    )[:5]

    kiosk_group = (
        df_slice.groupby("Interview Location")
        .agg(
            contracts_count=("Contract ID", "count"),
            terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            active_count=("contract_status", lambda s: int((s == "Active").sum())),
            avg_duration_days=("contract_duration_days", mean_without_na),
            median_duration_days=("contract_duration_days", median_without_na),
        )
        .reset_index()
    )
    kiosk_group["termination_rate"] = kiosk_group["terminated_count"] / kiosk_group["contracts_count"]
    kiosk_stats: List[Dict] = []
    for row in kiosk_group.sort_values("contracts_count", ascending=False).to_dict("records"):
        kiosk_stats.append(
            {
                "kiosk": row["Interview Location"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
                "avg_duration_days": round_float(row["avg_duration_days"]),
                "median_duration_days": round_float(row["median_duration_days"]),
            }
        )

    kiosk_threshold = 50
    eligible_kiosks = [k for k in kiosk_stats if k["contracts_count"] >= kiosk_threshold]
    best_kiosks = sorted(
        eligible_kiosks,
        key=lambda x: (x["termination_rate"] if x["termination_rate"] is not None else 0, -x["contracts_count"]),
    )[:5]
    worst_kiosks = sorted(
        eligible_kiosks,
        key=lambda x: (-(x["termination_rate"] if x["termination_rate"] is not None else -1), -x["contracts_count"]),
    )[:5]

    nk_group = (
        df_slice.groupby(["Interview Location", "Client Nationality"])
        .agg(
            contracts_count=("Contract ID", "count"),
            terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            active_count=("contract_status", lambda s: int((s == "Active").sum())),
            avg_duration_days=("contract_duration_days", mean_without_na),
            median_duration_days=("contract_duration_days", median_without_na),
        )
        .reset_index()
    )
    nk_group["termination_rate"] = nk_group["terminated_count"] / nk_group["contracts_count"]

    combination_stats: List[Dict] = []
    for row in nk_group.to_dict("records"):
        combination_stats.append(
            {
                "kiosk": row["Interview Location"],
                "nationality": row["Client Nationality"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
                "avg_duration_days": round_float(row["avg_duration_days"]),
                "median_duration_days": round_float(row["median_duration_days"]),
            }
        )

    risk_threshold_contracts = 10
    risk_threshold_rate = 0.6
    risk_segments = [
        combo
        for combo in combination_stats
        if combo["contracts_count"] >= risk_threshold_contracts
        and (combo["termination_rate"] or 0) >= risk_threshold_rate
    ]
    risk_segments = sorted(
        risk_segments,
        key=lambda x: (-(x["termination_rate"] if x["termination_rate"] is not None else -1), -x["contracts_count"]),
    )

    top_nationalities = [n["nationality"] for n in nationality_stats[:12]]
    top_kiosks = [k["kiosk"] for k in kiosk_stats[:12]]
    heatmap_nationalities = list(top_nationalities)
    heatmap_kiosks = list(top_kiosks)

    data_map = {
        (row["kiosk"], row["nationality"]): row for row in combination_stats if row["contracts_count"] > 0
    }
    termination_rate_matrix: List[List[float | None]] = []
    contracts_matrix: List[List[int | None]] = []
    tooltip_matrix: List[List[str]] = []

    for kiosk in top_kiosks:
        rate_row: List[float | None] = []
        count_row: List[int | None] = []
        tooltip_row: List[str] = []
        for nationality in top_nationalities:
            combo = data_map.get((kiosk, nationality))
            if combo:
                rate_row.append(combo["termination_rate"])
                count_row.append(combo["contracts_count"])
                tooltip_row.append(
                    f"{kiosk} / {nationality}<br>Contracts: {combo['contracts_count']}<br>"
                    f"Terminated: {combo['terminated_count']}<br>"
                    f"Avg duration: {combo['avg_duration_days'] if combo['avg_duration_days'] is not None else 'N/A'} days<br>"
                    f"Median duration: {combo['median_duration_days'] if combo['median_duration_days'] is not None else 'N/A'} days<br>"
                    f"Termination rate: {round_float(combo['termination_rate'], 3) if combo['termination_rate'] is not None else 'N/A'}"
                )
            else:
                rate_row.append(None)
                count_row.append(None)
                tooltip_row.append(f"{kiosk} / {nationality}<br>No contracts")
        termination_rate_matrix.append(rate_row)
        contracts_matrix.append(count_row)
        tooltip_matrix.append(tooltip_row)

    nationality_drilldown: Dict[str, List[Dict]] = {}
    kiosk_drilldown: Dict[str, List[Dict]] = {}

    for nationality, group in nk_group.groupby("Client Nationality"):
        sorted_rows = group.sort_values("contracts_count", ascending=False).to_dict("records")
        nationality_drilldown[nationality] = [
            {
                "kiosk": row["Interview Location"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
                "avg_duration_days": round_float(row["avg_duration_days"]),
                "median_duration_days": round_float(row["median_duration_days"]),
            }
            for row in sorted_rows
        ]

    for kiosk, group in nk_group.groupby("Interview Location"):
        sorted_rows = group.sort_values("contracts_count", ascending=False).to_dict("records")
        kiosk_drilldown[kiosk] = [
            {
                "nationality": row["Client Nationality"],
                "contracts_count": int(row["contracts_count"]),
                "terminated_count": int(row["terminated_count"]),
                "active_count": int(row["active_count"]),
                "termination_rate": round_float(row["termination_rate"]),
                "avg_duration_days": round_float(row["avg_duration_days"]),
                "median_duration_days": round_float(row["median_duration_days"]),
            }
            for row in sorted_rows
        ]

    client_group = (
        df_slice.groupby("Contract Client ID")
        .agg(
            client_contracts_count=("Contract ID", "count"),
            client_terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            client_active_count=("contract_status", lambda s: int((s == "Active").sum())),
            first_contract_date=("Contract Creation Date", "min"),
            last_contract_date=("Contract Creation Date", "max"),
        )
        .reset_index()
    )

    client_group["client_term_rate"] = (
        client_group["client_terminated_count"] / client_group["client_contracts_count"]
    ).replace([np.inf, -np.inf], np.nan)

    clients_with_one = client_group[client_group["client_contracts_count"] == 1]
    clients_with_multi = client_group[client_group["client_contracts_count"] > 1]

    clients_with_one_contract = int(len(clients_with_one))
    clients_with_multiple_contracts = int(len(clients_with_multi))

    repeat_rate = clients_with_multiple_contracts / unique_clients if unique_clients else 0
    single_client_termination_rate = float(
        (clients_with_one["client_terminated_count"] > 0).mean()
    ) if clients_with_one_contract else 0.0
    multi_client_termination_rate = float(
        (clients_with_multi["client_terminated_count"] > 0).mean()
    ) if clients_with_multiple_contracts else 0.0

    def bucket_contracts(count: int) -> str:
        return str(count) if count < 5 else "5+"

    if not client_group.empty:
        client_group["contracts_bucket"] = client_group["client_contracts_count"].apply(bucket_contracts)
    else:
        client_group["contracts_bucket"] = []

    distribution = (
        client_group.groupby("contracts_bucket").size().reindex(["1", "2", "3", "4", "5+"], fill_value=0)
        if not client_group.empty
        else pd.Series([0, 0, 0, 0, 0], index=["1", "2", "3", "4", "5+"], dtype=int)
    )
    distribution_records = [
        {"contracts": bucket, "clients": int(distribution.loc[bucket])} for bucket in ["1", "2", "3", "4", "5+"]
    ]

    bucket_summary = []
    if not client_group.empty:
        for bucket, bucket_df in client_group.groupby("contracts_bucket"):
            avg_term_rate = bucket_df["client_term_rate"].mean()
            bucket_summary.append(
                {
                    "bucket": bucket,
                    "clients": int(len(bucket_df)),
                    "avg_client_termination_rate": round_float(avg_term_rate),
                }
            )
        bucket_summary = sorted(bucket_summary, key=lambda x: ["1", "2", "3", "4", "5+"].index(x["bucket"]))

    client_metrics = {
        "total_unique_clients": unique_clients,
        "clients_with_one_contract": clients_with_one_contract,
        "clients_with_multiple_contracts": clients_with_multiple_contracts,
        "repeat_rate": round_float(repeat_rate),
        "single_client_termination_rate": round_float(single_client_termination_rate),
        "multi_client_termination_rate": round_float(multi_client_termination_rate),
        "distribution": distribution_records,
        "bucket_summary": bucket_summary,
    }

    contracts_by_month = df_slice.groupby("contract_month").size()
    terminations_by_month = df_slice[df_slice["contract_status"] == "Terminated"].groupby("termination_month").size()

    if timeline_months:
        time_series_months = timeline_months
    else:
        time_series_months = [
            month for month in sorted(df_slice["contract_month"].dropna().unique()) if isinstance(month, str) and month != "NaT"
        ]

    time_series = [
        {
            "month": month,
            "contracts": int(contracts_by_month.get(month, 0)),
            "terminations": int(terminations_by_month.get(month, 0)),
        }
        for month in time_series_months
    ]

    nationality_monthly_contracts = (
        df_slice.groupby(["Client Nationality", "contract_month"]).size()
    )
    nationality_monthly_terminations = (
        df_slice[df_slice["contract_status"] == "Terminated"].groupby(["Client Nationality", "termination_month"]).size()
    )
    unique_nationalities = sorted(df_slice["Client Nationality"].dropna().unique())
    time_series_by_nationality: Dict[str, List[Dict[str, int]]] = {}
    for nationality in unique_nationalities:
        series_entries = [
            {
                "month": month,
                "contracts": int(nationality_monthly_contracts.get((nationality, month), 0)),
                "terminations": int(nationality_monthly_terminations.get((nationality, month), 0)),
            }
            for month in time_series_months
        ]
        time_series_by_nationality[nationality] = series_entries
    time_series_by_nationality["All nationalities"] = time_series

    kiosk_monthly_counts = (
        df_slice.groupby(["Interview Location", "Client Nationality", "contract_month"])
        .size()
        .reset_index(name="contracts_count")
    )
    monthly_lookup = {
        (row["Interview Location"], row["Client Nationality"], row["contract_month"]): int(row["contracts_count"])
        for row in kiosk_monthly_counts.to_dict("records")
        if isinstance(row["contract_month"], str) and row["contract_month"] != "NaT"
    }
    status_monthly_counts = (
        df_slice.groupby(["Interview Location", "Client Nationality", "contract_month", "contract_status"])
        .size()
        .reset_index(name="contracts_count")
    )
    status_lookup = {
        (
            row["Interview Location"],
            row["Client Nationality"],
            row["contract_month"],
            row["contract_status"],
        ): int(row["contracts_count"])
        for row in status_monthly_counts.to_dict("records")
        if isinstance(row["contract_month"], str) and row["contract_month"] != "NaT"
    }

    kiosk_top_timeseries: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    for kiosk, entries in kiosk_drilldown.items():
        timeseries_nationalities = [entry["nationality"] for entry in entries[:5]]
        if not timeseries_nationalities:
            continue
        kiosk_top_timeseries[kiosk] = {
            "nationalities": timeseries_nationalities,
            "all": {},
            "Active": {},
            "Terminated": {},
        }
        for nationality in timeseries_nationalities:
            all_counts: List[int] = []
            active_counts: List[int] = []
            terminated_counts: List[int] = []
            for month in time_series_months:
                all_counts.append(monthly_lookup.get((kiosk, nationality, month), 0))
                active_counts.append(status_lookup.get((kiosk, nationality, month, "Active"), 0))
                terminated_counts.append(status_lookup.get((kiosk, nationality, month, "Terminated"), 0))
            kiosk_top_timeseries[kiosk]["all"][nationality] = all_counts
            kiosk_top_timeseries[kiosk]["Active"][nationality] = active_counts
            kiosk_top_timeseries[kiosk]["Terminated"][nationality] = terminated_counts

    city_group = (
        df_slice.groupby("City_clean")
        .agg(
            contracts_count=("Contract ID", "count"),
            terminated_count=("contract_status", lambda s: int((s == "Terminated").sum())),
            active_count=("contract_status", lambda s: int((s == "Active").sum())),
            avg_duration_days=("contract_duration_days", mean_without_na),
            median_duration_days=("contract_duration_days", median_without_na),
        )
        .reset_index()
    )
    city_group["termination_rate"] = city_group["terminated_count"] / city_group["contracts_count"]
    city_stats = [
        {
            "city": row["City_clean"],
            "contracts_count": int(row["contracts_count"]),
            "terminated_count": int(row["terminated_count"]),
            "active_count": int(row["active_count"]),
            "termination_rate": round_float(row["termination_rate"]),
            "avg_duration_days": round_float(row["avg_duration_days"]),
            "median_duration_days": round_float(row["median_duration_days"]),
        }
        for row in city_group.sort_values("contracts_count", ascending=False).to_dict("records")
    ]

    valid_payment_lag = df_slice["payment_lag_days"].dropna()
    payment_lag_summary = {
        "count": int(valid_payment_lag.count()),
        "min": round_float(valid_payment_lag.min(), 2) if not valid_payment_lag.empty else None,
        "q1": round_float(valid_payment_lag.quantile(0.25), 2) if not valid_payment_lag.empty else None,
        "median": round_float(valid_payment_lag.median(), 2) if not valid_payment_lag.empty else None,
        "q3": round_float(valid_payment_lag.quantile(0.75), 2) if not valid_payment_lag.empty else None,
        "max": round_float(valid_payment_lag.max(), 2) if not valid_payment_lag.empty else None,
    }

    paid_missing_count = int(df_slice["Paid Date"].isna().sum())
    client_name_missing_count = int(df_slice["Client Name"].isna().sum())
    city_missing_count = int(
        df_slice["City"].isna().sum()
        + df_slice["City"].astype(str).str.strip().eq("").sum()
    )
    termination_missing_count = int(df_slice["Termination Date"].isna().sum())
    nat_missing_count = int((df_slice["Client Nationality"] == "Unknown nationality").sum())

    invalid_city_counts = (
        df_slice["City"]
        .dropna()
        .astype(str)
        .str.strip()
        .pipe(lambda s: s[s.str.isdigit()])
        .value_counts()
        .to_dict()
    )

    negative_duration_count = int(df_slice.get("negative_duration_flag", pd.Series(dtype=bool)).sum())
    payment_lag_negative_count = int(df_slice.get("payment_lag_negative_flag", pd.Series(dtype=bool)).sum())

    missing_summary = [
        {
            "field": "Paid Date",
            "missing_count": paid_missing_count,
            "missing_pct": round_float(paid_missing_count / total_contracts if total_contracts else 0, 3),
        },
        {
            "field": "Client Name",
            "missing_count": client_name_missing_count,
            "missing_pct": round_float(client_name_missing_count / total_contracts if total_contracts else 0, 3),
        },
        {
            "field": "City",
            "missing_count": city_missing_count,
            "missing_pct": round_float(city_missing_count / total_contracts if total_contracts else 0, 3),
        },
        {
            "field": "Termination Date",
            "missing_count": termination_missing_count,
            "missing_pct": round_float(termination_missing_count / total_contracts if total_contracts else 0, 3),
        },
        {
            "field": "Client Nationality",
            "missing_count": nat_missing_count,
            "missing_pct": round_float(nat_missing_count / total_contracts if total_contracts else 0, 3),
        },
    ]

    data_quality = {
        "missing_summary": missing_summary,
        "invalid_city_values": [{"value": key, "count": count} for key, count in invalid_city_counts.items()],
        "payment_lag_summary": payment_lag_summary,
        "negative_duration_count": negative_duration_count,
        "payment_lag_negative_count": payment_lag_negative_count,
        "notes": [
            "Contracts without a termination date are treated as Active.",
            "Numeric city labels are recoded to Unknown for city-level reporting.",
            "Missing client nationality is consolidated under 'Unknown nationality'.",
            "Termination counts by month are based on termination dates.",
        ],
    }

    sanity_checks = {
        "nationality_total": sum(item["contracts_count"] for item in nationality_stats),
        "kiosk_total": sum(item["contracts_count"] for item in kiosk_stats),
        "combination_total": sum(item["contracts_count"] for item in combination_stats),
        "overall_total": total_contracts,
    }

    cancellation_insights = compute_cancellation_insights(df_slice)

    return {
        "overallKpis": overall_kpis,
        "nationalityStats": nationality_stats,
        "lowChurnNationalities": low_churn_nationalities,
        "highChurnNationalities": high_churn_nationalities,
        "kioskStats": kiosk_stats,
        "bestKiosks": best_kiosks,
        "worstKiosks": worst_kiosks,
        "combinationStats": combination_stats,
        "riskSegments": risk_segments,
        "heatmapData": {
            "kiosks": heatmap_kiosks,
            "nationalities": heatmap_nationalities,
            "terminationRateMatrix": termination_rate_matrix,
            "contractsMatrix": contracts_matrix,
            "tooltipMatrix": tooltip_matrix,
        },
        "nationalityDrilldown": nationality_drilldown,
        "kioskDrilldown": kiosk_drilldown,
        "kioskTopNationalityTimeseries": {
            "months": time_series_months,
            "series": kiosk_top_timeseries,
        },
        "cancellationInsights": cancellation_insights,
        "clientMetrics": client_metrics,
        "timeSeriesByNationality": time_series_by_nationality,
        "timeSeries": time_series,
        "cityStats": city_stats,
        "dataQuality": data_quality,
        "sanityChecks": sanity_checks,
    }


def prepare_data() -> Dict:
    df = pd.read_csv(DATA_PATH)

    for col in ["Contract ID", "Interview ID", "Contract Client ID"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    for col in ["Contract Creation Date", "Paid Date", "Termination Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Interview Location"] = df["Interview Location"].fillna("Unknown Location").astype(str).str.strip()

    def clean_nationality(value) -> str:
        if pd.isna(value):
            return "Unknown nationality"
        stripped = str(value).strip()
        return stripped if stripped else "Unknown nationality"

    df["Client Nationality"] = df["Client Nationality"].apply(clean_nationality)

    def clean_contract_type(value) -> str:
        if pd.isna(value):
            return "Unknown type"
        stripped = str(value).strip()
        return stripped if stripped else "Unknown type"

    df["contract_type"] = df["Contract Type"].apply(clean_contract_type)

    def clean_city(value) -> str:
        if pd.isna(value):
            return "Unknown"
        stripped = str(value).strip()
        if not stripped:
            return "Unknown"
        if stripped.isdigit():
            return "Unknown"
        return stripped

    df["City_clean"] = df["City"].apply(clean_city)

    terminated_mask = df["Termination Date"].notna()
    df["contract_status"] = np.where(terminated_mask, "Terminated", "Active")

    duration_raw = np.where(
        terminated_mask,
        (df["Termination Date"] - df["Contract Creation Date"]).dt.days,
        np.nan,
    )
    df["negative_duration_flag"] = np.where(np.isnan(duration_raw), False, duration_raw < 0)
    df["contract_duration_days"] = np.where(df["negative_duration_flag"], np.nan, duration_raw)

    df["payment_lag_days"] = (df["Paid Date"] - df["Contract Creation Date"]).dt.days
    df["payment_lag_negative_flag"] = df["payment_lag_days"] < 0

    df["contract_year"] = df["Contract Creation Date"].dt.year
    df["contract_month_period"] = df["Contract Creation Date"].dt.to_period("M")
    df["contract_month"] = df["contract_month_period"].astype(str)
    df["termination_month"] = df["Termination Date"].dt.to_period("M").astype(str)

    contract_types = sorted(df["contract_type"].unique())
    datasets: Dict[str, Dict] = {}

    all_types_label = "All Contract Types"
    datasets[all_types_label] = compute_dataset(df)

    for contract_type in contract_types:
        subset = df[df["contract_type"] == contract_type]
        datasets[contract_type] = compute_dataset(subset)

    return {
        "contractTypes": [all_types_label] + contract_types,
        "defaultContractType": all_types_label,
        "datasets": datasets,
    }


def build_html(data_bundle: Dict) -> str:
    # CSS, HTML, and JS template
    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Retail Sales & Churn Analysis by Nationality & Kiosk</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
        :root {{
            --bg-color: #f5f7fb;
            --card-bg: #ffffff;
            --accent: #2a6f97;
            --accent-soft: #bcd4e6;
            --text-primary: #1f2933;
            --text-muted: #52606d;
            --danger: #d9534f;
            --success: #2d936c;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        body {{
            margin: 0;
            background: var(--bg-color);
            color: var(--text-primary);
        }}

        .container {{
            max-width: 1280px;
            margin: 0 auto;
            padding: 32px 24px 64px;
        }}

        header {{
            margin-bottom: 32px;
        }}

        h1 {{
            margin: 0;
            font-size: 32px;
            font-weight: 700;
        }}

        header p {{
            margin-top: 8px;
            color: var(--text-muted);
            font-size: 16px;
        }}

        section {{
            margin-top: 48px;
        }}

        section h2 {{
            margin: 0 0 16px;
            font-size: 24px;
            font-weight: 600;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }}

        .kpi-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        }}

        .kpi-card h3 {{
            margin: 0;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.04em;
        }}

        .kpi-card p {{
            margin: 12px 0 0;
            font-size: 28px;
            font-weight: 700;
        }}

        .chart-card, .table-card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
            margin-top: 24px;
        }}

        .chart-card {{
            display: flex;
            flex-direction: column;
            gap: 18px;
        }}

        .chart-card h3 {{
            margin: 0;
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .chart-card .controls {{
            margin: 0;
            width: 100%;
            justify-content: flex-start;
        }}

        .table-card {{
            display: flex;
            flex-direction: column;
            gap: 18px;
        }}

        .table-card h3 {{
            margin: 0;
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }}

        .highlight-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            border-left: 5px solid var(--accent);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }}

        .highlight-card.danger {{
            border-left-color: var(--danger);
        }}

        .highlight-card.success {{
            border-left-color: var(--success);
        }}

        .highlight-card h3 {{
            margin-top: 0;
            font-size: 18px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            margin: 0;
        }}

        .table-card table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            margin: 0;
        }}

        thead {{
            background: var(--accent);
            color: #ffffff;
        }}

        th, td {{
            padding: 12px 14px;
            text-align: center;
            font-size: 14px;
            vertical-align: top;
        }}

        .table-card table th:first-child,
        .table-card table td:first-child {{
            text-align: left;
        }}

        .cancel-compact {{
            width: 100%;
            table-layout: fixed;
        }}
        .cancel-compact th, .cancel-compact td {{
            text-align: center;
            padding-left: 10px;
            padding-right: 10px;
        }}
        .cancel-compact th:first-child,
        .cancel-compact td:first-child {{
            text-align: left;
        }}
        .cancel-compact td:nth-child(3),
        .cancel-compact td:nth-child(5),
        .cancel-compact td:nth-child(7) {{
            white-space: nowrap;
        }}

        tbody tr:nth-child(every 2) {{
            background: #fafcff;
        }}

        tbody tr:nth-child(odd) {{
            background: #ffffff;
        }}

        tbody tr:hover {{
            background: #eef5ff;
        }}

        .table-actions {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            gap: 16px;
            flex-wrap: wrap;
        }}

        .search-input {{
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid #d1d5db;
            min-width: 200px;
        }}

        .tag {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            background: var(--accent-soft);
            color: var(--accent);
            margin-right: 8px;
        }}

        .tag.danger {{
            background: rgba(217, 83, 79, 0.18);
            color: var(--danger);
        }}

        .tag.success {{
            background: rgba(45, 147, 108, 0.15);
            color: var(--success);
        }}

        .pill {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: var(--card-bg);
            border: 1px solid #d9e2ec;
            font-size: 13px;
            font-weight: 500;
            margin-right: 8px;
        }}

        .controls {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}

        .controls label {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-muted);
            margin-right: 8px;
        }}

        .controls select, .controls input, .controls button {{
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid #cbd5e1;
            background: #ffffff;
            font-size: 14px;
            cursor: pointer;
        }}

        .search-dropdown {{
            position: relative;
            flex: 1 1 240px;
            max-width: 320px;
        }}

        .combo-input {{
            position: relative;
            padding-right: 34px;
            background:
                linear-gradient(45deg, transparent 50%, #52606d 50%),
                linear-gradient(135deg, #52606d 50%, transparent 50%);
            background-position: calc(100% - 18px) 50%, calc(100% - 12px) 50%;
            background-size: 6px 6px, 6px 6px;
            background-repeat: no-repeat;
            width: 100%;
        }}

        .search-dropdown-list {{
            position: absolute;
            top: calc(100% + 6px);
            left: 0;
            right: 0;
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
            max-height: 260px;
            overflow-y: auto;
            padding: 4px 0;
            display: none;
            z-index: 50;
        }}

        .search-dropdown-list.open {{
            display: block;
        }}

        .search-dropdown-item {{
            width: 100%;
            border: none;
            background: transparent;
            text-align: left;
            padding: 8px 14px;
            font-size: 14px;
            color: var(--text-primary);
            cursor: pointer;
        }}

        .search-dropdown-item:hover,
        .search-dropdown-item.highlighted {{
            background: #eef5ff;
        }}

        .search-dropdown-empty {{
            padding: 8px 14px;
            font-size: 13px;
            color: #52606d;
        }}

        .controls button.active {{
            background: var(--accent);
            color: #ffffff;
        }}

        .contract-type-control {{
            margin-top: 16px;
        }}

        .notes-list {{
            padding-left: 20px;
        }}

        .notes-list li {{
            margin-bottom: 8px;
            color: var(--text-muted);
        }}

        footer {{
            margin-top: 48px;
            color: var(--text-muted);
            font-size: 13px;
            text-align: center;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 24px 16px 48px;
            }}
        }}
    </style>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Retail Sales & Churn Analysis by Nationality & Kiosk</h1>
            <p>Interactive management dashboard highlighting nationality performance, kiosk trends, and client retention for retail contracts.</p>
            <div class="controls contract-type-control">
                <label for="contractTypeSelect">Contract Type</label>
                <select id="contractTypeSelect"></select>
            </div>
        </header>

        <section id="overview">
            <h2>Overall KPIs</h2>
            <div class="kpi-grid" id="kpiGrid"></div>
        </section>

        <section id="nationality-overview">
            <h2>Section 1 — Nationality Overview (Retail Level)</h2>
            <div class="chart-card" id="nationalityVolumes">
                <h3>Contracts by Nationality (Top 20)</h3>
                <div id="nationalityBarChart" style="height:420px;"></div>
            </div>
            <div class="table-card">
                <div class="table-actions">
                    <h3>Nationality Leaderboard</h3>
                    <input type="search" id="nationalitySearch" class="search-input" placeholder="Search nationality..." />
                </div>
                <div class="table-wrapper">
                    <table id="nationalityTable">
                        <thead></thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
            <div class="card-grid">
                <div class="highlight-card success">
                    <h3>Top Low-Churn Nationalities</h3>
                    <ul id="lowChurnList"></ul>
                </div>
                <div class="highlight-card danger">
                    <h3>High Churn Risk Nationalities</h3>
                    <ul id="highChurnList"></ul>
                </div>
            </div>
        </section>

        <section id="kiosk-overview">
            <h2>Section 2 — Kiosk Overview</h2>
            <div class="chart-card">
                <h3>Contracts by Kiosk (Top 15)</h3>
                <div id="kioskBarChart" style="height:420px;"></div>
            </div>
            <div class="table-card">
                <div class="table-actions">
                    <h3>Kiosk Performance</h3>
                    <input type="search" id="kioskSearch" class="search-input" placeholder="Search kiosk..." />
                </div>
                <table id="kioskTable">
                    <thead></thead>
                    <tbody></tbody>
                </table>
            </div>
            <div class="card-grid">
                <div class="highlight-card success">
                    <h3>Best Performing Kiosks</h3>
                    <ul id="bestKioskList"></ul>
                </div>
                <div class="highlight-card danger">
                    <h3>Highest Churn Kiosks</h3>
                    <ul id="worstKioskList"></ul>
                </div>
            </div>
        </section>

        <section id="nationality-kiosk">
            <h2>Section 3 — Nationality × Kiosk Insights</h2>
            <div class="chart-card">
                <div class="controls">
                    <button id="heatmapRateBtn" class="active" data-mode="rate">Termination Rate</button>
                    <button id="heatmapVolumeBtn" data-mode="volume">Contracts Volume</button>
                </div>
                <div id="heatmap" style="height:520px;"></div>
            </div>
            <div class="chart-card">
                <h3>Drill-down by Nationality</h3>
                <div class="controls">
                    <div class="search-dropdown">
                        <input id="nationalityCombo" class="search-input combo-input" type="search" placeholder="Search or pick nationality..." autocomplete="off" />
                        <div id="nationalityComboList" class="search-dropdown-list"></div>
                    </div>
                </div>
                <div id="nationalityDrilldownCounts" style="height:360px;"></div>
                <div id="nationalityDrilldownRates" style="height:360px;"></div>
            </div>
            <div class="chart-card">
                <h3>Drill-down by Kiosk</h3>
                <div class="controls">
                    <div class="search-dropdown">
                        <input id="kioskCombo" class="search-input combo-input" type="search" placeholder="Search or pick kiosk..." autocomplete="off" />
                        <div id="kioskComboList" class="search-dropdown-list"></div>
                    </div>
                </div>
                <div id="kioskDrilldownCounts" style="height:360px;"></div>
                <div id="kioskDrilldownRates" style="height:360px;"></div>
                <div class="controls">
                    <button id="kioskTimeseriesAll" class="active" data-timeseries-mode="all">All Contracts</button>
                    <button id="kioskTimeseriesActive" data-timeseries-mode="Active">Active Only</button>
                    <button id="kioskTimeseriesTerminated" data-timeseries-mode="Terminated">Terminated Only</button>
                </div>
                <div id="kioskDrilldownTimeseries" style="height:420px;"></div>
            </div>
            <div class="table-card">
                <div class="table-actions">
                    <h3>High-Risk Nationality × Kiosk Segments</h3>
                </div>
                <table id="riskTable">
                    <thead></thead>
                    <tbody></tbody>
                </table>
            </div>
        </section>

        <section id="client-analysis">
            <h2>Section 4 — Client-Level Analysis</h2>
            <div class="card-grid">
                <div class="highlight-card">
                    <h3>Client Base Overview</h3>
                    <p id="clientKpiSummary"></p>
                    <div id="clientTags"></div>
                </div>
                <div class="highlight-card">
                    <h3>Termination Perspective</h3>
                    <p id="clientTerminationSummary"></p>
                </div>
            </div>
            <div class="chart-card">
                <h3>Distribution of Contracts per Client</h3>
                <div id="clientDistributionChart" style="height:360px;"></div>
            </div>
            <div class="table-card">
                <h3>Client Buckets Summary</h3>
                <table id="clientBucketTable">
                    <thead></thead>
                    <tbody></tbody>
                </table>
            </div>
        </section>

        <section id="time-context">
            <h2>Section 5 — Time & Context</h2>
            <div class="chart-card">
                <h3 id="timeSeriesTitle">Contracts vs Terminations Over Time</h3>
                <div class="controls">
                    <div class="search-dropdown">
                        <input id="timeSeriesNationalityCombo" class="search-input combo-input" type="search" placeholder="All nationalities" autocomplete="off" />
                        <div id="timeSeriesNationalityList" class="search-dropdown-list"></div>
                    </div>
                </div>
                <div id="timeSeriesChart" style="height:420px;"></div>
            </div>
            <div class="chart-card">
                <h3>Contracts by City (Top 15)</h3>
                <div id="cityBarChart" style="height:420px;"></div>
            </div>
        </section>

        <section id="cancellation-insights">
            <h2>Section 6 �?? Cancellation Insights</h2>
            <div class="table-card">
                <h3>Cancellation Rate by Cohort</h3>
                <table id="cancelRateTable">
                    <thead></thead>
                    <tbody></tbody>
                </table>
            </div>
            <div class="table-card">
                <h3>Time-to-Cancel Breakdown</h3>
                <div id="cancelTimeTables"></div>
            </div>
        </section>

        <section id="data-quality">
            <h2>Section 7 ??? Data Quality & Assumptions</h2>
            <div class="table-card">
                <h3>Missing Data Summary</h3>
                <table id="missingTable">
                    <thead></thead>
                    <tbody></tbody>
                </table>
            </div>
            <div class="card-grid">
                <div class="highlight-card">
                    <h3>Rules & Assumptions</h3>
                    <ul class="notes-list" id="assumptionList"></ul>
                </div>
                <div class="highlight-card">
                    <h3>Data Quality Checks</h3>
                    <ul class="notes-list" id="dataQualityNotes"></ul>
                </div>
            </div>
        </section>

        <footer>
            Prepared for Senior Management &mdash; Retail Sales & Churn Analysis. All figures derived from Retail contract IDs dataset.
        </footer>
    </div>

    <script>
        const dataBundle = __DATA_BUNDLE__;
        const datasetsByType = dataBundle.datasets || {{}};
        const contractTypes = dataBundle.contractTypes || [];
        let currentContractType = dataBundle.defaultContractType || contractTypes[0] || '';
        let currentDataset = datasetsByType[currentContractType] || {{}};
        let currentTimeseriesMode = 'all';
        let currentHeatmapMode = 'rate';
        let currentTimeSeriesNationality = 'All nationalities';
        let timeSeriesFilterDropdown = null;

        function formatNumber(value) {{
            if (value === null || value === undefined) return '—';
            return value.toLocaleString();
        }}

        function formatPercent(value, digits = 1) {{
            if (value === null || value === undefined) return '—';
            return (value * 100).toFixed(digits) + '%';
        }}

        function formatRate(value, digits = 1) {{
            if (value === null || value === undefined) return '—';
            return (value * 100).toFixed(digits) + '%';
        }}

        function createSearchDropdown({{ input, listElement, options, onSelect }}) {{
            let filtered = [...options];
            let isOpen = false;
            let highlightedIndex = -1;

            const handleOutsideClick = event => {{
                if (event.target === input || listElement.contains(event.target)) {{
                    return;
                }}
                closeList();
            }};

            function openList() {{
                if (!options.length) {{
                    closeList();
                    return;
                }}
                if (!isOpen) {{
                    isOpen = true;
                    listElement.classList.add('open');
                    renderList();
                    input.setAttribute('aria-expanded', 'true');
                    document.addEventListener('click', handleOutsideClick);
                }} else {{
                    renderList();
                }}
            }}

            function closeList() {{
                if (!isOpen) {{
                    return;
                }}
                isOpen = false;
                highlightedIndex = -1;
                listElement.classList.remove('open');
                listElement.innerHTML = '';
                input.setAttribute('aria-expanded', 'false');
                document.removeEventListener('click', handleOutsideClick);
            }}

            function renderList() {{
                listElement.innerHTML = '';
                if (!filtered.length) {{
                    const empty = document.createElement('div');
                    empty.className = 'search-dropdown-empty';
                    empty.textContent = 'No matches';
                    listElement.appendChild(empty);
                    return;
                }}
                filtered.forEach((option, idx) => {{
                    const item = document.createElement('button');
                    item.type = 'button';
                    item.className = 'search-dropdown-item';
                    if (idx === highlightedIndex) {{
                        item.classList.add('highlighted');
                    }}
                    item.textContent = option;
                    item.onmousedown = event => {{
                        event.preventDefault();
                    }};
                    item.onclick = () => selectOption(option);
                    listElement.appendChild(item);
                }});
            }}

            function filterOptions(term) {{
                const normalized = term.toLowerCase();
                filtered = options.filter(option => option.toLowerCase().includes(normalized));
                highlightedIndex = filtered.length ? 0 : -1;
                if (isOpen) {{
                    renderList();
                }}
            }}

            function selectOption(value) {{
                input.value = value;
                closeList();
                if (onSelect) {{
                    onSelect(value);
                }}
            }}

            const handleFocus = () => {{
                filterOptions(input.value.trim());
                openList();
            }};

            const handleInput = event => {{
                filterOptions(event.target.value.trim());
                openList();
            }};

            const handleKeydown = event => {{
                if (event.key === 'ArrowDown') {{
                    event.preventDefault();
                    if (!isOpen) {{
                        filterOptions(input.value.trim());
                        openList();
                        return;
                    }}
                    if (!filtered.length) {{
                        return;
                    }}
                    highlightedIndex = (highlightedIndex + 1) % filtered.length;
                    renderList();
                }} else if (event.key === 'ArrowUp') {{
                    event.preventDefault();
                    if (!isOpen) {{
                        filterOptions(input.value.trim());
                        openList();
                        return;
                    }}
                    if (!filtered.length) {{
                        return;
                    }}
                    highlightedIndex = (highlightedIndex - 1 + filtered.length) % filtered.length;
                    renderList();
                }} else if (event.key === 'Enter') {{
                    if (isOpen && highlightedIndex >= 0) {{
                        event.preventDefault();
                        selectOption(filtered[highlightedIndex]);
                    }}
                }} else if (event.key === 'Escape') {{
                    if (isOpen) {{
                        event.preventDefault();
                        closeList();
                    }}
                }} else if ((event.altKey && event.key === 'ArrowDown') || event.key === 'F4') {{
                    event.preventDefault();
                    if (isOpen) {{
                        closeList();
                    }} else {{
                        filterOptions('');
                        openList();
                    }}
                }}
            }};

            const handleBlur = () => {{
                setTimeout(() => {{
                    if (!listElement.contains(document.activeElement)) {{
                        closeList();
                    }}
                }}, 120);
            }};

            const handlePointerDown = event => {{
                const arrowZoneStart = input.clientWidth - 32;
                if (event.offsetX >= arrowZoneStart) {{
                    event.preventDefault();
                    if (isOpen) {{
                        closeList();
                    }} else {{
                        input.focus();
                        filterOptions('');
                        openList();
                    }}
                }}
            }};

            input.setAttribute('autocomplete', 'off');
            input.setAttribute('role', 'combobox');
            input.setAttribute('aria-expanded', 'false');
            input.setAttribute('aria-autocomplete', 'list');

            input.addEventListener('focus', handleFocus);
            input.addEventListener('input', handleInput);
            input.addEventListener('keydown', handleKeydown);
            input.addEventListener('blur', handleBlur);
            input.addEventListener('pointerdown', handlePointerDown);

            return {{
                setValue(value, {{ silent = false }} = {{}}) {{
                    input.value = value || '';
                    if (!silent && value && onSelect) {{
                        onSelect(value);
                    }}
                }},
                destroy() {{
                    closeList();
                    input.removeEventListener('focus', handleFocus);
                    input.removeEventListener('input', handleInput);
                    input.removeEventListener('keydown', handleKeydown);
                    input.removeEventListener('blur', handleBlur);
                    input.removeEventListener('pointerdown', handlePointerDown);
                }},
            }};
        }}

        function createListItems(containerId, items, labelKey = 'nationality') {{
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            items.forEach(item => {{
                const li = document.createElement('li');
                const avgText = item.avg_duration_days !== null && item.avg_duration_days !== undefined
                    ? 'avg ' + formatNumber(item.avg_duration_days) + ' days'
                    : null;
                const medianText = item.median_duration_days !== null && item.median_duration_days !== undefined
                    ? 'median ' + formatNumber(item.median_duration_days) + ' days'
                    : null;
                const durationParts = [avgText, medianText].filter(Boolean);
                const durationText = durationParts.length ? ', ' + durationParts.join(' · ') : '';
                li.innerHTML = '<strong>' + item[labelKey] + '</strong> — ' + formatNumber(item.contracts_count) +
                    ' contracts, termination ' + formatPercent(item.termination_rate, 1) + durationText;
                container.appendChild(li);
            }});
        }}

        function setupTable({{ tableId, data, columns, searchInputId }}) {{
            const table = document.getElementById(tableId);
            const thead = table.querySelector('thead');
            const tbody = table.querySelector('tbody');
            const searchInput = searchInputId ? document.getElementById(searchInputId) : null;

            let currentData = [...data];
            let sortConfig = {{ key: null, direction: 'asc' }};

            function renderHeader() {{
                const headerRow = document.createElement('tr');
                columns.forEach(col => {{
                    const th = document.createElement('th');
                    th.textContent = col.label;
                    th.style.cursor = col.sortable === false ? 'default' : 'pointer';
                    if (col.tooltip) {{
                        th.title = col.tooltip;
                    }}
                    if (col.sortable !== false) {{
                        th.addEventListener('click', () => {{
                            if (sortConfig.key === col.key) {{
                                sortConfig.direction = sortConfig.direction === 'asc' ? 'desc' : 'asc';
                            }} else {{
                                sortConfig = {{ key: col.key, direction: 'asc' }};
                            }}
                            sortData();
                            renderBody();
                        }});
                    }}
                    headerRow.appendChild(th);
                }});
                thead.innerHTML = '';
                thead.appendChild(headerRow);
            }}

            function renderBody() {{
                tbody.innerHTML = '';
                currentData.forEach(row => {{
                    const tr = document.createElement('tr');
                    columns.forEach(col => {{
                        const td = document.createElement('td');
                        const value = row[col.key];
                        td.innerHTML = col.format ? col.format(value, row) : value;
                        tr.appendChild(td);
                    }});
                    tbody.appendChild(tr);
                }});
            }}

            function sortData() {{
                if (!sortConfig.key) return;
                const {{ key, direction }} = sortConfig;
                currentData.sort((a, b) => {{
                    const valA = a[key];
                    const valB = b[key];
                    if (valA === null || valA === undefined) return 1;
                    if (valB === null || valB === undefined) return -1;
                    if (typeof valA === 'number' && typeof valB === 'number') {{
                        return direction === 'asc' ? valA - valB : valB - valA;
                    }}
                    return direction === 'asc'
                        ? String(valA).localeCompare(String(valB))
                        : String(valB).localeCompare(String(valA));
                }});
            }}

            function filterData(term) {{
                const lower = term.trim().toLowerCase();
                if (!lower) {{
                    currentData = [...data];
                }} else {{
                    currentData = data.filter(row => {{
                        return columns.some(col => {{
                            const cell = row[col.key];
                            return cell !== null && cell !== undefined && String(cell).toLowerCase().includes(lower);
                        }});
                    }});
                }}
                sortData();
                renderBody();
            }}

            renderHeader();
            sortData();
            renderBody();

            if (searchInput) {{
                searchInput.addEventListener('input', (event) => {{
                    filterData(event.target.value);
                }});
            }}
        }}

        function colorForRate(rate) {{
            if (rate === null || rate === undefined) return '#CBD5E1';
            const clamp = Math.max(0, Math.min(rate, 1));
            const start = [45, 147, 108];   // fresh green
            const mid = [244, 211, 94];     // warm amber
            const end = [209, 73, 91];      // rich red
            let from, to, t;
            if (clamp <= 0.5) {{
                from = start;
                to = mid;
                t = clamp / 0.5;
            }} else {{
                from = mid;
                to = end;
                t = (clamp - 0.5) / 0.5;
            }}
            const r = Math.round(from[0] + (to[0] - from[0]) * t);
            const g = Math.round(from[1] + (to[1] - from[1]) * t);
            const b = Math.round(from[2] + (to[2] - from[2]) * t);
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        function renderKpis() {{
            const grid = document.getElementById('kpiGrid');
            const kpisData = currentDataset.overallKpis || {{}};
            const kpis = [
                {{ label: 'Total Contracts', value: formatNumber(kpisData.total_contracts) }},
                {{ label: 'Unique Clients', value: formatNumber(kpisData.unique_clients) }},
                {{ label: 'Active Contracts', value: formatNumber(kpisData.active_contracts) }},
                {{ label: 'Terminated Contracts', value: formatNumber(kpisData.terminated_contracts) }},
                {{ label: 'Termination Rate', value: formatPercent(kpisData.termination_rate, 1) }},
                {{
                    label: 'Date Coverage',
                    value: `${{kpisData.contract_date_min || '—'}} → ${{kpisData.contract_date_max || '—'}}`
                }},
                {{
                    label: 'Contract Type',
                    value: currentContractType || '—'
                }},
            ];
            grid.innerHTML = '';
            kpis.forEach(kpi => {{
                const card = document.createElement('div');
                card.className = 'kpi-card';
                card.innerHTML = `<h3>${{kpi.label}}</h3><p>${{kpi.value}}</p>`;
                grid.appendChild(card);
            }});
        }}

        function renderNationalityCharts() {{
            const stats = currentDataset.nationalityStats || [];
            const data = stats.slice(0, 20);
            const rates = stats.map(d => d.termination_rate || 0);
            const maxRate = rates.length ? Math.max(...rates) : 0;
            Plotly.newPlot('nationalityBarChart', [
                {{
                    type: 'bar',
                    x: data.map(d => d.nationality),
                    y: data.map(d => d.contracts_count),
                    marker: {{
                        color: data.map(d => colorForRate(maxRate ? (d.termination_rate || 0) / maxRate : 0)),
                    }},
                    hovertemplate: '<b>%{{x}}</b><br>Contracts: %{{y:,}}<br>Termination: %{{customdata[0]:.1%}}<br>Avg duration: %{{customdata[1]}} days<br>Median duration: %{{customdata[2]}} days<extra></extra>',
                    customdata: data.map(d => [
                        d.termination_rate || 0,
                        d.avg_duration_days !== null && d.avg_duration_days !== undefined ? formatNumber(d.avg_duration_days) : '—',
                        d.median_duration_days !== null && d.median_duration_days !== undefined ? formatNumber(d.median_duration_days) : '—',
                    ]),
                }}
            ], {{
                margin: {{ t: 30, r: 20, b: 120, l: 60 }},
                yaxis: {{ title: 'Contracts' }},
                xaxis: {{ automargin: true }},
                showlegend: false,
            }}, {{ responsive: true }});
        }}

        function renderKioskChart() {{
            const stats = currentDataset.kioskStats || [];
            const data = stats.slice(0, 15);
            const rates = stats.map(d => d.termination_rate || 0);
            const maxRate = rates.length ? Math.max(...rates) : 0;
            Plotly.newPlot('kioskBarChart', [
                {{
                    type: 'bar',
                    x: data.map(d => d.kiosk),
                    y: data.map(d => d.contracts_count),
                    marker: {{
                        color: data.map(d => colorForRate(maxRate ? (d.termination_rate || 0) / maxRate : 0)),
                    }},
                    hovertemplate: '<b>%{{x}}</b><br>Contracts: %{{y:,}}<br>Termination: %{{customdata[0]:.1%}}<br>Avg duration: %{{customdata[1]}} days<br>Median duration: %{{customdata[2]}} days<extra></extra>',
                    customdata: data.map(d => [
                        d.termination_rate || 0,
                        d.avg_duration_days !== null && d.avg_duration_days !== undefined ? formatNumber(d.avg_duration_days) : '—',
                        d.median_duration_days !== null && d.median_duration_days !== undefined ? formatNumber(d.median_duration_days) : '—',
                    ]),
                }}
            ], {{
                margin: {{ t: 30, r: 20, b: 140, l: 60 }},
                yaxis: {{ title: 'Contracts' }},
                xaxis: {{ automargin: true }},
                showlegend: false,
            }}, {{ responsive: true }});
        }}

        function renderHeatmap(mode) {{
            const effectiveMode = mode || currentHeatmapMode || 'rate';
            currentHeatmapMode = effectiveMode;
            const heatmapData = currentDataset.heatmapData || {{ kiosks: [], nationalities: [], terminationRateMatrix: [], contractsMatrix: [], tooltipMatrix: [] }};
            const z = effectiveMode === 'rate' ? heatmapData.terminationRateMatrix : heatmapData.contractsMatrix;
            const rateColorscale = [
                [0, '#2d936c'],
                [0.5, '#f4d35e'],
                [1, '#d1495b']
            ];
            const volumeColorscale = [
                [0, '#edf2fb'],
                [0.5, '#80a1d4'],
                [1, '#1d3557']
            ];
            const colorscale = effectiveMode === 'rate' ? rateColorscale : volumeColorscale;
            const colorbar = {{
                title: effectiveMode === 'rate' ? 'Termination Rate' : 'Contracts',
            }};
            if (effectiveMode === 'rate') {{
                colorbar.tickformat = '.0%';
            }}
            Plotly.newPlot('heatmap', [
                {{
                    type: 'heatmap',
                    x: heatmapData.nationalities,
                    y: heatmapData.kiosks,
                    z: z,
                    text: heatmapData.tooltipMatrix,
                    hovertemplate: '%{text}<extra></extra>',
                    colorscale: colorscale,
                    colorbar: colorbar,
                    zsmooth: false,
                }}
            ], {{
                margin: {{ t: 20, r: 20, b: 140, l: 140 }},
                xaxis: {{ tickangle: -45 }},
                yaxis: {{ automargin: true }},
            }}, {{ responsive: true }});

            const rateBtn = document.getElementById('heatmapRateBtn');
            const volBtn = document.getElementById('heatmapVolumeBtn');
            if (rateBtn && volBtn) {{
                rateBtn.classList.toggle('active', effectiveMode === 'rate');
                volBtn.classList.toggle('active', effectiveMode === 'volume');
            }}
        }}

        function setupHeatmapControls() {{
            const rateBtn = document.getElementById('heatmapRateBtn');
            const volBtn = document.getElementById('heatmapVolumeBtn');
            if (rateBtn) {{
                rateBtn.addEventListener('click', () => {{
                    renderHeatmap('rate');
                }});
            }}
            if (volBtn) {{
                volBtn.addEventListener('click', () => {{
                    renderHeatmap('volume');
                }});
            }}
        }}

        function renderNationalityDrilldown() {{
            const input = document.getElementById('nationalityCombo');
            const listElement = document.getElementById('nationalityComboList');
            const drilldown = currentDataset.nationalityDrilldown || {{}};
            const options = Object.keys(drilldown).sort();

            if (!input || !listElement) {{
                return;
            }}

            if (input._dropdownDestroy) {{
                input._dropdownDestroy();
                input._dropdownDestroy = null;
            }}

            function renderEmptyCharts() {{
                Plotly.newPlot('nationalityDrilldownCounts', [], {{
                    title: 'Contracts by Kiosk',
                    margin: {{ t: 60, r: 20, b: 80, l: 60 }},
                    annotations: [
                        {{
                            text: 'No data available for this contract type',
                            x: 0.5,
                            y: 0.5,
                            showarrow: false,
                            xref: 'paper',
                            yref: 'paper',
                            font: {{ color: '#52606d', size: 14 }}
                        }}
                    ],
                }}, {{ responsive: true }});
                Plotly.newPlot('nationalityDrilldownRates', [], {{
                    title: 'Termination Rate by Kiosk',
                    margin: {{ t: 60, r: 20, b: 80, l: 60 }},
                    annotations: [
                        {{
                            text: 'No data available for this contract type',
                            x: 0.5,
                            y: 0.5,
                            showarrow: false,
                            xref: 'paper',
                            yref: 'paper',
                            font: {{ color: '#52606d', size: 14 }}
                        }}
                    ],
                }}, {{ responsive: true }});
            }}

            if (!options.length) {{
                input.value = '';
                input.disabled = true;
                input.placeholder = 'No nationalities available';
                listElement.classList.remove('open');
                listElement.innerHTML = '';
                renderEmptyCharts();
                return;
            }}

            input.disabled = false;
            input.placeholder = 'Search or pick nationality...';
            listElement.classList.remove('open');
            listElement.innerHTML = '';
            let currentValue = null;

            function renderCharts(value) {{
                if (!value || !drilldown[value]) {{
                    renderEmptyCharts();
                    return;
                }}
                const rows = drilldown[value] || [];
                const topRows = rows.slice(0, 15);

                Plotly.newPlot('nationalityDrilldownCounts', [
                    {{
                        type: 'bar',
                        x: topRows.map(d => d.kiosk),
                        y: topRows.map(d => d.contracts_count),
                        marker: {{ color: '#2a6f97' }},
                        hovertemplate: '<b>%{{x}}</b><br>Contracts: %{{y:,}}<extra></extra>',
                    }}
                ], {{
                    title: `Contracts by Kiosk — ${{value}}`,
                    margin: {{ t: 60, r: 20, b: 160, l: 60 }},
                    xaxis: {{ automargin: true }},
                    showlegend: false,
                }}, {{ responsive: true }});

                Plotly.newPlot('nationalityDrilldownRates', [
                    {{
                        type: 'bar',
                        x: topRows.map(d => d.kiosk),
                        y: topRows.map(d => d.termination_rate),
                        marker: {{ color: topRows.map(d => colorForRate(d.termination_rate || 0)) }},
                        customdata: topRows.map(d => [
                            d.avg_duration_days !== null && d.avg_duration_days !== undefined ? formatNumber(d.avg_duration_days) : '—',
                            d.median_duration_days !== null && d.median_duration_days !== undefined ? formatNumber(d.median_duration_days) : '—',
                        ]),
                        hovertemplate: '<b>%{{x}}</b><br>Termination: %{{y:.1%}}<br>Avg duration: %{{customdata[0]}} days<br>Median duration: %{{customdata[1]}} days<extra></extra>',
                    }}
                ], {{
                    title: `Termination Rate by Kiosk — ${{value}}`,
                    margin: {{ t: 60, r: 20, b: 160, l: 60 }},
                    xaxis: {{ automargin: true }},
                    yaxis: {{ tickformat: '.0%' }},
                    showlegend: false,
                }}, {{ responsive: true }});
            }}

            const dropdown = createSearchDropdown({{
                input,
                listElement,
                options,
                onSelect: value => {{
                    currentValue = value;
                    renderCharts(value);
                }},
            }});

            input._dropdownDestroy = dropdown.destroy;

            const initialNormalized = (input.value || '').trim().toLowerCase();
            currentValue = options.find(opt => opt.toLowerCase() === initialNormalized) || options[0];

            dropdown.setValue(currentValue, {{ silent: true }});
            renderCharts(currentValue);
        }}

        function renderKioskDrilldown() {{
            const input = document.getElementById('kioskCombo');
            const listElement = document.getElementById('kioskComboList');
            const drilldown = currentDataset.kioskDrilldown || {{}};
            const timeseriesConfig = currentDataset.kioskTopNationalityTimeseries || {{ months: [], series: {{}} }};
            const tsMonths = timeseriesConfig.months || [];
            const linePalette = ['#2a6f97', '#d1495b', '#f4d35e', '#45a29e', '#6a4c93'];
            const modeButtons = Array.from(document.querySelectorAll('[data-timeseries-mode]'));
            const modeLabelMap = {{
                all: 'All Contracts',
                Active: 'Active Contracts',
                Terminated: 'Terminated Contracts',
            }};

            if (!input || !listElement) {{
                return;
            }}

            const options = Object.keys(drilldown).sort();
            listElement.classList.remove('open');
            listElement.innerHTML = '';

            if (input._dropdownDestroy) {{
                input._dropdownDestroy();
                input._dropdownDestroy = null;
            }}

            function applyModeStyles() {{
                modeButtons.forEach(btn => {{
                    btn.classList.toggle('active', btn.dataset.timeseriesMode === currentTimeseriesMode);
                }});
            }}

            let currentSelection = null;

            modeButtons.forEach(btn => {{
                btn.onclick = () => {{
                    const nextMode = btn.dataset.timeseriesMode || 'all';
                    currentTimeseriesMode = nextMode;
                    applyModeStyles();
                    renderTimeseries(currentSelection);
                }};
            }});
            applyModeStyles();

            function renderEmptyBar(chartId, title) {{
                Plotly.newPlot(chartId, [], {{
                    title,
                    margin: {{ t: 60, r: 20, b: 80, l: 60 }},
                    annotations: [
                        {{
                            text: 'No data available for this selection',
                            x: 0.5,
                            y: 0.5,
                            showarrow: false,
                            xref: 'paper',
                            yref: 'paper',
                            font: {{ color: '#52606d', size: 14 }}
                        }}
                    ],
                }}, {{ responsive: true }});
            }}

            function renderTimeseries(kioskValue) {{
                const chartId = 'kioskDrilldownTimeseries';
                const months = tsMonths;
                if (!kioskValue || !months.length) {{
                    Plotly.newPlot(chartId, [], {{
                        title: 'Monthly Contracts — No data available',
                        margin: {{ t: 60, r: 20, b: 80, l: 60 }},
                        xaxis: {{ title: 'Month', tickangle: -45, automargin: true }},
                        yaxis: {{ title: 'Contracts' }},
                        annotations: [
                            {{
                                text: 'No contract activity available for this selection',
                                showarrow: false,
                                xref: 'paper',
                                yref: 'paper',
                                x: 0.5,
                                y: 0.5,
                                font: {{ color: '#52606d', size: 14 }}
                            }}
                        ],
                    }}, {{ responsive: true }});
                    return;
                }}

                const kioskSeries = timeseriesConfig.series && timeseriesConfig.series[kioskValue];
                if (!kioskSeries || !kioskSeries.nationalities || !kioskSeries.nationalities.length) {{
                    Plotly.newPlot(chartId, [], {{
                        title: `Monthly Contracts — ${{kioskValue}}`,
                        margin: {{ t: 60, r: 20, b: 80, l: 60 }},
                        xaxis: {{ title: 'Month', tickangle: -45, automargin: true }},
                        yaxis: {{ title: 'Contracts' }},
                        annotations: [
                            {{
                                text: 'No contract activity available for this selection',
                                showarrow: false,
                                xref: 'paper',
                                yref: 'paper',
                                x: 0.5,
                                y: 0.5,
                                font: {{ color: '#52606d', size: 14 }}
                            }}
                        ],
                    }}, {{ responsive: true }});
                    return;
                }}

                const nationalities = kioskSeries.nationalities;
                const modeSeries = kioskSeries[currentTimeseriesMode] || {{}};
                const traces = nationalities.map((nationality, idx) => {{
                    const counts = modeSeries[nationality] || months.map(() => 0);
                    return {{
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: nationality,
                        x: months,
                        y: counts,
                        line: {{ color: linePalette[idx % linePalette.length], width: 3 }},
                        marker: {{ size: 6 }},
                        hovertemplate: `<b>${{nationality}}</b><br>Month: %{{x}}<br>Contracts: %{{y:,}}<extra></extra>`,
                    }};
                }});

                const hasData = traces.some(trace => (trace.y || []).some(value => value > 0));
                const layout = {{
                    title: `Monthly Contracts — ${{kioskValue}} (${{modeLabelMap[currentTimeseriesMode] || 'All Contracts'}})`,
                    margin: {{ t: 60, r: 20, b: 80, l: 60 }},
                    xaxis: {{ title: 'Month', tickangle: -45, automargin: true }},
                    yaxis: {{ title: 'Contracts' }},
                    legend: {{ orientation: 'h', y: -0.25 }},
                }};

                if (!hasData) {{
                    layout.annotations = [
                        {{
                            text: 'No contract activity for the selected filters',
                            showarrow: false,
                            xref: 'paper',
                            yref: 'paper',
                            x: 0.5,
                            y: 0.5,
                            font: {{ color: '#52606d', size: 14 }}
                        }}
                    ];
                }}

                Plotly.newPlot(chartId, traces, layout, {{ responsive: true }});
            }}

        function renderCharts(value) {{
            if (!value || !drilldown[value] || !drilldown[value].length) {{
                renderEmptyBar('kioskDrilldownCounts', 'Nationality Mix — No data available');
                renderEmptyBar('kioskDrilldownRates', 'Termination Rate by Nationality — No data available');
                return;
            }}

            const rows = drilldown[value] || [];
            const topRows = rows.slice(0, 15);

            Plotly.newPlot('kioskDrilldownCounts', [
                {{
                    type: 'bar',
                    x: topRows.map(d => d.nationality),
                    y: topRows.map(d => d.contracts_count),
                    marker: {{ color: '#195c70' }},
                    hovertemplate: '<b>%{{x}}</b><br>Contracts: %{{y:,}}<extra></extra>',
                }}
            ], {{
                title: `Nationality Mix — ${{value}}`,
                margin: {{ t: 60, r: 20, b: 160, l: 60 }},
                xaxis: {{ automargin: true }},
                showlegend: false,
            }}, {{ responsive: true }});

            Plotly.newPlot('kioskDrilldownRates', [
                {{
                    type: 'bar',
                    x: topRows.map(d => d.nationality),
                    y: topRows.map(d => d.termination_rate),
                    marker: {{ color: topRows.map(d => colorForRate(d.termination_rate || 0)) }},
                    customdata: topRows.map(d => [
                        d.avg_duration_days !== null && d.avg_duration_days !== undefined ? formatNumber(d.avg_duration_days) : '—',
                        d.median_duration_days !== null && d.median_duration_days !== undefined ? formatNumber(d.median_duration_days) : '—',
                    ]),
                    hovertemplate: '<b>%{{x}}</b><br>Termination: %{{y:.1%}}<br>Avg duration: %{{customdata[0]}} days<br>Median duration: %{{customdata[1]}} days<extra></extra>',
                }}
            ], {{
                title: `Termination Rate by Nationality — ${{value}}`,
                margin: {{ t: 60, r: 20, b: 160, l: 60 }},
                xaxis: {{ automargin: true }},
                yaxis: {{ tickformat: '.0%' }},
                showlegend: false,
            }}, {{ responsive: true }});
        }}

            if (!options.length) {{
                input.value = '';
                input.disabled = true;
                input.placeholder = 'No kiosks available';
                listElement.classList.remove('open');
                listElement.innerHTML = '';
                renderEmptyBar('kioskDrilldownCounts', 'Nationality Mix — No data available');
                renderEmptyBar('kioskDrilldownRates', 'Termination Rate by Nationality — No data available');
                renderTimeseries(null);
                return;
            }}

            input.disabled = false;
            input.placeholder = 'Search or pick kiosk...';

            const dropdown = createSearchDropdown({{
                input,
                listElement,
                options,
                onSelect: value => {{
                    currentSelection = value;
                    renderCharts(value);
                    renderTimeseries(value);
                }},
            }});

            input._dropdownDestroy = dropdown.destroy;

            const initialNormalized = (input.value || '').trim().toLowerCase();
            currentSelection = options.find(opt => opt.toLowerCase() === initialNormalized) || options[0];

            dropdown.setValue(currentSelection, {{ silent: true }});
            renderCharts(currentSelection);
            renderTimeseries(currentSelection);
        }}

        function renderRiskTable() {{
            setupTable({{
                tableId: 'riskTable',
                data: currentDataset.riskSegments || [],
                columns: [
                    {{ key: 'kiosk', label: 'Kiosk' }},
                    {{ key: 'nationality', label: 'Nationality' }},
                    {{ key: 'contracts_count', label: 'Contracts', format: v => formatNumber(v) }},
                    {{ key: 'terminated_count', label: 'Terminated', format: v => formatNumber(v) }},
                    {{
                        key: 'avg_duration_days',
                        label: 'Avg Duration (days)',
                        format: v => v === null || v === undefined ? '—' : formatNumber(v)
                    }},
                    {{
                        key: 'median_duration_days',
                        label: 'Median Duration (days)',
                        format: v => v === null || v === undefined ? '—' : formatNumber(v)
                    }},
                    {{
                        key: 'termination_rate',
                        label: 'Termination Rate',
                        format: v => formatPercent(v, 1)
                    }},
                ],
            }});
        }}

        function renderTables() {{
            setupTable({{
                tableId: 'nationalityTable',
                data: (currentDataset.nationalityStats || []).map(row => Object.assign({}, row, {{
                    contracts_share_pct: row.contracts_share,
                }})),
                columns: [
                    {{ key: 'nationality', label: 'Nationality' }},
                    {{ key: 'contracts_count', label: 'Contracts', format: v => formatNumber(v) }},
                    {{ key: 'contracts_share_pct', label: '% of Total Contracts', format: v => formatPercent(v, 1) }},
                    {{ key: 'active_count', label: 'Active', format: v => formatNumber(v) }},
                    {{ key: 'terminated_count', label: 'Terminated', format: v => formatNumber(v) }},
                    {{
                        key: 'avg_duration_days',
                        label: 'Avg Duration (days)',
                        format: v => v === null || v === undefined ? '—' : formatNumber(v),
                    }},
                    {{
                        key: 'median_duration_days',
                        label: 'Median Duration (days)',
                        format: v => v === null || v === undefined ? '—' : formatNumber(v),
                    }},
                    {{ key: 'termination_rate', label: 'Termination Rate', format: v => formatPercent(v, 1) }},
                ],
                searchInputId: 'nationalitySearch',
            }});

            setupTable({{
                tableId: 'kioskTable',
                data: currentDataset.kioskStats || [],
                columns: [
                    {{ key: 'kiosk', label: 'Kiosk' }},
                    {{ key: 'contracts_count', label: 'Contracts', format: v => formatNumber(v) }},
                    {{ key: 'active_count', label: 'Active', format: v => formatNumber(v) }},
                    {{ key: 'terminated_count', label: 'Terminated', format: v => formatNumber(v) }},
                    {{
                        key: 'avg_duration_days',
                        label: 'Avg Duration (days)',
                        format: v => v === null || v === undefined ? '—' : formatNumber(v),
                    }},
                    {{
                        key: 'median_duration_days',
                        label: 'Median Duration (days)',
                        format: v => v === null || v === undefined ? '—' : formatNumber(v),
                    }},
                    {{ key: 'termination_rate', label: 'Termination Rate', format: v => formatPercent(v, 1) }},
                ],
                searchInputId: 'kioskSearch',
            }});

            setupTable({{
                tableId: 'clientBucketTable',
                data: (currentDataset.clientMetrics && currentDataset.clientMetrics.bucket_summary) || [],
                columns: [
                    {{ key: 'bucket', label: 'Contracts Per Client' }},
                    {{ key: 'clients', label: 'Number of Clients', format: v => formatNumber(v) }},
                    {{
                        key: 'avg_client_termination_rate',
                        label: 'Avg Termination Rate',
                        format: v => formatPercent(v, 1)
                    }},
                ],
            }});

            setupTable({{
                tableId: 'missingTable',
                data: (currentDataset.dataQuality && currentDataset.dataQuality.missing_summary) || [],
                columns: [
                    {{ key: 'field', label: 'Field' }},
                    {{ key: 'missing_count', label: 'Missing Count', format: v => formatNumber(v) }},
                    {{ key: 'missing_pct', label: 'Missing %', format: v => formatPercent(v, 1) }},
                ],
            }});
        }}

        function renderLists() {{
            createListItems('lowChurnList', currentDataset.lowChurnNationalities || []);
            createListItems('highChurnList', currentDataset.highChurnNationalities || []);
            createListItems('bestKioskList', currentDataset.bestKiosks || [], 'kiosk');
            createListItems('worstKioskList', currentDataset.worstKiosks || [], 'kiosk');
        }}

        function renderClientSection() {{
            const metrics = currentDataset.clientMetrics || {{
                total_unique_clients: 0,
                clients_with_one_contract: 0,
                clients_with_multiple_contracts: 0,
                repeat_rate: 0,
                single_client_termination_rate: 0,
                multi_client_termination_rate: 0,
                distribution: [],
                bucket_summary: [],
            }};
            const kpiSummary = document.getElementById('clientKpiSummary');
            kpiSummary.innerHTML = `
                ${{formatNumber(metrics.total_unique_clients)}} unique clients — ${{formatPercent(metrics.repeat_rate, 1)}} are repeat purchasers.
                <br />${{formatNumber(metrics.clients_with_one_contract)}} have a single contract; ${{formatNumber(metrics.clients_with_multiple_contracts)}} hold multiple contracts.
            `;

            const tags = document.getElementById('clientTags');
            tags.innerHTML = `
                <span class="tag">Single-contract clients: ${{formatNumber(metrics.clients_with_one_contract)}}</span>
                <span class="tag">Repeat clients: ${{formatNumber(metrics.clients_with_multiple_contracts)}}</span>
            `;

            const termSummary = document.getElementById('clientTerminationSummary');
            termSummary.innerHTML = `
                Single-contract clients with a termination: ${{formatPercent(metrics.single_client_termination_rate, 1)}}<br/>
                Multi-contract clients with any termination: ${{formatPercent(metrics.multi_client_termination_rate, 1)}}
            `;

            Plotly.newPlot('clientDistributionChart', [
                {{
                    type: 'bar',
                    x: metrics.distribution.map(d => d.contracts),
                    y: metrics.distribution.map(d => d.clients),
                    marker: {{ color: '#2a6f97' }},
                    hovertemplate: 'Contracts: %{{x}}<br>Clients: %{{y:,}}<extra></extra>',
                }}
            ], {{
                margin: {{ t: 20, r: 20, b: 60, l: 60 }},
                xaxis: {{ title: 'Contracts per client' }},
                yaxis: {{ title: 'Number of clients' }},
                showlegend: false,
            }}, {{ responsive: true }});
        }}

        function setupTimeSeriesFilter() {{
            const input = document.getElementById('timeSeriesNationalityCombo');
            const listElement = document.getElementById('timeSeriesNationalityList');
            if (!input || !listElement) {{
                renderTimeSeries();
                return;
            }}

            if (timeSeriesFilterDropdown) {{
                timeSeriesFilterDropdown.destroy();
                timeSeriesFilterDropdown = null;
            }}

            const map = Object.assign({{}}, currentDataset.timeSeriesByNationality || {{}});
            if (!map['All nationalities']) {{
                map['All nationalities'] = currentDataset.timeSeries || [];
            }}

            const optionKeys = Object.keys(map);
            if (!optionKeys.length) {{
                input.value = '';
                input.disabled = true;
                listElement.classList.remove('open');
                listElement.innerHTML = '';
                currentTimeSeriesNationality = 'All nationalities';
                renderTimeSeries();
                return;
            }}

            const sortedOptions = optionKeys.filter(opt => opt !== 'All nationalities').sort();
            const finalOptions = ['All nationalities', ...sortedOptions];

            input.disabled = false;

            timeSeriesFilterDropdown = createSearchDropdown({{
                input,
                listElement,
                options: finalOptions,
                onSelect: value => {{
                    currentTimeSeriesNationality = value;
                    renderTimeSeries(value);
                }},
            }});

            const defaultSelection = finalOptions.includes(currentTimeSeriesNationality)
                ? currentTimeSeriesNationality
                : finalOptions[0];
            currentTimeSeriesNationality = defaultSelection;
            timeSeriesFilterDropdown.setValue(defaultSelection, {{ silent: true }});
            renderTimeSeries(defaultSelection);
        }}

        function renderTimeSeries(selectedNationality) {{
            const seriesMap = Object.assign({{}}, currentDataset.timeSeriesByNationality || {{}});
            if (!seriesMap['All nationalities']) {{
                seriesMap['All nationalities'] = currentDataset.timeSeries || [];
            }}
            const selectedKey = selectedNationality || currentTimeSeriesNationality || 'All nationalities';
            const series = seriesMap[selectedKey] || seriesMap['All nationalities'] || currentDataset.timeSeries || [];
            const titleEl = document.getElementById('timeSeriesTitle');
            const suffix = selectedKey && selectedKey !== 'All nationalities' ? ` — ${{selectedKey}}` : '';
            if (titleEl) {{
                titleEl.textContent = `Contracts vs Terminations Over Time${{suffix}}`;
            }}
            Plotly.newPlot('timeSeriesChart', [
                {{
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Contracts',
                    x: series.map(d => d.month),
                    y: series.map(d => d.contracts),
                    line: {{ color: '#2a6f97', width: 3 }},
                    marker: {{ size: 6 }},
                }},
                {{
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Terminations',
                    x: series.map(d => d.month),
                    y: series.map(d => d.terminations),
                    line: {{ color: '#d9534f', width: 3 }},
                    marker: {{ size: 6 }},
                }},
            ], {{
                margin: {{ t: 20, r: 20, b: 80, l: 60 }},
                xaxis: {{ title: 'Month', tickangle: -45, automargin: true }},
                yaxis: {{ title: 'Contracts / Terminations' }},
                legend: {{ orientation: 'h', y: -0.25 }},
            }}, {{ responsive: true }});
        }}

        function renderCityChart() {{
            const data = (currentDataset.cityStats || []).slice(0, 15);
            Plotly.newPlot('cityBarChart', [
                {{
                    type: 'bar',
                    x: data.map(d => d.city),
                    y: data.map(d => d.contracts_count),
                    marker: {{ color: data.map(d => colorForRate(d.termination_rate || 0)) }},
                    hovertemplate: '<b>%{x}</b><br>Contracts: %{y:,}<br>Termination: %{customdata[0]:.1%}<br>Avg duration: %{customdata[1]} days<br>Median duration: %{customdata[2]} days<extra></extra>',
                    customdata: data.map(d => [
                        d.termination_rate || 0,
                        d.avg_duration_days !== null && d.avg_duration_days !== undefined ? formatNumber(d.avg_duration_days) : '—',
                        d.median_duration_days !== null && d.median_duration_days !== undefined ? formatNumber(d.median_duration_days) : '—',
                    ]),
                }}
            ], {{
                margin: {{ t: 20, r: 20, b: 140, l: 60 }},
                yaxis: {{ title: 'Contracts' }},
                xaxis: {{ automargin: true }},
                showlegend: false,
            }}, {{ responsive: true }});
        }}

        function renderCancellationInsights() {{
            const data = currentDataset.cancellationInsights || {{}};
            setupTable({{
                tableId: 'cancelRateTable',
                data: data.rateSummary || [],
                columns: [
                    {{ key: 'cohort', label: 'Cohort' }},
                    {{ key: 'contracts', label: 'Contracts', format: v => formatNumber(v) }},
                    {{ key: 'terminations', label: 'Cancellations', format: v => formatNumber(v) }},
                    {{ key: 'cancellation_rate', label: 'Cancellation Rate', format: v => formatPercent(v, 1) }},
                ],
            }});

            const container = document.getElementById('cancelTimeTables');
            if (!container) {{ return; }}
            const rows = data.timeToCancel || [];
            const periodOrder = ['Pre-promoters', 'Post-promoters'];
            const cohorts = ['Retail (all)', 'Filipina', 'Other nationalities'];
            const types = ['CC', 'MV', 'All'];

            const map = new Map();
            rows.forEach(r => {{
                const key = `${{r.period}}|${{r.contract_type}}|${{r.cohort}}`;
                map.set(key, r);
            }});

            const fmt = v => formatPercent(v, 1);
            const cell = (period, cohort, type, key) => {{
                const row = map.get(`${{period}}|${{type}}|${{cohort}}`) || {{}};
                const val = row[key];
                return val === undefined || val === null ? '�??' : fmt(val);
            }};
            const cnt = (period, cohort, type) => {{
                const row = map.get(`${{period}}|${{type}}|${{cohort}}`) || {{}};
                const val = row.terminated;
                return val === undefined || val === null ? '�??' : formatNumber(val);
            }};

            const tablesHtml = periodOrder.map(period => {{
                const rowsHtml = cohorts.map(cohort => {{
                    const ccCount = cnt(period, cohort, 'CC');
                    const mvCount = cnt(period, cohort, 'MV');
                    const allCount = cnt(period, cohort, 'All');
                    return `
                        <tr>
                            <td>${{cohort}}</td>
                            <td>${{ccCount}}</td>
                            <td>${{cell(period, cohort, 'CC', 'pct_7d')}} | ${{cell(period, cohort, 'CC', 'pct_30d')}}</td>
                            <td>${{mvCount}}</td>
                            <td>${{cell(period, cohort, 'MV', 'pct_7d')}} | ${{cell(period, cohort, 'MV', 'pct_30d')}}</td>
                            <td>${{allCount}}</td>
                            <td>${{cell(period, cohort, 'All', 'pct_7d')}} | ${{cell(period, cohort, 'All', 'pct_30d')}}</td>
                        </tr>
                    `;
                }}).join('');
                return `
                    <div class="table-wrapper">
                        <h4>${{period}}</h4>
                        <table class="data-table cancel-compact">
                            <thead>
                                <tr>
                                    <th>Cohort</th>
                                    <th>CC cancels</th>
                                    <th>CC % <=7d | % <=30d</th>
                                    <th>MV cancels</th>
                                    <th>MV % <=7d | % <=30d</th>
                                    <th>All cancels</th>
                                    <th>All % <=7d | % <=30d</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${{rowsHtml}}
                            </tbody>
                        </table>
                    </div>
                `;
            }}).join('');

            container.innerHTML = tablesHtml;
        }}

        function renderDataQuality() {{
            const assumptionList = document.getElementById('assumptionList');
            assumptionList.innerHTML = '';
            (currentDataset.dataQuality && currentDataset.dataQuality.notes || []).forEach(note => {{
                const li = document.createElement('li');
                li.textContent = note;
                assumptionList.appendChild(li);
            }});

            const notesList = document.getElementById('dataQualityNotes');
            notesList.innerHTML = '';
            const invalidCities = (currentDataset.dataQuality && currentDataset.dataQuality.invalid_city_values) || [];
            const negativeDuration = currentDataset.dataQuality ? currentDataset.dataQuality.negative_duration_count : 0;
            const paymentNeg = currentDataset.dataQuality ? currentDataset.dataQuality.payment_lag_negative_count : 0;
            const sanity = currentDataset.sanityChecks || {{ nationality_total: 0, kiosk_total: 0, overall_total: 0 }};
            const invalidCitySummary = invalidCities.length
                ? invalidCities.map(item => item.value + ' (' + formatNumber(item.count) + ')').join(', ')
                : 'None observed.';
            const notesHtml = '<li>Invalid numeric city values recoded to Unknown: ' + invalidCitySummary + '.</li>' +
                '<li>Contracts with negative duration identified and set to null: ' + formatNumber(negativeDuration) + '.</li>' +
                '<li>Contracts paid before creation (likely data errors): ' + formatNumber(paymentNeg) + '.</li>' +
                '<li>Sanity check — totals by nationality: ' + formatNumber(sanity.nationality_total) + ', by kiosk: ' + formatNumber(sanity.kiosk_total) + ', overall: ' + formatNumber(sanity.overall_total) + '.</li>';
            notesList.innerHTML = notesHtml;
        }}

        function renderAll() {{
            renderKpis();
            renderNationalityCharts();
            renderKioskChart();
            renderHeatmap();
            renderNationalityDrilldown();
            renderKioskDrilldown();
            renderRiskTable();
            renderTables();
            renderLists();
            renderClientSection();
            setupTimeSeriesFilter();
            renderCityChart();
            renderCancellationInsights();
            renderDataQuality();
        }}

        function setContractType(type) {{
            if (datasetsByType[type]) {{
                currentContractType = type;
            }} else if (contractTypes.length) {{
                currentContractType = contractTypes[0];
            }} else {{
                currentContractType = '';
            }}
            currentDataset = datasetsByType[currentContractType] || {{}};
            const select = document.getElementById('contractTypeSelect');
            if (select && select.value !== currentContractType) {{
                select.value = currentContractType;
            }}
            currentTimeSeriesNationality = 'All nationalities';
            if (timeSeriesFilterDropdown) {{
                timeSeriesFilterDropdown.destroy();
                timeSeriesFilterDropdown = null;
            }}
            renderAll();
        }}

        function populateContractTypeSelect() {{
            const select = document.getElementById('contractTypeSelect');
            if (!select) {{
                return;
            }}
            select.innerHTML = '';
            contractTypes.forEach(opt => {{
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt;
                select.appendChild(option);
            }});
            if (!contractTypes.length) {{
                select.disabled = true;
                return;
            }}
            select.disabled = false;
            if (!datasetsByType[currentContractType]) {{
                currentContractType = contractTypes[0];
                currentDataset = datasetsByType[currentContractType] || {{}};
            }}
            select.value = currentContractType;
            select.onchange = event => {{
                setContractType(event.target.value);
            }};
        }}

        function init() {{
            populateContractTypeSelect();
            setupHeatmapControls();
            renderAll();
        }}

        document.addEventListener('DOMContentLoaded', () => {{
            init();
        }});
    </script>
</body>
</html>"""
    template = template.replace("{{", "{").replace("}}", "}")
    html = template.replace("__DATA_BUNDLE__", safe_json(data_bundle))
    return html


def main():
    data_bundle = prepare_data()
    html = build_html(data_bundle)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Report written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
