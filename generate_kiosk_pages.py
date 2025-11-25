#!/usr/bin/env python3
"""
Regenerate kiosk-level HTML pages so their metrics stay in sync with the latest dataset.

The script pulls from the CSV exports produced by the analysis pipeline:
- kiosk_nationality_latest_distribution.csv   → latest month snapshot per kiosk
- kiosk_nationality_recent_summary.csv        → trailing-quarter growth & churn stats
- kiosk_nationality_time_series.csv           → monthly active counts for time-series cards

For each kiosk we rebuild an HTML file that mirrors the existing layout, only
the numbers/tables get refreshed. Navigation links between kiosks are also
regenerated so we don't have to maintain them by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
LATEST_PATH = BASE_DIR / "kiosk_nationality_latest_distribution.csv"
RECENT_PATH = BASE_DIR / "kiosk_nationality_recent_summary.csv"
TIMESERIES_PATH = BASE_DIR / "kiosk_nationality_time_series.csv"

UNKNOWN_LABELS = {"unknown", "unknown nationality", "nan"}


@dataclass(frozen=True)
class KioskData:
    name: str
    slug: str
    latest_month: pd.Timestamp
    latest_rows: pd.DataFrame
    recent_rows: pd.DataFrame
    monthly_totals: pd.DataFrame


def load_source_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not LATEST_PATH.exists():
        raise FileNotFoundError(f"Missing latest distribution file: {LATEST_PATH}")
    if not RECENT_PATH.exists():
        raise FileNotFoundError(f"Missing recent summary file: {RECENT_PATH}")
    if not TIMESERIES_PATH.exists():
        raise FileNotFoundError(f"Missing time-series file: {TIMESERIES_PATH}")

    latest_df = pd.read_csv(LATEST_PATH, parse_dates=["Month"])
    recent_df = pd.read_csv(RECENT_PATH)
    ts_df = pd.read_csv(TIMESERIES_PATH, parse_dates=["Month"])
    return latest_df, recent_df, ts_df


def slugify(value: str) -> str:
    slug = value.strip().lower().replace("&", "and")
    out = []
    prev_dash = False
    for ch in slug:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    result = "".join(out).strip("-")
    if not result:
        raise ValueError(f"Cannot slugify value: {value!r}")
    return result


def format_number(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    if decimals == 0:
        return f"{int(round(float(value))):,}"
    return f"{float(value):,.{decimals}f}"


def format_percent(value, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def normalize_nat(value: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    value = str(value).strip()
    return value if value else "Unknown"


def build_navigation(kiosk_names: List[str]) -> str:
    links = ["<li><a href='index.html'>← Back to Executive Report</a></li>"]
    for name in kiosk_names:
        links.append(f"<li><a href='{slugify(name)}.html'>{name}</a></li>")
    return "\n".join(links)


def build_kiosk_payloads(
    latest_df: pd.DataFrame, recent_df: pd.DataFrame, ts_df: pd.DataFrame
) -> Dict[str, KioskData]:
    kiosk_names = sorted(latest_df["Kiosk"].unique())
    payloads: Dict[str, KioskData] = {}

    kiosk_ts_totals = (
        ts_df.groupby(["Kiosk", "Month"])[["active_contracts", "new_contracts"]]
        .sum()
        .reset_index()
        .sort_values("Month")
    )

    for kiosk in kiosk_names:
        kiosk_latest = latest_df[latest_df["Kiosk"] == kiosk].copy()
        if kiosk_latest.empty:
            continue
        kiosk_recent = recent_df[recent_df["Kiosk"] == kiosk].copy()
        kiosk_monthly = kiosk_ts_totals[kiosk_ts_totals["Kiosk"] == kiosk].copy()
        latest_month = kiosk_latest["Month"].max()
        payloads[kiosk] = KioskData(
            name=kiosk,
            slug=slugify(kiosk),
            latest_month=latest_month,
            latest_rows=kiosk_latest.sort_values(
                ["active_contracts", "Client Nationality"], ascending=[False, True]
            ),
            recent_rows=kiosk_recent.sort_values(
                ["net_change_total", "avg_churn_rate"], ascending=[False, False]
            ),
            monthly_totals=kiosk_monthly,
        )
    return payloads


def render_highlights(
    data: KioskData,
    total_active: int,
    prev_active: int | None,
    prev_month_label: str | None,
) -> str:
    rows = data.latest_rows
    statements: List[str] = []

    if not rows.empty:
        leader = rows.iloc[0]
        lead_name = leader["Client Nationality"]
        lead_share = leader.get("share_of_kiosk") or leader.get("share_pct")
        statements.append(
            f"{lead_name} is the leading cohort with {int(leader['active_contracts'])} active contracts "
            f"({format_percent(lead_share, 1)})."
        )

    unknown_mask = rows["Client Nationality"].str.lower().isin(UNKNOWN_LABELS)
    unknown_active = int(rows.loc[unknown_mask, "active_contracts"].sum())
    if unknown_active:
        share = unknown_active / total_active if total_active else 0
        statements.append(
            f"{unknown_active} contracts ({format_percent(share, 1)}) still have nationality marked as Unknown."
        )
    else:
        statements.append("All active contracts have a known nationality.")

    latest_month = data.latest_month.strftime("%Y-%m")
    if prev_active is None:
        statements.append("Historical comparison unavailable for the prior month.")
    else:
        prior_label = prev_month_label or "the prior month"
        if prev_active == total_active:
            change_text = f"were flat versus {prior_label}"
        elif total_active > prev_active:
            diff = total_active - prev_active
            change_text = f"grew by {diff} versus {prior_label}"
        else:
            diff = prev_active - total_active
            change_text = f"declined by {diff} versus {prior_label}"
        statements.append(
            f"Active contracts {change_text}."
        )

    growth_row = data.recent_rows.sort_values("net_change_total", ascending=False).head(1)
    if not growth_row.empty:
        gr = growth_row.iloc[0]
        statements.append(
            f"Largest net growth in the last quarter: {format_number(gr['net_change_total'], 1)} net {gr['Client Nationality']} contracts."
        )

    churn_row = (
        data.recent_rows.sort_values("avg_churn_rate", ascending=False)
        .replace({pd.NA: None})
        .head(1)
    )
    if not churn_row.empty and not pd.isna(churn_row.iloc[0]["avg_churn_rate"]):
        cr = churn_row.iloc[0]
        statements.append(
            f"Highest recent churn: {cr['Client Nationality']} at {format_percent(cr['avg_churn_rate'], 1)} (avg) with {format_number(cr['latest_active'])} active contracts."
        )

    items = "".join(f"<li>{text}</li>" for text in statements)
    return f"<ul>{items}</ul>"


def render_latest_table(rows: pd.DataFrame, headline_month: str) -> str:
    cols = [
        ("Client Nationality", "Client Nationality"),
        ("active_contracts", "Active"),
        ("share_pct", "Share"),
        ("new_contracts", "New"),
        ("terminations", "Terminations"),
        ("churn_rate", "Churn"),
        ("retention_rate", "Retention"),
    ]
    header = "".join(f"<th>{label}</th>" for _, label in cols)
    body = []
    for _, row in rows.iterrows():
        share = format_percent(row.get("share_of_kiosk") or row.get("share_pct"), 1)
        churn = format_percent(row.get("churn_rate"), 1)
        retention = format_percent(row.get("retention_rate"), 1)
        body.append(
            "<tr>"
            f"<td>{row['Client Nationality']}</td>"
            f"<td>{format_number(row['active_contracts'])}</td>"
            f"<td>{share}</td>"
            f"<td>{format_number(row['new_contracts'])}</td>"
            f"<td>{format_number(row['terminations'])}</td>"
            f"<td>{churn}</td>"
            f"<td>{retention}</td>"
            "</tr>"
        )

    return (
        f"<h3>Top nationalities ({headline_month})</h3>"
        "<table class='dataframe data-table'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
        "<p class='note'>Includes top 10 nationalities ranked by active contracts for the latest complete month.</p>"
    )


def render_recent_tables(recent_rows: pd.DataFrame) -> Tuple[str, str]:
    cols_growth = [
        ("Client Nationality", "Client Nationality"),
        ("net_change_total", "Net Change"),
        ("mean_active", "Avg Active"),
        ("avg_retention_rate", "Avg Retention"),
        ("latest_active", "Latest Active"),
    ]
    growth_body = []
    for _, row in recent_rows.sort_values("net_change_total", ascending=False).head(5).iterrows():
        growth_body.append(
            "<tr>"
            f"<td>{row['Client Nationality']}</td>"
            f"<td>{format_number(row['net_change_total'], 1)}</td>"
            f"<td>{format_number(row['mean_active'], 1)}</td>"
            f"<td>{format_percent(row.get('avg_retention_rate') or row.get('avg_retention_pct'), 1)}</td>"
            f"<td>{format_number(row['latest_active'])}</td>"
            "</tr>"
        )

    cols_churn = [
        ("Client Nationality", "Client Nationality"),
        ("avg_churn_rate", "Avg Churn"),
        ("avg_retention_rate", "Avg Retention"),
        ("latest_active", "Latest Active"),
        ("net_change_total", "Net Change"),
    ]
    churn_body = []
    for _, row in recent_rows.sort_values("avg_churn_rate", ascending=False).head(5).iterrows():
        churn_body.append(
            "<tr>"
            f"<td>{row['Client Nationality']}</td>"
            f"<td>{format_percent(row['avg_churn_rate'], 1)}</td>"
            f"<td>{format_percent(row.get('avg_retention_rate') or row.get('avg_retention_pct'), 1)}</td>"
            f"<td>{format_number(row['latest_active'])}</td>"
            f"<td>{format_number(row['net_change_total'], 1)}</td>"
            "</tr>"
        )

    growth_table = (
        "<table class='dataframe data-table'>"
        "<thead><tr>"
        + "".join(f"<th>{label}</th>" for _, label in cols_growth)
        + "</tr></thead>"
        f"<tbody>{''.join(growth_body)}</tbody>"
        "</table>"
    )
    churn_table = (
        "<table class='dataframe data-table'>"
        "<thead><tr>"
        + "".join(f"<th>{label}</th>" for _, label in cols_churn)
        + "</tr></thead>"
        f"<tbody>{''.join(churn_body)}</tbody>"
        "</table>"
    )
    return growth_table, churn_table


def render_trailing_table(monthly_totals: pd.DataFrame) -> str:
    if monthly_totals.empty:
        return "<p class='note'>No historical activity.</p>"
    trimmed = monthly_totals.sort_values("Month").tail(12)
    rows = []
    for _, row in trimmed.iterrows():
        signed = row["new_contracts"] if "new_contracts" in row else None
        value = signed if signed is not None else row.get("active_contracts", 0)
        rows.append(
            "<tr>"
            f"<td>{row['Month'].strftime('%Y-%m')}</td>"
            f"<td>{format_number(value)}</td>"
            "</tr>"
        )
    return (
        "<table class='dataframe data-table'>"
        "<thead><tr><th>Month</th><th>Contracts Signed (payment month)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_cards(total_active: int, latest_month: pd.Timestamp, latest_rows: pd.DataFrame) -> str:
    latest_month_label = latest_month.strftime("%Y-%m")
    new_contracts = int(latest_rows["new_contracts"].sum())
    terminations = int(latest_rows["terminations"].sum())
    unique_nats = latest_rows["Client Nationality"].nunique()
    unknown_active = int(
        latest_rows.loc[
            latest_rows["Client Nationality"].str.lower().isin(UNKNOWN_LABELS), "active_contracts"
        ].sum()
    )
    unknown_share = unknown_active / total_active if total_active else 0
    cards = [
        ("Active Contracts (current)", format_number(total_active), f"Active in latest month snapshot ({latest_month_label})"),
        ("New Contracts", format_number(new_contracts), "Signed during latest month"),
        ("Terminations", format_number(terminations), "Closed during latest month"),
        ("Nationalities", format_number(unique_nats), "Unique in latest month"),
        ("Unknown", format_number(unknown_active), f"{format_percent(unknown_share, 1)} of active base"),
    ]
    card_html = []
    for title, value, note in cards:
        card_html.append(
            "<div class='card'>"
            f"<h3>{title}</h3>"
            f"<p>{value}</p>"
            f"<span>{note}</span>"
            "</div>"
        )
    return "<div class='cards'>" + "".join(card_html) + "</div>"


def build_html_page(data: KioskData, nav_html: str) -> str:
    full_latest_rows = data.latest_rows
    latest_rows = full_latest_rows.head(10).copy()
    total_active = int(full_latest_rows["kiosk_total"].max())
    prev_month_ts = data.monthly_totals[data.monthly_totals["Month"] < data.latest_month]
    prev_active = None
    prev_month_label = None
    if not prev_month_ts.empty:
        prev_active = int(prev_month_ts.iloc[-1]["active_contracts"])
        prev_month_label = prev_month_ts.iloc[-1]["Month"].strftime("%Y-%m")

    cards_html = render_cards(total_active, data.latest_month, full_latest_rows)
    highlights_html = render_highlights(data, total_active, prev_active, prev_month_label)
    latest_table = render_latest_table(latest_rows, data.latest_month.strftime("%b %Y"))
    growth_table, churn_table = render_recent_tables(data.recent_rows)
    trailing_table = render_trailing_table(data.monthly_totals)

    refresh_date = data.latest_month.strftime("%Y-%m-%d")
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{data.name} — Kiosk Nationality Analysis</title>
<style>
body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; background: #f5f7fb; color: #1f2937; }}
.header {{ background: #0f172a; color: #f8fafc; padding: 28px 40px; position: sticky; top: 0; z-index: 20; display: flex; align-items: center; justify-content: space-between; }}
.header h1 {{ margin: 0; font-size: 24px; }}
.header span {{ color: #94a3b8; font-size: 14px; }}
.menu-button {{ width: 44px; height: 44px; border-radius: 8px; background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.3); display: grid; place-content: center; cursor: pointer; transition: background 0.2s ease; }}
.menu-button span {{ display: block; width: 20px; height: 2px; background: #f8fafc; margin: 3px 0; border-radius: 2px; }}
.menu-button:hover {{ background: rgba(148, 163, 184, 0.24); }}
.drawer {{ position: fixed; inset: 0; background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(3px); opacity: 0; pointer-events: none; transition: opacity 0.25s ease; z-index: 30; }}
.drawer.open {{ opacity: 1; pointer-events: auto; }}
.drawer-panel {{ position: absolute; top: 0; left: 0; bottom: 0; width: min(320px, 90vw); background: #0b1220; padding: 32px 28px; box-shadow: 8px 0 24px rgba(2, 6, 23, 0.2); overflow-y: auto; }}
.drawer-panel h2 {{ color: #e2e8f0; font-size: 18px; margin: 0 0 16px; }}
.drawer-panel ul {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }}
.drawer-panel a {{ display: block; padding: 10px 12px; border-radius: 8px; color: #e2e8f0; text-decoration: none; font-weight: 500; transition: background 0.2s ease, transform 0.2s ease; }}
.drawer-panel a:hover {{ background: rgba(148, 163, 184, 0.16); transform: translateX(4px); }}
.section {{ padding: 32px 40px; }}
.section h2 {{ margin-top: 0; color: #0f172a; }}
.cards {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 20px; }}
.card {{ background: #ffffff; border-radius: 12px; padding: 18px 20px; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08); }}
.card h3 {{ margin: 0; font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
.card p {{ margin: 8px 0 0; font-size: 24px; font-weight: 600; color: #0f172a; }}
.card span {{ font-size: 13px; color: #64748b; }}
.panel-grid {{ display: grid; gap: 22px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
.panel {{ background: #ffffff; border-radius: 12px; padding: 22px 24px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); }}
.panel h3 {{ margin-top: 0; }}
.panel ul {{ margin: 12px 0 0; padding-left: 20px; color: #475569; }}
.data-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
.data-table th, .data-table td {{ border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }}
.data-table th {{ background: #f1f5f9; color: #0f172a; }}
.data-table tr:nth-child(even) {{ background: #f8fafc; }}
.note {{ font-size: 13px; color: #64748b; margin-top: 10px; }}
.footer {{ padding: 24px 40px 60px; color: #64748b; font-size: 13px; }}
.back-link {{ color: #38bdf8; text-decoration: none; }}
@media (max-width: 720px) {{
  .section {{ padding: 24px; }}
  .header {{ padding: 24px; }}
}}
</style>
<script>
function toggleMenu() {{
  const drawer = document.getElementById('nav-drawer');
  drawer.classList.toggle('open');
}}
function closeMenu(event) {{
  if (event.target.id === 'nav-drawer') {{
    toggleMenu();
  }}
}}
</script>
</head>
<body>
<header class='header'>
  <div>
    <h1>{data.name}</h1>
    <span>Nationality and churn insights per kiosk</span>
  </div>
  <div class='menu-button' onclick='toggleMenu()'>
    <span></span>
    <span></span>
    <span></span>
  </div>
</header>
<div class='drawer' id='nav-drawer' onclick='closeMenu(event)'>
  <div class='drawer-panel'>
    <h2>Kiosks</h2>
    <ul>
      {nav_html}
    </ul>
  </div>
</div>
<section class='section'>
  <h2>Key Metrics</h2>
  {cards_html}
</section>
<section class='section'>
  <h2>Highlights</h2>
  <div class='panel'>
    <h3>What changed recently</h3>
    {highlights_html}
  </div>
</section>
<section class='section'>
  <h2>Latest Nationality Mix</h2>
  <div class='panel'>
    {latest_table}
  </div>
</section>
<section class='section'>
  <h2>Recent Growth & Churn</h2>
  <div class='panel-grid'>
    <div class='panel'>
      <h3>Growth (last 3 months)</h3>
      {growth_table}
    </div>
    <div class='panel'>
      <h3>Churn focus (last 3 months)</h3>
      {churn_table}
    </div>
  </div>
</section>
<section class='section'>
  <h2>Active Contracts Trend</h2>
  <div class='panel'>
    <h3>Trailing 12 months</h3>
    {trailing_table}
  </div>
</section>
<footer class='footer'>
  <a class='back-link' href='index.html'>Back to executive report</a> · Latest refresh: {refresh_date}
</footer>
</body>
</html>"""


def main() -> None:
    latest_df, recent_df, ts_df = load_source_frames()
    payloads = build_kiosk_payloads(latest_df, recent_df, ts_df)
    nav_html = build_navigation(sorted(payloads.keys()))

    for kiosk, data in payloads.items():
        html = build_html_page(data, nav_html)
        output_path = BASE_DIR / f"{data.slug}.html"
        output_path.write_text(html, encoding="utf-8")
        print(f"Updated {output_path.name}")


if __name__ == "__main__":
    pd.set_option("mode.copy_on_write", True)
    main()
