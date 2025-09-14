import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union  # <-- Add this import

DATA_FILES = {
    "facebook": "Facebook.csv",
    "google": "Google.csv",
    "tiktok": "TikTok.csv",
    "business": "business.csv",
}


def _read_marketing_csv(path: Path, platform: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])\
           .rename(columns={
               "impression": "impressions",
               "attributed revenue": "attributed_revenue"
           })
    df["platform"] = platform.capitalize()
    # Standardise column order
    return df[["date", "platform", "tactic", "state", "campaign", "impressions", "clicks", "spend", "attributed_revenue"]]


def load_raw_data(base_path: Path) -> dict:
    """Load all source CSVs into dataframes."""
    data = {}
    for key, fname in DATA_FILES.items():
        fpath = base_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Expected file not found: {fpath}")
        if key == "business":
            data[key] = pd.read_csv(fpath, parse_dates=["date"])\
                .rename(columns={
                    "# of orders": "orders",
                    "# of new orders": "new_orders",
                    "new customers": "new_customers",
                    "total revenue": "total_revenue",
                    "gross profit": "gross_profit",
                    "COGS": "cogs"
                })
        else:
            data[key] = _read_marketing_csv(fpath, key)
    return data


def unify_marketing(marketing_frames: list[pd.DataFrame]) -> pd.DataFrame:
    marketing = pd.concat(marketing_frames, ignore_index=True)
    # Derived metrics at row (campaign-day) level
    marketing["ctr"] = np.where(marketing["impressions"] > 0, marketing["clicks"] / marketing["impressions"], np.nan)
    marketing["cpc"] = np.where(marketing["clicks"] > 0, marketing["spend"] / marketing["clicks"], np.nan)
    marketing["cpm"] = np.where(marketing["impressions"] > 0, marketing["spend"] / marketing["impressions"] * 1000, np.nan)
    marketing["roas"] = np.where(marketing["spend"] > 0, marketing["attributed_revenue"] / marketing["spend"], np.nan)
    return marketing


def aggregate_marketing(marketing: pd.DataFrame, level: list[str]) -> pd.DataFrame:
    agg = (marketing
           .groupby(level, as_index=False)
           .agg({
               "impressions": "sum",
               "clicks": "sum",
               "spend": "sum",
               "attributed_revenue": "sum"
           }))
    agg["ctr"] = np.where(agg["impressions"] > 0, agg["clicks"] / agg["impressions"], np.nan)
    agg["cpc"] = np.where(agg["clicks"] > 0, agg["spend"] / agg["clicks"], np.nan)
    agg["cpm"] = np.where(agg["impressions"] > 0, agg["spend"] / agg["impressions"] * 1000, np.nan)
    agg["roas"] = np.where(agg["spend"] > 0, agg["attributed_revenue"] / agg["spend"], np.nan)
    return agg


def blend_with_business(marketing_daily: pd.DataFrame, business: pd.DataFrame) -> pd.DataFrame:
    """Join marketing (already aggregated by date) with business metrics.

    marketing_daily: expects columns: date, impressions, clicks, spend, attributed_revenue
    business: date-level actuals.
    """
    merged = marketing_daily.merge(business, on="date", how="left")
    merged["marketing_spend_pct_revenue"] = np.where(merged["total_revenue"] > 0, merged["spend"] / merged["total_revenue"], np.nan)
    merged["blended_cac"] = np.where(merged["new_customers"] > 0, merged["spend"] / merged["new_customers"], np.nan)
    merged["gross_margin_pct"] = np.where(merged["total_revenue"] > 0, merged["gross_profit"] / merged["total_revenue"], np.nan)
    return merged


def estimate_platform_orders(marketing: pd.DataFrame, business: pd.DataFrame) -> pd.DataFrame:
    """Estimate per-platform orders by allocating total daily orders by share of attributed revenue.

    Adds columns: est_orders, est_new_customers.
    These are heuristic and for directional funnel visuals only.
    """
    daily_platform = aggregate_marketing(marketing, ["date", "platform"])  # includes roas etc
    daily_totals = daily_platform.groupby("date")["attributed_revenue"].sum().rename("day_attr_rev")
    daily_platform = daily_platform.merge(daily_totals, on="date", how="left")
    business_slim = business[["date", "orders", "new_customers"]]
    daily_platform = daily_platform.merge(business_slim, on="date", how="left")
    daily_platform["rev_share"] = np.where(daily_platform["day_attr_rev"] > 0, daily_platform["attributed_revenue"] / daily_platform["day_attr_rev"], 0)
    daily_platform["est_orders"] = daily_platform["orders"] * daily_platform["rev_share"]
    daily_platform["est_new_customers"] = daily_platform["new_customers"] * daily_platform["rev_share"]
    daily_platform["est_cac"] = np.where(daily_platform["est_new_customers"] > 0, daily_platform["spend"] / daily_platform["est_new_customers"], np.nan)
    daily_platform["est_conv_rate_click_to_order"] = np.where(daily_platform["clicks"] > 0, daily_platform["est_orders"] / daily_platform["clicks"], np.nan)
    return daily_platform


def load_prepared(base_path: Union[str, Path]) -> dict:  # <-- Change here
    base = Path(base_path)
    raw = load_raw_data(base)
    marketing = unify_marketing([raw["facebook"], raw["google"], raw["tiktok"]])
    marketing_daily = aggregate_marketing(marketing, ["date"])
    blended = blend_with_business(marketing_daily, raw["business"])
    platform_estimates = estimate_platform_orders(marketing, raw["business"])
    return {
        "raw": raw,
        "marketing": marketing,
        "marketing_daily": marketing_daily,
        "business": raw["business"],
        "blended": blended,
        "platform_estimates": platform_estimates,
    }


if __name__ == "__main__":  # quick manual test
    data = load_prepared(Path(__file__).parent)
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            print(k, v.head())
