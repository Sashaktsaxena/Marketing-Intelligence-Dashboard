import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from data_prep import load_prepared
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    layout="wide",
    page_icon="📊"
)

@st.cache_data(show_spinner=False)
def get_data():
    return load_prepared(Path(__file__).parent)

data = get_data()
marketing = data["marketing"]
business = data["business"]
blended = data["blended"]
platform_est = data["platform_estimates"]

st.title("Marketing Intelligence Dashboard")
st.caption("Integrated view of marketing performance and business outcomes (demo dataset)")

with st.expander("About & Assumptions", expanded=False):
    st.markdown(
        """
        **Goal**: Provide a decision-focused view that links paid marketing activity with commercial results.
        **Assumptions**:
        - Business metrics are total brand daily actuals (not split by state/platform).
        - Per-platform orders & new customers are *estimated* by allocating daily totals by share of attributed revenue.
        - Use estimated orders ONLY for directional funnel & efficiency comparisons.
        - ROAS uses platform attributed revenue; blended CAC uses total spend / actual new customers.
        """
    )

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
date_min, date_max = marketing["date"].min(), marketing["date"].max()
date_range = st.sidebar.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = date_range, date_range

platform_options = sorted(marketing["platform"].unique())
platform_sel = st.sidebar.multiselect("Platforms", platform_options, default=platform_options)

tactic_options = sorted(marketing["tactic"].unique())
tactic_sel = st.sidebar.multiselect("Tactics", tactic_options, default=tactic_options)

state_options = sorted(marketing["state"].unique())
state_sel = st.sidebar.multiselect("States", state_options, default=state_options)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date)) &
              (df["platform"].isin(platform_sel)) & (df["tactic"].isin(tactic_sel)) & (df["state"].isin(state_sel))]

marketing_f = apply_filters(marketing)
platform_est_f = platform_est[(platform_est["date"] >= pd.to_datetime(start_date)) & (platform_est["date"] <= pd.to_datetime(end_date)) & (platform_est["platform"].isin(platform_sel))]
business_f = business[(business["date"] >= pd.to_datetime(start_date)) & (business["date"] <= pd.to_datetime(end_date))]

# Aggregated views
agg_overall = marketing_f.agg({"impressions":"sum", "clicks":"sum", "spend":"sum", "attributed_revenue":"sum"})
ctr = agg_overall["clicks"] / agg_overall["impressions"] if agg_overall["impressions"] else np.nan
cpc = agg_overall["spend"] / agg_overall["clicks"] if agg_overall["clicks"] else np.nan
cpm = agg_overall["spend"] / agg_overall["impressions"] * 1000 if agg_overall["impressions"] else np.nan
roas = agg_overall["attributed_revenue"] / agg_overall["spend"] if agg_overall["spend"] else np.nan
total_revenue = business_f["total_revenue"].sum()
gross_profit = business_f["gross_profit"].sum()
orders_sum = business_f["orders"].sum()
new_customers_sum = business_f["new_customers"].sum()
blended_cac = agg_overall["spend"] / new_customers_sum if new_customers_sum else np.nan
gm_pct = gross_profit / total_revenue if total_revenue else np.nan
spend_pct_rev = agg_overall["spend"] / total_revenue if total_revenue else np.nan

def kpi(label, value, suffix="", precision=2):
    st.metric(label, f"{value:,.{precision}f}{suffix}" if pd.notnull(value) else "–")

st.subheader("Executive Summary")
col_kpi = st.columns(8)
with col_kpi[0]: kpi("Spend", agg_overall["spend"], suffix="$")
with col_kpi[1]: kpi("Attributed Rev", agg_overall["attributed_revenue"], suffix="$")
with col_kpi[2]: kpi("ROAS", roas)
with col_kpi[3]: kpi("Blended CAC", blended_cac, suffix="$")
with col_kpi[4]: kpi("CTR", ctr*100, suffix="%")
with col_kpi[5]: kpi("CPM", cpm, suffix="$")
with col_kpi[6]: kpi("Gross Margin %", gm_pct*100 if not np.isnan(gm_pct) else np.nan, suffix="%")
with col_kpi[7]: kpi("Spend % Revenue", spend_pct_rev*100 if not np.isnan(spend_pct_rev) else np.nan, suffix="%")

# -----------------------------
# Time Series
# -----------------------------
st.markdown("### 1. Trend View")
daily = marketing_f.groupby("date").agg({"spend":"sum", "attributed_revenue":"sum", "impressions":"sum", "clicks":"sum"}).reset_index()
daily = daily.merge(business_f[["date", "orders", "total_revenue"]], on="date", how="left")
daily["roas"] = np.where(daily["spend"]>0, daily["attributed_revenue"]/daily["spend"], np.nan)

fig_trend = px.line(daily, x="date", y=["spend", "attributed_revenue"], title="Spend vs Attributed Revenue", markers=True)
fig_roas = px.line(daily, x="date", y="roas", title="Daily ROAS", markers=True)

col1, col2 = st.columns(2)
col1.plotly_chart(fig_trend, use_container_width=True)
col2.plotly_chart(fig_roas, use_container_width=True)

# Platform spend share
platform_share = marketing_f.groupby("platform").agg({"spend":"sum", "attributed_revenue":"sum"}).reset_index()
platform_share["roas"] = np.where(platform_share["spend"]>0, platform_share["attributed_revenue"]/platform_share["spend"], np.nan)
fig_share = px.bar(platform_share, x="platform", y="spend", color="platform", title="Spend by Platform", text_auto=True)
fig_share_roas = px.bar(platform_share, x="platform", y="roas", color="platform", title="ROAS by Platform", text_auto=True)
col3, col4 = st.columns(2)
col3.plotly_chart(fig_share, use_container_width=True)
col4.plotly_chart(fig_share_roas, use_container_width=True)

# -----------------------------
# Allocation & Efficiency
# -----------------------------
st.markdown("### 2. Allocation & Efficiency")
granularity = st.radio("Bubble granularity", ["platform", "tactic", "campaign"], horizontal=True)
if granularity == "platform":
    group_cols = ["platform"]
elif granularity == "tactic":
    group_cols = ["platform", "tactic"]
else:
    group_cols = ["platform", "tactic", "campaign"]

bubble = marketing_f.groupby(group_cols).agg({"spend":"sum", "attributed_revenue":"sum", "impressions":"sum", "clicks":"sum"}).reset_index()
bubble["roas"] = np.where(bubble["spend"]>0, bubble["attributed_revenue"]/bubble["spend"], np.nan)
bubble["ctr"] = np.where(bubble["impressions"]>0, bubble["clicks"]/bubble["impressions"], np.nan)

fig_bubble = px.scatter(bubble, x="spend", y="roas", size="spend", color=group_cols[0], hover_data=bubble.columns, title="Efficiency Quadrant: Scale (Spend) vs ROAS")
st.plotly_chart(fig_bubble, use_container_width=True)

# Pareto (cumulative revenue contribution by spend rank)
pareto = bubble.sort_values("spend", ascending=False).copy()
pareto["cum_spend_pct"] = pareto["spend"].cumsum() / pareto["spend"].sum()
pareto["cum_attr_rev_pct"] = pareto["attributed_revenue"].cumsum() / pareto["attributed_revenue"].sum()
fig_pareto = px.line(pareto, x="cum_spend_pct", y="cum_attr_rev_pct", title="Pareto: Cumulative Spend vs Attributed Revenue")
fig_pareto.add_hline(y=0.8, line_dash="dash", line_color="red")
fig_pareto.add_vline(x=0.2, line_dash="dash", line_color="red")
st.plotly_chart(fig_pareto, use_container_width=True)

# -----------------------------
# Funnel & Estimated Conversion
# -----------------------------
st.markdown("### 3. Funnel & Conversion (Estimated)")
funnel = platform_est_f.groupby("platform").agg({"impressions":"sum", "clicks":"sum", "spend":"sum", "est_orders":"sum", "est_new_customers":"sum", "attributed_revenue":"sum"}).reset_index()
funnel["click_through_rate"] = np.where(funnel["impressions"]>0, funnel["clicks"]/funnel["impressions"], np.nan)
funnel["est_order_rate"] = np.where(funnel["clicks"]>0, funnel["est_orders"]/funnel["clicks"], np.nan)
funnel["roas"] = np.where(funnel["spend"]>0, funnel["attributed_revenue"]/funnel["spend"], np.nan)
funnel["est_cac"] = np.where(funnel["est_new_customers"]>0, funnel["spend"]/funnel["est_new_customers"], np.nan)
st.dataframe(funnel[["platform","impressions","clicks","est_orders","est_new_customers","roas","est_cac","click_through_rate","est_order_rate"]])

# Simple regression: does spend predict orders?
st.markdown("### 4. Spend → Orders Relationship (Exploratory)")
reg_df = daily.dropna(subset=["spend", "orders"]).copy()
if len(reg_df) > 5 and reg_df["spend"].std() > 0:
    X = reg_df[["spend"]].values
    y = reg_df["orders"].values
    model = LinearRegression().fit(X, y)
    reg_df["pred_orders"] = model.predict(X)
    r2 = model.score(X, y)
    fig_reg = px.scatter(reg_df, x="spend", y="orders", trendline="ols", title=f"Daily Spend vs Orders (R²={r2:.2f})")
    st.plotly_chart(fig_reg, use_container_width=True)
    st.caption("Caution: Correlation ≠ causation; does not adjust for seasonality or tactics.")
else:
    st.info("Not enough data variation for regression in the selected filters.")

# -----------------------------
# Raw Data Access
# -----------------------------
with st.expander("Raw Aggregated Data"):
    st.dataframe(daily)

with st.expander("Download Prepared Data"):
    csv_bytes = marketing_f.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Marketing CSV", csv_bytes, file_name="marketing_filtered.csv", mime="text/csv")

st.markdown("---")
st.caption("© 2025 Marketing Intelligence Demo. For assessment purposes only.")
