# Marketing Intelligence Dashboard

Interactive BI dashboard linking multi-platform marketing performance to business outcomes (demo dataset, 120 days).

## Features

- Unified ingestion of Facebook, Google, TikTok campaign data + business KPIs.
- Derived metrics: CTR, CPC, CPM, ROAS, Blended CAC, Spend % Revenue, Gross Margin %.
- Allocation & efficiency analysis (bubble chart, Pareto curve).
- Estimated funnel (impressions → clicks → est orders → est new customers).
- Exploratory regression of Spend → Orders (directional).
- Rich filtering (date range, platform, tactic, state) & CSV export.

## Data Assumptions

- Business metrics are total brand-level; platform/state splits for orders & new customers are estimated by revenue share (directional only).
- Use estimated granular orders strictly for relative comparisons, not financial reporting.

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the provided local URL (e.g. http://localhost:8501).

## Deploy (Streamlit Community Cloud)

1. Push these files + CSVs to a public GitHub repo.
2. Go to https://share.streamlit.io → "New app" → select repo + `app.py`.
3. Add (optional) secrets or config if extending.
4. Deploy – first build installs `requirements.txt`.

### Alternative Hosting

- **Hugging Face Spaces** (Streamlit template) – drop in same files.
- **Render.com** – use a Python web service, start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Docker** (optional):
  ```Dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt ./
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

## Product-Oriented Insights to Look For

- Which platforms/tactics deliver outsized ROAS relative to spend share (efficiency quadrant)?
- Diminishing returns: does increasing spend sustain ROAS or erode it (trend & regression)?
- Allocation opportunities: Pareto cut—what % of campaigns drive 80% of attributed revenue.
- Funnel friction: platforms with strong CTR but weak click→order rate (optimization focus).
- Spend % of Revenue trending vs margin – early warning for profitability pressure.

## Extensibility Ideas

- Introduce incremental lift modeling (geo or time-based synthetic controls).
- Add cohort-level LTV to replace first-order revenue in ROAS (true Payback).
- Blend organic channels & unify into MMM / regression framework.
- Alerting (Slack/email) for metric anomalies beyond rolling z-score threshold.

## License

For assessment/demo use only.
