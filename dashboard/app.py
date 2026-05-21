"""
Healthcare RAG — Streamlit KPI Dashboard (M4)

4 sections:
  1. System Overview  — query count, uptime, last deployment
  2. Model Performance — BLEU / ROUGE-L trends across MLflow runs
  3. Query Analytics  — category pie chart, avg latency per category
  4. Data Quality     — preprocessing stats

Run:
    streamlit run dashboard/app.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare RAG Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load deployment config from environment (set in .env)
DEPLOY_ENV = os.getenv("DEPLOY_ENV", "local")
DEPLOY_DATE = os.getenv("DEPLOY_DATE", "—")


# ── Helpers to load real data ─────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_labelled_data() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "data" / "processed" / "pubmedqa_labelled.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=60)
def load_cleaned_data() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "data" / "processed" / "pubmedqa_cleaned.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=60)
def load_raw_data() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "data" / "raw" / "pubmedqa_raw.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=60)
def load_evaluation_results() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "reports" / "rag_evaluation_results.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=60)
def load_pipeline_test_log() -> list | None:
    path = PROJECT_ROOT / "reports" / "rag_pipeline_test_log.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=30)
def load_mlflow_runs() -> pd.DataFrame | None:
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        if not experiments:
            return None
        exp_id = experiments[0].experiment_id
        runs = client.search_runs(experiment_ids=[exp_id], max_results=20)
        rows = []
        for r in runs:
            rows.append({
                "run_id": r.info.run_id[:8],
                "status": r.info.status,
                **r.data.metrics,
                **{f"param_{k}": v for k, v in r.data.params.items()},
            })
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Healthcare RAG — KPI Dashboard")
st.caption(f"eyouth × DEPI 2025 | Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: System Overview
# ═════════════════════════════════════════════════════════════════════════════
st.header("1 · System Overview")

labelled = load_labelled_data()
pipeline_log = load_pipeline_test_log()

query_count = len(pipeline_log) if pipeline_log else 0
corpus_size = len(labelled) if labelled is not None else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Corpus Size", f"{corpus_size:,} pairs", help="Rows in pubmedqa_labelled.csv")
col2.metric("Test Queries Run", query_count, help="From rag_pipeline_test_log.json")
col3.metric("Deployment", DEPLOY_ENV, help="Set DEPLOY_ENV in .env")
col4.metric("Last Deploy", DEPLOY_DATE if DEPLOY_DATE else "—", help="Set DEPLOY_DATE in .env")

azure_url = os.getenv("AZURE_APP_URL", "")
if azure_url:
    st.info(f"🔗 Live API: [{azure_url}]({azure_url}/health) — `/health` and `/query`")
else:
    st.warning("⚠️ Set AZURE_APP_URL in .env to show the live API link.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Model Performance
# ═════════════════════════════════════════════════════════════════════════════
st.header("2 · Model Performance")

eval_df = load_evaluation_results()
mlflow_runs = load_mlflow_runs()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("BLEU / ROUGE-L (Evaluation Report)")
    if eval_df is not None:
        # Try to plot trend if there are multiple rows (one per run/config)
        metric_cols = [c for c in eval_df.columns if any(m in c.lower() for m in ["bleu", "rouge"])]
        if metric_cols:
            fig = px.line(
                eval_df.reset_index(),
                y=metric_cols,
                title="Evaluation Metrics",
                labels={"index": "Sample", "value": "Score"},
                markers=True,
            )
            fig.update_layout(legend_title_text="Metric", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(eval_df.head(10), use_container_width=True)
    else:
        # Placeholder with KPI targets
        targets = pd.DataFrame({
            "Metric": ["BLEU (RAG)", "BLEU (Baseline)", "ROUGE-L (RAG)", "BERTScore F1", "Hallucination Rate"],
            "Target": ["≥ +20% vs baseline", "baseline", "≥ 0.20", "≥ 0.80", "≤ 15%"],
            "Status": ["Run notebook 08 to populate", "—", "—", "—", "—"],
        })
        st.dataframe(targets, use_container_width=True, hide_index=True)
        st.caption("Run `notebooks/08_evaluation.ipynb` to populate this chart.")

with col_right:
    st.subheader("MLflow Experiment Runs")
    if mlflow_runs is not None:
        # Column names must match what mlflow_tracking.py actually logs:
        #   bleu_rag, rouge_rag, avg_latency_ms, bleu_improvement_pct, macro_f1
        METRIC_COLS = ["bleu_rag", "rouge_rag", "avg_latency_ms", "bleu_improvement_pct", "macro_f1"]
        metric_cols = [c for c in METRIC_COLS if c in mlflow_runs.columns]
        if metric_cols:
            fig2 = px.bar(
                mlflow_runs,
                x="run_id",
                y=metric_cols,
                barmode="group",
                title="Metrics per MLflow Run",
                height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.dataframe(mlflow_runs, use_container_width=True)
    else:
        st.info("MLflow not running or no experiments logged yet.\n\nRun `mlops/mlflow_tracking.py` to populate.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Query Analytics
# ═════════════════════════════════════════════════════════════════════════════
st.header("3 · Query Analytics")

col3a, col3b = st.columns(2)

with col3a:
    st.subheader("Category Distribution")
    if labelled is not None and "category" in labelled.columns:
        cat_counts = labelled["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig3 = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            title=f"6 Medical Categories ({len(labelled):,} labelled rows)",
            hole=0.35,
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Load `data/processed/pubmedqa_labelled.csv` to see category distribution.")

with col3b:
    st.subheader("Avg Latency per Category (Test Queries)")
    if pipeline_log:
        log_df = pd.DataFrame(pipeline_log)
        if "category" in log_df.columns and "latency_ms" in log_df.columns:
            lat_df = log_df.groupby("category")["latency_ms"].mean().reset_index()
            lat_df.columns = ["Category", "Avg Latency (ms)"]
            fig4 = px.bar(
                lat_df,
                x="Category",
                y="Avg Latency (ms)",
                title="Average Response Latency by Category",
                color="Category",
                height=350,
            )
            fig4.add_hline(y=5000, line_dash="dash", line_color="red",
                           annotation_text="KPI limit 5000ms")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Pipeline test log missing `category` or `latency_ms` columns.")
    else:
        st.info("No pipeline test log found.\n\nRun `notebooks/10_end_to_end_test.ipynb`.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Data Quality
# ═════════════════════════════════════════════════════════════════════════════
st.header("4 · Data Quality")

raw = load_raw_data()
cleaned = load_cleaned_data()

col4a, col4b, col4c, col4d = st.columns(4)

raw_rows = len(raw) if raw is not None else "N/A"
clean_rows = len(cleaned) if cleaned is not None else "N/A"
if raw is not None and cleaned is not None and len(raw) > 0:
    removed = len(raw) - len(cleaned)
    pct_kept = round(100 * len(cleaned) / len(raw), 1)
else:
    removed = "N/A"
    pct_kept = "N/A"

col4a.metric("Raw Rows", f"{raw_rows:,}" if isinstance(raw_rows, int) else raw_rows)
col4b.metric("Cleaned Rows", f"{clean_rows:,}" if isinstance(clean_rows, int) else clean_rows)
col4c.metric("Rows Removed", f"{removed:,}" if isinstance(removed, int) else removed,
             help="Duplicates + non-English + nulls")
col4d.metric("% Retained", f"{pct_kept}%" if isinstance(pct_kept, float) else pct_kept)

if cleaned is not None:
    st.subheader("Null Counts in Cleaned Dataset")
    null_counts = cleaned.isnull().sum().reset_index()
    null_counts.columns = ["Column", "Null Count"]
    null_counts["% Null"] = (null_counts["Null Count"] / len(cleaned) * 100).round(2)
    st.dataframe(null_counts, use_container_width=True, hide_index=True)
else:
    st.info("Run `notebooks/02_preprocessing.ipynb` to generate cleaned data.")

if labelled is not None and "category" in labelled.columns:
    st.subheader("Category Coverage (KPI: each ≥ 1%)")
    cov = labelled["category"].value_counts(normalize=True).mul(100).round(2).reset_index()
    cov.columns = ["Category", "% of Dataset"]
    cov["KPI Pass"] = cov["% of Dataset"] >= 1.0
    cov["KPI Pass"] = cov["KPI Pass"].map({True: "✅", False: "❌"})
    st.dataframe(cov, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Healthcare RAG-Powered Medical Q&A Assistant · eyouth × DEPI 2025 · "
    "Team: Abdelrahman · Ziad · Youssef · Doha · Eman"
)
