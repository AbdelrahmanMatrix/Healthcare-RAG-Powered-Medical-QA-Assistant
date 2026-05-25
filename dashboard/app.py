"""
Healthcare RAG — Homepage

Landing page with gradient hero, navigation cards,
and system status overview.
"""

import os

import streamlit as st

from home import (
    DEPLOY_ENV,
    DEPLOY_DATE,
    inject_custom_css,
    apply_plotly_theme,
    build_top_nav,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare RAG — Home",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
apply_plotly_theme()
build_top_nav("app.py")

# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-gradient">
        <h1>🏥 Healthcare RAG System Hub</h1>
        <p class="subtitle">
            Enterprise Medical Question Answering &amp; Verification Pipelines —
            Grounded in <strong>PubMedQA</strong> Literature
        </p>
        <div class="hero-badges">
            <span class="hero-badge">🔬 6 Core Medical Categories</span>
            <span class="hero-badge">🤖 Llama 4 Scout Engine</span>
            <span class="hero-badge">📚 200K+ Vector Embeddings</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Navigation Cards ─────────────────────────────────────────────────────────
st.subheader("System Access Points")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="nav-card">
            <p class="card-icon">⚡</p>
            <h3 class="card-title">Clinical Research Portal</h3>
            <p class="card-desc">
                Query the live vector store pipeline, track logical generation paths,
                and dissect context citations retrieved directly from authoritative
                clinical research records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/1_🔬_Medical_Query.py",
        label="Launch Pipeline",
        use_container_width=True,
    )

with col2:
    st.markdown(
        """
        <div class="nav-card">
            <p class="card-icon">◱</p>
            <h3 class="card-title">Analytics & Evaluation Hub</h3>
            <p class="card-desc">
                Audit continuous deployment validation metrics, inspect data label
                balances, system latencies, and MLflow experiments tracking
                system performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/2_📊_KPI_Dashboard.py",
        label="Open Telemetry",
        use_container_width=True,
    )

st.divider()

# ── System Status ────────────────────────────────────────────────────────────
st.subheader("Node Infrastructure Topology")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("Deployment Topology", DEPLOY_ENV.upper())
col_s2.metric("Release Timestamp", DEPLOY_DATE)

azure_url = os.getenv("AZURE_APP_URL", "")
if azure_url:
    col_s3.metric("API Endpoint Status", "🟢 Live Core")
    col_s4.markdown(f"[📘 System API Docs]({azure_url}/docs)")
else:
    col_s3.metric("API Endpoint Status", "🟡 Local Sandbox")
    col_s4.markdown("[📘 System API Docs](http://localhost:8000/docs)")

st.divider()

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-credits">
        <p class="footer-title">
            Healthcare RAG-Powered Medical Q&A Assistant · eyouth × DEPI 2026
        </p>
        <p class="footer-team">
            Team Topology: Abdelrahman · Ziad · Youssef · Doha · Eman
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
