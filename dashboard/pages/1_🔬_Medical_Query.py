"""
Healthcare RAG — Medical Query Interface

Chat-style question interface with styled answer bubbles,
category badges, and expandable source citations.
"""

import sys
import os

import streamlit as st
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from home import (  # noqa: E402
    API_BASE_URL,
    CATEGORY_COLORS,
    inject_custom_css,
    apply_plotly_theme,
    build_top_nav,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Query",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
apply_plotly_theme()
build_top_nav("pages/1_🔬_Medical_Query.py")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <span class="page-header-icon">🔬</span>
        <div>
            <h1>Ask a Medical Question</h1>
            <p>Powered by Groq Llama 4 · PubMedQA evidence retrieval</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Query Input ──────────────────────────────────────────────────────────────
with st.container():
    col_q, col_b = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            "Enter your medical question",
            placeholder="e.g., What are the symptoms of type 2 diabetes?",
            label_visibility="collapsed",
        )
    with col_b:
        submitted = st.button("Ask", type="primary", use_container_width=True)

# ── Chat Thread ──────────────────────────────────────────────────────────────
if submitted and question.strip():
    if len(question.strip()) < 5:
        st.warning("Please enter a more detailed question (at least 5 characters).")
    else:
        # User message bubble
        st.markdown(
            f"""
            <div class="chat-user-wrap">
                <div class="user-bubble">
                    <div class="bubble-header">
                        <span class="bubble-header-icon">👤</span>
                        <span class="bubble-header-label">You</span>
                    </div>
                    <p>{question.strip()}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Typing indicator
        with st.spinner("🔍 Retrieving evidence and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"question": question.strip()},
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    category = data.get("category", "Unknown")
                    sources = data.get("source_citations", [])

                    badge_color = CATEGORY_COLORS.get(category, "#5F6368")

                    # Bot answer bubble
                    st.markdown(
                        f"""
                        <div class="chat-bot-wrap">
                            <div class="bot-bubble">
                                <div class="bubble-header">
                                    <span class="bubble-header-icon">🤖</span>
                                    <span class="bubble-header-label">Healthcare RAG</span>
                                    <span class="category-badge" style="background: {badge_color};">
                                        {category}
                                    </span>
                                </div>
                                <p class="answer-text">{answer}</p>
                                <div class="bubble-footer">
                                    <span>📚 {len(sources)} sources retrieved</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Expandable source citations
                    if sources:
                        with st.expander(f"📚 Retrieved Sources ({len(sources)})", expanded=False):
                            for i, src in enumerate(sources[:5]):
                                q = src.get("question", "N/A")
                                cat = src.get("category", "N/A")
                                score = src.get("relevance_score", 0)
                                excerpt = src.get("excerpt", "")
                                excerpt_html = (
                                    f'<p class="source-excerpt">'
                                    f'Excerpt: {excerpt[:200]}</p>'
                                ) if excerpt else ""
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <p class="source-title">Source {i+1}: {q}</p>
                                        <p style="margin: 0; font-size: 0.8rem; color: var(--text-secondary);">
                                            Category: {cat} · Relevance: {score:.3f}
                                        </p>
                                        {excerpt_html}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                if i < len(sources[:5]) - 1:
                                    st.markdown(
                                        "<hr style='margin: 0.25rem 0; opacity: 0.3;'>",
                                        unsafe_allow_html=True,
                                    )
                else:
                    error_detail = resp.json().get("detail", f"HTTP {resp.status_code}")
                    st.error(f"❌ API Error: {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Could not connect to the API at {API_BASE_URL}. "
                    "Is the API server running?"
                )
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

elif submitted:
    st.warning("Please enter a question.")

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
