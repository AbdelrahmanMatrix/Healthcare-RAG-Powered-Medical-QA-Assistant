"""
Healthcare RAG — Shared helpers for multi-page dashboard.
"""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEPLOY_ENV = os.getenv("DEPLOY_ENV", "local")
DEPLOY_DATE = os.getenv("DEPLOY_DATE", "—")
API_BASE_URL = os.getenv("AZURE_APP_URL", "http://localhost:8000")

# ── Shared Constants ─────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Symptoms": "#00E5FF",    # Neon Cyan
    "Diagnosis": "#2979FF",   # Electric Blue
    "Treatment": "#00E676",   # Neon Green
    "Medication": "#D500F9",  # Neon Purple
    "Prevention": "#1DE9B6",  # Teal
    "General": "#8D99AE",     # Cool Grey
}

# ── Custom CSS Theme (Modern, Minimalist, Futuristic) ────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

    :root {
        --font-display: 'Space Grotesk', sans-serif;
        --font-body: 'Inter', sans-serif;

        /* Light Mode - Clean & Minimal */
        --bg: #F8F9FA;
        --card: #FFFFFF;
        --surface: #F1F3F5;
        --border: #E9ECEF;
        --text: #1A1D20;
        --text-secondary: #6C757D;
        --primary: #000000;
        --accent: #00E5FF;
        
        --shadow: 0 4px 12px rgba(0,0,0,0.05);
        --shadow-hover: 0 8px 24px rgba(0,229,255,0.15);
        --radius: 12px;
        --radius-sm: 8px;
    }

    /* ── Base layout ────────────────────────────────────────── */
    .stApp { background: var(--bg); }
    * { font-family: var(--font-body) !important; }
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-display) !important;
        color: var(--text) !important;
        letter-spacing: -0.03em !important;
    }
    p, li, div, span { color: var(--text-secondary); }
    label {
        color: var(--text) !important;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* ── Hero Section (Futuristic) ──────────────────────────── */
    .hero-gradient {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        color: white !important;
        padding: 3rem;
        border-radius: var(--radius);
        margin-bottom: 2rem;
        box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.2), 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-gradient::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(to right, transparent, rgba(0, 229, 255, 0.1), transparent);
        transform: skewX(-20deg);
        animation: shine 6s infinite;
    }
    @keyframes shine {
        100% { left: 200%; }
    }
    .hero-gradient h1 {
        color: #fff !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
    }
    .hero-gradient .subtitle {
        color: #94a3b8 !important;
        font-size: 1.1rem;
        max-width: 650px;
        margin: 0 auto;
    }
    .hero-badge {
        padding: 0.4rem 1rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(0, 229, 255, 0.05);
        color: #00E5FF;
        border: 1px solid rgba(0, 229, 255, 0.2);
        letter-spacing: 0.05em;
    }

    /* ── Top Navigation Bar ────────────────────────────────── */
    .top-nav-container {
        background: var(--card);
        border-radius: var(--radius);
        padding: 0.5rem 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .nav-brand-text {
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text);
        letter-spacing: -0.02em;
    }
    .top-nav-container [data-testid="stPageLink"] a {
        border-radius: 8px !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        transition: all 0.3s ease !important;
    }
    .top-nav-container [data-testid="stPageLink"] a:hover {
        background: rgba(0, 229, 255, 0.1) !important;
        color: var(--text) !important;
    }
    .top-nav-container [data-testid="stPageLink"] a[aria-current="page"] {
        background: rgba(0, 229, 255, 0.05) !important;
        color: var(--text) !important;
        border-bottom: 2px solid #00E5FF;
    }

    /* ── Navigation Cards (Homepage) ───────────────────────── */
    .nav-card {
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: var(--card);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-hover);
        border-color: rgba(0, 229, 255, 0.3);
    }
    .nav-card .card-icon { font-size: 3rem; margin-bottom: 1rem; }
    .nav-card .card-title {
        font-size: 1.4rem;
        color: var(--text);
    }
    .nav-card .card-desc {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Chat Bubbles ──────────────────────────────────────── */
    .user-bubble {
        background: var(--text);
        color: var(--bg);
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 4px 20px;
    }
    .user-bubble p, .user-bubble .bubble-header-label { color: var(--bg) !important; }
    
    .bot-bubble {
        background: var(--card);
        border: 1px solid var(--border);
        padding: 1.25rem 1.5rem;
        border-radius: 20px 20px 20px 4px;
        box-shadow: var(--shadow);
    }
    .bot-bubble .bubble-header-label {
        color: var(--text);
        font-weight: 700;
    }
    .bot-bubble .category-badge {
        color: #000;
        padding: 0.2rem 0.75rem;
        border-radius: 20px;
        font-family: var(--font-display);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.65rem;
        letter-spacing: 0.05em;
    }

    /* ── Metrics & DataFrames ──────────────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem;
        box-shadow: var(--shadow);
    }
    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-family: var(--font-display) !important;
        letter-spacing: -0.05em;
    }
    
    /* ── Input & Buttons ───────────────────────────────────── */
    /* All text-type inputs — explicit color ensures visibility in light mode */
    input:not([type="checkbox"]):not([type="radio"]):not([type="range"]) {
        color: var(--text) !important;
    }
    input::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.8;
    }
    textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.8;
    }
    .stTextInput input {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        background: var(--surface) !important;
    }
    .stTextInput input:focus {
        border-color: #00E5FF !important;
        box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.2) !important;
    }
    .stTextArea textarea {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        background: var(--surface) !important;
    }
    .stTextArea textarea:focus {
        border-color: #00E5FF !important;
        box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.2) !important;
    }
    /* Select boxes (stSelectbox, stMultiSelect) */
    div[data-baseweb="select"] > div {
        background: var(--surface) !important;
        border-color: var(--border) !important;
    }
    div[data-baseweb="select"] input, 
    div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        color: var(--text) !important;
    }
    .stButton button[kind="primary"] {
        background: var(--text) !important;
        color: var(--bg) !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-display) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.85rem !important;
    }
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4) !important;
        border-color: #00E5FF !important;
    }
    .stButton button[kind="secondary"] {
        background: white !important;
        color: var(--text) !important;
        border: 2px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }
    .stButton button[kind="secondary"]:hover {
        background: var(--surface) !important;
        border-color: #00E5FF !important;
    }

    /* ── Page headers (sub-pages) ────────────────────────────── */
    .page-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .page-header-icon {
        font-size: 2rem;
        line-height: 1;
        flex-shrink: 0;
    }
    .page-header h1 {
        margin: 0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .page-header p {
        margin: 0.1rem 0 0 !important;
        font-size: 0.9rem !important;
    }

    /* ── Chat bubbles (Medical Query page) ───────────────────── */
    .chat-user-wrap {
        display: flex;
        justify-content: flex-end;
        margin: 1.5rem 0 0.5rem;
    }
    .chat-bot-wrap {
        display: flex;
        margin: 0.5rem 0 1rem;
    }
    .bubble-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
    }
    .bubble-header-icon {
        font-size: 1.1rem;
    }
    .user-bubble .bubble-header-label {
        font-weight: 600;
        font-size: 0.8rem;
    }
    .user-bubble p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .bot-bubble .answer-text {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: var(--text);
    }
    .bot-bubble .bubble-footer {
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid var(--border);
        font-size: 0.75rem;
        color: var(--text-secondary);
        display: flex;
        justify-content: space-between;
    }

    /* ── Source citation cards (Medical Query) ───────────────── */
    .source-card {
        background: var(--surface);
        border-left: 3px solid #00E5FF;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    .source-card .source-title {
        font-weight: 500;
        font-size: 0.9rem;
        margin: 0 0 0.25rem;
        color: var(--text);
    }
    .source-excerpt {
        margin: 0.25rem 0 0;
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-style: italic;
    }

    /* ── Footer credits ──────────────────────────────────────── */
    .footer-credits {
        text-align: center;
        padding: 1rem 0;
    }
    .footer-credits p { margin: 0; }
    .footer-credits .footer-title {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    .footer-credits .footer-team {
        font-size: 0.75rem;
        color: #BDBDBD;
        margin-top: 0.25rem;
    }

    /* ── Metric Group Cards (KPI Dashboard) ──────────────────── */
    .metric-group {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        height: 100%;
        transition: all 0.3s ease;
    }
    .metric-group:hover {
        border-color: rgba(0, 229, 255, 0.2);
        box-shadow: var(--shadow-hover);
    }
    .metric-group-header {
        font-family: var(--font-display);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent);
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .metric-group-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid var(--border);
    }
    .metric-group-item:last-child {
        border-bottom: none;
    }
    .metric-group-item .mgi-label {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.3;
    }
    .metric-group-item .mgi-value {
        font-family: var(--font-display);
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.03em;
        white-space: nowrap;
        text-align: right;
        margin-left: 1rem;
    }
    .metric-group-item .mgi-delta {
        font-size: 0.65rem;
        font-weight: 600;
        color: #00E676;
        display: block;
        text-align: right;
        letter-spacing: 0.02em;
    }
    .metric-group-item .mgi-delta.warning {
        color: #FFB74D;
    }

    /* ── Dark mode toggle indicator ────────────────────────── */
    .mode-indicator {
        font-family: var(--font-display) !important;
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        text-align: center !important;
        padding: 0.1rem 0 !important;
        line-height: 1.2 !important;
    }

    /* ── Nav brand inline layout ─────────────────────────────── */
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.2rem 0;
        white-space: nowrap;
    }

    /* ── Page link base ──────────────────────────────────────── */
    div[data-testid="stPageLink"] a {
        font-family: var(--font-body) !important;
        text-decoration: none !important;
    }

    /* ── Alerts ──────────────────────────────────────────────── */
    div.stAlert {
        border-radius: var(--radius-sm) !important;
        border-left: 4px solid !important;
    }
    div[data-testid="stInfo"] { border-left-color: #2979FF !important; }
    div[data-testid="stSuccess"] { border-left-color: #00E676 !important; }
    div[data-testid="stWarning"] { border-left-color: #FFB74D !important; }
    div[data-testid="stError"] { border-left-color: #EF5350 !important; }

    /* ── Dividers ───────────────────────────────────────────── */
    hr {
        margin: 1.5rem 0 !important;
        border-color: var(--border) !important;
        opacity: 0.5;
    }

    /* ── Expander ────────────────────────────────────────────── */
    div.streamlit-expanderHeader {
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
        border-radius: var(--radius-sm) !important;
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }
    div.streamlit-expanderContent {
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 var(--radius-sm) var(--radius-sm);
        padding: 1rem !important;
        background: var(--surface);
    }

    /* ── DataFrames ──────────────────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border-radius: var(--radius-sm) !important;
        overflow: hidden;
        box-shadow: var(--shadow);
    }

    /* ── Caption ─────────────────────────────────────────────── */
    .stCaptionContainer p {
        font-size: 0.8rem !important;
        color: var(--text-secondary) !important;
    }

    /* ── Spinner ─────────────────────────────────────────────── */
    .stSpinner > div {
        border-color: #00E5FF transparent transparent transparent !important;
    }

    /* ── Custom scrollbar ────────────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 229, 255, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0, 229, 255, 0.5); }

    /* ── Tabs ────────────────────────────────────────────────── */
    button[data-testid="stTab"] {
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
    }

    /* Hide default Streamlit elements */
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="collapsedControl"] { display: none !important; }
</style>
"""

# ── Dark Mode CSS (True Cyberpunk Aesthetic) ─────────────────────────────────
DARK_MODE_CSS = """
<style>
    [data-theme="dark"] {
        --bg: #030712;
        --card: #0B0F19;
        --surface: #111827;
        --border: #1F2937;
        --text: #F9FAFB;
        --text-secondary: #9CA3AF;
        --shadow: 0 4px 20px rgba(0,0,0,0.5);
        --shadow-hover: 0 0 20px rgba(0, 229, 255, 0.15);
    }

    [data-theme="dark"] .nav-card {
        background: rgba(11, 15, 25, 0.7);
        backdrop-filter: blur(10px);
    }
    
    [data-theme="dark"] .user-bubble {
        background: #111827;
        border: 1px solid #1F2937;
        color: #F9FAFB;
    }
    [data-theme="dark"] .user-bubble p, 
    [data-theme="dark"] .user-bubble .bubble-header-label { color: #F9FAFB !important; }
    
    [data-theme="dark"] .bot-bubble {
        background: #0B0F19;
        border-color: #1F2937;
    }
    
    [data-theme="dark"] .stTextInput input {
        background: #0B0F19 !important;
        color: #F9FAFB !important;
    }
    [data-theme="dark"] .stTextArea textarea {
        background: #0B0F19 !important;
        color: #F9FAFB !important;
    }
    
    [data-theme="dark"] .stButton button[kind="primary"] {
        background: #00E5FF !important;
        color: #030712 !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.2) !important;
    }
    [data-theme="dark"] .stButton button[kind="primary"]:hover {
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.5) !important;
    }
    [data-theme="dark"] .stButton button[kind="secondary"] {
        background: transparent !important;
        color: #00E5FF !important;
        border-color: #1F2937 !important;
    }
    [data-theme="dark"] .stButton button[kind="secondary"]:hover {
        border-color: #00E5FF !important;
    }

    /* Page headers in dark mode */
    [data-theme="dark"] .page-header p {
        color: #9CA3AF !important;
    }

    /* Source citation cards in dark mode */
    [data-theme="dark"] .source-card {
        background: #111827 !important;
        border-left-color: #00E5FF !important;
    }
    [data-theme="dark"] .source-card .source-title {
        color: #F9FAFB !important;
    }
    [data-theme="dark"] .source-excerpt {
        color: #6B7280 !important;
    }

    /* Footer in dark mode */
    [data-theme="dark"] .footer-credits .footer-title {
        color: #6B7280 !important;
    }
    [data-theme="dark"] .footer-credits .footer-team {
        color: #4B5563 !important;
    }

    /* Alerts in dark mode */
    [data-theme="dark"] div[data-testid="stInfo"] { background: #1A2D3D !important; }
    [data-theme="dark"] div[data-testid="stSuccess"] { background: #0D2B1D !important; }
    [data-theme="dark"] div[data-testid="stWarning"] { background: #2A1F0E !important; }
    [data-theme="dark"] div[data-testid="stError"] { background: #2A1215 !important; }

    /* Tabs in dark mode */
    [data-theme="dark"] button[data-testid="stTab"] { color: #9CA3AF !important; }
    [data-theme="dark"] button[data-testid="stTab"][aria-selected="true"] { color: #00E5FF !important; }

    /* DataFrames in dark mode */
    [data-theme="dark"] div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
    }

    /* Caption in dark mode */
    [data-theme="dark"] .stCaptionContainer p {
        color: #6B7280 !important;
    }

    /* Scrollbar in dark mode */
    [data-theme="dark"] ::-webkit-scrollbar-thumb {
        background: rgba(0, 229, 255, 0.15) !important;
    }
    [data-theme="dark"] ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 229, 255, 0.3) !important;
    }

    /* Bot bubble text in dark mode */
    [data-theme="dark"] .bot-bubble .answer-text {
        color: #F9FAFB !important;
    }
    [data-theme="dark"] .bot-bubble .bubble-footer {
        border-top-color: #1F2937 !important;
        color: #6B7280 !important;
    }

    /* Metric groups in dark mode */
    [data-theme="dark"] .metric-group {
        background: rgba(11, 15, 25, 0.7);
        backdrop-filter: blur(10px);
    }
    [data-theme="dark"] .metric-group:hover {
        border-color: rgba(0, 229, 255, 0.3);
    }

    /* Expander in dark mode */
    [data-theme="dark"] div.streamlit-expanderHeader {
        background: #0B0F19 !important;
        border-color: #1F2937 !important;
        color: #F9FAFB !important;
    }
    [data-theme="dark"] div.streamlit-expanderContent {
        background: #111827 !important;
    }
</style>
"""

def _dark_mode_js():
    dark = st.session_state.get("dark_mode", False)
    theme = "dark" if dark else "light"
    return f'<script type="text/javascript">document.documentElement.setAttribute("data-theme", "{theme}");</script>'

# ── Clean Minimalist Plotly Themes ───────────────────────────────────────────
MEDICAL_PLOTLY_THEME = {
    "layout": {
        "font": {"family": "Space Grotesk, sans-serif", "size": 12, "color": "#6C757D"},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "colorway": ["#00E5FF", "#2979FF", "#00E676", "#D500F9", "#1DE9B6"],
        "hovermode": "x unified",
        "xaxis": {"gridcolor": "#E9ECEF", "zerolinecolor": "#E9ECEF"},
        "yaxis": {"gridcolor": "#E9ECEF", "zerolinecolor": "#E9ECEF"},
    }
}

MEDICAL_PLOTLY_DARK_THEME = {
    "layout": {
        "font": {"family": "Space Grotesk, sans-serif", "size": 12, "color": "#9CA3AF"},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "colorway": ["#00E5FF", "#2979FF", "#00E676", "#D500F9", "#1DE9B6"],
        "hovermode": "x unified",
        "xaxis": {"gridcolor": "#1F2937", "zerolinecolor": "#1F2937"},
        "yaxis": {"gridcolor": "#1F2937", "zerolinecolor": "#1F2937"},
    }
}

def inject_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    st.markdown(_dark_mode_js(), unsafe_allow_html=True)

def apply_plotly_theme():
    dark = st.session_state.get("dark_mode", False)
    if dark:
        pio.templates["medical_dark"] = MEDICAL_PLOTLY_DARK_THEME
        pio.templates.default = "plotly_dark+medical_dark"
    else:
        pio.templates["medical"] = MEDICAL_PLOTLY_THEME
        pio.templates.default = "plotly_white+medical"

def section_header(title: str, icon: str = ""):
    st.markdown(
        f"""
        <div style="margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); font-family: var(--font-display);">
            <h3 style="margin:0; font-size: 1.2rem; font-weight: 600;">{icon} {title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_top_nav(current_page: str):
    pages = [
        ("⎈", "Hub", "app.py"),
        ("⚡", "Query", "pages/1_🔬_Medical_Query.py"),
        ("◱", "Telemetry", "pages/2_📊_KPI_Dashboard.py"),
    ]

    st.markdown('<div class="top-nav-container">', unsafe_allow_html=True)
    cols = st.columns([2, 1, 1, 1, 0.8])

    with cols[0]:
        st.markdown(
            '<div class="nav-brand">'
            '<span class="nav-brand-text">HEALTHCARE // RAG</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    for i, (icon, label, path) in enumerate(pages):
        with cols[i+1]:
            st.page_link(path, label=f"{icon} {label}", use_container_width=True, disabled=(path == current_page))

    with cols[4]:
        dark = st.session_state.get("dark_mode", False)
        st.toggle("◑", key="dark_mode", label_visibility="collapsed")
        mode_label = "◑ Dark" if dark else "◐ Light"
        st.markdown(f'<div class="mode-indicator">{mode_label}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Data Loaders ──────────────────────────────────────────────────────────────

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
