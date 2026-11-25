"""Custom Streamlit styles for ALPRO UI."""

from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --alpro-bg: #ffffff;
  --alpro-surface: #f7fafc;
  --alpro-surface-2: #f1f5f9;
  --alpro-border: #e5e7eb;
  --alpro-text: #1f2937;
  --alpro-muted: #6b7280;
  --alpro-accent: #0f766e;
  --alpro-accent-soft: #ecfdf5;
}

.stApp {
  background: var(--alpro-bg);
  color: var(--alpro-text);
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 2rem;
  max-width: 1250px;
}

h1, h2, h3 {
  color: var(--alpro-text) !important;
  letter-spacing: -0.01em;
}

p, label, .stMarkdown, .stCaption, .stText, .stSelectbox label, .stSlider label, .stCheckbox label {
  color: var(--alpro-text);
}

small, .stCaption {
  color: var(--alpro-muted) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--alpro-surface);
  border-right: 1px solid var(--alpro-border);
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 1.5rem;
}

/* Cards */
.alpro-card {
  background: #ffffff;
  border: 1px solid var(--alpro-border);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
}
.alpro-card-muted {
  background: var(--alpro-surface);
  border: 1px solid var(--alpro-border);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
}
.alpro-card-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--alpro-text);
  margin-bottom: 0.25rem;
}
.alpro-help {
  color: var(--alpro-muted);
  font-size: 0.86rem;
  line-height: 1.45;
  margin-top: -0.2rem;
  margin-bottom: 0.65rem;
}

/* Metrics */
div[data-testid="stMetric"] {
  background: var(--alpro-surface);
  border: 1px solid var(--alpro-border);
  border-radius: 12px;
  padding: 0.75rem 0.9rem;
}
div[data-testid="stMetricLabel"] p {
  color: var(--alpro-muted) !important;
  font-size: 0.85rem;
}
div[data-testid="stMetricValue"] {
  color: var(--alpro-text) !important;
}

/* Buttons */
.stButton > button[kind="primary"] {
  background: var(--alpro-accent);
  border: 1px solid var(--alpro-accent);
  color: #ffffff;
  border-radius: 10px;
  font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
  background: #115e59;
  border-color: #115e59;
  color: #ffffff;
}
.stButton > button {
  border-radius: 10px;
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-testid="stFileUploaderDropzone"] {
  border-radius: 10px !important;
  border-color: var(--alpro-border) !important;
  background: #ffffff !important;
}
div[data-testid="stFileUploaderDropzone"] {
  background: var(--alpro-surface) !important;
}

/* Expanders */
details {
  border: 1px solid var(--alpro-border) !important;
  border-radius: 12px !important;
  background: #ffffff;
}
details summary p {
  font-weight: 600;
}

/* Info/warn/error soften */
div[data-testid="stAlert"] {
  border-radius: 10px;
}

/* Small badge */
.alpro-badge {
  display: inline-block;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  line-height: 1.4;
  background: var(--alpro-accent-soft);
  color: #065f46;
  border: 1px solid #a7f3d0;
}
</style>
""",
        unsafe_allow_html=True,
    )
