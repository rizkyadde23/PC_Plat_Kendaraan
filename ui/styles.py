"""Custom Streamlit styles for ALPRO - White Modern AI theme."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def section_head(title: str, heading: str, caption: Optional[str] = None) -> None:
    """Centered section header shared by every page (label + heading + caption)."""
    cap = f'<p class="alpro-section-caption">{caption}</p>' if caption else ""
    st.markdown(
        f'<div class="alpro-section-head">'
        f'<div class="alpro-section-title">{title}</div>'
        f"<h2>{heading}</h2>{cap}</div>",
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --alpro-bg: #ffffff;
  --alpro-bg-alt: #f8fafc;
  --alpro-card: #ffffff;
  --alpro-border: #e2e8f0;
  --alpro-text: #0f172a;
  --alpro-muted: #64748b;
  --alpro-primary: #2563eb;
  --alpro-primary-hover: #1d4ed8;
  --alpro-primary-dark: #1e3a8a;
  --alpro-teal: #0d9488;
  --alpro-teal-soft: #f0fdfa;
  --alpro-blue-soft: #eff6ff;
  --alpro-success: #16a34a;
  --alpro-warning: #d97706;
  --alpro-error: #dc2626;
  --alpro-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  --alpro-radius: 16px;

  /* Type scale */
  --fs-display: clamp(2rem, 3.6vw, 3rem);
  --fs-h1: clamp(1.9rem, 3vw, 2.5rem);
  --fs-h2: clamp(1.6rem, 2.6vw, 2.2rem);
  --fs-h4: 1.1rem;
  --fs-body: 1rem;
  --fs-small: 0.9rem;
  --fs-label: 1.1rem;
}

html, body, .stApp, [class*="css"] {
  font-family: Inter, "Plus Jakarta Sans", Manrope, "Segoe UI", sans-serif;
}
.stApp {
  background: var(--alpro-bg);
  color: var(--alpro-text);
  font-size: var(--fs-body);
}
/* Extra top space so content is never clipped by the Streamlit app bar */
.block-container {
  padding-top: 4.75rem !important;
  padding-bottom: 2.5rem;
  max-width: 1240px;
}
h1, h2, h3, h4 {
  color: var(--alpro-text) !important;
  letter-spacing: -0.02em;
  line-height: 1.2;
}
h1 { font-size: var(--fs-h1) !important; font-weight: 800 !important; }
h2 { font-size: var(--fs-h2) !important; font-weight: 800 !important; }
h4 { font-size: var(--fs-h4) !important; font-weight: 700 !important; }
p, li, label, .stMarkdown, .stCaption, .stText {
  color: var(--alpro-text);
}
small, .stCaption, .alpro-muted {
  color: var(--alpro-muted) !important;
}

/* Hide sidebar - navigation uses in-page buttons */
section[data-testid="stSidebar"] {
  display: none !important;
}
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
  display: none !important;
}

/* Buttons */
.stButton > button {
  border-radius: 10px;
  font-weight: 600;
  border: 1px solid var(--alpro-border);
  min-height: 2.75rem;
}
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
  background: var(--alpro-primary) !important;
  border-color: var(--alpro-primary) !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
  background: var(--alpro-primary-hover) !important;
  border-color: var(--alpro-primary-hover) !important;
}
/* White text on every primary button (including inner <p>/<span>/icon) */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] *,
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] * {
  color: #ffffff !important;
  fill: #ffffff !important;
}
.stButton > button[kind="primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled {
  opacity: 0.55;
}
.stButton > button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
  background: #fff;
  color: var(--alpro-primary);
}

/* Big CTA buttons (wrapped with st.container(key="bigbtn-...")) */
[class*="st-key-bigbtn"] .stButton > button {
  min-height: 3.5rem;
  font-size: 1.05rem;
  font-weight: 700;
  border-radius: 12px;
  padding: 0.8rem 1.75rem;
  box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
}
[class*="st-key-bigbtn"] .stButton > button:hover {
  box-shadow: 0 6px 18px rgba(37, 99, 235, 0.32);
  transform: translateY(-1px);
}
.st-key-bigbtn-hero { max-width: 340px; }
.st-key-bigbtn-cta { max-width: 420px; margin: 0 auto; }
.st-key-bigbtn-run .stButton > button { margin-top: 0.25rem; }

/* Top navigation buttons on inner pages */
.st-key-nav-back .stButton > button,
.st-key-nav-config .stButton > button {
  min-height: 2.9rem;
  font-weight: 700;
  border-radius: 10px;
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"],
div[data-testid="stFileUploaderDropzone"] {
  border-radius: 10px !important;
  border-color: var(--alpro-border) !important;
  background: #fff !important;
}
div[data-testid="stFileUploaderDropzone"] {
  background: var(--alpro-bg-alt) !important;
}
div[data-testid="stFileUploaderDropzone"] * {
  color: var(--alpro-muted);
}

/* Cards (st.container(key="card-...") / key="soft-...") */
[class*="st-key-card-"] {
  background: var(--alpro-card);
  border: 1px solid var(--alpro-border);
  border-radius: var(--alpro-radius);
  padding: 1.25rem;
  box-shadow: var(--alpro-shadow);
  margin-bottom: 1rem;
}
[class*="st-key-soft-"] {
  background: var(--alpro-bg-alt);
  border: 1px solid var(--alpro-border);
  border-radius: var(--alpro-radius);
  padding: 1.25rem;
  margin-bottom: 1rem;
}
.alpro-card {
  background: var(--alpro-card);
  border: 1px solid var(--alpro-border);
  border-radius: var(--alpro-radius);
  padding: 1.25rem;
  box-shadow: var(--alpro-shadow);
  margin-bottom: 1rem;
}
.alpro-card-soft {
  background: var(--alpro-bg-alt);
  border: 1px solid var(--alpro-border);
  border-radius: var(--alpro-radius);
  padding: 1.25rem;
  margin-bottom: 1rem;
}

/* Section header: centered, larger label */
.alpro-section-head {
  text-align: center;
  margin: 0.5rem auto 1.5rem auto;
  max-width: 760px;
}
.alpro-section-title {
  font-size: var(--fs-label);
  font-weight: 800;
  color: var(--alpro-primary);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  text-align: center;
  margin-bottom: 0.4rem;
}
.alpro-section-head h2 {
  margin: 0 !important;
  padding: 0 !important;
  text-align: center;
}
.alpro-section-caption {
  color: var(--alpro-muted) !important;
  font-size: var(--fs-body);
  line-height: 1.6;
  margin: 0.6rem 0 0 0;
  text-align: center;
}
.alpro-card h3, .alpro-card h4, .alpro-card-soft h3 {
  margin-top: 0;
  margin-bottom: 0.35rem;
}

/* Hero */
.alpro-hero {
  padding: 1.5rem 0 1rem 0;
}
.alpro-hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 0.9rem;
  background: var(--alpro-blue-soft);
  color: var(--alpro-primary);
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  margin-bottom: 1.1rem;
}
.alpro-hero h1 {
  font-size: var(--fs-display) !important;
  line-height: 1.15;
  margin: 0 0 1rem 0 !important;
  padding: 0 !important;
  font-weight: 800 !important;
}
.alpro-hero h1 .accent { color: var(--alpro-primary); }
.alpro-hero p {
  font-size: 1.1rem;
  color: var(--alpro-muted);
  max-width: 42rem;
  line-height: 1.75;
  margin-bottom: 1.5rem;
}
.alpro-hero-visual {
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid var(--alpro-border);
  border-radius: 24px;
  padding: 1.5rem;
  min-height: 320px;
  position: relative;
  overflow: hidden;
}
.alpro-visual-frame {
  background: #fff;
  border: 1px solid #dbeafe;
  border-radius: 18px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(37, 99, 235, 0.06);
}
.alpro-visual-car {
  background: var(--alpro-bg-alt);
  border: 1px dashed #cbd5e1;
  border-radius: 14px;
  height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  color: var(--alpro-muted);
  font-weight: 600;
}
.alpro-bbox {
  position: absolute;
  bottom: 28px;
  left: 50%;
  transform: translateX(-50%);
  width: 180px;
  height: 54px;
  border: 3px solid var(--alpro-primary);
  border-radius: 8px;
  background: rgba(37, 99, 235, 0.06);
}
.alpro-bbox-label {
  position: absolute;
  top: -12px;
  left: 10px;
  background: var(--alpro-primary);
  color: #fff;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
}
.alpro-visual-result {
  margin-top: 1rem;
  background: var(--alpro-blue-soft);
  border: 1px solid #bfdbfe;
  border-radius: 12px;
  padding: 0.85rem 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.alpro-visual-result .plate { font-size: 1.45rem; font-weight: 800; color: var(--alpro-text); letter-spacing: 0.03em; }
.alpro-visual-result .status { font-size: 0.78rem; color: var(--alpro-success); font-weight: 700; background: #f0fdf4; padding: 0.25rem 0.55rem; border-radius: 999px; border: 1px solid #bbf7d0; }

/* Feature/step cards */
.alpro-grid-card {
  background: #fff;
  border: 1px solid var(--alpro-border);
  border-radius: 16px;
  padding: 1.25rem;
  height: 100%;
  box-shadow: var(--alpro-shadow);
}
.alpro-icon {
  width: 46px; height: 46px; border-radius: 12px;
  display: inline-flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 1rem; margin-bottom: 0.85rem;
  background: var(--alpro-blue-soft); color: var(--alpro-primary);
}
.alpro-icon.teal { background: var(--alpro-teal-soft); color: var(--alpro-teal); }
.alpro-grid-card h4 { margin: 0 0 0.4rem 0 !important; font-size: 1.1rem !important; }
.alpro-grid-card p { margin: 0; color: var(--alpro-muted); font-size: var(--fs-small); line-height: 1.6; }

/* Tech chip */
.alpro-tech {
  background: #fff; border: 1px solid var(--alpro-border); border-radius: 14px; padding: 1.1rem 1rem;
  text-align: center; height: 100%;
}
.alpro-tech .name { font-weight: 700; font-size: 1.1rem; color: var(--alpro-text); margin-bottom: 0.25rem; }
.alpro-tech .desc { font-size: 0.86rem; color: var(--alpro-muted); line-height: 1.5; }

/* CTA */
.alpro-cta {
  background: var(--alpro-bg-alt);
  border: 1px solid var(--alpro-border);
  border-radius: 20px;
  padding: 2.5rem 1.5rem;
  text-align: center;
}
.alpro-cta h2 { margin: 0 0 0.5rem 0 !important; }
.alpro-cta p { color: var(--alpro-muted); margin-bottom: 0; font-size: 1.05rem; }

/* Result panel - dominant */
.alpro-result-panel {
  background: #fff;
  border: 1px solid #bfdbfe;
  border-radius: 20px;
  padding: 1.75rem;
  box-shadow: 0 8px 30px rgba(37, 99, 235, 0.06);
  margin-bottom: 1.25rem;
  position: relative;
  overflow: hidden;
  width: 100%;
}
.alpro-result-panel::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, var(--alpro-primary), #60a5fa);
}
.alpro-result-label {
  font-size: 0.85rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--alpro-primary);
  margin-bottom: 0.5rem;
}
.alpro-plate-text {
  font-size: clamp(2.2rem, 5.5vw, 3.8rem);
  font-weight: 800;
  letter-spacing: 0.04em;
  color: var(--alpro-text);
  line-height: 1.1;
  margin: 0.25rem 0 0.9rem 0;
  word-break: break-word;
}
.alpro-badge-status {
  display: inline-flex; align-items: center;
  padding: 0.35rem 0.8rem; border-radius: 999px; font-size: 0.82rem; font-weight: 700;
  border: 1px solid transparent;
}
.alpro-badge-success { background: #f0fdf4; color: #166534; border-color: #bbf7d0; }
.alpro-badge-warn { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.alpro-badge-error { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.alpro-badge-info { background: var(--alpro-blue-soft); color: #1d4ed8; border-color: #bfdbfe; }

/* Highlighted metrics: stretched edge-to-edge across the card */
.alpro-result-meta {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.9rem;
  width: 100%;
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--alpro-border);
}
.alpro-meta-item {
  background: var(--alpro-blue-soft);
  border: 1px solid #bfdbfe;
  border-radius: 16px;
  padding: 1.1rem 0.8rem;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 0;
}
.alpro-meta-item .k {
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--alpro-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.alpro-meta-item .v {
  font-size: clamp(1.5rem, 2.8vw, 2.4rem);
  font-weight: 800;
  line-height: 1.15;
  color: var(--alpro-primary);
  margin-top: 0.35rem;
  word-break: break-word;
}
.alpro-meta-item.strong {
  background: linear-gradient(135deg, var(--alpro-primary), var(--alpro-primary-hover));
  border-color: var(--alpro-primary);
  box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
}
.alpro-meta-item.strong .k { color: #dbeafe; }
.alpro-meta-item.strong .v { color: #ffffff; }

/* Metrics */
div[data-testid="stMetric"] {
  background: var(--alpro-bg-alt);
  border: 1px solid var(--alpro-border);
  border-radius: 12px;
  padding: 0.8rem 1rem;
}
div[data-testid="stMetricLabel"] p { color: var(--alpro-muted) !important; }
div[data-testid="stMetricValue"] { color: var(--alpro-text) !important; font-size: 1.4rem; }

/* Details/expander */
details {
  border: 1px solid var(--alpro-border) !important;
  border-radius: 14px !important;
  background: #fff;
  margin-bottom: 0.75rem;
}
details summary p { font-weight: 600; color: var(--alpro-text); }

/* Alerts */
div[data-testid="stAlert"] { border-radius: 12px; }

/* Dialog */
div[role="dialog"] {
  border-radius: 20px !important;
  border: 1px solid var(--alpro-border);
}
div[role="dialog"] > div { background: #fff; }

/* File uploader */
[data-testid="stFileUploader"] section {
  padding: 1rem;
}

/* Small helpers */
.alpro-divider { height: 1px; background: var(--alpro-border); margin: 1.25rem 0; }
.alpro-muted-text { color: var(--alpro-muted); font-size: var(--fs-small); line-height: 1.65; }
.alpro-kicker { color: var(--alpro-primary); font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; }

/* Responsive */
@media (max-width: 900px) {
  .alpro-result-meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 768px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 4.5rem !important; }
  .alpro-hero { padding-top: 0.5rem; }
  .alpro-result-panel { padding: 1.2rem; }
}
</style>
""",
        unsafe_allow_html=True,
    )
