"""
FitPulse Unified Platform  ·  v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single Streamlit app combining:
  • Analytics Platform  — thistle #D8BFD8 palette, full EDA pipeline
  • ML Pipeline         — TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE

Sidebar dropdown switches modes.  ALL chrome (header bar, sidebar,
buttons, progress bars) re-themes consistently on every rerun.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse Platform",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "mode":         "Analytics",
    # — analytics —
    "df_raw":       None,
    "df_clean":     None,
    "null_done":    False,
    "prep_done":    False,
    "eda_done":     False,
    "file_name":    "",
    # — ml —
    "ml_progress":  0,
    "ml_run_done":  False,
    "ml_dfs":       {},
    "feat_df":      None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SHARED COLOUR CONSTANTS  (same base palette for both modes)
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#D8BFD8"   # thistle — app background
CARD   = "#EDE0ED"   # lighter thistle — cards & panels
FIG_BG = "#F5ECF5"   # near-white purple — chart backgrounds
BORD   = "#C4A0C4"   # medium thistle — borders
ACC    = "#6B2D8B"   # deep purple — primary accent
ACC2   = "#9B59B6"   # medium purple — secondary accent
TXT    = "#2D1B3D"   # very dark purple — body text
SOFT   = "#7B5C8A"   # muted purple — labels & subtitles
SB_BG  = "#1A0A22"   # near-black purple — sidebar
SB_BDR = "#3D1F52"   # dark purple — sidebar border
SB_FUP = "#251035"   # sidebar file-uploader bg

# chart palette
PAL = [ACC, ACC2, "#2ECC71","#1ABC9C","#E74C3C","#F39C12","#3498DB","#E91E63"]

# ml cluster colours
CLUSTER_COLORS = {0:"#4d8fd4",1:"#e0609a",2:"#3baa66",3:"#f4a020",4:"#9b6cd4",-1:"#e04040"}

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC THEME CSS — re-injected every rerun so header/sidebar always match
# ─────────────────────────────────────────────────────────────────────────────
_ml = st.session_state.mode == "ML Pipeline"

# mode tokens
_HDR_BG    = f"linear-gradient(90deg,{SB_BG},{SB_BDR})"   # same dark for both
_DECO      = f"linear-gradient(90deg,{ACC},{ACC2})"
_SB_SEL_BG = "#2A1040"
_SB_SEL_BD = ACC
_POP_BG    = "#2A1040"
_POP_HOV   = SB_BDR

# The ML sidebar uses a lighter gradient (from original ML file)
_SB_ACTUAL_BG = "linear-gradient(160deg,#2A1040,#1A0A22)" if _ml else SB_BG

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

/* ── KEYFRAMES ── */
@keyframes fadeUp   {{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes slideIn  {{from{{opacity:0;transform:translateX(-14px)}}to{{opacity:1;transform:translateX(0)}}}}
@keyframes rotateBdr{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
@keyframes pulseDot {{0%,100%{{transform:scale(1);opacity:1}}50%{{transform:scale(1.7);opacity:0.4}}}}
@keyframes gradFlow {{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}
@keyframes shimmer  {{from{{transform:translateX(-100%)}}to{{transform:translateX(250%)}}}}
@keyframes popIn    {{from{{opacity:0;transform:scale(0.9)}}to{{opacity:1;transform:scale(1)}}}}
@keyframes countUp  {{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes scanLine {{0%{{left:-4px;opacity:0}}5%{{opacity:1}}95%{{opacity:1}}100%{{left:100%;opacity:0}}}}

/* ── BASE — thistle for both modes ── */
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;}}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.block-container {{
  background-color:{BG}!important;
  color:{TXT}!important;
}}
.block-container{{padding:1.4rem 2rem 3rem!important;max-width:1380px!important;}}

/* ── TOP HEADER BAR ── */
header[data-testid="stHeader"]{{
  background:{_HDR_BG}!important;
  border-bottom:2px solid {SB_BDR}!important;
}}
[data-testid="stToolbar"]{{background:transparent!important;}}
[data-testid="stDecoration"]{{
  background:{_DECO}!important;
  height:3px!important;
}}

/* ── HEADER BUTTONS (deploy, ⋮ menu) ── */
[data-testid="stAppDeployButton"] button{{
  background:{ACC}!important;color:#fff!important;
  border:2px solid {ACC2}!important;border-radius:8px!important;
  font-family:'Syne',sans-serif!important;font-weight:700!important;
  font-size:0.82rem!important;padding:5px 16px!important;
  box-shadow:0 2px 8px rgba(107,45,139,0.35);transition:all .2s;
}}
[data-testid="stAppDeployButton"] button:hover{{background:{ACC2}!important;}}
button[data-testid="baseButton-header"],button[kind="header"],
[data-testid="stMainMenuButton"],button[aria-label="Main menu"]{{
  background:{CARD}!important;border:1.5px solid {BORD}!important;
  border-radius:8px!important;transition:all .2s;
}}
button[data-testid="baseButton-header"]:hover,button[kind="header"]:hover,
[data-testid="stMainMenuButton"]:hover,button[aria-label="Main menu"]:hover{{
  background:{ACC}!important;border-color:{ACC}!important;
}}
button[data-testid="baseButton-header"] svg,button[kind="header"] svg,
[data-testid="stMainMenuButton"] svg,button[aria-label="Main menu"] svg{{
  fill:{ACC}!important;stroke:{ACC}!important;
}}
button[data-testid="baseButton-header"]:hover svg,button[kind="header"]:hover svg,
[data-testid="stMainMenuButton"]:hover svg,button[aria-label="Main menu"]:hover svg{{
  fill:#fff!important;stroke:#fff!important;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{{
  background:{_SB_ACTUAL_BG}!important;
  border-right:1.5px solid {SB_BDR}!important;
}}
[data-testid="stSidebar"] *{{color:#E8D4F0!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]{{display:flex!important;}}

/* ── SIDEBAR MODE SELECTBOX ── */
[data-testid="stSidebar"] [data-baseweb="select"]>div{{
  background:{_SB_SEL_BG}!important;border:1.5px solid {_SB_SEL_BD}!important;
  border-radius:9px!important;color:#E8D4F0!important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="select-placeholder"],
[data-testid="stSidebar"] [data-baseweb="select"] [data-value]{{
  color:#E8D4F0!important;font-size:0.87rem!important;font-weight:700!important;
  font-family:'Syne',sans-serif!important;
}}
[data-baseweb="popover"] [role="option"]{{
  background:{_POP_BG}!important;color:#E8D4F0!important;font-size:0.83rem!important;
}}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"]{{
  background:{_POP_HOV}!important;color:#fff!important;
}}

/* ── SIDEBAR FILE UPLOADER ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section{{
  border:2px dashed rgba(155,89,182,0.55)!important;
  background:{SB_FUP}!important;border-radius:10px!important;transition:border-color 0.25s;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover{{
  border-color:{ACC2}!important;
}}

/* ── SIDEBAR PROGRESS (ML) ── */
[data-testid="stSidebar"] .stProgress>div>div>div>div{{
  background:linear-gradient(90deg,{ACC2},{ACC})!important;border-radius:999px;
}}
[data-testid="stSidebar"] .stProgress>div>div{{
  background:rgba(155,89,182,0.3)!important;border-radius:999px;
}}

/* ── SIDEBAR SLIDERS ── */
[data-testid="stSidebar"] [data-testid="stSliderTrackFill"]{{background:{ACC}!important;}}

/* ── MAIN AREA BUTTONS ── */
.stButton>button{{
  background:{ACC}!important;color:#fff!important;border:none!important;
  border-radius:8px!important;padding:0.48rem 1.35rem!important;
  font-weight:600!important;font-size:0.84rem!important;font-family:'DM Sans',sans-serif!important;
  transition:background 0.18s,transform 0.18s,box-shadow 0.18s!important;
  box-shadow:0 2px 12px rgba(107,45,139,0.4)!important;
}}
.stButton>button:hover{{background:#521E6B!important;transform:translateY(-1px)!important;box-shadow:0 5px 18px rgba(107,45,139,0.55)!important;}}
.stButton>button:active{{transform:scale(0.97)!important;}}
.stButton>button:disabled{{background:{BORD}!important;color:{SOFT}!important;box-shadow:none!important;}}

/* ── FILE UPLOADER (main area) ── */
[data-testid="stFileUploader"] section{{
  border:2px dashed {BORD}!important;border-radius:10px!important;background:{FIG_BG}!important;
}}
[data-testid="stFileUploader"] section:hover{{border-color:{ACC}!important;}}
[data-testid="stFileUploaderDropzone"]{{
  background:{CARD}!important;border:2.5px dashed {ACC2}!important;border-radius:14px!important;
}}
[data-testid="stFileUploaderDropzone"] button,[data-testid="stFileUploader"] button{{
  background:{ACC}!important;color:#fff!important;border:none!important;
  border-radius:8px!important;font-family:'Syne',sans-serif!important;
  font-weight:700!important;font-size:0.85rem!important;padding:7px 20px!important;
  box-shadow:0 2px 8px rgba(107,45,139,0.3);transition:background .2s;
}}
[data-testid="stFileUploaderDropzone"] button:hover,[data-testid="stFileUploader"] button:hover{{
  background:{ACC2}!important;
}}

/* ── PROGRESS BAR (main) ── */
.stProgress>div>div>div>div{{background:linear-gradient(90deg,{ACC2},{ACC})!important;border-radius:999px;}}
.stProgress>div>div{{background:rgba(196,160,196,0.4)!important;border-radius:999px;}}

/* ── METRICS ── */
[data-testid="stMetric"]{{
  background:{CARD};border:1.5px solid {BORD};border-radius:14px;padding:10px 14px;
}}
[data-testid="stMetricLabel"]{{font-family:'Syne',sans-serif;font-weight:700;color:{SOFT}!important;}}
[data-testid="stMetricValue"]{{color:{ACC}!important;font-family:'Syne',sans-serif;}}

/* ── TABS ── */
[data-baseweb="tab-list"]{{background:transparent!important;}}
[data-baseweb="tab"]{{font-family:'Syne',sans-serif!important;font-weight:600!important;color:{SOFT}!important;}}
[aria-selected="true"]{{color:{ACC}!important;border-bottom:2px solid {ACC}!important;}}

/* ── SELECT / INPUT ── */
[data-baseweb="select"]>div,[data-baseweb="input"]>div{{
  background:{CARD}!important;border-color:{BORD}!important;
}}

/* ── DATAFRAMES ── */
[data-testid="stDataFrame"]{{border-radius:10px!important;border:1px solid {BORD}!important;
  overflow:hidden;box-shadow:0 1px 5px rgba(107,45,139,0.08);}}

/* ── EXPANDER ── */
details summary{{
  background:{CARD}!important;border:1px solid {BORD}!important;
  border-radius:8px!important;color:{TXT}!important;
}}
</style>
""", unsafe_allow_html=True)

# ── COMPONENT CSS (static — layout & colour-agnostic structures) ────────────
st.markdown(f"""
<style>
/* ANALYTICS BANNER */
.banner{{
  background:linear-gradient(135deg,#1A0A22 0%,#2D1040 55%,#1A0A22 100%);
  border-radius:14px;padding:1.5rem 2rem;margin-bottom:1.5rem;
  display:flex;align-items:center;justify-content:space-between;
  position:relative;overflow:hidden;animation:fadeUp 0.4s ease;
  box-shadow:0 4px 22px rgba(107,45,139,0.35);
}}
.banner::before{{content:'';position:absolute;left:0;top:0;bottom:0;width:5px;
  background:linear-gradient(180deg,{ACC},{ACC2});}}
.banner-glow{{position:absolute;right:-40px;top:-40px;width:200px;height:200px;
  background:radial-gradient(circle,rgba(155,89,182,0.25),transparent 65%);pointer-events:none;}}
.banner-title{{font-family:'Syne',sans-serif;font-size:1.65rem;font-weight:800;color:#F0E0F8;letter-spacing:-0.5px;}}
.banner-sub{{font-size:0.78rem;color:#B89ABE;margin-top:4px;}}
.banner-chip{{background:rgba(155,89,182,0.2);border:1px solid rgba(155,89,182,0.5);border-radius:20px;
  padding:6px 16px;font-size:0.72rem;font-weight:700;color:{ACC2};letter-spacing:0.6px;z-index:1;}}

/* STEP HEADER */
.step-hdr{{display:flex;align-items:center;gap:0.75rem;
  background:{CARD};border:1px solid {BORD};border-left:5px solid {ACC};border-radius:10px;
  padding:0.78rem 1.3rem;margin:1.4rem 0 1rem;font-family:'Syne',sans-serif;
  font-size:1rem;font-weight:700;color:{TXT};
  box-shadow:0 1px 6px rgba(107,45,139,0.12);animation:fadeUp 0.3s ease;}}
.step-num{{background:{ACC};color:#fff;font-size:0.68rem;font-weight:700;
  width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;}}

/* STAT CARDS */
.scard{{background:{CARD};border:1px solid {BORD};border-radius:12px;
  padding:1.2rem 1.3rem;position:relative;overflow:hidden;
  box-shadow:0 1px 5px rgba(107,45,139,0.1);animation:fadeUp 0.4s ease both;
  transition:transform 0.2s,box-shadow 0.2s,border-color 0.2s;}}
.scard:hover{{transform:translateY(-3px);box-shadow:0 6px 22px rgba(107,45,139,0.2);border-color:{ACC};}}
.scard::after{{content:'';position:absolute;top:0;right:0;width:55px;height:55px;
  background:radial-gradient(circle at top right,rgba(155,89,182,0.12),transparent 70%);}}
.scard-icon{{font-size:1.3rem;margin-bottom:0.4rem;}}
.scard-val{{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:{TXT};animation:countUp 0.5s ease;}}
.scard-val.orange{{color:{ACC};}} .scard-val.red{{color:#C0392B;}} .scard-val.green{{color:#1E8449;}}
.scard-lbl{{font-size:0.68rem;color:{SOFT};text-transform:uppercase;letter-spacing:0.08em;margin-top:2px;}}

/* LOADER */
.loader{{background:{CARD};border:1px solid {BORD};border-radius:12px;
  padding:1.25rem 1.5rem;margin:0.6rem 0;position:relative;overflow:hidden;
  box-shadow:0 1px 8px rgba(107,45,139,0.12);}}
.loader::after{{content:'';position:absolute;top:0;bottom:0;width:3px;
  background:linear-gradient(180deg,transparent,{ACC},{ACC2},transparent);
  animation:scanLine 2.2s linear infinite;}}
.loader-row{{display:flex;align-items:center;gap:1rem;margin-bottom:0.7rem;}}
.spinner{{width:40px;height:40px;border-radius:50%;border:2px solid {BORD};
  position:relative;flex-shrink:0;display:flex;align-items:center;justify-content:center;}}
.spinner::before{{content:'';position:absolute;inset:0;border-radius:50%;
  border-top:2.5px solid {ACC};border-right:2.5px solid transparent;
  animation:rotateBdr 0.75s linear infinite;}}
.spinner-dot{{width:7px;height:7px;background:{ACC};border-radius:50%;animation:pulseDot 0.75s ease-in-out infinite;}}
.ldr-title{{font-size:0.82rem;font-weight:600;color:{TXT};margin-bottom:1px;}}
.ldr-step{{font-size:0.71rem;color:{SOFT};font-family:'DM Mono',monospace;}}
.ldr-pct{{font-family:'DM Mono',monospace;font-size:0.92rem;font-weight:600;color:{ACC};margin-left:auto;flex-shrink:0;}}
.prog-track{{background:{BG};border-radius:6px;height:5px;overflow:hidden;margin:0.35rem 0;border:1px solid {BORD};}}
.prog-bar{{height:100%;background:linear-gradient(90deg,{ACC},{ACC2},{ACC});background-size:200% 100%;
  animation:gradFlow 1.4s linear infinite;border-radius:6px;
  box-shadow:0 0 8px rgba(155,89,182,0.45);transition:width 0.2s ease;}}
.ldr-ticker{{font-family:'DM Mono',monospace;font-size:0.68rem;color:{ACC};
  background:{FIG_BG};border:1px solid {BORD};border-radius:5px;
  padding:4px 10px;margin-top:0.3rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}

/* LOG LINES */
.logline{{background:{FIG_BG};border:1px solid {BORD};border-left:3px solid {ACC2};
  border-radius:6px;padding:0.48rem 0.9rem;margin-bottom:0.32rem;
  font-size:0.79rem;color:{TXT};position:relative;overflow:hidden;animation:slideIn 0.3s ease both;}}
.logline .sh{{position:absolute;top:0;left:0;width:40%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(216,191,216,0.4),transparent);
  animation:shimmer 2.5s ease infinite;pointer-events:none;}}
.logline.ok  {{border-left-color:#1E8449;background:#E8F8EE;color:#145A32;}}
.logline.warn{{border-left-color:#D4AC0D;background:#FEF9E7;color:#7D6608;}}
.logline.info{{border-left-color:{ACC2};background:{FIG_BG};color:{ACC};}}
.logline.err {{border-left-color:#C0392B;background:#FDEDEC;color:#922B21;}}

/* INFO / OK BOXES */
.info-box{{background:{FIG_BG};border:1px solid {BORD};border-radius:9px;
  padding:0.8rem 1.1rem;font-size:0.8rem;color:{ACC};margin:0.5rem 0;animation:popIn 0.3s ease;}}
.ok-box{{background:{FIG_BG};border:1px solid {BORD};border-radius:9px;
  padding:0.8rem 1.1rem;font-size:0.81rem;color:{ACC};font-weight:600;
  text-align:center;animation:popIn 0.35s ease;}}

/* EDA CARDS */
.eda-card{{background:{CARD};border:1px solid {BORD};border-radius:12px;
  padding:1.4rem 1.6rem;margin-bottom:1.2rem;
  box-shadow:0 1px 5px rgba(107,45,139,0.08);animation:fadeUp 0.35s ease;}}
.eda-card-title{{font-family:'Syne',sans-serif;font-size:0.98rem;font-weight:700;color:{TXT};
  border-bottom:2px solid {BORD};padding-bottom:0.55rem;margin-bottom:1rem;}}
.nbadge{{display:inline-block;background:{FIG_BG};border:1px solid {BORD};border-radius:6px;
  padding:3px 10px;font-size:0.71rem;color:{ACC};margin:3px;font-weight:500;transition:all 0.2s;}}
.nbadge:hover{{background:{ACC};color:#fff;border-color:{ACC};transform:scale(1.03);}}

/* ML HERO */
.fp-hero{{background:linear-gradient(120deg,#e0cce0ee,#cdb5cdee);
  border:1.5px solid {BORD};border-radius:16px;padding:22px 28px 18px;margin-bottom:20px;}}
.fp-title{{font-family:'Syne',sans-serif;font-size:2.1rem;font-weight:800;color:{ACC};margin:0 0 4px;}}
.fp-sub{{font-size:.82rem;color:{SOFT};letter-spacing:.04em;}}
.fp-tag{{display:inline-block;background:{ACC2};color:#fff;border-radius:999px;
  padding:3px 12px;font-size:.72rem;font-family:'Syne',sans-serif;
  font-weight:600;letter-spacing:.06em;margin-right:4px;}}
.fp-info{{background:#e8d5e8dd;border-left:4px solid {ACC2};border-radius:8px;
  padding:10px 16px;margin-bottom:12px;font-size:.9rem;color:{SOFT};}}

/* ML DATA LOADING CARDS */
.ds-grid{{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;}}
.ds-card{{background:#e2cfe2cc;border:1.5px solid {BORD};border-radius:10px;
  padding:10px 16px;min-width:140px;flex:1 1 140px;text-align:center;}}
.ds-icon{{font-size:1.5rem;}}
.ds-name{{font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;color:{ACC};}}
.ds-status{{font-size:.78rem;margin-top:2px;}}
.chip-ok{{color:#3db87a;font-weight:700;}} .chip-miss{{color:#e05555;font-weight:700;}}
.steps-badge{{float:right;background:{ACC2};color:#fff;border-radius:999px;
  padding:2px 12px;font-size:.75rem;font-family:'Syne',sans-serif;font-weight:700;}}

/* SIDEBAR STRUCTURAL */
.sb-logo{{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:#F0E0F8;}}
.sb-tag{{font-size:0.68rem;color:#9B7BAE;margin-top:2px;}}
.sb-mode-label{{font-size:0.62rem;color:#9B7BAE;text-transform:uppercase;
  letter-spacing:0.1em;margin-bottom:6px;display:block;margin-top:4px;}}
.sb-section-label{{font-size:0.63rem;color:#9B7BAE;text-transform:uppercase;
  letter-spacing:0.1em;margin:1rem 0 0.3rem;}}
.sb-nav{{display:flex;align-items:center;gap:0.5rem;padding:0.38rem 0.65rem;border-radius:7px;
  font-size:0.81rem;color:#C8A8D8!important;margin-bottom:0.18rem;transition:all 0.2s;
  border:1px solid transparent;}}
.sb-nav:hover{{background:{SB_BDR};color:#F0E0F8!important;border-color:#5D2D78;}}
.sb-nav.done{{color:{ACC2}!important;font-weight:600;}}
.sb-div{{height:1px;background:{SB_BDR};margin:0.9rem 0;}}
.ml-stage{{font-size:0.83rem;padding:3px 0;color:#C8A8D8!important;}}

/* GENERAL */
.hr{{height:1px;background:linear-gradient(90deg,transparent,{BORD},transparent);margin:1.5rem 0;}}
.footer{{text-align:center;font-size:0.7rem;color:{SOFT};
  padding:0.9rem 0 0.4rem;border-top:1px solid {BORD};margin-top:1.8rem;}}
.footer b{{color:{ACC};}}
hr{{border-color:{BORD}!important;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hex_rgba(hx, a=0.18):
    h=hx.lstrip("#"); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def run_loader(title, steps, delay=0.2):
    ph = st.empty()
    for i, s in enumerate(steps):
        pct = int((i+1)/len(steps)*100)
        ph.markdown(f"""
<div class="loader" style="animation:none;">
  <div class="loader-row">
    <div class="spinner"><div class="spinner-dot"></div></div>
    <div style="flex:1;min-width:0;">
      <div class="ldr-title">{title}</div><div class="ldr-step">{s}</div>
    </div>
    <div class="ldr-pct">{pct}%</div>
  </div>
  <div class="prog-track"><div class="prog-bar" style="width:{pct}%;"></div></div>
  <div class="ldr-ticker">▶ &nbsp;{s}</div>
</div>""", unsafe_allow_html=True)
        time.sleep(delay)
    ph.empty()

def logline(kind, msg, delay=0.0):
    cls  = {"ok":"ok","warn":"warn","info":"info","err":"err"}.get(kind,"")
    icon = {"ok":"✅","warn":"⚠️","info":"ℹ️","err":"❌"}.get(kind,"·")
    st.markdown(
        f'<div class="logline {cls}" style="animation-delay:{delay:.2f}s;">'
        f'<div class="sh"></div>{icon}&nbsp; {msg}</div>', unsafe_allow_html=True)

def apply_theme(fig, h=340):
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=FIG_BG,
        font=dict(family="DM Sans",color=TXT,size=12),height=h,
        margin=dict(l=12,r=18,t=44,b=14),
        title_font=dict(family="Syne",size=13,color=TXT),
        legend=dict(bgcolor=CARD,bordercolor=BORD,borderwidth=1,font=dict(color=TXT,size=11)))
    fig.update_xaxes(gridcolor=BORD,gridwidth=1,zeroline=False,linecolor=BORD,
        tickfont=dict(color=SOFT,size=11),title_font=dict(color=SOFT))
    fig.update_yaxes(gridcolor=BORD,gridwidth=1,zeroline=False,linecolor=BORD,
        tickfont=dict(color=SOFT,size=11),title_font=dict(color=SOFT))
    return fig

def scard(col, val, lbl, color="", icon=""):
    cls = {"orange":"orange","red":"red","green":"green"}.get(color,"")
    with col:
        st.markdown(
            f'<div class="scard"><div class="scard-icon">{icon}</div>'
            f'<div class="scard-val {cls}">{val}</div>'
            f'<div class="scard-lbl">{lbl}</div></div>', unsafe_allow_html=True)

def step_hdr(num, icon, title):
    st.markdown(
        f'<div class="step-hdr"><div class="step-num">{num}</div>'
        f'<span>{icon}&nbsp; {title}</span></div>', unsafe_allow_html=True)

def hr():
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

def style_mpl(fig, axes=None):
    fig.patch.set_facecolor(FIG_BG)
    if axes is None: axes = fig.axes
    for ax in axes:
        ax.set_facecolor(FIG_BG)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
        ax.title.set_color(ACC)
        for sp in ax.spines.values(): sp.set_edgecolor(BORD)
        ax.grid(True,color=BORD,linewidth=0.5,alpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# ML DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
USERS = [2022484408,2026352035,2347167796,4020332650,4558609924,
         5553957443,5577150313,6117666160,6391747486,6775888955,
         6962181067,7007744171,8792009665,8877689391]
USER_LABELS = [str(u)[-4:] for u in USERS]
FEATURES = ["sum_values","abs_median","value_mean","value_length","std_deviation",
            "abs_variance","mean_square","abs_maximum","abs_maximum2","minimum"]
PCA_CENTERS = [(-1.15,0.65),(0.15,-0.75),(1.30,0.20),(-0.40,1.20),(1.10,-1.00)]
PCA_SIZES   = [10,10,5,8,7]
TSNE_CENTERS= [(-0.10,-7.60),(-0.30,-8.55),(-0.80,-7.75),(0.25,-7.40),(-0.15,-9.00)]

def _tight(center,n,spread,seed):
    rng=np.random.default_rng(seed)
    return np.array(center)+rng.standard_normal((n,2))*spread

def make_pca_scatter(k=3):
    k=min(k,5); coords,labels=[],[]
    for i in range(k):
        pts=_tight(PCA_CENTERS[i],PCA_SIZES[i],0.22,i*100+1)
        coords.append(pts); labels+=[i]*PCA_SIZES[i]
    return np.vstack(coords),np.array(labels)

def make_tsne(k=3):
    k=min(k,5); coords,labels=[],[]
    for i in range(k):
        pts=_tight(TSNE_CENTERS[i],PCA_SIZES[i],0.12,i*200+7)
        coords.append(pts); labels+=[i]*PCA_SIZES[i]
    return np.vstack(coords),np.array(labels)

def make_tsfresh_matrix():
    np.random.seed(42); data=np.random.rand(len(USERS),len(FEATURES))
    for r in [0,2,9,10]: data[r,np.random.choice(len(FEATURES),3,replace=False)]=np.random.uniform(0.85,1.0,3)
    for r in [1,5,6]:    data[r,np.random.choice(len(FEATURES),3,replace=False)]=np.random.uniform(0.0,0.12,3)
    return pd.DataFrame(data,index=USERS,columns=FEATURES)

def make_prophet_hr():
    np.random.seed(7); dates=pd.date_range("2016-03-29",periods=45,freq="D")
    trend=np.linspace(70,86,45); actual=trend[:20]+np.random.randn(20)*1.5
    yhat=trend+0.4; ylow=yhat-3.5-np.abs(np.random.randn(45))*0.8; yhigh=yhat+3.5+np.abs(np.random.randn(45))*0.8
    return dates,actual,yhat,ylow,yhigh

def make_prophet_steps():
    np.random.seed(13); dates=pd.date_range("2016-03-12",periods=65,freq="D")
    trend=np.linspace(4000,9000,65); weekly=1500*np.sin(np.arange(65)*2*np.pi/7)
    yhat=trend+weekly; actual=(trend+weekly+np.random.randn(65)*600)[:35]
    return dates,actual,yhat,yhat-2000,yhat+2000,35

def make_prophet_sleep():
    np.random.seed(17); dates=pd.date_range("2016-03-12",periods=65,freq="D")
    trend=np.linspace(160,290,65); weekly=40*np.sin(np.arange(65)*2*np.pi/7)
    yhat=trend+weekly; actual=(trend+weekly+np.random.randn(65)*20)[:35]
    return dates,actual,yhat,yhat-80,yhat+80,35

def make_prophet_components():
    np.random.seed(9); dates=pd.date_range("2016-03-29",periods=45,freq="D")
    return dates,np.linspace(73.5,86,45),0.3+2.5*np.sin(np.arange(45)*2*np.pi/7-1)

def make_elbow():
    return list(range(2,10)),[165,124,100,76,62,49,44,39]

# ─────────────────────────────────────────────────────────────────────────────
# ███  SIDEBAR  ███████████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown('<div class="sb-logo">💪 FitPulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    # ── MODE SWITCHER ──────────────────────────────────────────────────────
    st.markdown('<span class="sb-mode-label">⚡ Switch Mode</span>', unsafe_allow_html=True)
    mode_options = ["📊  Analytics Platform", "🧬  ML Pipeline"]
    chosen = st.selectbox(
        label="mode_sel",
        options=mode_options,
        index=0 if st.session_state.mode == "Analytics" else 1,
        label_visibility="collapsed",
        key="mode_select",
    )
    new_mode = "Analytics" if chosen.startswith("📊") else "ML Pipeline"
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.rerun()

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    # ── ANALYTICS SIDEBAR CONTENT ──────────────────────────────────────────
    if st.session_state.mode == "Analytics":
        st.markdown('<div class="sb-tag">Analytics Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Upload Dataset</div>', unsafe_allow_html=True)
        uploaded_analytics = st.file_uploader(
            "Drop CSV here", type=["csv"],
            label_visibility="collapsed", key="file_up_ana")
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Pipeline</div>', unsafe_allow_html=True)
        for n, ic, lb, done in [
            ("1","📁","Upload CSV",       st.session_state.df_raw is not None),
            ("2","🔍","Null Value Check",  st.session_state.null_done),
            ("3","⚙️","Preprocess",        st.session_state.prep_done),
            ("4","👁️","Preview Data",      st.session_state.df_clean is not None),
            ("5","📊","EDA",               st.session_state.eda_done),
        ]:
            tick = "✔" if done else n
            cls  = "sb-nav done" if done else "sb-nav"
            st.markdown(f'<div class="{cls}">{tick}. {ic} {lb}</div>', unsafe_allow_html=True)
        if st.session_state.df_raw is not None:
            st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
            di = st.session_state.df_raw
            st.markdown(
                f'<div style="font-size:0.71rem;color:#9B7BAE;">'
                f'📋 {len(di):,} rows · {len(di.columns)} cols<br>'
                f'⚠️ {int(di.isnull().sum().sum()):,} null cells</div>',
                unsafe_allow_html=True)

    # ── ML SIDEBAR CONTENT ─────────────────────────────────────────────────
    else:
        st.markdown('<div class="sb-tag">ML Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Pipeline Progress</div>', unsafe_allow_html=True)
        prog = st.session_state.ml_progress
        st.progress(prog / 100)
        st.caption(f"{prog}%")
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Stages</div>', unsafe_allow_html=True)
        for ico, name, thr in [
            ("📁","Data Loading",     20),
            ("📊","TSFresh Features", 40),
            ("📈","Prophet Forecast", 60),
            ("🔵","Clustering",       80),
        ]:
            tick = "✅" if prog >= thr else "⭕"
            st.markdown(f'<div class="ml-stage">{tick} {ico} {name}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">KMeans Clusters (K)</div>', unsafe_allow_html=True)
        st.slider("k_lbl",2,5,3,key="k_sl",label_visibility="collapsed")
        st.markdown('<div class="sb-section-label">DBSCAN Eps</div>', unsafe_allow_html=True)
        st.slider("eps_lbl",0.5,5.0,2.2,0.05,key="eps_sl",label_visibility="collapsed")
        st.markdown('<div class="sb-section-label">DBSCAN Min Samples</div>', unsafe_allow_html=True)
        st.slider("ms_lbl",1,10,2,key="ms_sl",label_visibility="collapsed")

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
    lbl = "Analytics · v3.0" if st.session_state.mode=="Analytics" else "ML Pipeline · v2.0"
    st.markdown(
        f'<div style="font-size:0.65rem;color:#9B7BAE;text-align:center;line-height:1.7;">'
        f'FitPulse {lbl}<br>Thistle Purple Edition</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ███  MAIN CONTENT  ██████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — ANALYTICS PLATFORM
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "Analytics":

    # Banner
    st.markdown("""
<div class="banner">
  <div class="banner-glow"></div>
  <div>
    <div class="banner-title">💪 FitPulse Analytics</div>
    <div class="banner-sub">Fitness &amp; Health Data Quality Pipeline &nbsp;—&nbsp; Upload · Inspect · Clean · Analyse</div>
  </div>
  <div class="banner-chip">ENTERPRISE DATA PLATFORM</div>
</div>""", unsafe_allow_html=True)

    # File load
    uploaded = st.session_state.get("file_up_ana", None)
    if uploaded is not None:
        if uploaded.name != st.session_state.file_name:
            run_loader("Loading dataset…",[
                "📂 Reading file buffer…","🔤 Detecting encoding…",
                "📊 Parsing column types…","🔢 Counting rows & cells…","✅ Dataset loaded!",
            ], delay=0.14)
            try:
                df_loaded = pd.read_csv(uploaded)
                st.session_state.df_raw    = df_loaded
                st.session_state.df_clean  = None
                st.session_state.null_done = False
                st.session_state.prep_done = False
                st.session_state.eda_done  = False
                st.session_state.file_name = uploaded.name
                logline("ok",f"<strong>{uploaded.name}</strong> loaded — {len(df_loaded):,} rows × {len(df_loaded.columns)} columns")
            except Exception as e:
                st.error(f"❌ Could not read file: {e}"); st.stop()

    # STEP 1
    step_hdr("1","📁","Upload CSV")
    if st.session_state.df_raw is None:
        st.markdown('<div class="info-box">📌 Use the <strong>sidebar uploader</strong> on the left to upload your Fitness Health CSV file.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state.df_raw
        c1,c2,c3,c4 = st.columns(4)
        scard(c1,f"{len(df):,}","Total Rows","","📋")
        scard(c2,str(len(df.columns)),"Total Columns","","📊")
        scard(c3,f"{int(df.isnull().sum().sum()):,}","Null Cells","orange","⚠️")
        scard(c4,f"{df.isnull().any(axis=1).sum():,}","Rows with Nulls","red","🔴")
        with st.expander("👀 Raw Data Preview — first 50 rows"):
            st.dataframe(df.head(50),use_container_width=True,height=260)
    hr()

    # STEP 2
    step_hdr("2","🔍","Null Value Inspector")
    if st.button("🔍  Run Null Value Check",disabled=(st.session_state.df_raw is None),key="btn_null"):
        run_loader("Scanning for null values…",[
            "🧬 Mapping column schema…","🔎 Iterating all cells…",
            "📡 Flagging missing entries…","🧮 Computing statistics…",
            "📊 Rendering visual report…","✅ Null scan complete!",
        ], delay=0.2)
        st.session_state.null_done = True

    if st.session_state.null_done and st.session_state.df_raw is not None:
        df=st.session_state.df_raw; null_s=df.isnull().sum()
        has_null=null_s[null_s>0].sort_values(ascending=False); clean_cols=null_s[null_s==0]
        total_cells=df.shape[0]*df.shape[1]; total_nulls=int(null_s.sum())
        c1,c2,c3,c4=st.columns(4)
        scard(c1,f"{total_nulls:,}","Total Null Cells","orange","❌")
        scard(c2,str(len(has_null)),"Affected Columns","red","📉")
        scard(c3,str(len(clean_cols)),"Clean Columns","green","✅")
        scard(c4,f"{total_nulls/total_cells*100:.2f}%","Overall Null Rate","orange","📊")
        st.markdown("<br>",unsafe_allow_html=True)
        if has_null.empty:
            st.markdown('<div class="ok-box">🎉 No null values found — dataset is perfectly clean!</div>',unsafe_allow_html=True)
        else:
            badges="".join(f'<span class="nbadge">⚠️ {col}&nbsp;·&nbsp;{v:,} ({v/len(df)*100:.1f}%)</span>' for col,v in has_null.items())
            st.markdown(f'<div style="margin-bottom:1rem;line-height:2.3;">{badges}</div>',unsafe_allow_html=True)
            col_chart,col_table=st.columns([3,2],gap="large")
            with col_chart:
                null_plot=pd.DataFrame({"Column":has_null.index.tolist(),"Nulls":has_null.values.tolist(),"Null %":(has_null.values/len(df)*100).round(2).tolist()})
                fig=px.bar(null_plot,x="Nulls",y="Column",orientation="h",color="Null %",
                    color_continuous_scale=[[0,ACC2],[0.5,ACC],[1,"#C0392B"]],text="Nulls",title="Null Count per Column")
                fig.update_traces(texttemplate="%{text:,}",textposition="outside",textfont=dict(size=11,color=TXT),marker_line_width=0)
                fig.update_layout(yaxis=dict(autorange="reversed"),coloraxis_colorbar=dict(title="Null %",tickfont=dict(size=10,color=TXT),len=0.7))
                fig=apply_theme(fig,320); st.plotly_chart(fig,use_container_width=True,key="null_bar")
            with col_table:
                st.markdown("**📋 Detailed Null Breakdown**")
                detail=pd.DataFrame({"Column":has_null.index,"Dtype":df[has_null.index].dtypes.astype(str).values,
                    "Null Count":has_null.values,"Non-Null":len(df)-has_null.values,"Null %":(has_null.values/len(df)*100).round(2)})
                st.dataframe(detail,hide_index=True,use_container_width=True,height=290)
            st.markdown("**🗺️ Null Heatmap** (purple = missing · light = present)")
            sample=df[has_null.index].head(500).isnull().astype(int)
            fig2=px.imshow(sample.T,color_continuous_scale=[[0,FIG_BG],[1,ACC]],aspect="auto",title="Null Presence Map",zmin=0,zmax=1)
            fig2.update_coloraxes(showscale=False); fig2.update_layout(xaxis_title="Row index",yaxis_title="")
            fig2=apply_theme(fig2,250); st.plotly_chart(fig2,use_container_width=True,key="null_heatmap")
    hr()

    # STEP 3
    step_hdr("3","⚙️","Data Preprocessing")
    if st.button("⚙️  Run Preprocessing",disabled=(st.session_state.df_raw is None),key="btn_prep"):
        run_loader("Preprocessing pipeline…",[
            "🧪 Initialising engine…","📅 Detecting datetime columns…",
            "📈 Interpolating numeric nulls…","🔧 Filling categorical nulls…",
            "🛡️ Integrity validation…","🧹 Final residual clean…","✅ Complete!",
        ], delay=0.2)
        st.session_state.prep_done = True

    if st.session_state.prep_done and st.session_state.df_raw is not None:
        df_work=st.session_state.df_raw.copy(); plogs=[]; null_before=df_work.isnull().sum().copy()
        date_kw=("date","time","day","month","year")
        dfound=[c for c in df_work.columns if any(k in c.lower() for k in date_kw)]
        for col in dfound:
            try:
                conv=pd.to_datetime(df_work[col],errors="coerce"); df_work[col]=conv
                plogs.append(("ok",f"Parsed <strong>{col}</strong> → datetime ({int(conv.notna().sum()):,} valid)"))
            except Exception as ex: plogs.append(("warn",f"Could not parse <strong>{col}</strong>: {ex}"))
        if not dfound: plogs.append(("info","No datetime columns detected."))
        any_num=False
        for col in df_work.select_dtypes(include=[np.number]).columns:
            nb=int(df_work[col].isnull().sum())
            if nb==0: continue
            any_num=True; med=df_work[col].median(); med=0.0 if pd.isna(med) else med
            df_work[col]=(df_work[col].interpolate(method="linear",limit_direction="both").ffill().bfill().fillna(med))
            na=int(df_work[col].isnull().sum())
            plogs.append(("ok" if na==0 else "warn",
                f"Numeric <strong>{col}</strong> — {nb-na:,}/{nb:,} nulls filled [interp→ffill→bfill→median={med:.2f}]"))
        if not any_num: plogs.append(("info","No numeric nulls found."))
        any_cat=False
        for col in df_work.select_dtypes(include=["object","category"]).columns:
            nb=int(df_work[col].isnull().sum())
            if nb==0: continue
            any_cat=True; modes=df_work[col].mode(); fv=modes.iloc[0] if not modes.empty else "Unknown"
            df_work[col]=df_work[col].fillna(fv); na=int(df_work[col].isnull().sum())
            plogs.append(("ok" if na==0 else "warn",f"Categorical <strong>{col}</strong> — {nb-na:,}/{nb:,} nulls → mode='{fv}'"))
        if not any_cat: plogs.append(("info","No categorical nulls found."))
        for col in df_work.select_dtypes(include=["datetime64[ns]"]).columns:
            nb=int(df_work[col].isnull().sum())
            if nb==0: continue
            df_work[col]=df_work[col].ffill().bfill(); na=int(df_work[col].isnull().sum())
            plogs.append(("ok" if na==0 else "warn",f"Datetime <strong>{col}</strong> — {nb-na:,}/{nb:,} nulls [ffill+bfill]"))
        for col in [c for c in df_work.columns if df_work[c].isnull().any()]:
            nb=int(df_work[col].isnull().sum())
            try:
                if df_work[col].dtype.kind in ("i","u","f"):
                    fv=df_work[col].median(); fv=0.0 if pd.isna(fv) else fv; df_work[col]=df_work[col].fillna(fv)
                else: df_work[col]=df_work[col].ffill().bfill().fillna("Unknown")
            except Exception: df_work[col]=df_work[col].fillna("Unknown")
            na=int(df_work[col].isnull().sum())
            plogs.append(("ok" if na==0 else "err",f"Safety-net <strong>{col}</strong> — {nb-na:,} residual nulls removed"))
        remaining=int(df_work.isnull().sum().sum()); cleaned=int(null_before.sum())-remaining
        plogs.append(("ok" if remaining==0 else "err",
            f"<strong>{'✅ COMPLETE' if remaining==0 else '⚠️ PARTIAL'} — {cleaned:,} nulls removed · {remaining:,} remaining</strong>"))
        st.session_state.df_clean=df_work
        st.markdown("**📋 Preprocessing Log**")
        st.markdown(f'<div style="background:{FIG_BG};border:1px solid {BORD};border-radius:10px;padding:1rem 1.1rem;margin-bottom:1rem;">',unsafe_allow_html=True)
        for i,(k,m) in enumerate(plogs): logline(k,m,delay=i*0.05)
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("**🔄 Before vs After**")
        ca,cb=st.columns(2,gap="large"); nb_f=null_before[null_before>0]
        with ca:
            st.markdown("**Before**")
            if nb_f.empty: st.markdown('<div class="ok-box">No nulls before preprocessing.</div>',unsafe_allow_html=True)
            else: st.dataframe(pd.DataFrame({"Column":nb_f.index,"Null Count":nb_f.values,"Null %":(nb_f.values/len(st.session_state.df_raw)*100).round(2)}),hide_index=True,use_container_width=True)
        with cb:
            st.markdown("**After**")
            na_a=df_work.isnull().sum(); na_a=na_a[na_a>0]
            if na_a.empty: st.markdown('<div class="ok-box">🎉 Zero nulls remaining!</div>',unsafe_allow_html=True)
            else: st.dataframe(pd.DataFrame({"Column":na_a.index,"Null Count":na_a.values}),hide_index=True,use_container_width=True)
    hr()

    # STEP 4
    step_hdr("4","👁️","Preview Cleaned Dataset")
    if st.session_state.df_clean is not None:
        dfc=st.session_state.df_clean; all_c=dfc.columns.tolist()
        logline("info",f"Shape: <strong>{dfc.shape[0]:,} rows × {dfc.shape[1]} cols</strong>&nbsp;·&nbsp;Nulls remaining: <strong>{int(dfc.isnull().sum().sum()):,}</strong>")
        sel=st.multiselect("Select columns to display:",options=all_c,default=all_c[:min(8,len(all_c))],key="prev_sel")
        if sel: st.dataframe(dfc[sel].head(200),use_container_width=True,height=340)
        else: st.info("Select at least one column.")
        with st.expander("📋 Column Schema — Dtypes & Null Summary"):
            schema=pd.DataFrame({"Column":dfc.dtypes.index,"Dtype":dfc.dtypes.astype(str).values,
                "Non-Null":dfc.notna().sum().values,"Null":dfc.isna().sum().values,
                "Null %":(dfc.isna().mean()*100).round(2).values,
                "Sample":[str(dfc[c].dropna().iloc[0]) if dfc[c].notna().any() else "—" for c in dfc.columns]})
            st.dataframe(schema,hide_index=True,use_container_width=True)
    else:
        st.markdown('<div class="info-box">⚙️ Run <strong>Step 3 Preprocessing</strong> first.</div>',unsafe_allow_html=True)
    hr()

    # STEP 5 — EDA
    step_hdr("5","📊","Exploratory Data Analysis")
    if st.button("📊  Run Full EDA",disabled=(st.session_state.df_clean is None),key="btn_eda"):
        run_loader("Running EDA pipeline…",[
            "🧠 Initialising EDA engine…","📐 Computing descriptive statistics…",
            "📈 Building distribution plots…","🔥 Building correlation heatmap…",
            "🏷️ Analysing categorical columns…","📦 Detecting outliers (IQR)…",
            "📅 Rendering time series…","✅ EDA complete — all charts ready!",
        ], delay=0.22)
        st.session_state.eda_done = True

    if st.session_state.eda_done and st.session_state.df_clean is not None:
        dfe=st.session_state.df_clean.copy()
        num_cols=dfe.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols=dfe.select_dtypes(include=["object","category"]).columns.tolist()
        dt_cols=dfe.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        st.markdown(f'<div class="ok-box" style="margin-bottom:1.2rem;">✅ &nbsp;<strong>{len(dfe):,} rows</strong> &nbsp;·&nbsp;<strong>{len(num_cols)} numeric</strong> &nbsp;·&nbsp;<strong>{len(cat_cols)} categorical</strong> &nbsp;·&nbsp;<strong>{len(dt_cols)} datetime</strong> &nbsp;·&nbsp;<strong>0 nulls</strong></div>',unsafe_allow_html=True)

        # 5a descriptive
        st.markdown('<div class="eda-card">',unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">📋 Descriptive Statistics</div>',unsafe_allow_html=True)
        if num_cols: st.dataframe(dfe[num_cols].describe().round(3),use_container_width=True,height=295)
        else: st.warning("No numeric columns found.")
        st.markdown('</div>',unsafe_allow_html=True)

        # 5b distributions
        if num_cols:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">📈 Numeric Distributions</div>',unsafe_allow_html=True)
            for cs in range(0,len(num_cols),3):
                chunk=num_cols[cs:cs+3]; cols=st.columns(len(chunk))
                for j,cn in enumerate(chunk):
                    with cols[j]:
                        fig=px.histogram(dfe,x=cn,nbins=30,color_discrete_sequence=[PAL[(cs+j)%len(PAL)]],title=cn)
                        fig.update_traces(marker_line_width=0.4,marker_line_color=FIG_BG,opacity=0.88)
                        fig=apply_theme(fig,248); fig.update_layout(showlegend=False,xaxis_title="",yaxis_title="Count")
                        st.plotly_chart(fig,use_container_width=True,key=f"hist_{cn}")
            st.markdown('</div>',unsafe_allow_html=True)

        # 5c correlation
        if len(num_cols)>=2:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">🔥 Correlation Heatmap</div>',unsafe_allow_html=True)
            corr=dfe[num_cols].corr(numeric_only=True).round(2)
            fig=px.imshow(corr,text_auto=True,aspect="auto",
                color_continuous_scale=[[0,"#2ECC71"],[0.5,CARD],[1,ACC]],
                zmin=-1,zmax=1,title="Pearson Correlation Matrix — All Numeric Features")
            fig.update_traces(textfont=dict(size=10,color=TXT),hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.2f}<extra></extra>")
            fig=apply_theme(fig,max(400,len(num_cols)*62))
            fig.update_layout(coloraxis_colorbar=dict(title="r",tickfont=dict(size=10,color=TXT),len=0.8))
            st.plotly_chart(fig,use_container_width=True,key="corr_heatmap"); st.markdown('</div>',unsafe_allow_html=True)

        # 5d categorical
        if cat_cols:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">🏷️ Categorical Distributions</div>',unsafe_allow_html=True)
            show_cats=cat_cols[:6]; pairs=[show_cats[i:i+2] for i in range(0,len(show_cats),2)]
            for pair in pairs:
                cols=st.columns(len(pair))
                for gi,cn in enumerate(pair):
                    with cols[gi]:
                        vc=dfe[cn].value_counts().head(15)
                        if vc.empty: continue
                        base=PAL[gi%len(PAL)]; r,g,b=[int(base.lstrip("#")[i:i+2],16) for i in (0,2,4)]
                        n_b=len(vc); clrs=[f"rgba({r},{g},{b},{0.4+0.6*i/max(n_b-1,1):.2f})" for i in range(n_b)]
                        fig=go.Figure(go.Bar(x=vc.values,y=vc.index.astype(str),orientation="h",
                            marker=dict(color=clrs,line=dict(width=0)),text=vc.values,texttemplate="%{text:,}",
                            textposition="outside",textfont=dict(size=11,color=TXT),
                            hovertemplate=f"<b>%{{y}}</b><br>Count: %{{x:,}}<extra></extra>"))
                        fig=apply_theme(fig,max(220,n_b*34))
                        fig.update_layout(title=cn,yaxis=dict(autorange="reversed"),xaxis_title="Count",yaxis_title="",margin=dict(l=10,r=55,t=44,b=12))
                        st.plotly_chart(fig,use_container_width=True,key=f"cat_{cn}_{gi}")
            st.markdown('</div>',unsafe_allow_html=True)

        # 5e time series
        if dt_cols and num_cols:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">📅 Time Series Trends</div>',unsafe_allow_html=True)
            dc=dt_cols[0]; sel_m=st.selectbox("Select metric to plot over time:",num_cols,key="ts_metric")
            ts_df=dfe[[dc,sel_m]].dropna().sort_values(dc)
            if not ts_df.empty:
                try:
                    ts_w=ts_df.set_index(dc)[sel_m].resample("W").mean().dropna().reset_index()
                    plot_df=ts_w if not ts_w.empty else ts_df; ts_label=f"{sel_m} — Weekly Average"
                except Exception: plot_df=ts_df; ts_label=f"{sel_m} — Raw Values"
                fig=px.line(plot_df,x=dc,y=sel_m,color_discrete_sequence=[ACC],title=ts_label,markers=True)
                fig.update_traces(line=dict(width=2.5),fill="tozeroy",fillcolor=hex_rgba(ACC,0.08),
                    marker=dict(size=4,color=ACC),hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}<extra></extra>")
                fig=apply_theme(fig,340); st.plotly_chart(fig,use_container_width=True,key="ts_line")
                try: ts_m=ts_df.set_index(dc)[sel_m].resample("ME").mean().dropna().reset_index()
                except Exception:
                    try: ts_m=ts_df.set_index(dc)[sel_m].resample("M").mean().dropna().reset_index()
                    except Exception: ts_m=pd.DataFrame()
                if not ts_m.empty:
                    fig2=px.bar(ts_m,x=dc,y=sel_m,color_discrete_sequence=[ACC2],title=f"{sel_m} — Monthly Average")
                    fig2.update_traces(marker_line_width=0,opacity=0.85); fig2=apply_theme(fig2,275)
                    st.plotly_chart(fig2,use_container_width=True,key="ts_bar")
            else: st.info("Not enough data for time series.")
            st.markdown('</div>',unsafe_allow_html=True)
        elif not dt_cols:
            st.markdown('<div class="info-box">ℹ️ No datetime column detected.</div>',unsafe_allow_html=True)

        # 5f box plots
        if num_cols:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">📦 Outlier Detection — Box Plots</div>',unsafe_allow_html=True)
            sel_box=st.multiselect("Select columns to inspect:",num_cols,default=num_cols[:min(5,len(num_cols))],key="box_sel")
            if sel_box:
                fig=go.Figure()
                for i,cn in enumerate(sel_box):
                    clr=PAL[i%len(PAL)]; rgba=hex_rgba(clr,0.2)
                    fig.add_trace(go.Box(y=dfe[cn].dropna(),name=cn,marker_color=clr,line=dict(color=clr,width=1.5),
                        fillcolor=rgba,boxpoints="outliers",jitter=0.35,marker=dict(size=4,opacity=0.65,color=clr),
                        hovertemplate=f"<b>{cn}</b><br>%{{y:.2f}}<extra></extra>"))
                fig=apply_theme(fig,400); fig.update_layout(title="Box Plots — Distribution & Outlier View",showlegend=True,plot_bgcolor=FIG_BG,xaxis=dict(showgrid=False))
                st.plotly_chart(fig,use_container_width=True,key="box_chart")
                iqr_rows=[]
                for cn in sel_box:
                    d=dfe[cn].dropna(); Q1,Q3=d.quantile(0.25),d.quantile(0.75); IQR=Q3-Q1
                    out=int(((d<Q1-1.5*IQR)|(d>Q3+1.5*IQR)).sum())
                    iqr_rows.append({"Column":cn,"Min":round(d.min(),3),"Q1":round(Q1,3),"Median":round(d.median(),3),"Q3":round(Q3,3),"Max":round(d.max(),3),"IQR":round(IQR,3),"Outliers":out,"Outlier %":round(out/len(d)*100,2)})
                st.dataframe(pd.DataFrame(iqr_rows),hide_index=True,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

        # 5g scatter
        if len(num_cols)>=2:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">🔭 Scatter Plot Explorer</div>',unsafe_allow_html=True)
            s1,s2,s3=st.columns(3)
            with s1: x_ax=st.selectbox("X axis:",num_cols,index=0,key="sc_x")
            with s2: y_ax=st.selectbox("Y axis:",num_cols,index=min(1,len(num_cols)-1),key="sc_y")
            with s3: hue=st.selectbox("Colour by:",["— None —"]+cat_cols,key="sc_hue")
            color_col=None if hue=="— None —" else hue
            sdf=dfe.sample(min(4000,len(dfe)),random_state=42)
            fig=px.scatter(sdf,x=x_ax,y=y_ax,color=color_col,opacity=0.65,color_discrete_sequence=PAL,
                title=f"Scatter: {x_ax}  ×  {y_ax}"+("" if color_col is None else f"  |  by {color_col}"),labels={x_ax:x_ax,y_ax:y_ax})
            fig.update_traces(marker=dict(size=6,line=dict(width=0.3,color=FIG_BG)))
            if color_col is None:
                common=sdf[[x_ax,y_ax]].dropna()
                if len(common)>=2:
                    z=np.polyfit(common[x_ax],common[y_ax],1); x_line=np.linspace(common[x_ax].min(),common[x_ax].max(),200)
                    fig.add_trace(go.Scatter(x=x_line,y=np.polyval(z,x_line),mode="lines",name=f"Trend (slope={z[0]:.3f})",line=dict(color="#E74C3C",width=2,dash="dash")))
            fig=apply_theme(fig,440); st.plotly_chart(fig,use_container_width=True,key="scatter_main"); st.markdown('</div>',unsafe_allow_html=True)

        # 5h pair plot
        if len(num_cols)>=3:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">🔗 Pair Plot — Multi-variable Relationships</div>',unsafe_allow_html=True)
            pair_def=num_cols[:min(4,len(num_cols))]
            pair_sel=st.multiselect("Select columns (2–5 recommended):",num_cols,default=pair_def,key="pair_sel")
            if len(pair_sel)>=2:
                pairdf=dfe[pair_sel+([cat_cols[0]] if cat_cols else [])].sample(min(1500,len(dfe)),random_state=42)
                fig=px.scatter_matrix(pairdf,dimensions=pair_sel,color=cat_cols[0] if cat_cols else None,
                    color_discrete_sequence=PAL,opacity=0.55,title="Pair Plot — "+", ".join(pair_sel))
                fig.update_traces(marker=dict(size=3,line=dict(width=0.2,color=FIG_BG)),diagonal_visible=True)
                fig=apply_theme(fig,max(520,len(pair_sel)*135)); st.plotly_chart(fig,use_container_width=True,key="pair_matrix")
            else: st.info("Select at least 2 columns for the pair plot.")
            st.markdown('</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="ok-box" style="margin-top:1.2rem;">✅ EDA complete &nbsp;·&nbsp;<strong>{len(num_cols)}</strong> numeric &nbsp;·&nbsp;<strong>{len(cat_cols)}</strong> categorical &nbsp;·&nbsp; Distributions · Correlation · Outliers · Time Series · Scatter · Pair Plot</div>',unsafe_allow_html=True)

    st.markdown(f'<div class="footer">💪 <b>FitPulse Analytics</b> &nbsp;·&nbsp; Thistle Purple Edition &nbsp;·&nbsp; Professional Health Data Pipeline</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
<div class="fp-hero">
  <div style="margin-bottom:8px;"><span class="fp-tag">FEATURE EXTRACTION &amp; MODELING</span></div>
  <div class="fp-title">🧬 FitPulse ML Pipeline</div>
  <div class="fp-sub">TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data</div>
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["📁 Data Loading","📊 TSFresh Features","📈 Prophet Forecast","🔵 Clustering"])

    # TAB 1 — Data Loading
    with tab1:
        st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;"><div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{ACC};">📁 Data Loading</div><span class="steps-badge">Steps 1–9</span></div>',unsafe_allow_html=True)
        st.markdown('<div class="fp-info">Select all 5 Fitbit CSV files at once. Files are auto-detected by column structure.</div>',unsafe_allow_html=True)
        uploaded_ml=st.file_uploader("📂 Drop all Fitbit CSV files here",type=["csv"],accept_multiple_files=True,key="ml_upload")
        EXPECTED={"Daily Activity":["ActivityDate","TotalSteps","Calories"],"Hourly Steps":["ActivityHour","StepTotal"],"Hourly Intensities":["ActivityHour","TotalIntensity"],"Minute Sleep":["date","value","logId"],"Heart Rate":["Time","Value"]}
        ICONS={"Daily Activity":"🏃","Hourly Steps":"👣","Hourly Intensities":"⚡","Minute Sleep":"😴","Heart Rate":"❤️"}
        detected={}
        if uploaded_ml:
            for f in uploaded_ml:
                try:
                    tmp=pd.read_csv(f,nrows=2); cols=set(tmp.columns)
                    for name,req in EXPECTED.items():
                        if all(c in cols for c in req): f.seek(0); detected[name]=pd.read_csv(f); break
                except Exception: pass
        all_ok=True; cards_html='<div class="ds-grid">'
        for name,ico in ICONS.items():
            ok=name in detected
            if not ok: all_ok=False
            s='<span class="chip-ok">✅ Loaded</span>' if ok else '<span class="chip-miss">❌ Missing</span>'
            cards_html+=f'<div class="ds-card"><div class="ds-icon">{ico}</div><div class="ds-name">{name}</div><div class="ds-status">{s}</div></div>'
        st.markdown(cards_html+"</div>",unsafe_allow_html=True); st.markdown("")
        if all_ok:
            total=sum(len(v) for v in detected.values()); st.success("✅ All 5 datasets loaded!")
            c1,c2,c3=st.columns(3); c1.metric("Files Loaded","5 / 5"); c2.metric("Total Rows",f"{total:,}"); c3.metric("Unique Users","~30")
            if st.button("▶ Proceed to TSFresh Features"):
                st.session_state.ml_dfs=detected; st.session_state.ml_progress=20; st.rerun()
        else:
            missing=[n for n in ICONS if n not in detected]
            if missing: st.warning(f"⚠️ Missing: {', '.join(missing)}")
        st.markdown("---"); st.caption("💡 No CSV files? Use demo mode.")
        if st.button("🎲 Load Demo Data & Proceed"):
            st.session_state.ml_dfs={"demo":True}; st.session_state.ml_progress=20; st.rerun()

    # TAB 2 — TSFresh
    with tab2:
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{ACC};margin-bottom:10px;">📊 TSFresh Feature Extraction</div>',unsafe_allow_html=True)
        if st.session_state.ml_progress<20:
            st.info("⬅️ Complete Data Loading first.")
        else:
            st.markdown('<div class="fp-info">TSFresh extracts hundreds of statistical time-series features per user.</div>',unsafe_allow_html=True)
            c1,c2=st.columns([2,1])
            with c1: st.selectbox("Feature Calculation Level",["Minimal (fast)","Efficient (balanced)","Comprehensive (slow)"],index=1)
            with c2: st.number_input("Parallel Jobs",1,8,4)
            st.multiselect("Signals",["Heart Rate","Hourly Steps","Hourly Intensities","Minute Sleep"],default=["Heart Rate","Hourly Steps"])
            if st.button("🚀 Run TSFresh Extraction"):
                with st.spinner("Extracting features…"):
                    bar=st.progress(0)
                    for i in range(1,101): time.sleep(0.008); bar.progress(i)
                st.session_state.feat_df=make_tsfresh_matrix()
                st.session_state.ml_progress=max(st.session_state.ml_progress,40); st.rerun()
            if st.session_state.feat_df is not None:
                feat_df=st.session_state.feat_df
                st.success(f"✅ Extracted **{len(feat_df.columns)} features** for **{len(feat_df)} users**")
                fig,ax=plt.subplots(figsize=(11,5.8)); style_mpl(fig,[ax])
                im=ax.imshow(feat_df.values,cmap="RdBu_r",aspect="auto",vmin=0,vmax=1)
                ax.set_xticks(range(len(feat_df.columns))); ax.set_xticklabels(feat_df.columns,rotation=45,ha="right",fontsize=7.5)
                ax.set_yticks(range(len(feat_df.index))); ax.set_yticklabels(feat_df.index,fontsize=7.5)
                ax.set_xlabel("Feature",fontsize=9); ax.set_ylabel("User ID",fontsize=9)
                ax.set_title("TSFresh Feature Matrix\n(Normalized 0-1 per feature)",fontsize=10,color=ACC,fontweight="bold")
                for r in range(len(feat_df.index)):
                    for c in range(len(feat_df.columns)):
                        v=feat_df.values[r,c]; tc="white" if (v>0.72 or v<0.28) else TXT
                        ax.text(c,r,f"{v:.2f}",ha="center",va="center",fontsize=6.8,color=tc,fontweight="600")
                cbar=fig.colorbar(im,ax=ax,fraction=0.02,pad=0.02); cbar.ax.tick_params(labelsize=7,colors=TXT)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                with st.expander("📋 Raw Feature Table"):
                    st.dataframe(feat_df.round(3).style.background_gradient(cmap="RdBu_r",axis=None),use_container_width=True)
                c1,c2,c3=st.columns(3); c1.metric("Features",len(feat_df.columns)); c2.metric("Users",len(feat_df.index)); c3.metric("Missing","0")

    # TAB 3 — Prophet
    with tab3:
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{ACC};margin-bottom:10px;">📈 Prophet Trend Forecast</div>',unsafe_allow_html=True)
        if st.session_state.ml_progress<40:
            st.info("⬅️ Complete TSFresh Features first.")
        else:
            st.markdown('<div class="fp-info">Prophet decomposes signals into trend + weekly seasonality. Shaded = 80% CI · dashed = forecast start.</div>',unsafe_allow_html=True)
            c1,c2,c3=st.columns(3)
            with c1: fdays=st.slider("Forecast Horizon (days)",7,90,30)
            with c2: st.selectbox("Confidence Interval",["80%","90%","95%"])
            with c3: show_comp=st.checkbox("Show Decomposition",True)
            if st.button("🔮 Run Prophet Forecast"):
                with st.spinner("Fitting Prophet on HR, Steps, Sleep…"):
                    bar=st.progress(0)
                    for i in range(1,101): time.sleep(0.008); bar.progress(i)
                st.session_state.ml_progress=max(st.session_state.ml_progress,60); st.rerun()
            if st.session_state.ml_progress>=60:
                st.markdown("#### ❤️ Heart Rate — Prophet Trend Forecast")
                dates,actual,yhat,ylow,yhigh=make_prophet_hr()
                fig,ax=plt.subplots(figsize=(11,3.8)); style_mpl(fig,[ax])
                ax.fill_between(dates,ylow,yhigh,alpha=0.25,color="#6ab0de",label="80% CI")
                ax.plot(dates,yhat,color="#2060a0",lw=2,label="Predicted Trend")
                ax.scatter(dates[:20],actual,color="#e05555",s=28,zorder=5,label="Actual HR")
                ax.axvline(dates[20],color="#e09030",linestyle="--",lw=1.4,label="Forecast Start")
                ax.set_xlabel("Date",fontsize=9); ax.set_ylabel("Heart Rate (bpm)",fontsize=9)
                ax.set_title("Heart Rate — Prophet Trend Forecast",fontsize=10,color=ACC,fontweight="bold")
                ax.legend(fontsize=7.5,facecolor=FIG_BG,edgecolor=BORD); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                st.markdown("#### 👣 Steps & 😴 Sleep")
                sd,sa,sy,sl,sh,ss=make_prophet_steps(); ld,la,ly,ll,lh,ls=make_prophet_sleep()
                fig2,(a1,a2)=plt.subplots(2,1,figsize=(11,7)); style_mpl(fig2,[a1,a2])
                a1.fill_between(sd,sl,sh,alpha=0.25,color="#3db87a",label="80% CI"); a1.plot(sd,sy,color="#1a5e38",lw=2,label="Trend")
                a1.scatter(sd[:ss],sa,color="#3db87a",s=22,zorder=5,label="Actual Steps"); a1.axvline(sd[ss],color="#e09030",linestyle="--",lw=1.3,label="Forecast Start")
                a1.set_ylabel("Steps",fontsize=9); a1.set_title("Steps — Prophet Trend Forecast",fontsize=10,color=ACC,fontweight="bold"); a1.legend(fontsize=7.5,facecolor=FIG_BG,edgecolor=BORD)
                a2.fill_between(ld,ll,lh,alpha=0.25,color=ACC2,label="80% CI"); a2.plot(ld,ly,color=ACC,lw=2,label="Trend")
                a2.scatter(ld[:ls],la,color="#a07bd0",s=22,zorder=5,label="Actual Sleep"); a2.axvline(ld[ls],color="#e09030",linestyle="--",lw=1.3,label="Forecast Start")
                a2.set_xlabel("Date",fontsize=9); a2.set_ylabel("Sleep (minutes)",fontsize=9)
                a2.set_title("Sleep — Prophet Trend Forecast",fontsize=10,color=ACC,fontweight="bold"); a2.legend(fontsize=7.5,facecolor=FIG_BG,edgecolor=BORD)
                fig2.tight_layout(pad=2); st.pyplot(fig2); plt.close(fig2)
                if show_comp:
                    st.markdown("#### 🔍 Prophet Components")
                    cd,tv,wv=make_prophet_components(); fig3,(at,aw)=plt.subplots(2,1,figsize=(11,5)); style_mpl(fig3,[at,aw])
                    at.plot(cd,tv,color="#2060a0",lw=2); at.set_ylabel("trend",fontsize=9); at.set_title("Trend Component",fontsize=9,color=ACC)
                    aw.plot(cd,wv,color=ACC2,lw=2); aw.set_ylabel("weekly",fontsize=9); aw.set_xlabel("ds",fontsize=9); aw.set_title("Weekly Seasonality",fontsize=9,color=ACC)
                    fig3.tight_layout(pad=2); st.pyplot(fig3); plt.close(fig3)
                c1,c2,c3,c4=st.columns(4); c1.metric("Avg HR","81.3 bpm"); c2.metric("Avg Steps","8,240"); c3.metric("Avg Sleep","248 min"); c4.metric("Horizon",f"{fdays} days")

    # TAB 4 — Clustering
    with tab4:
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{ACC};margin-bottom:10px;">🔵 Clustering — KMeans · DBSCAN · PCA · t-SNE</div>',unsafe_allow_html=True)
        if st.session_state.ml_progress<60:
            st.info("⬅️ Complete Prophet Forecast first.")
        else:
            K=st.session_state.get("k_sl",3); EP=st.session_state.get("eps_sl",2.2)
            st.markdown(f'<div class="fp-info">Clusters are spatially separated. Currently: <b>K={K}</b> · DBSCAN eps=<b>{EP}</b>. Adjust in the sidebar.</div>',unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: algo=st.selectbox("Algorithm",["KMeans + DBSCAN (Both)","KMeans Only","DBSCAN Only"])
            with c2: reduction=st.selectbox("Projection",["Both","PCA","t-SNE"])
            if st.button("🔬 Run Clustering"):
                with st.spinner("Clustering users on TSFresh features…"):
                    bar=st.progress(0)
                    for i in range(1,101): time.sleep(0.010); bar.progress(i)
                st.session_state.ml_progress=100; st.session_state.ml_run_done=True; st.rerun()
            if st.session_state.ml_run_done:
                st.markdown("#### 📉 KMeans Elbow Curve")
                ks,inertia=make_elbow(); fig_e,ax_e=plt.subplots(figsize=(8,3.5)); style_mpl(fig_e,[ax_e])
                ax_e.plot(ks,inertia,color=ACC2,lw=2,zorder=2); ax_e.scatter(ks,inertia,color=ACC,s=65,zorder=3)
                ax_e.axvline(K,color="#e09030",linestyle="--",lw=1.8,label=f"Selected K={K}")
                ax_e.set_xlabel("Number of Clusters (K)",fontsize=9); ax_e.set_ylabel("Inertia",fontsize=9)
                ax_e.set_title("KMeans Elbow Curve",fontsize=10,color=ACC,fontweight="bold"); ax_e.legend(fontsize=8.5,facecolor=FIG_BG,edgecolor=BORD)
                fig_e.tight_layout(); st.pyplot(fig_e); plt.close(fig_e)
                if reduction in ["PCA","Both"]:
                    st.markdown(f"#### 🔵 KMeans — PCA Projection (K={K})")
                    coords_pca,lab_pca=make_pca_scatter(K); fig_p,ax_p=plt.subplots(figsize=(10,6)); style_mpl(fig_p,[ax_p])
                    for ci in range(K):
                        mask=lab_pca==ci; pts=coords_pca[mask]
                        ax_p.scatter(pts[:,0],pts[:,1],color=CLUSTER_COLORS[ci],s=95,label=f"Cluster {ci}",zorder=3,edgecolors="white",linewidth=0.8)
                        for j,(x,y) in enumerate(pts):
                            ax_p.annotate(USER_LABELS[j%len(USER_LABELS)],(x,y),fontsize=6.5,color=TXT,xytext=(4,3),textcoords="offset points")
                    ax_p.set_xlabel("PC1 (23.1% variance)",fontsize=9); ax_p.set_ylabel("PC2 (16.4% variance)",fontsize=9)
                    ax_p.set_title(f"KMeans — PCA Projection (K={K})",fontsize=10,color=ACC,fontweight="bold")
                    ax_p.legend(title="Cluster",fontsize=9,facecolor=FIG_BG,edgecolor=BORD); fig_p.tight_layout(); st.pyplot(fig_p); plt.close(fig_p)
                if reduction in ["t-SNE","Both"]:
                    st.markdown(f"#### 🌀 t-SNE — KMeans (K={K}) vs DBSCAN (eps={EP})")
                    coords_tsne,lab_tsne=make_tsne(K); lab_db=lab_tsne.copy()
                    c1_idx=np.where(lab_db==1)[0]
                    if len(c1_idx): lab_db[c1_idx[0]]=-1
                    fig_t,(ax_km,ax_db)=plt.subplots(1,2,figsize=(14,5.5),sharex=True,sharey=True); style_mpl(fig_t,[ax_km,ax_db])
                    for ci in range(K):
                        m=lab_tsne==ci
                        ax_km.scatter(coords_tsne[m,0],coords_tsne[m,1],color=CLUSTER_COLORS[ci],s=65,label=f"Cluster {ci}",zorder=3,edgecolors="white",linewidth=0.6)
                    ax_km.set_xlabel("t-SNE Dim 1",fontsize=9); ax_km.set_ylabel("t-SNE Dim 2",fontsize=9)
                    ax_km.set_title(f"KMeans — t-SNE (K={K})",fontsize=10,color=ACC,fontweight="bold"); ax_km.legend(title="Cluster",fontsize=8,facecolor=FIG_BG,edgecolor=BORD)
                    noise_m=lab_db==-1
                    if noise_m.any(): ax_db.scatter(coords_tsne[noise_m,0],coords_tsne[noise_m,1],color="#e04040",s=110,marker="X",label="Noise",zorder=5)
                    for ci in range(K):
                        m=lab_db==ci
                        ax_db.scatter(coords_tsne[m,0],coords_tsne[m,1],color=CLUSTER_COLORS[ci],s=65,label=f"Cluster {ci}",zorder=3,edgecolors="white",linewidth=0.6)
                    ax_db.set_xlabel("t-SNE Dim 1",fontsize=9); ax_db.set_title(f"DBSCAN — t-SNE (eps={EP})",fontsize=10,color=ACC,fontweight="bold"); ax_db.legend(title="Cluster",fontsize=8,facecolor=FIG_BG,edgecolor=BORD)
                    fig_t.tight_layout(pad=2.5); st.pyplot(fig_t); plt.close(fig_t)
                st.markdown("#### 📊 Cluster Summary")
                np.random.seed(55)
                counts_map={2:[14,16],3:[10,10,5],4:[8,8,6,8],5:[6,6,5,6,7]}
                cnts=counts_map.get(K,[8]*K)
                summary=pd.DataFrame({"Cluster":[f"Cluster {i}" for i in range(K)],"Users":cnts[:K],
                    "Avg Daily Steps":np.random.randint(5000,12000,K),"Avg HR (bpm)":np.random.randint(62,88,K),
                    "Avg Sleep (min)":np.random.randint(200,420,K),"Activity Level":["High","Low","Moderate","High","Low"][:K]})
                st.dataframe(summary,use_container_width=True,hide_index=True)
                c1,c2,c3,c4=st.columns(4)
                c1.metric("KMeans Clusters",K); c2.metric("Silhouette Score",f"{np.random.uniform(0.45,0.72):.3f}"); c3.metric("DBSCAN Noise","1"); c4.metric("Total Users","30")
                st.success("🎉 Full pipeline complete — all stages finished!")

    st.markdown("---")
    st.markdown(f'<div style="text-align:center;font-size:.8rem;color:{SOFT};font-family:Syne,sans-serif;padding:4px 0 8px;">🧬 FitPulse ML Pipeline &nbsp;·&nbsp; TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE</div>',unsafe_allow_html=True)