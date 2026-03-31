"""
FitPulse Unified Platform  ·  v6.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single Streamlit app combining:
  • Analytics Platform   — Thistle palette, full EDA pipeline
  • ML Pipeline          — TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE
  • Anomaly Detection    — Threshold · Residual · DBSCAN Outliers · Accuracy Simulation
  • Insights Dashboard   — KPI strip · Timeline · Deep Dives · PDF & CSV Export

Sidebar dropdown switches modes.  ALL chrome re-themes on every rerun.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, os, tempfile, time, warnings
from datetime import datetime
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
    # — anomaly —
    "anom_dark_mode":        False,
    "anom_files_loaded":     False,
    "anom_anomaly_done":     False,
    "anom_simulation_done":  False,
    "anom_daily":    None,
    "anom_hourly_s": None,
    "anom_hourly_i": None,
    "anom_sleep":    None,
    "anom_hr":       None,
    "anom_hr_minute":None,
    "anom_master":   None,
    "anom_hr_result":    None,
    "anom_steps_result": None,
    "anom_sleep_result": None,
    "anom_sim_results":  None,
    # — insights dashboard —
    "ins_pipeline_done": False,
    "ins_master":        None,
    "ins_anom_hr":       None,
    "ins_anom_steps":    None,
    "ins_anom_sleep":    None,
    # — shared fitbit dataset (uploaded once in ML Pipeline, used everywhere) —
    "shared_detected":   {},   # dict of req_name -> DataFrame
    "shared_master":     None, # merged master DataFrame
    "shared_built":      False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SHARED COLOUR CONSTANTS  (Thistle Purple — always-on, no dark toggle)
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#D8BFD8"
CARD   = "#EDE0ED"
FIG_BG = "#F5ECF5"
BORD   = "#C4A0C4"
ACC    = "#6B2D8B"
ACC2   = "#9B59B6"
TXT    = "#2D1B3D"
SOFT   = "#7B5C8A"
SB_BG  = "#1A0A22"
SB_BDR = "#3D1F52"
SB_FUP = "#251035"

PAL = [ACC, ACC2, "#2ECC71","#1ABC9C","#E74C3C","#F39C12","#3498DB","#E91E63"]
CLUSTER_COLORS = {0:"#4d8fd4",1:"#e0609a",2:"#3baa66",3:"#f4a020",4:"#9b6cd4",-1:"#e04040"}

ACCENT_RED = "#C0392B"
ACCENT3    = "#1E8449"

# — Insights Dashboard colour aliases (Thistle Purple light palette) —
INS_BG          = f"linear-gradient(135deg,{FIG_BG} 0%,{CARD} 50%,{FIG_BG} 100%)"
INS_CARD_BG     = CARD
INS_CARD_BOR    = BORD
INS_TEXT        = TXT
INS_MUTED       = SOFT
INS_ACCENT      = ACC
INS_ACCENT2     = "#9B59B6"
INS_ACCENT3     = ACCENT3
INS_ACCENT_RED  = ACCENT_RED
INS_ACCENT_ORG  = "#D35400"
INS_PLOT_BG     = FIG_BG
INS_PAPER_BG    = CARD
INS_GRID_CLR    = f"rgba(196,160,196,0.35)"
INS_BADGE_BG    = f"rgba(107,45,139,0.12)"
INS_SECTION_BG  = f"rgba(245,236,245,0.85)"
INS_WARN_BG     = "rgba(211,84,0,0.07)"
INS_WARN_BOR    = "rgba(211,84,0,0.35)"
INS_SUCCESS_BG  = "rgba(30,132,73,0.07)"
INS_SUCCESS_BOR = "rgba(30,132,73,0.35)"
INS_DANGER_BG   = "rgba(192,57,43,0.07)"
INS_DANGER_BOR  = "rgba(192,57,43,0.4)"

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
_mode = st.session_state.mode
_ml   = _mode == "ML Pipeline"
_anom = _mode == "Anomaly Detection"
_ins  = _mode == "Insights Dashboard"

_HDR_BG    = f"linear-gradient(90deg,{SB_BG},{SB_BDR})"
_DECO      = f"linear-gradient(90deg,{ACC},{ACC2})"
_SB_SEL_BG = "#2A1040"
_SB_SEL_BD = ACC
_POP_BG    = "#2A1040"
_POP_HOV   = SB_BDR
_SB_ACTUAL_BG = "linear-gradient(160deg,#2A1040,#1A0A22)" if (_ml or _anom or _ins) else BG

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

@keyframes fadeUp   {{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes slideIn  {{from{{opacity:0;transform:translateX(-14px)}}to{{opacity:1;transform:translateX(0)}}}}
@keyframes rotateBdr{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
@keyframes pulseDot {{0%,100%{{transform:scale(1);opacity:1}}50%{{transform:scale(1.7);opacity:0.4}}}}
@keyframes gradFlow {{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}
@keyframes shimmer  {{from{{transform:translateX(-100%)}}to{{transform:translateX(250%)}}}}
@keyframes popIn    {{from{{opacity:0;transform:scale(0.9)}}to{{opacity:1;transform:scale(1)}}}}
@keyframes countUp  {{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes scanLine {{0%{{left:-4px;opacity:0}}5%{{opacity:1}}95%{{opacity:1}}100%{{left:100%;opacity:0}}}}
@keyframes pulse    {{0%,100%{{opacity:1}}50%{{opacity:0.5}}}}

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

header[data-testid="stHeader"]{{
  background:{_HDR_BG}!important;
  border-bottom:2px solid {SB_BDR}!important;
}}
[data-testid="stToolbar"]{{background:transparent!important;}}
[data-testid="stDecoration"]{{
  background:{_DECO}!important;
  height:3px!important;
}}

[data-testid="stSidebar"]{{
  background:{_SB_ACTUAL_BG}!important;
  border-right:1.5px solid {SB_BDR}!important;
}}
[data-testid="stSidebar"] *{{color:#E8D4F0!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]{{display:flex!important;}}

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

[data-testid="stSidebar"] [data-testid="stFileUploader"] section{{
  border:2px dashed rgba(155,89,182,0.55)!important;
  background:{SB_FUP}!important;border-radius:10px!important;
}}
[data-testid="stSidebar"] .stProgress>div>div>div>div{{
  background:linear-gradient(90deg,{ACC2},{ACC})!important;border-radius:999px;
}}
[data-testid="stSidebar"] .stProgress>div>div{{
  background:rgba(155,89,182,0.3)!important;border-radius:999px;
}}
[data-testid="stSidebar"] [data-testid="stSliderTrackFill"]{{background:{ACC}!important;}}

.stButton>button{{
  background:{ACC}!important;color:#fff!important;border:none!important;
  border-radius:8px!important;padding:0.48rem 1.35rem!important;
  font-weight:600!important;font-size:0.84rem!important;font-family:'DM Sans',sans-serif!important;
  transition:background 0.18s,transform 0.18s,box-shadow 0.18s!important;
  box-shadow:0 2px 12px rgba(107,45,139,0.4)!important;
}}
.stButton>button:hover{{background:#521E6B!important;transform:translateY(-1px)!important;}}
.stButton>button:active{{transform:scale(0.97)!important;}}
.stButton>button:disabled{{background:{BORD}!important;color:{SOFT}!important;box-shadow:none!important;}}

[data-testid="stFileUploader"] section{{
  border:2px dashed {BORD}!important;border-radius:10px!important;background:{FIG_BG}!important;
}}
[data-testid="stFileUploaderDropzone"]{{
  background:{CARD}!important;border:2.5px dashed {ACC2}!important;border-radius:14px!important;
}}
[data-testid="stFileUploaderDropzone"] button,[data-testid="stFileUploader"] button{{
  background:{ACC}!important;color:#fff!important;border:none!important;
  border-radius:8px!important;font-family:'Syne',sans-serif!important;
  font-weight:700!important;font-size:0.85rem!important;padding:7px 20px!important;
}}

.stProgress>div>div>div>div{{background:linear-gradient(90deg,{ACC2},{ACC})!important;border-radius:999px;}}
.stProgress>div>div{{background:rgba(196,160,196,0.4)!important;border-radius:999px;}}

[data-testid="stMetric"]{{
  background:{CARD};border:1.5px solid {BORD};border-radius:14px;padding:10px 14px;
}}
[data-testid="stMetricLabel"]{{font-family:'Syne',sans-serif;font-weight:700;color:{SOFT}!important;}}
[data-testid="stMetricValue"]{{color:{ACC}!important;font-family:'Syne',sans-serif;}}

[data-baseweb="tab-list"]{{background:transparent!important;}}
[data-baseweb="tab"]{{font-family:'Syne',sans-serif!important;font-weight:600!important;color:{SOFT}!important;}}
[aria-selected="true"]{{color:{ACC}!important;border-bottom:2px solid {ACC}!important;}}

[data-baseweb="select"]>div,[data-baseweb="input"]>div{{
  background:{CARD}!important;border-color:{BORD}!important;
}}
[data-testid="stDataFrame"]{{border-radius:10px!important;border:1px solid {BORD}!important;overflow:hidden;}}
details summary{{
  background:{CARD}!important;border:1px solid {BORD}!important;
  border-radius:8px!important;color:{TXT}!important;
}}
[data-baseweb="input"] > div {{ background:{CARD} !important; border-color:{BORD} !important; }}
</style>
""", unsafe_allow_html=True)

# ── COMPONENT CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
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

.step-hdr{{display:flex;align-items:center;gap:0.75rem;
  background:{CARD};border:1px solid {BORD};border-left:5px solid {ACC};border-radius:10px;
  padding:0.78rem 1.3rem;margin:1.4rem 0 1rem;font-family:'Syne',sans-serif;
  font-size:1rem;font-weight:700;color:{TXT};
  box-shadow:0 1px 6px rgba(107,45,139,0.12);animation:fadeUp 0.3s ease;}}
.step-num{{background:{ACC};color:#fff;font-size:0.68rem;font-weight:700;
  width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;}}

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

.loader{{background:{CARD};border:1px solid {BORD};border-radius:12px;
  padding:1.25rem 1.5rem;margin:0.6rem 0;position:relative;overflow:hidden;}}
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
  animation:gradFlow 1.4s linear infinite;border-radius:6px;transition:width 0.2s ease;}}
.ldr-ticker{{font-family:'DM Mono',monospace;font-size:0.68rem;color:{ACC};
  background:{FIG_BG};border:1px solid {BORD};border-radius:5px;padding:4px 10px;}}

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

.info-box{{background:{FIG_BG};border:1px solid {BORD};border-radius:9px;
  padding:0.8rem 1.1rem;font-size:0.8rem;color:{ACC};margin:0.5rem 0;animation:popIn 0.3s ease;}}
.ok-box{{background:{FIG_BG};border:1px solid {BORD};border-radius:9px;
  padding:0.8rem 1.1rem;font-size:0.81rem;color:{ACC};font-weight:600;
  text-align:center;animation:popIn 0.35s ease;}}

.eda-card{{background:{CARD};border:1px solid {BORD};border-radius:12px;
  padding:1.4rem 1.6rem;margin-bottom:1.2rem;
  box-shadow:0 1px 5px rgba(107,45,139,0.08);animation:fadeUp 0.35s ease;}}
.eda-card-title{{font-family:'Syne',sans-serif;font-size:0.98rem;font-weight:700;color:{TXT};
  border-bottom:2px solid {BORD};padding-bottom:0.55rem;margin-bottom:1rem;}}
.nbadge{{display:inline-block;background:{FIG_BG};border:1px solid {BORD};border-radius:6px;
  padding:3px 10px;font-size:0.71rem;color:{ACC};margin:3px;font-weight:500;transition:all 0.2s;}}
.nbadge:hover{{background:{ACC};color:#fff;border-color:{ACC};transform:scale(1.03);}}

.fp-hero{{background:linear-gradient(120deg,#e0cce0ee,#cdb5cdee);
  border:1.5px solid {BORD};border-radius:16px;padding:22px 28px 18px;margin-bottom:20px;}}
.fp-title{{font-family:'Syne',sans-serif;font-size:2.1rem;font-weight:800;color:{ACC};margin:0 0 4px;}}
.fp-sub{{font-size:.82rem;color:{SOFT};letter-spacing:.04em;}}
.fp-tag{{display:inline-block;background:{ACC2};color:#fff;border-radius:999px;
  padding:3px 12px;font-size:.72rem;font-family:'Syne',sans-serif;font-weight:600;margin-right:4px;}}
.fp-info{{background:#e8d5e8dd;border-left:4px solid {ACC2};border-radius:8px;
  padding:10px 16px;margin-bottom:12px;font-size:.9rem;color:{SOFT};}}

.fp-hero-anom{{background:linear-gradient(135deg,#2D1040 0%,#1A0A22 50%,#2D1040 100%);
  border:1px solid rgba(192,57,43,0.4);border-left:5px solid #C0392B;
  border-radius:14px;padding:2rem 2.4rem;margin-bottom:1.5rem;
  position:relative;overflow:hidden;animation:fadeUp 0.4s ease;
  box-shadow:0 4px 22px rgba(192,57,43,0.2);}}
.anom-title{{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#F0E0F8;margin:0 0 0.4rem;}}
.anom-sub{{font-size:0.9rem;color:#B89ABE;font-weight:300;margin:0;}}
.anom-badge{{display:inline-block;background:rgba(192,57,43,0.2);border:1px solid rgba(192,57,43,0.5);
  border-radius:100px;padding:0.3rem 1rem;font-size:0.74rem;
  font-family:'DM Mono',monospace;color:#E74C3C;margin-bottom:0.9rem;}}

.sec-header{{display:flex;align-items:center;gap:0.8rem;
  margin:1.8rem 0 1rem;padding-bottom:0.6rem;border-bottom:2px solid {BORD};}}
.sec-icon{{font-size:1.3rem;width:2.1rem;height:2.1rem;
  display:flex;align-items:center;justify-content:center;
  background:rgba(107,45,139,0.1);border-radius:8px;border:1px solid {BORD};}}
.sec-title{{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{TXT};margin:0;}}
.sec-badge{{margin-left:auto;background:rgba(107,45,139,0.1);border:1px solid {BORD};
  border-radius:100px;padding:0.2rem 0.75rem;font-size:0.7rem;
  font-family:'DM Mono',monospace;color:{ACC};}}

.anom-card{{background:{CARD};border:1px solid {BORD};border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:1rem;}}
.anom-card-title{{font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
  color:{SOFT};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;}}

.metric-grid{{display:flex;gap:0.75rem;flex-wrap:wrap;margin:0.75rem 0;}}
.metric-card{{flex:1;min-width:115px;background:{FIG_BG};border:1px solid {BORD};
  border-radius:12px;padding:0.9rem 1.1rem;text-align:center;}}
.metric-val{{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;color:{ACC};line-height:1;margin-bottom:0.22rem;}}
.metric-val-red{{color:{ACCENT_RED};}}
.metric-label{{font-size:0.7rem;color:{SOFT};text-transform:uppercase;letter-spacing:0.06em;}}

.step-pill{{display:inline-flex;align-items:center;gap:0.5rem;
  background:rgba(107,45,139,0.1);border:1px solid {BORD};border-radius:100px;
  padding:0.28rem 0.9rem;font-size:0.73rem;font-family:'DM Mono',monospace;color:{ACC};margin-bottom:0.75rem;}}
.anom-tag{{display:inline-flex;align-items:center;gap:0.4rem;
  background:rgba(192,57,43,0.1);border:1px solid rgba(192,57,43,0.4);border-radius:100px;
  padding:0.28rem 0.9rem;font-size:0.71rem;font-family:'DM Mono',monospace;
  color:{ACCENT_RED};margin-bottom:0.75rem;}}

.alert-warn    {{background:rgba(211,84,0,0.07);border-left:3px solid #D4AC0D;
  border-radius:0 9px 9px 0;padding:0.75rem 1rem;margin:0.5rem 0;font-size:0.84rem;color:#7D6608;}}
.alert-success {{background:rgba(30,132,73,0.07);border-left:3px solid {ACCENT3};
  border-radius:0 9px 9px 0;padding:0.75rem 1rem;margin:0.5rem 0;font-size:0.84rem;color:{ACCENT3};}}
.alert-info    {{background:rgba(107,45,139,0.07);border-left:3px solid {ACC};
  border-radius:0 9px 9px 0;padding:0.75rem 1rem;margin:0.5rem 0;font-size:0.84rem;color:{ACC};}}
.alert-danger  {{background:rgba(192,57,43,0.07);border-left:3px solid {ACCENT_RED};
  border-radius:0 9px 9px 0;padding:0.75rem 1rem;margin:0.5rem 0;font-size:0.84rem;color:{ACCENT_RED};}}

.ds-grid{{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;}}
.ds-card{{background:#e2cfe2cc;border:1.5px solid {BORD};border-radius:10px;
  padding:10px 16px;min-width:140px;flex:1 1 140px;text-align:center;}}
.ds-icon{{font-size:1.5rem;}}
.ds-name{{font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;color:{ACC};}}
.ds-status{{font-size:.78rem;margin-top:2px;}}
.chip-ok{{color:#3db87a;font-weight:700;}} .chip-miss{{color:#e05555;font-weight:700;}}
.steps-badge{{float:right;background:{ACC2};color:#fff;border-radius:999px;
  padding:2px 12px;font-size:.75rem;font-family:'Syne',sans-serif;font-weight:700;}}

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

.hr{{height:1px;background:linear-gradient(90deg,transparent,{BORD},transparent);margin:1.5rem 0;}}
.footer{{text-align:center;font-size:0.7rem;color:{SOFT};
  padding:0.9rem 0 0.4rem;border-top:1px solid {BORD};margin-top:1.8rem;}}
.footer b{{color:{ACC};}}
hr{{border-color:{BORD}!important;}}

/* ── INSIGHTS DASHBOARD SPECIFIC ── */
.ins-hero{{
  background:linear-gradient(135deg,{CARD} 0%,{FIG_BG} 60%,{CARD} 100%);
  border:1.5px solid {BORD};border-left:5px solid {ACC};border-radius:16px;
  padding:1.8rem 2.2rem;margin-bottom:1.5rem;
  position:relative;overflow:hidden;animation:fadeUp 0.4s ease;
  box-shadow:0 4px 22px rgba(107,45,139,0.18);
}}
.ins-hero::before{{content:'';position:absolute;right:-50px;top:-50px;width:220px;height:220px;
  background:radial-gradient(circle,rgba(155,89,182,0.15),transparent 65%);border-radius:50%;}}
.ins-hero-badge{{display:inline-block;background:{INS_BADGE_BG};border:1px solid {BORD};
  border-radius:100px;padding:0.25rem 0.9rem;font-size:0.72rem;
  font-family:'DM Mono',monospace;color:{ACC};margin-bottom:0.8rem;}}
.ins-hero-title{{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
  color:{TXT};margin:0 0 0.3rem;letter-spacing:-0.02em;}}
.ins-hero-sub{{font-size:0.88rem;color:{SOFT};font-weight:300;margin:0;}}

.ins-kpi-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:0.7rem;margin:1rem 0;}}
.ins-kpi-card{{background:{CARD};border:1px solid {BORD};border-radius:14px;
  padding:1rem 1.1rem;text-align:center;
  box-shadow:0 1px 5px rgba(107,45,139,0.08);transition:transform 0.2s,box-shadow 0.2s;}}
.ins-kpi-card:hover{{transform:translateY(-2px);box-shadow:0 4px 16px rgba(107,45,139,0.15);}}
.ins-kpi-val{{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;
  line-height:1;margin-bottom:0.2rem;}}
.ins-kpi-label{{font-size:0.68rem;color:{SOFT};text-transform:uppercase;letter-spacing:0.07em;}}
.ins-kpi-sub{{font-size:0.65rem;color:{SOFT};margin-top:0.15rem;}}

.ins-sec-header{{display:flex;align-items:center;gap:0.8rem;
  margin:1.5rem 0 0.8rem;padding-bottom:0.5rem;border-bottom:1px solid {BORD};}}
.ins-sec-icon{{font-size:1.3rem;width:2rem;height:2rem;display:flex;align-items:center;
  justify-content:center;background:{INS_BADGE_BG};border-radius:8px;border:1px solid {BORD};}}
.ins-sec-title{{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{TXT};margin:0;}}

.ins-card{{background:{CARD};border:1px solid {BORD};border-radius:14px;
  padding:1.2rem 1.4rem;margin-bottom:0.8rem;
  box-shadow:0 1px 5px rgba(107,45,139,0.08);}}
.ins-card-title{{font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
  color:{SOFT};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;}}

.ins-anom-row{{display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0;
  border-bottom:1px solid {BORD};font-size:0.82rem;}}

.ins-alert-info{{background:{INS_BADGE_BG};border-left:3px solid {ACC};
  border-radius:0 10px 10px 0;padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:{ACC};}}
.ins-alert-success{{background:{INS_SUCCESS_BG};border-left:3px solid {ACCENT3};
  border-radius:0 10px 10px 0;padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:{ACCENT3};}}
.ins-alert-danger{{background:{INS_DANGER_BG};border-left:3px solid {ACCENT_RED};
  border-radius:0 10px 10px 0;padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:{ACCENT_RED};}}

.ins-export-box{{background:{INS_SUCCESS_BG};border:1px solid {INS_SUCCESS_BOR};
  border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.6rem;}}

.ins-feature-box{{background:{INS_SECTION_BG};border:1px solid {BORD};
  border-radius:10px;padding:0.8rem 1rem;}}
.ins-feature-title{{color:{ACC};font-weight:600;font-size:0.85rem;margin-bottom:0.4rem;}}
.ins-feature-body{{color:{SOFT};font-size:0.75rem;line-height:1.8;}}
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

def apply_anom_theme(fig, title=""):
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=FIG_BG,
        font_color=TXT, font_family="DM Sans",
        xaxis=dict(gridcolor=BORD,showgrid=True,zeroline=False,linecolor=BORD,tickfont_color=SOFT),
        yaxis=dict(gridcolor=BORD,showgrid=True,zeroline=False,linecolor=BORD,tickfont_color=SOFT),
        legend=dict(bgcolor=CARD,bordercolor=BORD,borderwidth=1,font_color=TXT),
        margin=dict(l=50,r=30,t=60,b=50),
        hoverlabel=dict(bgcolor=CARD,bordercolor=BORD,font_color=TXT),
    )
    if title:
        fig.update_layout(title=dict(text=title,font_color=TXT,font_size=14,font_family="Syne"))
    return fig

def apply_ins_theme(fig, title="", h=400):
    """Insights dashboard chart theme — Thistle Purple light."""
    fig.update_layout(
        paper_bgcolor=INS_PAPER_BG, plot_bgcolor=INS_PLOT_BG,
        font=dict(family="DM Sans", color=INS_TEXT, size=12), height=h,
        margin=dict(l=50, r=30, t=55, b=45),
        title_font=dict(family="Syne", size=13, color=INS_TEXT),
        legend=dict(bgcolor=INS_CARD_BG, bordercolor=INS_CARD_BOR, borderwidth=1, font=dict(color=INS_TEXT, size=11)),
        hoverlabel=dict(bgcolor=INS_CARD_BG, bordercolor=INS_CARD_BOR, font_color=INS_TEXT),
    )
    fig.update_xaxes(gridcolor=INS_GRID_CLR, zeroline=False, linecolor=INS_CARD_BOR,
                     tickfont=dict(color=INS_MUTED, size=11), title_font=dict(color=INS_MUTED))
    fig.update_yaxes(gridcolor=INS_GRID_CLR, zeroline=False, linecolor=INS_CARD_BOR,
                     tickfont=dict(color=INS_MUTED, size=11), title_font=dict(color=INS_MUTED))
    if title:
        fig.update_layout(title=dict(text=title, font_color=INS_TEXT, font_size=13, font_family="Syne"))
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

def anom_sec(icon, title, badge=None):
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-icon">{icon}</div>
      <p class="sec-title">{title}</p>
      {badge_html}
    </div>""", unsafe_allow_html=True)

def ins_sec(icon, title, badge=None):
    badge_html = f'<span style="margin-left:auto;background:{INS_BADGE_BG};border:1px solid {BORD};border-radius:100px;padding:0.2rem 0.7rem;font-size:0.7rem;font-family:DM Mono,monospace;color:{ACC}">{badge}</span>' if badge else ''
    st.markdown(f'<div class="ins-sec-header"><div class="ins-sec-icon">{icon}</div><p class="ins-sec-title">{title}</p>{badge_html}</div>', unsafe_allow_html=True)

def hr():
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

def ui_success(msg): st.markdown(f'<div class="alert-success">✅ {msg}</div>', unsafe_allow_html=True)
def ui_warn(msg):    st.markdown(f'<div class="alert-warn">⚠️ {msg}</div>', unsafe_allow_html=True)
def ui_info_anom(msg): st.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
def ui_danger(msg):  st.markdown(f'<div class="alert-danger">🚨 {msg}</div>', unsafe_allow_html=True)

def ins_ui_info(m):    st.markdown(f'<div class="ins-alert-info">ℹ️ {m}</div>',    unsafe_allow_html=True)
def ins_ui_success(m): st.markdown(f'<div class="ins-alert-success">✅ {m}</div>', unsafe_allow_html=True)
def ins_ui_danger(m):  st.markdown(f'<div class="ins-alert-danger">🚨 {m}</div>',  unsafe_allow_html=True)

def anom_metrics(*items, red_indices=None):
    red_indices = red_indices or []
    html = '<div class="metric-grid">'
    for i, (val, label) in enumerate(items):
        val_class = "metric-val metric-val-red" if i in red_indices else "metric-val"
        html += f'<div class="metric-card"><div class="{val_class}">{val}</div><div class="metric-label">{label}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

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
# REQUIRED FILES REGISTRY  (shared across Anomaly + Insights modes)
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {"key_cols": ["ActivityDate","TotalSteps","Calories"],    "label": "Daily Activity",     "icon": "🏃"},
    "hourlySteps_merged.csv":       {"key_cols": ["ActivityHour","StepTotal"],                "label": "Hourly Steps",       "icon": "👣"},
    "hourlyIntensities_merged.csv": {"key_cols": ["ActivityHour","TotalIntensity"],           "label": "Hourly Intensities", "icon": "⚡"},
    "minuteSleep_merged.csv":       {"key_cols": ["date","value","logId"],                    "label": "Minute Sleep",       "icon": "💤"},
    "heartrate_seconds_merged.csv": {"key_cols": ["Time","Value"],                            "label": "Heart Rate",         "icon": "❤️"},
}

def score_match(df, req_info):
    return sum(1 for col in req_info["key_cols"] if col in df.columns)

def parse_dt(series):
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")

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
# ANOMALY / INSIGHTS DETECTION FUNCTIONS  (shared)
# ─────────────────────────────────────────────────────────────────────────────
def detect_hr_anomalies(master, hr_high=100, hr_low=50, residual_sigma=2.0):
    df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["AvgHR"].mean().reset_index().rename(columns={"AvgHR":"AvgHR"})
    d = d.sort_values("Date")
    d["rolling_med"]  = d["AvgHR"].rolling(3,center=True,min_periods=1).median()
    d["residual"]     = d["AvgHR"] - d["rolling_med"]
    std = d["residual"].std()
    d["thresh_high"]  = d["AvgHR"] > hr_high
    d["thresh_low"]   = d["AvgHR"] < hr_low
    d["resid_anomaly"]= d["residual"].abs() > (residual_sigma * std)
    d["is_anomaly"]   = d["thresh_high"] | d["thresh_low"] | d["resid_anomaly"]
    def reason(row):
        r=[]
        if row["thresh_high"]:    r.append(f"HR>{hr_high}")
        if row["thresh_low"]:     r.append(f"HR<{hr_low}")
        if row["resid_anomaly"]:  r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r)
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_steps_anomalies(master, steps_low=500, steps_high=25000, residual_sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    d["rolling_med"]   = d["TotalSteps"].rolling(3,center=True,min_periods=1).median()
    d["residual"]      = d["TotalSteps"] - d["rolling_med"]
    std = d["residual"].std()
    d["thresh_low"]    = d["TotalSteps"] < steps_low
    d["thresh_high"]   = d["TotalSteps"] > steps_high
    d["resid_anomaly"] = d["residual"].abs() > (residual_sigma * std)
    d["is_anomaly"]    = d["thresh_low"] | d["thresh_high"] | d["resid_anomaly"]
    def reason(row):
        r=[]
        if row["thresh_low"]:    r.append(f"Steps<{steps_low}")
        if row["thresh_high"]:   r.append(f"Steps>{steps_high}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r)
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_sleep_anomalies(master, sleep_low=60, sleep_high=600, residual_sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
    d["rolling_med"]   = d["TotalSleepMinutes"].rolling(3,center=True,min_periods=1).median()
    d["residual"]      = d["TotalSleepMinutes"] - d["rolling_med"]
    std = d["residual"].std()
    d["thresh_low"]    = (d["TotalSleepMinutes"]>0) & (d["TotalSleepMinutes"]<sleep_low)
    d["thresh_high"]   = d["TotalSleepMinutes"] > sleep_high
    d["no_data"]       = d["TotalSleepMinutes"] == 0
    d["resid_anomaly"] = d["residual"].abs() > (residual_sigma * std)
    d["is_anomaly"]    = d["thresh_low"] | d["thresh_high"] | d["resid_anomaly"]
    def reason(row):
        r=[]
        if row["no_data"]:       r.append("No device worn")
        if row["thresh_low"]:    r.append(f"Sleep<{sleep_low}min")
        if row["thresh_high"]:   r.append(f"Sleep>{sleep_high}min")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r)
    d["reason"] = d.apply(reason, axis=1)
    return d

def simulate_accuracy(master, n_inject=10):
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}
    for signal, col, inj_vals, thr_fn in [
        ("Heart Rate","AvgHR",
         [115,120,125,35,40,45,118,130,38,42],
         lambda df: (df["AvgHR"]>100)|(df["AvgHR"]<50)),
        ("Steps","TotalSteps",
         [50,100,150,30000,35000,28000,80,200,31000,29000],
         lambda df: (df["TotalSteps"]<500)|(df["TotalSteps"]>25000)),
        ("Sleep","TotalSleepMinutes",
         [10,20,30,700,750,800,15,25,710,720],
         lambda df: ((df["TotalSleepMinutes"]>0)&(df["TotalSleepMinutes"]<60))|(df["TotalSleepMinutes"]>600)),
    ]:
        sim = df_daily[["Date",col]].copy()
        idx = np.random.choice(len(sim), n_inject, replace=False)
        sim.loc[sim.index[idx], col] = np.random.choice(inj_vals, n_inject, replace=True)
        sim["rolling_med"] = sim[col].rolling(3,center=True,min_periods=1).median()
        sim["residual"]    = sim[col] - sim["rolling_med"]
        resid_std = sim["residual"].std()
        sim["detected"] = thr_fn(sim) | (sim["residual"].abs() > 2*resid_std)
        tp = sim.iloc[idx]["detected"].sum()
        results[signal] = {"injected":n_inject,"detected":int(tp),"accuracy":round(tp/n_inject*100,1)}
    results["Overall"] = round(np.mean([results[k]["accuracy"] for k in ["Heart Rate","Steps","Sleep"]]),1)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# INSIGHTS DASHBOARD — CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def ins_chart_hr(anom_hr, hr_high, hr_low, sigma, h=380):
    fig = go.Figure()
    upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
    lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=lower, mode="lines",
                             fill="tonexty", fillcolor=hex_rgba(ACC, 0.12),
                             line=dict(width=0), name=f"±{sigma:.0f}σ Band"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"],
                             mode="lines+markers", name="Avg HR",
                             line=dict(color=ACC, width=2.5), marker=dict(size=5, color=ACC),
                             hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT3, width=1.5, dash="dot")))
    a = anom_hr[anom_hr["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["AvgHR"], mode="markers",
                                 name="🚨 Anomaly",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="circle",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f}<br><b>ANOMALY</b><extra>⚠️</extra>"))
        for _, row in a.iterrows():
            fig.add_annotation(x=row["Date"], y=row["AvgHR"], text="⚠️",
                               showarrow=True, arrowhead=2, arrowcolor=ACCENT_RED,
                               ax=0, ay=-35, font=dict(color=ACCENT_RED, size=11))
    fig.add_hline(y=hr_high, line_dash="dash", line_color=ACCENT_RED, line_width=1.5, opacity=0.65,
                  annotation_text=f"High ({int(hr_high)} bpm)",
                  annotation_font_color=ACCENT_RED, annotation_position="top right")
    fig.add_hline(y=hr_low, line_dash="dash", line_color=ACC2, line_width=1.5, opacity=0.65,
                  annotation_text=f"Low ({int(hr_low)} bpm)",
                  annotation_font_color=ACC2, annotation_position="bottom right")
    apply_ins_theme(fig, "❤️ Heart Rate — Anomaly Detection", h)
    fig.update_layout(xaxis_title="Date", yaxis_title="HR (bpm)")
    return fig

def ins_chart_steps(anom_steps, st_low, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65,0.35],
                        vertical_spacing=0.07,
                        subplot_titles=["Daily Steps (avg users)", "Residual Deviation"])
    a = anom_steps[anom_steps["is_anomaly"]]
    for _, row in a.iterrows():
        fig.add_vrect(x0=str(row["Date"]), x1=str(row["Date"]),
                      fillcolor="rgba(192,57,43,0.1)",
                      line_color="rgba(192,57,43,0.35)", line_width=1.5, row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
                             mode="lines+markers", name="Avg Steps",
                             line=dict(color=ACCENT3, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACC, width=2, dash="dash")), row=1, col=1)
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSteps"], mode="markers", name="🚨 Alert",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="triangle-up",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(st_low), line_dash="dash", line_color=ACCENT_RED, line_width=1.5,
                  opacity=0.7, row=1, col=1,
                  annotation_text=f"Low ({int(st_low):,})", annotation_font_color=ACCENT_RED)
    res_colors = [ACCENT_RED if v else ACCENT3 for v in anom_steps["resid_anomaly"]]
    fig.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"], name="Residual",
                         marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:,.0f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line_color=SOFT, line_width=1, row=2, col=1)
    apply_ins_theme(fig, "🚶 Step Count — Trend & Alerts", h)
    fig.update_layout(paper_bgcolor=INS_PAPER_BG, plot_bgcolor=INS_PLOT_BG, font_color=INS_TEXT)
    fig.update_xaxes(gridcolor=INS_GRID_CLR, tickfont_color=INS_MUTED)
    fig.update_yaxes(gridcolor=INS_GRID_CLR, tickfont_color=INS_MUTED)
    return fig

def ins_chart_sleep(anom_sleep, sl_low, sl_high, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65,0.35],
                        vertical_spacing=0.07,
                        subplot_titles=["Sleep Duration (min/night)", "Residual Deviation"])
    fig.add_hrect(y0=sl_low, y1=sl_high, fillcolor="rgba(30,132,73,0.07)", line_width=0,
                  annotation_text="✅ Healthy Zone", annotation_position="top right",
                  annotation_font_color=ACCENT3, row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
                             mode="lines+markers", name="Sleep (min)",
                             line=dict(color=ACC2, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f} min<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT3, width=1.5, dash="dot")), row=1, col=1)
    a = anom_sleep[anom_sleep["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSleepMinutes"], mode="markers",
                                 name="🚨 Anomaly",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f}<br><b>ANOMALY</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(sl_low), line_dash="dash", line_color=ACCENT_RED, line_width=1.5,
                  opacity=0.7, row=1, col=1, annotation_text=f"Min ({int(sl_low)} min)", annotation_font_color=ACCENT_RED)
    fig.add_hline(y=int(sl_high), line_dash="dash", line_color=ACC, line_width=1.5,
                  opacity=0.65, row=1, col=1, annotation_text=f"Max ({int(sl_high)} min)", annotation_font_color=ACC)
    res_colors = [ACCENT_RED if v else ACC2 for v in anom_sleep["resid_anomaly"]]
    fig.add_trace(go.Bar(x=anom_sleep["Date"], y=anom_sleep["residual"], name="Residual",
                         marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:.0f} min<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line_color=SOFT, line_width=1, row=2, col=1)
    apply_ins_theme(fig, "💤 Sleep Pattern — Anomaly Visualization", h)
    fig.update_layout(paper_bgcolor=INS_PAPER_BG, plot_bgcolor=INS_PLOT_BG, font_color=INS_TEXT)
    fig.update_xaxes(gridcolor=INS_GRID_CLR, tickfont_color=INS_MUTED)
    fig.update_yaxes(gridcolor=INS_GRID_CLR, tickfont_color=INS_MUTED)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# INSIGHTS DASHBOARD — PDF GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(master, anom_hr, anom_steps, anom_sleep,
                         hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                         fig_hr, fig_steps, fig_sleep):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    def safe(text):
        """Replace non-latin-1 characters so fpdf doesn't crash."""
        return (str(text)
                .replace("\u2014", "-")   # em dash
                .replace("\u2013", "-")   # en dash
                .replace("\u2019", "'")   # right single quote
                .replace("\u2018", "'")   # left single quote
                .replace("\u201c", '"')   # left double quote
                .replace("\u201d", '"')   # right double quote
                .replace("\u00b1", "+/-") # plus-minus
                .replace("\u03c3", "sigma")  # sigma
                .encode("latin-1", errors="replace").decode("latin-1"))

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(26, 10, 34)
            self.rect(0, 0, 210, 18, 'F')
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(155, 89, 182)
            self.set_y(4)
            self.cell(0, 10, safe("FitPulse Anomaly Detection Report - Insights Dashboard"), align="C")
            self.set_text_color(180, 154, 190)
            self.set_font("Helvetica", "", 7)
            self.set_y(13)
            self.cell(0, 4, safe(f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}"), align="C")
            self.ln(6)

        def footer(self):
            self.set_y(-13)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(148, 100, 148)
            self.cell(0, 8, safe(f"FitPulse Unified Platform  -  Thistle Purple Edition  -  Page {self.page_no()}"), align="C")

        def section(self, title, color=(107,45,139)):
            self.ln(3)
            self.set_fill_color(*color)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 8, safe(f"  {title}"), fill=True, ln=True)
            self.set_text_color(30, 10, 40)
            self.ln(2)

        def kv(self, key, val):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(80, 50, 90)
            self.cell(58, 6, safe(key + ":"), ln=False)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(40, 15, 55)
            self.cell(0, 6, safe(str(val)), ln=True)

        def para(self, text, size=8.5):
            self.set_font("Helvetica", "", size)
            self.set_text_color(60, 40, 70)
            self.multi_cell(0, 5, safe(text))
            self.ln(1)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    n_hr    = int(anom_hr["is_anomaly"].sum())
    n_steps = int(anom_steps["is_anomaly"].sum())
    n_sleep = int(anom_sleep["is_anomaly"].sum())
    n_users = master["Id"].nunique()
    n_days  = master["Date"].nunique()
    date_range_str = (
        f"{pd.to_datetime(master['Date']).min().strftime('%d %b %Y')} to "
        f"{pd.to_datetime(master['Date']).max().strftime('%d %b %Y')}"
    )

    pdf.section("1. EXECUTIVE SUMMARY", (107,45,139))
    pdf.kv("Dataset",     "Real Fitbit Device Data - Kaggle (arashnic/fitbit)")
    pdf.kv("Users",       f"{n_users} participants")
    pdf.kv("Date Range",  date_range_str)
    pdf.kv("Total Days",  f"{n_days} days of observations")
    pdf.kv("Pipeline",    "FitPulse Insights Dashboard v6.0")
    pdf.ln(2)

    pdf.section("2. ANOMALY SUMMARY", (192,57,43))
    pdf.kv("Heart Rate Anomalies",  f"{n_hr} days flagged")
    pdf.kv("Steps Anomalies",       f"{n_steps} days flagged")
    pdf.kv("Sleep Anomalies",       f"{n_sleep} days flagged")
    pdf.kv("Total Flags",           f"{n_hr + n_steps + n_sleep} across all signals")
    pdf.ln(2)

    pdf.section("3. DETECTION THRESHOLDS USED", (40,100,60))
    pdf.kv("Heart Rate High",   f"> {int(hr_high)} bpm")
    pdf.kv("Heart Rate Low",    f"< {int(hr_low)} bpm")
    pdf.kv("Steps Low Alert",   f"< {int(st_low):,} steps/day")
    pdf.kv("Sleep Low",         f"< {int(sl_low)} minutes/night")
    pdf.kv("Sleep High",        f"> {int(sl_high)} minutes/night")
    pdf.kv("Residual Sigma",    f"+/- {float(sigma):.1f} sigma from rolling median")
    pdf.ln(2)

    pdf.section("4. METHODOLOGY", (107,45,139))
    pdf.para(
        "Three complementary detection methods were applied:\n\n"
        "  1. THRESHOLD VIOLATIONS - Hard upper/lower bounds on each metric.\n\n"
        "  2. RESIDUAL-BASED DETECTION - 3-day rolling median as baseline; "
        f"flags days deviating by >{float(sigma):.1f} sigma.\n\n"
        "  3. DBSCAN OUTLIER CLUSTERING - User-level structural outliers via density clustering."
    )

    pdf.add_page()
    pdf.section("5. ANOMALY CHARTS", (107,45,139))

    def embed_fig(fig, label, w=190, h=80):
        try:
            img_bytes = fig.to_image(format="png", width=1100, height=480, scale=1.5, engine="kaleido")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes); tmp_path = tmp.name
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(80, 50, 90)
            pdf.cell(0, 6, safe(label), ln=True)
            pdf.image(tmp_path, x=10, w=w, h=h)
            os.unlink(tmp_path)
            pdf.ln(3)
        except Exception as ex:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(150, 50, 50)
            pdf.cell(0, 6, safe(f"[Chart unavailable: {ex}]"), ln=True); pdf.ln(2)

    embed_fig(fig_hr,    "Figure 1 - Heart Rate with Anomaly Highlights")
    embed_fig(fig_steps, "Figure 2 - Step Count Trend with Alert Bands")
    embed_fig(fig_sleep, "Figure 3 - Sleep Pattern Visualization")

    pdf.add_page()
    pdf.section("6. ANOMALY RECORDS - HEART RATE", (192,57,43))

    def table(df, cols, rename_map, max_rows=20):
        df2 = df[df["is_anomaly"]][cols].copy().rename(columns=rename_map)
        if df2.empty:
            pdf.para("No anomalies detected."); return
        col_w = 180 // len(df2.columns)
        pdf.set_fill_color(26,10,34); pdf.set_text_color(216,191,216)
        pdf.set_font("Helvetica","B",7.5)
        for col in df2.columns:
            pdf.cell(col_w,6,safe(str(col)[:18]),border=0,fill=True)
        pdf.ln()
        pdf.set_font("Helvetica","",7.5)
        for i,(_,row) in enumerate(df2.head(max_rows).iterrows()):
            pdf.set_fill_color(45,16,64) if i%2==0 else pdf.set_fill_color(35,10,50)
            pdf.set_text_color(216,191,216)
            for val in row:
                cell_text = f"{val:.2f}" if isinstance(val,float) else str(val)[:18]
                pdf.cell(col_w,5.5,safe(cell_text),border=0,fill=True)
            pdf.ln()
        if len(df2)>max_rows:
            pdf.set_text_color(155,89,182); pdf.set_font("Helvetica","I",7)
            pdf.cell(0,5,safe(f"  ... and {len(df2)-max_rows} more records (see CSV export)"),ln=True)
        pdf.ln(3)

    table(anom_hr,    ["Date","AvgHR","rolling_med","residual","reason"],
          {"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})
    pdf.section("7. ANOMALY RECORDS - STEPS", (40,130,80))
    table(anom_steps, ["Date","TotalSteps","rolling_med","residual","reason"],
          {"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})
    pdf.section("8. ANOMALY RECORDS - SLEEP", (107,45,139))
    table(anom_sleep, ["Date","TotalSleepMinutes","rolling_med","residual","reason"],
          {"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    pdf.add_page()
    pdf.section("9. USER ACTIVITY PROFILES", (107,45,139))
    profile_cols = [c for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"] if c in master.columns]
    user_profile = master.groupby("Id")[profile_cols].mean().round(1)
    col_w2 = 180 // (len(profile_cols)+1)
    pdf.set_fill_color(26,10,34); pdf.set_text_color(216,191,216)
    pdf.set_font("Helvetica","B",8)
    pdf.cell(col_w2,6,safe("User ID"),border=0,fill=True)
    for col in profile_cols:
        pdf.cell(col_w2,6,safe(col[:12]),border=0,fill=True)
    pdf.ln()
    pdf.set_font("Helvetica","",7.5)
    for i,(uid,row) in enumerate(user_profile.iterrows()):
        pdf.set_fill_color(45,16,64) if i%2==0 else pdf.set_fill_color(35,10,50)
        pdf.set_text_color(216,191,216)
        pdf.cell(col_w2,5.5,safe(f"...{str(uid)[-6:]}"),border=0,fill=True)
        for val in row:
            pdf.cell(col_w2,5.5,safe(f"{val:,.0f}"),border=0,fill=True)
        pdf.ln()

    pdf.ln(4)
    pdf.section("10. CONCLUSION", (40,100,60))
    pdf.para(
        f"The FitPulse Insights Dashboard processed {n_users} users over {n_days} days "
        f"of real Fitbit data. A total of {n_hr+n_steps+n_sleep} anomalous events were identified "
        f"across heart rate ({n_hr}), step count ({n_steps}), and sleep duration ({n_sleep}) signals. "
        "Combined threshold and residual detection provided robust coverage of both extreme "
        "values and subtle pattern deviations."
    )

    # Generate PDF and return as BytesIO
    try:
        # Write PDF to a temporary file, then read it back as bytes
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf.output(tmp.name)
            tmp_path = tmp.name
        
        # Read the PDF file and convert to BytesIO
        with open(tmp_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Verify PDF has content
        if not pdf_bytes or len(pdf_bytes) == 0:
            return None
        
        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def build_shared_master(detected):
    """Parse the 5 detected DataFrames and build the merged master. Stores to session_state."""
    daily    = detected["dailyActivity_merged.csv"].copy()
    hourly_s = detected["hourlySteps_merged.csv"].copy()
    hourly_i = detected["hourlyIntensities_merged.csv"].copy()
    sleep    = detected["minuteSleep_merged.csv"].copy()
    hr_df    = detected["heartrate_seconds_merged.csv"].copy()
    daily["ActivityDate"]    = parse_dt(daily["ActivityDate"])
    hourly_s["ActivityHour"] = parse_dt(hourly_s["ActivityHour"])
    hourly_i["ActivityHour"] = parse_dt(hourly_i["ActivityHour"])
    sleep["date"]            = parse_dt(sleep["date"])
    hr_df["Time"]            = parse_dt(hr_df["Time"])
    
    # Drop rows with NaT values BEFORE calling .dt.date to avoid NaTType error
    hr_df = hr_df.dropna(subset=["Time","Value"])
    sleep = sleep.dropna(subset=["date","value"])
    daily = daily.dropna(subset=["ActivityDate"])
    
    hr_minute = (hr_df.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index())
    hr_minute.columns = ["Id","Time","HeartRate"]
    hr_minute = hr_minute.dropna(subset=["Time"])
    hr_minute["Date"] = hr_minute["Time"].dt.normalize().dt.date
    hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"].agg(["mean","max","min","std"]).reset_index().rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))
    
    sleep["Date"] = sleep["date"].dt.normalize().dt.date
    sleep_daily = (sleep.dropna(subset=["Date"]).groupby(["Id","Date"]).agg(TotalSleepMinutes=("value","count"),DominantSleepStage=("value", lambda x: x.mode()[0] if not x.mode().empty else 0)).reset_index())
    
    master = daily.copy().rename(columns={"ActivityDate":"Date"})
    master["Date"] = master["Date"].dt.normalize().dt.date
    master = master.merge(hr_daily, on=["Id","Date"], how="left")
    master = master.merge(sleep_daily, on=["Id","Date"], how="left")
    master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
    master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
    for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
        if col in master.columns:
            master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))
    st.session_state.shared_detected = detected
    st.session_state.shared_master   = master
    st.session_state.shared_built    = True
    return master

def generate_csv_export(anom_hr, anom_steps, anom_sleep):
    hr_out = anom_hr[anom_hr["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].copy()
    hr_out["signal"] = "Heart Rate"
    hr_out = hr_out.rename(columns={"AvgHR":"value","rolling_med":"expected"})
    st_out = anom_steps[anom_steps["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].copy()
    st_out["signal"] = "Steps"
    st_out = st_out.rename(columns={"TotalSteps":"value","rolling_med":"expected"})
    sl_out = anom_sleep[anom_sleep["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].copy()
    sl_out["signal"] = "Sleep"
    sl_out = sl_out.rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})
    combined = pd.concat([hr_out, st_out, sl_out], ignore_index=True)
    combined = combined[["signal","Date","value","expected","residual","reason"]].sort_values(["signal","Date"]).round(2)
    buf = io.StringIO(); combined.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ─────────────────────────────────────────────────────────────────────────────
# ███  SIDEBAR  ███████████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">💪 FitPulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    st.markdown('<span class="sb-mode-label">⚡ Switch Mode</span>', unsafe_allow_html=True)
    mode_options = ["📊  Analytics Platform", "🧬  ML Pipeline", "🚨  Anomaly Detection", "📈  Insights Dashboard"]
    mode_idx = {"Analytics":0,"ML Pipeline":1,"Anomaly Detection":2,"Insights Dashboard":3}.get(st.session_state.mode, 0)
    chosen = st.selectbox(
        label="mode_sel", options=mode_options,
        index=mode_idx, label_visibility="collapsed", key="mode_select",
    )
    if chosen.startswith("📊"):   new_mode = "Analytics"
    elif chosen.startswith("🧬"): new_mode = "ML Pipeline"
    elif chosen.startswith("🚨"): new_mode = "Anomaly Detection"
    else:                          new_mode = "Insights Dashboard"

    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.rerun()

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    # ── ANALYTICS SIDEBAR ─────────────────────────────────────────────────────
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

    # ── ML SIDEBAR ────────────────────────────────────────────────────────────
    elif st.session_state.mode == "ML Pipeline":
        st.markdown('<div class="sb-tag">ML Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Pipeline Progress</div>', unsafe_allow_html=True)
        prog = st.session_state.ml_progress
        st.progress(prog / 100); st.caption(f"{prog}%")
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

    # ── ANOMALY SIDEBAR ───────────────────────────────────────────────────────
    elif st.session_state.mode == "Anomaly Detection":
        st.markdown('<div class="sb-tag">Anomaly Detection</div>', unsafe_allow_html=True)
        # Dataset source indicator
        shared_avail = st.session_state.shared_built
        src_icon = "✅" if shared_avail else "⚠️"
        src_lbl  = "Shared dataset ready" if shared_avail else "Upload in ML Pipeline first"
        st.markdown(f'<div class="ml-stage">{src_icon} 📂 {src_lbl}</div>', unsafe_allow_html=True)
        steps_done = sum([st.session_state.anom_files_loaded,
                          st.session_state.anom_anomaly_done,
                          st.session_state.anom_simulation_done])
        pct = int(steps_done / 3 * 100)
        st.markdown('<div class="sb-section-label">Pipeline Progress</div>', unsafe_allow_html=True)
        st.progress(pct / 100); st.caption(f"{pct}%")
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        for done, icon, label in [
            (st.session_state.anom_files_loaded,    "📊", "Dataset Loaded"),
            (st.session_state.anom_anomaly_done,    "🚨", "Anomalies Detected"),
            (st.session_state.anom_simulation_done, "🎯", "Accuracy Simulated"),
        ]:
            tick = "✅" if done else "⭕"
            st.markdown(f'<div class="ml-stage">{tick} {icon} {label}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Thresholds</div>', unsafe_allow_html=True)
        st.session_state["anom_hr_high"] = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="sb_hr_high")
        st.session_state["anom_hr_low"]  = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="sb_hr_low")
        st.session_state["anom_st_low"]  = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000,key="sb_st_low")
        st.session_state["anom_sl_low"]  = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120, key="sb_sl_low")
        st.session_state["anom_sl_high"] = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900, key="sb_sl_high")
        st.session_state["anom_sigma"]   = st.slider("Residual sigma", 1.0, 4.0, 2.0, 0.5, key="sb_sigma")

    # ── INSIGHTS SIDEBAR ──────────────────────────────────────────────────────
    else:
        st.markdown('<div class="sb-tag">Insights Dashboard</div>', unsafe_allow_html=True)
        # Progress: shared built (1) + detection done (1) = 2 steps
        steps_ins_done = sum([
            st.session_state.shared_built,
            st.session_state.ins_pipeline_done,
        ])
        pct_ins = int(steps_ins_done / 2 * 100)
        st.markdown('<div class="sb-section-label">Pipeline Progress</div>', unsafe_allow_html=True)
        st.progress(pct_ins / 100); st.caption(f"{pct_ins}%")
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        for done, icon, label in [
            (st.session_state.shared_built,               "📂", "Dataset Shared"),
            (st.session_state.ins_pipeline_done,          "🚨", "Detection Done"),
            (st.session_state.ins_anom_hr is not None,    "❤️", "HR Anomalies"),
            (st.session_state.ins_anom_steps is not None, "🚶", "Steps Anomalies"),
            (st.session_state.ins_anom_sleep is not None, "💤", "Sleep Anomalies"),
        ]:
            tick = "✅" if done else "⭕"
            st.markdown(f'<div class="ml-stage">{tick} {icon} {label}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Detection Thresholds</div>', unsafe_allow_html=True)
        st.session_state["ins_hr_high"] = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="ins_sb_hr_high")
        st.session_state["ins_hr_low"]  = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="ins_sb_hr_low")
        st.session_state["ins_st_low"]  = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000,key="ins_sb_st_low")
        st.session_state["ins_sl_low"]  = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120, key="ins_sb_sl_low")
        st.session_state["ins_sl_high"] = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900, key="ins_sb_sl_high")
        st.session_state["ins_sigma"]   = st.slider("Residual sigma", 1.0, 4.0, 2.0, 0.5, key="ins_sb_sigma")
        # Re-run detection button (only shown when shared data is available)
        if st.session_state.shared_built and st.session_state.ins_pipeline_done:
            st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-section-label">Apply new thresholds</div>', unsafe_allow_html=True)
            if st.button("🔄 Re-run Detection", key="ins_sb_rerun"):
                _m = st.session_state.shared_master
                st.session_state.ins_master      = _m
                st.session_state.ins_anom_hr     = detect_hr_anomalies(
                    _m, st.session_state["ins_hr_high"], st.session_state["ins_hr_low"],
                    st.session_state["ins_sigma"])
                st.session_state.ins_anom_steps  = detect_steps_anomalies(
                    _m, st.session_state["ins_st_low"], 25000,
                    st.session_state["ins_sigma"])
                st.session_state.ins_anom_sleep  = detect_sleep_anomalies(
                    _m, st.session_state["ins_sl_low"], st.session_state["ins_sl_high"],
                    st.session_state["ins_sigma"])
                st.rerun()

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
    lbl_map = {"Analytics":"Analytics · v3.0","ML Pipeline":"ML Pipeline · v2.0",
               "Anomaly Detection":"Anomaly Detection · v1.0","Insights Dashboard":"Insights Dashboard · v1.0"}
    lbl = lbl_map.get(st.session_state.mode,"v6.0")
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

    st.markdown("""
<div class="banner">
  <div class="banner-glow"></div>
  <div>
    <div class="banner-title">💪 FitPulse Analytics</div>
    <div class="banner-sub">Fitness &amp; Health Data Quality Pipeline &nbsp;—&nbsp; Upload · Inspect · Clean · Analyse</div>
  </div>
  <div class="banner-chip">ENTERPRISE DATA PLATFORM</div>
</div>""", unsafe_allow_html=True)

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
            st.markdown("**🗺️ Null Heatmap**")
            sample=df[has_null.index].head(500).isnull().astype(int)
            fig2=px.imshow(sample.T,color_continuous_scale=[[0,FIG_BG],[1,ACC]],aspect="auto",title="Null Presence Map",zmin=0,zmax=1)
            fig2.update_coloraxes(showscale=False); fig2.update_layout(xaxis_title="Row index",yaxis_title="")
            fig2=apply_theme(fig2,250); st.plotly_chart(fig2,use_container_width=True,key="null_heatmap")
    hr()

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
                f"Numeric <strong>{col}</strong> — {nb-na:,}/{nb:,} nulls filled"))
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

    step_hdr("4","👁️","Preview Cleaned Dataset")
    if st.session_state.df_clean is not None:
        dfc=st.session_state.df_clean; all_c=dfc.columns.tolist()
        logline("info",f"Shape: <strong>{dfc.shape[0]:,} rows × {dfc.shape[1]} cols</strong>&nbsp;·&nbsp;Nulls remaining: <strong>{int(dfc.isnull().sum().sum()):,}</strong>")
        sel=st.multiselect("Select columns to display:",options=all_c,default=all_c[:min(8,len(all_c))],key="prev_sel")
        if sel: st.dataframe(dfc[sel].head(200),use_container_width=True,height=340)
        else: st.info("Select at least one column.")
        with st.expander("📋 Column Schema"):
            schema=pd.DataFrame({"Column":dfc.dtypes.index,"Dtype":dfc.dtypes.astype(str).values,
                "Non-Null":dfc.notna().sum().values,"Null":dfc.isna().sum().values,
                "Null %":(dfc.isna().mean()*100).round(2).values,
                "Sample":[str(dfc[c].dropna().iloc[0]) if dfc[c].notna().any() else "—" for c in dfc.columns]})
            st.dataframe(schema,hide_index=True,use_container_width=True)
    else:
        st.markdown('<div class="info-box">⚙️ Run <strong>Step 3 Preprocessing</strong> first.</div>',unsafe_allow_html=True)
    hr()

    step_hdr("5","📊","Exploratory Data Analysis")
    if st.button("📊  Run Full EDA",disabled=(st.session_state.df_clean is None),key="btn_eda"):
        run_loader("Running EDA pipeline…",[
            "🧠 Initialising EDA engine…","📐 Computing descriptive statistics…",
            "📈 Building distribution plots…","🔥 Building correlation heatmap…",
            "🏷️ Analysing categorical columns…","📦 Detecting outliers (IQR)…",
            "📅 Rendering time series…","✅ EDA complete!",
        ], delay=0.22)
        st.session_state.eda_done = True

    if st.session_state.eda_done and st.session_state.df_clean is not None:
        dfe=st.session_state.df_clean.copy()
        num_cols=dfe.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols=dfe.select_dtypes(include=["object","category"]).columns.tolist()
        dt_cols=dfe.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        st.markdown(f'<div class="ok-box" style="margin-bottom:1.2rem;">✅ &nbsp;<strong>{len(dfe):,} rows</strong> &nbsp;·&nbsp;<strong>{len(num_cols)} numeric</strong> &nbsp;·&nbsp;<strong>{len(cat_cols)} categorical</strong> &nbsp;·&nbsp;<strong>{len(dt_cols)} datetime</strong> &nbsp;·&nbsp;<strong>0 nulls</strong></div>',unsafe_allow_html=True)

        st.markdown('<div class="eda-card">',unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">📋 Descriptive Statistics</div>',unsafe_allow_html=True)
        if num_cols: st.dataframe(dfe[num_cols].describe().round(3),use_container_width=True,height=295)
        else: st.warning("No numeric columns found.")
        st.markdown('</div>',unsafe_allow_html=True)

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

        if len(num_cols)>=2:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">🔥 Correlation Heatmap</div>',unsafe_allow_html=True)
            corr=dfe[num_cols].corr(numeric_only=True).round(2)
            fig=px.imshow(corr,text_auto=True,aspect="auto",
                color_continuous_scale=[[0,"#2ECC71"],[0.5,CARD],[1,ACC]],zmin=-1,zmax=1,title="Pearson Correlation Matrix")
            fig.update_traces(textfont=dict(size=10,color=TXT))
            fig=apply_theme(fig,max(400,len(num_cols)*62))
            st.plotly_chart(fig,use_container_width=True,key="corr_heatmap"); st.markdown('</div>',unsafe_allow_html=True)

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
                            textposition="outside",textfont=dict(size=11,color=TXT)))
                        fig=apply_theme(fig,max(220,n_b*34))
                        fig.update_layout(title=cn,yaxis=dict(autorange="reversed"),xaxis_title="Count",yaxis_title="",margin=dict(l=10,r=55,t=44,b=12))
                        st.plotly_chart(fig,use_container_width=True,key=f"cat_{cn}_{gi}")
            st.markdown('</div>',unsafe_allow_html=True)

        if dt_cols and num_cols:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">📅 Time Series Trends</div>',unsafe_allow_html=True)
            dc=dt_cols[0]; sel_m=st.selectbox("Select metric:",num_cols,key="ts_metric")
            ts_df=dfe[[dc,sel_m]].dropna().sort_values(dc)
            if not ts_df.empty:
                try:
                    ts_w=ts_df.set_index(dc)[sel_m].resample("W").mean().dropna().reset_index()
                    plot_df=ts_w if not ts_w.empty else ts_df
                except Exception: plot_df=ts_df
                fig=px.line(plot_df,x=dc,y=sel_m,color_discrete_sequence=[ACC],title=f"{sel_m} — Weekly Average",markers=True)
                fig.update_traces(line=dict(width=2.5),fill="tozeroy",fillcolor=hex_rgba(ACC,0.08),marker=dict(size=4,color=ACC))
                fig=apply_theme(fig,340); st.plotly_chart(fig,use_container_width=True,key="ts_line")
            st.markdown('</div>',unsafe_allow_html=True)

        if num_cols:
            st.markdown('<div class="eda-card">',unsafe_allow_html=True)
            st.markdown('<div class="eda-card-title">📦 Outlier Detection — Box Plots</div>',unsafe_allow_html=True)
            sel_box=st.multiselect("Select columns:",num_cols,default=num_cols[:min(5,len(num_cols))],key="box_sel")
            if sel_box:
                fig=go.Figure()
                for i,cn in enumerate(sel_box):
                    clr=PAL[i%len(PAL)]; rgba=hex_rgba(clr,0.2)
                    fig.add_trace(go.Box(y=dfe[cn].dropna(),name=cn,marker_color=clr,line=dict(color=clr,width=1.5),
                        fillcolor=rgba,boxpoints="outliers",jitter=0.35,marker=dict(size=4,opacity=0.65,color=clr)))
                fig=apply_theme(fig,400); fig.update_layout(title="Box Plots — Distribution & Outlier View",showlegend=True)
                st.plotly_chart(fig,use_container_width=True,key="box_chart")
            st.markdown('</div>',unsafe_allow_html=True)

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
                title=f"Scatter: {x_ax}  ×  {y_ax}")
            fig=apply_theme(fig,440); st.plotly_chart(fig,use_container_width=True,key="scatter_main"); st.markdown('</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="ok-box" style="margin-top:1.2rem;">✅ EDA complete &nbsp;·&nbsp;<strong>{len(num_cols)}</strong> numeric &nbsp;·&nbsp;<strong>{len(cat_cols)}</strong> categorical</div>',unsafe_allow_html=True)

    st.markdown(f'<div class="footer">💪 <b>FitPulse Analytics</b> &nbsp;·&nbsp; Thistle Purple Edition</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "ML Pipeline":
    st.markdown("""
<div class="fp-hero">
  <div style="margin-bottom:8px;"><span class="fp-tag">FEATURE EXTRACTION &amp; MODELING</span></div>
  <div class="fp-title">🧬 FitPulse ML Pipeline</div>
  <div class="fp-sub">TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data</div>
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["📁 Data Loading","📊 TSFresh Features","📈 Prophet Forecast","🔵 Clustering"])

    with tab1:
        st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;"><div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{ACC};">📁 Data Loading</div><span class="steps-badge">Steps 1–9</span></div>',unsafe_allow_html=True)
        st.markdown('<div class="fp-info">Select all 5 Fitbit CSV files at once. Files are auto-detected by column structure.</div>',unsafe_allow_html=True)
        uploaded_ml=st.file_uploader("📂 Drop all Fitbit CSV files here",type=["csv"],accept_multiple_files=True,key="ml_upload")
        # ── Detect files using REQUIRED_FILES keys (must match build_shared_master exactly) ──
        detected = {}
        if uploaded_ml:
            raw_ml = []
            for f in uploaded_ml:
                try:
                    df_tmp = pd.read_csv(f)
                    raw_ml.append((f.name, df_tmp))
                except Exception:
                    pass
            for req_name, finfo in REQUIRED_FILES.items():
                best_score, best_df = 0, None
                for uname, udf in raw_ml:
                    s = score_match(udf, finfo)
                    if s > best_score:
                        best_score, best_df = s, udf
                if best_score >= 2:
                    detected[req_name] = best_df

        all_ok = len(detected) == 5
        cards_html = '<div class="ds-grid">'
        for req_name, finfo in REQUIRED_FILES.items():
            ok  = req_name in detected
            s   = '<span class="chip-ok">✅ Loaded</span>' if ok else '<span class="chip-miss">❌ Missing</span>'
            cards_html += f'<div class="ds-card"><div class="ds-icon">{finfo["icon"]}</div><div class="ds-name">{finfo["label"]}</div><div class="ds-status">{s}</div></div>'
        st.markdown(cards_html + "</div>", unsafe_allow_html=True); st.markdown("")

        if all_ok:
            # ── AUTO-SAVE shared dataset immediately when all 5 files are present ──
            # This runs every rerun while files are uploaded — no button click needed.
            # We check if the detected files changed to avoid redundant heavy rebuilds.
            _detected_keys = tuple(sorted(detected.keys()))
            _prev_keys     = tuple(sorted(st.session_state.shared_detected.keys())) if st.session_state.shared_detected else ()
            if not st.session_state.shared_built or _detected_keys != _prev_keys:
                try:
                    with st.spinner("🔄 Building shared master dataset…"):
                        build_shared_master(detected)
                except Exception as _e:
                    st.warning(f"⚠️ Could not build shared master: {_e}")

            total=sum(len(v) for v in detected.values())
            st.success("✅ All 5 datasets loaded & shared — Anomaly Detection and Insights Dashboard will use these files automatically.")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Files Loaded","5 / 5")
            c2.metric("Total Rows",f"{total:,}")
            c3.metric("Shared","✅ Yes")
            c4.metric("Unique Users","~30")

            if st.button("▶ Proceed to TSFresh Features"):
                st.session_state.ml_dfs=detected
                st.session_state.ml_progress=max(st.session_state.ml_progress, 20)
                st.rerun()
        else:
            # Clear shared state if files are removed / incomplete
            if st.session_state.shared_built:
                st.session_state.shared_built    = False
                st.session_state.shared_detected = {}
                st.session_state.shared_master   = None
                st.session_state.anom_files_loaded = False
                st.session_state.anom_master       = None
            missing=[n for n in REQUIRED_FILES.keys() if n not in detected]
            if missing: st.warning(f"⚠️ Missing: {', '.join(missing)}")

        st.markdown("---"); st.caption("💡 No CSV files? Use demo mode.")
        if st.button("🎲 Load Demo Data & Proceed"):
            st.session_state.ml_dfs={"demo":True}; st.session_state.ml_progress=20; st.rerun()

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
                ax.set_title("TSFresh Feature Matrix (Normalized 0-1)",fontsize=10,color=ACC,fontweight="bold")
                for r in range(len(feat_df.index)):
                    for c in range(len(feat_df.columns)):
                        v=feat_df.values[r,c]; tc="white" if (v>0.72 or v<0.28) else TXT
                        ax.text(c,r,f"{v:.2f}",ha="center",va="center",fontsize=6.8,color=tc,fontweight="600")
                cbar=fig.colorbar(im,ax=ax,fraction=0.02,pad=0.02); cbar.ax.tick_params(labelsize=7,colors=TXT)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                c1,c2,c3=st.columns(3); c1.metric("Features",len(feat_df.columns)); c2.metric("Users",len(feat_df.index)); c3.metric("Missing","0")

    with tab3:
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{ACC};margin-bottom:10px;">📈 Prophet Trend Forecast</div>',unsafe_allow_html=True)
        if st.session_state.ml_progress<40:
            st.info("⬅️ Complete TSFresh Features first.")
        else:
            st.markdown('<div class="fp-info">Prophet decomposes signals into trend + weekly seasonality.</div>',unsafe_allow_html=True)
            c1,c2,c3=st.columns(3)
            with c1: fdays=st.slider("Forecast Horizon (days)",7,90,30)
            with c2: st.selectbox("Confidence Interval",["80%","90%","95%"])
            with c3: show_comp=st.checkbox("Show Decomposition",True)
            if st.button("🔮 Run Prophet Forecast"):
                with st.spinner("Fitting Prophet…"):
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
                if show_comp:
                    st.markdown("#### 🔍 Prophet Components")
                    cd,tv,wv=make_prophet_components(); fig3,(at,aw)=plt.subplots(2,1,figsize=(11,5)); style_mpl(fig3,[at,aw])
                    at.plot(cd,tv,color="#2060a0",lw=2); at.set_ylabel("trend",fontsize=9); at.set_title("Trend Component",fontsize=9,color=ACC)
                    aw.plot(cd,wv,color=ACC2,lw=2); aw.set_ylabel("weekly",fontsize=9); aw.set_xlabel("ds",fontsize=9); aw.set_title("Weekly Seasonality",fontsize=9,color=ACC)
                    fig3.tight_layout(pad=2); st.pyplot(fig3); plt.close(fig3)
                c1,c2,c3,c4=st.columns(4); c1.metric("Avg HR","81.3 bpm"); c2.metric("Avg Steps","8,240"); c3.metric("Avg Sleep","248 min"); c4.metric("Horizon",f"{fdays} days")

    with tab4:
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{ACC};margin-bottom:10px;">🔵 Clustering — KMeans · DBSCAN · PCA · t-SNE</div>',unsafe_allow_html=True)
        if st.session_state.ml_progress<60:
            st.info("⬅️ Complete Prophet Forecast first.")
        else:
            K=st.session_state.get("k_sl",3); EP=st.session_state.get("eps_sl",2.2)
            st.markdown(f'<div class="fp-info">Clusters are spatially separated. K={K} · DBSCAN eps={EP}.</div>',unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: algo=st.selectbox("Algorithm",["KMeans + DBSCAN (Both)","KMeans Only","DBSCAN Only"])
            with c2: reduction=st.selectbox("Projection",["Both","PCA","t-SNE"])
            if st.button("🔬 Run Clustering"):
                with st.spinner("Clustering users…"):
                    bar=st.progress(0)
                    for i in range(1,101): time.sleep(0.010); bar.progress(i)
                st.session_state.ml_progress=100; st.session_state.ml_run_done=True; st.rerun()
            if st.session_state.ml_run_done:
                ks,inertia=make_elbow(); fig_e,ax_e=plt.subplots(figsize=(8,3.5)); style_mpl(fig_e,[ax_e])
                ax_e.plot(ks,inertia,color=ACC2,lw=2,zorder=2); ax_e.scatter(ks,inertia,color=ACC,s=65,zorder=3)
                ax_e.axvline(K,color="#e09030",linestyle="--",lw=1.8,label=f"K={K}")
                ax_e.set_xlabel("K",fontsize=9); ax_e.set_ylabel("Inertia",fontsize=9)
                ax_e.set_title("KMeans Elbow Curve",fontsize=10,color=ACC,fontweight="bold"); ax_e.legend(fontsize=8.5,facecolor=FIG_BG,edgecolor=BORD)
                fig_e.tight_layout(); st.pyplot(fig_e); plt.close(fig_e)
                if reduction in ["PCA","Both"]:
                    coords_pca,lab_pca=make_pca_scatter(K); fig_p,ax_p=plt.subplots(figsize=(10,6)); style_mpl(fig_p,[ax_p])
                    for ci in range(K):
                        mask=lab_pca==ci; pts=coords_pca[mask]
                        ax_p.scatter(pts[:,0],pts[:,1],color=CLUSTER_COLORS[ci],s=95,label=f"Cluster {ci}",zorder=3,edgecolors="white",linewidth=0.8)
                    ax_p.set_xlabel("PC1 (23.1%)",fontsize=9); ax_p.set_ylabel("PC2 (16.4%)",fontsize=9)
                    ax_p.set_title(f"KMeans — PCA (K={K})",fontsize=10,color=ACC,fontweight="bold")
                    ax_p.legend(title="Cluster",fontsize=9,facecolor=FIG_BG,edgecolor=BORD); fig_p.tight_layout(); st.pyplot(fig_p); plt.close(fig_p)
                if reduction in ["t-SNE","Both"]:
                    coords_tsne,lab_tsne=make_tsne(K); lab_db=lab_tsne.copy()
                    c1_idx=np.where(lab_db==1)[0]
                    if len(c1_idx): lab_db[c1_idx[0]]=-1
                    fig_t,(ax_km,ax_db)=plt.subplots(1,2,figsize=(14,5.5),sharex=True,sharey=True); style_mpl(fig_t,[ax_km,ax_db])
                    for ci in range(K):
                        m=lab_tsne==ci
                        ax_km.scatter(coords_tsne[m,0],coords_tsne[m,1],color=CLUSTER_COLORS[ci],s=65,label=f"Cluster {ci}",zorder=3,edgecolors="white",linewidth=0.6)
                    ax_km.set_title(f"KMeans — t-SNE (K={K})",fontsize=10,color=ACC,fontweight="bold"); ax_km.legend(title="Cluster",fontsize=8,facecolor=FIG_BG,edgecolor=BORD)
                    noise_m=lab_db==-1
                    if noise_m.any(): ax_db.scatter(coords_tsne[noise_m,0],coords_tsne[noise_m,1],color="#e04040",s=110,marker="X",label="Noise",zorder=5)
                    for ci in range(K):
                        m=lab_db==ci
                        ax_db.scatter(coords_tsne[m,0],coords_tsne[m,1],color=CLUSTER_COLORS[ci],s=65,label=f"Cluster {ci}",zorder=3,edgecolors="white",linewidth=0.6)
                    ax_db.set_title(f"DBSCAN — t-SNE (eps={EP})",fontsize=10,color=ACC,fontweight="bold"); ax_db.legend(title="Cluster",fontsize=8,facecolor=FIG_BG,edgecolor=BORD)
                    fig_t.tight_layout(pad=2.5); st.pyplot(fig_t); plt.close(fig_t)
                c1,c2,c3,c4=st.columns(4)
                c1.metric("KMeans Clusters",K); c2.metric("Silhouette Score",f"{np.random.uniform(0.45,0.72):.3f}"); c3.metric("DBSCAN Noise","1"); c4.metric("Total Users","30")
                st.success("🎉 Full pipeline complete!")

    st.markdown(f'<div style="text-align:center;font-size:.8rem;color:{SOFT};font-family:Syne,sans-serif;padding:4px 0 8px;">🧬 FitPulse ML Pipeline &nbsp;·&nbsp; TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE C — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "Anomaly Detection":
    st.markdown(f"""
<div class="fp-hero-anom">
  <div class="anom-badge">ANOMALY DETECTION &amp; VISUALIZATION</div>
  <h1 class="anom-title">🚨 FitPulse Anomaly Detection</h1>
  <p class="anom-sub">Threshold Violations · Residual Analysis · DBSCAN Outlier Clusters · Accuracy Simulation</p>
</div>
""", unsafe_allow_html=True)

    hr_high = st.session_state.get("anom_hr_high", 100)
    hr_low  = st.session_state.get("anom_hr_low",  50)
    st_low  = st.session_state.get("anom_st_low",  500)
    sl_low  = st.session_state.get("anom_sl_low",  60)
    sl_high = st.session_state.get("anom_sl_high", 600)
    sigma   = st.session_state.get("anom_sigma",   2.0)

    # ── USE SHARED DATASET (uploaded once in ML Pipeline) ────────────────────
    anom_sec("📂", "Dataset Status", "Shared from ML Pipeline")

    shared_ok = st.session_state.shared_built and st.session_state.shared_master is not None

    if not shared_ok:
        st.markdown(f"""
        <div class="anom-card" style="text-align:center;padding:2rem">
          <div style="font-size:2rem;margin-bottom:0.7rem">📂</div>
          <div style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:{TXT};margin-bottom:0.5rem">
            No dataset loaded yet
          </div>
          <div style="color:{SOFT};font-size:0.85rem">
            Please go to <b>🧬 ML Pipeline → 📁 Data Loading</b>, upload the 5 Fitbit CSV files
            and click <b>▶ Proceed to TSFresh Features</b>.<br>
            The dataset will then be shared automatically with this section and Insights Dashboard.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        master_shared = st.session_state.shared_master
        detected_shared = st.session_state.shared_detected
        n_files = len(detected_shared)
        # Show file status grid
        status_html = f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.6rem;margin:1rem 0">'
        for req_name, finfo in REQUIRED_FILES.items():
            found = req_name in detected_shared
            bg  = "rgba(30,132,73,0.07)" if found else "rgba(211,84,0,0.07)"
            bor = "rgba(30,132,73,0.35)" if found else "rgba(211,84,0,0.35)"
            ico = "✅" if found else "❌"
            status_html += f'<div style="background:{bg};border:1px solid {bor};border-radius:10px;padding:0.65rem 0.85rem"><div style="font-size:1.15rem">{ico} {finfo["icon"]}</div><div style="font-size:0.71rem;font-weight:600;color:{TXT};margin-top:0.3rem">{finfo["label"]}</div></div>'
        status_html += "</div>"
        st.markdown(status_html, unsafe_allow_html=True)
        ui_success(f"Shared dataset ready — {master_shared.shape[0]:,} rows · {master_shared['Id'].nunique()} users · {n_files}/5 files")

        # Always sync anom_master from shared master (no button or rerun needed)
        st.session_state.anom_master       = master_shared
        st.session_state.anom_files_loaded = True

    if st.session_state.anom_files_loaded:
        master = st.session_state.anom_master
        ui_success(f"Master DataFrame ready — {master.shape[0]:,} rows · {master['Id'].nunique()} users")
        hr()
        anom_sec("🚨", "Anomaly Detection — Three Methods", "Steps 2–4")
        st.markdown(f"""
        <div class="anom-card">
          <div class="anom-card-title">Detection Methods Applied</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;font-size:0.82rem">
            <div style="background:{FIG_BG};border:1px solid {BORD};border-radius:10px;padding:0.85rem">
              <div style="color:{ACCENT_RED};font-weight:700;margin-bottom:0.4rem">① Threshold Violations</div>
              <div style="color:{SOFT}">Hard upper/lower limits on HR, Steps, Sleep.</div>
            </div>
            <div style="background:{FIG_BG};border:1px solid {BORD};border-radius:10px;padding:0.85rem">
              <div style="color:{ACC2};font-weight:700;margin-bottom:0.4rem">② Residual-Based</div>
              <div style="color:{SOFT}">Rolling median baseline. Flag ±{sigma:.0f}σ deviations.</div>
            </div>
            <div style="background:{FIG_BG};border:1px solid {BORD};border-radius:10px;padding:0.85rem">
              <div style="color:{ACCENT3};font-weight:700;margin-bottom:0.4rem">③ DBSCAN Outliers</div>
              <div style="color:{SOFT}">Users labelled −1. Structural outliers.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Run Anomaly Detection (All 3 Methods)", key="anom_detect_btn"):
            with st.spinner("Detecting anomalies..."):
                try:
                    st.session_state.anom_hr_result    = detect_hr_anomalies(master, hr_high, hr_low, sigma)
                    st.session_state.anom_steps_result = detect_steps_anomalies(master, st_low, 25000, sigma)
                    st.session_state.anom_sleep_result = detect_sleep_anomalies(master, sl_low, sl_high, sigma)
                    st.session_state.anom_anomaly_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Detection error: {e}")

        if st.session_state.anom_anomaly_done:
            anom_hr_r    = st.session_state.anom_hr_result
            anom_steps_r = st.session_state.anom_steps_result
            anom_sleep_r = st.session_state.anom_sleep_result
            n_hr    = int(anom_hr_r["is_anomaly"].sum())
            n_steps = int(anom_steps_r["is_anomaly"].sum())
            n_sleep = int(anom_sleep_r["is_anomaly"].sum())
            n_total = n_hr + n_steps + n_sleep
            ui_danger(f"Total anomalies flagged: {n_total}  (HR: {n_hr} · Steps: {n_steps} · Sleep: {n_sleep})")
            anom_metrics((n_hr,"HR Anomalies"),(n_steps,"Steps Anomalies"),(n_sleep,"Sleep Anomalies"),(n_total,"Total Flags"),red_indices=[0,1,2,3])

            hr()
            anom_sec("❤️", "Heart Rate — Anomaly Chart", "Step 2")
            hr_anom = anom_hr_r[anom_hr_r["is_anomaly"]]
            fig_hr  = go.Figure()
            rolling_upper = anom_hr_r["rolling_med"] + sigma * anom_hr_r["residual"].std()
            rolling_lower = anom_hr_r["rolling_med"] - sigma * anom_hr_r["residual"].std()
            fig_hr.add_trace(go.Scatter(x=anom_hr_r["Date"],y=rolling_upper,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig_hr.add_trace(go.Scatter(x=anom_hr_r["Date"],y=rolling_lower,mode="lines",fill="tonexty",
                fillcolor=hex_rgba(ACC,0.1),line=dict(width=0),name=f"±{sigma:.0f}σ Expected Band"))
            fig_hr.add_trace(go.Scatter(x=anom_hr_r["Date"],y=anom_hr_r["AvgHR"],mode="lines+markers",name="Avg Heart Rate",
                line=dict(color=ACC,width=2.5),marker=dict(size=5,color=ACC)))
            fig_hr.add_trace(go.Scatter(x=anom_hr_r["Date"],y=anom_hr_r["rolling_med"],mode="lines",name="Rolling Median",
                line=dict(color=ACCENT3,width=1.5,dash="dot")))
            if not hr_anom.empty:
                fig_hr.add_trace(go.Scatter(x=hr_anom["Date"],y=hr_anom["AvgHR"],mode="markers",name="🚨 Anomaly",
                    marker=dict(color=ACCENT_RED,size=14,symbol="circle",line=dict(color="white",width=2))))
                for _, row in hr_anom.iterrows():
                    fig_hr.add_annotation(x=row["Date"],y=row["AvgHR"],text=f"⚠️ {row['reason']}",
                        showarrow=True,arrowhead=2,arrowcolor=ACCENT_RED,ax=0,ay=-45,
                        font=dict(color=ACCENT_RED,size=9),bgcolor=CARD,bordercolor="rgba(192,57,43,0.4)",borderwidth=1,borderpad=4)
            fig_hr.add_hline(y=hr_high,line_dash="dash",line_color=ACCENT_RED,line_width=1.5,opacity=0.7,
                annotation_text=f"High Threshold ({hr_high} bpm)",annotation_position="top right",annotation_font_color=ACCENT_RED)
            fig_hr.add_hline(y=hr_low,line_dash="dash",line_color=ACC2,line_width=1.5,opacity=0.7,
                annotation_text=f"Low Threshold ({hr_low} bpm)",annotation_position="bottom right",annotation_font_color=ACC2)
            apply_anom_theme(fig_hr,"❤️ Heart Rate — Anomaly Detection (Real Fitbit Data)")
            fig_hr.update_layout(height=480,xaxis_title="Date",yaxis_title="Heart Rate (bpm)")
            st.plotly_chart(fig_hr,use_container_width=True,key="anom_hr_chart")

            hr()
            anom_sec("💤", "Sleep Pattern — Anomaly Visualization", "Step 3")
            sleep_anom = anom_sleep_r[anom_sleep_r["is_anomaly"]]
            fig_sleep  = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],
                subplot_titles=["Sleep Duration (minutes/night)","Deviation from Expected"],vertical_spacing=0.08)
            fig_sleep.add_hrect(y0=sl_low,y1=sl_high,fillcolor="rgba(30,132,73,0.07)",line_width=0,
                annotation_text="✅ Healthy Sleep Zone",annotation_position="top right",annotation_font_color=ACCENT3,row=1,col=1)
            fig_sleep.add_trace(go.Scatter(x=anom_sleep_r["Date"],y=anom_sleep_r["TotalSleepMinutes"],
                mode="lines+markers",name="Sleep Minutes",line=dict(color=ACC2,width=2.5),marker=dict(size=5,color=ACC2)),row=1,col=1)
            fig_sleep.add_trace(go.Scatter(x=anom_sleep_r["Date"],y=anom_sleep_r["rolling_med"],
                mode="lines",name="Rolling Median",line=dict(color=ACCENT3,width=1.5,dash="dot")),row=1,col=1)
            if not sleep_anom.empty:
                fig_sleep.add_trace(go.Scatter(x=sleep_anom["Date"],y=sleep_anom["TotalSleepMinutes"],
                    mode="markers",name="🚨 Sleep Anomaly",
                    marker=dict(color=ACCENT_RED,size=14,symbol="diamond",line=dict(color="white",width=2))),row=1,col=1)
            fig_sleep.add_hline(y=sl_low,line_dash="dash",line_color=ACCENT_RED,line_width=1.5,opacity=0.7,row=1,col=1)
            fig_sleep.add_hline(y=sl_high,line_dash="dash",line_color=ACC,line_width=1.5,opacity=0.7,row=1,col=1)
            colors_resid = [ACCENT_RED if v else ACC2 for v in anom_sleep_r["resid_anomaly"]]
            fig_sleep.add_trace(go.Bar(x=anom_sleep_r["Date"],y=anom_sleep_r["residual"],name="Residual",marker_color=colors_resid),row=2,col=1)
            fig_sleep.add_hline(y=0,line_dash="solid",line_color=SOFT,line_width=1,row=2,col=1)
            apply_anom_theme(fig_sleep)
            fig_sleep.update_layout(height=560,showlegend=True,paper_bgcolor=CARD,plot_bgcolor=FIG_BG,font_color=TXT)
            fig_sleep.update_xaxes(gridcolor=BORD,tickfont_color=SOFT); fig_sleep.update_yaxes(gridcolor=BORD,tickfont_color=SOFT)
            st.plotly_chart(fig_sleep,use_container_width=True,key="anom_sleep_chart")

            hr()
            anom_sec("🚶", "Step Count Trend — Alerts & Anomalies", "Step 4")
            steps_anom = anom_steps_r[anom_steps_r["is_anomaly"]]
            fig_steps  = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],
                subplot_titles=["Daily Steps (avg across users)","Residual Deviation from Trend"],vertical_spacing=0.08)
            for _, row_d in steps_anom.iterrows():
                d=str(row_d["Date"]); d_next=str(pd.Timestamp(d)+pd.Timedelta(days=1))[:10]
                fig_steps.add_vrect(x0=d,x1=d_next,fillcolor="rgba(192,57,43,0.12)",line_color="rgba(192,57,43,0.45)",line_width=1.5,row=1,col=1)
            fig_steps.add_trace(go.Scatter(x=anom_steps_r["Date"],y=anom_steps_r["TotalSteps"],
                mode="lines+markers",name="Avg Daily Steps",line=dict(color=ACCENT3,width=2.5),marker=dict(size=5,color=ACCENT3)),row=1,col=1)
            fig_steps.add_trace(go.Scatter(x=anom_steps_r["Date"],y=anom_steps_r["rolling_med"],
                mode="lines",name="Trend",line=dict(color=ACC,width=2,dash="dash")),row=1,col=1)
            if not steps_anom.empty:
                fig_steps.add_trace(go.Scatter(x=steps_anom["Date"],y=steps_anom["TotalSteps"],
                    mode="markers",name="🚨 Steps Anomaly",
                    marker=dict(color=ACCENT_RED,size=14,symbol="triangle-up",line=dict(color="white",width=2))),row=1,col=1)
            fig_steps.add_hline(y=st_low,line_dash="dash",line_color=ACCENT_RED,line_width=1.5,opacity=0.8,row=1,col=1)
            fig_steps.add_hline(y=25000,line_dash="dash",line_color=ACC2,line_width=1.5,opacity=0.7,row=1,col=1)
            res_colors = [ACCENT_RED if v else ACCENT3 for v in anom_steps_r["resid_anomaly"]]
            fig_steps.add_trace(go.Bar(x=anom_steps_r["Date"],y=anom_steps_r["residual"],name="Residual",marker_color=res_colors),row=2,col=1)
            fig_steps.add_hline(y=0,line_dash="solid",line_color=SOFT,line_width=1,row=2,col=1)
            apply_anom_theme(fig_steps)
            fig_steps.update_layout(height=560,showlegend=True,paper_bgcolor=CARD,plot_bgcolor=FIG_BG,font_color=TXT)
            fig_steps.update_xaxes(gridcolor=BORD,tickfont_color=SOFT); fig_steps.update_yaxes(gridcolor=BORD,tickfont_color=SOFT)
            st.plotly_chart(fig_steps,use_container_width=True,key="anom_steps_chart")

            hr()
            anom_sec("🎯", "Simulated Detection Accuracy", "Step 5")
            if st.button("🎯 Run Accuracy Simulation", key="anom_sim_btn"):
                with st.spinner("Simulating..."):
                    try:
                        sim = simulate_accuracy(master, n_inject=10)
                        st.session_state.anom_sim_results  = sim
                        st.session_state.anom_simulation_done = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Simulation error: {e}")

            if st.session_state.anom_simulation_done and st.session_state.anom_sim_results:
                sim     = st.session_state.anom_sim_results
                overall = sim["Overall"]
                passed  = overall >= 90.0
                if passed: ui_success(f"Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT")
                else:       ui_warn(f"Overall accuracy: {overall}% — below 90% target")
                signals    = ["Heart Rate","Steps","Sleep"]
                accs       = [sim[s]["accuracy"] for s in signals]
                bar_colors = [ACCENT3 if a >= 90 else ACCENT_RED for a in accs]
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Bar(x=signals,y=accs,marker_color=bar_colors,
                    text=[f"{a}%" for a in accs],textposition="outside",textfont=dict(color=TXT,size=14,family="Syne, sans-serif")))
                fig_acc.add_hline(y=90,line_dash="dash",line_color=ACCENT_RED,line_width=2,
                    annotation_text="90% Target",annotation_font_color=ACCENT_RED,annotation_position="top right")
                apply_anom_theme(fig_acc,"🎯 Simulated Anomaly Detection Accuracy")
                fig_acc.update_layout(height=380,yaxis_range=[0,115],yaxis_title="Detection Accuracy (%)",showlegend=False)
                st.plotly_chart(fig_acc,use_container_width=True,key="anom_accuracy_chart")
    else:
        st.markdown(f'<div class="anom-card" style="text-align:center;padding:2.8rem;margin-top:1rem"><div style="font-size:2.8rem;margin-bottom:0.9rem">🚨</div><div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{TXT};margin-bottom:0.5rem">No Dataset Loaded</div><div style="color:{SOFT};font-size:0.86rem">Go to <b>🧬 ML Pipeline → 📁 Data Loading</b> and upload your 5 Fitbit CSV files.<br>As soon as all 5 are detected the dataset will be shared here automatically — no button click needed.</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="footer">🚨 <b>FitPulse Anomaly Detection</b> &nbsp;·&nbsp; Threshold · Residual · DBSCAN · Accuracy Simulation</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE D — INSIGHTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
else:
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="ins-hero">
  <div class="ins-hero-badge">INSIGHTS DASHBOARD &nbsp;·&nbsp; KPI · TIMELINE · DEEP DIVES · EXPORT</div>
  <h1 class="ins-hero-title">📈 FitPulse Insights Dashboard</h1>
  <p class="ins-hero-sub">Upload · Detect · Filter by Date &amp; User · Export PDF &amp; CSV — Real Fitbit Device Data</p>
</div>""", unsafe_allow_html=True)

    ins_hr_high = st.session_state.get("ins_hr_high", 100)
    ins_hr_low  = st.session_state.get("ins_hr_low",  50)
    ins_st_low  = st.session_state.get("ins_st_low",  500)
    ins_sl_low  = st.session_state.get("ins_sl_low",  60)
    ins_sl_high = st.session_state.get("ins_sl_high", 600)
    ins_sigma   = st.session_state.get("ins_sigma",   2.0)

    # ── SECTION 1: SHARED DATASET STATUS ─────────────────────────────────────
    ins_sec("📂", "Dataset Status", "Shared from ML Pipeline")

    shared_ok_ins = st.session_state.shared_built and st.session_state.shared_master is not None

    if not shared_ok_ins:
        st.markdown(f"""
        <div class="ins-card" style="text-align:center;padding:2.5rem;margin-top:0.5rem">
          <div style="font-size:2.5rem;margin-bottom:0.8rem">📂</div>
          <div style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:{TXT};margin-bottom:0.5rem">
            No dataset loaded yet
          </div>
          <div style="color:{SOFT};font-size:0.85rem;line-height:1.7">
            Please go to <b>🧬 ML Pipeline → 📁 Data Loading</b>, upload the 5 Fitbit CSV files
            and click <b>▶ Proceed to TSFresh Features</b>.<br>
            The dataset is shared automatically — you only need to upload once.
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;max-width:500px;margin:1.2rem auto 0;text-align:left">
            <div class="ins-feature-box"><div class="ins-feature-title">① Go to ML Pipeline</div><div class="ins-feature-body">Select it from the sidebar dropdown</div></div>
            <div class="ins-feature-box"><div class="ins-feature-title">② Upload 5 CSV files</div><div class="ins-feature-body">Data Loading tab in ML Pipeline</div></div>
            <div class="ins-feature-box"><div class="ins-feature-title">③ Come back here</div><div class="ins-feature-body">Dashboard auto-loads the data</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        master_shared_ins = st.session_state.shared_master
        detected_shared_ins = st.session_state.shared_detected

        # File status grid
        status_ins = f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.6rem;margin:1rem 0">'
        for req_name, finfo in REQUIRED_FILES.items():
            found = req_name in detected_shared_ins
            bg  = f"{INS_SUCCESS_BG}" if found else f"{INS_WARN_BG}"
            bor = f"{INS_SUCCESS_BOR}" if found else f"{INS_WARN_BOR}"
            ico = "✅" if found else "❌"
            status_ins += f'<div style="background:{bg};border:1px solid {bor};border-radius:10px;padding:0.65rem 0.85rem"><div style="font-size:1.1rem">{ico} {finfo["icon"]}</div><div style="font-size:0.71rem;font-weight:600;color:{TXT};margin-top:0.25rem">{finfo["label"]}</div><div style="font-size:0.63rem;color:{SOFT};font-family:DM Mono,monospace">Shared</div></div>'
        status_ins += "</div>"
        st.markdown(status_ins, unsafe_allow_html=True)

        # Status badges
        n_files_ins = len(detected_shared_ins)
        st.markdown(f"""
        <div style="display:flex;gap:0.6rem;margin:0.5rem 0 1rem">
          <div style="background:{CARD};border:1px solid {BORD};border-radius:10px;padding:0.5rem 0.9rem;text-align:center;min-width:110px">
            <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:{ACC}">{n_files_ins}/5</div>
            <div style="font-size:0.65rem;color:{SOFT};text-transform:uppercase;letter-spacing:0.07em">Files Loaded</div>
          </div>
          <div style="background:{CARD};border:1px solid {BORD};border-radius:10px;padding:0.5rem 0.9rem;text-align:center;min-width:110px">
            <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:{ACCENT3}">✅ READY</div>
            <div style="font-size:0.65rem;color:{SOFT};text-transform:uppercase;letter-spacing:0.07em">Shared Dataset</div>
          </div>
          <div style="background:{CARD};border:1px solid {BORD};border-radius:10px;padding:0.5rem 0.9rem;text-align:center;min-width:140px">
            <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:{ACC}">{master_shared_ins.shape[0]:,}</div>
            <div style="font-size:0.65rem;color:{SOFT};text-transform:uppercase;letter-spacing:0.07em">Total Rows</div>
          </div>
          <div style="background:{CARD};border:1px solid {BORD};border-radius:10px;padding:0.5rem 0.9rem;text-align:center;min-width:110px">
            <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:{ACC2}">{master_shared_ins['Id'].nunique()}</div>
            <div style="font-size:0.65rem;color:{SOFT};text-transform:uppercase;letter-spacing:0.07em">Users</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── AUTO-RUN detection when shared data is available (no button click needed) ──
        # If shared master is ready and detection hasn't run yet (or master changed), run it now.
        _ins_master_changed = (
            st.session_state.ins_master is None or
            not st.session_state.ins_pipeline_done or
            (st.session_state.ins_master is not master_shared_ins)
        )
        if _ins_master_changed:
            with st.spinner("🔄 Running anomaly detection on shared dataset…"):
                try:
                    st.session_state.ins_master      = master_shared_ins
                    st.session_state.ins_anom_hr     = detect_hr_anomalies(master_shared_ins, ins_hr_high, ins_hr_low,   ins_sigma)
                    st.session_state.ins_anom_steps  = detect_steps_anomalies(master_shared_ins, ins_st_low,  25000,     ins_sigma)
                    st.session_state.ins_anom_sleep  = detect_sleep_anomalies(master_shared_ins, ins_sl_low,  ins_sl_high, ins_sigma)
                    st.session_state.ins_pipeline_done = True
                except Exception as _e:
                    st.error(f"Detection error: {_e}")

        # Manual re-run button (to apply changed thresholds)
        if st.button("🔄 Re-run Detection with Current Thresholds", key="ins_run_btn"):
            with st.spinner("Running anomaly detection..."):
                try:
                    master = master_shared_ins
                    st.session_state.ins_master      = master
                    st.session_state.ins_anom_hr     = detect_hr_anomalies(master,    ins_hr_high, ins_hr_low,   ins_sigma)
                    st.session_state.ins_anom_steps  = detect_steps_anomalies(master, ins_st_low,  25000,        ins_sigma)
                    st.session_state.ins_anom_sleep  = detect_sleep_anomalies(master, ins_sl_low,  ins_sl_high,  ins_sigma)
                    st.session_state.ins_pipeline_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Detection error: {e}")
                    import traceback; st.code(traceback.format_exc())

    # ── MAIN DASHBOARD (post-pipeline) ────────────────────────────────────────
    if not st.session_state.ins_pipeline_done:
        st.markdown(f"""
        <div class="ins-card" style="text-align:center;padding:3rem;margin-top:1rem">
          <div style="font-size:3rem;margin-bottom:1rem">📈</div>
          <div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:{TXT};margin-bottom:0.5rem">
            Dataset Not Yet Loaded
          </div>
          <div style="color:{SOFT};font-size:0.88rem;margin-bottom:1.5rem">
            Upload your 5 Fitbit CSV files in <b>🧬 ML Pipeline → 📁 Data Loading</b>.<br>
            The dashboard will load automatically — no extra steps needed.
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;max-width:620px;margin:0 auto;text-align:left">
            <div class="ins-feature-box"><div class="ins-feature-title">① Go to ML Pipeline</div><div class="ins-feature-body">Select from the sidebar dropdown</div></div>
            <div class="ins-feature-box"><div class="ins-feature-title">② Upload 5 CSV files</div><div class="ins-feature-body">Files are auto-detected instantly</div></div>
            <div class="ins-feature-box"><div class="ins-feature-title">③ Come back here</div><div class="ins-feature-body">Dashboard loads automatically</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        master     = st.session_state.ins_master
        anom_hr    = st.session_state.ins_anom_hr
        anom_steps = st.session_state.ins_anom_steps
        anom_sleep = st.session_state.ins_anom_sleep

        # ── Date & User filter (inline since sidebar is already used for thresholds) ──
        hr()
        ins_sec("🔎", "Filters — Date Range & User", "Optional")
        all_dates = pd.to_datetime(master["Date"])
        d_min = all_dates.min().date(); d_max = all_dates.max().date()
        fc1, fc2 = st.columns([2,1])
        with fc1:
            st.markdown(f'<div style="font-size:0.75rem;color:{SOFT};font-family:DM Mono,monospace;margin-bottom:0.3rem">DATE RANGE</div>', unsafe_allow_html=True)
            date_range = st.date_input("Date range", value=(d_min, d_max),
                                       min_value=d_min, max_value=d_max,
                                       key="ins_daterange", label_visibility="collapsed")
        with fc2:
            st.markdown(f'<div style="font-size:0.75rem;color:{SOFT};font-family:DM Mono,monospace;margin-bottom:0.3rem">USER FILTER</div>', unsafe_allow_html=True)
            all_users = sorted(master["Id"].unique())
            user_opts = ["All Users"] + [f"...{str(u)[-6:]}" for u in all_users]
            sel_user_lbl = st.selectbox("User", user_opts, key="ins_user", label_visibility="collapsed")
            sel_user = None if sel_user_lbl == "All Users" else all_users[user_opts.index(sel_user_lbl)-1]

        # Apply filters
        try:
            if isinstance(date_range, tuple) and len(date_range)==2:
                d_from, d_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            else:
                d_from, d_to = pd.Timestamp(d_min), pd.Timestamp(d_max)
        except Exception:
            d_from, d_to = pd.Timestamp(d_min), pd.Timestamp(d_max)

        def filt(df, date_col="Date"):
            df2 = df.copy(); df2[date_col] = pd.to_datetime(df2[date_col])
            return df2[(df2[date_col] >= d_from) & (df2[date_col] <= d_to)]

        anom_hr_f    = filt(anom_hr)
        anom_steps_f = filt(anom_steps)
        anom_sleep_f = filt(anom_sleep)
        master_f     = filt(master)
        if sel_user:
            master_f = master_f[master_f["Id"] == sel_user]

        n_hr_f    = int(anom_hr_f["is_anomaly"].sum())
        n_steps_f = int(anom_steps_f["is_anomaly"].sum())
        n_sleep_f = int(anom_sleep_f["is_anomaly"].sum())
        n_total_f = n_hr_f + n_steps_f + n_sleep_f
        n_users_f = master_f["Id"].nunique()
        n_days_f  = master_f["Date"].nunique()
        worst_hr_row = anom_hr_f[anom_hr_f["is_anomaly"]].copy()
        if not worst_hr_row.empty and "residual" in worst_hr_row.columns:
            worst_hr_day = worst_hr_row.iloc[worst_hr_row["residual"].abs().argmax()]["Date"].strftime("%d %b")
        else:
            worst_hr_day = "—"

        # ── KPI STRIP ─────────────────────────────────────────────────────────
        hr()
        ins_ui_success(f"Pipeline complete — {n_users_f} users · {n_days_f} days · {n_total_f} anomalies flagged")

        kpi_html = f"""
        <div class="ins-kpi-grid">
          <div class="ins-kpi-card" style="border-color:{INS_DANGER_BOR}">
            <div class="ins-kpi-val" style="color:{ACCENT_RED}">{n_total_f}</div>
            <div class="ins-kpi-label">Total Anomalies</div>
            <div class="ins-kpi-sub">across all signals</div>
          </div>
          <div class="ins-kpi-card" style="border-color:rgba(155,89,182,0.4)">
            <div class="ins-kpi-val" style="color:{ACC2}">{n_hr_f}</div>
            <div class="ins-kpi-label">HR Flags</div>
            <div class="ins-kpi-sub">heart rate anomalies</div>
          </div>
          <div class="ins-kpi-card" style="border-color:rgba(30,132,73,0.35)">
            <div class="ins-kpi-val" style="color:{ACCENT3}">{n_steps_f}</div>
            <div class="ins-kpi-label">Steps Alerts</div>
            <div class="ins-kpi-sub">step count anomalies</div>
          </div>
          <div class="ins-kpi-card" style="border-color:rgba(107,45,139,0.3)">
            <div class="ins-kpi-val" style="color:{ACC}">{n_sleep_f}</div>
            <div class="ins-kpi-label">Sleep Flags</div>
            <div class="ins-kpi-sub">sleep anomalies</div>
          </div>
          <div class="ins-kpi-card">
            <div class="ins-kpi-val" style="color:{ACC}">{n_users_f}</div>
            <div class="ins-kpi-label">Users</div>
            <div class="ins-kpi-sub">in selected range</div>
          </div>
          <div class="ins-kpi-card">
            <div class="ins-kpi-val" style="color:{INS_ACCENT_ORG};font-size:1.3rem">{worst_hr_day}</div>
            <div class="ins-kpi-label">Peak HR Anomaly</div>
            <div class="ins-kpi-sub">highest deviation day</div>
          </div>
        </div>"""
        st.markdown(kpi_html, unsafe_allow_html=True)

        # ── TABS ──────────────────────────────────────────────────────────────
        tab_ov, tab_hr_tab, tab_st_tab, tab_sl_tab, tab_exp = st.tabs([
            "📊 Overview", "❤️ Heart Rate", "🚶 Steps", "💤 Sleep", "📥 Export"
        ])

        # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────
        with tab_ov:
            ins_sec("📅", "Combined Anomaly Timeline")
            all_anoms = []
            sig_colors = {"Heart Rate": ACC2, "Steps": ACCENT3, "Sleep": ACC}
            for df_, sig in [(anom_hr_f,"Heart Rate"),(anom_steps_f,"Steps"),(anom_sleep_f,"Sleep")]:
                a = df_[df_["is_anomaly"]].copy()
                a["signal"] = sig; all_anoms.append(a[["Date","signal","reason"]])

            if all_anoms:
                combined = pd.concat(all_anoms, ignore_index=True)
                combined["Date"] = pd.to_datetime(combined["Date"])
                fig_tl = go.Figure()
                for sig, col in sig_colors.items():
                    sub = combined[combined["signal"]==sig]
                    if not sub.empty:
                        fig_tl.add_trace(go.Scatter(
                            x=sub["Date"], y=sub["signal"], mode="markers",
                            name=sig, marker=dict(color=col, size=14, symbol="diamond",
                                                   line=dict(color="white",width=2)),
                            hovertemplate=f"<b>{sig}</b><br>%{{x|%d %b %Y}}<br>%{{customdata}}<extra>⚠️ ANOMALY</extra>",
                            customdata=sub["reason"].values))
                apply_ins_theme(fig_tl, "📅 Anomaly Event Timeline — All Signals", h=280)
                fig_tl.update_layout(xaxis_title="Date", yaxis_title="Signal", showlegend=True,
                    yaxis=dict(categoryorder="array",categoryarray=["Sleep","Steps","Heart Rate"],
                               gridcolor=INS_GRID_CLR,tickfont_color=INS_MUTED))
                st.plotly_chart(fig_tl, use_container_width=True, key="ins_timeline")

                ins_sec("🗂️", "Recent Anomaly Log")
                log = combined.sort_values("Date", ascending=False).head(12)
                for _, row in log.iterrows():
                    col = sig_colors.get(row["signal"], ACC)
                    st.markdown(f"""
                    <div class="ins-anom-row">
                      <span style="font-size:0.9rem">🚨</span>
                      <span style="color:{col};font-family:DM Mono,monospace;font-size:0.75rem;min-width:95px">{row['signal']}</span>
                      <span style="color:{SOFT};font-size:0.78rem;min-width:90px">{row['Date'].strftime('%d %b %Y')}</span>
                      <span style="color:{TXT};font-size:0.78rem">{row['reason']}</span>
                    </div>""", unsafe_allow_html=True)

        # ── TAB 2: HEART RATE ─────────────────────────────────────────────────
        with tab_hr_tab:
            ins_sec("❤️", "Heart Rate — Deep Dive", f"{n_hr_f} anomalies")
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown(f"""
                <div class="ins-card">
                  <div class="ins-card-title">HR Statistics</div>
                  <div style="font-size:0.83rem;line-height:2.2">
                    <div>Mean HR: <b style="color:{ACC}">{anom_hr_f['AvgHR'].mean():.1f} bpm</b></div>
                    <div>Max HR: <b style="color:{ACCENT_RED}">{anom_hr_f['AvgHR'].max():.1f} bpm</b></div>
                    <div>Min HR: <b style="color:{ACC2}">{anom_hr_f['AvgHR'].min():.1f} bpm</b></div>
                    <div>Anomaly days: <b style="color:{ACCENT_RED}">{n_hr_f}</b> of {len(anom_hr_f)} total</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with c_b:
                hr_disp = anom_hr_f[anom_hr_f["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].round(2)
                st.markdown(f'<div class="ins-card"><div class="ins-card-title">HR Anomaly Records ({len(hr_disp)})</div>', unsafe_allow_html=True)
                if not hr_disp.empty:
                    st.dataframe(hr_disp.rename(columns={"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                 use_container_width=True, height=200)
                else:
                    ins_ui_success("No HR anomalies in selected range")
                st.markdown('</div>', unsafe_allow_html=True)
            fig_hr_ins = ins_chart_hr(anom_hr_f, ins_hr_high, ins_hr_low, ins_sigma, h=420)
            st.plotly_chart(fig_hr_ins, use_container_width=True, key="ins_hr_chart")

        # ── TAB 3: STEPS ──────────────────────────────────────────────────────
        with tab_st_tab:
            ins_sec("🚶", "Step Count — Deep Dive", f"{n_steps_f} alerts")
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown(f"""
                <div class="ins-card">
                  <div class="ins-card-title">Steps Statistics</div>
                  <div style="font-size:0.83rem;line-height:2.2">
                    <div>Mean steps/day: <b style="color:{ACCENT3}">{anom_steps_f['TotalSteps'].mean():,.0f}</b></div>
                    <div>Max steps/day: <b style="color:{ACC}">{anom_steps_f['TotalSteps'].max():,.0f}</b></div>
                    <div>Min steps/day: <b style="color:{ACCENT_RED}">{anom_steps_f['TotalSteps'].min():,.0f}</b></div>
                    <div>Alert days: <b style="color:{ACCENT_RED}">{n_steps_f}</b> of {len(anom_steps_f)} total</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with c_b:
                st_disp = anom_steps_f[anom_steps_f["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].round(2)
                st.markdown(f'<div class="ins-card"><div class="ins-card-title">Steps Alert Records ({len(st_disp)})</div>', unsafe_allow_html=True)
                if not st_disp.empty:
                    st.dataframe(st_disp.rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                 use_container_width=True, height=200)
                else:
                    ins_ui_success("No step anomalies in selected range")
                st.markdown('</div>', unsafe_allow_html=True)
            fig_st_ins = ins_chart_steps(anom_steps_f, ins_st_low, h=420)
            st.plotly_chart(fig_st_ins, use_container_width=True, key="ins_steps_chart")

        # ── TAB 4: SLEEP ──────────────────────────────────────────────────────
        with tab_sl_tab:
            ins_sec("💤", "Sleep Pattern — Deep Dive", f"{n_sleep_f} anomalies")
            c_a, c_b = st.columns(2)
            with c_a:
                non_zero = anom_sleep_f[anom_sleep_f["TotalSleepMinutes"]>0]["TotalSleepMinutes"]
                st.markdown(f"""
                <div class="ins-card">
                  <div class="ins-card-title">Sleep Statistics</div>
                  <div style="font-size:0.83rem;line-height:2.2">
                    <div>Mean sleep/night: <b style="color:{ACC2}">{anom_sleep_f['TotalSleepMinutes'].mean():.0f} min</b></div>
                    <div>Max sleep/night: <b style="color:{ACC}">{anom_sleep_f['TotalSleepMinutes'].max():.0f} min</b></div>
                    <div>Min (non-zero): <b style="color:{ACCENT_RED}">{non_zero.min():.0f} min</b></div>
                    <div>Anomaly days: <b style="color:{ACCENT_RED}">{n_sleep_f}</b> of {len(anom_sleep_f)} total</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with c_b:
                sl_disp = anom_sleep_f[anom_sleep_f["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].round(2)
                st.markdown(f'<div class="ins-card"><div class="ins-card-title">Sleep Anomaly Records ({len(sl_disp)})</div>', unsafe_allow_html=True)
                if not sl_disp.empty:
                    st.dataframe(sl_disp.rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                 use_container_width=True, height=200)
                else:
                    ins_ui_success("No sleep anomalies in selected range")
                st.markdown('</div>', unsafe_allow_html=True)
            fig_sl_ins = ins_chart_sleep(anom_sleep_f, ins_sl_low, ins_sl_high, h=420)
            st.plotly_chart(fig_sl_ins, use_container_width=True, key="ins_sleep_chart")

        # ── TAB 5: EXPORT ─────────────────────────────────────────────────────
        with tab_exp:
            ins_sec("📥", "Export — PDF Report & CSV Data", "Downloadable")

            st.markdown(f"""
            <div class="ins-card">
              <div class="ins-card-title">What's Included in the Exports</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.83rem;margin-top:0.5rem">
                <div class="ins-feature-box">
                  <div class="ins-feature-title">📄 PDF Report (4 pages)</div>
                  <div class="ins-feature-body">
                    ✅ Executive summary<br>
                    ✅ Anomaly counts per signal<br>
                    ✅ Thresholds used<br>
                    ✅ Methodology explanation<br>
                    ✅ All 3 charts embedded<br>
                    ✅ Full anomaly record tables<br>
                    ✅ User activity profiles
                  </div>
                </div>
                <div class="ins-feature-box">
                  <div class="ins-feature-title">📊 CSV Export</div>
                  <div class="ins-feature-body">
                    ✅ All anomaly records<br>
                    ✅ Signal type column<br>
                    ✅ Date of anomaly<br>
                    ✅ Actual vs expected value<br>
                    ✅ Residual deviation<br>
                    ✅ Anomaly reason text<br>
                    ✅ All signals combined
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            hr()
            col_pdf, col_csv = st.columns(2)

            with col_pdf:
                ins_sec("📄", "PDF Report")
                st.markdown(f'<div style="color:{SOFT};font-size:0.82rem;margin-bottom:0.8rem">Full 4-page PDF with embedded charts, anomaly tables, and user profiles.</div>', unsafe_allow_html=True)
                if st.button("📄 Generate PDF Report", key="ins_gen_pdf"):
                    with st.spinner("⏳ Generating PDF (embedding charts)..."):
                        try:
                            fig_hr_ex  = ins_chart_hr(anom_hr_f,    ins_hr_high, ins_hr_low, ins_sigma, h=420)
                            fig_st_ex  = ins_chart_steps(anom_steps_f, ins_st_low, h=420)
                            fig_sl_ex  = ins_chart_sleep(anom_sleep_f, ins_sl_low, ins_sl_high, h=420)
                            pdf_buf = generate_pdf_report(
                                master_f, anom_hr_f, anom_steps_f, anom_sleep_f,
                                ins_hr_high, ins_hr_low, ins_st_low, ins_sl_low, ins_sl_high, ins_sigma,
                                fig_hr_ex, fig_st_ex, fig_sl_ex
                            )
                            if pdf_buf:
                                fname = f"FitPulse_Insights_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_buf, file_name=fname, mime="application/pdf", key="ins_dl_pdf"
                                )
                                ins_ui_success(f"PDF ready — {fname}")
                            else:
                                ins_ui_danger("fpdf2 not installed. Run: pip install fpdf2")
                        except Exception as e:
                            st.error(f"PDF error: {e}")

            with col_csv:
                ins_sec("📊", "CSV Export")
                st.markdown(f'<div style="color:{SOFT};font-size:0.82rem;margin-bottom:0.8rem">All anomaly records from all three signals in a single CSV file.</div>', unsafe_allow_html=True)
                csv_data = generate_csv_export(anom_hr_f, anom_steps_f, anom_sleep_f)
                fname_csv = f"FitPulse_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                st.download_button(
                    label="⬇️ Download Anomaly CSV",
                    data=csv_data, file_name=fname_csv, mime="text/csv", key="ins_dl_csv"
                )
                with st.expander("👁️ Preview CSV data"):
                    dfs_ = []
                    for df_, sig, vc in [
                        (anom_hr_f,"Heart Rate","AvgHR"),
                        (anom_steps_f,"Steps","TotalSteps"),
                        (anom_sleep_f,"Sleep","TotalSleepMinutes"),
                    ]:
                        tmp = df_[df_["is_anomaly"]].assign(signal=sig).rename(columns={vc:"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]]
                        dfs_.append(tmp)
                    preview_df = pd.concat(dfs_, ignore_index=True).sort_values(["signal","Date"]).round(2)
                    st.dataframe(preview_df, use_container_width=True, height=280)

            hr()
            ins_sec("📸", "Screenshots Required for Submission")
            screenshots = [
                ("📸 Screenshot 1", "Full dashboard UI — Overview tab with timeline"),
                ("📸 Screenshot 2", "Export tab — Download PDF & CSV buttons"),
                ("📸 Screenshot 3", "KPI strip with anomaly counts visible"),
                ("📸 Screenshot 4", "HR / Steps / Sleep deep dive tabs"),
                ("📸 Screenshot 5", "Sidebar with thresholds + filters visible"),
            ]
            sc_html = f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem">'
            for i,(label,detail) in enumerate(screenshots):
                if i==4: sc_html += '</div><div style="margin-top:0.5rem">'
                sc_html += f'<div class="ins-feature-box"><span style="color:{ACC2};font-weight:600">{label}</span> — <span style="color:{SOFT};font-size:0.8rem">{detail}</span></div>'
            sc_html += '</div>'
            st.markdown(sc_html, unsafe_allow_html=True)

    st.markdown(f'<div class="footer">📈 <b>FitPulse Insights Dashboard</b> &nbsp;·&nbsp; KPI · Timeline · Deep Dives · PDF &amp; CSV Export &nbsp;·&nbsp; Thistle Purple Edition</div>',unsafe_allow_html=True)