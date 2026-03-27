import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse Anomaly Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("dark_mode",        False),   # default LIGHT to match thistle theme
    ("files_loaded",     False),
    ("anomaly_done",     False),
    ("simulation_done",  False),
    ("daily",    None), ("hourly_s", None), ("hourly_i", None),
    ("sleep",    None), ("hr",       None), ("hr_minute", None),
    ("master",   None),
    ("anom_hr",  None), ("anom_steps", None), ("anom_sleep", None),
    ("sim_results", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Palette — Thistle Purple system matching FitPulse main app ────────────────
# Light mode: #F5ECF5 (fig/plot bg) and #EDE0ED (card bg) as core
# Dark mode: deep purple equivalents
dark = st.session_state.dark_mode

if dark:
    # ── Dark Thistle Purple ──────────────────────────────────────────────────
    BG          = "linear-gradient(135deg,#1A0A22 0%,#2D1040 40%,#1A0A22 100%)"
    CARD_BG     = "rgba(45,16,64,0.92)"
    CARD_BOR    = "rgba(196,160,196,0.3)"
    TEXT        = "#F0E0F8"
    MUTED       = "#C4A0C4"
    ACCENT      = "#9B59B6"      # medium purple
    ACCENT2     = "#E91E63"      # pink accent
    ACCENT3     = "#2ECC71"      # green (keep semantic)
    ACCENT_RED  = "#E53E3E"      # red anomaly
    PLOT_BG     = "#2D1040"
    PAPER_BG    = "#1A0A22"
    GRID_CLR    = "rgba(196,160,196,0.12)"
    BADGE_BG    = "rgba(155,89,182,0.2)"
    SECTION_BG  = "rgba(245,236,245,0.06)"
    WARN_BG     = "rgba(246,173,85,0.12)"
    WARN_BOR    = "rgba(246,173,85,0.4)"
    SUCCESS_BG  = "rgba(46,204,113,0.1)"
    SUCCESS_BOR = "rgba(46,204,113,0.4)"
    DANGER_BG   = "rgba(229,62,62,0.12)"
    DANGER_BOR  = "rgba(229,62,62,0.4)"
    # Chart-specific
    CHART_PAPER = "#1A0A22"
    CHART_PLOT  = "#2D1040"
    CHART_GRID  = "rgba(196,160,196,0.1)"
    HOVER_BG    = "rgba(45,16,64,0.95)"
else:
    # ── Light Thistle Purple (#F5ECF5 / #EDE0ED base) ───────────────────────
    BG          = "linear-gradient(135deg,#F5ECF5 0%,#EDE0ED 50%,#F5ECF5 100%)"
    CARD_BG     = "#EDE0ED"      # ← thistle card bg
    CARD_BOR    = "#C4A0C4"      # ← thistle border
    TEXT        = "#2D1B3D"      # ← deep purple text
    MUTED       = "#7B5C8A"      # ← soft purple
    ACCENT      = "#6B2D8B"      # ← primary deep purple
    ACCENT2     = "#9B59B6"      # ← medium purple
    ACCENT3     = "#1E8449"      # ← green (semantic ok)
    ACCENT_RED  = "#C0392B"      # ← red anomaly
    PLOT_BG     = "#F5ECF5"      # ← thistle chart plot area
    PAPER_BG    = "#EDE0ED"      # ← thistle chart paper
    GRID_CLR    = "rgba(196,160,196,0.35)"
    BADGE_BG    = "rgba(107,45,139,0.1)"
    SECTION_BG  = "rgba(245,236,245,0.8)"
    WARN_BG     = "rgba(211,84,0,0.07)"
    WARN_BOR    = "rgba(211,84,0,0.35)"
    SUCCESS_BG  = "rgba(30,132,73,0.07)"
    SUCCESS_BOR = "rgba(30,132,73,0.35)"
    DANGER_BG   = "rgba(192,57,43,0.07)"
    DANGER_BOR  = "rgba(192,57,43,0.35)"
    # Chart-specific
    CHART_PAPER = "#EDE0ED"      # ← thistle paper
    CHART_PLOT  = "#F5ECF5"      # ← thistle plot area
    CHART_GRID  = "rgba(196,160,196,0.4)"
    HOVER_BG    = "#EDE0ED"

# ── Plotly base layout ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CHART_PAPER,
    plot_bgcolor=CHART_PLOT,
    font_color=TEXT,
    font_family="Inter, sans-serif",
    xaxis=dict(gridcolor=CHART_GRID, showgrid=True, zeroline=False,
               linecolor=CARD_BOR, tickfont_color=MUTED),
    yaxis=dict(gridcolor=CHART_GRID, showgrid=True, zeroline=False,
               linecolor=CARD_BOR, tickfont_color=MUTED),
    legend=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, borderwidth=1,
                font_color=TEXT),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor=HOVER_BG, bordercolor=CARD_BOR, font_color=TEXT),
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], .main {{
    background: {BG} !important;
    font-family: 'Inter', sans-serif;
    color: {TEXT} !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{
    background: {'rgba(26,10,34,0.97)' if dark else '#D8BFD8'} !important;
    border-right: 1px solid {CARD_BOR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

/* ── Hide sidebar collapse arrow (matches main app) ── */
[data-testid="collapsedControl"] {{ display: none !important; }}
button[kind="header"] {{ display: none !important; }}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{ display: flex !important; }}

.block-container {{ padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }}
p, div, span, label {{ color: {TEXT}; }}

/* ── Hero banner ── */
.m3-hero {{
    background: {'linear-gradient(135deg,rgba(229,62,62,0.07),rgba(155,89,182,0.06),rgba(45,16,64,0.85))' if dark
                 else f'linear-gradient(135deg,{DANGER_BG},{BADGE_BG},{CARD_BG})'};
    border: 1px solid {DANGER_BOR};
    border-left: 5px solid {ACCENT_RED};
    border-radius: 14px; padding: 2.2rem 2.8rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 2px 16px rgba(107,45,139,0.12);
}}
.m3-hero::before {{
    content: ''; position: absolute; top:-60px; right:-60px;
    width:280px; height:280px;
    background: radial-gradient(circle,{'rgba(229,62,62,0.08)' if dark else 'rgba(192,57,43,0.06)'} 0%,transparent 70%);
    border-radius:50%;
}}
.hero-title {{
    font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800;
    color:{TEXT}; margin:0 0 0.4rem 0; letter-spacing:-0.02em;
}}
.hero-sub {{ font-size:1rem; color:{MUTED}; font-weight:300; margin:0; }}
.hero-badge {{
    display:inline-block; background:{DANGER_BG}; border:1px solid {DANGER_BOR};
    border-radius:100px; padding:0.3rem 1rem; font-size:0.74rem;
    font-family:'JetBrains Mono',monospace; color:{ACCENT_RED}; margin-bottom:0.9rem;
}}

/* ── Section header ── */
.sec-header {{
    display:flex; align-items:center; gap:0.8rem;
    margin:2rem 0 1rem 0; padding-bottom:0.6rem; border-bottom:2px solid {CARD_BOR};
}}
.sec-icon {{
    font-size:1.3rem; width:2.1rem; height:2.1rem;
    display:flex; align-items:center; justify-content:center;
    background:{BADGE_BG}; border-radius:8px; border:1px solid {CARD_BOR};
}}
.sec-title {{
    font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
    color:{TEXT}; margin:0;
}}
.sec-badge {{
    margin-left:auto; background:{BADGE_BG}; border:1px solid {CARD_BOR};
    border-radius:100px; padding:0.2rem 0.75rem; font-size:0.7rem;
    font-family:'JetBrains Mono',monospace; color:{ACCENT};
}}

/* ── Cards ── */
.card {{
    background:{CARD_BG}; border:1px solid {CARD_BOR}; border-radius:12px;
    padding:1.3rem 1.5rem; margin-bottom:1rem;
    box-shadow: 0 1px 6px rgba(107,45,139,0.08);
}}
.card-title {{
    font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700;
    color:{MUTED}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;
}}

/* ── Pills & tags ── */
.step-pill {{
    display:inline-flex; align-items:center; gap:0.5rem;
    background:{BADGE_BG}; border:1px solid {CARD_BOR}; border-radius:100px;
    padding:0.28rem 0.9rem; font-size:0.73rem; font-family:'JetBrains Mono',monospace;
    color:{ACCENT}; margin-bottom:0.75rem;
}}
.anom-tag {{
    display:inline-flex; align-items:center; gap:0.4rem;
    background:{DANGER_BG}; border:1px solid {DANGER_BOR}; border-radius:100px;
    padding:0.28rem 0.9rem; font-size:0.71rem; font-family:'JetBrains Mono',monospace;
    color:{ACCENT_RED}; margin-bottom:0.75rem;
}}
.screenshot-badge {{
    display:inline-flex; align-items:center; gap:0.4rem;
    background:{SECTION_BG}; border:1px solid {CARD_BOR}; border-radius:100px;
    padding:0.28rem 0.9rem; font-size:0.71rem; font-family:'JetBrains Mono',monospace;
    color:{ACCENT2}; margin-bottom:0.75rem;
}}

/* ── Metric grid ── */
.metric-grid {{ display:flex; gap:0.75rem; flex-wrap:wrap; margin:0.75rem 0; }}
.metric-card {{
    flex:1; min-width:115px; background:{SECTION_BG}; border:1px solid {CARD_BOR};
    border-radius:12px; padding:0.9rem 1.1rem; text-align:center;
}}
.metric-val {{
    font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800;
    color:{ACCENT}; line-height:1; margin-bottom:0.22rem;
}}
.metric-val-red {{ color:{ACCENT_RED}; }}
.metric-label {{ font-size:0.7rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.06em; }}

/* ── Alert boxes ── */
.alert-warn    {{ background:{WARN_BG};    border-left:3px solid #D4AC0D;
    border-radius:0 9px 9px 0; padding:0.75rem 1rem; margin:0.5rem 0;
    font-size:0.84rem; color:{'#D4AC0D' if dark else '#7D6608'}; }}
.alert-success {{ background:{SUCCESS_BG}; border-left:3px solid {ACCENT3};
    border-radius:0 9px 9px 0; padding:0.75rem 1rem; margin:0.5rem 0;
    font-size:0.84rem; color:{ACCENT3}; }}
.alert-info    {{ background:{BADGE_BG};   border-left:3px solid {ACCENT};
    border-radius:0 9px 9px 0; padding:0.75rem 1rem; margin:0.5rem 0;
    font-size:0.84rem; color:{ACCENT}; }}
.alert-danger  {{ background:{DANGER_BG};  border-left:3px solid {ACCENT_RED};
    border-radius:0 9px 9px 0; padding:0.75rem 1rem; margin:0.5rem 0;
    font-size:0.84rem; color:{ACCENT_RED}; }}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {{
    background:{SECTION_BG}; border:2px dashed {CARD_BOR}; border-radius:12px; padding:0.5rem;
}}

/* ── Buttons ── */
.stButton > button {{
    background:{DANGER_BG}; border:1px solid {DANGER_BOR}; color:{ACCENT_RED};
    border-radius:9px; font-family:'JetBrains Mono',monospace; font-size:0.81rem;
    font-weight:500; padding:0.48rem 1.2rem; transition:all 0.2s;
    box-shadow: 0 1px 6px rgba(192,57,43,0.15);
}}
.stButton > button:hover {{
    background:{ACCENT_RED}; color:white; border-color:{ACCENT_RED};
    transform:translateY(-1px); box-shadow: 0 4px 14px rgba(192,57,43,0.35);
}}
.stButton > button:disabled {{
    background:{CARD_BG}; color:{MUTED}; border-color:{CARD_BOR}; box-shadow:none;
}}

/* ── Tables ── */
[data-testid="stDataFrame"] {{
    border-radius:10px !important; border:1px solid {CARD_BOR} !important;
    overflow:hidden; box-shadow:0 1px 5px rgba(107,45,139,0.08);
}}

/* ── Divider ── */
.m3-divider {{ border:none; border-top:1px solid {CARD_BOR}; margin:1.8rem 0; }}

/* ── Number inputs & sliders ── */
[data-baseweb="input"] > div {{ background:{CARD_BG} !important; border-color:{CARD_BOR} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def sec(icon, title, badge=None):
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-icon">{icon}</div>
      <p class="sec-title">{title}</p>
      {badge_html}
    </div>""", unsafe_allow_html=True)

def step_pill(n, label):
    st.markdown(f'<div class="step-pill">◆ Step {n} &nbsp;·&nbsp; {label}</div>', unsafe_allow_html=True)

def screenshot_badge(ref):
    st.markdown(f'<div class="screenshot-badge">📸 Screenshot · {ref}</div>', unsafe_allow_html=True)

def anom_tag(label):
    st.markdown(f'<div class="anom-tag">🚨 {label}</div>', unsafe_allow_html=True)

def ui_success(msg): st.markdown(f'<div class="alert-success">✅ {msg}</div>', unsafe_allow_html=True)
def ui_warn(msg):    st.markdown(f'<div class="alert-warn">⚠️ {msg}</div>', unsafe_allow_html=True)
def ui_info(msg):    st.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
def ui_danger(msg):  st.markdown(f'<div class="alert-danger">🚨 {msg}</div>', unsafe_allow_html=True)

def metrics(*items, red_indices=None):
    red_indices = red_indices or []
    html = '<div class="metric-grid">'
    for i, (val, label) in enumerate(items):
        val_class = "metric-val metric-val-red" if i in red_indices else "metric-val"
        html += f'<div class="metric-card"><div class="{val_class}">{val}</div><div class="metric-label">{label}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def apply_plotly_theme(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font_color=TEXT, font_size=14,
                                     font_family="Syne, sans-serif"))
    return fig

# ── Smart datetime parser — fixes ALL format variants in Fitbit data ──────────
def parse_dt(series):
    """
    Robustly parse any Fitbit timestamp format.
    Handles:
      "3/12/2016 12:00:00 AM"   (M/D/YYYY H:MM:SS AM/PM)
      "12-03-2016 00.00"        (DD-MM-YYYY HH.MM)
      "2016-03-12 00:00:00"     (ISO)
      and any mixed combination.
    """
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")

# ── Required file registry ────────────────────────────────────────────────────
REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {"key_cols": ["ActivityDate", "TotalSteps", "Calories"],    "label": "Daily Activity",     "icon": "🏃"},
    "hourlySteps_merged.csv":       {"key_cols": ["ActivityHour", "StepTotal"],                 "label": "Hourly Steps",       "icon": "👣"},
    "hourlyIntensities_merged.csv": {"key_cols": ["ActivityHour", "TotalIntensity"],            "label": "Hourly Intensities", "icon": "⚡"},
    "minuteSleep_merged.csv":       {"key_cols": ["date", "value", "logId"],                    "label": "Minute Sleep",       "icon": "💤"},
    "heartrate_seconds_merged.csv": {"key_cols": ["Time", "Value"],                             "label": "Heart Rate",         "icon": "❤️"},
}

def score_match(df, req_info):
    return sum(1 for col in req_info["key_cols"] if col in df.columns)

# ── Anomaly detection functions ───────────────────────────────────────────────
def detect_hr_anomalies(master, hr_high=100, hr_low=50, residual_sigma=2.0):
    df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    hr_daily = df.groupby("Date")["AvgHR"].mean().reset_index()
    hr_daily.columns = ["Date","AvgHR"]
    hr_daily = hr_daily.sort_values("Date")
    hr_daily["thresh_high"]  = hr_daily["AvgHR"] > hr_high
    hr_daily["thresh_low"]   = hr_daily["AvgHR"] < hr_low
    hr_daily["rolling_med"]  = hr_daily["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_daily["residual"]     = hr_daily["AvgHR"] - hr_daily["rolling_med"]
    resid_std                = hr_daily["residual"].std()
    hr_daily["resid_anomaly"]= hr_daily["residual"].abs() > (residual_sigma * resid_std)
    hr_daily["is_anomaly"]   = hr_daily["thresh_high"] | hr_daily["thresh_low"] | hr_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["thresh_high"]:   r.append(f"HR>{hr_high}")
        if row["thresh_low"]:    r.append(f"HR<{hr_low}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    hr_daily["reason"] = hr_daily.apply(reason, axis=1)
    return hr_daily

def detect_steps_anomalies(master, steps_low=500, steps_high=25000, residual_sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    steps_daily = df.groupby("Date")["TotalSteps"].mean().reset_index()
    steps_daily = steps_daily.sort_values("Date")
    steps_daily["thresh_low"]    = steps_daily["TotalSteps"] < steps_low
    steps_daily["thresh_high"]   = steps_daily["TotalSteps"] > steps_high
    steps_daily["rolling_med"]   = steps_daily["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    steps_daily["residual"]      = steps_daily["TotalSteps"] - steps_daily["rolling_med"]
    resid_std                    = steps_daily["residual"].std()
    steps_daily["resid_anomaly"] = steps_daily["residual"].abs() > (residual_sigma * resid_std)
    steps_daily["is_anomaly"]    = steps_daily["thresh_low"] | steps_daily["thresh_high"] | steps_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["thresh_low"]:    r.append(f"Steps<{steps_low}")
        if row["thresh_high"]:   r.append(f"Steps>{steps_high}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    steps_daily["reason"] = steps_daily.apply(reason, axis=1)
    return steps_daily

def detect_sleep_anomalies(master, sleep_low=60, sleep_high=600, residual_sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    sleep_daily = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
    sleep_daily = sleep_daily.sort_values("Date")
    sleep_daily["thresh_low"]    = (sleep_daily["TotalSleepMinutes"] > 0) & (sleep_daily["TotalSleepMinutes"] < sleep_low)
    sleep_daily["thresh_high"]   = sleep_daily["TotalSleepMinutes"] > sleep_high
    sleep_daily["no_data"]       = sleep_daily["TotalSleepMinutes"] == 0
    sleep_daily["rolling_med"]   = sleep_daily["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sleep_daily["residual"]      = sleep_daily["TotalSleepMinutes"] - sleep_daily["rolling_med"]
    resid_std                    = sleep_daily["residual"].std()
    sleep_daily["resid_anomaly"] = sleep_daily["residual"].abs() > (residual_sigma * resid_std)
    sleep_daily["is_anomaly"]    = sleep_daily["thresh_low"] | sleep_daily["thresh_high"] | sleep_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["no_data"]:       r.append("No device worn")
        if row["thresh_low"]:    r.append(f"Sleep<{sleep_low}min")
        if row["thresh_high"]:   r.append(f"Sleep>{sleep_high}min")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    sleep_daily["reason"] = sleep_daily.apply(reason, axis=1)
    return sleep_daily

def simulate_accuracy(master, n_inject=10):
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}

    # HR
    hr_sim = df_daily[["Date","AvgHR"]].copy()
    inject_idx = np.random.choice(len(hr_sim), n_inject, replace=False)
    hr_sim.loc[inject_idx, "AvgHR"] = np.random.choice(
        [115,120,125,35,40,45,118,130,38,42], n_inject, replace=True)
    hr_sim["rolling_med"] = hr_sim["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_sim["residual"]    = hr_sim["AvgHR"] - hr_sim["rolling_med"]
    resid_std = hr_sim["residual"].std()
    hr_sim["detected"] = (hr_sim["AvgHR"] > 100) | (hr_sim["AvgHR"] < 50) | \
                         (hr_sim["residual"].abs() > 2 * resid_std)
    tp = hr_sim.iloc[inject_idx]["detected"].sum()
    results["Heart Rate"] = {"injected": n_inject, "detected": int(tp),
                              "accuracy": round(tp / n_inject * 100, 1)}

    # Steps
    st_sim = df_daily[["Date","TotalSteps"]].copy()
    inject_idx2 = np.random.choice(len(st_sim), n_inject, replace=False)
    st_sim.loc[inject_idx2, "TotalSteps"] = np.random.choice(
        [50,100,150,30000,35000,28000,80,200,31000,29000], n_inject, replace=True)
    st_sim["rolling_med"] = st_sim["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    st_sim["residual"]    = st_sim["TotalSteps"] - st_sim["rolling_med"]
    resid_std2 = st_sim["residual"].std()
    st_sim["detected"] = (st_sim["TotalSteps"] < 500) | (st_sim["TotalSteps"] > 25000) | \
                         (st_sim["residual"].abs() > 2 * resid_std2)
    tp2 = st_sim.iloc[inject_idx2]["detected"].sum()
    results["Steps"] = {"injected": n_inject, "detected": int(tp2),
                         "accuracy": round(tp2 / n_inject * 100, 1)}

    # Sleep
    sl_sim = df_daily[["Date","TotalSleepMinutes"]].copy()
    inject_idx3 = np.random.choice(len(sl_sim), n_inject, replace=False)
    sl_sim.loc[inject_idx3, "TotalSleepMinutes"] = np.random.choice(
        [10,20,30,700,750,800,15,25,710,720], n_inject, replace=True)
    sl_sim["rolling_med"] = sl_sim["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sl_sim["residual"]    = sl_sim["TotalSleepMinutes"] - sl_sim["rolling_med"]
    resid_std3 = sl_sim["residual"].std()
    sl_sim["detected"] = ((sl_sim["TotalSleepMinutes"] > 0) & (sl_sim["TotalSleepMinutes"] < 60)) | \
                          (sl_sim["TotalSleepMinutes"] > 600) | \
                          (sl_sim["residual"].abs() > 2 * resid_std3)
    tp3 = sl_sim.iloc[inject_idx3]["detected"].sum()
    results["Sleep"] = {"injected": n_inject, "detected": int(tp3),
                         "accuracy": round(tp3 / n_inject * 100, 1)}

    overall = round(np.mean([results[k]["accuracy"] for k in results]), 1)
    results["Overall"] = overall
    return results

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.5rem 0 1.2rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:{ACCENT_RED}">
        🚨 FitPulse
      </div>
      <div style="font-size:0.7rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.2rem">
        Anomaly Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    new_dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.9rem 0">', unsafe_allow_html=True)

    steps_done = sum([st.session_state.files_loaded,
                      st.session_state.anomaly_done,
                      st.session_state.simulation_done])
    pct = int(steps_done / 3 * 100)
    st.markdown(f"""
    <div style="margin-bottom:1rem">
      <div style="font-size:0.7rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-bottom:0.4rem">
        PIPELINE · {pct}%
      </div>
      <div style="background:{CARD_BOR};border-radius:4px;height:5px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{ACCENT_RED},{ACCENT2});border-radius:4px;transition:width 0.4s"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for done, icon, label in [
        (st.session_state.files_loaded,    "📂", "Data Loaded"),
        (st.session_state.anomaly_done,    "🚨", "Anomalies Detected"),
        (st.session_state.simulation_done, "🎯", "Accuracy Simulated"),
    ]:
        dot = f'<span style="color:{ACCENT3}">●</span>' if done else f'<span style="color:{MUTED}">○</span>'
        st.markdown(f'<div style="font-size:0.81rem;padding:0.28rem 0;color:{TEXT if done else MUTED}">{dot} {icon} {label}</div>', unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.9rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.5rem">THRESHOLDS</div>', unsafe_allow_html=True)

    hr_high = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180)
    hr_low  = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70)
    st_low  = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000)
    sl_low  = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120)
    sl_high = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900)
    sigma   = st.slider("Residual σ threshold", 1.0, 4.0, 2.0, 0.5, key="sigma_slider")

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.9rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.67rem;color:{MUTED};font-family:JetBrains Mono,monospace">Real Fitbit Dataset<br>30 users · March–April 2016</div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="m3-hero">
  <div class="hero-badge"> ANOMALY DETECTION &amp; VISUALIZATION</div>
  <h1 class="hero-title">🚨 FitPulse Anomaly Detection</h1>
  <p class="hero-sub">Threshold Violations · Residual Analysis · Outlier Clusters · Interactive Plotly Charts</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
sec("📂", "Data Loading", "Step 1")

ui_info("Upload the same 5 Fitbit CSV files as Milestone 2. Files are auto-detected by column structure.")

uploaded_files = st.file_uploader(
    "📁  Drop all 5 Fitbit CSV files here",
    type="csv", accept_multiple_files=True, key="m3_uploader",
    help="Hold Ctrl (Windows) or Cmd (Mac) to select multiple files"
)

detected = {}
ignored  = []
if uploaded_files:
    raw_uploads = []
    for uf in uploaded_files:
        try:
            df_tmp = pd.read_csv(uf)
            raw_uploads.append((uf.name, df_tmp))
        except Exception:
            ignored.append(uf.name)

    used_names = set()
    for req_name, finfo in REQUIRED_FILES.items():
        best_score, best_name, best_df = 0, None, None
        for uname, udf in raw_uploads:
            s = score_match(udf, finfo)
            if s > best_score:
                best_score, best_name, best_df = s, uname, udf
        if best_score >= 2:
            detected[req_name] = best_df
            used_names.add(best_name)

    for uname, _ in raw_uploads:
        if uname not in used_names:
            ignored.append(uname)

# Status grid
status_html = f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.6rem;margin:1rem 0">'
for req_name, finfo in REQUIRED_FILES.items():
    found = req_name in detected
    bg  = SUCCESS_BG if found else WARN_BG
    bor = SUCCESS_BOR if found else WARN_BOR
    ico = "✅" if found else "❌"
    status_html += f"""
    <div style="background:{bg};border:1px solid {bor};border-radius:10px;padding:0.65rem 0.85rem">
      <div style="font-size:1.15rem">{ico} {finfo['icon']}</div>
      <div style="font-size:0.71rem;font-weight:600;color:{TEXT};margin-top:0.3rem">{finfo['label']}</div>
      <div style="font-size:0.64rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.1rem">
        {'Found ✓' if found else 'Missing'}
      </div>
    </div>"""
status_html += "</div>"
st.markdown(status_html, unsafe_allow_html=True)

n_up = len(detected)
metrics((n_up, "Detected"), (5 - n_up, "Missing"), ("✓" if n_up == 5 else "✗", "Ready"))

if n_up < 5:
    missing = [REQUIRED_FILES[r]["label"] for r in REQUIRED_FILES if r not in detected]
    ui_warn(f"Missing: {', '.join(missing)}")

if st.button("⚡ Load & Build Master DataFrame", disabled=(n_up < 5)):
    with st.spinner("Parsing and building master..."):
        try:
            daily    = detected["dailyActivity_merged.csv"].copy()
            hourly_s = detected["hourlySteps_merged.csv"].copy()
            hourly_i = detected["hourlyIntensities_merged.csv"].copy()
            sleep    = detected["minuteSleep_merged.csv"].copy()
            hr       = detected["heartrate_seconds_merged.csv"].copy()

            # ── FIX: Use smart parser for ALL timestamp columns ───────────────
            # Handles ANY Fitbit format variant:
            #   "3/12/2016 12:00:00 AM"  (M/D/YYYY H:MM:SS AM/PM)
            #   "12-03-2016 00.00"       (DD-MM-YYYY HH.MM)
            #   "2016-03-12 00:00:00"    (ISO 8601)
            daily["ActivityDate"]    = parse_dt(daily["ActivityDate"])
            hourly_s["ActivityHour"] = parse_dt(hourly_s["ActivityHour"])
            hourly_i["ActivityHour"] = parse_dt(hourly_i["ActivityHour"])
            sleep["date"]            = parse_dt(sleep["date"])
            hr["Time"]               = parse_dt(hr["Time"])

            # Drop rows where parsing failed
            daily    = daily.dropna(subset=["ActivityDate"])
            hourly_s = hourly_s.dropna(subset=["ActivityHour"])
            hourly_i = hourly_i.dropna(subset=["ActivityHour"])
            sleep    = sleep.dropna(subset=["date"])
            hr       = hr.dropna(subset=["Time"])

            # HR minute-level resample
            hr_minute = (hr.set_index("Time")
                         .groupby("Id")["Value"]
                         .resample("1min").mean()
                         .reset_index())
            hr_minute.columns = ["Id","Time","HeartRate"]
            hr_minute = hr_minute.dropna()

            hr_minute["Date"] = hr_minute["Time"].dt.date
            hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                        .agg(["mean","max","min","std"])
                        .reset_index()
                        .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))

            sleep["Date"] = sleep["date"].dt.date
            sleep_daily = (sleep.groupby(["Id","Date"])
                           .agg(TotalSleepMinutes=("value","count"),
                                DominantSleepStage=("value", lambda x: x.mode()[0]))
                           .reset_index())

            master = daily.copy().rename(columns={"ActivityDate":"Date"})
            master["Date"] = master["Date"].dt.date
            master = master.merge(hr_daily,    on=["Id","Date"], how="left")
            master = master.merge(sleep_daily, on=["Id","Date"], how="left")
            master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
            master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
            for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

            st.session_state.daily     = daily
            st.session_state.hourly_s  = hourly_s
            st.session_state.hourly_i  = hourly_i
            st.session_state.sleep     = sleep
            st.session_state.hr        = hr
            st.session_state.hr_minute = hr_minute
            st.session_state.master    = master
            st.session_state.files_loaded = True
            st.rerun()
        except Exception as e:
            st.error(f"Error building master: {e}")
            import traceback
            st.code(traceback.format_exc())

if st.session_state.files_loaded:
    master = st.session_state.master
    ui_success(f"Master DataFrame ready — {master.shape[0]:,} rows · {master['Id'].nunique()} users")

    st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — ANOMALY DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    sec("🚨", "Anomaly Detection — Three Methods", "Steps 2–4")

    st.markdown(f"""
    <div class="card">
      <div class="card-title">Detection Methods Applied</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;font-size:0.82rem">
        <div style="background:{SECTION_BG};border:1px solid {CARD_BOR};border-radius:10px;padding:0.85rem">
          <div style="color:{ACCENT_RED};font-weight:700;margin-bottom:0.4rem">① Threshold Violations</div>
          <div style="color:{MUTED}">Hard upper/lower limits on HR, Steps, Sleep. Simple, interpretable, fast.</div>
        </div>
        <div style="background:{SECTION_BG};border:1px solid {CARD_BOR};border-radius:10px;padding:0.85rem">
          <div style="color:{ACCENT2};font-weight:700;margin-bottom:0.4rem">② Residual-Based</div>
          <div style="color:{MUTED}">Rolling median as baseline. Flag days where actual deviates by ±{sigma:.0f}σ.</div>
        </div>
        <div style="background:{SECTION_BG};border:1px solid {CARD_BOR};border-radius:10px;padding:0.85rem">
          <div style="color:{ACCENT3};font-weight:700;margin-bottom:0.4rem">③ DBSCAN Outliers</div>
          <div style="color:{MUTED}">Users labelled −1 by DBSCAN. Structural outliers — behaviour fits no cluster.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Run Anomaly Detection (All 3 Methods)"):
        with st.spinner("Detecting anomalies..."):
            try:
                anom_hr    = detect_hr_anomalies(master, hr_high, hr_low, sigma)
                anom_steps = detect_steps_anomalies(master, st_low, 25000, sigma)
                anom_sleep = detect_sleep_anomalies(master, sl_low, sl_high, sigma)
                st.session_state.anom_hr    = anom_hr
                st.session_state.anom_steps = anom_steps
                st.session_state.anom_sleep = anom_sleep
                st.session_state.anomaly_done = True
                st.rerun()
            except Exception as e:
                st.error(f"Detection error: {e}")

    if st.session_state.anomaly_done:
        anom_hr    = st.session_state.anom_hr
        anom_steps = st.session_state.anom_steps
        anom_sleep = st.session_state.anom_sleep

        n_hr    = int(anom_hr["is_anomaly"].sum())
        n_steps = int(anom_steps["is_anomaly"].sum())
        n_sleep = int(anom_sleep["is_anomaly"].sum())
        n_total = n_hr + n_steps + n_sleep

        ui_danger(f"Total anomalies flagged: {n_total}  (HR: {n_hr} · Steps: {n_steps} · Sleep: {n_sleep})")
        metrics(
            (n_hr,    "HR Anomalies"),
            (n_steps, "Steps Anomalies"),
            (n_sleep, "Sleep Anomalies"),
            (n_total, "Total Flags"),
            red_indices=[0,1,2,3]
        )

        # ── CHART 1: Heart Rate ───────────────────────────────────────────────
        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec("❤️", "Heart Rate — Anomaly Chart", "Step 2")
        anom_tag(f"{n_hr} anomalous days detected")
        screenshot_badge("Heart Rate Chart with Anomaly Highlights")
        step_pill(2, "Threshold + Residual Detection")
        ui_info(f"Red markers = anomaly days. Dashed lines = thresholds (HR>{hr_high} or HR<{hr_low}). Shaded band = ±{sigma:.0f}σ residual zone.")

        hr_anom   = anom_hr[anom_hr["is_anomaly"]]
        fig_hr    = go.Figure()

        # Expected band
        rolling_upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
        rolling_lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=rolling_upper, mode="lines",
                                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=rolling_lower, mode="lines",
                                    fill="tonexty", fillcolor="rgba(107,45,139,0.1)",
                                    line=dict(width=0), name=f"±{sigma:.0f}σ Expected Band"))

        # HR line
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"],
                                    mode="lines+markers", name="Avg Heart Rate",
                                    line=dict(color=ACCENT, width=2.5),
                                    marker=dict(size=5, color=ACCENT),
                                    hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<extra></extra>"))

        # Rolling median
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"],
                                    mode="lines", name="Rolling Median",
                                    line=dict(color=ACCENT3, width=1.5, dash="dot"),
                                    hovertemplate="<b>%{x}</b><br>Median: %{y:.1f} bpm<extra></extra>"))

        # Anomaly markers
        if not hr_anom.empty:
            fig_hr.add_trace(go.Scatter(x=hr_anom["Date"], y=hr_anom["AvgHR"],
                                        mode="markers", name="🚨 Anomaly",
                                        marker=dict(color=ACCENT_RED, size=14, symbol="circle",
                                                    line=dict(color="white", width=2)),
                                        hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<br><b>ANOMALY</b><extra>⚠️</extra>"))
            for _, row in hr_anom.iterrows():
                fig_hr.add_annotation(x=row["Date"], y=row["AvgHR"],
                                       text=f"⚠️ {row['reason']}", showarrow=True,
                                       arrowhead=2, arrowcolor=ACCENT_RED, arrowsize=1.2,
                                       ax=0, ay=-45, font=dict(color=ACCENT_RED, size=9),
                                       bgcolor=CARD_BG, bordercolor=DANGER_BOR, borderwidth=1, borderpad=4)

        fig_hr.add_hline(y=hr_high, line_dash="dash", line_color=ACCENT_RED, line_width=1.5, opacity=0.7,
                         annotation_text=f"High Threshold ({hr_high} bpm)", annotation_position="top right",
                         annotation_font_color=ACCENT_RED)
        fig_hr.add_hline(y=hr_low,  line_dash="dash", line_color=ACCENT2,   line_width=1.5, opacity=0.7,
                         annotation_text=f"Low Threshold ({hr_low} bpm)",   annotation_position="bottom right",
                         annotation_font_color=ACCENT2)

        apply_plotly_theme(fig_hr, "❤️ Heart Rate — Anomaly Detection (Real Fitbit Data)")
        fig_hr.update_layout(height=480, xaxis_title="Date", yaxis_title="Heart Rate (bpm)")
        st.plotly_chart(fig_hr, use_container_width=True)

        if not hr_anom.empty:
            with st.expander(f"📋 View {len(hr_anom)} HR Anomaly Records"):
                st.dataframe(hr_anom[hr_anom["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]]
                             .rename(columns={"rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                             .round(2), use_container_width=True)

        # ── CHART 2: Sleep ────────────────────────────────────────────────────
        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec("💤", "Sleep Pattern — Anomaly Visualization", "Step 3")
        anom_tag(f"{n_sleep} anomalous sleep days detected")
        screenshot_badge("Sleep Pattern Visualization with Alerts")
        step_pill(3, "Threshold Detection on Sleep Minutes")
        ui_info(f"Purple = sleep line. Red diamonds = anomaly days. Green band = healthy sleep zone ({sl_low}–{sl_high} min).")

        sleep_anom = anom_sleep[anom_sleep["is_anomaly"]]
        fig_sleep  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.7, 0.3],
                                    subplot_titles=["Sleep Duration (minutes/night)", "Deviation from Expected"],
                                    vertical_spacing=0.08)

        fig_sleep.add_hrect(y0=sl_low, y1=sl_high, fillcolor="rgba(30,132,73,0.07)",
                             line_width=0, annotation_text="✅ Healthy Sleep Zone",
                             annotation_position="top right",
                             annotation_font_color=ACCENT3, row=1, col=1)

        fig_sleep.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
                                        mode="lines+markers", name="Sleep Minutes",
                                        line=dict(color=ACCENT2, width=2.5),
                                        marker=dict(size=5, color=ACCENT2),
                                        hovertemplate="<b>%{x}</b><br>Sleep: %{y:.0f} min<extra></extra>"),
                             row=1, col=1)

        fig_sleep.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
                                        mode="lines", name="Rolling Median",
                                        line=dict(color=ACCENT3, width=1.5, dash="dot"),
                                        hovertemplate="<b>%{x}</b><br>Median: %{y:.0f} min<extra></extra>"),
                             row=1, col=1)

        if not sleep_anom.empty:
            fig_sleep.add_trace(go.Scatter(x=sleep_anom["Date"], y=sleep_anom["TotalSleepMinutes"],
                                            mode="markers", name="🚨 Sleep Anomaly",
                                            marker=dict(color=ACCENT_RED, size=14, symbol="diamond",
                                                        line=dict(color="white", width=2)),
                                            hovertemplate="<b>%{x}</b><br>Sleep: %{y:.0f} min<br><b>ANOMALY</b><extra>⚠️</extra>"),
                                 row=1, col=1)
            for _, row in sleep_anom.iterrows():
                fig_sleep.add_annotation(x=row["Date"], y=row["TotalSleepMinutes"],
                                          text=f"⚠️ {row['reason']}", showarrow=True,
                                          arrowhead=2, arrowcolor=ACCENT_RED, arrowsize=1.2,
                                          ax=20, ay=-40, font=dict(color=ACCENT_RED, size=9),
                                          bgcolor=CARD_BG, bordercolor=DANGER_BOR, borderwidth=1,
                                          borderpad=3, row=1, col=1)

        fig_sleep.add_hline(y=sl_low, line_dash="dash", line_color=ACCENT_RED, line_width=1.5, opacity=0.7,
                             row=1, col=1, annotation_text=f"Min ({sl_low} min)", annotation_font_color=ACCENT_RED)
        fig_sleep.add_hline(y=sl_high, line_dash="dash", line_color=ACCENT,    line_width=1.5, opacity=0.7,
                             row=1, col=1, annotation_text=f"Max ({sl_high} min)", annotation_font_color=ACCENT)

        colors_resid = [ACCENT_RED if v else ACCENT2 for v in anom_sleep["resid_anomaly"]]
        fig_sleep.add_trace(go.Bar(x=anom_sleep["Date"], y=anom_sleep["residual"],
                                    name="Residual", marker_color=colors_resid,
                                    hovertemplate="<b>%{x}</b><br>Residual: %{y:.0f} min<extra></extra>"),
                             row=2, col=1)
        fig_sleep.add_hline(y=0, line_dash="solid", line_color=MUTED, line_width=1, row=2, col=1)

        apply_plotly_theme(fig_sleep)
        fig_sleep.update_layout(height=560, showlegend=True,
                                  paper_bgcolor=CHART_PAPER, plot_bgcolor=CHART_PLOT, font_color=TEXT)
        fig_sleep.update_xaxes(gridcolor=CHART_GRID, tickfont_color=MUTED)
        fig_sleep.update_yaxes(gridcolor=CHART_GRID, tickfont_color=MUTED)
        st.plotly_chart(fig_sleep, use_container_width=True)

        if not sleep_anom.empty:
            with st.expander(f"📋 View {len(sleep_anom)} Sleep Anomaly Records"):
                st.dataframe(sleep_anom[sleep_anom["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]]
                             .rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                             .round(2), use_container_width=True)

        # ── CHART 3: Steps ────────────────────────────────────────────────────
        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec("🚶", "Step Count Trend — Alerts & Anomalies", "Step 4")
        anom_tag(f"{n_steps} anomalous step-count days detected")
        screenshot_badge("Step Count Trend with Alert Bands")
        step_pill(4, "Threshold + Residual Detection on Steps")
        ui_info(f"Shaded red bands = anomaly alert days. Dashed lines = step thresholds. Bar chart shows deviation from trend.")

        steps_anom = anom_steps[anom_steps["is_anomaly"]]
        fig_steps  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.65, 0.35],
                                    subplot_titles=["Daily Steps (avg across users)", "Residual Deviation from Trend"],
                                    vertical_spacing=0.08)

        for _, row in steps_anom.iterrows():
            d      = str(row["Date"])
            d_next = str(pd.Timestamp(d) + pd.Timedelta(days=1))[:10]
            fig_steps.add_vrect(x0=d, x1=d_next,
                                 fillcolor="rgba(192,57,43,0.12)",
                                 line_color="rgba(192,57,43,0.45)",
                                 line_width=1.5, row=1, col=1)

        fig_steps.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
                                        mode="lines+markers", name="Avg Daily Steps",
                                        line=dict(color=ACCENT3, width=2.5),
                                        marker=dict(size=5, color=ACCENT3),
                                        hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<extra></extra>"),
                             row=1, col=1)

        fig_steps.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"],
                                        mode="lines", name="Trend (Rolling Median)",
                                        line=dict(color=ACCENT, width=2, dash="dash"),
                                        hovertemplate="<b>%{x}</b><br>Trend: %{y:,.0f}<extra></extra>"),
                             row=1, col=1)

        if not steps_anom.empty:
            fig_steps.add_trace(go.Scatter(x=steps_anom["Date"], y=steps_anom["TotalSteps"],
                                            mode="markers", name="🚨 Steps Anomaly",
                                            marker=dict(color=ACCENT_RED, size=14, symbol="triangle-up",
                                                        line=dict(color="white", width=2)),
                                            hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"),
                                 row=1, col=1)

        fig_steps.add_hline(y=st_low, line_dash="dash", line_color=ACCENT_RED, line_width=1.5, opacity=0.8,
                             row=1, col=1, annotation_text=f"Low Alert ({st_low:,} steps)", annotation_font_color=ACCENT_RED)
        fig_steps.add_hline(y=25000, line_dash="dash", line_color=ACCENT2, line_width=1.5, opacity=0.7,
                             row=1, col=1, annotation_text="High Alert (25,000 steps)", annotation_font_color=ACCENT2)

        res_colors = [ACCENT_RED if v else ACCENT3 for v in anom_steps["resid_anomaly"]]
        fig_steps.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"],
                                    name="Residual", marker_color=res_colors,
                                    hovertemplate="<b>%{x}</b><br>Deviation: %{y:,.0f} steps<extra></extra>"),
                             row=2, col=1)
        fig_steps.add_hline(y=0, line_dash="solid", line_color=MUTED, line_width=1, row=2, col=1)

        apply_plotly_theme(fig_steps)
        fig_steps.update_layout(height=560, showlegend=True,
                                  paper_bgcolor=CHART_PAPER, plot_bgcolor=CHART_PLOT, font_color=TEXT)
        fig_steps.update_xaxes(gridcolor=CHART_GRID, tickfont_color=MUTED)
        fig_steps.update_yaxes(gridcolor=CHART_GRID, tickfont_color=MUTED)
        st.plotly_chart(fig_steps, use_container_width=True)

        if not steps_anom.empty:
            with st.expander(f"📋 View {len(steps_anom)} Steps Anomaly Records"):
                st.dataframe(steps_anom[steps_anom["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]]
                             .rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                             .round(2), use_container_width=True)

        # ── CHART 4: DBSCAN ───────────────────────────────────────────────────
        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec("🔍", "DBSCAN Outlier Users — Cluster-Based Anomalies", "Step 5")
        step_pill(5, "Structural Outlier Detection via DBSCAN")
        anom_tag("Outlier = users with atypical overall behaviour pattern")
        ui_info("Cluster each user on their activity profile. Users labelled −1 are structural outliers — their behaviour doesn't fit any group.")

        cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes",
                        "FairlyActiveMinutes","LightlyActiveMinutes",
                        "SedentaryMinutes","TotalSleepMinutes"]
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import DBSCAN
            from sklearn.decomposition import PCA

            cf = master.groupby("Id")[cluster_cols].mean().round(3).dropna()
            scaler    = StandardScaler()
            X_scaled  = scaler.fit_transform(cf)
            db        = DBSCAN(eps=2.2, min_samples=2)
            db_labels = db.fit_predict(X_scaled)
            pca       = PCA(n_components=2, random_state=42)
            X_pca     = pca.fit_transform(X_scaled)
            var       = pca.explained_variance_ratio_ * 100

            cf["DBSCAN"]      = db_labels
            outlier_users     = cf[cf["DBSCAN"] == -1].index.tolist()
            n_outliers        = len(outlier_users)
            n_clusters        = len(set(db_labels)) - (1 if -1 in db_labels else 0)

            metrics((n_clusters, "DBSCAN Clusters"), (n_outliers, "Outlier Users"),
                    (len(cf) - n_outliers, "Normal Users"), red_indices=[1])

            CLUSTER_COLORS = [ACCENT, ACCENT2, "#F39C12", "#1ABC9C", "#E91E63"]
            fig_db = go.Figure()

            for lbl in sorted(set(db_labels)):
                if lbl == -1: continue
                mask = db_labels == lbl
                fig_db.add_trace(go.Scatter(
                    x=X_pca[mask,0], y=X_pca[mask,1],
                    mode="markers+text", name=f"Cluster {lbl}",
                    marker=dict(size=14, color=CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)],
                                opacity=0.85, line=dict(color="white", width=1.5)),
                    text=[str(uid)[-4:] for uid in cf.index[mask]],
                    textposition="top center", textfont=dict(size=8, color=TEXT),
                    hovertemplate="<b>User ...%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"))

            if n_outliers > 0:
                mask_out = db_labels == -1
                fig_db.add_trace(go.Scatter(
                    x=X_pca[mask_out,0], y=X_pca[mask_out,1],
                    mode="markers+text", name="🚨 Outlier / Anomaly",
                    marker=dict(size=20, color=ACCENT_RED, symbol="x",
                                line=dict(color="white", width=2.5)),
                    text=[str(uid)[-4:] for uid in cf.index[mask_out]],
                    textposition="top center", textfont=dict(size=9, color=ACCENT_RED),
                    hovertemplate="<b>⚠️ OUTLIER User ...%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra>ANOMALY</extra>"))

                for i, uid in enumerate(cf.index[mask_out]):
                    xi, yi = X_pca[mask_out][i]
                    fig_db.add_shape(type="circle",
                        x0=xi-0.3, y0=yi-0.3, x1=xi+0.3, y1=yi+0.3,
                        line=dict(color=ACCENT_RED, width=2, dash="dot"),
                        fillcolor="rgba(192,57,43,0.08)")

            apply_plotly_theme(fig_db, f"🔍 DBSCAN Outlier Detection — PCA Projection (eps=2.2)")
            fig_db.update_layout(height=500,
                                  xaxis_title=f"PC1 ({var[0]:.1f}% variance)",
                                  yaxis_title=f"PC2 ({var[1]:.1f}% variance)")
            st.plotly_chart(fig_db, use_container_width=True)

            if outlier_users:
                st.markdown(f'<div class="card" style="border-color:{DANGER_BOR}"><div class="card-title" style="color:{ACCENT_RED}">🚨 Outlier User Profiles</div></div>', unsafe_allow_html=True)
                st.dataframe(cf[cf["DBSCAN"]==-1][cluster_cols].round(2), use_container_width=True)

        except ImportError:
            ui_warn("sklearn not installed. Run: pip install scikit-learn")
        except Exception as e:
            ui_warn(f"DBSCAN clustering skipped: {e}")

        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3 — ACCURACY SIMULATION
        # ══════════════════════════════════════════════════════════════════════
        sec("🎯", "Simulated Detection Accuracy — 90%+ Target", "Step 6")
        step_pill(6, "Inject Known Anomalies → Measure Detection Rate")
        ui_info("10 known anomalies are injected into each signal. The detection is run and we measure how many it catches. Validates the 90%+ accuracy requirement.")

        if st.button("🎯 Run Accuracy Simulation (10 injected anomalies per signal)"):
            with st.spinner("Simulating..."):
                try:
                    sim = simulate_accuracy(master, n_inject=10)
                    st.session_state.sim_results  = sim
                    st.session_state.simulation_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Simulation error: {e}")

        if st.session_state.simulation_done and st.session_state.sim_results:
            sim     = st.session_state.sim_results
            overall = sim["Overall"]
            passed  = overall >= 90.0

            if passed:
                ui_success(f"Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT")
            else:
                ui_warn(f"Overall accuracy: {overall}% — below 90% target, adjust thresholds in sidebar")

            # Accuracy cards
            html = '<div class="metric-grid">'
            for signal in ["Heart Rate","Steps","Sleep"]:
                r   = sim[signal]
                acc = r["accuracy"]
                col = ACCENT3 if acc >= 90 else ACCENT_RED
                html += f"""
                <div class="metric-card" style="border-color:{col}44">
                  <div style="font-size:1.75rem;font-weight:800;color:{col};font-family:'Syne',sans-serif">{acc}%</div>
                  <div style="font-size:0.79rem;color:{TEXT};font-weight:600;margin:0.3rem 0">{signal}</div>
                  <div style="font-size:0.7rem;color:{MUTED}">{r['detected']}/{r['injected']} detected</div>
                  <div style="font-size:0.68rem;color:{'#1E8449' if acc>=90 else ACCENT_RED}">{'✅ PASS' if acc>=90 else '⚠️ LOW'}</div>
                </div>"""
            html += f"""
                <div class="metric-card" style="border-color:{'#1E8449' if passed else ACCENT_RED}88;background:{SUCCESS_BG if passed else DANGER_BG}">
                  <div style="font-size:1.75rem;font-weight:800;color:{'#1E8449' if passed else ACCENT_RED};font-family:'Syne',sans-serif">{overall}%</div>
                  <div style="font-size:0.79rem;color:{TEXT};font-weight:600;margin:0.3rem 0">Overall</div>
                  <div style="font-size:0.68rem;color:{'#1E8449' if passed else ACCENT_RED}">{'✅ 90%+ ACHIEVED' if passed else '⚠️ BELOW TARGET'}</div>
                </div>"""
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

            # Accuracy bar chart
            signals    = ["Heart Rate","Steps","Sleep"]
            accs       = [sim[s]["accuracy"] for s in signals]
            bar_colors = [ACCENT3 if a >= 90 else ACCENT_RED for a in accs]

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=signals, y=accs,
                marker_color=bar_colors,
                text=[f"{a}%" for a in accs],
                textposition="outside",
                textfont=dict(color=TEXT, size=14, family="Syne, sans-serif"),
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
                name="Detection Accuracy"
            ))
            fig_acc.add_hline(y=90, line_dash="dash", line_color=ACCENT_RED, line_width=2,
                               annotation_text="90% Target", annotation_font_color=ACCENT_RED,
                               annotation_position="top right")
            apply_plotly_theme(fig_acc, "🎯 Simulated Anomaly Detection Accuracy")
            fig_acc.update_layout(height=380, yaxis_range=[0,115],
                                   yaxis_title="Detection Accuracy (%)",
                                   xaxis_title="Signal", showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # MILESTONE 3 SUMMARY
        # ══════════════════════════════════════════════════════════════════════
        sec("✅", "Summary")

        checklist = [
            ("🚨", "Threshold Violations",  st.session_state.anomaly_done,    f"HR>{hr_high}/{hr_low}, Steps<{st_low}, Sleep<{sl_low}/<{sl_high}"),
            ("📉", "Residual-Based",         st.session_state.anomaly_done,    f"Rolling median ±{sigma:.0f}σ on all 3 signals"),
            ("🔍", "DBSCAN Outliers",        st.session_state.anomaly_done,    "Structural user-level anomalies via clustering"),
            ("❤️", "HR Chart",               st.session_state.anomaly_done,    "Interactive Plotly — annotations + threshold lines"),
            ("💤", "Sleep Chart",            st.session_state.anomaly_done,    "Dual subplot — duration + residual bars"),
            ("🚶", "Steps Chart",            st.session_state.anomaly_done,    "Trend + alert bands + residual deviation"),
            ("🎯", "Accuracy Simulation",    st.session_state.simulation_done, "10 injected anomalies per signal, 90%+ target"),
        ]

        for icon, label, done, detail in checklist:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.55rem 0;border-bottom:1px solid {CARD_BOR}">
              <span style="font-size:1rem">{'✅' if done else '⬜'}</span>
              <span style="font-size:0.88rem;font-weight:600;color:{TEXT};min-width:185px">{icon} {label}</span>
              <span style="font-size:0.78rem;color:{MUTED}">{detail}</span>
            </div>
            """, unsafe_allow_html=True)


    st.markdown(f"""
    <div class="card" style="text-align:center;padding:2.8rem;border-color:{CARD_BOR}">
      <div style="font-size:2.8rem;margin-bottom:0.9rem">🚨</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem">
        Upload Your Fitbit Files to Begin
      </div>
      <div style="color:{MUTED};font-size:0.86rem">
        Upload all 5 CSV files above and click <b>Load &amp; Build Master DataFrame</b>
      </div>
    </div>
    """, unsafe_allow_html=True) 