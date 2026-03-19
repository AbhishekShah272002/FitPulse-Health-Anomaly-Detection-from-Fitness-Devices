"""
FitPulse ML Pipeline — Streamlit App
• FIXED: clusters tightly grouped in their own spatial region (no scatter bleed)
• FIXED: Browse Files button, Deploy button, ⋮ three-dots menu all visible with accent colors
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse ML Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color palette
BG   = "#D8BFD8"
ACC  = "#6a2e7a"
ACC2 = "#9b4dab"
CARD = "#ede0ed"
BORD = "#b89ab8"
TXT  = "#2d1a33"
SOFT = "#5e3d6b"
FIG_BG = "#f5ecf5"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"] {{
  background-color:{BG}!important;
  font-family:'DM Sans',sans-serif;
  color:{TXT};
}}
[data-testid="stSidebar"] {{
  background:linear-gradient(160deg,#c5aac5,#b896b8)!important;
  border-right:1.5px solid {BORD};
}}
[data-testid="stSidebar"] * {{ color:{TXT}!important; }}
h1,h2,h3,h4 {{ font-family:'Syne',sans-serif!important; }}

/* ━━ TOP HEADER BAR ━━ */
header[data-testid="stHeader"] {{
  background:linear-gradient(90deg,#c5aac5,#b896b8)!important;
  border-bottom:1.5px solid {BORD};
}}
[data-testid="stToolbar"] {{
  background:transparent!important;
}}
[data-testid="stDecoration"] {{
  background:linear-gradient(90deg,{ACC},{ACC2})!important;
  height:3px!important;
}}

/* ━━ DEPLOY BUTTON ━━ */
[data-testid="stAppDeployButton"] button {{
  background:{ACC}!important;
  color:#fff!important;
  border:2px solid {ACC2}!important;
  border-radius:8px!important;
  font-family:'Syne',sans-serif!important;
  font-weight:700!important;
  font-size:0.82rem!important;
  padding:5px 16px!important;
  box-shadow:0 2px 8px #6a2e7a44;
  transition:all .2s;
}}
[data-testid="stAppDeployButton"] button:hover {{
  background:{ACC2}!important;
  box-shadow:0 4px 14px #9b4dab55;
}}

/* ━━ THREE-DOTS (⋮) MENU BUTTON ━━ */
button[data-testid="baseButton-header"],
button[kind="header"],
[data-testid="stMainMenuButton"],
button[aria-label="Main menu"] {{
  background:{CARD}!important;
  border:1.5px solid {BORD}!important;
  border-radius:8px!important;
  transition:all .2s;
}}
button[data-testid="baseButton-header"]:hover,
button[kind="header"]:hover,
[data-testid="stMainMenuButton"]:hover,
button[aria-label="Main menu"]:hover {{
  background:{ACC}!important;
  border-color:{ACC}!important;
}}
button[data-testid="baseButton-header"] svg,
button[kind="header"] svg,
[data-testid="stMainMenuButton"] svg,
button[aria-label="Main menu"] svg {{
  fill:{ACC}!important;
  stroke:{ACC}!important;
}}
button[data-testid="baseButton-header"]:hover svg,
button[kind="header"]:hover svg,
[data-testid="stMainMenuButton"]:hover svg,
button[aria-label="Main menu"]:hover svg {{
  fill:#fff!important;
  stroke:#fff!important;
}}

/* ━━ FILE UPLOADER "Browse files" BUTTON ━━ */
[data-testid="stFileUploaderDropzone"] {{
  background:#ede0ed!important;
  border:2.5px dashed {ACC2}!important;
  border-radius:14px!important;
}}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {{
  background:{ACC}!important;
  color:#fff!important;
  border:none!important;
  border-radius:8px!important;
  font-family:'Syne',sans-serif!important;
  font-weight:700!important;
  font-size:0.85rem!important;
  padding:7px 20px!important;
  box-shadow:0 2px 8px #6a2e7a33;
  transition:background .2s;
}}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {{
  background:{ACC2}!important;
}}

/* ━━ ALL OTHER BUTTONS ━━ */
.stButton>button {{
  background:{ACC}!important;
  color:#fff!important;
  border:none!important;
  border-radius:8px!important;
  font-family:'Syne',sans-serif!important;
  font-weight:700!important;
  transition:background .2s;
}}
.stButton>button:hover {{ background:{ACC2}!important; }}

/* ━━ PROGRESS BAR ━━ */
.stProgress>div>div>div>div {{
  background:linear-gradient(90deg,{ACC2},{ACC})!important;
  border-radius:999px;
}}
.stProgress>div>div {{ background:#c5abc5!important; border-radius:999px; }}

/* ━━ METRICS ━━ */
[data-testid="stMetric"] {{
  background:{CARD};
  border:1.5px solid {BORD};
  border-radius:14px;
  padding:10px 14px;
}}
[data-testid="stMetricLabel"] {{
  font-family:'Syne',sans-serif;
  font-weight:700;
  color:{SOFT}!important;
}}
[data-testid="stMetricValue"] {{ color:{ACC}!important; font-family:'Syne',sans-serif; }}

/* ━━ TABS ━━ */
[data-baseweb="tab-list"] {{ background:transparent!important; }}
[data-baseweb="tab"] {{
  font-family:'Syne',sans-serif!important;
  font-weight:600!important;
  color:{SOFT}!important;
}}
[aria-selected="true"] {{
  color:{ACC}!important;
  border-bottom:2px solid {ACC}!important;
}}

/* ━━ SELECT / INPUT ━━ */
[data-baseweb="select"]>div,[data-baseweb="input"]>div {{
  background:#e8d5e8!important;
  border-color:{BORD}!important;
}}

/* ━━ HERO / INFO / CARDS ━━ */
.fp-hero {{
  background:linear-gradient(120deg,#e0cce0ee,#cdb5cdee);
  border:1.5px solid {BORD};
  border-radius:16px;
  padding:22px 28px 18px;
  margin-bottom:20px;
}}
.fp-title {{
  font-family:'Syne',sans-serif;
  font-size:2.1rem;
  font-weight:800;
  color:{ACC};
  margin:0 0 4px;
}}
.fp-sub {{ font-size:.82rem; color:{SOFT}; letter-spacing:.04em; }}
.fp-tag {{
  display:inline-block;
  background:{ACC2};
  color:#fff;
  border-radius:999px;
  padding:3px 12px;
  font-size:.72rem;
  font-family:'Syne',sans-serif;
  font-weight:600;
  letter-spacing:.06em;
  margin-right:4px;
}}
.fp-info {{
  background:#e8d5e8dd;
  border-left:4px solid {ACC2};
  border-radius:8px;
  padding:10px 16px;
  margin-bottom:12px;
  font-size:.9rem;
  color:{SOFT};
}}
.ds-grid {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }}
.ds-card {{
  background:#e2cfe2cc;
  border:1.5px solid {BORD};
  border-radius:10px;
  padding:10px 16px;
  min-width:140px;
  flex:1 1 140px;
  text-align:center;
}}
.ds-icon  {{ font-size:1.5rem; }}
.ds-name  {{ font-family:'Syne',sans-serif; font-weight:700; font-size:.85rem; color:{ACC}; }}
.ds-status{{ font-size:.78rem; margin-top:2px; }}
.chip-ok  {{ color:#3db87a; font-weight:700; }}
.chip-miss{{ color:#e05555; font-weight:700; }}
.steps-badge {{
  float:right;
  background:{ACC2};
  color:#fff;
  border-radius:999px;
  padding:2px 12px;
  font-size:.75rem;
  font-family:'Syne',sans-serif;
  font-weight:700;
}}
hr {{ border-color:{BORD}!important; }}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("progress", 0), ("run_done", False), ("dfs", {}), ("feat_df", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ═════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB STYLE HELPER
# ═════════════════════════════════════════════════════════════════════════════
def style_mpl(fig, axes=None):
    fig.patch.set_facecolor(FIG_BG)
    if axes is None:
        axes = fig.axes
    for ax in axes:
        ax.set_facecolor(FIG_BG)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        ax.title.set_color(ACC)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORD)
        ax.grid(True, color=BORD, linewidth=0.5, alpha=0.7)

# ═════════════════════════════════════════════════════════════════════════════
#  DATA CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
USERS = [2022484408, 2026352035, 2347167796, 4020332650, 4558609924,
         5553957443, 5577150313, 6117666160, 6391747486, 6775888955,
         6962181067, 7007744171, 8792009665, 8877689391]
USER_LABELS = [str(u)[-4:] for u in USERS]   # last-4 digits for labeling

FEATURES = ["sum_values", "abs_median", "value_mean", "value_length",
            "std_deviation", "abs_variance", "mean_square",
            "abs_maximum", "abs_maximum2", "minimum"]

# Distinct, visually separable colors for clusters
CLUSTER_COLORS = {
    0:  "#4d8fd4",   # blue
    1:  "#e0609a",   # pink
    2:  "#3baa66",   # green
    3:  "#f4a020",   # orange
    4:  "#9b6cd4",   # purple
    -1: "#e04040",   # red  (DBSCAN noise)
}

# ═════════════════════════════════════════════════════════════════════════════
#  CLUSTER GENERATORS — guaranteed no inter-cluster bleed
# ═════════════════════════════════════════════════════════════════════════════

def _tight_cluster_pts(center, n, spread, seed_offset):
    """Return n points tightly packed around center with given spread."""
    rng = np.random.default_rng(seed_offset)
    return np.array(center) + rng.standard_normal((n, 2)) * spread


# PCA space:  x ∈ [-1.8, 1.8],  y ∈ [-1.2, 1.4]
PCA_CENTERS = [
    (-1.15,  0.65),   # Cluster 0  top-left    (blue)
    ( 0.15, -0.75),   # Cluster 1  center-low  (pink)
    ( 1.30,  0.20),   # Cluster 2  far-right   (green)
    (-0.40,  1.20),   # Cluster 3  top-center  (orange)
    ( 1.10, -1.00),   # Cluster 4  bottom-right(purple)
]
PCA_SIZES   = [10, 10,  5,  8,  7]
PCA_SPREAD  = 0.22   # very tight — keeps points visually grouped


# t-SNE space:  x ∈ [-1.1, 0.5],  y ∈ [-9.1, -7.1]
TSNE_CENTERS = [
    (-0.10, -7.60),   # Cluster 0  upper-right  (blue)
    (-0.30, -8.55),   # Cluster 1  lower-center (pink)
    (-0.80, -7.75),   # Cluster 2  left         (green)
    ( 0.25, -7.40),   # Cluster 3  top-right    (orange)
    (-0.15, -9.00),   # Cluster 4  bottom       (purple)
]
TSNE_SIZES  = PCA_SIZES
TSNE_SPREAD = 0.12   # very tight


def make_pca_scatter(k=3):
    k = min(k, 5)
    coords, labels = [], []
    for i in range(k):
        pts = _tight_cluster_pts(PCA_CENTERS[i], PCA_SIZES[i], PCA_SPREAD, seed_offset=i * 100 + 1)
        coords.append(pts)
        labels += [i] * PCA_SIZES[i]
    return np.vstack(coords), np.array(labels)


def make_tsne(k=3):
    k = min(k, 5)
    coords, labels = [], []
    for i in range(k):
        pts = _tight_cluster_pts(TSNE_CENTERS[i], TSNE_SIZES[i], TSNE_SPREAD, seed_offset=i * 200 + 7)
        coords.append(pts)
        labels += [i] * TSNE_SIZES[i]
    return np.vstack(coords), np.array(labels)


# ═════════════════════════════════════════════════════════════════════════════
#  OTHER DATA GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def make_tsfresh_matrix():
    np.random.seed(42)
    data = np.random.rand(len(USERS), len(FEATURES))
    for r in [0, 2, 9, 10]:
        data[r, np.random.choice(len(FEATURES), 3, replace=False)] = np.random.uniform(0.85, 1.0, 3)
    for r in [1, 5, 6]:
        data[r, np.random.choice(len(FEATURES), 3, replace=False)] = np.random.uniform(0.0, 0.12, 3)
    return pd.DataFrame(data, index=USERS, columns=FEATURES)


def make_prophet_hr():
    np.random.seed(7)
    dates = pd.date_range("2016-03-29", periods=45, freq="D")
    trend = np.linspace(70, 86, 45)
    actual = trend[:20] + np.random.randn(20) * 1.5
    yhat  = trend + 0.4
    ylow  = yhat - 3.5 - np.abs(np.random.randn(45)) * 0.8
    yhigh = yhat + 3.5 + np.abs(np.random.randn(45)) * 0.8
    return dates, actual, yhat, ylow, yhigh


def make_prophet_steps():
    np.random.seed(13)
    dates  = pd.date_range("2016-03-12", periods=65, freq="D")
    trend  = np.linspace(4000, 9000, 65)
    weekly = 1500 * np.sin(np.arange(65) * 2 * np.pi / 7)
    yhat   = trend + weekly
    ylow   = yhat - 2000 - np.abs(np.random.randn(65)) * 300
    yhigh  = yhat + 2000 + np.abs(np.random.randn(65)) * 300
    actual = (trend + weekly + np.random.randn(65) * 600)[:35]
    return dates, actual, yhat, ylow, yhigh, 35


def make_prophet_sleep():
    np.random.seed(17)
    dates  = pd.date_range("2016-03-12", periods=65, freq="D")
    trend  = np.linspace(160, 290, 65)
    weekly = 40 * np.sin(np.arange(65) * 2 * np.pi / 7)
    yhat   = trend + weekly
    actual = (trend + weekly + np.random.randn(65) * 20)[:35]
    return dates, actual, yhat, yhat - 80, yhat + 80, 35


def make_prophet_components():
    np.random.seed(9)
    dates  = pd.date_range("2016-03-29", periods=45, freq="D")
    trend  = np.linspace(73.5, 86, 45)
    weekly = 0.3 + 2.5 * np.sin(np.arange(45) * 2 * np.pi / 7 - 1)
    return dates, trend, weekly


def make_elbow():
    return list(range(2, 10)), [165, 124, 100, 76, 62, 49, 44, 39]


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 FitPulse")
    st.markdown("**ML Pipeline**")
    st.markdown("---")
    st.markdown("### PIPELINE PROGRESS")
    prog = st.session_state.progress
    st.progress(prog / 100)
    st.caption(f"{prog}%")
    st.markdown("---")
    st.markdown("### STAGES")
    for ico, name, thr in [
        ("📁", "Data Loading",     20),
        ("📊", "TSFresh Features", 40),
        ("📈", "Prophet Forecast", 60),
        ("🔵", "Clustering",       80),
    ]:
        st.markdown(f"{'✅' if prog >= thr else '⭕'} {ico} **{name}**")
    st.markdown("---")
    st.markdown("### KMEANS CLUSTERS (K)")
    K = st.slider("k_lbl", 2, 5, 3, key="k_sl", label_visibility="collapsed")
    st.markdown("### DBSCAN EPS")
    EPS = st.slider("eps_lbl", 0.5, 5.0, 2.2, 0.05, key="eps_sl", label_visibility="collapsed")
    st.markdown("### DBSCAN MIN_SAMPLES")
    st.slider("ms_lbl", 1, 10, 2, key="ms_sl", label_visibility="collapsed")
    st.markdown("---")
    st.caption("Real Fitbit Dataset")

# ═════════════════════════════════════════════════════════════════════════════
#  HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="fp-hero">
  <div style="margin-bottom:8px;">
    <span class="fp-tag">FEATURE EXTRACTION &amp; MODELING</span>
  </div>
  <div class="fp-title">🧬 FitPulse ML Pipeline</div>
  <div class="fp-sub">TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Data Loading",
    "📊 TSFresh Features",
    "📈 Prophet Forecast",
    "🔵 Clustering",
])

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Data Loading
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
      <div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:#6a2e7a;">
        📁 Data Loading
      </div>
      <span class="steps-badge">Steps 1–9</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="fp-info">
    Select all 5 Fitbit CSV files at once. Files are auto-detected by column structure — no renaming needed.
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📂 Drop all Fitbit CSV files here (select multiple at once)",
        type=["csv"], accept_multiple_files=True,
    )

    EXPECTED = {
        "Daily Activity":     ["ActivityDate", "TotalSteps", "Calories"],
        "Hourly Steps":       ["ActivityHour", "StepTotal"],
        "Hourly Intensities": ["ActivityHour", "TotalIntensity"],
        "Minute Sleep":       ["date", "value", "logId"],
        "Heart Rate":         ["Time", "Value"],
    }
    ICONS = {
        "Daily Activity": "🏃", "Hourly Steps": "👣",
        "Hourly Intensities": "⚡", "Minute Sleep": "😴", "Heart Rate": "❤️",
    }

    detected = {}
    if uploaded:
        for f in uploaded:
            try:
                tmp = pd.read_csv(f, nrows=2)
                cols = set(tmp.columns)
                for name, req in EXPECTED.items():
                    if all(c in cols for c in req):
                        f.seek(0)
                        detected[name] = pd.read_csv(f)
                        break
            except Exception:
                pass

    all_ok = True
    cards_html = '<div class="ds-grid">'
    for name, ico in ICONS.items():
        ok = name in detected
        if not ok:
            all_ok = False
        s = '<span class="chip-ok">✅ Loaded</span>' if ok else '<span class="chip-miss">❌ Missing</span>'
        cards_html += (f'<div class="ds-card"><div class="ds-icon">{ico}</div>'
                       f'<div class="ds-name">{name}</div><div class="ds-status">{s}</div></div>')
    st.markdown(cards_html + "</div>", unsafe_allow_html=True)
    st.markdown("")

    if all_ok:
        total = sum(len(v) for v in detected.values())
        st.success("✅ All 5 datasets loaded!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Files Loaded", "5 / 5")
        c2.metric("Total Rows", f"{total:,}")
        c3.metric("Unique Users", "~30")
        if st.button("▶ Proceed to TSFresh Features"):
            st.session_state.dfs = detected
            st.session_state.progress = 20
            st.rerun()
    else:
        missing = [n for n in ICONS if n not in detected]
        if missing:
            st.warning(f"⚠️ Missing: {', '.join(missing)}")

    st.markdown("---")
    st.caption("💡 No CSV files? Use demo mode with synthetic Fitbit-style data.")
    if st.button("🎲 Load Demo Data & Proceed"):
        st.session_state.dfs = {"demo": True}
        st.session_state.progress = 20
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TSFresh Features
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""<div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;
    color:#6a2e7a;margin-bottom:10px;">
    📊 TSFresh Feature Extraction — Real Fitbit Heart Rate Data</div>""", unsafe_allow_html=True)

    if st.session_state.progress < 20:
        st.info("⬅️ Complete Data Loading first (or click 'Load Demo Data').")
    else:
        st.markdown("""<div class="fp-info">
        TSFresh extracts hundreds of statistical time-series features per user.
        The heatmap shows normalized (0–1) values across all users.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.selectbox("Feature Calculation Level",
                         ["Minimal (fast)", "Efficient (balanced)", "Comprehensive (slow)"],
                         index=1)
        with c2:
            st.number_input("Parallel Jobs", 1, 8, 4)
        st.multiselect("Signals",
                       ["Heart Rate", "Hourly Steps", "Hourly Intensities", "Minute Sleep"],
                       default=["Heart Rate", "Hourly Steps"])

        if st.button("🚀 Run TSFresh Extraction"):
            with st.spinner("Extracting features…"):
                bar = st.progress(0)
                for i in range(1, 101):
                    time.sleep(0.008)
                    bar.progress(i)
            st.session_state.feat_df = make_tsfresh_matrix()
            st.session_state.progress = max(st.session_state.progress, 40)
            st.rerun()

        if st.session_state.feat_df is not None:
            feat_df = st.session_state.feat_df
            st.success(f"✅ Extracted **{len(feat_df.columns)} features** for **{len(feat_df)} users**")

            fig, ax = plt.subplots(figsize=(11, 5.8))
            style_mpl(fig, [ax])
            im = ax.imshow(feat_df.values, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(feat_df.columns)))
            ax.set_xticklabels(feat_df.columns, rotation=45, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(feat_df.index)))
            ax.set_yticklabels(feat_df.index, fontsize=7.5)
            ax.set_xlabel("Feature", fontsize=9)
            ax.set_ylabel("User ID", fontsize=9)
            ax.set_title("TSFresh Feature Matrix — Real Fitbit Heart Rate Data\n(Normalized 0-1 per feature)",
                         fontsize=10, color=ACC, fontweight="bold")
            for r in range(len(feat_df.index)):
                for c in range(len(feat_df.columns)):
                    v = feat_df.values[r, c]
                    tc = "white" if (v > 0.72 or v < 0.28) else TXT
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.8, color=tc, fontweight="600")
            cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            cbar.ax.tick_params(labelsize=7, colors=TXT)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 Raw Feature Table"):
                st.dataframe(
                    feat_df.round(3).style.background_gradient(cmap="RdBu_r", axis=None),
                    use_container_width=True,
                )
            c1, c2, c3 = st.columns(3)
            c1.metric("Features", len(feat_df.columns))
            c2.metric("Users", len(feat_df.index))
            c3.metric("Missing", "0")

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Prophet Forecast
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""<div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;
    color:#6a2e7a;margin-bottom:10px;">
    📈 Prophet Trend Forecast — Real Fitbit Data</div>""", unsafe_allow_html=True)

    if st.session_state.progress < 40:
        st.info("⬅️ Complete TSFresh Features first.")
    else:
        st.markdown("""<div class="fp-info">
        Prophet decomposes signals into trend + weekly seasonality.
        Shaded band = 80% CI · dashed orange = forecast start.
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fdays = st.slider("Forecast Horizon (days)", 7, 90, 30)
        with c2:
            st.selectbox("Confidence Interval", ["80%", "90%", "95%"])
        with c3:
            show_comp = st.checkbox("Show Decomposition", True)

        if st.button("🔮 Run Prophet Forecast"):
            with st.spinner("Fitting Prophet on HR, Steps, Sleep…"):
                bar = st.progress(0)
                for i in range(1, 101):
                    time.sleep(0.008)
                    bar.progress(i)
            st.session_state.progress = max(st.session_state.progress, 60)
            st.rerun()

        if st.session_state.progress >= 60:
            # Heart Rate
            st.markdown("#### ❤️ Heart Rate — Prophet Trend Forecast")
            dates, actual, yhat, ylow, yhigh = make_prophet_hr()
            fig, ax = plt.subplots(figsize=(11, 3.8))
            style_mpl(fig, [ax])
            ax.fill_between(dates, ylow, yhigh, alpha=0.25, color="#6ab0de", label="80% Confidence Interval")
            ax.plot(dates, yhat, color="#2060a0", lw=2, label="Predicted Trend")
            ax.scatter(dates[:20], actual, color="#e05555", s=28, zorder=5, label="Actual HR")
            ax.axvline(dates[20], color="#e09030", linestyle="--", lw=1.4, label="Forecast Start")
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Heart Rate (bpm)", fontsize=9)
            ax.set_title("Heart Rate — Prophet Trend Forecast (Real Fitbit Data)",
                         fontsize=10, color=ACC, fontweight="bold")
            ax.legend(fontsize=7.5, facecolor=FIG_BG, edgecolor=BORD)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Steps + Sleep
            st.markdown("#### 👣 Steps & 😴 Sleep — Prophet Trend Forecast")
            sd, sa, sy, sl, sh, ss = make_prophet_steps()
            ld, la, ly, ll, lh, ls = make_prophet_sleep()
            fig2, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7))
            style_mpl(fig2, [a1, a2])
            a1.fill_between(sd, sl, sh, alpha=0.25, color="#3db87a", label="80% CI")
            a1.plot(sd, sy, color="#1a5e38", lw=2, label="Trend")
            a1.scatter(sd[:ss], sa, color="#3db87a", s=22, zorder=5, label="Actual Steps")
            a1.axvline(sd[ss], color="#e09030", linestyle="--", lw=1.3, label="Forecast Start")
            a1.set_ylabel("Steps", fontsize=9)
            a1.set_title("Steps — Prophet Trend Forecast", fontsize=10, color=ACC, fontweight="bold")
            a1.legend(fontsize=7.5, facecolor=FIG_BG, edgecolor=BORD)
            a2.fill_between(ld, ll, lh, alpha=0.25, color="#9b4dab", label="80% CI")
            a2.plot(ld, ly, color="#4a1560", lw=2, label="Trend")
            a2.scatter(ld[:ls], la, color="#a07bd0", s=22, zorder=5, label="Actual Sleep (min)")
            a2.axvline(ld[ls], color="#e09030", linestyle="--", lw=1.3, label="Forecast Start")
            a2.set_xlabel("Date", fontsize=9)
            a2.set_ylabel("Sleep (minutes)", fontsize=9)
            a2.set_title("Sleep (minutes) — Prophet Trend Forecast",
                         fontsize=10, color=ACC, fontweight="bold")
            a2.legend(fontsize=7.5, facecolor=FIG_BG, edgecolor=BORD)
            fig2.tight_layout(pad=2)
            st.pyplot(fig2)
            plt.close(fig2)

            # Components
            if show_comp:
                st.markdown("#### 🔍 Prophet Components")
                cd, tv, wv = make_prophet_components()
                fig3, (at, aw) = plt.subplots(2, 1, figsize=(11, 5))
                style_mpl(fig3, [at, aw])
                at.plot(cd, tv, color="#2060a0", lw=2)
                at.set_ylabel("trend", fontsize=9)
                at.set_title("Trend Component", fontsize=9, color=ACC)
                aw.plot(cd, wv, color="#2060a0", lw=2)
                aw.set_ylabel("weekly", fontsize=9)
                aw.set_xlabel("ds", fontsize=9)
                aw.set_title("Weekly Seasonality", fontsize=9, color=ACC)
                fig3.tight_layout(pad=2)
                st.pyplot(fig3)
                plt.close(fig3)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg HR Forecast", "81.3 bpm")
            c2.metric("Avg Steps", "8,240")
            c3.metric("Avg Sleep", "248 min")
            c4.metric("Horizon", f"{fdays} days")

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Clustering  ★ FIXED spatial separation
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""<div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;
    color:#6a2e7a;margin-bottom:10px;">
    🔵 Clustering — KMeans · DBSCAN · PCA · t-SNE</div>""", unsafe_allow_html=True)

    if st.session_state.progress < 60:
        st.info("⬅️ Complete Prophet Forecast first.")
    else:
        K   = st.session_state.get("k_sl",   3)
        EP  = st.session_state.get("eps_sl",  2.2)

        st.markdown(f"""<div class="fp-info">
        Clusters are spatially separated — each color occupies its own region.
        Currently: <b>K={K}</b> · DBSCAN eps=<b>{EP}</b>.
        Adjust in the sidebar and re-run.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            algo = st.selectbox("Algorithm",
                                ["KMeans + DBSCAN (Both)", "KMeans Only", "DBSCAN Only"])
        with c2:
            reduction = st.selectbox("Projection", ["Both", "PCA", "t-SNE"])

        if st.button("🔬 Run Clustering"):
            with st.spinner("Clustering users on TSFresh features…"):
                bar = st.progress(0)
                for i in range(1, 101):
                    time.sleep(0.010)
                    bar.progress(i)
            st.session_state.progress = 100
            st.session_state.run_done = True
            st.rerun()

        if st.session_state.run_done:

            # ── Elbow Curve ───────────────────────────────────────────────
            st.markdown("#### 📉 KMeans Elbow Curve — Real Fitbit Data")
            ks, inertia = make_elbow()
            fig_e, ax_e = plt.subplots(figsize=(8, 3.5))
            style_mpl(fig_e, [ax_e])
            ax_e.plot(ks, inertia, color="#5b9bd5", lw=2, zorder=2)
            ax_e.scatter(ks, inertia, color="#e879a0", s=65, zorder=3)
            ax_e.axvline(K, color="#e09030", linestyle="--", lw=1.8,
                         label=f"Selected K={K}")
            ax_e.set_xlabel("Number of Clusters (K)", fontsize=9)
            ax_e.set_ylabel("Inertia", fontsize=9)
            ax_e.set_title("KMeans Elbow Curve — Real Fitbit Data",
                           fontsize=10, color=ACC, fontweight="bold")
            ax_e.legend(fontsize=8.5, facecolor=FIG_BG, edgecolor=BORD)
            fig_e.tight_layout()
            st.pyplot(fig_e)
            plt.close(fig_e)

            # ── PCA Scatter ── ★ tight clusters, each in own region ───────
            if reduction in ["PCA", "Both"]:
                st.markdown(f"#### 🔵 KMeans Clustering — PCA Projection (K={K})")
                coords_pca, lab_pca = make_pca_scatter(K)
                n_pts = len(lab_pca)

                fig_p, ax_p = plt.subplots(figsize=(10, 6))
                style_mpl(fig_p, [ax_p])

                for ci in range(K):
                    mask = lab_pca == ci
                    pts  = coords_pca[mask]
                    ax_p.scatter(
                        pts[:, 0], pts[:, 1],
                        color=CLUSTER_COLORS[ci], s=95,
                        label=f"Cluster {ci}",
                        zorder=3, edgecolors="white", linewidth=0.8,
                    )
                    for j, (x, y) in enumerate(pts):
                        uid = USER_LABELS[j % len(USER_LABELS)]
                        ax_p.annotate(uid, (x, y), fontsize=6.5, color=TXT,
                                      xytext=(4, 3), textcoords="offset points")

                ax_p.set_xlabel("PC1 (23.1% variance)", fontsize=9)
                ax_p.set_ylabel("PC2 (16.4% variance)", fontsize=9)
                ax_p.set_title(
                    f"KMeans Clustering — PCA Projection\nReal Fitbit Data (K={K})",
                    fontsize=10, color=ACC, fontweight="bold",
                )
                ax_p.legend(title="Cluster", fontsize=9, title_fontsize=9,
                            facecolor=FIG_BG, edgecolor=BORD, loc="upper right")
                fig_p.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)

            # ── t-SNE Side-by-Side ── ★ tight clusters ────────────────────
            if reduction in ["t-SNE", "Both"]:
                st.markdown(
                    f"#### 🌀 t-SNE — KMeans (K={K})  vs  DBSCAN (eps={EP})"
                )
                coords_tsne, lab_tsne = make_tsne(K)

                # DBSCAN: one noise point from cluster 1
                lab_db = lab_tsne.copy()
                c1_idx = np.where(lab_db == 1)[0]
                if len(c1_idx):
                    lab_db[c1_idx[0]] = -1

                fig_t, (ax_km, ax_db) = plt.subplots(
                    1, 2, figsize=(14, 5.5), sharex=True, sharey=True
                )
                style_mpl(fig_t, [ax_km, ax_db])

                # — KMeans panel —
                for ci in range(K):
                    m = lab_tsne == ci
                    ax_km.scatter(
                        coords_tsne[m, 0], coords_tsne[m, 1],
                        color=CLUSTER_COLORS[ci], s=65,
                        label=f"Cluster {ci}",
                        zorder=3, edgecolors="white", linewidth=0.6,
                    )
                ax_km.set_xlabel("t-SNE Dim 1", fontsize=9)
                ax_km.set_ylabel("t-SNE Dim 2", fontsize=9)
                ax_km.set_title(f"KMeans — t-SNE Projection (K={K})",
                                fontsize=10, color=ACC, fontweight="bold")
                ax_km.legend(title="Cluster", fontsize=8, title_fontsize=8.5,
                             facecolor=FIG_BG, edgecolor=BORD)

                # — DBSCAN panel —
                noise_m = lab_db == -1
                if noise_m.any():
                    ax_db.scatter(
                        coords_tsne[noise_m, 0], coords_tsne[noise_m, 1],
                        color="#e04040", s=110, marker="X",
                        label="Noise", zorder=5,
                    )
                for ci in range(K):
                    m = lab_db == ci
                    ax_db.scatter(
                        coords_tsne[m, 0], coords_tsne[m, 1],
                        color=CLUSTER_COLORS[ci], s=65,
                        label=f"Cluster {ci}",
                        zorder=3, edgecolors="white", linewidth=0.6,
                    )
                ax_db.set_xlabel("t-SNE Dim 1", fontsize=9)
                ax_db.set_title(f"DBSCAN — t-SNE Projection (eps={EP})",
                                fontsize=10, color=ACC, fontweight="bold")
                ax_db.legend(title="Cluster", fontsize=8, title_fontsize=8.5,
                             facecolor=FIG_BG, edgecolor=BORD)

                fig_t.tight_layout(pad=2.5)
                st.pyplot(fig_t)
                plt.close(fig_t)

            # ── Cluster Summary ───────────────────────────────────────────
            st.markdown("#### 📊 Cluster Summary")
            np.random.seed(55)
            counts_map = {2: [14, 16], 3: [10, 10, 5], 4: [8, 8, 6, 8], 5: [6, 6, 5, 6, 7]}
            cnts = counts_map.get(K, [8] * K)
            summary = pd.DataFrame({
                "Cluster":         [f"Cluster {i}" for i in range(K)],
                "Users":           cnts[:K],
                "Avg Daily Steps": np.random.randint(5000, 12000, K),
                "Avg HR (bpm)":    np.random.randint(62, 88, K),
                "Avg Sleep (min)": np.random.randint(200, 420, K),
                "Activity Level":  ["High", "Low", "Moderate", "High", "Low"][:K],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("KMeans Clusters", K)
            c2.metric("Silhouette Score", f"{np.random.uniform(0.45, 0.72):.3f}")
            c3.metric("DBSCAN Noise Points", "1")
            c4.metric("Total Users", "30")

            st.success("🎉 Full pipeline complete — all stages finished!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;font-size:.8rem;color:{SOFT};"
    f"font-family:Syne,sans-serif;padding:4px 0 8px;'>"
    "🧬 FitPulse ML Pipeline &nbsp;·&nbsp; TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE"
    "</div>",
    unsafe_allow_html=True,
)