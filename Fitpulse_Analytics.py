import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FitPulse Analytics",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — init all keys upfront so reruns never wipe EDA
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "df_raw":    None,
    "df_clean":  None,
    "null_done": False,
    "prep_done": False,
    "eda_done":  False,
    "file_name": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE  — base color #D8BFD8 (thistle) with purple accents
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#D8BFD8"   # thistle — app background
CARD   = "#EDE0ED"   # lighter thistle — cards & panels
FIG_BG = "#F5ECF5"   # near-white purple — chart backgrounds
BORD   = "#C4A0C4"   # medium thistle — borders
ACC    = "#6B2D8B"   # deep purple — primary accent (buttons, step num)
ACC2   = "#9B59B6"   # medium purple — secondary accent
TXT    = "#2D1B3D"   # very dark purple — body text
SOFT   = "#7B5C8A"   # muted purple — labels & subtitles
SB_BG  = "#1A0A22"   # near-black purple — sidebar background
SB_BDR = "#3D1F52"   # dark purple — sidebar border
SB_FUP = "#251035"   # sidebar file-uploader bg

# Chart palette: purples first, then readable distinct colors
PAL = [ACC, ACC2, "#2ECC71", "#1ABC9C", "#E74C3C", "#F39C12", "#3498DB", "#E91E63"]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — full #D8BFD8 thistle palette
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

/* ── KEYFRAMES ── */
@keyframes fadeUp   {{ from{{opacity:0;transform:translateY(14px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes slideIn  {{ from{{opacity:0;transform:translateX(-14px)}} to{{opacity:1;transform:translateX(0)}} }}
@keyframes rotateBdr{{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
@keyframes pulseDot {{ 0%,100%{{transform:scale(1);opacity:1}} 50%{{transform:scale(1.7);opacity:0.4}} }}
@keyframes gradFlow {{ 0%{{background-position:0% 50%}} 50%{{background-position:100% 50%}} 100%{{background-position:0% 50%}} }}
@keyframes shimmer  {{ from{{transform:translateX(-100%)}} to{{transform:translateX(250%)}} }}
@keyframes popIn    {{ from{{opacity:0;transform:scale(0.9)}} to{{opacity:1;transform:scale(1)}} }}
@keyframes countUp  {{ from{{opacity:0;transform:translateY(6px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes scanLine {{
  0%  {{left:-4px;opacity:0}} 5%{{opacity:1}} 95%{{opacity:1}} 100%{{left:100%;opacity:0}}
}}

/* ── BASE ── */
html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; }}
.stApp {{ background:{BG}; color:{TXT}; }}
.block-container {{ padding:1.4rem 2rem 3rem !important; max-width:1380px !important; }}

/* ── SIDEBAR COLLAPSE ARROW — hide >> so full-screen is truly clean ── */
[data-testid="collapsedControl"] {{ display:none !important; }}
button[kind="header"] {{ display:none !important; }}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{ display:flex !important; }}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
  background:{SB_BG} !important;
  border-right:1px solid {SB_BDR} !important;
}}
[data-testid="stSidebar"] * {{ color:#E8D4F0 !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
  border:2px dashed rgba(155,89,182,0.55) !important;
  background:{SB_FUP} !important;
  border-radius:10px !important;
  transition:border-color 0.25s;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {{
  border-color:{ACC2} !important;
}}

/* ── BANNER ── */
.banner {{
  background:linear-gradient(135deg,#1A0A22 0%,#2D1040 55%,#1A0A22 100%);
  border-radius:14px; padding:1.5rem 2rem; margin-bottom:1.5rem;
  display:flex; align-items:center; justify-content:space-between;
  position:relative; overflow:hidden;
  animation:fadeUp 0.4s ease;
  box-shadow:0 4px 22px rgba(107,45,139,0.35);
}}
.banner::before {{
  content:''; position:absolute; left:0; top:0; bottom:0; width:5px;
  background:linear-gradient(180deg,{ACC},{ACC2});
}}
.banner-glow {{
  position:absolute; right:-40px; top:-40px; width:200px; height:200px;
  background:radial-gradient(circle,rgba(155,89,182,0.25),transparent 65%);
  pointer-events:none;
}}
.banner-title {{ font-family:'Syne',sans-serif; font-size:1.65rem; font-weight:800; color:#F0E0F8; letter-spacing:-0.5px; }}
.banner-sub   {{ font-size:0.78rem; color:#B89ABE; margin-top:4px; }}
.banner-chip  {{
  background:rgba(155,89,182,0.2); border:1px solid rgba(155,89,182,0.5); border-radius:20px;
  padding:6px 16px; font-size:0.72rem; font-weight:700; color:{ACC2}; letter-spacing:0.6px; z-index:1;
}}

/* ── STEP HEADER ── */
.step-hdr {{
  display:flex; align-items:center; gap:0.75rem;
  background:{CARD};
  border:1px solid {BORD};
  border-left:5px solid {ACC};
  border-radius:10px;
  padding:0.78rem 1.3rem;
  margin:1.4rem 0 1rem;
  font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:{TXT};
  box-shadow:0 1px 6px rgba(107,45,139,0.12);
  animation:fadeUp 0.3s ease;
}}
.step-num {{
  background:{ACC}; color:#fff; font-size:0.68rem; font-weight:700;
  width:24px; height:24px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; flex-shrink:0;
}}

/* ── STAT CARDS ── */
.scard {{
  background:{CARD}; border:1px solid {BORD}; border-radius:12px;
  padding:1.2rem 1.3rem; position:relative; overflow:hidden;
  box-shadow:0 1px 5px rgba(107,45,139,0.1);
  animation:fadeUp 0.4s ease both;
  transition:transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}}
.scard:hover {{ transform:translateY(-3px); box-shadow:0 6px 22px rgba(107,45,139,0.2); border-color:{ACC}; }}
.scard::after {{ content:''; position:absolute; top:0; right:0; width:55px; height:55px;
  background:radial-gradient(circle at top right,rgba(155,89,182,0.12),transparent 70%); }}
.scard-icon {{ font-size:1.3rem; margin-bottom:0.4rem; }}
.scard-val  {{ font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; color:{TXT}; animation:countUp 0.5s ease; }}
.scard-val.orange {{ color:{ACC}; }}
.scard-val.red    {{ color:#C0392B; }}
.scard-val.green  {{ color:#1E8449; }}
.scard-lbl  {{ font-size:0.68rem; color:{SOFT}; text-transform:uppercase; letter-spacing:0.08em; margin-top:2px; }}

/* ── LOADER ── */
.loader {{
  background:{CARD}; border:1px solid {BORD}; border-radius:12px;
  padding:1.25rem 1.5rem; margin:0.6rem 0;
  position:relative; overflow:hidden;
  box-shadow:0 1px 8px rgba(107,45,139,0.12);
}}
.loader::after {{
  content:''; position:absolute; top:0; bottom:0; width:3px;
  background:linear-gradient(180deg,transparent,{ACC},{ACC2},transparent);
  animation:scanLine 2.2s linear infinite;
}}
.loader-row  {{ display:flex; align-items:center; gap:1rem; margin-bottom:0.7rem; }}
.spinner     {{ width:40px; height:40px; border-radius:50%; border:2px solid {BORD};
  position:relative; flex-shrink:0; display:flex; align-items:center; justify-content:center; }}
.spinner::before {{ content:''; position:absolute; inset:0; border-radius:50%;
  border-top:2.5px solid {ACC}; border-right:2.5px solid transparent;
  animation:rotateBdr 0.75s linear infinite; }}
.spinner-dot {{ width:7px; height:7px; background:{ACC}; border-radius:50%;
  animation:pulseDot 0.75s ease-in-out infinite; }}
.ldr-title {{ font-size:0.82rem; font-weight:600; color:{TXT}; margin-bottom:1px; }}
.ldr-step  {{ font-size:0.71rem; color:{SOFT}; font-family:'DM Mono',monospace; }}
.ldr-pct   {{ font-family:'DM Mono',monospace; font-size:0.92rem; font-weight:600; color:{ACC}; margin-left:auto; flex-shrink:0; }}
.prog-track {{ background:{BG}; border-radius:6px; height:5px; overflow:hidden; margin:0.35rem 0; border:1px solid {BORD}; }}
.prog-bar   {{ height:100%;
  background:linear-gradient(90deg,{ACC},{ACC2},{ACC}); background-size:200% 100%;
  animation:gradFlow 1.4s linear infinite; border-radius:6px;
  box-shadow:0 0 8px rgba(155,89,182,0.45); transition:width 0.2s ease; }}
.ldr-ticker {{ font-family:'DM Mono',monospace; font-size:0.68rem; color:{ACC};
  background:{FIG_BG}; border:1px solid {BORD}; border-radius:5px;
  padding:4px 10px; margin-top:0.3rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

/* ── LOG LINES ── */
.logline {{
  background:{FIG_BG}; border:1px solid {BORD}; border-left:3px solid {ACC2};
  border-radius:6px; padding:0.48rem 0.9rem; margin-bottom:0.32rem;
  font-size:0.79rem; color:{TXT}; position:relative; overflow:hidden;
  animation:slideIn 0.3s ease both;
}}
.logline .sh {{ position:absolute; top:0; left:0; width:40%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(216,191,216,0.4),transparent);
  animation:shimmer 2.5s ease infinite; pointer-events:none; }}
.logline.ok   {{ border-left-color:#1E8449; background:#E8F8EE; color:#145A32; }}
.logline.warn {{ border-left-color:#D4AC0D; background:#FEF9E7; color:#7D6608; }}
.logline.info {{ border-left-color:{ACC2}; background:{FIG_BG}; color:{ACC}; }}
.logline.err  {{ border-left-color:#C0392B; background:#FDEDEC; color:#922B21; }}

/* ── PANELS ── */
.info-box {{
  background:{FIG_BG}; border:1px solid {BORD}; border-radius:9px;
  padding:0.8rem 1.1rem; font-size:0.8rem; color:{ACC}; margin:0.5rem 0; animation:popIn 0.3s ease;
}}
.ok-box {{
  background:{FIG_BG}; border:1px solid {BORD}; border-radius:9px;
  padding:0.8rem 1.1rem; font-size:0.81rem; color:{ACC}; font-weight:600;
  text-align:center; animation:popIn 0.35s ease;
}}

/* ── EDA SECTION CARD ── */
.eda-card {{
  background:{CARD}; border:1px solid {BORD}; border-radius:12px;
  padding:1.4rem 1.6rem; margin-bottom:1.2rem;
  box-shadow:0 1px 5px rgba(107,45,139,0.08);
  animation:fadeUp 0.35s ease;
}}
.eda-card-title {{
  font-family:'Syne',sans-serif; font-size:0.98rem; font-weight:700; color:{TXT};
  border-bottom:2px solid {BORD}; padding-bottom:0.55rem; margin-bottom:1rem;
}}

/* ── NULL BADGES ── */
.nbadge {{
  display:inline-block; background:{FIG_BG}; border:1px solid {BORD};
  border-radius:6px; padding:3px 10px; font-size:0.71rem; color:{ACC};
  margin:3px; font-weight:500; transition:all 0.2s;
}}
.nbadge:hover {{ background:{ACC}; color:#fff; border-color:{ACC}; transform:scale(1.03); }}

/* ── SIDEBAR NAV ── */
.sb-logo  {{ font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:800; color:#F0E0F8; }}
.sb-tag   {{ font-size:0.68rem; color:#9B7BAE; margin-top:2px; }}
.sb-label {{ font-size:0.66rem; color:#9B7BAE !important; text-transform:uppercase; letter-spacing:0.1em; margin:1rem 0 0.35rem; }}
.sb-nav   {{ display:flex; align-items:center; gap:0.5rem; padding:0.38rem 0.65rem; border-radius:7px;
  font-size:0.81rem; color:#C8A8D8 !important; margin-bottom:0.18rem; transition:all 0.2s; border:1px solid transparent; }}
.sb-nav:hover {{ background:{SB_BDR}; color:#F0E0F8 !important; border-color:#5D2D78; }}
.sb-nav.done  {{ color:{ACC2} !important; font-weight:600; }}
.sb-div   {{ height:1px; background:{SB_BDR}; margin:0.9rem 0; }}

/* ── DIVIDER ── */
.hr {{ height:1px; background:linear-gradient(90deg,transparent,{BORD},transparent); margin:1.5rem 0; }}

/* ── BUTTONS ── */
.stButton > button {{
  background:{ACC} !important; color:#fff !important; border:none !important;
  border-radius:8px !important; padding:0.48rem 1.35rem !important;
  font-weight:600 !important; font-size:0.84rem !important; font-family:'DM Sans',sans-serif !important;
  transition:background 0.18s,transform 0.18s,box-shadow 0.18s !important;
  box-shadow:0 2px 12px rgba(107,45,139,0.4) !important;
}}
.stButton > button:hover   {{ background:#521E6B !important; transform:translateY(-1px) !important; box-shadow:0 5px 18px rgba(107,45,139,0.55) !important; }}
.stButton > button:active  {{ transform:scale(0.97) !important; }}
.stButton > button:disabled{{ background:{BORD} !important; color:{SOFT} !important; box-shadow:none !important; }}

/* ── TABLES ── */
[data-testid="stDataFrame"] {{ border-radius:10px !important; border:1px solid {BORD} !important;
  overflow:hidden; box-shadow:0 1px 5px rgba(107,45,139,0.08); }}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] section {{ border:2px dashed {BORD} !important;
  border-radius:10px !important; background:{FIG_BG} !important; }}
[data-testid="stFileUploader"] section:hover {{ border-color:{ACC} !important; }}

/* ── MULTISELECT & SELECTBOX ── */
[data-baseweb="select"] > div {{
  background:{CARD} !important;
  border-color:{BORD} !important;
}}

/* ── EXPANDER ── */
details summary {{
  background:{CARD} !important;
  border:1px solid {BORD} !important;
  border-radius:8px !important;
  color:{TXT} !important;
}}

/* ── FOOTER ── */
.footer {{ text-align:center; font-size:0.7rem; color:{SOFT};
  padding:0.9rem 0 0.4rem; border-top:1px solid {BORD}; margin-top:1.8rem; }}
.footer b {{ color:{ACC}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hex_rgba(hx, a=0.18):
    h = hx.lstrip("#"); r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
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
      <div class="ldr-title">{title}</div>
      <div class="ldr-step">{s}</div>
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
        f'<div class="sh"></div>{icon}&nbsp; {msg}</div>',
        unsafe_allow_html=True)

def apply_theme(fig, h=340):
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=FIG_BG,
        font=dict(family="DM Sans", color=TXT, size=12),
        height=h, margin=dict(l=12,r=18,t=44,b=14),
        title_font=dict(family="Syne", size=13, color=TXT),
        legend=dict(bgcolor=CARD, bordercolor=BORD, borderwidth=1,
                    font=dict(color=TXT, size=11)),
    )
    fig.update_xaxes(gridcolor=BORD, gridwidth=1, zeroline=False,
                     linecolor=BORD, tickfont=dict(color=SOFT, size=11),
                     title_font=dict(color=SOFT))
    fig.update_yaxes(gridcolor=BORD, gridwidth=1, zeroline=False,
                     linecolor=BORD, tickfont=dict(color=SOFT, size=11),
                     title_font=dict(color=SOFT))
    return fig

def scard(col, val, lbl, color="", icon=""):
    cls = {"orange":"orange","red":"red","green":"green"}.get(color,"")
    with col:
        st.markdown(
            f'<div class="scard"><div class="scard-icon">{icon}</div>'
            f'<div class="scard-val {cls}">{val}</div>'
            f'<div class="scard-lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

def step_hdr(num, icon, title):
    st.markdown(
        f'<div class="step-hdr"><div class="step-num">{num}</div>'
        f'<span>{icon}&nbsp; {title}</span></div>',
        unsafe_allow_html=True)

def hr():
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">💪 FitPulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Upload Dataset</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV here", type=["csv"],
                                 label_visibility="collapsed", key="file_up")

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Pipeline</div>', unsafe_allow_html=True)
    for n, ic, lb, done in [
        ("1","📁","Upload CSV",       st.session_state.df_raw   is not None),
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

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.65rem;color:#9B7BAE;text-align:center;line-height:1.7;">'
        'FitPulse Analytics · v3.0<br>Thistle Purple Edition</div>',
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <div class="banner-glow"></div>
  <div>
    <div class="banner-title">💪 FitPulse Analytics</div>
    <div class="banner-sub">Fitness &amp; Health Data Quality Pipeline &nbsp;—&nbsp; Upload · Inspect · Clean · Analyse</div>
  </div>
  <div class="banner-chip">ENTERPRISE DATA PLATFORM</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILE LOAD — only reset state on NEW file
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is not None:
    new_file = (uploaded.name != st.session_state.file_name)
    if new_file:
        run_loader("Loading dataset…", [
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
            logline("ok", f"<strong>{uploaded.name}</strong> loaded — {len(df_loaded):,} rows × {len(df_loaded.columns)} columns")
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
step_hdr("1","📁","Upload CSV")

if st.session_state.df_raw is None:
    st.markdown(
        '<div class="info-box">📌 Use the <strong>sidebar uploader</strong> '
        'on the left to upload your Fitness Health CSV file.</div>',
        unsafe_allow_html=True)
else:
    df = st.session_state.df_raw
    c1,c2,c3,c4 = st.columns(4)
    scard(c1, f"{len(df):,}",                      "Total Rows",      "",       "📋")
    scard(c2, str(len(df.columns)),                 "Total Columns",   "",       "📊")
    scard(c3, f"{int(df.isnull().sum().sum()):,}",  "Null Cells",      "orange", "⚠️")
    scard(c4, f"{df.isnull().any(axis=1).sum():,}", "Rows with Nulls", "red",    "🔴")
    with st.expander("👀 Raw Data Preview — first 50 rows"):
        st.dataframe(df.head(50), use_container_width=True, height=260)

hr()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — NULL CHECK
# ═════════════════════════════════════════════════════════════════════════════
step_hdr("2","🔍","Null Value Inspector")

if st.button("🔍  Run Null Value Check",
             disabled=(st.session_state.df_raw is None), key="btn_null"):
    run_loader("Scanning for null values…",[
        "🧬 Mapping column schema…","🔎 Iterating all cells…",
        "📡 Flagging missing entries…","🧮 Computing statistics…",
        "📊 Rendering visual report…","✅ Null scan complete!",
    ], delay=0.2)
    st.session_state.null_done = True

if st.session_state.null_done and st.session_state.df_raw is not None:
    df = st.session_state.df_raw
    null_s     = df.isnull().sum()
    has_null   = null_s[null_s > 0].sort_values(ascending=False)
    clean_cols = null_s[null_s == 0]
    total_cells= df.shape[0] * df.shape[1]
    total_nulls= int(null_s.sum())

    c1,c2,c3,c4 = st.columns(4)
    scard(c1, f"{total_nulls:,}",                    "Total Null Cells",  "orange","❌")
    scard(c2, str(len(has_null)),                    "Affected Columns",  "red",   "📉")
    scard(c3, str(len(clean_cols)),                  "Clean Columns",     "green", "✅")
    scard(c4, f"{total_nulls/total_cells*100:.2f}%", "Overall Null Rate", "orange","📊")
    st.markdown("<br>", unsafe_allow_html=True)

    if has_null.empty:
        st.markdown('<div class="ok-box">🎉 No null values found — dataset is perfectly clean!</div>',
                    unsafe_allow_html=True)
    else:
        badges = "".join(
            f'<span class="nbadge">⚠️ {col}&nbsp;·&nbsp;{v:,} ({v/len(df)*100:.1f}%)</span>'
            for col, v in has_null.items())
        st.markdown(f'<div style="margin-bottom:1rem;line-height:2.3;">{badges}</div>',
                    unsafe_allow_html=True)

        col_chart, col_table = st.columns([3,2], gap="large")
        with col_chart:
            null_plot = pd.DataFrame({
                "Column": has_null.index.tolist(),
                "Nulls":  has_null.values.tolist(),
                "Null %": (has_null.values/len(df)*100).round(2).tolist(),
            })
            fig = px.bar(null_plot, x="Nulls", y="Column", orientation="h",
                         color="Null %",
                         color_continuous_scale=[[0,ACC2],[0.5,ACC],[1,"#C0392B"]],
                         text="Nulls", title="Null Count per Column")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                              textfont=dict(size=11,color=TXT), marker_line_width=0)
            fig.update_layout(yaxis=dict(autorange="reversed"),
                              coloraxis_colorbar=dict(title="Null %",tickfont=dict(size=10,color=TXT),len=0.7))
            fig = apply_theme(fig, 320)
            st.plotly_chart(fig, use_container_width=True, key="null_bar")

        with col_table:
            st.markdown("**📋 Detailed Null Breakdown**")
            detail = pd.DataFrame({
                "Column":     has_null.index,
                "Dtype":      df[has_null.index].dtypes.astype(str).values,
                "Null Count": has_null.values,
                "Non-Null":   len(df)-has_null.values,
                "Null %":     (has_null.values/len(df)*100).round(2),
            })
            st.dataframe(detail, hide_index=True, use_container_width=True, height=290)

        st.markdown("**🗺️ Null Heatmap** (purple = missing · light = present)")
        sample = df[has_null.index].head(500).isnull().astype(int)
        fig2 = px.imshow(sample.T,
                         color_continuous_scale=[[0,FIG_BG],[1,ACC]],
                         aspect="auto", title="Null Presence Map", zmin=0, zmax=1)
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(xaxis_title="Row index", yaxis_title="")
        fig2 = apply_theme(fig2, 250)
        st.plotly_chart(fig2, use_container_width=True, key="null_heatmap")

hr()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
step_hdr("3","⚙️","Data Preprocessing")

if st.button("⚙️  Run Preprocessing",
             disabled=(st.session_state.df_raw is None), key="btn_prep"):
    run_loader("Preprocessing pipeline…",[
        "🧪 Initialising engine…","📅 Detecting datetime columns…",
        "📈 Interpolating numeric nulls…","🔧 Filling categorical nulls…",
        "🛡️ Integrity validation…","🧹 Final residual clean…","✅ Complete!",
    ], delay=0.2)
    st.session_state.prep_done = True

if st.session_state.prep_done and st.session_state.df_raw is not None:
    df_work     = st.session_state.df_raw.copy()
    plogs       = []
    null_before = df_work.isnull().sum().copy()

    # A — datetime
    date_kw = ("date","time","day","month","year")
    dfound = [c for c in df_work.columns if any(k in c.lower() for k in date_kw)]
    for col in dfound:
        try:
            conv = pd.to_datetime(df_work[col], errors="coerce")
            df_work[col] = conv
            plogs.append(("ok",f"Parsed <strong>{col}</strong> → datetime ({int(conv.notna().sum()):,} valid)"))
        except Exception as ex:
            plogs.append(("warn",f"Could not parse <strong>{col}</strong>: {ex}"))
    if not dfound:
        plogs.append(("info","No datetime columns detected."))

    # B — numeric: interpolate → ffill → bfill → median
    any_num = False
    for col in df_work.select_dtypes(include=[np.number]).columns:
        nb = int(df_work[col].isnull().sum())
        if nb == 0: continue
        any_num = True
        med = df_work[col].median(); med = 0.0 if pd.isna(med) else med
        df_work[col] = (df_work[col]
                        .interpolate(method="linear", limit_direction="both")
                        .ffill().bfill().fillna(med))
        na = int(df_work[col].isnull().sum())
        plogs.append(("ok" if na==0 else "warn",
            f"Numeric <strong>{col}</strong> — {nb-na:,}/{nb:,} nulls filled "
            f"[interp→ffill→bfill→median={med:.2f}]"))
    if not any_num:
        plogs.append(("info","No numeric nulls found."))

    # C — categorical: mode fill
    any_cat = False
    for col in df_work.select_dtypes(include=["object","category"]).columns:
        nb = int(df_work[col].isnull().sum())
        if nb == 0: continue
        any_cat = True
        modes = df_work[col].mode()
        fv    = modes.iloc[0] if not modes.empty else "Unknown"
        df_work[col] = df_work[col].fillna(fv)
        na = int(df_work[col].isnull().sum())
        plogs.append(("ok" if na==0 else "warn",
            f"Categorical <strong>{col}</strong> — {nb-na:,}/{nb:,} nulls → mode='{fv}'"))
    if not any_cat:
        plogs.append(("info","No categorical nulls found."))

    # D — datetime nulls
    for col in df_work.select_dtypes(include=["datetime64[ns]"]).columns:
        nb = int(df_work[col].isnull().sum())
        if nb == 0: continue
        df_work[col] = df_work[col].ffill().bfill()
        na = int(df_work[col].isnull().sum())
        plogs.append(("ok" if na==0 else "warn",
            f"Datetime <strong>{col}</strong> — {nb-na:,}/{nb:,} nulls [ffill+bfill]"))

    # E — safety net
    for col in [c for c in df_work.columns if df_work[c].isnull().any()]:
        nb = int(df_work[col].isnull().sum())
        try:
            if df_work[col].dtype.kind in ("i","u","f"):
                fv = df_work[col].median(); fv = 0.0 if pd.isna(fv) else fv
                df_work[col] = df_work[col].fillna(fv)
            else:
                df_work[col] = df_work[col].ffill().bfill().fillna("Unknown")
        except Exception:
            df_work[col] = df_work[col].fillna("Unknown")
        na = int(df_work[col].isnull().sum())
        plogs.append(("ok" if na==0 else "err",
            f"Safety-net <strong>{col}</strong> — {nb-na:,} residual nulls removed"))

    remaining = int(df_work.isnull().sum().sum())
    cleaned   = int(null_before.sum()) - remaining
    plogs.append(("ok" if remaining==0 else "err",
        f"<strong>{'✅ COMPLETE' if remaining==0 else '⚠️ PARTIAL'} — "
        f"{cleaned:,} nulls removed · {remaining:,} remaining</strong>"))

    st.session_state.df_clean = df_work

    st.markdown("**📋 Preprocessing Log**")
    st.markdown(
        f'<div style="background:{FIG_BG};border:1px solid {BORD};'
        'border-radius:10px;padding:1rem 1.1rem;margin-bottom:1rem;">',
        unsafe_allow_html=True)
    for i,(k,m) in enumerate(plogs):
        logline(k, m, delay=i*0.05)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**🔄 Before vs After**")
    ca,cb = st.columns(2, gap="large")
    nb_f = null_before[null_before > 0]
    with ca:
        st.markdown("**Before**")
        if nb_f.empty:
            st.markdown('<div class="ok-box">No nulls before preprocessing.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(pd.DataFrame({"Column":nb_f.index,"Null Count":nb_f.values,
                "Null %":(nb_f.values/len(st.session_state.df_raw)*100).round(2)}),
                hide_index=True, use_container_width=True)
    with cb:
        st.markdown("**After**")
        na_a = df_work.isnull().sum(); na_a = na_a[na_a>0]
        if na_a.empty:
            st.markdown('<div class="ok-box">🎉 Zero nulls remaining!</div>', unsafe_allow_html=True)
        else:
            st.dataframe(pd.DataFrame({"Column":na_a.index,"Null Count":na_a.values}),
                hide_index=True, use_container_width=True)

hr()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — PREVIEW
# ═════════════════════════════════════════════════════════════════════════════
step_hdr("4","👁️","Preview Cleaned Dataset")

if st.session_state.df_clean is not None:
    dfc   = st.session_state.df_clean
    all_c = dfc.columns.tolist()
    logline("info",
        f"Shape: <strong>{dfc.shape[0]:,} rows × {dfc.shape[1]} cols</strong>"
        f" &nbsp;·&nbsp; Nulls remaining: <strong>{int(dfc.isnull().sum().sum()):,}</strong>")
    sel = st.multiselect("Select columns to display:", options=all_c,
                          default=all_c[:min(8,len(all_c))], key="prev_sel")
    if sel:
        st.dataframe(dfc[sel].head(200), use_container_width=True, height=340)
    else:
        st.info("Select at least one column.")
    with st.expander("📋 Column Schema — Dtypes & Null Summary"):
        schema = pd.DataFrame({
            "Column":   dfc.dtypes.index,
            "Dtype":    dfc.dtypes.astype(str).values,
            "Non-Null": dfc.notna().sum().values,
            "Null":     dfc.isna().sum().values,
            "Null %":   (dfc.isna().mean()*100).round(2).values,
            "Sample":   [str(dfc[c].dropna().iloc[0]) if dfc[c].notna().any() else "—" for c in dfc.columns],
        })
        st.dataframe(schema, hide_index=True, use_container_width=True)
else:
    st.markdown(
        '<div class="info-box">⚙️ Run <strong>Step 3 Preprocessing</strong> first.</div>',
        unsafe_allow_html=True)

hr()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — EDA
# eda_done persists in session_state so widgets don't reset charts
# ═════════════════════════════════════════════════════════════════════════════
step_hdr("5","📊","Exploratory Data Analysis")

if st.button("📊  Run Full EDA",
             disabled=(st.session_state.df_clean is None), key="btn_eda"):
    run_loader("Running EDA pipeline…",[
        "🧠 Initialising EDA engine…",
        "📐 Computing descriptive statistics…",
        "📈 Building distribution plots…",
        "🔥 Building correlation heatmap…",
        "🏷️ Analysing categorical columns…",
        "📦 Detecting outliers (IQR)…",
        "📅 Rendering time series…",
        "✅ EDA complete — all charts ready!",
    ], delay=0.22)
    st.session_state.eda_done = True

if st.session_state.eda_done and st.session_state.df_clean is not None:
    dfe      = st.session_state.df_clean.copy()
    num_cols = dfe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = dfe.select_dtypes(include=["object","category"]).columns.tolist()
    dt_cols  = dfe.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    st.markdown(f"""
    <div class="ok-box" style="margin-bottom:1.2rem;">
      ✅ &nbsp;<strong>{len(dfe):,} rows</strong> &nbsp;·&nbsp;
      <strong>{len(num_cols)} numeric</strong> &nbsp;·&nbsp;
      <strong>{len(cat_cols)} categorical</strong> &nbsp;·&nbsp;
      <strong>{len(dt_cols)} datetime</strong> &nbsp;·&nbsp;
      <strong>0 nulls</strong>
    </div>""", unsafe_allow_html=True)

    # ── 5a DESCRIPTIVE STATISTICS ─────────────────────────────────────────────
    st.markdown('<div class="eda-card">', unsafe_allow_html=True)
    st.markdown('<div class="eda-card-title">📋 Descriptive Statistics</div>', unsafe_allow_html=True)
    if num_cols:
        st.dataframe(dfe[num_cols].describe().round(3), use_container_width=True, height=295)
    else:
        st.warning("No numeric columns found.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 5b NUMERIC DISTRIBUTIONS ──────────────────────────────────────────────
    if num_cols:
        st.markdown('<div class="eda-card">', unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">📈 Numeric Distributions</div>', unsafe_allow_html=True)
        per_row = 3
        for cs in range(0, len(num_cols), per_row):
            chunk = num_cols[cs: cs+per_row]
            cols  = st.columns(len(chunk))
            for j, cn in enumerate(chunk):
                color = PAL[(cs+j) % len(PAL)]
                with cols[j]:
                    fig = px.histogram(dfe, x=cn, nbins=30,
                                       color_discrete_sequence=[color], title=cn)
                    fig.update_traces(marker_line_width=0.4, marker_line_color=FIG_BG, opacity=0.88)
                    fig = apply_theme(fig, 248)
                    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True, key=f"hist_{cn}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 5c CORRELATION HEATMAP ────────────────────────────────────────────────
    if len(num_cols) >= 2:
        st.markdown('<div class="eda-card">', unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = dfe[num_cols].corr(numeric_only=True).round(2)
        fig  = px.imshow(corr, text_auto=True, aspect="auto",
                         color_continuous_scale=[[0,"#2ECC71"],[0.5,CARD],[1,ACC]],
                         zmin=-1, zmax=1,
                         title="Pearson Correlation Matrix — All Numeric Features")
        fig.update_traces(textfont=dict(size=10, color=TXT),
                          hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.2f}<extra></extra>")
        fig = apply_theme(fig, max(400, len(num_cols)*62))
        fig.update_layout(coloraxis_colorbar=dict(title="r",tickfont=dict(size=10,color=TXT),len=0.8))
        st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 5d CATEGORICAL DISTRIBUTIONS ─────────────────────────────────────────
    if cat_cols:
        st.markdown('<div class="eda-card">', unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">🏷️ Categorical Distributions</div>', unsafe_allow_html=True)
        show_cats = cat_cols[:6]
        pairs     = [show_cats[i:i+2] for i in range(0, len(show_cats), 2)]
        for pair in pairs:
            cols = st.columns(len(pair))
            for gi, cn in enumerate(pair):
                with cols[gi]:
                    vc    = dfe[cn].value_counts().head(15)
                    if vc.empty: continue
                    base  = PAL[gi % len(PAL)]
                    r,g,b = [int(base.lstrip("#")[i:i+2],16) for i in (0,2,4)]
                    n_b   = len(vc)
                    clrs  = [f"rgba({r},{g},{b},{0.4+0.6*i/max(n_b-1,1):.2f})"
                             for i in range(n_b)]
                    fig   = go.Figure(go.Bar(
                        x=vc.values, y=vc.index.astype(str), orientation="h",
                        marker=dict(color=clrs, line=dict(width=0)),
                        text=vc.values, texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=11, color=TXT),
                        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>"))
                    fig = apply_theme(fig, max(220, n_b*34))
                    fig.update_layout(title=cn, title_font=dict(size=13),
                                      yaxis=dict(autorange="reversed"),
                                      xaxis_title="Count", yaxis_title="",
                                      margin=dict(l=10,r=55,t=44,b=12))
                    st.plotly_chart(fig, use_container_width=True, key=f"cat_{cn}_{gi}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 5e TIME SERIES ────────────────────────────────────────────────────────
    if dt_cols and num_cols:
        st.markdown('<div class="eda-card">', unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">📅 Time Series Trends</div>', unsafe_allow_html=True)
        dc    = dt_cols[0]
        sel_m = st.selectbox("Select metric to plot over time:", num_cols, key="ts_metric")
        ts_df = dfe[[dc, sel_m]].dropna().sort_values(dc)
        if not ts_df.empty:
            try:
                ts_w = ts_df.set_index(dc)[sel_m].resample("W").mean().dropna().reset_index()
                plot_df  = ts_w if not ts_w.empty else ts_df
                ts_label = f"{sel_m} — Weekly Average"
            except Exception:
                plot_df  = ts_df; ts_label = f"{sel_m} — Raw Values"
            fig = px.line(plot_df, x=dc, y=sel_m,
                          color_discrete_sequence=[ACC], title=ts_label, markers=True)
            fig.update_traces(line=dict(width=2.5), fill="tozeroy",
                              fillcolor=hex_rgba(ACC, 0.08),
                              marker=dict(size=4, color=ACC),
                              hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}<extra></extra>")
            fig = apply_theme(fig, 340)
            st.plotly_chart(fig, use_container_width=True, key="ts_line")
            # Monthly bar
            try:
                ts_m = ts_df.set_index(dc)[sel_m].resample("ME").mean().dropna().reset_index()
            except Exception:
                try:
                    ts_m = ts_df.set_index(dc)[sel_m].resample("M").mean().dropna().reset_index()
                except Exception:
                    ts_m = pd.DataFrame()
            if not ts_m.empty:
                fig2 = px.bar(ts_m, x=dc, y=sel_m,
                              color_discrete_sequence=[ACC2],
                              title=f"{sel_m} — Monthly Average")
                fig2.update_traces(marker_line_width=0, opacity=0.85)
                fig2 = apply_theme(fig2, 275)
                st.plotly_chart(fig2, use_container_width=True, key="ts_bar")
        else:
            st.info("Not enough data for time series.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif not dt_cols:
        st.markdown(
            '<div class="info-box">ℹ️ No datetime column detected. '
            "Rename your date column to include 'date', 'time', 'day', 'month' or 'year'.</div>",
            unsafe_allow_html=True)

    # ── 5f BOX PLOTS / OUTLIERS ───────────────────────────────────────────────
    if num_cols:
        st.markdown('<div class="eda-card">', unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">📦 Outlier Detection — Box Plots</div>', unsafe_allow_html=True)
        sel_box = st.multiselect("Select columns to inspect:", num_cols,
                                  default=num_cols[:min(5,len(num_cols))], key="box_sel")
        if sel_box:
            fig = go.Figure()
            for i, cn in enumerate(sel_box):
                clr  = PAL[i % len(PAL)]
                rgba = hex_rgba(clr, 0.2)
                fig.add_trace(go.Box(
                    y=dfe[cn].dropna(), name=cn,
                    marker_color=clr, line=dict(color=clr, width=1.5),
                    fillcolor=rgba, boxpoints="outliers", jitter=0.35,
                    marker=dict(size=4, opacity=0.65, color=clr),
                    hovertemplate=f"<b>{cn}</b><br>%{{y:.2f}}<extra></extra>"))
            fig = apply_theme(fig, 400)
            fig.update_layout(title="Box Plots — Distribution & Outlier View",
                              showlegend=True, plot_bgcolor=FIG_BG,
                              xaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True, key="box_chart")
            st.markdown("**📊 IQR Outlier Statistics**")
            iqr_rows = []
            for cn in sel_box:
                d  = dfe[cn].dropna()
                Q1,Q3 = d.quantile(0.25),d.quantile(0.75); IQR = Q3-Q1
                out = int(((d < Q1-1.5*IQR)|(d > Q3+1.5*IQR)).sum())
                iqr_rows.append({"Column":cn,"Min":round(d.min(),3),
                    "Q1":round(Q1,3),"Median":round(d.median(),3),
                    "Q3":round(Q3,3),"Max":round(d.max(),3),
                    "IQR":round(IQR,3),"Outliers":out,
                    "Outlier %":round(out/len(d)*100,2)})
            st.dataframe(pd.DataFrame(iqr_rows), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 5g SCATTER EXPLORER ───────────────────────────────────────────────────
    if len(num_cols) >= 2:
        st.markdown('<div class="eda-card">', unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">🔭 Scatter Plot Explorer</div>', unsafe_allow_html=True)
        s1,s2,s3 = st.columns(3)
        with s1: x_ax = st.selectbox("X axis:", num_cols, index=0, key="sc_x")
        with s2: y_ax = st.selectbox("Y axis:", num_cols, index=min(1,len(num_cols)-1), key="sc_y")
        with s3: hue  = st.selectbox("Colour by:", ["— None —"]+cat_cols, key="sc_hue")
        color_col = None if hue == "— None —" else hue
        sdf = dfe.sample(min(4000,len(dfe)), random_state=42)
        fig = px.scatter(sdf, x=x_ax, y=y_ax, color=color_col,
                         opacity=0.65, color_discrete_sequence=PAL,
                         title=f"Scatter: {x_ax}  ×  {y_ax}"
                               + ("" if color_col is None else f"  |  by {color_col}"),
                         labels={x_ax:x_ax, y_ax:y_ax})
        fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color=FIG_BG)))
        if color_col is None:
            common = sdf[[x_ax,y_ax]].dropna()
            if len(common) >= 2:
                z      = np.polyfit(common[x_ax], common[y_ax], 1)
                x_line = np.linspace(common[x_ax].min(), common[x_ax].max(), 200)
                fig.add_trace(go.Scatter(
                    x=x_line, y=np.polyval(z,x_line), mode="lines",
                    name=f"Trend (slope={z[0]:.3f})",
                    line=dict(color="#E74C3C", width=2, dash="dash")))
        fig = apply_theme(fig, 440)
        st.plotly_chart(fig, use_container_width=True, key="scatter_main")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 5h PAIR PLOT ──────────────────────────────────────────────────────────
    if len(num_cols) >= 3:
        st.markdown('<div class="eda-card">', unsafe_allow_html=True)
        st.markdown('<div class="eda-card-title">🔗 Pair Plot — Multi-variable Relationships</div>', unsafe_allow_html=True)
        pair_def = num_cols[:min(4,len(num_cols))]
        pair_sel = st.multiselect("Select columns (2–5 recommended):",
                                   num_cols, default=pair_def, key="pair_sel")
        if len(pair_sel) >= 2:
            pairdf = dfe[pair_sel+([cat_cols[0]] if cat_cols else [])].sample(
                        min(1500,len(dfe)), random_state=42)
            fig = px.scatter_matrix(pairdf, dimensions=pair_sel,
                                    color=cat_cols[0] if cat_cols else None,
                                    color_discrete_sequence=PAL, opacity=0.55,
                                    title="Pair Plot — "+", ".join(pair_sel))
            fig.update_traces(marker=dict(size=3, line=dict(width=0.2, color=FIG_BG)),
                              diagonal_visible=True)
            fig = apply_theme(fig, max(520,len(pair_sel)*135))
            st.plotly_chart(fig, use_container_width=True, key="pair_matrix")
        else:
            st.info("Select at least 2 columns for the pair plot.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Final banner
    st.markdown(f"""
    <div class="ok-box" style="margin-top:1.2rem;">
      ✅ EDA complete &nbsp;·&nbsp;
      <strong>{len(num_cols)}</strong> numeric &nbsp;·&nbsp;
      <strong>{len(cat_cols)}</strong> categorical &nbsp;·&nbsp;
      Distributions · Correlation · Outliers · Time Series · Scatter · Pair Plot
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  💪 <b>FitPulse Analytics</b> &nbsp;·&nbsp;
  Built with Streamlit &nbsp;·&nbsp;
  Thistle Purple Edition (#D8BFD8) &nbsp;·&nbsp;
  Professional Health Data Pipeline
</div>
""", unsafe_allow_html=True)