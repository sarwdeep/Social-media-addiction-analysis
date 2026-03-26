import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Addiction Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #12101e 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.2);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #c4b8e8 !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0533 0%, #0d1b3e 50%, #0a1628 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero-sub {
    color: #9ca3af;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #a78bfa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #13111f, #1a1630);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(139, 92, 246, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #a78bfa;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8e6f0;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(139, 92, 246, 0.3);
}
.section-sub {
    color: #6b7280;
    font-size: 0.9rem;
    margin-top: -0.5rem;
    margin-bottom: 1.5rem;
}

/* ── Insight Cards ── */
.insight-card {
    background: #13111f;
    border-left: 3px solid #a78bfa;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    color: #c4b8e8;
    font-size: 0.9rem;
    line-height: 1.6;
}
.insight-card strong {
    color: #a78bfa;
}

/* ── Plotly chart container ── */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #13111f;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(139,92,246,0.2);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #6b7280;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(139, 92, 246, 0.2) !important;
    color: #a78bfa !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    padding: 0.6rem 1.8rem;
    transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(139,92,246,0.4);
}

/* ── Selectbox / Slider ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #13111f !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 8px !important;
    color: #e8e6f0 !important;
}

/* ── DataFrames ── */
.dataframe {
    background: #13111f !important;
    color: #e8e6f0 !important;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(139,92,246,0.15);
    margin: 1.5rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ────────────────────────────────────────────────────────
CHART_BG   = "#0f0f1a"
CHART_PAPER= "#0f0f1a"
GRID_COLOR = "rgba(139,92,246,0.1)"
TEXT_COLOR = "#9ca3af"
ACCENT     = ["#a78bfa","#60a5fa","#f472b6","#34d399","#fbbf24","#fb923c","#e879f9"]

def apply_dark_theme(fig, height=400):
    fig.update_layout(
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_BG,
        font=dict(family="Space Grotesk", color=TEXT_COLOR, size=12),
        height=height,
        margin=dict(t=40, b=40, l=40, r=20),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(139,92,246,0.2)", borderwidth=1),
    )
    return fig

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df



# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'></div>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#a78bfa;'>Dashboard</div>
        <div style='font-size:0.75rem; color:#6b7280;'>Social Media Addiction Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload Dataset (.csv)", type=["csv"])

    st.markdown("---")
    st.markdown("<div style='color:#6b7280; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;'>Navigation</div>", unsafe_allow_html=True)

    pages = {
        " Overview":          "overview",
        " Data Explorer":     "explorer",
        " Platform Analysis": "platform",
        " Academic Impact":   "academic",
        " Insights & Conclusions": "insights",
    }
    page = st.radio("", list(pages.keys()), label_visibility="collapsed")
    active = pages[page]

    st.markdown("---")
    st.markdown("<div style='color:#4b5563; font-size:0.75rem; text-align:center;'>Built with Streamlit & Python<br/>Course Project</div>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">Course Project</div>
    <div class="hero-title">Social Media Addiction<br/>Analysis Dashboard</div>
    <p class="hero-sub">Exploring usage patterns, mental health impacts & academic correlations among students</p>
</div>
""", unsafe_allow_html=True)

# ── Guard: no file ────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style='background:#13111f; border:1px dashed rgba(139,92,246,0.4); border-radius:16px;
                padding:3rem; text-align:center; margin-top:1rem;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>📂</div>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#a78bfa; margin-bottom:0.5rem;'>
            Upload Your Dataset to Begin
        </div>
        <div style='color:#6b7280; font-size:0.95rem;'>
            Upload <strong style="color:#c4b8e8">StudentsSocialMediaAddiction.csv</strong> using the sidebar
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data(uploaded_file)

# ── Auto-detect key columns (flexible) ───────────────────────────────────────
cols = df.columns.tolist()

def find_col(keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return None

COL_USAGE    = find_col(["daily_usage","usage_time","hours","daily_use","avg_daily"])
COL_PLATFORM = find_col(["platform","app","social_media"])
COL_GENDER   = find_col(["gender","sex"])
COL_AGE      = find_col(["age"])
COL_MENTAL   = find_col(["mental","anxiety","depression","stress","well","mood"])
COL_SLEEP    = find_col(["sleep"])
COL_ACADEMIC = find_col(["academic","grade","gpa","performance","score"])
COL_ADDICTED = find_col(["addicted","addiction","dependent","hooked"])
COL_COUNTRY  = find_col(["country","location","region"])

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if active == "overview":
    st.markdown("<div class='section-header'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (len(df), "Total Records"),
        (len(cols), "Total Features"),
        (df.isnull().sum().sum(), "Missing Values"),
        (len(cat_cols), "Categorical Cols"),
    ]
    for col_obj, (val, label) in zip([c1,c2,c3,c4], metrics):
        col_obj.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val:,}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown("<div class='section-header' style='font-size:1.1rem;'>📋 Sample Data</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=320)
    with c2:
        st.markdown("<div class='section-header' style='font-size:1.1rem;'>📐 Data Types</div>", unsafe_allow_html=True)
        dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str), "Non-Null": df.notnull().sum().values})
        st.dataframe(dtype_df, use_container_width=True, height=320)

    st.markdown("---")
    st.markdown("<div class='section-header' style='font-size:1.1rem;'>📈 Descriptive Statistics</div>", unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif active == "explorer":
    st.markdown("<div class='section-header'> Interactive Data Explorer</div>", unsafe_allow_html=True)

    with st.expander("Filter Data", expanded=True):
        filter_cols = st.multiselect("Filter by columns:", cat_cols, default=cat_cols[:2] if len(cat_cols)>=2 else cat_cols)
        filtered = df.copy()
        for fc in filter_cols:
            vals = st.multiselect(f"  ↳ {fc}:", df[fc].dropna().unique().tolist(), default=df[fc].dropna().unique().tolist())
            filtered = filtered[filtered[fc].isin(vals)]

    st.markdown(f"<div class='insight-card'>Showing <strong>{len(filtered):,}</strong> of <strong>{len(df):,}</strong> rows after filters.</div>", unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=400)

    st.download_button("⬇️ Download Filtered Data", filtered.to_csv(index=False), "filtered_data.csv", "text/csv")
# ─────────────────────────────────────────────────────────────────────────────

# PAGE: PLATFORM ANALYSIS
elif active == "platform":
    st.markdown("<div class='section-header'>📱 Platform Analysis</div>", unsafe_allow_html=True)

    if COL_PLATFORM is None:
        st.warning("No platform/app column detected. Please check your dataset column names.")
    else:
        vc = df[COL_PLATFORM].value_counts().reset_index()
        vc.columns = ["Platform","Count"]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(vc, x="Platform", y="Count", title="Users per Platform",
                         color="Count", color_continuous_scale=["#4f46e5","#a78bfa","#f472b6"])
            apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(vc, names="Platform", values="Count", title="Platform Share",
                          color_discrete_sequence=ACCENT, hole=0.45)
            apply_dark_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        if COL_USAGE:
            st.markdown("<div class='section-header' style='font-size:1.1rem;'>⏱️ Daily Usage by Platform</div>", unsafe_allow_html=True)
            fig3 = px.box(df, x=COL_PLATFORM, y=COL_USAGE, title="Daily Usage Hours by Platform",
                          color=COL_PLATFORM, color_discrete_sequence=ACCENT)
            apply_dark_theme(fig3, height=420)
            st.plotly_chart(fig3, use_container_width=True)

        if COL_GENDER:
            st.markdown("<div class='section-header' style='font-size:1.1rem;'>👥 Platform by Gender</div>", unsafe_allow_html=True)
            cross = pd.crosstab(df[COL_PLATFORM], df[COL_GENDER])
            fig4 = px.bar(cross.reset_index(), x=COL_PLATFORM,
                          y=cross.columns.tolist(), barmode="group",
                          title="Platform Preference by Gender",
                          color_discrete_sequence=ACCENT)
            apply_dark_theme(fig4, height=400)
            st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────

# PAGE: ACADEMIC IMPACT
elif active == "academic":
    st.markdown("<div class='section-header'>🎓 Academic Impact Analysis</div>", unsafe_allow_html=True)

    academic_candidates = [c for c in cols if any(k in c.lower() for k in ["academic","grade","gpa","performance","score","study","cgpa"])]

    if not academic_candidates:
        st.info("No academic performance column detected. Showing usage-based analysis.")
        if COL_USAGE and COL_AGE:
            fig = px.scatter(df, x=COL_AGE, y=COL_USAGE, title="Age vs Daily Usage",
                             color=COL_GENDER if COL_GENDER else None,
                             trendline="ols", color_discrete_sequence=ACCENT)
            apply_dark_theme(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
    else:
        sel_ac = st.selectbox("Academic metric:", academic_candidates)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=sel_ac, nbins=25, title=f"Distribution: {sel_ac}",
                               color_discrete_sequence=[ACCENT[2]])
            apply_dark_theme(fig, height=330)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if COL_USAGE:
                fig2 = px.scatter(df, x=COL_USAGE, y=sel_ac,
                                  title=f"Usage vs {sel_ac}",
                                  trendline="ols", opacity=0.65,
                                  color=COL_GENDER if COL_GENDER else None,
                                  color_discrete_sequence=ACCENT)
                apply_dark_theme(fig2, height=330)
                st.plotly_chart(fig2, use_container_width=True)

        if COL_PLATFORM:
            fig3 = px.box(df, x=COL_PLATFORM, y=sel_ac,
                          title=f"{sel_ac} by Platform",
                          color=COL_PLATFORM, color_discrete_sequence=ACCENT)
            apply_dark_theme(fig3, height=380)
            st.plotly_chart(fig3, use_container_width=True)

        if COL_ADDICTED:
            fig4 = px.violin(df, x=COL_ADDICTED, y=sel_ac,
                             title=f"{sel_ac} by Addiction Status",
                             color=COL_ADDICTED, box=True,
                             color_discrete_sequence=ACCENT)
            apply_dark_theme(fig4, height=380)
            st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif active == "insights":
    st.markdown("<div class='section-header'>💡 Key Insights & Conclusions</div>", unsafe_allow_html=True)

    insights = [
        ("📱 High Daily Usage", "Students who spend more than 4 hours/day on social media show higher anxiety and stress scores compared to low-usage peers."),
        ("😴 Sleep Disruption", "Elevated social media usage is strongly correlated with reduced sleep duration, particularly among younger students (18–22 years)."),
        ("🎓 Academic Decline", "Students categorized as 'addicted' tend to report lower academic performance scores and reduced study focus."),
        ("📊 Platform Patterns", "Instagram and TikTok dominate usage time, while Twitter/X is associated with higher stress score correlations."),
    ]

    for title, body in insights:
        st.markdown(f"<div class='insight-card'><strong>{title}</strong><br/>{body}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-header' style='font-size:1.1rem;'>📌 Recommendations</div>", unsafe_allow_html=True)
    recommendations = [
        "🔔 Implement digital wellness features like screen time reminders in apps.",
        "📚 Promote social media detox programs in educational institutions.",
        "🌙 Encourage students to set a no-phone period at least 1 hour before sleep.",
        "📖 Create awareness campaigns around mindful social media consumption.",
        "🏫 Schools should integrate digital literacy & addiction awareness into curriculum.",
    ]
    for r in recommendations:
        st.markdown(f"<div class='insight-card'>{r}</div>", unsafe_allow_html=True)

    st.markdown("---")
   