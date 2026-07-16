"""
Instacart E-Commerce Intelligence — Homepage
=============================================
Entry point for the Streamlit multi-page dashboard.
Run with:  streamlit run app/main.py
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

# ──────────────────────────── Path setup ────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"

# ──────────────────────────── Page config ───────────────────────────
st.set_page_config(
    page_title="Instacart E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ────────────────────────────
st.markdown(
    """
<style>
/* ── Global tweaks ───────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit header bar for cleaner look */
header[data-testid="stHeader"] {
    background: transparent;
}

/* ── Hero / gradient header ──────────────────────────────────────── */
.hero-section {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.hero-section::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255, 107, 107, 0.06) 0%, transparent 70%);
    animation: pulse 8s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #FF6B6B, #FFB347, #FF6B6B);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s linear infinite;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}

@keyframes gradientShift {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero-subtitle {
    color: #9ca3b0;
    font-size: 1.05rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Metric cards (glow effect) ──────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1a1f2e, #141824);
    border: 1px solid rgba(255, 107, 107, 0.15);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 0 20px rgba(255, 107, 107, 0.05),
                0 4px 15px rgba(0, 0, 0, 0.3);
    transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(255, 107, 107, 0.4);
    box-shadow: 0 0 30px rgba(255, 107, 107, 0.12),
                0 8px 25px rgba(0, 0, 0, 0.4);
    transform: translateY(-4px);
}

div[data-testid="stMetric"] label {
    color: #8892a4 !important;
    font-weight: 500;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FAFAFA !important;
    font-weight: 700;
    font-size: 1.9rem;
}

/* ── Section headers ─────────────────────────────────────────────── */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #FAFAFA;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(255, 107, 107, 0.3);
    display: inline-block;
}

/* ── Info cards ───────────────────────────────────────────────────── */
.info-card {
    background: linear-gradient(145deg, #1a1f2e, #141824);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1.6rem;
    height: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}

.info-card:hover {
    border-color: rgba(255, 107, 107, 0.25);
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
}

.info-card h3 {
    color: #FF6B6B;
    font-size: 1.15rem;
    margin-bottom: 0.6rem;
}

.info-card p {
    color: #b0b8c8;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── Tech badge pills ────────────────────────────────────────────── */
.tech-badge {
    display: inline-block;
    background: rgba(255, 107, 107, 0.10);
    border: 1px solid rgba(255, 107, 107, 0.25);
    color: #FF6B6B;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0.3rem 0.25rem;
    transition: all 0.25s ease;
}

.tech-badge:hover {
    background: rgba(255, 107, 107, 0.20);
    border-color: rgba(255, 107, 107, 0.5);
    box-shadow: 0 0 12px rgba(255, 107, 107, 0.15);
}

/* ── Overview box ────────────────────────────────────────────────── */
.overview-box {
    background: linear-gradient(145deg, #1a1f2e, #141824);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 4px solid #FF6B6B;
    border-radius: 0 14px 14px 0;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    color: #b0b8c8;
    font-size: 0.95rem;
    line-height: 1.75;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}

/* ── Sidebar styling ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0f0c29, #1a1f2e);
}

/* ── Footer ──────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    color: #555d6e;
    font-size: 0.78rem;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    margin-top: 3rem;
}

.footer a { color: #FF6B6B; text-decoration: none; }
.footer a:hover { text-decoration: underline; }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────── Data loaders ──────────────────────────
@st.cache_data(show_spinner=False)
def load_overview_stats() -> dict:
    """Load order & user counts from raw orders.csv."""
    try:
        orders = pd.read_csv(
            DATA_PATH / "raw" / "orders.csv",
            usecols=["order_id", "user_id"],
        )
        return {"n_orders": len(orders), "n_users": orders["user_id"].nunique()}
    except FileNotFoundError:
        return {"n_orders": 3_421_083, "n_users": 206_209}


@st.cache_data(show_spinner=False)
def load_product_count() -> int:
    """Load product count from raw products.csv."""
    try:
        products = pd.read_csv(
            DATA_PATH / "raw" / "products.csv",
            usecols=["product_id"],
        )
        return products["product_id"].nunique()
    except FileNotFoundError:
        return 49_688


@st.cache_data(show_spinner=False)
def load_cluster_metadata() -> dict:
    """Load customer segmentation metadata."""
    meta_path = MODELS_PATH / "cluster_metadata.json"
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"n_clusters": 4, "persona_map": {}}


# ──────────────────────────── Sidebar ───────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shopping-cart.png",
        width=64,
    )
    st.markdown("### 🛒 Instacart Intelligence")
    st.caption("E-Commerce Analytics Platform")
    st.divider()
    st.markdown(
        """
**Project Info**
- 📊  Multi-page analytics dashboard
- 🤖  ML-powered predictions
- 👥  Customer segmentation
- 📈  Interactive visualisations
"""
    )
    st.divider()
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&style=for-the-badge)]"
        "(https://github.com/PhamAnhQuoc-HTTT/instacart-ecommerce-intelligence)"
    )
    st.caption("Built with ❤️ using Streamlit")


# ──────────────────────────── Load data ─────────────────────────────
stats = load_overview_stats()
n_products = load_product_count()
cluster_meta = load_cluster_metadata()
n_segments = cluster_meta.get("n_clusters", 4)

# ──────────────────────────── Hero section ──────────────────────────
st.markdown(
    """
<div class="hero-section">
    <div class="hero-title">Instacart E-Commerce Intelligence</div>
    <div class="hero-subtitle">
        Nền tảng phân tích hành vi khách hàng thương mại điện tử — kết hợp
        khám phá dữ liệu, phân cụm khách hàng và dự đoán mua lại bằng Machine Learning
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────── KPI metrics ───────────────────────────
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.metric(label="Tổng khách hàng", value=f"{stats['n_users']:,}")
with col2:
    st.metric(label="Tổng đơn hàng", value=f"{stats['n_orders']:,}")
with col3:
    st.metric(label="Sản phẩm", value=f"{n_products:,}")
with col4:
    st.metric(label="Phân khúc KH", value=str(n_segments))

# ──────────────────────────── Project overview ──────────────────────
st.markdown('<div class="section-header">Giới thiệu dự án</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="overview-box">
    <strong>Instacart E-Commerce Intelligence</strong> là một nền tảng phân tích dữ liệu
    đầu-cuối (end-to-end) được xây dựng trên bộ dữ liệu
    <a href="https://www.kaggle.com/c/instacart-market-basket-analysis"
       target="_blank" style="color:#FF6B6B">
    Instacart Market Basket Analysis</a> gồm hơn 3.4 triệu đơn hàng. Dự án kết hợp
    <em>phân tích khám phá dữ liệu (EDA)</em>, <em>phân cụm khách hàng (K-Means Clustering)</em>,
    và <em>dự đoán mua lại (XGBoost Classification)</em> — tất cả được trình bày
    trên giao diện tương tác Streamlit.<br><br>
    Nền tảng giúp trả lời các câu hỏi kinh doanh:
    <ul style="margin-top:0.5rem">
        <li>Sản phẩm nào được mua lại nhiều nhất?</li>
        <li>Khách hàng có những nhóm hành vi nào khác biệt?</li>
        <li>Có thể dự đoán được khách hàng sẽ mua lại sản phẩm nào không?</li>
        <li>Thời điểm nào trong tuần/ngày khách hàng mua sắm nhiều nhất?</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────── Navigation guide ──────────────────────
st.markdown('<div class="section-header">Điều hướng trang</div>', unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3 = st.columns(3, gap="medium")

with nav_col1:
    st.markdown(
        """
<div class="info-card">
    <h3>Khám phá dữ liệu (EDA)</h3>
    <p>
        Phân tích xu hướng đặt hàng theo giờ, ngày trong tuần,
        phân bố sản phẩm theo ngành hàng và hành vi khách hàng
        qua hơn 3.4 triệu giao dịch.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

with nav_col2:
    st.markdown(
        """
<div class="info-card">
    <h3>Dự đoán mua lại</h3>
    <p>
        Sử dụng mô hình XGBoost để dự đoán khách hàng có mua lại
        sản phẩm không. Xem mức độ quan trọng từng đặc trưng và
        thử dự đoán trực tiếp trên giao diện.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

with nav_col3:
    st.markdown(
        """
<div class="info-card">
    <h3>Phân khúc khách hàng</h3>
    <p>
        Kết quả phân cụm K-Means chia 206K khách hàng thành 4 nhóm
        hành vi khác biệt — từ khách trung thành đến khách khám phá
        cuối tuần. So sánh qua biểu đồ radar và heatmap.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

# ──────────────────────────── Tech stack ────────────────────────────
st.markdown('<div class="section-header">Công nghệ sử dụng</div>', unsafe_allow_html=True)

tech_items = [
    "Python", "PySpark", "Scikit-learn", "XGBoost",
    "Streamlit", "Plotly", "Pandas", "NumPy",
]

badges_html = " ".join(
    f'<span class="tech-badge">{name}</span>' for name in tech_items
)
st.markdown(
    f'<div style="text-align:center; padding: 1rem 0;">{badges_html}</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────── Cluster personas (bonus) ──────────────
persona_map = cluster_meta.get("persona_map", {})
if persona_map:
    with st.expander("🎭 Customer Segment Personas", expanded=False):
        seg_cols = st.columns(len(persona_map))
        cluster_sizes = cluster_meta.get("cluster_sizes", {})
        colors = ["#FF6B6B", "#FFB347", "#4ECDC4", "#A78BFA"]
        for idx, (cid, name) in enumerate(sorted(persona_map.items())):
            with seg_cols[idx]:
                size = cluster_sizes.get(str(cid), "—")
                color = colors[idx % len(colors)]
                st.markdown(
                    f"""
<div style="text-align:center; padding:1rem 0.5rem;">
    <div style="font-size:2rem; margin-bottom:0.3rem;">{'👤👥🛍️🏠'[idx]}</div>
    <div style="color:{color}; font-weight:600; font-size:0.95rem;">{name}</div>
    <div style="color:#8892a4; font-size:0.82rem; margin-top:0.25rem;">{size:,} users</div>
</div>
""",
                    unsafe_allow_html=True,
                )

# ──────────────────────────── Footer ────────────────────────────────
st.markdown(
    """
<div class="footer">
    Instacart E-Commerce Intelligence &nbsp;·&nbsp;
    <a href="https://github.com/PhamAnhQuoc-HTTT/instacart-ecommerce-intelligence">GitHub</a>
    &nbsp;·&nbsp; Built with Streamlit &amp; ❤️
</div>
""",
    unsafe_allow_html=True,
)
