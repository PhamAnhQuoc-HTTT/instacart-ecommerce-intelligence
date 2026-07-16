"""
Data Overview — EDA Page
=========================
Exploratory data analysis of the Instacart dataset.
Visualises order timing, department distributions,
and user-level behavioural features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ──────────────────────────── Path setup ────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# ──────────────────────────── Page config ───────────────────────────
st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

# ──────────────────────────── Custom CSS ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container { padding-top: 1.5rem; }

    /* Page header */
    .page-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .page-header h1 {
        font-size: 2.4rem;
        background: linear-gradient(90deg, #FF6B6B, #FFB347, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .page-header p {
        color: #8888aa;
        font-size: 1.05rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #c0c0e0;
        border-left: 4px solid #FF6B6B;
        padding-left: 12px;
        margin: 30px 0 16px 0;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,30,60,0.7), rgba(20,20,40,0.9));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(255, 107, 107, 0.3);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetric"] label {
        color: #a0a0c0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #e0e0ff !important;
    }

    /* Caption styling */
    .stCaption {
        color: #7a7a9a !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555d6e;
        font-size: 0.78rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────── Plotly theme defaults ─────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c0c0e0"),
    margin=dict(t=50, b=40, l=50, r=30),
)

# ──────────────────────────── Data loaders ──────────────────────────

@st.cache_data(show_spinner="Loading orders…")
def load_orders() -> pd.DataFrame:
    """Load orders.csv from raw data."""
    return pd.read_csv(
        DATA_PATH / "raw" / "orders.csv",
        usecols=["order_id", "user_id", "order_dow", "order_hour_of_day", "days_since_prior_order"],
    )


@st.cache_data(show_spinner="Loading products…")
def load_products() -> pd.DataFrame:
    """Load products.csv from raw data."""
    return pd.read_csv(DATA_PATH / "raw" / "products.csv")


@st.cache_data(show_spinner="Loading departments…")
def load_departments() -> pd.DataFrame:
    """Load departments.csv from raw data."""
    return pd.read_csv(DATA_PATH / "raw" / "departments.csv")


@st.cache_data(show_spinner="Loading aisles…")
def load_aisles() -> pd.DataFrame:
    """Load aisles.csv from raw data."""
    return pd.read_csv(DATA_PATH / "raw" / "aisles.csv")


@st.cache_data(show_spinner="Loading user features…")
def load_user_features() -> pd.DataFrame:
    """Load user_features.parquet (Spark-partitioned folder)."""
    return pd.read_parquet(DATA_PATH / "processed" / "user_features.parquet")


# ──────────────────────────── Load data ─────────────────────────────
try:
    orders = load_orders()
    products = load_products()
    departments = load_departments()
    aisles = load_aisles()
    user_features = load_user_features()
    data_loaded = True
except FileNotFoundError as e:
    st.error(f"🚫 Data file not found: `{e}`")
    st.info("Please ensure the raw data files and processed features are in the `data/` directory.")
    data_loaded = False
except Exception as e:
    st.error(f"🚫 Error loading data: {e}")
    data_loaded = False

if not data_loaded:
    st.stop()

# ──────────────────────────── Page header ───────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Data Overview</h1>
    <p>Phân tích khám phá trên <b>3.4 triệu+</b> đơn hàng từ <b>200K+</b> khách hàng Instacart</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────── KPI Metrics ───────────────────────────
st.markdown('<div class="section-header">📋 Dataset at a Glance</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tổng đơn hàng", f"{len(orders):,}")
m2.metric("Khách hàng", f"{orders['user_id'].nunique():,}")
m3.metric("Sản phẩm", f"{len(products):,}")
m4.metric("Ngành hàng", f"{len(departments):,}")
m5.metric("Gian hàng", f"{len(aisles):,}")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — Order Timing Patterns
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⏰ Order Timing Patterns</div>', unsafe_allow_html=True)

col_hour, col_dow = st.columns(2, gap="large")

# ── Chart 1: Order Hour Distribution ────────────────────────────────
with col_hour:
    st.subheader("When Do Customers Shop?")

    hour_counts = orders["order_hour_of_day"].value_counts().sort_index()
    hour_df = pd.DataFrame({
        "Hour": hour_counts.index,
        "Orders": hour_counts.values,
    })

    fig_hour = px.bar(
        hour_df,
        x="Hour",
        y="Orders",
        color="Orders",
        color_continuous_scale="YlOrRd",
        labels={"Hour": "Hour of Day", "Orders": "Number of Orders"},
    )
    fig_hour.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(dtick=1, title="Hour of Day (0–23)"),
        yaxis=dict(title="Number of Orders"),
        coloraxis_showscale=False,
        showlegend=False,
    )
    fig_hour.update_traces(
        hovertemplate="<b>%{x}:00</b><br>Orders: %{y:,}<extra></extra>",
        marker_line_width=0,
    )
    st.plotly_chart(fig_hour, use_container_width=True)
    st.caption("Giờ mua sắm cao điểm là 9h – 16h, đỉnh điểm khoảng 10h sáng.")

# ── Chart 2: Day of Week Distribution ───────────────────────────────
with col_dow:
    st.subheader("Shopping by Day of Week")

    # Instacart convention: 0 and 1 are the busiest → Saturday/Sunday
    dow_map = {0: "Saturday", 1: "Sunday", 2: "Monday", 3: "Tuesday",
               4: "Wednesday", 5: "Thursday", 6: "Friday"}
    dow_colors = {
        "Saturday": "#FF6B6B", "Sunday": "#FFB347",
        "Monday": "#4ECDC4", "Tuesday": "#45B7D1",
        "Wednesday": "#A78BFA", "Thursday": "#F472B6", "Friday": "#34D399",
    }

    dow_counts = orders["order_dow"].value_counts().sort_index()
    dow_df = pd.DataFrame({
        "DOW_num": dow_counts.index,
        "Day": [dow_map[d] for d in dow_counts.index],
        "Orders": dow_counts.values,
    })
    # Preserve day ordering
    dow_df["Day"] = pd.Categorical(
        dow_df["Day"],
        categories=["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        ordered=True,
    )
    dow_df = dow_df.sort_values("Day")

    fig_dow = px.bar(
        dow_df,
        x="Day",
        y="Orders",
        color="Day",
        color_discrete_map=dow_colors,
    )
    fig_dow.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(title="Day of Week"),
        yaxis=dict(title="Number of Orders"),
        showlegend=False,
    )
    fig_dow.update_traces(
        hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>",
        marker_line_width=0,
    )
    st.plotly_chart(fig_dow, use_container_width=True)
    st.caption("Cuối tuần (Thứ 7 & Chủ nhật) có lượng đơn hàng cao nhất — phù hợp với thói quen mua sắm thực phẩm.")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Order Frequency
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📅 Days Since Prior Order</div>', unsafe_allow_html=True)

col_dspo, col_dept = st.columns(2, gap="large")

# ── Chart 3: Days Since Prior Order ─────────────────────────────────
with col_dspo:
    st.subheader("Order Frequency Distribution")

    dspo = orders["days_since_prior_order"].dropna()

    fig_dspo = px.histogram(
        dspo,
        nbins=31,
        color_discrete_sequence=["#4ECDC4"],
        labels={"value": "Days Since Prior Order", "count": "Frequency"},
    )
    # Add 30-day cap annotation
    fig_dspo.add_vline(
        x=30, line_dash="dash", line_color="#FF6B6B", line_width=2,
        annotation_text="30-day cap",
        annotation_position="top right",
        annotation_font=dict(color="#FF6B6B", size=12),
    )
    fig_dspo.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(title="Days Since Prior Order", dtick=5),
        yaxis=dict(title="Frequency"),
        showlegend=False,
    )
    fig_dspo.update_traces(
        hovertemplate="<b>%{x:.0f} days</b><br>Count: %{y:,}<extra></extra>",
        marker_line_width=0,
    )
    st.plotly_chart(fig_dspo, use_container_width=True)
    st.caption(
        "Các đỉnh rõ rệt ở 7, 14, 21, 30 ngày cho thấy thói quen mua hàng tuần. "
        "Giá trị bị giới hạn tối đa 30 ngày trong dữ liệu gốc."
    )

# ── Chart 4: Top 15 Departments by Products ────────────────────────
with col_dept:
    st.subheader("Top 15 Departments by Products")

    # Join products with departments
    prod_dept = products.merge(departments, on="department_id", how="left")
    dept_counts = (
        prod_dept.groupby("department")["product_id"]
        .count()
        .sort_values(ascending=True)
        .tail(15)
        .reset_index()
    )
    dept_counts.columns = ["Department", "Product Count"]

    fig_dept = px.bar(
        dept_counts,
        x="Product Count",
        y="Department",
        orientation="h",
        color="Product Count",
        color_continuous_scale="Viridis",
    )
    fig_dept.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(title="Number of Products"),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
    )
    fig_dept.update_traces(
        hovertemplate="<b>%{y}</b><br>Products: %{x:,}<extra></extra>",
        marker_line_width=0,
    )
    st.plotly_chart(fig_dept, use_container_width=True)
    st.caption("Personal care và snacks chiếm nhiều sản phẩm nhất, tiếp theo là pantry và beverages.")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — User Behavior Summary
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👤 User Behavior Summary</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8888aa; margin-bottom:16px;'>"
    "Distributions from aggregated user-level features (computed via PySpark feature engineering pipeline)."
    "</p>",
    unsafe_allow_html=True,
)

# Detect available columns dynamically
uf_cols = user_features.columns.tolist()

# ── Row of 3 distribution charts ───────────────────────────────────
col_u1, col_u2, col_u3 = st.columns(3, gap="large")

# Chart 5a: Total Orders distribution
with col_u1:
    total_orders_col = "total_orders" if "total_orders" in uf_cols else None
    if total_orders_col:
        st.subheader("Total Orders per User")
        fig_to = px.histogram(
            user_features,
            x=total_orders_col,
            nbins=50,
            color_discrete_sequence=["#FF6B6B"],
            labels={total_orders_col: "Total Orders"},
        )
        fig_to.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis=dict(title="Total Orders"),
            yaxis=dict(title="Number of Users"),
            showlegend=False,
        )
        fig_to.update_traces(
            hovertemplate="<b>%{x:.0f} orders</b><br>Users: %{y:,}<extra></extra>",
            marker_line_width=0,
        )
        st.plotly_chart(fig_to, use_container_width=True)
        st.caption(
            f"📌 Median: {user_features[total_orders_col].median():.0f} orders · "
            f"Mean: {user_features[total_orders_col].mean():.1f} orders"
        )
    else:
        st.info("Column `total_orders` not found in user_features.")

# Chart 5b: Average Basket Size distribution
with col_u2:
    basket_col = "avg_basket_size" if "avg_basket_size" in uf_cols else None
    if basket_col:
        st.subheader("Average Basket Size")
        fig_bs = px.histogram(
            user_features,
            x=basket_col,
            nbins=50,
            color_discrete_sequence=["#FFB347"],
            labels={basket_col: "Avg Basket Size"},
        )
        fig_bs.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis=dict(title="Avg Items per Order"),
            yaxis=dict(title="Number of Users"),
            showlegend=False,
        )
        fig_bs.update_traces(
            hovertemplate="<b>%{x:.1f} items</b><br>Users: %{y:,}<extra></extra>",
            marker_line_width=0,
        )
        st.plotly_chart(fig_bs, use_container_width=True)
        st.caption(
            f"📌 Median: {user_features[basket_col].median():.1f} items · "
            f"Mean: {user_features[basket_col].mean():.1f} items"
        )
    else:
        st.info("Column `avg_basket_size` not found in user_features.")

# Chart 5c: Reorder Ratio distribution
with col_u3:
    reorder_col = "reorder_ratio" if "reorder_ratio" in uf_cols else None
    if reorder_col:
        st.subheader("Reorder Ratio")
        fig_rr = px.histogram(
            user_features,
            x=reorder_col,
            nbins=50,
            color_discrete_sequence=["#4ECDC4"],
            labels={reorder_col: "Reorder Ratio"},
        )
        fig_rr.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis=dict(title="Reorder Ratio (0–1)", range=[-0.02, 1.02]),
            yaxis=dict(title="Number of Users"),
            showlegend=False,
        )
        fig_rr.update_traces(
            hovertemplate="<b>Ratio: %{x:.2f}</b><br>Users: %{y:,}<extra></extra>",
            marker_line_width=0,
        )
        st.plotly_chart(fig_rr, use_container_width=True)
        st.caption(
            f"📌 Median: {user_features[reorder_col].median():.2f} · "
            f"Mean: {user_features[reorder_col].mean():.2f}"
        )
    else:
        st.info("Column `reorder_ratio` not found in user_features.")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — User Features Summary Table
# ═══════════════════════════════════════════════════════════════════════
with st.expander("📋 User Features — Descriptive Statistics", expanded=False):
    # Show describe() for all numeric columns in user_features
    desc = user_features.describe().T
    desc.index.name = "Feature"
    st.dataframe(
        desc.style.format("{:.2f}"),
        use_container_width=True,
    )

# ──────────────────────────── Footer ────────────────────────────────
st.markdown("""
<div class="footer">
    Data Overview · Exploratory Data Analysis · Instacart E-Commerce Intelligence
</div>
""", unsafe_allow_html=True)
