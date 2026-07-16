"""
Customer Segmentation Page
===========================
Visualizes KMeans clustering results from Notebook 05.
Shows segment profiles, distributions, radar charts, heatmap,
and an interactive persona classifier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Path Setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'models'

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Segments", page_icon="👥", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Overall dark-theme polish */
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,30,60,0.7), rgba(20,20,40,0.9));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
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

    /* Page header */
    .page-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .page-header h1 {
        font-size: 2.4rem;
        background: linear-gradient(90deg, #9b59b6, #3498db, #2ecc71);
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
        border-left: 4px solid #9b59b6;
        padding-left: 12px;
        margin: 30px 0 16px 0;
    }

    /* Persona result card */
    .persona-card {
        background: linear-gradient(135deg, rgba(30,30,60,0.8), rgba(20,20,40,0.95));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 6px 30px rgba(0,0,0,0.4);
    }
    .persona-name {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .persona-desc {
        color: #a0a0c0;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Styled dataframe wrapper */
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ── Color Scheme ────────────────────────────────────────────────────────────
PERSONA_COLORS = {
    "Loyal Regulars": "#2ecc71",
    "Occasional Weekday Shoppers": "#f39c12",
    "Weekend Explorers": "#9b59b6",
    "Bulk Family Buyers": "#e74c3c",
}

PERSONA_DESCRIPTIONS = {
    "Loyal Regulars": (
        "Khách hàng trung thành, mua sắm thường xuyên mỗi tuần và luôn mua lại các sản phẩm yêu thích. "
        "Đây là nhóm tạo ra doanh thu định kỳ ổn định nhất."
    ),
    "Occasional Weekday Shoppers": (
        "Khách hàng ít mua sắm, chỉ mua tiện lợi vào ngày thường với giỏ hàng nhỏ. "
        "Nhóm này cần các chiến dịch re-engagement để tăng tần suất mua."
    ),
    "Weekend Explorers": (
        "Khách hàng thích mua sắm cuối tuần và thử các sản phẩm mới. "
        "Tỉ lệ mua lại thấp — họ thích khám phá hơn là mua theo thói quen."
    ),
    "Bulk Family Buyers": (
        "Nhóm mua sỉ với giỏ hàng lớn, đa dạng ngành hàng. "
        "Có thể là các gia đình mua thực phẩm cho cả tuần hoặc nửa tháng."
    ),
}

# ── Data Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading KMeans model…")
def load_kmeans_model():
    """Load the trained KMeans model."""
    return joblib.load(MODELS_PATH / "kmeans_model.joblib")


@st.cache_resource(show_spinner="Loading scaler…")
def load_scaler():
    """Load the fitted StandardScaler."""
    return joblib.load(MODELS_PATH / "scaler.joblib")


@st.cache_data(show_spinner="Loading cluster metadata…")
def load_metadata():
    """Load cluster metadata JSON (persona map, features, metrics)."""
    with open(MODELS_PATH / "cluster_metadata.json", "r") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading cluster profiles…")
def load_profiles():
    """Load cluster profiles CSV with mean feature values per persona."""
    return pd.read_csv(MODELS_PATH / "cluster_profiles.csv")


# ── Load everything ─────────────────────────────────────────────────────────
try:
    kmeans = load_kmeans_model()
    scaler = load_scaler()
    metadata = load_metadata()
    profiles = load_profiles()
    data_loaded = True
except FileNotFoundError as e:
    st.error(f"🚫 Required model artifact not found: `{e.filename}`")
    st.info("Please run **Notebook 05 – Customer Segmentation** first to generate the model artifacts.")
    data_loaded = False
except Exception as e:
    st.error(f"🚫 Error loading artifacts: {e}")
    data_loaded = False

if not data_loaded:
    st.stop()

# ── Derived data ────────────────────────────────────────────────────────────
features = metadata["features"]
persona_map = metadata["persona_map"]
metrics = metadata["metrics"]
cluster_sizes = metadata["cluster_sizes"]
n_clusters = metadata["n_clusters"]
n_users = metadata["n_users"]

# Build ordered persona list & color list for consistent indexing
persona_order = [persona_map[str(i)] for i in range(n_clusters)]
color_list = [PERSONA_COLORS[p] for p in persona_order]

# ── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Customer Segmentation</h1>
    <p>Phân cụm K-Means chia <b>206K+</b> khách hàng Instacart thành <b>4 nhóm hành vi</b> khác biệt</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Overview Metrics
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Overview Metrics</div>', unsafe_allow_html=True)

# Find the largest persona
largest_cluster_idx = max(cluster_sizes, key=lambda k: cluster_sizes[k])
largest_persona = persona_map[largest_cluster_idx]
largest_size = cluster_sizes[largest_cluster_idx]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng khách hàng", f"{n_users:,}")
c2.metric("Số phân khúc", n_clusters)
c3.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
c4.metric("Phân khúc lớn nhất", largest_persona, delta=f"{largest_size:,} users")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Persona Summary Table
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Persona Summary Profiles</div>', unsafe_allow_html=True)

# Prepare a nice display dataframe
display_df = profiles.copy()

# Ensure persona column exists and is nicely ordered
if "persona" in display_df.columns:
    display_df = display_df.sort_values("cluster").reset_index(drop=True)

# Round numeric columns for readability
numeric_cols = [c for c in display_df.columns if c not in ("cluster", "persona")]
for col in numeric_cols:
    if col == "n_users":
        display_df[col] = display_df[col].astype(int)
    else:
        display_df[col] = display_df[col].round(3)

# Rename columns for display
rename_map = {
    "cluster": "Cluster",
    "persona": "Persona",
    "total_orders": "Total Orders",
    "avg_basket_size": "Avg Basket Size",
    "reorder_ratio": "Reorder Ratio",
    "avg_days_between_orders": "Avg Days Between Orders",
    "weekend_ratio": "Weekend Ratio",
    "unique_departments": "Unique Depts",
    "n_users": "Users",
}
styled_df = display_df.rename(columns=rename_map)

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Users": st.column_config.NumberColumn(format="%d"),
        "Reorder Ratio": st.column_config.NumberColumn(format="%.3f"),
        "Weekend Ratio": st.column_config.NumberColumn(format="%.3f"),
    },
)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Cluster Distribution Donut Chart
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🍩 Cluster Distribution</div>', unsafe_allow_html=True)

sizes = [cluster_sizes[str(i)] for i in range(n_clusters)]
pct = [s / sum(sizes) * 100 for s in sizes]

fig_donut = go.Figure(data=[go.Pie(
    labels=persona_order,
    values=sizes,
    hole=0.55,
    marker=dict(colors=color_list, line=dict(color="#1a1a2e", width=2)),
    textinfo="label+percent",
    textfont=dict(size=13, color="white"),
    hovertemplate="<b>%{label}</b><br>Users: %{value:,}<br>Share: %{percent}<extra></extra>",
)])

fig_donut.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0c0e0"),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    height=480,
    margin=dict(t=30, b=50, l=30, r=30),
    annotations=[dict(
        text=f"<b>{n_users:,}</b><br>Users",
        x=0.5, y=0.5,
        font=dict(size=18, color="#e0e0ff"),
        showarrow=False,
    )],
)

st.plotly_chart(fig_donut, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Radar Charts (All Personas Overlaid)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🕸️ Persona Radar Comparison</div>', unsafe_allow_html=True)

# Normalize feature values to 0-1 range across clusters for the radar
feature_values = profiles[features].values  # shape: (n_clusters, n_features)
feat_min = feature_values.min(axis=0)
feat_max = feature_values.max(axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1  # avoid division by zero
normalized = (feature_values - feat_min) / feat_range

# Pretty feature labels for display
feature_labels = [
    "Tổng đơn hàng",
    "Giỏ hàng TB",
    "Tỉ lệ mua lại",
    "Ngày giữa đơn",
    "Tỉ lệ cuối tuần",
    "Số ngành hàng",
]

fig_radar = go.Figure()

for i in range(n_clusters):
    persona = persona_order[i]
    vals = normalized[i].tolist()
    vals.append(vals[0])  # close the polygon
    labels_closed = feature_labels + [feature_labels[0]]

    # Build hover text with actual values
    actual_vals = feature_values[i].tolist()
    actual_vals_closed = actual_vals + [actual_vals[0]]
    hover_texts = [
        f"<b>{persona}</b><br>{fl}: {av:.2f} (norm: {nv:.2f})"
        for fl, av, nv in zip(labels_closed, actual_vals_closed, vals)
    ]

    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=labels_closed,
        fill="toself",
        fillcolor=PERSONA_COLORS[persona].replace(")", ",0.1)").replace("rgb", "rgba")
                        if "rgb" in PERSONA_COLORS[persona]
                        else f"rgba({int(PERSONA_COLORS[persona][1:3],16)},{int(PERSONA_COLORS[persona][3:5],16)},{int(PERSONA_COLORS[persona][5:7],16)},0.1)",
        line=dict(color=PERSONA_COLORS[persona], width=2.5),
        name=persona,
        hovertext=hover_texts,
        hoverinfo="text",
    ))

fig_radar.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0c0e0"),
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(
            visible=True,
            range=[0, 1.05],
            showticklabels=True,
            tickfont=dict(size=10, color="#888"),
            gridcolor="rgba(255,255,255,0.08)",
        ),
        angularaxis=dict(
            tickfont=dict(size=12, color="#c0c0e0"),
            gridcolor="rgba(255,255,255,0.08)",
        ),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    height=550,
    margin=dict(t=40, b=60, l=80, r=80),
)

st.plotly_chart(fig_radar, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Cluster Heatmap
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Feature Intensity Heatmap</div>', unsafe_allow_html=True)

# Create annotation text with actual (unscaled) mean values
annotation_text = [[f"{feature_values[i, j]:.2f}" for j in range(len(features))]
                    for i in range(n_clusters)]

fig_heatmap = go.Figure(data=go.Heatmap(
    z=normalized,
    x=feature_labels,
    y=persona_order,
    text=annotation_text,
    texttemplate="%{text}",
    textfont=dict(size=12, color="white"),
    colorscale=[
        [0.0, "#1a1a2e"],
        [0.25, "#16213e"],
        [0.5, "#0f3460"],
        [0.75, "#e94560"],
        [1.0, "#f39c12"],
    ],
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Feature: %{x}<br>"
        "Actual Value: %{text}<br>"
        "Normalized: %{z:.2f}<extra></extra>"
    ),
    colorbar=dict(
        title="Normalized",
        titlefont=dict(color="#c0c0e0"),
        tickfont=dict(color="#888"),
    ),
))

fig_heatmap.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0c0e0"),
    xaxis=dict(tickfont=dict(size=12), tickangle=-30),
    yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
    height=380,
    margin=dict(t=20, b=60, l=200, r=30),
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Interactive Persona Classifier
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-header">🎯 Interactive Persona Classifier</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8888aa; margin-bottom:20px;'>"
    "Nhập thông tin hành vi mua sắm bên dưới để xem bạn thuộc nhóm khách hàng nào!"
    "</p>",
    unsafe_allow_html=True,
)

# Input sliders arranged in two columns
col_left, col_right = st.columns(2)

with col_left:
    inp_total_orders = st.slider(
        "🛒 Total Orders", min_value=3, max_value=99, value=15, step=1,
        help="How many orders have you placed in total?"
    )
    inp_avg_basket = st.slider(
        "🧺 Avg Basket Size", min_value=1.0, max_value=50.0, value=10.0, step=0.5,
        help="Average number of items per order"
    )
    inp_reorder_ratio = st.slider(
        "🔄 Reorder Ratio", min_value=0.0, max_value=1.0, value=0.43, step=0.01,
        help="Fraction of items that are reorders"
    )

with col_right:
    inp_avg_days = st.slider(
        "📅 Avg Days Between Orders", min_value=0.0, max_value=30.0, value=14.5, step=0.5,
        help="Average gap (in days) between consecutive orders"
    )
    inp_weekend_ratio = st.slider(
        "🌅 Weekend Ratio", min_value=0.0, max_value=1.0, value=0.33, step=0.01,
        help="Fraction of orders placed on weekends"
    )
    inp_unique_depts = st.slider(
        "🏪 Unique Departments", min_value=1, max_value=21, value=11, step=1,
        help="Number of different departments you shop from"
    )

# Predict button
if st.button("Phân loại nhóm khách hàng", use_container_width=True, type="primary"):
    # Assemble input vector in the same feature order
    user_input = np.array([[
        inp_total_orders,
        inp_avg_basket,
        inp_reorder_ratio,
        inp_avg_days,
        inp_weekend_ratio,
        inp_unique_depts,
    ]])

    # Scale and predict
    user_scaled = scaler.transform(user_input)
    predicted_cluster = kmeans.predict(user_scaled)[0]
    predicted_persona = persona_map[str(predicted_cluster)]
    persona_color = PERSONA_COLORS[predicted_persona]
    persona_desc = PERSONA_DESCRIPTIONS[predicted_persona]

    st.markdown("---")

    # ── Result card ──
    st.markdown(f"""
    <div class="persona-card">
        <p style="color:#8888aa; font-size:0.9rem; margin-bottom:6px;">NHÓM KHÁCH HÀNG CỦA BẠN</p>
        <div class="persona-name" style="color:{persona_color};">
            {predicted_persona}
        </div>
        <p class="persona-desc">{persona_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── User vs. Cluster Centroids Radar ──
    st.markdown(
        '<p style="color:#c0c0e0; font-weight:600; font-size:1.1rem; text-align:center;">'
        '🕸️ Your Profile vs. Cluster Centroids</p>',
        unsafe_allow_html=True,
    )

    # Normalize user input the same way as cluster profiles
    user_vals = user_input[0]
    # Use the same min/max from cluster profiles for normalization
    # but extend if user goes beyond cluster range
    combined = np.vstack([feature_values, user_vals.reshape(1, -1)])
    c_min = combined.min(axis=0)
    c_max = combined.max(axis=0)
    c_range = c_max - c_min
    c_range[c_range == 0] = 1

    user_norm = ((user_vals - c_min) / c_range).tolist()
    user_norm.append(user_norm[0])  # close polygon
    labels_closed = feature_labels + [feature_labels[0]]

    fig_user_radar = go.Figure()

    # Plot cluster centroids (faded)
    for i in range(n_clusters):
        persona = persona_order[i]
        vals = ((feature_values[i] - c_min) / c_range).tolist()
        vals.append(vals[0])
        fig_user_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels_closed,
            fill="toself",
            fillcolor=f"rgba({int(PERSONA_COLORS[persona][1:3],16)},{int(PERSONA_COLORS[persona][3:5],16)},{int(PERSONA_COLORS[persona][5:7],16)},0.05)",
            line=dict(color=PERSONA_COLORS[persona], width=1.5, dash="dot"),
            name=persona,
            opacity=0.5,
        ))

    # Plot user profile (bold)
    fig_user_radar.add_trace(go.Scatterpolar(
        r=user_norm,
        theta=labels_closed,
        fill="toself",
        fillcolor=f"rgba({int(persona_color[1:3],16)},{int(persona_color[3:5],16)},{int(persona_color[5:7],16)},0.25)",
        line=dict(color=persona_color, width=3),
        name="✨ You",
        marker=dict(size=8, color=persona_color),
    ))

    fig_user_radar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c0c0e0"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1.05],
                showticklabels=True,
                tickfont=dict(size=10, color="#888"),
                gridcolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#c0c0e0"),
                gridcolor="rgba(255,255,255,0.08)",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=500,
        margin=dict(t=30, b=70, l=80, r=80),
    )

    st.plotly_chart(fig_user_radar, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.85rem;'>"
    "Customer Segmentation · KMeans Clustering · Instacart E-Commerce Intelligence"
    "</p>",
    unsafe_allow_html=True,
)
