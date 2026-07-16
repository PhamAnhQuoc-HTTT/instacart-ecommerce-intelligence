"""
Reorder Prediction — ML-powered reorder likelihood prediction.

Demonstrates the XGBClassifier model trained in Notebook 04.
Users can explore model metrics, feature importances, and make
interactive predictions with custom feature inputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import joblib
from pathlib import Path

# ──────────────────────────────────────────────
# Path Setup
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'models'

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Reorder Prediction",
    page_icon="🔮",
    layout="wide",
)

# ──────────────────────────────────────────────
# Custom CSS — clean, professional look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; }

    /* Page header */
    .page-header { padding: 10px 0 6px 0; }
    .page-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 2px;
    }
    .page-header p { color: #8892a4; font-size: 1rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e, #141824);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"] label {
        color: #8892a4 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Section headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #c0c8d8;
        border-left: 3px solid #667eea;
        padding-left: 12px;
        margin: 28px 0 14px 0;
    }

    /* Info box */
    .info-box {
        background: rgba(102, 126, 234, 0.06);
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        color: #b0b8c8;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Prediction result */
    .result-card {
        background: linear-gradient(135deg, #1a1f2e, #141824);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .result-label {
        font-size: 0.85rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .result-value {
        font-size: 3.2rem;
        font-weight: 700;
        margin: 6px 0;
    }
    .result-high { color: #34d399; }
    .result-mid { color: #fbbf24; }
    .result-low { color: #f87171; }
    .result-note {
        font-size: 1rem;
        color: #b0b8c8;
        margin-top: 6px;
    }

    /* Explanation factors */
    .factor-row {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
    }
    .factor-icon {
        font-size: 1.1rem;
        width: 28px;
        text-align: center;
        flex-shrink: 0;
    }
    .factor-text {
        color: #b0b8c8;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-left: 8px;
    }
    .factor-text strong { color: #e2e8f0; }

    /* Form category label */
    .form-category {
        font-size: 0.95rem;
        font-weight: 600;
        color: #c0c8d8;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* Footer */
    .page-footer {
        text-align: center;
        color: #555d6e;
        font-size: 0.8rem;
        padding: 1.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Vietnamese Feature Descriptions
# ──────────────────────────────────────────────
FEATURE_VN = {
    "add_to_cart_order": ("Vị trí trong giỏ hàng", "Thứ tự sản phẩm được bỏ vào giỏ. Vị trí thấp = khách nhớ đến sớm = hay mua lại."),
    "order_dow": ("Ngày đặt hàng", "Ngày trong tuần (0 = Chủ nhật). Mỗi ngày có hành vi mua sắm khác nhau."),
    "order_hour_of_day": ("Giờ đặt hàng", "Giờ trong ngày khách đặt đơn (0–23h). Giờ cao điểm thường 9–16h."),
    "days_since_prior_order": ("Khoảng cách đơn hàng", "Bao nhiêu ngày kể từ lần đặt trước. Ngắn = mua thường xuyên."),
    "total_orders": ("Tổng đơn đã đặt", "Tổng số lần khách hàng đặt hàng. Càng nhiều = khách trung thành."),
    "avg_basket_size": ("Giỏ hàng trung bình", "Trung bình mỗi đơn mua bao nhiêu món."),
    "reorder_ratio": ("Tỉ lệ mua lại (user)", "Bao nhiêu % sản phẩm trong giỏ là hàng đã từng mua. Cao = khách có thói quen."),
    "unique_departments": ("Số ngành hàng", "Khách mua từ bao nhiêu ngành hàng khác nhau (1–21)."),
    "times_ordered": ("Số lần SP được mua", "Tổng số lần sản phẩm này đã được mua bởi tất cả khách hàng."),
    "reorder_rate": ("Tỉ lệ mua lại (SP)", "Trong tổng số đơn chứa SP này, bao nhiêu % là mua lại."),
    "avg_cart_position": ("Vị trí TB trong giỏ", "Trung bình SP được thêm vào giỏ ở vị trí thứ mấy. Thấp = được ưu tiên."),
    "total_items_bought": ("Tổng SP đã mua", "Tổng số sản phẩm khách đã mua qua tất cả đơn."),
    "total_reordered_items": ("Tổng SP mua lại", "Tổng số sản phẩm khách đã mua lại."),
    "avg_order_hour": ("Giờ đặt hàng TB", "Giờ trung bình mà khách thường đặt đơn."),
    "avg_days_between_orders": ("TB ngày giữa đơn", "Trung bình bao nhiêu ngày giữa 2 lần đặt hàng."),
    "avg_days_between_items": ("TB ngày giữa lần mua SP", "Trung bình bao nhiêu ngày giữa 2 lần mua cùng sản phẩm."),
    "weekend_ratio": ("Tỉ lệ mua cuối tuần", "% đơn hàng đặt vào thứ 7 và chủ nhật."),
    "peak_hour_ratio": ("Tỉ lệ giờ cao điểm", "% đơn hàng đặt vào khung giờ cao điểm (9–16h)."),
    "unique_products": ("Số SP khác nhau", "Khách đã mua bao nhiêu sản phẩm khác nhau."),
    "unique_buyers": ("Số người mua SP", "Bao nhiêu khách khác nhau đã mua SP này."),
    "unique_orders": ("Số đơn chứa SP", "SP này xuất hiện trong bao nhiêu đơn hàng."),
    "user_product_reorder_signal": ("Tín hiệu mua lại", "Khách này có xu hướng mua lại SP này không (0–1)."),
    "is_early_cart": ("Thêm sớm vào giỏ", "SP được thêm ở 3 vị trí đầu giỏ hàng = 1, ngược lại = 0."),
    "product_popularity_log": ("Độ phổ biến SP (log)", "Log(số lần SP được mua). Cao = sản phẩm phổ biến."),
    "user_engagement_log": ("Độ gắn bó user (log)", "Log(tổng số đơn × tổng SP). Cao = khách hàng tích cực."),
}

# Reference medians (from dataset) for explanation
MEDIANS = {
    "add_to_cart_order": 8, "order_dow": 3, "order_hour_of_day": 13,
    "days_since_prior_order": 7, "total_orders": 9, "total_items_bought": 94,
    "avg_basket_size": 9, "reorder_ratio": 0.43, "total_reordered_items": 37,
    "avg_order_hour": 13.0, "avg_days_between_orders": 14.5,
    "avg_days_between_items": 14.0, "weekend_ratio": 0.33, "peak_hour_ratio": 0.29,
    "unique_products": 50, "unique_departments": 11, "times_ordered": 5,
    "unique_buyers": 100, "unique_orders": 100, "reorder_rate": 0.5,
    "avg_cart_position": 8, "user_product_reorder_signal": 0.5,
    "is_early_cart": 1, "product_popularity_log": 4.5, "user_engagement_log": 4.5,
}

# Features where HIGHER value → MORE likely to reorder
POSITIVE_DIRECTION = {
    "total_orders", "reorder_ratio", "reorder_rate", "times_ordered",
    "total_reordered_items", "user_product_reorder_signal",
    "total_items_bought", "unique_products", "unique_departments",
    "product_popularity_log", "user_engagement_log", "avg_basket_size",
    "unique_buyers", "unique_orders", "peak_hour_ratio", "is_early_cart",
}

# Features where LOWER value → MORE likely to reorder
NEGATIVE_DIRECTION = {
    "add_to_cart_order", "avg_cart_position",
    "days_since_prior_order", "avg_days_between_orders", "avg_days_between_items",
}


# ──────────────────────────────────────────────
# Load Model Artifacts
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = MODELS_PATH / 'reorder_classifier.joblib'
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_feature_columns():
    path = MODELS_PATH / 'feature_columns.joblib'
    return joblib.load(path) if path.exists() else None

@st.cache_data
def load_metadata():
    path = MODELS_PATH / 'model_metadata.json'
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


model = load_model()
feature_columns = load_feature_columns()
metadata = load_metadata()

if model is None or feature_columns is None or metadata is None:
    st.error(
        "Model artifacts not found. "
        f"Please ensure files exist in `{MODELS_PATH}`:\n"
        "- `reorder_classifier.joblib`\n"
        "- `feature_columns.joblib`\n"
        "- `model_metadata.json`"
    )
    st.stop()


# ──────────────────────────────────────────────
# Page Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Reorder Prediction</h1>
    <p>Dự đoán xác suất khách hàng mua lại sản phẩm — sử dụng mô hình XGBClassifier với 25 đặc trưng</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ══════════════════════════════════════════════
# Section 1 — Model Overview
# ══════════════════════════════════════════════
st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

metrics = metadata["metrics"]

# Model info
info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Model", metadata.get("model_type", "XGBClassifier"))
info_col2.metric("Training Samples", f"{metadata.get('train_size', 1_107_693):,}")
info_col3.metric("Test Samples", f"{metadata.get('test_size', 276_924):,}")

st.write("")

# Performance metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
m2.metric("Precision", f"{metrics['precision']:.2%}")
m3.metric("Recall", f"{metrics['recall']:.2%}")
m4.metric("F1 Score", f"{metrics['f1']:.2%}")
m5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

st.markdown(
    '<div class="info-box">'
    f'Mô hình sử dụng <strong>{metadata.get("n_features", 25)} đặc trưng</strong> '
    f'được thiết kế từ dữ liệu gốc. Tỉ lệ sản phẩm được mua lại trong tập dữ liệu là '
    f'<strong>{metadata.get("class_balance", 0.5986):.1%}</strong>.'
    '</div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════
# Section 2 — Feature Importance
# ══════════════════════════════════════════════
st.markdown('<div class="section-title">Feature Importance — Top 15 đặc trưng quan trọng nhất</div>', unsafe_allow_html=True)

importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importances,
}).sort_values("importance", ascending=True)

top15 = feat_imp_df.tail(15).copy()

# Add Vietnamese labels for the chart
top15["label"] = top15["feature"].apply(
    lambda f: f"{FEATURE_VN.get(f, (f,))[0]}\n({f})" if f in FEATURE_VN else f
)

fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(
    x=top15["importance"],
    y=top15["label"],
    orientation="h",
    marker=dict(
        color=top15["importance"],
        colorscale=[[0, "#4a5568"], [0.5, "#667eea"], [1, "#a78bfa"]],
        line=dict(width=0),
    ),
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "Importance: %{x:.4f}<extra></extra>"
    ),
    customdata=list(zip(
        top15["feature"],
        top15["feature"].map(lambda f: FEATURE_VN.get(f, ("", ""))[1]),
    )),
))
fig_imp.update_layout(
    template="plotly_dark",
    height=520,
    margin=dict(l=10, r=30, t=20, b=30),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Feature Importance (Gain)", gridcolor="rgba(255,255,255,0.04)", zeroline=False),
    yaxis=dict(title="", gridcolor="rgba(255,255,255,0.04)"),
    font=dict(size=12, family="Inter, sans-serif"),
)
st.plotly_chart(fig_imp, use_container_width=True)


# ══════════════════════════════════════════════
# Section 3 — Interactive Prediction
# ══════════════════════════════════════════════
st.markdown('<div class="section-title">Interactive Prediction — Dự đoán tương tác</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-box">'
    'Điều chỉnh các thông số bên dưới rồi nhấn <strong>Dự đoán</strong> để xem '
    'mô hình ước tính xác suất khách hàng sẽ mua lại sản phẩm này. '
    'Các đặc trưng không hiển thị sẽ sử dụng giá trị trung vị (median) từ tập dữ liệu.'
    '</div>',
    unsafe_allow_html=True,
)

# Hidden defaults for features not shown in the form
HIDDEN_DEFAULTS = {
    "total_items_bought": 94, "total_reordered_items": 37,
    "avg_order_hour": 13.0, "avg_days_between_orders": 14.5,
    "avg_days_between_items": 14.0, "weekend_ratio": 0.33,
    "peak_hour_ratio": 0.29, "unique_products": 50,
    "unique_buyers": 100, "unique_orders": 100,
    "user_product_reorder_signal": 0.5, "is_early_cart": 1,
    "product_popularity_log": 4.5, "user_engagement_log": 4.5,
}

with st.form("prediction_form"):
    col_order, col_user, col_product = st.columns(3)

    with col_order:
        st.markdown('<div class="form-category">Bối cảnh đơn hàng</div>', unsafe_allow_html=True)
        add_to_cart_order = st.slider(
            "Vị trí trong giỏ hàng", min_value=1, max_value=50, value=5,
            help=FEATURE_VN["add_to_cart_order"][1],
        )
        order_dow = st.selectbox(
            "Ngày đặt hàng",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Chủ nhật", "Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy"][x],
            index=0,
            help=FEATURE_VN["order_dow"][1],
        )
        order_hour_of_day = st.slider(
            "Giờ đặt hàng", min_value=0, max_value=23, value=10,
            help=FEATURE_VN["order_hour_of_day"][1],
        )
        days_since_prior_order = st.slider(
            "Ngày kể từ đơn trước", min_value=0, max_value=30, value=7,
            help=FEATURE_VN["days_since_prior_order"][1],
        )

    with col_user:
        st.markdown('<div class="form-category">Hành vi khách hàng</div>', unsafe_allow_html=True)
        total_orders = st.slider(
            "Tổng đơn đã đặt", min_value=3, max_value=99, value=15,
            help=FEATURE_VN["total_orders"][1],
        )
        avg_basket_size = st.slider(
            "Giỏ hàng TB (số món/đơn)", min_value=1, max_value=50, value=10,
            help=FEATURE_VN["avg_basket_size"][1],
        )
        reorder_ratio = st.slider(
            "Tỉ lệ mua lại (user)", min_value=0.0, max_value=1.0, value=0.43, step=0.01,
            help=FEATURE_VN["reorder_ratio"][1],
        )
        unique_departments = st.slider(
            "Số ngành hàng", min_value=1, max_value=21, value=11,
            help=FEATURE_VN["unique_departments"][1],
        )

    with col_product:
        st.markdown('<div class="form-category">Đặc trưng sản phẩm</div>', unsafe_allow_html=True)
        times_ordered = st.slider(
            "Số lần SP được mua", min_value=1, max_value=100, value=5,
            help=FEATURE_VN["times_ordered"][1],
        )
        reorder_rate = st.slider(
            "Tỉ lệ mua lại (SP)", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help=FEATURE_VN["reorder_rate"][1],
        )
        avg_cart_position = st.slider(
            "Vị trí TB trong giỏ", min_value=1, max_value=50, value=8,
            help=FEATURE_VN["avg_cart_position"][1],
        )

    submitted = st.form_submit_button("Dự đoán xác suất mua lại", use_container_width=True)


# ── Run prediction & explain ──
if submitted:
    # Build feature dict
    form_values = {
        "add_to_cart_order": add_to_cart_order,
        "order_dow": order_dow,
        "order_hour_of_day": order_hour_of_day,
        "days_since_prior_order": days_since_prior_order,
        "total_orders": total_orders,
        "avg_basket_size": avg_basket_size,
        "reorder_ratio": reorder_ratio,
        "unique_departments": unique_departments,
        "times_ordered": times_ordered,
        "reorder_rate": reorder_rate,
        "avg_cart_position": avg_cart_position,
    }
    all_values = {**HIDDEN_DEFAULTS, **form_values}

    # Build DataFrame in correct column order
    input_df = pd.DataFrame([{col: all_values[col] for col in feature_columns}])

    # Predict
    proba = model.predict_proba(input_df)[0][1]
    proba_pct = proba * 100

    # Determine level
    if proba >= 0.7:
        level_class, interpretation, gauge_color = "result-high", "Khả năng cao sẽ mua lại", "#34d399"
    elif proba >= 0.4:
        level_class, interpretation, gauge_color = "result-mid", "Khả năng trung bình", "#fbbf24"
    else:
        level_class, interpretation, gauge_color = "result-low", "Khả năng thấp sẽ mua lại", "#f87171"

    st.markdown("---")

    # Results: probability card + gauge
    res_col1, res_col2 = st.columns([1, 1.3])

    with res_col1:
        st.markdown(
            f'<div class="result-card">'
            f'  <div class="result-label">Xác suất mua lại</div>'
            f'  <div class="result-value {level_class}">{proba_pct:.1f}%</div>'
            f'  <div class="result-note">{interpretation}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with res_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_pct,
            number=dict(suffix="%", font=dict(size=44, color="#e2e8f0")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#4a5568", dtick=20),
                bar=dict(color=gauge_color, thickness=0.3),
                bgcolor="rgba(30,30,46,0.5)",
                borderwidth=1,
                bordercolor="rgba(255,255,255,0.08)",
                steps=[
                    dict(range=[0, 40], color="rgba(248,113,113,0.08)"),
                    dict(range=[40, 70], color="rgba(251,191,36,0.08)"),
                    dict(range=[70, 100], color="rgba(52,211,153,0.08)"),
                ],
                threshold=dict(line=dict(color="#e2e8f0", width=2), thickness=0.75, value=proba_pct),
            ),
        ))
        fig_gauge.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=30, r=30, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Explanation: WHY this probability? ──
    st.markdown('<div class="section-title">Phân tích kết quả — Tại sao xác suất ở mức này?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Dưới đây là phân tích dựa trên <strong>mức độ quan trọng (feature importance)</strong> '
        'của từng đặc trưng trong mô hình, kết hợp với giá trị bạn đã nhập so với '
        '<strong>giá trị trung vị</strong> của toàn bộ tập dữ liệu.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Get top influential features that the user can control (form inputs only)
    imp_dict = dict(zip(feature_columns, importances))
    form_features_ranked = sorted(
        [(f, imp_dict.get(f, 0)) for f in form_values.keys()],
        key=lambda x: -x[1]
    )

    explanation_html = ""
    for feat, imp_score in form_features_ranked[:6]:
        val = form_values[feat]
        median = MEDIANS.get(feat, val)
        vn_name = FEATURE_VN.get(feat, (feat,))[0]
        vn_desc = FEATURE_VN.get(feat, ("", ""))[1]

        # Determine deviation
        if median != 0:
            pct_diff = (val - median) / abs(median) * 100
        else:
            pct_diff = 0

        # Determine if this pushes toward or away from reorder
        if feat in POSITIVE_DIRECTION:
            if val > median * 1.15:
                icon = "▲"
                effect = "tăng khả năng mua lại"
                color = "#34d399"
            elif val < median * 0.85:
                icon = "▼"
                effect = "giảm khả năng mua lại"
                color = "#f87171"
            else:
                icon = "●"
                effect = "ảnh hưởng trung tính"
                color = "#8892a4"
        elif feat in NEGATIVE_DIRECTION:
            if val < median * 0.85:
                icon = "▲"
                effect = "tăng khả năng mua lại"
                color = "#34d399"
            elif val > median * 1.15:
                icon = "▼"
                effect = "giảm khả năng mua lại"
                color = "#f87171"
            else:
                icon = "●"
                effect = "ảnh hưởng trung tính"
                color = "#8892a4"
        else:
            icon = "●"
            effect = "ảnh hưởng không xác định rõ"
            color = "#8892a4"

        # Format value display
        if isinstance(val, float) and val < 1:
            val_display = f"{val:.2f}"
            med_display = f"{median:.2f}"
        else:
            val_display = f"{val}"
            med_display = f"{median}"

        explanation_html += (
            f'<div class="factor-row">'
            f'  <div class="factor-icon" style="color:{color}">{icon}</div>'
            f'  <div class="factor-text">'
            f'    <strong>{vn_name}</strong> = {val_display} '
            f'    (trung vị: {med_display}) → '
            f'    <span style="color:{color}">{effect}</span>'
            f'  </div>'
            f'</div>'
        )

    st.markdown(explanation_html, unsafe_allow_html=True)

    # Summary interpretation
    if proba >= 0.7:
        summary = (
            "Kết luận: Mô hình đánh giá khách hàng này có **khả năng cao sẽ mua lại** sản phẩm. "
            "Các yếu tố chính đẩy tỉ lệ lên cao thường là tỉ lệ mua lại (user/product) cao "
            "và sản phẩm được thêm sớm vào giỏ hàng."
        )
    elif proba >= 0.4:
        summary = (
            "Kết luận: Mô hình đánh giá khách hàng này có **khả năng trung bình** mua lại sản phẩm. "
            "Một số yếu tố ủng hộ mua lại nhưng cũng có yếu tố kéo giảm. "
            "Chiến dịch nhắc nhở hoặc khuyến mãi có thể giúp tăng tỉ lệ chuyển đổi."
        )
    else:
        summary = (
            "Kết luận: Mô hình đánh giá khách hàng này **ít khả năng mua lại** sản phẩm. "
            "Có thể do khoảng cách giữa các đơn hàng quá dài, tỉ lệ mua lại thấp, "
            "hoặc sản phẩm chưa đủ phổ biến."
        )

    st.markdown(f"\n{summary}")

    # Expandable: all feature values
    with st.expander("Xem toàn bộ 25 đặc trưng đã gửi vào mô hình"):
        feat_display = pd.DataFrame({
            "Đặc trưng": feature_columns,
            "Tên tiếng Việt": [FEATURE_VN.get(c, (c,))[0] for c in feature_columns],
            "Giá trị": [all_values[c] for c in feature_columns],
            "Nguồn": [
                "Nhập từ form" if c in form_values else "Giá trị mặc định (median)"
                for c in feature_columns
            ],
        })
        st.dataframe(feat_display, use_container_width=True, hide_index=True)


# ── Footer ──
st.markdown(
    '<div class="page-footer">'
    'Reorder Prediction · XGBClassifier · Instacart E-Commerce Intelligence'
    '</div>',
    unsafe_allow_html=True,
)
