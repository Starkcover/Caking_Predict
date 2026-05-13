import subprocess
import streamlit as st

result = subprocess.run(["pip", "list"], capture_output=True, text=True)
st.code(result.stdout)
st.stop()  # stops rest of app from running
"""
app.py
------
Main Streamlit entry point for the Powder Caking Prediction dashboard.
All prediction logic, feature construction, and plot calls are wired
directly to the functions extracted from caking-prediction.ipynb.

Run locally:
    streamlit run app.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ── local imports ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from physics import (
    build_single_row,
    generate_caking_dataset,
    clean_dataset,
    engineer_physics_features,
    ALL_FEATURE_NAMES,
    CAKING_THRESHOLD_PA,
)
from plots import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_physics_relationships,
    plot_boxplots_by_class,
    plot_regression_evaluation,
    plot_classification_evaluation,
    plot_partial_dependence,
    plot_model_comparison,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Powder Caking Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS = Path(__file__).parent / "models" / "artifacts"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid #3498db;
        margin-bottom: 10px;
    }
    .caked-card   { border-left-color: #e74c3c; background: #fff5f5; }
    .free-card    { border-left-color: #2ecc71; background: #f0fff4; }
    .warn-card    { border-left-color: #f39c12; background: #fffbf0; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 4px;
        margin-bottom: 12px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Artifact loader (cached once per session) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    if not (ARTIFACTS / "best_reg_model.pkl").exists():
        return None, None, None, None, None
    reg_model  = joblib.load(ARTIFACTS / "best_reg_model.pkl")
    clf_model  = joblib.load(ARTIFACTS / "best_clf_model.pkl")
    scaler     = joblib.load(ARTIFACTS / "scaler_reg.pkl")
    feat_cols  = json.load(open(ARTIFACTS / "feature_cols.json"))
    metrics    = json.load(open(ARTIFACTS / "training_metrics.json"))
    return reg_model, clf_model, scaler, feat_cols, metrics


@st.cache_data(show_spinner="Generating dataset…")
def get_dataset():
    df = generate_caking_dataset(n_samples=1200, random_state=42)
    return clean_dataset(df)


reg_model, clf_model, scaler, feat_cols, metrics = load_artifacts()
models_ready = reg_model is not None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — navigation + inputs
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/test-tube.png", width=60)
    st.title("Powder Caking\nPredictor")
    st.caption("Physics-Informed ML · caking-prediction.ipynb")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📊 EDA & Physics", "📈 Model Results",
         "🧠 Explainability", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()

    # ── Input sliders (used by Predict page) ──────────────────────────────
    st.markdown('<div class="section-header">Process Conditions</div>',
                unsafe_allow_html=True)

    D50_um       = st.slider("D50 — Median Diameter (µm)",       10.0, 500.0, 100.0, 1.0)
    span_psd     = st.slider("PSD Span  (D90-D10)/D50  [–]",      0.5,   3.0,   1.5, 0.1)
    BET_m2g      = st.slider("BET Surface Area (m²/g)",            0.1,  20.0,   5.0, 0.1)
    shape_factor = st.slider("Shape Factor  [–]  (1=sphere)",      0.4,   1.0,   0.8, 0.05)
    Temp_C       = st.slider("Storage Temperature (°C)",          15.0,  60.0,  25.0, 0.5)
    RH_pct       = st.slider("Relative Humidity (%)",             20.0,  95.0,  60.0, 1.0)
    time_hr      = st.slider("Storage Time (hours)",               1.0, 720.0, 168.0, 1.0)
    pressure_kPa = st.slider("Compaction Pressure (kPa)",          0.0, 200.0,  10.0, 1.0)
    CRH_pct      = st.slider("Critical Relative Humidity (%)",    40.0,  85.0,  65.0, 1.0)
    Tg_C         = st.slider("Glass Transition Temp Tg (°C)",    -20.0,  80.0,  30.0, 1.0)

    st.divider()
    if not models_ready:
        st.error("⚠️ No trained models found.\nRun:  `python src/train.py`")
    else:
        st.success(f"✅ Models loaded\n\nBest reg: **{metrics['best_reg']}**\n\n"
                   f"Best clf: **{metrics['best_clf']}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("Caking Strength Predictor")
    st.caption(
        "Adjust the sliders in the sidebar, then read off the real-time prediction below."
    )

    if not models_ready:
        st.warning("Train models first: `python src/train.py`")
        st.stop()

    # ── Build feature row ──────────────────────────────────────────────────
    X_input = build_single_row(
        D50_um=D50_um, span_psd=span_psd, BET_m2g=BET_m2g,
        shape_factor=shape_factor, Temp_C=Temp_C, RH_pct=RH_pct,
        time_hr=time_hr, pressure_kPa=pressure_kPa,
        CRH_pct=CRH_pct, Tg_C=Tg_C,
    )
    X_scaled = scaler.transform(X_input[feat_cols])

    pred_strength = float(reg_model.predict(X_scaled)[0])
    pred_class    = int(clf_model.predict(X_scaled)[0])
    pred_prob     = float(clf_model.predict_proba(X_scaled)[0][1])

    # ── KPI cards ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        css_class = "caked-card" if pred_strength > CAKING_THRESHOLD_PA else "free-card"
        st.markdown(
            f'<div class="metric-card {css_class}">'
            f'<div style="font-size:0.85rem;color:#666">Predicted Caking Strength</div>'
            f'<div style="font-size:2rem;font-weight:700">{pred_strength:.1f} Pa</div>'
            f'<div style="font-size:0.75rem;color:#999">Threshold: {CAKING_THRESHOLD_PA} Pa</div>'
            f'</div>', unsafe_allow_html=True
        )

    with c2:
        label  = "🔴 CAKED" if pred_class == 1 else "🟢 FREE-FLOWING"
        css_cl = "caked-card" if pred_class == 1 else "free-card"
        st.markdown(
            f'<div class="metric-card {css_cl}">'
            f'<div style="font-size:0.85rem;color:#666">Classification</div>'
            f'<div style="font-size:2rem;font-weight:700">{label}</div>'
            f'<div style="font-size:0.75rem;color:#999">Model: {metrics["best_clf"]}</div>'
            f'</div>', unsafe_allow_html=True
        )

    with c3:
        bar_color = "#e74c3c" if pred_prob > 0.5 else "#2ecc71"
        st.markdown(
            f'<div class="metric-card warn-card">'
            f'<div style="font-size:0.85rem;color:#666">Caking Probability</div>'
            f'<div style="font-size:2rem;font-weight:700;color:{bar_color}">'
            f'{pred_prob*100:.1f}%</div>'
            f'<div style="font-size:0.75rem;color:#999">Calibrated from predict_proba</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.divider()

    # ── Probability gauge ──────────────────────────────────────────────────
    st.markdown("#### Caking Risk Gauge")
    col_g, col_info = st.columns([2, 1])
    with col_g:
        fig_g, ax_g = plt.subplots(figsize=(7, 1.2))
        ax_g.barh(0, 100, color="#eee", height=0.4)
        ax_g.barh(0, pred_prob * 100,
                  color="#e74c3c" if pred_prob > 0.5 else "#2ecc71", height=0.4)
        ax_g.axvline(50, color="gray", linestyle="--", lw=1.2)
        ax_g.set_xlim(0, 100)
        ax_g.set_yticks([])
        ax_g.set_xlabel("Caking Probability (%)")
        ax_g.text(pred_prob * 100 + 1, 0, f"{pred_prob*100:.1f}%",
                  va='center', fontsize=9, fontweight='bold')
        for spine in ax_g.spines.values():
            spine.set_visible(False)
        fig_g.patch.set_alpha(0)
        st.pyplot(fig_g, use_container_width=True)

    with col_info:
        st.markdown("**Physics drivers active:**")
        if RH_pct > CRH_pct:
            st.markdown(f"- 💧 RH ({RH_pct:.0f}%) > CRH ({CRH_pct:.0f}%) → liquid bridges")
        if Temp_C > Tg_C:
            st.markdown(f"- 🌡️ T ({Temp_C:.0f}°C) > Tg ({Tg_C:.0f}°C) → sintering active")
        if time_hr > 168:
            st.markdown(f"- ⏱️ Long storage ({time_hr:.0f} h > 1 week)")
        if D50_um < 50:
            st.markdown(f"- 🔬 Fine powder (D50={D50_um:.0f} µm) → high contact area")

    st.divider()

    # ── Feature vector table ───────────────────────────────────────────────
    with st.expander("🔍 Inspect full 25-feature vector sent to model"):
        feat_display = X_input.T.copy()
        feat_display.columns = ["Value"]
        feat_display["Value"] = feat_display["Value"].apply(lambda x: f"{x:.5g}")
        st.dataframe(feat_display, use_container_width=True)

    # ── SHAP waterfall for this prediction ────────────────────────────────
    if SHAP_AVAILABLE and st.button("🔬 Explain this prediction (SHAP waterfall)"):
        with st.spinner("Computing SHAP values…"):
            explainer  = shap.TreeExplainer(reg_model)
            shap_vals  = explainer.shap_values(X_scaled)
            fig_shap, ax_shap = plt.subplots(figsize=(9, 5))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=X_scaled[0],
                    feature_names=feat_cols,
                ),
                show=False, max_display=15,
            )
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & PHYSICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Physics":
    st.title("Exploratory Data Analysis")
    st.caption(
        "Plots reproduced from notebook Sections 4.2 – 4.6 on the "
        "1 200-sample physics-based synthetic dataset."
    )

    df = get_dataset()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Target Distribution",
        "Physics Relationships",
        "Correlation Heatmap",
        "Boxplots by Class",
    ])

    with tab1:
        st.markdown("### Target Variable Analysis  *(Section 4.2)*")
        st.pyplot(plot_target_distribution(df, threshold=CAKING_THRESHOLD_PA),
                  use_container_width=True)
        st.info(
            f"**Dataset stats:** n=1200 · "
            f"caking_strength_Pa in [{df['caking_strength_Pa'].min():.0f}, "
            f"{df['caking_strength_Pa'].max():.0f}] Pa · "
            f"Caked: {df['is_caked'].mean()*100:.1f}%  |  "
            f"Free-flowing: {(1-df['is_caked']).mean()*100:.1f}%"
        )

    with tab2:
        st.markdown("### Caking Strength vs Key Physics Variables  *(Section 4.5)*")
        st.pyplot(plot_physics_relationships(df), use_container_width=True)
        st.caption(
            "Green = free-flowing (≤800 Pa) · Red = caked (>800 Pa). "
            "Shapes confirm: moisture sigmoid, log-time, Arrhenius-T, "
            "exponential D50 decay, T–Tg step."
        )

    with tab3:
        st.markdown("### Feature Correlation Matrix  *(Section 4.4)*")
        st.pyplot(plot_correlation_heatmap(df), use_container_width=True)

    with tab4:
        st.markdown("### Feature Distributions by Class  *(Section 4.6)*")
        st.pyplot(plot_boxplots_by_class(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Results":
    st.title("Model Evaluation & Comparison")

    if not models_ready:
        st.warning("Run `python src/train.py` first.")
        st.stop()

    reg_metrics = metrics["regression"]
    clf_metrics = metrics["classification"]

    reg_df = (pd.DataFrame(reg_metrics).T
              .rename(columns={
                  'Test_R2': 'Test R²', 'Test_RMSE': 'Test RMSE (Pa)',
                  'Test_MAE': 'Test MAE (Pa)', 'Test_MAPE': 'Test MAPE (%)',
                  'CV_RMSE_mean': 'CV RMSE mean', 'CV_RMSE_std': 'CV RMSE std',
              })
              .sort_values('Test R²', ascending=False))

    clf_df = (pd.DataFrame(clf_metrics).T
              .rename(columns={
                  'CV_F1_mean': 'CV F1 mean', 'CV_F1_std': 'CV F1 std',
                  'ROC_AUC': 'ROC-AUC',
              })
              .sort_values('F1', ascending=False))

    tab_r, tab_c, tab_cmp = st.tabs(
        ["Regression", "Classification", "Side-by-Side Comparison"]
    )

    with tab_r:
        st.markdown("### Regression — Caking Strength (Pa)  *(Section 10)*")
        st.dataframe(
            reg_df.style
            .format({
                'Test R²': '{:.4f}', 'Test RMSE (Pa)': '{:.2f}',
                'Test MAE (Pa)': '{:.2f}', 'Test MAPE (%)': '{:.2f}',
                'CV RMSE mean': '{:.2f}', 'CV RMSE std': '{:.2f}',
            })
            .highlight_max(subset=['Test R²'], color='#d4edda')
            .highlight_min(subset=['Test RMSE (Pa)', 'Test MAE (Pa)'], color='#d4edda'),
            use_container_width=True,
        )
        best = reg_df.index[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model",     best)
        col2.metric("Best R²",        f"{reg_df.loc[best,'Test R²']:.4f}")
        col3.metric("Best RMSE",      f"{reg_df.loc[best,'Test RMSE (Pa)']:.2f} Pa")

    with tab_c:
        st.markdown("### Classification — Caked vs Free-Flowing  *(Section 10)*")
        st.dataframe(
            clf_df.style
            .format({
                'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}',
                'F1': '{:.4f}', 'ROC-AUC': '{:.4f}',
                'CV F1 mean': '{:.4f}', 'CV F1 std': '{:.4f}',
            })
            .highlight_max(subset=['F1', 'ROC-AUC'], color='#d4edda'),
            use_container_width=True,
        )
        best_c = clf_df.index[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model",  best_c)
        col2.metric("Best F1",     f"{clf_df.loc[best_c,'F1']:.4f}")
        col3.metric("Best AUC",    f"{clf_df.loc[best_c,'ROC-AUC']:.4f}")

    with tab_cmp:
        st.markdown("### Final Model Comparison *(Section 12)*")
        st.pyplot(plot_model_comparison(reg_df, clf_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.title("Model Explainability")
    st.caption("SHAP + Partial Dependence Plots — Sections 11 of the notebook.")

    if not models_ready:
        st.warning("Run `python src/train.py` first.")
        st.stop()

    # Rebuild test set for SHAP / PDP
    @st.cache_data(show_spinner="Preparing test set for explainability…")
    def get_test_set():
        df = get_dataset()
        df_eng = engineer_physics_features(df)
        fc = [c for c in df_eng.columns if c not in ('caking_strength_Pa','is_caked')]
        X_full = df_eng[fc]; y_r = df_eng['caking_strength_Pa']
        y_c = df_eng['is_caked']
        from sklearn.model_selection import train_test_split
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
            X_full, y_r, y_c, test_size=0.20, random_state=42, stratify=y_c)
        X_te_s = pd.DataFrame(scaler.transform(X_te), columns=fc)
        return X_te_s, yr_te, yc_te, X_te, fc

    X_te_s, yr_te, yc_te, X_te, fc = get_test_set()

    tab_shap, tab_pdp, tab_err = st.tabs(
        ["SHAP Summary", "Partial Dependence", "Error Analysis"]
    )

    with tab_shap:
        st.markdown("### SHAP Feature Importance  *(Section 11)*")
        if SHAP_AVAILABLE:
            with st.spinner("Computing SHAP values (TreeExplainer)…"):
                explainer  = shap.TreeExplainer(reg_model)
                shap_vals  = explainer.shap_values(X_te_s)

            col_s, col_b = st.columns(2)
            with col_s:
                st.markdown("**Beeswarm (summary) plot**")
                fig_sw, _ = plt.subplots(figsize=(8, 6))
                shap.summary_plot(shap_vals, X_te_s,
                                  feature_names=fc, show=False)
                plt.title(f"SHAP Summary — {metrics['best_reg']}",
                          fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_sw, use_container_width=True)
                plt.close("all")

            with col_b:
                st.markdown("**Bar plot — mean |SHAP|**")
                shap_mean  = np.abs(shap_vals).mean(axis=0)
                shap_series = (pd.Series(shap_mean, index=fc)
                               .sort_values(ascending=True).tail(15))
                fig_bar, ax_bar = plt.subplots(figsize=(7, 6))
                shap_series.plot(kind='barh', ax=ax_bar, color='steelblue')
                ax_bar.set_title("Top 15 SHAP Importances (Mean |SHAP|)",
                                 fontweight='bold')
                ax_bar.set_xlabel("Mean |SHAP value|")
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)
                plt.close("all")

            top_feat = shap_series.index[-1]
            st.info(
                f"🔬 **SHAP Key Finding:** `{top_feat}` is the most influential feature.\n\n"
                "Physics interpretation: High RH above CRH → liquid bridge formation "
                "→ rapid caking."
            )
        else:
            st.warning("Install SHAP: `pip install shap`")
            st.markdown("Showing permutation importance as fallback…")
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(reg_model, X_te_s, yr_te,
                                          n_repeats=10, random_state=42)
            perm_s = (pd.Series(perm.importances_mean, index=fc)
                      .sort_values(ascending=True).tail(15))
            fig_p, ax_p = plt.subplots(figsize=(8, 6))
            perm_s.plot(kind='barh', ax=ax_p, color='darkorange')
            ax_p.set_title("Permutation Importance (SHAP fallback)",
                           fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close("all")

    with tab_pdp:
        st.markdown("### Partial Dependence Plots — Physics Verification  *(Section 11)*")
        st.caption(
            "Expected physics shapes: RH → sigmoidal · time → log-monotone · "
            "T → Arrhenius rise · D50 → exponential decay · T−Tg → step at 0."
        )
        with st.spinner("Computing PDP…"):
            fig_pdp = plot_partial_dependence(reg_model, X_te_s, fc)
        st.pyplot(fig_pdp, use_container_width=True)
        plt.close("all")

    with tab_err:
        st.markdown("### Regression Error Analysis  *(Section 10)*")
        y_pred_best = reg_model.predict(X_te_s)
        fig_err = plot_regression_evaluation(
            metrics['best_reg'], yr_te, y_pred_best, X_te
        )
        st.pyplot(fig_err, use_container_width=True)
        plt.close("all")

        st.markdown("### Classification Evaluation  *(Section 10)*")
        yp_clf  = clf_model.predict(X_te_s)
        yp_prob = clf_model.predict_proba(X_te_s)[:, 1]
        fig_clf = plot_classification_evaluation(
            metrics['best_clf'], yc_te, yp_clf, yp_prob
        )
        st.pyplot(fig_clf, use_container_width=True)
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About this Project")

    st.markdown("""
## Powder Caking Prediction — Physics-Informed ML

This app is the Streamlit deployment of **caking-prediction.ipynb**, a
full end-to-end physics-informed ML pipeline for predicting powder caking.

### What is powder caking?
Powder caking is the spontaneous agglomeration of free-flowing particles
into rigid lumps during storage.  It costs industry billions annually
across food science, pharma, fertilisers, and bulk chemicals.

### Governing Physics (from notebook Section 1)

| Mechanism | Governing equation |
|---|---|
| Moisture (Kelvin) | ln(p/p₀) = 2γVₘ / rRT |
| JKR adhesion | F_pull = ³⁄₂ π W R* |
| Sintering rate | dNb/dt = A · exp(−Ea/RT) · σⁿ |

### Caking strength scale
| Range | Interpretation |
|---|---|
| 0 – 400 Pa | Free-flowing |
| 400 – 800 Pa | Mildly caked |
| 800 – 1500 Pa | Moderately caked |
| > 1500 Pa | Severely caked |

**Classification threshold: 800 Pa** (Johanson 2009)

### Pipeline summary
1. **Data** — 1 200 physics-based synthetic samples (Section 3)
2. **Cleaning** — domain clipping, IQR outlier detection (Section 5)
3. **Feature engineering** — 7 physics features: T/Tg, Kelvin ratio,
   JKR proxy, Arrhenius-time, RH×BET/D50, SSA, PSD×moisture (Section 6)
4. **Regression** — 9 models incl. RF, GBM, XGBoost (Section 7)
5. **Classification** — 7 models incl. RF, SVC, XGBoost (Section 7)
6. **PINN** — SiLU MLP + sintering physics loss (Section 8)
7. **Tuning** — RandomizedSearchCV 50 trials (Section 9)
8. **Explainability** — SHAP TreeExplainer + PDP (Section 11)

### References
- Johanson (2009) *Measurement and prediction of caking in bulk solids*
- Raissi et al. (2019) *Physics-informed neural networks* — J. Comp. Phys.
- Teunou & Fitzpatrick (1999) *Effect of T and RH on food powder flowability*
- Lundberg & Lee (2017) *A unified approach to interpreting model predictions*
""")
