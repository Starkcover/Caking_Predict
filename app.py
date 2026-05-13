"""
app.py  —  Powder Caking Prediction · Streamlit App
=====================================================
Cloud-hardened version:
  • torch / shap / xgboost are truly optional — missing them never crashes the app
  • A Debug page surfaces exactly what is/isn't installed at runtime
  • All model artifacts are loaded lazily with clear error messages
  • matplotlib backend forced to Agg before any import of pyplot
"""

# ── Force headless matplotlib BEFORE any pyplot import ─────────────────────
import matplotlib
matplotlib.use("Agg")

import json
import warnings
import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Optional heavy deps — never crash if missing ────────────────────────────
try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import sklearn                          # noqa: F401
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import xgboost                          # noqa: F401
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Local src on path ───────────────────────────────────────────────────────
SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

try:
    from physics import (
        build_single_row, generate_caking_dataset, clean_dataset,
        engineer_physics_features, ALL_FEATURE_NAMES, CAKING_THRESHOLD_PA,
    )
    PHYSICS_OK = True
except Exception as _physics_err:
    PHYSICS_OK = False
    _physics_err_msg = str(_physics_err)

try:
    import matplotlib.pyplot as plt
    import seaborn                          # noqa: F401
    from plots import (
        plot_target_distribution, plot_correlation_heatmap,
        plot_physics_relationships, plot_boxplots_by_class,
        plot_regression_evaluation, plot_classification_evaluation,
        plot_partial_dependence, plot_model_comparison,
        plot_pinn_training, plot_pinn_vs_models,
    )
    PLOTS_OK = True
except Exception as _plots_err:
    PLOTS_OK = False
    _plots_err_msg = str(_plots_err)

CORE_OK = JOBLIB_OK and SKLEARN_OK and PHYSICS_OK and PLOTS_OK

# ── App config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Powder Caking Predictor", page_icon="🧪",
    layout="wide", initial_sidebar_state="expanded",
)

ARTIFACTS = Path(__file__).parent / "models" / "artifacts"

st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:10px;padding:16px 20px;
  border-left:5px solid #3498db;margin-bottom:10px}
.caked-card{border-left-color:#e74c3c;background:#fff5f5}
.free-card {border-left-color:#2ecc71;background:#f0fff4}
.warn-card {border-left-color:#f39c12;background:#fffbf0}
.pinn-card {border-left-color:#9b59b6;background:#f8f0ff}
.section-header{font-size:1.1rem;font-weight:700;color:#2c3e50;
  border-bottom:2px solid #3498db;padding-bottom:4px;margin-bottom:12px}
div[data-testid="stMetricValue"]{font-size:1.6rem!important}
</style>
""", unsafe_allow_html=True)


# ── Artifact loaders ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    if not JOBLIB_OK:
        return None, None, None, None, None
    if not (ARTIFACTS / "best_reg_model.pkl").exists():
        return None, None, None, None, None
    try:
        return (
            joblib.load(ARTIFACTS / "best_reg_model.pkl"),
            joblib.load(ARTIFACTS / "best_clf_model.pkl"),
            joblib.load(ARTIFACTS / "scaler_reg.pkl"),
            json.load(open(ARTIFACTS / "feature_cols.json")),
            json.load(open(ARTIFACTS / "training_metrics.json")),
        )
    except Exception:
        return None, None, None, None, None


@st.cache_resource(show_spinner="Loading PINN weights…")
def load_pinn(input_dim: int):
    wp = ARTIFACTS / "pinn_weights.pt"
    sp = ARTIFACTS / "pinn_target_scaler.pkl"
    if not (wp.exists() and sp.exists() and TORCH_AVAILABLE and JOBLIB_OK):
        return None, None
    try:
        from pinn import CakingPINN
        m = CakingPINN(input_dim=input_dim)
        m.load_state_dict(torch.load(wp, map_location="cpu"))
        m.eval()
        return m, joblib.load(sp)
    except Exception:
        return None, None


@st.cache_data(show_spinner="Generating dataset…")
def get_dataset():
    df = generate_caking_dataset(1200, 42)
    return clean_dataset(df)


@st.cache_data(show_spinner="Preparing test split…")
def get_test_set(_scaler):
    from sklearn.model_selection import train_test_split
    df = get_dataset()
    df_eng = engineer_physics_features(df)
    fc = [c for c in df_eng.columns if c not in ("caking_strength_Pa", "is_caked")]
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        df_eng[fc], df_eng["caking_strength_Pa"], df_eng["is_caked"],
        test_size=0.20, random_state=42, stratify=df_eng["is_caked"],
    )
    X_te_s = pd.DataFrame(_scaler.transform(X_te), columns=fc)
    return (X_te_s,
            yr_te.reset_index(drop=True),
            yc_te.reset_index(drop=True),
            X_te.reset_index(drop=True),
            fc)


reg_model, clf_model, scaler, feat_cols, metrics = load_artifacts()
models_ready = reg_model is not None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/test-tube.png", width=60)
    st.title("Powder Caking\nPredictor")
    st.caption("Physics-Informed ML · caking-prediction.ipynb")
    st.divider()

    page = st.radio("Navigate", [
        "🔮 Predict", "📊 EDA & Physics",
        "📈 Model Results", "🧠 Explainability",
        "🔧 Debug", "ℹ️ About",
    ], label_visibility="collapsed")
    st.divider()

    st.markdown('<div class="section-header">Process Conditions</div>',
                unsafe_allow_html=True)
    D50_um       = st.slider("D50 — Median Diameter (µm)",      10.0, 500.0, 100.0, 1.0)
    span_psd     = st.slider("PSD Span (D90-D10)/D50 [–]",       0.5,   3.0,   1.5, 0.1)
    BET_m2g      = st.slider("BET Surface Area (m²/g)",           0.1,  20.0,   5.0, 0.1)
    shape_factor = st.slider("Shape Factor [–] (1=sphere)",       0.4,   1.0,   0.8, 0.05)
    Temp_C       = st.slider("Storage Temperature (°C)",         15.0,  60.0,  25.0, 0.5)
    RH_pct       = st.slider("Relative Humidity (%)",            20.0,  95.0,  60.0, 1.0)
    time_hr      = st.slider("Storage Time (hours)",              1.0, 720.0, 168.0, 1.0)
    pressure_kPa = st.slider("Compaction Pressure (kPa)",         0.0, 200.0,  10.0, 1.0)
    CRH_pct      = st.slider("Critical Relative Humidity (%)",   40.0,  85.0,  65.0, 1.0)
    Tg_C         = st.slider("Glass Transition Temp Tg (°C)",   -20.0,  80.0,  30.0, 1.0)
    st.divider()

    if not CORE_OK:
        st.error("Core deps missing — see Debug page")
    elif not models_ready:
        st.warning("Models not loaded — see Debug page")
    else:
        pinn_ok = (ARTIFACTS / "pinn_weights.pt").exists() and TORCH_AVAILABLE
        st.success(
            f"**Best reg:** {metrics['best_reg']}\n\n"
            f"**Best clf:** {metrics['best_clf']}\n\n"
            + ("✅ PINN ready" if pinn_ok else "⚠️ PINN not available")
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEBUG  (always works even if core deps are broken)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔧 Debug":
    st.title("🔧 Environment Diagnostics")
    st.caption("Use this page to diagnose Streamlit Cloud deployment issues.")

    # ── Runtime info ──────────────────────────────────────────────────────
    import platform
    col1, col2 = st.columns(2)
    col1.info(f"**Python:** {sys.version.split()[0]}  \n**Platform:** {platform.system()} {platform.machine()}")
    col2.info(f"**Working dir:** `{Path.cwd()}`  \n**src path:** `{SRC}`  \n**artifacts:** `{ARTIFACTS}`")

    st.divider()

    def _row(label, ok, detail=""):
        icon = "✅" if ok else "❌"
        color = "green" if ok else "red"
        st.markdown(
            f'<div style="padding:8px 12px;margin:4px 0;border-radius:6px;'
            f'background:{"#f0fff4" if ok else "#fff5f5"};'
            f'border-left:4px solid {"#2ecc71" if ok else "#e74c3c"}">'
            f'<b>{icon} {label}</b>'
            + (f'<br><span style="font-size:.8rem;color:#555">{detail}</span>' if detail else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    # ── Core packages ──────────────────────────────────────────────────────
    st.markdown("### Core Packages")

    def _ver(pkg):
        m = __import__(pkg)
        return getattr(m, "__version__", "?")

    for pkg, import_name in [
        ("numpy",       "numpy"),
        ("pandas",      "pandas"),
        ("joblib",      "joblib"),
        ("scikit-learn","sklearn"),
        ("scipy",       "scipy"),
        ("matplotlib",  "matplotlib"),
        ("seaborn",     "seaborn"),
        ("streamlit",   "streamlit"),
    ]:
        try:
            v = _ver(import_name)
            _row(pkg, True, f"version {v}")
        except Exception as e:
            _row(pkg, False, str(e))

    # ── Optional packages ──────────────────────────────────────────────────
    st.markdown("### Optional Packages")
    for pkg, import_name, note in [
        ("xgboost", "xgboost", "Used in regression + classification"),
        ("torch",   "torch",   "Required for PINN"),
        ("shap",    "shap",    "Required for SHAP explainability"),
    ]:
        try:
            v = _ver(import_name)
            _row(pkg, True, f"version {v} — {note}")
        except Exception as e:
            _row(pkg, False, f"NOT INSTALLED (optional) — {note}")

    # ── Local modules ──────────────────────────────────────────────────────
    st.markdown("### Local Modules")
    _row("src/physics.py", PHYSICS_OK,
         "generate_caking_dataset, engineer_physics_features, build_single_row"
         if PHYSICS_OK else _physics_err_msg if not PHYSICS_OK else "")
    _row("src/plots.py",   PLOTS_OK,
         "All plot helpers loaded"
         if PLOTS_OK else _plots_err_msg if not PLOTS_OK else "")

    # ── Artifact files ──────────────────────────────────────────────────────
    st.markdown("### Model Artifacts")
    for fname in ["scaler_reg.pkl", "best_reg_model.pkl",
                  "best_clf_model.pkl", "feature_cols.json",
                  "training_metrics.json"]:
        p = ARTIFACTS / fname
        if p.exists():
            _row(fname, True, f"{p.stat().st_size // 1024} KB")
        else:
            _row(fname, False, f"Missing — run `python src/train.py` locally then commit")

    # ── Model load round-trip ───────────────────────────────────────────────
    st.markdown("### Model Load Round-Trip")
    if JOBLIB_OK and (ARTIFACTS / "best_reg_model.pkl").exists():
        try:
            reg_t = joblib.load(ARTIFACTS / "best_reg_model.pkl")
            clf_t = joblib.load(ARTIFACTS / "best_clf_model.pkl")
            sc_t  = joblib.load(ARTIFACTS / "scaler_reg.pkl")
            fc_t  = json.load(open(ARTIFACTS / "feature_cols.json"))
            _row("Model load", True,
                 f"reg={type(reg_t).__name__}  clf={type(clf_t).__name__}  "
                 f"scaler={type(sc_t).__name__}  features={len(fc_t)}")
        except Exception as e:
            _row("Model load", False, str(e))
    else:
        _row("Model load", False, "Skipped — joblib missing or artifacts not found")

    # ── Quick prediction smoke test ─────────────────────────────────────────
    st.markdown("### Prediction Smoke Test")
    if models_ready and PHYSICS_OK:
        try:
            row = build_single_row(100, 1.5, 5.0, 0.8, 25.0, 70.0, 168.0, 10.0, 65.0, 30.0)
            xs  = scaler.transform(row[feat_cols])
            p   = float(reg_model.predict(xs)[0])
            _row("Prediction", True, f"build_single_row → {p:.1f} Pa (expected 0–2000 Pa)")
        except Exception as e:
            _row("Prediction", False, str(e))
    else:
        _row("Prediction", False, "Skipped — models or physics not ready")

    st.divider()
    st.markdown("""
**Common fixes for Streamlit Cloud:**

| Error | Fix |
|---|---|
| `No module named 'joblib'` | Add `runtime.txt` containing `3.11` to repo root |
| `No module named 'sklearn'` | Same — Python 3.14 has no sklearn wheels |
| `Model load failed` | Commit `.pkl` files; they're gitignored by default — remove from `.gitignore` |
| `physics.py failed` | Check `src/` folder is committed; not just the zip |
| Slow cold start | Normal — Streamlit Cloud takes 60–90 s on first boot |
""")


# ══════════════════════════════════════════════════════════════════════════════
# All other pages require CORE_OK
# ══════════════════════════════════════════════════════════════════════════════
elif not CORE_OK:
    st.error("Core dependencies are not installed correctly. Go to the **🔧 Debug** page for details.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("Caking Strength Predictor")
    st.caption("Adjust sliders in the sidebar — predictions update instantly.")
    if not models_ready:
        st.warning("Models not loaded. Go to **🔧 Debug** to diagnose.")
        st.stop()

    X_input  = build_single_row(D50_um, span_psd, BET_m2g, shape_factor,
                                Temp_C, RH_pct, time_hr, pressure_kPa, CRH_pct, Tg_C)
    X_scaled = scaler.transform(X_input[feat_cols])

    pred_str  = float(reg_model.predict(X_scaled)[0])
    pred_cls  = int(clf_model.predict(X_scaled)[0])
    pred_prob = float(clf_model.predict_proba(X_scaled)[0][1])

    pinn_m, pinn_ts = load_pinn(len(feat_cols))
    pinn_val = None
    if pinn_m is not None:
        from pinn import predict_pinn
        pinn_val = float(predict_pinn(pinn_m, pinn_ts, X_scaled)[0])

    cols = st.columns(4 if pinn_val is not None else 3)
    s_css = "caked-card" if pred_str > CAKING_THRESHOLD_PA else "free-card"
    cols[0].markdown(
        f'<div class="metric-card {s_css}">'
        f'<span style="font-size:.85rem;color:#666">ML Caking Strength</span><br>'
        f'<span style="font-size:2rem;font-weight:700">{pred_str:.1f} Pa</span><br>'
        f'<span style="font-size:.75rem;color:#999">Threshold {CAKING_THRESHOLD_PA} Pa</span>'
        f'</div>', unsafe_allow_html=True)

    c_css = "caked-card" if pred_cls == 1 else "free-card"
    lbl   = "🔴 CAKED" if pred_cls == 1 else "🟢 FREE-FLOWING"
    cols[1].markdown(
        f'<div class="metric-card {c_css}">'
        f'<span style="font-size:.85rem;color:#666">Classification</span><br>'
        f'<span style="font-size:2rem;font-weight:700">{lbl}</span><br>'
        f'<span style="font-size:.75rem;color:#999">{metrics["best_clf"]}</span>'
        f'</div>', unsafe_allow_html=True)

    bc = "#e74c3c" if pred_prob > 0.5 else "#2ecc71"
    cols[2].markdown(
        f'<div class="metric-card warn-card">'
        f'<span style="font-size:.85rem;color:#666">Caking Probability</span><br>'
        f'<span style="font-size:2rem;font-weight:700;color:{bc}">{pred_prob*100:.1f}%</span><br>'
        f'<span style="font-size:.75rem;color:#999">predict_proba</span>'
        f'</div>', unsafe_allow_html=True)

    if pinn_val is not None:
        cols[3].markdown(
            f'<div class="metric-card pinn-card">'
            f'<span style="font-size:.85rem;color:#666">PINN Prediction</span><br>'
            f'<span style="font-size:2rem;font-weight:700">{pinn_val:.1f} Pa</span><br>'
            f'<span style="font-size:.75rem;color:#999">Physics-constrained NN</span>'
            f'</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Caking Risk Gauge")
    cg, ci = st.columns([2, 1])
    with cg:
        fg, ag = plt.subplots(figsize=(7, 1.2))
        ag.barh(0, 100, color="#eee", height=0.4)
        ag.barh(0, pred_prob * 100,
                color="#e74c3c" if pred_prob > 0.5 else "#2ecc71", height=0.4)
        ag.axvline(50, color="gray", linestyle="--", lw=1.2)
        ag.set_xlim(0, 100); ag.set_yticks([]); ag.set_xlabel("Caking Probability (%)")
        ag.text(min(pred_prob * 100 + 1, 90), 0,
                f"{pred_prob*100:.1f}%", va="center", fontsize=9, fontweight="bold")
        for sp in ag.spines.values():
            sp.set_visible(False)
        fg.patch.set_alpha(0)
        st.pyplot(fg, use_container_width=True); plt.close("all")

    with ci:
        st.markdown("**Active physics drivers:**")
        if RH_pct > CRH_pct:
            st.markdown(f"- 💧 RH ({RH_pct:.0f}%) > CRH ({CRH_pct:.0f}%) → liquid bridges")
        if Temp_C > Tg_C:
            st.markdown(f"- 🌡️ T ({Temp_C:.0f}°C) > Tg ({Tg_C:.0f}°C) → sintering")
        if time_hr > 168:
            st.markdown(f"- ⏱️ Long storage ({time_hr:.0f} h > 1 wk)")
        if D50_um < 50:
            st.markdown(f"- 🔬 Fine powder D50={D50_um:.0f} µm")
        if pinn_val is not None:
            st.markdown(f"- 🧠 PINN offset: {pinn_val - pred_str:+.1f} Pa")

    st.divider()
    with st.expander("🔍 Full 25-feature vector"):
        fd = X_input.T.rename(columns={0: "Value"})
        fd["Value"] = fd["Value"].apply(lambda x: f"{x:.5g}")
        st.dataframe(fd, use_container_width=True)

    if SHAP_AVAILABLE and st.button("🔬 Explain (SHAP waterfall)"):
        with st.spinner("Computing SHAP…"):
            exp = shap.TreeExplainer(reg_model)
            sv  = exp.shap_values(X_scaled)
            shap.waterfall_plot(shap.Explanation(
                values=sv[0], base_values=exp.expected_value,
                data=X_scaled[0], feature_names=feat_cols),
                show=False, max_display=15)
            st.pyplot(plt.gcf(), use_container_width=True); plt.close("all")
    elif not SHAP_AVAILABLE:
        st.caption("SHAP not installed — explanation unavailable on this deployment.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & PHYSICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Physics":
    st.title("Exploratory Data Analysis")
    df = get_dataset()
    t1, t2, t3, t4 = st.tabs([
        "Target Distribution", "Physics Relationships",
        "Correlation Heatmap", "Boxplots by Class",
    ])
    with t1:
        st.pyplot(plot_target_distribution(df, CAKING_THRESHOLD_PA), use_container_width=True)
        plt.close("all")
        st.info(f"n=1 200 · [{df['caking_strength_Pa'].min():.0f}, "
                f"{df['caking_strength_Pa'].max():.0f}] Pa · "
                f"Caked {df['is_caked'].mean()*100:.1f}%")
    with t2:
        st.pyplot(plot_physics_relationships(df), use_container_width=True); plt.close("all")
    with t3:
        st.pyplot(plot_correlation_heatmap(df), use_container_width=True); plt.close("all")
    with t4:
        st.pyplot(plot_boxplots_by_class(df), use_container_width=True); plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Results":
    st.title("Model Evaluation & Comparison")
    if not models_ready:
        st.warning("Models not loaded. Go to **🔧 Debug** to diagnose.")
        st.stop()

    # RAW keys for plot functions; renamed only for display tables
    reg_raw = pd.DataFrame(metrics["regression"]).T.sort_values("Test_R2", ascending=False)
    clf_raw = pd.DataFrame(metrics["classification"]).T.sort_values("F1", ascending=False)

    reg_disp = reg_raw.rename(columns={
        "Test_R2": "Test R²", "Test_RMSE": "Test RMSE (Pa)",
        "Test_MAE": "Test MAE (Pa)", "Test_MAPE": "Test MAPE (%)",
        "CV_RMSE_mean": "CV RMSE mean", "CV_RMSE_std": "CV RMSE std",
    })
    clf_disp = clf_raw.rename(columns={
        "CV_F1_mean": "CV F1 mean", "CV_F1_std": "CV F1 std", "ROC_AUC": "ROC-AUC",
    })

    pinn_info      = metrics.get("pinn", {})
    pinn_available = pinn_info.get("available", False)

    tab_names = ["Regression", "Classification", "Side-by-Side Comparison"]
    if pinn_available:
        tab_names.append("🧠 PINN")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown("### Regression — Caking Strength (Pa)")
        st.dataframe(
            reg_disp.style
            .format({"Test R²": "{:.4f}", "Test RMSE (Pa)": "{:.2f}",
                     "Test MAE (Pa)": "{:.2f}", "Test MAPE (%)": "{:.2f}",
                     "CV RMSE mean": "{:.2f}", "CV RMSE std": "{:.2f}"})
            .highlight_max(subset=["Test R²"], color="#d4edda")
            .highlight_min(subset=["Test RMSE (Pa)", "Test MAE (Pa)"], color="#d4edda"),
            use_container_width=True,
        )
        br = reg_disp.index[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Model", br)
        c2.metric("Best R²",    f"{reg_disp.loc[br,'Test R²']:.4f}")
        c3.metric("Best RMSE",  f"{reg_disp.loc[br,'Test RMSE (Pa)']:.2f} Pa")

    with tabs[1]:
        st.markdown("### Classification — Caked vs Free-Flowing")
        st.dataframe(
            clf_disp.style
            .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                     "F1": "{:.4f}", "ROC-AUC": "{:.4f}",
                     "CV F1 mean": "{:.4f}", "CV F1 std": "{:.4f}"})
            .highlight_max(subset=["F1", "ROC-AUC"], color="#d4edda"),
            use_container_width=True,
        )
        bc = clf_disp.index[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Model", bc)
        c2.metric("Best F1",    f"{clf_disp.loc[bc,'F1']:.4f}")
        c3.metric("Best AUC",   f"{clf_disp.loc[bc,'ROC-AUC']:.4f}")

    with tabs[2]:
        st.markdown("### Final Model Comparison")
        st.pyplot(plot_model_comparison(reg_raw, clf_raw), use_container_width=True)
        plt.close("all")

    if pinn_available and len(tabs) == 4:
        with tabs[3]:
            st.markdown("### Physics-Informed Neural Network *(Section 8)*")
            st.markdown("""
**Architecture:** `Linear(25→128)→SiLU → (128→64)→SiLU → (64→32)→SiLU → (32→1)`

**Loss:** `L = MSE(ŷ,y) + 0.5×[MSE(∂ŷ/∂t, Arrhenius_rate) + mean(ReLU(−∂ŷ/∂t))]`
""")
            c1, c2, c3 = st.columns(3)
            for col, key, label in [
                (c1, "R2", "PINN R²"), (c2, "RMSE", "PINN RMSE"), (c3, "MAE", "PINN MAE")
            ]:
                suffix = " Pa" if key != "R2" else ""
                col.markdown(
                    f'<div class="metric-card pinn-card">'
                    f'<span style="font-size:.85rem;color:#666">{label}</span><br>'
                    f'<span style="font-size:2rem;font-weight:700">'
                    f'{pinn_info[key]:.4f if key == "R2" else pinn_info[key]:.2f}{suffix}</span>'
                    f'</div>', unsafe_allow_html=True)

            st.divider()
            X_te_s, yr_te, _, _, _ = get_test_set(scaler)
            pinn_m, pinn_ts = load_pinn(len(feat_cols))
            if pinn_m is not None:
                from pinn import predict_pinn
                pp = predict_pinn(pinn_m, pinn_ts, X_te_s.values)
                st.pyplot(plot_pinn_training(
                    pinn_info["train_losses"], pinn_info["phys_losses"],
                    yr_te.values, pp, pinn_info["R2"]),
                    use_container_width=True)
                plt.close("all")
            st.divider()
            st.pyplot(plot_pinn_vs_models(reg_raw, pinn_info["R2"], pinn_info["RMSE"]),
                      use_container_width=True)
            plt.close("all")
            st.info("RF/GBM/XGBoost typically outperform the PINN on this synthetic dataset "
                    "because trees can memorise the generated pattern. The PINN enforces "
                    "physics constraints (monotonicity, Arrhenius T-dependence) that trees cannot.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.title("Model Explainability")
    if not models_ready:
        st.warning("Models not loaded. Go to **🔧 Debug** to diagnose.")
        st.stop()

    X_te_s, yr_te, yc_te, X_te, fc = get_test_set(scaler)
    y_pred_best = reg_model.predict(X_te_s)

    t_shap, t_pdp, t_err = st.tabs(["SHAP Summary", "Partial Dependence", "Error Analysis"])

    with t_shap:
        st.markdown("### SHAP Feature Importance")
        if SHAP_AVAILABLE:
            with st.spinner("Computing SHAP…"):
                exp = shap.TreeExplainer(reg_model)
                sv  = exp.shap_values(X_te_s)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Beeswarm**")
                plt.figure(figsize=(8, 6))
                shap.summary_plot(sv, X_te_s, feature_names=fc, show=False)
                plt.tight_layout()
                st.pyplot(plt.gcf(), use_container_width=True); plt.close("all")
            with c2:
                st.markdown("**Mean |SHAP| (global)**")
                sm = np.abs(sv).mean(axis=0)
                ss = pd.Series(sm, index=fc).sort_values(ascending=True).tail(15)
                fb, ab = plt.subplots(figsize=(7, 6))
                ss.plot(kind="barh", ax=ab, color="steelblue")
                ab.set_title("Top 15 SHAP Importances", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fb, use_container_width=True); plt.close("all")
            st.info(f"**Top feature:** `{ss.index[-1]}`")
        else:
            st.warning("SHAP not installed — showing permutation importance instead.")
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(reg_model, X_te_s, yr_te, n_repeats=10, random_state=42)
            ps = pd.Series(pi.importances_mean, index=fc).sort_values(ascending=True).tail(15)
            fp, ap = plt.subplots(figsize=(8, 6))
            ps.plot(kind="barh", ax=ap, color="darkorange")
            ap.set_title("Permutation Importance", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fp, use_container_width=True); plt.close("all")

    with t_pdp:
        st.markdown("### Partial Dependence Plots — Physics Verification")
        with st.spinner("Computing PDP…"):
            st.pyplot(plot_partial_dependence(reg_model, X_te_s, fc),
                      use_container_width=True)
        plt.close("all")

    with t_err:
        st.markdown("### Regression Error Analysis")
        st.pyplot(plot_regression_evaluation(metrics["best_reg"], yr_te, y_pred_best, X_te),
                  use_container_width=True)
        plt.close("all")

        st.markdown("### Classification Evaluation")
        yp_clf  = clf_model.predict(X_te_s)
        yp_prob = clf_model.predict_proba(X_te_s)[:, 1]
        st.pyplot(plot_classification_evaluation(metrics["best_clf"], yc_te, yp_clf, yp_prob),
                  use_container_width=True)
        plt.close("all")

        pinn_info = metrics.get("pinn", {})
        if pinn_info.get("available", False) and TORCH_AVAILABLE:
            st.divider()
            st.markdown("### PINN vs Best ML — Residual Overlay")
            pinn_m2, pinn_ts2 = load_pinn(len(feat_cols))
            if pinn_m2 is not None:
                from pinn import predict_pinn
                pp     = predict_pinn(pinn_m2, pinn_ts2, X_te_s.values)
                res_ml = yr_te.values - y_pred_best
                res_pn = yr_te.values - pp
                fr, ar = plt.subplots(1, 2, figsize=(14, 4))
                ar[0].hist(res_ml, bins=40, color="steelblue",  edgecolor="white", alpha=0.7, label="Best ML")
                ar[0].hist(res_pn, bins=40, color="#9b59b6", edgecolor="white", alpha=0.6, label="PINN")
                ar[0].axvline(0, color="red", linestyle="--")
                ar[0].set_xlabel("Residual (Pa)"); ar[0].set_ylabel("Count")
                ar[0].set_title("Residual Distribution"); ar[0].legend()
                ar[1].scatter(y_pred_best, res_ml, alpha=0.4, s=12, color="steelblue", label="Best ML")
                ar[1].scatter(pp,          res_pn, alpha=0.4, s=12, color="#9b59b6",   label="PINN")
                ar[1].axhline(0, color="red", linestyle="--")
                ar[1].set_xlabel("Predicted (Pa)"); ar[1].set_ylabel("Residual (Pa)")
                ar[1].set_title("Residuals vs Fitted"); ar[1].legend()
                plt.tight_layout()
                st.pyplot(fr, use_container_width=True); plt.close("all")
                ml_r = float(np.sqrt(np.mean(res_ml**2)))
                pn_r = float(np.sqrt(np.mean(res_pn**2)))
                c1, c2 = st.columns(2)
                c1.metric("ML RMSE",   f"{ml_r:.2f} Pa")
                c2.metric("PINN RMSE", f"{pn_r:.2f} Pa", delta=f"{pn_r - ml_r:+.2f} Pa")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About this Project")
    st.markdown("""
## Powder Caking Prediction — Physics-Informed ML

### Governing Physics
| Mechanism | Equation |
|---|---|
| Kelvin vapour pressure | ln(p/p₀) = 2γVₘ/rRT |
| JKR contact adhesion  | F_pull = ³⁄₂ π W R* |
| Sintering rate        | dNb/dt = A·exp(−Ea/RT)·σⁿ |

### Caking Strength Scale
| Range | Label |
|---|---|
| 0–400 Pa    | Free-flowing |
| 400–800 Pa  | Mildly caked |
| 800–1500 Pa | Moderately caked |
| >1500 Pa    | Severely caked |

**Threshold: 800 Pa** (Johanson 2009)

### Pipeline
| Step | Detail |
|---|---|
| Data §3 | 1 200 physics-based synthetic samples, [0, 2000] Pa |
| Features §6 | 18 raw + 7 engineered = 25 total |
| Regression §7 | Ridge, Lasso, DT, RF, GBM, ExtraTrees, SVR, KNN, XGBoost |
| Classification §7 | LogReg, DT, RF, GBM, SVC, KNN, XGBoost |
| PINN §8 | SiLU MLP · data MSE + sintering residual + monotonicity |
| Tuning §9 | RandomizedSearchCV 50 trials |
| Explainability §11 | SHAP TreeExplainer + PDP |
""")
