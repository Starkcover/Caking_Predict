# 🧪 Powder Caking Prediction — Streamlit App

Physics-Informed ML deployment of **caking-prediction.ipynb**.  
Dual-task: regression (caking strength in Pa) + binary classification (caked / free-flowing).

---

## Project Structure

```
caking-prediction-app/
├── app.py                        ← Streamlit entry point (5 pages)
├── src/
│   ├── __init__.py
│   ├── physics.py                ← generate_caking_dataset, engineer_physics_features,
│   │                               build_single_row  (Sections 3 & 6)
│   ├── pinn.py                   ← CakingPINN, sintering_physics_loss,
│   │                               train_pinn, predict_pinn  (Section 8)
│   ├── plots.py                  ← All plot helpers  (Sections 4, 10, 11, 12)
│   └── train.py                  ← Offline training + serialisation  (Sections 3–9)
├── models/
│   └── artifacts/
│       ├── scaler_reg.pkl        ← PowerTransformer (Yeo-Johnson) fitted on X_tr
│       ├── best_reg_model.pkl    ← Best regression model by Test R²
│       ├── best_clf_model.pkl    ← Best classification model by Test F1
│       ├── feature_cols.json     ← Ordered list of 25 feature names
│       └── training_metrics.json ← All reg + clf metrics
├── .streamlit/
│   └── config.toml               ← Theme + server settings
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start — Local

### 1. Clone / copy the project

```bash
git clone https://github.com/YOUR_USERNAME/caking-prediction-app.git
cd caking-prediction-app
```

### 2. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# OR
.venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train models (once — writes artifacts to models/artifacts/)

```bash
python src/train.py
```

Expected output (takes ~3–5 min on CPU):
```
[1/6] Generating physics-based synthetic dataset …
[2/6] Engineering physics features …
[3/6] Splitting (80/20 stratified) and scaling …
[4/6] Training regression models …
      RandomForest   | R²=0.9xxx | RMSE=xxx Pa …
[5/6] Training classification models …
[6/6] Hyperparameter tuning …
  ✅ Artifacts saved to: models/artifacts/
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## App Pages

| Page | Description | Notebook Section |
|---|---|---|
| 🔮 Predict | Real-time prediction from sidebar sliders + SHAP waterfall | §3, §6, §11 |
| 📊 EDA & Physics | 4 tabs: target dist, physics scatter, correlation, boxplots | §4 |
| 📈 Model Results | Regression & classification leaderboards + comparison chart | §10, §12 |
| 🧠 Explainability | SHAP summary/bar, Partial Dependence, error analysis | §11 |
| ℹ️ About | Project overview, physics equations, pipeline summary | §1 |

---

## Streamlit Cloud Deployment

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — caking prediction app"
git remote add origin https://github.com/YOUR_USERNAME/caking-prediction-app.git
git push -u origin main
```

> **Important:** model `.pkl` files are excluded by `.gitignore` (they can be large).  
> Two options:  
> **Option A** — Remove `*.pkl` from `.gitignore` and commit the artifacts.  
> **Option B** — Run `train.py` inside a GitHub Actions workflow that uploads artifacts.

For quick demos, Option A is simplest. Compress first:
```bash
python -c "
import joblib
for f in ['scaler_reg','best_reg_model','best_clf_model']:
    obj = joblib.load(f'models/artifacts/{f}.pkl')
    joblib.dump(obj, f'models/artifacts/{f}.pkl', compress=3)
print('Compressed.')
"
git add models/artifacts/*.pkl models/artifacts/*.json
git commit -m "Add trained model artifacts"
git push
```

### 2. Connect Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Pick your repo, branch `main`, main file `app.py`
3. Click **Deploy**

### 3. Environment variables / secrets

If you need API keys, add them at **Streamlit Cloud → App → Settings → Secrets**:
```toml
# .streamlit/secrets.toml  (local only — never commit this)
[general]
MY_KEY = "..."
```
Access in Python: `st.secrets["general"]["MY_KEY"]`

---

## Runtime Optimisation

| Technique | Where applied |
|---|---|
| `@st.cache_resource` | Model loading — once per server lifetime |
| `@st.cache_data` | Dataset generation, test-set prep |
| `joblib compress=3` | Reduces `.pkl` size 3–5× |
| `plt.close("all")` | Prevents matplotlib memory leak across rerenders |
| `n_jobs=1` on deployed models | Avoids fork overhead in Streamlit's process |

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: best_reg_model.pkl` | `train.py` not run yet | `python src/train.py` |
| `ValueError: feature names mismatch` | Column order changed | Ensure `feature_cols.json` matches `ALL_FEATURE_NAMES` in `physics.py` |
| `ModuleNotFoundError: shap` | SHAP not installed | `pip install shap` — app degrades gracefully without it |
| `ModuleNotFoundError: xgboost` | XGBoost not installed | `pip install xgboost` — optional, notebook code guards with `XGB_AVAILABLE` |
| Streamlit Cloud memory limit (1 GB) | Too many large trees loaded | Set `n_estimators=100` in `train.py` for demo deployment |
| `torch` slow to import | Large package | Move PINN training to `train.py` only; `app.py` never imports torch |
| SHAP plots blank on Cloud | Matplotlib backend issue | Add `matplotlib.use('Agg')` at top of `plots.py` |

---

## Physics Reference

| Feature | Unit | Physical meaning |
|---|---|---|
| D10/D50/D90_um | µm | Particle size distribution percentiles |
| BET_m2g | m²/g | Specific surface area (contact point density) |
| water_activity | – | RH/100, drives moisture sorption |
| Ca_capillary | – | Viscous vs surface-tension forces |
| Bo_bond | – | Gravity vs surface-tension forces |
| T_minus_Tg | K | Glass-to-rubber sintering activation |
| RH_above_CRH | % | Deliquescence exceedance |
| kelvin_ratio | – | ln(RH/CRH) — Kelvin vapour pressure |
| arrhenius_time | h | Kinetic exposure = time × exp(−Ea/RT) |
| JKR_proxy | m²/g·µm⁰·⁵ | Pull-off force proxy (BET/√D50) |

**Caking threshold: 800 Pa** — Johanson (2009) literature-grounded mildly-caked boundary.

---

## References

1. Johanson (2009) *Measurement and prediction of caking in bulk solids* — Part. Sci. Tech. 27(2)
2. Raissi et al. (2019) *Physics-informed neural networks* — J. Comp. Phys. 378
3. Teunou & Fitzpatrick (1999) *Effect of T and RH on food powder flowability* — J. Food Eng.
4. Lundberg & Lee (2017) *SHAP: A unified approach to interpreting model predictions* — NeurIPS
5. Cranfield (2004) *Powder caking mechanisms and prevention* — Powder Handling & Processing
