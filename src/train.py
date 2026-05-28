"""
train.py
--------
Offline training script.
Reproduces the full pipeline from caking-prediction.ipynb (Sections 3–9)
and serialises the best models + scaler to models/artifacts/.

Run once before launching the Streamlit app:
    python src/train.py

Outputs (written to models/artifacts/)
---------------------------------------
scaler_reg.pkl          PowerTransformer fitted on X_tr  (25 features)
best_reg_model.pkl      Best regression model by Test R²
best_clf_model.pkl      Best classification model by Test F1
feature_cols.json       Ordered list of 25 feature column names
training_metrics.json   All model metrics (reg + clf)
"""

import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ── local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from physics import generate_caking_dataset, clean_dataset, engineer_physics_features

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
ARTIFACTS_DIR  = Path(__file__).parent.parent / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)


def main():
    print("=" * 65)
    print("  Powder Caking PIML — Training Pipeline")
    print("=" * 65)

    # ── Step 1: Generate data ─────────────────────────────────────────────
    print("\n[1/6] Generating physics-based synthetic dataset …")
    df = generate_caking_dataset(n_samples=1200, random_state=RANDOM_STATE)
    df = clean_dataset(df)
    print(f"      Shape: {df.shape}  |  "
          f"Caked: {df['is_caked'].sum()} ({df['is_caked'].mean()*100:.1f}%)")

    # ── Step 2: Feature engineering ───────────────────────────────────────
    print("\n[2/6] Engineering physics features …")
    df_eng = engineer_physics_features(df)
    FEATURE_COLS = [c for c in df_eng.columns
                    if c not in ('caking_strength_Pa', 'is_caked')]
    print(f"      Total features: {len(FEATURE_COLS)}")

    X_full   = df_eng[FEATURE_COLS]
    y_reg    = df_eng['caking_strength_Pa']
    y_clf    = df_eng['is_caked']

    # ── Step 3: Stratified 80/20 split + scale ────────────────────────────
    print("\n[3/6] Splitting (80/20 stratified) and scaling …")
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X_full, y_reg, y_clf,
        test_size=0.20, random_state=RANDOM_STATE, stratify=y_clf,
    )
    sc = PowerTransformer(method='yeo-johnson')
    X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), columns=FEATURE_COLS)
    X_te_s = pd.DataFrame(sc.transform(X_te),     columns=FEATURE_COLS)
    print(f"      Train: {X_tr_s.shape}  |  Test: {X_te_s.shape}")

    assert len(np.unique(yc_tr)) == 2, "Training set must contain both classes!"

    # ── Step 4: Regression models (Section 7 notebook) ───────────────────
    print("\n[4/6] Training regression models …")
    reg_models = {
        'Ridge':        Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso':        Lasso(alpha=0.1, random_state=RANDOM_STATE),
        'DecisionTree': DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(n_estimators=200,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        'GBM':          GradientBoostingRegressor(n_estimators=200,
                                                   random_state=RANDOM_STATE),
        'ExtraTrees':   ExtraTreesRegressor(n_estimators=200,
                                             random_state=RANDOM_STATE, n_jobs=-1),
        'SVR':          SVR(kernel='rbf', C=10, gamma='scale'),
        'KNN':          KNeighborsRegressor(n_neighbors=7),
    }
    if XGB_AVAILABLE:
        reg_models['XGBoost'] = XGBRegressor(
            n_estimators=200, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    reg_results = {}

    for name, model in reg_models.items():
        cv_rmse = -cross_val_score(model, X_tr_s, yr_tr,
                                   cv=kf, scoring='neg_root_mean_squared_error',
                                   n_jobs=-1)
        model.fit(X_tr_s, yr_tr)
        y_pred = model.predict(X_te_s)
        rmse = float(np.sqrt(mean_squared_error(yr_te, y_pred)))
        mae  = float(mean_absolute_error(yr_te, y_pred))
        r2   = float(r2_score(yr_te, y_pred))
        mape = float(np.mean(np.abs((yr_te - y_pred) / (yr_te + 1e-6))) * 100)
        reg_results[name] = {
            'CV_RMSE_mean': float(cv_rmse.mean()),
            'CV_RMSE_std':  float(cv_rmse.std()),
            'Test_RMSE': rmse, 'Test_MAE': mae,
            'Test_R2': r2,     'Test_MAPE': mape,
        }
        print(f"      {name:15s} | R²={r2:.4f} | RMSE={rmse:.2f} Pa | "
              f"CV RMSE={cv_rmse.mean():.2f}±{cv_rmse.std():.2f}")

    reg_df = pd.DataFrame(reg_results).T.sort_values('Test_R2', ascending=False)
    best_reg_name  = reg_df.index[0]
    best_reg_model = reg_models[best_reg_name]
    print(f"\n      ★ Best regression model: {best_reg_name} "
          f"(R²={reg_df['Test_R2'].iloc[0]:.4f})")

    # ── Step 5: Classification models (Section 7 notebook, fixed) ────────
    print("\n[5/6] Training classification models …")
    clf_models = {
        'LogisticReg':  LogisticRegression(C=1.0, max_iter=500,
                                            random_state=RANDOM_STATE,
                                            class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(max_depth=6,
                                                random_state=RANDOM_STATE,
                                                class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=200,
                                                random_state=RANDOM_STATE,
                                                n_jobs=-1,
                                                class_weight='balanced'),
        'GBM':          GradientBoostingClassifier(n_estimators=200,
                                                    random_state=RANDOM_STATE),
        'SVC':          SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                            random_state=RANDOM_STATE, class_weight='balanced'),
        'KNN':          KNeighborsClassifier(n_neighbors=7),
    }
    if XGB_AVAILABLE:
        clf_models['XGBoost'] = XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            verbosity=0, n_jobs=-1, eval_metric='logloss')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf_results = {}

    for name, model in clf_models.items():
        cv_f1 = cross_val_score(model, X_tr_s, yc_tr,
                                cv=skf, scoring='f1', n_jobs=-1)
        model.fit(X_tr_s, yc_tr)
        yp      = model.predict(X_te_s)
        yp_prob = (model.predict_proba(X_te_s)[:, 1]
                   if hasattr(model, 'predict_proba') else yp.astype(float))
        clf_results[name] = {
            'CV_F1_mean': float(cv_f1.mean()),
            'CV_F1_std':  float(cv_f1.std()),
            'Accuracy':   float(accuracy_score(yc_te, yp)),
            'Precision':  float(precision_score(yc_te, yp, zero_division=0)),
            'Recall':     float(recall_score(yc_te, yp, zero_division=0)),
            'F1':         float(f1_score(yc_te, yp, zero_division=0)),
            'ROC_AUC':    float(roc_auc_score(yc_te, yp_prob)),
        }
        print(f"      {name:15s} | F1={clf_results[name]['F1']:.4f} | "
              f"AUC={clf_results[name]['ROC_AUC']:.4f} | "
              f"Acc={clf_results[name]['Accuracy']:.4f}")

    clf_df = pd.DataFrame(clf_results).T.sort_values('F1', ascending=False)
    best_clf_name  = clf_df.index[0]
    best_clf_model = clf_models[best_clf_name]
    print(f"\n      ★ Best classification model: {best_clf_name} "
          f"(F1={clf_df['F1'].iloc[0]:.4f})")

    # ── Step 6: Hyperparameter tuning on best regressor (Section 9) ──────
    print("\n[6/6] Hyperparameter tuning (RandomizedSearchCV — 50 trials) …")
    param_dist_rf = {
        'n_estimators':      [100, 200, 300, 500],
        'max_depth':         [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['sqrt', 'log2', 0.5, 0.7],
        'bootstrap':         [True, False],
    }
    rf_tuned = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_dist_rf,
        n_iter=50,
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring='neg_root_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1, verbose=0,
    )
    rf_tuned.fit(X_tr_s, yr_tr)
    y_tuned  = rf_tuned.best_estimator_.predict(X_te_s)
    r2_tuned = float(r2_score(yr_te, y_tuned))
    rm_tuned = float(np.sqrt(mean_squared_error(yr_te, y_tuned)))
    print(f"      Tuned RF → R²={r2_tuned:.4f} | RMSE={rm_tuned:.2f} Pa")
    print(f"      Best params: {rf_tuned.best_params_}")

    # Replace best reg model with tuned version if it improves R²
    if r2_tuned > reg_df['Test_R2'].iloc[0]:
        best_reg_model = rf_tuned.best_estimator_
        best_reg_name  = 'RandomForest_tuned'
        print(f"      → Using tuned RF as best regression model")

    # ── Serialise artifacts ───────────────────────────────────────────────
    print("\n  Saving artifacts …")
    joblib.dump(sc,             ARTIFACTS_DIR / "scaler_reg.pkl",      compress=3)
    joblib.dump(best_reg_model, ARTIFACTS_DIR / "best_reg_model.pkl",  compress=3)
    joblib.dump(best_clf_model, ARTIFACTS_DIR / "best_clf_model.pkl",  compress=3)

    json.dump(FEATURE_COLS,
              open(ARTIFACTS_DIR / "feature_cols.json", "w"), indent=2)

    all_metrics = {
        'regression':     reg_results,
        'classification': clf_results,
        'best_reg':       best_reg_name,
        'best_clf':       best_clf_name,
    }
    json.dump(all_metrics,
              open(ARTIFACTS_DIR / "training_metrics.json", "w"), indent=2)

    print(f"\n  ✅ Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"     • scaler_reg.pkl")
    print(f"     • best_reg_model.pkl  [{best_reg_name}]")
    print(f"     • best_clf_model.pkl  [{best_clf_name}]")
    print(f"     • feature_cols.json   ({len(FEATURE_COLS)} features)")
    print(f"     • training_metrics.json")
    print("\n  Training complete. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
