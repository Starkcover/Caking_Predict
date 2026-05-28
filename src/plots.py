"""
plots.py
--------
All plotting helpers extracted from caking-prediction.ipynb (Sections 4, 10, 11).
Functions return Matplotlib Figure objects so Streamlit can display them with
st.pyplot(fig).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, r2_score,
)
from sklearn.inspection import PartialDependenceDisplay

plt.style.use('seaborn-v0_8-whitegrid')


# ── Section 4.2 ──────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, threshold: float = 800.0):
    """Histogram + KDE + class balance bar chart. (Section 4.2)"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(df['caking_strength_Pa'], bins=40,
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(threshold, color='red', linestyle='--',
                    label=f'Caking threshold ({threshold:.0f} Pa)')
    axes[0].set_xlabel('Caking Strength (Pa)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Caking Strength')
    axes[0].legend()

    x = np.linspace(df['caking_strength_Pa'].min(),
                    df['caking_strength_Pa'].max(), 300)
    kde = gaussian_kde(df['caking_strength_Pa'])
    axes[1].plot(x, kde(x), color='darkorange', linewidth=2)
    axes[1].fill_between(x, kde(x), alpha=0.3, color='darkorange')
    axes[1].set_xlabel('Caking Strength (Pa)')
    axes[1].set_title('KDE — Caking Strength')

    counts = df['is_caked'].value_counts()
    axes[2].bar(['Free-Flowing (0)', 'Caked (1)'], counts.values,
                color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
    axes[2].set_ylabel('Count')
    axes[2].set_title('Class Balance')
    for i, v in enumerate(counts.values):
        axes[2].text(i, v + 5, str(v), ha='center', fontweight='bold')

    plt.suptitle('Target Variable Analysis', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ── Section 4.4 ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """Lower-triangular Pearson correlation heatmap. (Section 4.4)"""
    fig, ax = plt.subplots(figsize=(18, 14))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={'size': 7}, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', fontsize=14,
                 fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


# ── Section 4.5 ──────────────────────────────────────────────────────────────

def plot_physics_relationships(df: pd.DataFrame):
    """Scatter: caking_strength vs 6 key physics features. (Section 4.5)"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_pairs = [
        ('RH_pct',       '📌 RH drives moisture sorption → liquid bridges'),
        ('time_hr',      '📌 Monotone increase (physics constraint)'),
        ('Temp_C',       '📌 Arrhenius: exponential T dependence'),
        ('D50_um',       '📌 Finer particles → more contact points'),
        ('T_minus_Tg',   '📌 T > Tg activates glass-rubber sintering'),
        ('RH_above_CRH', '📌 Above CRH: deliquescence onset'),
    ]
    colors = df['is_caked'].map({0: '#2ecc71', 1: '#e74c3c'})
    for ax, (feat, note) in zip(axes.flatten(), plot_pairs):
        ax.scatter(df[feat], df['caking_strength_Pa'],
                   c=colors, alpha=0.4, s=15)
        ax.set_xlabel(feat, fontweight='bold')
        ax.set_ylabel('Caking Strength (Pa)')
        ax.set_title(feat, fontsize=10)
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=7,
                va='top', ha='left', color='navy')

    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='Free-Flowing'),
        mpatches.Patch(facecolor='#e74c3c', label='Caked'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
    plt.suptitle('Caking Strength vs Key Physics Variables',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ── Section 4.6 ──────────────────────────────────────────────────────────────

def plot_boxplots_by_class(df: pd.DataFrame):
    """Boxplots of 9 key features split by caking class. (Section 4.6)"""
    key_features = ['RH_pct', 'Temp_C', 'time_hr', 'D50_um', 'BET_m2g',
                    'pressure_kPa', 'T_minus_Tg', 'RH_above_CRH', 'water_activity']
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, feat in enumerate(key_features):
        data_grouped = [
            df[df['is_caked'] == 0][feat].values,
            df[df['is_caked'] == 1][feat].values,
        ]
        bp = axes[i].boxplot(data_grouped, patch_artist=True,
                             labels=['Free-Flowing', 'Caked'], widths=0.5)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        for patch in bp['boxes']:
            patch.set_alpha(0.7)
        axes[i].set_title(feat, fontweight='bold')
        axes[i].set_ylabel(feat)
    plt.suptitle('Feature Distributions by Caking Class',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Section 10: Regression evaluation ────────────────────────────────────────

def plot_regression_evaluation(
    best_reg_name: str,
    yr_te: pd.Series,
    y_pred_best: np.ndarray,
    X_te: pd.DataFrame,
):
    """4-panel regression error analysis. (Section 10)"""
    residuals = yr_te.values - y_pred_best
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Actual vs Predicted
    lo, hi = yr_te.min(), yr_te.max()
    axes[0, 0].scatter(yr_te, y_pred_best, alpha=0.5, s=20, c='steelblue')
    axes[0, 0].plot([lo, hi], [lo, hi], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Caking Strength (Pa)')
    axes[0, 0].set_ylabel('Predicted (Pa)')
    r2 = r2_score(yr_te, y_pred_best)
    axes[0, 0].set_title(f'{best_reg_name}: Actual vs Predicted\nR²={r2:.4f}')

    # Residual plot
    axes[0, 1].scatter(y_pred_best, residuals, alpha=0.5, s=20, c='darkorange')
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Value')
    axes[0, 1].set_ylabel('Residual (Actual – Predicted)')
    axes[0, 1].set_title('Residual Plot (Heteroscedasticity Check)')

    # Residual distribution
    axes[1, 0].hist(residuals, bins=40, color='steelblue',
                    edgecolor='white', alpha=0.8)
    axes[1, 0].axvline(0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Residual (Pa)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(
        f'Residual Distribution\nμ={residuals.mean():.2f}, σ={residuals.std():.2f}'
    )

    # MAE by RH regime
    rh_groups = pd.cut(
        X_te['RH_pct'], bins=[0, 40, 60, 80, 100],
        labels=['<40%', '40–60%', '60–80%', '>80%']
    )
    error_by_rh = pd.Series(np.abs(residuals)).groupby(rh_groups).mean()
    error_by_rh.plot(kind='bar', ax=axes[1, 1],
                     color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'])
    axes[1, 1].set_title('Mean Absolute Error by RH Regime')
    axes[1, 1].set_ylabel('MAE (Pa)')
    axes[1, 1].set_xlabel('RH Range')
    axes[1, 1].tick_params(axis='x', rotation=0)

    plt.suptitle(f'Detailed Error Analysis — {best_reg_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Section 10: Classification evaluation ────────────────────────────────────

def plot_classification_evaluation(
    best_clf_name: str,
    yc_te,
    yp_clf: np.ndarray,
    yp_prob: np.ndarray,
):
    """Confusion matrix + ROC + PR curve. (Section 10)"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cm = confusion_matrix(yc_te, yp_clf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Free-Flowing', 'Caked'],
                yticklabels=['Free-Flowing', 'Caked'])
    axes[0].set_title(f'Confusion Matrix — {best_clf_name}')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    fpr, tpr, _ = roc_curve(yc_te, yp_prob)
    auc = roc_auc_score(yc_te, yp_prob)
    axes[1].plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC = {auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()

    prec, rec, _ = precision_recall_curve(yc_te, yp_prob)
    axes[2].plot(rec, prec, lw=2, color='darkorange')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')

    plt.suptitle(f'Classification Evaluation — {best_clf_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Section 11: Partial Dependence Plots ─────────────────────────────────────

def plot_partial_dependence(model, X_te_s: pd.DataFrame, feature_cols: list):
    """5-panel PDP physics-consistency check. (Section 11)"""
    key_phys_features = ['RH_pct', 'Temp_C', 'time_hr', 'D50_um', 'T_minus_Tg']
    key_phys_idx = [feature_cols.index(f) for f in key_phys_features]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, feat, feat_idx in zip(axes, key_phys_features, key_phys_idx):
        PartialDependenceDisplay.from_estimator(
            model, X_te_s, [feat_idx],
            feature_names=feature_cols,
            ax=ax, line_kw={'color': 'steelblue', 'lw': 2}
        )
        ax.set_title(f'PDP: {feat}', fontsize=9)

    plt.suptitle(
        'Partial Dependence Plots — Physics Consistency Check\n'
        '(RH, T, time should show monotone/sigmoidal patterns)',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    return fig


# ── Section 12: Final model comparison ───────────────────────────────────────

def plot_model_comparison(reg_df: pd.DataFrame, clf_df: pd.DataFrame):
    """R² and F1 bar-chart comparison. (Section 12)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    reg_df['Test_R2'].plot(
        kind='bar', ax=axes[0],
        color=['gold' if i == 0 else 'steelblue' for i in range(len(reg_df))],
        edgecolor='white'
    )
    axes[0].set_title('Model Comparison: R² (Regression)', fontweight='bold')
    axes[0].set_ylabel('R²')
    axes[0].axhline(0.9, color='red', linestyle='--', alpha=0.5,
                    label='R²=0.90 target')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    clf_df['F1'].plot(
        kind='bar', ax=axes[1],
        color=['gold' if i == 0 else '#3498db' for i in range(len(clf_df))],
        edgecolor='white'
    )
    axes[1].set_title('Model Comparison: F1-Score (Classification)',
                      fontweight='bold')
    axes[1].set_ylabel('F1-Score')
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('Final Model Comparison (Gold = Best Model)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ── PINN training curves ──────────────────────────────────────────────────────

def plot_pinn_training(train_losses: list, phys_losses: list,
                       y_actual, pinn_pred: "np.ndarray", pinn_r2: float):
    """PINN loss curves + actual vs predicted. (Section 8)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, color='steelblue', label='Data Loss (MSE)')
    axes[0].plot(phys_losses, color='darkorange',
                 linestyle='--', label='Physics Loss')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('PINN Training Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_actual, pinn_pred, alpha=0.5, s=20, color='mediumpurple')
    lims = [min(float(np.min(y_actual)), float(np.min(pinn_pred))),
            max(float(np.max(y_actual)), float(np.max(pinn_pred)))]
    axes[1].plot(lims, lims, 'r--', lw=2, label='Identity Line')
    axes[1].set_xlabel('Measured Caking Strength (Pa)')
    axes[1].set_ylabel('PINN Predicted (Pa)')
    axes[1].set_title(f'Actual vs. Predicted (R²={pinn_r2:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
# ── PINN vs ML Comparison ─────────────────────────────────────────────────────

def plot_pinn_vs_models(reg_df: pd.DataFrame, pinn_r2: float, pinn_rmse: float):
    """Bar charts comparing PINN to Traditional ML models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. R2 Comparison
    r2_series = reg_df['Test_R2'].copy()
    r2_series.loc['PINN'] = pinn_r2
    r2_series = r2_series.sort_values(ascending=False)
    colors_r2 = ['#9b59b6' if idx == 'PINN' else 'steelblue' for idx in r2_series.index]
    
    r2_series.plot(kind='bar', ax=axes[0], color=colors_r2, edgecolor='white')
    axes[0].set_title('R² Comparison (Higher is Better)', fontweight='bold')
    axes[0].set_ylabel('R² Score')
    axes[0].tick_params(axis='x', rotation=45)

    # 2. RMSE Comparison
    rmse_series = reg_df['Test_RMSE'].copy()
    rmse_series.loc['PINN'] = pinn_rmse
    rmse_series = rmse_series.sort_values(ascending=True)
    colors_rmse = ['#9b59b6' if idx == 'PINN' else 'darkorange' for idx in rmse_series.index]
    
    rmse_series.plot(kind='bar', ax=axes[1], color=colors_rmse, edgecolor='white')
    axes[1].set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
    axes[1].set_ylabel('RMSE (Pa)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('PINN vs Traditional ML Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig