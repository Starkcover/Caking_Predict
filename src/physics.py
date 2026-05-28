"""
physics.py
----------
Physics-based dataset generation and feature engineering.
All code extracted verbatim from caking-prediction.ipynb (Sections 3 & 6).

Units reference
---------------
D10_um, D50_um, D90_um : µm
BET_m2g                : m²/g
Temp_C                 : °C
RH_pct, CRH_pct        : % (0–100)
time_hr                : hours
pressure_kPa           : kPa
Tg_C                   : °C
water_activity         : dimensionless [0, 1]
Ca_capillary           : dimensionless
Bo_bond                : dimensionless
Zc_coordination        : dimensionless
T_minus_Tg             : K (same magnitude as °C differences)
RH_above_CRH           : % (zero-clipped)
caking_strength_Pa     : Pa  — regression target
is_caked               : {0, 1} — classification target (threshold 800 Pa)
"""

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────
CAKING_THRESHOLD_PA = 800   # [Pa] literature-grounded mildly-caked boundary
RANDOM_STATE = 42


# ── Section 3: Data Generator ──────────────────────────────────────────────

def generate_caking_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a physics-consistent synthetic caking dataset.
    Extracted verbatim from notebook Section 3 (FIXED version).

    Fix applied
    -----------
    Original 300 Pa threshold sat below the global minimum of CI_raw (~350 Pa),
    producing 100 % caked samples → single class → StratifiedKFold failure.
    Fix: normalise CI_raw to [0, 2000] Pa; threshold at 800 Pa (~50.8 % caked).
    """
    rng = np.random.default_rng(random_state)
    n = n_samples

    # ── RAW INPUT FEATURES ──────────────────────────────────────────────────
    D50_um       = rng.uniform(10, 500, n)          # median particle diameter  [µm]
    span_psd     = rng.uniform(0.5, 3.0, n)         # span = (D90-D10)/D50      [–]
    D10_um       = D50_um * (1 - span_psd * 0.3)    # derived 10th-percentile   [µm]
    D10_um       = np.clip(D10_um, 1, None)          # physical floor: 1 µm
    D90_um       = D50_um * (1 + span_psd * 0.7)    # derived 90th-percentile   [µm]

    BET_m2g      = rng.uniform(0.1, 15, n) + 500 / D50_um  # [m²/g]
    shape_factor = rng.uniform(0.4, 1.0, n)                 # circularity [–]

    Temp_C       = rng.uniform(15,  60,  n)   # [°C]
    RH_pct       = rng.uniform(20,  95,  n)   # [%]
    time_hr      = rng.uniform(1,  720,  n)   # [hours]
    pressure_kPa = rng.uniform(0,  200,  n)   # [kPa]

    CRH_pct      = rng.uniform(40, 85, n)     # critical RH [%]
    Tg_C         = rng.uniform(-20, 80, n)    # glass-transition temp [°C]

    # ── PHYSICS-DERIVED FEATURES ────────────────────────────────────────────
    a_w = RH_pct / 100.0                                             # [–]

    surface_tension = 0.072                                          # [N/m]
    viscosity_est   = 0.001 * np.exp(-0.05 * (Temp_C - 20))         # [Pa·s]
    velocity_est    = 1e-6 / (D50_um * 1e-6)                        # [m/s]
    Ca = (viscosity_est * velocity_est) / surface_tension            # [–]

    rho_particle = 1500                                              # [kg/m³]
    g = 9.81                                                         # [m/s²]
    r_m = D50_um * 1e-6 / 2                                          # [m]
    Bo = (rho_particle * g * r_m**2) / surface_tension               # [–]

    Zc           = 6.0 / (1 + (D50_um / 100)**0.5)                  # [–]
    T_minus_Tg   = Temp_C - Tg_C                                     # [K]
    RH_above_CRH = np.maximum(RH_pct - CRH_pct, 0)                  # [%]

    # ── TARGET GENERATION ───────────────────────────────────────────────────
    R_gas = 8.314;  T_K = Temp_C + 273.15;  Ea = 80_000
    arrhenius      = np.exp(-Ea / (R_gas * T_K))
    arrhenius_norm = (arrhenius - arrhenius.min()) / (arrhenius.max() - arrhenius.min())

    moisture_effect = 1 / (1 + np.exp(-0.15 * (RH_pct - CRH_pct)))
    time_effect     = np.log1p(time_hr) / np.log1p(720)
    size_effect     = np.exp(-D50_um / 200)
    ssa_effect      = np.tanh(BET_m2g / 5)
    pressure_effect = np.log1p(pressure_kPa) / np.log1p(200)
    tg_effect       = np.where(T_minus_Tg > 0, np.tanh(T_minus_Tg / 20), 0.0)

    CI_raw = (
        200 * arrhenius_norm
        + 500 * moisture_effect
        + 300 * time_effect
        + 400 * size_effect
        + 150 * ssa_effect
        + 100 * pressure_effect
        + 250 * tg_effect
    )

    # Normalise to [0, 2000] Pa  (physically justified: severe cake ~ 2000 Pa)
    CI_min, CI_max = CI_raw.min(), CI_raw.max()
    caking_strength_Pa_clean = (CI_raw - CI_min) / (CI_max - CI_min) * 2000  # [Pa]

    noise = rng.normal(0, 0.10 * caking_strength_Pa_clean.std(), n)
    caking_strength_Pa = np.clip(caking_strength_Pa_clean + noise, 0, None)   # [Pa]

    is_caked = (caking_strength_Pa > CAKING_THRESHOLD_PA).astype(int)

    df = pd.DataFrame({
        'D10_um':            D10_um,
        'D50_um':            D50_um,
        'D90_um':            D90_um,
        'span_psd':          span_psd,
        'BET_m2g':           BET_m2g,
        'shape_factor':      shape_factor,
        'Temp_C':            Temp_C,
        'RH_pct':            RH_pct,
        'time_hr':           time_hr,
        'pressure_kPa':      pressure_kPa,
        'CRH_pct':           CRH_pct,
        'Tg_C':              Tg_C,
        'water_activity':    a_w,
        'Ca_capillary':      Ca,
        'Bo_bond':           Bo,
        'Zc_coordination':   Zc,
        'T_minus_Tg':        T_minus_Tg,
        'RH_above_CRH':      RH_above_CRH,
        'caking_strength_Pa': caking_strength_Pa,
        'is_caked':          is_caked,
    })
    return df


# ── Section 5: Cleaning ─────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Production-quality data cleaning pipeline.
    Extracted verbatim from notebook Section 5.
    """
    df_clean = df.copy()

    # 1. Missing value imputation
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        for col in df_clean.columns[df_clean.isnull().any()]:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # 2. Duplicate removal
    df_clean.drop_duplicates(inplace=True)

    # 3. Physics-based domain validation
    df_clean['RH_pct']            = df_clean['RH_pct'].clip(0, 100)
    df_clean['Temp_C']            = df_clean['Temp_C'].clip(-50, 150)
    df_clean['caking_strength_Pa'] = df_clean['caking_strength_Pa'].clip(0, None)
    df_clean['water_activity']    = df_clean['water_activity'].clip(0, 1)

    return df_clean


# ── Section 6: Feature Engineering ─────────────────────────────────────────

def engineer_physics_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7 physics-engineered features per notebook Section 6.
    Extracted verbatim from engineer_physics_features() in the notebook.

    New features
    ------------
    T_over_Tg_ratio        : WLF-like sintering proximity [–]
    kelvin_ratio           : Kelvin vapour-pressure correction proxy [–]
    rh_particle_interaction: RH × BET / (D50+1)  [m²·%/g·µm]
    JKR_proxy              : BET / √D50  [m²/g·µm⁰·⁵]
    arrhenius_time         : time_hr × exp(-Ea/RT)  [h]
    surface_volume_proxy   : BET × D50⁻¹  [m²/g·µm]
    psd_moisture           : span_psd × water_activity  [–]
    """
    df_e = df_in.copy()

    df_e['T_over_Tg_ratio'] = (
        (df_e['Temp_C'] + 273.15) / (df_e['Tg_C'] + 273.15 + 1e-6)
    )

    df_e['kelvin_ratio'] = np.log(
        (df_e['RH_pct'] + 1e-3) / (df_e['CRH_pct'] + 1e-3)
    )

    df_e['rh_particle_interaction'] = (
        df_e['RH_pct'] * df_e['BET_m2g'] / (df_e['D50_um'] + 1)
    )

    df_e['JKR_proxy'] = df_e['BET_m2g'] / (df_e['D50_um'] ** 0.5 + 1e-6)

    T_K = df_e['Temp_C'] + 273.15
    R = 8.314;  Ea = 80000
    df_e['arrhenius_time'] = df_e['time_hr'] * np.exp(-Ea / (R * T_K))

    df_e['surface_volume_proxy'] = df_e['BET_m2g'] / (df_e['D50_um'].astype(float) + 1e-6)

    df_e['psd_moisture'] = df_e['span_psd'] * df_e['water_activity']

    return df_e


# ── Helper: build a single-row feature vector for inference ─────────────────

ENGINEERED_FEATURE_NAMES = [
    'T_over_Tg_ratio', 'kelvin_ratio', 'rh_particle_interaction',
    'JKR_proxy', 'arrhenius_time', 'surface_volume_proxy', 'psd_moisture',
]

RAW_FEATURE_NAMES = [
    'D10_um', 'D50_um', 'D90_um', 'span_psd', 'BET_m2g', 'shape_factor',
    'Temp_C', 'RH_pct', 'time_hr', 'pressure_kPa', 'CRH_pct', 'Tg_C',
    'water_activity', 'Ca_capillary', 'Bo_bond', 'Zc_coordination',
    'T_minus_Tg', 'RH_above_CRH',
]

ALL_FEATURE_NAMES = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES  # 25 total


def build_single_row(
    D50_um: float,
    span_psd: float,
    BET_m2g: float,
    shape_factor: float,
    Temp_C: float,
    RH_pct: float,
    time_hr: float,
    pressure_kPa: float,
    CRH_pct: float,
    Tg_C: float,
) -> pd.DataFrame:
    """
    Reconstruct all 25 model features from the 10 user-facing inputs.
    Returns a single-row DataFrame with columns matching ALL_FEATURE_NAMES.
    """
    D10_um = float(np.clip(D50_um * (1 - span_psd * 0.3), 1, None))
    D90_um = D50_um * (1 + span_psd * 0.7)

    a_w = RH_pct / 100.0

    surface_tension = 0.072
    viscosity_est   = 0.001 * np.exp(-0.05 * (Temp_C - 20))
    velocity_est    = 1e-6 / (D50_um * 1e-6)
    Ca = (viscosity_est * velocity_est) / surface_tension

    rho_particle = 1500;  g = 9.81
    r_m = D50_um * 1e-6 / 2
    Bo = (rho_particle * g * r_m**2) / surface_tension

    Zc           = 6.0 / (1 + (D50_um / 100)**0.5)
    T_minus_Tg   = Temp_C - Tg_C
    RH_above_CRH = float(np.maximum(RH_pct - CRH_pct, 0))

    raw = {
        'D10_um': D10_um, 'D50_um': D50_um, 'D90_um': D90_um,
        'span_psd': span_psd, 'BET_m2g': BET_m2g, 'shape_factor': shape_factor,
        'Temp_C': Temp_C, 'RH_pct': RH_pct, 'time_hr': time_hr,
        'pressure_kPa': pressure_kPa, 'CRH_pct': CRH_pct, 'Tg_C': Tg_C,
        'water_activity': a_w, 'Ca_capillary': Ca, 'Bo_bond': Bo,
        'Zc_coordination': Zc, 'T_minus_Tg': T_minus_Tg,
        'RH_above_CRH': RH_above_CRH,
    }
    df_raw = pd.DataFrame([raw])
    df_eng = engineer_physics_features(df_raw)
    return df_eng[ALL_FEATURE_NAMES]
