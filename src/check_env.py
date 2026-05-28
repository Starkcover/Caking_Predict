"""
check_env.py
------------
Environment diagnostic helper.
Called by the Debug page in app.py to surface exactly what is and isn't
available at runtime — useful for Streamlit Cloud troubleshooting.
"""
import sys
import platform
from pathlib import Path


def run_checks(artifacts_dir: Path) -> list[dict]:
    """
    Run all environment checks and return a list of result dicts:
      {label, ok, detail}
    """
    results = []

    def chk(label, fn):
        try:
            detail = fn()
            results.append({"label": label, "ok": True,  "detail": detail or "OK"})
        except Exception as e:
            results.append({"label": label, "ok": False, "detail": str(e)})

    # Python runtime
    chk("Python version",
        lambda: f"{sys.version}  |  platform: {platform.platform()}")

    # Core deps
    chk("numpy",       lambda: __import__("numpy").__version__)
    chk("pandas",      lambda: __import__("pandas").__version__)
    chk("joblib",      lambda: __import__("joblib").__version__)
    chk("scikit-learn",lambda: __import__("sklearn").__version__)
    chk("scipy",       lambda: __import__("scipy").__version__)
    chk("matplotlib",  lambda: __import__("matplotlib").__version__)
    chk("seaborn",     lambda: __import__("seaborn").__version__)
    chk("streamlit",   lambda: __import__("streamlit").__version__)

    # Optional
    chk("xgboost (optional)",
        lambda: __import__("xgboost").__version__)
    chk("torch / PINN (optional)",
        lambda: __import__("torch").__version__)
    chk("shap (optional)",
        lambda: __import__("shap").__version__)

    # Local src modules
    sys.path.insert(0, str(Path(__file__).parent))
    chk("src/physics.py",
        lambda: str(__import__("physics").generate_caking_dataset.__module__))
    chk("src/plots.py",
        lambda: str(__import__("plots").plot_target_distribution.__module__))

    # Artifacts
    for fname in ["scaler_reg.pkl", "best_reg_model.pkl",
                  "best_clf_model.pkl", "feature_cols.json",
                  "training_metrics.json"]:
        path = artifacts_dir / fname
        chk(f"artifact: {fname}",
            lambda p=path: f"exists, {p.stat().st_size // 1024} KB" if p.exists()
                           else (_ for _ in ()).throw(FileNotFoundError(f"not found: {p}")))

    # Model load round-trip
    def _load_model():
        import joblib, json
        sc  = joblib.load(artifacts_dir / "scaler_reg.pkl")
        reg = joblib.load(artifacts_dir / "best_reg_model.pkl")
        clf = joblib.load(artifacts_dir / "best_clf_model.pkl")
        fc  = json.load(open(artifacts_dir / "feature_cols.json"))
        return (f"scaler={type(sc).__name__}  "
                f"reg={type(reg).__name__}  "
                f"clf={type(clf).__name__}  "
                f"features={len(fc)}")
    chk("Model load round-trip", _load_model)

    return results
