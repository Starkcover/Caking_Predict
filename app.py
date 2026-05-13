import streamlit as st
st.title("Debug Mode")

# Test 1 - basic imports
try:
    import numpy, pandas, joblib, matplotlib, seaborn, scipy
    st.success("✅ Core imports OK")
except Exception as e:
    st.error(f"❌ Core import failed: {e}")

# Test 2 - sklearn
try:
    import sklearn
    st.success(f"✅ sklearn {sklearn.__version__} OK")
except Exception as e:
    st.error(f"❌ sklearn failed: {e}")

# Test 3 - local src imports
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from physics import build_single_row
    st.success("✅ physics.py OK")
except Exception as e:
    st.error(f"❌ physics.py failed: {e}")

# Test 4 - model loading
try:
    import joblib, json
    ARTIFACTS = Path(__file__).parent / "models" / "artifacts"
    reg = joblib.load(ARTIFACTS / "best_reg_model.pkl")
    st.success("✅ Models loaded OK")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")

st.stop()