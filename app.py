import streamlit as st
st.title("Debug")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    st.success("✅ OK")
except Exception as e:
    st.error(f"❌ {e}")