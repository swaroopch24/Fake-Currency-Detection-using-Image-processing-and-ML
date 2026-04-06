# app.py
# Streamlit demo app for Fake Currency Detection (prototype)
# - Uploads an image
# - Runs lightweight OpenCV heuristics (not a real CNN)
# - Returns "Likely Genuine" / "Potentially Fake" with a confidence score

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from detection import analyze_note

st.set_page_config(page_title="Fake Currency Detection - Prototype", page_icon="💵", layout="centered")

st.title("💵 Fake Currency Detection (Prototype)")
st.caption("Demo-only: heuristic checks using OpenCV. Replace with your CNN later.")

uploaded = st.file_uploader("Upload a banknote image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing security features (heuristic)..."):
        result = analyze_note(np.array(image))

    st.subheader(f"Result: {result['label']}")
    st.metric("Confidence", f"{result['confidence']:.2f}%")

    with st.expander("See analysis details"):
        st.write({
            "sharpness_var_laplacian": round(result["features"]["var_laplace"], 2),
            "edge_density": round(result["features"]["edge_density"], 4),
            "brightness_mean": round(result["features"]["mean_gray"], 2)
        })
        st.progress(min(1.0, result["score"]))

st.markdown("---")
st.write("**Note**: This is a _prototype_ with simple heuristics. For production, plug in a trained CNN and verified datasets.")
