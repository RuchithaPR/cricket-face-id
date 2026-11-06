"""Web demo:  streamlit run app.py"""
import joblib
import streamlit as st
from common import load_image
from predict import predict

st.title("🏏 Indian Cricketer Face Identifier")


@st.cache_resource
def get_model():
    return joblib.load("model.joblib")


thr = st.slider("Confidence threshold (below = Unknown)", 0.1, 0.95, 0.5)
up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
if up:
    img = load_image(up)
    st.image(img, width=300)
    label, top = predict(img, get_model(), thr)
    if label is None:
        st.warning("No face detected in this image.")
    else:
        st.subheader(label.replace("_", " "))
        for n, p in top:
            st.progress(p, text=f"{n.replace('_', ' ')} — {p:.1%}")
