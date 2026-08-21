# ==========================================================
# model_loader.py
# Load ML Model, Encoders and Feature Columns
# ==========================================================

import pickle
import streamlit as st


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    # -----------------------------
    # Load Trained Model
    # -----------------------------

    with open("patient_survival_model.pkl", "rb") as f:

        model = pickle.load(f)

    # -----------------------------
    # Load Label Encoders
    # -----------------------------

    with open("label_encoders.pkl", "rb") as f:

        encoders = pickle.load(f)

    # -----------------------------
    # Load Feature Columns
    # -----------------------------

    with open("feature_columns.pkl", "rb") as f:

        feature_columns = pickle.load(f)

    return model, encoders, feature_columns


# ==========================================================
# GLOBAL OBJECTS
# ==========================================================

model, encoders, feature_columns = load_model()