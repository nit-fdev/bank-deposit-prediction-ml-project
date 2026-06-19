import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="centered"
)


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "bank_model.keras"
PREPROCESSOR_PATH = BASE_DIR / "model" / "preprocessor.joblib"


# =========================
# Category Options
# =========================

JOB_OPTIONS = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]

MARITAL_OPTIONS = [
    "divorced",
    "married",
    "single",
]

EDUCATION_OPTIONS = [
    "primary",
    "secondary",
    "tertiary",
    "unknown",
]

YES_NO_OPTIONS = [
    "no",
    "yes",
]

CONTACT_OPTIONS = [
    "cellular",
    "telephone",
    "unknown",
]

MONTH_OPTIONS = [
    "apr",
    "aug",
    "dec",
    "feb",
    "jan",
    "jul",
    "jun",
    "mar",
    "may",
    "nov",
    "oct",
    "sep",
]

POUTCOME_OPTIONS = [
    "failure",
    "other",
    "success",
    "unknown",
]


# =========================
# Load Model and Preprocessor
# =========================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    return joblib.load(PREPROCESSOR_PATH)


model = load_model()
preprocessor = load_preprocessor()


# =========================
# Helper Function
# =========================

def predict_deposit(input_data: dict):
    input_df = pd.DataFrame([input_data])

    # Make sure column order matches training
    input_df = input_df[
        [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome",
            "age",
            "balance",
            "day",
            "duration",
            "campaign",
            "pdays",
            "previous",
        ]
    ]

    processed_data = preprocessor.transform(input_df)

    probability = float(model.predict(processed_data, verbose=0)[0][0])
    prediction = 1 if probability >= 0.5 else 0

    return prediction, probability, input_df


# =========================
# UI
# =========================

st.title("🏦 Bank Term Deposit Prediction")
st.write(
    "This app predicts whether a customer is likely to subscribe to a bank term deposit."
)

st.divider()

with st.form("prediction_form"):
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        job = st.selectbox(
            "Job",
            JOB_OPTIONS,
            index=JOB_OPTIONS.index("management")
        )

        marital = st.selectbox(
            "Marital Status",
            MARITAL_OPTIONS,
            index=MARITAL_OPTIONS.index("married")
        )

        education = st.selectbox(
            "Education",
            EDUCATION_OPTIONS,
            index=EDUCATION_OPTIONS.index("secondary")
        )

        default = st.selectbox(
            "Has Credit in Default?",
            YES_NO_OPTIONS,
            index=YES_NO_OPTIONS.index("no")
        )

        balance = st.number_input(
            "Account Balance",
            value=1000,
            step=100
        )

        housing = st.selectbox(
            "Has Housing Loan?",
            YES_NO_OPTIONS,
            index=YES_NO_OPTIONS.index("no")
        )

        loan = st.selectbox(
            "Has Personal Loan?",
            YES_NO_OPTIONS,
            index=YES_NO_OPTIONS.index("no")
        )

    with col2:
        contact = st.selectbox(
            "Contact Type",
            CONTACT_OPTIONS,
            index=CONTACT_OPTIONS.index("cellular")
        )

        day = st.number_input(
            "Last Contact Day",
            min_value=1,
            max_value=31,
            value=15,
            step=1
        )

        month = st.selectbox(
            "Last Contact Month",
            MONTH_OPTIONS,
            index=MONTH_OPTIONS.index("may")
        )

        duration = st.number_input(
            "Call Duration",
            min_value=0,
            value=180,
            step=10,
            help="Duration is call duration in seconds."
        )

        campaign = st.number_input(
            "Number of Contacts During Campaign",
            min_value=1,
            value=1,
            step=1
        )

        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=-1,
            value=-1,
            step=1,
            help="-1 means the client was not previously contacted."
        )

        previous = st.number_input(
            "Number of Previous Contacts",
            min_value=0,
            value=0,
            step=1
        )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            POUTCOME_OPTIONS,
            index=POUTCOME_OPTIONS.index("unknown")
        )

    submitted = st.form_submit_button("Predict")


if submitted:
    input_data = {
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
        "age": age,
        "balance": balance,
        "day": day,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
    }

    try:
        prediction, probability, input_df = predict_deposit(input_data)

        st.divider()
        st.subheader("Prediction Result")

        probability_percent = probability * 100

        st.metric(
            label="Subscription Probability",
            value=f"{probability_percent:.2f}%"
        )

        if prediction == 1:
            st.success("Result: The customer is likely to subscribe to a term deposit.")
        else:
            st.error("Result: The customer is not likely to subscribe to a term deposit.")

        st.progress(min(max(probability, 0.0), 1.0))

        with st.expander("View Input Data"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)


# =========================
# Footer
# =========================

st.divider()
st.caption("Built with Streamlit and TensorFlow")