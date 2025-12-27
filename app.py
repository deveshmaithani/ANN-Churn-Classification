import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

st.title("📊 Customer Churn Prediction (ANN)")
st.write(
    "This app predicts whether a bank customer is likely to churn using a trained "
    "**Artificial Neural Network (ANN)**."
)

# ---------------------------
# Load Model & Preprocessors
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# ---------------------------
# User Inputs
# ---------------------------
st.subheader("🔧 Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox(
        "Geography",
        onehot_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "Gender",
        label_encoder_gender.classes_
    )

    age = st.slider(
        "Age",
        min_value=18,
        max_value=92,
        value=35
    )

    tenure = st.slider(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=3
    )

with col2:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        max_value=300000.0,
        value=50000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=1000.0,
        max_value=200000.0,
        value=50000.0
    )

    num_of_products = st.slider(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# ---------------------------
# Prediction
# ---------------------------
st.markdown("---")

if st.button("🔍 Predict Churn"):
    # Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Prepare base input
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Combine features
    final_input = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # Scale input
    final_input_scaled = scaler.transform(final_input)

    # Predict
    prediction_proba = model.predict(final_input_scaled)[0][0]

    st.subheader("📈 Prediction Result")
    st.write(f"**Churn Probability:** `{prediction_proba:.2f}`")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption(
    "⚠️ This prediction is based on historical data and should be used "
    "as a decision-support tool, not as the sole decision-maker."
)
