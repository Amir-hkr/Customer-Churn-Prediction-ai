import streamlit as st
import pandas as pd
import joblib


# -------------------------
# Load model assets
# -------------------------
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer will leave or stay.")


# -------------------------
# Inputs
# -------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)


# -------------------------
# Build input
# -------------------------
input_df = pd.DataFrame([{
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": senior,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])


# Align with training features
input_df = input_df.reindex(columns=features, fill_value=0)


# Scale
input_scaled = scaler.transform(input_df)


# -------------------------
# Prediction
# -------------------------
if st.button("Predict Churn"):

    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.subheader("📌 Result")

    # Probability display
    st.write(f"🔢 Churn Probability: **{prob:.2%}**")

    # Progress bar
    st.progress(float(prob))

    # Decision
    if pred == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer will stay")


    # Extra insight
    st.markdown("---")
    st.info("💡 Higher probability means higher risk of losing the customer.")