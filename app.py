# IMPORT LIBRARIES
# ==============================
import streamlit as st
import numpy as np
import joblib

# LOAD MODEL & SCALER
# ==============================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# APP TITLE
# ==============================
st.title("Loan Approval Prediction System")
st.write("Enter applicant details below to predict loan approval status.")

# ==============================
# USER INPUTS
# ==============================
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income_annum = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term (in months)")
cibil_score = st.number_input("CIBIL Score")

residential_assets_value = st.number_input("Residential Assets Value")
commercial_assets_value = st.number_input("Commercial Assets Value")
luxury_assets_value = st.number_input("Luxury Assets Value")
bank_asset_value = st.number_input("Bank Asset Value")

# ==============================
# ENCODE INPUTS (MATCH TRAINING)
# ==============================
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# ==============================
# PREDICTION BUTTON
# ==============================
if st.button("Predict Loan Status"):

    # Arrange input in correct order
    input_data = np.array([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)



    # ==============================
    # OUTPUT RESULT
    # ==============================
    if prediction[0] == 0:
        st.success("Loan Approved ✅")
        st.write("This applicant is considered low risk based on financial and credit profile.")
    else:
        st.error("Loan Rejected ❌")
        st.write("High risk detected. Applicant may default on loan.")