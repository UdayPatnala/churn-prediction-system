"""
Streamlit UI for the Churn Prediction System.

Allows users to input customer details and get real-time churn predictions
by calling the FastAPI backend.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Churn Prediction", page_icon="📉")

st.title("📉 Customer Churn Prediction")
st.markdown("Enter customer details below to predict their likelihood of churning.")

st.sidebar.header("System Status")
try:
    health_res = requests.get(f"{API_URL}/health", timeout=2)
    if health_res.status_code == 200 and health_res.json().get("model_loaded"):
        st.sidebar.success("✅ API Connected & Model Loaded")
        metrics = requests.get(f"{API_URL}/metrics", timeout=2).json()
        st.sidebar.metric("Model Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        st.sidebar.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    else:
        st.sidebar.error("❌ API Degraded (Model missing?)")
except requests.exceptions.RequestException:
    st.sidebar.error("❌ API Offline")

st.header("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    has_internet = st.selectbox("Has Internet?", ["Yes", "No"])
    has_phone = st.selectbox("Has Phone?", ["Yes", "No"])
    support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=0)

if st.button("Predict Churn Probability", type="primary"):
    payload = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "has_internet": has_internet,
        "has_phone": has_phone,
        "support_tickets": support_tickets
    }
    
    try:
        with st.spinner("Analyzing profile..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"]
            label = result["label"]
            
            st.subheader("Prediction Result")
            
            if result["prediction"] == 1:
                st.error(f"⚠️ {label} ({prob:.1%} probability)")
                st.progress(prob)
                st.markdown("### Suggested Actions")
                st.markdown("- Offer a discounted 1-year contract")
                st.markdown("- Reach out for a customer success call")
            else:
                st.success(f"✅ {label} ({prob:.1%} probability)")
                st.progress(prob)
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
