import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the trained model
from xgboost import XGBClassifier

# Load trained models & scaler
scaler = joblib.load("scaler.pkl")  # StandardScaler
best_model = joblib.load("best_model.pkl")  # Best trained model

# Define feature names used in training
feature_names = ["Weight_in_gms", "Discount_offered", "Cost_of_the_Product",
                 "Customer_rating", "Prior_purchases", "Customer_care_calls"]

# Streamlit App UI
st.title("📦 E-Commerce Delivery Prediction App 🚚")
st.write("Enter details below to predict if an order will be delivered on time.")

# User Inputs
weight = st.number_input("Weight in grams", min_value=0, max_value=5000, value=500)
discount = st.number_input("Discount Offered (%)", min_value=0.0, max_value=50.0, value=10.0)
cost = st.number_input("Cost of Product ($)", min_value=1, max_value=1000, value=100)
rating = st.slider("Customer Rating", min_value=1, max_value=5, value=3)
prior_purchases = st.number_input("Prior Purchases", min_value=0, max_value=50, value=1)
customer_calls = st.number_input("Customer Care Calls", min_value=0, max_value=10, value=2)

# Format input data
features = np.array([[weight, discount, cost, rating, prior_purchases, customer_calls]])

# Convert to DataFrame (Fixes Feature Name Issue)
features_df = pd.DataFrame(features, columns=feature_names)

# Scale features
features_scaled = scaler.transform(features_df)

# Prediction
if st.button("Predict 🚀"):
    prediction = best_model.predict(features_scaled)[0]
    probability = best_model.predict_proba(features_scaled)[0][1] * 100  # Get probability

    # Show Results
    if prediction == 1:
        st.success(f"✅ Order is predicted to be **Delivered on Time**! 📦 (Confidence: {probability:.2f}%)")
    else:
        st.error(f"❌ Order may **NOT be Delivered on Time**. (Confidence: {100-probability:.2f}%)")

st.write("\n🔍 **Powered by Machine Learning** 🚀")
