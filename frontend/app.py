import streamlit as st
import requests

st.title("Wine Quality Prediction")

st.write("Enter the alcohol value to predict wine quality:")
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=10.0)

if st.button("Predict"):
    # Call backend API (assume running at http://localhost:8000/predict)
    response = requests.post(
        "http://linear-regression-backend:8000/predict",
        json={"alcohol": alcohol}
    )
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted wine quality: {result}")
    else:
        st.error("Prediction failed. Please check backend service.")
