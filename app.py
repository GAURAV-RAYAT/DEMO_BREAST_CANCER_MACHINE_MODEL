import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("ML Model Deployment using Streamlit")

input_text = st.text_input("Enter features (comma separated)", "1,2,3,4")

if st.button("Predict"):
    try:
        arr = np.array([float(x) for x in input_text.split(',')]).reshape(1, -1)
        output = model.predict(arr)[0]
        st.success(f"Prediction: {output}")
    except:
        st.error("Invalid input")
