import streamlit as st
import joblib
import numpy as np

# Load model using joblib
model = joblib.load("iris_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🌸 Iris Flower Classification App")

st.write("Enter Flower Measurements")

# User Inputs
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(input_data)
    flower_name = label_encoder.inverse_transform(prediction)
    
    st.success(f"Predicted Flower is: {flower_name[0]}")
