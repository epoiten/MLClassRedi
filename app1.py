import streamlit as st
import joblib
import numpy as np


model = joblib.load('decision_tree_model.pkl')
st.title("Iris Classification with Decision Tree")

sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.0)

if st.button("Predict"):
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(features)
    
    st.write(f"Predicted Class: {prediction[0]}")