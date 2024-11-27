import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your trained model (assuming you saved it with pickle or joblib)
# You can replace this with the actual path where the model is saved.
with open('log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)

# Load your LabelEncoder for the 'CLASS' column
with open('label_encoder.pkl', 'rb') as f:
    le_class = pickle.load(f)

# Title of the web app
st.title("Diabetes Prediction Web App")

# Subtitle or description
st.write("Enter your medical metrics to predict the class (Diabetes/Normal).")

# Create input fields for user to enter their metrics
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("AGE", min_value=20, max_value=79, value=30)
bmi = st.number_input("BMI", min_value=19.0, max_value=47.75, value=25.0)

# Convert user input into a list of values, ensuring the order matches the training data
gender_encoded = 1 if gender == "Male" else 0  # Assuming 1 for Male, 0 for Female
custom_input = [gender_encoded, age, bmi]

# Create a button for prediction
if st.button("Predict"):
    # Convert the input list into a NumPy array and reshape it
    custom_input_array = np.array(custom_input).reshape(1, -1)

    # Make a prediction using the trained model
    prediction = log_reg_model.predict(custom_input_array)

    # Convert the prediction back to the original class label (e.g., "Normal" or "Diabetes")
    predicted_class = le_class.inverse_transform(prediction)

       # Convert the prediction back to the original class label
    predicted_class = le_class.inverse_transform(prediction)[0]

    # Display a message based on the predicted class
    if predicted_class == 'Y':  # Assuming 'Y' indicates diabetes
        st.success("You have a high chance to have diabetes.")
    elif predicted_class == 'N':  # Assuming 'N' indicates normal
        st.success("Good news, you have a low chance to have diabetes. Keep it up!")
    elif predicted_class == 'P':  # Assuming 'P' indicates prediabetes
        st.warning("You might be prediabetic, make sure to monitor your sugar intake and consult a doctor.")
    else:
        st.error("Unexpected prediction result. Please check your inputs.")