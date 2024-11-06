import streamlit as st
import numpy as np
from keras.models import load_model
# Load the trained model
# model = joblib.load('linear_regression_model.pkl')

model = load_model('model.h5')
# Sidebar setup
st.sidebar.header("**Ahmad Ali Rafique**")
st.sidebar.write("AI & Machine Learning Expert")

st.sidebar.header("Contact Information", divider='rainbow')
st.sidebar.write("Feel free to reach out through the following")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/ahmad-ali-rafique/)")
st.sidebar.write("[GitHub](https://github.com/Ahmad-Ali-Rafique/)")
st.sidebar.write("[Email](mailto:arsbussiness786@gmail.com)")
st.sidebar.write("Developed by Ahmad Ali Rafique", unsafe_allow_html=True)

# Title of the application
st.title("diabete prediction jadid1")

# Description of the application
st.write("""
This application predicts the price of a house in the USA based on several input features.
Please enter the details below to get an estimated price for a house.
""")

# Input fields for user input
gender = st.number_input('gender')
age = st.number_input('age')
hypertension = st.number_input('hypertension')
heart_disease = st.number_input('heart_disease')
bmi = st.number_input('bmi')
HbA1c_level = st.number_input('HbA1c_level')
blood_glucose_level = st.number_input('blood_glucose_level')

# Prepare the input for prediction
input_data = np.array([[gender, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]])

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write("The estimated house price is: ")
    st.write(prediction[0])
    if prediction[0]>0.5:
        st.write("yes")
    if prediction[0]<0.5:
        st.write("no")

# Footer or additional information
st.write("""
*Note: This prediction is based on the model trained with historical data and may not reflect the current market conditions.*
""")

