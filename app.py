import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd  # @ برای وارد کردن pandas
import matplotlib.pyplot as plt  # @ برای وارد کردن matplotlib
import seaborn as sns  # @ Importing seaborn for creating heatmaps
import keras
import joblib

# Load the trained model
scaler = joblib.load('standard_scaler.pkl')  # Make sure to save your StandardScaler in project.py
load_diabetes = joblib.load('load_diabetes.pkl')

# model = tf.keras.models.load_model('model.h5')
model = keras.models.load_model('model.h5', compile = False)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Menu options
st.sidebar.header("MENU", divider='rainbow')
menu = st.sidebar.radio("Select a section:", ("Home", "Plots", "About Project"))

# Sidebar setup
st.sidebar.header("**Developed by :**", divider='rainbow')
st.sidebar.write("Omid Haghayegh (Fall 2024)")

if menu == "Home":
    # Title of the application
    st.title("Diabetes Prediction Application")

    # Description of the application
    st.write("""
    This application predicts the diabetes risk based on several input features.
    Please enter the details below to get a prediction.
    """)

    # Input fields for user input
    gender = st.selectbox("Gender (0: Female, 1: Male)", (0, 1))
    age = st.slider("Age", 0, 100)
    hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", (0, 1))
    heart_disease = st.selectbox("Heart Disease (0: No, 1: Yes)", (0, 1))
    bmi = st.number_input('BMI')
    HbA1c_level = st.number_input('HbA1c Level')
    blood_glucose_level = st.number_input('Blood Glucose Level')
    
    # Prepare the input for prediction
    input_data = np.array([[gender, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]])
   
    # Standardize the input data
    standardized_input = scaler.transform(input_data)

    # Make a prediction for user input
    if st.button('Predict'):
        prediction = model.predict(standardized_input)
        st.write("The estimated diabetes risk is: ")
        st.write(prediction[0]*100 )
        if prediction[0] > 0.5:
            st.write("Yes. High Risk")
        else:
            st.write("No. Low Risk")


elif menu == "Plots":
    # # Load test data from project.py
    # import project  # @ برای وارد کردن داده‌ها از فایل project.py
    # X_test = project.X_test  # @ فرض بر این است که X_test در project.py تعریف شده است
    # y_test = project.y_test  # @ فرض بر این است که y_test در project.py تعریف شده است
    # y_pred = model.predict(X_test)  # @ پیش‌بینی بر اساس داده‌های تست

    # # Plotting the results
    # if st.button('Show Prediction Plot'):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(y_test, label='Actual', color='blue')
    #     plt.plot(y_pred, label='Predicted', color='red')
    #     plt.title('Comparison of Actual and Predicted Values')
    #     plt.xlabel('Test Sample Index')
    #     plt.ylabel('Values')
    #     plt.legend()
    #     st.pyplot(plt)

    # Heatmap of the diabetes dataset from project.py
    # if st.button('Show Heatmap'):
        # import project
        diabetes = load_diabetes  # @ Access the load_diabetes DataFrame
        plt.figure(figsize=(12, 8))
        sns.heatmap(diabetes[['diabetes', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'heart_disease', 'hypertension']].corr(), 
                     annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Heatmap of Diabetes Dataset Features')
        st.pyplot(plt)
