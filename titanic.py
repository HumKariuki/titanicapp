# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:04:39 2024

@author: Hum
"""

import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the model
titanic_model = pickle.load(open(r"C:\Users\LENOVO\OneDrive\Desktop\ML\titanic.sav", 'rb'))

# Set up the web app
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

# Title and Image
st.title("Titanic Survival Prediction 🚢")
image = Image.open(r"C:\Users\LENOVO\OneDrive\Desktop\ML\titanic.jpg")
st.image(image, caption='RMS Titanic', use_column_width=True)

# Sidebar for user input
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", options=[1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", options=['Female', 'Male'])
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=25)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Fare Paid", min_value=0.0, value=30.0)
embarked = st.sidebar.selectbox("Port of Embarkation", options=['Cherbourg', 'Queenstown', 'Southampton'], index=2)

# Convert categorical data to numerical data
sex = 1 if sex == 'Male' else 0
embarked = {'Cherbourg': 1, 'Queenstown': 2, 'Southampton': 3}[embarked]

# Predict function
def predict_survival(model, input_data):
    input_data_as_numpy = np.asarray(input_data)
    reshaped = input_data_as_numpy.reshape(1, -1)
    prediction = model.predict(reshaped)
    return prediction[0]

# Display prediction result
if st.sidebar.button("Predict"):
    input_data = (pclass, sex, age, sibsp, parch, fare, embarked)
    result = predict_survival(titanic_model, input_data)
    if result == 0:
        st.markdown("<h2 style='color: red;'>The passenger did not survive 😢</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>The passenger survived 😊</h2>", unsafe_allow_html=True)

# Additional information
st.markdown(
    """
    <div style="background-color: lightblue; padding: 10px; border-radius: 10px;">
    <h3 style="color: navy;">About this App:</h3>
    <p style="color: darkblue;">This web app predicts the survival of a passenger on the RMS Titanic based on their personal details. 
    The model was trained on historical data from the Titanic disaster.</p>
    </div>
    """, unsafe_allow_html=True)
