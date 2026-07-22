import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load Model
model = joblib.load('./models/model.pkl')

st.title('Salary Prediction')
st.markdown("""
    Welcome 😊!!!\n
    In order to predict your Salary, follow the following rules:
    - Click on the sidebar menu
    - Fill up your information 
    - Close the sidebar
    - Click on "Predict Salary" to get the value
""")


st.sidebar.title('Salary Prediction')
st.sidebar.write('Enter your details')

age = st.sidebar.slider('How Old Are You?',
    min_value=18,
    max_value=65,
    value=20,
    width=150
)

gender = st.sidebar.selectbox(
    'What is Your Gender?',
    ['male', 'female'],
    width=150
)

education_level = st.sidebar.selectbox(
    'What is Your Highest Education Level?',
    ['high_school', 'bachelor', 'master', 'phd'],
    width=150
)

job_title = st.sidebar.selectbox(
    'What Role Are You Planning To Apply For?',
    ['back_end_developer', 'front_end_developer', 'full_stack_engineer',
     'data_analyst', 'data_scientist',
     'it_support',
     'product_designer', 'product_manager', 'project_manager',
     'software_developer', 'software_engineer',
     'ux_designer', 'web_designer', 'web_developer'],
     width=200
)

years_of_experience = st.sidebar.number_input(
    'How Many Years of Experience Do You Have?',
    min_value=0,
    max_value=30,
    value=5,
    width=150
)

sample_data = pd.DataFrame({
    'Age':[age],
    'Gender':[gender],
    'Education_Level':[education_level],
    'Job_Title':[job_title],
    'Years_of_Experience':[float(years_of_experience)],
})

st.divider()

if st.button('Predict Salary'):
    st.write('Here is Your Salary Based On The Details Filled In The Sidebar')
    predictions = np.round(model.predict(sample_data)[0], 2)
    st.success(predictions)

