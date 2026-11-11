import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Salary Estimator", page_icon="💼")

# Load Model
model = joblib.load('./models/model3.pkl')
df = pd.read_csv('./data/cleaned_data.csv')

# Constants
JOB_GROUPS = sorted([
    'Data_&_Analytics',
    'Finance_&_Accounting',
    'HR_&_Recruiting',
    'Management_&_Leadership',
    'Marketing_&_Design',
    'Operations_&_Admin',
    'Product_&_Research',
    'Sales_&_Business_Development',
    'Software_Engineering_&_IT',
])

EDUCATION_OPTIONS = ['high_school', 'bachelor', 'master', 'phd']

# --- UI ---
st.title('💼 Salary Estimator')
st.markdown("Fill in your details below to get an estimated salary range.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    education_level = st.selectbox('Highest Education Level', EDUCATION_OPTIONS)

with col2:
    job_group = st.selectbox('Job Category', JOB_GROUPS, format_func=lambda x: x.replace('_', ' '))

years_of_experience = st.slider('Years of Experience', min_value=0, max_value=30, value=5)

st.divider()

if st.button('Predict Salary', use_container_width=True):

    # --- Prediction ---
    sample_data = pd.DataFrame({
        'Education_Level': [education_level],
        'Job_Group': [job_group],
        'Years_of_Experience': [float(years_of_experience)],
    })

    prediction = model.predict(sample_data)[0]
    lower = round(prediction * 0.87, 2)
    upper = round(prediction * 1.13, 2)

    st.success(f"**Estimated Salary: ${prediction:,.0f}**")
    st.info(f"Typical Range: ${lower:,.0f} — ${upper:,.0f}")
    st.caption("Estimate based on US salary data. Actual salaries vary by company, location, and negotiation.")

    st.divider()

    # --- How does my profile compare ---
    st.subheader("How does your profile compare?")

    profile_df = df[
        (df['Job_Group'] == job_group) &
        (df['Education_Level'] == education_level)
    ]

    if len(profile_df) > 0:
        avg_salary = profile_df['Salary'].mean()
        min_salary = profile_df['Salary'].min()
        max_salary = profile_df['Salary'].max()
        count = len(profile_df)

        col3, col4, col5 = st.columns(3)
        col3.metric("Average Salary", f"${avg_salary:,.0f}")
        col4.metric("Lowest", f"${min_salary:,.0f}")
        col5.metric("Highest", f"${max_salary:,.0f}")
        st.caption(f"Based on {count} profiles in this category from the dataset.")
    else:
        st.warning("Not enough data for this combination to show a comparison.")

    st.divider()

    # --- Experience vs Salary chart ---
    st.subheader("How salary grows with experience")
    st.caption(f"{job_group.replace('_', ' ')} | {education_level.replace('_', ' ').title()}")

    exp_range = list(range(0, 31))
    chart_data = pd.DataFrame({
        'Education_Level': [education_level] * 31,
        'Job_Group': [job_group] * 31,
        'Years_of_Experience': [float(x) for x in exp_range],
    })

    chart_data['Predicted_Salary'] = model.predict(chart_data)

    chart_df = pd.DataFrame({
        'Years of Experience': exp_range,
        'Predicted Salary ($)': chart_data['Predicted_Salary'].values
    }).set_index('Years of Experience')

    st.line_chart(chart_df)
