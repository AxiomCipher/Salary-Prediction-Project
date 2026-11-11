# Salary Estimator

A machine learning web app that estimates salary based on job category, 
education level, and years of experience.

## Problem

Salary transparency is a real challenge. Job seekers often have no 
reliable way to estimate what they should earn for a given role. This 
project uses a trained Random Forest model to provide data-driven salary 
estimates with a typical range, based on US job market data.

## Approach

- Cleaned and preprocessed a dataset of 6,700+ salary records
- Grouped 190 job titles into 9 meaningful job categories to improve 
  model generalization
- Trained and compared four models: Logistic Regression, Decision Tree, 
  Random Forest, and KNN
- Fine-tuned Random Forest using RandomizedSearchCV with 5-fold 
  cross-validation
- Final model achieves R² of 0.89 and MAE of ~$11,792 on unseen data
- Cross-validation std of 0.009 confirms the model is stable and consistent

## Features

- Salary estimate with a typical range
- Profile comparison against real data from the dataset
- Experience vs Salary growth chart for the selected job category and 
  education level

## Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Joblib

## Live App

[Open Salary Estimator](https://salary-prediction-project-14.streamlit.app/)


## Author

Toluwalase Boluwajoko
[LinkedIn](https://www.linkedin.com/in/toluwalase-boluwajoko) · 
[GitHub](https://github.com/AxiomCipher)
