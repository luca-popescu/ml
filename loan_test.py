import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model and scaler
model = joblib.load("loan.pkl")
scaler = joblib.load("scaler.pkl")  # Load the saved scaler

# Title and introduction
st.title("Loan Repayment Predictor")
st.write("Welcome to the Loan Repayment Predictor, a sophisticated tool designed for investors and financial analysts. Utilizing a robust Random Forest Classifier model, this app assesses the likelihood of loan repayment based on comprehensive borrower data from Lending Club. Simply input the key financial indicators, and our predictive algorithm will provide you with a clear, data-driven assessment of the repayment probability.")

# Input field for credit_policy with 'Yes' and 'No' options
credit_policy_response = st.selectbox('Does the customer meet the credit underwriting criteria of LendingClub.com?', options=['Yes', 'No'])

# Convert 'Yes' to 1 and 'No' to 0 for the input data
credit_policy = 1 if credit_policy_response == 'Yes' else 0

# Input fields for all features using sliders
int_rate = st.slider("Loan's Interest Rate", min_value=0.00, max_value=1.00, value=0.10, step=0.01)
installments = st.slider('Monthly Installments', min_value=0, max_value=1500, value=100, step=10)
log_annual_inc = st.slider('Log of Annual Income', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
dti = st.slider('Debt-to-Income Ratio', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
fico = st.slider('FICO Credit Score', min_value=300, max_value=850, value=600, step=10)
days_with_cr_line = st.slider('Days with Credit Line', min_value=0, max_value=100000, value=5000, step=100)
revol_bal = st.slider('Revolving Balance', min_value=0, max_value=1000000, value=10000, step=1000)
revol_util = st.slider('Revolving Line Utilization Rate', min_value=0.0, max_value=100.0, value=30.0, step=0.1)
inq_last_6mths = st.slider('Inquiries in Last 6 Months', min_value=0, max_value=100, value=1, step=1)
delinq_2yrs = st.slider('Delinquencies in Last 2 Years', min_value=0, max_value=100, value=1, step=1)
pub_rec = st.slider('Public Records', min_value=0, max_value=100, value=1, step=1)

# Dropdown for 'purpose' feature with the same categories as in your training dataset
purpose = st.selectbox('Purpose of Loan', ['credit_card', 'debt_consolidation', 'educational', 'major_purchase', 'small_business', 'home_improvement', 'all_other'])

# Generate dummy variables for 'purpose'
purpose_categories = ['credit_card', 'debt_consolidation', 'educational', 'major_purchase', 'small_business', 'home_improvement', 'all_other']
purpose_dummies = pd.DataFrame(columns=['purpose_' + category for category in purpose_categories])
purpose_dummies.loc[0] = [0] * len(purpose_categories)  # Initialize all values to 0
purpose_dummies['purpose_' + purpose] = 1  # Set the selected purpose to 1

# Prepare the input data for prediction
input_features = [credit_policy, int_rate, installments, log_annual_inc, dti, fico, days_with_cr_line, revol_bal, revol_util, inq_last_6mths, delinq_2yrs, pub_rec]
input_data = pd.DataFrame([input_features], columns=['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec'])
input_data = pd.concat([input_data, purpose_dummies], axis=1)

# Convert all column names to string to ensure consistency
input_data.columns = input_data.columns.astype(str)

# Check for NaN values in the input data
if input_data.isnull().values.any():
    st.error("Error: Missing values detected in input data. Please check your inputs.")
else:
    # Scale only the continuous features (Ensure these are the same features scaled in your training)
    continuous_features = ['installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']
    input_data[continuous_features] = scaler.transform(input_data[continuous_features])

    # Prediction
    if st.button('Predict Loan Repayment'):
        prediction = model.predict(input_data.values)
        probability = model.predict_proba(input_data.values)[0]
        if prediction[0] == 1:
            st.write("The loan is likely not to be fully paid.")
            st.write(f"Probability of not being fully paid: {probability[1]:.2f}")
        else:
            st.write("The loan is likely to be fully paid.")
            st.write(f"Probability of being fully paid: {probability[0]:.2f}")
