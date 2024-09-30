import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model_finance_RandomForest.pkl')

st.title("Credit Score Classifier")

st.header("Input Features")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
monthly_salary = st.number_input("Monthly In-hand Salary", min_value=0, value=5000)
num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=1)
num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, value=1)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=5.0)
delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, value=0)
num_delayed_payments = st.number_input("Number of Delayed Payments", min_value=0, value=0)
num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=0)
outstanding_debt = st.number_input("Outstanding Debt", min_value=0, value=0)
credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, value=30.0)
credit_history_age = st.number_input("Credit History Age (years)", min_value=0, value=1)
total_emi_per_month = st.number_input("Total EMI per Month", min_value=0, value=0)
amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0, value=0)
monthly_balance = st.number_input("Monthly Balance", min_value=0, value=0)

# Create a DataFrame from the input
input_data = pd.DataFrame({
    'Age': [age],
    'Monthly_Inhand_Salary': [monthly_salary],
    'Num_Bank_Accounts': [num_bank_accounts],
    'Num_Credit_Card': [num_credit_cards],
    'Interest_Rate': [interest_rate],
    'Delay_from_due_date': [delay_from_due_date],
    'Num_of_Delayed_Payment': [num_delayed_payments],
    'Num_Credit_Inquiries': [num_credit_inquiries],
    'Outstanding_Debt': [outstanding_debt],
    'Credit_Utilization_Ratio': [credit_utilization_ratio],
    'Credit_History_Age': [credit_history_age],
    'Total_EMI_per_month': [total_emi_per_month],
    'Amount_invested_monthly': [amount_invested_monthly],
    'Monthly_Balance': [monthly_balance]
})

# Predict the credit score bracket
if st.button("Predict Credit Score"):
    prediction = model.predict(input_data)
    score_bracket = ["Poor", "Standard", "Good"]
    st.success(f"The predicted credit score bracket is: {score_bracket[prediction[0]]}")