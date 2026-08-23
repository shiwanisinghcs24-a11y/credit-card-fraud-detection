import streamlit as st
import pandas as pd
import joblib

st.title("Credit Card Fraud Detection Demo")
st.write("Select a sample transaction below to see how the model predicts it.")

model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

samples = pd.read_csv('sample_transactions.csv')

selected_index = st.selectbox("Choose a transaction:", samples.index)
selected_transaction = samples.loc[[selected_index]]

st.write("Transaction details:")
st.write(selected_transaction)

if st.button("Predict"):
    features = selected_transaction.drop('Actual', axis=1)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ Predicted: FRAUD (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Predicted: NORMAL (Probability of fraud: {probability:.2%})")
    
    actual = selected_transaction['Actual'].values[0]
    st.write(f"Actual label: {'Fraud' if actual == 1 else 'Normal'}")
