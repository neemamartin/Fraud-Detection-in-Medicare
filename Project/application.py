

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import bcrypt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the pre-trained models and scaler
model = joblib.load('random_forest.joblib')
scaler = joblib.load('scaler1.joblib')

# Define the fraud prediction function
def predict_fraud(npi, drug_name):
    # Load your dataset or input data
    data_features1 = pd.read_csv('data_features1.csv')  # Load the original data

    # Ensure 'NPI' column is of the same type for merging
    data_features1['NPI'] = data_features1['NPI'].astype(str)

    # Define the feature columns used during training
    feature_columns = [
        'Tot_Drug_Cst_sum_sum', 'Tot_Drug_Cst_mean_mean', 'Total_Amount_of_Payment_USDollars',
        'Tot_Drug_Cst_max_max', 'Tot_Clms_sum_sum', 'Tot_Clms_mean_mean', 'Tot_Clms_max_max',
        'Tot_Day_Suply_sum_sum', 'Tot_Day_Suply_mean_mean', 'Tot_Day_Suply_max_max',
        'claim_max-mean', 'supply_max-mean', 'drug_max-mean', 'drug_sum', 'Spec_Weight'
    ]

    # Prepare input data for prediction
    input_data = {
        'NPI': [str(npi)],  # Ensure input NPI is a string
        'DrugName': [drug_name],
    }
    df_input = pd.DataFrame(input_data)
    df_input['NPI'] = df_input['NPI'].astype(str)  # Convert input NPI to string

    # Merge with features data to get all necessary columns
    df_input = pd.merge(df_input, data_features1, on='NPI', how='left')
    df_input = df_input.fillna(0)
    
    # Ensure the input has the same feature set as used during training
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    
    # Extract features for prediction
    X_input = df_input[feature_columns].values
    
    # Check if the number of features matches the scaler's input features
    if X_input.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Feature mismatch: Expected {scaler.n_features_in_} features, but got {X_input.shape[1]} features.")
    
    # Scale the input data
    X_input_scaled = scaler.transform(X_input)
    
    # Predict using the loaded model
    prediction = model.predict_proba(X_input_scaled)[:, 1]
    
    return prediction[0]

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# Database functions
def create_user_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY, 
            username TEXT UNIQUE, 
            password TEXT, 
            email TEXT UNIQUE
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password, email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username, password, email) VALUES (?,?,?)', (username, password, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    return data

# Password hashing
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Email validation
def validate_email(email):
    email_regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.match(email_regex, email) is not None

# Sign-up function
def signup():
    st.markdown(
        """
        <style>
        .signup-container {
            max-width: 400px;
            margin: 5% auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            background-color: #ffffff;
        }
        .title {
            text-align: center;
            color: #00796b;
        }
        .btn {
            display: block;
            width: 100%;
            padding: 0.75rem;
            margin-top: 1rem;
            border: none;
            border-radius: 5px;
            background-color: #00796b;
            color: white;
            font-size: 1rem;
        }
        .btn:hover {
            background-color: #004d40;
        }
        .error {
            color: #d32f2f;
            font-size: 0.875rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="signup-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="title">Register</h2>', unsafe_allow_html=True)
    
    new_user = st.text_input('Username', key='signup_user')
    email = st.text_input('Email', key='signup_email')
    new_password = st.text_input('Password', type='password', key='signup_pass')

    if st.button('Sign Up', key='signup_btn'):
        if new_user and new_password and email:
            if not validate_email(email):
                st.markdown('<p class="error">Please enter a valid email address.</p>', unsafe_allow_html=True)
            else:
                create_user_table()
                hashed_new_password = hash_password(new_password)
                if add_user(new_user, hashed_new_password, email):
                    st.success('You have successfully created an account')
                    st.info('Redirecting to Login...')
                    st.session_state['page'] = 'login'
                    st.experimental_rerun()
                else:
                    st.markdown('<p class="error">Username or email already exists. Please choose a different username or email.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="error">Please enter all required fields.</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sign-in function
def signin():
    st.markdown(
        """
        <style>
        .signin-container {
            max-width: 400px;
            margin: 5% auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            background-color: #ffffff;
        }
        .title {
            text-align: center;
            color: #00796b;
        }
        .btn {
            display: block;
            width: 100%;
            padding: 0.75rem;
            margin-top: 1rem;
            border: none;
            border-radius: 5px;
            background-color: #00796b;
            color: white;
            font-size: 1rem;
        }
        .btn:hover {
            background-color: #004d40;
        }
        .error {
            color: #d32f2f;
            font-size: 0.875rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="signin-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="title">Login</h2>', unsafe_allow_html=True)
    
    username = st.text_input('Username', key='signin_user')
    password = st.text_input('Password', type='password', key='signin_pass')

    if st.button('Sign In', key='signin_btn'):
        user_data = get_user(username)
        if user_data:
            if check_password(password, user_data[2]):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.session_state['page'] = 'home'
                st.experimental_rerun()
            else:
                st.markdown('<p class="error">Incorrect username or password.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="error">Username does not exist.</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Home function with visualizations and user guidance
def home():
    st.markdown(
        """
        <style>
        .home-container {
            max-width: 600px;
            margin: 5% auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            background-color: #ffffff;
        }
        .title {
            text-align: center;
            color: #00796b;
        }
        .btn {
            display: block;
            width: 100%;
            padding: 0.75rem;
            margin-top: 1rem;
            border: none;
            border-radius: 5px;
            background-color: #00796b;
            color: white;
            font-size: 1rem;
        }
        .btn:hover {
            background-color: #004d40;
        }
        .form-control {
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="home-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="title">Fraud Prediction</h2>', unsafe_allow_html=True)

    npi = st.text_input('NPI', key='npi')
    drug_name = st.text_input('Drug Name', key='drug_name')
    threshold = 50  # Fixed threshold at 50%

    if st.button('Predict Fraud', key='predict_btn'):
        if npi and drug_name:
            try:
                prediction = predict_fraud(npi, drug_name)
                
                # Adjust threshold from percentage
                threshold_adjusted = threshold / 100.0
                
                fraud_percent = prediction * 100
                
                # Donut Pie Chart
                st.subheader("Fraud Prediction Probability")
                fig, ax = plt.subplots(figsize=(6, 6))
                labels = ['Fraud', 'Non-Fraud']
                sizes = [fraud_percent, 100 - fraud_percent]
                colors = ['#FF6347', '#20B2AA']
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
                ax.set_title('Fraud Prediction Probability', fontsize=16)
                st.pyplot(fig)

                # Bar Plot
                st.subheader("Fraud Probability vs Threshold")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['Prediction', 'Threshold'], [fraud_percent, threshold], color=['#FF6347', '#20B2AA'])
                ax.set_ylim(0, 100)
                ax.set_ylabel('Percentage (%)')
                ax.set_title('Fraud Probability vs Threshold')
                st.pyplot(fig)

                # Gauge Plot
                st.subheader("Fraud Severity Gauge")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_percent,
                    title={'text': "Fraud Probability"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#FF6347"},
                           'steps': [
                               {'range': [0, 50], 'color': "#20B2AA"},
                               {'range': [50, 100], 'color': "#FF6347"}],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}}))
                fig.update_layout(height=400)
                st.plotly_chart(fig)

                # Additional Plot: Distribution of Prediction
                st.subheader("Prediction Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist([fraud_percent], bins=[0, 25, 50, 75, 100], edgecolor='black', color='#FF6347', alpha=0.7)
                ax.set_xlabel('Fraud Probability Range (%)')
                ax.set_ylabel('Frequency')
                ax.set_title('Fraud Probability Distribution')
                st.pyplot(fig)

                # Detailed Metrics
                st.subheader("Detailed Prediction Metrics")
                metrics = {
                    "Predicted Fraud Probability (%)": f"{fraud_percent:.2f}",
                    "Threshold (%)": f"{threshold:.2f}",
                    "Fraud Likelihood": "High" if prediction > threshold_adjusted else "Low"
                }
                st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

                # User Guidance
                if prediction > threshold_adjusted:
                    st.warning(f"The model predicts a high likelihood of fraud ({fraud_percent:.2f}%).")
                    st.info("The prediction exceeds the threshold of 50%, indicating potential fraudulent activity. It is advised to reconsider using this drug, as the drug dealer has shown fraudulent activity in the past.")
                else:
                    st.success(f"The model predicts a low likelihood of fraud ({fraud_percent:.2f}%).")
                    st.info("The prediction is below the threshold of 50%, suggesting that the drug dealer has not shown significant fraudulent behavior. You are good to proceed with this drug.")

            except ValueError as e:
                st.error(f'Error: {e}')
        else:
            st.error('Please enter both NPI and Drug Name.')
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main logic
def main():
    if st.session_state['authenticated']:
        if st.session_state['page'] == 'home':
            home()
        else:
            st.error('Unknown page.')
    else:
        if st.session_state['page'] == 'login':
            signin()
        elif st.session_state['page'] == 'signup':
            signup()
        else:
            st.error('Unknown page.')

if __name__ == '__main__':
    main()
