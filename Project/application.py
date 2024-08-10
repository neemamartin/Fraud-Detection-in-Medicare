import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import bcrypt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import re

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
    
    # # Debug: Print the input data for inspection
    # st.write("Input Data for Prediction:")
    # st.write(df_input)

    # Extract features for prediction
    X_input = df_input[feature_columns].values
    
    # Check if the number of features matches the scaler's input features
    if X_input.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Feature mismatch: Expected {scaler.n_features_in_} features, but got {X_input.shape[1]} features.")
    
    # Scale the input data
    X_input_scaled = scaler.transform(X_input)
    
    # # Debug: Print the scaled input data for inspection
    # st.write("Scaled Input Data:")
    # st.write(X_input_scaled)
    
    # Predict using the loaded model
    prediction = model.predict_proba(X_input_scaled)[:, 1]
    
    # Debug: Print the prediction probability
    st.write("Prediction Probability:")
    st.write(prediction[0])
    
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

# Home page function
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

    if st.button('Predict Fraud', key='predict_btn'):
        if npi and drug_name:
            try:
                prediction = predict_fraud(npi, drug_name)
                if prediction > 0.5:
                    st.success(f'The model predicts a high likelihood of fraud ({prediction:.2f}).')
                else:
                    st.success(f'The model predicts a low likelihood of fraud ({prediction:.2f}).')
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
