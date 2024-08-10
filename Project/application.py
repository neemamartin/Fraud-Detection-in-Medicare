# import streamlit as st
# import sqlite3
# import bcrypt
# import re
# import joblib
# import numpy as np
# import pandas as pd

# # Initialize session state variables
# if 'authenticated' not in st.session_state:
#     st.session_state['authenticated'] = False
# if 'page' not in st.session_state:
#     st.session_state['page'] = 'login'
# if 'username' not in st.session_state:
#     st.session_state['username'] = ''

# # Load the database into a DataFrame
# def load_database():
#     conn = sqlite3.connect('users.db')
#     return pd.read_sql_query('SELECT * FROM user_predictions', conn)

# # Create the user table if it doesn't exist
# def create_user_table():
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS users(
#             id INTEGER PRIMARY KEY, 
#             username TEXT UNIQUE, 
#             password TEXT, 
#             email TEXT UNIQUE
#         )
#     ''')
#     conn.commit()
#     conn.close()

# # Add a new user to the database
# def add_user(username, password, email):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     try:
#         c.execute('INSERT INTO users(username, password, email) VALUES (?,?,?)', (username, password, email))
#         conn.commit()
#         return True
#     except sqlite3.IntegrityError:
#         return False
#     finally:
#         conn.close()

# # Retrieve user data from the database
# def get_user(username):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute('SELECT * FROM users WHERE username = ?', (username,))
#     data = c.fetchone()
#     conn.close()
#     return data

# # Password hashing
# def hash_password(password):
#     return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# def check_password(password, hashed):
#     return bcrypt.checkpw(password.encode('utf-8'), hashed)

# # Email validation
# def validate_email(email):
#     email_regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#     return re.match(email_regex, email) is not None

# # Sign-up function
# def signup():
#     st.markdown(
#         """
#         <style>
#         .signup-container {
#             max-width: 400px;
#             margin: 5% auto;
#             padding: 2rem;
#             border-radius: 10px;
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#             background-color: #ffffff; /* White background for the form */
#         }
#         .title {
#             text-align: center;
#             color: #00796b; /* Teal color for title */
#         }
#         .btn {
#             display: block;
#             width: 100%;
#             padding: 0.75rem;
#             margin-top: 1rem;
#             border: none;
#             border-radius: 5px;
#             background-color: #00796b; /* Teal color for button */
#             color: white;
#             font-size: 1rem;
#         }
#         .btn:hover {
#             background-color: #004d40; /* Darker teal for hover effect */
#         }
#         .error {
#             color: #d32f2f; /* Red color for errors */
#             font-size: 0.875rem;
#         }
#         </style>
#         """, unsafe_allow_html=True
#     )

#     st.markdown('<div class="signup-container">', unsafe_allow_html=True)
#     st.markdown('<h2 class="title">Register</h2>', unsafe_allow_html=True)
    
#     new_user = st.text_input('Username', key='signup_user')
#     email = st.text_input('Email', key='signup_email')
#     new_password = st.text_input('Password', type='password', key='signup_pass')

#     if st.button('Sign Up', key='signup_btn'):
#         if new_user and new_password and email:
#             if not validate_email(email):
#                 st.markdown('<p class="error">Please enter a valid email address.</p>', unsafe_allow_html=True)
#             else:
#                 create_user_table()
#                 hashed_new_password = hash_password(new_password)
#                 if add_user(new_user, hashed_new_password, email):
#                     st.success('You have successfully created an account')
#                     st.info('Redirecting to Login...')
#                     st.session_state['page'] = 'login'
#                     st.experimental_rerun()
#                 else:
#                     st.markdown('<p class="error">Username or email already exists. Please choose a different username or email.</p>', unsafe_allow_html=True)
#         else:
#             st.markdown('<p class="error">Please enter all required fields.</p>', unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# # Sign-in function
# def signin():
#     st.markdown(
#         """
#         <style>
#         .signin-container {
#             max-width: 400px;
#             margin: 5% auto;
#             padding: 2rem;
#             border-radius: 10px;
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#             background-color: #ffffff; /* White background for the form */
#         }
#         .title {
#             text-align: center;
#             color: #00796b; /* Teal color for title */
#         }
#         .btn {
#             display: block;
#             width: 100%;
#             padding: 0.75rem;
#             margin-top: 1rem;
#             border: none;
#             border-radius: 5px;
#             background-color: #00796b; /* Teal color for button */
#             color: white;
#             font-size: 1rem;
#         }
#         .btn:hover {
#             background-color: #004d40; /* Darker teal for hover effect */
#         }
#         .error {
#             color: #d32f2f; /* Red color for errors */
#             font-size: 0.875rem;
#         }
#         </style>
#         """, unsafe_allow_html=True
#     )

#     st.markdown('<div class="signin-container">', unsafe_allow_html=True)
#     st.markdown('<h2 class="title">Login</h2>', unsafe_allow_html=True)
    
#     username = st.text_input('Username', key='signin_user')
#     password = st.text_input('Password', type='password', key='signin_pass')

#     if st.button('Sign In', key='signin_btn'):
#         if username and password:
#             create_user_table()
#             user = get_user(username)
            
#             if user:
#                 if check_password(password, user[2]):
#                     st.session_state['authenticated'] = True
#                     st.session_state['username'] = username
#                     st.session_state['page'] = 'home'
#                     st.experimental_rerun()
#                 else:
#                     st.markdown('<p class="error">Incorrect username or password</p>', unsafe_allow_html=True)
#             else:
#                 st.markdown('<p class="error">User not found</p>', unsafe_allow_html=True)
#         else:
#             st.markdown('<p class="error">Please enter both username and password.</p>', unsafe_allow_html=True)
    
#     if st.button('Register'):
#         st.session_state['page'] = 'signup'
#         st.experimental_rerun()
    
#     st.markdown('</div>', unsafe_allow_html=True)

# # Feature computation function
# def compute_user_features(raw_data):
#     return {
#         'Tot_Drug_Cst_sum_sum': np.log10(raw_data.get('Tot_Drug_Cst_sum', 0) + 1.0),
#         'Tot_Clms_sum_sum': np.log10(raw_data.get('Tot_Clms_sum', 0) + 1.0),
#     }

# # Load pre-trained model and scaler
# model = joblib.load('GradientBoostingClassifier.joblib')  # Adjust this if using a different model
# scaler = joblib.load('scaler.joblib')

# # Make a prediction
# def make_prediction(features):
#     feature_values = list(features.values())
#     features_scaled = scaler.transform([feature_values])
#     prediction = model.predict(features_scaled)[0]
#     return prediction

# # Predict fraud
# def predict_fraud(database, npi, drug_name):
#     user_data = database[(database['NPI'] == npi) & (database['DrugName'] == drug_name)]
#     if user_data.empty:
#         return "No data available for the given NPI and Drug Name."

#     features = compute_user_features(user_data.iloc[0].to_dict())
#     prediction = make_prediction(features)
#     return prediction

# # Home page function
# def home():
#     database = load_database()
    
#     st.markdown(
#         """
#         <style>
#         .home-container {
#             max-width: 400px;
#             margin: 5% auto;
#             padding: 2rem;
#             border-radius: 10px;
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#             background-color: #ffffff; /* White background for the form */
#         }
#         .title {
#             text-align: center;
#             color: #00796b; /* Teal color for title */
#         }
#         .btn {
#             display: block;
#             width: 100%;
#             padding: 0.75rem;
#             margin-top: 1rem;
#             border: none;
#             border-radius: 5px;
#             background-color: #00796b; /* Teal color for button */
#             color: white;
#             font-size: 1rem;
#         }
#         .btn:hover {
#             background-color: #004d40; /* Darker teal for hover effect */
#         }
#         </style>
#         """, unsafe_allow_html=True
#     )

#     st.markdown('<div class="home-container">', unsafe_allow_html=True)
#     st.markdown('<h2 class="title">Home</h2>', unsafe_allow_html=True)
#     st.subheader(f'Welcome, {st.session_state["username"]}')

#     # Log out button
#     if st.button('Log Out'):
#         st.session_state['authenticated'] = False
#         st.session_state['username'] = ''
#         st.session_state['page'] = 'login'
#         st.experimental_rerun()

#     st.markdown('<h3 class="title">Fraud Detection</h3>', unsafe_allow_html=True)
    
#     npi = st.text_input('Enter NPI:', key='fraud_npi')
#     drug_name = st.text_input('Enter Drug Name:', key='fraud_drug_name')

#     if st.button('Predict Fraud'):
#         if npi and drug_name:
#             prediction = predict_fraud(database, npi, drug_name)
#             if prediction == 1:
#                 st.error('This NPI is likely to be fraudulent.')
#             else:
#                 st.success('This NPI is unlikely to be fraudulent.')
#         else:
#             st.error('Please enter both NPI and Drug Name.')

#     st.markdown('</div>', unsafe_allow_html=True)

# # Main app
# st.set_page_config(page_title='Authentication App', layout='centered')

# if st.session_state['authenticated']:
#     home()
# else:
#     if st.session_state['page'] == 'login':
#         signin()
#     elif st.session_state['page'] == 'signup':
#         signup()

import streamlit as st
import sqlite3
import bcrypt
import re
import joblib
import numpy as np
import pandas as pd

# Initialize session state variables
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# Load the database into a DataFrame
def load_database():
    conn = sqlite3.connect('users.db')
    return pd.read_sql_query('SELECT * FROM user_predictions', conn)

# Create the user table if it doesn't exist
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

# Add a new user to the database
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

# Retrieve user data from the database
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
            background-color: #ffffff; /* White background for the form */
        }
        .title {
            text-align: center;
            color: #00796b; /* Teal color for title */
        }
        .btn {
            display: block;
            width: 100%;
            padding: 0.75rem;
            margin-top: 1rem;
            border: none;
            border-radius: 5px;
            background-color: #00796b; /* Teal color for button */
            color: white;
            font-size: 1rem;
        }
        .btn:hover {
            background-color: #004d40; /* Darker teal for hover effect */
        }
        .error {
            color: #d32f2f; /* Red color for errors */
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
            background-color: #ffffff; /* White background for the form */
        }
        .title {
            text-align: center;
            color: #00796b; /* Teal color for title */
        }
        .btn {
            display: block;
            width: 100%;
            padding: 0.75rem;
            margin-top: 1rem;
            border: none;
            border-radius: 5px;
            background-color: #00796b; /* Teal color for button */
            color: white;
            font-size: 1rem;
        }
        .btn:hover {
            background-color: #004d40; /* Darker teal for hover effect */
        }
        .error {
            color: #d32f2f; /* Red color for errors */
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
        if username and password:
            create_user_table()
            user = get_user(username)
            
            if user:
                if check_password(password, user[2]):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['page'] = 'home'
                    st.experimental_rerun()
                else:
                    st.markdown('<p class="error">Incorrect username or password</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="error">User not found</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="error">Please enter both username and password.</p>', unsafe_allow_html=True)
    
    if st.button('Register'):
        st.session_state['page'] = 'signup'
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Feature computation function
def compute_user_features(raw_data):
    return {
        'Tot_Drug_Cst_sum_sum': np.log10(raw_data.get('Tot_Drug_Cst_sum', 0) + 1.0),
        'Tot_Clms_sum_sum': np.log10(raw_data.get('Tot_Clms_sum', 0) + 1.0),
    }

# Load pre-trained model and scaler
model = joblib.load('GradientBoostingClassifier.joblib')  # Adjust this if using a different model
scaler = joblib.load('scaler.joblib')

# Make a prediction
def make_prediction(features):
    feature_values = list(features.values())
    features_scaled = scaler.transform([feature_values])
    prediction = model.predict(features_scaled)[0]
    return prediction

# Predict fraud
def predict_fraud(database, npi, drug_name):
    user_data = database[(database['NPI'] == npi) & (database['DrugName'] == drug_name)]
    if user_data.empty:
        return "No data available for the given NPI and Drug Name."

    features = compute_user_features(user_data.iloc[0].to_dict())
    prediction = make_prediction(features)
    return prediction

# Home page function
def home():
    database = load_database()
    
    st.markdown(
        """
        <style>
        .home-container {
            max-width: 400px;
            margin: 5% auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            background-color: #ffffff; /* White background for the form */
        }
        .title {
            text-align: center;
            color: #00796b; /* Teal color for title */
        }
        .btn {
            display: block;
            width: 100%;
            padding: 0.75rem;
            margin-top: 1rem;
            border: none;
            border-radius: 5px;
            background-color: #00796b; /* Teal color for button */
            color: white;
            font-size: 1rem;
        }
        .btn:hover {
            background-color: #004d40; /* Darker teal for hover effect */
        }
        .error {
            color: #d32f2f; /* Red color for errors */
            font-size: 0.875rem;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="home-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="title">Fraud Prediction</h2>', unsafe_allow_html=True)
    
    npi = st.text_input('Enter NPI', key='npi')
    drug_name = st.text_input('Enter Drug Name', key='drug_name')
    
    if st.button('Predict Fraud', key='predict_btn'):
        if npi and drug_name:
            result = predict_fraud(database, npi, drug_name)
            st.write(f'Prediction: {result}')
        else:
            st.markdown('<p class="error">Please enter both NPI and Drug Name.</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main function
def main():
    st.title('Fraud Detection System')
    
    if st.session_state['authenticated']:
        if st.session_state['page'] == 'home':
            home()
        else:
            st.session_state['page'] = 'login'
            st.experimental_rerun()
    else:
        if st.session_state['page'] == 'signup':
            signup()
        else:
            signin()

if __name__ == '__main__':
    create_user_table()
    main()



