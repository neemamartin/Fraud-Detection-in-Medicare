import sqlite3

def recreate_user_table():
    # Connect to the database (or create it if it doesn't exist)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Drop existing tables if they exist
    c.execute('DROP TABLE IF EXISTS users')

    # Create users table
    c.execute('''
        CREATE TABLE users(
            id INTEGER PRIMARY KEY, 
            username TEXT UNIQUE, 
            password TEXT, 
            email TEXT UNIQUE
        )
    ''')

    # Create table for storing predictions

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# Call the function to recreate the tables
recreate_user_table()
