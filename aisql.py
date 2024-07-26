import streamlit as st
import openai
import mysql.connector
import pandas as pd

# Load secrets
sql_host = st.secrets["sql_database"]["host"]
sql_database = st.secrets["sql_database"]["database"]
sql_username = st.secrets["sql_database"]["username"]
sql_password = st.secrets["sql_database"]["password"]
openai_api_key = st.secrets["openai"]["api_key"]

# Configure OpenAI
openai.api_key = openai_api_key

# Database connection
def connect_to_db():
    return mysql.connector.connect(
        host=sql_host,
        database=sql_database,
        user=sql_username,
        password=sql_password
    )

# Function to execute SQL query
def execute_query(query):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

# Chatbot function
def get_chatbot_response(user_input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You specialize in concisely explaining complex topics to 12yo.",
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
    )
    return response.choices[0].message["content"]

# Streamlit UI
st.title("Chatbot with SQL Querying")

# User input for chatbot
user_input = st.text_input("Ask the chatbot anything:")

if user_input:
    chatbot_response = get_chatbot_response(user_input)
    st.write("Chatbot response:")
    st.write(chatbot_response)

# User input for SQL query
sql_query = st.text_input("Enter an SQL query:")

if sql_query:
    query_result = execute_query(sql_query)
    df = pd.DataFrame(query_result)
    st.write("SQL Query Result:")
    st.dataframe(df)
