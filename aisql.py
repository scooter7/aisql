import streamlit as st
import os
import pandas as pd
from uuid import uuid4
import psycopg2

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader

# Ensure necessary directories exist
folders_to_create = ['csvs', 'vectors']
for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)
    print(f"Directory '{folder}' checked or created.")

# Load environment and API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize language models and embeddings
language_model = OpenAI(api_key=openai_api_key)
chat_language_model = ChatOpenAI(api_key=openai_api_key, temperature=0.4)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

def fetch_table_details(cursor):
    sql = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public';
    """
    cursor.execute(sql)
    return cursor.fetchall()

def fetch_foreign_key_details(cursor):
    sql = """
        SELECT conrelid::regclass AS table_name, conname AS foreign_key,
               pg_get_constraintdef(oid) AS constraint_definition
        FROM pg_constraint
        WHERE contype = 'f' AND connamespace = 'public'::regnamespace;
    """
    cursor.execute(sql)
    return cursor.fetchall()

def create_vector_database(data, directory):
    loader = CSVLoader(file_path=data, encoding="utf8")
    document_data = loader.load()
    vector_db = Chroma.from_documents(document_data, embedding=embeddings, persist_directory=directory)
    vector_db.persist()

def save_database_details(uri):
    unique_id = str(uuid4()).replace("-", "_")
    conn = psycopg2.connect(uri)
    cur = conn.cursor()
    details = fetch_table_details(cur)
    df = pd.DataFrame(details, columns=['table_name', 'column_name', 'data_type'])
    csv_path = f'csvs/tables_{unique_id}.csv'
    df.to_csv(csv_path, index=False)
    create_vector_database(csv_path, f"./vectors/tables_{unique_id}")
    
    foreign_keys = fetch_foreign_key_details(cur)
    fk_df = pd.DataFrame(foreign_keys, columns=['table_name', 'foreign_key', 'constraint_definition'])
    fk_csv_path = f'csvs/foreign_keys_{unique_id}.csv'
    fk_df.to_csv(fk_csv_path, index=False)
    
    cur.close()
    conn.close()
    return unique_id

def generate_sql_query_template(query, db_uri):
    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(
            content=(
                f"You are an assistant capable of composing SQL queries. Use the details provided to write a relevant SQL query for the question below. DB connection string is {db_uri}."
                "Enclose the SQL query with three backticks '```'."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ])
    response = chat_language_model(template.format_messages(text=query))
    return response.content

# Streamlit application setup
st.title("Database Interaction Tool")
uri = st.text_input("Enter the RDS Database URI")
if st.button("Connect to Database"):
    if uri:
        try:
            unique_id = save_database_details(uri)
            st.success(f"Connected to database and data saved with ID: {unique_id}")
        except Exception as e:
            st.error(f"Failed to connect: {str(e)}")
    else:
        st.warning("Please enter a valid database URI.")

st.subheader("SQL Query Generator")
query = st.text_area("Enter your query here:")
if st.button("Generate SQL Query"):
    if uri and query:
        try:
            sql_query = generate_sql_query_template(query, uri)
            st.text_area("Generated SQL Query", value=sql_query, height=300)
        except Exception as e:
            st.error(f"Failed to generate SQL query: {str(e)}")
    else:
        st.warning("Please provide both a database URI and a query.")
