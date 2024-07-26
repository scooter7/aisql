import streamlit as st
import pandas as pd

# Initialize connection.
# Uses st.connection's secrets management.
conn = st.connection('mysql', type='sql')

# Perform query.
query = 'SELECT * FROM table1;'  # Change 'your_table' to your actual table name
df = conn.query(query, ttl=600)

# Print results.
st.write(df)
