# imports
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
import streamlit as st

import nest_asyncio
nest_asyncio.apply()

st.subheader(':file_cabinet: Chat with your SQL Database')
st.caption("Created by [Jay Shah](https://www.linkedin.com/in/jay-shah-qml) with :heart:")

# Read database connection details from Streamlit secrets
db_config = st.secrets["connections"]["mysql"]

with st.sidebar:
    # get OPENAI_API_KEY
    OPENAI_API_KEY = st.text_input("OPENAI API KEY", key="chatbot_api_key", type="password")

    if OPENAI_API_KEY:
        db_uri = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
        table_names = db.get_usable_table_names()
        num_tables = len(table_names)

        with st.expander('Database details'):
            st.text(f'Number of Tables: {num_tables}')
            st.text(f'Table Names: {table_names}')
    else:
        st.warning("Please enter your OpenAI API key to continue.")

    with st.expander('Model Details'):
        openai_llm = st.selectbox("OpenAI Model",
                                  ("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "gpt-4o"))

        temperature = st.slider(label='Temperature', min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                                help='Lesser value means less randomness in the output and makes output reproducible.')

        sql_toggle = st.checkbox("Show SQL queries", value=False)  # get verbose or only LLM response
        verbose_toggle = st.checkbox("Show Cost details", value=False)  # get verbose or only LLM response
    reset = st.button('Reset Chat!')  # reset the chat

    st.write("[Get your API key](https://platform.openai.com/account/api-keys)")
    st.write("[GitHub](https://github.com/Jayshah25/Chat-with-your-SQL-Database)")

if OPENAI_API_KEY:
    llm = ChatOpenAI(model=openai_llm,
                     temperature=temperature,
                     openai_api_key=OPENAI_API_KEY)
    agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=False, stream_runnable=False)

if "messages" not in st.session_state or reset:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    # if the user started chatting without setting the OPENAI API KEY
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    try:
        with st.spinner('Wait for output...'):
            if sql_toggle:
                added_prompt = 'If you use SQL queries to answer the question, list them too in the output.'
                prompt_new = prompt + added_prompt
                # query the agent
                with get_openai_callback() as cb:
                    result = agent.invoke(prompt_new)
            else:
                with get_openai_callback() as cb:
                    result = agent.invoke(prompt)
            # assistant response
            cb_ = f'Operation Details- Total Tokens:{cb.total_tokens}, Prompt Tokens:{cb.prompt_tokens}, Completion Tokens:{cb.completion_tokens}, Total Cost(USD):{cb.total_cost}'
            msg = result['output'] + cb_ if verbose_toggle else result['output']

        # write the response
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    except Exception as e:
        st.error(e)
