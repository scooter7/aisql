import os
import pandas as pd
import sqlalchemy
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, footer

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state['conversation'] = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        st.session_state['chat_history'] = None

    st.set_page_config(
        page_title="Chat with your SQL Server database",
        page_icon=":Database:"
    )

    #creating a title for the app
    st.title("Chat Application with Local Database using Generative AI and GPT-4")

    #footer
    st.markdown(footer, unsafe_allow_html=True)

    # Database connection
    if "db_connection" not in st.session_state:
        db_credentials = st.secrets["database"]
        db_url = f"mssql+pyodbc://{db_credentials['username']}:{db_credentials['password']}@{db_credentials['host']}/{db_credentials['database']}?driver=ODBC+Driver+17+for+SQL+Server"
        engine = sqlalchemy.create_engine(db_url)
        st.session_state.db_connection = engine.connect()

    query = st.text_area("Enter your SQL query:")

    if query:
        try:
            with st.spinner("Executing query..."):
                df = pd.read_sql_query(query, st.session_state.db_connection)
                st.write(df)
                # get the text chunks
                text_chunks = get_text_chunks(df.to_string())
                # create vector store
                openai_api_key = st.secrets["openai"]["api_key"]
                vectorstore = get_vectorstore(text_chunks, openai_api_key)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(traceback.format_exc())

    user_question = st.text_input("Ask a question about your data:")
    if user_question and "conversation" in st.session_state and st.session_state.conversation:
        with st.spinner("Processing"):
            handle_userinput(user_question)

# Start
if __name__ == '__main__':
    main()
