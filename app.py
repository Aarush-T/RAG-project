import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -------------------- SETUP --------------------

st.set_page_config(
    page_title="ChatGroq Demo",
    layout="wide"
)

load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# -------------------- VECTOR STORE --------------------

if "vectors" not in st.session_state:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    loader = WebBaseLoader("https://docs.smith.langchain.com/")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(docs[:50])

    st.session_state.vectors = FAISS.from_documents(splits, embeddings)

# -------------------- UI --------------------

st.title("ChatGroq Demo")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}
    """
)

retriever = st.session_state.vectors.as_retriever()

# -------------------- MODERN LCEL CHAIN --------------------

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    start = time.process_time()
    answer = chain.invoke(user_prompt)
    st.write(answer)
    print("Response time:", time.process_time() - start)
