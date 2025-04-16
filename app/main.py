import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

st.title("Self-Hosted AI Document Q&A")

uploaded_file = st.file_uploader("Upload document", type=["pdf", "docx", "txt", "xlsx"])
query = st.text_input("Ask a question about the document:")

if uploaded_file:
    path = f"docs/{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    loader = UnstructuredFileLoader(path)
    docs = loader.load()

    embeddings = OllamaEmbeddings(model="llama3")
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=Ollama(model="llama3"), retriever=retriever)

    if query:
        with st.spinner("Processing..."):
            answer = qa.run(query)
            st.success(answer)
