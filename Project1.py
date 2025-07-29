import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Groq
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="PDF QA with RAG", layout="wide")
st.title("📄 Ask Questions from your PDF (Groq + LLaMA3)")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Groq API Key
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

query = st.text_input("Ask a question from the document")

if uploaded_file and groq_api_key and query:
    with st.spinner("Processing..."):
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load PDF content
        loader = UnstructuredFileLoader(tmp_path)
        documents = loader.load()

        # Split into small chunks to reduce token size
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(docs, embedding=embeddings)

        # Use MMR retriever and fewer chunks to avoid hitting token limit
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

        # Set up Groq LLM
        llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)

        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Run QA
        response = qa.run(query)
        st.markdown("### 📌 Answer")
        st.success(response)

        # Clean up
        os.remove(tmp_path)
