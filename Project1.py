# RAG_Model
import os
import requests
import streamlit as st

from langchain_community.document_loaders import UnstructuredFileIOLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Set up Streamlit UI
st.title("📘 Ask Questions from a PDF using Groq + LangChain")

groq_api_key = st.text_input("Enter your GROQ API key:", type="password")
pdf_url = st.text_input("Enter a PDF URL to process:", value="https://alex.smola.org/drafts/thebook.pdf")

query = st.text_input("Ask a question from the PDF:")

if st.button("Submit") and groq_api_key and pdf_url and query:
    with st.spinner("Downloading and processing PDF..."):
        os.environ["GROQ_API_KEY"] = groq_api_key

        # Download PDF
        response = requests.get(pdf_url)
        with open("book.pdf", "wb") as f:
            f.write(response.content)

        # Load document
        with open("book.pdf", "rb") as f:
            loader = UnstructuredFileIOLoader(file=f)
            documents = loader.load()

        # Chunk text
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Embeddings and VectorDB
        embedding = HuggingFaceBgeEmbeddings()
        vectordb = Chroma.from_documents(texts, embedding)

        # LLM setup
        llm = ChatGroq(model="llama3-70b-8192", temperature=0)

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever()
        )

        # Run query
        result = qa_chain.invoke({"query": query})
        st.success(result["result"])
