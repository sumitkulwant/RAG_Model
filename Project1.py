import os
import requests
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

st.title("📘 Ask Questions from a PDF using Groq + LangChain")

groq_api_key = st.text_input("Enter your GROQ API key:", type="password")
pdf_url = st.text_input("Enter a PDF URL to process:", value="https://alex.smola.org/drafts/thebook.pdf")
query = st.text_input("Ask a question from the PDF:")

if st.button("Submit") and groq_api_key and pdf_url and query:
    with st.spinner("Downloading and processing PDF..."):
        os.environ["GROQ_API_KEY"] = groq_api_key

        # Download PDF
        response = requests.get(pdf_url)
        pdf_path = "book.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        # Extract text from PDF
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            st.error("Could not extract text from the PDF.")
        else:
            # Split text
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.create_documents([raw_text])

            # Embed and index
            embedding = HuggingFaceBgeEmbeddings()
            vectordb = Chroma.from_documents(texts, embedding)

            # LLM
            llm = ChatGroq(model="llama3-70b-8192", temperature=0)

            # RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordb.as_retriever()
            )

            # Query the document
            result = qa_chain.invoke({"query": query})
            st.success(result["result"])
