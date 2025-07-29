import os
import requests
import streamlit as st
import fitz  # PyMuPDF

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Helper to extract text from PDF using PyMuPDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI
st.set_page_config(page_title="PDF QA with Groq", layout="centered")
st.title("📘 Ask Questions from a PDF using Groq + LangChain")

groq_api_key = st.text_input("🔑 Enter your GROQ API key:", type="password")
pdf_url = st.text_input("🌐 Enter a PDF URL to process:", value="https://alex.smola.org/drafts/thebook.pdf")
query = st.text_input("❓ Ask a question from the PDF:")

if st.button("Submit"):
    if not groq_api_key or not pdf_url or not query:
        st.warning("Please provide all required inputs.")
    else:
        try:
            os.environ["GROQ_API_KEY"] = groq_api_key
            with st.spinner("📥 Downloading and processing PDF..."):

                # Download PDF
                response = requests.get(pdf_url)
                pdf_path = "book.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

                # Extract text from PDF
                raw_text = extract_text_from_pdf(pdf_path)
                if not raw_text.strip():
                    st.error("Could not extract text from the PDF.")
                    st.stop()

                # Chunk text
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.create_documents([raw_text])

                # Create embeddings and vector DB (FAISS)
                embedding = HuggingFaceBgeEmbeddings()
                vectordb = FAISS.from_documents(texts, embedding)

                # Set up LLM
                llm = ChatGroq(model="llama3-70b-8192", temperature=0)

                # Create RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectordb.as_retriever()
                )

                # Ask question
                result = qa_chain.invoke({"query": query})
                st.success(result["result"])

        except Exception as e:
            st.error(f"An error occurred: {e}")
