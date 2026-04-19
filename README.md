# RAG-Based Chat Application Using AWS Bedrock

## 📌 Overview
This is a Retrieval-Augmented Generation (RAG) based chat application that allows users to interact with their PDF documents using natural language. The system leverages AWS Bedrock for large language model inference, FAISS for efficient vector search, and Streamlit for an interactive web interface.

---

## 🚀 Features
- Chat with your PDF documents in natural language  
- Semantic search using FAISS vector database  
- Context-aware responses using AWS Bedrock LLMs  
- Upload and process multiple PDF files  
- Simple and interactive Streamlit UI  
- Retrieval-Augmented Generation (RAG) pipeline  

---

## 🧠 Architecture
1. Load PDF documents  
2. Split documents into chunks  
3. Convert text into embeddings using AWS Titan Embeddings  
4. Store embeddings in FAISS vector store  
5. Retrieve relevant chunks based on user query  
6. Pass context to AWS Bedrock LLM for response generation  

---

## 🛠️ Tech Stack
- Python  
- AWS Bedrock  
- FAISS  
- LangChain  
- Streamlit  
- Boto3  

---

## 📌 How it works

Ask questions about your uploaded PDFs, and the system will retrieve the most relevant context and generate accurate answers using AWS Bedrock LLMs.

## 📂 Project Structure
