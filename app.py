import json
import os
import sys
import boto3
import streamlit as st

## Using Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.messages import HumanMessage
from langchain_aws.chat_models import ChatBedrockConverse

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# FAISS Vector Embedding And Vector Store

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)


## Ingesting data
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_nova_llm():
    llm = ChatBedrockConverse(
        model="amazon.nova-micro-v1:0",
        client=bedrock
    )
    return llm
    

def get_mistral_llm():
    llm = ChatBedrockConverse(
        model="mistral.ministral-3-3b-instruct",
        client=bedrock
    )
    return llm

prompt_template = """

Human: Based on the context provided below, answer the question at the end in a clear and concise manner. 
Your response should include a well-explained summary of at least 250 words. If the information needed to answer 
the question is not present in the context, simply state that you don’t know rather than guessing or making up an answer
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):

    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = PROMPT.format(context=context, question=query)

    response = llm.invoke([
    HumanMessage(content=prompt)
])

    return response.content


def main():
    st.set_page_config("Chat with PDF")
    
    st.header("Chat with PDF using AWS Bedrock 🌐")

    user_question = st.text_input("Ask a question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Update Vectors"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Ask Nova Micro"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_nova_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Ask Mistral"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_mistral_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()














