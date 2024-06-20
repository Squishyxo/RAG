import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import requests
import os

# Initialize Elasticsearch and model
es = Elasticsearch(
    hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}]
)
model_name = 'all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(model_name)

# Create the Elasticsearch index if it doesn't exist
def create_index():
    if not es.indices.exists(index="documents"):
        es.indices.create(
            index="documents",
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {"type": "dense_vector", "dims": 384}
                    }
                }
            }
        )

create_index()

# Function to index a document
def index_document(doc_text):
    # Generate the embedding for the document
    embedding = sentence_model.encode(doc_text)
    
    # Index the document in Elasticsearch
    es.index(
        index="documents",
        body={
            "text": doc_text,
            "embedding": embedding.tolist()
        }
    )

# Function to call the local language model API
def call_local_model(user_input):
    url = "http://192.168.1.10:11434/api/chat"  # Ensure the correct URL
    payload = {
        "model": "llama3",
        "messages": [
            { "role": "user", "content": user_input }
        ],
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    
    try:
        response_json = response.json()
        print(response_json)  # Print the raw response for debugging
        return response_json
    except ValueError:
        st.error("Failed to decode JSON response")
        return None

# Function to handle the user question
def handle_question(question):
    # Generate the embedding for the question
    query_embedding = sentence_model.encode(question)
    
    # Search for the most relevant documents
    response = es.search(
        index="documents",
        body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding.tolist()}
                    }
                }
            },
            "size": 5
        }
    )
    
    # Extract the relevant text from the response
    retrieved_docs = [hit['_source']['text'] for hit in response['hits']['hits']]
    context = " ".join(retrieved_docs)
    
    # Display the retrieved relevant text
    st.write("### Retrieved Relevant Texts")
    for i, doc in enumerate(retrieved_docs, 1):
        st.write(f"**Document {i}:** {doc}")
    
    # Generate the response using the local language model API
    user_input = f"Question: {question}\nContext: {context}\nAnswer:"
    response_json = call_local_model(user_input)
    
    if response_json and 'message' in response_json:
        response_text = response_json['message']['content'].strip()
        # Append the response to the chat history
        st.session_state.chat_history.append({"question": question, "response": response_text})
    else:
        st.error("Failed to generate a response from the local model.")

# Streamlit app setup
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("ASK PDFs :books:")

# Add a text area to input and index a new document
new_document = st.text_area("Add a new document to the index:")
if st.button("Index Document"):
    if new_document:
        index_document(new_document)
        st.success("Document indexed successfully!")

# Input for user questions
user_question = st.text_input("Ask questions about the uploaded document:")
if user_question:
    handle_question(user_question)

# Display the chat history
for chat in st.session_state.chat_history:
    st.write("**You:**", chat["question"])
    st.write("**Bot:**", chat["response"])
