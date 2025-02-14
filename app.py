import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from qdrant_client.http.models import VectorParams, Distance, SearchParams
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
def query_ai_page():
    def query_embedding(query, api_key):
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        return embeddings_model.embed_query(query)

    def search_related_text(query_embedding, collection_name, top_k=3):
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            search_params=SearchParams(hnsw_ef=128),
            limit=top_k
        )
        return [result.payload["text"] for result in search_results.points]

    def generate_response(retriever, api_key, user_query):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.6, google_api_key=api_key)
        conversation = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, verbose=False)
        return conversation.run(user_query)

    def collections_list(qdrant_client):
        
        collections = [col.name for col in qdrant_client.get_collections().collections]
        return collections

    def pipeline(api_key, qdrant_client, collection_name, user_query, top_k=3):
        query_embeddings = query_embedding(user_query, api_key)
        related_texts = search_related_text(query_embeddings, collection_name, top_k)
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings_model,
            content_payload_key="text"
        )
        retriever = vector_store.as_retriever()
        return generate_response(retriever, api_key, user_query)

    st.title("AI Query Pipeline")
    st.write("Looking for specific information? Type your question and select the Hospital ID (Name) to get results instantly!")

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    api_key = os.getenv("GOOGLE_API_KEY")

    # Fetch unique IDs for the dropdown
    with st.spinner("Fetching Hospitals Names..."):
        try:
            hospitals = collections_list(qdrant_client)
            if not hospitals:
                hospitals = ["Hospital is not stored yet."]  # Fallback option if none are found
        except Exception as e:
            st.error(f"Error fetching Hospiatl ID: {e}")
            hospitals = ["Error fetching Hospiatl ID"]

    
    user_query = st.text_input("Enter your Query:")    
    collection_name = st.selectbox("Select Hospiatl ID/Name:", options=hospitals)
    
    if st.button("Run Query"):
        if api_key and qdrant_client and collection_name and user_query:
            try:
                with st.spinner("Processing your query..."):
                    response = pipeline(api_key, qdrant_client, collection_name, user_query)
                st.write("Generated Response:", response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide all the required inputs.")


query_ai_page()
