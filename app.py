import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.groq import Groq

# Set API Key
GROQ_API_KEY = "your_groq_api_key"

# Initialize LLM
llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

st.title("Streamlit + Groq + LlamaIndex")

# User input
query = st.text_input("Ask a question about the data:")

if query:
    # Load documents
    docs = [("Groq is a fast inference engine for LLMs.")]
    
    # Build index
    index = VectorStoreIndex.from_documents(docs)

    # Query engine
    query_engine = index.as_query_engine(llm=llm)
    
    # Get response
    response = query_engine.query(query)

    st.write("### Answer:")
    st.write(response)
