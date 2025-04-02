import streamlit as st
import pandas as pd
import plotly.express as px
import json
from llama_index.core import VectorStoreIndex, Document, get_response_synthesizer
from llama_index.llms.groq import Groq
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from sentence_transformers import SentenceTransformer
from llama_index.embeddings import HuggingFaceEmbedding # import HuggingFaceEmbedding
import re
import numpy as np

# ... (rest of your code)

# Initialize Embeddings
def setup_embeddings():
    try:
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", model=model) #wrap the model
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

# ... (rest of your code)
