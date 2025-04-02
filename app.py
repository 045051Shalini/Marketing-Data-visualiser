import streamlit as st
import pandas as pd
import plotly.express as px
import json
from llama_index.core import VectorStoreIndex, PromptTemplate, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import re
import numpy as np

# Custom JSON Encoder to handle NumPy types and other non-serializable objects
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()  # Convert pandas Series/DataFrames to dict
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # Convert any remaining non-serializable object to string

# Streamlit App Configuration
st.set_page_config(page_title="LLM-Powered Data Visualizer", layout="wide")

# Data Loading with Upload Functionality
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            
            date_cols = [col for col in df.columns if 'date' in col]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            num_cols = [col for col in df.columns if df[col].dtype == 'object']
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            return df.dropna(how='all').fillna(0)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

# LLM Configuration
def setup_llm():
    with st.sidebar:
        st.header("LLM Configuration")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your API key from https://console.groq.com/"
        )
        if api_key:
            try:
                return Groq(model="llama3-70b-8192", api_key=api_key)
            except Exception as e:
                st.error(f"Invalid Groq API key: {str(e)}")
                return None
        return None

# Initialize Embeddings
def setup_embeddings():
    try:
        return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

# Initialize Tools
def setup_tools(df, embed_model):
    def generate_metadata():
        def return_vals(col):
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return [float(df[col].max()), float(df[col].min()), float(df[col].mean())]
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    return [str(df[col].max()), str(df[col].min()), str(df[col].mean())]
                return df[col].astype(str).value_counts().nlargest(10).to_dict()
            except Exception as e:
                st.error(f"Error generating stats for column {col}: {e}")
                return {}

        metadata = {}
        for col in df.columns:
            try:
                metadata[col] = {
                    "type": str(df[col].dtype),
                    "stats": return_vals(col)
                }
            except Exception as e:
                st.error(f"Error generating metadata for column {col}: {e}")
                metadata[col] = {"type": "unknown", "stats": {}}
        return metadata

    metadata = generate_metadata()
    metadata_engine = VectorStoreIndex.from_documents([Document(text=json.dumps(metadata, cls=EnhancedJSONEncoder))], embed_model=embed_model).as_query_engine()
    
    insight_prompt = Document(text="""
    When analyzing charts:
    1. Identify trends and patterns
    2. Highlight statistical outliers
    3. Compare categorical performance
    4. Note temporal patterns
    5. Identify extremes
    6. Calculate percentage changes
    7. Correlate external events
    """)
    insight_index = VectorStoreIndex.from_documents([insight_prompt], embed_model=embed_model).as_query_engine()
    
    return metadata_engine, insight_index

# Streamlit UI
st.title("LLM-Powered Data Visualizer")

# File Upload Section
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], 
                                    help="Upload your dataset in CSV format",
                                    accept_multiple_files=False)

# Main Interface
df = load_data(uploaded_file)
llm = setup_llm()
embed_model = setup_embeddings()

if not df.empty:
    query = st.text_area("Analysis Request:", 
                        "Generate a bar chart showing final_price distribution with insights",
                        height=100)
    
    if st.button("Execute Analysis"):
        if llm and embed_model:
            metadata_engine, insight_index = setup_tools(df, embed_model)

            metadata_response = metadata_engine.query(query)
            insight_response = insight_index.query(query)

            full_query = f"{query}. Use this metadata: {metadata_response.response}. Use these insights: {insight_response.response}"

            response = llm.complete(full_query)

            code_match = re.search(r"```python\n(.*?)\n```", response.text, re.DOTALL)
            insight_match = re.search(r"Insights:(.*?)(?=```)", response.text, re.DOTALL)

            if code_match:
                code = code_match.group(1)
                st.subheader("Visualization")
                
                try:
                    exec(code, globals(), {'df': df, 'px': px, 'st': st})
                except Exception as e:
                    st.error(f"Execution error: {str(e)}")
                    st.code(code)
                
                if insight_match:
                    st.subheader("AI Insights")
                    st.markdown(insight_match.group(1).strip())
            else:
                st.error("Code generation failed")
                st.text(response.text)
        else:
            st.error("Please configure LLM and/or Embeddings.")

    # Data Inspection
    with st.sidebar:
        st.header("Dataset Preview")
        st.dataframe(df.head(5))
        
        st.header("Column Summary")
        st.json({
            col: {
                "type": str(df[col].dtype),
                "unique": df[col].nunique(),
                "nulls": df[col].isnull().sum()
            } for col in df.columns
        })
else:
    st.info("Please upload a CSV file to begin analysis")

# Security Features
def validate_csv(file):
    try:
        pd.read_csv(file)
        return True
    except:
        return False

if uploaded_file and not validate_csv(uploaded_file):
    st.error("Invalid CSV file detected")
    st.stop()
