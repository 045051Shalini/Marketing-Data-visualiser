import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
import re

# Custom JSON Encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Streamlit App Configuration
st.set_page_config(page_title="Marketing Data Visualizer", layout="wide")

# Data Loading with Enhanced Type Handling
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            
            # Auto-convert date columns
            date_cols = [col for col in df.columns if 'date' in col]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Convert numerical columns
            num_cols = [col for col in df.columns if df[col].dtype == 'object']
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # Ensure all columns are JSON serializable
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(float)
            
            return df.dropna(how='all').fillna(0)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

# LLM Configuration with API Key Management
def setup_llm():
    with st.sidebar:
        st.header("LLM Configuration")
        llm_provider = st.radio(
            "Select LLM Provider",
            ("Groq", "OpenAI", "Custom"),
            index=0
        )
        
        if llm_provider == "Groq":
            api_key = st.text_input("Groq API Key", type="password")
            if api_key:
                return Groq(model="llama3-70b-8192", api_key=api_key)
            
        elif llm_provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                return OpenAI(model="gpt-3.5-turbo", api_key=api_key)
            
        elif llm_provider == "Custom":
            model_name = st.text_input("Model Name")
            api_base = st.text_input("API Base URL")
            if model_name and api_base:
                from llama_index.llms.ollama import Ollama
                return Ollama(model=model_name, base_url=api_base)
                
        return None

# Initialize Embeddings with Error Handling
try:
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )
except Exception as e:
    st.error(f"Embedding initialization error: {str(e)}")
    st.stop()

# Enhanced Metadata Generation
def generate_metadata(df):
    def return_vals(col):
        if pd.api.types.is_numeric_dtype(df[col]):
            return [
                float(df[col].max()),
                float(df[col].min()),
                float(df[col].mean())
            ]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            return [
                str(df[col].max()),
                str(df[col].min()),
                str(df[col].mean())
            ]
        return df[col].astype(str).value_counts().nlargest(10).to_dict()
    
    return {
        col: {
            "type": str(df[col].dtype),
            "stats": return_vals(col),
            "sample": df[col].head(3).astype(str).tolist()
        } for col in df.columns
    }

# Tool Setup with JSON Serialization Fixes
def setup_tools(df):
    metadata = generate_metadata(df)
    try:
        metadata_str = json.dumps(metadata, cls=NumpyEncoder)
        metadata_engine = VectorStoreIndex.from_documents([Document(text=metadata_str)])
    except Exception as e:
        st.error(f"Metadata generation error: {str(e)}")
        st.stop()
    
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
    insight_index = VectorStoreIndex.from_documents([insight_prompt])
    
    return [
        QueryEngineTool(
            query_engine=metadata_engine.as_query_engine(),
            metadata=ToolMetadata(
                name="data_metadata",
                description="Access dataset statistics and column information"
            )
        ),
        QueryEngineTool(
            query_engine=insight_index.as_query_engine(),
            metadata=ToolMetadata(
                name="chart_insights",
                description="Provides analytical frameworks for chart interpretation"
            )
        )
    ]

# Agent Creation with Error Handling
def create_agent(df, llm):
    if llm is None:
        st.error("Please configure LLM settings first")
        return None
    
    try:
        tools = setup_tools(df)
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
        agent.update_prompts({
            "agent_worker:system_prompt": PromptTemplate("""
            You are an advanced data analysis assistant with capabilities to:
            1. Generate professional visualizations
            2. Provide business insights
            3. Handle complex data requests
            4. Explain technical concepts
            Always use available tools for metadata and insights.
            """)
        })
        return agent
    except Exception as e:
        st.error(f"Agent creation failed: {str(e)}")
        st.stop()

# Streamlit UI
st.title("Multi-LLM Marketing Analytics Platform")

# File Upload Section
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    df = load_data(uploaded_file)
    
    if not df.empty:
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

# LLM Initialization
llm = setup_llm()

# Main Analysis Interface
if not df.empty and llm:
    query = st.text_area("Analysis Request:", 
                        "Generate a bar chart showing price distribution with insights",
                        height=100)
    
    if st.button("Execute Analysis"):
        agent = create_agent(df, llm)
        
        if agent:
            response = agent.chat(query)
            
            # Enhanced Code Extraction
            code_match = re.search(r"``````", response.response, re.DOTALL)
            insight_match = re.search(r"Insights:(.*?)(?=```
            
            if code_match:
                code = code_match.group(1).strip()
                st.subheader("Generated Visualization")
                
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
                st.text(response.response)

# Security and Validation
if uploaded_file and not df.empty:
    def validate_csv(df):
        try:
            pd.testing.assert_frame_equal(df, df)
            return True
        except:
            return False
    
    if not validate_csv(df):
        st.error("Invalid data structure detected")
        st.stop()
