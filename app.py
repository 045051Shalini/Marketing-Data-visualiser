import streamlit as st
import pandas as pd
import plotly.express as px
import json
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
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
            return str(obj) # Convert any remaining non-serializable object to string

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

        llm_provider = st.radio(
            "Select LLM Provider",
            ("Groq", "OpenAI", "Custom"),
            index=0,
            help="Choose your preferred LLM provider"
        )

        if llm_provider == "Groq":
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

        elif llm_provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Get your API key from https://platform.openai.com/"
            )
            if api_key:
                try:
                    return OpenAI(model="gpt-3.5-turbo", api_key=api_key)
                except Exception as e:
                    st.error(f"Invalid OpenAI API key: {str(e)}")
                    return None

        elif llm_provider == "Custom":
            model_name = st.text_input(
                "Custom Model Name",
                help="Enter model name (e.g. 'llama3')"
            )
            api_base = st.text_input(
                "API Base URL",
                help="Local endpoint (e.g. 'http://localhost:11434')"
            )
            if model_name and api_base:
                try:
                    from llama_index.llms.ollama import Ollama
                    return Ollama(model=model_name, base_url=api_base)
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
                    return None

        return None

# Initialize Embeddings
try:
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )
except Exception as e:
    st.error(f"Error initializing embeddings: {e}")

# Initialize Tools
def setup_tools(df):
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
    metadata_engine = VectorStoreIndex.from_documents([Document(text=json.dumps(metadata, cls=EnhancedJSONEncoder))])

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

# Agent Configuration
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
        st.error(f"Error creating agent: {e}")
        return None

# Streamlit UI
st.title("Multi-LLM Data Analysis Platform")

# File Upload Section
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"],
                                   help="Upload your dataset in CSV format",
                                   accept_multiple_files=False)

# Main Interface
df = load_data(uploaded_file)
llm = setup_llm()

if not df.empty:
    query = st.text_area("Analysis Request:",
                        "Generate a bar chart showing final_price distribution with insights",
                        height=100)

    if st.button("Execute Analysis"):
        agent = create_agent(df, llm)

        if agent:
            response = agent.chat(query)

            code_match = re.search(r"``````", response.response, re.DOTALL)
            insight_match = re.search(r"Insights:(.*?)(?=```

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
                st.text(response.response)

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
