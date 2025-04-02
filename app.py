import streamlit as st
import pandas as pd
import plotly.express as px
import json
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.readers.json import JSONReader
from llama_index.embeddings.ollama import OllamaEmbedding
import re

# Streamlit App Configuration
st.set_page_config(page_title="Marketing Data Visualizer", layout="wide")

# Data Preparation (Move this to data loading section)
@st.cache_data
def load_data():
    df = pd.read_csv('/home/ashok/Downloads/ecommerce_dataset_updated.csv')
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], dayfirst=True, errors='coerce')
    numerical_cols = ['Price (Rs.)', 'Discount (%)', 'Final_Price(Rs.)']
    df[numerical_cols] = df[numerical_cols].astype(float)
    return df.fillna(0)

df = load_data()

# Initialize LLM and Embeddings
Settings.llm = Groq(model="llama3-70b-8192", api_key="YOUR_GROQ_API_KEY")
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Initialize Tools
def setup_tools():
    # Data Metadata Tool
    def generate_metadata():
        def return_vals(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                return [float(df[col].max()), float(df[col].min()), float(df[col].mean())]
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                return [str(df[col].max()), str(df[col].min()), str(df[col].mean())]
            return df[col].value_counts().nlargest(10).to_dict()

        metadata = {
            col: {
                "type": str(df[col].dtype),
                "stats": return_vals(col)
            } for col in df.columns
        }
        return metadata

    metadata = generate_metadata()
    metadata_engine = VectorStoreIndex.from_documents([Document(text=json.dumps(metadata))])
    
    # Chart Insight Tool
    insight_prompt = Document(text="""
    When analyzing charts:
    1. Identify trends (monthly/yearly patterns)
    2. Highlight anomalies (spikes/drops)
    3. Compare category performance
    4. Note seasonality patterns
    5. Identify top/bottom performers
    6. Calculate percentage changes
    7. Correlate with marketing events
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
def create_agent():
    tools = setup_tools()
    agent = ReActAgent.from_tools(tools, llm=Settings.llm, verbose=True)
    
    agent.update_prompts({
        "agent_worker:system_prompt": PromptTemplate("""
        You are an intelligent marketing data visualization assistant with capabilities to:
        1. Generate Plotly visualizations
        2. Provide analytical insights
        3. Handle complex data requests
        4. Explain technical concepts
        Always use available tools for metadata and insights.
        """)
    })
    return agent

# Streamlit UI
st.title("AI-Powered Marketing Data Visualizer")
query = st.text_area("Enter your visualization request:", 
                    "Generate a bar chart for Final_Price(Rs.) over Purchase_Date with insights")

if st.button("Generate Visualization"):
    agent = create_agent()
    response = agent.chat(query)
    
    # Code Extraction
    code_match = re.search(r"``````", response.response, re.DOTALL)
    insight_match = re.search(r"Insights:(.*?)(?=```
    
    if code_match:
        code = code_match.group(1)
        st.subheader("Generated Visualization")
        
        try:
            exec(code, globals(), {'df': df, 'px': px, 'st': st})
        except Exception as e:
            st.error(f"Error executing code: {str(e)}")
        
        if insight_match:
            st.subheader("AI Analysis")
            st.markdown(insight_match.group(1).strip())
    else:
        st.error("No valid code generated. Response:")
        st.text(response.response)

# Sidebar Configuration
with st.sidebar:
    st.header("Dataset Preview")
    st.dataframe(df.head(10))
    
    st.header("Column Summary")
    st.json({
        col: str(df[col].dtype) for col in df.columns
    })
