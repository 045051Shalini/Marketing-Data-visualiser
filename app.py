import streamlit as st
import pandas as pd
import plotly.express as px
import re
import json
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, ToolMetadata

# Streamlit App Configuration
st.set_page_config(page_title="Groq Data Visualizer", layout="wide")

# Data Loading
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
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass
            return df.dropna(how='all').fillna(0)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

# Groq LLM Setup
def setup_llm():
    with st.sidebar:
        st.header("Groq LLM Configuration")
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
        if api_key:
            try:
                return Groq(model="llama3-70b-8192", api_key=api_key)
            except Exception as e:
                st.error(f"Invalid Groq API key: {str(e)}")
                return None
        return None

# Tool for Generating Python Code
class GeneratePythonCodeTool(BaseTool):
    """Tool for generating Python code to visualize data."""

    metadata = ToolMetadata(
        name="generate_python_code",
        description="Generates Python code using pandas and plotly to visualize the given data.",
    )

    def __init__(self):
        pass

    def __call__(self, input: str, chart_type: str = None) -> str:
        """Use the Groq LLM to generate Python code based on the query."""

        llm = setup_llm()
        if not llm:
            return "Please provide a valid Groq API key."

        prompt = f"""Given the following data analysis request: "{input}", generate Python code using pandas and plotly to visualize the data. Ensure the code is self-contained and executable. Do not explain the code, just return the code. Return the code in a code block. Do not return any other text."""
        if chart_type:
            prompt += f" Generate a {chart_type} chart."

        response = llm.complete(prompt)
        code_match = re.search(r"```python\n(.*?)\n```", response.text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        else:
            return "Code generation failed."

# Tool for Generating Insights
class GenerateInsightsTool(BaseTool):
    """Tool for generating insights from the data."""

    metadata = ToolMetadata(
        name="generate_insights",
        description="Generates insights from the data based on the user's query.",
    )

    def __init__(self):
        pass

    def __call__(self, input: str) -> str:
        """Use the Groq LLM to generate insights based on the query."""

        llm = setup_llm()
        if not llm:
            return "Please provide a valid Groq API key."

        prompt = f"""Given the following data analysis request: "{input}", generate insights from the data. Be concise and focus on key observations. Do not return any code. Do not return any other text."""

        response = llm.complete(prompt)
        return response.text

# Streamlit UI
st.title("Groq Data Visualizer")

# File Upload
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Upload your dataset in CSV format", accept_multiple_files=False)

# Main Interface
df = load_data(uploaded_file)
llm = setup_llm()

if not df.empty:
    query = st.text_area("Analysis Request:", "Generate a bar chart showing final_price distribution with insights", height=100)
    if st.button("Execute Analysis"):
        if llm:
            tools = [GeneratePythonCodeTool(), GenerateInsightsTool()]
            agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, max_iterations=10)
            response = agent.chat(query)

            code_match = re.search(r"```python\n(.*?)\n```", str(response), re.DOTALL)
            insight_match = re.search(r"generate_insights: (.*?)(?:generate_python_code:|Final Answer:)", str(response), re.DOTALL)

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
            st.error("Please configure Groq LLM.")

    # Data Inspection
    with st.sidebar:
        st.header("Dataset Preview")
        st.dataframe(df.head(5))
        st.header("Column Summary")
        st.json({col: {"type": str(df[col].dtype), "unique": df[col].nunique(), "nulls": df[col].isnull().sum()} for col in df.columns})
else:
    st.info("Please upload a CSV file to begin analysis")
