import streamlit as st
import pandas as pd
import plotly.express as px
import re
import numpy as np
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.core.schema import ToolMetadata

def detect_column_types(df):
    """Dynamically detect column types."""
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or any(
            isinstance(x, str) and re.search(r'\d{4}-\d{2}-\d{2}', str(x)) for x in df[col].dropna()
        ):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"
    return column_types

def preprocess_data(df):
    """Handle missing values and convert data types dynamically."""
    df = df.fillna(0)
    column_types = detect_column_types(df)
    return df, column_types

def extract_columns_from_prompt(prompt, df):
    """Extract potential column names from the user prompt."""
    mentioned_cols = [col for col in df.columns if col.lower() in prompt.lower()]
    return mentioned_cols[:2] if len(mentioned_cols) >= 2 else None

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Marketing Data Visualizer with AI Insights")
    
    # Sidebar for input selections
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    
    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        
        st.sidebar.success("✅ Dataset uploaded successfully!")
        
        # Initialize AI agent
        llm = Groq(model="llama3-70b-8192", api_key=api_key)
        
        # Create a style index
        styling_instructions = ["Provide Plotly styling instructions for better visualization."]
        style_index = VectorStoreIndex.from_documents(styling_instructions)
        dataframe_index = VectorStoreIndex.from_documents(df.columns.tolist())
        
        # Build query engines
        dataframe_engine = dataframe_index.as_query_engine(similarity_top_k=1)
        styling_engine = style_index.as_query_engine(similarity_top_k=1)
        
        # Define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=dataframe_engine,
                metadata=ToolMetadata(
                    name="dataframe_index",
                    description="Provides information about the dataset columns and data distribution."
                ),
            ),
            QueryEngineTool(
                query_engine=styling_engine,
                metadata=ToolMetadata(
                    name="styling",
                    description="Provides Plotly styling instructions for data visualization."
                ),
            ),
        ]
        
        # Create agent
        agent = FunctionCallingAgentWorker.from_tools(query_engine_tools, llm=llm)
        runner = AgentRunner(agent)
        
        # Input for the user prompt
        user_prompt = st.text_area(
            "💬 Provide a prompt for generating charts:",
            """For example, you can type:
            'Create a bar chart showing sales over time and analyze trends.'
            """
        )
        generate_button = st.button("🚀 Generate Visualization & Insights")
        
        if generate_button:
            st.subheader("📈 Visualization")
            
            try:
                chart_type_map = {'bar': 'bar', 'line': 'line', 'scatter': 'scatter', 'histogram': 'histogram', 'pie': 'pie', 'box': 'box'}
                chart_type = next((v for k, v in chart_type_map.items() if k in user_prompt.lower()), 'bar')
                
                extracted_cols = extract_columns_from_prompt(user_prompt, df)
                
                if extracted_cols:
                    x_axis, y_axis = extracted_cols
                else:
                    x_axis = df.select_dtypes(include=[np.number]).columns[0]
                    y_axis = df.select_dtypes(include=[np.number]).columns[1] if len(df.select_dtypes(include=[np.number]).columns) > 1 else df.select_dtypes(include=[np.number]).columns[0]
                
                # Generate code using the AI agent
                ai_prompt = f"""
                    Generate a {chart_type} chart using '{x_axis}' on the x-axis and '{y_axis}' on the y-axis with proper styling.
                    Also, provide insights on the trends observed in the data.
                """
                response = runner.chat(ai_prompt)
                
                code_match = re.search(r"```python\n(.*?)```", response.response, re.DOTALL)
                if code_match:
                    extracted_code = code_match.group(1)
                    exec(extracted_code)
                else:
                    st.error("No valid Python code found in AI response.")
                
                st.subheader("💡 AI-Generated Insights")
                st.write(response.response)
                
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                return
            
            if st.button("📜 Show Python Code"):
                st.code(extracted_code, language='python')
    else:
        st.info("Upload a dataset and enter an API key to proceed.")

if __name__ == "__main__":
    main()
