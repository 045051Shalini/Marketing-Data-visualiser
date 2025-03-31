import streamlit as st
import pandas as pd
import re
import numpy as np
import json
import plotly.express as px
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

def detect_column_types(df):
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or any(isinstance(x, str) and re.search(r'\d{4}-\d{2}-\d{2}', str(x)) for x in df[col].dropna()):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"
    return column_types

def preprocess_data(df):
    df = df.fillna(0)
    column_types = detect_column_types(df)
    return df, column_types

def main():
    st.set_page_config(layout="wide")
    st.title("Marketing Data Visualizer with AI")
    
    # Sidebar inputs
    st.sidebar.header("Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    llm_provider = st.sidebar.selectbox("Choose LLM", ["Groq", "OpenAI"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    
    if uploaded_file and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        st.sidebar.success("Dataset processed successfully!")
        
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram", "pie", "box"])
        generate_button = st.sidebar.button("Generate Visualization")
        
        # LLM selection
        llm = Groq(model="llama3-70b-8192", api_key=api_key) if llm_provider == "Groq" else OpenAI(model="gpt-4", api_key=api_key)
        
        if generate_button:
            if x_axis == y_axis:
                st.error("X-axis and Y-axis cannot be the same.")
                return
            
            # AI prompt
            prompt = PromptTemplate(f"""
                You are an AI specialized in marketing data analysis. 
                Given the dataset provided by the user, generate Python code to visualize the selected X-axis ({x_axis}) and Y-axis ({y_axis}) as a {chart_type} chart.
                Also, provide marketing insights based on trends, customer behavior, and key performance indicators.
            """)
            agent = ReActAgent.from_tools([], llm=llm, verbose=True)
            agent.update_prompts({'agent_worker:system_prompt': prompt})
            query = f"Generate a {chart_type} chart for '{y_axis}' over '{x_axis}' and give marketing insights."
            response = agent.chat(query)
            
            # Extract Python code and insights
            code_match = re.search(r"```python\n(.*?)```", response.response, re.DOTALL)
            insights_match = re.search(r"Insights:\n(.*?)$", response.response, re.DOTALL)
            
            # Generate chart
            try:
                fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f'{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error creating the chart: {e}")
            
            # Show insights
            if insights_match:
                insights_text = insights_match.group(1).strip()
                st.subheader("Marketing Insights")
                st.write(insights_text)
            else:
                st.warning("No insights provided by AI.")
            
            # Show Python code button
            if st.button("Show Python Code"):
                if code_match:
                    st.code(code_match.group(1), language='python')
                else:
                    st.error("No valid Python code found.")
    else:
        st.info("Upload a CSV file and enter an API key to proceed.")

if __name__ == "__main__":
    main()

