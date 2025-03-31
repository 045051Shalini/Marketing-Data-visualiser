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
    """Dynamically detect column types."""
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
    """Handle missing values and convert data types dynamically."""
    df = df.fillna(0)
    column_types = detect_column_types(df)
    return df, column_types

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Marketing Data Visualizer with AI Insights")
    
    # Sidebar for input selections
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    llm_choice = st.sidebar.selectbox("Select AI Model", ["Groq (Llama3-70B)", "OpenAI (GPT-4)"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    
    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        
        st.sidebar.success("✅ Dataset uploaded successfully!")
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram", "pie", "box"])
        user_prompt = st.sidebar.text_area("💬 Custom AI Prompt (Optional)", "Provide marketing insights from this data.")
        generate_button = st.sidebar.button("🚀 Generate Visualization & Insights")
        
        if generate_button:
            st.subheader("📈 Visualization")
            try:
                fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f'{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                return
            
            # Initialize LLM
            llm = Groq(model="llama3-70b-8192", api_key=api_key) if "Groq" in llm_choice else OpenAI(model="gpt-4", api_key=api_key)
            
            # Prompt AI for insights
            ai_prompt = f"""
                You are an AI specialized in marketing analysis.
                Analyze the dataset and provide insights specific to marketing trends, consumer behavior, and campaign performance.
                Focus on:
                - Key trends in {x_axis} and {y_axis}.
                - Anomalies, patterns, or seasonal effects.
                - Insights that can help improve marketing decisions.
                {user_prompt}
            """
            agent = ReActAgent.from_tools([], llm=llm, verbose=True)
            response = agent.chat(ai_prompt)
            
            # Extract insights
            insights_match = re.search(r"Insights:\n(.*?)$", response.response, re.DOTALL)
            insights_text = insights_match.group(1).strip() if insights_match else "No insights provided."
            
            st.subheader("💡 AI-Generated Insights")
            st.write(insights_text)
            
            if st.button("📜 Show Python Code"):
                st.code(f"""
                import plotly.express as px
                fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
                fig.show()
                """, language='python')
    else:
        st.info("Upload a dataset and enter an API key to proceed.")

if __name__ == "__main__":
    main()
