import streamlit as st
import pandas as pd
import plotly.express as px
import re
import json
import numpy as np
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
    st.title("📊 Marketing Data Visualizer with AI")
    
    # Sidebar for input selections
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    llm_choice = st.sidebar.selectbox("Select AI Model", ["Groq (Llama3-70B)", "OpenAI (GPT-4)"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    
    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        
        st.sidebar.success("✅ Dataset uploaded successfully!")
        
        # Input for the user prompt
        user_prompt = st.text_area(
            "💬 Provide a prompt for generating charts and AI insights:",
            """For example, you can type:
            'Create a bar chart showing sales over time'
            """
        )
        generate_button = st.button("🚀 Generate Visualization")
        
        if generate_button:
            st.subheader("📈 Visualization")
            
            try:
                # Check if the user specified chart type in the prompt
                chart_type_map = {'bar': 'bar', 'line': 'line', 'scatter': 'scatter', 'histogram': 'histogram', 'pie': 'pie', 'box': 'box'}
                chart_type = next((v for k, v in chart_type_map.items() if k in user_prompt.lower()), 'bar')
                
                # Extract column names from the prompt if specified
                extracted_cols = extract_columns_from_prompt(user_prompt, df)
                
                if extracted_cols:
                    x_axis, y_axis = extracted_cols
                else:
                    x_axis = df.select_dtypes(include=[np.number]).columns[0]
                    y_axis = df.select_dtypes(include=[np.number]).columns[1] if len(df.select_dtypes(include=[np.number]).columns) > 1 else df.select_dtypes(include=[np.number]).columns[0]
                
                # Generate the plot using the user prompt
                fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f'{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig)
            
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                return
            
            # Initialize LLM based on the user's choice
            llm = Groq(model="llama3-70b-8192", api_key=api_key) if "Groq" in llm_choice else OpenAI(model="gpt-4", api_key=api_key)
            
            # Create AI prompt to provide insights
            ai_prompt = f"""
                You are an AI specialized in marketing analytics. Given the dataset and the generated visualization:
                - Identify key trends in '{x_axis}' and '{y_axis}'.
                - Provide actionable marketing insights based on the chart.
                - Analyze anomalies, patterns, seasonal variations, and customer behavior.
                - Ensure insights are specific to the provided dataset and visualization.
                {user_prompt}
            """
            
            agent = ReActAgent.from_tools([], llm=llm, verbose=True)
            response = agent.chat(ai_prompt)
            
            # Extract insights
            insights_text = response.response if response.response else "No insights provided by AI."
            
            st.subheader("💡 AI-Generated Insights")
            st.write(insights_text)
            
            # Show Python Code Button
            if st.button("📜 Show Python Code"):
                python_code = f"""
                import plotly.express as px
                fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
                fig.show()
                """
                st.code(python_code, language='python')
    else:
        st.info("Upload a dataset and enter an API key to proceed.")

if __name__ == "__main__":
    main()
