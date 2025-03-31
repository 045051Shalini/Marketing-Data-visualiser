import streamlit as st
import pandas as pd
import re
import plotly.express as px
from llama_index.llms.groq import Groq

# Function to detect column types
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

# Function to preprocess data
def preprocess_data(df):
    df = df.fillna(0)
    column_types = detect_column_types(df)
    return df, column_types

def main():
    st.set_page_config(layout="wide")
    st.title("\ud83d\udcca Marketing Data Visualizer with AI Insights")
    
    # Sidebar setup
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    
    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        
        st.sidebar.success("✅ Dataset uploaded successfully!")
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram", "pie", "box"])
        user_prompt = st.sidebar.text_area("\ud83d\udcac Custom AI Prompt (Optional)", "Provide marketing insights from this data.")
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
            
            # Initialize Groq LLM
            llm = Groq(model="llama3-70b-8192", api_key=api_key)
            
            # AI Insights Generation
            ai_prompt = f"""
                You are an AI specialized in marketing analysis.
                Analyze the dataset and provide insights using the provided visualization.
                Focus on:
                - Key trends in {x_axis} and {y_axis}.
                - Anomalies, patterns, or seasonal effects.
                - Insights that can help improve marketing decisions.
                Data:
                {df.to_string(index=False)}
                {user_prompt}
            """
            
            response = llm.complete(ai_prompt)
            insights_text = response.text if response.text else "No insights provided."
            
            st.subheader("💡 AI-Generated Insights")
            st.write(insights_text)
            
            # Show Python code
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
