import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index.core import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq

def detect_column_types(df):
    """Detect column types dynamically."""
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"
    return column_types

def preprocess_data(df):
    """Fill missing values and detect column types."""
    df = df.fillna(0)
    column_types = detect_column_types(df)
    return df, column_types

def generate_ai_insights(llm, df, x_axis, y_axis, user_prompt):
    """Generate AI insights using structured data and a well-defined prompt."""
    data_sample = df[[x_axis, y_axis]].head(10).to_json()
    prompt = f"""
    You are a data analyst. Analyze the provided dataset and visualization to extract meaningful insights:
    - Identify trends, patterns, and anomalies.
    - Provide business implications for marketing decisions.
    - Base insights on the given X-axis ({x_axis}) and Y-axis ({y_axis}).
    - Use the following data sample for reference: {data_sample}
    - User-specified request: {user_prompt}
    """
    return llm.complete(prompt)

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Marketing Data Visualizer with AI Insights")
    
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    llm_choice = st.sidebar.selectbox("Select AI Model", ["Groq (Llama3-70B)", "OpenAI (GPT-4)"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    
    if uploaded_file and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        
        st.sidebar.success("✅ Dataset uploaded!")
        x_axis = st.sidebar.selectbox("X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Y-axis", df.columns)
        chart_type = st.sidebar.selectbox("Chart Type", ["bar", "line", "scatter", "histogram", "pie", "box"])
        user_prompt = st.sidebar.text_area("💬 Custom AI Prompt", "Analyze trends and insights.")
        generate_button = st.sidebar.button("🚀 Generate")
        
        if generate_button:
            st.subheader("📈 Visualization")
            fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f'{chart_type.capitalize()} Visualization')
            fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
            st.plotly_chart(fig)
            
            # Initialize LLM
            llm = Groq(model="llama3-70b-8192", api_key=api_key) if "Groq" in llm_choice else OpenAI(model="gpt-4", api_key=api_key)
            
            # Get AI insights
            ai_response = generate_ai_insights(llm, df, x_axis, y_axis, user_prompt)
            insights = ai_response.text if ai_response else "No insights generated."
            
            st.subheader("💡 AI-Generated Insights")
            st.write(insights)
            
            with st.expander("📜 Show Python Code"):
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
