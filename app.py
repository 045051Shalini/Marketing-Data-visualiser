import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import re
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent

def preprocess_data(df):
    """Handle missing values and convert data types dynamically."""
    df = df.fillna(0)
    return df

def generate_metadata(df):
    """Generate dataset metadata for AI processing."""
    def return_vals(column):
        if df[column].dtype in [np.int64, np.float64]:
            return [int(df[column].max()), int(df[column].min()), float(df[column].mean())]
        else:
            return list(df[column].value_counts().head(10).index)
    
    metadata = {col: {'column_name': col, 'type': str(df[col].dtype), 'variable_information': return_vals(col)} for col in df.columns}
    return metadata

def generate_visualization(df, x_axis, y_axis, chart_type):
    """Create the requested visualization."""
    try:
        fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f'{chart_type.capitalize()} Visualization')
        fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
        return fig
    except Exception as e:
        st.error(f"Error creating the chart: {e}")
        return None

def main():
    st.title("Marketing Data Visualizer with AI Insights")

    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)
        metadata = generate_metadata(df)

        with open("dataframe.json", "w") as fp:
            json.dump(metadata, fp, default=str)

        st.sidebar.success("Dataset uploaded and processed successfully!")

        x_axis = st.sidebar.selectbox("Select X-axis:", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis:", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type:", ["bar", "line", "scatter", "histogram", "pie", "box"])

        generate_button = st.sidebar.button("Generate Visualization")

        if generate_button:
            if x_axis == y_axis:
                st.error("X-axis and Y-axis cannot be the same. Please select different columns.")
                return
            
            fig = generate_visualization(df, x_axis, y_axis, chart_type)
            if fig:
                st.plotly_chart(fig)
            
            # AI Insights
            llm = Groq(model="llama3-70b-8192", api_key="YOUR_GROQ_API_KEY")
            prompt = PromptTemplate(f"""
                Analyze the dataset and provide insights into marketing trends, customer behavior, and sales performance.
                Consider key factors such as sales patterns, customer demographics, and campaign success.
            """)
            
            agent = ReActAgent.from_tools([], llm=llm, verbose=True)
            agent.update_prompts({'agent_worker:system_prompt': prompt})
            response = agent.chat("Provide insights from the dataset.")

            insights_match = re.search(r"Insights:\n(.*?)$", response.response, re.DOTALL)
            
            if insights_match:
                insights_text = insights_match.group(1).strip()
                st.subheader("Marketing Insights")
                st.write(insights_text)
            else:
                st.warning("No insights provided by the AI.")

if __name__ == "__main__":
    main()
