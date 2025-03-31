import streamlit as st
import pandas as pd
import plotly.express as px
import re
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent

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

def generate_figure(df, x_axis, y_axis, chart_type):
    """Generate chart based on user's choice."""
    if chart_type == "bar":
        return px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart of {y_axis} vs {x_axis}")
    elif chart_type == "line":
        return px.line(df, x=x_axis, y=y_axis, title=f"Line Chart of {y_axis} vs {x_axis}")
    elif chart_type == "scatter":
        return px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot of {y_axis} vs {x_axis}")
    elif chart_type == "histogram":
        return px.histogram(df, x=x_axis, title=f"Histogram of {y_axis}")
    elif chart_type == "pie":
        return px.pie(df, names=x_axis, title=f"Pie Chart of {x_axis}")
    elif chart_type == "box":
        return px.box(df, x=x_axis, y=y_axis, title=f"Box Plot of {y_axis} vs {x_axis}")
    else:
        return None

def get_ai_insights(llm, ai_prompt):
    """Generate insights using the LLM."""
    agent = ReActAgent.from_tools([], llm=llm, verbose=True)
    response = agent.chat(ai_prompt)
    return response.response.strip() if response else "No insights generated."

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Marketing Data Visualizer with AI Insights")

    # Sidebar for input configurations
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    # Choose LLM model and API key
    llm_choice = st.sidebar.selectbox("Select AI Model", ["Groq (Llama3-70B)"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")

    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)

        st.sidebar.success("✅ Dataset uploaded successfully!")
        
        # Let the user input English text to describe the insights
        user_prompt = st.sidebar.text_area("💬 Custom AI Prompt (Optional)", "Provide insights from the visualized data.")
        
        # Input x and y columns from the dataset
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        
        # Let user describe the visualization type
        chart_type = st.sidebar.text_input("Enter chart type", "scatter")  # Free-form text input
        
        # Button to generate insights and visualization
        generate_button = st.sidebar.button("🚀 Generate Visualization & Insights")

        if generate_button:
            st.subheader("📈 Visualization")
            try:
                fig = generate_figure(df, x_axis, y_axis, chart_type)
                if fig:
                    st.plotly_chart(fig)
                else:
                    st.error(f"Error generating chart of type {chart_type}")
                    return
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                return

            # Initialize LLM
            llm = Groq(model="llama3-70b-8192", api_key=api_key)
            
            # Construct the prompt to send to the AI model
            ai_prompt = f"""
                You are an AI specialized in marketing analysis. 
                You have the following dataset with these columns: {list(df.columns)}.
                Analyze the chart type: {chart_type}.
                Focus on trends, anomalies, patterns, and give marketing-specific insights related to {x_axis} and {y_axis}.
                The following data points may be useful:
                {df[[x_axis, y_axis]].describe().to_dict()}
                {user_prompt}
            """
            
            # Get insights from AI
            ai_insights = get_ai_insights(llm, ai_prompt)
            
            st.subheader("💡 AI-Generated Insights")
            st.write(ai_insights)
            
            if st.button("📜 Show Python Code"):
                st.code(f"""
                fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
                fig.show()
                """, language='python')
    else:
        st.info("Upload a dataset and enter an API key to proceed.")

if __name__ == "__main__":
    main()
