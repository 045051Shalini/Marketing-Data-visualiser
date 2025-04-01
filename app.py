import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq

st.set_page_config(layout="wide")
st.title("📊 Marketing Data Visualizer with AI Insights")

# Sidebar for input
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if uploaded_file and api_key:
    df = pd.read_csv(uploaded_file)
    
    # Column selection
    x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
    chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter"])
    user_prompt = st.text_area("💬 Ask AI for Insights", "Analyze the trends in this data.")

    if st.button("🚀 Generate Visualization & Insights"):
        # Generate chart
        fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart")
        st.plotly_chart(fig)

        # Initialize LLM (Groq API)
        llm = Groq(model="llama3-70b-8192", api_key=api_key)
        agent = ReActAgent.from_tools([], llm=llm, verbose=True)

        # Pass the raw data and user prompt for AI insight generation
        ai_prompt = f"""
        I am analyzing a chart that shows the relationship between {x_axis} (x-axis) and {y_axis} (y-axis) using a {chart_type} chart.
        Here is the data from the chart:
        {df[[x_axis, y_axis]].to_dict(orient='records')}
        Please analyze this data and provide insights on the trends, patterns, and any key takeaways.
        {user_prompt}
        """
        
        # Get insights
        response = agent.chat(ai_prompt)
        insights_text = response.response if response.response else "No insights provided by AI."

        st.subheader("💡 AI-Generated Insights")
        st.write(insights_text)
