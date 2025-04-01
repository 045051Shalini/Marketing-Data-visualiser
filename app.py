import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq

# Streamlit page configuration
st.set_page_config(layout="wide")
st.title("📊 Marketing Data Visualizer with AI Insights")

# Sidebar for user input
st.sidebar.header("1️⃣ Upload & Configure")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if uploaded_file and api_key:
    # Load CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Show first few rows of the data
    st.write("### Preview of the dataset", df.head())
    
    # Column selection for visualization
    x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
    chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram"])
    user_prompt = st.text_area("💬 Custom AI Prompt (Optional)", "Analyze the trends in this data.")

    if st.button("🚀 Generate Visualization & Insights"):
        # Generate the selected chart type using Plotly
        fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart")
        st.plotly_chart(fig)
        
        # Initialize the Groq model (use the provided API key)
        llm = Groq(model="llama3-70b-8192", api_key=api_key)
        
        # Initialize ReActAgent with the Groq LLM
        agent = ReActAgent.from_tools([], llm=llm, verbose=True)
        
        # Extract and prepare data from the selected columns for insight generation
        data_description = f"""
        The dataset contains the following columns: {', '.join(df.columns)}.
        The chart visualizes the relationship between {x_axis} (x-axis) and {y_axis} (y-axis) using a {chart_type} chart.
        Below is the data used for the chart:
        {df[[x_axis, y_axis]].to_dict(orient='records')}
        """
        
        # Generate the AI prompt for insights
        ai_prompt = f"""
        Please analyze the data provided and the chart context. The chart shows the relationship between {x_axis} (x-axis) and {y_axis} (y-axis).
        Here is the data used for the chart:
        {data_description}
        {user_prompt}
        """
        
        # Get insights from the agent
        response = agent.chat(ai_prompt)
        insights_text = response.response if response.response else "No insights provided by AI."

        # Display AI-generated insights
        st.subheader("💡 AI-Generated Insights")
        st.write(insights_text)

else:
    st.info("Upload a dataset and enter your API key to proceed.")
