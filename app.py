import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
import json
import re
import datetime
import numpy as np

# Streamlit page configuration
st.set_page_config(layout="wide")
st.title("📊 Marketing Data Visualizer with AI-Generated Code")

# Sidebar for user input
st.sidebar.header("1️⃣ Upload & Configure")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if uploaded_file and api_key:
    # Load CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Convert date column to datetime format
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], dayfirst=True, errors='coerce')
    # Convert numerical columns to appropriate types
    df['Price (Rs.)'] = df['Price (Rs.)'].astype(float)
    df['Discount (%)'] = df['Discount (%)'].astype(float)
    df['Final_Price(Rs.)'] = df['Final_Price(Rs.)'].astype(float)
    # Handling missing values
    df = df.fillna(0)
    
    # Function to extract metadata
    def return_vals(df, c):
        if isinstance(df[c].iloc[0], (int, float, complex)):
            return [max(df[c]), min(df[c]), np.mean(df[c])]
        elif isinstance(df[c].iloc[0], datetime.datetime):
            return [str(max(df[c])), str(min(df[c])), str(np.mean(df[c]))]
        else:
            return list(df[c].value_counts()[:10])
    
    # Store dataset metadata
    dict_ = {}
    for c in df.columns:
        dict_[c] = {'column_name': c, 'type': str(type(df[c].iloc[0])), 'variable_information': return_vals(df, c)}
    
    # Save metadata as JSON
    with open("dataframe.json", "w") as fp:
        json.dump(dict_, fp)
    
    # Load metadata
    reader = JSONReader()
    documents = reader.load_data(input_file='dataframe.json')
    dataframe_index = VectorStoreIndex.from_documents(documents)
    
    # Define styling instructions
    styling_instructions = [
        Document(text="""
            For a line chart, use plotly_white template, set x & y axes line to 0.2, grid width to 1.
            Always include a bold title, use multiple colors for multiple lines.
            Annotate min & max values, show numbers in 'K' or 'M' if >1000/100000.
            Show percentages in 2 decimal points with '%'.
        """),
        Document(text="""
            For a bar chart, use plotly_white template, set x & y axes line to 0.2, grid width to 1.
            Include a bold title, use multiple colors, annotate values on y-axis.
            Display numbers in 'K' or 'M' if >1000/100000, percentages in 2 decimal points with '%'.
        """),
        Document(text="""
            General instructions: Use plotly_white template, set x & y axes line to 0.2, grid width to 1.
            Include a bold title, display numbers in 'K' or 'M' if >1000/100000.
            Show percentages in 2 decimal points with '%'.
        """),
    ]
    
    # Create style index
    style_index = VectorStoreIndex.from_documents(styling_instructions)
    
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
    
    # Initialize Groq LLM
    llm = Groq(model="llama3-70b-8192", api_key=api_key)
    
    # Create agent
    agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
    
    # Adjust agent prompt
    new_prompt_txt = """
    You are designed to help with marketing data visualization in Plotly using Python.
    You have access to tools providing dataset insights and styling instructions.
    Please follow a structured approach, use JSON format for tool actions, and ensure correctness.
    """
    new_prompt = PromptTemplate(new_prompt_txt)
    agent.update_prompts({'agent_worker:system_prompt': new_prompt})
    
    # Show first few rows of the data
    st.write("### Preview of the dataset", df.head())
    
    # User prompt for AI-generated code
    user_prompt = st.text_area("💬 Custom AI Prompt (Optional)", "Generate a bar chart based on categories in the data.")
    
    if st.button("🚀 Generate Visualization & Insights"):
        # Generate AI prompt based on user input
        ai_prompt = f"""
        Based on the following dataset, generate Python code to create a chart:
        {df.head().to_string()}
        {user_prompt}
        """
        
        # Get response from agent
        response = agent.chat(ai_prompt)
        
        # Extract code from response using regex (assuming code block is wrapped in triple backticks)
        code_match = re.search(r"```python\n(.*?)```", response.response, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1)
            st.subheader("💬 Generated Python Code:")
            st.code(extracted_code, language="python")
            
            # Try to execute the extracted code and display the chart using Plotly
            try:
                exec(extracted_code)
                # Assuming the chart is stored in a variable called `fig`
                if 'fig' in locals():
                    st.plotly_chart(fig)  # Display the Plotly chart
                else:
                    st.error("No chart generated. Please check the generated code.")
            except Exception as e:
                st.error(f"Error executing the generated code: {e}")
        else:
            st.error("No valid Python code found in response.")
else:
    st.info("Upload a dataset and enter your API key to proceed.")
