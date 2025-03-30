import streamlit as st
import pandas as pd
#import plotly.express as px
import json
import re
from llama_index.readers.json import JSONReader
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
#from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate

# Streamlit app
st.title("Marketing Data Visualization")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("Data Preview:")
    st.write(df.head())

    # Extract headers
    headers = df.columns.tolist()

    # Step 2: Select fields for visualization
    st.sidebar.title("Visualization Configuration")
    x_axis = st.sidebar.selectbox("Select X-axis", headers)
    y_axis = st.sidebar.selectbox("Select Y-axis", headers)

    # Step 3: Select LLM and provide API key
    st.sidebar.title("LLM Configuration")
    llm_option = st.sidebar.selectbox("Select LLM", ["Groq", "OpenAI"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")

    if api_key:
        # Initialize LLM based on selection
        if llm_option == "Groq":
            llm = Groq(model="llama3-70b-8192", api_key=api_key)
        else:
            from llama_index.llms.openai_like import OpenAILike
            llm = OpenAILike(
                temperature=0.7,
                model="gpt-3.5-turbo",
                api_base="http://127.0.0.1:8080/v1",
                api_key=api_key,
                timeout=1000.0,
                is_chat_model=True,
                is_function_calling_model=True,
            )

        # Function to extract metadata
        def return_vals(df, c):
            if isinstance(df[c].iloc[0], (int, float, complex)):
                return [max(df[c]), min(df[c]), df[c].mean()]
            elif isinstance(df[c].iloc[0], pd.Timestamp):
                return [str(max(df[c])), str(min(df[c])), str(df[c].mean())]
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

        # Step 4: Generate visualizations and insights
        query = f"Generate a visualization for '{y_axis}' over '{x_axis}'. Ensure proper styling."

        # Get response from agent
        response = agent.chat(query)

        # Extract code from response using regex (assuming code block is wrapped in triple backticks)
        code_match = re.search(r"```python\n(.*?)```", response.response, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1)
            st.code(extracted_code, language='python')
            exec(extracted_code)
        else:
            st.write("No valid Python code found in response.")
    else:
        st.write("Please enter your API key.")
else:
    st.write("Please upload a CSV file.")
