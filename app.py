import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.tools import FunctionTool

# ---------------------- App Setup ----------------------

st.set_page_config(page_title="Marketing Data Visualizer", page_icon="📊", layout="wide")

st.title("📊 Marketing Data Visualizer")

# ---------------------- API & Model Setup ----------------------

# Configure the Groq API (Replace with your actual key)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "your_api_key_here")

llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
Settings.llm = llm

# ---------------------- File Upload ----------------------

uploaded_file = st.file_uploader("📂 Upload your marketing dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # ---------------------- User Input for Chart ----------------------

    user_prompt = st.text_area("📌 Describe the chart you want to generate", "Create a bar chart for sales by category")

    # ---------------------- AI Agent to Generate Code ----------------------

    def generate_chart_code(query: str) -> str:
        """Generate Python code for visualizing the dataset based on user input."""
        code_prompt = f"""
        Given the following dataset:
        {df.head(5).to_csv(index=False)}

        Generate Python code (using matplotlib or altair) to create a visualization that meets this requirement:
        "{query}"
        Ensure the code is self-contained and correctly references column names.
        """
        response = llm.complete(code_prompt)
        return response.text.strip()

    chart_code_tool = FunctionTool.from_defaults(fn=generate_chart_code)
    
    agent_worker = FunctionCallingAgentWorker.from_tools([chart_code_tool], llm=llm, verbose=True)
    agent = AgentRunner(agent_worker)

    if st.button("🔍 Generate Chart"):
        with st.spinner("Generating visualization..."):
            generated_code = agent.chat(user_prompt).response
        
        # ---------------------- Display & Execute Code ----------------------

        st.subheader("📝 Generated Python Code")
        st.code(generated_code, language="python")

        st.subheader("📊 Visualization Output")
        try:
            exec(generated_code, {"plt": plt, "df": df, "alt": alt, "st": st})
        except Exception as e:
            st.error(f"Error executing generated code: {e}")

