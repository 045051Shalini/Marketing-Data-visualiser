import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index.core import Settings
from llama_index.llms.
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

# Configure LLM and embeddings
embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
Settings.embed_model = embed_model

llm = MistralAI(_key="VIScv20xwi7bmBbxZ6SiNJzkh35ZOWvM")
Settings.llm = llm

# Streamlit UI
st.set_page_config(page_title="AI-Powered Data Visualizer", page_icon="📊", layout="wide")
st.title("Chat-Based Data Visualizer 💬📊")

# Upload data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

# Visualization tool
async def visualize_data(query: str) -> str:
    """Generate a visualization based on user input."""
    if data is None:
        return "No data uploaded. Please upload a CSV file."
    
    try:
        if "bar" in query.lower():
            fig = px.bar(data, x=data.columns[0], y=data.columns[1])
        elif "scatter" in query.lower():
            fig = px.scatter(data, x=data.columns[0], y=data.columns[1])
        elif "line" in query.lower():
            fig = px.line(data, x=data.columns[0], y=data.columns[1])
        else:
            return "Unsupported chart type. Try bar, scatter, or line."
        
        st.plotly_chart(fig)
        return "Visualization generated successfully!"
    except Exception as e:
        return f"Error generating visualization: {str(e)}"

visualization_tool = FunctionTool.from_defaults(fn=visualize_data)

# AI Agent
agent_worker = FunctionCallingAgentWorker.from_tools([visualization_tool], llm=llm, verbose=True)
agent = AgentRunner(agent_worker)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Upload a CSV and ask me to visualize it!"}]

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = agent

if prompt := st.chat_input("Ask for a visualization (e.g., 'Show a bar chart')"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = st.session_state.chat_engine.chat(prompt)
        st.write(response.response)
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message)
