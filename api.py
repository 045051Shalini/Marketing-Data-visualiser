from flask import Flask, request, jsonify
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import PromptTemplate
import json

app = Flask(__name__)

# Initialize Groq LLM
llm = Groq(model="llama3-70b-8192", api_key="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")

# Load metadata
with open("dataframe.json", "r") as fp:
    dict_ = json.load(fp)

documents = [Document(text=json.dumps(dict_))]
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
Additionally, provide insights and analysis along with the visualizations.
"""
new_prompt = PromptTemplate(new_prompt_txt)
agent.update_prompts({'agent_worker:system_prompt': new_prompt})

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    data = request.json
    query = data.get('query')
    response = agent.chat(query)
    return jsonify(response=response.response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
