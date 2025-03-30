import json
import pandas as pd
import requests
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import PromptTemplate

# Initialize Groq LLM
llm = Groq(model="llama3-70b-8192", api_key="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")

# URL of the CSV file on GitHub
csv_url = "https://github.com/045051Shalini/Marketing-Data-visualiser/blob/main/ecommerce_dataset_updated.csv"

# Load CSV data from GitHub
df = pd.read_csv(csv_url)

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
        return [max(df[c]), min(df[c]), df[c].mean()]
    elif isinstance(df[c].iloc[0], pd.Timestamp):
        return [str(max(df[c])), str(min(df[c])), str(df[c].mean())]
    else:
        return list(df[c].value_counts()[:10])

# Store dataset metadata
dict_ = {}
for c in df.columns:
    dict_[c] = {'column_name': c, 'type': str(type(df[c].iloc[0])), 'variable_information': return_vals(df, c)}

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

def lambda_handler(event, context):
    data = json.loads(event['body'])
    query = data.get('query')
    response = agent.chat(query)
    return {
        'statusCode': 200,
        'body': json.dumps({'response': response.response})
    }
