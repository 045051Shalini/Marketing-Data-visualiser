import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.groq import Groq
import pandas as pd
import re
import numpy as np
import json
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

def generate_metadata(df):
    """Generate dataset metadata for AI processing."""
    def return_vals(column):
        if df[column].dtype in [np.int64, np.float64]:
            return [int(df[column].max()), int(df[column].min()), float(df[column].mean())]
        elif df[column].dtype == 'datetime64[ns]':
            return [str(df[column].max()), str(df[column].min())]
        else:
            return list(df[column].value_counts().head(10).index)
    
    metadata = {col: {'column_name': col, 'type': str(df[col].dtype), 'variable_information': return_vals(col)} for col in df.columns}
    return metadata

def is_complex_chart(df, x_axis, y_axis):
    """Determine if the chart is too complex based on the number of unique values."""
    x_unique = len(df[x_axis].unique())
    y_unique = len(df[y_axis].unique())
    return x_unique > 20 or y_unique > 20

def main():
    st.title("Marketing Data Visualizer with Groq AI")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df, column_types = preprocess_data(df)
            metadata = generate_metadata(df)
            
            # Ensure proper conversion before JSON serialization
            with open("dataframe.json", "w") as fp:
                json.dump(metadata, fp, default=str)
            
            st.success("Dataset uploaded and processed successfully!")
            
            # Ask user for visualization preference
            x_axis = st.selectbox("Select X-axis:", df.columns)
            y_axis = st.selectbox("_
