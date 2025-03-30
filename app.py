import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

# Load CSV data
df = pd.read_csv('https://raw.githubusercontent.com/your-username/your-repo/main/ecommerce_dataset_updated.csv')

# Convert date column to datetime format
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], dayfirst=True, errors='coerce')

# Convert numerical columns to appropriate types
df['Price (Rs.)'] = df['Price (Rs.)'].astype(float)
df['Discount (%)'] = df['Discount (%)'].astype(float)
df['Final_Price(Rs.)'] = df['Final_Price(Rs.)'].astype(float)

# Handling missing values
df = df.fillna(0)

# Streamlit app
st.title("Marketing Data Visualization")

# User input for chart type
chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart"])

# Generate chart based on user selection
if chart_type == "Bar Chart":
    query = "Generate a bar chart for 'Final_Price(Rs.)' over 'Purchase_Date'. Ensure proper styling."
elif chart_type == "Line Chart":
    query = "Generate a line chart for 'Final_Price(Rs.)' over 'Purchase_Date'. Ensure proper styling."

# Call the API
api_url = "https://your-api-gateway-url/your-endpoint"
response = requests.post(api_url, json={'query': query})

if response.status_code == 200:
    response_data = response.json()
    code = response_data['response']

    # Extract code from response using regex (assuming code block is wrapped in triple backticks)
    import re
    code_match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1)
        exec(extracted_code)
    else:
        st.write("No valid Python code found in response.")
else:
    st.write("Error in API call.")
