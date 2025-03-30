import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

# Streamlit app
st.title("Marketing Data Visualization")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert date column to datetime format
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], dayfirst=True, errors='coerce')

    # Convert numerical columns to appropriate types
    df['Price (Rs.)'] = df['Price (Rs.)'].astype(float)
    df['Discount (%)'] = df['Discount (%)'].astype(float)
    df['Final_Price(Rs.)'] = df['Final_Price(Rs.)'].astype(float)

    # Handling missing values
    df = df.fillna(0)

    # Step 2: Select LLM and provide API key
    st.sidebar.title("LLM Configuration")
    llm_option = st.sidebar.selectbox("Select LLM", ["Groq", "OpenAI"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")

    if api_key:
        # Step 3: Generate visualizations and insights
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart"])

        if chart_type == "Bar Chart":
            query = "Generate a bar chart for 'Final_Price(Rs.)' over 'Purchase_Date'. Ensure proper styling."
        elif chart_type == "Line Chart":
            query = "Generate a line chart for 'Final_Price(Rs.)' over 'Purchase_Date'. Ensure proper styling."

        # Call the API
        api_url = "https://your-api-gateway-url/your-endpoint"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(api_url, json={'query': query}, headers=headers)

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
    else:
        st.write("Please enter your API key.")
else:
    st.write("Please upload a CSV file.")
