import streamlit as st
import pandas as pd
import plotly.express as px
from langchain import LangChain
from langchain.streamlit import StreamlitCallbackHandler

# Streamlit page configuration
st.set_page_config(layout="wide")
st.title("📊 Marketing Data Visualizer with AI-Generated Code")

# Sidebar for user input
st.sidebar.header("1️⃣ Upload & Configure")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
api_key = st.sidebar.text_input("Enter LangChain API Key", type="password")

if uploaded_file and api_key:
    # Load CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Show first few rows of the data
    st.write("### Preview of the dataset", df.head())
    
    # User prompt for AI-generated code
    user_prompt = st.text_area("💬 Custom AI Prompt (Optional)", "Generate a bar chart based on categories in the data.")

    if st.button("🚀 Generate Visualization & Insights"):
        # Initialize LangChain
        lc = LangChain(api_key=api_key)
        
        # Define your data processing and querying logic
        def process_data(data):
            # Example: Generate a bar chart using Plotly
            fig = px.bar(data, x='Purchase_Date', y='Final_Price(Rs.)', title='Bar Chart of Final Price over Purchase Date')
            return fig
        
        # Generate AI prompt based on user input
        ai_prompt = f"""
        Based on the following dataset, generate Python code to create a chart:
        {df.head().to_string()}
        {user_prompt}
        """
        
        # Get response from LangChain
        response = lc.chat(ai_prompt)
        
        # Extract code from response using regex (assuming code block is wrapped in triple backticks)
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
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
