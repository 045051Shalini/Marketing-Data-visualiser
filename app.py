import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.groq import Groq
import pandas as pd
import re
import numpy as np
import json
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
import plotly.express as px

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

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        metadata = generate_metadata(df)

        with open("dataframe.json", "w") as fp:
            json.dump(metadata, fp, default=str)

        st.sidebar.success("Dataset uploaded and processed successfully!")

        # Sidebar selections
        x_axis = st.sidebar.selectbox("Select X-axis:", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis:", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type:", ["bar", "line", "scatter", "histogram", "pie", "box"])

        # Enter button to trigger visualization and insights
        enter_button = st.sidebar.button("Generate Visualization")

        if enter_button:
            # Ensure x and y columns are valid
            if x_axis == y_axis:
                st.error("X-axis and Y-axis cannot be the same. Please select different columns.")
                return

            # Initialize Groq LLM
            llm = Groq(model="llama3-70b-8192", api_key="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")
            
            # Create a custom prompt with the user's dataset information and focus on marketing-specific insights
            new_prompt = PromptTemplate(f"""
                You are an AI specialized in marketing data analysis.
                Given the marketing dataset provided by the user, analyze the data and generate Python code for a valid visualization and provide insights specific to marketing. 
                Consider trends, customer segments, campaign performance, and marketing metrics like conversion rates, spending, demographics, etc.
                
                Your task:
                - Analyze marketing performance based on the data provided.
                - If the chart is too complex, suggest simplifications or focus on key marketing metrics.
                - Provide actionable marketing insights, including trends in customer acquisition, spending, campaign performance, or demographic breakdowns.
                
                Make sure to generate Python code for a valid chart based on the user's choices:
                Example Chart Code:
                ```python
                import plotly.express as px
                
                fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='Marketing Insights Visualization')
                fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
                fig.show()
                ```
                Example Insights:
                - Analyze customer acquisition trends.
                - Identify top-performing customer segments.
                - Suggest optimizations for marketing campaigns.
                - Products and categories trends.
                - Consumer preference.
                - Sales trends.                
                
            """)
            
            # Create AI Agent
            agent = ReActAgent.from_tools([], llm=llm, verbose=True)
            agent.update_prompts({'agent_worker:system_prompt': new_prompt})
            
            # Query Groq for visualization insights
            query = f"Generate a {chart_type} chart for '{y_axis}' over '{x_axis}'. Use the dataset provided to give marketing-specific insights."
            response = agent.chat(query)
            
            # Display raw AI response for debugging
            st.subheader("AI Response")
            st.write(response.response)
            
            # Extract Python code and insights from the AI response
            code_match = re.search(r"```python\n(.*?)```", response.response, re.DOTALL)
            insights_match = re.search(r"Insights:\n(.*?)$", response.response, re.DOTALL)
            
            # Check if the chart is too complex
            if is_complex_chart(df, x_axis, y_axis):
                st.warning("The chart may be too complex. Here are some suggestions:\n- Try focusing on fewer categories or grouping data.\n- Consider plotting a summary or aggregate statistic.")
            
            # Display chart before insights
            if code_match:
                extracted_code = code_match.group(1)
                extracted_code = extracted_code.replace("pd.read_csv('your_data.csv')", "df")
                st.code(extracted_code, language='python')
                try:
                    exec_locals = {"df": df}
                    exec(extracted_code, globals(), exec_locals)
                    if 'fig' in exec_locals:
                        st.plotly_chart(exec_locals['fig'])
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
            else:
                st.error("❌ No valid Python code found in AI response.")
            
            # Display insights if available (below the chart)
            if insights_match:
                insights_text = insights_match.group(1).strip()
                st.subheader("Marketing Insights")
                st.write(insights_text)
            else:
                st.warning("No insights provided by the AI.")
                
        else:
            st.info("Select options and click 'Generate Visualization'.")

if __name__ == "__main__":
    main()
