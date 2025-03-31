import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.groq import Groq
import pandas as pd
import re
import numpy as np
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
            return [df[column].max(), df[column].min(), df[column].mean()]
        elif df[column].dtype == 'datetime64[ns]':
            return [str(df[column].max()), str(df[column].min())]
        else:
            return list(df[column].value_counts().head(10).index)
    
    metadata = {col: {'column_name': col, 'type': str(df[col].dtype), 'variable_information': return_vals(col)} for col in df.columns}
    return metadata

def main():
    st.title("Dynamic Data Visualizer with Groq AI")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df, column_types = preprocess_data(df)
            metadata = generate_metadata(df)
            with open("dataframe.json", "w") as fp:
                json.dump(metadata, fp)
            
            st.success("Dataset uploaded and processed successfully!")
            
            # Ask user for visualization preference
            x_axis = st.selectbox("Select X-axis:", df.columns)
            y_axis = st.selectbox("Select Y-axis:", df.columns)
            chart_type = st.selectbox("Select Chart Type:", ["bar", "line", "scatter"])
            
            # Ensure x and y columns are valid
            if x_axis == y_axis:
                st.error("X-axis and Y-axis cannot be the same. Please select different columns.")
                return
            
            # Initialize Groq LLM
            llm = Groq(model="llama3-70b-8192", api_key="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")
            
            # Define AI Prompt
            new_prompt = PromptTemplate(f"""
                You are an AI that generates Python visualizations using Plotly.
                Given a dataset, generate a valid visualization code in Python.
                Ensure that you use 'df' as the dataframe and do not use file paths.
                
                ```python
                import plotly.express as px
                
                # Visualization
                fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='Generated Visualization')
                fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
                fig.show()
                ```
                
                Also provide insights about the generated chart in plain text format.
            """)
            
            # Create AI Agent
            agent = ReActAgent.from_tools([], llm=llm, verbose=True)
            agent.update_prompts({'agent_worker:system_prompt': new_prompt})
            
            # Query Groq for visualization insights
            query = f"Generate a {chart_type} chart for '{y_axis}' over '{x_axis}'. Ensure proper styling and provide insights."
            response = agent.chat(query)
            
            # Debug: Show raw AI response in Streamlit
            st.subheader("AI Response")
            st.write(response.response)
            
            # Extract Python code from AI response
            code_match = re.search(r"```python\n(.*?)```", response.response, re.DOTALL)
            insights_match = re.search(r"Insights:\n(.*?)$", response.response, re.DOTALL)
            
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
                st.info("👉 Possible reasons:\n- API key issue\n- LLM failed to generate proper code\n- Formatting issue")
            
            # Display insights in Streamlit window
            if insights_match:
                insights_text = insights_match.group(1).strip()
                st.subheader("Visualization Insights")
                st.write(insights_text)
            else:
                st.warning("No insights provided by the AI.")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(traceback.format_exc())
            if "AxiosError: Request failed with status code 403" in str(e):
                st.warning("⚠️ File upload is restricted! Try renaming the file or checking permissions.")
                st.info("👉 Possible Fixes:\n- Rename the file\n- Check file permissions\n- Run Streamlit with admin rights\n- Ensure Groq API key is correct")
                
if __name__ == "__main__":
    main()
