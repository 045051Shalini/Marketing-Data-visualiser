import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import json
from llama_index.core import VectorStoreIndex, Document, PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq

def detect_column_types(df):
    """Dynamically detect column types in the dataset."""
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or any(
            isinstance(x, str) and re.search(r'\d{4}-\d{2}-\d{2}', str(x)) for x in df[col].dropna()
        ):
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
    """Generate metadata about the dataset for AI processing."""
    metadata = {}
    for col in df.columns:
        try:
            if df[col].dtype in [np.int64, np.float64]:
                metadata[col] = {
                    'column_name': col,
                    'type': str(df[col].dtype),
                    'variable_information': [
                        float(df[col].max()),  # Convert to native float
                        float(df[col].min()),
                        float(df[col].mean())
                    ]
                }
            elif df[col].dtype == 'datetime64[ns]':
                metadata[col] = {
                    'column_name': col,
                    'type': 'datetime',
                    'variable_information': [str(df[col].max()), str(df[col].min())]
                }
            else:
                metadata[col] = {
                    'column_name': col,
                    'type': 'categorical',
                    'variable_information': list(df[col].value_counts().head(10).index)
                }
        except Exception:
            metadata[col] = {'column_name': col, 'type': 'unknown', 'variable_information': []}
    return metadata

def main():
    st.title("Dynamic Data Visualizer with Groq AI")

    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"], key="file_uploader_unique")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df, column_types = preprocess_data(df)
            metadata = generate_metadata(df)
            
            with open("dataframe.json", "w") as fp:
                json.dump(metadata, fp, indent=4)
            
            st.success("✅ Dataset uploaded and processed successfully!")
            
            if len(df.columns) < 2:
                st.error("Dataset must have at least two columns for visualization.")
                return

            x_axis = st.selectbox("Select X-axis:", df.columns)
            y_axis = st.selectbox("Select Y-axis:", df.columns)
            chart_type = st.selectbox("Select Chart Type:", ["bar", "line", "scatter"])
            
            if x_axis == y_axis:
                st.error("⚠️ X-axis and Y-axis cannot be the same. Please select different columns.")
                return
            
            llm = Groq(model="llama3-70b-8192", api_key="your_groq_api_key_here")
            
            prompt_text = f"""
                You are an AI that generates Python visualizations using Plotly.
                Given a dataset, generate a valid visualization code in Python.
                Ensure that you use 'df' as the dataframe and do not use file paths.
                
                import plotly.express as px
                
                # Visualization
                fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='Generated Visualization')
                fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
            """
            
            new_prompt = PromptTemplate(prompt_text)
            agent = ReActAgent.from_tools([], llm=llm, verbose=True)
            agent.update_prompts({'agent_worker:system_prompt': new_prompt})
            
            query = f"Generate a {chart_type} chart for '{y_axis}' over '{x_axis}'. Ensure proper styling and provide insights."
            response = agent.chat(query)
            response_text = response.get_response() if hasattr(response, "get_response") else str(response)
            
            code_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
            insights_match = re.search(r"Insights:\n(.*?)$", response_text, re.DOTALL)
            
            if code_match:
                extracted_code = code_match.group(1)
                extracted_code = extracted_code.replace("pd.read_csv('your_data.csv')", "df")
                st.code(extracted_code, language='python')
                
                try:
                    exec_locals = {"df": df}
                    exec(extracted_code, globals(), exec_locals)
                    if 'fig' in exec_locals:
                        st.plotly_chart(exec_locals['fig'], use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Execution Error: {str(e)}")
            else:
                st.error("❌ No valid Python code found in AI response.")
                st.info("👉 Possible reasons:\n- API key issue\n- LLM failed to generate proper code\n- Formatting issue")
            
            if insights_match:
                insights_text = insights_match.group(1).strip()
                st.subheader("Visualization Insights")
                st.write(insights_text)
            else:
                st.warning("No insights provided by the AI.")
        
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
