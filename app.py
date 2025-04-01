import streamlit as st
import pandas as pd
import plotly.express as px
import re
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate

def configure_streamlit():
    st.set_page_config(
        page_title="AI Data Visualizer",
        page_icon="📊",
        layout="wide"
    )

def validate_data(df):
    if df.empty:
        st.error("Uploaded file is empty.")
        return False
    return True

def generate_visualization_code(llm, df, chart_type, x_col, y_col):
    prompt_template = PromptTemplate(
        """
        You are a data visualization expert. Generate Plotly Express Python code following these conditions:
        - Use the DataFrame 'df'.
        - Chart type: {chart_type}.
        - X-axis: {x_col} (Type: {x_dtype}).
        - Y-axis: {y_col} (Type: {y_dtype}).
        - Ensure correct data handling for categorical and numerical data.
        - Use modern color schemes and proper labels.
        - Return only the Python code wrapped in triple backticks.
        """
    )
    
    system_prompt = prompt_template.format(
        chart_type=chart_type,
        x_col=x_col,
        x_dtype=df[x_col].dtype,
        y_col=y_col,
        y_dtype=df[y_col].dtype
    )
    
    try:
        agent = ReActAgent.from_tools([], llm=llm, verbose=False)
        response = agent.chat(system_prompt)
        return response.response
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def execute_visualization_code(code_block, df):
    try:
        exec_globals = {'df': df, 'px': px}
        exec(code_block, exec_globals)
        return exec_globals.get('fig')
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        st.code(str(e), language='bash')
        return None

def main():
    configure_streamlit()
    st.title("📊 AI-Powered Data Visualizer")
    
    with st.sidebar:
        st.header("Upload & Configure")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        api_key = st.text_input("Groq API Key", type="password")
        model_name = st.selectbox("LLM Model", ["mixtral-8x7b-32768", "llama3-70b-8192"])
    
    if uploaded_file and api_key:
        df = pd.read_csv(uploaded_file)
        if not validate_data(df):
            return

        llm = Groq(model=model_name, api_key=api_key)
        
        with st.sidebar:
            st.header("Visualization Settings")
            x_col = st.selectbox("X-Axis", df.columns)
            y_col = st.selectbox("Y-Axis", df.columns)
            chart_type = st.selectbox("Chart Type", ["bar", "line", "scatter", "histogram", "box", "violin"])
        
        if st.button("Generate Visualization"):
            with st.spinner("Generating visualization..."):
                response = generate_visualization_code(llm, df, chart_type, x_col, y_col)
                if response:
                    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                        fig = execute_visualization_code(code, df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to render visualization.")
                    else:
                        st.error("No valid code extracted.")
                        st.code(response)  # Debugging output

if __name__ == "__main__":
    main()
