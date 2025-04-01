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
    st.markdown(
        """
        <style>
            .reportview-container .main .block-container { max-width: 1400px; }
            h1 { color: #4f8bff; }
        </style>
        """,
        unsafe_allow_html=True,
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
        - Provide a short explanation of insights after the code.
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

def extract_code_and_insights(response):
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    insights_match = re.search(r"Insights:(.*?$)", response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else None
    insights = insights_match.group(1).strip() if insights_match else "No insights provided."
    return code, insights

def execute_visualization_code(code_block, df):
    try:
        exec_globals = {'df': df, 'px': px}
        exec(code_block, exec_globals)
        return exec_globals.get('fig')
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        return None

def handle_user_question(llm, df, question, context):
    prompt_template = PromptTemplate(
        """
        Context: {context}
        Dataset Columns: {columns}
        Sample Data:
        {sample_data}
        
        Question: {question}
        
        Provide a detailed answer with:
        - Specific data points
        - Statistical analysis
        - Visualization interpretation
        - Potential next steps
        """
    )
    # Change to using to_string() instead of to_markdown()
    qa_prompt = prompt_template.format(
        context=context,
        columns=list(df.columns),
        sample_data=df.head(3).to_string(),
        question=question
    )
    try:
        agent = ReActAgent.from_tools([], llm=llm, verbose=False)
        response = agent.chat(qa_prompt)
        return response.response
    except Exception as e:
        return f"Error processing question: {str(e)}"

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
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("Generate Visualization", use_container_width=True):
                with st.spinner("Generating visualization..."):
                    response = generate_visualization_code(llm, df, chart_type, x_col, y_col)
                    if response:
                        code, insights = extract_code_and_insights(response)
                        if code:
                            fig = execute_visualization_code(code, df)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                with st.expander("📈 Detailed Analysis", expanded=True):
                                    st.markdown(insights)
                            else:
                                st.error("Failed to render visualization.")
                        else:
                            st.error("No valid code extracted.")
                            st.code(response)  # Debugging output
        
        with col2:
            with st.expander("🔍 Data Preview"):
                st.dataframe(df.head(10), height=300)
                
            with st.expander("📝 Ask Question"):
                user_question = st.text_input("Enter your question:")
                if user_question:
                    context = f"Visualization Context: {chart_type}, X: {x_col}, Y: {y_col}"
                    answer = handle_user_question(llm, df, user_question, context)
                    st.markdown(f"**AI Analysis:**\n{answer}")

if __name__ == "__main__":
    main()
