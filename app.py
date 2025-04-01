import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
import pyperclip  # To allow copying to clipboard

# Configure Streamlit layout and page settings
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

# Validate data uploaded
def validate_data(df):
    if df.empty:
        st.error("Uploaded file is empty.")
        return False
    return True

# Generate Plotly visualization code with insights
def generate_visualization_code(llm, df, chart_type, x_col, y_col):
    prompt_template = PromptTemplate(
        """
        You are a data visualization expert. Your task is to generate Plotly Express Python code for a chart. 
        The DataFrame to use is 'df', and the following chart characteristics should be followed:
        - Chart type: {chart_type}.
        - X-axis: {x_col} (Type: {x_dtype}).
        - Y-axis: {y_col} (Type: {y_dtype}).
        - Only provide the Python code wrapped in triple backticks. Exclude all explanations and insights in your response.
        Example response format:
        ```
        python
        import plotly.express as px
        fig = px.{chart_type}(df, x="{x_col}", y="{y_col}")
        fig.show()
        ```
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
        agent = ReActAgent.from_tools([], llm=llm, verbose=True)
        response = agent.chat(system_prompt)
        return response.response
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def execute_visualization_code(code_block, df):
    try:

        # Define the execution context (with 'df' and 'px' as required)
        exec_globals = {'df': df, 'px': px}

        # Execute the cleaned code
        exec(clean_code, exec_globals)

        # Return the generated figure (assuming it's assigned to 'fig')
        return exec_globals.get('fig')
    
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        return None
# Handle user questions and generate responses based on the data
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
    qa_prompt = prompt_template.format(
        context=context,
        columns=list(df.columns),
        sample_data=df.head(3).to_markdown(),
        question=question
    )
    try:
        agent = ReActAgent.from_tools([], llm=llm, verbose=False)
        response = agent.chat(qa_prompt)
        return response.response
    except Exception as e:
        return f"Error processing question: {str(e)}"

# Main function that drives the Streamlit app
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
                        # Check if code is correctly formatted
                        if "```python" in response:
                            start = response.index("```python") + len("```python")
                            end = response.index("```", start)
                            code_block = response[start:end].strip()
                            
                            # Display the code in the app
                            st.code(code_block, language="python")
                            
                            # Execute the generated code
                            fig = execute_visualization_code(code_block, df)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Failed to render visualization.")
                        else:
                            st.error("Invalid code format.")
                            st.write(response)  # Debugging output for raw response
        
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
