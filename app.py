import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re

#ME_CONFIG = {
    "primaryColor": "#4f8bff",
    "backgroundColor": "#0e1117",
    "textColor": "#f0f2f6"
}

def configure_streamlit():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="AI Data Visualizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{max-width: 1400px;}}
            h1 {{color: {THEME_CONFIG['primaryColor']};}}
            .stSelectbox, .stTextInput {{border: 1px solid {THEME_CONFIG['primaryColor']};}}
        </style>
    """, unsafe_allow_html=True)

def validate_data(df):
    """Perform data validation checks"""
    if df.empty:
        st.error("Uploaded file is empty")
        return False
    return True

def generate_visualization_code(llm, df, chart_type, x_col, y_col):
    """Generate visualization code using Groq LLM"""
    system_prompt = f"""
    You are an expert data visualization assistant. Generate valid Plotly Express code with these requirements:
    1. Use DataFrame 'df' provided in context
    2. Chart type: {chart_type}
    3. X-axis: {x_col} ({df[x_col].dtype})
    4. Y-axis: {y_col} ({df[y_col].dtype})
    5. Include proper axis labels and titles
    6. Use modern color schemes
    7. Return ONLY the Python code wrapped in ```
    8. After code, provide insights specific to this data relationship:
       - Statistical correlation between columns
       - Notable patterns/trends
       - Data distribution characteristics
       - Potential anomalies
    """
    
    try:
        agent = ReActAgent.from_tools([], llm=llm, verbose=False)
        response = agent.chat(system_prompt)
        st.write("LLM Response:", response.response)  # Debugging output
        return response.response
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def clean_code(code_block):
    """Clean the code block by removing triple backticks and extra spaces"""
    return code_block.strip().replace("```python", "").replace("```", "").strip()

def execute_visualization_code(code_block):
    """Send the code to the Java backend for execution"""
    try:
        response = requests.post("http://localhost:8080/api/execute", json={"code": code_block})
        return response.text
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        return None

def handle_user_question(llm, df, question, context):
    """Handle user questions with full context"""
    qa_prompt = f"""
    Context: {context}
    Dataset Columns: {list(df.columns)}
    Sample Data:
    {df.head(3).to_markdown()}
    
    Question: {question}
    
    Provide a detailed answer with:
    1. Specific data points from the dataset
    2. Statistical analysis
    3. Visualization interpretation
    4. Potential next steps
    """
    
    try:
        agent = ReActAgent.from_tools([], llm=llm, verbose=False)
        response = agent.chat(qa_prompt)
        return response.response
    except Exception as e:
        return f"Error processing question: {str(e)}"

def main():
    configure_streamlit()
    st.title("📊 Smart Data Visualizer with AI Analytics")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Data Configuration")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        api_key = st.text_input("Groq API Key", type="password")
        model_name = st.selectbox("AI Model", ["mixtral-8x7b-32768", "llama3-70b-8192"])
        
    if uploaded_file and api_key:
        df = pd.read_csv(uploaded_file)
        if not validate_data(df):
            return
            
        # Initialize Groq client correctly
        llm = Groq(model=model_name, api_key=api_key)
        
        with st.sidebar:
            st.header("Visualization Settings")
            cols = st.columns(2)  # Creates two columns
            x_col = cols[0].selectbox("X-Axis", df.columns)
            y_col = cols[1].selectbox("Y-Axis", df.columns)
                
            chart_type = st.selectbox("Chart Type", [
                "bar", "line", "scatter", 
                "histogram", "box", "violin"
            ])
            
            with st.expander("Advanced Options"):
                color_scale = st.selectbox("Color Scale", px.colors.named_colorscales())
                template = st.selectbox("Theme", ["plotly", "plotly_white", "plotly_dark"])
        
        # Main content area
        col1, col2 = st.columns([3,1])
        
        with col1:
            if st.button("Generate Visualization", use_container_width=True):
                with st.spinner("Analyzing data and creating visualization..."):
                    response = generate_visualization_code(llm, df, chart_type, x_col, y_col)
                    
                    if response:
                        # Extract code and insights
                        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
                        insights_match = re.search(r"Insights:(.*?$)", response, re.DOTALL)
                        
                        if code_match:
                            code = clean_code(code_match.group(1))
                            st.subheader("Generated Python Code")
                            st.code(code, language="python")
                            
                            execution_result = execute_visualization_code(code)
                            st.text(execution_result)
                            
                            # Display insights
                            if insights_match:
                                insights = insights_match.group(1).strip()
                                st.subheader("Insights")
                                st.markdown(insights)
                            else:
                                st.warning("No insights generated")
                        else:
                            st.error("No valid code generated")
                            st.code(response)  # Debug output

        with col2:
            with st.expander("🔍 Data Preview"):
                st.dataframe(df.head(10), height=300)
                
            with st.expander("📝 Ask Question"):
                user_question = st.text_input("Enter your question:")
                if user_question:
                    context = f"""
                        Visualization Context:
                        - Type: {chart_type}
                        - X: {x_col} ({df[x_col].dtype})
                        - Y: {y_col} ({df[y_col].dtype})
                        - Sample X Values: {df[x_col].sample(3).tolist()}
                        - Sample Y Values: {df[y_col].sample(3).tolist()}
                    """
                    answer = handle_user_question(llm, df, user_question, context)
                    st.markdown(f"**AI Analysis:**\n{answer}")

if __name__ == "__main__":
    main()
