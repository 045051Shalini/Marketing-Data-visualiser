import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index.readers import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent

def preprocess_data(df):
    """Handle missing values and convert data types dynamically."""
    df = df.fillna(0)
    return df

def generate_figure(df, x_axis, y_axis, chart_type):
    """Generate chart based on user's choice."""
    if chart_type == "bar":
        return px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart of {y_axis} vs {x_axis}")
    elif chart_type == "line":
        return px.line(df, x=x_axis, y=y_axis, title=f"Line Chart of {y_axis} vs {x_axis}")
    elif chart_type == "scatter":
        return px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot of {y_axis} vs {x_axis}")
    elif chart_type == "histogram":
        return px.histogram(df, x=x_axis, title=f"Histogram of {y_axis}")
    elif chart_type == "pie":
        return px.pie(df, names=x_axis, title=f"Pie Chart of {x_axis}")
    elif chart_type == "box":
        return px.box(df, x=x_axis, y=y_axis, title=f"Box Plot of {y_axis} vs {x_axis}")
    else:
        return None

def get_ai_insights(llm, ai_prompt):
    """Generate insights using the LLM."""
    agent = ReActAgent.from_tools([], llm=llm, verbose=True)
    response = agent.chat(ai_prompt)
    return response.response.strip() if response else "No insights generated."

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Marketing Data Visualizer with AI Insights")

    # Sidebar for input configurations
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    # Choose LLM model and API key
    llm_choice = st.sidebar.selectbox("Select AI Model", ["Groq (Llama3-70B)"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")

    if uploaded_file is not None and api_key:
        try:
            # Use LlamaIndex to load CSV using SimpleDirectoryReader
            with open(uploaded_file, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the data from the directory where the CSV is stored
            reader = SimpleDirectoryReader('path_to_directory')
            index = LlamaIndex(reader)
            data = index.get_data()
            df = pd.DataFrame(data)  # Convert data into pandas DataFrame

            # Preprocess data
            df = preprocess_data(df)
            st.sidebar.success("✅ Dataset uploaded and processed successfully!")

            # Input x and y columns from the dataset
            x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
            y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)

            # Let user describe the visualization type
            chart_type = st.sidebar.text_input("Enter chart type", "scatter")  # Free-form text input
            
            # Button to generate insights and visualization
            generate_button = st.sidebar.button("🚀 Generate Visualization & Insights")

            if generate_button:
                st.subheader("📈 Visualization")
                try:
                    fig = generate_figure(df, x_axis, y_axis, chart_type)
                    if fig:
                        st.plotly_chart(fig)
                    else:
                        st.error(f"Error generating chart of type {chart_type}")
                        return
                except Exception as e:
                    st.error(f"Error generating chart: {e}")
                    return

                # Initialize LLM
                llm = Groq(model="llama3-70b-8192", api_key=api_key)

                # Construct the prompt to send to the AI model
                ai_prompt = f"""
                    You are an AI specialized in marketing analysis. 
                    You have the following dataset with these columns: {list(df.columns)}.
                    Analyze the chart type: {chart_type}.
                    Focus on trends, anomalies, patterns, and give marketing-specific insights related to {x_axis} and {y_axis}.
                    The following data points may be useful:
                    {df[[x_axis, y_axis]].describe().to_dict()}
                """
                
                # Get insights from AI
                ai_insights = get_ai_insights(llm, ai_prompt)
                
                st.subheader("💡 AI-Generated Insights")
                st.write(ai_insights)
                
                if st.button("📜 Show Python Code"):
                    st.code(f"""
                    fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='{chart_type.capitalize()} Visualization')
                    fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
                    fig.show()
                    """, language='python')

        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    else:
        st.info("Upload a dataset and enter an API key to proceed.")

if __name__ == "__main__":
    main()
