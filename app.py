import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.groq import Groq
from llama_index import SimpleDirectoryReader

# Configure Streamlit app
st.set_page_config(layout="wide")
st.title("📊 Marketing Data Insights with AI 🧠")

# Sidebar for user input
st.sidebar.header("1️⃣ Upload & Configure")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Store chatbot message history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about the data!"}]

# Function to load data and create index
@st.cache_resource(show_spinner=False)
def load_data():
    # Load the CSV data
    df = pd.read_csv(uploaded_file)
    # Display first few rows of the dataset for preview
    st.write("### Preview of the Dataset", df.head())
    
    # Convert CSV data to documents for LlamaIndex
    docs = [Document(text=row.to_json()) for index, row in df.iterrows()]
    
    # Create a vector store index using the uploaded data
    service_context = ServiceContext.from_defaults(llm=Groq(model="llama3-70b-8192", api_key=api_key))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index, df

if uploaded_file and api_key:
    # Load and index the data when CSV and API key are provided
    index, df = load_data()

    # Create the chat engine (ReAct Agent)
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    # Sidebar UI for selecting chart options
    x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
    chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram"])

    # Custom prompt for AI (optional)
    user_prompt = st.text_area("💬 Custom AI Prompt", "Analyze the trends in this data.")

    if st.button("🚀 Generate Visualization & Insights"):
        # Generate selected chart type using Plotly
        fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart")
        st.plotly_chart(fig)

        # Data description for the agent
        data_description = f"""
        The dataset contains the following columns: {', '.join(df.columns)}.
        The chart visualizes the relationship between {x_axis} (x-axis) and {y_axis} (y-axis).
        Here is the data used for the chart:
        {df[[x_axis, y_axis]].to_dict(orient='records')}
        """
        
        # Construct AI prompt
        ai_prompt = f"""
        Please analyze the data provided and the chart context. The chart shows the relationship between {x_axis} (x-axis) and {y_axis} (y-axis).
        Below is the data used for the chart:
        {data_description}
        {user_prompt}
        """
        
        # Generate insights using the chat engine
        response = chat_engine.chat(ai_prompt)
        insights_text = response.response if response.response else "No insights provided by AI."

        # Display the AI-generated insights
        st.subheader("💡 AI-Generated Insights")
        st.write(insights_text)

else:
    st.info("Upload a dataset and enter your API key to proceed.")
