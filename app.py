import streamlit as st
import pandas as pd
import plotly.express as px
import traceback
from llama_index.llms.groq import Groq

# Apply custom styling with background texture & blue sidebar
st.markdown(
    """
    <style>
        /* Apply a dark overlay for readability */
        .main {
            background: rgba(0, 0, 0, 0.85);
            padding: 20px;
            border-radius: 10px;
        }
         /* Apply a light purple background for the main content area */
        .main {
            background-color: #d9d2e9 !important;  /* Light purple color for the main area */
            padding: 20px;
            border-radius: 10px;
        }
        /* Sidebar Styling - Pruple Background */
        section[data-testid="stSidebar"] {
            background-color: #b4a7d6 !important;  /* Purple */
            color: white !important;
            padding: 20px;
            border-radius: 0px 10px 10px 0px;
        }
        /* Sidebar text and headers */
        .sidebar-content {
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #66CCFF;
        }
        /* Buttons */
        .stButton>button {
            background-color: #004080;
            color: white;
            border-radius: 10px;
            padding: 10px 15px;
        }
        .stButton>button:hover {
            background-color: #00509E;
        }
        /* Links */
        .stMarkdown a {
            color: #66CCFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar for API Key and AI Query
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
st.sidebar.subheader("Ask AI About This Visualization")
user_query = st.sidebar.text_input("Enter your question for AI:")

# Function to detect column types
def detect_column_types(df):
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().any():
            df[col] = pd.to_datetime(df[col], errors='coerce')
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"
    return column_types

# Function to preprocess data
def preprocess_data(df):
    df = df.fillna(0)
    column_types = detect_column_types(df)
    return df, column_types

# Function to generate statistical insights
def generate_statistical_insights(df, selected_columns):
    insights = []
    for col in selected_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            insights.append(f"Mean of {col}: {df[col].mean():.2f}")
            insights.append(f"Median of {col}: {df[col].median():.2f}")
            insights.append(f"Standard Deviation: {df[col].std():.2f}")
        elif col in df.select_dtypes(include=['object', 'category']).columns:
            most_common = df[col].mode()[0]
            insights.append(f"Most common category in {col}: {most_common}")
    return "\n".join(insights)

# Function to query AI insights
def query_ai(prompt, df, selected_columns, chart_type):
    if not api_key:
        return "Please enter a valid API key in the sidebar."
    
    try:
        llm = Groq(model="llama3-70b-8192", api_key=api_key)
        df_summary = df[selected_columns].describe().to_dict()
        full_prompt = f"Dataset Summary: {df_summary}\nChart Type: {chart_type}\n{prompt}"
        response = llm.complete(full_prompt)
        return response.text.strip() if response and hasattr(response, "text") else "No AI insights available."
    except Exception as e:
        return f"AI error: {str(e)}"

# Main app
st.title("🔹 Advanced Data Visualizer with AI Insights 🔹")

# File uploader with sample data link
st.subheader("📂 Upload Your Dataset")
st.markdown(
    '[📥 Download Sample Data](https://github.com/045051Shalini/Marketing-Data-visualiser/raw/main/ecommerce_dataset_updated.csv)', 
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read file based on format
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df, column_types = preprocess_data(df)
        st.success("✅ Dataset uploaded and processed successfully!")

        # Chart selection and column selection
        st.subheader("📊 Select Columns & Chart Type")
        chart_type = st.selectbox("Select Chart Type", ["bar", "line", "scatter", "pie", "histogram", "box"])
        
        if chart_type == "pie":
            selected_columns = [st.selectbox("Select Column for Pie Chart", df.columns)]
        elif chart_type in ["histogram", "box"]:
            selected_columns = [st.selectbox("Select Numeric Column", df.columns)]
        else:
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            selected_columns = [x_axis, y_axis]

            if x_axis == y_axis:
                st.error("⚠️ X-axis and Y-axis cannot be the same. Please select different columns.")
                st.stop()

        # Generate chart
        st.subheader("📈 Generated Chart")
        if chart_type == "bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart")
        elif chart_type == "line":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart")
        elif chart_type == "pie":
            fig = px.pie(df, names=selected_columns[0], title="Pie Chart Distribution")
        elif chart_type == "histogram":
            fig = px.histogram(df, x=selected_columns[0], title="Histogram Distribution")
        elif chart_type == "box":
            fig = px.box(df, y=selected_columns[0], title="Box Plot Distribution")

        st.plotly_chart(fig)

        # Display insights
        st.subheader("📌 Graph Insights")
        statistical_insights = generate_statistical_insights(df, selected_columns)
        ai_prompt = f"Analyze this {chart_type} chart. Identify trends, anomalies, and patterns."
        ai_insights = query_ai(ai_prompt, df, selected_columns, chart_type)

        st.write(statistical_insights)
        st.write(ai_insights)

        # AI agent query from sidebar
        if user_query:
            query_prompt = f"Based on the provided dataset and {chart_type} chart, answer this question: {user_query}"
            ai_response = query_ai(query_prompt, df, selected_columns, chart_type)
            st.sidebar.subheader("💡 AI Response")
            st.sidebar.write(ai_response)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error(traceback.format_exc())
