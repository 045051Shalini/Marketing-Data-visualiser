import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import traceback
from llama_index.llms.groq import Groq

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

def preprocess_data(df):
    df = df.fillna(0)
    column_types = detect_column_types(df)
    return df, column_types

def generate_statistical_insights(df, x_axis, y_axis):
    insights = []
    if pd.api.types.is_numeric_dtype(df[y_axis]):
        insights.append(f"Mean of {y_axis}: {df[y_axis].mean():.2f}")
        insights.append(f"Median of {y_axis}: {df[y_axis].median():.2f}")
        insights.append(f"Standard Deviation: {df[y_axis].std():.2f}")
        if df[y_axis].skew() > 1:
            insights.append(f"{y_axis} is right-skewed, indicating a longer tail on the right.")
        elif df[y_axis].skew() < -1:
            insights.append(f"{y_axis} is left-skewed, indicating a longer tail on the left.")
    if x_axis in df.select_dtypes(include=['object', 'category']).columns:
        most_common = df[x_axis].mode()[0]
        insights.append(f"Most common category in {x_axis}: {most_common}")
    return "\n".join(insights)

def query_ai(prompt, df, x_axis, y_axis, chart_type):
    try:
        llm = Groq(model="llama3-70b-8192", api_key="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")
        df_summary = df[[x_axis, y_axis]].describe().to_dict()
        full_prompt = f"Dataset Summary: {df_summary}\nChart Type: {chart_type}\n{prompt}"
        response = llm.complete(full_prompt)
        return response.text.strip() if response and hasattr(response, "text") else "No AI insights available."
    except Exception as e:
        return f"AI error: {str(e)}"

def main():
    st.title("Enhanced Data Visualizer with AI Insights")
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df, column_types = preprocess_data(df)
            
            st.success("Dataset uploaded and processed successfully!")
            x_axis = st.selectbox("Select X-axis:", df.columns)
            y_axis = st.selectbox("Select Y-axis:", df.columns)
            chart_type = st.selectbox("Select Chart Type:", ["bar", "line", "scatter"])
            
            if x_axis == y_axis:
                st.error("X-axis and Y-axis cannot be the same. Please select different columns.")
                return
            
            # Generate graph
            if chart_type == "bar":
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart of {y_axis} vs {x_axis}")
            elif chart_type == "line":
                fig = px.line(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart of {y_axis} vs {x_axis}")
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart of {y_axis} vs {x_axis}")
            
            fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
            st.plotly_chart(fig)
            
            st.subheader("Graph Insights")
            statistical_insights = generate_statistical_insights(df, x_axis, y_axis)
            ai_prompt = f"Analyze this {chart_type} chart for {y_axis} over {x_axis}. Identify trends, anomalies, and patterns."
            ai_insights = query_ai(ai_prompt, df, x_axis, y_axis, chart_type)
            
            st.write(statistical_insights)
            st.write(ai_insights)
            
            user_query = st.text_input("Ask AI about this visualization:")
            if user_query:
                query_prompt = f"Based on the provided dataset and {chart_type} chart, answer this question: {user_query}"
                ai_response = query_ai(query_prompt, df, x_axis, y_axis, chart_type)
                st.subheader("AI Response")
                st.write(ai_response)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(traceback.format_exc())
            
if __name__ == "__main__":
    main()