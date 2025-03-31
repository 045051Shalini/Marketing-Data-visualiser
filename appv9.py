import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import datetime
import re
import traceback
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq

def detect_column_types(df):
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
            insights.append(f"{y_axis} is right-skewed (positively skewed), indicating a longer tail on the right.")
        elif df[y_axis].skew() < -1:
            insights.append(f"{y_axis} is left-skewed (negatively skewed), indicating a longer tail on the left.")
        
        correlation = df.corr().get(x_axis, {}).get(y_axis, None)
        if correlation and abs(correlation) > 0.5:
            direction = "positive" if correlation > 0 else "negative"
            insights.append(f"There is a {direction} correlation ({correlation:.2f}) between {x_axis} and {y_axis}.")
        
    if pd.api.types.is_categorical_dtype(df[x_axis]) or pd.api.types.is_categorical_dtype(df[y_axis]):
        most_common = df[x_axis].mode()[0]
        insights.append(f"Most common category in {x_axis}: {most_common}")
    
    return insights

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
            
            fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f"{chart_type.capitalize()} Chart of {y_axis} vs {x_axis}")
            fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
            st.plotly_chart(fig)
            
            insights = generate_statistical_insights(df, x_axis, y_axis)
            
            llm = Groq(model="llama3-70b-8192", api_key="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")
            ai_prompt = PromptTemplate(f"""
                Given a dataset and the generated {chart_type} visualization for {y_axis} over {x_axis},
                provide data-driven insights beyond basic statistics. 
                Highlight trends, anomalies, and possible interpretations. 
            """)
            
            try:
                ai_response = llm.complete(ai_prompt)
                ai_text = ai_response.text.strip() if ai_response and hasattr(ai_response, "text") else "No AI insights available."
            except Exception as ai_error:
                ai_text = f"AI analysis failed: {str(ai_error)}"
                st.warning("⚠️ AI Insights could not be generated. Showing statistical insights only.")
            
            st.subheader("Graph Insights")
            st.write("\n".join(insights))
            st.write(ai_text)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(traceback.format_exc())
            if "AxiosError: Request failed with status code 403" in str(e):
                st.warning("⚠️ File upload might be restricted! Try renaming the file or checking permissions.")
                
if __name__ == "__main__":
    main()
