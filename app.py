import streamlit as st
import pandas as pd
import plotly.express as px
from llama_index.llms.openai import OpenAI
from llama_index.indices.service_context import ServiceContextSimple

# Initialize LlamaIndex
def initialize_llama_index(api_key):
    try:
        llm = OpenAI(model="gpt-4", api_key=api_key)
        index = LlamaIndex(llm=llm)
        return index
    except Exception as e:
        st.error(f"Error initializing LlamaIndex: {e}")
        return None

# Generate insights using LlamaIndex
def generate_insights(index, df, x_axis, y_axis, user_prompt):
    try:
        # Create a ServiceContextSimple to store the data
        context = ServiceContextSimple(df)

        # Define the prompt for the AI agent
        prompt = f"""
            Analyze the data in the '{x_axis}' and '{y_axis}' columns.
            Generate actionable marketing insights based on the chart.
            {user_prompt}
        """

        # Use the LlamaIndex to generate insights
        insights = index.query(prompt, context=context)

        return insights
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Marketing Data Visualizer with AI Insights")

    # Sidebar for input selections
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    api_key = st.sidebar.text_input("Enter API Key", type="password")

    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)

        st.sidebar.success("✅ Dataset uploaded successfully!")
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram", "pie", "box"])
        user_prompt = st.sidebar.text_area("💬 Custom AI Prompt (Optional)", "Analyze the provided data and chart to generate insights.")
        generate_button = st.sidebar.button("🚀 Generate Visualization & Insights")

        if generate_button:
            st.subheader("📈 Visualization")
            try:
                fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f'{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                return

            # Initialize LlamaIndex
            index = initialize_llama_index(api_key)
            if index is None:
                return

            # Generate insights using LlamaIndex
            insights = generate_insights(index, df, x_axis, y_axis, user_prompt)
            if insights is None:
                return

            st.subheader("💡 AI-Generated Insights")
            st.write(insights)

            # Show Python Code Button
            if st.button("📜 Show Python Code"):
                python_code = f"""
                import plotly.express as px
                fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
                fig.show()
                """
                st.code(python_code, language='python')
    else:
        st.info("Upload a dataset and enter an API key to proceed.")

if __name__ == "__main__":
    main()
