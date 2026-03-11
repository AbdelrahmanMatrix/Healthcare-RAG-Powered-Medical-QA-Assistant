import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="Healthcare RAG Dashboard", layout="wide")

st.title("🏥 Healthcare RAG-Powered Assistant Monitoring")

# Sidebar for configuration
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

# Main KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Queries", "1,254", "+12%")
col2.metric("Avg Latency", "1.2s", "-5%")
col3.metric("BLEU Score", "0.52", "+0.05")
col4.metric("F1 Score", "0.89", "+2%")

# Query Interface
st.header("🔍 Test the Assistant")
query = st.text_input("Enter a medical question:")
if st.button("Ask"):
    try:
        response = requests.post(f"{api_url}/ask", json={"query": query})
        if response.status_code == 200:
            data = response.json()
            st.subheader("Response:")
            st.write(data["response"])
            st.info(f"Category: {data['category']}")

            with st.expander("View Retrieved Context"):
                st.write(data["retrieved_contexts"])
        else:
            st.error(f"Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection error: {e}")

# Analytics Charts
st.header("📊 Performance Analytics")
chart_col1, chart_col2 = st.columns(2)

# Sample data for charts
category_data = pd.DataFrame({
    'Category': ['Symptoms', 'Diagnosis', 'Treatment', 'Medication', 'General'],
    'Volume': [450, 300, 200, 150, 154]
})

fig_cat = px.pie(category_data, values='Volume', names='Category', title='Query Distribution by Category')
chart_col1.plotly_chart(fig_cat)

latency_data = pd.DataFrame({
    'Time': pd.date_range(start='2025-01-01', periods=10, freq='D'),
    'Latency (ms)': [1200, 1100, 1300, 1150, 1050, 1250, 1100, 1000, 950, 1100]
})

fig_lat = px.line(latency_data, x='Time', y='Latency (ms)', title='API Latency Trend')
chart_col2.plotly_chart(fig_lat)

st.sidebar.markdown("---")
st.sidebar.info("Developed for eyouth x DEPI Graduation Project 2025")
