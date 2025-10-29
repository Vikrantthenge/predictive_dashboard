import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import io
import warnings
warnings.filterwarnings("ignore")

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Predictive Sales Optimization",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Predictive Sales Optimization Dashboard")
st.write("An interactive tool to forecast future sales and optimize marketing or promotional strategies using time series analysis.")

# --- File Upload Section ---
st.sidebar.header("📂 Upload Your Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# --- Load dataset ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded successfully!")
else:
    df = pd.read_csv("enhanced_sales_data.csv")  # Preloaded default sample
    st.sidebar.info("🧾 Using sample dataset (enhanced_sales_data.csv)")

# --- Ensure Date column is datetime ---
date_col_candidates = [c for c in df.columns if 'date' in c.lower()]
if date_col_candidates:
    date_col = date_col_candidates[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)
    df = df.set_index(date_col)
else:
    st.error("❌ No date-like column found in dataset. Please include a column named 'Date'.")
    st.stop()

st.sidebar.success(f"📅 Using '{date_col}' as Date column")

# --- Data Overview ---
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# --- Target column (Sales) ---
if "Sales" not in df.columns:
    st.error("❌ No 'Sales' column found in dataset.")
    st.stop()

target_col = "Sales"

# --- Time series by date ---
daily_df = df.resample('D').sum(numeric_only=True).fillna(0)

st.subheader("📆 Historical Sales Trend")
fig_sales = px.line(daily_df, y=target_col, title="Daily Sales Over Time")
st.plotly_chart(fig_sales, use_container_width=True)

# --- Forecast Section ---
st.subheader("🔮 Sales Forecasting")
horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 365, 30)

try:
    model = SARIMAX(daily_df[target_col], order=(1,1,1), seasonal_order=(1,1,1,7))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=horizon)
    forecast_df = forecast.summary_frame()

    fig_forecast = px.line()
    fig_forecast.add_scatter(x=daily_df.index, y=daily_df[target_col], mode='lines', name='Historical')
    fig_forecast.add_scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast')
    fig_forecast.add_scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], mode='lines', name='Upper CI', line=dict(width=0.5, dash='dot'))
    fig_forecast.add_scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], mode='lines', name='Lower CI', line=dict(width=0.5, dash='dot'))
    fig_forecast.update_layout(title="Sales Forecast", xaxis_title="Date", yaxis_title="Sales")
    st.plotly_chart(fig_forecast, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Forecasting error: {e}")

# --- Regional / Product Analysis ---
if "Region" in df.columns and "Product" in df.columns:
    st.subheader("🌍 Regional & Product Analysis")

    col1, col2 = st.columns(2)

    with col1:
        region_sales = df.groupby("Region")[target_col].sum().reset_index()
        fig_region = px.bar(region_sales, x="Region", y="Sales", title="Total Sales by Region", color="Region")
        st.plotly_chart(fig_region, use_container_width=True)

    with col2:
        prod_sales = df.groupby("Product")[target_col].sum().reset_index()
        fig_prod = px.bar(prod_sales, x="Product", y="Sales", title="Total Sales by Product", color="Product")
        st.plotly_chart(fig_prod, use_container_width=True)

# --- Optimization Section ---
st.subheader("⚙️ Marketing Spend Optimization")

st.markdown("""
Estimate how sales performance responds to changes in **Ad Spend** and **Discounts**.  
Use sliders to simulate different strategies.
""")

base_sales = daily_df[target_col].mean()
discount = st.slider("Average Discount (%)", 0, 50, 10)
ad_spend = st.slider("Ad Spend Multiplier (x base)", 0.5, 3.0, 1.0)

# Simplified predictive logic
expected_sales = base_sales * (1 + (ad_spend - 1) * 0.3) * (1 - discount / 100 * 0.5)

st.metric(label="💰 Expected Daily Sales", value=f"{expected_sales:,.0f}")
st.caption("Simulated impact based on average historical performance.")

# --- Download Forecast Data ---
csv_buffer = io.StringIO()
forecast_df.to_csv(csv_buffer)
st.download_button("📥 Download Forecast CSV", data=csv_buffer.getvalue(), file_name="sales_forecast.csv")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Python | AWS EC2 | Time Series Forecasting")
