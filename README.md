
# Predictive Dashboard Generator (Streamlit)

Interactive **time-series forecasting** dashboard built with **Streamlit**, **scikit-learn**, **pandas**, and **Plotly**.

## Features
- Upload your CSV or use the included `sales_data.csv`
- Choose **Linear Regression** or **Random Forest**
- KPI cards: **MAE, RMSE, R²**
- Charts: **Actual vs Predicted**, **History + Forecast**
- Filters for categorical columns (e.g., product, region)
- Monthly aggregations & KPIs
- Download forecasts as CSV

## Run Locally
```bash
pip install -r requirements.txt
streamlit run predictive_dashboard.py
```

## Deploy on Streamlit Community Cloud
1. Push files to a **public GitHub** repo.
2. Go to https://streamlit.io/cloud → **New app**.
3. Set **Main file path** to `predictive_dashboard.py`.
4. Deploy and share your URL.
