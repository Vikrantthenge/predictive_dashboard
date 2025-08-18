# predictive_dashboard.py
import os
from datetime import timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Predictive Dashboard Generator", page_icon="📈", layout="wide")

# --- Header Image and Title ---
banner_path = "asset/predictive_dashboard_banner.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_column_width=True)
st.title("📈 Predictive Dashboard Generator")

# --- Sidebar Inputs ---
st.sidebar.header("User Input")
selected_feature = st.sidebar.selectbox("Select Feature (demo charts)", ["Feature A", "Feature B", "Feature C"])
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=(uploaded is None))
encoding_choice = st.sidebar.selectbox("File encoding", ["utf-8", "ISO-8859-1", "latin1"])
target_col = st.sidebar.text_input("Target column (numeric)", "sales")
date_col = st.sidebar.text_input("Date column", "date")
category_cols = st.sidebar.text_input("Category columns (optional)", "product,region")
model_name = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest"])
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30, step=1)
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
download_toggle = st.sidebar.checkbox("Enable predictions download", value=True)

# --- Load Sample or Uploaded Data ---
@st.cache_data
def load_sample():
    for fname in ("sales_data_small_iso.csv", "sales_data_small.csv", "sales_data.csv"):
        if os.path.exists(fname):
            return pd.read_csv(fname), fname
    raise FileNotFoundError("Sample CSV not found.")

def try_parse_dates(df, date_col):
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    if df[date_col].isna().mean() > 0.4:
        raise ValueError("Too many unparseable dates.")
    return df

# Load data
if use_sample:
    try:
        df, fname = load_sample()
        st.success(f"Sample loaded: {fname}")
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, encoding=encoding_choice)
            st.success("Uploaded file loaded.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
    else:
        st.stop()

# --- Parse and validate ---
if date_col not in df.columns:
    candidates = [c for c in df.columns if "date" in c.lower()]
    if candidates:
        date_col = candidates[0]
        st.warning(f"Auto-detected date column: {date_col}")
    else:
        st.error("Date column not found.")
        st.stop()

try:
    df = try_parse_dates(df, date_col)
except Exception as e:
    st.error(f"Date parsing error: {e}")
    st.stop()

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found.")
    st.stop()

df = df.dropna(subset=[date_col, target_col]).sort_values(by=date_col)

# --- Demo chart (optional) ---
demo_df = pd.DataFrame(np.random.randn(100, 3), columns=["Feature A", "Feature B", "Feature C"])
st.subheader("Dashboard Overview")
st.write("This dashboard uses predictive analytics to visualize trends and forecast outcomes.")
st.line_chart(demo_df[selected_feature])

# --- Feature Engineering ---
def make_features(df, date_col, target_col):
    df["ds"] = pd.to_datetime(df[date_col])
    df["y"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["y"])
    df["t"] = (df["ds"] - df["ds"].min()).dt.days
    df["dow"] = df["ds"].dt.dayofweek
    df["dom"] = df["ds"].dt.day
    df["month"] = df["ds"].dt.month
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["roll_7"] = df["y"].rolling(7, min_periods=1).mean()
    df["roll_14"] = df["y"].rolling(14, min_periods=1).mean()
    df = df.dropna()
    features = ["t", "dow", "dom", "month", "lag_1", "lag_7", "lag_14", "roll_7", "roll_14"]
    return df, features

Xy, features = make_features(df, date_col, target_col)
if len(Xy) < 60:
    st.warning("Less than 60 rows after feature engineering. Forecast may be unreliable.")

# --- Train/Test Split ---
X = Xy[features]
y = Xy["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, shuffle=False)

# --- Model Selection & Training ---
if model_name == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
pred = model.predict(X_test)
y_test = np.ravel(y_test)
pred = np.ravel(pred)

# --- Evaluation Metrics ---
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

st.subheader("📊 Model Evaluation Metrics")
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R²", f"{r2:.2f}")

# --- Results DataFrame ---
results_df = pd.DataFrame({
    "date": Xy.loc[X_test.index, "ds"],
    "actual": y_test,
    "predicted": pred
})
st.subheader("Actual vs Predicted")
st.dataframe(results_df)

# --- Download Results ---
if download_toggle:
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_results.csv",
        mime="text/csv"
    )

# --- Plot Predictions ---
fig = px.line(results_df, x="date", y=["actual", "predicted"], title="Actual vs Predicted")
st.plotly_chart(fig, use_container_width=True)

# --- Forecast Future ---
last_date = Xy["ds"].max()
future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)

fcast_df = pd.DataFrame({date_col: future_dates})
merged_df = pd.concat([df[[date_col, target_col]], fcast_df], ignore_index=True)
merged_df[date_col] = pd.to_datetime(merged_df[date_col])
F, _ = make_features(merged_df, date_col, target_col)
F_future = F[F["ds"].isin(future_dates)]

if len(F_future) > 0:
    yhat = model.predict(F_future[features])
    forecast_df = pd.DataFrame({"date": F_future["ds"], "forecast": yhat})

    hist = Xy[["ds", "y"]].rename(columns={"ds": "date", "y": "value"})
    hist["series"] = "history"
    fcast_plot = forecast_df.rename(columns={"forecast": "value"})
    fcast_plot["series"] = "forecast"
    full_plot = pd.concat([hist, fcast_plot])

    fig2 = px.line(full_plot, x="date", y="value", color="series", title="History + Forecast")
    st.plotly_chart(fig2, use_container_width=True)

    if download_toggle:
        st.download_button("⬇️ Download Forecast CSV", forecast_df.to_csv(index=False).encode("utf-8"), "forecast.csv", "text/csv")
else:
    st.warning("Not enough history to generate forecast.")

# --- Aggregations ---
with st.expander("📊 Aggregations & KPIs"):
    cat_cols = [c.strip() for c in category_cols.split(",") if c.strip() in df.columns]
    agg_dim = st.selectbox("Aggregate by", ["(None)"] + cat_cols)
    agg_metric = st.selectbox("Metric", ["mean", "sum", "median"], index=1)

    if agg_dim != "(None)":
        agg_df = df.groupby([agg_dim, pd.Grouper(key=date_col, freq="M")])[target_col].agg(agg_metric).reset_index()
        fig3 = px.line(agg_df, x=date_col, y=target_col, color=agg_dim, title=f"Monthly {agg_metric} by {agg_dim}")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        monthly = df.groupby(pd.Grouper(key=date_col, freq="M"))[target_col].agg(agg_metric).reset_index()
        fig4 = px.bar(monthly, x=date_col, y=target_col, title=f"Monthly {agg_metric.title()}")
        st.plotly_chart(fig4, use_container_width=True)

st.caption("Built by Vikrant • Streamlit + scikit-learn + Plotly")
