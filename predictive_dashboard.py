from PIL import Image

# Load and display logo
logo = Image.open("predictive_dashboard_banner.png")
st.image(logo, width=150)  # Adjust width as needed

# Optional: Add a centered title below the logo
st.markdown("<h1 style='text-align: center;'>📈 Predictive Dashboard</h1>", unsafe_allow_html=True)

import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload your CSV or use the sample. Required: a **date** column and a **numeric target** (e.g., sales). Optional: categorical columns (e.g., product, region).")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=(uploaded is None))
encoding_choice = st.sidebar.selectbox("File encoding", ["utf-8", "ISO-8859-1", "latin1"], index=0)
target_col = st.sidebar.text_input("Target column (numeric)", "sales")
date_col = st.sidebar.text_input("Date column", "date")
category_cols = st.sidebar.text_input("Category columns (comma-separated, optional)", "product,region")
model_name = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest"])
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30, step=1)
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
download_toggle = st.sidebar.checkbox("Enable predictions download", value=True)

# --- Load Sample or Uploaded Data ---
@st.cache_data
def load_sample():
    for fname in ("sales_data_small_iso.csv", "sales_data_small.csv", "sales_data.csv"):
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname)
                return df, fname
            except Exception:
                continue
    raise FileNotFoundError("No sample CSV found.")

def try_parse_dates(df, date_col):
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found.")
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        return df
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
        if parsed.notna().sum() / len(parsed) > 0.90:
            df[date_col] = parsed.dt.normalize()
            return df
    except Exception:
        pass
    def infer_date(x):
        for dfmt in ("%Y-%m-%d","%d-%m-%Y","%m-%d-%Y","%d/%m/%Y","%m/%d/%Y"):
            try: return pd.to_datetime(x, format=dfmt)
            except: continue
        try: return pd.to_datetime(x, errors="raise")
        except: return pd.NaT
    parsed = df[date_col].apply(infer_date)
    if parsed.notna().sum() / len(parsed) > 0.6:
        df[date_col] = pd.to_datetime(parsed).dt.normalize()
        return df
    raise ValueError("Date parsing failed.")

if use_sample:
    try:
        df, loaded_fname = load_sample()
        st.info(f"Loaded sample file: {loaded_fname}")
    except Exception as e:
        st.error(str(e))
        st.stop()
elif uploaded is not None:
    try:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding=encoding_choice)
        st.info("Uploaded file loaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
else:
    st.stop()

if date_col not in df.columns:
    candidates = [c for c in df.columns if "date" in c.lower()]
    if candidates:
        date_col = candidates[0]
        st.warning(f"Auto-detected date column: {date_col}")
    else:
        st.error("No valid date column found.")
        st.stop()

try:
    df = try_parse_dates(df, date_col)
except Exception as e:
    st.error(f"Date parsing failed: {e}")
    st.stop()

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found.")
    st.stop()

df = df.dropna(subset=[date_col, target_col]).sort_values(by=date_col)

# --- Demo Chart ---
demo_data = np.random.randn(100, 3)
demo_df = pd.DataFrame(demo_data, columns=["Feature A", "Feature B", "Feature C"])
st.subheader("Dashboard Overview")
st.line_chart(demo_df[selected_feature])

# --- Feature Engineering ---
def make_features(frame, date_col, target_col):
    Xy = frame[[date_col, target_col]].copy()
    Xy["ds"] = pd.to_datetime(Xy[date_col])
    Xy["y"] = pd.to_numeric(Xy[target_col], errors="coerce")
    Xy = Xy.dropna(subset=["y"])
    Xy["t"] = (Xy["ds"] - Xy["ds"].min()).dt.days
    Xy["dow"] = Xy["ds"].dt.dayofweek
    Xy["dom"] = Xy["ds"].dt.day
    Xy["month"] = Xy["ds"].dt.month
    for lag in [1, 7, 14]:
        Xy[f"lag_{lag}"] = Xy["y"].shift(lag)
    Xy["roll_7"] = Xy["y"].rolling(7, min_periods=1).mean()
    Xy["roll_14"] = Xy["y"].rolling(14, min_periods=1).mean()
    Xy = Xy.dropna()
    feats = ["t","dow","dom","month","lag_1","lag_7","lag_14","roll_7","roll_14"]
    return Xy, feats

Xy, feats = make_features(df, date_col, target_col)
if len(Xy) < 60:
    st.warning("Less than 60 rows after feature engineering. Forecasts may be unstable.")

X = Xy[feats]
y = Xy["y"]

# --- Modular Prediction Function ---
def run_model_prediction(X, y, model_name, test_size, enable_download=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, shuffle=False)
    model = LinearRegression() if model_name == "Linear Regression" else RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    y_test = np.ravel(y_test)
    pred = np.ravel(pred)
    results_df = pd.DataFrame({"actual": pd.Series(y_test, index=X_test.index), "predicted": pd.Series(pred, index=X_test.index)})
    st.subheader("Actual vs Predicted Results")
    st.dataframe(results_df, use_container_width=True)
    if enable_download:
        st.download_button("Download Results as CSV", results_df.to_csv(index=False).encode("utf-8"), "prediction_results.csv", "text/csv", key="download_results_unique")
    try:
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        st.subheader("📊 Model Evaluation Metrics")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")
    except Exception as e:
        st.error(f"❌ Metric calculation failed: {e}")
    return model, y_test, pred, X_test

# --- Run Prediction ---
model, y_test, pred, X_test = run_model_prediction(X, y, model_name, test_size, enable_download=download_toggle)

# --- Actual vs Predicted Chart ---
plot_df = pd.DataFrame({
    "date": Xy.loc[X_test.index, "ds"],
    "actual": y_test,
    "predicted": pred
})
fig = px.line(plot_df, x="date", y=["actual","predicted"], title="Actual vs Predicted")
st.plotly_chart(fig, use_container_width=True)

# --- Forecast Future ---
last_date = Xy["ds"].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
fcast_frame = pd.DataFrame({date_col: future_dates})
tmp = pd.concat([df[[date_col, target_col]].copy(), fcast_frame], ignore_index=True)
tmp[date_col] = pd.to_datetime(tmp[date_col])

F, feats2 = make_features(tmp, date_col, target_col)
F_future = F[F["ds"].isin(future_dates)]

if len(F_future) == 0:
    st.warning("Not enough history to generate features for the requested horizon. Try a smaller horizon or ensure daily frequency.")
else:
    yhat = model.predict(F_future[feats])
    forecast_df = pd.DataFrame({"date": F_future["ds"], "forecast": yhat})
    hist = Xy[["ds", "y"]].rename(columns={"ds": "date", "y": "value"})
    hist["series"] = "history"
    fplot = forecast_df.rename(columns={"forecast": "value"})
    fplot["series"] = "forecast"
    chart_df = pd.concat([hist, fplot], ignore_index=True)
    fig2 = px.line(chart_df, x="date", y="value", color="series", title="History + Forecast")
    st.plotly_chart(fig2, use_container_width=True)

    if download_toggle:
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
