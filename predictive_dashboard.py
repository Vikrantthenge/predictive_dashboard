# app.py
# Predictive Maintenance Optimization Dashboard
# Forecast equipment failures with selectable models, smoothing, downloads, and sample CSV support
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
import warnings
warnings.filterwarnings("ignore")

# Optional libs
try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import pmdarima as pm
except Exception:
    pm = None

# Provide SARIMAX fallback for AR-like modelling if pmdarima not available
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

# ---------------------------
# Page config and header
# ---------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🔧", layout="wide")

st.markdown("""
<div style='text-align: center'>
    <img src='https://github.com/Vikrantthenge/predictive_dashboard/blob/main/asset/create%20a%20matching%20he.png?raw=true' width='220' />
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.title-box {
    background: linear-gradient(90deg, #ff6a00, #ee0979, #9b59b6, #00c6ff);
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}
.title-text {
    font-size: 24px;
    font-weight: 700;
    color: white;
}
.small-muted { color: #6C757D; font-size:14px; text-align:center; }
</style>
<div class='title-box'>
    <div class='title-text'>📈🔧 Predictive Maintenance Optimization</div>
</div>
<p class='small-muted'>Forecast failures, visualize smoothed trends, and export results — deployable on Streamlit Cloud or EC2</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Sidebar: inputs and upload
# ---------------------------
st.sidebar.header("User inputs")
st.sidebar.markdown("Upload your CSV or use the built-in sample. Required: a date column and a numeric target column (failures).")

uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=(uploaded is None))
encoding_choice = st.sidebar.selectbox("File encoding", ["utf-8", "ISO-8859-1", "latin1"], index=0)

# Editable field names (defaults for maintenance)
target_col = st.sidebar.text_input("Target column (numeric)", value="failures")
date_col = st.sidebar.text_input("Date column", value="date")
category_cols = st.sidebar.text_input("Category columns (optional, comma-separated)", value="equipment,region")
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30, step=1)
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
download_toggle = st.sidebar.checkbox("Enable download buttons", value=True)

# Model choices
available_models = ["Linear Regression", "Random Forest"]
if Prophet is not None:
    available_models.append("Prophet")
if pm is not None or SARIMAX is not None:
    available_models.append("ARIMA")
model_name = st.sidebar.selectbox("Model", available_models)

st.sidebar.markdown("ℹ️ Sample: daily maintenance counts. Use `date` as Date column and `failures` as Target column.")

# ---------------------------
# Synthetic sample generation / load
# ---------------------------
@st.cache_data
def generate_synthetic_data():
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
    np.random.seed(42)
    base = np.sin(np.linspace(0, 8, len(dates))) * 2 + 6
    noise = np.random.normal(0, 0.6, len(dates))
    failures = (base + noise).round().astype(int)
    equipment = np.random.choice(["Pump-A", "Motor-B", "Compressor-C"], size=len(dates))
    region = np.random.choice(["Plant-1", "Plant-2"], size=len(dates))
    df = pd.DataFrame({date_col: dates, target_col: failures, "equipment": equipment, "region": region})
    return df

@st.cache_data
def load_sample():
    return generate_synthetic_data()

# ---------------------------
# Load data
# ---------------------------
def load_data():
    if uploaded is not None:
        try:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding=encoding_choice)
            st.sidebar.success("✅ Uploaded file loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    elif use_sample:
        df = load_sample()
        st.sidebar.info("Using sample data")
    else:
        st.error("No data source selected. Upload CSV or tick 'Use sample data'.")
        st.stop()
    return df

df = load_data()

# ---------------------------
# Validate and preprocess
# ---------------------------
if date_col not in df.columns:
    # try to infer a date-like column
    candidates = [c for c in df.columns if 'date' in c.lower() or 'day' in c.lower()]
    if candidates:
        inferred = candidates[0]
        st.warning(f"Date column '{date_col}' not found. Using '{inferred}' instead.")
        date_col = inferred
    else:
        st.error(f"Required date column '{date_col}' not found in dataset.")
        st.stop()

if target_col not in df.columns:
    st.error(f"Required numeric target column '{target_col}' not found in dataset.")
    st.stop()

# Parse date, ensure numeric target
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[date_col, target_col])
df = df.sort_values(by=date_col).reset_index(drop=True)

# optional categories
cat_cols = [c.strip() for c in category_cols.split(",") if c.strip() and c.strip() in df.columns]

# Display basic info
st.subheader("📊 Data preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown(f"**Rows:** {len(df):,}   |   **Date range:** {df[date_col].min().date()} to {df[date_col].max().date()}")

# ---------------------------
# Demo chart + smoothed trend
# ---------------------------
st.subheader("📈 Smoothed performance trend")
window_size = st.slider("Smoothing window size (days)", min_value=3, max_value=30, value=7, step=1)
df_sorted = df.sort_values(by=date_col)
df_sorted['moving_avg'] = df_sorted[target_col].rolling(window=window_size, min_periods=1).mean()

fig_trend = px.line(df_sorted, x=date_col, y='moving_avg', title=f"Smoothed Trend (window={window_size} days)")
fig_trend.update_layout(hovermode="x unified")
st.plotly_chart(fig_trend, use_container_width=True)
latest_val = df_sorted['moving_avg'].iloc[-1]
latest_dt = df_sorted[date_col].iloc[-1]
st.metric("Latest smoothed value", f"{latest_val:.2f}", delta=f"as of {latest_dt.date()}")

# ---------------------------
# Feature engineering for ML models
# ---------------------------
def make_features(df_local, date_col_local, target_col_local):
    Xy = df_local[[date_col_local, target_col_local]].copy().rename(columns={date_col_local: "ds", target_col_local: "y"})
    Xy["ds"] = pd.to_datetime(Xy["ds"])
    Xy = Xy.sort_values("ds").reset_index(drop=True)
    Xy["t"] = (Xy["ds"] - Xy["ds"].min()).dt.days
    Xy["dow"] = Xy["ds"].dt.dayofweek
    Xy["dom"] = Xy["ds"].dt.day
    Xy["month"] = Xy["ds"].dt.month
    for lag in [1, 7, 14]:
        Xy[f"lag_{lag}"] = Xy["y"].shift(lag)
    Xy["roll_7"] = Xy["y"].rolling(7, min_periods=1).mean()
    Xy["roll_14"] = Xy["y"].rolling(14, min_periods=1).mean()
    Xy = Xy.dropna().reset_index(drop=True)
    feats = ["t", "dow", "dom", "month", "lag_1", "lag_7", "lag_14", "roll_7", "roll_14"]
    return Xy, feats

Xy, feats = make_features(df_sorted, date_col, target_col)

if Xy.empty:
    st.error("Not enough data after feature engineering. Need more consecutive observations or reduce lag/rolling windows.")
    st.stop()

# ---------------------------
# Model training and evaluation
# ---------------------------
def run_model_prediction(Xy_local, feats_local, model_name_local, horizon_local, test_pct_local, enable_download=True):
    X = Xy_local[feats_local]
    y = Xy_local["y"]

    if model_name_local in ["Linear Regression", "Random Forest"]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct_local/100.0, shuffle=False)
        model = LinearRegression() if model_name_local == "Linear Regression" else RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📊 Actual vs Predicted (test set)")
        results_df = pd.DataFrame({
            "date": Xy_local.loc[X_test.index, "ds"].dt.date,
            "actual": y_test.values,
            "predicted": np.round(y_pred, 2)
        }).reset_index(drop=True)
        st.dataframe(results_df, use_container_width=True)

        if enable_download:
            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", csv_bytes, "predictions.csv", "text/csv")

        st.subheader("📊 Model metrics (test set)")
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.metric("R²", f"{r2_score(y_test, y_pred):.2f}")

        return model, X_test.index

    elif model_name_local == "Prophet" and Prophet is not None:
        m = Prophet()
        m.fit(Xy_local[["ds", "y"]])
        future = m.make_future_dataframe(periods=horizon_local)
        forecast = m.predict(future)
        st.subheader("📈 Prophet Forecast (history + forecast)")
        fig = px.line(forecast, x="ds", y="yhat", title="Prophet forecast")
        st.plotly_chart(fig, use_container_width=True)
        return m, None

    elif model_name_local == "ARIMA":
        # Prefer pmdarima if available
        if pm is not None:
            try:
                model = pm.auto_arima(Xy_local["y"], seasonal=False, error_action="ignore", suppress_warnings=True)
                st.info("ARIMA model built with pmdarima.auto_arima")
                return model, None
            except Exception as e:
                st.error(f"ARIMA (pmdarima) failed: {e}")
                return None, None
        elif SARIMAX is not None:
            try:
                sar = SARIMAX(Xy_local["y"], order=(1,1,1))
                res = sar.fit(disp=False)
                st.info("ARIMA-like model built with SARIMAX")
                return res, None
            except Exception as e:
                st.error(f"SARIMAX failed: {e}")
                return None, None
        else:
            st.error("ARIMA selected but neither pmdarima nor statsmodels SARIMAX is available.")
            return None, None
    else:
        st.error("Unsupported model or required library not available.")
        return None, None

model_obj, test_index = run_model_prediction(Xy, feats, model_name, horizon, test_size, enable_download=download_toggle)

# ---------------------------
# Forecast generation and plotting
# ---------------------------
last_date = Xy["ds"].max()
future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")

# Prepare a frame that includes history + future for feature creation
tmp = pd.concat([df_sorted[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"}), 
                 pd.DataFrame({"ds": future_dates, "y": [np.nan]*len(future_dates)})], ignore_index=True)
tmp["ds"] = pd.to_datetime(tmp["ds"])
F, feats2 = make_features(tmp, "ds", "y")  # may drop rows with NaN y in history but feature functions require y; handled below

# Get rows matching future dates from F
F_future = F[F["ds"].isin(future_dates)]
forecast_df = pd.DataFrame()

if model_obj is not None and not F_future.empty:
    try:
        if model_name in ["Linear Regression", "Random Forest"]:
            yhat = model_obj.predict(F_future[feats])
            forecast_df = pd.DataFrame({"date": F_future["ds"].dt.date, "forecast": np.round(yhat, 2)})
        elif model_name == "Prophet" and Prophet is not None:
            future_df = model_obj.make_future_dataframe(periods=horizon)
            forecast = model_obj.predict(future_df)
            forecast_df = forecast[forecast["ds"].dt.date > last_date.date()][["ds", "yhat"]].rename(columns={"ds":"date","yhat":"forecast"})
            forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.date
        elif model_name == "ARIMA":
            if pm is not None:
                yhat = model_obj.predict(n_periods=horizon)
                forecast_df = pd.DataFrame({"date": future_dates.date, "forecast": np.round(yhat,2)})
            elif SARIMAX is not None:
                pred = model_obj.get_forecast(steps=horizon)
                mean = pred.predicted_mean
                forecast_df = pd.DataFrame({"date": future_dates.date, "forecast": np.round(mean.values, 2)})
    except Exception as e:
        st.error(f"Forecast generation failed: {e}")

# Plot history + forecast
st.subheader("📈 Historical vs Forecasted")
hist_plot = df_sorted[[date_col, target_col]].rename(columns={date_col:"date", target_col:"value"})
hist_plot["date"] = pd.to_datetime(hist_plot["date"])
if not forecast_df.empty:
    fplot = forecast_df.rename(columns={"date":"date","forecast":"value"})
    fplot["date"] = pd.to_datetime(fplot["date"])
    chart_df = pd.concat([hist_plot.assign(series="history"), fplot.assign(series="forecast")], ignore_index=True)
else:
    chart_df = hist_plot.assign(series="history")

fig2 = px.line(chart_df, x="date", y="value", color="series", title="History vs Forecast")
st.plotly_chart(fig2, use_container_width=True)

if download_toggle and not forecast_df.empty:
    buf = io.StringIO()
    forecast_df.to_csv(buf, index=False)
    st.download_button("⬇️ Download Forecast CSV", buf.getvalue().encode("utf-8"), "forecast.csv", "text/csv")

if forecast_df.empty:
    st.warning("Not enough data or model unavailable to generate future forecast.")

# ---------------------------
# Optional category analysis
# ---------------------------
if cat_cols:
    st.subheader("🔍 Category breakdown")
    cols_present = [c for c in cat_cols if c in df.columns]
    if cols_present:
        for c in cols_present:
            agg = df.groupby(c)[target_col].sum().reset_index().sort_values(by=target_col, ascending=False)
            figc = px.bar(agg, x=c, y=target_col, title=f"Total {target_col} by {c}")
            st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("No configured category columns found in the dataset.")

# ---------------------------
# Simple optimization simulation (optional)
# ---------------------------
st.subheader("⚙️ Maintenance scheduling simulation")
st.markdown("Quick simulation: change inspection frequency and see simple expected failure reduction estimate.")

inspection_freq = st.slider("Inspection frequency (days)", 7, 60, 30)
response_time = st.slider("Average response time (hours)", 1, 72, 24)
# crude sim: more frequent inspections reduce failures linearly for demonstration only
base_failures = df_sorted[target_col].mean()
expected_failures = base_failures * (30/inspection_freq) * (1 - min(response_time/168, 0.5))
st.metric("Simulated expected daily failures", f"{expected_failures:.2f}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding-top: 12px; font-family: Arial, sans-serif;">
    <div style="font-size:16px; font-weight:600;">🔧 Built by Vikrant Thenge</div>
    <div style="color:#666; font-size:13px; margin-bottom:6px;">Data & Analytics | Predictive Maintenance | Streamlit</div>
    <div>
        <a href="mailto:vikrantthenge@outlook.com" title="Email" style="margin:0 8px;"><img src="https://img.icons8.com/color/48/000000/microsoft-outlook.png" width="22"/></a>
        <a href="https://github.com/Vikrantthenge" target="_blank" title="GitHub" style="margin:0 8px;"><img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" width="22"/></a>
        <a href="https://www.linkedin.com/in/vthenge/" target="_blank" title="LinkedIn" style="margin:0 8px;"><img src="https://img.icons8.com/color/48/000000/linkedin.png" width="22"/></a>
    </div>
</div>
""", unsafe_allow_html=True)
