# 🔧 Predictive Maintenance Optimization Dashboard
# Forecast equipment failures using ML and time-series models
# 📊 Interactive, deployable, and visually enhanced
# 💼 Built by Vikrant Thenge

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

# Optional libraries
try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import pmdarima as pm
except Exception:
    pm = None

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🔧", layout="wide")

st.markdown("""
<div style='text-align: center'>
    <img src='https://github.com/Vikrantthenge/predictive_dashboard/blob/main/asset/create%20a%20matching%20he.png?raw=true' width='230'/>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.title-box {
    background: linear-gradient(90deg, #ff6a00, #ee0979, #9b59b6, #00c6ff);
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.title-text {
    font-size: 26px;
    font-weight: 700;
    color: white;
}
.small-muted { color: #6C757D; font-size:14px; text-align:center; }
</style>
<div class='title-box'>
    <div class='title-text'>📈🔧 Predictive Maintenance Optimization</div>
</div>
<p class='small-muted'>Forecast failures, visualize smoothed trends, and export results — powered by Streamlit</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=(uploaded is None))
target_col = st.sidebar.text_input("Target column", "failures")
date_col = st.sidebar.text_input("Date column", "date")
category_cols = st.sidebar.text_input("Category columns (comma-separated)", "equipment,region")
horizon = st.sidebar.number_input("Forecast horizon (days)", 7, 365, 30)
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20)
download_toggle = st.sidebar.checkbox("Enable download", True)

available_models = ["Linear Regression", "Random Forest"]
if Prophet is not None:
    available_models.append("Prophet")
if pm is not None:
    available_models.append("ARIMA")
model_name = st.sidebar.selectbox("Model", available_models)

# ---------------------------
# Generate sample data
# ---------------------------
@st.cache_data
def generate_synthetic_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90, freq="D")
    np.random.seed(42)
    trend = np.linspace(3, 10, len(dates))
    noise = np.random.normal(0, 1, len(dates))
    failures = np.maximum(0, (trend + noise).round().astype(int))
    equipment = np.random.choice(["Pump-A", "Pump-B", "Compressor-C"], len(dates))
    region = np.random.choice(["Plant-1", "Plant-2", "Plant-3"], len(dates))
    return pd.DataFrame({"date": dates, "failures": failures, "equipment": equipment, "region": region})

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File loaded successfully")
        return df
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

# ---------------------------
# Load dataset
# ---------------------------
if uploaded is not None:
    df = load_data(uploaded)
elif use_sample:
    df = generate_synthetic_data()
else:
    st.error("Please upload a CSV or use sample data.")
    st.stop()

# Validate required columns
if date_col not in df.columns or target_col not in df.columns:
    st.warning("⚠️ Missing required columns. Loading sample data instead.")
    df = generate_synthetic_data()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[date_col, target_col])
df = df.sort_values(by=date_col).reset_index(drop=True)

if len(df) < 30:
    st.warning("⚠️ Dataset too small for forecasting. Using sample data instead.")
    df = generate_synthetic_data()

# ---------------------------
# Data preview
# ---------------------------
st.subheader("📊 Data Preview")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"Records: {len(df)} | Date range: {df[date_col].min().date()} → {df[date_col].max().date()}")

# ---------------------------
# Smoothed Trend
# ---------------------------
st.subheader("📈 Smoothed Failure Trend")
window_size = st.slider("Smoothing window", 3, 30, 7)
df["moving_avg"] = df[target_col].rolling(window=window_size, min_periods=1).mean()

fig_trend = px.line(
    df, x=date_col, y="moving_avg",
    title=f"Smoothed Trend (window={window_size})",
    color_discrete_sequence=["#ff6a00"]
)
fig_trend.update_layout(hovermode="x unified", title_x=0.0)
st.plotly_chart(fig_trend, use_container_width=True)

st.metric("Latest smoothed value", f"{df['moving_avg'].iloc[-1]:.2f}",
          delta=f"as of {df[date_col].iloc[-1].date()}")

# ---------------------------
# Feature Engineering
# ---------------------------
def make_features(df_local, date_col_local, target_col_local):
    df_local = df_local[[date_col_local, target_col_local]].rename(columns={date_col_local: "ds", target_col_local: "y"})
    df_local["ds"] = pd.to_datetime(df_local["ds"])
    df_local = df_local.sort_values("ds").reset_index(drop=True)
    df_local["t"] = (df_local["ds"] - df_local["ds"].min()).dt.days
    for lag in [1, 3, 7]:
        df_local[f"lag_{lag}"] = df_local["y"].shift(lag)
    df_local["roll_7"] = df_local["y"].rolling(7, min_periods=1).mean()
    df_local = df_local.dropna().reset_index(drop=True)
    feats = [c for c in df_local.columns if c not in ["ds", "y"]]
    return df_local, feats

Xy, feats = make_features(df, date_col, target_col)

# ---------------------------
# Model Training
# ---------------------------
def train_and_forecast(Xy, feats, model_name, horizon, test_size):
    if len(Xy) < 20:
        st.error("Not enough data after feature engineering.")
        return None, pd.DataFrame()

    X, y = Xy[feats], Xy["y"]

    if model_name in ["Linear Regression", "Random Forest"]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=False)
        model = LinearRegression() if model_name == "Linear Regression" else RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("📊 Actual vs Predicted")
        results_df = pd.DataFrame({
            "date": Xy.loc[X_test.index, "ds"].dt.date,
            "actual": y_test.values,
            "predicted": np.round(y_pred, 2)
        })
        st.dataframe(results_df, use_container_width=True)
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.metric("R²", f"{r2_score(y_test, y_pred):.2f}")

        # Forecast future
        last_date = Xy["ds"].max()
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
        future_feats = pd.DataFrame({
            "t": (future_dates - Xy["ds"].min()).days,
            "lag_1": [y.iloc[-1]] * horizon,
            "lag_3": [y.iloc[-3]] * horizon,
            "lag_7": [y.iloc[-7]] * horizon,
            "roll_7": [y.iloc[-7:].mean()] * horizon,
        })
        forecast = model.predict(future_feats)
        forecast_df = pd.DataFrame({"date": future_dates, "forecast": np.round(forecast, 2)})
        return model, forecast_df

    elif model_name == "Prophet" and Prophet is not None:
        m = Prophet()
        m.fit(Xy[["ds", "y"]])
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        forecast_df = forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "forecast"})
        return m, forecast_df.tail(horizon)

    elif model_name == "ARIMA" and pm is not None:
        model = pm.auto_arima(Xy["y"], seasonal=False, suppress_warnings=True)
        forecast = model.predict(n_periods=horizon)
        forecast_df = pd.DataFrame({"date": pd.date_range(Xy["ds"].max()+timedelta(days=1), periods=horizon), "forecast": np.round(forecast, 2)})
        return model, forecast_df

    else:
        st.error("Model unavailable or unsupported.")
        return None, pd.DataFrame()

# Run model
model, forecast_df = train_and_forecast(Xy, feats, model_name, horizon, test_size)

# ---------------------------
# Plot Historical vs Forecast
# ---------------------------
st.subheader("📈 Historical vs Forecasted")
hist = df[[date_col, target_col]].rename(columns={date_col: "date", target_col: "value"})
hist["series"] = "history"
if model is not None and not forecast_df.empty:
    fplot = forecast_df.rename(columns={"forecast": "value"})
    fplot["series"] = "forecast"
    chart_df = pd.concat([hist, fplot])
else:
    chart_df = hist

fig2 = px.line(chart_df, x="date", y="value", color="series",
               color_discrete_sequence=["#00c6ff", "#ff6a00"],
               title="History vs Forecast")
st.plotly_chart(fig2, use_container_width=True)

if download_toggle and not forecast_df.empty:
    st.download_button("⬇️ Download Forecast CSV",
                       forecast_df.to_csv(index=False).encode("utf-8"),
                       "forecast.csv", "text/csv")

# ---------------------------
# Category Breakdown (Enhanced)
# ---------------------------
cat_cols = [c.strip() for c in category_cols.split(",") if c.strip() in df.columns]
if cat_cols:
    st.subheader("🔍 Category Breakdown (Enhanced)")
    for c in cat_cols:
        agg = df.groupby(c)[target_col].sum().reset_index().sort_values(by=target_col, ascending=False)
        figc = px.bar(
            agg, x=c, y=target_col, color=c,
            color_discrete_sequence=px.colors.qualitative.Bold,
            text=target_col,
            title=f"Total {target_col.capitalize()} by {c.capitalize()}"
        )
        figc.update_traces(texttemplate="%{text}", textposition="outside",
                           marker=dict(line=dict(color="black", width=1)))
        figc.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figc, use_container_width=True)

# ---------------------------
# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding-top:15px; font-family:Arial, sans-serif;">
    <p style="font-size:15px; color:#555; margin-bottom:8px;">
        🔧 <b>Built by Vikrant Thenge</b> — Data & Analytics | Streamlit | Machine Learning
    </p>
    <div style="margin-top:8px;">
        <a href="https://www.linkedin.com/in/vthenge/" target="_blank" title="LinkedIn" style="margin:0 10px;">
            <img src="https://img.icons8.com/color/48/000000/linkedin.png" width="26" style="vertical-align:middle;"/>
        </a>
        <a href="https://github.com/Vikrantthenge" target="_blank" title="GitHub" style="margin:0 10px;">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" width="26" style="vertical-align:middle;"/>
        </a>
        <a href="mailto:vikrantthenge@outlook.com" title="Email" style="margin:0 10px;">
            <img src="https://img.icons8.com/color/48/000000/microsoft-outlook.png" width="26" style="vertical-align:middle;"/>
        </a>
    </div>
    <p style="color:#888; font-size:13px; margin-top:10px;">© 2025 Vikrant Thenge</p>
</div>
""", unsafe_allow_html=True)

