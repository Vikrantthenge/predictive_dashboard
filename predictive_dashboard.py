# predictive_dashboard.py
import io
from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Predictive Dashboard Generator", page_icon="📈", layout="wide")

# --- Header Image and Title ---
# If you have a banner image, place it at asset/predictive_dashboard_banner.png
banner_path = "asset/predictive_dashboard_banner.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_column_width=True)
st.title("📈 Predictive Dashboard Generator")

# ---------------- Sidebar Inputs (define these BEFORE data loading) ----------------
st.sidebar.header("User Input")
selected_feature = st.sidebar.selectbox("Select Feature (demo charts)", ["Feature A", "Feature B", "Feature C"])

st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload your CSV or use the sample. Required: a **date** column and a **numeric target** (e.g., sales). Optional: categorical columns (e.g., product, region).")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=(uploaded is None))

encoding_choice = st.sidebar.selectbox(
    "File encoding",
    options=["utf-8", "ISO-8859-1", "latin1"],
    index=0,
    help="Choose encoding if your file fails to load. UTF-8 is default."
)

target_col = st.sidebar.text_input("Target column (numeric)", "sales")
date_col = st.sidebar.text_input("Date column", "date")
category_cols = st.sidebar.text_input("Category columns (comma-separated, optional)", "product,region")

model_name = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest"])
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30, step=1)
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)

st.sidebar.markdown("---")
download_toggle = st.sidebar.checkbox("Enable predictions download", value=True)


# ---------------- Robust Data loading & date parsing ----------------
@st.cache_data
def load_sample():
    # prefer an ISO-formatted sample if available; fallback to other filenames
    for fname in ("sales_data_small_iso.csv", "sales_data_small.csv", "sales_data.csv"):
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname)
                return df, fname
            except Exception:
                continue
    raise FileNotFoundError("No sample CSV found. Place sales_data_small_iso.csv or sales_data.csv in the app folder.")

def try_parse_dates(df, date_col):
    """Try several date parsing strategies and return dataframe with parsed date column or raise."""
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not present in dataframe.")

    # if dtype already datetime-like, normalize and return
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        return df

    # attempt parse with dayfirst True (DD-MM-YYYY)
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        if parsed.notna().sum() / len(parsed) > 0.90:
            df[date_col] = parsed.dt.normalize()
            return df
    except Exception:
        pass

    # attempt parse with dayfirst False (MM-DD-YYYY)
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
        if parsed.notna().sum() / len(parsed) > 0.90:
            df[date_col] = parsed.dt.normalize()
            return df
    except Exception:
        pass

    # elementwise inference fallback (try common formats)
    def infer_date(x):
        for dfmt in ("%Y-%m-%d","%d-%m-%Y","%m-%d-%Y","%d/%m/%Y","%m/%d/%Y"):
            try:
                return pd.to_datetime(x, format=dfmt)
            except Exception:
                continue
        try:
            return pd.to_datetime(x, errors="raise")
        except Exception:
            return pd.NaT

    parsed = df[date_col].apply(infer_date)
    if parsed.notna().sum() / len(parsed) > 0.6:
        df[date_col] = pd.to_datetime(parsed).dt.normalize()
        return df

    raise ValueError("Date parsing failed — mixed or unrecognised formats detected; please use a consistent date format (prefer YYYY-MM-DD).")

# Load data (uploaded or sample)
if use_sample:
    try:
        df, loaded_fname = load_sample()
        st.info(f"Loaded sample file: {loaded_fname}")
    except FileNotFoundError as fe:
        st.error(str(fe))
        st.stop()
    except Exception as e:
        st.error(f"Unable to load sample data: {e}")
        st.stop()
else:
    if uploaded is not None:
        try:
            # read uploaded file using selected encoding
            # uploaded is a BytesIO-like object so pass directly to pandas
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding=encoding_choice)
            st.info("Uploaded file loaded.")
        except UnicodeDecodeError:
            st.error(f"Failed to decode file with encoding '{encoding_choice}'. Try a different option.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            st.stop()
    else:
        st.stop()

# If the user-specified date column is missing, try to guess a date-like column
if date_col not in df.columns:
    candidates = [c for c in df.columns if "date" in c.lower() or "day" in c.lower() or "dt" in c.lower()]
    if candidates:
        guessed = candidates[0]
        st.warning(f"Date column '{date_col}' not found. Auto-detected '{guessed}' as date column. If incorrect, set Date column in sidebar.")
        date_col = guessed
    else:
        st.error(f"Date column '{date_col}' not found in data. Please set the correct Date column name in the sidebar.")
        st.stop()

# Attempt robust date parsing (and normalize)
try:
    df = try_parse_dates(df, date_col)
except Exception as e:
    st.error(
        "Could not parse the date column automatically. Common fixes:\n"
        "- Ensure the CSV date column uses a consistent format like YYYY-MM-DD.\n"
        "- Set the correct Date column name in the sidebar.\n"
        f"Details: {e}"
    )
    st.stop()

# Validate required columns exist (date + target)
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in data. Please set the correct Target column name in the sidebar.")
    st.stop()

# Continue with cleaning
df = df.dropna(subset=[date_col, target_col])
df = df.sort_values(by=date_col)

# ---------------- Demo top chart (keeps your original demo) ----------------
# Demonstration lightweight chart using random data if you want to keep
# the earlier demo; otherwise the forecast charts below are the main UI.
demo_data = np.random.randn(100, 3)
demo_df = pd.DataFrame(demo_data, columns=["Feature A", "Feature B", "Feature C"])
st.subheader("Dashboard Overview")
st.write("This dashboard uses predictive analytics to visualize trends and forecast outcomes based on selected features.")
st.line_chart(demo_df[selected_feature])

# ---------------- Feature engineering ----------------
def make_features(frame: pd.DataFrame, date_col: str, target_col: str):
    Xy = frame[[date_col, target_col]].copy()
    Xy["ds"] = pd.to_datetime(Xy[date_col])
    Xy["y"] = pd.to_numeric(Xy[target_col], errors="coerce")
    Xy = Xy.dropna(subset=["y"])

    # Time features
    Xy["t"] = (Xy["ds"] - Xy["ds"].min()).dt.days
    Xy["dow"] = Xy["ds"].dt.dayofweek
    Xy["dom"] = Xy["ds"].dt.day
    Xy["month"] = Xy["ds"].dt.month

    # Lags & rolling means
    for lag in [1, 7, 14]:
        Xy[f"lag_{lag}"] = Xy["y"].shift(lag)
    Xy["roll_7"] = Xy["y"].rolling(7, min_periods=1).mean()
    Xy["roll_14"] = Xy["y"].rolling(14, min_periods=1).mean()

    Xy = Xy.dropna()
    feats = ["t","dow","dom","month","lag_1","lag_7","lag_14","roll_7","roll_14"]
    return Xy, feats

Xy, feats = make_features(df, date_col, target_col)
if len(Xy) < 60:
    st.warning("Data has fewer than 60 usable rows after feature engineering. Forecasts may be unstable.")

# ---------------- Train / Evaluate ----------------
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Split features and target
X = Xy[feats]
y = Xy["y"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import streamlit as st

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model selection via Streamlit
model_choice = st.selectbox("Choose model", ["Linear Regression", "Random Forest"])

if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
pred = model.predict(X_test)

# Flatten arrays if needed
y_test = np.ravel(y_test)
pred = np.ravel(pred)

# Create results DataFrame with index alignment
results_df = pd.DataFrame({
    "actual": pd.Series(y_test, index=X_test.index),
    "predicted": pd.Series(pred, index=X_test.index)
})

# Display results
st.subheader("Actual vs Predicted Results")
st.dataframe(results_df)

# Optional: Download button
st.download_button(
    label="Download Results as CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="prediction_results.csv",
    mime="text/csv"
)


# ✅ Define model here
model = LinearRegression()  # or whichever model you're using

# Fit model
model.fit(X_train, y_train)
# Train model
model.fit(X_train, y_train)

# Make predictions
pred = model.predict(X_test)

# Flatten arrays if needed
y_test = np.ravel(y_test)
pred = np.ravel(pred)

# ✅ Insert this block right here
results_df = pd.DataFrame({
    "actual": pd.Series(y_test, index=X_test.index),
    "predicted": pd.Series(pred, index=X_test.index)
})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100.0, shuffle=False
)

# Model selection
if model_name == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=300, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Ensure predictions and targets are 1D arrays
y_test = np.ravel(np.array(y_test))
pred = np.ravel(np.array(pred))

# Optional debug info
with st.expander("🔍 Debug Info"):
    st.write("📊 y_test sample:", y_test[:5])
    st.write("📈 pred sample:", pred[:5])
    st.write("🧮 Shapes:", y_test.shape, pred.shape)
    st.write("✅ Length match:", len(y_test) == len(pred))

# Validate lengths
assert len(y_test) == len(pred), "Mismatch in prediction and test label lengths"

# Metrics calculation with error handling
try:
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    # Display metrics
    st.subheader("📊 Model Evaluation Metrics")
    st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

except Exception as e:
    st.error(f"❌ Metric calculation failed: {e}")
    mae, rmse, r2 = None, None, None


# ---------------- Charts ----------------
# Actual vs Predicted
plot_df = pd.DataFrame({
    "date": Xy.loc[X_test.index, "ds"],
    "actual": y_test.values,
    "predicted": pred
})
fig = px.line(plot_df, x="date", y=["actual","predicted"], title="Actual vs Predicted")
st.plotly_chart(fig, use_container_width=True)

# Forecast future
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

    hist = Xy[["ds","y"]].rename(columns={"ds":"date","y":"value"})
    hist["series"] = "history"
    fplot = forecast_df.rename(columns={"forecast":"value"})
    fplot["series"] = "forecast"
    chart_df = pd.concat([hist, fplot], ignore_index=True)

    fig2 = px.line(chart_df, x="date", y="value", color="series", title="History + Forecast")
    st.plotly_chart(fig2, use_container_width=True)

    if download_toggle:
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")

# ---------------- Aggregations ----------------
with st.expander("📊 Aggregations & KPIs", expanded=False):
    if category_cols:
        # If user passed a comma-separated list earlier, convert to names present in df
        cat_cols = [c.strip() for c in category_cols.split(",") if c.strip() and c.strip() in df.columns]
    else:
        cat_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() < 200]

    if cat_cols:
        agg_dim = st.selectbox("Aggregate by", options=["(None)"] + cat_cols)
    else:
        agg_dim = "(None)"
    agg_metric = st.selectbox("Metric", options=["mean","sum","median"], index=1)

    if agg_dim != "(None)":
        agg_df = df.groupby([agg_dim, pd.Grouper(key=date_col, freq="M")])[target_col].agg(agg_metric).reset_index()
        fig3 = px.line(agg_df, x=date_col, y=target_col, color=agg_dim, title=f"Monthly {agg_metric} by {agg_dim}")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        monthly = df.groupby(pd.Grouper(key=date_col, freq="M"))[target_col].agg(agg_metric).reset_index()
        fig4 = px.bar(monthly, x=date_col, y=target_col, title=f"Monthly {agg_metric.title()}")
        st.plotly_chart(fig4, use_container_width=True)

st.caption("Built by Vikrant • Streamlit + scikit-learn + Plotly")
