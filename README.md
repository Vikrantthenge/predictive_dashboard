# 📈 Predictive Maintenance Dashboard

<p align="left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" alt="AWS Logo" height="25">
  <a href="http://16.171.175.12:8501/">
    <img src="https://img.shields.io/badge/Predictive-Maintenance-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="Live EC2 Demo">
  </a>
</p>

[![Email](https://img.shields.io/badge/Outlook-vikrantthenge@outlook.com-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white)](mailto:vikrantthenge@outlook.com)  
[![Tech Stack](https://img.shields.io/badge/🧠-Python_·_Streamlit_·_Pandas_·_Scikit--learn_·_Prophet_·_Plotly-6A5ACD?style=for-the-badge)](#)

---

## 🚀 Live Demo

Your Streamlit app is now live and auto-starts on reboot.

[![Launch App](https://img.shields.io/badge/Open%20App-%2316.171.175.12:8501-brightgreen?style=for-the-badge&logo=streamlit)](http://16.171.175.12:8501)

📌 **Auto-start enabled** via `systemd`  
🔁 No manual SSH needed after reboot  
🔒 **Indian IP restriction** preferred for public access control  
🌐 Optional: Map to `forecast.vikrantthenge.in`  
📎 Scan below to open on mobile

![QR Code](https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=http://16.171.175.12:8501)

---

## 🧭 Overview

**ForecastFlow** is a recruiter-facing Streamlit app that transforms raw business data into **actionable forecasts and insights**.  
It empowers users to upload datasets, explore trends, select forecasting models, and visualize predictions — all in a responsive, production-ready dashboard.

---

## ☁️ Cloud Deployment

Deployed on **AWS EC2 Free Tier**, optimized for public access with secure port bindings and auto-restart.  
Designed for recruiter-grade uptime and cost-efficient hosting of live dashboard previews.

---

## 🎯 Key Features

- 📁 **Upload or use sample data** — instantly start exploring insights  
- 📊 **Dynamic Model Selection** — choose Linear Regression, Random Forest, or Prophet  
- 📅 **Trend & Forecast Visualization** — seasonality, confidence intervals, and horizon control  
- 📈 **Smoothed Performance Trends** — rolling averages with adjustable window size  
- 📥 **Downloadable Outputs** — export predictions and forecasts in CSV format  
- 🧠 **Feature Engineering** — lag features, rolling averages, calendar signals  
- 🖼️ **Branded & Responsive UI** — sidebar controls, gradient header, emoji framing  
- ⚠️ **ARIMA Notice** — ARIMA disabled on cloud due to Cython limitations

---

## 📂 Sample Data Format

The app expects a CSV with:
- A **date column** (e.g., `date`)
- A **numeric target column** (e.g., `failures`, `sales`, `downtime`)
- Optional **categorical columns** (e.g., `product`, `region`)

Use the built-in synthetic dataset or upload your own.

---

## 🧪 Model Logic

- **Linear Regression** and **Random Forest** use lag features, rolling averages, and calendar-based signals  
- **Prophet** handles seasonality and trend decomposition automatically  
- Forecast horizon is user-defined (7–180 days)  
- Model metrics include **MAE**, **RMSE**, and **R²**

---

## 🧰 Usage Instructions

1. Upload your CSV or use the sample  
2. Configure model, forecast horizon, and test size  
3. View smoothed trends and forecasted values  
4. Download predictions and forecasts as CSV  

---

## 📦 Requirements

```txt
streamlit>=1.29.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
prophet>=1.1.5
