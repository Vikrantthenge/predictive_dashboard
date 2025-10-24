# 📈 Predictive Maintenance Dashboard

<p align="left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" alt="AWS Logo" height="25">
  <a href="http://16.171.175.12:8501/">
    <img src="https://img.shields.io/badge/Predictive-Maintenance-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="Live EC2 Demo">
  </a>
</p>

[![Open Dashboard](https://img.shields.io/badge/Live%20App-Streamlit%20Dashboard-brightgreen?logo=streamlit)](http://16.171.174.123:8501)

[![Email](https://img.shields.io/badge/Outlook-vikrantthenge@outlook.com-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white)](mailto:vikrantthenge@outlook.com)  

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](#)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Prophet](https://img.shields.io/badge/Prophet-003B73?style=for-the-badge&logo=python&logoColor=white)](#)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](#)
[![AWS EC2](https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](#)

[![Systemd](https://img.shields.io/badge/systemd-auto--restart-0078D4?style=for-the-badge&logo=linux&logoColor=white)](#)
[![QR Access](https://img.shields.io/badge/QR-Mobile_Access-34A853?style=for-the-badge&logo=qr-code&logoColor=white)](#)
[![IP Restriction](https://img.shields.io/badge/IP_Restricted-India_Only-FF0000?style=for-the-badge&logo=security&logoColor=white)](#)
[![Custom Domain](https://img.shields.io/badge/Custom_Domain-forecast.vikrantthenge.in-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](#)

☁️ Deployed on AWS EC2 Free Tier — auto-restarting, IP-restricted, and recruiter-accessible

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
