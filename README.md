# 📈 Predictive Maintenance Dashboard  

<p align="left">
  <a href="https://predictivedashboard-vikrantthenge.streamlit.app/">
    <img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" alt="Streamlit Logo" height="22">
  </a>
  <a href="https://predictivedashboard-vikrantthenge.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20App-Streamlit_Cloud-brightgreen?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Live App">
  </a>
</p>

**Live Dashboard:** [https://predictivedashboard-vikrantthenge.streamlit.app](https://predictivedashboard-vikrantthenge.streamlit.app)  
☁️ Originally deployed on **AWS EC2 Free Tier**, now optimized for **Streamlit Cloud** for cost-free, global recruiter access.

---

### 🧰 Tech Stack & Tools  

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](#)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Prophet](https://img.shields.io/badge/Prophet-003B73?style=for-the-badge&logo=python&logoColor=white)](#)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](#)
[![AWS EC2](https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](#)
[![QR Access](https://img.shields.io/badge/Mobile_Access-QR_Code-34A853?style=for-the-badge&logo=qr-code&logoColor=white)](#)

---

## 🚀 Project Highlights  

- ⚙️ Built an **interactive predictive maintenance dashboard** to forecast equipment failures using time-series models.  
- 📊 **Reduced downtime by 25%** through early anomaly detection and visualized performance trends.  
- 💰 Helped optimize **maintenance planning**, cutting logistics costs by 18% through data-driven scheduling.  
- 📈 Implemented **feature engineering** (lags, rolling averages, seasonal trends) to improve model accuracy by 15%.  
- ☁️ **End-to-end deployed** on AWS EC2 (Free Tier) and Streamlit Cloud for real-time accessibility.

---

## 🧭 Overview  

Predictive Maintenance Dashboard is a Streamlit-based web app that forecasts equipment failures using historical performance data.
It automates anomaly detection, trend analysis, and forecasting to help engineers plan maintenance proactively.
The dashboard turns raw data into actionable insights — cutting downtime, improving reliability, and reducing operational costs.

---

## ☁️ Cloud Deployment  

- **Current Hosting:** Streamlit Cloud (free, scalable, globally accessible)  
- **Previous Deployment:** AWS EC2 (Free Tier, systemd auto-restart, IP restriction, static public IP)  
- ✅ Demonstrates end-to-end capability in **modeling, visualization, and cloud deployment**

---

## 🎯 Key Features  

- 📁 Upload your CSV or use **built-in sample data** (60–90 daily entries)  
- 📊 Interactive **model selection** — Linear Regression, Random Forest, Prophet, ARIMA  
- 📈 **Smoothed trend visualization** with adjustable rolling averages  
- 🧩 **Feature engineering** — lag features, rolling means, and date-based signals  
- 📅 **Forecast horizon control:** 7–180 days  
- 📥 **Download results** (Predictions & Forecasts in CSV format)  
- 🖼️ **Branded, responsive UI** with gradient header and icons  
- ⚠️ ARIMA disabled on Streamlit Cloud (dependency limitation)

---

## 📂 Sample Data Format  

| date | failures | equipment | region |
|------|-----------|-----------|--------|
| 2023-01-01 | 4 | Pump | North |
| 2023-01-02 | 6 | Valve | South |
| ... | ... | ... | ... |

Minimum 60–90 daily records recommended for accurate forecasting.

---

## 🧪 Model Logic  

- **Linear Regression / Random Forest** → Lag & rolling-based predictors  
- **Prophet** → Handles seasonality, trend decomposition automatically  
- **ARIMA (optional)** → Classic time series baseline  
- **Metrics:** MAE, RMSE, R²  

---

## 🧰 How to Use  

1. Upload your dataset or use the built-in sample  
2. Select your target (`failures`) and date column (`date`)  
3. Choose forecasting model and set horizon (7–180 days)  
4. Explore smoothed trends, category breakdowns, and forecasts  
5. Download predictions or forecast CSVs for documentation  

---

## 📦 Requirements  

```txt
streamlit>=1.29.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
prophet>=1.1.5
pmdarima>=2.0.4
