# 📈 Predictive Dashboard Generator

[![Live Demo](https://img.shields.io/badge/🚀-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://predictivedashboard-vikrantthenge.streamlit.app)
[![Email](https://img.shields.io/badge/Outlook-vikrantthenge@outlook.com-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white)](mailto:vikrantthenge@outlook.com)
[![Tech Stack](https://img.shields.io/badge/🧠-Python_·_Streamlit_·_Pandas_·_Scikit--learn_·_Prophet_·_Plotly-6A5ACD?style=for-the-badge)](#)

---

## 🚀 Overview

The **Predictive Dashboard Generator** is a recruiter-facing Streamlit app that transforms raw business data into **actionable forecasts and insights**.  
It empowers users to upload datasets, explore trends, select forecasting models, and visualize predictions — all in a responsive, production-ready dashboard.

---

## 🎯 Key Features

- 📁 **Upload or use sample data** — instantly start exploring insights  
- 📊 **Dynamic Model Selection** — choose Linear Regression, Random Forest, or Prophet  
- 📅 **Trend and Forecast Visualization** — observe seasonality and confidence intervals  
- 📈 **Smoothed Performance Trends** — rolling averages with adjustable window size  
- 📥 **Downloadable Outputs** — export predictions and forecasts in CSV format  
- 🧠 **Feature Engineering** — lag features, rolling averages, calendar signals  
- 🖼️ **Branded & Responsive UI** — sidebar controls, gradient header, emoji framing  
- ⚠️ **ARIMA Notice** — ARIMA is disabled on cloud due to Cython limitations

---

## 📂 Sample Data Format

The app expects a CSV with:
- A **date column** (e.g., `date`)
- A **numeric target column** (e.g., `failures`, `sales`, `downtime`)
- Optional **categorical columns** (e.g., `product`, `region`)

You can use the built-in synthetic dataset or upload your own.

---

## 🧪 Model Logic

- **Linear Regression** and **Random Forest** use lag features, rolling averages, and calendar-based signals (day of week, month, etc.)
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

## 🖼️ App Preview

> 📌 *Preview image coming soon — this section will showcase a screenshot or GIF of the dashboard in action.*

![Dashboard Preview Placeholder](https://via.placeholder.com/800x400.png?text=Dashboard+Preview+Coming+Soon)

## 👨‍💻 Author

**Vikrant Thenge**  
Senior Data Analyst & Automation Strategist  
📫 [vikrantthenge@outlook.com](mailto:vikrantthenge@outlook.com)  
🔗 [GitHub Profile](https://github.com/vikrantthenge)  
🌐 [Live App](https://predictivedashboard-vikrantthenge.streamlit.app)


