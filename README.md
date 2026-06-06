# 📈 Bitcoin Price Prediction (Live ML)

> A Bitcoin-themed Machine Learning application that forecasts prices using historical trends and evaluates model accuracy in real-time.

---

## 📖 Overview

This project is an interactive web dashboard built with Streamlit that analyzes and predicts the future price of Bitcoin (BTC). It dynamically fetches the last three years of live market data from Yahoo Finance and trains a Scikit-Learn Linear Regression model on the fly. Users can review real-time model evaluation metrics, forecast BTC prices up to 15 years into the future, and estimate the projected value of their personal portfolio.

## 📸 Project Screenshots

**1. Dashboard Overview & Live Metrics**
*(Showcases the main dashboard, live pricing, and real-time model evaluation scores like R² and RMSE.)*
<img width="1861" height="770" alt="image" src="https://github.com/user-attachments/assets/0d3f6a6f-c761-42d3-9512-f7cf7ca7872d" />


**2. Price Prediction & Portfolio Calculator**
*(Demonstrates the future date selection and the calculated future portfolio value.)*

<img width="361" height="502" alt="image" src="https://github.com/user-attachments/assets/f65e5272-a072-4745-b21b-a6004236c769" />

<img width="1755" height="435" alt="image" src="https://github.com/user-attachments/assets/e04ada7d-f553-40f1-90a1-991bc5353b8d" />



---

## ✨ Key Features

* **Live Data Ingestion:** Automatically retrieves daily Bitcoin market data over a 3-year period using the `yfinance` API.
* **Dynamic ML Training:** Trains a Linear Regression model upon launch, splitting data for training and testing to ensure valid metric reporting.
* **Real-Time Evaluation:** Displays a clean metrics dashboard featuring Current BTC Price, Latest Market Date, Total Data Points, R² Score, and RMSE.
* **Interactive Forecasting:** Calculates projected prices based on user-selected future dates.
* **Portfolio Calculator:** Computes the future USD value of a user's current BTC holdings based on the predicted price.
* **Custom UI:** Features a highly customized, dark-themed Streamlit interface complete with CSS-injected gradients and a subtle Bitcoin watermark.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (LinearRegression, train_test_split, r2_score, mean_squared_error)
* **Data Processing:** Pandas, NumPy
* **Financial API:** yfinance

---

## 🚀 Getting Started

### Prerequisites

Ensure Python is installed on your system, then install the required dependencies:

```bash
pip install streamlit pandas numpy yfinance scikit-learn
