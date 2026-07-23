import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import timedelta

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Bitcoin Price Prediction (Live ML)",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL STYLING (BITCOIN THEME)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1a1a1a, #0b0b0b 60%);
    color: #f5f5f5;
}
.stApp::before {
    content: "₿";
    position: fixed;
    font-size: 400px;
    font-weight: 900;
    color: rgba(255, 165, 0, 0.04);
    top: 10%;
    left: 50%;
    transform: translateX(-50%);
    z-index: 0;
}
h1, h2, h3 {
    color: #ffa500;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1f1f1f, #2a2a2a);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 0 18px rgba(255, 165, 0, 0.15);
}
.stButton > button {
    background: linear-gradient(135deg, #ff9800, #ff5722);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📈 Bitcoin Price Prediction (Live ML)")
st.write(
    "A Bitcoin-themed Machine Learning app that predicts prices "
    "using historical trends and evaluates model accuracy."
)

st.markdown("---")

# --------------------------------------------------
# FETCH BITCOIN DATA
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_bitcoin_data():
    btc = yf.download(
        "BTC-USD",
        period="3y",
        interval="1d",
        progress=False
    )

    if btc.empty:
        return None

    btc.reset_index(inplace=True)
    btc = btc[["Date", "Close"]]
    btc["Date"] = pd.to_datetime(btc["Date"])
    return btc

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = fetch_bitcoin_data()
if df is None or df.empty:
    st.error("Failed to fetch Bitcoin data from Yahoo Finance.")
    st.stop()

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
df["Timestamp"] = df["Date"].apply(lambda x: x.timestamp())
X = df[["Timestamp"]]
y = df["Close"]

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# MODEL EVALUATION (NEW)
# --------------------------------------------------
y_pred = model.predict(X_test).flatten()
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------
latest_price = float(df["Close"].iloc[-1].iloc[0])
latest_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("💰 Current BTC Price", f"${latest_price:,.2f}")
with c2:
    st.metric("📅 Latest Market Date", latest_date)
with c3:
    st.metric("📊 Data Points", len(df))
with c4:
    st.metric("📐 R² Score", f"{r2:.3f}")
with c5:
    st.metric("📉 RMSE (USD)", f"${rmse:,.2f}")

st.markdown("---")

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
st.subheader("🔮 Predict Bitcoin Price")

left, right = st.columns([2, 1])

with left:
    min_date = df["Date"].iloc[0].date()
    future_limit = (df["Date"].iloc[-1] + timedelta(days=15 * 365)).date()

    input_date = st.date_input(
        "Select a date",
        value=df["Date"].iloc[-1].date(),
        min_value=min_date,
        max_value=future_limit
    )

with right:
    btc_amount = st.number_input(
        "Bitcoin owned (BTC)",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🚀 Predict Price", use_container_width=True):
    timestamp = pd.to_datetime(input_date).timestamp()
    prediction = float(model.predict([[timestamp]]).flatten()[0])

    st.markdown("---")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.success(
            f"📅 **Prediction for {input_date}**\n\n"
            f"💰 **${prediction:,.2f} per BTC**"
        )

    with p2:
        total_value = btc_amount * prediction
        st.info(
            f"🪙 **Value of {btc_amount} BTC**\n\n"
            f"💵 **${total_value:,.2f}**"
        )

    with p3:
        st.warning(
            f"📐 **R² Score:** {r2:.3f}\n\n"
            f"📉 **RMSE:** ${rmse:,.2f}"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ Educational ML project. Linear Regression captures long-term trends "
    "but does not model Bitcoin’s volatility. Not financial advice."
)
