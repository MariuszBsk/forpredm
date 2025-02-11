import os
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, jsonify, request
from sklearn.preprocessing import MinMaxScaler
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])

# Load the trained model with no_grad to save memory
def load_trained_model():
    model = LSTMModel()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Mapping common trading pairs to CoinGecko coin IDs
SYMBOL_MAPPING = {
    "BTCUSD": "bitcoin",
    "ETHUSD": "ethereum",
    "DOGEUSD": "dogecoin",
    "XRPUSD": "ripple",
    "ADAUSD": "cardano",
    "SOLUSD": "solana",
    "DOTUSD": "polkadot",
    "LTCUSD": "litecoin",
    "BCHUSD": "bitcoin-cash",
    "BNBUSD": "binancecoin",
    "AVAXUSD": "avalanche-2",
    "MATICUSD": "matic-network",
    "SHIBUSD": "shiba-inu",
    "LINKUSD": "chainlink",
    "UNIUSD": "uniswap",
    "XLMUSD": "stellar"
}

# Fetch OHLC data from CoinGecko (Last 60 days)
def fetch_ohlc_data(symbol, days=60):
    symbol = SYMBOL_MAPPING.get(symbol.upper(), symbol.lower())  # Convert symbol to CoinGecko ID
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "prices" not in data:
            logging.error(f"Error fetching data for {symbol}: {data}")
            return None

        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    except Exception as e:
        logging.error(f"Request failed: {e}")
        return None

# Prepare data for prediction (Using 60 days of past data)
def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['close']])

    X, _ = [], []
    lookback = 20  # Using 20 days for training

    for i in range(len(df_scaled) - lookback):
        X.append(df_scaled[i:i + lookback])

    return torch.tensor(X, dtype=torch.float32), scaler

# Predict next 30 days (Optimized for memory)
def predict_future(model, X, scaler):
    model.eval()
    future_inputs = X[-1].unsqueeze(0)  # Start from last available data
    predicted_prices = []

    for _ in range(30):  # Predict for 30 days
        with torch.no_grad():  # Prevents unnecessary memory usage
            pred = model(future_inputs).item()
        predicted_prices.append(pred)
        future_inputs = torch.cat((future_inputs[:, 1:, :], torch.tensor([[[pred]]], dtype=torch.float32)), dim=1)

    return scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_prices', methods=['GET'])
def fetch_prices():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400

    df = fetch_ohlc_data(symbol)
    if df is not None:
        times = df.index.strftime('%Y-%m-%d').tolist()
        prices = df['close'].tolist()
        return jsonify({'times': times, 'prices': prices})
    else:
        return jsonify({'error': f'Failed to fetch data for symbol: {symbol}'}), 400

@app.route('/predict_prices', methods=['GET'])
def predict_prices():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400

    df = fetch_ohlc_data(symbol)
    if df is not None:
        X, scaler = prepare_data(df)
        model = load_trained_model()
        predicted_prices = predict_future(model, X, scaler)

        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        predicted_df = pd.DataFrame(predicted_prices, columns=["Predicted"], index=future_dates)

        times = predicted_df.index.strftime('%Y-%m-%d').tolist()
        predicted_prices = predicted_df['Predicted'].tolist()

        return jsonify({'times': times, 'predicted_prices': predicted_prices})
    else:
        return jsonify({'error': f'Failed to fetch data for prediction of {symbol}'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
