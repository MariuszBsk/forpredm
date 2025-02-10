import os
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, jsonify, request
from sklearn.preprocessing import MinMaxScaler
import datetime
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Map Binance-style symbols to CoinGecko coin IDs
def get_coingecko_id(symbol):
    """
    Convert Binance-style symbols (BTCUSDT) to CoinGecko's coin IDs (bitcoin).
    """
    coingecko_mapping = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
        "SOL": "solana",
        "DOT": "polkadot",
        "DOGE": "dogecoin",
        "MATIC": "polygon",
        "LTC": "litecoin",
        "TRX": "tron",
        "LINK": "chainlink",
        "AVAX": "avalanche-2"
    }

    symbol = symbol.upper().replace("USDT", "").replace("USD", "")
    return coingecko_mapping.get(symbol, None)

# Fetching OHLC data from CoinGecko API
def fetch_ohlc_data(symbol, days=180):
    coingecko_id = get_coingecko_id(symbol)
    
    if not coingecko_id:
        logging.error(f"Coin not found: {symbol}")
        return None

    url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "prices" not in data:
            logging.error(f"Error fetching data: {data}")
            return None

        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    except Exception as e:
        logging.error(f"Request failed: {e}")
        return None

# Prepare data for model
def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['close']])

    X, y = [], []
    lookback = 20
    for i in range(len(df_scaled) - lookback):
        X.append(df_scaled[i:i + lookback])
        y.append(df_scaled[i + lookback])

    X, y = np.array(X), np.array(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])

# Train the model
def train_model(model, X, y, epochs=20, batch_size=16, lr=0.001):  # Reduced epochs to prevent timeouts
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    batch_size = min(batch_size, X.shape[0])

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Predict future prices
def predict_future(model, X, scaler):
    model.eval()
    future_inputs = X[-1].unsqueeze(0)
    predicted_prices = []

    for _ in range(180):  # 6 months (approx. 180 days)
        pred = model(future_inputs).item()
        predicted_prices.append(pred)
        future_inputs = torch.cat((future_inputs[:, 1:, :], torch.tensor([[[pred]]], dtype=torch.float32)), dim=1)

    return scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_prices', methods=['GET'])
def fetch_prices():
    symbol = request.args.get('symbol')
    df = fetch_ohlc_data(symbol)
    if df is not None:
        times = df.index.strftime('%Y-%m-%d').tolist()
        prices = df['close'].tolist()
        return jsonify({'times': times, 'prices': prices})
    else:
        return jsonify({'error': 'Failed to fetch data'}), 400

@app.route('/predict_prices', methods=['GET'])
def predict_prices():
    symbol = request.args.get('symbol')
    df = fetch_ohlc_data(symbol)
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = LSTMModel()
        train_model(model, X, y)
        predicted_prices = predict_future(model, X, scaler)

        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=180, freq='D')
        predicted_df = pd.DataFrame(predicted_prices, columns=["Predicted"], index=future_dates)

        times = predicted_df.index.strftime('%Y-%m-%d').tolist()
        predicted_prices = predicted_df['Predicted'].tolist()

        return jsonify({'times': times, 'predicted_prices': predicted_prices})
    else:
        return jsonify({'error': 'Failed to fetch data for prediction'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
