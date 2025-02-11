import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

# Define LSTM Model (Lighter for better performance)
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])

# Fetch OHLC data (180 days)
def fetch_ohlc_data(symbol, days=180):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {
        "vs_currency": "usd",  # Explicitly adding this parameter
        "days": days,
        "interval": "daily"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an error if request fails
        data = response.json()

        if "prices" not in data:
            logging.error(f"Error fetching data: {data}")
            return None

        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None

# Prepare data for training
def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['close']])

    X, y = [], []
    lookback = 20  # Use 20 days of past data for training

    for i in range(len(df_scaled) - lookback):
        X.append(df_scaled[i:i + lookback])
        y.append(df_scaled[i + lookback])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# Train the model
def train_model(X, y, epochs=10, batch_size=16, lr=0.001):
    model = LSTMModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model

# Main Execution
symbol = "bitcoin"  # Change to your desired coin
df = fetch_ohlc_data(symbol, days=180)

if df is not None:
    X, y, scaler = prepare_data(df)
    model = train_model(X, y, epochs=10)  # Training with 10 epochs for accuracy
    torch.save(model.state_dict(), "model.pth")  # Save model
    print("Model trained and saved as model.pth")
