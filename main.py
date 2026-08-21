from fastapi import FastAPI, HTTPException, Query
import torch
import torch.nn as nn
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device('cpu')
models = {}
scalers_X = {}
scalers_y = {}

# 🚨 [수정됨] 위안화(cny) 제거
SUPPORTED_CURRENCIES = ["usd", "jpy", "aud"]
TICKER_MAP = {
    "usd": "USDKRW=X",
    "jpy": "JPYKRW=X",
    "aud": "AUDKRW=X"
}

print("모든 통화 모델을 메모리에 로드 중...")
for cur in SUPPORTED_CURRENCIES:
    model_path = f"models/{cur}_lstm.pt"
    scaler_x_path = f"models/{cur}_scaler_X.pkl"
    scaler_y_path = f"models/{cur}_scaler_y.pkl"
    
    if os.path.exists(model_path):
        model = TimeSeriesLSTM(input_dim=5, hidden_dim=64, num_layers=2, output_dim=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        models[cur] = model
        scalers_X[cur] = joblib.load(scaler_x_path)
        scalers_y[cur] = joblib.load(scaler_y_path)
        print(f"✅ {cur.upper()} 모델 로드 완료")
    else:
        print(f"⚠️ {cur.upper()} 모델 파일이 없습니다. (경로 확인 필요)")

@app.get("/predict")
def predict_next_price(currency: str = Query("usd", description="통화 코드 (usd, jpy, aud)")):
    currency = currency.lower()
    
    if currency not in models:
        raise HTTPException(status_code=404, detail=f"{currency.upper()} 모델이 로드되지 않았거나 지원되지 않습니다.")
    
    ticker = TICKER_MAP[currency]
    
    df = yf.download(ticker, period='100d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()
    
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df = df.dropna()

    rsi_col = [col for col in df.columns if col.startswith('RSI_')][0]
    macd_col = [col for col in df.columns if col.startswith('MACD_')][0]
    bbl_col = [col for col in df.columns if col.startswith('BBL_')][0]
    bbu_col = [col for col in df.columns if col.startswith('BBU_')][0]
    
    features = ['Close', rsi_col, macd_col, bbl_col, bbu_col]
    recent_data = df[features].tail(30).values
    
    scaler_X_cur = scalers_X[currency]
    scaler_y_cur = scalers_y[currency]
    model_cur = models[currency]
    
    scaled_recent = scaler_X_cur.transform(recent_data)
    input_tensor = torch.FloatTensor(scaled_recent).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_scaled = model_cur(input_tensor).numpy()
    
    predicted_price = scaler_y_cur.inverse_transform(prediction_scaled)[0][0]
    
    current_close = recent_data[-1][0]
    upper_band = recent_data[-1][4]
    lower_band = recent_data[-1][3]
    
    volatility_status = "Normal"
    if predicted_price > upper_band:
        volatility_status = "High Volatility (Overbought Danger)"
    elif predicted_price < lower_band:
        volatility_status = "High Volatility (Oversold Danger)"

    return {
        "currency": currency.upper(),
        "current_price": float(current_close),
        "predicted_next_price": float(predicted_price),
        "volatility_status": volatility_status
    }

# uvicorn main:app --reload