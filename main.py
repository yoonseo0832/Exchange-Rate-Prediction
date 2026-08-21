from fastapi import FastAPI, HTTPException, Query
import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()

# 1. PyTorch LSTM 모델 클래스 정의
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

SUPPORTED_CURRENCIES = ["usd", "jpy", "aud"]
TICKER_MAP = {
    "usd": "USDKRW=X",
    "jpy": "JPYKRW=X",
    "aud": "AUDKRW=X"
}

# 2. 서버 시작 시 모델 및 스케일러 메모리 로드
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
        print(f"⚠️ {cur.upper()} 모델 파일이 없습니다: {model_path}")

# 3. 실시간 추론 엔드포인트
@app.get("/predict")
def predict_next_price(currency: str = Query("usd", description="통화 코드 (usd, jpy, aud)")):
    currency = currency.lower()
    
    if currency not in models:
        raise HTTPException(status_code=404, detail=f"{currency.upper()} 모델이 로드되지 않았거나 지원되지 않습니다.")
    
    ticker = TICKER_MAP[currency]
    
    # 최근 100일 데이터 수집
    df = yf.download(ticker, period='100d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()
    
    # -----------------------------------
    # 순수 Pandas 기반 보조지표 계산 (외부 라이브러리 미사용)
    # -----------------------------------
    # 1. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 2. MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema12 - ema26

    # 3. 볼린저 밴드 (20, 2)
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BBL_20_2.0'] = sma20 - (2 * std20)
    df['BBU_20_2.0'] = sma20 + (2 * std20)

    df = df.dropna()

    # 입력 피처 추출 (과거 30일)
    features = ['Close', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0']
    recent_data = df[features].tail(30).values
    
    # 통화별 전용 스케일러 및 모델로 추론
    scaler_X_cur = scalers_X[currency]
    scaler_y_cur = scalers_y[currency]
    model_cur = models[currency]
    
    scaled_recent = scaler_X_cur.transform(recent_data)
    input_tensor = torch.FloatTensor(scaled_recent).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_scaled = model_cur(input_tensor).numpy()
    
    predicted_price = scaler_y_cur.inverse_transform(prediction_scaled)[0][0]
    
    # 변동성 상태 판별
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