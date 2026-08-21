from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI 환율 인텔리전스 대시보드", layout="wide")

# 상단 헤더 & 디자인
st.markdown("## 📊 AI Global FX Intelligence & Forecast")
st.caption(
    "PyTorch LSTM 시계열 딥러닝 모델 기반 실시간 환율 예측 및 기술적 지표 분석 시스템"
)

# 사이드바 설정
with st.sidebar:
  st.header("⚙️ 분석 설정")
  selected_currency = st.selectbox(
      "통화 선택", ("USD/KRW", "JPY/KRW", "AUD/KRW")
  )
  time_period = st.select_slider(
      "조회 기간", options=["1mo", "3mo", "6mo", "1y"], value="3mo"
  )
  st.divider()
  st.markdown(
      "**모델 아키텍처**  \n- LSTM 2-Layer (Seq: 30)  \n- Feature: Close, RSI,"
      " MACD, BBands"
  )

# 데이터 수집 및 지표 계산
ticker_map = {
    "USD/KRW": "USDKRW=X",
    "JPY/KRW": "JPYKRW=X",
    "AUD/KRW": "AUDKRW=X",
}
ticker = ticker_map[selected_currency]
df = yf.download(ticker, period=time_period, progress=False)
if isinstance(df.columns, pd.MultiIndex):
  df.columns = df.columns.droplevel(1)
df = df.dropna()

# Pandas 기반 보조지표 계산
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-9)
df["RSI"] = 100 - (100 / (1 + rs))

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

sma20 = df["Close"].rolling(window=20).mean()
std20 = df["Close"].rolling(window=20).std()
df["BBL"] = sma20 - (2 * std20)
df["BBU"] = sma20 + (2 * std20)

# 백엔드 API 호출 (실제 Render URL 입력)
API_BASE_URL = "https://your-render-service.onrender.com"
currency_code = selected_currency.split("/")[0].lower()

pred_price, status = None, "대기 중"
try:
  res = requests.get(
      f"{API_BASE_URL}/predict?currency={currency_code}", timeout=10
  )
  if res.status_code == 200:
    data = res.json()
    pred_price = data["predicted_next_price"]
    status = data["volatility_status"]
except Exception:
  pass

# 1. 상단 핵심 지표 카드
curr_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])
day_diff = curr_close - prev_close
day_rate = (day_diff / prev_close) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    f"{selected_currency} 현재가",
    f"₩{curr_close:,.2f}",
    f"{day_diff:+.2f} ({day_rate:+.2f}%)",
)
if pred_price:
  p_diff = pred_price - curr_close
  c2.metric("내일 AI 예측가", f"₩{pred_price:,.2f}", f"{p_diff:+.2f} 원")
else:
  c2.metric("내일 AI 예측가", "서버 응답 대기")
c3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
c4.metric("변동성 리스크", status)

st.divider()

# 2. 전문 금융 멀티 서브플롯 차트 (캔들 + BBands + RSI + MACD)
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=(
        f"{selected_currency} 가격 & AI 예측 밴드",
        "RSI (14)",
        "MACD (12, 26, 9)",
    ),
)

# [Row 1] 캔들스틱 + 볼린저 밴드
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BBU"],
        line=dict(color="rgba(173, 204, 255, 0.4)", dash="dot"),
        name="Upper Band",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BBL"],
        line=dict(color="rgba(173, 204, 255, 0.4)", dash="dot"),
        fill="tonexty",
        fillcolor="rgba(173, 204, 255, 0.05)",
        name="Lower Band",
    ),
    row=1,
    col=1,
)

if pred_price:
  next_date = df.index[-1] + timedelta(days=1)
  fig.add_trace(
      go.Scatter(
          x=[df.index[-1], next_date],
          y=[curr_close, pred_price],
          mode="lines+markers",
          line=dict(color="#FF4B4B", width=3, dash="dash"),
          marker=dict(size=8, color="#FF4B4B"),
          name="AI Forecast",
      ),
      row=1,
      col=1,
  )

# [Row 2] RSI + 과매수/과매도 기준선
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["RSI"],
        line=dict(color="#FFAA00", width=1.5),
        name="RSI",
    ),
    row=2,
    col=1,
)
fig.add_hline(
    y=70,
    line_dash="dot",
    line_color="red",
    row=2,
    col=1,
    annotation_text="Overbought (70)",
)
fig.add_hline(
    y=30,
    line_dash="dot",
    line_color="green",
    row=2,
    col=1,
    annotation_text="Oversold (30)",
)

# [Row 3] MACD & Signal
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MACD"],
        line=dict(color="#2962FF", width=1.5),
        name="MACD",
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Signal"],
        line=dict(color="#E91E63", width=1.5),
        name="Signal",
    ),
    row=3,
    col=1,
)

fig.update_layout(
    height=800,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)