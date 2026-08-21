import streamlit as st
import os
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta
st.set_page_config(page_title='FX Insight',page_icon='◈',layout='wide')
st.markdown('''<style>.stApp{background:#0b1020;color:#e8ecf7}.block-container{max-width:1440px;padding:40px 5vw}.brand{font-size:18px;font-weight:800}.mark{background:#6478ff;color:white;border-radius:11px;padding:8px 11px;margin-right:10px}.eyebrow{color:#8490ad;font-size:12px;letter-spacing:1.5px;margin-top:35px}h1{color:#f5f7ff!important}.subtitle,.metric-label{color:#8d98b2}.section-title{font-weight:800;color:#f1f4ff;margin:26px 0 13px}.metric-card{background:#151c31;border:1px solid #27304a;border-radius:18px;padding:20px;min-height:120px}.metric-value{font-size:26px;font-weight:800;color:#f5f7ff}.metric-caption{color:#52d69a;font-size:12px;margin-top:8px}.info-box{background:#182344;color:#bec9ff;border-radius:16px;padding:17px 20px;margin:18px 0;font-size:13px}</style>''',unsafe_allow_html=True)
st.markdown('<div class="brand"><span class="mark">◈</span>FX INSIGHT</div>',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">AI-POWERED CURRENCY DASHBOARD</div>',unsafe_allow_html=True)
st.title('환율 흐름을 한눈에')
st.markdown('<div class="subtitle">PyTorch LSTM 모델이 최근 시장 데이터를 분석해 내일의 환율을 예측합니다.</div>',unsafe_allow_html=True)
st.markdown('<div class="section-title">분석 설정</div>',unsafe_allow_html=True)
a,b,_=st.columns([2,1,3])
with a: currency=st.selectbox('분석 통화',('USD/KRW','JPY/KRW','AUD/KRW'),label_visibility='collapsed')
with b: run=st.button('분석 실행 →',type='primary',use_container_width=True)
if not run: st.markdown('<div class="info-box">통화를 선택하고 <b>분석 실행</b>을 눌러 최신 환율과 AI 예측을 확인하세요.</div>',unsafe_allow_html=True)
else:
    try:
        code=currency.split('/')[0].lower(); 
        backend_url=os.getenv('BACKEND_URL',st.secrets.get('BACKEND_URL','https://exchange-rate-prediction.onrender.com')).rstrip('/'); data=requests.get(f'{backend_url}/predict?currency={code}',timeout=30).json(); curr=data['current_price']; pred=data['predicted_next_price']; status=data['volatility_status']; diff=pred-curr
        tick={'USD/KRW':'USDKRW=X','JPY/KRW':'JPYKRW=X','AUD/KRW':'AUDKRW=X'}; df=yf.download(tick[currency],period='30d',progress=False,auto_adjust=False); close=df['Close']; close=close.iloc[:,0] if len(getattr(close,'shape',()))>1 else close
        st.markdown('<div class="section-title">오늘의 시장 요약</div>',unsafe_allow_html=True); cols=st.columns(3)
        cards=[('현재 환율',f'₩ {curr:,.2f}','실시간 기준'),('내일의 AI 예측',f'₩ {pred:,.2f}',f'{diff:+.2f} ({diff/curr*100:+.2f}%)'),('변동성 상태','주의 필요' if 'Danger' in status else '안정적',status)]
        for col,(label,value,cap) in zip(cols,cards):
            with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-caption">{cap}</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-title">최근 30일 추세와 내일 예측</div>',unsafe_allow_html=True); fig=go.Figure(go.Scatter(x=close.index,y=close.values,name='실제 환율',mode='lines',line=dict(color='#7185ff',width=3))); nxt=close.index[-1]+timedelta(days=1); fig.add_trace(go.Scatter(x=[close.index[-1],nxt],y=[float(close.iloc[-1]),pred],name='AI 예측',mode='lines+markers',line=dict(color='#ff7180',width=3,dash='dot'))); fig.update_layout(template='plotly_dark',height=410,margin=dict(l=10,r=10,t=25,b=10),hovermode='x unified'); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        direction='상승' if diff>0 else '하락' if diff<0 else '보합'; level='높게' if diff>0 else '낮게' if diff<0 else '같게'; risk='변동성 경고 상태라 예측 오차가 커질 수 있습니다.' if 'Danger' in status else '변동성은 정상 범위로 분류되었습니다.'
        st.markdown(f'<div class="info-box"><b>왜 이렇게 예측했나요?</b><br>• 예측 방향은 <b>{direction}</b>이며 현재 환율보다 {abs(diff):,.2f}만큼 {level} 계산되었습니다.<br>• {risk}<br>• 최근 30일 환율 흐름과 RSI·MACD·볼린저 밴드 특징을 LSTM 모델에 입력해 다음 시점의 값을 추정했습니다.<br><small>※ AI 예측은 참고용이며 실제 시장 상황과 다를 수 있습니다.</small></div>',unsafe_allow_html=True)
    except Exception as e: st.error(f'분석 중 오류가 발생했습니다: {e}')
