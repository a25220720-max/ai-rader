import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ⚙️ 작전 설정
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

@st.cache_data(ttl=3600)
def gather_intel_pro(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="5y")
    if hist.empty: return None
    current_price = round(hist['Close'].iloc[-1], 2)
    
    # 지표 계산
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean(); avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    current_rsi = round(100 - (100 / (1 + rs)).iloc[-1], 2)

    # 7일 뉴스 수집
    all_news = ticker.news
    recent_news = [n.get('title') for n in all_news if datetime.fromtimestamp(n.get('providerPublishTime', 0)) >= datetime.now() - timedelta(days=7)]
    if not recent_news: recent_news = [n.get('title') for n in all_news[:5]]

    return {
        "ticker": ticker_symbol.upper(), "current_price": current_price,
        "ma20": round(hist['MA20'].iloc[-1], 2), "ma50": round(hist['MA50'].iloc[-1], 2),
        "rsi": current_rsi, "news": recent_news[:10], "history": hist
    }

def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    # 🔥 프롬프트 대폭 수정: 비관론/낙관론 토론 유도
    prompt = f"""
    [객관성 강화 퀀트 시뮬레이션]
    당신은 '냉철한 비관론자'와 '합리적 낙관론자' 두 명의 분석가입니다. 
    {intel_data['ticker']} 종목에 대해 서로 끝장 토론을 벌인 뒤, 가장 현실적인 예측치를 도출하세요.

    [현재 데이터]: 가격 ${intel_data['current_price']}, RSI {intel_data['rsi']}, MA20 ${intel_data['ma20']}
    [주요 뉴스]: {news_text}

    🚨 [분석 지침]
    1. '비관론자'의 관점에서 주가가 폭락하거나 정체될 수밖에 없는 리스크 3가지를 반드시 제시하세요.
    2. 무조건적인 우상향 그래프는 '오류'로 간주합니다. 경기 침체, 금리, 공급 과잉 등 하락 요인을 반영하세요.
    3. 예측 주가는 과거 5년의 변동 범위 내에서 현실적으로 산출하세요.

    [결과양식]
    ### 🗣️ AI 참모의 끝장 토론 브리핑
    - **낙관적 견해**: (요약)
    - **비관적 리스크**: (반드시 포함할 리스크 요인 2~3가지)
    - **최종 결론**: (두 의견을 종합한 냉정한 요약)

    [PRICE_START]
    1일: 000.00
    1주: 000.00
    2주: 000.00
    1개월: 000.00
    3개월: 000.00
    6개월: 000.00
    1년: 000.00
    2년: 000.00
    3년: 000.00
    [PRICE_END]

    ### 🎯 3년 마스터 투자 전략
    (공격적 매수보다는 '분할 매수' 및 '리스크 관리' 관점에서 가이드)
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e: return f"AI 오류: {e}"

# (이하 UI 및 차트 로직은 동일하되, 차트 색상을 조금 더 차분하게 조정)
st.set_page_config(page_title="AI-Radar Neutral", layout="centered", page_icon="⚖️")
st.title("⚖️ AI-Radar (객관적 토론 에디션)")
st.markdown("단순 예측을 넘어 **비관적 리스크**를 강제로 검토하여 객관성을 확보한 시스템입니다.")

TARGET = st.text_input("🎯 종목 코드를 입력하세요", "TSLA")
if st.button("🚀 냉정한 레이더 가동"):
    with st.spinner("비관론자와 낙관론자가 격렬하게 토론 중입니다..."):
        market_intel = gather_intel_pro(TARGET)
        if market_intel:
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            # (차트 및 보고서 출력 로직 동일)
            start_idx = final_report.find("[PRICE_START]"); end_idx = final_report.find("[PRICE_END]")
            if start_idx != -1 and end_idx != -1:
                price_block = final_report[start_idx + len("[PRICE_START]"):end_idx].strip()
                future_prices = [float(''.join(c for c in line.split(':')[1] if c.isdigit() or c == '.')) for line in price_block.split('\n') if ":" in line]
                if len(future_prices) == 9:
                    hist = market_intel['history']; past_dates = hist.index; past_prices = hist['Close'].values; today = past_dates[-1]
                    daily_vol = hist['Close'].pct_change().std()
                    x_f = [today + pd.Timedelta(days=d) for d in [1, 7, 14, 30, 90, 180, 365, 730, 1095]]
                    all_x = [today] + x_f; all_y = [market_intel['current_price']] + future_prices
                    f_dates = pd.date_range(start=today, end=x_f[-1], freq='B')
                    sim_p = np.interp(mdates.date2num(f_dates), mdates.date2num(all_x), all_y)
                    for i in range(len(all_x)-1):
                        m = (f_dates > all_x[i]) & (f_dates < all_x[i+1]); s = m.sum()
                        if s > 0: sim_p[m] += np.random.normal(0, daily_vol * sim_p[m], s) * np.sin(np.pi * np.arange(1, s + 1) / (s + 1))
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(past_dates, past_prices, color='#5D6D7E', label='History')
                    ax.plot(f_dates, sim_p, color='#CA6F1E', label='Balanced AI Forecast')
                    min_p = min(future_prices); buy_d = x_f[future_prices.index(min_p)]
                    ax.scatter(buy_d, min_p, color='#27AE60', s=200, marker='*', zorder=10, label='Potential Entry')
                    ax.legend(); st.pyplot(fig)
                    st.markdown(final_report[:start_idx].strip() + "\n\n" + final_report[end_idx + len("[PRICE_END]"):].strip())
