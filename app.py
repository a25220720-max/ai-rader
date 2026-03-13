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
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5y")
        if hist.empty: return None
        current_price = round(hist['Close'].iloc[-1], 2)
        
        # 지표 계산
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean(); avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = round(100 - (100 / (1 + rs)).iloc[-1], 2)

        # 7일 뉴스
        all_news = ticker.news
        recent_news = [n.get('title') for n in all_news if datetime.fromtimestamp(n.get('providerPublishTime', 0)) >= datetime.now() - timedelta(days=7)]
        if not recent_news: recent_news = [n.get('title') for n in all_news[:5]]

        return {
            "ticker": ticker_symbol.upper(), "current_price": current_price,
            "ma20": round(hist['MA20'].iloc[-1], 2), "rsi": rsi,
            "news": recent_news[:10], "history": hist
        }
    except: return None

def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    # 🔥 프롬프트를 더 단순하고 강력하게 수정 (에러 방지용)
    prompt = f"""
    [가상 퀀트 분석 모드]
    당신은 냉철한 분석가입니다. {intel_data['ticker']} 종목에 대해 비관적 리스크와 낙관적 기회를 분석하세요.
    현재가: ${intel_data['current_price']}, RSI: {intel_data['rsi']}
    
    필수 조건:
    1. 주가가 정체되거나 하락할 리스크 3가지를 반드시 포함하세요.
    2. 아래 양식을 절대 어기지 마세요.

    [결과양식]
    ### 🗣️ AI 참모의 끝장 토론 브리핑
    - 낙관: (요약)
    - 비관: (리스크 3개)
    - 최종 결론: (한줄 요약)

    [PRICE_START]
    1일: 000
    1주: 000
    2주: 000
    1개월: 000
    3개월: 000
    6개월: 000
    1년: 000
    2년: 000
    3년: 000
    [PRICE_END]
    """
    try:
        # 모델을 최신형 2.5-flash로 고정
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        if not response.text: return "AI가 응답을 거부했습니다."
        return response.text
    except Exception as e: return f"AI 통신 오류: {e}"

st.set_page_config(page_title="AI-Radar Balanced", layout="centered", page_icon="⚖️")
st.title("⚖️ AI-Radar (객관적 토론 에디션)")

TARGET = st.text_input("🎯 종목 코드를 입력하세요", "TSLA")

if st.button("🚀 냉정한 레이더 가동"):
    # 버튼을 누르는 순간 캐시를 초기화하지 않고 새로 고침 유도
    with st.spinner("데이터 분석 및 AI 토론 중..."):
        market_intel = gather_intel_pro(TARGET)
        if not market_intel:
            st.error("데이터 수집 실패. 종목명을 다시 확인하세요.")
        else:
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            
            if "AI 통신 오류" in final_report:
                st.error("🚨 구글 서버 과부하! 잠시 후 다시 시도해주세요.")
            else:
                start_idx = final_report.find("[PRICE_START]"); end_idx = final_report.find("[PRICE_END]")
                if start_idx != -1 and end_idx != -1:
                    try:
                        price_block = final_report[start_idx + 13:end_idx].strip()
                        future_prices = [float(''.join(c for c in line.split(':')[1] if c.isdigit() or c == '.')) for line in price_block.split('\n') if ":" in line]
                        
                        # 차트 그리기
                        hist = market_intel['history']; past_dates = hist.index; past_prices = hist['Close'].values; today = past_dates[-1]
                        x_f = [today + pd.Timedelta(days=d) for d in [1, 7, 14, 30, 90, 180, 365, 730, 1095]]
                        all_x = [today] + x_f; all_y = [market_intel['current_price']] + future_prices
                        f_dates = pd.date_range(start=today, end=x_f[-1], freq='B')
                        sim_p = np.interp(mdates.date2num(f_dates), mdates.date2num(all_x), all_y)
                        
                        vol = hist['Close'].pct_change().std()
                        for i in range(len(all_x)-1):
                            m = (f_dates > all_x[i]) & (f_dates < all_x[i+1]); s = m.sum()
                            if s > 0: sim_p[m] += np.random.normal(0, vol * sim_p[m], s) * np.sin(np.pi * np.arange(1, s + 1) / (s + 1))
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(past_dates, past_prices, color='#5D6D7E', label='History')
                        ax.plot(f_dates, sim_p, color='#CA6F1E', label='Balanced Forecast')
                        ax.legend(); st.pyplot(fig)
                        st.markdown(final_report[:start_idx].strip() + "\n\n" + final_report[end_idx + 11:].strip())
                    except: st.markdown(final_report)
                else: st.markdown(final_report)
