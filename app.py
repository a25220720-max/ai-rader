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
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean(); avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = round(100 - (100 / (1 + rs)).iloc[-1], 2)

        all_news = ticker.news
        recent_news = [n.get('title') for n in all_news if datetime.fromtimestamp(n.get('providerPublishTime', 0)) >= datetime.now() - timedelta(days=7)]
        if not recent_news: recent_news = [n.get('title') for n in all_news[:5]]

        return {
            "ticker": ticker_symbol.upper(), "current_price": current_price,
            "ma20": round(hist['MA20'].iloc[-1], 2), "ma50": round(hist['MA50'].iloc[-1], 2),
            "rsi": rsi, "news": recent_news[:10], "history": hist
        }
    except: return None

def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    prompt = f"""
    당신은 '비관론자'와 '낙관론자'입니다. {intel_data['ticker']} 종목에 대해 토론하세요.
    가격 ${intel_data['current_price']}, RSI {intel_data['rsi']}
    뉴스: {news_text}

    [결과양식]
    ### 🗣️ AI 참모의 끝장 토론 브리핑
    - 낙관: 내용
    - 비관: 리스크 3가지 필수
    - 결론: 냉정한 요약

    [PRICE_START]
    1일: 00.0
    1주: 00.0
    2주: 00.0
    1개월: 00.0
    3개월: 00.0
    6개월: 00.0
    1년: 00.0
    2년: 00.0
    3년: 00.0
    [PRICE_END]
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e: return f"AI 통신 오류: {e}"

st.set_page_config(page_title="AI-Radar Final", layout="centered", page_icon="⚖️")
st.title("⚖️ AI-Radar (객관적 토론 에디션)")

TARGET = st.text_input("🎯 종목 코드를 입력하세요", "TSLA")

if st.button("🚀 냉정한 레이더 가동"):
    with st.spinner("데이터 분석 및 AI 토론 중..."):
        market_intel = gather_intel_pro(TARGET)
        if not market_intel:
            st.error("데이터를 가져오지 못했습니다. 종목명을 확인하세요.")
        else:
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            
            if "AI 통신 오류" in final_report:
                st.warning("구글 서버가 과부하 상태입니다. 1분만 기다렸다가 다시 눌러주세요!")
            else:
                start_idx = final_report.find("[PRICE_START]"); end_idx = final_report.find("[PRICE_END]")
                if start_idx != -1 and end_idx != -1:
                    price_block = final_report[start_idx + 13:end_idx].strip()
                    try:
                        future_prices = [float(''.join(c for c in line.split(':')[1] if c.isdigit() or c == '.')) for line in price_block.split('\n') if ":" in line]
                        
                        hist = market_intel['history']; past_dates = hist.index; past_prices = hist['Close'].values; today = past_dates[-1]
                        x_f = [today + pd.Timedelta(days=d) for d in [1, 7, 14, 30, 90, 180, 365, 730, 1095]]
                        all_x = [today] + x_f; all_y = [market_intel['current_price']] + future_prices
                        f_dates = pd.date_range(start=today, end=x_f[-1], freq='B')
                        sim_p = np.interp(mdates.date2num(f_dates), mdates.date2num(all_x), all_y)
                        
                        daily_vol = hist['Close'].pct_change().std()
                        for i in range(len(all_x)-1):
                            m = (f_dates > all_x[i]) & (f_dates < all_x[i+1]); s = m.sum()
                            if s > 0: sim_p[m] += np.random.normal(0, daily_vol * sim_p[m], s) * np.sin(np.pi * np.arange(1, s + 1) / (s + 1))
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(past_dates, past_prices, color='#5D6D7E', label='History')
                        ax.plot(f_dates, sim_p, color='#CA6F1E', label='Forecast')
                        ax.legend(); st.pyplot(fig)
                        st.markdown(final_report[:start_idx].strip() + "\n\n" + final_report[end_idx + 11:].strip())
                    except: st.markdown(final_report)
                else: st.markdown(final_report)
