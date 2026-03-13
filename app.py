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
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = round(100 - (100 / (1 + gain/loss)).iloc[-1], 2)

        # 7일 뉴스 수집
        all_news = ticker.news
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_news = [n.get('title') for n in all_news if datetime.fromtimestamp(n.get('providerPublishTime', 0)) >= seven_days_ago]
        if not recent_news: recent_news = [n.get('title') for n in all_news[:10]]

        return {
            "ticker": ticker_symbol.upper(), "current_price": current_price,
            "ma20": round(hist['MA20'].iloc[-1], 2), "rsi": rsi, 
            "news": recent_news[:10], "history": hist
        }
    except: return None

def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    prompt = f"""
    당신은 '냉철한 비관론자'와 '합리적 낙관론자'입니다. 
    {intel_data['ticker']}에 대해 끝장 토론을 벌여 현실적인 예측치를 내놓으세요.
    가격 ${intel_data['current_price']}, RSI {intel_data['rsi']}
    주요 뉴스: {news_text}

    🚨 [지시] 하락 리스크 3가지를 반드시 포함하고, 무조건적인 우상향은 배제하세요.

    [결과양식]
    ### 🗣️ AI 참모의 끝장 토론 브리핑
    - **낙관적 견해**: 내용
    - **비관적 리스크**: 리스크 3가지 필수
    - **최종 결론**: 냉정한 요약

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

    ### 🎯 3년 마스터 투자 전략
    (리스크 관리를 포함한 가이드)
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "OVERLOAD"
        return f"AI 오류: {e}"

st.set_page_config(page_title="AI-Radar Neutral", layout="centered", page_icon="⚖️")
st.title("⚖️ AI-Radar (객관적 토론 에디션)")

TARGET = st.text_input("🎯 종목 코드를 입력하세요", "TSLA").upper()

if st.button("🚀 냉정한 레이더 가동"):
    with st.spinner("비관론자와 낙관론자가 격렬하게 토론 중입니다..."):
        market_intel = gather_intel_pro(TARGET)
        if not market_intel:
            st.error("데이터 수집 실패. 종목명을 확인하세요.")
        else:
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            
            if final_report == "OVERLOAD":
                st.warning("⚠️ 서버 과열! 1분만 기다렸다가 다시 눌러주세요.")
            elif "AI 오류" in final_report:
                st.error("API 키 문제나 서버 에러가 발생했습니다. 키 설정을 다시 확인해주세요.")
            else:
                start_idx = final_report.find("[PRICE_START]"); end_idx = final_report.find("[PRICE_END]")
                if start_idx != -1 and end_idx != -1:
                    try:
                        p_txt = final_report[start_idx + 13:end_idx].strip()
                        future_prices = [float(''.join(c for c in line.split(':')[1] if c.isdigit() or c == '.')) for line in p_txt.split('\n') if ":" in line]
                        
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
                        ax.plot(f_dates, sim_p, color='#CA6F1E', label='Forecast')
                        
                        min_p = min(future_prices); buy_d = x_f[future_prices.index(min_p)]
                        ax.scatter(buy_d, min_p, color='#27AE60', s=250, marker='*', zorder=10, label='BUY')
                        
                        ax.legend(); st.pyplot(fig)
                        st.markdown(final_report[:start_idx].strip() + "\n\n" + final_report[end_idx + 11:].strip())
                    except: st.markdown(final_report)
                else: st.markdown(final_report)
