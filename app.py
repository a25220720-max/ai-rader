import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ⚙️ 작전 설정 (비밀 금고에서 안전하게 키를 불러옵니다)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# 📡 정보 수집조 (🔥 1시간 주기 실시간 & 7일 백업 레이더)
@st.cache_data(ttl=3600)
def gather_intel_pro(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="5y")
    if hist.empty:
        return None

    current_price = round(hist['Close'].iloc[-1], 2)
    
    # 기술적 지표 계산
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    ma20 = round(hist['MA20'].iloc[-1], 2) if not pd.isna(hist['MA20'].iloc[-1]) else "데이터 부족"
    ma50 = round(hist['MA50'].iloc[-1], 2) if not pd.isna(hist['MA50'].iloc[-1]) else "데이터 부족"

    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else "데이터 부족"

    # 🔥 뉴스 수집 로직 강화 (최근 7일 데이터 확보)
    all_news = ticker.news
    recent_news = []
    
    # 현재 시간 기준 7일 전 타임스탬프 계산
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    for news in all_news:
        pub_time = datetime.fromtimestamp(news.get('providerPublishTime', 0))
        if pub_time >= seven_days_ago:
            recent_news.append(news.get('title', '제목 없음'))
    
    # 만약 뉴스가 하나도 없다면 '정보 없음' 메시지 대신 가용한 뉴스 중 가장 최근 것 유지
    if not recent_news:
        recent_news = [news.get('title', '제목 없음') for news in all_news[:5]]
        news_mode = "최근 기록 기반(7일 내 뉴스 없음)"
    else:
        news_mode = "실시간 및 7일 내 주요 소식"

    return {
        "ticker": ticker_symbol.upper(),
        "current_price": current_price,
        "ma20": ma20,
        "ma50": ma50,
        "rsi": current_rsi,
        "news": recent_news[:10], # 최대 10개
        "news_mode": news_mode,
        "history": hist
    }

# 🧠 월스트리트 AI 두뇌
def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    prompt = f"""
    [가상 퀀트 시뮬레이션 모드]
    당신은 월스트리트의 수석 분석가입니다. 아래 데이터를 바탕으로 주가를 예측하세요.
    
    [종목]: {intel_data['ticker']} / [현재가]: ${intel_data['current_price']}
    [기술적 지표]: 20일선 ${intel_data['ma20']}, 50일선 ${intel_data['ma50']}, RSI {intel_data['rsi']}
    [정보 모드]: {intel_data['news_mode']}
    [수집된 주요 뉴스]:
    {news_text}

    🚨 [작전 지시]
    1. 만약 뉴스가 7일 이내의 것이라면 '실시간 모멘텀'을 강조하세요.
    2. 뉴스가 부족하다면 지난 7일간의 시장 흐름을 바탕으로 가장 중요한 이벤트를 유추해 브리핑하세요.
    
    [결과양식]
    ### 🗣️ AI 참모의 상황 분석 보고 (뉴스 갱신 주기: 1H)
    (수집된 정보를 바탕으로 현재 상황을 명확히 브리핑)

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

    ### 🎯 향후 3년 마스터 투자 전략
    (비중 조절 및 최적의 매수 타이밍 상세 가이드)

    ### 📋 AI 심층 보고서
    **[🚀 단기 전망]**: (모멘텀 상세 분석)
    **[🗓️ 중장기 전망]**: (실적 및 추세 분석)
    **[📅 초장기 전망]**: (기업 가치 및 비전 제시)
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"AI 오류: {e}"

# 📱 앱 화면 구성
st.set_page_config(page_title="AI-Radar 1H & 7D", layout="centered", page_icon="📡")
st.title("📡 AI-Radar (실시간 & 7일 분석 에디션)")

st.markdown(
    "**1시간마다 정보를 갱신**하며, 최신 뉴스가 없을 경우 "
    "**최근 7일간의 주요 데이터**를 분석하여 브리핑을 생성합니다."
)

TARGET = st.text_input("🎯 종목 코드를 입력하세요 (예: TSLA, NVDA)", "TSLA")

if st.button("🚀 레이더 가동"):
    with st.spinner("최신 뉴스 및 지난 7일간의 작전 기록을 스캔 중..."):
        market_intel = gather_intel_pro(TARGET)
        
        if not market_intel:
            st.error("데이터를 불러오지 못했습니다.")
        else:
            st.success(f"[{market_intel['ticker']}] 정보 수집 완료 (모드: {market_intel['news_mode']})")
            
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            
            if "AI 오류:" in final_report:
                st.error("🚨 통신 에러 발생")
                st.code(final_report)
            else:
                start_idx = final_report.find("[PRICE_START]")
                end_idx = final_report.find("[PRICE_END]")
                
                if start_idx == -1 or end_idx == -1:
                    st.markdown(final_report)
                else:
                    # 차트 및 보고서 출력 (기존 로직 유지)
                    price_block = final_report[start_idx + len("[PRICE_START]"):end_idx].strip()
                    lines = price_block.split('\n')
                    future_prices = []
                    future_labels = []
                    for line in lines:
                        if ":" in line:
                            parts = line.split(':')
                            future_labels.append(parts[0].strip())
                            val = ''.join(c for c in parts[1] if c.isdigit() or c == '.')
                            if val: future_prices.append(float(val))
                    
                    if len(future_prices) == 9:
                        hist = market_intel['history']
                        past_dates = hist.index
                        past_prices = hist['Close'].values
                        current_price = market_intel['current_price']
                        today = past_dates[-1]
                        daily_volatility = hist['Close'].pct_change().std()
                        x_future_dates = [today + pd.Timedelta(days=d) for d in [1, 7, 14, 30, 90, 180, 365, 730, 1095]]
                        
                        all_x_dates = [today] + x_future_dates
                        all_y_prices = [current_price] + future_prices
                        future_daily_dates = pd.date_range(start=today, end=x_future_dates[-1], freq='B')
                        x_numeric = mdates.date2num(all_x_dates)
                        future_numeric = mdates.date2num(future_daily_dates)
                        base_trend = np.interp(future_numeric, x_numeric, all_y_prices)
                        
                        simulated_prices = np.copy(base_trend)
                        for i in range(len(all_x_dates) - 1):
                            mask = (future_daily_dates > all_x_dates[i]) & (future_daily_dates < all_x_dates[i+1])
                            seg_len = mask.sum()
                            if seg_len > 0:
                                noise = np.random.normal(0, daily_volatility * base_trend[mask], seg_len)
                                window = np.sin(np.pi * np.arange(1, seg_len + 1) / (seg_len + 1))
                                simulated_prices[mask] += noise * window
                                
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(past_dates, past_prices, color='#2E86C1', label='History', linewidth=1)
                        ax.plot(future_daily_dates, simulated_prices, color='#F39C12', linewidth=1.2, label='AI Simulated')
                        
                        min_price = min(future_prices)
                        buy_date = x_future_dates[future_prices.index(min_price)]
                        ax.scatter(buy_date, min_price, color='#00FF00', s=300, marker='*', zorder=10, label='BUY')
                        
                        ax.axvline(x=today, color='gray', linestyle=':', alpha=0.5)
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                        plt.xticks(rotation=45)
                        ax.legend(loc='upper left')
                        st.pyplot(fig)
                        
                        st.markdown(final_report[:start_idx].strip() + "\n\n" + final_report[end_idx + len("[PRICE_END]"):].strip())
