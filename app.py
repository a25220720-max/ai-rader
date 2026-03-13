import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ⚙️ 작전 설정 (비밀 금고에서 안전하게 키를 불러옵니다)
GEMINI_API_KEY = st.secrets["AIzaSyDEHcmpVBgZrIiz0y2pTEzMeyER4kN4Qxs"]

# 📡 정보 수집조 (심층 데이터 스캐너)
def gather_intel_pro(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="5y")
    if hist.empty:
        return None

    current_price = round(hist['Close'].iloc[-1], 2)
    
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

    news_items = ticker.news[:5]
    news_headlines = [news.get('title', '제목 없음') for news in news_items]

    return {
        "ticker": ticker_symbol.upper(),
        "current_price": current_price,
        "ma20": ma20,
        "ma50": ma50,
        "rsi": current_rsi,
        "news": news_headlines,
        "history": hist
    }

# 🧠 월스트리트 AI 두뇌 (Gemini 2.5 Pro 최신형 장착!)
def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    prompt = f"""
    [이것은 실제 금융 조언이 아닌, 학술적 목적의 가상 퀀트 시뮬레이션입니다.]
    당신은 가상의 월스트리트 가치 투자자이자 퀀트 애널리스트입니다. 
    아래 데이터를 바탕으로 해당 기업의 '단기(1일~30일)' 및 '중장기(3개월~3년)' 가상 주가 흐름을 예측하세요.
    
    [종목]: {intel_data['ticker']} / [현재가]: ${intel_data['current_price']}
    [기술적 지표]: 20일선 ${intel_data['ma20']}, 50일선 ${intel_data['ma50']}, RSI {intel_data['rsi']}
    [최신 주요 뉴스]: {news_text}

    🚨 [매우 중요] 반드시 아래의 마크다운 양식을 100% 똑같이 지켜서 답변하세요. 가격은 숫자만 정확히 적으세요.

    [결과양식]
    ### 🗣️ AI 참모의 현재 상황 요약
    (현재 상황에 대한 핵심 브리핑 1~2줄)

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

    ### 🎯 향후 3년 마스터 투자 전략 (로드맵)
    (앞으로 3년간 비중을 어떻게 조절하고 언제가 최적의 매수 타이밍인지, 장기 홀딩 전략은 어떻게 가져가야 하는지 펀드매니저의 시각에서 상세히 가이드)

    ### 📋 AI 심층 투자 보고서
    **[🚀 1일~30일 초단기/단기 전망]**: (시장 모멘텀, 기술적 지표, 뉴스의 단기적 영향을 아주 상세하게 분석)
    **[🗓️ 3개월~1년 중장기 전망]**: (거시 경제 흐름, 실적 전망, 경쟁 지위 변화를 기반으로 한 추세 분석)
    **[📅 2년~3년 초장기 전망]**: (기업의 본질적 가치, 산업의 구조적 성장성, 해자(Moat) 기반 비전 제시)
    """
    try:
        # 🔥 여기서 오류가 났던 엔진 이름을 최신형 2.5 Pro로 완벽하게 교체했습니다!
        response = client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
        return response.text
    except Exception as e:
        return f"AI 오류: {e}"

# 📱 앱 화면(UI) 구성
st.set_page_config(page_title="AI-Radar Pro Max", layout="centered", page_icon="🦅")
st.title("🦅 AI-Radar (2.5 Pro 최신형 에디션)")
st.markdown("**가장 강력한 Gemini 2.5 Pro 엔진**이 탑재되었습니다. 과거 변동성을 시뮬레이션하여 가장 저렴한 매수 타점을 포착합니다.")

TARGET = st.text_input("🎯 타겟 종목 코드를 입력하세요 (예: TSLA, AAPL, NVDA)", "TSLA")

if st.button("🚀 레이더 가동 (최신형 Pro 엔진)"):
    with st.spinner("최신형 AI가 과거 변동성 추출 및 미래 3년 시뮬레이션 중입니다. (약 15초 소요)..."):
        market_intel = gather_intel_pro(TARGET)
        
        if not market_intel:
            st.error("데이터를 불러오지 못했습니다. 종목 코드를 확인해 주세요.")
        else:
            st.success(f"[{market_intel['ticker']}] 전장 데이터 수집 완료! 작전 지도를 그립니다.")
            
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            
            # 🔥 [안전망 1] AI가 에러를 뿜었는지 확인
            if "AI 오류:" in final_report:
                st.error("🚨 통신 에러가 발생했습니다. 아래 내용을 확인해 주세요.")
                st.code(final_report)
            else:
                start_idx = final_report.find("[PRICE_START]")
                end_idx = final_report.find("[PRICE_END]")
                
                # 🔥 [안전망 2] AI가 양식을 어겼을 때
                if start_idx == -1 or end_idx == -1:
                    st.warning("⚠️ AI가 가격표 양식을 어겨 차트를 그릴 수 없습니다. 대신 AI의 원본 보고서를 그대로 출력합니다!")
                    st.markdown(final_report)
                else:
                    price_block = final_report[start_idx + len("[PRICE_START]"):end_idx].strip()
                    lines = price_block.split('\n')
                    future_prices = []
                    future_labels = []
                    
                    for line in lines:
                        if ":" in line:
                            parts = line.split(':')
                            future_labels.append(parts[0].strip())
                            val = ''.join(c for c in parts[1] if c.isdigit() or c == '.')
                            if val:
                                future_prices.append(float(val))
                    
                    if len(future_prices) == 9:
                        st.subheader("🗺️ 3년 장기 예상 작전 지도 (최적 매수 타점 포착)")
                        
                        hist = market_intel['history']
                        past_dates = hist.index
                        past_prices = hist['Close'].values
                        current_price = market_intel['current_price']
                        today = past_dates[-1]
                        
                        daily_volatility = hist['Close'].pct_change().std()
                        
                        x_future_dates = [
                            today + pd.Timedelta(days=1), today + pd.Timedelta(weeks=1), today + pd.Timedelta(weeks=2),
                            today + pd.Timedelta(days=30), today + pd.Timedelta(days=90), today + pd.Timedelta(days=180),
                            today + pd.Timedelta(days=365), today + pd.Timedelta(days=730), today + pd.Timedelta(days=1095)
                        ]
                        
                        all_x_dates = [today] + x_future_dates
                        all_y_prices = [current_price] + future_prices
                        
                        future_daily_dates = pd.date_range(start=today, end=x_future_dates[-1], freq='B')
                        x_numeric = mdates.date2num(all_x_dates)
                        future_numeric = mdates.date2num(future_daily_dates)
                        base_trend = np.interp(future_numeric, x_numeric, all_y_prices)
                        
                        simulated_prices = np.copy(base_trend)
                        for i in range(len(all_x_dates) - 1):
                            start_date, end_date = all_x_dates[i], all_x_dates[i+1]
                            mask = (future_daily_dates > start_date) & (future_daily_dates < end_date)
                            segment_length = mask.sum()
                            
                            if segment_length > 0:
                                noise = np.random.normal(0, daily_volatility * base_trend[mask], segment_length)
                                window = np.sin(np.pi * np.arange(1, segment_length + 1) / (segment_length + 1))
                                simulated_prices[mask] += noise * window
                                
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(past_dates, past_prices, color='#2E86C1', label='Past 5 Years History', linewidth=1)
                        ax.plot(future_daily_dates, simulated_prices, color='#F39C12', linewidth=1.2, alpha=0.9, label='AI Simulated Path')
                        ax.scatter(all_x_dates, all_y_prices, color='red', s=20, zorder=5)
                        
                        min_future_price = min(future_prices)
                        min_idx = future_prices.index(min_future_price)
                        buy_date = x_future_dates[min_idx]
                        
                        ax.scatter(buy_date, min_future_price, color='#00FF00', s=350, marker='*', edgecolor='black', zorder=10, label='🔥 Optimal BUY Point')
                        ax.annotate(f"BUY HERE\n${min_future_price}", (buy_date, min_future_price), textcoords="offset points", xytext=(0, -25), ha='center', fontsize=11, fontweight='bold', color='green')

                        for i, txt in enumerate(future_prices):
                            if future_prices[i] != min_future_price:
                                ax.annotate(f"{future_labels[i]}", (x_future_dates[i], future_prices[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
                            
                        ax.axvline(x=today, color='gray', linestyle=':', alpha=0.5)
                        ax.xaxis.set_major_locator(mdates.YearLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                        plt.xticks(rotation=45)
                        ax.set_ylabel("Price ($)")
                        ax.grid(True, linestyle='--', alpha=0.4)
                        ax.legend(loc='upper left')
                        
                        st.pyplot(fig)
                        
                        display_text = final_report[:start_idx].strip() + "\n\n" + final_report[end_idx + len("[PRICE_END]"):].strip()
                        st.markdown(display_text)
                        
                    else:
                        st.warning(f"⚠️ AI가 가격 데이터를 일부 누락했습니다 (찾은 개수: {len(future_prices)}/9). 차트 대신 원본 보고서를 출력합니다.")
                        st.markdown(final_report)
