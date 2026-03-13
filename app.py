import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ⚙️ 작전 설정 (준서 씨 전용 API 키 장착)
GEMINI_API_KEY = "AIzaSyCRpJwppWimK97OAhBQP85TkXN67aGHfxo"

# 📡 정보 수집조 (장기 데이터 스캐너)
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

    # RSI (14일)
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else "데이터 부족"

    news_items = ticker.news[:4]
    news_headlines = [news.get('title', '제목 없음') for news in news_items]

    return {
        "ticker": ticker_symbol.upper(),
        "current_price": current_price,
        "ma20": ma20,
        "ma50": ma50,
        "rsi": current_rsi,
        "news": news_headlines,
        "history_252d": hist['Close'].tail(252).tolist() # 차트용 과거 1년치 데이터 (약 252 거래일)
    }

# 🧠 월스트리트 AI 두뇌 (장기 뷰 장착)
def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    prompt = f"""
    당신은 월스트리트 최고의 가치 투자자이자 퀀트 애널리스트입니다. 
    아래 데이터를 바탕으로 해당 기업의 '중장기적 주가 흐름(1개월~3년)'을 예측하세요.
    
    [종목]: {intel_data['ticker']} / [현재가]: ${intel_data['current_price']}
    [기술적 지표]: 20일선 ${intel_data['ma20']}, 50일선 ${intel_data['ma50']}, RSI {intel_data['rsi']}
    [최신 주요 뉴스]: {news_text}

    🚨 [매우 중요] 반드시 아래의 양식을 100% 똑같이 지켜서 답변하세요.

    [결과양식]
    # 📊 구간별 AI 예상 주가 (장기 예측)
    [PRICE_START]
    1개월: 000.00
    3개월: 000.00
    6개월: 000.00
    1년: 000.00
    2년: 000.00
    3년: 000.00
    [PRICE_END]

    # 📋 AI 딥 다이브 장기 투자 보고서
    [1~3개월 중단기 뷰]: (현재 모멘텀과 뉴스 기반 상승/하락 확률 및 근거 2줄)
    [6개월~1년 중장기 뷰]: (거시 경제 흐름과 실적 전망 기반 상승/하락 확률 및 근거 2줄)
    [2~3년 초장기 뷰]: (해당 기업의 산업 성장성, 해자(Moat) 기반 상승/하락 확률 및 근거 2줄)
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"AI 오류: {e}"

# 📱 앱 화면(UI) 구성
st.set_page_config(page_title="AI-Radar Long-Term", layout="centered")
st.title("🦅 AI-Radar (장기 투자 에디션)")
st.markdown("가치 투자를 위한 중장기(1개월~3년) 예측 시스템입니다.")

TARGET = st.text_input("🎯 타겟 종목 코드를 입력하세요 (예: TSLA, AAPL, NVDA)", "TSLA")

if st.button("🚀 장기 예측 레이더 가동"):
    with st.spinner("과거 5년치 데이터 분석 및 미래 3년 시뮬레이션 중..."):
        market_intel = gather_intel_pro(TARGET)
        
        if not market_intel:
            st.error("데이터를 불러오지 못했습니다. 종목 코드를 확인해 주세요.")
        else:
            st.success(f"[{market_intel['ticker']}] 데이터 수집 완료! 심층 AI 분석을 시작합니다.")
            
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            
            st.subheader("📋 AI 장기 투자 전술 보고서")
            st.text(final_report)
            
            # 가격표 파싱
            start_idx = final_report.find("[PRICE_START]")
            end_idx = final_report.find("[PRICE_END]")
            
            if start_idx != -1 and end_idx != -1:
                price_block = final_report[start_idx + len("[PRICE_START]"):end_idx].strip()
                lines = price_block.split('\n')
                future_prices = []
                
                for line in lines:
                    if ":" in line:
                        val = ''.join(c for c in line.split(':')[1] if c.isdigit() or c == '.')
                        if val:
                            future_prices.append(float(val))
                
                # 6개의 미래 가격(1/3/6/12/24/36개월)을 성공적으로 뽑아냈다면 차트 그리기
                if len(future_prices) == 6:
                    st.subheader("🗺️ 3년 장기 예상 작전 지도")
                    
                    past_prices = market_intel['history_252d']
                    current_price = market_intel['current_price']
                    
                    # 과거 1년(-365일)부터 오늘(0일)까지의 X축
                    x_past = list(np.linspace(-365, 0, len(past_prices)))
                    
                    # 미래 1/3/6/12/24/36개월을 '일수(Days)'로 변환
                    x_future = [30, 90, 180, 365, 730, 1095]
                    future_labels = ["1M", "3M", "6M", "1Y", "2Y", "3Y"]
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # 과거 1년 선 (파란색)
                    ax.plot(x_past + [0], past_prices + [current_price], color='#2E86C1', label='Past 1 Year', linewidth=2)
                    
                    # 현재 가격 점 (빨간색)
                    ax.scatter(0, current_price, color='red', s=80, zorder=5, label=f'Current (${current_price})')
                    
                    # 미래 3년 선 (주황색 점선)
                    ax.plot([0] + x_future, [current_price] + future_prices, color='#F39C12', linestyle='--', marker='o', label='AI Prediction (up to 3 Years)')
                    
                    # 미래 가격 텍스트 달아주기
                    for i, txt in enumerate(future_prices):
                        ax.annotate(f"{future_labels[i]}\n${txt}", (x_future[i], future_prices[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, fontweight='bold')
                        
                    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5) # 오늘 기준선
                    ax.set_xlabel("Days from Today (0 = Today)")
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ 차트 생성 오류: AI가 장기 가격을 정확히 산출하지 못했습니다.")
