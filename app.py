import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd
import matplotlib.pyplot as plt

# ⚙️ 작전 설정 (준서 씨 전용 API 키 장착)
GEMINI_API_KEY = "AIzaSyA7hUSq_5FgAESEH-hjjEd-EhK2ilYcKf4"

# 📡 정보 수집조
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

    news_items = ticker.news[:3]
    news_headlines = [news.get('title', '제목 없음') for news in news_items]

    return {
        "ticker": ticker_symbol.upper(),
        "current_price": current_price,
        "ma20": ma20,
        "ma50": ma50,
        "rsi": current_rsi,
        "news": news_headlines,
        "history_14d": hist['Close'].tail(14).tolist()
    }

# 🧠 월스트리트 AI 두뇌
def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    prompt = f"""
    당신은 퀀트 애널리스트입니다. 아래 데이터를 바탕으로 주가를 예측하세요.
    [종목]: {intel_data['ticker']} / [현재가]: ${intel_data['current_price']}
    [지표]: 20일선 ${intel_data['ma20']}, 50일선 ${intel_data['ma50']}, RSI {intel_data['rsi']}
    [최신 뉴스]: {news_text}

    🚨 [매우 중요] 반드시 아래의 양식을 100% 똑같이 지켜서 답변하세요.

    [결과양식]
    # 📊 구간별 AI 예상 주가
    [PRICE_START]
    1일: 000.00
    3일: 000.00
    7일: 000.00
    15일: 000.00
    30일: 000.00
    [PRICE_END]

    # 📋 AI 딥 다이브 전술 보고서
    [1일/3일 단기 뷰]: (상승/하락 확률과 함께 근거 2줄 작성)
    [7일/15일 스윙 뷰]: (상승/하락 확률과 함께 근거 2줄 작성)
    [30일 중기 뷰]: (상승/하락 확률과 함께 근거 2줄 작성)
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"AI 오류: {e}"

# 📱 앱 화면(UI) 구성
st.set_page_config(page_title="AI-Radar", layout="centered")
st.title("🦅 AI-Radar Pro (모바일 앱 에디션)")
st.markdown("전술 참모 시스템입니다. 아래에 코드를 입력하고 가동 버튼을 누르세요.")

TARGET = st.text_input("🎯 타겟 종목 코드를 입력하세요 (예: TSLA, NVDA)", "TSLA")

if st.button("🚀 레이더 가동"):
    with st.spinner("위성 통신 연결 및 전장 데이터 수집 중..."):
        market_intel = gather_intel_pro(TARGET)
        
        if not market_intel:
            st.error("데이터를 불러오지 못했습니다. 종목 코드를 확인해 주세요.")
        else:
            st.success(f"[{market_intel['ticker']}] 데이터 수집 완료! AI 분석을 시작합니다.")
            
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            
            st.subheader("📋 AI 전술 보고서")
            st.text(final_report)
            
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
                
                if len(future_prices) == 5:
                    st.subheader("🗺️ 30일 예상 작전 지도")
                    past_prices = market_intel['history_14d']
                    current_price = market_intel['current_price']
                    
                    x_past = range(-14, 0)
                    x_future = [1, 3, 7, 15, 30]
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(list(x_past) + [0], past_prices + [current_price], color='#2E86C1', label='Past 14 Days', linewidth=2)
                    ax.scatter(0, current_price, color='red', s=80, zorder=5, label=f'Current (${current_price})')
                    ax.plot([0] + x_future, [current_price] + future_prices, color='#F39C12', linestyle='--', marker='o', label='AI Prediction')
                    
                    for i, txt in enumerate(future_prices):
                        ax.annotate(f"${txt}", (x_future[i], future_prices[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                        
                    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
