import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ⚙️ 작전 설정 (비밀 금고에서 안전하게 키를 불러옵니다)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

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

# 🧠 월스트리트 AI 두뇌 (빠르고 안정적인 Gemini 2.5 Flash 고속 엔진 장착!)
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
    (앞으로 3년간 비중을 어떻게 조절하고 언제가 최적의 매수 타이밍인지 상세히 가이드)

    ### 📋 AI 심층 투자 보고서
    **[🚀 1일~30일 초단기/단기 전망]**: (시장 모멘텀, 기술적 지표 상세 분석)
    **[🗓️ 3개월~1년 중장기 전망]**: (거시 경제 흐름, 실적 전망 기반 추세 분석)
    **[📅 2년~3년 초장기 전망]**: (기업의 본질적 가치, 해자 기반 비전 제시)
    """
    try:
        # 🔥 엔진을 빠르고 안정적인 Flash 모델로 교체했습니다.
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"AI 오류: {e}"

# 📱 앱 화면(UI) 구성
st.set_page_config(page_title="AI-Radar 고속 에디션", layout="centered", page_icon="⚡")
st.title("⚡ AI-Radar (고속 Flash 엔진 탑재)")
st.markdown("**빠르고 쾌적한 Gemini
