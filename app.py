import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ⚙️ 작전 설정 (준서 씨 전용 API 키 장착)
GEMINI_API_KEY = "AIzaSyCRpJwppWimK97OAhBQP85TkXN67aGHfxo"

# 📡 정보 수집조 (심층 데이터 스캐너)
def gather_intel_pro(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    # 차트 시각화 요구사항을 위해 지난 기록 전체(5년)를 사용합니다.
    hist = ticker.history(period="5y")
    if hist.empty:
        return None

    current_price = round(hist['Close'].iloc[-1], 2)
    
    # 지표 계산용 (기존 유지)
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
        "history": hist # 차트용 과거 5년치 데이터 전체
    }

# 🧠 월스트리트 AI 두뇌 (상세 단기 & 중장기 통합 뷰)
def predict_probability_pro(intel_data, ai_key):
    client = genai.Client(api_key=ai_key)
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    # 프롬프트 업데이트: AI 평가 한마디 추가, 1-30일 상세 예상 요구
    prompt = f"""
    당신은 월스트리트 최고의 가치 투자자이자 퀀트 애널리스트입니다. 
    아래 데이터를 바탕으로 해당 기업의 '단기(1일~30일)' 및 '중장기(3개월~3년)' 주가 흐름을 예측하세요.
    
    [종목]: {intel_data['ticker']} / [현재가]: ${intel_data['current_price']}
    [기술적 지표]: 20일선 ${intel_data['ma20']}, 50일선 ${intel_data['ma50']}, RSI {intel_data['rsi']}
    [최신 주요 뉴스]: {news_text}

    🚨 [매우 중요] 반드시 아래의 양식을 100% 똑같이 지켜서 답변하세요.

    [결과양식]
    # 🗣️ AI의 평가 한마디
    (현재 상황에 대한 아주 간결하고 명확한 요약 1줄)

    # 📊 구간별 AI 예상 주가
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

    # 📋 AI 심층 투자 보고서
    [🚀 1일~30일 상세 단기 전망]: 
    (1일, 1주, 2주, 1개월 단위로 시장 모멘텀, 기술적 지표, 뉴스의 단기적 영향을 아주 상세하게 분석하고 상승/하락 확률 및 근거 작성, 최소 5줄 이상)

    [🗓️ 3개월~1년 중장기 전망]: 
    (거시 경제 흐름, 실적 전망, 산업 내 경쟁 지위 변화를 기반으로 중장기 추세 분석 및 상승/하락 확률 근거 작성, 최소 3줄 이상)

    [📅 2년~3년 초장기 전망]: 
    (기업의 본질적 가치, 산업의 구조적 성장성, 해자(Moat)의 지속가능성을 기반으로 초장기 비전 제시 및 상승/하락 확률 근거 작성, 최소 3줄 이상)
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"AI 오류: {e}"

# 📱 앱 화면(UI) 구성
st.set_page_config(page_title="AI-Radar Pro", layout="centered")
st.title("🦅 AI-Radar (심층 투자 분석 에디션)")
st.markdown("가치 투자 및 단기 모멘텀 분석을 위한 통합 예측 시스템입니다.")

TARGET = st.text_input("🎯 타겟 종목 코드를 입력하세요 (예: TSLA, AAPL, NVDA)", "TSLA")

if st.button("🚀 심층 예측 레이더 가동"):
    with st.spinner("과거 5년치 데이터 전체 분석 및 미래 시뮬레이션 중..."):
        market_intel = gather_intel_pro(TARGET)
        
        if not market_intel:
            st.error("데이터를 불러오지 못했습니다. 종목 코드를 확인해 주세요.")
        else:
            st.success(f"[{market_intel['ticker']}] 데이터 수집 완료! 심층 AI 분석을 시작합니다.")
            
            final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)
            
            # AI 평가 한마디 추출 및 표시
            st.subheader("🗣️ AI의 평가 한마디")
            evaluation_start = final_report.find("# 🗣️ AI의 평가 한마디")
            price_start_section = final_report.find("# 📊 구간별 AI 예상 주가")
            if evaluation_start != -1 and price_start_section != -1:
                evaluation_text = final_report[evaluation_start + len("# 🗣️ AI의 평가 한마디"):price_start_section].strip()
                st.info(evaluation_text)
            
            # 심층 보고서 표시
            st.subheader("📋 AI 심층 투자 보고서")
            report_start = final_report.find("# 📋 AI 심층 투자 보고서")
            if report_start != -1:
                report_text = final_report[report_start + len("# 📋 AI 심층 투자 보고서"):].strip()
                st.text(report_text)
            
            # 가격표 파싱 및 차트 그리기
            start_idx = final_report.find("[PRICE_START]")
            end_idx = final_report.find("[PRICE_END]")
            
            if start_idx != -1 and end_idx != -1:
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
                
                # 9개의 미래 가격 포인트가 있는지 확인
                if len(future_prices) == 9:
                    st.subheader("🗺️ 3년 장기 예상 작전 지도 (과거 전체 기록 참고)")
                    
                    hist = market_intel['history']
                    past_dates = hist.index
                    past_prices = hist['Close']
                    current_price = market_intel['current_price']
                    today = past_dates[-1]
                    
                    # 미래 날짜 계산
                    x_future_dates = [
                        today + pd.Timedelta(days=1),
                        today + pd.Timedelta(weeks=1),
                        today + pd.Timedelta(weeks=2),
                        today + pd.Timedelta(days=30),
                        today + pd.Timedelta(days=90),
                        today + pd.Timedelta(days=180),
                        today + pd.Timedelta(days=365),
                        today + pd.Timedelta(days=730),
                        today + pd.Timedelta(days=1095)
                    ]
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # 1. 과거 5년 데이터 전체 (뾰족한 곡선 그래프 - 기본 라인 플롯)
                    ax.plot(past_dates, past_prices, color='#2E86C1', label='Past 5 Years History', linewidth=1)
                    
                    # 2. 현재 가격 점 (빨간색)
                    ax.scatter(today, current_price, color='red', s=100, zorder=5, label=f'Current (${current_price})')
                    
                    # 3. AI 미래 예측 선 (주황색 점선)
                    all_x_dates = [today] + x_future_dates
                    all_y_prices = [current_price] + future_prices
                    ax.plot(all_x_dates, all_y_prices, color='#F39C12', linestyle='--', marker='o', label='AI Prediction')
                    
                    # 미래 가격 텍스트 달아주기
                    for i, txt in enumerate(future_prices):
                        ax.annotate(f"{future_labels[i]}\n${txt}", (x_future_dates[i], future_prices[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, fontweight='bold')
                        
                    # 오늘 기준선
                    ax.axvline(x=today, color='gray', linestyle=':', alpha=0.5)
                    
                    # X축 포맷 설정 (연도 단위)
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    plt.xticks(rotation=45)
                    
                    ax.set_ylabel("Price ($)")
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ 차트 생성 오류: AI가 구간별 가격을 정확히 산출하지 못했습니다.")
