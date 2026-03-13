# Task
The AI prediction prompt will be updated to include longer-term forecasts.

I will modify the `predict_probability_pro` function in cell `a6dafbc9` to instruct the AI to generate forecasts for 1 year, 2 years, 3 years, and 4 years, in addition to the existing prediction periods (1 day, 3 days, 7 days, 15 days, 30 days, 3 months, 6 months). The `[결과양식]` (result format) will be adjusted to include these new longer-term prediction intervals and their tactical bases.

```python
import yfinance as yf
from google import genai
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ==========================================
#  1. 작전 설정 (준서 씨 전용 API 키 장착)
# ==========================================
GEMINI_API_KEY = "AIzaSyA7hUSq_5FgAESEH-hjjEd-EhK2ilYcKf4"

# ==========================================
# ☆ 2. 정보 수집조 (5년 심층 데이터 스캐너) - ENHANCED VERSION
# ==========================================
def gather_intel_pro(ticker_symbol):
    print(f"☆ [{ticker_symbol}] 타겟의 지난 5년간 전투 기록(차트), 최신 뉴스, 그리고 추가 지표를 긁어옵니다...")
    ticker = yf.Ticker(ticker_symbol)

    # 1) 5년 차트 데이터 수집
    hist = ticker.history(period="5y")

    if hist.empty:
        raise ValueError("데이터를 불러오지 못했습니다. 종목 기호를 확인하세요.")

    current_price = round(hist['Close'].iloc[-1], 2)
    high_5y = round(hist['High'].max(), 2) # 5년 최고가
    low_5y = round(hist['Low'].min(), 2)   # 5년 최저가

    # 실제 6개월 최고/최저 (약 126 거래일)
    hist_6m = hist.tail(126) # Get last ~6 months of data
    high_6m = round(hist_6m['High'].max(), 2) if not hist_6m.empty else "데이터 부족"
    low_6m = round(hist_6m['Low'].min(), 2) if not hist_6m.empty else "데이터 부족"

    # 이동평균선(20일선, 50일선) 계산 - 기관 투자자들의 핵심 지표
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()

    ma20 = round(hist['MA20'].iloc[-1], 2) if not pd.isna(hist['MA20'].iloc[-1]) else "데이터 부족"
    ma50 = round(hist['MA50'].iloc[-1], 2) if not pd.isna(hist['MA50'].iloc[-1]) else "데이터 부족"

    # 2) 최신 뉴스 수집
    news_items = ticker.news[:5]
    news_headlines = [news.get('title', '제목 없음') for news in news_items] # Changed to .get() for robustness

    # ==========================================
    # ENHANCEMENT: Additional Indicators
    # ==========================================

    # RSI (Relative Strength Index) Calculation (14-day)
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else "데이터 부족"

    # Volume Analysis
    current_volume = hist['Volume'].iloc[-1] if not hist['Volume'].empty else "데이터 부족"
    avg_volume_20d = round(hist['Volume'].rolling(window=20).mean().iloc[-1], 2) if len(hist) >= 20 and not hist['Volume'].empty else "데이터 부족"

    # VIX (Volatility Index)
    vix_data = yf.Ticker('^VIX').history(period='1d')
    current_vix = round(vix_data['Close'].iloc[-1], 2) if not vix_data.empty else "데이터 부족"

    # QQQ (NASDAQ 100 ETF)
    qqq_data = yf.Ticker('QQQ').history(period='1d')
    current_qqq = round(qqq_data['Close'].iloc[-1], 2) if not qqq_data.empty else "데이터 부족"

    # Earnings Date and Days Remaining
    earnings_date_str = "데이터 부족"
    days_until_earnings = "데이터 부족"
    try:
        info = ticker.info
        if 'earningsDate' in info and info['earningsDate']:
            earnings_timestamp = info['earningsDate']
            # yfinance sometimes returns a list of timestamps, take the first one
            if isinstance(earnings_timestamp, list):
                earnings_timestamp = earnings_timestamp[0]

            # Check if it's already a datetime object or a Unix timestamp
            if isinstance(earnings_timestamp, int):
                earnings_datetime = datetime.fromtimestamp(earnings_timestamp)
            else:
                # If it's a string, attempt to parse it (though yfinance usually gives int)
                earnings_datetime = datetime.strptime(str(earnings_timestamp).split('.')[0], '%Y-%m-%dT%H:%M:%S')

            earnings_date_str = earnings_datetime.strftime('%Y-%m-%d')
            today = datetime.now()
            time_to_earnings = earnings_datetime - today
            days_until_earnings = time_to_earnings.days + (1 if time_to_earnings.seconds > 0 or time_to_earnings.microseconds > 0 else 0)
            if days_until_earnings < 0: # If earnings date has passed
                earnings_date_str = f"지난 {earnings_date_str}"
                days_until_earnings = "이미 발표"

    except Exception as e:
        print(f"Earnings date fetch error for {ticker_symbol}: {e}")

    # WTI Crude Oil Prices
    wti_data = yf.Ticker('CL=F').history(period='1d')
    current_wti = round(wti_data['Close'].iloc[-1], 2) if not wti_data.empty else "데이터 부족"

    return {
        "ticker": ticker_symbol,
        "current_price": current_price,
        "high_5y": high_5y,
        "low_5y": low_5y,
        "high_6m": high_6m,
        "low_6m": low_6m,
        "ma20": ma20,
        "ma50": ma50,
        "news": news_headlines,
        "rsi": current_rsi,
        "current_volume": current_volume,
        "avg_volume_20d": avg_volume_20d,
        "vix": current_vix,
        "qqq": current_qqq,
        "earnings_date": earnings_date_str,
        "days_until_earnings": days_until_earnings,
        "wti_price": current_wti,
        "hist_data": hist # Return full historical data for plotting
    }

# ==========================================
# ⚙ 3. 전술 지도 시각화 (Tactical Map Visualization)
# ==========================================
def draw_tactical_map(hist_data, current_price, high_5y, low_5y, high_6m, low_6m, ma20, ma50, ticker_symbol):
    print(f"\u2699\u0003 [{ticker_symbol}] 전술 지도 생성 중...")
    # Use a shorter period for better visualization, e.g., last 6 months
    plot_data = hist_data.tail(126) # Approximately 6 months of trading days

    plt.figure(figsize=(12, 7))
    plt.plot(plot_data.index, plot_data['Close'], label='Close Price', color='blue', linewidth=1.5)

    if isinstance(ma20, float):
        plt.plot(plot_data.index, plot_data['MA20'], label='20-day MA', color='orange', linestyle='--')
    if isinstance(ma50, float):
        plt.plot(plot_data.index, plot_data['MA50'], label='50-day MA', color='green', linestyle='--')

    plt.axhline(current_price, color='purple', linestyle=':', linewidth=1, label=f'Current Price: ${current_price}')

    if isinstance(high_5y, float):
        plt.axhline(high_5y, color='red', linestyle='-.', linewidth=0.8, label=f'5-Year High: ${high_5y}')
    if isinstance(low_5y, float):
        plt.axhline(low_5y, color='red', linestyle='-.', linewidth=0.8, label=f'5-Year Low: ${low_5y}')

    if isinstance(high_6m, float):
        plt.axhline(high_6m, color='brown', linestyle=':', linewidth=0.8, label=f'6-Month High: ${high_6m}')
    if isinstance(low_6m, float):
        plt.axhline(low_6m, color='brown', linestyle=':', linewidth=0.8, label=f'6-Month Low: ${low_6m}')

    plt.title(f'{ticker_symbol} Tactical Map - Last 6 Months Overview', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==========================================
#  4. 월스트리트 AI 두뇌 (1일/3일/7일/15일/30일/3개월/6개월/1년/2년/3년/4년 완전 분리 예측기) - UPDATED PROMPT
# ==========================================
def predict_probability_pro(intel_data, ai_key):
    print("⏳ AI 참모가 5년 데이터를 바탕으로 1일/3일/7일/15일/30일/3개월/6개월/1년/2년/3년/4년 뒤를 '각각 따로' 예측 중입니다...\n")
    client = genai.Client(api_key=ai_key)

    # 뉴스 리스트를 텍스트로 변환
    news_text = "\n".join([f"- {news}" for news in intel_data['news']])

    prompt = f"""
    당신은 월스트리트 최고의 퀀트 애널리스트입니다.
    아래의 '지난 5년간 데이터'와 '최신 뉴스', 그리고 추가적인 시장 지표를 바탕으로, 향후 주가를 1일 뒤, 3일 뒤, 7일 뒤, 15일 뒤, 30일 뒤, 3개월 뒤, 6개월 뒤, 1년 뒤, 2년 뒤, 3년 뒤, 4년 뒤 11가지 기간으로 완전히 분리해서 각각의 확률을 도출하세요. \n이러한 데이터를 바탕으로 상승/하락 확률을 더욱 명확하고 단호하게 판단하여 제시해주세요. 단순히 예측하는 것을 넘어, 당신의 심층 분석과 전술적 근거를 바탕으로 가장 합리적인 확률을 부여하세요.

    [타겟 종목]: {intel_data['ticker']}
    [현재가]: ${intel_data['current_price']}
    [5년간 최고/최저]: 최고 ${intel_data['high_5y']} / 최저 ${intel_data['low_5y']}
    [6개월 최고/최저]: 최고 ${intel_data['high_6m']} / 최저 ${intel_data['low_6m']}
    [이동평균선]: 20일선(단기 생명선) ${intel_data['ma20']}, 50일선(중기 추세선) ${intel_data['ma50']}
    [RSI (14일)]: {intel_data['rsi']}
    [현재 거래량]: {intel_data['current_volume']}
    [20일 평균 거래량]: {intel_data['avg_volume_20d']}
    [VIX (변동성 지수)]: {intel_data['vix']}
    [NASDAQ 100 (QQQ)]: {intel_data['qqq']}
    [다음 실적 발표일]: {intel_data['earnings_date']} (남은 일수: {intel_data['days_until_earnings']})
    [WTI 유가]: ${intel_data['wti_price']}
    [최신 뉴스 동향]:
    {news_text}

    결과는 반드시 아래 [결과양식]에 맞춰서만 출력하고, 다른 인사말이나 부연 설명은 절대 추가하지 마세요.

    [결과양식]
    [1일 뒤 예측 (초단기 대응)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [뉴스 모멘텀, RSI, 거래량 중심 1줄 요약]

    [3일 뒤 예측 (단기 추이)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [단기 시장 심리, VIX, QQQ 등 외부 요인 중심 1줄 요약]

    [7일 뒤 예측 (단기 스윙 타점)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [20일 이동평균선, RSI 과매수/과매도, 주간 흐름 중심 1줄 요약]

    [15일 뒤 예측 (중단기 추세)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [최근 2주간 기술적 지표, 실적 발표 기대감, WTI 유가 영향 등 종합 1줄 요약]

    [30일 뒤 예측 (중기 추세 판단)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [6개월 고점/저점, 50일선, VIX 장기 추세 등 거시적 추세 중심 1줄 요약]

    [3개월 뒤 예측 (중장기 관점)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [주요 경제 지표, 거시 경제 상황, 기업 펀더멘털 변화 등 중장기적 관점 1줄 요약]

    [6개월 뒤 예측 (장기 투자 전략)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [산업 동향, 경쟁 환경, 기술 변화, 금리 인상/인하 기대감 등 장기적 관점 1줄 요약]

    [1년 뒤 예측 (장기 전략)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [산업 성장성, 시장 점유율 변화, 거시 경제 정책 등 장기적 관점 1줄 요약]

    [2년 뒤 예측 (초장기 비전)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [기술 혁신, 패러다임 변화, 기업의 지속 가능성 등 초장기적 관점 1줄 요약]

    [3년 뒤 예측 (메가트렌드 분석)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [글로벌 경제 변화, 인구 구조 변화, 신기술 도입 속도 등 메가트렌드 관점 1줄 요약]

    [4년 뒤 예측 (미래 가치 평가)]
    - 상승 확률: 00% / 하락 확률: 00%
    - 전술 근거: [미래 시장 주도권, 신사업 성공 여부, 규제 환경 변화 등 미래 가치 평가 관점 1줄 요약]
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI 통신 오류 발생: {e}"

# ==========================================
#  5. 작전 실행 (Main 스위치)
# ==========================================
if __name__ == "__main__":
    print("======================================")
    print("\u0001 [AI-Radar Pro] 1일/3일/7일/15일/30일/3개월/6개월/1년/2년/3년/4년 분리 예측 시스템")
    print("======================================\n")

    # 분석하고 싶은 종목의 티커를 입력하세요 (예: NVDA, AMZN, MSFT)
    TARGET = "googl"

    try:
        # 1단계: 데이터 수집
        market_intel = gather_intel_pro(TARGET)

        # 2단계: 전술 지도 시각화
        draw_tactical_map(
            market_intel['hist_data'],
            market_intel['current_price'],
            market_intel['high_5y'],
            market_intel['low_5y'],
            market_intel['high_6m'],
            market_intel['low_6m'],
            market_intel['ma20'],
            market_intel['ma50'],
            market_intel['ticker']
        )

        # 3단계: AI 분석
        final_report = predict_probability_pro(market_intel, GEMINI_API_KEY)

        # 결과 출력
        print("┣ [AI 전술 보고서 (Wall Street Edition)]")
        print("=" * 50)
        print(final_report.strip())
        print("=" * 50)
    except Exception as e:
        print(f"시스템 오류: {e}")
```
