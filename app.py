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

# 📡 정보 수집조 (1시간 주기 + 10개 뉴스 + 7일 백업)
@st.cache_data(ttl=3600)
def gather_intel_master(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5y")
        if hist.empty: return None
        
        # 기술 지표
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = round(100 - (100 / (1 + gain/loss)).iloc[-1], 2)

        # 7일 뉴스 수집 (최대 10개)
        all_news = ticker.news
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_news = [n.get('title') for n in all_news if datetime.fromtimestamp(n.get('providerPublishTime', 0)) >= seven_days_ago]
        
        if not recent_news:
            recent_news = [n.get('title') for n in all_news[:10]]
            mode = "7일간의 기록 기반 (백업 분석)"
        else:
            mode = "실시간 & 7일 주요 소식 (정밀 분석)"

        return {
            "ticker": ticker_symbol.upper(), "p": round(hist['Close'].iloc[-1], 2), 
            "rsi": rsi, "news": recent_news[:10], "mode": mode, 
            "h": hist, "vol": hist['Close'].pct_change().std()
        }
    except: return None

# 🧠 AI 끝장 토론 두뇌 (비관론 vs 낙관론)
def conduct_ai_debate(intel, key):
    client = genai.Client(api_key=key)
    news_text = "\n".join([f"- {n}" for n in intel['news']])
    
    prompt = f"""
    [가상 퀀트 끝장 토론] 종목: {intel['ticker']} / 현재가: ${intel['p']}
    [데이터]: RSI {intel['rsi']} / [정보 모드]: {intel['mode']}
    [수집 뉴스 10개]: {news_text}

    🚨 [작전 지시]
    당신은 '냉철한 비관론자'와 '합리적 낙관론자'입니다. 
    1. 비관론자는 주가가 하락하거나 정체될 리스크 3가지를 공격적으로 제시하세요.
    2. 낙관론자는 이에 반박하며 기회 요인을 제시하세요.
    3. 결과적으로 가장 현실적인 3년 주가 시뮬레이션을 도출하세요.

    [결과양식]
    ### 🗣️ AI 참모의 끝장 토론 브리핑
    - **낙관적 견해**: 내용
    - **비관적 리스크**: 리스크 3가지 필수 포함
    - **최종 전략 결론**: 냉정한 요약

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

    ### 🎯 향후 3년 마스터 투자 로드맵
    (리스크 관리를 포함한 상세 전략)
    """
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return resp.text
    except Exception as e:
        if "429" in str(e): return "OVERLOAD"
        return f"AI 오류: {e}"

# 📱 UI 구성
st.set_page_config(page_title="AI-Radar Master", layout="centered", page_icon="🦅")
st.title("🦅 AI-Radar (퀀트 마스터 최종본)")
st.markdown("1시간 주기 최신화 | 10개 뉴스 | 7일 백업 | 객관적 끝장 토론")

ticker = st.text_input("🎯 종목 코드를 입력하세요", "TSLA").upper()

if st.button("🚀 레이더 가동"):
    with st.spinner("최신 정보를 스캔하고 AI 분석가들이 격렬하게 토론 중입니다..."):
        info = gather_intel_master(ticker)
        if not info:
            st.error("데이터 수집 실패. 종목 코드를 확인하세요.")
        else:
            report = conduct_ai_debate(info, GEMINI_API_KEY)
            
            if report == "OVERLOAD":
                st.warning("⚠️ 구글 서버가 과열되었습니다! 딱 1분만 쉬었다가 다시 눌러주세요.")
            elif "AI 오류" in report:
                st.error(report)
            else:
                s_idx = report.find("[PRICE_START]"); e_idx = report.find("[PRICE_END]")
                if s_idx != -1 and e_idx != -1:
                    try:
                        p_txt = report[s_idx+13:e_idx].strip()
                        f_p = [float(''.join(c for c in l.split(':')[1] if c.isdigit() or c=='.')) for l in p_txt.split('\n') if ':' in l]
                        
                        # 차트 시뮬레이션 (뾰족 곡선)
                        h = info['h']; today = h.index[-1]
                        xf = [today + pd.Timedelta(days=d) for d in [1, 7, 14, 30, 90, 180, 365, 730, 1095]]
                        fd = pd.date_range(start=today, end=xf[-1], freq='B')
                        sp = np.interp(mdates.date2num(fd), mdates.date2num([today]+xf), [info['p']]+f_p)
                        
                        # 변동성 주입
                        for i in range(len(xf)):
                            m = (fd > (today if i==0 else xf[i-1])) & (fd <= xf[i])
                            if m.sum() > 0:
                                noise = np.random.normal(0, info['vol']*sp[m], m.sum())
                                window = np.sin(np.pi * np.arange(1, m.sum()+1)/(m.sum()+1))
                                sp[m] += noise * window
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(h.index, h['Close'], color='#5D6D7E', alpha=0.6, label='History')
                        ax.plot(fd, sp, color='#CA6F1E', label='AI Simulated Path')
                        
                        # 최저점(별)
                        min_p = min(f_p); buy_d = xf[f_p.index(min_p)]
                        ax.scatter(buy_d, min_p, color='#27AE60', s=350, marker='*', zorder=10, label='Optimal BUY')
                        ax.annotate(f"BUY HERE\n${min_p}", (buy_d, min_p), textcoords="offset points", xytext=(0, -25), ha='center', fontsize=9, color='green', fontweight='bold')
                        
                        ax.axvline(x=today, color='gray', linestyle=':', alpha=0.5)
                        ax.legend(); st.pyplot(fig)
                        
                        st.markdown(report[:s_idx].strip() + "\n\n" + report[e_idx+11:].strip())
                    except: st.markdown(report)
                else: st.markdown(report)
