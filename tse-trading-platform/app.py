import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime, timedelta

# افزودن مسیر ریشه پروژه به sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.indicators import engineer_features

# ============================================
# تنظیمات صفحه
# ============================================
st.set_page_config(
    page_title="CryptoAI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS حرفه‌ای با فونت Vazirmatn و افکت‌های مدرن
# ============================================
st.markdown("""
<link href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css" rel="stylesheet">
<style>
    /* فونت اصلی - همه جا */
    html, body, .stApp, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Vazirmatn', 'Tahoma', sans-serif !important;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%) !important;
        color: white !important;
    }
    
    /* اطمینان از پوشش کامل پس‌زمینه */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%) !important;
    }
    
    section[data-testid="stSidebar"], 
    [data-testid="stHeader"],
    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }
    
    /* متن‌ها سفید */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: white !important;
    }
    
    /* هدر اصلی */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2em;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1em;
        margin: 5px 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    /* کارت‌های شیشه‌ای */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .glass-card .label {
        color: #a0a8c0 !important;
        font-size: 0.9em;
        margin-bottom: 8px;
    }
    .glass-card .value {
        color: white !important;
        font-size: 1.6em;
        font-weight: bold;
    }
    .glass-card .delta {
        font-size: 0.95em;
        margin-top: 5px;
    }
    .delta-positive { color: #00d4aa !important; }
    .delta-negative { color: #ff4d6d !important; }
    
    /* کارت‌های سیگنال */
    .signal-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin: 20px 0;
    }
    .signal-long {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.15) 0%, rgba(0, 255, 136, 0.05) 100%);
        border: 2px solid #00d4aa;
        box-shadow: 0 0 30px rgba(0, 212, 170, 0.3);
    }
    .signal-short {
        background: linear-gradient(135deg, rgba(255, 77, 109, 0.15) 0%, rgba(255, 0, 85, 0.05) 100%);
        border: 2px solid #ff4d6d;
        box-shadow: 0 0 30px rgba(255, 77, 109, 0.3);
    }
    .signal-neutral {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 152, 0, 0.05) 100%);
        border: 2px solid #ffc107;
        box-shadow: 0 0 30px rgba(255, 193, 7, 0.3);
    }
    .signal-card h2 {
        font-size: 2em;
        margin: 0;
        text-shadow: 0 0 20px currentColor;
    }
    .signal-long h2 { color: #00d4aa !important; }
    .signal-short h2 { color: #ff4d6d !important; }
    .signal-neutral h2 { color: #ffc107 !important; }
    
    /* RSI Bar */
    .rsi-container {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    .rsi-bar {
        height: 12px;
        background: linear-gradient(to right, #00d4aa 0%, #00d4aa 30%, #ffc107 30%, #ffc107 70%, #ff4d6d 70%, #ff4d6d 100%);
        border-radius: 6px;
        position: relative;
        margin-top: 10px;
    }
    .rsi-indicator {
        position: absolute;
        top: -8px;
        width: 4px;
        height: 28px;
        background: white;
        border-radius: 2px;
        box-shadow: 0 0 10px white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%) !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.03);
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px;
        background: transparent;
        color: #a0a8c0 !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* عنوان‌ها */
    .section-title {
        color: white !important;
        font-size: 1.5em;
        font-weight: bold;
        margin: 25px 0 15px 0;
        padding-right: 15px;
        border-right: 4px solid #667eea;
    }
    
    /* دکمه‌ها */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Select Box */
    .stSelectbox label, .stSelectbox [data-baseweb="select"] {
        color: white !important;
    }
    
    /* فوتر */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        color: #a0a8c0 !important;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    /* ریسپانسیو */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.5em; }
        .glass-card .value { font-size: 1.3em; }
        .signal-card h2 { font-size: 1.5em; }
    }
        /* Selectbox - حل مشکل دیده نشدن */
    .stSelectbox {
        color: white !important;
    }
    
    .stSelectbox label {
        color: white !important;
        font-weight: bold;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    
    .stSelectbox [data-baseweb="select"] div {
        color: white !important;
        background-color: transparent !important;
    }
    
    .stSelectbox [data-baseweb="select"] span {
        color: white !important;
    }
    
    /* Dropdown Menu */
    .stSelectbox [data-baseweb="menu"] {
        background-color: #1a1f3a !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li {
        background-color: #1a1f3a !important;
        color: white !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li:hover {
        background-color: rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Button در Selectbox */
    .stSelectbox [data-baseweb="button"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Input Fields عمومی */
    .stTextInput input, 
    .stNumberInput input,
    .stDateInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Checkbox و Radio */
    .stCheckbox label, 
    .stRadio label {
        color: white !important;
    }
    
    /* Slider */
    .stSlider label {
        color: white !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        color: white !important;
    }
        /* راست‌چین کردن کامل */
    html, body, .stApp, [data-testid="stAppViewContainer"], 
    [data-testid="stSidebar"], .main, section {
        direction: rtl !important;
    }
    
    /* همه متن‌ها راست‌چین */
    h1, h2, h3, h4, h5, h6, p, span, label, div, li, td, th,
    .stMarkdown, .stText, .stCaption {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* کارت‌ها راست‌چین */
    .glass-card, .signal-card, .main-header, .section-title, .rsi-container {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* Selectbox راست‌چین */
    .stSelectbox label,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="menu"] li {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* Buttons راست‌چین */
    .stButton button {
        direction: rtl !important;
        text-align: center !important;
    }
    
    /* Metrics راست‌چین */
    [data-testid="stMetric"] {
        direction: rtl !important;
        text-align: right !important;
    }
    
    [data-testid="stMetricLabel"] {
        text-align: right !important;
    }
    
    [data-testid="stMetricValue"] {
        text-align: right !important;
    }
    
    /* Tabs راست‌چین */
    .stTabs [data-baseweb="tab"] {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* Sidebar راست‌چین */
    [data-testid="stSidebar"] * {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* Info/Warning/Error boxes راست‌چین */
    .stAlert, [data-testid="stAlert"] {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* Dataframe راست‌چین (جدول‌ها) */
    .stDataFrame {
        direction: rtl !important;
    }
    
    .stDataFrame th, .stDataFrame td {
        text-align: right !important;
    }
    
    /* Plotly charts راست‌چین */
    .js-plotly-plot {
        direction: rtl !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# توابع کمکی (غیر از اندیکاتورها که از ماژول مشترک ایمپورت شده‌اند)
# ============================================

@st.cache_data(ttl=300)
def load_data(sym, days=200):
    """بارگذاری داده‌ها از yfinance با مدیریت خطا"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # تبدیل فرمت نماد برای yfinance
        yf_symbol = sym.upper()
        
        df = yf.download(
            yf_symbol, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'), 
            progress=False, 
            auto_adjust=False
        )
        
        if df.empty:
            st.warning(f"⚠️ داده‌ای برای {sym} یافت نشد")
            return None
            
        # استانداردسازی ستون‌ها
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        df = df.reset_index()
        df.columns = [str(c).strip().title() for c in df.columns]
        
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        
        # حذف داده‌های ناقص
        df = df.dropna(subset=['Close', 'Volume'])
        df = df[df['Volume'] > 0]
        
        return df
        
    except Exception as e:
        st.error(f"❌ خطا در دریافت داده‌ها: {str(e)}")
        return None

def load_model(sym):
    """بارگذاری مدل آموزش‌دیده با مدیریت خطا"""
    try:
        # مسیر مطلق به پوشه models
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        
        if not os.path.exists(models_dir):
            return None, None, None
            
        safe = sym.replace('-', '_').lower()
        files = [f for f in os.listdir(models_dir) if safe in f.lower() and "xgboost" in f.lower()]
        
        if not files:
            return None, None, None
            
        mf = os.path.join(models_dir, files[0])
        sf = mf.replace("XGBoost", "scaler").replace("xgboost", "scaler")
        ff = mf.replace("XGBoost", "features").replace("xgboost", "features")
        
        if not all(os.path.exists(x) for x in [mf, sf, ff]):
            return None, None, None
            
        return joblib.load(mf), joblib.load(sf), joblib.load(ff)
        
    except Exception as e:
        st.error(f"❌ خطا در بارگذاری مدل: {str(e)}")
        return None, None, None

# ============================================
# هدر اصلی
# ============================================
st.markdown("""
<div class="main-header">
    <h1>📈 CryptoAI Pro</h1>
    <p>پلتفرم هوشمند تحلیل و پیش‌بینی بازار کریپتو با یادگیری ماشین</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# سایدبار
# ============================================
with st.sidebar:
    st.markdown("## ⚙️ تنظیمات")
    symbol_map = {
        "🟠 بیت‌کوین (BTC)": "BTC-USD",
        "🔷 اتریوم (ETH)": "ETH-USD",
        "🟡 بایننس کوین (BNB)": "BNB-USD",
        "🟣 سولانا (SOL)": "SOL-USD",
        "🔵 کاردانو (ADA)": "ADA-USD"
    }
    selected_name = st.selectbox("🪙 انتخاب ارز", list(symbol_map.keys()))
    symbol = symbol_map[selected_name]
    
    st.markdown("---")
    st.markdown(f"🕐 **آخرین به‌روزرسانی:**")
    st.markdown(f"`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    
    st.markdown("---")
    st.markdown("### 📊 وضعیت بازار")
    df_quick = load_data(symbol, days=7)
    if df_quick is not None and not df_quick.empty:
        chg = ((df_quick['Close'].iloc[-1] - df_quick['Close'].iloc[0]) / df_quick['Close'].iloc[0]) * 100
        if chg > 0:
            st.success(f"📈 {chg:+.2f}% در ۷ روز")
        else:
            st.error(f"📉 {chg:+.2f}% در ۷ روز")

# ============================================
# تب‌های اصلی
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 داشبورد", "🎯 سیگنال زنده", "📈 بک‌تست", "ℹ️ درباره"])

# ============================================
# تب ۱: داشبورد
# ============================================
with tab1:
    df = load_data(symbol, days=200)
    
    if df is not None and not df.empty:
        df_f = engineer_features(df)
        df_c = df_f.dropna().reset_index(drop=True)
        
        if not df_c.empty:
            cp = df_c['Close'].iloc[-1]
            pc = df_c['Daily_Return'].iloc[-1] * 100
            rsi = df_c['RSI_14'].iloc[-1]
            atr = df_c['ATR_14'].iloc[-1]
            vol = df_c['Volume'].iloc[-1]
            
            # کارت‌های اطلاعات با افکت شیشه‌ای
            st.markdown('<div class="section-title">💰 اطلاعات لحظه‌ای</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                delta_class = "delta-positive" if pc >= 0 else "delta-negative"
                st.markdown(f"""
                <div class="glass-card">
                    <div class="label">💰 قیمت فعلی</div>
                    <div class="value">${cp:,.2f}</div>
                    <div class="delta {delta_class}">{pc:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                rsi_color = "#00d4aa" if rsi < 30 else ("#ff4d6d" if rsi > 70 else "#ffc107")
                st.markdown(f"""
                <div class="glass-card">
                    <div class="label">📊 RSI (14)</div>
                    <div class="value" style="color: {rsi_color};">{rsi:.1f}</div>
                    <div class="delta" style="color: #a0a8c0;">{'اشباع فروش' if rsi < 30 else ('اشباع خرید' if rsi > 70 else 'خنثی')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="label">📈 نوسان (ATR)</div>
                    <div class="value">${atr:,.2f}</div>
                    <div class="delta" style="color: #a0a8c0;">میانگین ۱۴ روزه</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c4:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="label">📊 حجم معاملات</div>
                    <div class="value">{vol/1e9:.2f}B</div>
                    <div class="delta" style="color: #a0a8c0;">دلار در ۲۴ ساعت</div>
                </div>
                """, unsafe_allow_html=True)
            
            # نمودار پیشرفته
            st.markdown('<div class="section-title">📊 نمودار تحلیل تکنیکال</div>', unsafe_allow_html=True)
            
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=('قیمت و باندهای بولینگر', 'RSI', 'MACD')
            )
            
            fig.add_trace(go.Candlestick(
                x=df_c['Date'], open=df_c['Open'], high=df_c['High'],
                low=df_c['Low'], close=df_c['Close'], name='قیمت',
                increasing_line_color='#00d4aa', decreasing_line_color='#ff4d6d'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['BB_Upper'],
                line=dict(color='rgba(102, 126, 234, 0.5)', width=1), name='BB Upper'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['BB_Lower'],
                line=dict(color='rgba(102, 126, 234, 0.5)', width=1), fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)', name='BB Lower'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['SMA_20'],
                line=dict(color='#ffc107', width=1.5), name='SMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['SMA_50'],
                line=dict(color='#ff6b9d', width=1.5), name='SMA 50'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['RSI_14'],
                line=dict(color='#667eea', width=2), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ff4d6d", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa", row=2, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0, row=2, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0, row=2, col=1)
            
            colors = ['#00d4aa' if v >= 0 else '#ff4d6d' for v in df_c['MACD_Hist']]
            fig.add_trace(go.Bar(x=df_c['Date'], y=df_c['MACD_Hist'], name='MACD Hist',
                marker_color=colors), row=3, col=1)
            
            fig.update_layout(
                template='plotly_dark', height=700,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False, hovermode='x unified',
                font=dict(family='Vazirmatn, Tahoma', color='#a0a8c0'),
                legend=dict(bgcolor='rgba(0,0,0,0)')
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # جدول داده‌های اخیر
            st.markdown('<div class="section-title">📋 داده‌های ۷ روز اخیر</div>', unsafe_allow_html=True)
            display_df = df_c[['Date', 'Close', 'Volume', 'RSI_14', 'MACD_Hist', 'Daily_Return']].tail(7).copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df = display_df.sort_values('Date', ascending=False)
            st.dataframe(display_df.style.format({
                'Close': '${:,.2f}', 'Volume': '{:,.0f}',
                'RSI_14': '{:.1f}', 'MACD_Hist': '{:.2f}', 'Daily_Return': '{:.2%}'
            }), use_container_width=True, hide_index=True)

# ============================================
# تب ۲: سیگنال زنده
# ============================================
with tab2:
    if st.button("🔄 به‌روزرسانی تحلیل", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    df = load_data(symbol, days=200)
    model, scaler, feature_cols = load_model(symbol)
    
    if df is not None and model is not None:
        df_f = engineer_features(df)
        df_c = df_f.dropna().reset_index(drop=True)
        
        last_row = df_c.iloc[-1:][feature_cols]
        last_scaled = scaler.transform(last_row)
        pred = model.predict(last_scaled)[0]
        proba = model.predict_proba(last_scaled)[0]
        conf = max(proba) * 100
        
        cp = df_c['Close'].iloc[-1]
        atr = df_c['ATR_14'].iloc[-1]
        rsi = df_c['RSI_14'].iloc[-1]
        sma200 = df_c['Close'].rolling(200).mean().iloc[-1]
        
        sl = cp - (1.5 * atr)
        tp = cp + (3.0 * atr)
        
        # تعیین سیگنال
        if pred == 1 and conf > 60 and cp > sma200:
            action, sig_class = "🟢 LONG (خرید / لانگ)", "signal-long"
            lev = "5x-10x" if conf > 75 else ("3x-5x" if conf > 65 else "1x-2x")
            pos = "30-50%" if conf > 75 else ("20-30%" if conf > 65 else "10-20%")
        elif pred == 0 and conf > 60 and cp < sma200:
            action, sig_class = "🔴 SHORT (فروش تعهدی)", "signal-short"
            lev = "5x-10x" if conf > 75 else ("3x-5x" if conf > 65 else "1x-2x")
            pos = "30-50%" if conf > 75 else ("20-30%" if conf > 65 else "10-20%")
        else:
            action, sig_class = "🟡 NEUTRAL (صبر کن)", "signal-neutral"
            lev, pos = "0x (نقد)", "0%"
        
        # کارت سیگنال بزرگ
        st.markdown(f"""
        <div class="signal-card {sig_class}">
            <h2>{action}</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.1em; margin-top: 10px;">
                اطمینان مدل: <strong>{conf:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Chart برای اطمینان
        st.markdown('<div class="section-title">🎯 سطح اطمینان مدل</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf,
            number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': "#00d4aa" if conf > 65 else ("#ffc107" if conf > 50 else "#ff4d6d")},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 77, 109, 0.3)'},
                    {'range': [50, 65], 'color': 'rgba(255, 193, 7, 0.3)'},
                    {'range': [65, 100], 'color': 'rgba(0, 212, 170, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75, 'value': conf
                }
            }
        ))
        fig_gauge.update_layout(
            height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Vazirmatn')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # RSI Bar
        st.markdown('<div class="section-title">📊 وضعیت RSI</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rsi-container">
            <div style="display: flex; justify-content: space-between; color: #a0a8c0;">
                <span>اشباع فروش (0)</span>
                <span style="color: white; font-weight: bold; font-size: 1.2em;">RSI: {rsi:.1f}</span>
                <span>اشباع خرید (100)</span>
            </div>
            <div class="rsi-bar">
                <div class="rsi-indicator" style="left: {rsi}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # اطلاعات معامله
        if "NEUTRAL" not in action:
            st.markdown('<div class="section-title">💼 استراتژی فیوچرز</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="label">🔰 اهرم پیشنهادی</div>
                    <div class="value">{lev}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="label">💰 حجم پوزیشن</div>
                    <div class="value">{pos}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">🎯 نقاط ورود و خروج</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="glass-card" style="border-color: #667eea;">
                    <div class="label">🟢 نقطه ورود</div>
                    <div class="value">${cp:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                sl_p = ((sl - cp) / cp) * 100
                st.markdown(f"""
                <div class="glass-card" style="border-color: #ff4d6d;">
                    <div class="label">🔴 حد ضرر</div>
                    <div class="value">${sl:,.2f}</div>
                    <div class="delta delta-negative">{sl_p:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                tp_p = ((tp - cp) / cp) * 100
                st.markdown(f"""
                <div class="glass-card" style="border-color: #00d4aa;">
                    <div class="label">🟢 حد سود</div>
                    <div class="value">${tp:,.2f}</div>
                    <div class="delta delta-positive">+{tp_p:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.info("📐 **نسبت ریسک به ریوارد: 1:2** - این نسبت به معنای آن است که سود احتمالی ۲ برابر ضرر احتمالی است.")
        else:
            st.warning("⚠️ **توصیه:** در حال حاضر سیگنال معتبری وجود ندارد. نقد بمان و منتظر تایید روند باش.")
        
        # تحلیل تکنیکال
        st.markdown('<div class="section-title">📊 تحلیل تکنیکال</div>', unsafe_allow_html=True)
        trend = "📈 صعودی (بالای SMA 200)" if cp > sma200 else "📉 نزولی (زیر SMA 200)"
        rsi_s = "اشباع خرید ⚠️" if rsi > 70 else ("اشباع فروش ⚠️" if rsi < 30 else "خنثی ✓")
        
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='glass-card'><div class='label'>روند کلان</div><div style='color:white; font-size:1.1em;'>{trend}</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='glass-card'><div class='label'>وضعیت RSI</div><div style='color:white; font-size:1.1em;'>{rsi_s}</div></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='glass-card'><div class='label'>نوسان (ATR)</div><div style='color:white; font-size:1.1em;'>${atr:,.2f}</div></div>", unsafe_allow_html=True)

# ============================================
# تب ۳: بک‌تست
# ============================================
with tab3:
    st.markdown('<div class="section-title">📊 نتایج بک‌تست استراتژی</div>', unsafe_allow_html=True)
    
    ff = f"data/features/{symbol.replace('-', '_').lower()}_1d_features.csv"
    
    if os.path.exists(ff):
        df_bt = pd.read_csv(ff, parse_dates=['Date'])
        model, scaler, fcols = load_model(symbol)
        
        if model is not None:
            ts = int(len(df_bt) * 0.2)
            df_t = df_bt.iloc[-ts:].copy().reset_index(drop=True)
            
            X = scaler.transform(df_t[fcols])
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]
            
            df_t['Signal'], df_t['Conf'] = preds, probs
            df_t['SMA_200'] = df_bt['Close'].rolling(200).mean().iloc[-ts:].values
            
            df_t['Signal'] = df_t['Signal'] * (df_t['Close'] > df_t['SMA_200']).astype(int)
            
            pos_sz = np.where(df_t['Conf'] >= 0.75, 1.0,
                     np.where(df_t['Conf'] >= 0.55, (df_t['Conf'] - 0.55) / 0.20, 0.0))
            df_t['Pos'] = pos_sz * df_t['Signal']
            df_t.loc[df_t['Conf'] < 0.55, 'Signal'] = 0
            
            df_t['MR'] = df_t['Target_Next_Day_Return']
            df_t['SR'] = np.where(df_t['Signal'] == 1, df_t['MR'] * df_t['Pos'] - 0.001, 0.0)
            df_t['CM'] = (1 + df_t['MR']).cumprod()
            df_t['CS'] = (1 + df_t['SR']).cumprod()
            
            sr = (df_t['CS'].iloc[-1] - 1) * 100
            mr = (df_t['CM'].iloc[-1] - 1) * 100
            tr = df_t[df_t['Signal'] == 1]
            wr = (tr['MR'] > 0).mean() * 100 if len(tr) > 0 else 0
            dd = ((df_t['CS'] / df_t['CS'].cummax()) - 1).min() * 100
            
            # متریک‌های زیبا
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                ("📈 بازدهی استراتژی", f"{sr:+.2f}%", "delta-positive" if sr > 0 else "delta-negative"),
                ("📉 Buy & Hold", f"{mr:+.2f}%", "delta-positive" if mr > 0 else "delta-negative"),
                ("🎯 نرخ برد", f"{wr:.1f}%", "delta-positive" if wr > 50 else "delta-negative"),
                ("⚠️ Max Drawdown", f"{dd:.2f}%", "delta-negative")
            ]
            for col, (lbl, val, cls) in zip([c1, c2, c3, c4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="label">{lbl}</div>
                        <div class="value {cls}">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # نمودار رشد سرمایه
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_t['Date'], y=df_t['CS'], mode='lines',
                name='استراتژی هوشمند', line=dict(color='#00d4aa', width=3),
                fill='tozeroy', fillcolor='rgba(0, 212, 170, 0.1)'))
            fig.add_trace(go.Scatter(x=df_t['Date'], y=df_t['CM'], mode='lines',
                name='Buy & Hold', line=dict(color='#ffc107', width=2, dash='dash')))
            fig.update_layout(
                template='plotly_dark', height=500,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Vazirmatn', color='#a0a8c0'),
                legend=dict(bgcolor='rgba(0,0,0,0)')
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💼 **تعداد معاملات:** {len(tr)} | 📅 **دوره تست:** {len(df_t)} روز")
    else:
        st.warning("⚠️ فایل داده‌های بک‌تست یافت نشد. ابتدا `python main.py` را اجرا کنید.")

# ============================================
# تب ۴: درباره
# ============================================
with tab4:
    st.markdown('<div class="section-title">🎯 درباره این پلتفرم</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white;">🧠 یک سیستم کامل تحلیل هوشمند</h3>
        <p style="color: #a0a8c0; line-height: 1.8;">
        این پلتفرم با استفاده از <strong style="color: #667eea;">یادگیری ماشین (XGBoost)</strong> 
        و <strong style="color: #667eea;">۱۷ ویژگی تکنیکال</strong> ساخته شده است. 
        سیستم با سه لایه محافظتی (فیلتر روند، مدیریت سرمایه پویا، حد ضرر/سود) 
        ریسک معاملات را به حداقل می‌رساند.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">✨ قابلیت‌ها</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    features = [
        ("📊 داشبورد تعاملی", "نمودارهای زنده با RSI, MACD, Bollinger Bands"),
        ("🎯 سیگنال هوشمند", "پیش‌بینی روند با درصد اطمینان"),
        ("💼 مدیریت ریسک", "محاسبه خودکار SL/TP بر اساس ATR"),
        ("📈 بک‌تست حرفه‌ای", "ارزیابی عملکرد با Sharpe و Sortino Ratio"),
        ("📱 ریسپانسیو", "کار روی موبایل، تبلت و دسکتاپ"),
        ("🔒 سلب مسئولیت", "ابزار تحلیلی، نه توصیه مالی")
    ]
    for i, (title, desc) in enumerate(features):
        with c1 if i % 2 == 0 else c2:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #00d4aa; margin: 0;">{title}</h4>
                <p style="color: #a0a8c0; margin: 5px 0 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">🛠️ تکنولوژی‌ها</div>', unsafe_allow_html=True)
    techs = ["Python 3.14", "Streamlit", "XGBoost", "scikit-learn", 
             "pandas", "numpy", "yfinance", "Plotly"]
    cols = st.columns(4)
    for i, tech in enumerate(techs):
        with cols[i % 4]:
            st.markdown(f"<div class='glass-card' style='text-align:center;'><strong style='color:#667eea;'>{tech}</strong></div>", unsafe_allow_html=True)

# فوتر
st.markdown("""
<div class="footer">
    <p>🚀 ساخته شده با ❤️ توسط <strong>میثم شهرامی برای رفیقای خوب و دوست داشتنی خودم علی الخصوص حسین آقا</strong> | نسخه 3.0 Pro</p>
    <p style="font-size: 0.85em; margin-top: 10px;">
    ⚠️ این ابزار صرفاً جنبه آموزشی و تحلیلی دارد و توصیه مالی نیست.
    </p>
</div>
""", unsafe_allow_html=True)