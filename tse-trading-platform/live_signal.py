import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
import sys
from datetime import datetime, timedelta

# افزودن مسیر ریشه پروژه به sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.indicators import engineer_features

def get_live_signal(symbol="btc-usd"):
    """دریافت سیگنال زنده با مدیریت خطا و کش کردن داده‌ها"""
    print(f"\n🔍 در حال تحلیل زنده {symbol.upper()}...")
    print("="*70)
    
    try:
        # تبدیل فرمت نماد برای yfinance
        yf_symbol = symbol.replace('-', '_').upper().replace('_', '-')
        if not yf_symbol.endswith('-USD'):
            yf_symbol = symbol.upper() if '-' in symbol else f"{symbol}-USD"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=200)
        
        print(f"📡 در حال دریافت داده از yfinance ({yf_symbol})...")
        df = yf.download(yf_symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if df.empty:
            print("❌ خطا: داده‌ای دریافت نشد. اتصال اینترنت را بررسی کنید.")
            return
        
        # استانداردسازی ستون‌ها
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df.columns = [str(col).strip().title() for col in df.columns]
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        
        # حذف داده‌های ناقص
        df = df.dropna(subset=['Close', 'Volume'])
        df = df[df['Volume'] > 0]
        
        if len(df) < 50:
            print(f"❌ خطا: داده کافی نیست (فقط {len(df)} رکورد). حداقل 50 رکورد نیاز است.")
            return
        
        print(f"✅ {len(df)} رکورد دریافت شد.")
        
        # محاسبه ویژگی‌ها با تابع مشترک
        print("⚙️ در حال محاسبه اندیکاتورهای تکنیکال...")
        df_features = engineer_features(df)
        df_clean = df_features.dropna().reset_index(drop=True)
        
        if df_clean.empty:
            print("❌ خطا: پس از محاسبه اندیکاتورها داده‌ای باقی نماند.")
            return
        
        # بارگذاری مدل
        safe_symbol = symbol.replace('-', '_').lower()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")
        
        if not os.path.exists(model_dir):
            print(f"❌ خطا: پوشه models یافت نشد. ابتدا python main.py را اجرا کنید.")
            return
        
        model_files = [f for f in os.listdir(model_dir) if safe_symbol in f.lower() and "xgboost" in f.lower()]
        
        if not model_files:
            print(f"❌ خطا: مدل برای {symbol} یافت نشد. ابتدا python main.py را اجرا کنید.")
            return
        
        model_file = os.path.join(model_dir, model_files[0])
        scaler_file = model_file.replace("XGBoost", "scaler").replace("xgboost", "scaler")
        features_file = model_file.replace("XGBoost", "features").replace("xgboost", "features")
        
        if not all(os.path.exists(x) for x in [model_file, scaler_file, features_file]):
            print(f"❌ خطا: فایل‌های مدل کامل نیستند.")
            return
        
        print("🧠 در حال بارگذاری مدل...")
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        feature_cols = joblib.load(features_file)
        
        # بررسی وجود ستون‌های مورد نیاز
        missing_cols = [col for col in feature_cols if col not in df_clean.columns]
        if missing_cols:
            print(f"❌ خطا: ستون‌های مورد نیاز مدل وجود ندارند: {missing_cols}")
            return
        
        # پیش‌بینی
        last_row = df_clean.iloc[-1:][feature_cols]
        last_row_scaled = scaler.transform(last_row)
        prediction = model.predict(last_row_scaled)[0]
        proba = model.predict_proba(last_row_scaled)[0]
        confidence = max(proba) * 100
        
        # محاسبه شاخص‌های فعلی
        current_price = df_clean['Close'].iloc[-1]
        atr = df_clean['ATR_14'].iloc[-1]
        rsi = df_clean['RSI_14'].iloc[-1]
        
        # محاسبه SMA_200 برای فیلتر روند
        sma_200 = df_clean['Close'].rolling(200).mean().iloc[-1]
        trend_status = "صعودی (بالای SMA 200)" if current_price > sma_200 else "نزولی (زیر SMA 200)"
        
        # محاسبه نقاط ورود و خروج
        stop_loss_price = current_price - (1.5 * atr)
        take_profit_price = current_price + (3.0 * atr)
        stop_loss_pct = ((stop_loss_price - current_price) / current_price) * 100
        take_profit_pct = ((take_profit_price - current_price) / current_price) * 100
        
        # تعیین سیگنال نهایی
        if prediction == 1 and confidence > 60 and current_price > sma_200:
            action = "🟢 LONG (خرید / لانگ)"
            leverage = "5x - 10x" if confidence > 75 else ("3x - 5x" if confidence > 65 else "1x - 2x")
            position_size = "30% - 50%" if confidence > 75 else ("20% - 30%" if confidence > 65 else "10% - 20%")
        elif prediction == 0 and confidence > 60 and current_price < sma_200:
            action = "🔴 SHORT (فروش تعهدی / شورت)"
            leverage = "5x - 10x" if confidence > 75 else ("3x - 5x" if confidence > 65 else "1x - 2x")
            position_size = "30% - 50%" if confidence > 75 else ("20% - 30%" if confidence > 65 else "10% - 20%")
        else:
            action = "🟡 NEUTRAL (صبر کن - معامله نکن)"
            leverage = "0x (نقد بمان)"
            position_size = "0%"
        
        # نمایش نتایج
        print(f"\n📊 گزارش تحلیل لحظه‌ای - {symbol.upper()}")
        print(f"🕐 زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"💰 قیمت فعلی: ${current_price:,.2f}")
        print(f"📈 روند کلان: {trend_status}")
        print(f"📊 RSI (14): {rsi:.1f} | ATR (نوسان): ${atr:,.2f}")
        print("="*70)
        print(f"🎯 سیگنال هوش مصنوعی: {action}")
        print(f"🎯 اطمینان مدل: {confidence:.1f}%")
        print("="*70)
        
        if "NEUTRAL" not in action:
            print(f"💼 استراتژی فیوچرز پیشنهادی:")
            print(f"   🔰 اهرم: {leverage} | حجم پوزیشن: {position_size} سرمایه")
            print(f"🎯 نقاط ورود و خروج:")
            print(f"   🟢 ورود: ${current_price:,.2f}")
            print(f"   🔴 حد ضرر (SL): ${stop_loss_price:,.2f} ({stop_loss_pct:.2f}%)")
            print(f"   🟢 حد سود (TP): ${take_profit_price:,.2f} (+{take_profit_pct:.2f}%)")
            print(f"   📐 ریسک به ریوارد: 1:2")
        else:
            print(f"⚠️ توصیه: سیگنال معتبری نیست. نقد بمان و صبر کن.")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ خطای غیرمنتظره: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    get_live_signal("btc-usd")