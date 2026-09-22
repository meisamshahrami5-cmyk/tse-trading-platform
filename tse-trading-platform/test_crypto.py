import yfinance as yf

print("🔄 تست اتصال به یاهو فایننس (Crypto)...")

try:
    ticker = yf.Ticker("BTC-USD")
    # دریافت داده‌های 7 روز گذشته
    df = ticker.history(period="7d")
    
    if not df.empty:
        print("✅ اتصال موفقیت‌آمیز و دریافت داده!")
        print(f"💰 آخرین قیمت بیت‌کوین: ${df['Close'].iloc[-1]:,.2f}")
        print(f"📊 تعداد روزهای دریافت شده: {len(df)}")
    else:
        print("⚠️ داده‌ای دریافت نشد.")
        
except Exception as e:
    print(f"❌ خطا: {e}")