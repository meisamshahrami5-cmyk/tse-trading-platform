import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
from datetime import datetime

# افزودن مسیر ریشه پروژه به sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🚀 شروع بک‌تست استراتژی هوش مصنوعی...")

# 1. تنظیمات نماد
symbol = "btc-usd"
safe_symbol = symbol.replace('-', '_')
base_dir = os.path.dirname(os.path.abspath(__file__))
features_file = os.path.join(base_dir, f"data/features/{safe_symbol}_1d_features.csv")

# پیدا کردن فایل مدل (با در نظر گرفتن حروف بزرگ و کوچک)
model_dir = os.path.join(base_dir, "models")

if not os.path.exists(model_dir) or not os.path.exists(features_file):
    print("❌ خطا: فایل داده‌ها یا مدل پیدا نشد. لطفاً ابتدا python main.py را اجرا کنید.")
    exit()

model_files = [f for f in os.listdir(model_dir) if safe_symbol in f.lower() and "xgboost" in f.lower()]

if not model_files:
    print("❌ خطا: فایل مدل پیدا نشد. لطفاً ابتدا python main.py را اجرا کنید.")
    exit()

model_file = os.path.join(model_dir, model_files[0])
scaler_file = model_file.replace("XGBoost", "scaler").replace("xgboost", "scaler")
features_list_file = model_file.replace("XGBoost", "features").replace("xgboost", "features")

print(f"📂 بارگذاری داده‌ها از: {features_file}")
print(f"🧠 بارگذاری مدل از: {model_file}")

# 2. بارگذاری داده‌ها و مدل
df = pd.read_csv(features_file, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
feature_cols = joblib.load(features_list_file)

# 3. جداسازی داده‌های تست (۲۰٪ انتهایی برای شبیه‌سازی دنیای واقعی)
test_size = int(len(df) * 0.2)
df_test = df.iloc[-test_size:].copy().reset_index(drop=True)

print(f"📊 تعداد روزهای مورد بک‌تست: {len(df_test)} روز")

# 4. تولید سیگنال‌های مدل
X_test = df_test[feature_cols]
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)

df_test['Model_Signal'] = predictions  # 1 = صعودی, 0 = نزولی

# 5. محاسبه بازدهی استراتژی
# اگر مدل سیگنال خرید (1) بدهد، ما بازدهی روز بعد را کسب می‌کنیم. در غیر این صورت 0 (نقد).
# کارمزد صرافی را هم حدود 0.1% در نظر می‌گیریم.
fee = 0.001
df_test['Market_Return'] = df_test['Target_Next_Day_Return']
df_test['Strategy_Return'] = np.where(df_test['Model_Signal'] == 1, 
                                      df_test['Market_Return'] - fee, 
                                      0.0)

# محاسبه رشد تجمعی سرمایه (Equity Curve)
df_test['Cumulative_Market'] = (1 + df_test['Market_Return']).cumprod()
df_test['Cumulative_Strategy'] = (1 + df_test['Strategy_Return']).cumprod()

# 6. محاسبه متریک‌های کلیدی
total_market_return = (df_test['Cumulative_Market'].iloc[-1] - 1) * 100
total_strategy_return = (df_test['Cumulative_Strategy'].iloc[-1] - 1) * 100

winning_trades = df_test[df_test['Model_Signal'] == 1]
win_rate = (winning_trades['Market_Return'] > 0).mean() * 100 if len(winning_trades) > 0 else 0

# محاسبه حداکثر افت سرمایه (Max Drawdown)
rolling_max = df_test['Cumulative_Strategy'].cummax()
drawdown = (df_test['Cumulative_Strategy'] / rolling_max) - 1
max_drawdown = drawdown.min() * 100

print("\n" + "="*50)
print("📊 نتایج بک‌تست (دوره تست)")
print("="*50)
print(f"📈 بازدهی استراتژی مدل: {total_strategy_return:+.2f}%")
print(f"📉 بازدهی خرید و نگهداری (Buy & Hold): {total_market_return:+.2f}%")
print(f"🎯 نرخ برد معاملات (Win Rate): {win_rate:.2f}%")
print(f"⚠️ حداکثر افت سرمایه (Max Drawdown): {max_drawdown:.2f}%")
print(f"💼 تعداد سیگنال‌های خرید صادر شده: {len(winning_trades)}")
print("="*50)

# 7. رسم نمودار رشد سرمایه
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_test['Date'], 
    y=df_test['Cumulative_Strategy'], 
    mode='lines', 
    name='سرمایه استراتژی هوش مصنوعی',
    line=dict(color='#00FF00', width=2)
))

fig.add_trace(go.Scatter(
    x=df_test['Date'], 
    y=df_test['Cumulative_Market'], 
    mode='lines', 
    name='سرمایه خرید و نگهداری (Buy & Hold)',
    line=dict(color='#FFA500', width=2, dash='dash')
))

fig.update_layout(
    title=f'نمودار رشد سرمایه (Backtest) - {symbol.upper()}',
    xaxis_title='تاریخ',
    yaxis_title='رشد سرمایه (ضریب)',
    template='plotly_dark',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99)
)

print("\n🌐 در حال باز کردن نمودار رشد سرمایه در مرورگر...")
fig.show()
