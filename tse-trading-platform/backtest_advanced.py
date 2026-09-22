import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🚀 شروع بک‌تست پیشرفته (نسخه 2.0 - با سه لایه محافظتی)...")

# ============================================
# تنظیمات استراتژی (قابل تنظیم)
# ============================================
symbol = "btc-usd"
safe_symbol = symbol.replace('-', '_')
features_file = f"data/features/{safe_symbol}_1d_features.csv"

# تنظیمات فیلتر روند کلان
USE_TREND_FILTER = True
TREND_SMA_PERIOD = 200  # فقط وقتی قیمت بالای SMA 200 است، خرید کن

# تنظیمات مدیریت سرمایه پویا
USE_POSITION_SIZING = True
MIN_CONFIDENCE = 0.55  # حداقل اطمینان برای ورود (55%)
MAX_CONFIDENCE = 0.75  # حداکثر اطمینان برای ورود کامل (75%)

# تنظیمات حد ضرر و حد سود
USE_STOP_LOSS = True
STOP_LOSS_PCT = -0.03  # حد ضرر 3%
TAKE_PROFIT_PCT = 0.05  # حد سود 5%

# کارمزد صرافی
FEE = 0.001  # 0.1%

# ============================================
# بارگذاری داده‌ها و مدل
# ============================================
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if safe_symbol in f.lower() and "xgboost" in f.lower()]

if not model_files or not os.path.exists(features_file):
    print("❌ خطا: فایل داده‌ها یا مدل پیدا نشد.")
    exit()

model_file = f"models/{model_files[0]}"
scaler_file = model_file.replace("XGBoost", "scaler").replace("xgboost", "scaler")
features_list_file = model_file.replace("XGBoost", "features").replace("xgboost", "features")

print(f"📂 بارگذاری داده‌ها از: {features_file}")
print(f"🧠 بارگذاری مدل از: {model_file}")

df = pd.read_csv(features_file, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
feature_cols = joblib.load(features_list_file)

# ============================================
# جداسازی داده‌های تست و تولید سیگنال‌ها
# ============================================
test_size = int(len(df) * 0.2)
df_test = df.iloc[-test_size:].copy().reset_index(drop=True)

print(f"📊 تعداد روزهای مورد بک‌تست: {len(df_test)} روز")

# تولید سیگنال‌های اولیه و اطمینان مدل
X_test = df_test[feature_cols]
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]  # احتمال صعودی بودن

df_test['Model_Signal'] = predictions
df_test['Model_Confidence'] = probabilities

# ============================================
# لایه ۱: فیلتر روند کلان (Trend Filter)
# ============================================
if USE_TREND_FILTER:
    print(f"🛡️ لایه ۱: فیلتر روند کلان فعال (SMA {TREND_SMA_PERIOD})")
    # محاسبه SMA 200 روی کل داده‌ها (نه فقط تست)
    df['SMA_200'] = df['Close'].rolling(window=TREND_SMA_PERIOD).mean()
    df_test['SMA_200'] = df['SMA_200'].iloc[-test_size:].values
    
    # فقط سیگنال‌های خریدی را نگه دار که قیمت بالای SMA 200 است
    trend_filter = df_test['Close'] > df_test['SMA_200']
    df_test['Model_Signal'] = df_test['Model_Signal'] * trend_filter.astype(int)
    
    filtered_count = (df_test['Model_Signal'] == 1).sum()
    print(f"   ✅ {filtered_count} سیگنال پس از فیلتر روند باقی ماند")

# ============================================
# لایه ۲: مدیریت سرمایه پویا (Position Sizing)
# ============================================
if USE_POSITION_SIZING:
    print(f"⚖️ لایه ۲: مدیریت سرمایه پویا فعال (اطمینان {MIN_CONFIDENCE*100:.0f}%-{MAX_CONFIDENCE*100:.0f}%)")
    
    # محاسبه اندازه پوزیشن بر اساس اطمینان مدل
    position_size = np.where(
        df_test['Model_Confidence'] >= MAX_CONFIDENCE,
        1.0,  # ورود کامل
        np.where(
            df_test['Model_Confidence'] >= MIN_CONFIDENCE,
            (df_test['Model_Confidence'] - MIN_CONFIDENCE) / (MAX_CONFIDENCE - MIN_CONFIDENCE),
            0.0  # عدم ورود
        )
    )
    
    # اعمال اندازه پوزیشن روی سیگنال‌ها
    df_test['Position_Size'] = position_size * df_test['Model_Signal']
    
    # سیگنال‌هایی که اطمینان کافی ندارند را حذف کن
    df_test.loc[df_test['Model_Confidence'] < MIN_CONFIDENCE, 'Model_Signal'] = 0
    
    avg_position = df_test[df_test['Position_Size'] > 0]['Position_Size'].mean()
    print(f"   ✅ میانگین اندازه پوزیشن: {avg_position*100:.1f}%")

# ============================================
# لایه ۳: حد ضرر و حد سود (Stop Loss / Take Profit)
# ============================================
if USE_STOP_LOSS:
    print(f"🎯 لایه ۳: حد ضرر ({STOP_LOSS_PCT*100:.1f}%) و حد سود ({TAKE_PROFIT_PCT*100:.1f}%) فعال")

# ============================================
# شبیه‌سازی معاملات با در نظر گرفتن همه لایه‌ها
# ============================================
print("\n⚙️ در حال شبیه‌سازی معاملات...")

# بازدهی بازار
df_test['Market_Return'] = df_test['Target_Next_Day_Return']

# بازدهی استراتژی با در نظر گرفتن اندازه پوزیشن و کارمزد
df_test['Strategy_Return'] = np.where(
    df_test['Model_Signal'] == 1,
    (df_test['Market_Return'] * df_test['Position_Size']) - (FEE * df_test['Position_Size']),
    0.0
)

# اعمال حد ضرر و حد سود (شبیه‌سازی ساده)
if USE_STOP_LOSS:
    # اگر بازدهی روزانه کمتر از حد ضرر بود، آن را محدود کن
    df_test.loc[df_test['Market_Return'] < STOP_LOSS_PCT, 'Strategy_Return'] = np.where(
        df_test.loc[df_test['Market_Return'] < STOP_LOSS_PCT, 'Model_Signal'] == 1,
        STOP_LOSS_PCT * df_test.loc[df_test['Market_Return'] < STOP_LOSS_PCT, 'Position_Size'],
        0.0
    )
    
    # اگر بازدهی بیشتر از حد سود بود، آن را محدود کن
    df_test.loc[df_test['Market_Return'] > TAKE_PROFIT_PCT, 'Strategy_Return'] = np.where(
        df_test.loc[df_test['Market_Return'] > TAKE_PROFIT_PCT, 'Model_Signal'] == 1,
        TAKE_PROFIT_PCT * df_test.loc[df_test['Market_Return'] > TAKE_PROFIT_PCT, 'Position_Size'],
        0.0
    )

# رشد تجمعی سرمایه
df_test['Cumulative_Market'] = (1 + df_test['Market_Return']).cumprod()
df_test['Cumulative_Strategy'] = (1 + df_test['Strategy_Return']).cumprod()

# ============================================
# محاسبه متریک‌های پیشرفته
# ============================================
total_market_return = (df_test['Cumulative_Market'].iloc[-1] - 1) * 100
total_strategy_return = (df_test['Cumulative_Strategy'].iloc[-1] - 1) * 100

# نرخ برد
winning_trades = df_test[df_test['Model_Signal'] == 1]
win_rate = (winning_trades['Market_Return'] > 0).mean() * 100 if len(winning_trades) > 0 else 0

# حداکثر افت سرمایه
rolling_max = df_test['Cumulative_Strategy'].cummax()
drawdown = (df_test['Cumulative_Strategy'] / rolling_max) - 1
max_drawdown = drawdown.min() * 100

# Sharpe Ratio (نسبت شارپ - بازدهی تعدیل شده با ریسک)
daily_rf = 0.0  # نرخ بدون ریسک روزانه (فرض صفر)
excess_returns = df_test['Strategy_Return'] - daily_rf
sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

# Sortino Ratio (فقط ریسک نزولی)
downside_returns = excess_returns[excess_returns < 0]
sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

# Profit Factor
gross_profit = winning_trades[winning_trades['Market_Return'] > 0]['Market_Return'].sum()
gross_loss = abs(winning_trades[winning_trades['Market_Return'] <= 0]['Market_Return'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

# میانگین سود و ضرر
avg_win = winning_trades[winning_trades['Market_Return'] > 0]['Market_Return'].mean() * 100 if len(winning_trades[winning_trades['Market_Return'] > 0]) > 0 else 0
avg_loss = winning_trades[winning_trades['Market_Return'] <= 0]['Market_Return'].mean() * 100 if len(winning_trades[winning_trades['Market_Return'] <= 0]) > 0 else 0

# ============================================
# نمایش نتایج
# ============================================
print("\n" + "="*60)
print("📊 نتایج بک‌تست پیشرفته (با سه لایه محافظتی)")
print("="*60)
print(f"📈 بازدهی استراتژی: {total_strategy_return:+.2f}%")
print(f"📉 بازدهی Buy & Hold: {total_market_return:+.2f}%")
print(f"🎯 نرخ برد: {win_rate:.2f}%")
print(f"⚠️ حداکثر افت سرمایه: {max_drawdown:.2f}%")
print(f"💼 تعداد معاملات: {len(winning_trades)}")
print("-"*60)
print("📊 متریک‌های پیشرفته:")
print(f"   📐 Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"   📐 Sortino Ratio: {sortino_ratio:.2f}")
print(f"   💰 Profit Factor: {profit_factor:.2f}")
print(f"   📈 میانگین سود: {avg_win:+.2f}%")
print(f"   📉 میانگین ضرر: {avg_loss:.2f}%")
print("="*60)

# ============================================
# رسم نمودارهای حرفه‌ای
# ============================================
print("\n🌐 در حال رسم نمودارها...")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.5, 0.3, 0.2],
    subplot_titles=(
        'نمودار رشد سرمایه',
        'افت سرمایه (Drawdown)',
        'اندازه پوزیشن'
    )
)

# نمودار ۱: رشد سرمایه
fig.add_trace(
    go.Scatter(
        x=df_test['Date'],
        y=df_test['Cumulative_Strategy'],
        mode='lines',
        name='استراتژی هوشمند',
        line=dict(color='#00FF00', width=2)
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_test['Date'],
        y=df_test['Cumulative_Market'],
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#FFA500', width=2, dash='dash')
    ),
    row=1, col=1
)

# نمودار ۲: Drawdown
fig.add_trace(
    go.Scatter(
        x=df_test['Date'],
        y=drawdown * 100,
        mode='lines',
        name='Drawdown %',
        fill='tozeroy',
        line=dict(color='#FF4444', width=1)
    ),
    row=2, col=1
)

# نمودار ۳: اندازه پوزیشن
fig.add_trace(
    go.Scatter(
        x=df_test['Date'],
        y=df_test['Position_Size'] * 100,
        mode='lines',
        name='اندازه پوزیشن %',
        fill='tozeroy',
        line=dict(color='#4444FF', width=1)
    ),
    row=3, col=1
)

fig.update_layout(
    title=f'بک‌تست پیشرفته - {symbol.upper()} (با سه لایه محافظتی)',
    template='plotly_dark',
    height=800,
    showlegend=True,
    legend=dict(x=0.01, y=0.99)
)

fig.update_yaxes(title_text='رشد سرمایه', row=1, col=1)
fig.update_yaxes(title_text='Drawdown %', row=2, col=1)
fig.update_yaxes(title_text='پوزیشن %', row=3, col=1)

fig.show()
