"""
ماژول مشترک اندیکاتورها و مهندسی ویژگی
تمام فایل‌های پروژه باید از این ماژول استفاده کنند تا از ناهماهنگی جلوگیری شود.
"""
import pandas as pd
import numpy as np


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """محاسبه میانگین متحرک ساده"""
    return series.rolling(window=period).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """محاسبه میانگین متحرک نمایی"""
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """محاسبه شاخص قدرت نسبی (RSI) با روش Wilder"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """محاسبه MACD - برمی‌گرداند: (macd_line, signal_line, histogram)"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bbands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """محاسبه باندهای بولینگر - برمی‌گرداند: (upper, middle, lower)"""
    middle = calc_sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """محاسبه Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    محاسبه تمام ویژگی‌های تکنیکال برای یک دیتافریم OHLCV.
    این تابع واحد باید در تمام بخش‌های پروژه (آموزش، پیش‌بینی، بک‌تست) استفاده شود.
    
    ستون‌های مورد نیاز ورودی: Date, Open, High, Low, Close, Volume
    """
    df = df.copy()
    
    # === میانگین‌های متحرک ===
    df['SMA_20'] = calc_sma(df['Close'], 20)
    df['SMA_50'] = calc_sma(df['Close'], 50)
    
    # === RSI ===
    df['RSI_14'] = calc_rsi(df['Close'], 14)
    
    # === MACD ===
    macd_line, signal_line, histogram = calc_macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = histogram
    
    # === باندهای بولینگر ===
    bb_upper, bb_middle, bb_lower = calc_bbands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    
    # === بازدهی ===
    df['Daily_Return'] = df['Close'].pct_change()
    df['Daily_Return_2'] = df['Close'].pct_change(2)
    df['Daily_Return_5'] = df['Close'].pct_change(5)
    
    # === نسبت‌های قیمتی (نرمال‌سازی شده) ===
    df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
    df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
    df['SMA20_to_SMA50'] = df['SMA_20'] / df['SMA_50']
    
    # === نوسان ===
    df['Volatility_7'] = df['Daily_Return'].rolling(7).std()
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
    
    # === حجم نسبی ===
    df['Volume_to_Avg20'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # === موقعیت در باندهای بولینگر ===
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # === مومنتوم ===
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    # === ATR (میانگین محدوده واقعی) ===
    df['ATR_14'] = calc_atr(df, 14)
    
    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    اضافه کردن ستون‌های هدف برای آموزش مدل.
    فقط در فاز آموزش استفاده شود (نه در پیش‌بینی زنده).
    """
    df = df.copy()
    df['Target_Next_Day_Return'] = (df['Close'].shift(-1) / df['Close']) - 1
    df['Target_Is_Up'] = (df['Target_Next_Day_Return'] > 0).astype(int)
    return df
