"""
ماژول دریافت و پردازش داده‌های ارز دیجیتال
استفاده از yfinance با فرمت صحیح نمادها (پایدارترین روش در ایران)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.helpers import load_config, ensure_directory, print_separator


class CryptoDataPipeline:
    """کلاس مدیریت دریافت داده‌های ارز دیجیتال"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.raw_data_path = self.config['data']['raw_data_path']
        self.processed_data_path = self.config['data']['processed_data_path']
        ensure_directory(self.raw_data_path)
        ensure_directory(self.processed_data_path)

    def fetch_crypto_data(self, symbol: str, days_back: int = None, interval: str = '1d') -> pd.DataFrame:
        if days_back is None:
            days_back = self.config['data']['days_back']

        print(f"🔄 در حال دریافت داده‌های {symbol}...")

        try:
            yf_symbol = symbol.replace('-usdt', '-USD').upper()
            ticker = yf.Ticker(yf_symbol)
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            df = ticker.history(start=start_date, interval=interval)

            if df.empty:
                raise ValueError(f"داده‌ای برای {yf_symbol} یافت نشد")

            print(f"✅ {len(df)} رکورد دریافت شد.")
            print("🧹 در حال تمیز کردن داده‌ها...")

            df = df.reset_index()
            df.columns = [str(col).replace('(UTC)', '').strip() for col in df.columns]
            
            cols_to_drop = [col for col in ['Dividends', 'Stock Splits'] if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            df = df.dropna()
            df = df[df['Volume'] > 0]
            df = df.sort_values('Date', ascending=True).reset_index(drop=True)

            print(f"✅ {len(df)} رکورد پس از تمیز کردن باقی ماند.")
            return df

        except Exception as e:
            print(f"❌ خطا در دریافت داده‌ها: {str(e)}")
            return pd.DataFrame()

    def save_raw_data(self, df: pd.DataFrame, symbol: str, interval: str):
        if df.empty:
            return
        safe_symbol = symbol.replace('-', '_')
        filename = f"{self.raw_data_path}/{safe_symbol}_{interval}_raw.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"💾 ذخیره شد: {filename}")

    def save_processed_data(self, df: pd.DataFrame, symbol: str, interval: str):
        if df.empty:
            return
        safe_symbol = symbol.replace('-', '_')
        filename = f"{self.processed_data_path}/{safe_symbol}_{interval}_processed.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"💾 ذخیره شد: {filename}")

    def print_statistics(self, df: pd.DataFrame, symbol: str):
        if df.empty:
            return
        
        print(f"\n📊 آمار {symbol.upper()}:")
        print(f"   - تعداد کندل‌ها: {len(df)}")
        
        start_date = str(df['Date'].iloc[0]).split(' ')[0]
        end_date = str(df['Date'].iloc[-1]).split(' ')[0]
        print(f"   - بازه زمانی: از {start_date} تا {end_date}")
        
        print(f"   - میانگین حجم: {df['Volume'].mean():,.2f}")
        print(f"   - بالاترین قیمت: {df['High'].max():,.2f}")
        print(f"   - پایین‌ترین قیمت: {df['Low'].min():,.2f}")
        print(f"   - آخرین قیمت: {df['Close'].iloc[-1]:,.2f}")
        
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        print(f"   - تغییر قیمت در این بازه: {price_change:+.2f}%")

    def run_pipeline(self, symbols: list = None, interval: str = '1d'):
        if symbols is None:
            symbols = self.config['data']['symbols']

        print_separator("شروع پایپلاین دریافت داده‌ها (yfinance - پایدار)")

        results = {}
        for symbol in symbols:
            print_separator(f"پردازش نماد: {symbol.upper()}")
            df = self.fetch_crypto_data(symbol, interval=interval)

            if not df.empty:
                self.save_raw_data(df, symbol, interval)
                self.save_processed_data(df, symbol, interval)
                self.print_statistics(df, symbol)
                results[symbol] = df
            else:
                print(f"⚠️ نماد {symbol.upper()} پردازش نشد.")

            time.sleep(1)

        print_separator("پایپلاین کامل شد")
        print(f"✅ {len(results)} از {len(symbols)} نماد با موفقیت پردازش شدند.")
        return results


if __name__ == "__main__":
    pipeline = CryptoDataPipeline()
    results = pipeline.run_pipeline()
