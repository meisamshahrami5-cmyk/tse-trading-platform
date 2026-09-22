"""
ماژول مهندسی ویژگی نسخه 3.0
استفاده از توابع مشترک اندیکاتورها برای جلوگیری از ناهماهنگی
"""
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.helpers import load_config, ensure_directory, print_separator
from src.indicators import engineer_features, add_target_columns


class FeatureEngineer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.processed_path = self.config['data']['processed_data_path']
        self.features_path = "data/features"
        ensure_directory(self.features_path)

    def process_symbol(self, symbol, interval='1d'):
        safe_symbol = symbol.replace('-', '_')
        input_file = f"{self.processed_path}/{safe_symbol}_{interval}_processed.csv"
        
        print(f"⚙️ در حال محاسبه ویژگی‌های هوشمند برای {symbol.upper()}...")
        
        try:
            df = pd.read_csv(input_file, parse_dates=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # استفاده از تابع مشترک engineer_features
            df = engineer_features(df)
            
            # اضافه کردن ستون‌های هدف (فقط برای آموزش)
            df = add_target_columns(df)

            # حذف NaNها
            initial_rows = len(df)
            df = df.dropna().reset_index(drop=True)
            dropped_rows = initial_rows - len(df)
            
            print(f"✅ محاسبات کامل شد. ({dropped_rows} ردیف حذف شد)")
            print(f"📊 تعداد ویژگی‌های نهایی: {len(df.columns)} ستون")
            
            output_file = f"{self.features_path}/{safe_symbol}_{interval}_features.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"💾 ذخیره شد: {output_file}")
            
            return df

        except Exception as e:
            print(f"❌ خطا در پردازش {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def run_engineering(self, symbols=None, interval='1d'):
        if symbols is None:
            symbols = self.config['data']['symbols']
            
        print_separator("شروع فاز 2 (نسخه 3.0): ویژگی‌های هوشمند")
        
        results = {}
        for symbol in symbols:
            print_separator(f"پردازش: {symbol.upper()}")
            df = self.process_symbol(symbol, interval)
            if not df.empty:
                results[symbol] = df
                
        print_separator("فاز 2 کامل شد")
        print(f"✅ {len(results)} نماد با موفقیت پردازش شدند.")
        return results


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run_engineering()
