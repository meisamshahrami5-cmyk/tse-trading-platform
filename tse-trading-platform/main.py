"""
نقطه ورود اصلی پروژه TSE Trading Platform
"""
import sys
from src.data_collection.crypto_data_pipeline import CryptoDataPipeline
from src.indicators.feature_engineering import FeatureEngineer
from src.ml.model_trainer import ModelTrainer
from src.utils.helpers import print_separator

def main():
    print_separator("TSE Trading Platform - شروع عملیات", char="*")
    
    try:
        # ================= فاز 1 =================
        print("\n🚀 شروع فاز 1: دریافت داده‌های خام")
        pipeline = CryptoDataPipeline()
        raw_results = pipeline.run_pipeline()
        
        if not raw_results:
            print("\n❌ فاز 1 ناموفق بود. دریافت داده متوقف شد.")
            sys.exit(1)
            
        # ================= فاز 2 =================
        print("\n🚀 شروع فاز 2: مهندسی ویژگی و اندیکاتورها")
        engineer = FeatureEngineer()
        feature_results = engineer.run_engineering()
        
        if not feature_results:
            print("\n❌ فاز 2 ناموفق بود.")
            sys.exit(1)
            
        # ================= فاز 3 =================
        print("\n🚀 شروع فاز 3: آموزش مدل‌های هوش مصنوعی")
        trainer = ModelTrainer()
        model_results = trainer.train_all_models()
        
        if model_results:
            print("\n🎉 تبریک! سیستم کامل آماده استفاده است.")
            print("📂 مدل‌های آموزش‌دیده در پوشه models/ ذخیره شدند.")
            print("🎯 قدم بعدی: فاز 4 (ساخت API و رابط کاربری)")
        else:
            print("\n❌ فاز 3 ناموفق بود.")
            
    except Exception as e:
        print(f"\n❌ خطای غیرمنتظره: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
