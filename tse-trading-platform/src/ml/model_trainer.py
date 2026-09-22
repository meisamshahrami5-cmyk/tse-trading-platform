"""
ماژول آموزش مدل هوش مصنوعی نسخه 2.0
با فیلتر هوشمند ویژگی‌ها و استانداردسازی
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.helpers import load_config, ensure_directory, print_separator


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.features_path = "data/features"
        self.models_path = "models"
        ensure_directory(self.models_path)
        
        # لیست ستون‌هایی که نباید به عنوان ویژگی وارد مدل شوند
        # توجه: ATR_14 شامل می‌شود زیرا در engineer_features مشترک محاسبه می‌گردد
        self.exclude_cols = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Value',
            'Target_Next_Day_Return', 'Target_Is_Up',
            'SMA_20', 'SMA_50',  # این‌ها در نسبت‌ها استفاده شده‌اند
            'BB_Upper', 'BB_Middle', 'BB_Lower',  # این‌ها هم در BB_Position
            'MACD', 'MACD_Signal'  # MACD_Hist کافی است
        ]
        
        self.use_xgboost = False
        try:
            from xgboost import XGBClassifier
            self.use_xgboost = True
            print("✅ XGBoost فعال است")
        except ImportError:
            print("⚠️ از Random Forest استفاده می‌شود")

    def _prepare_data(self, df):
        # فقط ستون‌های اندیکاتور و ویژگی‌های هوشمند را نگه می‌داریم
        feature_cols = [col for col in df.columns if col not in self.exclude_cols]
        
        X = df[feature_cols]
        y = df['Target_Is_Up']
        
        return X, y, feature_cols

    def train_model(self, symbol, interval='1d'):
        safe_symbol = symbol.replace('-', '_')
        input_file = f"{self.features_path}/{safe_symbol}_{interval}_features.csv"
        
        print(f"🧠 در حال آموزش مدل برای {symbol.upper()}...")
        
        try:
            df = pd.read_csv(input_file)
            print(f"   📊 تعداد نمونه‌ها: {len(df)}")
            
            X, y, feature_cols = self._prepare_data(df)
            print(f"   🎯 تعداد ویژگی‌های انتخاب شده: {len(feature_cols)}")
            
            # استانداردسازی ویژگی‌ها
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # تقسیم داده (بدون shuffle برای حفظ ترتیب زمانی)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            print(f"   📈 آموزش: {len(X_train)} | 📉 تست: {len(X_test)}")
            
            # ساخت مدل
            if self.use_xgboost:
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                )
                model_name = "XGBoost"
            else:
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1
                )
                model_name = "RandomForest"
            
            print(f"   ⚙️ آموزش مدل {model_name}...")
            model.fit(X_train, y_train)
            
            # ارزیابی
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n   🎯 دقت مدل: {accuracy * 100:.2f}%")
            print(f"\n   📋 گزارش طبقه‌بندی:")
            print(classification_report(y_test, y_pred, target_names=['پایین', 'بالا'], zero_division=0))
            
            # اهمیت ویژگی‌ها
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"   🏆 ۵ ویژگی مهم:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"      {row['Feature']}: {row['Importance']:.4f}")
            
            # ذخیره مدل و scaler
            model_file = f"{self.models_path}/{safe_symbol}_{model_name}.joblib"
            scaler_file = f"{self.models_path}/{safe_symbol}_scaler.joblib"
            features_file = f"{self.models_path}/{safe_symbol}_features.joblib"
            
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            joblib.dump(feature_cols, features_file)
            
            print(f"\n   💾 مدل ذخیره شد: {model_file}")
            
            return {
                'model': model,
                'scaler': scaler,
                'model_name': model_name,
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'feature_cols': feature_cols
            }
            
        except Exception as e:
            print(f"❌ خطا در آموزش مدل {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def train_all_models(self, symbols=None, interval='1d'):
        if symbols is None:
            symbols = self.config['data']['symbols']
            
        print_separator("شروع فاز 3 (نسخه 2.0): آموزش مدل‌های بهینه")
        
        results = {}
        for symbol in symbols:
            print_separator(f"آموزش مدل: {symbol.upper()}")
            result = self.train_model(symbol, interval)
            if result:
                results[symbol] = result
                
        print_separator("فاز 3 کامل شد")
        print(f"✅ {len(results)} مدل با موفقیت آموزش دیدند.")
        
        print("\n📊 خلاصه دقت مدل‌ها:")
        for symbol, result in results.items():
            print(f"   {symbol.upper()}: {result['accuracy'] * 100:.2f}% ({result['model_name']})")
        
        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()
