import requests
import urllib3
import os

# غیرفعال کردن هشدارهای SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# نادیده گرفتن پروکسی ویندوز
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7'
}

print("🔍 تست ۱: نوبیتکس (با غیرفعال کردن بررسی سخت‌گیرانه SSL)...")
try:
    url = "https://api.nobitex.ir/v3/market/udf/history"
    params = {
        'symbol': 'btc-usdt',
        'resolution': 'D',
        'from': 1690000000,
        'to': 1700000000
    }
    # توجه: verify=False اضافه شد
    response = requests.get(url, params=params, headers=headers, timeout=10, verify=False)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('s') == 'ok':
            print("✅ نوبیتکس با موفقیت متصل شد و داده گرفت!")
        else:
            print(f"⚠️ نوبیتکس پاسخ داد اما داده‌ای نداشت: {data}")
    else:
        print(f"⚠️ نوبیتکس کد خطا داد: {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ نوبیتکس خطا داد: {e}")

print("\n🔍 تست ۲: والکس (Wallex) به عنوان جایگزین مطمئن...")
try:
    url = "https://api.wallex.ir/v1/markets/BTCUSDT/candles"
    params = {
        'resolution': '1D',
        'from': 1690000000,
        'to': 1700000000
    }
    response = requests.get(url, params=params, headers=headers, timeout=10, verify=False)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'success':
            print("✅ والکس با موفقیت متصل شد! (این یک جایگزین عالی برای نوبیتکس است)")
        else:
            print(f"⚠️ والکس پاسخ داد اما موفق نبود: {data}")
    else:
        print(f"⚠️ والکس کد خطا داد: {response.status_code}")
except Exception as e:
    print(f"❌ والکس خطا داد: {e}")
