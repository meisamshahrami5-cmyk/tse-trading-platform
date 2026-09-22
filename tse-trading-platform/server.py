# -*- coding: utf-8 -*-
"""
CryptoAI Pro - Web server (FastAPI) + HTML UI
---------------------------------------------
Run:   python server.py      ->  http://127.0.0.1:8000

XGBoost models cannot run inside a browser, so this small server loads the
trained models (models/*.joblib), fetches prices, and exposes JSON endpoints
that the single-file UI (web/index.html) consumes.
"""
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# افزودن مسیر ریشه پروژه به sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from src.indicators import engineer_features

WEB_DIR = BASE_DIR / "web"
MODELS_DIR = BASE_DIR / "models"

SYMBOLS = {
    "BTC-USD": {"name": "بیت‌کوین (BTC)", "icon": "🟠"},
    "ETH-USD": {"name": "اتریوم (ETH)", "icon": "🔷"},
    "BNB-USD": {"name": "بایننس کوین (BNB)", "icon": "🟡"},
    "SOL-USD": {"name": "سولانا (SOL)", "icon": "🟣"},
    "ADA-USD": {"name": "کاردانو (ADA)", "icon": "🔵"},
}

# 420 days of history so that SMA-200 (trend filter) is computable
# even after the warm-up rows of the other indicators are dropped.
HISTORY_DAYS = 420
CACHE_TTL = 300  # seconds


# ============================================================
# توابع کمکی (اندیکاتورها از ماژول مشترک ایمپورت شده‌اند)
# ============================================================


def build_features(df):
    """Features + SMA_200. Rows are dropped only for the warm-up of the model
    features; SMA_200 may stay NaN in the early rows (it is only read at the end)."""
    from src.indicators import calc_sma
    
    f = engineer_features(df)
    f["SMA_200"] = calc_sma(f["Close"], 200)
    core_cols = [c for c in f.columns if c != "SMA_200"]
    return f.dropna(subset=core_cols).reset_index(drop=True)


# ============================================================
# Data + models
# ============================================================
_cache = {}
_lock = threading.Lock()


def _safe(sym):
    return sym.replace("-", "_").lower()


def _normalise(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date").drop_duplicates("Date")
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)


def _download_live(sym):
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=HISTORY_DAYS)
    df = yf.download(
        sym,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def _load_local(sym):
    f = BASE_DIR / "data" / "raw" / f"{_safe(sym)}_1d_raw.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["Date"])
    return df.sort_values("Date").tail(HISTORY_DAYS)


def get_prices(sym, refresh=False):
    """Returns (df, source, timestamp). Falls back to data/raw/*.csv when offline."""
    now = time.time()
    with _lock:
        hit = _cache.get(sym)
        if hit and not refresh and now - hit["t"] < CACHE_TTL:
            return hit["df"], hit["source"], hit["stamp"]

    df, source = None, "live"
    try:
        df = _download_live(sym)
    except Exception as e:  # network blocked, yfinance missing, ...
        print(f"[data] live download failed for {sym}: {e}")

    if df is None or len(df) == 0:
        df, source = _load_local(sym), "local"
    if df is None or len(df) == 0:
        raise HTTPException(503, "دریافت داده ممکن نیست: اینترنت/Yahoo Finance در دسترس نیست و فایل داده‌ی محلی هم یافت نشد.")

    df = _normalise(df)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        _cache[sym] = {"t": now, "df": df, "source": source, "stamp": stamp}
    return df, source, stamp


def _model_paths(sym):
    s = _safe(sym)
    models = [
        p for p in MODELS_DIR.glob(f"{s}_*.joblib")
        if not p.stem.lower().endswith(("_scaler", "_features"))
    ]
    scaler = MODELS_DIR / f"{s}_scaler.joblib"
    feats = MODELS_DIR / f"{s}_features.joblib"
    if not models or not scaler.exists() or not feats.exists():
        return None
    return models[0], scaler, feats


@lru_cache(maxsize=16)
def load_model(sym):
    paths = _model_paths(sym)
    if paths is None:
        return None
    try:
        m, s, f = paths
        return joblib.load(m), joblib.load(s), list(joblib.load(f))
    except Exception as e:
        raise HTTPException(500, f"خطا در بارگذاری مدل {sym}: {e}")


def require_model(sym):
    bundle = load_model(sym)
    if bundle is None:
        raise HTTPException(404, f"مدل {sym} یافت نشد. ابتدا `python main.py` را برای آموزش مدل‌ها اجرا کنید.")
    return bundle


def check_symbol(symbol):
    sym = str(symbol).upper()
    if sym not in SYMBOLS:
        raise HTTPException(404, "نماد نامعتبر است.")
    return sym


# ============================================================
# JSON helpers
# ============================================================
def col(s, nd=6):
    out = []
    for v in np.asarray(s, dtype=float):
        out.append(None if not np.isfinite(v) else round(float(v), nd))
    return out


def num(v, nd=6):
    v = float(v)
    return None if not np.isfinite(v) else round(v, nd)


def dates(s):
    return [d.strftime("%Y-%m-%d") for d in s]


def meta(df, source, stamp):
    last = df["Date"].iloc[-1]
    return {
        "source": source,
        "updated": stamp,
        "data_date": last.strftime("%Y-%m-%d"),
        "stale_days": int((datetime.now() - last.to_pydatetime()).days),
    }


# ============================================================
# API
# ============================================================
app = FastAPI(title="CryptoAI Pro", version="3.1")


@app.get("/api/symbols")
def api_symbols():
    return [
        {"id": k, "name": v["name"], "icon": v["icon"], "has_model": _model_paths(k) is not None}
        for k, v in SYMBOLS.items()
    ]


@app.get("/api/dashboard")
def api_dashboard(symbol: str = "BTC-USD", refresh: bool = False):
    sym = check_symbol(symbol)
    df, source, stamp = get_prices(sym, refresh)
    core = build_features(df)
    if core.empty:
        raise HTTPException(422, "داده‌ی کافی برای محاسبه‌ی اندیکاتورها وجود ندارد.")

    view = core.tail(180).reset_index(drop=True)
    last = core.iloc[-1]
    week = None
    if len(df) >= 8:
        week = num((df["Close"].iloc[-1] / df["Close"].iloc[-8] - 1) * 100, 3)

    table = []
    for _, r in core.tail(7).iloc[::-1].iterrows():
        table.append({
            "date": r["Date"].strftime("%Y-%m-%d"),
            "close": num(r["Close"]),
            "volume": num(r["Volume"], 0),
            "rsi": num(r["RSI_14"], 2),
            "macd_hist": num(r["MACD_Hist"], 4),
            "ret": num(r["Daily_Return"] * 100, 3),
        })

    return {
        "symbol": sym,
        **meta(df, source, stamp),
        "week_change_pct": week,
        "kpis": {
            "price": num(last["Close"]),
            "change_pct": num(last["Daily_Return"] * 100, 3),
            "rsi": num(last["RSI_14"], 2),
            "atr": num(last["ATR_14"]),
            "volume": num(last["Volume"], 0),
        },
        "chart": {
            "date": dates(view["Date"]),
            "open": col(view["Open"]), "high": col(view["High"]),
            "low": col(view["Low"]), "close": col(view["Close"]),
            "bb_upper": col(view["BB_Upper"]), "bb_lower": col(view["BB_Lower"]),
            "sma20": col(view["SMA_20"]), "sma50": col(view["SMA_50"]),
            "rsi": col(view["RSI_14"], 2), "macd_hist": col(view["MACD_Hist"], 4),
        },
        "table": table,
    }


@app.get("/api/signal")
def api_signal(symbol: str = "BTC-USD", refresh: bool = False):
    sym = check_symbol(symbol)
    model, scaler, cols = require_model(sym)
    df, source, stamp = get_prices(sym, refresh)
    core = build_features(df)
    if core.empty:
        raise HTTPException(422, "داده‌ی کافی برای تحلیل وجود ندارد.")
    missing = [c for c in cols if c not in core.columns]
    if missing:
        raise HTTPException(422, f"ویژگی‌های مورد نیاز مدل محاسبه نشد: {', '.join(missing)}")

    last = core.iloc[-1]
    X = scaler.transform(core.iloc[[-1]][cols])
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    conf = float(max(proba)) * 100
    prob_up = float(proba[1]) * 100 if len(proba) > 1 else None

    cp, atr, rsi = float(last["Close"]), float(last["ATR_14"]), float(last["RSI_14"])
    sma200 = float(last["SMA_200"])
    have_trend = bool(np.isfinite(sma200))

    action, reason = "NEUTRAL", None
    if not have_trend:
        reason = "no_trend_data"
    elif conf <= 60:
        reason = "low_confidence"
    elif pred == 1 and cp > sma200:
        action = "LONG"
    elif pred == 0 and cp < sma200:
        action = "SHORT"
    else:
        reason = "against_trend"

    out = {
        "symbol": sym,
        **meta(df, source, stamp),
        "action": action,
        "reason": reason,
        "confidence": round(conf, 1),
        "prob_up": None if prob_up is None else round(prob_up, 1),
        "price": num(cp),
        "atr": num(atr),
        "rsi": num(rsi, 2),
        "sma200": num(sma200) if have_trend else None,
        "trend": None if not have_trend else ("up" if cp > sma200 else "down"),
        "leverage": "0x (نقد)", "position": "0%",
        "entry": None, "sl": None, "tp": None, "sl_pct": None, "tp_pct": None,
    }

    if action != "NEUTRAL":
        tier = 2 if conf > 75 else (1 if conf > 65 else 0)
        out["leverage"] = ["1x-2x", "3x-5x", "5x-10x"][tier]
        out["position"] = ["10-20%", "20-30%", "30-50%"][tier]
        # For a SHORT the stop is ABOVE the entry and the target BELOW it.
        sign = 1 if action == "LONG" else -1
        sl, tp = cp - sign * 1.5 * atr, cp + sign * 3.0 * atr
        out.update({
            "entry": num(cp), "sl": num(sl), "tp": num(tp),
            "sl_pct": num((sl - cp) / cp * 100, 3), "tp_pct": num((tp - cp) / cp * 100, 3),
        })
    return out


@app.get("/api/backtest")
def api_backtest(symbol: str = "BTC-USD"):
    sym = check_symbol(symbol)
    model, scaler, cols = require_model(sym)
    ff = BASE_DIR / "data" / "features" / f"{_safe(sym)}_1d_features.csv"
    if not ff.exists():
        raise HTTPException(404, "فایل داده‌های بک‌تست یافت نشد. ابتدا `python main.py` را اجرا کنید.")

    df = pd.read_csv(ff, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    ts = int(len(df) * 0.2)
    if ts < 30:
        raise HTTPException(422, "داده‌ی کافی برای بک‌تست وجود ندارد.")
    sma200 = df["Close"].rolling(200).mean()
    t = df.iloc[-ts:].copy().reset_index(drop=True)

    X = scaler.transform(t[cols])
    t["Signal"] = model.predict(X)
    t["Conf"] = model.predict_proba(X)[:, 1]          # probability of "up"
    t["SMA_200"] = sma200.iloc[-ts:].values

    # Layer 1: trend filter | Layer 2: confidence-based position sizing
    t["Signal"] = t["Signal"] * (t["Close"] > t["SMA_200"]).astype(int)
    size = np.where(t["Conf"] >= 0.75, 1.0,
                    np.where(t["Conf"] >= 0.55, (t["Conf"] - 0.55) / 0.20, 0.0))
    t["Pos"] = size * t["Signal"]
    t.loc[t["Conf"] < 0.55, "Signal"] = 0

    mr = t["Target_Next_Day_Return"].to_numpy(dtype=float)
    on = (t["Signal"] == 1).to_numpy()
    sr = np.where(on, mr * t["Pos"].to_numpy() - 0.001, 0.0)   # 0.1% fee per trade

    cs, cm = np.cumprod(1 + sr), np.cumprod(1 + mr)
    dd = cs / np.maximum.accumulate(cs) - 1

    n_trades = int(on.sum())
    win_rate = float((mr[on] > 0).mean() * 100) if n_trades else 0.0
    std = sr.std(ddof=1)
    sharpe = float(sr.mean() / std * np.sqrt(365)) if std > 0 else 0.0   # crypto trades 365 days/year
    neg = sr[sr < 0]
    dstd = neg.std(ddof=1) if len(neg) > 1 else 0.0
    sortino = float(sr.mean() / dstd * np.sqrt(365)) if dstd > 0 else 0.0
    gp, gl = sr[sr > 0].sum(), -sr[sr < 0].sum()

    return {
        "symbol": sym,
        "period": {
            "days": int(len(t)),
            "start": t["Date"].iloc[0].strftime("%Y-%m-%d"),
            "end": t["Date"].iloc[-1].strftime("%Y-%m-%d"),
        },
        "metrics": {
            "strategy_return": num((cs[-1] - 1) * 100, 2),
            "market_return": num((cm[-1] - 1) * 100, 2),
            "win_rate": num(win_rate, 1),
            "max_drawdown": num(dd.min() * 100, 2),
            "sharpe": num(sharpe, 2),
            "sortino": num(sortino, 2),
            "profit_factor": num(gp / gl, 2) if gl > 0 else None,
            "trades": n_trades,
            "exposure_pct": num(on.mean() * 100, 1),
        },
        "series": {
            "date": dates(t["Date"]),
            "strategy": col(cs), "market": col(cm),
            "drawdown": col(dd * 100, 3), "position": col(t["Pos"] * 100, 1),
        },
    }


# ============================================================
# UI
# ============================================================
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


def _open_browser(url):
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")   # local only; set HOST=0.0.0.0 to expose on the network
    port = int(os.getenv("PORT", "8000"))
    url = f"http://{host}:{port}"
    print(f"CryptoAI Pro is running at {url}   (Ctrl+C to stop)")
    if os.getenv("NO_BROWSER") != "1":
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
