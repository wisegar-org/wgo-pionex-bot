# ============================================================
# BOT CE+ADX + IA de Patrones (K-means) + AUTOENTRENADOR 30min
# - Mercado: BTC_USDT_PERP (Pionex) • TF=15m
# - Estrategia: CE(22,2.0) + ADX(14)>35, SL=1.5×ATR, TP=2.5×ATR (para sizing/IA)
# - Realismo: taker=0.05%, min notional=11 USDT, límite ≤1% $vol EMA, leverage 3x
# - IA: ventana 64 (log-returns, RSI, ADX), K=20, Beta(8,8), min_count=100
# - Autoentrenador: reentrena cada 30min con 180 días y actualiza en caliente
# - Conexión: API Key/Secret en .env (HMAC). DRYRUN para pruebas.
# ============================================================

import os, sys, time, json, math, hmac, hashlib, threading, uuid, warnings
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import requests


# --------- .env ----------
def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, "r", encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if os.environ.get(k) is None:
                os.environ[k] = v


load_env()

# --------- Config --------
CFG: Dict[str, Any] = dict(
    BASE_URL=os.getenv("PIONEX_BASE_URL", "https://api.pionex.com").rstrip("/"),
    API_KEY=os.getenv("PIONEX_API_KEY"),
    API_SECRET=os.getenv("PIONEX_API_SECRET"),
    SYMBOL=os.getenv("SYMBOL", "BTC_USDT_PERP"),
    TIMEFRAME=os.getenv("TIMEFRAME", "15m"),
    INITIAL_CASH=float(os.getenv("INITIAL_CASH", "34.66")),
    CE_LENGTH=22,
    CE_MULT=2.0,
    ADX_LENGTH=14,
    ADX_THRESHOLD=35.0,
    USE_EMA_FILTER=False,
    EMA_LENGTH=200,
    ALLOW_SHORTS=True,
    SL_ATR=1.5,
    TP_ATR=2.5,
    TAKER_FEE=0.0005,
    MIN_NOTIONAL=11.0,
    SPREAD_BPS_MEAN=2.0,
    SPREAD_BPS_MAX=15.0,
    SPREAD_VOL_K=90.0,
    LATENCY_K=0.10,
    LIQ_FRAC=0.01,
    LEVERAGE=3.0,
    ATR_FLOOR_PCT=0.002,
    PERCENT_RISK_ATR=0.02,
    USE_PATTERN_AI=True,
    PAT_WIN_LEN=64,
    PAT_K=20,
    PAT_MAX_TRAIN_SAMPLES=4000,
    PAT_ITERS=50,
    PAT_MIN_COUNT=100,
    PAT_ALPHA=8.0,
    PAT_BETA=8.0,
    PAT_AI_PMIN_OFFSET=0.03,
    USE_KELLY=True,
    KELLY_CAP=0.02,
    AUTO_INTERVAL_MINUTES=int(os.getenv("AUTO_INTERVAL_MINUTES", "30")),
    AUTO_LOOKBACK_DAYS=int(os.getenv("AUTO_LOOKBACK_DAYS", "180")),
    DRYRUN=("--dryrun" in sys.argv)
    or (os.getenv("DRYRUN", "false").lower() in {"1", "true", "yes", "y"}),
)


def tf_minutes(tf: str) -> int:
    t = tf.lower()
    return (
        int(t[:-1]) * (60 if t.endswith("h") else 1) if t.endswith(("m", "h")) else 15
    )


CFG["TF_MINUTES"] = tf_minutes(CFG["TIMEFRAME"])


# --------- Utils ---------
def utcnow():
    return pd.Timestamp.now(tz="UTC")


def wait_next_close(tfm: int):
    now = utcnow()
    sec = ((tfm - (now.minute % tfm)) * 60) - now.second
    if sec <= 1:
        sec += tfm * 60
    time.sleep(int(sec))


def zscore(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std()
    if sd < eps:
        sd = 1.0
    return (x - mu) / sd


# --------- API Pionex ----
def _signed_headers(method: str, path: str, params: dict = None, body: dict = None):
    if not CFG["API_KEY"] or not CFG["API_SECRET"]:
        if not CFG["DRYRUN"]:
            raise RuntimeError("Faltan API key/secret en .env")
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    items = sorted(params.items())
    q = "&".join(f"{k}={v}" for k, v in items)
    path_url = f"{path}?{q}" if q else path
    msg = method.upper() + path_url
    if body:
        msg += json.dumps(body, separators=(",", ":"))
    sig = (
        hmac.new(
            (CFG["API_SECRET"] or "").encode(), msg.encode(), hashlib.sha256
        ).hexdigest()
        if CFG["API_SECRET"]
        else ""
    )
    return {
        "PIONEX-KEY": CFG.get("API_KEY", ""),
        "PIONEX-SIGNATURE": sig,
        "Content-Type": "application/json",
    }, path_url


TF_MAP = {
    "1m": "1M",
    "5m": "5M",
    "15m": "15M",
    "30m": "30M",
    "1h": "60M",
    "4h": "4H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
}


def api_get_klines(
    symbol: str, interval_code: str, limit: int = 500, endTime: int = None
):
    params = {"symbol": symbol, "interval": interval_code, "limit": limit}
    if endTime is not None:
        params["endTime"] = endTime
    r = requests.get(
        f"{CFG['BASE_URL']}/api/v1/market/klines", params=params, timeout=30
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("result"):
        raise RuntimeError(data)
    return data["data"]["klines"]


def api_new_order(
    symbol: str,
    side: str,
    type_: str,
    size: str = None,
    amount: str = None,
    ioc: bool = True,
):
    if CFG["DRYRUN"]:
        return {"result": True, "data": {"orderId": "DRYRUN"}}
    body = {"symbol": symbol, "side": side, "type": type_}
    if size is not None:
        body["size"] = str(size)
    if amount is not None:
        body["amount"] = str(amount)
    if ioc:
        body["IOC"] = True
    headers, path_url = _signed_headers("POST", "/api/v1/trade/order", {}, body)
    r = requests.post(
        f"{CFG['BASE_URL']}{path_url}",
        headers=headers,
        data=json.dumps(body),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# --------- Datos ---------
def fetch_df(symbol: str, timeframe: str, bars: int = 2000) -> pd.DataFrame:
    code = TF_MAP[timeframe]
    end_ms = int(utcnow().timestamp() * 1000)
    ms_map = {
        "1m": 60000,
        "5m": 300000,
        "15m": 900000,
        "30m": 1800000,
        "1h": 3600000,
        "4h": 14400000,
        "8h": 28800000,
        "12h": 43200000,
        "1d": 86400000,
    }
    rows = []
    end_cursor = end_ms
    while len(rows) < bars:
        kl = api_get_klines(
            symbol, code, limit=min(500, bars - len(rows)), endTime=end_cursor
        )
        if not kl:
            break
        kl = sorted(kl, key=lambda x: x["time"])
        rows = kl + rows
        oldest = kl[0]["time"]
        end_cursor = oldest - ms_map.get(timeframe, 900000)
        if end_cursor < 0:
            break
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = (
        df.dropna(subset=["open", "high", "low", "close"])
        .drop_duplicates("time")
        .set_index("time")
        .sort_index()
    )
    return df


# --------- Indicadores ---
def rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(
        axis=1
    )


def atr(h, l, c, n):
    return rma(true_range(h, l, c), n)


def dmi_adx(h, l, c, n):
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    a = atr(h, l, c, n)
    plus_di = 100 * rma(plus_dm, n) / (a + 1e-12)
    minus_di = 100 * rma(minus_dm, n) / (a + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = rma(dx, n)
    return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)


def chandelier_exit(h, l, c, length, mult):
    a = atr(h, l, c, length)
    hh = h.rolling(length, min_periods=length).max()
    ll = l.rolling(length, min_periods=length).min()
    return hh - a * mult, ll + a * mult, a


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rd = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))


# --------- IA patrones ---
def build_vec(idx, side, L, close, rsi_s, adx_s):
    if idx - L + 1 < 0:
        return None
    sl = slice(idx - L + 1, idx + 1)
    c = close.iloc[sl].values
    r = rsi_s.iloc[sl].values
    a = adx_s.iloc[sl].values
    lr = np.diff(np.log(np.maximum(c, 1e-9)))
    lr = np.concatenate(([0.0], lr))
    if side == "short":
        lr = -lr
    return np.concatenate([zscore(lr), zscore(r), zscore(a)], axis=0).astype(np.float32)


def kmeans(X, k, iters=50, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = X[rng.choice(n, size=min(k, n), replace=False)].copy()
    if C.shape[0] < k:
        C = np.vstack([C, C[rng.choice(C.shape[0], size=k - C.shape[0], replace=True)]])
    lab = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            m = lab == j
            C[j] = X[rng.integers(0, n)] if not np.any(m) else X[m].mean(axis=0)
    return C, lab


def train_ai(
    df,
    ceL,
    ceS,
    adx_s,
    rsi_s,
    ema_s,
    atr_raw,
    L=64,
    K=20,
    adx_thr=35,
    use_ema=False,
    allow_shorts=True,
    max_samples=4000,
):
    close = df["close"]
    Xv = []
    y = []
    for j in range(1, len(df)):
        pc = float(df["close"].iloc[j - 1])
        c = float(df["close"].iloc[j])
        ceL_now, ceS_now = float(ceL.iloc[j]), float(ceS.iloc[j])
        ceL_prev, ceS_prev = float(ceL.iloc[j - 1]), float(ceS.iloc[j - 1])
        adx_now = float(adx_s.iloc[j])
        ema_ok_L = True if not use_ema else (c > float(ema_s.iloc[j]))
        ema_ok_S = True if not use_ema else (c < float(ema_s.iloc[j]))
        long_sig = (
            (pc <= ceS_prev) and (c > ceS_now) and (adx_now > adx_thr) and ema_ok_L
        )
        short_sig = (
            (pc >= ceL_prev)
            and (c < ceL_now)
            and (adx_now > adx_thr)
            and ema_ok_S
            and allow_shorts
        )
        side = "long" if long_sig else ("short" if short_sig else None)
        if side is None or j - L + 1 < 0:
            continue
        vec = build_vec(j, side, L, close, rsi_s, adx_s)
        if vec is None:
            continue
        # etiquetar por TP/SL (máximo 96 barras)
        entry = c
        atr_j = float(atr_raw.iloc[j])
        if atr_j <= 0:
            continue
        if side == "long":
            tp = entry + CFG["TP_ATR"] * atr_j
            sl = entry - CFG["SL_ATR"] * atr_j
            lab = None
            for k in range(j + 1, min(j + 97, len(df))):
                h = float(df["high"].iloc[k])
                l = float(df["low"].iloc[k])
                if l <= sl:
                    lab = 0
                    break
                if h >= tp:
                    lab = 1
                    break
        else:
            tp = entry - CFG["TP_ATR"] * atr_j
            sl = entry + CFG["SL_ATR"] * atr_j
            lab = None
            for k in range(j + 1, min(j + 97, len(df))):
                h = float(df["high"].iloc[k])
                l = float(df["low"].iloc[k])
                if h >= sl:
                    lab = 0
                    break
                if l <= tp:
                    lab = 1
                    break
        if lab is None:
            continue
        Xv.append(vec)
        y.append(lab)
    if not Xv:
        return None
    X = np.vstack(Xv).astype(np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(X) > max_samples:
        sel = np.random.choice(len(X), size=max_samples, replace=False)
        X = X[sel]
        y = y[sel]
    K = min(K, len(X)) if len(X) > 0 else 1
    C, lab = kmeans(X, K, iters=CFG["PAT_ITERS"], seed=42)
    wr = np.zeros(C.shape[0], dtype=np.float32)
    cnt = np.zeros(C.shape[0], dtype=np.int32)
    a, b = CFG["PAT_ALPHA"], CFG["PAT_BETA"]
    for k in range(C.shape[0]):
        m = lab == k
        cnt[k] = int(m.sum())
        wins = float(y[m].sum()) if cnt[k] > 0 else 0.0
        wr[k] = (wins + a) / (cnt[k] + a + b)
    return {"centers": C, "wr": wr, "counts": cnt, "L": CFG["PAT_WIN_LEN"]}


def pwin(model, vec):
    if model is None or vec is None:
        return None
    C = model["centers"]
    cnt = model["counts"]
    wr = model["wr"]
    d2 = ((C - vec[None, :]) ** 2).sum(axis=1)
    j = int(np.argmin(d2))
    return None if cnt[j] < CFG["PAT_MIN_COUNT"] else float(wr[j])


# --------- Slippage -----
def eff_spread_bps(atr_val, close):
    if not (np.isfinite(atr_val) and close > 0):
        return 0.0
    atr_pct = float(atr_val / close)
    bps = CFG["SPREAD_BPS_MEAN"] + CFG["SPREAD_VOL_K"] * max(atr_pct, 0.0) * 100.0
    return min(max(bps, 0.0), CFG["SPREAD_BPS_MAX"])


def price_effective(side, c, o, h, l, atr_i):
    half = c * (eff_spread_bps(atr_i, c) * 1e-4 / 2.0)
    px = c + (half if side == "long" else -half)
    px += (h - l) * CFG["LATENCY_K"] * (+1 if side == "long" else -1)
    low = min(o, h, l, c) - atr_i
    high = max(o, h, l, c) + atr_i
    return float(min(max(px, low), high))


# --------- Modelo global y autoentrenador ----------
MODEL_FILE = Path("pattern_model.npz")
_model_lock = threading.Lock()
pattern_model = None
_last_trained_bar = None


def save_model(m, bars, last_bar):
    np.savez(
        MODEL_FILE,
        centers=m["centers"],
        wr=m["wr"],
        counts=m["counts"],
        L=np.array([m["L"]], dtype=np.int32),
    )


def load_model():
    if not MODEL_FILE.exists():
        return None
    d = np.load(MODEL_FILE, allow_pickle=True)
    return {
        "centers": d["centers"],
        "wr": d["wr"],
        "counts": d["counts"],
        "L": int(d["L"][0]),
    }


def autotrain():
    global pattern_model, _last_trained_bar
    print(
        f"[AUTO] ON • cada {CFG['AUTO_INTERVAL_MINUTES']} min • ventana={CFG['AUTO_LOOKBACK_DAYS']} días"
    )
    while True:
        try:
            bars = 96 * CFG["AUTO_LOOKBACK_DAYS"] if CFG["TF_MINUTES"] == 15 else 2000
            df = fetch_df(CFG["SYMBOL"], CFG["TIMEFRAME"], bars=bars)
            if df.empty:
                time.sleep(CFG["AUTO_INTERVAL_MINUTES"] * 60)
                continue
            ceL, ceS, atr_raw = chandelier_exit(
                df["high"], df["low"], df["close"], CFG["CE_LENGTH"], CFG["CE_MULT"]
            )
            *_, adx = dmi_adx(df["high"], df["low"], df["close"], CFG["ADX_LENGTH"])
            ema_line = (
                ema(df["close"], CFG["EMA_LENGTH"])
                if CFG["USE_EMA_FILTER"]
                else pd.Series(np.nan, index=df.index)
            )
            rs_series = rsi(df["close"], 14)
            m = train_ai(
                df,
                ceL,
                ceS,
                adx,
                rs_series,
                ema_line,
                atr_raw,
                L=CFG["PAT_WIN_LEN"],
                K=CFG["PAT_K"],
                adx_thr=CFG["ADX_THRESHOLD"],
                use_ema=CFG["USE_EMA_FILTER"],
                allow_shorts=CFG["ALLOW_SHORTS"],
                max_samples=CFG["PAT_MAX_TRAIN_SAMPLES"],
            )
            if m:
                with _model_lock:
                    pattern_model = m
                save_model(m, len(df), df.index[-1])
                _last_trained_bar = df.index[-1]
                print(
                    f"[AUTO] Modelo actualizado ✓ K={m['centers'].shape[0]} | última={_last_trained_bar}"
                )
        except Exception as e:
            print("[AUTO] Error:", e)
        time.sleep(CFG["AUTO_INTERVAL_MINUTES"] * 60)


# --------- Trading loop ---
def place_market(side_api, notional, price_hint):
    if CFG["DRYRUN"]:
        print(f"[DRYRUN] {side_api} MARKET notional≈{notional:.2f}")
        return {"result": True, "data": {"orderId": "DRY"}}
    if side_api == "BUY":
        return api_new_order(CFG["SYMBOL"], "BUY", "MARKET", amount=f"{notional:.2f}")
    size = max(0.0, math.floor((notional / max(price_hint, 1e-9)) * 1e4) / 1e4)
    return api_new_order(CFG["SYMBOL"], "SELL", "MARKET", size=f"{size:.4f}")


def main_loop():
    global pattern_model
    # Cargar modelo previo si existe
    m = load_model()
    if m:
        with _model_lock:
            pattern_model = m
        print("[AUTO] Modelo cargado de disco.")
    # Primer entrenamiento si no hay
    df = fetch_df(CFG["SYMBOL"], CFG["TIMEFRAME"], bars=2000)
    if df.empty:
        raise RuntimeError("No hay datos iniciales")
    if pattern_model is None:
        print("[AUTO] Entrenando modelo inicial…")
        ceL, ceS, atr_raw = chandelier_exit(
            df["high"], df["low"], df["close"], CFG["CE_LENGTH"], CFG["CE_MULT"]
        )
        *_, adx = dmi_adx(df["high"], df["low"], df["close"], CFG["ADX_LENGTH"])
        ema_line = (
            ema(df["close"], CFG["EMA_LENGTH"])
            if CFG["USE_EMA_FILTER"]
            else pd.Series(np.nan, index=df.index)
        )
        rs_series = rsi(df["close"], 14)
        m = train_ai(
            df,
            ceL,
            ceS,
            adx,
            rs_series,
            ema_line,
            atr_raw,
            L=CFG["PAT_WIN_LEN"],
            K=CFG["PAT_K"],
            adx_thr=CFG["ADX_THRESHOLD"],
            use_ema=CFG["USE_EMA_FILTER"],
            allow_shorts=CFG["ALLOW_SHORTS"],
            max_samples=CFG["PAT_MAX_TRAIN_SAMPLES"],
        )
        with _model_lock:
            pattern_model = m
        save_model(m, len(df), df.index[-1])
    # Autoentrenador
    threading.Thread(target=autotrain, daemon=True).start()
    print("[BOT] Iniciando… DRYRUN:", CFG["DRYRUN"])
    while True:
        wait_next_close(CFG["TF_MINUTES"])
        df = fetch_df(CFG["SYMBOL"], CFG["TIMEFRAME"], bars=2000)
        if len(df) < 100:
            print("[BOT] pocas velas")
            continue
        ceL, ceS, atr_raw = chandelier_exit(
            df["high"], df["low"], df["close"], CFG["CE_LENGTH"], CFG["CE_MULT"]
        )
        *_, adx = dmi_adx(df["high"], df["low"], df["close"], CFG["ADX_LENGTH"])
        ema_line = (
            ema(df["close"], CFG["EMA_LENGTH"])
            if CFG["USE_EMA_FILTER"]
            else pd.Series(np.nan, index=df.index)
        )
        rs = rsi(df["close"], 14)
        dv_ema = (
            (df["close"] * df["volume"])
            .ewm(span=20, adjust=False, min_periods=1)
            .mean()
            .fillna(0.0)
        )
        atr_eff = pd.Series(
            np.maximum(atr_raw.values, CFG["ATR_FLOOR_PCT"] * df["close"].values),
            index=df.index,
        )
        i = len(df) - 1
        ts = df.index[i]
        o, h, l, c = [float(df[x].iloc[i]) for x in ["open", "high", "low", "close"]]
        pc = float(df["close"].iloc[i - 1])
        ceL_now, ceS_now = float(ceL.iloc[i]), float(ceS.iloc[i])
        ceL_prev, ceS_prev = float(ceL.iloc[i - 1]), float(ceS.iloc[i - 1])
        adx_now = float(adx.iloc[i])
        ema_ok_L = True if not CFG["USE_EMA_FILTER"] else (c > float(ema_line.iloc[i]))
        ema_ok_S = True if not CFG["USE_EMA_FILTER"] else (c < float(ema_line.iloc[i]))
        long_sig = (
            (pc <= ceS_prev)
            and (c > ceS_now)
            and (adx_now > CFG["ADX_THRESHOLD"])
            and ema_ok_L
        )
        short_sig = (
            (pc >= ceL_prev)
            and (c < ceL_now)
            and (adx_now > CFG["ADX_THRESHOLD"])
            and ema_ok_S
            and CFG["ALLOW_SHORTS"]
        )
        if not (long_sig or short_sig):
            print(f"[{ts}] sin señal")
            continue
        side = "long" if long_sig else "short"
        atr_i = float(atr_eff.iloc[i])
        base_px = price_effective(side, c, o, h, l, atr_i)
        stop_dist = CFG["SL_ATR"] * atr_i
        if stop_dist <= 0:
            print(f"[{ts}] stop<=0")
            continue
        # IA
        ok = True
        pw = None
        m = None
        with _model_lock:
            m = pattern_model
        if CFG["USE_PATTERN_AI"] and m is not None:
            vec = build_vec(i, side, m["L"], df["close"], rs, adx)
            pw = pwin(m, vec)
            if pw is not None:
                R = CFG["TP_ATR"] / CFG["SL_ATR"]
                pmin = 1.0 / (1.0 + R) + CFG["PAT_AI_PMIN_OFFSET"]
                ok = pw >= pmin
        if not ok:
            print(f"[{ts}] filtrado IA (p_win={pw:.3f})")
            continue
        # sizing
        cash = float(CFG["INITIAL_CASH"])
        risk = CFG["PERCENT_RISK_ATR"]
        if (pw is not None) and CFG["USE_KELLY"]:
            R = CFG["TP_ATR"] / CFG["SL_ATR"]
            f = (pw * (1 + R) - (1 - pw)) / max(R, 1e-9)
            f = max(0.0, min(CFG["KELLY_CAP"], f))
            risk = max(1e-4, min(CFG["PERCENT_RISK_ATR"], f))
        size = (cash * risk) / stop_dist
        notional = base_px * size
        max_lev = cash * CFG["LEVERAGE"]
        max_liq = CFG["LIQ_FRAC"] * float(dv_ema.iloc[i])
        cap = max(0.0, min(max_lev, max_liq))
        if cap > 0:
            notional = min(notional, cap)
        if notional < CFG["MIN_NOTIONAL"]:
            print(f"[{ts}] notional<{CFG['MIN_NOTIONAL']} → omite")
            continue
        side_api = "BUY" if side == "long" else "SELL"
        print(f"[{ts}] {side_api} MARKET notional≈{notional:.2f} (p_win={pw})")
        try:
            resp = place_market(side_api, notional, base_px)
            print("→", resp)
        except Exception as e:
            print("Orden error:", e)


if __name__ == "__main__":
    print("Pionex Bot — CE+ADX + IA + Autoentrenador")
    print("Símbolo:", CFG["SYMBOL"], "TF:", CFG["TIMEFRAME"], "DRYRUN:", CFG["DRYRUN"])
    if not CFG["DRYRUN"] and (not CFG["API_KEY"] or not CFG["API_SECRET"]):
        print("ERROR: Faltan PIONEX_API_KEY/PIONEX_API_SECRET en .env")
        sys.exit(1)
    main_loop()
