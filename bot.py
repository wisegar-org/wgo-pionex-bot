import os, time, hmac, hashlib, json, math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

def getenv(key, default=None, cast=str):
    v = os.getenv(key, default)
    if v is None: return None
    if cast is bool: return str(v).strip().lower() in ("1","true","yes","y","on")
    if cast in (float,int):
        try: return cast(v)
        except: return cast(default) if default is not None else None
    return str(v)

API_KEY     = getenv("PIONEX_API_KEY", "")
API_SECRET  = getenv("PIONEX_API_SECRET", "")
BASE_URL    = "https://api.pionex.com"
SYMBOL      = getenv("SYMBOL", "BTC_USDT_PERP")
TF          = getenv("TIMEFRAME", "15m")
PERCENT_RISK_ATR = getenv("PERCENT_RISK_ATR", 0.02, float)
SL_ATR      = getenv("SL_ATR", 1.5, float)
TP_ATR      = getenv("TP_ATR", 2.5, float)
CE_LENGTH   = getenv("CE_LENGTH", 22, int)
CE_MULT     = getenv("CE_MULT", 2.0, float)
ADX_LENGTH  = getenv("ADX_LENGTH", 14, int)
ADX_TH      = getenv("ADX_THRESHOLD", 35.0, float)
USE_EMA     = getenv("USE_EMA_FILTER", False, bool)
EMA_LEN     = getenv("EMA_LENGTH", 200, int)
ALLOW_SHORTS= getenv("ALLOW_SHORTS", True, bool)
MIN_NOTIONAL= getenv("MIN_NOTIONAL", 11.0, float)
PAPER_MODE  = getenv("PAPER_MODE", True, bool)
PAPER_LOG   = getenv("DRY_RUN_LOG", "trades_paper.csv", str)
FETCH_KLINES= getenv("FETCH_KLINES", 300, int)
POLL_SEC    = getenv("POLL_SEC", 5, int)

TF_MAP = {"1m":"1M","5m":"5M","15m":"15M","30m":"30M","1h":"60M","4h":"4H","8h":"8H","12h":"12H","1d":"1D"}
if TF not in TF_MAP: raise SystemExit(f"TIMEFRAME no soportado: {TF}")

def _ts_ms() -> int: return int(datetime.now(timezone.utc).timestamp()*1000)
def _qs(params: Dict[str,str]) -> str: return "&".join([f"{k}={v}" for k,v in sorted((k,str(v)) for k,v in params.items())])
def _headers(signature: str) -> Dict[str,str]: return {"PIONEX-KEY": API_KEY, "PIONEX-SIGNATURE": signature, "Content-Type": "application/json"}
def _sign(method: str, path: str, params: Dict[str,str], body_obj: Optional[dict]):
    query = params.copy() if params else {}; query["timestamp"] = _ts_ms(); qs = _qs(query); path_url = f"{path}?{qs}"
    payload = f"{method.upper()}{path_url}"; body_json = ""
    if method.upper() in ("POST","DELETE") and body_obj is not None:
        body_json = json.dumps(body_obj, separators=(",",":")); payload += body_json
    sign = hmac.new(API_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return path_url, sign, body_json
def _check(resp):
    resp.raise_for_status(); data = resp.json()
    if not data.get("result", False): raise RuntimeError(f"Pionex error: {data}")
    return data.get("data", {})

class Pionex:
    def __init__(self, base=BASE_URL):
        self.base = base; self.sess = requests.Session()
    def get_klines(self, symbol: str, tf_code: str, limit=500, endTime=None):
        params = {"symbol":symbol, "interval":tf_code, "limit":int(limit)}
        if endTime: params["endTime"] = int(endTime)
        r = self.sess.get(f"{self.base}/api/v1/market/klines", params=params, timeout=30); return _check(r).get("klines", [])
    def get_book_ticker(self, symbol: str):
        r = self.sess.get(f"{self.base}/api/v1/market/bookTickers", params={"symbol":symbol}, timeout=15); d = _check(r).get("tickers", []); return d[0] if d else None
    def balances(self):
        path="/api/v1/account/balances"; path_url, sig, _ = _sign("GET", path, {}, None)
        r = self.sess.get(f"{self.base}{path_url}", headers=_headers(sig), timeout=15); return _check(r).get("balances", [])
    def new_order(self, symbol:str, side:str, typ:str, size:Optional[str]=None, price:Optional[str]=None, amount:Optional[str]=None, ioc:bool=False):
        body = {"symbol":symbol, "side":side, "type":typ, "IOC":bool(ioc)}
        if size is not None: body["size"] = str(size)
        if price is not None: body["price"] = str(price)
        if amount is not None: body["amount"] = str(amount)
        path="/api/v1/trade/order"; path_url, sig, body_json = _sign("POST", path, {}, body)
        r = self.sess.post(f"{self.base}{path_url}", headers=_headers(sig), data=body_json, timeout=30); return _check(r)
    def cancel_order(self, symbol:str, orderId:int):
        body = {"symbol":symbol, "orderId": int(orderId)}; path="/api/v1/trade/order"; path_url, sig, body_json = _sign("DELETE", path, {}, body)
        r = self.sess.delete(f"{self.base}{path_url}", headers=_headers(sig), data=body_json, timeout=15); return _check(r)
    def open_orders(self, symbol:str):
        path="/api/v1/trade/openOrders"; params={"symbol":symbol, "timestamp":_ts_ms()}; qs=_qs(params)
        sign = hmac.new(API_SECRET.encode(), f"GET{path}?{qs}".encode(), hashlib.sha256).hexdigest()
        r = self.sess.get(f"{self.base}{path}?{qs}", headers=_headers(sign), timeout=15); return _check(r).get("orders", [])

def rma(x, n): return x.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
def true_range(h,l,c): pc = c.shift(1); return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
def atr(h,l,c,n): return rma(true_range(h,l,c), n)
def dmi_adx(h,l,c,n):
    up = h.diff(); dn = (-l.diff())
    plus_dm  = pd.Series(np.where((up>dn)&(up>0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn>up)&(dn>0), dn, 0.0), index=h.index)
    a = atr(h,l,c,n); plus_di  = 100 * rma(plus_dm,  n) / a; minus_di = 100 * rma(minus_dm, n) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,np.nan); return plus_di.fillna(0), minus_di.fillna(0), rma(dx, n).fillna(0)
def chandelier_exit(h,l,c,length,mult):
    a = atr(h,l,c,length); hh = h.rolling(length, min_periods=length).max(); ll = l.rolling(length, min_periods=length).min()
    ceL = hh - a*mult; ceS = ll + a*mult; return ceL, ceS, a
def ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()

@dataclass
class Position: side:str; size:float; entry_price:float; trail_ce:float; stop_lvl:float; tp_lvl:Optional[float]

class CEADXBot:
    def __init__(self, symbol:str, tf:str):
        self.symbol=symbol; self.tf=tf; self.pio=Pionex(); self.pos: Optional[Position]=None; self.logfile="bot_log.csv"
        if PAPER_MODE and not os.path.exists(PAPER_LOG): open(PAPER_LOG,"w").write("ts,symbol,side,action,price,size\n")
        if not os.path.exists(self.logfile): open(self.logfile,"w").write("ts,event,detail\n")
    def _log(self, event, detail):
        open(self.logfile,"a").write(f"{datetime.utcnow().isoformat()}Z,{event},{detail}\n"); print(f"[{event}] {detail}")
    def _paper_fill(self, side:str, price:float, size:float):
        open(PAPER_LOG,"a").write(f"{datetime.utcnow().isoformat()}Z,{self.symbol},{side},MARKET,{price},{size}\n")
    def _get_usdt_free(self)->float:
        for b in self.pio.balances():
            if b.get("coin")=="USDT": return float(b.get("free","0")); 
        return 0.0
    def _place_market(self, side:str, size:float):
        if PAPER_MODE:
            ticker = self.pio.get_book_ticker(self.symbol) or {}; px = float(ticker.get("askPrice") if side.upper()=="BUY" else ticker.get("bidPrice") or 0) or 0
            self._paper_fill(side.upper(), px, size); return {"orderId": 0, "price": px}
        else: return self.pio.new_order(self.symbol, side.upper(), "MARKET", size=str(size))
    def _close_market(self):
        if not self.pos: return
        side = "SELL" if self.pos.side=="long" else "BUY"; return self._place_market(side, self.pos.size)
    def compute_signals(self, df:pd.DataFrame):
        ceL, ceS, atr_v = chandelier_exit(df["high"], df["low"], df["close"], CE_LENGTH, CE_MULT); _,_, adx = dmi_adx(df["high"], df["low"], df["close"], ADX_LENGTH)
        ema_line = ema(df["close"], EMA_LEN) if USE_EMA else pd.Series(np.nan, index=df.index)
        pc = float(df["close"].iloc[-2]); c = float(df["close"].iloc[-1]); ceL_now, ceS_now = float(ceL.iloc[-1]), float(ceS.iloc[-1])
        ceL_prev, ceS_prev = float(ceL.iloc[-2]), float(ceS.iloc[-2]); adx_now = float(adx.iloc[-1])
        ema_ok_L = True if not USE_EMA else (c > float(ema_line.iloc[-1])); ema_ok_S = True if not USE_EMA else (c < float(ema_line.iloc[-1]))
        long_sig  = (pc <= ceS_prev) and (c > ceS_now) and (adx_now > ADX_TH) and ema_ok_L
        short_sig = (pc >= ceL_prev) and (c < ceL_now) and (adx_now > ADX_TH) and ema_ok_S and ALLOW_SHORTS
        atr_last = float(atr_v.iloc[-1]); return {"long": long_sig, "short": short_sig, "atr": atr_last, "ceL": ceL_now, "ceS": ceS_now}
    def refresh_stops(self, price:float, atr:float, ce_line:float):
        if not self.pos: return
        if self.pos.side=="long":
            self.pos.trail_ce = max(self.pos.trail_ce, ce_line); self.pos.stop_lvl = max(self.pos.trail_ce, self.pos.entry_price - SL_ATR*atr)
            self.pos.tp_lvl   = self.pos.entry_price + TP_ATR*atr if TP_ATR>0 else None
        else:
            self.pos.trail_ce = min(self.pos.trail_ce, ce_line); self.pos.stop_lvl = min(self.pos.trail_ce, self.pos.entry_price + SL_ATR*atr)
            self.pos.tp_lvl   = self.pos.entry_price - TP_ATR*atr if TP_ATR>0 else None
    def compute_size(self, balance_usdt:float, atr:float, last_price:float)->float:
        stop_dist = SL_ATR*atr; 
        if stop_dist<=0: return 0.0
        risk_usdt = max(balance_usdt * PERCENT_RISK_ATR, 0.0); size = risk_usdt / stop_dist
        if last_price*size < MIN_NOTIONAL: return 0.0
        return math.floor(size*1e4)/1e4
    def load_klines(self, limit:int)->pd.DataFrame:
        tf_code = TF_MAP[TF]; kl = self.pio.get_klines(self.symbol, tf_code, limit=min(max(limit,50),500))
        if not kl: raise RuntimeError("Sin klines")
        df = pd.DataFrame(kl); 
        for col in ["open","high","low","close","volume"]: df[col] = pd.to_numeric(df[col], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True); 
        return df.dropna(subset=["open","high","low","close"]).drop_duplicates("time").set_index("time").sort_index()
    def run(self):
        self._log("start", f"symbol={self.symbol} tf={TF} paper={PAPER_MODE}"); cooldown = 0
        while True:
            try:
                df = self.load_klines(FETCH_KLINES); sig = self.compute_signals(df); last_close = float(df["close"].iloc[-1]); price_now = last_close
                bt = self.pio.get_book_ticker(self.symbol) or {}; 
                if bt:
                    bid = float(bt.get("bidPrice") or 0.0); ask = float(bt.get("askPrice") or 0.0)
                    price_now = (bid+ask)/2 if (bid and ask) else (bid or ask or last_close)
                if self.pos:
                    ce_line = sig["ceL"] if self.pos.side=="long" else sig["ceS"]; self.refresh_stops(price_now, sig["atr"], ce_line)
                    if self.pos.side=="long":
                        if price_now <= self.pos.stop_lvl or (self.pos.tp_lvl and price_now >= self.pos.tp_lvl):
                            self._log("exit", f"{self.pos.side}@{price_now}, stop={self.pos.stop_lvl}, tp={self.pos.tp_lvl}"); self._close_market(); self.pos=None; cooldown = 3
                    else:
                        if price_now >= self.pos.stop_lvl or (self.pos.tp_lvl and price_now <= self.pos.tp_lvl):
                            self._log("exit", f"{self.pos.side}@{price_now}, stop={self.pos.stop_lvl}, tp={self.pos.tp_lvl}"); self._close_market(); self.pos=None; cooldown = 3
                if not self.pos:
                    if cooldown>0: cooldown -= 1
                    else:
                        if sig["long"] or sig["short"]:
                            balance = self._get_usdt_free() if not PAPER_MODE else 1000.0; size = self.compute_size(balance, sig["atr"], price_now)
                            if size>0:
                                side = "BUY" if sig["long"] else "SELL"; res = self._place_market(side, size); entry_px = float(res.get("price", price_now)) if isinstance(res, dict) else price_now
                                if sig["long"]: self.pos = Position("long", size, entry_px, sig["ceL"], 0.0, None)
                                else: self.pos = Position("short", size, entry_px, sig["ceS"], 0.0, None)
                                self.refresh_stops(price_now, sig["atr"], self.pos.trail_ce); self._log("entry", f"{self.pos.side} size={size} @ {entry_px} stop={self.pos.stop_lvl} tp={self.pos.tp_lvl}")
                time.sleep(POLL_SEC)
            except Exception as e:
                self._log("error", str(e)); time.sleep(2)

if __name__=="__main__":
    if not PAPER_MODE and (not API_KEY or not API_SECRET): raise SystemExit("Faltan PIONEX_API_KEY / PIONEX_API_SECRET (o activa PAPER_MODE=true).")
    CEADXBot(SYMBOL, TF).run()
